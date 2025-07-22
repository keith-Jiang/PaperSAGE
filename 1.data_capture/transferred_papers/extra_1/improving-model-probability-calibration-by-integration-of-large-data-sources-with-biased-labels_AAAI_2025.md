# Improving Model Probability Calibration by Integration of Large Data Sources with Biased Labels

Renat Sergazinov 1, 2, Richard Chen 2, Cheng Ji 2, Jing Wu 3, Daniel Cociorva 2, Hakan Brunzell 2

1Department of Statistics, Texas A&M University, College Station, TX USA 2Buyer Risk Prevention at Amazon, San Diego, CA USA 3WWPS Solution Architecture at Amazon, Herndon, VA USA renserg, rychardy, cjiamzn, jingwua, cociorva, brunzell @amazon.com

# Abstract

Probability calibration transforms raw output of a classification model into empirically interpretable probability. When the model is purposed to detect rare event and only a small expensive data source has clean labels, it becomes extraordinarily challenging to obtain accurate probability calibration. Utilizing an additional large cheap data source is very helpful, however, such data sources oftentimes suffer from biased labels. To this end, we introduce an approximate expectationmaximization (EM) algorithm to extract useful information from the large data sources. For a family of calibration methods based on the logistic likelihood, we derive closed-form updates and call the resulting iterative algorithm CalEM. We show that CalEM inherits convergence guarantees from the approximate EM algorithm. We test the proposed model in simulation and on the real marketing datasets, where it shows significant performance increases.

# 1 Introduction

Machine learning models, including neural networks and gradient boosted trees, often suffer from calibration issues (Zadrozny and Elkan 2001; Guo et al. 2017; Zadrozny and Elkan 2001; Kuleshov, Fenner, and Ermon 2018; Ferna´ndez et al. 2018). Calibration refers to the alignment of a model’s predicted probabilities with the actual likelihood of outcomes. For instance, if a model predicts a probability of 0.2 for a specific class over 100 instances, ideally, 20 of those instances should belong to that class. This aspect of model performance is crucial for model trustworthiness and safety. There has been significant research on developing metrics for calibration assessment (Nixon et al. 2019; Gupta et al. 2020; Gruber and Buettner 2022), methods to post-correct mis-calibrated models (Platt et al. 1999; Zadrozny and Elkan 2002; Kumar, Liang, and Ma 2019; Gupta et al. 2020), and techniques for creating better-calibrated classifiers (Bohdal, Yang, and Hospedales 2023).

In this paper, we tackle a separate challenge of training a probability calibration model under practical constraints of label imbalance and small sample size. Under such constraints, accurate probability calibration becomes challenging as we illustrate in Figure 1. As the sample size decreases and the label imbalance exacerbates, the variance (calibration error) of the probability calibration model increases significantly.

A prominent example of real-world data with severe class imbalance and small sample size comes from the marketing literature (Diemert et al. 2018; Ke et al. 2021; Liu et al. 2023). In this context, a key objective is to estimate the probability that a customer will naturally convert to a given brand. With accurate estimates, marketing firms can strategically allocate budgets to target the most promising customer segments and improve overall sales. The target variable, a binary conversion indicator, is highly imbalanced (often with fewer than $1 \%$ converting), and the only unbiased dataset is the control group that receives no marketing intervention – data which can be both costly and limited. Meanwhile, the treatment group, exposed to marketing campaigns, produces biased labels unsuitable for direct modeling (Ke et al. 2021). Similar scenarios arise in (1) medical modeling, where the target is disease susceptibility and the treatment is a vaccine; (2) fraud prevention, where the target is fraudulent activity and the treatment is a preventative measure; and (3) online education, where the target is course completion and the treatment is a learning intervention. In all of these cases, the goal is to estimate the target, but the control data may be expensive, while treated samples have modified outcomes and cannot be used directly.

To solve this challenge, we propose a modified EM algorithm to fit any likelihood-based calibration model on the combined control and treatment portions of the dataset. We first estimate the calibration on the control data and then proceed to estimating the transition probability between the treatment and control sets. The calibration model could then be re-fit on the combined dataset with weights computed according to the transition probabilities. The algorithm then proceeds by iteratively refining the transition probability and calibration model estimates. As we illustrate in Figure 1, the algorithm results in smaller variance (calibration error). We theoretically show that our approach could be formulated as an approximate EM (Dempster, Laird, and Rubin 1977) and demonstrate convergence guarantees. We empirically validate its performance in simulation and on real marketing datasets, where we show improvement over the baseline methods fitted on the control-only data.

The rest of the paper is structured as follows. In Section 3 we introduce a general EM algorithm with approximate Estep which has an application to data augmentation for model calibration, we also show the convergence property of this approximate EM algorithm. We introduce baseline calibration in Section 4.2, and design data-augmented calibration algorithm based on the approximate EM algorithm in Section 4.3. In Section 5 we demonstrate our data-augmented model calibration in an empirical study. We put the proof and derivation in Appendix.

![](images/5c061f04b4a767e3e9e1f74b57d1ba50c0bba6fe153b6f2f77a3e323be486a4a.jpg)  
Figure 1: Variance of the probability calibration models in simulation. Note that larger variance leads to higher expected calibration error, by bias-variance decomposition. The shaded area represents the $9 5 \%$ confidence intervals around the curves. The baseline model is fitted on the small control data, while our proposed approach uses EM to fit on the combined control and large noisy datasets. Total sample size of the combined dataset is 2,000.

# 2 Related Work

Probability Calibration A variety of calibration techniques has been proposed in the literature such as works by Groeneboom and Lopuhaa (1993); Platt et al. (1999); Zadrozny and Elkan (2001); Niculescu-Mizil and Caruana (2005b,a); Naeini, Cooper, and Hauskrecht (2015); Kull, Silva Filho, and Flach (2017); Kull et al. (2019); Kumar, Liang, and Ma (2019); Gupta et al. (2021); van der Laan et al. (2023). Key desiderata for calibration methods is to be accuracy-preserving and data-efficient. To achieve the first, calibration methods are usually constrained to be monotonic. The latter has been studied by Kumar, Liang, and Ma (2019), who established sample efficiency for various calibration methods.

In practice, probability calibration can be integrated during model training (Platt et al. 1999; Niculescu-Mizil and Caruana 2005a) or applied as a post-hoc process (Groeneboom and Lopuhaa 1993; Kull, Silva Filho, and Flach 2017; Gupta et al. 2021). The post-hoc approach is generally favored in practical applications due to its superior performance and modularity, which allows the base classification model to remain unaltered.

Despite their popularity, post-hoc calibration models suffer from the need to maintain a separate calibration set for fitting. As we show in Figure 1, the variance (calibration error) of the model is highly dependent on the class imbalance and sample size. In many practical scenarios, maintaining a large calibration data set could be prohibitively expensive. In this work, we propose an algorithm that could be utilized together with any existing likelihood-based calibration models to incorporate large biased data source that substantially reduces the calibration error, whilst also not requiring the access to the expensive unbiased data.

Latent Variables To treat the bias in the large data source, we use the latent variable formulation. Numerous statistical frameworks are designed to handle unobserved (latent) variables effectively. For cases involving missing data, inverse probability weighting methods have demonstrated excellent performance, supported by robust theoretical guar

Algorithm 1: EM with approximate E-step

Initialization: specify initial point $\theta ^ { ( 0 ) }$ , tolerance $\varepsilon$ ; and learn $\eta$ ; Expectation-maximization iterations for $t = 0 , 1 , 2 , \cdots$

1. implement E-step by

$$
Q _ { \eta } ( \theta | \theta ^ { ( t ) } ) = \mathbb { E } _ { \mathcal { U } \sim P _ { \theta ^ { ( t ) } , \eta } ( \mathcal { U } | \mathcal { O } ) } \big [ \log P _ { \theta } ( \mathcal { O } , \mathcal { U } ) \big ] ;
$$

2. implement M-step by

$$
\theta ^ { ( t + 1 ) } = \underset { \theta } { \operatorname { a r g m a x } } Q _ { \eta } ( \theta | \theta ^ { ( t ) } ) ;
$$

3. continue until $\left\| \theta ^ { ( t + 1 ) } - \theta ^ { ( t ) } \right\| < \varepsilon$ .

![](images/37c7ccbe4443ead3eb5f329facdb469d9ba3a1c75cdb82c60b2c19039ab33b5a.jpg)  
Figure 2: Intuition of the convergence of our approximate EM

antees (Wooldridge 2007). In our scenario, we have a small control set that is representative of the population, alongside a larger dataset where a biased proxy of the response variable is available. The work in Chatterjee et al. (2016) explores a similar issue, utilizing a large additional data source to estimate the distribution of covariates in a regression framework. The study in Yang and Ding (2020) aligns even more closely with our setup, with the key difference being the adoption of a causal inference framework: the authors are interested in estimating the causal effects. We draw inspiration from these methodologies to develop our algorithm, which we build on the EM framework (Dempster, Laird, and Rubin 1977).

# 3 Approximate EM Algorithm

Let $\theta$ be the parameter of interest, $\mathcal { O }$ be the observed data, and $\mathcal { U }$ be the unobserved (latent) data. We can analytically formulate the log-likelihood function $\log { P _ { \theta } ( \mathcal { O } , \mathcal { U } ) }$ with respect to both observed and unobserved data. However, without the unobserved data $\mathcal { U }$ , one cannot analytically formulate the log-likelihood function $\mathcal { L } _ { \mathcal { O } } ( \theta ) = \log P _ { \theta } ( \mathcal { O } )$ with respect to observed data $\mathcal { O }$ alone. EM algorithm circumvents this difficulty, it computes expectation of the log-likelihood function $\log { P _ { \theta } ( \mathcal { O } , \mathcal { U } ) }$ conditional on observed data, and then maximizes the conditional expectation.

Given a starting value of parameter, the EM algorithm alternates the following two steps iteratively:

• $\mathbf { E }$ -step: find the conditional distribution of latent variables given observed variables and current parameter value, and compute the conditional expectation of the full likelihood of both observed and latent data;

Algorithm 2: EM with approximate $\mathrm { ~ E ~ }$ -step for penalized likelihood

Initialization: specify penalty hyperparameter $\lambda$ , initial point $\theta ^ { ( 0 ) }$ , tolerance $\varepsilon$ ; and learn $\eta$ ; Expectation-maximization iterations for $t = 0 , 1 , 2 , \cdots$

1. implement $\mathrm { ~ E ~ }$ -step by

$$
\begin{array} { r l } & { \boldsymbol { Q } _ { \lambda , \eta } ( \boldsymbol { \theta } | \boldsymbol { \theta } ^ { ( t ) } ) = \mathbb { E } _ { \mathcal { U } \sim \boldsymbol { P } _ { \boldsymbol { \theta } ^ { ( t ) } , \eta } ( \mathcal { U } | \mathcal { O } ) } \left[ \log \boldsymbol { P } _ { \boldsymbol { \theta } } ( \mathcal { O } , \mathcal { U } ) \right] } \\ & { \qquad - \Omega _ { \lambda } ( \boldsymbol { \theta } ) ; } \end{array}
$$

2. implement $\mathbf { M }$ -step by maximizing the penalized expected log-likelihood function:

$$
\theta ^ { ( t + 1 ) } = \underset { \theta } { \operatorname { a r g m a x } } Q _ { \lambda , \eta } ( \theta | \theta ^ { ( t ) } ) ;
$$

3. continue until $\big \| \theta ^ { ( t + 1 ) } - \theta ^ { ( t ) } \big \| < \varepsilon$

• M-step: maximize the conditional expectation of the full likelihood and update the parameter value.

The E-step and M-step of the EM algorithm can be described by

$$
\begin{array} { r c l } { Q ( \theta | \theta ^ { ( t ) } ) } & { : = } & { \mathbb { E } _ { \mathcal { U } \sim P _ { \theta ^ { ( t ) } } ( \mathcal { U } | \mathcal { O } ) } \big [ \log P _ { \theta } ( \mathcal { O } , \mathcal { U } ) \big ] , } \\ { \theta ^ { ( t + 1 ) } } & { = } & { \underset { \theta } { \operatorname { a r g m a x } } Q ( \theta | \theta ^ { ( t ) } ) , } \end{array}
$$

where $P _ { \theta ^ { ( t ) } } ( \mathcal { U } | \mathcal { O } )$ is the conditional distribution of latent variables conditioning on observed variables, parameterized by $\boldsymbol { \theta } ^ { ( t ) }$ . The EM algorithm can reach a local maximum of $\dot { \mathcal { L } } _ { \mathcal { O } } ( \boldsymbol { \theta } )$ (or the global maximum if ${ \mathcal { L } } _ { { \mathcal { O } } } ( \theta )$ is concave), even though the log-likelihood ${ \mathcal { L } } _ { { \mathcal { O } } } ( \theta )$ is not computable.

In practice, the E-step of Equation (1) may be difficult to compute. The probabilistic model of latent variable conditional on observed data may be so complicated that it is prohibitive to express the expectation in Equation (1) analytically. We propose an approximate E-step in which we use a surrogate conditional distribution $P _ { \theta ^ { ( t ) } , \eta } ( \mathcal { U } | \mathcal { O } )$ as an approximation of the exact conditional distribution $P _ { \theta ^ { ( t ) } } ( \mathcal { U } | \mathcal { O } )$ in iteration $t$ . The additional parameter $\eta$ in the surrogate is learned from the large noisy dataset and does not need to be updated in EM iterations.

If we model $P _ { \theta , \eta }$ wisely, the $\mathrm { E }$ -step in Algorithm 1 is conveniently implementable. In addition, if the noisy data size is sufficient, $P _ { \theta ^ { ( t ) } , \eta } ( \mathcal { U } | \mathcal { O } )$ can be a close proxy to ensure convergence. Below we give a theoretical result with regard to the convergence property of our approximate EM algorithm. The proof of the theorem can be found in Appendix A.1.

Theorem 1. Suppose $\theta ^ { * }$ is a local maximum of the likelihood function. Assume we can learn the nuisance parameter $\eta$ so that $P _ { \theta , \eta }$ is sufficiently close to $P _ { \theta }$ in the following sense:

$$
D _ { \mathrm { K L } } \big ( P _ { \theta , \eta } ( \cdot | \mathcal { O } ) \big | P _ { \theta ^ { * } } ( \cdot | \mathcal { O } ) \big ) \geq D _ { \mathrm { K L } } \big ( P _ { \theta , \eta } ( \cdot | \mathcal { O } ) \big | P _ { \theta } ( \cdot | \mathcal { O } ) \big ) ,
$$

as long as $\lVert { \boldsymbol { \theta } } - { \boldsymbol { \theta } } ^ { * } \rVert < \gamma _ { ; }$ , where $D _ { \mathrm { K L } } ( \cdot | \cdot )$ is Kullback–Leibler divergence1 and $\gamma$ is a positive constant, then Algorithm $\jmath$ (EM algorithm with approximate $E$ -step) converges.

1Kullback–Leibler divergence is defined as $\begin{array} { r l } { D _ { \mathrm { K L } } ( p | q ) } & { { } = } \end{array}$

Intuitively speaking, as long as the surrogate conditional distribution is closer to the exact conditional distribution than to the true conditional distribution, our approximate EM algorithm converges. The Requirement (3) is not a stringent assumption and we provide its intuition in Figure 2. Note that the Requirement (3) is trivially true when $\theta = \theta ^ { * }$ .

Modern practices of maximum likelihood estimation often incorporate penalty, i.e., choose

$$
L _ { \lambda } ( \theta ) = \mathcal { L } _ { \mathcal { O } } ( \theta ) - \Omega _ { \lambda } ( \theta )
$$

as the objective function, where $\Omega _ { \lambda }$ is a penalty function with hyperparameter $\lambda$ . The purpose of penalizing log-likelihood is to achieve certain desired properties, e.g. sparsity, monotonicity, etc. For probability calibration in Section 4, we need a penalty term to impose a shape constraint on calibration function. We apply the idea of EM algorithm with approximate $\mathrm { ^ E }$ -step to penalized maximum likelihood estimation in Algorithm 2.

We can extend the convergence property of EM algorithm with approximate E-step in Theorem 1 to penalized maximum likelihood estimation.

Corollary 1. Suppose $\theta ^ { * }$ is a local maximum of (4). Assume we can learn the nuisance parameter $\eta$ such that $P _ { \theta , \eta }$ satisfies (3) as long as $\lVert \theta - \theta ^ { * } \rVert < \gamma$ for a positive constant $\gamma$ , then Algorithm 2 converges.

# 4 Data-Augmented Model Calibration 4 .1 Notation and Setup

Let $\mathbb { X }$ be a feature space and $\mathbb { Y } = \{ 0 , 1 \}$ be a binary label – we label a sample that belongs to the target category as 1 (positive), otherwise we label it as 0 (negative). Denote the random variables corresponding to features and labels by $X$ and $Y$ respectively. We call the variable $Y$ unbiased or control. In contrast, denote by $Z$ the biased or treatment variables, which have distribution $p _ { Z } \neq p _ { Y }$ . Denote a classification model by $f : \mathbb { X } \mapsto [ 0 , 1 ]$ , i.e. $f ( X )$ is raw model score. The raw score $f ( X )$ from most modern machine learning models does not represent empirical probability. To calibrate raw model output whilst retain its predictive capability, a monotonically increasing function $g _ { \theta }$ is used to post hoc process raw model output as $g _ { \theta } \circ f ( X )$ , where $\theta$ is a multidimensional parameter. We call this function $g _ { \boldsymbol { \theta } }$ a probability calibration model.

# 4.2 Vanilla Probability Calibration

As we outline in Section 2, a probability calibration model $g _ { \theta }$ can be learned in a variety of ways. In this work in order to utilize EM algorithm, we focus on the models $g _ { \theta }$ that can be learned via logistic regression. Specifically, we focus on the case when:

$$
\begin{array} { c l c r } { { \displaystyle Y | f ( X ) \sim \mathrm { B e r n o u l l i } \left( g _ { \theta } ( f ( X ) ) \right. } } \\ { { \displaystyle g _ { \theta } ( w ) = \frac { 1 } { 1 + \exp \{ - s _ { \theta } ( w ) \} } . } } \end{array}
$$

For example, Kull, Silva Filho, and Flach (2017) propose to parameterize $s _ { \theta }$ as a bi-variate function to make the logistic objective correspond to optimizing Beta distribution. Gupta et al. (2021) parameterize $s _ { \theta }$ as a monotonic spline function.

In particular, this probabilistic formulation allows us to directly write the estimation procedure as a likelihood optimization, which in turn allows us to introduce the EM step. Since we focus on the calibration methods that can be learned via logistic regression objective, in the rest of this paper, we generically denote a learning algorithm for $g _ { \theta }$ by

$$
g _ { \theta }  \mathrm { L o g R e g } ( \mathcal { D } ) ,
$$

where $\mathcal { D }$ are the observations of $( X , Y )$ .

Probability Calibration: Case Study In marketing data, the target category is customer conversion. A machine learning model is used to score the probability of converting a customer. The overall goal is to use the model to target customers with higher likelihood of converting, which would simultaneously save the resources whilst retaining the overall effectiveness. The majority of the traffic (usually $> 9 0 \%$ ) is subject to the marketing treatment. The remaining small portion of traffic is exempt from any treatment to serve as holdout or control group for performance measurement and calibration purposes. We denote small clean data from the minority group by $\mathcal { D } _ { S }$ and large noisy data from the majority group by $\mathcal { D } _ { B }$ :

$$
\begin{array} { l } { { \mathcal { D } _ { S } = \{ { \pmb x } _ { l } , y _ { l } \} _ { l \in \mathcal { S } } , } } \\ { { \mathcal { D } _ { B } = \{ { \pmb x } _ { k } , z _ { k } \} _ { k \in \mathcal { B } } , } } \end{array}
$$

where $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { l } }$ and $\scriptstyle { \mathbf { { \mathit { x } } } } _ { k }$ are feature vectors, $y _ { l }$ is an unbiased label, and $z _ { k }$ is the observed but potentially biased label.

Note that the treated sample labels $\{ z _ { k } \} _ { k \in B }$ may contain bias. Intuitively, we want to decouple the default probability of customer conversion from the effect of the marketing campaign. In the treatment group, these two effects are confounded, so vanilla probability calibration can only utilize the small clean dataset $\mathcal { D } _ { S }$ . In the cases of severe label imbalance, this could lead to especially large variance and instability. Therefore, it is of interest to incorporate information from the biased sample $\mathcal { D } _ { B }$ into fitting the calibration model.

# 4.3 Data-Augmented Calibration via Approximate EM Algorithm

We augment the data by incorporating latent variable in the following way

large noisy dataset: $\{ \pmb { x } _ { k } , z _ { k } , y _ { k } \} _ { k \in B }$ ,

where $\{ y _ { k } \} _ { k \in B }$ are latent counterfactual labels (would-be labels if no treatment was applied).

Under this formulation, the observed data and unobserved data become

$$
\mathcal { O } = { \mathcal { D } } _ { S } \cup { \mathcal { D } } _ { B } , \qquad \mathcal { U } = \{ y _ { k } \} _ { k \in \mathcal { B } } .
$$

When the cardinality of clean labels is far less than that of corrupted labels, i.e. $\left| S \right| \ll \left| B \right|$ , data augmentation combining $\mathcal { D } _ { S }$ and $\mathcal { D } _ { B }$ and extracting information from large biased labels provides significant improvement over vanilla probability calibration on $\mathcal { D } _ { S }$ .

The hurdle to utilize the large noisy data is treatmentinduced bias in observed labels $\{ z _ { k } \} _ { k \in B }$ . We consider the unobserved clean labels $\{ y _ { k } \} _ { k \in B }$ of samples in the large noisy data as latent information. We use the approximate EM algorithm introduced in Section 3 to learn probability calibration in the presence of latent data.

We formulate $P _ { \theta } ( \mathcal { U } | \mathcal { O } )$ for our use cases as follows. Let $Z$ denote the label from a customer that has undergone treatment. There are two possible outcomes:

• If a sample has a negative label, we suppose that the treatment did not sufficiently alter the outcome. Hence, we assume that the counterfactual outcome, $Y$ , would still be negative if there was no treatment; • If a sample has undergone a treatment and the outcome is positive, with a non-zero probability the positive outcome could have been caused by the treatment. In other words, counterfactual outcome, $Y$ , could have been positive if there was no treatment.

We formalize our intuition in the following statement.

Assumption 1 (Treatment effect).

We can find such function $h : \mathbb { X } \mapsto [ 0 , 1 ]$ as long as $P ( \boldsymbol { Y } =$ $0 | f ( \boldsymbol { X } ) = 0 ) < 1$ .2

Now let us deduce the form of the function $h$ . Using the law of total probability and Bayes’s theorem, we write

$$
P ( \boldsymbol { Y } = 0 | \boldsymbol { X } ) = P ( \boldsymbol { Z } = 0 | \boldsymbol { X } ) + h ( \boldsymbol { X } ) \cdot P ( \boldsymbol { Z } = 1 | \boldsymbol { X } ) .
$$

Therefore, we have that:

$$
h ( \pmb { X } ) = \frac { P ( Y = 0 | \pmb { X } ) - P ( Z = 0 | \pmb { X } ) } { P ( Z = 1 | \pmb { X } ) } .
$$

![](images/e663a6d59aa7ab91aeae518a0684cc718bd72a9d693ca3f26db0d63911571d5c.jpg)  
Step 1: Model $^ +$ True Labels   
Figure 3: Summary of the CalEM algorithm

Note that $h ( X )$ is a well-defined probability measure: $h ( X ) \in [ 0 , \dot { 1 } ]$ because $P ( Z = 0 | \bar { X _ { } } ) \leq P ( \bar { Y _ { } } = 0 | T =$ $1 , X ) < 1$ , $\forall X \in \mathbb { X }$ .

$$
P ( Y = 0 | Z , { \pmb X } ) = \left\{ \begin{array} { l l } { 1 } & { i f Z = 0 } \\ { h ( { \pmb X } ) } & { i f Z = 1 } \end{array} . \right.
$$

Based on the Assumption (1) and Algorithm 1, we propose a new algorithm called CalEM. We describe CalEM in Algorithm 3. The full derivation of Algorithm 3 can be found in Appendix A.2. We also provide a visual illustration in Figure 3. The algorithm works by first fitting a calibrator $g _ { \theta ^ { ( 0 ) } }$ on the model’s predictions and true labels, and a second calibrator $g _ { \eta }$ on the same predictions and biased labels. Next, it computes a transition function $h$ from $g _ { \theta ^ { ( 0 ) } }$ and $g _ { \eta }$ , and then uses an expectation-maximization (EM) approach to iteratively refit $g _ { \theta ^ { ( t ) } }$ on the combined dataset, guided by $h$ . This process adjusts for label bias and improves the calibration of the model’s predictions.

# 5 Experiments

# 5.1 Simulation

First, we investigate the performance of the proposed approach under controlled settings in simulations in Section 5.1. We then apply our approach to the real marketing datasets in Section 5.2.

In simulation our primary goals twofold: test how the model performance changes with respect to the observed data proportion, event probability, and the choice of transition and calibration functions, $h$ and $g$ respectively. As a baseline model, we choose spline-based calibration model (Gupta et al. 2021), which we denote by GAM. First, we first fit GAM on the clean sample. Then, we enhance GAM with CalEM and re-fit on the large biased dataset.

For simulation, we sample true probabilities from $B e t a ( \alpha , \beta )$ distribution, where $\alpha , \beta$ allows us to control the event probability. To imitate the effect of the miscalibrated classifier, we transform the true probabilities to model scores via function, $g ^ { - 1 }$ . The treatment effect is simulated via the transition function, $h$ . Overall, our sampling models for generating the data is:

$$
\begin{array} { r l r } & { p _ { i } \sim B e t a ( \alpha , \beta ) , } & \\ & { z _ { i } \sim B e r n o u l l i ( p _ { i } ) , } & \\ & { \tilde { p } _ { i } = p _ { i } - h ( p _ { i } ) * p _ { i } , } & \\ & { y _ { i } | p _ { i } , z _ { i } \sim \left\{ \begin{array} { l l } { 0 } & { \mathrm { i f ~ } z _ { i } = 0 } \\ { B e r n o u l l i ( 1 - h ( p _ { i } ) ) } & { \mathrm { i f ~ } z _ { i } = 1 } \end{array} \right. . } \end{array}
$$

In total, we generate $1 0 , 0 0 0$ samples and select some portion of the data to be the clean dataset, $\mathcal { D } _ { S } = \{ g ^ { - 1 } ( \tilde { p } _ { i } ) , \bar { y _ { i } } \}$ , and the rest to be the noisy data, $\mathcal { D } _ { B } = \{ g ^ { - 1 } ( \bar { p } _ { i } ) , z _ { i } \}$ .

For the miscalibration curve $g ^ { - 1 }$ , we explore two design options: $\begin{array} { r } { g _ { 1 } ^ { - 1 } ( p ) = \frac { 1 } { 1 + \exp \{ - 2 0 ( p - 0 . 5 ) \} } } \end{array}$ , which represents an overconfident classifier that pushes scores to the extremes, and $g _ { 2 } ^ { - 1 } ( p ) = p ^ { 3 }$ , which corresponds to a classifier trained on imbalanced data, tending to push scores closer to 0. Similarly, for the transition function, we consider two options: $h _ { 1 } ( p ) \stackrel { \cdot } { = }$ $0 . 5 p ^ { 2 }$ and $h _ { 2 } ( p ) = 0 . 3 \times 1 \{ p \geq 0 . 3 \}$ , where the primary goal is to assess how the smoothness of the transition function impacts the estimation procedure.

For each simulation setup, we train the baseline GAM model on the observed portion, $\mathcal { D } _ { S }$ . We then fit the proposed model on the combined dataset, $\mathcal { D } _ { S } \cup \mathcal { D } _ { B }$ . As a metric, we compute $L ^ { 2 }$ -error between the oracle calibration curve, $g$ , and the estimated calibration curves. To demonstrate the utility of our approach, we compute the percent improvement as $( L _ { G A M } ^ { 2 } - L _ { E M } ^ { 2 } ) / ( L _ { G A M } ^ { 2 } ) * 1 0 0 \%$ . We report the results in Table 1. We observe the proposed method outperform the baseline under all settings. We re-run the simulation 100 times and perform a paired T-test on the $L ^ { 2 }$ errors. We find that at $5 \%$ significance level, our proposed approach achieves a lower $L ^ { 2 }$ error under most scenarios.

Table 1: Simulation results: average percent improvement in $L ^ { 2 }$ -error of the proposed model over the baseline GAM fit on the observed-only data. We indicate with star $( ^ { * } )$ the results that are statistically significant at the $5 \%$ level using paired T-test.   

<html><body><table><tr><td rowspan="2" colspan="2">% I P(Y)</td><td colspan="4">91</td><td colspan="4">92</td></tr><tr><td>0.5</td><td>0.4</td><td>0.3</td><td>0.2</td><td>0.5</td><td>0.4</td><td>0.3</td><td>0.2</td></tr><tr><td rowspan="5">h1</td><td>10%</td><td>+35.55%*</td><td>+24.49%*</td><td>+32.00%*</td><td>+11.26%*</td><td>+21.62%*</td><td>+47.51%*</td><td>+46.96%*</td><td>+52.82%*</td></tr><tr><td>20%</td><td>+21.14%*</td><td>+18.68%*</td><td>+19.45%*</td><td>+7.17%</td><td>+23.97%*</td><td>+33.13%*</td><td>+48.55%*</td><td>+32.87%*</td></tr><tr><td>30%</td><td>+20.12%*</td><td>+24.23%*</td><td>+23.18%*</td><td>-3.62%</td><td>+18.79%*</td><td>+12.91%</td><td>+52.48%*</td><td>+33.40%*</td></tr><tr><td>40%</td><td>+12.49%*</td><td>+12.31%*</td><td>+10.30%*</td><td>+15.46%*</td><td>+15.03%*</td><td>+22.21%</td><td>+27.56%*</td><td>+38.04%*</td></tr><tr><td>50%</td><td>+4.94%</td><td>+7.12%*</td><td>+2.12%</td><td>+4.82%</td><td>+8.58%*</td><td>+4.46%</td><td>+20.08%*</td><td>+25.09%*</td></tr><tr><td rowspan="5">h2</td><td>10%</td><td>+3.95%*</td><td>+2.62%*</td><td>+4.77%*</td><td>+2.94%</td><td>+4.93%*</td><td>+12.23%*</td><td>+31.84%*</td><td>+17.34%*</td></tr><tr><td>20%</td><td>+7.98%*</td><td>+6.56%*</td><td>+10.32%*</td><td>+8.92%*</td><td>+19.60%*</td><td>+23.98%*</td><td>+46.98%*</td><td>+43.49%*</td></tr><tr><td>30%</td><td>+7.32%*</td><td>+4.27%*</td><td>+5.03%*</td><td>+4.55%*</td><td>+11.63%*</td><td>+14.61%*</td><td>+28.72%*</td><td>+29.33%*</td></tr><tr><td>40%</td><td>+3.39%*</td><td>+8.04%*</td><td>+1.44%</td><td>+0.79%</td><td>+6.56%*</td><td>+20.18%*</td><td>+15.64%*</td><td>+28.13%*</td></tr><tr><td>50%</td><td>+3.95%*</td><td>+2.62%*</td><td>+4.77%*</td><td>+2.94%</td><td>+4.93%*</td><td>+12.23%*</td><td>+31.84%*</td><td>+17.34%*</td></tr></table></body></html>

<html><body><table><tr><td></td><td colspan="3">Criteo</td><td colspan="3">Hillstorm</td><td colspan="3">Lenta</td></tr><tr><td></td><td>KS</td><td>Brier</td><td>Log-lik.</td><td>KS</td><td>Brier</td><td>Log-lik.</td><td>KS-error</td><td>Brier</td><td>Log-lik.</td></tr><tr><td>GAM</td><td>0.000489</td><td>0.001656</td><td>0.008835</td><td>0.016997</td><td>0.092418</td><td>0.326527</td><td>0.002450</td><td>0.077768</td><td>0.270673</td></tr><tr><td>IR</td><td>0.000199</td><td>0.001653</td><td>0.008229</td><td>0.016958</td><td>0.092875</td><td>0.343015</td><td>0.002484</td><td>0.077877</td><td>0.271399</td></tr><tr><td>BC</td><td>0.000224</td><td>0.001647</td><td>0.008089</td><td>0.015970</td><td>0.091948</td><td>0.324656</td><td>0.002073</td><td>0.077755</td><td>0.270462</td></tr><tr><td>GAM-EM</td><td>0.000389</td><td>0.001650</td><td>0.008773</td><td>0.016290</td><td>0.092224</td><td>0.325976</td><td>0.002357</td><td>0.077708</td><td>0.270407</td></tr><tr><td>BC-EM</td><td>0.000190</td><td>0.001643</td><td>0.008083</td><td>0.015884</td><td>0.091926</td><td>0.324489</td><td>0.001918</td><td>0.077715</td><td>0.270340</td></tr></table></body></html>

Table 2: Results on the marketing datasets with 10 re-runs. The best results are highlighted in bold; second best results are underlined. We measure Kolmogorov-Smirnov error (Gupta et al. 2020), Brier score (Brier 1950), and log-likelihood. Our EM approach consistently improves on the baseline method.

Based on Table 1, we hypothesize that datasets that would benefit the most from the proposed approach are the ones with mild label imbalance, have small observed data size compared to the latent data, and where the influence function $h$ is suspected to be relatively smooth. For the imbalanced datasets, the variance of the calibration curve is heterogenuous and high; therefore, the calibration error improves from adding more data. However, if the observed data sample size is large enough, then there is no room for improvement, so the proposed approach may not yield a significant boost. Regarding smoothness, since we use GAM as our baseline which is piece-wise polynomial, this dictates a smoothness assumption on $h$ by (9). Finally, the shape of the true calibration curve does not seem to affect our method.

# 5.2 Real Data Application

For our real data analysis, we compare our proposed approach against the mainstream calibration methods: Beta calibration (Kull, Silva Filho, and Flach 2017), isotonic regression (Groeneboom and Lopuhaa 1993), and GAM (Gupta et al. 2021).

We use the datasets Criteo (Diemert et al. 2018), Hillstrom (Hillstrom 2008), and Lenta (Lenta 2020). For all datasets, the goal is to develop a well-calibrated classifier to predict the treatment outcome. Each dataset contains a feature set, a target column, and a binary treatment indicator. Criteo (Diemert et al. 2018) consists of data of 13 million users, each one represented by 12 features with the overall treatment ratio of $8 4 . \mathrm { \bar { 6 } \% }$ . Hillstrom (Hillstrom 2008) consists of the records of 64,000 users, described by 8 features with the overall treatment ratio of 0.6. Finally, Lenta (Lenta 2020) contains information on 687,000 users, described by 193 features with the overall treatment ratio of $7 5 \%$ .

To test the proposed approach, we randomly split the observed portion of the dataset, consisting of untreated users, into training, calibration, and test sets with a ratio of $9 0 \% - 5 \% - 5 \%$ , respectively. We use the training dataset to train the XGBoost classifier (Chen and Guestrin 2016). We use the calibration set for fitting the calibration model. Finally, we test the calibration on the test set, measuring Kolmogorov-Smirnov error (Gupta et al. 2020), Brier score

Input: read a small clean dataset $\mathcal { D } _ { S } ~ + ~$ a large noisy dataset $\mathcal { D } _ { B }$ , and specify tolerance $\varepsilon$ ;   
Process: Construct dataset

$$
\begin{array} { l l l } { \mathcal { D } ^ { * } } & { = } & { \mathcal { D } _ { S } \cup \mathcal { D } _ { B } \cup \{ { \pmb x } _ { k } , y _ { k } = 0 \} _ { k \in \mathcal { B } } ; } \end{array}
$$

Initialization: learn an initial calibration function $g _ { \theta ^ { ( 0 ) } }$ using $\mathcal { D } _ { S }$ and a calibration function on the biased dataset using $\mathcal { D } _ { B }$ using logistic regression objective, i.e.,

$$
g _ { \theta ^ { ( 0 ) } } \gets \mathrm { L o g R e g } ( \mathcal { D } _ { S } ) \quad g _ { \eta } \gets \mathrm { L o g R e g } ( \mathcal { D } _ { B } ) ;
$$

# Computing an initial estimate of “treatment effect”:

$$
h _ { \theta ^ { ( 0 ) } , \eta } ( \pmb { x } _ { j } ) = \frac { g _ { \eta } ( f ( \pmb { x } _ { j } ) ) - g _ { \theta ^ { ( 0 ) } } ( f ( \pmb { x } _ { j } ) ) } { g _ { \eta } ( f ( \pmb { x } _ { j } ) ) } ;
$$

Expectation-maximization iterations for $t = 0 , 1 , 2 , \cdots$ : 1. compute weight vector:

$$
\mathbf { \boldsymbol { w } } ^ { ( t ) } = \left( \mathbf { 1 } _ { | \mathcal { S } | } ^ { \mathrm { T } } , \mathbf { \boldsymbol { w } } _ { 1 } ^ { ( t ) , \mathrm { T } } , \mathbf { \boldsymbol { w } } _ { 2 } ^ { ( t ) , \mathrm { T } } \right) ^ { \mathrm { T } } ,
$$

where $\mathrm { \Delta T }$ denotes vector transpose, and

• $\mathbf { 1 } _ { | \cal S | }$ is a column vector of 1’s whose dimension equals the cardinality of $s$ , i.e. $| { \cal S } |$ , $w _ { 1 } ^ { ( t ) }$ eisntaisc i,on $| B |$ , and its $j$ -th ${ \pmb w } _ { 1 , j } ^ { ( t ) } = 1 - h _ { \theta ^ { ( t ) } , \eta } ( { \pmb x } _ { j } )$   
${ \pmb w } _ { 2 } ^ { ( t ) }$ is a column vector of dimension $| B |$ , and its $j$ -th   
element is ${ \pmb w } _ { 2 , j } ^ { ( t ) } = h _ { \theta ^ { ( t ) } , \eta } ( { \pmb x } _ { j } )$ ;

2. update calibration function $g _ { \theta ^ { ( t ) } } \to g _ { \theta ^ { ( t + 1 ) } }$ by:

$$
g _ { \theta ^ { ( t + 1 ) } } \gets \mathrm { L o g R e g } ( \mathcal { D } ^ { * } ; \pmb { w } ^ { ( t ) } ) ;
$$

3. update “treatment effect” estimate:

$$
h _ { \theta ^ { ( t + 1 ) } , \eta } ( \pmb { x } _ { j } ) = \frac { g _ { \eta } ( f ( \pmb { x } _ { j } ) ) - g _ { \theta ^ { ( t + 1 ) } } ( f ( \pmb { x } _ { j } ) ) } { g _ { \eta } ( f ( \pmb { x } _ { j } ) ) } ;
$$

4. continue until $\left\| \theta ^ { ( t + 1 ) } - \theta ^ { ( t ) } \right\| < \varepsilon .$

(Brier 1950), and log-likelihood. We report the mean error results across 10 runs in Table 2. As we outline in Section 4.2 both Beta calibration (Kull, Silva Filho, and Flach 2017) and GAM (Gupta et al. 2020) are likelihood-based approaches. Following Algorithm 3, we combine them with our proposed CalEM method. Upon analyzing the results, we find that CalEM consistently enhances the performance of the baseline algorithms and achieves optimal performance across all metrics. These outcomes underscore the effectiveness of CalEM in incorporating additional biased data in fitting the calibration model.

# 6 Conclusion

In this paper, we introduce an iterative approach based on a modified EM algorithm that allows the incorporation of large, biased data sources into the estimation of the probability calibration function. The proposed algorithm can serve as a drop-in enhancement for any likelihood-based recalibration method. Additionally, the method benefits from convergence guarantees inherited from the EM formulation. The significant performance improvements demonstrated in both simulated environments and real-world marketing datasets underscore the effectiveness and practical applicability of the proposed model in enhancing the accuracy of probability calibration under a variety of conditions.

# Acknowledgments

Parts of this work were completed during Renat Sergazinov’s internship at Amazon. Renat Sergazinov would like to express sincere gratitude to the Amazon Buyer Risk Prevention team for their support, guidance, and invaluable feedback throughout the project. Their collective expertise and encouragement were instrumental in shaping the direction and depth of this research.