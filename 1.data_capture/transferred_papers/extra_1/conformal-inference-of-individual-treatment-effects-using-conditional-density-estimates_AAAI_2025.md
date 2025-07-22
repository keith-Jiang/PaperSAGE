# Conformal Inference of Individual Treatment Effects Using Conditional Density Estimates

Baozhen Wang, Xingye Qiao

Department of Mathematics and Statistics, Binghamton University bwang62, xqiao @binghamton.edu

# Abstract

In an era where diverse and complex data are increasingly accessible, the precise prediction of individual treatment effects (ITE) becomes crucial across fields such as healthcare, economics, and public policy. Current state-of-the-art approaches, while providing valid prediction intervals through Conformal Quantile Regression (CQR) and related techniques, often yield overly conservative prediction intervals. In this work, we introduce a conformal inference approach to ITE using the conditional density of the outcome given the covariates. We leverage the reference distribution technique to efficiently estimate the conditional densities as the score functions under a two-stage conformal ITE framework. We show that our prediction intervals are not only marginally valid but are narrower than existing methods. Experimental results further validate the usefulness of our method.

# 1 Introduction

Understanding the effect of interventions on an individual level is important across many domains, such as healthcare, economics, and public policy. Traditional average treatment effect estimates consider all individuals but fail to account for the heterogeneity in individual responses. As diverse data from various fields become more accessible, machine learning plays an increasingly significant role in revealing insights from the data.

In recent years, many research works have focused on machine learning algorithms that provide point estimates of the Conditional Average Treatment Effect (CATE) (Athey and Imbens 2016; Wager and Athey 2018; K¨unzel et al. 2019; Meng and Qiao 2020), which quantifies the expected difference in treatment outcomes for individuals with specific characteristics. While CATE takes into account the varying covariates of the individuals by averaging effects across similar individuals, it can still sometimes overlook individuallevel heterogeneity. Recent works on Individualized Treatment Effect (ITE) have made significant progress in addressing these limitations, offering more accurate predictions tailored to each individual (Hill 2011; Alaa and Van Der Schaar 2017; Shalit, Johansson, and Sontag 2017). They utilize Gaussian processes or Bayesian approaches to provide interval-valued predictions at the individual level. However, these methods could be model-specific, and not always ensure that the predicted confidence intervals achieve the nominal coverage probability, i.e., the proportion of times that the true treatment effect lies within the constructed intervals across different samples. This issue can limit the reliability and generalizability of the treatment effect estimates, particularly when applied to diverse patient populations or under varying clinical conditions.

Conformal prediction (Vovk, Gammerman, and Shafer 2005; Lei, Robins, and Wasserman 2013; Lei et al. 2018) provides a model-agnostic and distribution-free framework that outputs interval predictions with the desired coverage probability. The main idea of (split) conformal prediction is to compute a conformity score by fitting a predictive model on the training set, and then evaluate the conformity score on the calibration set to quantify the uncertainty of future predictions. Although the conformal framework is modelagnostic and always achieves a coverage guarantee, the average length of resulting prediction intervals can highly rely on the choice of the score functions. Lei and Cande\`s (2021) propose a two-stage methodology utilizing weighted conformal prediction (WCP) (Tibshirani et al. 2019) to address the covariate shift problem during counterfactual inference. This method is considered as the state-of-the-art method which provides prediction intervals for ITE problems, with desired coverage guarantee as well as reasonably short intervals. Chen et al. (2024) study a similar approach, where they use the joint density to compute weights in WCP. Alternatively, Alaa, Ahmad, and van der Laan (2023) develop a methodology called conformal meta-learner, which applies conformal prediction directly with imputed pseudo outcomes. Both methods utilize Conformal Quantile Regression (CQR) (Romano, Patterson, and Candes 2019), an approach that integrates the concept of conformal prediction with quantile regression. CQR is capable of adapting to any heteroscedasticity in the data, but may not guarantee the shortest prediction intervals. Additionally, the two-stage design inherent in both methods tends to produce conservative results. Experimental studies consistently show that these methods yield conservative prediction intervals with a coverage much greater than the desired level (Lei and Cande\`s 2021; Alaa, Ahmad, and van der Laan 2023). Motivated by the need for more precise prediction interval for ITE, this work aims to refine these methodologies to achieve shorter prediction intervals while maintaining the coverage guarantee.

Inspired by the conformal predictions under the classification setting (Lei 2014; Sadinle, Lei, and Wasserman 2019), one can show that directly using the conditional density of the outcome $Y$ given the covariates $X$ as the score function in conformal prediction will lead to the shortest prediction interval (see discussion in Section 3.1). While a substantial amount of research has focused on estimating the regression function $E ( y | x )$ , the task of estimating the full conditional density $f ( y | x )$ , particularly in scenarios where $x$ is high-dimensional, has received considerably less attention. Izbicki, Shimizu, and Stern (2022) proposed a framework that utilizes conditional densities under regression setting. However, their focus is on achieving local conditional coverage and they employ a non-parametric smoothing technique to estimate the conditional densities, which is less efficient.

In this paper, we introduce a novel approach to conformal inference of individual treatment effects (ITE) using conditional densities as the score function. To address the computational difficulties associated with estimating full conditional densities, we employ a reference distribution technique to alleviate the problem. We theoretically show that the proposed method achieves shorter prediction intervals as well as maintaining the desired coverage guarantee. Empirical studies, including both simulations and semi-synthetic benchmarks, strongly indicate that our proposed method surpasses existing state-of-the-art methods in prediction length.

The remainder of this article is outlined as follows. We begin by reviewing the background knowledge of the potential outcome framework and conformal prediction in Section 2. We introduce our methodologies along with the discussion of theoretical guarantee in Section 3. In Section 4, we present supporting simulation and semi-synthetic experiments. Section 5 concludes by summarizing our contributions and the practical implications. Proofs and additional discussions can be found in the supplementary material.

# 2 Background

In this section, we first introduce the standard potential outcome framework. Then we review the conformal prediction in Section 2.2 and the weighted conformal prediction under covariate shift in Section 2.3.

# 2.1 Potential Outcome Framework and Objective Statement

We focus on the standard potential outcome framework (Neyman 1923; Rubin 1974) with a binary treatment. Denote by $A ~ = ~ \{ 0 , 1 \}$ the binary treatment indicator, by $X \in \dot { \mathcal { X } } \subseteq \mathbb { R } ^ { d }$ the covariates, by $Y \in \mathcal { V } \subseteq \mathbb { R }$ the observed outcome. For each subject $i$ , let $Y _ { i } ( 1 )$ and $Y _ { i } ( 0 )$ be the pair of potential outcomes under $A = 1$ and $A = 0$ respectively. The fundamental problem of causal inference is that we can only observe one potential outcome out of $Y _ { i } ( 1 )$ and $Y _ { i } ( 0 )$ for each subject (Holland 1986). Let

$$
( X _ { i } , A _ { i } , Y _ { i } ( 1 ) , Y _ { i } ( 0 ) ) \stackrel { \mathrm { i . i . d } } { \sim } P ( X , A , Y ( 1 ) , Y ( 0 ) ) .
$$

The following assumptions are often considered: (1) Unconfoundedness (strong ignorability): $( Y ( 1 ) , Y ( 0 ) ) \perp A | X .$ , which allows us to interpret the differences in outcomes as causal effects, rather than being confounded by other factors. Under unconfoundedness, the conditional distributions of a potential outcome are invariant across treatment groups: $P ( { \bar { Y } } | X , A = a ) = P ( Y ( a ) | X )$ . (2) Stable unit treatment value assumption (SUTVA): $Y _ { i } = Y _ { i } ( A )$ , which ensures individual treatment effects are not influenced by other units. (3) Positivity: $0 < P ( A = 1 | X = x ) < 1$ , which ensures that every individual has a nonzero probability of receiving each treatment condition.

Existing methods mostly focused on conditional average treatment effects (CATE) (Athey and Imbens 2016; Wager and Athey 2018; Ku¨nzel et al. 2019), defined as $\tau ( \bar { x ) } = E ( Y ( 1 ) \bar { \mathbf { \tau } } - Y ( 0 ) | X = x )$ . In this work, our primary focus is on individual treatment effects (ITE), defined as $\dot { Y _ { i } } ( 1 ) - Y _ { i } ( 0 )$ for subject $i$ without knowing the treatment assignment, and to construct a prediction interval of the ITE. Given the observations $( X _ { i } , Y _ { i } , A _ { i } ) , i = 1 , \dots , n$ , our goal is to construct a predictive interval $\hat { C } ( x )$ that covers the true ITE for a new test individual $n { + 1 }$ with covariate $X _ { n + 1 }$ with high probability, i.e.

$$
\begin{array} { r } { \mathbb { P } \left( Y _ { n + 1 } ( 1 ) - Y _ { n + 1 } ( 0 ) \in \hat { C } ( X _ { n + 1 } ) \right) \geq 1 - \alpha , } \end{array}
$$

for a pre-specified level $\alpha \in ( 0 , 1 )$ . Typically, $\alpha$ is a small value, such as 0.05.

# 2.2 Conformal Prediction

Conformal prediction (Vovk, Gammerman, and Shafer 2005; Lei, Robins, and Wasserman 2013; Lei et al. 2018) provides a means to a prediction set that with a predetermined probability covers the true value for future individuals based on a finite sample. Below we describe the construction of the original conformal prediction set. We first choose a score function $S ( \cdot , \cdot )$ , whose arguments consist of a point $( x , y )$ , and some dataset $D$ . By convention, a low value of $S ( ( x , y ) , D )$ indicates that the point $( x , y )$ “conforms” to $D$ , whereas a high value indicates that $( x , y )$ is atypical relative to the points in $D$ . For convenience of the method development below in this article, we choose $S$ to be a conformity, instead of nonconformity, score; that is, a high value of $S$ indicates that $( x , y )$ “conforms” to $D$ .

Given a training data set $( X _ { i } , Y _ { i } ) , i = 1 , . . . , n$ , and a fixed $\boldsymbol { x } \in \mathbb { R } ^ { d }$ , we obtain ${ \hat { C } } ( x ) \subseteq \mathbb { R }$ , the conformal prediction set, by repeating the following procedure for each $y _ { \mathrm { t r i a l } } \ \in$ R: we first calculate the conformity scores Vi(x,ytrial) $S ( ( X _ { i } , Y _ { i } ) , \bigcup _ { i = 1 } ^ { n } ( X _ { i } , Y _ { i } ) \cup \{ ( x , y _ { \mathrm { t r i a l } } ) \} )$ , for $i = 1 , \ldots , n$ , aclnud $V _ { n + 1 } ^ { ( x , y _ { \mathrm { t r i a l } } ) } = S ( ( x , y _ { \mathrm { t r i a l } } ) , \bigcup _ { i = 1 } ^ { n } ( X _ { i } , Y _ { i } ) )$ $y _ { \mathrm { { t r i a l } } }$ $\hat { C } ( x )$ $V _ { n + 1 } ^ { ( x , y _ { \mathrm { t r i a l } } ) } ~ \geq ~ \mathrm { Q u a n t i l e } ( \alpha ; V _ { 1 : n } ^ { ( x , y _ { \mathrm { t r i a l } } ) } \cup$ $\{ \infty \} )$ , that is, if no less than $\alpha ( n + 1 )$ many of $V _ { i } ^ { ( x , y _ { \mathrm { t r i a l } } ) } \mathbf { \bar { s } }$ are no greater than $V _ { n + 1 } ^ { ( x , y _ { \mathrm { t r i a l } } ) }$ . Importantly, the symmetry in the construction of the conformity scores guarantees a satisfactory coverage rate in finite samples (Lei et al. 2018):

$$
\begin{array} { r } { \mathbb { P } ( Y \in \hat { C } ( X ) ) \geq 1 - \alpha , } \end{array}
$$

where $\mathbb { P }$ is taken over the $n + 1$ i.i.d. draws of training samples and the test point.

The above original version of conformal prediction provides a finite-sample guarantee of the coverage rate. However, it can be computationally expensive, especially if $S$ has to be computed through an expensive machine-learning method. A more popular alternative is the split-conformal method (Papadopoulos et al. 2002; Vovk, Gammerman, and Shafer 2005), where the entire training data is split into two parts. The first part is used to estimate the score function $S$ , which is then evaluated on the second part of the data. The splitting process not only alleviates the computational burden of the full conformal prediction but also mitigates the risk of overfitting, as the score function is calibrated on a separate set from where it was trained.

# 2.3 Weighted Conformal Prediction under Covariate Shift

Both the original and the split version of conformal prediction assume that the distributions of the target data and the training data are the same. Tibshirani et al. (2019) generalized conformal prediction for regression to WCP under covariate shift assumptions. Covariate shift (Shimodaira 2000; Sugiyama et al. 2007) refers to the scenario where the marginal distribution of covariates differs between the training and target data set, denoted as $P _ { X }$ and $Q _ { X }$ respectively, while the conditional distributions of the outcome given the covariates remain the same, denoted as $P _ { Y \mid X } .$ . Lei and Cande\`s (2021) studied the counterfactuals inference utilizing WCP under covariate shift, where

Target: $( X , Y ) \sim Q _ { X } \times P _ { Y \mid X }$ .

Here we briefly go over the algorithm. Assume that the probability measure of the target data covariates is absolutely continuous with respect to that of the training data covariates, we consider using the Radon-Nikodym derivative $w ( x ) = d Q _ { X } ( x ) / d P _ { X } ( x )$ to address the shift in covariate distribution. Define $p _ { i } ( \boldsymbol { x } ) \stackrel { } { = } w ( x _ { i } ) / [ \sum _ { i ^ { \prime } = 1 } ^ { n } w ( x _ { i ^ { \prime } } ) + w ( x ) ]$ , for $i = 1 , \ldots , n$ and $\begin{array} { r } { p _ { n + 1 } ( x ) \ = \ w ( x ) / [ \sum _ { i ^ { \prime } = 1 } ^ { n } w ( x _ { i ^ { \prime } } ) \ + \ } \end{array}$ $w ( x ) ]$ . We can use weighted quantile of the scores computed in the calibration data as the cutoff value, with $p _ { i } ( x )$ as the weight, to obtain the prediction set:

$$
\begin{array} { l } { \displaystyle \hat { C } ( \boldsymbol { x } ) = \biggl \{ y \in \mathbb { R } : V _ { n + 1 } ^ { ( x , y ) } } \\ { \geq \mathrm { Q u a n t i l e } \biggl ( \alpha ; \displaystyle \sum _ { i = 1 } ^ { n } p _ { i } ( x ) \delta _ { V _ { i } ^ { ( x , y ) } } + p _ { n + 1 } ( x ) \delta _ { \infty } \biggr ) \biggr \} , } \end{array}
$$

where $\delta _ { c }$ is a Dirac measure placing a point mass at $c$ . Assume we have the true value of $w ( x )$ , Tibshirani et al. (2019) showed that $\hat { C } ( \boldsymbol { x } )$ satisfies:

$$
\begin{array} { r } { \mathbb { P } _ { ( X , Y ) \sim Q _ { X } \times P _ { Y \mid X } } \big ( Y \in \hat { C } ( X ) \big ) \geq 1 - \alpha . } \end{array}
$$

Under the potential outcome framework, the covariate distribution of the training data is a mixture of $P _ { X \mid A = 1 }$ and $P _ { X \mid A = 0 }$ . WCP can not directly handle training data of a mixed type due to computational challenges associated with weighted calculations. Lei and Cande\`s (2021) propose a two-stage framework to overcome this issue. On the first stage, one can use training data from the treatment group $P _ { X \mid A = 1 } \times P _ { Y \mid X }$ , to produce interval estimates for those from control group $P _ { X | A = 0 } \times P _ { Y | X }$ via WCP and vice versa. Then, in the second stage, one can integrate the interval outcomes from both groups using a secondary conformal prediction procedure or a naive Bonferroni correction. In Section 3.3, we will explore this two-stage framework, and propose another less conservative alternative using the concept of X-learner (Ku¨ nzel et al. 2019).

# 3 Methodology

In this section, we begin by demonstrating how using the conditional density as the score function can optimize the length of the conformal prediction interval. In Section 3.2, we introduce a reference distribution technique for efficient estimation of conditional densities. In Section 3.3, we adopt the two-stage framework proposed by Lei and Cande\`s (2021) to develop algorithms that compute shorter prediction intervals for the Individual Treatment Effect (ITE) of new subjects, ensuring desired coverage guarantees.

# 3.1 Conditional Density as the Score Function

Let $\mathbb { P }$ denote the joint distribution of $( X , Y )$ and $f$ denote the density of $\mathbb { P }$ with respect to Lebesgue measure. Throughout the article, we denote $f ( y | x ) = \bar { f ( Y = y | X = x ) }$ as the conditional density of $Y$ equaling $y$ given $X$ equals $x$ .

We define $C : \dot { \mathbb { R } ^ { d } }  \dot { \mathcal { M } ( \mathbb { R } ) }$ , where $\mathcal { M } ( \mathbb { R } )$ represents the set of all measurable intervals over $\mathbb { R }$ . The function $C$ serves as a confidence interval predictor, which provides an interval $C ( x )$ intended to contain the response variable $y$ , based on input $x$ . Consider the following optimization problem that minimizes the expected length of these intervals while ensuring that the probability of $y$ falling within $C ( \boldsymbol x )$ is at least the predefined level $1 - \alpha$ :

$$
\operatorname* { m i n } _ { C } \mathbb { E } \left\{ | C ( X ) | \right\} \quad { \mathrm { s u b j e c t ~ t o } } \quad \mathbb { P } \left\{ y \in C ( X ) \right\} \geq 1 - \alpha
$$

Theorem 1. Let $t _ { \alpha }$ denote the $\alpha$ quantile of $f ( Y | X \ =$ $\mathbf { \Psi } _ { x } )$ . The solution that optimizes (3) is given by $\begin{array} { r l } { C _ { \alpha } ^ { * } } & { { } = } \end{array}$ $\{ ( x , y ) : f ( y | x ) \geq t _ { \alpha } \}$ . And the optimal predictor can be written as

$$
C _ { \alpha } ^ { * } ( x ) = \{ y : f ( y | x ) \geq t _ { \alpha } \} .
$$

The proof can be found in the supplementary material. A similar problem of (3) has been explored under classification contexts by Lei (2014) and Sadinle, Lei, and Wasserman (2019). According to Theorem 1, if we can consistently estimate $f ( y | x )$ and apply it as a score function in conformal prediction, as detailed in Algorithm 1, we can optimize the length of the prediction interval while ensuring a coverage guarantee of at least $1 - \alpha$ .

# 3.2 Estimate Conditional Density Using Reference Distribution Technique

Using $f ( y | x )$ as the score function in conformal prediction can be optimal; however, estimating the full conditional density $f ( y | x )$ presents significant challenges. Traditional methods like the non-parametric kernel density estimator (De Gooijer and Zerom 2003) must address each dimension of $\boldsymbol { x } ^ { \mathrm { ~ \scriptsize ~ \in ~ } \mathbb { R } ^ { d } }$ and often struggle due to the curse of dimensionality. Contemporary approaches, such as tree-based (Holmes, Gray, and Isbell 2012) and neural network methods (Rothfuss et al. 2019), involve complex computations and may prove less efficient for large-scale applications. To effectively address this issue, we adapt a reference distribution technique originally intended for unconditional density estimation (Hastie et al. 2009). The details of this adaptation are described as follows.

Suppose we have $n$ i.i.d. random samples drawn from the joint density $\begin{array} { r l r } { f ( x , y ) } & { { } = } & { f ( y | x ) h ( x ) } \end{array}$ , denoted as $( x _ { 1 } , y _ { 1 } ) , ( x _ { 2 } , y _ { 2 } ) , \ldots , ( x _ { n } , y _ { n } )$ . We use a reference probability density function $f _ { 0 } ( y )$ , from which a sample of size $n$ independent of $h ( x )$ is drawn using Monte Carlo methods, denoted as $\tilde { y } _ { 1 } , \tilde { y } _ { 2 } , \ldots , \tilde { y } _ { n }$ . We then combine a duplicate of $x _ { 1 } , x _ { 2 } , \ldots , x _ { n }$ with these $\tilde { y }$ to form a joint reference distribution $f _ { 0 } ( x , y ) = f _ { 0 } ( y ) h ( x )$ . By assigning $Z = 1$ to each data point from $f ( x , y )$ and $Z = 0$ to those from $f _ { 0 } ( x , y )$ , we estimate $\mu ( x , y ) : = E ( Z | x , y )$ by supervised learning using the aggregated dataset:

$$
\mu ( x , y ) = \frac { f ( x , y ) } { f ( x , y ) + f _ { 0 } ( x , y ) } = \frac { f / f _ { 0 } } { 1 + f / f _ { 0 } } ,
$$

The resulting estimate, $\hat { \mu } ( x , y )$ , can be inverted to provide an estimate for the joint density

$$
\hat { f } ( x , y ) = f _ { 0 } ( x , y ) \cdot \frac { \hat { \mu } ( x , y ) } { 1 - \hat { \mu } ( x , y ) } .
$$

Dividing $h ( x )$ on both sides, we obtain an estimator of conditional density:

$$
\hat { f } ( y | x ) = f _ { 0 } ( y ) \cdot \frac { \hat { \mu } ( x , y ) } { 1 - \hat { \mu } ( x , y ) } .
$$

Techniques such as logistic regression and random forests, which efficiently estimate log-odds $\log ( f / f _ { 0 } )$ , are natural choices for this procedure. Generally, many reference density can be used for $f _ { 0 } ( y )$ , provided that the support of the reference density covers that of the original one. However, in practice, the accuracy of ${ \hat { f } } ( y | x )$ can be influenced by the choice of $f _ { 0 } ( y )$ . We recommend using a Gaussian distribution with the same mean and a slightly larger variance than the original $y$ ’s to alleviate the extreme case of none overlapping.

# 3.3 Weighted Conformal Inference Using Conditional Density Estimates

By leveraging the reference distribution technique to efficiently estimate the conditional density, we can apply the weighted conformal prediction (Tibshirani et al. 2019) to derive an estimate of (6). As introduced in Section 2.3, Lei and Cande\`s (2021) propose a two-stage framework to overcome the challenges of WCP when training data exhibit a mixed distribution of covariates under the potential outcome framework. A two-stage framework seems a common and necessary choice to compensate for never observing the potential outcome. Alaa, Ahmad, and van der Laan (2023) also utilize a two-stage framework, where they first impute a pseudo outcome and then apply conformalized quantile regression (CQR) on these pseudo outcomes along with the covariates. A central motivation of this paper is to reduce the length of the prediction interval as much as possible, due to the inherent conservativeness of prediction intervals under the potential outcome framework.

Algorithm 1 below outlines the procedure of the first stage, where we implement WCP using the conditional density estimate ${ \hat { f } } ( Y | X )$ as the score function. Within the first stage of our framework, Algorithm 1 is implemented twice: once using training data from the treatment group to obtain interval estimates for the control group, and conversely, using control group data to estimate intervals for the treatment group. In the former scenario, the weight function $w ( x )$ is calculated as:

$$
\frac { d P _ { X | A = 0 } ( x ) } { d P _ { X | A = 1 } ( x ) } \propto \frac { P ( A = 0 | X = x ) } { P ( A = 1 | X = x ) } = \frac { 1 - \pi ( x ) } { \pi ( x ) }
$$

where $\pi ( x ) : = P ( A = 1 | X = x )$ is the propensity score (Rosenbaum and Rubin 1983). This score captures the treatment assignment mechanism under given covariate conditions.

Similarly, when using data from the control group to estimate intervals for the treatment group, the weight function is:

$$
w ( x ) = \frac { d P _ { X | A = 1 } ( x ) } { d P _ { X | A = 0 } ( x ) } \propto \frac { \pi ( x ) } { 1 - \pi ( x ) }
$$

Upon implementing Algorithm 1 in both scenarios, we first partition the training data into two splits, indexed by $\mathcal { T } _ { 1 }$ and $\mathcal { T } _ { 2 }$ . The first split is used to estimate the weight function $\hat { w } ( x )$ and the conditional density estimate ${ \hat { f } } ( Y | X )$ using the reference distribution technique. We then evaluate both $\hat { w } ( x )$ and ${ \hat { f } } ( Y | X )$ on the second split, then compute the threshold $\hat { t } _ { \alpha }$ as the $\alpha$ quantile of the weighted conformity scores.

Theorem 2. Under Algorithm $\boldsymbol { I }$ , if the non-conformity scores $V _ { i }$ have no ties almost surely, $Q _ { X }$ is absolutely continuous with respect to $P _ { X }$ , and $\mathbb { E } [ \hat { w } ( X ) ] < \infty .$ , then, given the true weights, i.e., $\hat { w } ( \cdot ) = w ( \cdot )$ :

$$
P _ { ( X , Y ) \sim Q _ { X } \times P _ { Y \mid X } } ( Y _ { n + 1 } \in \hat { C } ( X _ { n + 1 } ) ) \geq 1 - \alpha .
$$

In general, if $\hat { w } ( \cdot ) \neq w ( \cdot )$ , define $\begin{array} { r } { \Delta w = \frac { 1 } { 2 } \mathbb { E } _ { X \sim P _ { X } } | \hat { w } ( X ) - } \end{array}$ $w ( X ) |$ . In this case, coverage is lower bounded by $1 - \alpha -$ $\Delta w$ .

Theorem 2 establishes that for any choice of target covariate distribution $Q _ { X }$ , it is possible to obtain a prediction interval for the outcome with a desired coverage guarantee. The first part of Theorem 2 is a split version of Theorem 2 from (Tibshirani et al. 2019) and the second part adapts from Theorem 3 in (Lei and Cande\`s 2021). We provide proofs in the supplementary material for completeness.

In stage one, we implement Algorithm 1 twice: once using training data from the treatment group in $\mathcal { T } _ { 1 }$ with weights defined in (7) to compute $\hat { C } _ { i } ( X _ { i } )$ for $i \in \mathcal { I } _ { 2 }$ where $A _ { i } =$ 0, and once using data from the control group in $\mathcal { I } _ { 1 }$ with weights in (8) to compute $\hat { C } _ { i } ( X _ { i } )$ for $i \in \mathcal { I } _ { 2 }$ where $A _ { i } = 1$ .

Input: Level $\alpha$ , data $( X _ { i } , Y _ { i } )$ from $P _ { X } \times P _ { Y \mid X }$ where $i \in \mathcal { I }$ , and a test point $X _ { n + 1 }$ from $Q _ { X }$ Output: A prediction set $\hat { C } ( x )$   
1: Split $\boldsymbol { \mathcal { T } }$ into two equal sized subsets $\mathcal { I } _ { 1 }$ and $\mathcal { T } _ { 2 }$ .   
2: Estimate the weight function $\hat { w } ( x )$ using $\mathcal { T } _ { 1 }$ .   
3: For each $i \in \mathcal { I } _ { 1 }$ , generate $\tilde { Y } _ { i }$ from a normal distribution with the same mean and slightly larger variance of data in $\mathcal { I } _ { 1 }$ . Assign $Z = 1$ to $( X _ { i } , Y _ { i } )$ and $Z = 0$ to $( X _ { i } , \tilde { Y } _ { i } )$ , then fit a classification algorithm $\hat { \mu }$ to obtain ${ \hat { f } } ( Y | X )$ according to Section 3.2.   
4: For each $i \in \mathcal { I } _ { 2 }$ , compute the score $V _ { i } = { \hat { f } } ( Y _ { i } | X _ { i } )$ and the weight $\hat { w } ( X _ { i } )$   
5: Compute the normalized weights $\begin{array} { r l r l } { \hat { p } _ { i } ( x ) } & { { } } & { = } \end{array}$ $\begin{array} { r } { \hat { w } ( X _ { i } ) / \left[ \sum _ { i \in \mathcal { T } _ { 2 } } \hat { w } ( X _ { i } ) + \hat { w } ( X _ { n + 1 } ) \right] } \end{array}$ and $\hat { p } _ { n + 1 } ( x ) \ =$ $\textstyle \hat { w } ( X _ { n + 1 } ) / \left[ \sum _ { i \in \mathbb { Z } _ { 2 } } \hat { w } ( X _ { i } ) + \hat { w } ( X _ { n + 1 } ) \right]$   
6: Compute $\hat { t } _ { \alpha }$ as the $\alpha$ -th quantile of the distribution   
7: $\begin{array} { r l } & { \sum _ { i \in \mathcal { Z } _ { 2 } } \hat { p } _ { i } ( x ) \delta _ { V _ { i } } + \hat { p } _ { \infty } ( x ) \delta _ { \infty } } \\ & { \mathbf { r e t u r n } \hat { C } ( X _ { n + 1 } ) = \big \{ y : \hat { f } ( y | X _ { n + 1 } ) \geq \hat { t } _ { \alpha } \big \} } \end{array}$

We obtain prediction sets $\hat { C } _ { i } ( X _ { i } )$ for each point in $\mathcal { T } _ { 2 }$ , which satisfy

$$
\mathbb { P } ( Y _ { i } ( 1 - j ) \in \hat { C } _ { i } ( X _ { i } ) | A _ { i } = j ) \geq 1 - \alpha .
$$

Recall for each individual $\mathbf { \chi } _ { i }$ with $A _ { i } = j$ , we can observe its factual outcome $Y _ { i } = Y _ { i } ( j )$ . That means we can obtain a confidence interval of $\mathrm { I T E } _ { i } = Y _ { i } ( 1 ) - Y _ { i } ( 0 )$ for each $i \in \mathcal { I } _ { 2 }$ by subtraction. Specifically, for those in the treatment group, i.e., $A _ { i } = 1$ ,

$$
\begin{array} { r } { \hat { C } _ { i } = \big [ Y _ { i } ( 1 ) - \operatorname* { m a x } ( \hat { C } ( X _ { i } ) ) , Y _ { i } ( 1 ) - \operatorname* { m i n } ( \hat { C } ( X _ { i } ) ) \big ] , } \end{array}
$$

and for those in the control group, i.e., $A _ { i } = 0$ ,

$$
\hat { C } _ { i } = \left[ \operatorname* { m i n } ( \hat { C } ( X _ { i } ) ) - Y _ { i } ( 0 ) , \operatorname* { m a x } ( \hat { C } ( X _ { i } ) ) - Y _ { i } ( 0 ) \right] .
$$

Eq. (10) and (11) are useful, but they depend on knowing the value of $A _ { i }$ , which is not available for future data.

In stage two, we apply a secondary procedure to the new training data pairs $( X _ { i } , \hat { C } _ { i } )$ to eliminate the dependency on the treatment assignment $A$ . Algorithm 2 below sketches the Exact method, where we apply another split Conformal on $( X _ { i } , \hat { C } _ { i } )$ . Here we denote $\bar { \hat { C } _ { i } } \overset { \cdot } { = } \left( \hat { C } _ { i } ^ { L } , \hat { C } _ { i } ^ { U } \right)$ for simplicity.

Lemma 1 (Lei and Cande\`s (2021)). Assume $( X _ { i } , \hat { C } _ { i } )$ are i.i.d. from $( X , C )$ . Then, for a test point $X _ { n + 1 }$ under Algorithm $\jmath$ and 2, both with miscoverage level $\alpha / 2$ ,

$$
\begin{array} { r } { \mathbb { P } ( Y _ { n + 1 } ( 1 ) - Y _ { n + 1 } ( 0 ) \in \hat { C } _ { I T E } ( X _ { n + 1 } ) \geq 1 - \alpha . } \end{array}
$$

Lemma 1 can be directly proved using Theorem 2 above and Theorem 2 in (Lei and Cande\`s 2021). Lemma 1 shows that by using an exact method, we can construct a prediction interval that covers the true ITE with a desired coverage level as in (1).

Input: Level $\gamma$ , data $( X _ { i } , \hat { C } _ { i } ) , i \ \in \ \mathcal { T } _ { 2 }$ , where $\hat { C } _ { i }$ are obtained from Algorithm 1 using control and treatment   
group respectively, as shown in (10) and (11), and a test point $X _ { n + 1 }$   
Output: A prediction interval $\hat { C } _ { \mathrm { I T E } } ( X _ { n + 1 } )$   
1: Split $\boldsymbol { \mathcal { T } }$ into two equal sized subsets $\mathcal { T } _ { t r }$ and $\scriptstyle { \mathcal { T } } _ { c a }$   
2: On $\scriptstyle { \mathcal { T } } _ { t r }$ , fit conditional mean $\hat { \tau } _ { L }$ and $\hat { \tau } _ { U }$ of $\hat { C } _ { i } ^ { L }$ and $\hat { C } _ { i } ^ { U }$ given $X$   
3: For each $i \in \mathcal { T } _ { c a }$ , compute score $V _ { i } = \operatorname* { m a x } \{ \hat { \tau } _ { L } ( X _ { i } ) -$ $\hat { C } _ { i } ^ { L } , \hat { C } _ { i } ^ { U } - \hat { \tau } _ { U } ( X _ { i } ) \}$ .   
4: Compute $\eta$ as the $1 - \gamma$ quantile over scores $V _ { i }$ .   
5: return $\begin{array} { r } { \hat { C } _ { \mathrm { I T E } } ( X _ { n + 1 } ) = \left[ \hat { \tau } _ { L } ( X _ { n + 1 } ) - \eta , \hat { \tau } _ { U } ( X _ { n + 1 } ) + \eta \right] } \end{array}$

Another method with theoretical guarantees is the Naive method, which employs a straightforward approach using a naive Bonferroni correction, designed as follows:

$$
\begin{array} { r } { \hat { C } _ { \Pi \mathbb { E } } ( x ) = \big [ \operatorname* { m i n } ( \hat { C } ^ { 1 } ( x ) ) - \operatorname* { m a x } ( \hat { C } ^ { 0 } ( x ) ) , } \\ { \operatorname* { m a x } ( \hat { C } ^ { 1 } ( x ) ) - \operatorname* { m i n } ( \hat { C } ^ { 0 } ( x ) ) \big ] } \end{array}
$$

Here ${ \hat { C } } ^ { 1 }$ and $\hat { C } ^ { 0 }$ denote the prediction intervals computed for the treatment group and the control group, respectively. By setting the miscoverage level in Algorithm 1 to be $1 - \alpha / 2$ , the Naive method achieves the desired coverage probability of $1 - \alpha$ for the ITE, as specified in (1).

In practice, while the Exact and Naive methods tend to be overly conservative, utilizing the conditional density estimate as the score function in Algorithm 1 helps to mitigate this issue, as demonstrated in our empirical results in Section 4.2. A practical and favorable alternative is the Inexact method, which yields much shorter prediction intervals, albeit without theoretical guarantees. To implement the Inexact method, we fit plug-in estimates for the $40 \%$ conditional quantiles of $\hat { C } _ { i } ^ { L }$ and $60 \%$ conditional quantiles of $\hat { C } _ { i } ^ { U }$ , respectively. For a new test point, these quantiles are then used to straightforwardly compute the prediction interval.

Inspired by X-learner (Ku¨nzel et al. 2019), we propose a fourth, less conservative alternative method named $C D$ - $X$ . We fit four plug-in estimates for the conditional means of $\hat { C } _ { i } ^ { L }$ and $\hat { C } _ { i } ^ { U }$ for both $A _ { i } = 0$ and $A _ { i } = 1$ , denoted as $\tilde { C } _ { 0 } ^ { L } ( \dot { x } ) , \tilde { C } _ { 0 } ^ { U } ( \dot { x } ) , \tilde { C } _ { 1 } ^ { L } ( x ) , \tilde { C } _ { 1 } ^ { U } ( x )$ . In Algorithm 1, we also estimate the propensity scores ${ \hat { \pi } } ( x )$ using $\mathcal { T } _ { 1 }$ . Then, for a new test individual, the prediction interval can be computed using the formula

$$
\begin{array} { r } { \hat { C } _ { \Pi \mathrm { E } } ( x ) = \left[ \hat { \pi } ( x ) \tilde { C } _ { 1 } ^ { L } ( x ) + ( 1 - \hat { \pi } ( x ) ) \tilde { C } _ { 0 } ^ { L } ( x ) , \right. } \\ { \left. \hat { \pi } ( x ) \tilde { C } _ { 1 } ^ { U } ( x ) + ( 1 - \hat { \pi } ( x ) ) \tilde { C } _ { 0 } ^ { U } ( x ) \right] } \end{array}
$$

Although CD-X does not offer theoretical guarantees, it performs well in most cases within our numerical experiments, achieving the desired coverage meanwhile producing the shortest prediction intervals. In the next section, we will examine the empirical performance of methods ensemble with CD (Algorithm 1) and WCP.

Homosc. + has effect Heterosc. + has effect Homosc. + no effect Heterosc. + no effect CM_DR 1 A 1 I   
WCP_Inexact 中 中 + 世 WCP_Exact 1 1 1 1 WCP_Naive 一 1 一 1 CD−X 中 ·iT E · F CD−Inexact 南 中 中 中 CD−Exact 1 ！ 一 1 CD−Naive 1 1 一 1 0.00 0.25 0.50 0.75 1.000.00 0.25 0.50 0.75 1.000.00 0.25 0.50 0.75 1.000.00 0.25 0.50 0.75 1.00 Empirical Coverage CM_DR 一 中   
WCP_Inexact 中 !· · 中 :l. WCP_Exact 正 由 WCP_Naive 中 -EI 中 白: CD−X 中 中 中 ！ CD−Inexact F ! 中 ! CD−Exact 中 P 中 ! CD−Naive |日! 日 !： 0 5 10 15 0 5 10 15 0 5 10 15 0 5 10 15 Interval Length

# 4 Experimental Studies

# 4.1 Experimental Setup

The nature of potential outcomes limits our observations to factual outcomes, excluding counterfactuals. This characteristic necessitates the validation of ITE estimation primarily through simulations and semi-synthetic data. In this section, we will explore the performance of our methods across one simulation featuring four different settings, and three semisynthetic benchmarks.

We consider the following baselines known to produce valid prediction intervals for ITEs:

• WCP-Inexact, WCP-Exact, and WCP-Naive: We consider state-of-the-art (SOTA) methods based on WCP, each integrating the Inexact, Exact, and Naive approaches. • CM-DR: We also evaluate the Conformal Meta-learners (Alaa, Ahmad, and van der Laan 2023) with the doublyrobust learner. This method is one of the top performers among three methods proposed in (Alaa, Ahmad, and van der Laan 2023) and also offers theoretical guarantees. Throughout our experimental studies, we estimate the propensity score in CM-DR rather than assuming it is known, as done in the original study by (Alaa, Ahmad, and van der Laan 2023), to ensure a fair comparison.

# 4.2 Simulation Studies

In the simulation studies, we combine the data-generation processes described in Lei and Cande\`s (2021) and Alaa, Ahmad, and van der Laan (2023), which were originally proposed by Wager and Athey (2018). The data are generated as follows: Covariates $X$ are sampled from $\mathrm { U n i } \mathbf { \tilde { f } } ( [ 0 , 1 ] ^ { d } )$ . The propensity score is generated based on $\pi ( x ) \stackrel { - } { = } \frac { 1 } { 4 } \left[ 1 + \beta _ { 2 , 4 } ( x _ { 1 } ) \right]$ , and the treatment assignment $A | X$ is generated from $\operatorname { B e r n } ( \pi ( x ) )$ . The potential outcomes, $Y ( j )$ for $j ~ = ~ 0 , 1$ , are modeled based on the function $g ( x ) = 2 / \left\{ 1 + \exp \left[ - 1 2 ( x - 0 . 5 ) \right] \right\}$ , where $\mathbb { E } [ Y ( 1 ) | X ] =$ $g ( x _ { 1 } ) g ( x _ { 2 } )$ and $\mathbb { E } [ Y ( 0 ) | X ] = \gamma g ( x _ { 1 } ) g ( x _ { 2 } )$ . Here $\gamma$ controls the treatment effect. These outcomes are then generated from the model $\mathbb { E } [ Y ( j ) | X ] + \sigma ( X ) \epsilon$ , where $\epsilon \sim \bar { N } ( 0 , 1 )$ . We consider four scenarios, derived from a $2 \mathrm { ~ x ~ } 2$ factorial design: homoscedastic $( \sigma ( x ) = 1 )$ and heteroscedastic $( \sigma ( x ) = - \log ( x _ { 1 } ) )$ errors, and treatment has no effect $\langle \gamma = 1 \rangle$ ) and the effects are heterogeneous $( \gamma = 0 )$ ).

# 4.3 Semi-synthetic Datasets

We also explore the performance of our approaches on three semi-synthetic datasets, which feature real covariates combined with simulated outcomes. These datasets include the National Study of Learning Mindsets (NSLM) (Yeager et al. 2019), the 2016 Atlantic Causal Inference Conference Competition (ACIC) (Dorie et al. 2019), and the Infant Health and Development Program (IHDP) datasets (Hill 2011). Detailed descriptions of all datasets are available in the supplementary material.

Table 1: Performance of all methods on semi-synthetic datasets. Empirical coverage (in percentages) and average interval lengths are shown, with standard errors in parentheses. Bold numbers highlight the best performance. “With guarantee” indicates methods that provide theoretical coverage guarantees.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="2">NSLM</td><td colspan="2">ACIC</td><td colspan="2">IHDP</td><td rowspan="2">With guarantee</td></tr><tr><td>Coverage</td><td>Avg. len.</td><td>Coverage</td><td>Avg. len.</td><td>Coverage</td><td>Avg.len.</td></tr><tr><td>CM-DR</td><td>99.9 (0.00)</td><td>6.45 (0.14)</td><td>99.9 (0.02)</td><td>52.9 (1.80)</td><td>99.9 (0.03)</td><td>82.1 (5.31)</td><td></td></tr><tr><td>WCP-Inexact</td><td>93.6 (0.29)</td><td>2.20 (0.03)</td><td>96.1 (0.48)</td><td>16.6 (0.40)</td><td>83.2 (0.87)</td><td>8.86 (0.72)</td><td>×</td></tr><tr><td>WCP-Exact</td><td>99.9 (0.01)</td><td>4.48 (0.04)</td><td>99.8 (0.04)</td><td>40.7 (2.11)</td><td>99.5 (0.10)</td><td>34.9 (4.60)</td><td></td></tr><tr><td>WCP-naive</td><td>99.9 (0.01)</td><td>4.23 (0.03)</td><td>99.8 (0.04)</td><td>30.7 (1.38)</td><td>99.3 (0.13)</td><td>21.4 (2.09)</td><td></td></tr><tr><td>CD-X</td><td>86.3 (0.37)</td><td>1.78 (0.02)</td><td>93.9 (0.42)</td><td>12.2 (0.30)</td><td>79.9 (0.83)</td><td>6.40 (0.51)</td><td>×</td></tr><tr><td>CD-Inexact</td><td>92.0 (0.29)</td><td>2.09 (0.02)</td><td>94.7 (0.49)</td><td>14.4 (0.37)</td><td>80.3 (0.93)</td><td>8.31 (0.69)</td><td>×</td></tr><tr><td>CD-Exact</td><td>99.9 (0.01)</td><td>4.13 (0.03)</td><td>99.5 (0.07)</td><td>30.4 (0.63)</td><td>99.4 (0.10)</td><td>25.6 (2.95)</td><td></td></tr><tr><td>CD-Naive</td><td>99.8 (0.02)</td><td>4.02 (0.04)</td><td>99.2 (0.13)</td><td>29.3 (0.67)</td><td>96.6 (0.39)</td><td>20.2 (2.04)</td><td></td></tr></table></body></html>

# 4.4 Results and Discussion

As our focus is on the prediction interval for ITEs, we use two commonly used metrics across all experimental studies: empirical coverage for true ITEs and average prediction interval length. The empirical coverage is defined as the empirical probability that the true ITE falls within the predicted interval. The average prediction interval length is measured as the mean of the widths of these intervals across all test instances.

Figure 1 illustrates the outcomes of the simulation studies, averaging results over 100 replications for each scenario as described in Section 4.2. CD-X generally excels, achieving the shortest and near-optimal interval lengths while maintaining the desired coverage levels, except in the scenario with heteroscedastic errors and treatment effect where it undercovers by 0.01. When comparing methods incorporating CD with those using WCP, CD methods consistently provide shorter intervals. Specifically, CD-Inexact is on average 0.26 shorter than WCP-Inexact, CD-Exact is 1.43 shorter than WCP-Exact. CD-Naive and WCP-Naive achieve nearly the same interval lengths on average.

The semi-synthetic experiments further demonstrate the superiority of the proposed CD methods. Across all three benchmarks, CD methods consistently outperform others. Although CD-X and CD-Inexact lack coverage guarantees, they excel in the ACIC and NSLM datasets, respectively, with the shortest average length and the coverage guarantee. For each pair of matched CD and WCP methods, CD consistently provides shorter intervals in all cases. Regarding the IHDP results, Alaa and Van Der Schaar (2017) reported that CM-DR performed well with an average interval length of 16.7, assuming a known propensity score. We emphasize that in our studies, we estimate the propensity score for CMDR to ensure a fair comparison. This is particularly critical given the imbalanced setting of the IHDP dataset, which includes only 747 samples (139 treated and 608 control), making accurate propensity score estimation challenging. In this context, CD-Naive outperforms all other methods.

Although our primary goal is to mitigate the excessive conservatism of prediction intervals, the inherent challenges posed by unobserved counterfactuals naturally lead to conservative outcomes. In our experimental studies, the CDExact method consistently demonstrated coverage rates exceeding $9 9 \%$ , despite a targeted desired coverage of $90 \%$ . Our two-stage framework for predictive inference of ITE addresses the reduction of prediction interval length for counterfactuals in Stage 1 by efficiently using conditional density as the score function. However, further reducing conservatism in Stage 2 remains an area for future research, aiming to find an optimal balance between the less conservative Inexact methods and the overly conservative Exact methods.

# 5 Conclusions

In this paper, we developed a two-stage framework to provide interval estimates for Individual Treatment Effects (ITEs), a task inherently challenging due to the nature of unobservable potential outcomes. We successfully leveraged the reference distribution technique to efficiently estimate the optimal conformal scores—the conditional densities. Both theoretical and experimental results demonstrate that our framework outperforms existing state-of-theart Weighted Conformal Prediction (WCP) methods. The practical implications of our work can be substantial, particularly given the difficulty inherent in estimating ITEs. Our method’s success in reducing the average length of prediction intervals enhances their usability in real-world scenarios, particularly in decision-making processes requiring precise estimates, such as in clinical and policy-making fields.