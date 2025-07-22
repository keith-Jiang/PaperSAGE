# General Uncertainty Estimation with Delta Variances

Simon Schmitt, 1,2 John Shawe-Taylor, 2 Hado van Hasselt1

1 DeepMind 2 University College London, UK suschmitt@google.com

# Abstract

Decision makers may suffer from uncertainty induced by limited data. This may be mitigated by accounting for epistemic uncertainty, which is however challenging to estimate efficiently for large neural networks.

To this extent we investigate Delta Variances, a family of algorithms for epistemic uncertainty quantification, that is computationally efficient and convenient to implement. It can be applied to neural networks and more general functions composed of neural networks. As an example we consider a weather simulator with a neural-network-based step function inside – here Delta Variances empirically obtain competitive results at the cost of a single gradient computation.

The approach is convenient as it requires no changes to the neural network architecture or training procedure. We discuss multiple ways to derive Delta Variances theoretically noting that special cases recover popular techniques and present a unified perspective on multiple related methods. Finally we observe that this general perspective gives rise to a natural extension and empirically show its benefit.

# 1 Introduction

Decision makers often need to act given limited data. Accounting for the resulting uncertainty (epistemic uncertainty) may be helpful for active learning (MacKay 1992a), exploration (Duff 2002; Auer, Cesa-Bianchi, and Fischer 2002) and safety (Heger 1994).

How to measure epistemic uncertainty efficiently for large neural networks is active research. Computational efficiency is important because even a single evaluation (e.g. a forward pass through a neural network) can be expensive. Popular approaches compute an ensemble of predictions using bootstrapping or MC dropout and incur a multiplicative computational overhead. Other approaches are faster but require changes to the predictors architecture and training procedure (Van Amersfoort et al. 2020).

In this paper we propose the Delta Variance family of algorithms which connects and extends Bayesian, frequentist and heuristic notions of variance. Delta Variances require no changes to the architecture or training procedure while incurring the cost of little more than a gradient computation. We present further appealing properties and benefits in Table 1.

The approach can be applied to neural networks or functions that contain neural networks as building blocks to compute a quantity of interest. For instance we could learn a step-by-step dynamics model and then use it to infer some utility function for decision making.

![](images/afa8ffa8902e3a03555e69e395d68213973920d37420bf5c455861d6ebcfeba8.jpg)  
Figure 1: We compare the computational overhead of training and evaluating different variance estimators. Delta Variances are favourable in terms of computational efficiency. They incur negligible training overhead while inference incurs the cost of a regular gradient pass making them more efficient than the alternatives considered. Monte-Carlo Dropout also incurs negligible training overhead, but requires $K$ independent evaluations for inference. Most expensive are Bootstrapped Ensembles requiring $K \times$ repeated computations.

In Section 6 we consider the GraphCast weather forecasting system with a neural network step function (Lam et al. 2023). We then compute the epistemic uncertainty of various derived quantities such as the expected precipitation or wind-turbine-power at a particular location.

Section 5 observes how instances of the Delta Variance family can be derived using different assumptions and theoretical frameworks. We begin with a Bernstein-von Mises plus Delta Method derivation, which relies on strong assumptions. We conclude with an influence function based derivation, which relies only on mild assumptions. Interestingly the resulting instances are not only similar, but also become identical as the number of observed data-points grows.

Formalizing the Delta Variance family allows us to connect Bayesian and frequentist notions of variance, adversarial robustness and anomaly detection in a unified perspective.

Table 1: Delta Variances have appealing benefits: The prototypical variant with diagonal $\Sigma$ is computationally efficient requiring only a gradient pass for inference while other methods evaluate the neural network multiple times. Furthermore they easily build on the existing training procedure and can even be added post hoc after training. Delta Variances do not require architecture changes such as introducing dropout layers or training procedure changes as needed for ensembling or even hyper parameter search. Finally Delta Variances have a simple closed form expression that yields reproducible deterministic results.   

<html><body><table><tr><td></td><td>Delta Variances</td><td>Ensemble</td><td>MC-Dropout</td></tr><tr><td>Efficiency</td><td></td><td></td><td></td></tr><tr><td>Inference cost</td><td>1 × gradient</td><td>K× evaluations</td><td>K× evaluations</td></tr><tr><td>Training overhead</td><td>1×</td><td>K×</td><td>1×</td></tr><tr><td>Memoryoverhead</td><td>2×</td><td>K×</td><td>1×</td></tr><tr><td>Ease of Use</td><td></td><td></td><td></td></tr><tr><td>No architecture requirements</td><td>√</td><td>√</td><td></td></tr><tr><td>No change to training procedure</td><td>√</td><td></td><td></td></tr><tr><td>Deterministic result</td><td>√</td><td>√</td><td></td></tr></table></body></html>

This perspective can be used to answer questions such as: What happens if we use a Bayesian variance, but our neural network does not meet all theoretical assumptions – what is a theoretically sound interpretation of the number that we compute? To further highlight the generality of this unified perspective we propose a novel Delta Variance in Section 7.1 for which we observe empirically improvements in Section 6.

When applied to the state-of-the-art GraphCast weather forecasting system we observe favourable results. In comparison to popular related approaches such as ensemble methods our method exhibits similar quality while requiring fewer computational resources.

What is Epistemic Variance? Given limited data the parameters $\theta$ of a parametric model $f _ { \theta }$ can only be identified with limited certainty. The resulting parameter uncertainty translates into uncertainty in the outputs of $f _ { \theta } ( x )$ and any other function $u _ { \theta } ( z )$ that depends on $\theta$ . We define Epistemic Variance as the output variance induced by the posterior distribution over parameters given the model $f$ and its training data $p ( \theta | f , \mathcal { D } )$ :

$$
\mathbb { V } _ { \theta \sim p ( \theta | f , \mathcal { D } ) } \left[ f _ { \theta } ( x ) \right]
$$

Section 3 extends this definition to any function $u _ { \theta }$ that depends on parameters $\theta$ that were estimated using $f _ { \theta }$ and $\mathcal { D }$ . As an illustration consider Section 6 where $f _ { \theta }$ is a learned weather dynamics model and $u _ { \theta }$ are weather dependent utility functions - e.g. a wind turbine power yield forecast. The epistemic variance of $u _ { \theta } ( z )$ is then $\mathbb { V } _ { \theta \sim p ( \theta | f , D ) } \left[ u _ { \theta } ( z ) \right) ]$ .

What is a Delta Variance Estimator? Variance estimators of the Delta Variance family are all of the following parametric form:

$$
\Delta _ { f ( x ) } ^ { \top } \Sigma \Delta _ { f ( x ) }
$$

being a vector-matrix-vector product using the gradient vector of $f \colon \Delta _ { f ( x ) } : = \nabla _ { \theta } f _ { \theta } ( x )$ while leaving some flexibility in the choice of matrix $\Sigma$ . In Section 5 and Table 2 we discuss dbiefifnergetnhtecihnovicereseofF $\Sigma$ haenrdinthfoeirrmpartoipoenrtimeast.riFxord $\begin{array} { r } { \Sigma _ { f } = \frac { 1 } { N } F _ { f } ^ { - 1 } } \end{array}$ number of data-points $N$ , it can be shown under suitable conditions that

$$
\begin{array} { r } { \mathbb { V } _ { \theta \sim p ( \theta | f , \mathcal { D } ) } \left[ f _ { \theta } ( x ) \right] \approx \Delta _ { f ( x ) } ^ { \top } \Sigma _ { f } \Delta _ { f ( x ) } } \end{array}
$$

Its name is inspired by the closely related Delta Method (Lambert 1765; Gauss 1823; Doob 1935) – see Gorroochurn (2020) for a historic account – which provides one of many ways to derive Delta Variances.

What is a Quantity of Interest? Sometimes we use a neural network $f _ { \theta }$ to learn predictions, that are then used to compute a downstream quantity of interest. For instance we could learn a step-by-step weather dynamics model $f _ { \theta } ( x )$ and then use it to infer some utility function for decision making $u _ { \theta } ( z )$ . Given limited training data neither the neural network’s prediction nor the downstream quantity of interest will be exact. The definition of Epistemic Variance and the Delta Variance family both extend conveniently to quantities of interest $u _ { \theta }$ . In fact we only need to replace $\Delta _ { f ( x ) }$ by $\Delta _ { u ( z ) } : = \nabla _ { \theta } u _ { \theta } ( z )$ :

$$
\mathbb { V } _ { \theta \sim p ( \theta | f , \mathcal { D } ) } \left[ u _ { \theta } ( z ) \right] \approx \Delta _ { u ( z ) } ^ { \top } \Sigma _ { f } \Delta _ { u ( z ) }
$$

Conveniently we can still use the same $\Sigma _ { f }$ as before. It is independent of $u _ { \theta }$ and can be re-used for various quantities of interest.

# 2 Notation

We consider a function $f _ { \theta } ( x )$ with parameters $\theta$ trained on a dataset $\mathcal { D }$ of size $N$ . We strive to estimate the uncertainty introduced by training on limited data. To admit Bayesian interpretations we assume that $f _ { \theta } ( x )$ is a density model that is trained with log-likelihood (or a function that is trained with a log-likelihood-equivalent loss, such as $L _ { 2 }$ regression or cross-entropy). Unless otherwise specified we assume that $\theta$ has been trained until convergence – i.e. equals the (local) maximum likelihood estimate $\bar { \theta }$ . Under appropriate conditions $\bar { \theta }$ converges to the true distribution parameters $\theta _ { \mathrm { T r u e } }$ . Let $H _ { f }$ be the Hessian of the log-likelihood of all $\mathcal { D }$ evaluated at $\bar { \theta }$ . When adopting a Bayesian view with prior belief $p ( \theta )$ the posterior over parameters is defined as $p ( \theta | \mathcal { D } ) \propto p ( \mathcal { D } | \theta ) p ( \theta ) . \mathbb { E } _ { z \sim p ( z ) }$ refers to the expectation with respect to random variable $z$ with distribution $p ( z )$ – which we shorten to $\mathbb { E } _ { z }$ or $\mathbb { E }$ when the distribution is clear. Similarly let $\mathbb { V } \left[ X \right] : = \mathbb { E } \left[ X ^ { 2 } \right] - \mathbb { E } \left[ X \right] ^ { 2 }$ .

Let $\begin{array} { r l r } { F _ { f } } & { : = } & { \mathbb { E } _ { x \sim f _ { \theta _ { \mathrm { T r u e } } } } \nabla _ { \theta } \log f _ { \theta } ( x ) \nabla _ { \theta } \log f _ { \theta } ( x ) ^ { \top } \vert _ { \theta = \theta _ { \mathrm { T r u e } } } } \end{array}$ be the Fisher information matrix and let $\begin{array} { r l } { \hat { F } _ { f } } & { { } : = } \end{array}$ $\begin{array} { r } { \frac { 1 } { N } \sum _ { x _ { i } \in \mathcal { D } } \nabla _ { \theta } \log f _ { \theta } ( x _ { i } ) ^ { \top } \ \nabla _ { \theta } \log f _ { \theta } ( x _ { i } ) | _ { \theta = \bar { \theta } } } \end{array}$ be the empirical Fisher information. Note that $H _ { f } ^ { - 1 }$ and $\hat { F } _ { f } ^ { - 1 } / N$ are both $O \left( N ^ { - 1 } \right)$ . When strong conditions are met (see van der Vaart 1998, for details) the Bernstein-von Mises theorem ensures that the Bayesian posterior converges to the Gaussian distribution $\mathcal { N } ( \check { \theta } , \frac { 1 } { N } F _ { f } ^ { - 1 } )$ in total variation norm independently of the choice of prior as the number of data-points $N$ increases. In Definition 1 we consider Quantities of Interest that we denote $u _ { \theta }$ . In practice $u _ { \theta } ( z )$ may be a utility function that depends on some context provided by $z$ . For a simpler exposition but without loss of generality we assume that $u _ { \theta } ( z )$ is scalar valued. We require $f _ { \theta }$ and $u _ { \theta }$ to have bounded second derivatives wrt. $\theta$ in order to perform first order Taylor expansions. To simplify notations we assume that $\theta$ is evaluated at the learned parameters $\bar { \theta }$ unless specified otherwise: in particular we write $\nabla _ { \boldsymbol { \theta } } f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ in place of $\nabla _ { \theta } f _ { \theta } ( x ) | _ { \theta = \bar { \theta } }$ and $\Delta _ { f ( x ) } : = \nabla _ { \theta } f _ { \theta } ( x ) | _ { \theta = \bar { \theta } }$ .

# 3 Epistemic Variance of Quantities of Interest

Sometimes we use a neural network $f _ { \theta }$ to learn predictions, that are then used to compute a downstream quantity of interest $u _ { \theta } - \sec$ motivational examples below. Given limited training data neither the neural network’s prediction nor the downstream quantity of interest will be exact. This motivates our research question:

# If we estimate the parameters $\theta$ of $\mathbf { u } _ { \theta }$ by learning $\mathbf { f } _ { \boldsymbol { \theta } } ( \mathbf { x } )$ , how can we quantify the epistemic uncertainty of ${ \bf u } _ { \boldsymbol { \theta } } ( { \bf z } ) \mathbf { ? }$

For a simpler exposition and without loss of generality we assume that $u _ { \theta } ( z )$ predicts scalar quantities. The prototypical example is a utility function that depends on some context provided by $z$ and internally uses $f _ { \theta }$ to compute a utility value. The derivations carry over naturally to the multi-variate case. Note that $f _ { \theta }$ and $u _ { \theta }$ may have different input spaces. Our research focuses on the general case where $f _ { \theta } \neq u _ { \theta }$ which has received little attention. This naturally includes the case where $f _ { \theta } = u _ { \theta }$ .

Definition 1. We call the real-valued function $u _ { \theta } ( z )$ quantity of interest if it depends on the same parameters $\theta$ as a related parametric model $f _ { \theta } ( x )$ .

# 3.1 Motivational Examples

We consider three motivational examples for training on $f _ { \theta }$ but evaluating a different quantity of interest $u _ { \theta }$ . We will see that training $f _ { \theta }$ is straightforward while training a predictor for $u _ { \theta } ( z )$ is inefficient, impractical, or even impossible.

1. As a simple motivation let us consider estimating the 10-year survival chance using a neural network predictor $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ of 1-year outcomes given patient features $x$ :

$$
u _ { \theta } ( x ) = f _ { \theta } ( x ) ^ { 1 0 }
$$

This example illustrates that it may be impossible to train $\mathbf { u } _ { \theta } ( \mathbf { x } )$ directly unless we collect data for 9 more years, hence we train $f _ { \theta } ( x )$ and evaluate $u _ { \theta } ( z )$ .

2. Distinct input spaces: $u _ { \theta } ( z )$ might aggregate predictions of $f _ { \theta } ( x )$ for sets $z ~ = ~ \{ x _ { 1 } , \ldots , x _ { k } \}$ : E.g. the survival chance of everyone in set of patients $z$ via $u _ { \theta } ( z ) : =$ $\textstyle \prod _ { x _ { i } \in z } f _ { \theta } ( x _ { i } )$ , or the average value of some basket of items $z$ , or the chance of any advertisement from a presented set being clicked. Here training $f _ { \theta }$ may be more convenient than training $u _ { \theta }$ .

3. Multiple derived quantities: In Section 6 we compute multiple quantities of interest using the GraphCast weather forecasting system (Lam et al. 2023). Training a separate $u _ { \theta }$ for each of them would be cumbersome and expensive.

# 3.2 Epistemic Variance

Here we define Epistemic Variance first from a Bayesian and then from a frequentist perspective. This allows us to formalize and quantify how parameter uncertainty from training $f _ { \theta }$ translates to uncertainty of any quantity of interest $u _ { \theta } ( z )$ that also depends on $\theta$ .

Bayesian Definition Epistemic uncertainty can be formalized with a Bayesian posterior distribution over parameters given training data: $p ( \boldsymbol { \theta } | \mathcal { D } )$ . The Epistemic Variance of a function evaluation $u _ { \theta } ( z )$ is then defined to be the variance induced by the posterior over $\theta$ :

Definition 2. Given any function $u _ { \theta }$ and a posterior over parameters $p ( \theta | f , \mathcal { D } )$ resulting from training $f _ { \theta }$ on data $\mathcal { D }$ the Epistemic Variance of $u _ { \theta } ( z )$ is defined as

$$
\mathbb { V } _ { \theta \sim p ( \theta | f , \mathcal { D } ) } \left[ u _ { \theta } ( z ) \right]
$$

where $\mathbb { V } \left[ X \right] : = \mathbb { E } \left[ X ^ { 2 } \right] - \mathbb { E } \left[ X \right] ^ { 2 }$ .

Frequentist Definition Leave-one-out cross-validation (Quenouille 1949) is a frequentist counterpart to Epistemic Variance. It computes the variance of $u _ { \theta } ( z )$ induced by removing a random element from the training data and reestimating the parameters $\theta$ .

Definition 3. Let $\theta _ { \backslash i }$ be the leave-one-out parameters resulting from training $f _ { \theta }$ on data $\mathcal { D } \backslash \{ x _ { i } \}$ , then the Leave-one-out Variance is defined as

$$
\mathbb { V } _ { \theta \sim L O O } \left[ u _ { \theta } ( z ) \right] : = \mathbb { V } _ { i \sim U ( 1 , \dots , N ) } \left[ u _ { \theta _ { \setminus i } } ( z ) \right]
$$

where $U ( 1 , \ldots , N )$ is the uniform distribution over indices.

# 4 Delta Variance Approximators

Delta Variance estimators are a family of efficient and convenient approximators of epistemic uncertainty. They can be used to compute the Epistemic Variance of a quantity of interest $u _ { \theta } ( z )$ where the parameters $\theta$ are obtained by learning $f _ { \theta }$ with limited data. Given any quantity of interest $u _ { \theta } ( z )$ they approximate both the Bayesian Epistemic Variance as well as the frequentist leave-one-out analogue:

$$
\underbrace { \mathbb { V } _ { \theta \sim p ( \theta | f , \mathcal { D } ) } \left[ u _ { \theta } ( z ) \right] } _ { \mathrm { E p i s t e m i c ~ V a r i a n c e } } \approx \underbrace { \Delta _ { u ( z ) } ^ { \top } \Sigma \Delta _ { u ( z ) } } _ { \mathrm { D e l t a ~ V a r i a n c e } } \approx \underbrace { \mathbb { V } _ { \theta \sim L O O } \left[ u _ { \theta } ( z ) \right] } _ { \mathrm { L O O ~ V a r i a n c e } }
$$

Here the Delta $\Delta _ { u ( z ) } : = \nabla _ { \theta } u _ { \theta } ( z )$ is the gradient vector of $u _ { \theta }$ evaluated at the input $z . \Sigma$ is a suitable matrix for which the canonical choice is an approximation of the scaled inverse Fisher Information matrix of $f _ { \theta }$ .

Canonical Choice of $\pmb { \Sigma }$ The family of Delta Variances in principle supports any positive definitive matrix $\Sigma$ . We will see in Section 5.1 that it intuitively represents the posterior covariance of the parameters $\theta$ after learning $f _ { \theta }$ on the training data $\mathcal { D }$ . The canonical choice is $\begin{array} { r } { \Sigma : = \frac { 1 } { N } \hat { F } _ { f } ^ { - 1 } } \end{array}$ being the inverse empirical Fisher Information matrix scaled by the number of data-points $N$ . Plugged into the Delta Variance formula we obtain the following estimate for the Epistemic Variance of $u _ { \theta } ( z )$ :

$$
\mathbb { V } _ { \theta \sim p ( \theta | f , \mathcal { D } ) } \left[ u _ { \theta } ( z ) \right] \approx \frac { 1 } { N } \nabla _ { \theta } u _ { \theta } ( z ) ^ { \top } \hat { F } _ { f } ^ { - 1 } \nabla _ { \theta } u _ { \theta } ( z )
$$

It is worth emphasizing that the Fisher information is computed using $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ (the model that was used for training $\theta$ ) while the gradient delta vectors come from $u _ { \theta } ( z )$ the quantity of interest that is evaluated. Hence $\hat { F } _ { f } ^ { - 1 }$ can be precomputed and reused for various choices of $u _ { \theta }$ .

Intuition Section 5 explores multiple ways to theoretically justify the Delta Variance family. The Bayesian intuition is that $\Sigma$ captures the posterior covariance of the parameters $\theta$ while $\Delta _ { u ( z ) } ^ { \phantom { \dagger } } = \nabla _ { \theta } \bar { u _ { \theta } } ( z )$ translates this parameter uncertainty from variations in $\theta$ to variations in $u _ { \theta } ( z )$ . In Figure 2 we consider an illustrative example, where a survival rate of $f _ { \boldsymbol { \theta } } ( x ) = \boldsymbol { \theta }$ has been estimated and is used to make predictions 10 years ahead via $u _ { \theta } ( z ) = \theta ^ { 1 0 }$ .

Theoretical Motivation The family of Delta Variance estimators is motivated because under strong conditions (see Section 5.1) and for number of data-points $N$ it can be shown to recover the Epistemic Variance up to a diminishing error:

$$
\underbrace { \mathbb { V } _ { \theta \sim p \left( \theta \mid f , \mathcal { D } \right) } \left[ u _ { \theta } ( z ) \right] } _ { \mathrm { E p i s t e m i c ~ V a r i a n c e } } = \underbrace { \Delta _ { u ( z ) } ^ { \top } \Sigma \Delta _ { u ( z ) } } _ { \mathrm { D e l t a V a r i a n c e } } + O \left( N ^ { - 1 . 5 } \right)
$$

An additional motivation is that it can be derived using mild assumptions from a leave-one-out or an adversarial robustness perspective (see Sections 5.2 and 5.3).

Computational Convenience Delta Variances are convenient because $\Delta _ { u ( z ) } : = \nabla _ { \theta } u _ { \theta } ( z )$ can be computed using any auto-differentiation framework and because $\Sigma$ does not depend on $u _ { \theta }$ (e.g. can be re-used for many different quantities of interest $u _ { \theta }$ ). It is efficient because it is a vector-matrixvector product, where the matrix can be approximated efficiently (e.g. diagonally, low-rank, or using KFAC (Martens 2014)).

# 4.1 How to choose $\pmb { \Sigma }$

Principled choices of $\pmb { \Sigma }$ Theory suggests three principled choices for the covariance matrix, which all scale as Σ ∝ 1 . Each choice can be derived in at least two ways using statistics or using influence functions (see Section 5 for details and Table 2 for an overview).

1. The inverse Fisher Information $F _ { f } ^ { - 1 }$ divided by $N$ .   
2. The inverse Hessian of the training loss $H _ { f } ^ { - 1 }$ .   
3. The sandwich $H _ { f } ^ { - 1 } F _ { f } H _ { f } ^ { - 1 }$ times $N$ .

![](images/50466331f9d0cf8bf4c91c51cf92debff3b5b657002ef50bb8307dfffb5f1cd5.jpg)  
Figure 2: Illustrative survival prediction example. Actual epistemic variance (red) vs. predicted variance using the Delta Variance (orange) or a 10-fold Bootstrap (blue) as the dataset size $N$ grows. Shaded confidence areas contain $9 5 \%$ of the variance predictions. Bold lines are the median. Observe that the orange median line of the Delta Variance and the actual variance in red overlap largely. Top: variance of learned function $f _ { \boldsymbol { \theta } } ( x ) = \boldsymbol { \theta }$ Bottom: variance of quantity of interest $u _ { \theta } ( x ) : = \bar { \theta } ^ { \dot { 1 } 0 }$ evaluations. All methods yield reasonable results for $N > 1 0$ with ensemble methods exhibiting higher variance. Generally the variance for $u _ { \theta }$ is harder to estimate than for $f _ { \theta }$ .

For well-specified models the three covariance matrices become eventually equivalent as Hessian divided by $N$ and empirical Fisher converge to the true Fisher as data increases. In practice they need to be efficiently approximated from finite data (e.g. diagonally or using KFAC) and safely inverted. The first and third Bayesian approach use $F _ { f }$ which can be approximated by the empirical Fisher information $\hat { F } _ { f }$ and is easily invertible as it is non-negative by construction. Frequentist analogues use the empirical $\hat { F } _ { f }$ directly. In contrast $H _ { f }$ is only non-negative at a maximum which may not be reached precisely with stochastic optimization. Hence inverting $H _ { f }$ requires more careful regularization (Martens 2014). For simplicity we select $\Sigma$ to be a diagonal approximation of the empirical Fisher in our experiments, which alleviates the question of regularization.

Fine-tuning or Learning $\pmb { \Sigma }$ The analytic form of the Delta Variance permits to back-propagate into the values of $\Sigma$ . This enables approaches that learn better values for $\Sigma$ from scratch or improve the values via fine-tuning. We explore a simple example in Section 7.1 that improves empirically over the regular Fisher information by re-scaling some of its entries.

<html><body><table><tr><td>Bayesian Interpretations (Section 5.1)</td><td>Frequentist Interpretations</td><td>Choice of∑ (modulo factors of N)</td></tr><tr><td>Bernstein-von Mises Posterior4 Misspecified Bernstein-von Mises Posterior4 Laplace Posterior</td><td>OOD Detection1 (Sec. 5.4) Leave-one-out Variance² (Sec. 5.2) Adversarial Robustness² (Sec.5.3)</td><td>H-1FH-1 H-1 F-1</td></tr></table></body></html>

Table 2: For each choice of $\Sigma$ there exists a Bayesian and a frequentist interpretation. Each interpretation requires different assumptions on $f _ { \theta } ( x )$ and its loss. Due to their milder assumptions the frequentist interpretations can serve as fall-back interpretations if the stricter conditions on the Bayesian interpretations are not met. For example observe that assuming a Bernstein-von Mises Posterior is computationally equivalent to performing OOD Detection. Interestingly the former makes strong assumptions about $f _ { \theta }$ which typically do not apply to neural networks, while the later only requires differentiability of $f _ { \theta }$ and $u _ { \theta }$ and that $\theta$ converges locally. The Hessian is computed with respect to the training loss of $f _ { \theta }$ . The Bayesian interpretations start with the true Fisher matrix and approximate it from data e.g. with $\hat { F }$ . The frequentist approximations work with $\bar { \hat { F } }$ directly. In practice both $H$ and $\hat { F }$ are computed at the locally optimal parameters $\bar { \theta }$ . We consider $\Sigma$ modulo factors of $N$ as they do not change the interpretation.

# 5 Analysis

In this section we will investigate multiple ways to derive and motivate Delta Variances. Broadly speaking they can be separated into three classes:

1. In Section 5.1 we begin with the easiest derivations, which approximate the Bayesian posterior and make strong assumptions that may not always apply to neural networks.   
2. In Section 5.2 we consider the frequentist analogue of Epistemic Variance, that is compatible with neural networks and does not make assumptions about any posterior.   
3. In Section 5.3 and 5.4 we consider alternative derivations that are based on adversarial robustness and out-ofdistribution detection and rely on even fewer assumptions.

All of the considered derivations yield Delta Variances with principled covariance matrices. For an overview consider Table 5.1, where we can observe that assuming a Bernstein-von Mises Posterior is computationally equivalent to performing OOD Detection. Interestingly the former makes strong assumptions about $f _ { \theta }$ which typically do not apply to neural networks, while the later only requires that the covariance of gradients is finite and that $\theta$ converges locally. Due to their milder assumptions the frequentist interpretations can serve as fall-back interpretations if the stricter conditions on the Bayesian interpretations are not met.

# 5.1 Bayesian Interpretation

We begin with a derivation that gives rise to a bound on the approximation error. While requiring strong assumptions, it serves as a motivation and introduction. The error diminishes with the number of observed data-points $N$ :

$$
\underbrace { \mathbb { V } _ { \theta \sim p \left( \theta \mid f , \mathcal { D } \right) } \left[ u _ { \theta } ( z ) \right] } _ { \mathrm { E p i s t e m i c ~ V a r i a n c e } } = \underbrace { \Delta _ { u ( z ) } ^ { \top } \Sigma \Delta _ { u ( z ) } } _ { \mathrm { D e l t a V a r i a n c e } } + O \left( N ^ { - 1 . 5 } \right)
$$

Bernstein-von Mises Motivation As a motivational introduction we will derive the approximation error when the Bernstein-von Mises conditions are met (e.g. differentiability and unique optimum – see van der Vaart (1998) for details). Under such conditions the posterior converges to a Gaussian distribution centered around the maximum likelihood solution $\theta$ with a scaled inverse Fisher Information as covariance matrix.

$$
P ( \boldsymbol { \theta } | \mathcal { D } )  \mathcal { N } ( \boldsymbol { \theta } , \frac { 1 } { N } F _ { f } ^ { - 1 } )
$$

The Epistemic Variance can then be computed using the Delta Method resulting in Proposition 1.

Proposition 1. For a normally distributed posterior with mean $\bar { \theta }$ and a covariance matrix $\Sigma$ proportional to $\textstyle { \frac { 1 } { N } }$ it holds:

$$
\underbrace { \mathbb { V } _ { \theta \sim p ( \theta | f , \mathcal { D } ) } \left[ u _ { \theta } ( z ) \right] } _ { \substack { \theta \sim p ( \theta | f , \mathcal { D } ) } } = \underbrace { \Delta _ { u ( z ) } ^ { \top } \Sigma \Delta _ { u ( z ) } } _ { \substack { \theta \sim \bigcup } } + O \left( N ^ { - 1 . 5 } \right)
$$

where $\Delta _ { u ( z ) } : = \nabla _ { \theta } u _ { \theta } ( z ) | _ { \theta = \bar { \theta } }$ as usual.

Proof. See Schmitt, Shawe-Taylor, and van Hasselt (2025).

If the Bernstein-von Mises conditions are met Proposition 1 holds with $\Sigma = F _ { f } ^ { - 1 } / N$ .

Further Bayesian Interpretations Other Gaussian posterior approximations can be considered by plugging their respective posterior covariance matrix into Proposition 1: The misspecified Bernstein-von Mises theorem (see Kleijn and van der Vaart 2012) states that we obtain the sandwich covariance $H _ { f } ^ { - 1 } F _ { f } H _ { f } ^ { - 1 } \times N$ if the model $f _ { \theta }$ is misspecified (i.e. does not represent the data well). Proponents advocate that the sandwich estimate is more robust to heteroscedastic noise while others argue against it (Freedman 2006). Similarly a Laplace approximation (Laplace 1774; MacKay 1992b; Ritter, Botev, and Barber 2018) can be made resulting in a Delta Variance with Hf . Again those choices of Σ are ∝ N1 .

# 5.2 Frequentist Interpretation

To better cater to complex function approximators such as neural networks this section discusses a frequentist derivation of the Delta Variance, which relies on milder assumptions: As it is frequentist it does not consider posterior distributions. This allows us to side-step any questions about the shape and tractability of posterior distributions for neural networks. It does not require global convexity or a unique optimum. Convergence of the parameters to some local optimum together with locally bounded second derivatives is sufficient. In Proposition 2 we observe that the Delta Variance computes an infinitesimal approximation to the leave-one-out variance (see Definition 5) for choice of $\Sigma = H _ { f } ^ { - 1 } \hat { F } _ { f } H _ { f } ^ { - 1 }$ :

$$
\underbrace { \Delta _ { u ( z ) } ^ { \top } \Sigma \Delta _ { u ( z ) } } _ { \mathrm { D e l t a V a r i a n c e } } = \underbrace { \mathbb { V } _ { \theta \sim I J } \left[ u _ { \theta } ( z ) \right] } _ { \mathrm { I n f i n i t e s s i m a l L O O V a r i a n c e } } \approx \underbrace { \mathbb { V } _ { \theta \sim L O O } \left[ u _ { \theta } ( z ) \right] } _ { \mathrm { L O O V a r i a n c e } }
$$

The infinitesimal approximation to the leave-one-out variance (also known as the infinitesimal jackknife (Jaeckel 1972)) is defined as follows:

Definition 4. Let $\theta _ { i }$ be the parameters resulting from training $f _ { \theta }$ on data $\mathcal { D }$ with $x _ { i }$ down-weighted by $\epsilon$ (i.e. from weight 1 $t o \mathrm { ~ 1 ~ - ~ } \epsilon )$ , then the $\epsilon$ -Leave-One-Out Variance is defined as

$$
\mathbb { V } _ { \theta \sim \epsilon - L O O } \left[ u _ { \theta } ( z ) \right] : = \frac { N - \epsilon } { \epsilon ^ { 2 } } \mathbb { V } _ { i \sim U ( 1 , \dots , N ) } \left[ u _ { \theta _ { i } } ( z ) \right]
$$

Definition 5. With slight abuse of notation we define the Infinitesimal LOO Variance as the limit of the $\epsilon$ -Leave-OneOut Variance:

$$
\mathbb { V } _ { I J } [ u _ { \theta } ( z ) ] : = \operatorname* { l i m } _ { \epsilon  0 } \mathbb { V } _ { \theta \sim \epsilon - L O O } [ u _ { \theta } ( z ) ]
$$

Proposition 2. The Delta Variance equals the infinitesimal LOO Variance for $\Sigma = H _ { f } ^ { - 1 } \hat { F } _ { f } H _ { f } ^ { - 1 } \times N$ :

$$
\mathbb { V } _ { I J } \left[ u _ { \theta } ( z ) \right] = \Delta _ { u ( z ) } ^ { \top } \Sigma \Delta _ { u ( z ) }
$$

Proof. See Schmitt, Shawe-Taylor, and van Hasselt (2025).

# 5.3 Adversarial Data Interpretation

Sometimes it is of interest to quantify how much a prediction changes if the training dataset is subject to adversarial data injection. Intuitively this is connected to epistemic uncertainty: one may argue that predictions are more robust the more certain we are about their parameters and vice versa. In the appendix of Schmitt, Shawe-Taylor, and van Hasselt (2025) we show that this intuition also holds mathematically. In particular we observe that:

1. The Delta Variance with $\Sigma = H _ { f } ^ { - 1 }$ computes how much a quantity of interest $u _ { \theta } ( z )$ changes if an adversarial datapoint is injected.   
2. This adversarial interpretation is technically equivalent to the Laplace Posterior approximation (from Section 5.1) – even though interestingly both start with different assumptions and objectives.

# 5.4 Out-of-Distribution Interpretation

We show that a large Delta Variance of $u _ { \theta } ( z )$ implies that its input $z$ is out-of-distribution with respect to the training data. This relates to epistemic uncertainty intuitively: a model is likely to be uncertain about data-points that differ from its training data. The derivation in the appendix of Schmitt, Shawe-Taylor, and van Hasselt (2025) is based on the Mahalanobis Distance (Mahalanobis 1936) – a classic metric for out of distribution detection. It accounts for the possibility that $f _ { \theta } \neq u _ { \theta }$ and relies on minimal assumptions only requiring existence of gradients and that the training of $f _ { \theta }$ has converged.

# 6 Experiments

To empirically study the Delta Variance we build on the stateof-the-art GraphCast weather forecasting system (Lam et al. 2023) which trains a neural network $f _ { \theta } ( x )$ to predict the weather 6 hours ahead. This $f _ { \theta } ( x )$ is then iterated multiple times to make predictions up to 10 days into the future. We define various quantities of interest $u _ { \theta }$ such as the average rainfall in an area or the expected power of a wind turbine at a particular location and compute their Epistemic Variance. We assess the Epistemic Variance predictions on 5 years of hold-out data using multiple metrics such as the correlation between predicted variance and prediction error and the likelihood of the quantities of interest. Empirically Delta Variances with a diagonal Fisher approximation yield competitive results at lower computational cost – see Figure 3. Next we give an overview on the experimental methodology – please consider Schmitt, Shawe-Taylor, and van Hasselt (2025) for more technical details.

# 6.1 Weather Forecasting Benchmark

GraphCast Training We build on the state-of-the-art GraphCast weather prediction system. It trains a graph neural network to predict the global weather state 6 hours into the future. This step function $x _ { t + 1 } = f _ { \theta } ( x _ { t } )$ is then iterated to predict up to 10 days into the future. The global weather state $x$ is represented as a grid with 5 surface variables and and 6 atmospheric variables at 37 levels of altitude (see Lam et al. 2023, for details). The authors consider a grid-sizes of 0.25 degrees. To save resources we retrain the model for a grid size of 4 degrees and reduce the number of layers and latents each by factor a of 2. Finally we skip the fine-tuning curriculum for simplicity. Besides the graph neural network we also consider a standard convolutional neural network. Training data ranges from 1979-2013 with validation data from 2014-2017 and holdout data from 2018-2021 resulting in about $1 0 0 \mathrm { G B }$ of weather data.

Quantities of Interest First we define 126 different quantities of interest $u _ { \theta }$ based on 4 topics that we evaluate on the hold-out data (2018-2021) for two different neural network architectures: 1) Precipitation at various times into the future. 2) Inspired by wind turbine energy yield we measure the third power of wind-speed at various times into the future. 3) Inspired by flood risk we measure precipitation averaged over areas of increasing size five days into the future. 4) Inspired by health emergencies we predict the maximum temperature maximized over areas of increasing size five days into the future. The first two quantities are predicted $1 , \ldots , 5$ days ahead. The last two are measured 5 days ahead in quadratic areas with inradii ranging from 1 to 6. These measurements take place at 6 preselected capital cities. Finally note that we never train $u _ { \theta }$ as it can be derived using $f _ { \theta }$ .

![](images/03f6010a01b48da246307627db4c9d525560d84630e1baa6c0fa7beb3a1b2638.jpg)  
Figure 3: Comparison of variance estimators in terms of their inference cost and prediction quality. The quantities of interest are based on the GraphCast (Lam et al. 2023) weather prediction system that iterates a learned neural network dynamics model to form predictions. We evaluate the selected variance estimators based on three different evaluation criteria (Log-likelihood, correlation to prediction error and AUC akin to Van Amersfoort et al. (2020)). Lines indicate 2 standard errors. Delta Variances yield similar results as popular alternatives for lower computational cost. On average ensembles achieve the highest quality and Delta Variances the lowest computational overhead. See Section 7.1 for the finetuned Delta Variance.

Evaluation Methodology The data from 2018-2021 is held out for evaluation resulting in approximately $6 \times 1 0 ^ { 3 }$ different (input, target-value) pairs for each of the 252 quantities of interest $u _ { \theta }$ . For each pair we obtain a prediction error $| y - u _ { \theta } ( z ) |$ and corresponding variance predictions $\nu ( z )$ . Unfortunately many practical applications do not admit ground truth values for Epistemic Variance that one could compare variance estimators to. Instead there are multiple popular approaches in the literature relying on the prediction error, which is subject to both epistemic and aleatoric uncertainty. In Figure 3 we consider multiple such different criteria:

1. Akin to Van Amersfoort et al. (2020) AUC considers how fast the average $L _ { 1 }$ error decreases when data-points are removed from the dataset – in the order of their largest predicted Epistemic Variance.   
2. We consider the Pearson correlation between absolute error and predicted epistemic standard deviation: corr $( | y - |$ $u _ { \theta } ( z ) | , \sqrt { \nu ( z ) } )$ .   
3. To evaluate the Log-likelihood of observations $y$ we interpret $u _ { \theta } ( z )$ as the mean of a Laplace distribution with variance derived from the predicted Epistemic Variance $\nu ( z )$ . We parameterize the Laplace distribution such that its variance decomposes in a constant $\alpha$ and the predicted Epistemic Variance $\nu ( z )$ scaled by $\beta$ . Intuitively $\alpha$ represents the aleatoric variance and $\beta \nu ( z )$ represents the Epistemic Variance: Laplace $\mathbf { \bar { \rho } } ( \mu = \dot { u } _ { \theta } ( \mathbf { \bar { \rho } } _ { z } ) , 2 \dot { b ^ { 2 } } = \alpha + \beta \nu ( \mathbf { \bar { \rho } } _ { z } ) )$ . Both $\alpha$ and $\beta$ are learned on the validation data (2014-   
2017) that is used for hyper-parameter selection. We then observe how well it models the actual observed target values $y$ from the evaluation data (2018-2021).

Finally to reduce variance we define the Improvement of a variance estimator as the difference of its score to the score obtained by the ensemble estimator. Intuitively this indicates the loss in Quality when using an estimator in place of an ensemble. This procedure is repeated for each of the 252 quantities of interest $u _ { \theta }$ .

# 7 Illustrations and Extensions

To highlight the generality of our approach we illustrate two extensions in this section.

1. By learning $\Sigma$ to represent uncertainty well, we generalize the parametric from of Delta Variances beyond Fisher and Hessian matrices and observe improved results in the GraphCast benchmark – see Figure 3.   
2. We consider an example where $u _ { \theta }$ is not an explicit function but maps to a fixed-point of an iterative algorithm. We observe that it is possible to compute the Delta Variance of fixed-points using the implicit function theorem. Applied to an eigenvalue solver we observe empirically that the Delta Variance variance yields reasonable uncertainty estimates – see Figure 4.

# 7.1 Learning $\pmb { \Sigma }$

In Section 5 we observed that Delta Variances with special $\Sigma$ such as the Fisher Information approximate theoretically established measures of uncertainty. In this section we observe that $\Sigma$ may also be learned or fine-tuned. In an illustrative example we differentiate the Delta Variances with respect to $\Sigma$ and use gradient descent to obtain an improved $\Sigma$ . This may be helpful to improve the uncertainty prediction or to improve a downstream use-case if the variance is used in a larger system.

Fine-Tuning $\pmb { \Sigma }$ Example We present a simple instance of fine-tuning a few parameters of $\Sigma$ , which empirically yields improved results – see Figure 3. Note that $\Sigma$ is approximated block-diagonally in most practical cases to limit the computational requirements – with one block for each weight vector in each neural network layer. Hence the Delta Variance splits into a sum of per-block Delta Variances derived from per-block gradients $\Delta _ { i }$ :

$$
\Delta _ { f ( x ) } ^ { \top } \Sigma \Delta _ { f ( x ) } = \sum _ { i } \Delta _ { i } ^ { \top } \Sigma _ { i } \Delta _ { i }
$$

In this example we introduce a factor to rescale $\Sigma$ within each block. Intuitively this adjusts the importance of each layer. Since only a few parameters need to be estimated we only need little fine-tuning data. This is applicable in situations where there is a small amount of training data for $u _ { \theta }$ . In our experiments we optimize the coefficients of this linear combination using gradient descent to improve the loglikelihood or correlation on a small set of held-out validation data. Note that the per-layer variances can be cached which reduces the optimization problem significantly.

# 7.2 Epistemic Variance of Iterative Algorithms and Implicit Functions

So far we considered quantities of interest $u _ { \theta }$ that are explicit functions of the parameters $\theta$ . Here we consider an example where the quantity of interest is an implicit function: $u _ { \theta }$ maps to the fixed-point (solution) of an iterative algorithm for which there is no closed-form formula that we could differentiate to obtain its gradient.

Given some initial point $w _ { 0 }$ the iteration $w _ { k + 1 } = F _ { \theta } ( w _ { k } )$ may converge to a fixed-point $\boldsymbol { w } _ { \boldsymbol { \theta } } ^ { \ast }$ that depends on the parameters $\theta$ . To estimate $\mathbb { V } _ { \theta } \left[ w _ { \theta } ^ { \ast } \right]$ we need to define $u _ { \theta }$ as follows, which can not be differentiated with regular back-propagation due to the limit

$$
u _ { \theta } ( w _ { 0 } ) : = \operatorname* { l i m } _ { k \to \infty } \underbrace { F _ { \theta } \circ \dots \circ F _ { \theta } } _ { k { \mathrm { ~ t i m e s } } } ( w _ { 0 } ) = w _ { \theta } ^ { * }
$$

Implicit Epistemic Variance Calculation To compute the Delta Variance of an implicitly defined $u _ { \theta }$ we need its gradient $\nabla _ { \boldsymbol { \theta } } u _ { \boldsymbol { \theta } }$ . This can be obtained under mild conditions using the implicit function theorem. Let us denote $w _ { k + 1 } = F _ { \theta } ( w _ { k } )$ any fixed-point iteration converging to $w ^ { \ast }$ with the corresponding non-linear equation $\bar { G _ { \theta } } ( \bar { w } ) : = F _ { \theta } ( w ) - w = 0$ The implicit function theorem yields the gradient of $u _ { \theta }$ by considering the Jacobian of $G$ at the fixed-point $w ^ { \ast }$ :

$$
\Delta _ { w ^ { * } } : = \nabla _ { \theta } u _ { \theta } = - \left( \nabla _ { w ^ { * } } G _ { \theta } ( w ^ { * } ) \right) ^ { - 1 } \nabla _ { \theta } G _ { \theta } ( w ^ { * } )
$$

whenever $G$ is continuously differentiable and the inverse of $\nabla _ { w ^ { * } } G _ { \theta } ( w ^ { * } )$ exists. Now we can compute the Epistemic Variance as $\mathbb { V } _ { \theta } \left[ w ^ { * } \right] \approx \Delta _ { w ^ { * } } ^ { \top } \Sigma \Delta _ { w ^ { * } }$ .

Eigenvalue Example Eigenvalues are a quantity of interest in structural engineering. As an illustrative example we consider the eigenvalues $\lambda _ { i } ( A _ { \theta } )$ of a finite element model matrix $A _ { \theta } = M _ { \theta } ^ { - 1 } K _ { \theta }$ that indicates the stability of a physical structure. If the parameters $\theta$ are uncertain the eigenvalues will be uncertain as well. Recall that they are the solutions to $\operatorname* { d e t } ( A _ { \theta } - \lambda I ) = 0 .$ . We can estimate the Epistemic Variance of an eigenvalue $\mathbb { V } _ { \theta } \left[ \lambda _ { i } ( A _ { \theta } ) \right]$ using Delta Variances if we obtain the gradient of the eigenvalue $\nabla _ { \theta } \lambda _ { i } ( A _ { \theta } )$ . To this extend we need the implicit function theorem as $\lambda _ { i } ( A _ { \theta } )$ is an implicitly defined function – please consider Schmitt, ShaweTaylor, and van Hasselt (2025) for technical details.

![](images/49e31f5a3c44263370cb83bd1ffcf070cb910ff50e20b010311dcf402d4cbf67.jpg)  
Figure 4: To investigate more intricate quantities of interest, we consider the mapping from a matrix $A _ { \theta }$ to its eigenvalue $u _ { \theta } = \lambda _ { i } ( A _ { \theta } )$ . This function is not explicit and computed using iterative algorithms, but we can use the implicit function approach to estimate the Delta Variance. Here $A _ { \theta }$ is an illustrative finite-element problem with 11-dimensional parameters $\theta$ and 5 eigenvalues.

# 8 Related Work

The proposed Delta Variance family bridges and extends Bayesian, frequentist and heuristic notions of variance. Furthermore it generalizes related work by considering explicit and implicit quantities of interest other than the neural network itself $u _ { \theta } \neq f _ { \theta }$ and permits learning improved covariances $\Sigma -$ see Section 7. Below we give a brief historic account of related methods that mostly consider the $u _ { \theta } = f _ { \theta }$ case.

Delta Method The Delta Method dates back to Cotes (1722), Lambert (1765) and Gauss (1823) in the context of error propagation and received a modern treatment by Kelley (1928); Wright (1934); Doob (1935); Dorfman (1938) – see Gorroochurn (2020) for a historical account. Denker and LeCun (1990) apply the Delta Method to the outputs of neural networks $f _ { \theta } ( x )$ and Nilsen et al. (2022) improves computational efficiency. When applied to neural networks the Delta Method requires strong assumptions about the posterior (e.g. unique optimum) or training process, which have not been proven to hold. Delta Variances – named after the Delta Method – provide multiple alternative theoretical justification through its unifying perspective. Furthermore Delta Variances generalize to the $u _ { \theta } \neq f _ { \theta }$ case and other $\Sigma$ .

Laplace Approximation Building on work by Gull (1989), MacKay (1992b) and Ritter, Botev, and Barber (2018) apply the Laplace approximation to neural networks. Approximating functions at an optimum by what should later be called a Gaussian distribution dates back to Laplace (1774). While only applicable to a single optimum MacKay (1992b) heuristically argues for its applicability to posterior distributions of neural networks. Given such Gaussian posterior approximation they apply the Delta Method yielding a special instance of the Delta Variance family with uθ = fθ and Σ = Hf−1 – see Section 5.1.

Influence Functions and Jackknife Methods Influence functions were proposed in Hampel (1974) concurrently with the closely related Infinitesimal Jackknife by Jaeckel (1972) which approximates cross validation (Quenouille 1949). Koh and Liang (2017) apply the influence function analysis to neural networks to evaluate how training data influences predictions. In Sections 5.2 and 5.3 we apply similar techniques to general quantities of interest different from $f _ { \theta }$ .

Uncertainty Estimation for Deep Neural Networks We focus our comparison on two popular methods: Lakshminarayanan, Pritzel, and Blundell (2017) train multiple neural networks to form an ensemble and Gal and Ghahramani (2016) which re-interprets Monte-Carlo dropout as variational inference. In Table 1 we compare their properties with Delta Variances observing that they come at larger inference cost. Osband et al. (2023) aims to reduce the training costs of ensemble methods. To this extent they change the neural network architecture and training procedure, however how to reduce the remaining $k$ -fold inference cost and memory requirements remain open research questions. Other popular methods come with similar requirements to change the architecture or training procedure (Blundell et al. 2015; Van Amersfoort et al. 2020; Immer, Korzepa, and Bauer 2021), while approaches like Sun et al. (2022) are of nonparametric flavour exhibiting inference cost that increases with the dataset size. SWAG (Maddox et al. 2019) reduces the training and memory cost by considering an ensemble of parameters from a single learning trajectory with stochastic gradient descent and approximating it with a Gaussian posterior. For inference they employ expensive k-fold sampling. We note that it is natural to derive a SWAG-inspired Delta Variances that employs the $\Sigma$ from SWAG inside the computationally efficient Delta Variance formula – we leave those considerations for future research. Finally Kallus and McInerney (2022) propose a Delta Method inspired approach to approximate Epistemic Variance with an ensemble of two predictors and Schnaus et al. (2023) learn scale parameters in Gaussian prior distributions for transfer learning.

# 9 Conclusion

We have addressed the question of how the uncertainty from limited training data affects the computation of downstream predictions in a system that relies on learned components. To this extent we proposed the Delta Variance family, which unifies and extends multiple related approaches theoretically and practically. We discussed how the Delta Variance family can be derived from six different perspectives (including Bayesian, frequentist, adversarial robustness and outof-distribution detection perspectives) highlighting its wide theoretical support and providing a unifying view on those perspectives. Next we presented extensions and applications of the Delta Variance family such as its compatibility with implicit functions and ability to be improved through finetuning. Finally an empirical validation on a state-of-the-art weather forecasting system shows that Delta Variances yield competitive results more efficiently than other popular approaches.