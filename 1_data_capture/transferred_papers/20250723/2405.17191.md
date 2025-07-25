# MCGAN: Enhancing GAN Training with Regression-Based Generator Loss

Baoren Xiao1, Hao $\mathbf { N _ { i } ^ { \bullet 1 } }$ , Weixin Yang2

1University College London 2University of Oxford baoren.xiao.18@ucl.ac.uk, h.ni@ucl.ac.uk, wxy1290g@gmail.com

# Abstract

Generative adversarial networks (GANs) have emerged as a powerful tool for generating high-fidelity data. However, the main bottleneck of existing approaches is the lack of supervision on the generator training, which often results in undamped oscillation and unsatisfactory performance. To address this issue, we propose an algorithm called Monte Carlo GAN (MCGAN). This approach, utilizing an innovative generative loss function, termed the regression loss, reformulates the generator training as a regression task and enables the generator training by minimizing the mean squared error between the discriminator’s output of real data and the expected discriminator of fake data. We demonstrate the desirable analytic properties of the regression loss, including discriminability and optimality, and show that our method requires a weaker condition on the discriminator for effective generator training. These properties justify the strength of this approach to improve the training stability while retaining the optimality of GAN by leveraging strong supervision of the regression loss. Extensive experiments on diverse datasets, including image data (CIFAR-10/100, FFHQ256, ImageNet, and LSUN Bedroom), time series data (VAR and stock data), and video data, are conducted to demonstrate the flexibility and effectiveness of our proposed MCGAN. Numerical results show that the proposed MCGAN is versatile in enhancing a variety of backbone GAN models and achieves consistent and significant improvement in terms of quality, accuracy, training stability, and learned latent space.

# Code — https://github.com/DeepIntoStreams/MCGAN Extended version — https://arxiv.org/abs/2405.17191

# Introduction

In recent years, Generative Adversarial Network (GAN) (Goodfellow et al. 2014) has become one of the most powerful tools for realistic image synthesis. However, the instability of the GAN training and unsatisfying performance remains a challenge. To combat it, much effort has been put into developing regularization methods, see (Gulrajani et al. 2017; Mescheder, Geiger, and Nowozin 2018; Miyato et al. 2018; Kang, Shin, and Park 2022). Additionally, as (Arjovsky and Bottou 2017) pointed out, the generator usually suffers gradient vanishing and instability due to the singularity of the denominator showed in the gradient when the discriminator becomes accurate. To address this issue, some work has been done to develop better adversarial loss (Lim and Ye 2017; Mao et al. 2017; Arjovsky, Chintala, and Bottou 2017). As a variant of GAN, conditional GAN (cGAN) (Mirza and Osindero 2014) is designed to learn the conditional distribution of target variable given conditioning information. It improves the GAN performance by incorporating conditional information to both the discriminator and generator, we hence have better control over the generated samples (Zhou et al. 2021; Odena, Olah, and Shlens 2017).

Unlike these works on the regularization method and adversarial loss, our work focuses on the generative loss function to enhance the performance of GAN training. In this paper, we propose a novel generative loss, termed as the regression loss $\mathcal { L } _ { R }$ , which reformulates the generator training as the least-square optimization task. This regression loss underpins our proposed MCGAN, an enhancement of existing GAN models achieved by replacing the original generative loss with our regression loss. This approach leverages the expected discriminator $D ^ { \phi }$ under the fake measure induced by the generator. Benefiting from the strong supervision lying in the regression loss, our approach enables the generator to learn the target distribution with a relatively weak discriminator in a more efficient and stable manner.

The main contributions of our paper are three folds:

• We propose the MCGAN methodology for enhancing both unconditional and conditional GAN training. • We establish the theoretical foundation of the proposed regression loss, e.g., the discriminability, optimality, and improved training stability. A simple but effective toy example of Dirac-GAN is provided to show that our proposed MCGAN successfully mitigates the nonconvergence issues of conventional GANs by incorporating regression loss. • We empirically validate the consistent improvements of MCGAN over various GANs across diverse data types (i.e., images, time series, and videos). Our approach improves quality, accuracy, training stability, and learned latent space, showing its generality and flexibility.

Related work GANs have demonstrated their capacity to simulate high-fidelity synthetic data, facilitating data sharing and augmentation. Extensive research has focused on designing GAN models for various data types, including images (Han et al. 2018), time series (Yoon, Jarrett, and Van der Schaar 2019; Xu et al. 2020; Ni et al. 2021), and videos (Gupta, Keshari, and Das 2022). Recently, Conditional GANs (cGANs) have gained significant attention for their ability to generate synthetic data by incorporating auxiliary information (Yoon, Jarrett, and Van der Schaar 2019; Liao et al. 2024; Xu et al. 2019). For the integer-valued conditioning variable, conditional GANs can be roughly divided into two groups depending on the way of incorporating the class information: Classification-based and Projectionbased cGANs (Odena, Olah, and Shlens 2017; Miyato and Koyama 2018; Kang et al. 2021; Zhou et al. 2021; Mirza and Osindero 2014; Hou et al. 2022). For the case where conditioning variable is continuous, the training of conditioning GANs is more challenging. For example, conditional WGAN suffers difficulty in estimating the conditional expected discriminator of real data due to the need for recalibration per every discriminator update (Liao et al. 2024). Attempts are made to mitigate this issue, such as conditional SigWGAN (Liao et al. 2024), which is designed to tackle this issue for time series data.

# Preliminaries

# Generative Adversarial Networks

Generative adversarial networks (GANs) are powerful tools for learning the target distribution from real data to enable the simulation of synthetic data. To this goal, GAN plays a min-max game between two networks: Generator $( G )$ and Discriminator $( D )$ . Let $\chi$ denote the target space and $\mathcal { Z }$ be the latent space. Then Generator $G ^ { \theta }$ is defined as a parameterised function that maps latent noise $z \in { \mathcal { Z } }$ to the target data $x \in \mathcal { X }$ , where $\theta \in \Theta$ is the model parameter of $G$ . Discriminator $D ^ { \phi } : \mathcal { X }  \mathbb { R }$ discriminates between the real data and fake data generated by the generator.

Let $\mu$ and $\nu _ { \theta }$ denote the true measure and fake measure induced by $G ^ { \theta }$ . For generality, the objective functions of GANs can be written in the following general form:

$$
\begin{array} { r l r } & { \operatorname* { m a x } _ { \phi } \mathcal { L } _ { D } ( \phi ; \theta ) = \mathbb { E } _ { \mu } \left[ f _ { 1 } ( D ^ { \phi } ( X ) ) \right] + \mathbb { E } _ { \nu _ { \theta } } \left[ f _ { 2 } ( D ^ { \phi } ( X ) ) \right] , } & \\ & { \operatorname* { m i n } _ { \theta } \mathcal { L } _ { G } ( \theta ; \phi ) = \mathbb { E } _ { \nu _ { \theta } } \left[ h ( D ^ { \phi } ( X ) ) \right] , } & { ( 1 ) } \end{array}
$$

where $f _ { 1 } , \ f _ { 2 }$ and $h$ are real-valued functions. Different choices of $f _ { 1 } , f _ { 2 }$ and $h$ lead to different GAN models.

There are extensive studies concerned with how to measure the divergence or distance between $\mu$ and $\nu _ { \theta }$ as the improved GAN loss function, which are instrumental in stabilising the training and enhancing the generation performance. Examples include Hinge loss (Lim and Ye 2017), Wasserstein loss (Arjovsky, Chintala, and Bottou 2017), Least squares loss (Mao et al. 2017), Energy-based loss (Zhao 2016) among others. Many of them satisfy Eqn. (1).

Example 1. • classical GAN (Goodfellow et al. 2014): $f _ { 1 } ( w ) = \log ( w )$ and $f _ { 2 } ( w ) = - h ( w ) = \log ( 1 - w )$ .   
• HingeGAN (Lim and Ye 2017): $f _ { 1 } ( w ) ~ = ~ f _ { 2 } ( - w ) ~ =$ $- \operatorname* { m a x } ( 0 , 1 - w )$ , and $h ( w ) = - w$ .

• Wasserstein GAN (Arjovsky, Chintala, and Bottou 2017) : $f _ { 1 } ( w ) = f _ { 2 } ( - w ) \stackrel { . } { = } w$ , and $h ( w ) = - w + c _ { \mu }$ , where $c _ { \mu } : = \mathbb { E } _ { X \sim \mu } [ D ^ { \phi } ( X ) ]$ .

The Wasserstein distance is linked with the mean discrepancy. More specifically, let $d _ { \phi } ( \mu , \nu )$ denote the mean discrepancy between any two distributions $\mu$ and $\nu$ associated with test function $D ^ { \phi }$ defined as $d _ { \phi } ( \mu , \nu ) =$ $\mathbb { E } _ { X \sim \mu } [ D ^ { \phi } ( X ) ] - \mathbb { E } _ { X \sim \nu } [ D ^ { \phi } ( X ) ]$ . In this case, ${ \mathcal { L } } _ { G } ( \theta ; \phi )$ could be interpreted as $d _ { \phi } ( \mu , \nu _ { \theta } )$ .

# Conditional GANs

Conditional GAN (cGAN) is a conditional version of a generative adversarial network that can incorporate additional information, such as data labels or other types of auxiliary data into both the generator and discriminative loss (Mirza and Osindero 2014). The goal of conditional GAN is to learn the conditional distribution $\mu$ of the target data distribution $X \in { \mathcal { X } }$ (i.e., image ) given the conditioning variable (i.e., image class label) $Y \in \mathcal { V }$ . More specifically, under the real measure $\mu$ , $X \times Y$ denote the random variable taking values in the space $\mathcal { X } \times \mathcal { Y }$ . The marginal law of $X$ and $Y$ are denoted by $P _ { X }$ and $P _ { Y }$ , respectively.

The conditional generator $G ^ { \theta } : \mathcal { V } \times \mathcal { Z } \to \mathcal { X }$ incorporates the additional conditioning variable to the noise input, and outputs the target variable in $\chi$ . Given the noise distribution $Z , G ^ { \theta } ( y )$ induces the fake measure denoted by $\nu _ { \theta } ( y )$ , which aims to approximate the conditional law of $\mu ( y ) : = P ( X | Y = y )$ under real measure $\mu$ . The task of training an optimal conditional generator is formulated as the following min-max game:

$$
\begin{array} { r l r } & { \mathcal { L } _ { D } ( \phi , \theta ) = \mathbb { E } _ { Y } \left[ \mathbb { E } _ { \mu ( y ) } [ f _ { 1 } ( D ^ { \phi } ( X ) ) ] + \mathbb { E } _ { \nu _ { \theta } ( y ) } [ f _ { 2 } ( D ^ { \phi } ( X ) ] \right] , } & \\ & { \mathcal { L } _ { G } ( \theta ; \phi ) = \mathbb { E } _ { Y } \left[ \mathbb { E } _ { \nu _ { \theta } ( y ) } [ h ( D ^ { \phi } ( X ) ) ] \right] , } & { ( 2 ) } \end{array}
$$

where $f _ { 1 } , f _ { 2 }$ and $h$ are real value functions as before and $\mathbb { E } _ { Y }$ denotes that the expectation is taken over $y$ sampled from $Y$ . Different from the unconditional case, $\mathcal { L } _ { D }$ and $\mathcal { L } _ { G }$ has in the outer expectation $\mathbb { E } _ { y \sim P _ { Y } }$ due to $Y$ being a random variable.

# Monte-Carlo GAN

# Methodology

In this section, we propose the Monte-Carlo GAN (MCGAN) for both unconditional and conditional data generation. Without loss of generality, we describe our methodology in the setting of the conditional GAN task.1 Consider the general conditional GAN composed with the generator loss $\mathcal { L } _ { G }$ (Eqn. (2)) and the discrimination loss $\mathcal { L } _ { D }$ outlined in the last subsection. To further enhance GAN, we propose the MCGAN by replacing the generative loss $\mathcal { L } _ { G }$ with the following novel regression loss for training the generator from the perspective of the regression, denoted by $\mathcal { L } _ { R }$ ,

$$
\begin{array} { r } { \mathcal { L } _ { R } ( \theta ; \phi ) : = \mathbb { E } _ { ( x , y ) \sim \mu } \left[ | D ^ { \phi } ( x ) - \mathbb { E } _ { \hat { x } \sim \nu _ { \theta } ( y ) } [ D ^ { \phi } ( \hat { x } ) ] | ^ { 2 } \right] , } \end{array}
$$

where the expectation is taken under the joint law $\mu$ of $X$ and $Y$ . We optimize the generator’s parameters $\theta$ by minimizing the regression loss ${ \mathcal { L } } _ { R } ( \theta ; \phi )$ . We keep the discriminator loss and conduct the min-max training as before. The training algorithm of MCGAN is given in the Appendix.

The name for Monte Carlo in MCGAN is due to the usage of the Monte Carlo estimator of expected discriminator output under the fake measure. This innovative loss function reframes the conventional generator training into a meansquare optimization problem by computing the $l ^ { 2 }$ loss between real and expected fake discriminator outputs.

Next, we explain the intuition behind $\mathcal { L } _ { G }$ and its link with optimality of conditional expectation. Let us consider a slightly more general optimization problem for $\mathcal { L } _ { R }$ :

$$
\operatorname* { m i n } _ { f \in { \mathcal C } ( { \mathcal V } , \mathbb { R } ) } \mathbb { E } _ { \mu } [ | D ^ { \phi } ( X ) - f ( Y ) | ^ { 2 } ] ,
$$

It is well known that the conditional expectation is the optimal $l ^ { 2 }$ estimator. So the minimizer to Eqn (4) is given by the conditional expectation function $f ^ { * } : \mathcal { V }  \mathbb { R }$ , defined as $f ^ { * } ( y ) = \mathbb { E } _ { \mu } [ { \bar { D } } ^ { \phi } ( X ) | Y = y ]$ . This fact motivates us to consider the conditional expectation under the fake measure, $\mathbb { E } _ { \nu _ { \theta } ( Y ) } [ D ^ { \phi } ( X ) ]$ , as the model for the mean equation $f ^ { * }$ . It leads to our regression loss $\mathcal { L } _ { R }$ , where we replace $f$ by $\mathbb { E } _ { \nu _ { \theta } ( Y ) } [ D ^ { \phi } ( X ) ]$ in Eqn. (4).

Minimising the regression loss $\mathcal { L } _ { G }$ enforces the conditional expectation of ${ \overline { { D ^ { \phi } } } } ( X )$ under fake measure $\nu _ { \theta } ( Y )$ to approach that under the conditional true distribution $\mu ( Y ) =$ $\mathbb { P } ( X | Y )$ for any given $D ^ { \phi }$ . Assume that $( G ^ { \theta } ) _ { \theta \in \Theta }$ provides a rich enough family of distributions containing the real distribution $\mu$ . Then there exists $\theta ^ { \ast } \in \Theta$ , which is a minimizer of ${ \mathcal { L } } _ { R } ( \theta , \phi )$ for all discriminator’s parameter $\phi$ , satisfying that

$$
\mathbb { E } _ { \mu ( Y ) } [ D ^ { \phi } ( X ) ] = \mathbb { E } _ { \nu _ { \theta ^ { * } } ( Y ) } [ D ^ { \phi } ( X ) ] .
$$

It implies that no matter whether the discriminator $D ^ { \phi }$ achieves the equilibrium of GAN training, the regression loss $\mathcal { L } _ { R }$ is a valid loss to optimize the generator to match its expectation of $D ^ { \phi }$ between true and fake measure.

Moreover, our proposed regression loss can effectively mitigate the challenge of the conditional Wassaserstain GAN (c-WGAN). To compute the generative loss of cWGAN, one needs to estimate the conditional expectation $\mathbb { E } _ { \mu ( Y ) } [ D ^ { \phi } ( X ) ]$ . However, when the conditioning variable is continuous, it becomes computationally expensive or even infeasible due to the need for recalibration with each discriminator update. In contrast, our regression loss does not need the estimator for $\mathbb { E } _ { \mu ( Y ) } [ D ^ { \phi } ( X ) ]$ .

# Comparison Between $\mathcal { L } _ { R }$ and $\mathcal { L } _ { G }$

In this subsection, we delve into the training algorithm of the regression loss $\mathcal { L } _ { R }$ and illustrate its advantages of enhancing the training stability in comparison with the generator loss $\mathcal { L } _ { G }$ . For ease of notation, we consider the unconditional case. To optimize the generator’s parameters $\theta$ in our MCGAN, we apply gradient-descent-based algorithms and the updating rule of $\theta _ { n }$ is given by

$$
\begin{array} { r l r } {  { \theta _ { n + 1 } = \theta _ { n } - \lambda \frac { \partial \mathcal { L } _ { R } } { \partial \theta } \big \vert _ { \theta = \theta _ { n } } } } & { } & { ( 6 ) } \\ & { } & { = \theta _ { n } - 2 \lambda \underbrace { ( \mathbb { E } _ { \mu } [ D ^ { \phi } ( X ) ] - \mathbb { E } _ { \nu _ { \theta _ { n } } } [ D ^ { \phi } ( X ) ] ) } _ { d \phi ( \mu , \nu _ { \theta _ { n } } ) } H ( \theta _ { n } , \phi ) , } \end{array}
$$

where $\lambda$ is the learning rate and

$$
H ( \theta , \phi ) = \mathbb { E } _ { z \sim P _ { Z } } [ \nabla _ { \theta } G ^ { \theta } ( z ) ^ { T } \cdot \nabla _ { x } D ^ { \phi } ( G ^ { \theta } ( z ) ) ] .
$$

Note the gradient $\textstyle { \frac { \partial { \mathcal { L } } _ { R } } { \partial \theta } }$ takes into account not only $\nabla _ { x } D ^ { \phi } ( x )$ but also $d ( \mu , \nu _ { \theta } )$ - the discrepancy between the expected discriminator outputs under two measures $\mu$ and νθ.

In contrast, employing the generator loss $\mathcal { L } _ { G }$ , the generator parameter $\theta$ is updated by the following formula:

$$
\begin{array} { r l r } {  { \theta _ { n + 1 } = \theta _ { n } - \lambda \mathbb { E } _ { z \sim P _ { Z } } \Big [ h ^ { \prime } ( D ^ { \phi } ( G ^ { \theta _ { n } } ( z ) ) ) \nabla _ { \theta } G ^ { \theta } ( z ) ^ { T } \Big | _ { \theta = \theta _ { n } } } } \\ & { } & { \quad \cdot \nabla _ { x } D ^ { \phi } ( G ^ { \theta _ { n } } ( z ) ) \Big ] . } \end{array}
$$

One can see that Eqn. (8) depends on the discriminator gradients $\nabla _ { x } D ^ { \phi } ( G ^ { \theta _ { n } } ( { \dot { z } } ) )$ heavily.

MCGAN benefits from the strong supervision of $\mathcal { L } _ { R }$ , which provides more control over the gradient behaviour during the training. When $\theta$ is close to the optimal $\theta ^ { * }$ , even if $D ^ { \phi }$ is away from the optimal discriminator, $d _ { \phi } ( \mu , \nu _ { \theta } )$ would be small and hence leads to stabilize the generator training. However, it may not be the case for the generator loss as shown in Eq. (8), resulting in the instability of generator training. For example, this issue is evident for the Hinge loss where $h ( x ) = x$ as shown in (Mescheder, Geiger, and Nowozin 2018).

# Illustrative Dirac-GAN Example

To illustrate the advantages of MCGAN, we present a toy example from (Mescheder, Geiger, and Nowozin 2018), demonstrating its resolution of the training instability in Dirac-GAN. The Dirac-GAN example involves a true data distribution that is a Dirac distribution concentrated at 0. Besides, the Dirac-GAN model consists of a generator with a fake distribution $\nu _ { \theta } ( x ) = \delta ( x - \theta )$ with $\delta ( \cdot )$ is a Dirac function and a discriminator $D ^ { \phi } ( x ) = \phi x$ .

We consider three different loss functions for both $\mathcal { L } _ { D }$ and $\mathcal { L } _ { G }$ : (1) binary cross-entropy loss (BCE), (2) Non-saturating loss and (3) Hinge loss, resulting GAN, NSGAN and HingeGAN, respectively. In this case, the unique equilibrium point of the above GAN training objectives is given by $\phi = \theta = 0$ .

In this case, the update of training GAN is simplified to

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { \phi _ { n + 1 } = \phi _ { n } + \lambda f ^ { \prime } ( - \phi _ { n } \theta _ { n } ) \theta _ { n } , } \\ { \theta _ { n + 1 } = \theta _ { n } - \lambda h ^ { \prime } ( \phi _ { n } \theta _ { n } ) \phi _ { n } . } \end{array} \right. } \end{array}
$$

where $f$ is specified as $f ( x ) = - \log ( 1 + \exp ( x ) )$ . By applying MCGAN to enhance GAN training, the update rules for the model parameters $\theta$ and $\phi$ are modified as follows:

$$
\left\{ \begin{array} { l l } { \phi _ { n + 1 } } & { = \phi _ { n } + \lambda f ^ { \prime } ( \phi _ { n } \theta _ { n } ) \theta _ { n } , } \\ { \theta _ { n + 1 } } & { = \theta _ { n } - \lambda 2 ( \phi _ { n } \theta _ { n } - \phi _ { n } c ) \phi _ { n } . } \end{array} \right.
$$

Fig. 1 (a-c) demonstrates that GAN, NSGAN and Hinge GAN all fail to converge to obtain the optimal generator parameter $\theta ^ { * } = 0$ . That is because the updating scheme of $\theta$ depends heavily on the $\phi$ . When $\phi$ fails to converge to zero, $\theta$ continues to update even if it has reached zero, and the nonzero $\theta$ further encourages $\phi$ updating away from 0, which results in a vicious cycle and the failure of both generator and discriminator. In contrast, Fig. 1(d) of MCGAN training demonstrates that the generator parameter $\theta$ successfully converges to the optimal value 0 thanks to the regression loss in (3) to bring the training stability of the generator. A 2D Gaussian mixture example is also provided in Appendix, showing that MCGAN can help mitigate model collapse.

Table 1: List of common discriminative loss functions that satisfy strict discriminability   

<html><body><table><tr><td>Name</td><td>Discriminative loss</td><td>D*（x）</td><td>a</td><td>C</td></tr><tr><td>Vanilla GAN</td><td>Binary cross-entropy</td><td>Pu（）（（@）</td><td>1</td><td>1/2</td></tr><tr><td>LSGAN</td><td>Least square loss</td><td>p(x）+p(（(</td><td>sign (α - β)</td><td></td></tr><tr><td>Hinge GAN</td><td>Hinge loss</td><td>21{pμ(x）≥Pvg(x)-1</td><td>1</td><td>0</td></tr><tr><td>Energy GAN</td><td>Energy-based loss</td><td>m1{pμ(x)<pvq(x)}</td><td>sign(-m)</td><td>m2</td></tr><tr><td>f-GAN</td><td>VLB on f-divergence</td><td>（</td><td>1</td><td>f'(1)</td></tr></table></body></html>

-2.0 0 2.0   
2.0 φ O   
-2.0 (a) GAN (b) NSGAN (c)HingeGAN (d)MCGAN

# Discriminability and Optimality of MCGAN

To ensure that MCGAN training leads to the optimal generator $\nu _ { \theta ^ { * } } = \mu$ , one needs the sufficient discriminative power of $D ^ { \phi }$ . The discriminative power of $D ^ { \phi }$ is determined by the discriminative loss function $\mathcal { L } _ { D }$ , which is usually defined as certain divergences, such as JS divergence in GAN (Goodfellow et al. 2014). However, computing such divergence involves finding the optimal discriminator that optimizes the objective function, which might be challenging in practice. See (Liu, Bousquet, and Chaudhuri 2017) for a comprehensive description of the discriminative loss function.

Instead of needing an optimal discriminator, we introduce the weaker condition, discriminability of the discriminator $D ^ { \phi }$ , to ensure the generator’s optimality for the training.

Definition 1 (Discriminability). A discriminator

$$
\mathcal { P } ( \mathcal { X } ) \times \mathcal { P } ( \mathcal { X } ) \times \mathcal { X }  \mathbb { R } ; \quad ( \mu , \nu , x ) \mapsto D ^ { \phi _ { \mu , \nu } } ( x ) ,
$$

where $\phi . , . : \mathcal { P } ( \mathcal { X } ) \times \mathcal { P } ( \mathcal { X } )  \Phi$ , is said to have discriminability if there exist two constants $a \in \{ - 1 , 1 \}$ and $c \in \mathbb { R }$ such that for any two measures $\mu , \nu \in \mathcal { P } ( \mathcal { X } )$ , it satisfies that

$$
a ( D ^ { \phi _ { \mu , \nu } } ( x ) - c ) ( p _ { \mu } ( x ) - p _ { \nu } ( x ) ) > 0 ,
$$

for all $x \in { \mathcal { A } } ^ { \mu , \nu } : = \{ x \in \mathcal { X } : p _ { \mu } ( x ) \neq p _ { \nu } ( x ) \}$ . We denote the set of discriminators with discriminability as $\mathcal { D } _ { D i s }$ .

The discriminability of the discriminator can be interpreted as the ability to distinguish between $\nu$ and $\mu$ pointwisely over $\mathbf { \mathcal { A } } ^ { \mu , \nu }$ by telling the sign (or the opposite sign) of $p _ { \mu } ( x ) - p _ { \nu } ( x )$ . In (9), if $a = 1$ , the constant $c$ can be regarded as a criterion in the sense that $D ^ { \phi _ { \mu , \nu } } ( x ) - c$ is positive when $p _ { \mu } ( x ) > p _ { \nu } ( x )$ and vice versa.

The discriminability covers a variety of optimal discriminators in GAN variants. We present in Table 1 a list of optimal discriminators of some commonly used GAN variants along with their values of $a$ and $c$ . The detailed description can be found in the Appendix. Although discriminability can be obtained by training the discriminator via certain $\mathcal { L } _ { D }$ , it is worth emphasizing that the discriminator does not necessarily need to reach its optimum to obtain discriminability.

Assumption 1. Let $H$ be defined in Eqn. (7). The equality $H ( \theta , \phi ) = { \vec { 0 } }$ holds only if $( \theta , \phi )$ reaches the equilibrium point where $\nu _ { \theta } = \mu$ .

Now, we establish the optimality of $\mu = \nu _ { \theta }$ in the following theorem under the regularity condition (Assumption 1).

Theorem 1. Assume Assumption $\boldsymbol { { \mathit { 1 } } }$ holds, and let $\phi _ { \cdot , \cdot } ^ { \prime } :$ $\mathcal { P } ( \mathcal { X } ) \times \mathcal { P } ( \mathcal { X } )  \Phi$ be a parameterization map such that $D ^ { \phi ^ { \prime } } \cdot \cdot : \mathcal { P } ( \mathcal { X } ) \times \mathcal { P } ( \mathcal { X } ) \times \mathcal { X }  \mathbb { R }$ has discriminability, i.e. $D ^ { \phi ^ { \prime } } \in \mathcal { D } _ { D i s }$ . If $\theta ^ { * }$ is a local minimizer of $\mathcal { L } _ { G } ( \theta ; \phi _ { \mu , \nu _ { \theta } } ^ { \prime } , \mu )$ defined in (3), then $\nu _ { \theta ^ { * } } = \mu$ .

Theorem 1 implies that MCGAN can effectively learn the data distribution $\mu$ without requiring the discriminator to reach its optimum; the discriminability is sufficient, which is again attributed to the strong supervision provided by regression loss $\mathcal { L } _ { R }$ . We defer the proof of Theorem 1 and other theoretical properties of MCGAN, e.g., improved training stability and relation to $f$ -divergence to the Appendix.

# Numerical Experiments

To validate the efficacy of the proposed MCGAN method, we conduct extensive experiments on a broad range of data, including image, time series, and video data for various generative tasks. For image generation, the conditioning variables are categorical, whereas for time series and video generation tasks, the conditioning variables are continuous. To show the flexibility of MCGAN to enhance different GAN backbones, we choose several state-of-the-art GAN models with different discriminative losses (i.e., BCE and Hinge loss) as baselines. Various test metrics and qualitative analysis are employed to give a comprehensive assessment of the quality of synthetic data generation.

The full implementation details of numerical experiments, including models, test metrics, hyperparameters, optimizer and supplementary numerical results, can be found in Appendix. Moreover, we will open-source the codes and final checkpoints upon publication for reproducibility.

# Unconditional and Conditional Image Generation

Datasets We conduct conditional image generation tasks using the CIFAR-10 and CIFAR-100 datasets (Alex 2009), which are standard benchmarks with 60K 32x32 RGB images across 10 and 100 classes, respectively. To further validate our MC method on larger and higher-resolution datasets, we include: 1) the unconditional FFHQ256 dataset, which contains 70K 256x256 human face images, 2) the conditional ImageNet64 dataset, which has 1.2 million $6 4 \mathrm { x } 6 4$ images across 1,000 classes, and 3) the unconditional LSUN bedroom data, which has 3 million 256x256 images.

We validate our method using two different backbones, BigGAN (Brock, Donahue, and Simonyan 2018) and StyleGAN2 (Karras et al. 2020b). The test metrics include Inception Score (IS), Fr´echet Inception Distance (FID), and Intra Fr´echet Inception Distance (IFID) together with two recognizability metrics Weak Accuracy (WA) and Strong Accuracy (SA). To alleviate the overfitting and improve the generalization, we also increase data efficiency by using the Differentiable Augmentation (DiffAug) (Zhao et al. 2020).

We focus on the CIFAR-10 for in-depth analysis, with a brief summary of the results on the other datasets.

Faster Training Convergence In Figure 2, we plot the learning curves in terms of FID and IS during the training. It shows that the MC method tends to have much faster convergence and ends at a considerably better level in both baselines of using Hinge loss and BCE loss.

Improved Fidelity Metrics As shown in Table 2, our MC method considerably improves all the baselines independently of the choice of discriminative loss $( \mathcal { L } _ { D } )$ . Specifically, when using Hinge loss as $\mathcal { L } _ { D }$ along with DiffAug, the MC method improves the FID from 4.43 to 3.61, comparable to the state-of-the-art FID result of (Kang, Shin, and Park 2022). Also, its IS score is significantly increased from 9.61 to 9.96, indicating better diversity of the generated samples.

In addition, applying the MC method to the cStyleGAN2 backbone results in an FID improvement of approximately 0.08. Notably, the combination of Hinge $+ \mathrm { M C } +$ DiffAug achieves an FID of 2.16, which, to our knowledge, is the best FID achieved using StyleGAN2 as the backbone (Kang et al. 2021; Kang, Shin, and Park 2022; Tseng et al. 2021)

Table 2: Quantitative results of image generation on CIFAR10 using BigGAN/StyleGAN2 w/o and with our MC method and Differentiable Augmentation.   

<html><body><table><tr><td rowspan="2">Loss</td><td colspan="3">Hinge</td><td colspan="3">BCE</td></tr><tr><td>IS↑</td><td>FID↓</td><td>IFID↓</td><td>二 IS↑</td><td>FID↓</td><td>IFID↓</td></tr><tr><td>Metrics BigGAN</td><td>9.27</td><td>5.31</td><td>16.20</td><td>9.30</td><td>5.55</td><td>16.62</td></tr><tr><td>+DiffAug</td><td>9.61</td><td>4.43</td><td>14.60</td><td>9.51</td><td>4.71</td><td>14.83</td></tr><tr><td>+MC</td><td>9.66</td><td>4.51</td><td>14.71</td><td>9.62</td><td>4.61</td><td>14.82</td></tr><tr><td>+MC+DiffAug</td><td>9.96</td><td>3.61</td><td>13.60</td><td>9.94</td><td>3.93</td><td>13.72</td></tr><tr><td>StyleGAN2</td><td>-</td><td></td><td></td><td>10.17</td><td>3.7</td><td>14.04</td></tr><tr><td>+DiffAug</td><td>10.19</td><td>2.25</td><td>11.40</td><td>10.03</td><td>2.44</td><td>11.62</td></tr><tr><td>+MC+DiffAug</td><td>10.26</td><td>2.16</td><td>11.04</td><td>10.10</td><td>2.36</td><td>11.30</td></tr></table></body></html>

Improved Recognizability Metrics We generated 10k (the same setting as the test set) images using the BigGAN backbone. The WA rates are $6 2 . 5 6 \%$ , $5 2 . 0 9 \%$ , and $5 4 . 7 1 \%$ for the real test set, the generated set from Hinge baseline, and the generated set from Hinge $+ \mathbf { M } \mathbf { C }$ , respectively. Our MC method’s images perform closer to the real test set than the baseline’s, showing better distribution matching to the real data in terms of recognizability. The SA rate of our MC

8 Hinge+MC 10 BCE   
7 BCE+MC Hinge y   
6 9   
5 Hinge+MC BCE   
4 BCE+MC Hinge   
3 0 20 40 60 80 100 8 0 20 40 60 80 100 (a)FID↓ (b)IS ↑

method is $8 3 . 4 2 \%$ compared to $9 3 . 6 5 \%$ of the real test set, showing that we generate fairly recognizable fake images.

Qualitative Results The qualitative results are shown in Figure 3 and a figure in Appendix with only a small amount of images (in red boxes) misclassified by our classifier.

![](images/2795986b327e1b2f72659997a377eb85c62f7f3cc809c331e93aefa2eb86f975.jpg)  
Figure 2: Learning curves in terms of (a) Fre´chet Inception Distance and (b) Inception Score along the training on the CIFAR-10 using BigGAN with various loss combinations.   
Figure 3: CIFAR-10 samples generated by the BigGAN backbone trained via Hinge $^ +$ DiffAug $^ +$ MC. Images in each row belong to one of the 10 classes. Images misclassified by ResNet-50 are in red boxes.

Latent Space Analysis The latent space learned by the generator is expected to be continuous and smooth so that small perturbations on the conditional input can lead to smooth and meaningful modifications on the generated output. To explore the latent space, we interpolate between each pair of randomly generated images by linearly interpolating their conditional inputs. The results are shown in Figure 4. Intermediary images between a pair of images from two different classes are shown in each row with their confidence score distributions below. The labels of the two classes are shown on the left and right sides of each row, respectively. Each distribution of the confidence scores is calculated by the bottleneck representation of the ResNet50 classifier with a softened softmax function of temperature 5.0 for normalization. The score bars of the left class and the right class are shown in green and magenta, respectively. The red boxes highlight the images being classified as a third class, while the yellow boxes mark images with nonmonotonic confidence score transitions compared to their adjacent images. In other words, images in both red and yellow boxes are undesirable as they imply that the latent space is less continuous and less smooth. Comparing Figure 4a and 4b, the MC method performs better in the learned latent space, with most decision switches between classes occurring in the mid-range of interpolation.

Table 3: Quantitative results of image generation on CIFAR100 using BigGAN w/o and with our MC method and Differentiable Augmentation.   

<html><body><table><tr><td>Loss</td><td colspan="3">Hinge</td><td colspan="3">BCE</td></tr><tr><td>Metrics</td><td>IS↑</td><td>FID↓</td><td>IFID↓</td><td>IS↑</td><td>FID↓</td><td>IFID↓</td></tr><tr><td>BigGAN</td><td>10.73</td><td>8.31</td><td>83.36</td><td>10.81</td><td>8.37</td><td>81.89</td></tr><tr><td>+DiffAug</td><td>10.72</td><td>7.37</td><td>80.00</td><td>10.71</td><td>7.61</td><td>80.48</td></tr><tr><td>+MC</td><td>11.39</td><td>6.97</td><td>80.20</td><td>11.59</td><td>6.99</td><td>80.91</td></tr><tr><td>+MC+DiffAug</td><td>11.81</td><td>5.77</td><td>76.26</td><td>11.90</td><td>5.85</td><td>77.33</td></tr></table></body></html>

![](images/701cdd014617dab3a41d5d5bce91b40d70d7151b50d6955e51c89f22a760ccb9.jpg)  
Figure 4: Latent space interpolation based on cStyleGAN2 backbone trained via Hinge loss w/o and with our MC method. Red and yellow boxes highlight two types of undesirable transitions between generated images.

Quantitative Results on CIFAR-100 For completeness, we show the image generation performance on CIFAR-100 in Table 3. Significant improvements are achieved by using our MC method independently for both baseline discriminative losses, with an average improvement of 1.1 in IS, 1.6 in FID, and 3.7 in IFID. A detailed sensitive analysis w.r.t the Monte Carlo sample size is provided in the appendix.

Large-Scale and High-Resolution Dataset Results For the FFHQ256 (high-resolution), the lmageNet64 (largescale), and the LSUN bedroom (large-scale and highresolution) dataset, we use the StyleGAN2-ada (Karras et al.

2020a) as backbones. As shown in Table $4 ^ { 2 }$ , MCGAN achieved significant and consistent gains in both FID and IS, as evidenced by $1 6 . 4 \%$ $( 4 . 5 1  3 . 7 7 \$ ), $1 5 . 5 \%$ ( $1 9 . 8 3 $ 16.76), and $3 5 . 7 \%$ ( $4 . 3 4 \ :  \ : 2 . 7 9 )$ FID improvement, respectively, on FFHQ256, ImageNet64, and LSUN bedroom datasets. These improvements are significant and consistent during training periods and across various datasets, demonstrating faster convergence and better generation ability.

<html><body><table><tr><td>Dataset</td><td>Method</td><td>FID↓</td><td>IS↑</td><td>Precision↑</td><td>Recall ↑</td></tr><tr><td>FFHQ256</td><td>original +MC</td><td>4.51 3.77</td><td>5.10 5.25</td><td>0.69 0.69</td><td>0.40 0.45</td></tr><tr><td>ImageNet64</td><td>origima</td><td>19.3</td><td>13.67</td><td>0.65</td><td>0.3</td></tr><tr><td>LSUN bedroom</td><td>original +MC</td><td>4.34 2.79</td><td>2.45 2.45</td><td>0.57 0.61</td><td>0.22 0.23</td></tr></table></body></html>

Table 4: Quantitative results of image generation on largescale and high-resolution datasets using StyleGAN2-ada w/o and with our MC method; FID is 10-run average.

# Conditional Video Generation

The conditional video generation task aims to generate the next frame given the past frames of the videos. Here, we used the Moving MNIST data set (Srivastava, Mansimov, and Salakhudinov 2015), which consists of 10,000 20-frame $6 4 \mathrm { x } 6 4$ videos of moving digits. The whole dataset is divided into the training set (9,000 samples) and the test set (1,000 samples). For the architecture of both the generator and discriminator, we use the convolutional LSTM (ConvLSTM) unit proposed by (Shi et al. 2015) due to its effectiveness in video prediction tasks. In the model training, the generator takes in 5 past frames as the input and generates the corresponding 1-step future frame, then the real past frames and the generated future frames are concatenated along time dimension and put into the discriminator.

For comparison, we used classical GAN as the benchmark. We trained our model for 20,000 epochs with batch size 16. The model performance is evaluated by computing the MSE between the generated frames and the corresponding ground truth on the test set. Numerical results show that our proposed MC method reduces GAN’s MSE from 0.1012 to 0.0840. Compared to the baseline, the predicted frames from our MC method are clearer, more coherent, and visually closer to the ground truth, as shown in Figure 5.

# Conditional Time-Series Generation

Following (Liao et al. 2024), we consider the conditional time-series generation task on two types of datasets (1) $d .$ - dimensional vector auto-regressive (VAR) data and (2) empirical stock data. The goal is to generate 3-step future paths based on the 3-lagged value of time series. We apply the MCGAN to the RCGAN baseline (Esteban, Hyland, and Ra¨tsch 2017) and benchmark it with TimeGAN (Yoon, Jarrett, and Van der Schaar 2019), GMMN (Li, Swersky, and Zemel 2015) and SigWGAN (Liao et al. 2024) as the strong SOTA models for conditional time series generation. The model performance is evaluated using metrics in (Liao et al. 2024) including (1) ABS metric, (2) Correlation metric, (3)ACF metric and (4) $R ^ { 2 }$ error to assess the fitting of synthetic data in terms of marginal distribution, correlation, autocorrelation and usefulness, respectively.

![](images/a16296227838788575232843ee511628d66d209e65629dde5a7009420e874fdc.jpg)  
Figure 5: Results of predicting the next frame given the past 5 frames using ConvLSTM w/o and with our MC method.

VAR Dataset To validate MCGAN for multivariate time series systematically, we use VAR datasets with various path dimensions $d \in [ 1 , 1 0 0 ]$ and various parameter settings. For $d \in \{ 1 , 2 , 3 \}$ , MCGAN consistently outperforms the RCGAN and TimeGAN (see results in Appendix). Figure 6 shows that the MCGAN and SigCWGAN have better fitting than other baselines in terms of conditional law as the estimated mean is closer to that of the ground truth compared with the others for $d = 3$ . Note that SigCWGAN suffers the curse of dimensionality resulting from large $d$ and becomes infeasible for $d \geq 5 0$ , whereas MCGAN does not. In fact, as shown in Table 5, as $d$ increases, the performance gains of MCGAN become more pronounced. With $d = 1 0 0$ , the MC method improves all the metrics by $30 \% \mathrm { - } 4 0 \%$ , further highlighting its effectiveness in high-dimensional settings.

Stock Dataset The stock dataset is a 4-dimensional time series composed of the log return and log volatility data of $\tt S \& P 5 0 0$ and DJI spanning from 2005/01/01 to 2020/01/01. To cover the stylized facts of financial time series like leverage effect and volatility clustering, we also evaluate our generated samples using the ACF metric on the absolute return and squared return. Table 6 demonstrates that our MC method consistently improves the generator performance in terms of temporal dependency, cross-correlation and usefulness. Although RCGAN achieved comparable ABS metrics, it failed to capture the cross-correlation and temporal dependence. Specifically, using our proposed MC method, the correlation metric and ACF metric of RCGAN can be improved from 0.25184 to 0.15687 and from 0.03814 to 0.02905. The gap in the $R ^ { 2 }$ further showcases that our MC method can enhance the generator to generate high-fidelity samples.

![](images/ebfa35551405ac6d57b53ab31a6b6861f9c1e33408546ad4552c22dd9721ae6e.jpg)  
Figure 6: Comparison of models’ performance in fitting the conditional distribution of future time series given one past path sample. The real and generated paths are plotted in red and blue, respectively, with the shaded area as the $9 5 \%$ confidence interval. The synthesized data is VAR(1) for $d = 3$ .

Table 5: Quantitative results of time-series generation on VAR data with different path dimensions $d$ ranging from 10 to 100 using RCGAN w/o and with our MC method.   

<html><body><table><tr><td colspan="2">Loss</td><td colspan="3">Hinge</td><td colspan="3">BCE</td></tr><tr><td>d</td><td>Method</td><td>ABS↓</td><td>Corr↓</td><td>ACF↓</td><td>ABS↓</td><td>Corr↓</td><td>ACF↓</td></tr><tr><td>10</td><td>RCGAN +MC</td><td>0.0180 0.0155</td><td>0.0568 0.0436</td><td>0.0818 0.0651</td><td>0.0153 0.0139</td><td>0.0507 0.0459</td><td>0.0900 0.0719</td></tr><tr><td>50</td><td>RCGAN +MC</td><td>0.0353 0.0286</td><td>0.0700 0.0616</td><td>0.0884 0.0687</td><td>0.0363 0.0250</td><td>0.0710 0.0600</td><td>0.0877 0.0700</td></tr><tr><td>100</td><td>RCGAN +MC</td><td>0.0332 0.0230</td><td>0.0790 0.0498</td><td>0.1102 0.0669</td><td>0.0379 0.0234</td><td>0.0730 0.0502</td><td>0.1022 0.0614</td></tr></table></body></html>

<html><body><table><tr><td>Model</td><td>ABS↓</td><td>ACF↓</td><td>ACF(|x|)↓</td><td>ACF(x²)↓</td><td>Corr↓</td><td>R² (%)↓</td></tr><tr><td>RCGAN MCGAN (ours)</td><td>0.0087 0.0100</td><td>0.0381 0.0291</td><td>0.0788 0.0544</td><td>0.1393 0.0993</td><td>0.2518 0.1569</td><td>4.4968 2.8429</td></tr><tr><td>SigCWGAN GMMN TimeGAN</td><td>0.0096 0.0139 0.0110</td><td>0.0298 0.0599 0.0572</td><td>0.1339 0.2530 0.0690</td><td>0.0846 0.2696 0.1258</td><td>0.1172 0.3184 0.4734</td><td>3.8198 11.8758 4.5396</td></tr></table></body></html>

Table 6: Quantitative results of time-series generation on SPX/DJI data using RNN w/o and with our MC method.

# Conclusion

This paper presents a general MCGAN method to tackle the training instability, a key bottleneck of GANs. Our method enhances generator training by introducing a novel regression loss for (conditional) GANs. We establish the optimality and discriminability of MCGAN, and prove that the convergence of optimal generator can be achieved under a weaker condition of the discriminator due to the strong supervision of the regression loss. Moreover, extensive numerical results on various datasets, including image, time series data, and video data, are provided to validate the effectiveness and flexibility of our proposed MCGAN and consistent improvements over the benchmarking GAN models.

For future work, it is worthwhile to explore the application of MCGAN to enhance state-of-the-art GAN models for more challenging and complex tasks, such as text-to-image generation. Besides, given the flexibility and promising results of the MCGAN on different types of data, it can be effectively applied to generate multi-modality datasets simultaneously. Moreover, MCGAN can be extended to incorporate more advanced discriminative losses, than those used in our numerical study, for further performance improvement.