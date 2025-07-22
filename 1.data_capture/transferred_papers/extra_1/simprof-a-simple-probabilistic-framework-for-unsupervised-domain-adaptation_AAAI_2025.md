# SimProF: A Simple Probabilistic Framework for Unsupervised Domain Adaptation

Mengzhu Wang

Hebei University of Technology, School of Artificial Intelligence wangmz@hebut.edu.cn

# Abstract

Unsupervised domain adaptation (UDA) aims at knowledge transfer from a labeled source domain to an unlabeled target domain. Most UDA techniques achieve this by reducing feature discrepancies between the two domains to learn domain-invariant feature representations. In this paper, we enhance this approach by proposing a simple yet powerful probabilistic framework (SimProF) for UDA to minimize the domain gap between the two domains. SimProF estimates the feature space distribution for each class and generates contrastive pairs by leveraging the shared categories between the source and target domains. The concept behind SimProF is inspired by the observation that normalized features in contrastive learning tend to follow a mixture of von Mises-Fisher (vMF) distributions on the unit sphere. This characteristic allows for the generation of an infinite number of contrastive pairs and facilitates an efficient optimization method using a closed-form expression for the expected contrastive loss. As a result, target semantics can be effectively used to augment source features. To implement this, we create vMF distributions based on the inter-domain feature mean difference for each class. Notably, we derive and minimize an upper bound of the expected loss, which is implicitly achieved through an estimated supervised contrastive learning loss applied to the augmented source distribution. Comprehensive experiments on cross-domain benchmarks confirm the efficacy of the proposed method.

# Introduction

Deep learning models typically exhibit good performance on comparable datasets after being trained on a particular dataset. Nevertheless, these models frequently exhibit markedly worse performance when applied to data from a different distribution. Domain adaptation (Wang et al. 2024c,b,a) offers a way around this problem by allowing a model that was trained on a labeled source domain to adapt to an unlabeled or sparsely labeled target domain.

Generally, solving the domain shift issue is the fundamental challenge to domain adaptation. To tackle this, a substantial subset of current UDA techniques concentrates on aligning the feature distributions in the source and target domains such that a classifier trained on the former can be successfully extended to the latter (Liang, Hu, and Feng 2021). Although these techniques show promising performance, they rely on modifying features using a shared source-supervised classifier. However, this approach carries a substantial risk: the classifier, being supervised by source features, may become biased towards the source domain. As a result, directly applying this classifier to the target domain can limit its generalization ability, especially when cross-domain features are not perfectly aligned. To address this limitation, some studies (Xie et al. 2024; Li et al. 2021b) focus on classifier adaptation, introducing specialized network modules to facilitate the learning of a target-specific classifier.

![](images/aed5710a9f570d7f6946e42c4bcec1fb030c69551f25e2a33e23aa491a7c21c6.jpg)  
Figure 1: Each colored dot represents a class, and the different classes can be effectively separated by using von MisesFisher distributions.

To advance classifier adaptation, we propose a simple probabilistic framework for unsupervised domain adaptation (SimProF). This approach involves semantically augmenting source features toward the target domain, enabling the classifier to adapt to target data within an augmented feature space enriched with target-specific semantics. Our key insight is to directly sample an infinite number of contrastive pairs from the data distribution and then minimize the predicted loss to establish the optimization objective. This approach eliminates the need for large batch sizes by explicitly predicting and minimizing the expected loss. The concept of semantic augmentation draws inspiration from the exceptional ability of deep neural networks to linearize features and disentangle the underlying variations in the data. By modifying features in the deep feature space along certain axes, we can induce significant semantic changes in the original input space (Li et al. 2021a; Wang et al. 2021b). Traditional methods often model unconstrained features using a normal distribution for data augmentation, offering an upper bound on the estimated cross-entropy loss for optimization. However, direct modeling with a normal distribution is impractical in contrastive learning due to the need for feature normalization, and estimating the distribution of each class within small batches is challenging. To address these issues, we model the feature distribution using the von Mises-Fisher distribution on the unit sphere (see Figure. 1), which extends the normal distribution to the hypersphere. By adding directions sampled from this distribution to each source data point, we create a new target-like domain. This augmentation preserves the class identity of the data, allowing us to monitor the classifier’s training within the newly generated target-like domain. This method bypasses the need to directly sample a large number of contrastive pairs by instead focusing on minimizing a surrogate loss function, which can be efficiently optimized without adding complexity during inference.

The following is a summary of the SimProF’s main contributions:

• For unsupervised domain adaptation, we provide SimProF, a novel probabilistic contrastive learning technique. SimProF effectively estimates parameters across batches by modeling feature distributions using the von Mises-Fisher (vMF) distribution. This strategy directly samples contrastive pairs from the predicted distribution, effectively overcoming the inherent limitation of contrastive learning, which typically requires large batch sizes. • We minimize the upper bound loss in the limiting case to obtain a closed-form solution. It is possible to efficiently optimize the resulting surrogate loss function without increasing any additional overhead during inference. • We investigate the efficacy of the suggested SimProF method on a number of datasets, including VisDA-2017, Office-31, Office-Home, and DomainNet.

# Method

# Motivation and Preliminaries

This work explores the use of von Mises-Fisher (vMF) distributions on unit space for unsupervised domain adaptation (UDA), an area that has not been extensively investigated. We propose a straightforward probabilistic framework for UDA, termed SimProF, which captures the semantics of the target domain while enhancing discriminability. SimProF is inspired by prior research demonstrating that deep neural networks can effectively manipulate features along specific directions (Wang et al. 2019b) in the deep feature space to alter semantics, given their ability to disentangle underlying data variations. Figure. 2 provides an overview of SimProF, with further details described below.

Unsupervised domain adaptation aims to transfer models trained on a labeled source domain to an unlabeled target domain with a differing data distribution which depends on having access to both unlabeled data from the target domain and all labeled samples from the source domain during training. Formally, let $D _ { s } = ( { \boldsymbol { \chi } } _ { s } , { \boldsymbol { \ y } } _ { s } ) = \{ ( x _ { s i } , y _ { s i } ) \} _ { i = 1 } ^ { N _ { s } }$ represents a fully labeled source domain with $N _ { s }$ image-label pairs, and Dt = Xt = {xti}iN=t1 denote an unlabeled dataset from the target domain with $N _ { t }$ images. Both $\{ x _ { s i } \}$ and $\{ x _ { t i } \}$ are drawn from the same set of $M$ predefined categories.

# Contrastive Learning (CL)

In CL (Khosla et al. 2020), a feature extractor $F$ is trained to discriminate between negative pairings $( x _ { i } , x _ { j } )$ with distinct labels $y _ { i } \neq y _ { j }$ and positive pairs $( x _ { i } , x _ { j } )$ with the same label $y _ { i } = y _ { j }$ . In embedding space, clusters of points from the same class are drawn together, while clusters of samples from different classes are pushed apart. Given every batch of sample-label pairings $\boldsymbol { B } = \{ ( x _ { i } , y _ { i } ) _ { i = 1 } ^ { N ^ { B } } \}$ and a temperature parameter $\tau$ , there are two standard methods to construct the CL loss:

$$
\mathcal { L } _ { i n } ( \pmb { x } _ { i } , y _ { i } ) = - \log \Bigg \{ \frac { 1 } { N _ { y _ { i } } ^ { B } } \sum _ { p \in P ( y _ { i } ) } \frac { e ^ { \pmb { x } _ { i } \cdot \pmb { x } _ { p } / \tau } } { \displaystyle \sum _ { j = 1 } ^ { M } \sum _ { b \in A ( j ) } e ^ { \pmb { x } _ { i } \cdot \pmb { x } _ { a } / \tau } } \Bigg \} ,
$$

$$
\mathcal { L } _ { o u t } ( \pmb { x } _ { i } , y _ { i } ) = \frac { - 1 } { N _ { y _ { i } } ^ { B } } \sum _ { p \in A ( y _ { i } ) } \log \frac { e ^ { \pmb { x } _ { i } \cdot \pmb { x } _ { p } / \tau } } { \displaystyle \sum _ { j = 1 } ^ { M } \sum _ { b \in A ( j ) } e ^ { \pmb { x } _ { i } \cdot \pmb { x } _ { a } / \tau } } ,
$$

where $P ( y _ { i } ) = \{ b \in A ( j ) : y _ { p } = y _ { j } \}$ is the set of indices of all positives in the multi-viewed batch distinct from $i$ , and $N _ { y _ { i } } ^ { B } = | A ( y _ { i } ) |$ is the cardinality. $\scriptstyle { \mathbf { { \vec { x } } } }$ denotes the normalized features of $x$ extracted by $F$ :

$$
\pmb { x } _ { i } = \frac { F ( \pmb { x } _ { i } ) } { \| F ( \pmb { x } _ { i } ) \| } , \quad \pmb { x } _ { p } = \frac { F ( \pmb { x } _ { p } ) } { \| F ( \pmb { x } _ { p } ) \| } , \quad \pmb { x } _ { a } = \frac { F ( \pmb { x } _ { a } ) } { \| F ( \pmb { x } _ { a } ) \| } .
$$

Furthermore, with respect to the position of the log function, the values $\mathcal { L } _ { o u t }$ and $\mathcal { L } _ { i n }$ represent the sums over the positive pairs. The two loss formulations are not equivalent, as shown in (Khosla et al. 2020), and $\mathcal { L } _ { i n } \le \mathcal { L } _ { o u t }$ , according to Jensen’s inequality (Jensen 1906). Since $\mathcal { L } _ { o u t }$ offers an upper bound for $\mathcal { L } _ { i n }$ , CL utilizes it as the loss function.

# von Mises–Fisher (vMF) Distribution

Features in contrastive learning are restricted to lie on the unit hypersphere, as was previously mentioned. Since vMF distributions extend the normal distribution to hyperspherical spaces, we utilize a mixture of von Mises–Fisher (vMF) distributions (Mardia, Jupp, and Mardia 2000) to simulate these properties. For a random $p$ -dimensional unit vector $z$ , the probability density function of the vMF distribution can be written as follows:

![](images/1c812c997363fede51fa8f83c3413f2b4dcf13a8d719e7d86e56ce445e0c458f.jpg)  
Figure 2: An example of SimProF is as follows: SimProF creates contrastive pairs from the sample distribution by estimating it using characteristics from different batches. By sampling an infinite number of contrastive pairs, it also obtains a closed-form expression for the predicted contrastive loss. This method overcomes contrastive learning’s intrinsic drawback with respect to high batch sizes.

$$
\begin{array} { c } { { f _ { p } ( x ; \mu , \kappa _ { y } ) = \displaystyle \frac { 1 } { C _ { p } ( \kappa _ { y } ) } e ^ { \kappa _ { y } \mu ^ { \top } x } , } } \\ { { C _ { p } ( \kappa _ { y } ) = \displaystyle \frac { ( 2 \pi ) ^ { p / 2 } I _ { ( p / 2 - 1 ) } ( \kappa _ { y } ) } { \kappa _ { y } ^ { p / 2 - 1 } } , } } \end{array}
$$

where $I _ { ( p / 2 - 1 ) }$ , $\kappa \geq 0 , \| \mu \| _ { 2 } = 1$ , and $x$ are $p$ -dimensional unit vectors. indicates the first-kind modified Bessel function at order $p / 2 - 1$ . It has the following definition:

$$
I _ { ( p / 2 - 1 ) } ( { \pmb x } ) = \sum _ { k = 0 } ^ { \infty } \frac { 1 } { k ! \Gamma ( p / 2 - 1 + k + 1 ) } ( \frac { { \pmb x } } { 2 } ) ^ { 2 k + p / 2 - 1 } .
$$

The concentration parameter and the mean direction are denoted by the parameters $\kappa _ { y }$ and $\mu$ , respectively. The distribution becomes more concentrated around the mean direction $\mu$ as $\kappa _ { y }$ grows. On the other hand, the distribution becomes uniform throughout the sphere when $\kappa _ { y } = 0$ .

Based on the assumption above, we model the feature distribution using a mixture of vMF distributions:

$$
P ( \pmb { x } ) = \sum _ { y = 1 } ^ { M } P ( y ) P ( \pmb { x } | y ) = \sum _ { y = 1 } ^ { M } \pi _ { y } \frac { \kappa _ { y } ^ { p / 2 - 1 } e ^ { \kappa _ { y } \mu _ { y } ^ { \top } \pmb { x } } } { ( 2 \pi ) ^ { p / 2 } I _ { p / 2 - 1 } ( \kappa _ { y } ) } ,
$$

where $\pi _ { y }$ , which represents the probability of a class $y$ , indicates how frequently class $y$ occurs in the training dataset. Next, we use maximum likelihood estimation to estimate the mean vector $\mu _ { y }$ and the concentration parameter $\kappa _ { y }$ for the feature distribution.

Maximum Likelihood Estimation. To get the mean vector $\mu _ { y }$ and the concentration parameter $\kappa _ { y }$ , we perform maximum likelihood estimation on $( \kappa _ { y } , \mu _ { y } )$ as follows:

$$
\begin{array} { l } { { \displaystyle { \cal L } ( \mu _ { y } ; p , \kappa _ { y } , \lambda ) = n \log \left( C _ { p } ( \kappa _ { y } ) \right) } \ ~ } \\ { { \displaystyle ~ + \kappa _ { y } \mu _ { y } ^ { \top } \sum _ { i = 1 } ^ { n } { \bf x } _ { i } + \lambda ( \| \mu _ { y } \| - 1 ) } , } \end{array}
$$

Based on the maximum likelihood estimation from Eq. 7, the partial derivative of $L$ can be expressed as follows:

$$
\begin{array} { r l } { \displaystyle \frac { \partial L } { \partial \mu _ { y } } = \kappa _ { y } \sum _ { i = 1 } ^ { n } \pmb { x } _ { i } + \lambda \frac { \mu _ { y } } { \| \mu _ { y } \| } } & { { } = 0 , } \\ { \displaystyle \frac { \partial L } { \partial \kappa _ { y } } = n \frac { C _ { p } ^ { \prime } ( \kappa _ { y } ) } { C _ { p } ( \kappa _ { y } ) } + \mu _ { y } \top \sum _ { i = 1 } ^ { n } \pmb { x } _ { i } = 0 , } \\ { \displaystyle \frac { \partial L } { \partial \lambda } = \| \mu _ { y } \| - 1 } & { { } = 0 , } \end{array}
$$

Due to $\begin{array} { r } { C _ { p } ( \kappa _ { y } ) = \frac { \kappa _ { y } ^ { \frac { p } { 2 } - 1 } } { ( 2 \pi ) ^ { \frac { p } { 2 } } I _ { \frac { p } { 2 } - 1 } ( \kappa _ { y } ) } = ( 2 \pi ) ^ { - \frac { p } { 2 } } \frac { \kappa _ { y } ^ { \frac { p } { 2 } - 1 } } { I _ { \frac { p } { 2 } - 1 } ( \kappa _ { y } ) } } \end{array}$ hen from the Eq. 9, $C _ { p } ^ { \prime } ( \kappa _ { y } )$ is given by:

$$
C _ { p } ^ { \prime } ( \kappa _ { y } ) = ( 2 \pi ) ^ { - \frac { p } { 2 } } \frac { \left( \frac { p } { 2 } - 1 \right) \kappa _ { y } ^ { \frac { p } { 2 } - 2 } I _ { \frac { p } { 2 } - 1 } ( \kappa _ { y } ) - \kappa _ { y } ^ { \frac { p } { 2 } - 1 } I _ { \frac { p } { 2 } - 1 } ^ { \prime } ( \kappa _ { y } ) } { I _ { \frac { p } { 2 } - 1 } ^ { 2 } ( \kappa _ { y } ) }
$$

According to the results of $C _ { p } ( \kappa _ { y } )$ and $C _ { p } ^ { \prime } ( \kappa _ { y } )$ , the $\frac { C _ { p } ^ { \prime } ( \kappa _ { y } ) } { C _ { p } ( \kappa _ { y } ) }$ is derived as follows:

$$
\begin{array} { r l } & { \frac { C _ { \phi } ^ { \nu } ( k _ { \phi } ) ^ { 2 } } { C _ { \phi } ^ { \nu } ( k _ { \phi } ) } = \frac { ( 2 \pi ) ^ { - \frac { \nu } { \lambda } \frac { \nu } { \lambda } \frac { \lambda } { \xi } - 1 ( k _ { \phi } ) } - \frac { \lambda } { \eta _ { \phi } ^ { \frac { \nu } { \lambda } } - 1 ( k _ { \phi } ) - \frac { \nu } { \lambda } \frac { \nu } { \lambda } } \frac { \lambda } { \eta _ { \phi } ^ { \frac { \nu } { \lambda } } - 1 ( k _ { \phi } ) } } { ( 2 \pi ) ^ { - \frac { \nu } { \lambda } } \frac { \nu } { \lambda } \frac { \lambda } { \xi } - 1 ( k _ { \phi } ) } } \\ & { = \frac { ( \frac { \nu } { \lambda } - 1 ) ^ { \frac { \nu } { \lambda } } - 2 ( k _ { \phi } ) - ( \nu _ { \phi } ) ^ { \frac { \nu } { \lambda } } - 1 ( k _ { \phi } ) } { \pi _ { \xi } ^ { \frac { \nu } { \lambda } } - 1 ( k _ { \xi } ) } } \\ & { = \frac { ( \frac { \nu } { \lambda } - 1 ) ^ { \frac { \nu } { \lambda } } \frac { \nu } { \lambda } \frac { \nu } { \lambda } \frac { \nu } { \lambda } - 1 ( k _ { \phi } ) - ( \nu _ { \phi } ) ^ { \frac { \nu } { \lambda } } - 1 ( k _ { \xi } ) } { ( 2 \pi ) ^ { - \frac { \nu } { \lambda } } \frac { \nu } { \lambda } \frac { \nu } { \lambda } } } \\ & { = \frac { ( \frac { \nu } { \lambda } - 1 ) ^ { \frac { \nu } { \lambda } } \frac { \nu } { \lambda } \frac { \nu } { \lambda } \frac { \nu } { \lambda } - 1 ( k _ { \phi } ) } { \pi _ { \xi } ^ { \frac { \nu } { \lambda } } - 1 ( k _ { \xi } ) } - \frac { L _ { \xi } } { \lambda } ( k _ { \xi } ) } \\ & { = \frac { ( \frac { \nu } { \lambda } - 1 ) ^ { \frac { \nu } { \lambda } } k _ { \phi } ^ { - 1 } \frac { L _ { \xi } } { \lambda } - 1 ( k _ { \xi } ) - \big [ L _ { \xi } ( k _ { \xi } ) + ( \frac { \nu } { 2 } - 1 ) \xi _ { \phi } ^ { - 1 } \frac { L _ { \xi } } { \lambda } - 1 ( k _ { \xi } ) \big ] } { \frac { L _ { \xi } } { \frac { \nu } { \lambda } - 1 ( k _ { \xi } ) } } } \\ &  = - \frac { \Gamma _ { \xi } ( k _ { \xi } ) }  \pi _ { \xi } ^  \frac   \end{array}
$$

Suppose that $\begin{array} { r } { A _ { p } ( \kappa _ { y } ) = - \frac { C _ { p } ^ { \prime } ( \kappa _ { y } ) } { C _ { p } ( \kappa _ { y } ) } = \frac { I _ { \frac { p } { 2 } } ( \kappa _ { y } ) } { I _ { \frac { p } { 2 } - 1 } ( \kappa _ { y } ) } } \end{array}$ , then the ， maximum likelihood estimates of the mean direction $\mu _ { y }$ satisfy the following equations:

$$
\left( \kappa _ { y } \sum _ { i = 1 } ^ { n } x _ { i } \right) ^ { \top } \left( \kappa _ { y } \sum _ { i = 1 } ^ { n } x _ { i } \right) = ( - \lambda \mu _ { y } ) ^ { \top } ( - \lambda \mu _ { y } )
$$

$$
\begin{array} { r l r } {  { n \kappa _ { y } \frac { C _ { p } ^ { \prime } ( \kappa _ { y } ) } { C _ { p } ( \kappa _ { y } ) } + \kappa _ { y } \pmb { \mu } _ { y } ^ { \top } \sum _ { i = 1 } ^ { n } \pmb { x } _ { i } } } \\ & { } & \\ & { } & { = \kappa _ { y } \mu _ { y } ^ { \top } \sum _ { i = 1 } ^ { n } x _ { i } + n \kappa _ { y } \frac { C _ { p } ^ { \prime } ( \kappa _ { y } ) } { C _ { p } ( \kappa _ { y } ) } } \\ & { } & \\ & { } & { = 0 } \end{array}
$$

$$
\left. \begin{array} { c } { \displaystyle { A _ { p } ( \kappa _ { y } ) = \frac { 1 } { n } \left\| \sum _ { i = 1 } ^ { n } \pmb { x } _ { i } \right\| } } \\ { \displaystyle = \| \overline { { \boldsymbol { x } } } \| } \end{array} \right.
$$

where nκy Cp(κ ) $\begin{array} { r } { n \kappa _ { y } \frac { C _ { p } ^ { \prime } ( \kappa _ { y } ) } { C _ { p } ( \kappa _ { y } ) } = \lambda } \end{array}$ and ${ \bar { R } } = \| { \bar { x } } \|$ is the length of sample mean. A simple approximation (Wang et al. 2021a) to $\kappa _ { y }$ is:

$$
\hat { \kappa } _ { y } = \frac { \bar { R } ( p - \bar { R } ^ { 2 } ) } { 1 - \bar { R } ^ { 2 } } .
$$

Furthermore, an online method is employed to estimate the sample mean for each class by incorporating data from both the current and previous mini-batches. Specifically, we utilize the online estimation algorithm outlined below to continuously update the sample mean, starting from zero initialization at the beginning of the current epoch. The sample mean estimated from the previous epoch is then used for maximum likelihood estimation.

# Implicit Augmentation by vMF distribution

On the unit hypersphere $S ^ { p - 1 }$ , consider a set of $N$ independent unit vectors $( x _ { i } ) _ { i } ^ { N }$ sampled from a von Mises–Fisher (vMF) (Sra 2012; Mardia and Jupp 2009) distribution corresponding to class $y$ . The following equations provide the maximum likelihood estimates for the concentration parameter $\kappa _ { y }$ and the mean direction $\mu _ { y }$ :

$$
\begin{array} { c } { \displaystyle \mu _ { y } = \bar { x } / \bar { R } , } \\ { \displaystyle A _ { p } ( \kappa _ { y } ) = \frac { I _ { p / 2 } ( \kappa _ { y } ) } { I _ { p / 2 - 1 } ( \kappa _ { y } ) } = \bar { R } , } \end{array}
$$

where the sample mean is represented by $\begin{array} { r } { \bar { x } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } x _ { i } } \end{array}$ and the sample mean vector’s length is indicated by $\bar { R } =$ $| \bar { x } | _ { 2 } . \kappa _ { y }$ has an approximation expression that is as follows:

$$
\hat { \kappa } _ { y } = \frac { \bar { R } ( p - \bar { R } ^ { 2 } ) } { 1 - \bar { R } ^ { 2 } } .
$$

Additionally, by combining data from the previous and current mini-batches, a real-time estimate of the sample mean for each class is computed. More specifically, we use the sample mean estimated from the previous epoch for maximum likelihood estimation and update the sample mean from zero initialization in the current epoch using the online estimation algorithm below:

$$
\bar { x } _ { j } ^ { ( t ) } = \frac { n _ { j } ^ { ( t - 1 ) } \bar { x } _ { j } ^ { ( t - 1 ) } + m _ { j } ^ { ( t ) } \bar { x } _ { j } ^ { \prime ( t ) } } { n _ { j } ^ { ( t - 1 ) } + m _ { j } ^ { ( t ) } } ,
$$

where x¯′j(t $\bar { x } _ { j } ^ { \prime ( t ) }$ represents the sample mean of class $j$ in the current mini-batch, and $\bar { x } _ { j \_ j } ^ { ( t ) }$ indicates the estimated sample mean of class $j$ at step $t$ . The number of samples from the prior mini-batches and the current mini-batch are indicated by the variables n(jt−1) and m(j , respectively.

# Semantic Directions for Domain Augmentation

Every class has an average feature representation that can be thought of as its semantic prototype, from which all samples are created. For example, take a bicycle. Variants of this prototype provide particular samples, but the semantic prototype of the bicycle may comprise essential structural components like wheels, handlebars, and a seat. The average representations, or semantic prototypes, of source and target samples for the same class frequently vary as a result of domain shift. To fix this prototype mismatch, we translate each source feature along the direction given by the prototype vector corresponding to its class. More specifically, $\mu _ { s } ^ { c }$ and $\mu _ { t } ^ { c }$ represent the estimated mean feature for class $c$ in the source and target domains, respectively.

A straightforward method that builds on the predicted parameters would be to select contrastive pairs from the mixture of vMF distributions. During each training iteration, directly sampling a large number of data points from these distributions might not be as beneficial. We overcome this by expanding the number of samples to infinity and using mathematical analysis to obtain a closed-form formula for the anticipated contrastive loss function.

Proposition 1. Assuming the parameters of the mixture of vMF distributions are $\pi _ { y } , \mu _ { y }$ , and $\kappa _ { y }$ for $y = 1 , \ldots , M$ and letting the sample size $N$ approach infinity, the projected contrastive loss function can be expressed as follows:

$$
\mathcal { L } _ { \mathrm { o u t } } ( \pmb { x _ { i } } , y _ { i } ) = \frac { - \pmb { x _ { i } } \cdot \pmb { A _ { p } } ( \kappa _ { y _ { i } } ) \mu _ { y _ { i } } } { \tau } + \log \bigg ( \sum _ { j = 1 } ^ { M } \pi _ { j } \frac { C _ { p } ( \tilde { \kappa } _ { j } ) } { C _ { p } ( \kappa _ { j } ) } \bigg ) ,
$$

$$
\mathcal { L } _ { \mathrm { i n } } ( \boldsymbol { x } _ { i } , y _ { i } ) = - \log \bigg ( \pi _ { y _ { i } } \frac { C _ { p } ( \tilde { \kappa } _ { y _ { i } } ) } { C _ { p } ( \kappa _ { y _ { i } } ) } \bigg ) + \log \bigg ( \sum _ { j = 1 } ^ { M } \pi _ { j } \frac { C _ { p } ( \tilde { \kappa } _ { j } ) } { C _ { p } ( \kappa _ { j } ) } \bigg ) ,
$$

$$
\begin{array} { r } { w h e r e \tilde { x } _ { j } \sim \mathrm { v M F } ( \mu _ { j } , \kappa _ { j } ) , \tilde { \kappa } _ { j } = | | \kappa _ { j } \mu _ { j } + x _ { i } / \tau | | _ { 2 } . } \end{array}
$$

Proof. As per Eq. 2, which defines supervised contrastive loss, we obtain

Table 1: Recognition accuracy $( \% )$ using the Office-31 dataset.   

<html><body><table><tr><td>Methods</td><td>D→A</td><td>W→A</td><td>A→W</td><td>D→W</td><td>W→D</td><td>A→D</td><td>Avg.</td></tr><tr><td>ResNet-50</td><td>62.5</td><td>60.7</td><td>68.4</td><td>96.7</td><td>99.3</td><td>68.9</td><td>76.1</td></tr><tr><td>SimNet</td><td>73.4</td><td>71.6</td><td>88.6</td><td>98.2</td><td>99.7</td><td>85.3</td><td>86.2</td></tr><tr><td>CyCADA</td><td>72.8</td><td>71.4</td><td>89.5</td><td>97.9</td><td>99.8</td><td>87.7</td><td>86.5</td></tr><tr><td>CDAN</td><td>71.0</td><td>69.3</td><td>94.1</td><td>98.6</td><td>100.0</td><td>92.9</td><td>87.7</td></tr><tr><td>TADA</td><td>72.9</td><td>73.0</td><td>94.3</td><td>98.7</td><td>99.8</td><td>91.6</td><td>88.4</td></tr><tr><td>BSP</td><td>73.6</td><td>72.6</td><td>93.3</td><td>98.2</td><td>100.0</td><td>93.0</td><td>88.5</td></tr><tr><td>T²SA</td><td>78.2</td><td>78.5</td><td>94.6</td><td>97.2</td><td>99.8</td><td>92.4</td><td>90.1</td></tr><tr><td>SimProF</td><td>81.7</td><td>84.5</td><td>97.2</td><td>100.0</td><td>100.0</td><td>97.3</td><td>93.5</td></tr></table></body></html>

<html><body><table><tr><td>Method</td><td>Ar→C1</td><td>Ar-→Pr</td><td>Ar→Rw</td><td></td><td></td><td>Cl-AI Cl-Pr CI→RW Pr-→AI Pr→Cl Pr-→RW</td><td></td><td></td><td></td><td>Rw→Ar</td><td>Rw→Cl</td><td>Rw→Pr</td><td>Avg.</td></tr><tr><td>ResNet-50</td><td>34.9</td><td>50.0</td><td>58.0</td><td>37.4</td><td>41.9</td><td>46.2</td><td>38.5</td><td>31.2</td><td>60.4</td><td>53.9</td><td>41.2</td><td>59.9</td><td>46.1</td></tr><tr><td>CDAN</td><td>50.7</td><td>70.6</td><td>76.0</td><td>57.6</td><td>70.0</td><td>70.0</td><td>57.4</td><td>50.9</td><td>77.3</td><td>70.9</td><td>56.7</td><td>81.6</td><td>65.8</td></tr><tr><td>BSP</td><td>52.0</td><td>68.6</td><td>76.1</td><td>58.0</td><td>70.3</td><td>70.2</td><td>58.6</td><td>50.2</td><td>77.6</td><td>72.2</td><td>59.3</td><td>81.9</td><td>66.3</td></tr><tr><td>SAFN</td><td>52.0</td><td>71.7</td><td>76.3</td><td>64.2</td><td>69.9</td><td>71.9</td><td>63.7</td><td>51.4</td><td>77.1</td><td>70.9</td><td>57.1</td><td>81.5</td><td>67.3</td></tr><tr><td>TADA</td><td>53.1</td><td>72.3</td><td>77.2</td><td>59.1</td><td>71.2</td><td>72.1</td><td>59.7</td><td>53.1</td><td>78.4</td><td>72.4</td><td>60.0</td><td>82.9</td><td>67.6</td></tr><tr><td>SymNet</td><td>47.7</td><td>72.9</td><td>78.5</td><td>64.2</td><td>71.3</td><td>74.2</td><td>64.2</td><td>48.8</td><td>79.5</td><td>74.5</td><td>52.6</td><td>82.7</td><td>67.6</td></tr><tr><td>T²SA</td><td>61.0</td><td>78.5</td><td>83.1</td><td>71.4</td><td>80.1</td><td>79.9</td><td>70.3</td><td>60.1</td><td>83.5</td><td>75.6</td><td>62.6</td><td>86.6</td><td>74.4</td></tr><tr><td>SimProF</td><td>61.9</td><td>78.9</td><td>84.5</td><td>72.9</td><td>78.4</td><td>81.1</td><td>72.3</td><td>62.2</td><td>84.8</td><td>81.9</td><td>69.4</td><td>92.2</td><td>76.7</td></tr></table></body></html>

Table 2: Recognition accuracy $( \% )$ using the Office-Home dataset.

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { o u t } = \frac { - 1 } { N _ { y _ { i } } } \sum _ { p \in A ( y _ { i } ) } { \pmb x } _ { i } \cdot { \pmb x } _ { p } / \tau } \\ { \displaystyle \qquad + \log \left( \sum _ { j = 1 } ^ { M } N \frac { N _ { j } } { N } \frac { 1 } { N _ { j } } \sum _ { a \in A ( j ) } e ^ { { \pmb x } _ { i } \cdot { \pmb x } _ { a } / \tau } \right) , } \end{array}
$$

where $N _ { j }$ represents the sampling number of class $j$ and $\mathrm { l i m } _ { N \to \infty } \mathrm { \bar { \it { N } } } _ { j } / N = \pi _ { j }$ is satisfied.

The loss function is as follows once $N  \infty$ and the constant term $\log N$ are eliminated:

$$
\begin{array} { c } { { \mathcal { L } _ { o u t } = \displaystyle \frac { - { \pmb x } _ { i } \cdot \mathbb { E } [ { \tilde { \pmb x } } _ { y _ { i } } ] } { \tau } + \log \left( \displaystyle \sum _ { j = 1 } ^ { M } \pi _ { j } \mathbb { E } [ e ^ { { \pmb x } _ { i } \cdot { \tilde { \pmb x } } _ { j } / \tau } ] \right) } } \\ { { = \displaystyle \frac { - { \pmb x } _ { i } \cdot { A } _ { p } ( \kappa _ { y _ { i } } ) { \pmb \mu } _ { y _ { i } } } { \tau } + \log \left( \displaystyle \sum _ { j = 1 } ^ { M } \pi _ { j } \frac { C _ { p } ( \tilde { \kappa } _ { j } ) } { C _ { p } ( \kappa _ { j } ) } \right) . } } \end{array}
$$

By utilizing the expectation and moment-generating function of the vMF distribution, Eq. 25 is obtained:

$$
\begin{array} { c } { \displaystyle \mathbb { E } \left( \pmb { x } \right) = A _ { p } ( \kappa ) \mu , A _ { p } ( \kappa ) = \frac { I _ { p / 2 } ( \kappa ) } { I _ { p / 2 - 1 } ( \kappa ) } , } \\ { \displaystyle \mathbb { E } \left( e ^ { t ^ { \operatorname { T } } \pmb { x } } \right) = \frac { C _ { p } ( \tilde { \kappa } ) } { C _ { p } ( \kappa ) } , \tilde { \kappa } = \vert \vert \kappa \mu + t \vert \vert _ { 2 } . } \end{array}
$$

We can derive the other loss function from Eq. 1 in a manner similar to that of Eq. 21.

Table 3: Recognition accuracy $( \% )$ using the VisDA-2017 dataset.   

<html><body><table><tr><td>Methods</td><td>Synthetic →Real</td></tr><tr><td>DAN</td><td>53.0</td></tr><tr><td>DANN</td><td>57.4</td></tr><tr><td>SimNet</td><td>69.6</td></tr><tr><td>CDAN</td><td>70.0</td></tr><tr><td>MCD</td><td>73.7</td></tr><tr><td>DSAN</td><td>75.1</td></tr><tr><td>T²SA</td><td>83.2</td></tr><tr><td>CDAN+InterBN</td><td>87.9</td></tr></table></body></html>

$$
\mathcal { L } _ { i n } = - \log \Bigg ( \pi _ { y _ { i } } \frac { C _ { p } ( \tilde { \kappa } _ { y _ { i } } ) } { C _ { p } ( \kappa _ { y _ { i } } ) } \Bigg ) + \log \Bigg ( \sum _ { j = 1 } ^ { K } \pi _ { j } \frac { C _ { p } ( \tilde { \kappa } _ { j } ) } { C _ { p } ( \kappa _ { j } ) } \Bigg ) .
$$

# Overall Formulation

Mutual information $I ( X ; Y )$ (Csisza´r, Shields et al. 2004) measures the degree of dependency between two random variables, $X$ and $Y$ , in information theory. Strong correlations between target features and predictions improve our semantic augmentations by guaranteeing that important semantics relevant to predictions are captured in the extracted features, instead of insignificant information. Therefore, by reducing the loss function provided in Eq. 29, we use mutual information maximization for the goal data.

<html><body><table><tr><td>CDAN clp inf pnt qdr rel</td><td>skt Avg</td><td>T²SA clp inf</td><td></td><td></td><td>pnt qdr rel skt Avg.SimProFclp inf pnt qdr rel skt Avg.</td></tr><tr><td>clp</td><td>-20.436.69.050.742.331.8</td><td>clp</td><td>20.444.516.061.747.438.0</td><td>clp</td><td>-22.346.317.863.549.739.9</td></tr><tr><td>inf</td><td>27.5 - 25.71.834.72</td><td></td><td>43.3 8.8 56.535.637.4</td><td>inf</td><td>45.2- 46.1 9.9 58.337.939.5</td></tr><tr><td>pnt</td><td>42.620.0 - 2.555.638.531.8</td><td>pnt</td><td>50.521.3 - 8.8 63.345.137.8</td><td>pnt</td><td>51.822.5 - 9.6 64.546.839.0</td></tr><tr><td>qdr</td><td>21.0 4.5 8.1 - 14.315.7</td><td>di</td><td>35.38.2 17.3 - 28.824.922.9</td><td>qdr</td><td>36.48.6 19.7- 31.226.124.4</td></tr><tr><td>rel</td><td>51.923.350.45.4  - 41.434.5</td><td>rel</td><td>57.223.653.9 9.7 - 44.337.8</td><td>rel</td><td>59.424.555.210.6- 46.839.3</td></tr><tr><td>skt</td><td>50.820.343.02.950.8- 33.6</td><td>skt</td><td>60.421.250.416.761.4  - 42.0</td><td>skt</td><td>62.423.751.318.862.5-  43.7</td></tr><tr><td>Avg.</td><td>38.817.732.84.341.231.627.7</td><td>Avg.</td><td>49.219.041.912.054.339.536.0</td><td>Avg.</td><td>51.0 20.3 43.713.356.041.5 37.6</td></tr></table></body></html>

![](images/685fb00d850c4a840174a95eab72335da913aef35cc78d9f7c4e4e2174d005b1.jpg)  
Table 4: Recognition accuracy $( \% )$ using the DomainNet dataset.   
Figure 3: For the task $\mathbf { C } {  } \mathbf { P }$ in ImageCLEF-DA, the confusion matrices of the target samples obtained from various method are displayed. (For optimal clarity, please magnify the image.)

$$
\mathcal { L } _ { M I } = \sum _ { c = 1 } ^ { C } \hat { P } ^ { c } \log \hat { P } ^ { c } - \frac { 1 } { n _ { t } } \sum _ { j = 1 } ^ { n _ { t } } \sum _ { c = 1 } ^ { C } P _ { t j } ^ { c } \log P _ { t j } ^ { c } ,
$$

The expression $\begin{array} { r } { \hat { P } = \frac { 1 } { n _ { t } } \sum _ { j = 1 } ^ { n _ { t } } P _ { t j } } \end{array}$ is used. We use the average of the target predictions to approximate the ground-truth distribution because the target domain is unlabeled.

In conclusion, the general goal function of SimProF is:

$$
\mathcal { L } _ { S i m P r o F } = \mathcal { L } _ { i n } + \beta \mathcal { L } _ { M I } ,
$$

where the trade-off parameter is $\beta$ . Experiments will be conducted to thoroughly investigate the effects of various SimProF components.

# Experiments

# Datasets

Office-31 (Saenko et al. 2010) is a well-known crossdomain dataset used in office environments, consisting of images across 31 classes from three distinct domains: Amazon (A), Webcam (W), and DSLR (D). The dataset is imbalanced, with 2,817 images from Amazon, 795 images from Webcam, and 498 images from DSLR.

Office-Home (Venkateswara et al. 2017) is a collection of 15,500 photos from both home and business settings, organized into 65 categories. The dataset consists of four distinct domains: Product $( \mathrm { P r } )$ , Clipart (Cl), Real-World (Rw), and Art (Ar). Real-world camera catches, product photos, clipart images, and artistic renderings are included in these categories in that order.

VisDA-2017 (Peng et al. 2017) comprises more than 280,000 photos in 12 categories and presents a difficult benchmark for unsupervised domain adaptation (UDA). We employ the validation set as the target domain and the training set as the source domain.

DomainNet (Peng et al. 2019) comprise over 590,000 images in six domains Painting (pnt), Infograph (inf), Real (rel), Quickdraw (qdr), (Clipart (clp) and Sketch (skt)), is the largest image benchmark for domain adaptation.

# Implementation Details

We use PyTorch to implement our technique, and the backbone network for all datasets is ResNet (He et al. 2016), which has been pre-trained on ImageNet (Russakovsky et al. 2015). We used Pytorch (Paszke et al. 2019) to implement all of the experiments. Our Stochastic Gradient Descent (SGD) algorithm is used with momentum set to 0.9 and weight decay set to 0.001. For model optimization, we follow the learning rate annealing technique described in (Ganin et al. 2016). In this study, we specifically specify $\beta ~ = ~ 0 . 5$ and perform a sensitivity analysis to find out how hyper-parameter selection affects the results. We present the average accuracy from three random trials for each task. We compared with ResNet-50 (He et al. 2016), DDC (Tzeng et al. 2014), DAN (Long et al.

![](images/ce94a3acf268edac6e68cdde6f4a33bf3379b3328485be6b0006c9e336d9dea2.jpg)  
Figure 4: (a): $\mathcal { A }$ -distance of different methods, (b): SVD Analysis, (c): Sensitivity of $\beta$ , zoom in to see the details.

2015), DANN (Ganin and Lempitsky 2015), SimNet (Pinheiro 2018), CyCADA (Hoffman et al. 2018), CDAN (Long et al. 2017), TADA (Wang et al. 2019a), BSP (Chen et al. 2019), MCD (Saito et al. 2018), SAFN (Xu et al. 2019), DSAN (Zhu et al. 2020),TADA (Wang et al. 2019a), SymNet (Pinheiro 2018),ATM (Li et al. 2020), TSA (Li et al. 2021b), $\mathrm { T } ^ { \mathrm { 2 } } \mathrm { S A }$ (Xie et al. 2024).

# Results

We present our results in Table. 1 to Table. 4. Our SimProF method significantly enhances the performance of $\mathrm { T } ^ { 2 } \mathrm { S A }$ , achieving average accuracy improvements of $3 . 4 \%$ , $2 . 3 \%$ , $4 . 7 \%$ , and $1 . 6 \%$ on the Office-31, Office-Home, VisDA2017, and DomainNet datasets, respectively. Notably, we observe substantial accuracy gains on some of the most challenging tasks, such as in the Office-31 dataset, where $\mathrm { D } {  } \mathrm { A }$ improved from $78 . 2 \%$ to $8 1 . 7 \%$ and $\mathrm { { W } } {  } \mathrm { { A } }$ improved from $78 . 5 \%$ to $8 4 . 5 \%$ . On the largest dataset, DomainNet, our SimProF method also demonstrates impressive performance with a $1 . 6 \%$ increase compared to $\mathrm { T } ^ { 2 } \mathrm { S } \dot { \mathrm { A } }$ .

# Empirical Analysis

Confusion Matrices. We present the confusion matrices in Figure 3 to visually demonstrate the effectiveness of our proposed method. For the ResNet-50 model, the categories ”aeroplane,” ”car,” and ”motorbike” are particularly challenging to predict. Specifically, there are numerous misclassifications, such as many samples from the ”car” class being incorrectly predicted as ”boat.” In comparison to the domain adversarial method CDAN and the semantic augmentation method $\mathrm { T } ^ { 2 } \mathrm { S A }$ , our SimProF method shows a significant improvement, with a greater number of correct predictions appearing on the diagonal.

Distribution Discrepancy. The $\mathcal { A }$ -distance (Ben-David et al. 2010) is a widely used metric for measuring the distance between two distributions, with a larger $\mathcal { A }$ -distance indicating a greater distinction between the source and target domains. However, the complexity of directly computing the $\mathcal { A }$ -distance prompts us to use the proxy $\mathcal { A }$ -distance, $\hat { d } _ { \mathcal { A } }$ , defined as $\hat { d } _ { A } = 2 ( 1 - 2 \varepsilon )$ . In our analysis, we compute this proxy $\mathcal { A }$ -distance for various models, including ResNet, CDAN, $\mathrm { \dot { T } ^ { 2 } S A }$ , and SimProF, specifically focusing on the feature representations for the $\mathbf A \to \mathbf W$ task in Office-31. The results, shown in Figure 4(a), reveal that SimProF has the lowest proxy $\mathcal { A }$ -distance among the evaluated methods.

SVD Analysis. Singular value decomposition (SVD) offers valuable insights into unsupervised domain adaptation (UDA) through the analysis of singular value distributions (Chen et al. 2019). To explore this, we plot the singular values (using max-normalization) of features extracted by $\mathrm { T } ^ { \mathrm { 2 } } \mathrm { S A }$ and SimProF in Figure 4(b). The figure shows that SimProF effectively reduces the disparity between the largest and the remaining singular values, suggesting improved domain adaptation capability.

Hyper-parameter Sensitivity. Figure 4(c) illustrates the sensitivity of SimProF to the hyper-parameter $\beta$ , with values varying from 0.0, 0.1, 0.2, 0.5, 1.0. The results indicate that SimProF performs optimally when $\beta = 0 . 5$ , demonstrating the robustness and stability of our method.

# Conclusions

In this paper, we introduce SimProF, a straightforward probabilistic framework for UDA to ameliorate the adaptation ability of the classifier by optimizing the derived upper bound of the expected. We utilize a von Mises-Fisher distribution to model the normalized feature space of samples within the context of contrastive learning. Besides. it not only allow for efficient parameter estimation across different batches using maximum likelihood estimation, but also derive a closed-form expression for the expected supervised contrastive loss by sampling an infinite number of samples from the estimated distribution, overcoming the typical limitation of supervised contrastive learning that requires a large sample size to achieve satisfactory performance.