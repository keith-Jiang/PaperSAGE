# Achieving Optimal Clustering in Gaussian Mixture Models with Anisotropic Covariance Structures

Xin Chen Princeton University xc5557@princeton.edu

Anderson Ye Zhang University of Pennsylvania ayz@wharton.upenn.edu

# Abstract

We study clustering under anisotropic Gaussian Mixture Models (GMMs), where covariance matrices from different clusters are unknown and are not necessarily the identity matrix. We analyze two anisotropic scenarios: homogeneous, with identical covariance matrices, and heterogeneous, with distinct matrices per cluster. For these models, we derive minimax lower bounds that illustrate the critical influence of covariance structures on clustering accuracy. To solve the clustering problem, we consider a variant of Lloyd’s algorithm, adapted to estimate and utilize covariance information iteratively. We prove that the adjusted algorithm not only achieves the minimax optimality but also converges within a logarithmic number of iterations, thus bridging the gap between theoretical guarantees and practical efficiency.

# 1 Introduction

Clustering is a fundamentally important task in statistics and machine learning [7, 2]. The most widely recognized and extensively studied model for clustering is the Gaussian Mixture Model (GMM) [17, 19], which is formulated as

$$
Y _ { j } = \theta _ { z _ { j } ^ { * } } ^ { * } + \epsilon _ { j } , \mathrm { w h e r e } \epsilon _ { j } \stackrel { i n d } { \sim } { \mathcal { N } } ( 0 , \Sigma _ { z _ { j } ^ { * } } ^ { * } ) , \forall j \in [ n ] .
$$

Here $Y = ( Y _ { 1 } , \ldots , Y _ { n } ) $ are the observations with $n$ being the sample size. We define the set $[ n ] = \{ 1 , 2 , . . . , n \}$ . Assume $k$ is the known number of clusters. Let $\{ \theta _ { a } ^ { * } \} _ { a \in [ k ] }$ represent the unknown centers, and $\Sigma _ { a } ^ { * }$ denote the corresponding unknown covariance matrices. Define $z ^ { * } \in [ k ] ^ { n }$ as the cluster assignment vector, where for each index $j \in [ n ]$ , the value of $z _ { j } ^ { * }$ specifies which cluster the $j$ -th data point is assigned to. The goal is to recover $z ^ { * }$ from $Y$ . For any estimator $\hat { z }$ , its clustering performance is measured by the misclustering error rate $h ( \hat { z } , z ^ { * } )$ , which will be introduced later in (4).

There has been increasing interest in theoretical and algorithmic analysis of clustering under GMMs. In a scenario where a GMM is isotropic, meaning that all covariance matrices $\{ \Sigma _ { a } ^ { * } \} _ { a \in [ k ] }$ are equal to the identity matrix, [15] obtained the minimax rate for clustering, which takes the form of $\exp ( - ( 1 + o \bar { ( 1 ) } ) ( \operatorname* { m i n } _ { a \neq b } \| \theta _ { a } ^ { * } - \theta _ { b } ^ { * } \| ) ^ { 2 } / 8 )$ , with respect to the misclustering error rate. A diverse range of methods has been explored in the context of the isotropic setting. Among these, Lloyd’s algorithm [13] stands out as a particularly effective clustering algorithm, renowned for its extensive success in a myriad of disciplines. [15, 8] establish computational and statistical guarantees for the Lloyd’s algorithm. Specifically, they showed it achieves the minimax optimal rates after a few iterations provided with some decent initialization. Another popular approach to clustering especially for high dimensional data is the spectral clustering [21, 18, 20], which is an umbrella term for clustering after a dimension reduction through a spectral decomposition. [14] proves the spectral clustering also achieves the optimality under the isotropic GMM. Semidefinite programming (SDP)

is also used for clustering by exploiting its low-rank structure, and its statistical properties have been studied in literature, for example, [5].

Despite the numerous compelling findings, most existing research primarily focuses on isotropic GMMs. The understanding of clustering in an anisotropic context, where the covariance matrices are not constrained to be identity matrices, remains relatively limited. Some studies, including [15, 5, 16, 1, 9, 24], present results for sub-Gaussian mixture models, wherein the errors $\epsilon _ { j }$ are assumed to follow some sub-Gaussian distributions with the variance proxy $\sigma ^ { 2 }$ . At first glance, it might appear that these results encompass the anisotropic case, as distributions of the form $\{ \mathcal { N } ( \bar { 0 } , \Sigma _ { a } ^ { * } ) \bar  \} _ { a \in [ k ] }$ are indeed sub-Gaussian distributions. However, from a minimax perspective, the least favorable scenario among all sub-Gaussian distributions with variance proxy $\sigma ^ { 2 }$ —and thus the most challenging for clustering—is when the errors are distributed as ${ \mathcal { N } } ( { \bar { 0 } } , \sigma ^ { 2 } I )$ . Therefore, the minimax rate for clustering under the sub-Gaussian mixture model essentially equals the one under the isotropic GMM, and methods like Lloyd’s algorithm, which require no covariance matrix information, can be rate-optimal. As a result, the aforementioned findings primarily pertain to isotropic GMMs.

A few studies have explored the direction of clustering under anisotropic GMMs. [3] presents a polynomial-time clustering algorithm that provably performs well when Gaussian distributions are well-separated by hyperplanes. This idea is further developed in [11], which extends the approach to allow overlapping Gaussians, albeit only in two-cluster scenarios. [22] proposes a novel method for clustering under a balanced mixture of two elliptical distributions. They establish a provable upper bound on their clustering performance. Nevertheless, the fundamental limit of clustering under anisotropic GMMs, and whether a polynomial-time procedure can achieve it, remains unknown.

In this paper, we investigate the clustering task under two anisotropic GMMs. In Model 1, all covariance matrices are equal (i.e., homogeneous) to some unknown matrix $\Sigma ^ { * }$ . Model 2 offers more flexibility, with covariance matrices that are unknown and not necessarily identical (i.e., heterogeneous). The contribution of this paper is two-fold, summarized as follows:

• Our first contribution is on the minimax rates. We obtain minimax lower bounds for clustering under anisotropic GMMs with respect to the misclustering error rate. We show they take the form of

$$
\operatorname* { i n f } _ { \hat { z } } \operatorname* { s u p } _ { z ^ { * } } \mathbb { E } h ( \hat { z } , z ^ { * } ) \geq \exp \left( - ( 1 + o ( 1 ) ) \frac { ( \mathrm { s i g n a l - t o - n o i s e \ r a t i o } ) ^ { 2 } } { 8 } \right) ,
$$

where the signal-to-noise ratio under Model 1 is equal to $\begin{array} { r } { \operatorname* { m i n } _ { a , b \in [ k ] : a \neq b } \| ( \theta _ { a } ^ { * } - \theta _ { b } ^ { * } ) ^ { T } \Sigma ^ { * - \frac { 1 } { 2 } } \| } \end{array}$ . The signal-to-noise ratio for Model 2 is more intricate and will be introduced in Section 3. For both models, we can see the minimax rates depend not only on the centers but also on the covariance matrices. This is different from the isotropic case, whose signal-to-noise ratio is $\mathrm { m i n } _ { a \neq b } \| \theta _ { a } ^ { * } - \theta _ { b } ^ { * } \|$ . Our results precisely capture the role that covariance matrices play in the clustering problem. This shows that covariance matrices impact the fundamental limits of the clustering problem through complex interactions with the centers, especially in Model 2. We obtain the minimax lower bounds by drawing connections with Linear Discriminant Analysis (LDA) [6] and Quadratic Discriminant Analysis (QDA).

• Our second and more important contribution is on the computational side. We give a computationally feasible procedure and rate-optimal algorithm for the anisotropic GMM. Lloyd’s algorithm, developed for the isotropic case, is no longer optimal as it only considers distances among centers [3]. We study an adjusted Lloyd’s algorithm which estimates the covariance matrices in each iteration and adjusts the clusters accordingly. It can also be seen as a hard EM algorithm [4]. Here, we modify the E-step of the soft EM by implementing a maximization step that directly assigns data points to clusters, rather than calculating probabilities. As an iterative algorithm, we demonstrate that it achieves the minimax lower bound within $\log n$ iterations. This offers both statistical and computational guarantees, serving as valuable guidance for practitioners. Specifically, if we let $\bar { \boldsymbol { z } } ^ { ( t ) }$ denote the output of the algorithm after $t$ iterations, it holds with high probability that

$$
h ( z ^ { ( t ) } , z ^ { * } ) \leq \exp \left( - ( 1 + o ( 1 ) ) \frac { ( \mathrm { s i g n a l - t o - n o i s e ~ r a t i o } ) ^ { 2 } } { 8 } \right) ,
$$

for all $t \geq \log n$ . The algorithm can be initialized using popular methods like spectral clustering or Lloyd’s algorithm. In our numerical studies, we demonstrate that our algorithm significantly improves over the two aforementioned methods under anisotropic GMMs, and matches the optimal exponent specified in the minimax lower bound.

Paper Organization. The remaining paper is organized as follows. In Section 2, we study Model 1 where the covariance matrices are unknown but homogeneous. In Section 3, we consider Model 2 where covariance matrices are unknown and heterogeneous. For both cases, we establish the minimax lower bound for the clustering and give a computationally feasible and rate-optimal procedure. In Section 4, we provide a numerical comparison with other popular methods. Proofs are included in the supplement.

Notation. For any matrix $X \in \mathbb { R } ^ { d \times d }$ , we denote $\lambda _ { 1 } ( X )$ as its smallest eigenvalue and $\lambda _ { d } ( X )$ as its largest eigenvalue. In addition, we denote $\| X \|$ as its operator norm. For any two vectors $u , v$ of the same dimension, we denote $\langle u , v \rangle = u ^ { T } v$ as their inner product. For any positive integer $d$ , we denote $I _ { d }$ as the $d \times d$ identity matrix. We denote $\mathcal { N } ( \boldsymbol { \mu } , \boldsymbol { \Sigma } )$ as the normal distribution with mean $\mu$ and covariance matrix $\Sigma$ . We denote $\mathbb { I } \left\{ \cdot \right\}$ as the indicator function. For two positive sequences $\left\{ a _ { n } \right\}$ and $\left\{ b _ { n } \right\}$ , $a _ { n } \preceq b _ { n }$ and $a _ { n } = O ( b _ { n } )$ both mean $a _ { n } \leq C b _ { n }$ for some constant $C > 0$ independent of $n$ . We also write $a _ { n } = o ( b _ { n } )$ or $\textstyle { \frac { b _ { n } } { a _ { n } } } \to \infty$ when $\begin{array} { r } { \operatorname* { l i m } \operatorname* { s u p } _ { n } \frac { a _ { n } } { b _ { n } } = 0 } \end{array}$ .

# 2 GMM with Unknown but Homogeneous Covariance Matrices

# 2.1 Model

We first consider the GMM where the covariance matrices of different clusters are unknown but are assumed to be equal to each other. Then the data-generating process can be displayed as follows:

# Model 1:

$$
Y _ { j } = \theta _ { z _ { j } ^ { * } } ^ { * } + \epsilon _ { j } , \mathrm { ~ w h e r e ~ } \epsilon _ { j } \overset { i n d } { \sim } \mathcal { N } ( 0 , \Sigma ^ { * } ) , \forall j \in [ n ] .
$$

Throughout the paper, we call it Model $\jmath$ for simplicity and to distinguish it from a different and more complicated one that will be introduced in Section 3. The goal is to recover the underlying cluster assignment vector $z ^ { * }$ . If $\Sigma ^ { * }$ were known, then (1) can be converted into an isotropic GMM by a linear transformation $( \Sigma ^ { * } ) ^ { - \frac { 1 } { 2 } } Y _ { j }$ . However, the unknown nature of $\Sigma ^ { * }$ makes clustering under this model more challenging than under isotropic GMMs.

Signal-to-noise Ratio. Define the signal-to-noise ratio

$$
\mathrm { S N R } = \operatorname* { m i n } _ { \substack { a , b \in [ k ] : a \neq b } } \| ( \theta _ { a } ^ { * } - \theta _ { b } ^ { * } ) ^ { T } \Sigma ^ { * - \frac { 1 } { 2 } } \| ,
$$

which is a function of all the centers $\{ \theta _ { a } ^ { * } \} _ { a \in [ k ] }$ and the covariance matrix $\Sigma ^ { * }$ . As we will show later in Theorem 2.1, SNR captures the difficulty of the clustering problem and determines the minimax rate. We defer the geometric interpretation of SNR until after presenting Theorem 2.2.

A quantity closely related to SNR is the minimum distance among the centers. Define $\Delta$ as

$$
\Delta = \operatorname* { m i n } _ { a , b \in \left[ k \right] : a \neq b } \left. \theta _ { a } ^ { * } - \theta _ { b } ^ { * } \right. .
$$

Then we can see SNR and $\Delta$ are of the same order if all eigenvalues of the covariance matrix $\Sigma ^ { * }$ are assumed to be constants. If $\Sigma ^ { * }$ is further assumed to be $\sigma ^ { 2 } \bar { I } _ { d }$ , then SNR equals $\Delta / \sigma$ . As a result, in [15, 8, 14] where the isotropic GMMs are studied, $\Delta / \sigma$ plays the role of signal-to-noise ratio and appears in their rates. Since (2) represents a direct generalization, we refer to it as the signal-to-noise ratio for Model 1.

Loss Function. To measure the clustering performance, we consider the following loss function. For any $z , z ^ { * } \in [ k ] ^ { n }$ , we define

$$
h ( z , z ^ { * } ) = \operatorname* { m i n } _ { \psi \in \Psi } \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \mathbb { I } \left\{ \psi ( z _ { j } ) \neq z _ { j } ^ { * } \right\} ,
$$

where $\Psi = \{ \psi : \psi$ is a bijection from $[ k ]$ to $[ k ] \}$ . Here, the minimum is taken over all permutations of $[ k ]$ to address the identifiability issues of the labels $1 , 2 , \ldots , k$ . The loss function measures the

proportion of coordinates where $z$ and $z ^ { * }$ differ, modulo any permutation of label symbols. Thus, it is referred to as the misclustering error rate in this paper. Another loss that will be used is $\ell ( z , z ^ { * } )$ defined as

$$
\ell ( z , z ^ { * } ) = \sum _ { j = 1 } ^ { n } \left\| \theta _ { z _ { j } } ^ { * } - \theta _ { z _ { j } ^ { * } } ^ { * } \right\| ^ { 2 } .
$$

It measures the clustering performance of $z$ considering the distances among the true centers. It is related to $h ( z , z ^ { * } )$ as $h ( \bar { z } , \bar { z } ^ { * } ) \leq \ell ( z , z ^ { * } ) / ( n \Delta ^ { 2 } )$ and provides more information than $h ( z , z ^ { * } )$ . We will mainly use $\ell ( z , z ^ { * } )$ in the technical analysis but will present results using $h ( z , z ^ { * } )$ which is more interpretable.

# 2.2 Minimax Lower Bound

We first establish the minimax lower bound for the clustering problem under Model 1.

Theorem 2.1. Under the assumption √SloNgR k → ∞, we have

$$
\operatorname* { i n f } _ { \hat { z } } \operatorname* { s u p } _ { z ^ { * } \in [ k ] ^ { n } } \mathbb { E } h ( \hat { z } , z ^ { * } ) \geq \exp \left( - ( 1 + o ( 1 ) ) \frac { S N R ^ { 2 } } { 8 } \right) .
$$

If $S N R = O ( 1 )$ instead, we have $\begin{array} { r } { \operatorname* { i n f } _ { \hat { z } } \operatorname* { s u p } _ { z ^ { * } \in [ k ] ^ { n } } \mathbb { E } h ( \hat { z } , z ^ { * } ) \geq c } \end{array}$ for some constant $c > 0$

Theorem 2.1 allows the cluster numbers $k$ to grow with $n$ and shows that $\mathbf { S N R }  \infty$ is a necessary condition to have a consistent clustering. If $k$ is a constant, then $\mathrm { S N R }  \infty$ is also a sufficient condition. Theorem 2.1 holds for any arbitrary configurations of $\{ \theta _ { a } ^ { * } \} _ { a \in [ k ] }$ and $\Sigma ^ { * }$ , with the minimax lower bound depending on these through SNR. The parameter space i∈s only for $z ^ { * }$ while $\{ \theta _ { a } ^ { * } \} _ { a \in [ k ] }$ and $\Sigma ^ { * }$ are held fixed. Hence, (6) can be interpreted as a case-specific result, precisely capturing the explicit dependence of the minimax rates on $\{ \theta _ { a } ^ { * } \} _ { a \in [ k ] }$ and $\Sigma ^ { * }$ .

Theorem 2.1 is closely related to the LDA. If there are only two clusters with known centers and a covariance matrix, then estimating each $z _ { j } ^ { \ast }$ becomes exactly the task of the LDA: we aim to determine from which of two normal distributions, each with a different mean but the same covariance matrix, the observation $Y _ { j }$ is generated. In fact, this approach is also how Theorem 2.1 is proved: We first reduce the estimation problem of $z ^ { * }$ to two-point hypothesis testing for each individual $z _ { j } ^ { * }$ . The error of these tests is analyzed in Lemma A.1 using the LDA, and we then aggregate all these testing errors together.

![](images/8376c262909abbd2a0fe05b9ac93abefe155b9776df9bf16155215f72dbc15d8.jpg)  
Figure 1: A geometric interpretation of SNR.

With the help of Lemma A.1, we have a geometric interpretation of SNR. In the left panel of Figure 1, we have two normal distributions $\mathcal { N } ( \theta _ { 1 } ^ { * } , \Sigma ^ { * } )$ and $\mathcal { N } ( \theta _ { 2 } ^ { * } , \Sigma ^ { * } )$ that $X$ follows. The black line represents the optimal testing procedure $\phi$ displayed in Lemma A.1, dividing the space into two half-spaces. To calculate the testing error, we can make the transformation $X ^ { \prime } = ( \Sigma ^ { * } ) ^ { - \frac { 1 } { 2 } } ( X - \theta _ { 1 } ^ { * } )$ so that the two normal distributions become isotropic: $\mathcal { N } ( 0 , I _ { d } )$ and $\mathcal { N } ( ( \Sigma ^ { * } ) ^ { - \frac { 1 } { 2 } } ( \theta _ { 2 } ^ { * } - \theta _ { 1 } ^ { * } ) , I _ { d } )$ as displayed in the right panel. Then the distance between the two centers is $\lVert ( \Sigma ^ { * } ) ^ { - \frac { 1 } { 2 } } ( \theta _ { 2 } ^ { * } - \theta _ { 1 } ^ { * } ) \rVert$ , and the distance from a center to the black curve is half of that. Then, the probability that $\mathcal { N } ( 0 , I _ { d } )$ falls within the grayed area equals $\exp ( - ( 1 + o ( 1 ) ) \| ( \Sigma ^ { * } ) ^ { - \frac { 1 } { 2 } } ( \theta _ { 2 } ^ { * } - \theta _ { 1 } ^ { * } ) \| ^ { 2 } / 8 )$ , according to Gaussian tail probability. As a result, $\lVert ( \Sigma ^ { * } ) ^ { - \frac { 1 } { 2 } } ( \theta _ { 2 } ^ { * } - \theta _ { 1 } ^ { * } ) \rVert$ is the effective distance between the two centers of $\mathcal { N } ( \theta _ { 1 } ^ { * } , \Sigma ^ { * } )$ and $\mathcal { N } ( \theta _ { 2 } ^ { * } , \Sigma ^ { * } )$ for the clustering problem, taking into account the geometry of the covariance matrix. Since we have multiple clusters, SNR defined in (2) can be interpreted as the minimum effective distance among the centers $\{ \theta _ { a } ^ { * } \} _ { a \in [ k ] }$ , considering the anisotropic structure of $\Sigma ^ { * }$ . This measure captures the intrinsic difficulty of the clustering problem.

# 2.3 Rate-Optimal Adaptive Procedure

In this section, we give a computationally feasible and rate-optimal procedure for clustering under Model 1. Summarized in Algorithm 1, it is a variant of Lloyd’s algorithm. Starting with an initial setup, it iteratively updates the estimates of the centers $\{ \theta _ { a } ^ { * } \} _ { a \in [ k ] }$ (in (7)), the covariance matrix $\Sigma ^ { * }$ (in (8)), and the cluster assignment vector $z ^ { * }$ (in (9)). This algorithm differs from Lloyd’s algorithm in that the latter is designed for isotropic GMMs and does not incorporate the covariance matrix update outlined in (8). Furthermore, (9) updates the estimation of $z _ { j } ^ { * }$ using $\mathrm { ; a r g m i n } _ { a \in [ k ] } ( Y _ { j } - \theta _ { a } ^ { ( t ) } ) ^ { T } ( Y _ { j } - \theta _ { a } ^ { ( t ) } )$ instead. To differentiate clearly, we refer to the classic form as the vanilla Lloyd’s algorithm and our modified version, which accommodates the unknown and anisotropic covariance matrix, as the adjusted Lloyd’s algorithm.

Algorithm 1 can also be interpreted as a hard EM algorithm. When applying Expectation Maximization (EM) to Model 1, the M step estimates the parameters $\{ \theta _ { a } ^ { * } \} _ { a \in [ k ] }$ and $\Sigma ^ { * }$ , while the E step estimates $z ^ { * }$ . It turns out the updates on the parameters (7) - (8) are identical to those in the EM’s M step. However, the update of $z ^ { * }$ in Algorithm 1 differs from that in the EM. Instead of computing a conditional expectation typical of the E step, the algorithm performs maximization in (9). As a result, Algorithm 1 effectively consists solely of M steps for both parameters and $z ^ { * }$ , characterizing it as a hard EM algorithm.

Algorithm 1: Adjusted Lloyd’s Algorithm for Model 1.

Input: Data $Y$ , number of clusters $k$ , an initialization $z ^ { ( 0 ) }$ , number of iterations $T$ . Output: $z ^ { ( T ) }$

1 for $t = 1 , \dots , T$ do

2 Update the centers:

$$
\theta _ { a } ^ { ( t ) } = \frac { \sum _ { j \in [ n ] } Y _ { j } \mathbb { I } \left\{ z _ { j } ^ { ( t - 1 ) } = a \right\} } { \sum _ { j \in [ n ] } \mathbb { I } \left\{ z _ { j } ^ { ( t - 1 ) } = a \right\} } , \quad \forall a \in [ k ] .
$$

3

Update the covariance matrix:

$$
\Sigma ^ { ( t ) } = \frac { \sum _ { a \in [ k ] } \sum _ { j \in [ n ] } ( Y _ { j } - \theta _ { a } ^ { ( t ) } ) ( Y _ { j } - \theta _ { a } ^ { ( t ) } ) ^ { T } \mathbb { I } \left\{ z _ { j } ^ { ( t - 1 ) } = a \right\} } { n } .
$$

4

Update the cluster assignment vector:

$$
z _ { j } ^ { ( t ) } = \underset { a \in [ k ] } { \operatorname { a r g m i n } } ( Y _ { j } - \theta _ { a } ^ { ( t ) } ) ^ { T } ( \Sigma ^ { ( t ) } ) ^ { - 1 } ( Y _ { j } - \theta _ { a } ^ { ( t ) } ) , \quad \forall j \in [ n ] .
$$

In Theorem 2.2, we give a computational and statistical guarantee of Algorithm 1. We show that starting from a decent initialization, within $\log n$ iterations, Algorithm 1 achieves the error rate $\exp \bigl ( - ( 1 + o ( 1 ) ) \mathrm { S N R } ^ { 2 } / 8 \bigr )$ which matches the minimax lower bound given in Theorem 2.1. As a result, Algorithm 1 is a rate-optimal procedure. In addition, the algorithm is fully adaptive to the unknown $\{ \theta _ { a } ^ { * } \} _ { a \in [ k ] }$ and $\Sigma ^ { * }$ . The sole piece of information presumed to be known is $k$ , the number of clusters, as commonly assumed in clustering literature [15, 8, 14]. The theorem also shows that the number of iterations needed to achieve the optimal rate is at most $\log n$ , providing implementation guidance to practitioners.

Theorem 2.2. Assume $k = O ( 1 )$ , $d = O ( { \sqrt { n } } )$ , and $\begin{array} { r } { \operatorname* { m i n } _ { a \in [ k ] } \sum _ { j = 1 } ^ { n } \mathbb { I } \{ z _ { j } ^ { * } = a \} \geq \frac { \alpha n } { k } } \end{array}$ for some constant $\alpha > 0$ . Assume $S N R  \infty$ and $\lambda _ { d } ( \Sigma ^ { * } ) / \lambda _ { 1 } ( \Sigma ^ { * } ) = \bar { \cal O ( 1 ) }$ . For Algorithm $\jmath$ , suppose $z ^ { ( 0 ) }$ satisfies $\ell ( z ^ { ( 0 ) } , z ^ { * } ) = o ( n )$ with probability at least $1 - \eta$ . Then with probability at least $1 - \eta - \dot { n } ^ { - 1 } - \mathrm { e x p } ( - \dot { S } N R )$ , we have

$$
h ( z ^ { ( t ) } , z ^ { \ast } ) \leq \exp \left( - ( 1 + o ( 1 ) ) \frac { S N R ^ { 2 } } { 8 } \right) , \quad f o r a l l t \geq \log n .
$$

We make the following remarks on the assumptions of Theorem 2.2: When $k$ is constant, the assumption that $\mathrm { S N R }  \infty$ is a necessary condition for consistent recovery of $z ^ { * }$ , as outlined in the minimax lower bound presented in Theorem 2.1. The assumption on $\Sigma ^ { * }$ ensures that the covariance matrix is well-conditioned. The dimensionality $d$ is assumed to be $O ( { \sqrt { n } } )$ , a stronger assumption than in [15, 8, 14], where $d = O ( n )$ is sufficient. This is because, unlike these studies, our work requires estimating the covariance matrix $\Sigma ^ { * }$ and controlling the estimation error $\| \Sigma ^ { ( t ) } - \Sigma ^ { * } \|$ .

Theorem 2.2 needs a decent initialization $z ^ { ( 0 ) }$ in the sense that it is sufficiently close to the ground truth such that $\ell ( z ^ { ( 0 ) } , z ^ { * } ) = o ( n )$ . This is because our theoretical analysis requires the initialization being within a specific proximity to the true parameters. The requirement can be fulfilled by simple procedures. An example is the vanilla Lloyd’s algorithm whose performance is studied in [15, 8]. Though [15, 8] are for isotropic GMMs, their results can be extended to sub-Gaussian mixture models with nearly identical proof. Since $\epsilon _ { j }$ are sub-Gaussian random variables with proxy variance $\lambda _ { d } ( \Sigma ^ { * } )$ , [8] implies the vanilla Lloyd’s algorithm output $\hat { z }$ satisfies $\ell ( \hat { z } , z ^ { * } ) \leq$ $n \exp ( - ( 1 + o ( 1 ) ) \Delta ^ { 2 } / ( 8 \lambda _ { d } ( \Sigma ^ { * } ) ) )$ with probability at least $1 - \exp ( - \Delta / \sqrt { \lambda _ { d } ( \Sigma ^ { * } ) } ) - n ^ { - 1 }$ , under the assumption that $\Delta ^ { 2 } / ( k ^ { 2 } ( k d / n + 1 ) \lambda _ { d } ( \Sigma ^ { * } ) )  \infty$ . Then we have $\ell ( \hat { z } , z ^ { * } ) = o ( n )$ with high probability under the assumptions of Theorem 2.2, and hence it can be used as an initialization for the algorithm.

# 3 GMM with Unknown and Heterogeneous Covariance Matrices

# 3.1 Model

In this section, we study the GMM where the covariance matrices of each cluster are unknown and not necessarily equal to each other. The data-generation process can be displayed as follows,

# Model 2:

$$
Y _ { j } = \theta _ { z _ { j } ^ { * } } ^ { * } + \epsilon _ { j } , \mathrm { w h e r e } \epsilon _ { j } \stackrel { i n d } { \sim } { \mathcal { N } } ( 0 , \Sigma _ { z _ { j } ^ { * } } ^ { * } ) , \forall j \in [ n ] .
$$

We refer to this as Model 2 throughout the paper to distinguish it from Model 1, as discussed in Section 2. The key difference between (10) and (1) is that here we have distinct covariance matrices $\{ \Sigma _ { a } ^ { * } \} _ { a \in [ k ] }$ for each cluster, instead of a single shared $\Sigma ^ { * }$ . We use the same loss function as defined in (4).

Signal-to-noise Ratio. The signal-to-noise ratio for Model 2 is defined as follows. We use the notation $\mathrm { S N R ^ { \prime } }$ to distinguish it from the SNR used for Model 1. Compared to SNR, $\mathrm { S N R } ^ { \prime }$ is much more complicated and does not have an explicit formula. We first define a set $B _ { a , b } \subset \mathbb { R } ^ { d }$ for any $a , b \in [ k ]$ such that $a \neq b$ :

$$
\begin{array} { l } { \displaystyle { B _ { a , b } = \left\{ x \in \mathbb { R } ^ { d } : x ^ { T } \Sigma _ { a } ^ { * \frac { 1 } { 2 } } \Sigma _ { b } ^ { * - 1 } ( \theta _ { a } ^ { * } - \theta _ { b } ^ { * } ) + \frac { 1 } { 2 } x ^ { T } \Big ( \Sigma _ { a } ^ { * \frac { 1 } { 2 } } \Sigma _ { b } ^ { * - 1 } \Sigma _ { a } ^ { * \frac { 1 } { 2 } } - I _ { d } \Big ) x \right. } } \\ { \displaystyle { \left. \qquad \leq - \frac { 1 } { 2 } ( \theta _ { a } ^ { * } - \theta _ { b } ^ { * } ) ^ { T } \Sigma _ { b } ^ { * - 1 } ( \theta _ { a } ^ { * } - \theta _ { b } ^ { * } ) + \frac { 1 } { 2 } \log | \Sigma _ { a } ^ { * } | - \frac { 1 } { 2 } \log | \Sigma _ { b } ^ { * } | \right\} } . } \end{array}
$$

We then define $\begin{array} { r } { \mathbf { S N R } _ { a , b } ^ { \prime } = 2 \operatorname* { m i n } _ { x \in B _ { a , b } } \left\| x \right\| } \end{array}$ and

$$
\mathrm { S N R } ^ { \prime } = \operatorname* { m i n } _ { a , b \in [ k ] : a \neq b } \mathrm { S N R } _ { a , b } ^ { \prime } .
$$

The form of $\mathrm { S N R ^ { \prime } }$ is closely connected to the testing error of the QDA, which we will give in Lemma 3.1. The interpretation of the $\mathrm { S N R ^ { \prime } }$ , particularly from a geometric perspective, will be deferred until after the presentation of Lemma 3.1. Here let us consider a few special cases where we are able to simplify $\mathrm { S N R ^ { \prime } }$ : (1) When $\Sigma _ { a } ^ { * } = \Sigma ^ { * }$ for all $a \in [ k ]$ , by simple algebra, we have $\mathrm { S N R } _ { a , b } ^ { \prime } = \| ( \theta _ { a } ^ { * } - \theta _ { b } ^ { * } ) ^ { T } \Sigma ^ { * - \frac { 1 } { 2 } } \|$ for any $a , b \in [ k ]$ such that $a \neq b$ . Hence, $\mathrm { S N R } ^ { \prime } = \mathrm { S N R }$ and Model 2 effectively reduces to Model 1. (2) When $\Sigma _ { a } ^ { * } = \sigma _ { a } ^ { 2 } I _ { d }$ for any $a \in [ k ]$ where $\sigma _ { 1 } , \ldots , \sigma _ { k } > 0$ are large constants, we have $\mathrm { S N R } _ { a , b } ^ { \prime }$ , $\mathrm { S N R } _ { b , a } ^ { \prime }$ both close to $2 \lVert \theta _ { a } ^ { * } - \theta _ { b } ^ { * } \rVert / ( \sigma _ { a } ^ { - } + \sigma _ { b } )$ . From these examples, we can see $\mathbf { S N R ^ { \prime } }$ is determined by both the centers $\{ \theta _ { a } ^ { * } \} _ { a \in [ k ] }$ and the covariance matrices $\{ \Sigma _ { a } ^ { * } \} _ { a \in [ k ] }$ .

# 3.2 Minimax Lower Bound

We first establish the minimax lower bound for the clustering problem under Model 2.

Theorem 3.1. Assume $d = O ( 1 )$ and $\begin{array} { r } { \operatorname* { m a x } _ { a , b \in [ k ] } \lambda _ { d } \bigl ( \Sigma _ { a } ^ { * } \bigr ) / \lambda _ { 1 } \bigl ( \Sigma _ { b } ^ { * } \bigr ) = O ( 1 ) } \end{array}$ . Under the assumption √SlNoRg′k → ∞, we have

$$
\operatorname* { i n f } _ { \hat { z } } \operatorname* { s u p } _ { z ^ { * } \in [ k ] ^ { n } } \mathbb { E } h ( \hat { z } , z ^ { * } ) \geq \exp \left( - ( 1 + o ( 1 ) ) \frac { S N R ^ { ' 2 } } { 8 } \right) .
$$

If $S N R ^ { \prime } = O ( 1 )$ instead, we have $\begin{array} { r } { \operatorname* { i n f } _ { \hat { z } } \operatorname* { s u p } _ { z ^ { * } \in [ k ] ^ { n } } \mathbb { E } h ( \hat { z } , z ^ { * } ) \geq c } \end{array}$ for some constant $c > 0$ .

Although the statement of Theorem 3.1 appears similar to that of Theorem 2.1, the two minimax lower bounds differ due to the varying dependencies of the centers and covariance matrices on $\mathbf { S N R ^ { \prime } }$ versus SNR. Using the same argument as in Section 2.2, the minimax lower bound established in Theorem 3.1 closely relates to the QDA between two normal distributions with different means and different covariance matrices.

Lemma 3.1 (Testing Error for the QDA). Consider two hypotheses $\mathbb { H } _ { 0 } : X \sim \mathcal { N } ( \theta _ { 1 } ^ { * } , \Sigma _ { 1 } ^ { * } )$ and $\mathbb { H } _ { 1 } : X \sim { \mathcal { N } } ( \theta _ { 2 } ^ { * } , \Sigma _ { 2 } ^ { * } )$ . Define a testing procedure

$$
\phi = \mathbb { I } \left\{ \log | \Sigma _ { 1 } ^ { * } | + ( x - \theta _ { 1 } ^ { * } ) ^ { T } ( \Sigma _ { 1 } ^ { * } ) ^ { - 1 } ( x - \theta _ { 1 } ^ { * } ) \geq \log | \Sigma _ { 2 } ^ { * } | + ( x - \theta _ { 2 } ^ { * } ) ^ { T } ( \Sigma _ { 2 } ^ { * } ) ^ { - 1 } ( x - \theta _ { 2 } ^ { * } ) \right\} .
$$

Then we have inf $\overset { \cdot } { \phi } \bigl ( \mathbb { P } _ { \mathbb { H } _ { 0 } } ( \hat { \phi } = 1 ) + \mathbb { P } _ { \mathbb { H } _ { 1 } } ( \hat { \phi } = 0 ) \bigr ) = \mathbb { P } _ { \mathbb { H } _ { 0 } } ( \phi = 1 ) + \mathbb { P } _ { \mathbb { H } _ { 1 } } ( \phi = 0 )$ . Assume $d = O ( 1 )$ and $\operatorname* { m a x } _ { a , b \in \{ 1 , 2 \} }$ $_ { a , b \in \{ 1 , 2 \} } \lambda _ { d } ( \Sigma _ { a } ^ { * } ) / \lambda _ { 1 } ( \Sigma _ { b } ^ { * } ) = O ( 1 )$ . $I f \operatorname* { m i n } \{ S N R _ { 1 , 2 } ^ { \prime } , S N R _ { 2 , 1 } ^ { \prime } \}  \infty$ , we have

$$
\operatorname* { i n f } _ { \hat { \phi } } \bigl ( \mathbb { P } _ { \mathbb { H } _ { 0 } } ( \hat { \phi } = 1 ) + \mathbb { P } _ { \mathbb { H } _ { 1 } } ( \hat { \phi } = 0 ) \bigr ) \geq \exp \left( - ( 1 + o ( 1 ) ) \frac { \operatorname* { m i n } \big \{ S N R _ { 1 , 2 } ^ { \prime } , S N R _ { 2 , 1 } ^ { \prime } \big \} ^ { 2 } } { 8 } \right) .
$$

Otherwise, $\begin{array} { r } { \operatorname* { i n f } _ { \hat { \phi } } ( \mathbb { P } _ { \mathbb { H } _ { 0 } } ( \hat { \phi } = 1 ) + \mathbb { P } _ { \mathbb { H } _ { 1 } } ( \hat { \phi } = 0 ) ) \geq c } \end{array}$ for some constant $c > 0$ .

![](images/442de70d8508d6e66d489a1bb8ab58ff7dce03a122dfd0eb501802a381a894e1.jpg)  
Figure 2: A geometric interpretation of $\mathrm { S N R ^ { \prime } }$ .

Lemma 3.1 provides a geometric interpretation of $\mathrm { S N R } ^ { \prime }$ . In the left panel of Figure 2, we have two normal distributions $\mathcal { N } ( \theta _ { 1 } ^ { * } , \Sigma _ { 1 } ^ { * } )$ and $\bar { \mathcal { N } ( \theta _ { 2 } ^ { * } , \Sigma _ { 2 } ^ { * } ) }$ from which $X$ can be generated, and the black curve represents the optimal testing procedure $\phi$ , as detailed in Lemma 3.1. Since $\Sigma _ { 1 } ^ { * }$ is not necessarily equal to $\Sigma _ { 2 } ^ { * }$ , the black curve is not necessarily a straight line. If $\mathbb { H } _ { 0 }$ is true, the probability that $X$ is incorrectly classified occurs when $X$ falls into the gray area, represented by $\mathbb { P } _ { \mathbb { H } _ { 0 } } ( \phi \stackrel { \cdot } { = } 1 )$ . To calculate this, we transform $X$ to $X ^ { \prime } = ( \Sigma _ { 1 } ^ { * } ) ^ { - \frac { 1 } { 2 } } ( X - \theta _ { 1 } ^ { * } )$ , standardizing the first distribution. Then, as displayed in the right panel of Figure 2, the two distributions become $\mathcal { N } ( 0 , I _ { d } )$ and $\mathcal { N } ( ( \Sigma _ { 1 } ^ { * } ) ^ { - \frac { 1 } { 2 } } ( \theta _ { 2 } ^ { * } -$ $\theta _ { 1 } ^ { * } )$ , $( { \bar { \Sigma } _ { 1 } ^ { * } } ) ^ { - \frac { 1 } { 2 } } \Sigma _ { 2 } ^ { * } ( \bar { \Sigma _ { 1 } ^ { * } } ) ^ { - \frac { 1 } { 2 } } )$ , and the optimal testing procedure $\phi$ becomes $\mathbb { I } \left\{ X ^ { \prime } \in B _ { 1 , 2 } \right\}$ . As a result, in the right panel of Figure 2, $B _ { 1 , 2 }$ represents the space colored by gray, and the black curve is its boundary. Then $\bar { \mathbb { P } } _ { \mathbb { H } _ { 0 } } \bar { ( \phi = 1 ) }$ is equal to $\mathbb { P } ( \mathcal { N } ( 0 , \bar { I } _ { d } ) \in B _ { 1 , 2 } )$ . Under the assumption $d = O ( 1 )$ and maxa,b∈{1,2} $\lambda _ { d } ( \Sigma _ { a } ^ { * } ) / \lambda _ { 1 } ( \Sigma _ { b } ^ { * } ) = { \cal O } ( 1 )$ , in Lemma C.10, we can show $\mathbb { P } ( \mathcal { N } ( 0 , I _ { d } ) \in B _ { 1 , 2 } ) \ \stackrel { \cdot } { = }$ $\exp ( - ( 1 + o ( 1 ) ) \mathbf { S N R } _ { 1 , 2 } ^ { ' 2 } / 8 )$ . As a result, $\mathbf { S N R ^ { \prime } }$ can be interpreted as the minimum effective distance among the centers $\{ \theta _ { a } ^ { * } \} _ { a \in [ k ] }$ , considering the anisotropic and heterogeneous structure of $\{ \Sigma _ { a } ^ { * } \} _ { a \in [ k ] }$ , and it captures the intrinsic difficulty of the clustering problem under Model 2.

# 3.3 Optimal Adaptive Procedure

In this section, we give a computationally feasible and rate-optimal procedure for clustering under Model 2. Similar to Algorithm 1, Algorithm 2 is a variant of Lloyd’s algorithm, adjusted to accommodate unknown and heterogeneous covariance matrices. It can also be interpreted as a hard EM algorithm under Model 2. Algorithm 2 differs from Algorithm 1 in (13) and (14), as now there are $k$ covariance matrices instead of a common one.

Algorithm 2: Adjusted Lloyd’s Algorithm for Model 2.

Input: Data $Y$ , number of clusters $k$ , an initialization $z ^ { ( 0 ) }$ , number of iterations $T$ . Output: $z ^ { ( T ) }$

1 for $t = 1 , \dots , T$ do

2 Update the centers:

$$
\theta _ { a } ^ { ( t ) } = \frac { \sum _ { j \in [ n ] } Y _ { j } \mathbb { I } \left\{ z _ { j } ^ { ( t - 1 ) } = a \right\} } { \sum _ { j \in [ n ] } \mathbb { I } \left\{ z _ { j } ^ { ( t - 1 ) } = a \right\} } , \quad \forall a \in [ k ] .
$$

3 Update the covariance matrices:

$$
\Sigma _ { a } ^ { ( t ) } = \frac { \sum _ { j \in [ n ] } ( Y _ { j } - \theta _ { a } ^ { ( t ) } ) ( Y _ { j } - \theta _ { a } ^ { ( t ) } ) ^ { T } \mathbb { I } \left\{ z _ { j } ^ { ( t - 1 ) } = a \right\} } { \sum _ { j \in [ n ] } \mathbb { I } \left\{ z _ { j } ^ { ( t - 1 ) } = a \right\} } , \quad \forall a \in [ k ] .
$$

4

Update the cluster assignment vector:

$$
z _ { j } ^ { ( t ) } = \underset { a \in [ k ] } { \operatorname { a r g m i n } } ( Y _ { j } - \theta _ { a } ^ { ( t ) } ) ^ { T } ( \Sigma _ { a } ^ { ( t ) } ) ^ { - 1 } ( Y _ { j } - \theta _ { a } ^ { ( t ) } ) + \log | \Sigma _ { a } ^ { ( t ) } | , \quad \forall j \in [ n ] .
$$

In Theorem 3.2, we give a computational and statistical guarantee for Algorithm 2. We demonstrate that, with proper initialization, Algorithm 2 achieves the minimax lower bound within $\log n$ iterations. The assumptions needed in Theorem 3.2 are similar to those in Theorem 2.2, except that we require stronger assumptions on the dimensionality $d$ since now we have $k$ (instead of one) covariance matrices to be estimated. In addition, by assuming $\begin{array} { r } { \operatorname* { m a x } _ { a , b \in [ k ] } \lambda _ { d } \big ( \Sigma _ { a } ^ { * } \big ) / \lambda _ { 1 } \big ( \Sigma _ { b } ^ { * } \big ) = O ( 1 ) } \end{array}$ , we ensure not only that each of the $k$ covariance matrices is well-conditioned but also that they are comparable to one another.

Theorem 3.2. Assume $k = O ( 1 )$ , $d = O ( 1 )$ , and $\begin{array} { r } { \operatorname* { m i n } _ { a \in [ k ] } \sum _ { j = 1 } ^ { n } \mathbb { I } \{ z _ { j } ^ { * } = a \} \geq \frac { \alpha n } { k } } \end{array}$ for some constant $\alpha > 0$ . Assume $S N R ^ { \prime }  \infty$ and $\begin{array} { r } { \operatorname* { m a x } _ { a , b \in [ k ] } \lambda _ { d } \bigl ( \Sigma _ { a } ^ { * } \bigr ) / \lambda _ { 1 } \bigl ( \Sigma _ { b } ^ { * } \bigr ) = { \cal O } ( 1 ) } \end{array}$ . For Algorithm 2, suppose $z ^ { ( 0 ) }$ satisfies $\ell ( z ^ { ( 0 ) } , z ^ { * } ) = o ( n )$ with probability at least $1 - \eta$ . Then with probability at least $1 - \eta - 5 n ^ { - 1 } - \mathrm { e x p } ( - S N R ^ { \prime } )$ , we have

$$
h ( z ^ { ( t ) } , z ^ { * } ) \leq \exp \left( - ( 1 + o ( 1 ) ) { \frac { S N R ^ { ' 2 } } { 8 } } \right) , \quad f o r a l l t \geq \log n .
$$

The vanilla Lloyd’s algorithm can be used as the initialization for Algorithm 2. This is because Model 2 is also a sub-Gaussian mixture model. By the same argument as in Section 2.3, the output of the vanilla Lloyd’s algorithm $\hat { z }$ satisfies $\ell ( \hat { z } , z ^ { * } ) = o ( n )$ with high probability under the assumptions of Theorem 3.2.

We conclude this section with a time complexity analysis of Algorithm 2. Compared to the vanilla Lloyd’s algorithm, our method introduces additional computational overhead due to the need for computing the inverse and determinant of covariance matrices. Specifically, the time complexity of Algorithm 2 is $O ( n k d ^ { 3 } T )$ . In contrast, the vanilla Lloyd’s algorithm has a lower time complexity of $O ( n k d T )$ . The increase in complexity stems from matrix operations in $d$ dimensions, as both matrix inversion and determinant computation scale as $O ( d ^ { 3 } )$ .

# 4 Numerical Studies

In this section, we compare the performance of our methods with other popular clustering methods on synthetic and real datasets under different settings.

Model 1. The first simulation is designed for the GMM with unknown but homogeneous covariance matrices (i.e., Model 1). We independently generate $n = 1 2 0 0$ samples with dimension $d = 5 0$ from $k = 3 0$ clusters. Each cluster has 40 samples. We set $\Sigma ^ { * } = U ^ { T } \Lambda \dot { U }$ , where $\Lambda$ is a $5 0 \times 5 0$ diagonal matrix with diagonal elements selected from 0.5 to 8 with equal space and $U$ is a randomly generated orthogonal matrix. The centers $\{ \theta _ { a } ^ { * } \} _ { a \in [ n ] }$ are orthogonal to each other with $\lVert { \boldsymbol { \theta } } _ { 1 } ^ { * } \rVert = . . . = \lVert { \boldsymbol { \theta } } _ { 3 0 } ^ { * } \rVert = 9$ . We consider four popular clustering methods: (1) the spectral clustering method in [14] (denoted as “spectral”), (2) the vanilla Lloyd’s algorithm in [15] (denoted as “vanilla Lloyd”), (3) Algorithm 1 initialized by the spectral clustering (denoted as “spectral $+ \mathrm { A l g \ 1 ^ { \circ } } ,$ ), and (4) Algorithm 1 initialized by the vanilla Lloyd (denoted as “vanilla Lloyd $+ \mathrm { A l g } 1 ^ { \prime \prime } ,$ ). The comparison is presented in the left panel of Figure 3.

Model 2. We also compare the performances of four methods (spectral, vanilla Lloyd, spectral $+$ Alg 2, and vanilla Lloy $\mathrm { d } + \mathrm { A l g } 2 ,$ ) for the GMM with unknown and heterogeneous covariance matrices (i.e., Model 2). In this case, we take $n = 1 2 0 0$ , $k = 2$ , and $d = 9$ . We set $\Sigma _ { 1 } ^ { * } = I _ { d }$ and $\Sigma _ { 2 } ^ { * } = \Lambda _ { 2 }$ , a diagonal matrix where the first diagonal entry is 0.5 and the remaining entries are 5. We set the cluster sizes to be 900 and 300, respectively. To simplify the calculation of $\mathrm { S N R ^ { \prime } }$ , we set $\theta _ { 1 } ^ { * } = 0$ and $\theta _ { 2 } ^ { * } = 5 e _ { 1 }$ , with $e _ { 1 }$ being the vector that has a 1 in its first entry and 0s elsewhere. The comparison is presented in the right panel of Figure 3.

![](images/fc390f46aa95542442d0035e53d4949563c38046b6825e2515dc8c4182520bd5.jpg)  
Figure 3: Left: Performance of Algorithm 1 compared with other methods under Model 1. Right: Performance of Algorithm 2 compared with other methods under Model 2.

In Figure 3, the $x$ -axis is the number of iterations and the $y$ -axis is the logarithm of the misclustering error rate, i.e., $\log ( h )$ . Each of the curves plotted is an average of 100 independent trials. We can see both Algorithm 1 and Algorithm 2 outperform the spectral clustering and the vanilla Lloyd’s algorithm significantly. Additionally, the dashed lines in the left and right panels represent the optimal exponents $- \dot { \mathrm { S N R } } ^ { 2 } / 8$ and $- \dot { \mathrm { S N R } } ^ { \prime 2 } / 8$ of the minimax bounds, respectively. It is observed that both Algorithm 1 and Algorithm 2 meet these benchmarks after three iterations. This justifies the conclusion that both algorithms are rate-optimal.

Real Data. To further demonstrate the effectiveness of our methods, we conduct experiments using the Fashion-MNIST dataset [23]. In the first analysis, we use a total of $1 2 , 0 0 0 \ 2 \bar { 8 } { \times } 2 8$ grayscale images, consisting of 6,000 images each from the T-shirt/top class and the Trouser class. The left panel of Figure 4 gives a visualization of the data points using their first two principal components, showing the anisotropic and heterogeneous covariance structures. Since a large number of pixels have zero across most images, we apply PCA to reduce dimensionality from 784 to 50 by retaining the top 50 principal components. Our Algorithm 2 achieves a misclustering error of $5 . 7 1 \%$ , outperforming the vanilla Lloyd’s algorithm, which has an error of $8 . 2 4 \%$ . In the second analysis, we incorporate an additional class, the Ankle boot class, increasing the total to 18,000 images across three classes. Following the same preprocessing steps, the visualization of the dataset’s structure in the right panel of Figure 4 again confirms the presence of anisotropic and heterogeneous covariances. Here, Algorithm 2 achieves an error of $3 . 9 7 \%$ , an improvement over the $5 . 6 4 \%$ error rate observed with the vanilla Lloyd’s algorithm.

![](images/a0ba4571b9a9b72a6a043b5a2105ba35a045cadced134c27f3a77878993199ef.jpg)  
Figure 4: Visualization of the Fashion-MNIST dataset using the first two principal components. The data points are color-coded to indicate class membership: Red represents the T-shirt/top class, green denotes the Trouser class, and blue signifies the Ankle boot class. This illustration shows the existence of anisotropic and heterogeneous covariance structures.

# 5 Conclusion

This paper focuses on clustering methods and theory for GMMs, with anisotropic covariance structures, presenting new minimax bounds and an adjusted Lloyd’s algorithm tailored for varying covariance structures. Our theoretical and empirical analyses demonstrate the algorithm’s ability to achieve optimality within a logarithmic number of iterations. Despite these advances, our results have some limitations that are worth addressing in future work:

1. High-Dimensional Settings: Current results are restricted to dimensions $d$ growing at a rate slower than $n$ , specifically $\overset { \cdot } { d } = O ( \sqrt { n } )$ as stated in Theorem 2.2. Section 3 further requires a stronger assumption $d = O ( 1 )$ . These constraints stem from technical challenges in estimating covariance matrices accurately and in controlling matrix determinant. Adopting more sophisticated analytical tools could potentially relax these bounds to $d = O ( n )$ . In scenarios where $d$ exceeds $n$ , the misclustering error deviates from the simpler exponential decay observed under isotropic GMMs, as shown in [16]. This suggests that our model might also exhibit similar complexities, warranting further exploration into the technique used in [16] for potential extensions.

2. Ill-Conditioned Covariance Structures: Our analysis relies on the assumption of wellconditioned covariance matrices, where $\begin{array} { r } { \operatorname* { m a x } _ { a , b \in [ k ] } \lambda _ { d } ( \Sigma _ { a } ^ { * } ) / \lambda _ { 1 } ( \Sigma _ { b } ^ { * } ) = O ( 1 ) } \end{array}$ . This condition is crucial for the current analytical framework, as it helps manage the estimation errors of covariance matrices and their inverses. While more advanced techniques may allow for a relaxation of this assumption, handling ill-conditioned or degenerate covariance matrices remains challenging, particularly due to the difficulty of working with matrix inverses in such cases. While minimax lower bounds suggest that clustering is still possible even when the covariance matrix is degenerate, it raises computational challenges for our current algorithms. This highlights the need for developing new algorithms that can function effectively under less restrictive conditions.