# KOALA: Kernel Coupling and Element Imputation Induced Multi-View Clustering

Tingting $\mathbf { W _ { u } } ^ { * 1 , 2 }$ , Zhendong $\mathbf { L i } ^ { * 1 , 2 }$ , Zhibin $\mathbf { G } \mathbf { u } ^ { 3 }$ , Jiazheng Yuan4, Songhe Feng†1,2

1 Key Laboratory of Big Data & Artificial Intelligence in Transportation (Beijing Jiaotong University), Ministry of Education 2 School of Computer Science and Technology, Beijing Jiaotong University, Beijing, 100044, China 3 College of Computer and Cyber Security, Hebei Normal University, Hebei, China 4College of Science and Technology, Beijing Open University, Beijing, China {22120438, 23120371, guzhibin, shfeng} $@$ bjtu.edu.cn, jzyuan $@$ 139.com

# Abstract

Incomplete Multi-View Clustering (IMVC) has made significant progress by optimally merging multiple pre-specified incomplete views. Most existing IMVC algorithms operate under the assumption that view alignment is known, but in practice, the coupling information between views may be absent, thereby limiting the practical applicability of these methods. Being aware of this, we propose a novel IMVC method named Kernel cOupling And eLement imputAtion induced Multi-View Clustering (KOALA), which sufficiently explores the nonlinear relationship among features and optimally processes a group of kernels with missing and unaligned elements to simultaneously resolve multi-view clustering problem under both uncoupled and incomplete scenarios. Specifically, we first introduce a cross-kernel alignment learning strategy to reconstruct the coupling relationships among multiple kernels, which effectively captures high-order nonlinear relationships among samples and enhances alignment accuracy. Additionally, a low-rank tensor constraint is imposed on the optimizable alignment kernel tensor, facilitating the effective imputation of missing kernel elements by leveraging consistency information across views. Subsequently, we develop an alternative optimization approach with promising convergence to solve the resultant optimization problem. Extensive experimental results on various multi-view datasets demonstrate that the KOALA method achieves remarkable clustering performance.

# Introduction

Incomplete Multi-View Clustering (IMVC) has received considerable attention for exploring consistent and complementary knowledge from incomplete multi-view data (Hu and Chen 2019b; Liu et al. 2019b; Wen et al. 2019; Zhang et al. 2020; Liu et al. 2021; Wen et al. 2021; Lin et al. 2021, 2022b; Xu et al. 2022; Tang and Liu 2022; Zhao et al. 2023; Wen et al. 2023; Zhang et al. 2023b; Dong et al. 2023b,a; Jin et al. 2023). Generally speaking, existing IMVC methods can be categorized into three popular groups, including matrix factorization-based methods (Xu et al. 2018; Hu and Chen 2019a; Li et al. 2023; Zhang et al. 2023a; Xu et al.

2024a; Gu and Feng 2023; Gu, Li, and Feng 2024), graphbased methods (Wang et al. 2019; Wen et al. 2020; Li, Wan, and He 2021; Yan et al. 2022; Zhang et al. 2023c; Xu et al. 2024b; Gu et al. 2024) and kernel-based methods (Liu et al. 2019c,a, 2020; Zhang et al. 2021; Wan et al. 2022; Wang et al. 2024). i) Matrix factorization-based methods factorize each view data into two non-negative matrices and then attempt to find a unified view-consensus cluster partition. For example, partial multi-view clustering (Li, Jiang, and Zhou 2014) aims to identify a common latent representation for all views by assuming that samples are represented similarly across different views. ii) Graph-based methods first build undirected graphs using pairwise similarities between samples, followed by simultaneous spectral clustering and graph fusion. Typically, perturbation-oriented incomplete multiview clustering produces comprehensive similarity matrices and alleviates the difficulties associated with spectral tension. iii) Kernel-based methods typically attempt to derive a unified representation or multiple latent representations corresponding to all views by employing kernels that have been reconstructed from these views. For instance, the work in (Liu et al. 2018) proposes a late fusion approach to simultaneously clustering and impute the incomplete matrices.

While the aforementioned IMVC methods effectively extract rich information from incomplete multi-view data, realworld scenarios present challenges beyond mere missing information in some views. The misalignment of samples across views poses significant hurdles for existing methods in exploring complementarity and achieving consensus representation in multi-view data (Xie et al. 2018). Unfortunately, research on uncoupled incomplete multi-view data has been relatively scarce in recent years. As far as we know, Lin et al. (Lin et al. 2022a) proposed a pioneering solution that investigates the relationship between the inferred selfrepresentation matrix of misaligned data and the most reliable view. Despite advancement in handling uncoupled incomplete multi-view data, this method operates in Euclidean space, limiting its ability to capture only linear relationships between pairs of samples and between pairs of views. Consequently, high-order nonlinear relationships among samples and among views remain unexplored, leading to suboptimal clustering results.

To effectively cluster uncoupled incomplete multi-view data, this paper introduces a new multi-view clustering algorithm termed Kernel Coupling and Element Imputation Induced Multi-View Clustering. The framework of our method is illustrated in Fig. 1. Specifically, KOALA first learns multiple non-coupled and incomplete kernel matrices from the provided uncoupled incomplete multi-view data. Subsequently, we adopt a cross-kernel alignment learning strategy by introducing a pair of coupling matrices in the reproducing kernel Hilbert space. This approach facilitates exploration of high-order nonlinear relationships among samples while realigning disordered kernel elements. Additionally, we introduce a strategy for recovering missing kernel elements using low-rank tensor constraints to explore high-order correlations among all kernel matrices, effectively imputing absent kernel elements. Overall, the main contributions of this paper are as follows:

![](images/22628e88ada7714cc2eeecbf2234c7b9e3acb777045b65fb0f263356293c7007.jpg)  
Figure 1: The Framework of KOALA: A coupling-imputation-clustering framework for uncoupled incomplete multi-view clustering.

1) To the best of our knowledge, we are the first work to utilize a kernel-based method, which integrates information from multiple views and flexibly handles complex and nonlinear data relationships, to solve the uncoupled and incomplete multi-view clustering problem.

2) KOALA introduces a cross-kernel alignment learning strategy to reconstruct the coupling relationship among kernels. Furthermore, a low-rank tensor constraint is used on the optimizable alignment kernel tensor to capture the consistency information to impute missing kernel elements.

3) We employ a curvilinear search algorithm to solve the difficult optimization problem with orthogonality constraint, followed by a six-step alternative optimization method for KOALA. Comprehensive experimental results clearly demonstrate the effectiveness and efficiency of our proposed method.

# Preliminary

# A. Multiple Kernel $k$ -Means with Incomplete Kernels (MKKM-IK)

Given $\{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { n } \subseteq \mathcal { X }$ with $d$ dimensions and $n$ samples, and $\phi _ { p } ( \cdot ) : \textbf { x } \in \mathcal { X } \ \mapsto \ \mathcal { H } _ { p }$ denotes the $p$ th feature mapping that maps $\mathbf { X }$ onto a reproducing kernel Hilbert space $\mathcal { H } _ { p }$ $( 1 ~ \leq ~ p ~ \leq ~ m )$ . The kernel matrix $\mathbf { K } _ { p }$ can be expressed as $K _ { i , j } ~ = ~ \phi _ { p } ( \mathbf { x } _ { i } ) ^ { \top } \phi _ { p } ( \mathbf { x } _ { j } ) ~ = ~ \kappa _ { p } ( \grave { \mathbf { x } _ { i } } , \mathbf { x } _ { j } )$ . In a multiple kernel setting, each sample is depicted as $\begin{array} { r l r } { \phi _ { \beta } ( { \bf x } ) } & { { } = } & { [ \beta _ { 1 } \phi _ { 1 } ( { \bf x } ) ^ { \top } , \ldots , \beta _ { m } \phi _ { m } ( { \bf x } ) ^ { \top } ] ^ { \top } } \end{array}$ where $\beta \ =$ $[ \beta _ { 1 } , \ldots , \beta _ { m } ] ^ { \top }$ consists of the coefficients of the $m$ base kernels $\{ \kappa _ { p } ( \cdot , \cdot ) \} _ { p = 1 } ^ { m }$ . Based on the definition of $\phi _ { \beta } ( \mathbf { x } )$ , the kernel function is denoted as $\kappa _ { \beta } ( \mathbf { x } _ { i } , \mathbf { x } _ { j } ) = \phi _ { \beta } ( \mathbf { x } _ { i } ) ^ { \top } \phi _ { \beta } ( \mathbf { x } _ { j } ) =$ $\begin{array} { r } { \sum _ { p = 1 } ^ { m } \beta _ { p } ^ { 2 } \kappa _ { p } ( \mathbf { x } _ { i } , \mathbf { x } _ { j } ) } \end{array}$ . Let $\{ \mathbf { K } _ { p } \} _ { p = 1 } ^ { m }$ denote a group of kernel matrices pre-calculated from coupled and incomplete multiview data with $m$ views. As noted in $\mathrm { \Delta Y u }$ et al. 2011; Go¨nen and Margolin 2014), the fused kernel matrix $\mathbf { K } _ { \beta }$ can be described as $\begin{array} { r } { \mathbf { K } _ { \beta } = \sum _ { p = 1 } ^ { m } \beta _ { p } ^ { 2 } \mathbf { K } _ { p } } \end{array}$ , so the general formulation of MKKM-IK (L u et al. 2019c) is

$$
\begin{array} { r l } &  \underset { \{ \mathbf { H } , \beta , \} , \{ \mathbf { K } _ { p } \} _ { p = 1 } ^ { m } \} { \operatorname* { m i n } } \mathbf { T r } ( \mathbf { K } _ { \beta } ( \mathbf { I } _ { n } - \mathbf { H H } ^ { \top } ) ) , } \\ & { s . t . \mathbf { H } \in \mathbb { R } ^ { n \times k } , \mathbf { H } ^ { \top } \mathbf { H } = \mathbf { I } _ { k } , \beta ^ { \top } \mathbf { 1 } _ { m } = 1 , \beta _ { p } \geq 0 , } \end{array}
$$

$$
\mathbf { K } _ { p } ( s _ { p } , s _ { p } ) = \mathbf { K } _ { p } ^ { ( c c ) } , \mathbf { K } _ { p } \succeq 0 , \forall p ,
$$

where $s _ { p } ( 1 \leq p \leq m )$ denotes the index of the sample presents in the p-th kernel and K(pcc) r epresents the kernel sub-matrix computed with the corresponding samples. The constraint ${ \bf K } _ { p } ( s _ { p } , s _ { p } ) \ = \ { \bf K } _ { p } ^ { ( c c ) }$ is imposed to ensure that $\mathbf { K } _ { p }$ preserves the known entries throughout the procedure. Moreover, The matrix $\mathbf { H }$ denotes the cluster partition matrix.

# $B .$ . Tensor Nuclear Norm Theory

Let $\mathcal { Y } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ be a tensor, its block circulant matrix $b c i r c ( \mathcal { V } ) \in \mathbb { R } ^ { n _ { 1 } n _ { 3 } \times n _ { 2 } n _ { 3 } }$ as

$$
b c i r c ( \mathscr { Y } ) = \left( \begin{array} { c c c c } { Y ^ { ( 1 ) } } & { Y ^ { ( n _ { 3 } ) } } & { . . . } & { Y ^ { ( 2 ) } } \\ { Y ^ { ( 2 ) } } & { Y ^ { ( 1 ) } } & { . . . } & { Y ^ { ( 3 ) } } \\ { \vdots } & { \vdots } & { . . } & { \vdots } \\ { Y ^ { ( n _ { 3 } ) } } & { Y ^ { ( n _ { 3 } - 1 ) } } & { . . . } & { Y ^ { ( 1 ) } } \end{array} \right) .
$$

Also, the block circular vectorization and its inverse operation are

$$
b v e c ( \mathcal { Y } ) = \binom { Y ^ { ( 1 ) } } { Y ^ { ( 2 ) } } , b v f l o d ( b v e c ( \mathcal { Y } ) ) = \mathcal { Y } .
$$

Then, some related definitions are introduced below (Xie et al. 2018).

Definition 1 (t-product) Let $\mathcal { A } \ \in \ \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ and $B \ \in$ $\mathbb { R } ^ { n _ { 2 } \times n _ { 4 } \times n _ { 3 } }$ , the tensor $S \in \mathbb { R } ^ { n _ { 1 } \times n _ { 4 } \times n _ { 3 } }$ is obtained by $t \cdot$ - product of $\mathcal { A }$ and $\boldsymbol { B }$

$$
\mathcal { S } = \mathcal { A } \ast \mathcal { B } = b v f o l d ( b c i r c ( \mathcal { A } ) b v e c ( \mathcal { B } ) ) ,
$$

Definition 2 (t-SVD) Given a tensor $\mathcal { V } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ , it can be factorized by t-SVD as $\mathcal { V } = \mathcal { U } * S * \mathcal { V } ^ { \top }$ , where $\boldsymbol { \mathcal { U } } \in$ $\mathbb { R } ^ { n _ { 1 } \times n _ { 1 } \times n _ { 3 } }$ and $\boldsymbol { \mathcal { V } } \in \mathbb { R } ^ { n _ { 2 } \times n _ { 2 } \times n _ { 3 } }$ are orthogonal, and $s$ is an $f$ -diagonal tensor whose size is $n _ { 1 } \times n _ { 2 } \times n _ { 3 }$ .

Definition 3 (t-TNN) The $t$ -SVD based tensor nuclear norm $\mathit { \Pi } _ { t }$ -TNN) is given by

$$
\left. \mathcal { V } \right. _ { * } : = \sum _ { i = 1 } ^ { m i n ( n _ { 1 } , n _ { 2 } ) } \sum _ { k = 1 } ^ { n _ { 3 } } \lvert S _ { f } ( i , i , k ) \rvert ,
$$

where $S _ { f } = f f t ( S , \mathbb { I } , 3 )$ is the Fourier transform along the third dimension.

# Proposed Method

This section introduces our proposed approach, which addresses the simultaneous tasks of imputing missing elements and aligning uncoupled values. Subsequently, we will provide detailed explanations of two sub-modules within the KOALA method: the cross-kernel alignment learning strategy and the missing kernel element recovery strategy.

Cross-kernel alignment learning strategy. As mentioned before, one of the main challenges is the unknown arrangement information among kernel matrices, which hinders view fusion. Thereafter, a key issue lies in how to effectively reestablish the cross-kernel correspondence relationship under the incomplete scenario?

Existing coupling methods generally function in Euclidean space and neglect high-order nonlinear relationships among samples and among views, resulting in suboptimal clustering performance. To solve this issue, KOALA conducts elements alignment in reproducing kernel Hilbert space. Precisely, let $\{ \hat { \mathbf { K } } _ { p } \} _ { p = 1 } ^ { m }$ represent a collection of noncoupled and incomplete kernel matrices obtained from uncoupled incomplete multi-view data. Inspired by (Lin et al. 2022a), we explicitly introduce a set of coupling matrices $\mathbf { B } _ { p } \in \mathbb { R } ^ { n \times n } ( \bar { 1 } \leq p ^ { \prime } \leq m )$ to realign the uncoupled kernel matrices. This strategy randomly selects $\hat { \mathbf { K } } _ { q }$ ( $1 \leq q \leq m )$ as the most reliable kernel and uses the following constraint to recouple the kernel matrix of each view with the $q$ -th kernel:

$$
\left\| \mathbf { B } _ { p } ^ { \top } \hat { \mathbf { K } } _ { p } \mathbf { B } _ { p } - \hat { \mathbf { K } } _ { q } \right\| _ { F } ^ { 2 } ,
$$

By reconstructing the coupling relationship among kernels, each paired kernel matrix $\mathbf { B } _ { p } ^ { \top } \hat { \mathbf { K } } _ { p } \mathbf { B } _ { p }$ can be obtained, and the fused kernel matrix $\hat { \mathbf { K } } _ { \beta }$ can be derived as $\hat { \bf K } _ { \beta } \ = \ $ $\begin{array} { r } { \sum _ { p = 1 } ^ { m } \beta _ { p } ^ { 2 } \mathbf { B } _ { p } ^ { \top } \hat { \mathbf { K } } _ { p } \mathbf { B } _ { p } } \end{array}$ . This strategy maps complex nonlinear relationships in the original data into kernel space, making multi-view data more linearly separable, which captures the high-order nonlinear relationships sufficiently to reconstruct the alignment correlations among kernel matrices.

Missing kernel elements recovery strategy. For the incompleteness of kernel matrices, we propose a novel kernel tensor-based Missing Kernel Elements Recovery Strategy (MLR). This strategy focuses on how to efficiently find significant information to fill in missing kernel elements, which sufficiently considers intra-kernel correlations and high-order correlations among different kernels. Unlike traditional imputation strategies that operate at the kernel matrix level, MLR constructs a creative optimizable tensor representation and imposes a global low-rank constraint on the tensor. This approach captures high-order correlations among multiple kernels, thereby enhancing the accuracy of imputing missing kernel elements. Specifically, we construct a optimizable alignment kernel tensor structure defined as follows:

Definition 4 (Optimizable alignment kernel tensor) Given $a$ set of coupled kernel matrices $\{ \mathbf { B } _ { p } ^ { \top } \hat { \mathbf { K } } _ { p } \mathbf { B } _ { p } \} _ { p = 1 } ^ { m }$ , the optimizable alignment kernel tensor is assembled by stacking the kernel matrices of different views, that is

$\mathcal { K } = \Phi ( \mathbf { B } _ { 1 } ^ { \top } \hat { \mathbf { K } } _ { 1 } \mathbf { B } _ { 1 } , \mathbf { B } _ { 2 } ^ { \top } \hat { \mathbf { K } } _ { 2 } \mathbf { B } _ { 2 } , \cdot \cdot \cdot , \mathbf { B } _ { m } ^ { \top } \hat { \mathbf { K } } _ { m } \mathbf { B } _ { m } ) ,$ (5) where $\{ \mathbf { B } _ { p } \} _ { p = 1 } ^ { m }$ are optimizable coupling matrices and $\Phi ( \cdot )$ represents a mapping function that merges the kernel matrices $\mathbf { B } _ { p } ^ { \top } \hat { \mathbf { K } } _ { p } \mathbf { B } _ { p }$ into a 3-dimension tensor of size $n \times n \times m$ .

Based on Eq. (5), the incomplete optimizable alignment kernel tensor $\kappa$ is formed, and $\lVert K \rVert _ { * } ^ { - }$ (definition 3) is applied to its rotated form ${ \mathcal { K } } \in \mathbb { R } ^ { n \times m \times n }$ , aiming to capture high-order correlations. The use of t-TNN facilitates the imputation of missing elements by capturing the global structure and latent patterns within $\mathbf { K } _ { p }$ . Introducing the low-rank approximation on $\kappa$ reduces model complexity and allows for the rational inference and completion of missing values based on the global trends and structure of the observed data. In short, this strategy implicitly fills in missing elements in the kernel matrix by effectively exploiting high-order correlations among incomplete multi-view data.

By integrating Eq. (4) and the low-rank tensor constraint $\| \kappa \| _ { * }$ with Eq. (1), the objective formulation of our proposed KOALA is as follows:

$$
\begin{array} { r } { \underset { \{ \hat { \mathbf { K } } _ { p } , \mathbf { B } _ { p } \} _ { p = 1 } ^ { m } , \beta , \mathbf { H } } { \operatorname* { m i n } } \mathbf { T r } ( \hat { \mathbf { K } } _ { \beta } ( \mathbf { I } _ { n } - \mathbf { H H } ^ { \top } ) ) + \lambda \left. \boldsymbol { K } \right. _ { * } } \\ { + \theta \underset { p = 1 } { \overset { m } { \sum } } \left. \mathbf { B } _ { p } ^ { \top } \hat { \mathbf { K } } _ { p } \mathbf { B } _ { p } - \hat { \mathbf { K } } _ { q } \right. _ { F } ^ { 2 } , } \end{array}
$$

$$
\boldsymbol { : } , \boldsymbol { \mathcal { K } } = \Phi ( \mathbf { B } _ { 1 } ^ { \top } \hat { \mathbf { K } } _ { 1 } \mathbf { B } _ { 1 } , \mathbf { B } _ { 2 } ^ { \top } \hat { \mathbf { K } } _ { 2 } \mathbf { B } _ { 2 } , \boldsymbol { \cdot } \boldsymbol { \cdot } , \mathbf { B } _ { m } ^ { \top } \hat { \mathbf { K } } _ { m } \mathbf { B } _ { m } ) ,
$$

$$
\mathbf { H } \in \mathbb { R } ^ { n \times k } , ~ \mathbf { H } ^ { \top } \mathbf { H } = \mathbf { I } _ { k } , ~ \boldsymbol { \beta } ^ { \top } \mathbf { 1 } _ { m } = 1 , ~ \beta _ { p } \geq 0 ,
$$

$$
\hat { \bf K } _ { p } ( { \bf s } _ { p } , { \bf s } _ { p } ) = \hat { \bf K } _ { p } ^ { ( c c ) } , \hat { \bf K } _ { \beta } = \sum _ { p = 1 } ^ { m } \beta _ { p } ^ { 2 } { \bf B } _ { p } ^ { \top } \hat { \bf K } _ { p } { \bf B } _ { p } ,
$$

$$
\hat { \mathbf { K } } _ { p } \geq 0 , ~ \mathbf { B } _ { p } \geq 0 , \mathbf { B } _ { p } ^ { \top } \mathbf { B } _ { p } = \mathbf { I } _ { n } , ~ \forall p ,
$$

where $\theta$ and $\lambda$ are hyperparameters, $\| \mathcal { K } \| _ { * }$ denotes the nuclear norm of the tensor. In this paper, we simply set $q = 1$ by assigning the first kernel as the most reliable kernel.

As seen from Eq. (6), the missing kernel elements are first filled with zero values, and then the cross-kernel alignment learning strategy is used to rebuild the coupling relationship among kernels. Subsequently, we create the optimizable alignment kernel tensor $\kappa$ and employ the missing kernel elements recovery strategy to impute the missing kernel elements. In conclusion, we present a joint couplingimputation-clustering framework where the kernel matrices $\{ \hat { \mathbf { K } } _ { p } \} _ { p = 1 } ^ { m }$ , the coupling matrices $\{ \mathbf { B } _ { p } \} _ { p = 1 } ^ { m }$ and the clustering partition matrix $\mathbf { H }$ can be jointly optimized to enhance the final clustering performance.

# Optimization

Inspired by the alternating direction method of multipliers (ADMM) (Lin, Liu, and Su 2011), we introduce two auxiliary variables, $\mathcal { G } = \kappa$ and $\{ \mathbf { M } \} _ { p = 1 } ^ { m } = \{ \mathbf { B } \} _ { p = 1 } ^ { m }$ to make the variables separable. The Lagrange function of Eq. (6) can be expressed as the following unconstrained problem.

$$
\begin{array} { r l } {  { \mathcal { L } ( \mathbf { H } ; \{ \hat { \mathbf { K } } \} _ { p = 1 } ^ { m } ; \{ \mathbf { B } \} _ { p = 1 } ^ { m } ; \mathcal { G } ; \{ \mathbf { M } \} _ { p = 1 } ^ { m } ; \beta ) } } \\ & { = \mathbf { T r } ( \sum _ { p = 1 } ^ { m } \beta _ { p } ^ { 2 } \mathbf { B } _ { p } ^ { \top } \hat { \mathbf { K } } _ { p } \mathbf { B } _ { p } ( \mathbf { I } _ { n } - \mathbf { H } \mathbf { H } ^ { \top } ) ) + \lambda \| \mathcal { G } \| _ { * } +  \mathcal { W } , \mathcal { K } - \mathcal { G }  } \\ & { + \displaystyle \frac { \rho } { 2 } \| \mathcal { K } - \mathcal { G } \| _ { F } ^ { 2 } + \theta \sum _ { p = 1 } ^ { m } \| \mathbf { B } _ { p } ^ { \top } \hat { \mathbf { K } } _ { p } \mathbf { B } _ { p } - \hat { \mathbf { K } } _ { q } \| _ { F } ^ { 2 } } \\ & { + \displaystyle \sum _ { p = 1 } ^ { m } (  \mathbf { J } _ { p } , \mathbf { B } _ { p } - \mathbf { M } _ { p }  + \frac { \mu } { 2 } \| \mathbf { B } _ { p } - \mathbf { M } _ { p } \| _ { F } ^ { 2 } ) . } \end{array}
$$

Here, the tensor $\boldsymbol { \mathcal { W } }$ and the matrix $\{ \mathbf { J } _ { p } \} _ { p = 1 } ^ { m }$ serve as Lagrange multipliers, while $\rho$ and $\mu$ are penalty parameters controlling convergence. Then Eq. (6) is transformed into solving the following subproblems in a distinct manner.

# Optimization H

Optimizing $\mathbf { H }$ is equivalent to solving the following problem if other variables are fixed:

$$
\operatorname* { m i n } _ { \mathbf { H } } \mathbf { T r } ( \hat { \mathbf { K } } _ { \beta } ( \mathbf { I } _ { n } - \mathbf { H H } ^ { \top } ) ) , ~ s . t . ~ \mathbf { H } \in \mathbb { R } ^ { n \times k } , ~ \mathbf { H } ^ { \top } \mathbf { H } = \mathbf { I } _ { k } .
$$

The Eq. (8) in optimizing $\mathbf { H }$ is a traditional $k$ -means clustering problem that can be solved by existing packages.

# Optimization $\hat { \mathbf { K } } _ { p }$

Fixing the other variables derives the subproblem for $\hat { \mathbf { K } } _ { p }$

$$
\begin{array} { l } { \displaystyle \underset { \hat { \mathbf { K } } _ { p } } { \operatorname* { m i n } } \mathbf { T r } ( \beta _ { p } ^ { 2 } \mathbf { B } _ { p } ^ { \top } \hat { \mathbf { K } } _ { p } \mathbf { B } _ { p } ( \mathbf { I } _ { n } - \mathbf { H H } ^ { \top } ) ) + \langle \mathbf { W } _ { p } , \hat { \mathbf { K } } _ { p } - \mathbf { G } _ { p } \rangle } \\ { \displaystyle \quad + \frac { \rho } { 2 } \left\| \hat { \mathbf { K } } _ { p } - \mathbf { G } _ { p } \right\| _ { F } ^ { 2 } + \theta \displaystyle \sum _ { p = 1 } ^ { m } \left\| \mathbf { B } _ { p } ^ { \top } \hat { \mathbf { K } } _ { p } \mathbf { B } _ { p } - \hat { \mathbf { K } } _ { q } \right\| _ { F } ^ { 2 } , } \\ { \displaystyle \quad { s . t . } \hat { \mathbf { K } } _ { p } ( \mathbf { s } _ { p } , \mathbf { s } _ { p } ) = \hat { \mathbf { K } } _ { p } ^ { ( c c ) } , \hat { \mathbf { K } } _ { p } \succeq 0 , \forall p . } \end{array}
$$

By setting the derivative of Eq. (9), the closed form of $\hat { \mathbf { K } } _ { p }$ can be obtained by

$$
\hat { \mathbf { K } } _ { p } = \frac { \rho \mathbf { G } _ { p } + 2 \theta \mathbf { B } _ { p } \hat { \mathbf { K } } _ { q } \mathbf { B } _ { p } ^ { \top } - \mathbf { W } _ { p } - \beta _ { p } ^ { 2 } \mathbf { B } _ { p } ^ { \top } ( \mathbf { I } _ { n } - \mathbf { H } \mathbf { H } ^ { \top } ) \mathbf { B } _ { p } } { \rho + 2 \theta } .
$$

# Optimization $\mathcal { G }$

If $\hat { \mathbf { K } } _ { p }$ and $\mathbf { B } _ { p } ( p = 1 , 2 , \ldots , m )$ are fixed, for updating the tensor $\mathcal { G }$ , the problem can be written as

$$
\boldsymbol { \mathcal { G } ^ { * } } = \arg \operatorname* { m i n } _ { \boldsymbol { \mathcal { G } } } \boldsymbol { \lambda } \left\| \boldsymbol { \mathcal { G } } \right\| _ { * } + \frac { \rho } { 2 } \left\| \boldsymbol { \mathcal { G } } - ( K + \frac { 1 } { \rho } \mathcal { W } ) \right\| _ { F } ^ { 2 } .
$$

The following theorem can be used to solve the optimization problem Eq. (11).

Theorem 1 (Xie et al. 2018): Let $\tau \ > \ 0$ , and $\mathcal { G } ^ { \mathrm { ~ ~ } } \in$ $\mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ , $\mathcal { F } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ , the best possible solution on a global scale for $\begin{array} { r } { \operatorname* { m i n } _ { \mathcal { G } } \tau \left\| \mathcal { G } \right\| _ { * } + \frac { 1 } { 2 } \left\| \mathcal { G } - \mathcal { F } \right\| _ { F } ^ { 2 } } \end{array}$ is given by the tensor tubal-shrinkage op∗erator ${ \mathcal { G } } \ = \ { \mathcal { C } } _ { n 3 \tau } ( { \underline { { \mathcal { F } } } } ) \ =$ $\bar { \mathcal { U } } * \mathcal { C } _ { n 3 \tau } ( S ) * \mathcal { V } ^ { \top }$ . Noticed that $\mathcal { F } = \mathcal { U } \ast \mathcal { S } \ast \mathcal { V } ^ { \intercal }$ and $\mathcal { C } _ { n 3 \tau } \ : = \ : S \ast \mathcal { T }$ , where $\mathcal { I } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ represents an fmain being $\begin{array} { r } { \mathcal { I } _ { f } ( i , i , j ) = ( \bar { 1 } ^ { - } \frac { n 3 \tau } { S _ { f } ^ { ( j ) } } ( i , i ) ) _ { + } } \end{array}$ intheFfoaie o

# Optimization $\mathbf { M } _ { p }$

By fixing all variables to expect $\mathbf { M } _ { p }$ , the problem in Eq. (7) can be written as

$$
\operatorname* { m i n } _ { \mathbf { M } _ { p } } \langle \mathbf { J } _ { p } , \mathbf { B } _ { p } - \mathbf { M } _ { p } \rangle + \frac { \mu } { 2 } \left. \mathbf { B } _ { p } - \mathbf { M } _ { p } \right. _ { F } ^ { 2 } .
$$

By differentiating Eq. (12) and setting it to zero, the optimal solution $\mathbf { M } _ { p }$ can be obtained:

$$
\mathbf { M } _ { p } = \frac { \mathbf { J } _ { p } + \mu \mathbf { B } _ { p } } { \mu } ,
$$

# Optimization $\mathbf { B } _ { p }$

When other variables are fixed, the optimization for $\mathbf { B } _ { p }$ is formulated as

$$
\begin{array} { r l } & { \displaystyle \underset { { \bf B } _ { p } } { \operatorname* { m i n } } \beta _ { p } ^ { 2 } { \bf T r } ( { \bf B } _ { p } ^ { \top } \hat { \bf K } _ { p } { \bf B } _ { p } ( { \bf I } _ { n } - { \bf H } { \bf H } ^ { \top } ) ) + \theta \left\| { \bf B } _ { p } ^ { \top } \hat { \bf K } _ { p } { \bf B } _ { p } - \hat { \bf K } _ { q } \right\| _ { F } ^ { 2 } } \\ & { + \left. { \bf J } _ { p } , { \bf B } _ { p } - { \bf M } _ { p } \right. + \frac { \mu } { 2 } \left\| { \bf B } _ { p } - { \bf M } _ { p } \right\| _ { F } ^ { 2 } , } \\ & { s . t . { \bf B } _ { p } \geq 0 , { \bf B } _ { p } ^ { \top } { \bf B } _ { p } = { \bf I } _ { n } . } \end{array}
$$

Let $\mathbf { Z } _ { p } = \mathbf { J } _ { p } - \mu \mathbf { M } _ { p }$ , $\mathbf O _ { p } = \beta _ { p } ^ { 2 } ( \mathbf I _ { n } - \mathbf H \mathbf H ^ { \top } ) - 2 \theta \hat { \mathbf K } _ { q }$ . Due to the orthogonality constraint of $\mathbf { B } _ { p }$ , a curvilinear search algorithm is applied to optimize $\mathbf { B } _ { p }$ according to (Wen and Yin 2013). We first define $\mathcal { F } ( \mathbf { B } _ { p } ) = \mathbf { T r } ( \mathbf { O } _ { p } \mathbf { B } _ { p } ^ { \top } \hat { \mathbf { K } } _ { p } \mathbf { B } _ { p } +$ $\mathbf { B } _ { p } ^ { \top } \mathbf { Z } _ { p } )$ and rewrite Eq. (14) as

$$
\operatorname* { m i n } _ { \mathbf { B } _ { p } } \mathcal { F } ( \mathbf { B } _ { p } ) , \quad s . t . \mathbf { B } _ { p } ^ { \top } \mathbf { B } _ { p } = \mathbf { I } _ { n } .
$$

We define $\begin{array} { r } { \mathbf { G } = \frac { \partial \mathcal { F } } { \partial \mathbf { B } _ { p } } } \end{array}$ and $\mathbf { A } = \mathbf { G } \mathbf { B } _ { p } ^ { \top } - \mathbf { B } _ { p } \mathbf { G } ^ { \top }$ . To preserve the orthogonality constraint, the next iteration of $\mathbf { B } _ { p }$ can be obtained according to (Wen and Yin 2013) as follows,

$$
\hat { \mathbf { B } } _ { p } ( \tau ) = \mathbf { B } _ { p } - \frac { \tau } { 2 } \mathbf { A } ( \mathbf { B } _ { p } + \hat { \mathbf { B } } _ { p } ( \tau ) ) .
$$

Then we can easily do the calculation

$$
\hat { \mathbf { B } } _ { p } ( \tau ) = ( \mathbf { I } _ { n } + \frac { \tau } { 2 } \mathbf { A } ) ^ { \dagger } ( \mathbf { I } _ { n } - \frac { \tau } { 2 } \mathbf { A } ) \mathbf { B } _ { p } ,
$$

where $\tau \in \mathrm { R }$ is a step size, $\hat { \mathbf { B } } _ { p } ( \tau )$ can be solved since orthogonality is preserved by any skew-symmetric matrix $\mathbf { A }$ .

Input: $\{ \hat { \mathbf { K } } _ { p } ^ { ( c c ) } \} _ { p = 1 } ^ { m }$ , $\{ \mathbf { s } _ { p } \} _ { p = 1 } ^ { m }$ , λ, $\theta$ and $\epsilon _ { 0 } , \rho ~ = ~ \mu ~ = ~ 1 0 ^ { - 5 }$ , ρmax = µmax = 1010;   
Initialize: Initialize $\beta ^ { ( 0 ) } = \mathbf { 1 } _ { m } / m , \hat { \mathbf { K } } _ { p } ^ { ( 0 }$ ) and $ t = 1 , \mathcal { G } = \mathcal { W } = 0$ , $\{ { \bf B } _ { p } \} _ { p = 1 } ^ { m } = { \bf I } _ { n }$ , $\{ \mathbf { M } _ { p } \} _ { p = 1 } ^ { m } = \{ \mathbf { J } _ { p } \} _ { p = 1 } ^ { m } = \mathbf { 0 }$ ;   
1: while not converge do   
2: $\begin{array} { r } { \hat { \mathbf { K } } _ { \beta } ^ { ( t ) } = \sum _ { p = 1 } ^ { m } ( \beta _ { p } ^ { ( t - 1 ) } ) ^ { 2 } \mathbf { B } _ { p } ^ { \top ( t - 1 ) } \hat { \mathbf { K } } _ { p } ^ { ( t - 1 ) } \mathbf { B } _ { p } ^ { ( t - 1 ) } ; } \end{array}$ ;   
3: Update $\mathbf { H } ^ { t }$ by solving a kernel $\mathbf { k }$ -means clustering optimization problem Eq. (8)   
4: Update $\hat { \mathbf { K } } _ { p } ^ { ( t ) }$ by Eq. (10);   
5: Update $\mathcal { G } ^ { ( t ) }$ via subproblem Eq. (11) ;   
6: Update $\mathcal { W } ^ { ( t ) }$ by Eq. (20);   
7: Update $\mathbf { M } _ { p } ^ { ( t ) }$ by Eq. (13);   
8: Update $\mathbf { J } _ { p } ^ { ( t ) }$ according to Eq. (21);   
9: Update $\mathbf { B } _ { p } ^ { ( t ) }$ by Eq. (18);   
10: Update $\beta ^ { ( t ) }$ by Eq. (19);   
11: Check the convergence conditions: $o b j ^ { ( t ) } - o b j ^ { ( t - 1 ) } \leq \epsilon _ { 0 }$ ; 12: end while   
13: Output: H.

To reduce the complexity of the matrix inversion, we future improve Eq. (17) according to (Zhang et al. 2022). Let $\mathbf { U } = [ \mathbf { G } , \mathbf { B } _ { p } ]$ and $\mathbf { V } = [ \mathbf { B } _ { p } , \mathbf { G } ]$ , then the SMW formula is applied to $\begin{array} { r } { { \bf { I } } _ { n } + \frac { \tau } { 2 } { \bf A } = { \bf { I } } _ { n } + \frac { \tau } { 2 } { \bf { U } } { \bf { V } } ^ { \top } } \end{array}$ and $\begin{array} { r } { ( { \bf I } _ { n } + \frac { \tau } { 2 } { \bf A } ) ^ { \dag } = } \end{array}$ $\begin{array} { r } { \mathbf I _ { n } - \frac { \tau } { 2 } \mathbf U ( \mathbf I _ { n } + \frac { \tau } { 2 } \mathbf V ^ { \top } \mathbf U ) ^ { \dag } \mathbf V ^ { \top } } \end{array}$ . After that, Eq. (17) can be rewritten as

$$
\hat { \mathbf { B } } _ { p } ( \tau ) = \mathbf { B } _ { p } - \tau \mathbf { U } ( \mathbf { I } _ { n } + \frac { \tau } { 2 } \mathbf { V } ^ { \top } \mathbf { U } ) ^ { \dagger } \mathbf { V } ^ { \top } \mathbf { B } _ { p } ,
$$

where $\tau$ is selected using a one-dimensional line searching strategy.

# Optimization $\beta$

Fixing the other variables, the optimization in Eq. (6) regarding $\beta$ is transformed to

$$
\begin{array} { r l } & { \underset { \boldsymbol { \beta } } { \operatorname* { m i n } } \underset { \boldsymbol { p } = 1 } { \sum } \beta _ { p } ^ { 2 } \mathbf { T r } ( \mathbf { B } _ { p } ^ { \top } \hat { \mathbf { K } } _ { p } \mathbf { B } _ { p } ( \mathbf { I } _ { n } - \mathbf { H H } ^ { \top } ) ) , } \\ & { s . t . \boldsymbol { \beta } ^ { \top } \mathbf { 1 } _ { m } = 1 , \ \beta _ { p } \geq 0 , \ \forall p . } \end{array}
$$

We summarize the solution of Eq. (6) in Algorithm 1. In addition, the Lagrange multipliers $\boldsymbol { \mathcal { W } }$ and $\mathbf { G } _ { p }$ can be optimized as follows,

$$
\begin{array} { c } { \mathcal { W } ^ { * } = \mathcal { W } + \rho ( \mathcal { K } - \mathcal { G } ) , } \\ { \mathbf { J } _ { p } = \mathbf { J } _ { p } + \mu ( \mathbf { B } _ { p } - \mathbf { M } _ { p } ) . } \end{array}
$$

# Experiment

# Experimental Settings

Datasets: We assessed the performance of our method using seven benchmark multi-view datasets, covering various sizes and numbers of clusters: ORL, proteinFold, Cora, UCI Digits, Caltech101-7, Caltech101-20, and Caltech101-all. The detailed information for each dataset is presented in Table 1.

Table 1: Datasets used in our experiments.   

<html><body><table><tr><td>Datasets</td><td>Samples</td><td>Views</td><td>Classes</td></tr><tr><td>ORL</td><td>400</td><td>3</td><td>40</td></tr><tr><td>proteinFold</td><td>694</td><td>12</td><td>27</td></tr><tr><td>Caltech101-7</td><td>1474</td><td>6</td><td>7</td></tr><tr><td>UCI_Digits</td><td>2000</td><td>6</td><td>10</td></tr><tr><td>Caltech101-20</td><td>2386</td><td>6</td><td>20</td></tr><tr><td>Cora</td><td>2708</td><td>2</td><td>7</td></tr><tr><td>Caltech101-all</td><td>9144</td><td>6</td><td>102</td></tr></table></body></html>

Comparison methods: There are several commonly used imputation approaches including zero filling (ZF), mean filling (MF), and $k$ -nearest-neighbor filling (KNN). The widely used Multiple Kernel $k$ -Means (MKKM) (Go¨nen and Margolin 2014) is combined with the above imputation methods to form ”two-stage” kernel-based methods, and KOALA is compared with these two-stage methods, including $\mathbf { M K K M + Z F }$ and MKKM+MF. Similarly, we compare our method with ”one-stage” kernel-based approaches that optimize both imputation and clustering simultaneously. These are LRKT-IMVC (Wu, Feng, and Yuan 2024) and the variants of MKKM-IK (Liu et al. 2019c) include MKKM$\mathrm { I K + M F } ,$ , MKKM- $\mathrm { . I K + Z F }$ , MKKM- $\mathbf { \partial } _ { \cdot } \operatorname { I K + K N N }$ , and MKKMIK-MKC. In addition to the kernel-based IMKC methods, UMIC (Lin et al. 2022a), a technique specifically designed for uncoupled incomplete multi-view data, is also used for comparison with our approach. For a fair comparison, we directly employ the source codes provided by the respective literature.

Uncoupled incomplete data construction: In this article, the underlying assumption is that there is at least one view available for each sample. The proportion of samples with missing views is controlled by the missing ratio parameter $\epsilon$ . Specifically, we set $\epsilon$ to $[ 0 . 1 : 0 . 2 : 0 . 5 ]$ for all datasets. As the mapping relationship of uncoupled data is unknown, and not all instances of uncoupled data are completely independent in the real world, we randomly shuffle the data arrangement for each view to create the uncoupled incomplete multi-view data in our experiments. Additionally, we randomly generate the ’incomplete and uncoupled patterns ten times and present the statistical results.

Evaluation metrics: To evaluate clustering performance, five commonly used criteria are employed: clustering Accuracy (ACC), Purity, Adjusted Rand Index (ARI), Fscore, and Precision. Due to space limitations, only partial results are presented in Table 2. More experimental results are presented in the supplementary materials.

# Experimental Results

Overall Clustering Performances. We evaluate the clustering performance of all the comparison methods mentioned above, along with the proposed KOALA, considering different missing ratios. The partial results are listed in Table 2. The best results are highlighted in bold, and the second best results are underlined. From Table 2, three observations can be made.

Table 2: Partial results (mean(std)) of KOALA and other compared methods on seven datasets. ’\*’ represents ’MKKM-’.   

<html><body><table><tr><td>Dataset</td><td>E</td><td>Metric(%)</td><td>*ZF</td><td>*MF</td><td>LRKT-IMVC</td><td>*IK-ZF</td><td>*IK-MF</td><td>*IK-KNN</td><td>*IK-MKC</td><td>UIMC</td><td>Ours</td><td></td></tr><tr><td rowspan="4">ORL</td><td>0.1</td><td>ACC Purity Fscore</td><td>16.18 (0.60) 16.77 (0.69) 2.60 (0.20)</td><td>16.15 (0.70) 16.98 (0.81) 2.56 (0.26)</td><td>16.63 (0.60) 17.28 (0.70) 2.65 (0.24)</td><td>16.82 (0.92) 17.25 (0.89) 2.72 (0.37)</td><td>16.65 (0.80) 17.17 (0.71) 2.77 (0.34)</td><td>16.57 (0.74) 17.15 (0.78) 2.62 (0.35)</td><td>16.43 (0.69) 16.95 (0.76) 2.52 (0.14)</td><td>31.32 (13.36) 32.95 (14.13) 19.06 (14.71)</td><td></td><td>23.15 (1.13) 28.40 (0.83) 42.01 (2.72)</td></tr><tr><td>0.3</td><td>ACC Purity Fscore</td><td>16.22 (1.04) 16.72 (1.09) 2.81 (0.32)</td><td>16.40 (0.65) 16.98 (0.59) 2.79 (0.30)</td><td>16.45 (0.59) 17.05 (0.66) 2.51 (0.30)</td><td>16.30 (0.63) 16.83 (0.76) 2.53 (0.29)</td><td>16.40 (0.82) 16.95 (0.92) 2.66 (0.38)</td><td>16.30 (0.42) 17.05 (0.43) 2.54 (0.20)</td><td>16.45 (0.95) 17.18 (1.11) 2.64 (0.55)</td><td>16.96 (0.09) 2.66 (0.06)</td><td>16.27 (0.10)</td><td>21.52 (0.79) 25.95 (0.70)</td></tr><tr><td>0.5</td><td>ACC Purity Fscore</td><td>15.85 (0.58) 16.32 (0.70) 2.98 (0.52)</td><td>15.98 (0.75) 16.40 (0.79) 3.09 (0.26)</td><td>16.50 (0.69) 17.10 (0.74) 2.71 (0.47)</td><td>16.32 (0.78) 16.92 (0.92) 2.64 (0.35)</td><td>15.98 (0.41) 16.55 (0.48) 2.55 (0.18)</td><td>16.23 (0.91) 17.03 (0.88) 2.55 (0.38)</td><td>16.53 (0.86) 17.10 (1.04) 2.68 (0.41)</td><td>16.43 (0.17) 17.04 (0.28) 2.84 (0.08)</td><td>35.13 (3.20) 21.85 (1.30) 25.65 (0.72)</td><td></td></tr><tr><td>proteinFold</td><td>ACC 0.1 Purity</td><td>11.79 (0.40) 15.94 (0.77)</td><td>11.69 (0.45) 16.04 (0.55) 4.02 (0.18)</td><td>11.95 (0.55) 16.17 (0.47)</td><td>11.70 (0.31) 16.02 (0.59)</td><td>11.73 (0.49) 15.73 (0.52)</td><td>11.97 (0.36) 16.05 (0.81)</td><td>12.00 (0.44) 16.40 (0.52)</td><td></td><td>12.88 (1.03) 17.26 (1.20)</td><td>31.84 (1.56) 15.98 (0.43) 20.85 (0.34)</td></tr><tr><td rowspan="5">Cora</td><td>0.3</td><td>Fscore ACC Purity Fscore</td><td>4.06 (0.23) 12.07 (0.47) 16.73 (0.79) 4.15 (0.22)</td><td>12.13 (0.64) 15.98 (0.88) 5.48 (0.34)</td><td>3.86 (0.14) 11.86 (0.32) 16.04 (0.31) 3.84 (0.14)</td><td>3.99 (0.14) 11.77 (0.38) 16.07 (0.67) 3.93 (0.18)</td><td>3.96 (0.18) 11.67 (0.28) 15.91 (0.61) 3.94 (0.16)</td><td>3.96 (0.16) 12.07 (0.20) 16.04 (0.48) 3.95 (0.20)</td><td>3.98 (0.20) 11.59 (0.37) 15.79 (0.49) 3.73 (0.19)</td><td>4.33 (0.49) 11.90 (0.09) 16.06 (0.07) 4.36 (0.33)</td><td>42.65 (1.28) 15.71 (0.49) 20.26 (0.55)</td><td></td></tr><tr><td>0.5</td><td>ACC Purity Fscore</td><td>12.23 (0.56) 16.28 (0.75) 4.31 (0.29) 36.05 (1.16)</td><td>12.38 (0.46) 16.08 (0.74) 7.10 (1.36)</td><td>11.71 (0.38) 15.94 (0.60) 3.84 (0.14)</td><td>11.84 (0.35) 15.85 (0.45) 3.93 (0.16)</td><td>11.82 (0.51) 15.98 (0.57) 3.94 (0.21)</td><td>12.02 (0.38) 16.05 (0.49) 3.97 (0.15)</td><td>11.86 (0.34) 16.15 (0.54) 3.81(0.15)</td><td>11.81 (0.05) 15.90 (0.09) 5.55 (0.30)</td><td></td><td>36.78 (1.30) 15.40 (0.48) 19.38 (0.40) 29.42 (1.33)</td></tr><tr><td>0.1</td><td>ACC Purity Fscore ACC</td><td>45.54 (1.20) 27.09 (1.29) 31.66 (1.90)</td><td>36.34 (1.37) 45.89 (1.16) 27.59 (1.41) 29.67 (2.43)</td><td>48.58 (0.47) 54.30(0.78) 32.16(0.53) 43.14 (0.88)</td><td>36.35 (0.79) 45.72 (0.91) 27.72 (0.46) 32.69 (1.55)</td><td>36.66 (0.77) 46.11 (0.95) 28.04 (0.49) 33.21 (1.46)</td><td>36.55 (0.70) 45.93 (0.92) 27.82 (0.43) 32.61 (1.54)</td><td>48.53 (0.40) 54.21 (0.67) 31.97 (0.43) 44.17 (1.17)</td><td>14.07 (3.66) 21.35 (7.74) 14.58 (3.71) 8.53 (5.55)</td><td></td><td>55.27 (1.12) 58.08 (0.96) 35.01 (0.51)</td></tr><tr><td>0.3 0.5</td><td>Purity Fscore ACC Purity</td><td>40.16 (2.22) 23.22 (1.68) 27.53 (2.98) 35.97 (2.23)</td><td>39.00 (2.55) 22.73 (1.80) 27.96 (2.32)</td><td>50.16 (1.08) 29.14 (0.67) 39.79 (1.37)</td><td>41.81 (1.78) 25.28 (0.93) 32.01 (1.70)</td><td>42.49 (1.74) 26.59 (0.86) 32.47 (1.49)</td><td>41.75 (1.82) 25.21 (0.99) 32.28 (1.45)</td><td>51.29 (1.12) 28.84 (0.62) 40.29 (1.69)</td><td>5.96 (5.28)</td><td>14.40 (9.58) 8.25 (5.65)</td><td>50.87 (1.47) 53.74 (1.23) 31.28 (1.01) 46.18 (1.44)</td></tr><tr><td>0.1</td><td>Fscore ACC</td><td>21.59 (0.94) 13.12 (0.24) 20.00 (0.02)</td><td>36.77 (2.33) 22.27 (1.54) 13.11 (0.22)</td><td>48.38 (1.26) 27.19 (0.79) 13.13 (0.26)</td><td>40.14 (1.29) 23.15 (0.70) 13.10 (0.24)</td><td>41.35 (1.45) 26.44 (0.65) 13.10 (0.24)</td><td>40.48 (0.97) 23.21 (0.64) 13.16 (0.35)</td><td>46.85 (1.71) 25.87 (0.84) 13.08 (0.26)</td><td>9.03 (8.24) 5.79 (5.44) 9.91 (5.94)</td><td></td><td>49.78 (1.20) 27.78 (1.02)</td></tr><tr><td rowspan="4">UCIDigits</td><td></td><td>Purity Fscore ACC</td><td>10.26 (0.11) 12.89 (0.39)</td><td>20.00 (0.00) 10.23 (0.11) 12.83 (0.29)</td><td>20.01 (0.03) 10.28 (0.10) 12.93 (0.39)</td><td>20.00 (0.00) 10.27 (0.10)</td><td>20.01 (0.03) 10.22 (0.09) 12.84 (0.28)</td><td>20.00 (0.00) 10.20 (0.11) 12.92 (0.38)</td><td>20.00 (0.00) 10.21 (0.10)</td><td></td><td>10.23 (6.16) 7.66 (4.37)</td><td>13.66 (0.49) 20.38 (0.22) 20.58 (3.37) 13.42 (0.30)</td></tr><tr><td>0.3</td><td>Purity Fscore</td><td>20.00 (0.02) 10.33 (0.15) 13.11 (0.39)</td><td>20.01 (0.03) 10.47 (0.12)</td><td>20.02 (0.03) 10.31 (0.13)</td><td>12.91 (0.40) 20.00 (0.02) 10.33 (0.15)</td><td>20.01 (0.03) 10.38 (0.12)</td><td>20.00 (0.00) 10.31 (0.16)</td><td>12.86 (0.39) 20.00 (0.02) 10.28 (0.16)</td><td>10.68 (6.59) 11.34 (7.00) 8.93 (5.22)</td><td></td><td>20.27 (0.16) 14.58 (4.25)</td></tr><tr><td>0.5</td><td>ACC Purity Fscore ACC</td><td>20.06 (0.10) 11.86 (0.14) 17.90 (0.70)</td><td>13.09 (0.29) 20.06 (0.09) 12.87 (0.24) 17.94 (0.69)</td><td>13.15 (0.39) 20.05 (0.10) 11.81 (0.14) 17.02 (0.33)</td><td>13.07 (0.39) 20.06 (0.10) 11.84 (0.14) 17.59 (0.73)</td><td>13.13 (0.30) 20.04 (0.09) 12.49 (0.24) 17.69 (0.70)</td><td>13.13 (0.38) 20.06 (0.10) 11.13 (0.65) 17.59 (0.69)</td><td>13.16 (0.37) 20.06 (0.10) 11.77 (0.13)</td><td>11.26 (9.44) 12.15 (10.52) 9.48 (8.03)</td><td></td><td>13.47 (0.30) 20.34 (0.22) 16.28 (2.20)</td></tr><tr><td>Caltech101-7</td><td>0.1 Fscore ACC 0.3 Purity Purity</td><td>54.14 (0.00) 14.48 (0.12) 17.54 (1.06) 54.14 (0.00) 14.44 (0.17)</td><td>54.14 (0.00) 14.48 (0.13) 17.95 (0.77) 54.14 (0.00)</td><td>54.14 (0.00) 14.32 (0.07) 17.33 (0.49) 54.14 (0.00)</td><td>54.14 (0.00) 14.42 (0.12) 17.28 (0.40) 54.14 (0.00)</td><td>54.14 (0.00) 14.43 (0.13) 17.63 (0.58) 54.14 (0.00)</td><td>54.14 (0.00) 14.44 (0.11) 17.24 (0.49) 54.14 (0.00)</td><td>17.37 (0.37) 54.14 (0.00) 14.36 (0.06) 17.16 (0.46)</td><td>54.14 (0.00)</td><td>23.74 (3.73) 58.39 (3.82) 18.61 (2.86) 19.70 (0.17) 54.14 (0.00)</td><td>53.11 (0.41) 54.97 (0.08) 89.43 (0.74) 50.61 (0.87) 54.72 (0.10)</td></tr><tr><td>Caltech101-20</td><td>0.5 0.1</td><td>Fscore ACC Purity Fscore ACC</td><td>18.01 (0.93) 54.14 (0.00) 14.58 (0.27) 8.39 (0.31)</td><td>14.51 (0.14) 18.22 (1.12) 54.14 (0.00) 14.71 (0.29) 8.39 (0.35)</td><td>14.38 (0.09) 18.32 (0.91) 54.14 (0.00) 14.66 (0.31) 8.54 (0.20)</td><td>14.40 (0.10) 17.39 (0.49) 54.14 (0.00) 14.44 (0.13) 8.43 (0.39)</td><td>14.47 (0.12) 17.86 (1.49) 54.14 (0.00) 14.61 (0.33) 8.37 (0.42)</td><td>14.37 (0.08) 17.43 (0.40) 54.14 (0.00) 14.44 (0.12) 8.29 (0.31)</td><td>14.34 (0.08) 18.15 (1.19) 54.14 (0.00) 14.64 (0.29) 8.53 (0.42)</td><td>15.61 (0.09) 21.62 (1.79) 56.87 (2.44) 16.59 (0.79) 9.20 (4.65)</td><td></td><td>73.08 (1.11) 48.13 (0.82) 54.69 (0.15) 61.22 (0.65)</td></tr><tr><td rowspan="5"></td><td>0.3</td><td>Purity Fscore ACC Purity</td><td>33.46 (0.03) 5.05 (0.05) 8.52 (0.32) 33.48 (0.08)</td><td>33.47 (0.04) 5.05 (0.05) 8.54 (0.28) 33.47 (0.08)</td><td>33.47 (0.08) 5.12 (0.04) 8.65 (0.34)</td><td>33.45 (0.01) 5.04 (0.06) 8.56 (0.29)</td><td>33.45 (0.00) 5.04 (0.06) 8.67 (0.34)</td><td>33.46 (0.04) 5.04 (0.04) 8.67(0.30)</td><td>33.45 (0.00) 5.11 (0.07) 8.82 (0.30)</td><td></td><td>23.88 (10.16) 4.67 (2.32) 9.07 (4.85)</td><td>33.95 (0.25) 36.83 (0.26) 88.29 (0.85) 32.16 (0.58) 36.48 (0.31)</td></tr><tr><td></td><td>Fscore ACC</td><td>5.15 (0.06) 8.53 (0.21) 33.45 (0.00)</td><td>5.12 (0.08) 8.66 (0.14)</td><td>33.50 (0.11) 5.16 (0.08) 8.42 (0.16)</td><td>33.49 (0.09) 5.15 (0.09) 8.76 (0.41)</td><td>33.45 (0.00) 5.16 (0.08) 8.81(0.33) 33.46 (0.05)</td><td>33.47 (0.06) 5.19 (0.10) 8.76 (0.27) 33.47 (0.05)</td><td>33.52 (0.12) 5.18 (0.06) 8.40 (0.19)</td><td>28.96 (8.74) 4.93 (2.14) 6.86 (1.19)</td><td>70.41 (1.29) 30.42 (0.31)</td><td></td></tr><tr><td>0.5 0.1</td><td>Purity Fscore ACC Purity</td><td>5.14 (0.05) 4.04 (0.04) 10.71 (0.16)</td><td>33.45 (0.00) 5.13 (0.05) 4.02 (0.08) 10.73 (0.12)</td><td>33.45 (0.03) 5.09 (0.04) 4.01 (0.08) 10.70 (0.08)</td><td>33.45 (0.00) 5.23 (0.11) 4.06 (0.07) 10.66 (0.14)</td><td>5.23 (0.11) 4.04 (0.06) 10.64 (0.13)</td><td>5.22 (0.08) 4.07 (0.04) 10.66 (0.10)</td><td>33.45 (0.03)</td><td>5.09 (0.04) 4.09 (0.08) 10.74 (0.10)</td><td>26.45 (4.63) 4.08 (0.71) 4.69 (3.26) 11.04 (7.53)</td><td>36.00 (0.24) 58.37 (0.81) 9.36 (0.46) 11.34 (0.10)</td></tr><tr><td>0.3</td><td>Fscore ACC Purity</td><td>1.09 (0.02) 4.06 (0.09) 10.77 (0.16)</td><td>1.09 (0.01) 4.10 (0.09) 10.72 (0.18)</td><td>1.06 (0.02) 4.01 (0.07) 10.69 (0.17)</td><td>1.09 (0.02) 4.07 (0.09) 10.71 (0.10)</td><td>1.08 (0.02) 4.09 (0.09) 10.71 (0.11)</td><td>1.07 (0.01) 4.09 (0.08) 10.75 (0.14)</td><td>1.09 (0.01) 4.04 (0.06) 10.63 (0.17)</td><td></td><td>1.77 (1.33) 3.93 (2.60)</td><td>62.83 (16.40) 10.02 (0.12)</td></tr><tr><td>Caltech101-all</td><td>Fscore ACC 0.5 Purity</td><td>1.08 (0.02) 4.10 (0.11) 10.65 (0.05)</td><td>1.09 (0.02) 4.32 (0.05) 10.60 (0.10)</td><td>1.06 (0.02) 4.01 (0.05) 10.71 (0.12)</td><td>1.11 (0.02) 4.08 (0.07) 10.63 (0.13)</td><td>1.12 (0.01) 4.23 (0.12) 10.65 (0.14)</td><td>1.11 (0.03) 4.07 (0.10) 10.66 (0.10)</td><td></td><td>1.06 (0.01) 4.02 (0.05) 10.70 (0.09)</td><td>9.72 (6.03) 1.41 (1.07) 3.97 (2.60) 9.79 (6.23)</td><td>11.14 (0.09) 65.66 (1.46) 10.03 (0.20) 11.08 (0.12)</td></tr></table></body></html>

1) Our proposed method achieves superior clustering results on the widely used datasets compared to eight other approaches, validating its ability to cluster uncoupled incomplete multi-view data. Specifically, the improvements of our algorithm over the second best method in terms of ACC are $4 . { \bar { 0 } } 7 \%$ , $3 . 6 4 \%$ , $6 . 7 0 \%$ , $0 . 4 9 \%$ , $2 3 . 0 9 \%$ , $5 . 9 2 \%$ , $5 . 9 2 \%$ for the missing ratio of 0.3 on all benchmarks, demonstrating the superior performance of KOALA.

2) Our approach outperforms existing kernel-based IMVC methods. For instance, KOALA outperforms the best alternatives by $0 . 3 7 \%$ , $0 . 2 5 \%$ and $0 . 2 8 \%$ in terms of purity, despite different missing ratios on UCI Digits. The gaps for the other datasets have similar properties. The success of the proposed approach is attributed to the utilization of crosskernel coupling learning, which exploits existing consensus and complementary information to recouple kernel elements and improve clustering results.

3) UIMC also serves as an IMVC approach tailored for uncoupled incomplete multi-view data, which outperforms most IMVC methods in various circumstances due to its coupling learning. The comparison between our proposed UC-IMVC and UIMC is presented in Table 2. As observed from the table, our method shows better clustering performance than UIMC with different missing ratio in most cases. Specifically, in terms of ACC for the missing ratio of 0.5, our approach achieves better performance than UIMC on

ORL Caltech101-7 0.28 Unfill 0.55 Unfill Fill Fill   
0.27 0.548   
= 山   
P0.26 P0.546 0.25 0.544 0.1 0.3 0.5 0.1 0.3 0.5 missingratio missing ratio (a) ORL (b) Caltech101-7

proteinFold Caltech101-7 0.2 市 0.55 1.i 0.18 0.545 P 0.16 0.54 0.1 0.3 0.5 missing ratio missing ratio (a) proteinFold (b) Caltech101-7

all datasets, i.e., $5 . 4 2 \%$ , $3 . 5 9 \%$ , $4 0 . 2 2 \%$ , $2 . 2 1 \%$ , $2 6 . 5 1 \%$ , $2 3 . 5 6 \%$ and $6 . 0 6 \%$ respectively, which demonstrates the effectiveness of KOALA in addressing the uncoupled incomplete multi-view data.

Ablation Study. In order to analyze two primary contributions in our algorithm, we conduct two ablation experiments to illustrate the validity of our strategies separately.

Missing kernel elements recovery strategy. The purpose of this ablation experiment is to verify the effectiveness of the imputation of uncoupled incomplete multi-view data. We present the experimental results of the ablation study in Fig. 2, where ”Unfill” indicates not using our incomplete multi-view imputation strategy. As shown in Fig. 2, our method has significant advantages in clustering performance, which fully proves the validity of the strategy of kernel completion.

Cross-kernel coupling learning Strategy. To further verify the superiority of the cross-kernel mapping strategy, we recommend conducting the ablation study experiment presented in Fig. 3, where ”Uncoupled” manifests the correlation relationship among different kernels is unknown. As we can see from Fig. 3, our proposed cross-kernel coupling learning strategy enhances the performance under various missing ratios on proteinFold and Caltech101-7 datasets.

Convergence. Convergence to a local optimum of our proposed method is guaranteed. The objective value in Eq. (6) consistently decreases in each iteration when optimizing one variable while keeping the others fixed. This provides proof that the entire optimization algorithm converges to a local optimum. To demonstrate this point in practice, we generate plots of the objective value curves of our approach, specifi

Caltech101-20 Cora E F 373.5 1252 0 10 20 30 0 10 20 30 Number of iterations Number of iterations (a) Caltech101-20 (b) Cora

![](images/6d57e85943e23cb8b667bdd74e55e2c280a257f5961b1fabae6a3f919683a9d6.jpg)  
Figure 2: The ablation study of our missing kernel elements recovery strategy.   
Figure 3: The ablation study of our cross-kernel coupling learning.   
Figure 4: Caltech101-20 and Cora objective values from KOALA with missing ratio 0.1.   
Figure 5: Sensitivity analysis of $\lambda$ and $\theta$ of out method on two benchmark datasets.

cally showing the number of iterations with varying missing ratios on partial datasets. As observed in Fig. 4, the objective value of KOALA decreases monotonically with each iteration and typically converges quickly.

Parameter Sensitivity. KOALA introduces two parameters $\lambda$ and $\theta$ to trade off better clustering performance. $\lambda$ denotes the low-rank tensor constraint balance parameter and $\theta$ is the kernel alignment balance parameter. We perform experiments to investigate the impact and sensitivity of these parameters on clustering performance across all datasets. We set $\lambda$ in a range of $[ 1 0 ^ { = 5 } , 1 0 ^ { - 1 } ]$ and $\theta$ in $[ 1 0 ^ { 4 } , 1 0 ^ { 6 } ]$ . Fig. 5 shows the ACC of KOALA on two datasets with different missing ratios. From the observation, our approach achieves stable clustering performance over a wide range of $\lambda$ and $\theta$ .

# Conclusion

In this paper, we present a novel coupling-imputationclustering framework termed Kernel Coupling and Element Imputation Induced Multi-View Clustering (KOALA). The proposed method constructs a kernel for each uncoupled incomplete multiple view and uses cross-kernel coupling learning to refine the correspondence among multiple kernels. Meanwhile, KOALA explores high-order correlations to guide the imputation of absent kernel elements. We jointly optimize the reconstruction of incomplete kernels, crosskernel alignment, and clustering within an integrated framework to achieve superior clustering performance. Finally, a six-step alternative algorithm is designed to ensure KOALA convergence. The effectiveness of the model is demonstrated through extensive experiments on benchmark datasets.

# Acknowledgments

This work was supported by the Fundamental Research Funds for the Central Universities (No. 2022JBZY019) and the Beijing Natural Science Foundation (No. 4242046). Jiazheng Yuan is the co-corresponding author.