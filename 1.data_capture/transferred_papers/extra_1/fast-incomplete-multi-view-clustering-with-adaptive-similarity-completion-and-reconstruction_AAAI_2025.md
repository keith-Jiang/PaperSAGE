# Fast Incomplete Multi-view Clustering with Adaptive Similarity Completion and Reconstruction

Deng Xu, Chao Zhang\*, Cong Guo, Chunlin Chen, Huaxiong Li\*

Department of Control Science and Intelligence Engineering, Nanjing University dengxu, chzhang, congguo @smail.nju.edu.cn, clchen, huaxiongli @nju.edu.cn

# Abstract

Recently, anchor-based incomplete multi-view clustering (IMVC) has been widely adopted for fast clustering, but most existing approaches still encounter some issues: (1) They generally rely on the observed samples to construct anchor graphs, ignoring the potentially useful information of missing instances. (2) Most methods attempt to learn a consensus anchor graph, failing to fully excavate the complementary information and high-order correlations across views. (3) They generally apply post-processing on learned anchor graph to seek latent embeddings, making them not globallyoptimal. To address these issues, this paper proposes a novel fast IMVC approach with Adaptive Similarity Completion and Reconstruction (ASCR), which unifies anchor learning, anchor-sample similarity construction and completion, and latent multi-view embedding learning in a joint framework. Specifically, ASCR learns an anchor-sample similarity graph for each view, and the missing values are fulfilled to mitigate the adverse effects. To explore the consistent and complementary information across views, ASCR simultaneously seeks the view-specific anchor embeddings and sample embeddings in a latent subspace by similarity reconstruction, which not only preserves the semantic information into latent embeddings but also enhances the low-rank property of similarity graphs, achieving a reliable graph completion process. Furthermore, the high-order cross-view correlations are explored with tensor-based regularization. Extensive experimental results demonstrate the superiority and efficiency of ASCR compared with SOTA approaches.

Code — https://github.com/dengxu-nju/ASCR

# Introduction

In real-world scenarios, diverse sources and feature collectors generate multi-view data, whose exponential growth over the past decade has driven significant research in multiview learning, especially in multimedia and machine learning (Lu et al. 2019; Cai et al. 2024; Han et al. 2022; Fang et al. 2023). Among these tasks, multi-view clustering (MVC) (Sun et al. 2024; Zhang et al. 2024; Wang et al. 2021; Xu et al. 2024; Chen et al. 2022; Li et al. 2024; Gu, Li, and

Feng 2024) stands out for its good ability to leverage complementary information across multiple views for clustering. However, in practical scenarios, some samples may be only partially available due to detector failure or data corruption.

To effectively group these incomplete data, there has been considerable attention towards incomplete multi-view clustering (IMVC). Numerious existing IMVC approaches have proven effective, such as those employing matrix factorization (Li, Jiang, and Zhou 2014), graph construction (Zhang et al. 2023a; Wen, Xu, and Liu 2020), subspace learning (Hu and Chen 2018; Liu et al. 2021), and deep learning (Lin et al. 2021; Xu et al. 2023; Chao, Jiang, and Chu 2024). For instance, (Liu et al. 2021) proposed jointly performing data imputation and self-representation learning for better IMVC results. (Wen, Xu, and Liu 2020) exploits the graph learning and spectral clustering techniques to learn the common representation for IMVC. Furthermore, the deep IMVC methods usually try to infer missing data and extract representation with a deep neural network (Wen et al. 2020b). Although remarkable success achieved by these IMVC approaches, their high storage and computational complexity pose challenges to their application on large-scale datasets.

Recently, some anchor-based IMVC methods have been proposed for fast clustering, which are widely applied in large-scale IMVC tasks (Liu et al. 2022; Wang et al. 2022; Yu et al. 2022; Wen et al. 2023; Yu et al. 2024). Anchorbased IMVC focuses on exploring the similarity relations between some anchors and samples to construct anchor graphs. As a representative, (Wang et al. 2022) proposed to learn a consensus anchor matrix and an anchor graph in a latent subspace. (Wen et al. 2023) learned view-specific anchors and anchor graphs, and aligned all graphs with a consensus one. In the latest work by (Yu et al. 2024), they utilized observed samples to generate view-shared anchors with multi-dimensions and multi-sizes for large-scale IMVC tasks. Although these anchor-based IMVC methods have achieved great success in large-scale IMVC applications, most of them still encounter some issues: (1) They typically construct anchor-sample similarity graphs directly from observed samples, neglecting the potentially valuable information of missing samples. (2) Most methods mainly focus on learning a consensus anchor graph, which might not fully exploit complementary information and high-order correlations across views. (3) They generally need post-processing such as singular value decomposition (SVD) (Wen et al. 2023; Chen et al. 2023; Yu et al. 2024) for the learned anchor graph to seek latent embeddings for final $k$ -means, which separates the graph learning and embedding learning into two steps and may lead to sub-optimal results.

![](images/6d09c67b3ea7116ebba80692e618dfa408508465af1a060b638b69d6a2302914.jpg)  
Figure 1: The overall framework of ASCR.

To overcome these limitations, in this paper, we proposed a novel fast IMVC approach with Adaptive Similarity Completion and Reconstruction (ASCR), whose framework is shown in Figure 1. Specifically, ASCR learns an anchorsample similarity graph for each view, and the missing values are effectively fulfilled to alleviate the negative impact of incomplete data. To explore the latent consistent and complementary information across views, ASCR seeks the latent view-specific anchor embedding and sample embedding by reconstructing the similarity graphs, which not only preserves the similarities into latent embeddings but also enhances the low-rank property of similarity graphs, ensuring a robust completion process and mutual reinforcement. Furthermore, ASCR incorporates tensorized regularization on sample embeddings to explore high-order cross-view correlations, enabling a deeper understanding of the underlying data structure. Finally, the multi-view sample embeddings are combined for clustering, leveraging the comprehensive information captured across views. The main contributions of this work are summarized as follows:

• We propose a novel IMVC method termed ASCR, which integrates anchor leaning, similarity graph construction and completion, and latent multi-view embedding learning into a unified framework. • ASCR adaptively learns and completes anchor-sample similarity graphs, and simultaneously reconstructs the similarities in a latent subspace for discriminative embeddings learning. The high-order cross-view correlations are also explored with tensor-based regularization. • Extensive experiments on several popular datasets demonstrate the effectiveness and efficiency of ASCR compared to various state-of-the-art IMVC methods.

# Related Work

# Notations and Preliminaries

We adopt the following notation conventions this paper: bold lowercase letters (e.g., a) represent vectors, uppercase letters (e.g., A) denote matrices, and calligraphic letters (e.g., $\mathcal { A }$ ) signify tensors. For a matrix $\mathbf { A } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } }$ , its Frobenius norm and nuclear norm are defined as $\| \mathbf { A } \| _ { F } = \sqrt { \sum _ { i j } a _ { i j } ^ { 2 } }$ and $\begin{array} { r } { \| \mathbf { A } \| _ { * } = \sum _ { i } \delta _ { i } ( \mathbf { A } ) } \end{array}$ , respectively, where $a _ { i j }$ is the element of $\mathbf { A }$ a position $( i , j )$ , and $\delta _ { i } ( \mathbf { A } )$ is the $i$ -th singular value of A. $\mathbf { I } _ { k }$ is a $k$ -dimensional identity matrix. For a tensor $\mathcal { A } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ , we denote the $i$ -th frontal, lateral, and horizontal slice as $\hat { \mathcal { A } } ^ { ( : , : , i ) } , \mathcal { A } ^ { ( : , i , : ) }$ , and $\mathcal { A } ^ { ( i , : , : ) }$ , respectively. Additionally, for convenience, we use $\mathbf { \mathcal { A } } ^ { ( i ) }$ to represent $\mathcal { A } ^ { ( : , : , i ) }$ . $\ b { \mathcal { A } } _ { f }$ denotes the fast Fourier transformation (FFT) of $\mathcal { A }$ along the third dimension, i.e., $\mathcal { A } _ { f } = \mathrm { f f t } ( \mathcal { A } , [ ] , 3 )$ , and $\mathcal { A }$ can be recovered from $\ b { \mathcal { A } } _ { f }$ by the inverse FFT operation, i.e., $\mathcal { A } = \mathrm { i f f t } ( \mathcal { A } _ { f } , [ ] , 3 )$ (Lu et al. 2020).

Definition 1 (t-SVD (Kilmer et al. 2013)). For a tensor $\mathcal { A } \in$ $\mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ , its t-SVD is defined as

$$
\mathcal { A } = \mathcal { U } \ast \mathcal { S } \ast \mathcal { V } ^ { T } ,
$$

where $\mathcal { U } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 1 } \times n _ { 3 } }$ and $\boldsymbol { \mathcal { V } } \in \mathbb { R } ^ { n _ { 2 } \times n _ { 2 } \times n _ { 3 } }$ are orthogonal tensors, $S \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ is an f-diagonal tensor, and ” $^ { , } { * }$ ” denotes the t-product.

Definition 2 (t-SVD based tensor nuclear norm (Semerci et al. 2014)). Given a tensor $\mathcal { A } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ , its t-SVD based tensor nuclear norm is defined as

$$
\| \boldsymbol { \mathcal { A } } \| _ { \oplus } = \sum _ { k = 1 } ^ { n _ { 3 } } \| \mathbf { A } _ { f } ^ { ( k ) } \| _ { * } = \sum _ { i = 1 } ^ { \operatorname* { m i n } ( n _ { 1 } , n _ { 2 } ) } \sum _ { k = 1 } ^ { n _ { 3 } } \delta _ { i } ( \mathbf { A } _ { f } ^ { ( k ) } ) ,
$$

# Anchor-Based IMVC

Anchor graph has emerged as a powerful tool in exploring the data structure of large-scale data. Most anchor-based

IMVC methods focus on using some representative anchors as the basis to construct a consensus similarity matrix that captures the relationships between the anchors and samples. Let $\mathbf { X } = \{ \mathbf { X } ^ { v } \} _ { v = 1 } ^ { m }$ with $\mathbf { X } ^ { v } \in \mathbb { R } ^ { d _ { v } \times n _ { v } }$ denote an incomplete multi-view dataset, where $m$ is the view number, $d _ { v }$ denotes the feature dimension and $n _ { v }$ signifies the number of observed samples in view $v$ . The general framework of anchor-based IMVC can be described as

$$
\underset { \mathbf { A } ^ { v } , \mathbf { Z } } { \operatorname* { m i n } } \sum _ { v = 1 } ^ { m } \| \mathbf { X } ^ { v } - \mathbf { A } ^ { v } \mathbf { Z Q } ^ { v ^ { T } } \| _ { F } ^ { 2 } + \alpha \mathcal { R } ( \mathbf { Z } )
$$

where $\mathbf { A } ^ { v } \in \mathbb { R } ^ { d _ { v } \times k }$ is the pre-defined or learnable anchor matrix of the $\boldsymbol { v }$ -th view, and $k$ is the number of anchors (usually $k \ll n _ { v } ,$ . $\mathbf { Z } \in \mathbb { R } ^ { k \times n }$ is the consensus anchor graph. $\mathbf { Q } ^ { v } \in \mathbb { R } ^ { n _ { v } \times n }$ functions as an index matrix to indicate the positions of existing samples, and it is constructed by deleting those rows of an $n \times n$ identity matrix that correspond to the missing samples. By this way, $\mathbf { Z } \mathbf { Q } ^ { v ^ { T } } \in \mathbb { R } ^ { k \times n _ { v } }$ denotes the relationship between anchors and existing samples of the $v$ -th view. $\mathcal { R } ( \bar { \mathbf Z } )$ is a regularization term for some properties like sparse or low-rank. When the optimal anchor graph $\mathbf { Z }$ is obtained, SVD is then applied to seek the latent embedding for $k$ -means clustering (Wang et al. 2022; Liu et al. 2022; Chen et al. 2023).

Based on Eq. (1), (Wang et al. 2022) jointly learned a consensus anchor matrix and anchor graph in a latent subspace. Following it, (Liu et al. 2022) considered view-specific anchor matrices $\mathbf { A } ^ { v }$ and a shared anchor graph Z. They both use Frobenius norm regularization for a smooth graph. (Wen et al. 2023) learned the view-specific anchor graphs and fused them to a consensus one. However, these methods neglect the potential information of missing samples and also fail to explore the high-order cross-view correlations. Besides, all of them separate the graph learning and embedding learning into two individual steps, may making the embedding not globally-optimal. Although some methods like (Chen et al. 2023) employed the tensor nuclear norm for high-order correlations exploration, the rest two issues are still not addressed. In the upcoming section, we will propose a novel approach named ASCR, which is carefully designed to address these limitations effectively.

# The Proposed Method

# Model Formulation

From Eq. (1), we can observe that most anchor-based IMVC methods aim to develop a consensus anchor-sample similarity graph $\mathbf { Z }$ . However, this approach often neglects the discrepancies between different views. To address this issue, we propose learning a distinct anchor-sample similarity graph $\mathbf { Z } ^ { v }$ for each view. This approach enables a more thorough exploration of the anchor-sample similarity structure within each specific view, thereby enhancing the mining of cross-view complementary information.

It should be noted that $\mathbf { Z } ^ { v }$ only reveals the similarities between anchors and existing samples, and the similarities w.r.t. missing samples are unavailable. Previous methods directly discard the information. However, as the missing rate increases, the loss of such valuable information becomes increasingly significant. To tackle this issue, we aim to learn a complete anchor-sample similarity graph for each view by adaptively fulfilling the missing values, which can be formulated as

$$
\begin{array} { c } { \displaystyle \underset { \mathbf { A } ^ { v } , \mathbf { Z } ^ { v } } { \operatorname* { m i n } } \displaystyle \sum _ { v = 1 } ^ { m } \| \mathbf { X } ^ { v } - \mathbf { A } ^ { v } \mathbf { Z } ^ { v } \| _ { F } ^ { 2 } + \| \mathbf { Z } ^ { v } \mathbf { Q } ^ { v } + \mathbf { E } ^ { v } \mathbf { P } ^ { v } - \mathbf { S } ^ { v } \| _ { F } ^ { 2 } } \\ { s . t . \mathbf { A } ^ { v } { } ^ { T } \mathbf { A } ^ { v } = \mathbf { I } _ { k } , \ \mathbf { Z } ^ { v } \geq 0 , \ \mathbf { E } ^ { v } \geq 0 . } \end{array}
$$

Here, $\mathbf Z ^ { v } \in \mathbb R ^ { k \times n _ { v } }$ reflects the relationship between anchors and existing samples of the $v$ -th view. ${ \bf E } ^ { v } \in  { }$ Rk×(n−nv) is the adaptive anchor-sample similarity completion matrix which reflects the similarity between anchors and missing samples. The nonnegative constraint on $\mathbf { Z } ^ { v }$ and $\mathbf { E } ^ { v }$ makes the learned similarity matrix and completion matrix physically meaningful. $\mathbf { P } ^ { v } \in \mathbb { R } ^ { ( n - n _ { v } ) \times n }$ is also an index matrix to indicate the positions of existing samples, which is constructed by deleting those columns of an $n \times n$ identity matrix that corresponds to existing samples of the $v$ -th view. In this way, we can utilize $\mathbf { E } ^ { v }$ to fill the missing values and obtain a complete anchor-sample similarity matrix $\mathbf { S } ^ { v } \in \mathbb { R } ^ { k \times n }$ for each view.

For a desirable completion, it is usually expected that the similarity matrix $\mathbf { S } ^ { v }$ can reveal the cluster structure of samples. Low-rank regularization is widely used in clustering models via nuclear norm for similarity matrices. Whereas, it usually needs post-processing for embedding learning, and also introduces additional terms and hyper-parameters. In our method, inspired by matrix factorization, we factorize the similarity matrix $\mathbf { S } ^ { v }$ into two latent factors, i.e., $\mathbf { S } ^ { v } = \mathbf { G } ^ { v } \mathbf { H } ^ { v }$ , where $\mathbf { G } ^ { v } \in \mathbb { R } ^ { k \times c }$ and $\mathbf { H } ^ { v } \in \mathbb { R } ^ { c \times n }$ denote the latent anchor embedding and sample embedding, respectively, and $c$ is the number of clusters $( c \leq k )$ . Then, Eq. (2) becomes

$$
\begin{array} { r } { \displaystyle \operatorname* { m i n } _ { \mathbf { H } ^ { v } , \mathbf { G } ^ { v } } \displaystyle \sum _ { v = 1 } ^ { m } \| \mathbf { X } ^ { v } - \mathbf { A } ^ { v } \mathbf { Z } ^ { v } \| _ { F } ^ { 2 } + \| \mathbf { Z } ^ { v } \mathbf { Q } ^ { v } + \mathbf { E } ^ { v } \mathbf { P } ^ { v } - \mathbf { G } ^ { v } \mathbf { H } ^ { v } \| _ { F } ^ { 2 } } \\ { s . t . \mathbf { A } ^ { v ^ { T } } \mathbf { A } ^ { v } = \mathbf { I } _ { k } , ~ \mathbf { Z } ^ { v } \geq 0 , ~ \mathbf { E } ^ { v } \geq 0 , ~ \mathbf { G } ^ { v ^ { T } } \mathbf { G } ^ { v } = \mathbf { I } _ { c } . . } \end{array}
$$

The orthogonal constraint on $\mathbf { G } ^ { v }$ is to avoid the arbitrary scale. The anchor-sample similarity matrix factorization not only encourages the low-rank property of $\mathbf { S } ^ { v }$ , but also incorporates the latent embedding learning into a unified model without introducing additional terms. And the similarity relations are reconstructed by anchor embedding and sample embedding in a latent subspace.

Eq. (3) learns a latent semantic sample embedding for each view individually, and the cross-view correlations are not fully captured. Instead of enforcing a shared representation for them, we adopt the tensor nuclear norm regularization to capture the cross-view consistency. The final objective function of our ASCR model is

$$
\begin{array} { r l } { \displaystyle \operatorname* { m i n } _ { \boldsymbol { v } = 1 } \displaystyle \sum _ { \boldsymbol { v } = 1 } ^ { m } \| \mathbf { X } ^ { \boldsymbol { v } } - \mathbf { A } ^ { \boldsymbol { v } } \mathbf { Z } ^ { \boldsymbol { v } } \| _ { F } ^ { 2 } } & { } \\ { + \| \mathbf { Z } ^ { \boldsymbol { v } } \mathbf { Q } ^ { \boldsymbol { v } } + \mathbf { E } ^ { \boldsymbol { v } } \mathbf { P } ^ { \boldsymbol { v } } - \mathbf { G } ^ { \boldsymbol { v } } \mathbf { H } ^ { \boldsymbol { v } } \| _ { F } ^ { 2 } + \alpha \| \mathcal { H } \| _ { \mathfrak { P } } } & { } \\ { s . t . \mathbf { A } ^ { \boldsymbol { v } ^ { T } } \mathbf { A } ^ { \boldsymbol { v } } = \mathbf { I } _ { k } , \mathbf { Z } ^ { \boldsymbol { v } } \geq 0 , \mathbf { E } ^ { \boldsymbol { v } } \geq 0 , } & { } \\ { \mathbf { G } ^ { \boldsymbol { v } ^ { T } } \mathbf { G } ^ { \boldsymbol { v } } = \mathbf { I } _ { c } , \mathcal { H } = \Phi ( \mathbf { H } ^ { 1 } , \mathbf { H } ^ { 2 } , . . . , \mathbf { H } ^ { m } ) . } & { } \end{array}
$$

where $\Omega = \{ \mathbf { A } ^ { v } , \mathbf { Z } ^ { v } , \mathbf { E } ^ { v } , \mathbf { G } ^ { v } , \mathcal { H } \}$ is the target variables set. $\Phi$ is to convert all sample embeddings $\mathbf { H } ^ { v }$ into a three-order tensor $\mathcal { H } \in \mathbb { R } ^ { c \times m \times n }$ . Our model integrates anchor learning, similarity matrix construction and completion, and latent embedding learning in a unified framework. The missing information is recovered and used to enrich the semantic representation. After solving the above model, we concatenate the multi-view latent embeddings by ${ \textbf { H } } =$ $\left[ \mathbf { H } ^ { 1 ^ { T } } , \mathbf { H } ^ { 2 ^ { T } } , . . . , \mathbf { H } ^ { m ^ { T } } \right] \in \mathbb { R } ^ { n \times m c }$ for $k$ -means clustering.

# Optimization

To optimize the objective function with multiple variables, we use an alternate optimization strategy, updating one variable at a time while keeping the others fixed.

Updating $\mathbf { E } ^ { v }$ : When other variables are fixed, the optimization for similarity completion matrix $\mathbf { E } ^ { v }$ is

$$
\mathbf { E } _ { t + 1 } ^ { v } = \underset { \mathbf { E } ^ { v } \geq 0 } { \arg \operatorname* { m i n } } \| \mathbf { E } ^ { v } \mathbf { P } ^ { v } - \mathbf { L } _ { t } ^ { v } \| _ { F } ^ { 2 } ,
$$

where $\mathbf { L } _ { t } ^ { v } = \mathbf { G } _ { t } ^ { v } \mathbf { H } _ { t } ^ { v } - \mathbf { Z } _ { t } ^ { v } \mathbf { Q } ^ { v }$ . Since $\mathbf { P } ^ { v }$ is an index matrix, Eq. (5) can be solved by the following rule:

$$
\mathbf { E } _ { t + 1 } ^ { v } = \operatorname* { m a x } ( \hat { \mathbf { L } } _ { t } ^ { v } , 0 ) ,
$$

where $\hat { \mathbf { L } } _ { t } ^ { v } \in \mathbb { R } ^ { k \times ( n - n _ { v } ) }$ is a sub-matrix of $\mathbf { L } _ { t } ^ { v }$ that is composed of the rows that correspond to missing samples.

Updating $\mathbf { A } ^ { v }$ : When other variables are fixed, the optimization for view-specific anchor matrix $\mathbf { A } ^ { v }$ is

$$
\mathbf { A } _ { t + 1 } ^ { v } = \underset { \mathbf { A } ^ { v } } { \operatorname { a r g m i n } } \| \mathbf { X } ^ { v } - \mathbf { A } ^ { v } \mathbf { Z } ^ { v } \| _ { F } ^ { 2 } \quad s . t . \mathbf { A } ^ { v ^ { T } } \mathbf { A } ^ { v } = \mathbf { I } _ { k } .
$$

By transforming the Frobenius norm to the trace and eliminating terms irrelevant to $\mathbf { A } ^ { v }$ , the above formula can be equivalently reformulated as:

$$
\operatorname* { m a x } _ { \mathbf { A } ^ { v } } \mathrm { T r } ( \mathbf { A } ^ { v ^ { T } } \mathbf { M } _ { t } ^ { v } ) \quad s . t . \mathbf { A } ^ { v ^ { T } } \mathbf { A } ^ { v } = \mathbf { I } _ { k } ,
$$

where $\mathbf { M } _ { t } ^ { v } = \mathbf { X } ^ { v } \mathbf { Z } _ { t } ^ { v ^ { T } }$ . This subproblem can be efficiently solved using SVD. The closed-form solution is given by $\mathbf { A } _ { t + 1 } ^ { v } = \mathbf { J } \bar { \mathbf { K } } ^ { T }$ , where $\mathbf { J }$ and $\mathbf { K }$ are the left and right singular matrices of $\mathbf { M } _ { t } ^ { v }$ .

Updating $\mathbf { Z } ^ { v }$ : When other variables are fixed, the optimization for $\mathbf { Z } ^ { v }$ is

$$
\begin{array} { r } { \mathbf { Z } _ { t + 1 } ^ { v } = \underset { \mathbf { Z } ^ { v } \geq 0 } { \operatorname { a r g m i n } } \| \mathbf { X } ^ { v } - \mathbf { A } ^ { v } \mathbf { Z } ^ { v } \| _ { F } ^ { 2 } + \| \mathbf { Z } ^ { v } \mathbf { Q } ^ { v } + \mathbf { E } ^ { v } \mathbf { P } ^ { v } - \mathbf { G } ^ { v } \mathbf { H } ^ { v } \| _ { F } ^ { 2 } . } \end{array}
$$

It can be further simplified as

$$
\mathbf { Z } _ { t + 1 } ^ { v } = \underset { \mathbf { Z } ^ { v } \geq 0 } { \arg \operatorname* { m i n } } \| \mathbf { O } _ { t } ^ { v } - \mathbf { Z } ^ { v } \| _ { F } ^ { 2 } .
$$

where $\mathbf { O } _ { t } ^ { v } = ( ( \mathbf { G } _ { t } ^ { v } \mathbf { H } _ { t } ^ { v } - \mathbf { E } _ { t + 1 } ^ { v } \mathbf { P } ^ { v } ) \mathbf { Q } ^ { v ^ { T } } + \mathbf { A } _ { t + 1 } ^ { v ^ { T } } \mathbf { X } ^ { v } ) / 2 .$ . The optimal solution of $\mathbf { Z } _ { t } ^ { v }$ is given by

$$
\mathbf { Z } _ { t + 1 } ^ { v } = \operatorname* { m a x } ( \mathbf { O } _ { t } ^ { v } , 0 ) .
$$

Updating $\mathbf { G } ^ { v }$ : When other variables are fixed, the optimization for anchor embedding $\mathbf { G } ^ { v }$ is

$$
\begin{array} { r } { \mathbf { G } _ { t + 1 } ^ { v } = \underset { \mathbf { G } ^ { v } } { \arg \operatorname* { m i n } } \| \mathbf { Z } _ { t + 1 } ^ { v } \mathbf { Q } ^ { v } + \mathbf { E } _ { t + 1 } ^ { v } \mathbf { P } ^ { v } - \mathbf { G } ^ { v } \mathbf { H } _ { t } ^ { v } \| _ { F } ^ { 2 } } \\ { s . t . \mathbf { G } ^ { v ^ { T } } \mathbf { G } ^ { v } = \mathbf { I } _ { c } . } \end{array}
$$

Similar to $\mathbf { A } ^ { v }$ , Eq. (12) is converted to

$$
\operatorname* { m a x } _ { \mathbf { G } ^ { v } } \mathrm { T r } ( \mathbf { G } ^ { v ^ { T } } \mathbf { N } _ { t } ^ { v } ) \quad s . t . \mathbf { G } ^ { v ^ { T } } \mathbf { G } ^ { v } = \mathbf { I } _ { c } ,
$$

where $\mathbf { N } _ { t } ^ { v } \ = \ ( \mathbf { Z } _ { t + 1 } ^ { v } \mathbf { Q } ^ { v } + \mathbf { E } _ { t + 1 } ^ { v } \mathbf { P } ^ { v } ) \mathbf { H } _ { t } ^ { v ^ { T } }$ . As in solving Eq. (8), the optimal solution for $\mathbf { G } ^ { v }$ is ${ \bf G } _ { t + 1 } ^ { v } = { \bf R } { \bf T } ^ { T }$ , where $\mathbf { R }$ and $\mathbf { T }$ are the left and right singular matrices of $\mathbf { N } _ { t } ^ { v }$ .

Updating $\mathcal { H }$ : When other variables are fixed, the optimization for sample embedding tensor $\mathcal { H }$ is

$$
\mathcal { H } _ { t + 1 } = \underset { \mathcal { H } } { \arg \operatorname* { m i n } } \alpha \| \mathcal { H } \| _ { \mathfrak { F } } + \| \mathcal { H } - \mathcal { F } _ { t + 1 } \| _ { F } ^ { 2 } ,
$$

where $\mathcal { F } _ { t + 1 } = \Phi ( \mathbf { F } _ { t + 1 } ^ { 1 } , . . . , \mathbf { F } _ { t + 1 } ^ { m } )$ is constructed by $\mathbf { F } _ { t } ^ { v } = $ $\mathbf { G } _ { t + 1 } ^ { v ^ { T } } ( \mathbf { Z } _ { t + 1 } ^ { v } \mathbf { Q } ^ { v } + \mathbf { E } _ { t + 1 } ^ { v } \mathbf { P } ^ { v } )$ . Eq. (14) is a typical low-rank tensor norm minimization problem that has a closed-form solution and can be solved by Theorem 1 (Xie et al. 2018).

Theorem 1. Given two tensor $B \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ and $\mathcal { D } \in$ Rn1×n2×n3 with a constant ρ, the globally optimal solution of the problem

$$
\operatorname* { m i n } _ { \boldsymbol { \mathcal { B } } } \rho \| \boldsymbol { \mathcal { B } } \| _ { \circledast } + \frac { 1 } { 2 } \| \boldsymbol { \mathcal { B } } - \mathcal { D } \| _ { F } ^ { 2 }
$$

can be obtained by the tensor tubal-shrinkage operator

$$
\mathcal { B } = \mathcal { C } _ { n _ { 3 } \rho } ( \mathcal { D } ) = \mathcal { U } * \mathcal { C } _ { n _ { 3 } \rho } ( S ) * \mathcal { V } ^ { T } ,
$$

where $\boldsymbol { \mathcal { D } } ~ = ~ \mathcal { U } * \mathcal { S } * \mathcal { V } ^ { T }$ and $\mathcal { C } _ { n _ { 3 } \rho } ( \mathcal { D } ) \ : = \ : \mathcal { S } \ast \mathcal { K } . \mathcal { K } \in$ $\mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ is a $f$ -diagonal tensor and its diagonal element in the Fourier domain is Kf (i, i, j) = (1 − (jn)3(ρi,i) )

Algorithm 1 summarizes the optimization process.

# Computational Complexity Analysis

Time Complexity When updating $\mathbf { E } ^ { v }$ , the primary time cost is from matrix multiplication, which takes $O ( c k n +$ $d _ { v } k n )$ . Updating $\mathbf { A } ^ { v }$ and $\mathbf { G } ^ { v }$ involves the SVD operation and matrix multiplication, requiring $O ( k ^ { 2 } d _ { v } + d _ { v } \dot { k n _ { v } } )$ and $O ( k c ^ { 2 } + c k n )$ , respectively. The $\mathbf { Z } ^ { v }$ step costs $O ( d _ { v } k n _ { v } +$ $c k n )$ for matrix multiplication. Updating $\mathcal { H }$ involves matrix multiplication, FFT, inverse FFT, and SVD operations, with matrix multiplication taking $\mathcal { O } ( k c n )$ . For a $c \times m \times n$ tensor, FFT and inverse FFT operations take $\mathcal { O } ( c m n \log ( n ) )$ , and the SVD operation needs $\mathcal { O } ( c n m ^ { 2 } )$ . Since $( k , c , m ) \ll$ $( n , d _ { v } )$ , and $n _ { v } < n$ , the overall time complexity of Algorithm 1 is $O ( \tau ( c m n \log ( n ) + k d n ) )$ , where $\begin{array} { r } { d = \mathbf { \bar { \sum } } _ { v = 1 } ^ { m } \mathbf { \bar { \boldsymbol { d } } } _ { v } } \end{array}$ , and $\tau$ is the number of iterations.

Space Complexity The major memory costs of our algorithm are for variables $\mathbf { A } ^ { v ^ { \bf } } \in \mathbb { R } ^ { d _ { v } \times k }$ , $\mathbf { Z } ^ { v } \mathbf { Q } ^ { v } \in \mathbb { R } ^ { k \times n }$ , $\mathbf { \bar { P } } ^ { v } \mathbf { E } ^ { v } \in \mathbb { R } ^ { k \times n }$ , $\mathbf { G } ^ { v } \in \mathbb { R } ^ { k \times c }$ , and $\mathbf { H } ^ { v } \in \mathbb { R } ^ { c \times n }$ . Thus, the space complexity of Algorithm 1 is $O ( n )$ .

# Algorithm 1: ASCR algorithm

Input: Incomplete multi-view data $\{ \mathbf { X } ^ { v } \} _ { v = 1 } ^ { m }$ , parameter $\alpha$ , the number of anchor $k$ and cluster $c$ .   
Output:Perform $k$ -means on $\mathbf { H }$ .   
1: Initialize $\mathbf { E } ^ { v } = \mathbf { 0 }$ , $\mathbf { A } ^ { v } = \mathbf { 0 }$ , ${ \bf Z } ^ { v } = { \bf 0 }$ , $ { \mathbf { G } } ^ { v } = \mathbf { 0 }$ , $\mathcal { H } = 0$ , $\epsilon = 1 e - 5$ .   
2: Construct the index matrices $\{ \mathbf { Q } ^ { v } \} _ { v = 1 } ^ { m }$ and $\{ \mathbf { P } ^ { v } \} _ { v = 1 } ^ { m }$ . 3: while not converged do   
4: Update $\mathbf { E } ^ { v }$ by Eq. (6);   
5: Update $\mathbf { A } ^ { v }$ by solving (8);   
6: Update $\mathbf { Z } ^ { v }$ by Eq. (11);   
7: Update $\mathbf { G } ^ { v }$ by solving (13);   
8: Update $\mathcal { H }$ by solving. (14);   
9: Check the convergence conditions:   
$\begin{array} { r } { \sum _ { v = 1 } ^ { m } \| \mathbf H _ { t } ^ { v } - \mathbf H _ { t - 1 } ^ { \check { v } } \| _ { F } ^ { 2 } \big / \sum _ { v = 1 } ^ { m } \| \mathbf H _ { t - 1 } ^ { v } \| _ { F } ^ { 2 } \leq \epsilon ; } \end{array}$ ; 10: $t \longleftarrow t + 1$ ;   
11: end while   
12: Obtain $\mathbf { H }$ by concatenating $[ { \mathbf { H } ^ { 1 } } ^ { T } , { \mathbf { H } ^ { 2 } } ^ { T } , . . . , { \mathbf { H } ^ { m } } ^ { T } ] .$

# Experiment

# Experimental Setup

Datasets Eight popular datasets from diverse applications were used to validate our ASCR, consisting of text datasets $\mathbf { N G s } ^ { 1 }$ , digit dataset $ { \mathbf { H } }  { \mathbf { W } } ^ { 2 }$ , scene datasets Scene15 (Xie et al. 2018) and $\mathbf { S U N R G B D } ^ { 3 }$ , face dataset NHface (Cao et al. 2015), video dataset CCV (Wang et al. 2021), as well as object datasets Caltech1 ${ \bf 0 1 ^ { 4 } }$ and NUSWIDE (Chua et al. 2009). More details are shown in Table 1.

Incomplete Data Construction To construct incomplete multi-view data, following (Wen et al. 2020a), we set varying missing rates $p \in [ 0 . 1 : 0 . 2 : 0 . 9 ]$ to comprehensively investigate the robustness of our method against missing data. When the missing rate $p = 0 . 1$ , we randomly select $90 \%$ samples as complete data and randomly drop some views of the remaining $10 \%$ samples. At least one view is preserved for incomplete instances.

Evaluation Metrics Five commonly used metrics are employed to evaluate clustering performance: accuracy (ACC), normalized mutual information (NMI), purity (PUR), adjusted Rand index (ARI), and F-score. Higher values for each metric indicate better clustering performance.

Baselines We compare ASCR’s clustering performance with eight SOTA IMVC methods: IMVC-CBG (Wang et al. 2022), FIMVC-VIA (Liu et al. 2022), SIMVC-SA (Wen et al. 2023), DVSAI (Yu et al. 2024), LSIMVC (Liu et al. 2023), SEC-IMVC (Zhang et al. 2023b), TDASC (Chen et al. 2023), sFSR-IMVC (Long et al. 2023).

Parameter Settings The hyper-parameters in the baselines are tuned according to the corresponding papers. For ASCR, we fix the anchor number $k = c$ and tune $\alpha$ within the range $2 ^ { \{ 2 : 7 \} }$ . All methods are run 10 times to obtain clustering results and standard deviations for fair comparison. The experiments are conducted using MATLAB R2021a on a PC with an i5-12400 CPU and 16GB RAM.

Table 1: Details of the used datasets.   

<html><body><table><tr><td>Dataset</td><td>Sample</td><td>Cluster</td><td>View</td><td>Viewdimension</td></tr><tr><td>NGs</td><td>500</td><td>5</td><td>3</td><td>2000,2000,2000</td></tr><tr><td>HW</td><td>2000</td><td>10</td><td>6</td><td>240,76,216,47,64,6</td></tr><tr><td>Scene15</td><td>4485</td><td>15</td><td>3</td><td>1800,1180,1240</td></tr><tr><td>NHface</td><td>4660</td><td>5</td><td>3</td><td>6750,2000,3304</td></tr><tr><td>CCV</td><td>6773</td><td>20</td><td>3</td><td>20,20,20</td></tr><tr><td>Caltech101</td><td>9144</td><td>102</td><td>5</td><td>48,40,254,512,928</td></tr><tr><td>SUNRGBD</td><td>10335</td><td>45</td><td>2</td><td>4096,4096</td></tr><tr><td>NUSWIDE</td><td>30000</td><td>31</td><td>5</td><td>64,225,144,73,128</td></tr></table></body></html>

# Experimental Results

Figure 2 shows the ACC value of all approaches on all datasets with varying missing rates (5 cases). Additionally, Table 2 presents the average clustering results of the 5 cases on all datasets. The best and second-best results are highlighted in red and blue, respectively. ”OM” denotes unavailable results due to out-of-memory errors. Based on the results, we can draw the following conclusions:

(1) Our method generally significantly outperforms the baselines. For instance, on the CCV datasets, it achieves an average improvement of $3 4 . 2 1 \%$ , $3 6 . 9 0 \%$ , $3 2 . 4 6 \%$ , $2 7 . 9 6 \%$ , and $3 5 . 6 4 \%$ over the second-best method (i.e., sFSR-IMVC) in terms of five metrics, respectively. Notably, as shown in Figure 2, our method exhibits remarkable stability compared to others as the missing rate increases, demonstrating the robustness of our ASCR against missing data.

(2) Compared to both anchor-based (e.g., IMVC-CBG, FIMVC-VIA, SIMVC-SA, DVSAI, and TDASC) and graph-based (e.g., LSIMVC) methods, which rely on observed samples for anchor graph construction or consensus representation learning, our ASCR consistently outperforms them. As the missing rate increases, the superiority of ASCR becomes more pronounced, demonstrating the advantages of our adaptive similarity completion strategy.

(3) While SEC-IMVC and sFSR-IMVC tend to yield better results than the other five baselines in most datasets due to their spectral embedding completion or feature space recovery approach, our ASCR consistently achieves superior performance across nearly all datasets. This is attributed to our approach of reconstructing anchor-sample similarity graphs using sample embedding and anchor embedding. This process not only preserves similarities in latent embeddings but also enhances the low-rank property of similarity graphs, leading to a more robust completion process and obtaining more discriminative embeddings.

# Parameter Analysis

To assess the influence of $\alpha$ and the anchor number $k$ in ASCR, Figure 3 shows ACC values on NGs and NHface for different $\alpha$ and $k$ . The results show that ASCR is not sensitive to $k$ , consistently delivering excellent and stable clustering performance. Therefore, we set $k = c$ for all datasets to

NGs HW Scene15 NHface 100 100 100 80 680 460 80 60 + 冏甲 u 书 ↑ ? 60 40 40 A 0.1 0.3 0.5 0.7 0.9 0.1 0.3 0.5 0.7 0.9 0.1 0.3 0.5 0.7 0.9 0.1 0.3 0.5 0.7 0.9 Missing Ratio Missing Ratio Missing Ratio Missing Ratio CCV Caltech101 SUNRGBD NUSWIDE 60 30 20 60 20 20 1 20 H H --=-=-- o 10 0.1 0.3 0.5 0.7 0.9 0.1 0.3 0.5 0.7 0.9 0.1 0.3 0.5 0.7 0.9 0.1 0.3 0.5 0.7 0.9 Missing Ratio Missing Ratio Missing Ratio Missing Ratio IMVC-CBG FIMVC-VIA SIMVC-SA DVSAI → LSIMVC SEC-IMVC TDASC sFSR-IMVC Ours

Figure 2: Clustering performances w.r.t. ACC on 8 datasets with varying missing rates.   

<html><body><table><tr><td></td><td>Metric(%)</td><td>IMVC-CBG</td><td>FIMVC-VIA</td><td>SIMVC-SA</td><td>DVSAI</td><td>LSIMVC</td><td>SEC-IMVC</td><td>TDASC</td><td>sFSR-IMVC</td><td>Ours</td></tr><tr><td rowspan="5">SON</td><td>ACC</td><td>64.65±0.42</td><td>66.38±0.46</td><td>68.14±0.22</td><td>86.37±0.16</td><td>50.44±0.21</td><td>79.40±0.00</td><td>76.17±0.29</td><td>70.95±1.33</td><td>99.60±0.00</td></tr><tr><td></td><td></td><td>57.38±0.27</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PMR</td><td>53.73±0.371</td><td></td><td>58.08±0.28</td><td>71.01±0.16</td><td>33.86±0.22</td><td>56.62±0.00</td><td>62.78±0.20</td><td>6.57±1.8</td><td>99.60±0.00</td></tr><tr><td>ARI</td><td>40.63±0.26</td><td>47.29±0.26</td><td>50.82±0.19</td><td>70.45±0.04</td><td>19.49±0.18</td><td>56.83±0.00</td><td>54.28±0.24</td><td>60.99±1.71</td><td>99.00±0.00</td></tr><tr><td>F-score</td><td>46.20±0.27</td><td>47.54±0.15</td><td>52.64±0.55</td><td>77.52±0.06</td><td>36.56±0.15</td><td>65.68±0.00</td><td>62.00±0.23</td><td>69.41±1.21</td><td>99.20±0.00</td></tr><tr><td rowspan="7">MH</td><td>ACC</td><td>56.20±0.60</td><td>56.53±0.67</td><td>60.43±0.39</td><td>64.71±0.32</td><td>86.85±1.56</td><td>93.24±0.00</td><td>60.31±0.45</td><td>50.50±3.20</td><td>99.38±0.03</td></tr><tr><td>NMI</td><td>43.74±0.35</td><td>44.87±0.59</td><td>50.62±0.42</td><td>53.78±0.37</td><td>80.85±1.15</td><td>86.57±0.00</td><td>57.94±0.42</td><td>59.25±4.97</td><td>98.47±0.07</td></tr><tr><td>PUR</td><td>56.98±0.62</td><td>57.91±0.67</td><td>63.27±0.38</td><td>64.55±0.40</td><td>86.65±1.32</td><td>93.24±0.00</td><td>67.05±0.31</td><td>50.72±5.27</td><td>99.38±0.03</td></tr><tr><td>ARI</td><td>36.76±0.51</td><td>35.97±0.42</td><td>41.52±0.29</td><td>39.03±0.34</td><td>83.92±0.81</td><td>85.84±0.00</td><td>58.06±0.54</td><td>43.27±4.33</td><td>98.61±0.06</td></tr><tr><td>F-score</td><td>43.92±0.37</td><td>44.72±0.37</td><td>50.19±0.31</td><td>47.53±0.52</td><td>80.54±0.91</td><td>87.25±0.00</td><td>54.09±0.32</td><td>51.56±4.01</td><td></td></tr><tr><td>ACC</td><td>31.57±1.43</td><td>28.96±0.67</td><td>36.83±1.96</td><td>29.79±1.79</td><td>33.71±1.27</td><td>49.35±0.00</td><td>43.09±0.31</td><td>76.98±5.12</td><td>98.75±0.06</td></tr><tr><td>NMI</td><td>26.74±0.38</td><td>29.21±0.49</td><td>35.47±0.62</td><td>31.73±0.39</td><td>36.63±0.58</td><td>43.46±0.00</td><td>37.34±0.19</td><td>84.27±2.46</td><td>86.01±3.78</td></tr><tr><td rowspan="6"></td><td>PUR</td><td>30.27±1.16</td><td>28.90±0.58</td><td>40.13±0.81</td><td>35.13±0.69</td><td>37.40±0.95</td><td>51.10±0.00</td><td>40.98±0.37</td><td>82.32±3.76</td><td>90.90±1.35</td></tr><tr><td>ARI</td><td>20.40±0.92</td><td>19.58±0.24</td><td>20.86±0.54</td><td>22.36±1.21</td><td>16.06±0.90</td><td>30.02±0.00</td><td>27.79±0.19</td><td>72.08±5.40</td><td>89.81±2.35</td></tr><tr><td>F-score</td><td>27.59±0.35</td><td>27.54±0.20</td><td>32.59±0.49</td><td>30.32±0.54</td><td>23.35±0.74</td><td></td><td></td><td></td><td>83.12±3.93</td></tr><tr><td>ACC</td><td>67.51±0.05</td><td>70.84±1.04</td><td></td><td></td><td></td><td>35.03±0.00</td><td>29.85±0.20</td><td>74.17±4.95</td><td>84.31±3.57</td></tr><tr><td>NMI</td><td>59.12±0.17</td><td>57.52±0.60</td><td>75.58±0.14</td><td>72.36±0.09 61.11±0.20</td><td>92.19±0.46</td><td>58.00±0.00</td><td>73.86±0.00</td><td>88.80±0.07</td><td>97.48±0.01</td></tr><tr><td>PUR</td><td>68.91±0.04</td><td>68.59±0.25</td><td>64.45±0.09 75.20±0.16</td><td></td><td>84.70±0.73</td><td>40.10±0.00</td><td>64.07±0.01</td><td>82.29±0.39</td><td>94.19±0.01</td></tr><tr><td rowspan="6">CCC</td><td>ARI</td><td>65.79±0.11</td><td>59.49±1.37</td><td></td><td>72.76±0.09 58.02±0.23</td><td>92.28±0.46</td><td>63.16±0.00</td><td>72.02±0.00</td><td>84.88±0.07</td><td>97.48±0.01</td></tr><tr><td>F-score</td><td>60.03±0.09</td><td>57.28±0.81</td><td>63.67±0.11</td><td>59.69±0.27</td><td>85.80±0.85 88.93±0.49</td><td>34.34±0.00 49.15±0.00</td><td>60.73±0.01 69.14±0.00</td><td>83.56±0.64 86.85±0.34</td><td>95.43±0.01</td></tr><tr><td>ACC</td><td>15.81±0.36</td><td>17.61±0.25</td><td>60.44±0.08</td><td></td><td></td><td></td><td></td><td></td><td>96.33±0.01</td></tr><tr><td></td><td></td><td></td><td>17.43±0.32</td><td>19.85±0.29</td><td>18.57±0.34</td><td>17.39±0.00</td><td>14.81±0.03</td><td>30.25±1.45</td><td>64.46±2.72</td></tr><tr><td>PUR</td><td>11.43±0.48</td><td>12.21±0.29</td><td>10.314±0.22</td><td>15.32±0.27</td><td>16.07±0.19</td><td>11.94±0.00</td><td>19.4±0.09</td><td>34.23±0.71</td><td>71.13±0.77</td></tr><tr><td>ARI</td><td>14.11±0.36</td><td>14.30±0.11</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4"></td><td>F-score</td><td>9.96±0.21</td><td>10.54±0.20</td><td>14.36±0.11</td><td>16.08±0.28</td><td>15.69±0.11</td><td>14.77±0.00</td><td>12.56±0.04</td><td>26.71±0.85</td><td>54.67±2.15</td></tr><tr><td></td><td>2.77±10.97</td><td></td><td>10.61±0.11</td><td>12.37±0.32</td><td>11.24±0.12</td><td>10.12±0.00</td><td>9.00±0.03</td><td>21.50±0.75</td><td>57.14±2.00</td></tr><tr><td>AMC</td><td></td><td>18.02±0.54</td><td>137.36±0.4</td><td>31.52±0.3</td><td>43.0±2.46</td><td>38.28±0.00</td><td>24.48±0.40</td><td>4.5±1.4</td><td></td></tr><tr><td>PUR</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>58.76±2.15</td></tr><tr><td></td><td></td><td>42.15±0.52</td><td>30.55±0.43</td><td>29.92±0.45</td><td>48.43±0.32</td><td>59.85±2.43</td><td>55.65±0.00</td><td>40.65±0.24</td><td>72.33±0.73</td><td>81.69±1.45</td></tr><tr><td></td><td>ARI</td><td>10.90±0.58</td><td>4.90±0.48</td><td>5.98±0.50</td><td>19.62±0.39</td><td>23.19±0.40</td><td>23.26±0.00</td><td>16.90±0.50</td><td>36.45±1.43</td><td>45.22±2.86</td></tr><tr><td></td><td>F-score</td><td>14.95±1.51</td><td>8.81±0.31</td><td>9.66±0.40</td><td>20.96±0.26</td><td>20.72±1.02</td><td>24.48±0.00</td><td>18.44±0.46</td><td>37.50±1.42</td><td>46.16±2.64</td></tr><tr><td></td><td>ACC</td><td>13.78±0.35</td><td>14.85±0.36</td><td>15.53±0.40</td><td>20.29±0.44</td><td>14.59±0.35</td><td>OM</td><td>14.29±0.40</td><td>22.75±1.04</td><td>23.16±0.86</td></tr><tr><td></td><td>NMI</td><td>19.27±0.42</td><td>18.38±0.27</td><td>19.76±0.31</td><td>21.48±0.31</td><td>20.88±0.17</td><td>OM</td><td>13.62±0.14</td><td>27.61±1.05</td><td>39.05±0.36</td></tr><tr><td></td><td>PUR</td><td>29.19±0.35</td><td>30.01±0.42</td><td>31.85±0.52</td><td>32.63±0.30</td><td>32.24±0.33</td><td>OM</td><td>23.45±0.16</td><td>36.58±0.89</td><td>47.73±0.67</td></tr><tr><td></td><td>ARI</td><td>5.16±0.22</td><td>5.58±0.11</td><td>5.62±0.14</td><td>5.82±0.20</td><td>6.04±0.14</td><td>OM</td><td>2.59±0.07</td><td>12.14±1.08</td><td>14.24±0.65</td></tr><tr><td></td><td>F-score</td><td>8.87±0.32</td><td>10.30±0.11</td><td>9.85±0.14</td><td>10.41±0.17</td><td>9.48±0.15</td><td>OM</td><td>10.39±0.17</td><td>16.87±1.01</td><td>17.36±0.61</td></tr><tr><td></td><td>ACC</td><td>12.06±0.33</td><td>12.10±0.20</td><td>10.86±0.13</td><td>16.55±0.21</td><td>OM</td><td>OM</td><td>11.58±0.12</td><td>13.13±0.65</td><td>16.42±0.29</td></tr><tr><td>JOIMSON</td><td>NMI PUR</td><td>9.89±0.10 21.24±0.37</td><td>8.16±0.10 19.87±0.15</td><td>10.24±0.18 15.81±0.18</td><td>11.67±0.24 21.59±0.13</td><td>OM OM</td><td>OM OM</td><td>8.59±0.11 20.21±0.13</td><td>9.54±0.40 19.41±0.58</td><td>18.79±0.11 28.22±0.25</td></tr><tr></table></body></html>

Table 2: Clustering results (mean±std) of different methods on 8 datasets. OM indicates out of memory.

balance time consumption and clustering performance. The parameter $\alpha$ controls the weight of tensor low-rank regularization, and satisfactory results are obtained with $\alpha$ values from $2 ^ { \{ 2 : 5 \} }$ for NGs and $2 ^ { \{ 3 : 6 \} }$ for NHface.

# Convergence and Time Comparison

For Algorithm 1, we set the convergence condition as $\begin{array} { r } { L o s s ( \bar { \mathbf H } ) \ = \ \sum _ { v = 1 } ^ { m } \| \mathbf H _ { t } ^ { v } \ - \ \mathbf H _ { t - 1 } ^ { v } \| _ { F } ^ { 2 } / \sum _ { v = 1 } ^ { m } \| \mathbf H _ { t - 1 } ^ { v } \| _ { F } ^ { 2 } \ \leq \ \qquad } \end{array}$ $1 e - 5$ . Figu  4 displays the change of $L o s s ( \mathbf { H } )$ and ACC

![](images/d5ae3e82908403095179055b172232566db159010bd4608564e059935696db0c.jpg)  
Figure 3: Clustering performance of ASCR under different parameter settings on NGs and NHface datasets $( p = 0 . 1 )$ .   
Figure 4: Convergence curves and the change of ACC value on NGs and NHface datasets $( p = 0 . 1 )$ ).

NGs NHface P-O-O-O-O-O-O-O-0-0-O-O-O-O-O-O-O-OH- 100 H-- 卢店 15 Loss(H) Loss(H) 95   
U ACC 99 500 ACC 90 10-4 10   
5 1 85 98 0.5 0.5 80 0 00000000000000000 97 0 POO00O00OCO0OO 75 10 20 30 10 20 30 0 100000000000000000000000000000 0 00000000000000000000000000000 70 10 20 30 10 20 30 Number of iterations Number of iterations

Table 3: Running time (second) of all methods on 8 datasets.   

<html><body><table><tr><td>Method</td><td>NGs</td><td>HW</td><td>Scene15</td><td>NHface</td><td>CCV</td><td>Caltech</td><td>SUN</td><td>NUS</td></tr><tr><td>IMVC-CBG</td><td>0.68</td><td>1.09</td><td>5.14</td><td>8.85</td><td>5.54</td><td>27.96</td><td>25.62</td><td>74.56</td></tr><tr><td>FIMVC-VIA</td><td>0.31</td><td>0.51</td><td>2.29</td><td>3.76</td><td>1.92</td><td>5.87</td><td>13.46</td><td>18.29</td></tr><tr><td>SIMVC-SA</td><td>0.39</td><td>0.94</td><td>3.28</td><td>8.65</td><td>8.61</td><td>36.41</td><td>15.68</td><td>82.24</td></tr><tr><td>DVSAI</td><td>3.26</td><td>0.85</td><td>3.72</td><td>6.83</td><td>12.25</td><td>45.29</td><td>69.25</td><td>164.82</td></tr><tr><td>LSIMVC</td><td>5.12</td><td>3.43</td><td>2.85</td><td>5.89</td><td>6.97</td><td>19.52</td><td>17.71</td><td>OM</td></tr><tr><td>SEC-IMVC</td><td>0.81</td><td>29.27</td><td>170.89</td><td>131.96</td><td>556.02</td><td>2135.67</td><td>OM</td><td>OM</td></tr><tr><td>TDASC</td><td>1.12</td><td>4.54</td><td>8.05</td><td>11.53</td><td>9.26</td><td>62.03</td><td>42.98</td><td>128.78</td></tr><tr><td>sFSR-IMVC</td><td>6.08</td><td>46.11</td><td>52.57</td><td>63.19</td><td>5.77</td><td>47.05</td><td>276.8</td><td>354.22</td></tr><tr><td>Ours</td><td>0.1</td><td>0.32</td><td>0.6</td><td>1.99</td><td>1.45</td><td>15.53</td><td>9.27</td><td>20.41</td></tr></table></body></html>

values w.r.t. iterations on NGs and NHface datasets. It is noticeable that $L o s s ( \mathbf { H } )$ decreases rapidly and converges to a stable value within 30 iterations, which satisfies our convergence condition. Meanwhile, the ACC value also increases and remains stable quickly. The outcomes demonstrate the favorable convergence property of our algorithm.

Table 3 shows the running time of all methods across benchmark datasets. Our ASCR demonstrates competitive computational efficiency compared to most baselines. SECIMVC has $O ( n _ { v } ^ { 3 } )$ complexity to learn spectral embeddings, and it is not efficient in practice. Although some anchor-based methods have $O ( n )$ complexity, they compute the anchor-sample similarities for each sample one-by-one. However, all variables of ASCR are updated directly with closed-form solutions. Thus, our method generally spends less time than those methods. This efficiency highlights the potential of ASCR for large-scale incomplete tasks in practical applications.

# Ablation Study

To assess the effectiveness of the anchor-sample similarity graph completion and reconstruction strategy, as well as the tensor-based regularization, we derive three variants based on the original ASCR model: ASCR-C, ASCR-R, and ASCR-T. ASCR-C drops the similarity completion matrix $\mathbf { E } ^ { v }$ and just uses the observed samples to construct anchor graphs. ASCR-R replace $\mathbf { G } ^ { v } \mathbf { H } ^ { v }$ with $\mathbf { S } ^ { v }$ and perform SVD on ${ \frac { 1 } { m } } \sum _ { v = 1 } ^ { m } \mathbf { S } ^ { v }$ to get latent embedding U and then apply $k$ -means like TDASC. ASCR-T omits tensor nuclear-norm regularization and utilizes traditional nuclear norm on $\mathbf { H } ^ { v }$ .

Table 4: Ablation study results w.r.t. ACC values.   

<html><body><table><tr><td>Datasets</td><td>Method</td><td>p=0.1</td><td>p=0.3</td><td>p=0.5</td><td>p=0.7</td><td>p=0.9</td></tr><tr><td>NGs</td><td>ASCR-C ASCR-R ASCR-T ASCR</td><td>90.33 98.62 87.56 100</td><td>86.54 97.14 78.91 99.60</td><td>82.71 97.14 71.29 99.60</td><td>78.62 94.62 64.88 99.20</td><td>75.92 96.62 57.25 99.60</td></tr><tr><td>HW</td><td>ASCR-C ASCR-R ASCR-T ASCR</td><td>87.51 98.30 82.16 99.53</td><td>85.42 97.45 75.27 99.49</td><td>81.97 97.31 67.21 98.99</td><td>78.04 96.90 60.03 99.34</td><td>72.59 97.03 51.97 99.54</td></tr><tr><td>Scene15</td><td>ASCR-C ASCR-R ASCR-T ASCR</td><td>81.75 85.90 67.59 86.18</td><td>77.96 84.56 59.99 85.86</td><td>74.13 81.66 51.25 85.08</td><td>70.29 83.18 48.56 86.58</td><td>67.84 81.08 42.95 86.36</td></tr><tr><td>NHface</td><td>ASCR-C ASCR-R ASCR-T ASCR</td><td>88.25 94.66 86.55 98.10</td><td>83.27 94.96 79.24 97.45</td><td>80.52 93.88 73.57 96.95</td><td>77.21 94.61 64.85 97.56</td><td>73.19 93.56 53.92 97.33</td></tr></table></body></html>

Table 4 shows the clustering performance of the three variants and our ASCR with different missing ratios. We can observe that the performance of ASCR-C is not desirable. Especially when the missing rate increases, the ACC value decreases sharply. This proves the effectiveness of our anchor-sample similarity completion. ASCR-R can achieve very competitive results. However, our ASCR still outperforms it with large missing rates. It is attributed to our reconstruction strategy, which seeks the latent embedding for a low-rank graph completion process and then achieves more optimal clustering results. In addition, the poor performance of ASCR-T also demonstrates the effectiveness of tensor regularization on $\mathbf { H } ^ { v }$ , which helps to fully explore the highorder cross-view correlations, and improves the semantic consistency discovery.

# Conclusion

In this paper, we propose a novel fast IMVC approach with Adaptive Similarity Completion and Reconstruction (ASCR), which unifies anchor learning, anchor-sample similarity completion, and latent multi-view embedding learning into a joint framework. ASCR learns an anchor-sample similarity graph for each view and fulfills the missing values to mitigate the adverse effects. To explore consistent and complementary information across views, ASCR simultaneously seeks latent view-specific anchor embedding and sample embedding by reconstructing similarity relations. It not only preserves the similarities into latent embeddings but also enhances the low-rank property of similarity graphs, achieving a robust graph completion process and mutual reinforcement. Furthermore, the high-order cross-view semantic consistency is captured via low-rank tensor regularization. Extensive experimental results on several datasets demonstrate the superiority and efficiency of our proposed ASCR compared with state-of-the-art approaches.

# Acknowledgments

This work was supported by the National Natural Science Foundation of China under Grants 62176116, 62276136, 62073160.