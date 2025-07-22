# Single-View Graph Contrastive Learning with Soft Neighborhood Awareness

Qingqiang $\mathbf { S u n } ^ { 1 }$ , Chaoqi Chen2, Ziyue Qiao1\*, Xubin Zheng1, Kai Wang3†,

1Great Bay University 2Shenzhen University 3Central South University {qqsun, zyqiao, xbzheng}@gbu.edu.cn, cqchen1994 $@$ gmail.com, kaiwang@csu.edu.cn

# Abstract

Most graph contrastive learning (GCL) methods heavily rely on cross-view contrast, thus facing several concomitant challenges, such as the complexity of designing effective augmentations, the potential for information loss between views, and increased computational costs. To mitigate reliance on cross-view contrasts, we propose SIGNA, a novel single-view graph contrastive learning framework. Regarding the inconsistency between structural connection and semantic similarity of neighborhoods, we resort to soft neighborhood awareness for GCL. Specifically, we leverage dropout to obtain structurally-related yet randomly-noised embedding pairs for neighbors, which serve as potential positive samples. At each epoch, the role of partial neighbors is switched from positive to negative, leading to probabilistic neighborhood contrastive learning effect. Moreover, we propose a normalized Jensen-Shannon divergence estimator for a better effect of contrastive learning. Experiments on diverse node-level tasks demonstrate that our simple single-view GCL framework consistently outperforms existing methods by margins of up to $2 1 . 7 4 \%$ $( P P I )$ . In particular, with soft neighborhood awareness, SIGNA can adopt MLPs instead of complicated GCNs as the encoder in transductive learning tasks, thus speeding up its inference process by $1 0 9 \times$ to $3 3 1 \times$ .

# Code — https://github.com/sunisfighting/SIGNA

# Introduction

Over the past few years, Self-Supervised Learning (SSL) of representations has emerged as a promising and popular research topic since it does not rely on the quantity and quality of labels (Wu et al. 2021; Liu et al. 2022; Xie et al. 2022). As one of the most competitive SSL paradigms, Contrastive Learning (CL) has made its mark in computer vision (He et al. 2020; Chen et al. 2020), natural language processing (Tian, Krishnan, and Isola 2020), and graph domains (Velickovic et al. 2019; Zhu et al. 2020b; Peng et al. 2020b; Sun et al. 2024).

The core spirit of CL is to pull together representations of related instances (positive samples) and push apart representations of unrelated pairs (negative samples). Similar

contrast contrast contrast Augmented Augmented AugNmoen-ted AugNmoen-ted Latent Latent Latent View 1 Latent View 2 Latent Latent View View View 1 View 2 contrast 自 月 月 自 T 目 R 一 T 二 Input View 1 Input View Input View Input View (a) (b) (c) (d)

to CL in computer vision and natural language processing domains, Graph Contrastive Learning (GCL) methods typically rely on cross-view contrasts, i.e., each positive or negative pair consists of two latent embeddings from different views. These strategies include augmentation-based dualbranch methods (Zhu et al. 2020b; You et al. 2020; Shen et al. 2023), non-augmentation-based dual-branch methods (Thakoor et al. 2021; Mo et al. 2022), and input-latent single-branch methods (Peng et al. 2020b), which are summarized in Fig. 1 and discussed in detail in the related work.

Despite the significant progress achieved by cross-view contrastive learning methods on graph data, these approaches face several challenges. Firstly, designing effective augmentation techniques often requires substantial manual effort and fine-tuning, leading to increased implementation complexity. Secondly, cross-view contrast may result in information loss or inconsistencies between views, which can negatively impact model performance. Lastly, the computational cost of cross-view methods is typically high, especially for those dual-branch ones, making them less scalable.

To circumvent these challenges, this paper seeks to explore a simpler yet little-studied alternative: single-view contrastive learning. The most pivotal point is how to obtain intra-view positive and negative samples. Note that nodes in the graph are intricately correlated with each other rather than completely independent (Dang et al. 2021; Wang et al. 2022; Yu, Shi, and Wang 2024; Wang et al. 2023, 2024). Therefore, a natural potential solution within this framework is to conduct contrast according to the topological relationship between nodes, i.e., pull embeddings of neighbors together and push those of non-neighbors away. However, such a trivial solution may be prone to overfitting topological information and thus results in suboptimal performance. A typical example is GAE (Kipf and Welling 2016b), which aims to reconstruct the graph structure as much as possible but can only achieve less competitive performance than other self-supervised methods that ignore topological information, such as GRACE (Zhu et al. 2020b). In such a context, two questions are naturally raised: (i) What role should neighborhoods play in node-level GCL? (ii) Can neighborhood awareness help single-view contrastive learning surpass cross-view contrastive learning?

Regarding these two questions, we first investigate the homophily nature across diverse real-world datasets from both global and local perspectives. Statistical results indicate that there is an obvious inconsistency between structural connection and semantic similarity. Inspired by this, we propose Single-vIew Graph contrastive learning with soft Neighborhood Awareness (SIGNA). Instead of relying on augmentations or other cross-view contrastive techniques to generate contrastive pairs, SIGNA adopts soft neighborhood awareness to realize single-view contrast. Specifically, we first use dropout to obtain randomly-perturbed embeddings, which implicitly provides more diverse embedding combinations for contrast. At each epoch, the role of neighbors is allowed to switch from positive to negative rather than being consistently designated as positive, thus leading to probabilistic neighborhood contrastive learning. Furthermore, we propose a normalized Jensen-Shannon divergence (NormJSD) estimator, which combines the advantages of both JSD and InfoNCE, resulting in a better contrastive effect.

We evaluate SIGNA on three kinds of node-level tasks, including transductive node classification, inductive node classification, and node clustering. SIGNA consistently outperforms existing methods across all tasks, achieving performance gains of up to $2 1 . 7 4 \%$ $( P P I )$ . In particular, thanks to reasonable neighborhood awareness, SIGNA is enabled to use a simple Multilayer Perceptron (MLP) as the encoder in transductive learning tasks, which is $1 0 9 \times$ to $3 3 1 \times$ faster than a GCN-based encoder with the same settings in terms of the inference time. The distribution analysis and visualization of learned representations showcase the superiority of SIGNA in striking a better balance between intra-class aggregation and inter-class separation, compared with representative neighborhood-aware GCL methods. In summary, our contributions can be highlighted as follows:

We study the role of neighborhoods by comprehensively investigating their homophily nature on real-world datasets from both global and local perspectives, thus offering nuanced insights on soft neighborhood awareness.

We propose a simple yet effective single-view GCL framework, SIGNA, which alleviates the heavy reliance on traditional cross-view contrastive learning with the help of our soft neighborhood awareness strategy.

Extensive experiments demonstrate that SIGNA outperforms existing methods across various node-level tasks, showcasing the feasibility of single-view GCL and the rationality of soft neighborhood awareness.

# Related Works

To position our contributions in the literature, we briefly review related works here.

GCL Paradigms. Existing GCL methods mainly rely on cross-view contrast, which can be further grouped into three classes, i.e., augmentation based cross-view contrast, non-augmentation based cross-view contrast, and inputlatent cross-view contrast, as illustrated in Fig. 1. (a)- (c). (a) Augmentation based cross-view contrast is represented by DGI (Velickovic et al. 2019), MVGRL (Hassani and Khasahmadi 2020), GRACE (Zhu et al. 2020b), GraphCL (You et al. 2020), GCC (Qiu et al. 2020), and MERIT (Jin et al. 2021), NCLA (Shen et al. 2023), etc. Augmented views from the same instance are pulled closer while those views from distinct instances are pushed away. (b) Instead of augmenting input data, non-augmentation based cross-view contrast resorts to using two discrepant encoders or directly perturbing latent embeddings to obtain contrastive pairs, such as SUGRL (Mo et al. 2022), AFGRL (Lee, Lee, and Park 2022), SimGRACE (Xia et al. 2022), SimGCL (Yu et al. 2022), and COSTA (Zhang et al. 2022b). (c) Input-latent cross-view contrast is represented by GMI (Peng et al. 2020b), which aims to maximize mutual information between the input graph and latent embeddings.

In this paper, we aim to study the simpler single-view GCL, which has seldom been studied in the literature.

Neighborhood-aware GCL. Regarding the limitations of augmentation-invariant GCL, concurrent studies turn to neighborhood-aware techniques to retain semantics. For example, GMI (Peng et al. 2020b) maximizes the mutual information across both feature and edge representations between the input and output spaces. GraphCL-NS (Hafidi et al. 2022) makes use of graph structure to sample negatives, i.e., from $l$ -th order neighbors of the anchor node. Based on NTXent (Zhu et al. 2020b), NCLA (Shen et al. 2023) adopts a neighbor contrastive loss that regards both intra-view and inter-view neighbors as positives. Without generating augmented views, Local-GCL (Zhang et al. 2022a) fabricates positive samples for each node using first-order neighbors. Instead of binary contrastive justification, GSCL (Ning et al. 2022) uses fine-grained contrastive justification according to the hop distance of neighborhoods. AFGRL (Lee, Lee, and Park 2022) obtains local positives by jointly considering the adjacency matrix and k-NNs in the embedding space. SUGRL (Mo et al. 2022) employs both GCN and MLP encoders to obtain contrastive pairs. For each anchor node, its positive counterparts are constructed with its GCN output and the MLP outputs of its neighbors.

Unlike existing works that fully trust neighborhoods to capture semantic information for downstream tasks, we resort to soft neighborhood awareness to prevent the encoder from overfitting uncertain signals.

# Rethinking the Role of Neighborhoods

To implement effective sigle-view contrastive learning, it is of great significance to understand what role neighbours should play in graph contrastive learning. Thus, we first empirically investigate the homophily nature of neighborhoods.

![](images/157a53753935a67f6c69ce701e8085ce86d4bfb94d88d571c113c047ef72810d.jpg)  
Figure 2: Global and local homophily statistics. Top: global homophily ratios on different datasets. Bottom Left: the distribution of local homophily counts on Photo. Bottom Right: the distribution of local homophily ratios on Photo.

Let $\mathcal { G } ~ = ~ ( \nu , \mathcal { E } )$ be an unweighted graph with a node set $\nu$ and an edge set $\mathcal { E }$ . We denote the feature matrix and adjacency matrix by $\mathbf { X } \ = \ \{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { | \mathcal { V } | } \ \in \ \mathbb { R } ^ { | \mathcal { V } | \times F }$ and $\mathbf { A } \in \{ 0 , 1 \} ^ { | \mathcal { V } | \times | \mathcal { V } | }$ , respectively. The one-hot label matrix is denoted by $\mathbf { Y } = \{ \mathbf { y } _ { i } \} _ { i = 1 } ^ { | \mathcal { V } | } \in \mathbb { R } ^ { | \mathcal { V } | \times c }$ . Following (Zhu et al. 2020a), the definition of global homophily ratio is given by: Definition 0.1 (Global Homophily Ratio). Given a graph $\mathcal { G } = ( \nu , \mathcal { E } )$ , its global homophily ratio is defined as the probability that two connected nodes share the same label:

$$
\mathcal { H } _ { g l o b a l } = \frac { 1 } { | \mathcal { E } | } \sum _ { u , v \in \mathcal { V } } \mathbb { I } [ ( u , v ) \in \mathcal { E } ] \cdot \mathbb { I } [ \mathbf { y } _ { u } = \mathbf { y } _ { v } ] .
$$

Furthermore, we define local homophily count and local homophily ratio as well:

Definition 0.2 (Local Homophily Count). Given a node $u \in$ $\nu$ and its one-hop neighbors $\mathcal { N } _ { u }$ , its local homophily count is defined as the number of neighbors with the same label:

$$
\begin{array} { r } { \mathcal { H } _ { l o c a l } ^ { \# } ( u ) = \sum _ { v \in { \mathcal { N } } _ { u } } \mathbb { I } [ { \mathbf { y } } _ { u } = { \mathbf { y } } _ { v } ] . } \end{array}
$$

Definition 0.3 (Local Homophily Ratio). Given a node $u \in$ $\nu$ and its one-hop neighbors $\mathcal { N } _ { u }$ , its local homophily ratio is defined as the probability that its neighbors share the same label with it:

$$
\mathcal { H } _ { l o c a l } ( u ) = \frac { 1 } { \left| \mathcal { N } _ { u } \right| } \sum _ { v \in \mathcal { N } _ { u } } \mathbb { I } [ \mathbf { y } _ { u } = \mathbf { y } _ { v } ] .
$$

We compute global homophily ratios $\mathcal { H } _ { g l o b a l }$ on six realworld datasets, and explore the distributions of local homophily counts $\{ \mathcal { H } _ { l o c a l } ^ { \# } ( u ) : u \in \mathcal { V } \}$ and ratios $\{ \mathcal { H } _ { l o c a l } ( u ) :$ $u \in \mathcal V \}$ on Amazon Photo. The statistics are illustrated in Figure 2. As can be observed: (a) the global homophily ratio varies considerably across datasets, with values ranging from 0.3195 (Flickr) to $0 . 9 3 1 4 \ ( P h y s i c s )$ , indicating that it is ubiquitous that structural connections between neighbors do not coincide with their semantic relations; and (b) although local homophily is relatively more concentrated in smaller counts and greater ratios, the overall distribution is hard to estimate, let alone identify exact semantically positive neighbors in the context of unsupervised learning (Sun, Zhang, and Lin 2023). In a nutshell, overemphasizing neighborhood affinity in contrastive learning is risky since the model would be provided with noisy and even detrimental learning signals, which result in suboptimal performance. Motivated by the above findings, we consider soft neighborhood awareness to realize single-view GCL.

# Methodology

The overall framework of our proposed SIGNA is depicted in Figure 3. SIGNA employs a single-branch paradigm for single-view contrast, eliminating the need for augmented inputs, disparate encoders, or perturbed embeddings to generate multiple views. Simply pulling embeddings of neighbors and pushing away those of non-neighbors has shown to be less competitive (Kipf and Welling 2016b). To pursue soft neighborhood awareness, SIGNA is equiped with three main components, i.e., the encoder with dropout, the stochastic neighbor masking, and the normalized Jensen-Shannon divergence (Norm-JSD) estimator.

Encoder with Dropout. Dropout has been adopted as minimal data augmentation in contrastive learning (Gao, Yao, and Chen 2021; Xu et al. 2023). To obtain positive pairs, they pass the same instance to the dropout-contained encoder twice so as to generate two related yet discrepant embeddings. By contrast, we only pass the input graph to the encoder with dropouts ONCE. In this way, the embedding pairs of neighbors are structurally related yet randomly noised, making them competent for positive samples in single-view contrastive learning. Note that we do not rely on augmentations to generate contrastive pairs. The introduction of random noise just implicitly creates more diverse embedding combinations for contrast. Thus, the encoder is less likely to overfit the relationship between node pairs when conducting contrastive learning.

Specifically, the encoder is composed of a stack of $L$ identical layers (in this paper, we fix $L = 2$ to match the common depth of existing works). Each layer has a BaseEncoder which can either be a linear layer (on transductive tasks) or a graph convolutional layer (Kipf and Welling 2016a) (on inductive tasks). Before each BaseEncoder, we employ Dropout (with rescaling) (Srivastava et al. 2014) to inject noise. The output of BaseEncoder is further processed by a nonlinear Activation function and a LayerNorm. Let $\dot { \mathbf { H } } ^ { ( l - 1 ) }$ be the input embeddings of $l$ -th encoder layer (particularly, $\mathbf { H } ^ { ( 0 ) } = \mathbf { X } ,$ ), the forward encoding process within an encoder layer can be formally described as:

$$
\mathbf { H } _ { \mathrm { d } } ^ { ( l ) } = \operatorname { D r o p o u t } ( \mathbf { H } ^ { ( l - 1 ) } ; p ) ,
$$

$$
\mathbf { H } _ { \mathrm { e } } ^ { ( l ) } = \mathrm { L i n e a r } ( \mathbf { H } _ { \mathrm { d } } ^ { ( l ) } ) \ \mathrm { o r } \ \mathbf { H } _ { \mathrm { e } } ^ { ( l ) } = \mathrm { G C o n v } ( \mathbf { H } _ { \mathrm { d } } ^ { ( l ) } , \mathbf { A } ) ,
$$

$$
{ \bf H } _ { \mathrm { a } } ^ { ( l ) } = \mathrm { A c t i v a t i o n } ( { \bf H } _ { \mathrm { e } } ^ { ( l ) } ) ,
$$

$$
\mathbf { H } ^ { ( l ) } = \mathbf { H } _ { \mathrm { n } } ^ { ( l ) } = \mathrm { L a y e r N o r m } ( \mathbf { H } _ { \mathrm { a } } ^ { ( l ) } ) ,
$$

where $p$ denotes the dropout rate, $\begin{array} { r l } { \mathrm { L i n e a r } ( \mathbf { H } _ { \mathrm { d } } ^ { ( l ) } ) } & { { } = } \end{array}$ $\mathbf { H } _ { \mathrm { d } } ^ { ( l ) } \mathbf { W } ^ { ( l ) }$ denotes a linear layer, and $\mathrm { G C o n v } ( \mathbf { H } _ { \mathrm { d } } ^ { ( l ) } , \mathbf { A } ) =$ $\hat { \mathbf { D } } ^ { - \frac { 1 } { 2 } } \hat { \mathbf { A } } \hat { \mathbf { D } } ^ { - \frac { 1 } { 2 } } \mathbf { H } _ { \mathrm { d } } ^ { ( l ) } \mathbf { W } ^ { ( l ) }$ denotes a graph convolutional layer with $\hat { \textbf { A } } = \textbf { A } + \textbf { I } _ { N }$ and $\begin{array} { r } { \hat { D } _ { i i } ~ = ~ \sum _ { j = 1 } ^ { N } \hat { A } _ { i j } } \end{array}$ . We denote the parameter set of the to-be-tr ned encoder by $\theta \ =$ $\{ \mathbf { W } ^ { ( l ) } \} _ { l = 1 , \cdots , L }$ . The output of the last encoder layer is our learned representations, i.e., $\mathbf { H } = \mathbf { H } ^ { ( L ) }$ .

![](images/8ef776c7f853a1353f7549a6ce9e340691021e6cd0ba48188cd24c0f4d7ab8ec.jpg)  
Figure 3: The framework of our proposed SIGNA. The contrast is conducted within a single graph view, and thus SIGNA has only one branch. The encoder with dropout implicitly provides more embedding combinations for robust contrast. The role of neighbors is variable, while non-neighbors are fixed as negative samples. The normalized JSD estimator facilitates better contrastive effect. The goal of SIGNA is to realize soft neighborhood awareness.

Stochastic Neighbor Masking. According to previous homophily analyses, neighbors actually have a nonnegligible probability of owning different labels. In light of such uncertainty, we seek to mask a fraction of neighbors and consider the remaining neighbors as positive samples. To do that, we independently draw a masking indicator $m _ { v }$ for a neighboring node $\boldsymbol { v }$ of $u$ from a Bernoulli distribution with probability $\alpha \in [ 0 , 1 ]$ , i.e., $m _ { v } \sim \mathcal { B } ( \alpha ) , \forall v \ \in \ \mathcal { N } _ { u }$ . Specifically, the remaining neighbor set at each epoch is:

$$
\mathcal { N } _ { u } ^ { \prime } = \{ v \in \mathcal { N } _ { u } \mid m _ { v } = 0 \} .
$$

Just in case all neighbors are masked, we add the anchor node itself to its positive set for implementation convenience (no extra training signal is provided). As for negative set, it consists of all non-neighbors as well as those masked neighbors. Then, the positive and negative sets are:

$$
\mathcal { P } _ { u } = \mathcal { N } _ { u } ^ { \prime } \cup \{ u \} , ~ \mathcal { Q } _ { u } = \mathcal { V } \setminus \mathcal { P } _ { u } .
$$

Theorem 0.4 (Probabilistic Neighborhood Contrastive Learning). Let $\boldsymbol { S _ { u v } }$ be the target similarity between embeddings of the anchor node $u$ and any other node $\boldsymbol { v } \ne \boldsymbol { u }$ within the graph, and assume that $S _ { u v } = \delta$ if $v \in \mathcal { P } _ { u }$ otherwise $S _ { u v } = \lambda ( v \in \mathcal { Q } _ { u } ) ,$ , where $\delta , \lambda$ are determined by the objective function. Then, we have: (a) $\mathbb { E } _ { v \in \mathcal { N } _ { u } } ( S _ { u v } ) =$ $\begin{array} { r } { \delta ( 1 - \alpha ) + \lambda \alpha ; ( b ) \operatorname { \mathbb { E } } _ { v \not \in \mathcal { N } _ { u } } ( S _ { u v } ) = \lambda . } \end{array}$ .

Proof. (a) Since neighbors are randomly masked by a probability of $\alpha$ , we have $p ( v \in \mathcal { P } _ { u } | v \in \mathcal { N } _ { u } ) = 1 - \alpha$ and $p ( v \in$ $\mathcal { Q } _ { u } | v \in \mathcal N _ { u } ) = \alpha .$ . Thus, $\mathbb { E } _ { v \in \mathcal { N } _ { u } } ( S _ { u v } ) = \delta ( 1 - \alpha ) + \lambda \alpha$ . (b) Since $p ( v \in \mathcal { Q } _ { u } | v \not \in \mathcal { N } _ { u } ) = 1$ , $\mathbb { E } _ { v \notin \mathcal { N } _ { u } } ( S _ { u v } ) = \lambda$ holds.

Remark. With stochastic masking, neighbors are flipped back and forth as positive and negative samples along the training process, while non-neighbors consistently act as negative samples. As a result, the expectation of target similarity scores between neighbors lies between $\delta$ and $\lambda$ .

Normalized JSD Estimator. According to the empirical evidence provided by (Hjelm et al. 2018), InfoNCE (Noise Contrastive Estimation (Oord, Li, and Vinyals 2018)) and DV (Donsker-Varadhan representation of the KLdivergence (Donsker and Varadhan 1983)) require a large number of negative samples to be competitive, while JSD (Jensen-Shannon Divergence estimator (Nowozin, Cseke, and Tomioka 2016)) is less sensitive to the number of negative samples. Since we sample positive samples from neighbors, the number of negative samples is inherently reduced. Therefore, we are committed to optimizing an objective in the form of JSD estimator:

$$
\begin{array} { r l } & { \mathcal { I } _ { \mathrm { J S D } } ( u ) = \mathbb { E } _ { v ^ { + } \sim \mathcal { P } _ { u } } [ \log \mathcal { D } _ { \phi } ( u , v ^ { + } ) ] } \\ & { \qquad + \mathbb { E } _ { v ^ { - } \sim \mathcal { Q } _ { u } } [ \log ( 1 - \mathcal { D } _ { \phi } ( u , v ^ { - } ) ) ] , } \end{array}
$$

where $\mathcal { D } _ { \phi } : \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } \to \mathbb { R }$ is a discriminator function modeled by a neural network with parameters $\phi$ to measure the similarity between two instances and scale it into the range of [0,1]. Typically, the discriminator of JSD estimator is implemented using the inner product plus sigmoid function (Nowozin, Cseke, and Tomioka 2016; Hjelm et al. 2018; Peng et al. 2020b). Yet, it has been empirically verified that $\ell _ { 2 }$ normalization plays an important role in contrastive learning (Chen et al. 2020), which projects embeddings to the unit hypersphere before computing similarity (Wang and Isola 2020). Hence, we introduce a normalized discriminator for the JSD estimator:

$$
\mathcal { D } _ { \phi } ^ { \mathrm { n o r m } } ( u , v ) = \frac { 1 } { 2 } \left( \frac { \mathbf { z } _ { u } ^ { \top } \mathbf { z } _ { v } } { \| \mathbf { z } _ { u } \| _ { 2 } \| \mathbf { z } _ { v } \| _ { 2 } } + 1 \right) = \frac { \cos ( \mathbf { z } _ { u } , \mathbf { z } _ { v } ) + 1 } { 2 } ,
$$

where $\mathbf { z } = g _ { \phi } ( \mathbf { h } ) = \mathbf { W } _ { g } ^ { ( 2 ) } \sigma ( \mathbf { W } _ { g } ^ { ( 1 ) } \mathbf { h } )$ is a MLP projector parameterized by $\mathbf { \boldsymbol { \phi } } = \{ \mathbf { W } _ { g } ^ { ( 1 ) } , \mathbf { W } _ { g } ^ { ( 2 ) } \}$ . With $\ell _ { 2 }$ normalization, the similarity metric between embedding vectors is calculated by cosine similarity instead of inner product. And we adopt a simple linear scaler to restrict the range of similarity scores. The instantiated estimator with the normalized discriminator is called Norm-JSD. In Table 1, we compare Norm-JSd against two typical estimators, JSD and InfoNCE, which implies that Norm-JSD succeeds in combining the advantages of both JSD and InfoNCE while remaining concise.

Table 1: Comparing Norm-JSD against JSD and InfoNCE.   

<html><body><table><tr><td>Perspective</td><td>InfoNCE</td><td>JSD</td><td>Norm-JSD</td></tr><tr><td>Robust to #neg.</td><td>no</td><td>yes</td><td>yes</td></tr><tr><td>l2 normalization</td><td>yes</td><td>no</td><td>yes</td></tr><tr><td>Scaler</td><td>temperature</td><td>sigmoid</td><td>linear</td></tr></table></body></html>

The superiority of Norm-JSD over JSD and InfoNCE is also empirically demonstrated in our ablation studies.

Combining Equation (9), Equation (10), and Equation (11) together, we arrive at the loss function of SIGNA:

$$
\begin{array} { r l } & { \ell ( u ) = - \frac { 1 } { | \mathcal { P } _ { u } | } \sum _ { v ^ { + } \in \mathcal { P } _ { u } } \log \left( \mathcal { D } _ { \phi } ^ { \mathrm { n o r m } } \left( u , v ^ { + } \right) \right) } \\ & { \qquad - \frac { 1 } { | \mathcal { Q } _ { u } | } \sum _ { v ^ { - } \in \mathcal { Q } _ { u } } \log \left( 1 - \mathcal { D } _ { \phi } ^ { \mathrm { n o r m } } \left( u , v ^ { - } \right) \right) , } \end{array}
$$

$$
\mathcal { L } _ { \mathrm { S I G N A } } = \frac { 1 } { \lvert \mathcal { V } \rvert } \sum _ { u \in \mathcal { V } } \ell ( u ) .
$$

Note that the target similarity scores for positive and negative samples are $\mathcal { D } _ { \phi } ^ { \mathrm { n o r m } } \left( u , v ^ { + } \right) = 1$ and $\mathcal { D } _ { \phi } ^ { \mathrm { n o r m } } \left( u , v ^ { - } \right) = 0$ respectively. Namely, $\delta = 1$ and $\lambda = 0$ . Recalling Theo$\mathrm { r e m } 0 . 4$ , we have the following corollary:

Corollary 0.5. With Norm-JSD, the expectation of target similarity for neighbors equals $1 - \alpha$ and the expectation of target similarity for non-neighbors equals 0, i.e., $\mathbb { E } _ { v \in \mathcal { N } _ { u } } ( S _ { u v } ) = 1 - \alpha \in [ 0 , 1 ]$ and $\mathbb { E } _ { v \notin \mathcal { N } _ { u } } ( S _ { u v } ) = 0$ .

Remark. The above corollary indicates that, with soft neighborhood awareness, neighbors are expected to maintain a moderate level of local tolerance with anchors, which leaves some leeway for future tuning in downstream tasks. By contrast, a large number of non-neighbors will be separated as far as possible. In other words, SIGNA is allowed to learn a desirable globally-uniform yet locally-tolerant embedding space (Wang and Isola 2020; Wang and Liu 2021). Such an effect is observed in our experimental analysis.

# Experiments Experimental Setup

Datasets. We comprehensively evaluate SIGNA on three kinds of node-level tasks across 7 datasets with various scales and properties (Velickovic et al. 2019; Jiao et al. 2020; Thakoor et al. 2021; Lee, Lee, and Park 2022). Wiki CS, Amazon Photo, Amazon Computers, Coauthor CS, and Coauthor Physics are used for transductive node classification and node clustering tasks. Two larger-scale datasets, Flickr and PPI, are used for inductive node classification on a single graph and multiple graphs, respectively. Statistics of these datasets are presented in the Appendix.

Baselines. We primarily compare SIGNA against representative and state-of-the-art unsupervised methods for node representation learning, including GAE (Kipf and Welling 2016b), DGI (Velickovic et al. 2019), GMI (Peng et al. 2020b), MVGRL (Hassani and Khasahmadi 2020), GRACE (Zhu et al. 2020b), Subg-Con (Jiao et al. 2020),

$S ^ { 2 } \mathrm { G R L }$ (Peng et al. 2020a), CCA-SSG (Zhang et al. 2021), BGRL (Thakoor et al. 2021), GraphCL-NS (Hafidi et al. 2022), SUGRL (Mo et al. 2022), and AFGRL (Lee, Lee, and Park 2022).

Evaluation Protocols. We first train all models in a fully unsupervised manner, and then the trained encoder is frozen and used for testing in downstream tasks (see Appendix).

Implementation Details. Details about encoder implementation, hyperparameter selection, and computing infrastructure are provided in appendix due to the space limitation.

# Main Results and Analysis

Transductive Node Classification. The empirical performance of various methods across five datasets are summarized in Table 2. As demonstrated by the results, SIGNA consistently and substantially outperforms baseline methods across all five benchmark datasets. Specifically, SIGNA improves the accuracy by absolute margins of $1 . 5 5 \%$ (Wiki $C S )$ , $2 . 0 4 \%$ (Amazon Photo), $0 . 5 8 \%$ (Amazon Computers), $1 . 7 1 \%$ (Coauthor $C S$ ), and $0 . 6 6 \%$ (Coauthor Physics) over those competitive runner-ups. Besides, a notable finding is that the performances of those competitors are less stable across different benchmarks. For example, AFGRL (Lee, Lee, and Park 2022) outperforms BGRL (Thakoor et al. 2021) on four out of five datasets while being surpassed by BGRL (Thakoor et al. 2021) with a margin of $1 . 7 \hat { 4 } \%$ on Wiki CS. By contrast, our method achieves more robust leading performances, indicating the efficacy and universality of SIGNA. Furthermore, unlike existing methods that rely on graph convolutional layers to generate representations, simple linear layers are used as our BaseEncoders, which discard the complicated massage aggregation process on numerous edges. To show the benefit, we investigate the time cost of the inference phase with GCN and MLP being basic encoders (all other settings are kept consistent), which are reported in Table 3. As illustrated, our MLP-based encoder is about $1 0 9 \times$ to $3 3 1 \times$ faster than the GCN-based encoder with the same settings in terms of the inference time. In particular, the gap widens rapidly as the size of the edge set increases, since the inference of MLP-based encoders is independent of edge-oriented aggregation.

Inductive Node Classification. The empirical performances on a single graph $( F l i c k r )$ and multiple graphs $( P P I )$ are reported in Table 4 and Table 5, respectively. According to the results in terms of micro-averaged F1-score, SIGNA surpasses the previous best method by $3 . 1 3 \%$ (Flickr) and $2 1 . 7 4 \%$ $( P P I )$ , showcasing its remarkable superiority in inductive learning tasks. It is noteworthy that SIGNA even outperforms the best supervised baseline GraphSAGE (Hamilton, Ying, and Leskovec 2017) by $1 . 8 3 \%$ on Flickr. On the PPI dataset, while the performance of supervised baselines remains unsurpassed, our method significantly reduces the disparity between unsupervised approaches and those supervised benchmarks.

Node Clustering. The comparison of clustering performance against GRACE (Zhu et al. 2020b), GCA (Zhu et al. 2021), BGRL (Thakoor et al. 2021), and AFGRL (Lee, Lee, and Park 2022) is shown in Table 6. The closer the value of both NMI and Homogeneity is to 1, the better the clustering effect. As revealed by Table 6, SIGNA exhibits dominating performance over all competitors across five datasets. In particular, SIGNA consistently outperforms AFGRL (Lee, Lee, and Park 2022), a clustering-oriented method, demonstrating the superior capability of SIGNA to learn underlying cluster structure in the unsupervised context.

Table 2: Performance on transductive node classification tasks. The best and second-best performances among unsupervised methods are highlighted in bold and underlined, respectively.   

<html><body><table><tr><td>Method</td><td>Training Data</td><td>Wiki CS</td><td>Amazon Photo</td><td>Amazon Computers</td><td>Coauthor CS</td><td>Coauthor Physics</td></tr><tr><td>GCN</td><td>X,A, Y</td><td>77.19±0.12</td><td>92.42±0.22</td><td>86.51±0.54</td><td>93.03±0.31</td><td>95.65±0.16</td></tr><tr><td>GAT</td><td>X,A, Y</td><td>77.65±0.11</td><td>92.56±0.35</td><td>86.93±0.29</td><td>92.31±0.24</td><td>95.47±0.15</td></tr><tr><td>GAE</td><td>X,A</td><td>75.25±0.28</td><td>91.62±0.13</td><td>85.27±0.19</td><td>90.01±0.17</td><td>94.92±0.07</td></tr><tr><td>VGAE</td><td>X,A</td><td>75.63±0.19</td><td>92.20±0.11</td><td>86.37±0.21</td><td>92.11±0.09</td><td>94.52±0.00</td></tr><tr><td>DGI</td><td>X,A</td><td>75.35±0.14</td><td>91.61±0.22</td><td>83.95±0.47</td><td>92.15±0.63</td><td>94.51±0.52</td></tr><tr><td>GMI</td><td>X,A</td><td>74.85±0.08</td><td>90.68±0.17</td><td>82.21±0.31</td><td>00M</td><td>00M</td></tr><tr><td>MVGRL</td><td>X,A</td><td>77.52±0.08</td><td>92.08±0.01</td><td>87.52±0.21</td><td>92.18±0.15</td><td>95.33±0.03</td></tr><tr><td>GRACE</td><td>X,A</td><td>78.19±0.41</td><td>92.24±0.45</td><td>86.35±0.44</td><td>92.93±0.22</td><td>95.26±0.10</td></tr><tr><td>CCA-SSG</td><td>X,A</td><td>78.64±0.72</td><td>93.14±0.14</td><td>88.74±0.28</td><td>92.91±0.20</td><td>95.38±0.06</td></tr><tr><td>SUGRL</td><td>X,A</td><td>79.12±0.67</td><td>93.07±0.15</td><td>88.93±0.21</td><td>92.83±0.23</td><td>95.38±0.11</td></tr><tr><td>BGRL</td><td>X,A</td><td>79.36±0.53</td><td>92.87±0.27</td><td>89.68±0.31</td><td>93.21±0.18</td><td>95.56±0.12</td></tr><tr><td>AFGRL</td><td>X,A</td><td>77.62±0.74</td><td>93.22±0.28</td><td>89.88±0.33</td><td>93.27±0.17</td><td>95.69±0.08</td></tr><tr><td>SIGNA</td><td>X,A</td><td>80.91±0.46</td><td>95.32±0.19</td><td>90.46±0.25</td><td>94.98±0.20</td><td>96.35±0.09</td></tr></table></body></html>

Table 3: Inference time (millisecond) on transductive node classification tasks with different BaseEncoders.   

<html><body><table><tr><td>Encoder</td><td>Wiki CS</td><td>Photo</td><td>Comp.</td><td>CS</td><td>Phys.</td></tr><tr><td>GCN</td><td>37.67</td><td>30.62</td><td>60.75</td><td>62.85</td><td>145.70</td></tr><tr><td>MLP</td><td>0.25</td><td>0.28</td><td>0.31</td><td>0.35</td><td>0.44</td></tr><tr><td>Ratio</td><td>151×</td><td>109×</td><td>196×</td><td>180×</td><td>331×</td></tr></table></body></html>

Table 4: Performance on single-graph inductive node classification task (Flickr) in terms of micro-averaged F1-score.   

<html><body><table><tr><td>Method</td><td>Training Data</td><td>F1-score</td></tr><tr><td>FastGCN</td><td>X,A,Y</td><td>48.1±0.5</td></tr><tr><td>GCN GraphSAGE</td><td>X,A,Y X,A, Y</td><td>48.7±0.3 50.1±1.3</td></tr><tr><td>Unsup-GraphSAGE</td><td>X,A</td><td>36.5±1.0</td></tr><tr><td>DGI</td><td>X,A</td><td>42.9±0.1</td></tr><tr><td>GMI</td><td>X,A</td><td></td></tr><tr><td>Subg-Con</td><td>X,A</td><td>44.5±0.2</td></tr><tr><td>SIGNA</td><td>X,A</td><td>48.8±0.1 51.93±0.04</td></tr></table></body></html>

# Why Does SIGNA Work

To facilitate direct understanding of what SIGNA has done, we first analyze the relationship between neighborhood awareness and model performance, which is illustrate in Figure 4. Compared with representative neighborhood-aware (SUGRL) and non-neighborhood-aware (GRACE) GCL methods, SIGNA learns a moderate of structural information due to our soft neighborhood awarness strategies, thus yielding better generalization performances on downstream tasks.

Table 5: Performance on multi-graph inductive node classification task (PPI) in terms of micro-averaged F1-score.   

<html><body><table><tr><td>Method</td><td>Training Data</td><td>F1-score</td></tr><tr><td>GaAN-mean GAT FastGCN</td><td>X,A,Y X,A,Y X,A,Y</td><td>96.9±0.2 97.3±0.2 63.7±0.6</td></tr><tr><td>Unsup-GraphSAGE</td><td>X,A</td><td>46.5</td></tr><tr><td>Random-Init DGI</td><td>X,A</td><td>62.6±0.2</td></tr><tr><td>GMI</td><td>X,A X,A</td><td>63.8±0.2</td></tr><tr><td>S²GRL</td><td></td><td>64.6±0.0</td></tr><tr><td>GRACE</td><td>X,A</td><td>66.0±0.0</td></tr><tr><td></td><td>X,A</td><td>66.2±0.1</td></tr><tr><td>Subg-Con</td><td>X,A</td><td>66.9±0.2</td></tr><tr><td>GraphCL-NS</td><td>X,A</td><td>65.9±0.0</td></tr><tr><td>BGRL-GAT-Encoder</td><td>X,A</td><td></td></tr><tr><td>SIGNA</td><td>X,A</td><td>70.49±0.05 92.25±0.03</td></tr></table></body></html>

Besides, our histogram analysis reveals that pulling together neighbors is less rational than pushing away non-neighbors, and thus we should avoid overly pulling together neighbors, which exactly matches the spirit of SIGNA. Please refer to our appendix for details.

In Figure 5, we use t-SNE (Van der Maaten and Hinton 2008) to visualize representations learned by SIGNA, as well as those learned by GRACE (Zhu et al. 2020b) and SUGRL (Mo et al. 2022) (representatives of augmentationinvariant and neighbourhood-aware methods). For fairness comparison, their random seeds are kept consistent. As can be seen, GRACE learns a uniformly-distributed embedding space, which impairs intra-class aggregation (e.g., purple samples). On the contrary, SUGRL improves intra-class similarities yet fails to separate different classes (e.g., purple samples and their surroundings). By contrast, SIGNA strikes a better balance between intra-class aggregation and interclass separation (e.g., purple samples and their surroundings). Clearly, this ability contributes to its superiority in classification and clustering tasks.

Table 6: Performance on node clustering tasks in terms of NMI and Homogeneity.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="2">Wiki CS</td><td colspan="2">Am. Photo</td><td colspan="2">Am. Computers</td><td colspan="2">Co. CS</td><td colspan="2">Co. Physics</td></tr><tr><td>NMI</td><td>Homo.</td><td>NMI</td><td>Homo.</td><td>NMI</td><td>Homo.</td><td>NMI</td><td>Homo.</td><td>NMI</td><td>Homo.</td></tr><tr><td>Raw features</td><td>0.2633</td><td>0.2738</td><td>0.3273</td><td>0.3376</td><td>0.2389</td><td>0.2617</td><td>0.7103</td><td>0.7446</td><td>0.5207</td><td>0.5576</td></tr><tr><td>GRACE</td><td>0.4282</td><td>0.4423</td><td>0.6513</td><td>0.6657</td><td>0.4793</td><td>0.5222</td><td>0.7562</td><td>0.7909</td><td>0.5128</td><td>0.5546</td></tr><tr><td>GCA</td><td>0.3373</td><td>0.3525</td><td>0.6443</td><td>0.6575</td><td>0.5278</td><td>0.5816</td><td>0.7620</td><td>0.7965</td><td>0.5202</td><td>0.5654</td></tr><tr><td>BGRL</td><td>0.3969</td><td>0.4156</td><td>0.6841</td><td>0.7004</td><td>0.5364</td><td>0.5869</td><td>0.7732</td><td>0.8041</td><td>0.5568</td><td>0.6018</td></tr><tr><td>AFGRL</td><td>0.4132</td><td>0.4307</td><td>0.6563</td><td>0.6743</td><td>0.5520</td><td>0.6040</td><td>0.7859</td><td>0.8161</td><td>0.5782</td><td>0.6174</td></tr><tr><td>SIGNA</td><td>0.4593</td><td>0.4763</td><td>0.7635</td><td>0.7823</td><td>0.5608</td><td>0.6057</td><td>0.8047</td><td>0.8408</td><td>0.5907</td><td>0.6393</td></tr></table></body></html>

<html><body><table><tr><td>Variant</td><td>Wiki CS</td><td>Am. Photo</td><td>Am. Computers</td><td>Co. CS</td><td>Co. Physics</td></tr><tr><td>SIGNA</td><td>80.91±0.46</td><td>95.32±0.19</td><td>90.46±0.25</td><td>94.98±0.20</td><td>96.35±0.09</td></tr><tr><td>×Dropout</td><td>80.04±0.55</td><td>93.85±0.26</td><td>89.72±0.36</td><td>93.95±0.15</td><td>95.93±0.12</td></tr><tr><td>Dropout=→>NFM</td><td>75.21±0.45</td><td>91.16±0.53</td><td>87.43±0.36</td><td>93.92±0.20</td><td>95.95±0.09</td></tr><tr><td>×StochMask StochMask=AllMask</td><td>80.58±0.65 68.14±0.69</td><td>94.87±0.28 80.59±0.99</td><td>90.09±0.34 76.64±0.60</td><td>93.74±0.15 89.22±0.25</td><td>96.22±0.08 94.30±0.14</td></tr><tr><td>Norm-JSD=JSD</td><td>78.11±0.59</td><td>76.90±0.54</td><td></td><td></td><td></td></tr><tr><td>Norm-JSD=InfoNCE</td><td>79.03±0.71</td><td>93.64±0.39</td><td>70.02±0.77 88.10±0.35</td><td>91.84±0.18</td><td>92.42±0.13</td></tr><tr><td></td><td></td><td></td><td></td><td>94.88±0.15</td><td>95.96±0.11</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>× All</td><td>77.92±0.48</td><td>74.61±1.22</td><td>69.16±0.83</td><td>86.74±0.34</td><td>50.57±0.10</td></tr></table></body></html>

Table 7: Ablation study of SIGNA. $\times$ : remove components; $\Rightarrow$ : replace components with alternatives.

![](images/e2882e1cfd85c3c10b8bf40e8a235df8320fc9b4264119c5a86e22f26e8a8b87.jpg)  
Figure 4: The soft neighborhood awareness of SIGNA yields better performance.

![](images/d54b009c2784b7059db653f4722bb1b3a248a6ddc28f80eac6878e6dbd8c402c.jpg)  
Figure 5: Representation visualization via t-SNE.

# Ablation Study

In Table 7, we evaluate the contribution of different components of SIGNA. Overall, soft neighborhood awareness proves of great significance. Our three components are all reasonably designed, as removing any one of them results in a noticeable performance degradation, especially when all three are removed, the performance drops dramatically. Besides, our specific findings include: (i) when we replace dropout with node feature masking (NFM), a widely-used input augmentation technique (Zhu et al. 2020b, 2021; Jin et al. 2021), the performance decreases dramatically; (ii) our Norm-JSD surpasses two other estimators by large margins, while the performance of JSD lags far behind that of InfoNCE, indicating the importance of our normalized discriminator; (iii) masking all neighbors significantly reduces the performance due to the missing of structural information (recall that we use MLP as encoder for transductive tasks); Yet, overemphasizing neighborhood affinities without stochastic masking also yields an inferior performance. In short, our soft neighborhood awareness is more suitable for GCL.

# Conclusion

In this work, we resort to soft neighborhood awareness for single-view GCL. To begin with, we explore homophily statistics on real-world datasets, which reveal that overconfidence in neighborhoods would be risky. Motivated by this, we propose a simple yet effective GCL framework (SIGNA) to pursue a moderate level of neighborhood affinities. Specifically, SIGNA considers neighboring nodes as potential positive samples but is equipped with three nuanced designs to relax restrictions on the retention of structural information. Experimental results demonstrate that SIGNA consistently outperforms existing GCL methods in a wide variety of node-level tasks. This work is expected to serve as a pioneer in exploring reasonable neighborhood awareness for contrastive learning of node representation.

# Acknowledgments

The work of Ziyue Qiao was supported by National Natural Science Foundation of China under Grant No. 62406056. The work of Kai Wang was supported in part by National Natural Science Foundation of China under Grant No. U24A20270 and Grant No. 62373378.