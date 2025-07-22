# Towards Learnable Anchor for Deep Multi-View Clustering

Bocheng Wang1, Chusheng Zeng1, Mulin Chen1\*, Xuelong Li2

1School of Artificial Intelligence, OPtics and ElectroNics (iOPEN), Northwestern Polytechnical University, China 2Institute of Artificial Intelligence (TeleAI), China Telecom, China wangbocheng $@$ mail.nwpu.edu.cn, zcs $@$ mail.nwpu.edu.cn, chenmulin $@$ nwpu.edu.cn, xuelong li@ieee.org

# Abstract

Deep multi-view clustering incorporating graph learning has presented tremendous potential. Most methods encounter costly square time consumption w.r.t. data size. Theoretically, anchor-based graph learning can alleviate this limitation, but related deep models mainly rely on manual discretization approaches to select anchors, which indicates that 1) the anchors are fixed during model training and 2) they may deviate from the true cluster distribution. Consequently, the unreliable anchors may corrupt clustering results. In this paper, we propose the Deep Multi-view Anchor Clustering (DMAC) model that performs clustering in linear time. Concretely, the initial anchors are intervened by the positive-incentive noise sampled from Gaussian distribution, such that they can be optimized with a newly designed anchor learning loss, which promotes a clear relationship between samples and anchors. Afterwards, anchor graph convolution is devised to model the cluster structure formed by the anchors, and the mutual information maximization loss is built to provide cross-view clustering guidance. In this way, the learned anchors can better represent clusters. With the optimal anchors, the full sample graph is calculated to derive a discriminative embedding for clustering. Extensive experiments on several datasets demonstrate the superior performance and efficiency of DMAC compared to state-of-the-art competitors.

# Introduction

In real-world data acquisition, a sample is often recorded from different views or sources (Cui et al. 2024), constituting multi-view data. For instance, an image can be distilled by multiple feature descriptors (e.g., color, shape, and spatial relation), and a document can be interpreted from multiple perspectives (e.g., topic distribution, word sequence, and word frequency). To adapt to the actual application environment, many types of unsupervised Multi-View Clustering (MVC) have emerged. Among them, deep MVC incorporates the enormous advantages of neural networks on representation learning, and has presented outstanding performance in real-world scenarios.

In general, deep MVC sets up a separate auto-encoder for each view, and then the learned multiple view-specific embeddings are fused into a consensus to infer clusters. To mitigate the inter-view conflict and perceive multi-view consistency, many representation alignment strategies are leveraged, such as contrastive learning (Xu et al. 2022b; Hu et al. 2023), and label distribution alignment (Cheng et al. 2021; Xu et al. 2022a; Liu et al. 2024). Overall, most deep MVC models emphasize the discriminability of the output embedding to improve clustering.

Recently, some works integrate the graph learning theory into deep MVC, which considers the structure relationship between samples while learning the discriminative embedding (Yang et al. 2023; Yan et al. 2023; Wang et al. 2024). These novel models capture the topological structure by constructing a data similarity graph, and thus optimizing representation learning based on graph data mining techniques, such as graph convolution network and structure preservation scheme (Wang et al. 2023b; Xiao et al. 2023). These methods usually require computing the edge weights between any two samples to build a full sample graph, which leads to costly time complexity $O ( n ^ { 2 } )$ where $n$ is the amount of samples. One theoretical solution to this problem is anchor graph learning that can accelerate the training to linear time w.r.t. $n$ (Dong et al. 2023; Cui et al. 2023). The performance of anchor graph learning and final clustering heavily depends on the anchor quality. However, most deep models employ non-differentiable manual procedures (e.g., random selection and $k$ -means) to determine the final anchors, which means the anchors are not learnable. Ideally, each sample should be represented by a certain anchor, and the anchors should reflect the cluster distribution. If the selected anchors cannot represent the original samples and cluster centroids well, the clustering result may be adversely affected.

To remedy the problems, we establish the Deep Multiview Anchor Clustering (DMAC) model. As shown in Fig. 1, the pipeline of DMAC can be partitioned into three main items. Firstly, the positive-incentive perturbation is injected into the initial anchors to yield a learnable mechanism for the anchors. Then, anchor graph convolution is used to produce the cluster indicator of anchors for each view, and the cross-view agreement is obtained by the proposed mutual information maximization loss. Finally, the learned anchors are leveraged to reveal the structural relationship between samples to impel a discriminative embedding. The key contributions of this paper are listed as follows.

• Anchor graph learning is incorporated into the deep multi-view clustering framework. Compared to most competitors, the proposed model can capture structural information with learnable anchors in a high-efficiency linear time to ameliorate embedding learning.

![](images/5667e2ae27afc220116b2d63bd048b4bab06ffb27ad21b5746d99aedd4ebd94f.jpg)  
Figure 1: Pipeline of DMAC. Note that the encoders are omitted. For the $a$ -th view, $\mathbf { Z } ^ { ( a ) }$ is the data embedding, $\mathbf { A } ^ { ( a ) }$ is the anchor graph, $\mathrm { A G C N } _ { a }$ is the corresponding anchor graph convolution network, and $\mathbf { F } ^ { ( a ) }$ is the anchor clustering distribution that records the probability of an anchor belonging to each cluster. $\mathbf { Z }$ is the shared fusion embedding among views. U represents the learnable anchors injected with the perturbation. The overall framework is updated by minimizing Eq. (16). The final result is gained by performing $k$ -means on the convergent $\mathbf { Z }$ .

• A perturbation-driven mechanism is proposed to improve the anchor quality adaptively during model training. By infusing beneficial perturbations, the learned anchors are adjusted to foster a distinct similarity relationship with the samples, thus representing the original data more comprehensively.

• An anchor graph convolution network is devised to infer the anchor clustering distribution of each view. The mutual information among multiple distributions is maximized to explore the consistent anchor clustering distribution, so as to produce clustering-oriented anchors.

# Related Work

# Deep Multi-View Clustering

Benefiting from the powerful representation learning ability of neural networks, deep MVC has achieved dominant clustering performance in practical applications (Fang et al. 2023b). Generally, deep MVC leverages auto-encoders to learn a series of view-specific embeddings, and then executes representation fusion to infer the consensus cluster indications. Many models are proposed to promote the clustering-friendly deep embedding. DAMC (Li et al. 2019) introduces the adversarial training mechanism to improve the discriminability of the learned embedding. DEMVC (Xu et al. 2021) pursues a consistent cluster structure between views to exploit the cross-view complementary information. In (Trosten et al. 2021), the importance of representation alignment for MVC is analyzed theoretically, and a contrastive learning based deep MVC method is established. MFLVC (Xu et al. 2022b) performs the feature- and clusterlevel contrastive learning simultaneously to inhibit the adverse effects of view-private information. CVCL (Chen et al. 2023) advocates unifying the cluster assignment by multiview contrastive learning.

The abovementioned representatives have presented good clustering capacity in experiments. However, most models focus on the intrinsic features while neglecting the structure relationship among samples that is essential to detect the clustering distribution (Huang et al. 2019; Wang et al. 2019; Chen, Wang, and Li 2024). Recently, some scholars introduce the graph learning theory into deep MVC to mine the structural information explicitly. CMGEC (Wang et al. 2021) employs a graph fusion network to integrate multiple structural graphs into a consensus. DFP-GNN (Xiao et al. 2023) parallels graph neural networks with auto-encoders to learn the feature- and structure-level embedding simultaneously. DealMVC (Yang et al. 2023) constructs the data similarity graph to guide clustering-oriented contrastive learning. GCFAggMVC (Yan et al. 2023) utilizes the similarity relationship among embeddings to ameliorate representation fusion. SURER (Wang et al. 2024) concatenates multiple view-specific graphs into a heterogeneous graph to explore the complementary relationship among views via the heterogeneous graph neural network. Those deep MVC models that incorporate the graph structure information have presented enormous development potential. Nevertheless, existing methods usually require computing the full sample graph, which encounters a squared complexity $O ( n ^ { 2 } )$ where $n$ is the number of samples. In this paper, we plan to introduce the anchor theory to achieve deep MVC in linear time.

# Anchor-Based Multi-View Clustering

Anchors, also known as landmarks, are widely used in graph-based and sub-space MVC (Yang et al. 2024). An anchor is the representative of local data (Chen and Cai 2011). By learning an anchor graph that records the adjacency relation between samples and anchors, the similarity among samples can be estimated approximately to derive the clusters. SFMC (Li et al. 2020) fuses multiple anchor graphs into a consensus bipartite graph with the rank constraint. MSGL (Kang et al. 2021) uses the data self-expression property to realize the adaptive anchor selection and graph optimization. FDAGF (Zhang et al. 2023) allows multiple anchor combinations as inputs to improve the flexibility and generalization ability. $\bar { \mathrm { E } } ^ { 2 } \mathrm { O M V C }$ (Wang et al. 2023a) calculates the spectral embeddings of anchor graphs and fuses them into the final cluster representation. CAMVC (Zhang et al. 2024) utilizes the estimated labels to optimize cluster-wise anchors.

Since the number of anchors is much smaller than the sample size, anchor-based MVC has outstanding efficiency. Nevertheless, most relevant algorithms are limited by shallow graph learning that the anchor graph is calculated via the original features directly. There is little exploration (Dong et al. 2023; Cui et al. 2023) of anchor-based deep MVC, which neglects the optimization of anchor quality with model training. Inspired by the positive-incentive noise (Li 2022), we plan to generate positive noise perturbation to guide high-quality anchor learning, and design the anchor graph convolution module to capture cross-view anchor clustering consistency.

# Methodology

In this section, the proposed DMAC is elaborated. DMAC generates positive perturbation to ameliorate the anchors shared among views, and utilizes anchor graph convolution to extract the cluster distribution of anchors.

Notations: matrices and vectors are expressed as uppercase and lowercase letters, respectively. For a matrix $\mathbf { X }$ , both $\mathbf { X } _ { i }$ and $\mathbf { x } _ { i }$ mean the $i$ -th row. $| | \mathbf { x } _ { i } | | _ { 1 } , | | \mathbf { x } _ { i } | | _ { 2 }$ , and $| | \mathbf { X } | | _ { \mathrm { F } }$ denote $\ell _ { 1 } , \ell _ { 2 }$ , and Frobenius norm, respectively.

# Preliminary Work

Denote $\{ \mathbf { X } ^ { ( 1 ) } , \mathbf { X } ^ { ( 2 ) } , \cdot \cdot \cdot , \mathbf { X } ^ { ( v ) } \}$ as the multi-view data with $n$ samples, $v$ views, and $c$ clusters. DMAC follows the mainstream deep MVC framework consisting of view-specific embedding learning and embedding fusion. In this part, the framework is introduced briefly to pave the subsequent innovative modules.

Specifically, the unshared encoder is used to learn viewspecific deep embedding

$$
{ \bf Z } ^ { ( a ) } = \mathrm { E n c o d e r } _ { a } ( { \bf X } ^ { ( a ) } ) .
$$

Based on the resultant embeddings $\{ \mathbf { Z } ^ { ( 1 ) } , \ \mathbf { Z } ^ { ( 2 ) } , \ \cdots ,$ $\mathbf { Z } ^ { ( v ) } \}$ , the simple but effective average weighting (Wang et al. 2024) is employed to calculate the fusion embedding

$$
\begin{array} { r } { { \bf Z } = \frac { 1 } { v } \sum _ { i } ^ { v } { \bf Z } ^ { ( i ) } , } \end{array}
$$

which is fed into a clustering algorithm to gain the labels.

Very lately, graph learning theory is introduced into the above framework, aiming to extract the topological structure of sample space to enhance embedding learning. Most related models encounter an expensive time complexity $O ( n ^ { 2 } )$ . Differently, we integrate anchor graph learning into deep MVC to reduce the complexity to linear time.

# Perturbation-Driven Anchor Learning

Anchor graph learning requires estimating multiple anchors in advance. Existing models usually use the manual setting strategy to select anchors, which inhibits the learnability of anchors. Therefore, we construct a generator to learn the perturbation to adjust anchors, and design the anchor learning loss to obtain the positive perturbation that improves anchor quality.

Perturbation Generation Network. Denoting $\widehat { \mathbf { U } }$ as the $m$ initial shared anchors obtained by performing $k$ mbeans on $\mathbf { Z }$ , the generation network produces learnable perturbation $\varepsilon$ to inject $\widehat { \bf U }$ . In this way, the anchors can be optimized by updating $\varepsilon$ through backpropagation.

To begin with, the initial perturbation $\epsilon$ is sampled from the standard multivariate Gaussian distribution

$$
\epsilon \sim N ( 0 , { \bf I } ) ,
$$

where $\mathbf { I }$ is an identity matrix, and $\epsilon$ has the same dimensionality as $\widehat { \bf U }$ .

Then,bwe use a pseudo-siamese perceptron to simulate the mean $\mu$ and deviation $\sigma$ of the perturbation, which are formulated as

$$
\mu = \mathrm { M L P } _ { \mu } ( \widehat { \mathbf { U } } ) , \sigma = \mathrm { M L P } _ { \sigma } ( \widehat { \mathbf { U } } ) .
$$

Consequently, the perturb abtion $\varepsilon$ is updated bas

$$
\varepsilon = \mu + \sigma \odot \epsilon ,
$$

where $\odot$ refers to the Hadamard product. Eq. (5) meets the reparameterization trick (Kingma and Welling 2014) that optimizes $\varepsilon$ by backpropagation. Finally, the anchor matrix is updated as

$$
\mathbf { U } = { \widehat { \mathbf { U } } } + { \boldsymbol { \varepsilon } } .
$$

Since $\varepsilon$ is optimized graduablly during training, the learnability of the anchor matrix $\mathbf { U }$ is achieved.

Anchor Learning Loss. Given the learnable anchors, the relationship between samples and anchors is explored to improve the quality of anchors.

Intuitively, the ideal anchors can be considered as subcentroids, which are appropriately dispersed, and each sample is strongly associated with a corresponding anchor. Therefore, the similarity of the sample and anchors is crucial to evaluate the anchor quality. Denoting $q _ { i } ^ { ( a ) } \in \mathbb { R } ^ { 1 \times m }$ as the similarity vector between the sample $\mathbf { Z } _ { i } ^ { ( a ) }$ and $m$ anchors U, $q _ { i j } ^ { ( a ) }$ is computed as

$$
\begin{array} { r } { q _ { i j } ^ { \left( a \right) } = \frac { \left( 1 + | | \mathbf { Z } _ { i } ^ { \left( a \right) } - \mathbf { U } _ { j } | | _ { 2 } ^ { 2 } \right) ^ { - 1 } } { \underset { k } { \sum } \left( 1 + | | \mathbf { Z } _ { i } ^ { \left( a \right) } - \mathbf { U } _ { k } | | _ { 2 } ^ { 2 } \right) ^ { - 1 } } . } \end{array}
$$

Then, we introduce the positive-incentive noise theory to pave the anchor learning loss. In other words, the perturbation $\varepsilon$ is regarded as the potential positive noise, which satisfies the definition in (Li 2022).

Definition 1. Mathematically, the positive noise $\epsilon _ { \pi }$ satisfies

$$
E ( { \mathcal { T } } | \epsilon _ { \pi } ) < E ( { \mathcal { T } } ) ,
$$

where $\tau$ represents a specific downstream task, and $E ( \cdot )$ computes the information entropy.

According to Definition 1, the positive-incentive noise aims to reduce the uncertainty of a specific downstream task. Considering that the anchors are representatives of the original data, the uncertainty of anchor selection task is mainly reflected by the distribution of $q _ { i } ^ { ( a ) }$ . To be specific, for a sample $\mathbf { Z } _ { i } ^ { ( a ) }$ , if all values in $q _ { i } ^ { ( a ) }$ are very close, the relationship between the sample and anchors is ambiguous, which indicates a high uncertainty. That is to say, we need to learn a extremely unbalanced $\dot { q } _ { i } ^ { ( a ) }$ , wherein one element is much larger than the others.

Therefore, for the $a$ -th view, the task entropy $E ( \mathcal T \vert \epsilon _ { \pi } )$ in Eq. (8) can be quantified as

$$
\begin{array} { r } { \mathcal { L } _ { A L } ^ { ( a ) } = \frac { 1 } { n } \displaystyle \sum _ { i } ^ { n } E \left( q _ { i } ^ { ( a ) } \right) = - \frac { 1 } { n } \displaystyle \sum _ { i } ^ { n } \sum _ { j } ^ { m } q _ { i j } ^ { ( a ) } \log \left( q _ { i j } ^ { ( a ) } \right) . } \end{array}
$$

By minimizing Eq. (9), each sample tends to hold a highly correlated anchor, such that the anchors can represent the data distribution well. In the following part, we aim to make the anchors aligned with the cluster distribution.

# Anchor Clustering Consistency Maximization

To learn clustering-oriented anchors, we devise anchor graph convolution to infer the anchor clustering distribution of each view, and introduce mutual information to capture cross-view anchor clustering consistency, so as to provide training guidance for anchor learning.

Anchor Graph Learning. Since the proposed anchor graph convolution network requires graph data as input, we first present anchor graph learning.

Anchor graph records the structural dependence between samples and anchors. For the $a$ -th view, an ideal anchor graph $\mathbf { S } ^ { ( a ) }$ respects the following assumption.

Assumption 1. The edge si(ja is negatively correlated with the distance between the corresponding nodes $\mathbf { Z } _ { i } ^ { ( a ) }$ and $\mathbf { U } _ { j }$ .

Assumption 1 also reflects the fundamental clustering scenario, that points with small distances are more likely to be within the same cluster. Hence, for the $a$ -th view, the anchor graph learning problem can be expressed as

$$
\begin{array} { r l r } {  { \operatorname* { m i n } _ { { \bf S } ^ { ( a ) } } \sum _ { i } ^ { n } \sum _ { j } ^ { m } | | { \bf Z } _ { i } ^ { ( a ) } - { \bf U } _ { j } | | _ { 2 } ^ { 2 } s _ { i j } ^ { ( a ) } + \gamma | | { \bf S } ^ { ( a ) } | | _ { \mathrm { F } } ^ { 2 } , } } \\ & { } & { s . t . \forall i | | { \bf s } _ { i } ^ { ( a ) } | | _ { 1 } = 1 , 0 \le { \bf s } _ { i } ^ { ( a ) } \le 1 , } \end{array}
$$

where $\mathbf { s } _ { i } ^ { ( a ) }$ is the $i$ -th row of the anchor graph $\mathbf { S } ^ { ( a ) } \in \mathbb { R } ^ { n \times m }$ , and the second term evades that each sample only connects with the nearest anchor. The constraint that the sum of each row in $\mathbf { S } ^ { ( a ) }$ is 1 aims to approach Assumption 1.

According to (Nie, Zhu, and Li 2017), problem (10) can be solved with an efficient closed-form solution, which also evades the selection of parameter $\gamma$ .

Anchor Graph Convolution. Based on the view-specific anchor graph $\mathbf { S } ^ { ( a ) }$ and shared anchors U, we introduce graph convolution (Kipf and Welling 2016) to calculate the anchor clustering distribution, leading to Anchor Graph Convolution Network (AGCN).

For the $a$ -th view, the row in anchor clustering distribution $\mathbf { F } ^ { ( a ) } \in \mathbb { R } ^ { m \times c }$ records the probability of an anchor belonging to each cluster. Specifically, the forward propagation of $l$ -th hidden layer in $\mathrm { A G C N _ { a } }$ is

$$
\mathbf { F } ^ { \left( a \right) \left( l + 1 \right) } = \varphi \left( \mathbf { D } _ { \mathbf { S } ^ { \left( a \right) } } ^ { - 1 } \left( \mathbf { S } ^ { \left( a \right) } \right) ^ { \mathrm { T } } \mathbf { S } ^ { \left( a \right) } \mathbf { F } ^ { \left( a \right) \left( l \right) } \mathbf { W } ^ { \left( l \right) } \right) ,
$$

where $\mathbf { D _ { S ^ { ( a ) } } } \in \mathbb { R } ^ { m \times m }$ is the diagonal degree matrix of $\mathbf { S } ^ { ( a ) }$ that the $j$ -th item is $\begin{array} { r l } & { \sum _ { i } ^ { n } s _ { i j } ^ { ( a ) } , \left( \mathbf { D } _ { \mathbf { S } ^ { ( a ) } } ^ { - 1 } \left( \mathbf { S } ^ { ( a ) } \right) ^ { \mathrm { T } } \mathbf { S } ^ { ( a ) } \right) \in } \end{array}$ $\mathbb { R } ^ { m \times m }$ is the symmetric and doubly stochastic anchor similarity graph (Zhang et al. 2022) that conforms to the criterion of GCN, $\mathbf { \bar { W } } ^ { ( l ) }$ is the parameter matrix, and $\varphi ( \cdot )$ is a certain activation function. Note that $\mathbf { F } ^ { ( a ) ( 0 ) } = \mathbf { U }$ .

The neuron number of the last layer in AGCN is the cluster number $c$ , and the softmax function is used for activation. Multiple AGCNs do not share parameters to learn the anchor cluster structure for each view.

Consistency Maximization Loss. We adopt Mutual Information (MI) to measure the difference among $\{ \mathbf { F } ^ { ( 1 ) } , \mathbf { F } ^ { ( 2 ) }$ , $\langle \cdots , \mathbf { F } ^ { ( v ) } \}$ . Compared to the widespread KL divergence, MI satisfies symmetry to increase computational efficiency. The large MI, the more similar the two distributions is. Based on MI, the consistency maximization loss for the $a$ -th view is

$$
\begin{array} { r } { \mathcal { L } _ { C M } ^ { ( a ) } = - \frac { 1 } { m } \displaystyle \sum _ { b = a + 1 } ^ { v } \sum _ { i } ^ { m } M I \left( \mathbf { F } _ { i } ^ { ( a ) } , \mathbf { F } _ { i } ^ { ( b ) } \right) , } \end{array}
$$

where

$$
\begin{array} { r } { M I \left( \mathbf { F } _ { i } ^ { \left( a \right) } , \mathbf { F } _ { i } ^ { \left( b \right) } \right) = \displaystyle \sum _ { x \sim \mathbf { F } _ { i } ^ { \left( a \right) } } \sum _ { y \sim \mathbf { F } _ { i } ^ { \left( b \right) } } p \left( x , y \right) l o g \left( \frac { p \left( x , y \right) } { p \left( x \right) p \left( y \right) } \right) . } \end{array}
$$

By minimizing Eq. (12), multiple anchor clustering distributions are aligned to relieve the adverse effects of view conflict and view-private information. Unlike existing deep MVC, the module achieves cross-view consensus from the perspective of anchors rather than samples, which is conducive to propelling an explicit cluster structure of anchors.

# Structure Preservation via Anchor Graph

In this part, the structural graph of samples is calculated via the learned anchor graph. On this basis, the structure preservation loss is developed to promote a discriminative multiview embedding.

Based on the anchor graph $\mathbf { S } ^ { ( a ) }$ , the full sample graph of the $a$ -view can be measured with

$$
\mathbf { G } ^ { ( a ) } = \mathbf { S } ^ { ( a ) } \mathbf { D } _ { \mathbf { S } ^ { ( a ) } } ^ { - 1 } \left( \mathbf { S } ^ { ( a ) } \right) ^ { \mathrm { T } } .
$$

Obviously, the resultant $\mathbf { G } ^ { ( a ) } \in \mathbb { R } ^ { n \times n }$ is a symmetric and doubly stochastic graph (Zhang et al. 2023).

With the $a$ -th full sample graph $\mathbf { G } ^ { ( a ) }$ , the structure preservation loss is

$$
\mathcal { L } _ { S P } ^ { ( a ) } = \sum _ { i , j } ^ { n } | | \mathbf { Z } _ { i } - \mathbf { Z } _ { j } | | _ { 2 } ^ { 2 } g _ { i j } ^ { ( a ) } ,
$$

which can be replaced with Eq. (19) to accelerate the matrix multiplication. There is a concern that Eq. (15) may trivially cause all samples to be mapped to the same embedding, which is called representation collapse. In Theorem 1, we deduce that the anchor learning loss can be seen as a regularization term to penalize the trivial solution.

By minimizing Eq. (15), the learned fusion embedding $\mathbf { z }$ is actuated to reserve the internal data structure of each view, that is, the samples in the same class remain compact. The complementary structural information across views is mined to ameliorate a discriminative fusion embedding for clustering performance improvement.

# Joint Loss and Optimizer

Combining Eqs. (9), (12), and (15), the joint loss is

$$
\mathcal { L } = \sum _ { a } ^ { v } \left( \mathcal { L } _ { A L } ^ { ( a ) } + \alpha \mathcal { L } _ { C M } ^ { ( a ) } + \beta \mathcal { L } _ { S P } ^ { ( a ) } \right) ,
$$

where both $\alpha$ and $\beta$ are the trade-off parameters.

The classical RMSprop optimizer (Zhang and Sennrich 2019) is adopted to train DMAC. The final result is obtained by performing $k$ -means (Hartigan and Wong 1979) on the fusion representation $\mathbf { Z }$ .

# Discussion and Analysis Theoretical Advantage of Anchor Learning Loss

In this part, we discuss that the proposed anchor learning loss shown in Eq. (9) is beneficial for relieving representation collapse. The anchor learning loss can be regarded as a regularization term of the structure preservation loss shown in Eq. (15) to boost a discriminative fusion embedding $\mathbf { Z }$ .

Theorem 1. Minimizing Eq. (9) is equivalent to penalizing the trivial solution (i.e., representation collapse) to Eq. (15).

Proof. Without loss of generality, we develop the proof from the perspective of the $a$ -th view. The converse-negative proposition corresponding to the theorem is

$$
( \mathbf { Z }  \mathbf { Z } ^ { * } ) \Rightarrow ( \mathcal { L } _ { A L } ^ { ( a ) } \not  0 ) ,
$$

where $\mathbf { Z ^ { * } }$ is the trivial solution to Eq. (15), that is, all rows in $\mathbf { Z }$ tend to be the same.

Because $\mathbf { U } \sim \mathbf { Z }$ (i.e., $\mathbf { U }$ is sampled from $\mathbf { Z }$ ), all anchors also tend to be the same when $\mathbf { Z } = \mathbf { Z } ^ { * }$ . Hence, the probability distribution $q _ { i } ^ { ( a ) }$ between sample $\mathbf { Z } _ { i } ^ { ( a ) }$ and anchor matrix $\mathbf { U }$ is very smooth, and then the anchor learning loss reaches the upper bound. The above deduction can be formulized as

$$
\begin{array} { r l } & { ( \mathbf { Z } \to \mathbf { Z } ^ { * } ) \Rightarrow ( \mathbf { U } \to \mathbf { U } ^ { * } ) \Rightarrow \left( \forall i \forall j \ : q _ { i j } ^ { ( a ) } \to \frac { 1 } { m } \right) } \\ & { \qquad \Rightarrow \left( \mathcal { L } _ { A L } ^ { ( a ) } \to l o g ( m ) \not \to 0 \right) . } \end{array}
$$

The original proposition and converse-negative proposition possess the same truth and falsehood property. Proposition (17) is true, so the theorem is proven. The proof can be generalized to any view easily. □

# Linear Computation Complexity

DMAC is able to accomplish MVC in the linear time complexity $O ( n )$ . To avoid excessive symbol definition and improve readability, we only analyze the influence of sample size $n$ and anchor number $m$ .

Table 1: Descriptions of real-world datasets.   

<html><body><table><tr><td></td><td>Dataset Samples</td><td>Views</td><td>Classes</td><td>Dimensions</td></tr><tr><td>Yale</td><td>165</td><td>3</td><td>15</td><td>4096,3304,6750</td></tr><tr><td>PIE</td><td>680</td><td>3</td><td>68</td><td>484,256,279</td></tr><tr><td>BBC</td><td>685</td><td>4</td><td>5</td><td>4659,4633,4665,4684</td></tr><tr><td>NUS</td><td>2400</td><td>6</td><td>10</td><td>64,144,73,128,225,500</td></tr><tr><td>CCV</td><td>6773</td><td>3</td><td>20</td><td>4000,5000,5000</td></tr><tr><td>ALOI</td><td>10800</td><td>4</td><td>100</td><td>77,13,64,125</td></tr></table></body></html>

In each forward propagation, the computation complexity of embedding learning and fusion is $O ( n )$ . Then, the initial anchor selection needs $O ( n m )$ via $k$ -means, and the generator requires $O ( m )$ to output the perturbation matrix. Finally, considering that D S−(1a) is a diagonal matrix, the consumption of anchor graph convolution is $O ( n m )$ .

In each back propagation, the anchor learning loss needs $O ( n m )$ to calculate the entropies of all rows in $\mathbf { Q }$ . The structure preservation loss can be written as the trace form

$$
\mathrm { T r } \left( \mathbf { Z } ^ { \mathrm { T } } \mathbf { D } _ { \mathbf { G } } ^ { ( a ) } \mathbf { Z } - \mathbf { Z } ^ { \mathrm { T } } \mathbf { S } ^ { ( a ) } \mathbf { D } _ { \mathbf { S } ^ { ( a ) } } ^ { - 1 } \left( \mathbf { S } ^ { ( a ) } \right) ^ { \mathrm { T } } \mathbf { Z } \right) ,
$$

where $\mathbf { D } _ { \mathbf { G } } ^ { ( a ) } \in \mathbb { R } ^ { n \times n }$ is the degree matrix of $\mathbf { G } ^ { ( a ) }$ shown in Eq. (14). Since $\mathbf { G } ^ { ( a ) }$ is a doubly stochastic matrix, $\mathbf { D } _ { \mathbf { G } } ^ { ( a ) } \in$ $\mathbb { R } ^ { n \times n }$ is an identity matrix. The calculation consumption of structure preservation loss is also $O ( n m )$ with Eq. (19). Finally, the consistency maximization loss needs $O ( m )$ .

In conclusion, the time complexity of DMAC is ${ \dot { O } } ( n m )$ . Normally, the quantity of anchors is much smaller than the sample size (i.e., $m \ll n$ ), so the average complexity of each iteration can be seen as $O ( n )$ .

# Experiments

In this section, the proposed DMAC is compared with advanced competitors. The ablation analysis is also conducted.

# Real-World Datasets

Six public real-world datasets that are widely used in clustering study are collected as benchmarks, including imagetype Yale (Belhumeur, Hespanha, and Kriegman 1997), PIE (Gross et al. 2010) and ALOI (Houle et al. 2010), text-type BBC (Greene and Cunningham 2006) and NUS (Bryant and $\mathrm { N g } 2 0 1 5$ ), and video-type CCV (Jiang et al. 2011). Each sample is preprocessed with the $\ell _ { 2 }$ norm normalization. Table 1 displays the basic information of each dataset.

# Evaluation Metrics

Two widespread metrics are adopted to quantify the clustering result, including Accuracy (ACC) and Normalized Mutual Information (NMI). Both ACC and NMI are positively correlated with the clustering performance. The mathematical expression can be found in (Wang et al. 2020).

# Comparison with Competitors

Competitors. Nine state-of-the-art methods are selected as competitors, including four shallow algorithms GMC (Wang, Yang, and Liu 2019), MSGL (Kang et al. 2021), LMVSC (Kang et al. 2020) and UDBGL (Fang et al. 2023a), and five deep models CMGEC (Wang et al. 2021), DealMVC (Yang et al. 2023), GCFAggMVC (Yan et al. 2023), DFP-GNN (Xiao et al. 2023) and SURER (Wang et al. 2024). Among them, MSGL, LMVSC and UDBGL are anchor-based MVC, and all deep models incorporate the graph structure information.

<html><body><table><tr><td rowspan="2">Method</td><td colspan="2">Yale</td><td colspan="2">PIE</td><td colspan="2">BBC</td><td colspan="2">NUS</td><td colspan="2">CCV</td><td colspan="2">ALOI</td><td colspan="2">Avg</td></tr><tr><td>ACC</td><td>NMI</td><td>ACC</td><td>NMI</td><td>ACC</td><td>NMI</td><td>ACC</td><td>NMI</td><td>ACC</td><td>NMI</td><td>ACC</td><td>NMI</td><td>ACC</td><td>NMI</td></tr><tr><td>GMC</td><td>69.70</td><td>70.06</td><td>21.18</td><td>44.24</td><td>69.05</td><td>47.87</td><td>18.24</td><td>9.96</td><td>10.66</td><td>0.43</td><td>57.05</td><td>73.50</td><td>40.98</td><td>41.01</td></tr><tr><td>MSGL</td><td>40.61</td><td>47.32</td><td>15.74</td><td>46.08</td><td>46.28</td><td>23.15</td><td>15.25</td><td>5.27</td><td>12.42</td><td>7.11</td><td>15.81</td><td>39.66</td><td>24.35</td><td>28.10</td></tr><tr><td>LMVSC</td><td>57.58</td><td>58.10</td><td>36.32</td><td>63.33</td><td>66.42</td><td>53.92</td><td>20.67</td><td>8.58</td><td>18.29</td><td>14.09</td><td>58.51</td><td>76.37</td><td>42.97</td><td>45.73</td></tr><tr><td>UDBGL</td><td>53.33</td><td>58.76</td><td>24.26</td><td>52.72</td><td>72.85</td><td>50.94</td><td>24.08</td><td>13.11</td><td>25.57</td><td>20.83</td><td>52.44</td><td>61.02</td><td>42.09</td><td>42.90</td></tr><tr><td>CMGEC</td><td>36.36</td><td>42.60</td><td>14.77</td><td>45.33</td><td>87.37</td><td>71.44</td><td>24.87</td><td>10.83</td><td>22.21</td><td>23.67</td><td>56.42</td><td>72.89</td><td>40.33</td><td>44.46</td></tr><tr><td>DealMVC</td><td>75.18</td><td>76.81</td><td>23.82</td><td>52.14</td><td>64.75</td><td>41.20</td><td>20.04</td><td>9.49</td><td>13.95</td><td>6.87</td><td>17.50</td><td>44.50</td><td>35.87</td><td>38.50</td></tr><tr><td>GCFAggMVC</td><td>66.06</td><td>66.51</td><td>27.94</td><td>59.15</td><td>63.65</td><td>48.87</td><td>23.42</td><td>10.69</td><td>35.43</td><td>32.92</td><td>54.52</td><td>72.21</td><td>45.17</td><td>48.39</td></tr><tr><td>DFP-GNN</td><td>56.36</td><td>63.39</td><td>24.26</td><td>56.88</td><td>75.09</td><td>58.73</td><td>29.42</td><td>16.12</td><td>21.33</td><td>19.36</td><td>49.15</td><td>66.12</td><td>42.60</td><td>46.77</td></tr><tr><td>SURER</td><td>61.82</td><td>67.68</td><td>30.29</td><td>64.16</td><td>79.85</td><td>64.24</td><td>27.33</td><td>16.17</td><td>24.91</td><td>26.86</td><td>43.94</td><td>64.03</td><td>44.69</td><td>50.52</td></tr><tr><td>DMAC</td><td>78.18</td><td>78.06</td><td>43.24</td><td>68.16</td><td>88.61</td><td>74.49</td><td>29.29</td><td>16.20</td><td>36.18</td><td>33.17</td><td>60.35</td><td>74.29</td><td>55.98</td><td>57.40</td></tr></table></body></html>

Table 2: Clustering performance of ten methods on six datasets. Bold and underlined values mean the optimal and sub-optimal results respectively. The column termed avg displays the average ACC and NMI of each method.

![](images/24cc1511206cebae13ac3f37e42159f234b9d77cf2edcbdac6e8474c2c001e6d.jpg)  
Figure 2: Runtime (s) of deep models on four datasets. Note that all records are converted by logarithmic base 2.

Setups. The grid search is used to explore the optimal parameter setup for each algorithm. The parameter grid of competitors are set as the recommendations in the original article. For example, the parameter $\alpha$ of MSGL is selected from $\{ 0 . 0 0 1 , 0 . 0 1 , 0 . 1 , 1 , 1 0 , 5 0 \}$ . For the proposed DMAC, the number of anchors $m$ is set automatically according to (Nie, Wang, and Li 2019), i.e., the lower bound of $\scriptstyle { \sqrt { n { \dot { \times } } c } }$ . The grid for both $\alpha$ and $\beta$ is $\{ 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } , 1 , 1 \dot { 0 } ^ { 1 } , 1 0 ^ { 2 }$ , $\mathrm { 1 0 ^ { 3 } } \mathrm  \}$ . The maximal iterations are 100.

The traditional methods are executed on Matlab $2 0 1 9 \mathrm { a }$ with an Intel i9-12900HX CPU. All deep models are implemented via PyTorch, and trained with a NVIDIA RTX-3090 GPU. Each algorithm is repeated 10 times for objectivity.

Performance Comparison. Table 2 records the clustering performance of all algorithms. For ease of comparison, we also calculate the average ACC and NMI of each algorithm on all datasets. In general, DMAC presents the best clustering ability. The success of DMAC proves the feasibility of applying anchor graph learning to deep MVC. DMAC learns high-quality anchors with the proposed perturbation-driven anchor learning scheme, and then mines the multi-view anchor clustering consistency via anchor graph convolution and mutual information maximization to further accelerate clustering-oriented anchors, so as to accurately reveal the structural graph for clustering improvement. According to the experimental results, we also summarize the following viewpoints. Firstly, compared with the traditional shallow methods, the deep models achieve better clustering scores, which indicates the enormous potential of neural networks on improving MVC. Secondly, the performance of GCFAg$\mathrm { g M V C }$ and SURER are more prominent than other deep methods. SURER and GCFAggMVC leverage data similarity graphs to guide heterogeneous graph embedding learning and feature aggregation respectively, which reflects the positive effects of structural information on the two key steps of deep MVC, namely, view-specific representation learning and multi-view representation fusion.

![](images/a5b30c40210a4df5c47a5db9eeb4230434b3264886aa6437c7f6a9e2a3d0dc0d.jpg)  
Figure 3: Anchor similarity matrix $\mathbf { U U } ^ { \mathrm { T } }$ on BBC.

Efficiency Comparison. Fig. 2 displays the clustering efficiency of deep models. It is observed that DMAC has the shortest runtime. Compared with advanced deep MVC models that incorporate graph structure learning, DMAC avoids inefficient full sample graph learning and graph convolution. The new anchor learning mechanism and anchor graph convolution network have linear time complexity theoretically, so as to speed up the training process.

Table 3: Ablation results of main modules in DMAC. Bold values emphasize the optimal results.   

<html><body><table><tr><td>Dataset</td><td>Metric</td><td>wo/PD</td><td>wo/CM</td><td>DMAC</td></tr><tr><td>Yale</td><td>ACC NMI</td><td>66.02 66.90</td><td>72.73 75.15</td><td>78.18 78.06</td></tr><tr><td>PIE</td><td>ACC NMI</td><td>34.12 62.94</td><td>33.82 62.97</td><td>43.24 68.16</td></tr><tr><td>BBC</td><td>ACC NMI</td><td>87.15 70.68</td><td>88.47 73.99</td><td>88.61 74.49</td></tr><tr><td>NUS</td><td>ACC NMI</td><td>24.53 14.29</td><td>27.67 15.35</td><td>29.29 16.20</td></tr><tr><td>CCV</td><td>ACC NMI</td><td>32.28 29.96</td><td>34.26 31.72</td><td>36.18 33.17</td></tr><tr><td>ALOI</td><td>ACC NMI</td><td>44.45 65.86</td><td>54.79 70.10</td><td>60.35 74.29</td></tr></table></body></html>

![](images/b4719fd7d9ce07be411f2da92efdf8f994b0aa2d693c5d270fbfaea984fd20ce.jpg)  
Figure 4: Visualization of fusion embedding $\mathbf { Z }$ on BBC. Each point is drawn as its actual label value.

# Ablation Study and Visualization

In the ablation experiment, we design two variants based on the complete DMAC. Concretely, wo/PD removes the perturbation generation network and anchor learning loss, and wo/CM suspends the consistency maximization loss. Table 3 shows the ablation comparison. DMAC still maintains the best clustering performance, which proves the positive role of the proposed modules.

To intuitively display the effects of new modules, we visualize the similarity matrix of anchors (i.e., ${ \mathbf { U U } } ^ { \mathrm { T } } ,$ ) in Fig. 3. It is exhibited that the anchor similarity matrix learned by DMAC has the most distinct diagonal, which means the anchors are relatively dispersive to adequately represent the sample clusters. The significant degradation of wo/PD compared to DMAC further proves the advantages of learnable anchors. The disparity between wo/CM and DMAC indicates that mining cross-view anchor clustering information is beneficial to guide clustering-oriented anchor learning for performance improvement.

In addition, we utilize UMAP (McInnes, Healy, and Melville 2018) to visualize the learned fusion embedding Z on BBC. As shown in Fig. 4, the result derived by DMAC has a more obvious inter-class partition, which means a small inter-class similarity. The ablation comparison again indicates that, the new modules are conducive to learn highquality and clustering-friendly anchors for accurate structure learning, so as to improve the discriminative ability of fusion embedding via the structure preservation loss. The advantage of DMAC over wo/PD reflects the practicability of Theorem 1, that is, the proposed anchor learning loss can suppress representation collapse.

![](images/9f96a64910d2925f2ae584e207b2156645f159e8d88bbdf1e0b66192fc672dcc.jpg)  
Figure 5: ACC of DMAC with different parameters $\alpha$ and $\beta$ .

# Parameter Sensitivity

Finally, the sensitivity of DMAC to the trade-off parameters $\alpha$ and $\beta$ is explored. Fig. 5 exhibits ACC of DMAC under the predefined parameter grid. It can be seen that the influence of $\beta$ is more significant than that of $\alpha$ , because $\beta$ is directly related to the fusion representation $\mathbf { Z }$ that derives the final result. The preliminary observation suggests that the combination of large $\alpha$ and $\beta$ is more likely to facilitate a good clustering result. Overall, the performance fluctuation is relatively smooth within an appropriate range.

# Conclusion

In this paper, we propose an anchor-based deep multiview clustering model termed DMAC. Different from traditional manual anchor selection ways, DMAC introduces a perturbation-driven anchor learning mechanism to make the anchors learnable. Specifically, inspired by the positiveincentive noise theory, a noise generation network is established to produce the perturbation adaptively, which is injected into anchors under the guidance of anchor learning loss. Besides, the anchor graph convolution module is designed to extract the cluster structure of anchors within each view, and then the multi-view anchor clustering consistency can be perceived with mutual information maximization. In this manner, DMAC is able to optimize the anchors during the training procedure, and pursue a desired anchor distribution for clustering. Theoretical analysis shows that DMAC has a linear time complexity $O ( n )$ . Experiments report the superior performance and efficiency of DMAC.

# Acknowledgments

This work was supported by the National Key Research and Development Program of China (Grant No: 2022ZD0160803), and the National Natural Science Foundation of China (Grant No: 61871470).