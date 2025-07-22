# DHAKR: Learning Deep Hierarchical Attention-Based Kernelized Representations for Graph Classification

Feifei Qian1, Lu Bai1\*, Lixin ${ \bf C u i } ^ { 2 }$ , Ming $\mathbf { L i } ^ { 3 , 4 * }$ , Ziyu Lyu5, Hangyuan $\mathbf { D } \mathbf { u } ^ { 6 }$ , Edwin Hancock7

1 School of Artificial Intelligence, and Engineering Research Center of Intelligent Technology and Educational Application, Ministry of Education, Beijing Normal University, Beijing, China;   
2 School of Information, Central University of Finance and Economics, Beijing, China;   
3 Zhejiang Key Laboratory of Intelligent Education Technology and Application, Zhejiang Normal University, Jinhua, China;   
4 Zhejiang Institute of Optoelectronics, Jinhua, China;   
5 School of Cyber Science and Technology, Sun Yat-Sen University, Shenzhen, China;   
6 School of Computer and Information Technology, Shanxi University, Taiyuan, China;   
7 Department of Computer Science, University of York, York, United Kingdom. feifei qian $@$ mail.bnu.edu.cn, bailu $@$ bnu.edu.cn, cuilixin $@$ cufe.edu.cn, mingli $@$ zjnu.edu.cn, lvzy7 $@$ mail.sysu.edu.cn, duhangyuan $@$ sxu.edu.cn, edwin.hancock $@$ york.ac.uk

# Abstract

Graph-based representations are powerful tools for analyzing structured data. In this paper, we propose a novel model to learn Deep Hierarchical Attention-Based Kernelized Representations (DHAKR) for graph classification. To this end, we commence by learning an assignment matrix to hierarchically map the substructure invariants into a set of composite invariants, resulting in hierarchical kernelized representations for graphs. Moreover, we introduce the feature-channel attention mechanism to capture the interdependencies between different substructure invariants that will be converged into the composite invariants, addressing the shortcoming of discarding the importance of different substructures arising in most existing R-convolution graph kernels. We show that the proposed DHAKR model can adaptively compute the kernelbased similarity between graphs, identifying the common structural patterns over all graphs. Experiments demonstrate the effectiveness of the proposed DHAKR model.

# Introduction

Graph-based representations have been widely employed in various research domains, such as social networks (Maleki, Padmanabhan, and Dutta 2022), molecular chemistry (An et al. 2024; Kosmala et al. 2023; Kelvinius et al. 2023), recommendation systems (Wu et al. 2019; Wei et al. 2022), etc. One way to effectively capture the structural characteristics for graph data analysis is to employ graph kernels.

Graph kernels aim to measure the similarity between graphs by mapping their structural information into a highdimensional Hilbert space. Under this scenario, most existing graph kernels are based on counting the pairs of isomorphic substructures decomposed from original graphs. This is the so-called R-convolution framework proposed by (Haussler et al. 1999). Specifically, given two sample graphs $G _ { m }$

$$
\begin{array} { r l } & { ( \underbrace { \displaystyle { \sum _ { G = \bigotimes } ^ { \{ \{  { \mathbb { B } } \} \bigotimes } } } ( \frac { \langle \boldsymbol { \mathbb { D } } , \boldsymbol { \mathbb { Q } } \otimes \otimes \mathbb { G } \rangle } { \langle \boldsymbol { \mathbb { Q } } , \boldsymbol { \mathbb { Q } } \otimes \mathbb { G } \rangle } ) } _ { G _ { 1 } ^ { + } } ) ( \underbrace { \langle \boldsymbol { \mathbb { D } } , \boldsymbol { \mathbb { Q } } \otimes \mathbb { G } \rangle } _ { \mathrm { m a ~ \mathrm { e } } } )  \\ & { ( \underbrace { \langle \boldsymbol { \mathbb { Q } } , \boldsymbol { \mathbb { Q } } ^ { \flat } \rangle } _ { \boldsymbol { G } _ { 2 } ^ { + } } ) ( \underbrace { \langle \boldsymbol { \mathbb { D } } , \boldsymbol { \mathbb { Q } } \mathbb { Q } \rangle } _ { \mathrm { m a ~ \mathrm { e } } \mathrm { p e n a t e n t s t r u c t u r e s } } ) } \\ & { \underbrace { \langle \boldsymbol { \mathbb { Q } } ^ { \flat } \mathbin { \sum _ { G = \bigcirc } ^ { \{ \flat } } } \boldsymbol { \mathbb { Q } } \rangle } _ { \boldsymbol { G } _ { 2 } ^ { \flat } } ( \underbrace { | \operatorname { c o n v e r g e } _ { \flat \bigotimes } \boldsymbol { \mathbb { Q } } | } _ { \boldsymbol { \mathcal { Q } } , \boldsymbol { \mathbb { Q } } \otimes \boldsymbol { \mathbb { Q } } } )  \\ & { \underbrace { \langle \boldsymbol { \mathbb { Q } } , \boldsymbol { \mathbb { Q } } ^ { \flat } \rangle } _ { \boldsymbol { G } _ { 2 } ^ { \flat } } ) \underbrace { ( \langle \boldsymbol { \mathbb { D } } , \boldsymbol { \mathbb { Q } } \otimes \boldsymbol { \mathbb { Q } } \rangle } _ { \mathrm { c o n p o s i t e s u b s t r u c t u r e s } } } \end{array}
$$

Figure 1: Illustrative example of the motivation.

and $G _ { n }$ , a R-convolution graph kernel $K _ { \mathrm { R } }$ is defined as

$$
K _ { \mathrm { R } } ( G _ { m } , G _ { n } ) = \sum _ { g _ { m } \subseteq G _ { m } } \sum _ { g _ { n } \subseteq G _ { n } } s ( g _ { m } , g _ { n } ) ,
$$

where $s ( g _ { m } , g _ { n } )$ denotes the similarity between the substructures $g _ { m }$ and $g _ { n }$ . Based on different types of substructures, most $\mathtt { R }$ -convolution kernels can be divided as kernels based on walks, paths, and subgraphs or subtrees. For instance, (Ga¨rtner, Flach, and Wrobel 2003) have proposed a Random Walk Graph Kernel (RWGK) that compares the sequences of visited nodes during random walks. (Shervashidze et al. 2009) have proposed a Graphlet Count Graph Kernel (GCGK) by counting the frequency of small connected subgraphs (i.e., graphlets). (Borgwardt and Kriegel 2005) have developed the Shortest Path Graph Kernel (SPGK) (Borgwardt and Kriegel 2005) by counting the pairs of common shortest paths. (Shervashidze et al. 2011) have defined the Weisfeiler-Lehman Subtree Kernel (WLSK) (Shervashidze et al. 2011) by iteratively counting the number of pairwise isomorphic subtrees extracted from the Weisfeiler-Lehman (WL) algorithm. Other alternative Rconvolution kernels also include the Optimal Assignment Kernel (OAK) (Kriege, Giscard, and Wilson 2016), the Subgraph Alignment Kernel (SAK) (Kriege and Mutzel 2012), the Wasserstein WLSK (Togninalli et al. 2019), etc.

However, there are some common drawbacks arising in existing R-convolution kernels. First, these kernels only focus on counting the number of pairwise isomorphic substructures, neglecting the global structure information. As shown in Fig.1, there are two non-isomorphic graphs, $G _ { 1 }$ and $G _ { 2 }$ . We assume $\{ 1 , 2 3 4 \}$ as substructure $s .$ , $\{ 2 , 1 4 \}$ as substructure $s ^ { \prime }$ . $G _ { 1 }$ and $G _ { 2 }$ have the isomorphic substructure $s$ , contributing one unit to the WLSK kernel value, even though the global structures are not isomorphic. Second, these kernels cannot identify the significance of different substructures, neglecting the influence of redundant features. Third, the kernel construction is separated from the classifier (e.g., C-SVMs). Although graph deep learning methods can provide an end-to-end learning framework, they usually sum up the node features to derive graph representations, resulting in topological information loss.

The aim of this paper is to address the above drawbacks, by proposing a novel Deep Hierarchical Attention-based Kernelized Representation (DHAKR) model. One key innovation is the hierarchical framework that can converge the homogeneous substructures into new composite structures. For instance, as shown in Fig.1, the two independent substructures will be combined as a composite substructure $\{ \{ 1 , 2 3 4 \} , \{ 2 , 1 4 \} \}$ . Since $G _ { 2 }$ does not contain the composite substructure, then the kernel value is 0. This indicates that the composite features are effective for identifying non-isomorphic global structures. As a result, the proposed model can simultaneously capture the global information at higher levels and the local information at lower levels. Specifically, the main contributions of this paper are threefold.

First, we define an assignment matrix that can converge homogeneous substructures into a set of composite substructures. Moreover, to identify the importance of different substructures, we employ the attention mechanism to assign different weights to the homogeneous substructures. Since the substructures are converged associated with different weights, the resulting composite substructure can significantly discriminate the importance of the original substructures. By hierarchically performing the above computational procedure, we can compute a family of hierarchical attention-based substructure invariants for original graphs. Second, with the above hierarchical substructure invariants to hand, we define a novel kernel-based learning framework to compute the DHARK for graphs, by computing the Rconvolution kernel between pairwise graphs associated with the hierarchical invariants. We show that the resulting hierarchical kernel matrixes can be seen as the kernel-based similarity embedding vectors of all sample graphs, and are differentiable. As a result, the proposed DHAKR model is end-toend trainable. Third, we demonstrate that the DHAKR can outperform state-of-the-art graph classification methods.

# Literature Reviews of Related Works The R-convolution Graph Kernels

We briefly review two classical R-convolution kernels that are closely related to this work, including the WLSK and SPGK kernels. The WLSK kernel employs the iterative label refinement and measures the similarity between subtree patterns. Assuming $l _ { 0 } ( u )$ denotes the initial label of node $u$ , for each iteration $t$ , the WLSK concatenates the node’s current label with the sorted multiset of its neighbors’ labels $\mathcal { L } _ { \mathcal { N } } ^ { t - 1 } ( u ) )$ . Then the WLSK maps each unique concatenated label into a new label through a hash function, i.e.,

$$
l _ { t } ( u ) = \mathrm { H a s h } ( l _ { t - 1 } ( u ) , \mathcal { L } _ { \mathcal { N } } ^ { t - 1 } ( u ) ) ,
$$

where $l _ { t } ( u )$ refers a subtree rooted at $u$ of height $t$ . Given a pair of graphs $G _ { m }$ and $G _ { n }$ , the WLSK kernel is computed by counting the number of common subtree patterns, i.e.,

$$
K _ { \mathrm { W L } } = \sum _ { t = 0 } ^ { T _ { m a x } } \sum _ { i = 0 } ^ { | \mathcal { L } ^ { t } | } c ( G _ { m } , l _ { t } ^ { i } ) c ( G _ { n } , l _ { t } ^ { i } ) ,
$$

where $T _ { m a x }$ represents the maximum iteration number, $l _ { t } ^ { i }$ denotes the $i$ -th node label of the iteration $t$ , and $c ( G _ { m } , l _ { t } ^ { i } )$ counts the subtree pattern labelled by $l _ { t } ^ { i }$ in $G _ { m }$ .

The idea of the SPGK kernel is to compare the isomorphic shortest paths within pairwise graphs. Given a pair of graphs $G _ { m }$ and $G _ { n }$ , the SPGK kernel is defined as

$$
K _ { \mathrm { S P } } = \sum _ { p _ { i } \in \mathcal { P } } c ( G _ { m } , p _ { i } ) c ( G _ { n } , p _ { i } ) ,
$$

where $\mathcal { P }$ denotes the set of all possible shortest paths appearing in all graphs, $c ( G _ { m } , p _ { i } )$ is the function that counts the number of the shortest path of length $p _ { i }$ in $G _ { m }$ .

Remarks: Both the WLSK and SPGK kernels tend to use all substructures to compute the kernel matrix, without considering the varying roles of different substructures. Furthermore, the construction of the kernel matrix and the classifier are independent from each other, these kernels cannot provide an end-to-end kernelized learning architecture.

# The Graph Neural Networks (GNNs)

GNNs have been widely employed for handling graph structures (Niepert, Ahmed, and Kutzkov 2016). Specifically, they leverage either spectral or spatial-based convolution strategies to learn effective graph representations and have achieved outstanding performance for graph classification. For instance, (Atwood and Towsley 2016) have proposed the Diffusion Convolution Neural Network (DCNN) that utilizes the diffusion convolution operation to propagate the information across the graph. (Zhang et al. 2018) have proposed the Deep Graph Convolution Neural Network (DGCNN) that employs a unique SortPooling layer to sort graph nodes into a consistent order based on the substructure information. ( $\mathrm { \Delta X u }$ et al. 2019) have proposed the Graph Isomorphism Network (GIN) to maximize the ability to distinguish different graph structures, making it as powerful as the Weisfeiler-Lehman (WL) graph isomorphism test.

Remarks: Although the above GNNs can provide an endto-end learning framework, they still suffer from other drawbacks. First, the GIN and DCNN tend to directly sum up the node features as the global representation, not only discarding local node information but also ignoring the structural differences of different nodes. Second, the DGCNN only preserves the local structure information residing on the topranked nodes, resulting in topological information loss.

⊙ ：Dot Product⨁：Sum

![](images/15ef47d4c13d11838eacf71e0123f47c9bd57593a9cadf7368c64866e4d59817.jpg)  
Figure 2: The framework of the proposed DHAKR model.

# The Proposed DHAKR Model The Overall Framework of the DHAKR Model

The overall framework is exhibited in Fig.2. Specifically, it contains five main computational procedures. First, we compute the feature vector $\phi ( G _ { i } )$ for each graph $G _ { i }$ based on a specific R-convolution kernel. Since the WLSK and SPGK have more effective performance for graph classification and their associated substructures have proven to be powerful structural characteristics of the original graph, in this work we extract the shortest paths and WL-based subtrees for $\phi ( G _ { i } )$ . Second, we hierarchically combine original substructures into more meaningful structures, by adaptively computing the assignment matrix to hierarchically converge the homogeneous substructure invariants into composite substructure invariants. Third, during the above combination process, we employ the attention mechanism to discriminate the importance of different substructures for the substructure invariant converging. Fourth, we compute the hierarchical kernel matrices by concatenating the individual kernel matrices based on different hierarchical-level composite substructure invariants. Each individual kernel matrix is assigned attention-based distinct weights for the concatenation. Finally, we employ the above concatenated kernel matrices as the multi-scale kernel-based graph embedding representations for the classifier.

# The Detailed Definitions of the DHAKR Model

The Construction of Substructure Invariants. We employ the WLSK and SPGK kernels to extract the WL subtrees and shortest paths as the initial substructure invariants. For the DHAKR(WL) model, the subtree-based feature vector $\phi _ { \mathrm { { W L } } } ( G )$ of graph $G$ is defined as

$$
\begin{array} { r } { \phi _ { \mathrm { W L } } ( G ) = [ c ( G , l _ { 1 } ) , c ( G , l _ { 2 } ) , . . . , c ( G , l _ { | \mathcal { L } | } ) ] , } \end{array}
$$

where $l _ { i }$ is the node label defined by Eq.(2), $| { \mathcal { L } } |$ refers to the number of all possible WL subtree invariants and each element $c ( G , l _ { i } )$ corresponds to the number of the subtree

labeled by $l _ { i }$ in $G$ . Similarly, for the DHAKR(SP) model, the shortest path-based feature vector $\phi _ { \mathrm { S P } } ( G )$ is defined as

$$
\begin{array} { r } { \phi _ { \mathrm { S P } } ( G ) = [ c ( G , p _ { 1 } ) , c ( G , p _ { 2 } ) , . . . , c ( G , p _ { | \mathcal { P } | } ) ] , } \end{array}
$$

where $c ( G , p _ { i } )$ denotes the number of the shortest paths of length $p _ { i }$ in $G$ , and $| \mathcal { P } |$ indicates the number of all possible shortest path. With $\dot { \phi } _ { \mathrm { { W L } } } ( G )$ or $\phi _ { \mathrm { S P } } ( G )$ to hand, we derive the feature matrix $\mathbf { X } _ { ( \cdot ) }$ for the entire dataset $\mathbf { G }$ as

$$
\begin{array} { r } { \mathbf { X } _ { ( \cdot ) } = \left( \begin{array} { c } { \phi _ { ( \cdot ) } ( G _ { 1 } ) } \\ { \cdots } \\ { \phi _ { ( \cdot ) } ( G _ { i } ) } \\ { \cdots } \\ { \phi _ { ( \cdot ) } ( G _ { N } ) } \end{array} \right) , } \end{array}
$$

where $G _ { i } \in \mathbf { G }$ , and $( \cdot )$ corresponds to either the symbol WL for WLSK or the symbol SP for SPGK.

The Hierarchical Combination of Substructures. We propose the $H$ -hierarchical combination of substructure invariants to capture the dependence and homogeneity between the invariants. With the feature matrix $\mathbf { X } _ { ( \cdot ) } ^ { ( h ) } \in \mathbb { R } ^ { \bar { N } \times f _ { h } }$ at layer $h$ , we compute the soft cluster assignment matrix as

$$
\mathbf { S } ^ { ( h ) } = s o f t m a x ( \mathbf { X } ^ { ( h ) } ^ { T } \mathbf { W } + b ) \in \mathbb { R } ^ { f _ { h } \times f _ { h + 1 } } ,
$$

where W and $b$ are the weight and bias in the linear neural network, and $f _ { h }$ denotes the feature dimensions at layer $h$ . $\mathbf { S } ^ { ( h ) }$ represents the probability of substructures from layer $h$ being assigned to each cluster at layer $h + 1$ . This allows us to combine substructure invariants in a flexible manner, where substructures with high relevance are more likely to be grouped together. Thus, we can derive the feature matrix $\mathbf { X } ^ { ( h + 1 ) }$ at layer $h + 1$ as

$$
\mathbf { X } _ { ( \cdot ) } ^ { ( h + 1 ) } = \mathbf { X } _ { ( \cdot ) } ^ { ( h ) } \mathbf { S } ^ { ( h ) } \in \mathbb { R } ^ { N \times f _ { h + 1 } } .
$$

The Attention Mechanism for Feature Selection. During the hierarchical combination procedure, since the importance of each substructure is different, we introduce the attention mechanism to facilitate feature selection, aiming to eliminate the redundant substructure information unsuitable for graph classification. Inspired by the channel attention (Hu, Shen, and Sun 2018), we employ the Average Pooling (AP) operation to squeeze the representation into a feature channel. With the feature matrix $\mathbf { \bar { X } } ^ { ( h ) }$ at layer $h$ to hand, we derive the aggregation information $\mathbf { e } ^ { h } \in \mathbb { R } ^ { \mathrm { 1 } \times f _ { h } }$ as

![](images/c3256308faa2a45cf9dddd1e6389660177ce72bbb04331c97ab1ae2b7093a673.jpg)  
Figure 3: An instance of the construction of the $h$ -th layer weighted kernel matrix.

$$
e _ { k } ^ { h } = F _ { A P } ( X _ { ( \cdot ) } ^ { ( h ) } ( : , k ) ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } x _ { i , k } ^ { h } ,
$$

where $e _ { k } ^ { h }$ is the $k$ -th element of $\mathbf { e } ^ { h }$ . The resulting attention scores of different substructures are computed through two fully-connected layers with the activation function $\sigma$ as

$$
\pmb { \alpha } ^ { h } = s o f t m a x ( W _ { 2 } \sigma ( W _ { 1 } \mathbf { e } ^ { \mathbf { h } } ) ) ,
$$

where $W _ { 1 }$ and $W _ { 2 }$ are learnable weights. $\alpha ^ { h }$ represents the attention score for each substructure invariant. With the attention scores, the feature matrix $\mathbf { X } ^ { ( h ) }$ can be updated as the attention-based weighted feature matrix $\mathbf { X } _ { ( \cdot ) } ^ { ( h ) ^ { \prime } }$ , i.e.,

$$
\mathbf { X } _ { ( . ) } ^ { ( h ) ^ { \prime } } = \pmb { \alpha } ^ { h } \otimes \mathbf { X } _ { ( . ) } ^ { ( h ) } ,
$$

where $\otimes$ indicates the feature-wise multiplication. By replacing $\mathbf { X } _ { ( \cdot ) } ^ { ( h ) ^ { \prime } }$ of Eq.(10) with that of Eq.(12), we can compute the weighted composite substructure invariants.

The Construction of Hierarchical Kernel Matrices. Based on the definition in (Haussler et al. 1999), the kernel matrix ${ \bf K } _ { h }$ can be computed directly by the dot product of pairwise feature vectors, i.e.,

$$
\mathbf { K } _ { h } = \mathbf { X } _ { ( \cdot ) } ^ { ( h ) ^ { \prime } } \cdot \mathbf { X } _ { ( \cdot ) } ^ { ( h ) ^ { \prime } } ^ { T } .
$$

When we vary the parameter $h$ from 1 to $H$ , a family of $H$ -hierarchical kernel matrices is formed as

$$
{ \mathbb K } _ { H } = \{ { \bf K } _ { 1 } , . . . , { \bf K } _ { h } , . . . , { \bf K } _ { H } \} ,
$$

To capture complex dependencies among the kernel matrices and extract multi-scale kernel embeddings, we employ channel attention to assign the kernel matrices different weights. We treat the number of kernel matrices as the number of channels, i.e., concatenate $h$ kernel matrices into a 3D matrix $\mathcal { K } _ { h } \in \dot { \mathbb { R } } ^ { N \times N \times h }$ with $h$ channels. We first use the Global Average Pooling (GAP) to extract the aggregation information $\mathbf { e } ^ { h ^ { \prime } } \in \mathbb { R } ^ { 1 \times h }$ in the kernel channel, i.e.,

$$
e _ { j } ^ { h ^ { \prime } } = F _ { G A P } ( { K } _ { h } ( : , : , j ) ) = \frac { 1 } { N } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { N } K _ { h } ( i , k , j ) ,
$$

where $e _ { j } ^ { h ^ { \prime } }$ is the $j$ -th element of $\mathbf { e } ^ { h ^ { \prime } }$ . Similar to the feature attention, we also use two fully-connected layers with activation function as shown in Eq.(11). Then, the kernel attention score $\alpha ^ { h ^ { \prime } }$ is derived. Different from the feature attention of Eq.(12), we use the kernel-wise multiplication and summation for the hierarchical kernel matrices from layer 0 to $h$ , and update the $h$ -th layer weighted kernel matrix as

$$
\mathbf { K } _ { h } ^ { \prime } = \sum _ { i = 1 } ^ { h } \boldsymbol { \alpha } ^ { i ^ { \prime } } \otimes \mathbf { K } _ { i }
$$

An instance of the construction of the $h$ -th layer weighted kernel matrix is shown in Fig.3. Hence, the family of $H$ - hierarchical kernel matrices can be updated as

$$
{ \mathbb K } _ { H } ^ { \prime } = \{ { \bf K } _ { 1 } ^ { \prime } , . . . , { \bf K } _ { h } ^ { \prime } , . . . , { \bf K } _ { H } ^ { \prime } \} .
$$

The Kernel-based Graph Embedding Representations. We employ the kernel matrices as graph embedding representations. (Bunke and Riesen 2008) and (Bai, Hancock, and Han 2013) have demonstrated that a graph can be embedded into a feature vector by means of its (dis)similarities to prototype graphs. Inspired by this method, we employ all sample graphs as prototype graphs and embed each graph into vectors using the kernel-based graph similarities. The resulting kernelized graph embeddings at layer $h$ is

$\phi _ { ( \cdot ) } ^ { h } ( G ) = [ K _ { h ( \cdot ) } ^ { \prime } ( G , G _ { 1 } ) , . . . , K _ { h ( \cdot ) } ^ { \prime } ( G , G _ { i } ) , . . . , K _ { h ( \cdot ) } ^ { \prime } ( G , G _ { N } ) ] ,$ (18) where each element $K _ { h ( \cdot ) } ^ { \prime } ( G , G _ { i } )$ denotes the kernel value between $G$ and $G _ { i }$ at layer $h$ , and $( \cdot )$ represents either the WLSK or SPGK kernel.

Objective Functions. To introduce the supervision signals to guide the model, we feed the family of $H$ -hierarchical kernel embeddings $\mathbb { K } _ { H } ^ { \prime }$ to dense layers for classification. Given $N$ graphs and $M$ labels, we can derive the label prediction $\hat { \mathbf { Y } } \in \mathbb { R } ^ { N \times M }$ in the following way:

$$
\hat { { \mathbf Y } } = s o f t m a x ( \sum _ { h = 1 } ^ { H } { \mathbf W _ { h } \cdot \mathbf K _ { h } ^ { \prime } } + { \mathbf b } _ { \mathbf h } ) ,
$$

Table 1: Information of the graph datasets   

<html><body><table><tr><td>Datasets</td><td>MUTAG PTC(MR)</td><td></td><td>PROTEINS</td><td>IMDB-B</td><td>IMDB-M Shock</td><td></td></tr><tr><td>Max #vertices</td><td>28</td><td>109</td><td>620</td><td>136</td><td>89</td><td>33</td></tr><tr><td>Mean #vertices</td><td>17.93</td><td>25.56</td><td>39.06</td><td>19.77</td><td>13</td><td>13.16</td></tr><tr><td># graphs</td><td>188</td><td>344</td><td>1113</td><td>1000</td><td>1500</td><td>150</td></tr><tr><td>#classes</td><td>2</td><td>2</td><td>2</td><td>2</td><td>3</td><td>10</td></tr><tr><td>Description</td><td>Bio</td><td>Bio</td><td>Bio</td><td>SN</td><td>SN</td><td>CV</td></tr></table></body></html>

where $\mathbf { W _ { h } }$ and $\mathbf { b _ { h } }$ are the weight and bias at the $h$ -th layer. Then the cross-entropy loss for graph classification over all training graphs is represented as $L _ { C }$ :

$$
L _ { C } = - \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { M } y _ { i } l o g ( \hat { y } _ { i , j } ) ,
$$

where $y _ { i }$ denotes the real label of the $i$ -th graph. Note that, each substructure should be assigned to a specific cluster, i.e., the cluster assignment probability for each substructure should be close to a one-hot vector. Thus, we add the regulation loss for soft cluster assignment matrix S:

$$
L _ { E } = \frac { 1 } { H } \frac { 1 } { R } \sum _ { h = 1 } ^ { H } \sum _ { r = 1 } ^ { R } H _ { E } ( S _ { r , \cdot } ^ { ( h ) } ) ,
$$

where $S _ { r , \cdot } ^ { ( h ) }$ denotes the $r$ -th row of $\mathbf { S }$ at the $h$ -th layer. $H _ { E }$ indicates the entropy function. In summary, we have the following overall objective function:

$$
l o s s = L _ { C } + \gamma L _ { E } ,
$$

where $\gamma$ is the coefficient of cluster assignment regulation.

# Computational Complexity Analysis

With $| V |$ and $| E |$ as the average number of nodes and edges, $N$ is the number of graphs, $T _ { m a x }$ is the number of iterations in WLSK, $d$ is the largest feature dimension, and $H$ is the number of hierarchical combined layers. First, the time complexities of computing explicit feature vectors based on SPGK and WLSK are ${ \mathsf { \bar { O } } } ( { \dot { N } } | V | ^ { 3 } + N d ^ { 2 } )$ and $O ( T _ { m a x } | E | )$ respectively. Besides, the time complexities of feature attention and kernel attention are $O ( N d ^ { \cdot } + d ^ { 2 } )$ and $O ( N ^ { 2 } H + H ^ { 2 } )$ . The time complexities of computing assignment loss and classifier are both $O ( N ^ { 2 } d )$ . Since $H < <$ $N$ and $H \ < < \ d$ , the total computational complexities of DHAKR(SP) and DHAKR(WL) are $O ( N | V | ^ { 3 } + N d ^ { 2 } +$ $N ^ { 2 } d )$ and $O ( T _ { m a x } | E | + d ^ { 2 } + N ^ { 2 } d )$ .

# Discussions of the Proposed DHAKR Model

Unlike some existing state-of-the-art methods, the proposed DHAKR has some theoretical advantages. First, the proposed DHAKR can either utilize the independent substructures to capture the local information or employ the composite substructures to extract the global structure similarity. Second, we integrate the feature attention mechanism to assign larger weights to more significant substructures. Additionally, the end-to-end framework allows us to adaptively identify more important structural features. Third, the proposed DHAKR model can identify the common structural patterns by computing the kernel-based similarity over all sample graphs.

# Experiments

We evaluate the classification performance of the DHAKR model on standard graph datasets(Siddiqi et al. 1999; Morris et al. 2020) extracted from bioinformatics (Bio), social networks (SN), and computer vision (CV). The statistical information of the datasets is shown in Table 1.

# Comparisons with Graph Kernels

Experimental Settings. We compare the proposed DHAKR with six existing graph kernels, including 1) GCGK (Shervashidze et al. 2009) 2) RWGK (Ga¨rtner, Flach, and Wrobel 2003), 3) SPGK (Borgwardt and Kriegel 2005), 4) WLSK (Shervashidze et al. 2011), (5) CORE SP (Nikolentzos et al. 2018a), and (6) CORE WL (Nikolentzos et al. 2018b). To make a fair comparison, we perform a 10-fold cross-validation and repeat the experiments 10 times. Table 2 reports the average accuracies and standard deviations. Since some datasets are not evaluated in the original papers, we do not report the results. We use DHAKR(WL) and DHAKR(SP) to denote the DHAKR variants based on WLSK and SPSK. For our proposed methods, the assignment ratio is 0.5. The detailed descriptions of baselines and the implementation details are provided in the Arxiv version.

Experimental Results and Analysis. As we can observe from Table 2, our proposed DHAKR model achieves highly competitive performance. Compared to the state-of-the-art graph kernels, DHKAR has the highest accuracies except for PROTEINS. Note that, DHAKR(SP) still performs better than the original SPGK on PROTEINS. These observations demonstrate the theoretical advantages of the proposed DHAKR, i.e., adaptively identifying the importance of different substructures and hierarchically integrating the relationships between substructure invariants. The experimental results also indicate that the end-to-end kernel learning framework is more beneficial for classification.

# Comparisons with Graph Deep Learning

Experimental Settings. We compare the proposed DHAKR with eight alternative graph deep learning methods, including 1) DGCNN (Zhang et al. 2018), 2) DCNN (Atwood and Towsley 2016), 3) PATCHYSAN (Niepert, Ahmed, and Kutzkov 2016), 4) DGK (Yanardag and Vishwanathan 2015), 5) $q$ -RWNN (Nikolentzos and Vazirgiannis 2020) associated with three different random walk length $q$ $( q = 1 , 2 , 3 )$ 6) KerGNN (Feng et al. 2022) 7) CapsGNN (Xinyi and Chen 2019), and 8) GIN (Xu et al. 2019).

Table 2: Classification accuracy (in $\% \pm$ standard error) comparisons with graph kernels. A.R. is the Average Rank.   

<html><body><table><tr><td>Datasets</td><td>MUTAG</td><td>PTC(MR)</td><td>PROTEINS</td><td>IMDB-B</td><td>IMDB-M</td><td>Shock</td><td>A.R.</td></tr><tr><td>GCGK3</td><td>82.04±0.39</td><td>55.41±0.59</td><td>71.67±0.55</td><td></td><td></td><td>26.93±0.63</td><td>7.6</td></tr><tr><td>RWGK</td><td>80.77±0.72</td><td>55.91±0.37</td><td></td><td>74.20±0.40 67.94±0.77 46.72±0.30</td><td></td><td>2.31±1.13</td><td>6.8</td></tr><tr><td>SPGK</td><td>83.38±0.81</td><td>55.52±0.46</td><td>75.10±0.50</td><td></td><td>71.26±1.04 51.33±0.57 37.88±0.93</td><td></td><td>5.2</td></tr><tr><td>WLSK</td><td>82.88±0.57</td><td></td><td></td><td></td><td>58.26±0.47 73.52±0.43 71.88±0.77 49.50±0.49 36.40±1.00</td><td></td><td>5.5</td></tr><tr><td>CORE SP</td><td>88.29 ±1.55</td><td>59.06 ±0.93</td><td>一</td><td>72.62±0.59 49.43±0.42</td><td></td><td></td><td>6.6</td></tr><tr><td>COREWL</td><td>87.47±1.08</td><td>59.43±1.20</td><td>一</td><td></td><td>74.02±0.42 51.35±0.48</td><td>一</td><td>4.8</td></tr><tr><td>DHAKR(SP)</td><td>88.44± 0.64</td><td></td><td></td><td></td><td>67.33±1.52 76.09±0.97 76.11±0.61 53.21±0.54 55.41±1.67</td><td></td><td>1.5</td></tr><tr><td>DHAKR(WL)</td><td>89.87±1.03</td><td></td><td></td><td></td><td>68.76±0.96 77.47±0.53 75.21±0.54 52.07±0.39 48.75±1.26</td><td></td><td>1.7</td></tr></table></body></html>

<html><body><table><tr><td>Datasets</td><td>MUTAG</td><td>PTC(MR)</td><td>PROTEINS</td><td>IMDB-B</td><td>IMDB-M</td><td>A.R.</td></tr><tr><td>DGCNN</td><td>85.83±1.66</td><td>58.57±1.69</td><td>75.54±0.94</td><td>70.03±0.86</td><td>47.83±0.85</td><td>7.0</td></tr><tr><td>DCNN</td><td>66.98</td><td>56.60</td><td>61.29±1.60</td><td>49.06±1.37</td><td>46.72±0.30</td><td>10.8</td></tr><tr><td>PATCHY-SAN</td><td>88.95±4.37</td><td>62.29 ± 5.68</td><td>75.00±2.51</td><td>71.00±2.29</td><td>45.23±2.84</td><td>5.8</td></tr><tr><td>DGK</td><td>82.66±1.45</td><td>60.08 ± 2.55</td><td>71.68±0.50</td><td>66.96±0.56</td><td>44.55±0.52</td><td>9.4</td></tr><tr><td>1-RWNN</td><td>89.2±4.3</td><td>57.36±1.66</td><td>70.8±4.8</td><td>70.8±4.8</td><td>47.8±3.8</td><td>7.4</td></tr><tr><td>2-RWNN</td><td>88.1±4.8</td><td>57.61±1.43</td><td>74.7±3.3</td><td>70.6±4.4</td><td>48.8±2.9</td><td>6.8</td></tr><tr><td>3-RWNN</td><td>88.6±4.1</td><td>58.40±1.86</td><td>74.1±2.8</td><td>70.7±3.9</td><td>47.8±3.5</td><td>7</td></tr><tr><td>KerGNN</td><td></td><td></td><td>76.5±3.9</td><td>74.4±4.3</td><td>51.6±3.1</td><td>8.2</td></tr><tr><td>CapsGNN</td><td>86.67±6.88</td><td></td><td>76.28±3.63</td><td>73.10±4.83</td><td>50.27±2.65</td><td>5.6</td></tr><tr><td>GIN</td><td>84.7±6.7</td><td>64.29±1.26</td><td>74.3±3.3</td><td>71.23±3.9</td><td>48.53±3.3</td><td>6.0</td></tr><tr><td>DHAKR(SP)</td><td>88.44± 0.64</td><td>67.33±1.52</td><td>76.09±0.97</td><td>76.11±0.61</td><td>53.21±0.54</td><td>2.4</td></tr><tr><td>DHAKR(WL)</td><td>89.87±1.03</td><td>68.76±0.96</td><td>77.47±0.53</td><td>75.21±0.54</td><td>52.07±0.39</td><td>1.4</td></tr></table></body></html>

Table 3: Classification accuracy (in $\% \pm$ standard error) comparisons with deep learning methods. AR is the Average Rank.

Experimental Results and Analysis. Table 3 indicates that the proposed DHAKR outperforms graph deep learning methods on all datasets. DHAKR(SP) achieves the best classification performance on PTC(MR), IMDB-BINARY and IMDB-MULTI datasets. DHAKR(WL) also has the highest accuracy on MUTAG and PROTEINS datasets. The experimental results meet our expectations since the pooling operation discards the graph information for existing GNNs. Besides, these results also demonstrate the effectiveness of the kernel-based framework, i.e., considering the common patterns shared between all graphs instead of stacking more message-passing layers.

# The Further Analysis for DHAKR

The Ablation Study. In this section, we compare DHAKR with its variants on three datasets to validate the effectiveness of the feature attention and kernel attention components. From the results shown in Table 4, we can draw the following conclusions:(1) Whether based on SPGK or WLSK, the accuracies of DHAKR are better than models without kernel attention, verifying the usefulness of kernel attention. (2) When the model removes the feature attention component, the classification accuracy drops sharply, demonstrating the effectiveness of adaptively identifying the importance of substructure invariants. The additional results are shown in the Arxiv version.

Hyperparameter Analysis. We further investigate the sensitivity of $\gamma$ in Eq.(22). We vary the values of $\gamma$ from 0.0001 to 1.0 and test the graph classification performance on four datasets as shown in Fig.4. With the increase of the regularization coefficient $\gamma$ , the classification accuracies rise first and drop slowly on MUTAG and PROTEINS datasets, whereas it initially drops and then gradually improves on the IMDB-MULTI and Shock datasets. It indicates that different values of $\gamma$ may result in varying classification performances. Fig.4 also shows that the regularization coefficient is more sensitive in the DHAKR(WL) model, i.e., it has a higher impact on graph classification accuracy compared to the DHAKR(SP) model. We also report the results of hyperparameter analysis on PTC(MR) and IMDB-BINARY datasets in the Arxiv version.

Table 4: The ablation study of DHAKR. w/o KAtt means variants without kernel attentions and w/o FAtt denotes variants without feature attentions.   

<html><body><table><tr><td>Datasets</td><td>MUTAG</td><td>PROTEINS IMDB-M</td></tr><tr><td>DHAKR(WL)</td><td>89.87±1.03</td><td>77.47±0.53 52.07±0.39</td></tr><tr><td>DHAKR(WL)w/oKAtt 89.62±0.79</td><td></td><td>77.05±0.33 51.87±0.60</td></tr><tr><td>DHAKR(WL) w/o FAtt 86.81±1.29</td><td></td><td>72.74±0.96 49.39±0.79</td></tr><tr><td>DHAKR(SP)</td><td>88.44±0.64</td><td>76.09±0.97 53.21±0.54</td></tr><tr><td>DHAKR(SP) w/o KAtt</td><td>88.33±0.59 75.79±1.22</td><td>52.50±0.43</td></tr><tr><td>DHAKR(SP) w/o FAtt</td><td>80.05±2.40</td><td>74.55±0.51 44.36±0.51</td></tr></table></body></html>

Visualization of Graph Representations. To intuitively demonstrate the effectiveness of our proposed method in learning graph representation, we conduct the visualization using t-SNE (Van der Maaten and Hinton 2008) and compare the representations with the GIN backbone on the PTC(MR) and PROTEINS datasets as Fig.5. For the DHAKR(WL) model, we extract the graph embeddings before dense layers for classifying. For the GIN model, we extract the representations by the readout function. The results in Fig.5 show that DHAKR provides more distinct boundaries between graphs of different classes while effectively clustering graphs of the same class together.

![](images/43c44a0f1db7f333e34ad63f2b791ff3585dd7fac2d28357546a3f3042860a35.jpg)  
Figure 4: The hyper-parameter sensitivity study.

![](images/c4bb175bef6cd60e3ca303ebbb27158b6593b19a483799ab0a4266a5b98b59de.jpg)  
Figure 5: The t-SNE visualization of graph representations.

Visualization of Hierarchical Combination. We further visualize the assignment score heatmaps for the MUTAG dataset based on the DHAKR(WL) model. Since the DHAKR model has a three-layer hierarchical structure with two assignment matrices, we provide corresponding visualizations of the matrix for each layer. The results in Fig.6 demonstrate that DHAKR can adaptively integrate homogeneous substructure invariants to generate new features.

To present the process of hierarchical feature assignment more intuitively, we select 24 substructures from the MUTAG dataset and assign them to the cluster with the highest probability after 500 training epochs. The visualization of

1.0 0.8 0.6 0.6 0.4 0.4 -0.2 -0.2 18- -0.0 -0.0 0 2468   
Feature ID at layer 1 Feature ID at layer 2   
(a) Hierarchical Layer 1 (b) Hierarchical Layer 2

hierarchical composite features in Fig.7 indicates that homogeneous substructures tend to cluster together. Therefore, the combined hierarchical framework enables DHAKR to capture global topological information at higher layers and local substructures at lower layers.

![](images/7fce1502cf4736b8af71a0f66a04c10d1999022e36d6f7fc688744e6540417fe.jpg)  
Figure 6: The assignment score visualization on MUTAG.   
Figure 7: The hierarchical composite feature visualization.

# Conclusion

In this paper, we have proposed a novel DHAKR model for graph classification. This model cannot only hierarchically extract composite substructure invariants, but also adaptively identify the importance of different substructures. Moreover, the proposed model can provide an end-to-end learning architecture that captures the common structural patterns over all sample graphs. Experiments demonstrate the superior classification performance.

Our future work is to consider the hybrid informative substructures, including random walks, cycles, etc. Moreover, the structurally aligned (sub)structures, as successfully explored in our previous works (Bai et al. 2024; Cui et al. 2024; Bai et al. 2015), will also be further utilized for proposing novel aligned kernelized methods. Finally, we will also utilize the subtrees extracted from the Perron-Frobenius operator of hypergraphs (Bai, Ren, and Hancock 2014; Bai, Escolano, and Hancock 2016), resulting in novel hypergraphbased kernalized methods.

# Acknowledgments

This work is supported by the National Natural Science Foundation of China under Grants T2122020, 61602535, 61976235. Ming Li acknowledged the supports from the Pioneer and Leading Goose R&D Program of Zhejiang (No. 2024C03262), the National Natural Science Foundation of China (No. U21A20473, No. 62172370) and the Jinhua Science and Technology Plan (No. 2023-3-003a). Hangyuan Du acknowledged the support from the Humanity and Social Science Foundation of Ministry of Education (24YJAZH022). This work is also supported in part by the Program for Innovation Research in the Central University of Finance and Economics, and the Emerging Interdisciplinary Project of CUFE.