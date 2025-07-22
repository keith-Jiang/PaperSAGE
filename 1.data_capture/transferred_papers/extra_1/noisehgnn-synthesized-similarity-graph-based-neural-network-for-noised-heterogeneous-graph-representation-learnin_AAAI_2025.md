# NoiseHGNN: Synthesized Similarity Graph-Based Neural Network For Noised Heterogeneous Graph Representation Learning

Xiong Zhang1, Cheng Xie1\* Haoran Duan2 , Beibei $\mathbf { Y } \mathbf { u } ^ { 3 }$

1 School of Software, Yunnan University, Kunming, China 2 School of Cyber Science and Engineering, Wuhan University, Wuhan, China 3 Australian AI Institute, University of Technology Sydne, Sydney, Australia zhangxiong@stu, xiecheng $@$ ynu.edu.cn, hrduan $0 7 \textcircled { a }$ gmail.com, Beibei.Yu@student.uts.edu.au

# Abstract

Real-world graph data environments intrinsically exist noise (e.g., link and structure errors) that inevitably disturb the effectiveness of graph representation and downstream learning tasks. For homogeneous graphs, the latest works use original node features to synthesize a similarity graph that can correct the structure of the noised graph. This idea is based on the homogeneity assumption, which states that similar nodes in the homogeneous graph tend to have direct links in the original graph. However, similar nodes in heterogeneous graphs usually do not have direct links, which can not be used to correct the original noise graph. This causes a significant challenge in noised heterogeneous graph learning. To this end, this paper proposes a novel synthesized similarity-based graph neural network compatible with noised heterogeneous graph learning. First, we calculate the original feature similarities of all nodes to synthesize a similarity-based high-order graph. Second, we propose a similarity-aware encoder to embed original and synthesized graphs with shared parameters. Then, instead of graph-to-graph supervising, we synchronously supervise the original and synthesized graph embeddings to predict the same labels. Meanwhile, a target-based graph extracted from the synthesized graph contrasts the structure of the metapathbased graph extracted from the original graph to learn the mutual information. Extensive experiments in numerous realworld datasets show the proposed method achieves state-ofthe-art records in the noised heterogeneous graph learning tasks. In highlights, $+ 5 \sim 6 \%$ improvements are observed in several noised datasets compared with previous SOTA methods.

# 1 Introduction

Graph representation learning is one of the most significant research fields in artificial intelligence, since most intelligent applications are based on graph representations such as recommendation systems (Lv et al. 2021; Fan et al. 2019; Zhao et al. 2017), social networks (Qiu et al. 2018; Li and Goldwasser 2019; Wang et al. 2019a), biomedicine (Gaudelet et al. 2021; Fout et al. 2017; Davis et al. 2019), fraud detection (Shchur et al. 2018; Dou et al. 2020), e-commerce (Ji et al. 2021; Zhao et al. 2019), etc. However, real-world graph data environments inherently contain noise data (e.g., structure errors, etc.) that challenge the representation models. Specifically, representation models are required to maintain their effectiveness in noised graph data environments.

![](images/5821d91e93d8f86f552ca9b290f947e7f529f54eb346f043967a8cc51fe408fa.jpg)  
Figure 1: Homogeneity assumption works for noised homogeneous graph but failed in noised heterogeneous graph.

The latest works attempting to solve this problem by optimizing graph structure (Wang et al. 2021; Wei et al. 2022) or synthesizing graph structure (Fatemi, El Asri, and Kazemi 2021; Franceschi et al. 2019; Jin et al. 2020; Liu et al. 2022) to alleviate the noise. Nowadays, graph synthesizing-based methods achieve impressive performance in noised homogeneous graph representation learning, based on the homogeneity assumption to create a synthesized graph to correct the noise homogeneous graph. As an example shown in Fig.1 (a), a paper citation graph has four paper nodes: P1, P2, P3, and P4. The links between nodes represent the references among the papers. The red link means P1 and P4 are incorrectly realized to have a reference relationship. To fix these error links, a similarity graph can be constructed in advance by calculating the node-to-node feature similarities without using links. It can be observed that P1 and P4 have minimal similarity values (0.03), which means P1 is unlikely to cite P4 in the original graph. Thus, the original noised graph will likely be corrected by calculating the loss between it and the similarity graph. This idea is based on the homogeneity assumption, which states that similar nodes tend to have direct links in a homogeneous graph.

However, the above solution can not be simply applied to the heterogeneous graphs. As an example shown in Fig.1 (b), a heterogeneous graph represents the relationships between papers and authors. A similarity graph shows that P1, P2, and P3 are similar enough to have a link. We will obtain a wrong correction result if we directly calculate the loss between the original and similarity graphs. This is because links in a heterogeneous graph do not represent “similar” semantics. Different types of links represent different semantics that need to be specifically dealt with. Current metapath-dependent (Wang et al. 2019b; Fu et al. 2020) or edgerelationship-oriented (Zhou et al. 2023; Lv et al. 2021; Zhao et al. 2022; Zhu et al. 2019; Zhang et al. 2019; Schlichtkrull et al. 2018; Du et al. 2023) models prone to misconnections in links when there are errors in the graph structure, leading to effectiveness degradation. Thus, maintaining representation models’ effectiveness in noised heterogeneous graphs is still challenging.

To address this challenge, we propose a novel NoiseHGNN model compatible with noised heterogeneous graph learning. Since the similarity and heterogeneous graphs represent different semantics, we do not calculate their mutual information to correct the noised graph. Instead, two novel modules, Similarity-aware HGNN, and MetapathTarget contrastive learning, are proposed. In the Similarityaware HGNN module, the synthesized similarity graph is used to reinforce the node attention during the representation learning instead of directly supervising the noised graph. In the Metapath-Target contrastive module, instead of directly contrasting synthesized and noised graphs, we contrast the metapath-based graph (extracted from the noised graph) and the target-based graph (extracted from the synthesized graph) that represent the same semantics. Then, the representations of both the noised and synthesized graphs are jointly utilized to predict the labels during model training. Finally, the noised graph representation is used to predict the label during the testing. Extensive experiments on five real-world datasets demonstrate the effectiveness and generalization ability of NoiseHGNN. The proposed NoiseHGNN achieves state-of-the-art records on four out of five extensively benchmark datasets under the noised data environment. In highlights, in the complex and noise-sensitive dataset (DBLP, PubMed, and IMDB), $+ 3 \sim + 6 \%$ improvement is observed compared with peer methods. In the rest of the datasets, $+ 1 \sim + 2 \%$ improvement is observed compared with peer methods. The code and datasets are available at https://github.com/kg-cc/NoiseHGNN.

In summary, our contributions are as follows:

• It is the first work, to our best knowledge, to investigate noised heterogeneous graph representation learning and achieves state-of-the-art records.   
• The proposed Metapath-Target contrastive learning bridges the homogeneity assumption to the noised heterogeneous graph representation.   
• The proposed similarity-aware HGNN takes advantage of the similarity graph from traditional homogeneous graphs to the noised heterogeneous graph representation learning.

# 2 Related Work

# 2.1 Graph Structure Learning

For homogeneous graphs, existing methods address erroneous link perturbations by constructing and refining the graph structure based on the homogeneous graph assumption through graph structure learning. These methods correct the structure of noisy graphs by synthesizing similarity graphs using features of the original nodes. Specifically, existing methods employ probabilistic models (Wang et al. 2021; Franceschi et al. 2019) and metric learning models (Fatemi, El Asri, and Kazemi 2021; Franceschi et al. 2019; Jin et al. 2020; Liu et al. 2022) to parameterize the adjacency matrix and jointly optimize the parameters of the adjacency matrix and GNNs by solving downstream models. However, in heterogeneous graphs, similar nodes are often not directly connected, making it difficult to use these methods to correct the original noisy graph. For heterogeneous graphs, HGSL (Zhao et al. 2021) simultaneously performs Heterogeneous Graph Structure Learning and GNN parameter learning for classification. However, it does not account for the potential noise introduced by erroneous edges within the heterogeneous graph structure.

# 2.2 Heterogeneous Graph Neural Network

HGNNs can generally be categorized into two types based on their strategies for handling heterogeneity: messagepassing based HGNNs, meta-paths based HGNNs.

Message-passing based HGNNs. RGCN (Schlichtkrull et al. 2018) assigns different weight matrices to various relation types and aggregates one-hop neighbors. RSHN(Zhu et al. 2019) builds a coarsened line graph to get edge features and adopts message passing to propagate node and edge features. SimpleHGN(Lv et al. 2021) incorporates relational weight matrices and embeddings to characterize heterogeneous attention at each edge. Additionally, Space4HGNN (Zhao et al. 2022) defines a unified design space for HGNNs to exhaustively evaluate combinations of multiple technologies. SlotGAT (Zhou et al. 2023) designs a slot for each type of node according to the node type and uses the slot attention mechanism to construct the HGNN model.

Meta-paths based HGNNs. Another class of HGNNs captures higher-order semantic information through metapaths. HAN(Wang et al. 2019b) employs hierarchical attention mechanisms to capture both node-level importance between nodes and the semantic-level importance of metapaths. MAGNN(Fu et al. 2020) enhances this approach with several meta-path encoders to comprehensively encode information along meta-paths. In contrast, the Graph Transformation Network(Yun et al. 2019) (GTN) can automatically learn meta-paths through graph transformation layers. However, for heterogeneous graphs with multiple edge types, meta-path-based methods are less practical due to the high cost of acquiring meta-paths. HetGNN (Zhang et al. 2019) addresses this issue by using random walks to sample fixedsize neighbors for nodes of different types and then applying recurrent neural networks (RNNs) for representation learning. Seq-HGNN (Du et al. 2023) designs a sequential node representation learning mechanism to represent each node as a sequence of meta-path representations during the node message passing.

![](images/8fbb98fa7a116b67f6faa9655f7d6f56e039e598469a315a66b1417e9285f8f1.jpg)  
Figure 2: The overall framework of the proposed model.

# 3 Preliminaries

# 3.1 Heterogeneous Graph

A heterogeneous graph (Sun and Han 2012) can be defined as $G \ : = \ : \{ \mathcal { V } , \mathcal { E } , \phi , \psi \}$ , where $\nu$ is the set of nodes and $\mathcal { E }$ is the set of edges. Each node $v$ has a type $\phi ( v )$ and each edge $e$ has a type $\psi ( e )$ . The sets of possible node types and edge types are denoted by $T _ { v } ~ = ~ \{ \phi ( v ) ~ : ~ \forall v ~ \in ~ \mathcal { V } \}$ and $T _ { e } { \ ' } = { \ ' } { \bar { \{ \psi } }  ( e ) : \forall e \in { \mathcal { E } } \}$ , respectively. For a heterogeneous graph $| \psi | + | \phi | > 2$ . When $\left| T _ { v } \right| = \left| T _ { e } \right| = 1$ , the graph degenerates into an ordinary homogeneous graph.

A node $v$ has a feature vector $\mathbf { x } _ { v }$ . For node type $t \in \Phi$ , all type- $t$ nodes $v \in \{ v \in \mathcal { V } | \phi ( v ) = t \}$ have the same feature dimension $d _ { 0 } ^ { t } = d _ { 0 } ^ { \phi ( v ) }$ ,i.e., $\mathbf { x } _ { v } \in \mathbb { R } ^ { d _ { 0 } ^ { \phi ( v ) } }$ . Nodes of different types can have different feature dimensions (Lv et al. 2021). For input nodes feature type, we use $\eta = 0$ to denote using all given features, $\eta = 1$ to denote using only target node features, and $\eta = 2$ to denote all nodes with one-hot features.

# 3.2 Noised Heterogeneous Graph

Let $G = \{ \mathcal { V } , \mathcal { E } , \phi , \psi \}$ denote a heterogeneous graph. $N$ be the number of nodes, i.e., $N = | \nu |$ , and $M$ be the number of links, i.e., $M = | \mathcal { E } |$ . The sets of node types is denoted by $T _ { v } = \{ \phi ( v ) : \forall v \in \mathcal { V } \}$ . An edge $e$ corresponds to an edge type $\psi ( e )$ connecting two types of nodes $v _ { i }$ and $v _ { j }$ , with the node types $\phi ( v _ { i } )$ and $\phi ( v _ { j } )$ respectively.

We simulate the erroneous link scenario in real data by modifying the target node $v _ { j }$ from $T _ { v _ { j } }$ connected to $v _ { i }$ . Specifically, for all datasets, we randomly modify the target nodes of $3 0 \%$ of the $M$ links, preserving the heterogeneous graph edge type $\psi ( e )$ . This ensures that the modified edge types remain consistent with their original types.

# 4 The Proposed Method

The proposed model mainly consists of four modules: a graph synthesizer, a graph augmenter, a similarity-aware graph encoder, and a graph contrastive module, as shown in Fig 2. The synthesized similarity graph contains highorder semantics used to adapt the noise link weights by the similarity-aware graph encoder. The contrastive learning between the target graph (from the synthesized graph) and the meta-path graph (from the noised graph) can further alleviate noise links.

# 4.1 Similarity Graph Synthesizer

In this work, the synthesized graph is a similarity graph. The base assumption of the similarity graph is that if two nodes are similar, they will likely have natural relationships. Thus, the similarity graph can potentially supervise the noise graph. Two processes are essential to synthesize a similarity graph: (1) node feature projection and (2) similarity adjacent matrix synthesizing.

Node Feature Projection. The node features need to be projected into a unified feature space due to the heterogeneity of nodes. Therefore, we design type-specific linear transformations to project the features of different types of nodes

$X ^ { \phi }$ into the unified feature space $Z$ , as defined in equation 1.

$$
\begin{array} { r } { Z = \left[ \begin{array} { c } { Z ^ { \phi _ { 1 } } = W ^ { \phi _ { 1 } } X ^ { \phi _ { 1 } } + b ^ { \phi _ { 1 } } } \\ { Z ^ { \phi _ { 2 } } = W ^ { \phi _ { 2 } } X ^ { \phi _ { 2 } } + b ^ { \phi _ { 2 } } } \\ { \vdots } \\ { Z ^ { \phi _ { m } } = W ^ { \phi _ { m } } X ^ { \phi _ { m } } + b ^ { \phi _ { m } } } \end{array} \right] = \left[ \begin{array} { c } { z _ { 1 } } \\ { z _ { 2 } } \\ { \vdots } \\ { z _ { N } } \end{array} \right] } \end{array}
$$

where $\phi _ { m }$ denotes the node type, $X ^ { \phi _ { m } }$ represents the feature matrix of the node type $\phi _ { m }$ , $\mathbf { \hat { W } } ^ { \phi _ { m } }$ is a learnable matrix, $b ^ { \phi _ { m } }$ denotes vector bias. We transform the node features using type-specific linear transformations for nodes with features and use $X ^ { \phi _ { m } } \in \mathbb { R } ^ { N \times d _ { 0 } ^ { \phi _ { m } } }$ to denote the node feature matrix and $ { \mathrm { ~ Z ~ } } \in \mathbb { R } ^ { N \times d ^ { \prime } }$ denotes the transformed node features.

Graph Feature Projection. The graph projection projects the graph-independent feature $Z$ into the graphdependent feature $\bar { Z }$ . Let $\eta$ as an indicator to categorize the graph data environment on feature types. $\eta = 0$ indicates that all attributes of nodes are embedded into the node features (e.g., title, description, abstract, and name of a node). $\eta = 1$ means only the target node’s attributes are embedded into the node features. $\eta = 2$ denotes the node’s one-hot attributes are used. According to the value of $\eta$ , the graph feature projection is defined as equation 2.

$$
\begin{array} { r } { \bar { Z } = \left\{ \begin{array} { l l } { \sigma ( Z W + b ) , } & { \eta \in [ 0 ] } \\ { } & { \bar { Z } = \left[ \begin{array} { c } { \bar { z } _ { 1 } } \\ { \vdots } \\ { \bar { z } _ { N } } \end{array} \right] , } & { \eta \in [ 1 , 2 ] } \end{array} \right. , \bar { Z } = \left[ \begin{array} { c } { \bar { z } _ { 1 } } \\ { \vdots } \\ { \bar { z } _ { N } } \end{array} \right] } \end{array}
$$

where $W$ is the parameter matrix, $\sigma$ is a non-linear function that makes training more stable, $\widetilde { \mathbf { A } } = \mathbf { A } + \mathbf { I }$ is the adjacency matrix with self-loop while $\widetilde { \mathrm { D } }$ ies the degree matrix of $\widetilde { \mathbf { A } }$ .

When $\eta = 0$ , a simple linear projection is applied. This is because, in this case, the node features already contain the first-order graph features (i.e., attributes of the node). When $\eta = 1 , 2$ , a typical graph convolutional network is used to aggregate topology information into the node feature. This is because, in this case, the node itself contains no semantics that need to be enriched from the graph structure.

Similarity Adjacent Matrix Synthesizing. To construct the adjacent matrix, we first calculate the similarity value node-to-node based on the graph-dependent node feature $\bar { Z }$ . The equation 3 presents the process for calculating the node similarity.

$$
S _ { i , j } = \frac { \bar { z } _ { i } \cdot \bar { z } _ { j } } { | \bar { z } _ { i } | \cdot | \bar { z } _ { j } | }
$$

where $S _ { i , j }$ is the cosine similarity between feature $\bar { z _ { i } }$ and $\bar { z _ { j } }$ . $\mathrm { S } \in \mathbb { R } ^ { N \times N }$ , and $N$ is the total number of graph node.

Similarity adjacent matrix S is usually dense and represents fully connected graph structures, which are often not meaningful for most applications and can lead to expensive computational costs (Wang et al. 2021). Therefore, we apply the $\mathbf { k }$ -nearest neighbors (kNN)-based sparsification on

S. Specifically, we retain the links with the top- $\mathbf { \nabla } \cdot \mathbf { k }$ connection values for each node and set the rest to zero. Let $A ^ { \theta }$ represent the Similarity adjacent matrix, as defined in equation 4.

$$
\begin{array} { r } { \mathrm { A } _ { i j } ^ { \theta } = \left\{ \begin{array} { l l } { \mathrm { S } _ { i j } , \quad } & { \mathrm { S } _ { i j } \in \mathrm { t o p } \mathrm { - } \mathrm { k } ( \mathrm { S } _ { i } ) , } \\ { 0 , } & { \mathrm { S } _ { i j } \notin \mathrm { t o p } \mathrm { - } \mathrm { k } ( \mathrm { S } _ { i } ) , } \end{array} \right. } \end{array}
$$

where top- $\mathbf { \cdot k } ( \mathrm { S _ { i } } )$ is the set of top- $\mathbf { \nabla } \cdot \mathbf { k }$ values of row vector $\mathrm { S _ { i } }$ . For large-scale graphs, we perform the kNN sparsification with its locality-sensitive approximation (Fatemi, El Asri, and Kazemi 2021) where the nearest neighbors are selected from a batch of nodes instead of all nodes, reducing the memory requirement.

At last, combining the graph-independent node feature $Z$ and the synthesized adjacent matrix ${ \bar { A } } ^ { \theta }$ , the synthesized similarity graph $G ^ { \theta }$ can be expressed as follows:

$$
{ G } ^ { \theta } = \{ Z , { A } ^ { \theta } \}
$$

# 4.2 Graph Augmentation

Augmentation is widely used in graph contrastive learning and representation learning. It can enhance mutual information and improve the model’s generalization ability. In this work, we apply the masking mechanism to augment the graph. In detail, for a given adjacency matrix A, we first sample a masking matrix $\mathbf { M } \in \dot { \{ 0 , 1 \} } ^ { N \times N }$ , where each element of $M$ is drawn independently from a Bernoulli distribution with probability $p ^ { \bar { ( A ) } }$ . In NoiseHGNN, we use this graph enhancement scheme to generate enhanced graphs from both the noise and synthesized graphs. The adjacency matrix is then masked with $M$ and ${ \bf \nabla } \breve { M ^ { \theta } }$ :

$$
\begin{array} { c } { { \bar { A } = A \odot M } } \\ { { \bar { A } ^ { \theta } = A ^ { \theta } \odot M ^ { \theta } } } \end{array}
$$

where $\bar { \mathrm { ~ A ~ } }$ and ${ \bar { A } } ^ { \theta }$ are the augmented noise graph and augmented synthesized graph, respectively. To obtain different context structures in the two views, edge discarding for the two views is performed with different probabilities $p ^ { ( A ) } \neq p ^ { ( A ^ { \theta } ) }$ . Other advanced enhancement schemes can also be applied to NoiseHGNN, which is left for future research.

# 4.3 Similarity-Aware HGNN Encoder

The base idea of the similarity-aware encoder is to aggregate the neighbor feature through attention and similarity. In detail, the primary process of the similarity-aware encoder is (1) correlation coefficient, (2) similarity-aware attention, and (3) attention-based aggregation.

Correlation coefficient. The same as the classic Graph Attention Network (GAT), we first calculate the correlation coefficient $\boldsymbol { e } _ { i , j }$ between a node $i$ and its neighbors $j$ , as the equation 7 shows.

$$
\begin{array} { r } { e _ { i , j } = a ( [ W z _ { i } | | W z _ { j } ] ) , j \in \mathtt { I n d e x } ( \bar { A } _ { i } ) } \\ { e _ { i , j } ^ { \theta } = a ( [ W z _ { i } | | W z _ { j } ] ) , j \in \mathtt { I n d e x } ( \bar { A } _ { i } ^ { \theta } ) } \end{array}
$$

Here, $W$ is a learnable parameter, while $z$ is the projected node feature. $a ( \cdot )$ is a projection network that projects concatenated features to a scalar number ranging from 0 to 1.0. $\operatorname { I n d e x } ( { \bar { A } } _ { i } )$ denotes a set that contains all indexes of nodes that are neighbors of node $i$ in the noise graph.

$\mathtt { I n d e x } ( { \breve { A } } _ { i } ^ { \theta } )$ denotes a set that contains all indexes of nodes that are neighbors of node $i$ in the synthesized similarity graph $G ^ { \theta }$ .

Similarity-aware attention. Similar to GAT, we calculate the attention $a _ { i , j }$ through the Softmax $( \cdot )$ operation on correlation coefficient $e _ { i , j }$ . However, the difference is that the proposed attention also considers the similarity graph’s adjacent matrix, as shown in equation 8.

$$
\begin{array} { r } { \alpha _ { i , j } = \frac { \exp \left( \mathrm { L e a k y R e L U } \left( e _ { i , j } \right) \right) } { \sum _ { k \in \mathrm { I n d e x } \left( \bar { A } _ { i } \right) } \exp \left( \mathrm { L e a k y R e L U } \left( e _ { i , k } \right) \right) } \cdot \bar { A } _ { i , j } } \\ { \alpha _ { i , j } ^ { \theta } = \frac { \exp \left( \mathrm { L e a k y R e L U } \left( e _ { i , j } ^ { \theta } \right) \right) } { \sum _ { k \in \mathrm { I n d e x } \left( \bar { A } _ { i } ^ { \theta } \right) } \exp \left( \mathrm { L e a k y R e L U } \left( e _ { i , k } ^ { \theta } \right) \right) } \cdot \bar { A } _ { i , j } ^ { \theta } } \end{array}
$$

From the above attention, the similar node pairs will have a higher attention coefficient, while the dissimilar node pairs will only have zero attention value. This potentially intercepts the message propagation on error-prone links when the corresponding attention is zero.

Attention-based aggregation. The aggregation aims to obtain the graph representation $H$ from node feature $Z$ and adjacent matrix $A$ . Let $W ^ { \alpha }$ denote a learnable parameter of the aggregation process. The noise and synthesized similarity graphs share the same $W ^ { \alpha } . \sigma ( \cdot )$ denote a LeakyReLU(·) operation. Equation 9 defines the node representation calculated by similarity-aware attention $\alpha$ and node feature $Z$ .

$$
\begin{array} { r l } & { \bar { h } _ { i } = \sigma \left( \displaystyle \sum _ { j \in \mathrm { I n d e x } ( \bar { A } _ { i } ) } \alpha _ { i j } \cdot W ^ { \alpha } \cdot z _ { j } \right) } \\ & { \bar { h } _ { i } ^ { \theta } = \sigma \left( \displaystyle \sum _ { j \in \mathrm { I n d e x } ( \bar { A } _ { i } ^ { \theta } ) } \alpha _ { i j } ^ { \theta } \cdot W ^ { \alpha } \cdot z _ { j } \right) } \end{array}
$$

where $j \in \mathtt { I n d e x } ( { \bar { A } } _ { i } )$ and $j \in \tt I n d e x ( \bar { A } _ { i } ^ { \theta } )$ represent the neighbors of node $i$ under the noise and synthesized graphs respectively. $\bar { h } _ { i }$ and $\bar { h } _ { i } ^ { \theta }$ represent the node representations of node $\boldsymbol { v } _ { i }$ under the noise and synthesized similarity graphs, respectively.

Finally, all nodes get the graph representation $H$ , which can be represented as follows:

$$
\bar { H } = \left[ \begin{array} { c } { h _ { 1 } } \\ { \vdots } \\ { h _ { N } } \end{array} \right] , \bar { H } ^ { \theta } = \left[ \begin{array} { c } { h _ { 1 } ^ { \theta } } \\ { \vdots } \\ { h _ { N } ^ { \theta } } \end{array} \right]
$$

where $\bar { H }$ denotes all node representations under the noise graph, while ${ \bar { H } } ^ { \theta }$ denotes all node representations under the synthesized similarity graph.

# 4.4 Graph Learning

Learning Objective. So far, we have the graph representations $\bar { H }$ and ${ \check { H } } ^ { \theta }$ . The representations can be used for multiple downstream objectives such as node classifications, graph classifications, clustering, link prediction, etc. This work focuses on node classifications, which categorize a node instance into a class label. Let $\mathtt { M L P ( \cdot ) }$ represent a linear neural network. The learning objective ( node classification) is defined as follows:

$$
\begin{array} { r } { \bar { \mathrm { Y } } = \mathtt { S o f t m a x } ( \mathtt { M L P } ( \bar { H } ) ) , } \\ { \bar { Y } ^ { \theta } = \mathtt { S o f t m a x } ( \mathtt { M L P } ( \bar { H } ^ { \theta } ) ) } \end{array}
$$

where $\bar { \mathrm { Y } }$ and ${ \bar { Y } } ^ { \theta }$ are the predictions obtained under the noise graph and the synthesized similarity graph, respectively.

Classification loss. The datasets normally have two characteristics: single-label and multi-label. For single-label classification, softmax and cross-entropy loss are used. For multi-label datasets, sigmoid activation and binary crossentropy loss are used. Let $\mathcal { L } _ { C E } ( \cdot )$ represent the crossentropy for both single-label and multi-label in general. $Y$ is the ground truth label of the classification. The classification loss is defined as equation 12.

$$
\begin{array} { l } { { \mathcal { L } _ { o } = \mathcal { L } _ { C E } ( \bar { Y } , \Upsilon ) } } \\ { { \mathcal { L } _ { s } = \mathcal { L } _ { C E } ( \bar { Y } ^ { \theta } , \Upsilon ) } } \end{array}
$$

where $\mathcal { L } _ { o }$ is the classification loss for the noise graph representation. $\mathcal { L } _ { s }$ is the classification loss for the synthesized similarity graph representation.

Contrastive Loss. To better mitigate the error links in the noise graph structure, we extract the meta-path $A ^ { \varphi }$ graph from the noise graph structure (e.g., author-paper-author (APA) in the DBLP) and also extract the target graph ${ \hat { A } } ^ { \theta }$ from the synthesized similarity graph, as presented in equation 13.

$$
\begin{array} { c } { { A ^ { \varphi } = \mathsf { M e t a P a t h } ( A ) } } \\ { { \hat { A } ^ { \theta } = \mathsf { T a r g e t } ( A ^ { \theta } ) } } \end{array}
$$

Based on the above subgraphs, we then use the scaled cosine loss function to calculate graph contrastive loss $\mathcal { L } _ { g }$ . Since the higher-order synthesized similarity graph is an adjacency matrix with higher-order semantic information, it can alleviate the errors in the meta-paths graph. Specifically, the contrastive loss $\mathcal { L } _ { g }$ is calculated as follows:

$$
\mathcal { L } _ { g } = ( 1 - \frac { ( A ^ { \varphi } ) ^ { T } \cdot \hat { A } ^ { \theta } } { \Vert ( A ^ { \varphi } ) ^ { T } \Vert \cdot \Vert \hat { A } ^ { \theta } \Vert } ) ^ { \gamma } , \gamma \geq 1
$$

where the scaling factor $\gamma$ is a hyper-parameter adjustable over different datasets, $A ^ { \varphi }$ denotes the graph obtained from the noise graph via meta-paths and $\hat { A } ^ { \bar { \theta } }$ denotes the target graph extracted from the synthesized graph.

Final Loss. Combining equation 12 and 13, the final loss function is formulated as:

$$
\mathcal { L } = \mathcal { L } _ { o } + \mathcal { L } _ { s } + \mathcal { L } _ { g }
$$

The whole process of the above equations is organized as a pseudo-code algorithm, see the appendix for details.

<html><body><table><tr><td></td><td colspan="2">DBLP(0.3)</td><td colspan="2">IMDB(0.3)</td><td colspan="2">ACM(0.3)</td><td colspan="2">PubMed_NC(0.3)</td><td colspan="2">Freebase(0.3)</td></tr><tr><td></td><td>Macro-Fl</td><td>Micro-Fl</td><td>Macro-F1</td><td>Micro-F1</td><td>Macro-Fl</td><td>Micro-Fl</td><td>Macro-Fl</td><td>Micro-F1</td><td>Macro-F1</td><td>Micro-F1</td></tr><tr><td>GCN (2016)</td><td>50.97±0.14</td><td>52.35±0.17</td><td>37.86±3.93</td><td>50.60±1.73</td><td>86.38±1.21</td><td>86.46±1.14</td><td>36.28±2.78</td><td>41.39±2.81</td><td>17.16±0.22</td><td>53.66±0.18</td></tr><tr><td>GAT (2017)</td><td>63.11±1.48</td><td>64.38±1.23</td><td>38.53±5.65</td><td>49.65±4.58</td><td>79.02±3.35</td><td>79.61.60±3.26</td><td>34.08±4.66</td><td>39.53±3.89</td><td>17.83±0.73</td><td>54.63±0.23</td></tr><tr><td>RGCN (2018)</td><td>43.02±1.83</td><td>45.02±1.53</td><td>39.79±1.77</td><td>47.28±0.90</td><td>54.37±2.38</td><td>55.36±2.25</td><td>15.28±3.30</td><td>19.76±3.96</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RHAN (2019)</td><td>67.01±1.47</td><td>67.69±1.39</td><td>31.40+3.2</td><td>48.09±1.45</td><td>80.3±2.09</td><td>80.43±1.383</td><td></td><td></td><td></td><td></td></tr><tr><td>HetGNN (2019)</td><td>59.31±0.23</td><td>60.23±0.36</td><td>37.36±0.75</td><td>40.85±0.86</td><td>74.45±0.13</td><td>74.39±0.15</td><td></td><td></td><td></td><td></td></tr><tr><td>MAGNN (2020)</td><td>60.83±1.40</td><td>62.66±1.07</td><td>14.30±0.20</td><td>39.98±0.12</td><td>88.67±0.84</td><td>88.64±0.73</td><td></td><td></td><td></td><td></td></tr><tr><td>HGT (2020)</td><td>48.11±4.39</td><td>53.66±2.54</td><td>45.26±0.89</td><td>55.62±0.16</td><td>81.52±1.82</td><td>81.43±1.77</td><td>46.65±3.63</td><td>49.53±2.61</td><td>10.40±0.94</td><td>49.31±0.13</td></tr><tr><td>simpleHGN (2021)</td><td>65.54±1.03</td><td>67.63±0.75</td><td>58.21±2.20</td><td>62.93±0.92</td><td>89.65±0.52</td><td>89.64±0.48</td><td>42.25±4.67</td><td>47.90±3.40</td><td>27.10±1.54</td><td>56.79±0.48</td></tr><tr><td>space4HGNN(2022)</td><td>65.43±1.12</td><td>67.81±0.55</td><td>55.43±0.54</td><td>61.91±0.94</td><td>88.52±0.74</td><td>88.63±0.34</td><td>42.41±3.59</td><td>47.14±3.16</td><td>26.93±0.78</td><td>53.54±0.33</td></tr><tr><td>Seq-HGNN(2023)</td><td>67.89±0.26</td><td>68.93±0.37</td><td>56.23±0.43</td><td>62.34±0.29</td><td>90.02±0.23</td><td>90.09±0.31</td><td>44.41±4.31</td><td>48.54±4.01</td><td>27.93±0.84</td><td>55.94±0.48</td></tr><tr><td>SlotGAT(2024)</td><td>68.23±0.75</td><td>69.77±0.19</td><td>54.89±1.77</td><td>62.47±0.97</td><td>90.64±0.46</td><td>90.65±0.42</td><td>44.76±3.36</td><td>50.88±1.54</td><td>28.14±1.31</td><td>56.56±0.64</td></tr><tr><td>NoiseHGNN</td><td>71.96±0.33</td><td>73.16±0.19</td><td>60.46±0.69</td><td>63.94±0.26</td><td>90.16±0.49</td><td>90.10±0.46</td><td>50.92±3.52</td><td>55.81±3.20</td><td>30.57±2.94</td><td>56.96±0.67</td></tr></table></body></html>

Table 1: Peer comparison on node classification task. $30 \%$ error link rate is applied to the datasets. Vacant positions (“-”) mean out of memory in our computational environment. The best record is marked in bold, and the runner-up is underlined.

# 5 Experiments

# 5.1 Experiment Settings

Datasets. Table 2 reports the statistics of the benchmark datasets widely used in previous studies (Lv et al. 2021; Zhou et al. 2023; Zhao et al. 2022). These datasets span various domains, such as academic graphs (e.g., DBLP, ACM), information graphs (e.g., IMDB, Freebase), and medicalbiological graphs (e.g., PubMed). For node classification, each dataset contains a target node type, and all nodes of this target type are used for classification.

Table 2: Statistics of Datasets.   

<html><body><table><tr><td>Classifiation</td><td>Nodes</td><td>Nype</td><td>Edges</td><td>Types</td><td>ernk</td><td>Target</td><td>Classes</td></tr><tr><td>DBLP</td><td>26,128</td><td>4</td><td>239,566</td><td>6</td><td>30%</td><td>author</td><td>4</td></tr><tr><td>IMDB</td><td>21.420</td><td>4</td><td>86.642</td><td>6</td><td>30%</td><td>movie</td><td>5</td></tr><tr><td>ACM</td><td>10,942</td><td>4</td><td>547,872</td><td>8</td><td>30%</td><td>paper</td><td>3</td></tr><tr><td>Freebase</td><td>180,098</td><td>8</td><td>1,057,688</td><td>36</td><td>30%</td><td>book</td><td>7</td></tr><tr><td>PubMed_NC</td><td>63,109</td><td>4</td><td>244,986</td><td>10</td><td>30%</td><td>disease</td><td>8</td></tr></table></body></html>

Baselines. We compare our method with several stateof-the-art models, including HAN (Wang et al. 2019b), MAGNN (Fu et al. 2020), HetGNN (Zhang et al. 2019), HGT (Hu et al. 2020), RGCN (Schlichtkrull et al. 2018), RSHN (Zhu et al. 2019), SimpleHGN (Lv et al. 2021), Space4HGNN (Zhao et al. 2022), Seq-HGNN(Du et al. 2023) and SlotGAT (Zhou et al. 2023). Additionally, we include comparisons with GCN (Kipf and Welling 2016) and GAT (Velicˇkovic´ et al. 2017).

Evaluation Settings. For node classification, following the methodology described by Lv et al. (2021), we split the labeled training set into training and validation subsets at $8 0 \% { : } 2 0 \%$ . The testing data are fixed, with detailed numbers provided in the appendix. For each dataset, we conduct experiments on five random splits, reporting the average and standard deviation of the results. For our methods, we perform a grid search to select the best hyperparameters on the validation set. The results of the node classification are reported using the averaged Macro-F1 and Micro-F1 scores.

# 5.2 Main Results

Table 1 presents the node classification results on a $30 \%$ noise data environment compared with peer methods. The comparison metrics are mean Macro-F1 and Micro-F1 scores. It can be observed that NoiseHGNN consistently outperforms all baselines for Macro-F1 and Micro-F1 on DBLP, IMDB, Pubmed, and Freebase. Overall, NoiseHGNN achieves an average $+ 4 . 2 \%$ improvement on Macro-F1 and $+ 2 . 6 \%$ on Micro-F1 compared with runner-up. In highlight, NoiseHGNN achieves the best results on IMDB and Pubmed datasets, with an average improvement of $+ 5 . 9 \%$ . This is because IMDB and Pubmed are noise-sensitive datasets since their node features are just one-hot embedding, which contains no semantics. All the information is stored in the topology of the graphs, which have $30 \%$ error links. In the medium-size dataset, DBLP, NoiseHGNN performs stable, with a $+ 3 . 5 \%$ improvement compared with the previous SOTA method. In the large and non-trivial dataset, Freebase, NoiseHGNN still obtained $3 0 . 5 7 \%$ Macro-F1, which is $2 . 4 \%$ higher than the previous SOTA method. Notably, NoiseHGNN gets the runner-up results $( 9 0 . 1 6 \% )$ in the ACM dataset, with $- 0 . 4 8 \%$ lower than SlotGAT $( 9 0 . 6 4 \% )$ . This is because ACM is a noise-insensitive and trivial dataset, while the latest methods can achieve $90 \%$ and even higher results. Its nodes contain rich semantic embedding that can be simply used for node classification without using graph links.

# 5.3 Ablation Study

In this section, we demonstrate the effectiveness of NoiseHGNN through ablation experiments with w/o Graph Synthesized and Meta-Target Graph. As shown in table3, NoiseHGNN with graph synthesized and Meta-target graph achieves superior results across all datasets. Specifically, on the smaller DBLP dataset, NoiseHGNN with the graph synthesizer and Meta-target graph improves Macro-F1 and Micro-F1 scores by $2 . 4 1 \%$ and $3 . 8 1 \%$ , respectively. On the largest dataset, Freebase, the improvements are $1 1 . 4 0 \%$ in Macro-F1 and $3 . 2 4 \%$ in Micro-F1. Notably, on the IMDB dataset, NoiseHGNN achieves significant improvements of $4 2 . 0 1 \%$ in Macro-F1 and $2 5 . 8 3 \%$ in Micro-F1. These results demonstrate that the higher-order synthesized graphs generated by the graph synthesizer effectively mitigate the negative effects of erroneous links, leading to substantially better performance.

Table 3: Ablation Study: w/o Graph Synthesized and MetaTarget Graph.   

<html><body><table><tr><td>Datasets</td><td></td><td>w/o Graph Synthesizer</td><td>w/o Meta-Target Graph</td><td>w/ All</td></tr><tr><td rowspan="2">DBLP</td><td></td><td>70.26±1.23</td><td>71.6±0.58</td><td></td></tr><tr><td>Macro-FI</td><td></td><td></td><td>71.96±0.3</td></tr><tr><td rowspan="2">IMDB</td><td></td><td>43.48±6.58</td><td>59.77±1.77</td><td></td></tr><tr><td>Macro-FI</td><td></td><td></td><td>60.96±0.69</td></tr><tr><td rowspan="2">ACM</td><td>Macro-Fl</td><td>89.40±0.92</td><td>89.32±0.61</td><td>90.16±0.49</td></tr><tr><td>Micro-Fl</td><td>89.34±1.09</td><td>89.42±0.60</td><td>90.10±0.46</td></tr><tr><td rowspan="2">Pubmed_NC</td><td>Macro-Fl</td><td>49.92±1.32</td><td>49.40±1.74</td><td>50.92±3.52</td></tr><tr><td>Micro-Fl</td><td>51.32±1.52</td><td>52.42±3.52</td><td>55.81±3.20</td></tr><tr><td rowspan="2">Freebase</td><td>Macro-Fl</td><td>27.44±1.96</td><td>28.16±2.47</td><td>30.57±2.94</td></tr><tr><td>Micro-Fl</td><td>55.14±1.33</td><td>56.15±0.34</td><td>56.96±0.67</td></tr></table></body></html>

# 5.4 Parameter Analysis

In this subsection, we explore the sensitivity of the hyperparameter $k$ in NoiseHGNN. The parameter $k$ determines the number of neighbor nodes to be used in the S high-order synthesized graph based on $\mathbf { k }$ -nearest neighbors (kNN). To understand its impact on our model’s performance, we search the number of neighbors $k$ in the range of $\{ 5 , 1 5 , 2 5 , 3 5 , 4 5 \}$ for all datasets. Our results indicate that selecting an appropriate $\mathbf { k }$ value can significantly enhance the accuracy of NoiseHGNN across various datasets. As is demonstrated in Fig. 3, the best selection for each dataset is different, i.e., $k \ = \ 1 5$ for Freebase and IMDB, $k \ = \ 2 5$ for Cora and PubMed NC, and $k = 3 5$ for ACM. It is commonly observed that selecting a value of $k$ that is either too large or too small can lead to suboptimal performance. We hypothesize that an excessively small $k$ may restrict the inclusion of beneficial neighbors, while an overly large $k$ might introduce noisy connections, thereby degrading the overall performance.

![](images/9e8a5d10e84bb795d0b34b3280222d81c908edbe88390f305320d2ac6dba7c7f.jpg)  
Figure 3: Effect of parameter $\mathbf { k }$ in top- $\mathbf { \cdot k } ( \cdot )$ selection in equation 4.

# 5.5 Visualization Analysis

To visually represent and compare the quality of the embeddings, Fig. 4 presents the t-SNE plot (Van der Maaten and Hinton 2008) of the node embeddings generated by NoiseHGNN on the DBLP dataset. Consistent with the quantitative results, the 2D projections of the embeddings learned by NoiseHGNN show more distinguishable clusters, both visually and numerically, compared to simpleHGN and SlotGAT. The Silhouette scores support this (Rousseeuw 1987), where NoiseHGNN achieves a score of 0.21 on DBLP, significantly higher than the scores of 0.12 for simpleHGN and 0.13 for SlotGAT.

![](images/d4ffa6f57968a7bafc40f3a23f2b21836b045010fd354b1854a0f77dbfd401b3.jpg)  
Figure 4: The t-SNE visualization of the graph representation. The proposed method achieves a good silhouette score (0.21).

# 6 Conclusion, and Future Work

This paper presents the first study addressing the problem of error link perturbation in heterogeneous graphs. To tackle this issue, we propose a novel method, NoiseHGNN, which effectively extracts valid structural information from the original data, thereby mitigating the negative impact of erroneous links. Extensive experiments on numerous real-world datasets demonstrate that the proposed method achieves state-of-the-art performance in noisy heterogeneous graph learning tasks. In highlights, $+ 5 \sim 6 \%$ improvements are observed in several noised datasets compared with previous SOTA methods.

In the future, we intend to explore the performance of heterogeneous graphs in complex environments. Specifically, we intend to address the problem of model failure in the presence of missing features, missing links, and erroneous link perturbations. We hope to advance the robustness and applicability of heterogeneous graph neural networks in more complex and uncertain real-world scenarios.

# Acknowledgments

This paper is the result of the research project funded by the National Natural Foundation of China (Grant No. 62106216 and 62162064) and the Open Foundation of Yunnan Key Laboratory of Software Engineering under Grant No.2023SE104.