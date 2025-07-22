# Hypergraph Learning for Unsupervised Graph Alignment via Optimal Transport

Yuguang Yan1, Canlin Yang1, Yuanlin Chen1, Ruichu $\mathbf { C a i } ^ { 1 , 2 * }$ , Michael $\mathbf { N g } ^ { 3 }$

1School of Computer Science, Guangdong University of Technology, Guangzhou, China 2Peng Cheng Laboratory, Shenzhen, China 3Department of Mathematics, Hong Kong Baptist University, Hong Kong, China ygyan $@$ gdut.edu.cn, {yangcl0608,chenyuanlin27,cairuichu}@gmail.com, michael-ng@hkbu.edu.hk

# Abstract

Unsupervised graph alignment aims to find corresponding nodes across different graphs without supervision. Existing methods usually leverage the graph structure to aggregate features of nodes to find relations between nodes. However, the graph structure is inherently limited in pairwise relations between nodes without considering higher-order dependencies among multiple nodes. In this paper, we take advantage of the hypergraph structure to characterize higher-order structural information among nodes for better graph alignment. Specifically, we propose an optimal transport model to learn a hypergraph to capture complex relations among nodes, so that the nodes involved in one hyperedge can be adaptively based on local geometric information. In addition, inspired by the Dirichlet energy function of a hypergraph, we further refine our model to enhance the consistency between structural and feature information in each hyperedge. After that, we jointly leverage graphs and hypergraphs to extract structural and feature information to better model the relations between nodes, which is used to find node correspondences across graphs. We conduct experiments on several benchmark datasets with different settings, and the results demonstrate the effectiveness of our proposed method.

# Introduction

Graph alignment aims to recognize node correspondences across different graphs (Wang et al. 2018; Heimann et al. 2018). As structured data become ubiquitous, graph alignment has received much attention and been widely used in real-world applications, such as finding the same users on different social network platforms (Li et al. 2019), recognizing the same entities in different knowledge graphs (Liu et al. 2022). In the problem of graph alignment with supervision, some node correspondences are given as ground-truth information (Liu et al. 2016). However, it is difficult to obtain supervised information since annotation is usually expensive and time-consuming. Therefore, unsupervised graph alignment without requiring supervision has drawn more attention in recent years (Chen et al. 2019).

Existing methods of unsupervised graph alignment usually leverage features of nodes and the graph structure to find the node correspondences across two graphs (Chen et al. 2019; Heimann et al. 2018; Li et al. 2019). The first paradigm is to learn a shared node embedding space for two graphs, so that the embeddings of nodes in different graphs can be compared, and the node correspondences can be found by nearest neighbor matching (Derr et al. 2021). However, it is non-trivial to learn a shared embedding space especially when the feature spaces of two graphs are diverse, making it challenging to compare the node embeddings in different graphs. To avoid directly comparing node embeddings across graphs, another paradigm based on optimal transport between metric spaces is proposed for unsupervised graph alignment (Li et al. 2022). Specifically, the Gromov-Wasserstein model is built on two graphs to find the optimal transport plan across graphs, where the transport plan reveals the probabilistic correspondences between nodes in different graphs (Peyre´, Cuturi, and Solomon 2016). The transport cost across graphs is measured by the difference between two metric matrices, each of which is constructed within one graph.

Although existing methods achieve encouraging performance for unsupervised graph alignment, they still suffer from the limitation of graph structure that only pairwise information between nodes is considered, while higher-order relations among multiple nodes are neglected (Ju et al. 2024). For example, a community consisting of multiple nodes contains complex dependencies among them, which cannot be captured by pairwise edges between two nodes. Compared with graphs in which one edge can connect only two nodes, the hypergraph has shown a powerful ability to characterize high-order relations among nodes, owing to the property that one hyperedge can involve multiple nodes (Gao et al. 2020). Figure 1 depicts the differences between the hypergraph and the graph representing a co-authored citation network. In the figure, a node represents a paper, and the edges in the graph can only describe the co-author relationship between the pairs of nodes, while the hyperedges in the hypergraph can contain all the papers that an author participated in. This property also endows the hypergraph with the ability to capture higher-order information.

Motivated by this, we seek to leverage hypergraphs to extract high-order structural information for unsupervised graph alignment, and design a method named Hypergraph Learning via Optimal Transport(HLOT). Specifically, we propose an optimal transport model within one graph to construct a weighted hypergraph, which can adaptively assign weights to the pairs of nodes and hyperedges, so that the different correlations can be captured. Moreover, inspired by the Dirichlet energy function of hypergraph that measures the consistency between structural and feature information, we refine our optimal transport model to derive a hypergraph learning model. Based on our constructed hypergraphs, we exploit structural and feature information on both graph and hypergraph levels, so that a better metric modeling the relations between nodes can be built for graph alignment. We conduct experiments on real-world datasets with different settings to evaluate the performance of our method, and empirically study the effects of graphs and hypergraphs.

![](images/b8579416f9e161004baea5565c43015cc9c66a59a413c0b64daf4feabb3df264.jpg)  
Figure 1: An example of using graph and hypergraph to represent a co-authorship citation network.

We summarize our principal contributions as follows:

• To characterize high-order structural information among nodes, we propose an optimal transport model to learn hypergraphs, in which the consistency between structural and feature information is enhanced by the Dirichlet energy function of the hypergraph.   
• To better exploit structural and feature information, we construct intra-graph metrics from the perspective of both graphs and hypergraphs.   
• To evaluate the performance of our method and the effects of graphs and hypergraphs, we conduct experiments and ablation studies on real-world data sets with different settings.

# Related Works

# Unsupervised Graph Alignment

Graph alignment is usually treated as a supervised learning task, relying on the known node correspondence to enhance prediction performance (Zhang et al. 2015, 2021; Fey et al. 2020; Yan, Zhang, and Tong 2021). However, obtaining these correspondences is often expensive and could be impractical, leading to the development of unsupervised graph alignment models. (Zhang and Tong 2016) is a pioneer study that proposed the consistency principle based on the similarity of nodes, which guides the evolution of unsupervised graph alignment algorithms. Existing methods of unsupervised graph alignment can be divided into two mainstream paradigms.

The first paradigm, known as “embed-then-crosscompare alignment”, first embeds the representation of nodes, and then determines the graph alignment based on similarity measurement. For example, REGAL (Heimann et al. 2018) uses matrix decomposition to generate node representations and match similar node pairs. WAlign (Gao, Huang, and Li 2021) employs a graph convolutional network (Kipf and Welling 2017) architecture to learn node embeddings, updating them incrementally using a Wasserstein distance discriminator. GCNAlign (Wang et al. 2018) integrates structural and attribute information through graph convolution networks, updating embeddings based on distance loss and identifying alignments via cross-graph similarity. GAlign (Trung et al. 2020) enhances graph structures by introducing perturbations and dynamically adjusting weights between high-confidence node pairs.

The second paradigm introduces the concept of GromovWasserstein distance (Me´moli 2011), transforming the graph alignment problem into an optimal transport problem between metric spaces. GWL $\mathrm { \Delta X u }$ et al. 2019a) applies optimal transport for graph alignment, using the graph adjacency matrix to represent the cost matrix. FusedGW (Titouan et al. 2019a) extends the Gromov-Wasserstein distance to incorporate both structural and feature information of the graphs. However, since aligned graphs are typically constructed from different sources, there usually exists an inconsistency in structure and features between the two graphs.

To address this, SLOTAlign (Tang et al. 2023) integrated multi-view structures containing rich information and introduced a learnable weight coefficient to balance the contributions of optimal transport from different views. Our work further incorporates the concept of hypergraphs into the unsupervised graph alignment framework based on optimal transport. By leveraging the hypergraph to model higherorder node relations, our model is better equipped to capture intricate complex structures and information, thereby improving both the alignment accuracy and robustness of the alignment process.

# Hypergraph

Hypergraph an extension of the traditional graph allows edges to connect an arbitrary number of nodes, thus capturing complex interactions and higher-order information among nodes. Hypergraphs have been effectively applied in various domains (Zhou, Huang, and Scho¨lkopf 2006), including citation networks (Jiang et al. 2019), social networks (Yang et al. 2019), medical data analysis (Di et al. 2021), and recommendation systems (Wang et al. 2020). To leverage the hypergraph structure to learn node embeddings, HGNN (Feng et al. 2019) generalizes convolutional operators from Graph Convolutional Networks to hypergraph structures, significantly enhancing the representational power of hypergraph models. Building on this, subsequent works have introduced novel hypergraph convolutional operators and aggregation methods, further advancing the representation learning of hypergraphs (Yan et al. 2024; Wu, Yan, and $\mathrm { N g }$ 2022; Dong, Sawin, and Bengio 2020).

In this work, we combine the optimal transport framework with the hypergraph energy function to construct the hypergraph with a finer granularity, and then discover the higherorder relationship between the source graph and the target graph, and further enhance the effectiveness and robustness of the model.

# Optimal Transport

Optimal transport (OT) (Villani 2008) is a mathematical framework designed to find an optimal transport plan that minimizes the cost of transferring a distribution from a source to a target, given their marginal distributions and a predefined cost function. Optimal transport has drawn significant attention for its applicability in various domains such as computer vision (Solomon et al. 2015), domain adaptation (Courty et al. 2016), node classification (Titouan et al. 2019b) and graph classification (Wang et al. 2024).

The Gromov-Wasserstein distance, originally designed to compare metric spaces by aligning their underlying structures, captures both the structural and distributional differences between two distributions (Me´moli 2011; Peyre´, Cuturi, and Solomon 2016). The Gromov-Wasserstein distance can identify the optimal mapping between nodes that preserves the structural consistency of the graphs, making it a natural fit for unsupervised graph alignment tasks (Tang et al. 2023). This insight has led to the reformulation of unsupervised graph alignment as an OT problem, thereby introducing the above optimal transport graph alignment paradigm without requiring any ground-truth node correspondence.

# Preliminary

Graph An undirected graph can be represented as $G =$ $( V , \mathbf { A } , \mathbf { X } )$ , where $V = \{ v _ { 1 } , v _ { 2 } , . . . , v _ { | V | } \}$ denotes the set of nodes in graph $G$ , $\mathbf { X } \in \mathbb { R } ^ { | V | \times d }$ is the node attribute matrix of the graph $G$ , with $d$ as the dimension of the attribute. The structure of the graph $G$ is represented by theadjacency matrix $\mathbf { A } \in \{ 0 , 1 \} ^ { \top \top } \mathbb { V } ^ { \top \top }$ , where $A _ { i j } = 1$ if there is an edge connecting υi and υj, otherwise Aij = 0. Let D ∈ R|V |×|V | be the degree matrix of the graph $G$ , where each diagonal value $\begin{array} { r } { D _ { i i } = \sum _ { j } A _ { i j } } \end{array}$ represents the number of neighbors of $\upsilon _ { i }$ .

Hypergraph A hypergraph is the generalization of the graph, where each hyperedge can be viewed as a set of nodes, Formally, a hypergraph can be denoted as $\mathcal { G } ^ { \mathrm { ~ ~ } } =$ $( \gamma , \mathcal { E } , \mathbf { X } , \mathbf { W } _ { e } )$ , where $\mathcal { V } = \{ v _ { 1 } , v _ { 2 } , . . . , v _ { | \mathcal { V } | } \}$ is the set of nodes. $\mathcal { E } = \{ e _ { 1 } , e _ { 2 } , . . . , e _ { | \mathcal { E } | } \}$ is the set of hyperedges, with each element $e$ being a subset of $\nu$ . The matrix $\mathbf { X } \in \mathbb { R } ^ { | \nu | \times d }$ represents the feature matrix of the nodes, where the feature of node $v _ { i }$ is represented $\mathbf { x } _ { i } \in \mathbb { R } ^ { d }$ which is the transpose of the $i$ -th row of $\mathbf { X }$ . The diagonal matrix $\mathbf { W } _ { e } \in \mathbb { R } ^ { | \mathcal { E } | \times \bar { | \mathcal { E } | } }$ represents the weight of the hyperedge, where the $j$ -th diagonal element $w _ { j }$ indicating the importance of the hyperedge $e _ { j }$ . In general, an incidence matrix $\mathbf { H } \in \mathbb { R } ^ { | \mathcal { V } | \times | \mathcal { E } | }$ can be used to represent the structure of the hypergraph $\mathcal { G }$ , where each element $H _ { i j }$ indicates the connection between node $v _ { i }$ and hyperedge $e _ { j }$ . Specifically, $H _ { i j }$ equals to 1 if the node $v _ { i }$ is contained in the hyperedge $e _ { j }$ , otherwise 0. Additionally, the diagonal matrices $\mathbf { D } _ { v } \in \mathbb { R } ^ { | \mathcal { V } | \times | \mathcal { V } | }$ and $\mathbf { D } _ { e } \in \mathbb { R } ^ { | \mathcal { E } | \times | \mathcal { E } | }$ are used to denote the degree of nodes and hyperedges, respectively, where the diagonal values of $\mathbf { D } _ { v }$ and $\mathbf { D } _ { e }$ are given by $\textstyle \sum _ { j } { \dot { w } } _ { j } H _ { i j }$ and $\textstyle \sum _ { i } { \bar { H } } _ { i j }$ , respectively.

# Problem Statement

In this study, we consider an unsupervised strategy to solve the alignment problem of two attribute graphs. Specifically, Given two attribute undirected graphs, source graph $G _ { s } ~ = ~ ( V _ { s } , A _ { s } , X _ { s } )$ and target graph $G _ { t } \ = \ ( V _ { t } , A _ { t } , X _ { t } ) ,$ , where $v _ { i } ^ { s }$ , $\ v _ { j } ^ { t }$ represent the $i$ -th node of the source graph $G _ { s }$ , and the $j$ -th node in target graph $G _ { t }$ , respectively, and $n _ { s }$ , $n _ { t }$ are the numbers of nodes in $G _ { s }$ and $G _ { t }$ , respectively. There is a naturally occurring set of alignment node pairs $\mathcal { M } = \{ ( v _ { i } ^ { s } , v _ { j } ^ { t } ) | ( v _ { i } ^ { \bar { s } } , v _ { j } ^ { t } ) \in V _ { s } ^ { \setminus } \times V _ { t } \}$ , where $v _ { i } ^ { s }$ and $\ v _ { j } ^ { t }$ represent the same node in two different graphs. The goal of the unsupervised alignment task is to find a mapping function $\pi ( v _ { i } ^ { \bar { s } } , v _ { j } ^ { t } )$ that accurately identifies the correspondence between nodes without using any observable ground-truth node correspondence.

# Methodology

Figure 2 illustrates the overview of our proposed method HLOT. We first learn hypergraphs for source and target graphs, respectively. After that, we leverage both hypergraphs and graphs to extract structural information and feature information, and then construct intra-graph metrics to characterize the relations between nodes. Finally, we find node correspondences between two graphs based on a probabilistic coupling matrix. In the following, we detailedly describe our proposed method.

# Hypergraph Level Learning

Hypergraph Generation To capture high-order relations among nodes, we construct hypergraphs for $G _ { s }$ and $G _ { t }$ by generating hyperedges that connect multiple nodes. Traditional methods usually find the nearest neighbors for a node to generate a hyperedge (Feng et al. 2019). However, it is heuristic to determine the number of nodes in a hyperedge. Different from them, we propose to employ a doubly stochastic matrix $\mathbf { H } \in \mathcal { H }$ to represent the hypergraph structure, where the domain $\mathcal { H }$ is defined as $\mathcal { H } = \{ \mathbf { H } ~ \in$ R|+V|×|E| | H1 = |V1| 1, H⊤1 = |E1| 1}. The constraints mean that the sums of all the rows (resp., columns) are equal, indicating that all the nodes (resp., hyperedge) have the same weight. Based on this, we propose the following model to adaptively find nearest neighbors and construct hyperedges

$$
\operatorname* { m i n } _ { \mathbf { H } \in \mathcal { H } } \mathbf { \langle C , H \rangle } + \epsilon \Omega ( \mathbf { H } ) ,
$$

where $C _ { i j }$ is the squared Euclidean distance between two nodes, i.e.,

$$
C _ { i j } = \| \mathbf { x } _ { i } - \mathbf { x } _ { j } \| _ { 2 } ^ { 2 } ,
$$

![](images/cf23c1ce1e37a42d5d7e89e1003cc1acfe687a927f75125b260e2b56c50958e9.jpg)  
Figure 2: The overview of our proposed method HLOT.

and $\Omega ( \mathbf { H } )$ is the negative entropy of $\mathbf { H }$ defined as

$$
\Omega ( \mathbf { H } ) = \sum _ { i j } H _ { i j } ( \log H _ { i j } - 1 ) ,
$$

which is used to smoothen the hypergraph structure $\mathbf { H }$ so that one hyperedge can involve multiple nodes. Since each hyperedge is built based on one node, $C _ { i i } = 0$ will induce the trivial solution that the diagonal elements $H _ { i i }$ dominate the others. To avoid this, we set $C _ { i i }$ as a sufficiently large value $\rho$ to construct a matrix $\tilde { \mathbf { C } } = \mathbf { C } + \rho \mathbf { I }$ where I is the identity matrix, so that sufficiently small $H _ { i i }$ will be induced.

Moreover, we refine the hypergraph structure from the perspective of the total variation (or the Dirichlet energy function) of the hypergraph, which can be used to analyze the properties of a hypergraph or used as a regularization term for hypergraph learning (Hein et al. 2013). By extending the total variation of a hypergraph with $0 / 1$ incidence matrix $\mathbf { H }$ in (Hein et al. 2013), we define the total variation of a hypergraph with a weighted incidence matrix $\mathbf { H } \in \mathbb { R } ^ { | \mathcal { V } | \times | \mathcal { E } | }$ as follows,

$$
\mathrm { T V } ( \mathbf { H } ) = \sum _ { e _ { k } \in \mathcal { E } } w _ { k } \operatorname* { m a x } _ { i , j \in e _ { k } } H _ { i k } H _ { j k } \| \mathbf { x } _ { i } - \mathbf { x } _ { j } \| _ { 2 } ^ { 2 } ,
$$

which is based on the node pair in one hyperedge with the largest distance. For two nodes with a strong correlation in one hyperedge (i.e., a large $H _ { i k } H _ { j k } )$ , they are expected to have similar features and a small distance. Therefore, a small $\mathrm { T V } ( \mathbf { H } )$ comes from the consistency between structural information in $\mathbf { H }$ and distance information on features $\mathbf { X }$ .

Motivated by this, we seek to learn a hypergraph structural $\mathbf { H }$ by minimizing the total variation of it. However, the maximum operation in Eq. (4) is non-differentiable and difficult to address in optimization. To tackle this, we instead design the following loss function to approximate Eq. (4)

$$
\Psi ( \mathbf { H } ) = \sum _ { k } w _ { k } \sum _ { i j } H _ { i k } H _ { j k } C _ { i j } = \mathrm { t r } ( \mathbf { W } _ { e } \mathbf { H } ^ { \top } \mathbf { C } \mathbf { H } ) ,
$$

where the weighted distances between nodes in one hyperedge are considered. By minimizing this, the consistency between structural information and feature information is enhanced.

Finally, we achieve the following optimization problem to learn the hypergraph structure

$$
\operatorname* { m i n } _ { \mathbf { H } \in \mathcal { H } } \mathbf { \pi } \langle \tilde { \mathbf { C } } , \mathbf { H } \rangle + \theta \Psi ( \mathbf { H } ) + \epsilon \Omega ( \mathbf { H } ) ,
$$

where $\theta$ and $\epsilon$ are the trade-off parameters.

We employ the projected gradient descent method to solve this problem (Peyre´, Cuturi, and Solomon 2016). The technical details are given in the appendix. After obtaining the weighted hypergraph $\mathbf { H }$ , we construct two views to extract structural and feature information, respectively.

Hypergraph Structure View First, we leverage the hypergraph structure to model the relations between nodes. To this end, we accumulate the incidence degrees of all the hyperedges that are shared by two nodes to measure the relations between these two nodes. Formally, the relation between nodes $v _ { i }$ and $v _ { j }$ is calculated as $\begin{array} { r } { M _ { i j } ^ { \bar { k _ { s } } } = \sum _ { k } H _ { i k } H _ { j k } } \end{array}$ and the intra-graph relation matrix between nodes is calculated as $\mathbf { M } ^ { h s ^ { \smile } } = \mathbf { H } \mathbf { H } ^ { \top }$ , where the superscript $h s$ means hypergraph structure view. Different from a classical $0 / 1$ adjacency matrix, the matrix $\mathbf { M } ^ { h s }$ contains continuous values, which reflect the strengths of the relations between two nodes.

Hypergraph Feature View Second, we leverage the hypergraph structure to aggregate the features of nodes and their neighbors. To achieve this, we apply the hypergraph convolutional operation (Feng et al. 2019) to obtain the embeddings of nodes, and the embeddings of the $( l { + } 1 )$ -th layer is updated as

$$
\mathbf { Z } _ { h } ^ { ( l + 1 ) } = \mathbf { D } _ { v } ^ { - \frac { 1 } { 2 } } \mathbf { H } \mathbf { W } _ { e } \mathbf { D } _ { e } ^ { - 1 } \mathbf { H } ^ { \top } \mathbf { D } _ { v } ^ { - \frac { 1 } { 2 } } \mathbf { Z } _ { h } ^ { ( l ) } ,
$$

where ${ \bf Z } _ { h } ^ { 0 } = { \bf X }$ is the input feature for the hypergraph, $\mathbf { D } _ { v }$ is the node degree matrix, $\mathbf { D } _ { e }$ is the edge degree matrix, and ${ \bf W } _ { e }$ is a diagonal matrix with diagonal elements indicating the weights of the hyperedges. In practice, to filter weak incidence in $\mathbf { H }$ and speed up the calculation, for each node, we adopt $\kappa$ hyperedges with the highest incidence values, and set the values of them as one while the others as zero. As a result, we obtain a sparse incidence matrix $\mathbf { H }$ with strong relations between nodes and hyperedge preserved.

# Algorithm 1: HLOT

Input: Source graph $G _ { s } ~ = ~ ( U _ { s } , A _ { s } , X _ { s } )$ , target graph $G _ { t } = ( V _ { t } , A _ { t } , X _ { t } )$ ,   
Initialize: $\begin{array} { r } { \alpha _ { i } ^ { s } = \alpha _ { i } ^ { t } = \frac { 1 } { K } } \end{array}$ , $\boldsymbol \alpha ^ { 1 } = [ \boldsymbol \alpha ^ { s } , \boldsymbol \alpha ^ { t } ]$ , $\begin{array} { r } { \pi _ { i j } ^ { 1 } = \frac { 1 } { n _ { s } n _ { t } } } \end{array}$ 1: Construct hypergraph $\mathbf { H }$ through solving Problem (6); 2: Construct candidate structure bases $\mathbf { M } ^ { s }$ and $\mathbf { M } ^ { t }$ ;   
3: repeat   
4: Update $\alpha ^ { k + 1 }$ by solving Problem (15);   
5: Update $\pi ^ { k + 1 }$ by solving Problem (16);   
6: until convergence   
7: Generate node pairs according to $\pi$ .

After obtaining the final embeddings ${ \mathbf Z } _ { h }$ , we construct the intra-graph relations between nodes based on the similarity metric, which is implemented by the inner product here, i.e., $\mathbf { M } ^ { h f } = \mathbf { Z } _ { h } \mathbf { Z } _ { h } ^ { \top }$ , where the superscript $h f$ means hypergraph feature view.

# Graph Level Learning

In this part, we construct intra-graph metric matrices based on the graph structure. Following the above approach for hypergraphs, two views are constructed to extract structural and feature information, respectively.

Graph Structure View The graph structural information is well captured by the adjacency matrix A. Therefore, we construct the graph structure view as $\mathbf { M } ^ { g s } = \mathbf { A }$ , where the superscript $g s$ means graph structure view.

Graph Feature View To leverage feature information of nodes by the graph structure, we apply a parameter-free graph convolutional operation (Wu et al. 2019) to aggregate features of nodes and their neighbors. Specifically, given the node embeddings of the $( l )$ -the layer, the embeddings are updated as

$$
\mathbf { Z } _ { g } ^ { ( l + 1 ) } = \mathbf { D } ^ { - \frac { 1 } { 2 } } \tilde { \mathbf { A } } \mathbf { D } ^ { - \frac { 1 } { 2 } } \mathbf { Z } _ { g } ^ { ( l ) } ,
$$

where $\mathbf { Z } _ { g } ^ { ( 0 ) }$ is the input features for the graph, $\mathbf { D }$ is the degree matrix of the graph, $\tilde { \mathbf { A } } = \mathbf { A } + \mathbf { I }$ is the adjacency matrix with self loops, where I is an identity matrix. After obtaining the embeddings $\mathbf { Z } _ { g }$ from GCN, we use a similarity metric to calculate node similarity: $\mathbf { M } ^ { g f } = \mathbf { Z } _ { g } \mathbf { Z } _ { g } ^ { \top }$ , where the superscript $g f$ means graph feature view.

# Graph Alignment

Given structural and feature views obtained from both hypergraphs and graphs, we are ready to combine them to construct the optimal representations of two graphs and find node correspondences between them. For simplicity, for the source graph $G _ { s }$ (resp., the target graph $G _ { t }$ ), we build matrices $\bar { \mathbf { M } } ^ { s , \bar { 1 } }$ , $\mathbf { M } ^ { s , 2 }$ , $\bar { \mathbf { M } } ^ { s , 3 }$ , $\mathbf { M } ^ { s , \bar { 4 } }$ (resp., $\mathbf { M } ^ { t , 1 }$ , $\mathbf { M } ^ { t , 2 }$ , $\mathbf { M } ^ { t , 3 }$ , $\mathbf { M } ^ { t , 4 }$ , ) based on $\mathbf { M } ^ { h s }$ , $\mathbf { M } ^ { h f }$ , $\mathbf { M } ^ { g s }$ , $\bar { \mathbf { M } } ^ { g f }$ , respectively. We construct intra-graph metric matrices by the convex combinations of the above views, i.e.,

$$
\mathbf { M } ^ { s } = \sum _ { i } \alpha _ { i } ^ { s } \mathbf { M } ^ { s , i } , \quad \mathbf { M } ^ { t } = \sum _ { i } \alpha _ { i } ^ { t } \mathbf { M } ^ { t , i } ,
$$

where the weight vectors $\alpha ^ { s }$ and $\alpha ^ { t }$ are in the following domain

$$
\Sigma _ { K } = \{ \mathbf { a } \in \mathbb { R } ^ { K } | \mathbf { a } ^ { \top } \mathbf { 1 } = 1 , a _ { i } \in [ 0 , 1 ] \forall i \} .
$$

Following (Chen et al. 2020; $\mathrm { { X u } }$ et al. 2019b) we use a probabilistic coupling matrix $\pi$ to indicate the node correspondences between these two graphs, and apply the Gromov-Wasserstein model to learn the optimal coupling $\pi$ . Here, $\pi _ { i j }$ represents the matching degree between $v _ { i } ^ { s }$ and $\boldsymbol { v } _ { j } ^ { t }$ , and $\pi$ is in the following domain

$$
\Pi = \{ \pi \in \mathbb { R } _ { + } ^ { n _ { s } \times n _ { t } } \mid \pi { \bf 1 } = \frac { 1 } { n _ { s } } { \bf 1 } , \pi ^ { \top } { \bf 1 } = \frac { 1 } { n _ { t } } { \bf 1 } \} ,
$$

and the Gromov-Wasserstein model aims to minimize the following objective function

$$
F ( \pi , \alpha ^ { s } , \alpha ^ { t } ) = \sum _ { i , i ^ { \prime } , j , j ^ { \prime } } L _ { i , i ^ { \prime } , j , j ^ { \prime } } \cdot \pi _ { i j } \cdot \pi _ { i ^ { \prime } j ^ { \prime } } ,
$$

where the loss tensor is defined as

$$
L _ { i , i ^ { \prime } , j , j ^ { \prime } } = \frac { 1 } { 2 } ( M _ { i i ^ { \prime } } ^ { s } - M _ { j j ^ { \prime } } ^ { t } ) ^ { 2 } ,
$$

which measures the transport cost between node pairs $( v _ { i } ^ { s } , v _ { i ^ { \prime } } ^ { s } )$ and $( v _ { j } ^ { t } , v _ { j ^ { \prime } } ^ { t } )$ .

Combining the learning of the weight vectors $\alpha ^ { s }$ and $\alpha ^ { t }$ , we solve the following optimization problem to find the optimal weight vectors and node correspondences jointly

$$
\begin{array} { r l } & { \underset { \pi , \alpha ^ { s } , \alpha ^ { t } } { \operatorname* { m i n } } F ( \pi , \alpha ^ { s } , \alpha ^ { t } ) } \\ & { \quad \quad \mathrm { s . t . } \pi \in \Pi , \alpha ^ { s } \in \Sigma _ { K } , \alpha ^ { t } \in \Sigma _ { K } . } \end{array}
$$

We alternately update two blocks of the variables, i.e., $\pi$ and $\boldsymbol { \alpha } = [ \alpha ^ { s } ; \alpha ^ { \dot { t } } ]$ . At the $\tau$ -th iteration, for the weight variables $\alpha$ , we follow (Tang et al. 2023) to apply a projected proximal gradient method to find $\alpha \in \Sigma _ { 2 K }$ that minimizes the linearized problem, i.e.,

$$
\boldsymbol { \alpha } ^ { \tau + 1 } = \arg \operatorname* { m i n } _ { \boldsymbol { \alpha } \in \Sigma _ { 2 K } } \langle \nabla _ { \boldsymbol { \alpha } } F ( \boldsymbol { \pi } ^ { \tau } , \boldsymbol { \alpha } ^ { \tau } ) , \boldsymbol { \alpha } \rangle + \lambda \| \boldsymbol { \alpha } - \boldsymbol { \alpha } ^ { \tau } \| _ { 2 } ^ { 2 } ,
$$

where $\nabla _ { \alpha } F ( \pi ^ { \tau } , \alpha ^ { \tau } )$ is the gradient of $F ( \pi ^ { \tau } , \alpha ^ { \tau } )$ with respect to $\alpha$ .

For the probabilistic coupling matrix $\pi$ , to handle the doubly stochastic constraint defined in $\Pi$ , we apply a projected

Table 1: Statistical information of the datasets.   

<html><body><table><tr><td>Data</td><td>Type</td><td>#Node</td><td>#Edges</td><td># Attr.</td></tr><tr><td>Cora</td><td>Citation Network</td><td>2708</td><td>5028</td><td>1433</td></tr><tr><td>Citeseer</td><td>Citation Network</td><td>3327</td><td>4732</td><td>3703</td></tr><tr><td>PPI</td><td>Protein Interaction</td><td>1767</td><td>16159</td><td>171</td></tr><tr><td>Facebook</td><td>Social Network</td><td>4039</td><td>44117</td><td>1476</td></tr><tr><td>ACM</td><td>Co-Author Network</td><td>9872</td><td>39561</td><td>17</td></tr><tr><td>DBLP</td><td>Co-Author Network</td><td>9916</td><td>44808</td><td>17</td></tr><tr><td>Douban -Online</td><td>Social Network</td><td>3906</td><td>16328</td><td>538</td></tr><tr><td>Douban -Offline</td><td>Social Network</td><td>1118</td><td>3022</td><td>538</td></tr></table></body></html>

<html><body><table><tr><td></td><td colspan="4">Douban Online-Offline</td><td colspan="4">ACM-DBLP</td></tr><tr><td>Model</td><td>Hit@1</td><td>Hit@5</td><td>Hit@10</td><td>Hit@30</td><td>Hit@1</td><td>Hit@5</td><td>Hit@10</td><td>Hit@30</td></tr><tr><td>KNN</td><td>3.31</td><td>10.38</td><td>16.64</td><td>30.05</td><td>49.25</td><td>59.46</td><td>63.42</td><td>69.61</td></tr><tr><td>REGAL</td><td>30.32</td><td>54.83</td><td></td><td></td><td>34.09</td><td>46.58</td><td>51.35</td><td>56.34</td></tr><tr><td>GCNAlign</td><td>20.93</td><td>34.44</td><td>39.62</td><td>50.72</td><td>38.43</td><td>68.46</td><td>77.64</td><td>86.89</td></tr><tr><td>GATAlign</td><td>23.70</td><td>36.94</td><td>44.01</td><td>57.16</td><td>14.21</td><td>34.07</td><td>42.12</td><td>49.00</td></tr><tr><td>WAlign</td><td>35.69</td><td>57.87</td><td>69.05</td><td>83.09</td><td>50.61</td><td>72.87</td><td>80.84</td><td>89.47</td></tr><tr><td>GWD</td><td>3.04</td><td>7.96</td><td>9.21</td><td>11.90</td><td>56.24</td><td>77.14</td><td>82.20</td><td>84.92</td></tr><tr><td>FusedGW</td><td>29.61</td><td>62.79</td><td>66.46</td><td>68.07</td><td>30.80</td><td>38.39</td><td>39.26</td><td>39.60</td></tr><tr><td>SLOTAlign</td><td>51.43</td><td>73.43</td><td>77.73</td><td>82.02</td><td>66.04</td><td>84.06</td><td>87.95</td><td>90.32</td></tr><tr><td>HLOT</td><td>57.33</td><td>79.70</td><td>84.17</td><td>87.12</td><td>65.94</td><td>84.70</td><td>88.76</td><td>91.32</td></tr></table></body></html>

Table 2: Hit $@ \mathbf { k }$ results on two real-world graph alignment datasets.

proximal gradient method with the Kullback-Leibler (KL) divergence, i.e.,

$$
\pi ^ { \tau + 1 } = \arg \operatorname* { m i n } _ { \pi \in \Pi } \langle \nabla _ { \pi } F ( \pi ^ { \tau } , \alpha ^ { \tau + 1 } ) , \pi \rangle + \gamma \mathbf { K } \mathbf { L } ( \pi | \pi ^ { \tau } ) ,
$$

where the KL divergence is defined as

$$
\mathrm { K L } ( \pi | \pi ^ { \tau } ) \stackrel { \mathrm { d e f . } } { = } \sum _ { i j } \pi _ { i j } \log \left( \frac { \pi _ { i j } } { \pi _ { i j } ^ { \tau } } \right) - \pi _ { i j } + \pi _ { i j } ^ { \tau } .
$$

As a result, the Sinkhorn algorithm (Peyre´, Cuturi, and Solomon 2016) can be employed to efficiently update $\pi$ . Algorithm 1 summarizes our proposed method. The complexity analysis is given in the appendix.

# Experiments

# Datasets

We list the statistical information of the datasets in Table 1, and describe the datasets in the following.

• Douban (Tang et al. 2023) is a social network dataset and contains online and offline graphs. Nodes and edges have the same meaning as the Facebook dataset. The location information of a user is used as node features for the Douban dataset.   
• ACM-DBLP (Tang et al. 2023) is a co-author network dataset and contains two graphs extracted from the publication information in four research areas. It consists of a collection of papers and their authors. The node features of ACM-DBLP are bag-of-words vector representations.   
• Citeseer (Sen et al. 2008) is a co-citation dataset. It consists of the collection of papers and their citation links. The node features of Citeseer are bag-of-words vector representations.   
• PPI (Zitnik and Leskovec 2017) is a Protein-Protein interaction network dataset. The nodes represent the proteins in the network and the edges represent the ProteinProtein interactions.   
• Cora (Sen et al. 2008) is a citation network dataset. The node features of Cora are based on the term frequencyinverse document frequency (TF-IDF).

• Facebook (Leskovec and Mcauley 2012) is a social network dataset. The nodes represent users in a social network and edges represent the correspondence between users. Facebook employs user profile information as the node features.

Among the above six datasets, the first two are real-world graph alignment datasets, and the other four are semisynthetic datasets used to evaluate the graph alignment methods. The detailed settings are described in the following parts.

# Compared Methods

KNN: This method directly matches nodes with their closest $K$ nodes as candidates in feature space. REGAL (Heimann et al. 2018): It is a graph alignment method based on embedding that can be used for graphs with or without node features. WAlign (Gao, Huang, and Li 2021): This method uses a lightweight Graph Convolutional Network with a Wasserstein distance discriminator to obtain a matching matrix between the source and target graphs. GCNAlign (Wang et al. 2018): This method uses a graph convolutional network to calculate node embeddings for both the source and target graphs. GATAlign (Velickovic et al. 2017): This method is similar to GCNAlign, while a Graph Attention Network is used to obtain node embeddings for source and target graphs. GWD (Xu et al. 2019a): This method uses the Gromov-Wasserstien distance for graph alignment, which only pays attention to structural information by using the graph adjacency matrices. FuesdGW (Titouan et al. 2019b): FuesdGW contains both the Wasserstein distance and the Gromov-Wasserstein distance. It takes into account both structural and feature information for graph alignment. SLOTAlign (Tang et al. 2023): SLOTAlign proposes a multi-view structure to learn graph representations and reduce the effect of structural and feature inconsistency.

# Experiment Settings

For our proposed method, we employ a one-layer parameterfree graph convolutional network and one one-layer parameter-free HGNN network for all datasets except ACMDBLP. Since ACM-DBLP has more nodes and edges than other datasets, we use two-layer parameter-free GCN and two-layer parameter-free HGNN to capture information.

Table $3 \colon \mathrm { H i t } @ 1$ results of four disturbance datasets.   

<html><body><table><tr><td></td><td colspan="4">Permutation</td><td colspan="4">Compression</td></tr><tr><td>Model</td><td>Cora</td><td>Citeseer</td><td>PPI</td><td>Facebook</td><td>Cora</td><td>Citeseer</td><td>PPI</td><td>Facebook</td></tr><tr><td>KNN</td><td>89.55(0.027)</td><td>57.44(0.019)</td><td>6(0.003)</td><td>20.85(0.042)</td><td>0(0.004)</td><td>0.03(0.004)</td><td>0.06(0.004)</td><td>0.02(0.002)</td></tr><tr><td>GWD</td><td>50.37(0.001)</td><td>14.52(0.002)</td><td>84.95(0.003)</td><td>70.17(0.002)</td><td>50.37(0.02)</td><td>14.52(0.014)</td><td>84.95(0.32)</td><td>70.17(0.25)</td></tr><tr><td>FusedGW</td><td>97.45(0.002)</td><td>99.58(0.002)</td><td>78.53(0.002)</td><td>64.17(0.002)</td><td>0.85(0.004)</td><td>0.57(0.001)</td><td>0.85(0.003)</td><td>0.87(0.002)</td></tr><tr><td>SLOTAlign</td><td>99.25(0.001)</td><td>99.25(0.002)</td><td>89.64(0.001)</td><td>98.71(0.003)</td><td>98.45(0.027)</td><td>99.76(0.02)</td><td>86.47(0.3)</td><td>98.61(0.043)</td></tr><tr><td>HLOT</td><td>99.56(0.004)</td><td>99.73(0.003)</td><td>89.98(0.005)</td><td>99.46(0.004)</td><td>98.56(0.003)</td><td>99.73(0.007)</td><td>87.44(0)</td><td>98.71(0.002)</td></tr></table></body></html>

Table 4: Hit $@ \mathbf { k }$ results of ablation studies.   

<html><body><table><tr><td></td><td colspan="3">Douban Online-Offline</td></tr><tr><td>Model</td><td>Hit@1</td><td>Hit@5</td><td>Hit@10</td><td>Hit@30</td></tr><tr><td>HLOT-w/o Mh s</td><td>45.35</td><td>63.42</td><td>67.62</td><td>72.18</td></tr><tr><td>HLOT-w/o Mh f</td><td>52.95</td><td>77.91</td><td>81.84</td><td>85.87</td></tr><tr><td>HLOT-w/o Mgs</td><td>1.07</td><td>5.1</td><td>8.77</td><td>19.32</td></tr><tr><td>HLOT-w/o Mgf</td><td>47.14</td><td>64.4</td><td>68.25</td><td>73.08</td></tr><tr><td>HLOT</td><td>57.33</td><td>79.7</td><td>84.17</td><td>87.12</td></tr></table></body></html>

Following (Tang et al. 2023), for each experiment, we tune the hyperparameters $\theta$ in the range $\{ 0 . 0 0 1 , 0 . 0 1 , 0 . 1 , 0 . 2 , \bar { 0 . 5 } , \bar { 1 } \}$ , $\epsilon$ in the range $\{ 0 . 0 0 1 , 0 . 0 1 , 0 . 1 , 1 , 1 0 , 1 0 0 \}$ , and select the highest $\mathrm { H i t } @ \mathrm { k }$ value that performs best as a result. For the updating of $\alpha ^ { s } , \alpha ^ { t }$ , we train the learnable weight for 500 epochs by gradient descent for all the datasets except that the epochs of ACM-DBLP are set to 1000.

We run all the methods on the Pytorch platform and the experiments are conducted on a Linux server with an NVIDIA RTX 4090 (24GB) graphics card. The running time results are reported in the appendix.

Evaluation Metrics We adopt $\operatorname { H i t } @ \operatorname { k }$ to evaluate the performance of the graph alignment methods. This metric calculates the percentage of nodes in $\mathcal { V } _ { t }$ whose ground-truth alignment results in $\mathcal { V } _ { s }$ are among the top- $\mathbf { \nabla } \cdot \mathbf { k }$ candidates.

# Results and Discussions

For fair comparisons, we strictly follow the experimental setting in (Tang et al. 2023), in which we directly align two graphs in noisy real-world graph alignment datasets to test the performance of graph alignment of our method. We align the source graph with the target graph. Table 2 presents the results of graph alignment in terms of the $\operatorname { H i t } @ \operatorname { k }$ , which shows that our method outperforms the other methods. Compared with the state-of-the-art method SLOTAlign, HLOT has $5 . 1 \%$ improvement in $\mathrm { H i t } @ 3 0$ on the Douban data , and $1 . 0 \%$ improvement in $\mathrm { H i t } @ 3 0$ on the ACM-DBLP data. It verifies that high-order information involved in hypergraphs is helpful for graph alignment.

We also use two kinds of disturbing ways to disturb the source graph to obtain the target graph, which are used for graph alignment. The first type is a permutation, meaning we randomly select $p \%$ feature columns and then rearrange the feature columns at random. The second type is compression. We use some dimensionality reduction methods such as PCA to reduce dimension to $( 1 0 0 - p ) \%$ . We set the $p = 5 0$ in the following experiments. Table 3 presents the results, where the results of the mean and standard derivation of 10 times are reported. Our method also outperforms other methods and maintains a high performance of all disturbance settings for all datasets. It means that our method can steadily perform hypergraph learning and graph learning in the case of feature disturbance. We also conduct experiments by randomly removing $10 \%$ nodes from the source graph and report the results in the appendix.

# Ablation Studies

To investigate the impacts of structural and feature information of hypergraphs and graphs, we conduct ablation studies on Douban and report the $\operatorname { H i t } @ \operatorname { k }$ results of different variants of our method in Table 4, in which each variant removes a view. We observe that the results of Douban drop significantly after the graph structure view is removed, which is consistent with the observation in (Tang et al. 2023) for SLOTAlign, indicating that the structural information ${ { \bf { M } } _ { g s } }$ is important to the datasets. Compared with the variants without the hypergraph level information, i.e., without $\mathbf { M } _ { h s }$ or $\mathbf { M } _ { h f }$ , HLOT achieves better performance, which verifies that hypergraphs are beneficial for extracting high-order information for better graph alignment. We also test a version that replaces the hypergraph view with a similarity metric, and show our results in the appendix.

# Conclusion

In this paper, we propose to learn hypergraphs to capture high-order information for unsupervised graph alignment. We apply an optimal transport model to learn hypergraphs, which are refined by the total variation function of the hypergraph for structure and feature consistency. Both hypergraphs and graphs are applied to model the relations between nodes to find the node correspondences across graphs. We conduct experiments with different settings to demonstrate the effectiveness of our proposed method.

# Acknowledgments

This work was supported in part by the National Science and Technology Major Project (2021ZD0111501), National Natural Science Foundation of China (62206061, U24A20233), National Science Fund for Excellent Young Scholars (62122022), Guangdong Basic and Applied Basic Research Foundation (2024A1515011901), and Guangzhou Basic and Applied Basic Research Foundation (2023A04J1700). The work of Michael $\mathrm { N g }$ was supported in part by the Hong Kong Research Grant Council GRF (17201020, 17300021), C7004-21GF, and Joint NSFC-RGC (N-HKU76921).