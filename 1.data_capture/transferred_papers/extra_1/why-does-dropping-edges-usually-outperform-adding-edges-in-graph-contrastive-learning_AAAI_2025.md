# Why Does Dropping Edges Usually Outperform Adding Edges in Graph Contrastive Learning?

Yanchen $\mathbf { X } \mathbf { u } ^ { 1 , 2 * }$ , Siqi Huang1,2\*, Hongyuan Zhang2,3†, Xuelong $\mathbf { L i } ^ { 2 \dagger }$

1 School of Artificial Intelligence, OPtics and ElectroNics (iOPEN), Northwestern Polytechnical University 2Institute of Artificial Intelligence (TeleAI), China Telecom 3The University of Hong Kong {yanchenxu.tj, 4777huang, hyzhang98}@gmail.com, xuelong li@ieee.org

# Abstract

Graph contrastive learning (GCL) has been widely used as an effective self-supervised learning method for graph representation learning. However, how to apply adequate and stable graph augmentation to generating proper views for contrastive learning remains an essential problem. Dropping edges is a primary augmentation in GCL while adding edges is not a common method due to its unstable performance. To our best knowledge, there is no theoretical analysis to study why dropping edges usually outperforms adding edges. To answer this question, we introduce a new metric, namely Error Passing Rate (EPR), to quantify how a graph fits the network. Inspired by the theoretical conclusions and the idea of positive-incentive noise, we propose a novel GCL algorithm, Error-PAssing-based Graph Contrastive Learning (EPAGCL), which uses both edge adding and edge dropping as its augmentations. To be specific, we generate views by adding and dropping edges based on the weights derived from EPR. Extensive experiments on various real-world datasets are conducted to validate the correctness of our theoretical analysis and the effectiveness of our proposed algorithm.

# Code — https://github.com/hyzhang98/EPAGCL Extended version — https://arxiv.org/abs/2412.08128

# 1 Introduction

Graph contrastive learning (GCL) has been a hot research topic in graph representation learning (Zhu et al. 2021; Thakoor et al. 2022; Zhang et al. 2022; Zhang, Zhu, and Li 2024). It originates from contrastive learning (Chen et al. 2020b; He et al. 2020; Gao, Yao, and Chen 2021; Qu et al. 2021; Radford et al. 2021), which generates different augmented views and maximizes the similarity between positive pairs (van den Oord, Li, and Vinyals 2018). In particular, compared with CL, it has been shown that how to generate proper graph augmentation views is a crucial problem in GCL (Tian et al. 2020; Wu et al. 2020). Due to the particularity of the structure of graph data, the reliable unsupervised data augmentation schemes are greatly limited. Inspired by Dropout (Srivastava et al. 2014), many researchers chose to

Training Time Training Time {199s, 371s} {26s,35s} Dropping Dropping Adding Adding   
Acc. Acc. Acc. Acc.   
Mean std. Mean std.   
{78.23, {0.11, {60.75, {0.22,   
79.77} 0.14} 68.81} 0.38} Memory Cost Memory Cost {6.07G,8.53G] {1.09G, 1.25G}   
(a) Performance on WikiCS (b) Performance on CiteSeer

randomly mask the features as the attribute augmentations. Moreover, for many GCL methods (Zhu et al. 2020; Thakoor et al. 2022; You et al. 2020), the most popular scheme is to change the graph topology, most of which randomly drops edges to generate views. Apart from random edge-dropping (Rong et al. 2020), researchers have tried various methods to generate stable views. MVGRL (Hassani and Khasahmadi 2020) introduced diffusion kernels for augmentation. GCA (Zhu et al. 2021) augmented the graph based on the importance of edges. And GCS (Wei et al. 2023) adaptively screened the semantic-related substructure in graphs for contrastive learning. Although various works have been done to investigate augmentation schemes for GCL, edge adding, as a simple way of graph topology augmentation, is only used in supervised learning (Zhao et al. 2021; Chen et al. 2020a). Why is edge dropping a widely accepted way for GCL other than edge adding? Can edge adding be modified to achieve similar effect as edge dropping? In this paper, we attempt to answer the above questions theoretically.

To begin with, a simple experiment is conducted on WikiCS (Mernyei and Cangea 2020) and CiteSeer (Sen et al. 2008) to compare the two strategies of augmentations. We use the same hyper-parameters and employ Adding Edges or Dropping Edges as the only augmentation, respectively. The result is shown in Figure 1. It is easy to conclude the disadvantage of edge adding in all respects.

Generally speaking, there are two main challenges for edge adding as augmentation:

1 For most graphs, the number of edges takes only a small portion of all node pairs, which means that there are much more choices for edge-adding than edge-dropping. Hence, the memory and time burden of edge-adding are significant.   
$\mathcal { C } 2$ Without labels, we have no idea about the influence of the edge perturbation. In that case, a suitable metric is needed to measure it.

To deal with the challenges above, we introduce a metric of graph, namely Error Passing Rate (EPR), to quantify the fitness between the graph and Graph Neural Network (GNN). To be specific, EPR measures how much message is wrongly passed during the aggregation in GNNs. For a certain graph, a lower EPR means that the aggregation is more efficient, which is what we expect. Consequently, the aim of augmentation can be converted to generating a view with low EPR.

In this paper, through mathematical derivation, we prove that the degree attribute of two nodes can measure the effect on EPR of adding or dropping the edge between them. To maintain a relatively low EPR, it is important to ensure that the EPR of the graph will not increase too much even if an edge is wrongly added or dropped. To this end, for stability, node pairs that correspond to low-level effect will be chosen to add or drop the edges between them, as a solution to $\mathcal { C } 2$ . As a result, the nodes can be pre-screened to reduce the memory and time burden, which resolves $\mathcal { C } 1$ .

Inspired by our theoretical analysis, we propose a novel adaptive augmentation framework for GCL, where a graph is augmented through selective edge perturbation and random feature mask. For edge perturbation, the possibility that an edge is added to or dropped from the graph is based on the magnitude of its effect on EPR. Briefly speaking, the possibility is high if the magnitude of effect is relatively low. This helps to maintain the EPR of augmented views in a low level. The augmentation is then equipped with an InfoNCElike objective (van den Oord, Li, and Vinyals 2018; Zhu et al. 2020) for contrastive learning, namely Error-PAssing-ratebased Graph Contrastive Learning (EPAGCL).

# 2 Related Works

# 2.1 Graph Contrastive Learning

Various graph contrastive learning methods have been proposed (Gao et al. 2021; Zhang et al. 2021; Liu et al. 2023) in recent years. In general, Graph Contrastive Learning (GCL) methods aims to learn representations by contrasting positive and negative samples.

The researchers mainly focus on the contrastive objective. For the sake of simplicity, random edge dropping and feature masking ars widely used (Rong et al. 2020; Zhu et al. 2020; Thakoor et al. 2022). Furthermore, GraphCL (You et al. 2020) and InfoGCL (Xu et al. 2021) employ two more augmentations, node dropping and subgraph sampling, which change the node attribute and structure property of the graph at the same time. There are also some methods using their own augmentation methods. MVGRL (Hassani and Khasahmadi 2020) generates views through graph diffusion. SimGCL (Yu et al. 2022) adds uniform noise to the embedding space to create contrastive views.

# 2.2 Adaptive Data Augmentations for Graph

Some researchers have investigated methods for adaptive data augmentations (Li 2022; Zhang et al. 2024) for GCL. JOAO (You et al. 2021) adaptively and dynamically selects data augmentations for specific data. GCA (Zhu et al. 2021) sets the probability of dropping edges and masking features according to their importance. ADGCL (Suresh et al. 2021) optimizes adversarial graph augmentation strategies for contrastive learning. GCS (Wei et al. 2023) screens substructure in graphs with the help of a gradient-based graph contrastive saliency. NCLA (Shen et al. 2023) learns graph augmentations by multi-head graph attention mechanism.

Our proposed augmentation method is similar to GCA. However, edges that EPAGCL tends to add or drop are likely to be preserved in GCA. What’s more, our theory is rigorously derived quantitatively, and our augmentation scheme is proved to be more stable.

# 2.3 Adding Edges as Augmentation

Adding edges is also used as augmentation for graph data is some supervised learning methods. AdaEdge (Chen et al. 2020a) and GAUG (Zhao et al. 2021) both add edges based on prediction metrics, which cannot be used in selfsupervised learning.

Both methods tend to minimize the negative impact of the augmentation, which is the same as our proposed framework. To be specific, both of them reduce the probability of error augmentation, while EPAGCL, inspired by pi-noise (a theoretical framework to learn beneficial noise) (Li 2022), reduces the magnitude of the impact if an error occurs.

# 3 The Proposed Method

In this section, we first introduce the formal definition of Error Passing Rate (EPR) to quantify how a graph fits the GNN. Then, the augmentations of graph topology are interpreted based on EPR. Finally, the Error-PAssing-based Graph Contrastive Learning (EPAGCL) method induced by the theoretical analysis is elaborated. The Framework is illustrated in Figure 2.

# 3.1 Preliminaries

Let $\mathcal { G } = ( \nu , \mathcal { E } )$ denote an undirected graph without isolated points, whose node set is $\mathcal { V } = \{ v _ { 1 } , v _ { 2 } , \bar { \cdot } \cdot \bar { \cdot } , v _ { N } \}$ and edge set is $\mathcal { E } \subseteq \mathcal { V } \times \mathcal { V }$ . The adjacency matrix of $\mathcal { G }$ is denoted as $A \in$ $\mathbb { R } ^ { N \times N }$ with $A _ { i j } = 1$ if $( v _ { i } , v _ { j } ) \in \mathcal { E }$ and $A _ { i j } = 0$ otherwise. The degree matrix of $\mathcal { G }$ , denoted as $D$ , is a diagonal matrix with $\begin{array} { r } { D _ { i i } = \sum _ { j } A _ { i j } } \end{array}$ . Furthermore, let $\stackrel { \sim } { D } = D + \stackrel { \sim } { I }$ , ${ \widetilde { A } } = A +$ $I$ , where $I$ s an $N$ -dimensional ident ey matrix. $\forall v \in \mathcal { V }$ , the set of its neighbours in graph $\mathcal { G }$ is denoted as $N _ { v }$ with $| N _ { v } | =$ $d _ { v } = D _ { v v }$ . Let $d _ { m i n } = \operatorname* { m i n } _ { v \in V } d _ { v }$ , $d _ { m a x } = \operatorname* { m a x } _ { v \in V } d _ { v }$ .

For $v _ { i } , v _ { j } \ \in \ \mathcal { V } \ s . t . \ ( v _ { i } , v _ { j } ) \ \notin \ \mathcal { E }$ , suppose that adding edges $( v _ { i } , \bar { v _ { j } } )$ to graph $\mathcal { G }$ yields graph $\mathcal { G } ^ { \prime } = ( \nu , \mathcal { E } ^ { \prime } )$ , where ${ \mathcal { E } } ^ { \prime } = { \mathcal { E } } \cup \{ ( v _ { i } , v _ { j } ) \}$ . Relatively, $A ^ { \prime } , D ^ { \prime } , \widetilde { A } ^ { \prime }$ , and ${ \widetilde { D } } ^ { \prime }$ of graph $\mathcal { G } ^ { \prime }$ are defined in the same way as abov

![](images/107732cf70e32e82c37cf26f98ea081a506a2b4b537f308238b037dd52855af3.jpg)  
Figure 2: Framework of EPAGCL: Before training, the weight of all existing edges and candidate edge for adding is computed according to the graph structure. We then generate two views adaptively based on the weights. Specifically, we add edges to and drop edges from the graph to obtain one view while drop edges only from the graph to obtain another. A random feature mask is then employed. After that, the two views are fed to a shared Graph Neural Network (GNN) with a projection head for representation learning. The model is trained with a contrastive objective.

# 3.2 Error Passing Rate

As message passing is a popular paradigm for GNNs, we introduce Error Passing Rate based on message passing mechanisms to measure how the graph fits the network:

Definition 3.1. For a graph $\mathcal { G }$ , the Error Passing Rate (EPR) $r _ { \mathcal G }$ denotes the ratio of the amount of message wrongly passed in the network, i.e.,

$$
r _ { \mathcal { G } } = M _ { w p } / M ,
$$

where $M _ { w p }$ is the amount of message wrongly passed, while $M$ is the amount of all the message passed.

Note that $r _ { \mathcal G }$ will change with different network structure. In this paper, Graph Convolutional Networks (GCN) (Kipf and Welling 2017) are taken as the target. The feed forward propagation in GCN can be recursivly conducted as

$$
H ^ { ( l + 1 ) } = \sum ( \widetilde { D } ^ { - 1 / 2 } \widetilde { A } \widetilde { D } ^ { - 1 / 2 } H ^ { ( l ) } W ^ { ( l ) } ) ,
$$

$H ^ { ( l + 1 ) } = \{ h _ { 1 } ^ { ( l + 1 ) } , \cdots , h _ { N } ^ { ( l + 1 ) } \}$ , h(l+1)} is the hidden vector of the (l + 1)-th layer with hi(l) as the hidden feature of node $\boldsymbol { v } _ { i }$ , and $\sum ( \cdot )$ is a nonlinear function. Eq. (1) suggests that the topological structure of the graph is represented as $\widetilde { D } ^ { - 1 / 2 } \widetilde { A } \widetilde { D } ^ { - 1 / 2 } \overset { \triangle } { = } \hat { A }$ in the network, corresponding to th emessage -epassing mechanism. Specifically, the amount of message passed from node $\boldsymbol { v } _ { i }$ to $v _ { j }$ can be measured by $\hat { A } _ { i j }$ . Thus, we can obtain the representation of $r _ { \mathcal G }$ .

Proposition 3.2. Given an undirected graph $\mathcal { G }$ , its $\mathtt { E P R } r _ { \mathcal G }$ in GCN can be formulated as

$$
r _ { \mathcal { G } } = \sum _ { ( v _ { i } , v _ { j } ) \in E } \hat { A } _ { i j } / \sum _ { ( v _ { i } , v _ { j } ) \in \mathcal { E } } \hat { A } _ { i j } ,
$$

where $E$ denotes the set of edges in $\mathcal { E }$ that links nodes of different, underlying classes which are agnostical during the self-supervised training, namely error edge set.

# 3.3 Why is Adding Edges Usually Worse?

An ordinary idea is to persue a smaller $r _ { \mathcal G }$ . This naturally leads to a question: how will $r _ { \mathcal G }$ change after adding $( v _ { i } , v _ { j } )$ into graph $\mathcal { G }$ as an augmentation?

It is complex to calculate the change of $r _ { \mathcal G }$ with Eq. (2). To simplify the calculation, we group the edges according to their relationship with $\boldsymbol { v } _ { i }$ and $v _ { j }$ , between which no edge lies. To be specific, for graph $\mathcal { G }$ , we divide $\mathcal { E }$ into three parts:

$$
\begin{array} { r l } & { \mathcal { E } _ { 1 } = \{ ( u , v ) | u , v \in V \backslash \{ v _ { i } , v _ { j } \} \} \cap \mathcal { E } , } \\ & { \mathcal { E } _ { 2 } = \{ ( v _ { i } , w ) | w \in V \} \cap \mathcal { E } , } \\ & { \mathcal { E } _ { 3 } = \{ ( v _ { j } , w ) | w \in V \} \cap \mathcal { E } . } \end{array}
$$

Note that $\mathcal { E } _ { 1 } \cup \mathcal { E } _ { 2 } \cup \mathcal { E } _ { 3 } = \mathcal { E }$ . Similarly, the error edge set $E$ is also divided into three parts, $E _ { 1 } , E _ { 2 }$ , and $E _ { 3 }$ . As a result, $\mathcal { E } _ { 1 }$ and $E _ { 1 }$ are sets of edges whose endpoints are not $v _ { i }$ or $v _ { j } ; \mathcal { E } _ { 2 }$ and $E _ { 2 }$ are edge sets with $v _ { i }$ as an endpoint; $\mathcal { E } _ { 3 }$ and $E _ { 3 }$ are edge sets with $v _ { j }$ as an endpoint.

As for graph $\mathcal { G } ^ { \prime }$ which yields from adding edge $( v _ { i } , v _ { j } )$ to $\mathcal { G }$ , the split is completed in the same way: $\mathcal { E } ^ { \prime } = \mathcal { E } _ { 1 } \cup \mathcal { E } _ { 2 } \cup \dot { \mathcal { E } } _ { 3 } \cup$ $\mathcal { E } _ { i j }$ , $E ^ { \prime } = E _ { 1 } \cup E _ { 2 } \cup E _ { 3 } \cup E _ { i j }$ . Among them, $\mathcal { E } _ { i j } = \{ ( v _ { i } , v _ { j } ) \}$ $\dot { E } _ { i j } = \mathcal { E } _ { i j }$ if $v _ { i }$ and $v _ { j }$ are of different classes and $E _ { i j } = \mathcal { D }$ if they are of the same class.

Owing to the split, the $\textstyle \sum _ { ( i , j ) } { \hat { A } } _ { i j }$ in Eq. (2) can be simplified. To begin with, denote the normalized adjacency matrix of $\mathcal { G } ^ { \prime }$ , $\widetilde { D } ^ { \prime - 1 / 2 } \widetilde { A } ^ { \prime } \widetilde { D } ^ { \prime - 1 / 2 }$ as ${ \hat { A } } ^ { \prime }$ . Furthermore, define

$$
m _ { l } = \sum _ { ( v _ { i } , v _ { j } ) \in { \mathcal E } _ { l } } \hat { A } _ { i j } , e _ { l } = \sum _ { ( v _ { i } , v _ { j } ) \in E _ { l } } \hat { A } _ { i j } , l = 1 , 2 , 3 ,
$$

and

$$
m _ { l } ^ { \prime } = \sum _ { ( v _ { i } , v _ { j } ) \in { \mathscr { E } } _ { l } } \hat { A } _ { i j } ^ { \prime } , e _ { l } ^ { \prime } = \sum _ { ( v _ { i } , v _ { j } ) \in E _ { l } } \hat { A } _ { i j } ^ { \prime } , l = 1 , 2 , 3 .
$$

Thus, $r _ { \mathcal G }$ and $r _ { \mathcal { G } ^ { \prime } }$ can be rewritten as

$$
r _ { \mathcal { G } } = \frac { \sum _ { i = 1 } ^ { 3 } e _ { i } } { \sum _ { i = 1 } ^ { 3 } m _ { i } } , r _ { \mathcal { G } ^ { \prime } } = \frac { \sum _ { i = 1 } ^ { 3 } e _ { i } ^ { \prime } + \sum _ { ( v _ { i } , v _ { j } ) \in E _ { i j } } \hat { A } _ { i j } ^ { \prime } } { \sum _ { i = 1 } ^ { 3 } m _ { i } ^ { \prime } + \sum _ { ( v _ { i } , v _ { j } ) \in E _ { i j } } \hat { A } _ { i j } ^ { \prime } } .
$$

Considering the change in node degrees in graph $\mathcal { G }$ after adding edge $( v _ { i } , v _ { j } )$ to it, it is easy to know that each node except $v _ { i }$ and $v _ { j }$ maintains its own degree while the degrees of $v _ { i }$ and $v _ { j }$ are increased by 1, indicating that

$$
\left\{ \begin{array} { l l } { m _ { 1 } ^ { \prime } = m _ { 1 } , } \\ { \displaystyle m _ { 2 } ^ { \prime } = \sqrt { \frac { d _ { i } } { d _ { i } + 1 } } m _ { 2 } , } \\ { m _ { 3 } ^ { \prime } = \sqrt { \frac { d _ { j } } { d _ { j } + 1 } } m _ { 3 } , } \\ { \displaystyle \hat { A } _ { i j } ^ { \prime } = \hat { A } _ { j i } ^ { \prime } = \frac { 1 } { \sqrt { ( d _ { i } + 1 ) ( d _ { j } + 1 ) } } . } \end{array} \right.
$$

As a result, $r _ { \mathcal G }$ and $r _ { \mathcal { G } ^ { \prime } }$ can be formulated as

$$
\begin{array} { l } { r _ { \mathcal { G } } = \frac { e _ { 1 } + e _ { 2 } + e _ { 3 } } { m _ { 1 } + m _ { 2 } + m _ { 3 } } , } \\ { r _ { \mathcal { G } ^ { \prime } } = \frac { e _ { 1 } + \sqrt { \frac { d _ { i } } { d _ { i } + 1 } } e _ { 2 } + \sqrt { \frac { d _ { j } } { d _ { j } + 1 } } e _ { 3 } + \xi \cdot \frac { 2 } { \sqrt { ( d _ { i } + 1 ) ( d _ { j } + 1 ) } } } { m _ { 1 } + \sqrt { \frac { d _ { i } } { d _ { i } + 1 } } m _ { 2 } + \sqrt { \frac { d _ { j } } { d _ { j } + 1 } } m _ { 3 } + \frac { 2 } { \sqrt { ( d _ { i } + 1 ) ( d _ { j } + 1 ) } } } , } \end{array}
$$

where $\xi = 0$ if $v _ { i }$ and $v _ { j }$ are of the same class and $\xi = 1$ otherwise.

Let m = Pi3=1 mi, m′ = Pi3=1 m′i + √(d +12)(d +1) , which are the sum of elements in $\hat { A }$ and ${ \hat { A } } ^ { \prime }$ , respectively. To compare $r _ { \mathcal G }$ with $r _ { \mathcal { G } ^ { \prime } }$ , we introduce

$$
\delta _ { \mathcal { G } , \mathcal { G } ^ { \prime } } = \boldsymbol { m } \cdot \boldsymbol { m } ^ { \prime } \cdot ( \boldsymbol { r } _ { \mathcal { G } } - \boldsymbol { r } _ { \mathcal { G } ^ { \prime } } ) .
$$

As $\delta _ { \mathcal { G } , \mathcal { G } ^ { \prime } } \propto \left( r _ { \mathcal { G } } - r _ { \mathcal { G } ^ { \prime } } \right)$ , it can effectively reflects the trends of changes in EPR. With a simple constraint, $\delta _ { { \mathcal G } , { \mathcal G } ^ { \prime } }$ can be related to the edge added, which is formally summarzied as Theorem 3.3.

Theorem 3.3. Given a graph $\mathcal { G }$ with $d _ { m a x } \leq 4 d _ { m i n } - 1$ and $d _ { m i n } \geq 1$ , then:

1. If $\mathcal { G } ^ { \prime }$ yields from adding an edge between two same-class nodes to $\mathcal { G }$ (i.e., $\xi = 0 .$ ), then $\delta _ { \mathcal { G } , \mathcal { G } ^ { \prime } } > 0$ .

2. If $\mathcal { G } ^ { \prime }$ yields from adding an edge between two differentclass nodes to $\mathcal { G }$ (i.e., $\xi = 1 \mathrm { \AA }$ ), then $\delta _ { \mathcal { G } , \mathcal { G } ^ { \prime } } < 0$ .

The proof of Theorem 3.3 is in Appendix A.

Theorem 3.3 indicates that adding edge between sameclass nodes decreases EPR of the graph, while adding edge between different-class nodes leads to the opposite result. The conclusion also meets our expectation of EPR and edge adding as well.

Moreover, motivating by the proof of Theorem 3.3.2, an additional assumption is used to precisely quantify $\delta _ { { \mathcal G } , { \mathcal G } ^ { \prime } }$ . With the assumption, we can further compare the effect of edge adding and dropping.

Assumption 3.4. For each node in the graph $\mathcal { G }$ , constantly $k$ of all the message passed in is error message, i.e.

$$
M _ { w p , i } / M _ { i } = k , \forall i \in \{ 1 , 2 , \cdots , N \} ,
$$

where $M _ { w p , i }$ is the amount of message wrongly passed to $v _ { i }$ , while $M _ { i }$ is the amount of all the message passed to $v _ { i }$ .

With Assumption 3.4, it can be derived that

$$
\delta _ { \mathcal { G } , \mathcal { G } ^ { \prime } } = ( k - \xi ) \cdot \alpha _ { i , j } \cdot m ,
$$

where $\alpha _ { i , j } = 2 / \sqrt { ( d _ { i } + 1 ) ( d _ { j } + 1 ) }$ . The detailed derivation is in Appendix B.

Now we apply the above analysis to edge dropping to answer the question raised in the title of this subsection. Specifically, we may consider $\mathcal { G } ^ { \prime }$ as the original graph and $\mathcal { G }$ as the graph augmented by dropping edge $( v _ { i } , v _ { j } )$ . Hence, for edge dropping, it holds that

$$
\delta _ { \mathcal { G } ^ { \prime } , \mathcal { G } } = ( \xi - k ) \cdot \alpha _ { i , j } \cdot m .
$$

The detailed derivation is in Appendix C. Hence, the following conclusions are drawn from Eq. (4), Eq. (5), and Eq. (6):

Theorem 3.5. Under Assumption 3.4, if graph $\mathcal { G }$ is obtained through adding edge $( v _ { i } , v _ { j } )$ to $\mathcal { G }$ , it holds that

$$
{ \left\{ \begin{array} { l l } { \displaystyle r g \prime - r _ { \mathscr { G } } = { \frac { k } { m ^ { \prime } } } \cdot \alpha _ { i , j } } & { { \mathrm { i f ~ } } c ( v _ { i } ) = c ( v _ { j } ) , } \\ { \displaystyle r _ { \mathscr { G } ^ { \prime } } - r _ { \mathscr { G } } = { \frac { 1 - k } { m ^ { \prime } } } \cdot \alpha _ { i , j } } & { { \mathrm { i f ~ } } c ( v _ { i } ) \neq c ( v _ { j } ) , } \end{array} \right. }
$$

where $c ( v )$ denotes the class of node $v$ .

For most graphs, the EPR is less than 0.5, i.e., $k < 0 . 5$ , which explains why dropping edges is more stable than adding edges for most of the time. What’s more, as mentioned in Section 2.2, GCA tends to drop edges with low importance, which corresponds to a higher $\alpha _ { i , j }$ and leads to a more unstable result according to Theorem 3.5.

# 3.4 Adding Edges with Retaining EPR

As shown in Section 3.3, adding edges is usually worse than dropping edges. Since the edge dropping may not work when the graph is too sparse, a direct question is that whether edge adding can work like edge dropping? According to Theorem 3.5, for dropping edge, the change of EPR $\propto \alpha _ { i , j }$ , while for adding edges, the change of EPR $\propto \alpha _ { i , j } / m ^ { \prime } \overset { \sim } { \approx } \alpha _ { i , j } / m \propto \alpha _ { i , j }$ .

Following the theoretical conclusion, we propose ErrorPAssing-based Graph Contrastive Learning (EPAGCL),

Algorithm 1: Algorithm to select added edges

Input: Vertex set $\nu$ , edge set $\mathcal { E }$ .   
Output: Drop-edge weight $w ^ { d }$ , add-edge weight $w ^ { a }$ , set of edges to be added $\mathcal { E } _ { a }$ .   
1: $l \gets | \mathcal { E } |$   
2: for $e _ { i , j }$ in $\mathcal { E }$ do   
3: $w _ { i j } ^ { d }  \alpha _ { i , j }$ ; drop // $\alpha _ { i , j }$ of $( \nu , \mathcal { E } )$ as $\mathcal { G } ^ { \prime }$ . 4: end for   
5: $\nu _ { a } \gets$ vertex in $\nu$ of top $\sqrt { 2 l }$ degrees   
6: $\mathcal { E } _ { a }  \mathcal { V } _ { a } \times \mathcal { V } _ { a } - \mathcal { E }$ $\begin{array} { r } { \operatorname { \mu } / / l \leq | \mathcal { E } _ { a } | \leq 2 l . } \end{array}$ 7: for $\boldsymbol { e } _ { i , j }$ in $\mathcal { E } _ { a }$ do   
8: wiaj ← αi,j; add // $\alpha _ { i , j }$ of $( \nu , \mathcal { E } )$ as $\mathcal { G }$ . 9: end for   
10: return $w ^ { d }$ , $w ^ { a }$ , $\mathcal { E } _ { a }$

which generates views for GCL based on the $\alpha _ { i , j }$ corresponding to each edge, ensuring that the EPR of the graph will not increase too much even if the edge is wrongly added or dropped.

To be specific, edges are added or dropped by sampling from the corresponding probability (Zhu et al. 2020). The edge set $\widetilde { \mathcal { E } }$ of the generated view can be formulated as $\widetilde { \mathcal { E } } =$ ${ \widetilde E } \cup { \widetilde E } ^ { \prime }$ , weith probability

$$
P \{ ( v _ { i } , v _ { j } ) \in \widetilde { E } \} = 1 - p _ { i j } ^ { d }
$$

and

$$
P \{ ( v _ { i } , v _ { j } ) \in \widetilde { E } ^ { \prime } \} = p _ { i j } ^ { a } ,
$$

where $\widetilde { E }$ is a subset of the original edge set $\mathcal { E }$ and ${ \widetilde { E } } ^ { \prime }$ is a subset ef the to-be-added edge set $\mathcal { E } _ { a \cdot } p _ { i j } ^ { a }$ and $p _ { i j } ^ { d }$ s end for the probability of dropping and adding $( v _ { i } , v _ { j } )$ respectively. Algorithm 1 shows how to get the to-be-added edge set $\mathcal { E } _ { a }$ along with the weights $w _ { d }$ and $w _ { a }$ . Note that for edges to be dropped, $\alpha _ { i , j } = 2 / \sqrt { d _ { i } d _ { j } }$ , while for edges to be added, $\alpha _ { i , j } ~ = ~ 2 / \sqrt { ( d _ { i } + 1 ) ( d _ { j } + 1 ) }$ (refer to Appendix C). The weights are then transformed into probability through a normalization step (Zhu et al. 2021).

$$
\left\{ \begin{array} { l l } { p _ { i j } ^ { a } = \operatorname* { m i n } ( \frac { \operatorname* { m a x } ( w ^ { a } ) - w _ { i j } ^ { a } } { \operatorname* { m a x } ( w ^ { a } ) - \mu _ { w ^ { a } } } \cdot p _ { \mathrm { a d d } } , \ p _ { \tau } ) , } \\ { p _ { i j } ^ { d } = \operatorname* { m i n } ( \frac { w _ { i j } ^ { d } - \operatorname* { m i n } ( w ^ { d } ) } { \mu _ { w ^ { d } } - \operatorname* { m i n } ( w ^ { d } ) } \cdot p _ { \mathrm { d r o p } } , \ p _ { \tau } ^ { \prime } ) . } \end{array} \right.
$$

In Eq. (9), $p _ { \mathrm { a d d } }$ and $p _ { \mathrm { d r o p } }$ are hyper-parameters that controls the overall probability. $\dot { p } _ { \tau } , p _ { \tau } ^ { \prime }$ are cut-off probability that is no greater than 1. $\mu _ { w ^ { a } }$ and $\mu _ { w ^ { d } }$ stand for the average of $w ^ { a }$ and $\boldsymbol { w } ^ { d }$ , respectively.

Note that the weights is obtained based on the original graph, they will be computed only once, which adds almost nothing to the burden. As for graphs with a significant number of nodes, thanks to the nodes filtering steps (line 5 in Algorithm 1), the computation is greatly accelerated and can be finished within an acceptable timeframe.

The training algorithm is summarized as pseudo-code in Algorithm 2. As Theorem 3.5 shows, dropping edges is a more stable way to generate augmented views, so it is used when generating both the views, while adding edges is used only for generating one views. Other than edge perturbation, random feature mask, which is widely used in graph presentation learning (Zhu et al. 2020) (Hassani and Khasahmadi 2020), is also employed. After the views are generated, an InfoNCE-like objective (van den Oord, Li, and Vinyals 2018) is employed. For each positive pair $( u _ { i } , v _ { i } )$ in $G _ { 1 } , G _ { 2 }$ , which is the embedding corresponds the same node of the original graph, we define $s ( u _ { i } , v _ { i } )$ as the cosine similarity of $g ( u _ { i } )$ and $g ( v _ { i } )$ , where $g ( \cdot )$ is a projection head (Tschannen et al. 2020), and

<html><body><table><tr><td>Algorithm 2:The EPAGCL trainingalgorithm</td></tr><tr><td>Input:Original graph G =（V,E) with feature X,weights wa,wd,to-be-added edge set εa. 1: for epoch ← 1,2,.  do 2: Obtain Pij, Pij according to wa, wd through Eq. (9). 3: Obtain E1, E2 according to pdj,E through Eq. (7). 4: Obtain E' according to pij,εa through Eq. (8).</td></tr></table></body></html>

$$
\begin{array} { l } { \displaystyle { l ( u _ { i } , v _ { i } ) = } } \\ { \displaystyle { \log \frac { e ^ { s ( u _ { i } , v _ { i } ) / \tau } } { e ^ { s ( u _ { i } , v _ { i } ) / \tau } + \sum _ { i \neq j } e ^ { s ( u _ { i } , v _ { j } ) / \tau } + \sum _ { i \neq j } e ^ { s ( u _ { i } , u _ { j } ) / \tau } } , } } \end{array}
$$

where $\tau$ is a temperature parameter. The contrastive loss is then computed by added up $l ( u _ { i } , v _ { i } )$ and $l ( v _ { i } , u _ { i } )$ for all $i \in$ $\{ 1 , 2 , \cdots , N \}$ .

# 4 Experiments

In this section, we perform experiments to investigate the following questions:

Q1 Does EPAGCL outperforms the existing baseline methods on node classification?   
Q2 What is the time and memory burden of EPAGCL?   
Q3 How does each part of the proposed augmentation strategy affect the effectiveness of training?

Table 1: Statistics of datasets used in experiments.   

<html><body><table><tr><td>Dataset</td><td>Nodes</td><td>Edges</td><td>Features</td><td>Classes</td></tr><tr><td>Cora</td><td>2,708</td><td>5,278</td><td>1,433</td><td>7</td></tr><tr><td>CiteSeer</td><td>3,327</td><td>4,552</td><td>3,703</td><td>6</td></tr><tr><td>PubMed</td><td>19,717</td><td>44,324</td><td>500</td><td>3</td></tr><tr><td>WikiCS</td><td>11,701</td><td>216,123</td><td>300</td><td>10</td></tr><tr><td>Amazon-Photo</td><td>7,650</td><td>119,081</td><td>745</td><td>8</td></tr><tr><td>Coauthor-Physics</td><td>34,493</td><td>247,962</td><td>8,415</td><td>5</td></tr><tr><td>Ogbn-Arxiv</td><td>169,343</td><td>1,166,243</td><td>128</td><td>40</td></tr></table></body></html>

Table 2: Results in terms of classification accuracy (in percent $\pm$ standard deviation) on seven datasets. \* means that we do not employ feature mask to augment graph data. OOM indicates Out-Of-Memory on a 40GB GPU. The best and runner-up results of self-supervised methods on each dataset are highlighted with bold and underline, respectively.   

<html><body><table><tr><td>Method</td><td>Cora</td><td>CiteSeer</td><td>PubMed</td><td>WikiCS</td><td>Amz. Photo</td><td>Co. Physics</td><td>ogbn-arxiv</td><td>Rank</td></tr><tr><td>GCN</td><td>84.15 ± 0.31</td><td>72.00± 0.39</td><td>85.39 ± 0.28</td><td>79.93 ± 0.53</td><td>92.69± 0.26</td><td>95.14 ± 0.41</td><td>69.44± 0.06</td><td>=</td></tr><tr><td>GAT</td><td>84.08 ± 0.51</td><td>72.17 ± 0.39</td><td>84.91 ± 0.20</td><td>80.84 ± 0.21</td><td>92.39 ± 0.30</td><td>95.48 ± 0.14</td><td>0OM</td><td>=</td></tr><tr><td>DGI</td><td>82.47± 0.38</td><td>71.03 ± 0.88</td><td>85.64± 0.39</td><td>79.43± 0.27</td><td>91.00 ±0.37</td><td>0OM</td><td>0OM</td><td>7.8</td></tr><tr><td>GMI</td><td>82.90 ± 0.69</td><td>69.51 ± 0.72</td><td>83.37 ± 0.52</td><td>80.23 ± 0.29</td><td>90.88 ± 0.31</td><td>OOM</td><td>OOM</td><td>8.6</td></tr><tr><td>MVGRL</td><td>81.46 ± 0.40</td><td>70.78 ± 0.53</td><td>84.33 ± 0.25</td><td>80.10 ± 0.26</td><td>90.91 ± 0.68</td><td>94.91 ± 0.16</td><td>0OM</td><td>8.2</td></tr><tr><td>GRACE</td><td>85.34 ± 0.29</td><td>71.66 ± 0.35</td><td>86.74 ± 0.18</td><td>80.65 ± 0.20</td><td>93.13 ± 0.12</td><td>95.63 ± 0.05</td><td>68.49 ± 0.01</td><td>3.6</td></tr><tr><td>GCA*</td><td>84.55 ± 0.43</td><td>70.81 ± 0.57</td><td>86.48 ± 0.17</td><td>81.45 ± 0.15</td><td>93.08 ±0.18</td><td>95.20 ± 0.09</td><td>69.26 ± 0.01</td><td>4.6</td></tr><tr><td>GCA</td><td>85.59 ± 0.35</td><td>71.21 ± 0.55</td><td>86.60 ± 0.19</td><td>79.18 ± 0.34</td><td>93.19 ± 0.24</td><td>95.28 ± 0.19</td><td>69.18 ± 0.01</td><td>4.7</td></tr><tr><td>BGRL</td><td>84.34 ± 0.36</td><td>70.02 ± 0.70</td><td>85.88 ± 0.14</td><td>80.43 ± 0.47</td><td>92.78 ± 0.65</td><td>95.56 ± 0.07</td><td>68.80 ±0.10</td><td>6.1</td></tr><tr><td>GREET</td><td>80.42 ± 0.25</td><td>71.48 ± 0.99</td><td>86.27 ± 0.32</td><td>79.91 ± 0.50</td><td>93.56 ± 0.14</td><td>96.06 ± 0.11</td><td>OOM</td><td>5.0</td></tr><tr><td>EPAGCL *</td><td>85.04 ± 0.33</td><td>71.97 ± 0.62</td><td>86.72 ± 0.11</td><td>81.81 ± 0.18</td><td>93.05 ±0.23</td><td>95.41 ± 0.03</td><td>69.29 ± 0.01</td><td>3.0</td></tr><tr><td>EPAGCL</td><td>86.07 ± 0.32</td><td>71.94 ± 0.57</td><td>86.77 ± 0.14</td><td>81.19 ± 0.11</td><td>93.42 ± 0.12</td><td>95.87 ± 0.04</td><td>69.25 ± 0.01</td><td>2.0</td></tr></table></body></html>

# 4.1 Datasets

Seven benchmark graph datasets are utilized for experimental study, including three citation network Cora, CiteSeer and PubMed (Sen et al. 2008), a reference network WikiCS (Mernyei and Cangea 2020), a co-purchase network Amazon-Photo (Shchur et al. 2018), a co-authorship network Coauthor-Physics (Shchur et al. 2018) and a largescale citation network ogbn-arxiv (Hu et al. 2020). The details of the datasets are summarized in Table 1.

# 4.2 Experimental Settings

Backbone For our proposed method, a two-layer GCN network (Kipf and Welling 2017) with PReLU activation is applied. The dimension of the hidden layer is 512 and the dimension of the final embedding is set as 256. Also, we employ a projection head, which consists of a 256- dimension fully connected layer with ReLU activation and a 256-dimension linear layer. The hyper-parameters of the training vary for different datasets, the details of which are shown in Appendix D.

Baselines We compare EPAGCL with two groups of baseline methods: (1) semi-supervised learning methods (i.e., GCN (Kipf and Welling 2017) and GAT (Velicˇkovic´ et al. 2018)); (2) contrastive learning methods (i.e., DGI (Velicˇkovic´ et al. 2019), GMI (Peng et al. 2020), MVGRL (Hassani and Khasahmadi 2020), GRACE (Zhu et al. 2020), GCA (Zhu et al. 2021), BGRL (Thakoor et al. 2022), and GREET (Liu et al. 2023)).

Evaluation Settings To evaluate the proposed method, we follow the standard linear evaluation scheme introduced in (Velicˇkovic´ et al. 2019). Firstly, each model is trained in an unsupervised manner. The resulting embeddings are then utilized to train and test a simple $l _ { 2 }$ -regularized logistic regression classifier, which is initialized randomly and trained with an Adam SGD optimizer (Kingma and Ba 2015) for 3000 epochs. The learning rate and weight decay factor of the optimizer are fixed to 0.01 and 0.0 respectively.

For each method on each dataset, the accuracy is averaged over 5 runs. For each run, the dataset is randomly split, where $10 \%$ , $10 \%$ and the rest $80 \%$ of the nodes are selected for the training, validation and test set, respectively.

The experiments are conducted on an NVIDIA A100 GPU with 40 GB memory.

# 4.3 Performance Analysis (Q1)

The classification accuracy is shown in Table 2 with a comparative rank. Specifically, despite the perturb target is different, our edge perturbation method is similar to GCA, while the feature mask is randomly applied. Further experiments are conducted for comparison. In the table, \* means that we do not employ feature mask as one of the augmentations on graph data.

It’s easy to find out that EPAGCL achieves better performance than baselines on almost every dataset. For instance, on Cora, our method achieves a $0 . 4 8 \%$ accuracy gain than the next best approach. What’s more, with the same contrastive objective, EPAGCL outperforms GCA by an average $0 . 6 1 \%$ increase, which indicates that our augmentation method is more effective. Thirdly, our method shows a lower standard deviation than GCA on most datasets, revealing its stability. This also verifies the inference from Theorem 3.5 that dropping edges corresponding to higher $\alpha _ { i , j }$ will lead to a more unstable result.

In additional, Figure 3 displays t-SNE (Van der Maaten and Hinton 2008) plots of the raw feature and learned embeddings on Cora and CiteSeer.

# 4.4 Efficiency Analysis (Q2)

To illustrate the efficiency of our model, we compare our method with other graph contrastive methods in terms of data pre-processing time, flops and training time of one epoch, and memory costs. MVGRL (Hassani and Khasahmadi 2020) is a method that makes use of graph diffusion for contrast, which is kind of similar to edge adding. GCA (Zhu et al. 2021) calculates different drop possibilities for edge dropping. And GREET (Liu et al. 2023) extracts information from feature and structure to benefit training. Experiments are conducted on Cora, PubMed, and ogbn-arxiv, corresponding to small, big and huge datasets. On ogbn-arxiv,

![](images/cb5a5a042bf67e988e0e7ef89bf42ceec224a580fb1bb2596213ee4ed79cbd64.jpg)

Figure 3: t-SNE embeddings of the raw features and learned embeddings obtained through EPAGCL on Cora and CiteSeer.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="4">Cora</td><td colspan="4">PubMed</td><td colspan="4">ogbn-arxiv</td></tr><tr><td>Proc.</td><td>FLOPs</td><td>Time</td><td>Mem.</td><td>Proc.</td><td>FLOPs</td><td>Time</td><td>Mem.</td><td>Proc.</td><td>FLOPs</td><td>Time</td><td>Mem.</td></tr><tr><td>MVGRL</td><td>0.18s</td><td>5.55e6</td><td>0.03s</td><td>3.49G</td><td>15.88s</td><td>4.04e7</td><td>0.16s</td><td>21.25G</td><td>-</td><td>1</td><td>1</td><td></td></tr><tr><td>GCA</td><td>0.40s</td><td>3.57e8</td><td>0.04s</td><td>0.93G</td><td>0.14s</td><td>2.60e9</td><td>0.42s</td><td>13.87G</td><td>0.13s</td><td>2.23e10</td><td>3.89s</td><td>18.36G</td></tr><tr><td>GREET</td><td>4.46s</td><td>5.90e9</td><td>0.15s</td><td>1.47G</td><td>240.01s</td><td>2.18e10</td><td>6.38s</td><td>39.50G</td><td>15.95h</td><td>1</td><td>1</td><td>1</td></tr><tr><td>EPAGCL</td><td>0.40s</td><td>3.57e8</td><td>0.04s</td><td>0.90G</td><td>0.57s</td><td>2.60e9</td><td>0.42s</td><td>17.10G</td><td>21.94s</td><td>2.23e10</td><td>3.89s</td><td>22.93G</td></tr></table></body></html>

Table 3: Comparison in terms of data pre-processing time, flops and training time of one epoch, and memory costs between different graph contrastive methods. Proc. and Mem. stand for data processing time and Memroy cost, respectively. ‘-’ indicates Out-Of-Memory on a 40GB GPU. On ogbn-arxiv, the model is trained with a batch size of 256 due to the memory limit.

the model is trained with a batch size of 256 because of the memory limit. The results are shown in Table 3.

The experiments show that although our methods takes more time to pre-process data and more memory for training than GCA, the additional burden is relatively small even when it is applied on a huge dataset like ogbn-arxiv. As for MVGRL and GREET, the pre-processing time and memory requirement are much higher. Specifically, on ogbn-arxiv, MVGRL reports Out-Of-Memory during the pre-processing phase. While GREET takes a lot of time to pre-process data and finally reports OOM when training in regardless of batch size. Moreover, the pre-processing time of EPAGCL grows in a slow rate with the increase of the scales of graph, which further indicates that EPAGCL also fits the huge datasets. What’s more, EPAGCL also shows a relatively fast training speed, especially compared with GREET.

# 4.5 Ablation Study (Q3)

Further experiments are conducted to demonstrate the effect of augment strategies. We investigate the performance of the following six augmentations without feature mask on some benchmark datasets: randomly add and drop edges as ‘Random Add’; add and drop edges adaptively on both views as ‘Add to Both Views’; our method as ‘EPAGCL’; drop edges adaptively as ‘Drop Only’; drop edges randomly as ‘Random Drop’; and add edges adaptively as ‘Add Only’.

The results are shown in Figure 4. To better manifest the difference between the performance, we illustrate the performance improvement of the five augmentations compared with ‘Random Add’. It can be observed that our method achieves the best performance on each dataset. Moreover, ‘Drop Only’ has an advantage over ‘Add Only’. This indicates that adding edges is a relatively poorer augmentation way, which is consistent with our theory. Thirdly, the adaptive strategy ‘Add to Both Views’ leads to a great accuracy increase compared with the random strategy ‘Random Add’, which proves the effectiveness of our method.

![](images/22cb665ce0ce87293c4cd18ef8390ce00b0eb2924d1aea5485ac0087981fdb6a.jpg)  
Figure 4: Performance improvement of five augmentation strategies compared to ‘Random Add’.

To sum up, our adaptive augmentation strategy is effective for edge-dropping and edge-adding as well. And adding edges to only one view fully utilizes the augmentation.

# 5 Conclusion

In this paper, we propose a novel algorithm EPAGCL for GCL. The main idea of EPAGCL is to use adding edges as one of the augmentations to generate views. We firstly introduce Error-Passing Rate (EPR) to measure the quality of a view. Based on EPR, the magnitude of effect of edge perturbation is quantified. Thus, we are able to add edges adaptively with low time and memory burden and without labels, which is also valid for edge dropping. Extensive experiments validate the correctness of our theoretical and reveal the effectiveness of our algorithm.