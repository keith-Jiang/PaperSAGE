# Multi-View Multi-Label Classification via View-Label Matching Selection

Hao Wei1, Yongjian Deng1, Qiuru Hai1, Yuena $\mathbf { L i n } ^ { 1 , 2 }$ , Zhen Yang1, Gengyu Lyu1\*

1College of Computer Science, Beijing University of Technology 2Idealism Beijing Technology Co., Ltd. haowei@emails.bjut.edu.cn, yjdeng@bjut.edu.cn, haiqiuru@emails.bjut.edu.cn, yuenalin $@$ 126.com, yangzhen@bjut.edu.cn, lyugengyu $@$ gmail.com

# Abstract

In multi-view multi-label classification (MVML), each object is described by several heterogeneous views while annotated with multiple related labels. The key to learn from such complicate data lies in how to fuse cross-view features and explore multi-label correlations, while accordingly obtain correct assignments between each object and its corresponding labels. In this paper, we proposed an advanced MVML method named VAMS, which treats each object as a bag of views and reformulates the task of MVML as a “view-label” matching selection problem. Specifically, we first construct an object graph and a label graph respectively. In the object graph, nodes represent the multi-view representation of an object, and each view node is connected to its K-nearest neighbor within its own view. In the label graph, nodes represent the semantic representation of a label. Then, we connect each view node with all labels to generate the unified “viewlabel” matching graph. Afterwards, a graph network block is introduced to aggregate and update all nodes and edges on the matching graph, and further generating a structural representation that fuses multi-view heterogeneity and multi-label correlations for each view and label. Finally, we derive a prediction score for each view-label matching and select the optimal matching via optimizing a weighted cross-entropy loss. Extensive results on various datasets have verified that our proposed VAMS can achieve superior or comparable performance against state-of-the-art methods.

# Introduction

Multi-view multi-label classification (MVML) is a crucial task in the field of machine learning and data mining, which aims to extract both consensus and complementary information from multiple high-dimensional heterogeneous views and assign multiple semantically relevant labels to the given samples (Liu et al. 2023b; Li et al. 2024). For example, in the task of news classification (Figure 1), multiple sources of information (image, video, text) are integrated to provide multiple semantically relevant labels such as disaster, wildfire, and rescue. MVML provides an effective framework to learn a desired multi-label classifier for bridging these various information (features) to the diverse topics (labels) and

# Wildfires 101 | National Geographic

![](images/9c3b9b34bbd369ed8426d6d717eac5490f8620f08f11b8eac8bb59f2cfcc1711.jpg)  
Figure 1: An application of MVML in news classification.

further provides support for subsequent tasks, such as public opinion analysis and monitoring on social media platforms.

The key to learn from MVML data lies in how to effectively fuse these heterogeneous features while comprehensively characterizing all relevant labels. Based on different multi-view fusion strategies, existing MVML methods can be roughly divided into two categories: Feature-fusion strategy and Decision-fusion strategy. Feature-fusion strategy based methods (Liu et al. 2015; Zhang, Jia, and Li 2020) usually conduct multi-view feature fusion first to obtain a common feature representation and then they employ the common representation to learn final multi-label classifier. Decision-fusion strategy (Luo et al. 2015) based methods directly learn multiple multi-label base classifiers for different views, and they make the final prediction by averaging the results of these base classifiers. Intuitively, Featurefusion strategy pays more attention to cross-view consensus information, which tends to correspond to significant labels, while some rare or insignificant labels may be overwhelmed. Decision-fusion strategy focuses more on the individualview specificity information, which tends to correspond to view-specific labels, where some labels with difficult feature representations can not be easily detected. Recently, some MVML methods (Tan et al. 2021; Wu et al. 2019; Lyu et al. 2022) attempt to simultaneously take both consensus and specific information into consideration, which intend to comprehensively characterize all relevant labels. However, most of these methods extract cross-view consensus features and individual-view specificity features in a separate manner and they independently learn a common classifier and multiple view-specific classifier to make the final prediction. Basically, these methods still split up the connection of exploiting consistency and specificity in multi-view data, and meanwhile, they also have not well described the inherent view-label correspondence relationship, which inevitably leads the final prediction model to be sub-optimal.

In order to address these issues, in this paper, we propose a matching-based MVML method named VAMS, which integrates cross-view consensus information and individualview specificity information into a unified framework and directly constructs the explicit matching correspondences between each view and label. Specifically, we first construct an object graph by connecting different view representations for fusing cross-view consistency and individual-view specificity, where the neighbor information of each view node is incorporated as supplementary information to enhance its feature expressiveness. Next, a label graph is established to capture multi-label semantic correlations, and all label nodes are connected to each view node to form the unified “view-label” matching graph. Afterwards, a graph network block (GN Block) is employed to perform node interactions among all views and labels, where cross-view feature consistency, view-specific feature specificity, and multi-label semantic correlations are simultaneously incorporated by a graph convolution propagation mechanism, thereby obtaining structural representation for each view and label. Finally, we derive prediction confidence of each view-label matching by optimizing the output of our model with a weighted cross-entropy loss. In summary, the contributions of our paper lie in the following aspects:

• We propose an advanced MVML method named VAMS, which leverages “view-label” matching to jointly fuse cross-view consensus, individual-view specificity, and multi-label correlation into the whole learning process, finally achieving accurate MVML prediction. • To the best of our knowledge, it is the first time to incorporate matching selection mechanism into MVML task, which avoids the split of consistency and specificity exploration in previous MVML methods, and accordingly improving the performance of the prediction model. • Enormous experimental results as well as comprehensive experimental analysis on various datasets have demonstrated that our proposed VAMS can achieve superior performance against state-of-the-art methods.

# Related Work

# Multi-View Learning (MVL)

Multi-View Learning aims to learn from different feature spaces to enhance model performance. Existing MVL methods can be roughly categorized into the following types: (Zhang et al. 2018) proposed a type of co-training method, which considers both view-specific and shared representations. (Liu et al. 2023a) proposed a multi-kernel learning, where contrastive learning is employed to leverage complementary information from data for high-quality kernel computation and combines kernels to enhance learning performance. (Zhang et al. 2023a; Luo et al. 2018) proposed subspace learning methods, which combine view consistency and specificity for effective subspace representation learning in multi-view clustering problems. Besides, there are also many other MVL methods for different tasks, such as clustering (Wang et al. 2023; Gu et al. 2023; Zhang et al. 2023b), retrieval (Dong et al. 2024) and classification (Jiang et al. 2021; Wen et al. 2024; Tan et al. 2024), etc.

# Multi-Label Learning (MLL)

Multi-Label Learning focuses on learning from data with multiple labels, and existing MLL methods can be broadly categorized into traditional methods and deep learningbased methods. 1) Traditional methods for handling MLL problems include problem transformation-based methods, which transfer multi-label problems into single-label learning, such as BR (Tsoumakas and loannis Katakis 2007), Classifier Chains (Liu, Tsang, and Mu¨ller 2017) and algorithm adaption-based methods, which convert the task of multi-label classification to some well-established learning scenarios, including CAMEL (Feng, An, and He 2019), RMFL (Feng et al. 2022), Metric Learning (Zhang et al. 2024; Liu et al. 2019) etc. 2) deep learning-based MLL methods tend to utilize deep neural networks to automatically learn complex relationships among labels. The most frequently used deep learning based methods for MLL include deep neural networks (Zhao et al. 2024, 2022; Wei et al. 2023; Yang et al. 2023), convolution (Wu et al. 2021; Feng et al. 2020), and transformer (Lyu et al. 2024b), etc.

# Multi-View Multi-Label Learning (MVML)

Multi-View Multi-Label Learning combines the characteristics of MVL and MLL, making it more complex when dealing with multi-view data (Zhong, Lyu, and Yang 2024; Lyu et al. 2022). To learn from such complicated data, the main methods include Feature-fusion strategy and Decisionfusion strategy. Feature-fusion strategy typically obtains a unified feature representation by fusing multi-view features, which is then used to train the final multi-label classifier (Liu et al. 2015). Decision-fusion strategy trains multiple multilabel base classifiers for different views and makes the final prediction by averaging the outputs of these base classifiers (Tan et al. 2021). Recently, some methods attempt to simultaneously take both consensus and specific information into consideration, which intend to comprehensively characterize all relevant labels, such as (Lyu et al. 2024a).

# The Proposed Method

Formally speaking, we define $\mathcal { X } = R ^ { d _ { 1 } } \times R ^ { d _ { 2 } } \times \cdot \cdot \cdot \times R ^ { d _ { V } }$ as the feature space with $V$ views, where each view has $d _ { m }$ dimension. For a given dataset $D = \{ ( X _ { i } , y _ { i } ) | 1 \leq i \leq$ $N \}$ , we denote each object $X _ { i }$ consists of $V$ feature vectors $[ { \pmb x } _ { i } ^ { 1 } ; { \pmb x } _ { i } ^ { 2 } ; . . . ; { \pmb x } _ { i } ^ { V } ]$ , $\pmb { y } _ { i } = [ y _ { i } ^ { 1 } , y _ { i } ^ { 2 } , . . . , y _ { i } ^ { Q } ]$ as the groundtruth label for $X _ { i }$ , where $Q$ is the total number of labels in the dataset. $y _ { i } ^ { c } = 1 ( 1 \leq c \leq Q )$ indicates that object $X _ { i }$ is annotated with label $c$ , $y _ { i } ^ { c } = 0$ otherwise. Our proposed VAMS method aims to integrate these diverse representations from different views to construct a robust multilabel classifier and further assign the predictive labels for test samples. Figure 2 illustrates the overview architecture of VAMS, which consists of three key components: ViewLabel Matching Graph Construction, Graph Network Convolution, and Multi-Label Classification.

![](images/101be9b139df0f35f6c63ea2e41f74528aba108476df36675e08477efa5516cc.jpg)  
Figure 2: The framework of our proposed VAMS, which consists of three components: (1) View-Label Matching Graph Construction, which connects the inter-view features of an object and links each view node to $k$ intra-view neighbors to form an object graph, while also constructing a fully connected label graph, ultimately connecting each view node to all labels to form a view-label matching; (2) Graph Network Convolution, using a Graph Network Block (GN Block) containing edge convolution and node convolution to perform aggregating and updating; (3) Multi-Label Classification, where a decoder derives view-label matching scores from the updated graph state, and averages them to obtain the final label prediction.

# View-Label Matching Graph Construction

In order to explicitly characterize the direct view-label correspondence relationship, we construct a unified view-label matching graph $\mathbb { G } ^ { m } = \mathbf { \bar { \Psi } } ( \mathbb { G } ^ { o } , \mathbb { G } ^ { l } , \mathbb { E } ^ { m } )$ , which consists of an object graph $\mathbb { G } ^ { o } = ( \mathbb { V } ^ { o } , \mathbb { E } ^ { o } )$ , a label graph $\mathbb { G } ^ { l } = ( \mathbb { V } ^ { l } , \mathbb { E } ^ { l } )$ , and their matching edges $\mathbb { E } ^ { m }$ . Specifically, we first connect $V$ feature representations of an object to construct the object graph $\bar { \mathbb { G } ^ { o } } = ( \mathbb { V } ^ { o } , \mathbb { E } ^ { o } )$ , where cross-view feature nodes $\mathbf { \bar { \{ } }  \mathbf { \Psi } \mathbf { \bar { { \xi } } } \mathbf { { \xi } } \mathbf { { \xi } } \mathbf { { \bar { \{ } } }  \mathbf { \bar { { \xi } } } \mathbf { \bar { { \xi } } } \mathbf { \bar { { \xi } } } \mathbf { \bar { { \xi } } } \mathbf { \xi } \mathbf { \bar { \{ } }  \mathbf { \xi } \mathbf { \xi } \mathbf { \xi } \mathbf { \xi }$ are integrated into a unified graph to explore the cross-view consensuses. Meanwhile, in order to enhance the individual-view specificities, we introduce the $K$ nearest neighbors $\{ \pmb { v } _ { i _ { k } } ^ { o } | _ { k = 1 } ^ { K } \}$ of each view node as complementary information and connect them to their corresponding view node $\pmb { v } _ { i } ^ { o }$ . Note that $\mathbb { G } ^ { o }$ is an undirected graph, where each node $\pmb { v } _ { i } ^ { o } \in \mathbb { V } ^ { o }$ is represented by the feature vector $\mathbf { \boldsymbol { x } } ^ { i }$ , and each edge $\boldsymbol { e } _ { i j } ^ { o } \in \mathbb { E } ^ { o }$ is described by concatenating the feature vectors of the connected nodes $\pmb { v } _ { i } ^ { o }$ and $\pmb { v } _ { j } ^ { o }$ :

$$
\pmb { v } _ { i } ^ { o } = \pmb { x } ^ { i } , \quad \pmb { e } _ { i j } ^ { o } = [ \pmb { x } ^ { i } , \pmb { x } ^ { j } ] ,
$$

where $[ \cdot , \cdot ]$ represents the vector concatenation operation.

After obtaining the object graph $\mathbb { G } ^ { o }$ , we further construct a fully connected label graph $\mathbb { G } ^ { \hat { l } } = ( \mathbb { V } ^ { l } , \mathbb { E } ^ { l } )$ to capture the label correlations, where each label node $\{ \pmb { v } _ { i } ^ { l } | _ { i = 1 } ^ { Q } \} \in \mathbb { V } ^ { l }$ is represented by one-hot embedding $c ^ { i }$ of the $i$ -th class. For the edge $e _ { i j } ^ { l } \ \in \ E ^ { l }$ , we employ the same strategy in object graph to concatenate the embedding vectors of the two connected label nodes $\pmb { v } _ { i } ^ { l }$ and $\boldsymbol { v } _ { j } ^ { l }$ , forming its edges attributes:

$$
\pmb { v } _ { i } ^ { l } = { \pmb { c } } ^ { i } , \quad \pmb { e } _ { i j } ^ { l } = [ { \pmb { c } } ^ { i } , { \pmb { c } } ^ { j } ] ,
$$

Finally, to establish the full matching correspondence relationship between views and labels, we connect each view node $\pmb { v } _ { i } ^ { o }$ in object graph $\mathbb { G } ^ { o }$ with each label node $\pmb { v } _ { i } ^ { l }$ in label graph $\mathbb { G } ^ { l }$ to generate the unified View-Label Matching Graph $\bar { \mathbb { G } } ^ { m } = ( \mathbb { G } ^ { \breve { o } } , \mathbb { G } ^ { l } , \mathbb { E } ^ { m } )$ , where the edges $\boldsymbol { e } _ { i j } ^ { m } \in \mathbb { E } ^ { m }$ connecting view nodes and labels remain undirected, and their attributes are generated by the feature concatenation of the connected nodes:

$$
\begin{array} { r } { e _ { i j } ^ { m } = [ \pmb { v } _ { i } ^ { o } , \pmb { v } _ { j } ^ { l } ] . } \end{array}
$$

According to the above operations, we explicitly construct the “view-label” matching correspondence between object and labels, which jointly integrates cross-view consensuses, individual-view specificities, and multi-label correlations into a unified framework. Meanwhile, such unified framework avoids the separation of previous MVML methods in exploiting multi-view consistency and specificity and significantly enhances the comprehensive semantic characterization capability of the final model.

# Graph Network Convolution

To better fuse the above multi-granularity relationships in MVML data and generate more distinctive graph representations, motivated by (Wang et al. 2020), we introduce graph network block (GN Block) to aggregate and update nodes and edges in the “view-label” matching graph. This module consists of a node convolution layer, which collects the attributes of all the nodes and edges adjacent to each node to compute per-node updates, and an edge convolution layer, which assembles the attributes of the two nodes associated with each edge to generate a new attribute of this edge.

Node Convolution. Each node convolution layer consists of a group of aggregation functions, which gather the information from its adjacent nodes and associated edges, and an update function, which updates node attributes according to these gathered information. In our model, there are two different types of nodes that need to be aggregated and updated, including view nodes and label nodes.

Specifically, for a view node $\pmb { v } _ { i } ^ { o }$ in $\mathbb { G } ^ { o }$ , its connected view nodes $\{ \pmb { v } _ { j } ^ { o } \vert _ { j = 1 } ^ { V - 1 } \}$ via view edges $e _ { i j } ^ { o }$ , neighbor view nodes $\{ \pmb { v } _ { i _ { k } } ^ { o } | _ { k = 1 } ^ { K } \}$ via neighbor edges $\boldsymbol { e } _ { i _ { k } } ^ { o }$ and label nodes $\{ \boldsymbol { v } _ { j } ^ { l } | _ { j = 1 } ^ { Q } \}$ via matching edges $e _ { i j } ^ { m }$ are gathered to update its attribute representations, i.e.,

$$
\begin{array} { l l l } { { \displaystyle \bar { \boldsymbol { v } } _ { i } ^ { o } = \frac { 1 } { V - 1 } \bar { \rho } _ { n } ^ { o } ( [ e _ { i j } ^ { o } , { \boldsymbol { v } } _ { j } ^ { o } ] ) , } } & { { \displaystyle \hat { \boldsymbol { v } } _ { i } ^ { o } = \frac { 1 } { K } \hat { \rho } _ { n } ^ { o } ( [ e _ { i _ { k } } ^ { o } , { \boldsymbol { v } } _ { i _ { k } } ^ { o } ] ) , } } \\ { { \displaystyle \tilde { \boldsymbol { v } } _ { i } ^ { o } = \tilde { \rho } _ { n } ^ { o } ( [ e _ { i j } ^ { m } , { \boldsymbol { v } } _ { j } ^ { l } ] ) , } } & { { \displaystyle { \boldsymbol { v } } _ { i } ^ { o } \gets \phi _ { n } ^ { o } ( [ { \boldsymbol { v } } _ { i } ^ { o } , \bar { { \boldsymbol { v } } } _ { i } ^ { o } , \hat { { \boldsymbol { v } } } _ { i } ^ { o } , \tilde { { \boldsymbol { v } } } _ { i } ^ { o } ] ) , } } \end{array}
$$

where $\bar { \rho } _ { n } ^ { o } , \ \hat { \rho } _ { n } ^ { o } , \ \tilde { \rho } _ { n } ^ { o }$ are aggregation functions, which gather the information from view nodes in $\mathbb { G } ^ { o }$ and label nodes in $\mathbb { G } ^ { l }$ respectively. $\phi _ { n } ^ { o }$ concatenates the current attributes of $\pmb { v } _ { i } ^ { o }$ with the gathered information $\bar { \pmb { v } } _ { i } ^ { o } , \hat { \pmb { v } } _ { i } ^ { o }$ and $\tilde { \pmb { v } } _ { i } ^ { o }$ to derive the updated attributes for $\pmb { v } _ { i } ^ { o }$ .

For a label node $\pmb { v } _ { i } ^ { l }$ in $\mathbb { G } ^ { l }$ , its connected label nodes $\{ \pmb { v } _ { j } ^ { l } | _ { j = 1 } ^ { Q - 1 } \}$ via label edges $e _ { i j } ^ { l }$ and view nodes $\{ { \pmb v } _ { j } ^ { o } | _ { j = 1 } ^ { V } \}$ via matching edges $e _ { i j } ^ { m }$ are gathered to update its attribute representation, i.e.,

$$
\bar { \pmb { v } } _ { i } ^ { l } = \frac { 1 } { Q - 1 } \bar { \rho } _ { n } ^ { l } ( [ \pmb { e } _ { i j } ^ { l } , \pmb { v } _ { j } ^ { l } ] ) , \quad \tilde { \pmb { v } } _ { i } ^ { l } = \frac { 1 } { V } \tilde { \rho } _ { n } ^ { l } ( [ \pmb { e } _ { i j } ^ { m } , \pmb { v } _ { j } ^ { o } ] ) ,
$$

$$
\pmb { v } _ { i } ^ { l }  \phi _ { n } ^ { l } ( [ \pmb { v } _ { i } ^ { l } , \bar { \pmb { v } } _ { i } ^ { l } , \tilde { \pmb { v } } _ { i } ^ { l } ] ) .
$$

where $\bar { \rho } _ { n } ^ { l }$ and $\tilde { \rho } _ { n } ^ { l }$ are aggregation functions, which gather the information from $\mathbb { G } ^ { o }$ and $\mathbf { \bar { \mathbb { G } } } ^ { l }$ respectively. $\phi _ { n } ^ { l }$ concatenates the current attributes of $\pmb { v } _ { i } ^ { l }$ with the gathered information $\bar { \pmb v } _ { i } ^ { l }$ and $\tilde { \boldsymbol { v } } _ { i } ^ { l }$ to derive the updated attributes for $\pmb { v } _ { i } ^ { l }$ .

Edge Convolution. Each edge convolution layer consists of an aggregation function that gathers the information from its associated nodes and an update function that updates node attributes via these gathered information. In our model, three different types of edges are aggregated and updated, including view edges, label edges, and matching edges.

Specifically, for a view edge $e _ { i j } ^ { o } , e _ { i _ { k } } ^ { o }$ in $\mathbb { G } ^ { o }$ , its attribute representation is updated by:

$$
\begin{array} { r } { \hat { \pmb { e } } _ { i j } ^ { o } = \rho _ { e } ^ { o } ( [ \pmb { v } _ { i } ^ { o } , \pmb { v } _ { j } ^ { o } ] ) , \qquad \pmb { e } _ { i j } ^ { o }  \phi _ { e } ^ { o } ( [ \pmb { e } _ { i j } ^ { o } , \hat { \pmb { e } } _ { i j } ^ { o } ] ) , } \\ { \hat { \pmb { e } } _ { i _ { k } } ^ { o } = \rho _ { e } ^ { o } ( [ \pmb { v } _ { i } ^ { o } , \pmb { v } _ { i _ { k } } ^ { o } ] ) , \qquad \pmb { e } _ { i _ { k } } ^ { o }  \phi _ { e } ^ { o } ( [ \pmb { e } _ { i _ { k } } ^ { o } , \hat { \pmb { e } } _ { i _ { k } } ^ { o } ] ) . } \end{array}
$$

Similarly, for a label edge $e _ { i j } ^ { l }$ in $\mathbb { G } ^ { l }$ , its attribute representation is aggregated and updated by:

$$
\begin{array} { r } { \hat { e } _ { i j } ^ { L } = \rho _ { e } ^ { l } ( [ { \pmb v } _ { i } ^ { l } , { \pmb v } _ { j } ^ { l } ] ) , \quad e _ { i j } ^ { l }  \phi _ { e } ^ { l } ( [ { \pmb e } _ { i j } ^ { l } , \hat { \pmb e } _ { i j } ^ { l } ] ) . } \end{array}
$$

Algorithm 1: The training process of VAMS

Input: Multi-view data: $D = \{ ( X _ { i } , y _ { i } ) | 1 \leq i \leq N \}$ , The number of convolutions: $C$ , The number of epochs: $I _ { m }$ . Output: Prediction Model.

# Process:

1: Construct view-label matching graph $G ^ { m }$ by Eq.(1)-(3);   
2: Intialized the attributes of nodes and edges;   
3: for epoch ${ \bf \mu } = 1 { \bf \rho }$ to $I _ { m }$ do   
4: // Forward Propagation   
5: for conv ${ \bf \mu } = 1 { \bf \rho }$ to $C$ do   
6: Conduct node convolution via Eq. (4)-(6);   
7: Conduct edge convolution via Eq. (7)-(9);   
8: end for   
9: Calculate matching score $r _ { v c }$ by Eq. (10);   
10: Calculate prediction score $r _ { c } ^ { i }$ of $X _ { i }$ ;   
11: // Backward Propagation   
12: Update the model parameters by optimizing Eq. (11);   
13: end for

As for the matching edges $e _ { i j } ^ { m }$ in $\mathbb { E } ^ { m }$ , its structural attributes can be represented by:

$$
\begin{array} { r } { \hat { \pmb { e } } _ { i j } ^ { m } = \rho _ { e } ^ { m } ( [ \pmb { v } _ { i } ^ { o } , \pmb { v } _ { j } ^ { l } ] ) , \quad \pmb { e } _ { i j } ^ { m }  \phi _ { e } ^ { m } ( [ \pmb { e } _ { i j } ^ { m } , \hat { \pmb { e } } _ { i j } ^ { m } ] ) . } \end{array}
$$

All the aforementioned aggregation and update functions in node convolution layer and edge convolution layer are implemented as MLPs, with different structures and parameters. Additionally, in order to better integrate the consistency and specificity information from different views and further enhance the feature expression of each graph node, we repeat the above convolution operation to fuse more information into each node, then obtain desired structural representations for subsequent multi-label classification.

# Multi-Label Classification

In our model, the task of MVML classification is transferred as “view-label” matching selection problem. Thus, we directly map the attributes of view-label matching edges to the view-label prediction scores for subsequent evaluation, i.e.,

$$
\begin{array} { r } { r _ { v c } = \Phi _ { e } ^ { d e c } ( e _ { v c } ^ { m } ) . } \end{array}
$$

Here, $r _ { v c } \in [ 0 , 1 ] ^ { V \times Q }$ represents the prediction score on $c$ - th label in $v$ -th view, and $\mathbf { \dot { \Phi } } _ { e } ^ { d e c }$ is an MLP.

Considering the contributions of different views, for each object $X _ { i }$ , we calculate its label prediction scores $r _ { c } ^ { i }$ by averaging the outputs from all $V$ views, i.e., $\begin{array} { r } { r _ { c } ^ { i } = \frac { 1 } { V } \sum _ { v = 1 } ^ { V } r _ { v c } ^ { i } } \end{array}$ where $r _ { v c } ^ { i }$ is the prediction score of $X _ { i }$ on $c$ -th label in $v$ -th view. Accordingly, we can train the model by optimizing:

$$
\mathcal { L } = \sum _ { i = 1 } ^ { N } \sum _ { c = 1 } ^ { Q } w _ { c } [ y _ { i } ^ { c } \log ( r _ { c } ^ { i } ) + ( 1 - y _ { i } ^ { c } ) \log ( 1 - r _ { c } ^ { i } ) ) ] ,
$$

where $w _ { c } = y _ { i } ^ { c } \cdot e ^ { \beta ( 1 - p ^ { c } ) } + ( 1 - y _ { i } ^ { c } ) \cdot e ^ { \beta p ^ { c } }$ is employed to alleviate the class imbalance, $\beta$ is hyperparameter and $p ^ { c }$ is the ratio of label $c$ in the whole data set. Algorithm 1 illustrates the whole training process of our proposed method.

<html><body><table><tr><td colspan="8"></td></tr><tr><td>H-L</td><td>LrMMC</td><td>LSPC</td><td>SIMM</td><td>FIMAN</td><td>IMvMLC</td><td>ML-BVAE</td><td>VAMS</td></tr><tr><td>Emotions</td><td>0.196±0.011</td><td>0.251±0.014</td><td>0.307±0.004</td><td>0.231±0.013</td><td>0.330±0.021</td><td>0.317±0.012</td><td>0.193±0.019</td></tr><tr><td>Plant</td><td>0.115±0.002</td><td>0.168±0.007</td><td>0.090±0.001</td><td>0.238±0.011</td><td>0.202±0.038</td><td>0.090±0.001</td><td>0.090±0.001</td></tr><tr><td>scene</td><td>0.082±0.006</td><td>0.221±0.008</td><td>0.179±0.002</td><td>0.195±0.005</td><td>0.197±0.007</td><td>0.179±0.001</td><td>0.086±0.007</td></tr><tr><td>Yeast</td><td>0.255±0.020</td><td>0.297±0.008</td><td>0.241±0.011</td><td>0.216±0.004</td><td>0.313±0.007</td><td>0.232±0.004</td><td>0.204±0.007</td></tr><tr><td>human</td><td>0.096±0.002</td><td>0.176±0.003</td><td>0.085±0.001</td><td>0.151±0.002</td><td>0.112±0.006</td><td>0.085±0.001</td><td>0.083±0.001</td></tr><tr><td>Corel5k</td><td>0.013±0.000</td><td>0.020±0.000</td><td>0.013±0.018</td><td>0.018±0.000</td><td>0.158±0.008</td><td>0.013±0.000</td><td>0.012±0.008</td></tr><tr><td>Pascal</td><td>0.073±0.000</td><td>0.219±0.003</td><td>0.060±0.001</td><td>0.116±0.002</td><td>0.074±0.000</td><td>0.062±0.001</td><td>0.110±0.013</td></tr><tr><td>R-L</td><td>LrMMC</td><td>LSPC</td><td>SIMM</td><td>FIMAN</td><td>IMvMLC</td><td>ML-BVAE</td><td>VAMS</td></tr><tr><td>Emotions</td><td>0.133±0.016</td><td>0.185±0.022</td><td>0.344±0.047</td><td>0.161±0.026</td><td>0.183±0.019</td><td>0.423±0.035</td><td>0.144±0.013</td></tr><tr><td>Plant</td><td>0.371±0.014</td><td>0.576±0.037</td><td>0.378±0.025</td><td>0.277±0.028</td><td>0.210±0.015</td><td>0.238±0.012</td><td>0.184±0.024</td></tr><tr><td>scene</td><td>0.115±0.011</td><td>0.233±0.023</td><td>0.280±0.026</td><td>0.107±0.006</td><td>0.086±0.004</td><td>0.171±0.017</td><td>0.070±0.010</td></tr><tr><td>Yeast</td><td>0.275±0.011</td><td>0.530±0.018</td><td>0.218±0.018</td><td>0.187±0.005</td><td>0.180±0.005</td><td>0.204±0.007</td><td>0.168±0.004</td></tr><tr><td>human</td><td>0.358±0.006</td><td>0.618±0.018</td><td>0.261±0.046</td><td>0.186±0.011</td><td>0.149±0.007</td><td>0.181±0.008</td><td>0.148±0.009</td></tr><tr><td>Corel5k</td><td>0.173±0.004</td><td>0.860±0.005</td><td>0.160±0.005</td><td>0.085±0.000</td><td>0.114±0.003</td><td>0.188±0.008</td><td>0.082±0.004</td></tr><tr><td>Pascal</td><td>0.336±0.005</td><td>0.868±0.003</td><td>0.097±0.006</td><td>0.118±0.003</td><td>0.063±0.002</td><td>0.106±0.001</td><td>0.101±0.002</td></tr><tr><td>Cov</td><td>LrMMC</td><td>LSPC</td><td>SIMM</td><td>FIMAN</td><td>IMvMLC</td><td>ML-BVAE</td><td>VAMS</td></tr><tr><td>Emotions</td><td>2.198±0.094</td><td>1.905±0.138</td><td>0.457±0.051</td><td>7.796±0.189</td><td>1.873±0.037</td><td>3.163±0.242</td><td>1.659±0.055</td></tr><tr><td>Plant</td><td>4.256±0.147</td><td>6.606±0.369</td><td>4.325±0.270</td><td>3.216±0.350</td><td>2.461±0.171</td><td>2.749±0.141</td><td>2.184±0.256</td></tr><tr><td>scene</td><td>0.677±0.057</td><td>1.252±0.109</td><td>0.248±0.020</td><td>0.628±0.020</td><td>0.516±0.024</td><td>0.942±0.090</td><td>0.434±0.058</td></tr><tr><td>Yeast</td><td>10.32±0.195</td><td>11.27±0.171</td><td>7.368±0.293</td><td>6.673±0.074</td><td>6.538±0.094</td><td>6.706±0.094</td><td>6.376±0.096</td></tr><tr><td>human</td><td>5.281±0.072</td><td>8.848±0.276</td><td>3.797±0.640</td><td>2.817±0.095</td><td>2.271±0.099</td><td>2.666±0.112</td><td>2.245±0.102</td></tr><tr><td>Corel5k</td><td>96.72±1.300</td><td>257.3±0.499</td><td>95.99±3.146</td><td>53.94±0.790</td><td>70.80±2.256</td><td>108.7±4.792</td><td>53.20±2.218</td></tr><tr><td>Pascal</td><td>7.900±0.060</td><td>17.08±0.065</td><td>2.772±0.140</td><td>3.486±0.081</td><td>1.908±0.065</td><td>2.966±0.028</td><td>2.231±0.038</td></tr><tr><td>A-P</td><td>LrMMC</td><td>LSPC</td><td>SIMM</td><td>FIMAN</td><td>IMvMLC</td><td>ML-BVAE</td><td>VAMS</td></tr><tr><td>Emotions</td><td>0.763±0.020</td><td>0.773±0.025</td><td>0.634±0.043</td><td>0.806±0.027</td><td>0.782±0.021</td><td>0.572±0.022</td><td>0.826±0.012</td></tr><tr><td>Plant</td><td>0.464±0.016</td><td>0.376±0.023</td><td>0.369±0.029</td><td>0.492±0.030</td><td>0.544±0.017</td><td>0.505±0.020</td><td>0.585±0.027</td></tr><tr><td>scene</td><td>0.852±0.012</td><td>0.647±0.020</td><td>0.608±0.027</td><td>0.827±0.010</td><td>0.844±0.004</td><td>0.717±0.023</td><td>0.878±0.014</td></tr><tr><td>Yeast</td><td>0.610±0.013</td><td>0.554±0.012</td><td>0.712±0.014</td><td>0.740±0.007</td><td>0.738±0.006</td><td>0.712±0.007</td><td>0.763±0.009</td></tr><tr><td>human</td><td>0.480±0.006</td><td>0.304±0.021</td><td>0.495±0.042</td><td>0.583±0.015</td><td>0.600±0.010</td><td>0.536±0.010</td><td>0.609±0.014</td></tr><tr><td>Corel5k</td><td>0.215±0.010</td><td>0.075±0.004</td><td>0.292±0.004</td><td>0.430±0.007</td><td>0.333±0.008</td><td>0.286±0.000</td><td>0.452±0.009</td></tr><tr><td>Pascal</td><td>0.422±0.004</td><td>0.116±0.003</td><td>0.685±0.010</td><td>0.721±0.003</td><td>0.695±0.008</td><td>0.659±0.002</td><td>0.759±0.008</td></tr><tr><td>Micro-F1</td><td>LrMMC</td><td>LSPC</td><td>SIMM</td><td>FIMAN</td><td>IMvMLC</td><td>ML-BVAE</td><td>VAMS</td></tr><tr><td>Emotions</td><td>0.685±0.018</td><td>0.653±0.022</td><td>0.034±0.051</td><td>0.671±0.014</td><td>0.482±0.010</td><td>0.107±0.091</td><td>0.692±0.015</td></tr><tr><td>Plant</td><td>0.339±0.016</td><td>0.230±0.025</td><td>0.004±0.005</td><td>0.299±0.014</td><td>0.181±0.006</td><td>0.146±0.036</td><td>0.370±0.027</td></tr><tr><td>scene</td><td>0.772±0.017</td><td>0.544±0.019</td><td>0.001±0.002</td><td>0.616±0.008</td><td>0.308±0.002</td><td>0.304±0.095</td><td>0.748±0.016</td></tr><tr><td>Yeast</td><td>0.516±0.012</td><td>0.385±0.010</td><td>0.428±0.042</td><td>0.111±0.008</td><td>0.468±0.003</td><td>0.479±0.008</td><td>0.656±0.012</td></tr><tr><td>human</td><td>0.391±0.009</td><td>0.212±0.014</td><td>0.001±0.002</td><td>0.177±0.012</td><td>0.159±0.000</td><td>0.389±0.011</td><td>0.438±0.017</td></tr><tr><td>Corel5k</td><td>0.273±0.009</td><td>0.153±0.004</td><td>0.038±0.010</td><td>0.361±0.009</td><td>0.029±0.000</td><td>0.025±0.001</td><td>0.372±0.005</td></tr><tr><td>Pascal</td><td>0.283±0.015</td><td>0.084±0.003</td><td>0.343±0.044</td><td>0.008±0.002</td><td>0.136±0.001</td><td>0.385±0.031</td><td>0.426±0.026</td></tr><tr><td>Macro-F1</td><td>LrMMC</td><td>LSPC</td><td>SIMM</td><td>FIMAN</td><td>IMvMLC</td><td>ML-BVAE</td><td>VAMS</td></tr><tr><td>Emotions</td><td>0.684±0.017</td><td>0.647±0.023</td><td>0.032±0.051</td><td>0.657±0.016</td><td>0.478±0.012</td><td>0.052±0.031</td><td>0.679±0.017</td></tr><tr><td>Plant</td><td>0.111±0.007</td><td>0.105±0.014</td><td>0.002±0.003</td><td>0.202±0.018</td><td>0.160±0.003</td><td>0.024±0.008</td><td>0.196±0.026</td></tr><tr><td>scene</td><td>0.772±0.017</td><td>0.522±0.019</td><td>0.001±0.002</td><td>0.622±0.008</td><td>0.307±0.002</td><td>0.304±0.078</td><td>0.758±0.014</td></tr><tr><td>Yeast</td><td>0.398±0.012</td><td>0.243±0.005</td><td>0.133±0.007</td><td>0.356±0.004</td><td>0.425±0.003</td><td>0.122±0.001</td><td>0.478±0.015</td></tr><tr><td>human</td><td>0.157±0.010</td><td>0.089±0.012</td><td>0.00±0.001</td><td>0.166±0.005</td><td>0.143±0.001</td><td>0.111±0.009</td><td>0.211±0.017</td></tr><tr><td>Corel5k</td><td>0.174±0.018</td><td>0.005±0.000</td><td>0.003±0.001</td><td>0.078±0.002</td><td>0.025±0.000</td><td>0.003±0.000</td><td>0.094±0.002</td></tr><tr><td>Pascal</td><td>0.231±0.017</td><td>0.034±0.001</td><td>0.164±0.012</td><td>0.416±0.008</td><td>0.127±0.001</td><td>0.169±0.007</td><td>0.420±0.007</td></tr></table></body></html>

Table 1: Experimental comparisons of VAMS with other comparing methods on six evaluation metrics, where the best performances on each metric are shown in bold face. For H-L, R-L and Cov, the lower value indicates the better performance. For A-P, Micro-F1 and Macro-F1, the higher value indicates the better performance.

![](images/b537f67aad754ab80323a19307401f69d1342e4329d86f5d80e4c92f698071de.jpg)  
Figure 3: Experimental comparisons of our proposed VAMS against other comparing algorithms with the Bonferroni-Dunn test. Algorithms not connected with VAMS are significantly inferior to VAMS $\mathrm { \langle C D } = 3 . 0 4 6$ at 0.05 significance level).

# Experiments

# Experimental Setup

To evaluate our proposed VAMS method, we conducted experiments on seven widely-used MVML datasets, including Emotions, Scene, Yeast, Plant, Human, Corel5k and Pascal, which can be downloaded from Mulan website1. Meanwhile, we compare it with several state-of-the-art methods, including LrMMC (Liu et al. 2015), LSPC (Szyman´ski, Kajdanowicz, and Kersting 2016), SIMM (Wu et al. 2019), FIMAN (Wu et al. 2020), IMvMLC (Wen et al. 2024) and ML-BVAE (Fu et al. 2024). The configured parameters of all comparing methods are set according to the suggestions in their corresponding literature. Additionally, we adopt six widely used multi-label metrics to evaluate each method, including Hamming Loss (H-L), Ranking Loss (R-L), Coverage (Cov), Average Precision (A-P), Micro-F1 and MacroF1. The detailed definitions of these metrics are available in (Sun and Zong 2021). Finally, we conduct experimental comparison between VAMS and all other methods, where five-fold cross-validation is performed on each data set.

# Experimental Results

Table 1 illustrates the experimental comparison between our proposed VAMS and other six comparing methods across all evaluation metrics, where the mean results and standard deviations are recorded. According to the 252 statistical comparisons, some key observations are clearly revealed:

• Among all comparing methods, our proposed VAMS is superior to LSPC in all cases. Meanwhile, it also outperforms FIMAN in $9 7 . 6 \%$ cases, ML-BVAE in $9 5 . 2 \%$ cases, IMvMLC in $9 2 . 9 \%$ cases, SIMM in $8 8 \%$ cases, and LrMMC in $8 5 \%$ cases, respectively. • Among all evaluation metrics, our proposed VAMS achieves the best performance on Average Precision metric, and it also outperforms other methods over $98 \%$ cases on Micro-F1, $9 3 . 9 \%$ on Ranking Loss and Coverage, $9 1 . 8 \%$ on Macro-F1, and $8 9 . 8 \%$ on Hamming Loss.

Table 2: Friedman statics $\tau _ { F }$ in terms of each evaluation metric (at 0.05 significance level).   

<html><body><table><tr><td>Evaluation Metric</td><td>TF</td><td>critical value</td></tr><tr><td>Hamming Loss</td><td>7.203</td><td></td></tr><tr><td>Ranking Loss</td><td>11.367</td><td></td></tr><tr><td>Coverage</td><td>11.974</td><td>2.365</td></tr><tr><td>Average Precision</td><td>7.235</td><td>Methods:7,Data Set:7</td></tr><tr><td>Micro-F1</td><td>8.967</td><td></td></tr><tr><td>Macro-F1</td><td>20.902</td><td></td></tr></table></body></html>

• Additionally, the improvements of our proposed VAMS against other methods are quite significant, especially compared with the second-best method, our proposed VAMS shows a significant improvement of $2 \%$ to $4 \%$ on Average Precision and $2 \%$ to $14 \%$ on Micro-F1.

In order to comprehensively evaluate the superiority of the proposed VAMS, we employ the Friedman test (Demsˇar 2006) as the statistical method to analyze relative performance among the comparing algorithms. As shown in Table 2, the null hypothesis of distinguishable performance among the comparing algorithms is rejected at 0.05 significance level. Consequently, we utilize the post-hoc BonferroniDunn test (Demsˇar 2006) to further compare the relative performance among the algorithms. Figure 3 illustrates the Critical Difference (CD) diagrams on each evaluation metric, with the average rank of each algorithm marked along the axis. According to Figure 3, it is observed that VAMS always ranks 1st on all evaluation metrics and it performs significant superiority against most comparing methods.

# Further Analysis

# Ablation Study

To evaluate the effectiveness of each component in our proposed framework, we perform an ablation study comparing VAMS with its three degenerated versions: VAMS-w/o IntraKNN, VAMS-w/o InterConn, VAMS-w/o SemCor, where

<html><body><table><tr><td>Emotions</td><td>Hamming Loss</td><td>Ranking Loss</td><td>Coverage</td><td>Average precision</td><td>Micro-F1</td><td>Macro-F1</td></tr><tr><td>VAMS-w/o IntraKNN</td><td>0.224±0.025</td><td>0.149±0.016</td><td>1.707±0.079</td><td>0.818±0.018</td><td>0.689±0.019</td><td>0.676±0.020</td></tr><tr><td>VAMS-w/o InterConn</td><td>0.223±0.020</td><td>0.151±0.024</td><td>1.727±0.155</td><td>0.812±0.031</td><td>0.614±0.058</td><td>0.601±0.062</td></tr><tr><td>VAMS-w/o SemCor</td><td>0.218±0.018</td><td>0.145±0.016</td><td>1.692±0.087</td><td>0.826±0.019</td><td>0.682±0.016</td><td>0.665±0.024</td></tr><tr><td>VAMS</td><td>0.193±0.019</td><td>0.144±0.013</td><td>1.659±0.055</td><td>0.826±0.012</td><td>0.692±0.015</td><td>0.679±0.017</td></tr><tr><td>Scene</td><td>Hamming Loss</td><td>Ranking Loss</td><td>Coverage</td><td>Average precision</td><td>Micro-F1</td><td>Macro-F1</td></tr><tr><td>VAMS-w/o IntraKNN</td><td>0.089±0.009</td><td>0.071±0.007</td><td>0.442±0.037</td><td>0.872±0.011</td><td>0.746±0.018</td><td>0.757±0.013</td></tr><tr><td>VAMS-w/o InterConn</td><td>0.094±0.008</td><td>0.075±0.008</td><td>0.463±0.054</td><td>0.865±0.013</td><td>0.713±0.022</td><td>0.722±0.019</td></tr><tr><td>VAMS-w/o SemCor</td><td>0.094±0.010</td><td>0.070±0.003</td><td>0.434±0.018</td><td>0.876±0.009</td><td>0.740±0.016</td><td>0.749±0.016</td></tr><tr><td>VAMS</td><td>0.086±0.007</td><td>0.070±0.010</td><td>0.434±0.008</td><td>0.878±0.014</td><td>0.748±0.016</td><td>0.758±0.014</td></tr><tr><td>Corel5k</td><td>Hamming Loss</td><td>Ranking Loss</td><td>Coverage</td><td>Average precision</td><td>Micro-F1</td><td>Macro-F1</td></tr><tr><td>VAMS-w/o IntraKNN</td><td>0.014±0.003</td><td>0.146±0.005</td><td>55.787±2.253</td><td>0.410±0.005</td><td>0.331±0.021</td><td>0.056±0.015</td></tr><tr><td>VAMS-w/o InterConn</td><td>0.016±0.007</td><td>0.151±0.006</td><td>54.088±2.427</td><td>0.428±0.006</td><td>0.315±0.047</td><td>0.047±0.043</td></tr><tr><td>VAMS-w/o SemCor</td><td>0.014±0.002</td><td>0.149±0.003</td><td>54.009±2.352</td><td>0.408±0.004</td><td>0.340±0.035</td><td>0.071±0.026</td></tr><tr><td>VAMS</td><td>0.013±0.008</td><td>0.082±0.004</td><td>53.200±2.218</td><td>0.452±0.009</td><td>0.372±0.005</td><td>0.094±0.002</td></tr></table></body></html>

Table 3: The experimental results of our proposed VAMS and its three degenerated methods over all employed evaluation metrics on Emotions, Scene and Pascal data sets, where VAMS-w/o IntraKNN, VAMS-w/o InterConn and VAMS-w/o SemCor do not consider the intra-view correlations, inter-view alignments, and label semantic correlations, respectively.

1   
0.8 0.8 Hamming Loss Hamming Loss   
0.6 \*Ranking Loss 0.6 \*Ranking Loss +Coverage +Coverage   
0.4 -AveragePrecision 0.4 +Average Precision 中Micro-F1 中Micro-F1   
0.2 +Macro-F1 0.2 +Macro-F1 C 0 0 3 4 5 6 7 8 9 10 1 2 3 4 5 (a) The number of neighbors (b) The number of conv layers

each degenerated algorithm ignores the intra-view K-nearest neighbors, inter-view connections, and label semantic correlations, respectively. Table 3 records the experimental results on Emotions, Scene and Corel5k data sets. Specifically, compared with removing label semantic correlations, the model performance drops more significantly when removing intraview K-nearest neighbors or inter-view connections, which indicates that multi-view data fusion has more contribution to the performance of model than label semantic correlations. Meanwhile, VAMS significantly surpasses its three degenerated algorithms, which also strongly demonstrates the superiority of utilizing such three multi-granularity relationships simultaneously when learning from MVML data.

# Sensitivity Analysis

We study the sensitivity analysis of VAMS with regard to its two parameters: the number of intra-view neighbors $k$ and convolution layers $C$ . Figure 4(a) shows the performance changes as $k$ increases from 3 to 10, where the performance gradually improves and then slightly declines. In our experiments, we set $k$ to 5. Figure 4(b) illustrates the perfor

1 -Hamming Loss -Hamming Loss   
0.8 -Ranking Loss 0.8 Ranking Loss -Coverage +Coverage   
0.6 Avecro-Fe Precision 0.6 Avrage Precison -Macro-F1 Macro-F1   
0.4 0.4   
0.2 0.2 0 0 0 10 20304050607080 90100 0 10 203040 5060708090100 (a) Emotions (b) Corel5k

mance of VAMS with different numbers of convolution layers, where the optimal result is reached when $C$ is set to 3.

# Convergence Analysis

We conduct convergence analysis on Emotions and Corel5k data sets. Figure 5 illustrates the performance of VAMS as the number of epochs increases, where the Coverage results are normalized to make all metric results be characterized in a unified figure. According to Figure 5, the performance of VAMS gradually improves and becomes stable. Therefore, the convergence of VAMS is empirically demonstrated.

# Conclusion

In this paper, we proposed a new matching selection based MVML method, which integrates cross-view consensus information and individual-view specificity information into a unified framework. To the best of our knowledge, it is the first time to incorporate matching selection mechanism into MVML task, which avoids the separation of multi-view consistency and specificity, and significantly enhances the semantic characterization capability of the model.

# Acknowledgments

This work was supported by the National Key Research and Development Program of China (No. 2023YFB3107100), the National Natural Science Foundation of China (No. 62306020, 62203024, 62173286), the Young Elite Scientist Sponsorship Program by BAST (No. BYESS2024199), the R&D Program of Beijing Municipal Education Commission (No. KM202310005027), the Major Research Plan of National Natural Science Foundation of China (No. 92167102), and the Beijing Natural Science Foundation (No. L244009).