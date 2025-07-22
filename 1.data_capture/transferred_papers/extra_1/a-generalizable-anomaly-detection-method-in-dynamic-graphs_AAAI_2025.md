# A Generalizable Anomaly Detection Method in Dynamic Graphs

Xiao Yang1,2,3, Xuejiao Zhao1,2\*, Zhiqi Shen1\*

1 College of Computing and Data Science, Nanyang Technological University (NTU), Singapore 2 Joint NTU-UBC Research Centre of Excellence in Active Living for the Elderly (LILY), NTU, Singapore 3 Joint NTU-WeBank Research Centre on Fintech, NTU, Singapore yangxiao.cs $@$ gmail.com, {xjzhao,zqshen}@ntu.edu.sg

# Abstract

Anomaly detection aims to identify deviations from normal patterns within data. This task is particularly crucial in dynamic graphs, which are common in applications like social networks and cybersecurity, due to their evolving structures and complex relationships. Although recent deep learningbased methods have shown promising results in anomaly detection on dynamic graphs, they often lack of generalizability. In this study, we propose GeneralDyG, a method that samples temporal ego-graphs and sequentially extracts structural and temporal features to address the three key challenges in achieving generalizability: Data Diversity, Dynamic Feature Capture, and Computational Cost. Extensive experimental results demonstrate that our proposed GeneralDyG significantly outperforms state-of-the-art methods on four realworld datasets.

# Code — https://github.com/YXNTU/GeneralDyG

# Introduction

Graphs are extensively employed to model complex systems across various domains, such as social networks (Wang et al. 2019), human knowledge networks (Ji et al. 2021), e-commerce (Qu et al. 2020), and cybersecurity (Gao et al. 2020). Although the bulk of researches focus on static graphs, real-world graph data often evolves over time (Skarding, Gabrys, and Musial 2021). Taking knowledge networks as an example, there is new knowledge being added to the network every month, with connections between different concepts evolving over time. To model and analyze graphs where nodes and edges change over time, mining dynamic graphs gains increasing popularity in the graph analysis. Anomaly detection in dynamic graphs (Ma et al. 2021; Ho, Karami, and Armanfard 2024) is vital for identifying outliers that significantly deviate from normal patterns such as anomalous edges or anomalous nodes,including the detection of fraudulent transactions, social media spam, and network intrusions (Dou et al. 2020). By utilizing the temporal information and relational structures inherent in dynamic graphs, researchers can more effectively identify anomalies, thereby enhancing the security and integrity of various systems (Pourhabibi et al. 2020).

Recently, techniques based on deep learning have facilitated significant advancements in anomaly detection within dynamic graphs. For example, methods like GDN (Deng and Hooi 2021), StrGNN (Cai et al. 2021) focus on extracting structural information from graphs, while approaches such as LSTM-VAE (Park, Hoshi, and Kemp 2018) and TADDY (Liu et al. 2021) concentrate on capturing temporal information.In addition, self-supervised (Lee, Kim, and Shin 2024) and semi-supervised (Tian et al. 2023) methods have also been applied to dynamic graph anomaly detection.

Despite their improved performance, current deep learning-based methods lack the crucial generalizability (Brennan 1992) needed for dynamic graph tasks across different tasks or datasets. A model with strong generalization can adapt to different tasks without significant adjustments to its architecture or parameters, reducing the need for retraining or redesigning for new tasks(Bai, Ling, and Zhao 2022). Conversely, in anomaly detection, where identifying potential risks or issues is crucial, poor generalization may lead to missed critical anomalies in new scenarios, thereby diminishing the model’s reliability in realworld applications. Specifically, the inadequate encoding of anomalous events1 in existing methods results in poor generalization. Firstly, in the absence of raw event attributes, they fail to generate informative event encodings that accurately represent the properties of the events. For example, SimpleDyG (Wu, Fang, and Liao 2024) nearly discards all topological structure information, tokenizing only the nodes while ignoring the edges, which leads to the loss of critical structural information during node prediction tasks, making it unsuitable for node anomaly detection tasks and even less so for edge anomaly detection. The positional encoding method in TADDY (Liu et al. 2021) may not capture structural similarities and could fail to model the structural interactions between events, as demonstrated in SAT (Chen, O’Bray, and Borgwardt 2022). TADDY’s node positionspecific encoding may result in ambiguous structural information, leading to suboptimal results in node anomaly detection tasks. Furthermore, some methods, such as GDN (Deng and Hooi 2021), exhibit inadequate temporal information capture capabilities. For instance, GDN does not incorporate the information provided by specific time values when modeling temporal data, resulting in poor performance on time-sensitive datasets such as Bitcoin-Alpha and BitcoinOTC (Liu et al. 2021).

Developing a highly generalizable dynamic graph anomaly detection method presents several challenges, primarily in: 1. Data Diversity: Differences across dynamic graph datasets, such as topological structures and node and edge attributes, can be substantial. The method must identify and adapt to a wide range of feature distributions. 2. Dynamic Feature Capture: Anomalies in dynamic graphs may occur locally (e.g., anomalous behavior of specific nodes or edges) or globally (e.g., abnormal changes in network topology). The method must capture both local and global dynamic features. 3. Computational Cost: Dynamic graph anomaly detection often involves large-scale graph data, making computational resources and time efficiency significant challenges.

Hence, in this work, we propose a novel approach for anomaly detection named GeneralDyG, which addresses the three key challenges mentioned above and ensures generalizability in dynamic graph anomaly detection tasks. It ensures simplicity by sampling ego-graphs around anomalous events, then uses a novel GNN extractor to capture structural information, and finally employs a Transformer module to capture temporal information. Specifically, the main contributions of our work are:

• We design a novel GNN extractor, which embeds nodes, edges, and topological structures into the feature space. By alternating the message-passing perspective between nodes and edges, it performs graph convolution on both simultaneously. This ensures that GeneralDyG adapts to diverse feature distributions.   
• We introduce special tokens into the feature sequences to distinguish the hierarchical relationships between anomalous events, ensuring that the method captures global temporal information while maintaining focus on local dynamic features.   
• We design a novel ego-graph2 sampling method for training anomalous events instead of using the entire graph. This approach significantly reduces computational resources, enhancing the overall efficiency of the method.   
• We demonstrate the effectiveness of GeneralDyG on four benchmark datasets for detecting anomalous events, showing that it achieves better performance than state-ofthe-art anomaly detection methods.

# Related Work

# Anomaly Detection in Dynamic Graphs

Anomalies are infrequent observations that significantly deviate from the rest of the sample, such as data records or events. Dynamic graph anomaly detection primarily focuses on identifying unusual events within a dynamic graph (Ekle and Eberle 2024; Ho, Karami, and Armanfard 2024; Ma et al. 2021). Recently, deep learning methods have made significant advancements in anomaly detection for dynamic graphs. Modeling time series-related tasks as anomalous node detection in dynamic graphs is considered a viable approach (Su et al. 2019; Chen et al. 2022; Zhang, Zhang, and Tsung 2022; Dai and Chen 2022). Specifically, M-GAT employs a multi-head attention mechanism along with two relational attention modules—namely, intra-modal and intermodal attention—to explicitly model correlations between different modalities (Ding, Sun, and Zhao 2023). MTADGAT incorporates two parallel graph attention layers to capture the complex dependencies in multivariate time series across both temporal and feature dimensions (Zhao et al. 2020). GDN integrates structural learning with graph neural networks and leverages attention weights to enhance the explainability of detected anomalies (Deng and Hooi 2021). FuSAGNet optimizes reconstruction and forecasting by combining a Sparse Autoencoder with a Graph Neural Network to model multivariate time series relationships and predict future behaviors (Han and Woo 2022).

Detection of edge anomalies in dynamic graphs has also garnered increasing attention. Classical methods include the randomized algorithm SEDANSPOT (Eswaran and Faloutsos 2018) and the hypothesis-based approach Midas (Bhatia et al. 2020). Many recent methods have employed discrete approaches to address this task. For instance, Addgraph utilizes a GCN to extract graph structural information from slices, followed by GRU-attention (Zheng et al. 2019). StrGNN extracts $h$ -hop closed subgraphs centered on edges and employs GCN to model structural information on snapshots, with GRU capturing correlations between snapshots (Cai et al. 2021). Recently, SAD introduced a continuous dynamic approach for anomaly detection using a semisupervised method (Tian et al. 2023).

# Transformer on Dynamic Graphs

Transformers are a type of neural network that rely exclusively on attention mechanisms to learn representative embeddings for various types of data, as initially introduced in (Vaswani et al. 2017). Recent works have also applied Transformers to dynamic graph tasks. For instance, GraphERT pioneers the use of Transformers to seamlessly integrate graph structure learning with temporal analysis by employing a masked language model on sequences of graph random walks (Beladev et al. 2023). GraphLSTA captures the evolution patterns of dynamic graphs by effectively extracting and integrating both long-term and shortterm temporal features through a recurrent attention mechanism (Gao et al. 2023). Taddy employs a Transformer to handle diffusion-based spatial encoding, distance-based spatial encoding, and relative time encoding, subsequently deriving edge representations through a pooling layer to calculate anomaly scores (Liu et al. 2021). SimpleDyG reinterprets dynamic graphs as a sequence modeling problem and presents an innovative temporal alignment technique. This approach not only captures the intrinsic temporal evolution patterns of dynamic graphs but also simplifies their modeling process (Wu, Fang, and Liao 2024).

# Preliminaries

Notations. A continuous-time dynamic graph (CTDG) is used to represent relational data in evolving systems. A CTDG is defined as $\mathcal { G } = ( \nu , \mathcal { E } )$ , where $\nu$ is the set of nodes that participate in temporal edges, and $\mathcal { E }$ is a chronologically ordered series of edges. Each edge ${ \boldsymbol { \delta } } ( t ) ~ = ~ ( v _ { i } , v _ { j } , { \overline { { t } } } , e _ { i j } { \dot { ) } }$ represents an interaction from node $\boldsymbol { v } _ { i }$ to node $v _ { j }$ at time $t$ with an associated feature $e _ { i j }$ . The node attributes for nodes $v _ { i } , v _ { j } \in \mathcal { V }$ are denoted by $\bar { x } _ { v _ { i } } , x _ { v _ { j } } \in \mathbb { R } ^ { d }$ , and the node attributes for all nodes are stored in $\boldsymbol { \mathcal { X } } \in \mathbb { R } ^ { n \times d }$ . Additionally, the edge attributes for edges $e _ { i j } \ \in \mathcal { E }$ are denoted by $y _ { e _ { i j } } \in \mathbb { R } ^ { d }$ , and the edge attributes for all edges are stored in $\bar { \boldsymbol { y } } \in \mathbb { R } ^ { m \times d }$ , where $n$ is the number of nodes and $m$ is the number of edges in the CTDG. In this paper, we explore a method called GeneralDyG for handling node-level and edge-level anomalies. Therefore, in the following text, we treat nodes $\nu$ and edges $\mathcal { E }$ collectively as anomaly events $\mathcal { A }$ . Similarly, we consider node features $\chi$ and edge features $y$ together as anomaly features $\mathcal { Z }$ .

Transformer on CTDG. While Graph Neural Networks (GNNs) directly leverage the inherent structure of graphs, Transformers take a different approach by inferring relationships between nodes using their attributes rather than the explicit graph structure (Dwivedi and Bresson 2020). Transformer treats the dynamic graph as a collection of edges, utilizing the self-attention mechanism to identify similarities between them. The architecture of the Transformer(Fang et al. 2023a,b) consists of two fundamental components: a self-attention module and a feed-forward neural network.

In the self-attention module, the input anomaly features $\mathcal { Z }$ are initially projected onto the query $( Q )$ , key $( K )$ , and value $( V )$ matrices through linear transformations, such that $Q = \mathcal { Z } W _ { Q }$ , $K = \mathcal { Z } W _ { K }$ , and $V = \mathcal { Z } W _ { V }$ , respectively. The self-attention can then be computed as follows:

$$
\mathrm { A t t n } ( \mathcal { Z } ) = \mathrm { s o f t m a x } \left( \frac { Q K ^ { T } } { \sqrt { d _ { \mathrm { o u t } } } } \right) V \in \mathbb { R } ^ { ( m + n ) \times d _ { \mathrm { o u t } } } .
$$

To address dynamic graph tasks, multiple Transformer layers can be stacked to build a model that provides nodelevel representations of the graph (Wang et al. 2021). However, due to the permutation invariance of the self-attention mechanism, the Transformer generates identical representations for nodes with the same attributes, regardless of their positions or surrounding structures within the graph. This characteristic necessitates the incorporation of positional and contextual information into the Transformer, typically achieved through positional encoding (Cong et al. 2021; Sun et al. 2022).

Absolute encoding. Absolute encoding involves adding or concatenating positional or structural representations of the graph to the input node features before feeding them into the main Transformer model. Examples of such encoding methods include Laplacian positional encoding (Dwivedi and Bresson 2020), Random Walk Positional Encoding (Dwivedi et al. 2021), and Node Encoding (Liu et al. 2021). A key limitation of these methods is that they typically fail to capture the structural similarity between nodes and their neighborhoods, thereby not effectively leveraging the graph’s structural information.

Problem Definition. The goal of this paper is to detect anomalous edges and nodes at each timestamp. Based on the previously mentioned notations, we model anomaly detection in dynamic graphs as a task of computing anomaly scores.

Definition 1. Given a dynamic graph $\mathcal { G }$ , where each $\mathcal { G } _ { t } ~ =$ $( \nu _ { t } , \mathcal { E } _ { t } )$ represents the graph at timestamp $t$ , the goal of anomaly detection is to identify unusual edges and nodes within this evolving structure. For each edge $e \in \mathcal { E } _ { t }$ and each node $v \in \mathcal V _ { t }$ , the objective is to compute an anomaly score $f ( e )$ and $f ( v )$ , respectively, where $f$ is a learnable anomaly score function. The anomaly score quantifies the degree of abnormality for both edges and nodes, with a higher score $f ( e )$ or $f ( v )$ indicating a greater likelihood of anomaly for edge $e$ or node $v$ .

Building on previous research, we adopt an unsupervised approach for anomaly detection in dynamic graphs. During training, all edges and nodes are considered normal. Binary labels indicating anomalies are provided during the testing process to assess the performance of the algorithms. Specifically, a label $y _ { e } = 1$ signifies that $e$ is anomalous, whereas $y _ { e } = 0$ denotes that $e$ is normal. Similarly, a label $y _ { n } = 1$ indicates that a node is anomalous. It is important to note that anomaly labels are often imbalanced, with the number of normal edges and nodes typically being much greater than the number of anomalous ones.

# Methodology

In this section, we introduce the general framework of our approach, which consists of three main components: Temporal ego-graph sampling, Temporal ego-graph GNN extractor, and Temporal-Aware Transformer. An overview of the proposed framework is illustrated in Figure 1. Initially, we extract ego-graphs at the level of each anomaly event, capturing $k$ -hop temporal dynamics. These temporal ego-graphs are then transformed into anomaly feature sequences, preserving their temporal and structural order, as demonstrated in Figure 1(a). To fully understand the structural information of these sequences, they are processed through a GNN model to extract the structural details of the temporal egographs, as depicted in Figure 1(b). Finally, both the original sequence features and the structure-enriched sequence features are fed into the Transformer to evaluate the anomaly detection task, as shown in Figure 1(c).

# Temporal ego-graph sampling

Unlike conventional methods that map dynamic graphs into a series of snapshots to obtain tokens, we use a more lightweight approach by employing anomalous events as tokens for the Transformer. Additionally, to acquire the contextual representation and hierarchical information of these anomalous events, we extract the temporal $k$ -hop ego-graph of each event to capture historical interaction information across different structures.

Specifically, we denote $\ u _ { a _ { i } } ~ \in ~ \mathcal { A }$ as an event in $\mathcal { G }$ . For each event $a _ { i }$ , we utilize a $k$ -hop algorithm to extract the historically interacted events and construct a series of $k$ -hop ego-graphs centered around $a _ { i }$ , representing subsets of the largest $k$ -hop ego-graph. Explicitly, we denote the temporal $k$ -hop ego-graph for $a _ { i }$ as a chronologically ordered sequence $\begin{array} { r l } { w _ { i } } & { { } = } \end{array}$ sa $m p l i n g ( \langle a _ { i } ^ { 1 } \rangle , \langle a _ { i } ^ { 1 } , a _ { i } ^ { 2 } , a _ { i } ^ { 3 } \rangle , \dots , \langle a _ { i } ^ { 1 } , a _ { i } ^ { 2 } , a _ { i } ^ { 3 } , \dots , a _ { i } ^ { | w _ { i } | } \rangle )$ , where $\left| w _ { i } \right|$ is the number of previously interacting events, and the maximum value of $\left| w _ { i } \right|$ is the total number of events that have interacted with $a _ { i }$ . Note that $\forall 1 \leq j < j ^ { \prime } \leq | w _ { i } |$ , $a _ { i } ^ { j }$ and $a ^ { j ^ { \prime } } i$ represent historical interactions $( a _ { i } , a _ { i } ^ { j } , e i , j )$ and $( a _ { i } , a ^ { j ^ { \prime } } i , e i , j ^ { \prime } )$ , respectively, such that $e _ { i , j } \leq e _ { i , j ^ { \prime } }$ .

![](images/16fa4408772ecd02308c9fdfb9eb67bc480d6d22d5bcf0dd791f5affd384cef5.jpg)  
Figure 1: The proposed generalizable anomaly detection framework. GeneralDyG consists of three main components: (a)Temporal ego-graph sampling. (b)Temporal ego-graph GNN extractor. (c)Temporal-Aware Transformer

When implementing feature sequences sorted by time, it is crucial to simultaneously consider the hierarchical information introduced by the $k$ -hop algorithm. Specifically, for the central event $a _ { i }$ , the set of events $a _ { i ; k } ^ { j }$ extracted by the $k$ -hop algorithm exhibits greater similarity compared to the saes  hofeyevsehnatrse $a _ { i ; k + 1 } ^ { j }$ mextsrhaoctrtedstbpyathet $( k { + } 1 )$ c-ehnotpraal pgoirinth $a _ { i }$ To better capture this hierarchical information, we draw inspiration from natural language processing methods and add special tokens to the feature sequence. These tokens ensure that the event sets between two special tokens maintain a chronological order. During training, the Transformer module and GNN extractor can receive the following input:

$$
\begin{array} { r l r } & { } & { \mathrm { i n p u t } _ { i } = \langle | K H S | \rangle , a _ { i } , \langle | K H S | \rangle , a { i } ; 1 ^ { 1 } , a _ { i ; 1 } ^ { 2 } , \dotsc , a _ { i ; 1 } ^ { | a _ { i ; 1 } | } , \dotsc } \\ & { } & { \langle | K H S | \rangle , a _ { i ; k } ^ { 1 } , a _ { i ; k } ^ { 2 } , \dotsc , a _ { i ; k } ^ { | a _ { i ; k } | } , \langle | K H S | \rangle , \qquad ( 2 ) } \end{array}
$$

where $\langle | K H S | \rangle$ is a special token signifying the beginning and end of the input hierarchical sequence. Specifically, adding such special tokens helps the model recognize and differentiate between the hierarchical layers of the ego-graph. It should be noted that we use sampled egographs here to enhance the model’s generalization capability. Therefore, the raw features obtained by the Transformer module and GNN extractor are denoted as $z _ { i }$ , where $z _ { i }$ is a subset of inputi.

# Temporal ego-graph GNN extractor

A practical approach to extracting local structural information at an event $a _ { i }$ is to apply an existing GNN model to the input graph with event feature sequences $z _ { i }$ , and utilize the output representation at $a _ { i }$ as the ego-graph representation $\varphi ( \bar { z } _ { i } )$ . It is important to highlight that, to showcase the flexibility of our model, the GNN model employed here should be both straightforward and capable of simultaneously processing node features $\chi$ and edge features $y$ . Formally, we denote the selected GNN model with $\kappa$ layers applied to $k$ - hop ego-graphs $k$ -DG as $\mathrm { G N N } _ { k } ^ { \kappa }$ . The output representation $\varphi ( z _ { i } )$ can be expressed as:

$$
\varphi ( z _ { i } ) = \mathbf { G } \mathbf { N } \mathbf { N } _ { k } ^ { \kappa } ( z _ { i } ) .
$$

Next, we discuss the choice of the $\mathrm { G N N } _ { k } ^ { \mathcal { K } }$ model. When the dataset information is known prior to anomaly event prediction—such as in cases where the CTDG consists solely of node features—a conventional GNN model like GCN, GAT, or GIN can be effectively utilized to extract ego-graph structural information. However, for CTDGs with diverse attributes, including both node and edge features, we introduce the Temporal Edge-Node Based Structure Extractor GNN (TensGNN). TensGNN is specifically designed to accommodate more complex scenarios by concurrently processing both types of features.

TensGNN encodes events by alternately applying node and edge layers, thereby embedding events into a shared feature space. Specifically, TensGNN employs operations analogous to spectral graph convolution for message passing on events. The node Laplacian-adjacency matrix with self-loops is defined as:

$$
\bar { A } _ { v } = D _ { v } ^ { \frac { 1 } { 2 } } \left( A _ { v } + I _ { v } \right) D _ { v } ^ { \frac { 1 } { 2 } } ,
$$

where $D _ { v }$ is the diagonal degree matrix of $A _ { v } + I _ { v }$ , and $I _ { v }$ is the identity matrix. The node-level propagation rule for node features in the $( K + 1 )$ -th layer is defined as:

$$
H _ { v } ^ { ( K + 1 ) } = \sigma \left( T ^ { T } H _ { e } ^ { ( K ) } W _ { e } ^ { \prime } \odot \bar { A } _ { v } H _ { v } ^ { ( K ) } W _ { v } \right) ,
$$

where $\sigma$ represents the activation function, the matrix $T \in$ $\mathbb { R } ^ { N _ { v } \times N _ { e } }$ is a binary transformation matrix, with $T _ { i j }$ indicating whether edge $j$ connects to node $i$ . The symbol $\odot$ represents the Hadamard product. $W _ { e } ^ { \prime }$ and $W _ { v }$ are learnable parameters for edges and nodes, respectively. Similarly, the Laplacianized edge adjacency matrix is defined as:

$$
\bar { A } _ { e } = D _ { e } ^ { \frac { 1 } { 2 } } \left( A _ { e } + I _ { e } \right) D _ { e } ^ { \frac { 1 } { 2 } } ,
$$

where $D _ { e }$ is the diagonal degree matrix of $A _ { e } + I _ { e }$ , and $I _ { e }$ is the identity matrix. The propagation rule for edge features is then defined as:

$$
H _ { e } ^ { ( K + 1 ) } = \sigma \left( T ^ { T } H _ { v } ^ { ( K ) } W _ { v } ^ { \prime } \odot \bar { A } _ { e } H _ { e } ^ { ( K ) } W _ { e } \right) .
$$

Here, the matrix $T$ is defined analogously to that in Equation 5, with $\boldsymbol { W } _ { v } ^ { \prime }$ and $W _ { e }$ representing the learnable weights for the nodes and edges, respectively. TensGNN alternates between stacking node layers and edge layers to iteratively refine the embeddings of both types of events. Specifically, to derive the final encoding of nodes, the last layer before the output is a node layer. Conversely, to obtain the final encoding of edges, the last layer before the output is an edge layer.

# Temporal-Aware Transformer

To enhance the Transformer’s understanding of the topological structure of the temporal ego-graph while preserving the original event features, we overlay the topological structure information onto the Query and Key, while retaining the original event features as the Value. This approach allows the model to leverage structural information for the attention mechanism while maintaining the integrity of the original feature values for effective representation. Formally, for the event feature to be predicted, $z _ { i } \in \mathcal { Z }$ , we adopt the method proposed in (Mialon et al. 2021) and rewrite the selfattention as kernel smoothing. The final embedding calculation is then given by:

$$
\mathrm { A t t n } ( z _ { i } ) = \sum _ { z _ { j } \in k - D G } \frac { \mathcal { F } _ { \mathrm { e x p } } ( z _ { i } , z _ { j } ) } { \sum _ { z _ { w } \in k - D G } \mathcal { F } _ { \mathrm { e x p } } ( z _ { i } , z _ { w } ) } \mathbf { w } _ { V } z _ { j } ,
$$

where $\mathbf { w } _ { V }$ is the linear value function of the original event feature $z _ { i }$ , and $\mathcal { F } e x p$ is an exponential kernel (nonsymmetric), parameterized by $\mathbf { w } _ { Q }$ and $\mathbf { w } _ { K }$ :

$$
\begin{array} { r l } & { { \mathscr F } e x p ( x , x ^ { \prime } ) : = \exp \left( \frac { \left. \mathbf { w } _ { Q } x , \mathbf { w } _ { K } x ^ { \prime } \right. } { \sqrt { d _ { o u t } } } \right) , } \\ & { \qquad \mathbf { w } _ { V } = \mathbf { W } z _ { i } + b , } \\ & { \qquad \mathbf { w } _ { Q } = \mathbf { W } \varphi ( z _ { i } ) + b , } \\ & { \qquad \mathbf { w } _ { K } = \mathbf { W } \varphi ( z _ { i } ) + b , } \end{array}
$$

where $\langle \cdot , \cdot \rangle$ denotes the dot product. By optimizing the objective function, we obtain the final embeddings for each anomaly feature $z _ { i }$ . These final embeddings are then fed into the scoring module to compute the anomaly scores. It is important to note that the scoring modules for node-level and edge-level anomalies differ in the datasets used in this paper. For edge-level anomalies, we directly use the final output embedding from the training process as the anomaly score. Conversely, for node-level anomalies, the final output consists of a set of binary labels indicating whether each time step is anomalous, which serves as the final anomaly score.

# Experiments

# Experimental Setup

Datasets. We use four real-world datasets, categorized into two types: Node-Level and Edge-Level anomaly detection tasks. For Node-Level, we utilize SWaT (Secure Water Treatment), a small-scale Cyber-Physical system managed by Singapore’s Public Utility Board, and WADI (Water Distribution), an extension of SWaT that includes a more extensive water distribution network. Both datasets provide data from normal operations and controlled attack scenarios to simulate real-world anomalies. For Edge-Level, we employ Bitcoin-Alpha and Bitcoin-OTC, which are trust networks of Bitcoin users trading on platforms from www.btcalpha.com and www.bitcoin-otc.com, respectively. In these datasets, nodes represent users, and edges indicate trust ratings between them, capturing interactions and trust dynamics within the Bitcoin trading community.

Experimental Design. In our experiments, The settings for Bitcoin-Alpha and Bitcoin-OTC are identical to those used in TADDY (Liu et al. 2021). We inject anomalies into the test set at proportions of $1 \%$ , $5 \%$ , and $10 \%$ . SWaT and WADI are identical to those used in GDN (Deng and Hooi 2021). AUC3, $\mathrm { A P ^ { 4 } }$ and $\mathrm { F } 1 ^ { 5 }$ are used as the primary metrics to evaluate the performance of the proposed GeneralDyG and baselines.

Baselines. We evaluated GeneralDyG against 20 advanced baselines, which are classified into two categories: graph embedding methods and anomaly detection methods. A detailed description of the baselines can be found in the Appendix.

• Graph Embedding Methods: node2vec (Grover and Leskovec 2016), DeepWalk (Perozzi, Al-Rfou, and

<html><body><table><tr><td rowspan="3">Methods</td><td colspan="6">Bitcoin-Alpha</td><td colspan="6">Bitcoin-OTC</td></tr><tr><td colspan="2">1%</td><td colspan="2">5%</td><td colspan="2">10%</td><td colspan="2">1%</td><td colspan="2">5%</td><td colspan="2">10%</td></tr><tr><td>AUC</td><td>AP</td><td>AUC</td><td>AP</td><td>AUC</td><td>AP</td><td>AUC</td><td>AP</td><td>AUC</td><td>AP</td><td>AUC</td><td>AP</td></tr><tr><td>node2vec</td><td>69.10</td><td>9.17</td><td>68.02</td><td>7.31</td><td>67.85</td><td>9.95</td><td>69.51</td><td>8.31</td><td>68.83</td><td>6.45</td><td>67.45</td><td>4.77</td></tr><tr><td>DeepWalk</td><td>69.85</td><td>8.56</td><td>68.74</td><td>9.68</td><td>67.93</td><td>10.78</td><td>74.23</td><td>10.58</td><td>73.56</td><td>9.41</td><td>72.87</td><td>8.22</td></tr><tr><td>TGAT</td><td>85.32</td><td>11.36</td><td>84.16</td><td>11.08</td><td>83.98</td><td>12.05</td><td>88.87</td><td>16.87</td><td>87.59</td><td>15.24</td><td>87.55</td><td>15.37</td></tr><tr><td>TGN</td><td>86.92</td><td>13.00</td><td>86.78</td><td>16.85</td><td>86.21</td><td>17.00</td><td>84.33</td><td>11.33</td><td>83.49</td><td>11.25</td><td>83.47</td><td>10.79</td></tr><tr><td>ADDGRAPH</td><td>83.41</td><td>13.21</td><td>84.70</td><td>13.01</td><td>83.69</td><td>14.28</td><td>86.00</td><td>16.04</td><td>84.98</td><td>15.21</td><td>84.77</td><td>14.21</td></tr><tr><td>StrGNN</td><td>85.74</td><td>12.56</td><td>86.67</td><td>13.99</td><td>86.27</td><td>14.68</td><td>90.12</td><td>18.34</td><td>87.75</td><td>18.68</td><td>88.36</td><td>18.10</td></tr><tr><td>TADDY</td><td>94.51</td><td>16.51</td><td>93.41</td><td>18.32</td><td>94.23</td><td>19.67</td><td>94.55</td><td>16.10</td><td>93.40</td><td>18.47</td><td>94.25</td><td>18.92</td></tr><tr><td>SAD</td><td>90.69</td><td>19.99</td><td>90.55</td><td>21.08</td><td>90.33</td><td>22.99</td><td>91.88</td><td>26.32</td><td>90.99</td><td>27.33</td><td>90.04</td><td>26.79</td></tr><tr><td>SLADE</td><td>90.32</td><td>18.78</td><td>89.99</td><td>22.02</td><td>88.71</td><td>24.41</td><td>91.53</td><td>20.32</td><td>91.24</td><td>22.11</td><td>91.01</td><td>20.04</td></tr><tr><td>GeneralDyG</td><td>94.01</td><td>24.00</td><td>95.41</td><td>24.02</td><td>96.28</td><td>26.73</td><td>94.66</td><td>27.89</td><td>94.86</td><td>29.97</td><td>95.59</td><td>27.13</td></tr></table></body></html>

Table 1: Anomaly detection performance comparison on Edge-Level datasets. The best performing method in each experimen is in bold and the second-best method is indicated with underlining.

Skiena 2014), TGAT (Xu et al. 2020), TGN (Rossi et al.   
2020).

• Anomaly Detection Methods: ADDGRAPH (Zheng et al. 2019), StrGNN (Cai et al. 2021), TADDY (Liu et al. 2021), SAD (Tian et al. 2023), SLADE (Lee, Kim, and Shin 2024), PCA (Shyu et al. 2003), KNN (Angiulli and Pizzuti 2002), GDN (Deng and Hooi 2021), BTAD (Ma, Han, and Zhou 2023), GRN-100 (Tang et al. 2023), DAGMM (Zong et al. 2018), MST-GAT (Ding, Sun, and Zhao 2023), FuSAGNet (Han and Woo 2022), LSTMVAE (Park, Hoshi, and Kemp 2018), MTAD-GAT (Zhao et al. 2020).

# Overall Performance

Edge Level. We compared our methods, GeneralDyG, with nine strong edge-level baseline methods, as shown in Table 1. Our methods consistently outperformed the baselines across both Bitcoin-Alpha and Bitcoin-OTC datasets. The baselines, lacking sufficient structural or temporal information, did not achieve state-of-the-art results. Specifically, GeneralDyG demonstrated an average AUC improvement of approximately $3 . 2 \%$ and $4 . 5 \%$ , respectively, compared to the best-performing baseline on the Bitcoin-Alpha dataset. In terms of Average Precision (AP), GeneralDyG achieved a significant improvement, with up to $24 \%$ in the $1 \%$ anomaly detection setting, representing a $1 9 . 8 \%$ increase over the best-performing baseline.

On the Bitcoin-OTC dataset, GeneralDyG also exhibited substantial gains, with an average AUC increase of about $3 . 6 \%$ and an AP improvement of up to $2 0 . 2 \%$ over the baselines. This demonstrates that our methods are more effective in generalizing and capturing the temporal dynamics necessary for robust anomaly detection in these datasets. Node Level. We compared our methods, GeneralDyG, with ten strong node-level baseline methods, as shown in Table 2. Our methods generally outperformed the baselines. Specifically, GeneralDyG achieved the highest F1 score on the SWaT dataset with $8 5 . 1 9 \%$ , surpassing the second-best method, FuSAGNet, by $1 . 8 \%$ . On the WADI dataset, GeneralDyG reached an F1 score of $6 0 . 4 3 \%$ , which is slightly below FuSAGNet’s $6 0 . 7 0 \%$ , but still demonstrates competitive performance.

Table 2: Anomaly detection F1 scoring comparison on Node-Level datasets. The best performing method in each experiment is in bold and the second-best method is indicated with underlining.   

<html><body><table><tr><td>Methods</td><td>SWaT</td><td>WADI</td></tr><tr><td>PCA KNN</td><td>23.16 7.83</td><td>9.35 7.75</td></tr><tr><td>GDN</td><td>80.82</td><td>56.92</td></tr><tr><td>BTAD</td><td>81.43</td><td>53.77</td></tr><tr><td>GRN-100</td><td>74.96</td><td>48.28</td></tr><tr><td>DAGMM</td><td>39.37</td><td>36.09</td></tr><tr><td>MST-GAT</td><td>83.55</td><td>60.31</td></tr><tr><td>FuSAGNet</td><td>83.69</td><td>60.70</td></tr><tr><td>LSTM-VAE</td><td>73.85</td><td>24.82</td></tr><tr><td>MTAD-GAT</td><td>31.71</td><td>16.94</td></tr><tr><td>GeneralDyG</td><td>85.19</td><td>60.43</td></tr></table></body></html>

The baselines, particularly those lacking robust temporal modeling capabilities like PCA and KNN, showed significantly lower F1 scores, with KNN performing the worst on both datasets. Compared to these methods, GeneralDyG shows a notable improvement of approximately $58 \%$ on SWaT and $5 3 \%$ on WADI in F1 score. Overall, these results highlight that our methods are better at generalizing and capturing the temporal dynamics necessary for effective anomaly detection in node-level datasets.

# Ablation Study

We conducted an ablation study to assess the contribution of each component in the proposed GeneralDyG, as detailed below:

• w/o ego-graph. This variant omits the temporal egograph sampling process and directly uses the entire graph as input features.   
• w/o TensGNN. This variant removes the GNN extractor, thereby omitting the extraction of structural information from the events.

Table 3: The performance of GeneralDyG and its variants on both Node-Level and Edge-Level datasets.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="2">Bitcoin-Alpha</td><td>WADI</td></tr><tr><td>AUC</td><td>AP</td><td>F1</td></tr><tr><td>GeneralDyG</td><td>96.28</td><td>26.73</td><td>60.43</td></tr><tr><td>w/o ego-graph</td><td>96.01</td><td>19.33</td><td>59.45</td></tr><tr><td>w/o TensGNN</td><td>92.02</td><td>22.63</td><td>55.13</td></tr><tr><td>w/o Transformer</td><td>93.71</td><td>20.20</td><td>58.46</td></tr></table></body></html>

• w/o Transformer. This variant excludes the Transformer module, thus omitting the extraction of temporal information from the events.

The ablation study results in Table 3 highlight the significance of each component in GeneralDyG. Removing the temporal ego-graph sampling (w/o ego-graph) results in a decrease of AUC by $0 . 2 7 \%$ and AP by $2 7 . 9 \%$ on BitcoinAlpha, and a reduction in F1 score by $1 . 6 \%$ on WADI, indicating its critical role in capturing temporal dependencies. Excluding the GNN extractor (w/o TensGNN) leads to a significant decrease in AUC by $4 . 2 6 \%$ , AP by $1 5 . 8 \%$ on Bitcoin-Alpha, and F1 score by $8 . 6 \%$ on WADI, underscoring the importance of structural information. Removing the Transformer module (w/o Transformer) results in a decrease of AUC by $2 . 5 7 \%$ , AP by $2 4 . 6 \%$ on Bitcoin-Alpha, and F1 score by $3 . 3 \%$ on WADI, emphasizing the need for temporal information processing. These results confirm that each component is crucial for achieving optimal performance.

# How to Set Up the Optimal GeneralDyG

![](images/c6c47b66b4b5dc31cc6eb55eab7e9df81a51a9165cfa7dd5605e0f3528c85d1b.jpg)  
Figure 2: Effect of Parameters $k$ and $\kappa$ on Bitcoin-Alpha

The heatmap in Figure 2 illustrates the impact of the parameters $k$ and $\kappa$ on model performance for the BitcoinAlpha dataset. It indicates that higher values of $\kappa$ (the number of layers in the TensGNN) generally lead to decreased performance, suggesting that having too many layers can be detrimental to the model’s effectiveness. This could be due to overfitting or increased model complexity without corresponding gains in performance.

On the other hand, the parameter $k$ (which controls the temporal ego-graph sampling) has a less pronounced effect on performance. While increasing $k$ does affect the results, it mainly impacts the training process due to the additional parameters it introduces.

Thus, the optimal setup should aim for a balance: choosing a modest number of layers $( \kappa )$ to avoid overfitting while selecting an appropriate $k$ that provides sufficient temporal information without excessively complicating the model. This balance will help in achieving both efficient training and robust performance.

Generalizable Analysis   
Table 4: Generalizable analysis on Node-Level and EdgeLevel tasks   

<html><body><table><tr><td rowspan="2">Node-Level Method</td><td colspan="2">Bitcoin-Alpha</td><td rowspan="2">Edge-Level Method</td><td>WADI</td></tr><tr><td>AUC</td><td>AP</td><td>F1</td></tr><tr><td>GeneralDyG</td><td>96.28</td><td>26.73</td><td>GeneralDyG</td><td>60.43</td></tr><tr><td>GDN</td><td>83.84</td><td>13.28</td><td>TADDY</td><td>40.05</td></tr><tr><td>MST-GAT</td><td>86.66</td><td>18.97</td><td>SimpleDyG</td><td>33.24</td></tr><tr><td>FuSAGNet</td><td>87.76</td><td>20.01</td><td>SAD</td><td>36.75</td></tr></table></body></html>

In Table 4, GeneralDyG demonstrates its strong generalizability across different types of tasks. Specifically, GeneralDyG consistently outperforms the baseline methods that were evaluated in a mismatched dataset context. For instance, when the edge-level baselines are applied to the node-level dataset (WADI), their performance significantly drops, with metrics such as AUC and F1 score showing substantial declines compared to GeneralDyG. Similarly, nodelevel baselines tested on the edge-level dataset (BitcoinAlpha) exhibit poor performance, further emphasizing their lack of generalizability.

GeneralDyG, on the other hand, maintains high performance across both types of datasets, showcasing its robustness and adaptability. This indicates that GeneralDyG is capable of effectively handling both node-level and edge-level tasks, whereas the baseline methods exhibit considerable performance degradation when faced with different dataset types. These results underline the superior generalizability of GeneralDyG, as it maintains stable and effective performance across diverse scenarios where other methods fail to deliver consistent results.

# Conclusion

In this work, we introduced a novel approach for anomaly detection in dynamic graphs called GeneralDyG, which effectively addresses the challenges of data diversity, dynamic feature capture, and computational cost, thereby demonstrating the generalizability of our method. GeneralDyG achieves this by mapping node, edge, and topological structure information into the feature space, incorporating hierarchical tokens, and sampling temporal ego-graphs to efficiently capture dynamic features. GeneralDyG excels across multiple benchmarks, demonstrating its effectiveness and high performance. For future work, we can build on this work to explore the interpretability of anomaly detection in dynamic graphs, providing more robust theoretical support.

# Acknowledgments

This research is supported by the Joint NTU-WeBank Research Centre on Fintech, Nanyang Technological University, Singapore. It is also supported by the Joint NTU-UBC Research Centre of Excellence in Active Living for the Elderly (LILY) and the College of Computing and Data Science (CCDS) at NTU Singapore. This work is partially supported by the Wallenberg Al, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation.