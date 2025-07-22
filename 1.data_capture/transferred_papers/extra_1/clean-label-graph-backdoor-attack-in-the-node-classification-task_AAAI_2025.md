# Clean-Label Graph Backdoor Attack in the Node Classification Task

Hui Xia, Xiangwei Zhao, Rui Zhang, Shuo Xu, Luming Wang

Ocean University of China xiahui $@$ ,zhaoxiangwei@stu.,zhangrui $0 5 0 4 \textcircled { a }$ stu.,xushuo@stu.,wangluming@stu. ouc.edu.cn

# Abstract

Graph neural networks (GNNs) have achieved impressive results in various graph learning tasks. Backdoor attacks pose a significant threat to GNNs, with a focus on dirty-label attacks. However, these attacks often necessitate the inclusion of blatantly incorrect inputs into the training set, rendering them easily detectable through simple filtering. In response to this challenge, we introduce Clean-Label Graph Backdoor Attack (CGBA). The majority of features in the generated poisoned nodes align with their true labels, significantly enhancing the difficulty of detecting the attack. Firstly, leveraging the uncertainty inherent in the GNNs, we develop a low-budget strategy for selecting poisoned nodes. This approach focuses on nodes in the target class with uncertain and low-degree classifications, allowing for efficient attacks within a limited budget while mitigating the impact on other clean nodes. Secondly, we present an innovative strategy for generating feature triggers. By boosting the confidence of poisoned samples in the target class, this tactic establishes a robust association between the trigger and the target class, even without modifying the labels of poisoned nodes. Additionally, we incorporate two constraints to reduce disruption to the graph structure. In conclusion, comprehensive experimental results unequivocally showcase CGBA’s exceptional attack performance across three benchmark datasets and four GNNs models. Notably, the attack targeting the GraphSAGE model attains a $100 \%$ success rate, accompanied by a marginal benign accuracy drop of no more than $0 . { \overset { } { 5 } } \%$ .

# Introduction

Graph Neural Networks (GNNs) have achieved remarkable success in processing graph-structured data (Kipf and Welling 2016a; Velickovic et al. 2017), demonstrating widespread applications across various domains, e.g., drug design (Wu et al. 2022), recommendation systems (Wu et al. 2019), and the financial sector (Wang et al. 2019). GNNs, specifically, aggregate information from neighbors based on graph features and topology to update node representations, enabling predictions and classifications for nodes or entire graphs (Kipf and Welling 2016a). However, akin to Convolutional Neural Networks (CNNs) (Cheng et al. 2021; Zhang et al. 2022), GNNs are vulnerable to backdoor attacks (Xi et al. 2021; Zhang et al. 2021; Xu and Picek 2022; Dai et al.

2023). The exploration of GNNs backdoor attacks is essential for uncovering potential vulnerabilities in GNNs, providing crucial insights and guidance for constructing more secure GNNs models in the future.

The standard backdoor attack (termed dirty-label backdoor attacks hereinafter) (Chen et al. 2017) involves attaching triggers to poisoned samples during the training phase and assigning a target label to them, thereby associating the trigger with the target class. However, this method has a significant weakness: the poisoned samples with attached triggers are conspicuously mislabeled, making it relatively easy to classify these samples as outliers through simple filtering during the model inference phase, prompting further investigation of the attack (Xu and Picek 2022). To overcome this limitation, (Turner, Tsipras, and Madry 2018) introduced the concept of a clean-label backdoor attack. In this form of attack, only a small number of inputs from the target class are poisoned, without any modification to their labels, making it considerably more challenging to detect the presence of poisoned samples. Despite the effectiveness of clean-label backdoor attacks, there is currently no such attack specifically designed for GNNs node classification tasks. To address this gap, we propose a solution in this study.

The implementation of clean-label backdoor attacks is more challenging compared to dirty-label backdoor attacks. Firstly, existing GNNs dirty-label backdoor attacks often involve randomly poisoning a larger number of non-target class nodes, requiring a larger attack budget (Xu, Xue, and Picek 2021; Chen et al. 2023). In contrast, clean-label backdoor attacks can only select a small number of nodes within the target class for the attack. Secondly, existing GNNs dirty-label backdoor attacks mostly use random triggers (Zhang et al. 2021; Xu, Xue, and Picek 2021) and rely on the modification of labels of the poisoned samples to establish a strong association between the triggers and the target label. However, clean-label attacks cannot modify the labels of the poisoned samples. In summary, we face the following challenges: (1) Choosing which nodes to designate as poisoned nodes under a lower budget. (2) The need to establish a firm connection between triggers and the target label under clean-label conditions.

To address the challenges mentioned earlier, we present the Clean-Label Graph Backdoor Attack (CGBA). In response to challenge (1), we’ve developed a tailored strategy for selecting poisoned nodes in clean-label attacks to optimize the use of node resources. By leveraging the principle of uncertainty in GNNs, we pinpoint nodes with ambiguous classification outcomes for the attack. Additionally, we introduced a constraint to bias the selection toward nodes with lower degrees, minimizing the impact on other clean nodes. For the poisoned features, we drew inspiration from (Xu, Xue, and Picek 2021), utilizing graph interpretability methods to select the most important features within the poisoned nodes for attaching triggers. To tackle challenge (2), we opt for node feature triggers to poison the data and introduce a novel feature trigger generation strategy. This strategy enhances the connection between triggers and the target label by boosting the confidence of the poisoned samples within the target class. Additionally, to minimize disruption to the graph structure, we impose constraints on the similarity between poisoned nodes and their neighboring nodes, as well as the similarity between the poisoned nodes before and after the poisoning process. These measures decrease the likelihood of detecting the attack. It’s important to note that our entire attack process operates within a black-box scenario, devoid of any knowledge about the model parameters. In summary, our contributions can be succinctly summarized as follows:

We introduce the inaugural clean-label backdoor attack explicitly crafted for graph neural network node classification tasks. Unlike previous approaches, this attack refrains from altering the labels of poisoned nodes, simplifying implementation and reducing the likelihood of detection.   
• We craft strategies for poisoning node selection and trigger generation specifically for clean-label backdoor attacks. These strategies establish a robust association between triggers and the target class, even when the labels of poisoned nodes remain unchanged, all within a low budget. To further minimize disruption to the graph structure, we introduced multiple constraints.   
We perform attacks on four types of GNNs using three widely utilized graph datasets. The experimental results reveal exceptional attack performance, especially in scenarios with constrained budgets. For example, on the Pubmed dataset, achieving a modest poisoning rate of only $0 . 5 \%$ led to an impressive $9 6 . 9 4 \%$ attack success rate.

# Related Works

# Backdoor Attacks

Based on whether the injected samples possess consistent characteristics and labels, existing backdoor attacks can be categorized into dirty-label attacks and clean-label attacks.

Dirty-label attacks. Most existing backdoor attacks (Xi et al. 2021; Zhang et al. 2021; Chen et al. 2023; Yang et al. 2022) fall into this category. These attacks typically start by selecting a set of clean examples from non-target classes, apply the backdoor trigger to these examples, and reset their labels to the target class. Training on such a poisoned dataset leads the model to memorize the association between the trigger and the target label. Since the remaining features in the poisoned inputs do not represent the target class, the model is easily able to learn the trigger pattern. Because triggers are typically small patterns, the poisoned inputs still appear to come from non-target classes, making them detectable even through relatively basic data filtering (Turner, Tsipras, and Madry 2018).

Clean-label attacks. To enhance the stealthiness of backdoor attacks, clean-label attacks have emerged. In this type of attack, the poisoned inputs have labels that are consistent with their true labels, making them appear benign even under human inspection. Turner et al. (Turner, Tsipras, and Madry 2018) initially introduced clean label backdoor attacks and implemented them in the field of images. Subsequently, various clean label backdoor attacks targeting CNNs have emerged, such as LC (Turner, Tsipras, and Madry 2019), HTBA (Saha, Subramanya, and Pirsiavash 2020), SAA (Souri et al. 2022), but research on clean label backdoor attacks for GNNs is still limited.

# Graph Neural Network Backdoor Attacks

Recent research has shown that GNNs are also susceptible to backdoor attacks. (Xi et al. 2021) and (Zhang et al. 2021) were the first to demonstrate the feasibility of graph backdoor attacks using subgraphs as triggers. They use subgraph triggers to replace the original subgraph in the graph and modify the labels of poisoned graphs to deceive the model into classifying graphs with triggers as the target class. However, their attacks did not extensively explore attack locations. (Xu, Xue, and Picek 2021) and (Wang et al. 2024) guided by graph explanation models, explored the optimal locations for attacks. To reduce the structural disruption caused by the attached trigger, (Chen et al. 2023) improved the similarity of adjacent nodes by pruning low-similarity links, and (Dai et al. 2023) designed imperceptible backdoor attacks by constraining the similarity between attached nodes and poisoned nodes. Xu et al. (Xu and Picek 2022) first demonstrated the feasibility of implementing cleanlabel backdoor attacks on graph neural network graph classification tasks. However, they did not verify whether it is feasible for node-level classification tasks, and our work fills this gap.

# Preliminary Analysis

# Notations

We use $G \ = \ ( V , E , X )$ to denote a graph, where $V \ =$ $\{ v _ { 1 } , . . . , v _ { N } \}$ represents the set of nodes in the graph. $E$ is the set of edges in the graph, and $X = \{ x \left( v _ { 1 } \right) , . . . , x \left( v _ { N } \right) \}$ represents the set of node features, where $x \left( v _ { i } \right)$ represents the features of a node. In this paper, we focus on node classification tasks, where during training, a set of nodes $V _ { L } \subseteq V$ is associated with labels $\bar { Y _ { L } } = \{ y _ { 1 } , . . . , y _ { N _ { L } } \} .$ . Here, $\theta$ represents the trained GNN model, and $f _ { \theta } \left( x \left( v _ { t } \right) ; G \right)$ represents the output for a node $\boldsymbol { v } _ { t }$ in the graph $G$ . We used $g$ to represent the trigger, and $\boldsymbol { v } _ { t } : \boldsymbol { g }$ represents a node v with an attached trigger. Nodes with attached triggers in the training set are referred to as poisoned nodes, while nodes with attached triggers in the test set are referred to as victim nodes.

We use $G ^ { \prime }$ to represent a graph that contains poisoned nodes or victim nodes.

# Threat Model

Attacker’s Knowledge and Capability. Similar to most current backdoor attacks, the training data of the target model is accessible to the attacker. The information about the target GNN model, including its architecture, is unknown to the attacker. The attacker can attach triggers to poisoned nodes before the target model’s training. In the inference phase, the attacker can attach triggers to the victim nodes.

Attacker’s Goal. Successful attacks require the model to have high accuracy on clean data and high attack accuracy on victim data. We can formalize our attack objective as follows:

$$
\left\{ \begin{array} { l } { f _ { \theta ^ { \ast } } \left( \boldsymbol { x } \left( \boldsymbol { v } : \boldsymbol { g } \right) ; \boldsymbol { G } ^ { \prime } \right) = y _ { t } } \\ { f _ { \theta ^ { \ast } } \left( \boldsymbol { x } ( \boldsymbol { v } ) ; \boldsymbol { G } ^ { \prime } \right) = f _ { \theta _ { 0 } } \left( \boldsymbol { x } \left( \boldsymbol { v } \right) ; \boldsymbol { G } ^ { \prime } \right) } \end{array} \right.
$$

Where $\theta ^ { * }$ represents the poisoned model trained with a mixed dataset containing poisoned data, $\theta _ { 0 }$ represents the clean model, and $y _ { t }$ represents the target class set by the attacker. The first objective indicates that poisoned victim nodes will be misclassified as the target class, and the second objective indicates that for clean data, the output of the poisoned model and the clean model should be close.

# Methodology

In this section, we will provide a detailed introduction to the CGBA model. Firstly, we will provide a brief overview of the attack and outline the attack process. Next, we will explain our strategies for selecting the nodes and feature positions for the attack. Finally, we will describe the trigger generation process in detail.

# Attack Overview

Figure 1 displays the framework of our clean-label node backdoor attack. Our attack occurs in two phases: the training phase and the inference phase. In the training phase, we first train a surrogate model using a clean dataset, which will guide the selection of poisoned nodes and trigger generation. Next, under the guidance of the surrogate model, we select a few nodes with label $y _ { t }$ and attach triggers to a small number of their features. This dataset containing nodes with attached triggers is referred to as the backdoor training dataset, and throughout this process, the labels of all nodes remain unchanged. Finally, the poisoned model is trained using the backdoor training dataset to create a GNN model known as the backdoor GNN. Since the nodes with attached triggers all have the label $y _ { t }$ , the backdoor GNN directly associates the triggers with the label $y _ { t }$ . In the inference phase, for clean data, the backdoor GNN will produce correct outputs. However, when the attacker attaches triggers to the nodes, the backdoor GNN will predict them as the target labelyt.

# Poisoned Node Selection

In this subsection, we provide the details of the poisoned node selection algorithm. Unlike dirty-label attacks, for clean label attacks, the poisoned nodes are chosen from the target class nodes. Intuitively, choosing to attach triggers to nodes in the target class that are more important could be less effective, as these nodes already have high confidence, and slight feature modifications might not have a significant impact on them. On the other hand, selecting nodes with uncertain classification results can confuse the backdoor GNN into believing that the poisoned nodes are classified as the target class due to the attachment of triggers, thereby strengthening the association between triggers and the target class. Therefore, we recommend selecting nodes in the target class with more uncertain classification results as poisoned nodes.

First, we train a simple surrogate model using the training data. Based on the output of the surrogate model, we select nodes with the lowest confidence in the target class for poisoning. In this case, we use a two-layer GCN model as the surrogate model. The loss function for the surrogate model is as follows:

$$
\theta = \underset { \theta } { \arg \operatorname* { m i n } } \sum _ { v _ { i } \in V _ { L } } l \left( f _ { \theta } \left( x ( v _ { i } ) ; G \right) , y _ { i } \right)
$$

Where $y _ { j }$ represents the true label of node $v _ { i }$ , and $l \left( , \right)$ represents the cross-entropy loss. Furthermore, we also take into account that if the degree of poisoned nodes is high, then modifying them will affect more nodes, which would have a more significant impact on the classification of other nodes, thereby reducing the accuracy of clean data. Therefore, we introduce a constraint to make the selected nodes more inclined to choose nodes with lower degrees. Finally, we compute an overall score for each node with the target class label, which is formulated as follows:

$$
\begin{array} { r } { S c o r e \left( v _ { i } \right) = - \left( m \left( f _ { \theta } \left( v _ { i } , y _ { t } ; G ^ { \prime } \right) \right) + \beta \cdot d e g \left( v _ { i } \right) \right) } \end{array}
$$

Where $m \left( f _ { \theta } \left( v _ { i } , y _ { t } ; G ^ { \prime } \right) \right)$ represents the confidence of node $\boldsymbol { v } _ { i }$ on target class $y _ { t }$ in the proxy model $\theta$ , and $d e g \left( v _ { i } \right)$ denotes the degree of node $v _ { i }$ , and $\beta$ is an empirical parameter used to control the contribution of node degree in the node selection process. After obtaining the score for each node, we select the top n nodes with the highest scores from all the target class nodes to form the poisoned node set $V _ { p }$

# Poisoned Feature Selection

Inspired by (Xu, Xue, and Picek 2021), we use the graph explanation network Graphlime (Huang et al. 2022) to select the most important features for the poisoned nodes. Graphlime takes as input a $1 \times N$ node feature and the parameters of the graph neural network, and it outputs a $1 \times N$ tensor where each value represents the importance of the corresponding feature. A higher value indicates higher importance. In CGBA, we calculate the average importance factor for all the features of the poisoned nodes:

$$
A I F = \frac { 1 } { l e n \left( V _ { p } \right) } \sum _ { x _ { i } \in V _ { p } } G r a p h l i m e \left( x _ { i } \right)
$$

Where $V _ { p }$ represents the set of poisoned nodes, Graphlime $( x _ { i } )$ represents the output of the node $\mathbf { \boldsymbol { x } }$ in the graph explanation network Graphlime. Finally, we select the positions of the top m values in $A I F$ as the feature attack locations. Our feature selection method has shown significant improvements in attack effectiveness, as demonstrated in the results in section 5.2.4.

![](images/fa0c42ff87d32f230e4a8dbb336f2846bcd4bc566dde71e86af01cf3e8aa9cf7.jpg)  
Figure 1: The framework of CGBA. In the lower right corner is the testing phase, while outside the box is the training phase

# Feature Trigger Generation

Our attack utilizes fixed triggers, and next, we will describe the generation process of the fixed trigger $g$ .

Design of Feature Trigger. In order to achieve our goal, we aim for the victim nodes to produce a classification result that strongly favors the target class $y _ { t }$ after attaching the trigger. For other non-poisoned nodes, we want to maintain clean outputs. First, we attach the trigger to the poisoned nodes in a way that maximizes the confidence for class $y _ { t }$ while minimizing confidence for other classes. This is done to confuse the backdoor GNN into thinking that the poisoned nodes’ classification results are strongly associated with the target class, thanks to the trigger. Next, for nodes that haven’t been attached with the trigger, we increase their confidence in their true labels. We can formulate our trigger loss as follows:

$$
\begin{array} { c } { { L _ { c } = \displaystyle \operatorname* { m i n } _ { g } \sum _ { v _ { i } \in V _ { p } } l \left( f _ { \theta } \left( v _ { i } : g , y _ { t } ; G ^ { \prime } \right) \right) } } \\ { { + \sum _ { v _ { j } \in V _ { L } / V _ { p } } l \left( f _ { \theta } \left( v _ { j } , y _ { j } ; G ^ { \prime } \right) \right) } } \end{array}
$$

Where $V / V _ { p }$ represents nodes that are not in the poisoned dataset $V _ { p }$ . $\theta$ is the GCN proxy model used in Section 4.2.

Design of Concealment. In order to ensure that nodes with attached triggers remain sufficiently concealed, we introduce two constraints. First, considering the homogeneity of neighboring nodes in the graph structure, if a node has a very low similarity with its neighboring nodes, it is likely to be considered an outlier. Therefore, we aim to make nodes with attached feature triggers as similar as possible to their neighboring nodes, to avoid being detected as outliers. We introduce the following loss to guide the generation of more

# Algorithm 1: CGBA

Input: $\theta , D _ { t r a i n } , y _ { t } , n , m , e p o c h .$ .

Output: $g$ .   
1: get the node set $V _ { y _ { t } }$ with label $y _ { t }$ from $D _ { t r a i n }$ . //get the poisoned node set $V _ { p }$ .   
2: for $v _ { i } \in V _ { y _ { t } }$ do   
3: get $S c o r e ( v _ { i } )$ by using Eq.(3).   
4: end for   
5: $V _ { p } \gets t o p n ( S c o r e ( v _ { i } ) )$ . //get the poisoned features positions $p$ .   
6: for $v _ { j } \in V _ { p }$ do   
7: get $A I F$ by using Eq.(4).   
8: end for   
9: $p \gets t o p m ( A I F )$ . //train feature trigger.   
10: for $k = 1  e p o c h$ do   
11: $g ^ { k }  M L \bar { P } ( g ^ { k - 1 } )$ .   
12: $V _ { p } [ p ]  { g ^ { k } } [ p ]$ ; //Replace the features at position $p$ with $g ^ { k }$ for all nodes in $V _ { p }$ .   
13: calculate the loss $L _ { a l l }$ by using Eq.(8) and optimize $M L P$ .   
14: end for   
15: if $D _ { t r a i n }$ is discrete dataset then   
16: update $g ^ { k }$ by using Eq.(10).   
17: end if   
18: return $g ^ { k }$

concealed triggers:

$$
L _ { n } = \operatorname* { m i n } _ { g } \sum _ { v _ { i } \in V _ { p } } \left( \sum _ { v _ { j } \in V _ { i } } T - s i m \left( x \left( v _ { i } : g \right) - x \left( v _ { j } \right) \right) \right)
$$

Where $x \left( v _ { i } : g \right)$ represents the features of node $\boldsymbol { v } _ { i }$ after attaching trigger $g$ , $V _ { i }$ represents the set of all neighboring nodes of node $v _ { i }$ , sim denotes the similarity between the features of two nodes, and we use cosine similarity, and $T$ represents the threshold for neighbor node similarity.

Second, we noticed that using Eq.(5) to train the generated triggers could result in some abrupt values. To constrain the magnitude of modifications and make the nodes with attached triggers look more like ”normal nodes,” we applied a second constraint on the extent of modifications to the training set. This ensures that the nodes with attached triggers are as similar as possible to their original state. We formulate this constraint as follows:

$$
L _ { s } = \operatorname* { m i n } _ { g } \sum _ { v _ { i } \in V _ { p } } \left( K - s i m \left( x \left( v _ { i } : g \right) - x \left( v _ { i } \right) \right) \right)
$$

Where $K$ represents the threshold for the extent of modifications, and the similarity function used is still the cosine similarity. Our final trigger generation loss consists of three parts from Eq.(5), Eq.(6)and Eq.(7):

$$
L _ { a l l } = L _ { c } + \lambda _ { 1 } L _ { s } + \lambda _ { 2 } L _ { n }
$$

Here $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are weight parameters. Our optimization model uses a two-layer MLP. We optimize the MLP model using Eq.(8) to obtain the final feature trigger, resulting in the final feature trigger $g$ , where $x ^ { 0 }$ is the randomly generated feature.

$$
x ^ { ( k ) } = \mathrm { M L P } \left( x ^ { k - 1 } \right) \qquad g = x ^ { ( k ) }
$$

In particular, when the dataset is a discrete dataset, we adopt a truncation method to obtain the final result. For example, when the features consist of only 0 and 1, if the final generated feature value is greater than 0.5, it is set to 1, otherwise, it is set to 0:

$$
\left\{ \begin{array} { l l } { g \left[ i \right] = 1 , g \left[ i \right] \geq 0 . 5 } \\ { g \left[ i \right] = 0 , g \left[ i \right] < 0 . 5 } \end{array} \right.
$$

Algorithm 1 describes the details of the CGBA attack. The input includes the proxy model $\theta$ , the training dataset $D _ { \mathrm { t r a i n } }$ , the target class $y _ { t }$ , the number of poisoned nodes $n$ , and the number of poisoned features $m$ . The output is the trained feature trigger $g$ .

# Experiments

In this section, we will evaluate the proposed method on different datasets and GNN models to answer the following research questions:

RQ1: Can our proposed method effectively perform backdoor attacks on GNNs while remaining unnoticed? RQ2: How do the number of poisoned nodes and the number of poisoned features affect our attack? RQ3: Do our node selection and feature selection strategies work effectively?

# Experimental Settings

5.1.1 Datasets To demonstrate the effectiveness of our two methods, we conducted experiments on three publicly available real-world datasets: Cora, Citeseer, and Pubmed. Cora and Pubmed are continuous datasets, while Citeseer is a discrete dataset (Sen et al. 2008). These datasets are widely used for inductive semi-supervised node classification. Dataset summaries are provided in Table 1. For each dataset, we split the nodes into a training dataset $( 7 0 \% ) , \mathrm { a }$ validation dataset $( 1 0 \% )$ , a clean test dataset $( 1 0 \% )$ , and a test dataset with embedded triggers $( 1 0 \% )$ . For the target class of the attack, we choose the class located at class count / 2 to ensure randomness.

Table 1: Dataset Statistics   

<html><body><table><tr><td>Datasets</td><td>Nodes</td><td>Edges</td><td>Classes</td><td>Target class</td></tr><tr><td>Cora</td><td>2708</td><td>5429</td><td>7</td><td>3</td></tr><tr><td>Citeseer</td><td>3327</td><td>4608</td><td>6</td><td>3</td></tr><tr><td>Pubmed</td><td>19717</td><td>44338</td><td>3</td><td>1</td></tr></table></body></html>

5.1.2 Evaluation In this paper, we evaluate the effectiveness and evasion capability of our model using two metrics: clean accuracy drop and attack success rate.

(1) Benign Accuracy (BA): BA refers to the accuracy of the GNN model’s output when tested on a clean test dataset. (2) Attack Success Rate (ASR): ASR is the rate at which the targeted nodes in the test dataset, which have been attacked, are successfully classified as the target class.

$$
{ \mathrm { A t t a c k ~ S u c c e s s ~ R a t e } } \left( A S R \right) = { \frac { \# s u c c e s s f u l t r i a l s } { \# t o t a l t r i a l s } }
$$

5.1.3Baseline We compared our approach with the state-ofthe-art subgraph trigger backdoor attack method GTA (Xi et al. 2021), and EGNN (Wang et al. 2024), the node injection backdoor attack UGBA (Dai et al. 2023), the feature trigger backdoor attack methods MIA (Xu, Xue, and Picek 2021), and NFTA (Chen et al. 2023). In addition, we transferred CBA (Xu and Picek 2022) from graph classification task attacks to node classification task attacks (using randomly generated feature triggers injected into random nodes for the attack). MIA, NFTA, and CBA maintain the same experimental settings as CGBA, with the same number of poisoned nodes and attacked features. GTA and EGNN attacked the same number of subgraphs as CBAN. UGBA attached the same number of nodes as CBAN had poisoned nodes. In particular, since NFTA is primarily designed for discrete data sets and flips attacked positions, in our settings, when attacking continuous data sets, its way of generating features is the same as MIA, i.e., randomly generated.

5.1.4 Parameter settings The poisoning ratio in the training set for Cora and Citeseer is $2 \%$ , and for Pubmed, it is $1 \%$ . The feature modification ratio for CGBA and the other two feature modification attacks is $5 \%$ . The value of parameter $\beta$ is 0.01. We deployed a 2-layer GCN as the proxy model, and CGBA used a 2-layer MLP to train the feature triggers.

# Attack Results

5.2.1Comparisons with baseline To answer RQ1, we will compare CGBA with baselines on three real-world graph datasets to understand the attack performance and stealthiness.

In Table 2, we report the results of CGBA and three feature trigger attack methods on four different GNN mod

<html><body><table><tr><td>Datasets</td><td>GNN Type</td><td>clean</td><td>MIA</td><td>NFTA</td><td></td><td>CBA</td><td>CGBA</td></tr><tr><td rowspan="4">Cora</td><td>GCN</td><td>86.44</td><td>87.22|85.30</td><td>90.56|85.56</td><td></td><td>61.67|85.24</td><td>98.89|86.17</td></tr><tr><td>GAT</td><td>86.42</td><td>90.55|85.80</td><td>95.56|85.67</td><td></td><td>73.33|85.42</td><td>98.33|86.54</td></tr><tr><td>GraphSAGE</td><td>85.56</td><td>93.89|85.18</td><td>99.44|85.18</td><td></td><td>83.22|85.56</td><td>100.0|85.62</td></tr><tr><td>GAE</td><td>85.30</td><td>86.67|81.60</td><td>86.67|81.17</td><td></td><td>73.17|82.04</td><td>93.89|84.94</td></tr><tr><td rowspan="4">Citeseer</td><td>GCN</td><td>75.78</td><td>90.5675.80</td><td>80.00|74.19</td><td></td><td>40.56|74.38</td><td>98.8975.37</td></tr><tr><td>GAT</td><td>74.52</td><td>70.56|72.08</td><td>80.37|74.55</td><td></td><td>43.89|72.03</td><td>96.6774.60</td></tr><tr><td>GraphSAGE</td><td>74.46</td><td>99.44|74.14</td><td>78.89|73.04</td><td></td><td>74.33|73.43</td><td>100.0|74.40</td></tr><tr><td>GAE</td><td>72.95</td><td>87.78|73.89</td><td>79.63|72.24</td><td></td><td>41.15|73.80</td><td>98.33|74.19</td></tr><tr><td rowspan="4">Pubmed</td><td>GCN</td><td>86.98</td><td>38.8984.75</td><td>55.01|85.77</td><td></td><td>47.22|86.67</td><td>96.6787.13</td></tr><tr><td>GAT</td><td>86.25</td><td>41.11|85.74</td><td>55.56|85.77</td><td></td><td>56.11|85.83</td><td>98.3386.18</td></tr><tr><td>GraphSAGE</td><td>88.57</td><td>67.78|88.69</td><td>45.56|88.47</td><td></td><td>62.78|88.46</td><td>100.0|88.52</td></tr><tr><td>GAE</td><td>88.12</td><td>54.44|88.17</td><td>41.67|88.11</td><td></td><td>61.674|87.95</td><td>100.0|88.20</td></tr></table></body></html>

Table 2: Backdoor attack results (ASR $( \% )$ |BA $( \% ) _ { . }$ ). Only clean accuracy is reported for clean graphs. The optimal effect is achieved by using bold font to highlight the representation.

GTA GTA GTA GTA UGBA UGBA UGBA UGBA 1.00 EGNN 10 10 1.0 EGN 1 EGNN 280 28 0 0 0.2- 0.2- 0.2 0.2- 0.0+ 0.0+ 0.0 0.0+ Cora CiteseerPumbed Cora CiteseerPumbed Cora CiteseerPumbed Cora CiteseerPumbed Dataset Dataset Dataset Dataset (a) GCN (b) GTA (c) GraphSage (d) GAE

els (GCN (Kipf and Welling 2016a), GraphSAGE (Hamilton, Ying, and Leskovec 2017),GAT (Velickovic et al. 2017),GAE (Kipf and Welling 2016b)). Overall, our proposed backdoor attack exhibits excellent attack effectiveness and stealthiness on all three datasets. Specifically, our attack method achieves an ASR of over $9 6 \%$ on all three datasets, indicating that our attack can successfully change the labels of victim nodes in the test set to the target label. Moreover, CGBA’s BA is advantageous compared to the baseline model, outperforming the baseline model in most cases. This suggests that our attack can maintain the accuracy of predictions for clean nodes, achieving a stealthy attack. Finally, the consistent success across different GNN models demonstrates the robust transferability of our model.

In addition, we compare CGBA with the subgraph trigger attack method GTA and EGNN and the node injection backdoor attack UGBA in terms of ASR in the Figure 2. From the figure, we can see that CGBA has a significant advantage in attack success rate compared to GTA and UGBA. In the Table 3, we list the comparison of CGBA with GTA ,EGNN and UGBA in terms of BA on different neural network models. From the table, it can be seen that CGBA’s performance is generally better than the baseline models. Overall, CGBA shows significant improvements in both attack success rate and stealthiness compared to the baseline models.

In order to further validate the stealthiness of our attack, we also tested the performance of our attack against potential defenses. Two defense methods are based on measuring the similarity of neighboring nodes. When the similarity between adjacent nodes falls below a fixed threshold, these methods consider the edge to be anomalous. The Prune defense method removes these anomalous edges, while the Isolate method not only removes the anomalous edges but also isolates anomalous nodes connected to the anomalous edges. The results are averaged over four different GNN models. The results are presented in Table4. When facing two defense schemes, the success rates of other attack methods all decrease to varying degrees. However, CGBA shows almost no decrease in the face of defense, and even exhibits an increase in ASR on the Cora and PubMed datasets. This may be because our triggers are deliberately designed to maintain a high level of similarity between poisoned nodes and their neighboring nodes. When the defense is applied, the poisoned nodes are not affected, and clean nodes may be removed, leading to an increase in ASR.

# Impact of the Sizes of Poisoned Nodes and Features

To answer RQ2, we conducted experiments to explore the impact of different numbers of poisoned nodes and modified features during the training phase on our attack.

Table 3: Comparison Results with Other Attack Methods (BA $( \% ) _ { . } ^ { . }$ ). Only clean accuracy is reported for clean graphs. The optimal effect is achieved by using bold font to highlight the representation.   

<html><body><table><tr><td>Datasets</td><td>GNN Type</td><td colspan="2">clean UGBA GTA EGNNCGBA</td></tr><tr><td>Cora</td><td>GCN GAT GraphSAGE GAE</td><td>86.4485.86 86.10 85.55 86.42 86.10 85.68 86.00 85.56 85.56 84.0785.43 85.30 83.27 80.18 82.98</td><td>86.17 86.54 85.62 84.94 75.37 74.60</td></tr><tr><td>Citeseer</td><td>GCN GAT GraphSAGE GAE</td><td>75.78 75.05 75.00 74.52 74.50 74.46 73.79 72.95 72.14 72.20</td><td>75.22 74.40 74.11 73.6773.56 74.40 72.81 74.19</td></tr><tr><td>Pubmed</td><td>GCN GAT GraphSAGE GAE</td><td>86.98 86.94 86.76 86.25 85.89 85.52 88.57 88.73 88.35 88.12 88.15 88.20</td><td>86.74 87.13 85.69 86.18 87.53 88.52 88.13 88.20</td></tr></table></body></html>

1.0 1.0 0.876 0.876  
R0.8 0.8 0.874 0.874  
S MIA B M0.6 NETA 0.6 0.872 NETA 0.872UGBA UGBAGTA 0.870 GTA -0.8700.4 CGBA0.4 CGBA3060 90 120150 306090　120　150SizesofPoisonedNodes Sizes of Poisoned Nodes(a) ASR (b) BA

Impacts of sizes of poisoned nodes. We conducted attacks by poisoning the training set with $\{ 3 0 , 6 0 , 9 0 , 1 2 0 , 1 5 0 \}$ poisoned nodes, while keeping the other settings consistent with those mentioned in section 5.1.4. The results, presented in Figure 3, show the average ASR and BA on the Pubmed dataset for four different graph neural network models. Similar results were obtained for the other datasets. As expected, the ASR increases and BA decreases as the number of poisoned nodes increases. ASR remains at a high level when the number of poisoned nodes reaches 90. Moreover, our results consistently outperform the baseline across different numbers of poisoned nodes, providing strong evidence for the effectiveness of our attack method. Additionally, even at lower poisoning rates, our method maintains a high level of attack performance, demonstrating that our approach can achieve successful attacks on a lower budget.

Impacts of sizes of poisoned features. Meanwhile, we conducted experiments with varying numbers of poisoned node features. In Figure 4, we report the variation in ASR with the increase in the number of poisoned features in the Cora and Pubmed datasets. The results indicate that the effect of modifying features is similar to that of the number of poisoned nodes, as ASR increases with the increase in the number of

1.0 1.0 1.0 1.0 0.8 0.8 0.8 0.8 80.6 0.6 R0.6 0.6 S 0.4 MIA 0.4 0.4 0.4 MIA 0.2 -CBA NFTA 0.2 0.2 NFTA 0.2 CGBA CBBA 0.0 0.0 0.0 0.0 10 203040506070 10 20 304050 6070 SizesofPodisonedFeatures Sizes of Podisoned Features (a) Cora (b) Pubmed

![](images/69a03e9e3b83a2397c92267c433f90926439cf7154ef2e55e47d1a5418f6087f.jpg)  
Figure 3: Impacts of sizes of poisoned nodes on Pubmed.   
Figure 4: Impacts of sizes of poisoned features.   
Figure 5: Impact of Different Node Selection Strategies   
Figure 6: Impact of Different Feature Selection Strategies. (a) Present the results of poisoning 20 nodes on the Pubmed dataset, and (b) display the results of poisoning 50 nodes on the Pubmed dataset.

poisoned features. Most importantly, our approach significantly outperforms the baseline model with the same number of feature attacks. It is worth noting that even with a very small number of feature attacks, our attack still achieved significant success.

# Ablation Experiments

To answer RQ3, we conducted ablation experiments to examine the impact of different node selection and feature selection strategies on our model.

# Parameter Sensitivity Analysis

Impact of Different Node Selection Strategies. We conducted a variant experiment called CGBA/S, which randomly selects and poisons nodes. To highlight the effectiveness of our node selection strategy, we conducted attacks with fewer poisoned nodes, using one-third of the poisoning

1.0 RF 1.0 RF MIF MIF 0. 0.4 1 0 0.4 1 0.2 0.2 0.0 0.0 MIANFTA CABN MIANFTA CABN AttackMethods AttackMethods (a) Pubmed-20 (b) Pubmed-50

<html><body><table><tr><td>Datasets</td><td>Defense</td><td>MIA</td><td>NFTA</td><td>CBA</td><td>UGBA</td><td>GTA</td><td>EGNN</td><td>CGBA</td></tr><tr><td rowspan="3">Cora</td><td>None</td><td>90.83</td><td>90.56</td><td>70.12</td><td>92.56</td><td>44.43</td><td>91.36</td><td>98.89</td></tr><tr><td>Prune</td><td>90.75</td><td>86.67</td><td>66.56</td><td>91.11</td><td>21.06</td><td>83.53</td><td>97.26</td></tr><tr><td>Isolate</td><td>62.91</td><td>80.10</td><td>65.36</td><td>89.99</td><td>35.01</td><td>80.27</td><td>98.63</td></tr><tr><td rowspan="3">Citeseer</td><td>None</td><td>86.67</td><td>80.00</td><td>55.23</td><td>94.45</td><td>44.43</td><td>92.98</td><td>99.16</td></tr><tr><td>Prune</td><td>85.61</td><td>76.33</td><td>41.52</td><td>85.00</td><td>35.05</td><td>89.36</td><td>98.89</td></tr><tr><td>Isolate</td><td>69.95</td><td>74.11</td><td>43.29</td><td>91.11</td><td>6.67</td><td>84.56</td><td>98.33</td></tr><tr><td rowspan="3">Pubmed</td><td>None</td><td>49.56</td><td>55.01</td><td>60.25</td><td>89.70</td><td>54.43</td><td>86.55</td><td>96.67</td></tr><tr><td>Prune</td><td>44.34</td><td>54.44</td><td>51.25</td><td>83.88</td><td>37.75</td><td>77.53</td><td>96.59</td></tr><tr><td>Isolate</td><td>31.11</td><td>54.29</td><td>49.36</td><td>89.44</td><td>12.74</td><td>79.12</td><td>95.55</td></tr></table></body></html>

Table 4: Comparative $\mathrm { A S R } ( \% )$ Evaluation under Varied Defensive Strategies. None indicates that no defense was used.

![](images/f2e1afb580786edd19f2720c3a810fd8b754fe036fb39dceed3eb664c041f0e1.jpg)  
Figure 7: The Impact of Parameters.

quantity from the experiments in Section 5.2.1, on all three datasets. All other settings remained the same for fairness. Figure 5 presents the results of our experiments on the three datasets. From the table, we can observe that our node selection strategy significantly outperforms the variant experiment. This demonstrates that selecting nodes with the lowest confidence in the target label from the training set can effectively enhance the attack performance.

Impact of Different Feature Selection Strategies. We explored the impact of different feature selection strategies by comparing Most Important Feature (MIF) selection to Random Feature (RF) selection. We conducted experiments with two different feature attack quantities, which were 20 and 50. The results, as shown in Figure 6, Selecting the most important features for poisoning had a significantly better effect compared to random feature selection, especially when attacking a smaller number of features. This clearly demonstrates the effectiveness of choosing features with the highest average importance for poisoning. Additionally, even when attacking the same features, our ASR outperforms the baseline model, indicating that our trained feature triggers can establish a stronger association with the target labels.

In this subsection, we further investigate the impact of hyperparameters $K , T , \lambda _ { 1 }$ , and $\lambda _ { 2 }$ on the performance of CGBA. Here, $K$ and $T$ represent the thresholds for feature modification magnitude and the similarity threshold between a poisoned node and its neighboring nodes, while $\lambda _ { 1 }$ and $\lambda _ { 2 }$ represent the weights for these two components. $K$ and $\lambda _ { 1 }$ Parameter Analysis. We set the value of $K$ as $\{ 0 . 0 1 , 0 . 0 0 5 , 0 . 1 , 0 . 3 , 0 . 5 \}$ and the value of $\lambda _ { 1 }$ as $\{ 0 , 1$ ,

$1 0 , 5 0 , 1 0 0 , 5 0 0 \}$ . In Figure 7(a), we report the impact of $K$ and $\lambda _ { 1 }$ on the Attack Success Rate (ASR) of CGBA on the Pubmed dataset. We observe that as $K$ decreases and $\lambda _ { 1 }$ increases, the magnitude of feature modifications to nodes becomes smaller, leading to a reduction in ASR. This is as expected, because when the restriction on the magnitude of modifications to nodes is smaller, the estimated budget will increase, leading to a stronger attack effect. It is worth noting that when $K$ exceeds 0.1, the ASR can be maintained at a high level, and when $K$ reaches 0.5, the ASR can remain at $100 \%$ . This indicates that our attack can achieve a very strong effect on the Pubmed dataset with a maximum node modification magnitude of 0.5, which is within our expectations.

$T$ and $\lambda _ { 2 }$ Parameter Analysis. We set the value of $T$ as $\{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 8 , 1 . 0 \}$ , and the value of $\lambda _ { 2 }$ as $\{ 0 , 1 , 1 0 \$ , ${ \dot { 5 } } 0 , 1 0 0 , 5 0 0 \}$ . In Figure 7(b), we report the effects on the Pubmed dataset. We can observe that when $\lambda _ { 2 }$ has a lower value, ASR increases with an increase in $T$ . We hypothesize that this effect may be due to the homogeneity of the graph, where neighboring nodes tend to have similar classification results. Poisoned nodes are more likely to have neighbors with classifications biased towards the target class $y _ { t }$ . As we increase the similarity with neighboring nodes, it further boosts the confidence of poisoned nodes in the target class. When we increase the similarity with neighboring nodes, it further boosts the confidence of the poisoned nodes in the target class. Conversely, when $\lambda _ { 2 }$ has a higher value, ASR decreases as $T$ increases. This is possibly because excessive weight negatively impacts the optimization of $L _ { c }$ , leading to a reduction in the effectiveness of the attack.

# Conclusion

In this paper, we propose CGBA, a groundbreaking cleanlabel backdoor attack approach tailored specifically for GNNs node classification tasks. Unlike traditional graph backdoor attacks, clean-label backdoor attacks achieve a backdoor attack by modifying only the output data without altering the labels. Our approach addresses the gap in cleanlabel node-level backdoor attacks on graph neural networks. Experiments have shown that CGBA can effectively achieve clean-label node-level backdoor attacks.