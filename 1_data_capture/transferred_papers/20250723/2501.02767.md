# Enhancing Trustworthiness of Graph Neural Networks with Rank-Based Conformal Training

Ting Wang1, Zhixin Zhou2, Rui Luo1\*

1Department of Systems Engineering, City University of Hong Kong, Hong Kong SAR, China 2Alpha Benito Research, Los Angeles, USA t.wang123@outlook.com, zzhou $@$ alphabenito.com, ruiluo $@$ cityu.edu.hk

# Abstract

Graph Neural Networks (GNNs) has been widely used in a variety of fields because of their great potential in representing graph-structured data. However, lacking of rigorous uncertainty estimations limits their application in high-stakes. Conformal Prediction (CP) can produce statistically guaranteed uncertainty estimates by using the classifier’s probability estimates to obtain prediction sets, which contains the true class with a user-specified probability. In this paper, we propose a Rank-based CP during training framework to GNNs (RCP-GNN) for reliable uncertainty estimates to enhance the trustworthiness of GNNs in the node classification scenario. By exploiting rank information of the classifier’s outcome, prediction sets with desired coverage rate can be efficiently constructed. The strategy of CP during training with differentiable rank-based conformity loss function is further explored to adapt prediction sets according to network topology information. In this way, the composition of prediction sets can be guided by the goal of jointly reducing inefficiency and probability estimation errors. Extensive experiments on several real-world datasets show that our model achieves any predefined target marginal coverage while significantly reducing the inefficiency compared with state-of-the-art methods.

Code — https://github.com/CityU-T/RCP-GNN

# Introduction

Graph Neural Networks (GNNs) has been widely used in many applications, such as weather forecasting (Lam et al. 2023), drug discovery (Li, Huang, and Zitnik 2022) and recommendation systems (Wu et al. 2022). However, predictions made by GNNs are inevitably present uncertainty. Though to understand the uncertainty in the predictions they produce can help to enhance the trustworthiness of GNNs (Huang et al. 2023), most existing uncertainty quantification (UQ) methods can not be easily adopted to graph-structured data (Stadler et al. 2021). Among various UQ methods, Conformal Prediction (CP) is an effective approach for achieving trustworthy GNNs (Wang et al. 2024). It relaxes the assumption of existing UQ methods, making it suitable for graphstructured data.

Instead of relying solely on uncalibrated predicted distribution $\mu ( \boldsymbol { y } | \boldsymbol { x } )$ , CP constructs a prediction set that informs a plausible range of estimates aligned with the true outcome distribution $\bar { p ( \boldsymbol { y } | \boldsymbol { x } ) }$ . The post-training calibration step make the output prediction sets provably include the true outcome with a user-specified coverage of $1 - \alpha$ . A conformity score function is the key component for CP to quantify the agreement between the input and the candidate label.

Some studies have draw attention to the application of CP to graph-structured data (Clarkson 2023; H. Zargarbashi, Antonelli, and Bojchevski 2023; Lunde 2023; Lunde, Levina, and Zhu 2023; Marandon 2024). However, CP usually suffer inefficiency when there are no well-calibrated probabilities, with the intuition that larger prediction sets covers higher uncertainty. How to achieve desirable efficiency beyond validity is still a noteworthy challenge. Existing studies (Sadinle, Lei, and Wasserman 2019; Romano, Sesia, and Candes 2020) are typically changing the definition of the conformity scores for inefficiency reduction. The challenge is that CP is always used as a post-training calibration step, thus hindering its ability of underlying model to adapting to the prediction sets.

Recently, (Stutz et al. 2022; Bellotti 2021) try to using CP as a training step to make model parameter $\theta$ dependent with the calibration step, so as to modifying prediction sets towards reducing inefficiency. However, the integration of conformal training and GNNs still remain largely unexplored. The very resent work (Huang et al. 2023) proposed a conformal graph neural network which develops a topology-aware calibration step during training. Differently, we focus this problem with two lines. One is that a suitable conformity score is applied for GNNs who often struggle with miscalibration predictions (Wang et al. 2021). Another is that a conformal training framework based on the differentiable variant of this conformity score is designed to adjust the prediction sets along with model parameters’ optimization.

In conclusion, our contributions are two-fold. First, we propose a novel rank-based conformity scores that emphasizes the rank of prediction probabilities which is more robust to GNNs. Second, we develops a calibration step during training for adjusting the prediction sets along with the model parameters. We demonstrate that the rank-based conformal prediction method we introduce is performancecritical for efficiently constructing prediction sets with expected coverage rate. And the proposed method can outperform state-of-the-art methods for the graph node classification tasks on several popular network datasets in terms of the converge and inefficiency metrics.

# Preliminaries

Let $G = ( \nu , \mathcal { E } , X )$ be a graph, where $\nu$ is a set of nodes, $\mathcal { E }$ is a set of edges, and $\mathbf { \bar { \Phi } } _ { X } \mathbf { \bar { \Phi } } = \{ x _ { v } \} _ { v \in \mathcal { V } }$ is the attributes. We denote $y$ as the discrete set of possible label classes. Let $\{ ( x _ { v } , y _ { v } ) \} _ { v \in D }$ be the random variables from the training data, where $x _ { v } \ \in \ R ^ { d }$ is the $d$ -dimensional vector for node $v$ and $y _ { v } \in \mathcal { V }$ is its corresponding class. The training data $\mathcal { D }$ is randomly split into $\mathcal { D } _ { t r } / \mathcal { D } _ { v a l } / \mathcal { D } _ { c a l i b }$ as training/validation/calibration set. Note that the subset $\mathcal { D } _ { { c a l i b } }$ is withhold as calibration data for conformal prediction. Let $\{ ( x _ { v } ) \} _ { v \in D _ { t e } }$ be the random variables from the test data whose true labels $\{ ( y _ { v } ) \} _ { v \in D _ { t e } }$ is unknown for model. The goal of node classification tasks is to obtain a classifier $\mu : X \to Y$ , which can approximate the posterior distribution over classes $y _ { v } \in \textit { Y }$ . During the training step, $\{ ( x _ { v } , y _ { v } ) \} _ { v \in D _ { t r } \cup D _ { v a l i d } }$ , $\{ ( x _ { v } ) \} _ { v \in D _ { t e } \cup D _ { c a l i b } }$ and the graph structure $( \nu , \mathcal { E } )$ are available to GNN model for node representations.

# Graph Neural Networks (GNNs)

In this paper, we focus on GNNs in the node classification scenario. GNNs is the most common encoder to learn compact node representations, which is generated by a series of propagation layers. For each layer $l$ , each node representation $\bar { h _ { u } ^ { ( l ) } }$ is updated by its previous representations $h _ { u } ^ { ( l - 1 ) }$ , and aggregated features $\hat { m } _ { u } ^ { ( \bar { l } ) }$ obtained through passing message from its neighbours $\mathcal { N } _ { ( u ) }$ :

$$
h _ { u } ^ { ( l ) } = \mathrm { F } _ { \mathrm { u p d } } \big ( h _ { u } ^ { ( l - 1 ) } , \hat { m } _ { u } ^ { ( l ) } \big )
$$

$$
\hat { m } _ { u } ^ { ( l ) } = \mathrm { F _ { a g g } } \big ( m _ { ( u v ) } , | v \in \mathcal { N } _ { ( u ) } \big )
$$

$$
m _ { ( u v ) } = \mathrm { F } _ { \mathrm { m s g } } ( h _ { u } ^ { ( l - 1 ) } , h _ { v } ^ { ( l - 1 ) } )
$$

where $\mathrm { F _ { u p d } ( \cdot ) }$ is a non-linear function to update node representations. $\dot { \mathrm { F _ { a g g } ( \cdot ) } }$ is the aggregation function while $\bar { \mathrm { F } _ { \mathrm { m s g } } ( \cdot ) }$ is the message passing function. We use node representations in the last layer as the input of a classifier to obtain a prediction $\mu _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ .

For CP on GNNs, a valid coverage guarantee requires the exchangeability of the calibration and test data. Since our model is transduction node classification, the calibration examples are drawn exchangeability from the test distribution following (Huang et al. 2023).

# Conformal Prediction

For a new test point $x _ { n + 1 }$ , the goal of CP is to construct a reasonably small prediction set $C ( x _ { n + 1 } )$ , which contains corresponding true label $y _ { n + 1 } \in \mathcal { V }$ with pre-defined coverage rate $1 - \alpha$ :

$$
P ( y _ { n + 1 } \in C ( x _ { n + 1 } ) ) \geq 1 - \alpha
$$

where $\alpha \in [ 0 , 1 ]$ is the user-specific miscoverage rate. The standard $\mathrm { C P }$ is usually conduct at test time after the classification model $\mu _ { \boldsymbol { \theta } }$ is trained, which is achieved in two steps: 1) In the calibration step, a cut-off threshold $\hat { \eta }$ is calculated by the quantile function of the conformity scores $V : X \times Y \to R$ on the hold-out calibration set $\mathcal { D } _ { c a l i b }$ . During calibration, the true classes $y _ { i }$ are used for computing the threshold to ensure coverage $1 - \alpha . 2 \rangle$ ) In the prediction step, the prediction sets $C ( x )$ depending on the threshold $\hat { \eta }$ and the model parameters $\theta$ are constructed. The conformity score function is designed to measure the predicted probability of a class, and it is typically changed for various objectives. Two popular conformity scores are described in details below.

Threshold Prediction Set (THR) The threshold $\hat { \eta }$ for THR (Sadinle, Lei, and Wasserman 2019) is calculated by the $\alpha$ quantile of the conformity scores:

$$
\hat { \eta } = Q \vert \{ V ( x _ { i } , y _ { j } ) \vert i \in \mathcal { D } _ { c a l i b } \} , \alpha ( 1 + \frac { 1 } { | \mathcal { D } _ { c a l i b } | } ) )
$$

where $Q ( \cdot )$ is the quantile function. The prediction sets including labels with sufficiently large prediction values are constructed by thresholding probabilities:

$$
C ( x ) = \{ k \in \mathcal { V } : V ( x , k ) \geq \hat { \eta } \}
$$

$$
V ( x , k ) = \mu _ { k } ( x )
$$

Adaptive Prediction Set (APS) APS (Romano, Sesia, and Candes 2020) takes the cumulative sum of ordered probabilities $\mu _ { \pi ( 1 ) } ( x ) > \mu _ { \pi ( 2 ) } ( x ) > \cdots \mu _ { \pi ( | \mathcal { V } | ) } ( x )$ for prediction set construction:

$$
C ( x ) = \{ k \in \mathcal { V } : V ( x , k ) \leq \hat { \eta } \}
$$

$$
V ( x , k ) = \sum _ { j = 1 } ^ { k } \mu _ { \pi ( j ) } ( x )
$$

where $\pi$ is a permutation of $y$ , and the $( 1 \mathrm { ~ - ~ } \alpha ) ( 1 \mathrm { ~ + ~ }$ $1 / | \mathcal { D } _ { c a l i b } | )$ -quantile is also required for calibration to ensure marginal coverage.

# RANK: Rank-based Conformal Prediction

Following our previous work (Luo and Zhou 2024), we advancing CP to GNNs through rank-based conformity scores, named RANK, to directly reduce the inefficiency. Assuming that a higher value of $\mu _ { k } ( x _ { i } )$ indicates a greater likelihood of $x _ { i }$ belonging to class $k$ . Consequently, if class $k$ is included in the prediction set, and $\mu _ { k ^ { \prime } } ( x _ { i } ) > \dot { \mu } _ { k } ( x _ { i } )$ satisfied, then $k ^ { \prime }$ must be in the prediction set. According to this assumption, the size of the prediction set including the true label can be evaluated in the calibration step. The smallest prediction set that includes the true label $y _ { i }$ will be constructed by the rank of $\mu _ { y _ { i } } ( x _ { i } )$ within the sequence $\{ \mu _ { 1 } ( x ) , \cdot \cdot \cdot \mu _ { \mathcal { V } } ( x ) \}$ .

Ranked Threshold Prediction Sets For each $i \in \mathcal { D } _ { c a l i b }$ , the following rank is defined to establish a rule to choose labels:

$$
\boldsymbol { r } _ { i } = \mathrm { r a n k } \mathrm { o f } \mu _ { y _ { i } } ( \boldsymbol { x } _ { i } ) \mathrm { i n } \left\{ \mu _ { k } ( \boldsymbol { x } _ { i } ) : k \in \mathcal { V } \right\}
$$

so that the order statistics can be find: $r _ { ( 1 ) } \geq r _ { ( 2 ) } \cdot \cdot \cdot \geq r _ { ( n ) }$ . Let $r _ { \alpha } ^ { * } = r _ { \left( \left\lfloor ( n + 1 ) \alpha \right\rfloor \right) }$ , either top- $( r _ { \alpha } ^ { * } - 1 )$ or top- $\left( r _ { \alpha } ^ { \ast } \right)$ classes will be included in the prediction set. The top- $( r _ { \alpha } ^ { * } )$ classes refers to the classes corresponding to the $\left( r _ { \alpha } ^ { \ast } \right)$ -th largest prediction values. To achieve the target coverage, the $\mu ^ { * }$ is defined to determine when the $\left( r _ { \alpha } ^ { \ast } \right)$ -th class should be included in the prediction sets:

$\mu ^ { * } = \lceil n p \rceil$ -th largest value in $\{ \hat { \mu } _ { r _ { \alpha } ^ { * } } ( x _ { i } ) : i \in \mathcal { D } _ { c a l i b } \}$ (11) where $n$ is the size of $\mathcal { D } _ { { c a l i b } }$ , $p$ is the proportion of instances we should included in the $r _ { a } ^ { * }$ -th label, and $\hat { \mu } _ { k } ( x _ { i } )$ denotes the $k$ -th order statistics in $( \mu _ { 1 } \mathsf { \bar { ( } } x _ { i } ) , . . . , \mu _ { n } ( x _ { i } ) )$ . The prediction set is defined as follows:

$$
C ( x ) = \left\{ \begin{array} { r l } & { \{ k \in \mathcal { V } : \hat { \mu } _ { k } ( x ) \geq \hat { \mu } _ { r _ { \alpha } ^ { * } } ( x ) \} , } \\ & { \mathrm { i f } \quad \hat { \mu } _ { r _ { \alpha } ^ { * } } ( x ) \geq \mu ^ { * } ; } \\ & { \{ k \in \mathcal { V } : \hat { \mu } _ { k } ( x ) \geq \hat { \mu } _ { r _ { \alpha ^ { * } - 1 } } ( x ) \} , } \\ & { \mathrm { o t h e r w i s e } ; } \end{array} \right.
$$

According to above analysis, the rank-based conformity scores calculated on the calibration set can be defined following:

$$
\begin{array} { l } { { \displaystyle V ( x _ { i } , y _ { i } ) = } \ ~ } \\ { { \displaystyle \left[ \mathrm { r a n k ~ o f ~ } \hat { \mu } _ { y _ { i } } ( x _ { i } ) \mathrm { ~ i n ~ } \{ \hat { \mu } _ { 1 } ( x _ { i } ) , \cdot \cdot \cdot , \hat { \mu } _ { \mathcal { V } } ( x _ { i } ) \} \right] - 1 } } \\ { { \displaystyle ~ + \frac { 1 } { n } [ \mathrm { r a n k ~ o f ~ } \hat { \mu } _ { y _ { i } } ( x _ { i } ) \mathrm { ~ i n ~ } \{ \hat { \mu } _ { y _ { i } } ( x _ { 1 } ) , \cdot \cdot \cdot , \hat { \mu } _ { y _ { i } } ( x _ { n } ) \} ] } } \end{array}
$$

and the quantile $Q$ as the $\lfloor ( n + 1 ) \alpha \rfloor$ -th largest value among the conformity scores, defining the prediction set is equivalent to selecting the calibration samples that satisfy the condition $V ( x _ { i } , y _ { i } ) \leq Q$ , is employed to construct the prediction set with 1 α coverage.

# RCP-GNN: Rank-Based Conformal Prediction on Graph Neural Networks

We propose RCP-GNN into two-stage: model training stage and conformal training stage, as Figure 1 shows. In model training stage, the base model ${ \mathrm { G N N } } _ { b a s e }$ is trained only by prediction loss (i.e., cross-entropy loss), and $\mu ( X )$ is the estimator of base model.

In conformal training stage, the correct model ${ \bf G N N } _ { c o r }$ is trained by both prediction loss and conformity loss. We set $\tilde { \mu } ( X ) ~ \doteq ~ \mathrm { G N N } \bar { \mathrm { N } } _ { c o r } ( \mu ( X ) , G )$ as the estimator of correct model. The prediction set will be optimized when CP performing on each mini-batch. For the reason that training with original rank-based CP may cause limited gradient flow, a differentiable implementation for RANK is designed to enable smooth sorting and ranking, and is further used to construct the conformity loss function, which sharpens the prediction set.

# Conformal Training

We try to train our model end-to-end with the conformal wrapper in order to allow fine-grained control over the prediction sets $C ( x )$ . Following the split CP approach (Lei, Rinaldo, and Wasserman 2013), we randomly split the test data set $\mathcal { D } _ { t e }$ into folds with $5 0 \% / 5 0 \%$ as $\hat { \mathcal { D } } _ { c a l i b } / \bar { \mathcal { D } } _ { t e }$ for calibration and constructing prediction sets. Before splitting the test data, a fraction of test data is withhold for further standard rank-based conformal prediction stage.

Differentiable Prediction and Calibration Steps A differentiable CP method which involves differentiable prediction and calibration step is defined for the training process: 1) In the prediction step, the prediction sets $C ( x )$ w.r.t. the threshold $\hat { \eta }$ and the predictions $\tilde { \mu } _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ is set to be differentiable. 2) In the calibration step, the conformity scores w.r.t. the predictions ${ \tilde { \mu } } _ { \theta } ( x )$ as well as quantile function is set to be differentiable. Notably, the predictions $\tilde { \mu } _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ are always differentiable throughout calibration and prediction steps. Therefore, The key component of differentiating through CP is the differentiable conformity scores and the differentiable quantile computation.

Given the prediction probabilities ${ \tilde { \mu } } ( X )$ , the smooth sorting designed by a sigmoid $\begin{array} { r } { ( x ) = \frac {  { \mathrm { ~ 1 ~ } } } { 1 + e ^ { - x } } } \end{array}$ function and a temperature hyper-parameter $\tau \in \ [ 0 , 1 ]$ is utilized to replace “hard” rank manipulation for the smoothed rank-based conformity scores:

$$
\tilde { V } ( x _ { i } , k ) = \sum _ { j = 1 } ^ { | \mathcal { V } | } \mathrm { s i g m o i d } ( \frac { \tilde { \mu } _ { j } ( x _ { i } ) - \tilde { \mu } _ { k } ( x _ { i } ) } { \tau } )
$$

After that, a differentiable quantile computation is employed for smoothed thresholding under smooth sorting.

$$
\hat { \eta } = \tilde { Q } ( \{ V ( x _ { i } , y _ { i } ) | i \in \hat { \mathcal { D } } _ { c a l i b } \} , \alpha ( 1 + \frac { 1 } { | \hat { \mathcal { D } } _ { c a l i b } | } ) )
$$

where $\tilde { Q } ( \cdot )$ is the smooth quantile function that are well-established in (Blondel et al. 2020; Chernozhukov, Ferna´ndez-Val, and Galichon 2007).

Loss Function The conformal training stage performs differentiable CP on data batch during stochastic gradient descent (SGD) training. As mentioned above, the $\hat { \eta }$ is calibrated by $\alpha ( 1 + 1 / | \hat { \mathcal { D } } _ { c a l i b } | )$ -quantile of the conformity scores in a differentiable way. Under the constraint of hyperparameter $\tau$ , we empirically make coverage close to $1 - \alpha$ by approximating “hard” sorting. Then we propose a conformity loss function to further optimize the inefficiency through training. Given the estimator $\tilde { \mu } _ { j } ( x _ { i } )$ for the conditional probability of $Y$ being class $k \in \mathcal { V }$ at $X = x _ { i }$ and the true label $y _ { i }$ . Similar with Eq.14, the smooth conformity scores on test data is defined as:

$$
\tilde { V } ( x _ { i } , y _ { i } ) = \sum _ { k = 1 } ^ { | \mathcal { V } | } \mathrm { s i g m o i d } ( \frac { \tilde { \mu } _ { k } ( x _ { i } ) - \tilde { \mu } _ { y _ { i } } ( x _ { i } ) } { \tau } )
$$

Given $i \in \hat { \mathcal { D } } _ { c a l i b }$ , a soft assignment (Stutz et al. 2022; Huang et al. 2023) of each class $k$ to the prediction set is defined smoothly as follows:

$$
c _ { i } = \operatorname* { m a x } ( 0 , \sum _ { k \in \mathcal { V } } \mathrm { s i g m o i d } ( \frac { \tilde { V } ( x _ { i } , k ) - \hat { \eta } } { \tau } ) - \kappa )
$$

Then the conformity loss function is defined by:

$$
\mathcal { L } _ { c p } = \frac { 1 } { | \hat { D } _ { t e } | } \frac { 1 } { | \mathcal { V } | } \sum _ { i \in \hat { D } _ { t e } } c _ { i }
$$

Thus, the loss function optimized in conformal training stage is defined as follows:

$$
\mathcal { L } = \mathcal { L } _ { p r e d } + \lambda * \mathcal { L } _ { c p }
$$

![](images/69bdc5f05df5759bf686a2292df8bb9828aab114df924a90bd94e990d2a0cdb0.jpg)  
Figure 1: The framework of RCP-GNN. (a) Model training stage. The GNN model ${ \mathrm { G N N } } _ { b a s e }$ is trained by optimizing a prediction loss using a standard deep learning step. And the prediction probabilities of node $i$ : $\mu ( x _ { i } )$ is obtained. (b) Conformal training stage. The novel rank-based conformal training step is proposed to adjust the prediction set for desirable properties jointly with improve estimation accuracy. The topology-aware correction model $\mathbf { G N N } _ { c o r }$ that takes $\tilde { \mu } ( x )$ as the output is updated by the conformal training step. (c) Rank-based Conformal Prediction. The rank-based CP is employed to produce a prediction set based on $\tilde { \mu }$ which includes true label with a user-specified probability.

where $\lambda$ is a hyper-parameter to balance the items and $\mathcal { L } _ { p r e d }$ is the prediction loss for optimizing model parameters $\theta$ :

$$
\begin{array} { r l } {  { \mathcal { L } _ { p r e d } = } } \\ & { - \sum _ { i \in \mathcal { D } _ { t r } } [ y _ { i } l o g ( \tilde { \mu } _ { \theta } ( x _ { i } ) ) + ( 1 - y _ { i } ) l o g ( 1 - \tilde { \mu } _ { \theta } ( x _ { i } ) ) ] } \end{array}
$$

After training, standard rank-based CP are conduct on ${ \tilde { \mu } } ( X )$ for prediction sets construction.

# Experiment

We conduct experiments to demonstrate the advantages of our model over existing methods in achieving empirical marginal coverage for graph data, as well as the efficiency improvement. We also conduct systematic ablation and parameter analysis to show the robustness of our model.

# Experiment Setup

Dataset We choose eight popular graph-structured datasets, i.e., Cora, DBLP, CiteSeer and PubMed (Yang, Cohen, and Salakhudinov 2016), Amazon-Computers and Amazon-Photo, Coauthor-CS and Coauthor-Physics (Shchur et al. 2019) for evaluation. We randomly split them with $2 0 \% / 1 0 \% / 7 0 \%$ as training/validation/testing set following previous works (Huang et al. 2023; Stutz et al. 2022). The statistical information of datasets is summarized in Table 1.

Table 1: Statistics of Datasets.   

<html><body><table><tr><td>Data</td><td># Nodes</td><td>#Edges #Features #Labels</td><td></td></tr><tr><td>Cora</td><td>2,995</td><td>16,346</td><td>2.879 7</td></tr><tr><td>DBLP</td><td>17,716</td><td>105,734</td><td>1,639 4</td></tr><tr><td>CiteSeer</td><td>4,230</td><td>10,674</td><td>602 6</td></tr><tr><td>PubMed</td><td>19,717</td><td>88,648</td><td>500 3</td></tr><tr><td>Computers</td><td>13,752</td><td>491,722</td><td>767 10</td></tr><tr><td>Photos</td><td>7,650</td><td>238,162</td><td>745 8</td></tr><tr><td>CS</td><td>18,333</td><td>163,788</td><td>6.805 15</td></tr><tr><td>Physics</td><td>34,493</td><td>495,924</td><td>8,415 5</td></tr></table></body></html>

Baseline We consider both general statistical calibration approaches temperate, i.e., temperate scaling (Guo et al. 2017), vector scaling (Guo et al. 2017), ensemble temperate scaling (Zhang, Kailkhura, and Han 2020) and SOTA GNN-specific calibration methods, i.e., CaGCN (Wang et al. 2021), GATS (Hsu et al. 2022) and CF-GNN(Huang et al. 2023).

Table 2: Hyper-parameters setting.   

<html><body><table><tr><td>Param.</td><td>Value</td></tr><tr><td>入</td><td>{1e-2,1e-1,1, 10}</td></tr><tr><td>T</td><td>{1e-2, 1e-1, 1, 10}</td></tr><tr><td>K GNN Layers</td><td>{0,1}</td></tr><tr><td>GNN Hidden Dimension</td><td>{1,2,3,4} {16,32,64,128, 256}</td></tr></table></body></html>

Implementation. Our model and baselines are trained on Intel(R) Core(TM) i7-5820K CPU $\textcircled { \alpha } ~ 3 . 3 0 \mathrm { G H z }$ , 64G RAM computing server, equipped with NVIDIA GTX TITAN X graphics cards. All hyper-parameters are chosen via random search. Details of hyper-parameter setting ranges are listed in Table 2.

Metrics Marginal coverage and inefficiency are two commonly used metrics for measuring the performance of CP. Given the test set $\boldsymbol { \mathcal { D } } _ { t e }$ , the marginal coverage metric is defined as follows:

$$
\mathrm { C o v e r a g e } : = \frac { 1 } { | \mathcal { D } _ { t e } | } \sum _ { i \in \mathcal { D } _ { t e } } \delta [ y _ { i } \in C ( x _ { i } ) ]
$$

Coverage vs.Size on CiteSeer Coveragevs.Size on Photo 1.6 TS 1.6 TS 1.4 ETS ETS   
212 CAGN ↓ CaGSN CF-GNN CF-GNN RCP-GNN RCP-GNN 1.0 1.0 0.8 0.8 0.70 0.75 0.80 0.85 0.90 0.70 0.75 0.80 0.85 0.90 Coverage Coverage (a) (b) Coverage vs. Size on Cora Coverage vs. Size on Computers TS 1.8 TS 1.6 ETS 1.6 ETS   
S GATSN e1.4 GATSN RCP-GNN S1.2 RCP-GNN 1.0 1.0 0.8 0.8 0.70 0.75 0.80 0.85 0.90 0.70 0.75 0.80 0.85 0.90 Coverage Coverage (c） (d)

where $\delta [ \cdot ]$ is the indicator function, it is 1 when its argument is true and 0 otherwise. The coverage is empirically guaranteed when marginal coverage exceeds $_ { 1 - \alpha }$ . In cases where it exceeds this threshold, the results improve as they get closer to the target. The marginal coverage guarantee ensures that the output prediction sets for new test points provably include the true outcome with probability at least $1 - \alpha$ . Then we focus on desirable prediction set size to enable further comparisons across CP methods. The inefficiency metric is defined by the size of the prediction set:

$$
\mathrm { I n e f f : = } \frac { 1 } { | \mathcal { D } _ { t e } | } \sum _ { i \in \mathcal { D } _ { t e } } | C ( x _ { i } ) |
$$

where smaller values indicate better performance.

# Results

Marginal Coverage Results. The marginal coverage of different methods are reported in Table 3. All methods use the same pre-trained base model to avoid randomness. The results that achieves the target coverage are marked with underline. The most closest values among covered coverage are marked by bold font. Some observations are introduced as follows:

Temperate scaling, vector scaling and ensemble temperate scaling do not perform well because they lack to aware the topology information in graphs. Although CaGCN and GATS try to integrate topology information to per-node temperature scaling, their performance is still unsatisfactory because they only use CP as a post-training calibration step. Among SOTA methods, only CF-GNN reached target coverage on all datasets. It confirms that fixing prediction sets during training is helpful for reliable uncertainty estimates.

![](images/c3ef79617a08b9108173311a65c7116632fcafa6cfe61c614aafc140e6cbaaf9.jpg)  
Figure 2: Results on different datasets. A lower curve means that the method can achieve the desired coverage using a smaller prediction set size.   
Figure 3: The coverage and inefficiency results with (a) $\tau$ and (b) $\lambda$ changes.

Among all methods, our model successfully reached target coverage on all datasets. Moreover, our model empirically achieves coverage rate closest to target. In summary, our model achieves superior empirical marginal coverage than existing methods.

Inefficiency Results. In Table 4, we summarize the inefficiency reductions of our methods in comparison to other baselines. It can be observed that our model achieve efficiency improvement across datasets with up to $1 1 . 2 8 \%$ reduction in the prediction size. We also empirical present the inefficiency of different methods on various tasks for $\alpha$ ranging from 0.1 to 0.3 in Figure 2. Though CF-GNN try to reduce inefficiency through training with conformal prediction, it does not consistently improve inefficiency across all datasets. Specifically, on Amazon-Photo and AmazonComputers, efficiency becomes even worse. Our RCP-GNN, in contrast, reduces inefficiency consistently.

The reason may be that RCP-GNN constructs and adjusts prediction sets based on ranking and probability of the labels, while CF-GNN only rely on assumptions about the model’s probabilities. Therefore, CF-GNN may not fully capture the model’s uncertainty, which hinders its performance. Additionally, RCP-GNN uses differentiable method to approximate “hard” sorting procedure, which helps to improve the performance and scalability of our framework.

Table 3: Empirical marginal coverage of different methods with $\alpha = 0 . 0 5$ . The result takes the average and standard deviation across 10 runs with 100 calib/test splits. Marked: Covered, Closest.   

<html><body><table><tr><td>Datasets</td><td>Temp. Scale.</td><td>Vector Scale.</td><td>Ensemble TS</td><td>CaGCN</td><td>GATS</td><td>CF-GNN</td><td>RCP-GNN</td></tr><tr><td>Cora</td><td>0.946(.003)</td><td>0.944(.004)</td><td>0.947(.003)</td><td>0.939(.005)</td><td>0.939(.005)</td><td>0.952(.001)</td><td>0.950(.002)</td></tr><tr><td>DBLP</td><td>0.920(.009)</td><td>0.921(.009)</td><td>0.920(.008)</td><td>0.922(.004)</td><td>0.921(.004)</td><td>0.952(.001)</td><td>0.950(.001)</td></tr><tr><td>CiteSeer</td><td>0.952(.004)</td><td>0.951(.004)</td><td>0.953(.003)</td><td>0.949(.005)</td><td>0.951(.005)</td><td>0.953(.001)</td><td>0.951(.001)</td></tr><tr><td>PubMed</td><td>0.899(.002)</td><td>0.899(.003)</td><td>0.899(.002)</td><td>0.898(.003)</td><td>0.898(.002)</td><td>0.953(.001)</td><td>0.950(.002)</td></tr><tr><td>Computers</td><td>0.929(.002)</td><td>0.932(.002)</td><td>0.930(.002)</td><td>0.926(.003)</td><td>0.925(.002)</td><td>0.952(.001)</td><td>0.951(.001)</td></tr><tr><td>Photo</td><td>0.962(.002)</td><td>0.963(.002)</td><td>0.964(.002)</td><td>0.956(.002)</td><td>0.957(.002)</td><td>0.953(.001)</td><td>0.951(.001)</td></tr><tr><td>CS</td><td>0.957(.001)</td><td>0.958(.001)</td><td>0.958(.001)</td><td>0.954(.003)</td><td>0.957(.001)</td><td>0.952(.001)</td><td>0.950(.001)</td></tr><tr><td>Physics</td><td>0.969(.000)</td><td>0.969(.000)</td><td>0.969(.000)</td><td>0.968(.001)</td><td>0.968(.000)</td><td>0.952(.001)</td><td>0.950(.001)</td></tr></table></body></html>

Table 4: Empirical inefficiency results of different methods across various datasets at test time with $\alpha = 0 . 1$ . Marked: First. Second. The average inefficiency reduction relative to the best results of baselines in percentage is reported in parentheses.   

<html><body><table><tr><td>Methods</td><td>Cora</td><td>DBLP</td><td>CiteSeer</td><td>Computers</td><td>Photo</td></tr><tr><td>Temp. Scale.</td><td>1.37</td><td>1.19</td><td>1.14</td><td>1.31</td><td>1.15</td></tr><tr><td>Vector Scale.</td><td>1.36</td><td>1.20</td><td>1.15</td><td>1.25</td><td>1.13</td></tr><tr><td>Ensemble TS.</td><td>1.37</td><td>1.19</td><td>1.14</td><td>1.30</td><td>1.15</td></tr><tr><td>CaGCN.</td><td>1.41</td><td>1.18</td><td>1.19</td><td>1.22</td><td>1.14</td></tr><tr><td>GATS</td><td>1.33</td><td>1.18</td><td>1.16</td><td>1.28</td><td>1.12</td></tr><tr><td>CF-GNN</td><td>1.72</td><td>1.23</td><td>0.99</td><td>1.81</td><td>1.66</td></tr><tr><td>RCP-GNN</td><td>1.18(11.28%↓)</td><td>1.17(0.85%↓)</td><td>0.97(2.02%↓)</td><td>1.20(1.64%↓)</td><td>1.04(7.14%↓)</td></tr></table></body></html>

In summary, our model can significantly reduce the inefficiency while maintaining satisfactory marginal coverage compared with other state-of-the-art methods.

# Ablation Study

We conduct ablations in Table 5 to test main components in RCP-GNN. 1) RCP-THR. It is a variant of RCP-GNN that using THR to compute the conformity scores. Since the THR conformity scores is naturally differentiable w.r.t. the model parameters $\theta$ according to Eq. 7, we only need to ensure the quantile function differentiable in the conformal training stage. 2) RCP-APS. Similar to RCP-THR, this variant leverage APS to compute the conformity scores. The differentiable implementation closely follows the one for RANK outlined in Eq. 14:

$$
V ( x _ { i } , y _ { i } ) = \sum _ { k = 1 } ^ { | \mathcal { V } | } \mathrm { s i g m o i d } ( \frac { \tilde { \mu } _ { y _ { i } } ( x _ { i } ) - \tilde { \mu } _ { k } ( x _ { i } ) } { \tau } ) \tilde { \mu } _ { k } ( x _ { i } )
$$

3) w/o Conf. $T r .$ In order to figure out the power of conformal training step, we remove the conformity loss and replace it with standard prediction loss. Compared with RCPTHR and RCP-APS, our model can achieve pre-defined marginal coverage with satisfactory inefficiency reduction, which demonstrates that the rank-based conformal prediction component is performance-critical to ensure valid coverage guarantees while simultaneously enhancing efficiency. Compared with w/o Conf.Tr., our model achieves consistent efficiency improvement, which demonstrates that prediction sets can be optimized along with conformal training.

# Hyper-Parameter Sensitivity

We also conduct experiments for major hyper-parameters of our model to test the robustness of RCP-GNN. In concrete, the hyper-parameter temperature is changed from 0.01 to 10 and the results show that our model is not sensitivity to the temperature. And we select the median value 1 for its relatively better performance. The hyper-parameter $\lambda$ in Eq. 19 is used to balance the prediction loss and the conformity loss. We report the converge and inefficiency results as $\lambda$ changes from 0.01 to 10 and we can observe that a proper weight of conformity loss can help to inefficiency reduction.

# Related Works

Uncertainty Quantification in Deep Learning. It is important for trustworthy modern deep learning models to mitigate overconfidence (Wang et al. 2020; Slossberg et al. 2022; Jiang et al. 2018). Uncertainty quantification (UQ), which aims to construct model-agnostic uncertain estimates, have great potential in many high-stakes applications (Abdar et al. 2021; Gupta et al. 2021; Guo et al. 2017; Kull et al. 2019; Zhang, Kailkhura, and Han 2020). Most of existing UQ methods rely on the i.i.d assumption. Thus make them be not easily adopt to inter-dependency graph-structure data. Some network principle-based UQ methods (Wang et al. 2021; Hsu et al. 2022) designing for GNNs have been proposed in recent years. However, these UQ methods fail to achieve valid coverage guarantee.

<html><body><table><tr><td rowspan="2">Methods</td><td colspan="2">Cora</td><td colspan="2">CiteSeer</td><td colspan="2">Photo</td></tr><tr><td>Coverage</td><td>Size</td><td>Coverage</td><td>Size</td><td>Coverage</td><td>Size</td></tr><tr><td>RCP-THR</td><td>0.954</td><td>1.65</td><td>0.953</td><td>1.57</td><td>0.953</td><td>1.26</td></tr><tr><td>RCP-APS</td><td>0.952</td><td>2.32</td><td>0.952</td><td>1.86</td><td>0.953</td><td>1.87</td></tr><tr><td>w/o Conf.Tr.</td><td>0.950</td><td>2.08</td><td>0.951</td><td>1.97</td><td>0.951</td><td>2.00</td></tr><tr><td>RCP-GNN</td><td>0.950</td><td>1.92</td><td>0.951</td><td>1.26</td><td>0.951</td><td>1.23</td></tr></table></body></html>

Table 5: Empirical marginal coverage and inefficiency of variants at $\alpha = 0 . 0 5$ . Marked: Best.

Standard Conformal Prediction. Conformal prediction (CP) is early proposed on (Vovk, Gammerman, and Shafer 2005). Compared with other CP framework, e.g., crossvalidation (Vovk 2015) or jackknife (Barber et al. 2021), most approaches follow a split CP method (Lei, Rinaldo, and Wasserman 2013), where a held-out calibration set is necessary. For the reason that it defines faster and more scalable CP algorithms. However, it sacrifices statistical efficiency.

Different variants of CP are struggled to the balance between statistical and computational efficiency. Some contributions made in conformity score function have been explored (Sadinle, Lei, and Wasserman 2019; Angelopoulos et al. 2021; Romano, Sesia, and Candes 2020) for inefficiency reduction. Other studies (Bates et al. 2021; Yang and Kuchibhotla 2024) have explored in the context of ensembles to obtain smaller confidence sets while avoiding to sacrifice the obtained empirical coverage. But these methods do not solve the major limitation of CP methods: the model is independent, leaving CP little to no control over the prediction sets (Guzma´n-rivera, Batra, and Kohli 2012). Recently, the work of (Bellotti 2021) and (Stutz et al. 2022) try to better integrated CP into deep learning models by simulating CP during training to make full use of CP benefits. For GNNs, how to define a trainable calibration step still remains an open space for exploration.

Conformal Prediction for Graphs. Some efforts have been done for CP to GNNs. The work of (Wijegunawardana, Gera, and Soundarajan 2020) adapted CP for node classification to achieve bounded error, and (Clarkson 2023) adapted weighted exchangeability without any lowerbound on the coverage. Furthermore, the assumption for a valid guarantee that the exchangeability between the calibration set and the test set is proved by (H. Zargarbashi, Antonelli, and Bojchevski 2023; Huang et al. 2023), which makes CP applicable to transduction node classification tasks. Different from these works, we leverage the rank of prediction probabilities of nodes to reduce its miscalibration. We also provide its differentiable variant for calibration during training to make prediction sets become aware of network topology information.

# Conclusion

In this work, we extend CP to GNNs by proposing a trainable rank-based CP framework for marginal coverage guaranteed and inefficiency reduction. In future work we will focus on more tasks like link prediction, and extensions to graph-based applications such as molecular prediction and

recommendation systems.