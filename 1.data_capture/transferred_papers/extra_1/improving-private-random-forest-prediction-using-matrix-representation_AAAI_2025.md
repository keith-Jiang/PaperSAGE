# Improving Private Random Forest Prediction Using Matrix Representation

Arisa Tajima1, Joie $\mathbf { W } \mathbf { u } ^ { 2 }$ , Amir Houmansadr1

1University of Massachusetts Amherst 2Independent Researcher atajima@cs.umass.edu, joie.y.wu $@$ gmail.com, amir@cs.umass.edu

# Abstract

We introduce a novel matrix representation for differentially private training and prediction methods tailored to random forest classifiers. Our approach involves representing each root-to-leaf decision path in all trees as a row vector in a matrix. Similarly, inference queries are represented as a matrix. This representation enables us to collectively analyze privacy across multiple trees and inference queries, resulting in optimal DP noise allocation under the Laplace Mechanism. Our experimental results show significant accuracy improvements of up to $40 \%$ compared to state-of-the-art methods.

# Introduction

Recent advances in machine learning services empower users to make inference queries using deployed models through black-box APIs. However, the outcomes of these predictions raise concerns about potential information disclosure regarding the training data, making them vulnerable to attacks like membership inference attacks (Shokri et al. 2017; Hu et al. 2022). To address this, differential privacy (DP) (Dwork and Lei 2009) has emerged as a promising solution, aiming to mitigate the risk of model predictions revealing sensitive information.

Two primary DP-based approaches exist for private prediction: $D P$ training and $D P$ prediction (Ponomareva et al. 2023). DP training (Chaudhuri, Monteleoni, and Sarwate 2011; Jayaraman and Evans 2019; Abadi et al. 2016; Fletcher and Islam 2019) integrates DP noise into model parameters, ensuring that predictions from privately trained models do not reveal information about the underlying training data. By contrast, DP prediction (Dwork and Feldman 2018; Nissim, Raskhodnikova, and Smith 2007; Bassily, Thakkar, and Guha Thakurta 2018) introduces perturbations to the predictions of non-private models. Unfortunately, despite the widespread adoption of DP in machine learning, a significant utility-privacy gap persists between DP-enabled machine learning and its non-private counterparts.

DP-enabled Random Forests: In this paper, we focus on random forest classifiers—a versatile machine learning algorithm known for its strong performance with tabular data like demographic surveys and medical records. We address the challenge of private prediction through both DP training and prediction approaches while maintaining high accuracy.

DP random forest classifiers have been extensively studied. Past approaches primarily differ in their use of base tree algorithms and node splitting functions to balance privacy budget usage and accuracy (Patil and Singh 2014; Jagannathan, Pillaipakkamnatt, and Wright 2009; Fletcher and Islam 2017, 2019; Hou et al. 2019). Regarding the treebuilding process, there are greedy approaches, where optimal split points are determined (Hou et al. 2019; Patil and Singh 2014) and random approaches, where split attributes are chosen randomly to save privacy budget for leaf nodes (Fletcher and Islam 2017; Jagannathan, Pillaipakkamnatt, and Wright 2009; Holohan et al. 2019). The latter method, which we focus on, is known as random decision trees. It is predominantly adapted in recent works for its consistently strong performance; see the comprehensive survey by Fletcher et al. (Fletcher and Islam 2019).

While distributing the privacy budget is crucial for DP random forests, many approaches allocate it equally among trees (Patil and Singh 2014; Jagannathan, Pillaipakkamnatt, and Wright 2009; Fletcher and Islam 2017; Hou et al. 2019). However, this per-tree privacy analysis faces challenges when a large number of trees are used, resulting in low accuracy despite the ensemble learning principle that training more trees generally improves performance.

To our knowledge, no prior work has addressed DP prediction techniques specifically for random forests. Alternatively, a well-known model-agnostic DP prediction method—the subsample-and-aggregate framework—allocates the privacy budget equally to each inference query (Dwork and Feldman 2018; Nissim, Raskhodnikova, and Smith 2007). However, this heuristic allocation results in poor accuracy when many inference queries are required (van der Maaten and Hannun 2020).

# Main Contributions

Existing methods for DP training and DP prediction add suboptimal levels of noise, hampering utility. Our primary contribution lies in addressing the inefficiencies of current methods by translating random forest training and prediction into matrix multiplication. This new representation allows us to optimize solutions for training models and predicting class labels under DP, introducing finer-grained noise than

the state-of-the-art.

We introduce DP batch training, which allocates the privacy budget across an ensemble of trees. The key insight behind our approach is the observation that some leaf values can be expressed as linear combinations of others. We optimize budget allocation based on leaf values along decision paths, preserving high accuracy even as the number of trees learned increases. Our allocation strategy is dataindependent, unlike a recent method that uses weighted privacy budget allocation for trees, which incurs budget expenditure (Li et al. 2022). Thus, our approaches offer privacyfree hyperparameter tuning, a significant advantage over existing methods including DP-SGD (Abadi et al. 2016).

Additionally, we introduce $D P$ batch prediction which takes into account prediction results collectively rather than isolating individual queries. This technique optimizes DP noise addition across a set of inference queries for majority voting in an ensemble. Batch privacy analysis closes the performance gap between DP prediction performance and DP training due to the former’s limit on inference queries. Our results show that this approach maintains good accuracy, even as the number of inference queries increases.

We validate our methods on real-world datasets, demonstrating a significant accuracy improvement of up to $40 \%$ compared to existing approaches. When tested on our Car dataset, both of our DP batch training and DP batch prediction approaches achieve $8 5 \%$ accuracy under a privacy budget of $\epsilon = 2$ when predicting 345 test samples on 128 random decision trees, exceeding the accuracy of existing solutions by $30 \%$ . Finally, our matrix representation can be extended to work within the subsample-and-aggregate framework, yielding an improvement of up to $50 \%$ . Our code and technical appendix will be available at: https://github.com/ arisa77/mrf-public.git.

# Background

# Data and Schema

Schema. Consider a schema consisting of $d$ attributes: $\{ A _ { 1 } , A _ { 2 } , \ldots , A _ { d } \}$ . Each attribute $A _ { i }$ has a finite domain $\Phi ( A _ { i } )$ of size $n _ { i }$ . The full domain size is then $\textstyle \prod _ { i = 1 } ^ { d } n _ { i }$ . For clarity, we assume the first $d - 1$ attributes represent feature attributes with domains denoted as $\mathcal { X } _ { i }$ for $i \in [ d - 1 ]$ . Let $\begin{array} { r } { \mathcal { X } = \prod _ { i = 1 } ^ { d - 1 } \mathcal { X } _ { i } } \end{array}$ , and $\begin{array} { r } { n = \prod _ { i = 1 } ^ { d - 1 } n _ { i } } \end{array}$ . The last attribute is the targe class attribute, denoted as $\mathcal { V }$ with $k$ variables. We distinguish $\chi$ as the $d { - } 1$ dimensional feature space and $y$ as the 1-dimensional target space.

Example 1 (Tennis Schema) We use a simplified tennis dataset with attributes outlook, windy, play and domains $\begin{array} { r c l } { \Phi ( o u t l o o k ) } & { = } & { \{ s u n n y , o \nu e r c a s t , r a i n y \} } \end{array}$ , $\begin{array} { r l } { \Phi ( w i n d y ) } & { { } = } \end{array}$ $\{ f a l s e , t r u e \}$ , and $\Phi ( p l a y ) ~ = ~ \{ n o , y e s \}$ . The features are ’outlook’ and ’windy’, and the target is ’play’. The feature domain $\chi$ contains all tuples from $\Phi ( o u t l o o k ) \times \Phi ( w i n d y ) ,$ , resulting in $\mathcal { X } = \{ ( s u n n y , f a l s e )$ , (sunny, true), (overcast, false), (overcast, true), (rainy, false), (rainy, true)}. The target domain is $\mathcal { Y } = \{ n o , y e s \}$ .

Data Matrix. We have a sensitive training dataset consisting of $N$ individuals, represented as an $N \times d$ matrix $X$ .

Each row $X _ { i }$ is an individual record and $X _ { i , j } \in \Phi ( A _ { j } )$ is the value of attribute $A _ { j }$ . We often represent this dataset $X$ as a 2-way contingency table of feature variables $\chi$ by target variables $y$ , denoted as a matrix $\mathbf { D } _ { X } \in \mathbb { R } ^ { n \times k }$ . We may drop the subscript $X$ and write $\mathbf { D }$ if the context is clear. This matrix captures the frequency of every possible tuple $( x , y ) \in \mathcal { X } \times \mathbf { \bar { \mathcal { X } } }$ in $X$ . Although the frequency representation is favored for mathematical convenience, our implementation uses the record-by-record format for efficiency.

Example 2 (Tennis Dataset) Consider the tennis dataset with six samples shown in Figure 1a. The corresponding $\boldsymbol { \mathscr { \sigma } }$ by 2 frequency matrix D is illustrated in Figure 1c. For instance, $\mathbf { D } _ { 1 , 1 } = 1$ indicates a single sample with features (sunny, false) and a target class of no.

# Differential Privacy

Differential privacy (DP) (Dwork, Roth et al. 2014) is the defacto standard for data privacy, formally defined as follows.

Definition 1 (Differential Privacy) $A$ randomized mechanism $\mathcal { M }$ satisfies $\epsilon$ -DP if for any two neighboring datasets $X , X ^ { \prime } \in { \mathcal { D } }$ and for any subset of outputs $S \subseteq R a n g e ( { \mathcal { M } } )$ it holds that: $\operatorname* { P r } \left[ \mathcal { M } ( X ) \in \mathcal { S } \right] \leq e ^ { \epsilon } \operatorname* { P r } \left[ \mathcal { M } ( X ^ { \prime } ) \in \mathcal { S } \right]$ .

DP offers valuable properties such as the post-processing and composition theorem (Dwork, Roth et al. 2014), which we utilize in this work. Following the literature on DP random forests, we focus on employing the Laplace mechanism (Fletcher and Islam 2019; Holohan et al. 2019).

Matrix Mechanism. Matrix Mechanism (Li et al. 2015) is a variant of the Laplace mechanism designed for answering input queries defined by matrix W by using a set of underlying queries A such that there exists some X for which $\mathbf { W } = \mathbf { X } \mathbf { A }$ . A is referred to as the strategy matrix. The naive case where $\mathbf { A } = \mathbf { W }$ corresponds to the vectorized Laplace mechanism. Heuristics for choosing the best strategy matrix have been widely studied (Xiao, Gardner, and Xiong 2012; Xiao, Wang, and Gehrke 2010).

Definition 2 (Matrix Mechanism) Given $\mathbf { \xi } _ { l }$ by $n$ dataindependent query matrix W, m by $n$ data-independent strategy matrix A (possibly induced by W), and n by $k$ data matrix $\mathbf { D } _ { X }$ , the following Matrix Mechanism satisfies $\epsilon { - } D P .$ $\mathcal { M } _ { \mathbf { A } } ( \mathbf { W } , \mathbf { D } _ { X } ) = \mathbf { W } \mathbf { D } _ { X } + \mathbf { W } \mathbf { A } ^ { + } \mathrm { L a p } ( | | \mathbf { A } | | _ { 1 } / \epsilon ) ^ { \tilde { m } \times k }$ .

The sensitivity of a query matrix is defined as its L1 norm $| | \mathbf { A } | | _ { 1 }$ . ${ { \bf A } ^ { + } }$ denotes the pseudoinverse of A. The original work utilizes a data vector instead of a data matrix, although both expressions are essentially interchangeable. The meansquared error of query answers to W under the strategy matrix A is given as follows, independent of the actual data input. $| | \cdot | | _ { F }$ denotes the Frobenius norm.

# Definition 3 (Error of strategy query answering)

Given query matrix W, strategy matrix A, the total mean squared error of Matrix Mechanism is given as: $\mathrm { E r r } _ { \epsilon } ( \bar { \mathbf { W } } , \mathbf { A } ) = 2 / \epsilon ^ { 2 } | | \bar { \mathbf { A } } | | _ { 1 } ^ { 2 } | | \mathbf { W } \mathbf { A } ^ { + } | | _ { F } ^ { 2 }$ .

The state-of-the-art minimizes this error through parameterized optimization, identifying a $( p + n ) \times n$ strategy matrix A by finding a $p \times n$ parameter matrix (McKenna et al. 2018). We treat the optimization routine as a black box, denoted as $\mathbf { A } \gets \mathrm { O P T } _ { p } \mathbf { \bar { ( W ) } }$ , where $p$ is a hyperparameter.

![](images/983e2ce6c8a406e2d2213d4864c6c7191a6b4931fb8fe666ade3730e3b482a2a.jpg)  
Figure 1: Matrix multiplication for computing leaf values for training (c) and weighted voting for prediction (d) in random decision trees. The matrices D, T, and $\mathbf { Q }$ correspond to the training data (a), tree decision paths (b), and test samples (d).

# Random Decision Trees

Random decision tree models were initially proposed as efficient classifiers for large databases in non-private settings (Fan et al. 2003; Geurts, Ernst, and Wehenkel 2006). They have since become a dominant algorithm for state-of-the-art DP random forests due to their data-independent tree construction (Jagannathan, Pillaipakkamnatt, and Wright 2009; Holohan et al. 2019; Fletcher and Islam 2017). Unlike traditional decision trees such as ID3 and CART, random decision trees are constructed by randomly selecting test attributes for nodes. Because this attribute selection occurs before examining any training data, there is no need to allocate a privacy budget for finding optimal split points.

Base Classifiers. We consider $\tau$ random decision trees each with depth $h$ , denoted as $F = \{ T _ { 1 } , \dots , T _ { \tau } \}$ . We may denote $F _ { S }$ as those built over a specific feature subset $S$ . We may build $q$ ensembles of such random decision trees, each associated with a random feature subset, denoted as $F _ { . } = \{ F _ { S _ { j } } \} _ { j = 1 } ^ { q }$ . The tree structure, including random feature selection, is pre-computed using schema information alone. Training data $X$ is then used to calculate class count distributions at all leaves. Each tree is fitted on the same training dataset unless stated otherwise. We provide the full algorithm of random decision trees in the technical appendix. During prediction, the trained random decision trees $F$ take a sample $s \in \mathcal { X }$ and output a predicted label $y \in \mathcal { V }$ , denoted as $y \gets F ( s )$ . Each tree casts a vote, and the label with the most votes is chosen as the final prediction.

Private Prediction and Problem Statement. Given $b$ inference queries $Q \in { \mathcal { X } } ^ { b }$ , our goal is to release prediction results $\mathbf { y } \gets F ( Q )$ (abusing the notation above) under DP to limit information leakage about the training data $X$ from the predictions. We will introduce formulations for both $D P$ training and $D P$ prediction. DP training is achieved by making the random forest $F$ DP. Thus, the subsequent prediction algorithm maintains privacy due to the post-processing theorem. DP prediction, on the other hand, is achieved by adding DP noise to the predictions of the non-DP random forest $F$ .

# Matrix Random Forests

We introduce a novel approach to representing random forest training and prediction using matrices, which we call Matrix Random Forest $( M { \cdot } R F )$ . This approach is essential in generating the optimal amount of DP noise, as we will detail later.

# Inference Query Matrix

A sample $s \in \mathcal { X }$ is represented as an indicator vector $\mathbf { q }$ of length $n$ over $\mathcal { X }$ , where the cell at the corresponding tuple is set to 1. Otherwise, it is set to 0. Thus, a set of $b$ inference queries $\{ s _ { 1 } , \dotsc , s _ { b } \} \in { \mathcal { X } } ^ { b }$ can be expressed as a $b \times n$ matrix $\mathbf { Q } \ = \ ( \mathbf { q } _ { 1 } ^ { \intercal } , \ldots , \mathbf { q } _ { b } ^ { \intercal } )$ . In contrast to the data matrix $\mathbf { D }$ , the inference query matrix $\mathbf { Q }$ is not treated as sensitive.

Example 3 (Inference Query) Figure 1d shows three unlabeled samples and the corresponding $3 \times 6$ inference query matrix Q. For instance, the first row of $\mathbf { Q }$ encodes the tuple (sunny, true) from the dataset.

# Decision Path Query Matrix for Training

We introduce the decision path matrix that encodes every root-to-leaf decision path in random decision trees. Each decision path represents a predicate $P ( x )$ over the feature domain $\chi$ . This predicate can be represented as a binary vector of length $n = | { \mathcal { X } } |$ , denoted as $\mathbf { p }$ , where $\mathbf { p } _ { i } = 1$ if the corresponding element in $\chi$ is in the truth set (the set of all elements in $\chi$ that make $P ( x )$ true); otherwise $\mathbf { p } _ { i } = 0$ . For instance, if $P ( x )$ stands for “outlook is sunny” in Example 1, the truth set is $\{$ (sunny, false), (sunny, true) $\}$ and thus the corresponding predicate results in $\mathbf { p } = ( 1 , 1 , 0 , 0 , 0 , 0 )$ .

A predicate is used in a counting query to evaluate the number of instances in the dataset satisfying its corresponding decision path. For a single predicate $\mathbf { p }$ , the expression $\mathbf { p } \mathbf { \breve { D } } \in \mathbb { R } ^ { k }$ will yield instance counts for each class label.

Thus, a set of $o$ decision paths can be organized into an $o$ by $n$ matrix where each row corresponds to a predicate, denoted as $\mathbf { T } = ( \mathbf { p } _ { 1 } ^ { \mathsf { T } } , \ldots , \mathbf { p } _ { o } ^ { \mathsf { T } } )$ . The matrix encodes the dataindependent tree structure. The class counts at every leaf node can be computed as $\mathbf { C } = \mathbf { T } \mathbf { D } \in \mathbb { R } ^ { o \times k }$ . Each $\mathbf { C } _ { i , j }$ represents the frequency of data instances that reach the $i$ -th leaf node whose class label is $y _ { j } \in \mathcal { V }$ ; see Figure 1c. We may write $\mathbf { C _ { D } }$ to emphasize its dependence on $\mathbf { D }$ .

# Decision Path Query Matrix for Prediction

For prediction, we construct the decision path matrix for individual instances. Given the decision path matrix $\textbf { T } \in$ $\mathbb { R } ^ { o \times n }$ and the inference query matrix $\mathbf { Q } \in \mathbb { R } ^ { b \times n }$ introduced earlier, their matrix multiplication $\mathbf { W } = \mathbf { Q } \mathbf { T } ^ { \intercal } \in \mathbb { R } ^ { b \times o }$ produces the specific decision paths for each sample. Specifically, $\mathbf { W } _ { i , j } ~ = ~ 1$ if the $i$ -th sample reaches the $j$ -th leaf node on the $j$ -th decision path defined by $\mathbf { T } _ { j }$ ; otherwise, $\mathbf { W } _ { i , j } = 0$ . Thus, each row vector $\mathbf { W } _ { i }$ encodes the decision paths taken by sample $i$ across the random decision trees.

# Weight Voting Matrix Operation

We introduce the matrix operations underlying the majority voting process for random decision trees. This method, which we call weight voting, involves aggregating leaf values across trees to make predictions. The aggregated votes represent a weighted count of data points that agree on the predicted class for each sample.

Given random decision trees, let $\mathbf { T }$ denote the decision path matrix and $\mathbf { C _ { D } }$ the leaf value matrix. For an inference query matrix $\mathbf { Q }$ , the weight votes for each target class are calculated using the matrix product: $\mathbf { V } = \mathbf { Q } \mathbf { T } ^ { \mathsf { T } } \mathbf { C } _ { \mathbf { D } } \in$ $\mathbb { R } ^ { b \times k }$ . Here $\mathbf { V } _ { i , j }$ represents the number of training instances that agree with the class label $y _ { j } \in \mathcal { D }$ for the $i$ -th inference query. When $\mathbf { C } = \mathbf { T D }$ , this operation can be expressed in terms of $\mathbf { D }$ as: $\mathbf { V } = \mathbf { Q } \mathbf { T } ^ { \top } \mathbf { T } \mathbf { D }$ . Here, QT⊺T represents weighted queries that aggregate tuples in the feature domain; see Figure 1f for an example matrix.

# DP M-RF Training and Prediction

In this section, we present novel techniques for DP training and DP prediction of random forests. Our approach leverages the matrix representation formulated in the previous sections, enabling the derivation of optimal solutions that significantly improve the accuracy of DP random forests.

# DP Batch Training Approach

We present our DP random forest training, referred to as batch training. This method provides DP leaf values while maintaining high accuracy by optimizing a batch of decision path queries. The leaf labels are computed from DP leaf values, ensuring the resultant random forests satisfy DP.

Algorithm 1: DP M-RF Training   

<html><body><table><tr><td>Input: Data matrix D,decision path matrix T,privacy bud gete.</td></tr><tr><td>Output: DPleaf label vector 1</td></tr><tr><td>1:A←OPT(T)</td></tr><tr><td>2: C = TD + TA+Lap(IIA|l1/e)m×k</td></tr><tr><td>3: Ii = arg max1≤j≤k Ci,j, ∀ leaf i.</td></tr></table></body></html>

Optimizing Decision Paths Utilizing our matrix representation, leaf values are computed as the matrix product $\mathbf { C } = \mathbf { T D }$ , where $\mathbf { T }$ represents root-to-leaf paths in random decision trees. This serves as a query matrix, where each row vector counts the number of training data instances satisfying the classification rule of the corresponding leaf node. Many existing DP approaches apply the Laplace mechanism by adding the Laplace noise to the leaf values with an equal privacy budget allocation: $\mathrm { T D } + \mathrm { L a p } ( \tau / \epsilon ) ^ { o \times k }$ , assigning each tree a budget of $\tau / \epsilon$ . However, this method can lead to suboptimal accuracy if decision paths in one tree are linearly dependent on those in others, wasting the privacy budget.

To address this, we optimize leaf-level counts by finding an optimal strategy $\mathbf { A } \gets \mathrm { O P T } ( \mathbf { T } )$ . By specifying the decision path query matrix $\mathbf { T }$ into the optimization procedure OPT, we derive an alternative strategy matrix $\mathbf { A } \in \mathbb { R } ^ { m \times n }$ that minimizes the error in estimating class counts at leaf nodes, i.e., $\operatorname { E r r } ( \mathbf { T } , \mathbf { A } )$ . Under this strategy, $\epsilon$ -DP leaf values are estimated as $\begin{array} { r } { \tilde { \mathbf { C } } = \mathbf { T } \mathbf { D } + \mathbf { T } \mathbf { A } ^ { + } \mathrm { L a p } ( | | \mathbf { A } | | _ { 1 } / \epsilon ) ^ { m \times k } } \end{array}$ .

DP Matrix Random Forest Training Algorithm 1 presents our approach for training DP random forests. The decision path matrix $\mathbf { T }$ is precomputed by building random decision trees. With the data matrix $\mathbf { D }$ and the decision path matrix $\mathbf { T }$ , we estimate leaf values under DP, following our above optimization approach. The resulting DP leaf values have the following error, a direct implication from Definition 3. Finally, the most frequent class is assigned as the label for each leaf node based on the noisy leaf values.

Theorem 4 Leaf values $\tilde { \mathbf { C } }$ resulting from Algorithm 1 have MSE of $\mathrm { E r r } _ { \epsilon } ( \mathbf { T } , \mathbf { A } )$ .

# Theorem 5 Algorithm 1 satisfies ϵ-DP.

The strategy matrix selection in Line 1 incurs no privacy loss since it only depends on the data-independent decision path matrix T. The computation on leaf values in Line 2 satisfies $\epsilon$ -DP, following the privacy of the Matrix Mechanism. Finally, from the post-processing theorem, the computation of leaf labels in Line 3 does not degrade privacy.

# DP Batch Prediction Approach

We introduce our DP random forest prediction approach, referred to as: batch prediction. Our approach predicts labels for given inference queries under DP while maintaining high accuracy. Unlike DP batch training, which optimizes leaf values for classifiers, DP batch prediction focuses on optimizing prediction results for specified inference queries.

<html><body><table><tr><td>Algorithm2:DPM-RFPrediction</td></tr><tr><td>Input:Data matrix D,decision path matrix T,inference</td></tr><tr><td>querymatrix Q,privacy budget e.</td></tr><tr><td>Output: DP predicted label vector y</td></tr><tr><td>1:A ←OPT(QT'T) 2: V = QTTD + QTTA+Lap(|IA|l1/e)mxk</td></tr></table></body></html>

Optimizing Decision Paths for Prediction The random forest prediction problem using weight voting can be expressed as the matrix product: $\mathbf { V } \ = \ \mathbf { Q } \mathbf { T } ^ { \mathsf { T } } \mathbf { T } \mathbf { D }$ , where $\mathbf { Q }$ denotes the inference query matrix, and $\mathbf { T }$ represents decision paths in the trees. We consider $\mathbf { Q T ^ { \mathsf { T } } T }$ as a query matrix where each vector aggregates instance counts (i.e., weighted votes) at predicted leaf nodes across all trees for the corresponding inference query. Using the optimization procedure OPT, we find a $m \times n$ strategy matrix $\mathbf { A } \gets \mathrm { O P T } ( \mathbf { W } )$ , where $\mathbf { W } = \mathbf { Q } \mathbf { T } ^ { \mathsf { T } } \mathbf { T }$ , that minimizes the error of answering votes for inference queries, i.e., $\mathrm { E r r } ( \mathbf { W } , \mathbf { A } )$ . Under this strategy, $\boldsymbol { \epsilon } { - } \mathbf { D } \mathbf { P }$ vote counts are estimated as: $\tilde { \mathbf { V } } = \mathbf { W } \mathbf { D } + \mathbf { \Sigma }$ $\mathbf { W } \mathbf { A } ^ { + } \mathrm { L a p } ( | | \mathbf { A } | | _ { 1 } / \epsilon ) ^ { m \times k }$ .

DP Matrix Random Forest Prediction Algorithm 2 shows our DP prediction approach for random forests. Given non-private random decision trees, the resulting tree structure are encoded into a decision path matrix T. Training data and inference queries are similarly transformed into matrices $\mathbf { D }$ and $\mathbf { Q }$ . We then estimate votes tallied across the forest for every inference query under DP, following the above optimization approach. The resulting DP vote counts have the following error, a direct implication from Definition 3. Lastly, class labels with the majority noisy votes are returned as the final prediction results.

Theorem 6 The vote counts V in Algorithm 2 have MSE of $\operatorname { E r r } _ { \epsilon } ( \mathbf { Q } \mathbf { T } ^ { \intercal } \mathbf { T } , \mathbf { A } )$ .

Theorem 7 Algorithm 2 satisfies ϵ-DP. Similar to the privacy proof of Theorem 5, the privacy of Algorithm 2 primarily follows from the privacy of the Matrix Mechanism.

# Optimizing Subsample-and-Aggregate Framework

We enhance the subsample-and-aggregate method (Dwork and Feldman 2018) using our batch prediction technique. In this approach, non-private random decision trees are fitted on disjoint training datasets to estimate weighted votes $\mathbf { V } = \mathbf { Q } \mathbf { T } ^ { \intercal } \mathbf { C }$ under DP. Since $\mathbf { C } \neq \mathbf { T D }$ , the above DP batch prediction is not directly applicable. To address this, our approach identifies an $m { \times } o$ strategy matrix $\mathbf { A } \gets \mathrm { O P T } ( \mathbf { Q } \mathbf { T } ^ { \bar { \top } } )$ that minimizes $\mathrm { E r r } _ { \epsilon } ( \mathbf { Q } \mathbf { T } ^ { \intercal } , \mathbf { A } )$ . The DP vote counts are then computed as $\begin{array} { r } { \tilde { \mathbf { V } } = \mathbf { Q } \mathbf { T } ^ { \top } \mathbf { C } + \mathbf { Q } \mathbf { T } ^ { \top } \mathbf { A } ^ { + } \mathrm { L a p } ( \| \mathbf { A } \| _ { 1 } / \epsilon ) ^ { m \times k } } \end{array}$ . This method ensures query sensitivity remains independent of the number of inference queries, unlike the existing method, which allocates an equal budget among queries, significantly degrading accuracy with more queries. Further details appear in the technical appendix.

# Our Full DP M-RF Framework

Our complete framework for private prediction with random decision trees is shown in Algorithm 3, capturing both DP batch training and DP batch prediction. The framework involves: 1) building trees, 2) optimizing strategies, 3) fitting training data, and 4) making predictions, detailed below.

We construct $q$ ensembles of random decision trees, where each ensemble $F _ { i }$ is built over a random feature subset $S _ { i }$ of size $\bar { d }$ . Our method involves performing strategy optimizations within each ensemble over the schema $S _ { i }$ , allowing potentially expensive computations to be performed in parallel across ensembles. The best strategy matrix $\mathbf { A } ^ { ( i ) }$ is chosen from OPT for generating refined DP noise. The query matrix $\mathbf { W } ^ { ( i ) }$ is instantiated with $\mathbf { T } ^ { ( i ) }$ or $\mathbf { Q } ( \mathbf { T } ^ { ( i ) } ) ^ { \top } \mathbf { T } ^ { ( i ) }$ , depending on batch training or batch prediction, recalling the prior sections. Here, $\mathbf { T } ^ { ( i ) }$ denotes the decision path matrix associated with ensemble $i$ . The optimality of our approach comes from the theoretical optimality of the strategy matrix from OPT for a fixed strategy. The derived strategy matrix remains optimal within the constraints of each individual ensemble.

Private training data is fit on each ensemble forest by updating the leaf values and labels. For DP training, the refined DP noise is added to leaf values, and the resulting random decision trees $\tilde { F } _ { i }$ are privatized. For DP prediction, the DP noise is added to the predicted vote counts, with predictions made on non-DP random decision trees. Finally, prediction labels are determined collectively across the $q$ ensembles.

Algorithm 3 can capture our improved subsample-andaggregate method as well by setting $\mathbf { W } ^ { ( i ) } = \mathbf { Q } ( \mathbf { T } ^ { ( i ) } ) ^ { \top }$ and fitting disjoint subsets of data on each ensemble forest.

For privacy, following Theorem 5 and 7, each ensemble algorithm with a privacy budget of $\epsilon / q$ satisfies $\epsilon / q$ -DP for batch training and prediction. Thus, from the composition theorem, Algorithm 3 satisfies $\epsilon$ -DP.

Complexity. The primary overhead of DP M-RF in Algorithm 3 comes from the optimization for refining DP noise, crucial for balancing accuracy and efficiency. This process is parallelizable across ensembles for efficiency. Assuming each feature has $n$ values with a total domain size of $n ^ { \bar { d } }$ , the size of $\mathbf { W } ^ { ( i ) }$ is $o \times n ^ { \bar { d } }$ , $b \times n ^ { \bar { d } }$ , $b \times o$ for DP batch training, DP batch prediction, and subsample-and-aggregate. Here, $o$ is the total number of leaves per ensemble and $b$ is the number of inference queries. Our implementation adopts an implicit matrix representation, effectively reducing the matrix size to e.g., $o \times { \bar { d } } n$ for batch training (McKenna et al. 2018).

The optimization cost depends on the chosen strategies, producing a $( p _ { i } + n ^ { \bar { d } } ) \times n ^ { \bar { d } }$ matrix $\mathbf { A } ^ { ( i ) }$ with $p _ { i } = O ( \bar { n } ^ { \bar { d } } )$ . The space and time complexity of noise generation in Line 6 are both $O ( n ^ { \bar { d } } )$ for DP batch training and prediction; for subsample-and-aggregate, they are $O ( o { + } b )$ and $O ( ( o + b ) o )$ , independent of domain sizes.

Tree building and optimization do not require the actual training data and can be preprocessed efficiently. Thus, the overhead for DP training and prediction is negligible. Moreover, matrices $\mathbf { Y } ^ { ( i ) }$ and $\mathbf { V } ^ { ( i ) }$ are computed without materializing the data matrix $\mathbf { D }$ , ensuring efficient space complexity.

Input: features $S$ , training data $X$ , inference queries $Q$ , privacy budget $\epsilon$ , number of ensembles $q$ , number of trees $\tau _ { 1 } , \ldots , \tau _ { q }$ , tree depth $h _ { 1 } , \ldots , h _ { q }$ , max feature size $\bar { d }$ .   
Output: DP predicted label vector $\tilde { \mathbf { y } }$ 1: for every ensemble $i = 1 \dots q$ do 2: $S _ { i }  \bar { d }$ random features from $S$ 3: $\begin{array} { r l } & { F _ { i } \gets \cup ^ { \tau _ { i } } \mathbf { B U I L D T R E } ( S _ { i } , h _ { i } ) } \\ & { \mathbf { W } ^ { ( i ) } \gets \mathbf { W o R K L O A D M A T R I X } ( F _ { i } ; Q ) } \\ & { \mathbf { A } ^ { ( i ) } \gets \mathbf { O P T } _ { p _ { i } } \big ( \mathbf { W } ^ { ( i ) } \big ) } \\ & { \mathbf { B } ^ { ( i ) } \gets \mathbf { W } ^ { ( i ) } \left( \mathbf { A } ^ { ( i ) } \right) ^ { + } \mathbf { L a p } ( q | | \mathbf { A } ^ { ( i ) } | | _ { 1 } / \epsilon ) ^ { m _ { i } \times k } } \\ & { F _ { i } \gets \mathbf { U P D A T E L A V E S } ( X , F _ { i } ) } \end{array}$ 4: 5: 6: 7: 8: if DP Training then 9: Let $\mathbf { Y } ^ { ( i ) }$ be leaf values of $F _ { i }$   
10: $\tilde { F } _ { i } \gets$ update $F _ { i }$ with leaf values $\mathbf { Y } ^ { ( i ) } + \mathbf { B } ^ { ( i ) }$   
11: $\tilde { \mathbf { V } } ^ { ( i ) }  \tilde { F } _ { i } ( Q )$   
12: else if DP Prediction then   
13: $\mathbf { V } ^ { ( i ) }  F _ { i } ( Q ) ; \tilde { \mathbf { V } } ^ { ( i ) }  \mathbf { V } ^ { ( i ) } + \mathbf { B } ^ { ( i ) }$   
14: end if   
15: end for   
16: $\begin{array} { r } { \tilde { \mathbf { y } } _ { i } = \arg \operatorname* { m a x } _ { 1 \leq l \leq k } \sum _ { j = 1 } ^ { q } \tilde { \mathbf { V } } _ { i , l } ^ { ( j ) } , \forall } \end{array}$ inference query $i$

Setting Hyperparameters. Following Theorem 4 and 6, the utility of Algorithm 3 is measured by $\mathrm { E r r } _ { \epsilon / q } ( \mathbf { W } ^ { ( i ) } , \mathbf { A } ^ { ( i ) } )$ . This metric corresponds to errors in leaf values for DP training and vote counts for DP prediction. The error rates are affected by the matrix $\mathbf { W } ^ { ( i ) }$ , which is defined by the number of trees $\tau _ { i }$ , depth $h _ { i }$ , and schema information, including the number of features $\bar { d } .$ . Since this error metric does not depend on the actual training data, we can effectively perform privacy-free hyperparameter tuning without consuming any privacy budget.

The hyperparameters $q$ (number of ensembles) and $\bar { d }$ (size of the feature subspace) involve a trade-off between accuracy and efficiency. Smaller feature sets enable more efficient optimizations but might exclude important features, reducing learning capability. Increasing the number of ensembles reduces the privacy budget per ensemble, potentially degrading accuracy. For smaller datasets, we recommend using a single ensemble forest $( q = 1 )$ ) with the original feature set $\bar { \boldsymbol { d } } = \mathsf { \bar { \Pi } } | \boldsymbol { S } | )$ to maximize accuracy. For larger datasets, $\bar { d }$ and $q$ should be chosen to balance efficiency and accuracy.

# Experiments

We empirically evaluate the performance of our DP training and prediction techniques for random forests, demonstrating higher accuracy compared to competing techniques.

# Experimental Setup

Datasets. We use six popular classification datasets from the UCI ML Repository (Kelly, Longjohn, and Nottingham 2023) with feature dimensions ranging from 4 to 128: Car, Iris, Scale, Adult, Heart, and Mushroom. Certain datasets with continuous values were preprocessed using public domain knowledge, including discretization. The Adult dataset was already discretized (Chang and Lin 2011).

Implementation and Competing Techniques. We evaluate Algorithm 3 against various competing techniques. For consistency, all DP methods use random decision trees. All implementations are in Python and experiments were conducted on a MacBook Air (M2 chip with 16GB RAM). We adopt a multi-way tree structure as in ID3. It can be easily extended to binary trees.

We compare our DP batch training against two widelyused methods. The first baseline employs the same batch training but applies the Laplace mechanism, commonly used in state-of-the-art works (Jagannathan, Pillaipakkamnatt, and Wright 2009; Maddock et al. 2022). This corresponds to Algorithm 3 with $\mathbf { B } ^ { ( i ) } \ = \ \mathrm { L a p } ( q \tau _ { i } / \epsilon ) ^ { o \times k }$ and $\dot { q } = 1 , \bar { d } = \check { | } S |$ . The second baseline trains each tree on disjoint datasets using the Laplace mechanism, as seen in prior works (Holohan et al. 2019; Fletcher and Islam 2017). For all methods above, we adopt the standard hard voting for prediction. We do not compare against work that uses a weaker definition of DP, such as (Rana, Gupta, and Venkatesh 2015). For DP prediction techniques, we compare our DP batch prediction and subsample-and-aggregate approach from Algorithm 3 against the existing subsampleand-aggregate (Dwork and Feldman 2018). Additionally, we benchmark our techniques against an optimized nonprivate random forest algorithm, the Extra-Trees classifier from scikit-learn to obtain an empirical upper bound on accuracy for private algorithms (Pedregosa et al. 2011). For runtime comparisons, we evaluate the non-private version of Algorithm 3 to ensure consistency.

# Results

We measure the accuracy and runtime of Algorithm 3 with at least 5 trials under fixed parameters. Unless explicitly denoted, each dataset is split into train and test subsets with a 80:20 ratio. Each ensemble consists of $\tau / q$ trees of depth $h$ .

Main Results. Figure 2 shows the test accuracy of our DP batch training and prediction techniques, varying values of $\epsilon$ . We consider commonly used values of $\tau = 6 4 \sim 1 2 8$ . We use $q = 1 , \bar { d } = d - 1$ for small to medium-sized datasets like Car, Iris, and Balance; for larger datasets, multiple ensembles are employed. Our optimized approaches consistently outperform the baselines, showing a significant accuracy improvement of $20 \%$ for small to medium-sized datasets. Larger datasets such as Heart, Mushroom, and Adult show a notable accuracy improvement of $10 \%$ , particularly with smaller $\epsilon$ values. Our novel matrix representation improves DP batch training and prediction, enabling the learning of numerous base classifiers and the prediction of a large number of test samples, all without sacrificing accuracy.

Increasing the number of trees does not negatively impact the accuracy of DP Batch training. With random forests, predictive power is supposed to increase with the number of trees. However, DP batch and disjoint training with the Laplace mechanism both suffer from tree scaling issues, making the use of a smaller number of trees optimal. Batch training with the Laplace mechanism employs equal budget allocation; as the number of trees increases,

ours batch-LM disj.-LM ours subs.&aggr. 1.0 1.0   
X non-private non-private 0.5 0.5 m 0 50 100 0 500 1000   
(a) Numberof trees (b) Number of inference queries

![](images/a3009d91d6bb2ac554e103487cfa9568003ed94223b6cfefb030c9dc663d4f5c.jpg)  
Figure 2: Test accuracy of different private prediction techniques on various datasets, varying values of privacy loss $\epsilon$ with fixed parameters: $h = 4 , \tau = 1 2 8$ for Car, $\mathbf { \bar { \boldsymbol { h } } } = 2 , \mathbf { \bar { \boldsymbol { \tau } } } = 6 4$ for Iris, $\bar { h ^ { \prime } } = 2 , \tau = 1 2 8$ for Balance, $\bar { h ^ { + } } = { \top } , \tau = 1 2 8 , q = \bar { 2 } , \bar { d } = 4$ for Heart, $h = 3 , \tau = 1 2 5$ , $q = 5 , \bar { d } = 4$ for Mushroom and $h = 8 , \tau = 1 0 0 , q = 4 , \bar { d } = 1 0$ for Adult.   
Figure 3: a) Test accuracy of DP training methods on the Car dataset vs. number of trees $( \epsilon = 2 , h = 4 , q = 1 )$ . b) Accuracy of DP prediction methods on the Car dataset vs. number of inference queries $( \epsilon = 2 , h = 4 , \tau = 1 6 , q = 1 )$ .   
Figure 4: Test accuracy of different subsample-andaggregate approaches on the Car dataset with various privacy loss budget of $\epsilon$ when $\tau = 1 6 , h = 3 , q = 1$ .

0.75 D Subs.& aggr.(ours) 0.50 Subs.& aggr.-LM 0.25 x--X--X- X 0.5 1.0 1.5 2.0 Privacy lossε

the privacy budget per tree decreases. The same tree scaling issue occurs during disjoint training; as the number of trees increases, the number of training samples per tree decreases. In both cases, the predictive power of a single tree is inversely proportional to the total number of trees. On the other hand, our DP batch training method preserves good accuracy as the number of trees increases, reflecting the scaling principle of ensemble learning; see Figure 3a.

Increasing the number of test samples has minimal impact on the accuracy of DP batch prediction. Figure3b illustrates the relationship between accuracy and the number of inference queries for the Car dataset, comparing our DP batch prediction against the baseline subsample-andaggregate. Because of the limited number of samples, we use the entire Car dataset to train a model and report the prediction accuracy for 5 to 1000 inference queries. The existing approach exhibits low accuracy as the number of inference queries increases, reaching $30 \%$ accuracy for 100 samples. The per-sample budget allocation leads to an immediate degradation in accuracy. In contrast, with our DP batch prediction, accuracy remains at $90 \%$ even when predicting as many as 1000 samples. This effectively addresses potential challenges that existing DP prediction approaches face when making predictions on numerous samples.

Table 1: Runtime of different privacy prediction techniques for datasets with feature dimension $d$ , broken down into noise optimization, training, and prediction.   

<html><body><table><tr><td rowspan="2">Dataset</td><td rowspan="2">d</td><td rowspan="2">Method</td><td colspan="3">Time (sec)</td></tr><tr><td>Opti.</td><td>Train.</td><td>Pred.</td></tr><tr><td rowspan="3">Adult</td><td rowspan="3">124</td><td>DP Train.</td><td>255.7</td><td>12.84</td><td>30.09</td></tr><tr><td>DP Pred.</td><td>58.58</td><td>12.88</td><td>29.63</td></tr><tr><td>Non-DP</td><td>N/A</td><td>13.20</td><td>29.24</td></tr><tr><td rowspan="3">Heart</td><td rowspan="3">14</td><td>DP Train.</td><td>0.60</td><td>1.05</td><td>0.52</td></tr><tr><td>DP Pred.</td><td>0.79</td><td>0.82</td><td>0.57</td></tr><tr><td>Non-DP</td><td>N/A</td><td>0.72</td><td>0.36</td></tr><tr><td rowspan="3">Mushroom</td><td rowspan="3">23</td><td>DP Train.</td><td>16.24</td><td>12.22</td><td>10.17</td></tr><tr><td>DP Pred.</td><td>24.68</td><td>11.97</td><td>10.07</td></tr><tr><td>Non-DP</td><td>N/A</td><td>13.09</td><td>9.82</td></tr></table></body></html>

Applying to the subsample-and-aggregate framework. We show the generalizability of our techniques by adapting them to the subsample-and-aggregate framework. Figure 4 shows the test accuracy of our optimized approach compared to the existing non-optimized version, when performing prediction on 345 examples with the Car dataset. Our technique improves the accuracy of existing solutions by up to $50 \%$ , with similar improvement in other datasets.

Runtime. Table 1 compares the runtime of Algorithm 3 to non-private random decision trees with the same hyperparameters as in Figure 2. The runtime for data-independent tree building is excluded as it incurs no DP overhead. The primary overhead comes from optimization, which is crucial for high accuracy. Despite this, our approach maintains minimal runtime overhead during training and prediction. Although our DP baselines have similar training and prediction complexities, they lack noise optimization. Our results show that the DP M-RF algorithm is feasible on commodity hardware, even with large datasets. The optimization overhead can be reduced by reducing the number of features per ensemble, potentially at the expense of some accuracy. Further experiments are detailed in the technical appendix.

# Acknowledgments

This work was supported in part by the NSF grant 2131910.