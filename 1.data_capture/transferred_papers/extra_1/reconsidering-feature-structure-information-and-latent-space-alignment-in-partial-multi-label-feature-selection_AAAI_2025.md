# Reconsidering Feature Structure Information and Latent Space Alignment in Partial Multi-label Feature Selection

Hanlin Pan1,2, Kunpeng ${ { \bf { L i u } } ^ { 3 } }$ , Wanfu Gao1,2\*

1College of Computer Science and Technology, Jilin University, China 2Key Laboratory of Symbolic Computation and Knowledge Engineering of Ministry of Education, Jilin University, China 3Department of Computer Science, Portland State University, Portland, OR 97201 USA panhl23@mails.jlu.edu.cn, kunpeng@pdx.edu, gaowf@jlu.edu.cn

# Abstract

The purpose of partial multi-label feature selection is to select the most representative feature subset, where the data comes from partial multi-label datasets that have label ambiguity issues. For label disambiguation, previous methods mainly focus on utilizing the information inside the labels and the relationship between the labels and features. However, the information existing in the feature space is rarely considered, especially in partial multi-label scenarios where the noises is considered to be concentrated in the label space while the feature information is correct. This paper proposes a method based on latent space alignment, which uses the information mined in feature space to disambiguate in latent space through the structural consistency between labels and features. In addition, previous methods overestimate the consistency of features and labels in the latent space after convergence. We comprehensively consider the similarity of latent space projections to feature space and label space, and propose new feature selection term. This method also significantly improves the positive label identification ability of the selected features. Comprehensive experiments demonstrate the superiority of the proposed method.

# Introduction

Partial Multi-Label Learning (PML) (Xie and Huang 2018; Yu et al. 2018) is a emerging framework within the realm of weakly supervised learning. In this paradigm, each instance is linked to a set of potential candidate labels, in which only part of these labels represent the true label. The characteristic of not requiring the label set to be completely correct makes this type of method more aligned with real-world scenarios and more robust in the face of noises in the dataset. It also diverges from traditional supervised learning methodologies, which precisely annotating label on each sample, thereby incurring substantial labeling cost. These two natures of PML renders it a viable solution for a multitude of practical applications, such as image recognition (Pham et al. 2022; Zeng et al. 2013), web mining (Chen et al. 2021), and ecoinformatics (Yilmaz et al. 2021; Zhou et al. 2018).

Formally speaking, let $\mathcal { D } = \{ ( x _ { i } , y _ { i } ) \mid 1 \leq i \leq n \}$ be the partial multi-label training set. $X ~ = ~ \{ x _ { i } \mid 1 \leq i \leq n \} ~ \in$

# Candidate labels

house   
• tree   
• street lamp   
• people   
• mountain   
• bike   
• flower

![](images/016310a17044af19bd3c64084cb48ea10ad9bce3c2fc8c27d0867fdbd19eab2f.jpg)  
Figure 1: An example of partial multi-label learning. The image is partially labeled by noisy annotators. Among the candidate labels, house, street lamp, people and bike are ground-truth labels while tree, mountain and flower are noisy labels.

$\mathbb { R } ^ { n \times d }$ which represents $n$ samples with $d$ -dimensional feature space and ${ \bf { \dot { Y } } } \in R ^ { n \times l } \in { \bf { \dot { \{ 0 , 1 \} } } } ^ { n \times l }$ be the label space with $l$ labels. In the label matrix, $Y _ { i k } = 1$ means the $k$ -th label is one of the candidate labels of $X _ { i }$ .

The core of addressing the problem of PML is how to deal with ambiguation in label set, which essentially involves identifying the ground-truth label of an instance from its candidate label sets. A common idea is to exploit existing information to disambiguate noises in labels. Several methods utilize relationships between labels (Xie and Huang 2018, 2021), while others focus on analyzing the relationships between labels and features (Zhang and Fang 2020; Xu, Liu, and Geng 2020; Yu et al. 2018). Additionally, some methods combine these two ideas (Xie and Huang 2020; Sun et al. 2021; Li, Lyu, and Feng 2021).

However, existing research predominantly leverages the information contained within the labels or the relationship between labels and features. While features also contain valuable information that can be exploited. In partial label learning scenarios, the information within features is considered to be relatively accurate in contrast to the potentially erroneous label information. By using the accurate information within the features, the ground-truth label of a sample can be identified from its candidate set. Moreover, the structural consistency between labels and features has not been thoroughly considered, i.e., similar features correspond to similar labels. By integrating these two characters, we can utilize the accurate information within the features to remove noises from the label set.

![](images/960e800b82850aceea525d69bcb86e48c2fc9dd9ae97aa9131cd1e6e684a8fea.jpg)  
Figure 2: The Process of PML-FSLA. First, the feature matrix and the label matrix are projected into the $k$ -dimension space determined by OPTICS. Then noisy labels are removed through latent space alignment. Finally two weight matrices are employed for feature selection.

Another problem is the sparsity of multi-label datasets. In multi-label datasets, the number of positive labels is much smaller than the number of negative labels, which naturally interferes with the learning process of the model due to data imbalance. Due to the significant difference in the number of learning examples for positive labels compared to negative labels, the models’ ability to distinguish positive labels is weak. However, in practical application scenarios such as fault detection (Abid, Khan, and Iqbal 2021) and disease diagnosis (Kumar et al. 2023), the importance of distinguishing positive labels is much higher than that of distinguishing negative labels. Therefore, how to improve the model’s ability to distinguish true positive labels when constructing a model is a key issue.

label matrix and feature matrix into a latent space, and the information of features is employed to reduce the noises of labels through alignment. Finally, feature selection is performed using the projection coefficient matrix of label matrix and feature matrix. Our main contributions are:

Feature selection can find the most informative and representative features, remove redundant and irrelevant features, and reduce the curse of dimensionality. During this process, we find that there are some problems in the common used feature selection terms in embedded feature selection: certain feature selection methods extend the linear projection of features to labels, resulting in limitations when handling high-order relations. Others rely on latent space coefficients for feature selection, but these methods often lack robustness when dealing with redundancy, erroneous information, and structural inconsistencies.

• A new feature selection term is utilized, involving the product of the projection coefficient matrix of the feature matrix and the label matrix. This method, compared to traditional coefficient matrices or single projection coefficient matrices, is more flexible and better suited for handling high-order relationships. The structural consistency of the feature matrix and label matrix is also utilized more directly, enhancing the identification capabilities for true positive labels. • The correct information in the feature matrix is leveraged through latent space alignment, while noises in the label matrix are reduced by utilizing the structural consistency between the label matrix and the feature matrix. • The OPTICS (Ankerst et al. 1999) clustering method is employed to determine the dimension of the latent space. This allows the model to determine the number of clusters based on the characteristics of the dataset, making the construction of the latent space more reasonable.

To solve these problems, we propose Partial Multi-label Feature Selection based on Latent Space Alignment (PMLFSLA). The whole process is shown in Figure 2. First OPTICS clustering is utilized to identify dimensions of latent space. Then matrix decomposition is applied to project the

# Related Work

# OPTICS Clustering

OPTICS clustering refers to “Ordering Points To Identify the Clustering Structure”, is a classic clustering algorithm first introduced in 1999 (Ankerst et al. 1999). It is an extension of DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm (Ester et al. 1996), the main idea is to find the cluster of the dataset by identifying the densityconnected points. The algorithm builds a density-based representation of the data by creating an ordered list of points called the reachability plot. Each point in the list is assigned with a reachability distance to measure how far it is to reach that point from other points in the dataset. Points with similar reachability distances are likely to be in the same cluster.

Compared to tradition cluster method like KNN, the number of clusters does not need to be set in advance. It provides more flexibility in selecting the number of clusters, can better identify clusters of arbitrary shapes and effectively handle noises and outliers. Compared to DBSCAN, OPTICS can handle data clusters with varying density and generate richer output and it is less sensitive to noises.

# Embedding Method Feature Selection

In embedded feature selection, a common idea is to take term $\| X W - Y \| _ { F } ^ { 2 }$ as the core term, then add other regularization terms with different meanings to form the objective function, and obtain the final feature selection term $W$ by optimizing the objective function and select the feature according to $\Vert \boldsymbol { W } _ { i . } \Vert _ { 2 }$ . Some methods remove the noises or cluster $X$ or $Y$ by adding the matrix decomposition term (He et al. 2019; Li and Wang 2020; Yu et al. 2020). Hu et. al. propose a method eliminates the noises by decomposing the label matrix into the low-dimensional space (Hao,

Hu, and Gao 2023). The feature matrix is also decomposed to steer the direction for label disambiguation. Finally, the coherent subspace is constructed through the shared coefficient weight matrix. Some methods add manifold terms for structural consistency of the matrix decomposition (Tibshirani 1996; Qi et al. 2018; Wei and Philip 2016). Shang et. al. propose a graph regularized feature selection framework based on subspace learning, which is extended by introducing data graphs. Data graph and feature graph are introduced into subspace learning to maintain the geometric structure of data manifold and feature manifold (Shang et al. 2020). Some methods add regularization term to ensure the sparsity of the matrix (Nie et al. 2010; Akbari and Hesamian 2019). Benefit from the joint advantages of the dual-manifold learning and the hesitant fuzzy correlation, it adds a feature manifold regularization term based on HFCM between features to the objective function to maintain the similarity between features and feature weights (Mokhtia, Eftekhari, and SaberiMovahed 2021). In addition, the regularization term of the sample manifold is also considered to maintain the local correlation between each class of samples.

The main drawback of linear feature selection term is to depict the mapping from $X$ to $Y$ through low-order linear relationship. However, in high-dimensional and large-scale multi-label data, the high-order relationship between features and labels is more complex, and it is difficult to depict it simply with low-order relationship.

Another common method is to exploit the latent space for feature selection. Some methods project the original matrix into the latent space to reduce noises or redundancy. Braytee et. al. solve the high-dimensional problem of data by matrix decomposition of label space and feature space respectively (Braytee et al. 2017). By projecting the original space to the low-dimensional space, it achieves the purpose of distinguishing irrelevant labels, related labels and wrong labels. Others believe that the label matrix and the feature matrix have structural consistency, and are the mapping to different dimensions of the same high-dimensional space (Hu et al. 2020b). These methods fit the projection of the initial matrix in the objective function, and select features through the coefficient matrix corresponding to the projection matrix. By projecting the label matrix and the feature matrix into the same dimension, Gao et. al. explore the impact of potential feature structure on label relationship, and design a latent shared term to share and preserve both latent feature structure and latent label structure (Gao, Li, and Hu 2021).

The main disadvantage of single latent feature selection term is that the projection coefficient matrix used means the weight coefficient of the latent space projected to the feature matrix. The premise of feature selection in this way is that the latent space projection of the feature matrix and the label matrix eventually converge, so the corresponding feature weight and label weight also converge. While the problem is that even if the feature matrix and label matrix have structural consistency, the accuracy cannot be guaranteed due to: (1) the existence of redundant information and error information; (2) the structure cannot be completely consistent.

# The Proposed Method

To find the latent space dimensions that are consistent with the dataset, the OPTICS clustering method is first used on the features. The radius is set as $r$ , after which the latent dimension $k$ is obtained. Then, the feature matrix $X ~ \in$ $\mathbb { R } ^ { n \times d }$ is decomposed into two low-dimensional matrices $\ b { L } \in \mathbb { R } ^ { n \times k }$ and $\mathbf { \Psi } ^ { \mathbf { \prime } } Q ^ { T } \in \mathbb { R } ^ { k \times d }$ . To minimize the reconstruction error, the following form is obtained:

$$
\operatorname* { m i n } _ { \boldsymbol { Q } , L } \| \boldsymbol { X } - L \boldsymbol { Q } ^ { T } \| _ { F } ^ { 2 } .
$$

Where $L$ represents the latent cluster matrix of feature matrix, with $\dot { Q ^ { T } }$ denoting a coefficient matrix. Formula 1 indicates that $d$ -dimensional features reduce to $k$ -dimensional features, which is obtained by OPTICS. It can be explained that the original $d$ -dimensions features are clustered into $k$ different clusters, relevant features are in the same cluster, while features of different clusters are independent. The original feature matrix $X$ can be seen as projected from the latent matrix $L$ , and $Q ^ { T }$ is the weight coefficient matrix. The row of matrix $Q$ represents the coefficient of each feature in these $k$ latent feature variables.

Similarly, label matrix $Y$ can also be decomposed into latent cluster matrix $P$ and coefficient matrix $R$ . As the correct label matrix $T$ should have a strong corresponding relationship with the characteristic matrix, the projection of the feature matrix and the label matrix should be consistent in the latent space. In addition, in the partial multi-label scenario, the label matrix is considered correct and noises appears in the label matrix. Combining these two points, we can utilize the latent space alignment of the feature matrix and the label matrix to reduce the noises in the label by using the correct information in the feature matrix. As shown in Figure 3, by comparing the projection of the label matrix and the feature matrix in the latent space, the label projection matrix can be updated according to the feature projection matrix:

$$
\operatorname* { m i n } _ { Q , L , P , R , T } \| X - L Q ^ { T } \| _ { F } ^ { 2 } + \alpha \| T - P R \| _ { F } ^ { 2 } + \beta \| L - P \| _ { F } ^ { 2 } .
$$

Where $\alpha$ and $\beta$ are parameters employed to balance the contribution of label decomposition and latent space alignment. $T$ is the correct label matrix, and its initial value is $Y$ . The third term is the alignment term to match the projections of the feature matrix and the label matrix so that the correct information of the feature matrix can be exploited. Unlike the traditional feature selection using $W \in \mathring { R } ^ { d \times l }$ where $d , l$ are the dimension of feature and label as the feature selection term, this method adopts the product of $Q$ and $R$ as the feature selection term, so the final function is reformed as:

$$
\begin{array} { r l } {  { \operatorname* { m i n } _ { Q , L , P , R , T } \| X - L Q ^ { T } \| _ { F } ^ { 2 } + \alpha \| T - P R \| _ { F } ^ { 2 } + \beta \| L - P \| _ { F } ^ { 2 } } } \\ & { + \gamma \| Q R \| _ { 2 , 1 } , } \\ & { s . t . Q , L , P , R , T \geq 0 . } \end{array}
$$

Where $\theta$ is a parameter that ensures the sparsity of the objective function and carry out feature selection. In the $Q R$ method, $Q$ represents the coefficients of the clustering $L$ projected onto $X$ , while $R$ represents the coefficients of the clustering $T$ projected onto $Y$ . With the iterations progress, the latent projections of the feature matrix and the label matrix tend to converge due to the alignment term’s influence. $Q R$ thus indicates the similarity of the same class $K$ in terms of their projections onto $X$ and $Y$ , $Q R _ { i j }$ reflects the similarity between a feature $X _ { i }$ and a label $Y _ { j }$ in latent space. The higher the $Q R _ { i j }$ , the more similar $X _ { i }$ and $Y _ { j }$ are in latent space, the more important $X _ { i }$ is to $Y _ { j }$ . Compared to feature in latent space label in latent space ★★ ★ ★\*★ ★

commonly used feature selection term $W$ , the starting point of this method is to deploy the structural consistency between labels and features to project to the common dimension $k$ for comparison and then select features. And it is more flexible for processing high-order relationships as traditional $\lVert X W - \bar { Y } \rVert _ { F } ^ { 2 }$ is relatively linear.

Compared to another commonly used feature selection term $Q$ , the meaning of each column is the coefficient of cluster matrix projected onto $X$ . The premise of feature selection in this method is that the learned $L$ and $P$ eventually converge. Therefore, the feature weight of $L$ for $X$ and the label weight of $P$ for $Y$ also converge. However, the problem is that even though $X$ and $Y$ have structural consistency, the accuracy cannot be guaranteed due to the existence of redundant information and noises and the structure cannot be completely consistent. By incorporating $R$ , which is related to $Y$ , this method can effectively reduce the impact of redundancy and noises, and mitigate potential inconsistencies within the structure.

Moreover, incorporating $R$ in the $Q R$ method inherently enhances the ability to identify true positive labels during feature selection. Since $X , \dot { Y } , V , Q , \mathsf { \bar { Z } } , P , T$ are all nonnegative, in if $Y _ { i j }$ is equal to zero, then the coefficient of the latent matrix must be zero. In other words, if $P _ { i j }$ is not zero, the corresponding y is not equal to zero. Therefore, in $Q R$ , increasing of the weight is only related to positive labels. The features selected according to the $Q R$ weight are more consistent with positive labels in latent space, which strengthens the identification of positive labels.

# Optimization

As the Formula 4 is joint nonconvex, the global optimal solution cannot be obtained. Moreover, the Formula 4 is

Algorithm 1: Pseudo code of PML-FSLA

Input:   
Feature matrix $X$ and label matrix $Y$ ;   
Regularization parameters $\alpha$ , $\beta$ , and $\gamma$ ;   
Cluster radius $r$ .   
Output:Return the ranked features.   
1: Cluster $X$ with the given radius $r$ using the OPTICS and determine the cluster number $k$ ;   
2: while not coverage do   
3: Update $L$ by Formula 16 with other variables fixed; 4: Update $Q$ by Formula 17 with other variables fixed; 5: Update $P$ by Formula 18 with other variables fixed; 6: Update $R$ by Formula 19 with other variables fixed; 7: Update $T$ by Formula 20 with other variables fixed; 8: end while   
9: return Return features according to $\left\| Q R _ { i } \right\| _ { 2 }$ .

nonsmooth due to the existence of $l _ { 2 , 1 }$ -norm feature selection term. Therefore, three iterative rules are introduced to obtain the local optimal solution. First $Q R$ is relaxed as $\mathrm { T r } ( Q R ) ^ { T } D ( Q R )$ where $D \in \mathcal { R } ^ { d \times d }$ is a diagonal matrix and the $i$ -th diagonal element $D _ { i i } = ( 1 / ( 2 \| \bar { Q } R _ { i \cdot } \| _ { 2 } + \epsilon ) )$ . $\epsilon$ is a extremely small positive constant. So the Formula 4 can be rewritten as:

$$
\begin{array} { r l } & { \Theta ( Q , L , P , R , T ) = \mathrm { T r } \left( \left( X ^ { T } - Q L ^ { T } \right) \left( X - L Q ^ { T } \right) \right) } \\ & { \qquad + \alpha \mathrm { T r } \left( \left( T ^ { T } - R ^ { T } P ^ { T } \right) \left( T - P R \right) \right) } \\ & { \qquad + \beta \mathrm { T r } \left( \left( L ^ { T } - P ^ { T } \right) \left( L - P \right) \right) } \\ & { \qquad + 2 \gamma \mathrm { T r } \left( R ^ { T } Q ^ { T } D Q R \right) . } \end{array}
$$

After that nonnegative constraints are integrated into Formula 4 and the lagrange function of it can be obtained:

$$
\begin{array} { r l } & { \quad \mathcal { L } ( Q , L , P , R , T ) } \\ & { = \mathrm { T r } \left( \left( X ^ { T } - Q L ^ { T } \right) \left( X - L Q ^ { T } \right) \right) } \\ & { \quad + \alpha \mathrm { T r } \left( \left( T ^ { T } - R ^ { T } P ^ { T } \right) \left( T - P R \right) \right) } \\ & { \quad + \beta \mathrm { T r } \left( \left( L ^ { T } - P ^ { T } \right) \left( L - P \right) \right) } \\ & { \quad + 2 \gamma \mathrm { T r } \left( R ^ { T } Q ^ { T } D Q R \right) - \mathrm { T r } \left( \Omega Q ^ { T } \right) - \mathrm { T r } \left( \Psi L ^ { T } \right) } \\ & { \quad - \mathrm { T r } \left( \Phi P ^ { T } \right) - \mathrm { T r } \left( \Upsilon R ^ { T } \right) - \mathrm { T r } \left( \tau T ^ { T } \right) . } \end{array}
$$

Where $\Omega \in R ^ { d \times k }$ , $\Psi \in R ^ { n \times k }$ , $\Phi \in R ^ { n \times k }$ , $\Upsilon \in R ^ { k \times l }$ and $\tau \in R ^ { n \times l }$ are Lagrange multipliers. The partial derivatives of the function w.r.t variables $Q , L , P , R$ and $T$ are:

$$
\frac { \partial \mathcal { L } } { \partial Q } = - 2 X ^ { T } L + 2 Q L ^ { T } L + 2 \gamma Q R R ^ { T } - \Omega .
$$

$$
\frac { \partial \mathcal { L } } { \partial L } = - 2 X Q + 2 L Q ^ { T } Q + 2 \beta L - 2 \beta P - \Psi .
$$

$$
\frac { \partial \mathcal { L } } { \partial P } = - 2 \alpha T R ^ { T } + 2 \alpha P R ^ { T } R + 2 \beta P - 2 \beta L - \Phi .
$$

$$
\frac { \partial \mathcal { L } } { \partial R } = - 2 \alpha P ^ { T } T + 2 \alpha P ^ { T } P R + 2 \gamma Q Q ^ { T } R - \Upsilon .
$$

$$
\frac { \partial \mathcal { L } } { \partial T } = 2 \alpha T - 2 \alpha P R - \tau .
$$

Based on Karush–Kuhn–Tucker condition we obtain that:

$$
( - 2 \boldsymbol { X } ^ { T } \boldsymbol { L } + 2 Q \boldsymbol { L } ^ { T } \boldsymbol { L } + 2 \gamma Q R R ^ { T } ) _ { i j } \Omega _ { i j } = 0 .
$$

$$
( - 2 X Q + 2 L Q ^ { T } Q + 2 \beta L - 2 \beta P ) _ { i j } \Psi _ { i j } = 0 .
$$

$$
( - 2 \alpha T R ^ { T } + 2 \alpha P R ^ { T } R + 2 \beta P - 2 \beta L ) _ { i j } \Phi _ { i j } = (
$$

$$
\begin{array} { r } { ( - 2 \alpha P ^ { T } T + 2 \alpha P ^ { T } P R + 2 \gamma Q Q ^ { T } R ) _ { i j } \Upsilon _ { i j } = 0 . } \end{array}
$$

$$
( 2 \alpha T - 2 \alpha P R ) _ { i j } \tau _ { i j } = 0 .
$$

Then $Q , L , P , R$ and $T$ can be presented as:

$$
L _ { i j } ^ { t + 1 } \gets L _ { i j } ^ { t } \frac { \left( X Q + \beta P \right) _ { i j } } { \left( L Q ^ { T } Q + \beta L \right) _ { i j } } .
$$

$$
Q _ { i j } ^ { t + 1 } \gets Q _ { i j } ^ { t } \frac { \left( X ^ { T } L \right) _ { i j } } { \left( Q L ^ { T } L + \gamma Q R R ^ { T } \right) _ { i j } } .
$$

$$
P _ { i j } ^ { t + 1 }  P _ { i j } ^ { t } \frac { ( \alpha T R ^ { T } + 2 \beta L ) _ { i j } } { ( \alpha P R ^ { T } R + \beta P ) _ { i j } } .
$$

$$
R _ { i j } ^ { t + 1 } \gets R _ { i j } ^ { t } \frac { \left( \alpha P ^ { T } T \right) _ { i j } } { \left( \alpha P ^ { T } P R + \gamma Q Q ^ { T } R \right) _ { i j } } .
$$

$$
\begin{array} { r } { T _ { i j } ^ { t + 1 }  T _ { i j } ^ { t } \frac { ( P R ) \rangle _ { i j } } { ( T ) _ { i j } } . } \end{array}
$$

Where $t$ indicates the iterative number. To ensure that the denominator does not reach zero during model iteration, a very small constant is added to the denominator of each term. The whole process is presented in Algorithm 1.

# Experiment Experimental Setup

To prove the effectiveness of the proposed method, we compare our method with the nine state-of-the-art methods of partial multi-label learning and multi-label learning method. Six partial multi-label learning methods (PML-LC (Xie and Huang 2018), PML-FP (Xie and Huang 2018), PAR-VLS (Zhang and Fang 2020), PAR-MAP (Zhang and Fang 2020), FPML (Yu et al. 2018) and PML-FSSO (Hao, Hu, and Gao 2023)) and three multi-label feature selection methods (MLKNN (Zhang and Zhou 2007), MIFS (Jian et al. 2016) and DRMFS (Hu et al. 2020a)) are involved. One classical and two low-rank multi-label feature selection methods are adopted. Due to the lack of feature selection method in partial multi-label learning, the weight matrix is extracted from model to reflect the importance of features. We adopt tenfold cross-validation to train these models and the selected features are compared on SVM classifier.

# Datasets and Evaluation Metrics

We conduct experiments on eight datasets from multiple fields: HumanPseAAC for Biology, CAL500 for music classification, Chess and Corel5K for image annotation, $L L O G \mathbf { \mathcal { F } }$ and for text categorization, Water for chemistry, Yeast for gene function prediction and CHD49 for medicine. Detailed information of datasets is shown in Table 1.

<html><body><table><tr><td>Name</td><td>Domain</td><td>#Instances</td><td>#Features</td><td>#Labels</td></tr><tr><td>CAL</td><td>music</td><td>555</td><td>49</td><td>6</td></tr><tr><td>CHD_49</td><td>medicine</td><td>555</td><td>49</td><td>6</td></tr><tr><td>Chess</td><td>imagine</td><td>585</td><td>258</td><td>15</td></tr><tr><td>Corel5K</td><td>image</td><td>5000</td><td>499</td><td>374</td></tr><tr><td>HumanPseAAC</td><td>biology</td><td>3106</td><td>40</td><td>14</td></tr><tr><td>LLOG_F</td><td>text</td><td>1460</td><td>1004</td><td>75</td></tr><tr><td>Water</td><td>chemistry</td><td>1060</td><td>16</td><td>14</td></tr><tr><td>Yeast</td><td>biology</td><td>2417</td><td>103</td><td>14</td></tr></table></body></html>

![](images/72e59b7cb812e5c6ae97da50021236188f119913ca1ae4d330f5ba497023bf80.jpg)  
Table 1: Characteristics of experimental datasets.   
Figure 4: Ten methods on Corel5K in terms of Marco-F1, Average Precision, Ranking Loss and Coverage Error.

# Results

Four widely-used metrics are selected for our performance evaluation: Ranking loss, Coverage, Average Precision, Macro-F1 and Micro-F1. Ranking Loss and Coverage are optimized when their values are minimized, while higher values of Average Precision, Macro-F1 and Micro-F1 indicate better performance.

Tables 2-6 show the detail of the experiment result, PMLFSLA is adopted as the abbreviation of our method. For all the datasets except Water, one to twenty percent features are selected according to the importance as descending order. For each dataset, five used metrics are recorded in form of mean and standard deviation among different percentages. As Water only have 16 features, we select 1 to 16 features. To show our performance more clearly, we also demonstrated a dataset for four metrics in Figure 4. From the overall results, following observations is obtained:

• In terms of Marco-F1 and Mirco-F1, PML-FSLA ranks first on all eight datasets. As for Average Precision, Ranking Loss and Coverage, PML-FSLA ranks first except CHD49 and Water. Among all the cases, PMLFSLA ranks first in $8 5 \%$ of cases, and in only $7 . 5 \%$ cases PML-FSLA doesn’t rank in top two. This result comprehensively proves the effectiveness of PML-FSLA.

Table 2: Experimental results (mean $\pm$ std) in terms of Mirco-f1 where the best performance is shown in boldface.   

<html><body><table><tr><td>Datasets</td><td>PML-FSLA</td><td>PML-LC</td><td>PML-FP</td><td>PAR-VLS</td><td>PAR-MAP</td><td>FPML</td><td>PML-FSSO</td><td>MLKNN</td><td>MIFS</td><td>DRMFS</td></tr><tr><td>CAL</td><td>0.538±0.019</td><td>0.018±0.031</td><td>0.069±0.049</td><td>0.020±0.001</td><td>0.079±0.078</td><td>0.122±0.070</td><td>0.000±0.000</td><td>0.043±0.043</td><td>0.000±0.000</td><td>0.000±0.000</td></tr><tr><td>CHD_49</td><td>0.637±0.059</td><td>0.310±0.149</td><td>0.244±0.118</td><td>0.202±0.136</td><td>0.262±0.120</td><td>0.012±0.014</td><td>0.484±0.160</td><td>0.310±0.188</td><td>0.157±0.124</td><td>0.090±0.127</td></tr><tr><td>Chess</td><td>0.579±0.061</td><td>0.404±0.082</td><td>0.432±0.128</td><td>0.497±0.127</td><td>0.536±0.145</td><td>0.050±0.000</td><td>0.035±0.003</td><td>0.552±0.147</td><td>0.299±0.081</td><td>0.154±0.027</td></tr><tr><td>Corel5K</td><td>0.263±0.094</td><td>0.075±0.020</td><td>0.067±0.047</td><td>0.054±0.058</td><td>0.043±0.016</td><td>0.000±0.000</td><td>0.004±0.004</td><td>0.000±0.000</td><td>0.000±0.000</td><td>0.035±0.039</td></tr><tr><td>LLOG F</td><td>0.339±0.006</td><td>0.178±0.056</td><td>0.162±0.034</td><td>0.033±0.000</td><td>0.230±0.123</td><td>0.315±0.085</td><td>0.081±0.035</td><td>0.305±0.127</td><td>0.187±0.065</td><td>0.024±0.006</td></tr><tr><td>HumanPseAAC</td><td>0.185±0.002</td><td>0.011±0.018</td><td>0.074±0.022</td><td>0.145±0.091</td><td>0.188±0.114</td><td>0.000±0.000</td><td>0.008±0.001</td><td>0.056±0.025</td><td>0.053±0.042</td><td>0.000±0.000</td></tr><tr><td>Water</td><td>0.556±0.000</td><td>0.331±0.149</td><td>0.346±0.176</td><td>0.094±0.051</td><td>0.151±0.106</td><td>0.106±0.073</td><td>0.186±0.139</td><td>0.149±0.137</td><td>0.455±0.018</td><td>0.056±0.088</td></tr><tr><td>Yeast</td><td>0.496±0.022</td><td>0.000±0.000</td><td>0.000±0.000</td><td>0.000±0.000</td><td>0.000±0.000</td><td>0.000±0.000</td><td>0.026±0.025</td><td>0.020±0.026</td><td>0.228±0.159</td><td>0.021±0.005</td></tr></table></body></html>

Table 3: Experimental results (mean $\pm$ std) in terms of Marco-f1 where the best performance is shown in boldface.   

<html><body><table><tr><td>Datasets</td><td>PML-FSLA</td><td>PML-LC</td><td>PML-FP</td><td>PAR-VLS</td><td>PAR-MAP</td><td>FPML</td><td>PML-FSSO</td><td>MLKNN</td><td>MIFS</td><td>DRMFS</td></tr><tr><td>CAL</td><td>0.538±0.019</td><td>0.017±0.029</td><td>0.074±0.055</td><td>0.017±0.000</td><td>0.075±0.075</td><td>0.104±0.062</td><td>0.033±0.026</td><td>0.041±0.042</td><td>0.000±0.000</td><td>0.000±0.000</td></tr><tr><td>CHD_49</td><td>0.615±0.066</td><td>0.272±0.131</td><td>0.226±0.112</td><td>0.164±0.116</td><td>0.237±0.118</td><td>0.011±0.013</td><td>0.187±0.133</td><td>0.277±0.179</td><td>0.134±0.104</td><td>0.076±0.107</td></tr><tr><td>Chess</td><td>0.588±0.042</td><td>0.050±0.000</td><td>0.343±0.088</td><td>0.405±0.141</td><td>0.476±0.139</td><td>0.131±0.016</td><td>0.020±0.000</td><td>0.517±0.156</td><td>0.539±0.147</td><td>0.511±0.154</td></tr><tr><td>Corel5K</td><td>0.231±0.094</td><td>0.064±0.016</td><td>0.045±0.034</td><td>0.041±0.042</td><td>0.030±0.011</td><td>0.000±0.000</td><td>0.003±0.002</td><td>0.000±0.000</td><td>0.000±0.000</td><td>0.022±0.024</td></tr><tr><td>LLOG F</td><td>0.321±0.005</td><td>0.027±0.113</td><td>0.047±0.050</td><td>0.033±0.000</td><td>0.227±0.114</td><td>0.221±0.058</td><td>0.104±0.037</td><td>0.273±0.100</td><td>0.148±0.060</td><td>0.019±0.005</td></tr><tr><td>HumanPseAAC</td><td>0.185±0.002</td><td>0.009±0.015</td><td>0.067±0.019</td><td>0.125±0.085</td><td>0.189±0.115</td><td>0.000±0.000</td><td>0.004±0.000</td><td>0.042±0.019</td><td>0.035±0.028</td><td>0.000±0.000</td></tr><tr><td>Water</td><td>0.551±0.000</td><td>0.324±0.146</td><td>0.337±0.172</td><td>0.104±0.056</td><td>0.167±0.112</td><td>0.111±0.074</td><td>0.134±0.091</td><td>0.149±0.142</td><td>0.444±0.015</td><td>0.065±0.092</td></tr><tr><td>Yeast</td><td>0.488±0.022</td><td>0.000±0.000</td><td>0.000±0.000</td><td>0.000±0.000</td><td>0.000±0.000</td><td>0.000±0.000</td><td>0.029±0.026</td><td>0.019±0.025</td><td>0.203±0.150</td><td>0.032±0.008</td></tr></table></body></html>

Table 4: Experimental results (mean $\pm$ std) in terms of Average Precision where the best performance is shown in boldface.   

<html><body><table><tr><td>Datasets</td><td>PML-FSLA</td><td>PML-LC</td><td>PML-FP</td><td>PAR-VLS</td><td>PAR-MAP</td><td>FPML</td><td>PML-FSSO</td><td>MLKNN</td><td>MIFS</td><td>DRMFS</td></tr><tr><td>CAL</td><td>0.583±0.004</td><td>0.388±0.008</td><td>0.407±0.013</td><td>0.476±0.045</td><td>0.545±0.040</td><td>0.554±0.049</td><td>0.554±0.053</td><td>0.541±0.041</td><td>0.468±0.075</td><td>0.429±0.038</td></tr><tr><td>CHD_49</td><td>0.712±0.010</td><td>0.683±0.007</td><td>0.663±0.008</td><td>0.763±0.023</td><td>0.713±0.007</td><td>0.646±0.020</td><td>0.773±0.035</td><td>0.739±0.018</td><td>0.748±0.024</td><td>0.765±0.031</td></tr><tr><td>Chess</td><td>0.761±0.037</td><td>0.543±0.085</td><td>0.546±0.093</td><td>0.616±0.117</td><td>0.663±0.133</td><td>0.253±0.033</td><td>0.714±0.133</td><td>0.666±0.124</td><td>0.476±0.129</td><td>0.426±0.122</td></tr><tr><td>Corel5K</td><td>0.413±0.061</td><td>0.302±0.034</td><td>0.315±0.039</td><td>0.285±0.040</td><td>0.239±0.020</td><td>0.266±0.015</td><td>0.374±0.082</td><td>0.266±0.025</td><td>0.282±0.051</td><td>0.333±0.066</td></tr><tr><td>LLOG F</td><td>0.591±0.000</td><td>0.456±0.004</td><td>0.478±0.005</td><td>0.484±0.005</td><td>0.565±0.028</td><td>0.600±0.036</td><td>0.462±0.012</td><td>0.539±0.027</td><td>0.542±0.026</td><td>0.473±0.004</td></tr><tr><td>HumanPseAAC</td><td>0.425±0.000</td><td>0.333±0.064</td><td>0.358±0.048</td><td>0.447±0.117</td><td>0.407±0.088</td><td>0.309±0.046</td><td>0.347±0.063</td><td>0.336±0.051</td><td>0.280±0.070</td><td>0.337±0.072</td></tr><tr><td>Water</td><td>0.566±0.003</td><td>0.518±0.009</td><td>0.530±0.008</td><td>0.602±0.013</td><td>0.591±0.023</td><td>0.607±0.019</td><td>0.576±0.044</td><td>0.588±0.041</td><td>0.525±0.027</td><td>0.552±0.046</td></tr><tr><td>Yeast</td><td>0.690±0.000</td><td>0.506±0.031</td><td>0.521±0.029</td><td>0.613±0.043</td><td>0.570±0.006</td><td>0.633±0.041</td><td>0.574±0.042</td><td>0.597±0.040</td><td>0.678±0.077</td><td>0.578±0.051</td></tr></table></body></html>

Table 5: Experimental results (mean $\pm$ std) in terms of Ranking Loss where the best performance is shown in boldface.   

<html><body><table><tr><td>Datasets</td><td>PML-FSLA</td><td>PML-LC</td><td>PML-FP</td><td>PAR-VLS</td><td>PAR-MAP</td><td>FPML</td><td>PML-FSSO</td><td>MLKNN</td><td>MIFS</td><td>DRMFS</td></tr><tr><td>CAL</td><td>0.318±0.002</td><td>0.629±0.120</td><td>0.613±0.152</td><td>0.687±0.134</td><td>0.716±0.067</td><td>0.684±0.088</td><td>0.394±0.098</td><td>0.605±0.093</td><td>0.579±0.255</td><td>0.574±0.174</td></tr><tr><td>CHD_49</td><td>0.327±0.046</td><td>0.463±0.155</td><td>0.470±0.133</td><td>0.339±0.143</td><td>0.609±0.057</td><td>0.720±0.082</td><td>0.314±0.186</td><td>0.337±0.132</td><td>0.347±0.116</td><td>0.326±0.172</td></tr><tr><td>Chess</td><td>0.150±0.053</td><td>0.258±0.154</td><td>0.262±0.164</td><td>0.453±0.136</td><td>0.398±0.155</td><td>0.873±0.039</td><td>0.223±0.180</td><td>0.395±0.143</td><td>0.543±0.183</td><td>0.554±0.199</td></tr><tr><td>Corel5K</td><td>0.440±0.159</td><td>0.652±0.124</td><td>0.638±0.122</td><td>0.843±0.082</td><td>0.896±0.063</td><td>0.924±0.022</td><td>0.626±0.192</td><td>0.899±0.047</td><td>0.757±0.145</td><td>0.708±0.160</td></tr><tr><td>LLOG F</td><td>0.269±0.000</td><td>0.763±0.024</td><td>0.692±0.016</td><td>0.806±0.051</td><td>0.669±0.106</td><td>0.533±0.176</td><td>0.446±0.108</td><td>0.530±0.195</td><td>0.501±0.206</td><td>0.630±0.197</td></tr><tr><td>HumanPseAAC</td><td>0.249±0.000</td><td>0.451±0.175</td><td>0.489±0.152</td><td>0.463±0.277</td><td>0.608±0.167</td><td>0.848±0.070</td><td>0.493±0.249</td><td>0.802±0.082</td><td>0.733±0.239</td><td>0.586±0.241</td></tr><tr><td>Water</td><td>0.387±0.001</td><td>0.458±0.055</td><td>0.436±0.056</td><td>0.483±0.028</td><td>0.440±0.029</td><td>0.443±0.033</td><td>0.388±0.063</td><td>0.384±0.073</td><td>0.441±0.033</td><td>0.418±0.071</td></tr><tr><td>Yeast</td><td>0.222±0.000</td><td>0.432±0.005</td><td>0.453±0.018</td><td>0.339±0.137</td><td>0.623±0.013</td><td>0.339±0.123</td><td>0.354±0.136</td><td>0.339±0.138</td><td>0.245±0.101</td><td>0.373±0.158</td></tr></table></body></html>

<html><body><table><tr><td>Datasets</td><td>PML-FSLA</td><td>PML-LC</td><td>PML-FP</td><td>PAR-VLS</td><td>PAR-MAP</td><td>FPML</td><td>PML-FSSO</td><td>MLKNN</td><td>MIFS</td><td>DRMFS</td></tr><tr><td>CAL</td><td>0.653±0.000</td><td>0.807±0.006</td><td>0.792±0.006</td><td>0.739±0.034</td><td>0.714±0.024</td><td>0.709±0.028</td><td>0.659±0.039</td><td>0.703±0.027</td><td>0.737±0.062</td><td>0.724±0.042</td></tr><tr><td>CHD_49</td><td>0.523±0.014</td><td>0.620±0.038</td><td>0.622±0.032</td><td>0.657±0.016</td><td>0.657±0.016</td><td>0.666±0.014</td><td>0.485±0.072</td><td>0.532±0.043</td><td>0.528±0.041</td><td>0.492±0.069</td></tr><tr><td>Chess</td><td>0.137±0.017</td><td>0.200±0.062</td><td>0.206±0.062</td><td>0.258±0.060</td><td>0.233±0.068</td><td>0.448±0.017</td><td>0.154±0.082</td><td>0.230±0.064</td><td>0.317±0.071</td><td>0.322±0.085</td></tr><tr><td>Corel5K</td><td>0.364±0.069</td><td>0.507±0.030</td><td>0.483±0.032</td><td>0.551±0.025</td><td>0.585±0.006</td><td>0.574±0.005</td><td>0.426±0.084</td><td>0.562±0.016</td><td>0.510±0.069</td><td>0.472±0.067</td></tr><tr><td>LLOG F</td><td>0.536±0.000</td><td>0.575±0.077</td><td>0.583±0.011</td><td>0.591±0.002</td><td>0.563±0.011</td><td>0.550±0.017</td><td>0.592±0.018</td><td>0.563±0.015</td><td>0.568±0.019</td><td>0.593±0.002</td></tr><tr><td>HumanPseAAC</td><td>0.253±0.000</td><td>0.355±0.063</td><td>0.394±0.030</td><td>0.286±0.101</td><td>0.371±0.052</td><td>0.468±0.011</td><td>0.339±0.073</td><td>0.465±0.011</td><td>0.415±0.082</td><td>0.362±0.073</td></tr><tr><td>Water</td><td>0.727±0.000</td><td>0.764±0.010</td><td>0.747±0.014</td><td>0.731±0.002</td><td>0.726±0.007</td><td>0.738±0.004</td><td>0.734±0.038</td><td>0.733±0.016</td><td>0.735±0.014</td><td>0.746±0.032</td></tr><tr><td>Yeast</td><td>0.520±0.000</td><td>0.562±0.080</td><td>0.576±0.015</td><td>0.555±0.047</td><td>0.782±0.000</td><td>0.532±0.043</td><td>0.579±0.046</td><td>0.549±0.049</td><td>0.522±0.068</td><td>0.578±0.058</td></tr></table></body></html>

Table 6: Experimental results (mean $\pm$ std) in terms of Coverage where the best performance is shown in boldface.

• The superiority in Marco-F1 and Mirco-F1 proves the ability of PML-FSLA in dealing with sparsity and identifying positive labels. We attribute it to the reconstructed feature selection term. By matrix multiplication, the feature selection term $Q R$ represents the feature weight by the latent space similarity between the feature and the label, and eliminates the influence of unbalanced negative labels, so that the selected feature has strong identification ability for positive labels. The subsequent ablation experiments further confirmed this opinion.

• For the superiority in Average Precision, Ranking Loss and Coverage. We attribute it to the effective latent space alignment. Through this process, the information in the feature space and the the structural consistency between features and labels is utilized, and the noises in the label space is effectively eliminated, so the effect of the model is improved. As for the relatively poor effect on some datasets, it is because of the trade off of enhancing the positive label identification ability through feature selection. It leads to the weakening of the identification effect of negative labels, which is reflected in the decline of these metrics that show the overall effect.

![](images/f071829960d7b015b75350334ae1a5e2a60ed79cbf3117a84db491cfca4eef86.jpg)  
Figure 5: Parameter sensitivity studies on the $L L O G \mathbf { \mathcal { F } }$ in terms of Coverage.

# Parameter Analysis

In PML-FSMIR, three parameters $\alpha , \beta$ , and $\gamma$ could influence the experimental outcomes. Figure 5 illustrates the impact of these parameters on the model’s performance on $\pmb { C H D _ { - } 4 9 }$ in terms of Ranking Loss. Each parameter was independently tuned over a range from 0.001 to 1000, and the model’s performance was evaluated when selecting between $8 \%$ to $20 \%$ of the features. Ranking Loss decreases slightly as the number of selected features increases. It is evident that the model demonstrates a clear insensitivity to variations in these parameters, which highlights the robustness and stability of the model under different conditions.

# Ablation Study

In order to prove that the feature selection term we designed is effective in enhancing the ability of the selected features to identify positive labels, another feature selection term commonly used when involving latent space is employed as a comparison: the weight sparse matrix $Q$ of the latent space projected to the feature space. Two methods are compared on seven datasets with Marco-F1 and Mirco-F1. The results are shown in Tables 7 and 8, PML-FSLA- $\mathsf { q }$ is utilized to represent this comparison method utilizing the weight matrix $Q$ , and the larger value in each case is marked in bold.

Table 7: Ablation experimental results of PML-FSLA in terms of Mirco-F1.   

<html><body><table><tr><td>Datasets</td><td>CAL</td><td>CHD_49</td><td>Chess</td><td>Corel5K</td><td>LLOG F</td><td>Water</td><td>Yeast</td></tr><tr><td>PML-FSLA</td><td>0.538</td><td>0.637</td><td>0.579</td><td>0.263</td><td>0.339</td><td>0.556</td><td>0.496</td></tr><tr><td>PML-FSLA-q</td><td>0.149</td><td>0.477</td><td>0.000</td><td>0.000</td><td>0.022</td><td>0.222</td><td>0.121</td></tr></table></body></html>

Table 8: Ablation experimental results of PML-FSLA in terms of Marco-F1.   

<html><body><table><tr><td>Datasets</td><td>CAL</td><td>CHD_49</td><td>Chess</td><td>Corel5K</td><td>LLOG F</td><td>Water</td><td>Yeast</td></tr><tr><td>PML-FSLA</td><td>0.538</td><td>0.615</td><td>0.588</td><td>0.231</td><td>0.321</td><td>0.551</td><td>0.488</td></tr><tr><td>PML-FSLA-q</td><td>0.148</td><td>0.472</td><td>0.000</td><td>0.000</td><td>0.020</td><td>0.212</td><td>0.132</td></tr></table></body></html>

In all cases, PML-FSLA beats PML-FSLA-q: the new feature selection term improves by at least $10 \%$ over baseline. This result shows that only using the feature space projection weight matrix as feature selection term is insufficient. This method only considers the global structural consistency of the feature space and the label space, but cannot deal with the possible noises and local inconsistency problems. while both problems can be solved by comprehensively considering the projection weight matrices of the feature space and label space. Different from the above method that only relies on global structural consistency, this method measures the similarity of the feature space and label space through two weight matrices, so that when local inconsistency occurs, the corresponding weights in the feature selection terms will be adjusted to alleviate this problem. In addition, this method strongly strengthens the role of positive labels in the feature selection process because negative labels with a corresponding weight of zero will not be considered in this projection method. Although this operation will relatively reduce the effect of PML-FSLA in other evaluation metrics, the results demonstrate the trade off is worthwhile.

# Conclusion

In this paper, the label disambiguation problem under the partial labeling scenario is addressed within the latent space by leveraging the inherent information in feature space as well as maintaining the structural consistency between the feature space and the label space. Additionally, rather than relying on traditional feature selection term commonly used in embedded feature selection methods when dealing with latent spaces, we start from another perspective and solve it by considering the similarity of latent space projections. By doing so, we significantly enhance the model’s capability to accurately identify positive labels, leading to more reliable and robust classification results. Extensive experimental results demonstrate that our method outperforms existing approaches in various challenging scenarios, underscoring its effectiveness and robustness.

# Acknowledgments

This work is funded by: by Science Foundation of Jilin Province of China under Grant No. 20230508179RC, and China Postdoctoral Science Foundation funded project under Grant No. 2023M731281 and Changchun Science and Technology Bureau Project 23YQ05.