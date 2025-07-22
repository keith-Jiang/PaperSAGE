# Scalable One-Pass Incomplete Multi-View Clustering by Aligning Anchors

Yalan Qin, Guorui Feng \*, Xinpeng Zhang \*,

School of Communication and Information Engineering, Shanghai University, Shanghai, China ylqin, grfeng, xzhang @shu.edu.cn

# Abstract

Multi-view clustering has gained increasing attention by utilizing the complementary and consensus information across views. To alleviate the computation cost for the existing multi-view clustering approaches on datasets with large scales, studies based on anchor have been presented. Although extensively adopted in the real scenarios, most of these works ignore to learn an integral subspace revealing the cluster structure with anchors from different views being aligned, where the centroid and cluster assignment matrix can be directly achieved based on the integral subspace. Moreover, these works neglect to perform the alignment among anchors and integral subspace learning in a unified model on the incomplete multi-view dataset. Then the mutual improvements among aligning anchors and learning integral subspace are not guaranteed in optimizing the objective function, which inevitably limit the representation ability of the model and result in the suboptimal clustering performance. In this paper, we propose a novel anchor learning method for incomplete multi-view dataset termed Scalable One-pass incomplete Multi-view clustEring by Aligning anchorS (SOMEAS). Specifically, we capture the complementary information among multiple views by building the anchor graph for each view on the incomplete dataset. The integral subspace reflecting the cluster structure is learned with the alignment among anchors from different views being considered. We build the cluster assignment and centroid representation with orthogonal constraint to approximate the integral subspace. Then the subspace itself and the partition are simultaneously taken into account in this manner. Besides, the mutual improvements among aligning anchors and learning integral subspace are able to be ensured. Experiments on several incomplete multi-view datasets validate the efficiency and effectiveness of SOME-AS.

# Introduction

Multi-view data are usually extracted from different sensors in real applications (Li et al. 2022a; Wu et al. 2023a; Qin et al. 2022a; Sun et al. 2023; Qin et al. 2022c,b, 2023b; Li et al. 2022b; Qin et al. 2023c; Yuan et al. 2024; Qin, Pu, and Wu 2023a,b; Sun et al. 2024; Qin et al. 2024d; Li et al. 2023b; Qin and Qian 2024; Qin et al. 2024a, 2025; Li et al. 2023a; Qin et al. 2024b,c), which is different for the data with single view (Qin, Wu, and Feng 2021; Qin et al. 2023d; Pu et al. 2024; Qin et al. 2023a, 2022d). For instance, a clip of the video can be decomposed into the text descriptions, sound records and pictures, which are achieved by producer, microphone and camera, respectively. Multi-view information of the data can be widely employed to improve the expressive ability of the model. Due to the promising capability in revealing the intrinsic structure of data, multi-view clustering has drawn increasing attentions by leveraging the consensus and complementary information across views in learning a desired representation. Consequently, how to utilize the information from multiple views is the key to increase the final performance for clustering.

The existing algorithms for multi-view clustering can be roughly classified into several classes including subspace clustering (Zhang et al. 2020), methods based on kernel (Liu 2021), non-negative matrix factorization (NMF) (Cai, Nie, and Huang 2013) and methods built on graph (Liu et al. 2022). To be specific, subspace clustering methods aim to learn consistent subspace representation across views. Methods based on kernel adopt the specified kernel learning framework, which apply the kernel matrix from the predefined kernel matrices. NMF is able to utilize the multi-view data in matrix factorization. Methods built on graph employ a unified graph structure to explore the multi-view data. Although these methods can achieve promising performance, their capabilities in scaling up are heavily restricted by the high complexity. Then, it is necessary to give efficient algorithms for dealing with multi-view clustering issues of large-scale data. To alleviate the computation cost for the existing multi-view clustering, studies based on anchor have been recently presented (Li et al. $2 0 2 2 \mathrm { c }$ ; Kang et al. 2020; Sun et al. 2021; Yu et al. 2023). Instead of building the full graph, these methods aim to construct the anchor graph between the entire dataset and anchors. Then the time complexity reduces from $O ( n ^ { 3 } )$ to $O ( n ^ { 2 } )$ in this manner. For instance, Li et al. (Li et al. 2022c) attempted to achieve a joint graph based on an adaptive weighted and parameterfree fusion framework. Kang et al. (Kang et al. 2020) independently obtained the anchor graph of each view by employing the pre-defined clustering centers as anchors. Sun et al. (Sun et al. 2021) incorporated the anchor into the optimization process, resulting in a unified anchor graph among different views. Despite numerous methods proposed for improving multi-view clustering in diverse manners, most of them are under the assumption that all data are available (Gao et al. 2015). However, due to sensor malfunction or data corruption in real scenarios, data points are usually partially available (Wu et al. 2023b; Bi, He, and Luo 2023). The absence of data points from different views increases the difficulty of studying complementary and consensus information, leading to the incomplete multi-view clustering problem. Liu et al. (Liu et al. 2022) utilized the anchor graph to reduce the complexity of algorithm, which is able to capture the structure for clustering with incomplete views. Li et al. (Li, Jiang, and Zhou 2014) adopted the $L _ { 1 }$ regularization and non-negative matrix decomposition terms to learn a shared potential representation based on the incomplete data points. Wen et al. (Wen et al. 2023) captured the complementary information across views by building the view-specific anchor graph, which refines the correspondence of anchors from different views.

![](images/adb11a72853df7d534f70986b6ebe43f037b0bd0acdcc5943a3cccd3fa227513.jpg)  
Figure 1: Framework of the proposed SOME-AS.

Although extensively adopted in the real scenarios, most of the existing methods based on anchor for large-scale datasets ignore to learn an integral subspace revealing the cluster structure with anchors from different views being aligned, where the centroid and cluster assignment matrix can be directly achieved based on the integral subspace. As is known, the distribution of data points across views would be biased since the incompleteness of multi-view data (Wen et al. 2023) and the corresponding misalignment issue for the learned anchors arises. Moreover, they neglect to perform the alignment among anchors and learning the integral subspace in a unified model on the incomplete multi-view dataset. Then the mutual improvements among aligning anchors and learning integral subspace are not ensured in optimizing the objective function, which inevitably influence the representation ability of the model and result in the suboptimal clustering performance.

In order to deal with the above challenging issues, we propose a novel anchor learning method for incomplete multiview dataset termed Scalable One-pass incomplete Multiview clustEring by Aligning anchorS (SOME-AS), shown in Fig. 1. To be specific, we capture the complementary information across views by building the anchor graph for each view on the incomplete dataset. The integral subspace reflecting the cluster structure is learned based on the alignment among anchors from different views. The alignment is realized by designing a mapping to adequately obtain the anchor correspondence among views in the learning process. We build the cluster assignment and centroid representation with orthogonal constraint to approximate the integral subspace. Then the subspace itself and the partition are simultaneously taken into account in this manner. Meanwhile, the alignment among anchors and integral subspace learning are integrated into a unified model to improve the quality for clustering. We then design an alternative algorithm with five six steps to handle the formulated optimization problem. The major contributions in our work are summarized as:

1. We learn the integral subspace reflecting the cluster structure with the alignment among anchors across views being considered, which guarantees that the integral subspace learning and the partition are simultaneously performed. The mapping for adequately obtaining the anchor correspondence among views is designed to solve the alignment issue among anchors in this work.   
2. We formulate the alignment among anchors and learning integral subspace into a unified model, which ensures the mutual improvements among these two parts in optimizing the objective function. An alternative algorithm with six steps is designed for solving the formulated optimization problem.   
3. Extensive experiments are performed on several incomplete benchmark multi-view datasets to demonstrate the superiority of SOME-AS compared with the representive methods based on effectiveness and efficiency under different metrics.

# The Proposed Method

Problem Formulation To learn an integral subspace revealing the cluster structure with anchors across views being aligned for incomplete multi-view clustering on large-scale dataset, we develop a scalable one-pass incomplete multiview clustering by aligning anchors. It is able to align the anchors among different views in learning an integral subspace, which reveals the shared cluster partition and depicts the structure of data. Meanwhile, the alignment among anchors and the integral subspace learning are integrated into a unified model, resulting in more promising representation capability. Given multi-view dataset $\{ X ^ { p } \} _ { p = 1 } ^ { v }$ , we introduce the alignment mapping $P _ { p }$ , centroid representation $F \in R ^ { l \times k }$ and cluster assignment ${ \bar { Y } } \in R ^ { k \times n }$ , which satisfy ${ \underset { - } { P _ { p } ^ { T } } } { P _ { p } } = I$ and $F ^ { T } F = I$ . Then the corresponding objective function is written as

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { P _ { p } , F , Y } \| P _ { p } Z _ { p } - F Y \| _ { F } ^ { 2 } , ~ s . t . ~ P _ { p } ^ { T } P _ { p } = I , ~ F ^ { T } F = I , } \\ { \displaystyle Y _ { i j } \in \{ 0 , 1 \} , ~ \displaystyle \sum _ { i = 1 } ^ { k } Y _ { i j } = 1 , ~ \forall j = 1 , 2 , \cdots , n , } \end{array}
$$

where $Z _ { p } \in R ^ { l \times n }$ denotes the anchor graph of each view. $Y _ { i j } ~ = ~ 1$ if the $j$ -th data point belongs to the $i$ -th cluster and 0 otherwise. Accordingly, the integral subspace can be approximated based on the centroid and cluster assignment matrix, which directly ensures the reliability of the built model from aligned anchor graphs and the underlying data structure.

To improve the clustering performance, we choose to learn anchors instead of using the traditional anchor strategy, which avoids relying on the quality of anchor initialization. The procedure is formulated as:

$$
\begin{array} { r l } & { \underset { A _ { p } , Z _ { p } } { \operatorname* { m i n } } \Vert X ^ { p } H _ { p } - A _ { p } Z _ { p } H _ { p } \Vert _ { F } ^ { 2 } , } \\ & { ~ s . t . ~ A _ { p } ^ { T } A _ { p } = I , ~ Z _ { p } ^ { T } 1 _ { l } = 1 _ { n } , ~ Z _ { p } \geq 0 , } \end{array}
$$

where the orthogonal constraints $A _ { p } ^ { T } A _ { ! } = \underline { { I } }$ is imposed to make the learned $A _ { p }$ more discriminative. The term $Z _ { p } H _ { p }$ can describe the similarity between anchors and the available data points of the $p$ -th view. We then unify anchor learning, alignment among anchors and integral subspace learning into a model to enhance the clustering performance as follows:

$$
\begin{array} { r l } & { \displaystyle \operatorname* { m i n } _ { { P _ { p } , F , \bar { X } } , \bar { X } , \bar { X } , p , \bar { Z } _ { p } = 1 } \gamma _ { p } ^ { \nu } \| \bar { X } ^ { p } H _ { p } - A _ { p } Z _ { p } H _ { p } \| _ { F } ^ { 2 } } \\ & { \displaystyle + \sum _ { p = 1 } ^ { n } \alpha \| Z _ { p } \| _ { F } ^ { 2 } + \sum _ { p = 1 } ^ { \nu } \beta \| P _ { p } \bar { Z } _ { p } - { F } Y \| _ { F } ^ { 2 } , } \\ & { \displaystyle _ { s , l , \bar { A } , p } A _ { p } ^ { T } A _ { p } = I , \ Z _ { p } ^ { T } 1 _ { i } = 1 _ { n } , } \\ & { Z _ { p } \geq 0 , P _ { p } ^ { T } P _ { p } = I , F ^ { T } F = I , \gamma ^ { T } 1 = 1 , } \\ & { \displaystyle \quad Y _ { i j } \in \{ 0 , 1 \} , \sum _ { i = 1 } ^ { k } Y _ { i j } = 1 , \forall j = 1 , 2 , \cdots , n , } \end{array}
$$

where $\gamma _ { p }$ determines the weight contribution of each view, $\alpha > 0$ and $\beta > 0$ indicate the balance parameter. Note that $\gamma _ { p }$ would be large if $\| X ^ { p } H _ { p } - A _ { p } Z _ { p } H _ { p } \| _ { F } ^ { 2 }$ is small. The above formulation is a multi-view least-square residual model to some extent, which is able to mitigate the influence of outliers with the two power residual for inducing the robustness.

Optimization The optimization problem in Eq. (3) is nonconvex with all variables being considered. We develop an iterative algorithm to solve this formulated problem, which consists of the following optimization subproblems.

1) Optimization of $\{ Z _ { p } \} _ { p = 1 } ^ { v } \colon \mathrm { W i t h } P _ { p } , F , Y$ and $A _ { p }$ being fixed, we write the optimization of $\{ Z _ { p } \} _ { p = 1 } ^ { v }$ as follows:

$$
\begin{array} { r l } { \displaystyle } & { \displaystyle \underset { Z _ { p } } { \operatorname* { m i n } } \sum _ { p = 1 } ^ { v } \gamma _ { p } ^ { 2 } \| X ^ { p } H _ { p } - A _ { p } Z _ { p } H _ { p } \| _ { F } ^ { 2 } + \sum _ { p = 1 } ^ { v } \alpha \| Z _ { p } \| _ { F } ^ { 2 } , } \\ & { + \displaystyle \sum _ { p = 1 } ^ { v } \beta \| P _ { p } Z _ { p } - F Y \| _ { F } ^ { 2 } , s . t . Z _ { p } ^ { T } 1 _ { l } = 1 _ { n } , Z _ { p } \geq 0 . } \end{array}
$$

We then remove the irrelevant items and rewrite Eq. (4) as:

$$
\begin{array} { r l } & { \underset { Z _ { p } } { \operatorname* { m i n } } T r \big ( Z _ { p } ^ { T } Z _ { p } ( \gamma _ { p } ^ { 2 } H _ { p } H _ { p } ^ { T } + ( \beta + \alpha ) I ) \big ) } \\ & { - 2 T r \big ( Z _ { p } ^ { T } ( \gamma _ { p } ^ { T } A _ { p } ^ { T } X ^ { p } H _ { p } H _ { p } ^ { T } + \beta P _ { p } ^ { T } F Y ) \big ) , } \\ & { s . t . Z _ { p } ^ { T } 1 _ { l } = 1 _ { n } , Z _ { p } \ge 0 . } \end{array}
$$

After denoting $z _ { p } ^ { j }$ as the $j$ -th column in $Z _ { p }$ , we can obtain

$$
\operatorname* { m i n } _ { Z _ { p } } \frac { 1 } { 2 } \| z _ { p } ^ { j } - f _ { p } ^ { j } \| _ { F } ^ { 2 } , ~ s . t . ~ z _ { p } ^ { j } \geq 0 , ~ ( z _ { p } ^ { j } ) ^ { T } 1 _ { l } = 1 ,
$$

where $\begin{array} { r } { f _ { p } ^ { i j } \ = \ \frac { \beta [ P _ { p } ^ { T } F Y ] _ { i j } + \gamma _ { p } ^ { i j } [ A _ { p } ^ { T } ( X ^ { p } H _ { p } H _ { p } ^ { T } ) ] _ { i j } } { \gamma _ { p } ^ { 2 } \sum _ { i = 1 } ^ { n } H _ { p } ^ { i j } + \beta + \alpha } } \end{array}$ and $[ P _ { p } ^ { T } F Y ]$ indicates the entry for the $j$ -th column and $i$ -th row in $P _ { p } ^ { T } F Y$ . Based on the Kahn-Kuhn-Tucker (KKT) conditions, we can achieve:

$$
z _ { p } ^ { j } = \operatorname* { m a x } ( f _ { p } ^ { j } + \eta _ { j } 1 _ { l } , 0 ) ,
$$

where $\eta$ is the Lagrangian multipliers obtained by Newton’s method. It needs the time complexity of $O ( n l d )$ to update $\{ Z _ { p } \} _ { p = 1 } ^ { v }$ , where $\begin{array} { r } { d = \sum _ { p = 1 } ^ { v } d _ { p } ^ { \ } } \end{array}$ and $d _ { p }$ is the dimension of $X ^ { p }$ .

2) Optimization of $\{ P _ { p } \} _ { p = 1 } ^ { v } \colon \mathrm { W i t h } \ : Z _ { p } , F , Y$ and $A _ { p }$ being fixed, we write the optimization of $P _ { p _ { p = 1 } } ^ { \ v }$ as follows:

$$
\operatorname* { m i n } _ { P _ { p } } \sum _ { p = 1 } ^ { v } \beta \| P _ { p } Z _ { p } - F Y \| _ { F } ^ { 2 } , ~ s . t . ~ P _ { p } ^ { T } P _ { p } = I .
$$

It equals to

$$
\operatorname * { m a x } _ { P _ { p } } T r ( P _ { p } ^ { T } W _ { p } ) , s . t . P _ { p } ^ { T } P _ { p } = I ,
$$

where $W _ { p } = F Y Z _ { p } ^ { T }$ . The optimal solution for $P _ { p }$ is $\Psi _ { P } \Sigma _ { P } ^ { T }$ , where $\Sigma _ { P }$ and $\Psi _ { P }$ are matrices comprising the first $l$ left and right singular vectors of $W _ { p }$ .

3) Optimization of $\{ A _ { p } \} _ { p = 1 } ^ { \mathit { v } } \colon \mathrm { W i t h } \ : Z _ { p } , F , Y$ and $P _ { p }$ being fixed, we write the optimization of $\{ A _ { p } \} _ { p = 1 } ^ { v }$ as follows:

$$
\operatorname* { m i n } _ { A _ { p } } \sum _ { p = 1 } ^ { v } \gamma _ { p } ^ { 2 } \lVert X ^ { p } H _ { p } - A _ { p } Z _ { p } H _ { p } \rVert _ { F } ^ { 2 } ,
$$

It can be rewritten as

$$
\operatorname* { m a x } _ { A _ { p } } T r ( A _ { p } ^ { T } J _ { p } ) , s . t . A _ { p } ^ { T } A _ { p } = I ,
$$

where $J _ { p } = X ^ { p } H _ { p } H _ { p } ^ { T } Z _ { p } ^ { T }$ . The optimal solution for $A _ { p }$ is $\Psi _ { A } \Sigma _ { A } ^ { T }$ , where $\Sigma _ { A }$ and $\Psi _ { A }$ are matrices consisting of the first $l$ left and right singular vectors of $J _ { p }$ .

4) Optimization of $F$ : With $Z _ { p }$ , $A _ { p }$ , $Y$ and $P _ { p }$ being fixed, we write the optimization of $F$ as follows:

$$
\operatorname* { m i n } _ { F } \sum _ { p = 1 } ^ { v } \beta \| P _ { p } Z _ { p } - F Y \| _ { F } ^ { 2 } , ~ s . t . ~ F _ { p } ^ { T } F _ { p } = I .
$$

It can be rewritten as follows:

$$
\operatorname* { m a x } _ { F } T r ( F ^ { T } B ) , s . t . F ^ { T } F = I ,
$$

where $\begin{array} { r } { B = \sum _ { p = 1 } ^ { v } \beta P _ { p } Z _ { p } Y ^ { T } } \end{array}$ . The optimal solution for $F$ is $\Psi _ { F } \Sigma _ { F } ^ { T }$ , where $\Sigma _ { F }$ and $\Psi _ { F }$ are matrices comprising the first $l$ left and right singular vectors of $B$ .

5) Optimization of $Y$ : With $Z _ { p }$ , $A _ { p }$ , $F$ and $P _ { p }$ being fixed, we write the optimization of $Y$ as follows:

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { Y } \sum _ { p = 1 } ^ { v } \beta \| P _ { p } Z _ { p } - F Y \| _ { F } ^ { 2 } , } \\ { \displaystyle s . t . Y _ { i j } \in \{ 0 , 1 \} , \sum _ { i = 1 } ^ { k } Y _ { i j } = 1 , \forall j = 1 , 2 , \cdots , n . } \end{array}
$$

Since the above optimization problem can be independently solved for each object, we can obtain

$$
\begin{array} { r l } & { \underset { Y _ { : , j } } { \operatorname* { m i n } } \displaystyle \sum _ { p = 1 } ^ { v } \| P _ { p } ^ { : , j } Z _ { p } ^ { : , j } - F Y _ { : , j } \| ^ { 2 } , } \\ & { s . t . Y _ { : , j } \in \{ 0 , 1 \} ^ { k } , \ \| Y _ { : , j } \| _ { 1 } = 1 . } \end{array}
$$

The optimal row is achieved by

$$
i ^ { * } = \arg _ { i } \operatorname* { m i n } \sum _ { p = 1 } ^ { v } \lVert P _ { p } ^ { : , j } Z _ { p } ^ { : , j } - F Y _ { : , j } \rVert ^ { 2 } .
$$

6) Optimization of $\gamma$ : With $Z _ { p } , A _ { p } , F , Y$ and $P _ { p }$ being fixed, we write the optimization of $\gamma$ as follows:

$$
\operatorname* { m i n } _ { \boldsymbol { \gamma } _ { p } ^ { 2 } } \sum _ { p = 1 } ^ { \boldsymbol { v } } \boldsymbol { \gamma } _ { p } ^ { 2 } \| \boldsymbol { X } ^ { p } H _ { p } - A _ { p } Z _ { p } H _ { p } \| _ { F } ^ { 2 } , ~ \boldsymbol { s . t . } ~ \boldsymbol { \gamma } ^ { T } \boldsymbol { 1 } = \boldsymbol { 1 } .
$$

According to Cauchy-schwarz inequality, we then update $\gamma$ as follows:

$$
\gamma _ { p } = \frac { 1 / \| X ^ { p } H _ { p } - A _ { p } Z _ { p } H _ { p } \| _ { F } ^ { 2 } } { \sum _ { p = 1 } ^ { v } 1 / \| X ^ { p } H _ { p } - A _ { p } Z _ { p } H _ { p } \| _ { F } ^ { 2 } } .
$$

Due to the optimal solutions of all sub-problems and convex property, the objective function in Eq. (3) monotonically decreases in each iteration until the convergence is achieved. Considering that the lower boundary of the objective function is zero, the proposed SOME-AS is able to converge to the local optimum. The procedure of solving SOME-AS is shown in Algorithm 1.

# Algorithm 1: Algorithm of SOME-AS

<html><body><table><tr><td>Input: Incomplete dataset {X𝑝}p=1, number of clusters k, missing index {Hp}p=1; Output: Cluster assignment Y. Initialize: Initialize Pp,F,Y,Ap,Zp and γ;</td></tr><tr><td>repeat Update Pp with Eq. (9); Update Ap with Eq. (11); Update Zp with Eq. (7); UpdateFwithEq. (13);</td></tr></table></body></html>

# Complexity Analysis

The proposed SOME-AS consists of six optimization subproblems as above mentioned. It costs $\bar { O ( } n l d )$ for updating $\{ Z _ { p } \} _ { p = 1 } ^ { v }$ . The computation cost of calculating $\{ P _ { p } \} _ { p = 1 } ^ { v }$ is $O ( ( n l ^ { \bar { 2 } } + l ^ { 3 } ) v )$ for all columns. It needs $O ( n l d + l ^ { 2 } d )$ in obtaining the optimal $\{ A _ { p } \} _ { p = 1 } ^ { v }$ . It costs $O ( n l ^ { 2 } v )$ time to optimize $F$ . The computation cost of $O ( l n k )$ is needed for computing $Y$ . It needs $O ( n l d )$ to update $\gamma$ . Therefore, the overall computation cost to solve SOME-AS is $O ( n l d + ( n l ^ { 2 } + l ^ { 3 } ) v ^ { \ast } + l ^ { 2 } d + n l ^ { 2 } v + l n k )$ . Consequently, the total time cost of optimizing SOME-AS is $O ( n )$ considering the property of multi-view datasets with large scales (i.e., $n \gg k , d , m )$ .

# Experiment

In this section, we conduct extensive experiments to verify the superiority of SOME-AS in terms of clustering results and running time on seven real-world datasets, which includes several multi-view datasets with large scales. Meanwhile, the parameter analysis, ablation experiments and convergence investigation are also performed to achieve a comprehensive study.

![](images/21f46c055ffc856dfa930992191fab1094f7b0939a193cd8ae324f16ded9141c.jpg)  
Figure 2: Clustering results of SOME-AS on dataset against different $\alpha$ .

Table 1: Clustering performance $\mathrm { ( A C C \% 1 5 T D \% ) }$ ) on all datasets.   

<html><body><table><tr><td>Data sets</td><td>BSV</td><td>MIC</td><td>MKKM-IK</td><td>AWP</td><td>DAIMC</td><td>APMC</td><td>MKKM-IK-MKC EEIMVC</td><td>VH</td><td>FIMVC-VIA</td><td>AUP-ID Ours</td></tr><tr><td>ORL</td><td></td><td>24.30±0.5037.60±1.50 59.82±2.00 68.62±0.05 68.00±2.3065.52±1.80</td><td></td><td></td><td></td><td></td><td>64.92±2.40</td><td></td><td></td><td>73.20±2.2067.00±1.42 76.30±2.90 76.00±2.40 78.20±0.05</td></tr><tr><td>ProteinFold</td><td></td><td>22.20±0.5015.80±0.50 26.00±1.05 29.00±0.95 28.70±1.60</td><td></td><td></td><td></td><td></td><td>17.90±0.80</td><td></td><td></td><td>27.90±1.7017.40±0.50 28.10±1.30 30.20±1.40 32.56±0.90</td></tr><tr><td>BDGP</td><td></td><td>34.95±1.00 25.40±0.60 32.20±0.20 23.60±0.20 28.10±0.03 28.14±0.04</td><td></td><td></td><td></td><td></td><td>40.80±0.22</td><td></td><td></td><td>44.00±0.03 43.62±0.65 39.85±0.05 48.10±0.10 50.22±0.03</td></tr><tr><td>SUNRGBD</td><td></td><td>6.12±0.05 14.60±0.4011.32±0.50 17.00±0.00 17.05±0.00 17.35±0.60</td><td></td><td></td><td></td><td></td><td>16.80±0.50</td><td>16.75±0.50</td><td></td><td>16.89±0.50 17.20±0.50 18.00±0.10</td></tr><tr><td>NUSWIDEOBJ12.00±0.00</td><td></td><td></td><td></td><td></td><td>13.80±0.40</td><td></td><td></td><td>12.70±0.20</td><td></td><td>12.95±0.0515.42±0.3017.00±0.00</td></tr><tr><td>Cifar10</td><td></td><td></td><td></td><td>1</td><td>95.80±0.50</td><td></td><td></td><td></td><td></td><td>96.20±0.05 96.30±0.2098.35±0.10</td></tr><tr><td>MNIST</td><td></td><td></td><td></td><td>1</td><td>95.60±0.40</td><td></td><td></td><td>-</td><td></td><td>98.10±0.02 98.15±0.05 99.20±0.01</td></tr></table></body></html>

Table 2: Clustering performance $( \mathrm { N M I } \% \pm \mathrm { S T D } \%$ ) on all datasets.   

<html><body><table><tr><td>Data sets</td><td>BSV</td><td>MIC</td><td>MKKM-IK</td><td>AWP</td><td>DAIMC</td><td>APMC</td><td>MKKM-IK-MKC</td><td>EEIMVC</td><td>VH</td><td>FIMVC-VIA AUP-ID</td></tr><tr><td>ORL</td><td></td><td>48.50±0.50 56.50±0.80 75.92±1.20 83.80±0.15 82.90±1.00 80.10±0.50</td><td></td><td></td><td></td><td></td><td>79.72±1.50</td><td></td><td></td><td>85.35±1.3081.00±0.30 88.02±1.00 87.50±1.20 89.80±0.50</td></tr><tr><td>ProteinFold</td><td>27.55±0.5016.70±1.00 33.65±0.50 36.12±0.50 37.80±1.00</td><td></td><td></td><td></td><td></td><td></td><td>24.90±0.80</td><td></td><td></td><td>36.05±0.7822.78±0.52 36.20±0.80 37.70±0.90 39.80±0.05</td></tr><tr><td>BDGP</td><td>12.90±0.75 4.50±0.80</td><td></td><td></td><td></td><td>7.45±0.18 4.70±0.15 8.71±0.03 8.09±0.02</td><td></td><td>16.40±0.20</td><td></td><td></td><td>19.90±0.1224.20±0.50 15.10±0.20 23.70±0.20 25.20±0.02</td></tr><tr><td>SUNRGBD</td><td></td><td></td><td></td><td></td><td>3.25±0.05 21.30±0.30 15.30±0.20 23.72±0.10 21.55±0.50 22.50±0.30</td><td></td><td>20.52±0.40</td><td>20.85±0.30</td><td></td><td>21.50±0.30 22.55±0.40 24.00±0.50</td></tr><tr><td>NUSWIDEOBJ2.70±0.05</td><td></td><td></td><td></td><td></td><td>11.95±0.30</td><td></td><td></td><td>10.40±0.20</td><td></td><td>10.30±0.05 11.80±0.1012.20±0.05</td></tr><tr><td>Cifar10</td><td></td><td></td><td></td><td></td><td>90.50±0.42</td><td></td><td></td><td>-</td><td></td><td>91.20±0.02 91.25±0.2093.00±0.50</td></tr><tr><td>MNIST</td><td></td><td></td><td></td><td></td><td>93.85±0.40</td><td>-</td><td></td><td></td><td></td><td>95.78±0.04 95.62±0.2095.70±0.02</td></tr></table></body></html>

![](images/fd1a0618a7338d066b18033547b379add0bb38d8a7db54eb2e1ce67fa7f6c51c.jpg)  
Figure 3: Clustering results of SOME-AS on dataset against different $\beta$ .

# Datasets and Compared Methods

We adopt seven widely used datasets in the evaluation. ORL consists of 400 face images and 3 views, whose dimensions are 3304, 4096 and 6750, respectively. BDGP contains 2500 samples from 5 categories. ProteinFold is a protein dataset consisting of 27 classes, whose data size is 2500. SUNRGBD is the dataset containing 10335 samples from 45 categories. NUSWIDEOBJ consists of 30000 samples from 31 classes. Cifar10 contains 50000 images from 10 categories. MNIST is the digit dataset consisting of 60000 samples. For these datasets, we randomly remove samples from different views to achieve the corresponding incomplete multiview dataset, which is consistent with the works in (Li et al. 2022d).

In the experiment, we adopt 11 representive methods for incomplete multi-view clustering, which consists of BSV $\mathrm { N g }$ , Jordan, and Weiss 2001), AWP (Nie, Tian, and Li 2018), MIC (Shao, He, and Yu 2015), MKKM-IK (Liu et al. 2017), DAIMC (Hu and Chen 2018), EEIMVC (Liu et al. 2021), $\mathrm { v ^ { 3 } h }$ (Fang et al. 2020), APMC (Guo and Ye 2019), MKKM-IK-MKC (Liu et al. 2020), FIMVC-VIA (Liu 2021) and AUP-ID (Wen et al. 2023).

![](images/5b3a3bbecd9c27fe936ef724dca29ebe65ec40f834ea1e82c6e221490a86d6ce.jpg)  
Figure 4: Clustering performance of SOME-AS on dataset with different missing ratios under ACC.

![](images/065c78796cf7f18abc684f018a74fd7294eee78a3ee598c733a4d3d4ada08dad.jpg)  
Figure 5: Clustering performance of SOME-AS on dataset with different missing ratios under NMI.

Table 3: Clustering performance (F1-score% STD%) on all datasets.   

<html><body><table><tr><td>Data sets</td><td>BSV</td><td>MIC</td><td>MKKM-IK</td><td>AWP</td><td>DAIMC</td><td>APMC</td><td>MKKM-IK-MKC</td><td>EEIMVC</td><td>VH</td><td>FIMVC-VIA AUP-ID</td></tr><tr><td>ORL</td><td></td><td></td><td></td><td></td><td>9.00±0.50 17.50±1.10 46.30±2.40 58.70±0.00 56.80±2.55 50.80±2.30</td><td></td><td>53.40±2.95</td><td></td><td></td><td>63.70±2.90 54.32±1.20 68.22±3.20 67.70±2.90 68.90±2.00</td></tr><tr><td>ProteinFold</td><td></td><td></td><td></td><td></td><td>12.35±0.0510.40±0.4014.30±0.5012.40±0.20 16.99±1.00</td><td></td><td>8.90±0.50</td><td></td><td></td><td>15.65±1.30 10.52±0.17 15.58±0.80 16.79±1.00 17.20±0.50</td></tr><tr><td>BDGP</td><td></td><td></td><td></td><td></td><td>28.72±0.52 29.90±0.05 25.20±0.1232.60±0.50 31.20±0.0031.22±0.05</td><td></td><td>30.20±0.15</td><td></td><td></td><td>32.95±0.05 35.40±0.30 31.68±0.02 35.52±0.10 38.00±0.05</td></tr><tr><td>SUNRGBD</td><td></td><td></td><td></td><td></td><td>6.90±0.02 9.50±0.31 7.00±0.15 11.60±0.1210.96±0.40 10.78±0.05</td><td></td><td>10.08±0.50</td><td>10.22±0.15</td><td></td><td>7.89±0.05 11.76±0.1013.20±0.02</td></tr><tr><td>NUSWIDEOBJ10.92±0.00</td><td></td><td></td><td></td><td></td><td>8.62±0.20</td><td></td><td></td><td>7.80±0.05</td><td></td><td>7.82±0.05 11.72±0.5013.28±0.05</td></tr><tr><td>Cifar10</td><td></td><td></td><td></td><td>1</td><td>92.20±0.50</td><td></td><td></td><td></td><td></td><td>92.82±0.12 92.95±0.15 94.20±0.10</td></tr><tr><td>MNIST</td><td></td><td></td><td></td><td>1</td><td>95.31±0.61</td><td></td><td></td><td>-</td><td></td><td>96.68±0.05 96.90±0.0097.10±0.00</td></tr></table></body></html>

Table 4: Clustering performance (Purity $\% \pm \mathrm { S T D } \%$ ) on all datasets.   

<html><body><table><tr><td>Data sets</td><td>BSV</td><td>MIC</td><td>MKKM-IK</td><td>AWP</td><td>DAIMC</td><td>APMC</td><td>MKKM-IK-MKC</td><td>EEIMVC</td><td>VH FIMVC-VIA</td><td>AUP-ID Ours</td></tr><tr><td>ORL</td><td></td><td>26.92±0.80 40.90±1.20 62.80±2.00 70.50±0.00 71.85±1.5069.20±1.20</td><td></td><td></td><td></td><td></td><td>67.60±2.20</td><td></td><td></td><td>76.00±2.10 70.20±1.00 78.60±2.20 78.81±2.00 80.50±0.05</td></tr><tr><td>ProteinFold</td><td></td><td>25.40±0.10 19.80±0.80 30.90±1.00 31.92±0.00 34.95±1.52</td><td></td><td></td><td></td><td></td><td>22.80±0.80</td><td></td><td></td><td>33.10±0.0722.40±0.50 33.70±1.10 35.99±1.7737.20±0.02</td></tr><tr><td>BDGP</td><td></td><td>36.72±0.60 25.70±0.20 33.40±0.20 24.00±0.00 28.42±0.0128.45±0.03</td><td></td><td></td><td></td><td></td><td>41.25±0.15</td><td></td><td></td><td>46.50±0.12 45.42±0.70 40.10±0.17 49.00±0.15 49.20±0.30</td></tr><tr><td>SUNRGBD</td><td></td><td>13.00±0.1032.40±0.30 27.00±0.3037.50±0.00 34.42±0.0733.25±0.52</td><td></td><td></td><td></td><td></td><td>32.95±0.30</td><td>33.60±0.50</td><td></td><td>34.30±0.50 34.55±0.60 36.20±0.35</td></tr><tr><td>NUSWIDEOBJ13.70±0.05</td><td></td><td></td><td></td><td></td><td>23.45±0.50</td><td></td><td></td><td>21.90±0.15</td><td></td><td>21.95±0.1023.70±0.20</td></tr><tr><td>Cifar10</td><td></td><td></td><td></td><td></td><td>95.80±0.50</td><td></td><td></td><td></td><td></td><td>96.20±0.00 96.32±0.2097.00±0.05</td></tr><tr><td>MNIST</td><td></td><td></td><td></td><td></td><td>97.60±0.50</td><td>-</td><td></td><td>-</td><td></td><td>98.27±0.05 98.48±0.05 99.20±0.03</td></tr></table></body></html>

![](images/3b4bca1bc3f04853b438aa5b7664f91a1f330d7ce4286064bdc15f5ae78f39e9.jpg)  
Figure 6: Sensity investigation of anchor number on dataset.

![](images/1a4933d4f044e76211bb86d834b898cebb6005627ad2e8154309896449b43a81.jpg)  
Figure 7: Ablation study on dataset.

To measure the performance for clustering, we employ four commonly used metrics including ACC, NMI, F1-score and Purity. Each experiment is repeated for 10 cycles to achieve the average results and variance. For the parameters of the compared methods, we set them as the recommended ones in the corresponding literatures. We perform all experiments on a computer with AMD Ryzen 5 1600X Six-Core Processor.

There are total two parameters needed to be determined in the experiment, i.e., $\alpha$ and $\beta$ , where $\alpha$ is the parameter of Frobenius norm and $\beta$ corresponds to the alignment mapping parameter. We select them in the range of $[ 0 . 0 0 1 , 0 . 0 1 , 0 . 1 , 1 , 1 0 ]$ on dataset. According to Figs. 2-3, we find that satisfied results can be achieve when $\alpha$ and $\beta$ are both 0.1. Moreover, it is observed that the clustering performance is relatively stable in the given range of parameters on dataset.

# Experimental Results

We report the clustering results on different multi-view datasets under four metrics in this section and adopt N/A to denote the unavailable results caused by out-of-memory errors. According to Tables 1-4, we can have conclusions as:

1. The proposed SOME-AS shows better performance than compared methods on most datasets, demonstrating its superiority in effectiveness under different metrics. For example, the proposed SOME-AS achieves more desired performance than MKKM-IK on SUNRGBD dataset under four metrics, i.e., $1 . 2 0 \%$ , $3 . 4 8 \%$ , $2 . 1 2 \%$ and $3 . 2 5 \%$ .

![](images/800f8a3dc216520b3fb87c18a3d7880c86a7d34543e7bac0d1e0642d0477b285.jpg)  
(a) Time comparison of different methods on all incomplete (b) Objective value of SOME-AS on ORL dataset. datasets.   
Figure 8: Time comparison and objective value.

2. Different from the traditional multi-view clustering methods built on subspace, the anchor-based works usually behave better on most cases, which are suitable for datasets with large scales.   
3. Among the compared methods based on anchor, the proposed SOME-AS is obviously more superior by introducing the latent integral subspace built on the cluster structure of the data with anchors from different views being aligned.

We also compare the clustering results of different methods (FIMVC-VIA and AUP-ID) with several missing rates for comparison. According to Figs. 4-5, we find that the proposed SOME-AS is more stable than other incomplete multi-view clustering methods on dataset. The advantages can be explained by the fact that learning the integral subspace reflecting the cluster structure with anchors from different views being aligned is helpful for incomplete multiview clustering with different missing rates.

# Sensity Investigation and Ablation Study

To analyze how the amount of anchors influences the final performance, we perform experiments by varying the number of anchors $l$ in the range of $[ 1 k , 2 k , 3 k ]$ and showing the corresponding results. As shown in Fig. 6, we observe that the proposed SOME-AS is not significantly affected by the number of anchors.

To validate the effectiveness of the alignment among anchors across views in learning an integral subspace revealing the cluster structure, we perform an ablation study and report the experimental results in Fig. 7, where “case 1” indicates that this strategy is not adopted and “case $2 ^ { \circ }$ represents the opposite one. According to Fig. 7, we find that the strategy of learning an integral subspace revealing the cluster structure with anchors from different views being aligned is able to improve the clustering performance on dataset, which demonstrates the necessarity of this adopted strategy in the proposed SOME-AS.

# Running Time Analysis and Convergence Study

We list the running time of different incomplete multi-view clustering methods to demonstrate the efficiency of the proposed SOME-AS. According to Fig. 8, we observe that SOME-AS is able to significantly reduce the running time with the guidance of the constructed anchor graphs compared with methods based on the graph on most datasets. Besides, SOME-AS achieves desired clustering performance and needs relatively little running time, which well balances the effectiveness and efficiency. Compared to some incomplete multi-view clustering methods based on anchor, i.e., FIMVC-VIA, the proposed SOME-AS requires more running time introduced by the alignment mapping. Considering the achieved desired performance on most datasets, the time costs consumed by SOME-AS is worthwhile.

We perform experiments to analyze the convergence of SOME-AS on some datasets. Based on Fig. 8, we observe that the objective value of SOME-AS monotonically decreases in each iteration, verifying the convergence.

# Conclusion

This paper proposes a novel scalable one-pass incomplete multi-view clustering by aligning anchors. The complementary information across views is captured by building the anchor graph for each view on the incomplete dataset. The integral subspace revealing the cluster structure is learned by taking the alignment among anchors from different views into consideration. The cluster assignment and centroid representation with orthogonal constraint are built to approximate the integral subspace. Extensive experiments demonstrate the efficiency and effectiveness of SOME-AS on several incomplete multi-view datasets under different metrics.

# Acknowledgments

This work was supported by the National Key Research and Development Program of China under Grant 2023YFF0905000.