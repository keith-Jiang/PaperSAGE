# Tensorized Label Learning Based Fast Fuzzy Clustering

Xingyu Xue\*, Jingjing Xue\*, Quanxue Gao†, Qianqian Wang

School of Telecommunication Engineering, Xidian University, Shaanxi 710071, China. 23011211150@stu.xidian.edu.cn,xuejingjing $@$ xidian.edu.cn,qxgao $@$ xidian.edu.cn,qqwang@xidian.edu.cn

# Abstract

Multi-view graph clustering methods have been widely concerned due to the ability of dealing with arbitrarily shaped datasets. However, many methods with higher time and space complexity make them challenging to deal with large-scale datasets. Besides, many fuzzy clustering methods needs additional regularization terms or hyper-parameters to obtain the membership matrix or avoid trivial solutions, which weakens the model generalization ability. Furthermore, inconsistent clustering labels can arise when there are significant discrepancies between views, making it challenging to effectively leverage the complementary information from different views. To this end, we propose Tensorized Label Learning based Fast Fuzzy Clustering (TLLFFC). Specifically, we design a novel balanced regularization term to reduce pressure of tuning regularization parameters for fuzzy clustering. The label transmission strategy with the anchor graph makes TLLFFC suitable for large-scale datasets. Moreover, incorporating the Schatten $p$ -norm regularization on the label matrices can effectively unearth the complementary information distributed among views, thereby align the labels across views more consistently. Extensive experiments verify the superiority of TLLFFC.

# Introduction

In recent decades, data mining and machine learning techniques have been increasingly applied in fields such as image processing (Gong, Yuan, and Bao 2021a), video annotation (Liu and Tsang 2015), and data analysis (Liu and Tsang 2017). Meanwhile, with the widespread use of various sensors and technologies, descriptions of the same object have become increasingly diverse and heterogeneous (Gong et al. 2022; Gong, Yuan, and Bao 2021b). Multi-view clustering methods (Cui et al. 2023) can identify these multiview datasets with common features and integrate them into groups, ensuring that samples within the same cluster exhibit high similarity. Among them, multi-view graph clustering methods (Wan et al. 2023) become popular due to their abilities of dealing with arbitrarily shaped multimedia data better. Multi-view graph clustering methods (Wang, Yang, and Liu 2019; Wu, Lin, and Zha 2019) usually need to construct an $n \times n$ large graph, and implement the eigenvalue decomposition step, or solve the optimal problem directly (Wang, Yang, and Liu 2019), whose time complexity are $\mathcal { O } ( V n ^ { 3 } )$ or $\breve { \mathcal { O } } ( V n ^ { 2 } )$ . Therefore, they are not suitable for large-scale multi-view datasets.

The anchor technique (Xia et al. 2023) has been utilized to reduce the size of the similarity matrix and expedite the optimization process, yielding impressive performance across various applications. Anchor points, which are representative data points, effectively cover the entire data point cloud and capture the internal structure of the data. Anchor-based methods first generate representative anchors from the original samples using a specific strategy and then construct a similarity graph to measure the similarity between samples and anchors. Following this, clustering processes are applied to the representation matrix.

Further, some researches use the two-step strategy to get final label matrices (Wang et al. 2021; Kang et al. 2020), whose solutions are suboptimal, as they are far away from the solutions obtained by directly solving the original problem. Therefore, (1) the rank constraint are introduce to ensure $c$ connected components to avoid the post-process (Xia et al. 2023; Li et al. 2020; Fang et al. 2023; Xue et al. 2024); (2) some novel optimization strategies (Qiang et al. 2023; Nie et al. 2022) or Nonnegative Matrix Factorization (NMF) models (Yang et al. 2020) are proposed, which can obtain the final solution directly. The former has high requirements on parameters, and in some cases it may not be possible to obtain bipartite graphs with clear connected components. The latter employs the crisp partitioning to make each data point belong to a cluster. In contrast to crisp partitioning, fuzzy clustering methods (Zhang et al. 2024) can represent the degree of ambiguity or certainty with which a data point belongs to a cluster. In practical scenarios, the class attributes of most objects are inherently ambiguous, making fuzzy clustering techniques a more accurate reflection of the real world. Consequently, many researchers have chosen to explore and advance this technique.

However, in order to obtain the membership matrix or avoid trivial solutions, some regularization terms or hyperparameters are introduced (Jiang and Gao 2022; Hu et al. 2024; Yang et al. 2024). Considering these parameters do not have practical interpretations, how to tactfully determine them is a severe challenge. A common way is to search these parameters in a wide range for each dataset, which weakens the model generalization ability. Therefore, designing a fuzzy clustering model with less parameters is necessary.

![](images/b8cd6119312278a28de5e990b19affef844961eba4da4e063cafe9abdd26413c.jpg)  
Figure 1: Construction of 3rd-order tensor

To this end, we design a Tensorized Label Learning based Fast Fuzzy Clustering (TLLFFC) model to solve the above problems simultaneously. First, in order to cope with largescale multi-view datasets and avoid the post-process, we utilize anchor and label transmission strategies. Second, we design a novel balanced regularization term which, combined with prior anchor graph information, can avoid trivial solutions and achieve fuzzy clustering without additional regularization parameters. Finally, in order to explore the complementary information of different views better, inspired by the tensor Schatten $p$ -norm (Li et al. 2023, 2024), we use the soft labels of each view to form a 3rd-order tensor as illustrated in Figure 1, and exploit tensor Schatten $p$ -norm regularizer on this term. This helps to get the complementary information embedded in the soft labels of different views. The main contributions are summarized as follows

• A novel balanced regularization term is proposed to avoid trivial solutions without additional regularization terms. Thus, TLLFFC has no pressure of tuning regularization parameters w.r.t. trivial solutions or fuzziness.   
• The label transmission strategy with the anchor graph transmits the label information from samples to anchors, which makes TLLFFC suitable for large-scale problems and can mine the intrinsic clustering structure between samples and anchors.   
• We leverage the tensor Schatten $p$ -norm regularization to fully exploit the structural and complementary information among different views. This ensures that labels of samples from different views are more likely to align, thereby improving clustering performance.   
• We provide an efficient and effective optimization algorithm for the proposed model. Extensive experiments on different datasets show the superiority of proposed TLLFFC.

# Notations

X(v) ∈ Rdv×n denotes the sample data of v-views, xi(jv) is $( i , j )$ -element of $\mathbf { X } ^ { ( v ) } , \mathbf { x } _ { j } ^ { ( v ) }$ and $\mathbf { x } ^ { ( v ) ^ { i } }$ are the $j$ th column and ith row of X(v) respectively. x(jv)T a nd X(v)T are the transpose of $\mathbf { x } _ { j } ^ { ( v ) }$ and respectively. 1 is a column vector of all ones. $\mathbf { H } ^ { ( v ) } \in \mathbb { R } ^ { n \times c }$ is a membership matrix, and $h _ { i j } ^ { ( v ) }$ represents the membership of the ith sample to the $j$ th class, satisfying $\begin{array} { r } { \sum _ { j = 1 } ^ { c } h _ { i j } ^ { ( v ) } = 1 } \end{array}$ . We use bold calligraphy letters for 3rd-order tensors, $\mathcal { H } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ . Moreover, the ith frontal slice of $\varkappa$ is $\dot { \mathbf { H } } ^ { ( i ) }$ . $\overline { { \mathcal { H } } }$ is the discrete Fourier transform of $\varkappa$ along the third dimension, $\overline { { \pmb { \mathscr { H } } } } = \mathrm { f f t } ( \pmb { \mathscr { H } } , [ \mathbf { \Lambda } ] , \mathbf { 3 } )$ . Thus, $\pmb { \mathcal { H } } = \mathrm { i f f t } ( \overline { { \pmb { \mathcal { H } } } } , [ ] , 3 )$ . Besides, all boldface uppercase letters represent matrices, all boldface lowercase letters represent vectors.

![](images/65de7b9cfc1c76d1ed129dccdf963222354baf3816d4878801fc18bf88cc68d9.jpg)  
Figure 2: The illustration of probability matrix $\mathbf { B }$ and $\mathbf { H }$ .

Definition 1 (Tensor Schatten $p$ -norm (Gao et al. 2021)) Given $\mathcal { H } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ , $h = m i n ( n _ { 1 } , n _ { 2 } )$ , the tensor Schatten $p$ -norm of $\varkappa$ is

$$
\left. \mathbf { \mathcal { H } } \right. _ { \mathbb { S } _ { \mathcal { P } } } = \left( \sum _ { i = 1 } ^ { n _ { 3 } } \left. \mathbf { \overline { { \mathcal { H } } } } ^ { ( i ) } \right. _ { \mathcal { S } _ { \mathcal { P } } } ^ { p } \right) ^ { \frac { 1 } { p } } = \left( \sum _ { i = 1 } ^ { n _ { 3 } } \sum _ { j = 1 } ^ { h } \sigma _ { j } \left( \mathbf { \overline { { \mathcal { H } } } } ^ { ( i ) } \right) ^ { p } \right) ^ { \frac { 1 } { p } } ,
$$

where $\sigma _ { j } ( \overline { { \pmb { \mathcal { H } } } } ^ { ( i ) } )$ denotes the jth singular value of $\overline { { \mathcal { H } } } ^ { ( i ) }$ .

# Methodology

# Label Transmission Strategy

The anchor graph $\mathbf { B } \in \mathbb { R } ^ { n \times m }$ shows the prior probability knowledge between $n$ samples and $m$ anchors. Given the label of samples $\mathbf { H } \in \mathbb { R } ^ { n \times c }$ , the label of anchors $\mathbf { F } \in \mathbb { R } ^ { m \times c }$ can be obtained by the Label Transmission strategy (Liu, He, and Chang 2010), i.e., $\mathbf { F } = \mathbf { B } ^ { \mathrm { T } } \mathbf { H }$ . The label of anchors can be decided by the label of its neighboring nodes, i.e., $\begin{array} { r } { f _ { i l } = \sum _ { j = 1 } ^ { n } b _ { i j } ^ { \mathrm { T } } \dot { h _ { j l } } ( i = 1 , 2 , \cdots , m ) \dot { \bar { \iota = 1 } } , 2 , \cdots , c ) } \end{array}$ . The greater the similarity to the node, the greater the influence weight of its neighboring nodes on its label. Figure 2 shows the illustration of $\mathbf { B }$ and $\mathbf { H }$ .

# Balanced Clustering

If we only require that the elements of the anchor label matrix $\mathbf { F }$ are non-negative, and each row sums to one. Then, the solution is not unique, and it cannot guarantee the balance of anchor across clusters. In order to avoid the trivial solution and obtain a clear clustering partition, our goal is to make the degree of each sample’s membership to each class vary as much as possible. Therefore, some researchers propose the

following clustering model

$$
\operatorname* { m a x } _ { \mathbf { F } \mathbf { 1 } = \mathbf { 1 } , f _ { i j } \geqslant 0 } T r ( \mathbf { F } ^ { \mathrm { T } } \mathbf { F } ) = \sum _ { j = 1 } ^ { c } \sum _ { i = 1 } ^ { n } f _ { i j } ^ { 2 } .
$$

Obviously, as $\mathbf { F 1 } = \mathbf { 1 }$ , the model (2) tends to make the maximum value of each row of $\mathbf { F }$ be 1. That is, it tends to make the probability that each data point belongs to a certain category as large as possible, in short, it has a clear clustering partition. However, this problem has a trivial solution $\mathbf { F } ^ { ( 1 ) }$ in which all data points are divided into one category.

Furthermore, in order to avoid this trivial solution, some researchers always add additional regularization terms which increase the pressure of tuning parameters. On the contrary, we design a novel non-parameter clustering model (3) that can avoid the above trivial solution

$$
\operatorname* { m a x } _ { \mathbf { F } \mathbf { 1 } = \mathbf { 1 } , \mathbf { F } > \mathbf { 0 } } \| \mathbf { F } ^ { \mathrm { T } } \| _ { 2 , 1 } = \operatorname* { m a x } _ { \mathbf { F } \mathbf { 1 } = \mathbf { 1 } , \mathbf { F } > \mathbf { 0 } } \sum _ { j = 1 } ^ { c } \sqrt { \mathbf { f } _ { j } ^ { \mathrm { T } } \mathbf { f } _ { j } } .
$$

Obviously, Model (3) can avoid trivial solution $\mathbf { F } ^ { ( 1 ) }$ , its optimal solution is $\mathbf { F } ^ { ( 2 ) }$ , a discrete indicator matrix, where, each sample belongs to only one category and the number of samples in each class is the same. The objective values of model (3) about two label matrices are as follows

$$
\begin{array} { r } { o b j ( \mathbf { F } ^ { ( 1 ) } ) = \sqrt { n } ; } \\ { o b j ( \mathbf { F } ^ { ( 2 ) } ) = et { } { ' } \sum _ { i = 1 } ^ { c } \sqrt { n _ { i } } . } \end{array}
$$

It is clear that the inequality $\begin{array} { r } { \sum _ { i = 1 } ^ { c } \sqrt { n _ { i } } \geqslant \sqrt { \sum _ { i = 1 } ^ { c } n _ { i } } } \end{array}$ holds when $n _ { i } \geqslant 0$ . Thus, $\mathbf { F } ^ { ( 1 ) }$ is not optimal solution of model (3), which can avoid trivial solution without additional regularization terms. Next, Theorem 1 gives a further analysis of model (3).

Theorem 1 The following problem arrives its maximum value when $F$ is the discrete indicator matrix and $\begin{array} { r } { n _ { j } = \frac { n } { c } } \end{array}$ .

$$
\operatorname* { m a x } _ { F I = I , F \geqslant 0 } \| F ^ { \mathrm { T } } \| _ { 2 , 1 } = \operatorname* { m a x } _ { F I = I , F \geqslant 0 } \sum _ { j = 1 } ^ { c } \sqrt { f _ { j } ^ { \mathrm { T } } f _ { j } } .
$$

Proof 1 Obviously, when $F$ is the discrete indicator matrix, we have $\begin{array} { r } { \sum _ { j = 1 } ^ { c } \sqrt { f _ { j } ^ { \mathrm { T } } f _ { j } ^ { \mathrm { } } } = \sum _ { j = 1 } ^ { c } \sqrt { n _ { j } } , } \end{array}$ , where $n _ { j }$ is the number of samples belonging to the jth category, $\textstyle \sum _ { j = 1 } ^ { c } n _ { j } = n$ . According to Cauchy–Schwarz inequality (Bhatia and Davis 1995), we have

$$
\| F ^ { \mathrm { T } } \| _ { 2 , 1 } ^ { 2 } = \left( \sum _ { j = 1 } ^ { c } \sqrt { f _ { j } ^ { \mathrm { T } } f _ { j } } \right) ^ { 2 } = \left( \sum _ { j = 1 } ^ { c } \sqrt { n _ { j } } \right) ^ { 2 }
$$

$$
\leqslant { \sum } _ { j = 1 } ^ { c } n _ { j } { \sum } _ { j = 1 } ^ { c } 1 = c n ,
$$

the two sides are equal if and only $i f { \sqrt { n _ { 1 } } } = { \sqrt { n _ { 2 } } } = \cdot \cdot \cdot =$ ${ \sqrt { n _ { c } } } ,$ i.e., $\begin{array} { r } { n _ { j } ~ = ~ \frac { n } { c } } \end{array}$ . At this time, $\textstyle \sum _ { j = 1 } ^ { c } { \sqrt { f _ { j } ^ { \mathrm { T } } f _ { j } } }$ reaches its maximum value.

# The Proposed Model

Therefore, motivated by the above, we can tend this idea to the multi-view field and propose a parameter-free model (the first term of model (6)) to avoid trivial solution, $\mathbf { H } ^ { ( \nu ) } \in \mathbb { R } ^ { n \times c }$ is the label matrix of samples in the $\boldsymbol { v }$ th view. Furthermore, in order to harness the complementary information and spatial structure across views, we employ the tensor Schatten $p { \cdot }$ - norm constraint on the tensorial form of $\mathbf { H } ^ { ( \nu ) }$ to ensure that the labels for samples within each view remain consistent. The objective function is

$$
\operatorname* { m a x } _ { \mathbf { H } ^ { ( \nu ) } \geqslant \mathbf { 0 } , \mathbf { H } ^ { ( \nu ) } \mathbf { 1 } = \mathbf { 1 } , } \sum _ { v = 1 } ^ { V } \| \mathbf { H } ^ { ( \nu ) ^ { \mathrm { T } } } \mathbf { B } ^ { ( \nu ) } \| _ { 2 , 1 } - \lambda \| \mathbf { \mathcal { H } } \| _ { \mathfrak { S } \{ p \} } ^ { p } ,
$$

According to the definition of the $\ell _ { 2 , 1 }$ -norm, the objective function can be written as:

$$
\operatorname* { m a x } _ { \mathbf { H } ^ { ( \nu ) } \geqslant \mathbf { 0 } , \mathbf { H } ^ { ( \nu ) } \mathbf { 1 } = \mathbf { 1 } } \sum _ { v = 1 } ^ { V } \sum _ { j = 1 } ^ { c } \sqrt { \mathbf { h } _ { j } ^ { ( \nu ) } { } ^ { \mathrm { T } } \mathbf { B } ^ { ( \nu ) } { \mathbf B } ^ { ( \nu ) } { } ^ { \mathrm { T } } \mathbf { h } _ { j } ^ { ( \nu ) } } - \lambda \| \mathcal { H } \| _ { \mathcal { S } \{ p \} } ^ { p } ,
$$

where, $0 < p \leqslant 1 , \lambda$ is the hyper-parameter of the Schatten $p$ -norm term. $\mathbf { H } ^ { ( \nu ) }$ represents the $\boldsymbol { v }$ th lateral slice of tensor $\mathcal { H } \in \mathbb { R } ^ { n \times V \times c }$ . The tensor construction process is illustrated in Figure 1.

Remark 1 The regularizer in the model (6) is used to explore the complementary information embedded in inter-views cluster assignment matrices $\pmb { H } ^ { ( v ) }$ $( v = 1 , 2 , \cdots , V )$ . The c-th frontal slice $\pmb { \Delta } ^ { ( c ) }$ in Figure $^ { \small 1 }$ describes the similarity between n sample points and the c clusters in different views. The ideal label matrix $\pmb { H } ^ { ( v ) }$ is consistent in different views. Since different views usually show different cluster structures, imposing tensor Schatten $p$ -norm minimization (Gao et al. 2021; Xia et al. 2023) constraint on $\varkappa$ can make sure each $\pmb { \Delta } ^ { ( c ) }$ has spatial low-rank structure. Thus, $\pmb { \Delta } ^ { ( c ) }$ can well characterize the complementary information embedded in inter-views.

# Optimization

We employ the Augmented Lagrange Multiplier (ALM) method to address problem (6). Introduce the auxiliary variable $\mathcal { I }$ and set $\mathbf { \mathcal { H } } = \mathbf { \mathcal { I } }$ , we have

$$
\begin{array} { r l } { \displaystyle \operatorname* { m a x } _ { \mathbf { H } ^ { ( \nu ) } \geqslant \mathbf { 0 } , \mathbf { H } ^ { ( \nu ) } \mathbf { 1 } = \mathbf { 1 } , \mathcal { T } } \sum _ { v = 1 } ^ { c } \sum _ { j = 1 } ^ { c } \sqrt { \mathbf { h } _ { j } ^ { ( \nu ) } { } ^ { \mathrm { T } } \mathbf { B } ^ { ( \nu ) } { \mathbf { B } ^ { ( \nu ) } } ^ { \mathrm { T } } \mathbf { h } _ { j } ^ { ( \nu ) } } - \lambda \| \mathcal { T } \| _ { \mathcal { S } { p } } ^ { p } } & { } \\ { \displaystyle \quad \quad - \frac { \mu } { 2 } \| \mathcal { H } - \mathcal { I } + \frac { Q } { \mu } \| _ { F } ^ { 2 } , } & { } \end{array}
$$

where $\mathfrak { Q }$ represents the Lagrange multiplier, $\mu$ is the penalty parameter. The optimization process can be separated into two steps:

(1) Solve $\varkappa$ with fixed $\mathcal { I }$ . (7) becomes

$$
\begin{array} { c } { \displaystyle \operatorname* { m a x } _ { \mathbf { H } ^ { ( \nu ) } \geq \mathbf { 0 } , \mathbf { H } ^ { ( \nu ) } \mathbf { 1 } = \mathbf { 1 } } \displaystyle \sum _ { v = 1 } ^ { V } \sum _ { j = 1 } ^ { c } \sqrt { \mathbf { h } _ { j } ^ { ( \nu ) } { } ^ { \mathrm { T } } \mathbf { B } ^ { ( \nu ) } { \mathbf { B } ^ { ( \nu ) } } ^ { \mathrm { T } } \mathbf { h } _ { j } ^ { ( \nu ) } } } \\ { \displaystyle \qquad - \frac { \mu } { 2 } \| \mathcal { H } - \mathcal { I } + \frac { \underline { { Q } } } { \mu } \| _ { F } ^ { 2 } . } \end{array}
$$

Algorithm 1: Solving problem (6)

Input: Data matrices X(v) V $\{ \mathbf { X } ^ { ( \nu ) } \} _ { v = 1 } ^ { V } \in \mathbb { R } ^ { d _ { v } \times n }$ , the number of anchors $m$ , the number of classes $c$ .

Parameter: $\mu = 1 0 ^ { - 3 }$ , $\eta = 1 . 1$ , $m a x _ { - } \mu = 1 0 ^ { 9 } , \rangle$ .

Output: Cluster assignment matrix $\mathbf { H } \in \mathbb { R } ^ { n \times c }$ of each data points.

1: Initialize $\boldsymbol { \mathcal { Q } } = \mathcal { I } = \mathbf { 0 } , \mathbf { H } ^ { ( }$ $\mathbf { H } ^ { ( \nu ) }$ is an identity matrix. Select anchor points and construct anchor graphs $\mathbf { B } ^ { ( \nu ) } ( v =$ $1 , 2 , \cdots , V )$ by (Xia et al. 2023).   
2: while not converge do   
3: Update $\mathcal { I }$ by (18);   
4: Update $\mathbf { H } ^ { ( \nu ) }$ by solving problem (9);   
5: Update $\mathfrak { Q }$ and $\mu$ $\therefore \mathcal { Q } = \mathcal { Q } + \mu ( \mathcal { H } - \mathcal { T } ) , \mu = \eta \mu ;$   
6: end while   
7: Directly achieve $c$ clusters based on the cluster assignment matrix $\begin{array} { r } { \mathbf { H } = \sum _ { v = 1 } ^ { V } \mathbf { H } ^ { ( \nu ) } } \end{array}$ .   
8: return Clustering results.

Since all $\mathbf { H } ^ { ( \nu ) } ( v = 1 , 2 , \cdot \cdot \cdot , V )$ are independent, then (8) can be decomposed into $V$ independent sub-optimization problems. Then, the optimization problem of solving $\mathbf { H } ^ { ( \nu ) }$ is

$$
\begin{array} { c } { { \displaystyle \operatorname* { m a x } _ { { \bf { H } } ^ { ( \nu ) } \geqslant { \bf { 0 } } , { \bf { H } } ^ { ( \nu ) } { \bf { 1 } } = { \bf { 1 } } } \sum _ { j = 1 } ^ { c } \sqrt { { \bf { h } } _ { j } ^ { ( \nu ) } { \bf { ^ { T } } } { \bf { B } } ^ { ( \nu ) } { \bf { B } } ^ { ( \nu ) } { \bf { ^ { T } } } { \bf { h } } _ { j } ^ { ( \nu ) } } } } \\ { { - \displaystyle \frac { \mu } { 2 } \| { \bf { H } } ^ { ( \nu ) } - { \bf { G } } ^ { ( \nu ) } \| _ { F } ^ { 2 } , } } \end{array}
$$

where, ${ \bf G } ^ { ( \nu ) } { = } { \bf J } ^ { ( \nu ) } { - } \frac { { \bf Q } ^ { ( \nu ) } } { \mu }$ . As $\sqrt { { \bf h } _ { j } ^ { ( \nu ) } { } ^ { \mathrm { T } } { \bf B } ^ { ( \nu ) } { \bf B } ^ { ( \nu ) } { } ^ { \mathrm { T } } { \bf h } _ { j } ^ { ( \nu ) } }$ is convex w.r.t. $\mathbf { h } _ { j _ { . } } ^ { ( \nu ) }$ , we use Iteratively Re-Weighted (IRW) (Nie et al. 2016) algorithm to solve problem (9) iteratively, where, the intermediate variable $\mathbf { D } ^ { ( \nu ) }$ is

$$
{ \bf { d } } _ { j } ^ { ( \nu ) } = \frac { { { \bf { B } } ^ { ( \nu ) } } { { \bf { B } } ^ { ( \nu ) } } ^ { \mathrm { T } } { { \bf { h } } _ { j } ^ { ( \nu ) } } } { { \sqrt { { \bf { h } } _ { j } ^ { ( \nu ) } } ^ { \mathrm { T } } } { { \bf { B } } ^ { ( \nu ) } } { { \bf { B } } ^ { ( \nu ) } } ^ { \mathrm { T } } { { \bf { h } } _ { j } ^ { ( \nu ) } } }  .
$$

Therefore, $\mathbf { H } ^ { ( \nu ) }$ is updated by solving the following problem

$$
\operatorname* { m a x } _ { \mathbf { H } ^ { ( \nu ) } \geqslant \mathbf { 0 } , \mathbf { H } ^ { ( \nu ) } \mathbf { 1 } = \mathbf { 1 } } T r ( \mathbf { D } ^ { ( \nu ) } ^ { \mathrm { T } } \mathbf { H } ^ { ( \nu ) } ) - \frac { \mu } { 2 } \| \mathbf { H } ^ { ( \nu ) } - \mathbf { G } ^ { ( \nu ) } \| _ { F } ^ { 2 } .
$$

Let $\mathbf { A } ^ { ( \nu ) } = \mathbf { D } ^ { ( \nu ) } + \mu \mathbf { G } ^ { ( \nu ) }$ , the above problem is

$$
\operatorname* { m i n } _ { \mathbf { H } ^ { ( \nu ) } \geq \mathbf { 0 } , \mathbf { H } ^ { ( \nu ) } \mathbf { 1 } = \mathbf { 1 } } T r ( \mathbf { H ^ { ( \nu ) } } ^ { \mathrm { T } } \mathbf { H ^ { ( \nu ) } } ) - \frac { 2 } { \mu } T r ( \mathbf { A ^ { ( \nu ) } } ^ { \mathrm { T } } \mathbf { H ^ { ( \nu ) } } ) .
$$

As each row of $\mathbf { H } ^ { ( \nu ) }$ is independent, solving (12) is equivalent to solving the following $n$ problems

$$
\operatorname* { m i n } _ { \mathbf { h } ^ { ( \nu ) ^ { i } } \mathbf { 1 } = 1 , \mathbf { h } ^ { ( \nu ) ^ { i } } \geqslant \mathbf { 0 } } \sum _ { i = 1 } ^ { n } \vert \vert \mathbf { h } ^ { ( \nu ) ^ { i } } - \frac { 1 } { \mu } \mathbf { a } ^ { ( \nu ) ^ { i } } \vert \vert _ { 2 } ^ { 2 } ,
$$

where, $\mathbf { h } ^ { ( \nu ) ^ { i } }$ denotes the $i$ th row of $\mathbf { H } ^ { ( \nu ) }$ . Problem (13) can be solved by an efficient iterative method (Huang, Nie, and Huang 2015). Therefore, problem (9) can be solved by updating $\mathbf { D } ^ { ( \nu ) }$ by Eq. (10) and updating $\mathbf { H } ^ { ( \nu ) }$ by solving problem (13) iteratively.

(2) Solve $\mathcal { I }$ with fixed $\varkappa$ . (7) becomes

$$
\operatorname* { m i n } _ { \mathcal { T } } \frac { \lambda } { \mu } \Vert \mathcal { T } \Vert _ { \mathfrak { S } \mathfrak { p } } ^ { { p } } + \frac { 1 } { 2 } \Vert \mathcal { H } - \mathcal { T } + \frac { \mathcal { Q } } { \mu } \Vert _ { F } ^ { 2 } .
$$

After completing the square regarding $\mathcal { I }$ , we can deduce

$$
\mathcal { T } ^ { * } = \arg \operatorname* { m i n } \frac { \lambda } { \mu } \| \mathcal { I } \| _ { \mathfrak { S } \mathcal { P } } ^ { p } + \frac { 1 } { 2 } \| \mathcal { H } - \mathcal { I } + \frac { \mathcal { Q } } { \mu } \| _ { F } ^ { 2 } ,
$$

which has a closed-form solution as Theorem 2 (Gao et al. 2021):

Theorem 2 Let $\pmb { S } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ have the $t$ -SVD $\pmb { \mathscr { s } } = \pmb { \mathscr { u } } *$ $\pmb { A } * \pmb { \nu } ^ { \mathrm { T } }$ . For the problem

$$
\operatorname* { m i n } _ { \pmb { x } } \frac { 1 } { 2 } \left\| \pmb { x } - \pmb { S } \right\| _ { F } ^ { 2 } + \tau \left\| \pmb { X } \right\| _ { \pmb { \mathcal { G } } p } ^ { p } ,
$$

the optimal solution is

$$
\pmb { \chi } ^ { \ast } = \Gamma _ { \tau } \left( \pmb { S } \right) = \pmb { \mathcal { U } } \ast \mathrm { i f f t } \left( \pmb { P } _ { \tau } \left( \pmb { \overline { { \pmb { S } } } } \right) \right) \ast \pmb { \mathcal { V } } ^ { \mathrm { T } } ,
$$

where $\pmb { P } _ { \tau } ( \overline { { \pmb { S } } } ) \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ is a $f$ -diagonal tensor, whose diagonal elements can be obtained by the General Shrinkage Thresholding algorithm (Gao et al. 2021).

Therefore, the solution of (15) is

$$
\mathcal { I } ^ { * } = \Gamma _ { \frac { \lambda } { \mu } } ( \mathcal { H } + \frac { \mathcal { Q } } { \mu } ) .
$$

Algorithm 1 summaries the details.

# Time and Space Complexity Analysis

Time Complexity: TLLFFC consists of two stages: 1) Construction of B(v) V , needs $\mathcal { O } ( V n m d + V n m \log ( m ) )$ , where $\begin{array} { r } { d = \sum _ { v = 1 } ^ { V } d _ { v } ; 2 ) } \end{array}$ Iterative updating $\mathcal { I }$ and $\{ \mathbf { H } ^ { ( \nu ) } \} _ { v = 1 } ^ { V }$ , needs $\mathcal { O } ( V n c \bar { \log ( V n ) } + V ^ { 2 } n c )$ and $\mathcal { O } ( V n c + V m ^ { 2 } c )$ respectively. $V , n , m , d _ { v } , c$ are the number of views, samples, anchors, features and classes, respectively. Given that $m , c , V$ are relatively small, the overall time complexity of our proposed method is $\mathcal { O } ( V n m d + V m ^ { 2 } c )$ .

Space Complexity: The storage memories for {B(v)}vV= $\mathcal { I } , \mathcal { H }$ and $\mathfrak { Q }$ needs $\mathcal { O } ( V n m ) , \ O ( V n c ) , \ O ( \dot { V } n c )$ and ${ \mathcal { O } } ( V n c )$ , respectively. The overall space complexity of TLLFFC is $\mathcal { O } ( V n m + 3 V n c )$ .

# Experiments

# Datasets, Compared Methods and Metrics

Datasets: Our experiments are executed on five widelyrecognized datasets to validate our model’s effectiveness. Table 1 gives a brief description of these datasets. All experiments are implemented on a standard Windows 10 Server with two Intel (R) Xeon (R) Gold 6230 CPUs 2.1 GHz and 128 GB RAM, MATLAB R2020a.

• MSRC-v5(MSRC) (Winn and Jojic 2005) includes 7 types of objects with a total of 210 images. We selected CM, HOG, GIST, LBP, and CENT features as five different views. • HandWritten (Asuncion, Newman et al. 2007) includes 10 digits with a total of 2,000 images. We selected FOU, FAC, ZER, and MOR features as four different views.

Table 1: Multi-view Datasets   

<html><body><table><tr><td>Datasets</td><td># of samples</td><td># of views</td><td># of classes</td><td>#of features</td></tr><tr><td>MSRC</td><td>210</td><td>5</td><td>7</td><td>24,576,512,256,254</td></tr><tr><td>HandWritten</td><td>2000</td><td>4</td><td>10</td><td>76,216,47,6</td></tr><tr><td>MNIST</td><td>4000</td><td>3</td><td>4</td><td>30,9,30</td></tr><tr><td>Scene</td><td>4485</td><td>3</td><td>15</td><td>1800,1180,1240</td></tr><tr><td>Reuters</td><td>18758</td><td>5</td><td>6</td><td>21531,24892,34251,15506,11547</td></tr></table></body></html>

Table 2: Clustering performance   

<html><body><table><tr><td rowspan="3">Datasets</td><td colspan="3">MSRC</td><td colspan="3">HandWritten</td><td colspan="3">MNIST</td><td colspan="3">Scene</td></tr><tr><td>m</td><td>P</td><td>入</td><td>m</td><td>P</td><td>入</td><td>m</td><td>p</td><td>入</td><td>m</td><td></td><td>入</td></tr><tr><td>Methods</td><td>0.7n ACC</td><td>0.5 NMI</td><td>50 Purity</td><td>n ACC</td><td>0.3 NMI</td><td>1000 Purity</td><td>0.3n ACC</td><td>0.9 NMI</td><td>100 Purity</td><td>0.9n ACC</td><td>0.4 NMI</td><td>500 Purity</td></tr><tr><td>CSMSC</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GMC</td><td>0.862 0.895</td><td>0.767</td><td>0.862 0.895</td><td>0.806</td><td>0.793</td><td>0.867 0.879</td><td>0.643 0.921</td><td>0.601</td><td>0.728 0.921</td><td>0.576 0.409</td><td>0.574 0.430</td><td>0.629 0.417</td></tr><tr><td>ETLMSC</td><td>0.962</td><td>0.809</td><td></td><td>0.879</td><td>0.882</td><td></td><td></td><td>0.807</td><td>0.934</td><td>0.218</td><td>0.166</td><td>0.221</td></tr><tr><td>LMVSC</td><td>0.814</td><td>0.937</td><td>0.962</td><td>0.938</td><td>0.894</td><td>0.938</td><td>0.934</td><td>0.847</td><td>0.892</td><td></td><td></td><td>0.581</td></tr><tr><td>FMCNOF</td><td></td><td>0.717</td><td>0.814</td><td>0.904</td><td>0.831</td><td>0.904</td><td>0.892</td><td>0.726</td><td></td><td>0.561</td><td>0.512 0.249</td><td></td></tr><tr><td>SFMC</td><td>0.713 0.809</td><td>0.648</td><td>0.714</td><td>0.541</td><td>0.484</td><td>0.54</td><td>0.686</td><td>0.513</td><td>0.695</td><td>0.263</td><td>0.522</td><td>0.258 0.346</td></tr><tr><td>FPMVS-CAG</td><td>0.843</td><td>0.721 0.738</td><td>0.781 0.843</td><td>0.853 0.850</td><td>0.871 0.787</td><td>0.873 0.850</td><td>0.917 0.887</td><td>0.801 0.719</td><td>0.917 0.887</td><td>0.343 0.618</td><td>0.566</td><td>0.651</td></tr><tr><td>OMVFC-LICAG</td><td>0.728</td><td>0.590</td><td>0.738</td><td>0.837</td><td>0.798</td><td>0.838</td><td>0.888</td><td>0.725</td><td>0.888</td><td>0.484</td><td>0.517</td><td>0.537</td></tr><tr><td>Orth-NTF</td><td>0.990</td><td>0.978</td><td>0.990</td><td>0.985</td><td>0.969</td><td>0.985</td><td>0.977</td><td>0.926</td><td>0.977</td><td>0.758</td><td>0.804</td><td>0.759</td></tr><tr><td>TLL-AG</td><td>0.986</td><td>0.968</td><td>0.986</td><td>0.982</td><td>0.962</td><td>0.982</td><td>0.981</td><td>0.935</td><td>0.981</td><td>0.719</td><td>0.785</td><td>0.752</td></tr><tr><td>TLLFFC</td><td>1</td><td>1</td><td>1</td><td>0.999</td><td>0.996</td><td>0.999</td><td>1</td><td>1</td><td>1</td><td>0.840</td><td>0.874</td><td>0.870</td></tr></table></body></html>

<html><body><table><tr><td rowspan="2">Datasets</td><td colspan="3">一 Reuters</td></tr><tr><td>m 0.001n</td><td>p 0.4</td><td>入 10</td></tr><tr><td>Methods</td><td>ACC</td><td>NMI</td><td>Purity</td></tr><tr><td>CSMSC GMC ETLMSC LMVSC FMCNOF</td><td>OM OM 0.589 0.343</td><td>OM OM 0.335</td><td>OM OM 0.615</td></tr></table></body></html>

Table 3: Clustering performance. ’OM’ means out of memory. ’-’ means it takes more than 3 hours

• Mnist4 (Deng 2012) includes four categories of handwritten digits from 0 to 3, totaling 4,000 images. We used ISO, LDA, and NPE features as three different views. • Scene15 (Oliva and Torralba 2001) comprises 15 categories of natural scenes encompassing indoor and outdoor environments. We selected PHOW, LBP, and CENTRIST as three different views. • Retuers (Apt´e, Damerau, and Weiss 1994) consists of document data with six categories, totaling 18,758 samples. We selected five different views for each category, described by different languages and their corresponding translations.

![](images/1894eea33ce62547bff16631983fde6b92692b33ea6d41dbf79ac1f14cabb5d5.jpg)  
Figure 3: Clustering performance w.r.t. $m$ .

Compared Methods: We choose the following 10 state-ofart multi-view clustering algorithms to compare with our proposed methods: CSMSC (Luo et al. 2018), GMC (Wang, Yang, and Liu 2019), ETLMSC (Wu, Lin, and Zha 2019), LMVSC (Kang et al. 2020), FMCNOF (Yang et al. 2020),

![](images/d80b28a1c840af7667811cdc04b838925114ab2a14adc602365f02417f9e411f.jpg)  
Figure 4: Clustering performance w.r.t. $p$ .

SFMC (Li et al. 2020), FPMVS-CAG (Wang et al. 2021), Orth-NTF (Li et al. 2023), OMVFC-LICAG (Zhang et al. 2024), TLL-AG (Li et al. 2024).

Metrics: We employ three criteria to comprehensively measure the clustering quality including ACCuracy (ACC), Normalized Mutual Information (NMI) and Purity. These criteria measure the consistency between real labels and predicted labels from different viewpoints. They are between 0 and 1, and the larger they are, the better the performance is.

# Experimental Results

The experimental results are delineated in Tables 2 and 3 along with the hyper-parameters setting on five datasets, the best results are highlighted in boldface. From these results, we found that our TLLFFC significantly competing methods across all datasets, i.e.,

(1) Compared with CSMSC, GMC, and ETLMSC, which have higher time and space complexity leading to time and memory overload when processing large-scale datasets, TLLFFC utilizes an anchor strategy to reduce computational demands. This enables it to efficiently manage and produce impressive results on large-scale datasets.

(2) Compared with discrete and fuzzy anchor graph methods, LMVSC, FMCNOF, SFMC, FPMVS, OMVFC-LICAG, where, LMVSC and FPMVS are two-step methods, TLLFFC performs better. As TLLFFC with the tensor constraint can extract complementary information from different perspectives, ensuring that the cluster indicator matrices across various views align as closely as possible. Besides, TLLFFC can obtain fuzzy clustering results with one-step.

(3) For Orth-NTF, TLL-AG and our method TLLFFC, which have tensor constraints, it is clear that these are the first three best methods, which demonstrate the necessity of the tensor constraint further. However, TLLFFC is better among them. As Orth-NTF and TLL-AG are NMF-based methods, our proposed fuzzy method TLLFFC with label transmission strategy can mine the intrinsic clustering structure between samples and anchors.

![](images/4b0755658e23d0056c93a70a6cbea4f04fa4b30c5ceb1c66501dd0c4764001d1.jpg)  
Figure 5: Clustering performance w.r.t. λ.

# Parameter Analysis

• parameter $m$ : We investigate the clustering performance w.r.t. the number of anchors $m$ , the results are illustrated in Figure 3. The abscissa is the anchor rate, the ratio of anchors to the total number of samples, varies from 0.1 to 1. Form it, we know that the clustering performance does not exhibit a consistent upward trend with increasing anchor rates. The quality of the anchor graph influences the clustering performance. Therefore, the construction of the anchor graph is critical.

• parameter $p \colon p$ varies from 0.1 to 1, and the corresponding results are depicted in Figure 4. Form it, it is clear that TLLFFC is insensitive to $p$ . And the model performs better when $p \ < \ 1$ than $p = 1$ . As when $p \ < \ 1$ , the Schatten $p$ -norm ensures that the tangent plane of the tensor has a spatially low-rank structure, which can mine of complementary information among multiple views better.

• parameter $\lambda$ : We tune $\lambda$ over the set $\{ 0 . 1 , 1 , 1 0 , 5 0 , 1 0 0 \}$ , 500, 1000, 5000, $1 0 0 0 0 \}$ . Figure 5 shows that $\lambda$ significantly affects the clustering results. Selecting extremely high or low values can degrade the clustering quality.

# Convergence Analysis

Figure 6 gives the convergence curves with the clustering performance of TLLFFC. We monitor the changes in reconstruction errors (i.e., $\| { \mathcal { H } } - { \mathcal { I } } \| _ { \infty } )$ over increasing iterations. Notably, the errors exhibit significant fluctuations during the initial 80 iterations but stabilize. Our model typically achieves convergence after 100 iterations. At the same time, within the first 80 iterations, ACC fluctuates at lower levels owing to the fact that the model has not reached convergence at this time.

Table 4: The result of ablation study   

<html><body><table><tr><td>Datasets</td><td colspan="3">MSRC</td><td colspan="3">HandWritten</td><td colspan="3">MNIST</td><td colspan="3">scene</td></tr><tr><td>Methods</td><td>ACC</td><td>NMI</td><td>Purity</td><td>ACC</td><td>NMI</td><td>Purity</td><td>ACC</td><td>NMI</td><td>Purity</td><td>ACC</td><td>NMI</td><td>Purity</td></tr><tr><td>w.o.H</td><td>0.742</td><td>0.654</td><td>0.776</td><td>0.487</td><td>0.578</td><td>0.584</td><td>0.612</td><td>0.552</td><td>0.702</td><td>0.470</td><td>0.502</td><td>0.526</td></tr><tr><td>TLLFFC</td><td>1</td><td>1</td><td>1</td><td>0.999</td><td>0.996</td><td>0.999</td><td>1</td><td>1</td><td>1</td><td>0.840</td><td>0.874</td><td>0.870</td></tr></table></body></html>

![](images/dc8f80917d08f595d9a5c815c74aa4186bf17be3407179e3728cc265c5207e13.jpg)  
Figure 6: Clustering performance and convergence curves.

At about 100 iterations, ACC peaks and remains relatively stable.

# Ablation Study

To examine the significance of the tensor regularization term, we perform ablation studies by omitting the Schatten $p$ -norms. The outcomes are presented in Table 4. Observations reveal that relying solely on fusion of independent view results results in significantly poorer clustering performance, with ACC over $2 5 \%$ below that of our proposed TLLFFC. Incorporating Schatten $p$ -norms enhances the clustering outcome, with the term having a more pronounced impact. As utilizing tensors uncovers complementary information across views, yielding more precise clustering labels.

# T-SNE Visualization

We show the T-SNE visualization results on the Handwritten dataset in Figure 7, with the number of iterations varies from 10 to 120. When the number of iterations is small, the majority of the samples belong to the same cluster, indicating poor clustering performance. However, as the number of iterations increases, the distinction between clusters gradually becomes more apparent. Samples belonging to the same cluster gather together, and the clustering accuracy steadily improves. At the 120th iteration, the model has converged, dividing the pieces into 10 distinct clusters. The clustering accuracy reaches 0.999, which provides further evidence of the model’s effectiveness.

![](images/d1c06914efc1667e845a14d5d98a2cc5b4f0c7b05c80556908dea5f195813083.jpg)  
Figure 7: T-SNE visualization results on the Handwritten dataset with different numbers of iterations.

# Conclusions

This study proposes a novel fuzzy label learning model with multi-view graph clustering. To be specific, the proposed novel balanced regularization term can reduce the pressure of tuning more regularization parameters for fuzzy clustering. Introducing the label transmission and anchor graph strategies can deal with large-scale datasets. Introducing the tensor Schatten $p$ -norm minimization, the complementary information and spatial structure embedded in data with different types can be well utilized to make the clustering results more reliable. Comprehensive experiments verify the superiority of TLLFFC.