# Stability-based Generalization Analysis of Randomized Coordinate Descent for Pairwise Learning

Liang ${ \bf W } { \bf u } ^ { 1 , 2 }$ , Ruixi $\mathbf { H } \mathbf { u } ^ { 1 * }$ , Yunwen Lei3

1Center of Statistical Research, School of Statistics, Southwestern University of Finance and Economics, Chengdu, China   
2Big Data Laboratory on Financial Security and Behavior, SWUFE (Laboratory of Philosophy and Social Sciences, Ministry of Education), Chengdu, China 3Department of Mathematics, University of Hong Kong, Pok Fu Lam, Hong Kong wuliang $@$ swufe.edu.cn, hu ruixi $@$ 163.com, leiyw $@$ hku.hk

# Abstract

Pairwise learning includes various machine learning tasks, with ranking and metric learning serving as the primary representatives. While randomized coordinate descent (RCD) is popular in various learning problems, there is much less theoretical analysis on the generalization behavior of models trained by RCD, especially under the pairwise learning framework. In this paper, we consider the generalization of RCD for pairwise learning. We measure the on-average argument stability for both convex and strongly convex objective functions, based on which we develop generalization bounds in expectation. The early-stopping strategy is adopted to quantify the balance between estimation and optimization. Our analysis further incorporates the low-noise setting into the excess risk bound to achieve the optimistic bound as $O ( 1 / n )$ , where $n$ is the sample size.

# Introduction

The paradigm of pairwise learning has found wide applications in machine learning. Several popular examples are shown as the following. In ranking, we aim to find a model that can predict the ordering of instances (Cle´men¸con, Lugosi, and Vayatis 2008; Rejchel 2012). In metric learning, we wish to build a model to measure the distance between instances (Cao, Guo, and Ying 2016; Ye, Zhan, and Jiang 2019; Dong et al. 2020). Besides, various problems such as AUC maximization (Cortes and Mohri 2003; Gao et al. 2013; Ying, Wen, and Lyu 2016; Liu et al. 2018) and learning tasks with minimum error entropy (Hu et al. 2015) can also be formulated as this paradigm. For all these pairwise learning tasks, the performance of models needs to be measured on pairs of instances. In contrast to pointwise learning, this paradigm is characterized by pairwise loss functions $f : \mathcal { H } \times \mathcal { \bar { Z } } \times \mathcal { \bar { Z } } \mapsto \mathbb { R }$ , where $\mathcal { H }$ and $\mathcal { Z }$ denote the hypothesis space and the sample space respectively. To understand and apply the paradigm better, there is a growing interest in the study under the uniform framework of pairwise learning.

Randomized coordinate descent (RCD) is one of the most commonly used first-order methods in optimization. In each iteration, RCD updates a randomly chosen coordinate along the negative direction of the gradient and keeps other coordinates unchanged. This makes RCD especially effective for large-scale problems (Nesterov 2012), where the computational cost is rather hard to handle.

The extensive applications of RCD have motivated some interesting theoretical analysis on its empirical behavior (Nesterov 2012; Richta´rik and Taka´cˇ 2014; Beck and Teboulle 2021; Chen, Li, and Lu 2023), which focuses on iteration complexities and empirical risks in the optimization process. However, there is much less work considering the generalization performance of RCD, i.e., how models trained by RCD would behave on testing samples. It is notable that the relative analysis only considers the case of pointwise learning (Wang, Wu, and Lei 2021), which is different from pairwise learning in the structure of loss functions. Besides, this work fails to establish the generalization bound based on the $\ell _ { 2 }$ on-average argument stability in a strongly convex case. Therefore, the existing theoretical analysis of RCD is not enough to describe the discrepancy between training and testing for pairwise learning. How to quantify the balance between statistics and optimization under this setting still remains a challenge. In this paper, we develop a more systematical and fine-grained generalization analysis of RCD for pairwise learning to refine the above study. Our analysis can lead to a more appropriate design for the optimization algorithm and the machine learning model.

In this paper, we present the generalization analysis based on the concept of algorithmic stability (Bousquet and Elisseeff 2002). The comparison between the existing work and this paper is presented in Table 1. Our contributions are summarized as follows.

1. Under general assumptions on $L$ -smoothness of loss functions, coordinate-wise smoothness and convexity of objective functions, we study the $\ell _ { 2 }$ on-average argument stability and the corresponding generalization bounds of RCD for pairwise learning. To achieve optimal performance, we consider the balance between the generalization error and the optimization error. The result shows that the early stopping strategy is beneficial to the generalization. The excess risk bounds enjoy the order of $\mathcal { \bar { O } } ( 1 / \sqrt { n } )$ and $O ( \sqrt { \log ( n ) } / n )$ for convex and strongly-convex objective functions respectively, where $n$ denotes the sample size and $\sigma$ is the strong-convexity parameter.

2. We use the low noise condition $F ( \mathbf { w } ^ { * } ) = O ( 1 / n )$ to develop shaper generalization bounds under the convex case. This motivates the excess risk bound $O ( 1 / n )$ , which matches the approximate optimal rate under the strongly convex case. However, we should note that the approximate optimal rate is accessible with a faster computing $T \asymp \log ( \bar { n } )$ for strongly convex empirical risks.

Table 1: All the above convergence rates are based on excess risk bounds in expectation. C meas the convexity as Assumption 4 and SC means the strong convexity as Assumption 5. $G$ -Lip refers to Assumption 1, $L { - } S$ refers to Assumption 2 and Lip-grad refers to Assumption 3. Furthermore, Lei (2020) refers to Lei and Ying (2020), Lei (2021) refers to Lei, Liu, and Ying (2021).   

<html><body><table><tr><td>LearningParadigm</td><td>Algorithm</td><td>Reference</td><td>Assumption</td><td>Noise Seeting</td><td>Iteration</td><td>Rate</td></tr><tr><td rowspan="5">Pointwise Learning</td><td rowspan="6">SGD</td><td rowspan="3">Lei (2020)</td><td>C, L-S</td><td></td><td rowspan="5">T×n</td><td>0(1/√n) 0(1/n)</td></tr><tr><td>C, G-Lip, L-S</td><td>F(w*)=0</td><td></td></tr><tr><td>SC, G-Lip, L-S</td><td></td><td>0(1/√n)</td></tr><tr><td rowspan="2">C, L-S</td><td rowspan="2"></td><td>0(1/no)</td></tr><tr><td>0(1/√n)</td></tr><tr><td rowspan="6">Pairwise Learning</td><td rowspan="3">Lei (2021)</td><td></td><td>F(w*)=0(1/n)</td><td></td><td>0(1/n)</td></tr><tr><td>C, G-Lip SC, L-S</td><td rowspan="4"></td><td>T×n² T×n</td><td>0(1/√n)</td></tr><tr><td>SC, G-Lip</td><td></td><td></td><td>0(1/no)</td></tr><tr><td rowspan="2"></td><td rowspan="2">C, L-S,Lip-grad</td><td></td><td>T×n² T×√n</td><td>0(1/no)</td></tr><tr><td></td><td></td><td>0(1/√n)</td></tr><tr><td></td><td>RCD</td><td>This Work</td><td> SC, L-S, Lip-grad</td><td>F(w*)=0(1/n)</td><td>T×n T× log(n)</td><td>0(1/n) O(√log(n)/n)</td></tr></table></body></html>

The main work is organized according to the convexity of the empirical risk. We consider the on-average argument stability and develop the corresponding excess risk bounds. The early-stopping strategy is useful for balancing optimization and estimation, by which we present the optimal convergence rate. Furthermore, there are two key points about our proof in comparison with the pointwise case (Wang, Wu, and Lei 2021): One is applying the coercivity property to bound the expansiveness of RCD updates since the expectation of randomized coordinates leads to the gradient descent operator. The other is following the optimization error bounds for pointwise learning directly since they both use unbiased gradient estimations.

# Related Work

In this section, we review the related work on RCD and generalization analysis for pairwise learning.

Randomized Coordinate Descent (RCD). The realworld performance of RCD has demonstrated its significant efficiency in many large-scale optimization tasks, including regularized risk minimization (Chang, Hsieh, and Lin 2008; Shalev-Shwartz and Tewari 2009), low-rank matrix completion and learning (Hu and Kwok 2019; Callahan, Vu, and Raich 2024), and optimal transport problems (Xie, Wang, and Zhang 2024). The convergence analysis of RCD and its accelerated variant was first proposed by Nesterov (2012), where global estimates of the convergence rate were considered. Then the strategies to accelerate RCD were further explored (Richta´rik and Taka´cˇ 2014), for which the corresponding convergence properties were established for structural optimization problems (Zhao et al. 2014; Lu and Xiao 2015, 2017). RCD was also studied under various settings including nonconvex optimization (Beck and Teboulle 2021;

Chen, Li, and Lu 2023), volume sampling (Rodomanov and Kropotov 2020) and differential privacy (Damaskinos et al. 2021). The above study mainly considered the empirical behavior of RCD. However, the aim of this paper is to quantify the generalization performance of machine learning models trained by RCD.

Generalization for Pairwise Learning. The generalization ability shows how models based on training datasets will adapt to testing datasets. It serves as an important indicator for the enhancement of models and algorithms in the view of statistical learning theory (SLT). To investigate the generalization performance for pairwise learning, methods of uniform convergence analysis and stability analysis have been applied under this wide learning framework. More details are described below.

The uniform convergence approach considers the connection between generalization errors and U-statistics, from which generalization bounds via corresponding U-processes are developed sufficiently. Complexity measures including VC dimension (Vapnik, Levin, and Le Cun 1994), covering numbers (Zhou 2002) and Rademacher complexities (Bartlett and Mendelson 2001) for the hypothesis space play a key role in this approach. For pairwise learning, these measures have been used for studying the generalization of specific tasks such as ranking (Cle´menc¸on, Lugosi, and Vayatis 2008; Rejchel 2012) and metric learning (Cao, Guo, and Ying 2016; Ye, Zhan, and Jiang 2019; Dong et al. 2020). Recently, some work also explored the generalization of deep networks with these tasks (Huang et al. 2023; Zhou, Wang, and Zhou 2024). Furthermore, generalizations for the pairwise learning framework were studied under various settings, including PL condition (Lei, Liu, and Ying 2021), regularized risk minimization (Lei, Ledent, and Kloft 2020) and online learning (Wang et al. 2012; Kar et al. 2013). As compared to the stability analysis, the complexity analysis enjoys the ability of yielding generalization bounds for non-convex objective functions (Mei, Bai, and Montanari 2018; Davis and Drusvyatskiy 2022). However, generalization bounds yielded by the uniform convergence approach are inevitably associated with input dimensions (Agarwal and Niyogi 2009; Feldman 2016; Schliserman, Sherman, and Koren 2024), which can be avoided in the stability analysis.

Algorithmic stability serves as an important concept in SLT, which is closely related to learnability and consistency (Feldman 2016; Rakhlin, Mukherjee, and Poggio 2005). The basic framework for stability analysis was proposed by Bousquet and Elisseeff (2002), where the concept of uniform stability was introduced and then extended to study randomized algorithms (Elisseeff et al. 2005). The power of algorithmic stability for generalization analysis further inspired several other stability measures including uniform argument stability (Liu et al. 2017), on-average loss stability (Shalev-Shwartz et al. 2010; Lei, Ledent, and Kloft 2020; Lei, Liu, and Ying 2021), on-average argument stability (Lei and Ying 2020; Deora et al. 2024), locally elastic stability (Deng, He, and Su 2021; Lei, Sun, and Liu 2023) and Bayes stability (Li, Luo, and Qiao 2020). While various stability measures were useful for deriving generalization bounds in expectation, applications of uniform stability implied elegant high-probability generalization bounds (Feldman and Vondrak 2019; Bousquet, Klochkov, and Zhivotovskiy 2020; Klochkov and Zhivotovskiy 2021). Furthermore, the stability analysis promoted the study for the generalization of stochastic gradient descent (SGD) effectively (Deng et al. 2023), which was considered under the paradigm of pairwise learning (Lei, Ledent, and Kloft 2020; Lei, Liu, and Ying 2021) or pointwise and pairwise learning (Wang et al. 2023; Chen et al. 2023). In contrast to SGD, a more sufficient generalization analysis of RCD is needed under the framework of pairwise learning. It provides us guidelines to apply RCD in large-scale optimization problems for pairwise learning.

Other than the approach based on uniform convergence or algorithmic stability, the generalization for pairwise learning was also studied from the perspective of algorithmic robustness (Bellet and Habrard 2015; Christmann and Zhou 2016), convex analysis (Ying and Zhou 2016), integral operators (Fan et al. 2016; Guo et al. 2017) and information theoretical analysis (Dong et al. 2024).

# Preliminaries

Let $S = \{ z _ { 1 } , . . . , z _ { n } \}$ be a set drawn independently from a probability measure $\rho$ defined over a sample space $\mathcal { Z } =$ $\mathcal { X } \times \mathcal { Y }$ , where $\chi$ is an input space and $\mathcal { y } \subset \mathbb { R }$ is an output space. For pairwise learning, our aim is to build a model $h : \mathcal { X } \mapsto \mathbb { R }$ or $h : \mathcal { X } \times \mathcal { X } \mapsto \mathbb { R }$ to simulate the potential mapping lying on $\rho$ . We further assume that the model is parameterized as $h _ { \mathbf { w } }$ and the vector w belongs to a parameter space $\mathcal { W } \subseteq \mathbb { R } ^ { d }$ . As the essential feature of pairwise learning, the nonnegative loss function takes the form of $f : \mathcal { W } \times \mathcal { Z } \times \mathcal { Z } \mapsto \mathbb { R }$ . Since ranking and metric learning are the most popular applications of pairwise learning, we take them as examples here to show how the learning framework involves various learning tasks. Besides, we present details of AUC maximization below, which is used as the experimental validation for our results.

Example 1. (Ranking). Ranking models usually take the form of $h _ { \mathbf { w } } : \mathcal { X } \mapsto \mathbb { R }$ . Given two instances $z = ( \dot { x } , y ) , z ^ { \prime } =$ $( x ^ { \prime } , y ^ { \prime } )$ , we adopt the ordering of $h _ { \mathbf { w } } ( x ) , h _ { \mathbf { w } } ( x ^ { \prime } )$ as the prediction of the ordering for $y , y ^ { \prime }$ . As a result, the prediction $h _ { \mathbf { w } } ( x ) - h _ { \mathbf { w } } ( x ^ { \prime } )$ and the true ordering $s g n ( y - y ^ { \prime } )$ jointly formulate the approach to measure the performance of models. The loss function in this problem is further defined as the pairwise formulation of $f ( \mathbf { w } ; z , z ^ { \prime } ) ~ = ~ \phi ( s g n ( y ~ -$ $y ^ { \prime } ) ( h _ { \bf w } \dot { ( } x ) - h _ { \bf w } ( x ^ { \prime } ) ) )$ . Here we can choose the logistic loss $\phi ( t ) = \log ( 1 + \exp ( - t ) )$ or the hinge loss $\phi ( t ) = $ max $\{ 1 - t , 0 \}$ .

Example 2. (Supervised metric learning). For this problem with output space as $\mathcal { V } = \{ + 1 , - 1 \}$ , the most usual aim is to learn a Mahalanobis metric $d _ { \mathbf { w } } ( x , \acute { x } ^ { \prime } ) = ( x - x ^ { \prime } ) ^ { \top } \mathbf { w } ( x - x ^ { \prime } )$ . Under the parameter $\mathbf { w } \in \mathbb { R } ^ { d \times d }$ and the corresponding metric, we hope that the distance metric between two instances is consistent with the similarity of labels. Let $\phi$ be the logistic or the hinge loss defined in Example 1. We can formulate this metric learning problem under the framework of pairwise learning by the loss function as $f ( \mathbf { w } ; z , z ^ { \prime } ) =$ $\phi ( \bar { \tau } ( y , y ^ { \prime } ) d _ { \mathbf { w } } ( x , x ^ { \prime } ) )$ , where $\tau ( y , y ^ { \prime } ) ~ = ~ 1$ if $y = y ^ { \prime }$ and $\tau ( y , y ^ { \prime } ) = - 1$ if $y \ne y ^ { \prime }$ .

Example 3. (AUC Maximization). AUC score is widely applied to measure the performance of classification models for imbalanced data. With the binary output space $y =$ $\{ + 1 , - 1 \}$ , it shows the probability that the model $h _ { \mathbf { w } } : \mathcal { X } \mapsto$ $\mathbb { R }$ scores a positive instance higher than a negative instance. Therefore, the loss function for AUC maximization usually takes the form of f (w; z, z′) = g(w⊤(x − x′))I[y=1,y′= 1], where $g$ can be chosen in the same way as $\phi$ in Example 1 and I denotes the indicator function. This demonstrates that AUC maximization also falls into the framework of pairwise learning.

With the pairwise loss function, the population risk is defined as the following

$$
F ( \mathbf { w } ) = \mathbb { E } _ { z _ { i } , z _ { j } \sim \rho } \left[ f ( \mathbf { w } ; z _ { i } , z _ { j } ) \right] ,
$$

which can measure the performance of $h _ { \mathbf { w } }$ in real applications. Since $\rho$ is unknown, we consider the empirical risk

$$
F _ { S } ( \mathbf { w } ) = \frac { 1 } { n ( n - 1 ) } \sum _ { \substack { i , j \in [ n ] : i \neq j } } f ( \mathbf { w } ; z _ { i } , z _ { j } ) ,
$$

where $[ n ] : = \{ 1 , \dots , n \}$ . Let $\mathbf { w } ^ { * } = \arg \operatorname* { m i n } _ { \mathbf { w } \in \mathcal { W } } F ( \mathbf { w } )$ and $\mathbf { w } _ { S } = \arg \operatorname* { m i n } _ { \mathbf { w } \in \mathcal { W } } F _ { S } ( \mathbf { w } )$ . To approximate the best model $h _ { \mathbf { w } ^ { * } }$ , we apply a randomized algorithm $A$ to the training dataset $S$ and get a corresponding output model. We then use $A ( S )$ to denote the parameter of the output model.

Comparing the acquired parameter $A ( S )$ and the best parameter $\mathbf { w } ^ { * }$ , the excess risk $F ( A ( S ) ) - { \dot { F } } ( \mathbf { w } ^ { * } )$ can quantify the performance of $A ( S )$ appropriately. We are interested in bounding the excess risk to provide theoretical supports for the practice of learning tasks. To study the risk adequately, we introduce the following decomposition

$$
\begin{array} { r } { F ( A ( S ) ) - F ( { \mathbf w } ^ { * } ) = \left[ F ( A ( S ) ) - F ( { \mathbf w } ^ { * } ) \right] - \left[ F _ { S } ( A ( S ) ) \right. } \\ { \left. - F _ { S } ( { \mathbf w } ^ { * } ) \right] + \left[ F _ { S } ( A ( S ) ) - F _ { S } ( { \mathbf w } ^ { * } ) \right] . \qquad ( 1 ) } \end{array}
$$

Taking expectation on both sides of the above equation and noting $\mathbb { E } _ { \boldsymbol { S } } \mathbf { \bar { \Pi } } [ F _ { S } ( \mathbf { w } ^ { * } ) ] = F ( \mathbf { w } ^ { * } )$ , we further decompose the

excess risk as

$$
\begin{array} { r l } & { \mathbb { E } _ { S , A } \big [ F ( A ( S ) ) - F ( { \mathbf w } ^ { * } ) \big ] = \mathbb { E } _ { S , A } \left[ F ( A ( S ) ) - F _ { S } ( A ( S ) ) \right] } \\ & { \qquad + \mathbb { E } _ { S , A } \left[ F _ { S } ( A ( S ) ) - F _ { S } ( { \mathbf w } ^ { * } ) \right] . \qquad ( 2 ) } \end{array}
$$

The first and the second term on the right-hand side are referred to as estimation error (generalization gap) and optimization error respectively. We incorporate SLT and optimization theory to control the two errors, respectively.

In this paper, we consider the learning framework below, which combines RCD and pairwise learning.

Definition 1. (RCD for pairwise learning). Let $\mathbf { w } _ { 1 } \in \mathcal { W }$ be the initial point and $\{ \eta _ { t } \}$ be a nonnegative stepsize sequence. At the $t$ -th iteration, we first draw $i _ { t }$ from the discrete uniform distribution over $\{ 1 , \ldots , d \}$ and then update along the $i _ { t }$ -th coordinate as

$$
\begin{array} { r } { \mathbf { w } _ { t + 1 } = \mathbf { w } _ { t } - \eta _ { t } \nabla _ { i _ { t } } F _ { S } ( \mathbf { w } _ { t } ) \mathbf { e } _ { i _ { t } } , } \end{array}
$$

where $\nabla _ { i _ { t } } F _ { S } ( \mathbf { w } _ { t } )$ denotes the gradient of the empirical risk w.r.t. to the $i _ { t }$ -th coordinate and $\mathbf { e } _ { i _ { t } }$ is a vector with the $i _ { t }$ -th coordinate being 1 and other coordinates being 0.

Considering the generalization for the above paradigm, we leverage the concept of algorithmic stability to handle the estimation error. Algorithmic stability shows how algorithms react to perturbations of training datasets. Various stability measures have been proposed to study the generalization gap in SLT, including uniform stability, argument stability and on-average stability. Here we introduce the uniform stability and the on-average argument stability, with the latter being particularly useful for generalization analysis in this paper. It is notable that we follow Lei and Ying (2020) in the definition of $\ell _ { 1 }$ and $\ell _ { 2 }$ on-average argument stabilities. The $\ell _ { 1 }$ on-average argument stability refers to the $\ell _ { 1 }$ -norm of the vector $( \| \\b { \dot { A } } ( S ) ^ { - } - A ( S _ { 1 } ) \| _ { 2 } , \dots , \| A ( S ) - A ( S _ { n } ) \| _ { 2 } )$ , while the $\ell _ { 2 }$ on-average argument stability refers to the $\ell _ { 2 }$ - norm of this vector.

Definition 2. (Algorithmic Stability). Drawing independently from $\rho$ , we get the following two datasets

$$
S = \{ z _ { 1 } , \dots , z _ { n } \} \qquad { a n d } \qquad S ^ { \prime } = \{ z _ { 1 } ^ { \prime } , \dots , z _ { n } ^ { \prime } \} .
$$

We then replace $z _ { i }$ in $S$ with $z _ { i } ^ { \prime }$ for any $i \in [ n ]$ and have

$$
S _ { i } = \{ z _ { 1 } , \ldots , z _ { i - 1 } , z _ { i } ^ { \prime } , z _ { i + 1 } , \ldots , z _ { n } \} .
$$

Let $\mathbf { x } \in \mathbb { R } ^ { d }$ be a vector of dimension $d$ . Then we denote the $p$ -norm∈ $\begin{array} { r } { \| \mathbf { x } \| _ { p } = ( \sum _ { i = 1 } ^ { d } | \mathbf { x } _ { i } | ^ { p } ) ^ { 1 / p } } \end{array}$ and show several stability measures below.

(a) Randomized algorithm $A$ is $\epsilon$ -uniformly stable if for any $S , S _ { i } \in \mathcal { Z } ^ { n }$ the following inequality holds

$$
\operatorname* { s u p } _ { z , \tilde { z } } \left[ f ( A ( S ) , z , \tilde { z } ) - f ( A ( S _ { i } ) , z , \tilde { z } ) \right] \leq \epsilon .
$$

(b) We say $A$ is $\ell _ { 1 }$ on-average argument $\epsilon$ -stable if

$$
\mathbb { E } _ { S , S ^ { \prime } , A } \Big [ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \| A ( S ) - A ( S _ { i } ) \| _ { 2 } \Big ] \leq \epsilon .
$$

(c) We say $A$ is $\ell _ { 2 }$ on-average argument $\epsilon$ -stable if

$$
\mathbb { E } _ { S , S ^ { \prime } , A } \Big [ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \| A ( S ) - A ( S _ { i } ) \| _ { 2 } ^ { 2 } \Big ] \leq \epsilon ^ { 2 } .
$$

As indicated below, We prepare several necessary assumptions so that relative generalization bounds can be derived effectively. Assumption 1 and Assumption 2 are useful for bounding the on-average argument stability. Assumption 3 is mainly applied in the proof of the optimization error. The other two assumptions show the convexity of the empirical risk, which is the basic condition for the establishment of relative theorems.

Assumption 1. For all $( z , z ^ { \prime } ) \in \mathcal { Z } \times \mathcal { Z }$ and $\mathbf { w } \in \mathcal { W }$ , the loss function satisfies the $G$ -Lipschitz continuity condition as $\| \nabla f ( \mathbf { w } , z , z ^ { \prime } ) \| _ { 2 } \leq G$ .

Assumption 2. For all $( z , z ^ { \prime } ) ~ \in ~ \mathcal { Z } \times \mathcal { Z }$ and $\mathbf { w } , \mathbf { w } ^ { \prime } \in$ $\mathcal { W }$ , the loss function is $L$ -smooth as $\| \nabla f ( \mathbf { w } ; z , z ^ { \prime } ) \ -$ $\nabla f ( \mathbf { w } ^ { \prime } ; z , z ^ { \prime } ) \| _ { 2 } \leq L \| \mathbf { w } - \mathbf { w } ^ { \prime } \| _ { 2 }$ .

Assumption 3. For any $S , F _ { S }$ has coordinate-wise Lipschitz continuous gradients with parameter $\widetilde L > 0$ , i.e., we have the following inequality for all $\alpha \in \mathbb { R }$ $\mathbf { w } \in { \mathcal { W } }$ , $i \in [ d ]$

$$
F _ { S } ( \mathbf { w } + \alpha \mathbf { e } _ { i } ) \leq F _ { S } ( \mathbf { w } ) + \alpha \nabla _ { i } F _ { S } ( \mathbf { w } ) + \widetilde L \alpha ^ { 2 } / 2 .
$$

Assumption 4. $F _ { S }$ is convex for any $S$ , i.e., $F _ { S } ( \mathbf { w } ) \mathbf { \Sigma } - \mathbf { \Sigma }$ $F _ { S } ( \mathbf { w } ^ { \prime } ) \geq \langle \mathbf { w } - \mathbf { w } ^ { \prime } , \nabla F _ { S } ( \mathbf { w } ^ { \prime } ) \rangle$ holds for all $\pmb { w } , \pmb { w } ^ { \prime } \in \mathcal { W }$ .

Assumption 5. $F _ { S }$ is $\sigma$ -strongly convex for any $S$ , i.e., the following inequality holds for all $\boldsymbol { \ w } , \boldsymbol { \mathbf { w } } ^ { \prime } \in \mathcal { W }$

$$
F _ { S } ( \mathbf { w } ) - F _ { S } ( \mathbf { w } ^ { \prime } ) \geq \langle \mathbf { w } - \mathbf { w } ^ { \prime } , \nabla F _ { S } ( \mathbf { w } ^ { \prime } ) \rangle + \sigma \| \mathbf { w } - \mathbf { w } ^ { \prime } \| _ { 2 } ^ { 2 } / 2 ,
$$

where $\langle \cdot , \cdot \rangle$ denotes the inner product of two vectors.

With Definition 1 and Definition 2, we can further quantify stabilities of RCD for pairwise learning. Then we show connections between the estimation error and stability measures by the following lemma, which is the key to apply algorithmic stability effectively in generalization analysis. While part (a) of Lemma 1 is motivated by the case of pointwise learning (Hardt, Recht, and Singer 2016) and derived with the technique similar to Lei, Liu, and Ying (2021), part (b) and part (c) are introduced from Lei, Liu, and Ying (2021) and Lei, Ledent, and Kloft (2020) respectively. In part (c), the base of the natural logarithm takes the symbol as $e$ and $\lceil \alpha \rceil$ means rounding up for $\alpha$ .

Lemma 1. Let $S , S _ { i }$ be constructed as Definition 2. Then we bound estimation errors with stability measures below. (a) Let Assumption $\jmath$ hold. Then the estimation error can be bounded by the $\ell _ { 1 }$ on-average argument stability below

$$
\begin{array} { r l } {  { \mathbb { E } _ { S , A } [ F ( A ( S ) ) - F _ { S } ( A ( S ) ) ] } \quad } & { } \\ & { \leq \frac { 2 G } { n } \displaystyle \sum _ { i = 1 } ^ { n } \mathbb { E } _ { S , S ^ { \prime } , A } [ \| A ( S _ { i } ) - A ( S ) \| _ { 2 } ] . } \end{array}
$$

(b) Let Assumption 2 hold. Then for any $\gamma > 0$ we have the following estimation error bound with the $\ell _ { 2 }$ on-average argument stability

$$
\begin{array} { r l r } & { } & { \displaystyle \mathbb { E } _ { S , A } \left[ F ( A ( S ) ) - F _ { S } ( A ( S ) ) \right] \leq \frac { L } { \gamma } \mathbb { E } _ { S , A } \left[ F _ { S } ( A ( S ) ) \right] } \\ & { } & { \displaystyle + \frac { 2 ( L + \gamma ) } { n } \sum _ { i = 1 } ^ { n } \mathbb { E } _ { S , S ^ { \prime } , A } \left[ \| A ( S _ { i } ) - A ( S ) \| _ { 2 } ^ { 2 } \right] . } \end{array}
$$

(c) Let n denote the sample size of $S$ . Assume for any $S$ and $( z , z ^ { \prime } ) \in \mathcal { Z } \times \mathcal { Z }$ , $| f ( A ( S ) ; z , z ^ { \prime } ) | \le R$ holds for $R > 0$ . Suppose that $A$ is $\epsilon$ -uniformly-stable and $\delta \in ( 0 , 1 / e )$ , then the following inequality holds with probability at least $1 - \delta$

$$
\begin{array} { r l } & { | F ( A ( S ) ) - F _ { S } ( A ( S ) ) | \le 4 \epsilon + e \Big ( 1 2 \sqrt { 2 } R \sqrt { \displaystyle \frac { \log ( e / \delta ) } { n - 1 } } } \\ & { ~ + ~ 4 8 \sqrt { 6 } \epsilon \lceil \log _ { 2 } ( n - 1 ) \rceil \log ( e / \delta ) \Big ) . ~ } \end{array}
$$

Remark 1. While estimation error bound (4) is established under the Lipschitz continuity condition, (5) remove this condition based on the $\ell _ { 2 }$ on-average argument stability measure. Inequality (5) holds with the $L$ - smoothness of the loss function, which replaces the Lipschitz constant in (4) by the empirical risk. Furthermore, if $A$ is $\ell _ { 2 }$ on-average argument $\epsilon$ -stable, we can take $\begin{array} { r c l } { \gamma } & { = } & { \sqrt { L \mathbb { E } _ { S , A } \left[ F _ { S } ( A ( S ) ) \right] } / ( \sqrt { 2 } \epsilon ) } \end{array}$ in part (b) and get $\begin{array} { r } { \mathbb { E } _ { S , A } \left[ F ( A ( S ) ) - F _ { S } ( A ( S ) ) \right] \le \sqrt { 2 L \mathbb { E } _ { S , A } \left[ F _ { S } ( A ( S ) ) \right] } \epsilon + } \end{array}$ $2 L \epsilon ^ { 2 }$ . If the empirical risk $\mathbb { E } _ { S , A } \left[ F _ { S } ( A ( S ) ) \right] \ : = \ : O ( 1 / n )$ , then we further know $\begin{array} { r l } { \operatorname { \mathbb { E } } _ { S , A } \left[ F ( A ( S ) ) - F _ { S } ( A ( S ) ) \right] } & { = } \end{array}$ $O ( \epsilon ^ { 2 } + \epsilon / \sqrt { n } )$ , which means the estimation error bound is well dependent on the stability measure $\epsilon$ via the small risk of the output model (Hardt, Recht, and Singer 2016). Other than the generalization error in expectation, the link in high probability (6) presents the convergence rate of $O ( n ^ { - \frac { 1 } { 2 } } + \epsilon \sqrt { \log _ { 2 } ( n ) } )$ for $\epsilon$ -uniformly stable algorithm. This result is achieved by combining a concentration inequality from Bousquet, Klochkov, and Zhivotovskiy (2020) and the decoupling technique in Lei and Ying (2020).

Besides the estimation error, we need to tackle the optimization error to achieve complete excess risk bounds. The optimization error analysis for pointwise learning can be directly extended to pairwise learning since they both use unbiased gradient estimations. Since pointwise learning and pairwise learning mainly differ in terms of loss structure, Lemma 2 from pointwise learning also works for pairwise learning.

Lemma 2. Let $\left\{ \mathbf { w } _ { t } \right\}$ be produced by RCD (3) with nonincreasing step sizes $\eta _ { t } \leq 1 / { \widetilde { L } } .$ . Let Assumptions 3,4 hold, then the following two inequalitie s holds for any w $\mathbf { \Psi } \in \mathcal { W }$

$$
\begin{array} { r } { \mathbb { E } _ { A } [ F _ { S } ( \mathbf { w } _ { t } ) - F _ { S } ( \mathbf { w } ) ] \leq \frac { d \left( \| \mathbf { w } _ { 1 } - \mathbf { w } \| _ { 2 } ^ { 2 } + 2 \eta _ { 1 } F _ { S } ( \mathbf { w } _ { 1 } ) \right) } { 2 \sum _ { j = 1 } ^ { t } \eta _ { j } } } \end{array}
$$

and

$$
\begin{array} { r l r } {  { 2 \sum _ { j = 1 } ^ { t } \eta _ { j } ^ { 2 } \mathbb { E } _ { A } [ F _ { S } ( \mathbf { w } _ { j } ) - F _ { S } ( \mathbf { w } ) ] } } \\ & { } & { \qquad \leq d \eta _ { 1 } \| \mathbf { w } _ { 1 } - \mathbf { w } \| _ { 2 } ^ { 2 } + 2 d \eta _ { 1 } ^ { 2 } F _ { S } ( \mathbf { w } _ { 1 } ) . } \end{array}
$$

Let Assumption $5$ hold and $\begin{array} { r } { \mathbf { w } _ { S } = \arg \operatorname* { m i n } _ { \mathbf { w } \in \mathcal { W } } F _ { S } ( \mathbf { w } ) , } \end{array}$ , then we have the following inequality

$$
\begin{array} { r l } & { \mathbb { E } _ { A } [ F _ { S } ( \mathbf { w } _ { t + 1 } ) - F _ { S } ( \mathbf { w } _ { S } ) ] } \\ & { \qquad \quad \leq ( 1 - \eta _ { t } \sigma / d ) \mathbb { E } _ { A } [ F _ { S } ( \mathbf { w } _ { t } ) - F _ { S } ( \mathbf { w } _ { S } ) ] . } \end{array}
$$

In the arXiv version, Appendix B restates the above two lemmas and prepares some other lemmas. The proof for part (a) of Lemma 1 is given in Appendix B.1. Considering the stability analysis, we introduce the coercivity property of the gradient descent operator in Appendix B.3 (Hardt, Recht, and Singer 2016). Then we show the self-bounding property of $L$ -smooth functions in Appendix B.4 (Srebro, Sridharan, and Tewari 2010), which plays a key role in introducing empirical risks into the $\ell _ { 2 }$ on-average argument stability.

# Main Results

In this section, we show our results on generalization analysis of RCD for pairwise learning. For both convex and strongly convex cases, we derive the on-average argument stability bounds and as well as the corresponding excess risk bounds. Results are organized according to the convexity of the empirical risk.

# Generalization for Convex Case

This subsection describes the $\ell _ { 2 }$ on-average argument stabilities for the convex empirical risk. Based on stability analysis, we consider generalization bounds in expectation under the setting that applies RCD for pairwise learning.

If the empirical risk is convex and $L$ -smooth, then the gradient descent operator enjoys the coercivity property according to Hardt, Recht, and Singer (2016). Since taking expectations for the coordinate descent operator yields the gradient descent operator, the coercivity property is useful to bound the expansiveness of RCD updates in the stability analysis. With the coercivity property of the coordinate descent operator in expectation, we further incorporate the self-bounding property of $L$ -smooth functions to measure the $\ell _ { 2 }$ on-average argument stability. Then we handle the estimation error by plugging the stability measure into part (b) of Lemma 1. We finally introduce the optimization error and derive the corresponding excess risk bound. The proof is given in Appendix C of the arXiv version.

Theorem 3. Let Assumptions 2, 3, 4 hold. Let $\left\{ { \bf w } _ { t } \right\}$ , $\{ \mathbf { w } _ { t } ^ { ( i ) } \}$ be produced by (3) with $\eta _ { t } \leq 1 / L$ based on $S$ and $S _ { i }$ respectively. Then the $\ell _ { 2 }$ on-average argument stability satisfies

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { E } _ { S , S ^ { \prime } , A } \left[ | | \mathbf { w } _ { t + 1 } - \mathbf { w } _ { t + 1 } ^ { ( i ) } | | _ { 2 } ^ { 2 } \right] } \\ & { \displaystyle \leq \frac { 1 2 8 L } { n ^ { 2 } d } ( \frac { t } { d } + 1 ) \sum _ { j = 1 } ^ { t } \eta _ { j } ^ { 2 } \mathbb { E } _ { S , A } [ F _ { S } ( \mathbf { w } _ { j } ) ] . } \end{array}
$$

Assume that the nonincreasing step size sequence $\{ \eta _ { t } \}$ satisfies $\eta _ { t } \leq 1 / \widetilde { L }$ . Then, for any $\gamma \geq 0$ , we have

$$
\begin{array} { r l } & { \mathbb { E } _ { S , A } \left[ F ( \mathbf { w } _ { T } ) - F ( \mathbf { w } ^ { * } ) \right] } \\ & { \quad = O \left( \frac { d ( 1 + L \gamma ^ { - 1 } ) } { \sum _ { t = 1 } ^ { T } { \eta _ { t } } } + \frac { L ( L + \gamma ) ( T + d ) } { { { n } ^ { 2 } } d } \right) } \\ & { \quad + O \left( \frac { L } { \gamma } + \frac { L ( L + \gamma ) ( T + d ) } { { { n } ^ { 2 } } d ^ { 2 } } \sum _ { t = 1 } ^ { T } { \eta _ { t } ^ { 2 } } \right) \times F ( \mathbf { w } ^ { * } ) . } \end{array}
$$

Furthermore, for a constant step size as $\eta _ { t } \equiv \eta ,$ , we choose $T \asymp n ^ { \frac { 1 } { 2 } } d L ^ { - \frac { 1 } { 2 } }$ and get

$$
\mathbb { E } _ { S , A } \left[ F ( \mathbf { w } _ { T } ) - F ( \mathbf { w } ^ { * } ) \right] = O \Big ( \sqrt { \frac { L } { n } } \Big ) .
$$

Assuming that $F ( \mathbf { w } ^ { * } ) = O ( L n ^ { - 1 } )$ , we choose $T \asymp n d L ^ { - 1 }$ to give

$$
\mathbb { E } _ { S , A } \left[ F ( \mathbf { w } _ { T } ) - F ( \mathbf { w } ^ { * } ) \right] = O \left( \frac { L } { n } \right) .
$$

Remark 2. For pairwise learning, Eq. (10) shows that RCD enjoys the $\ell _ { 2 }$ on-average argument stability is of the order of $\begin{array} { r } { O \big ( L ( t + d ) \sum _ { j = 1 } ^ { t } \eta _ { j } ^ { 2 } \mathbb { E } _ { S , A } [ F _ { S } ( \mathbf { w } _ { j } ) ] / ( n ^ { 2 } d ^ { 2 } ) \big ) } \end{array}$ . This bound means that th e output model of RCD becomes more and more stable with the sample size increasing or the number of iterations decreasing. In further detail, since the estimation error can be bounded by the stability bound according to (5), decreasing the number of iterations is beneficial to controlling the estimation error. However, increasing the number of iterations corresponds to the optimization process, which is the key to control the optimization error. As a result, the early stopping strategy is adopted to balance the estimation and optimization for a good generalization.

Remark 3. Fixing $F ( \mathbf { w } ^ { * } )$ , we choose an appropriate number of iterations for the excess risk bound (12). Besides, we incorporate $F ( \mathbf { w } ^ { * } )$ into the excess risk bound and get the convergence rate (13). It is obvious that (13) exploits the low noise setting to yield the optimistic bound (Srebro, Sridharan, and Tewari 2010). Furthermore, since RCD updates in expectation is closely related to the gradient descent operator, we consider the batch setting (Nikolakakis, Karbasi, and Kalogerias 2023) and find that results here are identical to those of full-batch GD (Nikolakakis et al. 2023). Turning to SGD for pairwise learning (Lei, Liu, and Ying 2021), the $\ell _ { 2 }$ on-average argument stability takes a slower rate as $O ( 1 / n )$ under the same setting. The excess risk bound can achieve the rate of $( 1 / { \sqrt { n } } )$ in a general setting with $\eta _ { t } = \eta \asymp 1 / \sqrt { T }$ and $T \asymp n$ . With the low noise setting $F ( \mathbf { w } ^ { * } ) = O ( { \dot { n } } ^ { - 1 } )$ , the optimistic bound $O ( 1 / n )$ is also derived.

# Generalization for Strongly Convex Case

This subsection presents generalization analysis of RCD for pairwise learning in a strongly convex setting. In the strongly convex case of pointwise learning (Wang, Wu, and Lei 2021), the $\ell _ { 2 }$ on-average argument stability and the corresponding generalization bound were not taken into consideration. Therefore, we not only measure the stability here but also derive the excess risk bound for the strongly convex empirical risk. We show the proof in Appendix D of the arXiv version.

Theorem 4. Let Assumptions 2, 3, 5 hold. Let $\left\{ { { \bf { w } } _ { t } } \right\}$ , $\{ \mathbf { w } _ { t } ^ { ( i ) } \}$ be produced by (3) with $\eta _ { t } \leq \beta / L$ for any $\beta \in ( 0 , 1 )$ based on $S$ and $S _ { i }$ , respectively. Then the $\ell _ { 2 }$ on-average argument

stability is

$$
\begin{array} { l } { \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { E } _ { S , S ^ { \prime } , A } \left[ \| \mathbf { w } _ { t + 1 } - \mathbf { w } _ { t + 1 } ^ { ( i ) } \| _ { 2 } ^ { 2 } \right] } \\ { \displaystyle \leq \frac { 1 2 8 L } { n ^ { 2 } d } \sum _ { j = 1 } ^ { t } \Big ( \frac { t } { d } \prod _ { k = j + 1 } ^ { t } \big ( 1 - \frac { 2 \eta _ { k } ( 1 - \beta ) ( n - 2 ) \sigma } { n d } \big ) ^ { 2 } } \\ { + \prod _ { k = j + 1 } ^ { t } \big ( 1 - \frac { 2 \eta _ { k } ( 1 - \beta ) ( n - 2 ) \sigma } { n d } \big ) \Big ) \eta _ { j } ^ { 2 } \mathbb { E } _ { S , A } [ F _ { S } ( \mathbf { w } _ { j } ) ] . } \end{array}
$$

Let step sizes be fixed as $\eta _ { t } \equiv \eta \le 1 / \widetilde { L }$ . For any $\gamma \geq 0$ , we develop the excess risk bound as

$$
\begin{array} { r l } {  { \mathbb { E } _ { S , A } [ F ( \mathbf { w } _ { T + 1 } ) - F _ { S } ( \mathbf { w } _ { S } ) ] } \quad } & { } \\ & { = O ( \big ( 1 + \frac { L } { \gamma } \big ) ( 1 - \eta \sigma / d ) ^ { T } + \frac { L ( d + T ) ( L + \gamma ) } { ( n - 2 ) ^ { 2 } \sigma ^ { 2 } ( 1 - \beta ) ^ { 2 } } ) } \\ & { + O ( \frac { L } { \gamma } + \frac { L ( d + T ) ( L + \gamma ) } { ( n - 2 ) ^ { 2 } \sigma ^ { 2 } ( 1 - \beta ) ^ { 2 } } ) \times \mathbb { E } [ F _ { S } ( \mathbf { w } _ { S } ) ] . \quad ( 1 ; } \end{array}
$$

Choosing $T \asymp d \sigma ^ { - 1 } \log ( n \sigma L ^ { - 1 } )$ yields

$$
\mathbb { E } _ { S , A } [ F ( \mathbf { w } _ { T + 1 } ) - F _ { S } ( \mathbf { w } _ { S } ) ] = O \left( \frac { L d ^ { \frac { 1 } { 2 } } } { n \sigma ^ { \frac { 3 } { 2 } } } \sqrt { \log \left( \frac { n \sigma } { L } \right) } \right) .
$$

Remark 4. As shown in (14), the stability measure involves a weighted sum of empirical risks. This demonstrates that low risks of output models can improve the stability along the training process. The measure also shows information including the convexity parameter $\sigma$ and learning rates $\eta _ { j }$ which are closely associated with the interplay between RCD and training datasets. Furthermore, the strong convexity of the empirical risk obviously leads to a better stability as compared to the convex case (10).

Remark 5. The convergence rate (16) gives the choice of $T$ to balance the estimation and optimization. Indeed, the optimal convergence rate lies between $O ( 1 / ( n \sigma ) )$ and $O \big ( \sqrt { \log ( n \sigma ) } / ( n \sigma ^ { \frac { 5 } { 2 } } ) \big )$ , for which the corresponding choices of $T$ are smaller than that we give. It is notable that the approximate optimal rate here almost matches the optimistic bound (13). Besides, the strong convexity promotes the fast computing $T \asymp \log ( n )$ as compared to $T \asymp n$ under the convex case. Results here are the same as those for full-batch GD (Nikolakakis et al. 2023), which can verify the theorem since the expectation for RCD leads to the gradient descent operator. Considering SGD, Lei, Liu, and Ying (2021) present the generalization bounds for pairwise learning. Under the same setting of smoothness and strong convexity, SGD achieves the excess risk bound $O ( 1 / ( n \sigma ) )$ . However, the convergence rate of SGD requires the number of iterations as $T \overset { \cdot } { \sim } O ( n / \sigma )$ and a small $F ( \mathbf { w } ^ { * } )$ .

# Experimental Verification

Here we choose the example of AUC maximization to verify the theoretical results on stability measures. Figure 1 shows results based on the usps dataset from LIBSVM (Chang and Lin 2011). More sufficient experiments about other datasets are presented in Appendix E of the arXiv version.

![](images/1b086d75f62204e430762da80fd80345ae62d51accd6bdaf970f61d2219da3b0.jpg)  
Figure 1: Euclidean distance $\Delta _ { t }$ as a function of the number of passes for the hinge loss.

We follow the settings of SGD for pairwise learning (Lei, Liu, and Ying 2021) and compare the results of RCD and SGD. In each experiment, we randomly choose 80 percents of the dataset as the training set $S$ . Then we perturb a signal example of $S$ to construct the neighboring dataset $S ^ { \prime }$ . We apply RCD or SGD to $S , S ^ { \prime }$ and get two iterate sequences, with which we plot the Euclidean distance ${ \Delta _ { t } } = \| \dot { \mathbf { w } _ { t } } - \mathbf { w } _ { t } ^ { \prime } \| _ { 2 }$ in each iteration. While the learning rates are set as $\eta _ { t } = \eta / \sqrt { T }$ with $\eta \in \{ 0 . 0 5 , 0 . 2 5 , 1 , 4 \}$ for RCD, we only compare RCD and SGD under the setting of $\eta \ : = \ : 0 . 0 5$ . Letting $n$ be the sample size, we report $\Delta _ { t }$ as a function of $T / n$ (the number of passes). We repeat the experiments 100 times, and consider the average and the standard deviation.

Considering the comparison between SGD and RCD, while the term $( T / n ) ^ { 2 }$ dominates the convergence rates of stability bounds for SGD according to Lei, Liu, and Ying (2021), the on-average argument stability bound (10) for RCD takes the order of $\check { O ( } T / n ^ { 2 } )$ . The results for the comparison are consistent with the theoretical stability bounds. Furthermore, the Euclidean distance under the logistic loss is significantly smaller than that under the hinge loss, which is consistent with the discussions of Lei, Liu, and Ying (2021).

# Conclusion

In this paper, we study the generalization performance of RCD for pairwise learning. We measure the on-average argument stability develop the corresponding excess risk bound. Results for the convex empirical risk show us how the early-stopping strategy can balance estimation and optimization. The excess risk bounds enjoy the convergence rates of $O ( 1 / \sqrt { n } )$ and $O ( \sqrt { \log ( n ) } / n )$ under the convex and strongly convex cases, respectively. Furthermore, we incorporate the low noise setting $F ( \mathbf { w } ^ { * } ) = O ( 1 / n )$ to explore better generalizations under the convex case.

There remain several questions for further investigation. Explorations under the nonparametric or the non-convex case are important for extending the applications of RCD. RCD for the specific large-scale matrix optimization also deserves a fine-grained generalization analysis.

# Acknowledgments

The work of L. Wu is supported by the National Natural Science Foundation of China (72431008, 61903309) and the Sichuan Science and Technology Program (2023NSFSC1355). The work of Y. Lei is supported by the Research Grants Council of Hong Kong [Project No. 22303723]. We are also grateful to the anonymous AAAI reviewers for their insightful and constructive comments.