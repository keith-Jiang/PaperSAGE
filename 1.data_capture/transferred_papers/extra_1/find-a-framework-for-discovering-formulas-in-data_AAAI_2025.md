# FIND: A Framework for Discovering Formulas in Data

Tingxiong Xiao1, Yuxiao Cheng1, Jinli Suo1,2,\*

1Department of Automation, Tsinghua University, Beijing 100084, China 2Institute for Brain and Cognitive Science, Tsinghua University, Beijing 100084, China jlsuo $@$ tsinghua.edu.cn

# Abstract

Scientific discovery serves as the cornerstone for advances in various fields, from the fundamental laws of physics to the intricate mechanisms of biology. However, two existing mainstream methods—symbolic regression and dimensional analysis, are significantly limited in this task: the former suffers from low computational efficiency due to the vast search space and often results in formulas without physical meaning; the latter provides a useful theoretical framework but also struggles in searching in a huge space because of lacking effective analysis for the latent variables. To address this issue, here we propose a framework for efficiently discovering underlying formulas in data, named FIND. We draw inspiration from Buckingham’s Pi theorem, imposing dimensional constraints on the input and output, thereby ensuring discovered expressions possess physical meaning. Additionally, we propose a theoretical scheme for identifying the latent structure as well as a coarse-to-fine framework, significantly reducing the search space of latent variables. This framework not only improves computational efficiency but also enhances model interpretability. From comprehensive experimental validation, FIND showcases its potential to uncover meaningful scientific insights across various domains, providing a robust tool for advancing our understanding of unknown systems.

# Introduction

Over the years, researchers have actively devoted large efforts in the field of scientific discovery (Bergen et al. 2019; Bhaskar and Nigam 1990; Camps-Valls et al. 2023; Xiao et al. $2 0 2 4 \mathrm { a }$ ; Bongard and Lipson 2007; Wang et al. 2023; Iten et al. 2020). In the quest for valuable domain knowledge and deeper field understanding, it is pivotal to explore the underlying formulas buried under data observations, which can assist in dimension reduction, prediction, or discovering the laws of nature.

There are two mainstream methods for formula discovery. Symbolic regression (SR) (Schmidt and Lipson 2009; Makke and Chawla 2024) is a machine learning technique that aims to automatically discover mathematical expressions to fit the given data, which can be used to discover physical laws, establish mathematical models, and predict unknown data. SR typically employs methods like genetic algorithms (Koza 1994) to explore the search space and find the best mathematical expression, and many researchers have begun to integrate deep learning with SR. Hernandez et al. develop a machine-learning algorithm based on SR in the form of genetic programming that is capable of discovering accurate, computationally efficient many-body potential models (Hernandez et al. 2019). Weng et al. use SR to guide the design of new oxide perovskite catalysts with improved oxygen evolution reaction activities (Weng et al. 2020). Cranmer et al. adopt sparse latent representations when training a GNN in a supervised setting, and then apply SR to components of the learned model to extract explicit physical relations (Cranmer et al. 2020). Kamienny et al. task a Transformer to directly predict the full mathematical expression, constants included and subsequently refined the predicted constants by feeding them to the non-convex optimizer as an informed initialization (Kamienny et al. 2022). Above SR methods can generate explicit expressions, which helps in understanding the underlying mechanisms of the observations. However, due to the typically large and complex search space of symbolic regression, the algorithms often suffer from limited efficiency and scalability, and most discovered formulas have no physical meaning.

Dimensional analysis (DA) aims to discover dimensionless equations with Buckingham’s Pi theorem (Buckingham 1914). As is described, almost all physical laws can be expressed as dimensionless relationships with fewer dimensionless numbers and in a more compact form (Barenblatt 2003). Dimensionless numbers are power-law monomials of some physical quantities (Tan 2011), which can simplify a problem by reducing the number of variables. Xie et al. propose two-level optimization schemes with dimensional invariance to efficiently discover dimensionless numbers in static and dynamic systems (Xie et al. 2022). HiDeNN draws on the Pi theorem and designs a universal dimensionless learning AI framework to solve challenging computational science and engineering problems with little or no available physics as well as with extreme computation cost (Saha et al. 2021). Bakarji et al. develop three data-driven techniques that use the Buckingham Pi theorem as a constraint and show decent accuracy, robustness, and computational complexity (Bakarji et al. 2022). DHC-GEP effectively discovers function forms and coefficients using basic mathematical operators and physical variables, without preassumed candidate functions, while the constraint of dimensional homogeneity filters out overfitting equations (Ma et al. 2024). DA has its theoretical system, but there is no effective analysis method for its latent variables, and its search space is still relatively large. Despite these progresses, DA-based methods are also slow and a high-efficiency discovery approach is highly demanded.

Here, we propose a framework, named FIND, for discovering formulas in data by designing a network structure including a latent layer and an expression layer. The latent layer is used to reduce dimensionality and discover meaningful input combinations, while the expression layer is used to find complete expressions. This scheme is inspired by Buckingham’s Pi theorem, which imposes dimensional constraints on the input and output, and thereby ensures the discovered expressions possess physical meaning. Overall, the proposed framework can support both accuracy and efficiency. For reliable discovery, we analyze the relationship between network weights and data derivatives to estimate the number of latent variables, their connection relationships, and the weight ratios, providing theoretical guidance for scientific discovery. For high efficiency, we propose a coarse-to-fine (C2F) searching scheme to progressively depict the probability distribution map of the optimal solution, which significantly shortens the running time and reduces the likelihood of being trapped in local optima. The source code can be found at https://github.com/HarryPotterXTX/FIND.git.

# The FIND Framework

Given a dataset $( \mathbf { X } , \mathbf { Y } )$ , where $\mathbf { X } \in \mathbb { R } ^ { b \times p }$ is the input composed of variable $\mathbf { x } \in \mathbb { R } ^ { p }$ , $\mathbf { Y } \in \mathbb { R } ^ { b \times 1 }$ is the output composed of $\mathbf { y } \in \mathbb { R }$ , and both $\mathbf { x }$ and $\mathbf { y }$ have units. We assume there exists a mapping $\mathbf { y } = f ( \mathbf { x } )$ from $\mathbf { X }$ to $\mathbf { Y }$ and designed FIND to discover this underlying relationship from data observations, with the scheme shown in Fig. 1.

![](images/15ee913a08842681d2e68f229764a13c6353bf9546be9b6579f20a92d2034ec3.jpg)  
Figure 1: The scheme of the FIND framework. (a) The network structure that consists of a latent layer for dimensionality reduction and discovering meaningful input combinations, and an expression layer in polynomial form. (b) Search in the latent space with dimensional invariance.

# Explainable Structure

Assuming that underlying natural laws can be represented by a concise and elegant equation, here we propose to decompose $f ( \mathbf { x } )$ into two parts—a latent layer $\mathbf { z } \ = f _ { 1 } ( \mathbf { x } ) \in$ $\mathbb { R } ^ { s }$ and an expression layer $\mathbf { y } = f _ { 2 } ( \mathbf { z } )$ , as shown in Fig. 1a.

For the latent layer, drawing inspiration from Buckingham’s $\mathrm { P i }$ theorem (Buckingham 1914), we set

$$
\mathbf { z } _ { i } = \prod _ { j = 1 } ^ { p } \mathbf { x } _ { j } ^ { \mathbf { W } _ { i j } } , i = 1 , \ldots , s ,
$$

where $\mathbf { W } \in \mathbb { R } ^ { s \times p }$ is the power matrix. The latent layer $f _ { 1 } ( \cdot )$ transforms the input $\mathbf { x }$ into $\mathbf { z }$ , achieving dimensionality reduction and meaningful combinations of inputs.

Regarding the expression layer, as is well known, most functions or even deep neural networks can be expanded into a Taylor series (Xiao et al. 2024b), allowing us to approximate them using polynomials. Therefore, for $f _ { 2 } ( \cdot )$ , we adopted a polynomial form.

# Structure Identification

From Eq. (1),

$$
\frac { \partial { \bf z } _ { i } } { \partial { \bf x } _ { j } } = { \bf W } _ { i j } { \bf x } _ { j } ^ { { \bf W } _ { i j } - 1 } \prod _ { k \neq j } { \bf x } _ { k } ^ { { \bf W } _ { i k } } = { \bf W } _ { i j } \frac { { \bf z } _ { i } } { { \bf x } _ { j } } ,
$$

and further we get

$$
\frac { \partial \mathbf { y } } { \partial \mathbf { x } _ { j } } = \sum _ { i = 1 } ^ { s } \frac { \partial \mathbf { z } _ { i } } { \partial \mathbf { x } _ { j } } \frac { \partial \mathbf { y } } { \partial \mathbf { z } _ { i } } = \sum _ { i = 1 } ^ { s } \mathbf { W } _ { i j } \frac { \mathbf { z } _ { i } } { \mathbf { x } _ { j } } \frac { \partial \mathbf { y } } { \partial \mathbf { z } _ { i } } .
$$

If $\mathbf { x } _ { j }$ and $\mathbf { x } _ { k }$ only connect to $\mathbf { z } _ { i }$ , we can get

$$
\frac { \partial { \bf y } } { \partial { \bf x } _ { j } } / \frac { \partial { \bf y } } { \partial { \bf x } _ { k } } = ( { \bf W } _ { i j } \frac { { \bf z } _ { i } } { { \bf x } _ { j } } \frac { \partial { \bf y } } { \partial { \bf z } _ { i } } ) / ( { \bf W } _ { i k } \frac { { \bf z } _ { i } } { { \bf x } _ { k } } \frac { \partial { \bf y } } { \partial { \bf z } _ { i } } ) ,
$$

and further

$$
\frac { \mathbf { W } _ { i j } } { \mathbf { W } _ { i k } } = \frac { \mathbf { x } _ { j } \frac { \partial \mathbf { y } } { \partial \mathbf { x } _ { j } } } { \mathbf { x } _ { k } \frac { \partial \mathbf { y } } { \partial \mathbf { x } _ { k } } } .
$$

By comparing xj ∂xy and xk ∂y , we can determine whether $\mathbf { x } _ { j }$ and $\mathbf { x } _ { k }$ exist in the same latent variable. If there is a clear linear relationship between xj ∂xy and xk ∂y calculated at multiple points, it indicates a high probability that $\mathbf { x } _ { j }$ and ${ \bf x } _ { k }$ exist in the same latent variable, or that they contribute to this latent variable more than other latent variables. Because the dataset is discrete, we cannot obtain the exact value of x j ∂xy , but we can get reliable estimation based on the difference. Here we define

$$
\rho _ { j } = \mathbf { x } _ { j } { \frac { \Delta \mathbf { y } _ { j } } { \Delta \mathbf { x } _ { j } } } ,
$$

where $\Delta \mathbf { x } _ { j }$ is the change in $\mathbf { x } _ { j }$ and $\Delta \mathbf { y } _ { j }$ is the corresponding change in $\mathbf { y }$ , and have

$$
\frac { \mathbf { W } _ { i j } } { \mathbf { W } _ { i k } } \approx \frac { \pmb { \rho } _ { j } } { \pmb { \rho } _ { k } } .
$$

Based on this, we can calculate the Pearson correlation coefficient between $\pmb { \rho } _ { 1 }$ and $\rho _ { p }$ . A larger correlation indicates a higher probability of being associated with the same latent variable, allowing us to estimate the number of latent variables, the ratios between weights as well as their positive or sign of the correlation, which greatly reduces the search space.

# Parameter Optimization

Dimensional Invariance. To ensure that the resulting equation has physical meaning, it is necessary to ensure dimension consistency between $f _ { 2 } \circ f _ { 1 } ( \mathbf { x } )$ and $\mathbf { y }$ , i.e.,

$$
f _ { 2 } ( { \bf D } { \bf W } ^ { T } ) = { \bf d } ,
$$

where $ { \mathbf { D } } \in \mathbb { R } ^ { 7 \times p }$ is the dimension matrix of $\mathbf { x }$ and $\mathbf { d } \in \mathbb { R } ^ { 7 }$ is the dimension vector of $\mathbf { y }$ . As shown in Fig. 1b, the dimension matrix $\mathbf { D } = [ \mathbf { d } _ { 1 } , \dots , \mathbf { d } _ { p } ]$ consists of dimension vectors for each corresponding input variable. The dimension vector represents the exponents of physical quantities concerning the fundamental dimensions in the natural world— mass [M], length [L], time [T], temperature $[ \Theta ]$ , electric current [I], luminous intensity $[ \mathrm { J } ]$ , and amount of substance [N]. For example, the dimension for gravitational acceleration is $m / s ^ { 2 }$ , and its dimension vector is $[ 0 , 1 , - 2 , 0 , 0 , 0 , 0 ] ^ { T }$ .

From Eq. (1), the $i$ -th latent variable $\mathbf { z } _ { i }$ is determined by W’s $i$ -th row $\mathbf { W } _ { i }$ . When the output y has no unit, i.e., $\mathbf { d } =$ 0, we set $\mathbf { D } \mathbf { W } _ { i } ^ { T } = 0$ to ensure $\mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { s }$ are dimensionless numbers, and the degree of $f _ { 2 } ( \cdot )$ is not limited. When y has a unit, i.e., $\mathbf { d } \neq 0$ , we set $\dot { \bf D } \dot { \bf W } _ { i } ^ { T } = { \bf d }$ to ensure the latent variables have the same unit as the output, and use a linear regression model to fit $\mathbf { y }$ and $\mathbf { z }$ . Incorporating the above two cases, we have the following constraints on the weight

$$
\begin{array} { r } { \mathbf { D W } _ { i } ^ { T } = \mathbf { d } , } \end{array}
$$

which ensures consistency among the dimensions of latent variables and the output. One can get the closed-form solution to the above equation

$$
\mathbf { W } _ { i } ^ { T } = \sum _ { k = 1 } ^ { p - r ( \mathbf { D } ) } \lambda _ { i k } \mathbf { w } _ { k } + \mathbf { w } ^ { * } ,
$$

where $\left\{ { \bf w } _ { k } \right\}$ is the set of homogeneous solutions that satisfies $\mathbf { D } \mathbf { w } _ { k } = 0$ , and $\mathbf { w } ^ { * }$ is a particular solution to $\mathbf { D } \mathbf { w } ^ { * } = \mathbf { d }$ . As shown in Fig. 1b, once we have obtained $\left\{ { \bf w } _ { k } \right\}$ and $\mathbf { w } ^ { * }$ , the task of searching for $\mathbf { W } \in \mathbb { R } ^ { s \times p }$ turns into searching for $\boldsymbol { \lambda } \in \mathbb { R } ^ { s \times ( p - r ( \mathbf { D } ) ) }$ with $r ( \cdot )$ denoting the rank of a matrix.

Prior Constraints. Eq. (10) transforms the search space from $s \times p$ to $s \times ( p - r ( \mathbf { D } ) )$ , but the search space remains large. Here, we restrict W from various perspectives to narrow down $\lambda$ ’s search space.

(i) Dataset Constraint. The dataset contains a wealth of information, which can be utilized to refine the search scope. From Eq. (1), there exists xjWij term in zi. If ∃xj < 0 in the dataset $( \mathbf { X } , \mathbf { Y } )$ , we let $\mathbf { W } _ { i j } \in \mathbb { Z }$ to avoid the occurrence of imaginary numbers. If $\exists \mathbf { x } _ { j } = 0$ in the dataset, we force $\mathbf { W } _ { i j } \geq 0$ to avoid a zero divisor.

$$
\left\{ \begin{array} { l l } { \mathbf { W } _ { i j } \in \mathbb { Z } , } & { \mathrm { i f } \exists \mathbf { x } _ { j } < 0 } \\ { \mathbf { W } _ { i j } \geq 0 . } & { \mathrm { i f } \exists \mathbf { x } _ { j } = 0 } \end{array} \right.
$$

(ii) Equivalence Constraint. Some weight coefficients are equivalent when the number of latent variables is greater than 1, e.g., if ${ \bf z } _ { 1 } = { \bf z } _ { 2 }$ , it is unnecessary to introduce an additional variable $\mathbf { z } _ { 2 }$ , since the weight $[ \dot { \mathbf { W } } _ { 1 } , \mathbf { W } _ { 2 } ] \sim [ \mathbf { W } _ { 1 } ]$ . Besides, exchanging two rows of $\mathbf { W }$ will not affect the result, e.g., the case $\mathbf { z } _ { 1 } = \mathbf { x } _ { 1 } ^ { 1 } \mathbf { x } _ { 2 } ^ { 2 } \mathbf { x } _ { 3 } ^ { 3 } , \mathbf { z } _ { 2 } = \mathbf { x } _ { 1 } ^ { 4 } \mathbf { x } _ { 2 } ^ { 5 } \mathbf { x } _ { 3 } ^ { 6 }$ is equivalent to $\mathbf { z } _ { 1 } = \mathbf { x } _ { 1 } ^ { 4 } \mathbf { x } _ { 2 } ^ { 5 } \mathbf { x } _ { 3 } ^ { 6 }$ , ${ \bf z } _ { 2 } = { \bf x } _ { 1 } ^ { 1 } { \bf \bar { x } } _ { 2 } ^ { 2 } { \bf \bar { x } } _ { 3 } ^ { 3 }$ . Mathematically, we have

$$
\left\{ \begin{array} { l l } { [ \mathbf { W } _ { i } , \mathbf { W } _ { k } ] \sim [ \mathbf { W } _ { i } ] , } & { \mathrm { i f } \mathbf { W } _ { i } = \mathbf { W } _ { k } } \\ { [ \mathbf { W } _ { i } , \mathbf { W } _ { k } ] \sim [ \mathbf { W } _ { k } , \mathbf { W } _ { j } ] . } & { \mathrm { e l s e } } \end{array} \right.
$$

(iii) Sparsity Constraint. In fact, in most cases, each latent variable is only composed of a partial combination of input variables, i.e., $\mathbf { W }$ is a sparse matrix

$$
\mathbf { W } _ { i j } \{ \begin{array} { l l } { = 0 , } & { \mathbf { x } _ { j } \nrightarrow \mathbf { z } _ { i } } \\ { \neq 0 . } & { \mathbf { x } _ { j }  \mathbf { z } _ { i } } \end{array} 
$$

To find concise and meaningful input combinations, we impose restrictions on the sparsity of the data. Specifically, We force W to have at most $\kappa _ { 1 }$ non-zero value and each column has no more than $\kappa _ { 2 }$ non-zero entries, i.e., each input is associated with at most $\kappa _ { 2 }$ latent variables:

$$
\left\{ \begin{array} { l } { \| \{ \mathbf { W } _ { i j } | \mathbf { W } _ { i j } \neq 0 , i = 1 , \ldots , s , j = 1 , \ldots , p \} \| \leq \kappa _ { 1 } } \\ { \| \{ \mathbf { W } _ { i j } | \mathbf { W } _ { i j } \neq 0 , i = 1 , \ldots , s \} \| \leq \kappa _ { 2 } , \ j = 1 , \ldots , p . } \end{array} \right.
$$

C2F Search. Assuming we have an estimated version of $\lambda { - } \hat { \lambda }$ , the latent variables for dataset $( \mathbf { X } , \mathbf { Y } )$ can be estimated as

$$
\hat { \mathbf { Z } } = f _ { 1 } ( \mathbf { X } | \hat { \lambda } ) .
$$

We minimize the least squares error to perform polynomial regression on $\hat { \mathbf { Z } }$ and $\mathbf { Y }$ and obtain $f _ { 2 } ( \cdot | \hat { \lambda } )$ , an estimate of $f _ { 2 } ( \cdot )$ . The predicted data $\hat { \mathbf Y }$ can be calculated as

$$
\hat { \mathbf { Y } } = f _ { 2 } ( \hat { \mathbf { Z } } | \hat { \lambda } ) .
$$

We use the coefficient of determination $R ^ { 2 }$ to measure the performance

$$
R ^ { 2 } = 1 - \frac { \sum _ { i = 1 } ^ { b } ( { \bf Y } _ { i } - \hat { \bf Y } _ { i } ) ^ { 2 } } { \sum _ { i = 1 } ^ { b } ( { \bf Y } _ { i } - \bar { \bf Y } ) ^ { 2 } } ,
$$

where $\bar { \mathbf { Y } } = ( \sum _ { i = 1 } ^ { b } \mathbf { Y } _ { i } ) / b$ is the mean of $\mathbf { Y }$ .

When $\lambda$ is determined, the polynomial coefficients for $f _ { 2 } ( \cdot )$ can be quickly calculated, so the challenge lies in searching $\lambda$ . In most cases, people tend to use input combinations with small exponents, like the law of universal gravitation $F = G m _ { 1 } m _ { 2 } r ^ { - 2 }$ , and Kepler’s third law $T = \check { k } a ^ { 1 . 5 }$ . Here, we limit $\lambda \in [ - 2 , 2 ] ^ { c }$ , where $c = s \times ( p - r ( \mathbf { D } ) )$ .

![](images/f324938190c4d436f421a7c89d318b565e7eb2536439f0034dc34efdf2810b20.jpg)  
Figure 2: An example of C2F search. The initialization stage searches $[ - 2 , 2 ] ^ { 2 }$ with a step of 1, obtaining a rough probability distribution map of the solution. The three refinement stages, by iteratively searching with smaller steps around the top coefficients, gradually shrink the solution’s range and move toward the optimal position.

There exist some typical options for searching the $\lambda$ . If we use a gradient optimization algorithm for searching, $\lambda$ tends to get stuck in local optima and results in irregular decimals such as 0.5234, rather than concise ones like 0.5. If linear searching is applied, we can avoid local optimal issues but encounter a huge search space. Therefore, we propose a coarse-to-fine (C2F) optimization framework to gradually search for the optimal solution from coarse to fine. By initialization, we locate the rough position of the target and then progressively refine the searches to increasingly smaller ranges and toward the optimal value.

(i) Initialization. We firstly divide $[ - 2 , 2 ] ^ { c }$ with a step of 1 to obtain $5 ^ { c }$ initial estimations of $\lambda$ and then exclude a lot of infeasible searching candidates with the prior constraints proposed before. For the left estimations $\hat { \lambda }$ , one can obtain their $R ^ { 2 }$ scores according to Eqns. (15)(16)(17) and record the candidates with top performances.

(ii) Refinement. We perform the next round of search with a step of 0.5 around the recorded top coefficients to obtain the new $R ^ { 2 }$ distribution and update the top coefficients. Then we repeat this process progressively, decreasing the step from 0.5 to 0.2 and finally to 0.1.

If we want $\lambda$ to be precise to 0.1, the number of candidates to be searched with the linear search algorithm and C2F algorithm is respectively

$$
\left\{ \begin{array} { c l } { { \mathrm { L i n e a r ~ S e a r c h } \mathrm { : } } } & { { n ( \hat { \lambda } ) = 4 1 ^ { c } } } \\ { { \mathrm { C 2 F ~ S e a r c h } \mathrm { : } } } & { { n ( \hat { \lambda } ) \ll 5 ^ { c } + 3 t 2 ^ { c } } } \end{array} \right.
$$

where $t$ refers to searching around the top $t$ candidates.

A simple example is shown in Fig. 2, in which we need to find the best estimation for $( \lambda _ { 1 } , \lambda _ { 2 } )$ , whose true solution is $( 0 . 6 , - 1 . 2 )$ . In the initialization stage, we divide $[ - 2 , 2 ] ^ { 2 }$ with a step of 1 and obtain 25 candidate values, with 6 exclusion items due to prior constraints. According to the $R ^ { 2 }$ metric for each candidate, $( 0 . 0 , - 1 . 0 )$ and $( 1 . 0 , - 1 . 0 )$ have the highest $R ^ { 2 }$ scores. In the first round of refinement, we explore the vicinity of $( 0 . 0 , - 1 . 0 )$ and $( 1 . 0 , - 1 . 0 )$ with a step size of 0.5, resulting in 7 new candidates, and the top 2 results are $( 0 . 5 , - 1 . 0 )$ and $( 1 . 0 , - 1 . 0 )$ . In the subsequent round of refinement, we focus on the updated top coefficients with a smaller step size. The search path progresses as follows: $( 1 . 0 , - 1 . 0 ) \ \overset { ^ { \cdot } } {  } \ ( 0 . 5 , - 1 . 0 ) \ $ $\mathsf { \bar { ( 0 . 5 , - 1 . 2 ) } } \to ( 0 . 6 , - 1 . 2 )$ . By iteratively exploring smaller step sizes around the leading coefficients and gradually narrowing the target range, we notably improve the likelihood of locating the optimal solution.

Note that the C2F search is quite flexible. When the search space for $\lambda$ is large, we can change the initialization step from 1.0 to 2.0 or even larger, and the refine step size to [1.0,0.5,0.2,0.1], which can greatly reduce the search space. Besides, the C2F search reduces the possibility of falling into local optima and greatly reduces the search space. Compared to irregular decimals, the coefficients obtained by C2F align better with human intuitions.

# Experiments

In this section, we first validate the conclusion in Eq. (7) and demonstrate that by estimating the $\rho$ -values, we can obtain the number of latent variables, the connection relationships as well as the weight ratios. We then introduce three typical applications of the FIND framework, including discovering dimensionless functions, dimensionless numbers, and physical laws. All the experiment details are accessible in the Supplementary Material.

# Identifying the Latent Variables

We designed two functions—5D and 7D respectively, to demonstrate our capability of identifying latent variables, as illustrated in Fig. 3a.

5D-Function Example. The latent variables and expression of the first function is

$$
\begin{array} { r l } & { \mathbf { z } _ { 1 } = \mathbf { x } _ { 1 } ^ { - 1 . 7 } \mathbf { x } _ { 2 } ^ { - 1 . 0 } , \mathbf { z } _ { 2 } = \mathbf { x } _ { 3 } ^ { - 1 . 2 } \mathbf { x } _ { 4 } ^ { 1 . 4 } , \mathbf { z } _ { 3 } = \mathbf { x } _ { 5 } ^ { 1 . 0 } } \\ & { \mathbf { y } = 3 + 0 . 4 \mathbf { z } _ { 1 } + 1 . 3 \mathbf { z } _ { 2 } - 0 . 7 \mathbf { z } _ { 3 } + 0 . 6 \mathbf { z } _ { 1 } \mathbf { z } _ { 2 } } \\ &  \phantom { \mathbf { z } _ { 1 } = \mathbf { x } _ { 1 } ^ { - 1 . 7 } \mathbf { x } _ { 3 } ^ { 2 } + \mathbf { z } _ { 1 } \mathbf { z } _ { 2 } \mathbf { z } _ { 3 } . } \end{array}
$$

We sampled 3125 points in the input domain, calculated the difference with $\Delta { \bf x } = 0 . 0 4$ to estimate the partial derivatives of each point, and obtained the $\rho _ { 1 } \sim \rho _ { 5 }$ values on each point with Eq. (6). According to Eq. (7), if both $\mathbf { x } _ { j }$ and ${ \bf x } _ { k }$

 PPMCC(,) /   
 z +1.00 +0.99 +0.42 −0.42 −0.91 +1.00 +1.76 +1.10 −0.01 −0.01   
   +0.9492 +10.0402 +01.4020 −0.4929 −0.9013 +01.65.74 +12.80.09 +01.00 −0.0902 −0.002   
  −0.42 −0.42 −0.99 +1.00 +0.03 −17.8 −31.3 −1.08 +1.00 +0.02   
 −0.91 −0.91 −0.03 +0.03 +1.00 −62.0 −109 −0.05 +0.05 +1.00   
 （b）   
 z PPMCC(,) /   
  7 +1.00 +01.9090 −0.43 +0.435 −0.10 +10.040 +2.52 −0.020 +0.031 −0.010   
 −0.43 −0.45 +1.00 −0.99 +0.00 −8.58 −22.5 +1.00 −1.42 +0.01   
 +0.43 +0.45 −0.99 +1.00 −0.00 +5.97 +15.6 −0.69 +1.00 −0.00  −0.10 −0.10 +0.00 −0.00 +1.00 −0.79 −2.04 +0.00 −0.00 +1.00   
 （a） （c）   
PPMCC(,) /   
+1.00 −0.56 −0.99 +0.56 −0.84 +0.93 +1.00 −0.00 −8.56 +0.00 0.90 +0.57   
−0.56 +1.00 +0.57 −0.99 +0.37 −0.46 −97.1 +1.00 +835 −0.69 +68.6 −48.7   
−0.99 +0.57 +1.00 −0.57 +0.84 −0.93 −0.12 +0.00 +1.00 −0.00 +0.11 −0.07   
+0.56 −0.99 −0.57 +1.00 −0.37 +0.46 +141 −1.45 −1209 +1.00 −99.3 +70.4 0.00 0.00 0.00 0.00 0.00 0.00 −0.84 +0.37 +0.84 −0.37 +1.00 −0.98 −0.78 +0.00 +6.67 −0.00 +1.00 −0.56   
+0.93 −0.46 −0.93 +0.46 −0.98 +1.00 +1.50 −0.00 −12.9 +0.00 −1.71 +1.00 （d）

are connected to $\mathbf { z } _ { i }$ and their weights on $\mathbf { z } _ { i }$ are much greater than their weights on other latent variables, then $\rho _ { j }$ and $\rho _ { k }$ show a clear proportional relationship.

We calculated the Pearson product-moment correlation coefficient (PPMCC) between $\rho _ { j }$ and $\rho _ { k }$ , and used the least squares method to calculate their slope to estimate ${ \rho } _ { j } / { \rho } _ { k }$ . The PPMCC and ratio tables are shown in Fig. 3b: from which we can get $\{ { \bf x } _ { 1 } , { \bf x } _ { 2 } \} \to { \bf z } _ { 1 }$ , $\{ \mathbf { x } _ { 3 } , \mathbf { x } _ { 4 } \}  \bar { \mathbf { z } } _ { 2 }$ , $\{ { \bf x } _ { 5 } \} $ $\mathbf { z } _ { 3 }$ , directly displaying the number of latent variables and the connection relationships; the ratio table shows the estimations for the weight ratios $\rho _ { 1 } / \rho _ { 2 } ~ = ~ 1 . 7 6 , \rho _ { 3 } / \rho _ { 4 } ~ =$ $- 0 . 9 2$ , which consist with the true values $- 1 . 7 / - 1 . 0 =$ $1 . 7 , - 1 . 2 / 1 . 4 \ : = \ : - 0 . 8 6$ and can greatly reduce the search space by providing a rough range via partial derivatives.

In Fig. 3c, we increased $\Delta \mathbf { x }$ to 0.4. The PPMCC results remain consistent with the original function. However, the estimated ratio values are beginning to diverge from the true value due to inaccurate derivative estimation. Nevertheless, we can still get the sign of the between-weight correlation.

7D-Function Example. The latent variables and expression of the second examplar function is

$$
\begin{array} { r l } & { \mathbf z _ { 1 } = \mathbf x _ { 1 } ^ { - 1 . 7 } \mathbf x _ { 3 } ^ { 0 . 2 } \mathbf x _ { 7 } ^ { - 1 . 0 } , \mathbf z _ { 2 } = \mathbf x _ { 2 } ^ { 1 . 0 } \mathbf x _ { 4 } ^ { - 1 . 3 } , \mathbf z _ { 3 } = \mathbf x _ { 6 } ^ { - 0 . 6 } \mathbf x _ { 7 } ^ { 0 . 7 } , } \\ & { \mathbf y = \sin ( 2 \mathbf z _ { 1 } + \pi / 3 ) - \mathbf z _ { 1 } \mathbf z _ { 2 } + \mathrm e ^ { \mathbf z _ { 1 } \mathbf z _ { 3 } } + \sin ( \mathbf z _ { 3 } ^ { 2 } ) } \\ & { \quad \quad + \mathbf z _ { 1 } \mathbf z _ { 2 } \mathbf z _ { 3 } + \mathbf z _ { 2 } ^ { 2 } . } \end{array}
$$

We set $\Delta \mathbf { x } = 0 . 0 4$ and get its PPMCC and ratio tables in Fig. 3d. The PPMCC results show that $\{ { \bf x } _ { 1 } , { \bf x } _ { 3 } , { \bf x } _ { 7 } \} \to { \bf z } _ { 1 }$ , $\{ \bar { \mathbf { x } _ { 2 } } , \mathbf { x } _ { 4 } \}  \mathbf { z } _ { 2 }$ , $\{ { \bf x } _ { 6 } , { \bf x } _ { 7 } \} \to { \bf z } _ { 3 }$ , and ${ \bf x } _ { 5 }$ is an independent variable. One can see that although the expression is not in polynomial form, we can still find the latent variables and the connection relationship correctly. The ratio table also reflects the ratio relationship between weights very well, except for ${ \bf x } _ { 7 }$ which cannot be accurately estimated due to the simultaneous connection with two latent variables.

Recall that not all observations are complete enough for reliable derivative estimation, so this estimation method might be inapplicable for extremely sparse data. In the subsequent experiments, we show the results of our C2F framework searching for the optimal solution from sparse data.

# Application #1: Finding Dimensionless Functions

We collected datasets from 7 distinct systems and employed our FIND framework to identify the original functions, as demonstrated in Tab. 1. All 7 datasets consist of simulation data with $1 \%$ Gaussian noise. Notably, no unit is assigned to the input and output, i.e., ${ \bf D } = 0 , { \bf d } = 0$ .

The experimental findings demonstrate that FIND excels in identifying latent variables across all scenarios. In the first experiment, employing C2F search only necessitates exploring 186 potential points, contrasting with a linear search that would entail investigating $4 1 ^ { 3 }$ points. Throughout experiments 1 to 3, augmenting the input variables from 3 to 5 did not impede the successful identification of both latent variables and expressions. Despite the original expression in experiment 4 deviating from a polynomial form, our method adeptly uncovers the latent variable and derives a polynomial surrogate for the initial expression. In experiments 5 to 7, where there are 2 or 3 latent variables, the original functions can still be efficiently and accurately determined.

Under the C2F framework, we iteratively refine the search space to pinpoint the optimal point effectively. This process enables us to identify the optimal solution even when the original expression deviates from a polynomial form. In such cases, we can still derive a polynomial representation that serves as a viable replacement.

# Application #2: Finding Dimensionless Numbers

Dimensionless numbers are quantities used in physics and engineering to describe and analyze problems without specific units, playing a crucial role in understanding and predicting natural phenomena and designing engineering systems. These numbers normalize problems, remove unit dependencies, and facilitate comparisons across various scenarios. Here, we assess our method for identifying dimensionless numbers in both static and dynamic systems.

Static System. The laser–metal interaction is an important problem. During this interaction, a depression filled with vapor, known as a keyhole, typically emerges on the molten metal surface. The formation of the keyhole stems from the recoil pressure induced by vaporization. Owing to its intricate reliance on numerous physical mechanisms, comprehending the kinetic essence of the keyhole poses inherent challenges.

The keyhole size $e$ is related to a lot of parameters, such as the laser power $\eta P$ , the laser scan speed $V _ { s }$ , the laser beam radius $r _ { 0 }$ , the thermal diffusivity $\alpha$ , the material density $\rho _ { 0 }$ , the heat capacity $C _ { p }$ , and the difference between melting and ambient temperatures $\Delta T$ . We assess the performance of our FIND framework using a dataset about keyholes (Xie et al. 2022), encompassing 90 experiments conducted on three distinct materials: titanium alloy (Ti6Al4V), aluminum alloy (Al6061), and stainless steel (SS316) (Zhao et al. 2019; Gan et al. 2021). The output variable $e$ is normalized as the keyhole aspect ratio denoted as $e ^ { * } = e / r _ { 0 }$ .

When the coefficient accuracy is set to 0.1, the outcome is

$$
\begin{array} { r l } & { z = \frac { \eta P ^ { 1 . 6 } } { V _ { s } ^ { 0 . 7 } r _ { 0 } ^ { 2 . 3 } \alpha ^ { 0 . 9 } \rho _ { 0 } ^ { 1 . 6 } C _ { p } ^ { 1 . 6 } \Delta T ^ { 1 . 6 } } , } \\ & { e ^ { * } = - 0 . 0 4 + 0 . 0 2 z , } \end{array}
$$

with a high $R ^ { 2 } ~ = ~ 0 . 9 8 6 5$ . The weights obtained in this case are 1.6, -0.7, which may not align with human conventions. Consequently, we adjusted the accuracy to 0.5 and conducted another test, yielding

$$
\begin{array} { c } { \displaystyle { z = \frac { \eta P } { \rho _ { 0 } C _ { p } \Delta T \sqrt { \alpha V _ { s } r _ { 0 } ^ { 3 } } } , } } \\ { \displaystyle { e ^ { * } = - 0 . 6 1 + 0 . 1 5 z , } } \end{array}
$$

with $R ^ { 2 } = 0 . 9 8 1 0$ . The latent variable $z$ divided by $\pi$ is a discovered keyhole number Ke (Gan et al. 2021; Ye et al. 2019), which can be derived from heat transfer theory.

Dynamic System. We use a dataset of Navier-Stokes equations with different Reynolds numbers (Xie et al. 2022) to demonstrate FIND’s potential in discovering dimensionless numbers in partial differential equations (PDEs). By changing dynamic viscosity $\mu$ , cylinder diameter $l$ , inlet velocity $v$ , fluid density $\rho _ { 0 }$ , and the pressure difference $p _ { 0 }$ , different PDEs are created. In each PDE scenario, there are six variables $t , x , y , u , v , w$ , and we use SINDy (Brunton, Proctor, and Kutz 2016) to process the data for each scenario and obtain the corresponding PDE equation. All the discovered PDEs have the following form

$$
\frac { \partial w } { \partial t } = \lambda _ { 1 } u \frac { \partial w } { \partial x } + \lambda _ { 2 } v \frac { \partial w } { \partial y } + \lambda _ { 3 } \frac { \partial ^ { 2 } w } { \partial x ^ { 2 } } + \lambda _ { 4 } \frac { \partial ^ { 2 } w } { \partial y ^ { 2 } } .
$$

There are three sets of PDE parameters here, which are

$$
\lambda = \left\{ \begin{array} { l l } { \left[ - 0 . 9 9 2 5 , - 0 . 9 9 2 5 , + 0 . 0 2 1 2 , + 0 . 0 2 1 2 \right] ^ { T } , } \\ { \left[ - 0 . 9 9 0 9 , - 0 . 9 9 0 9 , + 0 . 0 1 2 6 , + 0 . 0 1 2 6 \right] ^ { T } , } \\ { \left[ - 0 . 9 9 4 1 , - 0 . 9 9 4 1 , + 0 . 0 1 1 1 , + 0 . 0 1 1 1 \right] ^ { T } . } \end{array} \right.
$$

We can infer that $\lambda _ { 1 } = \lambda _ { 2 } = - 1$ are fix coefficients, while $\lambda _ { 3 } = \lambda _ { 4 }$ are dynamic dimensionless numbers. We use $\rho _ { 0 } , \mu , v , l , p _ { 0 }$ as inputs and $\lambda _ { 3 } , \lambda _ { 4 }$ as the outputs to search for dimensionless numbers, and find that

$$
\lambda _ { 3 } = \lambda _ { 4 } = \frac { \mu } { \rho _ { 0 } v l } = \frac { 1 } { R e } ,
$$

where $R e$ is the Reynolds number (Reynolds 1883). Finally, we get a unified PDE form

$$
\frac { \partial w } { \partial t } = - u \frac { \partial w } { \partial x } - v \frac { \partial w } { \partial y } + \frac { 1 } { R e } \left( \frac { \partial ^ { 2 } w } { \partial x ^ { 2 } } + \frac { \partial ^ { 2 } w } { \partial y ^ { 2 } } \right) .
$$

# Application #3: Finding Physical Laws

We examined a real-world dataset of planets in the solar system sourced from the NASA Planetary Fact Sheet (NASA 2017). The variables tested include planet mass $m$ , planet diameter $d _ { 0 }$ , planet density $\rho _ { 0 }$ , gravitational acceleration $g$ , escape velocity $v _ { e }$ , rotation period $t _ { r }$ , length of day $t _ { d }$ , distance from sun $r _ { s }$ , perihelion $r _ { p }$ , aphelion $r _ { a }$ , orbital period $t _ { o }$ , orbital velocity ${ \boldsymbol { v } } _ { o }$ . The set $S = \{ m , d _ { 0 } , \rho _ { 0 } , g , v _ { e } , t _ { r } , t _ { d } , r _ { s } , r _ { p } , r _ { a } , t _ { o } , v _ { o } \}$ encompasses all these variables.

We explore the formulas in two ways, as shown in Tab. 2. The initial method involves utilizing physical units to confine the search space, thereby directly deriving relevant physical formulas. In the second method, input/output units are disregarded, and a dimensional constant is appended after the formula.

Consider $m$ as the output and the other variables $S \backslash \{ m \}$ as the input, FIND yielded the result $m = 0 . 4 5 d _ { 0 } ^ { 3 } \rho _ { 0 }$ . This formula establishes the connection among mass $m$ , volume $\pi d _ { 0 } ^ { 3 } / 6$ , and density $\rho _ { 0 }$ , with the theoretical expression being $\dot { m } = 0 . 5 2 d _ { 0 } ^ { 3 } \dot { \rho _ { 0 } }$ . Taking $v _ { e }$ as the output and $\cal { S } \backslash \{ v _ { e } \}$ as the input, we discovered the relationship $v _ { e } = 1 . 0 \dot { 4 } \sqrt { g d _ { 0 } }$ that corresponds to the escape velocity formula, while the ground-truth formulation is $\dot { v _ { e } } ~ = ~ \dot { \sqrt { g d _ { 0 } } }$ . When designating $t _ { o }$ as the output and $S \backslash \{ t _ { o } \}$ as the input, the result is $t _ { o } = 6 . 2 1 r _ { s } / v _ { o }$ . This formula describes the connection among orbital period $t _ { o }$ , orbital circumference $2 \pi r _ { s }$ , and orbital velocity $\scriptstyle v _ { 0 }$ , and the true expression is $t _ { o } = 6 . 2 8 r _ { s } / v _ { o }$ .

In the preceding cases, all units of the inputs and outputs were considered, resulting in formulas with inherent physical meaning without additional adjustments. Subsequently, the exploration of formulas was conducted disregarding the input and output units. To prevent the rediscovery of already established formulas, a subset of variables was chosen for each experiment. For instance, after selecting $m$ and $d _ { 0 }$ , we exclude $\rho _ { 0 }$ because $m = 0 . 5 2 d _ { 0 } ^ { 3 } \rho _ { 0 }$ .

Table 1: Toy dataset with $1 \%$ Gaussian noise.   

<html><body><table><tr><td>ID</td><td>Latent</td><td></td><td>Expression</td><td>n(入）</td><td>R²</td><td></td><td>Time</td></tr><tr><td rowspan="2">1-2</td><td></td><td>1.2x2</td><td></td><td></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td rowspan="2"></td><td colspan="3">1.2x2</td></tr><tr><td></td><td></td><td></td><td></td><td rowspan="2">800</td><td rowspan="2"></td><td rowspan="2">000</td><td rowspan="2">4</td></tr><tr><td rowspan="2">3</td><td rowspan="2"></td><td colspan="3"></td></tr><tr><td>1X2</td><td></td><td></td><td rowspan="2">5777</td><td rowspan="2"></td><td rowspan="2">0.99</td><td rowspan="2">14.04</td></tr><tr><td rowspan="2"></td><td rowspan="2">Z=x1 X</td><td colspan="3"></td></tr><tr><td>1.7x2 1.2x2</td><td></td><td>y=6+3.6z-180z2²-13z</td><td rowspan="2"></td><td rowspan="2">198</td><td rowspan="2">0.99</td><td rowspan="2">0.51</td></tr><tr><td rowspan="2">4 5</td><td rowspan="2">Z=x1</td><td colspan="3">2 1.2x2</td></tr><tr><td></td><td>17x2 1.3×5 X</td><td>y=0.7+n2+））.+z2</td><td></td><td rowspan="2">3343</td><td rowspan="2">0.99</td><td rowspan="2"></td><td rowspan="2">12.23</td></tr><tr><td rowspan="2">6</td><td rowspan="2">z1=x1</td><td colspan="3">1.7x X1</td></tr><tr><td>Z1=x1</td><td>7x</td><td>1.3×5</td><td>y=++1++</td><td rowspan="2">3679</td><td rowspan="2">0.99</td><td rowspan="2"></td><td rowspan="2">15.83</td></tr><tr><td rowspan="2">7</td><td rowspan="2"></td><td colspan="3">17x</td><td>y=-12++++π/3+</td></tr><tr><td></td><td>1.7x2</td><td></td><td></td><td rowspan="2">2992</td><td rowspan="2"></td><td rowspan="2">0.99</td><td rowspan="2">185.63</td></tr><tr><td rowspan="2"></td><td rowspan="2">z1=x1</td><td colspan="3">1.7x2</td><td>y=+132+06++</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table></body></html>

Table 2: Formulas found with FIND.   

<html><body><table><tr><td></td><td>Input</td><td>Output</td><td>FIND</td><td>Theory</td></tr><tr><td rowspan="3">D≠0,d≠0</td><td>Sm}</td><td>m</td><td>m = 0.49do</td><td>m = 0.52dp</td></tr><tr><td>S{ve</td><td>Ue</td><td>Ue = 1.04√gdo</td><td>Ue=√gdo</td></tr><tr><td>S{to</td><td>to</td><td>to=6.21rs/Uo</td><td>to=6.28rs/Uo</td></tr><tr><td rowspan="3">D=0,d=0</td><td>do,g,tr,ta,rs,t</td><td>m</td><td>m=4.04e+09dg</td><td>m=3.75e+09d5g</td></tr><tr><td>m,do,g,td,rs</td><td>Ue</td><td>Ue =1.63e-05 m/do</td><td>Ue /m/do =1.63e-05</td></tr><tr><td>m,do，g,tr,td,rs</td><td>to</td><td>to=5.43e-1r</td><td>to=5.46e-1r5</td></tr></table></body></html>

Here we provide the law discovery for three different outputs. (i) When $m$ is designated as the output, the obtained formula is $m = 4 . 0 4 \mathrm { { \overline { { e } } ^ { + 0 9 } } } d _ { 0 } ^ { 2 } g$ . This formula is typically used for calculating planet mass, and the theoretical expression is $m = d _ { 0 } ^ { 2 } g \bar { / } ( \bar { 4 } G ) = 3 . 7 5 e ^ { + 0 9 } d _ { 0 } ^ { 2 } g$ , where $G$ represents the constant of universal gravitation. To maintain consistency in dimensions on both sides of the formula, the unit $s ^ { 2 } k g / m ^ { 3 }$ is appended to the constant $4 . 0 4 e ^ { + 0 9 }$ . (ii) When $v _ { e }$ is considered as the output, the formula obtained is $v _ { e } ~ = ~ 1 . 6 3 e ^ { - 0 5 } \sqrt { m / d _ { 0 } }$ . This represents an alternative calculation approach for escape velocity, where $v _ { e } = \sqrt { 4 G m / d _ { 0 } } = 1 . \bar { 6 3 } e ^ { - 0 5 } \sqrt { m / d _ { 0 } }$ . The unit of the constant $1 . 6 3 e ^ { - 0 5 }$ is $m ^ { 1 . 5 } s ^ { - 1 } k g ^ { - \dot { 0 } . 5 }$ . (iii) When $t _ { o }$ is the output, the result is $t _ { o } ~ = ~ 5 . 4 \bar { 3 } e ^ { - 1 0 } r _ { s } ^ { 1 . 5 }$ . This formula corresponds to Kepler’s third law, which states $t _ { o } = r _ { s } ^ { 1 . 5 } / \sqrt { K } =$ $5 . 4 6 e ^ { - 1 0 } r _ { s } ^ { 1 . 5 }$ , where $K$ denotes the Kepler consstant. The unit for the constant $5 . 4 3 e ^ { - 1 0 }$ is $s / m ^ { 1 . 5 }$ .

# Summary and Conclusions

To efficiently discover formulas from observations, we propose the FIND framework consisting of a latent layer and an expression layer: the former explores meaningful input combinations and reduces data dimension, while the latter pursues explicit expressions from latent variables to output.

To analyze the latent structure, we analyze the relationship between weights and derivatives. By statistically analyzing the linear correlation and ratio of $\rho$ values from multiple points, we can obtain the connection relationships and the weight ratios. To get the optimal weights, we propose the C2F framework, which gradually depicts the optimal probability graph from coarse to fine, which greatly reduces the optimization time and avoids getting stuck in local optima.

FIND has built a simple, general, and explainable framework to obtain field knowledge from data observations quickly. The typical applications include discovering dimensionless functions, dimensionless numbers, and natural physical laws. We have conducted comprehensive experiments to verify the high accuracy and efficiency of FIND, as well as its wide applicability.

Limitations. Due to its fixed structure, FIND can only obtain solutions in polynomial form. We use grid search to reduce the likelihood of local optima but at the cost of increased difficulty in handling high-dimensional data. Moreover, FIND is intrinsically a data-driven method, so the accuracy of results might degenerate when the input dataset is highly dispersed.

Future Work. We will further explore better structures and more efficient solutions, especially from low-quality observations. We also plan to integrate SR techniques to convert polynomials into symbolic expressions. Moreover, we aim to broaden FIND’s applications beyond fluid mechanics and astronomy, extending its use to areas such as industry, medicine, and chemistry.

# Acknowledgments

This work is jointly funded by Ministry of Science and Technology of China (Grant No. 2024YFF0505703) and National Natural Science Foundation of China (Grant Nos. 61931012 and 62088102).