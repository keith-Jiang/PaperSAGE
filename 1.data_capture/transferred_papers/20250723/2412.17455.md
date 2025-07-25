# Learning from Summarized Data: Gaussian Process Regression with Sample Quasi-Likelihood

Yuta Shikuri

Tokio Marine Holdings, Inc. Tokyo, Japan shikuriyuta@gmail.com

# Abstract

Gaussian process regression is a powerful Bayesian nonlinear regression method. Recent research has enabled the capture of many types of observations using non-Gaussian likelihoods. To deal with various tasks in spatial modeling, we benefit from this development. Difficulties still arise when we can only access summarized data consisting of representative features, summary statistics, and data point counts. Such situations frequently occur primarily due to concerns about confidentiality and management costs associated with spatial data. This study tackles learning and inference using only summarized data within the framework of Gaussian process regression. To address this challenge, we analyze the approximation errors in the marginal likelihood and posterior distribution that arise from utilizing representative features. We also introduce the concept of sample quasi-likelihood, which facilitates learning and inference using only summarized data. Non-Gaussian likelihoods satisfying certain assumptions can be captured by specifying a variance function that characterizes a sample quasi-likelihood function. Theoretical and experimental results demonstrate that the approximation performance is influenced by the granularity of summarized data relative to the length scale of covariance functions. Experiments on a real-world dataset highlight the practicality of our method for spatial modeling.

# 1 Introduction

Gaussian process regression is a Bayesian nonlinear regression method that can handle many types of observation data (Rasmussen and Williams 2006). A key application of Gaussian process regression is spatial modeling (Cressie 1993) across various fields, such as geology, agriculture, marketing, and public health. For example, it is used to estimate the distribution of soil moisture and regional medical expenses. Specifying likelihood functions according to these tasks impacts prediction performance. While a conventional choice is a Gaussian likelihood that is conjugate with a Gaussian process prior, recent research has focused extensively on applying Gaussian process regression to non-Gaussian likelihoods. The challenge of specifying non-Gaussian likelihoods lies in posterior distributions that lack closed-form expressions. Gaussian approximations of the posterior distribution are commonly used to overcome this challenge.

Laplace approximation represents the posterior distribution as a Gaussian centered at the maximum a posteriori estimate (Williams and Barber 1998). Expectation propagation finds the parameters in a Gaussian that approximate the posterior distribution through an iterative scheme of matching moments (Minka 2001). Variational inference maximizes a lower bound on the marginal likelihood via a Gaussian approximation (Sheth, Wang, and Khardon 2015). These methods for specifying non-Gaussian likelihoods allow us to handle various types of observations.

Difficulties have been encountered when only summarized data is available. This situation arises particularly in the context of spatial modeling. Due to privacy concerns associated with location-specific observation information, spatial data is often summarized to include representative locations, summary statistics, and counts. Too fine locations might identify individuals even if direct identifiers are removed. Hence, the granularity of summarized data tends to be intentionally coarse to ensure that personal information (e.g., financial assets, interests, purchase activity, and medical history) is not exposed. In addition to protecting individual privacy, management costs necessitate aggregating data by specific units (e.g., cities, branches, hospitals, and schools). To analyze regional trends, the latitude and longitude pair for each unit is linked to the corresponding summary statistics. Consequently, techniques are required for spatial modeling given only summarized data.

In this study, we address learning and inference from summarized data in Gaussian process regression. Figure 1 illustrates the overview of our approach. For input approximation using representative features, we demonstrate the theoretical errors in the marginal likelihood and posterior distribution within the Gaussian process regression framework. We also propose the concept of sample quasi-likelihood. This approach enables the computation of the marginal likelihood and posterior distribution using only summarized data, offering straightforward implementation and low computational complexity. The sample quasi-likelihood is defined by a variance function. Specifying this variance function allows for capturing non-Gaussian likelihoods that satisfy certain conditions. Theoretical and experimental results indicate that the approximation performance strongly depends on the granularity of summarized data relative to the length scale of covariance functions. Experiments using real-world spatial data demonstrate the usability of our approach for supervised learning tasks when only summarized data is available.

![](images/f76fcbdf1e8515cb3281e7e9171b45a15e53118c9351fc91639e56878d5411be.jpg)  
Figure 1: Overview of our approach. We address Gaussian process regression using only summarized data, whihc includes representative features, summary statistics, and data point counts. Our approach involves the input approximation and the introduction of sample quasi-likelihood under specific assumptions. Specifying a variance function that characterizes a sample quasi-likelihood allows us to capture non-Gaussian likelihoods.

# 2 Related Work

There are existing methods capable of handling aggregated outputs. Composite likelihood is derived by multiplying a collection of component likelihoods (Besag 1975; Varin, Reid, and Firth 2011). This approach is advantageous for dealing with aggregated outputs while retaining some properties of the full likelihood. Additionally, synthetic likelihood (Wood 2010; Price et al. 2018) and approximate Bayesian computation (Beaumont 2019) are also promising. These methods approximate the posterior distribution by comparing summary statistics with simulated data generated from a tractable probability model. Outputs sometimes take distributional forms, such as random lists, intervals, and histograms. Beranger, Lin, and Sisson 2022 constructed likelihood functions for these forms. Considering that a Gaussian process has parameters corresponding to its inputs, handling outputs not directly linked to inputs becomes crucial. Some studies (Law et al. 2018; Tanaka et al. 2019) have provided methods for learning and inference from aggregated outputs in Gaussian process regression, assuming that input data points or their distribution are available.

# 3 Preliminaries

# 3.1 Notations

Appendix A provides the list of symbols. Let $\boldsymbol { \mathcal { X } } \subset \mathbb { R } ^ { d }$ denote the domain of inputs, and $y$ denote the domain of outputs. The symbol of $\kappa .$ is a Gram matrix. The $( i , j )$ -th element of the Gram matrix for $( A , B )$ is the return value of a covariance function when the inputs are the $i$ -th row of $A$ and the $j$ -th row of $B$ . The matrix with the opposite order of indices is the transposed matrix (e.g., $K _ { * \mathrm { f } } \stackrel { \cdot \cdot } { = } K _ { \mathrm { f } * } ^ { \top } )$ . Let $\mathcal { N }$ denote the probability density function of Gaussian distribution, the vertical line with equality denote substitution (e.g., $p ( \pmb { y } \mid \pmb { f } ) \rvert _ { \pmb { f } = \pmb { W } _ { \mathbf { f u } } \pmb { u } } )$ , $O$ denote the Landau symbol, and $F$ denote the upper cumulative distribution function of the chi-square distribution given by

$$
F ( a , b ) \equiv \biggl ( \int _ { 0 } ^ { \infty } t ^ { \frac { a } { 2 } - 1 } \exp ( - t ) d t \biggr ) ^ { - 1 } \int _ { \frac { b } { 2 } } ^ { \infty } t ^ { \frac { a } { 2 } - 1 } \exp ( - t ) d t
$$

for any $a \in \mathbb { N }$ and $b \in [ 0 , \infty )$ . The Landau symbol in matrix operations is applied to each matrix element. We assume that symmetric Gram matrices are positive-definite, that likelihood functions and their gradients are differentiable and bounded, and that covariance functions are symmetric and continuous.

# 3.2 Gaussian Process Regression

In this subsection, we describe the basic framework of Gaussian process regression. A more detailed introduction is presented in (Rasmussen and Williams 2006).

A Gaussian process $f \sim \mathcal { G P } ( \tau ( \cdot ) , k ( \cdot , \cdot ) )$ is a distribution over functions characterized by a mean function $\tau : \mathcal { X }  \mathbb { R }$ and a covariance function $k : \mathcal { X } \times \mathcal { X } \to ( 0 , \infty )$ . A stochastic process $\{ f ( { \pmb x } ) \ | \ { \pmb x } \in { \pmb \chi } \}$ is a Gaussian process if and only if the random variables $\hat { \{ f ( { \pmb x } ) ~ | ~ { \pmb x } \in \mathcal { X } ^ { \bar { \prime } } \} }$ for any finite set $\chi ^ { \prime } \subseteq \chi$ follow a multivariate normal distribution. For simplicity, we take $\tau ( \cdot ) = 0$ . Given inputs $\pmb { X } \equiv ( \pmb { x } _ { i } ) _ { i = 1 } ^ { n }$ with $\pmb { x } _ { i } \in \mathcal { X }$ and outputs $\pmb { y } \equiv ( y _ { i } ) _ { i = 1 } ^ { n }$ with $y _ { i } \in \mathcal { V }$ , the hyperparameters of the covariance function are learned to maximize the log marginal likelihood defined as

$$
\mathcal { L } \equiv \log \int _ { \mathbb { R } ^ { n } } p ( \pmb { y } \mid \pmb { f } ) p ( \pmb { f } ) d \pmb { f } ,
$$

where ${ \pmb f } \equiv ( f ( { \pmb x } _ { i } ) ) _ { i = 1 } ^ { n }$ , $p ( f ) \equiv \mathcal { N } ( f ; 0 , K _ { \mathrm { f f } } )$ , $p ( { \pmb y } \mid { \pmb f } )$ is a likelihood function of $f$ , and $K _ { \mathbb { f } }$ is the Gram matrix of $( X , X )$ . The complexity of optimizing $\mathcal { L }$ given a Gaussian

<html><body><table><tr><td>Function Name</td><td colspan="2">k(x,x')</td><td colspan="4">S1(α,z,z')</td></tr><tr><td>Laplacian</td><td>exp(-|x - x'|)</td><td></td><td>1- exp(-2α)</td><td></td><td>1-exp(-α)</td><td></td></tr><tr><td>Gaussian</td><td>exp(</td><td>|xc-2x|²)</td><td>1-exp(-2α(|z- z|+α))</td><td></td><td>1- exp(-α(|≥-x*|+a))</td><td></td></tr></table></body></html>

Table 1: Errors of covariance functions. Given $z , z ^ { \prime } , x ^ { * } \in \mathcal { X }$ , for any $\alpha \in ( 0 , \infty )$ and $\mathbf { \boldsymbol { x } } , \mathbf { \boldsymbol { x } } ^ { \prime } \in \mathcal { X }$ , $\operatorname* { m a x } \{ | x - z | , | x ^ { \prime } - z ^ { \prime } | \} <$ $\alpha \Rightarrow | k ( x , x ^ { \prime } ) - k ( z , z ^ { \prime } ) | < \zeta _ { 1 } ( \alpha , z , z ^ { \prime } )$ and $| k ( \pmb { x } , \pmb { x } ^ { * } ) - k ( \pmb { z } , \pmb { x } ^ { * } ) | < \zeta _ { 2 } ( \alpha , z , \pmb { x } ^ { * } )$ . See appendix B.1.

likelihood is $O ( n ^ { 3 } )$ . For new inputs $\pmb { X } _ { * } \equiv ( \pmb { x } _ { i } ^ { * } ) _ { i = 1 } ^ { n _ { * } }$ with $\pmb { x } _ { i } ^ { * } \in \mathcal { X }$ , the posterior distribution of ${ f } _ { * } \equiv ( f ( \pmb { x } _ { i } ^ { * } ) ) _ { i = 1 } ^ { n _ { * } }$ is

$$
p ( \pmb { f } _ { * } \mid \pmb { y } ) \equiv \exp ( - \mathcal { L } ) \int _ { { \mathbb { R } } ^ { n } } p ( \pmb { f } _ { * } \mid \pmb { f } ) p ( \pmb { y } \mid \pmb { f } ) p ( \pmb { f } ) d \pmb { f } ,
$$

where $\kappa _ { \mathrm { f } * }$ is the Gram matrix of $( X , X _ { * } ) , K _ { * * }$ is that of $( X _ { * } , X _ { * } )$ , and $p ( \pmb { f } _ { * } \mid \pmb { f } ) \equiv \mathcal { N } ( \pmb { f } _ { * } ; \pmb { K } _ { * \mathrm { f } } \pmb { K } _ { \mathrm { f } } ^ { - 1 } \pmb { f } , \pmb { K } _ { * * } -$ $K _ { * \mathrm { f } } K _ { \mathrm { f f } } ^ { - 1 } K _ { \mathrm { f } * } )$ .

# 4 Input Approximation

To clarify the discussion in this study, we define summarized data and the error $\beta \in ( 0 , \infty )$ of a covariance function. Assume that data points $( { \pmb x } _ { i } , y _ { i } ) _ { i = 1 } ^ { n }$ are associated with assignments $\omega \equiv ( \omega _ { i } ) _ { i = 1 } ^ { n }$ with $\omega _ { i } \in \{ 1 , \cdots , m \}$ . Let summarized data be represented as a set of tuples $( z _ { j } , \bar { y } _ { j } , n _ { j } ) _ { j = 1 } ^ { m }$ , where each $z _ { j } \in \mathcal { X }$ is a representative feature, each $\bar { y _ { j } } \in \mathbb { R }$ is a summary statistic that depends only on the outputs $y _ { i }$ with $\omega _ { i } = j$ , and each $n _ { j } > 0$ is the number of indices such that $\omega _ { i } = j$ . Let $\beta$ satisfy the following conditions:

• $| k ( \pmb { x } _ { i } , \pmb { x } _ { j } ) - k ( \pmb { z } _ { \omega _ { i } } , \pmb { z } _ { \omega _ { j } } ) | < \beta$ for all $1 \leq i \leq j \leq n$ . • $| k ( \pmb { x } _ { i } , \pmb { x } _ { j } ^ { * } ) - k ( \pmb { z } _ { \omega _ { i } } , \pmb { x } _ { j } ^ { * } ) | < \beta$ for all $1 \leq i \leq n$ and $1 \leq j \leq n _ { * }$ .

Even in situations where only summarized data is available, the range of complete data inputs is often accessible. This range allows us to evaluate the approximation errors of covariance functions when the inputs are representative features, as described in table 1. For Gaussian process regression, we can flexibly design many types of covariance functions (e.g., covariance functions equivalent to an infinitely wide deep network (Lee et al. 2018) or relying on non-Euclidean metric (Feragen, Lauze, and Hauberg 2014)). Lemma 4.1 gurantees that a small range improves the approximation accuracy for any covariance function.

Lemma 4.1. Given $z , z ^ { \prime } , x ^ { * } \in \mathcal { X }$ , there exists $\alpha \in ( 0 , \infty )$ such that $\operatorname* { m a x } \{ | x - z | , | x ^ { \prime } - z ^ { \prime } | \} < \alpha \Rightarrow \operatorname* { m a x } \{ | k ( x , x ^ { \prime } ) -$ $k ( z , z ^ { \prime } ) | , | k ( \pmb { x } , \pmb { x } ^ { * } ) - k ( z , \pmb { x } ^ { * } ) | \} < \beta ,$ for any $\beta \in ( 0 , \infty )$ and $\pmb { x } , \pmb { x } ^ { \prime } \in \mathcal { X }$ .

# Proof. See appendix B.2.

Here we approximate the marginal likelihood and posterior distribution using representative features, and evaluate the approximation errors. Initially, we consider the case where the inputs are given by $( z _ { \omega _ { i } } ) _ { i = 1 } ^ { n }$ and the parameters corresponding to the centroids ${ \cal Z } \equiv ( z _ { i } ) _ { i = 1 } ^ { m }$ are represented by $\pmb { u } \equiv ( f ( \pmb { z } _ { i } ) ) _ { i = 1 } ^ { m }$ . In this case, the prior distribution and the likelihood function are $p ( \boldsymbol { \mathbf { \rho } } u ) \equiv \mathcal { N } ( \boldsymbol { \mathbf { \rho } } u ; \boldsymbol { \mathbf { 0 } } , K _ { \mathrm { \mathbf { u u } } } )$ and $p ( \pmb { y } \mid W _ { \mathbf { f u } } \pmb { u } ) \equiv p ( \pmb { y } \mid \pmb { f } ) | _ { \pmb { f } = W _ { \mathbf { f u } } \pmb { u } }$ , respectively, where

$K _ { \mathbf { u u } }$ is the Gram matrix of $( Z , Z )$ , and $W _ { \mathbf { f u } }$ is $n \times m$ matrix with $[ W _ { \mathbf { f u } } ] _ { i j } = 1$ if $\omega _ { i } = j$ ; $[ W _ { \mathbf { f u } } ] _ { i j } = 0$ otherwise. Using this prior distribution and likelihood function, the log marginal likelihood is

$$
\mathcal { E } \equiv \log \int _ { \mathbb { R } ^ { m } } p ( \pmb { y } \mid W _ { \mathbf { f u } } \pmb { u } ) p ( \pmb { u } ) d \pmb { u } .
$$

The posterior distribution of $f _ { * }$ is $p ( f _ { * } \mid \pmb { y } , \omega )$ defined as

$$
\exp ( - \mathcal { E } ) \int _ { { \mathbb { R } } ^ { m } } p ( f _ { * } \mid W _ { \mathrm { f u } } u ) p ( y \mid W _ { \mathrm { f u } } u ) p ( u ) d u ,
$$

where $p ( \pmb { f } _ { * } \ \vert \ W _ { \mathbf { f } \mathbf { u } } \ b { u } ) \equiv p ( \pmb { f } _ { * } \ \vert \ \pmb { f } ) \vert _ { \pmb { f } = W _ { \mathbf { f } \mathbf { u } } \pmb { u } } .$ . We evaluate their errors in comparison to Gaussian process regression with complete data. Let $p ( \pmb { f } \mid \pmb { u } )$ be defined as

$$
\mathcal { N } ( f ; K _ { \mathrm { f u } } K _ { \mathrm { u u } } { - 1 } u , K _ { \mathrm { f f } } - K _ { \mathrm { f u } } K _ { \mathrm { u u } } ^ { - 1 } K _ { \mathrm { u f } } ) ,
$$

where $K _ { \mathbf { f u } }$ is the Gram matrix of $( X , Z )$ . As in eq. (1) and eq. (2), the marginal likelihood and posterior distribution of the original model contain $\begin{array} { r } { p ( \pmb { f } ) = \int _ { \mathbb { R } ^ { m } } p ( \pmb { f } \ \lvert \textbf { \em u } ) p ( \pmb { u } ) d \pmb { u } } \end{array}$ . Considering that eq. (3) and eq. (4) correspond to the case where $f \mathrm { ~ = ~ } W _ { \mathrm { f u } } u$ holds, we proceed to analyze the integral of $p ( \pmb { f } \mid \pmb { u } )$ . To facilitate this analysis, we introduce the following lemma.

Lemma 4.2. Let $\gamma$ denote the maximum absolute value of the elements in $W _ { \mathrm { f u } } ^ { \mathrm { } } - K _ { \mathrm { f u } } K _ { \mathrm { u u } } { } ^ { - 1 }$ . Then we have

$$
K _ { \mathrm { f f } } - K _ { \mathrm { f u } } K _ { \mathrm { u u } } ^ { - 1 } K _ { \mathrm { u f } } = O ( \beta + m \beta \gamma ) .
$$

Proof. See appendix B.3.

The error $\gamma$ tends to become small when $\beta$ is sufficiently small. In such cases, from lemma 4.1 and lemma 4.2, the integral of $p ( \pmb { f } \mid \pmb { u } )$ over the region within a certain distance from the point $f \ = \ W _ { \mathrm { f u } } u$ approaches 1 as the range of inputs becomes small. The following lemmas evaluate this dynamic. For simplicity, we assume $\gamma \neq 0$ .

Lemma 4.3. For $\delta _ { 1 } \in [ 0 , \infty )$ and $\delta _ { 2 } \in [ 0 , \delta _ { 1 } ]$ , define

$$
\epsilon ( \delta _ { 1 } , \delta _ { 2 } ) \equiv F \Bigl ( m , \frac { \kappa ( \delta _ { 1 } - \delta _ { 2 } ) ^ { 2 } } { \lambda _ { 1 } } m \Bigr ) + F \Bigl ( n , \frac { \delta _ { 2 } ^ { 2 } } { \lambda _ { 2 } } n \Bigr ) ,
$$

where $\begin{array} { r } { \kappa \equiv \operatorname* { i n f } _ { \boldsymbol { u } \in \mathbb { R } ^ { m } ; | ( W _ { \mathrm { f u } } - K _ { \mathrm { f u } } K _ { \mathrm { u u } } - 1 ) \boldsymbol { u } | = 1 } \frac { n } { m } | \boldsymbol { u } | ^ { 2 } , \boldsymbol { \lambda } _ { 1 } } \end{array}$ is the maximum eigenvalue of $K _ { \mathbf { u u } }$ , and $\lambda _ { 2 }$ is that of $K _ { \mathbb { f } } \mathrm { - }$ $K _ { \mathrm { f u } } K _ { \mathrm { u u } } ^ { - 1 } K _ { \mathrm { u f } }$ . Then we have

$$
\int _ { \mathbb { R } ^ { m } } \int _ { \mathbb { R } ^ { n } \setminus R ( \pmb { \mathscr { s } } , \delta _ { 1 } ) } p ( \pmb { \mathscr { f } } | \pmb { \mathscr { u } } ) p ( \pmb { \mathscr { u } } ) d \pmb { \mathscr { f } } d \pmb { \mathscr { u } } \leq \epsilon ( \delta _ { 1 } , \delta _ { 2 } ) ,
$$

Proof. See fig. 2 and appendix B.4.

![](images/831409a026183710d98f1828dd51280131ea808bbf95b962c69cd8f49d0e1861.jpg)  
Figure 2: Sketch of the proof of lemma 4.3. The first and second terms in eq. (7) correspond respectively to the left and right sides. Evaluating the integral in eq. (8) requires analyzing $p ( \pmb { f } \mid \pmb { u } )$ over $\mathbb { R } ^ { \bar { n } } \backslash R ( \pmb { u } , \bar { \delta } _ { 1 } )$ , posing a significant challenge. Consequently, to assess the upper bound, we consider the hypersphere centered at the mean vector in $p ( \pmb { f } \ | \ \pmb { u } )$ . The space of $\mathbf { \Delta } _ { \pmb { u } }$ is divided based on whether $R ( { \pmb u } , \delta _ { 1 } )$ encompasses the hypersphere. Gray indicates the space that is not integrated with respect to $f$ .

Lemma 4.4. The following holds:

$$
\kappa ^ { - 1 } = O ( m \gamma ^ { 2 } ) , \lambda _ { 2 } = O ( \beta + m \beta \gamma ) .
$$

Proof. See appendix B.5.

Lemma 4.5. Suppose $\xi \ge \xi _ { 0 }$ , where $\xi _ { 0 }$ is the larger value satisfying $2 \sqrt { \pi \xi _ { 0 } } \exp ( - \xi _ { 0 } ) = 1 .$ . Then $F ( m , \xi m )$ is monotonically decreasing with respect to $m$ .

Proof. See appendix B.6.

The percentiles of the chi-square distributions appearing in eq. (7) depend on $\kappa$ and $\lambda _ { 2 }$ . Lemma 4.4 shows that the integral in eq. (8) approaches zero as $\beta$ and $\gamma$ decrease. The evaluation of the integral behaves oppositely with respect to $m$ in lemma 4.4 and lemma 4.5. While the integral decreases as $\delta _ { 1 }$ increases, $f$ within $R ( { \pmb u } , \delta _ { 1 } )$ move away from $W _ { \mathrm { f u } } u$ . Considering this behavior, we derive the error bounds for the marginal likelihood and posterior distribution.

Theorem 4.6. Let η be defined as

$$
\eta \equiv \operatorname* { i n f } _ { \xi \in [ \xi _ { 0 } , \infty ) } \bigl ( \sqrt { \xi \lambda _ { 1 } \kappa ^ { - 1 } } + \sqrt { \xi \lambda _ { 2 } } + F ( m , \xi m ) \bigr ) .
$$

Given the hyperparameters and complete data, we have

$$
\begin{array} { r l r } & { } & { \mathcal { L } - \mathcal { E } = O ( \eta ) , } \\ & { } & { \left| \left| \mathbb { E } _ { p ( f _ { * } | y ) } [ f _ { * } ] - \mathbb { E } _ { p ( f _ { * } | y , \omega ) } [ f _ { * } ] \right| \right| = O ( \eta ) , } \end{array}
$$

where $\left\| \cdot \right\|$ denote the norm of a vector.

Proof. See appendix B.7.

Theorem 4.6 demonstrates that the performance of input approximation improves as $\eta$ decreases. While the first and second terms in $\eta$ increase as $\xi$ increases, the last term decreases. Lemma 4.4 suggests that smaller values of $\beta$ and $\gamma$ , relative to the square root of $m$ , help prevent the first and second terms from increasing. From lemma 4.5, the last term decreases as the summarized data becomes finer. Figure 3 shows that $\eta$ decreases with finer summarized data. Furthermore, the length scale of the covariance functions significantly affects $\eta$ . As the length scale increases, $\eta$ decreases due to smaller $\beta$ and $\gamma$ . Note that this result does not directly prove the effect of the length scale on input approximation, as the proportional constants in eq. (11) and eq. (12) contain the hyperparameters.

![](images/2c307a7bdf1399b8873b87afc04272a59321fb05aba000246b25f15e3f4d0bdf.jpg)  
Figure 3: Behavior of $\eta$ in a toy model. Let $\theta$ represent the length scale. Light blue: $\theta = 0 . 1$ . Light green: $\theta = 1$ . Pink: $\theta = 1 0$ . The left and right figures correspond to the covariance functions $\exp ( - \frac { 1 } { \theta } | x - \overline { { x ^ { \prime } } } | )$ and $\mathrm { e x p } \big ( - \frac { 1 } { 2 \theta ^ { 2 } } | x - x ^ { \prime } | ^ { 2 } \big )$ , respectively. In each figure, the vertical and horizontal axes represent $\eta$ and $m$ , respectively. The $n = 1 0 0 0$ inputs and $m$ representative features are equally spaced within $[ 0 , 2 \pi ]$ . The assignment $\omega _ { i }$ of a data point $x _ { i }$ is the index $j$ of the closest centroid $z _ { j }$ . Appendix B.8 explains the process of obtaining $\kappa$ and $\eta$ .

# 5 Sample Quasi-Likelihood

Since the likelihood function $p ( \pmb { y } \mid \pmb { W } _ { \mathbf { f u } } \pmb { u } )$ requires the outputs of complete data, the log marginal likelihood $\mathcal { E }$ and the posterior distribution $p ( \pmb { f } _ { * } \mid \pmb { y } , \omega )$ still cannot be computed using only summarized data. Consequently, we replace the likelihood function with a function that excludes complete data outputs and incorporates summary statistics $\bar { \pmb { y } } \equiv ( \bar { y } _ { i } ) _ { i = 1 } ^ { m }$ . As one such function, we propose the concept of sample quasi-likelihood characterized by a variance function $v : \mathbb { R } \to ( 0 , \infty )$ as follows:

Definition 5.1. A sample quasi-likelihood function $\bar { Q }$ : $\mathcal { V } ^ { m } \times \mathbb { R } ^ { m } \to \mathbb { R }$ is a function such that

$$
\frac { \partial \bar { Q } ( \bar { y } , u ) } { \partial u } = { \cal V } _ { \bf u u } ^ { - 1 } ( \bar { y } - u ) ,
$$

where $V _ { \bf u u } \equiv \mathrm { d i a g } ( n _ { 1 } ^ { - 1 } v ( \bar { y } _ { 1 } ) , \cdot \cdot \cdot , n _ { m } ^ { - 1 } v ( \bar { y } _ { m } ) ) .$ .

Suppose that the prior distribution is $p ( \pmb { u } )$ , and the likelihood function is $\mathcal { N } ( \bar { y } ; u , V _ { \mathrm { u u } } )$ . Then the log marginal likelihood is

$$
\begin{array} { c } { { \displaystyle \mathcal { Q } \equiv - \frac { m } { 2 } \log ( 2 \pi ) - \frac { 1 } { 2 } \log \lvert K _ { \bf u u } + V _ { \bf u u } \rvert } } \\ { { \displaystyle ~ - \frac { 1 } { 2 } \bar { \pmb { y } } ^ { \top } ( K _ { \bf u u } + V _ { \bf u u } ) ^ { - 1 } \bar { \pmb { y } } . } } \end{array}
$$

The posterior distribution of $f _ { * }$ is $\mathcal { N } ( f _ { * } ; \mu _ { q } , \Sigma _ { q } )$ , where $\mu _ { q } \equiv K _ { \ast \mathbf { u } } ( K _ { \mathbf { u u } } + V _ { \mathbf { u u } } ) ^ { - 1 } \bar { \mathbf { y } }$ , $\Sigma _ { q } \equiv K _ { * * } - K _ { * \mathbf { u } } ( K _ { \mathbf { u u } } +$

$V _ { \mathbf { u u } } ) ^ { - 1 } K _ { \mathbf { u } * }$ , and $K _ { { \mathbf { u } } * }$ is the Gram matrix of $( Z , X _ { * } )$ . See appendix C.1. These marginal likelihood and posterior distribution are computed using only summarized data. The implementation is straightforward, as it is equivalent to Gaussian process regression with a Gaussian likelihood. The computational complexity is $O ( m ^ { 3 } )$ .

Despite these advantages, we still have not discussed the difference between likelihood and sample quasi-likelihood. Here we demonstrate the asymptotic behavior of the sample quasi-likelihood. The assumptions outlined below enable us to apply the Laplace approximation using summary statistics. Note that the asymptotic behavior in Bayesian statistics can be found in (Watanabe 2018).

Assumption 5.2. $- \nabla \nabla \log { p ( \pmb { y } | W _ { \mathbf { f u } } \pmb { u } ) } | _ { \pmb { u } = \pmb { \bar { y } } } = V _ { \mathbf { u } \mathbf { u } } ^ { - 1 } .$

Assumption 5.3. $p ( \pmb { y } \mid \pmb { W } _ { \mathbf { f u } } \pmb { u } )$ is unimodal with respect to $\mathbf { \Delta } _ { \pmb { u } }$ , having its mode at $\bar { y }$ .

The Laplace approximation is conventionally used to handle non-Gaussian likelihoods by approximating the posterior with a Gaussian centered at the maximum a posteriori estimate. Unlike this, our approach replaces the likelihood with a Gaussian centered at the maximum likelihood estimate.

Theorem 5.4. Suppose that assumption 5.2 and assumption 5.3 hold. Then we have

$$
\begin{array} { c l } { { \displaystyle \mathcal { E } - \mathcal { Q } = \log p ( \pmb { y } \mid W _ { \mathbf { f u } } \pmb { u } ) \vert _ { \pmb { u = \bar { y } } } + \frac { m } { 2 } \log ( 2 \pi ) } } \\ { { \displaystyle ~ + \frac { 1 } { 2 } \log \vert V _ { \mathbf { u u } } \vert + o _ { p } ( m ) , } } \end{array}
$$

where $o _ { p }$ denote convergence in probability. Additionally, the posterior distribution $p ( \pmb { f } _ { * } \mid \pmb { y } , \omega )$ asymptotically converges to $\mathcal { N } ( f _ { * } ; \mu _ { p } , \Sigma _ { p } )$ , where

![](images/1ee8612bcf099777164c63cc7ab2f1032e0e56f3d96a884ecd9dfbca680af3dd.jpg)  
Figure 4: Behavior of marginal likelihood and posterior distribution in a toy model. The vertical axis in the upper figure represents the absolute difference between $\mathcal { Q } + \log p ( \pmb { y } \ |$ $\begin{array} { r } { \dot { W _ { \mathrm { f u } } } u ) | _ { u = \bar { y } } + \frac { m } { 2 } \log ( 2 \pi ) + \frac { 1 } { 2 } \log | V _ { \mathbf { u u } } | } \end{array}$ and $\mathcal { L }$ . The vertical axis in the lower figure represents the root mean squared error (RMSE) between $K _ { * \mathrm { f } } ( K _ { \mathrm { f f } } + \mathrm { d i a g } ( 1 , \cdot \cdot \cdot , 1 ) \bar { ) } ^ { - 1 } y$ and $\mu _ { q }$ . Each solid line represents the average over 100 trials, with the shaded region in each color showing the range between the maximum and minimum values. The $n _ { * } = 1 0 0 0$ new inputs were uniformly generated within $[ 0 , 2 \pi ]$ . Each output $y _ { i }$ was generated from $\mathcal { N } ( \sin ( \pmb { x } _ { i } ) , 1 )$ . The summary statistic is the sample mean. The likelihood function is $\begin{array} { r } { \dot { \prod } _ { i = 1 } ^ { n } \mathcal { N } ( y _ { i } ; f ( \pmb { x } _ { i } ) , 1 ) } \end{array}$ . Other displays and conditions are identical to those in fig. 3.

$$
\begin{array} { r l } & { \mu _ { p } \equiv K _ { \ast \mathrm { f } } K _ { \mathrm { f } } ^ { - 1 } W _ { \mathrm { f u } } ( V _ { \mathrm { u u } } ^ { - 1 } + K _ { \mathrm { u u } } ^ { - 1 } ) ^ { - 1 } V _ { \mathrm { u u } } ^ { - 1 } \bar { y } , } \\ & { \Sigma _ { p } \equiv K _ { \ast \ast } - K _ { \ast \mathrm { f } } K _ { \mathrm { f f } } ^ { - 1 } K _ { \mathrm { f \ast } } } \\ & { ~ + K _ { \ast \mathrm { f } } K _ { \mathrm { f f } } ^ { - 1 } W _ { \mathrm { f u } } ( V _ { \mathrm { u u } } ^ { - 1 } + K _ { \mathrm { u u } } ^ { - 1 } ) ^ { - 1 } W _ { \mathrm { u f } } K _ { \mathrm { f f } } ^ { - 1 } K _ { \mathrm { f \ast \ast } } . } \end{array}
$$

Theorem 5.5. Suppose that $( { V _ { \bf u u } ^ { - 1 } + K _ { \bf u u } } ^ { - 1 } ) ^ { - 1 } , K _ { \bf u u } ^ { - 1 }$ , $K _ { \mathbf { u } * } , K _ { \mathbf { f } } ^ { - 1 }$ , $W _ { \mathbf { f u } }$ , and $V _ { \mathbf { u u } } ^ { - 1 } \bar { y }$ become $O ( \beta )$ when multiplied by a matrix of $O ( \beta )$ . Then we have

Proof. See appendix C.3.

$$
\pmb { \mu _ { p } } - \pmb { \mu _ { q } } = O ( \beta ) , \pmb { \Sigma _ { p } } - \pmb { \Sigma _ { q } } = O ( \beta + m \beta ^ { 2 } ) .
$$

Theorem 5.4 describes the asymptotic behavior of the marginal likelihood and posterior distribution. This approximation performs well when $n$ is sufficiently larger than $m$ . Since the right side of eq. (15) does not depend on the hyperparameters of a covariance function, we can employ $\mathcal { Q }$ to optimize $\mathcal { E }$ . Regarding the posterior distribution, we cannot compute $\mathcal { N } ( f _ { * } ; \mu _ { p } , \bar { \Sigma } _ { p } )$ given only summarized data since it contains $K _ { \mathbb { f } }$ and $\kappa _ { * \mathrm { f } }$ . Theorem 5.5 allows us to avoid the computation of them. From eq. (16), the sample quasilikelihood can be used for a smaller range of inputs. Figure 4 illustrates that the approximation errors of the marginal likelihood and posterior distribution are affected by the relative granularity of summarized data with respect to the length

Proof. See appendix C.2.

scale of the covariance functions. Their behavior largely aligns with the dynamics of $\eta$ in fig. 3. While the input domain of this toy model is one-dimensional, this behavior is expected to persist in higher-dimensional data. However, it is important to note that the range of inputs expands as the input dimension increases.

The discussion so far supports the use of the sample quasilikelihood for learning from summarized data. Addressing non-Gaussian likelihoods with summarized data presents a significant challenge, just as it does in Gaussian process regression with complete data. Our approach tackles this challenge by specifying an appropriate variance function, ensuring that assumption 5.2 is satisfied for a implicit likelihood function. Furthermore, the sample quasi-likelihood allows for the use of various summary statistics that satisfy assumption 5.3. This assumption requires that the summary statistic corresponds to the maximum likelihood estimate of the implicit likelihood function. For example, the sample median serves as the maximum likelihood estimate of the location parameter in the Laplace distribution. Our approximation performs poorly when the likelihood function associated with the variance function and summary statistic is flat. The concept of sample quasi-likelihood is analogous to the quasi-likelihood proposed by (Wedderburn 1974, 1976). While our approach also benefits from specifying a variance function instead of a likelihood function, its motivation and definition differ slightly. The formulation of the sample quasi-likelihood provides a computational advantage by expressing the marginal likelihood and posterior distribution in closed form. The comparison between both approaches is presented in appendix C.4.

Table 2: Approximation performance. To compare our approximation with the original regression, we present the mean and standard deviation of the RMSE between the mean vectors of their posterior distributions over 100 trials. The RMSE was normalized by dividing it by the standard deviation of the training data outputs. Let the likelihood be Gaussian and let $n = 1 0 0 0$ .   

<html><body><table><tr><td rowspan="2">Output</td><td rowspan="2">Covariance</td><td colspan="6">Grid Size</td></tr><tr><td>1.6 × 1.6</td><td>0.8 × 0.8</td><td>0.4 × 0.4</td><td>0.2 × 0.2</td><td>0.1 × 0.1</td><td>0.05 × 0.05</td></tr><tr><td>MedInc</td><td>Laplacian</td><td>0.57 ± 0.04</td><td>0.60 ± 0.05</td><td>0.59 ± 0.05</td><td>0.54 ± 0.05</td><td>0.44 ± 0.05</td><td>0.32 ± 0.05</td></tr><tr><td>MedInc</td><td>Gaussian</td><td>0.56 ± 0.09</td><td>0.59 ± 0.11</td><td>0.57 ± 0.12</td><td>0.50 ± 0.14</td><td>0.39 ± 0.12</td><td>0.28 ± 0.10</td></tr><tr><td>HouseAge</td><td>Laplacian</td><td>0.63 ± 0.03</td><td>0.57 ± 0.03</td><td>0.52 ± 0.04</td><td>0.41 ± 0.04</td><td>0.30 ± 0.04</td><td>0.19 ± 0.03</td></tr><tr><td>HouseAge</td><td>Gaussian</td><td>0.62 ± 0.04</td><td>0.57 ± 0.04</td><td>0.54 ± 0.05</td><td>0.40 ± 0.06</td><td>0.28 ± 0.07</td><td>0.17 ± 0.06</td></tr><tr><td>AveRooms</td><td>Laplacian</td><td>0.62 ± 0.12</td><td>0.59 ± 0.11</td><td>0.54 ± 0.11</td><td>0.49 ± 0.09</td><td>0.40 ± 0.10</td><td>0.29 ± 0.08</td></tr><tr><td>AveRooms</td><td>Gaussian</td><td>0.62 ± 0.16</td><td>0.63 ± 0.16</td><td>0.61 ± 0.18</td><td>0.59 ± 0.23</td><td>0.48± 0.26</td><td>0.32 ± 0.24</td></tr><tr><td>AveBedrms</td><td>Laplacian</td><td>0.62 ± 0.15</td><td>0.58 ± 0.15</td><td>0.51 ± 0.12</td><td>0.42 ± 0.10</td><td>0.32 ± 0.13</td><td>0.22 ± 0.10</td></tr><tr><td>AveBedrms</td><td>Gaussian</td><td>0.77 ± 0.23</td><td>0.77 ± 0.24</td><td>0.75 ± 0.22</td><td>0.71 ± 0.27</td><td>0.59 ± 0.30</td><td>0.44 ± 0.33</td></tr><tr><td>Population</td><td>Laplacian</td><td>0.26 ± 0.10</td><td>0.26 ± 0.10</td><td>0.25 ± 0.11</td><td>0.25 ± 0.12</td><td>0.22 ± 0.11</td><td>0.19 ± 0.09</td></tr><tr><td>Population</td><td>Gaussian</td><td>0.16 ± 0.12</td><td>0.16 ± 0.12</td><td>0.15 ± 0.13</td><td>0.14 ± 0.13</td><td>0.13 ± 0.13</td><td>0.11 ± 0.12</td></tr><tr><td>AveOccup</td><td>Laplacian</td><td>0.47 ± 0.17</td><td>0.47 ± 0.18</td><td>0.46 ± 0.18</td><td>0.45 ± 0.17</td><td>0.41 ± 0.16</td><td>0.30 ± 0.12</td></tr><tr><td>AveOccup</td><td>Gaussian</td><td>0.46 ± 0.33</td><td>0.46 ± 0.33</td><td>0.45 ± 0.34</td><td>0.45 ± 0.33</td><td>0.43 ± 0.32</td><td>0.36 ± 0.32</td></tr><tr><td>MedValue</td><td>Laplacian</td><td>0.79 ± 0.03</td><td>0.78 ± 0.04</td><td>0.74 ± 0.05</td><td>0.63 ± 0.05</td><td>0.47 ± 0.05</td><td>0.32 ± 0.04</td></tr><tr><td>MedValue</td><td>Gaussian</td><td>0.85 ± 0.08</td><td>0.84 ± 0.11</td><td>0.75 ± 0.14</td><td>0.58 ±0.20</td><td>0.41 ± 0.18</td><td>0.27 ± 0.13</td></tr></table></body></html>

# 6 Spatial Modeling

We investigate the usage of our method in spatial modeling tasks, using the California housing dataset 1. Our approach leverages the input approximation via representative features and utilizes the sample quasi-likelihood. We evaluate its performance by comparing it with Gaussian process regression given complete data. To compute the exact marginal likelihood and posterior distribution, we used a Gaussian likelihood in this comparison. We also perform predictive testing with the Gaussian and Poisson likelihoods.

Summarized Data. The dataset was randomly shuffled and then split to obtain the $n$ training data points and $n _ { * } = $ $2 0 6 4 0 - n$ test data points. The inputs were the pairs of latitude and longitude included among the attributes. The seven remaining attributes were used as outputs, respectively. The domain $[ \bar { 3 } 2 . 5 4 , 4 1 . 9 5 ] \times [ - 1 2 4 . 3 5 , - \bar { 1 } 1 4 . 3 1 ]$ in latitude and longitude was divided into girds. The data points within each grid were summarized, with the centers of the grids serving as the representative features, and the summary statistics being the sample means. Assume that the sample variances are also provided and that the distribution of input data points for learning is unavailable.

Functions. The likelihood function is presented in appendix D.1. Let $g ( \pmb { x } )$ denote $f ( { \pmb x } )$ for the Gaussian, and $\exp ( { \pmb x } )$ for the Poisson. We replace the summary statistics $\bar { y }$ with $( g ^ { - 1 } ( \bar { y } _ { i } ) ) _ { i = 1 } ^ { m }$ . This replacement makes the likelihood functions consistent with assumption 5.3. Similarly, we specified the variance functions based on the likelihoods, ensuring assumption 5.2 is satisfied. The mean function was $\begin{array} { r } { \tau ( \cdot ) = \breve { g } ^ { - 1 } ( \frac { 1 } { n } \dot { \sum } _ { i = 1 } ^ { n } y _ { i } ) } \end{array}$ . The covariance functions were designed by multiplying a constant kernel with the functions used in fig. 3 and adding white noise.

Hyperparameters. The hyperparameters of the covariance functions were optimized using the L-BFGS-B algorithm, starting from an initial value of 1. This optimization was implemented with version 1.7.3 of the SciPy software 2, with the parameters set to their default values. We used a 64- bit Windows machine with the Intel Core i9-9900K $@ 3 . 6 0$ GHz and $6 4 \mathrm { G B }$ of RAM. The code was implemented in the python programming language 3.7.3 version. As a hyperparameter, we also optimized the variance $\sigma \in ( 0 , \infty )$ of the Gaussian likelihood. In this case, the right side of eq. (15) contains the hyperparameter. Therefore, we employed $\mathcal { E }$ instead of $\mathcal { Q }$ for the Gaussian likelihood, which can be computed using the sample mean and sample variances.

Baselines. For the prediction test baselines, we adopted the sparse variational inference method (Hensman, Fusi, and Lawrence 2013; Hensman, Matthews, and Ghahramani 2015) implemented in version 1.12.0 of the GPy software 3. The parameters in GPy were set to their default values. The initial locations of $\lfloor { \sqrt { m ^ { 3 } n ^ { - 1 } } } \rfloor$ inducing points were randomly allocated from $m$ representative locations. Two patterns using variational inference were attempted. One involved the standard learning and inference with complete data. In the other, the inputs were replaced with representative locations (i.e., $( z _ { \omega _ { i } } ) _ { i = 1 } ^ { n } )$ . The complete outputs were used in both cases, contrasting with our setting where only summary statistics are available. We refer to these as the complete VI and summarized VI, respectively. The other settings were identical to those in our approach.

Table 3: Prediction performance. The mean and standard deviation of the RMSE and execution time [s] over 100 trials are shown. The RMSE between predictors and test outputs was normalized by dividing it by the standard deviation of the test data outputs. The prediction was obtained by applying the mapping $g$ to the mean vector of the posterior distribution of $f _ { * }$ . The execution time was measured from the start of learning to the end of prediction. The covariance function was the Gaussian. The grid size was $0 . 4 \times 0 . 4$ . Let $n = 1 0 0 0 0$ . The displays are bolded when the mean loss of the summarized VI and our method differs by greater than 0.01. Similarly, bolding is applied when the difference in the mean execution time exceeds 1 second. The results of the $0 . 8 \times 0 . 8$ grid size can be found in appendix D.2.   

<html><body><table><tr><td rowspan="2">Output</td><td rowspan="2">Likelihood</td><td colspan="2">Complete VI</td><td colspan="2">Summarized VI</td><td colspan="2">Our Approach</td></tr><tr><td>RMSE</td><td>Time</td><td>RMSE</td><td>Time</td><td>RMSE</td><td>Time</td></tr><tr><td>MedInc</td><td>Gaussian</td><td>0.881 ± 0.014</td><td>41±4</td><td>1.015 ± 0.015</td><td>30 ± 14</td><td>0.970 ±0.004</td><td>6±1</td></tr><tr><td>MedInc</td><td>Poisson</td><td>0.887 ± 0.024</td><td>59 ±14</td><td>1.014 ± 0.016</td><td>59 ±12</td><td>0.977 ±0.008</td><td>6±2</td></tr><tr><td>HouseAge</td><td>Gaussian</td><td>0.880 ± 0.049</td><td>39 ±11</td><td>0.970 ± 0.024</td><td>29 ±17</td><td>1.000 ± 0.000</td><td>4±0</td></tr><tr><td>HouseAge</td><td>Poisson</td><td>0.861 ± 0.049</td><td>55±18</td><td>0.966 ± 0.021</td><td>51± 20</td><td>1.032 ± 0.060</td><td>7±2</td></tr><tr><td>AveRooms</td><td>Gaussian</td><td>0.988 ± 0.031</td><td>15 ±12</td><td>0.995 ± 0.014</td><td>12 ± 12</td><td>0.975 ± 0.051</td><td>10±3</td></tr><tr><td>AveRooms</td><td>Poisson</td><td>0.990 ± 0.031</td><td>19 ±19</td><td>0.995 ± 0.015</td><td>19 ±20</td><td>0.949 ±0.012</td><td>8±1</td></tr><tr><td>AveBedrms</td><td>Gaussian</td><td>0.989 ± 0.029</td><td>15 ±13</td><td>0.996 ± 0.012</td><td>14 ± 12</td><td>0.964 ± 0.041</td><td>12±3</td></tr><tr><td>AveBedrms</td><td>Poisson</td><td>0.927 ± 0.021</td><td>62±9</td><td>0.941 ± 0.016</td><td>61 ±11</td><td>0.962 ± 0.044</td><td>10±3</td></tr><tr><td>Population</td><td>Gaussian</td><td>1.242 ± 0.224</td><td>5±3</td><td>1.041 ± 0.085</td><td>18±15</td><td>1.000 ± 0.000</td><td>1±0</td></tr><tr><td>Population</td><td>Poisson</td><td>0.990 ± 0.009</td><td>45 ±26</td><td>1.000 ± 0.005</td><td>37± 24</td><td>1.004 ± 0.013</td><td>6±1</td></tr><tr><td>AveOccup</td><td>Gaussian</td><td>1.002 ± 0.008</td><td>12 ± 11</td><td>1.002 ± 0.006</td><td>10 ± 10</td><td>1.002 ± 0.006</td><td>6±2</td></tr><tr><td>AveOccup</td><td>Poisson</td><td>1.021 ± 0.141</td><td>59 ± 23</td><td>1.015 ± 0.063</td><td>43 ±15</td><td>1.001 ± 0.002</td><td>6±2</td></tr><tr><td>MedValue</td><td>Gaussian</td><td>0.699 ±0.009</td><td>41±2</td><td>1.008 ± 0.015</td><td>40±2</td><td>0.904 ± 0.010</td><td></td></tr><tr><td>MedValue</td><td>Poisson</td><td>0.746 ± 0.033</td><td>59 ±15</td><td>0.973 ± 0.038</td><td>58±14</td><td>0.909 ± 0.011</td><td>6±1 5±0</td></tr></table></body></html>

Evaluation. Table 2 shows that the granularity of summarized data strongly impact the approximation performance of our approach. This result is consistent with the theoretical findings and the simple simulation using the toy model. The superiority or inferiority of the two covariance functions depended on the output attributes, even though the same inputs were used. The comparison between the complete VI and summarized VI in table 3 suggests a negative impact of the coarse inputs. Our approach offered advantages in both loss and execution time compared to the summarized VI. For the AveRooms, specifying the variance function corresponding to the Poisson distribution was beneficial.

# 7 Conclusion

This study focuses on learning and inference when only summarized data is available. The introduction of the sample quasi-likelihood facilitates them, including in cases where the likelihood is non-Gaussian. Our approach is straightforward to implement and incurs low computational costs. We have demonstrated the approximation errors of the marginal likelihood and posterior distribution, both theoretically and experimentally. This analysis highlights when our approach works well. In particular, the approximation performance depends on the granularity relative to the length scale of covariance functions. Experiments on spatial modeling show the impact of using summarized data instead of complete data. The feasibility of our approach was supported by the series of discussions in this study.

Limitations. The input approximation does not perform well when the granularity of summarized data is rough. Since our approach relies on the Laplace approximation applied to each summarized group, the prediction performance for non-Gaussian observations deteriorates as the number of data points decreases. Further investigation is needed for extending our approach to cases where statistics other than the sample mean are available. The proportional constants in eq. (11) and eq. (12) contain the hyperparameters of a covariance function, as shown in lemma B.3 and lemma B.4. The theoretical behavior of our approximation concerning the hyperparameters requires further analysis.

Future Works. Spatial modeling is valuable for decisionmaking in many fields, but a significant concern is protecting privacy. While our discussion in this study is primarily from the perspective of a data user, the design of datasets to maintain confidentiality, such as through differential privacy (Dwork et al. 2006), is also important. Our approach potentially contributes to encouraging discussions on the granularity of summarized data, considering the trade-offs between usability and confidentiality. Additionally, optimizing representative locations and data point assignments to improve approximation performance is an appealing direction for future work.