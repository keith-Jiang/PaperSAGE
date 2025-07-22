# Knockoffs Inference for Partially Linear Models with Automatic Structure Discovery

Hao Wang1, Biqin $\mathbf { S o n g ^ { 1 , 2 , * } }$ , Hao Deng1,2, Hong Chen1, 2

1College of Informatics, Huazhong Agricultural University, Wuhan 430062, China 2Engineering Research Center of Intelligent Technology for Agriculture, Ministry of Education, Wuhan, China hao wang $@$ webmail.hzau.edu.cn, biqin.song@mail.hzau.edu.cn, dengh@mail.hzau.edu.cn, chenh@mail.hzau.edu.cn

# Abstract

Partially linear models (PLM) have attracted much attention in the field of statistical machine learning. Specially, the ability of variable selection of PLM has been studied extensively due to the high requirement of model interpretability. However, few of the existing works concerns the false discovery rate (FDR) controllability of variable selection associated with PLM. To address this issue, we formulate a new Knockoffs Inference scheme for Linear And Nonlinear Discoverer (called KI-LAND), where FDR is controlled with respect to both linear and nonlinear variables for automatic structure discovery. For the proposed KI-LAND, theoretical guarantees are established for both FDR controllability and power, and experimental evaluations are provided to validate its effectiveness.

# Introduction

Linear and non-parametric models are two important classes statistical modeling tools, each with their own unique advantages. The linear model is simple and has fine interpretability, but it is limited by the linearity assumption. In contrast, the non-parametric model is much more flexible and can adapt to a wide range of functional forms present in the data, but it tends to be less interpretable. Therefore, a fundamental accuracy-interpretability tradeoff is need to be taken into account. Semi-parametric model inherits the interpretation of the linear model and flexibility of non-parametric model by allowing covariates to be linear or nonlinear, which can model complex data in some scientific fields, such as econometrics, social sciences, information sciences, biomedicine, and so on (Ruppert, Wand, and Carroll 2003; Ha¨rdle et al. 2004; Fuller 2009). Partially linear model (PLM) is the classic example of Semi-parametric model (Engle et al. 1986; Ha¨rdle, Liang, and Gao 2000). For the response variable $y \in$ $\mathcal { R }$ and input variables $( \mathbf { x , t } ) \ = \ ( x _ { 1 } , \ldots , x _ { p } , t _ { 1 } , \ldots , t _ { q } ) \ \in$ $\mathcal { R } ^ { p + q }$ , PLM is given by:

$$
y = b + \mathbf { x } ^ { \top } { \boldsymbol { \beta } } + f ( \mathbf { t } ) + \varepsilon ,
$$

where $b$ and $\beta$ are the intercept and the vector of coefficients for linear terms, respectively. $f$ is a nonlinear function from $\mathcal { R } ^ { q }$ to $\mathcal { R }$ , and $\varepsilon \sim \dot { \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ .

Variable selection for PLM has drawn extensive attention over the past two decades. It has usually been studied by using a specially designed penalty function under different smooth regression conditions, such as the component selection and smoothing operator penalty (Lin and Zhang 2006), the smoothly clipped absolute deviation penalty (Xie and Huang 2009), and the doubly penalized procedure (Wang et al. 2014a). Additional methods of variable selection for PLM can be found in the works of Wang et al. (2014b); Su and Cande\`s (2016); Lian, Zhao, and Lv (2019); Lv and Lian (2022). While these methods primarily focus on variable selection, they often lack control of error selections, such as the false discovery rate (FDR).

One very recent approach for FDR control is knockoffs which was first proposed by Barber and Cande\`s (2015). The key idea of knockoff is to construct fake variables which have a similar structure to the original ones but have no effect on the response, then computing an importance score for each variable and selecting those with higher scores than their fake copies. This work achieved effective FDR control in the context of Gaussian linear model, where the dimensionality $d$ does not exceed the sample size $n$ . The knockoff filter was subsequently extended to high-dimensional linear models with the help of data splitting and feature screening techniques (Barber and Cande\`s 2016). To achieve FDR control in general high dimensional nonlinear model, model-X knockoff (Cande\`s et al. 2018) was proposed. It can accommodate arbitrary dependence structure of the response variable $y$ on input variables $\mathbf { x }$ and bypass the need of calculating accurate $p$ -values. Fan et al. (2020) expanded model-X knockoff to nonparametric additive model. Building on this work, Xiaowu Dai and Li (2023) proposed a kernel knockoffs selection method. This allowed the knockoff approach to be applied in more flexible, non-linear settings, beyond the original linear model assumptions. Su et al. (2023) first applied the generalized knockoffs framework for variable selection in PLM, but it was a linear model by nature. As far as we know, few existing works studies about knockoffs inference for PLM with nonlinear settings.

The works described above have all accomplished FDR control, but a few studies have conducted comprehensive theoretical analysis for power as in (Fan et al. 2020; Weinstein et al. 2022). However, their power analysis was limited to the linear model setting, and a more general theoretical understanding of power across different models remains an important open challenge.

To fill the aforementioned gaps, a new variable selection procedure called Knockoffs Inference for Linear And Nonlinear Discoverer (KI-LAND) is proposed to study FDR control and power analysis for model (1). This is an attempt to extend knockoff framework to PLM, because there is no need for limit on $p$ and $q$ in (1). The main contributions of this paper are summarized as below:

i) PLM-based knockoff inference with automatic structure discovery. Knockoff framework is expanded to PLM, where FDR is controlled with respect to both linear and nonlinear variables for automatic structure discovery.   
ii) Theoretical guarantees and empirical effectiveness. Theoretical analyses for FDR and power are established. Meanwhile, the experiments validate the effectiveness of the proposed method.

# Methodology

# Problem Setup

This paper assume that all covariates are scaled to $[ 0 , 1 ]$ without loss of generality. Consider the i.i.d (independent and identically distributed) observation $\{ \mathbf { x } _ { i } , y _ { i } \} _ { i = 1 } ^ { n }$ , where $y _ { i }$ is the response, $\mathbf { x } _ { i } = ( x _ { i 1 } , . . . , x _ { i d } ) ^ { \top }$ is the input, and the true regression model is

$$
y _ { i } = b _ { 0 } + \sum _ { j \in \mathcal { T } _ { L } } x _ { i j } \beta _ { j } + \sum _ { j \in \mathcal { T } _ { N } } f _ { j } ( x _ { i j } ) + \sum _ { j \in \mathcal { T } _ { O } } 0 ( x _ { i j } ) + \varepsilon _ { i } .
$$

Here $b _ { 0 }$ is an intercept, $\beta _ { j }$ is the coefficient for linear term, $f _ { j }$ is an unknown unary nonlinear function, $0 ( x _ { i j } )$ is null function, and $\varepsilon _ { i }$ is noise. $\mathcal { T } _ { L } , \mathcal { T } _ { N } , \mathcal { T } _ { O }$ denote the index sets of nonzero linear effects, nonzero nonlinear effects, and null effects, respectively. We denote the total set as $\mathcal { T } =$ $\{ 1 , \dots , d \} = { \mathcal { T } } _ { L } \cup { \mathcal { T } } _ { N } \cup { \mathcal { T } } _ { O }$ and assume the three subgroups do not intersect each other. In applications, $\mathcal { T } _ { L } , \mathcal { T } _ { N } , \mathcal { T } _ { O }$ are generally unknown.

The model (2) can also be considered as a special additive model

$$
y _ { i } = b _ { 0 } + g _ { 1 } ( x _ { i 1 } ) + \cdot \cdot \cdot + g _ { d } ( x _ { i d } ) + \varepsilon _ { i } ,
$$

where each $g _ { j } , 1 \leq j \leq d$ is called the component function. Usually, we assume that each component function satisfies some smoothness conditions and $\bar { \mathbb { E } } [ g _ { j } ( x _ { i j } ) ] = 0 , j =$ $1 , \ldots , d$ .

Specially, we make assumption that $g _ { j } \in \mathcal { H } _ { j }$ , the secondorder Sobolev space on $\mathcal { X } _ { i } = [ 0 , 1 ]$ , that means, $\mathcal { H } _ { j } = \{ g : g$ and $g ^ { \prime }$ are absolutely continuous, $g ^ { \prime \prime } \in L ^ { 2 } [ 0 , 1 ] \bar  \}$ . In functional analysis, $\mathcal { H } _ { j }$ is a reproducing kernel Hilbert space (RKHS), when having the following norm:

$$
\left\| g _ { j } \right\| _ { \mathcal { H } _ { j } } ^ { 2 } = \{ \int _ { 0 } ^ { 1 } g _ { j } ( x ) d x \} ^ { 2 } + \{ \int _ { 0 } ^ { 1 } g _ { j } ^ { \prime } ( x ) d x \} ^ { 2 } + \int _ { 0 } ^ { 1 } \{ g _ { j } ^ { \prime \prime } ( x ) \} ^ { 2 } d x .
$$

The reproducing kernel (RK) in $\mathcal { H } _ { j }$ is $\begin{array} { r l } { R ( x , x ^ { \prime } ) } & { { } = } \end{array}$ $R _ { 0 } ( x , x ^ { \prime } ) { \bf \bar { \Psi } } + R _ { 1 } ( x , { \bf \bar { x } ^ { \prime } } )$ with $R _ { 0 } ( x , x ^ { \prime } ) \stackrel {  } { = } k _ { 1 } ( x ) \dot { k } _ { 2 } ( x ^ { \prime } )$ and $R _ { 1 } ( x , x ^ { \prime } ) = k _ { 2 } ( x ) k _ { 2 } ( x ^ { \prime } ) - k _ { 4 } ( x - x ^ { \prime } )$ , where $k _ { 1 } ( x ) \ =$ $\begin{array} { r } { x - \frac { 1 } { 2 } , k _ { 2 } ( x ) = \frac { 1 } { 2 } \{ k _ { 1 } ^ { 2 } ( x ) - \frac { 1 } { 1 2 } \} } \end{array}$ , and $\begin{array} { r } { k _ { 4 } ( x ) = \frac { 1 } { 2 4 } \{ k _ { 1 } ^ { 4 } ( \stackrel { \cdot } { x } ) - } \end{array}$ $\textstyle { \frac { 1 } { 2 } } k _ { 1 } ^ { 2 } ( x ) + { \frac { 7 } { 2 4 0 } } \}$ .naMlyosries idnetahielseitnw(oWraehfebraen1c9e8s3,)thaendsp(aCche $\mathcal { H } _ { j }$ has the following orthogonal decomposition:

$$
\mathcal { H } _ { j } = \{ 1 \} \oplus \mathcal { H } _ { 0 j } \oplus \mathcal { H } _ { 1 j } ,
$$

where $\{ 1 \}$ is the mean space, $\mathcal { H } _ { 0 j } = \{ g _ { j } : g _ { j } ^ { \prime \prime } ( x ) \equiv 0 \}$ is the linear contrast subspace, and $\begin{array} { r } { \mathcal { H } _ { 1 j } = \{ g _ { j } : \int _ { 0 } ^ { 1 } g _ { j } ( x ) \ d x = } \end{array}$ $\begin{array} { r } { 0 , \int _ { 0 } ^ { 1 } g _ { j } ^ { \prime } ( x ) \ d x = 0 , g _ { j } ^ { \prime \prime } \in \ L ^ { 2 } [ 0 , 1 ] \} } \end{array}$ is the nonlinear contrast space. Both $\mathcal { H } _ { 0 j }$ and $\breve { \varkappa } _ { 1 j }$ as subspace of $\mathcal { H } _ { j }$ are also RKHS, and have the reproducing kernels $R _ { 0 }$ and $R _ { 1 }$ respectively. Then, by the decomposition (4), for any function $g _ { j } \in \mathcal { H } _ { j }$ ,we can decompose it to linear and nonlinear components

$$
g _ { j } ( x _ { i j } ) = b _ { j } + \beta _ { j } ( x _ { i j } - \frac { 1 } { 2 } ) + g _ { 1 j } ( x _ { i j } ) ,
$$

$$
g ( \mathbf { x } _ { i } ) = b _ { 0 } + g _ { 1 } \left( x _ { i 1 } \right) + \cdot \cdot \cdot + g _ { d } \left( x _ { i d } \right) ,
$$

where $b _ { j }$ is the intercept for component function $g _ { j }$ , the term $\beta _ { j } ( x _ { i j } - \frac { 1 } { 2 } ) = \beta _ { j } k _ { 1 } ( x _ { i j } ) \in \mathcal { H } _ { 0 j }$ is the linear component and $g _ { 1 j } \left( \bar { x _ { i j } } \right) \in \mathcal { H } _ { 1 j }$ is the nonlinear component. $g ( \mathbf { x } _ { i } )$ is the function estimated in space $\mathcal { H }$ . Further more we have the decomposition of $\mathcal { H }$ :

$$
\begin{array} { c } { { \displaystyle \mathcal { H } = \bigoplus _ { j = 1 } ^ { d } \mathcal { H } _ { j } = \{ 1 \} \oplus \bigoplus _ { j = 1 } ^ { d } \mathcal { H } _ { 0 j } \oplus \bigoplus _ { j = 1 } ^ { d } \mathcal { H } _ { 1 j } } } \\ { { = \{ 1 \} \oplus \mathcal { H } _ { 0 } \oplus \mathcal { H } _ { 1 } , } } \end{array}
$$

where $\mathcal { H } _ { 0 } ~ = ~ \bigoplus _ { j = 1 } ^ { d } \mathcal { H } _ { 0 j }$ and $\mathcal { H } _ { j } ~ = ~ \oplus _ { j = 1 } ^ { d } \mathcal { H } _ { 1 j }$ . By the above decom sition, (6) can be rewri en as the following form:

$$
g ( \mathbf { x } _ { i } ) = b + \sum _ { j = 1 } ^ { d } \beta _ { j } k _ { 1 } \left( x _ { i j } \right) + \sum _ { j = 1 } ^ { d } g _ { 1 j } \left( x _ { i j } \right) ,
$$

where $\begin{array} { r } { b = b _ { 0 } + \sum _ { j = 1 } ^ { d } b _ { j } } \end{array}$

By the above analysis, we are able to distinguish between linear, nonlinear, and null variables by the following criteria:

$$
\begin{array} { r l } & { L i n e a r i n d e x s e t { : } \mathbb { Z } _ { L } = \{ j : \beta _ { j } \neq 0 , g _ { 1 j } \equiv 0 \} } \\ & { N o n l i n e a r i n d e x s e t { : } \mathbb { Z } _ { N } = \{ j : g _ { 1 j } \neq 0 \} , } \\ & { N u l l i n d e x s e t { : } \mathbb { Z } _ { O } = \{ j : \beta _ { j } = 0 , g _ { 1 j } \equiv 0 \} . } \end{array}
$$

# Knockoff Variable Construction

Now we will introduce how to construct a knockoff random variable. In (Cande\`s et al. 2018), the model-X knockoff random variable $\tilde { \mathbf { x } } = ( \tilde { x } _ { 1 } , \ldots , \tilde { x } _ { d } ) \in \mathcal { R } ^ { d }$ of a random variable $\mathbf { x } = ( x _ { 1 } , \ldots , x _ { d } ) \in { \mathcal { R } } ^ { d }$ is constructed with the following two properties (exchangeability and independence):

$$
\begin{array} { r l } { i ) } & { { } ( \mathbf { x } , \tilde { \mathbf { x } } ) \stackrel { d } { = } \big ( \mathbf { x } , \tilde { \mathbf { x } } \big ) _ { \mathrm { s w a p } ( \mathcal { S } ) } , \ \mathrm { f o r ~ a n y } \ \mathcal { S } \subseteq \{ 1 , \ldots , d \} , } \\ { i i ) } & { { } y \ \bot \ L \ \tilde { \mathbf { x } } \mid \mathbf { x } . } \end{array}
$$

Above, $\stackrel { d } { = }$ denotes identically distributed, the vector $( \mathbf { x } , \tilde { \mathbf { x } } ) _ { \mathrm { s w a p } ( S ) }$ is obtained from $( { \bf x } , \tilde { { \bf x } } )$ by swapping the entries $\mathbf { x } _ { j }$ and $\mathbf { \widetilde { x } } _ { j }$ for each $j \in S$ ; for instance, with $p = 3$ and $S = \{ \bar { 2 } , 3 \}$ ,

$$
\big ( x _ { 1 } , x _ { 2 } , x _ { 3 } , \tilde { x } _ { 1 } , \tilde { x } _ { 2 } , \tilde { x } _ { 3 } \big ) _ { \mathrm { s w a p } ( \{ 2 , 3 \} ) } \stackrel { d } { = } \big ( x _ { 1 } , \tilde { x } _ { 2 } , \tilde { x } _ { 3 } , \tilde { x } _ { 1 } , x _ { 2 } , x _ { 3 } \big ) .
$$

There are two classic data-based methods to generate knockoff variables. The first method is the model-X knockoffs, which constructs knockoffs by using the expectation and variance, like the follows:

$$
\mathbb { E } [ { \bf x } ] = \mathbb { E } [ \tilde { \bf x } ] ,
$$

$$
\operatorname { c o v } [ ( \mathbf { x } , { \tilde { \mathbf { x } } } ) ] = \left[ { \begin{array} { c c } { \Psi } & { \Psi - \mathrm { d i a g } ( s ) } \\ { \Psi - \mathrm { d i a g } ( s ) } & { \Psi } \end{array} } \right] ,
$$

where $\Psi$ is covariance matrix of $\mathbf { X }$ , $\mathrm { d i a g } ( s )$ is a diagonal matrix with all components of $s$ being positive and such that $\operatorname { c o v } [ ( X , { \tilde { X } } ) ]$ is positive definite. To obtain a good selection power, $\mathrm { d i a g } ( s )$ should be constructed as large as possible (Cande\`s et al. 2018). A variant of model-X knockoffs in (Barber, Cande\`s, and Samworth 2020) is generated by approximating the distribution of $\mathbf { X }$ .

The second method is deep knockoffs (Yaniv Romano and Cande\`s 2020), a sampling machine based on deep generative models, which approximates model-X knockoffs for unspecified data distributions. The key idea is to optimize the criterion evaluating the validity of the generated knockoffs by refining the knockoff sampling mechanism iteratively. This method leverages higher-order moments, enabling a better approximation of exchangeability.

In this article, we construct knockoffs with model-X knockoff when the input data approximately follows a multivariate normal distribution, and with deep knockoffs otherwise. In the next subsection, we will illustrate how to use knockoff procedure to implement future selection.

# Model Setup

Now we state the following regularization scheme originally proposed in (Zhang and Liu 2011):

$$
\begin{array} { l } { { \displaystyle \operatorname* { m i n } _ { g \in \mathcal { H } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left[ y _ { i } - g \left( \mathbf { x } _ { i } , \tilde { \mathbf { x } } _ { i } \right) \right] ^ { 2 } } } \\ { { \displaystyle ~ + \lambda _ { 1 } \sum _ { j = 1 } ^ { 2 d } w _ { 0 j } \left. \mathcal { P } _ { 0 j } g \right. _ { \mathcal { H } _ { 0 } } + \lambda _ { 2 } \sum _ { j = 1 } ^ { 2 d } w _ { 1 j } \left. \mathcal { P } _ { 1 j } g \right. _ { \mathcal { H } _ { 1 } } , } } \end{array}
$$

where

$$
\begin{array} { r } { g \left( \mathbf { x } _ { i } , \tilde { \mathbf { x } } _ { i } \right) = b + g _ { 1 } ( x _ { i 1 } ) + \cdot \cdot \cdot + g _ { d } ( x _ { i d } ) + ~ } \\ { g _ { d + 1 } ( \tilde { x } _ { i 1 } ) + \cdot \cdot \cdot + g _ { 2 d } ( \tilde { x } _ { i d } ) , ~ } \end{array}
$$

$\mathbf { x } _ { i } , \tilde { \mathbf { x } } _ { i } \in \mathcal { R } ^ { d }$ are original and knockoff variables, and $\mathcal { P } _ { 0 j } g , \mathcal { P } _ { 1 j } g$ project $\mathcal { H }$ into $\mathcal { H } _ { 0 j }$ and $\mathcal { H } _ { 1 j }$ , respectively. For the linear component, we set ∥P0jg∥ $\begin{array} { r l r } { \parallel \mathcal { P } _ { 0 j } \bar { g } \parallel _ { \mathcal { H } _ { 0 } } } & { { } = } & { | \beta _ { j } | , } \\ { . } & { { } } & { \parallel \mathcal { P } _ { 0 j } \bar { g } \parallel _ { \mathcal { H } _ { 0 } } } \end{array}$ , which is equivalent to the lasso penalty (Tibshirani 1996). In the nonlinear counterpart, we impose ∥P1jg∥ $\{ \int _ { 0 } ^ { 1 } \left[ g _ { 1 j } ^ { \prime \prime } ( x ) \right] ^ { 2 } d x \} ^ { 1 / 2 }$ to penalize the nonlinear component of $g _ { j }$ . As shown in (Zhang and Liu 2011),

$$
w _ { 0 j } = \frac { 1 } { | \tilde { \beta } _ { j } | ^ { \alpha } } , w _ { 1 j } = \frac { 1 } { \| \tilde { g } _ { 1 j } \| _ { 2 } ^ { \gamma } } , j = 1 , \dots , 2 d ,
$$

are weight parameters used in optimization computation, where $\bar { \tilde { \beta } } _ { j } , \tilde { g } _ { 1 j }$ are the decomposition of $\tilde { g }$ by (5), $| | \cdot | | _ { 2 }$ is $L _ { 2 }$ norm, $\alpha \geq 3 / 2$ and $\gamma \geq 3 / 2$ are hyperparameters to tune.

This weight design ensures that components deemed less significant are penalized more heavily, thereby shrinking them towards zero. Conversely, components that are crucial to the function are assigned lighter penalties, which helps in preserving their non-zero values during the selection process. This nuanced weighting strategy aims to balance the preservation of essential variables with the suppression of less relevant elements. The similar strategies have been used in linear models (Zhang and Lu 2007; Zou 2006; Wang, Li, and Jiang 2007) and SS-ANOVA model (Storlie et al. 2011).

To obtain the weights, we first need to get an initial estimation of the function $g$ , which will then allow us to determine the values of $\tilde { \beta } _ { j }$ and $\tilde { g } _ { 1 j }$ . For this initial estimation, we choose SS-ANOVA model (9):

$$
\operatorname* { m i n } _ { g \in \mathcal { H } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left[ y _ { i } - g \left( \mathbf { x } _ { i } , \tilde { \mathbf { x } } _ { i } \right) \right] ^ { 2 } + \lambda \sum _ { j = 1 } ^ { 2 d } \left. \mathcal { P } _ { 1 j } g \right. _ { \mathcal { H } _ { 1 } } ^ { 2 } .
$$

Now we could consider to solve (8). As referenced in (Zhang and Liu 2011), we don’t solve (8) directly, but solve an equivalent and more convenient problem :

$$
\begin{array} { r l r } { \displaystyle \operatorname* { m i n } _ { \theta \geq \mathbf { 0 } , g \in \mathcal { H } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } [ y _ { i } - g ( \mathbf { x } _ { i } , \tilde { \mathbf { x } } _ { i } ) ] ^ { 2 } + \lambda _ { 1 } \sum _ { j = 1 } ^ { 2 d } w _ { 0 j }  \mathcal { P } _ { 0 j } g  } & { } & \\ { \displaystyle +  \tau _ { 0 } \sum _ { j = 1 } ^ { 2 d } \theta _ { j } ^ { - 1 } w _ { 1 j }  \mathcal { P } _ { 1 j } g  _ { \mathcal { H } _ { 1 } } ^ { 2 } + \tau _ { 1 } \sum _ { j = 1 } ^ { 2 d } w _ { 1 j } \theta _ { j } , } & { } & \end{array}
$$

where $\tau _ { 0 }$ is a fixed constant, $( \lambda _ { 1 } , \tau _ { 1 } )$ are hyperparameters to tune. The following lemma states that optimization problems (8) and (10) are equivalent (Zhang and Liu 2011).

Lemma 1. Set $\tau _ { 1 } = \lambda _ { 2 } ^ { 2 } / \left( 4 \tau _ { 0 } \right)$ . (i) If $\hat { g }$ minimizes (8), set $ { \hat { \theta } } _ { j } ~ = ~ \tau _ { 0 } ^ { 1 / 2 } \tau _ { 1 } ^ { - 1 / 2 } \| \mathcal { P } _ { 1 j }  { \hat { g } } \|$ ; then the pair $( \hat { \pmb \theta } , \hat { g } )$ minimizes $( I O ) . ( i i ) I f ( \hat { \pmb { \theta } } , \hat { g } )$ minimizes $( l O )$ , then $\hat { g }$ minimizes (8).

Supposing that the input $\{ { \bf x } _ { i } \} _ { i = 1 } ^ { n }$ is given, we can use the representation theory to get the solution of (10):

$$
\begin{array} { l } { { \displaystyle { \hat { g } } ( { \bf x } , { \tilde { \bf x } } ) = { \hat { b } } + \sum _ { j = 1 } ^ { d } { \hat { \beta } } _ { j } k _ { 1 } \left( x _ { j } \right) + \sum _ { j = d + 1 } ^ { 2 d } { \hat { \beta } } _ { j } k _ { 1 } \left( { \tilde { x } } _ { j } \right) } } \\ { ~ + \sum _ { j = 1 } ^ { d } { \hat { \theta } } _ { j } w _ { 1 j } ^ { - 1 } \sum _ { i = 1 } ^ { n } { \hat { c } } _ { i } R _ { 1 j } \left( x _ { i j } , x _ { j } \right) }  \\ { { ~ + \sum _ { j = d + 1 } ^ { 2 d } { \hat { \theta } } _ { j } w _ { 1 j } ^ { - 1 } \sum _ { i = 1 } ^ { n } { \hat { c } } _ { i } { \tilde { R } } _ { 1 j } \left( { \tilde { x } } _ { i j } , { \tilde { x } } _ { j } \right) } . } \end{array}
$$

The expression (11) indicates that linearity or nonlinearity of $g _ { j }$ is determined by the case where two parameters $\hat { \beta } _ { j }$ and $\hat { \theta } _ { j }$ are 0 or not. Therefore, two parameters are used to select linear and nonlinear components in (7). It is clearly that $\hat { g } _ { 1 j } = 0$ when $\hat { \theta } _ { j } = 0$ , we define following index sets:

$$
\begin{array} { r l } & { \hat { \mathcal { Z } } _ { L } = \left\{ j = 1 , \dots , 2 d : \hat { \beta } _ { j } \neq 0 , \hat { \theta } _ { j } = 0 \right\} , } \\ & { \hat { \mathcal { Z } } _ { N } = \left\{ j = 1 , \dots , 2 d : \hat { \theta } _ { j } \neq 0 \right\} , } \\ & { \hat { \mathcal { Z } } _ { O } = \left\{ j = 1 , \dots , 2 d : \hat { \beta } _ { j } = 0 , \hat { \theta } _ { j } = 0 \right\} . } \end{array}
$$

In the next subsection, we will show how to construct the important score and knockoff statistic which help us to select linear and nonlinear components and control the false discovery rate (FDR).

# KI-LAND

In this article, we employ a subsampling strategy analogous to the one described in (Meinshausen and Bu¨hlmann 2010; Du¨mbgen, Samworth, and Schuhmacher 2013). Let $I \subset \{ 1 , \ldots , n \}$ denote the subsample indices with size $\lfloor n / 2 \rfloor$ , where $\lfloor \cdot \rfloor$ means round down. we make set $I$ as training samples on model (9) and model (10). Then, we will get the following index sets:

$$
\begin{array} { c } { { \hat { S } ( I ) ^ { L } = \left\{ j = 1 , \ldots , 2 d : \hat { \beta } _ { j } \neq 0 \right\} , } } \\ { { \hat { S } ( I ) ^ { N } = \left\{ j = 1 , \ldots , 2 d : \hat { \theta } _ { j } \neq 0 \right\} . } } \end{array}
$$

Where ${ \hat { S } } ( I ) ^ { L }$ and $\hat { S } ( I ) ^ { N }$ are the index sets of the linear and nonlinear components based on training data $I$ , respectively. We repeat subsampling subset $I$ without replacement $U$ times. The probabilities of the linear component and the nonlinear component being selected can be represented as:

$$
\Pi _ { j } ^ { L } = \mathcal { P } ( j \in \hat { \mathcal { S } } ( I ) ^ { L } ) , \Pi _ { j } ^ { N } = \mathcal { P } ( j \in \hat { \mathcal { S } } ( I ) ^ { N } ) , j = 1 , \dotsc , 2 d .
$$

Where the symbol $\mathcal { P }$ denotes the probability. The selection probabilities $\Pi _ { j } ^ { L }$ and $\Pi _ { j } ^ { N }$ are defined as important scores that reflect the importance of linear and nonlinear components of the original and knockoff variables, respectively. Specifically, the larger the probability value, the more important the corresponding component is.

Due to the selection probabilities are unknown, they can be estimated accurately by the empirical selection probabilities. More specifically, repeating the above subsampling and estimating model (8) and (10) $U$ times, we denote $I _ { u }$ as the uth time subsampling training set. Then, $\Pi _ { j } ^ { L }$ and $\Pi _ { j } ^ { N }$ in (14) can be transform into the follows:

$$
\begin{array} { l } { { \displaystyle { \widehat { \Pi } } _ { j } ^ { L } = \frac { 1 } { U } \sum _ { u = 1 } ^ { U } \mathbf { 1 } \left( j \in { \widehat { \mathcal { S } } } \left( I _ { u } \right) ^ { L } \right) , \quad j = 1 , \ldots , 2 d , } } \\ { { \displaystyle { \widehat { \Pi } } _ { j } ^ { N } = \frac { 1 } { U } \sum _ { u = 1 } ^ { U } \mathbf { 1 } \left( j \in { \widehat { \mathcal { S } } } \left( I _ { u } \right) ^ { N } \right) , \quad j = 1 , \ldots , 2 d . } } \end{array}
$$

Based on the selection probabilities, we can define the knockoff statistics for linear and nonlinear components of the original variable $\mathbf { x } = ( x _ { 1 } , \ldots , x _ { d } ) \in { \mathcal { R } } ^ { d }$ .

$$
\Delta _ { j } ^ { L } = \Pi _ { j } ^ { L } - \Pi _ { j + d } ^ { L } , \Delta _ { j } ^ { N } = \Pi _ { j } ^ { N } - \Pi _ { j + d } ^ { N } , j = 1 , \dots , d .
$$

The higher the value of knockoff statistic $\Delta _ { j } ^ { L }$ or $\Delta _ { j } ^ { N }$ , the more important the $j$ th linear component or nonlinear component of original variables is.

The most important step is to control FDR of the selected linear and nonlinear components. Given the target nominal FDR $q$ , we need to choose data-dependent threshold values $T ^ { L }$ and $T ^ { N }$ for linear and nonlinear components selection respectively. There are two ways to construct threshold, one is the classic knockoff filter (Barber and Cande\`s 2015; Xiaowu Dai and Li 2023) to choose thresholds as the follows:

$$
T = \operatorname* { m i n } \left\{ t \in \{ | \Delta _ { j } | : | \Delta _ { j } | > 0 \} : \frac { \# \left\{ j : \Delta _ { j } \leqslant - t \right\} } { \# \left\{ j : \Delta _ { j } \geqslant t \right\} } \leqslant q \right\} .
$$

$\Delta _ { j }$ is the knockoff statistic and we set $T = \infty$ if the above set is null set. Another one is a more conservative knockoff filter but being used commonly as follows:

$$
T _ { + } = \operatorname* { m i n } \left\{ t \in \{ | \Delta _ { j } | : | \Delta _ { j } | > 0 \} : \frac { \# \{ j : \Delta _ { j } \leqslant - t \} + 1 } { \# \left\{ j : \Delta _ { j } \geqslant t \right\} } \leqslant q \right\} .
$$

After threshold values $T ^ { L }$ and $T ^ { N }$ are constructed via (17) or (18), the final sets of selected linear and nonlinear components are $\hat { S } ^ { L }$ and $\hat { S } ^ { N }$ respectively:

$$
\begin{array} { r l r } & { } & { \hat { \mathcal { S } } ^ { L } = \left\{ j \in \{ 1 , \dots , p \} : \Delta _ { j } ^ { L } \geqslant T ^ { L } \right\} , } \\ & { } & { \hat { \mathcal { S } } ^ { N } = \left\{ j \in \{ 1 , \dots , p \} : \Delta _ { j } ^ { N } \geqslant T ^ { N } \right\} . } \end{array}
$$

Now we can choose linear variables and nonlinear variables based on (19). The index set (12) can be redefined as follow:

$$
\begin{array} { r l } & { \hat { \mathcal { Z } } _ { L } = \left\{ j = 1 , \dots , d : j \in \hat { \mathcal { S } } ^ { L } , j \notin \hat { \mathcal { S } } ^ { N } \right\} , } \\ & { \hat { \mathcal { Z } } _ { N } = \left\{ j = 1 , \dots , d : j \in \hat { \mathcal { S } } ^ { N } \right\} , } \\ & { \hat { \mathcal { Z } } _ { O } = \left\{ j = 1 , \dots , d : j \notin \hat { \mathcal { S } } ^ { L } , j \notin \hat { \mathcal { S } } ^ { N } \right\} . } \end{array}
$$

# Algorithm

The knockoff procedure described before can be summarized in 10 steps, please see Algorithm 1. Step 1 is to generate the knockoff variables via model-X knockoff or deep knockoffs. Steps 2 to 5 are carried out $U$ times repeatedly over a number of subsampling replications. By doing this, it not only can calculate the importance scores, but can also reduce the impact of randomness. Steps 6 to 9 are for variables selection and the final step is output.

# Theoretical Analysis

In this section, we build theoretical guarantees for the KILAND procedure in terms of controlling FDR and power asymptomatic property. All proofs can be seen in supplementary material.

# FDR Analysis

In this subsection, theoretical analysis shows that KI-LAND procedure is able to control FDR at any given nominal level $q$ and sample size. In subsection Problem Setup, we employ kernel function method to learn the function $g _ { j }$ in (3) of each dimension’s variable, and build a mapping between the RKHS and the original function space. By construction, the kernel matrix of the knockoff variables is designed to mimic the structure of the kernel matrix of the original variables, despite the knockoffs being unassociated with the response when conditioned on the original variables (Xiaowu Dai and Li 2023). For the stability of the results, we use a subsampling technique. Finally, KI-LAND controls the

Input: training data $\{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ , the number of subsampling replications $U$ and the nominal FDR level $q \in ( 0 , 1 )$ . Output: $\hat { \mathcal { T } } _ { L } , \hat { \mathcal { T } } _ { N } , \hat { \mathcal { T } } _ { O }$ .

1: Step 1: construct the knockoff $\{ \tilde { \mathbf { x } } _ { \mathbf { i } } \} _ { i = 1 } ^ { n }$ via model-X knockoff or deep knockoffs.   
2: for $u = 1$ to $U$ do   
3: Step 2: sampling without replacement to obtain a training set $I _ { u } \subset \{ 1 , \ldots , n \}$ of size $\lfloor n / 2 \rfloor$ .   
4: Step 3: estimate the weights $w _ { 0 j }$ and $w _ { 1 j }$ by the SSANOVA model (9) .   
5: Step 4: solve the equivalence problem (10) to get $\hat { \beta } _ { j }$ and $\hat { \theta } _ { j }$ .   
6: Step 5: record the selected index sets (13) of the linear and nonlinear components.   
7: end for   
8: Step 6: compute the important scores by (15), i.e., the empirical selection frequency based on $U$ times estimation of model (9) and (10).   
9: Step 7: compute knockoff statistics based on the important scores $\dot { \Delta } _ { j } ^ { L }$ and $\Delta _ { j } ^ { N }$ via (16).   
10: Step 8: select data-dependent thresholds $T ^ { L }$ and $T ^ { N }$ by (17) or (18).   
11: Step 9: choose the final sets of linear and nonlinear components via (19).   
12: Step 10: output linear variable set $\hat { \mathcal { T } } _ { L }$ , nonlinear variable set $\hat { \mathcal { T } } _ { N }$ and null variable set $\scriptstyle { \hat { \mathcal { I } } } _ { O }$ .

finite-sample FDR, the same to the existing knockoff methods (Barber and Cand\`es 2015; Cande\`s et al. 2018).

We first show that the knockoff statistics $\Delta _ { j } ^ { L }$ and $\Delta _ { j } ^ { N }$ in (16) have symmetric properties for a null linear or nonlinear component $j \in \hat { S } ^ { L }$ or $\hat { S } ^ { N }$ respectively. The symmetric property of the null components are crucial for the KI-LAND procedure, which then selects a data-dependent threshold while controlling the FDR (Barber and Cande\`s 2015).

Assumption 1 (PLM decomposition). For input $\mathrm { ~ \bf ~ x ~ } = $ $( x _ { 1 } , \hdots , \overline { { \ u { x _ { d } } } } ) \in \mathcal { R } ^ { d }$ , the response $y$ can be represented as follow by the decomposition (5)

$$
\begin{array} { l } { { \displaystyle y = y ^ { L } + y ^ { N } + \varepsilon ^ { L } + \varepsilon ^ { N } } } \\ { { \displaystyle \quad = \sum _ { j = 1 } ^ { d } \beta _ { j } k _ { 1 } \left( x _ { j } \right) + \sum _ { j = 1 } ^ { d } g _ { 1 j } \left( x _ { j } \right) + \varepsilon ^ { L } + \varepsilon ^ { N } . } } \end{array}
$$

Assumption 2 (Irrepresentable Condition in RKHS). For $j \in \{ 1 , \bar { \ } . . . , d \} , x _ { j } \ne \sum _ { k = 1 ; k \ne j } ^ { d } \beta _ { k } x _ { k }$ , for any $\beta _ { k } \in \mathcal { R }$ ; and $\begin{array} { r } { g _ { j } \left( x _ { j } \right) \ne \sum _ { k = 1 ; k \ne j } ^ { d } g _ { k } \left( x _ { k } \right) } \end{array}$ , for any functions $g _ { k } \in \mathcal { H } _ { 1 }$ .

Proposition 1. Suppose Assumption $\jmath$ and 2 hold. i) $\bar { \boldsymbol { j } } \in  { \mathcal { S } } _ { 0 } ^ { L }$ , if and only $i f \beta _ { j } = 0$ , for $j = 1 , \ldots , d$ ; ii) $j \in \mathcal { S } _ { 0 } ^ { N }$ , if and only if $g _ { j } = 0$ , for $j = 1 , \ldots , d .$

Lemma 2. Take any subset $s$ of null components, then $i )$ For $ { \boldsymbol { S } } \subset  { \boldsymbol { S } } _ { 0 } ^ { L }$ , $[ \mathbf { X } , \tilde { \mathbf { X } } ] \mid Y ^ { L } \stackrel { \mathrm { d } } { = } [ \mathbf { X } , \tilde { \mathbf { X } } ] _ { s w a p ( \mathcal { S } ) } \mid Y ^ { L }$ ; ii) For ${ \mathcal { S } } \subset S _ { 0 } ^ { N }$ , $[ \mathbf { R } , \tilde { \mathbf { R } } ] \mid Y ^ { N } \overset { \mathrm { d } } { = } [ \mathbf { R } , \tilde { \mathbf { R } } ] _ { s w a p ( S ) } \mid Y ^ { N }$ .

Remark 1. $[ \mathbf { X } , \tilde { \mathbf { X } } ] \in \mathcal { R } ^ { n \times 2 d }$ is the gram matrix of original and knockoff variables. $[ \mathbf { R } , \tilde { \mathbf { R } } ] \in \mathcal { R } ^ { n \times 2 n d }$ is consist of kernel matrix of original and knockoff variables. For more details, please see the appendix. $S _ { 0 } ^ { L }$ , $S _ { 0 } ^ { N }$ are true nulls of nonlinear and nonlinear component index set. $S ^ { L } , S ^ { N }$ are true linear and nonlinear component index set.

Lemma 3 (Sign-flip property for the nulls). Suppose Assumptions $( l )$ and (2) hold. Let $( \epsilon _ { 1 } , \hdots , \epsilon _ { d } )$ be a set of independent random variables,   
$i ,$ such that $\epsilon _ { j } ~ = ~ 1$ $f j \in \ S ^ { L }$ , and $\epsilon _ { j } ~ = ~ \pm 1$ with equal probability $I / 2 i f j \in S _ { 0 } ^ { L }$ Then, $\left( \Delta _ { 1 } ^ { L } , \ldots , \Delta _ { d } ^ { L } \right) \stackrel { d } { = }$ $\left( \Delta _ { 1 } ^ { L } \cdot \epsilon _ { 1 } , \dots , \Delta _ { d } ^ { L } \cdot \epsilon _ { d } \right)$ .   
$i i _ { , }$ ) such that $\epsilon _ { j } ~ = ~ 1 ~ i f ~ j \in ~ S ^ { N } $ , and $\epsilon _ { j } ~ = ~ \pm 1$ with equal probability $I / 2 i f j \in S _ { 0 } ^ { N }$ Then, $\left( \Delta _ { 1 } ^ { N } , \ldots , \Delta _ { d } ^ { N } \right) \stackrel { d } { = }$ $\left( \Delta _ { 1 } ^ { N } \cdot \epsilon _ { 1 } , \dots , \Delta _ { d } ^ { N } \cdot \epsilon _ { d } \right)$ .

The following part introduces that KI-LAND procedure controls the false discovery for any sample size well. The results which are established without the need for prior knowledge of the noise level are robust to the distribution and number of predictor variables. we adopt a modified FDR (mFDR) to approximate FDR when using the threshold $T$ (17) (Barber and Cande\`s 2015). For a true set $s$ , selected set $\hat { \boldsymbol { S } }$ and true nulls $ { \boldsymbol { S } } _ { 0 }$ , FDR and mFDR are defined as,

$$
\mathrm { F D R } ( \hat { S } ) = \mathbb { E } \left[ \frac { \vert \hat { S } \cap S _ { 0 } \vert } { \vert \hat { S } \vert \vee 1 } \right] , \mathrm { m F D R } ( \hat { S } ) = \mathbb { E } \left[ \frac { \vert \hat { S } \cap S _ { 0 } \vert } { \vert \hat { S } \vert + 1 / q } \right] .
$$

Theorem 1 (FDR control of KI-LAND). For any $q \in ( 0 , 1 )$ and any sample size $n$ , choose the threshold $\smash { T ^ { L } \ > \ 0 }$ and $T ^ { N } ~ > ~ 0$ via $T$ (17) . Then $\mathbf { m } \mathbf { F } \mathbf { D } \mathbf { R } ( \widehat { \mathcal { T } } _ { L } ) \ \leq \ q$ and $\mathbf { m } \mathbf { F } \mathbf { D } \mathbf { R } ( \widehat { \mathcal { T } } _ { N } ) \leq q ,$ ; Meanwhile, choose the thbreshold $T ^ { L } >$ 0 and $T ^ { N } ~ > ~ 0$ via $T _ { + }$ (18). Then $\mathbf { F D R } ( \widehat { \mathcal { T } } _ { L } ) \ \leq \ q$ and $\mathbf { F D R } ( \widehat { \mathcal { T } } _ { N } ) \leq q$ .

Remarbk 2. Theorem 1 controls the valid FDR with no restriction on the dimension $d$ and the sample size $n$ .

# Power Analysis

As far as we know, the current literature lacks thorough theoretical analysis about power for knockoff methods, except (Fan et al. 2020; Weinstein et al. 2022) which analyzed the power for linear regressions under the model-X knockoff framework and for thresholded Lasso knockoffs respectively. In this section, we will show that the KI-LAND procedure has a full power as the $n \to \infty$ . To obtain the theoretical results, some basic regularity conditions are introduced.

Assumption 3 (Minimum signal). For some slowly diverging sequence $\begin{array} { r l r l } { \kappa _ { n } } & { { } \to } & { \infty } & { } \end{array}$ , as $\begin{array} { r l r l } { n } & { { } \to } & { \infty } & { } \end{array}$ , let $\eta ~ \equiv ~ - c _ { \eta } \bar { \left\{ { n ^ { - \beta / ( 2 \beta + 1 ) } } \ + \ \left[ ( \log d ) / n \right] ^ { 1 / 2 } \right\} }$ .→For∞some constant $\begin{array} { r l r } { c _ { \eta } } & { { } > } & { 0 } \end{array}$ , such that, $\begin{array} { r l } { \operatorname* { i n i n } _ { j \in \cal S ^ { L } } | \beta _ { j } | } & { { } \ge } \end{array}$ $\begin{array} { r } { \kappa _ { n } \{ ( \log d ) / n \} ^ { 1 / 2 } , \operatorname* { m i n } _ { j \in S ^ { N } } \| g _ { 1 j } \left( x _ { j } \right) \| _ { 2 } \geqslant \kappa _ { n } \eta } \end{array}$ , where the $R K H S \ \mathcal { H } _ { 1 j }$ is embedded to a βth order Sobolev space with $\beta > 1$ .

In order to introduce the following conditions, some notations are given. Σ = [R, R˜ ] ∈ Rn×2nd, ΣSN ∈ Rn×2n|SN| is the design matrix consisted of the $j$ th and $j + d \mathrm { t h }$ columns of $\pmb { \Sigma }$ for $j \in \mathcal S ^ { N }$ . The same to other matrixs $[ \mathbf { X } , \tilde { \mathbf { X } } ] _ { S ^ { L } } ^ { \top } , \mathbf { X } _ { S ^ { L } }$ . Assumption 4 (Minimal eigenvalue). Suppose here is $a$ constant $C _ { \mathrm { m i n } } > 0$ , such that the minimal eigenvalue $\lambda _ { m i n }$ of matrix $\mathbb { E } [ n ^ { - 1 } [ \mathbf { X } , \tilde { \mathbf { X } } ] _ { \mathcal { S } ^ { L } } ^ { \top } [ \mathbf { X } , \tilde { \mathbf { X } } ] _ { \mathcal { S } ^ { L } } ]$ and $\mathbb { E } [ n ^ { - 1 } \Sigma _ { S ^ { N } } ^ { \top } \Sigma _ { S ^ { N } } ]$ satisfies that,

$$
\begin{array} { r l } & { \lambda _ { \operatorname* { m i n } } \bigl ( \mathbb { E } \bigl [ n ^ { - 1 } [ \mathbf { X } , \tilde { \mathbf { X } } ] _ { \mathcal { S } ^ { L } } ^ { \top } [ \mathbf { X } , \tilde { \mathbf { X } } ] _ { \mathcal { S } ^ { L } } \bigr ] \bigr ) \geqslant C _ { \operatorname* { m i n } } , } \\ & { \lambda _ { \operatorname* { m i n } } \bigl ( \mathbb { E } \bigl [ n ^ { - 1 } \Sigma _ { \mathcal { S } ^ { N } } ^ { \top } \Sigma _ { \mathcal { S } ^ { N } } \bigr ] \bigr ) \geqslant C _ { \operatorname* { m i n } } . } \end{array}
$$

Assumption 5 (Mutual incoherence). Suppose there exists a constant $0 \leqslant \xi < 1$ , such that,

$$
\begin{array} { r l } & { \underset { j \not \in \mathcal { S } ^ { L } } { \operatorname* { m a x } } \| [ \mathbf { X } , \tilde { \mathbf { X } } ] _ { j } ^ { \mathrm { T } } [ \mathbf { X } _ { \mathcal { S } ^ { L } } , \tilde { \mathbf { X } } _ { \mathcal { S } ^ { L } } ] } \\ & { ( [ \mathbf { X } _ { \mathcal { S } ^ { L } } , \tilde { \mathbf { X } } _ { \mathcal { S } ^ { L } } ] ^ { \mathrm { T } } [ \mathbf { X } _ { \mathcal { S } ^ { L } } , \tilde { \mathbf { X } } _ { \mathcal { S } ^ { L } } ] ) ^ { - 1 } \| _ { 2 } \leq 1 - \xi . } \end{array}
$$

Assumption 6 (Mutual incoherence). Suppose $d < e ^ { n }$ and there exists a constant $0 \leqslant \xi \_ { \Sigma } < 1$ , such that,

$$
\begin{array} { r l } & { \underset { j \not \in S ^ { N } } { \operatorname* { m a x } } \| \{ \boldsymbol { \Sigma } _ { j } \} ^ { \top } \boldsymbol { \Sigma } _ { \mathcal { S } ^ { N } } [ \boldsymbol { \Sigma } _ { \mathcal { S } ^ { N } } ^ { \top } \boldsymbol { \Sigma } _ { \mathcal { S } ^ { N } } ] ^ { - 1 } \| _ { 2 } \leqslant \xi _ { \Sigma } , } \\ & { \frac { \xi _ { \Sigma } \sqrt { | \mathcal { S } ^ { N } | } + 1 } { \lambda _ { 2 } } \eta + \xi _ { \Sigma } \sqrt { | \mathcal { S } ^ { N } | } < 1 , } \end{array}
$$

where $\Sigma _ { j }$ is the the jth columns of $\pmb { \Sigma }$ .

Remark 3. All these assumptions are reasonable and used frequently in current literature. In detail, Assumption 3 is a minimum regulatory effect condition which ensures that the solution of LAND model (8) does not overlook a great portion of important variables. We can find the same assumption in (Zhao and Yu 2006; Ravikumar, Wainwright, and Lafferty 2010). Assumption 4 is the minimal eigenvalue condition, which states that the Gram matrix of the important set on the augment design matrix is invertible. Similar assumptions have been imposed in Lasso regressions (Ravikumar, Wainwright, and Lafferty 2010; Raskutti, J Wainwright, and Yu 2012; Fan et al. 2020). Assumptions 5 and 6 indicate that the correlation between the true signals and nulls should not exert an strong effect (Zhao and Yu 2006; Ravikumar, Wainwright, and Lafferty 2010; Wainwright 2019).

With these regularity conditions in place, we are now ready to characterize the statistical power of the KI-LAND procedure. For a true set $s$ and a selected set $\hat { \boldsymbol { S } }$ , the power is defined as,

$$
\mathrm { P o w e r } ( \widehat { S } ) = \mathbb { E } \left[ \frac { \vert \widehat { S } \cap \widehat { S } \vert } { \vert S \vert \vee 1 } \right] .
$$

Theorem 2 (Power of KI-LAND). Suppose Assumptions 3- 6 hold, Then, the oracle KI-LAND procedure has a asymptomatic property that $\mathrm { P o w e r } ( \hat { \mathcal { T } } _ { L } ) \stackrel { - } {  } 1$ and Power $( { \hat { \mathcal { I } } } _ { N } ) \to$ 1, as $n \to \infty$ .

Remark 4. Theorems 1 and 2 show that the KI-LAND procedure can simultaneously provide good false discovery rate (FDR) control and statistical power properties under some regularity settings. In contrast to Theorem 1, Theorem 2 holds for $d < n$ and $n < d < e ^ { n }$ .

# Experiments

In this section, we conduct experiments to evaluate the empirical performance of the proposed KI-LAND, where various data settings are considered including the different correlation, the different sample size and the different dimension of variables. In all cases, we set $L = 1 0 0$ , and select the tuning parameter $( \lambda _ { 1 } , \tau _ { 0 } , \tau _ { 1 } )$ by cross-validation. And the target FDR of $\hat { \mathcal { T } } _ { L }$ and $\hat { \mathcal { T } } _ { N }$ are 0.2.

The simulated data is generated to demonstrate the empirical performance of the KI-LAND procedure. We compare the KI-LAND with LAND (Zhang and Liu 2011), KKO (Xiaowu Dai and Li 2023), and Model-X (Cande\`s et al. 2018). The following functions on [0, 1] are used for building components of response $y$ :

$$
\begin{array} { r l } & { f _ { 1 } ( x ) = x , \quad f _ { 2 } ( x ) = \cos ( 2 \pi x ) , } \\ & { f _ { 3 } ( x ) = \sin ( 2 \pi x ) / ( 2 - \sin ( 2 \pi x ) ) , } \\ & { f _ { 4 } ( x ) = \cos ( 2 \pi x ) / ( 2 - \cos ( 2 \pi x ) ) , } \\ & { f _ { 5 } ( x ) = 0 . 1 \sin ( 2 \pi x ) + 0 . 2 \cos ( 2 \pi x ) + 0 . 3 ( \sin ( 2 \pi x ) ) ^ { 2 } } \\ & { \qquad + 0 . 4 ( \cos ( 2 \pi x ) ) ^ { 3 } + 0 . 5 ( \sin ( 2 \pi x ) ) ^ { 3 } , } \\ & { f _ { 6 } ( x ) = ( 3 x - 1 ) ^ { 2 } , \quad f _ { 7 } ( x ) = ( 3 x - 1 ) ^ { 3 } . } \end{array}
$$

where $f _ { 1 }$ is a linear function, $f _ { 2 } , . . . , f _ { 5 }$ are pure nonlinear function, and $f _ { 6 } , f _ { 7 }$ are consist of linear and nonlinear functions. For $\mathbf { x } = ( x _ { 1 } , \ldots , x _ { d } ) \in { \mathcal { R } } ^ { d }$ , $y$ is generated by the model:

$$
\begin{array} { c } { { y = \displaystyle \sum _ { j = 1 } ^ { 1 0 } f _ { 1 } ( x _ { j } ) + f _ { 2 } ( x _ { 1 1 } ) + 3 f _ { 2 } ( x _ { 1 2 } ) + f _ { 3 } ( x _ { 1 3 } ) } } \\ { { + \left. 2 f _ { 3 } ( x _ { 1 4 } ) + f _ { 4 } ( x _ { 1 5 } ) + 4 f _ { 4 } ( x _ { 1 6 } ) + f _ { 5 } ( x _ { 1 7 } ) \right. } } \\ { { + \left. f _ { 5 } ( x _ { 1 8 } ) + f _ { 6 } ( x _ { 1 9 } ) + f _ { 7 } ( x _ { 2 0 } ) + \varepsilon , \right. } } \end{array}
$$

where $\varepsilon \sim \mathcal { N } ( 0 , 1 )$ , the designed matrix $\mathbf { X } \ \sim \ { \mathcal { N } } ( \mathbf { 0 } , { \boldsymbol { \Omega } } )$ , $\pmb { \Omega } = \left( \rho ^ { | i - j | } \right) _ { 1 \leq i , j \leq p } .$ , $\rho = \operatorname { c o r r } \left( X _ { i } , X _ { j } \right)$ for all $i \neq j$ . It is clearly that $x _ { 1 } , . . . , x _ { 1 0 }$ are the linear variables, $x _ { 1 1 } , . . . , x _ { 2 0 }$ are the nonlinear variables, and the rest is $d - 2 0$ noise variables. So $| \mathcal { T } _ { L } | = 1 0$ , $\vert \mathcal { I } _ { N } \vert = 1 0$ , $| \mathcal { T } _ { O } | = d - 2 0$ .

For the different variable dimensions $\begin{array} { r l } { d } & { { } = } \end{array}$ $\{ 5 0 , 1 0 0 , 2 0 0 , 4 0 0 \}$ , we set $n \ = \ 2 0 0$ , $\rho \ = \ 0 . 2$ , these cases examine whether the proposed method performs well in terms of FDR and statistical power when the important variables become more sparse. For the different sample size $n = \{ 5 0 , 1 0 0 , 2 0 0 , 4 0 0 \}$ , we set $d = 1 0 0 , \rho = 0 . 2$ , these cases examine whether the proposed method performs well in term of power when the sample size becomes larger. The above cases include both $d > n$ and $d < n$ . For the different correlations $\rho = \{ 0 , 0 . 2 , 0 . 5 , 1 \}$ , we set $d = 1 0 0$ , $n = 2 0 0$ , these cases examine whether the proposed method performs well when the correlation between variables is increasing.

Table 1 and Figure 1 show that KI-LAND procedure can achieve a good performance in terms of FDR control and power. As the sample size increasing, the power of both the linear and nonlinear components exhibits an upward trend. Conversely, the power shows an opposite trend of change when the variable dimension decreases or the correlation increases. This phenomenon is in line with expectations. Although KI-LAND has a weaker power than KKO, it

<html><body><table><tr><td colspan="6">FDR</td><td colspan="5">Power</td></tr><tr><td></td><td>KI-LAND (L) KI-LAND (N)</td><td></td><td>KKO</td><td></td><td></td><td>Model-X LAND KI-LAND (L) KI-LAND (N)</td><td></td><td>KKO</td><td>Model-X LAND</td><td></td></tr><tr><td>50</td><td>0.1667</td><td>0.2000</td><td>0.1875</td><td>0.2000</td><td>0.3333</td><td>50.00%</td><td>40.00%</td><td>65.00%</td><td>20.00%</td><td>60.00%</td></tr><tr><td>100</td><td>0.1250</td><td>0.1429</td><td>0.1500</td><td>0.1667</td><td>0.3684</td><td>70.00%</td><td>60.00%</td><td>85.00%</td><td>25.00%</td><td>60.00%</td></tr><tr><td>n 200</td><td>0.2000</td><td>0.1429</td><td>0.1429</td><td>0.1429</td><td>0.3182</td><td>80.00%</td><td>60.00%</td><td>90.00%</td><td>30.00%</td><td>75.00%</td></tr><tr><td>400</td><td>0.1818</td><td>0.1111</td><td>0.1429</td><td>0.2000</td><td>0.2963</td><td>90.00%</td><td>80.00%</td><td>90.00%</td><td>40.00%</td><td>95.00%</td></tr><tr><td>50</td><td>0.1111</td><td>0.1111</td><td>0.1429</td><td>0.2000</td><td>0.2727</td><td>80.00%</td><td>80.00%</td><td>90.00%</td><td>40.00%</td><td>80.00%</td></tr><tr><td>100 d</td><td>0.1818</td><td>0.1111</td><td>0.1905</td><td>0.1667</td><td>0.3000</td><td>90.00%</td><td>80.00%</td><td>85.00%</td><td>25.00%</td><td>70.00%</td></tr><tr><td>200</td><td>0.1250</td><td>0.1429</td><td>0.1500</td><td>0.1667</td><td>0.3684</td><td>70.00%</td><td>60.00%</td><td>85.00%</td><td>25.00%</td><td>60.00%</td></tr><tr><td>400</td><td>0.1429</td><td>0.1429</td><td>0.2000</td><td>0.2000</td><td>0.4211</td><td>60.00%</td><td>60.00%</td><td>80.00%</td><td>20.00%</td><td>55.00%</td></tr><tr><td>0</td><td>0.1000</td><td>0.2000</td><td>0.0500</td><td>0.1429</td><td>0.2500</td><td>90.00%</td><td>80.00%</td><td>95.00%</td><td>30.00%</td><td>75.00%</td></tr><tr><td>0.2 p</td><td>0.1250</td><td>0.1429</td><td>0.1500</td><td>0.1667</td><td>0.3684</td><td>70.00%</td><td>60.00%</td><td>85.00%</td><td>25.00%</td><td>60.00%</td></tr><tr><td>0.5</td><td>0.1429</td><td>0.1429</td><td>0.1000</td><td>0.2000</td><td>0.3000</td><td>60.00%</td><td>60.00%</td><td>90.00%</td><td>20.00%</td><td>70.00%</td></tr><tr><td>1</td><td>0.1667</td><td>0.1667</td><td>0.2000</td><td>0.2000</td><td>0.4000</td><td>50.00%</td><td>50.00%</td><td>80.00%</td><td>20.00%</td><td>60.00%</td></tr></table></body></html>

Table 1: Comparisons in terms of FDR and power with the different sample sizes, variable dimensions and correlations.

![](images/c2262bd55966fd4d2621e274972be231bb5f588c340b43d1107f0f69279f6608.jpg)  
Figure 1: Empirical performance and comparisons in terms of FDR and power with the different (a) sample sizes, (b) variable dimensions and (c) correlations. (L) and (N) denote the experiments of KI-LAND on the set $\hat { \mathcal { T } } _ { L }$ and $\hat { \mathcal { I } } _ { N }$ , respectively.

can identify linear and nonlinear variables while KKO just implements variable selection. The low power of modelX requires further explanation, as it stems from the use of Lasso which is unable to capture nonlinear relationships. The changing trend of FDR value are more complex, but we focus on whether it can be controlled below the target level $q = 0 . 2$ . Table 1 shows that FDR can be controlled by all knockoffs methods including KI-LAND, KKO and modelX. Compared to LAND, KI-LAND can control FDR successfully while LAND inflates FDR. The experiments with real-world dataset are in the supplementary materials.

# Conclusion

This paper formulates a new Knockoffs Inference scheme for Linear And Nonlinear Discoverer (KI-LAND), where FDR is controlled with respect to both linear and nonlinear variables for automatic structure discovery. Compared with the most related works (Zhang and Liu 2011; Xiaowu Dai and Li 2023), experimental results have shown that the proposed KI-LAND exhibits strong competitiveness. In theory, we have established the fine-grained characterizations on the FDR controllability and the power performance. The full version of the paper (including supplementary materials) can be found at arXiv.

# Acknowledgments

The work was supported in part by the National Natural Science Foundation of China under Grants 62376104, 12426512, and 12071166, in part by the Fundamental Research Funds for the Central Universities of China (No. 2662023LXPY005), and in part by Hubei Provincial Natural Science Foundation of China (No.2023AFB523).