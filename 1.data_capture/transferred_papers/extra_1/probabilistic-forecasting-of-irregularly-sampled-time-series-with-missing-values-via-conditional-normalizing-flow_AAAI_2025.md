# Probabilistic Forecasting of Irregularly Sampled Time Series with Missing Values via Conditional Normalizing Flows

Vijaya Krishna Yalavarthi\*1, Randolf Scholz\*1, Stefan Born 2, Lars Schmidt-Thieme 1

1 Information Systems and Machine Learning Lab, University of Hildesheim, Germany 2 Institute of Mathematics, TU Berlin, Germany {yalavarthi, scholz, schmidt-thieme}@ismll.de, born $@$ math.tu-berlin.de

# Abstract

Probabilistic forecasting of irregularly sampled multivariate time series with missing values is crucial for decision making in various domains, including health care, astronomy, and climate. State-of-the-art methods estimate only marginal distributions of observations in single channels and at single timepoints, assuming a Gaussian distribution for the data. In this work, we propose a novel model, ProFITi using conditional normalizing flows to learn multivariate conditional distribution: joint distribution of the future values of the time series conditioned on past observations and specific channels and timepoints, without assuming any fixed shape of the underlying distribution. As model components, we introduce a novel invertible triangular attention layer and an invertible non-linear activation function on and onto the whole real line. Through extensive experiments on 4 real-world datasets, ProFITi demonstrates significant improvement, achieving an average log-likelihood gain of 2.0 compared to the previous state-of-the-art method.

# 1 Introduction

Irregularly sampled multivariate time series with missing values (IMTS) are common in various real-world scenarios such as health, astronomy and climate. Accurate forecasting of IMTS is important for decision-making, but estimating uncertainty is crucial to avoid overconfidence. State-ofthe-art models applied to this task are Ordinary Differential Equations (ODE) based models (Schirmer et al. 2022; De Brouwer et al. 2019; Bilosˇ et al. 2021) which are 1) computationally inefficient, and 2) offer only marginal likelihoods. In practice, joint or multivariate distributions are desired to capture dependencies and study forecasting scenarios. With joint distributions one can estimate the likelihood of specific combinations of future variables ex. likelihood of having rain and strong winds, which marginal or point forecast models cannot deliver.

For this, we propose a novel conditional normalizing flow model called ProFITi, for Probabilistic Forecasting of Irregularly sampled Multivariate Time series. ProFITi is designed to learn conditional joint distributions. We also propose two novel model components that can be used in flow models: a sorted invertible triangular attention layer, SITA, parametrized by conditioning input to learn joint distributions, and an invertible non-linear activation function designed for flows, Shiesh, that is on and onto whole real line. ProFITi consists of several invertible blocks build using SITA and Shiesh functions. Being a flow-based model, ProFITi can learn any random conditional joint distribution, while existing models (Schirmer et al. 2022; De Brouwer et al. 2019; Bilosˇ et al. 2021) learn only Gaussians.

Experiments on 4 real-world IMTS datasets, attest the superior performance of ProFITi. Our contributions are:

1. Introduced ProFITi, a novel and, to the best of our knowledge, first normalizing flow based probabilistic forecasting model for predicting multivariate conditional distributions of irregularly sampled time series with missing values (Section 6).   
2. Proposed a novel invertible triangular attention layer, named SITA, which enables target variables to interact and capture dependencies within a conditional normalizing flow framework (Section 4).   
3. Proposed a novel non-linear, invertible, differentiable activation function on and onto the whole real line, Shiesh (Section 5). This activation function can be used in normalizing flows.   
4. Conducted experiments on 4 IMTS datasets for normalized joint negative loglikelihood. On average, ProFITi provides a loglikelihood gain of 2.0 over the previously best model (Section 7).

A detailed version of this work, along with additional experiments and proofs to lemmas, is available in the technical report (Yalavarthi et al. 2024b). Implementation: github.com/ yalavarthivk/ProFITi

# 2 Literature Review

There have been multiple works dealing point forecasting of irregular time series (Yalavarthi et al. 2024a; Ansari et al. 2023; Chen et al. 2024). Very few models provide uncertainty quantification for such forecasts.

Probabilistic Forecasting Models for IMTS. Probabilistic IMTS forecasting often relies on variational inference or predicting distribution parameters. Neural ODE models (Chen et al. 2018) combine probabilistic latent states with deterministic networks. Other approaches like latentODE (Rubanova, Chen, and Duvenaud 2019), GRU-ODEBayes (De Brouwer et al. 2019), Neural-Flows (Bilosˇ et al. 2021), and Continuous Recurrent Units (Schirmer et al. 2022) provide only marginal distributions, no joint distributions. Similarly, probabilistic interpolation models, such as HETVAE (Shukla and Marlin 2022) and TripletFormer (Yalavarthi, Burchert, and Schmidt-Thieme 2023), also provide only Gaussian marginal distributions. In contrast, Gaussian Process Regression models (GPR; Du¨richen et al. 2015; Li and Marlin 2015, 2016; Bonilla, Chai, and Williams 2007) offer full joint posterior distributions for forecasts, but struggle with the computational demands on long time series due to the dense matrix inversion operations. All the models assume the data distribution is to be Gaussian and fail if the true distribution is different.

Normalizing Flows for variable input size. We deal with predicting distributions for variable many targets. This utilizes equivariant transformations, as shown in (Bilosˇ and Gu¨nnemann 2021; Satorras, Hoogeboom, and Welling 2021; Liu et al. 2019). All the models apply continuous normalizing flows which require solving an ODE driven by a neural network using a slow numerical integration process. Additionally, they cannot incorporate conditioning inputs.

Conditioning Normalizing Flows. Learning conditional densities has been largely explored within computer vision (Khorashadizadeh et al. 2023; Winkler et al. 2019; Anantha Padmanabha and Zabaras 2021). They apply normalizing flow blocks such as affine transformations (Dinh, Sohl-Dickstein, and Bengio 2017), autoregressive transformations (Kingma and Welling 2013) or Sylvester flow blocks (van den Berg et al. 2018). Often the conditioning input is appended to the target while passing through the flow layers as shown in (Winkler et al. 2019). For continuous data representations only a few works exist (Kumar et al. 2020; de Be´zenac et al. 2020; Rasul et al. 2021; Si, Kuleshov, and Bishop 2022). However, methods that deal with regular multivariate time series (such as Rasul et al. 2021) cannot handle IMTS due to its missing values. We solve this by using invertible attention that allows flexible size.

Flows with Invertible Attention. To the best of our knowledge, there have been only two works that develop invertible attention for Normalizing Flows. Sukthanker et. al. (Sukthanker et al. 2022) proposed an invertible attention by adding the identity matrix to a softmax attention. However, softmax yields only positive values in the attention matrix and does not learn negative covariances. Zha et. al. (Zha et al. 2021) introduced residual attention similar to residual flows (Behrmann et al. 2019) that suffer from similar problems as residual flows such as the lack of an explicit inverse making inference slow. Additionally, computing determinants of dense attention matrices has cubic complexity which is not desired.

# 3 Problem Setting & Analysis The IMTS Forecasting Problem. An irregularly sampled multivariate times series with missing values (called

briefly just IMTS in the following), is a sequence $x ^ { \mathrm { o B S } } =$ $( ( t _ { \tau } , v _ { \tau } ) ) _ { \tau = 1 : T }$ where $v _ { \tau } \in \{ \mathbb { R } \cup \bar { \mathbb { N } } \mathrm { a N } \} ^ { C }$ is an observatio=n event at timepoint $t _ { \tau } \in \mathbb { R }$ $v _ { \tau , c } \in \mathbb { R }$ indicates observed value and $\mathtt { N a N }$ indicates a m∈issing va∈lue. Horn et. al. (Horn et al. 2020) introduced set notation where only observed values are considered and missing values are ignored. For notational continence, we use sequences whose order does not matter to represent sets.

Now, $\boldsymbol { x } ^ { \mathrm { O \hat { B } S } } ~ = ~ \big ( \big ( t _ { i } ^ { \mathrm { O B S } } , c _ { i } ^ { \mathrm { O B S } } , o _ { i } ^ { \mathrm { O B S } } \big ) \big ) _ { i = 1 : I }$ is a sequence of unique triple =w(h(ere $t _ { i } ^ { \mathrm { { o B } \bar { s } } } ~ \in ~ \bar { \mathbb { R } }$ R d)e)n=ot∶es the time, cOB $c _ { i } ^ { \mathrm { o B S } } \in$ $\{ 1 , . . . , C \}$ the channel and $o _ { i } ^ { \mathrm { { o B S } } } \in \mathbb { R }$ the value of an obser∈- v{ation, $I \in \mathbb { N }$ the total number ∈of observations across all channels ∈nd $C \in \mathbb { N }$ the number of channels.

An IMTS query is a sequence $\begin{array} { r l } { x ^ { \mathrm { Q R Y } } } & { { } \phantom { \left[ \frac { \Theta ^ { \mathrm { Q R Y } } } { \Theta } \right] } = } \end{array}$ $\left( \big ( t _ { k } ^ { \mathrm { Q R Y } } , c _ { k } ^ { \mathrm { Q R Y } } \big ) \right) _ { k = 1 : K }$ of just timepoints and channel=s (also unique), a= s∶equence $\boldsymbol { y } \in \mathbb { R } ^ { K }$ we call an answer. It is understood that $y _ { k }$ is the an w∈ er to the query $( t _ { k } ^ { \mathrm { Q R Y } } , c _ { k } ^ { \mathrm { Q R Y } } )$ .

The IMTS probabilistic forecasting pro(blem then) is, given a dataset $\mathcal { D } ^ { \mathrm { t r a i n } } : = \left( \left( x ^ { \mathrm { O B S } n } , x ^ { \mathrm { Q R Y } n } , \mathcal { \bar { Y } } ^ { \hat { n } } \right) \right) _ { n = 1 : N }$ of triples of time series, queries and answers from an u=n∶known distribution $p$ (with earliest query timepoint is beyond the latest observed timepoint for series $n$ $\mathrm { , \dot { m i n } } _ { k } t _ { k } ^ { \mathrm { Q R Y } } \mathrm { \dot { > } m a x } _ { i } t _ { i } ^ { \mathrm { O B S } } \mathrm { ) }$ , to find a model $\hat { p }$ that maps each observatio>n/query pair $( x ^ { \mathrm { o B S } } , x ^ { \mathrm { Q R Y } } )$ to a joint density over answers, $\hat { p } ( y _ { 1 } , \dots , y _ { k } \ |$ $x ^ { \mathrm { { o B S } } } , x ^ { \mathrm { { Q R Y } } } )$ ,)such that the expected joint negativ(e log likel -∣ hood is mi)nimal:

$$
\ell ^ { \mathrm { j N L L } } ( \hat { p } ; p ) : = - \mathbb { E } _ { ( x ^ { \mathrm { o b s } } , x ^ { \mathrm { o R Y } } , y ) \sim p } \log \hat { p } ( y | x ^ { \mathrm { o B s } } , x ^ { \mathrm { o R Y } } )
$$

Please note, that the number $C$ of channels is fixed, but the number $I$ of past observations and the number $K$ of future observations queried may vary over instances $( x ^ { \mathrm { o B S } } , x ^ { \mathrm { Q R Y } } , y )$ . If query sizes $K$ vary, instead of (joint) neg(ative log likel)ihood one also can normalize by query size to make numbers comparable over different series and limit the influence of large queries, the normalized joint negative log likelihood njNLL:

$$
\ell ^ { \mathrm { n j N L L } } ( \hat { p } ; p ) : = - \underbrace { \mathbb { E } } _ { ( x ^ { \mathrm { o b s } } , x ^ { \mathrm { o R Y } } , y ) \sim p } \frac { 1 } { | y | } \log \hat { p } ( y | x ^ { \mathrm { o B s } } , x ^ { \mathrm { o R Y } } )
$$

Problem Analysis and Characteristics. As the problem is not just an (unconditioned) density estimation problem, but the distribution of the outputs depends on both, the past observations and the queries, a conditional density model is required (requirement 1).

A crucial difference from many settings addressed in the related work (Schirmer et al. 2022; Bilosˇ et al. 2021; De Brouwer et al. 2019) is that we look for probabilistic models of the joint distribution of all queried observation values $( y _ { 1 } , \dotsc , y _ { K } )$ , not just at the single variable Tmhaergpirnoabl edmis o(rfibmutairoginsa $p \big ( y _ { k } \mid \overline { { x } } ^ { \mathrm { O B S } } , x _ { k } ^ { \mathrm { Q R Y } } \big )$ s(fpoerc $k = 1 { : } K )$ our formulation where all queries happen to have just one element (always $K = 1$ ). So for joint probabilistic forecasting of IMTS, mode s=need to output densities on a variable number of variables (requirement 2).

Furthermore, since we deal with the set representation of IMTS, whenever two query elements get swapped, a generative model should swap its output accordingly, a density model should yield the same density value, i.e., the model should be permutation invariant (requirement 3). For any permutation $\pi$ :

$$
\begin{array} { c } { { \hat { p } ( y _ { 1 } , \dots , y _ { K } \mid x ^ { \mathrm { O B S } } , x _ { 1 } ^ { \mathrm { Q R Y } } , \dots , x _ { K } ^ { \mathrm { Q R Y } } ) = } } \\ { { \hat { p } ( y _ { \pi ( 1 ) } , \dots , y _ { \pi ( K ) } \mid x ^ { \mathrm { O B S } } , x _ { \pi ( 1 ) } ^ { \mathrm { Q R Y } } , \dots , x _ { \pi ( K ) } ^ { \mathrm { Q R Y } } ) } } \end{array}
$$

# 4 Invariant Conditional Normalizing Flow Models

Normalizing flows. Parametrizing a specific distribution such as the Gaussian is a simple and robust approach to probabilistic forecasting. It can be added on top of any point forecasting model (for marginal distributions or fixed-size queries at least). However, such models are less suited for targets having a more complex distribution. Then typically normalizing flows are used (Rippel and Adams 2013; Papamakarios et al. 2021). A normalizing flow is an (unconditional) density model for variables $\breve { y } ~ \in ~ \mathbb { R } ^ { K }$ consisting of a simple base distribution, typically ∈standard normal $p _ { Z } ( z ) : = \mathcal { N } ( z ; 0 _ { K } , \mathbb { I } _ { K \times K } )$ , and an invertible, differentiable, parametrized map $f ( z ; \theta ) : \mathbb { R } ^ { K } \to \mathbb { R } ^ { K }$ ; then

$$
\hat { p } ( y ; \theta ) : = p _ { Z } ( f ^ { - 1 } ( y ; \theta ) ) \left| \operatorname* { d e t } \left( \frac { \partial f ^ { - 1 } ( y ; \theta ) } { \partial y } \right) \right|
$$

is a proper density, i.e., integrates to 1, and can be fitted to data minimizing negative log likelihood via gradient descent algorithms. A normalizing flow can be conditioned on predictor variables $\boldsymbol { x } \in \mathbb { R } ^ { M }$ by simply making $f$ dependent on predictors $x$ , too: $f ( z ; x , \theta )$ (satisfying requirement 1). $f$ then has to be invertible w.r.t. $z$ for any $x$ and $\theta$ (Trippe and Turner 2018).

Invariant conditional normalizing flows. A conditional normalizing flow represents an invariant conditional distribution in the sense of eq. 2 (requirement 3), if i) its predictors $x$ also can be grouped into $K$ elements $x _ { 1 } , \ldots , x _ { K }$ and possibly common elements $x ^ { \mathrm { { c o m } } }$ : $\boldsymbol { x } ~ = ~ \left( x _ { 1 } , \ldots , x _ { K } , x ^ { \mathrm { c o m } } \right)$ , and ii) its transformation $f$ is equiva=r (nt in stacked $x _ { 1 : K }$ and $z _ { 1 : k }$ :

$$
\begin{array} { r l } & { f ( z ^ { \pi } ; x _ { 1 : K } ^ { \pi } , x ^ { \mathrm { c o m } } , \theta ) ^ { \pi ^ { - 1 } } = f ( z ; x _ { 1 : K } , x ^ { \mathrm { c o m } } , \theta ) } \\ & { ~ \forall \mathrm { p e r m u t a t i o n s } ~ \pi } \end{array}
$$

where $z ^ { \pi } : = \bigl ( z _ { \pi ( 1 ) } , \ldots , z _ { \pi ( K ) } \bigr )$ denotes a permuted vector. We call t= (s an( i)nvariant(co)n)ditional normalizing flow model. If $K$ is fixed, we call it fixed size, otherwise dynamic size. In our work, we consider $x _ { 1 : K }$ as the embedding of $x _ { 1 } ^ { \mathrm { Q R Y } } , \dotsc , x _ { K } ^ { \mathrm { Q R Y } }$ and $x ^ { \mathrm { o B s } }$ and ignore $x ^ { \mathrm { { c o m } } }$ (see eq. 16).

Invariant conditional normalizing flows via invertible attention. The primary choice for a dynamic size (requirement 2), equivariant, parametrized function is attention (Attn; Vaswani et al. 2017):

$$
\begin{array} { r l } & { ~ A ( X _ { \mathrm { Q } } , X _ { \mathrm { K } } ) : = X _ { \mathrm { Q } } W _ { \mathrm { Q } } ( X _ { \mathrm { K } } W _ { \mathrm { K } } ) ^ { T } , } \\ & { ~ A ^ { \mathrm { s o f t m a x } } ( X _ { \mathrm { Q } } , X _ { \mathrm { K } } ) : = \mathrm { s o f t m a x } ( A ( X _ { \mathrm { Q } } , X _ { \mathrm { K } } ) ) } \\ & { \mathrm { A t t n } ( X _ { \mathrm { Q } } , X _ { \mathrm { K } } , X _ { \mathrm { V } } ) : = A ^ { \mathrm { s o f t m a x } } ( X _ { \mathrm { Q } } , X _ { \mathrm { K } } ) \cdot X _ { \mathrm { V } } W _ { \mathrm { V } } } \end{array}
$$

where $X _ { \mathrm { Q } } , X _ { \mathrm { K } } , X _ { \mathrm { V } }$ are query, key and value matrices, $W _ { \mathrm { Q } } , W _ { \mathrm { K } } , \dot { W } _ { \mathrm { V } }$ are parameter matrices (not depending on the number of rows of $X _ { \mathrm { Q } } , X _ { \mathrm { K } } , X _ { \mathrm { V } } )$ and the softmax is taken rowwise.

Self attention mechanism $\mathrm { \langle } X _ { 0 } = X _ { \mathrm { K } } = X _ { \mathrm { V } } \mathrm { \rangle }$ ) has been used in the literature as is for uncondi i=onal v=ector fields (Ko¨ hler, Klein, and Noe 2020; Li et al. 2020; Bilosˇ and Gu¨nnemann 2021). To be used in a conditional vector field, $X _ { \mathrm { Q } } = X _ { \mathrm { K } } =$ $X$ will have to contain the condition elements $x _ { 1 : K }$ , $X _ { \mathrm { V } } = Z$ contains the base samples $z _ { 1 : K }$ and $W _ { \mathrm { V } } = 1$ .

$$
X : = \left[ \begin{array} { c } { \boldsymbol { x } _ { 1 } ^ { T } } \\ { \vdots } \\ { \boldsymbol { x } _ { K } ^ { T } } \end{array} \right] , \quad Z : = \left[ \begin{array} { c } { \boldsymbol { z } _ { 1 } } \\ { \vdots } \\ { \boldsymbol { z } _ { K } } \end{array} \right]
$$

Now, we make attention matrix itself invertible. To get invertible attention (iAttn), we regularize the attention matrix $A$ sufficiently to become invertible (see Lemma 1 in Yalavarthi et al. 2024b for proof)

$$
\begin{array} { c } { { A ^ { \mathrm { r e g } } ( X ) : = \displaystyle \frac { 1 } { \| A ( X , X ) \| _ { 2 } + \epsilon } A ( X , X ) + \mathbb { I } } } \\ { { \mathrm { i A t t n } ( Z , X ) : = A ^ { \mathrm { r e g } } ( X ) . Z } } \end{array}
$$

where $\epsilon > 0$ is a hyperparameter. We note that, $A ^ { \mathrm { r e g } } ( X )$ is not a par>ameter of the model, but computed from th(e c)onditioners $x$ . Our approach is different from iTrans attention (Sukthanker et al. 2022, fig. 17) that makes attention invertible more easily via $\mathring { A } ^ { { \mathrm { i T r a n s } } } ( X ) : = A ^ { \mathrm { s o f t m a x } } ( X , X ) + \mathbb { I }$ using the fact that the spectral ra(dius) $\sigma ( A ^ { \mathrm { s o f t m a x } } ( X , X ) ) \leq 1$ , but therefore is restricted to non-nega(tive intera(ction )w)ei≤ghts.

The attention matrix $A ^ { \mathrm { r e g } } ( X )$ will be dense in general and thus slow to invert, takin(g $\dot { \mathcal { O } } ( K ^ { 3 } )$ operations. Following ideas for autoregressive flows and coupling layers, a triangular matrix would allow a much more efficient inverse pass, as its determinant can be computed in $\mathcal { O } ( K )$ and linear systems can be solved in $\mathcal { O } ( K ^ { 2 } )$ . This doeOs (not)restrict the expressivity of the model, as due to the Knothe–Rosenblatt rearrangement (Villani 2009) from optimal transport theory, any two probability distributions on $\textstyle { \dot { \mathbb { R } } } ^ { K }$ can be transformed into each other by flows with a locally triangular Jacobian. Unfortunately, just masking the upper triangular part of the matrix will destroy the equivariance of the model. We resort to the simplest way to make a function equivariant: we sort the inputs before passing them into the layer and revert the outputs to the original ordering. We call this approach sorted invertible triangular attention (SITA):

$$
\pi : = \operatorname { a r g s o r t } ( x _ { 1 } S , \dots , x _ { K } S )
$$

$$
A ^ { \mathrm { t r i } } ( X ) : = \mathrm { s o f t p l u s - d i a g } ( \log \mathrm { e r - t r i a n g } ( A ( X , X ) ) ) + \epsilon \mathbb { I }
$$

$$
\mathrm { S I T A } ( Z , X ) : = { ( A ^ { \mathrm { t r i } } ( X ^ { \pi } ) \cdot Z ^ { \pi } ) ^ { \pi ^ { - 1 } } }
$$

where $\pi$ operates on the rows of $X$ and $Z$ . Softplus activation is applied to diagonal elements making them positive. Sorting is a simple lexicographic sort along the dimensions of vector $x _ { k } S$ . The matrix $S$ allows to specify a sorting criterion, e.g., a permutation matrix. Note that sorting is unique only when $x$ has unique elements. In practice, we compute $\pi$ from $x ^ { \mathrm { Q R Y } }$ instead of $x$ , first sort by timepoint, and then by channel.

Table 1: Properties of existing activation functions.   

<html><body><table><tr><td>Activation</td><td>E1</td><td>E2</td><td>E3</td></tr><tr><td>ReLU</td><td>×</td><td>×</td><td>×</td></tr><tr><td>Leaky-ReLU</td><td>√</td><td>√</td><td>√</td></tr><tr><td>P-ReLU</td><td>√</td><td>√</td><td>√</td></tr><tr><td>ELU</td><td>√</td><td>×</td><td>√</td></tr><tr><td>SELU</td><td>√</td><td>×</td><td>√</td></tr><tr><td>GELU</td><td>×</td><td>×</td><td>√</td></tr><tr><td>Tanh</td><td>√</td><td>×</td><td>√</td></tr><tr><td>Sigmoid</td><td>√</td><td>×</td><td>√</td></tr><tr><td>Tanh-shrink</td><td>√</td><td>√</td><td>×</td></tr><tr><td>Shiesh</td><td>√</td><td>√</td><td>√</td></tr></table></body></html>

Example 1 (Demonstration of sorting in SITA). Given $\begin{array} { r l r } { x ^ { \mathrm { { Q R Y } } } } & { { } = } & { \left( ( 1 , 2 ) , ( 0 , 2 ) , ( 2 , 1 ) , ( 3 , 1 ) , ( \bar { 0 } , 1 ) , ( 3 , 3 ) \right) } \end{array}$ where first an=d se(c(ond)el(emen)ts(in $x _ { k } ^ { \mathrm { Q R Y } }$ ind)ic(ate q)ue(ried)t)ime and channel respectively. Assume $\textstyle { \ddot { S } } = { \bigl ( } { } _ { 0 } ^ { 1 } _ { 1 } ^ { 0 } { \bigr ) }$ . Then

$$
\begin{array} { l } { \pi = a r g s o r t ( x _ { 1 } ^ { \mathrm { o R Y } } S , \ldots , x _ { 5 } ^ { \mathrm { o R Y } } S ) } \\ { = a r g s o r t ( ( 1 , 2 ) , ( 0 , 2 ) , ( 2 , 1 ) , ( 3 , 1 ) , ( 0 , 1 ) , ( 3 , 3 ) ) } \\ { = ( 5 , 2 , 1 , 3 , 4 , 6 ) } \end{array}
$$

# 5 Shiesh: A New Activation Function for Normalizing Flows

The transformation function $f$ of a normalizing flow usually is realized as a stack of several simple functions. As in any other neural network, elementwise applications of a function, called activation functions, is one of those layers that allows for non-linear transformations. However, most common activation functions used in deep learning are not diffeomorphic and do not have both their domain and codomain on the entire real line, making them unsuitable for normalizing flows. For instance:

• ReLU is not invertible (E1)   
• ELU cannot be used consistently throughout the layer stack because its output domain, $\mathbb { R } ^ { + }$ , does not span the entire real number line (E2)   
• Tanh-shrink (tanhshrink $( u ) : = u { \mathrm { - t a n h } } ( u ) )$ is invertible and covers the entire real(li)ne∶,=bu−t it has( a)zero gradient at some points (e.g., at $u = 0$ ). This zero gradient makes it impossible to compute t=he inverse of the normalizing factor, det ∂f∂(u ) , required for normalizing flows (E3)

To serve as a standalone layer in a normalizing flow, an activation function must fulfill these three requirements: E1. be invertible, E2. cover the whole real line and E3. have no zero gradients. Out of all activation functions in the pytorch library $( \mathrm { V } 2 . 2 )$ only Leaky-ReLU and P-ReLU meet all three requirements (see Table ??). Leaky-ReLU and P-ReLU usually are used with a slope on their negative branch being well less than 1, so that stacking many of them might lead to small gradients also causing problems for the normalizing constant of a normalizing flow.

To address these challenges, we propose a new activation function derived from unconstrained monotonic neural networks (UMNN; Wehenkel and Louppe 2019). UMNN have been proposed as versatile, learnable activation functions for normalizing flows, being basically a continuous flow for each scalar variable $u$ separately and a scalar field $g$ implemented by a neural network:

![](images/02d82750d152b10f9b0c811a66114a5aa90930c0ae6b2a3f0224da58bf84f36c.jpg)  
Figure 1: (left) Shiesh function, (right) partial derivative.

$$
a ( u ) : = v ( 1 ) \mathrm { ~ w i t h ~ } v : [ 0 , 1 ] \to \mathbb { R }
$$

$$
\mathrm { b e i n g ~ t h e ~ s o l u t i o n ~ o f ~ } ~ \frac { \partial v } { \partial \tau } = g \big ( \tau , v ( \tau ) \big ) , ~ v ( 0 ) : = u
$$

In consequence, they suffer from the same issues as any continuous normalizing flow: they are slow as they require explicit integration of the underlying ODE. Besides requirements E1–E3, activation functions will profit from further desired properties: D1. having an explicit inverse, D2. having an explicit Jacobian and D3. having a bounded gradient. UMNN do not have desired property D1 and provide no guarantees for property D3.

Instead of parameterizing the scalar field $g$ and learn it from data, we make an educated guess and choose a specific function with few parameters for which eq. 11 becomes explicitly solvable and requires no numerics at runtime: for the scalar field $g ( \tau , a ; b ) : = \mathrm { t a n h } ( b \cdot a ( \tau ) )$ the resulting ODE

$$
\frac { \partial v } { \partial \tau } = \mathrm { t a n h } ( b \cdot v ( \tau ) ) , \quad v ( 0 ) : = u
$$

has an explicit solution (Sec. G; Yalavarthi et al. 2024b)

$$
v ( \tau ; u , b ) = \frac { 1 } { b } \sinh ^ { - 1 } \left( e ^ { b \cdot \tau } \cdot \sinh ( b \cdot u ) \right)
$$

yielding our activation function Shiesh:

$$
\begin{array} { l } { { \mathrm { S h i e s h } ( u ; b ) : = a ( u ) : = v ( 1 ; u , b ) \nonumber } } \\ { { \qquad = { \displaystyle { \frac { 1 } { b } } \sinh ^ { - 1 } \left( e ^ { b } \sinh ( b \cdot u ) \right) } \nonumber } } \end{array}
$$

being invertible, covering the whole real line and having no zero gradients (E1–E3) and additionally with analytical inverse and gradient (D1 and D2)

$$
\begin{array} { l } { { \displaystyle \mathrm { S h i e s h } ^ { - 1 } ( u ; b ) = \frac { 1 } { b } \sinh ^ { - 1 } \left( e ^ { - b } \cdot \sinh ( b \cdot u ) \right) } } \\ { { \displaystyle \frac { \partial } { \partial u } \mathrm { S h i e s h } ( u ; b ) = \frac { e ^ { b } \cosh ( b \cdot u ) } { \sqrt { 1 + \left( e ^ { b } \sinh ( b \cdot u ) \right) ^ { 2 } } } } } \end{array}
$$

and bounded gradient (D3) in the range $( 1 , e ^ { b } ]$ (G.4, Yalavarthi et al. 2024b). Figure 1 shows a f (ction ]plot. In our experiments we fixed its parameter $b = 1$ .

Since the Shiesh is applied element-wis=e, it does not violate the Requirements 1, 2 and 3 in Section 3.

![](images/f53ffeafa259fb49d55ecaec3867ead35642e7c00b5d561715460ab447c48d51.jpg)  
Figure 2: ProFITi architecture; $\otimes$ : dot product, $\odot$ : Hadamard product, $\oplus$ : addition. F ⊗ctions referred to  e⊙ir equation numbers: s⊕ort, argsort (eq. 8), GraFITi (eq. 16), SITA (eq. 10), EL (eq. 14), Shiesh (eq. 12). For efficiency, we perform, sorting only once directly on $x ^ { \mathrm { Q R Y } }$ and $y$ .

# 6 Overall ProFITi Model Architecture

Invertible attention and the Shiesh activation function systematically model inter-dependencies between variables and non-linearity respectively, but do not move the zero point. To accomplish the latter, we use a third layer called elementwise linear transformation layer (EL):

$$
\mathrm { E L } \left( y _ { k } ; x _ { k } \right) : = y _ { k } \cdot \mathrm { N N } ^ { \mathrm { s c a } } \left( x _ { k } \right) + \mathrm { N N } ^ { \mathrm { t r s } } \left( x _ { k } \right)
$$

where $\mathrm { { N N ^ { s c a } } }$ and $\mathrm { N N } ^ { \mathrm { t r s } }$ are neural networks for scaling and translation. $\mathrm { { N N ^ { s c a } } }$ is equipped with a $\exp ( \mathrm { t a n h } )$ output function to make it positive and bounded, gu(arante)eing the inverse. We combine all three layers from eq. 7, 12, and 14 to a block

$$
\mathrm { p r o f i t i - b l o c k } ( y ; x ) : = \mathrm { S h i e s h } ( \mathrm { E L } ( \mathrm { S I T A } ( y ; x ) ; x ) )
$$

and stack $L$ of those blocks to build the inverse transformation $f ^ { - 1 }$ of our conditional invertible flow model ProFITi. We add a transformation layer with slope fixed to 1 as initial encoding on the $y$ -side of the model. See Figure 2 for an overview of its architecture. As shown in the figure, for efficiency reasons we perform sorting (eq. 10) only once directly on the queries $x ^ { \mathrm { Q R Y } }$ and answers $y$ .

Encoder for Query embedding. As discussed in Section 4, for probabilistic time series forecasting we have to condition on both, the past observations $x ^ { \mathrm { o B S } }$ and the queried time point/channel pairs $x ^ { \mathrm { Q R Y } }$ of interest. While in principle any equivariant encoder could be used, an encoder that leverages the relationships between those two pieces of the conditioner is crucial. We use GraFITi (Yalavarthi et al. 2024a), a graph based equivariant point forecasting model for IMTS that provides state-of-the-art performance (in terms of accuracy and efficiency) as encoder

$$
\left( x _ { 1 } , \dots , x _ { K } \right) : = { \mathrm { G r a F I T i } } ( x _ { 1 } ^ { \mathrm { { o R Y } } } , \dots , x _ { K } ^ { \mathrm { { o R Y } } } , x ^ { \mathrm { { o B S } } } )
$$

The Grafiti encoder is trained end-to-end within the Profiti model, we did not pretrain it.

Note that for each query, other IMTS forecasting models yield a scalar, the predicted value, not an embedding vector. While it would be possible to use IMTS forecasting models as (scalar) encoders, due to their limitations to a single dimension we did not follow up on this idea.

Training. We train the ProFITi model $\hat { p }$ for the normalized joint negative log-likelihood loss (njNLL; eq. 1) which written in terms of the transformation $f ^ { - 1 } ( \cdot ; \cdot ; \theta )$ of the normalizing flow and its parameters $\theta$ yields:

$$
\begin{array} { r l r } {  { \ell ^ { \mathrm { n j N L L } } \big ( \theta \big ) : = \ell ^ { \mathrm { n j N L L } } \big ( \hat { p } ; p \big ) } } \\ & { = \underbrace { - \mathbb { E } } _ { \big ( x ^ { \mathrm { o s s } } , x ^ { \mathrm { o u r } } , y \big ) \sim p } \frac { 1 } { \big | y \big | } \log p _ { Z } \big ( f ^ { - 1 } \big ( y ; x ^ { \mathrm { o b s } } , x ^ { \mathrm { o R Y } } ; \theta \big ) \big ) } \\ & { } & { \bigg | \mathrm { d e t } \bigg ( \frac { \partial f ^ { - 1 } \big ( y ; x ^ { \mathrm { o s s } } , x ^ { \mathrm { o R Y } } ; \theta \big ) } { \partial y } \bigg ) \bigg | \quad \quad \quad ( 1 7 ) } \end{array}
$$

# 7 Experiments

# 7.1 Experiment for Joint Likelihoods

Datasets. We use 3 publicly available real-world medical IMTS datasets: MIMIC-III (Johnson et al. 2016), MIMIC-IV (Johnson et al. 2021), and Physionet’12 (Silva et al. 2012). Datasets contain ICU patient records collected over 48 hours. The preprocessing procedures outlined in (Yalavarthi et al. 2024a; Bilosˇ et al. 2021; De Brouwer et al. 2019) were applied. Observations in Physionet’12, MIMIC-III and MIMIC-IV were rounded to intervals of 1 hr, $3 0 \mathrm { m i n }$ and 1 min respectively. We also evaluated on publicly available climate dataset USHCN (Menne, Williams Jr, and Vose 2015). It consists of climate data observed for 150 years from 1218 weather stations in USA.

Baseline Models. ProFITi is compared to 3 probabilistic IMTS forecasting models: CRU (Schirmer et al. 2022), Neural-Flows (Bilosˇ et al. 2021), and GRU-ODEBayes (De Brouwer et al. 2019). To disentangle lifts originating from GraFITi (encoder) and those originating from ProFITi, we add GraFITi+ as a baseline. GraFIT $^ +$ predicts an elementwise mean and variance of a normal distribution. As often interpolation models can be used seamlessly for forecasting, too, we include HETVAE (Shukla and Marlin 2022), a state-of-the-art probabilistic interpolation model, for comparison. Furthermore, we include Multi-task Gaussian Process Regression (GPR; Du¨richen et al. 2015) as a baseline able to provide joint densities.

Protocol. We split the dataset into Train, Validation and Test in ratio 70:10:20, respectively. We select the hyperparameters from 10 random hyperparameter configurations based on their validation performance. We perform 5 fold cross validation with the chosen hyperparameters. Following (Bilosˇ et al. 2021) and (Yalavarthi et al. 2024a), we use the first 36 hours as observation range and forecast the next 3 time steps for medical datasets and first 3 years as observation range and forecast the next 3 time steps for climate dataset. Note that 3 time steps meaning do not mean 3 observations. For example, in Physionet’12, $K$ varies between 3 and 49. All models are implemented in PyTorch and run on GeForce RTX-3090 and 1080i GPUs. We compare the models for Normalized Joint Negative Log-likelihood (njNLL) loss (eq. 1). Except for GPR, and ProFITi, we take the average of the marginal negative log-likelihoods of all the observations in a series to compute njNLL for that series.

Table 2: Normalized Joint Negative Log-likelihood (njNLL), lower the better, best in bold, OOM indicates out of memory error, shows imporvement in njNLL w.r.t. next best model.   

<html><body><table><tr><td></td><td>USHCN</td><td>time epoch</td><td>Physioinet'12</td><td>time epoch</td><td>MIMIC-III</td><td>time epoch</td><td>MIMIC-IV</td><td>time epoch</td></tr><tr><td>GPR</td><td>2.011±1.376</td><td>2s</td><td>1.367±0.074</td><td>35s</td><td>3.146±0.359</td><td>71s</td><td>2.789±0.057</td><td>227s</td></tr><tr><td>HETVAE</td><td>198.9±397.3</td><td>1s</td><td>0.561±0.012</td><td>8s</td><td>0.794±0.032</td><td>8s</td><td>OOM</td><td></td></tr><tr><td>GRU-ODE</td><td>0.766±0.159</td><td>100s</td><td>0.501±0.001</td><td>155s</td><td>0.961±0.064</td><td>511s</td><td>0.823±0.318</td><td>1052s</td></tr><tr><td>Neural-Flows</td><td>0.775±0.152</td><td>21s</td><td>0.496±0.001</td><td>34s</td><td>0.998±0.177</td><td>272s</td><td>0.689±0.087</td><td>515s</td></tr><tr><td>CRU</td><td>0.761±0.191</td><td>35s</td><td>0.741±0.001</td><td>40s</td><td>1.234±0.076</td><td>131s</td><td>OOM</td><td></td></tr><tr><td>GraFITi+</td><td>0.489±0.173</td><td>3s</td><td>0.367±0.021</td><td>32s</td><td>0.721±0.053</td><td>80s</td><td>0.287±0.040</td><td>84s</td></tr><tr><td>ProFITi (ours)</td><td>-3.226±0.225</td><td>6s</td><td>-0.647±0.078</td><td>59s</td><td>-0.377±0.032</td><td>97s</td><td>-1.777±0.066</td><td>123s</td></tr><tr><td>→</td><td>3.5</td><td></td><td>1.0</td><td></td><td>1.1</td><td></td><td>2.1</td><td></td></tr></table></body></html>

<html><body><table><tr><td></td><td>USHCN</td><td>Physionet'12</td><td>MIMIC-III</td><td>MIMIC-IV</td></tr><tr><td>HETVAE</td><td>168.1±335.5</td><td>0.519±0.018</td><td>0.947±0.071</td><td>OOM</td></tr><tr><td>GRU-ODE</td><td>0.776±0.172</td><td>0.504±0.061</td><td>0.839±0.030</td><td>0.876±0.589</td></tr><tr><td>Neural-Flows</td><td>0.775±0.180</td><td>0.492±0.029</td><td>0.866±0.097</td><td>0.796±0.053</td></tr><tr><td>CRU</td><td>0.762±0.180</td><td>0.931±0.019</td><td>1.209±0.044</td><td>OOM</td></tr><tr><td>GraFITi+</td><td>0.462±0.122</td><td>0.505±0.015</td><td>0.657±0.040</td><td>0.351±0.045</td></tr><tr><td>ProFITi_marg (ours)</td><td>-2.575±1.336</td><td>-0.368±0.033</td><td>0.092±0.036</td><td>-0.782±0.023</td></tr></table></body></html>

Table 3: Results for Marginal Negative Log-likelihood (mNLL), lower the better. Best in bold and second best in italics. ProFITi marg is ProFITi trained for marginals.

Sampling-based metrics like the Energy Score (Gneiting and Raftery 2007) not only suffer from the curse of dimensionality but also evaluate multivariate distributions improperly (Marcotte et al. 2023). Similarly, Continuous Ranked Probability Score sum (CRPS-sum; Rasul et al. 2021) for multivariate probabilistic forecasting can be misled by simple noise models, where random forecasts may outperform state-of-the-art methods (Koochali et al. 2022).

Results. Table 2 demonstrates the Normalized Joint Negative Log-likelihood (njNLL, lower the better) and run time per epoch for all the datasets. Best results are presented in bold. ProFITi outperforms all the prior approaches with significant margin on all the four datasets. While $\mathrm { G r a F I T i + }$ is the second-best performing model, ProFITi outperforms it by an average gain of 2.0 in njNLL. We note that although GPR is predicting joint likelihoods, it performs poorly, likely because of having very few parameters. We note that njNLL of HETVAE is quite high for the USHCN dataset. The reason is HETVAE predicted a very small variance $( 1 0 ^ { - 4 } )$ for 1 sample whose predicted mean is farther from the target. We do not provide results for CRU on MIMIC-IV as our GPU (48GB VRAM) gives out of memory errors. The reason for such high likelihoods compared to the baseline models is i.) not assuming a Gaussian underlying distribution and ii.) directly predicting joint distributions (see Section 7.3).

# 7.2 Auxiliary Experiments for Marginals

Existing models in the related work (De Brouwer et al. 2019) and (Bilosˇ et al. 2021) cannot predict multivariate distributions, hence their evaluation was restricted to Marginal Negative Log-likelihood (mNLL):

$$
\ell ^ { \mathrm { m N L L } } ( \hat { p } ; \mathcal { D } ^ { \mathrm { t e s t } } ) : = - \frac { \sum _ { \substack { x ^ { \mathrm { o b s } } , x ^ { \mathrm { o b r } } , y ) \in D ^ { \mathrm { t s t } } } } \displaystyle \sum _ { k = 1 } ^ { | y | } \log \hat { p } \left( y _ { k } | x ^ { \mathrm { o b s } } , x _ { k } ^ { \mathrm { o R Y } } \right) } { \sum _ { ( x ^ { \mathrm { o b s } } , x ^ { \mathrm { o b r } } , y ) \in D ^ { \mathrm { t e s t } } } | y | }
$$

We trained ProFITi only for marginals by removing SITA from the architecture and called ProFITi marg. Results presented in Table 3 shows that ProFITi marg outperforms baseline models again.

In addition to mNLL, we also compare the models in terms of CRPS score, a metric for marginal probabilistic forecasting in Table 4; and MSE, a metric for point forecasting in Table 5. While all the baseline models can compute CRPS and MSE explicitly from Gaussian parameters, ProFITi requires sampling. For this, we randomly sampled 100 samples. We computed MSE from robustious mean of samples which is the mean of the samples with outliers removed.

Table 4: Results for CRPS score, lower the better. Best in bold and second best in italics   

<html><body><table><tr><td></td><td>USHCN</td><td>Physionet'12</td><td>MIMIC-III</td><td>MIMIC-IV</td></tr><tr><td>HETVAE</td><td>0.229±0.017</td><td>0.278±0.001</td><td>0.359±0.009</td><td>OOM</td></tr><tr><td>GRU-ODE</td><td>0.313±0.012</td><td>0.278±0.001</td><td>0.308±0.005</td><td>0.281±0.004</td></tr><tr><td>Neural-flows</td><td>0.306±0.028</td><td>0.277±0.003</td><td>0.308±0.004</td><td>0.281±0.004</td></tr><tr><td>CRU</td><td>0.247±0.010</td><td>0.363±0.002</td><td>0.410±0.005</td><td>OOM</td></tr><tr><td>GraFITi+</td><td>0.222±0.011</td><td>0.256±0.001</td><td>0.279±0.006</td><td>0.217±.005</td></tr><tr><td>ProFITi_marg (ours)</td><td>0.192±0.019</td><td>0.253±0.001</td><td>0.276±0.001</td><td>0.206±0.001</td></tr></table></body></html>

<html><body><table><tr><td></td><td>USHCN</td><td>Physionet'12</td><td>MIMIC-III</td><td>MIMIC-IV</td></tr><tr><td>HETVAE</td><td>0.298±0.073</td><td>0.304±0.001</td><td>0.523±0.055</td><td>OOM</td></tr><tr><td>GRU-ODE</td><td>0.410±0.106</td><td>0.329±0.004</td><td>0.479±0.044</td><td>0.365±0.012</td></tr><tr><td>Neural-Flows</td><td>0.424±0.110</td><td>0.331±0.006</td><td>0.479±0.045</td><td>0.374±0.017</td></tr><tr><td>CRU</td><td>0.290±0.060</td><td>0.475±0.015</td><td>0.725±0.037</td><td>OOM</td></tr><tr><td>GraFITi</td><td>0.256±0.027</td><td>0.286±0.001</td><td>0.401±0.028</td><td>0.233±0.005</td></tr><tr><td>ProFITi_marg (ours)</td><td>0.300±0.053</td><td>0.295±0.002</td><td>0.443±0.028</td><td>0.246±0.004</td></tr></table></body></html>

Table 5: Results for MSE, lower the better. Best in bold and second best in italics.

ProFITi marg outperforms all the baseline models in terms of CRPS score. On the other hand, for point forecasting, ProFITi marg is the second best model in terms of MSE, and GraFITi remains the best. While we leverage GraFiTi as the encoder for ProFITi, ProFITi incorporates various components specifically designed to predict distributions, even if this sacrifices some point forecast accuracy. Also, GraFITi is trained to predict a Gaussian distribution, and the Mean Squared Error (MSE) is probabilistically equivalent to the negative log-likelihood of a Gaussian distribution with a fixed variance. This probabilistic interpretation can lead to better performance compared to models that are not specifically trained for Gaussian distributions. Additionally, models for uncertainty quantification often suffer from slightly worse point forecasts, as observed in several studies (Lakshminarayanan, Pritzel, and Blundell 2017; Seitzer et al. 2021; Shukla and Marlin 2022). In the domain of probabilistic forecasting, the primary metric of interest is negative loglikelihood, and ProFITi demonstrates superior performance.

7.3 Ablation studies: Varying model components   

<html><body><table><tr><td>Model</td><td>Physionet2012</td></tr><tr><td>ProFITi</td><td>-0.647±0.078</td></tr><tr><td>ProFITi-SITA ProFITi-Shiesh</td><td>-0.470±0.017</td></tr><tr><td>ProFITi-SITA-Shiesh</td><td>0.285±0.061 0.372±0.021</td></tr><tr><td>ProFITi-Shiesh+PReLU</td><td>0.384±0.060</td></tr><tr><td>ProFITi-Shiesh+LReLU ProFITi-Atri+AiTrans</td><td>NaNError</td></tr><tr><td></td><td>-0.199±0.141</td></tr><tr><td>ProFITi-Ari+Areg</td><td>-0.778±0.016</td></tr></table></body></html>

Table 6: Varying model components; Metric: njNLL; ProFITi- $\mathbf { A } { + } \mathbf { B }$ : component A is removed and B is added.

To analyze ProFITi’s superior performance, we conduct an ablation study on Physionet’12 (Table 6). The Shiesh activation significantly improves performance by enabling learning of non-Gaussian distributions (compare ProFITi and ProFITi-Shiesh). Similarly, learning joint distributions (ProFITi) outperforms ProFITi-SITA in njNLL. ProFITiSITA-Shiesh, which learns only Gaussian marginals, performs worse than ProFITi. Replacing Shiesh with PReLU (ProFITi-Shiesh+PReLU) degrades performance, and using Leaky-ReLU (LReLU) results in small Jacobians and vanishing gradients. The $A ^ { \mathrm { i T r a n s } }$ (ProFITi- $A ^ { \mathrm { t r i } } + A ^ { \mathrm { i T r a n s } } )$ ) variant performs poorly due to its limitation to p+ositive covariances. While ProFITi with $A ^ { \mathrm { r e g } }$ or $A ^ { \mathrm { t r i } }$ shows similar results, $A ^ { \mathrm { r e g } }$ faces scalability challenges: computing the determinant of a full attention matrix has $\mathcal { O } ( K ^ { \hat { 3 } } )$ complexity, compared to $\mathcal { O } ( K )$ for triangular matOri(ces.)Additionally, $A ^ { \mathrm { r e g } }$ underpe oOr(ms)for longer forecast horizons (see appendix H.2; Yalavarthi et al. 2024b). Further experiments on varying observation and forecast horizons and sparsity levels are provided in appendix $\mathrm { ~ H ~ }$ of Yalavarthi et al. (2024b).

# Conclusions

In this work, we propose a novel model ProFITi for probabilistic forecasting of irregularly sampled multivariate time series with missing values using conditional normalizing flows. ProFITi is designed to learn multivariate conditional distribution of varying length sequences. To the best of our knowledge, ProFITi is the first normalizing flow based model that can handle irregularly sampled time series with missing values. We propose two novel model components, sorted invertible triangular attention and Shiesh activation function in order to learn any random target distribution. Our experiments on four real-world datasets demonstrate that ProFITi provides significantly better likelihoods than existing models.

# Acknowledgments

This work was supported by the Federal Ministry for Economic Affairs and Climate Action (BMWK), Germany, within the framework of the IIP-Ecosphere project (project number: 01MK20006D); co-funded by the Lower Saxony Ministry of Science and Culture under grant number ZN3492 within the Lower Saxony “Vorab” of the Volkswagen Foundation and supported by the Center for Digital Innovations (ZDIN); and also by the German Federal Ministry of Education and Research (BMBF) through the Program ”International Future Labs for Artificial Intelligence” (Grant 1DD20002A – KIWI-BioLab)”