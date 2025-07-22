# PowerMLP: An Efficient Version of KAN

Ruichen $\mathbf { Q i u } ^ { 1 , 2 }$ , Yibo Miao2,3, Shiwen Wang3, Yifan $\mathbf { Z } \mathbf { h } \mathbf { u } ^ { 2 , 3 }$ , Lijia $\mathbf { Y u } ^ { 4 , 5 }$ , Xiao-Shan Gao2,3\*

1School of Advanced Interdisciplinary Sciences, UCAS, Beijing 100049, China   
2Academy of Mathematics and Systems Science, CAS, Beijing 100190, China 3University of Chinese Academy of Sciences, Beijing 101408, China 4Institute of Software, CAS, Beijing 100190, China 5State Key Laboratory of Computer Science   
qiuruichen $2 0 @$ mails.ucas.ac.cn, miaoyibo $@$ amss.ac.cn, xgao@mmrc.iss.ac.cn

# Abstract

The Kolmogorov-Arnold Network (KAN) is a new network architecture known for its high accuracy in several tasks such as function fitting and PDE solving. The superior expressive capability of KAN arises from the Kolmogorov-Arnold representation theorem and learnable spline functions. However, the computation of spline functions involves multiple iterations, which renders KAN significantly slower than MLP, thereby increasing the cost associated with model training and deployment. The authors of KAN also noted that “the biggest bottleneck of KANs lies in their slow training. KANs are usually 10x slower than MLPs, given the same number of parameters.” To address this issue, we propose a novel MLP-type neural network PowerMLP that employs simpler non-iterative spline function representation, offering approximately the same training time as MLP while theoretically demonstrating stronger expressive power than KAN. Furthermore, we compare the FLOPs of KAN and PowerMLP, quantifying the faster computation speed of PowerMLP. Our comprehensive experiments demonstrate that PowerMLP generally achieves higher accuracy and a training speed about 40 times faster than KAN in various tasks.

# 1 Introduction

A long-standing problem of deep learning is the identification of more effective neural network architectures. The Kolmogorov-Arnold Network (KAN), introduced by Liu et al. (2024), presents a new architecture. Unlike traditional MLP that places activation functions on nodes, KAN employs learnable univariate spline functions as activation functions placed on edges. The expressive power of KAN is derived from the Kolmogorov-Arnold representation theorem and the property of spline functions. Due to its exceptional expressiveness, KAN achieves high accuracy and interpretability in multiple tasks such as function fitting and PDE solving. Significant performance improvements using KAN have been observed in time series prediction (Inzirillo and Genet 2024), graph data processing (Kiamari et al. 2024), and explainable natural language processing (Galitsky 2024).

![](images/bf1f7c7dced2ade093c906cea1cd0e171021f312605a49b08a1f58527282f87b.jpg)  
Figure 1: PowerMLPs define a strictly larger function space than KANs over $\mathbb { R } ^ { n }$ (Corollary 3), and define the same function space over $[ - E , E ] ^ { n }$ for any $E \in \mathbb { R } _ { + }$ (Corollary 7), where $n$ is the input dimension. $\mathcal { P } _ { d , w , k , p }$ is the set of all PowerMLPs with depth $d$ , width $w$ , $k$ -th power ReLU activation function, and $p$ nonzero parameters. $\mathcal { K } _ { d , w , k , G , p }$ is the set of all KANs with depth $d$ , width $w$ , using $( k , G )$ -spline (see Eq. (3)), and $p$ nonzero parameters.

Unfortunately, despite the impressive performance of KAN, it faces a critical drawback: slow inference and training speeds, which increase the cost associated with training and deploying the model. Liu et al. (2024) highlight in the KAN paper that “the biggest bottleneck of KANs lies in their slow training. KANs are usually 10x slower than MLPs, given the same number of parameters.” The inefficient training and inference latency of KAN plays a crucial role in ensuring a positive user experience and low computational resource requirements.

Upon examining the structure of KAN, we contribute the primary cause of the slow computation speed to the spline activation function. Specifically, a $k$ -order spline function is a linear combination of $k$ -order B-splines, and each $k$ - order B-spline requires construction through $\mathcal { O } ( k ^ { 2 } )$ iterations by the de Boor-Cox formula in KAN’s paper, resulting in slow computation. To address this issue, we need to identify a more efficient method to obtain B-splines, moving away from the recursive computation of the de Boor-Cox formula.

The $k$ -order B-spline was initially defined as $k$ -th divided difference of the truncated power function (Curry and Schoenberg 1947). Consequently, B-splines can also be expressed as linear combinations of powers of ReLUs (Greville 1969). Inspired by this, we introduce a novel MLP-type nueral network by incorporating a basis function into the MLP with powers of ReLU as activation function, termed PowerMLP, as shown in Figure 2. From Figure 1, PowerMLPs define a strictly larger function space than KANs over $\mathbb { R } ^ { n }$ and define the same function space over $[ - E , E ] ^ { n }$ , indicating that PowerMLP can serve as a viable substitute for KAN. Intuitively, the B-spline can be represented without iterative recursions in PowerMLP, leading to faster computation. In addition, the activation functions of PowerMLP are on the nodes instead of the edges, inhering the fast training advantages of MLP. We further choose Floating Point Operations (FLOPs) as the metric and demonstrate that the FLOPs of KAN are more than 10 times those of PowerMLP, theoretically explaining why PowerMLP is significantly faster than KAN.

We conducted extensive experiments to validate the advantages of PowerMLP in various experimental settings, including those of KAN and more complicated tasks such as image classification and language processing. The results show that PowerMLP trains about 40 times faster than KAN and achieves the best performance in most experiments.

# 2 Related Work

Network Architecture. Multilayer Perceptron (MLP) is a basic form of neural networks used primarily for supervised learning tasks (Haykin 1998). One of the activation functions commonly used with MLP is the rectified linear unit (ReLU) (Glorot, Bordes, and Bengio 2011), which is further extended to LeakyReLU (Maas et al. 2013), PReLU (He et al. 2015), RePU (Li, Tang, and Yu 2020a,b), and GeLU (Hendrycks and Gimpel 2023). LeCun et al. (1989) introduced CNNs specialized for images and Vaswani (2017) introduced Transformers, which have become a cornerstone for large language models. PowerMLP can be obtained from the RePU network (Li, Tang, and Yu 2020a,b) by adding a basis function to each layer.

Kolmogorov-Arnold Network. KAN (Liu et al. 2024) is a new network architecture using learnable activation functions on edges. In small-scale $\mathrm { A I } +$ Science tasks, KAN outperforms MLP in terms of both accuracy and interpretability. Additionally, KAN also achieves remarkable performance in time series prediction (Xu, Chen, and Wang 2024; Inzirillo and Genet 2024), graph-structured data processing (Kiamari et al. 2024; Carlo et al. 2024), and explainable natural language processing (Galitsky 2024). Meanwhile, notable approaches are taken to improve KAN’s interpretability and performance, including incorporating wavelet functions into KAN (Bozorgasl and Chen 2024), replacing the basis function with rational functions (Aghaei 2024), and faster implementation by approximation of radial basis functions (Li 2024). However, KAN and these variants have a computational speed significantly slower than MLP, since they have a structure similar to that of KAN.

PowerMLP vs. KAN. PowerMLP can be considered as an MLP-type representation of KAN. In Section 6 of (Liu et al. 2024), “KANs’ limitations and future directions” are discussed. Most of these limitations of KAN can be eliminated or mitigated by PowerMLP. (1) In algorithmic aspects, the authors noted that “KANs are usually $1 0 \mathrm { x }$ slower than MLPs.” PowerMLP, which is about 40 times faster than

KAN in our experiments, eliminates this limitation. (2) In mathematical aspects, the authors call for a more general approximation result beyond the depth-2 Kolmogorov-Arnold representations. This result is given in Corollary 8 using our proposed PowerMLP. (3) In the application aspects, the authors call for “integrating KANs into current architectures.” This can be done by replacing the MLP in these architectures with PowerMLP more naturally than KAN.

# 3 Preliminaries

In this section, we introduce the preliminary knowledge of Kolmogorov-Arnold Network (KAN) (Liu et al. 2024).

# 3.1 Spline Function

The following de Boor-Cox formula (Cox 1972; de Boor 1978) for $\mathbf { B }$ -spline is used to define KAN.

Definition 1 (B-spline). Let $t : = ( t _ { j } )$ be a nondecreasing sequence of real numbers, called knot sequence. The zeroorder $B$ -spline on $( t _ { j } , t _ { j + 1 } )$ is defined as 1

$$
B _ { j , 0 , t } ( x ) = \left\{ \begin{array} { r l r } { 1 , } & { { } } & { t _ { j } \leq x < t _ { j + 1 } , } \\ { 0 , } & { { } } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

Then the $j$ -th $k$ -order normalized B-spline for the knot sequence $t$ is defined recursively as

$$
\begin{array} { l } { \displaystyle B _ { j , k , t } ( x ) = \frac { x - t _ { j } } { t _ { j + k } - t _ { j } } B _ { j , k - 1 , t } ( x ) } \\ { \displaystyle + \frac { t _ { j + k + 1 } - x } { t _ { j + k + 1 } - t _ { j + 1 } } B _ { j + 1 , k - 1 , t } ( x ) , \quad \mathrm { f o r } k \geq 1 . } \end{array}
$$

To maintain consistency with KAN’s setting, let $\textit { t } =$ $\left( t _ { - k } , \cdot \cdot \cdot , t _ { - 1 } , t _ { 0 } , t _ { 1 } , \cdot \cdot \cdot , t _ { G } , t _ { G + 1 } , \cdot \cdot \cdot , t _ { G + k } \right)$ be an increasing knot sequence, called $a$ $( k , G )$ -grid. Through a linear combination of $\mathbf { B }$ -splines on $t$ , we provide the definition of spline functions as follows.

Definition 2 (Spline Function). Let t be a $( k , G )$ -grid. A $k$ - order spline function for the knot sequence $t$ is given by the following linear combination of $B$ -splines

$$
\mathrm { s p l i n e } _ { k , G } ( x ) = \sum _ { j = - k } ^ { G - 1 } c _ { j } B _ { j , k , t } ( x ) ,
$$

where $c _ { j } \in \mathbb { R }$ are the coefficients of the spline function. For simplicity, (3) is called a $( k , G )$ -spline function.

# 3.2 Kolmogorov-Arnold Networks

A KAN network (Liu et al. 2024) is a composition of $L$ layers: given an input vector $\mathbf { x } = ( x _ { 1 } , \cdot \cdot \cdot , \mathbf { \bar { x } } _ { n _ { 0 } } ) \in \mathbb { R } ^ { n _ { 0 } }$ , xn ) ∈ Rn0, the output of KAN is

$$
\begin{array} { r } { \mathrm { K A N } ( \mathbf { x } ) = ( \Phi _ { L - 1 } \circ \cdot \cdot \cdot \circ \Phi _ { 1 } \circ \Phi _ { 0 } ) ( \mathbf { x } ) , } \end{array}
$$

where $\Phi _ { \ell }$ is the function matrix corresponding to the $\ell { - } t h$ layer. The dimension of the input vector of $\Phi _ { \ell }$ is denoted as $n _ { \ell }$ , and $\Phi _ { \ell }$ is defined below:

$$
\Phi _ { \ell } ( \cdot ) = \left( { \begin{array} { c c c c } { \phi _ { \ell , 1 , 1 } ( \cdot ) } & { \phi _ { \ell , 1 , 2 } ( \cdot ) } & { \cdot \cdot \cdot } & { \phi _ { \ell , 1 , n _ { \ell } } ( \cdot ) } \\ { \phi _ { \ell , 2 , 1 } ( \cdot ) } & { \phi _ { \ell , 2 , 2 } ( \cdot ) } & { \cdot \cdot \cdot } & { \phi _ { \ell , 2 , n _ { \ell } } ( \cdot ) } \\ { \vdots } & { \vdots } & { \vdots } \\ { \phi _ { \ell , n _ { \ell + 1 } , 1 } ( \cdot ) } & { \phi _ { \ell , n _ { \ell + 1 } , 2 } ( \cdot ) } & { \cdot \cdot \cdot } & { \phi _ { \ell , n _ { \ell + 1 } , n _ { \ell } } ( \cdot ) } \end{array} } \right) ,
$$

1Note that de Boor-Cox formula defines the function in Eq. (1) as order 1. To keep the same mark with KAN, we adjust it to 0.

where $\phi _ { \ell , q , p }$ is a residual activation function:

$$
\phi _ { \ell , q , p } ( x _ { p } ) = u _ { \ell , q , p } b ( x _ { p } ) + v _ { \ell , q , p } \mathrm { s p l i n e } _ { k , G } ( x _ { p } ) .
$$

$b ( \boldsymbol { x } )$ is a non-parameter basis function 2 similar to residual shortcut, which is included to improve training. $\mathrm { s p l i n e } _ { k , G } ( x )$ is defined in Definition 2. The KAN defined above, called a KAN of $k$ -order, has depth $L$ , width $W =$ $\operatorname* { m a x } _ { i = 0 } ^ { L - 1 } n _ { i }$ , and $O ( W ^ { 2 } L ( G + k ) )$ parameters.

# 4 PowerMLP

While using spline functions as activations enables KAN to achieve excellent performance, computing spline functions involves multiple iterations, leading to a high number of FLOPs (see Table 1). This results in KAN’s computation speed being slower than that of MLP, thereby increasing the cost associated with training and deploying the model.

To replace KAN with a network structure that offers similar expressiveness but faster computation, we present a simpler representation of spline functions, avoiding the recursive calculations of the de Boor-Cox formula. In Section 4.1, we introduce a novel network, named PowerMLP. Through theorems in Sections 4.2 and 4.3, we demonstrate that KAN and PowerMLP define the same function space over bounded intervals, indicating their interchangeability. Furthermore, in Section 4.4, we compare the FLOPs of KAN and PowerMLP, revealing that PowerMLP theoretically achieves faster speeds than KAN. Proofs are given in Appendix A.

# 4.1 PowerMLP

Referring back to the de Boor-Cox formula in Eq. (2), we attribute the slow computation speed of KAN to the $\mathcal { O } ( k ^ { 2 } )$ iterations of calculation required for constructing $k$ -order Bspline. According to the initial definition of $k$ -order B-spline from the $k$ -th divided difference of the truncated power function (Curry and Schoenberg 1947) and subsequent work by Greville (1969), we express B-spline through a non-iterative approach as a linear combination of powers of ReLUs. Thus, we introduce a novel MLP network that incorporates a basis function into the MLP with powers of ReLU as the activation, termed PowerMLP.

The $k$ -th power of ReLU (Mhaskar 1993; Li, Tang, and Yu 2020a) is defined as:

$$
\sigma _ { k } ( x ) = \left( { \mathrm { R e L U } } ( x ) \right) ^ { k } = \left( { \mathrm { m a x } } ( 0 , x ) \right) ^ { k } \quad k \in \mathbb { Z } _ { + } .
$$

The fully connected feedforward neural network with the $k$ - th power of ReLU as the activation function is referred to as ReLU- $k$ MLP. We integrate a basis function into ReLU- $k$ MLP and define PowerMLP following the form of KAN.

Definition 3 (PowerMLP). A PowerMLP is a neural network composition of $L$ layers:

$$
\begin{array} { r l } & { \mathrm { P o w e r M L P } ( \mathbf x ) = ( \Psi _ { L - 1 } \circ \cdot \cdot \cdot \circ \Psi _ { 1 } \circ \Psi _ { 0 } ) ( \mathbf x ) , \quad \mathrm { w h e r e } } \\ & { \Psi _ { \ell } ( \mathbf x _ { \ell } ) = \left\{ \begin{array} { l l } { \alpha _ { \ell } b ( \mathbf x _ { \ell } ) + \sigma _ { k } ( \omega _ { \ell } \mathbf x _ { \ell } + \gamma _ { \ell } ) , } & { \mathrm { f o r } \ell < L - 1 , } \\ { \omega _ { L - 1 } \mathbf x _ { L - 1 } + \gamma _ { L - 1 } , } & { \mathrm { f o r } \ell = L - 1 . } \end{array} \right. } \end{array}
$$

2Basis function is $b ( x ) = x / ( 1 + e ^ { - x } )$ in KAN’s paper.

![](images/4a7076e38fae1c4f5b751497b3007c3ad1f4808b1a46e250bad96bfd8e1e513d.jpg)  
Figure 2: Structure of a 3-layer PowerMLP. The first two layers are calculated by: (1) affine transformation, (2) $k$ -th power of ReLU activation, (3) addition with a basis function. The last layer contains only an affine transformation.

$\alpha _ { \ell } \in \mathbb { R } ^ { m _ { \ell + 1 } \times m _ { \ell } } , \omega _ { \ell } \in \mathbb { R } ^ { m _ { \ell + 1 } \times m _ { \ell } } , \gamma _ { \ell } \in \mathbb { R } ^ { m _ { \ell } \times 1 }$ are trainable parameters. $b ( \mathbf { x } )$ is a basis function that performs the same operation as the basis function in KAN on each component of $\mathbf { x }$ , and $\sigma _ { k }$ is the ReLU- $k$ activation function. A PowerMLP using $\sigma _ { k }$ as the activation function is called $a$ $k$ -order PowerMLP. Width and depth of the PowerMLP are defined to be width $= \mathrm { m a x } _ { l = 0 } ^ { L - 1 } \{ \bar { m } _ { l } \}$ , $\mathrm { { d e p t h } } = L .$ . Refer to Figure 2 for an illustration.

Through non-parameter activation function $\sigma _ { k }$ and noniteration calculation, PowerMLP computes faster than KAN. Furthermore, they can represent each other within bounded intervals, a condition usually met in practical scenarios.

# 4.2 PowerMLP Can Represent KAN

In this section, we show that any KAN can be represented by a PowerMLP. We first give the following connection between B-spline and the $\mathbf { k }$ -th power of ReLU $\sigma _ { k }$ .

Lemma 1 (Represent the B-spline with $\mathbf { k }$ -th power of ReLU). If $t _ { u } \ne t _ { v } ( \forall u \ne v )$ , then the $k$ -order $B$ -spline on the knot sequence $t = ( t _ { j } , \cdot \cdot \cdot , t _ { j + k + 1 } )$ can be represented as a linear combination of $\sigma _ { k }$ functions:

$$
B _ { j , k , t } ( x ) = \sum _ { i = j } ^ { j + k + 1 } \frac { t _ { j + k + 1 } - t _ { j } } { \prod _ { l = j } ^ { j + k + 1 , l \ne i } ( t _ { l } - t _ { i } ) } \sigma _ { k } ( x - t _ { i } ) .
$$

Since each spline activation function in KAN is a linear combination of $\mathbf { B }$ -splines, we can represent it as a linear combination of ReLU- $k$ . By performing this operation on each component of the input vector $\mathbf { x }$ and incorporating a basis function, we represent a KAN layer as a PowerMLP layer. Since KAN and PowerMLP are both compositions of layers, it means that a KAN can be represented as a PowerMLP. We have the following theorem.

Theorem 2 (KAN is a Subset of PowerMLP). Fix the input dimension of networks as $n$ . Let $\mathcal { \kappa } _ { d , w , k , G , p }$ be the set of all KAN networks with depth $d$ , width $w$ , p nonzero parameters, using $( k , G )$ -spline; and $\mathcal { P } _ { d , w , k , p }$ be the set of all PowerMLPs with depth $d _ { \mathrm { { z } } }$ , width $w$ , $k$ -th power of ReLU and $p$ nonzero parameters. Then it holds

$$
\begin{array} { r } { \mathcal { K } _ { d , w , k , G , p } \subset \mathcal { P } _ { d , w ^ { 2 } ( G + k ) , k , p } . } \end{array}
$$

By Theorem 2 and the fact that a spline function is zero outside of certain interval (see Eq. (1)) while PowerMLPs include all polynomials (Li, Tang, and Yu 2020a), we have Corollary 3. PowerMLPs define a strictly larger function space than KANs over $\mathbb { R } ^ { n }$ .

# 4.3 KAN Can Represent PowerMLP over Intervals

In this section, we prove the inclusion relationship in another direction. A PowerMLP layer

$$
\mathbf { z } = \sigma _ { k } ( \omega \mathbf { x } + \boldsymbol { \gamma } ) + \alpha b ( \mathbf { x } )
$$

can be decomposed into 3 operations: (1) an affine transformation: $\mathbf { y } = \omega \mathbf { x } + \boldsymbol { \gamma }$ ; (2) a ReLU- $k$ activation: $\mathbf { u } = \sigma _ { k } ( \mathbf { y } )$ ; (3) an addition with basis function: $\mathbf { z } = \mathbf { u } + \alpha b ( \mathbf { x } )$ . By Lemmas 4 and 5, operations (1) and (2) can be represented by spline functions, while operation (3) can be easily achieved.

Lemma 4 (Affine Transformation). Consider an affine transformation on $\mathbb { R }$ : $\mathcal { A } ( x ) ~ = ~ \omega x + \gamma$ . For any $G$ , we can find a $( k , G )$ -grid $t ~ = ~ ( \underline { { t } } _ { - k } , \cdot \cdot \cdot , t _ { - 1 } , t _ { 0 } , t _ { 1 } , \cdot \cdot \cdot , t _ { G }$ , $\underline { { t _ { G + 1 } , \cdot \cdot \cdot , t _ { G + k } } } )$ , and a $k$ -order spline function

$$
\mathrm { s p l i n e } _ { k , G } ( x ) = \sum _ { j = - k } ^ { G - 1 } c _ { j } B _ { j , k , t } ( x ) ,
$$

where $c _ { j } = \textstyle \left( \sum _ { i = j + 1 } ^ { j + k } t _ { i } / k \right) \omega + \gamma , k > 0$ , such that $\boldsymbol { \mathcal { A } } ( \boldsymbol { x } ) =$ ${ \mathrm { p l i n e } } _ { k , G } ( x ) f o r t _ { 0 } \leq x \leq t _ { G }$ .

Lemma 5 (ReL $\scriptstyle { \mathrm { U } } - k { \mathrm { F } }$ unction). We can find a $( k , 2 )$ -grid $t =$ $\left( t _ { - k } , \cdot \cdot \cdot , t _ { - 1 } , t _ { 0 } , 0 , t _ { 2 } , t _ { 3 } , \cdot \cdot \cdot , t _ { k + 2 } \right)$ and a $k$ -order spline function defined on $t$

$$
\mathrm { s p l i n e } _ { k , 2 } ( \boldsymbol { x } ) = \sum _ { j = - k } ^ { 1 } \left[ \left( \prod _ { l = j + 1 } ^ { j + k } \sigma _ { 1 } ( t _ { l } ) \right) B _ { j , k , t } ( \boldsymbol { x } ) \right] ,
$$

such that $\sigma _ { k } ( x ) = \mathrm { s p l i n e } _ { k , 2 } ( x )$ for $t _ { 0 } \le x \le t _ { 2 }$ .

By the above lemmas, we can use two layers of spline functions to represent operations (1) and (2) successively, as shown in Figure 3. For the first layer, we set the coefficients before the basis function to be 0. By Lemma 4, we can represent affine transformation $\mathcal { A } ( x ) = \omega x + \gamma$ by activation functions $\phi _ { 1 , q , p } ( x _ { p } )$ (see Eq. (4)). Then, for $1 \leq p \leq m$ , we take the activation function as follows:

$$
\phi _ { 1 , q , p } ( x _ { p } ) = \left\{ \begin{array} { l l } { \omega _ { q , p } x _ { p } + \gamma _ { q , p } , } & { \mathrm { f o r } 1 \leq q \leq m , } \\ { \delta _ { q - m , p } x _ { p } , } & { \mathrm { f o r } 1 \leq q - m \leq n , } \end{array} \right.
$$

where $\delta _ { i j }$ equals to 1 if $i = j$ and 0 otherwise. Thus, we have $\begin{array} { r } { y _ { q } = \sum _ { p = 1 } ^ { n } \omega _ { q , p } x _ { p } + \gamma _ { q , p } } \end{array}$ for $1 \leq q \leq m$ and $y _ { q } = x _ { q - m }$ for $m + 1 \leq q \leq m + n$ . In the second layer, we represent the addition of ReLU- $k$ activation and the basis function. By Lemma 5, spline function can represent $\sigma _ { k }$ . So for $1 \leq r \leq$ $m$ , we set $\phi _ { 2 , r , q } ( y _ { q } )$ as follows:

$$
\phi _ { 2 , r , q } ( y _ { q } ) = \left\{ \begin{array} { c c } { { \delta _ { r , q } \sigma _ { k } ( y _ { q } ) , } } & { { \mathrm { f o r } 1 \leq q \leq m , } } \\ { { \alpha _ { r , q - m } b ( y _ { q } ) , } } & { { \mathrm { f o r } 1 \leq q - m \leq n . } } \end{array} \right.
$$

By direct computation, we show that the output of the two-layer KAN is the same as our PowerMLP layer in Eq. (8). Hence, we derive Theorem 6 and Corollary 7. Then by (Li, Tang, and $\mathtt { Y u 2 0 2 0 a }$ , Theorem 3.3), we obtain a general approximation result for KAN in Corollary 8.

![](images/4cd9f6b132b4a2c4d0c808e288f2f2ffbe9a2fa5543f6bd84cbb6bc492d22e0a.jpg)  
Figure 3: Represent a PowerMLP layer with a 2-layer KAN. $\delta _ { i j }$ equals to 1 if $i = j$ and 0 otherwise. The first layer represents the affine transformation $\begin{array} { r } { y _ { q } = \sum _ { p = 1 } ^ { n } \omega _ { q , p } x _ { p } + \gamma _ { q , p } } \end{array}$ for $1 \leq q \leq m$ and keeps $y _ { q } = x _ { q - m }$ for $m + 1 \leq q \leq m + n$ . The second layer represents the ReLU- $k$ activation and adds the basis function: $\begin{array} { r } { z _ { r } = \sigma _ { k } ( y _ { r } ) + \sum _ { q = m + 1 } ^ { m + n } \alpha _ { r , q - m } b ( y _ { q } ) } \end{array}$ .

Theorem 6 (PowerMLP is a subset of KAN over interval). Use notations in Theorem 2. For any $E \in \mathbb { R } _ { + }$ , it holds

$$
\mathcal { P } _ { d , w , k , p } \subset { \mathcal K } _ { 2 d , 2 w , k , 2 , \mathcal { O } ( k p ) } \mathrm { o v e r } [ - E , E ] ^ { n } .
$$

Corollary 7. PowerMLPs and KANs define the same function space over $[ - E , E ] ^ { n }$ for any $E \in \mathbb { R } _ { + }$ .

Corollary 8. Let $f$ be $a$ continuous, first-order differentiable function on $[ - 1 , 1 ] ^ { n }$ , which satisfies $\begin{array} { r } { \sum _ { i = 1 } ^ { n } \int _ { [ - 1 , 1 ] ^ { n } } ( \partial _ { x _ { i } } f ( x ) ) ^ { 2 } d x \leq 1 } \end{array}$ . Then, for any $\epsilon \in ( 0 , 1 )$ $a$ 2-order KAN $K$ requires at most $\mathcal { O } ( n \log _ { 2 } \frac { 1 } { \epsilon } )$ layers and $\mathcal { O } ( \epsilon ^ { - n } )$ nonzero parameters to ensure $\| K - f \| _ { L _ { 2 } } \leq \epsilon$ .

# 4.4 FLOPs: Comparison of Computing Cost

In this section, we show that PowerMLP exhibits significantly faster training and inference speeds compared to KAN in terms of FLOPs metric. FLOPs, an acronym for Floating Point Operations, is a metric used to quantify the computational complexity of neural networks, especially in frameworks like PyTorch (Molchanov et al. 2017). It measures the number of floating-point operations required to perform one forward pass through the network.

Following Yu, Yu, and Wang (2024), we consider FLOPs for any arithmetic operations like $+ , - , \times , \div$ to be 1, and for Boolean operations to be 0. Meanwhile, any operation of comparing two numbers is set to be 0 FLOPs, which means that the FLOPs of ReLU function are 0. We denote FLOPs of basis function as $\lambda$ . Then we can calculate FLOPs of one layer of MLP (with ReLU), KAN and PowerMLP below:

$$
\begin{array} { r l } & { \mathcal { F } _ { \mathrm { M L P } } = 2 d _ { i n } d _ { o u t } , } \\ & { \mathcal { F } _ { \mathrm { K A N } } = d _ { i n } d _ { o u t } \big ( 9 k G + 1 3 . 5 k ^ { 2 } + 2 G - 2 . 5 k + 3 \big ) + \lambda d _ { i n } , } \\ & { \mathcal { F } _ { \mathrm { P o w e r M L P } } = 4 d _ { i n } d _ { o u t } + ( k - 1 ) d _ { o u t } + \lambda d _ { i n } } \end{array}
$$

where $d _ { i n }$ and $d _ { o u t }$ denote the input and output dimensions of the layer. The KAN layer uses $( k , G )$ -grid, and the PowerMLP layer is $k$ -order. Refer to Appendix A for details.

Then we compare FLOPs of three network layers with same number of parameters, given by the ratio of FLOPs $\mathcal { F }$ to numbers of parameters:

$$
\begin{array} { l } { { r _ { \mathrm { M L P } } = \displaystyle \frac { 2 d _ { i n } d _ { o u t } } { d _ { i n } d _ { o u t } + d _ { o u t } } , } } \\ { { r _ { \mathrm { K A N } } = \displaystyle \frac { d _ { i n } d _ { o u t } \bigl ( 9 k G + 1 3 . 5 k ^ { 2 } + 2 G - 2 . 5 k + 3 \bigr ) + \lambda d _ { i n } } { d _ { i n } d _ { o u t } \bigl ( k + G + 2 \bigr ) } , } } \\ { { r _ { \mathrm { P o w e r M L P } } = \displaystyle \frac { 4 d _ { i n } d _ { o u t } + ( k - 1 ) d _ { o u t } + \lambda d _ { i n } } { 2 d _ { i n } d _ { o u t } + d _ { o u t } } . } } \end{array}
$$

Assuming that $d _ { i n }$ and $d _ { o u t }$ increase at the same rate, $r _ { \mathrm { M L P } }$ and rPowerMLP tend towards values less than 2, while $r _ { \mathrm { K A N } }$ approaches to numbers larger than 20 for $k \geq 3 , G \geq$ 3 $( k \ : = \ : 3 , G = 3$ are the smallest values used in KAN’s paper). Thus, PowerMLP shares a close computing speed with MLP, and is over 10 times faster than KAN under the FLOPs metric. We give an example of FLOPs of KAN, MLP and PowerMLP with almost the same number of parameters in Table 1. Training speed comparisons in experimental settings are in Section 5.3.

Table 1: Comparison of KAN, MLP and PowerMLP. With almost the same parameters, MLP and PowerMLP have much fewer FLOPs than KAN.   

<html><body><table><tr><td>Network</td><td>Shape</td><td>#Params</td><td>FLOPs</td></tr><tr><td>KAN (G=3,k = 3)</td><td>[2, 1,1]</td><td>24</td><td>564</td></tr><tr><td>MLP (ReLU)</td><td>[2,6,1]</td><td>25</td><td>36</td></tr><tr><td>PowerMLP(k = 3)</td><td>[2, 4, 1]</td><td>25</td><td>40</td></tr></table></body></html>

![](images/24203c406de2ade38626d6e4a96b55b9460a13f6b6f473797e021587cbad7a76.jpg)  
Figure 4: In the upper figure, PowerMLP can correctly find that 3 of 17 geometric invariants have influence on the output. Additionally, PowerMLP outperforms KAN in 15 of 17 input cases while KAN fails to converge with Symmetry $D _ { 3 }$ or $D _ { 8 }$ as input. In the bottom figure, trained on part or all of the 3 influencing geometric invariants, PowerMLP achieves much higher test accuracy than KAN in 3 cases.

# 5 Experiments

In Section 4, PowerMLPs are shown to define the same function space as KANs over bounded intervals and achieve faster computation. In this section, we employ several experiments to validate these theoretical findings and demonstrate the advantages of PowerMLP.

Four experiments are conducted. (1) We consider AI for science tasks in the KAN paper (Liu et al. 2024) under the same settings, showing that PowerMLP performs better. (2) More complex tasks such as machine learning, natural language processing, and image classification are considered. We show that PowerMLP outperforms KAN in all tasks. (3) We compare training and convergence time of KAN and PowerMLP, validating that PowerMLP can be much faster. (4) We conduct an ablation experiment to show that both the basis function and ReLU- $k$ activation are needed for the performance of PowerMLP.

All KANs in the paper are the latest version (0.2.5) up to 2024-8-14. More details are given in Appendix B.3

# 5.1 AI for Science Tasks

Function Fitting PowerMLP is tested on a regression task for 16 special functions in KAN’s experiments (Liu et al. 2024). For KAN, MLP, and PowerMLP, we choose two sizes for the networks: (1) Small size: KAN with $k = 3 , G = 3$ , shape [2, 1, 1], and 24 parameters; MLP with shape [2, 6, 1] and 25 parameters; 3-order PowerMLP with shape [2, 4, 1] and 25 parameters; (2) Large size: KAN with $k = 3 , G =$ 100, shape $[ 2 , 2 , 1 , 1 ]$ and 735 parameters; MLP with shape [2, 32, 18, 1] and 709 parameters; 3-order PowerMLP with shape [2, 32, 8, 1] and 689 parameters.

Results are given in Table 2. We see that PowerMLP achieves the best results on 11/10 (small size/large size) out of 16 cases. This is attributed to that PowerMLP has ReLU- $k$ for stronger power of expression than MLP and can better converge than KAN with a simpler structure of nonparameter activations.

Knot Theory Davies et al. (2021) used MLP to discover highly non-trivial relations among 17 geometric invariants and signatures in knot theory. With 17 geometric invariants as inputs and the corresponding signature as output, they trained MLPs and found a strong connection between signatures and 3 of all 17 geometric invariants: the longitudinal translation $\lambda$ , real and image part of meridional translation $\mu _ { r } , \mu _ { i }$ . Based on this observation, they proved a theorem and explained the connection theoretically, which is an interesting example of AI for math tasks (Davies et al. 2022).

Liu et al. (2024) reproduced the same experiment with KANs. Following their settings, we conduct the experiment with our PowerMLP. In Figure 4, we show the test accuracy of using a single geometric invariant as input each time to predict the signature. Most geometric invariants are close to random inputs, but the longitudinal translation, real and image part of the meridional translation show relationships with the signature. For better comparison, we also show results of a KAN with the same depth and almost same number of parameters. KAN performs worse than PowerMLP in 15 of 17 geometric invariants. In particular, KAN fails to converge with Symmetry $D _ { 3 }$ or $D _ { 8 }$ as input.

<html><body><table><tr><td rowspan="2">Function Name</td><td colspan="3">Small Size:~ 25 parameters</td><td colspan="3">Large Size:~ 7OO parameters</td></tr><tr><td>KAN</td><td>MLP</td><td>PowerMLP</td><td>KAN</td><td>MLP</td><td>PowerMLP</td></tr><tr><td>JE</td><td>4.63×10-3</td><td>4.98×10-3</td><td>5.79×10-4</td><td>1.04×10-4</td><td>5.88×10-4</td><td>7.23 × 10-5</td></tr><tr><td>IE1</td><td>1.34 × 10-2</td><td>5.79 × 10-3</td><td>3.43 × 10-3</td><td>4.52 × 10-5</td><td>5.57 × 10-4</td><td>3.37 × 10-5</td></tr><tr><td>IE2</td><td>1.16 × 10-2</td><td>4.71 × 10-3</td><td>1.73 × 10-3</td><td>1.18 × 10-3</td><td>5.98 × 10-4</td><td>3.20 × 10-5</td></tr><tr><td>B1</td><td>7.71 × 10-1</td><td>3.94 × 10-2</td><td>3.93 × 10-2</td><td>1.70 × 10-2</td><td>5.47 × 10-3</td><td>2.77 × 10-3</td></tr><tr><td>B2</td><td>7.94 × 10-2</td><td>6.02 × 10-2</td><td>7.75 × 10-2</td><td>1.77 × 10-3</td><td>4.62 × 10-3</td><td>2.61 ×10-3</td></tr><tr><td>MB1</td><td>2.29×10°</td><td>3.76 × 10-2</td><td>3.48 × 10-2</td><td>1.70 × 10-2</td><td>5.11 × 10-3</td><td>3.75 × 10-3</td></tr><tr><td>MB2</td><td>7.97 × 10-1</td><td>1.04 × 10-2</td><td>7.47 × 10-3</td><td>9.44 × 10-5</td><td>1.01 × 10-3</td><td>7.42 × 10-5</td></tr><tr><td>AL (m=0)</td><td>1.09 × 10-1</td><td>8.14 × 10-2</td><td>7.21 × 10-2</td><td>1.88 × 10-3</td><td>7.45 × 10-3</td><td>6.72 × 10-3</td></tr><tr><td>AL (m = 1)</td><td>1.25 × 10-1</td><td>7.49 × 10-2</td><td>6.94 × 10-2</td><td>1.40 ×10-2</td><td>1.20 × 10-2</td><td>1.02 × 10-2</td></tr><tr><td>AL (m = 2)</td><td>2.30 ×10-1</td><td>1.10 × 10-1</td><td>9.95 × 10-2</td><td>1.67 × 10-3</td><td>8.83 × 10-3</td><td>7.29 ×10-3</td></tr><tr><td>SH(m=0,n=1)</td><td>4.05 × 10-5</td><td>2.02 × 10-3</td><td>2.59 × 10-5</td><td>8.39 × 10-6</td><td>1.45 × 10-4</td><td>7.77 × 10-6</td></tr><tr><td>SH(m=1,n =1)</td><td>1.92 × 10-2</td><td>5.57 × 10-3</td><td>1.20 × 10-2</td><td>7.03 × 10-5</td><td>4.16 × 10-4</td><td>2.60×10-4</td></tr><tr><td>SH(m=0,n= 2)</td><td>5.94 × 10-5</td><td>4.46 × 10-3</td><td>4.79 × 10-4</td><td>1.02 × 10-5</td><td>3.17 × 10-4</td><td>2.55×10-5</td></tr><tr><td>SH(m=1,n = 2)</td><td>3.25×10-2</td><td>1.81 × 10-2</td><td>2.37 × 10-3</td><td>2.27× 10-3</td><td>8.90 × 10-4</td><td>1.45 × 10-4</td></tr><tr><td>SH(m=2,n = 2)</td><td>1.17 × 10-2</td><td>1.59 × 10-2</td><td>2.16× 10-2</td><td>1.09 × 10-2</td><td>8.32 ×10-2</td><td>3.66 × 10-5</td></tr></table></body></html>

Table 2: Fitting Special Functions. JE: Jacobian elliptic functions, IE1/IE2: Incomplete elliptic integral of the first/second kind, B1/B2: Bessel function of the first/second kind, MB1/MB2: Modified Bessel function of the first/second kind, AL: Associated Legendre function, SH: spherical harmonics. All the values are test RMSE loss, the less the better. The best results are marked as boldface. PowerMLP achieves the best results on 11/10 (small/large size) out of 16 cases.

Furthermore, in Figure 4, we show test accuracy of PowerMLPs trained on all or part of the three relevant geometric invariants $\lambda , \mu _ { i }$ and $\mu _ { r }$ . With a test accuracy of $8 6 . 7 4 \%$ , PowerMLP successfully validates the connection among signature and $\lambda$ , $\mu _ { i }$ , $\mu _ { r }$ . Furthermore, with a test accuracy of $8 0 . 6 2 \%$ , PowerMLP can also find the connection among $\lambda , \mu _ { r }$ and the signature discovered by KAN. More importantly, compared to the close test accuracy of KAN and PowerMLP in single geometric invariant input, PowerMLP achieves much higher test accuracy than KAN in 3 out of 4 cases of combination input. This indicates that PowerMLP can better utilize the correlation between inputs.

# 5.2 More Complex Tasks

We perform three more complex tasks. We use three networks with approximately the same number of parameters.

Machine Learning For basic machine learning tasks, we conduct two experiments on Titanic and Income (Becker and Kohavi 1996), which are classification tasks of small input dimension. In Figure 5, we show the test accuracy of three networks, and PowerMLP outperforms MLP and KAN.

Natural Language Processing We conduct two experiments on SMS Spam Collection (Spam) (Go´mez Hidalgo et al. 2006) and AG NEWS (Zhang, Zhao, and LeCun 2015) dataset, which are text classification tasks. We use TF-IDF transformation (Ramos et al. 2003) to convert text into vectors, and networks need to deal with high-dimensional sparse inputs. In Figure 5, we show the test accuracy of three networks, and PowerMLP outperforms MLP and KAN.

![](images/890310d0ab2c7246f08402861c33e244aacd98ed1f1758a2f6a3187a6ee94b2d.jpg)  
Figure 5: Test accuracy on multiple classification tasks.

Image Classification For image classification tasks, we conduct two experiments on MNIST (LeCun et al. 1998) and SVHN (Netzer et al. 2011) datasets. We convert SVHN data to greyscale images, and flatten image tensor into 1- dimension vectors. In this task, networks need to deal with high-dimensional inputs with strong connection among different input dimensions. In Figure 5, we show the test accuracy of three networks, and PowerMLP outperforms KAN.

In summary, with almost the same number of parameters and the same depth, PowerMLP achieves better accuracy than KAN in all tasks. PowerMLP outperforms MLP in machine learning (Income, Titanic) and language processing (AG NEWS, Spam), while performing worse in image classification (SVHN, MNIST).

Table 3: Training times on 8 tasks. Training times of PowerMLP are about 40 times smaller than KAN on average.   

<html><body><table><tr><td rowspan="2">Tasks</td><td rowspan="2">#Params</td><td colspan="3">Time(s)</td></tr><tr><td>KAN</td><td>MLP</td><td>PowerMLP</td></tr><tr><td>JE</td><td>~25</td><td>17.73</td><td>0.583</td><td>0.738</td></tr><tr><td>IE1</td><td>~25</td><td>25.32</td><td>0.430</td><td>0.674</td></tr><tr><td>IE2</td><td>~25</td><td>31.91</td><td>0.584</td><td>0.709</td></tr><tr><td>B1</td><td>~25</td><td>29.68</td><td>0.422</td><td>0.646</td></tr><tr><td>B2</td><td>~25</td><td>35.20</td><td>0.559</td><td>0.696</td></tr><tr><td>Titanic</td><td>~100</td><td>35.07</td><td>0.529</td><td>0.591</td></tr><tr><td>Spam</td><td>~800</td><td>62.06</td><td>0.568</td><td>0.655</td></tr><tr><td>SVHN</td><td>~1.3 × 105</td><td>89.82</td><td>1.53</td><td>2.39</td></tr></table></body></html>

![](images/65b3d769d7fd4ef20a0fc74accb71a10a9998722c4f7b9e5a6b061491bfbd938.jpg)  
Figure 6: Time of convergence. MLP and PowerMLP converge much faster than KAN.

# 5.3 Training Time

In Table 3, eight tasks are considered. The first five tasks are function regression tasks in Section 5.1, and the last three tasks are machine learning, language processing, and image classification tasks in Section 5.2. For better comparison, the experiments are on a single NVIDIA GeForce RTX 4090 GPU, repeated each task 10 times to take an average, and networks in each task are trained with the same hyperparameters. From Table 3, among all tasks and in different numbers of parameters, the training time of PowerMLP is close to MLP, which is about 40 times less than KAN. This is consistent with our theoretical analysis in Section 4.4.

To be more comprehensive, in Figure 6, we give the variation curves of the test RMSE loss relative to the training time in one training progress. Two curves with similar colors represent two training processes of the same network. We can see that MLP and PowerMLP also converge much faster than KAN in terms of training time.

# 5.4 Ablation Study

We show that both the basis function and the ReLU- $k$ are useful in PowerMLP. In Figure 7, we compare three networks, PowerMLP, PowerMLP without basis function, and PowerMLP without ReLU- $k$ activation (ReLU instead) on a function regression task for $f ( x , y ) = x \exp ( - y )$ .

The upper graph shows the RMSE loss on the test set with variant network depth for fixed width of 4. Although PowerMLPs without basis function perform well in shallow networks, they fail to converge for large depth. The bottom graph shows the influence of the number of parameters, with a fixed network depth of 2. PowerMLPs without basis function achieve nearly the same test RMSE loss with PowerMLP for some networks, while performing much worse in other cases, indicating that basis function enhances training stability. In all situations, PowerMLPs perform better than PowerMLPs without ReLU- $k$ , consistent with ReLU$k$ ’s better expressive ability.

![](images/365f678b2260f3eedd635e735d5b311eb0bdb8d6cb567bc5a8f6b6c2127fd841.jpg)  
Figure 7: Ablation study. Basis function enhances training stability, while ReLU- $k$ improves expressive ability.

# 6 Conclusion

In this paper, we introduce a novel neural network architecture, PowerMLP, which employs the powers of ReLU as activation and expresses linear combination as inner product to represent spline function. PowerMLP can be viewed as a more efficient and stronger version of KAN. In terms of expressiveness, PowerMLPs define a larger or equal function space than KANs. In terms of efficiency, PowerMLPs can be trained with over 10 times fewer FLOPs than KANs. We conducted comprehensive experiments to demonstrate that PowerMLPs achieve higher accuracy in most cases and can be trained about 40 times faster compared to KANs.

Limitations and Future Works. Despite its powerful expressive abilities, a limitation of our proposed PowerMLP is that it fails to achieve exceptional performance in complicated computer vision tasks and long-text processing. This stems from PowerMLP’s relatively simple architecture, which lacks specialized structures such as convolutional layers or attention mechanisms. However, given that PowerMLP shares a similar underlying architecture with traditional MLP, it is feasible to substitute the MLP parts in CNNs or Transformers with PowerMLP naturally and expect that more complicated problems can be solved better by using PowerMLP as a basic building block.