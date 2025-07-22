# OTLRM: Orthogonal Learning-based Low-Rank Metric for Multi-Dimensional Inverse Problems

Xiangming Wang1, Haijin $\mathbf { Z e n g } ^ { 2 * }$ , Jiaoyang Chen1, Sheng Liu3, Yongyong Chen1\*, Guoqing Chao4

1 Harbin Institute of Technology (Shenzhen) 2 Gent University 3 University of Electronic Science and Technology of China 4 Harbin Institute of Technology (Weihai) xmwang $2 8 @$ gmail.com

# Abstract

In real-world scenarios, complex data such as multispectral images and multi-frame videos inherently exhibit robust lowrank property. This property is vital for multi-dimensional inverse problems, such as tensor completion, spectral imaging reconstruction, and multispectral image denoising. Existing tensor singular value decomposition (t-SVD) definitions rely on hand-designed or pre-given transforms, which lack flexibility for defining tensor nuclear norm (TNN). The TNNregularized optimization problem is solved by the singular value thresholding (SVT) operator, which leverages the tSVD framework to obtain the low-rank tensor. However, it is quite complicated to introduce SVT into deep neural networks due to the numerical instability problem in solving the derivatives of the eigenvectors. In this paper, we introduce a novel data-driven generative low-rank t-SVD model based on the learnable orthogonal transform, which can be naturally solved under its representation. Prompted by the linear algebra theorem of the Householder transformation, our learnable orthogonal transform is achieved by constructing an endogenously orthogonal matrix adaptable to neural networks, optimizing it as arbitrary orthogonal matrices. Additionally, we propose a low-rank solver as a generalization of SVT, which utilizes an efficient representation of generative networks to obtain low-rank structures. Extensive experiments highlight its significant restoration enhancements.

# Code — https://github.com/xianggkl/OTLRM Extended version — https://arxiv.org/abs/2412.11165

# Introduction

Real-world multi-dimensional data, such as multispectral images (MSIs), videos, and Magnetic Resonance Imaging (MRI) data, are usually affected by unpredictable factors during capture and transmission. To reconstruct the original tensors, many tasks are extended under different observations, such as Tensor Completion (TC) (Qin et al. 2022; Mai, Lam, and Lee 2022), spectral imaging (Cai et al. 2022), and spectral denoising (Wang et al. 2022a).

Current approaches have achieved outstanding results benefiting from the low-rank nature of the original data tensor. However, the tensor rank is still not well defined. Various definitions of the tensor rank are not deterministic and all have specific advantages and limitations. For example, the CANDECOMP PARAFAC (CP) rank (Carroll and Chang 1970) is defined as the number of rank-1 tensors obtained by the CP decomposition. Since computing the CP decomposition of a tensor is NP-hard and finding its accurate convex approximation is challenging, CP rank may not be appropriate to give solutions tailored to practical application areas. Additionally, Tucker rank (Tucker 1966; Sun, Vong, and Wang 2022) is based on the tensor unfolding scheme, which unfolds the tensor into matrices along different dimensions. As a result, it leads to a broken structure of the original tensor, and such an unfolding describes the correlation between only one mode of the tensor and all other modes (one mode versus the rest), which may bring undesirable results to the tensor reconstruction. Overall, how to define an appropriate tensor rank in different tensor decompositions is a problem worth discussing and analyzing.

With the tensor-tensor product (t-product) (Kernfeld, Kilmer, and Aeron 2015) gradually becoming a comprehensive approach for tensor multiplication, the tensor Singular Value Decomposition (t-SVD) (Zhang et al. 2014; Lu et al. 2019; Liu et al. 2024) which constructs the original tensor as the t-product of two orthogonal tensors and an $f$ -diagonal tensor, has received widespread attention and research. Explicitly, the t-product is defined in a transform domain based on an arbitrary invertible linear transform. Given a tensor $\mathcal { X } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ and a transform matrix $\mathbf { L } \in \mathbb { R } ^ { n _ { 3 } \times n _ { 3 } }$ , the transformed tensor $L ( \mathcal { X } )$ can be formulated as

$$
L ( \mathcal { X } ) = \mathcal { X } \times _ { 3 } \mathbf { L } ,
$$

where $\times { _ 3 }$ denotes the mode-3 tensor product. Following the definition, t-SVD can be defined in the transform domain, which captures the low-rank structure of the tensor. Based on some fixed linear invertible transforms such as Discrete Fourier Transform (DFT) (Zhang and Aeron 2016) and Discrete Cosine Transform (DCT) (Lu 2021), current transforms have theoretical guarantees, yet are fixed and not data-adaptive, which cannot fit different data instances well.

Under various t-SVD definitions of the transform domain, there are many algorithms and models for solving the lowrank problem. Since directly minimizing the tensor tubal rank of a tensor is an NP-hard problem, the tensor nuclear norm (TNN) (Zhang et al. 2014) was proposed as its convex approximation, which is

$$
\| \mathcal { X } \| _ { \mathbf { L } , * } : = \sum _ { i = 1 } ^ { n _ { 3 } } \| L ( \mathcal { X } ) ( : , : , i ) \| _ { * } ,
$$

where $\| \cdot \| _ { * }$ is the matrix nuclear norm and $\| \cdot \| _ { \mathbf { L } , * }$ is the tensor nuclear norm based on the transform L. To solve the matrix nuclear norm, the singular value thresholding (SVT) operator was proposed (Cai, Cande\`s, and Shen 2010), which can be formulated as:

$$
\operatorname { S V T } _ { \gamma } ( \mathbf { X } ) = \mathbf { U S } _ { \gamma } \mathbf { V } ^ { \mathbf { T } } ,
$$

where ${ \bf S } _ { \gamma } = m a x \{ { \bf S } - \gamma , 0 \}$ . With the assistance of SVT, recent models utilizing optimization algorithms (Wang et al. $2 0 2 2 \mathrm { a }$ ; Qin et al. 2022) and deep unfolding networks (Mai, Lam, and Lee 2022) have demonstrated promising outcomes. However, training neural networks based on SVD (Ionescu, Vantzos, and Sminchisescu 2015a; Wang et al. 2023) poses challenges due to its complexity and numerical instability when handling derivatives of eigenvectors. This can result in sub-optimal convergence during the training process. Despite the robust differentiability of eigenvectors, the partial derivatives may become large in cases where two eigenvalues are equal or nearly equal (Wang et al. 2022b).

To achieve a data-adaptive and theoretically invertible transform, fostering compatibility with deep networks, we introduce a novel approach: the learnable Orthogonal Transform-induced generative Low-Rank t-SVD Model (OTLRM). Compared with existing t-SVD methods, our OTLRM has two merits: (1) Orthogonality and data adaptability. We construct the learnable orthogonal transform L by several learnable Householder transformations, possessing inherent orthogonality and adaptability. It enables the adjustment of the transform for each dataset while maintaining a theoretical guarantee for the requirement of “arbitrary invertible linear transform”. While current t-SVD methods (such as DTNN (Kong, Lu, and Lin 2021) and Tensor $Q$ -rank (Jiang et al. 2023)) can only accommodate one or the other. (2) Generative framework. Under the t-SVD representation, OTLRM can directly generate the expected tensor with the guidance of the observations within the DNN optimization framework. While others decompose the target tensor by SVD and truncate the singular values. Especially, we introduce a dense rank estimation operator, which stores and enriches the rank information of each band in the transform domain. Primary contributions are outlined as follows:

• Prompted by the linear algebra theorem of the Householder transformation, we construct a learnable endogenously orthogonal transform into the neural network. Different from the predefined orthogonal transform, the proposed learnable endogenously orthogonal transform can be naturally embedded in the neural network, which has more flexible data-adaptive capability. • With the endogenously orthogonal transform, we propose a low-rank solver as a generalization of SVT, which utilizes efficient t-SVD representation to obtain low-rank structures. In contrast to SVT-induced low-rank algorithms, OTLRM is solved by gradient descent-based algorithms within the DNN optimization framework.

• We conduct extensive experiments on several tasks: TC, snapshot compressive imaging, spectral denoising, under three types of datasets: MSIs, videos and MRI data. Abundant and superior experimental results validate the effectiveness of our method.

# Related Works

Transform-based t-SVD. Inspired by TNN (Zhang et al. 2014) in the Fourier domain, many t-SVD algorithms have been proposed. Specifically, Zhang et al. proposed the DFTbased t-SVD method (Zhang and Aeron 2016). Zhou et al. (Zhou and Cheung 2019) introduced the Bayesian version of tensor tubal rank to automatically determine the tensor rank. The weighted TNN (Mu et al. 2020) is also proposed to distinguish different tensor singular values under Fast Fourier Transform (FFT). Considering that t-product is defined on any invertible linear transforms, other transforms are explored. For example, Lu et al. (Lu, Peng, and Wei 2019) proposed the DCT-induced TNN with theoretical guarantees for exact recovery. While Song et al. (Song, Ng, and Zhang 2020) employed unitary transform matrices captured from the original datasets. However, fixed transforms can not be inherently suitable for current data instances. Therefore, Jiang et al. (Jiang et al. 2023) proposed a dictionarybased TNN (DTNN), which constructed a learnable dataadaptive dictionary as the transform. Additionally, Kong et al. (Kong, Lu, and Lin 2021) gave a definition of the tensor rank with a data-dependent transform, based on the learnable matrix Q. Recently, Luo et al. (Luo et al. 2022) induced DNN as the transform and generated the transformed tensor by gradient descent algorithms. Due to the nonlinear transform used in the DNN, it does not fulfill the theoretical requirements for arbitrary invertible linear transforms.

SVT in Neural Network. Cai et al. proposed the SVT algorithm for matrix completion (Cai, Cande\`s, and Shen 2010). To implement the SVT algorithm into the deep learning network, Ionescu et al. gave a sound mathematical apparatus and derived the gradient propagation formulas for SVD in deep networks (Ionescu, Vantzos, and Sminchisescu 2015a,b). However, although robustly calculating the derivatives of the SVD gradient propagation is direct, it becomes numerically unstable during the calculation of certain partial derivatives. Wang et al. introduced a Taylor expansionbased approach to compute the eigenvector gradients (Wang et al. 2022b) and Song et al. induced the orthogonality loss to improve the generalization abilities and training stability of the SVD (Song, Sebe, and Wang 2022). These approaches involve integrating the matrix-based SVT directly into deep neural networks, which still introduces complex SVD to solve low-rank problems.

# Notations and Preliminaries

# Notations

In this paper, scalars, vectors, matrices, and tensors are denoted respectively as lowercase letters, e.g. $a$ , boldface lowercase letters, e.g. a, boldface capital letters, e.g. A and boldface Calligraphy letters, e.g. $\mathcal { A }$ . $\mathbf { \mathcal { A } } ^ { ( i ) }$ represents the $i$ -th frontal slice of the tensor $\mathcal { A }$ .

# Orthogonal Transform

Orthogonal transform could maintain the original orthogonality between vectors without losing fine-level details. Based on the following linear algebra theorem, an orthogonal matrix L is constructed, which satisfies LTL = LLT = I and I is the identity matrix. Proofs of Theorem 1 are presented in the supplementary material.

Theorem 1. (Uhlig 2001) Every real orthogonal $n \times n$ matrix L is the product of at most n real orthogonal Householder transformations. And this also is true for complex unitary L and complex Householders.

Following the lines of Theorem 1, we can randomly generate a parameter matrix $\mathbf { W } \in \mathbb { R } ^ { n \times n }$ containing $n$ column vectors $\mathbf { w } _ { i } ~ \in ~ \mathbb { R } ^ { n \times 1 }$ , and construct $n$ orthogonal Householder transformations as below:

$$
\mathbf { F } _ { i } = \mathbf { I } - 2 \frac { \mathbf { w } _ { i } \mathbf { w } _ { i } ^ { \mathbf { T } } } { \| \mathbf { w } _ { i } \| ^ { 2 } } ,
$$

where $1 \leq i \leq n$ . Then the orthogonal matrix $\mathbf { L }$ is represented as the product of the $n$ orthogonal Householder transformations $\mathbf { F } _ { i }$ :

$$
\mathbf { L } = \mathbf { F } _ { 1 } \mathbf { F } _ { 2 } \cdot \cdot \cdot \mathbf { F } _ { n } = \prod _ { i = 1 } ^ { n } ( \mathbf { I } - 2 { \frac { \mathbf { w } _ { i } \mathbf { w } _ { i } ^ { \mathbf { T } } } { \| \mathbf { w } _ { i } \| ^ { 2 } } } ) ,
$$

and it is worth noting that $\mathbf { W }$ is the parameter matrix to be optimized and $\mathbf { L }$ is the endogenous orthogonal matrix which is another form of W based on the Householder transformation operations of Eq. (4) and Eq. (5).

In contrast to approaches that attain orthogonality by incorporating an orthogonality metric into the loss function, matrices formed through Householder transformations possess inherent orthogonality and learnability. This intrinsic property eliminates the need for external orthogonal constraints and ensures identical learnable parameters as a conventional transform matrix. Specifically, for optimizing the orthogonal transform, this chain multiplication enables the loss to be derived parallelly for each column vector $\mathbf { w } _ { i }$ . And all the optimization can be automatically solved by the builtin differentiation engine in PyTorch, facilitating straightforward integration into deep neural networks.

# $L$ -transformed Tensor Singular Value Decomposition

Distinguished from matrix SVD, t-SVD is based on the tensor-tensor product, which is defined directly in a transform domain with an arbitrary invertible linear transform $\mathbf { L }$ (Kernfeld, Kilmer, and Aeron 2015).

Definition 1. Mode-3 tensor-matrix product (Kolda and Bader 2009) For any third-order tensor $\mathcal { A } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ with a matrix $\mathbf { U } \in \dot { \mathbb { R } ^ { n \times n _ { 3 } } }$ , the mode-3 tensor-matrix product is defined as

$$
{ \hat { \cal A } } = { \cal A } \times _ { 3 } { \bf U } \Leftrightarrow \hat { \bf A } _ { ( 3 ) } = { \bf U } { \bf A } _ { ( 3 ) } ,
$$

where $\hat { \mathcal { A } } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n }$ , $\mathbf { A } _ { ( 3 ) }$ and $\hat { \mathbf { A } } _ { ( 3 ) }$ are mode-3 matricization of $\mathcal { A }$ and $\hat { \cal A }$ , respectively.

Definition 2. Tensor-Tensor face-wise product (Kernfeld, Kilmer, and Aeron 2015) Given two tensors $\mathcal { A } \in$

$\mathbb { R } ^ { n _ { 1 } \times \ell \times n _ { 3 } }$ and $B \in \mathbb { R } ^ { \ell \times n _ { 2 } \times n _ { 3 } }$ , the face-wise product of $\mathcal { A }$ and $\boldsymbol { B }$ is defined as

$$
( \boldsymbol { \mathcal { A } } \triangle \boldsymbol { \mathcal { B } } ) ^ { ( i ) } = \boldsymbol { \mathcal { A } } ^ { ( i ) } \boldsymbol { \mathcal { B } } ^ { ( i ) } ,
$$

where $\mathbf { \mathcal { A } } ^ { ( i ) }$ is $i$ -th frontal slice of $\mathcal { A }$ .

Definition 3. Tensor-tensor product in L-transform domain (Kernfeld, Kilmer, and Aeron 2015) Define $* _ { L }$ as $\mathbb { R } ^ { n _ { 1 } \times \ell \times n _ { 3 } } \times \mathbb { R } ^ { \ell \times n _ { 2 } \times n _ { 3 } }  \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ , we have the tensortensor product:

$$
\mathcal { A } * _ { L } \mathcal { B } = L ^ { - 1 } ( L ( \mathcal { A } ) \triangle L ( \mathcal { B } ) ) ,
$$

where $L ( \mathcal { A } ) = \mathcal { A } \times _ { 3 } \mathbf { L }$ and $L ^ { - 1 } ( \cdot )$ is the inverse transform operator of $L ( \cdot )$ .

Definition 4. Special tensors (Braman 2010; Kilmer et al. 2013) Identity tensor: for an identity tensor $\boldsymbol { \mathcal { T } }$ , its every frontal slice is an identity matrix. $f$ -diagonal tensor: for an $f$ -diagonal tensor $s$ , its every frontal slice is a diagonal matrix. Orthogonal tensor: if the tensor $\mathcal { U }$ satisfies that ${ \mathcal { U } } ^ { \mathbf { T } } * _ { L } { \mathcal { U } } = { \bar { \mathcal { U } } } * _ { L } { \mathcal { U } } ^ { \mathbf { T } } = { \mathcal { T } }$ , it is called orthogonal tensor. Semi-orthogonal tensor: if the tensor $\mathcal { U }$ satisfies $\mathcal { U } ^ { \mathbf { T } } * _ { L } \mathcal { U } = \mathcal { T }$ , it is called semi-orthogonal tensor.

Subsequently, the definition of the $\mathbf { \Delta t } { \mathbf { - } } { \mathbf { S } } { \mathbf { V } } { \mathbf { D } }$ is expressed in terms of the $\mathbf { L }$ -transform and its inverse.

Lemma 1. Tensor singular value decomposition (t-SVD) (Braman 2010; Kilmer and Martin 2011; Kilmer et al. 2013) Given a tensor $\mathcal { X } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ , the t-SVD can be formulated as

$$
\mathcal { X } = \mathcal { U } * _ { L } \mathcal { S } * _ { L } \mathcal { V } ^ { \mathbf { T } } ,
$$

where $\mathcal { U } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 1 } \times n _ { 3 } }$ , $\boldsymbol { \mathcal { V } } \in \mathbb { R } ^ { n _ { 2 } \times n _ { 2 } \times n _ { 3 } }$ are orthogonal tensors and $S \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ is $f$ -diagonal.

Definition 5. Tensor tubal-rank (Zhang et al. 2014) The tensor tubal-rank $r$ of the target tensor $\chi$ is defined as the number of the non-zero singular tubes of the $f$ -diagonal tensor $S$ , which is

$$
r a n k ( \mathcal { X } ) = \# \{ i , S ( i , i , : ) \neq 0 \} .
$$

And an alternative definition is that the tenor tubal-rank of $\chi$ is the largest rank of every frontal slice of the $L$ -transformed tensor $\bar { L ( \mathcal { X } ) }$ .

Remark 1. Skinny t-SVD (Kilmer et al. 2013; Zhang and Aeron 2016) Given a tensor $\mathcal { X } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ which has tensor tubal-rank $r$ , it’s more efficient to compute the skinny t-SVD. And the decomposition can be reformulated as

$$
\mathcal { X } = \mathcal { U } * _ { L } \mathcal { S } * _ { L } \mathcal { V } ^ { \mathbf { T } } ,
$$

where $\mathcal { U } \in \mathbb { R } ^ { n _ { 1 } \times r \times n _ { 3 } }$ , $\boldsymbol { \mathcal { V } } \in \mathbb { R } ^ { n _ { 2 } \times r \times n _ { 3 } }$ are semi-orthogonal tensors, $\boldsymbol { S } \in \mathbb { R } ^ { r \times r \times n _ { 3 } }$ is $f$ -diagonal and especially, $\bar { \mathcal { U } } ^ { \mathbf { T } } * _ { L }$ $\mathcal { U } = \mathcal { T }$ , $\gamma ^ { \mathbf { T } } * _ { L } \gamma = \mathcal { T }$ .

Lemma 2. Tensor Singular Value Thresholding (t-SVT) (Wang et al. 2022a) Let $\mathcal { X } = \mathcal { U } * _ { L } \mathcal { S } * _ { L } \mathcal { V } ^ { \mathbf { T } }$ be the t-SVD for tensor $\mathcal { X } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ . The t-SVT operator $\delta$ is:

$$
\delta _ { \gamma } ( \mathcal { X } ) = \mathcal { U } * _ { L } S _ { \gamma } * _ { L } \mathcal { V } ^ { \mathbf { T } } ,
$$

where $S _ { \gamma } = L ^ { - 1 } ( \operatorname* { m a x } \{ L ( S ) - \gamma , 0 \} )$ and $\gamma$ is the threshold value which controls the degree of the rank restriction.

# Learnable Endogenous Orthogonal Transform based Generative t-SVD Low-rank Model

Suppose that $\mathcal { V } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ denotes the unknown pure tensor (real scene or video), $\mathcal { X } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ is the desired low-rank tensor and $\mathbf { H } ( \cdot )$ is defined as the capture operation that obtains the source data (hence, ${ \bf { H } } ( \boldsymbol { y } )$ is the observed measurement). Based on the definition of transposition and orthogonality of the transform of the t-product, the desired low-rank tensor can be generated by the skinny low-rank tSVD representation, which is $\mathcal { X } = \dot { \mathcal { U } } \ast _ { L } \mathcal { S } \ast _ { L } \dot { \mathcal { V } } ^ { \mathbf { T } }$ . With the endogenous orthogonal transform $\mathbf { L } \in \mathbb { R } ^ { n _ { 3 } \times n _ { 3 } }$ (and the parameter matrix to be optimized is $\mathbf { W }$ ) which naturally satisfies ${ \bf L } ^ { \bf T } { \bf L } = { \bf L } { \bf L } ^ { \bf T } \dot { = } { \bf \Omega } { \bf I }$ , our generative t-SVD low-rank optimization model can be formulated as

$$
\operatorname* { m i n } _ { \boldsymbol { u } , \boldsymbol { \nu } , \boldsymbol { s } , \mathbf { w } } \phi ( \mathbf { H } ( \boldsymbol { \chi } ) , \mathbf { H } ( \boldsymbol { \mathcal { V } } ) ) , \boldsymbol { \mathrm { s . t . } } \boldsymbol { \mathcal { X } } = \mathcal { U } \ast _ { L } \boldsymbol { S } \ast _ { L } \boldsymbol { \mathcal { V } } ^ { \mathbf { T } } ,
$$

where $\phi ( \cdot )$ is the fidelity loss function which can be adjusted according to the target application, $\mathcal { U } \in \mathbb { R } ^ { n _ { 1 } \times r \times n _ { 3 } }$ , $\boldsymbol { \mathcal { V } } \in \mathbb { R } ^ { n _ { 2 } \times r \times n _ { 3 } }$ and $S \in \mathbb { R } ^ { r \times r \times n _ { 3 } }$ . And explicitly, $* _ { L }$ denotes the orthogonal transform L-based tensor-tensor product according to Eq. (5) and Eq. (8).

Since $s$ is an $f$ -diagonal tensor, we construct the rank tensor by a transformed matrix $\mathbf { S } \in \mathbb { R } ^ { n _ { 3 } \times r }$ instead, with the diagonalization $D i a g ( \cdot )$ . It is reformulated as (please refer to the supplementary material for details):

$$
\operatorname* { m i n } _ { \mathcal { U } , \mathcal { V } , \mathbf { S } , \mathbf { W } } \phi \big ( \mathbf { H } ( L ^ { - 1 } ( L ( \mathcal { U } ) \triangle D i a g ( \mathbf { S } ) \triangle L ( \mathcal { V } ) ^ { \mathbf { T } } ) ) , \mathbf { H } ( \mathcal { V } ) \big ) .
$$

# Dense Rank Estimation Operator

Traditionally, t-SVT is adopted to truncate and restrict the rank matrix by the soft thresholding operator. However, controlling the low-rank degree of the generated tensor $\chi$ by simply adjusting the rank $r$ and the threshold value $\gamma$ is coarse, which does not well exploit the data adaptability in DNNs. Inspired by the threshold contraction and to find the most suitable tensor tubal rank, we inject a DNN-induced dense rank estimation operator $\rho ( \cdot )$ into the optimal solution of the factor $s$ , which captures and enriches the rank information. Specifically, $\rho ( \cdot )$ can be viewed as a rank information extractor. In this paper, we use the fully connected layer for experiments. Given the rank matrix $\mathbf { S }$ as input, the dense rank estimation operator can be formulated as:

$$
\rho ( \mathbf { S } ) = L R e L U \cdots \left( L R e L U ( \mathbf { S } \times _ { 3 } \mathbf { G } _ { 1 } ) \times _ { 3 } \cdot \cdot \cdot \right) \times _ { 3 } \mathbf { G } _ { k } ,
$$

where $k$ denotes the number of the layers, $L R e L U$ is the LeakyReLU (He et al. 2015) function and each $\mathbf { G }$ is the learnable rank feature matrix. The optimization model is reformulated as

$$
\operatorname* { m i n } _ { \mathcal { U } , \mathcal { V } , \mathbf { S } , \mathbf { W } } \phi ( \mathbf { H } ( L ^ { - 1 } ( L ( \mathcal { U } ) \triangle D i a g ( \rho ( \mathbf { S } ) ) \triangle L ( \mathcal { V } ) ^ { \mathbf { T } } ) ) , \mathbf { H } ( \mathcal { V } ) ) .
$$

# Orthogonal Total Variation

To improve the generative capability and get a more suitable transform, we adopt the Orthogonal Total Variation (OTV)

constraint for $\Theta = \{ L ( \mathcal { U } ) , L ( \mathcal { V } ) , \theta _ { L ^ { - 1 } } \}$ , where $\theta _ { L ^ { - 1 } }$ denotes the weight matrix of the $\dot { L } ^ { - 1 } ( \cdot )$ module, aiming to enhance the local smoothness prior structure, which is

$$
\mathrm { O T V } ( \Theta ) = \| \nabla _ { x } L ( \mathcal { U } ) \| _ { \ell _ { 1 } } + \| \nabla _ { y } L ( \mathcal { V } ) ^ { \mathbf { T } } \| _ { \ell _ { 1 } } + \| \nabla _ { x } \theta _ { L ^ { - 1 } } \| _ { \ell _ { 1 } } .
$$

As a result, the final optimization model is

$$
\begin{array} { r l } & { \underset { u , \mathcal { V } , \mathbf { S } , \mathbf { W } } { \operatorname* { m i n } } \phi ( \mathbf { H } ( L ^ { - 1 } ( L ( \mathcal { U } ) \triangle D i a g ( \rho ( \mathbf { S } ) ) \triangle L ( \mathcal { V } ) ^ { \mathbf { T } } ) ) , \mathbf { H } ( \mathcal { V } ) ) } \\ & { ~ + \lambda \mathrm { O T V } ( \Theta ) , } \end{array}
$$

where $\lambda$ is a trade-off parameter, and the complete algorithm is depicted in Algorithm 1.

# Algorithm 1: The Proposed OTLRM Algorithm.

Input: The coarse estimated rank $r$ , hyperparameter $\lambda$ , and the maximum iteration $t _ { m a x }$ .   
Output: The reconstructed tensor $\mathcal { X } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ .   
1: Initialization: The iteration $t = 0$ .   
2: while $t < t _ { m a x }$ do   
3: Compute $\mathbf { L }$ via Eq. (5);   
4: Compute $\hat { \mathcal { U } }$ via $\hat { \mathcal { U } } = L ( \mathcal { U } )$ ;   
5: Compute $\hat { \mathcal { V } }$ via $\hat { \mathcal { V } } = L ( \mathcal { V } )$ ;   
6: Compute $\hat { \boldsymbol { S } }$ via $\hat { S } = D i a g ( \rho ( \mathbf { S } ) )$ ;   
7: Compute the loss via Eq. (18);   
8: Perform gradient backpropagation;   
9: end while   
10: Get the final low-rank tensor $\mathcal { X } = L ^ { - 1 } ( \hat { \mathcal { U } } \triangle \hat { \mathcal { S } } \triangle \hat { \mathcal { V } } ^ { \mathbf { T } } )$ .

# Applications for The Proposed Model Tensor Completion

Given the unknown pure tensor $\boldsymbol { \mathcal { V } } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ , tensor completion aims to recover underlying data from the observed entries $\Omega = \{ ( i _ { 1 } , i _ { 2 } , i _ { 3 } ) | \zeta _ { i _ { 1 } , i _ { 2 } , i _ { 3 } } = 1 \}$ , which follows the Bernoulli sampling scheme $\bar { \Omega _ { \mathbf { \lambda } } } \sim \mathbf { B } \mathbf { \bar { e } } \mathbf { r } ( \mathbf { \zeta } _ { p } )$ . And $p$ is the probability of taking target Bernoulli variables with independent and identically distributed. Based on the above definition, $\mathbf { H } ( \cdot )$ can be specified as $\mathbf { H } _ { \Omega } ( \cdot ) : \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } } $ $\mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ , which keeps the entries in $\Omega$ fixed and sets the rest zero. The loss function of $\phi ( \cdot )$ can be formulated as

$$
\phi ( \boldsymbol { \mathcal { X } } , \boldsymbol { \mathcal { Y } } ) = \| \mathbf { H } _ { \Omega } ( \boldsymbol { \mathcal { X } } ) - \mathbf { H } _ { \Omega } ( \boldsymbol { \mathcal { Y } } ) \| _ { F } ^ { 2 } .
$$

# MSI Reconstruction in CASSI System

Coded aperture snapshot compressive imaging (CASSI), which aims at scanning scenes with spatial and spectral dimensions, has achieved impressive performance. With a coded aperture and a disperser, it encodes and shifts each band of the original scene $\mathcal { V } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ with known mask $\mathbf { M } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } }$ and later blends all the bands to generate a 2-D measurement $\mathbf { X } \in \mathbb { R } ^ { n _ { 1 } \times ( n _ { 2 } + d \times ( n _ { 3 } - 1 ) ) }$ , where $d$ is the shift step. Considering the measurement noise $\mathbf { N } \in \mathbb { R } ^ { n _ { 1 } \times ( n _ { 2 } + d \times ( n _ { 3 } - 1 ) ) }$ generated in the coding system, the whole process can be formulated as

$$
\mathbf { X } = \sum _ { k = 1 } ^ { n _ { 3 } } s h i f t ( \mathcal { V } ( : , : , k ) \odot \mathbf { M } ) + \mathbf { N } .
$$

For convenience, by the definition of the $\mathbf { H } ( \cdot )$ , the above operations can be simplified to the following formula:

$$
\begin{array} { r } { \mathbf { X } = \mathbf { H } ( \mathcal { Y } ) + \mathbf { N } , } \end{array}
$$

where for MSI reconstruction, the $\mathbf { H } ( \cdot )$ operator is defined as $\mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }  \mathbb { R } ^ { n _ { 1 } \times ( n _ { 2 } + d \times ( n _ { 3 } - 1 ) ) }$ in CASSI application. And the loss function of $\phi ( \cdot )$ can be formulated as

$$
\phi ( \mathbf { X } , \mathcal { Y } ) = \| \mathbf { X } - \mathbf { H } ( \mathcal { Y } ) \| _ { F } ^ { 2 } .
$$

# MSI Denoising

The purpose of MSI denoising is to recover clean MSIs from noise-contaminated observations. In that case, the $\mathbf { H } ( \cdot )$ operator can be defined as a noise-adding operation. The loss function of $\phi ( \cdot )$ can be formulated as

$$
\phi ( \mathcal { X } , \mathcal { Y } ) = \| \mathcal { X } - \mathbf { H } ( \mathcal { Y } ) \| _ { \ell _ { 1 } } .
$$

# Computational Complexity Analysis

Suppose that the target tensor $\mathcal { X } \ \in \ \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ and the orthogonal transform $\textbf { L } \in \mathbb { R } ^ { n _ { 3 } \times n _ { 3 } }$ . For the process of the transform construction, the computational complexity is $O ( ( n _ { 3 } ) ^ { 4 } )$ . For the transformed tensors $\hat { \mathcal { U } } , \hat { \mathcal { V } }$ , the computational complexity is $O ( n _ { 1 } r ( n _ { 3 } ) ^ { 2 } ) + O ( r n _ { 2 } ( n _ { 3 } ) ^ { 2 } )$ . For the $\hat { \boldsymbol { s } }$ , given the number of DNN layers $k$ , the computational complexity is $O ( n _ { 3 } r ^ { 2 } k )$ . And for the product of the $\hat { \mathcal { U } } , \hat { \mathcal { V } } , \hat { \mathcal { S } }$ , the computational complexity is $O ( n _ { 3 } ( n _ { 1 } r ^ { 2 } n _ { 2 } ) )$ . Thus, the computational complexity of our method is ${ \overset { \underset { \mathrm { ~ \tiny ~ \hat { O } ~ } } { } } { O } } ( ( n _ { 3 } ) ^ { 4 } ) ~ +$ $O ( \bar { n _ { 1 } } r ( n _ { 3 } ) ^ { 2 } ) + O ( \bar { r n _ { 2 } } ( n _ { 3 } ) ^ { 2 } ) + O ( n _ { 3 } r ^ { 2 } k ) + O ( n _ { 3 } ( \bar { n _ { 1 } } r ^ { 2 } n _ { 2 } ) )$ . Due to the fact that $r < < \operatorname* { m i n } \{ n _ { 1 } , n _ { 2 } \}$ , it can be simplified as $O ( ( n _ { 3 } ) ^ { 4 } + ( n _ { 1 } + n _ { 2 } ) r ( n _ { 3 } ) ^ { 2 } + n _ { 1 } \dot { n _ { 2 } } n _ { 3 } r ^ { 2 } )$ .

# Experiments

# Datasets and Settings

In this section, we introduce the datasets (videos, MSIs and MRI data) and the experiment settings used in three multidimensional inverse problems. Please refer to the supplementary material for extended experiments.

Overall Settings: All the experiments are implemented in PyTorch and conducted on one NVIDIA GeForce RTX 3090 GPU with 20GB RAM. Adam (Kingma and Ba 2015) is used to optimize Eq. (16) and Eq. (18). For the rank estimation module $\rho ( \cdot )$ , we used a two-layer DNN (composed of two linear transforms and a LeakyReLU (He et al. 2015) function) in all experiments (the $k$ in Eq. (15) is 2). For the initialization of the learnable parameters in the model, we follow the typical kaiming initialization (He et al. 2015) and please refer to the supplementary material for the initialization ablation experiments. It is noteworthy that to enhance the model’s data adaptability, we employ the endogenously orthogonal transforms $\mathbf { L } _ { 1 }$ and $\mathbf { L } _ { 2 }$ to operate on $\mathcal { U }$ and $\nu$ , respectively, while $\mathbf { L } _ { 3 }$ serves as the inverse.

Tensor Completion: The evaluation encompasses four MSIs sourced from the CAVE database 1, namely Balloon, Beer, Pompom and Toy, along with two videos obtained from the NTT database 2, labeled Bird and Horse. Each MSI is resized to dimensions of $2 5 6 \times 2 5 6 \times 3 1$ , while the videos have a spatial size of $2 8 8 \times 3 5 2$ . In our experiments, we utilize the initial 30 frames of each video. Regarding the MSI datasets, we employ sampling rates (SRs) of 0.05, 0.10, and 0.15, while for the video datasets, the sampling rates are set to 0.10, 0.15, and 0.20.

MSI Reconstruction in CASSI System: We select five datasets (scene01-scene05, $2 5 6 \times 2 5 6 \times 3 1 \times$ ) in KAIST 3 (Choi et al. 2017) for simulation. The shift step $d$ is 2, which means that the measurement is of size $2 5 6 \times 3 1 0$ .

MSI Denoising: We select three scenes (scene01, scene02 and scene10) in KAIST (Choi et al. 2017) database (all of size $2 5 6 \times 2 5 6 \times 3 1 \times$ for testing. And the noise cases are set for Gaussian noise with standard deviations 0.2 and 0.3.

# Comparisons with State-of-the-Arts

In this section. we compare our method with state-of-thearts. The best and second-best are highlighted in bold and underlined, respectively. OTLRM\* represents the model without OTV loss, and OTLRM denotes the whole model.

Evaluation Metrics: For numerical comparison, we use peak signal-to-noise ratio (PSNR) and structural similarity (SSIM) as metrics in all three problems. And feature similarity index measure (FSIM) is added for MSI denoising problem. The higher PSNR, SSIM, and FSIM the better.

Tensor Completion: Especially, we compare with other state-of-the-art methods, including TNN (Lu et al. 2019), TQRTNN (Wu et al. 2021), UTNN (Song, $\mathrm { N g }$ , and Zhang 2020), DTNN (Jiang et al. 2023), LS2T2NN (Liu et al. 2023) and HLRTF (Luo et al. 2022).

Table 1 and 2 shows the numerical results with 1 to 4 dB improvement in PSNR, especially in low sampling rate. Compared to the traditional TNN-based methods, our model demonstrates excellent results of learnable orthogonal transforms and the generative t-SVD model, which maintains flexible data-adaption low-rank property. More results are provided in the supplementary material.

MSI Reconstruction in CASSI System: We compare our method with DeSCI(Liu et al. 2018), $\lambda$ -Net(Miao et al. 2019), TSA-Net(Meng, Ma, and Yuan 2020), HDNet(Hu et al. 2022), DGSMP(Huang et al. 2021), ADMM-Net(Ma et al. 2019), GAP-Net(Meng, Yuan, and Jalali 2023), PnPCASSI (Zheng et al. 2021), DIP-HSI(Meng et al. 2021) and HLRTF(Luo et al. 2022).

Table 3 exhibits clear advantages over self-supervised and zero-shot methods, even having a slight increase of at least 0.34 dB on average over supervised methods. More results are presented in the supplementary material.

MSI Denoising: For MSI denoising, we select: NonLRMA(Chen et al. 2017), TLRLSSTV(Zeng et al. 2020), LLxRGTV(Zeng and Xie 2021), 3DTNN(Zheng et al. 2019), LRTDCTV(Zeng et al. 2023), E3DTV(Peng et al. 2020), DDRM(Kawar et al. 2022), DDS2M(Miao et al. 2023) and HLRTF (Luo et al. 2022).

Table 1: Evaluation on CAVE of tensor completion by different methods for MSIs under different SRs. Top Left:Balloons, Bottom Left:Beer; Top Right:Pompom, Bottom Right:Toy.   

<html><body><table><tr><td rowspan="2">Method</td><td rowspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2">ReferencePSNRIM↑ PSNRSSM↑ PSNRSSITime(S)PSNRSIM↑ PSNRIM PSRIM T()</td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td rowspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Observed</td><td>None</td><td>13.53</td><td>0.12</td><td>13.76</td><td>0.15</td><td>14.01 0.18</td><td>None</td><td>11.66</td><td></td><td>0.05 11.89</td><td>0.08</td><td>12.14</td><td>0.11</td><td>None</td></tr><tr><td>TNN(Lu et al. 2019)</td><td>TPAMI</td><td>31.69</td><td>0.87</td><td>36.27</td><td>0.94</td><td>39.66 0.97</td><td>12</td><td>23.34</td><td>0.56</td><td>28.39</td><td>0.76</td><td>31.74</td><td>0.85</td><td>13</td></tr><tr><td>TQRTNN (Wu et al. 2021)</td><td>TCI</td><td>31.68</td><td>0.87</td><td>36.26</td><td>0.94</td><td>39.66 0.97</td><td>27</td><td>23.33</td><td>0.56</td><td>28.39</td><td>0.76</td><td>31.70</td><td>0.85</td><td>33</td></tr><tr><td>UTNN(Song,Ng,and Zhang 2020)</td><td>NLAA</td><td>33.78</td><td>0.92</td><td>39.28</td><td>0.97</td><td>43.20 0.99</td><td>234</td><td>25.32</td><td>0.66</td><td>31.34</td><td>0.86</td><td>35.62</td><td>0.93</td><td>150</td></tr><tr><td>DTNN(Jiang et al.2023)</td><td>TNNLS</td><td>35.44</td><td>0.94</td><td>40.51</td><td>0.98</td><td>44.22 0.99</td><td>417</td><td>28.05</td><td>0.78</td><td>32.48</td><td>0.89</td><td>36.17</td><td>0.94</td><td>409</td></tr><tr><td>LS2T2NN (Liu et al. 2023)</td><td>TCSVT</td><td>38.35</td><td>0.97</td><td>42.99</td><td>0.99</td><td>45.58 0.99</td><td>54</td><td>31.75</td><td>0.87</td><td>36.57</td><td>0.95</td><td>39.23</td><td>0.97</td><td>220</td></tr><tr><td>HLRTF(Luo et al.2022)</td><td>CVPR</td><td>37.25</td><td>0.96</td><td>42.43</td><td>0.98</td><td>45.65 0.99</td><td>22</td><td>30.04</td><td>0.82</td><td>37.65</td><td>0.96</td><td>40.92</td><td>0.98</td><td>21</td></tr><tr><td>OTLRM*</td><td>Ours</td><td>37.64</td><td>0.95</td><td>42.68</td><td>0.98</td><td>46.03 0.99</td><td>117</td><td>28.99</td><td>0.74</td><td>35.12</td><td>0.91</td><td>39.44</td><td>0.96</td><td>108</td></tr><tr><td>OTLRM</td><td>Ours</td><td>40.20</td><td>0.98</td><td>44.34</td><td>0.99</td><td>46.92 0.99</td><td>126</td><td>35.42</td><td>0.93</td><td>39.41</td><td>0.97</td><td>42.20</td><td>0.98</td><td>125</td></tr><tr><td>Observed</td><td>None</td><td>9.65</td><td>0.02</td><td>9.89</td><td>0.03</td><td>10.13 0.04</td><td>None</td><td>11.17</td><td>0.25</td><td>11.41</td><td>0.29</td><td>11.66</td><td>0.32</td><td>None</td></tr><tr><td>TNN (Lu et al. 2019)</td><td>TPAMI</td><td>32.41</td><td>0.88</td><td>37.81</td><td>0.96</td><td>41.35 0.98</td><td>13</td><td>27.38</td><td>0.82</td><td>31.45</td><td>0.90</td><td>34.42</td><td>0.94</td><td>13</td></tr><tr><td>TQRTNN(Wu et al. 2021)</td><td>TCI</td><td>32.46</td><td>0.88</td><td>37.57</td><td>0.95</td><td>41.28 0.98</td><td>33</td><td>27.12</td><td>0.81</td><td>31.43</td><td>0.90</td><td>34.37</td><td>0.94</td><td>33</td></tr><tr><td>UTNN(Song,Ng,and Zhang 2020)</td><td>NLAA</td><td>35.62</td><td>0.95</td><td>41.04</td><td>0.98</td><td>44.66 0.99</td><td>171</td><td>29.40</td><td>0.87</td><td>35.00</td><td>0.95</td><td>39.05</td><td>0.98</td><td>190</td></tr><tr><td>DTNN(Jiang et al.2023)</td><td>TNNLS</td><td>35.95</td><td>0.95</td><td>41.29</td><td>0.97</td><td>45.05 0.99</td><td>415</td><td>30.34</td><td>0.90</td><td>35.38</td><td>0.96</td><td>39.33</td><td>0.97</td><td>411</td></tr><tr><td>LS2T2NN (Liu et al. 2023)</td><td>TCSVT</td><td>38.67</td><td>0.96</td><td>43.36</td><td>0.98</td><td>46.25 0.98</td><td>87</td><td>32.12</td><td>0.91</td><td>36.56</td><td>0.96</td><td>40.23</td><td>0.98</td><td>142</td></tr><tr><td>HLRTF(Luo et al. 2022)</td><td>CVPR</td><td>37.20</td><td>0.95</td><td>42.76</td><td>0.98</td><td>45.73 0.99</td><td>21</td><td>33.00</td><td>0.92</td><td>39.01</td><td>0.97</td><td>42.95</td><td>0.99</td><td>21</td></tr><tr><td>OTLRM*</td><td>Ours</td><td>39.16</td><td>0.97</td><td>43.53</td><td>0.99</td><td>46.92 0.99</td><td>111</td><td>32.63</td><td>0.91</td><td>38.41</td><td>0.97</td><td>42.46</td><td>0.98</td><td>113</td></tr><tr><td>OTLRM</td><td>Ours</td><td>41.74</td><td>0.98</td><td>45.09</td><td>0.99</td><td>47.52 0.99</td><td>118</td><td>36.24</td><td>0.97</td><td>41.11</td><td>0.99</td><td>44.14</td><td>0.99</td><td>128</td></tr></table></body></html>

Table 2: Evaluation on NTT of tensor completion by different methods for videos under different SRs. Left:Bird, Right: Horse   

<html><body><table><tr><td rowspan="2">Method</td><td rowspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td rowspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2">Referece</td><td rowspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Observed</td><td>None</td><td>7.74</td><td>0.02</td><td>7.98</td><td>0.03</td><td>8.25</td><td>0.04</td><td>None</td><td>6.53</td><td>0.01</td><td>6.78</td><td>0.02</td><td>7.04</td><td>0.02</td><td>None</td></tr><tr><td>TNN(Lu et al. 2019)</td><td>TPAMI</td><td>27.22</td><td>0.76</td><td>29.28</td><td>0.83</td><td>31.08</td><td>0.87</td><td>17</td><td>26.81</td><td>0.66</td><td>28.42</td><td>0.73</td><td>29.79</td><td>0.79</td><td>17</td></tr><tr><td>TQRTNN(Wu et al. 2021)</td><td>TCI</td><td>27.41</td><td>0.77</td><td>29.55</td><td>0.83</td><td>31.33</td><td>0.88</td><td>58</td><td>26.93</td><td>0.67</td><td>28.64</td><td>0.74</td><td>29.95</td><td>0.79</td><td>51</td></tr><tr><td>UTNN(Song,Ng,and Zhang 2020)</td><td>NLAA</td><td>27.83</td><td>0.79</td><td>30.11</td><td>0.85</td><td>31.95</td><td>0.89</td><td>156</td><td>27.27</td><td>0.68</td><td>29.00</td><td>0.76</td><td>30.30</td><td>0.81</td><td>146</td></tr><tr><td>DTNN(Jiang etal.2023)</td><td>TNNLS</td><td>28.55</td><td>0.83</td><td>30.60</td><td>0.88</td><td>32.51</td><td>0.91</td><td>503</td><td>27.52</td><td>0.74</td><td>29.42</td><td>0.80</td><td>30.98</td><td>0.85</td><td>580</td></tr><tr><td>LS2T2NN (Liu et al. 2023)</td><td>TCSVT</td><td>30.40</td><td>0.85</td><td>33.38</td><td>0.91</td><td>35.55</td><td>0.94</td><td>63</td><td>29.48</td><td>0.76</td><td>31.44</td><td>0.83</td><td>33.45</td><td>0.88</td><td>83</td></tr><tr><td>HLRTF (Luo et al. 2022)</td><td>CVPR</td><td>26.25</td><td>0.70</td><td>28.50</td><td>0.78</td><td>31.27</td><td>0.86</td><td>22</td><td>25.73</td><td>0.60</td><td>26.37</td><td>0.63</td><td>28.61</td><td>0.76</td><td>23</td></tr><tr><td>OTLRM</td><td>Ours</td><td>34.24</td><td>0.93</td><td>36.04</td><td>0.95</td><td>37.44</td><td>0.96</td><td>124</td><td>30.64</td><td>0.83</td><td>32.20</td><td>0.87</td><td>33.71</td><td>0.90</td><td>125</td></tr></table></body></html>

Table 4 shows the performance with three metrics. More results are provided in the supplementary material. Compared with model-based methods, our OTLRM can deeply capture the low-rankness and help improve the abilities of MSI denoising within DNN. For diffusion-based methods and deep prior-induced $\mathrm { P n P }$ methods, our self-supervised method can also enhance the performance and achieve comparable results. Compared with the tensor-based method HLRTF, our method is more informative and smoother.

# Ablation Analysis

In this section, we do ablation experiments on the hyperparameter tensor rank $r$ , the trade-off parameter $\lambda$ , the number $k$ of the DNN layers in the rank estimation module $\rho ( \cdot )$ and the visual results of the weight matrices in $\rho ( \cdot )$ respectively to demonstrate the advantages of our model. Additionally, in traditional skinny t-SVD algorithm , $\mathcal { U }$ and $\nu$ are semi-orthogonal tensors. However, our model does not introduce semi-orthogonal constraints on these tensors. We explain this in the below ablation analysis. What’s more, for the advantages of the orthogonal transform L, we also compare our OTLRM with two typical methods, DTNN (Kong, Lu, and Lin 2021) and Tensor $Q$ -rank (Jiang et al. 2023). And more ablation analysis and experiments are provided in the supplementary material. MSI Balloons (of size $2 5 6 \times 2 5 6 \times 3 1 \times$ in CAVE datasets is selected for TC.

Effect of $r$ : The tensor rank $r \in [ 1 , \operatorname* { m i n } \{ n _ { 1 } , n _ { 2 } \} ]$ which is an integer characterizes the low-rankness of the generated tensor $\bar { \boldsymbol { \mathcal { X } } } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ . In the experiment, we choose the rank $r$ ranging from 10 to 100 at intervals of 10. From Figure 12 in the supplementary material, when the rank $r$ is low, the performance is less effective due to the lost information of the original tensor. When the rank is too high, the result tensor is not well guaranteed to be low-rank, which leads to a sub-optimal performance. Thus, given $n = \operatorname* { m i n } \{ n _ { 1 } , n _ { 2 } \}$ , the rank $r$ can be set between $[ n / 2 \bar { 0 } , n / 5 ]$ .

Effect of $\lambda { \mathrm { : } }$ : The hyperparameter $\lambda$ which controls the OTV loss is set using a strategy of sampling one point every order of magnitude from 1e-1 to 1e-10. From Figure 13 in the supplementary material, we can find that when $\lambda$ is large, the performance is decreased, probably because too strong the OTV loss aggravates the local similarities and loses the detailed features. Our method remains effective when $\lambda$ falls between 1e-7 and 1e-9. Thus, our model is easy to tune for the best $\lambda$ . And to highlight our novelty in low-rank, more results are available in the supplementary material.

Effect of $k$ : In all experiments, we simply use a two-layer DNN to represent the rank estimation module $\rho ( \cdot )$ . And we analyze the effect of $\rho ( \cdot )$ on the results by adjusting the number $k$ of layers of the DNN. Here, $k = 1$ indicates that only one LeakyReLU activation function is added. From Table 5, deeper DNN may have the potential for better results due to their better fitting ability.

Effect of $\rho ( \cdot )$ : For the rank information extractor $\rho ( \cdot )$ , we adopted a two-layer deep neural network that contains one LeakyReLU layer sandwiched between two linear layers as a simple implementation. Visual results of the two weight matrices are shown in Figure 15 in the supplementary material. It can be seen that the weight matrices learn different and rich rank information for different dimensions and different positions of the rank matrix S with low-rank property. Analysis for The Semi-orthogonality of $\mathcal { U }$ and $\nu$ : The semi-orthogonality of the $\mathcal { U }$ and $\nu$ is naturally and strictly guaranteed by the process of the SVD. However, our generative t-SVD model is solved based on the gradient descentbased algorithm, which means that we do not need the SVD. To verify the validity of the semi-orthogonality of $\mathcal { U }$ and $\nu$ , we construct a semi-orthogonal loss term for ablation experiments, which is:

Table 3: Evaluation on KAIST of MSI Reconstruction in CASSI. $*$ for supervised methods, ◦ for self-supervised methods   

<html><body><table><tr><td>Method</td><td>Category</td><td></td><td colspan="2">scene01 Reference|PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑</td><td colspan="2">scene02</td><td colspan="2">scene03</td><td colspan="2">scene04</td><td colspan="2"></td><td colspan="2">scene05</td><td colspan="2">Avg</td></tr><tr><td>DeSCI(Liu et al. 2018)</td><td>Model</td><td>TPAMI</td><td>28.38</td><td>0.80</td><td></td><td>26.00</td><td>0.70</td><td>23.11</td><td>0.73</td><td>28.26</td><td>0.86</td><td>25.41</td><td></td><td>0.78</td><td>26.23</td><td>0.77</td></tr><tr><td>X-Net(Miao et al.2019)</td><td>CNN (*)</td><td>ICCV</td><td>30.10</td><td>0.85</td><td></td><td>28.49</td><td>0.81</td><td>27.73</td><td>0.87</td><td>37.01</td><td>0.93</td><td></td><td>26.19</td><td>0.82</td><td>29.90</td><td>0.86</td></tr><tr><td>TSA-Net(Meng,Ma,and Yuan 2020)</td><td>CNN (*)</td><td>ECCV</td><td>32.31</td><td>0.89</td><td></td><td>31.03</td><td>0.86</td><td>32.15</td><td>0.92</td><td>37.95</td><td>0.96</td><td></td><td>29.47</td><td>0.88</td><td>32.58</td><td>0.90</td></tr><tr><td>HDNet(Hu et al. 2022)</td><td>Transformer (*)</td><td>CVPR</td><td>34.96</td><td>0.94</td><td></td><td>35.64</td><td>0.94</td><td>35.55</td><td>0.94</td><td>41.64</td><td>0.98</td><td></td><td>32.56</td><td>0.95</td><td>36.07</td><td>0.95</td></tr><tr><td>DGSMP(Huang et al. 2021)</td><td>Unfolding (*)</td><td>CVPR</td><td>33.26</td><td>0.92</td><td></td><td>32.09</td><td>0.90</td><td>33.06</td><td>0.93</td><td>40.54</td><td>0.96</td><td></td><td>28.86</td><td>0.88</td><td>33.56</td><td>0.92</td></tr><tr><td>ADMM-Net(Ma et al. 2019)</td><td>Unfolding (*)</td><td>ICCV</td><td>34.03</td><td>0.92</td><td></td><td>33.57</td><td>0.90</td><td>34.82</td><td>0.93</td><td>39.46</td><td>0.97</td><td></td><td>31.83</td><td>0.92</td><td>34.74</td><td>0.93</td></tr><tr><td>GAP-Net(Meng, Yuan, and Jalali 2023)</td><td>Unfolding (*)</td><td>IJCV</td><td>33.63</td><td>0.91</td><td></td><td>33.19</td><td>0.90</td><td>33.96</td><td>0.93</td><td>39.14</td><td>0.97</td><td></td><td>31.44</td><td>0.92</td><td>34.27</td><td>0.93</td></tr><tr><td>PnP-CASSI(Zheng et al. 2021)</td><td>PnP (Zero-Shot)</td><td>PR</td><td>29.09</td><td>0.80</td><td></td><td>28.05</td><td>0.71</td><td>30.15</td><td>0.85</td><td>39.17</td><td>0.94</td><td></td><td>27.45</td><td>0.80</td><td>30.78</td><td>0.82</td></tr><tr><td>DIP-HSI(Meng et al.2021)</td><td>PnP (Zero-Shot)</td><td>ICCV</td><td>31.32</td><td>0.86</td><td></td><td>25.89</td><td>0.70</td><td>29.91</td><td>0.84</td><td>38.69</td><td>0.93</td><td></td><td>27.45</td><td>0.80</td><td>30.65</td><td>0.82</td></tr><tr><td>HLRTF(Luo et al. 2022)</td><td>Tensor (o)</td><td>CVPR</td><td>34.56</td><td>0.91</td><td></td><td>33.37</td><td>0.87</td><td>35.55</td><td>0.94</td><td>43.56</td><td>0.98</td><td></td><td>33.08</td><td>0.93</td><td>36.02</td><td>0.93</td></tr><tr><td>OTLRM</td><td>Tensor (o)</td><td>Ours</td><td>35.03</td><td>0.91</td><td></td><td>32.90</td><td>0.82</td><td>36.25</td><td>0.95</td><td>44.75</td><td>0.98</td><td>33.13</td><td></td><td>0.92</td><td>36.41</td><td>0.92</td></tr></table></body></html>

<html><body><table><tr><td>Method</td><td colspan="5">Reference PSNR ↑SSIM ↑FSIM↑Time (s)</td><td>Method</td><td colspan="5">ReferencePSNR↑SSIM↑FSIM↑Time (s)</td></tr><tr><td>Noisy</td><td>None</td><td>16.18</td><td>0.12</td><td>0.401</td><td>None</td><td>Noisy</td><td>None</td><td>12.98</td><td>0.06</td><td>0.320</td><td>None</td></tr><tr><td>NonLRMA (Chen et al. 2017)</td><td>TGRS</td><td>21.26</td><td>0.41</td><td>0.803</td><td>1</td><td>NonLRMA (Chen et al. 2017)</td><td>TGRS</td><td>20.30</td><td>0.36</td><td>0.772</td><td>11</td></tr><tr><td>TLRLSSTV_(Zeng et al.2020)</td><td>TGRS</td><td>24.88</td><td>0.53</td><td>0.767</td><td>76</td><td>TLRLSSTV (Zeng et al. 2020)</td><td>TGRS</td><td>22.82</td><td>0.41</td><td>0.689</td><td>76</td></tr><tr><td>LLxRGTV (Zeng and Xie 2021)</td><td>SP</td><td>31.15</td><td>0.80</td><td>0.917</td><td>38</td><td>LLxRGTV (Zeng and Xie 2021)</td><td>SP</td><td>27.64</td><td>0.68</td><td>0.868</td><td>38</td></tr><tr><td>3DTNN (Zheng et al.2019)</td><td>TGRS</td><td>28.04</td><td>0.78</td><td>0.881</td><td>20</td><td>3DTNN (Zheng et al. 2019)</td><td>TGRS</td><td>26.04</td><td>0.72</td><td>0.848</td><td>20</td></tr><tr><td>LRTDCTV (Zeng et al. 2023)</td><td>JSTAR</td><td>25.95</td><td>0.66</td><td>0.816</td><td>43</td><td>LRTDCTV (Zeng et al. 2023)</td><td>JSTAR</td><td>24.59</td><td>0.53</td><td>0.739</td><td>42</td></tr><tr><td>E3DTV (Peng et al. 2020)</td><td>TIP</td><td>30.34</td><td>0.87</td><td>0.926</td><td>10</td><td>E3DTV (Peng et al.2020)</td><td>TIP</td><td>28.36</td><td>0.82</td><td>0.900</td><td>9</td></tr><tr><td>DDRM (Kawar et al. 2022)</td><td>NeurIPS</td><td>29.41</td><td>0.87</td><td>0.922</td><td>20</td><td>DDRM (Kawar et al. 2022)</td><td>NeurIPS</td><td>27.81</td><td>0.79</td><td>0.893</td><td>23</td></tr><tr><td>DDS2M (Miao et al.2023)</td><td>ICCV</td><td>32.80</td><td>0.79</td><td>0.895</td><td>354</td><td>DDS2M (Miao et al.2023)</td><td>ICCV</td><td>30.08</td><td>0.67</td><td>0.834</td><td>318</td></tr><tr><td>HLRTF(Luo et al. 2022)</td><td>CVPR</td><td>30.42</td><td>0.83</td><td>0.940</td><td>23</td><td>HLRTF(Luo et al. 2022)</td><td>CVPR</td><td>30.34</td><td>0.69</td><td>0.874</td><td>25</td></tr><tr><td>OTLRM</td><td>Ours</td><td>30.82</td><td>0.72</td><td>0.881</td><td>45</td><td>OTLRM</td><td>Ours</td><td>32.56</td><td>0.82</td><td>0.902</td><td>47</td></tr></table></body></html>

Table 4: Evaluation on KAIST of MSI Denosing. Left: case: $\mathcal { N } ( 0 , 0 . 2 )$ -scene10, Right: case: $\mathcal { N } ( 0 , 0 . 3 )$ -scene01.

Table 5: Effect of $k$ in the rank estimation module $\rho ( \cdot )$ .   

<html><body><table><tr><td>k-layers</td><td colspan="2">SR=0.05 PSNR↑ SSIM↑</td><td colspan="2">SR=0.10 PSNR↑ SSIM↑</td><td colspan="2">SR=0.15</td></tr><tr><td>0</td><td>35.82</td><td>0.94</td><td></td><td>0.98</td><td>PSNR↑</td><td>SSIM↑ 0.98</td></tr><tr><td>1</td><td>38.72</td><td></td><td>40.39 43.10</td><td>0.99</td><td>41.92 45.12</td><td>0.99</td></tr><tr><td>2</td><td>40.20</td><td>0.97</td><td>44.34</td><td>0.99</td><td>46.92</td><td></td></tr><tr><td>3</td><td>40.15</td><td>0.98 0.98</td><td>44.56</td><td>0.99</td><td>46.85</td><td>0.99 0.99</td></tr></table></body></html>

$$
\Psi ( \mathcal { U } , \mathcal { V } ) = \beta ( \| \mathcal { U } ^ { \mathbf { T } } * _ { L } \mathcal { U } - \mathcal { T } \| _ { F } ^ { 2 } + \| \mathcal { V } ^ { \mathbf { T } } * _ { L } \mathcal { V } - \mathcal { T } \| _ { F } ^ { 2 } ) .
$$

Figure 14 in the supplementary material shows the completion results under different penalty coefficient $\beta$ . It can be observed that the results show a decreasing trend as $\beta$ continues to rise. Therefore, we relax the semi-orthogonal constraints on $\mathcal { U }$ and $\nu$ to enhance the performance.

Comparisons on Orthogonal Transform L: Table 3 in the supplementary material shows the evaluation results of the above methods. (1) The transform of DTNN is a redundant dictionary with constraints on each column of the dictionary to have an F-norm of one. It may not technically be called orthogonal, which makes its inverse transform difficult to calculate. To obtain the transform L ∈ Rn3×d, the computational complexity of DTNN is $O ( ( n _ { 3 } ) ^ { 3 } d ^ { 2 } )$ and ours is $\mathcal { \dot { O } } ( ( n _ { 3 } ) ^ { 4 } )$ (running time 415s vs 118s). (2) Tensor $Q$ -rank imposes constraints on the transform process with a “two-step” strategy. This involves first finding an appropriate transform based on certain selection criteria and then incorporating it into the reconstruction process. This “separate” strategy may not well capture the intrinsic characteristics of the data. In contrast, our learnable transform can be directly embedded and inherently achieves orthogonality.

# Conclusion

In this paper, we proposed a learnable orthogonal transforminduced generative low-rank framework based on t-SVD for multi-dimensional tensor recovery, possessing the orthogonal and learnable transform which enables flexible dataadaptive capability while maintains theoretical guarantee for “arbitrary invertible linear transform”. Constructed by a series of Householder transformation units, this transform can be learnable and seamlessly integrated into the neural network with endogenous orthogonality. Compared to traditional solutions of t-SVD, our generative t-SVD representation model can naturally maintain the low-rank structure and be solved by gradient descent-based algorithms. Comprehensive experiments verify the effectiveness of our method in MSIs and videos for three practical problems.

# Acknowledgments

This work was supported in part by the National Natural Science Foundation of China under Grants 62106063 and 62276079, by the Guangdong Natural Science Foundation under Grant 2022A1515010819, and by National Natural Science Foundation of China Joint Fund Project Key Support Project under U22B2049.