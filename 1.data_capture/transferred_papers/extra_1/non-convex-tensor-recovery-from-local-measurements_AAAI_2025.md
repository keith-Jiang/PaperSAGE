# Non-Convex Tensor Recovery from Local Measurements

Tongle $\mathbf { W } \mathbf { u } ^ { 1 }$ , Ying $\mathbf { S u n } ^ { 1 * }$ , Jicong Fan2\*

1School of Electrical Engineering and Computer Science, Pennsylvania State University 2School of Data Science, The Chinese University of Hong Kong, Shenzhen {tfw5381, ybs5190}@psu.edu, fanjicong@cuhk.edu.cn

# Abstract

Motivated by the settings where sensing the entire tensor is infeasible, this paper proposes a novel tensor compressed sensing model, where measurements are only obtained from sensing each lateral slice via mutually independent matrices. Leveraging the low tubal rank structure, we reparameterize the unknown tensor $x ^ { \star }$ using two compact tensor factors and formulate the recovery problem as a nonconvex minimization problem. To solve the problem, we first propose an alternating minimization algorithm, termed Alt-PGD-Min, that iteratively optimizes the two factors using a projected gradient descent and an exact minimization step, respectively. Despite nonconvexity, we prove that Alt-PGD-Min achieves $\epsilon$ -accuracy recovery with $\mathcal { O } \left( \kappa ^ { 2 } \log \frac { 1 } { \epsilon } \right)$ iteration complexity and $\mathcal { O }$ $\left( \kappa ^ { 6 } r n _ { 3 } \log n _ { 3 } \right.$ $\begin{array} { r } { \left( \kappa ^ { 2 } r \left( n _ { 1 } + \bar { n _ { 2 } } \right) + n _ { 1 } \log \frac { 1 } { \epsilon } \right) , } \end{array}$ sample complexity, where $\kappa$ denotes tensor condition number of $x ^ { \star }$ . To further accelerate the convergence, especially when the tensor is ill-conditioned with large $\kappa$ , we prove AltScalePGD-Min that preconditions the gradient update using an approximate Hessian that can be computed efficiently. We show that Alt-ScalePGD-Min achieves $\kappa$ independent iteration complexity $\mathcal { O } ( \log \frac { 1 } { \epsilon } )$ and improves the sample complexity to $\begin{array} { r } { \mathcal { O } \left( \kappa ^ { 4 } r n _ { 3 } \log n _ { 3 } \left( \kappa ^ { 4 } r ( n _ { 1 } + n _ { 2 } ) + n _ { 1 } \log \frac { 1 } { \epsilon } \right) \right. } \end{array}$ . Experiments validate the effectiveness of the proposed methods.

# Introduction

Motivated by the well-known compressed sensing and matrix sensing (Cande\`s, Romberg, and Tao 2006; Recht, Fazel, and Parrilo 2010) problems, tensor compressed sensing (TCS) has attracted increasing attention in recent years (Shi et al. 2013; Rauhut, Schneider, and Stojanac 2017; Tong et al. 2022b; Chen, Raskutti, and Yuan 2019). The goal of TCS is to recover a tensor $\pmb { \mathcal { X } } ^ { \star } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ from a few measurements $\pmb { y } \in \mathbb { R } ^ { m } ( m \ll n _ { 1 } n _ { 2 } n _ { 3 } )$ , where $\pmb { y } = \pmb { \mathcal { A } } ( \pmb { \mathcal { X } } ^ { \star } )$ and $\mathcal { A } : \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }  \mathbb { R } ^ { m }$ is a linear operator. Since this problem is ill-posed for arbitrary $x ^ { \star }$ , the success of TCS relies on the existence of the low dimensional intrinsic structure in the original high-order tensor $x ^ { \star }$ . Such a structure has been widely validated and utilized in many real-world applications such as dynamic MRI ( $\mathrm { \Delta Y u }$ et al. 2014; Gong and Zhang 2024), video compression (Li et al. 2022; Ma et al. 2019; Liu et al. 2023b), snapshot compressive imaging (Ma et al. 2019; Liu et al. 2023b), quantum computing (Ran et al. 2020; Kuzmin et al. 2024), image recovery (Fan et al. 2023), and collaborative filtering (Fan 2022).

In the literature, different tensor decompositions can induce different definitions of tensor rank (Kolda and Bader 2009), which are often more complex than matrix rank. Consequently, the non-uniqueness and complexity of the tensor rank make TCS a non-trivial extension of matrix sensing. Most works of TCS assume that the ground truth tensor $x ^ { \star }$ has a low Tucker rank (Han, Willett, and Zhang 2022; Luo and Zhang 2023; Ahmed, Raja, and Bajwa 2020; Mu et al. 2014) or a low tubal rank (Zhang et al. 2020b; Hou et al. 2021; Lu et al. 2018; Liu et al. 2023a), which are induced by Tucker decomposition (Tucker 1966) and tensor Singular Value Decomposition (t-SVD) (Kilmer and Martin 2011) respectively. In these works, the measurements are given by

$$
y _ { i } = \langle A _ { i } , { \pmb x } ^ { \star } \rangle , i \in [ m ] ,
$$

where $\mathbf { \mathcal { A } } _ { i } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ has i.i.d. zero-mean Gaussian entries and can sense entire $x ^ { \star }$ .

It should be pointed out that in many scenarios, it is difficult to sense the entire tensor $x ^ { \star }$ , preventing the application of the sensing model (1). For instance, due to memory or privacy limitations, large-scale tensor data may be partitioned into multiple smaller tensors stored in a distributed network (Moothedath and Vaswani 2024; Singh and Vaswani 2024; Wu and Sun 2024). Another example is when the tensor $x ^ { \star }$ , such as images or videos, is collected in an on-the-fly streaming setting (Srinivasa et al. 2019).

To address the challenge, we propose a novel tensor compressed sensing model where each measurement is generated from locally sensing a slice of $x ^ { \star }$ . Our model is detailed as follows.

Definition 1 (Local TCS) For each lateral slice $i \in [ n _ { 2 } ]$ , its $j$ -th local measurement $y _ { j i }$ is obtained by

$$
y _ { j i } = \langle \pmb { A } _ { i } ( : , j , : ) , \pmb { \mathcal { X } } ^ { \star } ( : , i , : ) \rangle , i \in [ n _ { 2 } ] , j \in [ m ] ,
$$

where the $\pmb { \mathcal { A } } _ { i } ( : , j , : ) \in \mathbb { R } ^ { n _ { 1 } \times n _ { 3 } }$ denotes the $j$ -th sensing matrix for $i$ -th lateral slice of $x ^ { \star }$ .

Take the dynamic video sensing mentioned before as an example. Under the local TCS model in (2), the entire video is modeled as $x ^ { \star }$ , with each lateral slice $\boldsymbol { x } ^ { \star } ( : , i , : )$ representing the video frame at $i$ -th timestamp (Li, Ye, and $\mathrm { X u }$ t2i0o1n7s; $\left\{ y _ { j i } \right\} _ { j = 1 } ^ { m }$ iasl o2b0ta1i7n;edWbayngmetasalu. 2n0g2t0h)e. $i$ -thhe forbasmerevoafvideo $x ^ { \star }$ . We aim to recover $\pmb { \mathcal { X } } ^ { \star } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ from the measurements {yj i,j=1 i}i=n2,j=m obtained via (2). Under this framework, fundamental questions to understand are:

Under what conditions can we provably recover $x ^ { \star }$ , and how to compute the solution efficiently?

This paper considers the problem above under the structural assumption that the ground truth $x ^ { \star }$ is of low tubal rank with $r \ll \operatorname* { m i n } \left\{ n _ { 1 } , n _ { 2 } , n _ { 3 } \right\}$ (see Definition 8). We focus on low tubal rank for two key reasons. First, it can be computed more efficiently by solving multiple SVDs in the Fourier domain compared to CP and Tucker ranks (Zhang et al. 2014). Second, the convolution operator in this model is particularly effective at capturing the “spatial-shifting” properties of data (Liu et al. 2019; Wu et al. 2022; Wu and Fan 2024). Our main contributions are summarized as follows.

• We introduce a novel local TCS model (2) for tensor compressed sensing with measurements obtained by lateral slice-wise sensing. Compared to the traditional TCS model (1), local sensing does not rely on the availability of the entire tensor, which greatly enlarges its applicable scenarios, such as real-time and distributed processing. • We formulate the recovery problem as a nonconvex minimization problem based on the low tubal rank tensor factorization for $x ^ { \star }$ . An alternating minimization algorithm, called Alt-PGD-Min, is proposed to solve the problem with efficient computations per iteration. We show that under suitable conditions on the sensing operator, with $\mathcal { O } \left( \kappa ^ { 6 } r n _ { 3 } \log n _ { 3 } \left( \kappa ^ { 2 } r \left( n _ { 1 } + n _ { 2 } \right) + n _ { 1 } \breve { \log } \frac { 1 } { \epsilon } \right) \right)$ ) samples Alt-PGD-Min computes a solution that is $\epsilon$ - close to $x ^ { \star }$ in $\mathcal { O } \left( \kappa ^ { 2 } \log \frac { 1 } { \epsilon } \right)$ iterations, where $\kappa$ is the tensor condition number of $x ^ { \star }$ . • To improve the dependency of both the sample and iteration complexity on $\kappa$ , we further proposed AltScalePGD-Min that preconditions the gradient step in Alt-PGD-Min using an approximation of the Hessian matrix that is cheap to compute. We show that by incorporating the preconditioner, Alt-ScalePGDMin iteration complexity $\mathcal { O } ( \log \frac { 1 } { \epsilon } )$ that is independent of $\kappa$ , and improves the sample complexity to $\begin{array} { r } { \mathcal { O } \left( \kappa ^ { 4 } r n _ { 3 } \log n _ { 3 } \left( \kappa ^ { 4 } \dot { r } ( n _ { 1 } + n _ { 2 } ) + n _ { 1 } \log \frac { 1 } { \epsilon } \right) \right) } \end{array}$ . • We validated the proposed local sensing model and algorithms on both synthetic and real-world data. Numerical results show that the proposed algorithms can achieve effective performance in the local TCS model (2).

# Related Work

Tensor Compressed Sensing (TCS). Canonical TCS problems (1) based on Tucker and t-SVD decompositions have been extensively investigated in recent years. Studies in (Shi et al. 2013; Rauhut, Schneider, and Stojanac 2017; Ahmed, Raja, and Bajwa 2020; Mu et al. 2014; Chen, Raskutti, and Yuan 2019; Han, Willett, and Zhang 2022; Luo and Zhang 2023) utilized convex or non-convex optimization methods to solve low Tucker rank based tensor CS. For

TCS with low tubal rank under the t-SVD framework, Lu et al. (2018) proposed a convex method that minimizes the tensor nuclear norm (TNN) with order optimal sample complexity. Zhang et al. (2020b) proposed a regularized TNN minimization method with provable robust recovery performance from noisy observations based on the defined tensor Restricted Isometry Property (RIP). Hou et al. (2021) proposed convex methods to solve one-bit TCS from binary observations and provided robust recovery guarantees. Liu et al. (2024) developed theoretical guarantees for the nonconvex gradient descent method, which deals with exact low tubal rank and overparameterized tensor factorizations for solving the model (1). Liu et al. (2023a) fused low-rankness and local-smoothness of real-world tensor data in TCS and obtained the provable enhanced recovery guarantee. However, all existing TCS studies focus on models where random measurement tensors have access to the entire ground truth tensors, rather than recovering the low-rank tensor through local measurements as proposed in our TCS model in (2).

CS from Local Measurements. Compared to the canonical CS model, the investigation of CS that recovers from local measurements as model (2) is less explored. Nayer and Vaswani (2022); Vaswani (2024); Srinivasa et al. (2019); Srinivasa, Kim, and Lee (2023); Lee et al. (2023) considered matrix sensing model that involves recovering a low-rank matrix from independent compressed measurements of each of its columns. Srinivasa et al. (2019); Srinivasa, Kim, and Lee (2023); Lee et al. (2023) proposed convex programming methods that minimize relevant mixed norms with provable guarantees, while Nayer and Vaswani (2022); Vaswani (2024) proposed efficient non-convex method and obtained improved iteration and sample complexities.

Nevertheless, to our knowledge, all existing studies on CS from local measurements focused solely on twodimensional matrices, rather than higher-order tensors that widely exist in science and engineering. Reshaping a tensor into a matrix format to apply matrix methods overlooks the interactions across all dimensions and destroys the inherent structures of data. Therefore, it is essential to study TCS from local measurements, as proposed in our model (2), which has not been addressed in the literature, despite its significant applications in areas like video compression for online or distributed settings (Wu and Sun 2024; Srinivasa et al. 2019). Our experimental results such as Figures 1 and 4 will demonstrate the superiority of tensor CS over matrix CS.

# Preliminaries

# Notations

We use letters $x , x , X , X$ to denote scalars, vectors, matrices, and tensors, respectively. Let $\{ a _ { n } , b _ { n } \} _ { n > 1 }$ be any two positive series. We write $a _ { n } \gtrsim b _ { n }$ (or $a _ { n } \ \stackrel { \textstyle = } { \sim } \ b _ { n } )$ if there exists a universal constant $c > 0$ such that $a _ { n } \ \geq \ c b _ { n }$ (or $a _ { n } \leq c b _ { n } ,$ ). The notations of $a _ { n } = \Omega ( b _ { n } )$ and $a _ { n } = \mathcal { O } \left( b _ { n } \right)$ share the same meaning with $a _ { n } \gtrsim b _ { n }$ and $a _ { n } \lesssim b _ { n }$ .

The $i$ -th horizontal, lateral, and frontal slice matrix of $x$ are denoted as $\pmb { \mathcal { X } } ( i , : , : ) , \pmb { \mathcal { X } } ( : , i , : )$ , and $\pmb { x } ( : , : , i )$ respectively. The $( i , j , k )$ -th element is denoted as $x _ { i j k }$ . For simplicity, we also use $\boldsymbol { X } ^ { ( i ) }$ to denote the $i$ -th frontal slice. $\mathbf { \bar { \mathcal { X } } } ( i ) \mathbf { \bar { \Psi } } \in \mathbb { R } ^ { n _ { 1 } \times 1 \times n _ { 3 } }$ denotes the tensor that only composed of the $i$ -th lateral slice of $x$ . The inner product of tensors is denoted as $\begin{array} { r } { \langle { \pmb x } , { \pmb y } \rangle = \sum _ { i j k } { \pmb x } _ { i j k } { \pmb y } _ { i j k } } \end{array}$ . The Frobenius norm of tensor is denoted as $\begin{array} { r } { \| \pmb { \mathcal { X } } \| _ { F } = \sqrt { \sum _ { i j k } \pmb { \mathcal { X } } _ { i j k } ^ { 2 } } . } \end{array}$ . We use fft $( \pmb { \mathcal { X } } , [ \mathbf { \Lambda } ] , 3 ) = \overline { { \pmb { \mathcal { X } } } } \in \mathbb { C } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ to denote performing DFT on all the tubes of $\pmb { \mathcal { X } } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ . The inverse FFT on $\overline { { \mathcal { X } } }$ can turn it back to the original tensor, i.e., $\pmb { \chi } _ { = \mathrm { i f f t } ( } \overline { { \pmb { x } } } , [ ] , 3 )$ .

# Definitions, Tensor Factorization and Tubal Rank

Unfold and Fold operators for a tensor are defined as

$$
\begin{array} { l } { { \mathrm { U n f o l d } \left( \pmb { \mathcal { X } } \right) : = \left[ \pmb { X } ^ { ( 1 ) } ; \dots ; \pmb { X } ^ { ( n _ { 3 } ) } \right] } } \\ { { \mathrm { F o l d } \left( \mathrm { U n f o l d } \left( \pmb { \mathcal { X } } \right) \right) : = \pmb { \mathcal { X } } . } } \end{array}
$$

Denote the block circulant matrix of $x$ as

$$
\operatorname { b c i r c } ( \pmb { \mathscr X } ) : = \left[ \begin{array} { c c c c } { \pmb X ^ { ( 1 ) } } & { \pmb X ^ { ( n _ { 3 } ) } } & { \cdots } & { \pmb X ^ { ( 2 ) } } \\ { \pmb X ^ { ( 2 ) } } & { \pmb X ^ { ( 1 ) } } & { \cdots } & { \pmb X ^ { ( 3 ) } } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { \pmb X ^ { ( n _ { 3 } ) } } & { \pmb X ^ { ( n _ { 3 } - 1 ) } } & { \cdots } & { \pmb X ^ { ( 1 ) } } \end{array} \right] .
$$

The above bcirc $( \pmb { \mathscr { X } } )$ can be block diagonalized as

$$
\left( { \cal F } _ { n _ { 3 } } \otimes { \cal I } _ { n _ { 1 } } \right) \cdot \mathrm { b c i r c } ( \pmb { \chi } ) \cdot \left( { \cal F } _ { n _ { 3 } } \otimes { \cal I } _ { n _ { 2 } } \right) = \overline { { \pmb X } } ,
$$

where ${ \pmb F } _ { n }$ denotes the $n$ -dimensional discrete Fourier transformation matrix, $\otimes$ denotes the Kronecker product and $\overline { { \boldsymbol { X } } }$ is defined as:

$$
\overline { { \boldsymbol X } } : = \mathrm { b d i a g } \left( \overline { { \boldsymbol X } } \right) : = \mathrm { d i a g } \left( \overline { { \boldsymbol X } } ^ { ( 1 ) } ; \dots ; \overline { { \boldsymbol X } } ^ { ( 3 ) } \right) .
$$

With the above definitions, we introduce the following arithmetic operations for tensors.

Definition 2 (Tensor-Tensor product (T-product )) (Kilmer and Martin 2011) The tensor product between tensors $\pmb { \chi } \in$ $\mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ and $\pmb { y } \in \mathbb { R } ^ { n _ { 2 } \times n _ { 4 } \times n _ { 3 } }$ is defined as:

$$
\begin{array} { r l } { \pmb { \mathcal { X } } * \pmb { \mathcal { Y } } = F o l d \big ( b i c r c ( \pmb { \mathcal { X } } ) U n f o l d ( \pmb { \mathcal { Y } } ) \big ) } & { { } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 4 } \times n _ { 3 } } . } \end{array}
$$

Definition 3 (Conjugate transpose) $\mathit { \Delta } ^ { \prime } L u$ et al. 2020) The conjugate transpose of $\pmb { \chi } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ is $\pmb { \mathcal { X } } ^ { c }$ that $x ^ { c } ($ : $, : , 1 ) = \Big ( \pmb { X } ^ { ( 1 ) } \Big ) ^ { c } , \pmb { \chi } ^ { c } ( : , : , n _ { 3 } + 2 - i ) = \Big ( \pmb { X } ^ { ( i ) } \Big ) ^ { c }$ for $2 \leq i \leq n _ { 3 }$ , where $X ^ { c }$ is conjugate transpose of $X$ .

Definition 4 (Identity tensor) (Kilmer and Martin 2011) If $\pmb { \mathcal { I } } ( : , : , 1 ) = \pmb { I } _ { n }$ and $\pmb { \mathcal { I } } ( : , : , i ) = \mathbf { 0 } _ { n }$ for $2 \leq i \leq n _ { 3 }$ , then $\pmb { \mathcal { T } } \in \mathbb { R } ^ { n \times n \times n _ { 3 } }$ is defined as the identity tensor.

Definition 5 (Orthogonal tensor) (Kilmer and Martin 2011) $A$ tensor $\mathcal { Q } \in \mathbb { R } ^ { n \times n \times n _ { 3 } }$ is defined as the orthogonal tensor i $f \mathcal { Q } * \mathcal { Q } ^ { c } = \mathcal { Q } ^ { c } * \mathcal { Q } = \mathcal { I }$ .

Definition 6 (Tensor inverse) (Kilmer and Martin $2 0 I I ) A n$ $n { \times } n { \times } n _ { 3 }$ tensor $x$ has an inverse $\pmb { y } _ { i f . } \pmb { x } _ { * } \pmb { y } = \pmb { \tau }$ and $^ { y _ { * } }$ $\scriptstyle x = \tau$ . If $x$ is invertible, we use $x ^ { - 1 }$ to denote its inverse.

Definition 7 ${ \bf \nabla } ^ { F }$ -diagonal tensor) (Kilmer and Martin 2011) If all of frontal slices of $x$ are diagonal matrices, then $x$ is called an $f$ -diagonal tensor.

Theorem 1 (t-SVD) (Lu et al. 2020) Let $\pmb { \chi } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ . Then it can be factorized as

$$
\pmb { x } = \pmb { \mathcal { U } } * \pmb { \mathcal { S } } * \pmb { \mathcal { V } } ^ { c } ,
$$

where $\boldsymbol { u } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 1 } \times n _ { 3 } }$ , $\pmb { \nu } \in \mathbb { R } ^ { n _ { 2 } \times n _ { 2 } \times n _ { 3 } }$ are orthogonal tensors and $\pmb { S } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ is an $f$ -diagonal tensor.

Similar to matrices, the tensor QR factorization is defined as follows.

Theorem 2 (T-QR) (Kilmer and Martin 2011) Let $x \in$ $\mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ . Then it can be factorized as

$$
\pmb { x } = \pmb { \mathcal { Q } } * \pmb { \mathcal { R } } ,
$$

where $\pmb { \mathcal { Q } } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 1 } \times n _ { 3 } }$ is orthogonal, and $\mathcal { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } } \ i s$ an $f$ -upper triangular tensor whose frontal slices are all upper triangular matrices.

Definition 8 (Tubal rank) (Lu et al. 2020) For a tensor $\pmb { \chi } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ , its tubal rank is defined as the number of nonzero singular tubes of $s$ , where $s$ is the $f$ -diagonal tensor obtained from t-SVD of $x$ . Specially,

$$
r a n k _ { t } \left( \pmb { \mathcal { X } } \right) = \# \left\{ i : \pmb { \mathcal { S } } ( i , i , : ) \neq \pmb { 0 } \right\} .
$$

Lastly, we introduce the tensor spectral norm and condition number.

Definition 9 (Tensor spectral norm ) (Lu et al. 2018) The spectral norm of $\pmb { \chi } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ is defined as

$$
\| \pmb { \mathcal { X } } \| = \sigma _ { \operatorname* { m a x } } \left( b c i r c \left( \pmb { \mathcal { X } } \right) \right) = \operatorname* { m a x } _ { i \in n _ { 3 } } \sigma _ { \operatorname* { m a x } } \left( \overline { { \pmb { X } } } ^ { ( i ) } \right) ,
$$

where $\sigma _ { \mathrm { m a x } } ( X )$ denotes maximum singular value of $X$ .

Definition 10 (Tensor condition number) The condition number of $\pmb { \chi } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ is defined as the condition number of bcirc $( x )$ as

$$
\kappa \left( \pmb { \mathcal { X } } \right) = \kappa \left( b c i r c ( \pmb { \mathcal { X } } ) \right) = \frac { \sigma _ { \operatorname* { m a x } } \left( b c i r c \left( \pmb { \mathcal { X } } \right) \right) } { \sigma _ { \operatorname* { m i n } } \left( b c i r c \left( \pmb { \mathcal { X } } \right) \right) } ,
$$

where the $\sigma _ { \mathrm { m i n } } \left( b c i r c \left( \pmb { \mathcal { X } } \right) \right)$ denotes the smallest nonzero singular value of $b c i r c ( \pmb { \mathscr { X } } )$ .

If the condition number $\kappa \left( \mathcal { X } \right)$ is close to 1, the tensor $x$ is said to be well-conditioned. Conversely, if the condition number $\kappa \left( \mathcal { X } \right)$ is large, then $x$ is deemed ill-conditioned (Tong et al. 2022a). From Figure 1 and Figure 2, recovering the ill-conditioned low-rank tensor $x ^ { \star }$ in the TCS problems is more challenging.

# Algorithms and Theoretical Results

Given the local TCS model (2), a natural formulation is to minimize the fitting loss

$$
\hat { f } ( \pmb { \mathscr { X } } ) : = \sum _ { i = 1 } ^ { n _ { 2 } } \sum _ { j = 1 } ^ { m } \left( y _ { j i } - \langle \pmb { A } _ { i } ( : , j , : ) , \pmb { \mathscr { X } } ( : , i , : ) \rangle \right) ^ { 2 } .
$$

under the constraint that the tubal rank of $x$ is at most equal to $r$ . Based on the t-SVD, we can reparameterize the variable as $\pmb { x } = \pmb { u } * \pmb { \nu }$ to incorporate the low-rank constraint, with $\pmb { \mathcal { U } } \in \mathbb { R } ^ { n _ { 1 } \times r \times n _ { 3 } }$ satisfying the orthogonality constraint $\pmb { \mathcal { U } } ^ { c } \ast \pmb { \mathcal { U } } = \pmb { \mathcal { T } } _ { r }$ and $\pmb { \nu } \in \mathbb { R } ^ { r \times n _ { 2 } \times n _ { 3 } }$ . Overall, the optimization problem is written as:

$$
\operatorname* { m i n } _ { u , \mathcal { V } } \ f ( \mathcal { U } , \mathcal { V } ) = \sum _ { i = 1 } ^ { n _ { 2 } } \sum _ { j = 1 } ^ { m } \left( y _ { j i } - \langle \pmb { A } _ { i } ( : , j , : ) , \mathcal { U } * \mathcal { V } ( : , i , : ) \rangle \right) ^ { 2 }
$$

# $\pmb { \mathcal { U } } ^ { c } * \pmb { \mathcal { U } } = \pmb { \mathcal { I } } _ { r }$

Note that the reparameterization also significantly reduces the number of variables under small $r$ , unlocking the potential of designing low-complexity algorithms. However, as a tradeoff, it introduces nonconvexity through the tensor product in the objective and the orthogonality constraint.

This section proposes algorithms to solve (14) to the global minimum despite nonconvexity. The approach consists of two stages. The first stage employs a spectral initialization to find an initial point $u _ { \mathrm { 0 } }$ that is sufficiently close to the minimizer, thus providing a warm start for the second stage. The second stage is based on a local search strategy that alternately optimizes $u$ and $\nu$ according to (14). In the remainder of this section, we provide a detailed introduction to the two stages.

The following mild assumptions are made on $x ^ { \star }$ and the sensing operator $\{ \mathcal { A } _ { i } \} _ { i = 1 } ^ { n _ { 3 } }$ for obtaining our results.

Assumption 1 The ground truth $\pmb { \chi } ^ { \star } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ has tubal rank $r \ll \operatorname* { m i n } \{ n _ { 1 } , n _ { 2 } , n _ { 3 } \}$ . Its skinny t-SVD is $x ^ { \star } =$ $\mathcal { U } ^ { \star } { * } \mathcal { S } ^ { \star } { * } ( \mathcal { V } ^ { \star } ) ^ { c }$ that $\mathcal { \dot { U } } ^ { \star } \in \mathbb { R } ^ { n _ { 1 } \times \hat { r } \times n _ { 3 } } , \pmb { \mathcal { S } } ^ { \star } \in \mathbb { R } ^ { r \times r \times n _ { 3 } } , \pmb { \mathcal { V } } ^ { \star } \in$ $\mathbb { R } ^ { r \times n _ { 2 } \times n _ { 3 } }$ and $\mathcal { Z } ^ { \star } = \pmb { S } ^ { \star } * ( \pmb { \nu } ^ { \star } ) ^ { c }$ . There exists a finite constant $\mu$ such that $\operatorname* { m a x } _ { i \in [ n _ { 2 } ] } \| \pmb { \mathcal { Z } } ^ { \star } ( : , i , : ) \| _ { F } \leq \mu \sqrt { \frac { r } { n _ { 2 } } } \| \pmb { \mathcal { X } } ^ { \star } \| .$ .

This assumption is similar to the tensor incoherence condition in the low tubal rank tensor recovery literature (Zhang and Aeron 2016; Lu et al. 2018, 2020; Zhang et al. 2020a) and ensures that our problem remains well-posed.

Assumption 2 Each sensing tensor $\mathbf { \mathcal { A } } _ { i } \in \mathbb { R } ^ { n _ { 1 } \times m \times n _ { 3 } } ,$ $\forall i \in$ $[ n _ { 2 } ]$ , has i.i.d. standard Gaussian entries.

# Stage I: Truncated Spectral Initialization

The idea of spectral-based initialization, which is used for providing a “warm start” within the basin of attraction for $x ^ { \star }$ , has been extensively utilized in various non-convex low-rank matrix and tensor recovery problems (Cai et al. 2019; Liu et al. 2024). Inspired by the truncation technique (Chen and Candes 2015; Wang, Giannakis, and Eldar 2017; Vaswani 2024), we design the following initialization method for the local sensing model (2). Specifically, let

$$
\hat { \pmb { \mathscr { X } } } _ { 0 } ( : , i , : ) = \frac { 1 } { m _ { 0 } } \sum _ { j = 1 } ^ { m _ { 0 } } y _ { j i } \pmb { A } _ { i } ( : , j , : ) \cdot \mathbf { 1 } _ { \left\{ | y _ { j i } | \leq \sqrt { \alpha } \right\} } ,
$$

where $^ 1 \{ | y _ { i j } | \leq \sqrt { \alpha } \}$ denotes an indicator function that is equal to 1 when $| y _ { i j } | \le \sqrt { \alpha }$ and 0 otherwise. $m _ { 0 }$ is the number of sensing matrix for per lateral slice. The $\alpha$ is the threshold in truncation and its formula is given in the subsequent theorem. The reason for truncation is that we can use tight sample complexity to bound the concentration of $\hat { \pmb x } _ { 0 }$ on $x ^ { \star }$ . Subsequently, we perform the QR decomposition:

$$
\begin{array} { r } { \hat { \pmb { x } } _ { 0 } = \pmb { \mathcal { Q } } _ { 0 } * \pmb { \mathcal { R } } _ { 0 } , } \end{array}
$$

and initialize the orthogonal tensor $\boldsymbol { u } _ { 0 } \in \mathbb { R } ^ { n _ { 1 } \times r \times n _ { 3 } }$ to be the first $r$ lateral slices of $\scriptstyle { \mathfrak { Q } } _ { 0 }$ , i.e.,

$$
\mathcal { U } _ { 0 } = \mathcal { Q } _ { 0 } ( : , 1 : r , : ) .
$$

The following measure defines the sine of the largest angle between tensor subspaces spanned by their lateral slices.

Definition 11 (Principal angle distance) For two orthogonal tensors $\mathbf { \mathcal { A } } _ { 1 } , \mathbf { \mathcal { A } } _ { 2 } \mathbf { \Psi } \in \mathbb { R } ^ { n _ { 1 } \times r \times n _ { 3 } }$ , the principal angle distance between $\pmb { A } _ { 1 }$ and $\pmb { A } _ { 2 }$ is defined as

$$
D i s \left( \pmb { \mathscr { A } } _ { 1 } , \pmb { \mathscr { A } } _ { 2 } \right) = \| ( \pmb { \mathscr { T } } _ { r } - \pmb { \mathscr { A } } _ { 1 } * \pmb { \mathscr { A } } _ { 1 } ^ { c } ) * \pmb { \mathscr { A } } _ { 2 } \| .
$$

Based on this measure, we can prove the effectiveness of the proposed initialization method in the following theorem.

Theorem 3 Consider the TCS model (2) under Assumption 1 and 2. The initialization $u _ { 0 }$ in Algorithm $\jmath$ satisfies

$$
D i s \left( { { \mathcal { U } } _ { 0 } } , { { \mathcal { U } } ^ { \star } } \right) \leq \frac { 0 . 0 1 6 } { \sqrt { r } \kappa ^ { 2 } }
$$

with a probability at least

$$
\begin{array} { c } { { 1 - \exp \left( c _ { 1 } \left( n _ { 1 } + n _ { 2 } \right) \log n _ { 3 } - \frac { c _ { 2 } m _ { 0 } n _ { 2 } } { \kappa ^ { 8 } \mu ^ { 2 } n _ { 3 } r ^ { 2 } } \right) } } \\ { { - \exp \left( - c _ { 3 } \frac { m _ { 0 } n _ { 2 } } { \kappa ^ { 8 } \mu ^ { 2 } r ^ { 2 } } \right) , } } \end{array}
$$

where $\kappa$ is the tensor condition number of $x ^ { \star }$ and $c _ { 1 } , c _ { 2 } , c _ { 3 }$ are universal constants that are independent of model parameters.

Theorem 3 immediately implies the following sample complexity for our spectral initialization scheme.

Corollary 1 In the same setting as Theorem 3, if the sample size $m _ { 0 }$ for initialization satisfies

$$
m _ { 0 } n _ { 2 } \gtrsim \kappa ^ { 8 } \mu ^ { 2 } r ^ { 2 } n _ { 3 } ( n _ { 1 } + n _ { 2 } ) \log n _ { 3 } ,
$$

then (18) holds with probability at least 1 − (n1+1n3)10 .

Compared to the total number of entries in $u ^ { \star }$ and ${ { \mathcal { Z } } ^ { \star } }$ , which is $r n _ { 3 } ( n _ { 1 } + n _ { 2 } )$ , Corollary 1 shows a good initialization can be achieved with a sample size only having an additional factor of $r \log n _ { 3 }$ (modulus constants $\kappa , \mu )$ .

# Stage II: Local Search

The second stage concerns iteratively refining the initial point $u _ { 0 }$ computed by Stage I by local search. Since the objective $f$ in (14) is bi-convex in $u$ and $\nu$ , based on this structure, we first propose the Alt-PGD-Min algorithm that alternately updates these two factors.

The Alt-PGD-Min Algorithm. Let $\boldsymbol { u } _ { t }$ and $\nu _ { t }$ be the values of $u$ and $\nu$ at iteration $t$ .

1) Exact minimization for $\nu$ : Fixing $\boldsymbol { u } _ { t }$ , the lateral slices of $\nu$ are decoupled in problem (14). Thus, we can update each lateral slice $\boldsymbol { \nu } ( : , i , : )$ in parallel by solving the following minimization problem

$$
\mathcal { V } _ { t } ( i ) \in \underset { \pmb { \mathscr { B } } \in \mathbb { R } ^ { r \times 1 \times n _ { 3 } } } { \arg \operatorname* { m i n } } \sum _ { j = 1 } ^ { m _ { c } } \big ( y _ { j i } - \langle \pmb { \mathscr { U } } _ { t } ^ { c } \ast \pmb { \mathscr { A } } _ { i } ( j ) , \pmb { \mathscr { B } } \rangle \big ) ^ { 2 } ,
$$

which can be reformulated as follows based on Definition 2 $\pmb { v } _ { t , i } \in \underset { \pmb { v } \in \mathbb { R } ^ { r n _ { 3 } } } { \arg \operatorname* { m i n } } \left\| \left( \mathrm { b c i r c } \left( \mathscr { U } _ { t } ^ { c } \right) \cdot \mathrm { U n f o l d } \left( \pmb { A } _ { i } \right) \right) ^ { c } \cdot \pmb { v } - \pmb { y } _ { i } \right\| ^ { 2 } .$

The problem (22) is a least squares problem. Thus, for every $i \in [ n _ { 2 } ]$ , a closed-form solution can be derived as follows:

$$
\begin{array} { r l } & { H _ { t , i } = \mathrm { b c i r c } \left( \mathcal { U } _ { t } ^ { c } \right) \cdot \mathrm { U n f o l d } \left( \mathcal { A } _ { i } \right) , } \\ & { v _ { t , i } = \left( H _ { t , i } H _ { t , i } ^ { c } \right) ^ { - 1 } H _ { t , i } y _ { i } , } \\ & { \mathcal { V } _ { t } ( i ) = \mathrm { F o l d } \left( v _ { t , i } \right) . } \end{array}
$$

2) Projected gradient descent for $u$ : Although for fixed $\nu _ { t }$ , $f$ is also convex in $u$ , in pursuit of computation-efficient update, instead of performing exact minimization, we employ a first-order gradient descent step to update $u$ (Gu et al. 2024), followed by a projection step onto the orthogonality constraint set. Specifically, we first compute

$$
\begin{array} { r } { \hat { \mathcal { U } } _ { t + 1 } = \mathcal { U } _ { t } - \eta \displaystyle \sum _ { i = 1 } ^ { n _ { 2 } } \sum _ { j = 1 } ^ { m _ { c } } \left( y _ { j i } - \langle \mathcal { U } _ { t } ^ { c } \ast \mathcal { A } _ { i } ( j ) , \mathcal { V } _ { t } ( i ) \rangle \right) } \\ { \cdot \mathcal { A } _ { i } ( j ) \ast \left( \mathcal { V } _ { t } ( i ) \right) ^ { c } , \phantom { x x x x x x x x x x x x x x x x x x x x x x x x x x } } \end{array}
$$

with step size $\eta > 0$ . Then we obtain a tensor $\hat { \pmb { \mathscr { Q } } } _ { t + 1 }$ by the QR decomposition $\hat { \mathcal { U } } _ { t + 1 } = \hat { \mathcal { Q } } _ { t + 1 } * \hat { \mathcal { R } } _ { t + 1 }$ . The updated of $u$ is given by

$$
\mathcal { U } _ { t + 1 } = \hat { \mathcal { Q } } _ { t + 1 } ( : , 1 : r , : ) .
$$

The complete Alt-PGD-Min algorithm is described in Algorithm 1. It is worth mentioning that we use the samplesplitting technique in Algorithm 1, where we divide the total samples and measurements pairs of each slice into $2 T + 1$ groups as $\{ \mathcal { A } _ { i } ^ { ( k ) } \} _ { k = 1 } ^ { 2 T + 1 }$ . The last two groups are used for initialization that the sample size for each lateral slice is $m _ { 0 }$ , and Alt-PGD-Min draws two fresh groups of samples from the remaining groups per iteration, where the sample size for each lateral slice is $m _ { c }$ . The splitting strategy ensures statistical independence of the samples across iterations, which is a key component to simplifying the convergence analysis and has been used in various low matrix and tensor learning algorithms (Hardt and Wootters 2014; Jain and Netrapalli 2015; Ding and Chen 2020; Cai, Li, and Xia 2022).

The computational complexity per iteration in Alt-PGD-Min for updating $u$ and $\nu$ is $\mathcal { O } \left( n _ { 1 } n _ { 3 } m _ { c } r + n _ { 3 } m _ { c } r + n _ { 1 } \bar { n _ { 2 } } n _ { 3 } r \bar { + } n _ { 1 } n _ { 3 } r ^ { 2 } \right)$ and $\mathcal { O } \left( n _ { 1 } n _ { 2 } n _ { 3 } m _ { c } r + m _ { c } ( r n _ { 3 } ) ^ { 2 } n _ { 2 } + ( n _ { 3 } r ) ^ { 3 } n _ { 2 } \right)$ , respectively. Theorem 4 In the same setting as Theorem 3, if the initialization $u _ { 0 }$ satisfies (18) and m cη ⋆ 2 with cη ≤ 0.9, then the iterates generated by Alt-PGD-Min satisfies $\begin{array} { r l } & { D i s ( \mathcal { U } _ { t } , \mathcal { U } ^ { \star } ) \leq \bigg ( 1 - \frac { 0 . 8 4 c _ { \eta } } { \kappa ^ { 2 } } \bigg ) ^ { t } \cdot D i s ( \mathcal { U } _ { 0 } , \mathcal { U } ^ { \star } ) , } \\ & { \| \pmb { \mathcal { X } } _ { t } ( i ) - \pmb { \mathcal { X } } ^ { \star } ( i ) \| _ { F } \leq 1 . 4 D i s ( \mathcal { U } _ { t } , \mathcal { U } ^ { \star } ) \cdot \| \pmb { \mathcal { X } } ^ { \star } ( i ) \| _ { F } } \end{array}$ (26) fo $\cdot \forall t \geq 0 , i \in [ n _ { 2 } ]$ with a probability at least $\begin{array} { l } { { 1 - \exp \left( c _ { 4 } ( n _ { 1 } + r ) \log n _ { 3 } - \frac { c _ { 5 } m _ { c } n _ { 2 } } { \kappa ^ { 4 } \mu ^ { 2 } n _ { 3 } r } \right) } } \\ { { - \exp \left( \log n _ { 2 } + r \log n _ { 3 } - c _ { 6 } m _ { c } \right) , } } \end{array}$ (27)

Input: Number of iteration $T$ , total sensing tensor with sample splitting $\{ \{ \pmb { A } _ { i } ^ { ( k ) } \} _ { k = 1 } ^ { 2 T + 1 } \} _ { i = 1 } ^ { n _ { 2 } }$ , corresponding sample-splitting local measureme s $\{ \{ \pmb { y } _ { i } ^ { ( k ) } \} _ { k = 1 } ^ { 2 T + 1 } \} _ { i = 1 } ^ { n _ { 2 } }$ $r , \kappa , \mu , n _ { 2 }$ , step size $\eta$ .   
1: for $t = 0 , 1 , \dots , T - 1$ do   
2: ▷ Update $u$   
3: if $t = 0$ then $D$ Initialization   
4: Set $\pmb { \mathcal { A } } _ { i } = \pmb { \mathcal { A } } _ { i } ^ { ( 2 T ) } , \pmb { \mathscr { y } } _ { i } = \pmb { \mathscr { y } } _ { i } ^ { ( 2 T ) } , \forall i \in [ n _ { 2 } ] ,$   
5: Calculate α = C κ2µ2 Pin=21 Pjm=1 yji ,   
6: $\pmb { \mathscr { A } } _ { i } = \pmb { \mathscr { A } } _ { i } ^ { ( 2 T + 1 ) } , \pmb { \mathscr { y } } _ { i } = \pmb { \mathscr { y } } _ { i } ^ { ( 2 T + 1 ) } , \forall i \in [ n _ { 2 } ] ,$ 7: Construct $\hat { \pmb x } _ { 0 }$ as (15),   
8: Conduct QR decomposition $\hat { \pmb { x } } _ { 0 } = \hat { \pmb { \mathscr { Q } } } _ { 0 } * \hat { \pmb { \mathscr { R } } } _ { 0 }$ , 9: Initialize $u _ { 0 }$ by top- $\cdot r$ lateral slices of $\hat { \mathcal { Q } } _ { 0 }$ . 10: else   
11: 12: for $\begin{array} { r l } & { \quad : = \mathbf { 1 } , \boldsymbol { \mathscr { L } } , \ldots , \boldsymbol { n } _ { 2 } \mathbf { \Phi } \mathbf { u } \mathbf { 0 } } \\ & { \quad \mathcal { A } _ { i } = \pmb { A } _ { i } ^ { ( T + t ) } , \pmb { y } _ { i } = \pmb { y } _ { i } ^ { ( T + t ) } , \forall i \in [ n _ { 2 } ] , } \\ & { \quad H _ { t - 1 , i } = \mathrm { b c i r c } ( \mathscr { U } _ { t - 1 } ^ { c } ) \cdot \mathrm { U n f o l d } ( \pmb { A } _ { i } ) , } \\ & { \quad b _ { t - 1 , i } = H _ { t - 1 , i } ^ { c } \mathrm { U n f o l d } ( \mathscr { V } _ { t - 1 } ( i ) ) - \pmb { y } _ { i } , } \\ & { \quad \mathscr { T } _ { t - 1 } ( : , i , : ) = \sum _ { j = 1 } ^ { m } ( b _ { t - 1 , i } ) _ { j } \pmb { A } _ { i } ( : , j , : ) , } \end{array}$ $i = 1 , 2 , \ldots , n _ { 2 }$ do 13:   
14:   
15:   
16: end for   
/\*\*\*Alt-PGD-Min\*\*\*/   
17: $\begin{array} { r l } & { \quad \hat { \mathcal { U } } _ { t } ^ { \sf H - P \sf U L - m - n ! \sf U I I I } ^ { - \cdots \cdots } } \\ & { \quad \hat { \mathcal { U } } _ { t } = \mathcal { U } _ { t } - \eta \mathcal { T } _ { t - 1 } * \mathcal { V } _ { t - 1 } ^ { c } , } \\ & { \quad * * * \sf A | { \sf t - S c a l e P G D - M i n } ^ { * * * / } } \\ & { \quad \hat { \mathcal { U } } _ { t } = \mathcal { U } _ { t } - \eta \mathcal { T } _ { t - 1 } * \mathcal { V } _ { t - 1 } ^ { c } * ( \mathcal { V } _ { t - 1 } * \mathcal { V } _ { t - 1 } ^ { c } ) ^ { - 1 } , } \end{array}$ /   
18:   
19: Calculate $\boldsymbol { u } _ { t }$ as (25) by QR decomposition, 20: end if   
▷ Update $\nu$   
21: for $i = 1 , 2 , \ldots , n _ { 2 }$ do   
22: $\begin{array} { r l } & { \quad _ { \pmb { \mathscr { A } } _ { i } } = \pmb { \mathscr { A } } _ { i } ^ { ( t + 1 ) } , \pmb { \mathscr { y } } _ { i } = \pmb { \mathscr { y } } _ { i } ^ { ( t + 1 ) } , \forall i \in [ n _ { 2 } ] , } \\ & { \pmb { H } _ { t , i } = \mathrm { b c i r c } ( \pmb { \mathscr { U } } _ { t } ^ { c } ) \cdot \mathrm { U n f o l d } ( \pmb { \mathscr { A } } _ { i } ) , } \\ & { \pmb { v } _ { t , i } = \left( \pmb { H } _ { t , i } \pmb { H } _ { t , i } ^ { c } \right) ^ { - 1 } \pmb { H } _ { t , i } \pmb { y } _ { i } , } \\ & { \pmb { \mathscr { V } } _ { t } ( i ) = \mathrm { F o l d } ( \pmb { v } _ { t \ { i } } ) . } \end{array}$   
23:   
24:   
25:   
26: end for   
▷ Update   
27: $\pmb { \mathscr { X } } _ { t } = \pmb { \mathscr { U } } _ { t } * \pmb { \mathscr { V } } _ { t }$ ,   
28: end for   
Output: Recover tensor $\scriptstyle x _ { T - 1 }$ .

where $c _ { 4 } , c _ { 5 } , c _ { 6 }$ are universal positive constants independent from model parameters.

Theorem 4 shows that even (14) is non-convex, if the initialization $u _ { 0 }$ is sufficiently close to the minimizer, the iterations of Alt-PGD-Min will converge at a linear rate to $x ^ { \star }$ .

Corollary 2 In the same setting as Theorem 4, if the $\eta =$ mc0.8 ⋆ and the sample size mc satisfies

$m _ { c } \gtrsim \operatorname* { m a x } \left\{ \kappa ^ { 4 } \mu ^ { 2 } r n _ { 1 } n _ { 3 } \log n _ { 3 } / n _ { 2 } , \log n _ { 2 } , r \log n _ { 3 } \right\}$ (28) then it takes $T = c _ { 7 } \kappa ^ { 2 } \log \frac { 1 } { \epsilon }$ iterations for Alt-PGD-Min to achieve $\epsilon$ -accuracy recovery, i.e.,

$$
\begin{array} { r l } & { D i s \left( \mathcal { U } _ { T } , \mathcal { U } ^ { \star } \right) \leq \epsilon , } \\ & { \| \pmb { \mathcal { X } } _ { T } ( i ) - \pmb { \mathcal { X } } ^ { \star } ( i ) \| _ { F } \leq 1 . 4 \epsilon \| \pmb { \mathcal { X } } ^ { \star } ( i ) \| _ { F } , \forall i \in [ n _ { 2 } ] } \end{array}
$$

with probability at least 1 − (n1+1r)10 .

The sample complexity given by (28) scales linearly with $r$ , showing improved dependence on $r$ compared to that for the initialization Stage I. When the number of lateral slices is large enough such that $n _ { 2 } \gtrsim \kappa ^ { 4 } \mu ^ { 2 } n _ { 1 } \log n _ { 3 }$ , the order of $m _ { c }$ becomes $\mathcal { O } ( r n _ { 3 } )$ , which is significantly smaller than the size of lateral slice $n _ { \mathrm { 1 } } n _ { \mathrm { 3 } }$ as $r \ll \operatorname* { m i n } ( n _ { 1 } , n _ { 2 } )$ .

Combining the results of Corollary 1 and Corollary 2, we can immediately conclude the overall sample complexity of Alt-PGD-Min as follows.

Corollary 3 Consider the TCS model (2) under Assumption $\jmath$ and 2. For Alt-PGD-Min to achieve ϵ-accuracy recovery as described by (29) with high probability at least − (n +2r)10 , the total sample complexity m for each lateral slice is

$$
m n _ { 2 } \gtrsim \kappa ^ { 6 } \mu ^ { 2 } r n _ { 3 } \log n _ { 3 } \left( \kappa ^ { 2 } r \left( n _ { 1 } + n _ { 2 } \right) + n _ { 1 } \log \frac { 1 } { \epsilon } \right)
$$

Corollary 4 In the same setting as Theorem 5, if $\begin{array} { r } { \eta = \frac { 0 . 8 } { m _ { c } } } \end{array}$ and sample size for each lateral slice $m _ { c }$ satisfies (28), then to obtain the $\epsilon$ -accuracy recovery as described by (29), the iteration complexity for Alt-Scale-GD is

$$
T = c _ { 7 } \log { \frac { 1 } { \epsilon } } .
$$

.Theorem 5 and Corollary 4 show Alt-ScalePGD-Min converges linearly at a rate that is independent of the condition number $\kappa$ , significantly improving over the $\mathcal { O } \left( \kappa ^ { 2 } \log \frac { 1 } { \epsilon } \right)$ complexity of Alt-PGD-Min.

Corollary 5 Consider the TCS model (2) under Assumption 1 and 2. The total sample complexity $m$ for each lateral slice to achieve $\epsilon$ -accuracy recovery as (29) with high probability at least $\begin{array} { r } { 1 - \frac { 2 } { ( n _ { 1 } + r ) ^ { 1 0 } } } \end{array}$ is

$$
m n _ { 2 } \gtrsim \kappa ^ { 4 } \mu ^ { 2 } r n _ { 3 } \log n _ { 3 } \left( \kappa ^ { 4 } r ( n _ { 1 } + n _ { 2 } ) + n _ { 1 } \log \frac { 1 } { \epsilon } \right)
$$

and $\begin{array} { r } { m \gtrsim \kappa ^ { 2 } \operatorname* { m a x } \left\{ \log n _ { 2 } , r \log n _ { 3 } \right\} \log \frac { 1 } { \epsilon } . } \end{array}$

The total sample complexity in (30) comprises two parts: one is from initialization and the other is from iterative refinements. The dependency of the sample complexity on recovery accuracy $\epsilon$ is because of the sample splitting strategy introduced in the algorithm, which is common in all the analyses where such a technique is adopted (Jain, Netrapalli, and Sanghavi 2013; Hardt and Wootters 2014; Ding and Chen 2020; Vaswani 2024).

Alt-ScalePGD-Min: Acceleration by Preconditioning. To mitigate the influence of large $\kappa$ and improve the algorithm efficiency, especially for ill-conditioned problems, we propose to accelerate by pre-conditioning the gradient step that is sensitive to $\kappa$ . Recall (24) and let $\bar { \pmb { b } } _ { t , i } : = { \pmb { H } } _ { t , i } ^ { c } { \pmb { v } } _ { t , i } \bar { \ - }$ $\mathbf { \boldsymbol { y } } _ { i } , i \in [ n _ { 2 } ]$ and $\begin{array} { r } { \pmb { \mathcal { T } } _ { t } ( : , i , : ) : = \sum _ { j = 1 } ^ { m _ { c } } ( \pmb { b } _ { t , i } ) _ { j } \cdot \pmb { \mathcal { A } } _ { i } ( : , j , : ) } \end{array}$ , we rewrite the updating step as

$$
\hat { \mathcal { U } } _ { t + 1 } = \mathcal { U } _ { t } - \eta \mathcal { T } _ { t } * \mathcal { V } _ { t } ^ { c } .
$$

Comparing to the gradient step (31), the key difference of Alt-ScalePGD-Min is that it preconditions the search direction of $\boldsymbol { u } _ { t }$ by inverse of $\nu _ { t } * \nu _ { t } ^ { c }$ , i.e.,

$$
\begin{array} { r } { \hat { \mathcal { U } } _ { t + 1 } = \mathcal { U } _ { t } - \eta \mathcal { T } _ { t } * \mathcal { V } _ { t } ^ { c } * \left( \mathcal { V } _ { t } * \mathcal { V } _ { t } ^ { c } \right) ^ { - 1 } . } \end{array}
$$

Note that the inverse in (32) is easy to compute because the related tensor has a size of $\boldsymbol { r } \times \boldsymbol { r } \times \boldsymbol { n } _ { 3 }$ , significantly smaller than the dimension of the tensor factors. Thus, each iteration of scaled GD incurs minor additional complexity $\mathcal { O } \left( n _ { 1 } r ^ { 2 } n _ { 3 } + r ^ { 3 } n _ { 3 } \right)$ than the GD of Alt-PGD-Min in (31).

The convergence of Alt-ScalePGD-Min is given in the following theorem.

Theorem 5 Consider the TCS model (2) under Assumption 1 and 2. Let the step size $\begin{array} { r } { \eta = \frac { c _ { \eta } } { m _ { c } } } \end{array}$ cη and all other parameters are set the same as Theorem 4, then the iterates generated by Alt-ScalePGD-Min satisfy

$$
D i s \left( \mathcal { U } _ { t } , \mathcal { U } ^ { \star } \right) \leq \left( 1 - 0 . 8 9 c _ { \eta } \right) ^ { t } \cdot D i s \left( \mathcal { U } _ { 0 } , \mathcal { U } ^ { \star } \right)
$$

with at least the same probability as (27).

and $m \gtrsim \operatorname* { m a x } \left\{ \log n _ { 2 } , r \log n _ { 3 } \right\} \log \frac { 1 } { \epsilon } ,$

Due to the same initialization, the first term in (30) and (35) are the same. However, Alt-ScalePGD-Min improves over Alt-PGD-Min by a factor of $\kappa ^ { 2 }$ in the second term due to improved convergence rate. When the recovery accuracy is sufficiently high such that the second term dominates the first, i,e., $\begin{array} { r } { \operatorname* { l o g } \frac { 1 } { \epsilon } \gtrsim \kappa ^ { 4 } r ( 1 + \frac { n _ { 2 } } { n _ { 1 } } ) } \end{array}$ , the total sample complexity of Alt-ScalePGD-Min is $\begin{array} { r } { \stackrel { \cdots } { \mathcal { O } } \left( \kappa ^ { 4 } \mu ^ { 2 } r n _ { 1 } n _ { 3 } \log n _ { 3 } \log \frac { 1 } { \epsilon } \right) } \end{array}$ , significantly improving upon the $\mathcal { O } \left( \kappa ^ { 6 } \mu ^ { 2 } r n _ { 1 } n _ { 3 } \log n _ { 3 } \log \frac { 1 } { \epsilon } \right)$ ） of Alt-PGD-Min for large κ.

# Experiments

We evaluate our proposed methods on both synthetic and real-world data. Since model (2) has not been studied in existing works, we can only compare our method with the low-rank matrix column-wise CS method (LRcCS) (Nayer and Vaswani 2022), which is closest to our TCS model in application. To compare with this method, we conduct $\operatorname { U n f o l d } ( \mathcal { X } ^ { \star } )$ operation which reshapes each lateral slice into a vector. The performance of algorithms is measured by the relative recovery err or ∥X t− ⋆X ⋆∥F , which is plotted concerning the iteration number.

# Synthetic Data

We generate a synthetic tensor with $n _ { 1 } = n _ { 3 } = 2 0 , n _ { 2 } =$ $4 0 0 , r = 4$ , of which the details are in the Appendix. The sample sizes are $m _ { 0 } = 2 0 0$ and $m _ { c } = 1 0 0$ without sample splitting. We test three algorithms with the same step size and sample sizes under different $\kappa = 1 , 2 , 4$ . The results are plotted in Figure 1, which shows that both our proposed methods have linear convergence rates while LRcCS fails in all cases. The convergence rate of Alt-PGD-Min becomes slower with increasing $\kappa$ while Alt-ScalePGD-Min converges with independence on $\kappa$ , with all curves overlapping.

In the second setting, the data generation is the same as the first. However, we validate the performance under

Alt-PGD-Min $\kappa = 1$ 0 Alt-PGD-Min $\kappa = 2$ Alt-PGD-Min $\kappa = 4$   
-10 -Alt-ScalePGD-Min- $\kappa = 1$ -Alt-ScalePGD-Min $\kappa = 2$ -Alt-ScalePGD-Min $\kappa = 4$ 1   
-20 .LRcCS $\kappa = 1$ LRcCS-k = 2 LRcCS-κ=4   
-30   
-40 0 500 1000 Iteration number

random initialization, which has i.i.d. standard Gaussian entries. Without good initialization, we increase the $m _ { c }$ slightly to $m _ { c } = 1 2 0$ . The results depicted in Figure 2 show that both of the proposed algorithms converge with linear rates after a small number of iterations during the initial phase while LRcCS still does not work. With larger $\kappa$ , AltPGD-Min slows down significantly while the convergence speed of Alt-ScalePGD-Min remains almost the same with almost negligible initial phases.

Alt-PGD-Min-k=1   
10 Alt-PGD-Min-k = 2 Alt-PGD-Min-κ=4 0 Alt-ScalePGD-Min $\kappa = 1$ Alt-ScalePGD-Min-κ = 2   
-10 -Alt-ScalePGD-Min-k = 4 LRcCS-κ =1   
-20 LRcCS-k=2 LRcCS-K=4   
-30   
-40 0 100 200 300 400 Iteration number

# Video Compressed Sensing

We test the proposed TCS model (2) in the plane video sequence (only approximately low tubal rank) that has been used in previous work such as (Nayer, Narayanamurthy, and Vaswani 2019). It has 105 frames and each frame has been resized into $4 8 \times 6 4$ . We use the same samples with size $m \ : = \ : 1 6 0 0$ for initialization and iteration. We set $r = 1 0$ for three methods. For each method, we tune the step size to guarantee it converges and achieves performance as good as possible. The visual and quantitative comparison results are shown in Figure 3 and Figure 4, respectively. Alt-ScalePGDMin has the fastest convergence rate with the best recovery performance (Figure 3. (a) selected at 20-th iteration). Both Alt-PGD-Min and LRcCS converge very slowly with unsatisfied performance (Figure 3. (c) and (d) are selected by running 8000 iterations). This is because the video has very large matrix and tensor condition numbers that result in slow convergence rates.

We can observe that while Alt-PGD-Min outperforms LRcCS in synthetic data experiments, it performs worse in video compressive sensing. This discrepancy is because the synthetic data is generated to be exactly low-tubal-rank with a controlled tensor condition number. In contrast, the video data is only approximately low tubal rank and has a significantly larger tensor condition number compared to its reshaped matrix condition number. As a result, given the same number of iterations, Alt-PGD-Min performs worse than LRcCS in the video CS scenario.

![](images/48dc9b7c3543d1f19bbc47dfcb787f51317f035f035dba3fc379af659cf2b910.jpg)  
Figure 1: Spectral initialization with $m _ { 0 } = 2 0 0 , m _ { c } = 1 0 0$ .   
Figure 2: Random initialization with $m _ { c } = 1 2 0$ .   
Figure 3: Visualization of frame-7 in recovered videos.   
Figure 4: Quantitative comparison

45 45 40 40 LRcCS PR 35 Alt-PGD-Min Alt-ScalePGD-Min 30 LRcCS 25 eAlt-PGD-Min 25 Alt-ScalePGD-Min 20 20 20 40 60 80 100 0 50 100 Frame Iteration number (a) PSNR for each frame (b) Convergence comparison

# Conclusion

We introduced a novel local TCS model for low-tubal-rank tensors and developed two algorithms to solve this model with theoretical guarantees. The numerical results demonstrated the effectiveness of our method. There are several problems worth studying in the future. For instance, it would be valuable to investigate whether the current theoretical guarantees can be established without the sample-splitting technique. Additionally, exploring the possibility of achieving global convergence under random initialization, without spectral initialization, remains challenging. Lastly, improving the complexity dependence on the tensor condition number is another interesting direction for future work.

# Acknowledgments

The work of Fan was partially supported by the Youth Program 62106211 of the National Natural Science Foundation of China.