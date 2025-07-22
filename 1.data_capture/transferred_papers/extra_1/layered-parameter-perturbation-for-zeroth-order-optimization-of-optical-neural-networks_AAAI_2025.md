# Layered-Parameter Perturbation for Zeroth-Order Optimization of Optical Neural Networks

Hiroshi Sawada1, Kazuo Aoyama1, Masaya Notomi2

1Communication Science Laboratories, NTT Corporation, Japan 2Basic Research Laboratories, NTT Corporation, Japan hrsh.sawada@ntt.com, kazuo.aoyama@ntt.com, masaya.notomi@ntt.com

# Abstract

Optical neural networks (ONNs) have attracted great attention due to their low power consumption and high-speed processing. When training an ONN implemented on a chip with possible fabrication variations, the well-known backpropagation algorithm cannot be executed accurately because the perfect information inside the chip cannot be observed. Instead, we employ a black-box optimization method such as zeroth-order (ZO) optimization. In this paper, we first discuss how ONN parameters should be perturbed to search for better values in a black-box manner. Conventionally, parameter perturbations are sampled from a normal distribution with an identity covariance matrix. This is plausible if the parameters are not interrelated in a module, like a linear module of an ordinary neural network. However, this is not the best way for ONN modules with layered parameters, which are interrelated by optical paths. We then propose to perturb the parameters by a normal distribution with a special covariance matrix computed by our novel method. The covariance matrix is designed so that the perturbations appearing at the module output caused by the parameter perturbations become as isotropic as possible to uniformly search for better values. Experimental results show that the proposed method using the special covariance matrix significantly outperformed conventional methods.

# 1 Introduction

Neural networks (NNs) have demonstrated remarkable performance in a wide variety of tasks at the ever-increasing cost of their size and power consumption. In this regard, optical neural networks (ONNs) are expected to be one of the promising systems due to their intrinsic characteristics of low power consumption and high-speed processing (Shen et al. 2017; Fang et al. 2019; Tang et al. 2021; Ashtiani, Geers, and Aflatouni 2022). An ONN is implemented on a programmable optical circuit based on silicon photonics. It processes complex-valued analog signals with coherent light (Carolan et al. 2015; Bogaerts et al. 2020). Figure 1 shows the structure of an ONN, which consists of linear and nonlinear modules as well as an ordinary NN. A linear module in an ONN typically comprises an array of Mach-Zehnder interferometers (MZIs) connected by waveguides (Reck and Zeilinger 1994; Clements et al. 2016). The MZI consists of two pairs of a phase shifter (PS) with a phase parameter $\theta$ and a beam splitter (BS). A nonlinear module may also have parameters. The programmability (or trainability) of ONNs is achieved by adjusting such parameters.

A major approach to training NNs in general is to compute the gradient of a loss function with respect to parameters. And the backpropagation algorithm (Bishop 2006; LeCun, Bengio, and Hinton 2015) is very common to compute the gradient. However, for training ONNs, we cannot accurately compute the gradient by backpropagation. This is because each ONN circuit implemented on a chip has its own fabrication variations (Fang et al. 2019; Banerjee, Nikdast, and Chakrabarty 2023). As shown in the bottom area of Fig. 1, let us model such variations by errors from ideal component characteristics, such as splitting angle errors $\gamma \in \mathbb { R }$ (Mikkelsen, Sacher, and Poon 2014; Vadlamani, Englund, and Hamerly 2023) and attenuation-phase errors $\zeta \in \mathbb { C }$ . We may be able to use backpropagation with an ONN model simulated on an ordinary computer, with (Zheng et al. 2023; Sawada, Aoyama, and Ikeda 2024) or without modeling these errors, for an early stage of training. However, it is not a good idea to rely only on such imperfect gradients obtained from the backpropagation until the final stage of training, where accurate control of the parameters is of great importance.

As the perfect information inside an ONN chip is not precisely observable, black-box optimization methods have been applied to ONN training, including particle swarm optimization (Zhang et al. 2019), covariance matrix adaptation evolution strategy (CMA-ES) (Chen et al. 2022; Lupo et al. 2023), and zeroth-order (ZO) optimization (Shen et al. 2017; Zhou et al. 2020; Gu et al. 2020, 2021; Bandyopadhyay et al. 2022). Among these methods, ZO optimization (Liu et al. 2020) has become a major approach as (Gu et al. 2020) has experimentally shown that ZO optimization was superior to particle swarm optimization. We continue to focus on ZO optimization not only because of the shown superiority but also because it computes the (approximate) gradient of a loss function like backpropagation and is expected to scale to an ONN with a large number of parameters.

ZO optimization randomly perturbs the parameters to search for better values. The random parameter perturbations are generally sampled from a normal distribution $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ with an identity covariance matrix I (Liu et al. 2020;

moduleu (linear) module u' (nonlinear) parameters: 0u ∈ RNu parameters: 0u ∈ RNu' ONN Module Module ONN input input output output vector vector vector vector x;∈ CK xiu∈CKu Yiu E CMu yi ∈ CM ， C Phase shifter (PS) Mach-Zehnder Beam splitter (BS) Phase parameter $\theta$ interferometer (MZI) Splitting angle: Attenuation-phase $\bar { \phi } { = } \{ \bar { ( \pi / 2 ) } + \gamma \} / 2$ error: $\zeta \in \mathbb { C }$ $| \zeta | \le 1$ Angle error: $\gamma \in \mathbb { R }$ $\zeta = 1$ : no error PS BS PS BS $\gamma = 0$ : no error

Gu et al. 2020), or similarly from independent Bernoulli distributions (Bandyopadhyay et al. 2022). This means that the parameters are perturbed independently in a module as well as in a whole NN. This is plausible for a linear module of an ordinary NN, where each parameter as an element of a matrix is independently responsible for each input-output pair. However, the parameters of MZI-based ONN linear modules are not independent, but interrelated to each other, as represented by their layered structures shown in the upper part of Fig. 2. By interrelation, we mean that a change in one parameter can be approximated by changes in its upstream/downstream parameters. This fact suggests us to consider another way of generating random perturbations than the one from $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ . One way is to use coordinate-wise perturbations (Liu et al. 2020; Gu et al. 2020, 2021). It avoids the effect of interrelation because only one parameter has a non-zero element in its perturbation vector. However, it is not very efficient for a need to change many parameters.

This paper proposes a new notion we call layeredparameter perturbation, which fits well to ONN parameters. Specifically, we propose to perturb the parameters of a linear module $u$ with a layered structure by a normal distribution $\mathcal { N } ( \mathbf { 0 } , \pmb { \Sigma } _ { u } )$ with a covariance matrix $\Sigma _ { u }$ computed by our novel method. The covariance matrix is designed so that the perturbations of the output vector $\pmb { y } _ { i u }$ (Fig. 1), caused by parameter perturbations, become as isotropic as possible (Fig. 3). As a consequence, the whole ONN can be better trained in a non-informative black-box setting by uniformly searching for better values at the output of linear modules. Note that our special covariance matrices designed as above are fundamentally different from those in CMA-ES (Hansen 2016), which are computed directly from selected solutions of the population.

The notion of layered-parameter perturbation is considered for each linear module $u$ in this paper, although it can be extended to the whole ONN. The reasons are twofold. The first is to keep our discussion and proposal simple. The second is to make the complexity of computing the covariance matrix $\Sigma _ { u }$ feasible. If it were of the whole ONN, the number of matrix elements would become the square of the total parameters. This would be infeasible for a large ONN with many parameters.

The rest of this paper is organized as follows. Section 2 provides the prerequisite knowledge of ONNs, training NNs, ZO optimization, and Jacobian matrix. After we show motivating examples in Sec. 3 regarding how parameters are interrelated, we propose a novel method for calculating the covariance matrix $\Sigma _ { u }$ in Sec. 4. Section 5 shows the experimental results. Section 6 concludes the paper. Table 1 lists the notations used in the paper.

# 2 Preliminaries Optical Neural Network (ONN)

As shown in Fig. 1, an ONN consists of linear and nonlinear modules. Let a module $u$ have $N _ { u }$ real-valued parameters $\pmb { \theta } _ { u }$ . As a total, the ONN has $\begin{array} { r } { N = \sum _ { u } N _ { u } } \end{array}$ parameters $\pmb \theta =$ $[ \theta _ { 1 } , \dots , \theta _ { N } ] \in \mathbb { R } ^ { N }$ . The ONN imp ements a function $f$ that produces a complex-valued output vector ${ \pmb y } _ { i } = { \pmb f } ( { \pmb x } _ { i } , { \pmb \theta } ) \in$ $\mathbf { \bar { \mathbb { C } } } ^ { M }$ for an input vector $\pmb { x } _ { i } \in \mathbb { C } ^ { \hat { K } }$ . If we look at a module $u$ , it implements a function $f _ { u }$ that produces an output vector $\pmb { y } _ { i u } \overset { \cdot } { = } f _ { u } ( \pmb { x } _ { i u } , \theta _ { u } ) \in \mathbb { C } ^ { \tilde { M _ { u } } }$ for an input vector $\pmb { x } _ { i u } \in \mathbb { C } ^ { K _ { u } }$ . Outside the ONN, a loss function $\ell ( y _ { i } , t _ { i } ) \in \mathbb { R }$ with a target vector $\mathbf { \Delta } _ { t _ { i } }$ is defined to train the ONN.

A linear module is typically an array of MZIs. An MZI consists of two PS-BS pairs. One PS-BS pair serves as the transfer matrix

$$
{ \frac { 1 } { \sqrt { 2 } } } \left( { \begin{array} { c c } { 1 } & { j } \\ { j } & { 1 } \end{array} } \right) \left( e ^ { j \theta } \quad 0 \right)
$$

programmable by a phase parameter $\theta$ . The imaginary unit $j$ satisfies ${ j ^ { 2 } = - 1 }$ . Here, we assume an ideal situation without error, i.e., $\zeta = 1$ and $\gamma = 0$ . Then, an MZI represents a $2 \times 2$ unitary matrix $\mathbf { U }$ , which satisfies $\mathbf { U } \mathbf { U } ^ { \mathsf { H } } = \mathbf { I }$ with H being a conjugate transpose operator. For a nonlinear module, we employ modReLU (Arjovsky, Shah, and Bengio 2016; Maduranga, Helfrich, and Ye 2019; Williamson et al. 2020) with a bias parameter $\theta$ whose element-wise function is given by

Table 1: Notations   

<html><body><table><tr><td>S∈C xi∈CK Yi∈ CM 0ERN YER</td><td>Input vector to ONN Output vector of ONN ONN parameters splitting angle error attenuation-phase error</td></tr><tr><td>xiu ECKu Yiu ∈CMu 0u∈RNu Jiu ECMuXNu</td><td>Input vector to module u Output vector of module u Module parameters Jacobian matrix from 0u to Yiu</td></tr><tr><td>D={xi,ti=1 BCD l∈R</td><td>Trainingdataset Mini-batch dataset Loss function</td></tr><tr><td>80q∈CN 80uq ∈CNu</td><td>Perturbation vector for ZO optimization Perturbation for module u</td></tr><tr><td>∑u∈CNuXNu Cyiu F(i) ∈CNuXNu ∈CMuxMu</td><td>Covariance matrix for sampling δ0uq Fisherinformationmatrix Output covariance matrix</td></tr></table></body></html>

$$
{ \mathrm { m o d R e L U } } ( y ) = { \left\{ \begin{array} { l l } { y ( | y | + \theta ) / | y | } & { { \mathrm { i f ~ } } | y | + \theta \geq 0 } \\ { 0 } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

Figure 2 shows module examples used in NNs. The Clements mesh (Clements et al. 2016) is the most popular ONN module, whose $K _ { u } = 8$ case is illustrated as Clements(8,8). It realizes an arbitrary unitary matrix together with a diagonal matrix, illustrated as PSdiag(8), whose diagonal elements are of the form $e ^ { j \theta }$ . In this case, the connected modules Clements(8,8) $+ \mathsf { P }$ Sdiag(8) have $5 6 + 8 = 6 4$ parameters in total, which corresponds to the required and sufficient number of parameters for an arbitrary 8-dimensional unitary matrix. One can reduce the circuit size by employing a truncated version (Fang et al. 2019) of the Clements mesh, whose 4-layer case is illustrated as Clements(8,4). In this case, the number of parameters reduces to 28, while the degree of freedom for realizing a unitary matrix is also decreased. The Clements mesh and its truncated version are layered arrays of MZIs. Thus, the phase parameters in the module are interrelated. Examples will be shown in Sec. 3. On the other hand, the parameters of PSdiag and modReLU are not interrelated when viewed as a single module. A linear module of an ordinary NN has weight parameters $\textbf { \em w }$ that are also not interrelated.

# Training NN by Zeroth-Order Optimization

Given a training dataset $\mathcal { D } = \{ \boldsymbol { x } _ { i } , t _ { i } \} _ { i = 1 } ^ { D }$ of size $D$ , training an NN is to optimize the parameters $\pmb \theta$ so as to minimize the loss function

$$
\ell _ { \mathcal { D } } ( \pmb { \theta } ) = \frac { 1 } { D } \sum _ { i = 1 } ^ { D } \ell ( \pmb { f } ( \pmb { x } _ { i } , \pmb { \theta } ) , \pmb { t } _ { i } ) .
$$

A standard way to optimize the parameters $\pmb \theta$ is by minibatch stochastic gradient descent (SGD) (Bottou 2012;

![](images/8e8faad49064258c5401d2aa55506c76ccc34b72d8f468b1d379df6b184ac3b4.jpg)  
Figure 2: Module examples used in neural networks. The parameters of the upper modules are arranged in a layered structure in depth and therefore interrelated, while those of the lower modules are not.

Goodfellow, Bengio, and Courville 2016) in which we iterate the parameter updates

$$
\begin{array} { c } { \displaystyle \pmb { \theta } \gets \pmb { \theta } - \eta \cdot \nabla _ { \pmb { \theta } } \ell _ { B } ( \pmb { \theta } ) , } \\ { \ell _ { B } ( \pmb { \theta } ) = \displaystyle \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \ell ( \pmb { f } ( \pmb { x } _ { i } , \pmb { \theta } ) , \pmb { t } _ { i } ) , } \end{array}
$$

where $\mathcal { B } = \{ \boldsymbol { x } _ { i } , t _ { i } \} _ { i = 1 } ^ { B } \subset \mathcal { D }$ is a mini-batch dataset drawn from $\mathcal { D }$ , $\nabla _ { \theta }$ is the gradient with respect to $\pmb \theta$ typically calculated by backpropagation (Bishop 2006; LeCun, Bengio, and Hinton 2015), and $\eta > 0$ is a learning rate.

For training ONNs, we cannot accurately compute the gradient by backpropagation because of fabrication variations. Therefore, we use ZO optimization (Liu et al. 2020) as a black-box optimization method to approximately compute the gradient:

$$
\nabla _ { \pmb \theta } \ell _ { B } ( \pmb \theta ) \approx \hat { \nabla } _ { \pmb \theta } \ell _ { B } ( \pmb \theta ) = \frac { \lambda } { Q } \sum _ { q = 1 } ^ { Q } \delta \ell _ { q } \cdot \delta \pmb \theta _ { q } ,
$$

where $\lambda > 0$ is a scale hyperparameter, $\delta \pmb { \theta } _ { q }$ , $q = 1 , \ldots , Q$ are random perturbation vectors, and

$$
\delta \ell _ { q } = \frac { \ell _ { B } ( \pmb \theta + \mu \cdot \delta \pmb \theta _ { q } ) - \ell _ { B } ( \pmb \theta ) } { \mu } , q = 1 , \dots , Q
$$

are difference quotients by the finite difference method with a smoothing hyperparameter $\mu > 0$ .

Typically, the random perturbation vectors $\delta \pmb { \theta } _ { q }$ are sampled from a zero-mean normal distribution $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { N } )$ with the covariance matrix being an identity matrix of size $N$ , i.e., $\delta \pmb { \theta } _ { q } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { N } )$ . This means that the parameters $\pmb { \theta } _ { u }$ of a module $u$ are perturbed by

$$
\begin{array} { r } { \delta \pmb { \theta } _ { u q } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { N _ { u } } ) . } \end{array}
$$

In Sec. 3, we demonstrate by examples that an identity matrix $\mathbf { I } _ { N _ { u } }$ is not the best covariance matrix for a module $u$

(a) 01u= 02u=-π 03u=0 (c) (d) e-j (2.041 0.993 0.2821.760) 1.973 -0.020 -0.270 -1.665)   
xiu yiu 1. 8728:8 £u -0270-1.0407070.0 e 04u=0   
(b) 2 3 (e) 2 3 3 1 1 -2 -3 -2 -1 0 1 -2 -1 0 1 2 -2 -3 -2 -1 0 -1-2 -1 0 1 Re([yiul1) Im([yiul1) Re([yiul1) Im([yiu]1) Eigenvalues $\mathbf { \tau } = \mathbf { \tau }$ [0.242,0.503] Eigenvalues $\mathbf { \tau } = \mathbf { \tau }$ [0.231,0.260]

whose parameters $\pmb { \theta } _ { u }$ are interrelated due to its layered structure. We therefore instead perturb such parameters

$$
\delta \pmb { \theta } _ { u q } \sim \mathcal { N } ( \mathbf { 0 } , \pmb { \Sigma } _ { u } )
$$

with a covariance matrix $\Sigma _ { u }$ .

Let us explain (Rasmussen and Williams 2006) how to generate such samples $\delta \pmb { \theta } _ { u q }$ from $\Sigma _ { u }$ when we have solely a subroutine generating a random vector $\pmb { r } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ with an identity covariance matrix, i.e., $\mathbb { E } \left[ r r ^ { \mathsf { T } } \right] = \mathbf { I }$ with T being a transpose operator. We first compute the Cholesky decomposition $\pmb { \Sigma } _ { u } \overset { \cdot } { = } \mathbf { L } \mathbf { L } ^ { \mathsf { T } }$ , where $\mathbf { L }$ is a lower triangular matrix. We then compute what we need as $\delta \pmb { \theta } _ { u q } = \mathbf { L } \pmb { r }$ , which conforms to $\begin{array} { r } { \mathbb { E } _ { q } \left[ \delta \pmb { \theta } _ { u q } \delta \pmb { \theta } _ { u q } ^ { \top } \right] = \mathbf { L } \mathbb { E } \left[ \pmb { r } \pmb { r } ^ { \top } \right] \mathbf { L } ^ { \top } = \mathbf { L } \mathbf { L } ^ { \top } = \pmb { \Sigma } _ { u } . } \end{array}$ .

# Jacobian Matrix

To model how perturbations propagate from the parameters $\pmb { \theta } _ { u }$ to the output vector $\pmb { y } _ { i u }$ on a module $u$ for a specific input vector $\pmb { x } _ { i u }$ , let us introduce a Jacobian matrix as

$$
\mathbf { J } _ { i u } = \frac { \partial y _ { i u } } { \partial \pmb { \theta } _ { u } } = \left( \begin{array} { c c c } { \frac { \partial [ y _ { i u } ] _ { 1 } } { \partial [ \pmb { \theta } _ { u } ] _ { 1 } } } & { \cdot \cdot \cdot } & { \frac { \partial [ \pmb { y } _ { i u } ] _ { 1 } } { \partial [ \pmb { \theta } _ { u } ] _ { N _ { u } } } } \\ { \vdots } & { \ddots } & { \vdots } \\ { \frac { \partial [ \pmb { y } _ { i u } ] _ { M _ { u } } } { \partial [ \pmb { \theta } _ { u } ] _ { 1 } } } & { \cdot \cdot \cdot } & { \frac { \partial [ \pmb { y } _ { i u } ] _ { M _ { u } } } { \partial [ \pmb { \theta } _ { u } ] _ { N _ { u } } } } \end{array} \right) \in \mathbb { C } ^ { M _ { u } \times N _ { u } } .
$$

Then, the output perturbation $\delta { \bf { y } } _ { i u q }$ caused by a parameter perturbation $\delta \pmb { \theta } _ { u q }$ can be expressed as a Jacobian-vector product

$$
\delta { \bf y } _ { i u q } = { \bf J } _ { i u } \delta { \bf \pmb { \theta } } _ { u q } ,
$$

which can be computed by forward mode automatic differentiation (AD) (Baydin et al. 2018). Conversely, the parameter perturbation $\delta \pmb { \theta } _ { u p } ^ { ( i ) }$ caused by an output perturbation $\delta { \bf { y } } _ { i u p }$ (here we use index $p$ to distinguish it from the above using $q$ ) can be expressed as a vector-Jacobian product (the following form is transposed ${ } ^ { \mathsf { T } }$ for column vectors)

$$
\begin{array} { r } { \delta \pmb { \theta } _ { u p } ^ { ( i ) } = \mathbf { J } _ { i u } ^ { \top } \delta \pmb { y } _ { i u p } , } \end{array}
$$

which can be computed by backpropagation, or reverse mode AD in this context. When we perform forward and reverse mode ADs, we have no choice but to assume an ideal situation without errors, i.e., $\zeta = 1$ and $\gamma = 0$ , because both need the precise information inside the module $u$ .

# 3 Motivating Examples

This section shows some examples on what happens if we use (8) for parameter perturbations, and how the situation improves by using layered-parameter perturbations (9).

# 2-Dimensional Case

Figure 3 shows a simple 2-dimensional case where we can inspect the situation in details. Panel (a) depicts a linear module $u$ that represents a 2-dimensional unitary matrix with 4 phase parameters $\pmb { \theta } _ { u } = [ \frac { \pi } { 4 } , - \frac { \pi } { 4 } , 0 , 0 ] ^ { \mathsf { T } }$ . For an input vector $\underline { { { \pmb x } } } _ { i u } ~ = ~ [ e ^ { - j \frac { \pi } { 4 } } , e ^ { j \frac { \pi } { 2 } } ] ^ { \mathsf { T } }$ , the module produces $\mathsf { \pmb { y } } _ { i u } = [ - 1 , j ] ^ { \mathsf { T } }$ at the output. Panels (b) and (e) represent the output’s 2-dimensional complex spaces by the plots of the real parts $\mathrm { R e } ( [ { \pmb y } _ { i u } ] _ { 1 } ) , \mathrm { R e } ( [ { \pmb y } _ { i u } ] _ { 2 } )$ and the imaginary parts $\mathrm { I m } ( [ { \pmb y } _ { i u } ] _ { 1 } ) , \mathrm { I m } ( [ { \pmb y } _ { i u } ] _ { 2 } )$ . In both panels, the blue squares represent the output values $[ - 1 , j ] ^ { \dagger }$ with no perturbation, i.e., with the current parameters $\pmb { \theta } _ { u }$ , and the red stars represent the output values with the optimal parameters that we aim at. The light blue dots in panel (b) show the output perturbations (11) caused by typical parameter perturbations $\delta \pmb { \theta } _ { u q }$ , $q = 1 , \ldots , Q$ with $Q \ = \ 1 0 0 0$ , sampled from a distribution $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { 4 } )$ with a 4-dimensional identity matrix. We observe that the output perturbations $\delta { \pmb y } _ { i u q }$ are not isotropic, i.e., strong in some specific directions and weak in others.

The reason comes from the fact that the parameters are interrelated by the layered structure shown in panel (a). One way to measure the degree of interrelation is to calculate the Fisher information matrix (Martens 2020) for the input $\pmb { x } _ { i u }$ with respect to the parameters $\pmb { \theta } _ { u }$ . In our context, it is computed as the covariance matrix

$$
\mathbf { F } _ { u } ^ { ( i ) } = \mathbb { E } _ { p } \left[ \delta \pmb { \theta } _ { u p } ^ { ( i ) } \delta \pmb { \theta } _ { u p } ^ { ( i ) \top } \right] \approx \frac { 1 } { R _ { \mathrm { o u t } } } \sum _ { p = 1 } ^ { R _ { \mathrm { o u t } } } \delta \pmb { \theta } _ { u p } ^ { ( i ) } \delta \pmb { \theta } _ { u p } ^ { ( i ) \top }
$$

caused by newly generated isotropic perturbations

$$
\delta { \pmb y } _ { i u p } \sim \mathcal { N } ( { \bf 0 } , { \bf I } _ { M _ { u } } ) , p = 1 , \dots , R _ { \mathrm { o u t } }
$$

at the output $\pmb { y } _ { i u }$ . Here, $\delta \pmb { \theta } _ { u p } ^ { ( i ) }$ are the caused perturbations at the parameters $\pmb { \theta } _ { u }$ , and can be computed by backpropagating (12) the generated output perturbation vectors $\delta \pmb { y } _ { i u p }$ . Panel (c) shows the average of such matrices $\begin{array} { r } { \frac { 1 } { R _ { \mathrm { i n } } } \sum _ { i = 1 } ^ { R _ { \mathrm { i n } } } \mathbf { F } _ { u } ^ { ( i ) } } \end{array}$ over $R _ { \mathrm { i n } } = 1 0 0 0$ random input vectors $\pmb { x } _ { i u }$ and $R _ { \mathrm { o u t } } = 1 0$ output perturbations. It represents that all parameter pairs except $( \theta _ { 3 u } , \theta _ { 4 u } )$ , which are not connected by an optical path as shown in panel (a), are interrelated by non-negligible values.

This paper proposes to use parameter perturbations sampled from a distribution $\mathcal { N } ( \mathbf { 0 } , \pmb { \Sigma } _ { u } )$ with a covariance matrix $\Sigma _ { u }$ computed by the method explained in Sec. 4. In this specific case, $\Sigma _ { u }$ is given by panel (d), which apparently differs from $\mathbf { I } _ { 4 }$ . The light blue dots in panel (e) show the output perturbations $\delta \pmb { y } _ { i u q }$ , $q = 1 , \ldots , Q$ caused by the parameter perturbations $\delta \pmb { \theta } _ { u q }$ sampled from $\mathcal { N } ( \mathbf { 0 } , \pmb { \Sigma } _ { u } )$ . We observe that these in panel (e) are more isotropic than those in panel (b). Since we aim at the optimal parameters that produce the red stars at the output by ZO optimization, the output perturbations in panel (e) are better at easily approaching the red stars than those in panel (b). Note that complete isotropy for each $i$ -th output $\pmb { y } _ { i u }$ is difficult because the covariance matrix $\Sigma _ { u }$ is computed from the average of $\mathbf { F } _ { u } ^ { ( i ) }$ . One way to grasp how good or bad the output perturbations are is by inspecting the output covariance matrix

$$
\mathbf { C } _ { \pmb { y } _ { i u } } = \mathbb { E } _ { \pmb { q } } \left[ \delta \pmb { y } _ { i u q } \delta \pmb { y } _ { i u q } ^ { \mathrm { H } } \right] \approx \frac { 1 } { Q } \sum _ { q = 1 } ^ { Q } \delta \pmb { y } _ { i u q } \delta \pmb { y } _ { i u q } ^ { \mathrm { H } } ,
$$

whose eigenvalues especially tell us the degree of isotropic. The more similar the eigenvalues are, the more isotropic the output perturbations are. The eigenvalues in this example are shown at the bottom of panels (b) and (e).

# Clements Mesh and Its Truncated Version

Here, we examine the situations of Clements(8,8) and Clements(8,4), more practical modules than that of the previous subsection. In the left part of Fig. 4, the first column shows the averaged Fisher information matrices $\begin{array} { r } { \frac { 1 } { R _ { \mathrm { i n } } } \sum _ { i = 1 } ^ { R _ { \mathrm { i n } } } \mathbf { F } _ { u } ^ { ( i ) } } \end{array}$ over $R _ { \mathrm { { i n } } } ~ = ~ 1 0 0$ random input vectors $\pmb { x } _ { i u }$ and $R _ { \mathrm { o u t } } ~ = ~ 1 0$ output perturbations. These two matrices correspond to panel (c) of Fig. 3. We observe that the parameters are interrelated as these matrices contain many off-diagonal non-negligible values. They are caused by the Clements(8,8) and Clements(8,4) structures depicted in Fig. 2. The second column shows the computed covariance matrices $\Sigma _ { u }$ for parameter perturbations, which correspond to panel (d) of Fig. 3. We observe that these matrices apparently differ from identity matrices $\mathbf { I } _ { 5 6 }$ and $\mathbf { I } _ { 2 8 }$ .

The right hand side of Fig. 4 shows the eigenvalue distributions of the output covariance matrices ${ { \bf { C } } _ { { \bf { y } } _ { i u } } } , i { \mathrm { ~  ~ \Gamma ~ } } =$

![](images/dd93734f7d31b25360476bf49a19d23f2957c48a2286d1426e2cdeb703a62054.jpg)  
Figure 4: Clements(8,8) and Clements(8,4) examples. Left: averaged Fisher information matrices $\begin{array} { r } { \frac { 1 } { R _ { \mathrm { i n } } } \sum _ { i = 1 } ^ { R _ { \mathrm { i n } } } \mathbf { \bar { F } } _ { u } ^ { ( i ) } } \end{array}$ and covariance matrices $\Sigma _ { u }$ . Right: eigenvalue distributions of output covariance matrices $\mathbf { C } _ { y _ { i u } }$ , $i = 1 , \ldots , R _ { \mathrm { i n } }$ to show how isotropic or not the output perturbations are.

$1 , \ldots , R _ { \mathrm { i n } }$ with $Q = 1 0 0$ parameter perturbations. The vertical bars represent one standard deviation over the $R _ { \mathrm { i n } } ~ { = } ~$ 100 different input vectors. We observe that there are many large eigenvalues when we use the identity matrices $\mathbf { I } _ { 5 6 }$ and $\mathbf { I } _ { 2 8 }$ for perturbing the parameters. This means that the output perturbations are directional like panel (b) of Fig. 3. Contrary, if we use the covariance matrices $\Sigma _ { u }$ for perturbing the parameters, the eigenvalues become similar, meaning that the output perturbations are close to isotropic like panel (e) of Fig. 3.

# 4 Proposed Method

Our proposed method improves ZO optimization for ONN training. The core of the improvement is to perturb module parameters $\pmb { \theta } _ { u }$ as $\pmb { \theta } _ { u } + \pmb { \mu } \cdot \delta \pmb { \theta } _ { u q }$ by (9) with $\Sigma _ { u }$ instead of (8) with $\mathbf { I } _ { N _ { u } }$ for modules $u$ with layered parameters, such as those in the upper part of Fig. 2.

# Covariance Matrix $\Sigma _ { u }$ for Perturbations

We design the covariance matrix $\Sigma _ { u }$ in (9) so that it minimize a cost function

$$
\mathcal { C } \big ( \boldsymbol { \Sigma } _ { u } \big ) = \frac { 1 } { R _ { \mathrm { i n } } } \sum _ { i = 1 } ^ { R _ { \mathrm { i n } } } \mathcal { L } _ { M _ { u } } \big ( \mathbf { C } _ { \boldsymbol { y } _ { i u } } , \mathbf { I } _ { M _ { u } } \big ) + \rho \mathbf { \cdot } \mathcal { L } _ { N _ { u } } \big ( \boldsymbol { \Sigma } _ { u } , \mathbf { I } _ { N _ { u } } \big ) ,
$$

where $\mathcal { L } _ { M (  M _ { u } , N _ { u } ) }$ is the LogDet (Davis et al. 2007) (or Burg matrix (Davis and Dhillon 2006) or multichannel Itakura-Saito (Sawada et al. 2013)) divergence for matrices of size $M \times M$ :

$$
\mathscr { L } _ { M } ( { \mathbf { A } } , { \mathbf { B } } ) = \operatorname { t r a c e } ( { \mathbf { A } } { \mathbf { B } } ^ { - 1 } ) - \log \operatorname* { d e t } ( { \mathbf { A } } { \mathbf { B } } ^ { - 1 } ) - M .
$$

The LogDet divergence measures how much two covariance matrices differ, and becomes zero if they are the same.

The first term of (16) is motivated by the examples and discussions in Sec. 3. They suggest to make the output covariance matrices $\mathbf { C } _ { y _ { i u } }$ close to an identity matrix, whose eigenvalues are the same, for various input vectors $\pmb { x } _ { i u }$ , $i = 1 , \ldots , R _ { \mathrm { i n } }$ . Consequently, the output perturbations become as isotropic as possible, as shown in panel (e) of Fig. 3.

The second term with a hyperparameter $\rho > 0$ regularizes to prevent $\Sigma _ { u }$ from being singular by not making it far away from an identity matrix $\mathbf { I } _ { N _ { u } }$ . The hyperparameter $\rho$ should not be very large because $\Sigma _ { u }$ should be different from $\mathbf { I } _ { N _ { u } }$ according to our aim that replaces (8) with (9).

As a consequence, the covariance matrix $\Sigma _ { u }$ is given by

$$
\boldsymbol { \Sigma } _ { u } = ( 1 + \rho ) \left( \frac { 1 } { R _ { \mathrm { i n } } } \sum _ { i = 1 } ^ { R _ { \mathrm { i n } } } \mathbf { F } _ { u } ^ { ( i ) } + \rho \cdot \mathbf { I } _ { N _ { u } } \right) ^ { - 1 } ,
$$

with the Fisher information matrices $\mathbf { F } _ { u } ^ { ( i ) }$ averaged over $R _ { \mathrm { i n } }$ random input vectors $\pmb { x } _ { i u }$ . How to derive (18) from (16) will be explained in the next subsection. As already explained when we introduced $\mathbf { F } _ { u } ^ { \left( i \right) }$ in (13), $\mathbf { F } _ { u } ^ { ( i ) }$ is computed first by sampling output perturbations $\delta { \bf y } _ { i u p } \sim \mathcal { N } ( \mathbf { 0 } , \bar { \mathbf { I } } _ { M _ { u } } )$ , $p = 1 , \ldots , R _ { \mathrm { o u t } }$ as (14), then by backpropagating them up to parameters $\delta \pmb { \theta } _ { u p } ^ { ( i ) }$ as (12), and finally averaging their outer products over $p = \mathrm { { i } } , \ldots , R _ { \mathrm { { o u t } } }$ as (13).

# Derivation of (18)

Let us first rewrite the two equations in Sec. 3. The first one (13) can be written as

$$
\mathbf { F } _ { u } ^ { ( i ) } = \mathbf { J } _ { i u } ^ { \top } \mathbb { E } _ { p } \left[ \delta \pmb { y } _ { i u p } \delta \pmb { y } _ { i u p } ^ { \sf H } \right] \mathbf { J } _ { i u } ^ { * } = \mathbf { J } _ { i u } ^ { \top } \mathbf { J } _ { i u } ^ { * } ,
$$

where ∗ is the element-wise conjugate operator. Here, we use (12) and Ep δyiupδyiHup = IMu because of δyiup ∼ $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { M _ { u } } )$ . The second one (15) can be written as

$$
\begin{array} { r } { \mathbf { C } _ { { y } _ { i u } } = \mathbf { J } _ { i u } \mathbb { E } _ { q } \left[ \delta \pmb { \theta } _ { u q } \delta \pmb { \theta } _ { u q } ^ { \top } \right] \mathbf { J } _ { i u } ^ { \sf H } = \mathbf { J } _ { i u } \sum _ { u } \mathbf { J } _ { i u } ^ { \sf H } , } \end{array}
$$

where we use (11) and $\mathbb { E } _ { q } \left[ \delta { \pmb { \theta } } _ { u q } \delta { \pmb { \theta } } _ { u q } ^ { \top } \right] = \pmb { \Sigma } _ { u }$ because of $\delta \pmb { \theta } _ { u q } \sim \mathcal { N } ( \mathbf { 0 } , \pmb { \Sigma } _ { u } )$ .

According to the definition of LogDet divergence (17) and the new form (20) of the output covariance matrix, the cost function (16) with ignoring constant terms can be written as

$$
\begin{array} { r l r } {  { \mathcal { C } \big ( \Sigma _ { u } \big ) \stackrel { c } { = } \frac { 1 } { R _ { \mathrm { i n } } } \sum _ { i = 1 } ^ { R _ { \mathrm { i n } } } [ \mathrm { t r a c e } ( \mathbf { J } _ { i u } \Sigma _ { u } \mathbf { J } _ { i u } ^ { \mathsf { H } } ) - \log \operatorname* { d e t } ( \Sigma _ { u } ) ] } } \\ & { } & { + \rho \cdot [ \mathrm { t r a c e } \big ( \Sigma _ { u } \big ) - \log \operatorname* { d e t } \big ( \Sigma _ { u } \big ) ] . \quad ( 2 1 \rho \mathsf { H } ^ { \Sigma } ) } \end{array}
$$

The complex gradient matrix (Petersen and Pedersen 2008) with respect to $\Sigma _ { u }$ is given by

$$
\begin{array} { r l r } { \displaystyle \frac { \partial \mathcal { C } ( \Sigma _ { u } ) } { \partial \Sigma _ { u } ^ { * } } } & { = } & { \displaystyle \frac { 1 } { R _ { \mathrm { i n } } } \sum _ { i = 1 } ^ { R _ { \mathrm { i n } } } \left[ \mathbf { J } _ { i u } ^ { \top } \mathbf { J } _ { i u } ^ { * } - \Sigma _ { u } ^ { - 1 } \right] + \boldsymbol { \rho } \cdot \left[ \mathbf { I } _ { N _ { u } } - \Sigma _ { u } ^ { - 1 } \right] } \\ & { = } & { \displaystyle \frac { 1 } { R _ { \mathrm { i n } } } \sum _ { i = 1 } ^ { R _ { \mathrm { i n } } } \mathbf { F } _ { u } ^ { ( i ) } + \boldsymbol { \rho } \cdot \mathbf { I } _ { N _ { u } } - ( 1 + \boldsymbol { \rho } ) \cdot \Sigma _ { u } ^ { - 1 } , ( 2 \boldsymbol { 2 } ) } \end{array}
$$

where we use (19). Setting (22) to a zero matrix gives the covariance matrix (18) as the minimizer of (16).

# Overall Procedure of ZO Optimization

Algorithm 1 shows the overall procedure of ZO optimization that generates layered-parameter perturbations. Lines 2, 3,

Require: training dataset $\mathbf { \mathcal { D } } = \{ \mathbf { x } _ { i } , t _ { i } \} _ { i = 1 } ^ { D }$   
Ensure: parameters $\pmb \theta$   
1: while not converged, in $\tau$ -th iteration do   
2: Sample mini-batch $\boldsymbol { B }$ from $\mathcal { D }$   
3: Evaluate loss $\ell _ { B } ( { \pmb \theta } )$ (5)   
4: for all module $u$ with layered parameters do   
5: if $\tau$ mod $T _ { \mathrm { u d } } = 0$ then   
6: Update ${ { \bf { F } } _ { u } }$ by (24), (13), (12), and (14)   
7: Update $\Sigma _ { u }$ by (23)   
8: end if   
9: Sample perturbations $\delta \pmb { \theta } _ { u q }$ , $q = 1 , . . . , Q$ by (9)   
10: end for   
11: for all module $u$ with non-layered parameters do   
12: Sample perturbations $\delta \pmb { \theta } _ { u q }$ , $q = 1 , . . . , Q$ by (8)   
13: end for   
14: Concatenate module perturbations $\delta \pmb { \theta } _ { u q }$ into the   
whole perturbation vectors $\delta \pmb { \theta } _ { q }$ , $q = 1 , \ldots , Q$   
15: Get $\delta \ell _ { q }$ , $q = 1 , \ldots , Q$ by (7)   
16: Calculate the ZO gradient estimate (6)   
17: Update parameters $\pmb \theta$ (4)   
18: end while

14-17 concerns the basic procedure of ZO optimization explained in Sec. 2. Lines from 4 to 10 are specific to our new proposal. Line 9 is the key operation of our proposal, which samples perturbations $\delta \pmb { \theta } _ { u q }$ for a module $u$ with layered parameters by (9) as $\delta \pmb { \theta } _ { u q } \sim \mathcal { N } ( \mathbf { 0 } , \pmb { \Sigma } _ { u } )$ . For a module $u$ with non-layered parameters, we sample perturbations $\delta \pmb { \theta } _ { u q }$ by (8) as with the ordinary ZO optimization (line 12). The procedure reduces to the ordinary one if we skip lines from 4 to 10 and apply line 12 for all the modules $u$ .

In the iterations (lines 1 through 18), exponential smoothing (Gardner Jr 2006) leads to robust estimations of some statistics. In our particular case, we modify (18) to

$$
\pmb { \Sigma } _ { u } = \left( 1 + \rho \right) \left( \mathbf { F } _ { u } + \boldsymbol { \rho } \cdot \mathbf { I } _ { N _ { u } } \right) ^ { - 1 } ,
$$

$$
\mathbf { F } _ { u } \gets \alpha \cdot \frac { 1 } { R _ { \mathrm { i n } } } \sum _ { i = 1 } ^ { R _ { \mathrm { i n } } } \mathbf { F } _ { u } ^ { ( i ) } + ( 1 - \alpha ) \cdot \mathbf { F } _ { u } ,
$$

where ${ { \bf { F } } _ { u } }$ is initially set as an identity matrix and $0 \leq \alpha \leq 1$ is a smoothing hyperparameter. In addition, we have empirically confirmed that the performance of ZO optimization is not degraded much even if we reduce the update frequency of these statistics. Therefore, we introduce another hyperparameter $T _ { \mathrm { u d } }$ (line 5) to reduce the computational burden.

# 5 Experiments Compared Methods

Four methods “ZO-I”, “ZO-co”, “ZO- $\Sigma _ { u }$ ” and “CMA” are compared. The first three are ZO optimization based on an identity matrix $\mathbf { I }$ as (8), coordinate-wise perturbations where $\delta \pmb { \theta } _ { q }$ is a one-hot vector (only one element is 1 and all other elements are 0), and proposed special covariance matrices $\Sigma _ { u }$ as (9). “CMA” is CMA-ES (Hansen, Akimoto, and Baudis 2019), which also computes special covariance matrices, but in a fundamentally different way.

Table 2: Hyperparameter settings   

<html><body><table><tr><td>B=100 Q=K 入=1/N μ= 0.001/√N</td><td>Mini-batch size Number of perturbation vectors Scale (ZO optimization) Smoothing (ZO optimization)</td><td>Eq.(5) Eq.(6) Eq.(6) Eq.(7)</td></tr><tr><td>α=0.01 p=0.1 Tud = 100 Rin = 100 Rout =100 Num.of output perturbation vectors</td><td>Exponential smoothing ofFu Regularizing weight for Σu Update interval of Fu and Σu Number of random input vectors</td><td>Eq.(24) Eq.(23) Alg.(1) Eq.(24)</td></tr></table></body></html>

The parameters of layered modules Clements(K,L) were randomly initialized, while those of non-layered modules, e.g., PSdiag(K) and modReLU(K), were initialized to zero. The training procedure consisted of two steps. In the first step, we used backpropagation with error-free assumptions, i.e., $\zeta = 1$ and $\gamma = 0$ , for a small number of 10 epochs. The intention of the first step was to quickly and roughly minimize the loss function from the initial parameters, even though the calculated gradients were inaccurate due to the wrong error assumption. In the second step, we ran some of the four methods, taking into account the actual error situation, with a sufficient number of 100 epochs.

For better convergence of the ZO optimization algorithm, we used the Adam optimizer (Kingma and Ba 2014) instead of vanilla SGD (4). The learning rate $\eta$ of Adam and the step-size sigma0 of CMA-ES were optimized by using Optuna (Akiba et al. 2019) for each combination of task, dimensionality $( K = 1 6 , 3 2 , 6 4$ ), and method. Table 2 shows the default settings of other hyperparameters. The upper four settings were common to the three ZO methods. The lower five settings were specific to the proposed method $ { \mathrm {  ~ \tilde { ~ } { ~ Z } O - } }  { \Sigma _ { u } } ^ { \mathrm { \tiny ~ , ~ } }$ .

# Experimental Settings

We simulated ONN circuits with fabrication variations in an ordinary computer. For this purpose, we set the splitting angle errors $\gamma$ and the attenuation-phase errors $\zeta$ (see Fig. 1) randomly by

$$
\gamma = \sigma _ { \gamma } \cdot r _ { 0 } , \zeta = \left( 1 - \sigma _ { \zeta , r } r _ { 1 } \right) \exp [ j \cdot \sigma _ { \zeta , a } ( 2 r _ { 2 } { - } 1 ) ] ,
$$

where $r _ { 0 }$ is a random number sampled from a normal distribution $\mathcal { N } ( 0 , 1 )$ , $r _ { 1 }$ and $r _ { 2 }$ are ones sampled from a uniform distribution $[ 0 , 1 )$ . We examined the following error settings:

$$
\sigma _ { \gamma } = 1 0 ^ { - 2 } \beta , \sigma _ { \zeta , r } = 1 0 ^ { - 3 } \beta , \sigma _ { \zeta , a } = 1 0 ^ { - 1 } \beta
$$

where $\beta$ controlled the amount of errors. The setting $\beta =$ 1 corresponded to our estimate on an actual calibrated Clements mesh on silicon photonics. We set $\beta = 1$ unless otherwise noted.

We employed PyTorch (Paszke et al. 2019) to simulate and train ONNs. We built our customized CUDA kernels (Luebke 2008) for computational acceleration, and ran the program on an NVIDIA RTX A6000 (48 GB) as the GPU.

9 K 1 ONN xi yi

# Image Classification Task

Figure 5 shows our configuration for image classification tasks. We used the MNIST (LeCun and Cortes 2010) and FashionMNIST (Xiao, Rasul, and Vollgraf 2017) as image datasets, which were of appropriate difficulties for ONN circuits without memory. We first applied the discrete Fourier transform to a $7 8 4 = 2 8 \times 2 8$ pixel image and took the frequency bins from the second lowest to the $K + 1$ lowest (discarding the $0 ~ \mathrm { H z }$ ) as an input vector $\pmb { x } _ { i }$ . Then, $\mathbf { \boldsymbol { x } } _ { i }$ was fed into the ONN to produce the output vector $\mathbf { \mu } _ { \mathbf { \mathcal { Y } } _ { i } }$ , whose central 10 dimensions were extracted. Finally, the powers of the extracted elements were calculated to produce a 10- dimensional real-valued vector indicating the image classification result. As in many related studies, test accuracy and training loss are used as evaluation metrics.

Table 3 shows the test accuracies over eight independent runs with settings $L = K$ . In addition to the four methods compared, the row labeled “BP w error info” shows the results by backpropagation with perfect error information, which are unrealistic but can be considered as upper bounds. We observe that the proposed method $\widetilde { \mathbf { \Gamma } } \mathbf { Z } \mathbf { O } \mathbf { - } \mathbf { \sum } _ { u } \mathbf { \Gamma } ,$ ” generally outperformed the conventional methods (italic values indicate that they are statistically different from the corresponding best bold values according to the Mann-Whitney U test at a significance level of $p { = } 0 . 0 5 \$ ). We consider that “ZO-I” suffered from the interrelation effect discussed in this paper. Also that “ZO-co” avoided the interrelation effect, but could not train (or change) parameters efficiently because its ZO gradient estimates had only $Q$ non-zero elements. “CMA” did not scale well as one with $K { = } 6 4$ did not complete a run even in one day. Let us examine the three ZO methods in more detail below.

<html><body><table><tr><td>method</td><td>MNIST K=16 K=32 K=64</td><td>FashionMNIST K=16 K=32</td></tr><tr><td>ZO-I</td><td>79.18% 91.95% 94.37% ±0.43 ±0.19 ±0.09</td><td>K=64 64.73% 76.73% 81.51% ±0.34 ±0.39 ±0.15</td></tr><tr><td>ZO-co</td><td>79.37% 92.17%94.43% ±0.21 ±0.16 ±0.15</td><td>64.89% 77.12% 81.48% ±0.34 ±0.26 ±0.18</td></tr><tr><td>ZO-£u</td><td>79.60% 92.29% 94.70% ±0.32 ±0.21 ±0.14</td><td>65.34% 77.43% 81.87% ±0.29 ±0.18 ±0.22</td></tr><tr><td>CMA</td><td>78.88% 91.92% 1 ±0.33 ±0.22</td><td>64.69% 76.62% ±0.35 ±0.35 1</td></tr><tr><td>BP w error info</td><td>80.27% 93.96% 95.91% ±0.65 ±0.11 ±0.20</td><td>66.69% 79.01% 83.74% ±0.64 ±0.75 ±0.41</td></tr></table></body></html>

Table 3: Test accuracies $\ @ \mathrm { e p o c h } = 1 0 0$ . Mean $\pm$ standard deviation of eight runs. The best mean values among the four methods compared are shown in bold.

MNIST FashionMNIST\*\*\* ZO-I \*\*\* ZO-Ins ZO-co \* ZO-co  
1 Z0-Eu 1.10 \* ZO-Eu\*\* I \*\*\* \*\*\*1.08广 ns ns 站 \* \*： 工 1 宝0.62 T 1.00 白8 12 16 8 12 16\*\*\* \*\*\*\* \*  
2 0.68 1I I \*\*\*\*\*\*中 共 \* 1□ 中 1 T0.62中0.26? 0.6016 24 32 16 24 32\*\*\* \*\*\*0.52 1\*\*\* 水\* 卓 理 \*\*\*1 广 I 广 点 0.18 ： 中 0.4632 48 64 32 48 64L L

![](images/d4fe28a86078a9cce7b063554f6541c022fb412448820ca59ce4d8bd0c5a0d78.jpg)  
Figure 6: Training losses for image classification tasks. The $p$ -value annotations are based on the Mann-Whitney U test. The legends are: $\star \star \star$ for $1 0 ^ { - 4 } < p \le 1 0 ^ { - 3 }$ , $\star \star$ for $\mathrm { \dot { 1 } 0 ^ { - 3 } < }$ $p \leq 1 0 ^ { - 2 }$ , \* for $1 0 ^ { - 2 } < p \le 0 . 0 5$ , and ns for $0 . 0 5 < p \leq 1$ .   
Figure 7: Convergence behaviors for $K = L = 6 4$ MNIST

The box plots in Fig. 6 show the distributions of the training losses with $p$ -value annotations by the Mann-Whitney U test. We performed eight independent runs for each combination of dataset, method, dimensionality $K$ and the number $L$ of layers of Clements(K,L). We observe that “ZO-co” outperformed “ZO-I” and then the proposed method “ZO$\textstyle \Sigma _ { u } ^ { \phantom { + } }$ ” outperformed the winner $^ { \mathrm { 6 6 } } Z O  – \mathbf { c o } ^ { \prime \prime }$ among the conventional methods. These were with statistical significance in many cases except for the four ns cases in the MNIST plots. Most notably, the proposed method enabled some truncated Clements meshes such as Clements $( k = 6 4 , L = 4 8 )$ to outperform the full Clements meshes Clements(K,K) trained by the conventional methods, leading to significant circuit size savings without sacrificing classification performance.

Figure 7 shows some examples of convergence behavior as a function of the elapsed time. Although the proposed method $ { \mathrm { ~  ~ \omega ~ } } ^ {  { \mathrm { ~ \tiny ~ 6 ~ } } } Z ( 0 - \Sigma _ { u } ) ,$ apparently had a computational overhead over the other two ZO methods to complete the 100 epochs, the overhead was worth paying because the proposed method outperformed the other two methods most of the time, even with the same amount of elapsed time.

# 6 Conclusion

To better train MZI-based ONNs by ZO optimization in a black-box manner, we have discussed a new notion layeredparameter perturbation with some motivating examples. We have then proposed to perturb the parameters $\pmb { \theta } _ { u }$ of a layered module $u$ using a special covariance matrix $\Sigma _ { u }$ computed by our novel method. Experimental results show that the proposed method using $\Sigma _ { u }$ significantly outperformed the conventional methods. The more layers the modules had, the greater the advantage of the proposed method. The computational overhead of the proposed method was not very large in the experiments and was worth paying for the improvement of the ZO optimization.

As the first proposal of layered-parameter perturbation, this paper demonstrated its effectiveness with a simple, small-scale feed-forward network. To achieve higher accuracy in the image classification task, optical convolution operations (Feldmann et al. 2021; Wirth-Singh et al. 2024; Zheng et al. 2024; Liu et al. 2024a) or diffractive optical neural networks (Lin et al. 2018) could be used. In such situations, error compensation networks (Mengu et al. 2020; Liu et al. 2024b) can be implemented with an MZI-based ONN, whose parameters can be efficiently trained by the proposed method. Future work also includes verifying the proposed method on an actual silicon photonic device.