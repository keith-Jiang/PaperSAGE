# A Theoretical Framework for an Efficient Normalizing Flow-Based Solution to the Electronic Schro¨dinger Equation

Daniel Freedman1, Eyal Rozenberg2, Alex Bronstein2

1Independent Researcher 2Technion - Israel Institute of Technology

# Abstract

A central problem in quantum mechanics involves solving the Electronic Schr¨odinger Equation for a molecule or material. The Variational Monte Carlo approach to this problem approximates a particular variational objective via sampling, and then optimizes this approximated objective over a chosen parameterized family of wavefunctions, known as the ansatz. Recently neural networks have been used as the ansatz, with accompanying success. However, sampling from such wavefunctions has required the use of a Markov Chain Monte Carlo approach, which is inherently inefficient. In this work, we propose a solution to this problem via an ansatz which is cheap to sample from, yet satisfies the requisite quantum mechanical properties. We prove that a normalizing flow using the following two essential ingredients satisfies our requirements: (a) a base distribution which is constructed from Determinantal Point Processes; (b) flow layers which are equivariant to a particular subgroup of the permutation group. We then show how to construct both continuous and discrete normalizing flows which satisfy the requisite equivariance. We further demonstrate the manner in which the non-smooth nature (“cusps”) of the wavefunction may be captured, and how the framework may be generalized to provide induction across multiple molecules. The resulting theoretical framework entails an efficient approach to solving the Electronic Schr¨odinger Equation.

Extended version — https://arxiv.org/abs/2406.00047

# 1 Introduction

The Electronic Schr¨odinger Equation A central problem in quantum mechanics involves solving the Electronic Schr¨odinger Equation to compute the ground state energy and wavefunction of a molecule or material. This problem has manifold applications in chemistry, condensed matter physics, and materials science. However, solving the Electronic Schr¨odinger Equation can be tricky: analytical solutions only exist for certain very simple problems; and numerical solutions must cope with the curse of dimensionality, due to the fact that the wavefunction’s dimensionality grows linearly with the number of electrons.

Variational Monte Carlo A standard computational approach to this problem is based on Variational Monte Carlo (Ceperley and Alder 1986; Austin, Zubarev, and Lester Jr

2012; Gubernatis, Kawashima, and Werner 2016; Foulkes et al. 2001; Needs et al. 2009). Solving for the ground state is equivalent to finding the eigenfunction of the Hamiltonian corresponding to the smallest eigenvalue (energy); this problem can be posed as the minimization of a Rayleigh quotient. The issue is that both the numerator and denominator of the Rayleigh quotient correspond to very high dimensional integrals. Variational Monte Carlo approximates these integrals via sampling, and the approximated objective is optimized over a family of wavefunctions, yielding an upper bound on the ground state energy. The heart of this method is therefore the wavefunction family, also known as the ansatz: the more expressive the ansatz, the tighter the upper bound will be, yielding better approximations to the ground state energy.

Neural Networks as the Ansatz Recent work has proposed using neural networks as a flexible ansatz, and has achieved very high quality results. The network must be adapted to respect the properties of electronic wavefunctions – particularly antisymmetry, as electrons are Fermions. Important examples of this type of ansatz are PauliNet (Hermann, Sch¨atzle, and No´e 2020; Sch¨atzle, Hermann, and No´e 2021) and FermiNet (Pfau et al. 2020; Spencer et al. 2020), both of which attain excellent ground state energies (e.g. FermiNet achieves $9 9 . 8 \%$ of the correlation energy for boron atoms).

The Problem: Sampling Inefficiency In order to be able to apply the Variational Monte Carlo formalism to the ans¨atze just described, such as PauliNet or FermiNet, one must be able to sample from the densities corresponding to the wavefunctions given by their neural networks. In general, this is only possibly using Markov Chain Monte Carlo (MCMC) techniques such as Langevin Monte Carlo (Umrigar, Nightingale, and Runge 1993) or any of several variations. The issue with using such MCMC approaches to sampling is that they are inherently time-consuming: each sample is itself the solution of a stochastic differential equation as time goes to infinity.

Goals and Contributions The main goal of this paper is to solve the problem of sampling inefficiency, thereby yielding faster algorithms for solving the Electronic Schr¨odinger Equation. We achieve this goal by specifying a wavefunction ansatz which is easy to sample from, yet satisfies the requisite quantum mechanical properties. In particular, we use an ansatz based on specially designed normalizing flows. More specifically, we provide the following contributions:

• We establish that an ansatz satisfying our desiderata can be instantiated as a normalizing flow with these characteristics: (a) its base distribution is symmetric under a particular subgroup of the permutation group, and vanishes for identical electrons; (b) the flow transformation is equivariant to the same subgroup of the permutation group.   
• We show that the base distribution can be constructed using a particular combination of Determinantal Point Processes.   
• We construct both continuous and discrete normalizing flows obeying the requisite equivariance.   
• We provide a training regimen based on standard stochastic gradient descent.   
• We show how to accommodate cusps, which encapsulate non-smooth aspects of the wavefunction.   
• We generalize the framework so that induction across multiple molecules may be accommodated, while including the necessary additional invariances, in particular rigid motion invariance.

We end by noting that our contributions are of a theoretical character. Our interest is to elucidate a theoretical approach to solving the Electronic Schro¨dinger Equation based on neural networks, which retains both the expressivity of the neural network function class while at the same time providing considerably greater efficiency. These contributions should be of importance and interest to the growing community of researchers working on the boundary of machine learning and quantum physics; including those researchers whose work involves the practical computation of ground state energies.

# 2 Related Work

Neural Networks and Physics We begin by noting that neural networks have recently found broad and quite varied use in physics problems. Amongst others, examples include solution of physics-based partial differential equations (Raissi, Perdikaris, and Karniadakis 2019; Li et al. 2020; Lu et al. 2021); density functional theory (Ryczko, Strubbe, and Tamblyn 2019; Kirkpatrick et al. 2021); topological states (Deng, Li, and Das Sarma 2017a); quantum state tomography (Torlai et al. 2018); and quantum optics (Rozenberg et al. 2021, 2022). Useful surveys include (Carleo et al. 2019; Zhang et al. 2023).

Pure Spin Systems Various works have used neural networks as the ansatz in the case of pure-spin systems, sometimes also referred to as “discrete space systems”, in which the spins are at fixed locations. These include (Carleo and Troyer 2017; Deng, Li, and Das Sarma 2017b; Gao and Duan 2017; Levine et al. 2019; Sharir, Shashua, and Carleo 2022; Passetti et al. 2023).

Continuous Space Systems In terms of continuous space problems of the sort that interest us, DeepWF (Han, Zhang, and Weinan 2019) bases its on ansatz on the classical SlaterJastrow formalism, but learns both the symmetric and antisymmetric parts; the latter contains only two-electron terms, limiting the accuracy. PauliNet (Hermann, Scha¨tzle, and Noe´ 2020; Scha¨tzle, Hermann, and Noe´ 2021) also bases its ansatz on the Slater-Jastrow-Backflow form, but does so in a way that captures many-electron interactions, while respecting permutation-equivariance; this, as well as the inclusion of cusp terms, leads to much higher accuracy (e.g. $9 7 . 3 \%$ of the correlation energy for boron atoms). FermiNet (Pfau et al. 2020; Spencer et al. 2020) attains still higher accuracy (e.g. $9 9 . 8 \%$ of the correlation energy for boron atoms) by using an appropriately designed neural network to represent the entire wavefunction, which contains a generalization of Slater determinants to account for all-electron interactions. A hybrid solution which improves upon both PauliNet and FermiNet is presented in (Gerard et al. 2022). Techniques for learning / induction across several molecules or materials at once are presented in (Gao and G¨unnemann 2023; Scherbela, Gerard, and Grohs 2024; Gerard et al. 2024). We briefly mention applications to periodic systems (Wilson et al. 2022; Li, Li, and Chen 2022; Pescia et al. 2022; Cassella et al. 2023); techniques that use Diffusion Monte Carlo (Wilson et al. 2021; Ren et al. 2023); and methods that deal with excited states (Entwistle et al. 2023; Pfau et al. 2023; Naito et al. 2023).

Normalizing Flows and Quantum Problems There is a body of work which has applied normalizing flows to problems in lattice field theories, e.g. (Kanwar et al. 2020; Boyda et al. 2021; Abbott et al. 2023b,a); due to the different physical scenario, the setup in these papers is naturally quite different from ours. The work by (Xie, Zhang, and Wang 2022) introduced the use of normalizing flows in the context of Fermionic problems; follow on papers for specific applications include (Xie, Zhang, and Wang 2023; Saleh et al. 2023). Our paper considerably broadens this approach, by (a) establishing all results rigorously, as in Theorems 1 - 6; (b) showing how to practically implement the flows with discrete layers, as in Theorem 7; (c) providing important extensions to deal with phase computation (see Theorem 9), the crucial issue of cusps (see Theorem 10), and induction across multiple molecules (see Theorems 11 - 12). Finally, we mention two further papers: (Stokes, Chen, and Veerapaneni 2023) focuses on geometric aspects of the problem; while WaveFlow (Thiede, Sun, and Aspuru-Guzik 2022), develops a specialized normalizing flow architecture for onedimensional systems, which is interesting, albeit of limited applicability.

# 3 Problem Setup

# 3.1 Goals

The Setting Our overall goal is to compute the ground state wavefunction and energy of a molecule given its molecular parameters and spin multiplicity. Denote $x _ { i } = ( r _ { i } , s _ { i } )$ to be the pair consisting of the position and spin for the $i ^ { t h }$ electron; $x$ will denote the entire ordered list $( x _ { 1 } , \ldots , x _ { n } )$ , with corresponding definitions for $r$ and $s$ . We specify wavefunctions as $\psi ( x )$ ; due to the fact that electrons are Fermions, valid wavefunctions must be antisymmetric, that is if $\pi \in \mathbb { S } _ { n }$ is a permutation, then

$$
\psi ( \pi x ) = ( - 1 ) ^ { \pi } \psi ( x )
$$

where as usual, $( - 1 ) ^ { \pi }$ is shorthand for $( - 1 ) ^ { N ( \pi ) }$ where $N ( \pi )$ is the minimal number of flips to produce $\pi$ .

Let $R _ { I }$ and $Z _ { I }$ denote the position and atomic number of the $I ^ { t h }$ nucleus, and let the Laplacian for the $i ^ { t h }$ electron

$\begin{array} { r } { \Delta _ { i } = \frac { \partial ^ { 2 } } { \partial r _ { i 1 } ^ { 2 } } + \frac { \partial ^ { 2 } } { \partial r _ { i 2 } ^ { 2 } } + \frac { \partial ^ { 2 } } { \partial r _ { i 3 } ^ { 2 } } } \end{array}$ ∂∂r2 , and denote the sum of the Laplacians by $\Delta = \textstyle \sum _ { i } ^ { \cdot \bar { 1 } } \Delta _ { i }$ ; then the Hamiltonian is given by

$$
\begin{array} { r } { H = - \frac { 1 } { 2 } \Delta + V ( x ) } \end{array}
$$

where the potential $V ( x )$ is given by

$$
\begin{array} { r } { V ( x ) = \sum _ { i > j } \frac { 1 } { \| r _ { i } - r _ { j } \| } - \sum _ { i I } \frac { Z _ { I } } { \| r _ { i } - R _ { I } \| } + \sum _ { I > J } \frac { Z _ { I } Z _ { J } } { \| R _ { I } - R _ { J } \| } } \end{array}
$$

Our goal is to compute the ground state wavefunction, which we denote as $\psi _ { 0 } ( x )$ and corresponding ground state energy $E _ { 0 }$ . They may be computed using the variational principle, i.e. by minimizing the Rayleigh quotient:

$$
\psi _ { 0 } = \operatorname * { a r g m i n } _ { \psi \in \Psi } \frac { \langle \psi | H | \psi \rangle } { \langle \psi | \psi \rangle } \quad \mathrm { a n d } \quad E _ { 0 } = \frac { \langle \psi _ { 0 } | H | \psi _ { 0 } \rangle } { \langle \psi _ { 0 } | \psi _ { 0 } \rangle }
$$

where $\Psi$ is the set of all possible valid wavefunctions, and $H$ is the Hamiltonian. If we specify the wavefunction ansatz as a neural network with parameters $\theta$ , this becomes

$$
\theta ^ { * } = \operatorname * { a r g m i n } _ { \theta } \frac { \langle \psi ( \cdot ; \theta ) | H | \psi ( \cdot ; \theta ) \rangle } { \langle \psi ( \cdot ; \theta ) | \psi ( \cdot ; \theta ) \rangle }
$$

and

$$
E ^ { * } = \frac { \langle \psi ( \cdot ; \theta ^ { * } ) | H | \psi ( \cdot ; \theta ^ { * } ) \rangle } { \langle \psi ( \cdot ; \theta ^ { * } ) | \psi ( \cdot ; \theta ^ { * } ) \rangle } \geq E _ { 0 }
$$

That is, we compute an upper bound $E ^ { * }$ to the ground state energy $E _ { 0 }$ . The more expressive the ansatz, the tighter the bound will be.

Variational Monte Carlo The issue with the formulation to this point is the need to compute the inner products in Equations (4) and (6), which correspond to very high-dimensional integrals. A standard solution to this problem is based on a Monte Carlo scheme. To begin with, let us define the local energy $\mathcal { E } ( x )$ and its real part $\mathcal { E } _ { r } ( x )$ as

$$
\mathcal { E } ( x ) \equiv \frac { H \psi ( x ) } { \psi ( x ) } = - \frac { \Delta \psi ( x ) } { 2 \psi ( x ) } + V ( x ) , \mathcal { E } _ { r } ( x ) = \mathfrak { R e } \{ \mathcal { E } ( x ) \}
$$

In this case, one can simplify the minimand in Equation (4) (see the Extended Version) as

$$
\frac { \langle \psi | H | \psi \rangle } { \langle \psi | \psi \rangle } = \mathbb { E } _ { x \sim \rho ( \cdot ) } \left[ \mathcal { E } _ { r } ( x ) \right] \approx \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathcal { E } _ { r } \left( x ^ { ( k ) } \right)
$$

where the $x ^ { ( k ) }$ are sampled from

$$
\rho ( x ) = \frac { | \psi ( x ) | ^ { 2 } } { \langle \psi | \psi \rangle } .
$$

# 3.2 The General Approach

As detailed in Sections 1 and 2, a number of recent works have followed the above approach using a variety of neural networks as the ansatz for the wavefunction $\psi ( \cdot ; \theta )$ . In order to do so, one must be able to sample from $\rho ( x ; \theta ) =$ $| \psi ( x ; \theta ) | ^ { 2 } / \langle \psi | \psi \rangle$ ; and as the networks are quite general, the only feasible method for sampling is a Markov Chain Monte Carlo technique such as Langevin Monte Carlo (Umrigar, Nightingale, and Runge 1993) or any of several variations.

This can be time-consuming, as each sample is the solution of a stochastic differential equation as time goes to infinity.

A solution to this problem presents itself if we can somehow specify a wavefunction $\psi ( x )$ which is easy to sample from. We are interested in wavefunctions which satisfy the following three properties:

(W1) There is an explicit functional form for the wavefunction $\psi ( x )$ .   
(W2) $\psi$ is antisymmetric.   
(W3) We can sample non-iteratively (in constant time) from $| \psi ( \cdot ) | ^ { 2 }$ .

The first two properties are necessary for any form of Variational Monte Carlo: (W1) allows us to evaluate the local energy $\mathcal { E } _ { r }$ in (7) for use in (8); and (W2) is required for valid electronic (Fermionic) wavefunctions. But (W3) is the new ingredient: if we have a family of wavefunctions $\psi$ satisfying (W1)-(W3), then solving the minimization in (6) via the Monte Carlo approach in (8) will be considerably accelerated, as each sample will only require constant time to generate. We add a fourth property, which is not strictly necessary but is both desirable and will prove useful:

(W4) $\psi$ is normalized, that is $\begin{array} { r } { \int | \psi ( x ) | ^ { 2 } d x = 1 } \end{array}$ .

It turns out that generating such wavefunctions is possible using the following procedure:

Theorem 1. Let $\rho ( \cdot )$ be a probability density function which we can sample from in constant time. Let $\rho ( \cdot )$ satisfy two additional properties:

$( D I )$ $\rho ( x )$ is symmetric: $\rho ( \pi x ) = \rho ( x )$ for all permutations $\pi \in \mathbb { S } _ { n }$ .   
$( D 2 )$ $\rho ( x ) = 0$ if $x _ { i } = x _ { j }$ for any $i , j$ .

Finally, let $\kappa ( \boldsymbol { x } )$ be a complex function which satisfies $| \kappa ( x ) | = 1 \forall x$ , and is nearly antisymmetric:

$$
\kappa ( \pi x ) = \left\{ \begin{array} { l l } { ( - 1 ) ^ { \pi } \kappa ( x ) } & { i f x _ { i } \neq x _ { j } f o r a l l i , j } \\ { \bar { \kappa } } & { o t h e r w i s e } \end{array} \right.
$$

where $\bar { \kappa } ~ \in ~ \mathbb { C }$ is an arbitrary value with $| \bar { \kappa } | = 1$ . Then $\psi$ satisfies $( W I ) _ { - } ( W 4 )$ if and only if $\psi$ can be written as $\psi ( x ) = \kappa ( x ) \sqrt { \rho ( x ) }$ with $\kappa$ and $\rho$ satisfying the above-stated properties.

Proof: See the Extended Version.

The general idea expressed in Theorem 1 is that we can build the wavefunction $\psi$ out of an easy-to-sample-from density function satisfying additional properties (D1)-(D2); and a nearly antisymmetric phase function $\kappa$ . In what follows, we will show how to construct both of these ingredients. But before doing so, we take a short detour to address the most important practical scenario, that of fixed spin multiplicity.

# 3.3 Fixed Spin Multiplicity

Notation As in most approaches to this problem, we assume that the spin multiplicity of the molecule is specified, which is equivalent to fixing the number of spin up and spin down electrons, denoted $n _ { u }$ and $n _ { d }$ respectively, with $n _ { u } + n _ { d } = n$ . Define the canonical spin vector to be given by $\bar { s } = [ \uparrow , \dots , \uparrow$ $, \downarrow , . . . , \downarrow ]$ , i.e. the first $n _ { u }$ are $\uparrow$ , the last $n _ { d }$ are $\downarrow$ . We let the sets of indices of up and down spin electrons for the canonical spin vector be denoted by $\bar { \mathcal { N } _ { u } } = \{ 1 , \dots , n _ { u } \}$ and $\mathcal { N } _ { d } = \{ n _ { u } { \stackrel { - } { + } } 1 , . . . , n \}$ . Finally, we will be interested in the subgroup of permutations in which a permutation is applied separately to spin-up and spin-down electrons. We denote this subgroup by

$$
\begin{array} { r l r } { \mathbb G \equiv \mathbb S _ { N _ { u } } \times \mathbb S _ { \mathcal N _ { d } } } & { { } } & { \left( \mathbb G \mathrm { ~ i s ~ a ~ s u b g r o u p ~ o f ~ } \mathbb S _ { n } \right) } \end{array}
$$

Specification of the Density In the case of fixed spin multiplicity, the specification of the density $\rho ( x )$ is simplified:

Theorem 2. Given a configuration $\boldsymbol { x } = \left( \boldsymbol { r } , \boldsymbol { s } \right)$ , let a permutation which maps the spin vector s to the canonical spin vector s¯ be given by $\bar { \pi } _ { s }$ , i.e. $\bar { s } = \bar { \pi } _ { s } s$ . Let $\bar { \rho } ( \boldsymbol { r } )$ be a density function on electron positions (i.e. no spins) satisfying

$( R I )$ ) $\bar { \rho }$ is $\mathbb { G }$ -invariant: $\bar { \rho } ( \pi r ) = \bar { \rho } ( r )$ for all $\pi \in \mathbb { G }$ $( R 2 )$ $\bar { \rho } ( r ) = 0$ if $r _ { i } = r _ { j }$ , for $i , j \in \mathcal { N } _ { u } o r i , j \in \mathcal { N } _ { d }$

A density $\rho ( x ) = \rho ( r , s )$ satisfies conditions $( D I ) _ { - } ( D 2 )$ in Theorem 1 if and only if it may be written as $\rho ( r , s ) = \bar { \rho } ( \bar { \pi } _ { s } r )$ for a density $\bar { \rho } ( \boldsymbol { r } )$ satisfying conditions $( R I )$ and $( R 2 )$ .

Proof: See the Extended Version.

To summarize: in the case of fixed spin multiplicity, specifying a wavefunction $\psi$ satisfying our desired conditions (W1)- (W4) is equivalent to specifying a density $\bar { \rho } ( \boldsymbol { r } )$ satisfying conditions (R1)-(R2); and then applying the transformations given in Theorems 1 and 2 to map from $\bar { \rho }$ to $\psi$ .1 Therefore, henceforth we will focus exclusively on specifying densities $\bar { \rho } ( \boldsymbol { r } )$ satisfying conditions (R1)-(R2). To avoid unnecessary notational complexity, we will drop the bars and simply write $\rho ( r )$ .

# 4 Using Normalizing Flows to Construct the Wavefunction Ansatz

# 4.1 Sufficient Properties of the Normalizing Flow’s Base Density and Transformation

Our goal is to use a normalizing flow to construct the density $\rho ( r )$ . Let $D$ be the ambient dimension (i.e. $D = 3$ ) and $n$ be the number of electrons. The relevant vectors will live in the space $\mathbb { R } ^ { D n }$ construed as the Cartesian product $\mathbb { R } ^ { D } \times \cdots \times \mathbb { R } ^ { D }$ (which is of course isomorphic to $\mathbb { R } ^ { D _ { n } ^ { - } }$ ). A normalizing flow will consist of two ingredients: (1) a base random variable $z$ , which lives in $\mathbb { R } ^ { D n }$ , and is described by the density $\rho _ { z } ( z )$ ; (2) an invertible transformation $T : \mathbb { R } ^ { D { \dot { n } } }  \mathbb { R } ^ { D n }$ , such that $r = T ( z )$ . In this case, the density $\rho ( r )$ is the push-forward of $\rho _ { z }$ along $T$ , and is given by the change of variables formula

$$
\rho ( r ) = \rho _ { z } ( T ^ { - 1 } ( r ) ) | \operatorname* { d e t } J _ { T ^ { - 1 } } ( r ) |
$$

Recall that we would like our density $\rho ( r )$ to satisfy conditions (R1)-(R2) laid out in Theorem 2. The following theorem establishes conditions for this to occur:

Theorem 3. Suppose that we have a normalizing flow, whose base density $\rho _ { z }$ satisfies properties $( R I )$ and $( R 2 )$ from Theorem 2, and whose transformation $T$ is $\mathbb { G }$ -equivariant. Then the density resulting from the normalizing flow will satisfy properties $( R I )$ and $( R 2 )$ .

Proof: See the Extended Version.

Armed with this key result, we now set out to design the base density $\rho _ { z }$ and transformation $T$ which satisfy the conditions of Theorem 3.

# 4.2 Base Density: Determinantal Point Processes

In most cases in machine learning, the base density for a normalizing flow is taken to be a standard distribution, most often a Gaussian. In our case, we require that the base density have certain special properties, namely (R1) and (R2) from Theorem 2. It turns out that Determinantal Point Processes (DPPs) have just the properties we require. In particular, we are interested in the class of DPPs known as Projection DPPs (Gautier, Bardenet, and Valko 2019; Lavancier, Møller, and Rubak 2015), which can be specified as follows. We will let $y$ specify a generic point in $\mathbf { \mathbb { R } } ^ { D }$ . Let $h _ { k } : \mathbb { R } ^ { D }  \mathbb { R }$ for $k = 1 , \dots , n$ be a set of $n$ functions which are orthogonal, that is $\begin{array} { r } { \langle h _ { i } , h _ { j } \rangle = \int _ { \mathbb { R } ^ { D } } h _ { i } ( y ) h _ { j } ( y ) d y = \delta _ { i j } } \end{array}$ . Let $H ( y )$ be the column vector composed by stacking the individual functions $h _ { i } ( y )$ and define the kernel function as $K ( y , y ^ { \prime } ) = H ( y ) ^ { T } H ( y ^ { \prime } )$ . Then for a given collection of $n$ points in $\mathbb { R } ^ { D }$ , that is $\boldsymbol { r } = ( r _ { 1 } , \ldots , r _ { n } )$ , we define the $n \times n$ kernel matrix ${ \bf K } _ { n } ( r )$ :

$$
\mathbf { K } _ { n } ( r ) = \left[ \begin{array} { c c c } { K ( r _ { 1 } , r _ { 1 } ) } & { \ldots } & { K ( r _ { 1 } , r _ { n } ) } \\ { \vdots } & { \ddots } & { \vdots } \\ { K ( r _ { n } , r _ { 1 } ) } & { \ldots } & { K ( r _ { n } , r _ { n } ) } \end{array} \right]
$$

Using this kernel matrix, we specify the density of the Projection DPP as follows:

$$
\rho _ { d p p } ( r ; n ) = \frac { 1 } { n ! } \operatorname* { d e t } \mathbf { K } _ { n } ( r )
$$

Since ${ \bf K } _ { n } ( r )$ is positive semi-definite, it follows that its determinant is non-negative so that $\rho _ { d p p } ( r ; n )$ is non-negative, as desired. A proof that $\rho _ { d p p } ( r ; n )$ is properly normalized (i.e. integrates to 1) can be found, for example, in Proposition 2.10 of (Johansson 2006).

Given the notion of a Projection DPP, we may define the base density as follows. As above, let the base random variable be $z$ , where $z$ can be broken into spin-up and spin-down pieces, denoted $z _ { u }$ and $z _ { d }$ . (Specifically, $z _ { u }$ and $z _ { d }$ are the parts of $z$ corresponding to electrons in $\mathcal { N } _ { u }$ and $\mathcal { N } _ { d }$ , respectively.) The base density can then be constructed by taking

$$
\rho _ { z } ( z ) = \rho _ { d p p } ( z _ { u } ; n _ { u } ) \rho _ { d p p } ( z _ { d } ; n _ { d } )
$$

That is, $z _ { u }$ and $z _ { d }$ are chosen from two independent Projection DPPs. We then have the following theorem:

Theorem 4. Let $\rho _ { z }$ be the density specified in Equation (15).   
Then $\rho _ { z }$ satisfies conditions $( R I )$ and $( R 2 )$ from Theorem 2.

Proof: See the Extended Version.

We therefore have an explicit form for the base density from Equations (14) and (15). Furthermore, sampling from the base density amounts to sampling from two independent Projection DPPs. A sampling procedure for Projection DPPs is specified in the Extended Version.

We note that it is possible to make the base density itself learnable. This is achieved by making each Projection DPP learnable, through the introduction of learnable orthogonal functions $h _ { k } : \breve { \mathbb { R } } ^ { D } \to \mathbb { R }$ , that is $h _ { k } ( y ; \theta )$ , from which the kernel function $K ( y , y ^ { \prime } ; \theta ) ~ = ~ H ( y ; \theta ) ^ { T } H ( y ; \theta )$ becomes learnable. In order to retain orthogonality, one may proceed as follows. Let $B _ { \theta } : \mathbb { R } ^ { D }  \mathbb { R } ^ { n }$ be a specified by a network with parameters $\theta$ , for example a Multilayer Perceptron (in which case $B _ { \theta }$ is a type of neural field), and let $H ( \bar { y } ; \theta ) = A B _ { \theta } ( y )$ for a square $n \times n$ matrix $A$ . Let the Gram matrix of the network $B _ { \theta }$ be given by $\Xi _ { \theta }$ , that is $( \Xi _ { \theta } ) _ { i j } = \langle ( B _ { \theta } ) _ { i } , ( B _ { \theta } ) _ { j } \rangle$ . If the eigendecomposition of $\Xi _ { \theta }$ is given by $\Xi _ { \theta } = U _ { \theta } \Lambda _ { \theta } U _ { \theta } ^ { T }$ , then orthogonality of $H ( \cdot ; \theta )$ is achieved by choosing $A =$ $A _ { \theta } = \Lambda _ { \theta } ^ { - 1 7 2 } U _ { \theta } ^ { T }$ .

# 4.3 $\mathbb { G }$ -Equivariant Layers

As noted in Section 4, we require the normalizing flow transformation to be $\mathbb { G }$ -equivariant. Of course, chaining together many layers which are each $\mathbb { G }$ -equivariant results in an overall transformation which is also $\mathbb { G }$ -equivariant. Now, suppose that a particular layer $\ell$ can be written as

$$
r ^ { \ell + 1 } = T ^ { \ell } ( r ^ { \ell } )
$$

where $r ^ { \ell } = ( r _ { 1 } ^ { \ell } , \ldots , r _ { n } ^ { \ell } )$ and likewise for $r ^ { \ell + 1 }$ . We will need to see the action on the spin-up and spin-down electrons separately, so we denote $r _ { u } ^ { \ell } = ( \dot { r } _ { i } ^ { \ell } ) _ { i \in \mathcal { N } _ { u } }$ and $\boldsymbol { r } _ { d } ^ { \ell } = ( \boldsymbol { r } _ { i } ^ { \ell } ) _ { i \in \mathcal { N } _ { d } }$ ; and we may write

$$
r _ { u } ^ { \ell + 1 } = T _ { u } ^ { \ell } ( r _ { u } ^ { \ell } , r _ { d } ^ { \ell } ) \quad { \mathrm { a n d } } \quad r _ { d } ^ { \ell + 1 } = T _ { d } ^ { \ell } ( r _ { u } ^ { \ell } , r _ { d } ^ { \ell } )
$$

For notational convenience, we use $\alpha \in \{ u , d \}$ to denote the spin, and the complement of the spin is given by $\hat { \alpha }$ (i.e. if $\alpha =$ $u$ then $\hat { \alpha } = d$ and vice-versa). Then we have the following theorem:

Theorem 5. The transformation $T ^ { \ell }$ is $\mathbb { G }$ -equivariant if and only $i f$

$$
T _ { \alpha } ^ { \ell } ( \pi _ { \alpha } r _ { \alpha } ^ { \ell } , \pi _ { \hat { \alpha } } r _ { \hat { \alpha } } ^ { \ell } ) = \pi _ { \alpha } T _ { \alpha } ^ { \ell } ( r _ { \alpha } ^ { \ell } , r _ { \hat { \alpha } } ^ { \ell } ) \qquad \alpha \in \{ u , d \}
$$

That is, $T _ { \alpha } ^ { \ell }$ is equivariant with respect to $r _ { \alpha } ^ { \ell }$ , and invariant with respect to rℓαˆ.

Proof: See the Extended Version.

We now show how to specify continuous and discrete normalizing flows satisfying Theorem 5.

# 4.4 Continuous Normalizing Flows

According to Theorem 3, we are required a find a transformation which is $\mathbb { G }$ -equivariant. We now show this can be achieved via a continuous normalizing flow. We specify this flow via the ordinary differential equation (ODE)

$$
\frac { d v } { d t } = \Gamma _ { t } ( v ) , \quad \mathrm { w i t h } \quad v ( 0 ) = z \sim \rho _ { z } ( \cdot ) \quad \mathrm { a n d } \quad r = v ( 1 )
$$

That is, the transformation $r = T ( z )$ is derived as follows: the initial condition is sampled from the base density; and $r$ is gotten by integrating the ODE forward to time $t = 1$ . $\Gamma$ ’s $t$ -dependence is indicated via a subscript for notational convenience. We then have the following theorem:

Theorem 6. Let the transformation $r = T ( z )$ be specified as in Equation (19). Then $T$ is $\mathbb { G }$ -equivariant if $\Gamma _ { t }$ is $\mathbb { G }$ - equivariant for all $t$ .

Proof: See the Extended Version.

It therefore suffices to design a $\mathbb { G }$ -equivariant function $\Gamma _ { t }$ . Let us break this down by spin: from Theorem 5, we know that this implies that for all $t$ , we have that $\Gamma _ { t } ( \pi _ { \alpha } r _ { \alpha } , \pi _ { \hat { \alpha } } r _ { \hat { \alpha } } ) =$ $\pi _ { \alpha } \Gamma _ { t } ( r _ { \alpha } , r _ { \hat { \alpha } } )$ for $\alpha \in \{ u , d \}$ . We show in the Extended Version how to implement a layer of $\Gamma$ with a combination of multihead attention, fully connected layers, and linear projections $\Gamma$ can be composed of many such layers).

Continuous normalizing flows are elegant; however, they can present some numerical difficulties. In particular, the issue of ODE stiffness frequently arises in deep learning pipelines involving continuous normalizing flows. Thus, we now present an alternative method, based on discrete normalizing flows.

# 4.5 Discrete Normalizing Flows

Our goal is now to design such functions $T _ { u } ^ { \ell }$ and $\textstyle T _ { d } ^ { \ell }$ which satisfy Equation (18), and for which the overall transformation $\dot { T } ^ { \ell } \overset { \bullet } { = } ( T _ { u } ^ { \ell } , \dot { T } _ { d } ^ { \ell } )$ is invertible. The goal of the layer we propose here is to not sacrifice on expressivity, especially when compared to many layers which are designed for discrete normalizing flows. In particular, the main issue will be to show that the expressivity can be retained even with the joint requirements of invertibility and $\mathbb { G }$ -equivariance. We note that the kind of transformation we propose below is not generally used for normalizing flows, as the determinant of its Jacobian is not fast to compute; however, this is not an issue in our case, as the dimension of the spaces we are dealing with is moderate in size. For a more detailed discussion, see the Extended Version.

To solve this problem, we introduce the Split Subspace Layer; we note that this layer may be of broader interest in machine learning, independent of the current setting. As before, we take $D$ to represent the ambient spatial dimension; in our case, $D = 3$ . A key parameter for the $\dot { \ell } ^ { t h }$ layer will be the orthogonal matrix $\Lambda _ { \alpha } ^ { \bar { \ell } } \in O ( D )$ ; in particular, we divide this matrix into 2 pieces

$$
\Lambda _ { \alpha } ^ { \ell } = [ \beta _ { \alpha } ^ { \ell } , \xi _ { \alpha } ^ { \ell } ] \mathrm { w i t h } \beta _ { \alpha } ^ { \ell } \in \mathbb { R } ^ { D \times D _ { \beta } } \mathrm { a n d } \xi _ { \alpha } ^ { \ell } \in \mathbb { R } ^ { D \times ( D - D _ { \beta } ) }
$$

That is, $\beta _ { \alpha } ^ { \ell }$ represents the first $D _ { \beta }$ columns of $\Lambda _ { \alpha } ^ { \ell }$ , and $\xi _ { \alpha } ^ { \ell }$ represents the final $D - D _ { \beta }$ columns. For each electron $i$ , we compute the inner product of its coordinates with $\beta _ { \alpha } ^ { \ell }$ , i.e.

$$
\begin{array} { r l } { \gamma _ { \alpha , i } ^ { \ell } = ( \beta _ { \alpha } ^ { \ell } ) ^ { T } r _ { \alpha , i } ^ { \ell } } & { { } \mathrm { s o } \mathrm { t h a t } \gamma _ { \alpha , i } ^ { \ell } \in \mathbb { R } ^ { D _ { \beta } } } \end{array}
$$

We can collect the individual vectors $\gamma _ { \alpha , i } ^ { \ell }$ into a list $\gamma _ { \alpha } ^ { \ell } =$ $( \gamma _ { \alpha , i } ^ { \ell } ) _ { i \in \mathcal { N } _ { \alpha } }$ . Given this, we define the Split Subspace Layer $T _ { \alpha } ^ { \ell }$ on a per-electron basis by

$$
r _ { \alpha , i } ^ { \ell + 1 } = T _ { \alpha , i } ^ { \ell } ( r _ { \alpha } ^ { \ell } , r _ { \hat { \alpha } } ^ { \ell } ) = r _ { \alpha , i } ^ { \ell } + \xi _ { \alpha } ^ { \ell } \varphi _ { \alpha , i } ^ { \ell } ( \gamma _ { \alpha } ^ { \ell } , \gamma _ { \hat { \alpha } } ^ { \ell } )
$$

where $\varphi _ { \alpha } ^ { \ell }$ is a network, $\varphi _ { \alpha , i } ^ { \ell }$ is the part of (the output of) $\varphi _ { \alpha } ^ { \ell }$ corresponding to the $i ^ { t h }$ electron, and $\varphi _ { \alpha , i } ^ { \ell } ( \gamma _ { \alpha } ^ { \ell } , \gamma _ { \hat { \alpha } } ^ { \ell } ) \in$ $\mathbb { R } ^ { D - D _ { \beta } }$ . A crucial aspect of this $\varphi _ { \alpha } ^ { \ell }$ network is that it captures dependencies between all electrons. This can been seen by examining Equation (22): $\varphi _ { \alpha , i } ^ { \ell }$ depends on both $\gamma _ { \alpha } ^ { \ell }$ and $\gamma _ { \hat { \alpha } } ^ { \ell }$ . But recall that $\gamma _ { \alpha } ^ { \ell }$ is a list of the vectors $\gamma _ { \alpha , i } ^ { \ell }$ for each electron $i$ with spin $\alpha$ , and $\gamma _ { \hat { \alpha } } ^ { \ell }$ is the corresponding list for electrons with the complement spin $\hat { \alpha }$ . As a result, $\varphi _ { \alpha , i } ^ { \ell }$ depends on all electrons, as desired.

The layer is referred to as the Split Subspace Layer due to the fact that its input is one subspace of $\mathbb { R } ^ { \mathbf { \hat { D } } }$ , given by $\beta _ { \alpha } ^ { \ell }$ ; whereas its output is in the orthogonal complement of this subspace, given by $\xi _ { \alpha } ^ { \ell }$ . Note that there are multiple possible versions of this layer, as any choice $D _ { \beta }$ in the set $\bar  \left\{ 1 , \ldots , D - \right.$ $1 \}$ is valid. Since in our case $D = 3$ , this gives us exactly two choices: $D _ { \beta } = 1$ or $D _ { \beta } = 2$ .

The main ingredient of the layer is the network $\varphi _ { \alpha } ^ { \ell }$ . We now show two things: (1) the layer is invertible for any choice of $\varphi _ { \alpha } ^ { \ell }$ (2) we derive conditions on $\varphi _ { \alpha } ^ { \ell }$ to achieve $\mathbb { G }$ -equivariance of $T _ { \alpha } ^ { \ell }$ .

Theorem 7. Let $T ^ { \ell }$ be a Split Subspace Layer, as given in Equation (22). Then $T ^ { \ell }$ is invertible. In particular, let $\underline { { \gamma } } _ { \alpha , i } ^ { \ell + 1 } = ( \beta _ { \alpha } ^ { \ell } ) ^ { T } r _ { \alpha , i } ^ { \ell + 1 } .$ ; then the inverse of the layer is given by

$$
r _ { \alpha , i } ^ { \ell } = r _ { \alpha , i } ^ { \ell + 1 } - \xi _ { \alpha } ^ { \ell } \varphi _ { \alpha , i } ^ { \ell } ( \underline { { \gamma } } _ { \alpha } ^ { \ell + 1 } , \underline { { \gamma } } _ { \hat { \alpha } } ^ { \ell + 1 } )
$$

Furthermore, the layer $T ^ { \ell }$ is $\mathbb { G }$ -equivariant if

$$
\varphi _ { \alpha } ^ { \ell } ( \pi _ { \alpha } \gamma _ { \alpha } ^ { \ell } , \pi _ { \hat { \alpha } } \gamma _ { \hat { \alpha } } ^ { \ell } ) = \pi _ { \alpha } \varphi _ { \alpha } ^ { \ell } ( \gamma _ { \alpha } ^ { \ell } , \gamma _ { \hat { \alpha } } ^ { \ell } )
$$

i.e. if $\varphi _ { \alpha } ^ { \ell } ( \gamma _ { \alpha } ^ { \ell } , \gamma _ { \hat { \alpha } } ^ { \ell } )$ is equivariant with respect to permutations on $\gamma _ { \alpha } ^ { \ell }$ and invariant with respect to permutations on $\gamma _ { \hat { \alpha } } ^ { \ell }$ .

Proof: See the Extended Version.

The Split Subspace Layer therefore depends on implementation of the network $\varphi _ { \alpha } ^ { \ell }$ so that it satisfies Equation (24). We show in the Extended Version how $\varphi _ { \alpha } ^ { \ell }$ can be implemented with a combination of multihead attention, fully connected layers, and linear projections. We specify a more general version of the Split Subspace Layer in the Extended Version.

We now comment on the complexity of this method vs. that of standard MCMC approaches to sampling. In our case, a single sample may be drawn by passing a sample from the base density through the flow network; in other words, only a single call to the flow network is required. By contrast, when sampling using MCMC techniques, each sample requires many calls to the network representing the wavefunction or density. For example, in using Langevin Monte Carlo techniques, a network representing the (unnormalized) density $\rho ( r )$ may be sampled by the iterations

$$
r _ { k + 1 } = r _ { k } + \tau \nabla \log \rho ( r _ { k } ) + \sqrt { 2 \tau } \xi _ { k }
$$

where $\xi _ { k }$ is Gaussian noise. In the limit of $\tau  0$ , as $k $ $\infty$ , the samples thus generated will be distributed according to (the normalized version of) the density $\rho ( r )$ . We note that in the case of finite $\tau$ , one often adds treats each new iterate as a Metropolis-Hastings style proposal, which can be correspondingly accepted or rejected. The key point, however, is that a single sample requires many calls to the network representing $\rho$ , whereas in our case only a single network call is required.

# 4.6 Training via SGD

Log Domain: Density In order to avoid numerical issues, it is best to operate in the log domain. Suppose that

$$
\psi ( r ) = e ^ { q ( r ) + i w ( r ) }
$$

so that

$\begin{array} { r } { q ( r ) = \frac { 1 } { 2 } \log \rho ( r ) \mathrm { a n d } w ( r ) = \mathrm { a t a n } 2 \left( \kappa _ { i } ( r ) , \kappa _ { r } ( r ) \right) } \end{array}$ (27) where $\kappa _ { r } ( r )$ and $\kappa _ { i } ( \boldsymbol { r } )$ are the real and imaginary parts of the phase $\kappa ( r )$ , respectively; and atan2 is the “full” arctangent.

The log-density $q ( r ; \theta )$ may be computed for both continuous and discrete normalizing flows, where we now introduce the parameters $\theta$ of the network explicitly. Consider a sample $z$ chosen from the base density $\rho _ { z } ( z )$ , and in analogy to $q ( r )$ , define $\begin{array} { r } { q _ { z } ( z ) = \frac { 1 } { 2 } \log \rho _ { z } ( z ) } \end{array}$ . Now, in the case of a continuous normalizing flow, let $\boldsymbol { v } ( t )$ satisfy Equation (19); then $q ( r ; \theta )$ can be by computed (Chen et al. 2018) by solving the ODE

$$
\frac { d a } { d t } = - \mathrm { T r a c e } \left( \frac { \partial \Gamma _ { t } } { \partial v } ( v ( t ) ; \theta ) \right)
$$

with $a ( 0 ) = q _ { z } ( z )$ and $q ( r ; \theta ) = a ( 1 )$ . This is the continuous analogue of the change of variables formula. In the case of a discrete normalizing flow, fix the following notation: $r ^ { 0 } = z$ , $r = r ^ { L + 1 }$ , and $\breve { T } = T ^ { L } \circ \cdots \circ T ^ { 0 }$ . Then we may use a logarithmic version of the standard change of variables formula (12):

$$
q ( r ; \theta ) = q _ { z } \left( T ^ { - 1 } ( r ; \theta ) \right) + \frac { 1 } { 2 } \sum _ { \ell = 0 } ^ { L } \log \left| \operatorname * { d e t } J _ { ( T ^ { \ell } ) ^ { - 1 } } ( r ^ { \ell + 1 } ; \theta ) \right|
$$

Log Domain: Gradient of the Objective Recall that our goal in finding an approximation to the ground state wavefunction is to solve the optimization problem in Equation (6). Using Equation (8) and noting that $\langle \psi ( \cdot ; \theta ) | \psi ( \cdot ; \bar { \theta } ) \rangle = 1$ since $\rho ( \cdot ; \theta )$ is normalized, we may write the objective function to be minimized as

$$
\begin{array} { l } { \displaystyle \mathcal { L } ( \theta ) = \langle \psi ( \cdot ; \theta ) | H | \psi ( \cdot ; \theta ) \rangle } \\ { \displaystyle = \mathbb { E } _ { r \sim \rho ( \cdot ; \theta ) } \left[ \mathcal { E } _ { r } ( r ; \theta ) \right] \approx \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathcal { E } _ { r } \left( r ^ { ( k ) } ; \theta \right) } \end{array}
$$

with samples $r ^ { ( k ) } \sim \rho ( \cdot ; \theta )$ . Then we have the following theorem, which shows that the local energy can be written entirely as a function of $q ( r ; \theta )$ and the potential $V ( r )$ , so that the phase $w ( r ; \theta )$ does not appear; and furthermore gives the gradient of the objective function $\mathcal { L } ( \boldsymbol { \theta } )$ .

Theorem 8. The local energy can be written as

$\begin{array} { r } { \mathcal { E } _ { r } ( r ; \theta ) = - \frac { 1 } { 2 } \Delta _ { r } q ( r ; \theta ) - \frac { 1 } { 2 } \| \nabla _ { r } q ( r ; \theta ) \| ^ { 2 } + V ( r ) } \end{array}$ (31) In particular, the local energy is independent of the phase $w ( r ; \theta )$ . Furthermore, let

$$
\begin{array} { r } { \Omega ( r ; \theta ) = 2 \left( \mathcal { E } _ { r } ( r ; \theta ) - \mathbb { E } _ { r \sim \rho ( \cdot ; \theta ) } \left[ \mathcal { E } _ { r } ( r ; \theta ) \right] \right) \nabla _ { \theta } q ( r ; \theta ) . } \end{array}
$$

Then the gradient of the loss function may be written as

$$
\nabla _ { \theta } \mathcal { L } ( \theta ) = \mathbb { E } _ { r \sim \rho ( \cdot ; \theta ) } \left[ \Omega ( r ; \theta ) \right] \approx \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \Omega \left( r ^ { ( k ) } ; \theta \right)
$$

with samples $r ^ { ( k ) } \sim \rho ( \cdot ; \theta )$ .

Proof: See the Extended Version.

Thus, in order to optimize the objective in Equation (30), we may use gradient descent using the estimate for the gradient in Equation (33). A detailed version of the optimization routine is given in the Extended Version.

# 5 Further Details: Phase, Cusps, and Induction

# 5.1 The Phase

Since the Hamiltonian is time-reversal invariant and Hermitian, both its eigenvalues and its eigenfunctions are real. Since the ground-state wavefunction we are looking for is real, the phase can be taken to belong to the two element set $\{ 0 , \pi \}$ . Given that we now know how to solve for an approximation to the density $\rho _ { 0 } ( r )$ corresponding to the ground state wavefunction, we now show one way of assigning the phase so that the resulting ground state wavefunction $\bar { \psi } _ { 0 } ( \boldsymbol { r } )$ is appropriately antisymmetric.

Theorem 9. Let $\rho _ { 0 } ( r )$ be the the density for the ground state wavefunction. Let $\prec b e$ a strict total order on $\mathbb { R } ^ { \breve { D } }$ , and define the set

$$
\begin{array} { c } { { \mathcal R = \{ r = ( r _ { 1 } , \dots r _ { n } ) : r _ { 1 } \prec r _ { 2 } \prec \dots \prec r _ { n _ { u } } a n d } } \\ { { r _ { n _ { u } + 1 } \prec r _ { n _ { u } + 2 } \prec \dots \prec r _ { n } \} } } \end{array}
$$

For any $r$ without $r _ { i } = r _ { j }$ , define the permutation $\pi _ { \prec } ( r ) \in \mathbb { G }$ by $\pi _ { \prec } ( r ) r \in \mathcal { R }$ . Then a valid antisymmetric ground state wavefunction is given by

$$
\psi _ { 0 } ( r ) = { \left\{ \begin{array} { l l } { ( - 1 ) ^ { \pi _ { \prec } ( r ) } { \sqrt { \rho _ { 0 } ( r ) } } } & { i f r _ { i } \neq r _ { j } \forall i , j } \\ { 0 } & { o t h e r w i s e } \end{array} \right. }
$$

Proof: See the Extended Version.

Thus, given the density $\rho _ { 0 }$ , we can use Theorem 9 to easily compute the ground state wavefunction $\psi _ { 0 }$ . A question remains: what is the strict total order $\prec \ ?$ Any choice is valid, but the simplest thing to do is to use lexicographic ordering on the coordinates of the two points in $\mathbb { R } ^ { \breve { D } }$ that are being compared.

# 5.2 Incorporating Cusps

Electron-Electron Cusps Wavefunctions are known to have certain non-smooth properties, known as cusps. In particular, the gradient of the wavefunction should exhibit a discontinuity when two electrons coincide. One way to incorporate such gradient discontinuities is via the introduction of terms which depend on the distance between electrons (Pfau et al. 2020); as the distance is itself a continuous but non-smooth function of the electron positions, using distances can allow us to model such cusps. In the case of the discrete normalizing flow, our goal will be to design a layer which incorporates the inter-electron distances directly. Given the requirements of a normalizing flow, the challenge is to enforce invertibility for such a layer. We have the following result:

Theorem 10. Let the set of distances be given by $\delta ^ { \ell } =$ $\left\{ \delta _ { i j } ^ { \ell } \right\} _ { i < j }$ where $\delta _ { i j } ^ { \ell } = \| r _ { i } ^ { \ell } - \check { r } _ { j } ^ { \ell } \|$ . Given a layer of the form

$$
r _ { i } ^ { \ell + 1 } = \Theta ^ { \ell } ( \delta ^ { \ell } ; \theta ) r _ { i } ^ { \ell } + t ^ { \ell } ( \delta ^ { \ell } ; \theta )
$$

with $\Theta ^ { \ell } ( \delta ^ { \ell } ; \theta ) \in O ( D )$ and $t ^ { \ell } ( \delta ^ { \ell } ; \boldsymbol { \theta } ) \in \mathbb { R } ^ { D }$ . Then the layer is both $\mathbb { G }$ -equivariant as well as invertible.

Proof: See the Extended Version.

The essence of this layer to rotate all electrons in a given configuration $r = ( r _ { 1 } , \ldots , r _ { n } )$ by the same rotation matrix $\Theta ^ { \ell }$ and translation vector $t ^ { \ell }$ ; and the rotation matrix and translation vector are both functions the configuration $r$ entirely through the distances $\delta ^ { \ell }$ . The latter fact is crucial, as it means that different configurations $r$ are treated differently, which gives the layer expressivity. An implementation of this layer based on a Deep Set architecture (Zaheer et al. 2017) is given in the Extended Version.

It is also known that the gradient of the wavefunction should exhibit a discontinuity when an electron and nucleus coincide. The treatment is similar, and is given in the Extended Version.

# 5.3 Induction Across Multiple Molecules

In an effort to accelerate the ground state computation, we may try to learn the ground state wavefunctions and energies for an entire class of molecules simultaneously, as in (Gao and G¨unnemann 2023; Scherbela, Gerard, and Grohs 2024; Gerard et al. 2024). In particular, the molecular parameters are given by $R = ( R _ { 1 } , \ldots , R _ { N } )$ , the nuclear positions; and $Z \doteq ( Z _ { 1 } , \dot { \ldots } , Z _ { N } )$ , the atomic numbers of each nucleus. Then our goal is to learn a function of the form $\psi _ { 0 } ( x ; R , Z )$ , i.e. a ground state wavefunction which is explicitly parameterized by the molecular parameters. This entails computing the density $\rho ( r ; R , Z )$ . However, this latter task is made more complicated by the fact that two new invariances are required. The first is nuclear permutation invariance:

$$
\rho ( r ; \pi R , \pi Z ) = \rho ( r ; R , Z ) \quad { \mathrm { f o r } } \pi \in \mathbb { S } _ { N }
$$

and the second is joint rigid motion invariance:

$$
\begin{array} { r } { \rho ( \tau r ; \tau R , Z ) = \rho ( r ; R , Z ) \quad \mathrm { f o r } \tau \in E ( D ) } \end{array}
$$

We henceforth assume that the nuclei have their center of mass at the origin, i.e. $\begin{array} { r } { \hat { R } \ = \ \frac { 1 } { N } \sum _ { I = 1 } ^ { N } R _ { I } \ = \ 0 } \end{array}$ ; this removes the need to deal with translations, which generally require special (and uninteresting) treatment, e.g. see (Satorras, Hoogeboom, and Welling 2021). Thus, Equation (38) becomes joint rotation invariance:

$$
\rho ( \Theta r ; \Theta R , Z ) = \rho ( r ; R , Z ) \quad \mathrm { f o r } \Theta \in O ( D )
$$

We now show that densities satisfying Equations (37) and (39) can be realized via a variation of the continuous normalizing flow we have introduced in Section 4.4:

Theorem 11. Let $\begin{array} { r } { \bar { R } = \frac { 1 } { N } \sum _ { I = 1 } ^ { N } R _ { I } = 0 } \end{array}$ . Given a continuous normalizing flow of the form $d v / d t = \Gamma _ { t } ( v ; R , Z )$ with $v ( 0 ) = z \sim \rho _ { z } \bar { ( \cdot ) }$ and $r = v ( 1 )$ . Let the function $\Gamma _ { t }$ be invariant with respect to nuclear permutations and equivariant with respect to joint rotations, i.e. for all $t$

$$
\begin{array} { r l } & { \Gamma _ { t } ( v ; \pi R , \pi Z ) = \Gamma _ { t } ( v ; R , Z ) \forall \pi \in \mathbb { S } _ { N } } \\ & { \Gamma _ { t } ( \Theta v ; \Theta R , Z ) = \Theta \Gamma _ { t } ( v ; R , Z ) \forall \Theta \in O ( D ) } \end{array}
$$

Furthermore, suppose that the base density is invariant with respect to rotations, $\rho _ { z } ( \Theta z ) = \rho _ { z } ( z )$ for $\Theta \in O ( D )$ . Then the resulting density $\rho ( r ; R , Z )$ satisfies Equations (37) and (39).

Proof: See the Extended Version.

First, we note that the base density in Equation (15) can be made invariant to rotations by constructing the relevant Projection DPP from a kernel function $K ( y , y ^ { \prime } ) = H ( y ) ^ { T } H ( y )$ , where the functions $h _ { i } ( y )$ are derived from taking arbitrary rotationally-invariant functions $\tilde { h } _ { i } ( y )$ , and orthogonalizing them with Gram-Schmidt; e.g. one may use Gaussians of varying bandwidths, ˜hi(y) = e−∥y∥2/σi2 .

Now, we turn to the construction of $\Gamma _ { t }$ . Recall from Theorem 6 that $\Gamma _ { t } ( \cdot ; R , Z )$ must be $\mathbb { G }$ -equivariant for all $t$ . Furthermore, we have already noted that $\mathbb { G }$ -equivariant functions may be constructed using a combination of standard pieces: multihead attention, fully connected layers, and linear projections. It would be nice if we were able to use this result while also incorporating the extra conditions in Equation (38). We now show that this is possible:

Theorem 12. Let $\phi _ { t } ( v ; R , Z )$ be a function which is $\mathbb { G }$ -equivariant with respect to $v$ i.e. $\begin{array} { r l } { \phi _ { t } ( g v ; R , Z ) } & { { } = } \end{array}$ $g \phi _ { t } ( v ; R , Z )$ for $g \in \mathbb { G }$ . Let $\omega _ { t } ( v ; R , z )$ be a function whose output is itself a rotation, i.e. $\omega _ { t } ( v ; R , z ) \in O ( D )$ . Let $\omega _ { t } \ b e$ $\mathbb { G }$ -invariant with respect to $v$ , and $O ( D )$ -equivariant jointly with respect to $v$ and $R$ i.e. $\omega _ { t } ( \Theta v ; \Theta R , Z ) = \Theta \omega _ { t } ( v ; R , Z )$ . Finally, let both $\phi _ { t }$ and $\omega _ { t }$ be permutation-invariant jointly with respect to $R$ and $Z$ i.e. $\overset { \cdot } { \phi _ { t } } ( v ; \pi R , \pi Z ) = \phi _ { t } ( \overset { \cdot } { v ; } R , Z )$ and likewise for $\omega _ { t }$ . Then the function

$$
\Gamma _ { t } ( v ; R , Z ) = \zeta \phi _ { t } ( \zeta ^ { - 1 } v ; \zeta ^ { - 1 } R , Z )
$$

where $\zeta = \omega _ { t } ( v ; R , Z )$ satisfies the properties in Equation (40) and is $\mathbb { G }$ -equivariant with respect to $v$ .

Proof: See the Extended Version.

We can use the previously mentioned recipe in the Extended Version in order to construct a $\mathbb { G }$ -equivariant $\phi _ { t }$ , with an extra path in the network for the $R , Z$ dependence, based on either Deep Set or a Transformer architecture with pooling to gain the requisite invariance. The function $\omega _ { t }$ can be constructed by using an $E ( D )$ Equivariant Graph Neural Network (Satorras, Hoogeboom, and Welling 2021) whose output is a rotation matrix, similar to what is done in (Kaba et al. 2023). More detailed information is contained in the Extended Version.

# 6 Conclusions

We have demonstrated a theoretical framework for efficiently solving the Electronic Schro¨dinger Equation using normalizing flows. Using these flows allows us to sample efficiently from the wavefunction, thereby side-stepping the need for time-consuming MCMC approaches to sampling. Future work will focus on using diffusion techniques (Yang et al. 2023) to model wavefunctions, which is not straightforward due to the need to maintain the requisite quantum mechanical properties; and on adapting flow-matching (Lipman et al. 2022) and related techniques for the optimization of the variational objective.