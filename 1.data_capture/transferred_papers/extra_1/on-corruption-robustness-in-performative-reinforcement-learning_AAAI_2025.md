# On Corruption-Robustness in Performative Reinforcement Learning

Vasilis Pollatos1\*, Debmalya Mandal2†, Goran Radanovic3

1 Archimedes/Athena RC, Greece 2University of Warwick 3Max Planck Institute for Software Systems v.pollatos $@$ athenarc.gr, debmalya.mandal $@$ warwick.ac.uk, gradanovic $@$ mpi-sws.org

# Abstract

In performative Reinforcement Learning (RL), an agent faces a policy-dependent environment: the reward and transition functions depend on the agent’s policy. Prior work on performative RL has studied the convergence of repeated retraining approaches to a performatively stable policy. In the finite sample regime, these approaches repeatedly solve for a saddle point of a convex-concave objective, which estimates the Lagrangian of a regularized version of the reinforcement learning problem. In this paper, we aim to extend such repeated retraining approaches, enabling them to operate under corrupted data. More specifically, we consider Huber’s $\epsilon$ -contamination model, where an $\epsilon$ fraction of data points is corrupted by arbitrary adversarial noise. We propose a repeated retraining approach based on convex-concave optimization under corrupted gradients and a novel problemspecific robust mean estimator for the gradients. We prove that our approach exhibits last-iterate convergence to an approximately stable policy, with the approximation error linear in $\sqrt { \epsilon }$ . We experimentally demonstrate the importance of accounting for corruption in performative RL.

# Introduction

In performative reinforcement learning (Mandal, Triantafyllou, and Radanovic 2023; Rank et al. 2024), the learner operates in a policy-dependent environment, where the reward and transition functions are influenced by the learner’s policy. A canonical example of such a setting is an RL system involving human users, such as RL-based recommender systems or chatbots: the RL policy affects user preferences, which in turn alters user engagement and behavior, thereby modifying the RL environment.

For an offline learner, performativity represents a specific type of distribution shift. Similar to standard reinforcement learning (RL), where the learner’s policy influences the data it encounters due to the sequential decision-making process, performativity introduces an additional layer of complexity. The learner not only affects the observed data but also alters the underlying data generation process through its policy choices. This dual influence complicates the learning process, impacting the learner’s ability to generalize effectively.

To alleviate this challenge prior work places sensitivity assumptions, formalized as Lipschitz conditions, which limit the extent the learner’s policy can influence the reward and transition functions of the underlying environment. These conditions enable convergence guarantees to a performatively stable policy during repeated retraining. This process involves optimizing a regularized RL objective after each deployment round to update the current policy. In the finite data regime, the retraining step solves for a saddle point of a convex-concave objective, which approximates the Lagrangian of the regularized reinforcement learning problem.

In this paper, we additionally consider a third type of distribution shift, specifically induced by an adversary capable of corrupting training data. This scenario is particularly relevant in practical settings where performativity plays a role. For example, recommendation systems are vulnerable to Sybil attacks, while chatbots can be susceptible to poisoning poisoning attacks (e.g., see the Tay.ai incident (Neff 2016)). Therefore, it is crucial to explore strategies for achieving corruption-robustness in performative environments.

Our goal is to extend algorithmic and convergence results from prior work on perfomrative RL by allowing an $\epsilon$ fraction of data points in training data to be corrupted by adversarial noise—commonly referred to as the Huber ϵ-contamination model. We recognize two main challenges:

• First, this involves solving a convex-concave optimization problem where access to a given objective function is corrupted. For instance, gradient-based methods, which are commonly used for convex-concave optimization, rely on first-order gradient oracles to access the objective. However, existing guarantees for these methods typically assume that the outputs of gradient oracles are clean and not adversarially corrupted. • Second, generic robust estimators, such as those used for estimating gradients, do not account for the problem structure, often making them impractical in challenging high-dimensional settings.

Contributions: Our work aims to resolve these challenges within the performative RL framework. Our contributions are as follow:

• We propose a robust version of Optimistic Optimistic

Follow the Regularized Leader (OFTRL) for optimizing convex-concave objectives under corruption and provide theoretical analysis of thereof. This analysis shows that the robust OFTRL achieves optimal convergence rates and information-theoretically optimal terminal error. These results are of an independent interest and complement the concurrent work on gradient-based algorithms for convex-concave optimization under corrupted gradients. See the related work section for more details.

• We extend the existing algorithmic approach to performative reinforcement learning, making it robust against corruption. More specifically, we propose a novel repeated retraining approach based on robust OFTRL and a novel coordinate-wise robust mean estimator for estimating the gradients. The robust mean estimator is particularly suited for performative RL. We theoretically analyze our repeated retraining approach, showing that it exhibits last-iterate convergence to an approximately stable policy, with the approximation error that scales linearly with the square root of the corruption level ϵ. • Using a simulation-based experimental testbed, we showcase the importance of accounting for corrupted gradients, and the efficacy of our approach.

The extended version of the paper (Pollatos, Mandal, and Radanovic 2024) provides additional information, including the proofs of our formal results and implementation details for the experimental analysis.

# Related Work

We recognize several lines of works related to this paper: performative prediction and RL, corruption-robust offline RL, and convex-concave optimization, and convex optimization under corrupted gradients.

Performative Prediction and RL: Performative prediction models data distribution shifts due to the model deployment (Perdomo et al. 2020). Much prior work has studied convergence properties of algorithms to different solution concepts, including performative stability (Perdomo et al. 2020; Mendler-Du¨nner et al. 2020) and performative optimality (Izzo, Ying, and Zou 2021; Miller, Perdomo, and Zrnic 2021). In recent years, other variants of the canonical setting have been proposed, including multi-player variants (Narang et al. 2023; Piliouras and Yu 2023), variants that consider more nuanced state-dependent distribution shifts (Brown, Hod, and Kalemaj 2022; Li and Wai 2022), or variants that introduce constraints (Yan and Cao 2024) or bilevel optimization (Lu 2023). The closest to our setting is the work of (Mandal, Triantafyllou, and Radanovic 2023), who introduced the concept of performativity in reinforcement learning. The performative RL framework relates to Stackelberg stochastic games (Letchford et al. 2012; Zhong et al. 2021), in which a principal agent commits to a policy to which follower agents respond—making the principal’s effective environment performative. However, the performative RL framework abstracts away from game-theoretic considerations, as it does not model performativity through game-theoretic agents. (Rank et al. 2024) extends the performative RL framework by considering gradual environment shifts, akin to those considered in the performative precision settings of (Brown, Hod, and Kalemaj 2022; Li and Wai 2022). Our paper contributes to the extensive literature on performative prediction (Hardt and Mendler-Du¨nner 2023) by introducing corruption-robustness to performative RL.

Corruption-Robust Offline RL: From a technical perspective, our results build on the analysis (Mandal, Triantafyllou, and Radanovic 2023), who adapted the minimax optimization problem (Zhan et al. 2022) for the offline setting of performative RL. Hence, our work relates to the vast literature on corruption robustness in offline RL (Zhang et al. 2022; Wu et al. 2022; Ye et al. 2024; Nika et al. 2024; Mandal et al. 2024). However, these works do not utilize corruption-robust minimax optimization in their frameworks, nor do their underlying framework model performativity effects. An additional discussion of how the bounds in some of these works compares to ours is provided in the convergence analysis of our approach.

Convex-Concave Optimization: Our approach relies on convex-concave optimization. There has been a vast literature on this topic, from the fictitious play algorithm (Robinson 1951) and the extra-gradient method (Korpelevich 1976; Tseng 1995) for solving bilinear optimization problems, to Gradient Ascent Descent-based algorithms for the general convex-concave optimization problem (Nemirovski 2004; Nesterov 2007; Tseng 2008; Nedic´ and Ozdaglar 2009; Mokhtari, Ozdaglar, and Pattathil 2019, 2020). Most relevant to our setting are works on convex-concave optimization that assume inexact oracles. (Juditsky, Nemirovski, and Tauvel 2011; Huang and Zhang 2022) consider biased stochastic gradient oracles. In (Beznosikov, Sadiev, and Gasnikov 2020), a zeroth-order biased oracle with stochastic and bounded deterministic noise is assumed and (Dvinskikh et al. 2022) consider a zeroth-order oracle corrupted by adversarial noise. Our work differs from these results as our setting has a bounded gradient corruption (bias) and we provide guarantees for the actual error of our algorithm instead of the expected. Arguably, the closest related work on convex-concave optimization to ours is the concurrent work of (Zhang et al. 2024). They provide convergence guarantees under adversarial noise for smooth convex-concave functions, but achieve this through a different algorithm. Hence, our results on convex-concave optimization complement theirs. Our information-theoretic lower bound on the duality gap has a different flavor than the lower bounds in (Zhang et al. 2024); the latter focuses on notions related to algorithmic reproducibility. Hence, our lower bound and the analysis we used for deriving it is novel.

Convex Optimization under Corrupted Gradients: In convex optimization gradient corruption has been more extensively studied (Polyak 1987; d’Aspremont 2008; Devolder 2013; Devolder, Glineur, and Nesterov 2014). From this line of research most relevant are the works of (Prasad et al. 2020) and (Wang, Mianjy, and Arora 2021), who study corruption in gradient descent due to poisoning attacks with applications to machine learning, as well as works of (Ahn et al. 2022), that study reproducibility in optimization in the face of inexact gradients.

# Preliminaries

In this section, we provide the necessary background. Our main approach builds on convex-concave optimization under corrupted gradients. More specifically, we consider smooth convex-concave objectives. We propose a robust version of Optimistic Follow the Regularized Leader (OFTRL), and provide a convergence guarantee for it. In particular, we show that it exhibits $\bar { O } ( 1 / \bar { T } )$ convergence rate for the duality gap. This result and the analysis are of independent interest, complementing prior work that studied convex-concave analysis under gradient corruption.

Notation. In the following sections, we use $\| \cdot \|$ to denote the $L _ { 2 }$ norm $\| \cdot \| _ { 2 } , [ N ]$ to denote the set $\{ 1 , 2 , . . . N \}$ and $\Pi _ { \mathcal { X } } ( x )$ to denote the projection of a vector $x$ on a set $\mathcal { X }$ .

# Convex-concave optimization

We consider the following constrained minimax problem

$$
\operatorname* { m i n } _ { x \in \mathcal { X } } \operatorname* { m a x } _ { y \in \mathcal { Y } } f ( x , y ) ,
$$

where $f$ is a convex-concave function, and $\chi$ and $y$ are convex bounded domains. We denote the upper bounds to the radius of domains $\chi$ and $y$ by $D _ { X }$ and $D _ { Y }$ respectively, i.e., $\operatorname* { m a x } _ { x \in { \mathcal { X } } } \| x \| ~ \le ~ D _ { X }$ and $\operatorname* { m a x } _ { y \in \mathcal { y } } \| y \| ~ \le ~ D _ { Y }$ . We assume that gradients of this function are bounded, i.e., $\operatorname* { m a x } _ { x \in \mathcal { X } , y \in \mathcal { y } } \| \nabla _ { x } \mathbf { \bar { f } } ( x , y ) \| \leq G _ { X }$ and $\operatorname* { m a x } _ { x \in \mathcal { X } , y \in \mathcal { y } } \| \nabla _ { y } f ( x , y ) \| \leq$ $G _ { Y }$ , for some constants $G _ { X } , G _ { Y }$ . These assumptions are satisfied in the objective function that we will consider in the sections on perfomrative RL.

# Contamination Model

We are interested in designing a robust optimization method for finding a saddle point of $f$ which only has access to $f$ through noisy (first order) gradient oracles with bounded noise norm. We assume we are given a sampling procedure that generates unbiased gradient samples an $1 - \epsilon$ fraction of the time and adversarial samples (potentially with unbounded corruption) otherwise. This is a strong contamination model known as Huber contamination. We aim to design an optimization method that is robust in the sense that: a) its convergence guarantees have an information theoretically optimal dependence on the noise norm of the inexact gradient oracle, matching our lower bound in 2 and b) it deploys robust gradient estimators that trim the unbounded corruption down to a bounded error.

# Smooth Convex-Concave Objectives

The convex-concave objective that we will study in the sections on performative RL satisfies smoothness. Hence, in this section, we consider a class of convex-concave functions $f$ that are smooth. In particular, we assume that for all $x \in \mathcal { X }$ and $y \in \mathcal { V }$ functions $\nabla _ { x } f ( \cdot , y ) , \nabla _ { x } f ( x , \cdot ) , \nabla _ { y } f ( \cdot , y )$ and $\nabla _ { y } f ( x , \cdot )$ have Lipschitz constants $L _ { X X } , L _ { X Y } , L _ { X Y }$ and $L _ { Y Y }$ respectively, w.r.t. $\ell _ { 2 }$ norm. Similarly to the setting of exact gradient oracles, these smoothness conditions enable us to achieve a better convergence rate.

Robust OFTRL. To find a saddle point of $f$ , we propose a robust version of gradient-based Optimistic Follow the Regularized Leader, as defined in Algorithm 1. The algorithm follows standard OFTRL steps (e.g., see (Orabona 2019)). It alternates between updating $\scriptstyle { \boldsymbol { x } } _ { t }$ and $y _ { t }$ , in optimizing for each a regularized objective. Regularizer $\psi _ { X }$ (resp. $\psi _ { Y } )$ is $\lambda _ { X }$ (resp. $\lambda _ { Y }$ ) strongly convex and bounded over $\chi$ (resp. $\mathcal { V }$ ). For instance $\psi _ { X } ( x )$ could be equal to $\begin{array} { r l } {  { \frac { \lambda _ { X } } { 2 } \| x \| ^ { 2 } } } \end{array}$ and $\psi _ { Y } ( y )$ could be equal to $\textstyle { \frac { \lambda _ { Y } } { 2 } } \| y \| ^ { 2 }$ . Importantly, the algorithm utilizes robust gradient estimates in line 7. While adversarial samples can potentially have unbounded corruption, prior work has shown that there exist robust estimators for the Huber $\epsilon$ -contamination model (e.g. (Diakonikolas, Kane, and Pensia 2020)) that can filter the samples and guarantee that the gradient estimation has a bounded error w.r.t. the true gradient and this error scales with ϵ. For the analysis in the next subsection, we will assume black box access to a (robust) gradient estimation oracle upon query on some point $( x _ { t } , y _ { t } )$ . For the results on performative RL, we provide a problem-specific robust estimator.

# Algorithm 1: Robust OFTRL

1: Initialize $\alpha , b , c > 0$   
2: $( \lambda _ { X } , \lambda _ { Y } ) \gets 3 ( L _ { X X } + L _ { X Y } \alpha + b , L _ { Y Y } + L _ { X Y } / \alpha + c )$   
3: $( g _ { X , 0 } , g _ { Y , 0 } )  ( \mathbf { 0 } , \mathbf { 0 } )$   
4: for $t = 1 , . . . T$ do   
5: $\begin{array} { r l } { \displaystyle } & { \displaystyle x _ { t } \gets \arg \operatorname* { m i n } _ { x \in \mathcal { X } } \psi _ { X } ( x ) + \langle g _ { X , t - 1 } , x \rangle + \underset { i = 1 } { \overset { t - 1 } { \sum } } \langle g _ { X , i } , x \rangle } \\ { \displaystyle y _ { t } \gets \arg \operatorname* { m i n } _ { y \in \mathcal { V } } \psi _ { Y } ( y ) + \langle g _ { Y , t - 1 } , y \rangle + \underset { i = 1 } { \overset { t - 1 } { \sum } } \langle g _ { Y , i } , y \rangle } \end{array}$   
6:   
7: Calculate robust estimations $g _ { X , t }$ and $g _ { Y , t }$ of   
$\nabla _ { x } f ( x _ { t } , y _ { t } )$ and $- \nabla _ { y } f ( x _ { t } , y _ { t } )$ respectively   
8: end for   
$\begin{array} { l } { { \displaystyle 9 \colon ( \bar { x } , \bar { y } )  ( \frac { 1 } { T } \sum _ { t = 1 } ^ { T } x _ { t } , \frac { 1 } { T } \sum _ { t = 1 } ^ { T } y _ { t } ) } } \\ { { \displaystyle 1 0 \colon { \bf R e t u r n } \bar { x } , \bar { y } } } \end{array}$

# Analysis of Robust OFTRL

Next, we analyze the convergence guarantees of Robust OFTRL. Denoting the errors of robust gradient estimation as $\zeta _ { t } ^ { X } = g _ { X , t } - \bar { \nabla } _ { x } f ( x _ { t } , y _ { t } )$ and $\zeta _ { t } ^ { Y } = \overline { { g } } _ { Y , t } - \nabla _ { y } f ( x _ { t } , y _ { t } )$ , we obtain the following result.

Theorem 1. The output $( { \bar { x } } , { \bar { y } } )$ of Algorithm $\boldsymbol { { \mathit { 1 } } }$ satisfies for all $x \in \mathcal { X }$ and $y \in \mathcal { V }$ :

$$
\begin{array} { l } { f \left( \bar { x } , y \right) - f \left( x , \bar { y } \right) \leq \displaystyle \frac { \psi _ { X } \left( x \right) + \psi _ { Y } \left( y \right) } { T } + \frac { \| \nabla _ { x } f \left( x _ { 1 } , y _ { 1 } \right) \| ^ { 2 } } { \lambda _ { X } \cdot T } } \\ { + \frac { \| \nabla _ { y } f \left( x _ { 1 } , y _ { 1 } \right) \| ^ { 2 } } { \lambda _ { Y } \cdot T } + \frac { 6 D _ { X } } { T } \displaystyle \sum _ { t = 1 } ^ { T } \| \zeta _ { t } ^ { X } \| + \frac { 6 D _ { Y } } { T } \displaystyle \sum _ { t = 1 } ^ { T } \| \zeta _ { t } ^ { Y } \| . } \end{array}
$$

The proof of this theorem is based on the results from (Orabona 2019), which consider the exact gradients. We see that the algorithm converges to an approximate saddle point with the $\frac { \mathbf { \nabla } _ { \overline { { 1 } } } } { T }$ convergence rate and the approximation error, i.e., asymptotic duality gap, dependent on the errors of robust gradient estimation. The $\dot { \frac { 1 } { T } }$ convergence rate is optimal for smooth convex-concave problems, as shown in prior work (Ouyang and $\mathtt { X u 2 0 2 1 }$ ). Next, we show that the dependence of the asymptotic duality gap on the gradient noise and domain radius is information theoretically optimal.

Let $\mathcal { A }$ be some deterministic algorithm that estimates saddle points. $\mathcal { A }$ has only access to a noisy gradient oracle of $f$ that can be called $T$ times. At timestep $t$ the algorithm chooses a point $( x _ { t } , y _ { t } )$ and the oracle returns $g _ { x } ( x _ { t } , y _ { t } ) =$ $\nabla _ { x } f ( x _ { t } , y _ { t } ) + \zeta _ { t } ^ { X }$ and $g _ { y } ( x _ { t } , y _ { t } ) = \nabla _ { y } f ( x _ { t } , y _ { t } ) + \zeta _ { t } ^ { Y } .$ . The noise is bounded as follows: $\| \zeta _ { t } ^ { X } \| \leq Z _ { X }$ and $\| \zeta _ { t } ^ { Y } \| \leq$ $Z _ { Y } \forall t \in [ T ]$ . The algorithm has knowledge of the constants $Z _ { X }$ and $Z _ { Y }$ but does not know the exact noise values. In this setting we can derive the following lower bound:

Theorem 2. Consider a deterministic algorithm $\mathcal { A }$ that estimates saddle points of convex concave functions $f ( x , y )$ over the domain $\| x \| \leq D _ { X }$ , $\| y \| \le D _ { Y }$ , where $x$ and $y$ are $d$ -dimensional vectors, using $T$ adaptive queries on noisy gradient oracles with $\| \zeta _ { t } ^ { X } \| \leq Z _ { X }$ and $\| \zeta _ { t } ^ { Y } \| \leq Z _ { Y }$ for all $\textit { t } \in \ [ T ]$ and $Z _ { Y } ~ \le ~ D / 2$ , $Z _ { X } \ \le \ D / 2$ , where $D = m i n \{ D _ { X } , D _ { Y } \}$ . For any such $\mathcal { A }$ :

There exists a convex concave (bilinear) function $f ( x , y )$ and a noise sequence realisation, such that $\mathcal { A }$ returns $a$ point $( x _ { 0 } , y _ { 0 } )$ that has distance at least $\frac { Z _ { X } + Z _ { Y } } { \sqrt { 2 } }$ from any saddle point of $f$ and duality gap $f ( x _ { 0 } , y ) - f ( x , y _ { 0 } ) \geq$ ${ \textstyle \frac { 1 } { 4 } } Z _ { Y } \bar { D _ { Y } } + { \textstyle \frac { 1 } { 4 } } \bar { Z _ { X } } D _ { X }$ for some pair $( x , y )$ inside the domain $\| x \| \leq { \bar { D } } x$ , $\| y \| \leq D _ { Y }$ .

Theorem 1 and Theorem 2 provide a rather complete characterization of the convergence properties of robust OFTRL under corrupted gradients. To the best of our knowledge, these characterization results are novel (see the related work section for comparison to prior work). Moreover, we can prove similar results for Optimistic Mirror Descent Ascent, which is known to have an exponential last iterate convergence rate for objectives that satisfy the Metric Subregularity (MS) condition (Wei et al. 2021). We provide this analysis in the appendix of the extended version of the paper. The latter results are novel and of independent interest for minimax optimization and they could be useful for performative RL if we prove that the Lagrangian (3) satisfies the MS condition or if we simply make it satisfy MS by adding regularization on both variables.

# Formal Setting

The focus of this work is on the performative reinforcement learning framework, introduced by (Mandal, Triantafyllou, and Radanovic 2023).

# Policy-dependent Markov Decision Process

The performative reinforcement learning framework considers a policy-dependent Markov Decision Process (MDP) defined as tuple $M ( \pi ) = ( S , A , P ^ { \pi } , r ^ { \pi } , \rho , \gamma )$ , where: $s$ is a finite state space, $\mathcal { A }$ is a finite action space, $\pi : S \to { \mathcal { P } } ( A )$ is a stochastic policy, $P ^ { \pi } : S \times { \mathcal { A } } \to { \mathcal { P } } ( S )$ is a transition function, with $P ^ { \pi } ( s , a , s ^ { \prime } )$ denoting the probability of transition to state $s ^ { \prime }$ when action $a$ is taken in state s, $r ^ { \pi } : \mathcal { S } \times \mathcal { A }  \mathbb { R }$ is the reward function, $\rho \in { \mathcal { P } } ( S )$ is the initial state distribution, and $\gamma \in [ 0 , 1 )$ is the discount factor. We denote $S = | S |$ and $A = | { \mathcal { A } } |$ , and we assume that rewards are bounded, i.e., $r ^ { \pi } ( s , a ) \leq R$ for some (unknown) constant $R$ .

# Performatively Stable Policy

To define a solution concept in this framework, we define the value of policy $\pi$ in $M ( \pi ^ { \prime } )$ given initial state distribution $\rho$ as $\begin{array} { r } { V _ { \pi ^ { \prime } } ^ { \pi } ( \rho ) = \mathbb E _ { \tau } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \cdot r ^ { \pi ^ { \prime } } ( s _ { t } , a _ { t } ) | \rho \right] } \end{array}$ , where $\tau = ( s _ { 0 } , a _ { 0 } , s _ { 1 } , a _ { 1 } , \ldots ) $ is a trajectory obtained by executing policy $\pi$ in MDP $M ( \pi ^ { \prime } )$ . The solution concept of interest is a performatively stable policy $\pi _ { S }$ , which satisfies: $\pi _ { S } \in \arg \operatorname* { m a x } _ { \pi } V _ { \pi _ { S } } ^ { \pi } ( \rho )$ .

# Occupancy Measures

We denote by $d ^ { \pi } ( s , a )$ the occupancy measure of policy $\pi$ in MDP M (π), i.e., dπ = Eτ [ ∞ γt $d ^ { \pi } = \mathbb { E } _ { \tau } \left[ \sum _ { t = 0 } ^ { \tilde { \infty } } \gamma ^ { \hat { t } } \cdot \mathbb { 1 } \left[ s _ { t } = s , \bar { a } _ { t } = \bar { a } \right] | \rho \right]$ . Occupancy measure $d ^ { \pi }$ satisfies the Bellman flow constraint

$$
\forall s : \rho ( s ) + \gamma \sum _ { s ^ { \prime } , a } d ^ { \pi } ( s ^ { \prime } , a ) P ^ { \pi } ( s ^ { \prime } , a , s ) = \sum _ { a } d ^ { \pi } ( s , a ) .
$$

For a generic $d \geq 0$ , we define policy $\pi ^ { \downarrow d }$ as

$$
\pi ^ { \downarrow d } ( s , a ) = \left\{ \begin{array} { l l } { { \frac { d ( s , a ) } { \sum _ { a ^ { \prime } } d ( s , a ^ { \prime } ) } } } & { { \mathrm { ~ i f ~ } \sum _ { a ^ { \prime } } d ( s , a ^ { \prime } ) > 0 } } \\ { { \frac { 1 } { A } } } & { { \mathrm { ~ o t h w . } } } \end{array} \right. .
$$

If $d$ is a valid occupancy measure in $M ( \pi ^ { \downarrow d } )$ (i.e., if it satisfies the Bellman flow constraints), the occupancy measure of π↓d, i.e., dπ↓d , is equal to d. In general, dπ↓d and d may differ. The occupancy measure of a performatively stable policy is denoted by $d _ { S }$ .

# Data Generation Process

We are interested in a finite sample, offline RL regime. The data generation process is assumed to be i.i.d.: $( s _ { i } , a _ { i } )$ is sampled from normalized $d _ { n }$ , i.e., $( s _ { i } , a _ { i } ) \sim ( 1 - \gamma ) \cdot d _ { n } ^ { }$ $r _ { i } = r _ { n } ( s _ { i } , a _ { i } )$ and $s _ { i + 1 }$ is sampled from $P _ { n }$ transition kernel, i.e., $s _ { i + 1 } \sim P _ { n } ( s _ { i } , a _ { i } , \cdot )$ . We also make a coverage assumption that $d _ { n } ( s , a )$ is positive for all $s \in \mathcal { S } , a \in \mathcal { A }$ . This assumption can be satisfied if the dynamics make all states reachable and we mix some exploratory random policy with $\pi _ { n }$ . In particular, we assume that if $d ( s , a ) \geq c$ then $\forall ( s , a ) \in S \times A , d ^ { \pi ^ { \downarrow d } } ( s , a ) \geq B ( c ) > 0$ . For the rest of this work, all deployed policies will satisfy this condition and thus the coverage assumption will hold with constant $B ( c )$ .

# Contamination Model

We consider the Huber $\epsilon$ -contamination model, where an $\epsilon$ fraction of data points can be corrupted. In this case, both transition and reward can be corrupted. In particular, when the adversary corrupts a sample $( \bar { s _ { i } } , \bar { a _ { i } } , s _ { i } ^ { \prime } , \bar { r _ { i } } )$ , next state $s _ { i } ^ { \prime }$ can be replaced by any $s _ { c } \in \mathcal S$ and reward $\boldsymbol { r } _ { i }$ can be placed by any $r _ { c } \in \mathbb { R }$ . We assume that $s _ { i }$ and $a _ { i }$ are not corrupted. The latter assumption is made for technical simplicity. In the appendix of the extended version of the paper, we show how the assumption could be avoided and still derive similar results with a more complicated analysis.

# Corruption-Robust Performative RL

We follow prior work on performative RL, and study repeated retraining and its convergce to an approximate performatively stable point. In repeated retraining, the policy is retrained after each deployment round. In a canonical setting, we have access to the MDP model $M ( \pi _ { n } )$ , where $\pi _ { n }$ is the policy deployed in round $n$ . We will denote this MDP model by $M _ { n } = M ( \pi _ { n } )$ , and its reward and transition function by $\boldsymbol { r } _ { n }$ and $P _ { n }$ , respectively. Assuming access to $M _ { n }$ , repeated retraining optimizes the following regularized RL objective after each round $n$ :

$$
d _ { n + 1 } ^ { * } \in \arg \operatorname* { m a x } _ { d \geq \mathcal { C } ( M _ { n } ) } \sum _ { s , a } d ( s , a ) \cdot r _ { n } - \frac { \lambda } { 2 } \cdot \left. d \right. _ { 2 } ^ { 2 } ,
$$

where $\mathcal { C } ( M _ { n } )$ is the space of occupancy measures compatible with the MDP $M _ { n }$ , satisfying:

$$
\forall s : \rho ( s ) + \gamma \sum _ { s ^ { \prime } , a } d ( s ^ { \prime } , a ) P _ { n } ( s ^ { \prime } , a , s ) = \sum _ { a } d ( s , a ) .
$$

Note that $\pi _ { n }$ is obtained from $d _ { n } ^ { * }$ , and is defined as $\pi _ { n } : =$ $\pi ^ { \downarrow d _ { n } ^ { * } }$ . To directly optimize from data, Mandal, Triantafyllou, and Radanovic (2023) consider the following minimax optimization problem:

$$
d _ { n + 1 } ^ { * } , h _ { n + 1 } ^ { * } \in \arg \operatorname* { m a x } _ { d \geq 0 } \arg \operatorname* { m i n } _ { h } \mathcal { L } ( d , h , M _ { n } ) ,
$$

where objective $\mathcal { L }$ is the Lagrangian of the regularized RL objective, defined as:

$$
\begin{array} { c l c r } { { \displaystyle { \mathcal { L } } ( d , h , M _ { n } ) = - \frac { \lambda } { 2 } \| d \| _ { 2 } ^ { 2 } + \displaystyle \sum _ { s } h ( s ) \rho ( s ) + \displaystyle \sum _ { s , a } d ( s , a ) \times } } \\ { { \displaystyle \left[ r _ { n } ( s , a ) - h ( s ) + \gamma \displaystyle \sum _ { s ^ { \prime } } P _ { n } ( s , a , s ^ { \prime } ) h ( s ^ { \prime } ) \right] . } } \end{array}
$$

Given dataset $D _ { n }$ containing $m$ samples $( s _ { i } , a _ { i } , s _ { i } ^ { \prime } , r _ { i } )$ , we can replace $\mathcal { L }$ with its empirical version $\hat { \mathcal { L } }$ :

$$
\begin{array} { l } { \displaystyle \hat { \mathcal { L } } ( d , h , M _ { n } ) = - \frac { \lambda } { 2 } \| d \| _ { 2 } ^ { 2 } + \displaystyle \sum _ { s } h ( s ) \rho ( s ) } \\ { + \displaystyle \sum _ { ( s , a , r , s ^ { \prime } ) \in D _ { n } } \frac { d ( s , a ) } { d _ { n } ( s , a ) } \cdot \frac { r - h ( s ) + \gamma \cdot h ( s ^ { \prime } ) } { m \cdot ( 1 - \gamma ) } , } \end{array}
$$

where $m$ is the size of $D _ { n }$ and $d _ { n } ( s , a )$ is the occupancy measure of policy $\pi _ { n }$ in $M _ { n }$ . This allow us to directly optimize from data and establish last-iterate convergence guarantees from finite samples.

Remark 1. In the above framework we assumed for simplicity knowledge of the occupancy measure $d _ { n }$ that generates the samples. In practice, we can estimate it up to arbitrary accuracy from the samples using Monte-Carlo.

# Robust Repeated Retraining

We build on the repeated retraining approach described above, but consider the case where dataset $D _ { n }$ is corrupted, according to the corruption model specified in the formal setting. Our approach is depicted in Algorithm 2.

In each round $n$ , the algorithm first collects a contaminated data $D _ { n }$ by deploying policy $\pi _ { n }$ in $M ( \pi _ { n } )$ . The next step is to approximately solve problem $( 3 ) -$ given that $D _ { n }$ is corrupted directly utilizing $\bar { \hat { \mathcal { L } } }$ instead of $\mathcal { L }$ may not yield any guarantees. Hence, we apply robust OFTR, described in the preliminaries, with $f = { \mathcal { L } }$ and the robust gradient estimators from the next subsection. Finally, the algorithm calculates new policy $\pi _ { n + 1 }$ by mixing the occupancy measure obtained via robust OFTRL with an exploratory random policy and applying Eq. (2). We do this by adding some positive constant $c$ to $\bar { d }$ . The mixing step ensures the coverage property for the next iteration will be satisfied.

In the next subsection, we first propose a novel robust gradient estimator of $\mathcal { L }$ , which can be combined with robust OFTRL to approximately solve (3). We provide guarantees on the estimation error for this estimator that scales with the corruption level $\epsilon$ . We then focus on the convegence analysis of Algorithm 2, and show that it exhibits last-iterate convergence to an approximately stable policy, with the approximation error proportional to $\sqrt { \epsilon }$ .

# Algorithm 2: Robust Repeated Retraining

1: $\pi _ { 0 } = 0$   
2: for $n = 1 , . . . , N$ do   
3: $D _ { n } \gets$ Sample dπn + Huber $\epsilon$ -contamination   
4: $\bar { d } _ { n + 1 } \gets$ Apply Robust OFTRL with $f = { \mathcal { L } }$ and gra  
dient estimators from Section Robust Gradient Esti  
mation on $D _ { n }$   
5: $\tilde { d } _ { n + 1 } \gets \bar { d } _ { n + 1 } + c .$ , where $c > 0$   
6: $\pi _ { n + 1 }  \pi ^ { \downarrow \tilde { d } _ { n + 1 } }$ , where $\pi ^ { \downarrow d }$ is defined in Eq. (2).   
7: end for   
8: Return ${ \tilde { d } } _ { N }$

# Robust Gradient Estimation

We now focus on robust estimation of gradients $g _ { d } : = \begin{array} { l l l } { \begin{array} { r l r l } \end{array} } \end{array}$ $\nabla _ { d } \mathcal { L } ( d , h , M _ { n } )$ and $g _ { h } : = \nabla _ { h } \mathcal { L } ( d , h , M _ { n } )$ . If some of the samples are corrupted, naive averaging may not suffice. Therefore, we explore robust alternatives. We propose the following steps:

• For the gradient w.r.t. $d$ , given a subset $\{ ( s _ { i } , s _ { i } ^ { \prime } , a _ { i } , \bar { r _ { i } } ) | i \in [ \tilde { m } ] \}$ of $D _ { n }$ , with corruption level $\epsilon$ , we apply a robust mean estimator to the dataset $D _ { d } \ : = \ \{ \hat { g } _ { d } ^ { i ^ { - } \bar { | } \ \bar { i } } \ \in \ [ \tilde { m } ] \}$ , where each sample $\hat { g } _ { d } ^ { i }$ is a single-entry $| S | \cdot | A |$ -dimensional vector constructed by sample $( s _ { i } , s _ { i } ^ { \prime } , a _ { i } , r _ { i } )$ according to the formula $\hat { g } _ { d } ^ { i } ( s , a ) ~ = ~ 1 \left[ ( s _ { i } , a _ { i } ) = ( s , a ) \right]$ · $\frac { \gamma h ( s _ { i } ^ { \prime } ) - h ( s _ { i } ) + r _ { i } } { ( 1 - \gamma ) \cdot d _ { n } ( s _ { i } , a _ { i } ) }$ . $D _ { d }$ contains both corrupted and clean samples. Each clean sample $\hat { g } _ { d } ^ { i }$ is an unbiased estimator of $g _ { d } + \lambda d$ . Finally, we add $- \lambda d ( s , a )$ to the robust mean of $D _ { d }$ . • For the gradient w.r.t. $h$ , given a subset $\{ ( s _ { i } , s _ { i } ^ { \prime } , a _ { i } , \bar { r _ { i } } ) | i \in [ \tilde { m } ] \}$ of $D _ { n }$ (disjoint with that used for $g _ { d , \ l }$ , with corruption level $\epsilon$ , we apply a robust mean estimator to the dataset $D _ { h } : = \{ \hat { g } _ { h } ^ { i ^ { - } \bar { | } \bar { i } } \in [ \tilde { m } ] \}$ , where each sample $\hat { g } _ { h } ^ { i }$ is a $\vert { \cal S } \vert$ -dimensional vector constructed by sample $( s _ { i } , s _ { i } ^ { \prime } , \dot { a _ { i } } , \dot { r _ { i } } )$ according to the formula $\begin{array} { r l r } { \hat { g } _ { h } ^ { i } \left( s ^ { \prime } \right) } & { = } & { d ( s _ { i } , a _ { i } ) \frac { \gamma \mathbb { 1 } \left[ s ^ { \prime } = s _ { i } ^ { \prime } \right] - \mathbb { 1 } \left[ s ^ { \prime } = s _ { i } \right] } { ( 1 - \gamma ) \cdot d _ { n } \left( s _ { i } , a _ { i } \right) } } \end{array}$ . Each clean sample $\hat { g } _ { h } ^ { i }$ is an unbiased estimator of $g _ { h } \ - \ \rho$ . Finally, we add $\rho$ to the robust mean of $D _ { h }$ .

1: Input: $\{ \hat { g } _ { d } ^ { i } | i \in [ \tilde { m } ] \}$ , ϵ   
2: for $k = 1 , . . . , S \cdot A$ do   
3: $d a t a \gets \{ \hat { g } _ { d } ^ { 1 } [ k ] , . . . , \hat { g } _ { d } ^ { \tilde { m } } [ k ] \}$   
4: $M e d _ { k } \gets m e d i a n ( d a t a )$   
5: $c l e a n \gets ( 1 - \epsilon ) \cdot \tilde { m }$ closest data entries to $M e d _ { k }$   
6: $\hat { g _ { d } } [ k ] \gets m e a n ( c l e a n )$   
7: end for   
8: Return $\hat { g _ { d } }$

Robust Mean Estimators. As a robust mean estimator we consider any estimator whose error has one (statistical) term, vanishing with the number $\tilde { m }$ of samples and one bias term polynomial to the frequency $\epsilon$ of corrupted samples. The error of a robust estimator should not scale with the magnitude of corruption, especially when corruption can be unbounded, as it can happen in the reward samples in our setting. For the estimation of $g _ { d }$ , we use Algorithm 3. For the estimation of $g _ { h }$ it suffices to apply naive averaging to achieve the kind of result that we wish. The errors of gradient estimators are analysed in the following theorem.

Theorem 3. Let us use $\hat { g } _ { h } = \textstyle { \frac { 1 } { \tilde { m } } } \sum _ { i = 1 } ^ { \tilde { m } } \hat { g } _ { h } ^ { i }$ to estimate $g _ { h }$ and Algorithm $3$ to estimate $g _ { d } ,$ , and assume that the corruption level in the respective datasets is bounded by $\epsilon < 0 . 5$ . Then with probability at least $1 - \delta$ the estimation errors satisfy the following guarantees:

$$
\begin{array} { r l } & { \| \widehat { g } _ { h } - g _ { h } \| _ { 1 } \leq \underbrace { \frac { 4 } { ( 1 - \gamma ) ^ { 2 } B ( c ) } \left( \frac { \sqrt { S \log ( 4 S / \delta ) } } { \sqrt { \tilde { m } } } + \epsilon \right) } _ { E _ { 1 } ( \tilde { m } , \epsilon , \delta ) } , } \\ & { \| \widehat { g } _ { d } - g _ { d } \| _ { 2 } \leq \underbrace { 6 \sqrt { S A } \frac { 2 h _ { m a x } + R } { ( 1 - \gamma ) B ( c ) } \left( \frac { \sqrt { 2 \log \left( \frac { 4 S A } { \delta } \right) } } { \sqrt { \tilde { m } } } + 2 \epsilon \right) } _ { E _ { 2 } ( \tilde { m } , \epsilon , \delta ) } . } \end{array}
$$

# Convergence Analysis

Our goal is to show that the repeated optimization approach as specified in Algorithm 2, outputs a solution that is approximately stable. We restrict the domain of variables $d$ and $h$ in Algorithm 2 as follows: $\textstyle { \mathcal { D } } = \{ d : 0 \leq d ( s , a ) \leq { \frac { 1 } { 1 - \gamma } } \}$ and $\mathcal { H } = \{ h : - h _ { m a x } \leq h ( s , a ) \leq h _ { m a x } \}$ , where $h _ { m a x } > 0$ . Furthermore, we will assume that $D _ { n }$ has a large enough number $m$ of samples and a bounded corruption level, as specified by the following assumption, whose role we explain in the next paragraph.

Assumption 1. (bounded corruption) Every $D _ { n }$ can be split in $2 T$ batches, each having corruption level of at most $\epsilon < 0$ .

Robust OFTRL Guarantees. To prove the convergence of Algorithm 2, we first need to provide an upper bound on the quality of the solution that robust OFTRL (Algorithm 1) outputs. We show that after sufficiently many iterations $T$ , robust OFTRL with gradient estimators defined in the previous section can find an approximate saddle point to (3), but with bounded domains $\mathcal { D }$ and $\mathcal { H }$ . Similarly to (Prasad et al. 2020), to avoid statistical issues, we split the original dataset of $m$ samples in $2 T$ equal batches, assuming that each batch has corruption level at most $\epsilon < 0 . 5$ . In each iteration, we apply each gradient estimator on a fresh batch. We refer to this process as batch-splitting.

Lemma 1. There exists $T$ such that the output of Algorithm $\jmath$ run for $T$ iterations on $f = { \mathcal { L } }$ , with $\boldsymbol { \mathcal { X } } = \boldsymbol { \mathcal { D } }$ , $\mathscr { y } = \mathscr { H }$ , $g _ { X , t } = \hat { g } _ { h }$ , $g _ { Y , t } = \hat { g } _ { d } ,$ , and batch-splitting, satisfies

$$
\operatorname* { m a x } _ { d \in \mathcal { D } } \mathcal { L } ( d , \bar { h } , M _ { n } ) - \operatorname* { m i n } _ { h \in \mathcal { H } } \mathcal { L } ( \bar { d } , h , M _ { n } ) \leq 7 C ( \delta )
$$

under Assumption $\boldsymbol { { l } }$ , with probability at least $1 - \delta$ , where

$$
\begin{array} { r } { C ( \delta ) : = \sqrt { S } \Bigg ( E _ { 1 } \left( \frac { m } { 2 T } , \epsilon , \frac { \delta } { T } \right) h _ { m a x } + \frac { \sqrt { A } E _ { 2 } \left( \frac { m } { 2 T } , \epsilon , \frac { \delta } { T } \right) } { 1 - \gamma } \Bigg ) . } \end{array}
$$

Now, we want to provide a bound on the quality of the output of robust OFTRL w.r.t. the true solution of (3). Consider the set of $d \geq 0$ that satisfy the Bellman flow constraint in $M _ { n }$ and denote its Hoffman constant (Garber 2019) by $\sigma _ { n }$ . To simplify the exposition, we define quantity $\alpha ( M _ { n } , \delta )$ :

$$
\alpha ( M _ { n } , \delta ) = \sqrt { \frac { 1 4 C ( \delta ) } { \lambda } } + C _ { n } ^ { \prime } + \sqrt { 2 C _ { n } ^ { \prime } \left( \frac { \| r _ { n } \| } { \lambda } + \frac { 1 } { 1 - \gamma } \right) }
$$

where C′ = ∥rn∥2 Sσn−1/2 . For a generic MDP $M ( \pi )$ , we analogously define $\overset { \cdot } { \alpha } ( M ( \pi ) , \delta )$ . Next we show that robust OFTRL outputs an approximately optimal solution to (3).

Theorem 4. Consider the robust OFTRL from Lemma $I$ , and assume its number of iterations $T$ is s.t. (4) is holds. Under Assumption $\boldsymbol { { l } }$ , the output of robust OFTRL $\bar { d }$ satisfies $\lVert d _ { n } ^ { * } - \bar { d } \rVert _ { 2 } \leq \alpha ( M _ { n } , \delta )$ with probability at least $1 - \delta$ .

Convergence of Algorithm 2. Now, we are ready to derive convergence guarantees of Algorithm 2. To do so, we need two additional assumptions, which we use to establish contraction properties of repeated retraining. The first one, $\epsilon$ -sensitivity is a standard in the literature on performative prediction, and we take it from prior work on performative RL (Mandal, Triantafyllou, and Radanovic 2023).

Assumption 2. (ϵ˜-sensitivity) For any two MDPs $M ( \pi )$ and $M ( \pi ^ { \prime } )$ , the following holds $\| r ^ { \pi } - r ^ { \pi ^ { \prime } } \| _ { 2 } \le \tilde { \epsilon } _ { r } \| d ^ { \pi } - d ^ { \pi ^ { \prime } } \| _ { 2 }$ and $\| P ^ { \pi } - P ^ { \pi ^ { \prime } } \| _ { 2 } \le \tilde { \epsilon } _ { p } \| d ^ { \pi } - d ^ { \pi ^ { \prime } } \| _ { 2 }$ .

The second one, is a rather weak assumption requiring that any MDP induced by the deployed policy does not have an infinite factor $\alpha$ for fixed $\delta$ .

Assumption 3. ( $\bar { \alpha }$ -boundedness) For any MDP $M ( \pi )$ induced by stationary policy $\pi$ we have $\alpha ( M ( \pi ) , \delta ) \leq { \bar { \alpha } } ( \delta )$ .

Note that after each round $n$ , the distance between $d _ { n }$ and $d _ { n } ^ { * }$ is at most $\bar { C } ( \delta ) : = \bar { \alpha } ( \delta ) + c \sqrt { S A }$ . This is due to the definition of ${ \tilde { d } } _ { n }$ —we obtain $\tilde { d }$ from $\bar { d }$ by adding a constant $c > 0$ to each of its entry — and the robust OFTRL guarantees for $\bar { d }$ . Using this bound and the previous two assumptions we derive a convergence result for Algorithm 2.

Theorem 5. (Informal Statement) Under Assumption $\jmath$ , Assumption 2 and Assumption 3, there exist $\lambda$ and $N$ such that the output of Algorithm 2 satisfies $\tilde { d } _ { N } \in \{ d \in \mathcal { D } :$ $\| d - d _ { S } \| _ { 2 } \le \tilde { C } : = 4 \cdot \bar { C } ( \delta / N ) \}$ with probability at least $1 - \delta$ , where $d _ { S }$ is a performatively stable policy.

Z=10 Z=10 = 0.005 = 0.005 20 Z=1000 100 Z=1000 1 = 0.015 1.0 = 0.015 = 0.1 = 0.1 0 0 10 > 20 0 0 10 20 0 0 10 20 0.0 0 10 20 Iteration n Iteration n Iteration n Iteration n (a) Fixed ϵ, naive (b) Fixed ϵ, robust (c) Fixed Z, naive (d) Fixed $Z$ , robust

Theorem 5 states that Robust Repeated Retraining (Algorithm 2) exhibits last-iterate convergence to an approximate stable point. Ignoring $\epsilon$ -independent additive terms, the approximation error is proportional $\sqrt { \epsilon }$ . Namely, factor $\bar { C } ( \delta / N )$ depends on $\bar { \alpha } ( \delta )$ , which has square root dependence on $\epsilon$ since $C$ is linear in $\epsilon$ . In prior work on corruptionrobust offline RL, suboptimality gaps can scale as $O ( \sqrt { \epsilon } )$ or $O ( \epsilon )$ , depending on the exact setting and assumption (e.g., see (Zhang et al. 2022; Nika et al. 2024)). However, we note that our setting is not directly comparable, since it combines offline RL with repeated retraining. We leave the analysis of the tightness of this bound for future work.

The result in Theorem 5 provides an asymptotic convergence guarantee w.r.t. $\lambda$ and $N$ . Following the proof of Theorem 1 from (Mandal, Triantafyllou, and Radanovic 2023), it is easy to show that λ > λ0 and N > 1 1λ · $\begin{array} { r } { N > \frac { 1 } { 1 - \lambda _ { 0 } } \cdot \log \bigl ( \frac { 2 } { \bar { C } _ { 0 } \cdot ( 1 - \gamma ) } \bigr ) } \end{array}$ guarantee the convergence.1 Here, λ0 = 24S3/(21(2γϵ˜r4)+5ϵ˜p) , while $\bar { C } _ { 0 }$ is a lower bound on $4 \cdot { \bar { C } } ( \delta / N )$ independent of $N$ , but possibly dependent on $\lambda$ .

We can further assess the return that policy defined by ${ \tilde { d } } _ { N }$ , i.e., $\pi ^ { \downarrow \tilde { d } _ { N } }$ , achieves in $\mathcal { M } ( \pi _ { N } )$ . Assuming that the initial state distribution has a full support over state space, one can show that this return is comparable to the return of a performatively stable policy $\pi _ { S }$ in $M ( \pi _ { S } )$ : it is worse by at most an instance-specific constant proportional to $\tilde { C }$ .

# Experimental Evaluation

In this section we experimentally test the efficacy of our approach in performative RL under corruption.

Environment. Our MDP model is a $W \times W$ gridworld environment, inspired by the gridworld environment in (Triantafyllou, Singla, and Radanovic 2021), where state $s =$ $( i , j )$ encodes the location/cell that the agent occupies. In round $n$ , the reward function is defined as $\bar { r } _ { n } ( s , a ) \stackrel { - } { = } R _ { i , j } -$ $c _ { p } \cdot \sum _ { a \in \mathcal { A } } d _ { n } ( i \cdot W + j , a )$ . We set $W = 8$ , while the values $R _ { i , j }$ are defined as in the gridworld environment used in (Triantafyllou, Singla, and Radanovic 2021). The transitions are deterministic and only controlled by the four agent’s actions (left, right, up, down). Samples are transition tuples $( s _ { i } , s _ { i } ^ { \prime } , a _ { i } , r _ { i } )$ . In corrupted samples we add Gaussian noise $N ( Z , 0 . 5 )$ to $\boldsymbol { r } _ { i }$ and we replace $s _ { i } ^ { \prime }$ with a random state $s ^ { \prime }$ with probability exponentially decreasing with the distance between $s ^ { \prime }$ and $s _ { i } ^ { \prime }$ on the grid. We study the convergence of the robust repeated retraining based on OFTRL (Algorithm 2). As a baseline approach, we consider a version of repeated retraining based on OFTRL which uses a naive estimator of $g _ { d }$ instead of Algorithm 3. In all the experiments, we set $\gamma = 0 . 9 9 , c _ { p } = 1$ and $\lambda = 0 . 0 0 1$ . To create transition samples, we collect 1000 trajectories with an effective horizon of $1 / ( 1 - \gamma ) = 1 0 0$ .

Results. The plots in Fig. 1 show the convergence results for different values of the noise magnitude $Z$ and the corruption frequency $\epsilon$ . We observe that naive gradient estimation results in more noisy convergence of repeated retraining, compared to robust gradient estimation. The effect is stronger when we fix $\epsilon$ and progressively increase $Z$ . Then the curve of repeated retraining with a naive estimator oscillates with magnitude scaling with $Z$ , while the curve of repeated retraining with a robust estimator stays virtually unaffected. Fixing $Z$ and progressively increasing $\epsilon$ we see that both robust and naive retraining are affected, with the error increasing with $\epsilon$ . All these effects are in agreement with our theoretical results and they showcase the utility of robust gradient estimation.

# Conclusion

We considered performative reinforcement learning under corrupted data. We introduced a repeated retraining approach and showed that it converges to an approximately stable policy, where the approximation error depends on the level of corruption. One of the most interesting future research directions is to investigate the tightness of the approximation error. Extending this work to RL settings with function approximation is another important avenue for future research.