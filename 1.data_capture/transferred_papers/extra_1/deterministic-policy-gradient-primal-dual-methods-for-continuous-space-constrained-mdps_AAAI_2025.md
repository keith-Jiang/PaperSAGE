# Deterministic Policy Gradient Primal-Dual Methods for Continuous-Space Constrained MDPs

Sergio Rozada 1, Dongsheng Ding 2, Antonio G. Marques 1, Alejandro Ribeiro 2

1Dept. of Signal Theory and Communications, King Juan Carlos University 2Dept. of Electrical and Systems Engineering, University of Pennsylvania s.rozada.2019 $@$ alumnos.urjc.es, dongshed@seas.upenn.edu, antonio.garcia.marques $@$ urjc.es, alejandro.ribeiro $@$ seas.upenn.edu

# Abstract

We study the problem of computing deterministic optimal policies for constrained Markov decision processes (MDPs) with continuous state and action spaces, which are widely encountered in constrained dynamical systems. Designing deterministic policy gradient methods in continuous state and action spaces is particularly challenging due to the lack of enumerable state-action pairs and the adoption of deterministic policies, hindering the application of existing policy gradient methods. To this end, we develop a deterministic policy gradient primal-dual method to find an optimal deterministic policy with non-asymptotic convergence. Specifically, we leverage regularization of the Lagrangian of the constrained MDP to propose a deterministic policy gradient primal-dual (D-PGPD) algorithm that updates the deterministic policy via a quadraticregularized gradient ascent step and the dual variable via a quadratic-regularized gradient descent step. We prove that the primal-dual iterates of D-PGPD converge at a sub-linear rate to an optimal regularized primal-dual pair. We instantiate D-PGPD with function approximation and prove that the primal-dual iterates of D-PGPD converge at a sub-linear rate to an optimal regularized primal-dual pair, up to a function approximation error. Furthermore, we demonstrate the effectiveness of our method in two continuous control problems: robot navigation and fluid control. This appears to be the first work that proposes a deterministic policy search method for continuous-space constrained MDPs.

# Code — https://github.com/sergiorozada12/d-pg-pd Extended version — https://arxiv.org/abs/2408.10015

tion (Paternain et al. 2022), video compression (Mandhane et al. 2022), and finance (Chow et al. 2018).

This paper is motivated by two observations. First, continuous state-action spaces are pervasive in dynamical systems, yet most methods in constrained RL are designed for discrete state and/or action spaces (Borkar 2005; Efroni, Mannor, and Pirotta 2020; Ding et al. 2022; Singh, Gupta, and Shroff 2022). Second, the literature on constrained RL largely focuses on stochastic policies. However, randomly taking actions by following a stochastic policy is often prohibitive in practice, especially in safety-critical domains (Sehnke et al. 2010; Li et al. 2022; Gao et al. 2023). Deterministic policies alleviate such concerns, but (i) they might lead to sub-optimal solutions (Ross 1989; Altman 2021); and (ii) computing them is NP-complete (Feinberg 2000; Dolgov 2005). Nevertheless, there is a rich body of constrained control literature that studies problems where optimal policies are deterministic (Posa, Kuindersma, and Tedrake 2016; Tsiamis et al. 2020; Zhao and You 2021; Ma et al. 2022). Viewing this gap, we study the problem of finding optimal deterministic policies for constrained MDPs with continuous state-action spaces.

A key consideration of this paper is the fact that deterministic policies are sub-optimal in finite state-action spaces, but sufficient for constrained MDPs with continuous state-action spaces (Feinberg and Piunovskiy 2002, 2019). This enables our formulation of a constrained RL problem with deterministic policies. To develop a tractable deterministic policy search method, we introduce a regularized Lagrangian approach that leverages proximal optimization methods. Moreover, we use function approximation to ensure scalability in continuous state-action spaces. Our main contribution is four-fold.

# 1 Introduction

Constrained Markov decision processes (MDPs) are a standard framework for incorporating system specifications into dynamical systems (Altman 2021; Brunke et al. 2022). In recent years, constrained MDPs have attracted significant attention in constrained Reinforcement Learning (RL), whose goal is to derive optimal control policies through interaction with unknown dynamical systems (Achiam et al. 2017; Tessler, Mankowitz, and Mannor 2018). Policy gradient-based constrained learning methods have become the workhorse driving recent successes across various disciplines, e.g., naviga

(i) We introduce a deterministic policy constrained RL problem for a constrained MDP with continuous state-action spaces and prove that the problem exhibits zero duality gap, despite being constrained to deterministic policies.   
(ii) We propose a regularized deterministic policy gradient primal-dual (D-PGPD) algorithm that updates the primal policy via a proximal-point-type step and the dual variable via a gradient descent step, and we prove that the primaldual iterates of D-PGPD converge to a set of regularized optimal primal-dual pairs at a sub-linear rate.   
(iii) We propose an approximation for D-PGPD by including function approximation. We prove that the primal-dual

iterates of the approximated D-PGPD converge at a sublinear rate, up to a function approximation error.

(iv) We demonstrate that D-PGPD addresses the classical constrained navigation problem involving several types of cost functions and constraints. We show that D-PGPD can solve non-linear fluid control problems under constraints.

Related work. Deterministic policy search has been studied in the context of unconstrained MDPs (Silver et al. 2014; Lillicrap et al. 2015; Kumar et al. 2020; Lan 2022). In constrained setups, however, deterministic policies have been largely restricted to occupancy measure optimization in finite state-action spaces (Dolgov 2005) or are embedded in hyperpolicies (Sehnke et al. 2010; Montenegro et al. 2024a,b). This work extends deterministic policy search to constrained MDPs with continuous state-action spaces, overcoming two main roadblocks: the sub-optimality of deterministic policies and the NP-completeness of computing them (Ross 1989; Feinberg 2000; Dolgov 2005; Altman 2021; McMahan 2024). First, we show that deterministic policies are sufficient for constrained MDPs in continuous state-action spaces (Feinberg and Piunovskiy 2002, 2019), leveraging the convexity of the value image to establish strong duality in the deterministic policy space. Second, we overcome computational intractability by introducing a quadratic regularization of the reward function and proposing a regularization-based primaldual algorithm. This algorithm exploits the structure of value functions and achieves last-iterate convergence to an optimal deterministic policy. While last-iterate convergence of primal-dual algorithms has been explored in constrained RL (Moskovitz et al. 2023; Ding et al. 2024; Ding, Huan, and Ribeiro 2024), existing methods focus on stochastic policies and finite-action spaces. In control, extensive work addresses deterministic policies in constrained setups with continuous state-action spaces (Scokaert and Rawlings 1998; Lim and Zhou 1999). However, these approaches are typically model-based and tailored to specific structured problems (Posa, Kuindersma, and Tedrake 2016; Tsiamis et al. 2020; Zhao, You, and Ba¸sar 2021; Zhao and You 2021; Ma et al. 2022). Bridging constrained control and RL has also been explored (Kakade et al. 2020; Zahavy et al. 2021; Li et al. 2023), but these methods remain model-based and focus on stochastic policies. In contrast, we propose a model-free deterministic policy search method for constrained MDPs with continuous state-action spaces.

# 2 Preliminaries

We consider a discounted constrained MDP, denoted by the tuple $( S , A , p , r , u , b , \gamma , \rho )$ . Here, $S \subseteq \mathbb { R } ^ { d _ { s } }$ and $A \subseteq \mathbb { R } ^ { \bar { d _ { a } } }$ are continuous state-action spaces with dimensions $d _ { s }$ and $d _ { a }$ , and bounded actions $\| a \| \leq A _ { \operatorname* { m a x } }$ for all $a \in A ; p ( \cdot | s , a )$ is a probability measure over $S$ parametrized by the state-action pairs $( s , a ) \in S \times A ; r , u \colon S \times A \mapsto [ 0 , 1 ]$ are reward/utility functions; $b$ is a constraint threshold; $\gamma \in [ 0 , 1 )$ is a discount factor; and $\rho$ is a probability measure that specifies an initial state. We consider the set of all deterministic policies $\pi$ in which a policy $\pi \colon S \mapsto A$ maps states to actions. The transition $p$ , the initial state distribution $\rho$ , and the policy π define a distribution over trajectories {st, at, rt, ut}t∞=0, where $s _ { 0 } \sim \rho , a _ { t } = \pi ( s _ { t } ) , r _ { t } = r ( s _ { t } , a _ { t } ) , u _ { t } = u ( s _ { t } , a _ { t } )$ and $s _ { t + 1 } \sim p ( \cdot | s _ { t } , a _ { t } )$ . Given $\pi$ , we define the value function $V _ { r } ^ { \pi } \colon S  \mathbb { R }$ as the expected sum of discounted rewards

$$
V _ { r } ^ { \pi } ( s ) : = \mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ( s _ { t } , a _ { t } ) \mid s _ { 0 } = s \right] .
$$

For the utility function, we define the corresponding value function $V _ { u } ^ { \pi }$ . Their expected values over the initial state distribution $\rho$ are denoted as $V _ { r } ( \pi ) : = \mathbb { E } _ { \rho } [ V _ { r } ^ { \pi } ( s ) ]$ and $V _ { u } ( \pi ) : =$ $\mathbb { E } _ { \rho } [ V _ { u } ^ { \pi } ( s ) ]$ , where we drop the dependence on $\rho$ for simplicity of notation. Boundedness of $r$ and $u$ leads to $V _ { r } ( \pi )$ , $V _ { u } ( \pi ) \in$ $\left[ 0 , 1 / ( 1 - \gamma ) \right]$ . We introduce a discounted state visitation distribution $\begin{array} { r } { \dot { d } _ { s _ { 0 } } ^ { \pi } ( B ) : = ( 1 - \gamma ) \sum _ { t = 0 } ^ { \infty } \operatorname* { P r } ( s _ { t } \in B \mid \pi , s _ { 0 } ) } \end{array}$ for any $B \subseteq S$ and define $d _ { \rho } ^ { \pi } ( s ) : = \mathbb { E } _ { s _ { 0 } \sim \rho } [ d _ { s _ { 0 } } ^ { \pi } ( s ) ]$ . For the reward function $r$ , we define the state-action value function $Q _ { r } ^ { \pi } \colon S \times A \to \mathbb { R }$ given an initial action $a$ while following $\pi$ ,

$$
Q _ { r } ^ { \pi } ( s , a ) : = \mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ( s _ { t } , a _ { t } ) | s _ { 0 } = s , a _ { 0 } = a \right] .
$$

We let the associated advantage function $A _ { r } ^ { \pi } \colon S \times A \to  { \mathbb { R } }$ be $A _ { r } ^ { \pi } ( s , a ) : = Q _ { r } ^ { \pi } ( s , a ) - V _ { r } ^ { \pi } ( s )$ . Similarly, we define $Q _ { u } ^ { \pi }$ : $S \times A \to \mathbb { R }$ and $A _ { u } ^ { \pi } \colon S \times A \to  { \mathbb { R } }$ for the utility function $u$ .

A policy is optimal for a given reward function when it maximizes the corresponding value function. However, the value functions $V _ { r } ( \pi )$ and $V _ { u } ( \pi )$ are usually in conflict, e.g., a policy that maximizes $V _ { r } ( \pi )$ is not necessary good for $\bar { V _ { u } ( \pi ) }$ . To trade off two conflicting objectives, constrained MDP aims to find an optimal policy $\pi ^ { \star }$ that maximizes the reward value function $\mathrm { \bar { \it V } } _ { r } ( \pi )$ subject to an inequality constraint on the utility value function $V _ { u } ( \pi ) \geq b$ , where we assume $b \in ( 0 , 1 / ( \overset { \cdot } { 1 } - \gamma ) ]$ to avoid trivial solutions. We use a single constraint for the sake of simplicity, but our results extend to problems with multiple constraints. We translate the constraint $V _ { u } ( \pi ) \geq b$ into the constraint $V _ { g } ( \pi ) \geq 0$ for $g : = u - ( 1 - \gamma ) \dot { b }$ , where $g$ : $S \times A \mapsto [ - 1 , 1 ]$ denotes the translated utility. This leads to the following problem

$$
\begin{array} { r l } { \underset { \pi \in \Pi } { \operatorname* { m a x } } } & { { } ~ V _ { r } ( \pi ) } \\ { \mathrm { s . ~ t . ~ } } & { { } ~ V _ { g } ( \pi ) ~ \geq ~ 0 . } \end{array}
$$

Restricting Problem (1) to deterministic policies poses several challenges. Deterministic policies can be sub-optimal in constrained MDPs with finite state-action spaces (Ross 1989; Altman 2021), and when they exist, finding them is a NP-complete (Feinberg 2000). Problem (1) is non-convex in the policy but can be reformulated as a linear program using occupancy measures with stochastic policies (Paternain et al. 2019). However, the occupancy measure representation of (1) is a non-linear and non-convex problem when only deterministic policies are considered (Dolgov 2005). Finally, multiple policies can achieve the optimal value function $V _ { P } ^ { \pi ^ { \star } }$ while satisfying the constraint. We denote the set of all maximizers of (1) that attain $V _ { P } ^ { \pi ^ { \star } }$ as $\Pi ^ { \star }$ . To address these points, we observe that deterministic policies are sufficient in constrained MDPs with continuous state-action spaces under the following assumption (Feinberg and Piunovskiy 2002, 2019).

Assumption 1 (Non-atomicity). The MDP is non-atomic, i.e., $\rho ( s ) = { \bar { 0 } }$ and $p ( s ^ { \prime } \mid s , a ) = 0$ for all $s$ , $s ^ { \prime } \in S$ and $a \in A$ .

Assumption 1 is mild in practice. Since stochastic perturbations are common in physical systems with continuous state and action spaces (Anderson and Moore 2007), the probability measures $\rho$ and $p ( \cdot \mid s , a )$ are normally atomless, i.e., for any measurable set $B \subseteq S$ with probability measures $\rho ( B )$ and $p ( B \mid s , a )$ , there exists a measurable subset $B ^ { \prime } \subset B$ that has smaller non-zero probability measures $\rho ( B ) > \rho ( B ^ { \prime } ) > 0$ and $p ( B \mid s , a ) > p ( B ^ { \prime } \mid s , a ) > 0$ for any $s \in S$ and $a \in A$ . In other words, the transition probability and the initial probability do not concentrate in a single state (Feinberg and Piunovskiy 2019). When a constrained MDP is non-atomic, only considering deterministic policies is sufficient (Feinberg and Piunovskiy 2019). Specifically, let $V ( \pi ) : = [ V _ { r } ( \pi ) ^ { \smile } V _ { g } ( \pi ) ] ^ { \top }$ denote the vector of value functions for a given policy $\pi$ . We define a deterministic value image $\mathcal { V } _ { D } : = \{ V ( \pi ) | \pi \in \Pi \}$ , which is a set of all attainable vector value functions for deterministic policies. We denote by $\nu _ { T }$ a value image for all policies. The deterministic value image $\nu _ { D }$ and the value image $\nu _ { T }$ are equivalent under Assumption 1 for discounted MDPs (see Lemmas 2 and 4 in Appendix B). Therefore, the optimal value function of a non-atomic constrained MDP is contained in the deterministic value image $\nu _ { D }$ . Furthermore, the deterministic value image $\gamma _ { D }$ is a convex set, even though each value function $V ( \pi ) \in \mathcal { V } _ { D }$ is non-convex in policy $\pi$ (see Lemmas 2 and 3 in Appendix B). These observations are summarized below.

Lemma 1 (Sufficiency of deterministic policies). For a nonatomic discounted MDP, the deterministic value image $\nu _ { D }$ is convex, and equals the value image $\nu _ { T }$ , i.e., $\nu _ { D } = \nu _ { T }$ .

# 2.1 Zero Duality Gap

With the convexity of the deterministic value image $\nu _ { D }$ in hand, we next establish zero duality gap for Problem (1). We begin with a standard feasibility assumption.

Assumption 2 (Feasibility). There exists a deterministic policy $\tilde { \pi } \in \Pi$ and $\xi > 0$ such that $V _ { g } ( \tilde { \pi } ) \geq \xi$ .

We dualize the constraint by introducing the dual variable $\lambda \in \mathbb { R } ^ { + }$ and the Lagrangian $\begin{array} { r } { L ( \pi , \lambda ) : = V _ { r } ( \pi ) + \lambda V _ { g } ( \pi ) } \end{array}$ . For a fixed $\lambda$ , let $\Pi ( \lambda )$ be the set of Lagrangian maximizers. The Lagrangian $L ( \pi , \lambda )$ is equivalent to the value function $V _ { \lambda } ( \pi )$ associated with the combined reward/utility function $r _ { \lambda } ( \dot { s } , a ) = r ( s , a ) + \lambda g ( s , a )$ . The dual function $D ( \boldsymbol { \lambda } ) : =$ $\operatorname* { m a x } _ { \pi \in \Pi } V _ { \lambda } ( \pi )$ is an upper bound of Problem (1), and the dual problem searches for the tightest primal upper bound

$$
\operatorname* { m i n } _ { \lambda \in \mathbb { R } ^ { + } } ~ D ( \lambda ) .
$$

We denote by $V _ { D } ^ { \lambda ^ { \star } }$ the optimal value of the dual function, where $\lambda ^ { \star }$ is a minimizer of the dual Problem (2). Despite being non-convex in the policy, if we replace the deterministic policy space in Problem (1) with the stochastic policy space, then it is known that Problem (1) has zero duality gap (Paternain et al. 2019). The proof capitalizes on the convexity of the occupancy measure representation of (1) for stochastic policies. However, this occupancy-measure-based argument does not carry to deterministic policies, since the occupancy measure representation of Problem (1) is non-convex when only deterministic policies are used (Dolgov 2005). Instead, we leverage the convexity of the deterministic value image $\nu _ { D }$ to prove that strong duality holds for Problem (1); see Appendices A and C.2 for more details and the proof.

Theorem 1 (Zero duality gap). Let Assumption 1 hold. Then, Problem (1) has zero duality gap, i.e., $V _ { P } ^ { \pi ^ { \star } } = V _ { D } ^ { \lambda ^ { \star } }$ .

Theorem 1 states that the optimal values of Problems (1) and (2) are equivalent, extending the zero duality gap result in (Paternain et al. 2019) to deterministic policies under the nonatomicity assumption. However, recovering an optimal policy $\pi ^ { \star }$ can be non-trivial even if an optimal dual variable $\lambda ^ { \star }$ is obtained from the dual problem (Zahavy et al. 2021). The root cause is that the maximizers of the primal problem $\Pi ^ { \star }$ and those of the Lagrangian for an optimal multiplier $\Pi ( { \lambda } ^ { \star } )$ are different sets (Calvo-Fullana et al. 2023, Proposition 1). To address this, we employ Theorem 1 to interpret Problem (1) as a saddle point problem. Zero duality gap implies that an optimal primal-dual pair $( \pi ^ { \star } , \lambda ^ { \star } )$ is a saddle point of the Lagrangian $L ( \pi , \lambda )$ , and satisfies the mini-max condition

$L ( \pi , \lambda ^ { \star } ) \ \leq \ L ( \pi ^ { \star } , \lambda ^ { \star } ) \ \leq \ L ( \pi ^ { \star } , \lambda ) \quad \forall ( \pi , \lambda ) \in \Pi \times \Lambda _ { 1 }$ , where $\lambda$ is bounded in the interval $\Lambda : = ~ [ 0 , \lambda _ { \operatorname* { m a x } } ]$ , with $\lambda _ { \operatorname* { m a x } } : = 1 / ( ( 1 - \gamma ) \xi )$ ; see Lemma 9 in Appendix B. In this paper, we refer to saddle points that satisfy the mini-max condition for all pairs $( \pi , \lambda ) \in \Pi \times \Lambda$ as global saddle points. Our main task in Section 3 is to find a global saddle point of the Lagrangian $L ( \pi , \lambda )$ that is a solution to Problem (1).

# 2.2 Constrained Regulation Problem

We illustrate Problem (1) using the following example

$$
\begin{array} { r l } & { \displaystyle \underset { \pi \in \Pi } { \operatorname* { m a x } } ~ \mathbb { E } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \left( s _ { t } ^ { \top } G _ { 1 } s _ { t } + a _ { t } ^ { \top } R _ { 1 } a _ { t } \right) \right] } \\ & { ~ \mathrm { s . t . } ~ \mathbb { E } \left[ \displaystyle \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \left( s _ { t } ^ { \top } G _ { 2 } s _ { t } + a _ { t } ^ { \top } R _ { 2 } a _ { t } \right) \right] \geq b } \end{array}
$$

$$
- b _ { s } \leq C _ { s } s _ { t } \leq b _ { s } , - b _ { a } \leq C _ { a } a _ { t } \leq b _ { a }
$$

$$
s _ { t + 1 } = B _ { 0 } s _ { t } + B _ { 1 } a _ { t } + \omega _ { t } , \ s _ { 0 } \sim \rho
$$

where $B _ { 0 } \in \mathbb { R } ^ { d _ { s } \times d _ { s } }$ and $B _ { 1 } \in \mathbb { R } ^ { d _ { s } \times d _ { a } }$ denote the system dynamics, $\omega _ { t }$ is the standard Gaussian noise, $\rho$ is the initial state distribution, and $G _ { 1 }$ , $G _ { 2 } \in \mathbb { R } ^ { d _ { s } \times d _ { s } }$ and $R _ { 1 }$ , $R _ { 2 } \in \mathbb { R } ^ { d _ { a } \times d _ { a } }$ are negative semi-definite reward matrices. The constraint threshold is $b$ , with $C _ { s } \in \mathbb { R } ^ { d _ { s } \times d _ { s } }$ , $C _ { a } \in \mathbb { R } ^ { d _ { a } \times d _ { a } }$ , $b _ { s } \in \mathbb { R } ^ { d _ { s } }$ , and $b _ { a } \in \mathbb { R } ^ { d _ { a } }$ specifying state-action constraints, e.g., if $C _ { s }$ , $C _ { a }$ are identity matrices, $b _ { s } , b _ { a }$ limit state and action ranges. Equations (3a), (3c), and (3d) describe the constrained regulation problem under Gaussian disturbances (Bemporad et al. 2002; Stathopoulos, Korda, and Jones 2016), where the optimal policy is deterministic (Scokaert and Rawlings 1998). We add a general constraint (3b). The Markovian transition dynamics (3d) are linear, and the Gaussian noise $\omega _ { t }$ is nonatomic, rendering the transition probabilities non-atomic. If $\rho$ is non-atomic, the underlying MDP of (3) is also non-atomic. The reward function $\begin{array} { r } { r ( \bar { s } , a ) : = s ^ { \top } G _ { 1 } s + a ^ { \top } R _ { 1 } a } \end{array}$ induces a value function $V _ { r } ( \pi )$ , bounded within $[ r _ { \operatorname* { m i n } } / ( 1 - \gamma ) , 0 ]$ , with $r _ { \mathrm { m i n } } : = b _ { s } ^ { \top } G _ { 1 } \dot { b _ { s } } + b _ { a } ^ { \top } R _ { 1 } b _ { a }$ . Similarly, for $u ( s , a ) : =$ $s ^ { \top } G _ { 2 } s + a ^ { \top } R _ { 2 } a$ , the utility value $V _ { u }$ is also bounded. Therefore, this problem is an instance of Problem (1), assuming the state space is bounded with $\| s \| \leq S _ { \operatorname* { m a x } }$ .

# 3 Method and Theory

While our problem has zero duality gap, finding an optimal dual $\lambda ^ { \star }$ poses a significant challenge, due to the presence of multiple saddle points in the Lagrangian. To address it, we resort to the regularization method. More specifically, we introduce two regularizers. First, the term $h \overset { \cdot } { ( \lambda ) } : = \overset { \cdot } { \lambda ^ { 2 } }$ promotes convexity in the Lagrange multiplier $\lambda$ . Second, the term $h _ { a } ( a ) : = \dot { - } \| a \| ^ { 2 }$ promotes concavity in the reward function $r$ by penalizing large actions selected by the policy $\pi$ . The associated value function is defined as $H ^ { \pi } ( s ) : =$ $\mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } h _ { a } ( a _ { t } ) \vert s \right]$ , and leads to the regularizer $H ( \pi ) : =$ $\mathbb { E } _ { \rho } [ H ^ { \pi } ( s ) ]$ . Now, we consider the problem

$$
\operatorname* { m i n } _ { \lambda \in \Lambda } \operatorname* { m a x } _ { \pi \in \Pi } L _ { \tau } ( \pi , \lambda ) : = V _ { \lambda } ( \pi ) + \frac { \tau } { 2 } H ( \pi ) + \frac { \tau } { 2 } h ( \lambda ) ,
$$

where $\tau \geq 0$ is the regularization parameter and $L _ { \tau } ( \pi , \lambda )$ is the regularized Lagrangian. For a fixed $\lambda$ , the objective of Problem (4) is equivalent to an unconstrained regularized MDP plus a regularization of the dual variable. Consider the composite regularized reward function $r _ { \lambda , \tau } ( s , a ) \ : =$ $r ( s , a ) \bar { + } \lambda g ( s , a ) - { \textstyle \frac { \tau } { 2 } } h _ { a } ( a )$ . The value function associated with the reward function ${ \boldsymbol { r } } _ { \lambda , \tau }$ can be expressed as $\begin{array} { r } { V _ { \lambda , \tau } ( \pi ) = V _ { \lambda } ( \pi ) + \frac { \tau } { 2 } H ( \pi ) } \end{array}$ . Then, we can reformulate the regularized Lagrangian as $\begin{array} { r } { L _ { \tau } ( \pi , \lambda ) : = V _ { \lambda , \tau } ( \pi ) + \frac { \tau } { 2 } \lambda ^ { 2 } } \end{array}$ . The global saddle points of the regularized Lagrangian $\smash { \tilde { \Pi } _ { \tau } ^ { \star } \times \Lambda _ { \tau } ^ { \star } }$ are guaranteed to exist; see Lemma 13 in Appendix C. Moreover, a global saddle point $( \pi _ { \tau } ^ { \star } , \lambda _ { \tau } ^ { \star } )$ satisfies

$$
V _ { \lambda _ { \tau } ^ { \star } } ( \pi ) + \frac { \tau } { 2 } H ( \pi ) \leq V _ { \lambda _ { \tau } ^ { \star } } ( \pi _ { \tau } ^ { \star } ) \leq V _ { \lambda } ( \pi _ { \tau } ^ { \star } ) + \frac { \tau } { 2 } \lambda ^ { 2 }
$$

for all $( \pi , \lambda ) \in \Pi \times \Lambda$ . Hence, $( \pi _ { \tau } ^ { \star } , \lambda _ { \tau } ^ { \star } )$ is also a global saddle point of the original Lagrangian $L ( \pi , \lambda )$ up to two $\tau$ -terms.

# 3.1 Deterministic Policy Search Method

We propose a deterministic policy gradient primal-dual (DPGPD) method for finding a global saddle point $( \pi _ { \tau } ^ { \star } , \lambda _ { \tau } ^ { \star } )$ of $L _ { \tau } ( \pi , \lambda )$ . In the primal update, as is customary in RL, we maximize the advantage function rather than the value function directly. Specifically, we use the regularized advantage $\begin{array} { r } { A _ { \lambda , \tau } ^ { \pi } ( s , a ) : = Q _ { \lambda , \tau } ^ { \pi } ( s , a ) - V _ { \lambda , \tau } ^ { \pi } ( s ) - \bar { \frac { \tau } { 2 } } ( \| a \| ^ { 2 } - \| \pi ( s ) \| ^ { 2 } ) } \end{array}$ associated with the regularized reward ${ \boldsymbol { r } } _ { \lambda , \tau }$ . The primal update (6a) performs a proximal-point-type ascent step that solves a quadratic-regularized maximization sub-problem, while the dual update (6b) performs a gradient descent step that solves a quadratic-regularized minimization sub-problem

$$
\begin{array} { r l } & { \pi _ { t + 1 } ( s ) = \underset { a \in A } { \mathrm { a r g m a x ~ } } A _ { \lambda _ { t } , \tau } ^ { \pi _ { t } } ( s , a ) - \displaystyle \frac { 1 } { 2 \eta } \| a - \pi _ { t } ( s ) \| ^ { 2 } } \\ & { \quad \lambda _ { t + 1 } = \underset { \lambda \in \Lambda } { \mathrm { a r g m i n ~ } } \lambda \left( V _ { g } ( \pi _ { t } ) + \tau \lambda _ { t } \right) + \displaystyle \frac { 1 } { 2 \eta } \| \lambda - \lambda _ { t } \| ^ { 2 } , } \end{array}
$$

where $\eta$ is the step-size. D-PGPD is a single-time-scale algorithm, in the sense that the primal and the dual updates are computed concurrently in the same time-step. We remark that implementing D-PGPD is difficult in practice, and to make it tractable, we will leverage function approximation in Section 4. Before proceeding, we show that the primal-dual iterates (6) converge in the last iterate to the set of global saddle points of the regularized Lagrangian $\Pi _ { \tau } ^ { \star } \times \Lambda _ { \tau } ^ { \star }$ .

# 3.2 Non-Asymptotic Convergence

Finding deterministic optimal policies is a computationally challenging problem (Feinberg 2000; Dolgov 2005). To render the problem tractable, we assume concavity and Lipschitz continuity of the regularized action value functions.

Assumption 3 (Concavity). The regularized state-action value function $Q _ { \lambda , \tau } ^ { \pi } ( s , a ) \dot { - } \tau _ { 0 } \| \pi _ { 0 } ( s ) - a \| ^ { 2 }$ is concave in action a for any policy $\pi _ { 0 }$ and some $\tau _ { 0 } \in [ 0 , \tau )$ .

Assumption 4 (Lipschitz continuity). The actionvalue functions $Q _ { r } ^ { \pi } ( s , a ) , Q _ { g } ^ { \pi } ( s , a )$ , and $\begin{array} { r l } { H ^ { \pi } ( s , a ) } & { { } : = } \end{array}$ $\begin{array} { r } { \mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } h _ { a } ( a _ { t } ) \vert s _ { 0 } = s , a _ { 0 } ^ { \mathrm { ~ } } = a \right] } \end{array}$ are Lipschitz in action $a$ with Lipschitz constants $L _ { r } , L _ { g }$ , and $L _ { h }$ , i.e.,

$$
\begin{array} { r l } & { \| Q _ { r } ^ { \pi } ( s , a ) - Q _ { r } ^ { \pi } ( s , a ^ { \prime } ) \| \leq L _ { r } \| a - a ^ { \prime } \| } \\ & { \| Q _ { g } ^ { \pi } ( s , a ) - Q _ { g } ^ { \pi } ( s , a ^ { \prime } ) \| \leq L _ { g } \| a - a ^ { \prime } \| } \\ & { \| H ^ { \pi } ( s , a ) - H ^ { \pi } ( s , a ^ { \prime } ) \| \leq L _ { h } \| a - a ^ { \prime } \| , \forall a , a ^ { \prime } \in A . } \end{array}
$$

Assumption 3 states that there exists a $\tau _ { 0 }$ -strongly concave regularizer that renders $Q _ { \lambda , \tau } ^ { \pi }$ concave in the action $a$ . When $\tau _ { 0 } = 0 , Q _ { \lambda , \tau } ^ { \pi }$ is concave in the action $a$ . An example of this is Problem (3), where the original reward and utility functions are concave and the transition dynamics are linear, leading to concavity of the associated regularized value function. Assumption 4 implies Lipschitz continuity of the reward function and the probability transition kernel, which holds for several dynamics that can be expressed as a deterministic function of the actual state-action pair and some stochastic perturbation; see Appendix D.1 for a detailed explanation over the example introduced in Section 2.2.

To show convergence of D-PGPD, we introduce first two projection operators. The operator $\mathcal { P } _ { \Pi _ { \tau } ^ { \star } }$ projects a policy into the non-empty set of optimal policies with state visitation distribution $d _ { \rho } ^ { \star }$ , and the operator $\mathcal { P } _ { \Lambda _ { \tau } ^ { \star } }$ projects a Lagrangian multiplier onto the non-empty set of optimal Lagrangian multipliers $\Lambda _ { \tau } ^ { \star }$ . Then, we characterize the convergence of the primal-dual iterates of D-PGPD using a potential function

$$
\Phi _ { t } : = \frac { 1 } { 2 } \mathbb { E } _ { d _ { \rho } ^ { \star } } \left[ \| \mathcal { P } _ { \Pi _ { \tau } ^ { \star } } ( \pi _ { t } ( s ) ) - \pi _ { t } ( s ) \| ^ { 2 } \right] + \frac { \| \mathcal { P } _ { \Lambda _ { \tau } ^ { \star } } ( \lambda _ { t } ) - \lambda _ { t } \| ^ { 2 } } { 2 ( 1 + \eta ( \tau - \tau _ { 0 } ) ) } ,
$$

which measures the distance between a iteration pair $( \pi _ { t } , \lambda _ { t } )$ of D-PGPD and the set of global saddle points of the regularized Lagrangian $\Pi _ { \tau } ^ { \star } \times \Lambda _ { \tau } ^ { \star }$ . Theorem 2 shows that as $t$ increases, the potential function $\Phi _ { t }$ decreases linearly, up to an error; see Appendix C.4 for the proof.

Theorem 2 (Linear convergence). Let Assumptions 2–4 hold. For $\eta > 0$ and $\tau > \tau _ { 0 }$ , the primal-dual iterates (6) satisfy

$$
\Phi _ { t + 1 } \ : \le \ : \mathrm { e } ^ { - \beta _ { 0 } t } \Phi _ { 1 } + \beta _ { 1 } C _ { 0 } ^ { 2 } , \ : \ : \ : w h e r e
$$

$$
\beta _ { 0 } : = \frac { \eta ( \tau - \tau _ { 0 } ) } { 1 + \eta ( \tau - \tau _ { 0 } ) } a n d \beta _ { 1 } : = \frac { \eta ( 1 + \eta ( \tau - \tau _ { 0 } ) ) } { \tau - \tau _ { 0 } }
$$

$$
C _ { 0 } : = \mathrm { \Delta } L _ { r } + \lambda _ { \operatorname* { m a x } } L _ { g } + \tau L _ { h } + \tau \sqrt { d _ { a } } A _ { m a x } + \frac { 1 + \frac { \tau } { \xi } } { 1 - \gamma } .
$$

Theorem 2 states that the primal-dual updates of D-PGPD converge to a neighborhood of the set of global saddle points of the regularized Lagrangian $\Pi _ { \tau } ^ { \star } \times \Lambda _ { \tau } ^ { \star }$ in a linear rate. The size of the neighborhood depends polynomially on the parameters $( L _ { r } , L _ { g } , L _ { h } , A _ { \mathrm { m a x } } , \tau )$ . When $\tau _ { 0 } = 0$ , the regularization parameter $\tau$ can be arbitrarily small. Reducing the size of the convergence neighborhood can be achieved by selecting a sufficiently small $\eta$ . However, a smaller the value of $\eta$ leads to slower convergence. To be more specific, for $\eta \stackrel { . } { = } \epsilon ( \tau - \tau _ { 0 } ) C _ { 0 } ^ { - 2 }$ , the size of the convergence neighborhood is $O ( \epsilon )$ , and when $t \geq \Omega ( \epsilon ^ { - 1 } \log ( \epsilon ^ { - 1 } \bar { ) } )$ , the potential function $\Phi _ { t }$ is $O ( \epsilon )$ too, where $\Omega$ encapsulates some problemdependent constants. After $O ( \epsilon ^ { - 1 } )$ iterations, the primal-dual iterates $( \pi _ { t } , \lambda _ { t } )$ of D-PGPD are $\epsilon$ -close to the set $\Pi _ { \tau } ^ { \star } \times \Lambda _ { \tau } ^ { \star }$ .

The relationship between the solution to Problem (1) and the solution to the regularized Problem (4) is given by Corollary 1; see its proof in Appendix C.5.

Corollary 1 (Near-optimality). Let Assumptions 2–4 hold. I $f \eta = O \dot { ( } \epsilon ^ { 4 } )$ and $\tau \stackrel { \cdot } { = } O ( \epsilon ^ { 2 } ) \stackrel { \cdot } { + } \tau _ { 0 }$ , and $t = \mathsf { \bar { \Omega } } \mathsf { ( \epsilon ^ { - 6 } l o g ^ { 2 } } \epsilon ^ { - 1 } )$ , then the primal-dual iterates (6) satisfy

$$
\begin{array} { r l } { V _ { r } ( \pi ^ { \star } ) - V _ { r } ( \pi _ { t } ) \ : \le \ : \epsilon - \tau _ { 0 } H ( \pi ^ { \star } ) } & { } \\ { V _ { g } ( \pi _ { t } ) \ : \ge \ : - \epsilon + \tau _ { 0 } H ( \pi ^ { \star } ) ( \lambda _ { \operatorname* { m a x } } - \lambda ^ { \star } ) ^ { - 1 } . } \end{array}
$$

Corollary 1 highlights that the value functions corresponding to the policy iterates of D-PGPD can closely approximate the optimal solution to Problem (1). Specifically, in problems where $\tau _ { 0 } = 0$ , the final policy iterate of D-PGPD achieves $\epsilon$ -optimality for Problem (1) after $\Omega ( \epsilon ^ { - 6 } )$ iterations. When $\tau _ { 0 } > 0$ , D-PGDP converges to a saddle point of the original problem. However, the proximity of the final policy iterate to the optimal solution to Problem (1) is proportional to $H ( \pi ^ { \star } )$ .

This work presents the first primal-dual convergence result for general constrained RL problems that directly work with deterministic policies and continuous state-action spaces. In the context of control, the convergence of different algorithms for solving constrained problems has been analyzed (Stathopoulos, Korda, and Jones 2016; Zhang et al. 2020; Garg, Arabi, and Panagou 2020). However, these analyses are limited to linear utility functions and box constraints. DPGPD is a general algorithm that can be used for a broad range of transition dynamics and cost functions.

# 4 Function Approximation

To instantiate D-PGPD (6) with function approximation we begin by expanding the objective in (6a) and dropping the terms that do not depend on the action $a$ ,

$$
Q _ { \lambda , \tau } ^ { \pi } ( s , a ) + \frac { 1 } { \eta } \pi ( s ) ^ { \top } a - \left( \frac { \tau } { 2 } + \frac { 1 } { 2 \eta } \right) \| a \| ^ { 2 } .
$$

The usual function approximation approach (Agarwal et al. 2021; Ding et al. 2022) is to introduce a parametric estimator of the policy $\pi$ , and a compatible parametric estimator of the action value function $Q _ { \lambda , \tau } ^ { \pi }$ . Instead, we approximate the augmented action-value function $J ^ { \pi } ( s , a ) : = Q _ { \lambda , \tau } ^ { \pi } ( s , a ) +$ $\begin{array} { r } { \frac { 1 } { \eta } \pi ( s ) ^ { \top } a } \end{array}$ using a linear estimator $\tilde { J } _ { \theta } ( s , a ) = \phi ( s , a ) ^ { \top } \theta$ over the basis $\phi$ . At time $t$ , we estimate $J ^ { \pi _ { t } } ( s , a )$ by computing the parameters $\theta _ { t }$ via a mean-squared-error minimization

$$
\theta _ { t } : = \operatorname * { a r g m i n } _ { \theta } \mathbb { E } _ { ( s , a ) \sim \nu } \left[ \| \phi ( s , a ) ^ { \top } \theta - J ^ { \pi _ { t } } ( s , a ) \| ^ { 2 } \right] ,
$$

where $\nu$ is a pre-selected state-action distribution. Problem (8) can be easily addressed using, e.g., stochastic approximation. A subsequent policy $\pi _ { t + 1 }$ results from a primal update based on $\tilde { J } _ { \theta _ { t } }$ . This leads to an approximated D-PGPD algorithm (AD-PGPD) that updates $\pi _ { t }$ and $\lambda _ { t }$ via

$$
\begin{array} { r l } & { \pi _ { t + 1 } ( s ) = \underset { a \in A } { \mathrm { a r g m a x ~ } } \tilde { J } _ { \theta _ { t } } ( s , a ) - \left( \frac { \tau } { 2 } + \frac { 1 } { 2 \eta } \right) \| a \| ^ { 2 } } \\ & { \quad \lambda _ { t + 1 } = \underset { \lambda \in \Lambda } { \mathrm { a r g m i n ~ } } \lambda ( V _ { g } ( \pi _ { t } ) + \tau \lambda _ { t } ) + \displaystyle \frac { 1 } { 2 \eta } \| \lambda - \lambda _ { t } \| ^ { 2 } . } \end{array}
$$

Solving the sub-problem (9a) requires inverting the gradient of (9a) with respect to $a$ , which is a challenge when the MDP model is unknown or the value functions cannot be computed in closed form. This is the focus of Section 5.

# 4.1 Non-Asymptotic Convergence

To ease the computational tractability of AD-PGPD, we assume concavity of the approximated augmented action-value function and bounded approximation error.

Assumption 5 (Concavity of approximation). The function $\tilde { J } _ { \theta _ { t } } ( s , a ) - \tau _ { 0 } \| \pi _ { 0 } ( s ) - a \| ^ { 2 }$ is concave with respect to the action a for some arbitrary policy $\pi _ { 0 }$ and some $\tau _ { 0 } \in [ 0 , \tau )$ . Assumption 6 (Approximation error). The approximation error $\delta _ { \theta _ { t } } ( s , a )$ is bounded, $\mathbb { E } _ { s \sim d _ { \rho } ^ { \star } , a \sim \mathbf { u } } [ \| \delta _ { \theta _ { t } } ( s , a ) \| ] \le$ $\frac { \epsilon _ { \mathrm { a p p r o x } } } { 2 ( 2 A _ { m a x } ) ^ { d _ { a } } }$ , where u is the uniform distribution and $\epsilon _ { \mathrm { a p p r o x } } \geq 0$ is a positive error constant.

The concavity of $\tilde { J } _ { \theta _ { t } } ( s , a )$ with respect to $a$ depends on the selection of the basis function $\phi$ . When the augmented actionvalue function $J ^ { \pi _ { t } }$ is a concave quadratic function, it can be represented as a weighted linear combination of concave and quadratic basis functions. If these basis functions are known, $J ^ { \pi _ { t } }$ can be perfectly approximated, i.e., $\epsilon _ { \mathrm { a p p r o x } } = 0$ Furthermore, when $J ^ { \pi _ { t } }$ is concave with respect to the action $a$ , the regularization parameter $\tau$ can be arbitrarily small.

Theorem 3 (Linear convergence). Let Assumptions 2, 4– $\boldsymbol { \mathscr { \sigma } }$ hold. If $\mathrm { \Delta } \eta > 0$ and $\tau > \tau _ { 0 }$ , the primal-dual iterates (9) satisfy

$$
\Phi _ { t + 1 } \leq \mathrm { e } ^ { - \beta _ { 0 } t } \Phi _ { 1 } + \beta _ { 1 } C _ { 0 } ^ { 2 } + \beta _ { 2 } \epsilon _ { \mathrm { a p p r o x } } ,
$$

where $\beta _ { 0 } , \beta _ { 1 }$ , and $C _ { 0 }$ are defined in Theorem 2, and

$$
\beta _ { 2 } : = \frac { 1 + \eta ( \tau - \tau _ { 0 } ) } { \tau - \tau _ { 0 } } .
$$

Theorem 3 shows that the primal-dual iterates of ADPGPD converge to a neighborhood of $\Pi _ { \tau } ^ { \star } \times \Lambda _ { \tau } ^ { \star }$ at a linear rate. The result is similar to Theorem 2, up to an approximation error $\epsilon _ { \mathrm { a p p r o x } }$ . In fact, when $\epsilon _ { \mathrm { a p p r o x } } = 0$ , Theorem 3 is equivalent to Theorem 2. Linear models can achieve $\epsilon _ { \mathrm { a p p r o x } } = 0$ when the augmented action-value function $J ^ { \pi _ { t } }$ can be expressed as a linear combination of the selected basis function $\phi$ , e.g. when $J ^ { \pi _ { t } }$ is convex. When the error is small, the following result relates Problem (1) to the regularized Problem (4).

Corollary 2 (Near-optimality of approximation). Let Assumptions 2 and $_ { 4 - 6 }$ hold. If $\eta = \bar { O ( \epsilon ^ { 4 } ) }$ , $\tau = O ( \epsilon ^ { 2 } ) + \tau _ { 0 }$ , $\epsilon _ { \mathrm { a p p r o x } } = O ( \epsilon ^ { 4 } )$ , and $t = \Omega ( \epsilon ^ { - 6 } \log ^ { 2 } \epsilon ^ { - 1 } )$ , then the primaldual iterates (9) satisfy

$$
\begin{array} { r l } { V _ { r } ( \pi ^ { \star } ) - V _ { r } ( \pi _ { t } ) \ : \le \ : \epsilon - \tau _ { 0 } H ( \pi ^ { \star } ) } & { } \\ { V _ { g } ( \pi _ { t } ) \ : \ge \ : - \epsilon + \tau _ { 0 } H ( \pi ^ { \star } ) ( \lambda _ { \operatorname* { m a x } } - \lambda ^ { \star } ) ^ { - 1 } . } \end{array}
$$

Corollary 2 states that Corollary 1 extends to function approximation. When the approximation error is sufficiently small, i.e., $\epsilon _ { \mathrm { a p p r o x } } = O ( \epsilon ^ { 4 } )$ , the proof of Corollary 1 holds (see Appendix C.5), and the value functions corresponding to the policy iterates of AD-PGPD closely approximate an optimal solution to Problem (1). In fact, when $\tau _ { 0 } = 0$ and $\epsilon _ { \mathrm { a p p r o x } }$ are small, then the last policy iterate of AD-PGPD is an $\epsilon$ -optimal solution to Problem (1) after $\Omega ( \epsilon ^ { - 6 } )$ iterations.

# 5 Model-Free Algorithm

When the model of the MDP is unknown or when valuefunctions cannot be computed in closed form, we can leverage sample-based approaches to compute the primal and dual iterates of AD-PGPD. To that end, we assume access to a simulator of the MDP from where we can sample trajectories given a policy $\pi$ . The sample-based algorithm requires modifying the policy evaluation step in (8), and the dual update in (9b). For the former, in time-step $t$ for a given policy $\pi _ { t }$ , we have the following linear function approximation problem

$$
\operatorname* { m i n } _ { \theta , \| \theta \| \leq \theta _ { \operatorname* { m a x } } } \mathbb { E } _ { s , a \sim \nu } \left[ \| \phi ( s _ { n } , a _ { n } ) ^ { \top } \theta - \hat { J } ^ { \pi _ { t } } ( s _ { n } , a _ { n } ) \| ^ { 2 } \right] ,
$$

where the parameters $\theta$ are bounded, i.e., $\| \theta \| ~ \le ~ \theta _ { \operatorname* { m a x } }$ , and $\phi$ is the basis function. The approximated augmented value-function $\begin{array} { r } { \hat { J } ^ { \pi _ { t } } : = \hat { Q } _ { \lambda , \tau } ^ { \pi _ { t } } ( s _ { n } , \overset {  } { a _ { n } } ) + \frac { 1 } { \eta } \pi ( s _ { n } ) ^ { \top } \overset {  } { a _ { n } } } \end{array}$ is estimated from samples, which comes down to approximating $\hat { Q } _ { \lambda , \tau } ^ { \pi _ { t } } ( s _ { n } , a _ { n } )$ . The dual update $( 9 \mathfrak { b } )$ also requires the approximated value-function $\hat { V } _ { g } ( \pi _ { t } )$ to be estimated. We detail how to estimate $\hat { V } _ { g } ( \pi _ { t } )$ and $\hat { Q } _ { \lambda , \tau } ^ { \pi _ { t } } ( s _ { n } , a _ { n } )$ via rollouts in Algorithms 1 and 2, which can be found in Appendix E. We use random horizon rollouts (Paternain et al. 2020; Zhang et al. 2020) to guarantee that the stochastic estimates of $\hat { Q } _ { \lambda , \tau } ^ { \pi _ { t } }$ and $\hat { V } _ { g } ( \pi _ { t } )$ are unbiased. From (Paternain et al. 2020, Proposition 2), we have $Q _ { \lambda , \tau } ^ { \pi _ { t } } ( s , a ) = \mathbb { E } [ \hat { Q } _ { \lambda , \tau } ^ { \pi _ { t } } ( s , a ) | s , a ]$ and $V _ { g } ( \pi _ { t } ) = \mathbb { E } [ \hat { V } _ { g } ^ { \pi _ { t } } ( s ) ]$ , where the expectations $\mathbb { E }$ are taken over the randomness of drawing trajectories following $\pi _ { t }$ . We solve Problem (11) at time $t$ using projected stochastic gradient descent (SGD),

$$
\begin{array} { r c l } { { g _ { t } ^ { ( n ) } } } & { { = } } & { { 2 \left( \phi ( s _ { n } , a _ { n } ) ^ { \top } \theta _ { t } ^ { ( n ) } - \hat { J } ^ { \pi _ { t } } ( s _ { n } , a _ { n } ) \right) \phi \left( s _ { n } , a _ { n } \right) } } \\ { { } } & { { } } & { { } } \\ { { \theta _ { t } ^ { ( n + 1 ) } } } & { { = } } & { { \mathcal { P } _ { \| \theta \| \leq \theta _ { \operatorname* { m a x } } } \left( \theta _ { t } ^ { ( n ) } - \alpha _ { n } g _ { t } ^ { ( n ) } \right) , \ ~ ( 1 2 ) } } \end{array}
$$

where $n \geq 0$ is the iteration index, $\alpha _ { n }$ is the step-size, $g _ { t } ^ { ( n ) }$ (n) is the stochastic gradient of (11), and $\mathcal { P } _ { | | \theta | | \leq \theta _ { \operatorname* { m a x } } }$ is an operator that projects onto the domain $\lVert \boldsymbol { \theta } \rVert \leq \theta _ { \mathrm { m a x } }$ , which is convex and bounded. Each projected SGD update (12) forms the estimate $\widehat { \theta } _ { t }$ . We run $N$ projected SGD iterations and form the weighted average $\begin{array} { r } { \hat { \theta } _ { t } : = \frac { 2 } { N ( N + 1 ) } \sum _ { n = 0 } ^ { N - 1 } ( n + 1 ) \hat { \theta } _ { t } } \end{array}$ , which is the estimation of the parameters $\theta _ { t }$ . Combining (9), the SGD rule in (12), and averaging techniques lead to a sample-based algorithm presented in Algorithm 3, in Appendix E.

The convergence analysis of Algorithm 3 has to account for the estimation error induced by the sampling process. The error $\delta _ { \hat { \theta } _ { t } } ( s , a ) = \tilde { J } _ { \hat { \theta } _ { t } } ( s , a ) - \bar { J } ^ { \pi _ { t } } ( s , a )$ can be decomposed as $\delta _ { \hat { \theta } _ { t } } ( s , a ) = \delta _ { \hat { \theta } _ { t } } ( s , a ) - \delta _ { \theta _ { t } } ( s , a ) + \delta _ { \theta _ { t } } ( s , a )$ . The bias error term $\delta _ { \theta _ { t } } ( s , a )$ is similar to the approximation error of AD-PGPD and captures how good the model approximates the true augmented value function. The term $\delta _ { \hat { \theta } _ { t } } ( s , a ) \ - \$ $\delta _ { \theta _ { t } } ( s , a )$ is a statistical error that reflects the error introduced by the sampling mechanism for a given state-action pair. To deal with the randomness of the projected SGD updates, we assume that the bias error and the feature basis are bounded. We also assume that the feature covariance matrix is positive definite, and that the sampling distribution $\nu$ and the optimal state visitation frequency $d _ { \rho } ^ { \star }$ are uniformly equivalent.

Assumption 7 (Bounded feature basis). The feature function is bounded, i.e., $\| \phi ( s , a ) \| \leq 1$ for all $s \in S$ and $a \in A$ .

Assumption 8 (Positive covariance). The feature covariance matrix $\Sigma _ { \nu } = \mathbb { E } _ { s , a \sim \nu } [ \phi ( s , a ) \phi ( s , a ) ^ { \top } ]$ is positive definite $\Sigma _ { \nu } \geq \kappa _ { 0 } I$ for the state-action distribution $\nu$ .

Assumption 9 (Bias error). The bias error $\delta _ { \theta _ { t } } ( s , a )$ is bounded $\begin{array} { r } { \mathbb { E } _ { s \sim d _ { \rho } ^ { \star } , a \sim \mathbf { u } } [ \| \delta _ { \theta _ { t } } ( s , a ) \| ] \ \le \ \frac { \epsilon _ { \mathrm { b i a s } } } { 2 ( 2 A _ { m a x } ) ^ { d _ { a } } } } \end{array}$ , where u is the uniform distribution and $\epsilon _ { \mathrm { b i a s } }$ is a positive error constant.

Assumption 10 (Uniformly equivalence). The state-action distribution induced by the state-visitation frequency $d _ { \rho } ^ { \star }$ and the uniform distribution u is uniformly equivalent to the stateaction distribution $\nu$ , i.e.

$$
\frac { d _ { \rho } ^ { \star } ( s ) \mathsf { u } ( a ) } { \nu ( s , a ) } \ \leq \ L _ { \nu } \ f o r \ a l l \left( s , a \right) \in S \times A .
$$

Assumption 7 holds without loss of generality, as the basis functions are a design choice. Assumption 8 ensures that the minimizer of (11) is unique, since $\Sigma _ { \nu } \geq \kappa _ { 0 } I$ for some $\kappa _ { 0 } > 0$ . Assumption 9 states that the selected model achieves a bounded error, and Assumption 10 ensures that the sampling distribution $\nu$ is sufficiently representative of the optimal state visitation frequency $d _ { \rho } ^ { \star }$ . We characterize the convergence using the expected potential function $\mathbb { E } [ \Phi _ { t } ]$ , where the expectation is taken over the randomness of $\theta _ { t } ^ { ( n ) }$ . We have the following corollary; see the proof in Appendix C.7.

Corollary 3 (Linear convergence). Let Assumptions 2, 4, 5, and 7–10 hold. Then, the sample-based AD-PGPD in Algorithm 3 satisfies

$$
\mathbb { E } [ \Phi _ { t + 1 } ] \le e ^ { - \beta _ { 0 } t } \mathbb { E } [ \Phi _ { 1 } ] + \beta _ { 1 } C _ { 0 } ^ { 2 } + \beta _ { 2 } \left( \frac { C _ { 1 } ^ { 2 } } { \eta ^ { 2 } ( N + 1 ) } + \epsilon _ { \mathrm { b i a s } } \right) ,
$$

where $\beta _ { 0 } , \beta _ { 1 } , \beta _ { 2 }$ , and $C _ { 0 }$ are given in Theorems 2 and $3$ , and

$$
C _ { 1 } : = \sqrt { 2 ^ { d _ { a } + 5 } A _ { m a x } ^ { d _ { a } } L _ { \nu } } \left( \theta _ { m a x } + 2 ( 1 - \gamma ) ^ { - 2 } \xi ^ { - 1 } + d _ { a } A _ { m a x } ^ { 2 } \right) \kappa _ { 0 } ^ { - 1 } .
$$

Corollary 3 is analogous to Theorem 3, but accounting for the use of sample-based estimates. The sampling effect appears as the number $N$ of projected SGD steps performed at each time-step $t$ . Corollary 2 holds when the bias error $\epsilon _ { \mathrm { b i a s } } = { \cal O } ( \epsilon ^ { 4 } )$ and the estimation error $C _ { 1 } ^ { 2 } \eta ^ { - 2 } ( N + 1 ) ^ { - 1 } =$ $O ( \epsilon ^ { 4 } )$ . As $\eta ^ { ' } = { \cal O } ( \epsilon ^ { 4 } )$ , the latter holds when $N = \Omega ( \epsilon ^ { - 1 2 } )$ , where $\Omega$ encapsulates problem-dependent constants. Therefore, the number of rollouts required to output an $\epsilon$ -optimal policy is $t N = \Omega ( \epsilon ^ { - 1 8 } )$ . While this result suggests potential improvement, it stands as the first sample-complexity result in the context of constrained MDPs with continuous spaces.

![](images/d5938cfab6d1a979c6801d0c8a5dc72f963fcdab1f92cb40565a827ca1e6abb2.jpg)  
Figure 1: Navigation trajectories of an agent (Left) and velocity profile of the fluid over time (Right).

# 6 Computational Experiments

We test D-PGPD on constrained robot navigation and fluid control problems (Figure 1). See Appendix F for more details. Navigation Problem. An agent moves in a horizontal plane following some linearized dynamics with zero-mean Gaussian noise (Shimizu et al. 2020; Ma et al. 2022). We aim to drive the agent to the origin while constraining its velocity. When the dynamics are known and the reward function linearly weights quadratic penalties on position and action, this problem is an instance of the constrained linear regulation problem (Scokaert and Rawlings 1998), which has closedform solution. Hence, we can directly apply D-PGPD (6) and AD-PGPD (9) (See Appendix F.1). However, we consider the dynamics to be unknown, and we leverage our samplebased implementation of AD-PGPD. Furthermore, we use absolute value penalties instead of quadratic ones, as the latter can result in unstable behavior in sample-based scenarios (Engel and Babuška 2014). Conventional methods do not solve this problem straightforwardly. We compare our sample-based AD-PGPD with PGDual, a dual method with linear function approximation (Zhao and You 2021; Brunke et al. 2022). Figure 2 shows the value functions of the policy iterates generated by AD-PGPD and PGDual over 40, 000 iterations. The oscillations of AD-PGPD are damped over time, and it converges to a feasible solution with low variance in reward and utility, indicating a near-deterministic behavior without constraint violation. In contrast, PGDual exhibits large variance, indicating that the resultant policy violates the constraint. Nevertheless, the final primal return performance of PGDual is similar to that of AD-PGPD on average.

Fluid Velocity Control. We apply D-PGPD (6) to the control of the velocity of an incompressible Newtonian fluid described by the one-dimensional Burgers’ equation (Baker, Armaou, and Christofides 2000), a non-linear stochastic control problem. The velocity profile of the fluid $z$ varies in a one-dimensional space $x \in [ 0 , 1 ]$ and time $t \in [ 0 , 1 ]$ , and the goal is to drive the velocity of the fluid towards zero via the control action $a$ , e.g., injection of polymers. By discretizing Burgers’ equation, we have a non-linear system $s _ { t + 1 } = \stackrel { \cdot } { B } _ { 0 } s _ { t } + B _ { 1 } \stackrel { \cdot } { a _ { t } } + B _ { 2 } s _ { t } ^ { 2 } + \omega _ { t }$ , where $s _ { t } \in \mathbb { R } ^ { d }$ is the state, $s _ { t } ^ { 2 }$ is the element-wise squared state vector, $a _ { t } \in \mathbb { R } ^ { d }$ is the control input, and $B _ { 0 }$ , $B _ { 1 }$ , $B _ { 2 } \in \mathbb { R } ^ { d \times d }$ are matrices representing the discretized spatial operators and non-linear terms (Borggaard and Zietsman 2020). The details can be found in Appendix F. We consider a reward function that penalizes the state quadratically, and a budget constraint that limits the total control action. We compare our sample-based AD-PGPD with PGDual. Figure 3 shows the value functions of the policy iterates generated by AD-PGPD and PGDual over 10, 000 iterations. The results are consistent with those of the navigation problem. The AD-PGPD algorithm successfully mitigates oscillations and converges to a feasible solution with low return variance. In contrast, although PGDual achieves similar objective value, it does not dampens oscillations, as indicated by the variance of the solution. This implies that PGDual violates the constraint in the last iterate.

![](images/3cf093cbaf9687489a3685be4fb725a16b7a02a82b8e61615c2ad73b5442d8f5.jpg)  
Figure 2: Avg. reward/utility value functions of AD-PGPD $\mathrm { ~ ( ~ ) ~ }$ and PGDual $\mathrm { ~ ( ~ ) ~ }$ iterates in the navigation problem.

![](images/ce59bcc87b9bd67dbccabef6473301ffe98e1d89aaf34d1bd40556f8e4cc34ff.jpg)  
Figure 3: Avg. reward/utility value functions of AD-PGPD $( - )$ and PGDual $\mathrm { ~ ( ~ ) ~ }$ iterates in a fluid velocity control.

# 7 Concluding Remarks

We have presented a deterministic policy gradient primal-dual method for continuous state-action constrained MDPs with non-asymptotic convergence guarantees. We have leveraged function approximation to make the implementation practical and developed a sample-based algorithm. Furthermore, we have shown the effectiveness of the proposed method in navigation and non-linear fluid constrained control problems. Our work opens new avenues for constrained MDPs with continuous state-action spaces, such as (i) minimal assumption on value functions; (ii) online exploration; (iii) optimal sample complexity; and (iv) general function approximation.

# Appendix

All the theoretical proofs and additional materials referenced in this paper, the supplementary experiments and introductions to key concepts are included in the extended version of the paper, available at https://arxiv.org/abs/2408.10015.

# Acknowledgments

We thank the anonymous reviewers for their insightful comments. This work has been partially supported by the Spanish NSF (AEI/10.13039 /501100011033) grants TED2021- 130347B-I00 and PID2022-136887NB-I00, and the Community of Madrid via the Ellis Madrid Unit and grant TEC2024/COM-89.