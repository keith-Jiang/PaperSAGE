# Risk-averse Total-reward MDPs with ERM and EVaR

Xihong $\mathbf { S } \mathbf { u } ^ { 1 }$ , Marek Petrik1, Julien Grand-Cle´ment2

University of New Hampshire, 33 Academic Way, Durham, NH, 03824 USA 2 HEC Paris, 1 Rue de la Libe´ration, Jouy-en-Josas, 78350 France xihong.su@unh.edu, mpetrik@cs.unh.edu, grand-clement@hec.fr

# Abstract

Optimizing risk-averse objectives in discounted MDPs is challenging because most models do not admit direct dynamic programming equations and require complex history-dependent policies. In this paper, we show that the risk-averse total reward criterion, under the Entropic Risk Measure (ERM) and Entropic Value at Risk (EVaR) risk measures, can be optimized by a stationary policy, making it simple to analyze, interpret, and deploy. We propose exponential value iteration, policy iteration, and linear programming to compute optimal policies. Compared with prior work, our results only require the relatively mild condition of transient MDPs and allow for both positive and negative rewards. Our results indicate that the total reward criterion may be preferable to the discounted criterion in a broad range of risk-averse reinforcement learning domains.

# 1 Introduction

Risk-averse Markov decision processes (MDP) (Puterman 2005) that use monetary risk measures as their objective have been gaining in popularity in recent years (Kastner, Erdogdu, and Farahmand 2023; Marthe, Garivier, and Vernade 2023; Lam et al. 2022; Li, Zhong, and Brandeau 2022; Ba¨uerle and Glauner 2022; Hau, Petrik, and Ghavamzadeh 2023; Hau et al. 2023; Su, Petrik, and Grand-Cl´ement 2024a,b). Riskaverse objectives, such as Value at Risk (VaR), Conditional Value at Risk (CVaR), Entropic Risk Measure (ERM), or Entropic Value at Risk (EVaR), penalize the variability of returns (Follmer and Schied 2016). As a result, these risk measures yield policies with stronger guarantees on the probability of catastrophic losses, which is important in domains like healthcare or finance.

In this paper, we target the total reward criterion (TRC) (Kallenberg 2021; Puterman 2005) instead of the common discounted criterion. TRC also assumes an infinite horizon but does not discount future rewards. To control for infinite returns, we assume that the MDP is transient, i.e. that there is a positive probability that the process terminates after a finite number of steps, an assumption commonly used in the TRC literature (Filar and Vrieze 2012). We consider the TRC with both positive and negative rewards. When the rewards are non-positive, the TRC is equivalent to the stochastic shortest path problem, and when they are nonnegative, it is equivalent to the stochastic longest path (Dann, Wei, and Zimmert 2023).

Two reasons motivate our departure from discounted objectives in risk-averse MDPs. First, considering risk affects discounted objectives significantly. It is common to use discounted objectives because they admit optimal stationary policies and value functions that can be computed using dynamic programs. However, most risk-averse discount objectives, such as VaR, CVaR, or EVaR, require that optimal policies are history-dependent (B¨auerle and Ott 2011; Hau et al. 2023; Hau, Petrik, and Ghavamzadeh 2023) and do not admit standard dynamic programming optimality equations.

Second, TRC captures the concept of stochastic termination, which is common in reinforcement learning (Sutton and Barto 2018). In risk-neutral objectives, discounting can serve well to model the probability of termination because it guarantees the same optimal policies (Puterman 2005; Su and Petrik 2023). However, as we show in this work, no such correspondence exists with risk-averse objectives, and the difference between them may be arbitrarily significant. Modeling stochastic termination using a discount factor in risk-averse objectives is inappropriate and leads to dramatically different optimal policies.

As our main contribution, we show that the risk-averse TRC with ERM and EVaR risk measures admit optimal stationary policies and optimal value functions in transient MDPs. We also show that the optimal value function satisfies dynamic programming equations and can be computed with exponential value iteration, policy iteration, or linear programming algorithms. These algorithms are simple and closely resemble the algorithms for solving MDPs.

Our results indicate that EVaR is a particularly interesting risk measure in reinforcement learning. ERM and the closely related exponential utility functions have been popular in sequential decision-making problems because they admit dynamic programming decompositions (Patek and Bertsekas 1999; de Freitas, Freire, and Delgado 2020; Smith and Chapman 2023; Denardo and Rothblum 1979; Hau, Petrik, and Ghavamzadeh 2023; Hau et al. 2023). Unfortunately, ERM is difficult to interpret; it is scale-dependent; and it is incomparable with popular risk measures like VaR and CVaR. Because EVaR reduces to an optimization over ERM, it preserves most of the computational advantages of ERM, and since

Table 1: Structure of optimal policies in risk-averse $\mathbf { M D P s }$ : “S”, “M” and “H” refer to Stationary, Markov and Historydependent policies respectively.   

<html><body><table><tr><td rowspan="2"></td><td colspan="2">Risk properties</td><td colspan="2">Optimal policy</td></tr><tr><td>Risk measure Coherent</td><td>Law inv.</td><td>Disc.</td><td>TRC</td></tr><tr><td>E</td><td>yes</td><td>yes</td><td>S</td><td>S</td></tr><tr><td>EVaR</td><td>yes</td><td>yes</td><td>M</td><td>S</td></tr><tr><td>ERM</td><td>no</td><td>yes</td><td>M</td><td>S</td></tr><tr><td>NCVaR</td><td>yes</td><td>no</td><td>S</td><td>S</td></tr><tr><td>VaR</td><td>yes</td><td>yes</td><td>H</td><td>H</td></tr><tr><td>CVaR</td><td>yes</td><td>yes</td><td>H</td><td>H</td></tr></table></body></html>

EVaR closely approximates CVaR and VaR at the same risk level, its value is also much easier to interpret. Finally, EVaR is also a coherent risk measure, unlike ERM (Ahmadi-Javid 2012; Ahmadi-Javid and Pichler 2017).

Table 1 puts our contribution in the context of other work on risk-averse MDP objectives. Optimal policies for VaR and CVaR are known to be history-dependent in the discounted objective (Ba¨uerle and Ott 2011; Hau et al. 2023) and must be history-dependent in TRC because TRC generalizes the finitehorizon objective. The TRC with Nested risk measures, such as Nested CVaR (NCVaR), applies the risk measure in each level of the dynamic program independently and preserves most of the favorable computational properties of risk-neutral MDPs (Ahmadi et al. 2021a). Unfortunately, nested risk measures are difficult to interpret; their value depends on the sequence in which the rewards are obtained in a complex and unpredictable way (Kupper and Schachermayer 2006) and may be unbounded even if MDPs are transient.

While we are unaware of prior work on the TRC objective with ERM or EVaR risk-aversion allowing both positive and negative rewards, the ERM risk measure is closely related to exponential utility functions. Prior work on TRC with exponential utility functions also imposes constraints on the sign of the instantaneous rewards, such as all positive rewards (Blackwell 1967) or all negative rewards (Bertsekas and Tsitsiklis 1991; Freire and Delgado 2016; Carpin, Chow, and Pavone 2016; de Freitas, Freire, and Delgado 2020; Fei et al. 2021; Fei, Yang, and Wang 2021; Ahmadi et al. 2021a; Cohen et al. 2021; Meggendorfer 2022). Disallowing a mix of positive and negative rewards limits the modeling power of prior work because it requires that either all states are more desirable or all states are less desirable than the terminal state. Allowing rewards with mixed signs raises some technical challenges, which we address by employing a squeeze argument that takes advantage of MDP’s transience.

Notation. We use a tilde to mark random variables, e.g. $\tilde { x }$ . Bold lower-case letters represent vectors, and upper-case bold letters represent matrices. Sets are either calligraphic or upper-case Greek letters. The symbol $\mathbb { X }$ represents the space of real-valued random variables. When a function is defined over an index set, such as $z \colon \{ 1 , 2 , \ldots , N \} \to \mathbb { R }$ , we also treat it interchangeably as a vector $z \in \mathbb { R } ^ { n }$ such that $z _ { i } = z ( i ) , \forall i = 1 , . . . , n ,$ . Finally, $\mathbb { R } , \mathbb { R } _ { + } , \mathbb { R } _ { + + }$ denote real, non-negative real, and positive real numbers, respectively. $\bar { \mathbb { R } } = \bar { \mathbb { R } } \cup \{ - \infty , \infty \}$ . Given a finite set $\mathcal { Y }$ , the probability simplex is $\Delta _ { \mathcal { Y } } : = \{ \overset { \cdot } { x } \in \mathbb { R } _ { + } ^ { \mathcal { Y } } \mid \mathbf { 1 } ^ { \mathrm { T } } x = 1 \}$ .

# 2 Background on Risk-averse MDPs

Markov Decision Processes We focus on solving Markov decision processes (MDPs) (Puterman 2005), modeled by a tuple $( \bar { \mathcal { S } } , \bar { \mathcal { A } } , \bar { p } , \bar { r } , \bar { \mu } )$ , where $\bar { \mathcal { S } } = \{ 1 , 2 , . . . , S , S + 1 \}$ is the finite set of states and $\mathcal { A } = \{ 1 , 2 , \dotsc , A \}$ is the finite set of actions. The transition function $\bar { p } \colon \bar { \mathcal { S } } \times \mathcal { A }  \Delta _ { \bar { \mathcal { S } } }$ represents the probability $\bar { p } ( s , a , s ^ { \prime } )$ of transitioning to $s ^ { \prime } \in \bar { \mathcal { S } }$ after taking $a \in { \mathcal { A } }$ in $s \in \bar { \mathcal { S } }$ and $\bar { p } _ { s a } \in \Delta _ { \bar { \mathcal { S } } }$ is such that $( { \bar { p } } _ { s a } ) _ { s ^ { \prime } } =$ $\bar { p } ( s , a , s ^ { \prime } )$ . The function $\bar { r } \colon \bar { \mathcal { S } } \times \mathcal { A } \times \bar { \mathcal { S } }  \mathbb { R }$ represents the reward $\bar { r } ( s , a , s ^ { \prime } ) \in \mathbb { R }$ associated with transitioning from $s \in \bar { \mathcal { S } }$ and $a \in { \mathcal { A } }$ to $s ^ { \prime } \in \bar { \mathcal { S } }$ . The vector $\bar { \pmb { \mu } } \in \Delta _ { \bar { \mathcal { S } } }$ is the initial state distribution.

We designate the state $e : = S + 1$ as a sink state and use $\mathcal { S } = \{ 1 , . . . , S \}$ to denote the set of all non-sink states. The sink state $e$ must satisfy that $\bar { p } ( e , a , e ) = 1$ and $\bar { r } ( e , a , e ) = 0$ for each $a \in { \mathcal { A } }$ , and $\bar { \mu } _ { e } ~ = ~ 0$ . Throughout the paper, we use a bar to indicate whether the quantity involves the sink state $e$ . Note that the sink state can indicate a goal when all rewards are negative and an undesirable terminal state when all rewards are positive.

The following technical assumption is needed to simplify the derivation. To lift the assumption, one needs to carefully account for infinite values, which adds complexity to the results and distracts from the main ideas.

Assumption 2.1. The initial distribution $\pmb { \mu }$ satisfies that

$$
{ \boldsymbol { \mu } } > \mathbf { 0 } .
$$

The solution to an MDP is a policy. Given a horizon $t \in \mathbb { N }$ , a history-dependent policy in the set $\Pi _ { \mathrm { H R } } ^ { t }$ maps the history of states and actions to a distribution over actions. A Markov policy $\pi \in \Pi _ { \mathrm { M R } } ^ { t }$ is a sequence of decision rules $\pi = ( d _ { 0 } , d _ { 1 } , \ldots , d _ { t - 1 } )$ with $d _ { k } \colon \mathcal { S } \  \ \Delta _ { \mathcal { A } }$ the decision rule for taking actions at time $k$ . The set of all randomized decision rules is $\mathcal { D } = ( \Delta _ { \mathcal { A } } ) ^ { \mathcal { S } }$ . Stationary policies $\Pi _ { \mathrm { S R } }$ are Markov policies with $\pi : = ( d ) _ { \infty } : = ( d , d , . . . )$ with the identical decision rule in every timestep. We treat decision rules and stationary policies interchangeably. The sets of deterministic Markov and stationary policies are denoted by $\Pi _ { \mathrm { M D } } ^ { t }$ and $\Pi _ { \mathrm { S D } }$ . Finally, we omit the superscript $t$ to indicate infinite horizon definitions of policies.

The risk-neutral Total Reward Criterion (TRC) objective is:

$$
\operatorname* { s u p } _ { \pi \in \Pi _ { \mathrm { H R } } } \operatorname* { l i m } _ { t  \infty } \operatorname* { i n f } _ { } \mathbb { E } ^ { \pi , \mu } [ \sum _ { k = 0 } ^ { t - 1 } r ( \tilde { s } _ { k } , \tilde { a } _ { k } , \tilde { s } _ { k + 1 } ) ] ,
$$

where the random variables are denoted by a tilde and $\tilde { s } _ { k }$ and $\tilde { a } _ { k }$ represent the state from ¯S and action at time $k$ . The superscript $\pi$ denotes the policy that governs the actions $\tilde { a } _ { k }$ when visiting $\tilde { s } _ { k }$ and $\pmb { \mu }$ denotes the initial distribution. Finally, note that lim inf gives a conservative estimate of a policy’s return since the limit does not necessarily exist for non-stationary policies.

Unlike the discounted criterion, the risk-neutral TRC may be unbounded, optimal policies may not exist, or may be nonstationary (Bertsekas and $\mathrm { Y u } 2 0 1 3$ ; James and Collins 2006).

![](images/1f7de1e5eb3e53630142fc185322c01e96215a7e428351dcfa27c2fb28be0f6c.jpg)  
Figure 1: left: a discounted MDP, right: a transient MDP

To circumvent these issues, we assume that all policies have a positive probability of eventually transitioning to the sink state.

Assumption 2.2. The MDP is transient for any $\pi \in \Pi _ { \mathrm { S D } }$ :

$$
\sum _ { t = 0 } ^ { \infty } \mathbb { P } ^ { \pi , s } \left[ \tilde { s } _ { t } = s ^ { \prime } \right] < \infty , \qquad \forall s , s ^ { \prime } \in \mathcal { S } .
$$

Assumption 2.2 underlies most of our results. Transient MDPs are important because their optimal policies exist and can be chosen to be stationary deterministic (Kallenberg 2021, theorem 4.12). Transient MDPs are also common in stochastic games (Filar and Vrieze 2012) and generalize the stochastic shortest path problem (Bertsekas and $\mathrm { Y u } 2 0 1 3$ ).

An important tool in their analysis is the spectral radius $\rho \colon \mathbb { R } ^ { n \times n }  \mathbb { R }$ which is defined for each $\pmb { A } \in \mathbb { R } ^ { n \times n }$ as the maximum absolute eigenvalue: $\rho ( A ) : = \operatorname* { m a x } _ { i = 1 , \ldots , n } | \lambda _ { i } |$ where $\lambda _ { i }$ is the $i$ -th eigenvalue (Horn and Johnson 2013).

Lemma 2.3 (Theorem 4.8 in Kallenberg (2021)). An MDP is transient if and only if $\rho ( P ^ { \pi } ) < 1$ for all $\pi \in \Pi _ { \mathrm { S R } }$ .

Now, let us understand the differences between a discounted MDP and a transient MDP, which are useful in demonstrating the behavior of risk-averse objectives. Consider the MDPs in Figure 1. There is one non-sink state $s$ and one action $a$ . A triple tuple represents an action, transition probability, and a reward separately. Note that every discounted MDP can be converted to a transient MDP as described in Su, Grand-Cle´ment, and Petrik (2024, appendix B). For the discounted MDP, the discount factor is $\gamma$ . For the transient MDP, $e$ is the sink state, and there is a positive probability $1 - \epsilon$ of transiting from state $s$ to state $e$ . Once the agent reaches the state $e$ , it stays in $e$ . For the risk-neutral objective, if $\gamma$ equals $\epsilon$ , their value functions have identical values. However, for risk-aversion objectives, such as ERM, we show that the value functions in a discounted MDP can diverge from those in a transient MDP in Section 5.

Monetary risk measures Monetary risk measures aim to generalize the expectation operator to account for the spread of the random variable. Entropic risk measure (ERM) is a popular risk measure, defined for any risk level $\beta > 0$ and $\tilde { x } \in \mathbb { X }$ as (Follmer and Schied 2016)

$$
\operatorname { E R M } _ { \beta } \left[ \tilde { { \boldsymbol { x } } } \right] = - \beta ^ { - 1 } \cdot \log \operatorname { \mathbb { E } } \exp \left( - \beta \cdot \tilde { { \boldsymbol { x } } } \right) .
$$

and extended to $\beta \in [ 0 , \infty ]$ as $\mathrm { E R M _ { 0 } } [ \tilde { { \boldsymbol { x } } } ]$ $\ c =$ $\begin{array} { r l r } { \operatorname* { l i m } _ { \beta \to 0 ^ { + } } \mathrm { E R M } _ { \beta } \left[ \tilde { x } \right] } & { { } = } & { \mathbb { E } [ \tilde { x } ] } \end{array}$ and $\begin{array} { r l } { \mathrm { E R M } _ { \infty } [ \tilde { x } ] } & { { } = } \end{array}$ $\mathrm { l i m } _ { \beta \to \infty } \mathrm { E R M } _ { \beta } \left[ \tilde { x } \right] \ = \ \mathrm { e s s i n f } [ \tilde { x } ]$ . ERM plays a unique role in sequential decision-making because it is the only law-invariant risk measure that satisfies the tower property (e.g., Su, Grand-Cl´ement, and Petrik (2024, proposition A.1)), which is essential in constructing dynamic programs (Hau, Petrik, and Ghavamzadeh 2023). Unfortunately, two significant limitations of ERM hinder its practical applications. First, ERM is not positively homogenous and, therefore, the risk value depends on the scale of the rewards, and ERM is not coherent (Follmer and Schied 2016; Hau, Petrik, and Ghavamzadeh 2023; Ahmadi-Javid 2012). Second, the risk parameter $\beta$ is challenging to interpret and does not relate well to other standard risk measures, like VaR or CVaR.

For these reasons, we focus on the Entropic Value at Risk (EVaR), defined as, for a given $\alpha \in ( 0 , 1 )$ ,

$$
\begin{array} { r l } & { \mathrm { E V a R } _ { \alpha } \left[ \tilde { x } \right] = \underset { \beta > 0 } { \operatorname* { s u p } } - \beta ^ { - 1 } \log \left( \alpha ^ { - 1 } \mathbb { E } \exp \left( - \beta \tilde { x } \right) \right) } \\ & { = \underset { \beta > 0 } { \operatorname* { s u p } } \mathrm { E R M } _ { \beta } \left[ \tilde { x } \right] + \beta ^ { - 1 } \log \alpha , } \end{array}
$$

and extended to $\mathrm { E V a R _ { 0 } } \left[ \tilde { x } \right] = \mathrm { e s s i n f } [ \tilde { x } ]$ and $\mathrm { E V a R _ { 1 } } \left[ \tilde { x } \right] =$ $\mathbb { E } \left[ \tilde { { \boldsymbol { x } } } \right]$ (Ahmadi-Javid 2012). It is important to note that the supremum in (4) may not be attained even when $\tilde { x }$ is a finite discrete random variable (Ahmadi-Javid and Pichler 2017).

EVaR addresses the limitations of ERM while preserving its benefits. EVaR is coherent and positively homogenous. EVaR is also a good approximation to interpretable quantilebased risk measures, like VaR and CVaR (Ahmadi-Javid 2012; Hau, Petrik, and Ghavamzadeh 2023).

Risk-averse MDPs. Risk-averse MDPs, using static VaR and CVaR risk measures, under the discounted criterion received abundant attention (Hau et al. 2023; B¨auerle and Ott 2011; Ba¨uerle and Glauner 2022; Pflug and Pichler 2016; Li, Zhong, and Brandeau 2022), showing that these objectives require history-dependent optimal policies. In contrast, nested risk measures under the TRC may admit stationary policies that can be computed using dynamic programming (Ahmadi et al. 2021a; Meggendorfer 2022; de Freitas, Freire, and Delgado 2020; Gavriel, Hanasusanto, and Kuhn 2012). However, the TRC with nested CVaR can be unbounded (Su, GrandCl´ement, and Petrik 2024, proposition C.1). Recent work has shown that optimal Markov policies exist for EVaR discounted objectives, and they can be computed via dynamic programming (Hau, Petrik, and Ghavamzadeh 2023), building upon similar results established for ERM (Chung and Sobel 1987). However, in TRC with ERM, the value functions may also be unbounded (Su, Grand-Cle´ment, and Petrik 2024, proposition D.1).

# 3 Solving ERM Total Reward Criterion

This section shows that an optimal stationary policy exists for ERM-TRC and that the value function satisfies dynamic programming equations. We then outline algorithms for computing it.

Our objective in this section is to maximize the ERM-TRC objective for some given $\beta > 0$ defined as

$$
\operatorname* { s u p } _ { \pi \in \Pi _ { \mathrm { H R } } } \operatorname* { l i m } _ { t  \infty } \operatorname* { i n f } _ { 0 } \mathrm { E R M } _ { \beta } ^ { \pi , \mu } [ \sum _ { k = 0 } ^ { t - 1 } r ( \tilde { s } _ { k } , \tilde { a } _ { k } , \tilde { s } _ { k + 1 } ) ] .
$$

The definition employs limit inferior because the limit may not exist for non-stationary policies. Return functions $g _ { t } \colon \Pi _ { \mathrm { H R } } \times \mathbb { R } _ { + + } \to \mathbb { R }$ and $g _ { t } ^ { \star } \colon \mathbb { R } _ { + + } ~ \to ~ \mathbb { R }$ for a horizon $t \in \mathbb { N }$ and the infinite-horizon versions $g _ { t } \colon \Pi _ { \mathrm { H R } } \times \mathbb { R } _ { + + } \to \bar { \mathbb { R } }$ and $g _ { t } ^ { \star } : \mathbb { R } _ { + + } \to \bar { \mathbb { R } }$ are defined

$$
\begin{array} { r l } & { \displaystyle g _ { t } ( \pi , \beta ) : = \mathrm { E R M } _ { \beta } ^ { \pi , \mu } [ \sum _ { k = 0 } ^ { t - 1 } r \big ( \widetilde s _ { k } , \widetilde a _ { k } , \widetilde s _ { k + 1 } \big ) ] , } \\ & { \quad \quad \quad g _ { t } ^ { \star } ( \beta ) : = \displaystyle \operatorname* { s u p } _ { \pi \in \mathrm { I I R } } g _ { t } ( \pi , \beta ) , } \\ & { \displaystyle g _ { \infty } ( \pi , \beta ) : = \operatorname* { l i m i n f } _ { t  \infty } g _ { t } ( \pi , \beta ) , } \\ & { \quad \quad \quad g _ { \infty } ^ { \star } ( \beta ) : = \operatorname* { l i m i n f } _ { t  \infty } g _ { t } ^ { \star } ( \beta ) . } \end{array}
$$

Note that the functions $g _ { \infty }$ and $g _ { \infty } ^ { \star }$ can return infinite values and that (5) differs from $g _ { \infty } ^ { \star }$ in the order of the limit and the supremum. Finally, when $\beta = 0$ , we assume that all $g$ functions are defined as the expectation. In the remainder of the section, we assume that the risk level $\beta > 0$ is fixed and omit it in notations when its value is unambiguous from the context.

# 3.1 Finite Horizon

We commence the analysis with definitions and basic properties for the finite horizon criterion. To the best of our knowledge, this analysis is original in the context of the ERM but builds on similar approaches employed in the study of exponential utility functions.

Finite-horizon functions ${ \pmb v } ^ { t } ( \pi ) \in \mathbb { R } ^ { S }$ and $v ^ { t , \star } \in \mathbb { R } ^ { S }$ are defined for each horizon $t \in \mathbb { N }$ and policy $\pi \in \Pi _ { \mathrm { M D } }$ , $s \in \mathcal S$ as

$$
\begin{array} { r l } & { v _ { s } ^ { t } ( \pi ) : = \mathrm { E R M } _ { \beta } ^ { \pi , s } \left[ \displaystyle \sum _ { k = 0 } ^ { t - 1 } r ( \tilde { s } _ { k } , \tilde { a } _ { k } , \tilde { s } _ { k + 1 } ) \right] , } \\ & { \quad v _ { s } ^ { t , \star } : = \displaystyle \operatorname* { m a x } _ { \pi \in \Pi _ { \mathrm { M D } } } v _ { s } ^ { t } ( \pi ) , } \end{array}
$$

and $v _ { e } ^ { t } ( \pi ) : = 0$ .

Because the nonlinearity of ERM complicates the analysis, it will be convenient to instead rely on exponential value function $\pmb { w } ^ { t } ( \pi ) \in \mathbb { R } ^ { S }$ for $\pi \in \Pi _ { \mathrm { M D } }$ , $t \in \mathbb { N }$ , and $s \in \mathcal S$ that satisfy

$$
\begin{array} { c } { { w _ { s } ^ { t } ( \pi ) : = - \exp \left( - \beta \cdot v _ { s } ^ { t } ( \pi ) \right) , } } \\ { { v _ { s } ^ { t } ( \pi ) = - \beta ^ { - 1 } \log ( - w _ { s } ^ { t } ( \pi ) ) . } } \end{array}
$$

The optimal ${ \pmb w } ^ { t , \star } \in \mathbb { R } ^ { S }$ is defined analogously from $\scriptstyle v ^ { t , \star }$ . Note that $\mathbf { \Delta } w ^ { t } < \mathbf { 0 }$ (componentwise) and $\mathbf { \bar { w } } ^ { 0 } ( \pi ) = \pmb { w } ^ { 0 , \star } =$ $^ { - 1 }$ for any $\pi \in \Pi _ { \mathrm { M D } }$ . Similar exponential value functions have been used previously in exponential utility function objectives (Denardo and Rothblum 1979; Patek 2001), in the analysis of robust MDPs, and even in regularized MDPs (see Grand-Cl´ement and Petrik (2022) and references therein).

One can define a corresponding exponential Bellman operator for any ${ \boldsymbol w } \in \mathbb { R } ^ { S }$ as

$$
\begin{array} { r l } & { L ^ { d } \boldsymbol { w } : = B ^ { d } \boldsymbol { w } - b ^ { d } , } \\ & { L ^ { \star } \boldsymbol { w } : = \displaystyle \operatorname* { m a x } _ { d \in \mathcal { D } } L ^ { d } \boldsymbol { w } = \displaystyle \operatorname* { m a x } _ { d \in \mathrm { e x t \mathcal { D } } } L ^ { d } \boldsymbol { w } , } \end{array}
$$

where $\operatorname { e x t } \ \mathrm { \mathcal { D } }$ is the set of extreme points of $\mathrm { \textmathcal { D } }$ corresponding to deterministic decision rules and Bd ∈ RS+×S and $b ^ { d } \in$ $\mathbb { R } _ { + } ^ { S }$ are defined for $s , s ^ { \prime } \in \mathcal { S }$ and $\pmb { d } \in \mathcal { D }$ as

$$
\begin{array} { r } { B _ { s , s ^ { \prime } } ^ { d } : = \displaystyle \sum _ { a \in \mathcal { A } } p ( s , a , s ^ { \prime } ) \cdot d _ { a } ( s ) \cdot e ^ { - \beta \cdot r ( s , a , s ^ { \prime } ) } , } \\ { b _ { s } ^ { d } : = \displaystyle \sum _ { a \in \mathcal { A } } p ( s , a , e ) \cdot d _ { a } ( s ) \cdot e ^ { - \beta \cdot r ( s , a , e ) } . } \end{array}
$$

The following theorem shows that $L$ can be used to compute $\textbf { \em w }$ . We use the shorthand notation $\pi _ { 1 : t - 1 } =$ $( d _ { 1 } , \cdot \cdot \cdot , d _ { t - 1 } ) \in \Pi _ { \mathrm { M R } } ^ { t - 1 }$ to denote the tail of $\pi$ that starts with $\mathbf { \ b { d } } _ { 1 }$ instead of $\mathbf { { d } } _ { 0 }$ .

Theorem 3.1. For each $\begin{array} { r l r } { t } & { { } = } & { 1 , \dots . } \end{array}$ and $\begin{array} { r l } { \pi } & { { } = } \end{array}$ $( d _ { 0 } , \dotsc , d _ { t - 1 } ) \in \Pi _ { \mathrm { M R } } ^ { t }$ , the exponential values satisfy that

$$
\begin{array} { r l r } & { } & { { \pmb w } ^ { t } ( \pi ) = L ^ { d _ { t } } { \pmb w } ^ { t - 1 } ( \pi _ { 1 : t - 1 } ) , \qquad { \pmb w } ^ { 0 } ( \pi ) = - { \bf 1 } , } \\ & { } & { { \pmb w } ^ { t , \star } = L ^ { \star } { \pmb w } ^ { t - 1 , \star } = { \pmb w } ^ { t } ( \pi ^ { \star } ) \geq { \pmb w } ^ { t } ( \pi ) , \qquad { \pmb w } ^ { 0 , \star } = - { \bf 1 } , } \end{array}
$$

for some $\pi ^ { \star } \in \Pi _ { \mathrm { M D } } ^ { t }$

The proof of Theorem 3.1 is standard and has been established both in the context of ERMs (Hau, Petrik, and Ghavamzadeh 2023) and utility functions (Patek 1997).

The following corollary follows directly from Theorem 3.1 by algebraic manipulation and by the monotonicity of exponential value function transformation and the ERM.

Corollary 3.2. We have that

$$
\begin{array} { r l } & { g _ { t } ( \pi , \beta ) = \mathrm { E R M } _ { \beta } ^ { \mu } \left[ v _ { \tilde { s } _ { 0 } } ^ { t } ( \pi ) \right] , } \\ & { ~ g _ { t } ^ { \star } ( \beta ) = \mathrm { E R M } _ { \beta } ^ { \mu } \left[ v _ { \tilde { s } _ { 0 } } ^ { t , \star } \right] = \underset { \pi \in \Pi _ { \mathrm { M D } } } { \operatorname* { m a x } } ~ \mathrm { E R M } _ { \beta } ^ { \mu } \left[ v _ { \tilde { s } _ { 0 } } ^ { t } ( \pi ) \right] . } \end{array}
$$

# 3.2 Infinite Horizon

We now turn to construct infinite-horizon optimal policies as a limiting case of the finite horizon. An important quantity is the infinite-horizon exponential value function defined for each $\pi \in \Pi _ { \mathrm { H R } }$ as

$$
\pmb { w } ^ { \infty } ( \pi ) : = \operatorname* { l i m i n f } _ { t \to \infty } \pmb { w } ^ { t } ( \pi ) , \quad \pmb { w } ^ { \infty , \star } : = \operatorname* { l i m i n f } _ { t \to \infty } \pmb { w } ^ { t , \star } .
$$

Note again that we use the inferior limit because the limit may not be defined for non-stationary policies. The limiting infinite-horizon value functions $\scriptstyle w ^ { \infty } ( \pi )$ and $w ^ { \infty , \star }$ are defined analogously from ${ \pmb v } ^ { t } ( \pi )$ and $\scriptstyle v ^ { t , \star }$ using the inferior limit. The following theorem is the main result of this section. It shows that for an infinite horizon, the optimal exponential value function is attained by a stationary deterministic policy and is a fixed point of the exponential Bellman operator.

Theorem 3.3. Whenever $w ^ { \infty , \star } > - \infty$ there exists $\pi ^ { \star } =$ $( d ^ { \star } ) _ { \infty } \in \Pi _ { \mathrm { S D } }$ such that

$$
w ^ { \infty , \star } = w ^ { \infty } ( \pi ^ { \star } ) = L ^ { d ^ { \star } } w ^ { \infty , \star } ,
$$

and $w ^ { \infty , \star }$ is the unique value that satisfies this equation.

Corollary 3.4. Asuming the hypothesis of Theorem 3.3, we have that $v ^ { \infty , \star } = v ^ { \infty } ( \pi ^ { \star } )$ and

$$
g _ { \infty } ^ { \star } ( \beta ) = \mathrm { E R M } _ { \beta } ^ { \mu } \left[ v _ { \tilde { s } _ { 0 } } ^ { \infty , \star } \right] = \operatorname* { m a x } _ { \pi \in \Pi _ { \mathrm { S D } } } \mathrm { E R M } _ { \beta } ^ { \mu } \left[ v _ { \tilde { s } _ { 0 } } ^ { \infty } ( \pi ) \right] .
$$

We now outline the proof of Theorem 3.3; see Su, GrandCl´ement, and Petrik (2024, appendix D) for details. To establish Theorem 3.3, we show that $\mathbf { \Delta } _ { w } t , \star$ converges to a fixed point as $t \to \infty$ . Standard arguments do not apply to our setting (Puterman 2005; Kallenberg 2021; Patek 2001) because the ERM-TRC Bellman operator is not an $L _ { \infty }$ -contraction, it is not linear, and the values in value iteration do not increase or decrease monotonically. Although the exponential Bellman operator $L ^ { d }$ is linear, it may not be a contraction.

The main idea of the proof is to show that whenever the exponential value functions are bounded, the exponential Bellman operator must be weighted-norm contraction with a unique fixed point. To facilitate the analysis, we define $\pmb { w } ^ { t } \colon \dot { \Pi } _ { \mathrm { S R } } ^ { t } \times \mathbb { R } ^ { S ^ { \ast } }  \mathbb { R } ^ { S } , t \in \mathbb { N }$ for $z \in \mathbb { R } ^ { S }$ , $\dot { \pi } \in \Pi _ { \mathrm { S R } } ^ { t }$ , as

$$
\begin{array} { l } { { { \pmb w } ^ { t } ( \pi , z ) = L ^ { d } { \pmb w } ( \pi _ { 1 : t - 1 } ) = L ^ { d } L ^ { d } . . . L ^ { d } ( - z ) } } \\ { { \mathrm { } } } \\ { { \mathrm { } = - ( { \pmb B } ^ { d } ) ^ { t } z - \displaystyle \sum _ { k = 0 } ^ { t - 1 } ( { \pmb B } ^ { d } ) ^ { k } { \pmb b } ^ { d } . } } \end{array}
$$

The value $z$ can be interpreted as the exponential value function at the termination of the process following $\pi$ for $t$ periods. Note that $\mathbf { \Delta } \mathbf { w } ^ { t } ( \pi ) = \mathbf { w } ^ { t } ( \pi , \mathbf { 1 } \bar { ) }$ , $\forall \pi \in \Pi _ { \mathrm { M R } } , t \in \mathbb { N }$ .

An important technical result we show is that the only way a stationary policy’s return can be bounded is if the policy’s matrix has a spectral radius strictly less than 1.

Lemma 3.5. For each $\pi = ( d ) _ { \infty } \in \Pi _ { \mathrm { S R } }$ and ${ \boldsymbol { z } } \geq \mathbf { 0 }$ :

$$
\begin{array} { r } { { \pmb w } ^ { \infty } ( \pi , z ) > - \infty \quad \Rightarrow \quad { \ b \rho } ( { \pmb B } ^ { d } ) < 1 . } \end{array}
$$

Lemma 3.5 uses the transience property to show that the Perron vector (with the maximum absolute eigenvalue) $f$ of $B ^ { d }$ satisfies that $f ^ { \top } b ^ { d } > 0$ . Therefore, $\rho ( \mathbf { \bar { \boldsymbol { B } } } ^ { d } ) < \mathrm { ~ i ~ }$ is necessary for the series in (12) to be bounded.

The limitation of Lemma 3.5 is that it only applies to stationary policies. The lemma does not preclude the possibility that all stationary policies have unbounded returns, but a Markov policy with a bounded return exists. We construct an upper bound on $\mathbf { \Delta } w ^ { t , \star }$ that decreases monotonically in $t$ and converges to show this is impossible. The proof then concludes by squeezing $\mathbf { \Delta } w ^ { t , \star }$ between a lower and the upper bound with the same limits. This technique allows us to relax the limiting assumptions from prior work (Patek 2001; de Freitas, Freire, and Delgado 2020). Finally, our results imply an optimal stationary policy exists whenever the planning horizon $T$ is sufficiently large. Because the set $\Pi _ { \mathrm { S D } }$ is finite, one policy must be optimal for a sufficiently large $T$ . This property suggests behavior similar to turnpikes in discounted MDPs (Puterman 2005).

# 3.3 Algorithms

We now briefly describe the algorithms we use to compute the optimal ERM-TRC policies. Surprisingly, the main algorithms for discounted MDPs, including value iteration, policy iteration, and linear programming, can be adapted to this risk-averse setting with only minor modifications.

Value iteration is the most direct method for computing the optimal value function (Puterman 2005). The value iteration computes a sequence of $\begin{array} { r } { { \pmb w } ^ { k } , k = 0 , . . . } \end{array}$ such that

$$
\pmb { w } ^ { k + 1 } = L ^ { \star } \pmb { w } ^ { k } , \quad \pmb { w } ^ { 0 } = \mathbf { 0 } .
$$

The initialization of ${ \bf w } ^ { 0 } = { \bf 0 }$ is essential and guarantees convergence directly from the monotonicity argument used to prove Theorem 3.3.

Policy iteration (PI) starts by initializing with a stationary policy $\pi _ { 0 } = ( { \pmb d } ^ { 0 } ) _ { \infty } \in \Pi _ { \mathrm { S D } }$ . Then, for each iteration $k =$ $0 , \ldots , \operatorname { P I }$ alternates between the policy evaluation step and the policy improvement step:

$$
w ^ { k } = - ( { \boldsymbol { I } } - { \boldsymbol { B } } ^ { d ^ { k } } ) ^ { - 1 } { \boldsymbol { b } } ^ { d ^ { k } } , \ d ^ { k + 1 } \in \operatorname * { a r g m a x } _ { d \in \mathcal { D } } { \boldsymbol { B } } ^ { d } w ^ { k } - { \boldsymbol { b } } ^ { d } .
$$

PI converges because it monotonically improves the value functions when initialized with a policy $\dot { \pmb d } ^ { 0 }$ with bounded return (Patek 2001). However, we lack a practical approach to finding such an initial policy.

Finally, linear programming is a fast and convenient method for computing optimal exponential value functions:

$$
\operatorname* { m i n } \left\{ \mathbf { 1 } ^ { \top } \pmb { w } \mid \pmb { w } \in \mathbb { R } ^ { S } , \pmb { w } \geq - \pmb { b } ^ { a } + \pmb { B } ^ { a } \pmb { w } , \forall a \in \mathcal { A } \right\} .
$$

Here, $B _ { s , . . } ^ { a } = ( B _ { s , s _ { 1 } } ^ { a } , \cdot \cdot \cdot , B _ { s , s _ { S } } ^ { a } ) , B _ { s , s ^ { \prime } } ^ { a }$ and $b _ { s } ^ { a }$ are constructed as in (11). We use the shorthand $B ^ { a } = B ^ { d }$ and $\pmb { b } ^ { a } = \pmb { b } ^ { d }$ where $d _ { a ^ { \prime } } ( s ) = 1$ if $a = a ^ { \prime }$ for each $s \in \mathcal { S } , a ^ { \prime } \in \mathcal { A }$ .

It is important to note that the value functions, as well as the coefficients of $B ^ { d }$ may be irrational. It is, therefore, essential to study the sensitivity of the algorithms to errors in the input. However, this question is beyond the scope of the present paper, and we leave it for future work.

# 4 Solving EVaR Total Reward Criterion

This section shows that the EVaR-TRC objective can be reduced to a sequence of ERM-TRC problems, similarly to the discounted case (Hau, Petrik, and Ghavamzadeh 2023). As a result, an optimal stationary EVaR-TRC policy exists and can be computed using the methods described in Section 3.

Formally, we aim to compute a policy that maximizes the EVaR of the random return at some given fixed risk level $\alpha \in ( 0 , 1 )$ defined as

$$
\operatorname* { s u p } _ { \pi \in \Pi _ { \mathrm { H R } } } \operatorname* { l i m i n f } _ { t  \infty } \mathrm { E V a R } _ { \alpha } ^ { \pi , \mu } [ \sum _ { k = 0 } ^ { t - 1 } r ( \widetilde s _ { k } , \widetilde a _ { k } , \widetilde s _ { k + 1 } ) ] .
$$

In contrast with Ahmadi et al. (2021b), the objective in (14) optimizes EVaR rather than Nested EVaR.

# 4.1 Reduction to ERM-TRC

To solve (14), we exploit that EVaR can be defined in terms of ERM as shown in (4). To that end, define a function $h _ { t } \colon \Pi _ { \mathrm { H R } } \times \mathbb { R } \to \bar { \mathbb { R } }$ for $t \in \mathbb { N }$ as

$$
h _ { t } ( \pi , \beta ) : = g _ { t } ( \pi , \beta ) + \beta ^ { - 1 } \log ( \alpha ) ,
$$

where $g _ { t }$ is the ERM value of the policy defined in (6). Also, $h _ { t } ^ { \star } , h _ { \infty } , h _ { \infty } ^ { \star }$ are defined analogously in terms of $g _ { t } ^ { \star }$ , $g _ { \infty }$ , and $g _ { \infty } ^ { \star }$ respectively. The functions $h$ are useful, because by (4):

$$
\operatorname { E V a R } _ { \alpha } ^ { \pi , \mu } \left[ \sum _ { k = 0 } ^ { t - 1 } r ( { \tilde { s } } _ { k } , { \tilde { a } } _ { k } , { \tilde { s } } _ { k + 1 } ) \right] = \operatorname* { s u p } _ { \beta > 0 } h _ { t } ( \pi , \beta ) ,
$$

for each $\pi \in \Pi _ { \mathrm { H R } }$ and $t \in \mathbb { N }$ . However, note that the limit in the definition of $\mathrm { s u p } _ { \beta > 0 } h _ { \infty } ^ { \star } ( \beta )$ is inside the supremum unlike in the objective in (14).

There are two challenges with solving (14) by reducing it to (16). First, the supremum in the definition of EVaR in (4) may not be attained, as mentioned previously. Second, the functions $g _ { t } ^ { \star }$ and $h _ { t } ^ { \star }$ may not converge uniformly to $g _ { \infty } ^ { \star }$ and $h _ { \infty } ^ { \star }$ . Note that Theorem 3.3 only shows pointwise convergence when the functions are bounded.

To circumvent the challenges described above, we replace the supremum in (16) with a maximum over a finite set $\mathcal { B } ( \beta _ { 0 } , \bar { \delta } )$ of discretized $\beta$ values:

$$
\begin{array} { r } { \mathfrak { B } ( \beta _ { 0 } , \delta ) : = \{ \beta _ { 0 } , \beta _ { 1 } , \dots , \beta _ { K } \} , } \end{array}
$$

where $\delta > 0$ $> 0 , 0 < \beta _ { 0 } < \beta _ { 1 } < \cdot \cdot \cdot < \beta _ { K }$ , and

$$
\beta _ { k + 1 } : = \frac { \beta _ { k } \log \frac { 1 } { \alpha } } { \log \frac { 1 } { \alpha } - \beta _ { k } \delta } , \quad \beta _ { K } \geq \frac { \log \frac { 1 } { \alpha } } { \delta } ,
$$

for an appropriately chosen value $K$ for each $\beta _ { 0 }$ and $\delta$ . We assume that the denominator in the expression for $\beta _ { k + 1 }$ in Equation (17b) is positive; otherwise $\beta _ { k + 1 } = \infty$ and $\beta _ { k }$ is sufficiently large.

The construction in (17) resembles equations (19) and (20) in Hau, Petrik, and Ghavamzadeh (2023) but differs in the choice of $\beta _ { 0 }$ because Hoeffding’s lemma does not readily bound the TRC criterion.

The following proposition upper-bounds the value of $K$ ; see (Hau, Petrik, and Ghavamzadeh 2023, theorem 4.3) for a proof that $K$ is polynomial in $\delta$ .

Proposition 4.1. Assume a given $\beta _ { 0 } > 0$ and $\delta \in ( 0 , 1 )$ such that $\beta _ { 0 } \delta < \log { \frac { 1 } { \alpha } }$ . Then, to satisfy the condition in (17b), it is sufficient to choose $K$ as

$$
K : = \frac { \log z } { \log ( 1 - z ) } , \quad w h e r e \quad z : = \frac { \beta _ { 0 } \delta } { \log \frac { 1 } { \alpha } } .
$$

The following theorem shows that one can obtain an optimal ERM policy for an appropriately chosen $\beta$ that approximates an optimal EVaR policy arbitrarily closely.

Theorem 4.2. For any $\delta > 0$ , let

$$
( \pi ^ { \star } , \beta ^ { \star } ) \in \underset { ( \pi , \beta ) \in \Pi _ { \mathrm { S D } } \times \mathcal { B } ( \beta _ { 0 } , \delta ) } { \mathrm { a r g m a x } } h _ { \infty } ( \pi , \beta ) ,
$$

<html><body><table><tr><td>Algorithm1:SimpleEVaRalgorithm</td></tr><tr><td>Data:MDP and desired precision δ > O Result: δ-optimal policy π* ∈ IsD</td></tr><tr><td>while g(O)-g(β)>δdo β←β/2；</td></tr><tr><td>Construct B(βo,δ) as described in(17a) ; Compute</td></tr></table></body></html>

# 4.2 Algorithms

where $\beta _ { 0 } > 0$ is chosen such that $g _ { \infty } ^ { \star } ( 0 ) \leq g _ { \infty } ^ { \star } ( \beta _ { 0 } ) - \delta$ . Then the limits below exist and satisfy:

$$
\begin{array} { r l } & { \underset { t  \infty } { \mathrm { l i m } } \mathrm { E V a R } _ { \alpha } ^ { \pi ^ { \star } , \mu } [ \displaystyle \sum _ { k = 0 } ^ { t - 1 } r ( \tilde { s } _ { k } , \tilde { a } _ { k } , \tilde { s } _ { k + 1 } ) ] } \\ & { \quad \quad \quad \geq \underset { \pi \in \Pi _ { \mathrm { H R } } } { \operatorname* { s u p } } \underset { t  \infty } { \operatorname* { l i m } } \underset { \beta > 0 } { \operatorname* { s u p } } h ( \pi , \beta ) - \delta . } \end{array}
$$

We now propose a simple algorithm for computing a $\delta$ - optimal EVaR policy described in Algorithm 1. The algorithm reduces finding optimal EVaR-TRC policies to solving a sequence of ERM-TRC problems in (5). As Theorem 4.2 shows, there exists a $\delta$ -optimal policy such that it is ERMTRC optimal for some $\bar { \beta } \in \mathcal { B } ( \bar { \beta } _ { 0 } , \delta )$ . It is, therefore, sufficient to compute an ERM-TRC optimal policy for one of those $\beta$ values.

The second implication of Theorem 4.2 is that it suggests an algorithm for computing the optimal, or near-optimal, stationary policy. We summarize it in Section 4.2.

Note that the right-hand side in (19) is the $\delta$ -optimal objective in (14).

Corollary 4.4. Algorithm 1 computes the $\delta$ -optimal policy $\pi ^ { \star } \in \Pi _ { \mathrm { S D } }$ that satifies the condition (19).

The analysis above shows that Algorithm 1 is correct.

The first implication of Theorem 4.2 is that there exists an optimal stationary deterministic policy.

Corollary 4.3. There exists an optimal stationary deterministic policy $\pi ^ { \star } \in \Pi _ { \mathrm { S D } }$ that attains the supremum in (14).

Corollary 4.4 follows directly from Theorem 4.2 and from the existence of a sufficiently small $\beta _ { 0 }$ from the continuity of $g _ { \infty } ^ { \star } ( \beta )$ for positive $\beta$ around 0.

Algorithm 1 prioritizes simplicity over computational complexity and could be accelerated significantly. Evaluating each $\bar { h } _ { \infty } ^ { \star } ( \beta )$ requires computing an optimal ERM-TRC solution which involves solving a linear program. One could reduce the number of evaluations of $h _ { \infty } ^ { \star }$ needed by employing a branch-and-bound strategy that takes advantage of the monotonicity of $g _ { \infty } ^ { \star }$ .

An additional advantage of Algorithm 1 is that the overhead of computing optimal solutions for multiple risk levels $\alpha$ can be small if one selects an appropriate set $\mathcal { B }$ .

# 5 Numerical Evaluation

In this section, we illustrate our algorithms and formulations on tabular MDPs that include positive and negative rewards.

The ERM returns for the discounted and transient MDPs in Figure 1 with parameters $r = - 0 . 2$ , $\gamma = 0 . 9$ , $\epsilon = 0 . 9$ are shown in Figure 2. The figure shows that, as expected, the returns are identical in the risk-neutral objective (when $\beta = 0$ ). However, for $\beta > 0$ , the discounted and TRC returns differ significantly. The discounted return is unaffected by $\beta$ while the ERM-TRC return decreases with an increasing $\beta$ . Please see Su, Grand-Cl´ement, and Petrik (2024, appendix B) for more details.

To evaluate the effect of risk-aversion on the structure of the optimal policy, we use the gambler’s ruin problem (Hau, Petrik, and Ghavamzadeh 2023; B¨auerle and Ott 2011). In this problem, a gambler starts with a given amount of capital and seeks to increase it up to a cap $K$ . In each turn, the gambler decides how much capital to bet. The bet doubles or is lost with a probability $q$ and $1 - q$ , respectively. The gambler can quit and keep the current wealth; the game also ends when the gambler goes broke or achieves the cap $K$ The reward equals the final capital, except it is $^ { - 1 }$ when the gambler is broke. The initial state is chosen uniformly. In the formulation, we use $q = 0 . 6 8$ , and a cap is $K = 7$ . The algorithm was implemented in Julia 1.10, and is available at https://github.com/suxh2019/ERMLP. Please see Su, GrandCl´ement, and Petrik (2024, appendix F) for more details.

![](images/7c50da23e9489a918fcc4e19822694690c20fcbbfb421f9a3280785b7d6416fd.jpg)  
Figure 2: ERM values with TRC and discounted criteria.

![](images/91a8c27cb3e7b70daa8a721d8a4708c8d397b675f165c1b288573db6475817f4.jpg)  
Figure 3: The optimal EVaR-TRC policies.

Figure 3 shows optimal policies for four different EVaR risk levels $\alpha$ computed by Algorithm 1. The state represents how much capital the gambler holds. The optimal action indicates the amount of capital invested. The action 0 means quitting the game. Note that there is only one action when the capital is 0 and 7 for all policies so that action is neglected in Figure 3. Because the optimal policy is stationary, we can interpret and analyze it. The policies become notably less risk-averse as $\alpha$ increases. For example, when $\alpha = 0 . 2$ , the gambler is very risk-averse and always quits with the current capital. When $\alpha = 0 . 4$ , the gambler invests 1 when capital is greater than 1 and quits otherwise to avoid losing it all. When $\alpha = 0 . 9$ , the gambler makes bigger bets, increasing the probability of reaching the cap and losing all capital.

![](images/b3539d42ce431e532b7128a2f81f2cf044d3f221716b916144fbe246e7ae9de9.jpg)  
Figure 4: Distribution of the final capital for EVaR optimal policies.

To understand the impact of risk-aversion on the distribution of returns, we simulate the resulting policies over 7,000 episodes and show the distribution of capitals in Figure 4. When $\alpha = 0 . 2$ , the return follows a uniform distribution on [1, 7]. When $\alpha = 0 . 4$ , the returns are 1 and 7. When $\alpha = 0 . 7$ or 0.9, the returns are $- 1$ and 7. Overall, the figure shows that for lower values of $\alpha$ , the gambler gives up some probability of reaching the cap in exchange for a lower probability of losing all capital.

# 6 Conclusion and Future Work

We analyze transient MDPs with two risk measures: ERM and EVaR. We establish the existence of stationary deterministic optimal policies without any assumptions on the sign of the rewards, a significant departure from past work. Our results also provide algorithms based on value iteration, policy iteration, and linear programming for computing optimal policies.

Future directions include extensions to infinite-state TRC problems, risk-averse MDPs with average rewards, and partial-state observations.

# Acknowledgments

We thank the anonymous reviewers for their detailed reviews and thoughtful comments, which significantly improved the paper’s clarity. This work was supported, in part, by NSF grants 2144601 and 2218063. Julien Grand-Cl´ement was supported by Hi! Paris and Agence Nationale de la Recherche (Grant 11-LABX-0047).