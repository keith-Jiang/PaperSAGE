# Doubly-Bounded Queue for Constrained Online Learning: Keeping Pace with Dynamics of Both Loss and Constraint

Juncheng Wang\*, Bingjie Yan, Yituo Liu

Department of Computer Science, Hong Kong Baptist University

# Abstract

We consider online convex optimization with time-varying constraints and conduct performance analysis using two stringent metrics: dynamic regret with respect to the online solution benchmark, and hard constraint violation that does not allow any compensated violation over time. We propose an efficient algorithm called Constrained Online Learning with Doubly-bounded Queue (COLDQ), which introduces a novel virtual queue that is both lower and upper bounded, allowing tight control of the constraint violation without the need for the Slater condition. We prove via a new Lyapunov drift analysis that COLDQ achieves $\mathcal { O } ( T ^ { \frac { 1 + V _ { x } } { 2 } } )$ dynamic regret and $\mathcal { O } ( T ^ { V _ { g } } )$ hard constraint violation, where $V _ { x }$ and $V _ { g }$ capture the dynamics of the loss and constraint functions. For the first time, the two bounds smoothly approach to the bestknown $\mathcal { O } ( T ^ { \frac { 1 } { 2 } } )$ regret and $\mathcal { O } ( 1 )$ violation, as the dynamics of the losses and constraints diminish. For strongly convex loss functions, COLDQ matches the best-known ${ \mathcal { O } } ( \log T )$ static regret while maintaining the $\mathcal { O } ( T ^ { V _ { g } } )$ hard constraint violation. We further introduce an expert-tracking variation of COLDQ, which achieves the same performance bounds without any prior knowledge of the system dynamics. Simulation results demonstrate that COLDQ outperforms the stateof-the-art approaches.

# 1 Introduction

In many online learning applications, optimization losses and constraints are dynamic over time. Online Convex Optimization (OCO) (Shalev-Shwartz 2012; Hazan 2016), as the intersection of learning, optimization, and game, is a vital framework for solving online learning problems under uncertainty. It has broad applications such as advertisement placement (Balseiro, Lu, and Mirrokni 2020), load balancing (Hsu et al. 2021), network virtualization (Shi, Lin, and Fahmy 2021), and resource allocation (Wang et al. 2023).

In the standard OCO setting, a learner selects online decisions from a known convex set to minimize a sequence of time-varying convex loss functions. The information of each loss function, however, is only revealed to the learner after the decision has been made. Given this lack of current information, the objective of the learner becomes to minimize the regret, which is the accumulated difference between the losses incurred by their online decisions and those of some benchmark solutions. Zinkevich (2003) considered both static regret to an offline benchmark and dynamic regret to an online benchmark. The proposed online projected gradient descent algorithm provided a dynamic regret bound that smoothly approaches to $\mathcal { O } ( T ^ { \frac { 1 } { 2 } } )$ static regret, as the accumulated variation of the loss functions reduces, i.e., the OCO algorithm keeps pace with the dynamics of the losses.

The projection operation to strictly satisfy the constraints at each time can incur heavy computation. Furthermore, in many applications, the online decisions are subject to constraints that are allowed to be violated at certain time slots. Mahdavi, Jin, and Yang (2012) initiated the study on OCO with soft constraint violation, which measures the amount of compensated violations over time. In contrast, with a goal to limit the instantaneous violation, Yuan and Lamperski (2018) introduced a stronger notion of hard constraint violation that does not allow any compensated violation over time. For fixed constraints, the best-known soft and hard constraint violation bounds are both $\mathcal { O } ( 1 )$ (Yu and Neely 2020; Guo et al. 2022).

Most existing works on OCO with time-varying constraints focused on the static regret (Yu, Neely, and Wei 2017; Wei, Yu, and Neely 2020; Cao, Zhang, and Poor 2021; Sinha and Vaze 2024). Dynamic regret for time-varying constrained OCO was more recently studied (Chen, Ling, and Giannakis 2017; Cao and Liu 2019; Liu et al. 2022; Guo et al. 2022; Yi et al. 2023; Wang et al. 2023). As the accumulated variation of the constraint functions reduces, the bestknown soft and hard constraint violation bounds for timevarying constraints approach to $\mathcal { O } ( T ^ { \frac { 1 } { 2 } } )$ and $\mathcal { O } ( T ^ { \frac { 1 } { 2 } } \log T )$ , respectively (Wang et al. 2023; Sinha and Vaze 2024). However, none of the constraint violation bound recovers the best-known $\mathcal { O } ( 1 )$ violation for fixed constraints, i.e., the constrained OCO algorithms do not keep pace with the dynamics of the constraints.

The above discrepancies motivate us to pose the following key question: Can a constrained OCO algorithm provide a dynamic regret bound and a constraint violation bound that smoothly approach to the best-known $\mathcal { O } ( T ^ { \frac { 1 } { 2 } } )$ regret and $\mathcal { O } ( 1 )$ violation, respectively, as the dynamics of the losses and constraints diminish? Our answer is yes.

Contributions. We summarize our contributions below.   
• We propose an effective algorithm named Constrained Online Learning with Doubly-bounded Queue (COLDQ) for tackling OCO problems with time-varying constraints. Existing virtual-queue-based approaches rely on either a lower or an upper bound of the virtual queue to bound the constraint violation. In contrast, we introduce a novel virtual queue that enforces both a lower and an upper bound, without the commonly assumed Slater condition, to strictly control the constraint violation.   
• We analyze the performance of COLDQ via a new Lyapunov drift design that leverages both the lower and upper bounds of the virtual queue. We show that COLDQ provides $\mathcal { O } ( T ^ { \frac { 1 + V _ { x } } { 2 } } )$ dynamic regret and $\mathcal { O } ( T ^ { V _ { g } } )$ hard constraint violation, where $V _ { x }$ and $V _ { g }$ capture the dynamics of the losses and constraints (see definitions in (6) and (7)). For the first time, the two bounds smoothly approach to the best-known $\mathcal { O } ( T ^ { \frac { 1 } { 2 } } )$ regret and $\mathcal { O } ( 1 )$ violation as $V _ { x }  0$ and $V _ { g } \to 0$ .   
• When the loss functions are strongly convex, we show that COLDQ matches the best-known ${ \mathcal { O } } ( \log T )$ static regret, while maintaining the $\mathcal { O } ( T ^ { V _ { g } } )$ hard constraint violation. We further propose a variation of COLDQ with expert tracking that can achieve the same $\mathcal { O } ( T ^ { \frac { 1 + V _ { x } } { 2 } } )$ dynamic regret and $\mathcal { O } ( T ^ { V _ { g } } )$ hard constraint violation, without any prior knowledge about the system dynamics.   
• We conduct experiments to evaluate the practical performance of COLDQ on various applications involving both time-varying and fixed constraints. Numerical results confirm the effectiveness of COLDQ over the stateof-the-art approaches.

# 2 Related Work

# 2.1 OCO with Fixed Constraints

The seminal OCO work (Zinkevich 2003) achieved $\mathcal { O } ( T ^ { \frac { 1 } { 2 } } )$ static regret and a more meaningful $\mathcal { O } ( T ^ { \frac { 1 + V _ { x } } { 2 } } )$ dynamic regret. For strongly convex loss functions, Hazan, Agarwal, and Kale (2007) further improved the static regret bound to $\mathcal { O } ( \log T )$ . Dynamic regret has gained increased attention in subsequent OCO works (Hall and Willett 2015; Jadbabaie et al. 2015; Zhang, Lu, and Zhou 2018; Eshraghi and Liang 2020). Some further improvements in the dynamic regret have been achieved by exploiting the strong convexity and smoothness properties (Mokhtari et al. 2016; Zhang et al. 2017; Zhao and Zhang 2021). These works all used projection operations to strictly satisfy the constraints at each time.

To reduce the computational complexity incurred by the projection operation, Mahdavi, Jin, and Yang (2012) relaxed the complicated short-term constraints to long-term constraints, which need to be satisfied in the time-averaged manner. The proposed saddle-point-type algorithm achieved $\mathcal { O } ( T ^ { \frac { 1 } { 2 } } )$ static regret and $\mathcal { O } ( T ^ { \frac { 3 } { 4 } } )$ constraint violation. Subsequently, Jenatton, Huang, and Archambeau (2016) provided a trade-off between $\bar { \mathcal { O } } ( T ^ { \operatorname* { m a x } \{ c , 1 - c \} } )$ static regret and $\mathcal { O } ( T ^ { 1 - \frac { c } { 2 } } )$ constraint violation. For constraints satisfying the Slater condition, which excludes equality constraints, the virtual-queue-based algorithm (Yu and Neely 2020) reached $\mathcal { O } ( T ^ { \frac { 1 } { 2 } } )$ static regret and the best-known $\mathcal { O } ( 1 )$ constraint violation. These works all adopted the soft constraint violation that allows compensated violations over time.

In contrast, Yuan and Lamperski (2018) aimed at limiting the instantaneous constraint violation and considered a stronger notion of hard constraint violation, which does not allow any compensated violation over time. The proposed online algorithm obtained $\mathcal { O } ( T ^ { \operatorname* { m a x } \{ c , 1 - c \} } )$ static regret and $\mathcal { O } ( T ^ { 1 - \frac { c } { 2 } } )$ violation. The online algorithm in (Yi et al. 2021) provided $\mathcal { O } ( T ^ { \frac { 1 + V _ { x } } { 2 } } )$ dynamic regret and $\mathcal { O } ( T ^ { \frac { 1 } { 2 } } )$ hard constraint violation.

# 2.2 OCO with Time-Varying Constraints

For OCO problems with stochastic constraints, Yu, Neely, and Wei (2017) proposed a virtual-queue-based algorithm and achieved $\mathcal { O } ( T ^ { \frac { 1 } { 2 } } )$ expected static regret and $\mathcal { O } ( \bar { T } ^ { \frac { 1 } { 2 } } )$ expected soft constraint violation, under the Slater condition. Similar $\mathcal { O } ( T ^ { \frac { 1 } { 2 } } )$ performance guarantees were obtained under a weaker assumption on the Lagrangian multiplier (Wei, $\mathrm { Y u }$ , and Neely 2020). For time-varying constraints with unknown statistics, Cao, Zhang, and Poor (2021) reached $\mathcal { O } ( T ^ { \frac { 1 } { 2 } } )$ static regret and $\mathcal { O } ( T ^ { \frac { 3 } { 4 } } )$ soft constraint violation.

The modified saddle-point-type algorithm (Chen, Ling, and Giannakis 2017) attained $\mathcal { O } ( T ^ { \operatorname* { m a x } \{ \frac { 1 + V _ { x } } { 2 } , \frac { 1 + V _ { g } } { 2 } \} } )$ dynamic regret and $\mathcal { O } ( T ^ { \operatorname* { m a x } \{ 1 - V _ { x } , 1 - V _ { g } \} } )$ soft constraint violation, when the Slater constant is sufficiently large. Another saddle-point-type algorithm (Cao and Liu 2019) achieved $\mathcal { O } ( T ^ { \frac { 1 + V _ { x } } { 2 } } )$ dynamic regret and $\mathcal { O } ( T ^ { \frac { 3 + V _ { x } } { 4 } } )$ soft constraint violation. Liu et al. (2022) proposed a virtualqueue-based algorithm and obtained $\bar { \mathcal { O } } ( \bar { T } ^ { \frac { 1 + V _ { x } } { 2 } } )$ dynamic regret and $\mathcal { O } ( T ^ { \operatorname* { m a x } \{ \frac { 3 } { 4 } , V _ { g } \} } )$ soft constraint violation without the Slater condition. The delay-tolerant algorithm in (Wang et al. 2023) provided $\mathcal { O } ( T ^ { \operatorname* { i n a x } \{ \frac { 1 + V _ { x } } { 2 } , V _ { g } \} } )$ dynamic regret and $\mathcal { O } ( T ^ { \operatorname* { m a x } \{ \frac { 1 - V _ { x } } { 2 } , V _ { g } \} } )$ soft constraint violation under the Slater condition. Unfortunately, as the dynamics of the loss and constraint functions decrease, i.e., $V _ { x }  0$ and $V _ { g } \to 0$ , none of the above soft constraint violation bounds approaches to $\mathcal { O } ( 1 )$ .

For fixed constraints, Guo et al. (2022) provided the bestknown $\mathcal { O } ( 1 )$ hard constraint violation , and was able to keep the $\mathcal { O } ( T ^ { \frac { 1 } { 2 } } )$ static regret. For time-varying constraints, Guo et al. (2022) provided $\mathcal { O } ( T ^ { \frac { 3 } { 4 } } )$ violation and $\mathcal { O } ( T ^ { \frac { 1 } { 2 } + V _ { x } } )$ dynamic regret. Yi et al. (2023) achieved $\mathcal { O } ( T ^ { \frac { 1 } { 2 } } )$ static regret and $\mathcal { O } ( T ^ { \frac { 3 } { 4 } } )$ hard constraint violation under the distributed setting. Sinha and Vaze (2024) achieved the current best $\mathcal { O } ( T ^ { \frac { 1 } { 2 } } \log T )$ hard constraint violation and $\mathcal { O } ( T ^ { \frac { 1 } { 2 } } )$ static regret. Unfortunately still, none of the above hard constraint violation bounds smoothly approaches to $\mathcal { O } ( 1 )$ as the system dynamics reduce.

Comparisons. In Tables 1 and 2, we compare the performance bounds of COLDQ with the most relevant prior works. The comparison demonstrates that COLDQ keeps pace with the dynamics of both the losses and constraints. Below are a few points we would like to highlight.

Table 1: Performance bounds for time-varying constraints $( V _ { g } > 0 )$ ).   

<html><body><table><tr><td>Reference</td><td>Loss Function</td><td>Static Regret, Hard Constraint Violation</td><td>Dynamic Regret, Hard Constraint Violation</td></tr><tr><td>Guo et al. (2022)</td><td>Convex</td><td>O(T）， O(T）</td><td>O(T+Ve）， O(T）</td></tr><tr><td>Yi et al. (2023)</td><td>Convex</td><td>0(T2）， O(T)</td><td>N/A</td></tr><tr><td>Sinha and Vaze (2024)</td><td>Convex</td><td>O(T）， O(T2 log T)</td><td>N/A</td></tr><tr><td>COLDQ (this work)</td><td>Convex</td><td>O(T) O(TVg)</td><td>O(T+V）, O(TVg)</td></tr><tr><td>Guo et al. (2022)</td><td>Strongly convex</td><td>O(log T), O(T² (log T))</td><td>O(T+Vx）， O(T² (logT)²)</td></tr><tr><td>Yi et al. (2023)</td><td>Strongly convex</td><td>O(TC), O(T1 ）</td><td>N/A</td></tr><tr><td>Sinha and Vaze (2024)</td><td>Strongly convex</td><td>O(log T), O(T²(logT))</td><td>N/A</td></tr><tr><td>COLDQ (this work)</td><td>Strongly convex</td><td>O(log T), O(TV）</td><td>O(T V）， O(TVg)</td></tr></table></body></html>

Table 2: Performance bounds for fixed constraints $( V _ { g } = 0 )$ ).   

<html><body><table><tr><td>Reference</td><td>Loss Function</td><td>StaticRegret,Hard Constraint Violation</td><td>Dynamic Regret,Hard Constraint Violation</td></tr><tr><td>Yi et al. (2021)</td><td>Convex</td><td>O(T²)， O(T）</td><td>(T）， O(T）</td></tr><tr><td>Guo et al. (2022)</td><td>Convex</td><td>O(T2）， 0(1)</td><td>O(Ty O(log T)</td></tr><tr><td>Sinha and Vaze (2024)</td><td>Convex</td><td>O(T2）, O(T2 log T)</td><td>N/A</td></tr><tr><td>COLDQ (this work)</td><td>Convex</td><td>O(T2）, 0(1)</td><td>O(T ）， 0(1) V</td></tr><tr><td>Yi et al. (2021)</td><td>Strongly convex</td><td>O(log T), O(log T)</td><td>O(T O(T）</td></tr><tr><td>Guo et al. (2022)</td><td>Strongly convex</td><td>O(log T), 0(1)</td><td>O(T O(log T)</td></tr><tr><td>Sinha and Vaze (2024)</td><td>Strongly convex</td><td>O(log T), O(T²(logT)²)</td><td>N/A</td></tr><tr><td>COLDQ (this work)</td><td>Strongly convex</td><td>O(log T), 0(1)</td><td>O(T V） 0(1)</td></tr></table></body></html>

• For time-varying constraints and convex loss functions, COLDQ improves upon the current best $\mathcal { O } ( T ^ { \frac { 1 } { 2 } } \log T )$ hard constraint violation bound (Sinha and Vaze 2024) and achieves an $\mathcal { O } ( T ^ { V _ { g } } )$ bound instead. Furthermore, COLDQ enhances the current best $\mathcal { O } ( T ^ { \frac { 1 } { 2 } + V _ { x } } )$ dynamic regret (Guo et al. 2022) to $\mathcal { O } ( T ^ { \frac { 1 + V _ { x } } { 2 } } )$ . • For time-varying constraints and strongly convex loss functions, COLDQ improves the current best $\mathcal { O } ( T ^ { \frac { 1 } { 2 } + V _ { x } } )$ dynamic regret and $\mathcal { O } ( T ^ { \frac { 1 } { 2 } } ( \log T ) ^ { \frac { 1 } { 2 } } )$ hard constraint violation (Guo et al. 2022) to $\mathcal { O } ( T ^ { \frac { 1 + V _ { x } } { 2 } } )$ and $\mathcal { O } ( T ^ { V _ { g } } )$ . • For fixed constraints and both convex and stronglyconvex loss functions, COLDQ improves the current best $\mathcal { O } ( \log T )$ hard constraint violation (Guo et al. 2022) to $\mathcal { O } ( 1 )$ , while maintaining the $\mathcal { O } ( T ^ { \frac { 1 + V _ { x } } { 2 } } )$ dynamic regret.

# 3 Constrained Online Convex Optimization

We can consider the constrained OCO problem as an iterative game between a learner and the system over $T$ time slots. At each time $t$ , the learner first selects a decision $\mathbf { x } _ { t }$ from a known feasible set $\boldsymbol { \mathcal { X } } \subseteq \mathbb { R } ^ { p }$ . The loss function $f _ { t } ( \mathbf { x } ) \ : \ \mathbb { R } ^ { p } \ \to \ \mathbb { R }$ and the constraint function $\begin{array} { r } { \mathbf { g } _ { t } ( \mathbf { x } ) \mathbf { \Theta } = \mathbf { \Theta } } \end{array}$ $[ g _ { t } ^ { 1 } ( \mathbf { x } ) , \ldots , g _ { t } ^ { N } ( \mathbf { x } ) ] ^ { \top } : \mathbb { R } ^ { p }  \mathbb { R } ^ { N }$ are then revealed to the learner, incurring a loss of $f _ { t } ( \mathbf { x } _ { t } )$ and a constraint violation of ${ \bf g } _ { t } ( { \bf x } _ { t } )$ . Both the loss function $f _ { t } ( \mathbf { x } )$ and the constraint function $\mathbf { g } _ { t } ( \mathbf { x } )$ are unknown a priori and are allowed to change arbitrarily over time.

The goal of the learner is to select from the feasible set an online decision sequence that minimizes the total accumulated loss under time-varying constraints. This gives rise to the following time-varying constrained OCO problem

$$
\begin{array} { r l } { \mathbf { P } : \quad \displaystyle \operatorname* { m i n } _ { \{ \mathbf { x } _ { t } \in \mathcal { X } \} } } & { \displaystyle \sum _ { t = 1 } ^ { T } f _ { t } ( \mathbf { x } _ { t } ) } \\ { \mathrm { s . t . } \quad \quad \mathbf { g } _ { t } ( \mathbf { x } _ { t } ) \preceq \mathbf { 0 } , \quad \forall t . } \end{array}
$$

When $\mathbf { g } _ { t } ( \mathbf { x } ) = \mathbf { g } ( \mathbf { x } ) , \forall t ,$ P becomes the OCO problem with fixed constraints.

# 3.1 Assumptions

We make some mild and common assumptions on $\chi$ , $f _ { t } ( \mathbf { x } )$ , and $\mathbf { g } _ { t } ( \mathbf { x } )$ in the constrained OCO literature.

Assumption 1. The feasible set $\chi$ is convex and bounded, i.e., $\exists R > 0$ , such that $\| \mathbf x - \mathbf y \| \leq R , \forall \mathbf x , \mathbf y \in \mathcal { X }$ .

Assumption 2. The loss functions are convex with bounded subgradient over $\chi$ , i.e., $\exists D > 0$ , such that $\left| \left| \nabla f _ { t } ( \mathbf { x } ) \right| \right| \leq$ $D , \forall { \mathbf x } \in \boldsymbol { \mathcal X } , \forall t$ .

Assumption 3. The constraint functions are convex and bounded over $\chi$ , i.e., $\exists G > 0$ , such that $| g _ { t } ^ { n } ( \mathbf { x } ) | \leq G , \forall \mathbf { x } \in$ $\boldsymbol { \mathscr { X } } , \forall t , \forall n$ .

Note that we do not require the commonly assumed Slater condition (or any of its relaxed version), on each of the constraint function at each time, i.e., $\exists \tilde { \mathbf { x } } _ { t } \in \mathcal { X }$ and $\delta > 0$ , such that $g _ { t } ^ { n } ( \tilde { \mathbf { x } } _ { t } ) < - \delta , \forall t , \forall n$ , (Yu, Neely, and Wei 2017; Chen, Ling, and Giannakis 2017; Yu and Neely 2020; Wei, Yu, and Neely 2020; Wang et al. 2023). The Slater condition, i.e., the existence of a shared interior point assumption, excludes equality constraints that are common in many practical applications.

# 3.2 Performance Metrics

Finding an optimal solution to $\mathbf { P }$ is known to be impossible since the current information about $f _ { t } ( \mathbf { x } )$ and $\bar { \mathbf { g } _ { t } } ( \mathbf { x } )$ is not available when selecting $\mathbf { x } _ { t }$ at each time $t$ . Instead, the OCO literature measures the performance of a constrained online algorithm, by comparing it with some solution benchmarks. There are two commonly used benchmarks. One is the fixed offline solution benchmark $\mathbf { x } ^ { \star } \in$ arg $\begin{array} { r } { \operatorname* { m i n } _ { \mathbf { x } \in \mathcal { X } } \{ \sum _ { t = 1 } ^ { T } f _ { t } ( \mathbf { x } ) | \mathbf { g } _ { t } ( \mathbf { x } ) \ \preceq \ \mathbf { 0 } , \forall t \} } \end{array}$ . The resulting static regret is defined as

$$
\mathrm { R E G } _ { \mathrm { s } } ( T ) \triangleq \sum _ { t = 1 } ^ { T } \big [ f _ { t } ( \mathbf { x } _ { t } ) - f _ { t } ( \mathbf { x } ^ { \star } ) \big ] .
$$

Another one is the dynamic online solution benchmark $\mathbf { x } _ { t } ^ { \star } \in$ arg $\begin{array} { r } { \operatorname* { m i n } _ { \mathbf { x } \in \mathcal { X } } \big \{ f _ { t } ( \mathbf { x } ) \big | \mathbf { g } _ { t } ( \mathbf { x } ) \preceq \mathbf { 0 } , \forall t \big \} } \end{array}$ . The resulting dynamic regret is defined as

$$
\mathrm { R E G } _ { \mathrm { d } } ( T ) \triangleq \sum _ { t = 1 } ^ { T } \big [ f _ { t } ( \mathbf { x } _ { t } ) - f _ { t } ( \mathbf { x } _ { t } ^ { \star } ) \big ] .
$$

The difference between the dynamic regret in (3) and the static regret in (2) can scale linearly with $T$ , i.e., $\mathrm { R E G } _ { \mathrm { d } } ( T ) -$ $\mathrm { R E G } _ { \mathrm { s } } ( \bar { T } ) = \mathcal { O } ( T )$ (Besbes, Gur, and Zeevi 2015). For a thorough analysis, in this work, we provide upper bounds on both the dynamic regret and the static regret.

There are also two commonly used performance metrics to quantify how much the time-varying constraints (1) are violated. One is the soft constraint violation defined as

$$
\mathrm { V I O } _ { \mathrm { s } } ( T ) \triangleq \sum _ { n = 1 } ^ { N } \bigg [ \sum _ { t = 1 } ^ { T } g _ { t } ^ { n } ( \mathbf { x } _ { t } ) \bigg ] _ { + } ,
$$

where $[ \cdot ] _ { + }$ is the projector onto the non-negative space. The above soft constraint violation allows the violation at individual time slots to be compensated over time. Another one is the hard constraint violation defined as

$$
\operatorname { V I O } _ { \mathrm { h } } ( T ) \triangleq \sum _ { n = 1 } ^ { N } \sum _ { t = 1 } ^ { T } \big [ g _ { t } ^ { n } ( \mathbf { x } _ { t } ) \big ] _ { + } .
$$

This hard constraint violation does not allow the violation at a time slot to be compensated by any other time slot. From the definitions of the soft and hard constraint violations in (4) and (5), we readily have $\mathrm { V I O } _ { \mathrm { s } } ( T ) \leq \mathrm { V I O } _ { \mathrm { d } } ( T )$ . In this work, we provide upper bounds on the hard constraint violation, which apply to the soft constraint violation as well.

# 3.3 Variation Measures

In the context of time-varying constrained OCO, it is desirable for an online algorithm to simultaneously achieve sublinear dynamic regret and sublinear constraint violation. This dual objective, however, can be intractable due to the adversarial variations of the losses and constraints. The performance guarantees of a constrained OCO algorithm are inherently linked to the temporal variations of both $\{ f _ { t } ( \mathbf { x } ) \} _ { t = 1 } ^ { T }$ and $\{ { \bf \dot { g } } _ { t } ( { \bf x } ) \} _ { t = 1 } ^ { T }$ . Therefore, it is necessary to quantify the dynamics of the underlying time-varying constrained OCO problem $\mathbf { P }$ .

There are two common variation measures in the literature. The first one measures the fluctuations in the dynamic online solution benchmark $\{ \mathbf { x } _ { t } ^ { \star } \} _ { t = 1 } ^ { T }$ , which is also referred to as the path length (Chen, Ling, and Giannakis 2017; Cao and Liu 2019; Yi et al. 2021; Guo et al. 2022; Liu et al. 2022; Wang et al. 2023), given by

$$
\sum _ { t = 2 } ^ { T } \| \mathbf x _ { t } ^ { \star } - \mathbf x _ { t - 1 } ^ { \star } \| = \mathcal { O } \big ( T ^ { V _ { x } } \big ) ,
$$

where $V _ { x } \in [ 0 , 1 ]$ represents the time variability of the dynamic online solution benchmark.

The other one focuses on the fluctuations in the constraint functions $\{ \mathbf { g } _ { t } \} _ { t = 1 } ^ { T }$ (Chen, Ling, and Giannakis 2017; Liu et al. 2022; Wang et al. 2023)

$$
\sum _ { t = 2 } ^ { T } \operatorname* { m a x } _ { \mathbf { x } \in \mathcal { X } } \| \mathbf { g } _ { t } ( \mathbf { x } ) - \mathbf { g } _ { t - 1 } ( \mathbf { x } ) \| = \mathcal { O } \big ( T ^ { V _ { g } } \big ) ,
$$

where $V _ { g } \in [ 0 , 1 ]$ . Note that for the fixed offline solution benchmark, i.e., $\mathbf { x } _ { t } ^ { \star } = \mathbf { x } ^ { \star } , \forall t$ , we have $V _ { x } = 0$ . Similarly, for fixed constraint functions, i.e., ${ \bf g } _ { t } ( { \bf x } ) = { \bf g } ( { \bf x } ) , \forall t$ , we have $V _ { g } = 0$ .

# 4 Constrained Online Learning with Doubly-bounded Queue (COLDQ)

We present the COLDQ algorithm for solving $\mathbf { P }$ . In COLDQ, we introduce a novel doubly-bounded virtual queue and a new Lyapunov drift design, which will be shown to provide improved regret and constraint violation bounds.

# 4.1 Doubly-Bounded Virtual Queue

We introduce a novel virtual queue $Q _ { t } ^ { n }$ to track the amount of violation for each time-varying constraint $n$ . At the end of each time $t > 1$ , after observing the constraint function ${ \bf g } _ { t } ( { \bf x } )$ , we update the virtual queue as:

$$
Q _ { t } ^ { n } = \operatorname* { m a x } \big \{ ( 1 - \eta ) Q _ { t - 1 } ^ { n } + [ g _ { t } ^ { n } ( \mathbf { x } _ { t } ) ] _ { + } , \gamma \big \} ,
$$

where $\eta ~ \in ~ ( 0 , 1 )$ and $\gamma \in ( 0 , \frac { G } { \eta } )$ are two algorithm parameters. Our virtual queue updating rule (8) includes an additional penalty term $- \eta Q _ { t - 1 } ^ { n }$ to avoid the virtual queue from becoming excessively large. Furthermore, (8) enforces a minimum virtual queue length $\gamma$ to prevent the constraint violation being overly large. In the following lemma, we show that without the Slater condition, (8) leads to both a lower bound and an upper bound on the virtual queue.1

Lemma 1. Under Assumption 3, the virtual queue in (8) has both a lower and an upper bound for each time $t$ and each constraint $n$ , given by

$$
\gamma \leq Q _ { t } ^ { n } \leq { \frac { G } { \eta } } .
$$

As shown in Lemma 1, the parameter $\eta$ can be seen as a virtual Slater constant for the constraints (1) in $\mathbf { P }$ . This means that the virtual queue upper bound is independent of the actual Slater constant. Furthermore, the parameter $\gamma$ ensures that the virtual queue length is always strictly positive. Our virtual queue updating rule (8) leads to straightforward lower and upper bounds on the virtual queue itself. These virtual queue bounds, however, cannot be directly translated into a bound on the constraint violation. In the following section, we will establish a connection between our virtual queue and the hard constraint violation via a new Lyapunovdrift-based approach.

# 4.2 Lyapunov Drift

We define a new Lyapunov drift for each $t > 1$ as

$$
\Delta _ { t - 1 } \triangleq \frac 1 2 \sum _ { n = 1 } ^ { N } ( Q _ { t } ^ { n } - \gamma ) ^ { 2 } - \frac 1 2 \sum _ { n = 1 } ^ { N } ( Q _ { t - 1 } ^ { n } - \gamma ) ^ { 2 } .
$$

Compared with the standard Lyapunov drift that uses the quadratic virtual queue as the Lyapunov function, each virtual queue $Q _ { t } ^ { n }$ is penalized by its lower bound $\gamma$ in (10). The subsequent lemma establishes an upper bound for $\Delta _ { t - 1 }$ , leveraging both the lower and upper bounds of $Q _ { t } ^ { n }$ in (9).

Lemma 2. Under Assumption 3, the Lyapunov drift in $( l O )$ is upper bounded for any $t > 1$ by

$$
\begin{array} { l } { \displaystyle \Delta _ { t - 1 } \leq \sum _ { n = 1 } ^ { N } Q _ { t - 1 } ^ { n } [ g _ { t - 1 } ^ { n } ( \mathbf { x } _ { t } ) ] _ { + } - \gamma \sum _ { n = 1 } ^ { N } [ g _ { t } ^ { n } ( \mathbf { x } _ { t } ) ] _ { + } } \\ { \displaystyle \quad \quad + \frac { G \sqrt { N } } { \eta } \underset { \mathbf { x } \in \mathcal { X } } { \operatorname* { m a x } } \| \mathbf { g } _ { t } ( \mathbf { x } ) - \mathbf { g } _ { t - 1 } ( \mathbf { x } ) \| + 2 N G ^ { 2 } . } \end{array}
$$

The above Lyapunov drift upper bound comprises two key terms. The second term on the right-hand side (RHS) of (11) accounts for the hard constraint violation $\textstyle \sum _ { n = 1 } ^ { N } [ g _ { t } ^ { n } ( \mathbf { x } _ { t } ) ] _ { + }$ , scaled by the virtual queue lower bound $\gamma$ . The third term on the RHS of (11) captures the fluctuation in the two adjacent constraint functions $\mathrm { m a x } _ { \mathbf { x } \in \mathcal { X } } \| \mathbf { g } _ { t } ( \mathbf { x } ) - \mathbf { g } _ { t - 1 } ( \mathbf { x } ) \|$ , scaled by the virtual queue upper bound $\frac { G } { \eta }$ . These two terms are crucial for relating the hard constraint violation $\mathrm { V I O } _ { \mathrm { h } } ( T )$ to the constraint variation measure in (7), leading to improved performance bounds over the current-best results.

# 4.3 Algorithm Intuition

We solve the following per-slot optimization problem $\mathbf { P } _ { t }$ to determine the decision $\mathbf { x } _ { t }$ at each time $t > 1$

$$
\begin{array} { l } { \displaystyle \mathbf { P } _ { t } : \displaystyle \operatorname* { m i n } _ { \mathbf { x } \in \mathcal { X } } \big \langle \nabla f _ { t - 1 } \big ( \mathbf { x } _ { t - 1 } \big ) , \mathbf { x } - \mathbf { x } _ { t - 1 } \big \rangle + \alpha _ { t - 1 } \| \mathbf { x } - \mathbf { x } _ { t - 1 } \| ^ { 2 } } \\ { \displaystyle \qquad + \displaystyle \sum _ { n = 1 } ^ { N } Q _ { t - 1 } ^ { n } \big [ g _ { t - 1 } ^ { n } ( \mathbf { x } ) \big ] _ { + } } \end{array}
$$

Algorithm 1 Constrained Online Learning with Doublybounded Queue (COLDQ)   

<html><body><table><tr><td>1:Initialize non-decreasing sequence {αt} ∈ (O,+00). η ∈ (0,1),and γ ∈ (0,).Choose X1 ∈ X arbitrar- ily and let Q’ = γ, ∀n. At each time t = 2,..,T, do the following: 2:Update decision Xt by solving Pt.</td></tr></table></body></html>

where $\alpha _ { t - 1 } > 0$ is another algorithm parameter and is nondecreasing, i.e., $\alpha _ { t } \geq \alpha _ { t - 1 } , \forall t > 1 .$ . From the Lyapunov drift upper bound established in Lemma 2, we can see the intuition behind solving $\mathbf { P } _ { t }$ . Specifically, the objective is to greedily minimize the upper bound on the following drift plus penalty term:

$$
\begin{array} { r } { \Delta _ { t - 1 } + \langle \nabla f _ { t - 1 } ( { \mathbf x } _ { t - 1 } ) , { \mathbf x } - { \mathbf x } _ { t - 1 } \rangle + \alpha _ { t - 1 } \| { \mathbf x } - { \mathbf x } _ { t - 1 } \| ^ { 2 } . } \end{array}
$$

Note that the last two terms on the RHS of (11) are independent of ${ \bf x } _ { t }$ , and second term is omitted in $\mathbf { P } _ { t }$ since $\mathbf { g } _ { t } ( \mathbf { x } )$ is not available when choosing $\mathbf { x } _ { t }$ .

Minimizing the above penalty term $\langle \nabla f _ { t - 1 } ( \mathbf { x } _ { t - 1 } ) , \mathbf { x } -$ $\mathbf { x } _ { t - 1 } \rangle + \alpha _ { t - 1 } \mathbf { \bar { \| } x - x } _ { t - 1 } \| ^ { \bar { 2 } }$ itself is equivalent to performing the standard gradient descent xt 1 − 2α1 $\begin{array} { r } { { \bf x } _ { t - 1 } \stackrel { * } { - } \frac { 1 } { 2 \alpha _ { t - 1 } } \nabla \dot { f } _ { t - 1 } \big ( { \bf x } _ { t - 1 } \big ) } \end{array}$ The optimal solution to $\mathbf { P } _ { t }$ depends on the amount of constraint violation induced by such gradient descent. If $\begin{array} { r l } { g _ { t - 1 } ^ { n } ( \mathbf { x } _ { t - 1 } - \frac { 1 } { 2 \alpha _ { t - 1 } } \nabla f _ { t - 1 } ( \mathbf { x } _ { t - 1 } ) ) } & { \leq \ \mathbf { \widetilde { 0 } } , \forall n . } \end{array}$ , i.e., the gradient descent does not incur any constraint violation, then $\begin{array} { r } { \mathbf { x } _ { t } \in \arg \operatorname* { m i n } _ { \mathbf { x } \in \mathcal { X } } \{ \mathbf { x } _ { t - 1 } - \frac { 1 } { 2 \alpha _ { t - 1 } } \nabla { f _ { t - 1 } } ( \mathbf { x } _ { t - 1 } ) \} } \end{array}$ is the optimal solution to $\mathbf { P } _ { t }$ . Otherwise, the gradient descent direction is shifted towards minimizing $Q _ { t - 1 } ^ { n } \big [ g _ { t - 1 } ^ { n } ( \mathbf { x } _ { t } ) \big ] _ { + }$ to reduce the constraint violation. The virtual queue $Q _ { t - 1 } ^ { n }$ balances between loss minimization and violation reduction.

# 4.4 The COLDQ Algorithm

In Algorithm 1, we summarize the proposed COLDQ algorithm. COLDQ consists of two main steps. The first step updates the decision variable $\mathbf { x } _ { t }$ at the beginning of each time $t$ based on the gradient of the previous loss function $\nabla f _ { t - 1 } \big ( \mathbf { x } _ { t - 1 } \big )$ and the previous constraint function $\mathbf { g } _ { t - 1 } ( \mathbf { x } )$ . This primal update is designed to balance the accumulated loss minimization and the constraint violation control. The second step updates the virtual queue $Q _ { t } ^ { n } , \forall n$ at the end of each $t$ , after observing the constraint function ${ \bf g } _ { t } ( { \bf x } )$ . This dual update is to track the amount of hard constraint violation. Note that COLDQ solves at each time $t$ a convex optimization problem $\mathbf { P } _ { t }$ , which can be efficiently solved in polynomial time. We will discuss the algorithm parameters $\alpha _ { t } , \eta , \gamma$ to derive the best performance bounds for COLDQ in Section 5.5.2

# Performance Bounds of COLDQ 5.1 Preliminary Analysis

The subsequent lemma establishes a per-slot performance guarantee of the COLDQ algorithm.

Lemma 3. Under Assumptions 1-3, the online decision sequence generated by COLDQ satisfies the following inequality for any $t > 1$ :

$$
\left[ f _ { t - 1 } ( \mathbf { x } _ { t - 1 } ) - f _ { t - 1 } ( \mathbf { x } _ { t - 1 } ^ { \star } ) \right] + \sum _ { n = 1 } ^ { N } Q _ { t - 1 } ^ { n } [ g _ { t - 1 } ^ { n } ( \mathbf { x } _ { t } ) ] _ { + }
$$

$$
\begin{array} { r l } {  { \le 2 R \alpha _ { t - 1 } \| \mathbf { x } _ { t } ^ { \star } - \mathbf { x } _ { t - 1 } ^ { \star } \| + R ^ { 2 } ( \alpha _ { t } - \alpha _ { t - 1 } ) + \frac { D ^ { 2 } } { 4 \alpha _ { t - 1 } } } } \\ & { \quad + ( \alpha _ { t - 1 } \| \mathbf { x } _ { t - 1 } ^ { \star } - \mathbf { x } _ { t - 1 } \| ^ { 2 } - \alpha _ { t } \| \mathbf { x } _ { t } ^ { \star } - \mathbf { x } _ { t } \| ^ { 2 } ) . \quad ( 1 ^ { \prime } } \end{array}
$$

Lemma 3 is the key to bridge the per-slot optimization problem $\mathbf { P } _ { t }$ and the performance bounds of COLDQ. From Lemma 3, we can separately bound the dynamic regret and the hard constraint violation by substituting different lower bounds on $Q _ { t - 1 } ^ { n } [ g _ { t - 1 } ^ { n } ( \mathbf { x } _ { t } ) ] _ { + }$ into (12). Note that adopting the soft constraint violation measure necessitates jointly bounding the regret and constraint violation.

# 5.2 Bounding Dynamic Regret

The virtual queue length is always positive due to its lower bound, and the hard constraint violation is non-negative by definition. Hence, their product $Q _ { t - 1 } ^ { n } [ g _ { t - 1 } ^ { n } ( { \bf x } _ { t } ) ] _ { + }$ in (12) is guaranteed to be non-negative. Unlike the analysis for softconstrained OCO algorithms, this unique property enables us to bound the dynamic regret of COLDQ in the following theorem, without needing to explicitly consider the hard constraint violation.

Theorem 1. Under Assumptions 1-3, the dynamic regret of the COLDQ algorithm is upper bounded by

$$
\begin{array} { r l r } {  { \mathrm { R E G } _ { \mathrm { d } } ( T ) \le 2 R \displaystyle \sum _ { t = 2 } ^ { T } \alpha _ { t - 1 } \| \mathbf { x } _ { t } ^ { \star } - \mathbf { x } _ { t - 1 } ^ { \star } \| + \frac { D ^ { 2 } } { 4 } \sum _ { t = 1 } ^ { T } \frac { 1 } { \alpha _ { t } } } } \\ & { } & { \quad + R ^ { 2 } \alpha _ { T } + D R . } \end{array}
$$

From Theorem 1, we readily have an upper bound on the static regret $\mathrm { R E G } _ { \mathrm { s } } ( T )$ by substituting $\mathbf { x } _ { t } ^ { \star } = \mathbf { x } ^ { \star } , \forall t$ into the dynamic regret bound (13).

# 5.3 Bounding Hard Constraint Violation

The following theorem establishes a bound on the hard constraint violation incurred by the COLDQ algorithm. This is achieved by converting the term $Q _ { t - 1 } ^ { n } [ g _ { t - 1 } ^ { n } ( \mathbf { \bar { x } } _ { t } ) ] _ { + }$ in (12) to $\left[ g _ { t } ^ { n } ( \mathbf { x } _ { t } ) \right] _ { + }$ through the Lyapunov drift upper bound (11).

Theorem 2. Under Assumptions 1-3, the hard constraint violation of COLDQ is upper bounded by

$$
\begin{array} { l } { { \displaystyle \mathrm { V I O } _ { \mathrm { h } } ( T ) \leq \frac { G \sqrt { N } } { \eta \gamma } \sum _ { t = 2 } ^ { T } \operatorname* { m a x } _ { \mathbf { x } \in \mathcal { X } } \| \mathbf { g } _ { t } ( \mathbf { x } ) - \mathbf { g } _ { t - 1 } ( \mathbf { x } ) \| } } \\ { ~ + \frac { 2 R } { \gamma } \displaystyle \sum _ { t = 2 } ^ { T } \alpha _ { t - 1 } \| \mathbf { x } _ { t } ^ { \star } - \mathbf { x } _ { t - 1 } ^ { \star } \| + \frac { D ^ { 2 } } { 4 \gamma } \displaystyle \sum _ { t = 1 } ^ { T } \frac { 1 } { \alpha _ { t } } } \\ { ~ + \left( D R + 2 N G ^ { 2 } \right) \frac { T } { \gamma } + R ^ { 2 } \frac { \alpha _ { T } } { \gamma } + N G . } \end{array}
$$

To establish a hard constraint violation bound for fixed constraints, we can simply substitute ${ \bf g } _ { t } ( { \bf x } ) = { \bf g } ( { \bf x } ) , \forall t$ into the bound for time-varying constraints (14).

# 5.4 Strongly Convex Case

We further consider the case of strongly convex loss functions as in (Yi et al. 2021; Guo et al. 2022; Yi et al. 2023).

Assumption 4. The loss functions are $\mu$ -strongly convex in $\chi$ for some $\mu > 0$ i.e., $f _ { t } ( \mathbf { y } ) \geq f _ { t } ( \mathbf { x } ) + \langle \nabla f _ { t } ( \mathbf { x } ) , \mathbf { y } - \mathbf { x } \rangle +$ $\mu \| \mathbf { y } - \mathbf { x } \| ^ { 2 } , \forall \mathbf { x } , \mathbf { y } \in { \mathcal { X } } , \forall t$ .

The following theorem provides a static regret bound for COLDQ with Assumption 4.

Theorem 3. Under Assumptions 1-4, the static regret of the COLDQ algorithm is upper bounded by

$$
\begin{array} { r l } & { \displaystyle \mathrm { R E G } _ { s } ( T ) \leq \sum _ { t = 2 } ^ { T - 1 } \big ( \alpha _ { t } - \alpha _ { t - 1 } - \mu \big ) \| \mathbf { x } ^ { \star } - \mathbf { x } _ { t } \| ^ { 2 } } \\ & { \quad \quad \quad + \displaystyle \frac { D ^ { 2 } } { 4 } \sum _ { t = 1 } ^ { T } \frac { 1 } { \alpha _ { t } } + ( \alpha _ { 1 } - \mu ) R ^ { 2 } + D R . } \end{array}
$$

# 5.5 Regret and Constraint Violation Bounds

From Theorems 1-3, we can derive the following corollaries on the regret and constraint violation bounds of COLDQ.

Corollary 1 (Convex Loss). Under Assumptions $\boldsymbol { { I } }$ -3, for any $V _ { x } \in [ 0 , 1 ]$ and $V _ { g } \in [ 0 , 1 ]$ , let $\alpha _ { t } = t ^ { \frac { 1 - V _ { x } } { 2 } }$ , $\eta = T ^ { - 1 }$ and $\gamma = \epsilon T$ , where $\epsilon \in ( 0 , G )$ , COLDQ achieves:

$$
\mathrm { R E G } _ { \mathrm { d } } ( T ) = \mathcal { O } \big ( T ^ { \frac { 1 + V _ { x } } { 2 } } \big ) , \quad \mathrm { V I O _ { h } } ( T ) = \mathcal { O } \big ( T ^ { V _ { g } } \big ) .
$$

Corollary 2 (Strongly Convex Loss). Under Assumptions $\jmath$ - 4, for any $V _ { g } \in [ 0 , 1 ]$ , let $\alpha _ { t } = \mu t$ , $\eta = T ^ { - 1 }$ , and $\gamma = \epsilon T$ , where $\epsilon \in ( 0 , G )$ , COLDQ achieves:

$$
\mathrm { R E G } _ { \mathrm { s } } ( T ) = \mathcal { O } \big ( \log T \big ) , \quad \mathrm { V I O _ { h } } ( T ) = \mathcal { O } \big ( T ^ { V _ { g } } \big ) .
$$

From Corollary 1, we readily have a static regret bound $\mathrm { R E G } _ { \mathrm { s } } ( T ) = \mathcal { O } \left( T ^ { \frac { 1 } { 2 } } \right)$ by setting $V _ { x } ~ = ~ 0$ , and a hard constraint violation bound $\mathrm { V I O } _ { \mathrm { h } } ( T ) ~ = ~ \mathcal { O } ( 1 )$ for fixed constraints by setting $V _ { g } = 0$ . From Corollary 2, we also have $\mathrm { V I O } _ { \mathrm { h } } ( T ) \dot { = } \mathcal { O } ( 1 \bar { ) }$ for fixed constraints.

Remark. The same $\mathcal { O } ( T ^ { \frac { 1 + V _ { x } } { 2 } } )$ dynamic regret and $\mathcal { O } ( T ^ { V _ { g } } )$ hard constraint violation in Corollary 1 can be achieved without the knowledge of $V _ { x }$ to set the algorithm parameter $\alpha _ { t }$ . In the full version (Wang, Yan, and Liu 2025), we extend the basic COLDQ algorithm with expert tracking, which can achieve the same performance bounds as COLDQ without any prior knowledge of the system dynamics.

# 6 Experiments

We conduct experiments to evaluate the performance of COLDQ for both time-varying and fixed constraints. In the full version (Wang, Yan, and Liu 2025), we provide all the algorithm parameters used in our experiments, and detailed problem settings of the application to online job scheduling.

![](images/7a3cdbd8f8a705bbf593e82dbb62e6b2b51d295d3d41119af7086b3fdb640aa6.jpg)  
Figure 1: Experiment on time-varying constraints.

×10 5101520Hard Constraint Violation   
0.51.01.52.0Accumulated Loss COLDQ RECOO COLDQ RECOO Yi et al. 2021 Yi et al. 2021 ×104 2.2 2.1 0 4500 5000   
0.0 4500 5000 0   
0 1000 2000 3000 4000 5000 0 1000 2000 3000 4000 5000 Time Time

# 6.1 Experiment on Time-Varying Constraints

Similar to the problem considered in (Guo et al. 2022; Yi et al. 2023), we set the loss function as $\begin{array} { r } { f _ { t } ( { \bf x } ) = \frac { 1 } { 2 } | | { \bf H } _ { t } { \bf x } - { \bf \nabla } } \end{array}$ $\mathbf { y } _ { t } | | ^ { 2 }$ , where $\mathbf { H } _ { t } ~ \in ~ \mathbb { R } ^ { 4 \times 1 0 }$ , $\textbf { x } \in \ \mathbb { R } ^ { 1 0 }$ , and $\mathbf { y } _ { t } ~ \in ~ \mathbb { R } ^ { 4 }$ . Each element of $\mathbf { H } _ { t }$ is uniformly distributed, i.e., $H _ { t } ^ { i , j } \sim$ $U ( - 1 , 1 ) , \forall i , j$ . Each element of $\mathbf { y } _ { t }$ is generated as $\dot { y } _ { t } ^ { i } =$ $\textstyle \sum _ { j = 1 } ^ { 1 0 } H _ { t } ^ { i , j } + \epsilon _ { i }$ , where $\boldsymbol { \epsilon } _ { i }$ follows a standard normal distribution. We set the constraint function as $\mathbf { g } _ { t } ( \mathbf { x } ) = \mathbf { A } _ { t } \mathbf { x } - \mathbf { b } _ { t } ,$ , where $\mathbf { A } _ { t } \in \mathbb { R } ^ { 2 \times 1 0 }$ and ${ \bf b } _ { t } \in \mathbb { R } ^ { 2 }$ , and $\mathcal { X } = \{ \mathbf { x } \mid \mathbf { 0 } \preceq \mathbf { x } \preceq $ $\mathbf { 5 } \}$ . We generate $A _ { t } ^ { i , j } \sim U ( 0 , 1 ) , \forall i , j$ and $b _ { t } ^ { i } \sim U ( 0 , 1 ) , \forall i$ .

We compare COLDQ with the state-of-the-art timevarying constrained OCO algorithms: RECOO (Guo et al. 2022) and Algorithm 1 (Yi et al. 2023). Fig 1 shows the accumulated loss and hard constraint violation. We can see that COLDQ achieves over $4 0 \%$ lower constraint violation than RECOO without sacrificing the accumulated loss.

# 6.2 Experiment on Fixed Constraints

We consider an online quadratic programming problem similar to (Yi et al. 2021). We set the loss function as $f _ { t } ( \mathbf { x } ) =$ $| | \mathbf { x } - \pmb { \theta } _ { t } | | ^ { 2 } + 2 0 \langle \pmb { \theta } _ { t } , \mathbf { x } \rangle$ , where $\pmb { \theta } _ { t } \ = \ \pmb { \theta } _ { t } ^ { 1 } + \pmb { \theta } _ { t } ^ { 2 } + \pmb { \theta } _ { t } ^ { 3 } \ \in$ ${ \dot { \mathbb { R } } } ^ { 2 }$ and $\textbf { x } \in \ \mathbb { R } ^ { 2 }$ . The time-varying parameters ${ \mathbf { } } \theta _ { t }$ are set as $\theta _ { t } ^ { 1 , j } \ \sim \ U ( - t ^ { 1 / 1 0 } , t ^ { 1 / 1 0 } ) , \forall j$ ; $\theta _ { t } ^ { 2 , j } \sim U ( - 1 , 0 ) , \forall j$ for $t ~ \in ~ [ 1 , 1 5 0 0 ] \cup [ 2 0 0 0 , 3 5 0 0 ] \cup [ 4 0 0 0 , 5 0 0 0 ]$ , and $\theta _ { t } ^ { 2 , j } \sim$ $U ( 0 , 1 ) , \forall j$ otherwise; and $\theta _ { t } ^ { 3 , j } ~ = ~ ( - 1 ) ^ { \mu _ { t } } , \forall j$ with the sequence of $\mu _ { t }$ being a random permutation of the vector $\left[ 1 : 5 0 0 0 \right]$ . We set the constraint function as ${ \bf g } ( { \bf x } ) { \bf \theta } = { \bf \theta }$ $\mathbf { A } \mathbf { x } - \mathbf { b }$ , where $\textbf { A } \in \ \mathbb { R } ^ { 3 \times 2 }$ and $\textbf { b } \in \ \mathbb { R } ^ { 3 }$ with $A _ { i , j } \sim$ $U ( 0 . 1 , 0 . 5 ) , \forall i , j$ , and $b _ { i } \sim U ( 0 , 0 . 3 ) , \forall i$ , and the feasible set as ${ \mathcal { X } } = \{ \mathbf { x } \mid \mathbf { 0 } \preceq \mathbf { x } \preceq \mathbf { 1 } \}$ . We also experiment on the online linear programming problem considered in (Yi et al. 2021) and (Guo et al. 2022), by setting $f _ { t } ( \mathbf { x } ) = \langle \pmb { \theta } _ { t } , \mathbf { x } \rangle$ and keeping the rest of the problem settings unchanged.

![](images/798ddcf101d30ee9de872144ff5e6d0586393e330ce49029a61008132815974c.jpg)  
Figure 3: Experiment on online linear programming.

![](images/f76c42ec45414b34fdb4ba8aed11c44237be3d10b3e01311bc657562e37d3e81.jpg)  
Figure 2: Experiment on online quadratic programming.   
Figure 4: Experiment on online job scheduling.

We compare COLDQ with the current-best time-invariant constrained OCO algorithms: Algorithm 1 (Yi et al. 2021) and RECOO (Guo et al. 2022). As shown in Figures 2 and 3, COLDQ demonstrates significant reductions in the hard constraint violation compared to RECOO (Guo et al. 2022).

# 6.3 Application to Online Job Scheduling

We further apply COLDQ to online job scheduling using real-world datasets similar to (Yu, Neely, and Wei 2017; Guo et al. 2022). Figure 4 shows the time-averaged energy cost and the number of delayed jobs. COLDQ demonstrates a significant reduction in the energy cost without compromising the service quality, compared with RECOO (Guo et al. 2022) and Algorithm 1 (Yi et al. 2023).

# 7 Conclusions

We propose an effective COLDQ algorithm for OCO with time-varying constraints. We design a novel virtual queue that is bounded both from above and below to strictly control the hard constraint violation. Through a new Lyapunov drift analysis, COLDQ achieves $\mathcal { O } ( T ^ { \frac { 1 + \mathbf { \bar { V } } _ { x } } { 2 } } )$ dynamic regret and $\mathcal { O } ( \hat { T } ^ { V _ { g } } )$ hard constraint violation. For the first time, the two bounds smoothly approach to the best-known $\mathcal { O } ( T ^ { \frac { 1 } { 2 } } )$ regret and $\mathcal { O } ( 1 )$ violation, as the dynamics of the losses and constraints represented by $V _ { x }$ and $V _ { g }$ diminish. We further study the case of strongly-convex loss functions, and demonstrate that COLDQ matches the best-known ${ \mathcal { O } } ( \log T )$ static regret while maintaining the $\mathcal { O } ( T ^ { V _ { g } } )$ hard constraint violation. Moreover, we extend COLDQ with expert tracking capability, which allows it to achieve the same dynamic regret and hard constraint violation bounds without any prior knowledge of the system dynamics. Finally, experimental results complement our theoretical analysis.

# Acknowledgments

This work was supported in part by the Hong Kong Research Grants Council (RGC) Early Career Scheme (ECS) under grant 22200324.