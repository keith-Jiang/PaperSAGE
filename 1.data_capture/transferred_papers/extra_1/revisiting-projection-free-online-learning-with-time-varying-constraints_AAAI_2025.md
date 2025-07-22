# Revisiting Projection-Free Online Learning with Time-Varying Constraints

Yibo Wang1,2, Yuanyu $\mathbf { W a n } ^ { 3 , 1 , * }$ , Lijun Zhang1,2,\*

1 National Key Laboratory for Novel Software Technology, Nanjing University, Nanjing 210023, China 2 School of Artificial Intelligence, Nanjing University, Nanjing 210023, China 3 School of Software Technology, Zhejiang University, Ningbo 315048, China {wangyb, zhanglj}@lamda.nju.edu.cn, wanyy $@$ zju.edu.cn

# Abstract

We investigate constrained online convex optimization, in which decisions must belong to a fixed and typically complicated domain, and are required to approximately satisfy additional time-varying constraints over the long term. In this setting, the commonly used projection operations are often computationally expensive or even intractable. To avoid the timeconsuming operation, several projection-free methods have been proposed with an $\mathcal { O } ( T ^ { 3 / 4 } \sqrt { \log T } )$ regret bound and an $\mathcal { O } ( T ^ { 7 / 8 } )$ cumulative constraint violation (CCV) bound for general convex losses. In this paper, we improve this result and further establish novel regret and CCV bounds when loss functions are strongly convex. The primary idea is to first construct a composite surrogate loss, involving the original loss and constraint functions, by utilizing the Lyapunov-based technique. Then, we propose a parameter-free variant of the classical projection-free method, namely online Frank-Wolfe (OFW), and run this new extension over the online-generated surrogate loss. Theoretically, for general convex losses, we achieve an $\mathcal { O } ( T ^ { 3 / 4 } )$ regret bound and an $\mathcal { O } ( T ^ { 3 / 4 } \log T )$ CCV bound, both of which are order-wise tighter than existing results. For strongly convex losses, we establish new guarantees of an $\mathcal { O } ( T ^ { 2 / 3 } )$ regret bound and an $\mathcal { O } ( T ^ { 5 / 6 } )$ CCV bound. Moreover, we also extend our methods to a more challenging setting with bandit feedback, obtaining similar theoretical findings. Empirically, experiments on real-world datasets have demonstrated the effectiveness of our methods.

defined as the difference between the cumulative loss of the learner and that of the best fixed decision.

In the literature, there have been abundant theoretical appeals for OCO, such as the $\mathcal { O } ( \sqrt { T } )$ regret bound for general convex losses (Zinkevich 2003) and the $\mathcal { O } ( \log T )$ regret bound for strongly convex losses (Hazan, Agarwal, and Kale 2007). In practice, besides the hard and fixed domain $\kappa$ , the decisions made by the learner are typically governed by a series of soft and time-varying constraints, which may be violated in several rounds but should be satisfied on average over the long term. For example, in wireless communication systems, operators manage varying transmission power consumption to ensure the reception of messages (Mannor, Tsitsiklis, and $\mathrm { Y u } ~ 2 0 0 9 )$ ; in online advertisement systems, advertisers employ dynamic budgets to maximize the clickthrough-rates for their advertisements (Liakopoulos et al. 2019). These practical applications thus motivate the development of constrained online convex optimization (COCO) (Mahdavi, Jin, and Yang 2012; Neely and $\mathrm { Y u } 2 0 1 7 \mathrm { , }$ ).

In the framework of COCO, the time-varying constraints are typically captured by the inequality $g _ { t } ( \mathbf { x } ) \ \leq \ 0$ where $g _ { t } ( \cdot ) : \mathcal { K } \mapsto \mathbb { R }$ is a convex function revealed by the adversary at the end of each round $t$ . Consequently, in addition to minimizing (1), the learner also aims to ensure the cumulative constraint violation (CCV):

# Introduction

Online convex optimization (OCO) has become a popular paradigm for modeling online decision-making problems (Shalev-Shwartz 2012; Hazan 2016; Orabona 2019), such as online portfolio optimization (Agarwal et al. 2006) and online advertisement system (McMahan et al. 2013). Formally, OCO can be viewed as a structured iterative game between a learner and an adversary. Specifically, at each round $t$ , the learner first chooses a decision $\mathbf { x } _ { t }$ from a convex and fixed domain $\mathcal { K } \subseteq \mathbb { R } ^ { d }$ . Then, the adversary reveals a convex loss function $f _ { t } ( \cdot ) : \mathcal { K } \mapsto \mathbb { R }$ , and the learner suffers the cost $f _ { t } ( \mathbf { x } _ { t } )$ . The goal of the learner is to minimize the regret:

$$
Q _ { T } = { \sum ^ { T } } _ { t = 1 } ^ { T } g _ { t } ^ { + } ( { \mathbf x } _ { t } )
$$

$$
{ \mathrm { R e g r e t } } _ { T } = \sum _ { t = 1 } ^ { T } f _ { t } ( \mathbf { x } _ { t } ) - \underset { \mathbf { x } \in \mathcal { K } } { \operatorname* { m i n } } \sum _ { t = 1 } ^ { T } f _ { t } ( \mathbf { x } ) ,
$$

to be sublinear with respect to the time horizon $T$ , where $g _ { t } ^ { + } ( \mathbf { x } ) \triangleq \operatorname* { m a x } \{ 0 , g _ { t } ( \mathbf { x } ) \}$ . To optimize (1) and (2) concurrently, various efforts have been made recently (Cao and Liu 2019; $\mathrm { Y u }$ and Neely 2020; Yi et al. 2021; Guo et al. 2022; Yi et al. 2023; Sinha and Vaze 2024), and established plentiful guarantees, including the regret and CCV bounds of $\overset { \cdot } { \mathcal { O } } ( \sqrt { T } )$ for general convex losses (Yu, Neely, and Wei 2017).

The key operation in these COCO methods is the projection that pulls an infeasible decision back into the hard constraint $\kappa$ . In many practical scenarios, the domain $\kappa$ is typically high-dimensional and complex, rendering projections onto $\kappa$ computationally expensive or even intractable, which significantly limits the applicability of these methods. To address this issue, several studies (Lee, Ho-Nguyen, and Lee 2023; Garber and Kretzu 2024) propose projectionfree methods for COCO, which replace the time-consuming projection with the more efficient linear optimization operation. One prominent example is online semidefinite optimization, where the hard constraint is a positive semidefinite cone with the bounded trace. In this case, the linear optimization has been proven at least an order of magnitude faster than projection (Hazan and Kale 2012). Unfortunately, existing projection-free methods can only guarantee the $\mathcal { O } ( T ^ { 3 / 4 } \sqrt { \log T } )$ regret bound and the $\mathcal { O } ( T ^ { 7 / 8 } )$ CCV bound for general convex losses (Garber and Kretzu 2024).

In this paper, we improve the above bounds and introduce new theoretical guarantees for strongly convex losses. The key idea is to first construct a composite surrogate loss that consists of the original loss $f _ { t } ( \bar { \cdot } )$ and the timevarying constraint $g _ { t } ( \cdot )$ , based on a carefully designed Lyapunov function. Rigorous analysis reveals that both (1) and (2) are simultaneously controlled by the regret in terms of the surrogate losses, so that we can directly apply classical projection-free methods, e.g., online Frank-Wolfe (OFW) (Hazan and Kale 2012), over the surrogate losses to minimize the two metrics. Notably, since the surrogate loss is generated in an online manner, essential prior knowledge for OFW (e.g., the gradient norm bound) is unavailable beforehand. Therefore, we need to employ the methods that are agnostic to the prior parameters about the surrogate loss. To this end, we propose the first parameter-free variant of OFW for general convex losses based on the doubling trick technique (Cesa-Bianchi et al. 1997). By running the parameterfree variant over the composite surrogate losses, we establish an $\mathcal { O } ( T ^ { 3 / 4 } )$ regret bound and an $\mathcal { O } ( T ^ { 3 / 4 } \log T )$ CCV bound for the general convex loss. Both of our results are better than the state-of-the-art bounds achieved by Garber and Kretzu (2024). Additionally, we further investigate the strongly convex loss and achieve an $\mathcal { O } ( T ^ { 2 / 3 } )$ regret bound and an $\mathcal { O } ( T ^ { 5 / 6 } )$ CCV bound, by constructing the surrogate loss based on a different Lyapunov function and running the strongly convex variant of OFW (Wan and Zhang 2021).

Furthermore, to handle the more challenging bandit setting, we combine our proposed methods with the classical one-point estimator (Flaxman, Kalai, and McMahan 2005), which can approximate the gradient with only the loss value. Theoretically, for general convex losses, we establish the $\mathcal { O } ( T ^ { 3 / 4 } )$ regret bound and the $\mathcal { O } ( T ^ { 3 / 4 } \log T )$ CCV bound. For strongly convex losses, we achieve the $\mathcal { O } ( T ^ { 2 / 3 } \log T )$ regret bound and the $\mathcal { O } ( T ^ { 5 / 6 } \log T )$ CCV bound.

Contributions. We summarize our contributions below.

• For general convex losses, we deliver an $\mathcal { O } ( T ^ { 3 / 4 } )$ regret bound and an $\mathcal { O } ( T ^ { 3 / 4 } \log T )$ CCV bound, both of which improve the previous results of Garber and Kretzu (2024). During the analysis, we propose the first parameter-free variant of OFW, which may be an independent of interest;   
• For strongly convex losses, we establish the novel results of an $\mathcal { O } ( T ^ { 2 / 3 } )$ regret bound and an $\mathcal { O } ( T ^ { 5 / 6 } )$ CCV bound for projection-free COCO;   
• We extend our methods to the bandit setting and achieve similar bounds as those in the full-information setting;

• We verify our theoretical findings by conducting experiments on real-world datasets. The empirical results have demonstrated the effectiveness of our methods.

# Related Work

In this section, we briefly overview the recent progress on projection-free and constrained online convex optimization.

# Projection-Free Online Convex Optimization

The pioneering work of Hazan and Kale (2012) introduces the first projection-free online method, namely online FrankWolfe (OFW), which is an online extension of the classical Frank-Wolfe algorithm (Frank and Wolfe 1956). The basic idea is to replace the time-consuming projection with the following linear optimization steps:

$$
\begin{array} { r } { \mathbf { v } _ { t } = \underset { \mathbf { x } \in \mathcal { K } } { \mathrm { a r g m i n } } \left. \nabla F _ { t } ( \mathbf { x } _ { t } ) , \mathbf { x } \right. , \mathbf { x } _ { t + 1 } = \mathbf { x } _ { t } + \sigma _ { t } ( \mathbf { v } _ { t } - \mathbf { x } _ { t } ) } \end{array}
$$

where $\sigma _ { t } > 0$ denotes the step size, and $F _ { t } ( \mathbf { x } )$ is defined as

$$
F _ { t } ( \mathbf { x } ) = \eta \sum _ { \tau = 1 } ^ { t - 1 } \nabla f _ { \tau } ( \mathbf { x } _ { \tau } ) ^ { \top } \mathbf { x } + \| \mathbf { x } - \mathbf { x } _ { 1 } \| _ { 2 } ^ { 2 }
$$

parameterized by $\eta > 0$ . With the prior knowledge about $f _ { t } ( \cdot )$ (e.g., the gradient norm bound) and appropriate configurations on $\eta$ and $\sigma _ { t }$ , OFW ensures an $\mathcal { O } ( T ^ { 3 / 4 } )$ regret bound for general convex losses.

Based on OFW, plenty of investigations deliver tighter regret bounds by utilizing additional properties on $f _ { t } ( \cdot )$ , such as the smoothness (Hazan and Minasyan 2020), the strong convexity (Wan and Zhang 2021; Kretzu and Garber 2021), and the exponential concavity (Garber and Kretzu 2023; Mhammedi 2024). Moreover, several efforts improve the regret bounds by leveraging special structures of $\kappa$ (Garber and Hazan 2016; Levy and Krause 2019; Molinaro 2020; Wan and Zhang 2021; Mhammedi 2022; Gatmiry and Mhammedi 2023). Additionally, there exist other studies exploring more practical scenarios, e.g., the bandit feedback (Chen, Zhang, and Karbasi 2019; Garber and Kretzu 2020; Zhang et al. 2024), the delayed feedback (Wan et al. 2022b), the distributed setting (Zhang et al. 2017; Wan, Tu, and Zhang 2020; Wan et al. 2022a; Wang et al. 2023; Wan et al. 2024) and non-stationary environments (Kalhan et al. 2021; Wan, Xue, and Zhang 2021; Garber and Kretzu 2022; Lu et al. 2023; Wan, Zhang, and Song 2023; Wang et al. 2024).

# Constrained Online Convex Optimization

In the literature, there are two lines of research for COCO. One is the time-invariant setting, where the soft constraints are assumed to be fixed, i.e., $g _ { t } ( \cdot ) = g ( \cdot )$ , and known to the learner at the beginning round. In this setting, for general convex losses, Mahdavi, Jin, and Yang (2012) originally develop an $\mathcal { O } ( \sqrt { T } )$ regret bound and an $\mathcal { O } ( T ^ { 3 / 4 } )$ CCV bound. Then, subsequent studies generalize the results, and obtain tighter bounds for both regret and CCV under additional conditions (Jenatton, Huang, and Archambeau 2016; Yuan and Lamperski 2018; Yu and Neely 2020; Yi et al. 2021).

The other is the time-variant setting, where the soft constraints change over time and are only revealed after the learner submits the decision. Under the stochastic timevarying constraints and the Slater’s condition, Yu, Neely, and Wei (2017) deliver an $\mathcal { O } ( \sqrt { T } )$ regret bound and an $\mathcal { O } ( \sqrt { T } )$ CCV bound. Subsequently, extensive studies focus on the more general adversarial time-varying constraints and attempt to remove the Slater’s condition (Neely and $\mathrm { Y u }$ 2017; Sun, Dey, and Kapoor 2017; Liakopoulos et al. 2019; Cao and Liu 2019; Guo et al. 2022; Yi et al. 2023; Sinha and Vaze 2024). One of the key techniques in these work is to analyze a refined bound based on the Lyapunov drift of a virtual queue, which partially inspires our methods. To the best of our knowledge, the state-of-the-art results in this setting are delivered by Sinha and Vaze (2024), who establish the $\mathcal { O } ( \sqrt { T } )$ regret bound and the $\mathcal { O } ( \sqrt { T } \log T )$ CCV bound for general convex losses, and the ${ \mathcal { O } } ( \log T )$ regret bound and the $\mathcal { O } ( \sqrt { T \log T } )$ CCV bound for strongly convex losses.

<html><body><table><tr><td>Methods</td><td>Losses</td><td>Constraints</td><td>Feedback</td><td>Regret</td><td>CCV</td></tr><tr><td rowspan="2">Lee,Ho-Nguyen,and Lee (2023)</td><td>cVX</td><td>sto</td><td>full-info</td><td>O(T5/6)</td><td>O(T5/6)</td></tr><tr><td>cvX</td><td>adv</td><td>full-info</td><td>O(T5/6+α)</td><td>O(T1i/12-α/2)</td></tr><tr><td>Garber and Kretzu (2024)</td><td>cvX</td><td>adv</td><td>full-info</td><td>O(T3/4√IogT)</td><td>O(T7/8)</td></tr><tr><td>Theorem 1 (this work)</td><td>cvX</td><td>adv</td><td>full-info</td><td>O(T3/4)</td><td>O(T3/4 log T)</td></tr><tr><td>Theorem 2 (this work)</td><td>str-cvx</td><td>adv</td><td>full-info</td><td>O(T2/3)</td><td>O(T5/6)</td></tr><tr><td>Garber and Kretzu (2024)</td><td>cvx</td><td>adv</td><td>bandits</td><td>O(T3/4√log T)</td><td>O(T7/8 log T)</td></tr><tr><td>Theorem 3 (this work)</td><td>cvX</td><td>adv</td><td>bandits</td><td>O(T3/4)</td><td>O(T3/4 log T)</td></tr><tr><td>Theorem 4 (this work)</td><td>str-cvx</td><td>adv</td><td>bandits</td><td>O(T2/3 1og T)</td><td>O(T5/6 log T)</td></tr></table></body></html>

Table 1: Comparisons of our results with existing projection-free methods for COCO. Abbreviations: convex $\to \mathbf { c v x }$ , strongly convex $$ str-cvx, stochastic $$ sto, adversarial $$ adv, full-information $$ full-info.

As mentioned before, the above methods still rely on the inefficient projection for decision updates, which thereby motivates the development of projection-free COCO. Lee, Ho-Nguyen, and Lee (2023) first obtain an $\mathcal { O } ( T ^ { 5 / 6 + \alpha } )$ regret bound and an $\mathcal { O } ( T ^ { 1 1 / 1 2 - \alpha / 2 } )$ CCV bound with the parameter $\alpha ~ \in ~ ( 0 , 1 )$ for general convex losses and the full-information feedback. Later, Garber and Kretzu (2024) propose to apply a recent projection-free method, named LOO-BOGD (Garber and Kretzu 2022), under the driftplus-penalty framework (Neely 2010) that is extensively used in previous COCO methods (Yu, Neely, and Wei 2017; Guo et al. 2022), and thus deliver an $\mathcal { O } ( T ^ { 3 / 4 } \sqrt { \log T } )$ regret bound and an $\mathcal { O } ( T ^ { 7 / 8 } )$ CCV bound. When only the bandit feedback (i.e., the function value) is accessible, they obtain the same regret bound and a slightly worse $\mathcal { O } ( T ^ { 7 / 8 } \log T )$ CCV bound. More details can be found in Table 1.

# Preliminaries

In this section, we recall the basic assumptions and definitions that are commonly used in prior studies (Mahdavi, Jin, and Yang 2012; Hazan and Kale 2012; Agrawal and Devanur 2014).

Assumption 1. The convex decision set $\kappa$ contains the ball of radius r centered at the origin 0, and is contained in an ball with the diameter $D = 2 R _ { ☉ }$ , i.e., $r B \subseteq \mathcal { X } \subseteq R B$ where

$$
B = \{ \mathbf { x } \in \mathbb { R } ^ { d } \mid \| \mathbf { x } \| _ { 2 } \leq 1 \} .
$$

Assumption 2. At each round $t$ , the loss function $f _ { t } ( \cdot )$ and the constraint function $g _ { t } ( \cdot )$ are $G$ -Lipschitz over $\kappa$ , i $\begin{array} { r } { : e . , \forall \mathbf { x } , \mathbf { y } \in \mathcal { K } , | f _ { t } ( \mathbf { x } ) - f _ { t } ( \mathbf { y } ) | \leq G \| \mathbf { x } - \mathbf { y } \| _ { 2 } } \end{array}$ and $| g _ { t } ( { \bf x } ) -$ $g _ { t } ( \mathbf { y } ) | \leq G \| \mathbf { x } - \mathbf { y } \| _ { 2 }$ .

Assumption 3. At each round $t ,$ , the loss function value $f _ { t } ( \mathbf { x } )$ is bounded over $\kappa$ , i.e., $\forall \mathbf { x } \in \mathcal { K }$ , $| f _ { t } ( \mathbf { x } ) | \leq M$ .

Definition 1. Let $\Phi ( \boldsymbol { x } ) : \mathbb { R } ^ { + } \mapsto \mathbb { R }$ be a convex function. It is called Lyapunov if $\Phi ( x )$ satisfies (i) $\Phi ( 0 ) = 0$ ; (ii) $\Phi ( x ) \geq$ $0 , \forall x \in \operatorname { \bar { \mathbb { R } } } ^ { \bar { + } }$ ; (iii) $\Phi ( x )$ is non-decreasing.

Definition 2. Let $f ( \mathbf { x } ) : \mathcal { K } \mapsto \mathbb { R }$ be a function over $\kappa$ . It is called $\alpha _ { f }$ -strongly convex if for all $\mathbf { x } , \mathbf { y } \in \kappa$

$$
f ( \mathbf { y } ) \geq f ( \mathbf { x } ) + \langle \nabla f ( \mathbf { x } ) , \mathbf { y } - \mathbf { x } \rangle + \frac { \alpha _ { f } } { 2 } \| \mathbf { y } - \mathbf { x } \| _ { 2 } ^ { 2 } .
$$

In analysis, we will make use of the following property of strongly convex functions (Garber and Hazan 2015).

Lemma 1. Let $f ( \mathbf { x } )$ be an $\alpha _ { f }$ -strongly convex function over $\kappa$ and $\begin{array} { r } { \mathbf { x } ^ { * } = \mathop { \mathrm { a r g m i n } } _ { \mathbf { x } \in \mathcal { K } } f ( \bar { \mathbf { x } } ) } \end{array}$ . Then, for any $\mathbf { x } \in \mathcal { K }$

$$
\frac { \alpha _ { f } } { 2 } \left\| \mathbf { x } - \mathbf { x } ^ { * } \right\| _ { 2 } ^ { 2 } \leq f ( \mathbf { x } ) - f \left( \mathbf { x } ^ { * } \right) .
$$

# Main Results

In this section, we initially present our methods as well as their theoretical guarantees for the full-information setting. Then, we extend our investigations to the bandit setting. Due to the limitation of space, all proofs are deferred in the supplementary material.

# Algorithms for Full-Information Setting

Overall, we first construct a composite surrogate loss function based on the loss $f _ { t } ( \mathbf { x } )$ , the constraint $g _ { t } ( \mathbf { x } )$ and a specially designed Lyapunov function that depends on the type of $f _ { t } ( \mathbf { x } )$ . Then, we employ parameter-free variants of OFW to optimize the surrogate loss.

Specifically, let $Q _ { t }$ be the cumulative constraint violation at the round $t$ , and $\Phi ( \cdot )$ be a convex Lyapunov function. According to (2), $Q _ { t }$ can be formalized recursively as

$$
Q _ { t } = Q _ { t - 1 } + g _ { t } ^ { + } ( \mathbf { x } _ { t } ) , \forall t \geq 1
$$

1: Choose any $\mathbf { x } _ { 1 } \in \mathcal { K }$ , and set $\tilde { G } _ { 1 } = s _ { 1 } = k = 1$   
2: for $t = 1$ to $T$ do   
3: Play $\mathbf { x } _ { t }$ , and suffer $f _ { t } ( \mathbf { x } _ { t } )$ and $g _ { t } ( \mathbf { x } _ { t } )$   
4: Construct $Q _ { t }$ and $\tilde { f } _ { t } ( { \bf x } )$ according to (4) and (6)   
5: while $\tilde { G } _ { k } < \beta G ( \gamma + \Phi ^ { \prime } ( \beta Q _ { t } ) )$ ) do   
6: Se $\dot { \boldsymbol { G } } _ { k + 1 } = 2 \tilde { \boldsymbol { G } } _ { k } , \boldsymbol { s } _ { k + 1 } = t , \boldsymbol { k } = \boldsymbol { k } + 1$   
7: end while   
8: Set $\eta _ { k }$ and $F _ { s _ { k } : t } ( \mathbf { x } )$ according to (9) and (10)   
9: Compute $\mathbf { v } _ { t }$ and $\sigma _ { s _ { k } , t }$ according to (11) and (12)   
10: Update $\mathbf { x } _ { t + 1 }$ according to (13)

and $Q _ { 0 } = 0$ . By utilizing the convexity of $\Phi ( \cdot )$ , the Lyapunov drift of $Q _ { t }$ at the round $t$ , i.e., $\Phi ( \beta Q _ { t } ) - \Phi ( \beta Q _ { t - 1 } )$ , is upper bounded by

$$
\begin{array} { r l } & { \Phi ( \beta Q _ { t } ) - \Phi ( \beta Q _ { t - 1 } ) \le \Phi ^ { \prime } ( \beta Q _ { t } ) \beta [ Q _ { t } - Q _ { t - 1 } ] } \\ & { \qquad \quad \stackrel { ( 4 ) } { = } \Phi ^ { \prime } ( \beta Q _ { t } ) \beta g _ { t } ^ { + } ( { \bf x } _ { t } ) } \end{array}
$$

where $\beta > 0$ denotes a hyper-parameter. To simultaneously minimize $f _ { t } ( \mathbf { x } )$ and $g _ { t } ( \mathbf { x } )$ , we follow the drift-plus-penalty framework (Neely 2010), and construct the surrogate loss function $\tilde { f } _ { t } ( { \bf x } )$ by combining the loss $f _ { t } ( \mathbf { x } )$ and the upper bound of the Lyapunov drift in (5):

$$
\widetilde { f } _ { t } ( \mathbf { x } ) = \gamma \beta f _ { t } ( \mathbf { x } ) + \Phi ^ { \prime } ( \beta Q _ { t } ) \beta g _ { t } ^ { + } ( \mathbf { x } )
$$

where $\gamma ~ > ~ 0$ denotes a hyper-parameter. In fact, it can be verified that the regret in terms of $\tilde { f } _ { t } ( { \bf x } )$ , denoted by $\mathrm { R e g r e t } _ { T } ^ { \prime }$ , concurrently captures (1) and (2) (Sinha and Vaze 2024):

$$
\mathrm { R e g r e t \prime } _ { T } ^ { ( 5 ) , ( 6 ) } \geq \gamma \beta \mathrm { R e g r e t } _ { T } + \Phi ( \beta Q _ { T } ) .
$$

With an appropriate configuration on $\Phi ( \cdot )$ , (1) and (2) can be decoupled from (7), delivering corresponding theoretical guarantees. It should be noticed that the specific choice of $\bar { \Phi } ( \cdot )$ is quite involved, since (i) it is employed to construct the surrogate loss in (6), necessitating a simple form that does not incur expensive computational costs; (ii) it appears in (7) and is required to adeptly balance the regret and CCV.

In the following, we investigate the general convex losses and the strongly convex losses.

General Convex Losses. Given the favorable property of minimizing $\tilde { f } _ { t } ( { \bf x } )$ shown in (7), one may attempt to apply the classical OFW method over $\tilde { f } _ { t } ( { \bf x } )$ for the simultaneous minimization on (1) and (2). However, such a straightforward application is not suitable, since $\tilde { f } _ { t } ( { \bf x } )$ is generated in an online manner, and thus the prior knowledge required by OFW is unavailable beforehand. For example, the $\ell _ { 2 }$ -norm of the subgradient $\nabla \tilde { f } _ { t } ( { \bf x } )$ is bounded by:

$$
\begin{array} { r l } & { \| \nabla \tilde { f } _ { t } ( \mathbf { x } ) \| _ { 2 } \leq \gamma \beta \| \nabla f _ { t } ( \mathbf { x } ) \| _ { 2 } + \Phi ^ { \prime } ( \beta Q _ { t } ) \beta \| \nabla g _ { t } ( \mathbf { x } ) \| _ { 2 } } \\ & { \qquad \leq \beta G ( \gamma + \Phi ^ { \prime } ( \beta Q _ { T } ) ) \triangleq \tilde { G } , } \end{array}
$$

in which the last step follows the fact that $\Phi ( \cdot )$ is convex and hence its derivative $\Phi ^ { \prime } ( \cdot )$ is non-decreasing. From (8), it can be observed that $\tilde { G }$ is unknown due to the uncertainty of $Q _ { T }$ at the round $t$ . For this reason, we propose the first parameter-free variant of OFW, which is agnostic to $\tilde { G }$ , and thereby can be employed to minimize $\tilde { f } _ { t } ( \mathbf { x } )$ . The basic idea is to utilize an estimation of $\tilde { G }$ for decision updating. If the estimation is too low, we repeatedly double the current guess and employ the first valid value for updates. We summarize our method in Algorithm 1.

Specifically, at the Step 1, we choose any point $\mathbf { x } _ { 1 } \in \mathcal { K }$ as the decision for the first round and make the estimation $\tilde { G } _ { 1 } = 1$ . Then, at each round $t$ , we submit the decision $\mathbf { x } _ { t }$ , suffer the cost $f _ { t } ( \mathbf { x } _ { t } )$ and the constraint $g _ { t } ( \mathbf { x } _ { t } )$ (Step 3). At the Step 4, we construct $Q _ { t }$ and the surrogate loss function $\tilde { f } _ { t } ( { \bf x } )$ according to (4) and (6), respectively. Next, we verify the feasibility of the estimation $\tilde { G } _ { k }$ . If it is lower than $\ddot { \beta G } ( \gamma + \Phi ^ { \prime } ( \beta Q _ { t } ) )$ , we continuously double the current estimation until an appropriate value is found (Steps 5-7). After that, we set the learning rate

$$
\eta _ { k } = D ( 2 \tilde { G } _ { k } T ^ { 3 / 4 } ) ^ { - 1 } ,
$$

and construct the function

$$
F _ { s _ { k } : t } ( \mathbf { x } ) = \eta _ { k } \sum _ { \tau = s _ { k } } ^ { t } \left. \nabla \tilde { f } _ { \tau } \left( \mathbf { x } _ { \tau } \right) , \mathbf { x } \right. + \left\| \mathbf { x } - \mathbf { x } _ { s _ { k } } \right\| _ { 2 } ^ { 2 }
$$

based on the historical gradients $\nabla \tilde { f } _ { t } ( { \bf x } _ { t } )$ since the round $s _ { k }$ (Step 8), where $s _ { k }$ denotes the first round that utilizes the estimation $\tilde { G } _ { k }$ . At the Step 9, we compute $\mathbf { v } _ { t }$ according to

$$
\begin{array} { r } { \mathbf { v } _ { t } \in \underset { \mathbf { x } \in \mathcal { K } } { \mathrm { a r g m i n } } \left. \nabla F _ { s _ { k } : t } \left( \mathbf { x } _ { t } \right) , \mathbf { x } \right. , } \end{array}
$$

and set the step size as

$$
\sigma _ { s _ { k } , t } = 2 ( t - s _ { k } + 1 ) ^ { - 1 / 2 } .
$$

Finally, we update the decision $\mathbf { x } _ { t + 1 }$ for the next round as shown below (Step 10):

$$
{ \bf x } _ { t + 1 } = { \bf x } _ { t } + \sigma _ { s _ { k } , t } \left( { \bf v } _ { t } - { \bf x } _ { t } \right) .
$$

By choosing the exponential Lyapunov function $\Phi ( x ) \ = \quad$ $\exp ( 2 ^ { - 1 } T ^ { - 3 / 4 } x ) - 1$ , we establish the following theorem.

Theorem 1. Let $\beta ~ = ~ ( 2 ^ { 6 } G D ) ^ { - 1 }$ and $\gamma ~ = ~ 1$ . Under Assumptions $\boldsymbol { l }$ and 2, if the loss functions and the constraint functions are general convex, Algorithm $\jmath$ ensures the bounds of

$$
\mathrm { R e g r e t } _ { T } = \mathcal { O } ( T ^ { 3 / 4 } ) , Q _ { T } = \mathcal { O } ( T ^ { 3 / 4 } \log T ) .
$$

Remark. Compared to the $\mathcal { O } ( T ^ { 3 / 4 } \sqrt { \log T } )$ regret bound and the $\mathcal { O } ( T ^ { 7 / 8 } )$ CCV bound in Garber and Kretzu (2024), our results for both metrics are tighter. The underlying reasons can be attributed to: (i) the choice of projection-free methods. Under the drift-plus-penalty framework, Garber and Kretzu (2024) choose to run the projection-free LOOBOGD method, which, due to its complex design, necessitates additional effort to balance the costs of linear optimization and the performance. In contrast, our proposed method is inherently simpler, naturally requiring only one linear optimization per round; (ii) the specification of $\Phi ( x )$ . Garber and Kretzu (2024) implicitly choose $\Phi ( x ) = x$ , which potentially fails to balance regret and CCV for general convex losses, leading to looser results. Furthermore, it should be emphasized that even if $\Phi ( x )$ in Garber and Kretzu (2024) is replaced with the exponential function, the complex management of linear optimization costs in LOO-BOGD still prevents it from yielding the same results as ours.

<html><body><table><tr><td>Algorithm 2: Strongly Convex Variant of OFW with Time- Varying Constraints (SCOFW-TVC)</td></tr><tr><td>Input: Hyper-parameters β,y，and the function Φ(-),and 1: Choose any X1 ∈K the modulus of strong convexity α f</td></tr></table></body></html>

Strongly Convex Losses. In this case, note that for the $\alpha _ { f }$ -strong convex $f _ { t } ( \mathbf { x } )$ , the surrogate loss $\tilde { f } _ { t } ( { \bf x } )$ defined in (6) is $\gamma \beta \alpha _ { f }$ -strongly convex. Therefore, we can employ the strongly convex variant of OFW to minimize $\tilde { f } _ { t } ( { \bf x } )$ . In this paper, we choose the SCOFW method proposed by Wan and Zhang (2021), because of its simplicity and agnosticism to $\tilde { G }$ . The detailed procedures are given in Algorithm 2.

Specifically, we first choose any decision $\mathbf { x } _ { 1 } \in \mathcal { K }$ for initialization (Step 1). Then, at each round $t$ , we make the decision $\mathbf { x } _ { t }$ , suffer the loss $f _ { t } ( \mathbf { x } _ { t } )$ and the constraint $g _ { t } ( \mathbf { x } _ { t } )$ , and construct $Q _ { t }$ and $\tilde { f } _ { t } ( { \bf x } )$ according to (4) and (6) (Steps 3-4). At Steps 5-6, we construct $F _ { t } ^ { s c } ( \mathbf { x } )$ in the following way:

$$
F _ { t } ^ { s c } ( \mathbf { x } ) = \sum _ { \tau = 1 } ^ { t } \left[ \left. \nabla \tilde { f } _ { \tau } \left( \mathbf { x } _ { \tau } \right) , \mathbf { x } \right. + C _ { 1 } \| \mathbf { x } - \mathbf { x } _ { \tau } \| _ { 2 } ^ { 2 } \right] .
$$

where we denote $C _ { 1 } = \gamma \beta \alpha _ { f } / 2$ for brevity, and compute $\mathbf { v } _ { t }$ according to:

$$
\mathbf { v } _ { t } \in \underset { \mathbf { x } \in \mathcal { K } } { \mathrm { a r g m i n } } \left. \nabla F _ { t } ^ { s c } \left( \mathbf { x } _ { t } \right) , \mathbf { x } \right.
$$

and $\sigma _ { t } ^ { s c }$ according to

$$
\sigma _ { t } ^ { s c } = \underset { \sigma \in [ 0 , 1 ] } { \operatorname { a r g m i n } } F _ { t } ^ { s c } \big ( \mathbf { x } _ { t } + \sigma \big ( \mathbf { v } _ { t } - \mathbf { x } _ { t } \big ) \big ) .
$$

At the Step 7, we update the decision $\mathbf { x } _ { t + 1 }$ for the next round according to

$$
{ \bf x } _ { t + 1 } = { \bf x } _ { t } + \sigma _ { s _ { k } , t } \left( { \bf v } _ { t } - { \bf x } _ { t } \right) .
$$

By choosing the quadratic Lyapunov function $\Phi ( x ) = x ^ { 2 } + $ $x$ , we establish the following theoretical results.

Theorem 2. Let $\beta = G ^ { - 1 } D ^ { - 1 } T ^ { - 2 / 3 }$ and $\gamma = G / ( G +$ $\alpha _ { f } D )$ . Under Assumptions $\jmath$ and 2, if the loss functions are $\alpha _ { f }$ -strongly convex, and the constraint functions are general convex, Algorithm 2 ensures the bounds of

$$
\mathrm { R e g r e t } _ { T } = \mathcal { O } ( T ^ { 2 / 3 } ) , Q _ { T } = \mathcal { O } ( T ^ { 5 / 6 } ) .
$$

Remark. Theorem 2 provides the first theoretical guarantees for the strongly convex losses in projection-free COCO, which are tighter than those in Garber and Kretzu (2024) for general convex losses.

# Algorithms for Bandit Setting

In this section, we investigate the bandit setting, where only the function value is available. To handle the more challenging setting, we introduce the one-point gradient estimator (Flaxman, Kalai, and McMahan 2005), which can approximate the gradient with a single function value.

One-Point Gradient Estimator. For a function $f ( \mathbf { x } )$ , we define its $\delta$ -smooth version as

$$
\hat { f } _ { \delta } ( \mathbf { x } ) = \mathbb { E } _ { \mathbf { u } \sim B ^ { d } } [ f ( \mathbf { x } + \delta \mathbf { u } ) ]
$$

which satisfies the following lemma (Flaxman, Kalai, and McMahan 2005, Lemma 1).

Lemma 2. Let $\delta > 0$ , $\hat { f } _ { \delta } ( \mathbf { x } )$ defined in (18) ensures

$$
\nabla \hat { f } _ { \delta } ( \mathbf x ) = \mathbb { E } _ { \mathbf { u } \sim S ^ { d } } \left[ ( d / \delta ) f ( \mathbf x + \delta \mathbf u ) \mathbf u \right]
$$

where $S ^ { d }$ denotes the unit sphere in $\mathbb { R } ^ { d }$ .

To exploit the one-point gradient estimator, we define the shrunk set of $\kappa$ as stated below

$$
\begin{array} { r } { \mathcal { K } _ { \delta } = ( 1 - \delta / r ) \mathcal { K } = \{ ( 1 - \delta / r ) \mathbf { x } \mid \mathbf { x } \in \mathcal { K } \} , } \end{array}
$$

where $0 < \delta < r$ denotes the shrunk parameter.

Compared to our methods for the full-information setting, we make the following modifications:

• At each round $t$ , the decision $\mathbf { x } _ { t }$ consists of two parts:

$$
\mathbf { x } _ { t } = \mathbf { y } _ { t } + \delta \mathbf { u } _ { t }
$$

where $\mathbf { y } _ { t } ~ \in ~ \mathcal { K } _ { \delta }$ denotes an auxiliary decision learned from historical information, and ${ \mathbf { u } } _ { t } \sim { \mathcal { S } } ^ { d }$ is uniformly sampled from Sd;

• The gradient of $\tilde { f } _ { t } ( { \bf x } _ { t } )$ is approximated by the one-point gradient estimator:

$$
\tilde { \nabla } _ { t } = ( d / \delta ) [ \tilde { f } _ { t } ( \mathbf { x } _ { t } ) ] \mathbf { u } _ { t } ,
$$

so that we can adhere to the update rules in our previous methods;

• To manage the approximate error introduced by (21), we employ the blocking technique (Garber and Kretzu 2020; Hazan and Minasyan 2020) for decision updates, i.e., dividing the time horizon $T$ into equally-sized blocks and only updating decisions at the end of each block.

In the bandit setting, we also investigate the general convex losses and the strongly convex losses.

General Convex Losses. In this case, we incorporate the modifications (20) and (21), and the blocking technique into Algorithm 1. The detailed procedures are summarized in Algorithm 3. Specifically, for initialization, we set ${ \tilde { G } } _ { 1 } = m =$ 1, and choose any $\hat { \mathbf { y } } _ { 1 } \in \mathcal { K } _ { \delta }$ (Step 1). At each round $t$ , we update $\mathbf { y } _ { t } = \hat { \mathbf { y } } _ { m }$ where $\hat { \mathbf { y } } _ { m } \in \mathcal { K } _ { \delta }$ is the auxiliary decision used in the block $m$ , make the decision $\mathbf { x } _ { t }$ according to (20),

Algorithm 3: Bandit Frank-Wolfe with Time-Varying Constraints (BFW-TVC)   
Input: Hyper-parameters $\beta , \gamma , c , K , \epsilon$ , and the function $\overline { { \Phi ( \cdot ) } }$   

<html><body><table><tr><td></td></tr><tr><td>Input:Hyper-parameters β,γ，δ,K,L,and the function Φ(-), and the modulus of strong convexity α f 1: Choose any y1 ∈ Ks,and set m = 1 2:for t=1 to Tdo 3: Set yt = ym,and play Xt according to (20) 4: Suffer ft(xt) and gt(Xt) 5: Construct Qt and ft(x) according to (4) and (6)</td></tr></table></body></html>

1: Choose any $\hat { \mathbf { y } } _ { 1 } \in \mathcal { K } _ { \delta _ { 1 } }$ , and set ${ \tilde { G } } _ { 1 } = m = 1$   
2: for $t = 1$ to $T$ do   
3: Set $\mathbf { y } _ { t } = \hat { \mathbf { y } } _ { m }$ , and play $\mathbf { x } _ { t }$ according to (20)   
4: Suffer $f _ { t } ( \mathbf { x } _ { t } )$ and $g _ { t } ( \mathbf { x } _ { t } )$   
5: Construct $Q _ { t }$ and $\tilde { f } _ { t } ( { \bf x } )$ according to (4) and (6)   
6: Compute $\tilde { \nabla } _ { t }$ according to (21)   
7: if $t$ mod $K = 0$ then   
8: while $\tilde { G } _ { k } < \beta G ( \gamma + \Phi ^ { \prime } ( \beta Q _ { \tau } ) ) , \forall \tau$ in block do   
9: Set $\tilde { G } _ { k + 1 } = 2 \tilde { G } _ { k }$ , $k = k + 1$   
10: end while   
11: Compute ∇ˆ m = tτ=t K+1   
12: Set $\eta _ { k }$ and $F _ { b _ { k } : m } ( \mathbf { y } )$ according to (22) and (23)   
13: Set $\tilde { \mathbf { y } } _ { 1 } = \hat { \mathbf { y } } _ { m }$ and $\tau = 0$   
14: repeat   
15: Set $\tau = \tau + 1$   
16: Update $\mathbf { v } _ { \tau }$ and $\sigma _ { \tau }$ according to (24) and (25)   
17: Compute $\tilde { \mathbf { y } } _ { \tau + 1 }$ according to (26)   
18: until $\langle \nabla F _ { b _ { k } : m } ( \tilde { \mathbf { y } } _ { \tau } ) , \tilde { \mathbf { y } } _ { \tau } - \mathbf { v } _ { \tau } \rangle \leq \epsilon$   
19: Set $\hat { \mathbf { y } } _ { m + 1 } = \tilde { \mathbf { y } } _ { \tau + 1 }$ and $m = m + 1$   
20: end if   
21: end for

and suffer $f _ { t } ( \mathbf { x } _ { t } )$ and $g _ { t } ( \mathbf { x } _ { t } )$ (Steps 3-4). Then, we construct the CCV $Q _ { t }$ , the surrogate loss function $\tilde { f } _ { t } ( { \bf x } )$ and the gradient estimation $\tilde { \nabla } _ { t }$ according to (4), (6) and (21), respectively (Steps 5-6). At the end of the block $m$ , we update our decision. To be precise, we first evaluate the current guess $\tilde { G } _ { k }$ for the gradient norm bound: if it is unsuitable, we double the value until an appropriate $\tilde { G } _ { k }$ is found (Steps 8-10). At the Step 11, we compute the cumulative gradient estimation $\begin{array} { r } { \hat { \nabla } _ { m } = \sum _ { \tau = t - K + 1 } ^ { t } \tilde { \nabla } _ { \tau } } \end{array}$ tτ=t K+1 ∇˜ τ where K denotes the block size. At the Step 12, we set $\eta _ { k }$ according to

$$
\eta _ { k } = c D ( d M \tilde { G } _ { k } T ^ { 3 / 4 } ) ^ { - 1 } ,
$$

and construct $F _ { b _ { k } : m }$ according to

$$
F _ { b _ { k } : m } ( { \mathbf y } ) = \eta _ { k } \sum _ { \tau = b _ { k } } ^ { m } \Big \langle \hat { \nabla } _ { \tau } , { \mathbf y } \Big \rangle + \left\| { \mathbf y } - { \mathbf y } _ { s _ { k } } \right\| _ { 2 } ^ { 2 } ,
$$

where $b _ { k }$ denotes the first block that utilizes the estimation $\tilde { G } _ { k }$ , and $s _ { k }$ denotes the first round of $b _ { k }$ . Next, we update the auxiliary decision for the next block, and set $\tilde { \mathbf { y } } _ { 1 } = \mathbf { y } _ { m }$ and $\tau = 0$ (Step 12). At Steps 14-18, we repeat the following procedures: updating $\tau = \tau + 1$ , computing $\mathbf { v } _ { \tau }$ according to

$$
\begin{array} { r } { \mathbf { v } _ { \tau } \in \underset { \mathbf { y } \in \mathcal { K } _ { \delta } } { \mathrm { a r g m i n } } \left. \nabla F _ { b _ { k } : m } \left( \tilde { \mathbf { y } } _ { \tau } \right) , \mathbf { y } \right. , } \end{array}
$$

and $\sigma _ { \tau }$ according to

$$
\sigma _ { \tau } = \underset { \sigma \in [ 0 , 1 ] } { \operatorname { a r g m i n } } F _ { b _ { k } : m } \left( \tilde { \mathbf { y } } _ { \tau } + \sigma \left( \mathbf { v } _ { \tau } - \tilde { \mathbf { y } } _ { \tau } \right) \right) ,
$$

and updating $\tilde { \mathbf { y } } _ { \tau + 1 }$ according to

$$
\tilde { \mathbf { y } } _ { \tau + 1 } = \tilde { \mathbf { y } } _ { \tau } + \sigma _ { \tau } \left( \mathbf { v } _ { \tau } - \tilde { \mathbf { y } } _ { \tau } \right) ,
$$

until the stop condition $\langle \nabla F _ { b _ { k } : m } ( \tilde { \mathbf { y } } _ { \tau } ) , \tilde { \mathbf { y } } _ { \tau } - \mathbf { v } _ { \tau } \rangle \leq \epsilon$ is satisfied. After that, we set the auxiliary decision $\hat { \mathbf { y } } _ { m + 1 } = \tilde { \mathbf { y } } _ { \tau + 1 }$ for the next block.

With the configurations of the exponential Lyapunov function $\Phi ( x ) = \exp ( 2 ^ { - 1 } T ^ { - 3 / 4 } x ) - 1$ and suitable parameters, we obtain the following theorem.

Theorem 3. Let $\gamma = 1$ , $K = T ^ { 1 / 2 }$ , $\epsilon = 4 D ^ { 2 } T ^ { - 1 / 2 }$ and $c >$ 0 be a constant satisfying $\delta = c T ^ { - 1 / 4 } \leq r$ , and $\beta = C _ { 2 } ^ { - 1 }$ where $\gamma _ { 2 } = 2 ^ { 4 } G ( c D / r + 3 c + 1 + 2 c D / ( d M ) + d M D / \bar { c }$ ). Under Assumptions $^ { l }$ and 2, if the loss functions and the constraint functions are general convex, Algorithm 3 ensures the bounds of

$$
\mathbb { E } [ \mathrm { R e g r e t } _ { T } ] = \mathcal { O } ( T ^ { 3 / 4 } ) , Q _ { T } = \mathcal { O } ( T ^ { 3 / 4 } \log T ) .
$$

Remark. Theorem 3 presents tighter regret and CCV bounds, compared to the $\mathcal { O } ( T ^ { 3 / 4 } \sqrt { \log T } )$ regret bound and the $\mathcal { O } ( T ^ { 7 / 8 } \log T )$ CCV bound in Garber and Kretzu (2024).

Strongly Convex Losses. In this case, we also employ the one-point gradient estimator and the blocking technique, and summarize the procedures in Algorithm 4. Overall, our method for strongly convex losses is similar to Algorithm 3, with the primary difference in the update of decisions. tShpe ciufimcaullayt, vaet tghreadeiennd oefsteiamcaht bolno $m$ $\begin{array} { r } { \hat { \nabla } _ { m } = \sum _ { \tau = t - K + 1 } ^ { t } \mathsf { \bar { V } } _ { \tau } } \end{array}$ (Step 8), and then construct $F _ { m } ^ { s c } ( \mathbf { y } )$ as shown below:

$$
F _ { m } ^ { s c } ( \mathbf { y } ) = \sum _ { \tau = 1 } ^ { m } \left. \hat { \nabla } _ { \tau } , \mathbf { y } \right. + C _ { 3 } \| \mathbf { y } \| _ { 2 } ^ { 2 }
$$

where we denote $C _ { 3 } = \gamma \beta \alpha _ { f } t / 2$ for brevity, and set $\tilde { \bf y } _ { 1 } =$ ${ \bf y } _ { m }$ (Step 9). Next, we repeat the following procedures for $L$ times to refine the auxiliary decision (Steps 10-14): at the iteration $\tau \in [ L ]$ , computing $\mathbf { v } _ { \tau } ^ { s c }$ according to

$$
\begin{array} { r } { \mathbf { v } _ { \tau } ^ { s c } \in \underset { \mathbf { y } \in \mathcal { K } _ { \delta } } { \mathrm { a r g m i n } } \left. \nabla F _ { m } ^ { s c } \left( \tilde { \mathbf { y } } _ { \tau } \right) , \mathbf { y } \right. , } \end{array}
$$

![](images/8df81dba28446008e59f2ebdc00e0d737e8a2e2d573420c3cb080331e5908a55.jpg)  
Figure 1: Experimental results on the MovieLens dataset.

calculating $\sigma _ { t } ^ { s c }$ according to

$$
\begin{array} { r } { \sigma _ { \tau } ^ { s c } = \underset { \sigma \in [ 0 , 1 ] } { \operatorname { a r g m i n } } F _ { m } ^ { s c } \left( \tilde { \mathbf { y } } _ { \tau } + \sigma \left( \mathbf { v } _ { \tau } ^ { s c } - \tilde { \mathbf { y } } _ { \tau } \right) \right) , } \end{array}
$$

and updating $\tilde { \mathbf { y } } _ { \tau + 1 }$ according to

$$
\tilde { \mathbf { y } } _ { \tau + 1 } = \tilde { \mathbf { y } } _ { \tau } + \sigma _ { \tau } ^ { s c } \left( \mathbf { v } _ { \tau } ^ { s c } - \tilde { \mathbf { y } } _ { \tau } \right) .
$$

Finally, we set the auxiliary decision for the next block as $\mathbf { y } _ { m + 1 } = \tilde { \mathbf { y } } _ { L + 1 }$ (Step 15).

By setting the quadratic Lyapunov function $\Phi ( x ) = x ^ { 2 }$ and proper parameters, we obtain the following theorem.

Theorem 4. Let $\beta = G ^ { - 1 } D ^ { - 1 } T ^ { - 2 / 3 }$ , $\gamma = G / ( G + \alpha _ { f } D )$ , $K \ = \ L \ = \ T ^ { 2 / 3 }$ , and $\delta { \bf \Psi } = c T ^ { - 1 / 3 }$ with $c > 0$ satisfying $c T ^ { - 1 / 3 } < r$ , and $\gamma ~ = ~ \mathcal { O } ( T ^ { 2 / 3 } )$ . Under Assumptions $^ { l }$ and 2, if the loss functions are $\alpha _ { f }$ -strongly convex, and the constraint functions are general convex, Algorithm 4 ensures the bounds of

$$
\mathbb { E } \left[ \mathrm { R e g r e t } _ { T } \right] = \mathcal { O } ( T ^ { 2 / 3 } \log T ) , Q _ { T } = \mathcal { O } ( T ^ { 5 / 6 } \log T ) .
$$

Remark. Theorem 4 provides the first regret and CCV bounds for the strongly convex case with bandit feedback in projection-free COCO. By utilizing the strong convexity of $f _ { t } ( \cdot )$ , both of our results are tighter than those established for the general convex losses in Garber and Kretzu (2024).

# Experiments

In this section, we conduct empirical studies on real-world datasets to evaluate our theoretical findings.

General Setup. We investigate the online matrix completion problem (Hazan and Kale 2012; Lee, Ho-Nguyen, and Lee 2023), the goal of which is to generate a matrix $X$ in an online manner to approximate the target matrix $M \in \mathbb { R } ^ { m \times n }$ . Specifically, at each round $t$ , the learner receives a sampled data $( i , j )$ with the value $M _ { i j }$ from the observed subset $O$ of $M$ . Then, the learner chooses a matrix $X$ from the trace norm ball $\mathcal { K } = \{ X \vert \Vert X \Vert _ { * } \ \leq \ \delta , X \in \mathbb { R } ^ { m \times n } \}$ where $\delta > 0$ is the parameter, and suffers the strongly convex cost loss $\begin{array} { r } { f _ { t } ( X _ { t } ) = \sum _ { ( i , j ) \in O } ( X _ { i j } - M _ { i j } ) ^ { 2 } / 2 } \end{array}$ and the constraint loss $g _ { t } ( X _ { t } ) = \mathrm { T r } ( P _ { t } X _ { t } )$ where $P _ { t }$ is uniformly sampled from $[ - 1 , 1 ] ^ { n \times m }$ . The experiments are conducted with $\bar { \delta } \ = \ 1 0 ^ { 4 }$ on two real-world datasets: MovieLens1 for the full-information setting, and Film Trust (Guo, Zhang, and Yorke-Smith 2013) for the bandit setting.

![](images/65ccd9037a8e92260958361306a0992dc04904574e33fe4b05f5c2cf68062d6d.jpg)  
Figure 2: Experimental results on the Film Trust dataset.

Baselines. We choose three projection-free COCO methods as the contenders: (i) OPDP (Lee, Ho-Nguyen, and Lee 2023, Algorithm 1) and LPM (Garber and Kretzu 2024, Algorithm 4) for the full-information setting; (ii) LBPM (Garber and Kretzu 2024, Algorithm 5) for the bandit setting. All parameters of each method are set according to their theoretical suggestions, and we choose the best hyper-parameters from the range of $[ 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , \allowbreak . . . , 1 0 ^ { 4 } , 1 0 ^ { 5 } ]$ .

Results. All experiments are repeated 10 times and we report experimental results (mean and standard deviation) in Figures 1 and 2. As evident from the results, in the fullinformation setting, OFW-TVC outperforms its competitors significantly in terms of both two metrics. Moreover, by utilizing the strong convexity of $f _ { t } ( \cdot )$ , our SCOFW-TVC yields the lowest cumulative cost loss, albeit with a slight compromise on CCV. Similarly, in the bandit setting, it can be observed that our methods consistently outperform others, aligning with the theoretical guarantees.

# Conclusion and Future Work

In this paper, we investigate projection-free COCO and propose a series of methods for the full-information and bandit settings. The key idea is to utilize the Lyapunov-based technique to construct a composite surrogate loss, consisting of the original cost and the constraint loss, and employ parameter-free variants of OFW running over the surrogate loss to simultaneously optimize the regret and CCV. In this way, we improve previous results for general convex cost losses and establish novel regret and CCV bounds for strongly convex cost losses. During the analysis, we propose the first parameter-free variant of OFW for general convex losses, which may hold independent interest. Finally, empirical studies have verified our theoretical findings.

Currently, for strongly convex losses, we improve the regret bound from $\mathcal { O } ( T ^ { 3 / 4 } )$ to $\mathcal { O } ( T ^ { 2 / 3 } )$ , but sacrifice another metric CCV with a marginally looser bound of $\mathcal { O } ( T ^ { 5 / 6 } )$ , compared to our results for general convex losses. This phenomenon may be due to the potential impropriety of the quadratic Lyapunov function. Hence, one possible solution is to choose other more powerful functions, which seems highly non-trivial, and we leave it as future work.

# Acknowledgments

This work was partially supported by National Science and Technology Major Project (2022ZD0114801), and NSFC (U23A20382, 62306275). We would like to thank the anonymous reviewers for their constructive suggestions.