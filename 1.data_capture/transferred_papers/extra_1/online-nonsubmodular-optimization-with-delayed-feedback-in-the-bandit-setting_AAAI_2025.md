# Online Nonsubmodular Optimization with Delayed Feedback in the Bandit Setting

Sifan Yang1,2, Yuanyu $\mathbf { W a n } ^ { 3 , 1 , * }$ , Lijun Zhang1,3,\*

1National Key Laboratory for Novel Software Technology, Nanjing University, Nanjing, China 2School of Artificial Intelligence, Nanjing University, Nanjing, China   
2State Key Laboratory of Blockchain and Data Security, Zhejiang University, Hangzhou, China {yangsf,zhanglj}@lamda.nju.edu.cn, wanyy $@$ zju.edu.cn

# Abstract

We investigate the online nonsubmodular optimization with delayed feedback in the bandit setting, where the loss function is $\alpha$ -weakly DR-submodular and $\beta$ -weakly DRsupermodular. Previous work has established an $( \alpha , \beta )$ -regret bound of $\mathcal { O } ( n d ^ { 1 / 3 } T ^ { 2 / 3 } )$ , where $n$ is the dimensionality and $d$ is the maximum delay. However, its regret bound relies on the maximum delay and is thus sensitive to irregular delays. Additionally, it couples the effects of delays and bandit feedback as its bound is the product of the delay term and the $\mathcal { O } ( n T ^ { 2 / 3 } )$ regret bound in the bandit setting without delayed feedback. In this paper, we develop two algorithms to address these limitations, respectively. Firstly, we propose a novel method, namely DBGD-NF, which employs the one-point gradient estimator and utilizes all the available estimated gradients in each round to update the decision. It achieves a better $\mathcal { O } ( n \bar { d } ^ { 1 / 3 } T ^ { 2 / 3 } )$ regret bound, which is relevant to the average delay $\begin{array} { r } { \bar { d } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } d _ { t } \leq d } \end{array}$ . Secondly, we extend DBGD-NF by employing a blocking update mechanism to decouple the joint effect of the delays and bandit feedback, which enjoys an $\mathcal { O } ( n ( T ^ { 2 / 3 } + \dot { \sqrt { d T } } ) )$ regret bound. When $d = \mathcal { O } ( T ^ { 1 / 3 } )$ , our regret bound matches the $\mathcal { O } ( n T ^ { 2 / 3 } )$ bound in the bandit setting without delayed feedback. Compared to our first $\mathcal { O } ( n \bar { d } ^ { 1 / 3 } T ^ { 2 / 3 } )$ bound, it is more advantageous when the maximum delay $d = o ( \bar { d } ^ { 2 / 3 } T ^ { 1 / 3 } )$ . Finally, we conduct experiments on structured sparse learning to demonstrate the superiority of our methods.

# Introduction

Online learning is a powerful framework that has been used to model various sequential prediction problems (ShalevShwartz 2012). It can address scenarios in which decisions are made from a small set (Hazan and Kale 2012), a continuous space (Hazan et al. 2016), or a combinatorial feasible domain (Cesa-Bianchi and Lugosi 2012). In this paper, we study the online nonsubmodular optimization (Lin et al. 2022), an emerging branch of the online learning, which appears in many machine learning tasks like structured sparse learning (El Halabi and Cevher 2015), Bayesian optimization (Gonza´lez et al. 2016), and column subset selection (Sviridenko, Vondra´k, and Ward 2017), etc. Similar to the classical online convex optimization (OCO) (Zinkevich 2003), it is typically formulated as a game between an online player and an adversary. In each round $t \in [ T ]$ , the player begins by selecting a subset decision $S _ { t } \subseteq [ n ]$ . When the player submits its decision $S _ { t }$ , the adversary chooses a nonsubmodular loss function $f _ { t } ( \cdot ) : 2 ^ { [ n ] } \mapsto \bar { \mathbb { R } }$ and then the player suffers a loss $f _ { t } ( S _ { t } )$ , where $f _ { t } ( \cdot ) \ =$ $\bar { f } _ { t } \left( \cdot \right) - \underline { { f } } _ { t } \left( \cdot \right) , \bar { f } _ { t } \left( \cdot \right)$ is $\alpha$ -weakly diminishing return submodular (DR-submodular), $\underline { { f } } _ { t } \left( \cdot \right)$ is $\beta$ -weakly diminishing return supermodular (DR-supermodular) and $2 ^ { [ n ] }$ represents all the subset of $[ n ]$ .1

The player aims to minimize the cumulative loss over $T$ rounds, equivalently minimizing the regret:

$$
\mathbf { R e g } ( T ) \triangleq \sum _ { t = 1 } ^ { T } f _ { t } ( S _ { t } ) - \operatorname* { m i n } _ { S \subseteq [ n ] } \sum _ { t = 1 } ^ { T } f _ { t } ( S ) ,
$$

which compares the excess loss suffered by the player with that of the best decision chosen in hindsight. As pointed out by El Halabi and Jegelka (2020), the optimization problem $\begin{array} { r } { \operatorname* { m i n } _ { S \subseteq [ n ] } \sum _ { t = 1 } ^ { T } f _ { t } ( \bar { S } ) } \end{array}$ is NP-hard, thus it is impossible to find the optimal decision in polynomial time. For this reason, we follow the previous work (Lin et al. 2022) and apply the $( \alpha , \beta )$ -regret to measure the performance of the online player, which is defined as

$$
\mathbf { R e g } _ { \alpha , \beta } ( T ) \triangleq \sum _ { t = 1 } ^ { T } f _ { t } ( S _ { t } ) - \left( \frac { 1 } { \alpha } \bar { f } _ { t } \left( S ^ { \star } \right) - \beta \underline { { f } } _ { t } \left( S ^ { \star } \right) \right) ,
$$

where $S ^ { * } = \arg \operatorname* { m i n } _ { S \subseteq [ n ] } \sum _ { t = 1 } ^ { T } f _ { t } ( S )$ and $( \alpha , \beta )$ are the approximation factors achieved by a certain offline algorithm. Lin et al. (2022) are the first to investigate the online nonsubmodular optimization and develop a method that achieves an $( \alpha , \beta )$ -regret bound of $\mathcal { O } ( \sqrt { n T } )$ , building on the Lova´sz extension (Lova´sz 1983) and the convex relaxation model (El Halabi and Jegelka 2020), where $n$ is the dimensionality.

In lots of real-world scenarios, there may be a potential delay between the query of the player and the corresponding response (Quanrud and Khashabi 2015; Wan et al. 2022, 2023). To address the delayed scenario, Lin et al. (2022) also explore the problem of online nonsubmodular optimization with delayed feedback, where the online player incurs an arbitrary delay $d _ { t } ~ \geq ~ 1$ in receiving the response. To handle the delayed feedback, they utilize the pooling strategy (He´liou, Mertikopoulos, and Zhou 2020) to propose delay online approximate gradient descent (DOAGD), which enjoys an $\mathcal { O } ( \sqrt { n d T } )$ regret bound, where $d$ is the maximum delay. Moreover, they consider a more challenging setting, bandit feedback, in which the online player does not receive any additional information about the loss function $f _ { t } ( \cdot )$ (e.g., its gradient) beyond the value $f _ { t } ( S _ { t } )$ , and develop delay online approximate gradient descent (DBAGD), achieving an $\mathcal { O } ( n d ^ { 1 / 3 } T ^ { 2 / 3 } )$ regret bound. However, their result for bandit setting with delayed feedback, summarized in Table 1, has two limitations. Firstly, it relies on the maximum delay $d$ , which renders it sensitive to irregular delays. Secondly, it is the product of the delay term and the $\mathcal { O } ( n T ^ { 2 / 3 } )$ regret bound in the non-delayed bandit setting. This arises from DBAGD coupling the effects of the delays and bandit feedback, resulting in a discontented regret bound.

Table 1: Summary of results for online nonsubmodular optimization under different settings, where $n$ is the dimensionality, $d$ is the maximum delay and $\begin{array} { r } { \bar { d } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } d _ { t } } \end{array}$ is the average delay over $T$ rounds. For simplicity, we use the abbreviations: full $$ full-information setting with delayed feedback, bandit $$ bandit setting with delayed feedback.   

<html><body><table><tr><td>Setting</td><td>Method</td><td>(α,β)-regret bound</td></tr><tr><td>bandit</td><td>DBAGD (Lin et al. 2022)</td><td>0 (nd1/3T2/3)</td></tr><tr><td>bandit</td><td>BDGD-NF(Theorem 1)</td><td>O (nd1/3T2/3)</td></tr><tr><td>bandit</td><td>BDBGD-NF (Theorem 3) O</td><td>(n(T2/3 + √dT))</td></tr><tr><td>full</td><td>DOAGD (Lin et al. 2022)</td><td>O(√ndT)</td></tr><tr><td>full</td><td>DOGD-NF (Theorem 2)</td><td>0(VndT)</td></tr></table></body></html>

To overcome these limitations, we revisit the online nonsubmodular optimization with delayed feedback in the bandit setting. Specifically, we first develop a delayed algorithm to establish a regret bound that is relevant to the average delay. Our proposed method, named delayed bandit gradient descent for nonsubmodular function (DBGD-NF), achieves an enhanced $\mathcal { O } \left( n \bar { d } ^ { 1 / 3 } T ^ { 2 / 3 } \right)$ regret bound, where $\begin{array} { r } { \bar { d } ~ = ~ \frac { 1 } { T } \sum _ { t = 1 } ^ { T } d _ { t } ~ \le ~ d _ { \bar { \mathbf { \Gamma } } } } \end{array}$ represents the average delay. The primary idea is to employ the one-point gradient estimator (Hazan and Kale 2012; Lin et al. 2022) and use all available estimated gradients to update the decision in each round, instead of utilizing the oldest one like DBAGD. Furthermore, as a by-product, if the online player has access to the gradient of the loss function, we can substitute the estimated gradient in DBGD-NF with the true gradient. Our algorithm, referred to as delayed online gradient descent for nonsubmodular function (DOGD-NF), enjoys a better $\mathcal { O } ( \sqrt { n \bar { d } T } )$

regret bound for the full-information setting with delayed feedback.

In our pursuit of decoupling the joint effect of delayed feedback and the gradient estimator, we develop the blocking delayed bandit gradient descent for nonsubmodular function (BDBGD-NF). Drawing inspiration from Wan et al. (2024), we adopt the blocking update mechanism (Zhang et al. 2019; Garber and Kretzu 2020; Wang et al. 2023, 2024) with DBGD-NF. Particularly, we divide the total $T$ rounds into several blocks of size $K$ and update the decision at the end of each block using the estimated gradients from the blocks where all gradients are available. By setting a appropriate block size $K$ , we can reduce the variance of the one-point gradient estimator. Leveraging this technique, BDBGD-NF achieves a superior $\mathcal { O } ( n ( T ^ { 2 / 3 } + \sqrt { d T } ) )$ regret bound. When the algorithm faces small delays, i.e., maximum delay $d = \mathcal { O } \bar { ( } T ^ { 1 / 3 } )$ , this regret bound matches the existing $\mathcal { O } ( n T ^ { 2 / 3 } )$ regret bound in the non-delayed setting (Lin et al. 2022), benefiting from the blocking update mechanism.

On the other hand, when the impact of the delayed feedback $d$ is substantial, i.e., $d = \Omega \bar { ( } T ^ { 1 / 3 } )$ , our regret bound is on the same order as the $\mathcal { O } ( \sqrt { n \bar { d } T } )$ bound we establish for the full-information setting in terms of $d$ and $T$ under the worst case, where $\bar { d } = \Theta ( \bar { d } )$ . Moreover, it is better than our first $\mathcal { O } \left( n \bar { d } ^ { 1 / 3 } T ^ { 2 / 3 } \right)$ bound when the maximum delay $d = o ( \bar { d } ^ { 2 / 3 } T ^ { 1 / 3 } )$ . Notably, compare to Wan et al. (2024), BDBGD-NF is specifically designed for online nonsubmodular optimization, making it more challenging to analyze. Finally, we compare our algorithms with the state-of-the-art (SOTA) methods through numerical experiments to demonstrate the robustness and effectiveness in handling delayed and bandit feedback effects.

To summarize, this paper makes the following contributions to online nonsubmodular optimization with delayed feedback:

• We propose two algorithms for online nonsubmodular optimization with delayed feedback to derive the regret bounds that are relevant to the average delay. Our methods reduce the regret bounds to $\mathcal { O } ( \sqrt { n \bar { d } T } )$ and $\mathcal { O } \left( n \bar { d } ^ { 1 / 3 } T ^ { 2 / 3 } \right)$ for full-information and bandit feedback settings, respectively.   
• To decouple the joint effect of the delayed and bandit feedback, we develop a novel algorithm by utilizing a blocking update technique, which enjoys an $\dot { \mathcal { O } } ( n ( T ^ { 2 / 3 } +$ $\sqrt { d T } )$ ) regret bound.

# Related Work

In this section, we briefly introduce the related work of submodular optimization and nonsubmodular optimization.

# Submodular Optimization

Submodular optimization has garnered increasing interest in various practical applications, such as sparse reconstruction (Bach 2010; Das, Dasgupta, and Kumar 2012; Liao et al. 2023), graph inference (Gomez-Rodriguez and Scho¨lkopf

2012; Defazio and Caetano 2012), video analysis (Zheng et al. 2014), object detection (Song et al. 2014), etc. The primary property of a submodular function is diminishing returns, meaning that adding an element to a larger set provides less incremental gain than adding it to a smaller set. Early research (Badanidiyuru and Vondra´k 2014; Krause and Golovin 2014) on submodular optimization mainly focuses on the offline setting, which may be unsuitable for sequential decision-making problems. To overcome this issue, Hazan and Kale (2012) investigate the online submodular optimization, where the player chooses a subset from a ground set of $n$ elements in round $t$ , and then observe a submodular loss function. Based on the Lova´sz extension (Lova´sz 1983), they extend online gradient descent (OGD) (Zinkevich 2003) and bandit gradient descent (BGD) (Flaxman, Kalai, and McMahan 2005) into submodular optimization in both full-information and bandit feedback settings, establishing $\mathcal { O } ( \sqrt { n T } )$ and $\mathcal { O } ( n T ^ { 2 / 3 } )$ regret bounds, respectively, where $n$ is the dimensionality.

# Nonsubmodular Optimization

Although submodularity is a natural assumption, the objective function is not always exactly submodular in various applications, such as structured sparse learning (El Halabi and Cevher 2015), Bayesian optimization (Gonza´lez et al. 2016), and column subset selection (Sviridenko, Vondra´k, and Ward 2017), etc. Instead, it satisfies a weaker version of the DR property, like $\alpha$ -weakly DR-submodular and $\beta$ - weakly DR-supermodular (Lehmann, Lehmann, and Nisan 2006). El Halabi and Jegelka (2020) provide the first approximation guarantee for nonsubmodular minimization by developing an approximate projected subgradient method. Nevertheless, they only focus on the offline setting.

Lin et al. (2022) pioneer the study of online nonsubmodular optimization and introduce the $( \alpha , \beta )$ -regret to measure the performance of the online player, which is defined in (2). Based on the Lova´sz extension (Lova´sz 1983) and the convex relaxation model (El Halabi and Jegelka 2020), they propose online approximate gradient descent (OAGD), which obtains an $\mathcal { O } ( \sqrt { n T } )$ regret bound. Particularly, it uses the subgradient of the convex relaxation function to perform a gradient descent step over the Lova´sz extension domain. Then it samples the decision $S _ { t }$ from a certain distribution over all possible sets at round $t$ . Moreover, they also consider the full-information and bandit settings with delay feedback, which are more complex to analyze. To handle the delay feedback, they further extend OAGD by adopting a pooling strategy (He´liou, Mertikopoulos, and Zhou 2020). Their method, namely DOAGD, which keeps a pool to store all the available delayed information and utilizes the oldest received but not utilized gradient to update the decision in each round, achieves an $\mathcal { O } ( \sqrt { n d T } )$ regret bound, where $d$ is the maximum delay. Nevertheless, it depends on the maximum delay, and thus is sensitive to irregular delays. In the bandit setting, since the player can only observe the loss value of its decision $S _ { t }$ , they employ the importance weighting technique to estimate the gradient. In particular, $S _ { t }$ is chosen from a distribution that is related to the decision with probability $1 - \mu$ $0 < \mu < 1 _ { \cdot }$ ) and a random distribution with probability $\mu$ for exploration, ensuring the variance of the gradient is upper bounded by $\mathcal { O } ( n ^ { 2 } / \bar { \mu } )$ . While they establish an $\mathcal { O } ( n d ^ { 1 / 3 } T ^ { 2 / 3 } )$ regret bound, their method couples the effects of the delays and gradient estimator.

# Preliminary

In this section, we will provide essential definitions and basic setup for optimization of the nonsubmodular functions.

# Definitions and Assumptions

Definition 1 For any function $f ( \cdot ) : 2 ^ { [ n ] } \mapsto \mathbb { R } ,$ , we define $f ( i \mid S ) = f ( \{ i \} \bigcup { S } ) - f ( S )$ to denote the marginal gain of adding an element i to $S$ . Moreover, $f ( \cdot )$ is normalized $i f$ and only if $f ( \varnothing ) \ = \ 0$ and nondecreasing if and only $i f$ $f ( A ) \leq f ( B )$ for any $A \subseteq B . \Pi _ { [ 0 , 1 ] ^ { n } }$ is the projection onto the domain $[ 0 , 1 ] ^ { n }$ , which can be efficiently implemented as a simple clipping operation.

Definition 2 A function $f ( \cdot ) : 2 ^ { [ n ] } \mapsto \mathbb { R }$ is $\alpha$ -weakly DRsubmodular with $\alpha > 0$ if

$$
f ( i \mid A ) \ge \alpha f ( i \mid B ) , f o r a l l A \subseteq B , i \in [ n ] \backslash B .
$$

If this inequality holds when $\alpha = 1 , f ( \cdot )$ is submodular.

Similarly, $f ( \cdot ) : 2 ^ { [ n ] } \mapsto \mathbb { R }$ is $\beta$ -weakly $D R$ –supermodular with $\beta > 0 \ i f$

$$
f ( i \mid B ) \geq \beta f ( i \mid A ) , \quad f o r a l l \ A \subseteq B , i \in [ n ] \backslash B .
$$

$f ( \cdot )$ is supermodular when $\beta = 1$

We define that $f ( \cdot )$ is $( \alpha , \beta )$ -weakly $D R$ -modular if both of the above inequalities hold simultaneously.

Building on the above definitions, we formulate the problem of minimizing structured nonsubmodular functions (El Halabi and Jegelka 2020; Lin et al. 2022):

$$
f ( S ) : = { \bar { f } } ( S ) - \underline { { f } } ( S ) ,
$$

where $S \subseteq [ n ]$ . Afterwards, we introduce two common assumptions in the online nonsubmodular optimization.

Assumption 1 All the nonsubmodular functions $f ( \cdot )$ defined in (3) satisfy ${ \bar { f } } ( S ) + \underline { { f } } ( S ) \leq L$ for all $S \subseteq [ n ]$ .

Assumption 2 $\bar { f } ( \cdot )$ and $\underline { { f } } ( \cdot )$ defined in (3) are normalized and non-decreasing. $\bar { f } ( \cdot )$ is $\alpha$ -weakly $D R$ -submodular and $\underline { { f } } ( \cdot )$ is $\beta$ -weakly $D R$ -supermodular.

Then we give an application of the online nonsubmodular optimization to provide practical insights of these assumptions, which is also used in our later experiments.

Structured sparse learning. This problem aims to learn a sparse parameter vector whose support satisfies a specific structure, such as group-sparsity, clustering, tree-structure, or diversity (Kyrillidis et al. 2015). It is typically formulated as $\begin{array} { r } { \operatorname* { m i n } _ { \mathbf { x } \in \mathbb { R } ^ { n } } \ell ( \mathbf { x } ) + \gamma F ( \mathrm { s u p p } ( \mathbf { x } ) ) } \end{array}$ , where $\ell ( \cdot ) : \mathbb { R } ^ { n } \mapsto \mathbb { R }$ is the loss function, $F ( \cdot ) : 2 ^ { [ n ] } \mapsto \mathbb { R }$ is a set function that imposes restrictions on the support set and $\gamma$ is a tradeoff parameter. Previous approaches (Bach 2010; El Halabi and Cevher 2015) often replace the discrete regularizer $F ( \operatorname { s u p p } ( \mathbf { x } ) )$ with its closest convex relaxation, which is computationally tractable only when $F ( \cdot )$ is submodular. El Halabi and Jegelka (2020) introduce an alternative formulation by using a nonsubmodular regularizer, which is better in practice, defined as

$$
\operatorname* { m i n } _ { S \subseteq [ n ] } H ( S ) = \gamma F ( S ) - G ( S ) ,
$$

where $G ( S ) = \ell ( 0 ) - \mathrm { { m i n } _ { s u p p } ( \mathbf { x } ) } { \subseteq } S \ell ( \mathbf { x } )$ is a normalized non-decreasing set function. El Halabi and Jegelka (2020) have pointed out that when $\ell ( \cdot )$ is smooth, strongly convex and is generated from random data, $G ( \cdot )$ is weakly DRmodular. Moreover, if $F ( \cdot )$ is $\alpha$ -weakly DR-submodular, the equation (4) can be transformed into (3) so that we can handle it directly. For example, $F ( \cdot )$ is often chosen as the range cost function (Bach 2010) $\begin{array} { r } { \dot { } \alpha = \frac { 1 } { n - 1 } } \end{array}$ n1 1 and n is the dimensionality), applied in the time-series and cancer diagnosis (Rapaport, Barillot, and Vert 2008), and the cost function $\begin{array} { r } { ( \alpha = \frac { 1 + a } { 1 + b - a } } \end{array}$ 1+a and a, b are cost parameters), applied in the healthcare (Sakaue 2019).

Since nonsubmodular functions are defined over the discrete domain, determining their minimum values is a challenging task. Therefore, we introduce the Lova´sz extension (Lova´sz 1983) to transform a function $f ( \cdot )$ defined over a discrete domain $[ n ]$ to a new function $f _ { L } ( \cdot )$ over the unit hypercube $[ 0 , 1 ] ^ { n }$ . The extended function $f _ { L } ( \cdot )$ is convex if and only if $f ( \cdot )$ is submodular. For nonsubmodular functions, we can exploit the convex relation (El Halabi and Jegelka 2020), enabling the use of convex optimization algorithms on the transformed function.

# Lova´sz Extension and Convex Relaxation

Lova´sz extension ensures that the minima of the function over the domain $[ 0 , 1 ] ^ { n }$ also recover the minima of the original function $f ( \cdot )$ . In this way, we can reduce the complex optimization task over domain $[ n ]$ to a simpler convex optimization problem. To clarify this reduction process, we start with some necessary definitions.

Definition 3 A max chain of subsets of $[ n ]$ is a collection of sets $\{ A _ { 0 } , . . . , A _ { n } \}$ , and $\varnothing = A _ { 0 } \subseteq A _ { 1 } \subseteq \ldots \subseteq A _ { n } = [ n ] .$ .

For any $\mathbf { x } \in [ 0 , 1 ] ^ { n } = \mathcal { X }$ , we introduce a unique associated permutation $\pi : [ n ] \mapsto [ n ]$ such that $\pi ( i ) = j$ , meaning that $\mathbf { x } _ { j }$ is the $i$ -th largest number in x. Notably, we have $1 \geq$ $\mathbf { x } _ { \pi ( 1 ) } \geq . . . \geq \mathbf { x } _ { \pi ( n ) } \geq 0$ and let $\mathbf { x } _ { \pi ( 0 ) } = 1 , \mathbf { x } _ { \pi ( n + 1 ) } = 0$ for simplicity. If we set $A _ { i } = \{ \pi ( 1 ) , . . . , \pi ( i ) \}$ for all $i \in [ n ]$ and $A _ { 0 } = \varnothing$ , the vector x can be expressed as a convex combination, i.e., $\begin{array} { r } { \mathbf { x } = \sum _ { i = 0 } ^ { n } \lambda _ { i } \chi ( A _ { i } ) } \end{array}$ , where $\lambda _ { i } = \mathbf { x } _ { \pi ( i ) } - \mathbf { x } _ { \pi ( i + 1 ) }$ and $\textstyle \sum _ { i = 0 } ^ { n } \lambda _ { i } = 1 , \lambda _ { i } \in [ 0 , 1 ]$ (Hazan and Kale 2012). For any set $S \subseteq [ n ]$ , $\chi ( \cdot ) : 2 ^ { [ n ] } \mapsto \{ 0 , 1 \} ^ { n }$ is an indicator function $\chi ( S ) _ { i } = 1$ for all $i \in S$ and $\chi \big ( S \big ) _ { i } = 0$ for all $i \not \in S$ . Next, we give the definition of the Lov´asz extension.

Definition 4 For any submodular function $f ( \cdot )$ , its Lov´asz extension $f _ { L } ( \cdot ) : \mathcal { X } = [ 0 , 1 ] ^ { n } \mapsto \mathbb { R }$ is defined as $f _ { L } ( \mathbf { x } ) =$ $\begin{array} { r } { \sum _ { i = 0 } ^ { n } ( \mathbf { x } _ { \pi ( i ) } - \mathbf { x } _ { \pi ( i + 1 ) } ) f ( A _ { i } ) = \sum _ { i = 0 } ^ { n } \mathbf { x } _ { \pi ( i ) } f ( \pi ( i ) \mid \dot { A } _ { i - 1 } ) } \end{array}$ .

It is not hard to verify that $f _ { L } ( \chi ( S ) ) = f ( S )$ for any $S \subseteq$ $[ n ]$ . Therefore, minimizing the Lova´sz extension is equivalent to minimizing the original submodular function over all possible sets. Moreover, Edmonds (1970) has pointed out that the subgradient $\mathbf { g }$ of $f _ { L } ( \mathbf { x } )$ can be computed by

$$
\begin{array} { r } { \mathbf { g } _ { \pi ( i ) } = f \left( A _ { i } \right) - f \left( A _ { i - 1 } \right) \mathrm { f o r } \mathrm { a l l } i \in [ n ] . } \end{array}
$$

However, when $f ( \cdot )$ is not submodular, many properties break down. For example, $f _ { L } ( \cdot )$ is non-convex, which is harder to analyze. To tackle the nonsubmodular functions, we adopt the convex closure $f _ { C } ( \cdot )$ , which is defined as:

Definition 5 The convex closure $f _ { C } ( \cdot ) : [ 0 , 1 ] ^ { n } \mapsto \mathbb { R }$ for a nonsubmodular function $f ( \cdot )$ is the point-wise largest convex function which always lower bounds $f ( \cdot )$ . Additionally, $f _ { C } ( \cdot )$ is the tightest convex extension of $f ( \cdot )$ and $\begin{array} { r } { \operatorname* { m i n } _ { S \subseteq [ n ] } f ( S ) = \operatorname* { m i n } _ { x \in [ 0 , 1 ] ^ { n } } f _ { C } ( \mathbf { x } ) . } \end{array}$ .

Definition 5 gives us a simpler way to analyze the nonsubmodular function. Unfortunately, it is NP-hard to evaluate and optimize $f _ { C } ( \cdot )$ (Vondra´k 2007). Nevertheless, we can utilize the proposition 3.1 in Lin et al. (2022) to derive the approximation:

Lemma 1 Assuming $f ( \cdot )$ satisfies Assumption 2 and $\mathbf { g }$ is calculated according to (5) for all $A \subseteq [ n ]$ and $\mathbf { x } \in \mathcal { X }$ , we have the following guarantees

$$
\begin{array} { c } { f _ { L } ( \mathbf { x } ) = \displaystyle \langle \mathbf { g } , \mathbf { x } \rangle \geq f _ { C } ( \mathbf { x } ) , } \\ { \displaystyle \sum _ { i \in A } \mathbf { g } _ { i } \leq \frac { 1 } { \alpha } \bar { f } ( A ) - \beta \underline { { f } } ( A ) , } \\ { f _ { C } ( \mathbf { x } ) \leq f _ { L } ( \mathbf { x } ) = \displaystyle \langle \mathbf { g } , \mathbf { x } \rangle \leq \frac { 1 } { \alpha } \bar { f } _ { C } ( \mathbf { x } ) - \beta \underline { { f } } _ { C } ( \mathbf { x } ) . } \end{array}
$$

Remark 1 Lemma 1 demonstrates how Lova´sz extension $f _ { L } ( \mathbf { x } )$ approximates the convex closure $f _ { C } ( \mathbf { x } )$ so that the subgradient of $f _ { L } ( \mathbf { x } )$ can serve as the approximate subgradient for $f _ { C } ( \mathbf { x } )$ (El Halabi and Jegelka 2020), which plays an important role in our analysis.

# Problem Setup

We consider the online nonsubmodular optimization with delayed feedback in the bandit setting, where the loss function $f _ { t } ( \cdot )$ is defined in (3), and satisfies Assumption 1 and Assumption 2. In each round $t$ , the player makes a decision $S _ { t } \subseteq [ n ]$ and then triggers a delay $d _ { t }$ when receiving the loss value. The response will arrive at round $t + d _ { t } - 1$ and the player receives $\{ f _ { k } ( S _ { k } ) | k \in \mathcal { F } _ { t } \}$ , where $\mathcal { F } _ { t } = \{ k \ | \ k + d _ { k } - 1 = t \}$ represents the index set of received loss values in round $t$ . To measure the performance of the online player, we follow the previous work (Lin et al. 2022) and apply the $( \alpha , \beta )$ -regret defined in (2). It compares the loss of the player’s decisions to the result returned by an offline algorithm that approximately solves the optimization problem $\begin{array} { r } { \operatorname* { m i n } _ { S \subseteq [ n ] } \sum _ { t = 1 } ^ { T } f _ { t } ( S ) } \end{array}$ in polynomial time, which is different from the vanilla regret (Zinkevich 2003).

# Main Results

In this section, we first develop a delayed algorithm for bandit setting to establish a regret bound that is relevant to the average delay. As a by-product, we demonstrate that it can be slightly adjusted into the full-information setting to obtain a better regret bound. Finally, we present our blocking method to decouple the joint effect of the delays and bandit feedback, further enhancing the regret bound.

# Algorithm 1: DBGD-NF

# Require: Learning rate $\eta$

1: Initialize $\mathbf { x } _ { 1 } \in [ 0 , 1 ] ^ { n }$   
2: for $t = 1$ to $T$ do   
3: Let $\begin{array} { r } { 1 \ \geq \ \mathbf { x } _ { t , \pi _ { ( \frac { 1 } { s } ) } } \ \geq \ \mathbf { x } _ { t , \pi _ { ( 2 ) } } \ \geq \ \cdot \ \cdot \cdot \mathbf { x } _ { t , \pi _ { ( n ) _ { s } } } \ \geq \ 0 } \end{array}$ be the sorted entries in decreasing order with $A _ { t , i } \ =$ $\{ \pi _ { ( 1 ) } , \ldots , \pi _ { ( i ) } \}$ for all $i \in [ n ]$ and $A _ { t , 0 } = \varnothing$ . Define $\mathbf { x } _ { t , \pi _ { ( 0 ) } } = 1 , \mathbf { x } _ { t , \pi _ { ( n + 1 ) } } = 0$   
4: For 0 ≤ i ≤ n, calculate λt,i = xt,π(i) − xt,π(i+1)   
5: Sample $S _ { t }$ from the distribution defined in (6)   
6: Observe the loss $f _ { t } ( S _ { t } )$   
7: Calculate $\hat { f } _ { t , i }$ according to (7)   
8: Compute the estimated gradient $\hat { \bf g }$ by (8) and incur a delay $d _ { t } \geq 1$   
9: Receive the gradient set $\{ \hat { \bf g } _ { k } | k \in \mathcal { F } _ { t } \}$   
10: Update $\mathbf { x } _ { t }$ according to (9)

# Results Related to the Average Delay

Existing literature (Lin et al. 2022) on online nonsubmodular optimization with delayed feedback in the bandit setting adopts a pooling strategy (He´liou, Mertikopoulos, and Zhou 2020) to handle the arbitrary delays, which only uses the oldest available information in a gradient pool. Since the gradient may be delayed by $d$ rounds, its regret bound relies on the maximum delay $d$ . To mitigate the effect of delayed feedback, motivated by previous work (Quanrud and Khashabi 2015), we utilize all the gradients received in round $t$ to update the decision, rather than the oldest one. In the bandit setting, the online player only has access to the loss value. To deal with this issue, we employ the one-point estimator (Hazan and Kale 2012; Lin et al. 2022) to compute the unbiased gradient. Our method, DBGD-NF, is detailed in Algorithm 1. In each round $t$ , $S _ { t }$ is sampled from the distribution

$$
P \left( S _ { t } = A _ { t , i } \right) = ( 1 - \mu ) \lambda _ { t , i } + \frac \mu { n + 1 } ,
$$

where $\lambda _ { t , i } = \mathbf { x } _ { t , \pi _ { ( i ) } } - \mathbf { x } _ { t , \pi _ { ( i + \pm , 1 ) } }$ and $\mu \in ( 0 , 1 )$ is the exploration probability. Then we utilize the one-point estimator to derive the gradient; that is, we compute

$$
\hat { f } _ { t , i } = \frac { \mathrm { \bf ~ 1 } \left( S _ { t } = A _ { t , i } \right) } { \left( 1 - \mu \right) \lambda _ { t , i } + \frac { \mu } { n + 1 } } f _ { t } \left( S _ { t } \right) ,
$$

where $\mathbf { 1 } ( \cdot )$ is an indicator function, and calculate the unbiased gradient

$$
\begin{array} { r } { \hat { \bf g } _ { t , \pi ( i ) } = \hat { f } _ { t , i } - \hat { f } _ { t , i - 1 } . } \end{array}
$$

Notably, we do not assume that the information is immediately available. Instead, owing to incurring a delay $d _ { t }$ , we only receive the information at the end of the round $t + d _ { t } - 1$ , and we only present the calculation of the gradient for simplicity. Then we use all the available estimated gradients to update the decision

$$
\mathbf { x } _ { t + 1 } = \Pi _ { [ 0 , 1 ] ^ { n } } \left[ \mathbf { x } _ { t } - \sum _ { k \in \mathcal { F } _ { t } } \hat { \mathbf { g } } _ { k } \right]
$$

Next, we establish the regret bound of Algorithm 1.

# Algorithm 2: DOGD-NF

# Require: Learning rate $\eta$

1: Initialize $\mathbf { x } _ { 1 } \in [ 0 , 1 ] ^ { n }$ 2: for $t = 1$ to $T$ do 3: Let $1 \ \geq \ \mathbf { x } _ { t , \pi _ { ( \frac { 1 } { t } ) } } \ \geq \ \mathbf { x } _ { t , \pi _ { ( 2 ) } } \ \geq \ \cdot \ \cdot \cdot \mathbf { x } _ { t , \pi _ { ( n ) _ { . } } } \ \geq \ 0$ be the sorted entries in decreasing order with $A _ { t , i } \ =$ $\{ \pi _ { ( 1 ) } , \ldots , \pi _ { ( i ) } \}$ for all $i \in [ n ]$ and $A _ { t , 0 } = \varnothing$ . Define $\mathbf { x } _ { t , \pi _ { ( 0 ) } } = 1 , \mathbf { x } _ { t , \pi _ { ( n + 1 ) } } = 0$ 4: For 0 ≤ i ≤ n, calculate λt,i = xt,π(i) − xt,π(i+1) 5: Sample $S _ { t }$ from the distribution defined in (10) 6: Observe the loss $f _ { t } ( S _ { t } )$ 7: Compute the estimated gradient $\mathbf { g } _ { t }$ by (11) and incur a delay $d _ { t } \geq 1$ 8: Receive the gradient set $\{ \mathbf { g } _ { k } | k \in \mathcal { F } _ { t } \}$ 9: Update $\mathbf { x } _ { t }$ according to (12) 10: end for

Theorem 1 Under Assumption $\jmath$ and Assumption 2, by setting $\begin{array} { r } { \mu = \frac { n \bar { d } ^ { 1 / 3 } } { T ^ { 1 / 3 } } , \eta = \frac { 1 } { L \bar { d } ^ { 1 / 3 } T ^ { 2 / 3 } } } \end{array}$ Ld¯1/31T 2/3 , DBGD-NF ensures

$$
\begin{array} { r } { \mathbb { E } \left[ { \pmb { R e g } } _ { \alpha , \beta } ( T ) \right] \leq \mathcal { O } \left( n \bar { d } ^ { 1 / 3 } T ^ { 2 / 3 } \right) , } \end{array}
$$

where n is the dimensionality and $\begin{array} { r } { \bar { d } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } d _ { t } } \end{array}$ is the average delay.

Remark 2 Compared to the existing $\mathcal { O } ( n d ^ { 1 / 3 } T ^ { 2 / 3 } )$ regret bound, DBGD-NF reduces the effect of delay and achieves a better $\mathcal { O } ( n \bar { d } ^ { 1 / 3 } T ^ { 2 / 3 } )$ regret bound. It is worth noting that DBGD-NF requires the prior knowledge of the average delay to set the learning rate. Quanrud and Khashabi (2015) also encounter this issue and introduce a simple solution by utilizing the doubling trick (Cesa-Bianchi et al. 1997) to adaptively adjust the learning rate, thereby overcoming this limitation, which we can also employ to attain an equivalent $\mathcal { O } \left( n \bar { d } ^ { 1 / 3 } T ^ { 2 / 3 } \right)$ bound.

Additionally, we also observe that DBGD-NF can be slightly adjusted to the full-information setting to derive a regret bound that relies on the average delay. Our modified method, DOGD-NF, is summarized in Algorithm 2. In each round $t$ , we sample $S _ { t }$ from the distribution

$$
P ( S _ { t } = A _ { t , i } ) = \lambda _ { t , i } ,
$$

where $A _ { t , i } = \{ \pi _ { ( 1 ) } , \ldots , \pi _ { ( i ) } \} , A _ { t , 0 } = \emptyset$ and $\lambda _ { t , i } = \mathbf { x } _ { t , \pi _ { ( i ) } } -$ $\mathbf { x } _ { t , \pi _ { ( i + 1 ) } } .$ Similar to DOAGD (Lin et al. 2022), we also employ the convex relaxation based on the Lova´sz extension to compute the approximate subgradient

$$
\begin{array} { r } { \mathbf { g } _ { t , \pi ( i ) } = f _ { t } ( A _ { t , i } ) - f _ { t } ( A _ { t , i - 1 } ) . } \end{array}
$$

Finally, we use all the available gradients to perform a gradient descent step

$$
\mathbf { x } _ { t + 1 } = \Pi _ { [ 0 , 1 ] ^ { n } } \left[ \mathbf { x } _ { t } - \sum _ { k \in \mathcal { F } _ { t } } \mathbf { g } _ { k } \right] .
$$

Then we present the theoretical guarantee of Algorithm 2.

Theorem 2 Under Assumption 1 and Assumption 2, by set  
ting $\begin{array} { r } { \eta = \frac { \sqrt { n } } { L \sqrt { d T } } } \end{array}$ , DOGD-NF ensures ,

$$
\mathbb { E } \left[ R e g _ { \alpha , \beta } ( T ) \right] \leq \mathcal { O } ( \sqrt { n \bar { d } T } ) .
$$

To better showcase the improvement of our results, we provide an example that clearly demonstrates the enhancement through the better exponents of $T$ .

Example 1 Consider a situation: $d _ { 1 : T }$ satisfy $\begin{array} { r } { d _ { 1 : \sqrt { T } } = \frac { T } { 2 } } \end{array}$ and $d _ { \sqrt { T } + 1 : T } = 1$ . Our methods achieve $\mathcal { O } ( \sqrt { n } T ^ { 3 / 4 } )$ and $\mathcal { O } ( n T ^ { 5 / 6 } )$ regret bounds, while the bounds of DOAGD and DBAGD are $\bar { \mathcal { O } } ( \sqrt { n } T )$ and $\mathcal { O } ( n T )$ , respectively.

# Decoupled Result

While BOGD-NF achieves a SOTA regret bound with respect to the average delay $\bar { d }$ , its dependence on the maximum delay $d$ remains suboptimal. In the following, we introduce a modification of BOGD-NF aimed at improving the regret dependence on the maximum delay. Inspired by Wan et al. (2024), we combine DBGD-NF with a blocking update technique (Zhang et al. 2019; Garber and Kretzu 2020; Wang et al. 2023, 2024) to reduce the effect of delays. The essential idea is dividing total $T$ rounds into blocks with size $K$ and choosing the decision $S _ { t }$ from the same mixture distribution per block, so that we can reduce the bound of the estimated gradients suffered within each block. It is not hard to verify that the bound on the sum of gradients within a block is

$$
\mathbb { E } \left[ \left. \sum _ { t = m K + 1 } ^ { ( m + 1 ) K } \hat { \mathbf { g } } _ { t } \right. \right] \leq \mathcal { O } \left( n \left( \sqrt { K / \mu } + K \right) \right) .
$$

By choosing a proper block size $K = \Theta ( 1 / \mu )$ , this bound can be enhanced to $\mathcal { O } \left( n K \right)$ , which is smaller than the $\mathcal { O } ( n K / \sqrt { \mu } )$ bound of BDAGD. Based on this blocking update mechanism, we reduce the effect of delays on the regret from $\mathcal { O } ( n \eta d T \sqrt { \mu } )$ to $\mathcal { O } ( n \eta d T )$ , so that we decouple the effect of the delays and gradient estimator.

Our designed algorithm, BDBGD-NF, is presented in Algorithm 3. We first simply initialize $\mathbf { x } _ { 1 } , \mathbf { y } _ { 1 }$ to be any point in $[ 0 , 1 ] ^ { n }$ and available gradient set for the each block $\bar { \mathcal { P } } _ { i } = \emptyset$ . In each block $m$ , we sample the decision set $S _ { t }$ from the same mixture distribution (6) related to ${ \bf y } _ { m }$ and employ the one-point gradient estimator to derive the gradient.

At the end of each block $m$ , we will identify the block where all the queries are obtained and use the all gradients from available blocks to update ym

$$
\mathbf { y } _ { m + 1 } = \Pi _ { [ 0 , 1 ] ^ { n } } \left[ \mathbf { y } _ { m } - \sum _ { i \in A _ { m } } \sum _ { \hat { \mathbf { g } } _ { k } \in \mathcal { P } _ { i } } \hat { \mathbf { g } } _ { k } \right] .
$$

In the following, we present the following theorem to establish the theoretical guarantee for BDBGD-NF.

Theorem 3 Under Assumption $\jmath$ and Assumption 2, by setting K = T 13 , µ = nT −1/3, η = min{ LT 12/3 , L√1dT } BDBGD-NF ensures

$$
\mathbb { E } \left[ { \pmb { R e g } } _ { \alpha , \beta } ( T ) \right] \leq { \mathcal { O } } \left( n ( T ^ { \frac { 2 } { 3 } } + \sqrt { d T } ) \right) .
$$

# Algorithm 3: BDBGD-NF

# Require: $\eta , \mu \in ( 0 , 1 ) , K$

1: Initialize point $\mathbf { x } _ { 1 } \in [ 0 , 1 ] ^ { n }$ , set $\mathbf { y } _ { 1 } = \mathbf { x } _ { 1 }$ and set the gradient pool for each block $\mathcal { P } _ { i } = \emptyset , i = 1 , . . . , \lceil T / K \rceil$ 2: for $m = 1$ to $\lceil T / K \rceil$ do 3: Set block pool $A _ { m } = \varnothing$ 4: for time step $t = ( m - 1 ) K + 1$ to $\operatorname* { m i n } \{ m K , T \}$ do 5: Choose xt = ym 6: Let $1 \geq \mathbf { x } _ { t , \pi _ { ( \frac { 1 } { \cdot } ) } } \geq \mathbf { x } _ { t , \pi _ { ( 2 ) } } \geq . . . \mathbf { x } _ { t , \pi _ { ( n ) , } } \geq 0$ be the sorted entries in decreasing order with $A _ { t , i } =$ $\{ \pi _ { ( 1 ) } , \ldots , \pi _ { ( i ) } \}$ for all $i \in [ n ]$ and $A _ { t , 0 } = \emptyset$ . Define $\mathbf { x } _ { t , \pi _ { ( 0 ) } } = 1 , \mathbf { x } _ { t , \pi _ { ( n + 1 ) } } = 0$ 7: For 0 ≤ i ≤ n, calculate λt,i = xt,π(i) − xt,π(i+1) 8: Sample $S _ { t }$ from the distribution defined in (6) 9: Observe the loss $f _ { t } ( S _ { t } )$ 10: Calculate $\hat { f } _ { t , i }$ according to (7) 11: Compute the estimated gradient $\hat { \bf g } _ { t }$ by (8) and incur a delay $d _ { t } \geq 1$ 12: Receive the set $\{ \hat { \bf g } _ { k } | k \in \mathcal { F } _ { t } \}$ and update each gradient to its gradient pool $\mathcal { P } _ { j } = \mathcal { P } _ { j } \bigcup \{ \hat { \bf g } _ { k } \}$ , where $j = \lceil k / K \rceil$ is the block that $\hat { \bf g } _ { k }$ beloSngs to 13: end for 14: If $| \mathcal { P } _ { i } | = K$ , $\mathcal { A } _ { m } = \mathcal { A } _ { m } \cup \{ i \}$ , which denotes the index of the block where all the queries arrive. 15: Perform gradient descent to $\mathbf { y } _ { m + 1 }$ according to (13) 16: For $i \in \mathcal { A } _ { m }$ , set $\mathcal { P } _ { i } = \emptyset$ 17: end for

Remark 3 When $d = \mathcal { O } ( T ^ { 1 / 3 } )$ , the regret bound of our method matches the previous $\mathcal { O } ( n T ^ { 2 / 3 } )$ regret bound (Lin et al. 2022) in the bandit setting without delayed feedback. Otherwise, this regret bound is also on the same order in terms of $d$ and $T$ with the $\mathcal { O } ( \sqrt { n \bar { d } T } )$ regret in the fullinformation setting with delayed feedback in the worst case, where $\bar { d } \ = \ \Theta ( d ) \breve { }$ . Moreover, it is better than the former $\mathcal { O } \left( n \bar { d } ^ { 1 / 3 } T ^ { 2 / 3 } \right)$ bound when $d = o ( \bar { d } ^ { 2 / 3 } T ^ { 1 / 3 } )$ .

# Experiments

In this section, we evaluate the effectiveness of our proposed methods through numerical experiments. We compare our methods with the SOTA methods, DOAGD and DBAGD (Lin et al. 2022) on structured sparse learning with delayed feedback. When it comes to hyper-parameter tuning, we set $\begin{array} { r } { \eta = \frac { \sqrt { n } } { L \sqrt { d T } } } \end{array}$ for DOGD-NF, $\begin{array} { r } { \eta = \frac { 1 } { L \bar { d } ^ { 1 / 3 } T ^ { 2 / 3 } } } \end{array}$ and $\begin{array} { r } { \mu = \frac { q n \bar { d } ^ { 1 / 3 } } { T ^ { 1 / 3 } } } \end{array}$ for DBGD-NF and η = min{ LT 12/3 , L√1dT }, $\begin{array} { r } { \mu = \frac { q n } { T ^ { 1 / 3 } } } \end{array}$ T q1n/3 and K = T 1/3 for BDBGD-NF. As for other methods, we choose $\begin{array} { r } { \eta \ = \ \frac { \sqrt { n } } { L \sqrt { d T } } } \end{array}$ for DOAGD, and $\eta \ =$ $\frac { 1 } { L d ^ { 1 / 3 } T ^ { 2 / 3 } }$ $\textstyle \mu = { \frac { q n d ^ { 1 / 3 } } { T ^ { 1 / 3 } } }$   
to Theorem 5.2 and Theorem 5.4 in Lin et al. (2022). Moreover, we perform a grid search to select the parameter $q$ from the set $\{ 0 . 0 1 , 0 . 1 , 1 \}$ . All the experiments are conducted in Python 3.7 with two $3 . 1 \ : \mathrm { G H z }$ Intel Xeon Gold 6346 CPUs and 32GB memory.

Setup. We conduct experiments on structured sparse learning defined in (4). Following the setup in Lin et al. (2022), we choose $F _ { t } ( S ) = \mathrm { m a x } ( S ) - \mathrm { m i n } ( S ) + 1$ for all $\textit { S } \neq \emptyset$ and $F ( \varnothing ) \ = \ 0$ . We consider a simple linear regression problem, where the sparse optimal solution $\mathbf { x } ^ { * } \in \mathbb { R } ^ { n }$ only consists $k$ consecutive 1, with the remaining positions being 0. We compute $\mathbf { y } _ { t } = A _ { t } \mathbf { x } ^ { * } + \boldsymbol { \epsilon } _ { t }$ , where each row of $A _ { t } \in \mathbb { R } ^ { s \times n }$ is a vector i.i.d. sampled from a Gaussian distribution and $\boldsymbol { \epsilon } _ { t } \in \mathbb { R } ^ { s }$ is a Gaussian noise vector with standard deviation 0.1. We choose the square loss $\ell _ { t } ( { \mathbf { x } } ) = \| { A } _ { t } { \mathbf { x } } - { \mathbf { y } } _ { t } \| ^ { 2 } / 2$ and random delays in our experiments, i.e., $1 \leq d _ { t } \leq d$ , where $d$ is the max delay. For all experiments, we set rounds $T = 8 0 0 0$ such that the block $K$ is 20, dimension $n = 1 0$ , number of samples $s = 1 2 8$ , trade-off parameter $\gamma = 0 . 1$ and sparse parameter $k = 2$ .

![](images/28bc5f43f5984749d7967ab2fddb9e1e6a6ad0695b03147b232f796245927113.jpg)  
Figure 1: Results for full-information.

![](images/088f485cc099bd874d62954e768746e6db92797a15f7989937570761f97430e3.jpg)  
Figure 2: Results for bandit feedback.

Results. We report the regret against the number of rounds in full-information and bandit settings under different maximum delays $d$ within the set $\{ 1 0 , 2 0 , 5 0 0 \}$ in Figure 1 and Figure 2, respectively. For the delays $d _ { t }$ , we sample them uniformly at random from the range $[ 1 , d ]$ . We observe that DOGD-NF suffers less loss compared to DOAGD under full-information with delayed feedback because it utilizes all available gradients rather than just the oldest one. When the maximum delay increases, the superiority becomes more pronounced due to the large gap between the average delay $\bar { d }$ and maximum delay $d$ . As evident from Figure 2, our DBGD-NF also experiences less loss, which is consistent with our theories. Additionally, BDBGD-NF obtains the best performance and consistently yields lowest regret as the delay changes in the bandit setting, which aligns with its theoretical guarantee.

# Conclusion and Future Work

In this paper, we study the online nonsubmodular optimization with delayed feedback in the bandit setting and develop several algorithms to improve the existing regret bound. Firstly, our BDGD-NF and DOGD-NF achieve better $\mathcal { O } ( n \bar { d } ^ { 1 / 3 } T ^ { 2 / 3 } )$ and $\mathcal { O } ( \sqrt { n \bar { d } T } )$ regret bounds for bandit and full-information settings, respectively, which are relevant to the average delay. Furthermore, to decouple the joint effect of delays and bandit feedback, we combine BDGDNF with a blocking update technique. Our BDBGD-NF obtains a superior $\mathcal { O } ( n ( T ^ { 2 / 3 } + \sqrt { d T } ) )$ regret bound. Finally, the experimental results also demonstrate the effectiveness of our methods.

One might notice that the regret bound of BDBGD-NF depends on the maximum delay rather than the average delay, unlike the former results. In the future, we will investigate how to develop a decoupled algorithm to obtain the regret bound that relies on the average delay. Moreover, we will try to deal with the online nonsubmodular optimization in the non-stationary environments.

# Acknowledgments

This work was partially supported by NSFC (U23A20382, 62361146852), the Collaborative Innovation Center of

Novel Software Technology and Industrialization, and the Open Research Fund of the State Key Laboratory of Blockchain and Data Security, Zhejiang University. The authors would like to thank the anonymous reviewers for their constructive suggestions.