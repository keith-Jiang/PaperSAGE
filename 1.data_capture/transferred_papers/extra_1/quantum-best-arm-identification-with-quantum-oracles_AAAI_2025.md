# Best Arm Identification with Quantum Oracles

Xuchuang Wang1, Yu-Zhen Janice Chen1, Matheus Guedes de Andrade1, Jonathan Allcock2, Mohammad Hajiesmaili1, John C.S. Lui3, and Don Towsley

1College of Information and Computer Sciences, University of Massachusetts, Amherst, Massachusetts, USA 2Tencent Quantum Laboratory, Tencent, Hong Kong 3Department of Computer Science and Engineering, The Chinese University of Hong Kong, Hong Kong {xuchuangwang, yuzhenchen, mguedesdeand, hajiesmaili, towsley} $@$ cs.umass.edu, jonallcock@tencent.com, cslui@cse.cuhk.edu.hk

# Abstract

Best arm identification (BAI) is a key problem in stochastic multi-armed bandits, where $K$ arms each has an associated reward distribution, and the objective is to minimize the number of queries needed to identify the best arm with high confidence. In this paper, we explore BAI using quantum oracles. For the case where each query probes only one arm $( m = 1 )$ ), we devise a quantum algorithm with a query complexity upper bound of $\tilde { O } ( K \Delta ^ { - 1 } \log ( 1 / \delta ) )$ , where $\delta$ is the confidence parameter and $\Delta$ is the reward gap between best and second best arms. This improves on the classical bound by a factor of $\Delta ^ { - 1 }$ . For the general case where a single query can probe $m$ arms $( 1 \leqslant m \leqslant K )$ simultaneously, we propose an algorithm with an upper bound of $\tilde { O } ( ( K / \sqrt { m } ) \Delta ^ { - \mathrm { 1 } } \log ( 1 / \delta ) )$ , improving by a factor of $\sqrt { m }$ compared to the $m = 1$ case. We also provide query complexity lower bounds for both scenarios, which match the upper bounds up to logarithmic factors, and validate our theoretical results with Qiskit-based simulations.

# 1 Introduction

Best arm identification (BAI) is a fundamental problem in the bandits and online learning communities (Audibert, Bubeck, and Munos 2010; Bubeck, Munos, and Stoltz 2011; Mannor and Tsitsiklis 2004). Given $K ~ \in ~ \mathbb { N } ^ { + }$ arms, each arm $k$ is associated with a reward distribution with unknown mean $\mu _ { k }$ , and the goal of BAI is to identify the arm with the largest mean reward, with a confidence of $1 - \delta$ , using as few queries as possible. The number of queries required is called the query complexity. Each query, in the classical setting, corresponds to the learner pulling (sampling) one arm and observing a reward drawn from the arm’s reward distribution. As the learner only cares about finding the best arm, the BAI problem is a pure exploration problem. BAI has many real world applications, such as, clinical trials (Robbins 1952), network routing (Barrachina-Mun˜oz and Bellalta 2017), and crowdsourcing (Zhou, Chen, and Li 2014).

Recent progress in building quantum computers (Arute et al. 2019; Chow, Dial, and Gambetta 2021) and quantum networks (Wehner, Elkouss, and Hanson 2018; Azuma et al. 2022) has been encouraging, and wide applications of quantum systems are envisaged in the near future. In these quantum systems, BAI problems also emerge. For example, a quantum network may contain multiple channels between source and destination nodes. Among these channels, one may want to determine the “best” one, where “best” may refer, for example, to the channel with the highest fidelity (Liu et al. 2024) or with the lowest noise (Li, Deng, and Zhou 2008). Another example is in distributed quantum computing (Cacciapuoti et al. 2019), where different quantum computers may have different performances when applied to the same problem, and one wants to identify the quantum computer that provides the best performance for a given task. Although one can still apply classical BAI algorithms to address these problems, we aim to show that the quantum information feedback from these quantum systems can be leveraged to improve the learning efficiency.

In this paper, we study the BAI problem in quantum systems, where the learner can query the arms using quantum queries. More specifically, we study two key advantages of quantum feedback in BAI: (1) quantum parallelism (Chuang and Yamamoto 1995), and (2) quantum entanglement (Einstein, Podolsky, and Rosen 1935). The quantum Monte Carlo estimator (parallelism, Lemma 1) provides a more efficient estimator for the learner to estimate arm rewards. Additionally, multi-qubit oracles with entangled quantum superposition inputs enable the learner to query multiple arms simultaneously (coherently) within a single query. We model the former advantage by weak quantum oracles, one for each arm, and the latter by a constrained quantum oracle which can query several arms coherently (both detailed in Section 2.2). When a constrained oracle can query all arms coherently, we call it a strong quantum oracle.

The development of effective algorithms for both oracles necessitates the use of quantum computing to manage quantum information feedback and leverage quantum parallelism and entanglement. However, obtaining a valid output from a quantum computing subroutine, such as amplitude amplification (Brassard et al. 2002), typically demands multiple consecutive queries on the same arm or a subset of arms for the constrained oracle. This characteristic renders the classical BAI algorithm design and analysis ineffective for BAI with quantum oracles. Consequently, it is imperative to contemplate new algorithm designs and analyses for BAI with quantum oracles. On the other hand, to investigate the fundamental limit of quantum BAI problems, we need to establish query complexity lower bounds. However, given that quantum information (including parallelism and entanglement) offers more informative and inherently different query feedback than classical BAI, the classical proofs of BAI complexity lower bound are not applicable. Instead, one needs to adapt quantum computation and quantum information approaches to examine quantum BAI problems. Additionally, to empirically validate the performance of devised algorithms for BAI with quantum oracles, one has to utilize quantum circuits (Nielsen and Chuang 2002) and implement the necessary quantum computation subroutines using basic quantum logic gates.

Table 1: Comparison of query complexity bounds with classical and quantum BAI   

<html><body><table><tr><td>Oracle</td><td>Lower Bound</td><td>Upper Bound</td></tr><tr><td>Classical (2) Strong quantum (5)</td><td>Ω (∑klog ） (Mannor and Tsitsiklis 2004) (Wang et al. 2021)</td><td>O (∑k  log $） (Karnin,Koren,and Somekh 2013) 6</td></tr><tr><td>Weak quantum (3)</td><td>Ω Ω</td><td>√10g（ (Wang et al. 2021)</td></tr><tr><td>Constrained quantum (4)</td><td>(∑k  10g( Ω ∑s∈B1 £kEs</td><td>0 (∑  log( 0(∑seB√Ek∈s1og())</td></tr></table></body></html>

We summarize the key contributions of this paper as follows:

• For BAI with the weak quantum oracle, we derive a query complexity lower bound $\Omega \left( \sum _ { k } ( 1 / \Delta _ { k } ) \log ( 1 / \delta ) \right)$ , showing that no quantum algorithm can achieve a smaller query complexity. Then, we propose an eliminationbased quantum algorithm $( \mathsf { Q } - \mathsf { E } \mathtt { l i m } )$ and derive its query complexity upper bound $\tilde { O }$ $\tilde { \mathcal { O } } \left( \sum _ { k } ( 1 / \Delta _ { k } ) \log ( 1 / \delta ) \right)$ , where the suboptimality gap $\Delta _ { k } \ : = \ \mu _ { 1 } - \mu _ { k }$ is the difference in the mean rewards of the optimal arm and arm $k$ , and ${ \tilde { O } } ( \cdot )$ hides poly-logarithmic factors. This implies that $\mathsf { Q }$ -Elim is near-optimal up to logarithmic factors for BAI with the weak quantum oracle (Section 3).   
• For BAI with the $m$ -constrained quantum oracle, we propose a partition-based quantum algorithm $\left( \scriptstyle \ Q - \ P \ a x \ t \right)$ , and derive its query complexity upper bound $\begin{array} { r } { \tilde { O } ( \sum _ { S \in \mathfrak { B } } \sqrt { \sum _ { k \in S } 1 / \Delta _ { k } ^ { 2 } } \log \left( 1 / \delta \right) ) } \end{array}$ , where $\mathfrak { B }$ is a partition of the full arm set, i.e., a set of arm subsets, each subset $s$ containing $m$ arms. We also derive a query complexity lower bound of $\Omega ( \sum _ { S \in \mathfrak { B } } \sqrt { \sum _ { k \in S } 1 / \Delta _ { k } ^ { 2 } } )$ for the partition algorithm class, which matches the upper bound of $\mathsf { Q }$ -Part up to logarithmic factors (Section 4).   
• We implement our quantum algorithms using the IBM Qiskit (Qiskit contributors 2023). We first corroborate the superiority of our quantum algorithms over classical BAI algorithms. We then evaluate our algorithms under simulated quantum noise (Section 5).   
Related Works Prior works on multi-armed bandits

(MAB) typically focus on regret minimization and BAI. This paper focuses on the BAI setting (Even-Dar, Mannor, and Mansour 2002; Even-Dar et al. 2006; Mannor and Tsitsiklis 2004). The BAI setting can be divided into two categories: (1) BAI with fixed confidence—find the best arm with a confidence of at least $1 - \delta ( \delta \in ( 0 , 1 ) )$ using as few samples as possible (Bubeck, Munos, and Stoltz 2011); and (2) BAI with fixed budget—given a fixed budget of $\boldsymbol { Q }$ queries, find the best arm with as high a probability as possible (Karnin, Koren, and Somekh 2013). In this paper, we focus on the former category which, for brevity, we will refer to simply as the BAI problem. BAI with the strong quantum oracle was first studied by Casale´ et al. (2020); Wang et al. (2021), where they proposed a near-optimal quantum algorithm that enjoys a quadratic speedup in query complexity. We are the first to study the BAI problem with the weak and constrained quantum oracles. Besides BAI, another objective, regret minimization in bandit theory, has also been studied with quantum oracles, including Wan et al. (2023); Dai et al. (2023); Wu et al. (2023), etc. We defer a more detailed discussion of these works and other loosely related works to Appendix A.

In Table 1, we summarize the key results in this paper and compare them to prior works. Comparing the $\Delta _ { k }$ dependence of the complexities, we have

$$
\sqrt { \sum _ { k } \frac { 1 } { \Delta _ { k } ^ { 2 } } } \leqslant \sum _ { S \in \mathfrak { B } } \sqrt { \sum _ { k \in S } \frac { 1 } { \Delta _ { k } ^ { 2 } } } \leqslant \sum _ { k } \frac { 1 } { \Delta _ { k } } \leqslant \sum _ { k } \frac { 1 } { \Delta _ { k } ^ { 2 } }
$$

All BAI problems with quantum oracles enjoy smaller query complexities than the classical one. The query complexity of the weak quantum oracle is worst among quantum oracles, which is due to the fact that the weak oracle cannot exploit quantum entanglement to probe multiple arms in parallel. The query complexity of the $m$ -constrained quantum oracle lies between that of strong and weak oracles, and when $m =$ 1 (resp., $m = K )$ ) the complexity coincides with that of weak (resp., strong) oracles.

# 2 Model

# 2.1 Preliminaries

Best arm identification (BAI). Consider a multi-armed bandit (MAB) consisting of $K$ arms, where each arm $k \in$ $\mathcal { K } : = \{ 1 , 2 , \dots , K \}$ is associated with a Bernoulli distribution $\mathcal { B } ( \mu _ { k } )$ with mean $\mu _ { k } ~ \in ~ ( 0 , 1 )$ .1 An MAB instance is determined by the mean rewards of its arms, and we denote an instance $\boldsymbol { \tau }$ with means $\mu _ { 1 } , \dotsc , \mu _ { K }$ as ${ \cal J } : = \{ \mu _ { 1 } , \ldots , \mu _ { K } \}$ .

For simplicity, we assume the $K$ arms are labeled in descending order of their means: $\mu _ { 1 } > \mu _ { 2 } \geqslant . . . \geqslant \mu _ { K }$ , unknown to the learner, and denote the mean reward (suboptimality) gap as $\Delta _ { k } : = \mu _ { 1 } - \mu _ { k }$ for suboptimal arms $k > 1$ and $\Delta _ { 1 } : = \Delta _ { 2 }$ for the optimal arm. We assume a unique optimal arm for the simplicity of the later presentation of the algorithms and analysis. One could extend the results to multiple optimal arms with techniques in bandits literature, e.g., find an $\epsilon$ - optimal arm (Even-Dar, Mannor, and Mansour 2002). Then, given confidence parameter $\delta \in ( 0 , 1 )$ , the best arm identification (BAI) problem is to correctly output the best arm with a probability of at least $1 - \delta$ using as few queries as possible, noted as the query complexity $\boldsymbol { Q }$ .

Next, we present some basics notation from quantum computation and information (Nielsen and Chuang 2002).

Bra-ket notation. We make use of bra-ket notation to represent quantum states, where the “ket” $\begin{array} { r l } { | x \rangle } & { { } : = } \end{array}$ $( x _ { 1 } , x _ { 2 } , \ldots , x _ { n } ) ^ { \widehat { T } } \in \mathbb { C } ^ { n }$ denotes a column vector of $n$ complex numbers, while the “bra” $\langle x | : = | x \rangle ^ { \dagger } = ( x _ { 1 } ^ { * } , x _ { 2 } ^ { * } , \ldots , x _ { n } ^ { * } )$ , a row vector, is the conjugate transpose of $\vert x \rangle$ . For two quantum states $\left| x \right. , \left| y \right. \in \mathbb { C } ^ { n }$ , their inner product is denoted as $\textstyle \langle x | y \rangle : = \sum _ { i = 1 } ^ { n } x _ { i } ^ { * } y _ { i } \ \in \ \mathbb { C }$ , and given another quantum state $| z \rangle \in \mathbb { C } ^ { m }$ the tensor product between $| x \rangle$ and $| z \rangle$ is denoted as $\left| x \right. \left| z \right. = \left| x \right. \otimes \left| z \right. : = \left( x _ { 1 } z _ { 1 } , x _ { 1 } z _ { 2 } , \ldots , x _ { n } z _ { m } \right) \in { \mathbb { C } } ^ { n } \otimes { \mathbb { C } } ^ { m } .$

Qubit. A “qubit” is a two-level quantum system $| \phi \rangle ~ =$ $\mathsf { \Gamma } ( \alpha , \beta ) \ \in \ \mathbb { C } ^ { 2 } ;$ , often written as $\left| \underline { { \phi } } \right. = \alpha \left| 0 \right. + \beta \left| 1 \right.$ , where $| 0 \rangle \ : = \ : ( 1 , 0 ) ^ { T }$ and $| 1 \rangle \ : = \ : ( 0 , 1 ) ^ { T }$ are two basis states, and $\alpha , \beta \in \mathbb { C }$ are complex numbers, called amplitudes, satisfying $| \alpha | ^ { 2 } + | \beta | ^ { 2 } = 1$ . A measurement of the qubit in the $\left\{ \left| 0 \right. , \left| 1 \right. \right\}$ basis will give a $\cdot _ { 0 } ,$ with probability $| { \boldsymbol { \alpha } } | ^ { 2 }$ and a ‘1’ with probability $| \beta | ^ { 2 }$ .

Quantum query model. In the quantum query model, one has access to a black-box unitary operator (i.e., oracle) which implements a given transformation. The objective is to study the query complexity, i.e., the number of calls $\boldsymbol { Q }$ to the oracle needed to solve a given task; all other possible costs, e.g., gate complexity, are ignored. This is a commonly used model for studying quantum algorithms (Childs 2017, $\ S 2 0 _ { , }$ and can be used, for instance, to obtain algorithmic running time lower bounds (Klauck, Sˇpalek, and De Wolf 2007). In this paper, we study the query complexity of best arm identification with fixed confidence under weak and constrained quantum oracles.

# 2.2 Quantum Oracles

Before introducing the quantum oracles, we first recall the classical oracle for the BAI problem. That is, when querying an arm $k$ , one obtains a reward drawn from a Bernoulli distribution $\mathcal { B } ( \mu _ { k } )$ with unknown mean $\mu _ { k }$ , i.e.,

$$
X _ { k } \sim { \mathcal { B } } ( \mu _ { k } ) .
$$

We refer to (2) as the classical oracle.

In the quantum setting, the Bernoulli distributions can be mapped to oracles ${ \cal O } _ { \mathrm { w e a k } } ^ { ( k ) }$ (one for each $k$ ) that act as follows,

$$
O _ { \mathrm { w e a k } } ^ { ( k ) } : | 0 \rangle _ { R } \mapsto \sqrt { 1 - \mu _ { k } } | 0 \rangle _ { R } + \sqrt { \mu _ { k } } | 1 \rangle _ { R } ,
$$

where the register $| \cdot \rangle _ { R }$ represents a single-qubit “bandit reward” register with basis states $| 0 \rangle$ and 1 . The output qubit encodes the Bernoulli reward, meaning that if one measures the output in the basis $\{ \left| 0 \right. , \left| 1 \right. \}$ , the probability of observing $| 1 \rangle$ is $\mu _ { k }$ , while the probability of observing $| 0 \rangle$ is $1 - \mu _ { k }$ . We refer to (3) as the weak quantum oracle.

Note that directly measuring the output qubits reduces the weak oracle to a Bernoulli distribution. However, aside from direct measurement, the output qubits enable efficient quantum parallelism through quantum computing algorithms, which we elaborate in Section 3.

To harness the entanglement properties of quantum information in real-world quantum systems, we consider a more general quantum oracle that allows simultaneous querying of multiple arms. In addition to the reward register $| \cdot \rangle _ { R }$ , we introduce an “arm index” register $\left| \cdot \right. _ { I }$ , which has $K$ orthogonal basis states {|𝑘⟩𝐼 }𝑘𝐾=1, each corresponding to an arm. A quantum state in the ${ \dot { | \cdot \rangle } } _ { I }$ register can be expressed as $\begin{array} { r } { \sum _ { k = 1 } ^ { K } a _ { k } \left| k \right. _ { I } } \end{array}$ , where $\boldsymbol { a } _ { k } \in \mathbb { C }$ are the amplitudes of the arms, and normalization requires that $\begin{array} { r } { \sum _ { k = 1 } ^ { K } | a _ { k } | ^ { 2 } = 1 } \end{array}$ .

With the assistance of the arm index register, we define a constrained quantum oracle that outputs states entangling the arm index and reward registers. Assuming the oracle can access $m \in \{ 1 , 2 , \ldots , K \}$ arms simultaneously, for any subset of arms $s \subseteq \mathcal { K }$ with $| S | = m$ and $\begin{array} { r } { \sum _ { k \in \mathcal { S } } \mathbf { \dot { | } } a _ { k } \mathbf { | } ^ { 2 } = \mathbf { \dot { 1 } } } \end{array}$ , the oracle is defined as follows:

$$
\begin{array} { l } { { \displaystyle { \cal O } _ { \mathrm { c o n s } } ^ { ( S ) } : \sum _ { k \in { \cal S } } a _ { k } \left| k \right. _ { I } \left| 0 \right. _ { R } } } \\ { { \displaystyle ~ \mapsto \sum _ { k \in { \cal S } } a _ { k } \left| k \right. _ { I } \left( \sqrt { 1 - \mu _ { k } } \left| 0 \right. _ { R } + \sqrt { \mu _ { k } } \left| 1 \right. _ { R } \right) . } } \end{array}
$$

when $m = 1$ , the oracle reduces to the weak quantum oracle in (3), and when $m \ = K$ , it becomes the strong quantum oracles as follows,

$$
\begin{array} { l } { { \displaystyle O _ { \mathrm { s t r o } } : \sum _ { k = 1 } ^ { K } a _ { k } \left| k \right. _ { I } \left| 0 \right. _ { R } } } \\ { { \displaystyle \qquad \mapsto \sum _ { k = 1 } ^ { K } a _ { k } \left| k \right. _ { I } \left( \sqrt { 1 - \mu _ { k } } \left| 0 \right. _ { R } + \sqrt { \mu _ { k } } \left| 1 \right. _ { R } \right) . } } \end{array}
$$

The constrained quantum oracle in (4) is more powerful than the weak oracle in (3) because it can access multiple arms coherently in a single query, whereas the weak oracle only allows access to one arm at a time. In Section 4, we present a BAI algorithm using the $m$ -constrained oracle, which outperforms the weak oracle when $m > 1$ .

In practice, coherently querying a large number of channels may be technologically challenging, which motivates the general $m \ \leqslant \ K$ case. This limitation reflects a technology constraint where more options exist than can be accessed simultaneously. Such technological constraints may also affect, for example, access to quantum states stored in memory. In this case, a weak oracle would support individual calls to memory, while an $m$ -constrained oracle functions like a dynamically loadable quantum random access memory (QRAM, see Appendix B), capable of querying multiple entries at once.

# 3 BAI with Weak Quantum Oracle

In this section, we address the BAI problem using a weak quantum oracle as described in (3). Querying this oracle for arm $k$ yields the state $\sqrt { 1 - \mu _ { k } } \left| 0 \right. + \sqrt { \mu _ { k } } \left| 1 \right.$ . To estimate $\mu _ { k }$ efficiently, we use the following lemma:

Lemma 1 (Performance of QuEst, adapted from Montanaro (2015); Grinko et al. (2021)). For a weak quantum oracle ${ O } _ { \mathrm { w e a k } } ^ { ( k ) }$ in (3), there exists a constant $C _ { 1 } > 1$ and a quantum estimation algorithm QuEst(O (𝑘) , weak 𝜖 , 𝛿) that estimates $\mu _ { k }$ with precision $\epsilon$ and confidence $\delta$ (i.e., $\mathbb { P } ( | \hat { \mu } _ { k } - \mu _ { k } | \ \geqslant$ $\epsilon ) \leqslant \delta ,$ ), using at most $\frac { C _ { 1 } } { \epsilon } \log { \frac { 1 } { \delta } }$ queries.

This quantum estimator QuEst achieves a quadratic speedup over the classical estimators that require $\dot { O } ( ( 1 / \dot { \epsilon } ^ { 2 } ) \log ( 1 / \delta ) )$ queries. Unfortunately, QuEst lacks flexibility: it does not generate any information before the entire procedure has completed, unlike classical estimators that improve estimates incrementally during the samples arriving and allows for sample reuse.

To address this issue, we first use QuEst to develop a batch-based elimination algorithm for BAI with the weak quantum oracle in Section 3.1. We then establish an upper bound on the query complexity of this algorithm in Section 3.2. Finally, in Section 3.3, we present a lower bound for any BAI algorithm using a weak quantum oracle, highlighting the fundamental limits of the task.

# 3.1 Algorithm Design

Algorithm 1 presents a quantum elimination algorithm $\left( \mathrm { Q - E 1 i m } \right)$ for BAI. The core idea of the elimination process is to maintain a candidate arm set $c$ (initially set to the full arm set $\mathcal { K } )$ , gradually identify and remove suboptimal arms from $c$ as learning progresses, and terminate when $c$ contains only one arm, which is then declared as the optimal arm.

Although several classical elimination algorithms, such as successive elimination (Even-Dar et al. 2006), have been proposed for BAI using classical oracles, these cannot be directly adapted by simply replacing classical estimators with the quantum estimator from Lemma 1 due to the rigidity of the quantum estimator (one cannot acquire any information from QuEst before the entire procedure completed).

A significant challenge in designing our quantum algorithm is determining when to perform quantum estimation QuEst and arm elimination. We address this by proposing a batch-based exploration and elimination scheme, where $j ~ \in ~ \{ 1 , 2 , \dots \}$ denotes the batch number. In each batch, we query all remaining arms in the candidate arm set $c$ a number of times depending on the batch number $j$ (Line 2), conduct $\begin{array} { r l } & { \mathsf { Q u E s t } \left( \bar { O _ { \mathrm { w e a k } } ^ { ( k ) } } , \bar { 2 ^ { - j } } , \frac { \delta } { 2 ^ { j } | C | } \right) } \end{array}$ to estimate the mean rewards of arms in $c$ based on the queries from this batch (Line 3), and eliminate newly identified suboptimal arms (Line 5) at the end of the batch. As $j$ increases, we progressively increase both the number of queries (Line 6) and the estimation accuracy of QuEst (Lines 2 and 3).

Algorithm 1 Q-Elim: Quantum elimination algorithm for BAI with weak quantum oracle

Input: fixed confidence parameter $\delta$ and number of arms $K$   
Initialize: empirical mean $\hat { \mu } _ { k } \gets 0$ , candidate arm set $c \gets$ $\mathcal { K }$ , batch number $j  1$ 1: while $| C | > 1$ do   
2: Query each arm $k \in { \cal { C } }$ for $C _ { 1 } 2 ^ { j } \log \left( 2 ^ { j } | C | / \delta \right)$ times   
3: Run QuEst $\begin{array} { r } { \left( O _ { \mathrm { w e a k } } ^ { ( k ) } , 2 ^ { - j } , \frac { \delta } { 2 ^ { j } | C | } \right) } \end{array}$ for each arm $k$ in $c$ and update these arms’ estimates $\hat { \mu } _ { k }$   
4: 5: $\begin{array} { r l } & { \mu _ { \operatorname* { m a x } }  \operatorname* { m a x } _ { k \in { \mathcal { C } } } \mu _ { k } } \\ & { { \mathcal { C } }  { \mathcal { C } } \setminus \{ k \in { \mathcal { C } } : \hat { \mu } _ { k } + 2 \cdot 2 ^ { - j } \leqslant \hat { \mu } _ { \operatorname* { m a x } } \} } \end{array}$ ⊲ Arm elimination   
6: $j \gets j + 1$

# 3.2 Query Complexity Upper Bound for Elimination Algorithm

Theorem 1 (Query complexity upper bound of Algorithm 1). Given confidence parameter $\delta \in ( 0 , 1 )$ , the query complexity of $\mathsf { Q }$ -Elim is upper bounded as follows,

$$
Q \leqslant \sum _ { k \in \mathcal { K } } \log _ { 2 } \left( \frac { 4 } { \Delta _ { k } } \right) \frac { 1 6 C _ { 1 } } { \Delta _ { k } } \log \frac { K } { \delta } ,
$$

where log is the natural logarithm, and $\log _ { 2 }$ is the logarithmic function base 2.

Comparison with the query complexity lower bound in Theorem 2 shows that our upper bound in Theorem 1 is tight up to logarithmic factors. Compared to the classical oracle sample complexity upper bound of $\begin{array} { r } { O ( \sum _ { k \in \mathcal { K } } ( 1 / \Delta _ { k } ) ^ { 2 } \log ( 1 / \delta ) ) } \end{array}$ (Karnin, Koren, and Somekh 2013), the query complexity upper bound in Theorem 1 has a quadratic improvement in the dependence on $1 / \Delta _ { k }$ for each individual arm. In contrast, the strong quantum oracle sample complexity upper bound $\tilde { O } ( \sqrt { \sum _ { k } 1 / \Delta _ { k } ^ { 2 } } \log ( 1 / \delta ) )$ (Wang et al. 2021) achieves an overall quadratic speedup. That is, as the first inequality of (1) shows, the coefficient of the query complexity lower bound of the weak quantum oracle is larger than that of the strong oracle, and is, in the worst case, $\sqrt { K }$ times larger.

# 3.3 Lower Bounds for BAI with weak quantum oracle

Lastly, we present a query complexity lower bound for BAI with a weak quantum oracle. This lower bound describes the fundamental limits of the BAI task with a weak quantum oracle and is independent of the specific algorithm used.

Theorem 2 (Query complexity lower bound for best arm identification). Given a quantum multi-armed bandits instance ${ { J } _ { 0 } } = \{ \mu _ { 1 } , . . . , \mu _ { K } \}$ where $\mu _ { k } \in ( 0 , 1 / 2 )$ for all $k$ and $\mu _ { 1 } > \mu _ { 2 } \geqslant \mu _ { k }$ for any $k \neq 1$ , any algorithm that identifies the optimal arm with a given confidence $1 - \delta$ , $\delta \in ( 0 , 1 )$ requires $\boldsymbol { Q }$ queries to the weak quantum oracle, where

$$
Q \geqslant \sum _ { k \in \mathcal { K } } \frac { 1 } { 4 \Delta _ { k } } \log \frac { 1 } { 4 \delta } .
$$

Thus, to identify the best arm with confidence $1 - \delta$ , it is necessary to pull each arm $k$ at least $1 / ( 4 \Delta _ { k } ) \log 1 / ( 4 \delta )$ times. The proof of this lower bound consists of two steps: (1) apply the quantum hypothesis testing techniques to prove a lower bound for the task of two arm identification, and (2) extend the lower bound of the two-arm case to multiple arms via adapting the lower bound proof of the classical best arm identification. The detailed proof is presented in Appendix F.

First, Theorem 2 demonstrates that the query complexity of Q-Elim, as established in Theorem 1, is near-optimal (up to some logarithm factors). Compared to the classical oracle’s sample lower bound $\begin{array} { r } { \Omega \left( \sum _ { k \in \mathcal { K } } \frac { 1 } { \Delta _ { k } ^ { 2 } } \log \frac { 1 } { \delta } \right) } \end{array}$ (Mannor and Tsitsiklis 2004), our lower bound shows a linear dependence on $1 / \Delta _ { k }$ rather than quadratic. When compared to the strong quantum oracle’s sample complexity lower bound $\Omega \left( \sqrt { \sum _ { k } \frac { 1 } { \Delta _ { k } ^ { 2 } } } ( 1 - \sqrt { \delta ( 1 - \delta ) } ) \right)$ (Wang et al. 2021, Theorem 5), the weak oracle’s query complexity lower bound has a larger coefficient, which can be up to $\sqrt { K }$ times greater in the worst case. However, our lower bound improves on the dependence on $\delta$ , as $\log ( 1 / \delta )$ is significantly larger than $1 - \sqrt { \delta ( 1 - \delta ) }$ when $\delta$ is small.

# BAI with m-Constrained Quantum Oracle

In this section, we present a partition algorithm for BAI with the $m$ -constrained quantum oracle. We first present some key subroutines in Section 4.1 on quantum computing, and then present our algorithm in Section 4.2, followed by the algorithm’s query complexity upper bound in Section 4.3, as well as a lower bound for any partition algorithms in Section 4.4.

# 4.1 Key Quantum Subroutines

Variable-Time Algorithm Construction The variabletime algorithm of Ambainis (2010); Wang et al. (2021) can be used to transform an $m$ -constrained quantum oracle with a reward register $| \cdot \rangle _ { R }$ into an oracle (VTA) that outputs a state with a flag register $| \cdot \rangle _ { F }$ which distinguishes arms with large mean rewards from other arms. For an $m$ -constrained oracle $O _ { \mathrm { c o n s } } ^ { ( S ) }$ and a subset $s$ , VTA takes an interval $\boldsymbol { I } = \left[ a , b \right]$ with $0 < a < b < 1$ and a parameter $\alpha \in ( 0 , 1 )$ as inputs. It divides $s$ into three subsets: $S _ { \mathrm { r i g h t } } : = \{ k \in S : \mu _ { k } \ \geqslant$ $\begin{array} { r } { b - \frac { b - a } { 8 } \big \} } \end{array}$ (high rewards); $S _ { \mathrm { l e f t } } : = \{ k \in S : \mu _ { k } < b - \frac { b - a } { 2 } \}$ (low rewards); $S _ { \mathrm { m i d d l e } } : = S \ \backslash$ $( S _ { \mathrm { r i g h t } } \cup S _ { \mathrm { l e f t } } )$ (intermediate rewards). The output state is:

$$
\begin{array} { r l } & { \mathrm { V T } \mathbb { A } ( O _ { \mathrm { c o n s } } ^ { ( S ) } , S , I = [ a , b ] , \alpha ) : \frac { 1 } { \sqrt { m } } \displaystyle \sum _ { k \in S } | k \rangle _ { I } | 1 \rangle _ { F }  } \\ & { \frac { 1 } { \sqrt { m } } \displaystyle ( \displaystyle \sum _ { k \in S _ { \mathrm { r i g h t } } } | k \rangle _ { I } | 1 \rangle _ { F } + \displaystyle \sum _ { k \in S _ { \mathrm { l e f t } } } | k \rangle _ { I } | 0 \rangle _ { F } + \displaystyle \sum _ { k \in S _ { \mathrm { m i d d l e } } } | k \rangle _ { I } | \phi _ { k } \rangle _ { F } ) , } \end{array}
$$

where $| \cdot \rangle _ { F }$ indicates the subsets $S _ { \mathrm { r i g h t } }$ and $\boldsymbol { S _ { \mathrm { l e f t } } }$ with $| 1 \rangle _ { F }$ and $| 0 \rangle _ { F }$ respectively. Arms in $S _ { \mathrm { m i d d l e } }$ are represented by $\left| \phi _ { k } \right. _ { F }$ , with specific states depending on $\alpha$ and the MAB instance. The probability of observing $| 1 \rangle _ { F }$ is $p _ { \mathrm { g o o d } } : =$

$\begin{array} { r } { \frac { 1 } { m } \left( \left| S _ { \mathrm { r i g h t } } \right| + \sum _ { k \in S _ { \mathrm { m i d d l e } } } | \beta _ { k } | ^ { 2 } \right) } \end{array}$ , where $\beta _ { k }$ depends on $\left| \phi _ { k } \right. _ { F }$ .   
The algorithm’s pseudocode is in Appendix C.1.

Amplitude amplification (Amplify) and amplitude estimation (Estimate) The Amplify and Estimate are two fundamental quantum computing algorithms (Brassard et al. 2002). Amplify enhances the amplitude of a target basis state, while Estimate estimates the amplitude of that state. Since these algorithms are well-established, we omit their pseudocode and direct interested readers to Brassard et al. (2002) for details. In this work, we apply both algorithms with the VTA oracle in (6), using $| 1 \rangle _ { F }$ as the target state. The performance of Amplify and Estimate in this context is discussed in Lemma 5 in Appendix C.1.

Good Ratio (GoodRatio) Subroutine The good ratio subroutine is based on the variable-time algorithm (VTA) and amplitude estimation (Estimate). It takes an $m$ - constrained oracle $O _ { \mathrm { c o n s } } ^ { ( S ) }$ for a subset of arms $s$ , an interval $\boldsymbol { I } = [ a , b ]$ , and a confidence parameter $\delta$ as inputs, and outputs an estimate of the ratio of “good arms” in $s$ , where the “good arms” are the arms with mean reward greater than $a$ in the interval $I$ . The subroutine is detailed in Algorithm 4 in Appendix C.2. Lemma 2 provides the subroutine’s performance guarantees.

Lemma 2 (Performance of GoodRatio). Given an interval $\boldsymbol { I } ~ = ~ \left[ a , b \right]$ and a confidence parameter $0 ~ < ~ \delta ~ < ~ 1$ , there exists a GoodRati $\mathsf { o } ( O _ { c o n s } ^ { ( S ) } , S , I = [ a , b ] , \delta )$ subroutine which uses $O ( G )$ queries to output an estimate $\hat { p } _ { \mathrm { g o o d } }$ of the “good arm” ratio $p _ { \mathrm { g o o d } }$ such that

$$
0 . 9 \left( p _ { \mathrm { g o o d } } - { \frac { 0 . 1 } { m } } \right) < \hat { p } _ { \mathrm { g o o d } } < 1 . 1 \left( p _ { \mathrm { g o o d } } + { \frac { 0 . 1 } { m } } \right)
$$

$$
\begin{array} { r } { \sqrt { \frac { 1 } { ( b - a ) ^ { 2 } } + \frac { 1 } { \left| S _ { \mathrm { r i g h t } } \right| } \sum _ { k \in S _ { \mathrm { l e f t } } \cup S _ { \mathrm { m i d d e } } } \frac { 1 } { ( b - \mu _ { k } ) ^ { 2 } } } \mathrm { p o l y l o g } \left( \frac { m } { \delta ( b - a ) } \right) . } \end{array}
$$

Lemma 2 guarantees that GoodRatio provides a good estimate of the ratio of good arms in the subset $s$ with high probability and with in a reasonable number of queries.

Partition Shrink (PartShrink) Subroutine The partition shrink subroutine takes as input the $m$ -constrained oracles $O _ { \mathrm { c o n s } } ^ { ( S ) }$ for each subset $s$ in the partition set $\mathfrak { B }$ , the partition set $\mathfrak { B }$ itself, an interval $I$ , and parameters $h \in \{ 1 , 2 \}$ and $\delta \in ( 0 , 1 )$ . The parameter $h = 1$ (resp. $h = 2$ ) directs the algorithm to shrink the input interval $I$ so that the best arm $\mu _ { 1 }$ (resp. the second best arm $\mu _ { 2 }$ ) lies inside the output interval $J$ . Utilizing a technique from quantum ground state preparation (Lin and Tong 2020), PartShrink divides the input interval $I = [ a , b ]$ into five sub-intervals of equal length and outputs a new interval $J$ consisting of three consecutive subintervals, as illustrated below:

$$
\begin{array} { r l } & { \xrightarrow [ ] { \mathrm { I n p u t } \mathrm { I n t e r v a l } } \xrightarrow [ ] { I } \overbrace { \sum \mathrm { \Lambda } \mathrm { e } ^ { a + \epsilon } \mathrm { \Lambda } \overbrace { - \mathrm { \Lambda } \mathrm { e } ^ { a + 2 \epsilon } \mathrm { \Lambda } \mathrm { \Lambda } \mathrm { \Lambda } \mathrm { \Lambda } \mathrm { \Lambda } \mathrm { \Lambda } \mathrm { \Lambda } } ^ { a + \epsilon \mathrm { \Lambda } \mathrm { \Lambda } \mathrm { \Lambda } a + 3 \epsilon \mathrm { \Lambda } a + 4 \epsilon \mathrm { \Lambda } b } } ^ { a + \epsilon \mathrm { \Lambda } \mathrm { \Lambda } \mathrm { \Lambda } \mathrm { \Lambda } \mathrm { \Lambda } \mathrm { \Lambda } \mathrm { \Lambda } \mathrm { \Lambda } } } \\ &  \xrightarrow [ ] { \mathrm { O u t p u t } \mathrm { \Lambda } _ { \mathrm { \Lambda } } ^ { c a s e \mathrm { \Lambda } ( 0 , 0 ) } \overbrace { \left( \mathrm { \Lambda } , \mathrm { \Lambda } \right) \mathrm { \Lambda } \left( \mathrm { \Omega } , 1 \right) \mathrm { \Lambda } \left( \overbrace { \mathrm { \Lambda } \mathrm { \Lambda } \mathrm { e } ^ { a } \mathrm { \Lambda } \mathrm { \Lambda } \mathrm { \Lambda } \mathrm { \Lambda }  } ^ { \mathrm { O u t p u } \mathrm { \Lambda } \mathrm { \Lambda } } } } \en\right)d{array} \end{array}
$$

Which case the output interval $J$ above corresponds to depends on the input parameters and the mean reward

Algorithm 2 Q-Part: Partition Algorithm for BAI with coherent query constrained $m$

Input: full arm set $\mathcal { K }$ , confidence parameter $\delta$ , constraint $m$   
Initialize: $\delta \gets \delta / 2 , I _ { 1 } , I _ { 2 } \gets [ 0 , 1 ] , \delta ^ { \prime } \gets \delta$ 1: Partition the full arm sets to $\lceil K / m \rceil$ subsets, each with $m$ arms, together denoted as a set $\mathfrak { B }$ ⊲ Stage (i): identify best arm subset 2: while $\mathrm { n i n } I _ { 1 } - \mathrm { m a x } I _ { 2 } < 2 | I _ { 1 } |$ or $| \mathfrak { B } | > 1$ do 3: $I _ { 1 } \gets \mathtt { P a r t s h r i n k } \left( ( O _ { \mathrm { c o n s } } ^ { ( S ) } ) _ { \forall S \in \mathfrak { B } } , \mathfrak { B } , I _ { 1 } , 1 , \delta ^ { \prime } \right)$ 4: $I _ { 2 } \gets \mathtt { P a r t s h r i n k } \left( ( O _ { \mathrm { c o n s } } ^ { ( S ) } ) _ { \forall S \in \mathfrak { B } } , \mathfrak { B } , I _ { 2 } , 2 , \delta ^ { \prime } \right)$ 5: for ${ \boldsymbol { S } } \in { \mathfrak { B } }$ do 6: if GoodRatio $\left( O _ { \mathrm { c o n s } } ^ { ( S ) } , I _ { 1 } , \delta ^ { \prime } \right) = 0$ then ⊲ If no good arm inside subset $s$ 7: ${ \mathfrak { B } } \gets { \mathfrak { B } } \backslash S$ ⊲ Subset elimination 8: $\delta ^ { \prime } \gets \delta ^ { \prime } / 2 \flat$ Halve confidence parameter 9: $\ell _ { 1 } \gets \operatorname* { m i n } I _ { 1 }$ , $\ell _ { 2 } \gets \operatorname* { m a x } I _ { 2 }$   
10: $s \gets \mathfrak { B }$ ⊲ Only remaining subset in $\mathfrak { B }$ ⊲ Stage (ii): identify best arm   
11: Construct variable-time quantum algorithm $\mathcal { A } $ $\mathrm { V T } \mathbb { A } ( O _ { \mathrm { c o n s } } ^ { ( S ) } , S , I = [ \ell _ { 2 } , \ell _ { 1 } ] , \bar { 0 } . 0 1 \delta )$   
12: $k \gets \tt { A m p l i f y } ( \mathcal { A } , \delta ^ { \prime } )$   
Output: arm $k$

of the arms in the interval $I$ . We refer the detail to the PartShrink subroutine in Algorithm 5 in Appendix C.3. The performance guarantees are provided in Lemma 3.

Lemma 3 (Performance of PartShrink). Given $\textit { h } \in$ $\{ 1 , 2 \}$ , an interval $I = [ a , b ]$ , and a confidence parameter $0 < \delta < 1$ , supposing $ { \mu } _ { h } \in I$ and $| I | \geqslant \Delta _ { 2 } / 8$ , there exists a PartShrink $\left( ( O _ { c o n s } ^ { ( S ) } ) _ { \forall S \in \mathfrak { B } } , \mathfrak { B } , I , h , \delta \right)$ subroutine which

1. outputs an interval $J$ with $| J | = 3 | I | / 5$ such that $ { \mu } _ { h } \in J$ with a probability of at least $1 - \delta$ , and

2. uses $\begin{array} { r } { O \left( \sum _ { S \in \mathfrak { B } } \sqrt { \sum _ { k \in S } \frac { 1 } { \Delta _ { k } ^ { 2 } } } \mathrm { p o l y l o g } \left( \frac { K } { m \delta \Delta _ { 2 } } \right) \right) q u e r i e s . } \end{array}$

Lemma 3 guarantees that PartShrink outputs an interval $J$ containing the mean reward $\mu _ { h }$ with high probability and in a reasonable number of queries. The proofs of Lemmas 2 and 3 are presented in Appendix E.1.

Next, we present the partition algorithm for BAI with the $m$ -constrained quantum oracle that builds on the GoodRatio and PartShrink subroutines.

# 4.2 Algorithm Design

This section presents the partition algorithm (Q-Part in Algorithm 2) using the $m$ -constrained quantum oracle. The algorithm partitions $K$ arms into $K / m$ subsets2 and queries arms within each subset to find the optimal one.

Initially, Q-Part partitions the $K$ arms into $K / m$ subsets $\boldsymbol { S } _ { 1 } , \ldots , \boldsymbol { S } _ { K / m }$ (Line 1), each containing $m$ arms, and denote $\mathfrak { B } : = \{ S _ { 1 } , \ldots , S _ { K / m } \}$ . The algorithm has two main stages: (i) identifying the subset containing the optimal arm (Lines 2-8) and (ii) finding the best arm within that subset (Lines 9-12).

To find the optimal arm’s subset, Q-Part uses an elimination process. It starts with all subsets in $\mathfrak { B }$ and progressively removes those without the best arm until one remains. The algorithm maintains two intervals, $I _ { 1 }$ and $I _ { 2 }$ , both initialized to 0, 1 . Within the while loop (Line 2), PartShrink is applied to shrink the intervals $I _ { 1 }$ and $I _ { 2 }$ (Lines 3-4). Then, GoodRatio checks each remaining subset to see if it contains an arm with a mean reward in $I _ { 1 }$ . If not, the subset is eliminated (Line 7). The loop ends when only one subset remains, and $I _ { 1 }$ and $I _ { 2 }$ are separated by a gap of at least $2 | I _ { 1 } |$ (i.e., $\operatorname* { m i n } I _ { 1 } - \operatorname* { m a x } I _ { 2 } \geqslant 2 | I _ { 1 } | )$ .

Upon completion of Stage (i), Q-Part identifies the subset containing the best arm, with $I _ { 1 }$ containing the mean reward $\mu _ { 1 }$ of the best arm and $I _ { 2 }$ containing the mean reward $\mu _ { 2 }$ of the second-best arm. The endpoints $\ell _ { 1 } = \operatorname* { m i n } I _ { 1 }$ and $\ell _ { 2 } ~ = ~ \operatorname* { m a x } I _ { 2 }$ separate the best arm from the rest (Line 9). To find the optimal arm, Q-Part uses a variabletime algorithm (VTA) in (6) with the remaining subset $s$ and interval $[ \ell _ { 2 } , \ell _ { 1 } ]$ as inputs (Line 11), which produces the expected output $\begin{array} { r } { \frac { 1 } { \sqrt { m } } ( \left| k ^ { * } \right. _ { I } \left| 1 \right. _ { F } + \sum _ { k \in \cal S \backslash \{ k ^ { * } \} } \left| k \right. _ { I } \left| 0 \right. _ { F } ) } \end{array}$ . Amplify then determines the index of the optimal arm $k ^ { * }$ (Line 12), which guarantees to output the best arm in the set $s$ with a probability of at least $1 - \delta ^ { \prime }$ .

# 4.3 Query Complexity Upper Bound for Partition Algorithm

We derive a query complexity upper bound for $\mathsf { Q } \mathrm { - } \mathsf { P }$ art (Algorithm 2), and its detail proof is deferred to Appendix E.2.

Theorem 3 (Query complexity upper bound for $\mathsf { Q { \mathrm { - } } P a r t }$ of the $m$ -constrained quantum oracle). With confidence parameter $\delta \in ( 0 , 1 )$ and an arm partition $\mathfrak { B }$ , the query complexity of Algorithm 2 is $\begin{array} { r } { O \left( \sum _ { S \in \mathfrak { B } } \sqrt { \sum _ { k \in S } \frac { 1 } { \Delta _ { k } ^ { 2 } } } \operatorname { p o l y l o g } \left( \frac { K } { \delta \Delta _ { 2 } } \right) \right) } \end{array}$ where $\Delta _ { k } = \mu _ { 1 } - \mu _ { k }$ is the reward gap of arm $k$ , and $\Delta _ { 2 }$ is the minimal reward gap.

As $\Delta _ { 2 }$ is the smallest reward gap, Theorem 3 simplifies the upper bound to $\tilde { O } ( ( K / \sqrt { m } ) \bar { \Delta } _ { 2 } ^ { - 1 } )$ . Thus, a smaller $m$ (better coherence) reduces the query complexity. When $\textit { m } = \textit { 1 }$ (weak quantum oracle), $\mathsf { Q } \mathrm { - } \mathsf { P }$ art’s complexity is $\begin{array} { r } { \tilde { O } ( \sum _ { k \in \mathcal { K } } \Delta _ { k } ^ { - 1 } \log \frac { 1 } { \delta } ) } \end{array}$ , which matches $\mathsf { Q }$ -Elim’s bound for a weak oracle (Theorem 2). However, Q-Elim’s bound $O ( \log { \frac { 1 } { \Delta } } \log { \frac { K } { \delta } } )$ is better than $\mathsf { Q }$ -Part’s polylogarithmic factor $\begin{array} { r } { O ( \mathrm { p o l y l o g } \frac { K } { \delta \Delta _ { 2 } } ) } \end{array}$ (at most $\begin{array} { r } { \log ^ { 3 } \frac { K } { \delta \Delta _ { 2 } } ) } \end{array}$ because $\mathsf { Q }$ -Elim’s parameters are optimized for weak oracles. When $m = K$ (strong quantum oracle), Q-Part reduces to the algorithm by Wang et al. (2021), as no further partitioning is needed $( { \dot { \mathfrak { B } } } \ = \ { \bar { \{ \{ \mathcal { K } } }   )$ . For $1 ~ < ~ m ~ < ~ K$ , $\mathsf { Q }$ -Part’s complexity lies between that of Wang et al. (2021)’s strong oracle and Q-Elim’s weak oracle (see (1)).

# 4.4 Query Complexity Lower Bounds for the m-Constrained Quantum Oracle

In Section 4.4, we establish lower bounds to demonstrate the tightness and optimality of the Q-Part algorithm. The key

×104 ×104 SuccElim 1.2 Q-Part (m=2) Q-Elim 1.0 Q-Part (m=4) fail 105 Q-Part (m=8) 0.8 10 GRn ideal qiskit with simulated noise 0.110.09 0.07 0.05 0.03 0.01 0.11 0.09 0.07 0.05 0.030.01 Suboptimality Gap Suboptimality Gap $\overline { { m = 2 } } \quad \overline { { m = 4 } } \quad \overline { { m = 8 } }$ (a) SuccElim vs. Q-Elim (b) Q-Part with different 𝑚 (c) Impact of noise on Q-Part

challenge is proving the lower bound with the outer summation over all subsets in $\mathfrak { B }$ (i.e., $\Sigma _ { S \in \mathfrak { B } } )$ ). This summation indicates that queries on each subset are “orthogonal”, meaning information gained from one subset does not overlap with others. To address this, we define a class of partition algorithms for $m$ -constrained oracles, which ensures that queries on one subset of arms cannot be used to infer information about arms in any of the other subsets. We then derive a lower bound for any partition algorithm, as stated in Theorem 4, with a proof provided in Appendix G.2.

Definition 1 (Partition algorithm class). A partition algorithm for BAI with the 𝑚-constrained oracle is one that partitions the full arm set into several subsets at initialization, each with at most 𝑚 arms, and always follows this fixed partition when querying arms during algorithm execution.

Theorem 4 (Query complexity lower bound for $m$ -constrained oracle). To identify the best arm with a probability of at least $1 - \delta$ with the 𝑚-constrained oracle with parameter 𝑚, any partition algorithm needs to spend at least the following number of queries, $\Omega ( \sum _ { S \in \mathfrak { B } } \sqrt { \sum _ { k \in S } 1 / \Delta _ { k } ^ { 2 } } )$ , where $\mathfrak { B }$ is the partition of arms.

Note that Q-Part (Algorithm 2) belongs to this partition algorithm class, and if the arm partition $\mathfrak { B }$ in the lower bound is the same as the one chosen in $\mathsf { Q { - } P a r t }$ , then this lower bound matches the upper bound for $\mathsf { Q }$ -Part in Theorem 3 up to some logarithmic factors. This implies that the bounds in both Theorems 3 and 4 are tight, and $\mathsf { Q { \mathrm { - } } P a r t }$ is near-optimal within the partition algorithm class.

# 5 Qiskit-based Simulation

We compare the quantum algorithms Q-Elim (for the weak quantum oracle) and $\mathsf { Q { \mathrm { - } } P a r t }$ (for the $m$ -constrained quantum oracle) with the classical successive elimination SuccElim (Even-Dar et al. 2006).

We set $\delta = 0 . 1$ and $K = 8$ arms with mean rewards ranging from $0 . 9 9 - ( k - 1 ) \times \Delta$ (where $k \in \{ 1 , \ldots , K \} )$ and vary $\Delta$ from 0.11 to 0.01 in steps of 0.02 to analyze its effect on query complexity. Details of the Qiskit implementation are in Appendix $\mathrm { ~ H ~ }$ , and the code is provided in the supplementary material. We implement $\mathsf { Q }$ -Elim with $m = 1$ and $\mathsf { Q { \mathrm { - } } P a r t }$ with $m = 2 , 4$ , and 8. For $m = 8$ , the $m$ -constrained oracle is equivalent to the strong quantum oracle. The default confidence parameter for SuccElim is $c = 4$ (Even-Dar et al. 2006). Results, averaged over 50 trials, are shown in Figure 1.

Figure 1a (with y-axis in log-scale) shows that $\mathsf { Q }$ -Elim outperforms SuccElim, demonstrating the benefits of quantum information with the weak oracle. As $\Delta$ decreases, SuccElim’s query complexity increases faster than $\mathsf { Q }$ -Elim’s, validating the quantum improvement of dependence on $\Delta$ from $\Delta ^ { = 2 }$ to $\dot { \Delta } ^ { - 1 }$ (see Appendix H for curve-fitting). Figure 1b compares Q-Part’s performance for $m \ = \ 2 , 4 , 8$ as $\Delta$ varies. Increasing $m$ improves performance, confirming the advantage of quantum parallelism predicted by the $\tilde { O } ( ( K / \sqrt { m } ) \Delta ^ { - 1 } )$ bound from Theorem 3.

We also assess the impact of noise using Qiskit’s simulation of IBM’s 127-qubit device. Figure 1c shows that Q-Elim’s performance decreases by $2 . 3 8 \%$ and $8 . 0 9 \%$ for $m = 2$ and $m = 4$ , respectively. For $\scriptstyle m = 8$ , $\scriptstyle \mathrm { Q - P a r t }$ fails due to high noise, as the increased qubit and gate requirements exceed practical limits, impairing the algorithm’s functionality. This highlights the importance to study the restrictive $m$ -constrained quantum oracles under a noisy environment.

# 6 Conclusion

In this paper, we explore the best arm identification (BAI) problem using weak and $m$ -constrained quantum oracles. We introduce the $m$ -constrained oracle, which generalizes both the weak oracle $( m = 1 )$ ) and the strong oracle $( m = K )$ . Our quantum algorithms, Q-Elim for the weak oracle and Q-Part for the constrained oracle, offer significant improvements over classical methods. Specifically, $\mathsf { Q } \mathrm { . }$ -Elim achieves a quadratic speedup at the arm level due to quantum parallelism, while $\mathsf { Q { \mathrm { - } } P a r t }$ provides a quadratic speedup at the subset level due to quantum entanglement. We establish query complexity lower bounds for both quantum BAI problems that align with our upper bounds, indicating nearoptimal performance. Our experiments using Qiskit confirm these theoretical results.

# Acknowledgments

The work of Mohammad Hajiesmaili is supported by NSF CNS-2325956, CAREER-2045641, CPS-2136199, CNS2102963, and CNS-2106299. The work of John C.S. Lui is supported in part by SRFS2122-4S02. The work of Don Towsley is supported in part by the NSF grant CNS1955744, NSF-ERC Center for Quantum Networks grant EEC-1941583, and DOE Grant AK000000001829.