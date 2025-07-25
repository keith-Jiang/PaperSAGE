# Dynamic-Width Speculative Beam Decoding for LLM Inference

Zongyue Qin, Zifan He, Neha Prakriya, Jason Cong, Yizhou Sun

University of California Los Angeles, CA, USA {qinzongyue, zifanhe, nehaprakriya, cong, yzsun}@cs.ucla.edu

# Abstract

Large language models (LLMs) based on transformer architecture have shown outstanding performance across numerous real-world tasks. However, the autoregressive nature of these models makes the inference process slow and costly. Speculative decoding has emerged as a promising solution, leveraging a smaller auxiliary model to draft future tokens, which are then validated simultaneously by the larger model, achieving a speed-up of $_ { 1 - 2 \times }$ . Although speculative decoding matches the same distribution as multinomial sampling, multinomial sampling itself is prone to suboptimal outputs, whereas beam sampling is widely recognized for producing higher-quality results by maintaining multiple candidate sequences at each step. This paper explores the novel integration of speculative decoding with beam sampling. However, there are four key challenges: (1) how to generate multiple sequences from the larger model’s distribution given draft sequences from the small model; (2) how to dynamically optimize the number of beams to balance efficiency and accuracy; (3) how to efficiently verify the multiple drafts in parallel; and (4) how to address the extra memory costs inherent in beam sampling. To address these challenges, we propose dynamicwidth speculative beam decoding (DSBD). Specifically, we first introduce a novel draft and verification scheme that generates multiple sequences following the large model’s distribution based on beam sampling trajectories from the small model. Then, we introduce an adaptive mechanism to dynamically tune the number of beams based on the context, optimizing efficiency and effectiveness. Besides, we extend tree-based parallel verification to handle multiple trees simultaneously, accelerating the verification process. Finally, we illustrate a simple modification to our algorithm to mitigate the memory overhead of beam sampling. Experimental results show that our approach achieves a $1 . 5 \substack { - 1 . 9 \times }$ speed-up and $1 . 8 – 2 . 5 \times$ lower energy consumption compared to beam sampling, with no loss in downstream performance. Moreover, it can produce significantly higher-quality outputs than speculative decoding, while maintaining similar time, memory, and energy costs. In summary, our method offers a more efficient and effective inference process for LLMs.

# 1 Introduction

et al. 2023), Llama-3 (AI $@$ Meta 2024), and PALM (Anil et al. 2023), have demonstrated remarkable performance across a wide range of real-world tasks, including text generation, summarization, and translation. However, the autoregressive nature of these models, where tokens are generated one at a time, leads to slow inference speeds and high computational costs. As the size and complexity of LLMs continue to increase, the demands on computational resources and energy consumption during inference have become major concerns, limiting their scalability and accessibility.

Speculative decoding has emerged as a promising technique to accelerate LLM inference by leveraging a smaller auxiliary model to generate draft tokens. These tokens are then validated by the large model, resulting in a significant reduction in inference time. The primary advantage of speculative decoding is its ability to maintain the same quality of output as multinomial sampling while achieving a $1 { - } 2 \times$ speed-up. However, multinomial sampling itself is limited to generating a single sequence based on local optimality. This limitation makes it susceptible to returning suboptimal results, as it lacks the diversity that could be achieved by considering multiple candidate sequences simultaneously.

In recent years, large language models based on transformer architecture (Vaswani et al. 2017), such as GPT-4 (Achiam

Motivated by the need to improve the output quality, we explore the integration of speculative decoding with beam sampling, a technique that maintains multiple candidate sequences (beams) at each step to enhance the diversity and quality of the generated output. This fusion, however, presents several challenges. First, while previous studies focused on obtaining a single token from the large model distribution given draft tokens from the smaller model, our approach requires generating multiple tokens (beams) simultaneously, which necessitates a new verification scheme. Second, determining the optimal number of beams is critical: too many beams can lead to inefficiency due to a high rejection rate, while too few beams may result in under-utilization of the small model’s potential and low effectiveness. Third, efficiently verifying multiple draft sequences in parallel requires a technique that can process and validate multiple beams concurrently. Fourth, addressing the additional memory cost of storing multiple key-value caches is crucial to enable LLMs to use beam sampling in practice.

To address these challenges, we propose dynamic-width speculative beam decoding (DSBD) that combines speculative decoding with beam sampling through a series of innovations. First, we introduce a draft and verification scheme that processes beam decoding trajectories as forests of trees, which are verified layer by layer by the large model. This approach allows us to efficiently generate multiple beams while maintaining the large model’s sampling distribution. Second, we propose a mechanism to dynamically adjust the number of beams based on the context, ensuring a balance between efficiency and effectiveness. Third, we extend existing tree-based parallel verification techniques (Miao et al. 2023) to operate on multiple trees, incorporating a forestbased parallel verification strategy that enhances the speed of the verification process. Finally, we introduce a simple modification to DSBD that reduces the memory cost by storing only one set of key-value caches, while still delivering better output quality than multinomial sampling.

Experimental results show that our approach achieves a $1 . 5 \substack { - 1 . 9 \times }$ speed-up and $1 . 8 – 2 . 5 \times$ smaller energy consumption than beam sampling, without sacrificing performance on downstream tasks. Besides, it can produce significantly higher-quality outputs than speculative decoding, while maintaining comparable time, memory, and energy costs. These findings suggest that DSBD successfully bridges the gap between speculative decoding and beam sampling, providing a more efficient and effective decoding method for LLMs. Our code is open source1.

# 2 Preliminaries

# 2.1 Decodings of LLMs

Let $p$ denote the distribution defined by a large language model $M _ { p }$ . Given an input prefix, the optimal decoding algorithm is to generate a sequence of $N$ tokens with maximum likelihood $\bar { p } ( x _ { 1 : N } | i n p u t )$ .

Multinomial Sampling. Multinomial sampling, also known as standardized sampling, samples the next token $\boldsymbol { x } _ { t }$ based on $\tau \circ p ( \cdot | x _ { 1 : t - 1 } , i n p u t )$ , where $\tau$ is a warping operation applied to enhance the high probability region. Some common warping operations include top- $k$ warping, which limits the selection to the top $k$ tokens, and top- $p$ warping, where tokens are sampled from the smallest possible subset of the vocabulary whose cumulative probability mass exceeds a specified threshold $p$ . The deterministic version of multinomial sampling is a special case when $k = 1$ .

Beam Sampling. Beam decoding aims to do a better job than multinomial sampling. For each position $t$ $( 1 \leq t \leq$ $N _ { , }$ ), it maintains $W > 1$ candidate sequences, which are also called beams. Assume we have already kept the $W$ sequences $\mathcal { T } _ { t - 1 } = \{ x _ { 1 : t - 1 } ^ { ( 1 ) } , \dots , x _ { 1 : t - 1 } ^ { ( W ) } \}$ at position $t - 1 , W$ sequences with length $t$ are then s−ampled from $\tau \circ p _ { b e a m }$ , where $p _ { b e a m } \colon \mathcal { T } _ { t - 1 } \times V  [ 0 , 1 ]$ is the beam sampling probability:

$$
p _ { b e a m } ( x _ { 1 : t - 1 } ^ { ( i ) } , x _ { t } ) = \frac { p ( x _ { 1 : t - 1 } ^ { ( i ) } , x _ { t } | i n p u t ) } { \sum _ { x _ { 1 : t - 1 } ^ { ( j ) } , x _ { t } ^ { \prime } \in \mathcal { T } _ { t - 1 } \times V } p ( x _ { 1 : t - 1 } ^ { ( j ) } , x _ { t } ^ { \prime } | i n p u t ) }
$$

Notice that $p ( x _ { 1 : t - 1 } ^ { ( i ) } , x _ { t } | i n p u t ) \ = \ p ( x _ { t } | x _ { 1 : t - 1 } ^ { ( i ) } , i n p u t ) .$ $p ( x _ { 1 : t - 1 } ^ { ( i ) } | i n p u t )$ . In practice, beam sampling stores the likelihood $p ( x _ { 1 : t - 1 } ^ { ( i ) } | i n p u t )$ for each beam, and the computation complexity of $p _ { b e a m }$ is $O ( W \cdot | V | )$ . In deterministic beam sampling, the top $W$ sequences with the highest likelihood $p _ { b e a m } ( x _ { 1 : t } )$ will be kept.

(Shi et al. 2024) shows that beam sampling in general has better downstream effectiveness than multinomial sampling. Figure 1 shows an example where beam decoding returns a better output.

# 2.2 Vanilla Speculative Decoding

Speculative decoding utilizes a small model to generate the next $\gamma$ tokens and then employs the large model to verify these drafted tokens in parallel. The process is summarized as follows:

1. Given input, the small model samples $\gamma$ draft tokens $x _ { 1 } , \ldots , x _ { \gamma }$ using greedy decoding, based on the warped predicted conditional probability $\tilde { q } ( x _ { t } | x _ { 1 : t - 1 } , i n p u t )$ for $t = 1 , \dots , \gamma$ , where $\tilde { q } = \mathcal { T } \circ q$ and $q$ is the small model’s output distribution.   
2. The large model verifies the draft tokens in parallel by computing the conditional probability $\tilde { p } \ = \ \tau \circ$ $p ( x _ { t } | x _ { 1 : t - 1 } , i n p u t )$ for $t = 1 , \dots , \gamma$ .   
3. Each draft token $\boldsymbol { x } _ { t }$ is accepted with probability $\operatorname* { m i n } ( 1 , \tilde { p } ( x _ { t } ) / \tilde { q } ( x _ { t } ) )$ . The draft tokens before the first rejected token are kept as the decoding output. An additional token is sampled from a residual distribution as a correction for the first rejected token. The accepted tokens and the resampled token are then appended to the context prefix as input for the next iteration.   
4. Repeat steps 1-3 until reaching the stopping criteria, such as a length limit.

By verifying $\gamma$ tokens in parallel with one run of the large model, speculative decoding reduces the time cost compared to calling the large model $\gamma$ times. Additionally, although the small model still runs in an autoregressive manner, its inference speed is much faster than the large model. This makes speculative decoding an effective method to accelerate the inference process of LLMs. Moreover, it has been proven that each token $\boldsymbol { x } _ { t }$ generated by speculative sampling follows the identical sampling distribution as multinomial sampling.

# 3 Methodology

The primary goal of our method is to enhance the efficiency and effectiveness of large language model (LLM) inference by combining the speed advantages of speculative decoding with the accuracy and diversity benefits of beam sampling. We first introduce a novel draft and verification scheme that keeps identical distribution as beam sampling. Then, we describe an adaptive beam management strategy. Next, we illustrate a forest-based parallel verification mechanism. Finally, we discuss how to resolve the additional memory cost inherent in beam sampling.

![](images/0a81a48ffce37da68fd4fc5b46d1efe357a1968d0b0190f8205b67c009729af8.jpg)  
Figure 1: Examples of greedy and beam sampling. Some nodes are omitted in the figures. Assume the sampling probability is warped to always sample the tokens with the largest probabilities. Given prefix “h”, multinomial sampling generates “hello” with an average perplexity of 1.55. Beam sampling generates “happy” with an average perplexity of 1.49.

(a) Draft forest from the small model.

![](images/501abc247816e75e5b1180e3157e17e780b084dab88641018e2829a862dd8b59.jpg)  
(b) Verification result of the 1st draft layer. (c) Verification result of the 2nd draft layer.   
Figure 2: Illustration of one iteration of Speculative Beam Decoding. (a) Draft Stage: given the input beams “who” and “why”, the small model first generates a trace of beam sampling. (b)(c): Verification Stage. When verify the first draft layer, “who are” and “why do” are accepted, while “why is” is rejected. When verify the second draft layer, “why is it” is directly rejected because its parent is rejected. Then “who are they” is accepted, while “who are it” is rejected. And another beam “who are you” is sampled from the residual distribution.

# 3.1 Draft and Verification Scheme

Overview As illustrated in Figure 2, the core idea of our method is to leverage a smaller, auxiliary model to generate multiple draft sequences, referred to as draft beams, which are then verified and refined by the larger model. This approach enables us to maintain multiple candidate sequences throughout the decoding process, thereby achieving better output quality than multinomial sampling, while improving the overall efficiency of beam sampling.

For now, assume that the number of beams (also referred to as the width, denoted as $W _ { L }$ ) is fixed. In each iteration of our method, the input consists of the beams generated in the previous iteration. For the first iteration, the input is the initial input context. At each iteration, our method first uses the small model to perform beam sampling with a width of $W _ { S }$ for $\gamma$ steps. Notice that we want $W _ { S } > W _ { L }$ because some draft beams might be rejected later. As illustrated in Figure 2a, it generates a trajectory that can be represented as a forest consisting of $W _ { L }$ trees, which we refer to as the “draft forest”. In this forest, each tree originates from an input beam, with the maximum depth of each tree being $\gamma + 1$ . Starting from the second layer, each layer of the forest contains $W _ { S }$ nodes, representing the intermediate beams at each step of the beam sampling process.

Once the draft forest is generated, our method leverages the large model to predict the distribution for the next token of each node (beam) in parallel. Using these distributions, DSBD then verifies each layer of the draft forest sequentially. For each layer, it calculates the joint probability of the beams and sequentially determines whether each beam should be accepted. If $W _ { L }$ beams are accepted in a given layer, the remaining beams are discarded, and the method is moved on to verify the next layer. If fewer than $W _ { L }$ beams are accepted in layer $l$ , the method rejects this layer and terminates the verification process.

When verification ends, either because it is terminated or because there are no more layers to verify, our method samples an additional layer with $W _ { L }$ beams. This additional layer either corrects the first rejected layer or adds a new layer if all draft layers are accepted. The output beams from this additional layer then serve as the input beams for the next iteration, continuing until the stopping criteria are met (e.g., reaching the maximum number of tokens).

This approach allows each run of the large model to produce at least one, and possibly multiple, steps of beam sampling. Previous studies have shown that memory operations during LLM runs contribute significantly to both runtime and energy consumption (Leviathan, Kalman, and Matias 2023; Allen and Ge 2016; Chen et al. 2011). By generating multiple tokens in a single run, DSBD reduces the number of memory operations required, which in turn improves both the speed and the energy efficiency of LLM inference.

Details Let $p$ denote the output distribution of the large model and $q$ denote the distribution of the small model. We will start by explaining how to verify the first draft layer (which is the second layer of the draft forest) during each iteration.

Let $\overrightarrow { \tau } = \{ x _ { 1 : t } ^ { ( 1 ) } , \cdot \cdot \cdot , x _ { 1 : t } ^ { ( W _ { L } ) } \}$ x(1:Wt L)} represent the input beams, in hSe fi t l1a:yt+er  ·o·f  the td+ra1f}t forest. Note that and represent the draft beams $x _ { 1 : t + 1 } ^ { ( i ) }$ is sampled from the distribution $q _ { b e a m } ( x _ { 1 : t + 1 } ^ { ( i ) } ) ~ = ~ \mathcal { T } ~ \circ$ $\frac { q ( x _ { 1 : t + 1 } ^ { ( i ) } ) } { \mathcal { Q } }$ , where $\tau$ denotes the warping operation and x1:t+1 V q(x1:t+1). Similarly, let pbeam denote the beam sampling distribution of the large model, $\begin{array} { r } { p _ { b e a m } ( x _ { 1 : t + 1 } ^ { ( i ) } ) = \ T \circ \frac { p ( x _ { 1 : t + 1 } ^ { ( i ) } ) } { \mathcal { P } } } \end{array}$ p(x(1i:)t+1) , where = $\begin{array} { r } { \sum _ { x _ { 1 : t + 1 } \in \mathbb { Z } \times V } p \big ( x _ { 1 : t + 1 } \big ) } \end{array}$

During verification, our method starts by setting $p ^ { \prime } \ =$ $p _ { b e a m }$ . For each draft beam $x _ { 1 : t } ^ { ( i ) }$ , DSBD accepts it with prob$\begin{array} { r } { \operatorname* { m i n } ( 1 , \frac { q _ { b e a m } ( x _ { 1 : t } ^ { ( i ) } ) } { p ^ { \prime } ( x _ { 1 : t } ^ { ( i ) } ) } ) . } \end{array}$ If $x _ { 1 : t } ^ { ( i ) }$ is rejected, the method updates $p ^ { \prime }$ by setting it to $n o r m ( \operatorname* { m a x } ( 0 , p ^ { \prime } - q _ { b e a m } ) )$ , where norm denotes the normalization operation. Then it continues to verify the next draft beam with the updated $p ^ { \prime }$ . If the beam is accepted, $p ^ { \prime }$ is reset to $p _ { b e a m }$ . If $W _ { L }$ draft beams have already been accepted in this layer, the method will reject all remaining beams.

Now we illustrate how to verify the second draft layer. The difference is that some beams in the first layer have already been rejected. In this case, all the beams stem from the rejected beams are directly rejected. For the remaining beams, DSBD applies the same verification process as above.

If all layers in the draft forest have $W _ { L }$ accepted beams, the method proceeds to sample an additional layer with $W _ { L }$ beams directly from $p _ { b e a m }$ . However, if at any layer $l$ fewer than $W _ { L }$ beams are accepted, the method will first sample one beam from the adjusted distribution $p ^ { \prime }$ . If the number of accepted beams still falls short of $W _ { L }$ , additional beams will be sampled from the original distribution $p _ { b e a m }$ to meet the required number.

Theorem 3.1. Correctness of Draft and Verification Scheme. Let I = {x(11:t), · $\mathcal { T } = \{ x _ { 1 : t } ^ { ( 1 ) } , \cdot \cdot \cdot , x _ { 1 : t } ^ { ( W _ { L } ) } \}$ x(1:Wt L)} denote input beams, $\mathcal { S } = \{ x _ { 1 : t + 1 } ^ { ( 1 ) } , \cdot \cdot \cdot , x _ { 1 : t + 1 } ^ { ( W _ { S } ) } \}$ denote draft beams, and $\mathcal { O } =$ $\{ \tilde { x } _ { 1 : t + 1 } ^ { ( 1 ) } , \cdot \cdot \cdot , \tilde { x } _ { 1 : t + 1 } ^ { ( W _ { L } ) } \}$ denote the output beams obtained by our algorithm. We have $1 , \ldots , W _ { L } )$ , where $p _ { b e a m } ( x _ { 1 : t + 1 } ^ { ( i ) } ) \ = \ T \circ ( p ( x _ { 1 : t + 1 } ^ { ( i ) } ) / \mathcal { P } ) .$ $\tilde { x } _ { 1 : t + 1 } ^ { ( i ) }$ pbeam ( i = , $\begin{array} { r } { \mathcal { P } = \sum _ { x _ { 1 : t + 1 } \in \mathbb { Z } \times V } p ( x _ { 1 : t + 1 } ) } \end{array}$ .

The proof is illustrated in (Qin et al. 2024).

# 3.2 Dynamic-Width Speculative Beam Decoding

The draft and verification scheme described above ensures that our method matches the sampling distribution of beam sampling. However, it has a limitation: the beam width $W _ { L }$ remains fixed across all layers. While this fixed width works well for standard beam sampling, it is not suitable for our method. The key challenge is that the discrepancy between the small model’s predictions $( q _ { b e a m } )$ and the large model’s true distribution $( p _ { b e a m } )$ vary from token to token. In some

1: Input: Draft Forest with $\gamma$ draft layers, Small model   
distribution $q$ , Large model distribution $p$ , Beam width   
$W _ { L } , W _ { S }$ .   
2: Output: Verified beams for the next iteration   
3: $l _ { l a s t }  \gamma + 1$   
4: for $l = 1 , \ldots , \gamma$ do   
5: $/ / \mathcal { T } ^ { ( l ) }$ is the beams of layer $l - 1$ in the forest.   
6: $\boldsymbol { \mathcal { T } ^ { ( l ) } } \gets$ input beams of layer $l$ .   
7: // $\mathbf { \mathcal { S } } ^ { ( l ) }$ is the beams of layer $l$ in the forest.   
8: $\boldsymbol { S } ^ { ( l ) } $ draft beams of layer $l$ .   
9: // remove beams stem from beams rejected in the last   
10: $\begin{array} { r } { \dot { S ^ { ( l ) } } \gets \{ x _ { 1 : t + 1 } ^ { ( l , i ) } | x _ { 1 : t + 1 } ^ { ( l , i ) } \in S ^ { ( l ) } , x _ { 1 : t } ^ { ( l , i ) } . } \end{array}$ is not rejected}   
11: $\mathit { 1 / t + 1 }$ is the length of sequence in $\mathbf { \mathcal { S } } ^ { ( l ) }$ , $t = l - 1$ .   
12: compute p(ble)am based on next-token distributions $p$   
13: $p ^ { \prime } \gets p _ { b e a } ^ { ( l ) }$ m   
14: Compute $W _ { L } ^ { ( l ) }$ based on Eq 2 - Eq. 6   
15: for $x _ { 1 : t + 1 } ^ { ( l , i ) } \in S ^ { ( l ) }$ do   
16: $r \gets \dot { U } ( 0 , 1 )$   
17: if $\begin{array} { r } { r \leq \frac { q _ { b e a m } ^ { ( l ) } \left( x _ { 1 : t + 1 } ^ { ( i ) } \right) } { p ^ { \prime } ( x _ { 1 : t + 1 } ^ { ( i ) } ) } } \end{array}$ qb(le)am(x(1i:)t+1) then   
18: accept x(1l:,ti+) 1   
19: p′ ← pbeam (l)   
20: else   
21: reject x(1l:,ti+) 1   
22: $p ^ { \prime } \gets \mathrm { n o r m } ( \operatorname* { m a x } ( 0 , p ^ { \prime } - q _ { b e a m } ^ { ( l ) } ) )$   
23: if $W _ { L } ^ { ( l ) }$ beams are accepted then   
24: reject remaining beams   
25: break   
26: if less than $W _ { L } ^ { ( l ) }$ beams are accepted then   
27: sample $\overline { { x _ { 1 : t + 1 } } } \sim p ^ { \prime }$ and add it to accepted beams   
28: while less than $W _ { L } ^ { ( l ) }$ beams are accepted do   
29: sample $x _ { 1 : t + 1 } \sim p _ { b e a m } ^ { ( l ) }$ and add it to accepted   
beams   
30: ${ l _ { l a s t } } \gets l$   
31: break   
32: if ${ l _ { l a s t } = = \gamma + 1 }$ then   
33: compute p(bγea+m1)   
34: sample $W _ { L }$ beams from $p _ { b e a m } ^ { ( \gamma + 1 ) }$   
35:   
36: return accepted beams at the layer $l _ { l a s t }$

layers, $q _ { b e a m }$ closely aligns with $p _ { b e a m }$ , resulting in a high acceptance rate. In other layers, the gap is much wider, leading to a lower acceptance rate.

To address this variability, the decoding algorithm should dynamically adjust the number of beams it expects to accept based on the alignment between $q _ { b e a m }$ and $p _ { b e a m }$ . By doing so, it can (1) reduce the target width for challenging layers, preventing the entire layer from being rejected and thus maintaining progress, and (2) increase the target width for easier layers, enhancing the exploration of diverse sequences and reducing the risk of getting trapped in local optima. This adaptive approach would optimize the balance between efficiency and accuracy, making the decoding process more robust and effective. So we propose a self-adjusting method where the target width $W _ { L } ^ { ( l ) }$ for layer $l$ is determined based on the context of that layer.

Let $P _ { p , q } ^ { ( l ) } ( m , k )$ represent the probability that $k$ out of $m$ draft beams are accepted at the $l .$ -th layer. This probability is computed using the following recursive equation:

$$
P _ { p , q } ^ { ( l ) } ( m , k ) = \sum _ { i = 1 } ^ { m } \tilde { P } _ { p , q } ^ { ( l ) } ( m , i ) P _ { p , q } ( m - i , k - 1 ) )
$$

Here, P˜p(,lq)(m, i) is the probability that the i-th beam is the first to be accepted among the $m$ draft beams:

$$
\tilde { P } _ { p , q } ^ { ( l ) } ( m , i ) = \alpha _ { i } ^ { ( l ) } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ^ { ( l ) } )
$$

where $\alpha _ { j } ^ { ( l ) }$ is the probability that the $j$ -th beam is accepted, given that all previous beams (from the 1st to the $( j - 1 )$ -th) were rejected.

$$
\alpha _ { j } ^ { ( l ) } = \sum q _ { b e a m } \operatorname* { m i n } ( p _ { j } ^ { ( l ) } / q _ { b e a m } ^ { ( l ) } , 1 )
$$

where p $p _ { 1 } ^ { ( l ) } = p _ { b e a m } ^ { ( l ) }$ , $p _ { k } ^ { ( l ) } = n o r m ( \operatorname* { m a x } ( p _ { k - 1 } ^ { ( l ) } - q _ { b e a m } ^ { ( l ) } , 0 ) )$

Using these equations and the fact that $P _ { p , q } ^ { ( l ) } ( m , k ) = 0$ if $k > m$ and $P _ { p , q } ^ { ( l ) } ( 0 , 0 ) = 1$ , we can calculate the probability that at least $K$ beams are accepted at the $l$ -th layer as:

$$
1 - \sum _ { k = 1 } ^ { K - 1 } P _ { p , q } ^ { ( l ) } ( M _ { S } , k )
$$

Finally, the width $W _ { L } ^ { ( l ) }$ for the $l$ -th layer is set based on Eq 5, ensuring that it is not less than a minimum width $W _ { m i n }$ :

$$
W _ { L } ^ { ( l ) } = \operatorname * { m a x } ( W _ { m i n } , \tilde { W } _ { L } ^ { ( l ) } ( t ) )
$$

In this expression, $t \in [ 0 , 1 ]$ is a pre-defined threshold. The value of $\tilde { W } _ { L } ^ { ( l ) } ( t )$ is computed as follows:

$$
\tilde { W } _ { L } ^ { ( l ) } ( t ) = \operatorname* { m a x } \{ K \in \mathbb { N } | 1 - \sum _ { k = 0 } ^ { K - 1 } P _ { p , q } ^ { ( l ) } ( M _ { S } , k ) \geq t \}
$$

This formula gives us the maximum width $\tilde { W } _ { L } ^ { ( l ) } ( t )$ such that the probability of accepting at least $\tilde { W } _ { L } ^ { ( l ) } ( t )$ beams at the $l$ -th layer is greater than or equal to the threshold $t$ . Eq 6 ensures that the width is dynamically adjusted to maintain a high likelihood of accepting a sufficient number of beams, while also ensuring that it does not fall below the minimum width W min. Algorithm 1 illustrates the pseudocode for the draft and verification scheme.

Let $\begin{array} { r } { \beta _ { W _ { m i n } } ^ { ( l ) } \ = \ \sum _ { k = W _ { m i n } } ^ { W _ { S } } P _ { p , q } ^ { ( l ) } ( W _ { S } , k ) ) } \end{array}$ Pp(,lq)(WS, k)), which is the probability that at east $W _ { m i n }$ beams are accepted at layer l. Based on the definition of $W _ { L } ^ { ( l ) }$ , the probability that layer

$L$ is accepted is $\operatorname* { m i n } ( t , \beta _ { W _ { m i n } } ^ { ( l ) } )$ . So $t$ and $W _ { m i n }$ both control the average acceptance rate of our algorithm, and hence determine efficiency. Let β¯ = Eβ(l) , we have the following theorem for the efficiency of DSBD.

Theorem 3.2. The expected number of steps generated per iteration is 1−min(t,β)γ .

Proof. As described above, the average acceptance rate is $\operatorname* { m i n } ( t , \beta )$ . With the Theorems in (Leviathan, Kalman, and Matias 2023), we can calculate the average number of generated layers as $\frac { 1 - \operatorname* { m i n } ( t , \beta ) ^ { \gamma + 1 } } { 1 - \operatorname* { m i n } ( t , \beta ) }$ □

# 3.3 Forest-based Parallel Decoding

As noted in (Miao et al. 2023), efficient management of the key-value cache is crucial to avoid redundant computations when running the large model during verification, which affects overall efficiency. SpecInfer (Miao et al. 2023) introduced tree-based parallel decoding, which reuses the same key-value cache and employs a topology-aware causal mask to accelerate the computation of the large model. However, this tree-based parallel decoding approach cannot be directly applied to our algorithm because, unlike SpecInfer, our method retains multiple beams as inputs at each iteration. Although these beams share the same initial input, the tokens generated in each beam can differ significantly as the sequence length increases. As a result, the draft tokens in DSBD form not a single tree but a forest.

So we propose forest-based parallel decoding, an extension of tree-based parallel decoding that accommodates multiple trees. As shown in Figure 3, DSBD maintains a separate key-value cache for each input beam. For each beam, we apply tree-based parallel decoding to compute the tree attention across all tokens. Finally, after the iteration ends, DSBD updates the key-value caches according to the output beams. For example, if the output beams in Figure 3 are $b _ { 5 }$ and $b _ { 6 }$ , which both originate from $b _ { 1 }$ , then the caches for $b _ { 5 }$ and $b _ { 6 }$ are kept for the next iteration.

# 3.4 Reducing Memory Cost

In practice, key-value caches take up a large portion of memory cost for LLM inference (Kang et al. 2024). A critical disadvantage of beam sampling is that it has to maintain a separate key-value cache for each beam, significantly increasing the memory cost. But our method can mitigate this issue with a simple modification. Notice that with the forest-based parallel decoding, the number of key-value caches kept during generation equals the number of input beams. So an effective way to reduce the memory cost of our method is to limit the number of input beams. This can be achieved by selecting only the output beam with the lowest perplexity as the input beam for the next iteration. In this way, only one key-value cache is needed during generation, so the memory cost will be similar to the cost of multinomial sampling and speculative decoding. Notice that although there is only one input beam, more than one beam can be accepted at each layer of the draft forest. Hence, it will be more effective than multinomial sampling.

![](images/e4206a682912348a42cd892589d46952d27f3f3fe4b7a886a8f382f7d32f07d6.jpg)  
Figure 3: Illustration of forest-based parallel decoding. Given the draft forest, the large model converts the two trees into sequences in depth-first search order and verifies them in parallel with the topology-aware attention mask. Empty cells in the matrices indicate that attention is masked.   
Figure 4: Evaluation on SQuAD. Exact match (EM) is higher the better. The blue points represent performances of DSBD under different parameter settings $( \gamma , W _ { S } , t )$ . The blue and yellow lines mark the Pareto fronts of DSBD and beam sampling. (SpD: SpecDecode, SI: SpecInfer)

# 4 Experiments

# 4.1 Experiment Setups

LLMs. We evaluate our method using three publicly available LLM families: OPT (Zhang et al. 2022), Llama-2 and Llama-3 (Touvron et al. 2023; AI $@$ Meta 2024). We use Llama-2-13B, Llama-3.1-8B, and OPT-13B as the large models as they are the largest models our GPU could run. And we use Llama-68M (Miao et al. 2023) , Llama-3.2-1B, and OPT-125M as the small models.

Datasets. We use public datasets: SQuAD (Rajpurkar, Jia, and Liang 2018), Spider (Yu et al. 2018), and MTBench (Zheng et al. 2023). SQuAD is a natural language QA dataset using exact match (EM) as the evaluation metric. Spider is a text-to-SQL code dataset that uses execution accuracy (EA) as the metric. MT-bench covers various tasks including writing, roleplay, extraction, stem, humanities, reasoning, math, and coding. It uses GPT-4 to rate the output quality on a scale of 1-10 (the higher the better).2

# 4.2 Comparison with Beam Sampling

We begin by comparing DSBD with beam sampling, focusing on the relationship between efficiency (e.g., energy consumption and throughput) and effectiveness. The width of beam sampling ranges from 1 to 4. When width equals 1, beam sampling is equivalent to multinomial sampling. In addition, we observe the improvement in downstream effectiveness and output perplexity begins to converge when the width reaches around 4. For our method, we vary the draft beam width $W _ { S } ~ \in ~ \{ 2 , 3 , 4 , 5 , 6 \}$ , the threshold $t \in$ $\{ 0 . 7 , 0 . 9 \}$ , and set $W _ { m i n } ~ \in ~ \{ 1 , 2 , 3 \}$ . We also include speculative decoding (Leviathan, Kalman, and Matias 2023) (SpD) and SpecInfer (Miao et al. 2023) (SI) as references.

Figure 4 and Figure 5 illustrate the points that mark the performance of different methods under different parameter settings on SQUAD and Spider datasets, respectively. SpD and SpecInfer each have only one point in the figures because they do not offer a trade-off between efficiency and effectiveness. We plot the curves of beam sampling and the Pareto fronts of DSBD. Notably, we omit the results of the OPT model on the Spider dataset as its execu

86 86 84 84   
8 8 DSBD   
78 78 spam 76 76 SI 74 74 白夕 白 Throughput (token/second) Energy (Joule/Token) (a) Throughput vs EM Score (b) Energy vs EM Score (Llama-2) (Llama-2) 82 82 80 80 78 76 DSBD   
M 74 74 Beam 日 72 日 72 SpD 70 70 SI 68 68 白白白 8 古六白白 Throughput (token/second) Energy (Joule/Token) (c) Throughput vs EM Score (d) Energy vs EM Score (Llama-3) (Llama-3) 76 76   
DSBD   
68 M 68 Beam 66 66 SpD 64 64 SI 白小名 又 6 8 0 → 1 Throughput (token/second) Energy (Joule/Token) (e) Throughput vs EM Score (f) Energy vs EM Score (OPT) (OPT)

tion accuracy remains consistently close to zero, rendering it uninformative for this analysis. The figures demonstrate that DSBD consistently outperforms beam sampling, signifying that it achieves higher quality at any given level of throughput or energy consumption. More importantly, when the effectiveness is fixed, DSBD can be $1 . 5 \ – 1 . 9 \times$ faster than beam sampling, while reducing energy consumption by 1.8- $2 . 5 \times$ , as demonstrated by the Pareto fronts of DSBD. Table 1 presents the results on MT-Bench. Due to the cost and time of GPT-4 evaluations, we report results for SpecInfer, beam sampling $W = 5$ ), and DSBD. DSBD achieves comparable efficiency to SpecInfer while significantly improving output quality. It is also $1 . 5 3 \times$ faster and $1 . 5 4 \times$ more energy-efficient than beam sampling. These results highlight DSBD’s advantages in efficiency and effectiveness, making it ideal for real-world applications.

# 4.3 Comparison under Memory Constraint

As discussed in Section 3.4, DSBD can mitigate the memory issue of beam sampling by selecting only one output

45 45 ！   
45 DSBD s   
A30 30 Beam 25 25 SpD SI 20 20 白白呂 8 9 ☆夕 Throughput (token/second) Energy (Joule/Token) (a) Speed vs EA Score (b) Energy vs EA Score (Llama-2) (Llama-2) 75 75 70 70   
65 65 DSBD 5060   
A 55 E55 spam 50 50 SI 6 8 0 2 Throughput (token/second) Energy (Joule/Token) (c) Speed vs EA Score (d) Energy vs EA Score (Llama-3) (Llama-3)

Table 1: Evaluation on MT-Bench with SpecInfer, beam sampling and DSBD.   

<html><body><table><tr><td>Model</td><td>Method</td><td>Score</td><td>Token/s</td><td>J/token</td></tr><tr><td rowspan="3">Llama-2-13B</td><td>SpecInfer</td><td>2.86</td><td>21.8</td><td>21.2</td></tr><tr><td>Beam (W=5)</td><td>3.51</td><td>12.1</td><td>26.3</td></tr><tr><td>DSBD</td><td>3.52</td><td>16.5</td><td>16.1</td></tr><tr><td rowspan="3">Llama-3-8B</td><td>SpecInfer</td><td>3.46</td><td>20.2</td><td>19.8</td></tr><tr><td>Beam (W=5)</td><td>4.10</td><td>10.5</td><td>33.3</td></tr><tr><td>DSBD</td><td>4.11</td><td>17.8</td><td>22.9</td></tr></table></body></html>

beam for the next iteration. It allows DSBD to only keep one sequence of key-value cache and to achieve memory usage comparable to that of multinomial sampling. To assess the performance of DSBD under memory constraints (i.e., only keeps one sequence of key-value cache), we compare it with multinomial sampling, SpD, and SpecInfer, as shown in Table 2. In addition, the DSBD in Table 1 also only keeps one sequence of key-value cache. The results show that DSBD achieves speed and energy efficiency close to that of SpD. Moreover, DSBD delivers a significant improvement in output quality, far surpassing the baselines in downstream scores.

# 5 Related Work

EFFICIENT LLM INFERENCE. Numerous studies have focused on improving the efficiency of large model inference, including model quantization (Frantar et al. 2022; Lin et al. 2023), model pruning (Gale, Elsen, and Hooker 2019;

Table 2: Comparison under memory constraints: each method stores key-value caches for only one sequence.   

<html><body><table><tr><td></td><td>Method</td><td>EM/EA</td><td>tokens/s</td><td>J/token</td></tr><tr><td rowspan="3">Llama-2 SQuAD</td><td>Multinomial</td><td>74</td><td>21.14</td><td>12.36</td></tr><tr><td>SpD</td><td>75</td><td>27.11</td><td>8.34</td></tr><tr><td>SpecInfer DSBD</td><td>74 86</td><td>27.15 26.67</td><td>9.22 8.75</td></tr><tr><td rowspan="4">Llama-2 Spider</td><td>Multinomial</td><td>20</td><td>21.98</td><td>11.14</td></tr><tr><td>SpD</td><td>19</td><td>31.74</td><td>7.06</td></tr><tr><td>SpecInfer</td><td>19</td><td>32.00</td><td>8.01</td></tr><tr><td>DSBD</td><td>31</td><td>30.30</td><td>7.17</td></tr></table></body></html>

Sanh, Wolf, and Rush 2020), and model distillation (Hinton, Vinyals, and Dean 2015). While these techniques achieve significant speed-ups, they often sacrifice the model’s overall effectiveness. A closely related direction to our work is non-autoregressive decoding, enabling parallel generation of multiple tokens (Gu et al. 2017; Wang et al. 2019; Sun et al. 2019; Ghazvininejad et al. 2019; Lee, Mansimov, and Cho 2018; Guo, Xu, and Chen 2020). However, these methods typically require extensive retraining of the model and often face challenges in either maintaining model effectiveness or achieving comparable performance without relying on taskspecific techniques (Kim et al. 2023).

SPECULATIVE DECODING. Speculative decoding is initially introduced in (Leviathan, Kalman, and Matias 2023; Chen et al. 2023). More recent works (Sun et al. 2023; Miao et al. 2023; Yang et al. 2024) extend this concept by allowing the small model to generate multiple draft sequences. All these methods only maintains a single sequence during generation, making them prone to sub-optimal results. Recently, Andronov et al. (Andronov et al. 2024) proposed a decoding method called “speculative beam search”. While it retains multiple candidate sequences to handle the chemical synthesis planning task, it does not preserve the same distribution as either beam sampling or multinomial sampling, and their method is fundamentally different from ours. Another complementary direction to enhance speculative decoding is to improve the effectiveness of the small draft model. A more effective draft model leads to a higher acceptance rate of draft tokens, which in turn accelerates the overall inference process (Kim et al. 2023; Liu et al. 2023; He et al. 2023). EAGLE (Li et al. 2024) and MEDUSA (Cai et al. 2024) train additional heads in the target model to generate draft tokens and achieve better acceptance rate. These works are orthogonal to our work because our algorithm can be directly applied to their draft models.

# 6 Conclusion

This work introduces a novel method that integrates speculative decoding with beam sampling to enhance the efficiency and effectiveness of large language model (LLM) inference. Experimental results show that DSBD outperforms beam sampling, achieving a significant speed-up and energy reduction without compromising downstream task performance. This work enhances the effectiveness of speculative decoding and opens new avenues for exploration.