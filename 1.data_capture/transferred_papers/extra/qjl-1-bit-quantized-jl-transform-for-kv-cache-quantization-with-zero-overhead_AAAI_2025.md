# QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead

Amir Zandieh1, Majid Daliri2, Insu Han3

1Google Research 2New York University 3KAIST zandieh@google.com, daliri.majid@nyu.edu, insu.han@kaist.ac.kr

# Abstract

Serving LLMs requires substantial memory due to the storage requirements of Key-Value (KV) embeddings in the KV cache, which grows with sequence length. An effective approach to compress KV cache is quantization. However, traditional quantization methods face significant memory overhead due to the need to store quantization constants (at least a zero point and a scale) in full precision per data block. Depending on the block size, this overhead can add 1 or 2 bits per quantized number. We introduce QJL, a new quantization approach that consists of a Johnson-Lindenstrauss (JL) transform followed by signbit quantization. In contrast to existing methods, QJL eliminates memory overheads by removing the need for storing quantization constants. We propose an asymmetric estimator for the inner product of two vectors and demonstrate that applying QJL to one vector and a standard JL transform without quantization to the other provides an unbiased estimator with minimal distortion. We have developed an efficient implementation of the QJL sketch and its corresponding inner product estimator, incorporating a lightweight CUDA kernel for optimized computation. When applied across various LLMs and NLP tasks to quantize the KV cache to only 3 bits, QJL demonstrates a more than fivefold reduction in KV cache memory usage without compromising accuracy, all while achieving faster runtime.

# Code — https://github.com/amirzandieh/QJL Extended version — https://arxiv.org/pdf/2406.03482

# Introduction

Large language models (LLMs) have garnered significant attention and demonstrated remarkable success in recent years. Their applications span various domains, including chatbot systems (Achiam et al. 2023; Team 2024a) to textto-image (Ramesh et al. 2022; Team 2023a, 2022), text-tovideo synthesis (Team 2024c), coding assistant (Team 2023b) and even multimodal domain across text, audio, image, and video (OpenAI 2024). The Transformer architecture with self-attention mechanism (Vaswani et al. 2017) is at the heart of these LLMs as it enables capturing intrinsic pairwise correlations across tokens in the input sequence. The ability of LLMs grows along with their model size (Kaplan et al. 2020), which leads to computational challenges in terms of huge memory consumption.

Deploying auto-regressive transformers during the generation phase is costly because commercial AI models must simultaneously serve millions of end users while meeting strict latency requirements. One significant challenge is the substantial memory needed to store all previously generated key-value (KV) embeddings in cache to avoid recomputations. This has become a major memory and speed bottleneck, especially for long context lengths. Additionally, the GPU must load the entire KV cache from its main memory to shared memory for each token generated, resulting in low arithmetic intensity and leaving most GPU threads idle. Therefore, reducing the KV cache size while maintaining accuracy is crucial.

There are several approaches to address this challenge. One method involves reducing the number of heads in the KV cache using multi-query attention (Shazeer 2019) and multi-group attention (Ainslie et al. 2023), but these require fine-tuning the pre-trained models or training from scratch. Another line of work tries to reduce the KV cache size by pruning or evicting unimportant tokens (Zhang et al. 2024b; Liu et al. $2 0 2 4 \mathrm { a }$ ; Xiao et al. 2023; Zandieh et al. 2024). Additionally, some recent works tackle the issue from a system perspective, such as offloading (Sheng et al. 2023) or using virtual memory and paging techniques in the attention mechanism (Kwon et al. 2023).

A simple yet effective approach is to quantize the floatingpoint numbers (FPN) in the KV cache using fewer bits. Several quantization methods have been proposed specifically for the KV cache (Yue et al. 2024; Yang et al. 2024; Dong et al. 2024; Kang et al. 2024; Zhang et al. 2024a). Most recently, KIVI (Liu et al. 2024b) and KVQuant (Hooper et al. 2024) proposed per-channel quantization for the key cache to achieve better performance. However, all existing quantization methods for the KV cache face significant “memory overhead” issues. Specifically, all these methods group the data into blocks, either channel-wise or token-wise, and calculate and store quantization constants (at least a zero point and a scale) for each group. Depending on the group size, this overhead can add approximately 1 or 2 additional bits per quantized number, which results in significant computational overhead. In this work, our goal is to develop an efficient, data-oblivious quantization method, based on sketching techniques. This method, which we call QJL, does not need to be tuned by or adapted to the input data with significantly less overhead than prior works, without any loss in performance.

# Overview of Contributions

The decoding phase in the attention mechanism involves the following computations: (1) computing attention scores by applying the softmax function to the inner product between the current query embedding and all previously generated keys, and (2) multiplying the attention scores with all previously generated values. To make the attention score calculations in step (1) more memory efficient, we quantize the keys in the cache. We introduce a quantization scheme for key embeddings, named QJL, leveraging randomized sketching techniques. Alongside, we develop a high-accuracy estimator for the inner product of query/key pairs, crucial for mitigating errors amplified by the softmax operation in attention score calculations.

Firstly, we revisit a fundamental concept in numerical linear algebra: applying a Johnson-Lindenstrauss (JL) transform, i.e., a random Gaussian projection, to a pair of vectors and then computing the inner product of the projected vectors provides an unbiased and low-distortion estimator for their original inner product (Dasgupta and Gupta 2003). To address the key cache quantization problem, our aim is to quantize the result after applying the JL transform to a key embedding, ideally to just a single bit. Surprisingly, we prove that by applying the JL transform to a key embedding and then quantizing the result to a single bit (the sign bit), while applying the same JL transform to the query embedding without quantization, we still obtain an unbiased estimator of their inner product (see Lemma 2). Moreover, the distortion of this estimator is small and comparable to that of the standard JL transform (see Lemma 3). In Theorem 4, we demonstrate that the proposed inner product estimator based on QJL achieves a relative distortion of $1 \pm \varepsilon$ on the final attention scores. Notably, the number of required bits for representing quantized keys is independent of the embedding dimension and scales logarithmically with the context length, using a fixed number of bits per token.

Thus the QJL sketch combines a JL transform—a random Gaussian projection—with quantization to the sign bit. An overview of this approach is illustrated in Figure 1. Unlike previous methods, the QJL sketch can quantize vectors with zero overhead because it does not require grouping the data and storing quantization constants (zeros and scales) per group. Furthermore, this is a data-oblivious algorithm that does not rely on specific input, requires no tuning, and can be easily parallelized and applied in real-time.

The value cache quantization used to make step (2) memory efficient is known to be a straightforward task, and a standard token-wise quantization is very effective and efficient in practice, as observed in prior work (Liu et al. 2024b; Hooper et al. 2024). Hence, we follow the same approach for the value therein.

Furthermore, we analyzed the distribution of outliers in large language models (LLMs). We observed that while there are no significant outliers in the initial layers, certain fixed key embedding channels (coordinates) in the deeper layers exhibit considerably larger magnitudes (see Figure 2). To address this, we identify these outlier channels during the prompt phase and simply apply two independent copies of our quantizer to the outliers and inliers separately.

The QJL transform and its accompanying inner product estimator are highly efficient and GPU-friendly algorithms. In particular, we provide a lightweight CUDA kernel for their efficient computation. We apply QJL and our inner product estimator to compress the KV cache in several LLMs, including Llama-2 (Touvron et al. 2023) and its fine-tuned models by long sequence (Li et al. 2023), under various NLP tasks. Our results show that quantizing the KV cache to only 3 bits per FPN results in no accuracy drop compared to the exact model with 16 bits per FPN while reducing cache memory usage by over fivefold and increasing the generation speed significantly for long contexts. For example, our proposed quantization shows better F1 scores on longrange question-answering tasks from LongBench (Bai et al. 2023) (a collection of long-context datasets) compared to the recent KV cache quantization methods, while minimizing memory overheads.

# Related Work

Improving the inference efficiency of LLMs requires minimizing memory transactions, as data transfer between memory and compute elements is significantly slower than computation itself. Notably, FlashAttention (Dao 2023) significantly accelerates the prefill phase of LLMs by reducing memory transactions. Additionally, approximation methods leveraging sparsity (Zandieh et al. 2023; Han et al. 2023) have been proposed to enhance prefill speed further.

Two effective direction for reducing the KV cache size in the generation phase are based on quantization and dimensionality reduction. Quantization and sketching methods have been extensively studied for compressing highdimensional data in various applications. One prominent family of methods relies on the Johnson-Lindenstrauss (JL) transform, which reduces dimensionality while approximately preserving pairwise distances. Quantized variants of JL transform have been developed to further reduce memory requirements, as demonstrated in works (Plan, Vershynin, and Yudovina 2017) and (Matsumoto and Mazumdar 2024), which apply 1-bit compressed sensing techniques in different domains. Sampling methods have also been proposed for sketching inner products, offering strong theoretical guarantees (Daliri et al. 2024). Vector quantization techniques have also been widely explored, particularly in the context of vector search, where they are used to compress indices for efficient similarity search. Recent work (Gao and Long 2024) introduces a novel approach to 1-bit index quantization using random rotations. Their method not only achieves efficient compression but also provides an unbiased estimator for inner products.

# Preliminaries: Token Generation in Attention

Deploying auto-regressive language models for inference involves performing attention decoding in an online setting, where key and value embeddings from each transformer layer are cached in memory to remove redundant computations. The model sequentially uses and updates the KV cache to generate the next token, one at a time.

![](images/8add1a00fe75d04e2d4d22bd31421d9eefd62431ce600d3ec8538f885f53a0c3.jpg)  
Figure 1: Overview of the KV cache quantization via Quantized JL (QJL) transform

More precisely, in every phase of token generation, the stream of tokens is represented by a triplet of vectors called by the query, key, and value embeddings, respectively. Let $\dot { \mathbf { q } _ { i } } , \pmb { k } _ { i } , \bar { \mathbf { v } _ { i } } \in \mathbf { \mathbb { R } } ^ { d }$ be the triplet at $i$ -th generation phase and $n$ be the total number of tokens in the stream so far either in the prompt encoding (prefill) or the generation (decoding) phase. Then, the attention output in $n$ -th generation phase can be written as

$$
\bullet _ { n } = \sum _ { i \in [ n ] } \operatorname { S c o r e } ( i ) \cdot \pmb { v } _ { i } ,
$$

where Score $\in \mathbb { R } ^ { n }$ is the vector of attention scores defined as:

$$
\mathtt { S c o r e : = s o f t m a x \left( \right)} [ \langle q _ { n } , k _ { 1 } \rangle , \langle q _ { n } , k _ { 2 } \rangle , \ldots \langle q _ { n } , k _ { n } \rangle ]  .
$$

The output embedding $\scriptstyle { o _ { n } }$ will be used for computing the next tokens in the stream $\mathbf { q } _ { n + 1 } , \mathbf { k } _ { n + 1 } , \mathbf { v } _ { n + 1 }$ unless the generation phase terminates. Observe that to compute output $\scriptstyle { o _ { n } }$ , one needs to store all previous key and value embeddings $\{ k _ { i } , { \pmb v } _ { i } \} _ { i \in [ n ] }$ and keeping them in full precision requires significant memory for long-context inputs. The time complexity to compete Equation (2) is $O ( n d )$ due to the computation of $n$ inner products. Additionally, the inference speed is also impacted by the KV cache size, as the KV cache must be loaded from GPU main memory for every token generated, resulting in low arithmetic intensity and underutilization of GPU cores (Pope et al. 2023). In this work, we focus on compressing the KV cache by quantizing tokens, thereby reducing the memory required to store each key or value

embedding in the cache.

# Quantized Johnson-Lindenstrauss (QJL)

Our goal is to save memory space for storing the KV cache while the inner product between query and key remains undistorted. To achieve this, we first transform the embedding vectors using a random projection that preserves the inner products, acting as a preconditioning step, and then quantize the result. Specifically, we project the input vectors onto a random subspace by applying the Johnson-Lindenstrauss (JL) transform (Johnson, Lindenstrauss, and Schechtman 1986), which amounts to multiplying by a random Gaussian matrix. The inner product of the resulting vectors after applying this projection provides an unbiased and low-distortion estimator for the inner product of the original vectors (Dasgupta and Gupta 2003). We introduce a 1-bit Johnson-Lindenstrauss transform, comprising a JL transformation followed by quantization to a single sign bit, and demonstrate its ability to offer an unbiased and low-distortion inner product estimator. We complement our binary quantizer by developing an unbiased estimator for the inner product of the quantized vector with any arbitrary vector. This inner product estimator is asymmetric, as one of the vectors is quantized to a single bit while the other remains unquantized, making it well-suited for the KV cache mechanism. The Quantized Johnson-Lindenstrauss (QJL) transformation, acting as a 1-bit quantizer, alongside our proposed estimator, is formally defined in the following definition:

Definition 1 (QJL and inner product estimator). For any positive integers $d , m$ , let $\pmb { S } \in \mathbb { R } ^ { m \times d }$ be a $\scriptstyle { \mathrm { J L } }$ transform matrix, i.e., entries of $\boldsymbol { s }$ are i.i.d. samples from the zero mean and unit variance Normal distribution. The QJL is a mapping function $\mathcal { H } _ { S } : \mathbb { R } ^ { d }  \{ - 1 , + 1 \} ^ { m }$ defined as:

$$
\begin{array} { r } { \mathcal { H } _ { S } ( k ) : = \mathrm { s i g n } ( S k ) \quad \mathrm { f o r ~ a n y } \ k \in \mathbb { R } ^ { d } . } \end{array}
$$

Furthermore, for any pair of vectors $\pmb { k } , \pmb { q } \in \mathbb { R } ^ { d }$ the estimator for their inner product $\langle q , k \rangle$ based on the aforementioned quantizer is defined as:

$$
\operatorname* { P r o d } _ { \mathbb { Q } \operatorname { J L } } ( \pmb q , \pmb k ) : = \frac { \sqrt { \pi / 2 } } { m } \cdot \| \pmb { k } \| _ { 2 } \cdot \langle S \pmb q , \mathcal H _ { S } ( \pmb k ) \rangle .
$$

Now, we show that the inner product estimator $\operatorname { P r o d } _ { \mathbb { Q } \mathrm { J L } } ( q , k )$ , exactly like the inner product of JLtransformed vectors without quantization to sign bit, is an unbiased estimator. The crucial point to note is that if we applied QJL to both vectors $\pmb q$ and $k$ in Equation (4), we would obtain an unbiased estimator for the angle between these vectors, as shown in (Charikar 2002). However, to estimate the inner product one needs to apply the cosine function on top of the angle estimator, which results in a biased estimation. Thus, to achieve an unbiased inner product estimator, it is necessary to asymmetrically apply quantization to the JL transform of only one of the vectors $\pmb q$ and $k$ .

Lemma 2 (Inner product estimator $\mathtt { P r o d } _ { \mathtt { Q J L } }$ is unbiased). For any vectors $\pmb q , \pmb k \in \mathbb { R } ^ { d }$ the expected value of the estimator $\mathrm { P r o d } _ { \mathrm { Q J L } } ( q , k )$ defined in Equation (4) is:

$$
\begin{array} { r } { \frac { \mathbb { E } } { s } [ \mathrm { P r o d } _ { \mathbb { Q } , \mathbf { k } } ( \pmb q , \pmb k ) ] = \langle \pmb q , \pmb k \rangle , } \end{array}
$$

where the expectation is over the randomness of the JL matrix $\boldsymbol { s }$ in Definition $\jmath$ .

The complete proof is provided in the extended version of this paper (Zandieh, Daliri, and Han 2024).

Now we show that the inner product estimator $\mathtt { P r o d } _ { \mathtt { Q J L } }$ in Definition 1, just like the estimators based on the standard JL transform, has a bounded distortion with high probability.

Lemma 3 (Distortion of inner product estimator $\mathtt { P r o d } _ { \mathtt { Q J L } } ,$ ). For any vectors $\ b q , \ b k \in \mathbb { R } ^ { d }$ if the estimator $\mathrm { P r o d } _ { \mathrm { Q J L } } ( q , k )$ is defined as in Equation (4) for $\boldsymbol { Q } \boldsymbol { J } \boldsymbol { L }$ with dimension $m \geq$ 4 · 1+2 log 2 , then:

$$
\operatorname* { P r } _ { S } \left[ | \operatorname* { P r o d } _ { \mathbb { Q } , \mathbf { k } } ( \pmb { q } , \pmb { k } ) - \langle \pmb { q } , \pmb { k } \rangle | > \varepsilon \| \pmb { q } \| _ { 2 } \| \pmb { k } \| _ { 2 } \right] \leq \delta ,
$$

where the probability is over the randomness of the JL matrix $\boldsymbol { s }$ in Definition $\jmath$ .

The complete proof is provided in the extended version of this paper (Zandieh, Daliri, and Han 2024).

Note that the distortion bound in Lemma 3 has remarkably small constants, even smaller than those of the original unquintized JL transform. This indicates that quantizing one of the vectors to just a single sign bit does not result in any loss of accuracy. We use these properties of QJL and our inner product estimator to prove the final approximation bound on our KV cache quantizer.

# Key Cache Quantization via QJL

The key cache is used in the computation of attention scores as shown in Equation (2). To calculate these scores, we need to compute the inner products of the current query embedding with all key embeddings in the cache. We design a quantization scheme that allows for a low-distortion estimate of the inner products between an arbitrary query and all keys in the cache. In this section, we develop a practical algorithm with provable guarantees based on QJL and the inner product estimator defined in Definition 1.

Algorithm 1: QJL Key Cache Quantizer   

<html><body><table><tr><td>Input: Stream of key tokens k1,k2,... ∈ Rd,integer m 1: Draw a random sketch S ∈ Rmxd with i.i.d. entries Si,j ~ N(O,1) as per Definition 1 2: repeat 3: Compute ki ← sign(Ski) and Vi ← ||kill2 4: store the quantized vector ki and the key norm Vi in thecache</td></tr><tr><td>5: until token stream ends Procedure ESTIMATESCORES(qn) 1 /元/2 m (Sqn,kj) for every j ∈[n] 7: Score ← softmax (qK 6: Compute inner product estimators qK(j） ← Vi</td></tr></table></body></html>

The quantization scheme presented in Algorithm 1 applies QJL, defined in Definition 1, to each key embedding, mapping them to binary vectors and storing the results in the key cache. We show in the following theorem that the attention scores calculated by Algorithm 1 have very small $( 1 \pm \varepsilon )$ relative distortion with high probability:

Theorem 4 (Distortion bound on QJL key cache quantizer). For any sequence of key tokens $\bar { \pmb { k } _ { 1 } } , \dotsc \bar { \pmb { k } _ { n } } \in \mathbf { R } ^ { \bar { d } }$ and any integer m, Algorithm 1 stores binary vectors $\tilde { k } _ { 1 } , \dots \tilde { k } _ { n } \ \in$ $\{ - 1 , + 1 \} ^ { m }$ along with scalar values $\nu _ { 1 } , \ldots \nu _ { n }$ in the cache. If the key embeddings have bounded norm $\mathrm { m a x } _ { i \in [ n ] } \| { \pmb { k } } _ { i } \| _ { 2 } \leq$ $r$ and $m \ge 2 r ^ { 2 } \varepsilon ^ { - 2 } \log n ,$ , then for any query embedding ${ \bf q } _ { n } ~ \in ~ { \bf R } ^ { d }$ with bounded norm $\| \pmb { q } _ { n } \| _ { 2 } ~ \le ~ r$ the output of the procedure ESTIMATESCORES $( q _ { n } )$ satisfies the following with probability poly1(n) sinultaneously for all i ∈ [n]:

$$
\begin{array} { r } { \Big | \widetilde { \mathsf { S c o r e } } ( i ) - \mathsf { S c o r e } ( i ) \Big | \le 3 \varepsilon \cdot \mathsf { S c o r e } ( i ) , } \end{array}
$$

where Score is the vector of attention scores defined in Equation (2).

The complete proof is provided in the extended version of this paper (Zandieh, Daliri, and Han 2024).

This theorem shows that if the query and key embeddings have constant norms, as is common in practical scenarios, we can quantize each key embedding such that only $m \approx$ $\varepsilon ^ { - 2 } \log { n }$ bits are needed to store each key token. This is independent of the embedding dimension of the tokens and scales only logarithmically with the sequence length.

# Value Cache Quantization

We quantize the value cache using a standard quantization method, i.e., normalizing each token’s entries and then rounding each entry to a few-bit integer representation. This approach aligns with prior work, which has shown that standard token-wise quantization is highly effective for the value cache and results in a minimal accuracy drop (Liu et al. 2024b; Hooper et al. 2024).

# Experiments

In this section, we validate the empirical performance of our algorithm. All experiments are conducted under a single A100 GPU with 80GB memory. We implement two main CUDA kernels for our core primitives: one for quantizing embedding vectors using various floating point data types such as bfloat16, FP16, and FP32, and the other for computing the inner product of an arbitrary embedding vector with all quantized vectors in the cache. The algorithm’s wrapper is implemented in PyTorch, handling all the housekeeping tasks. We plan to complete implementation in the CUDA for future work, which will further accelerate our algorithm.

# Practical Consideration

Outliers. As reported in recent works e.g., KIVI (Liu et al. 2024b), KVQuant (Hooper et al. 2024), key embeddings typically contain outliers exhibiting a distinct pattern. Specifically, certain coordinates of key embeddings display relatively large magnitudes. To further investigate these observations, we analyze the distribution of the magnitudes of key embedding coordinates across different layers. Firstly, we observe that there are no significant outliers in the initial attention layers. However, in the deeper layers, certain fixed coordinates of key embeddings consistently exhibit large magnitudes, and this pattern persists within these channels across all tokens. The distribution of outliers across different layers for the Llama-2 model is plotted in Figure 2. It is evident that in the initial layers, outliers are rare, but as we approach the final layers, their frequency and impact increase significantly. Secondly, the outliers show a persistent pattern in specific fixed coordinates of the key embeddings. This observation aligns with previous findings that certain fixed embedding coordinates exhibit larger outliers (Dettmers et al. 2022; Lin et al. 2023; Liu et al. 2024b; Hooper et al. 2024).

As demonstrated in Theorem 4, the distortion on the attention scores is directly proportional to the norms of the embeddings. Therefore, capturing these outlier coordinates is essential, as their large magnitudes contribute significantly to the norms of key embeddings. By identifying and isolating these outlier channels, we can reduce the norm of the key embeddings and, consequently, significantly decrease the final distortion. Next, we quantize the outliers using an independent instance of our QJL quantizer but with a lower compression rate, utilizing more bits to accurately represent each outlier coordinate.

Orthogonalized JL transform. We observed that orthogonalizing the rows of the $\mathrm { J L }$ matrix $S$ in Definition 1 almost always improves the performance of our QJL quantizer. This finding aligns with previous work on various applications of the JL transform, such as random Fourier features (Yu et al. 2016) and locality sensitive hashing (Ji et al. 2012). Consequently, in our implementation and all experiments, we first generate a random $\scriptstyle { \mathrm { J L } }$ matrix $S$ with i.i.d. Gaussian entries and then orthogonalize its rows using QR decomposition. We then use this orthogonalized matrix in our QJL quantizer, as described in Algorithm 1.

# Ablation Study

Here, we perform an ablation study on the relative distortion of the attention scores in one attention layer after applying QJL on key embeddings. The distortion for various layers of the Llama2-7B model is plotted against the number of bits per token and embedding channel $m / d$ , where $d = 1 2 8$ is the embedding dimension, as shown in Figure 3. Our theoretical result from Theorem 3.6 suggests that $m \sim 1 / \varepsilon ^ { 2 }$ which aligns with our observations in Figure 3. An interesting observation is that the first layer has a much higher distortion compared to all other layers, suggesting that the first layer is more challenging to quantize and requires a higher number of bits per FPN. This finding is noteworthy and indicates the need for tailored quantization strategies for different layers. This is consistent with the outlier distribution depicted in Figure 2, where the first layer appears distinct from the others.

# End-to-End Text Generation

Next we benchmark our method on LongBench (Bai et al. 2023), a benchmark of long-range context on various tasks. We choose the base model as longchat-7b-v1.5-32k (Li et al. 2023) (fine-tuned Llama-2 with 7B parameter with 16,384 context length) and apply following quantization methods to this model; KIVI (Liu et al. 2024b), KVQuant (Yue et al. 2024) and our proposed quantization via QJL. Each floatingpoint number (FPN) in the base model is represented by 16 bits, and we choose proper hyper-parameters of KIVI and QJL so that their bits per FPN become 3. For KVQuant, we follow the default setting which holds its bits per FPN as 4.3.

QJL evaluation on Llama2/Llama3 model and LongBench dataset. We benchmarked QJL on LongBench (Bai et al. 2023), a suite of tasks designed to evaluate performance with long-range contexts. We choose the base model as longchat-7b-v1.5-32k (Li et al. 2023) (fine-tuned Llama-2 with 7B parameter with 32,768 context length) and Llama3.1-8B-Instruct (Team 2024b) and apply following quantization methods to this model; KIVI (Liu et al. 2024b), KVQuant (Hooper et al. 2024) and our proposed QJL. Each floating-point number (FPN) in the base model is represented by 16 bits. We chose several hyperparameters for QJL to match the bits per FPN of the competing methods KVQuant and KIVI. There are two versions of KIVI, with bits per FPN of 3 and 5, respectively. For KVQuant, the default setting results in 4.3 bits per FPN. To validate the quality of those quantized models, we benchmark them on 6 question-answer datasets from LongBench, and we set the maximum sequence length to 31,500. We follow the same approach of prompting and evaluating to evaluate the prediction of the model from the original repository. Table 1 summarizes the results. Our proposed QJL achieves the highest score within the quantization methods for NarrativeQA, Qasper and 2WikiMultiQA.

Experiments with Llama3 and Llama2 models. We additionally test our method on datasets Lambada-OpenAI,

![](images/9ca79fbf6e98a704e0a69969fbc9b17abaeb9314d67090551a7b9ea8df9e58db.jpg)  
Figure 2: The magnitude of key cache entries for different layers of the Llama-2 model, for an example prompt, reveals notable patterns. Channels are sorted by their average magnitudes. In initial layers, no significant outliers are observed. However, in the deeper layers, few channels (approximately four) exhibit visibly larger magnitudes (outliers), highlighting the importance of addressing these outliers to improve quantization accuracy for the key cache.   
Figure 3: The mean relative distortion square on the attention scores $\varepsilon$ vs. the number of bits of QJL per token and embedding channels, i.e., $m / d$ , for layers at different depths of Llama 2 model.

<html><body><table><tr><td rowspan="3">Methods</td><td rowspan="3">Bits</td><td colspan="8">Datasets from LongBench (Bai et al.2023)</td></tr><tr><td>NarrativeQA</td><td>Qasper</td><td>MultiQA-en</td><td></td><td>MultifQA-zh</td><td>HotpotQA</td><td>2WikiMultiQA</td><td></td></tr><tr><td></td><td>20.79</td><td>29.42</td><td>42.83</td><td>34.33</td><td></td><td>33.05</td><td>24.14</td></tr><tr><td>FP16 (Longchat-7b-v1.5-32k) KIVI (Liu et al. 2024b)</td><td>16 3</td><td colspan="2">20.96</td><td></td><td></td><td></td><td></td><td colspan="2"></td></tr><tr><td>QJL (ours)</td><td>3</td><td colspan="2">20.67</td><td>29.01 28.48</td><td>40.93 40.94</td><td>34.75 29.71</td><td>32.79 35.62</td><td colspan="2">23.01 23.60</td></tr><tr><td rowspan="2">KVQuant (Hooper et al.2024)</td><td></td><td colspan="2"></td><td></td><td></td><td colspan="2"></td><td colspan="2"></td></tr><tr><td>4.3 4.3</td><td colspan="2">20.14</td><td>28.77 30.02</td><td>44.22 41.18</td><td>34.44 31.73</td><td>34.06 34.22</td><td colspan="2">23.05</td></tr><tr><td>QJL (ours) KIVI (Liu et al. 2024b)</td><td>5</td><td colspan="2">20.72</td><td></td><td></td><td></td><td>33.07</td><td colspan="2">22.63</td></tr><tr><td rowspan="2">QJL (ours)</td><td>5</td><td colspan="2">20.49 21.09</td><td>28.90</td><td>43.24</td><td colspan="2">34.66</td><td colspan="2">24.86 24.61</td></tr><tr><td></td><td colspan="2">29.11</td><td colspan="2">41.58</td><td colspan="2">31.86</td><td colspan="2"></td></tr><tr><td rowspan="2">Methods</td><td>Bits</td><td>Qasper</td><td>MultiNews</td><td></td><td>Datasets from LongBench (Bai et al.2023)</td><td></td><td></td><td></td><td rowspan="2">Avg.</td></tr><tr><td>Qmsum 25.22</td><td></td><td></td><td>TREC 73.5</td><td>TriviaQA 91.79</td><td>SAMSum 43.87</td><td>LCC 62.99</td><td>RepoBench-P 56.39</td></tr><tr><td>BP16 (Llama-3.1-8B-Instruct) QJL (ours)</td><td>16 3</td><td></td><td>44.98</td><td>26.99</td><td></td><td>88.55</td><td></td><td></td><td>53.47</td></tr><tr><td>QJL (ours)</td><td>4</td><td>23.68 23.40</td><td>42.76 43.09</td><td>26.82 27.02</td><td>72.5 72.5</td><td>43.43 89.69 43.30</td><td>63.98 64.75</td><td>56.09 56.79</td><td>52.48 52.82</td></tr></table></body></html>

Table 1: Evaluation on long-context tasks from LongBench. The top table corresponds to the longchat-7b-v1.5-32k model (fine-tuned Llama-2 with 7B parameter with 32k context) and the bottom table corresponds to Llama-3.1-8B-Instruct model.

Layer 0 Layer 8 Layer 23 Layer 4 Layer 15 Layer 31   
10   
MSE Distortion (ϵ) ( m 二. 10 10 6 8 Bits per Channel (m)

HellaSwag, PIQA, MathQA, and MMLU, which have shorter sequence lengths. We benchmark our method using LM-eval (Gao et al. 2023) framework to ensure a thorough evaluation across various metrics. We evaluate quantization methods with accuracy across Llama-2-7B (Touvron et al.

2023) and Llama-3-8B (Team 2024b) models. Note that KIVI only supports a half-precision floating point, whereas our method can be used for any precision format type. This makes it unable to run KIVI on the Llama-3 model.

As we observe, QJL can significantly reduce memory usage by utilizing only 3 bits per FPN, compared to the 16 bits per FPN in the baseline, achieving around an $81 \%$ reduction in memory. We observe that this efficiency does not compromise performance significantly. Across all datasets, our method’s accuracy is generally comparable to the baseline, with slight variations. In Table 2, our QJL on the Llama-3-8B performs on average about slightly better than the baseline across all datasets.

Runtime and Peak-Memory Evaluations. To evaluate the runtime and memory consumption of QJL we additionally report runtimes of: (1) prompt encoding, (2) KV cache quantization, and (3) decoding (token generation) in a single attention layer as well as the (4) peak memory consumption during prompt encoding and decoding. Figure 4 shows the wall-clock time to encode a prompt and quantize the KV cache, generate 128 tokens for Llama2 model, and generate 64 tokens for Llama3 model using different quantization

![](images/8f592c59daeb3c01b1fe64135ae422900713e1cf12818f8cd33e2cb95f085848.jpg)  
Figure 4: Wall-clock time to encode a prompt and quantize the KV cache (left), generate 128 tokens for llama2 model (middle), and generate 64 tokens for llama3 model (right) using different quantization methods in a single attention layer. The input sequence length varies from 1k to 64k. Both KIVI and QJL (ours) with 3 bits per FPN show faster decoding time than the baseline. However, KVQuant is significantly slower during both quantizing and decoding phases. QJL is the only method that can quantize Llama3, as our kernels support grouped query attention and BF16 data type. We observe the same speed for Llama3 as the exact method for generation. Our memory usage is at least 5-fold less than the exact method and can support all data types.

<html><body><table><tr><td rowspan="2">Models</td><td rowspan="2">Methods</td><td rowspan="2">Bits</td><td colspan="5">Datasets from LM-eval (Gao et al.2023)</td></tr><tr><td>Lambada-OpenAI</td><td>HellaSwag</td><td>PIQA</td><td>MathQA</td><td>MMLU</td></tr><tr><td rowspan="3">Llama-2-7B</td><td>FP16 (baseline)</td><td>16</td><td>73.90</td><td>57.18</td><td>78.07</td><td>28.11</td><td>41.85</td></tr><tr><td>KIVI (Liu et al. 2024b)</td><td>3</td><td>73.88</td><td>57.13</td><td>78.07</td><td>28.11</td><td>41.81</td></tr><tr><td>QJL (ours)</td><td>3</td><td>73.88</td><td>57.14</td><td>78.07</td><td>28.17</td><td>41.78</td></tr><tr><td rowspan="2">Llama-3-8B</td><td>BF16 (baseline)</td><td>16</td><td>75.59</td><td>60.17</td><td>79.65</td><td>40.64</td><td>62.09</td></tr><tr><td>QJL (ours)</td><td>3</td><td>75.61</td><td>60.13</td><td>79.87</td><td>40.60</td><td>62.12</td></tr></table></body></html>

Table 2: Accuracy of quantization methods on standard-length datasets from LM-eval (Gao et al. 2023) shows our 3-bit QJL performing comparably to the 16-bit baseline, even without long-context focus.

![](images/4d74f47fecbda98b6ad6713ec03e9c3244bc630066ce13db22bcf617ef5a3cf0.jpg)  
Figure 5: Peak memory usage for prompt encoding and generating 128 tokens with Llama2.

methods in a single attention layer of these models. Note that QJL is the only method that can quantize Llama3, as our kernels support grouped query attention and BF16 data type. we observe the same speed for Llama3 as the exact method for generation. The input sequence lengths vary between 1k to $1 2 8 \mathbf { k }$ . As shown in Figure 4, KVQuant runs slower than other methods during both prompt encoding and decoding phases, as it requires a huge amount of preprocessing which leads to slow runtime. On the other hand, both KIVI and our QJL with 3 bits per FPN show marginal runtime overhead compared to the exact baseline during prompting but reduce KV cache memory usage by at least a factor of 5.

Next, we compare the peak memory consumption of various KV cache quantization methods applied to the Llama2 model for encoding prompts of different lengths and generating 128 new tokens, as shown in Figure 5. Both QJL and KIVI quantize the KV cache to 3/5 bits per FPN. However, peak memory consumption also includes the memory required to store model parameters. Even considering total memory consumption, we observe an over two-fold reduction in peak memory usage. We did not include KVQuant in the peak memory study as this method was extremely slow.