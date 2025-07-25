# CSR:Achieving 1 Bit Key-Value Cache via Sparse Representation

Hongxuan Zhang1, 2\*†, Yao Zhao2†, Jiaqi Zheng1, Chenyi Zhuang2, Jinjie $\mathbf { G } \mathbf { u } ^ { 2 }$ , Guihai Chen1

1Nanjing University 2Ant Group x zhang $@$ smail.nju.edu.cn, nanxiao.zy $@$ antgroup.com, jzheng@nju.edu.cn, {chenyi.zcy, jinjie.gujj} $@$ antgroup.com, gchen@nju.edu.cn

# Abstract

The emergence of long-context text applications utilizing large language models (LLMs) has presented significant scalability challenges, particularly in memory footprint. The linear growth of the Key-Value (KV) cache, which stores attention keys and values to reduce redundant computations, can significantly increase memory usage and may prevent models from functioning properly in memory-constrained environments. To address this issue, we propose a novel approach called Cache Sparse Representation (CSR), which converts the KV cache by transforming the dense Key-Value cache tensor into sparse indexes and weights, offering a more memory-efficient representation during LLM inference. Furthermore, we introduce NeuralDict, a novel neural networkbased method to automatically generate the dictionary used in our sparse representation. Our extensive experiments demonstrate that CSR matches the performance of state-of-theart KV cache quantization algorithms while ensuring robust functionality in memory-constrained environments.

Cache Sparse Representation Decomposition 2 767 ≈ 0.4× + 0.7×   
Dictionary in Transformer Layer ≈ 0.3× + 0.4× 𝑥 873 Extract Offline Part by Layer Index Sparse Storage 0.4 0.7 Coefficients: 0.3 0.4 FP16 0 1 2 3 ... 764 ... 864 Index   
… … Indexes: 1 786773 INT16 Basis Vector Restoration 0.4× + 0.7× = 2 767 Collect Online Part 0.3× + 0.4× = during Inference 873 𝑥"

# Introduction

The introduction of large language models (LLMs) has brought about a new wave of exciting AI applications, including document summarization, code analysis, extended multi-turn applications, tool learning, and more. Among these applications, those involving long text have garnered significant interest, such as RAG (Retrieval-Augmented Generation). RAG tackles the challenge of generating accurate and pertinent content, particularly in scenarios where queries extend beyond the training data or require up-todate knowledge, by integrating external information sources. This fusion of RAG with LLMs expands the scope of LLMs and makes them increasingly applicable for specialized and knowledge-driven tasks in real-world contexts. However, the significant number of parameters in LLMs, amounting to tens or hundreds of billions, results in high memory and computation requirements during generation tasks, especially when handling long contexts like RAG. To effectively support large language models (LLMs), it is crucial to batch multiple requests together to minimize the cost per request.

The key-value (KV) cache utilized to store attention keys and values, and thereby avoid redundant computations, can lead to a substantial increase in memory usage and become a bottleneck for both speed and memory efficiency. The cache’s memory consumption grows linearly with the length of the input prompt and the number of tokens generated, overshadowing even the substantial memory requirements of the model parameters. This presents scalability challenges for large language models, as the cache’s linearly expanding memory footprint hinders its ability to handle longer sequences. Therefore, it is imperative to develop methods for compressing the KV cache to enable long-sequence inference in a memory-efficient manner.

We provide an overview of existing methods that help mitigate KV cache overhead as follows:(Shazeer 2019) introduces Multi-Query Attention, a variant of Multi-Head Attention (MHA)(Vaswani et al. 2017). MQA allows different attention heads to share the same KV caches, thereby reducing memory usage. Furthermore, (Ainslie et al. 2023) introduces Grouped-Query Attention (GQA), which strikes a balance between the performance degradation caused by MQA and the memory footprint associated with Multi

Head Attention (MHA). Recent works like YOCO(2024), which reuses global KV caches via cross-attention, and MLA(2024), which compresses KV caches into latent vectors, both improve memory efficiency too. These modifications to the attention mechanism also contribute to a reduction in the memory footprint of the KV cache. Another set of techniques utilizes quantization to reduce the number of bits used by the original stored data type, sacrificing data precision for memory footprint. Additionally, some researchers have taken an alternative approach by lowering the memory footprint of the KV cache through the eviction of unimportant cache parts from the GPU. We will delve into a detailed discussion of these two methods in Related Work.

In this paper, we propose CSR (Cache Sparse Representation), which offers a sparse representation of the KV cache and provides an equivalent but less memory-intensive representation for the original KV cache during LLM inference. Our contributions are outlined as follows:

1. CSR provides an innovative approach to reducing the high memory footprint of the KV cache in long-text LLM applications. It works across different attention mechanisms in transformers while remaining independent of established methods such as KV cache eviction.   
2. Our extensive experiments on various models and datasets demonstrate that CSR performs comparably to 4-bit or 2-bit KV cache quantization algorithms under conditions with sufficient memory, while also maintaining robust performance with less than 1 bit per channel in memory-constrained environments.

# Preliminary

# Sparse Representation

Sparse representation is a well-researched fields in computer vision and pattern recognition. However, to the best of our knowledge, no work yet try using sparse representation to reduce the memory footprint used during inference of LLM. Suppose we have a dictionary $D = [ \mathbf { d _ { 1 } } , \mathbf { \bar { d } _ { 2 } } , . . . , \mathbf { d _ { N } } ] \in$ Rd×N and each basis vector dn in D is an l2-norm unity vector. Define the sparsity of a representation vector $\mathbf { r }$ as the $l _ { 0 }$ norm of $\mathbf { r }$ , which means the number of the nonzero elements of vector $\mathbf { r }$ . Given dictionary $D$ and limit maximum representation sparsity as $s$ , for a dense origin vector $\mathbf { x } \in \mathbb { R } ^ { \dot { d } }$ , the way to find the sparse representation of $\mathbf { x }$ is solving the following optimization problem:

$$
\mathbf { r } ( \mathbf { x } , D , s ) = \arg \operatorname* { m i n } \| \mathbf { x } - D \mathbf { r } \| ^ { 2 } \quad \mathrm { s . t . } \quad \| \mathbf { r } \| _ { 0 } \leq s
$$

where ${ \bf r } \in \mathbb { R } ^ { N }$ means sparse representation of $\mathbf { x }$ with sparsity no greater than $s$ . Among different types of algorithms for solving equation (1), Matching Pursuit(MP) (Mallat and Zhang 1993) is the most widely used one to generate sparse representation satisfying sparsity limitations. MP iteratively choose the best atom from the dictionary based on a certain similarity measurement to approximately obtain the sparse solution. First of all, the residual vector is initialized as ${ \bf R } _ { 0 } = { \bf x }$ , $\mathbf { r } = \mathbf { 0 } \in \mathbb { R } ^ { d }$ . The MP algorithm will determine the optimal atom vector index $i _ { g }$ and the corresponding coefficient $c _ { g }$ through the following formulas:

$$
\begin{array} { c } { { \displaystyle c _ { g } = s u p \left. { \bf R } _ { g } \cdot { \bf d } _ { n } \right. } } \\ { { \displaystyle i _ { g } = \arg \operatorname* { m a x } _ { n } \left. { \bf R } _ { g } \cdot { \bf d } _ { n } \right. } } \end{array}
$$

where $1 \leq g \leq s$ represents the number of current iterations. Subsequently, update $\mathbf { r } [ i _ { g } ] = c _ { g }$ , and the residual vector $R _ { g }$ is updated based on the part that was already approximated in the previous iteration as following:

$$
\mathbf { R } _ { g + 1 } = \mathbf { R } _ { g } - c _ { g } \times \mathbf { d } _ { i _ { g } }
$$

MP will repeat calculating (2)-(4) until $c _ { s }$ and $i _ { s }$ are calculated and $\| \mathbf { r } \| _ { 0 } = s$ exactly.

# KV Cache in Attention

LLM inference can be divided into the prefill phase and the decoding phase. In the prefill phase, each token of the input prompt is used to generate KV cache for every transformer layer of LLMs. The model uses and updates the KV cache to generate the next token autoregressively in the decoding phase. Since the KV cache mechanism for different attention heads is the same, we will not consider the attention head index in the subsequent discussion.

Assuming a model’s hidden size is $d$ and the number of key (or value) attention heads is $h$ , let $X _ { p } ^ { \lambda } \in \mathbb { R } ^ { b \times l \times h \times d _ { h } }$ represent the activations of the input prompt $p$ ’s tokens after being forwarded into transformer layer $\lambda$ , where $b$ is batch size, $l$ is the length of prompt tokens, and $d _ { h } = d / / h$ is the tensor size for each attention head. $W _ { K } ^ { \lambda }$ , $W _ { V } ^ { \lambda } \in \mathbb { R } ^ { \dot { d } \times d }$ of the current layer will map $X _ { p } ^ { \lambda }$ to key and value cache through the following equation:

$$
X _ { p , \{ K , V \} } ^ { \lambda } = X _ { p } ^ { \lambda } W _ { \{ K , V \} } ^ { \lambda }
$$

Here, $\lambda$ is the transformer layer index. $X _ { p , K } ^ { \lambda } , ~ X _ { p , V _ { . } } ^ { \lambda }$ pλ,V are cached in the memory as KV cache of prompt $p$ for layer $\lambda$ (Here we temporarily ignore the effect of position embedding). During the autoregressive decoding phase, each forward pass generates a new token $t$ , and its corresponding activations after passing through layer $\lambda$ are represented as $X _ { t } ^ { \lambda } \in \mathbb { R } ^ { b \times 1 \times d }$ . After being mapped to the key (K) and value (V space using $W _ { K } ^ { \lambda }$ and $\mathbf { \bar { \boldsymbol { W } } } _ { V } ^ { \lambda }$ , the corresponding $X _ { t , K } ^ { \lambda }$ and $X _ { t , V } ^ { \lambda }$ are appended to the KV cache of layer $\lambda$ . Throughout the remainder of the paper, we will use X{λK,V } to refer to the $\mathrm { \sf ~ K }$ or $\mathrm { \Delta V }$ cache space of the transformer layer $\lambda$ .

# KV Cache Sparse Representation:CSR

In this section, we introduce our method, Cache Sparse Representation (CSR), which utilizes the dictionary that fully extracts KV cache features and replaces dense KV cache vectors with sparse indexes and coefficients to significantly reduce the memory footprint during inference. We initially present our intuitions collected during the LLM inference stage, which directly guide the dictionary construction of CSR. Subsequently, we provide a comprehensive overview of the CSR procedure and delve into the detailed process of constructing the dictionary required by CSR.

Key in Layer 5 Value in Layer 5 5 Key in Layer 30 Value in Layer 30 PCA C1 PCA C1 (a) Distribution analysis of $X _ { K }$ and $X _ { V }$ across different promp Key Cache before RoPE Key Cache after RoPE 0 PCA C1 PCA C1 (b) Comparison of $X _ { K }$ with and without RoPE processing.

Figure 2: (a) The distribution of $X _ { K }$ among prompts is nearly identical. While there is substantial spatial overlap in $X _ { V }$ in the shallow layer, few noticeable differences still emerge in the deep layers (e.g., Layer 30). (b) $X _ { K }$ from the same layer is evenly segmented into 8 groups based on their positions. It is evident that, in comparison with the keys processed by RoPE, the keys that have not undergone RoPE processing are thoroughly intermingled, which is better for extracting offline basis vectors.

# Intuitions

We extracted a range of prompts from wikitext dataset(Merity et al. 2016), and forward them into Llama2- 7B-Chat which is a widely used LLM. Subsequently, we gathered the KV cache generated during model inference. To aid in subsequent observation and research, we reduced the collected KV cache to a two-dimensional space through PCA in the channel dimension. This allowed us to derive the following observations through analysis.

Difference among prompts is nearly ignorable. Following PCA dimensionality reduction to a two-dimensional space, we observe that the spaces covered by different prompts are nearly identical, as depicted in Figure 2a where different colors mean different prompts. This finding suggests that a portion of the constructed dictionary can be shared across different query prompts. We refer this query-independent part as the offline part. Note that few noticeable differences still exist in the deep transformer layers. We propose the online part to deal with this issue.

Position embedding makes nonstationary Keys. An important consideration in determining the sparse representation for Keys is managing the positional embedding, such as Rotary Positional Embedding (RoPE for short)(Su et al. 2024) which are applied to Keys and Queries in most LLMs to embed relative positional information between Keys and Queries. The nature of these positional embedding causes the Keys to be relatively unstable with regard to position, as depicted in Figure 2b. Due to this phenomenon, we opt to pre-process the Key cache of tokens before introducing the position embedding.

Adjacent transformer layers’ KV space is similar. To analyze the differences in $X _ { \{ K , V \} } ^ { \lambda }$ between Transformer layers, we first normalize the collected KV cache into $l _ { 2 }$ -norm unity vectors, then perform PCA in pairs on adjacent layers to reduce the dimension to 2. After that we generate a two-dimensional histogram of $2 0 0 \times 2 0 0$ bins to obtain the discrete distribution of $X _ { \{ K , V \} }$ . Finally, we measure the difference of KV cache space from two transformer layers by calculating the JS divergence between these discrete distributions, part of the results are shown in the figure 3. We observes that the distribution of $X _ { \{ K , V \} }$ between most adjacent layers is similar. So we decide to to construct a multi-layers shared offline dictionary based on the similarity between layers in order to save memory footprint as much as possible. Take $X _ { K }$ as an example to illustrate the heuristic idea we follow, the set of aggregated layers is denoted as $\mathcal { M } _ { K } = \{ \Lambda _ { 1 } , . . . \Lambda _ { i } \}$ , and $\boldsymbol { \Lambda _ { i } } \ = \ \{ \lambda _ { m } , . . . \lambda _ { n } \}$ . For $\forall i \neq j , \Lambda _ { i } \cap \Lambda _ { j } = \tilde { \emptyset }$ , and $\mathsf { U } _ { i } \Lambda _ { i }$ is the set of all transformer layers. Any transformer layer pair $\left( \lambda _ { m } , \lambda _ { n } \right)$ in the same $\Lambda _ { i }$ satisfies the following:

![](images/268eb46658bbda5758abc176f59d4fc8617dab994e75f4294a3459bde15abfad.jpg)  
Figure 3: JS divergence for $X _ { \{ K , V \} }$ from different transformer layers. The lighter the color, the smaller the distribution difference between two layers. We use blue boxes to highlight adjacent layers with similar KV cache space.

$$
J S D ( P _ { K } ^ { \lambda _ { m } } \| P _ { K } ^ { \lambda _ { n } } ) \leq \delta _ { 1 } , \forall \lambda _ { m } , \lambda _ { n } \in \Lambda _ { i }
$$

$$
\sum _ { \lambda _ { m } \in \Lambda _ { i } } J S D ( P _ { K } ^ { \lambda _ { m } } \| P _ { K } ^ { \lambda _ { m + 1 } } ) \leq \delta _ { 2 } , w i t h \lambda _ { m + 1 } \in \Lambda _ { i }
$$

where $P _ { K } ^ { \lambda _ { m } }$ represents the discrete distribution obtained through dimensional histogram after reducing the dimensions of the vector in the attention head to 2 dimensions using PCA, $\delta _ { 1 }$ and $\delta _ { 2 }$ are thresholds used to adjust the aggregation strategy. Both aim to prevent the cache space from becoming excessively large after aggregation.

Based on these observations, we propose the following guidelines for constructing the dictionary needed for CSR:

• First, the construction of the dictionary will be divided into two parts: offline and online.   
• Second, we choose to preprocess $X _ { K }$ prior to the embedding of positional information.

# Algorithm 1: NeuralDict

1: procedure $\mathrm { N E U R D I C T } ( { \mathcal { C } } , m , N , E )$ 2: Input: The calibration corpus dataset $\mathcal { C }$ , language model $m$ , offline dictionary size $N$ and training procedure epochs number $E$ . 3: Perform inference on dataset $\mathcal { C }$ using model $m$ and collect $X _ { K } ^ { m } , X _ { V } ^ { m }$ for each layer in model $m$ 4: Generate $\mathcal { M } _ { K }$ and $\mathcal { M } _ { V }$ based on Equation 6 and 7. 5: TRAINONMERGEDLAYERS $( \mathcal { M } _ { K } , N , X , E )$ 6: TRAINONMERGEDLAYERS $( \mathcal { M } _ { V } , N , X , E )$ 7: procedure TRAINONMERGEDLAYERS $( \mathcal { M } , N , \boldsymbol { X } , E )$ 8: for $\boldsymbol { \Lambda } _ { i } \in \mathcal { M }$ do 9: $X { = }$ concatenate $[ X ^ { \lambda _ { n } }$ for $\lambda _ { n } \in \Lambda _ { i } ]$ 10: TRAINNEURDICT $( N , X , E )$ 11: procedure TRAINNEURDICT $( N , X _ { \{ K , V \} } , e )$ 12: Input: Offline dictionary size $N$ , and Key cache or Value cache $X _ { \{ K , V \} }$ in corpus dataset, epochs $e$ to train 13: Initialize $W _ { D } = [ d _ { 1 } , \dots , d _ { N } ]$ with cluster centroids of X K,V . 14: $W _ { D } \doteq \mathbf { R E N O R M } ( W _ { D } )$ 15: for $\hat { e } = [ 1 , \ldots , e ]$ do 16: for Batch B ∈ Xl,Kh,V do 17: calculate $\mathcal { L }$ using equation 11 18: backward $\mathcal { L }$ and update $W _ { D }$ 19: $W _ { D } = \mathrm { R E N O R M } ( W _ { D } )$ 20: procedure RENORM $W _ { D } )$ 21: ▷ Normalize the vector to ensure its l2 norm unity ◁ 22: for i in $[ 1 , 2 , . . . , \mathrm { N } ]$ do 23: $\begin{array} { r } { \widehat { w } _ { i } = \frac { w _ { i } } { \| w _ { i } \| _ { 2 } } \qquad } \\ { \widehat { W } = [ \widehat { w } _ { 1 } ; \widehat { w } _ { 2 } ; \ldots ; \widehat { w } _ { N } ] \qquad } \\ { \mathrm { r e t u r n } \widehat { W } \qquad } \end{array}$ 24: 25:

• Third, we fuse the $X _ { \{ K , V \} }$ from adjacent transformer layers to construct a multi-layers shared offline dictionary based on the similarity between layers in order to save memory footprint as much as possible.

# CSR’s Workflow

We divide CSR into two stages. The first stage is the Preparation Stage, in which CSR probes $X _ { \{ K , V \} }$ of each transformer layer using the calibration corpus dataset for a new language model, and then aggregates $X _ { \{ K , V \} }$ of each layer following (6) and (7), and trains to obtain an offline dictionary that can be shared by multiple layers. Another stage is Inference Stage, in which CSR replaces the original KV cache of the language model and utilizes the sparse representations to reduce the GPU memory footprint.

# Preparation Stage

The primary concern is how to construct a dictionary that can approximately represent each KV cache tensor in LLM generated by the current query by selecting only $s$ bases in the dictionary. Clustering is a widely used unsupervised learning method for extracting features from a vector space. However, the clustering algorithm does not directly interact with the process of calculating the sparse representation in CSR. As a result, the dictionary constructed by clustering does not take into consideration the features of residual tensors beyond the first iteration in MP. To address these issues, we propose a novel neural network-based method named NeuralDict to automatically resolve this problem.

NeuralDict The offline dictionary construction remains consistent across $\mathcal { M } _ { K }$ or $\mathcal { M } _ { V }$ of the model. We utilize the calibration set $\mathcal { C }$ as the corpus dataset to assess the distribution of $X _ { \{ K , V \} }$ in each layer of the large language model $m$ . For a model $m$ with hidden states size in each attention head as $d _ { h }$ , CSR split the hidden states evenly into $s _ { n }$ chunks according to the order of channels, and given the dictionary $D$ with a size of $N$ , the dictionary we aim to create can be viewed as a matrix $W _ { D } \in \mathbb { R } ^ { ( d _ { h } / / s _ { n } ) \times N }$ . This matrix $W _ { D }$ can be considered as the learnable weights in a single linear layer neural network without any bias or activation function. We utilize the mean squared error as shown in Equation 8 to train $W _ { D }$ . Take Key cache as example:

$$
\mathcal { L } _ { M S E } = \sum _ { x \in { X _ { K } } } \Vert \mathbf { x } - W _ { D } \mathbf { r } ( \mathbf { x } , W _ { D } , s ) \Vert _ { 2 } ^ { 2 }
$$

where $\mathbf { r } ( \mathbf { x } , W _ { D } , s )$ represents the sparse representation vector calculated by the MP algorithm, and $W _ { D }$ serves as the basis vector’s dictionary. The mean squared error $\mathcal { L } _ { M S E }$ will be excessively large and difficult to optimize when $s$ is too small, while a large $s$ will result in a prolonged MP process, leading to lower training efficiency. After updating $W _ { D }$ through loss backpropagation, we apply an additional update to $W _ { D }$ as:

$$
W _ { D } = \mathrm { R e N o r m } ( W _ { D } )
$$

where ReNorm denotes the normalize each vector to $l _ { 2 }$ norm unity as shown in Algorithm 1.

Adaptive Regularization to Encourage Diversity. We include the following regularization term to promote the diversity of vectors in $W _ { D }$ to prevent the training from getting trapped in local optima:

$$
\mathcal { L } _ { d i v } = \frac { 1 } { N ^ { 2 } } \| I - W _ { D } ^ { T } W _ { D } \| _ { F } ^ { 2 }
$$

$\| \cdot \| _ { F }$ denotes the Frobenius norm, and $I \in \mathbb { R } ^ { N \times N }$ represents the identity matrix. Since the magnitude of the mean squared error (MSE) loss varies with the transformer layer while the diversity term does not, we incorporate an adaptive coefficient to adjust the weight of $\mathcal { L } _ { M S E }$ and $\mathcal { L } _ { d i v }$ :

$$
\mathcal { L } = \mathcal { L } _ { M S E } + \beta \mathcal { L } _ { d i v }
$$

where $\begin{array} { r } { \beta = \mathrm { c l a m p } ( 0 . 1 \times \frac { \hat { \mathcal { L } } _ { M S E } } { \hat { \mathcal { L } } _ { d i v } } , 0 , 1 . 0 ) } \end{array}$ . Note that $\hat { \mathcal { L } } _ { M S E }$ and $\hat { \mathcal { L } } _ { d i v }$ represent the calculated values without any gradient information. The purpose of limiting $\beta$ to 1.0 is to prevent the model from overly focusing on reducing $\mathcal { L } _ { d i v }$ and disregarding $\mathcal { L } _ { M S E }$ when $\mathcal { L } _ { d i v }$ is sufficiently small. The whole training procedure is shown in Algorithm 1.

# Inference Stage

When the language model’s transformer layer is loaded into the GPU, CSR will load the layer’s corresponding offline dictionary onto the same device. Note that due to the existence of Merged Layers, we prefer to load layers corresponding to the same offline dictionary onto the same single device. The whole process of how CSR take place of original KV cache is illustrated in Figure 1.

Build Dictionary For prompt $p$ , CSR build $D _ { K } ^ { \lambda } ( p )$ and $D _ { V } ^ { \lambda } ( p )$ as dictionaries for the Key and Value cache. For each transformer layer $\lambda$ . CSR will extract the corresponding part from the offline dictionary for the transformer layer according to the layer index as show in Figure 1. The bank of the online part is divided into two sections. During the prefill phase, CSR will perform online sampling methods from the calculated KV cache. In order to prevent poor fitting results caused by out-of-distribution entries in the KV cache during inference, we follow the KV quantization framework, such as (Kang et al. 2024), and design a separate module named Guard part for handling outlier entries.

KV Decomposition and Sparse Storage CSR will compute the sparse representation for the tokens in the prompt using $D _ { K } ^ { \lambda } \dot { ( q ) }$ or $\hat { D } _ { V } ^ { \lambda } ( q )$ for the $X _ { K , V }$ by solving problem 1 using the Matching Pursuit algorithm. The maximum sparsity is set to be $s$ which is so-called $M P$ -level, the Matching Pursuit algorithm will perform $s$ iterations to generate sparse representations with a sparsity of $s$ for the entire $X _ { \{ K , V \} }$ . We denote the sparse representations of the KV cache as ${ \bf r } ( X _ { \{ K , V \} } , D _ { \{ K , V \} } ^ { \lambda } ( q ) , s \hat { ) }$ with sparsity $s$ in layer $\lambda$ . Please note that there are no more than $s$ non-zero elements in r.Therefore, it is only necessary to store the index and coefficient of these non-zero elements. The index indicates the position of the selected basis vector in the dictionary, while the value represents the corresponding coefficient.

De-Sparse to Restore To meet the needs of calculating attention scores, CSR will de-sparse $\mathbf { r }$ into tensor form:

$$
\tilde { X } _ { \{ K , V \} } ^ { \lambda } = D _ { \{ K , V \} } ^ { \lambda } { \bf r } ( X _ { \{ K , V \} } ^ { \lambda } , D _ { \{ K , V \} } ^ { \lambda } , s )
$$

Here, $\tilde { X } _ { \{ K , V \} } \in \mathbb { R } ^ { b \times l _ { g } \times h \times d _ { h } }$ , and $l _ { g }$ represents the number of prompt tokens and generated tokens. $\tilde { X } _ { \{ K , V \} }$ will be used in the attention score calculation instead of original KV cache. When new tokens are generated, the corresponding KV cache will also be replaced by CSR.

Analysis for CSR The initial $X _ { K , V }$ of each attention head comprises $d _ { h }$ floating-point values, with fp16 as the prevalent datatype in LLM inference. With CSR, only $s \times s _ { n }$ fp16 values for coefficients, accompanied by $s \times s _ { n }$ INT16 values for indexes. The compression rate can be calculated as 16s sn×+1h6s sn $\begin{array} { r } { \frac { 1 6 \times d _ { h } } { 1 6 s \times s _ { n } + 1 6 s \times s _ { n } } \ = \ \frac { \bar { d } _ { h } } { 2 s \times s _ { n } } } \end{array}$ dh , which implies that for $\mathrm { C S R } ( s , s _ { n } )$ , the number of bits of the corresponding quantization algorithm is d /21s6 s $\begin{array} { c c c } { \frac { 1 6 } { d _ { h } / 2 s \times s _ { n } } } & { = } & { \frac { 3 2 s \times s _ { n } ^ { \star } } { d _ { h } } } \end{array}$ 32s×sn bits. Taking LLaMA3-8B whose $d _ { h } = 1 2 8$ as an example, for ${ \mathrm { C S R } } ( s =$ $4 , s _ { n } = 1 \rangle$ or $( s = 2 , s _ { n } = 2 )$ , the corresponding quantization bit count is 1 bit.

# Experiments

# Experiments Settings

Models We applied CSR to multiple large language models (LLMs) based on the HuggingFace Transformers library1. To evaluate CSR’s effectiveness across different attention mechanisms, we conducted experiments on Llama2-7Bchat(Touvron et al. 2023b,a), Llama3-8B-Instruct(AI@Meta 2024), and Baichuan2-7B-chat(Baichuan 2023). Specifically, Llama2-7B-chat and Baichuan2-7B-chat utilize MHA, while Llama3-8B-Instruct adopts GQA.

Benchmark The primary goal of CSR is to reduce the memory usage of the KV cache by identifying sparse representations for the KV cache within a long context setting. To evaluate its effectiveness, we utilized the LongBench benchmark (Bai et al. 2023), which is a bilingual and multitask benchmark designed to assess the long context understanding capabilities of LLM. In our evaluation, we relied on standard metrics such as F1 score, ROUGE score, and similarity score. These metrics align with the settings established in (Liu et al. 2024) for different datasets within the LongBench. CSR In the experiments, unless stated otherwise, the Value Cache uses $s _ { n } = 2$ , and the Key Cache uses $s _ { n } = 1$ . For simplicity, CSR- $s$ denotes the MP-level. Since $s _ { n } = 2$ for the Value Cache, its maximum MP-level is half that of the Key Cache. For example, CSR-8 corresponds to $s = 8 , s _ { n } =$ 1 for the Key Cache, and $s = 4$ , $s _ { n } = 2$ for the Value Cache. The experimental results presented in the main content uniformly adopt reverse sequential sampling for the online sampling part due to space constraints. In CSR’s online part, the Guard size per layer is 8192, and the sampling size is 4096 for Llama2 and Baichuan2. For Llama3, the Guard size is 2048, with a sampling size of 1024. For further details on the offline part and additional ablation studies, please refer to our extended version2.

Baselines We selected state-of-the-art (SOTA) KV cache quantization algorithms to establish a robust baseline for measuring CSR performance. These included:

• KIVI(2024): KIVI introduced a tuning-free quantization algorithm known as KIVI-2 and KIVI-4 for 2 bits and 4 bits correspondingly, and quantize the Key cache perchannel and the Value cache per-token. • GEAR(2024): GEAR applies 4-bit quantization to the majority of entries in the KV cache and utilizes a lowrank matrix to approximate the quantization error. Additionally, GEAR uses a sparse matrix to handle outliers.

Hardware Environment A single NVIDIA A100 GPU (80GB) with 128GB memory.

# Robust Performance on Various Tasks

Initially, we present a comparison between CSR and various quantization algorithms on the Llama2-7B-chat model. For KIVI and GEAR, we perform grid search on hyperparameters and show the results of the best obtained. The hidden size of each attention head in the Llama2-7B-chat model is

Table 1: We conducted experiments on CSR methods and corresponding quantization methods with an identical number of bits, using Llama2-7B-chat. We highlight the data where our method performs better within the same precision group. As there is no equivalent method for quantization below 2 bits, we only present CSR-4 and CSR-6.   

<html><body><table><tr><td rowspan="2">Method</td><td rowspan="2">2wiki-</td><td rowspan="2">hoqa-</td><td rowspan="2">musi- que</td><td rowspan="2">trec qa</td><td rowspan="2">arr-</td><td rowspan="2">qas- per</td><td rowspan="2">sum</td><td rowspan="2">sam-</td><td rowspan="2">lcc</td><td rowspan="2">trivi-</td><td rowspan="2">meldi- qa_en</td><td rowspan="2">Avg</td></tr><tr><td></td></tr><tr><td>FP16</td><td>26.47</td><td>33.84</td><td>9.33</td><td>68.14</td><td>16.65</td><td>17.17</td><td>20.81</td><td>40.88</td><td>58.25</td><td>82.74</td><td>35.62</td><td>37.26</td></tr><tr><td>GEAR</td><td>26.52</td><td>32.65</td><td>9.01</td><td>68.17</td><td>16.78</td><td>18.03</td><td>21.12</td><td>41.66</td><td>57.59</td><td>83.53</td><td>36.65</td><td>37.42</td></tr><tr><td>KIVI-4</td><td>26.08</td><td>33.48</td><td>9.53</td><td>68.14</td><td>17.14</td><td>17.16</td><td>20.61</td><td>40.33</td><td>58.07</td><td>82.49</td><td>36.06</td><td>37.19</td></tr><tr><td>CSR-16</td><td>26.19</td><td>34.19</td><td>9.22</td><td>68.14</td><td>17.44</td><td>16.99</td><td>21.00</td><td>40.62</td><td>58.07</td><td>82.85</td><td>36.23</td><td>37.36</td></tr><tr><td>KIVI-2 CSR-8</td><td>26.06</td><td>32.13</td><td>9.71</td><td>68.14</td><td>16.69</td><td>19.30</td><td>20.79</td><td>39.31</td><td>57.21</td><td>83.23</td><td>36.32</td><td>37.17</td></tr><tr><td></td><td>26.86</td><td>33.52</td><td>9.06</td><td>68.17</td><td>16.70</td><td>16.15</td><td>20.48</td><td>38.52</td><td>57.47</td><td>82.91</td><td>35.41</td><td>36.93</td></tr><tr><td>CSR-6</td><td>26.28</td><td>33.76</td><td>9.13</td><td>67.75</td><td>16.18</td><td>14.96</td><td>20.68</td><td>38.24</td><td>56.53</td><td>83.41</td><td>33.43</td><td>36.39</td></tr><tr><td>CSR-4</td><td>25.54</td><td>31.90</td><td>8.55</td><td>66.29</td><td>15.70</td><td>13.42</td><td>20.29</td><td>37.37</td><td>54.19</td><td>81.64</td><td>31.50</td><td>35.13</td></tr></table></body></html>

<html><body><table><tr><td rowspan="2">Model</td><td rowspan="2">Meth- od</td><td rowspan="2">2wiki- mqa</td><td rowspan="2">hotp- otqa</td><td rowspan="2">musi- que</td><td rowspan="2">trec</td><td rowspan="2">atarr- qa</td><td rowspan="2">qas- per</td><td rowspan="2">qm- sum</td><td rowspan="2">sam- sum</td><td rowspan="2">lcc</td><td rowspan="2">trivi- aqa</td><td rowspan="2">meld- qa_en</td><td rowspan="2">Avg</td></tr><tr><td></td></tr><tr><td rowspan="5">Lla- ma3</td><td>FP16</td><td>32.17</td><td>33.23</td><td>17.87</td><td>74.54</td><td>20.24</td><td>29.22</td><td>22.83</td><td>42.63</td><td>56.49</td><td>88.79</td><td>38.81</td><td>41.51</td></tr><tr><td>CSR-16</td><td>31.83</td><td>35.26</td><td>17.87</td><td>74.29</td><td>20.13</td><td>29.56</td><td>22.39</td><td>41.19</td><td>57.58</td><td>89.04</td><td>40.05</td><td>41.74</td></tr><tr><td>CSR-8</td><td>29.42</td><td>35.82</td><td>17.39</td><td>73.92</td><td>19.68</td><td>26.98</td><td>22.44</td><td>40.73</td><td>57.10</td><td>89.62</td><td>36.90</td><td>40.91</td></tr><tr><td>CSR-6</td><td>29.51</td><td>34.76</td><td>18.04</td><td>73.16</td><td>19.99</td><td>24.26</td><td>21.94</td><td>39.86</td><td>55.79</td><td>89.35</td><td>35.73</td><td>40.22</td></tr><tr><td>CSR-4</td><td>28.52</td><td>35.28</td><td>16.87</td><td>71.61</td><td>19.70</td><td>22.03</td><td>21.54</td><td>38.19</td><td>54.95</td><td>89.01</td><td>33.14</td><td>39.17</td></tr><tr><td rowspan="5">Baic- huan2</td><td>FP16</td><td>20.13</td><td>26.52</td><td>11.51</td><td>73.46</td><td>17.66</td><td>21.02</td><td>22.04</td><td>17.05</td><td>63.53</td><td>81.02</td><td>39.78</td><td>35.79</td></tr><tr><td>CSR-16</td><td>19.64</td><td>25.99</td><td>11.68</td><td>73.46</td><td>17.98</td><td>19.94</td><td>21.79</td><td>17.82</td><td>62.86</td><td>80.44</td><td>39.69</td><td>35.57</td></tr><tr><td>CSR-8</td><td>18.90</td><td>25.92</td><td>11.63</td><td>73.29</td><td>17.24</td><td>19.26</td><td>21.70</td><td>20.24</td><td>61.67</td><td>81.50</td><td>38.37</td><td>35.43</td></tr><tr><td>CSR-6</td><td>18.73</td><td>24.38</td><td>11.61</td><td>72.96</td><td>17.85</td><td>16.70</td><td>21.47</td><td>22.28</td><td>60.87</td><td>79.26</td><td>38.20</td><td>34.94</td></tr><tr><td>CSR-4</td><td>18.06</td><td>24.36</td><td>10.40</td><td>70.97</td><td>16.79</td><td>15.01</td><td>20.86</td><td>23.12</td><td>58.36</td><td>78.29</td><td>33.58</td><td>33.62</td></tr></table></body></html>

Table 2: Experiments on Longbench using Llama3-8B and Baichuan2-7B show that CSR is also effective for these models.

128 so according to the previous analysis, CSR-8 is equivalent to 2 bits in quantization, and CSR-16 is equivalent to 4 bits in quantization. We grouped several methods according to equivalent quantization levels, namely FP16 corresponding to 16 bits, GEAR, KIVI-4 and CSR-16 corresponding to 4 bits, and KIVI-2 and CSR-8 corresponding to 2 bits in Table 1. The performance of various methods on multiple datasets, is presented in Table 1. For the 4-bit group, our method performs better than KIVI and GEAR on most datasets, while for the 2-bit group, our method and KIVI have their own advantages and disadvantages. We conclude that CSR, KIVI, and GEAR exhibit similar performances and CSR can provide performance comparable to state-ofthe-art 4-bit or 2-bit quantization algorithms.

Effective CSR with Less Than 2 bit: There is no way to reduce from 2bit to 1bit for quantization based methods. However, CSR can provide sparse representation for all KV caches at less than 2 bits or even 1 bit per channel, thus alleviating the tight memory resources of the GPU without any KV cache eviction. We conducted extensive experiments with CSR-6, equivalent to 1.5 bit, and CSR-4, equivalent to only 1 bit. In this scenario, CSR can still maintain performance on most datasets, with only a slight performance drop as shown in Table 1. Even CSR-4 only drops $5 . 7 \%$ in model performance compared to FP16, with less than $\frac { 1 } { 1 0 }$ memory occupied by KV cache.

# CSR Works Well for Various Language Models

CSR is independent of the attention mechanism utilized in LLM, making it theoretically applicable to various models. In order to validate the versatility of our method across different models, we conducted more experiments on Baichuan2-7B, Llama3-8B-Instruct. As depicted in Table 2. The results demonstrate that despite providing at least an $8 \mathbf { x }$ compression ratio compared to the original data type, CSR still delivers strong performance across all models. The performance loss of CSR-4 compared to FP16 is $5 . 7 \%$ for Llama3-8B-Instruct and $6 . 0 \%$ for Baichuan2-7B-chat according to the value in Avg column.

# Memory Foorpint

We plotted the relationship between the inference length and memory footprint of KV cache for different models using different methods. As shown in Figure 4, the additional

1020304050Memory Usage(GB) Original Llama3 Llama3 $^ +$ KIVI-2 Llama3 $^ +$ CSR(6) Llama3 $^ +$ CSR(4) LOlraigminaa2 $^ +$ lKaImVaI-2 Llama2 $^ +$ CSR(6)   
1000 2000 4000 8000 30000 Llama2 $^ +$ CSR(4) Sequence Length

memory overhead introduced by the offline or online dictionary is almost negligible. Compared with the original KV cache, both CSR and quantization algorithms have greatly reduced the memory occupied by the KV cache. Compared with quantization, which cannot be further reduced from 2 bits, CSR provides the possibility of further reducing memory usage in long context scenarios.

# Effect of $s _ { n }$ and NeuralDict Size

We analyze the role of $s _ { n }$ based on the training results on NeuralDict for Llama2-7B-chat model. The MSE loss on the test dataset is shown in the table 3. Overall, from the perspective of Key cache and Value cache, the improvement of splitting the Value cache is very obvious, but there is almost no improvement on Key cache. Specifically, when $s = 8$ , the most significant improvement occurs when $s _ { n }$ increases from 1 to 2 for Value cache. After a comprehensive trade-off between performance and compression rate, we fix $s _ { n } = 1$ for the Key and $s _ { n } = 2$ for Value in CSR in actual use.

Table 3: MSE loss on test dataset of the converged NeuralDict trained with different $s$ and $s _ { n }$ parameters.   

<html><body><table><tr><td colspan="4">Value Cache</td><td colspan="4">Key Cache</td></tr><tr><td>size</td><td>s</td><td>Sn</td><td>Loss</td><td>二 size</td><td>S</td><td>Sm</td><td>Loss</td></tr><tr><td rowspan="3">8192</td><td>4 4 4</td><td>1 2 4</td><td>0.259 0.253 0.093</td><td rowspan="3">8192</td><td>4 4 4</td><td>1 2 4</td><td>0.079 0.068 0.052</td></tr><tr><td>8 8</td><td>1 2</td><td>0.213 0.094</td><td>8 8</td><td>1 1</td><td>0.055 0.047</td></tr><tr><td>8 4</td><td>4 1</td><td>0.012 0.263</td><td>8 4</td><td>1 1</td><td>0.037 0.077</td></tr><tr><td rowspan="3">16384</td><td>4 4</td><td>2 4</td><td>0.223 0.078</td><td rowspan="3">16384</td><td>4 4</td><td>2 4</td><td>0.058 0.042</td></tr><tr><td>8 8</td><td>1 2</td><td>0.194 0.079</td><td>8 8</td><td>1 2</td><td>0.047 0.043</td></tr><tr><td>8</td><td>4</td><td>0.008</td><td>8</td><td>4</td><td>0.033</td></tr></table></body></html>

# Related Work

KV Cache Quantization Quantization is an alternative method for reducing memory and compute requirements during generation tasks, particularly in processing extremely long contexts. Prior research, such as that by (Hooper et al. 2024; Yue et al. 2024), has focused on quantizing the KV cache. Meanwhile, (Liu et al. 2024) proposes quantizing the key cache per-channel and the value cache per-token, (Kang et al. 2024) propose to use SVD to reduce the quantization error. However, these approaches are not applicable when the per-token quantization falls below 2 bits.

KV Cache Eviction Various approaches exist to minimize the KV cache footprint, with the common objective of retaining only a small subset of keys and values. One technique utilizes the attention mechanism’s localized pattern, namely the attention sink, as proposed by (Xiao et al. 2023). This involves employing a finite attention window to retain only the sink token and a fixed number of recent tokens. Another strategy involves implementing a KV cache eviction policy considering the attention mechanism’s sparsity. For example, (Zhang et al. 2023; Ge et al. 2023) suggest discarding non-essential parts of the KV cache to reduce memory usage during large language model (LLM) inference. Moreover, (Liu et al. 2023) identifies a repetitive attention pattern during inference processes, recommending the retention of only pivotal tokens. Additionally, (Anagnostidis et al. 2023) employs a learnable mechanism to identify uninformative tokens, implementing adaptive sparse attention that requires fine-tuning on the pre-trained model.

# Conclusion

This paper introduces CSR, a framework for optimizing the memory footprint of the KV cache during LLM inference, based on compressed sensing algorithms. Our experiments on widely-used LLMs and long-context datasets have demonstrated that CSR’s performance comparable to quantized algorithms when memory resources are relatively abundant (in comparison to 2-bit or 4-bit KV cache quantized algorithms). Furthermore, CSR exhibits robust performance even when memory is more constrained, aiming for less than 2 bits per channel. Notably, even with a per-channel bit count as low as 1, CSR can maintain robust performance. We believe that CSR provides an alternative approach for compressing the KV cache independently of quantizationrelated algorithms. We conclude that CSR can operate effectively in memory-constrained environments and maintain strong performance across a variety of tasks, even with extremely low memory usage of KV cache.

# Limitations

Compared with the quantization algorithm, CSR further reduces the memory occupied by the KV cache. However, the process of detecting the KV cache space of the model through the calibration dataset and then obtaining a part of the dictionary through offline training is time-consuming. We leave the research for a more efficient way to obtain the offline dictionary as future exploration.