# Sequence Accumulation and Beyond: Infinite Context Length on Single GPU and Large Clusters

Weigao $\mathbf { S u n } ^ { 1 * }$ , Yongtuo Liu2, Xiaqiang Tang3, Xiaoyu $\mathbf { M _ { 0 } } ^ { 4 * }$

1Shanghai AI Laboratory 2University of Amsterdam 3The Hong Kong University of Science and Technology (Guangzhou) 4Nanyang Technological University sunweigao $@$ outlook.com, y.liu6 $@$ uva.nl, xtang771 $@$ connect.hkust-gz.edu.cn, xiaoyu006@e.ntu.edu.sg

# Abstract

Linear sequence modeling methods, such as linear attention, state space modeling, and linear RNNs, have recently been recognized as potential alternatives to softmax attention thanks to their linear complexity and competitive performance. However, although their linear-memory advantage during training enables dealing with long sequences, it is still hard to handle extremely long sequences with very limited computational resources. In this paper, we propose Sequence Accumulation (SA) which leverages the common recurrence feature of linear sequence modeling methods to manage infinite context length even on a single GPU. Specifically, SA divides long input sequences into fixed-length sub-sequences and accumulates intermediate states sequentially, which reaches only constantmemory consumption. Additionally, we further propose Sequence Accumulation with Pipeline Parallelism (SAPP), to train large models with infinite context length, without incurring any additional synchronization costs in the sequence dimension. Extensive experiments with a wide range of context lengths are conducted to validate the effectiveness of SA and SAPP on both single and multiple GPUs. Results show that SA and SAPP enable the training of infinite context length on even very limited resources, and are well compatible with the out-of-the-box distributed training techniques.

# Introduction

The attention mechanism (Vaswani et al. 2017; Qu et al. 2024) is recognized as a crucial component for effective sequence modeling. However, softmax attention has a complexity that grows quadratically with context length, making it inherently costly. Although recent advancements (Dao et al. 2022; Dao 2023; Shah et al. 2024) have enabled the scaling of softmax attention to longer sequences by optimizing intermediate computations for hardware, these approaches still necessitate the storage of key and value vectors. Managing this "KV cache" can become cumbersome when dealing with extremely long sequences.

Meanwhile, linear sequence modeling techniques, such as linear attention (Katharopoulos et al. 2020; Qin et al. 2022a; Yang et al. 2023; Qin et al. 2024c; Shen et al. 2024), state space modeling (SSM) (Gu and Dao 2023; Dao and Gu 2024), and linear RNNs (Peng et al. 2023, 2024; Qin et al.

2024d), have recently emerged as compelling alternatives to the conventional softmax-attention-based transformer architecture. These approaches are characterized by their linear complexity in both training computation and memory usage, which also eliminates the need for the KV cache, allowing for constant-memory inference. Although these methods originate from different technical backgrounds, recent works (Qin et al. $2 0 2 4 \mathrm { a }$ ; Yang et al. 2024) underscore the commonalities among these models, leading to the development of common techniques that boost the efficiency of linear sequence modeling methods.

Although the linear sequence modeling methods show the advantages of linear complexity and competitive performance, they still face challenges when modeling extremely long sequences with limited GPU resources. This is due to their linear-memory complexity with respect to the context length during training. To model extremely long sequences, previous methods rely on sequence or context parallelism techniques. However, sequence or context parallelism does not change the linear-memory complexity of linear sequence models, which also requires a lot more GPU resources and introduces heavy communication overheads.

In the paper, we propose Sequence Accumulation (SA) for linear sequence modeling methods to achieve constantmemory training. In this way, SA can enable infinite context length modeling even on a single GPU. Specifically, SA works by dividing a long input sequence into multiple sub-sequences (abbreviated as sub-seqs) and sequentially accumulating the intermediate states from each sub-seq into a single state. SA offers a distinct advantage when training with extremely long context lengths on limited computational resources. Additionally, we further propose Sequence Accumulation with Pipeline Parallelism (SAPP) to integrate SA with PP for training large models with infinite context length. SAPP accumulates intermediate states across model partitions on different devices without incurring any additional synchronization costs in the sequence dimension.

Through experiments conducted both on a single GPU and across multiple GPUs, we demonstrate the effectiveness of SA and SAPP on managing extremely long context lengths. Additionally, we evaluate the compatibility of SAPP with other parallel training strategies such as data parallelism (DP), tensor parallelism (TP), and context parallelism (CP), on a cluster with multiple nodes. The key contributions of this paper are summarized as follows:

Table 1: Instances of Unified Linear Sequence Modeling. All instances listed follow the unified formulation in Eq. 3. Here, $\lambda$ represents a fixed constant, $\lambda _ { t } \in \mathcal { R }$ refers to a time-dependent scalar, and $\lambda _ { t } \in \mathcal { R } ^ { d }$ indicates a time-dependent vector. It is important to note that $\lambda$ may denotes different constants or variables in each specific method.   

<html><body><table><tr><td>Modeling Method</td><td>Formulation</td><td>Choice of 入</td></tr><tr><td>Linear Attention (Katharopoulos et al. 2020)</td><td>KVt=KVt-1+KTVt</td><td></td></tr><tr><td>Lightning Attention (Qin et al. 2023b)</td><td>KVt =XKVt-1+KTVt</td><td>入ER</td></tr><tr><td>RetNet (Sun et al. 2023)</td><td>KVt =λKVt-1+KTVt</td><td>入∈R</td></tr><tr><td>GLA (Yang et al. 2023)</td><td>KVt =diag{λt}KVt-1+KTVt</td><td>At∈Rd</td></tr><tr><td>DeltaNet (Yang et al. 2024)</td><td>KVt = (I-XtKτKt)KVt-1+λtKTVt</td><td>入t∈R</td></tr><tr><td>Mamba2 (Dao and Gu 2024)</td><td>KVt=XtKVt-1+KTVt</td><td>入t∈R</td></tr><tr><td>RWKV-6 (Peng et al. 2024)</td><td>KVt = diag{λt}KVt-1+KTVt</td><td>At∈Rd</td></tr><tr><td>HGRN2 (Qin et al. 2024d)</td><td>KVt = diag{λt}KVt-1+(1-λt)TVt</td><td>入t∈Rd</td></tr></table></body></html>

• Sequence Accumulation. We introduce the SA method, which accumulates intermediate states from each sub-seq, enabling constant-memory training with infinite context length even on a single GPU.   
• Sequence Accumulation Pipeline Parallelism. SA is integrated with PP, allowing the accumulation of intermediate states across model partitions on different devices, thus facilitating the training of large models with infinite context lengths on large clusters.   
• Compatibility with Hybrid Parallelism. We implement and evaluate the compatibility of SAPP with existing parallel training techniques, including DP, TP, and CP.

# Preliminary

Notation Throughout this paper, we maintain consistent notations for arithmetic expressions. For clarity, matrices are denoted by uppercase letters, while vectors are represented by lowercase letters, both in non-boldface. Scalar or matrix multiplication is indicated by the symbol " " or is implied when the symbol is omitted, while Hadamard (element-wise) multiplication is represented by $" \odot "$ ". The scalars $N$ and $d$ refer to the context length and hidden dimension, respectively. To simplify the notation, we omit the dimensions related to batch size and the number of heads in the tensor shapes.

Linear Attention The standard softmax attention (Vaswani et al. 2017), commonly used in transformer models, can typically be expressed as:

$$
O = \operatorname { s o f t m a x } ( Q \cdot K ^ { \top } / \sqrt { d } ) \cdot V .
$$

Here, the matrices $Q , K , V , O \in \mathcal { R } ^ { N \times d }$ correspond to the query, key, value, and output matrices, respectively. The matrices $Q , K$ , and $V$ are linear projections of the input matrix $X ~ \in ~ \mathcal { R } ^ { N \times d }$ , defined as $Q \ = \ X W _ { Q }$ , $K \ : = \ : X W _ { K }$ , and $V = X W _ { V }$ , where $W _ { Q } , W _ { K } , W _ { V } \in \mathcal { R } ^ { d \times d }$ are learnable weight matrices.

Linear attention (Katharopoulos et al. 2020) introduces two key modifications to the standard softmax attention: 1) it eliminates the softmax $( \cdot )$ operation, thereby removing the need for the scaling factor $1 / { \sqrt { d } }$ ; and 2) it alters the order of matrix multiplications by first computing $K ^ { \top } V$ , followed by $Q ( K ^ { \top } V )$ . These adjustments reduce both the computational and memory complexity of attention from $O (  { \bar { N } } ^ { 2 } d )$ to $O ( N d ^ { 2 } )$ . This technique is often referred to as the rightproduct kernel trick because it prioritizes the multiplication on the right side first.

Another important feature of linear attention is that its computation of $K V$ state can be processed in a recurrent form, similar to RNN, as follows:

$$
K V _ { t } = K V _ { t - 1 } + K _ { t } ^ { \top } \cdot V _ { t } .
$$

In this formulation, the intermediate state $K V$ is updated at each time step $t$ by adding the product of $K ^ { \top } \cdot V$ from the $t$ -th input to the previous computed state $K V _ { t - 1 }$ . Note that the term $K V$ here represents a single state or matrix, which is distinct from the key $K$ and value $V$ matrices.

Unified Linear Sequence Modeling Linear attention simplifies the standard softmax attention using the right-product kernel trick, which is a constructive step towards linear sequence modeling. Follow-up studies on state space modeling and linear RNNs, also demonstrate the linear sequence modeling is comparable with transformer architecture from the perspectives of control theory and routine RNN. Recent studies (Qin et al. $2 0 2 4 \mathrm { a }$ ; Yang et al. 2024) suggest that the linear attention, state space, and linear RNN sequence modeling methods can be expressed within a unified recurrence framework as:

$$
\begin{array} { l } { \widehat { M } _ { t } = f ( K _ { t } ^ { \top } , V _ { t } ) , } \\ { M _ { t } = \Theta _ { t } \diamond M _ { t - 1 } + \widehat { M } _ { t } . } \end{array}
$$

In this formulation, $\widehat { M } _ { t } \in \mathcal { R } ^ { d \times d }$ represents the memory state corresponding to thec $t$ -th input, which is a function of $K _ { t } ^ { \top }$ and $V _ { t }$ . And $\Theta _ { t }$ denotes a coefficient matrix that may be time-varying or constant (and also can be a vector or scalar). The operator $" 0 "$ can denote either standard matrix multiplication or a Hadamard product. We collect some recent linear sequence modeling methods which follow the unified formulation in Eq. 3 and list them in Table 1.

# Method

In this section, we begin by introducing the SA method, followed by an explanation of its integration with PP. The visual

representations of SA and SAPP are provided in Figure 1 and Figure 2, respectively.

# Sequence Accumulation

SA is a co-design of algorithm and system techniques aimed at enabling the training of models with long context length using minimal GPU resources, which introduces a new concept that differs from sequence parallelism. In existing sequence or context parallelism approaches, the input sequence is divided into multiple shards, which are then distributed across a group of devices to compute the individual $Q , K , V$ components concurrently. These methods require complex communication operations to gather these shards, allowing attention computation to be performed with complete information across the sequence dimension.

Similar to sequence or context parallelism, SA also divides the entire input sequence $X$ into $T$ sub-sequences, denoted as $\{ X _ { 1 } , X _ { 2 } , \cdot \cdot \cdot , X _ { T } \}$ . Differently, SA sequentially computes the query, key, and value states for each sub-sequence on the same device:

$$
\begin{array} { r l } & { Q _ { t } = X _ { t } W _ { Q } , \quad K _ { t } = X _ { t } W _ { K } , } \\ & { V _ { t } = X _ { t } W _ { V } , \quad t \in \{ 1 , 2 , \cdot \cdot \cdot , T \} . } \end{array}
$$

Leveraging the recurrent form of linear sequence modeling as described in Eq. 3, the memory state variation $\widehat { M _ { t } }$ for each sub-seq $t$ can be computed and accumulated ntco $M _ { t }$ sequentially. The finally updated memory state $M _ { t }$ is then utilized to produce the output $O _ { t }$ . The accumulation process in SA closely mirrors the recurrent computation form of linear sequence modeling and can be expressed as:

$$
\begin{array} { r l } & { \widehat { M } _ { t } = f ( K _ { t } ^ { \top } , V _ { t } ) , } \\ & { M _ { t } = \Theta \diamond M _ { t - 1 } + \widehat { M } _ { t } , } \\ & { O _ { t } = Q _ { t } M _ { t } , } \\ & { \quad t = 1 , 2 , \cdots , T , } \end{array}
$$

with the initial memory state

$$
M _ { 0 } = 0 \in \mathcal { R } ^ { d \times d } .
$$

The specific selections of $f , \Theta$ , and $\diamond$ in Eq. 5 determine how the SA computation process is instantiated. For instance, in the basic linear transformer model used in our experiments, $f ( K _ { t } ^ { \top } , V _ { t } )$ is defined as $K _ { t } ^ { \top } V _ { t }$ , $\Theta$ is set as a constant $\lambda \in \mathcal { R }$ , and $\diamond$ corresponds to the matrix multiplication.

The forward computation for the $t$ -th sub-sequence depends on the memory state from the preceding $( t - 1 )$ subsequences, denoted as $M _ { t - 1 }$ , which is precomputed and can be reused on the same device. During the backward pass, gradient computation for the $t$ -th sub-sequence requires $M _ { t }$ as an intermediate activation. To prevent redundant computation of $M _ { t }$ and to expedite subsequent forward passes, we store the updated memory state $M _ { t }$ in the High-Bandwidth Memory (HBM) of the GPU. This allows $M _ { t }$ to be efficiently accessed during both the backward pass and the next forward pass. The forward procedure for SA is detailed in Algorithm 1, and the backward pass is provided in Algorithm 2. The backward pass is not performing any sort of truncation as is typical in truncated backpropagation through time in RNNs.

# Algorithm 1 Sequence Accumulation (Forward)

1: Input: input sequence in embedding space X ∈ RN×d.   
2: Split $X$ into $T$ sub-seqs $\{ X _ { 1 } , X _ { 2 } , \cdot \cdot \cdot , \bar { { , } } X _ { T } \}$ , obtain sub  
seq length $S = N / T$ .   
3: Initialize memory state $M _ { 0 } = 0 \in \mathcal { R } ^ { d \times d }$ .   
4: Instantiate $f , \Theta$ and $\diamond$ .   
5: for sub-seq $t \in \{ 1 , 2 , \cdots , T \}$ on the same device do   
6: Compute $Q _ { t } = X _ { t } W _ { Q }$ , $K _ { t } = X _ { t } W _ { K }$ , $V _ { t } = X _ { t } W _ { V }$ .   
7: Compute $\widehat { M _ { t } } = f ( K _ { t } ^ { \top } , V _ { t } )$ .   
8: Accumulatce $M _ { t } = \Theta _ { t } \diamond M _ { t - 1 } + \widehat { M } _ { t }$ .   
9: Compute $O _ { t } = Q _ { t } M _ { t }$ .   
0： end fo

# Algorithm 2 Sequence Accumulation (Backward)

1: Input: $Q _ { t } , K _ { t } , V _ { t } , O _ { t } , d O _ { t }$ for $t = T$ .   
2: Initialize $d M _ { T + 1 } = 0 \in \mathcal { R } ^ { d \times d }$ .   
3: for $t \in \{ T , \cdots , 2 , 1 \}$ on the same device do   
4: Compute $d Q _ { t } = d O _ { t } M _ { t } ^ { \top }$ .   
5: Compute ${ \widehat { d M } } _ { t } = Q _ { t } ^ { \top } d O _ { t }$ .   
6: Accumulated $d M _ { t } = \widehat { \Theta } \diamond d M _ { t + 1 } + \widehat { d M } _ { t }$ .   
7: Compute $d K _ { t } = V _ { t } d M _ { t } ^ { \top }$ .   
8: Compute $d V _ { t } = K _ { t } d M _ { t }$ .   
9: end for   
10: Return: $d Q = [ d Q _ { t } ]$ , $d K = [ d K _ { t } ]$ , $d V = [ d V _ { t } ]$ , wi   
$t \in \{ 1 , 2 , \cdots , T \}$ .

Note that the accumulated memory state $M _ { t }$ has a shape of $d \times d$ , so storing $M _ { t }$ incurs only a small memory cost, which is independent of the context length $N$ or sub-seq length $S$ . Additionally, no communication operations are required in SA, as the entire process is executed on the same device.

Otherwise, SA can be used in conjunction with DP, TP or even CP. In these scenarios, the SA process runs concurrently on each device, with synchronization occurring between devices at each iteration within the data, tensor or context parallelism communication groups.

# Sequence Accumulation Pipeline Parallelism

To enable the application of SA to large models with long sequences, we discovered that SA can be seamlessly integrated with PP. This integration allows for efficient training of long context lengths on large models on limited GPU resources. For the sake of simplifying the analysis of SAPP, we use GPipe (Huang et al. 2019; Kim et al. 2020) as a representative example of PP. A detailed exploration of more complex PP methods is outside the scope of this paper.

Consider a deep neural network with a sequence of $L$ layers, each layer $L _ { i }$ is composed of a forward computation function $f _ { i }$ . Considering the consecutive layers between layers $i$ and $j$ , its forward function would be $F _ { i j } = f _ { j } \circ \cdot \cdot \cdot \circ f _ { i + 1 } \circ f _ { i }$ . The corresponding back-propagation function $B _ { i j }$ can be computed from $F _ { i j }$ using automatic symbolic differentiation. PP partitions the network into $P$ cells and places the $p$ -th cell on the $p$ -th accelerator. Communication primitives are automatically inserted at partition boundaries to allow data transfer between neighboring partitions. During the forward pass, PP first divides every mini-batch into $B$ equal microbatches, which are pipelined through the $P$ accelerators. During the backward pass, gradients for each micro-batch are computed based on the same model parameters used for the forward pass. At the end of each mini-batch, gradients from all $B$ micro-batches are accumulated and applied to update the model parameters across all accelerators. This process of PP is illustrated in Figure 2.

![](images/1017c57313ccbea8217db63013be7cccf250a4466f53a76ac0d0ac55f8151430.jpg)  
Figure 1: Sequence Accumulation (SA). The entire input sequence is divided into multiple sub-sequences, which are processed sequentially on the same device to compute their respective memory states. These memory states are progressively accumulated into a single state, which is then used to generate the output.

![](images/0dbf16a67beb64bb105eb2fb0abe830eecb1a220c7548bcd327918b8b1a0f0eb.jpg)  
Figure 2: Sequence Accumulation Pipeline Parallelism (SAPP). We illustrate the operation of SAPP using a setup with 4 GPUs, employing code-style indexing to maintain consistency with other PP illustrations. The model is divided into 4 shards, each allocated to one of the 4 GPUs, and the input sequence is split into 4 sub-seqs, all initially placed on GPU0. Here, $F _ { 0 , 1 }$ represents the forward pass of sub-seq1 on GPU0, while $B _ { 0 , 1 }$ denotes its backward pass. Black arrows indicate the pipeline communication of layer activations via point-to-point send/receive operations, while red arrows represent the local write/read operations for SA on the same device. Memory states for each model shard are accumulated on their respective GPUs without requiring any inter-device communication, facilitating the training of large models with long sequences.

When integrating SA with PP, the concept of $B$ microbatches in traditional pipeline parallelism is replaced by $S$ sub-sequences, representing the partitioned segments of the entire input sequence. In this setup, the data flow of SA occurs exclusively within the forward functions on the same accelerator, which is also within the same model shard. For instance, the data flow follows the pattern on device with a index of 0:

$$
F 0 , 0 \to F 0 , 1 \to \cdots \to F 0 , S - 1 .
$$

Since these operations all take place on the same device, there is no need for inter-device communication, making the integration of SA with PP highly efficient with no additional communication overhead. This inherent efficiency allows the combined approach to handle increasingly longer context lengths as the number of sub-sequences $S$ increases. Furthermore, similar to traditional PP, the "bubble time"—the idle time during pipeline execution—decreases as the number of sub-sequences grows, further optimizing the overall training process. This characteristic makes the SAPP particularly well-suited for efficiently training large models on extremely long sequences, leveraging the full capacity of the limited computational resources.

# Experiments

We performed thorough evaluations of the SA and SAPP methods, focusing on their convergence behavior, computational speed, and memory efficiency. To test SA’s capability of handling infinite context lengths, we first evaluated it on a single GPU, ensuring its effectiveness in scenarios with extremely long sequences. We then expanded our evaluation to multiple GPUs, integrating SA with DP to explore its scalability. For SAPP, we assessed its performance in both singlenode and multi-node configurations, further demonstrating its flexibility and efficiency when combined with various parallel training techniques such as DP, TP, and CP. These experiments were conducted on a state-of-the-art GPU cluster comprising 64 A100 GPUs, each with 80GB of memory. The implementation was built on the Megatron-LM framework (Shoeybi et al. 2019; Korthikanti et al. 2022), which provided a robust foundation for testing the scalability and applicability of our methods on large-scale models and extensive computational resources.

![](images/1db9349732a2147061de5f4c6ff91a7bad3fd8b2371c5efd7254b4b6ad6394d7.jpg)  
Figure 3: Performance of SA on Single GPU. The experiment utilizes a model with 1 billion parameters and a batch size of 4, executed on a single GPU.

Table 2: Convergence Results of SA and SAPP. All experiments were conducted on a single node with 8x A100 80GB GPUs, using a basic linear transformer model with 0.4 billion parameters. The sequence length was set to 16K (16,384 tokens), with a batch size of 4. Each experiment utilized a specific number of GPUs, determined by DP size $\times \mathrm { P P }$ size. Additionally, the total number of sub-seqs processed in each iteration was calculated as DP size $\mathbf { \nabla } \times \mathbf { S } \mathbf { A }$ size. The baseline experiment, which did not involve DP, PP, or SA, required 512K iterations. To ensure that all experiments processed an equivalent amount of data tokens, we adjusted the number of iterations accordingly.   

<html><body><table><tr><td>Method</td><td>SA-PP-DP</td><td>Num of GPUs (DP × PP)</td><td>Sub-seqs per Iter (DP × SA)</td><td>Iterations</td><td>Loss</td></tr><tr><td>Baseline</td><td>1-1-1</td><td>1</td><td>1</td><td>512K</td><td>3.710</td></tr><tr><td>SA</td><td>8-1-1</td><td>1</td><td>8</td><td>64K</td><td>3.705</td></tr><tr><td>PP</td><td>1-8-1</td><td>8</td><td>1</td><td>512K</td><td>3.709</td></tr><tr><td>DP</td><td>1-1-8</td><td>8</td><td>8</td><td>64K</td><td>3.714</td></tr><tr><td>PP&DP</td><td>1-2-4</td><td>8</td><td>4</td><td>128K</td><td>3.711</td></tr><tr><td>SA&DP</td><td>2-1-4</td><td>4</td><td>8</td><td>64K</td><td>3.708</td></tr><tr><td>SAPP</td><td>2-4-1</td><td>4</td><td>2</td><td>256K</td><td>3.712</td></tr><tr><td>SAPP&DP</td><td>4-4-2</td><td>8</td><td>8</td><td>64K</td><td>3.709</td></tr></table></body></html>

# Experimental Setup

In all experiments presented in this paper, we employ the Adam optimizer, configured with beta values of 0.9 and 0.999, to ensure efficient training while mitigating potential issues such as gradient instability (Tang et al. 2023). To further enhance regularization and prevent overfitting, a weight decay rate of 0.01 is applied. We set the learning rate at 0.0005, accompanied by a warmup phase spanning 2000 updates, which gradually increases the learning rate to stabilize the early stages of training (Zhou et al. 2020; Sun et al. 2024b). The total number of updates varies depending on the specific training configurations used in each experiment. For pretraining, we utilize the Pile (Gao et al. 2020) dataset, a comprehensive and diverse open-source language modeling dataset consisting of 825 GB of text data (Qin et al. 2023a).

# Convergence

Both SA and SAPP are theoretically accurate training methods. To validate their convergence, we conducted experiments using a 0.4B parameter basic linear transformer model with a fixed context length of 16K. The results are summarized in Table 2. In this experiment, we utilized a total of 8 GPUs, applying various configurations of SA, PP, and DP to explore different method combinations. The baseline model, which does not employ DP, PP, or SA, required 512K iterations to achieve a loss value of 3.710. For the other configurations, the number of GPUs used was determined by the product of the DP size and PP size, while the number of sub-sequences processed in each iteration was calculated as the product of the DP size and SA size. To ensure a fair comparison, we adjusted the number of iterations across different experiments so that all models processed the same amount of data tokens, thus targeting a similar loss value. As shown in the table, regardless of whether SA, SAPP, or their combinations with DP were used, all methods achieved comparable loss values, demonstrating the convergence accuracy of these approaches.

![](images/aa4096a6c153983dbe90ec87985b121ca8671c63b5ba992c4a34605cb11607a0.jpg)  
Figure 4: Performance Comparison of PP and SAPP. This experiment uses a model with 1 billion parameters and a batch size of 4 across four GPUs, which also represents the size of PP configuration. It is observed that PP encounters OOM when the context length reaches 32K.

<html><body><table><tr><td>Context Length</td><td>SA-PP-DP-TP-CP</td><td>GPU Memory (GB)</td><td>Throughput (tokens/s)</td><td>Iteration Time (s)</td></tr><tr><td>2K</td><td>1-2-4-2-4</td><td>18.2</td><td>11256</td><td>2.4</td></tr><tr><td>4K</td><td>2-2-4-2-4</td><td>18.2</td><td>11984</td><td>4.6</td></tr><tr><td>8K</td><td>4-2-4-2-4</td><td>18.2</td><td>12565</td><td>9.3</td></tr></table></body></html>

Table 3: Compatibility with Hybrid Parallelism. This experiment involves a 32-layer model with 7 billion parameters and utilizes a distributed setup across 64 GPUs. The length of each sub-sequence is fixed at 2K.

# Memory and Speed

We conducted a series of experiments to evaluate the memory usage and speed performance of both SA and SAPP. The SA method was tested on a single GPU, while SAPP was tested on a node with 4 GPUs, both using a linear transformer model containing 1 billion parameters. The experiments employed a batch size of 4, and the context length was progressively increased from 2K to 1024K to assess the changes in GPU memory usage, the number of sub-seqs, throughput, and the time per iteration.

When testing SA on a single GPU, as depicted in Figure 3, the context length of each sub-seq was fixed at 2K. Since SA accumulates the KV states of all sub-seqs, the memory usage per GPU remained constant as the context length increased. However, the number of sub-sequences and the time per iteration increased rapidly with longer context lengths. Throughput saw only a modest increase, which was expected, given that SA sequentially computes all sub-sequences and accumulates their KV states on a single device.

In contrast, Figure 4 illustrates the performance of SAPP on 4 GPUs within a single node and compares it to the standard PP approach. The results show that with PP, both the memory usage per GPU and throughput increased sharply as the context length doubled, eventually leading to an outof-memory (OOM) error when the context length reached 32K. On the other hand, SAPP maintained a constant memory usage per GPU and exhibited a more gradual increase in throughput as the context length grew, demonstrating its efficiency in handling long sequences without overwhelming the available memory resources.

# Hybrid Parallelism: SA-PP-DP-TP-CP

To showcase the capabilities of SAPP when integrated with existing distributed training methods, we evaluate SAPP in conjunction with DP, TP, and CP. As detailed in Table 3, we maintained a fixed sub-sequence length of 2K and varied the SA sizes to $\{ 1 , 2 , 4 \}$ , corresponding to total context lengths of $\{ \mathrm { 2 K , 4 K , 8 K } \}$ . The results reveal that, similar to our previous findings with SA and SAPP, while the iteration time increases significantly with longer context lengths, the memory usage remains consistent. Throughput shows a moderate increase. This stability in memory usage and gradual improvement in throughput are attributed to the way SA accumulates intermediate computation states across multiple sub-sequences.

# Performance on Benchmarks

We have performed additional benchmark experiments, with the results summarized in Table 4. To ensure consistency, we utilized the pretrained models from the Baseline, SA, and SAPP experiments, as outlined in Table 2. The results indicate that both SA and SAPP achieve generalization performance that is very close to that of the Baseline across standard benchmarks.

Table 4: Performance Comparison Across Various Benchmarks. Both SA and SAPP achieve very close generalization performance comparing with that of the Baseline.   

<html><body><table><tr><td>Method</td><td>LMB. PIQA</td><td>Hella.</td><td>Wino.</td><td>ARC-e</td><td>ARC-c</td></tr><tr><td>Baseline</td><td>30.9</td><td>64.8 36.1</td><td>50.5</td><td>42.9</td><td>23.6</td></tr><tr><td>SA</td><td>30.7</td><td>65.3 36.5</td><td>49.6</td><td>42.2</td><td>23.7</td></tr><tr><td>SAPP</td><td>30.6</td><td>65.6 35.8</td><td>49.9</td><td>42.7</td><td>23.9</td></tr></table></body></html>

# Discussion

The limitations of linear attention in comparison to softmax attention can be summarized in two main points: 1) Modeling performance: Linear attention sometimes fails to match the performance of softmax attention, particularly in tasks that require high recall. 2) Support technologies: The supporting technologies for linear attention models, such as CUDA kernels, parallel training, quantization, inference acceleration, and post-training enhancements, are not as mature as those available for softmax attention models.

When comparing linear attention with methods like RWKV, both are sequence modeling techniques that operate with linear complexity, utilizing similar RNN-like recurrence structures. These methods benefit from the advantages of linear-time training and constant memory usage during inference. However, their underlying designs are distinct: linear attention is a variant of softmax attention, whereas RWKV is rooted in linear RNN principles. The primary advantage of linear attention lies in its simplicity, as it only modifies standard attention by removing the softmax operation and reordering matrix multiplications. On the other hand, RWKV, based on linear RNNs, introduces a more complex architecture with advanced mechanisms for token mixing, time mixing, and channel mixing.

The proposed SA shares similarities with the gradient accumulation (GA) approach, both in terms of technical idea and practical application. GA has proven useful for simulating larger batch sizes on devices with limited memory, making it possible to conduct large (theoretically infinite) batch training on a single GPU. Similarly, SA enables training on long (theoretically infinite) sequences with restricted memory resources. For users with limited computational resources, such as a few GPUs, personal computers, or even mobile devices, SA allows for model fine-tuning on very-long-context inputs (which is infeasible without SA). Meanwhile, for those with access to extensive resources, like large GPU clusters or data centers, SAPP provides a solution for training large models on lengthy input sequences.

# Related Work Distributed Training

Distributed training techniques enhance model training efficiency across multiple devices (Team 2023). Pipeline parallelism (Huang et al. 2019; Kim et al. 2020; Narayanan et al. 2019) divides the model into stages, allowing simultaneous computation and communication across devices. Sequence or context parallelism (Li et al. 2021; Korthikanti et al. 2022; Jacobs et al. 2023; Liu, Zaharia, and Abbeel 2023; Sun et al. 2024a) processes long sequences in smaller segments concurrently to manage memory and speed up training. Gradient accumulation (You et al. 2019) are often utilized to accumulates gradients over several mini-batches before updating model parameters, simulating large batch training with limited resources.

# Linear Sequence Modeling

Linear Transformer models avoid using Softmax attention by employing various approximation techniques (Katharopoulos et al. 2020; Choromanski et al. 2020; Peng et al. 2021; Qin et al. 2022b,a). These methods leverage the "kernel trick" to accelerate the computation of attention matrix. The chunk-wise approach to linear attention was initially introduced in FLASH (Hua et al. 2022), who utilized a hybrid model that combined linear and nonlinear attention mechanisms. This chunk-wise linear attention also has been independently developed in several other studies (Kacham, Mirrokni, and Zhong 2023; Sun et al. 2023; Yang et al. 2023; Qin et al. 2024b,c). GLA (Yang et al. 2023) and Lightning Attention (Qin et al. 2024b,c) focus on optimizing I/O operations for chunk-wise linear attention, while LASP (Sun et al. 2024a) extends these methods to multi-node distributed training environments. Additionally, Mamba2 (Dao and Gu 2024) provides a comprehensive theoretical framework connecting Structured Semi-Separable Matrices with linear attention through matrix de-compositions. And DeltaNet (Yang et al. 2024) presents hardware-efficient DeltaNet training.

# Conclusion

In this paper, we have introduced SA and its extension SAPP, to address the challenge of managing extremely long sequences with limited computational resources. SA leverages the recurrence feature inherent in linear sequence modeling methods to handle infinite context lengths efficiently, even on a single GPU. By dividing long sequences into fixed-length sub-sequences and accumulating intermediate states sequentially, SA achieves constant memory consumption, making it feasible to work with extremely contexts without excessive memory usage. Furthermore, we have extended SA by integrating it with PP to develop SAPP, which allows for the training of large models with infinite context lengths on large GPU clusters without additional communication costs in the sequence dimension. Our extensive experiments demonstrate the effectiveness of both SA and SAPP across various context lengths and GPU configurations. The results highlight that SA and SAPP not only enable the training of infinite context lengths under constrained resources but also integrate seamlessly with existing distributed training methods.

# Acknowledgments

This work was supported by the National Key R&D Program of China (No. 2022ZD0160201) and Shanghai Artificial Intelligence Laboratory.