# Practical Offloading for Fine-Tuning LLM on Commodity GPU via Learned Sparse Projectors

Siyuan Chen1\*, Zhuofeng Wang2, Zelong Guan1, Yudong ${ { \bf { L i u } } ^ { 1 } }$ , Phillip B. Gibbons1

1 Carnegie Mellon University, 2 Peking University siyuanc3@andrew.cmu.edu, wangzf2003 $@$ stu.pku.edu.cn, zelongg $@$ andrew.cmu.edu, yudongltech $@$ gmail.com, gibbons@cs.cmu.edu

# Abstract

Fine-tuning large language models (LLMs) requires significant memory, often exceeding the capacity of a single GPU. A common solution to this memory challenge is offloading compute and data from the GPU to the CPU. However, this approach is hampered by the limited bandwidth of commodity hardware, which constrains communication between the CPU and GPU, and by slower matrix multiplications on the CPU. In this paper, we present an offloading framework, LSPOffload, that enables near-native speed LLM fine-tuning on commodity hardware through learned sparse projectors. Our data-driven approach involves learning efficient sparse compressors that minimize communication with minimal precision loss. Additionally, we introduce a novel layer-wise communication schedule to maximize parallelism between communication and computation. As a result, our framework can fine-tune a 1.3 billion parameter model on a 4GB laptop GPU and a 6.7 billion parameter model on a 24GB NVIDIA RTX 4090 GPU. Compared to state-of-the-art offloading frameworks, our approach reduces end-to-end fine-tuning time by $3 3 . 1 \% - 6 2 . 5 \%$ when converging to the same accuracy.

Code — https://github.com/gulang2019/LSP-Offload

# 1 Introduction

Recent years have highlighted the remarkable success of billion scale LLMs. Hand-to-hand with task performance improvements are the ever-growing model sizes and the strong demand for powerful computing resources that are available only in high-end clusters. Fortunately, fine-tuning provides everyday ML practitioners the accessibility to LLMs by allowing them to adapt a pre-trained model to downstream tasks using less onerous computational effort. However, finetuning’s memory and compute demands are still daunting. For example, under a default fine-tuning configuration that uses the fp16 data type with the Adam optimizer (Kingma and Ba 2014), the memory footprint is $8 \times$ #Parameters bytes, which means top-notch commodity workstation GPUs (e.g., NVIDIA 4090 GPU and AMD 7900XTX with 24GB memory each) are able to hold only smaller LLMs (3B parameters). With commodity laptop GPUs (e.g., NVIDIA A1000 with 4GB memory), even 0.77B parameter LLMs do not fit.

![](images/8cb22ab94e71c5a295fdb073a520b2ee69a93c011791460e3d904757a059ce11.jpg)  
Figure 1: LSP-Offload

A variety of techniques have been proposed to reduce the memory demand during fine-tuning. A typical solution from system researchers is to offload part of the compute and memory from GPU to CPU, leveraging the fact that commodity laptop CPUs typically have $4 \mathrm { x }$ the memory of laptop GPUs and commodity workstation CPUs can provide 4TBs of memory (per socket). Although offloading is able to scale the trainable model size, large batch sizes are essential to remain efficient despite the limited PCIe bandwidth between CPU and GPU (Rajbhandari et al. 2021). In fact, we show that training with offloading is inherently bounded by either the CPU-GPU communication or the compute on CPU, especially on commodity hardware where the limited GPU memory dictates small batch sizes. Therefore, offloading itself can hardly save us from the scaling challenge.

Meanwhile, another promising method from ML researchers for memory-reduction is parameter-efficient finetuning (PEFT). The key idea of PEFT is to limit the trainable parameters to a carefully designed subspace (e.g., a low rank subspace (Hu et al. 2021; Zhao et al. 2024) or only part of the model (Guo, Rush, and Kim 2020)), so the GPU can train the model without offloading as long as it can hold the parameters and minimal optimizer states for the trainable parameters. However, though more memory-efficient, PEFT methods can suffer from slow convergence or sub-optimal training results due to their overly constrained space for parameter updates.

In this paper, we show how to mitigate the memory challenge by combining both approaches. We present LSPOffload (Fig. 1), a novel fine-tuning framework that (i) mitigates bottlenecks in offloading approaches by a new approach to refactor the offloading process and (ii) trains efficiently by a new approach to constrain the optimization space.

Specifically, to alleviate the compute pressure on the CPU as well as the communication overhead back-and-forth between CPU and GPU, we constrain the updates to happen on a periodically-changing subspace ( $S$ in Fig. 1). Because the updates from different subspaces are projected back and accumulate together in the original space, the model is able to update in the full-rank optimization space. State-of-theart (SOTA) approaches (Hu et al. 2021; Zhao et al. 2024) for constraining the parameter update space suffer from linear memory and compute complexity that limits them from optimizing in large subspaces. We solve this problem by the introduction of $( d , r )$ -sparse projectors $\cdot P _ { t }$ and $Q _ { t }$ in Fig. 1), sparse embedding matrices that represent a subspace but whose memory consumption is independent of the subspace’s size. In this way, given the same memory budget as PEFT, we are able to optimize in an arbitrary-size subspace. To further improve the compression quality of the subspace, we adopt a data-driven approach similar to (Liu et al. 2020) that adapts the subspace to the gradient matrices, which is empirically proven necessary for fast convergence.

Moreover, at the system level, we demonstrate that the SOTA offloading framework Zero-Offload (Rajbhandari et al. 2020) suffers from limited parallelism between communication and compute when running on commodity hardware. This is due to the limited GPU memory relative to the model size, which implies that only small batch sizes can be used during training. We improve Zero’s schedule by performing fine-grained communication on the granularity of layers and communicating components of the gradient ahead of time. The new schedule enables us to explore the full parallelism between CPU compute, GPU compute, CPU-to-GPU communication, and GPU-to-CPU communication.

In summary, our paper makes the following contributions:

• We analyze LLM training on commodity hardware (both laptop and workstation) to show that current offloading workflows are fundamentally bounded by either the communication or the CPU’s compute. • We design LSP-Offload to enable near-native speed finetuning on commodity hardware. The system is built on the key idea of learned sparse projectors, which enables finetuning on high-dimensional subspaces with constant memory and compute overhead. We open source our framework at https://github.com/gulang2019/LSP-Offload. • We verify that LSP-Offload converges to the same accuracy as native training on the GLUE dataset. For instruction-tuning, LSP-Offload reduces end-to-end finetuning time by $3 3 . 1 \%$ to $6 2 . 5 \%$ over SOTA offloading, when converging to the same accuracy. Moreover, LSPOffload improves accuracy by $2 7 . 8 \%$ to $30 \%$ over SOTA PEFT approaches on the Alpaca and Humaneval datasets.

# 2 Background and Related Work

Memory breakdown for training large language models. Training a deep learning model requires memory for parameters, activations, and optimizer states. Activations include intermediate results used in backward propagation. The optimizer states are used by the optimizer to update the parameters. Of the three, the memory for parameters $( M _ { p a r a m } )$ and optimizer states $( M _ { o p t } )$ consume most of the memory. When trained with the Adam optimizer and half precision, $M _ { p a r a m } + M _ { o p t } \approx 8 \times \#$ Parameters bytes, which easily exceeds the single GPU’s memory for billion-scale models.

Memory offloading. Memory offloading techniques (Huang, Jin, and Li 2020; Rajbhandari et al. 2020, 2021; Ren et al. 2021; Zhang et al. 2023) enable training a full model on limited GPU memory by utilizing CPU memory or SSDs. Among these, Zero series are the SOTA approaches for fine-tuning large models. Zero-Offload (Ren et al. 2021) offloads the optimizer states and the update step to the CPU. Compared to other approaches that offload only the memory to CPU and do all computations on GPU, Zero-Offload achieves the optimal communication volume for full parameter training. Nevertheless, we found that Zero’s training is severely bottlenecked by the communication (see Fig. 2).

Parameter-efficient fine-tuning. PEFT enables pretrained models to rapidly adapt to downstream tasks with minimal extra memory required. LoRA (Hu et al. 2021) is among the most popular PEFT techniques by constraining the optimization onto a decomposed low-rank subspace. However, recent works (Lialin et al. 2023; Valipour et al. 2022) found LoRA is sensitive to hyperparameter tuning and can struggle with tasks requiring significant change to the base model. To break the low-dimensional constraint of LoRA, GaLore (Zhao et al. 2024) explores a similar idea to ours that periodically changes the subspace computed by singularvalue-decomposition (SVD). However, both LoRA and GaLore have the limitation that their extra memory and compute requirements scale linearly with the subspace’s size (rank), which inherently prevent them from tuning on a higher dimensional subspace. Our work mitigates this problem via novel subspace projectors whose compute and memory demands are independent of the subspace size, enabling us to achieve better model accuracy by tuning in a larger subspace. Moreover, a contemporary work (He et al. 2024) explores the similar idea to reduce memory overhead via PEFT but using a sparse matrix approach.

Other methods for memory-efficient training. Various approaches such as quantization (Dettmers et al. 2024) and gradient checkpointing (Chen et al. 2016) have been proposed to reduce the memory demand for training/fine-tuning LLMs. The quantization approach uses data types with fewer bits for training, and is fully compatible with our techniques (we use fp16 in our evaluations). Meanwhile, the gradient checkpointing technique trades computation for memory by recomputing activations during the backward pass. We include this technique in our implementation.

# 3 Motivation

# Numerical Analysis for Fine-tuning on a GPU

We motivate our work by an analysis on the fundamental limits of vanilla offloading on a single commodity GPU. We use the example setting of fine-tuning a llama-7B model on a commodity workstation GPU (Nvidia RTX 4090), which provides only 2 $4 / ( 1 4 + 4 2 + 8 ) = 3 7 . 5 \%$ of the required memory (Tab. 1). (A similar analysis can be done for the GPT2-1.3B model on a commodity laptop GPU (Nvidia A100)–see the full version of our paper (Chen et al. 2024).)

Table 1: Configurations and timings for training/fine-tuning the llama-7B Model (using fp16) on commodity workstation hardware—the Nvidia RTX 4090 GPU and AMD Ryzen Threadripper 3970X CPU. For UPD, we measure the fused Adam kernel with thread-level parallelism and SIMD optimizations. Bandwidth is the PCIe bandwidth with a pinned memory buffer.   

<html><body><table><tr><td>Parameters</td><td>Optimizer State</td><td>Activations</td><td>CPU-GPU Bandwidth</td><td>#Layers</td><td>GPU Memory</td></tr><tr><td>14GB</td><td>42GB</td><td>8GB</td><td>10-20GB/s</td><td>32</td><td>24GB</td></tr><tr><td>FWD on CPU</td><td>BWD on CPU</td><td>UPD on CPU</td><td>FWD on GPU</td><td>BWD on GPU</td><td>UPD on GPU</td></tr><tr><td>1.61s/layer</td><td>3.30s/layer</td><td>0.06s/layer</td><td>12.2ms/layer</td><td>28.1ms/layer</td><td>1ms/layer</td></tr></table></body></html>

Current offloading techniques can be categorized into two classes: (i) those that offload only memory to the CPU, and (ii) those that offload both memory and compute to the CPU. The first class is represented by (Huang, Jin, and Li 2020; Zhang et al. 2023), which perform all compute on the GPU while swapping in and out of GPU memory on the fly. An example of this type of schedule is shown in Fig. 3c. However, this type of offloading schedule is inherently bounded by the communication under the following observation:

Observation. Training a model demanding $M _ { t o t }$ memory on a GPU with only $M _ { g p u }$ memory, such that the GPU performs all the computation, requires $\ge M _ { t o t } - M _ { g p u }$ of communication per iteration.

For our setting, we need 2-4s communication per iteration $( 6 4 - 4 0 \$ divided by 10-20). Since the per-iteration training time is 1.32s (32 layers $\times ( . 0 1 2 2 + . 0 2 8 1 + . 0 0 1 )$ ), this adds $0 . 5 \mathrm { x }$ to $2 . 0 \mathrm { x }$ overhead compared to the GPU compute even if compute and communication are fully overlapped.

The second class divides the workload between the CPU and GPU. Because of the CPU’s limited computing power, only the parameter update step (UPD) is suitable to run on the CPU. For example, assigning the $\mathrm { F W D + B W D }$ pass of just one layer to the CPU directly adds $1 . 6 1 + 3 . 3 0 = 4 . 9 1 \mathrm { s }$ overhead, which is already $3 . 7 \mathrm { x }$ the per-iteration GPU compute. Moreover, offloading UPD to the CPU (more specifically, the computation of $\Delta W$ to the CPU—applying these deltas to the model parameters remains on the GPU) means that the 42GB optimizer state can reside on the CPU, enabling larger models like llama-7B to fit in the GPU memory.

Offloading UPD to the CPU was first realized in ZeroOffload (Ren et al. 2021), whose schedule is displayed in Fig. 3a. In their schedule, $2 M _ { p a r a m }$ communication happens every iteration (gradients to CPU, deltas to GPU, both having the same size of Parameters), implying the communication time is 1.4-2.8s, which is already up to $2 . 1 \mathrm { x }$ the GPU compute time (1.29s). Moreover, the CPU compute can become the bottleneck for Zero’s schedule. For the example setting, UPD on the CPU takes 1.92s per iteration. When there is no overlap between CPU compute and GPU compute (Fig. 3a), this adds $1 . 5 \mathrm { x }$ overhead compared to GPU compute.

This analysis shows that training with offloading is computationally inefficient on modern commodity hardware due to fundamental bottlenecks in communication and/or CPU compute. This motivates us to design a lossy (PEFT

5 Laptop 4.28 Workstation Normalized 4 Oof 4GB GPU 24GB GPU 3.38 Lrnaf 3 2 2.43 1.93 S 1 0 GPT2-774M GPT2-1.3B llama-3B llama-7B BS: 4 1 16 8 GPU Comm CPU Other style) algorithm for reduced overheads when offloading.

# Case Study on Zero’s Schedule

Complementing our analysis, we study Zero-Offload in two settings: (i) training a GPT2 model on a 4GB laptop GPU, and (ii) training a llama model on a 24GB workstation GPU. The slowdown normalized by the GPU compute time is shown in Fig. 2. Under both configurations, Zero’s schedule slows training by $1 . 9 3 \mathrm { x }$ to $4 . 2 8 \mathrm { x }$ , for the following two reasons.

Communication and CPU compute overhead. The primary source of overhead comes from the unavoidable high communication volume and slow CPU compute as demonstrated in our previous analysis. Shown in Fig. 2, although Zero is able to overlap part of the GPU/CPU compute with communication, the non-overlapped communication brings $0 . 6 1 \mathrm { x }$ to $2 . 0 9 \mathrm { x }$ added slowdown compared to the GPU compute time. For both the laptop and workstation GPUs, the situation is worse for the larger model due to the decrease in the largest batch size that fits. When training a 1.3B model on a 4GB GPU, the non-overlapped communication and CPU compute are $2 . 0 9 \mathrm { x }$ , 0.63x the GPU compute, respectively.

Limited parallelism between CPU and GPU, communication and compute. The second source of overhead comes from Zero’s limited parallelism between compute and communication. Fig. 3a shows Zero’s standard training pipeline, which is suboptimal for two reasons: (i) FWD and BWD on the GPU are not overlapped with the CPU’s compute. This results in significant slowdown when the CPU compute is

Step 𝐼𝐼 Step 𝐼𝐼 + 1… Step 𝐼𝐼 Step 𝐼𝐼 + 1 GPU 𝐹𝐹𝐹𝐹𝐷𝐷𝐼𝐼 𝐵𝐵𝐵𝐵𝐷𝐷𝐼𝐼 𝐹𝐹𝐹𝐹𝐷𝐷𝐼𝐼+1 𝐵𝐵𝐵𝐵𝐷𝐷𝐼𝐼+1 GPU 𝐹𝐹𝐹𝐹𝐷𝐷𝐼𝐼 𝐵𝐵𝐵𝐵𝐷𝐷𝐼𝐼 𝐹𝐹𝐹𝐹𝐷𝐷𝐼𝐼+1 𝐵𝐵𝐵𝐵𝐷𝐷𝐼𝐼+1 𝐹𝐹𝐹𝐹𝐷𝐷𝐼𝐼+2 𝐵𝐵𝐵𝐵𝐷𝐷𝐼𝐼+2   
GPU  CPU ∇𝑊𝑊𝐼𝐼 ∇𝑊𝑊𝐼𝐼+1 GPU  CPU ∇𝑊𝑊𝐼𝐼 ∇𝑊𝑊𝐼𝐼+1 ∇𝑊𝑊𝐼𝐼+2   
CPU  GPU Δ𝑊𝑊𝐼𝐼 CPU  GPU Δ𝑊𝑊𝐼𝐼−1 Δ𝑊𝑊𝐼𝐼 Δ𝑊𝑊𝐼𝐼+1 CPU UPDI CPU UPDI UPDI+1 Comm. Contention Data Dependency a) Zero-Offload Schedule b) Zero-Offload Schedule w/ Delayed Parameter Update Step 𝐼𝐼 Step 𝐼𝐼 + 1… Step 𝐼𝐼 Step 𝐼𝐼 + 1   
GCPU $$ CGPU 1 231F  3  4W2  4    53D 5 64 6 6 564B   4 53W   3 42D 231 1 GCPU  CGPU 𝐹𝐹𝐹𝐹𝐷𝐷𝐼𝐼෡∇𝑊𝑊𝐼𝐼෡Δ𝐵𝑊𝐵 𝑊𝐼𝐼𝐵𝐵𝐷𝐷𝐼𝐼 𝐹𝐹𝐹𝐹𝐹𝐹𝐹𝐷𝐹 𝐷𝐼෡∇𝐼𝐷+𝐷𝐼𝐼𝑊1 𝑊𝐼𝐼+෡Δ1𝐵𝑊𝐵 𝑊𝐵𝐵 𝐵𝐵𝐷𝐵 𝐷𝐼𝐷𝐼+𝐷𝐼𝐼1 $$ CPU idle CPU 𝑈𝑈𝑈𝑈𝐷𝐷𝐼𝐼 𝑈𝑈𝑈𝑈𝐷𝐷𝐼𝐼+1 c) Layer-wise Schedule with all compute on GPU d) LSP-Offload

around the same scale as the GPU compute. (ii) There is no overlap between the GPU-to-CPU communication and the CPU-to-GPU communication, which implies that the full duplex PCIe channel is at least $50 \%$ underutilized.

To mitigate the first issue, Zero proposed delayed parameter updates (Fig. 3b), which use stale parameter values to calculate current gradients, allowing the CPU to perform the previous step’s update at the same time the GPU performs the current step’s forward and backward passes. Although increasing throughput, this method can affect the accuracy of training. Also, in order not to incur additional memory for buffering communication, the CPU-to-GPU communication and GPU-to-CPU communication cannot be parallelized.

These limitations inspire our design of a layer-wise scheduling strategy that maximizes parallelism between computation and communication. Unlike prior works that focus on parameter pulling or collective communication (Wang, Pi, and Zhou 2019) in distributed training (Lee et al. 2017), our approach applies layer-wise overlapping to offloading, achieving optimal parallelization across CPU and GPU computations and their communications.

# 4 LSP-Offload’s Approach

In this section, we present LSP-Offload, a practical offloading framework for fine-tuning high-quality models efficiently under memory-constrained settings. We will introduce our training algorithm for mitigating the compute and communication overhead, and then illustrate our new schedule design for maximized parallelism in the offloading’s schedule.

# Efficient and High-quality Offloading via Learned Sparse Projectors

As discussed before, on commodity hardware, the large optimization space combined with limited communication bandwidth causes offloading with a standard training algorithm to result in significant communication and compute overheads. To mitigate this problem, our key insight is to assist the offloading algorithm by using PEFT to configure the size of the optimization subspace, but to do so using novel techniques that avoid the pitfalls of prior PEFT.

<html><body><table><tr><td></td><td>LoRA</td><td>GaLore</td><td>LSP-Offload</td></tr><tr><td>Weight Matrix Trainable Parameters</td><td>W+ABT A,B ∈Rmxr,nxr</td><td>W+AtBT Bt∈Rnxr</td><td>W+PtStQT St∈Rdxd</td></tr><tr><td>GPUMemory Rank(Optim. Space)</td><td>mn+β(m+n)r r</td><td>mn+(m+ βn)r 1rT</td><td>mn+(m+n)r 72dT</td></tr></table></body></html>

Table 2: Comparison between different fine-tuning approaches, where $n , d , r$ are tensor dimensions satisfying $n \gg d \gg r$ . $W \in R ^ { m \times n }$ is the frozen pre-trained weight matrix. $\beta \geq 1$ is the scale factor for storing the optimizer state ( $\beta = 3$ for Adam), $\tau$ is the number of updates on the subspace, and $\gamma _ { 1 } , \gamma _ { 2 } \in ( 0 , 1 ]$ are scaling factors that adjust the rank based on how the individual subspaces interact when added together. LSP-Offload both reduces GPU memory and increases the optimization space rank.

Fig. 1 illustrates our approach. Following previous work (Hu et al. 2021; Zhao et al. 2024), we focus on matrix multiplication operations. Similarly to LoRA and GaLore, we freeze the pre-trained weight matrix and optimize on a decomposed subspace. However, LoRA’s and GaLore’s extra GPU memory to store the projectors and the optimization states grows linearly with the rank of their optimization spaces, preventing them from optimizing in a sufficiently large subspace. E.g., as shown in (Zhao et al. 2024), fine-tuning a 1B model with a hidden size of 2048 on a rank-512 subspace in half precision requires 4.38GB for LoRA and 6.17GB for GaLore, $2 . 2 \mathrm { x }$ and 3.1x the GPU memory needed for just the model.

To overcome this limitation, we made the key innovation to design the projector as sparse matrices, decoupling the dependence between the GPU memory overhead and the rank of the optimization space. Specifically, we use $( d , r )$ - sparse projectors as the template projector (see the properties of this projector in the full version (Chen et al. 2024)).

Definition 1 $( d , r )$ -Sparse Projector). We define the projection bases $P \in \mathbb { R } ^ { m \times d } , Q \in \mathbb { R } ^ { n \times d } a s \left( d , r \right)$ -sparse projectors if both $P , Q$ have $r$ nonzero values per row.

As shown in Fig. 1, by using $( d , r )$ -sparse projectors to replace the dense projectors, we project the weights on a $d { \times } d$ dimensional subspace. Meanwhile, the sparsity allows us to

LoRA Subspace 2 Subspace 3 P2Δ𝑆𝑆2𝑄𝑄2𝑇𝑇 𝑃𝑃3Δ𝑆𝑆3𝑄𝑄3𝑇𝑇 GaLore Subspace1 P1Δ𝑆𝑆1𝑄𝑄1𝑇𝑇 LSP-Offload Full-Param FT

store only the $O ( ( m + n ) r )$ non-zero values of the projectors on the GPU. This brings LSP-Offload two benefits:

• LSP-Offload is capable of optimizing in a larger subspace while using less GPU memory than SOTA PEFT. For our 2GB model example setting, LSP-Offload requires only 15MB extra GPU memory when using $r = 4$ . • LSP-Offload’s optimization space scales linearly with the parameter size. LSP-Offload optimizes in subspaces of size $O ( d ^ { 2 } )$ , with $d$ set to $n / 2$ to hide communication overhead. This results in a scaling of $O ( n ^ { 2 } )$ as the model size grows, outperforming LoRA and Galore’s $O ( n \times r )$ scaling, especially when $n \gg r$ for large models.

In all, the optimization space for a matrix multiplication operation with pre-trained matrix $W _ { 0 } \in R ^ { m \times n }$ constrains as

$$
\Delta { W } = P _ { 1 } S _ { 1 } Q _ { 1 } ^ { T } + P _ { 2 } S _ { 2 } Q _ { 2 } ^ { T } + . . . + P _ { \tau } S _ { \tau } Q _ { \tau } ^ { T } ,
$$

where $P _ { t } \in R ^ { m \times d } , Q _ { t } \in R ^ { n \times d }$ are periodically updated $( d , r )$ -sparse projectors, and $S _ { t } \in R ^ { d \times \mathbf { \hat { d } } }$ is a dense trainable matrix. As illustrated in Fig. 4, LSP-Offload optimizes in a larger subspace than LoRA and GaLore for the same GPU memory overhead, underscoring LSP-Offload’s efficiency.

Training algorithm. The above design leads to the LSPOffload’s core training algorithm listed in Alg. 1. In every iteration, the gradient is projected onto a subspace (line 15) before transferred to the CPU. The weight delta is then computed on CPU by optimizing on the subspace (line 16) before transferred back to GPU and projected to the original space (line 17). This way, both communication and compute complexity for offloading is reduced from $O ( m \cdot n )$ to $O ( d ^ { 2 } )$ , which guarantees our algorithm’s efficiency. Moreover, we optionally update the subspace (lines 18-21) by checking its quality. (In the next subsection, the steps are further pipelined between layers to hide latencies.)

Learned sparse projectors. Further, we boost the performance of the sparse projectors with a data-driven approach. Specifically, we initialize the $( d , r )$ -sparse projectors by randomly sampling the $r$ nonzero positions for each row and randomly sampling the nonzero values from $\mathcal { N } ( 0 , 1 / \sqrt { r } )$ . Random sampling ensures an unbiased estimation gradient with good approximation properties, as supported by the $\scriptstyle { \mathrm { J L } }$ lemma (Kane and Nelson 2014). After that, we fit the projectors on the calibration dataset to minimize the following estimation bias on the gradient:

Definition 2 (Estimation Bias). For a $( d , r )$ -sparse projector $P$ , $Q$ and a matrix $\Sigma \in R ^ { m \times n }$ , the estimation bias is $\pmb { b } ^ { P , Q } ( \Sigma ) : = P P ^ { T } \Sigma Q Q ^ { T } - \Sigma$ .

1: HyperParam: $d , r \colon ( d , r )$ -sparse projectors. CheckF req, $\alpha$ : check frequency, threshold for updating projectors.   
2: Function MAYBEUPDATE $\nabla _ { W }$ : the gradient, $P _ { p r e v }$ , $Q _ { p r e v }$ : previous projectors, $M , V$ : optimizer state)   
3: if $\| \mathbf { b } ^ { P , Q } ( \mathsf { \bar { V } } _ { W } ) \| _ { F } / \| \nabla _ { W } \| _ { F } ^ { \bullet } \leq \alpha$ then   
4: Return Pprev, Qprev   
5: $P , Q \gets I n i t i a l i z e ( d , r )$   
6: Minimize $l o s s : = \| \boldsymbol { \mathsf { b } } ^ { P , Q } ( \boldsymbol { \nabla } _ { W } ) \| _ { F } + \beta \cdot ( \| P \| _ { F } ^ { 2 } + \| Q \| _ { F } )$ until $\| \mathbf { b } ^ { P , Q } ( \nabla _ { W } ) \| _ { F } / \| \nabla _ { W } \| _ { F } \leq \alpha$ or Timeout.   
7: {Project previous M and $\mathrm { v }$ tensors to new subspace}   
8: $\check { M } \in \dot { \mathbb R } ^ { d \times d } \gets P ^ { T } P _ { p r e v } M Q _ { p r e v } ^ { T } Q$   
9: $V \in \mathbb R ^ { d \times d }  ( P ^ { T } P _ { p r e v } ) ^ { 2 } V ( Q _ { p r e v } ^ { T } Q ) ^ { 2 }$   
10: Return $P , Q$   
11: Function MAIN( $\mathcal { M }$ : Model, $\mathcal { D }$ : Dataset, $\textbf { \textit { W } } \in \mathbb { R } ^ { m \times n }$ : Weights, $M , V \in \mathbb { R } ^ { d \times d }$ : 1 t,D 2nd order optim∈izer state, $\boldsymbol { P } \in \mathbf { \mathbb { R } } ^ { m \times d }$ , $\boldsymbol { Q } \in \mathbb { R } ^ { n \times d }$ the sparse projectors)   
12: for $t \gets 1$ to $T$ do   
13: Sample $x \sim \mathcal { D }$   
14: $\nabla _ { W } ~ \gets ~ f o r w a r d B a c k w a r d ( \mathcal { M } , x ) \{ \mathrm { F W D + B W D } $ on GPU}   
15: $g r a i \gets S e n d T o C P U ( P ^ { T } \nabla _ { W } Q )$ {Compress on GPU and gradient offload}   
16: $\Delta _ { W }  S e n d T o G P U ( U p d a t e ( g r a d ) ) \{ { \mathrm { U P D } }$ on CPU and delta upload}   
17: $W \gets \mathbf { \hat { \boldsymbol { W } } } - \eta _ { t } P \Delta _ { W } Q ^ { T } \mathbf  \boldsymbol  \{ \$ {Decompress, apply deltas on GPU}   
18: if $( t - 1 )$ mod $C h e c k F r e q = 0$ then   
19: $\nabla _ { \boldsymbol { W } } \gets$ gradient on sampled subset $\mathcal { D } ^ { \prime } \subset \mathcal { D }$ .   
20: $P _ { \cdot }$ , $Q $ MAYBEUPDATE $\mathopen : ( \nabla _ { W } , P , Q , M , V )$   
21: end if   
22: end for

Particularly, we optimize the following problem for better projectors:

$$
\operatorname* { m i n } _ { P , Q } \underbrace { \lVert  { \mathbf { b } } ^ { P , Q } ( \nabla _ { W } ) \rVert _ { F } } _ { \mathrm { e s t i m a t i o n ~ e r r o r ~ o f ~ g r a d i e n t } } + \beta \cdot \underbrace { ( \lVert P \rVert _ { F } + \lVert Q \rVert _ { F } ) } _ { \mathrm { r e g u l a r i z a t i o n } }
$$

Compared to GaLore, which uses SVD decomposition as the projection matrix, we empirically find that our datadriven approach has a lower generalization error when using the same amount of extra GPU memory (Fig. 6b).

Convergence analysis of Alg. 1. For dataset $\mathcal { D }$ , weight matrix $\bar { W } \in R ^ { m \times n }$ , we consider minimizing $f ( W ) ~ =$ $\Sigma _ { x \sim \mathcal { D } } f _ { x } ( W ) / | \mathcal { D } |$ using Alg. 1 with $C h e c k F r e q = 1$ . That is, $\boldsymbol { W _ { t + 1 } } = \dot { \boldsymbol { W _ { t } } } - \eta P _ { t } \dot { P } _ { t } ^ { T } \boldsymbol { \nabla } f _ { x _ { t } } ( \boldsymbol { W _ { t } } ) Q _ { t } Q _ { t } ^ { T } , t = 1 , 2 , \dots , T ,$ where $P _ { t } , Q _ { t }$ are $( d , r )$ -sparse projectors. We derive the convergence theorem based on $\mathrm { L }$ -smooth functions, which indicate convexity and smoothness and are widely used in prior work (Ajalloeian and Stich 2020; Garrigos and Gower 2023).

Assumption 1 (Effectiveness of the subspace). The relative error on the subspace is kept under α in Alg. 1.

Assumption 2 (Bounded bias). There exists $\gamma > 0$ , such that for any weight $W$ and $x \sim \mathcal { D }$ , $\lVert \pmb { b } ^ { P _ { t } , Q _ { t } } ( \nabla f _ { x } ( W ) ) \rVert <$ $\gamma , \lVert \dot { \nabla } f _ { x } ( \dot { W } ) \rVert < \gamma$ .

Assumption 3 (Sparse bias). There exists a constant $\begin{array} { c c c c c } { { 0 } } & { { < } } & { { c } } & { { < } } & { { { \frac { 1 } { \sqrt { 2 } \alpha } } } } \end{array}$ √12α, such that ∥bPt,Qt(∇f(W ))∥F < $c \| \pmb { b } ^ { P _ { t } , Q _ { t } } ( \nabla f ( W ) ) \| _ { 2 }$ holds for any weight matrix $W$ .

We show the following convergence rate of our algorithm— see the full version (Chen et al. 2024) for the proof. The key idea is that a small gradient estimation error on the full dataset, which drives convergence, can be inferred from a bounded gradient estimation error on sub-sampled datasets.

Theorem 1. For any $\beta > 0$ and $0 < \delta < 1$ , suppose that $f$ is an $L$ -smooth function, Assumptions $\boldsymbol { { \mathit { 1 } } }$ , 2, 3 hold and that we check every iteration in Alg. 1 with the subsampled data set ${ \mathcal { D } } ^ { \prime }$ of size $\begin{array} { r } { \mathcal { O } \big ( \frac { 8 \gamma ^ { 2 } } { 3 \beta ^ { 2 } } \log \frac { ( m + \bar { n _ { \cdot } } ) T } { \delta } \big ) } \end{array}$ , and stepsize $\begin{array} { r } { \eta = \frac { 1 } { L } } \end{array}$ . Denote $F : = \mathbb { E } [ f ( W _ { 0 } ) ] - f ^ { * }$ . Then with probability $1 - \delta$ , $\begin{array} { r } { T = \mathcal { O } ( \frac { 1 } { \epsilon } ) \cdot \frac { \overline { { L } } F } { ( 1 - 2 c ^ { 2 } \alpha ^ { 2 } ) } } \end{array}$ iterations are sufficient to obtain mint [T ] E∥∇f (Wt)∥2 = O(ϵ + 2c2β2(12+2α)2 ).

Remark 1. The relative error ( $\alpha$ in Assumption $^ { l }$ ) is critical both for the final accuracy and for the time to convergence.

Remark 2. The logarithmic sample efficiency in our optional update indicates low overhead for subsampling $\mathcal { D } ^ { \prime }$ .

# Layer-wise Schedule for Maximal Parallelism

At the system level, we propose a new scheduling approach that addresses both issues in Zero’s schedule, based on the observation that optimization update steps for different layers are independent. This allows us to overlap GPU computation, CPU-GPU communication in both directions, and parameter updates on the CPU across different layers. The key idea and its benefits are illustrated in Fig. 3d (see the full version (Chen et al. 2024) for pseudocode). We split the GPU-to-CPU, CPU update, and CPU-to-GPU communication into small blocks to unlock the parallelism between layers without the accuracy loss of Zero’s use of stale parameter values. We parallelize the CPU’s and GPU’s compute by executing the deeper layers’ update step on CPU while doing the backward pass of shallower layers on GPU. We also parallelize the doublesided communication by executing deeper layer’s upload step while doing the shallower layer’s offload step. Compared to Zero-Offload, LSP-Offload reduces the CPU’s involvement in the critical path from the entire parameter update step to the update for only one layer, a $3 2 \mathrm { x }$ improvement for llama7B. We show in the full version (Chen et al. 2024) how to avoid a deeper layer’s workload from blocking a shallower layer’s computation that executes earlier in the next iteration.

# 5 Evaluation

We first verify the convergence of LSP-Offload on the GLUE dataset and then evaluate the end-to-end training performance on instruction-tuning. Detailed configurations for the experiments are described in the full version (Chen et al. 2024).

Accuracy validation of LSP-Offload on GLUE. Tab. 3 summarizes the accuracy of LSP-Offload for fine-tuning the pre-trained RoBertA-base (Liu et al. 2019) (117M) model on the GLUE dataset (Wang et al. 2018), which is a set of language understanding tasks that is widely adopted to evaluate fine-tuning (Hu et al. 2021; Zhao et al. 2024). We measure the best accuracy within one hour of training for end-to-end comparison given potential overheads. As shown in Tab. 3, LSP-Offload outperforms full parameter tuning by $1 . 9 \%$ accuracy, despite using only 253MB GPU memory vs. 747MB. Furthermore, the full version (Chen et al. 2024) shows that LSP-Offload converges at the same rate as the full parameter tuning. Compared to Galore, LSP-Offload achieves $1 . 2 \%$ higher accuracy. We attribute this to LSPOffload’s larger parameter update space (for the same GPU memory), which is $1 0 \mathrm { x }$ for this experiment.

Table 3: Accuracy $( \% )$ Comparison after 1 hour fine-tuning the pre-trained RoBertA-base model on GLUE.   

<html><body><table><tr><td></td><td>MNLI</td><td>SST2</td><td></td><td>MRPC CoLA</td><td>QNLI QQP</td><td></td><td>SST2</td><td>STS-B</td><td>Avg</td></tr><tr><td>Full Parameter</td><td>81.11</td><td>93.4</td><td>86.6</td><td>55.0</td><td>90.4</td><td>80.8</td><td>93.3</td><td>88.4</td><td>83.63</td></tr><tr><td>GaLore (Rank=16)</td><td>83.0</td><td>92.0</td><td>88.0</td><td>56.7</td><td>88.1</td><td>85.2</td><td>92.0</td><td>90.0</td><td>84.38</td></tr><tr><td>LSP (d=512,r=16)</td><td>81.4</td><td>91.7</td><td>91.1</td><td>61.65</td><td>91.78</td><td>83.39</td><td>92.0</td><td>91.0</td><td>85.53</td></tr></table></body></html>

Table 4: Evaluation accuracy $( \% )$ on the Humaneval dataset instruction after fine-tuning Deepseek-Coder-1.3B (top) and Deepseek-Coder-6.7b (bottom) with bfloat16 on the laptop GPU (top) and workstation GPU (bottom). Memory is measured in GB of used GPU memory, time is measured in hours.   

<html><body><table><tr><td></td><td>Mem</td><td>Time</td><td>python</td><td> java</td><td>cpp</td><td>js</td><td>ts</td><td>php</td><td>Avg</td></tr><tr><td>Zero-Offload</td><td>3.3</td><td>120</td><td>57.93</td><td>37.97</td><td>39.75</td><td>52.80</td><td>47.17</td><td>40.99</td><td>45.5</td></tr><tr><td>LoRA (Rank=8)</td><td>3.6</td><td>120</td><td>43.29</td><td>41.7735.4041.61</td><td></td><td></td><td></td><td>43.4031.68</td><td>39.3</td></tr><tr><td>GaLore (Rank=256)</td><td>7.9</td><td>120</td><td>39.63</td><td>36.08</td><td>31.68</td><td>34.78</td><td>40.88</td><td>36.02</td><td>36.4</td></tr><tr><td>LSP (d=1280,r=4)</td><td>3.6</td><td>120</td><td>55.49</td><td>42.41</td><td>40.99</td><td>50.31</td><td></td><td>48.43 38.51</td><td>45.6</td></tr><tr><td>Zero-Offload</td><td>16.8</td><td>15</td><td>73.78</td><td>61.39</td><td>64.60</td><td>66.46</td><td>64.15</td><td>58.39</td><td>64.8</td></tr><tr><td>Zero-Offload</td><td>16.8</td><td>30</td><td>75.00</td><td>64.56</td><td>61.49</td><td>70.81</td><td>65.41</td><td>62.73</td><td>66.7</td></tr><tr><td>LSP(d=2048,r=8)</td><td>17.0</td><td>15</td><td>74.39</td><td>62.66</td><td>61.49</td><td>66.46</td><td>67.30</td><td>65.84</td><td>66.4</td></tr></table></body></html>

End-to-end evaluation. Next, we evaluate the end-to-end performance of LSP-Offload for instruction-tuning. We perform our evaluation using four settings: (1) fine-tuning the GPT2-774M model on the Alpaca dataset (Taori et al. 2023) on a laptop with Nvidia A1000 Laptop GPU (4GB) and Intel Core-i7 12800H CPU (32GB), (2) fine-tuning the Llama-3B model on Alpaca on a workstation with Nvidia RTX 4090 GPU (24 GB) and AMD Ryzen Threadripper 3970X CPU (252GB), and (3,4) fine-tuning the DeepseekCoder-1.3B model (Deepseek-Coder-6.7B model) on an open-source code instruction dataset generated using WizardCoder’ method (Luo et al. 2023) on the laptop GPU (workstation GPU). We choose the $r$ in LSP-Offload and the ranks in LoRA and GaLore such that they all use similar amounts of memory below the GPU memory capacity.

Comparison with Zero-Offload. Compared to ZeroOffload, LSP-Offload achieves faster convergence while achieving similar convergence accuracy. For the instructiontuning task, LSP-Offload uses around $6 2 . 5 \%$ (Fig. 5a) and $3 3 . 1 \%$ (Fig. 5b) less time when converging to similar accuracy. E.g., when training on the Laptop GPU, LSP-Offload achieves the evaluation perplexity of 1.82 after 2 hours of training, while reaching the same perplexity takes 4.5 hours with Zero-Offload. Moreover, as shown in Fig. 5c and Tab. 4, within the 120 hour training budget, LSP-Offload trains $1 . 9 7 \mathrm { x }$ more epochs than Zero-Offload, resulting in lower training losses. Similarly, shown in Fig. 5d, for the Deepseek-Coder

Ours (d=512, r=16) 2.5 Ours (d=256, r=16) 2.4 Ours (d=512, r=8) Ours (d=512, r=16) 2.4 Ours (d=512, r=32) Ours (d=1024, r=16) 2.02 Zero-Offload 2.12 LoRA (Rank=16) LoRA (Rank=16) Zero-Offload 1.8 1.9 1.8 1.6 0 2 4 6 8 10 12 14 0 1 2 3 4 5 6 Time (h) Time (h)

(a) Evaluation perplexity of finetuning GPT2-774M w/ the laptop GPU.

![](images/7ed1476dfcdd34f49f032513b5bcfa7e636e496a8259f0e07b21efae9ed07806.jpg)  
(c) Simulated training loss of fine- (d) Simulated training loss of finetuning Deepseek-Coder-1.3B w/ tuning Deepseek-Coder-6.7B w/ the laptop GPU. workstation GPU for one epoch.

Figure 5: End-to-end evaluation of LSP-Offload. Rolling average is applied. Shading depicts the standard deviation.

![](images/bd53446261f15e9c93722d81754f43e2821646706fa79e2a769ebc1c846e2b18.jpg)  
(b) Evaluation perplexity of finetuning Llama-3B w/ the workstation GPU.   
Figure 6: Analysis on Coding task.   
Figure 7: Ablation on training throughput.

5 1.3B w/ 6.7B w/ RTX Relative…estimation…error…( ) .15 Sparse…Proj. S Laptop GPU 4090 GPU 1.05 (571628, r) (1280, r) Galore 0 0.75 Zero Ours Zero Ours 21 23 25 27 2 GPU Compute Comm CPU Other r (a) Breakdown for training 1 iter- (b) Estimation bias w/ Deepseekation on the coding task. 1.3B Model on validation set.

6.7B model , LSP-Offload completes the fine-tuning for an epoch $2 \mathrm { x }$ faster than Zero-Offload while achieving close accuracy (0.820 vs. 0.824). When trained for 15 hours, LSPOffload outperforms Zero-Offload on average accuracy by $2 . 4 \%$ .

Comparison with PEFT approaches. LSP-Offload achieves $30 \%$ lower evaluation perplexity than LoRA in Alpaca (Fig. 5a), and outperforms GaLore in all coding tasks with $2 7 . 8 \%$ higher average accuracy on the Humaneval (Chen et al. 2021; Cassano et al. 2023) dataset (Tab. 4), even if GaLore trains $60 \%$ more epochs than LSP-Offload.

Training time breakdown. Fig. 6a shows the time breakdown of LSP-Offload for training a single iteration. Compared to Zero-Offload, LSP-Offload cuts $50 \%$ the periteration latency by reducing the wall-clock time of CPU compute and communication. Because of the layer-wise parallel schedule, the communication and compute on both CPU and GPU are fully in parallel, resulting in minimal nonoverlapped overhead for communication and CPU compute.

Hyperparameters. We measured the estimation bias across different configurations on the Deepseek Coding task. As shown in Fig. 6b, larger $d$ with $r$ at 4 or 8 minimizes estimation bias, leading to faster and higher quality convergence. Therefore, it is advisable to set larger $d \mathrm { s }$ as long as the communication is hidden from the compute with smaller rs.

Ablation study. Fig. 7 shows an ablation study on training throughput with different techniques. Training throughput is

5 Normalized Throughput Laptop 4GB GPU Workstation 24GB GPU Zero-wO/f fLloayaedrwise Sch. Ours (d=2048, $\mathsf { r } { = } 1 6 )$ 1 Ours $( \mathsf { d } = 1 0 2 4 , \mathsf { r } = 1 6 )$ Ours (d=512, $\mathsf { r } { = } 1 6$ ) Ours (d=256, r=16) FWD $^ +$ BWD $^ +$ UPD GPT2-774M GPT2-1.3B Llama-3B Llama-7B

measured by the number of training iterations executed in unit time. First, by adding layer-wise scheduling (blue columns), we improve Zero-Offload (leftmost column) throughput by $18 \%$ . After that, we apply LSP-Offload with different configurations. Compared to a native training setup (rightmost column) where only FWD, BWD, and UPD operations are performed on the GPU without CPU computation or communication, LSP-Offload incurs an average slowdown of just $1 0 . 6 \%$ , $1 6 . 7 \%$ for subspace sizes of 256, 512 respectively.

# 6 Limitation

LSP-Offload introduces a list of hyperparameters for the selection of the $( d , r )$ -sparse projector, the frequency of subspace updates, the threshold for these updates, and others. We refer the reader to the full version (Chen et al. 2024) for more empirical insights into their selection.

# 7 Conclusion

In this paper, inspired by PEFT methods, we developed LSPOffload to enable near-native speed fine-tuning by constraining parameter updates to a subspace. Using a sparse projector and minimizing empirical bias, LSP-Offload optimizes in larger spaces than GaLore and LoRA for the same GPU memory size. In the GLUE data set, LSP-Offload achieves convergence at the same rate as native training. Compared to zero-offload, it reduces the fine-tuning time by $3 3 . 1 \% - 6 2$ $5 \%$ in instruction-tuning tasks while maintaining accuracy. Furthermore, it improves accuracy by $2 7 . 8 \% - 3 0 \%$ in the Alpaca and Humaneval datasets over GaLore and LoRA.