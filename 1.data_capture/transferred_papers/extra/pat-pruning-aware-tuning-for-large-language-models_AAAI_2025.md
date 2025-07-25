# PAT: Pruning-Aware Tuning for Large Language Models

Yijiang Liu1, Huanrui Yang2\*, Youxin Chen3, Rongyu Zhang1, Miao Wang1, Yuan $\mathbf { D } \mathbf { u } ^ { 1 , 4 }$ , Li Du1,4\*

1School of Electronic Science and Engineering, Nanjing University 2University of Arizona 3Samsung Electronic Research Centre of China 4Interdisciplinary Research Center for Future Intelligent Chips, Nanjing University, Suzhou {liuyijiang, rongyuzhang, wangmiao}@smail.nju.edu.cn huanruiyang@arizona.edu, yx113.chen $@$ samsung.com, {yuandu, ldu}@nju.edu.cn

# Abstract

Large language models (LLMs) excel in language tasks, especially with supervised fine-tuning after pre-training. However, their substantial memory and computational requirements hinder practical applications. Structural pruning, which reduces less significant weight dimensions, is one solution. Yet, traditional post-hoc pruning often leads to significant performance loss, with limited recovery from further finetuning due to reduced capacity. Since the model fine-tuning refines the general and chaotic knowledge in pre-trained models, we aim to incorporate structural pruning with the finetuning, and propose the Pruning-Aware Tuning (PAT) paradigm to eliminate model redundancy while preserving the model performance to the maximum extend. Specifically, we insert the innovative Hybrid Sparsification Modules (HSMs) between the Attention and FFN components to accordingly sparsify the upstream and downstream linear modules. The HSM comprises a lightweight operator and a globally shared trainable mask. The lightweight operator maintains a training overhead comparable to that of LoRA, while the trainable mask unifies the channels to be sparsified, ensuring structural pruning. Additionally, we propose the Identity Loss which decouples the transformation and scaling properties of the HSMs to enhance training robustness. Extensive experiments demonstrate that PAT excels in both performance and efficiency. For example, our Llama2-7b model with a $2 5 \%$ pruning ratio achieves $1 . 3 3 \times$ speedup while outperforming the LoRA-finetuned model by up to $1 . 2 6 \%$ in accuracy with a similar training cost.

# Code — https://github.com/kriskrisliu/PAT

# Introduction

Large language models (LLMs) (Touvron et al. $2 0 2 3 \mathrm { a }$ ; Brown et al. 2020; Chowdhery et al. 2022) have transformed the field of NLP (Vaswani et al. 2017; Bahdanau, Cho, and Bengio 2014; Zhang, Zhao, and LeCun 2015; Yang et al. 2016) with their exceptional performance on various complex language benchmarks. Despite their success, these models often necessitate substantial computational resources and present challenges for practical deployment due to their billions of parameters. Their extensive scales result in high latency and complications in deployments (Pan et al. 2023; Zhang et al. 2024). To mitigate these issues, various techniques have been proposed, including model pruning (Ma, Fang, and Wang 2023; Ashkboos et al. 2024; Sun et al. 2023; Santacroce et al. 2023; Fang, Ma, and Wang 2023), knowledge distillation (Agarwal et al. 2024; Tunstall et al. 2023; Sun et al. 2019, 2020; Ma et al. 2020), and quantization (Liu et al. 2022; Yao et al. 2022; Bai et al. 2020; Zafrir et al. 2019) within the context of pre-trained language models (PLMs).

![](images/9b062c619e2a9f9a7fb959da2f1674e5fa19f25fa86442c6cc082602a2f1e3b3.jpg)  
Figure 1: Comparison of zero-shot accuracy averaged on downstream tasks. Various pruning methods at a $2 5 \%$ pruning ratio, as well as the unpruned LoRA, are employed. Our PAT (red) notably outperforms LLM-Pruner and SliceGPT, and is comparable to LoRA (blue), surpassing LoRA by $1 . 2 6 \%$ on the Llama2-7B model.

Network pruning (Syed, Guo, and Sundarapandiyan 2023; Xu et al. 2021a; Liu et al. 2021; Guo et al. 2019), which reduces model size by eliminating specific weights, has gained significant attention. Especially for structural pruning (Ashkboos et al. 2024; Li et al. 2016; Wang et al. 2019b) which promises practical acceleration on current hardware architectures. However, as shown in Fig. 1, traditional pruning methods (Ma, Fang, and Wang 2023; Ashkboos et al. 2024) usually results in significant performance loss, whether applied before or after recovery model finetuning with Pre/Post-Trainig Pruning (P2F/F2P).

On the other hand, since the pretraining-fine-tuning pipeline has become standard practice in both academic and industrial scenarios, Parameter-Efficient Fine-Tuning (PEFT) methods (Xu et al. 2023a; Lin, Madotto, and Fung 2020; Mahabadi et al. 2021; Liu et al. 2024b), e.g., LowRank Adapter (LoRA) (Hu et al. 2021), have emerged as prevailing solutions for streamlined training. Meanwhile, since model fine-tuning can be seen as refining the universal and chaotic knowledge in the pre-trained model, thereby transforming the general LLM into a task-specific expert, combining structural pruning and PEFT for model efficiency and quick adaptation becomes a natural thought.

Drawing inspiration from quantization methods that often work synergistically, including the training-free PostTraining Quantization (PTQ) (Dettmers et al. 2022; Frantar et al. 2022; Lin et al. 2023; Lee et al. 2023) and the performance-enhancing Quantization-Aware Training (QAT) (Liu et al. 2023; Kim et al. 2023; Dettmers et al. 2023), we aim to incorporate structure pruning into the fine-tuning process while further boosting the model performance. This prompts us to introduce a new Pruning-Aware Tuning (PAT) paradigm to facilitate efficient inference and practical deployment in real-world applications, such as autonomous vehicles which require fast and accurate model inference to make real-time decisions and avoid obstacles while a fine-tuned RAG model must quickly and precisely retrieve and generate relevant responses from a compact knowledge base for different customer support. Unlike traditional P2F/F2P methods that remove model weights based on fixed prior knowledge, our proposed PAT method enables simultaneous pruning and fine-tuning. This allows the model to adaptively learn which parameters are most redundant and should be pruned during the PAT process. As a result, we achieve an automatic, end-to-end structured pruning process that not only maximizes but can also enhance the capabilities of the fine-tuned model.

Specifically, we propose the integration of plug-in Hybrid Sparsification Modules (HSMs). These HSMs are strategically positioned between the Attention and FFN components. Initially, they are set as identity matrices to maintain stable gradients at the onset of the fine-tuning process. As fine-tuning progresses, the HSMs selectively attenuate the channel values of the hidden dimensions, resulting in the exclusion of the corresponding linear projection weights. However, directly integrating dense-structured HSMs introduces an excess of trainable parameters. To mitigate this issue, we leverage the Hybrid-Identity-Operator (HIO), which reduces the number of trainable parameters. Compared with other PEFT methods, our approach not only achieves parameter efficiency but also decreases the overall model complexity. Furthermore, we introduce the Identity Loss (IL) applied to the HSMs to enhance training robustness and efficacy. This technique regularizes the HSMs while delegating the scaling functionality to independent trainable parameters.

In addition, the pruning operation across all HSMs is governed by a single trainable Unified Sparsification Mask (USM), ensuring consistent retention of channel indices across modules. This approach standardizes and streamlines the transformer decoder structure. As the trainable mask gradually converges to the target sparsity, the knowledge encoded in weights from pruned channels are seamlessly updated and redistributed to the remaining active channels.

Extensive experiments on widely recognized Large Language Models (LLMs) demonstrate the effectiveness of our proposed Pruning-Aware Tuning (PAT) compared to state-of-the-art baselines, including Parameter-Efficient Fine-Tuning (PEFT) and Pre/Post-Training Pruning (PTP) methods. Notably, on the Llama2-7B model, PAT surpasses the performance of LoRA-64 by $1 . 2 6 \%$ while achieving $2 5 \%$ weight pruning. The contribution of this paper can be summarized as follows:

• We propose an innovative paradigm called Pruning-Aware Tuning (PAT). Unlike traditional pre- or post-training pruning methods, PAT achieves simultaneous structural pruning and fine-tuning, leading to improved model performance.   
• To decrease overall model complexity, we integrate plug-in Hybrid Sparsification Modules (HSMs) with the Hybrid-Identity-Operator. Additionally, we design an Identity Loss (IL) applied to the HSMs to further enhance fine-tuning efficiency and robustness.   
• We utilize a single Unified Sparsification Mask (USM) that governs all HSMs, ensuring consistent retention of channel indices across modules.

# Related Work

# Pruning

Network pruning (LeCun, Denker, and Solla 1989) has long been recognized as an effective method for model compression and acceleration. Earlier research primarily focused on small-scale networks (Fang et al. 2023; Yang et al. 2023; Chen et al. 2021; Wu et al. 2024). However, with the advent of large-scale models, pruning techniques have increasingly been applied to large language models (LLMs). According to the pruning granularity, pruning methods can be broadly categorized into unstructured and structured pruning. In the realm of unstructured pruning (Frantar and Alistarh 2023; Sun et al. 2023), techniques such as SparseGPT (Frantar and Alistarh 2023) and Wanda (Sun et al. 2023) have been proposed. SparseGPT addresses the layer-wise reconstruction problem by utilizing Hessian inverses, while Wanda employs the product of weight magnitudes and input feature norms as its pruning criterion. Despite their effectiveness, these unstructured sparsification methods do not guarantee on-device speedup without hardware-specific support. In contrast, the structured pruning (Zafrir et al. 2021; Kurtic et al. 2022; Xia, Zhong, and Chen 2022; Yang, Wen, and Li 2019; Yang et al. 2023) removes organized patterns within the network, enabling significant acceleration in a hardware-agnostic manner. For instance, ShortenedLLaMA (Kim et al. 2024) removes Transformer blocks, resulting in depth pruning. Sheared-LLaMA (Xia et al. 2023)

incorporates the learnable mask to prune both the network’s width and depth. LLM-Pruner (Ma, Fang, and Wang 2023) and SliceGPT (Ashkboos et al. 2024) prune the network width while retaining the number of layers: LLM-Pruner sparsifies the intermediate dimension while SliceGPT focuses on the hidden dimension. However, existing structured pruning models still suffer from accuracy loss, necessitating further exploration and improvement.

# Parameter-Efficient Fine-Tuning

Compared to full fine-tuning of LLMs, Parameter-Efficient Fine-Tuning (PEFT) can achieve comparable performance while significantly reducing the computation and memory cost. PEFT methods can be broadly classified into five categories: additive fine-tuning, partial fine-tuning, reparameterized fine-tuning, hybrid fine-tuning, and unified fine-tuning. Additive fine-tuning methods introduce new additional parameters into the model, including adapter-based (Hu et al. 2021; Zhang et al. 2023b; He et al. 2021; Ru¨ckl´e et al. 2020) and soft prompt-based (Li and Liang 2021; Wang et al. 2023; Vu et al. 2021) approaches. For example, LoRA (Hu et al. 2021), one of the most popular used PEFT method, freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. DoRA (Liu et al. 2024a), a successful variant of LoRA, achieves enhanced performance by decomposing the pre-trained weights into magnitude and direction for subsequent fine-tuning. Partial fine-tuning selects only the parameters that are important for the downstream task to be trained (Ben-Zaken, Ravfogel, and Goldberg 2021; Lawton et al. 2023; Xu et al. 2021b). Reparameterized fine-tuning methods (Edalati et al. 2022; Zhang et al. 2023a; Xu et al. 2023b) often use low-rank transformations to reduce the number of trainable parameters. Hybrid finetuning (Zhou et al. 2023; Hu et al. 2022) combines multiple PEFT methods together. Unified fine-tuning (He et al. 2022; Wang et al. 2022) integrates various fine-tuning methods into a unified structure, but only utilizes one of them during fine-tuning. In this study, we mainly employ LoRA and DoRA as the fine-tuning techniques to explore our proposed PAT paradigm.

# Methodology

In this section, we detail the components of our proposed Pruning-Aware Tuning (PAT). Firstly, we introduce the foundational concept of the zero-preservation property inherent in the RMSNorm operation. Subsequently, we elaborate on the Hybrid Sparsification Module (HSM) and the Unified Sparsification Mask (USM). Furthermore, we outline the comprehensive process of PAT and introduce the innovative Identity Loss (IL). Finally, we expound on the overall optimization objective.

# Preliminary: Zero-Preservation of RMSNorm

RMSNorm (Zhang and Sennrich 2019), an abbreviation for root mean square layer normalization, is widely used in LLMs, such as Llama (Touvron et al. 2023b), Gemma (Team

et al. 2024), and $\mathrm { Y i }$ (Young et al. 2024). The general form of the RMSNorm is defined as the following:

$$
\bar { x } _ { i } = \mathrm { R M S N o r m } ( x _ { i } ) = \frac { x _ { i } } { \mathrm { R M S } ( \mathbf { x } ) } g _ { i } ,
$$

where $\bar { x } _ { i }$ is the $i$ -th value of vector $\bar { \mathbf { x } } \in \mathbb { R } ^ { d }$ , and $\mathbf { g } \in \mathbb { R } ^ { d }$ is the gain parameter. $\mathrm { R M S } ( \cdot )$ is the Root Mean Square operation, defined as:

$$
\mathrm { R M S } ( \mathbf { x } ) = \sqrt { \frac { 1 } { d } \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } }
$$

Given the layer input $\mathbf { X } \in \mathbb { R } ^ { d \times n }$ with specific (e.g., 1st and 2nd) channels all equal to 0 :

$$
\mathbf { X } = \left( \begin{array} { c c c c } { 0 } & { 0 } & { \cdots } & { 0 } \\ { 0 } & { 0 } & { \cdots } & { 0 } \\ { x _ { 3 } ^ { ( 1 ) } } & { x _ { 3 } ^ { ( 2 ) } } & { \cdots } & { x _ { 3 } ^ { ( n ) } } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { x _ { d } ^ { ( 1 ) } } & { x _ { d } ^ { ( 2 ) } } & { \cdots } & { x _ { d } ^ { ( n ) } } \end{array} \right)
$$

where $\boldsymbol { x } _ { j } ^ { ( i ) }$ is the $j$ -th value of the $i$ -th vector in $\mathbf { X }$ . Referring to Eq. (1), the RMSNorm operation will preserve these zero values, thereby making it feasible to prune the corresponding channels.

# Hybrid Sparsification Module (HSM)

Our objective is to prune the hidden dimensions of LLMs during fine-tuning, which would involve selecting the channels to be pruned in a linear layer, and convert the knowledge of pruned weights into those remained. To achieve this, we design a specific module to be applied after a linear layer, namely Hybrid Sparsification Module (HSM). HSM consists of a trainable channel selection mask M and a knowledge transformation weight $\mathbf { D }$ . Specifically, the computation involving the HSM and the upstream linear layer with weight $\mathbf { W } \in \mathbf { \overline { { R } } } ^ { d _ { o } \times d _ { i } }$ is formulated as follows:

$$
\begin{array} { r l } & { \mathbf { Z } = \left( \mathbf { M } \odot \mathbf { D } \right) \cdot \mathbf { W } \mathbf { X } } \\ & { \quad = \left( \mathbf { M } \odot \mathbf { D } \mathbf { W } \right) \cdot \mathbf { X } } \\ & { \quad = \mathbf { W } _ { D } \cdot \mathbf { X } , } \end{array}
$$

where $d _ { i }$ and $d _ { o }$ are the input and output dimension, respectively, $\mathbf { X } \in \mathbb { R } ^ { d _ { i } \times n }$ is the input value, $\mathbf { Z } \in \mathbb { R } ^ { d _ { o } \times n }$ is the output value, $\mathbf { M } \in \mathbb { R } ^ { d _ { o } }$ denotes the trainable mask whose values converge to either 0 or 1, $\mathbf { D } \in \mathbb { R } ^ { d _ { o } \times d _ { o } }$ is the HSM weight, $\textbf { W } \in \ \mathbb { R } ^ { d _ { o } \times d _ { i } }$ is the upstream linear weight, and $\mathbf { W } _ { D } \in \mathbb { R } ^ { d _ { o } \times d _ { i } }$ is the merged weight that replaces W after training. Notably, the zero values in M effectively cause the corresponding output channels of $\mathbf { W } _ { D }$ to be pruned.

To prune all linear layers in LLMs such as Llama2, which encompass the Q, K, V, and O projections in Attentions, as well as Up, Gate, and Down projections in FFNs, a straightforward approach is to apply the HSM after all linear layers. However, considering the sheer number of the linear layers in an LLM, this approach would incur significant overhead. We propose a novel and efficient alternative: placing pruning modules only between the Attention and FFN components, as illustrated in Fig. 2. The “pruned1” HSM’s output, Z, will first undergo the addition with the residual connection, which has already been pruned by the previous HSM, and then be fed into the RMSNorm operator before the next Attention/FFN component. As demonstrated previously in the preliminary, the RMSNorm has no impact on zero-valued channels, and since the downstream linear projection receives input with certain channels set to zero, the input dimensions of the following block can be pruned accordingly. In cases where LLMs involve the LayerNorm which projects zero-valued channels to non-zero, we can convert it to the RMSNorm before incorporating HSMs. This transformation is mathematically equivalent, as described by SliceGPT (Ashkboos et al. 2024).

![](images/2045ef21f9906ab2869e2d1094f33bee78b76300959098b53b18d33a1855dca3.jpg)  
Figure 2: Framework of our Pruning-Aware Tuning (PAT), featuring Hybrid Sparsification Modules (HSMs) positioned between the Attention and Feed-Forward Network (FFN) components. Each HSM includes a Hybrid-Identity-Operator (HIO) and a globally shared trainable mask. At training stage, the mask values will be updated until convergence. At inference stage, the pruned HSMs and the upstream linear layers will be merged, and the downstream layers which receive inputs with zero-valued channels will be pruned accordingly.

Although inserting HSMs between Attention and FFN components reduces trainable parameters compared to directly applying them to each linear module, the overall training overhead remains significantly larger than that of PEFT methods. To mitigate this issue, we propose the HybridIdentity-Operator (HIO) as a replacement for the dense structure of HSMs, which is formulated as:

$$
\mathbf { D } = \mathbf { L } _ { 1 } \cdot \mathbf { L } _ { 0 } + \mathbf { I } ,
$$

where ${ \bf L } _ { 0 } \in \mathbb { R } ^ { r \times d _ { o } }$ , ${ \bf L } _ { 1 } \in \mathbb { R } ^ { d _ { o } \times r }$ , $r$ is the rank value of $\mathbf { L } _ { 1 } \mathbf { L } _ { 0 }$ , and $\mathbf { I } \in \mathbb { R } ^ { d _ { o } \times d _ { o } }$ is the identity matrix with diagonal values set to 1 and other values set to 0. During fine-tuning, I is frozen, allowing gradients to flow through ${ \bf L } _ { 0 }$ and $\mathbf { L } _ { 1 }$ . HIO significantly reduces the number of trainable parameters. For example, a dense HSM consists of $d _ { o } \times d _ { o }$ parameters, while the HIO consists of $2 \times d _ { o } \times r$ . By determining $r < d _ { o } / 2$ , we can decrease the number of trainable parameters. In practice, we set $r$ to approximately $5 \%$ of $d$ , which in turn only accounts for $10 \%$ parameter of dense HSMs.

# Unified Sparsification Mask (USM)

We utilize a single trainable mask $M$ as in Eq. (4) to adaptively set channel values of hidden states to zero. The mask acts uniformly across all HSMs, ensuring consistency in the pruned channel indices throughout the computation flow. This unified pruning mask is particularly necessary at the residual connections between Attention and FFN components, as it guarantees that the pruned channels are correctly aligned throughout the entire data flow.

To insure structural sparsity at the convergence of the model, we employ a continuous sparsification strategy with a tailored regularizer to ensure that the mask converges to discrete values of 0 or 1 and achieves the desired sparsity at the end of the training process. This involves applying a differentiable gating function, $\mathscr { G } ( \cdot )$ , to the trainable proxy weights ${ \bf W } _ { M }$ of the mask. The gating function utilizes a modified Sigmoid function with a variable temperature $\tau$ , which is defined as:

$$
\tau ( s ) = { \left\{ \begin{array} { l l } { \displaystyle { \frac { 1 } { 1 - { \frac { \ln ( s ) } { \ln ( s _ { 0 } ) } } } } } & { { \mathrm { i f } } s < s _ { 0 } , } \\ { \displaystyle { } } \\ { \displaystyle { \epsilon ^ { - 1 } } } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

$$
\beta ( s ) = { \left\{ \begin{array} { l l } { { \displaystyle { \frac { - s } { s _ { 0 } } } + 0 . 5 } } & { { \mathrm { i f } s < s _ { 0 } / 2 , } } \\ { { } } & { { } } \\ { { 0 } } & { { \mathrm { o t h e r w i s e } . } } \end{array} \right. }
$$

$$
\mathbf { M } = { \mathcal { G } } ( s , \mathbf { W } _ { M } ) = { \frac { 1 } { 1 + e ^ { - \tau ( s ) \cdot \mathbf { W } _ { M } } } } + \beta ( s ) ,
$$

![](images/74ca544538db4451ac82b4b05a8a5147b61529f70d8793e6468194b5fffbae06.jpg)  
Figure 3: The differentiable gating function $\mathcal { G } ( \cdot )$ .

where $s$ denotes the current training step which dynamically determines the temperature, $s _ { 0 }$ is the milestone step which indicates that the temperature stay unchanged in the remaining training steps. In practice, we set $s _ { 0 }$ to $1 / 3$ of the total training steps. $\beta ( \cdot )$ denotes the offset which varies according to the step. Fig. 3 demonstrates some typical training stages. Initially, when $s = 0$ , the gating function maps all proxy weights of the mask to 1. This is achieved by initializing ${ \bf W } _ { M }$ to zero, which keeps the model weights unchanged, ensuring stable gradients at the beginning. As the temperature increases, the slope near 0 rises, and the offset term decreases. By halfway to the milestone step, the offset term reaches 0 and stops updating, while the slope continues to increase. At the milestone step, the slope near 0 becomes very steep, while the slope elsewhere approaches 0. At this point, the mask values will be enforced to either 0 or 1, where 0 refers to the channel being pruned. Moreover, to achieve the target sparsity, specifically the proportion of values equal to 0, we propose regularizing the number of active channels. This is achieved through the following regularization term:

$$
\mathcal { L } _ { a c t i v e } = \| N _ { t a r g e t } - \sum _ { i } \mathbb { 1 } _ { ( m _ { i } > 0 ) } \| _ { 2 } ,
$$

where $N _ { t a r g e t }$ denotes the target channel number of active channels, $m _ { i }$ represents the $i$ -th value of the proxy weight $M$ , and $\mathbb { 1 } _ { ( c o n d i t i o n ) }$ is the indicator function that equals 1 if the condition is true, and 0 otherwise.

# Pruning-Aware Tuning

We perform model fine-tuning by updating the proposed HSM modules and applying LoRA on all linear layers $\mathrm { \Delta H u }$ et al. 2021). Besides the standard instruction fine-tuning loss $\mathcal { L } _ { I n s t r u c t }$ , we propose the innovative Identity Loss (IL) to decompose the scaling and rotation in the HSM transformations. Specifically, we alter the formulation of Eq. (5) into:

$$
\mathbf { D } = \mathbf { L } _ { 1 } \cdot \mathrm { d i a g } ( \mathbf { v } ) \cdot \mathbf { L } _ { 0 } + \mathbf { I } ,
$$

where $\mathbf { v } \in \mathbb { R } ^ { r }$ is the trainable scaling values, and $L _ { 0 }$ and $L _ { 1 }$ are constrained to be orthogonal with the identity regularization

$$
\mathcal { L } _ { I d e n t i t y } = \| \mathbf { L } _ { 0 } \cdot \mathbf { L } _ { 0 } ^ { T } - \mathbf { I } \| _ { 2 } + \| \mathbf { L } _ { 1 } ^ { T } \cdot \mathbf { L } _ { 1 } - \mathbf { I } \| _ { 2 }
$$

The overall optimization objective is defined by a composite loss function $\mathcal { L }$ , which is expressed as follows:

$$
\mathcal { L } = \mathcal { L } _ { I n s t r u c t } + \mathcal { L } _ { a c t i v e } + \mathcal { L } _ { I d e n t i t y } ,
$$

where $\mathcal { L } _ { I n s t r u c t }$ represents the loss associated with instruction fine-tuning.

# Experiments

In this section, we present the experimental results and analysis. We begin by describing the experimental setup. Next, we showcase our main results across various Language Models (LLMs). We then delve into the efficiency and accuracy trade-off, examining memory and latency considerations. Finally, we conduct ablation studies on the trainable mask and identity loss.

# Experimental Setup

Models. We utilize model frameworks and checkpoints from HuggingFace (Jain 2022; Wolf et al. 2019), which includes Llama-2 7B and 13B (Touvron et al. 2023b), Gemma 2B and 7B (Team et al. 2024), Yi-1.5-34B (Young et al. 2024).

Baselines. The pruning baselines include LLMPruner (Ma, Fang, and Wang 2023), and SliceGPT (Ashkboos et al. 2024). We also involve the common LoRA (Hu et al. 2021) approach with the rank set to 64. Unless otherwise stated, we adjust the number of trainable parameters in all fine-tuning approaches to match the number of the LoRA. Additionally, we conduct complementary tests by applying ${ } ^ { 6 6 } \mathrm { P } $ FT” (Pruning before Fine-Tuning) and $\mathrm { \cdot \mathrm { F T } \to P ^ { \prime \prime } }$ (Fine-Tuning before Pruning) strategies on LLM-Pruner and SliceGPT. The pruning ratios are set to $20 \%$ , $2 5 \%$ , and $30 \%$ , respectively.

Datasets. We employ the LaMini-instruction dataset (Wu et al. 2023) for fine-tuning. To reduce training costs, we randomly drop $50 \%$ of the samples, resulting in a final dataset of 1 million samples. Unless otherwise stated, all experimental results are based on this setting. We conduct zero-shot evaluation on 14 datasets, including ARCChallenge (Clark et al. 2018), ARC-Easy (Clark et al. 2018), BOOLQ (Wang et al. 2019a), COPA (Wang et al. 2019a), HellaSwag (Zellers et al. 2019), MMLU (Hendrycks et al. 2021), MultiRC (Wang et al. 2019a), OpenBookQA (Mihaylov et al. 2018), PIQA (Bisk et al. 2020), RTE (Wang et al. 2019a), SIQA (Sap et al. 2019), WIC (Wang et al. 2019a), WinoGrande (Sakaguchi et al. 2021), WSC (Wang et al. 2019a). The accuracy is calulated by First-CapitalWord2 (Contributors 2023) method.

Implementation Details. Experiments are conducted using A100 GPUs. The models are fine-tuned over 3 epochs using the Alpaca instruction template. The learning rate is set to $5 \times 1 0 ^ { - 5 }$ with a cosine schedule. The batch size is set to 128, and the sequence length is 256 tokens. The milestone step of our PAT, $s _ { 0 }$ , is set to $1 / 3$ of the total training steps. The settings of our HIOs are derived to match the number of trainable parameters with LoRA-64. For example, we set the rank values of HIO and LoRA modules to 200 and 20 in the Llama2-7B experiments, respectively.

# Experimental Results and Analysis

Performance Comparison. Tab. 1 shows the zero-shot evaluations of different pruning methods across 14 wellknown tasks, where various types and sizes of LLMs are tested. We obtain that: (1) Our method, employing the Pruning-Aware Tuning (PAT) strategy, achieves the highest accuracy across pruned models. In contrast, LLM-Pruner and SliceGPT, which use either the Pruning before FineTuning $( \mathrm { P \to F T } )$ ) or Fine-Tuning before Pruning $\mathrm { ( F T  P ) }$ , suffer from non-negligible accuracy degradation. However, the“P $$ FT” significantly outperforms the $\mathrm { ^ { * } F T  P ^ { * } }$ . (2) The feasibility of pruning varies across different models. We observe that Llama2 with PAT maintains comparable performance to the un-pruned LoRA approach even at a $30 \%$ pruning rate, whereas Gemma 7b shows the trending of accuracy degradation at a $20 \%$ pruning rate. (3) Surprisingly, Llama2 7B and 13B with PAT under less than $30 \%$ and $20 \%$ pruning ratio, respectively, exhibit accuracy better than the unpruned LoRA.

Efficiency and Accuracy Trade-off. The implementation of HIO significantly reduces the number of trainable parameters, but this reduction may directly impact the model accuracy. We conducted experiments using various scales of training parameters on the Llama 2 7B model, and illustrate the results in Fig. 4. The total number of trainable parameters is adjusted by the rank values of HIO and LoRA modules. For example, the “LoRA- $. 6 4 ^ { , 9 }$ in dark represents the traditional LoRA fine-tuning with a rank value set to 64, and the “HIO-200, LoRA- $2 0 ^ { \prime \prime }$ in purple represents our PAT with a rank of 200 in HIO and a rank of 20 in LoRA modules. We find that our PAT demonstrates a performance trend correlated to the number of trainable parameters. “Dense3, LoRA- $\cdot 8 ^ { \circ }$ with $1 4 . 1 5 \%$ trainable parameters achieves $6 4 . 1 9 \%$ accuracy, outperforming “LoRA64” by $5 . 4 3 \%$ . Conversely, “HIO-8, LoRA- $\mathbf { \nabla } \cdot { } 8 ^ { , 9 }$ with merely $0 . 3 6 \%$ trainable parameters results in a $6 \%$ accuracy reduction. In practice, we opt for “HIO-200, LoRA- ${ \cdot 2 0 ^ { \circ } }$ in Llama 2 7B experiments, aligning the parameter count with that of “LoRA-64”. For others, Gemma 2B with “HIO-300, LoRA20”, Gemma 7B with “HIO-300, LoRA20”, Llama2 13B with “HIO-200, LoRA20” , and Yi-1.5 34B with “HIO200, LoRA20”.

Memory and Latency. We conducted an evaluation of the VRAM usage and the inference latency comparing the base Llama2 7B and 13B models with pruned versions, as illustrated in Fig. 5 and Fig. 6. The GPU memory is tested by loading the model without any proceeding tokens. The latency is tested by the time of the first token prediction in a batch with an initial context length of 128. Specifically, we assessed the models pruned at $20 \%$ , $2 5 \%$ , and $30 \%$ ratios across various batch sizes. Our $30 \%$ pruned models achieve $1 . 3 3 \times$ speedup on average. Moreover, the base Llama2 13B model encounters Out-Of-Memory (OOM) errors at a batch size of larger than 288 when executed on a single A100- 80GB GPU. In contrast, our pruned models work reliably under these conditions.

Llama2 7B Training Efficiency and Accuracy 70 (HIO-2048, (HIO-1024, 65 (HIO-512, LoRA-32) LoRA-64) LoRA-16) 5650 (LoRA(-L8)oR (HIO-200, LoRA-20) A-64) (LoRA-128) (LoRA-256) (LoRA-512) 50 (HIO-8, LoRA-8) PAT LoRA 45 0 2 4 6 8 10 12 14 16 % Trainable Parameters

Llama2 7B Llama2 13B 14 64 26 75 13 62 24   
112 60 22 70   
190 58 1280 605 56 16 14 7 52 12 55 6 50 10 0% 20% 25% 30% 0% 20% 25% 30% Pruning Ratio Pruning Ratio (a) Llama 2 7B (b) Llama 2 13B

![](images/18a93ab0d509b516e18de120fa318f7b840bde24c8fc7bd1a2f94bf662de7a4a.jpg)  
Figure 4: The training efficiency and the accuracy comparison for Llama2 7B. Our PAT results are represented as “HIO-M, LoRA-N”, where M and N denote the rank value in the HIO and the LoRA, respectively. The LoRA results are “LoRA-N”.   
Figure 5: The VRAM usage and the evaluation accuracy of Llama2 models under various pruning ratios.   
Figure 6: The speedup of Llama2 models according to different pruning ratios and batch sizes.

Table 1: Zero-shot evaluations of different pruning methods with $20 \%$ , $2 5 \%$ , and $30 \%$ pruning ratios across various LLMs. “FT” represents Fine-Tuning. $\bf { \hat { \Pi } } ^ { 6 6 } \bf { P } \mathrm { \to } \bf { F T } ^ { 9 }$ denotes Pruning the base model and then Fine-Tuning the pruned model via LoRA. “FT $ \mathbf { P } ^ { \prime }$ denotes Fine-Tuning the base model via LoRA and then Pruning the fine-tuned model. “PAT” denotes our proposed Pruning-Aware Tuning strategy. The accuracy is averaged across 14 datasets. More details are available in the Appendix.   

<html><body><table><tr><td>Ratio</td><td>Method</td><td>Mode</td><td>Gemma-2B</td><td>Gemma-7B</td><td>Llama2-7B</td><td>Llama2-13B</td><td>Yi-1.5-34B</td></tr><tr><td>0%</td><td>LoRA-64</td><td>FT</td><td>53.82</td><td>71.59</td><td>58.76</td><td>66.74</td><td>81.21</td></tr><tr><td rowspan="5">20%</td><td>LLM-Pruner</td><td>P-→FT</td><td>48.87</td><td>65.45</td><td>58.53</td><td>65.28</td><td>73.86</td></tr><tr><td>LLM-Pruner</td><td>FT→P</td><td>40.64</td><td>54.87</td><td>40.68</td><td>41.43</td><td>53.88</td></tr><tr><td>SliceGPT</td><td>P→FT</td><td>48.21</td><td>66.60</td><td>57.81</td><td>65.86</td><td>76.81</td></tr><tr><td>SliceGPT</td><td>FT→P</td><td>41.67</td><td>56.17</td><td>47.77</td><td>50.67</td><td>67.60</td></tr><tr><td>Ours</td><td>PAT</td><td>53.95</td><td>68.68</td><td>61.04</td><td>69.37</td><td>81.02</td></tr><tr><td rowspan="5">25%</td><td>LLM-Pruner</td><td>P→FT</td><td>42.32</td><td>60.50</td><td>52.50</td><td>58.64</td><td>70.10</td></tr><tr><td>LLM-Pruner</td><td>FT→P</td><td>40.20</td><td>50.29</td><td>39.72</td><td>39.82</td><td>51.03</td></tr><tr><td>SliceGPT</td><td>P-→FT</td><td>45.23</td><td>62.22</td><td>52.98</td><td>60.69</td><td>73.88</td></tr><tr><td>SliceGPT</td><td>FT→P</td><td>39.72</td><td>52.13</td><td>41.97</td><td>46.75</td><td>60.63</td></tr><tr><td>Ours</td><td>PAT</td><td> 52.98</td><td>66.68</td><td>60.02</td><td>66.58</td><td>78.90</td></tr><tr><td rowspan="5">30%</td><td>LLM-Pruner</td><td>P→FT</td><td>39.71</td><td>50.28</td><td>50.60</td><td>51.28</td><td>66.85</td></tr><tr><td>LLM-Pruner</td><td>FT→P</td><td>40.05</td><td>41.35</td><td>39.73</td><td>39.70</td><td>45.36</td></tr><tr><td>SliceGPT</td><td>P-→FT</td><td>40.07</td><td>53.14</td><td>50.91</td><td>56.12</td><td>71.81</td></tr><tr><td>SliceGPT</td><td>FT→P</td><td>39.89</td><td>44.30</td><td>40.14</td><td>46.19</td><td>56.10</td></tr><tr><td>Ours</td><td>PAT</td><td>45.33</td><td>64.58</td><td>57.81</td><td>65.15</td><td>77.89</td></tr></table></body></html>

Trainable and Frozen Mask. The frozen mask is implemented by linearly attenuating a fixed portion of the mask values during training. In our experiment, this attenuation is applied to the first $N$ values of the hidden dimension in LLMs, where $N$ is determined by the pruning ratio. The results presented in Tab. 2 demonstrate the significant advantage of the trainable mask over the frozen counterpart. For instance, in the case of the Llama2 13B model with $30 \%$ pruning, the trainable mask yields an accuracy improvement of $4 . 0 6 \%$ over the frozen mask.

Table 2: Ablation study on trainable mask and identity loss.   

<html><body><table><tr><td>Model</td><td>Ratio</td><td>Method</td><td>Traiable</td><td>Identity</td><td>Accuracy</td></tr><tr><td rowspan="5">Llama2 7B</td><td>0%</td><td>LoRA</td><td>N/A</td><td>N/A</td><td>58.76</td></tr><tr><td rowspan="4">25%</td><td rowspan="4">PAT</td><td>×</td><td>√</td><td>54.97</td></tr><tr><td></td><td>X</td><td>58.62</td></tr><tr><td>√</td><td>√</td><td>60.02</td></tr><tr><td>×</td><td>√</td><td>52.72</td></tr><tr><td rowspan="5">Llama2</td><td>30%</td><td>PAT</td><td>√ √</td><td>× √</td><td>56.59 57.81</td></tr><tr><td>0%</td><td>LoRA</td><td>N/A</td><td>N/A</td><td>66.74</td></tr><tr><td rowspan="3">25%</td><td rowspan="3">PAT</td><td>×</td><td>√</td><td>62.35</td></tr><tr><td></td><td>×</td><td>65.81</td></tr><tr><td>/√</td><td>√</td><td>66.58</td></tr><tr><td rowspan="3"></td><td rowspan="3">30%</td><td rowspan="3">PAT</td><td></td><td></td><td>61.09</td></tr><tr><td>X</td><td>√ ×</td><td>64.85</td></tr><tr><td>√</td><td>√</td><td>65.15</td></tr></table></body></html>

Ablation on Identity Loss. The incorporation of Identity Loss contributes to an enhanced accuracy improvement. As depicted in Tab. 2, Llama2 7B achieves $1 . 4 \%$ enhancement with the pruning ratio of $2 5 \%$ .

Downstream Task Capability. Following the downstream task adaptation detailed in DoRA (Liu et al. 2024a), we leverage PAT to fine-tune on specific tasks, including ARC, SuperGlue, OpenBookQA, PIQA, SIQA, MMLU, and WinoGrande. The setting of HSMs is “HIO-200, LoRA/DoRA- $2 0 ^ { \circ }$ . Our $2 5 \%$ pruned PAT-L and PAT-D achieve performance levels on par with those achieved by traditional DoRA and LoRA, shown in Tab. 3.

Table 3: Downstream task performance of LoRA, DoRA, and PAT. PAT-L and PAT-D denote our PAT with LoRA and DoRA fine-tuning, respectively.   

<html><body><table><tr><td>Method</td><td>Ratio</td><td>Llama2 7B</td><td>Llama213B</td></tr><tr><td>LoRA-64</td><td>0%</td><td>72.85</td><td>76.24</td></tr><tr><td>PAT-L</td><td>25%</td><td>72.05</td><td>76.08</td></tr><tr><td>DoRA-64</td><td>0%</td><td>73.50</td><td>77.36</td></tr><tr><td>PAT-D</td><td>25%</td><td>72.98</td><td>77.02</td></tr></table></body></html>

# Conclusion

We propose Pruning-Aware Tuning (PAT), a novel structured pruning approach for Large Language Models (LLMs). PAT prunes the hidden dimensions during the finetuning, while preserving the linguistic capabilities. We develop a trainable mask to adaptively set channel values to zero, and efficient Hybrid Sparsification Modules to enable pruning of all linear layers accordingly. The efficiency design reduces the training overhead of PAT to levels comparable to traditional LoRA fine-tuning. Additionally, we propose the Identity Loss to enhance the training robustness by decoupling the rotation and scaling properties of the HSMs. In the zero-shot evaluation, our $30 \%$ -PAT Llama2 7B and 13B models maintains $98 \%$ performance of those achieved from the LoRA fine-tuning.