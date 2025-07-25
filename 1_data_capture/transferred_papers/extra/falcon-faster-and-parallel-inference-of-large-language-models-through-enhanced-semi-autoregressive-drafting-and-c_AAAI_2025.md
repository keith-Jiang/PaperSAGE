# Falcon: Faster and Parallel Inference of Large Language Models Through Enhanced Semi-Autoregressive Drafting and Custom-Designed Decoding Tree

Xiangxiang Gao\*, Weisheng Xie \*†, Yiwei Xiang, Feng Ji

Bestpay AI Lab, Shanghai, 200080 China gaoxiangxiang, xieweisheng, xiangyiwei, jifeng @bestpay.com

# Abstract

Striking an optimal balance between minimal drafting latency and high speculation accuracy to enhance the inference speed of Large Language Models remains a significant challenge in speculative decoding. In this paper, we introduce Falcon, an innovative semi-autoregressive speculative decoding framework fashioned to augment both the drafter’s parallelism and output quality. Falcon incorporates the Coupled Sequential Glancing Distillation technique, which fortifies inter-token dependencies within the same block, leading to increased speculation accuracy. We offer a comprehensive theoretical analysis to illuminate the underlying mechanisms. Additionally, we introduce a Custom-Designed Decoding Tree, which permits the drafter to generate multiple tokens in a single forward pass and accommodates multiple forward passes as needed, thereby boosting the number of drafted tokens and significantly improving the overall acceptance rate. Comprehensive evaluations on benchmark datasets such as MT-Bench, HumanEval, and GSM8K demonstrate Falcon’s superior acceleration capabilities. The framework achieves a lossless speedup ratio ranging from 2.91x to $3 . 5 1 \mathrm { x }$ when tested on the Vicuna and LLaMA2-Chat model series. These results outstrip existing speculative decoding methods for LLMs, including Eagle, Medusa, Lookahead, SPS, and PLD, while maintaining a compact drafter architecture equivalent to merely two Transformer layers.

# 1 Introduction

Large Language Models (LLMs) have demonstrated exceptional performance across various benchmarks, reinforcing their practical significance. These models are primarily built on the Transformer architecture and utilize autoregressive (AR) decoding, effectively capturing complex dependencies and generating coherent sequences (Wan et al. 2024). However, due to their considerable model sizes, LLMs also face significant computational overhead and latency bottlenecks during inference. For example, the inference speed of the GLM 10B model is a mere 101 tokens per second when operated on a single Nvidia A100 GPU (Du et al. 2022). This presents considerable challenges for the widespread deployment and application of LLMs (Zhu et al. 2024; Xiao et al.

2023; Ghazvininejad et al. 2019; Xu et al. 2021; Ding et al.   
2021; Guo et al. 2021; Wang et al. 2019; Guo et al. 2019a).

Intriguingly, the inference speed of LLMs is considerably hampered by memory bandwidth, with the main latency bottleneck stemming from reading and writing model parameters to memory, rather than from arithmetic computations (Xia et al. 2024). To address this issue, researchers have proposed speculative decoding to expedite the inference speed of LLMs without sacrificing accuracy (Zhang et al. 2024; Guo et al. 2020). This strategy employs a drafter that efficiently generates $\mathbf { k }$ tokens as speculation of future decoding steps within the LLMs. Subsequently, the LLM verifies these tokens, with only successfully validated tokens being accepted as decoded tokens, thus maintaining the quality of the generated output (Xia et al. 2023). By directing computational resources towards validating pre-generated tokens, speculative decoding significantly diminishes the memory operations needed to access LLM parameters, consequently boosting overall inference efficiency (Xia et al. 2024).

While speculative decoding presents a potential solution to decrease the inference latency of LLMs, it also raises several important questions that need further exploration. Notably, existing speculative decoding methods primarily employ two drafting strategies: AR and semi-autoregressive (SAR) drafting. AR drafting generates tokens sequentially, each dependent on the preceding ones. This sequential dependency constrains the parallelism of the draft models, leading to substantial time overheads (Guo et al. 2019b; Sun et al. 2019; Geng, Feng, and Qin 2021; Graves et al. 2006). In contrast, SAR drafting generates multiple tokens simultaneously, enhancing the parallelization of the drafting process. However, a significant limitation of SAR drafting is its inability to fully capture the inter-dependencies between tokens within the same block, potentially resulting in a lower acceptance rate for the generated tokens (Bao et al. 2022; Ran et al. 2020). Consequently, balancing low drafting latency with high speculation accuracy poses a significant challenge in speculative decoding aimed at accelerating the inference speed of LLMs (Xia et al. 2024).

In this paper, we introduce Falcon, an advanced SAR speculative decoding framework engineered to boost both the drafter’s parallelism and the output quality, thereby amplifying the inference efficiency of LLMs. Falcon integrates the Coupled Sequential Glancing Distillation method, which elevates the token acceptance rate of SAR drafting. We offer an in-depth theoretical explanation for this accuracy enhancement, ascribing it to the strengthened dependencies between tokens within the same block. Additionally, we have developed a specialized decoding tree to support SAR drafting, enabling the drafter to generate multiple tokens in a single forward pass and also accommodate multiple forward passes. This design leads to a higher acceptance rate of tokens by the LLMs, further hastening inference speed.

Our comprehensive evaluations of Falcon on various benchmarks showcase its outstanding acceleration capabilities. It achieves a lossless speedup ratio ranging from $2 . 9 1 \mathrm { x }$ to $3 . 5 1 \mathrm { x }$ on the Vicuna and LLaMA2 model series. This performance outshines top-tier speculative decoding methods that use either AR drafting (such as Eagle, SPS, and PLD) or SAR drafting (like Medusa and Lookahead). Additionally, unlike other methods that necessitate billions of parameters for drafters, Falcon can attain faster inference speeds using parameters equivalent to just two Transformer blocks. Therefore, Falcon is highly beneficial for applications requiring real-time responses while working under computationally constrained environments. Our key contributions to the community are summarized as:

• We introduce Falcon, an advanced SAR speculative decoding framework that enhances both the parallelism and output quality of the drafter. Falcon achieves a better equilibrium between reduced inference latency and elevated speculation accuracy.   
• We develop Coupled Sequential Glancing Distillation, which markedly improves the accuracy of the SAR drafter. Additionally, a theoretical explanation is provided to clarify how this method enhances output quality.   
• We design a specialized decoding tree to support SAR drafting, enabling the drafter to generate multiple tokens in a single forward and allowing multiple forward passes, resulting in an improved token acceptance rate and inference speed of the LLMs.   
• Falcon outperforms existing speculative decoding methodologies for LLMs, attaining a speedup ratio ranging from $2 . 9 1 \mathrm { x }$ to $3 . 5 1 \mathrm { x }$ . Remarkably, it accomplishes this while preserving a compact drafter size, which is comparable to merely two Transformer layers.

# 2 Related Work

# 2.1 Autoregressive Drafting

Speculative decoding with AR drafting is a straightforward approach that employs a small language model as the drafter to generate tokens sequentially, each conditioned on its predecessors. The generated tokens are then validated by the LLMs to ensure alignment (Leviathan, Kalman, and Matias 2023; Spector and Re 2023; Chen et al. 2024; Zhao et al. 2024). SPS (Chen et al. 2023) is the pioneering work in this field, which generates draft tokens by invoking a 4B parameter AR drafter and validating these drafts using 70B parameter LLMs. PLD (Saxena 2023) replaces the draft model with simple string matching in the prompt to generate candidate token sequences. BiLD (Kim et al. 2023) utilizes the T5- small model to generate tokens, which are then validated by a T5-XXL model. SpecInfer (Miao et al. 2024) accelerates LLM serving through a tree-based inference and verification mechanism, organizing the drafts into a token tree. The correctness of all candidate token sequences is verified against the LLM using a tree-based decoding mechanism.

While AR drafting considerably boosts inference speed, it also imposes extra GPU memory costs, particularly for draft models with billions of parameters. Moreover, it adds time overhead from the drafter that could potentially counterbalance the advantages of the improved inference speed. EAGLE (Li et al. 2024), the current state-of-the-art speculative decoding method with AR drafting, tackles the issue of high GPU memory usage by incorporating a token sequence advanced by one time step. However, due to its inherently sequential nature, EAGLE is still constrained by time overhead, which hampers its potential to further speed up the inference of LLMs.

# 2.2 Semi-autoregressive Drafting

SAR speculative decoding simultaneously generates multiple tokens, maintaining the AR feature globally while easing it locally (Wang, Zhang, and Chen 2018; Łukasz Kaiser et al. 2018; van den Oord et al. 2018, 2016). Santilli et al. (Santilli et al. 2023) propose that AR decoding can be restructured as parallel resolution of a non-linear equation through Jacobi and Gauss-Seidel fixed-point iterations. This technique directly appends multiple [PAD] tokens to the end of the input prompt, facilitating parallel generation and speeding up existing models. Lookahead (Zhao et al. 2024) utilizes this generation method to enable the LLMs to produce several separate n-grams concurrently within a single forward pass, thereby reducing the latency of LLM. PASS (Monea, Joulin, and Grave 2023) introduces multiple learnable tokens and fine-tunes these token embeddings to enhance parallel decoding performance. However, these methods deviate from the AR pre-training patterns, which can result in less optimal draft quality.

Medusa (Cai et al. 2024) represents the most significant advancements in SAR drafting, building upon the research of Stern et al. (Stern, Shazeer, and Uszkoreit 2018). It optimizes the process by freezing the backbone model and incorporating additional lightweight heads into it, enabling the concurrent generation of multiple tokens. Medusa effectively alleviates the computational cost typically associated with AR drafting, thus achieving remarkable speedups. However, its inference speed is constrained by the low accuracy of the drafter, which currently stands at 0.6. This drop in accuracy results from employing the parallel processing mechanism, whose predictions are solely based on input tokens without accounting for inter-token dependencies (Xia et al. 2023; Wertheimer et al. 2024). Therefore, the key to improving the output quality of SAR drafters lies in strengthening the inter-token dependencies within the same block. Such enhancements would enable an optimal balance between low drafting latency and high speculative accuracy.

Target LLMs assist you Enhanced Semi-Autoregressive Drafting get to can 1 heap with to do the post help to the baby 个 个 个 个 Sampling Sampling multiple times 个 个 个 LM head LM head how 小 can 个 个 fiassist fi make fi help get fyou to 个 本 f to the 木 个 fahelp 个 f with the 个 Transformer Layers S Rmi-xud Causal Me sHed 个 个 本 f how can fiassist figet fimake fi help e how ecan ecani e assist you eget to emakea e help with 个 个 个 个 个 个 Embedding Embedding 个 个 how can cani assist you get to make a help with 小 Forward 1 Forward 1 Forward 2

# 3 Falcon

The Falcon framework is an advanced SAR speculative decoding architecture that concurrently improves the drafter’s parallelism and token acceptance rate. A comprehensive overview and computational process of this framework are presented in Section 3.1. To improve the drafter’s accuracy, we introduce the Coupled Sequential Glancing Distillation (CSGD) method, which strengthens the inter-dependencies among tokens within each block. An in-depth description of the CSGD method is provided in Section 3.2, followed by a theoretical discussion in Section 3.3. In addition, a CustomDesigned Decoding Tree has been developed to support SAR drafting, which is described in Section 3.4.

# 3.1 Framework and Computational Process

The architecture and computational process of the Falcon are depicted in Figure 1. It comprises three main components: the Embedding Layer, the Language Model (LM) Head, and the Relaxed Causal SAR Decoding Head. The Embedding Layer transforms a sequence of tokens into a corresponding sequence of token embeddings. The LM Head computes the probability distribution based on the aggregated features, from which the next token is sampled. Both the Embedding Layer and the LM Head leverage the parameters of LLMs. Relaxed Causal SAR Head will be introduced in section 3.2.

EAGLE’s findings (Li et al. 2024) indicate that concatenating feature and token sequences from one time step ahead encapsulate richer semantic context. This enhancement allows the model to make more informed predictions by leveraging a broader scope of information. Consequently, we concatenate consecutive feature sequences and token sequences from one time step ahead to predict the next $k$ tokens concurrently. For instance, when $k = 2$ , Falcon predicts the feature sequence $( f _ { 3 } , f _ { 4 } )$ using the initial feature sequence $( f _ { 1 } , f _ { 2 } )$ and the token sequence $( t _ { 2 } , t _ { 3 } )$ advanced by one time step. Subsequently, the predicted features $( f _ { 3 } , f _ { 4 } )$ , along with the next token sequence $( t _ { 4 } , t _ { 5 } )$ are concatenated to form the new input sequence. This is used to predict subsequent feature sequences $( f _ { 5 } , f _ { 6 } )$ and token sequences $( t _ { 6 } , t _ { 7 } )$ , facilitating the continuation of the drafting process.

# 3.2 Coupled Sequential Glancing Distillation

We have designed an SAR decoding method based on CSGD, aimed at enhancing the accuracy of the drafter. The training procedure is illustrated in Figure 2. Here, the feature and token sequences from one time step ahead are concatenated and fed into the drafter, resulting in a fused sequence of dimensions $( b s , s e q \_ l e n , 2 * h i d d e n \_ d i m )$ . The drafter is composed of a hybrid Transformer network, which includes two layers of LSTM (Hochreiter and Schmidhuber 1997), Relaxed Causal-Masked Multi-Head Self-Attention, and MLP networks. The LSTM network reduces the dimension of the fused sequence to (bs, seq len, hidden dim) and retains information about past tokens, thereby improving the model’s accuracy. The Relaxed Causal-Masked Multi-Head Self-Attention mechanism enables the model to focus on relevant parts of the input sequence while preserving causality. The MLP layers further process this information to make the final predictions.

After the sequence passes through the drafter for the first time, an initial prediction of tokens, denoted as $\hat { Y }$ is generated. We compare the Hamming Distance (Roman 1992) between the prediction from the drafter $\hat { Y }$ and the predic

Enhanced Semi-Autoregressive Head   
SFquturee quture Coupled Enhanced Compute A LA PM → Sequence Semi-Auto Distillation Token Glancing Token regressive Loss   
Sequence Sequence ahead ahead' √ ￥ 业 Y Yt choice 1 choice2 choice3   
h t2 y Compute yt : Sample s(Y,Y) t ht ht Distance words   
h t V2 y h t h ！ ? + 4 h y3 y features cand h t4 h t4 ！ ： ${ h } _ { 3 } ^ { t }$ $t _ { 4 } ^ { t }$ t5 y4 y4 tokens h4 t5 h4 t5 ： h4 t5

tion from LLMs $Y ^ { t }$ . Then, we replace a certain number of continuously predicted token sequence $t _ { i }$ and features sequence $h _ { i }$ with the correct ones $\bar { t } _ { i } ^ { t }$ and $h _ { i } ^ { t }$ from LLMs. The number $N$ is computed as $N = \lambda \cdot d ( Y ^ { t } , \hat { Y } )$ , where $d ( \cdot )$ is the hamming distance, $\begin{array} { r } { \lambda = \frac { 0 . 4 * ( e p _ { t } - e p _ { c } ) } { e p _ { t } } } \end{array}$ , $e p _ { t }$ is the total epoch number and $e p _ { c }$ is the current epoch number. Note that our approach differs from the conventional glancing method (Qian et al. 2021), which only replaces tokens randomly. Instead, we concurrently replace continuous token and feature sequences preceding those to be predicted, which is illustrated in the dashed boxes labeled Choice 1, 2, 3 in Figure 2. This enhances the comprehension of intertoken relationships, as well as ensures the drafter can effectively utilize token sequences from ahead time step. This is particularly beneficial in SAR decoding, where preserving the sequence’s integrity is crucial in maintaining optimum performance. Subsequently, the revised token and feature sequences are re-input into the drafter to compute the training loss. The training loss consists of two components: the regression loss and the distillation loss. For regression loss, we utilize the Smooth L1 loss:

$$
\begin{array} { r } { f _ { i : i + k } ^ { d r a f t } = D r a f t ( t _ { i : i + k } , f _ { i - 1 : i + k - 1 } ) } \\ { L _ { \mathrm { r e g } } = \mathrm { S m o o t h L 1 } ( f _ { i : i + k } , f _ { i : i + k } ^ { d r a f t } ) } \end{array}
$$

Correspondingly, we optimize the drafter by calculating the distillation loss between the LLMs and the drafter:

$$
\begin{array} { r l } & { L _ { \mathrm { s o f t } } = K L \_ D i v ( p _ { i : i + k } , p _ { i : i + k } ^ { d r a f t } ) } \\ & { L _ { \mathrm { h a r d } } = C r o s s \_ E n t r o p y ( t _ { i : i + k } , t _ { i : i + k } ^ { d r a f t } ) } \\ & { L _ { \mathrm { d i s t } } = \alpha L _ { \mathrm { s o f t } } + ( 1 - \alpha ) L _ { \mathrm { h a r d } } } \end{array}
$$

where $f$ and $f ^ { d r a f t }$ denote features, $p$ and $p ^ { d r a f t }$ denote the probability distribution, $t$ and $t ^ { d r a f t }$ represent the tokens produced by the LLM and the drafter, respectively; and $\alpha$ , a constant coefficient set to 0.9. The losses $\displaystyle L _ { s o f t }$ and $L _ { h a r d }$ , represent the soft and hard label losses as described in (Hinton, Vinyals, and Dean 2015), are independently computed using the Kullback-Leibler divergence and Cross-Entropy, respectively.

Using the same weight matrix of the LM Head, the logits can be calculated as follows:

$$
\begin{array} { r } { p _ { i : i + k } = S o f t m a x ( L M ( f _ { i - 1 : i + k - 1 } ) ) } \\ { p _ { i : i + k } ^ { d r a f t } = S o f t m a x ( L M ( f _ { i - 1 : i + k - 1 } ^ { d r a f t } ) ) } \end{array}
$$

By integrating regression and distillation loss, we train the SAR Head with the combined loss function:

$$
L = L _ { \mathrm { r e g } } + \omega _ { \mathrm { d i s t } } L _ { \mathrm { d i s t } }
$$

$\omega _ { d i s t }$ is set to 0.1. Moreover, we employ data augmentation by adding random noise sampled from a uniform distribution $U ( - 0 . 1 , 0 . 1 )$ to avoid error accumulation in features.

# 3.3 Theoretical analysis of CSGD

We use a theory of information argument to illustrate the impact of CSGD. Take $k = 2$ as an example, let $X$ represent the next token, $Y$ the second-next token, and $C$ the input context (Omitted from equations for simplicity). Traditional AR methods focus on ${ \bar { H ( X ) } }$ , whereas SAR decoding with $k = 2$ targets $H ( X ) + H ( Y )$ , decomposed as (Gloeckle et al. 2024; Olsson et al. 2022):

$$
\begin{array} { c } { { H ( X ) = H ( X | Y ) + I ( X ; Y ) } } \\ { { H ( X ) + H ( Y ) = H ( Y | X ) + 2 I ( X ; Y ) + H ( X | Y ) } } \end{array}
$$

In Equation (10), the left two terms represent the training loss of 2-token prediction models. The right terms decompose the loss into a local cross-entropy term for the prefix $( C , X )$ , denoted as $H ( Y | X )$ , a mutual information term that captures the information about $Y$ contained in $X$ , denoted as $I ( X ; Y )$ , and term $H ( X | Y )$ describes the uncertainty about $X$ given the prefix $C$ and suffix $Y$ . We can observe that SAR decoding increases the weight of $I ( X ; Y )$ . However, conventional training methods for SAR decoding typically consider only the classical next-token entropy $H ( Y | X )$ , while the terms $I ( X ; Y )$ and $H ( X | Y )$ are always overlooked. As a result, these methods do not effectively learn the dependencies between tokens within a block. However, CSGD addresses this issue. When predicting $X$ , CSGD can leverage the features and tokens from one time step ahead in $C$ . The feature sequences represent the prefix, and the token sequences represent the mutual information capturing the correlation between $C$ and $X$ . Therefore, the term $H ( X | Y )$ should be changed to $H ( X | C )$ , and $I ( X ; Y )$ should be changed to $I ( X ; C )$ . In addition, when predicting token $Y$ , CSGD can see the information about $C$ and $X$ simultaneously. Therefore, the training loss of CSGD is changed to:

$$
\begin{array} { c } { { H ( X ) = H ( X | C ) + I ( X ; C ) } } \\ { { H ( X ) + H ( Y ) = H ( X | C ) + I ( X ; C ) } } \\ { { + H ( Y | X ) + I ( X ; Y ) } } \end{array}
$$

Equations (11) and (12) denote CSGD improves the dependency between tokens within a block, making the training loss of the SAR approach more similar to that of the AR approach. As a result, the probability distribution of multiple tokens predicted simultaneously by SAR decoding becomes more aligned with the distribution of tokens predicted sequentially by AR decoding. It further increases the probability that the tokens generated by SAR drafting will be accepted by the LLMs.

# 3.4 Custom-Designed Decoding Tree

Root Forward 1Forward 2 Can 1 assist you get to make a helpwith to do   
can assist to 1 1 1 0 0 0 0 0 0 1   
you do 1 1 1 0 0 0 0 0 1 1   
get the 1 1 0 0 0 0 0 0 0   
L post 0 0 1 0 0 0 0 0 0   
1   
make Chelp 0 0 0 0 1 1 0 0 0 0   
to 1 0 0 0 0 1 0 0 0   
Chelp the + 1 0 0 0 0 0 0 0 0   
with (baby) 1 0 0 0 0 0 0 1 0 0

A Custom-Designed Decoding Tree has been introduced to accommodate SAR drafting. It enables the drafter to perform m forward passes and generate $\mathrm { n ^ { * } k }$ tokens for each forward pass, where n is the number of the tree nodes. These $\mathrm { m } ^ { \ast } \mathrm { n } ^ { \ast } \mathrm { k }$ tokens are then organized in a tree structure. Meanwhile, the generated tokens and the corresponding features are concatenated for the subsequent forward passes. In traditional AR decoding, a causal mask is used, structured as a lower triangular matrix. It ensures that the former tokens can not access the later information. In contrast, Falcon employs a Relaxed Causal Mask, which allows the model to access tokens within the same $\mathbf { k } ^ { * } \mathbf { k }$ block and their predecessors in the tree, illustrated in Figure 3.

Subsequently, the drafts are organized as a token tree, whose nodes each represent a candidate token sequence. The correctness of the candidate token sequences are verified by the LLMs using a tree-based parallel decoding mechanism, which is consistent with SpecInfer (Miao et al. 2024). We record the accepted tokens and their corresponding features for the subsequent drafting phase.

Our custom-designed decoding tree, engineered specifically for SAR, is more efficient than those used for fully AR approaches. In SAR decoding, a tree-structure drafter generates $\mathbf { k }$ times as many tokens as an AR decoding drafter in a single forward pass. As a result, m forward passes can propose $\mathtt { k } ^ { * } \mathrm { m } ^ { * } \mathrm { n }$ tokens, which is k times greater than an AR tree would generate. This enhancement significantly boosts the efficiency of token generation for drafters, allowing the LLMs to verify more tokens concurrently. Therefore, this improvement increases the likelihood of drafting tokens being accepted by the LLMs, thereby accelerating the overall inference speed of the LLMs.

# 4 Experiments and Analysis

# 4.1 Models and Tasks

We conducted experiments on Vicuna models (7B, 13B) and LLaMA2-Chat models (7B, 13B). We evaluated Falcon across multiple tasks, including multi-turn dialogue, code generation, and mathematical reasoning, employing the MTbench (Zheng et al. 2024), HumanEval (Chen et al. 2021), and GSM8K (Cobbe et al. 2021) datasets, respectively. We conducted experiments with a batch size of 1, a temperature of 0 (greedy decoding), and adopted experimental settings consistent with other works like Eagle and Medusa.

# 4.2 Metrics

Like other speculative decoding methods, Falcon primarily focuses on latency rather than throughput. We assess acceleration effects using the following metrics:

• Wall-time speedup ratio: The actual test speedup ratio relative to vanilla AR decoding.   
• Acceptance rate $( \alpha )$ : The ratio of tokens generated by the drafter to all tokens gauges draft accuracy.   
• Average acceptance length $\left( \tau \right)$ : The number of accepted tokens in each draft phase.

# 4.3 Training

Falcon was trained on the ShareGPT dataset, utilizing 68,000 dialogue iterations. We employed the AdamW optimizer with beta values $( \beta _ { 1 } , \beta _ { 2 } )$ set to (0.9, 0.95) with a learning rate set to 3e-5. The settings are consistent with Eagle (Li et al. 2024). The semi-autoregressive head is trainable within two days on an H800 server.

![](images/44d563b602b9cd9079c57054d2c71f1d1c9fb874d7d5cad41b394f9033628582.jpg)  
Figure 4: Speedup ratio of Vicuna and LLaMA2-Chat on MT-bench for greedy (temperature $_ { = 0 }$ ).

Table 1: Comparison among Falcon and other speculative decoding methods in terms of speedup ratio.   

<html><body><table><tr><td colspan="4">MT-Bench HumEval</td><td>GSM8K</td></tr><tr><td>Model</td><td>Method</td><td>speedup</td><td>speedup</td><td>speedup</td></tr><tr><td rowspan="6">V7B</td><td>SpS PLD</td><td>1.82x</td><td>1.99x</td><td>1.71x</td></tr><tr><td></td><td>1.61x</td><td>1.82x</td><td>1.82x</td></tr><tr><td>Lookahead</td><td>1.63x</td><td>1.72x</td><td>1.84x</td></tr><tr><td>Medusa</td><td>2.06x</td><td>2.41x</td><td>2.22x</td></tr><tr><td>Eagle</td><td>2.82x</td><td>2.95x</td><td>2.72x</td></tr><tr><td>Falcon</td><td>3.10x</td><td>3.21x</td><td>2.92x</td></tr><tr><td rowspan="6">V13B</td><td>SpS</td><td>1.93x</td><td>2.23x</td><td>1.77x</td></tr><tr><td>PLD</td><td>1.58x</td><td>1.85x</td><td>1.68x</td></tr><tr><td>Lookahead</td><td>1.65x</td><td>1.71x</td><td>1.81x</td></tr><tr><td>Medusa</td><td>2.32x</td><td>2.44x</td><td>2.32x</td></tr><tr><td>Eagle</td><td>2.85x</td><td>2.97x</td><td>2.52x</td></tr><tr><td>Falcon</td><td>2.97x</td><td>3.12x</td><td>3.13x</td></tr><tr><td rowspan="4">LC7B</td><td>PLD</td><td>1.38x</td><td>1.52x</td><td>1.32x</td></tr><tr><td>Lookahead</td><td>1.61x</td><td>1.72x</td><td>1.58x</td></tr><tr><td>Eagle</td><td>2.71x</td><td>2.93x</td><td>2.89x</td></tr><tr><td>Falcon</td><td>2.94x</td><td>2.95x</td><td>2.91x</td></tr><tr><td rowspan="4">LC13B</td><td>PLD</td><td>1.42x</td><td>1.63x</td><td>1.41x</td></tr><tr><td>Lookahead</td><td>1.61x</td><td>1.72x</td><td>1.58x</td></tr><tr><td>Eagle</td><td>2.95x</td><td>3.25x</td><td>2.93x</td></tr><tr><td>Falcon</td><td>3.11x</td><td>3.51x</td><td>3.10x</td></tr></table></body></html>

# 4.4 Effectiveness

Figure 4 and Table 1 show the speedup comparison among Falcon and other state-of-the-art speculative decoding methods on MT-bench, HumanEval and GSM8K for greedy (temperature $\scriptstyle = 0$ ). Speedup ratios of Falcon, Eagle, and Medusa were fairly tested on an H800 server. The results of Speculative Sampling (SpS), Prompt lookup decoding (PLD), and Lookahead were copied from their technical reports. Falcon has achieved a speedup ratio ranging from $2 . 9 1 \mathrm { x } { - 3 . 5 1 \mathrm { x } }$ . Compared to AR drafters, Lookahead and Eagle, Falcon is faster by $1 . 7 9 \mathrm { x } \mathrm { - } 2 . 0 6 \mathrm { x }$ and $1 . 0 1 \mathrm { x } \mathrm { - } 1 . 2 4 \mathrm { x }$ , respectively. This improvement is because of Falcon’s SAR property; the draft model can generate multiple tokens each forward, which reduces the latency of each drafting phase and thus increases the total speedup ratio. Compared to Medusa, which is a SAR method as well, Falcon is faster by $1 . 2 4 \mathrm { x } \mathrm { - } 1 . 5 0 \mathrm { x }$ . It is attributed to the consideration of the dependency among tokens. Falcon adopts a CSGD method to maintain token dependency in the training stage. This enables the drafter to predict tokens more accurately than Medusa, effectively reducing the cost from the verification phase.

Table 2: Comparison among Falcon and other speculative decoding methods in terms of Acceptance rate $( \alpha )$ .   

<html><body><table><tr><td colspan="2"></td><td>MT-Bench</td><td>HumEval</td><td>GSM8K</td></tr><tr><td>Model</td><td>Method</td><td></td><td></td><td></td></tr><tr><td rowspan="3">V7B</td><td>Medusa</td><td>60.53</td><td>63.26</td><td>61.09</td></tr><tr><td>Eagle</td><td>74.59</td><td>76.18</td><td>75.47</td></tr><tr><td>Falcon</td><td>77.64</td><td>79.92</td><td>78.03</td></tr><tr><td rowspan="3">V13B</td><td>Medusa</td><td>61.77</td><td>64.58</td><td>62.39</td></tr><tr><td>Eagle</td><td>74.91</td><td>76.26</td><td>75.46</td></tr><tr><td>Falcon</td><td>77.60</td><td>80.34</td><td>80.22</td></tr><tr><td rowspan="2">LC7B</td><td>Eagle</td><td>72.98</td><td>74.76</td><td>73.89</td></tr><tr><td>Falcon</td><td>74.31</td><td>77.07</td><td>75.51</td></tr><tr><td rowspan="2">LC13B</td><td>Eagle</td><td>73.64</td><td>76.51</td><td>75.28</td></tr><tr><td>Falcon</td><td>75.27</td><td>78.52</td><td>77.13</td></tr></table></body></html>

Table 2 illustrates the Acceptance rate $( \alpha )$ of Falcon, Medusa, and Eagle. The experiments were performed on the H800 server. Falcon outperforms Eagle by $3 \%$ to $5 \%$ , indicating that more tokens are generated by the draft model rather than the target LLM. Compared to the SAR method (Medusa), Falcon achieves a higher $\alpha$ by $1 5 . 7 6 \% - 1 7 . 8 3 \%$ , which illustrates the importance of the token dependency in the draft phase.

Table 3 compares the average acceptance length $( \tau )$ . Although the Vicuna 7B model has a lower $\tau$ than Eagle, other sizes of models have shown a higher acceptance rate. It is demonstrated that the drafter of Falcon has a high prediction accuracy, even surpassing the AR models, and thus illustrates the effectiveness of CSGD. Compared to Medusa, the intrinsic design restricts it to conduct at most one forward each draft phase, which leads to a low $\tau$ . However, there is no such limitation to Falcon. In the experiment, Falcon performs four forward passes each drafting phase, achieving a higher $\tau$ than Medusa by 1.78-2.22.

Table 3: Comparison among Falcon, Eagle, and Medusa in terms of Average acceptance length $( \tau )$ .   

<html><body><table><tr><td colspan="2"></td><td>MT-Bench</td><td>HumEval</td><td>GSM8K</td></tr><tr><td>Model</td><td>Method</td><td>T</td><td>T</td><td>T</td></tr><tr><td rowspan="3">V7B</td><td>Medusa</td><td>1.51</td><td>1.71</td><td>1.55</td></tr><tr><td>Eagle</td><td>3.94</td><td>4.29</td><td>4.00</td></tr><tr><td>Falcon</td><td>3.34</td><td>3.88</td><td>3.70</td></tr><tr><td rowspan="3">V13B</td><td>Medusa</td><td>1.59</td><td>1.81</td><td>1.64</td></tr><tr><td>Eagle</td><td>2.91</td><td>3.13</td><td>2.97</td></tr><tr><td>Falcon</td><td>3.37</td><td>3.97</td><td>3.86</td></tr><tr><td rowspan="2">LC7B</td><td>Eagle</td><td>2.43</td><td>2.91</td><td>2.78</td></tr><tr><td>Falcon</td><td>2.82</td><td>3.30</td><td>3.02</td></tr><tr><td rowspan="2">LC13B</td><td>Eagle</td><td>2.73</td><td>3.21</td><td>2.98</td></tr><tr><td>Falcon</td><td>2.95</td><td>3.60</td><td>3.47</td></tr></table></body></html>

# 4.5 Ablation Study

Tree Attention Falcon, similar to Eagle and Medusa, employs tree attention in both drafting and verification phases, while methods like speculative decoding do not use it. To remove the effect of tree attention, we applied a chain to Falcon, whose length is equal to the tree height, in order to maintain the same forward times of the draft model. Table 4 shows the comparative results indicating the impact of tree attention. In Falcon, the implementation of tree attention mechanisms has been demonstrated to enhance the acceleration ratio by $1 . 1 2 \mathrm { x }$ , increase $\alpha$ by 1.22, and improve $\tau$ by $4 . 9 6 \%$ . This improvement is attributed to the fact that the tree attention increases the number of tokens validated and accepted, thereby augmenting the overall token throughput.

Coupled Sequential Glancing Distillation The SAR approach of Falcon enhances the token numbers of each forward at the cost of reduced accuracy. The CSGD training method was employed to mitigate the drop in precision. We conducted tests on Vicuna 7B under two conditions: training the SAR head with Eagle’s token-shift method (Li et al. 2024) alone and with the CSGD method. Table 4 illustrates the impact of CSGD. The implementation of CSGD resulted in a $1 . 1 7 \mathrm { x }$ increase in acceleration ratio, 0.56 in $\tau$ , and $3 . 2 6 \%$ improvement in $\alpha$ . The above results demonstrate that CSGD can significantly improve the drafter’s accuracy.

$k$ factor The $k$ factor determines the number of tokens the drafter generates in a single forward pass, which is important to SAR drafting. We tested conditions with $k = \{ 2 , 3 , 4 \}$ on Vicuna-7B. Considering the accuracy descending of SAR, we adopted fewer forward pass times in the drafting phase for $k = \{ 3 , 4 \}$ to achieve speedup ratios as high as possible. The corresponding heights of the decoding trees are 6 and 8, respectively. The results are presented in Table ??. With the increasing of the $k$ factor, we see a drop in all three metrics, but the speedup ratios are still higher than Eagle $( 2 . 8 2 \mathrm { x } )$ and Medusa (2.06x). This is due to the reduced time it takes for an SAR drafter to pass forward with the same number of tokens generated by AR drafting, indicating the inherent advantages of enhanced SAR drafting.

Table 4: The ablation study of Falcon of the impact of Tree Attention (TA) and CSGD on MT-Bench.   

<html><body><table><tr><td>Model</td><td>TA</td><td>CSGD</td><td>speedup</td><td>T</td><td></td></tr><tr><td rowspan="4">V7B</td><td>×</td><td>×</td><td>2.06x</td><td>1.64</td><td>62.86</td></tr><tr><td>×</td><td>√</td><td>2.27x</td><td>2.13</td><td>68.68</td></tr><tr><td>√</td><td>×</td><td>2.65x</td><td>2.78</td><td>74.37</td></tr><tr><td>√</td><td>√</td><td>3.10x</td><td>3.34</td><td>77.64</td></tr></table></body></html>

<html><body><table><tr><td>Model</td><td>k</td><td>speedup</td><td>T</td><td>a</td></tr><tr><td rowspan="3">V7B</td><td>2</td><td>3.10x</td><td>3.34</td><td>77.64</td></tr><tr><td>3</td><td>2.96x</td><td>2.52</td><td>72.21</td></tr><tr><td>4</td><td>2.86x</td><td>2.35</td><td>70.78</td></tr></table></body></html>

# 5 Conclusion and Future Work

In this paper, we propose Falcon, which utilizes the Coupled Sequential Glancing Distillation method to bolster the interdependencies among tokens, thereby enhancing the quality of the drafts. We also provide a theoretical analysis that clarifies the inner workings of our method. Additionally, we have developed a specialized decoding tree to support SAR drafting, which increases the token acceptance rate. Comprehensive evaluations indicate Falcon’s superior performance over the Vicuna and LLaMA2 model series. On the MT-bench, HumanEval, and GSM8K datasets, Falcon is $2 . 9 1 \mathrm { x } { - } 3 . 5 1 \mathrm { x }$ faster than vanilla Transformer, while maintaining a compact size comparable to two Transformer layers. Falcon appeals to applications that demand real-time responses and have limited computing resources.

The main challenge in accelerating the LLMs through speculative decoding lies in improving the accuracy and efficiency of the drafter under resource-limited conditions. Therefore, our future research efforts will focus on developing drafters that attain a high token acceptance rate with minimal overhead. This goal will be met by advanced semiautoregressive or potentially non-autoregressive decoding techniques that fortify the interdependence among tokens. In addition, the dynamic modification of the decoding tree is another avenue that warrants further exploration.