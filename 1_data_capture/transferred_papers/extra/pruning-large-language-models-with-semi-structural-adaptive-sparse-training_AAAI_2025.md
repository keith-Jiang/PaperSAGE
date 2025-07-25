# Pruning Large Language Models with Semi-Structural Adaptive Sparse Training

Weiyu Huang, Yuezhou Hu, Guohao Jian, Jun Zhu, Jianfei Chen†

Dept. of Comp. Sci. and Tech., Institute for AI, BNRist Center, THBI Lab, Tsinghua-Bosch Joint ML Center, Tsinghua University {hwy23, huyz21, jgh22}@mails.tsinghua.edu.cn;{dcszj, jianfeic}@tsinghua.edu.cn

# Abstract

The remarkable success of Large Language Models (LLMs) relies heavily on their substantial scale, which poses significant challenges during model deployment in terms of latency and memory consumption. Recently, numerous studies have attempted to compress LLMs using one-shot pruning methods. However, these methods often suffer from considerable performance degradation on complex language understanding tasks, raising concerns about the feasibility of pruning in LLMs. To address this issue, we propose Adaptive Sparse Trainer (AST), a novel and efficient retraining framework tailored for semi-structured sparse models. AST enables models to learn optimal masks during the weight update process without incurring additional computational overhead. Furthermore, we demonstrate that incorporating knowledge distillation significantly improves retraining efficiency and enhances model performance under fixed computational constraints. Additionally, a supplementary set of well-initialized parameters is integrated to further augment the model’s efficacy. AST achieves state-of-the-art performance with minimal training cost. When applied to the LLaMA2-7B model, AST reduces the perplexity and zero-shot accuracy gap between dense and 2:4 semi-structured sparse models to 0.6 and $1 . 1 6 \%$ , respectively, utilizing less than $0 . 4 \%$ of the pretraining tokens and GPU hours. Our work demonstrates the feasibility of deploying semi-structured sparse LLMs and offers a promising alternative for achieving highly compressed models when combined with existing quantization techniques.

Code — https://github.com/thu-ml/Adaptive-Sparse-Trainer Extended Version — https://arxiv.org/abs/2407.20584

# 1 Introduction

Transformer-based Large Language Models (LLMs) are equipped to handle complex tasks (Devlin et al. 2018; Brown et al. 2020; Achiam et al. 2023) and exhibit emergent abilities (Wei et al. 2022) due to their expanding parameter count. However, this continual growth in model size poses significant challenges for practical deployment. In particular, inference speed suffers due to the increasing computational and memory demands. This has spurred a series of efforts to develop effective model compression techniques aimed at reducing memory footprints and easing the constraints associated with deploying these large-scale models.

Model pruning (Frantar and Alistarh 2023; Han, Mao, and Dally 2016; Sun et al. 2023) is an effective technique for compressing LLMs by setting a proportion of weights to zero, thereby adhering to a specific sparsity pattern. Recently, N:M sparsity has emerged as a type of semistructured sparsity pattern that offers an optimal balance between precision and hardware efficiency. Specifically, N:M sparsity retains only N nonzero elements out of every group of M elements. This sparsity pattern can accelerate both matrix multiplication and memory access, potentially enhancing the performance of both pre-filling and decoding processes on off-the-shelf GPUs. Despite the promise of N:M sparsity, current state-of-the-art methods for pruning LLMs, such as SparseGPT (Frantar and Alistarh 2023) and Wanda (Sun et al. 2023), employ a post-training approach that determines the sparsity pattern in a layer-by-layer fashion without back-propagation. Although these methods improve efficiency, they can lead to significant performance degradation, particularly in knowledge-intensive tasks (Nowak et al. 2024), raising concerns about the feasibility of pruning LLMs. Additionally, while retraining pruned models has been successful in the pre-LLM era (Wang, Wohlwend, and Lei 2019; Lee, Ajanthan, and Torr 2019; Evci et al. 2020), its application to models with billions of parameters remains under-explored.

Although retraining sparse LLMs holds significant potential, it introduces several unique challenges: (1) Retraining is computationally expensive, necessitating techniques that ensure rapid convergence. (2) The output model must adhere to strict sparsity pattern, which adds additional constraints to the retraining process. (3) In order to achieve optimal performance, both masks and weights should be learnable during training. (4) Pruning LLMs may compromise essential language understanding and reasoning abilities, which are significantly harder to restore compared to simpler metrics such as perplexity. To this end, we propose a novel and efficient training method, Adaptive Sparse Trainer (AST), designed to produce high-performance sparse LLMs. For the first time, we demonstrate that pretrained 2:4 sparse LLMs can achieve competitive performance not only in terms of perplexity but also in more demanding knowledge-intensive tasks, making them viable for practical deployment. AST integrates the transition from dense to sparse models directly into the training process, gradually decaying unimportant weights to zero through a carefully designed decay mechanism. This approach allows for the revival of pruned weights during training, enabling a smooth and dynamic adjustment of the model’s connectivity pattern while adhering to the N:M sparsity structure. Furthermore, AST applies knowledge distillation using dense model as the teacher, which can significantly speed up model convergence and prevent sparse models from settling into local optima. It also helps the sparse model retain the valuable world knowledge and performance characteristics of the original dense model, thereby enhancing its generalization ability and compensating for the use of weaker training datasets. To further enhance performance, a supplementary set of parameters is integrated using information from pruned weights. When applied to the LLaMA2-7B model, AST achieves a 2:4 sparse configuration with minimal performance loss, demonstrating that compressed models can still perform effectively in practice. Our model further benefits from AWQ quantization, achieving competitive compression rates with state-ofthe-art performance.

The key contributions of this work are as follows:

• We propose Adaptive Sparse Trainer (AST), an novel semi-structured pruning framework designed to compress large language models efficiently while maintaining high performance.   
• AST introduces a gradual decay scheduler (Annealing SR-STE) combined with knowledge distillation to accelerate model convergence and improve performance on complex tasks.   
• We introduce SLoRB, a technique that boosts model performance by adding a supplementary set of wellinitialized parameters.   
• When applied to LLaMA2-7B model, our 2:4 sparse model experiences only a 0.6 increase in perplexity and a $1 . 1 6 \%$ accuracy drop in zero-shot tasks, with less than $0 . 4 \%$ of the pretraining cost.

# 2 Related Work

Network Pruning Pruning is a prevalent technique aimed at reducing model size and computational costs. It originated from methods like OBD (LeCun, Denker, and Solla 1990) and OBS (Hassibi, Stork, and Wolff 1993). Based on sparsity patterns, pruning methods can be broadly categorized into unstructured, structured, and semi-structured pruning. Unstructured pruning removes individual weights (Han, Mao, and Dally 2016; Paul et al. 2023), which can maintain performance even with high sparsity. However, due to its random pattern, unstructured models are difficult to accelerate. Structured pruning (Liu et al. 2017; Molchanov et al. 2019; Nova, Dai, and Schuurmans 2023; Shen et al. 2022), on the other hand, removes entire neurons, filters, or attention heads, resulting in models that are easier to accelerate on standard hardware but often suffer from severe performance loss. Semi-structured sparsity (e.g., N:M sparsity) (Hubara et al. 2021) has been applied as a trade-off between performance and achieving actual speedup. Recently, a series of works (Frantar and Alistarh 2023; Sun et al. 2023; Zhang et al. 2024a,b) have made progress in pruning LLMs with billions of parameters. However, these pruned models from training-free methods still fall short in complex zeroshot performance.

Retraining Pruned Models Another line of research (Han et al. 2015; Singh and Alistarh 2020; Renda, Frankle, and Carbin 2020; Zhou et al. 2023) has focused on retraining pruned models to enhance performance. While retraining has been shown to be effective with smaller models and simpler tasks (Kurtic and Alistarh 2022; Zhu and Gupta 2018), it often involves additional training steps (Frankle and Carbin 2019) or introduces extra parameters for the pruning process (Shi et al. 2023), which limits its application to large-scale models due to the substantial computational resources required. Other retraining methods focus on unstructured sparsity, which may struggle when given stricter sparsity patterns (Lee, Ajanthan, and Torr 2019; Evci et al. 2020). Recently, Sheared LLaMA (Xia et al. 2024) employed a twostage structured pruning process to train models that outperform others of similar size. However, they employ a computationally intensive pruning stage, unsuitable for more finegrained sparsity. Our work proposes a lightweight pruning process that enables the retraining of semi-structured sparse large language models with minimal training costs.

Combining Pruning with Quantization Pruned models can be further compressed using quantization. Earlier methods like Deep Compression (Han, Mao, and Dally 2016) combined pruning and quantization to significantly reduce the size of neural networks. More recent work (Frantar and Alistarh 2023; Sun et al. 2023; Guo et al. 2024) has combined sparsity with quantization in large language models. In our work, we report results using AWQ quantization (Lin et al. 2024) with our semi-structured sparse model.

# 3 Methods

We begin by revisiting a naive approach to training sparse models before introducing our method. The full training pipeline is illustrated in Figure 1.

# 3.1 Sparse Training

Pruning model weights is equivalent to multiplying them by an element-wise mask. For example, consider the matrix multiplication in a linear layer:

$$
Z = X W ^ { \top } , \quad Z \in \mathbb { R } ^ { N \times D } , \quad X \in \mathbb { R } ^ { N \times C } , \quad W \in \mathbb { R } ^ { D \times C } ,
$$

where $X , W$ , and $Z$ represent the model input, weight matrix, and output activation, respectively. The pruned weight matrix can be expressed as:

$$
\tilde { W } = m ( W ) \odot W , \quad m ( W ) \in \{ 0 , 1 \} ^ { D \times C } ,
$$

where $m ( \cdot )$ is a mapping that selects a mask based on the weight matrix $W$ . In this work, we focus on N:M sparsity (Hubara et al. 2021), where there are $N$ nonzero elements in every $M$ consecutive weights in the same row. However, when implementing the backward pass for a sparse model with automatic differentiation, the gradient cannot flow through the discrete mask $m ( W )$ . A straightforward approach to solve this issue is the straight-through estimator (STE) (Bengio, Le´onard, and Courville 2013) which updates the parameters using the gradient with respect to the masked weight $\tilde { W } _ { t }$ , where $W _ { t }$ is the parameter at iteration $t$ :

![](images/cf287ff6d9933856329a7c3dec17914d1f6288089c2734ba72f43bfbc5d7be2d.jpg)  
Figure 1: (Left) In the naive training baseline, the mask remains constant during training, which can result in suboptimal performance. (Right) Adaptive sparse training enables both mask and weight learning through a scheduled decay term. AST also utilizes distillation and SLoRB parameters to speed up convergence and improve performance.

$$
W _ { t + 1 }  W _ { t } - \gamma _ { t } g ( \tilde { W } _ { t } ) ,
$$

here $g ( \tilde { W } _ { t } )$ represents the gradient with respect to $\tilde { W } _ { t }$ and $\gamma _ { t }$ indicates the learning rate at iteration $t$ .

A previous method (Sun et al. 2023) employs a fixed mask derived from one-shot pruning techniques and updates only the remaining parameters using language modeling loss. However, this strategy has two significant limitations. First, it discards the valuable information embedded in the pruned weights and prevents a smooth transition from a dense to a sparse model, resulting in slower convergence and suboptimal performance. Second, relying solely on the language modeling loss during retraining increases the risk of the sparse model becoming trapped in a local optimum, leading to poor outcomes.

To address these issues, we maintain all model weights and select the mask on-the-fly while gradually decaying non-essential weights to zero, thereby enabling a smooth transition ( 3.2). Additionally, we leverage knowledge distillation to guide the model toward a global optimum, significantly accelerating convergence under a fixed computational budget ( 3.3). Finally, we offer an optional method to further enhance model performance by incorporating an additional set of well-initialized parameters, as detailed in ( 3.4).

# 3.2 Adaptive Mask Learning with Annealing SR-STE

As highlighted in previous work (Frankle and Carbin 2019; Evci et al. 2020), the connectivity pattern is as crucial as the weights themselves in determining the performance of sparse models. Therefore, we enable the model to adaptively learn the optimal mask during training instead of fixing it permanently. Specifically, we recalculate $m ( W _ { t } )$ based on the magnitude criterion every $\Delta t$ iterations and add a decay term to the masked weights. The intuition behind this approach is straightforward: important weights typically receive strong gradients of the same sign during training, causing their magnitude to grow despite the decay, and they will eventually be revived. In contrast, less important weights tend to receive weaker, mixed gradient signals, leading them to decay to zero and remain masked under our settings. We build upon SR-STE (Zhou et al. 2021) by applying L2 decay to masked weights, as this approach helps preserve overall model performance.

However, selecting the appropriate decay strength is challenging. If the decay signal is too strong, masked weights will struggle to regenerate, leading to an almost fixed mask. On the other hand, if the decay signal is too weak, the model will fail to filter out important weights, resulting in heavy oscillation that hinders model convergence. To address this dilemma between stability and exploration, we propose adding a moving decay schedule, with weaker decay strength at the beginning to encourage the model to explore various connectivity patterns and a stronger decay signal toward the end to facilitate model convergence.

Formally, we update the model weights with the following equation:

$$
\lambda _ { W } ( t ) = { \left\{ \begin{array} { l l } { \alpha t , } & { { \mathrm { i f ~ } } 0 \leq t \leq T _ { 0 } , } \\ { \alpha T _ { 0 } , } & { { \mathrm { i f ~ } } T _ { 0 } \leq t \leq T _ { 1 } , } \end{array} \right. }
$$

$$
W _ { t + 1 }  W _ { t } - \gamma _ { t } ( g ( { \tilde { W } } _ { t } ) + \lambda _ { W } ( t ) ( { \overline { { m ( W _ { t } ) } } } \odot W _ { t } ) ) ,
$$

where $\lambda _ { W } ( t )$ denotes the decay factor, $\overline { { m ( W _ { t } ) } }$ denotes the mask for pruned weights at iteration $t$ , $T _ { 1 }$ denotes the total number of training batches, and $T _ { 0 }$ represents the batch at which the decay factor stops increasing. Compared with STE in Equation 1, masked weights receive a decay signal proportional to the decay factor $\lambda _ { W } ( t )$ , which starts small at the beginning and increases later on. This method, termed Annealing SR-STE (ARS-STE), has been shown to improve model performance compared to naive SR-STE. We also provide a detailed analysis of mask oscillation behavior during training in Section 7 of the Appendix.

We experimented with various one-shot pruning criteria incorporating activation information (Frantar and Alistarh 2023; Zhang et al. 2024a) for mask selection. However, the classic magnitude criterion consistently outperformed these methods in both computational efficiency and performance. As activation-based approaches incur higher computational costs and are less reliable due to activation fluctuations caused by weight updates, leading to degraded performance in later training stages.

Since the naive SR-STE is only compatible with the SGD optimizer, we introduce further adjustments to support more advanced optimizers such as AdamW. Specifically, as AdamW maintains both first-order and second-order moments of the gradient, we decouple the weight decay term from the first-order moment instead of directly adding decay to the gradient, as done in (Hu et al. 2024). This separation ensures that the weight decay term depends solely on the current weight, avoiding interference with momentum calculations. We then use the decoupled first-order signal to update the second-order moment. Detailed expressions and explanations are provided in Section 6 of the Appendix.

# 3.3 Alleviating the Retraining Dilemma through Knowledge Distillation

A key distinction between pretraining and retraining pruned models lies in how their parameters are initialized. In the pretraining setting, parameters are typically randomly sampled, whereas in retraining, they are inherited from a welltrained model. Consequently, pruned models retain a portion of the patterns acquired during pretraining.

We discovered that this feature tends to trap pruned models in local optima. Specifically, while retraining can achieve much faster initial convergence, it often fails to reach an optimal state later on; we refer to this phenomenon as the Retraining Dilemma. As illustrated in Figure 2, although using cross-entropy loss during retraining initially leads to a quicker reduction in training loss compared to pretraining, the test set perplexity remains unstable and elevated. We attribute this to the fact that, unlike randomly initialized models, the pruned model is more prone to overfitting on the limited data available at the start due to its prior knowledge. This overfitting hampers the model’s ability to learn global features and achieve optimal convergence. Notably, this issue persists even when using a significantly smaller learning rate than in pretraining, suggesting that simply reducing the learning rate is insufficient to resolve the problem.

As a solution, we found that applying KL-divergence loss (Kullback and Leibler 1951) can address this issue. Unlike the language modeling loss, KL-divergence loss measures the difference between two probability distributions, offering richer signals that help mitigate overfitting in the early stages of training. Consequently, we employ the following loss function:

$$
\mathcal { L } _ { l o g i t } = D _ { K L } \big ( p _ { \theta _ { t } } | | p _ { \theta _ { s } } \big ) = \frac { 1 } { B \times S } \sum _ { i = 1 } ^ { B \times \mathrm { s e q } } p _ { \theta _ { t } } \big ( x _ { i } \big ) \log \frac { p _ { \theta _ { t } } \big ( x _ { i } \big ) } { p _ { \theta _ { s } } \big ( x _ { i } \big ) } ,
$$

$$
\mathcal { L } = \alpha \mathcal { L } _ { l o g i t } + ( 1 - \alpha ) \mathcal { L } _ { t a s k } ,
$$

![](images/b1c5fd2069ead9327660c6b639b3b41c05c90b2349a28f18f708201fce38ce0b.jpg)  
Figure 2: (Upper) The Wikitext perplexity curve for retraining GPT2 with and without knowledge distillation. (Lower) The training loss curve for pretraining and retraining.

where $\mathcal { L } _ { t a s k }$ is the cross-entropy loss, $B$ is the batch size, $S$ is the sequence length, and $p _ { \theta _ { t } }$ and $p _ { \theta _ { s } }$ are the probability distributions of the teacher and student models, respectively.

We observe that the Retraining Dilemma is more pronounced in smaller models; therefore, we typically apply a larger $\alpha$ for these models. We also experimented with various distillation methods that utilize intermediate information for distillation. However, we found that adding constraints to intermediate outputs hinders the model’s generalization ability and leads to undesirable outcomes.

# 3.4 Sparse Low-Rank Boosting

The pruned model with 2:4 sparsity, which retains only half of the parameters under strict structural constraints, experiences a reduction in expressive capacity. To mitigate this, we incorporate a LoRA (Low-Rank Adaptation) (Hu et al. 2022) shape parameter that is trained alongside the original parameters, helping to bridge the performance gap between dense and sparse models with minimal memory overhead. Unlike traditional LoRA fine-tuning, where the original model parameters are frozen and only the adapters are trained, our approach trains both the sparse parameters and the adapter weights simultaneously. This approach recognizes that while the classic LoRA method freezes the original model to prevent overfitting and catastrophic forgetting during downstream tasks, our method enhances generalization by training both the additional and original weights on a pretraining dataset, thereby enabling concurrent training.

Another notable aspect of our method is the initialization strategy. Classic LoRA typically employs random initialization techniques, such as Xavier initialization (Glorot and Bengio 2010) or Kaiming initialization (He et al. 2015). However, in our approach, we leverage the pruned weights as additional information to initialize the adapter weights, thereby accelerating the training process. Given that in 2:4 sparsity, each neuron can only access half of the inputs, we find that incorporating additional information to retain the weight’s first-order information helps preserve the model’s capacity. Specifically, for a weight matrix $W$ of size $N \times d$ and its corresponding mask matrix $M$ , we select the rank $r$ as $\textstyle { \frac { d } { k } }$ , where $k$ can be 64, 32, 16, and so on. A smaller $k$ improves performance but also increases memory usage. We select a projection matrix $X$ of size $\begin{array} { r } { \frac { d } { k } \times d } \end{array}$ and a SLoRB weight matrix $S$ of size $N \times { \frac { d } { k } }$ . The matrices $X$ and $S$ are defined as follows:

![](images/d53075c8ea79e31e3c015e2051d8e283cdb0095384681784ab8f5be2241c331a.jpg)  
Figure 3: Visualization of SLoRB Initialization: Consider a weight matrix of size 3 by 8, with $k = 4$ . Weight $S _ { i j }$ is broadcasted within group $G _ { i j }$ after multiplication with the projection matrix.

$$
x _ { i j } = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } i \cdot k \leq j \leq ( i + 1 ) \cdot k - 1 , } \\ { 0 , } & { { \mathrm { o t h e r w i s e , } } } \end{array} \right. }
$$

$$
S _ { i j } = \frac { 1 } { k } \sum _ { p = j \cdot k } ^ { ( j + 1 ) \cdot k - 1 } W _ { i p } \cdot \neg M _ { i p } .
$$

We define Group $G _ { i j }$ as the elements from $j \cdot k$ to $( i + 1 ) \cdot k - 1$ in row $i$ . As illustrated in Fig 3, each SLoRB weight $S _ { i j }$ is broadcast within Group $G _ { i j }$ . By setting $S _ { i j }$ as the mean of all pruned weights in Group $G _ { i j }$ , this design ensures that the total mean of groups $G _ { i j }$ remains consistent after pruning. Although different initialization methods may ultimately converge to similar outcomes given sufficient training steps, our method converges more rapidly, making it particularly advantageous when the computational budget is limited, as demonstrated in Section 4 of the Appendix. We refer to this method as Sparse LowRank Boosting (SLoRB), which is an optional technique that trades off memory overhead for enhanced performance. The complete pseudo-code for our method is provided in Algorithm 1.

# 4 Experiments

# 4.1 Experiment Setup

Model Configurations. We report the performance of Adaptive Sparse Trainer (AST) across three different LLM model families: LLaMA2-7B (Touvron et al. 2023), OPT (Zhang et al. 2022), and GPT2 (Brown et al. 2020). We

Input: Weight of linear layers $\mathcal { W } ^ { ( 0 ) }$ ; Mask update frequency $\Delta t$ ; Total training iteration $T _ { 0 }$ ; SLoRB parameter $\mathbf { k }$ ; Decay increase iteration $T _ { 1 }$ ;

1: for $W ^ { ( 0 ) } \in \mathcal { W } ^ { ( 0 ) }$ do   
2: Initialize the mask $m ( W ^ { ( 0 ) } )$ based on magnitude;   
3: if Use SLoRB then   
4: Initialize SLoRB weight $S ^ { ( 0 ) } \in S ^ { ( 0 ) }$ and $X ^ { ( 0 ) } \in$   
$\mathcal { X } ^ { ( 0 ) }$ by Equation 5 and 6;   
5: end if   
6: end for   
7: for $\mathfrak { t } = 1 , 2 , . . T _ { 0 }$ do   
8: if t mod $\Delta t = = 0$ then   
9: for $W ^ { ( t ) } \in \mathcal { W } ^ { ( t ) }$ do   
10: Update model mask mask $m ( W ^ { ( t ) } )$ ;   
11: end for   
12: end if   
13: Compute decay for term $\lambda _ { W } ( t )$ from Equation 2;   
14: Compute gradient $g ( W ^ { ( t ) } )$ by back-propagation on   
distillation loss $\mathcal { L }$ from Equation 4;   
15: Update weight $W ^ { ( t ) }$ by Equation 3;   
16: if Use SLoRB then   
17: Update $S ^ { ( t ) }$ and $X ^ { ( t ) }$ by gradient;   
18: end if   
19: end for   
20: return the pruned model.

present results for two different sparsity patterns: ASTNaive, which adheres to a strict 2:4 sparsity pattern without additional weights, and AST-Boosted, which incorporates extra SLoRB weights. For AST-Boosted models, we selected $k ~ = ~ 1 6$ , introducing an additional $1 2 . 5 \%$ of parameters. Optimal hyperparameters were identified through a grid search, with the specific hyperparameters and training details provided in Section 3 of the Appendix.

Data. For training smaller models like the OPT and GPT2 model families, we utilized the C4 dataset (Raffel et al. 2020). For the LLaMA2-7B model, we employed a more comprehensive dataset, RedPajama-v11, which encompasses data from seven domains: CommonCrawl, C4, GitHub, Wikipedia, Books, ArXiv, and StackExchange. Additionally, we leveraged the dynamic batching feature provided in the ShearedLLaMA (Xia et al. 2024) codebase.

Baseline. For the GPT2 and OPT models, we compare our methods with both training-free and training-based approaches. Among training-free methods, we include comparisons with SparseGPT (Frantar and Alistarh 2023) and Wanda (Sun et al. 2023). For training-based methods, given the lack of existing results for retraining generative language models with N:M sparsity, we implemented iterative magnitude pruning (Frankle and Carbin 2019) and gradual magnitude pruning (Kurtic and Alistarh 2022), both of which can be adapted to achieve the desired sparsity. To ensure a fair comparison in terms of computational cost, we report results for training-based methods that include distillation, as we have observed that incorporating distillation significantly enhances performance. Due to computational constraints, we report the results of training-based baselines only for the GPT2 and OPT models.

Table 1: Perplexity results on raw-WikiText2 on 2:4 sparsified language models. AST outperforms both training-free and other training-based methods with similar computational costs.   

<html><body><table><tr><td rowspan="2">Method</td><td rowspan="2">Training</td><td colspan="3">OPT</td><td colspan="4">GPT2</td></tr><tr><td>125M</td><td>350M</td><td>1.3B</td><td>124M</td><td>350M</td><td>774M</td><td>1.5B</td></tr><tr><td>Dense</td><td></td><td>27.76</td><td>22.00</td><td>14.62</td><td>29.95</td><td>21.72</td><td>19.43</td><td>17.40</td></tr><tr><td>SparseGPT</td><td>X</td><td>45.58</td><td>40.33</td><td>29.03</td><td>50.09</td><td>31.03</td><td>25.98</td><td>21.14</td></tr><tr><td>Wanda</td><td>X</td><td>60.91</td><td>50.16</td><td>23.92</td><td>115.64</td><td>63.71</td><td>49.97</td><td>30.44</td></tr><tr><td> Iterative Magnitude Pruning</td><td>√</td><td>38.37</td><td>30.29</td><td>23.94</td><td>40.08</td><td>29.86</td><td>24.31</td><td>20.83</td></tr><tr><td>Gradual Magnitude Pruning</td><td>√</td><td>31.51</td><td>25.98</td><td>16.78</td><td>33.48</td><td>24.83</td><td>22.01</td><td>18.96</td></tr><tr><td>AST-Naive(Ours)</td><td>√</td><td>30.22</td><td>24.65</td><td>15.85</td><td>32.23</td><td>23.65</td><td>21.29</td><td>18.33</td></tr><tr><td>AST-Boosted(Ours)</td><td>√</td><td>28.68</td><td>24.03</td><td>15.43</td><td>31.13</td><td>23.03</td><td>20.66</td><td>18.01</td></tr></table></body></html>

Table 2: Accuracy $( \% )$ of various open-sourced models on seven zero-shot tasks.   

<html><body><table><tr><td>Models (Sparsity Pattern)</td><td>Parameters</td><td>BoolQ</td><td>RTE</td><td>HellaSwag</td><td>WinoGrande</td><td>ARC-e</td><td>ARC-c</td><td>OBQA</td><td>Mean</td></tr><tr><td>LLaMA2-7B (Dense)</td><td>6.7B</td><td>77.73</td><td>63.89</td><td>57.18</td><td>69.04</td><td>76.17</td><td>42.91</td><td>31.60</td><td>59.78</td></tr><tr><td>Sheared-LLaMA-2.7B (Dense)</td><td>2.7B</td><td>65.99</td><td>50.54</td><td>51.21</td><td>65.03</td><td>67.29</td><td>33.27</td><td>28.80</td><td>51.73</td></tr><tr><td>INCITE-Base-3B (Dense)</td><td>2.8B</td><td>67.40</td><td>52.34</td><td>47.91</td><td>62.98</td><td>67.68</td><td>31.74</td><td>27.60</td><td>51.09</td></tr><tr><td>Open-LLaMA-3B-v2 (Dense)</td><td>3.4B</td><td>65.89</td><td>55.69</td><td>52.12</td><td>63.59</td><td>68.34</td><td>34.32</td><td>26.00</td><td>52.13</td></tr><tr><td>LLaMA-7B (Sparse)</td><td>3.4B</td><td>73.21</td><td>61.34</td><td>54.86</td><td>66.18</td><td>70.24</td><td>35.68</td><td>31.80</td><td>56.19</td></tr><tr><td>LLaMA2-7B (Sparse)</td><td>3.4B</td><td>73.12</td><td>66.06</td><td>54.66</td><td>67.87</td><td>73.61</td><td>39.93</td><td>28.60</td><td>57.68</td></tr><tr><td>LLaMA2-7B (Sparse+SLoRB)</td><td>4.2B</td><td>75.04</td><td>66.06</td><td>55.24</td><td>68.48</td><td>74.91</td><td>41.11</td><td>29.40</td><td>58.62</td></tr></table></body></html>

For the LLaMA2-7B model, we compare our approach against strong dense pre-trained LLMs with a similar number of non-zero parameters. Additionally, we include a comparison with the LLaMA-7B 2:4 sparsity model provided in the Wanda (Sun et al. 2023) paper.

Evaluation. We evaluated the WikiText-2 perplexity for all models and assessed zero-shot and few-shot performance on LLaMA2-7B using EleutherAI’s LM Harness (Gao et al. 2021). Our evaluation tasks included zero-shot ARC Easy (Clark et al. 2018), OpenBookQA (Mihaylov et al. 2018), WinoGrande (Sakaguchi et al. 2021), RTE from the GLUE benchmark (Wang et al. 2018), HellaSwag (Zellers et al. 2019), ARC Challenge (Clark et al. 2018), BoolQ (Clark et al. 2019), as well as more complex tasks like the MMLU benchmark (Hendrycks et al. 2021a), GSM8K (Cobbe et al. 2021), and MATH (Hendrycks et al. 2021b).

# 4.2 Main Results

Language Modeling In Table 1, we compare the perplexity results of our sparse models with those of the baselines. Our methods significantly outperform both training-free and training-based approaches across various models, achieving results close to those of the dense baseline. A plausible reason why training-based methods like IMP fail is that while a lottery ticket may exist for a randomly initialized model, this may not be the case for a well-trained model. As for the gradual baseline, applying a proportion of training steps at low sparsity may not provide the model with sufficient time to converge.

Zero-shot Result In Table 2, we present the accuracy results of seven zero-shot tasks for AST and baseline models. Our method performed best in most of the tasks. It should be noted that our sparse models consistently outperform smaller dense models with a similar parameter count, suggesting a promising approach for obtaining parameterefficient models. Moreover, our method requires only a minimal amount of training tokens to converge. For example, when training the LLaMA2-7B model, we utilized only 7B tokens, which is less than $0 . 4 \%$ of those used in pretraining, making AST applicable to open-source models.

Our method can also be seamlessly adapted to current quantization techniques to achieve extremely compressed models with minimal performance drop. We provide results using AWQ quantization methods in section 8 of the Appendix.

# 4.3 Additional Results for LLaMA2-7B

Recent findings (Nowak et al. 2024) have shown that previous pruning methods suffer significant performance degradation in knowledge-intensive tasks. To address this, we tested our LLaMA2-7B model on more complex tasks. As shown in Table 3, our model preserves most of the knowledge from the dense model and continues to perform well in knowledge-intensive tasks compared with previous pruning methods. This provides strong evidence that 2:4 pruned models possess considerable potential, contrary to the observations in previous studies.

Table 3: Results of perplexity and knowledge-intensive tasks for LLaMA2-7B models with 2:4 sparsity. The symbols $\downarrow$ and $\uparrow$ indicate that lower and higher values are better, respectively. (LoRA methods are finetuned with $\scriptstyle r = 6 4$ )   

<html><body><table><tr><td>LLaMA2-7B</td><td>Dense</td><td>Wanda</td><td>AST-Naive</td><td>AST-Boosted</td></tr><tr><td>Perplexity↓</td><td>5.12</td><td>11.02</td><td>5.82</td><td>5.69</td></tr><tr><td>MMLU (5-shot) ↑</td><td>45.3</td><td>27.6</td><td>37.9</td><td>38.2</td></tr><tr><td>MATH (4-shot) ↑</td><td>5.38</td><td>2.86</td><td>4.42</td><td>4.64</td></tr><tr><td>GMS8K (LoRA Finetuned) ↑</td><td>40.3</td><td>32.1</td><td>35.6</td><td>36.2</td></tr></table></body></html>

Table 4: Ablation study of different methods in training sparse models.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="2">GPT</td></tr><tr><td>124M 350M</td><td>OPT 774M 125M</td></tr><tr><td>Dense</td><td>29.95 21.72</td><td>19.43 27.76</td></tr><tr><td>Naive Training</td><td>28.65</td><td></td></tr><tr><td>No distillation</td><td>40.34 29.79</td><td>39.46</td></tr><tr><td>Static SR-STE</td><td>39.29 29.08 27.21</td><td>36.97</td></tr><tr><td>FixedMask</td><td>32.84 24.04 21.73</td><td>31.08</td></tr><tr><td>AST-Naive(Ours)</td><td>32.93 24.18 21.95 32.23 23.65 21.29</td><td>31.06 30.22</td></tr></table></body></html>

# 4.4 Ablation Study

Training Sparse Models. Our AST-Naive method employed distillation, Annealing SR-STE, and adaptive masking during training. In Table 4, we present an ablation study to assess the effects of each of these components. For a fair comparison, we apply additional training steps to nondistillation methods. Specifically, we analyze the impact of training without distillation, using a naive SR-STE decay factor, and fixing the mask during training, each component in isolation. Additionally, we provide results for the naive training baseline in Figure 1 mentioned above. We demonstrate that all three approaches contribute to performance gains compared to naive training across various models.

Distillation Functions We also conducted ablation studies on different distillation functions used in previous work, including TinyBERT (Jiao et al. 2020), MobileBERT (Sun et al. 2020), and Sparse-Finetuning (Kurtic et al. 2023), which use attention and hidden states for distillation. Additionally, we examined MiniLLM (Gu et al. 2024), which employs reverse KL divergence. We find that using intermediate information during the distillation of generative language models is detrimental, and that KL loss is sufficient for optimal performance. Reverse-KL yields performance similar to that of forward KL. Detailed descriptions of the distillation loss functions for each of these methods are provided in Section 5 of the Appendix.

# 4.5 Speedup

Table 6 presents speedup results obtained using TensorRT$\mathrm { L L M } ^ { 2 }$ . We evaluate the actual end-to-end decoding speedup on two GPU architectures with the LLaMA2-7B 2:4 sparse model. We employ throughput, measured as the number of tokens processed per second, as the primary evaluation metric. Across a range of input and output lengths, the 2:4 sparse model demonstrates an overall acceleration of $1 . 3 3 \times$ to $1 . 8 3 \times$ compared to its dense counterpart, highlighting its potential for practical deployment.

Table 5: Ablation study on different distillation loss for training sparse models.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="3">GPT</td><td rowspan="2">OPT 125M</td></tr><tr><td>124M</td><td>350M</td><td>774M</td></tr><tr><td>Dense</td><td>29.95</td><td>21.72</td><td>19.43</td><td>27.76</td></tr><tr><td>TinyBERT</td><td>42.75</td><td>33.35</td><td>30.86</td><td>39.39</td></tr><tr><td>MobileBERT</td><td>44.87</td><td>31.67</td><td>29.75</td><td>40.33</td></tr><tr><td>Sparse-Fintuning</td><td>41.19</td><td>29.42</td><td>26.19</td><td>38.96</td></tr><tr><td>MiniLLM</td><td>32.20</td><td>23.64</td><td>21.31</td><td>30.24</td></tr><tr><td>KL Loss(Ours)</td><td>32.23</td><td>23.65</td><td>21.29</td><td>30.22</td></tr></table></body></html>

Table 6: Speedup results using TensorRT-LLM on RTX4090 and L20 GPUs with different input and output sequence lengths, measured by throughput (tokens/s).   

<html><body><table><tr><td colspan="3">RTX4090</td></tr><tr><td>Inp Len, OutLen</td><td>Sparse Dense</td><td>Speedup</td></tr><tr><td>128,128</td><td>70.23 52.94</td><td>1.33x</td></tr><tr><td>128,1024</td><td>69.11 52.00</td><td>1.33x</td></tr><tr><td>1024,128</td><td>68.06 51.10</td><td>1.33x</td></tr><tr><td>1024,1024</td><td>67.41 50.37</td><td>1.34x</td></tr><tr><td colspan="3">L20</td></tr><tr><td>Inp Len, Out Len</td><td>Sparse Dense</td><td>Speedup</td></tr><tr><td>128,128</td><td>54.75 29.86</td><td>1.83x</td></tr><tr><td>128,1024</td><td>53.81 29.57</td><td>1.82x</td></tr><tr><td>1024,128</td><td>52.49 29.18</td><td>1.80x</td></tr><tr><td>1024,1024</td><td>51.64 28.94</td><td>1.78x</td></tr></table></body></html>

# 5 Conclusion

In this paper, we introduce the Adaptive Sparse Trainer (AST), a novel and efficient training pipeline for semistructured sparse models. AST effectively narrows the precision gap between dense and sparse LLMs in terms of perplexity and accuracy on zero-shot tasks, while keeping training costs minimal. Our results demonstrate that pruning LLMs is feasible with minimal performance loss on knowledge-intensive tasks, and that large semi-structured sparse models can outperform dense models of similar decoding speed when supported by the appropriate software and hardware. Although our findings contribute to advancing the retraining of pruned models with billions of parameters, we acknowledge that the limited number of training tokens, due to computational constraints, remains an area for future exploration. Expanding this work to larger models or increasing training tokens count could provide valuable insights and further enhance the effectiveness of our methods.