# COSEE: Consistency-Oriented Signal-Based Early Exiting via Calibrated Sample Weighting Mechanism

Jianing $\mathbf { H } \mathbf { e } ^ { 1 }$ , Qi Zhang1, Hongyun Zhang1, Xuanjing Huang2, Usman Naseem3, Duoqian Miao1

1Tongji University, China 2Fudan University, China 3Macquarie University, Australia {jnhe, zhangqi cs, zhanghongyun, dqmiao} $@$ tongji.edu.cn, xjhuang@fudan.edu.cn, engr.usmannaseem87@gmail.com

# Abstract

Early exiting is an effective paradigm for improving the inference efficiency of pre-trained language models (PLMs) by dynamically adjusting the number of executed layers for each sample. However, in most existing works, easy and hard samples are treated equally by each classifier during training, which neglects the test-time early exiting behavior, leading to inconsistency between training and testing. Although some methods have tackled this issue under a fixed speed-up ratio, the challenge of flexibly adjusting the speedup ratio while maintaining consistency between training and testing is still under-explored. To bridge the gap, we propose a novel Consistency-Oriented Signal-based Early Exiting (COSEE) framework, which leverages a calibrated sample weighting mechanism to enable each classifier to emphasize the samples that are more likely to exit at that classifier under various acceleration scenarios. Extensive experiments on the GLUE benchmark demonstrate the effectiveness of our COSEE across multiple exiting signals and backbones, yielding a better trade-off between performance and efficiency.

# Code — https://github.com/He-Jianing/COSEE Extended version — https://arxiv.org/abs/2412.13236

# 1 Introduction

Although tremendous improvements have been achieved by pre-trained language models (PLMs) in natural language processing tasks (Devlin et al. 2019; Lan et al. 2020; Radford et al. 2019; Liu et al. 2019), high computational costs of PLMs in both training and inference still hinder their deployment in resource-constrained devices and real-time scenarios. Besides, overthinking problem (Kaya, Hong, and Dumitras 2019) also restricts the application of PLMs. Precisely, for easy samples, PLMs can generate correct predictions according to the representations indicated by shallow layers. However, high-level representations may focus on more intricate or unrelated details, leading to incorrect answers.

To address these issues, early exiting (Xin et al. 2020; Zhou et al. 2020; Xin et al. 2021; Liao et al. 2021; Sun et al. 2022; Zeng et al. 2024), a kind of adaptive inference strategy, has been proposed to accelerate the inference of

PLMs. As illustrated in Figure 2, each intermediate layer of the PLM is coupled with an internal classifier to give an early prediction. This enables the early exiting of samples once the early predictions are sufficiently reliable, eliminating the need for passing them through the entire model. This method employs a sample-wise inference strategy to deal with easy samples with shallow classifiers and process hard samples with deeper classifiers, significantly improving inference efficiency without sacrificing accuracy and alleviating the overthinking problem.

Signal-based early exiting methods (Xin et al. 2020; Liu et al. 2020; Zhou et al. 2020; Schwartz et al. 2020; Li et al. 2021; Liao et al. 2021; Ji et al. 2023; Zhu 2021; Zhu et al. 2021; Gao et al. 2023; Zhang et al. 2023; He et al. 2024; Akbari, Banitalebi-Dehkordi, and Zhang 2022; Xin et al. 2021; Balagansky and Gavrilov 2022) are typical implementations of early exiting, which rely on carefully designed exiting signals (e.g. entropy, energy score, softmax score, and patience) to dynamically adjust the number of executed layers for each sample. The inference process is terminated once the exiting signal meets a certain condition. These methods can easily adapt to various acceleration requirements during inference by simply adjusting the threshold, without incurring additional training costs. However, existing works simply use the (weighted) sum of cross-entropy losses from all classifiers as the training objective, where each classifier treats the loss of both easy and hard samples equally. This treatment ignores the dynamic early exiting behavior during inference (as shown in Figure 1), leading to a gap between training and testing.

To bridge the gap, router-based early exiting methods (Sun et al. 2022; Mangrulkar, MS, and Sembium 2022; Zeng et al. 2024) have been successively proposed. These methods employ a router (e.g. a hash function or a network) to determine the exiting layer of samples during both training and inference, and each sample only incurs a crossentropy loss at its exiting classifier, ensuring consistency between training and testing. However, router-based early exiting methods fail to meet various acceleration requirements during inference, as a router can only generate a fixed exiting strategy, leading to unadjustable speed-up ratios.

In this paper, we aim to bridge the gap between training and testing while enabling flexible adjustments of the speed-up ratio. To this end, building upon the signal-based early exiting framework, we propose to assign sample-wise weights on the cross-entropy loss of all classifiers, such that each classifier is encouraged to emphasize samples that are more likely to exit at that classifier. Unfortunately, samples exit at different classifiers under various acceleration scenarios, bringing extreme challenges to weight assignment.

To address the challenges, we propose a novel framework of Consistency-Oriented Signal-based Early Exiting (COSEE). Specifically, at each training step, we mimic the test-time early exiting process at multiple randomly selected thresholds to find where the samples tend to exit under different accelerations. Subsequently, we adopt a heuristic sample weighting mechanism (SWM) to assign weights on the cross-entropy loss of each sample across all classifiers, where each sample is emphasized by the classifiers near its exiting layer. Accordingly, we minimize the mean of cross-entropy losses across different thresholds to ensure the model’s generalization ability in various acceleration scenarios. In addition, we further devise an online signal calibration (OSC) objective to generate highly discriminative exiting signals for more reliable exiting decisions, thus encouraging more proper loss weights based on exiting layers.

Our method is simple yet effective. Extensive experiments on the GLUE benchmark demonstrate that our COSEE framework with energy score consistently outperforms the state-of-the-art methods across all tasks, yielding a better trade-off between performance and efficiency with faster convergence speed and negligible additional storage overhead. In addition, an in-depth analysis further confirms the generalization of the COSEE framework on different exiting signals and backbones. Our main contributions can be summarized as follows:

• We disclose that the performance bottleneck of current early exiting methods primarily stems from the challenge of ensuring consistency between training and testing while flexibly adjusting the speed-up ratios. • We propose a novel Consistency-Oriented Signal-based Early Exiting (COSEE) framework to bridge the gap, which incorporates a sample weighting mechanism (SWM) and an online signal calibration (OSC) objective. • Extensive experiments verify the effectiveness of our COSEE across multiple exiting signals and backbones.

# 2 Preliminaries

In this section, we provide the necessary background for signal-based early exiting. Related works are detailed in the extended version.

# 2.1 Problem Definition

Per Figure 2, given a BERT-style PLM with $M$ layers, we denote the hidden states at the $m$ th layer as $h ^ { ( m ) }$ . To enable early exiting during inference on a classification task involving $C$ classes, each intermediate layer is equipped with an internal classifier $F _ { m } , m \in \{ 1 , 2 , \cdots , M - 1 \}$ to produce an early prediction $p ^ { ( m ) } = F _ { m } ( h ^ { ( m ) } )$ , i.e., a probability distribution over the $C$ classes. Classifiers in different layers do not share parameters.

![](images/7389ee7d0b564f57e2b130f996aafc1dbdefb93d86708522199399491e5f8760.jpg)  
Figure 1: Exiting layer distribution on the QNLI development set with entropy-based exiting signal (Threshold $\mathbf { \tau } = \mathbf { \tau }$ 0.4). Neg and Pos denote negative and positive samples, respectively. Samples near the classification boundary (hard samples) tend to exit at deep classifiers, while samples far from the classification boundary (easy samples) typically exit at shallow classifiers.

# 2.2 Signal-based Early Exiting

For a given sample $x$ , the inference process is terminated once the exiting signal at the current layer meets a certain condition. For exiting signals that exhibit a positive correlation with sample difficulty (e.g. entropy and energy score), early exiting is triggered once the exiting signal falls below a predefined threshold. A higher threshold leads to a higher speed-up ratio and potentially some performance degradation. Conversely, for exiting signals negatively correlated with sample difficulty (e.g. patience and softmax score), the exiting condition is met when the exiting signal surpasses the threshold. A higher threshold leads to a lower speed-up ratio and performance improvements.

# 2.3 Conventional Training Methods

In current signal-based early exiting methods, a widely used training objective involves the (weighted) sum of crossentropy losses across all classifiers:

$$
L = \sum _ { m = 1 } ^ { M } w _ { m } L ^ { ( m ) } ,
$$

where $L ^ { ( m ) }$ denotes the cross-entropy loss of the $m$ th classifier and $w _ { m }$ denotes the corresponding loss weight. Under Eq.(1), each classifier treats the loss of both easy and hard samples equally, which is inconsistent with the dynamic early exiting behavior during inference.

# 3 The COSEE Framework

# 3.1 Framework Overview

We propose a novel Consistency-Oriented Signal-based Early Exiting (COSEE) framework for PLMs, aiming to ensure consistency between training and testing while maintaining flexible adjustments of the speed-up ratio. Figure 2 provides an overview of our framework. We first propose a sample weighting mechanism (SWM) that identifies the potential exiting layer of samples by simulating the test-time early exiting process during training and then uses this information to produce sample-wise loss weights across all classifiers. Additionally, we further devise an online signal calibration (OSC) objective to encourage highly discriminative exiting signals for more reliable exiting decisions, thus ensuring more proper loss weights based on exiting layers. Finally, regarding the exiting signal, we introduce a normalized energy score to align energy distributions across different layers for easy threshold selection. We primarily use it to implement the COSEE framework.

![](images/ef3151679911c79531b4bac4b8c6db5dcfe98fb8fccf3cbe49b63f08c07579da.jpg)  
Figure 2: Comparison between the conventional signal-based early exiting framework and our COSEE. The conventional framework simply minimizes the (weighted) sum of cross-entropy losses from all classifiers, where each classifier treats all samples equally during training. Instead, our COSEE enables each classifier to emphasize samples that are more likely to exit at that classifier, ensuring consistency between training and testing. We also incorporate an online signal calibration objective LossOSC for each internal classifier to encourage highly discriminative exiting signals for more reliable exiting decisions and loss weights.

# 3.2 Sample Weighting Mechanism

Our goal is to identify the potential exiting layer of samples in various acceleration scenarios, and then assign greater weights to the cross-entropy loss of each sample on classifiers closer to its exiting layer. Accordingly, at each training step, all samples are passed through the entire model to generate predictions and exiting signals at all classifiers. Subsequently, we randomly select $K$ thresholds and simulate the early exiting process based on exiting signals at each threshold to find where the samples exit. This information is used to produce sample-wise loss weights across all classifiers.

Range for Threshold Selection. For threshold selection, we collect the maximum and minimum values of exiting signals across all layers for training samples within each epoch and use them to create the selection range for the next epoch. We start with the thresholds randomly selected between 0 and 1 in the first epoch.

Weight Assignment. For a given threshold $\tau$ , we impose sample-wise loss weights across all classifiers based on the exiting layer of samples and then compute the classification loss at threshold $\tau$ :

$$
\begin{array} { c } { { \displaystyle { \cal L } _ { \mathrm { C E } , \tau } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \sum _ { m = 1 } ^ { M } w _ { n } ^ { ( m ) } \cdot \mathrm { { C E } } ( \hat { y } _ { n } ^ { ( m ) } , y _ { n } ) , } } \\ { { w _ { n } ^ { ( m ) } = \frac { e ^ { - \beta _ { t } \cdot | m - m _ { n } ^ { * } | } } { \sum _ { m = 1 } ^ { M } e ^ { - \beta _ { t } \cdot | m - m _ { n } ^ { * } | } } , } } \end{array}
$$

where $\mathrm { C E } ( \hat { y } _ { n } ^ { ( m ) } , y _ { n } )$ and $w _ { n } ^ { ( m ) }$ denote the cross-entropy loss and the loss weight for the nth sample at the mth classifier respectively, and w(nm) satisfies $\begin{array} { r } { \sum _ { m = 1 } ^ { M } w _ { n } ^ { ( m ) } = 1 . ~ N } \end{array}$ denotes the number of samples. $m _ { n } ^ { * }$ denotes the index of exiting layer for the nth sample at threshold $\tau$ , and $\beta _ { t }$ denotes the decay factor at the tth training step. According to Equation 3, classifiers closer to the exiting layer are assigned greater weights compared to those further away, i.e., each sample is emphasized by the classifiers near its exiting layer. Note that the loss weights of classifiers are symmetrical around the exiting layer for easy parameter selection. Different from router-based early exiting methods, which employ one-hot sample-wise loss weights such that each sample only incurs a cross-entropy loss on its exiting classifier, we employ a softer sample weighting mechanism to enable the generality of our COSEE on unseen thresholds.

During the early training stage, unstable exiting layers often lead to fluctuating loss weights, consequently impacting the model’s convergence. To mitigate this problem, we conduct a warm-up operation for the decay factor $\beta _ { t }$ to gradually increase the impact of the sample’s exiting layers on the loss weights during training:

$$
\beta _ { t } = \gamma _ { t } \cdot \beta _ { 0 } ,
$$

where $\beta _ { 0 }$ is positive, and $\gamma _ { t }$ is the ratio of the current training step to the total training steps.

Classification Objective. To enable various acceleration ratios during inference, the classification objective is defined as the mean of classification losses across all $K$ thresholds:

$$
L _ { \mathrm { C E } } = \frac { 1 } { K } \sum _ { \tau } L _ { \mathrm { C E } , \tau } .
$$

# 3.3 Online Signal Calibration

While SWM effectively facilitates the training of multi-exit networks, exiting signals may not consistently reflect sample difficulty, particularly during the early training stages. This affects the reliability of exiting decisions, leading to

0.5 Layer-2 0.8 Layer-2   
0.4 Layer-6 Layer-10 0.6 GR 20.4 Layer-10 Layer-6 0.1 F0.2 0.0 0.0 -4 -2 0 0.0 0.2 0.4 Exiting Signals Exiting Signals (a) Original Energy Score (b) Normalized Energy Score

sub-optimal loss weights based on exiting layers. Therefore, we introduce an online signal calibration (OSC) objective to explicitly enlarge the distribution divergence of exiting signals between easy and hard samples. Specifically, for exiting signals that indicate the sample difficulty (e.g. entropy and energy score), our OSC objective is formulated as:

$$
L _ { \mathrm { O S C } } = \frac { 1 } { M - 1 } \sum _ { m = 1 } ^ { M - 1 } L _ { \mathrm { O S C } } ^ { ( m ) } ,
$$

$$
L _ { \mathrm { O S C } } ^ { ( m ) } = \operatorname* { m a x } ( 0 , \overline { { S } } _ { e a s y } ^ { ( m ) } - \overline { { S } } _ { h a r d } ^ { ( m ) } + \epsilon ) ,
$$

where $L _ { \mathrm { O S C } } ^ { ( m ) }$ is the signal calibration loss at the $m$ th layer. $\overline { { S } } _ { e a s y } ^ { ( m ) }$ and $\overline { { S } } _ { h a r d } ^ { ( m ) }$ are the mean of exiting signals on easy and hard samples at the mth layer, respectively, and $\epsilon$ is the margin parameter shared across layers. For exiting signals negatively correlated with sample difficulty (e.g. softmax score), the calculation for $L _ { \mathrm { O S C } } ^ { ( m ) }$ in Eq.(7) needs to be replaced with:

$$
L _ { \mathrm { O S C } } ^ { ( m ) } = \operatorname* { m a x } ( 0 , \overline { { S } } _ { h a r d } ^ { ( m ) } - \overline { { S } } _ { e a s y } ^ { ( m ) } + \epsilon ) .
$$

Note that we only minimize the signal calibration loss for the first $M - 1$ layers, since there is no need to exit at the last layer. Additionally, we define samples as easy or hard depending on whether the internal classifier can predict them correctly, thus the partition may differ across layers.

# 3.4 Training Objective

The training objective of the COSEE is formulated as the weighted sum of the classification and OSC objective:

$$
\begin{array} { r } { L = L _ { \mathrm { C E } } + \alpha \times L _ { \mathrm { O S C } } , } \end{array}
$$

where $\alpha$ is a hyper-parameter used to balance the classification and OSC objectives. All internal classifiers are jointly trained with the backbone.

# 3.5 Exiting Signal

Following E-LANG (Akbari, Banitalebi-Dehkordi, and Zhang 2022), we primarily implement our COSEE with the energy-based exiting signal. The energy score is defined as:

$$
E ( x ; F _ { m } ) = - \log \sum _ { i = 1 } ^ { C } e ^ { f _ { i } ^ { ( m ) } } ,
$$

where $C$ is the number of classes, and $f _ { i } ^ { ( m ) }$ denotes the logit value of sample $x$ on class $\mathbf { \chi } _ { i }$ suggested by the mth internal classifier $F _ { m }$ . A lower energy score indicates lower sample difficulty. The exiting criterion is met when the energy score falls below a predefined threshold. To align the energy distribution across different layers for threshold selection, we normalize the original energy scores to $( 0 , 1 )$ :

$$
E _ { n o r m } ( x ; F _ { m } ) = \big ( 1 + e ^ { - E ( x ; F _ { m } ) } \big ) ^ { - 1 } .
$$

Figure 3 confirms the superiority of the normalized energy score over the original energy score. In this paper, we mainly conduct experiments with the normalized energy score. Nevertheless, we also verify the effectiveness of the COSEE framework on other exiting signals, i.e., entropy and softmax score (see Section 5.2).

# 4 Experiments

# 4.1 Tasks and Datasets

Following Li et al. (2021); Liao et al. (2021), we evaluate COSEE on six classification tasks from the GLUE benchmark (Wang et al. 2019), including SST-2, MRPC, QNLI, RTE, QQP, and MNLI. Data statistics are shown in Table 1.

# 4.2 Baselines

We compare our COSEE model with three groups of representative and state-of-the-art baselines.

Backbone. We adopt the widely used BERT-base (Devlin et al. 2019) as the backbone for convincing comparisons.

Budget Exiting. We directly train a BERT-base with 6 layers (BERT-6L) to obtain a speed-up ratio of $2 . 0 0 \times$ , establishing a lower bound for early exiting methods as no techniques are employed.

Early Exiting. For signal-based early exiting methods, we choose DeeBERT (Xin et al. 2020), PABEE (Zhou et al. 2020), BERxiT (Xin et al. 2021), LeeBERT (Zhu 2021), GPFEE (Liao et al. 2021), GAML-BERT (Zhu et al. 2021), PALBERT (Balagansky and Gavrilov 2022), and DisentangledEE (Ji et al. 2023). For router-based early exiting methods, we choose state-of-the-art ConsistentEE (Zeng et al. 2024). Notably, some early exiting methods (Sun et al. 2022; Mangrulkar, MS, and Sembium 2022; Zhang et al. 2023; Zhu et al. 2023) are not included due to the difference in backbones. CascadeBERT (Li et al. 2021) and ELANG (Akbari, Banitalebi-Dehkordi, and Zhang 2022) are excluded for fair comparisons since they implement early exiting within several complete networks instead of a multiexit network. Refer to the extended version for more details.

# 4.3 Experimental Settings

Measurement. Since the runtime is unstable across different runs, following Zhang et al. (2022) and Liao et al. (2021), we utilize the saved layers to measure the speed-up ratio:

$$
{ \mathrm { S p e e d - u p R a t i o } } = { \frac { \sum _ { m = 1 } ^ { M } M \times N ^ { m } } { \sum _ { m = 1 } ^ { M } m \times N ^ { m } } } ,
$$

where $M$ is the total number of layers and $N ^ { m }$ is the number of samples exiting from the $m$ th layer. According to Xin et al. (2020), this metric is proportional to actual runtime.

Table 1: Dataset Statistics. NLI is the Natural Language Inference task, and QA is the Question Answering task.   

<html><body><table><tr><td>Dataset</td><td>Classes</td><td>|Train|</td><td>|Test|</td><td>Task</td></tr><tr><td>SST-2</td><td>2</td><td>67k</td><td>1.8k</td><td>Sentiment</td></tr><tr><td>MRPC</td><td>2</td><td>3.7k</td><td>1.7k</td><td>Paraphrase</td></tr><tr><td>QQP</td><td>2</td><td>364k</td><td>391k</td><td>Paraphrase</td></tr><tr><td>MNLI</td><td>3</td><td>393k</td><td>20k</td><td>NLI</td></tr><tr><td>QNLI</td><td>2</td><td>105k</td><td>5.4k</td><td>QA/NLI</td></tr><tr><td>RTE</td><td>2</td><td>2.5k</td><td>3k</td><td>NLI</td></tr></table></body></html>

![](images/deab00f4f1a0b1cadbd0ccb5afa0611f316e3b742711483cd87cfc963eac8ad9.jpg)  
Figure 4: Impact of SWM and OSC on the trade-off between performance and efficiency for COSEE with energy.   
Figure 5: DIS heatmap of different models for layers 2, 6, and 10 on the SST-2 and QNLI development sets.

Training. Our implementation is based on Hugging Face’s Transformers (Wolf et al. 2020). Each internal classifier consists of a single linear layer. We mainly implement our COSEE framework with the normalized energy score if not specified. We also conduct experiments with entropy and softmax scores for generalization analysis. Following the previous work (Zhou et al. 2020; Zhang et al. 2022; Liao et al. 2021), we perform a grid search over learning rates of $\{ 1 \mathrm { e } { - } 5 , 2 \mathrm { e } { - } 5 , 3 \bar { \mathrm { e } } { - } 5 , 5 \mathrm { e } { - } 5 \}$ , batch sizes of $\{ 1 6 , 3 2 , 1 \bar { 2 } 8 \}$ , $\alpha$ values in Eq.(9) of $\{ 0 . 0 0 1 , 0 . 0 1 , 0 . 1 , 1 . 0 \}$ , and $\beta _ { 0 }$ values in Eq.(4) of $\{ 0 . 0 5 , 0 . 2 , 1 . 0 , 1 0 . 0 \}$ . We set $\epsilon$ to 0.3 in Eq.(7) and $K$ to 5 in Eq.(5). The maximum sequence length is fixed at 128. We employ a linear decay learning rate scheduler and the AdamW optimizer (Loshchilov and Hutter 2019). We conduct experiments on two RTX4090 GPUs with 24GB.

Inference. Following previous work (Zhang et al. 2022; Liao et al. 2021), we adopt a batch size of 1 for inference, emulating a typical industry scenario where requests from various users arrive one by one. For fair comparisons, we carefully adjust the threshold $\tau$ for each task to achieve a similar speed-up ratio as the baseline methods (approximately $2 . 0 0 \times \mathrm { \ i }$ ) and further compare the trade-off between task performance and inference efficiency.

S S 2 C么 71.5 77.885.7 85 N C么4 61.677.179.3 80 W O 74.6 82.2 86.7 -80 z 66.580.5 82.5 -70 77.3 84.0 87.0 -75 S 68.3 81.983.3 2 6 10 2 6 10 (a) SST-2 (b) QNLI

# 4.4 Overall Performance Comparison

Table 2 reports the test results of each early exiting method on the GLUE benchmark with BERT-base as the backbone model. The speed-up ratio is approximately $2 . 0 0 \times ( \pm 3 8 \% )$ . Overall, our COSEE framework with normalized energy score demonstrates a superior performance-efficiency tradeoff across different tasks compared to the baseline methods, which verifies the effectiveness of our design. Notably, our COSEE can even outperform the original BERT-base on RTE and QQP tasks, indicating that our method can effectively alleviate the overthinking problem of PLMs. This suggests that, for easy samples, predictions from intermediate layers may outperform those from the final layer. Our method enables easy samples to exit at shallow classifiers, thereby reducing the inference time while maintaining or even improving the task performance. Besides, our method can save training costs (see Section 4.5) and introduce negligible additional storage overhead (see Section 5.3). We also explore the impact of hyperparameters and statistically analyze the failure cases in the extended version.

Although using energy scores and BERT-base in the primary experiments, we also verify the generality of COSEE on various exiting signals and backbones (see Section 5.2).

# 4.5 Ablation Studies

Performance-Efficiency Trade-Off. To investigate the effectiveness of SWM and OSC, we plot the performanceefficiency trade-off curves of models trained using different methods on four GLUE development sets, as shown in Figure 4. We can observe both SWM and OSC significantly improve the performance of early exiting across all tasks, especially under high speed-up ratios. This confirms the advantage of our COSEE under high acceleration scenarios, indicating the proposed SWM and OSC effectively facilitate the training of internal classifiers, particularly shallow ones.

Evaluation of Exiting Signals. Difficulty Inversion Score (DIS) is first proposed by Li et al. (2021), an evaluation metric for exiting signals. A higher value indicates a greater correlation between the exiting signal and sample difficulty, thus enabling more reliable exiting decisions. Figure 5 illustrates the DIS of exiting signals generated by different models. The results indicate that OSC explicitly enhances the correlation between the exiting signal and sample difficulty by enlarging the distribution divergence of exiting signals

Table 2: Performance comparison on the GLUE test set with BERT-base as the backbone. $\dagger$ denotes the results taken from GPFEE (Liao et al. 2021), and $\ddag$ denotes the results taken from DisentangledEE (Ji et al. 2023). Other baseline results are from their original papers. Our COSEE uses the normalized energy score as the exiting signal. Best results are marked in bold.   

<html><body><table><tr><td>Method</td><td>RTE</td><td> MRPCc/Mean</td><td>F1/Acc/Mean</td><td>SST-2</td><td>QNLI</td><td>MNLI</td></tr><tr><td>BERT-base</td><td>66.4 (1.00×)</td><td>88.9/-/- (1.00x)</td><td>71.2/-/- (1.00x)</td><td>93.5 (1.00×)</td><td>90.5 (1.00×)</td><td>84.6 (1.00×)</td></tr><tr><td>BERT-6L†</td><td>63.9 (2.00×)</td><td>85.1/78.6/81.9 (2.00×)</td><td>69.7/88.3/79.0 (2.00×)</td><td>91.0 (2.00×)</td><td>86.7 (2.00×)</td><td>80.8 (2.00×)</td></tr><tr><td>DeeBERTt</td><td>64.3 (1.95×)</td><td>84.4/77.4/80.9 (2.07×)</td><td>70.4/88.8/79.6 (2.13×)</td><td>90.2 (2.00×)</td><td>85.6 (2.09×)</td><td>74.4 (1.87×)</td></tr><tr><td>PABEE†</td><td>64.0 (1.81x)</td><td>84.4/77.4/80.9 (2.01×)</td><td>70.4/88.6/79.5 (2.09×)</td><td>89.3 (1.95×)</td><td>88.0 (1.87×)</td><td>79.8 (2.07×)</td></tr><tr><td>BERxiT</td><td>65.7 (2.17×)</td><td>86.2/-/- (2.27×)</td><td>70.5/-/- (2.27x)</td><td>91.6 (2.86×)</td><td>89.6 (1.72×)</td><td>82.1 (2.33×)</td></tr><tr><td>LeeBERT</td><td>■</td><td>87.1/-l- (1.97x)</td><td></td><td>92.6 (1.97x)</td><td>1</td><td>83.1 (1.97x)</td></tr><tr><td>GPFEE</td><td>64.5 (2.04×)</td><td>87.0/81.8/84.4(1.98×)</td><td>71.2/89.4/80.3 (2.18×)</td><td>92.8 (2.02×)</td><td>89.8 (1.97×)</td><td>83.3 (1.96×)</td></tr><tr><td>GAML-BERT</td><td>64.3 (1.96x)</td><td>87.2/-/- (1.96×)</td><td>70.9/-1- (1.96x)</td><td>92.8 (1.96×)</td><td>84.2 (1.96×)</td><td>83.3 (1.96×)</td></tr><tr><td>PALBERT‡</td><td>64.3 (1.48×)</td><td>-/-/80.7 (1.48×)</td><td>-/-/79.3 (1.48×)</td><td>91.8 (1.48×)</td><td>89.1 (1.48×)</td><td>83.0 (1.48×)</td></tr><tr><td>DisentangledEE</td><td>66.8 (1.25×)</td><td>-/-/83.8 (1.25×)</td><td>-1-/79.4 (1.25x)</td><td>92.9 (1.25x)</td><td>88.5 (1.25×)</td><td>83.0 (1.25×)</td></tr><tr><td>ConsistentEE</td><td>69.0 (1.85×)</td><td>89.0/-/- (1.59 ×)</td><td>-/89.0/- (1.82x)</td><td>92.9 (1.85x)</td><td>89.9 (1.72×)</td><td>83.4 (1.45×)</td></tr><tr><td>COSEE (ours)</td><td>68.7 (1.96x)</td><td>88.0/82.0/85.0 (2.70×)</td><td>71.4/89.4/80.4 (2.01×)</td><td>93.0 (2.14×)</td><td>90.2 (2.56×)</td><td>83.4 (1.92×)</td></tr></table></body></html>

0.6 COSEE-energy 0.6 COSEE-energy w/o OSC w/o OSC 0.4 -w/o OSC w/o SWM . w/o OSC w/o SWM 0.2 0 500 1000 1500 0 1000 2000 Iterations Iterations (a) SST-2 (b) QNLI

between easy and hard samples. Meanwhile, SWM encourages highly discriminative exiting signals by enabling each classifier to emphasize a subset of samples with certain difficulty levels. Also, it is noticeable that the improvements brought by SWM and OSC appear to be significant on shallow layers, which aligns with the observation shown in Figure 4. This is due to the constrained capability of shallow classifiers, which allows for greater potential for improvements in training than deep classifiers.

Training Curves. To further explore the convergence speed of our COSEE during training, we plot the model’s training curves across different training methods on SST-2 and QNLI tasks, as shown in Figure 6. The results indicate that the proposed SWM effectively accelerates the model’s convergence during training. We attribute this to SWM’s ability to reduce data complexity during training by enabling each classifier to emphasize a different subset of samples. Additionally, we can observe that incorporating the OSC objective can slightly impact the model’s convergence speed. Nevertheless, our COSEE framework still maintains an advantage over the vanilla training method.

# 5 In-depth Analysis

# 5.1 Visualization of Sample Exiting Layers

To examine the consistency between training and testing under our COSEE framework, we visualize the exiting layer distribution in training and development sets at various thresholds, respectively. Figure 7 shows the visualization results for the SST-2 task. At different thresholds, training and development sets demonstrate consistent exiting layer distributions, which verifies the interpretability of our design. Additionally, we can observe that samples near the classi

![](images/ba8c0b3f97ac70c2e0fd7b4028d2cee5a56f97c790bac1ae3562628cb97a9a8f.jpg)  
Figure 6: Impact of SWM and OSC on training convergence for the SST-2 and QNLI tasks.   
Figure 7: Exiting layer distribution on the training and development sets of SST-2 task under different thresholds $\tau$ . Neg and Pos denote negative and positive samples, respectively. The results exhibit consistency between training and development sets across different thresholds.

0.925 0.9   
50.00 00.8 S 0.875 0.850 2345 6 0.7 2345 6 Speed-up Ratio Speed-up Ratio (a) SST-2 (b) QNLI 0.85 0.90   
0.80 .86 COSEE-energy   
\$0.75 COSEE-entropy - w/o OSC 0.84 -w/oOSCw/oSWM 0.70 23 4 5 6 1 2 3 4 5 6 Speed-up Ratio Speed-up Ratio (c) MNLI (d) QQP 0.925 0.9   
e0.900 0 00.8   
S s 0.875 0.850 2345 6 0.7 2345 6 Speed-up Ratio Speed-up Ratio (a) SST-2 (b) QNLI 0.85 0.90 0.88   
\$0.75 .86OE-x - w/o OSC 0.84 w/o OSCw/o SWM 0.70 23 4 5 6 1 2 3 4 5 6 Speed-up Ratio Speed-up Ratio (c) MNLI (d) QQP

fication boundary (hard samples) tend to exit at deep classifiers while samples far from the classification boundary (easy samples) tend to exit at shallow classifiers. Furthermore, increasing the threshold will cause a decrease in exiting layers, thus achieving a higher speed-up ratio. These observations align with our intuitive understanding.

# 5.2 Generality of the COSEE Framework

In this subsection, we explore the generality of our method on various exiting signals and backbones.

Figure 8 and Figure 9 present the experimental results of our COSEE framework with entropy and softmax scores, respectively. The results demonstrate the generality of our framework across various exiting signals. Notably, COSEE with energy outperforms COSEE with entropy or softmax scores on most tasks (SST-2, QNLI, and QQP). We attribute this to the superiority of energy scores in distinguishing easy and hard samples compared to entropy and softmax scores, as theoretically demonstrated by Akbari, BanitalebiDehkordi, and Zhang (2022). Therefore, we primarily implement COSEE with energy scores in this paper.

Table 3: Test results of different early exiting methods with ALBERT-base as the backbone. The speed-up ratio is averaged across 4 tasks. We report the mean of accuracy and F1-score for QQP, and accuracy for other tasks. $\dagger$ denotes results taken from GPFEE (Liao et al. 2021). Other baseline results are taken from DisentangledEE (Ji et al. 2023).   

<html><body><table><tr><td>Method</td><td>Speed-up</td><td>QQP</td><td>SST-2</td><td>QNLI</td><td>MNLI</td><td>AVG</td></tr><tr><td>ALBERT-base†</td><td>1.00×</td><td>79.6</td><td>93.3</td><td>92.0</td><td>85.2</td><td>87.5</td></tr><tr><td>PABEE†</td><td>1.95×</td><td>79.8</td><td>92.4</td><td>90.9</td><td>84.2</td><td>86.8</td></tr><tr><td>PALBERT</td><td>1.21×</td><td>79.1</td><td>91.4</td><td>90.9</td><td>83.2</td><td>86.2</td></tr><tr><td>DisentangledEE</td><td>1.26×</td><td>79.3</td><td>92.2</td><td>91.0</td><td>83.5</td><td>86.5</td></tr><tr><td>COSEE-energy</td><td>2.12×</td><td>79.6</td><td>92.9</td><td>91.8</td><td>84.8</td><td>87.3</td></tr></table></body></html>

Table 4: Parameter volume comparison. $C$ is the number of classes. COSEE introduces negligible extra parameters.   

<html><body><table><tr><td>Model</td><td colspan="2">#Params</td></tr><tr><td></td><td>C=2</td><td>C=3</td></tr><tr><td>BERT-base</td><td>109.48M</td><td>109.48M</td></tr><tr><td>COSEE</td><td>+16.92K</td><td>+25.38K</td></tr></table></body></html>

Table 3 presents the performance comparison with backbone ALBERT-base. We observe that our COSEE with energy outperforms competitive baseline methods on most tasks, demonstrating its generality across different PLMs.

# 5.3 Storage Costs Analysis

Table 4 compares the parameter volume of our COSEE model with that of the original BERT-base. Our COSEE model introduces less than $\bar { 0 . 0 3 \% }$ additional parameters due to incorporating internal classifiers. Notably, the proposed SWM is parameter-free, yet it effectively generates appropriate loss weights for each sample to enhance the training of multi-exit networks.

# 6 Conclusion

In this paper, we point out that the performance bottleneck of existing early exiting methods primarily lies in the challenge of ensuring consistency between training and testing while enabling flexible adjustments of the speed-up ratio. To remedy this, we propose COSEE, which mimics the testtime early exiting process under various acceleration scenarios based on calibrated exiting signals and then produces the sample-wise loss weights at all classifiers according to the sample’s exiting layer. Our framework is both simple and intuitive. Extensive experiments on the GLUE benchmark demonstrate the superiority and generality of our framework across various exiting signals and backbones.