# VERO: Verification and Zero-Shot Feedback Acquisition for Few-Shot Multimodal Aspect-Level Sentiment Classification

Kai Sun,1,2 Hao Wu,3 Bin Shi, $^ { 1 , 2 * }$ Samuel Mensah,4 Peng Liu,3∗ Bo Dong1,5

1Shaanxi Provincial Key Laboratory of Big Data Knowledge Engineering, Xi’an Jiaotong University, China 2School of Computer Science and Technology, Xi’an Jiaotong University, China 3School of computer science and engineering, Guangxi Normal University, Guilin, China 4Department of Computer Science, University of Sheffield, Sheffield, United Kingdom 5School of Continuing Education, Xi’an Jiaotong University, China sunkai $@$ xjtu.edu.cn, wuhao $@$ stu.gxnu.edu.cn, shibin $@$ xjtu.edu.cn, s.mensah@sheffield.ac.uk, liupeng@gxnu.edu.cn, dong.bo $@$ xjtu.edu.cn

# Abstract

Deep learning approaches for multimodal aspect-level sentiment classification (MALSC) often require extensive data, which is costly and time-consuming to obtain. To mitigate this, current methods typically fine-tune small-scale pretrained models like BERT and BART with few-shot examples. While these models have shown success, Large VisionLanguage Models (LVLMs) offer significant advantages due to their greater capacity and ability to understand nuanced language in both zero-shot and few-shot settings. However, there is limited work on fine-tuning LVLMs for MALSC. A major challenge lies in selecting few-shot examples that effectively capture the underlying patterns in data for these LVLMs. To bridge this research gap, we propose an acquisition function designed to select challenging samples for the few-shot learning of LVLMs for MALSC. We compare our approach, Verification and ZERO-shot feedback acquisition (VERO), with diverse acquisition functions for few-shot learning in MALSC. Our experiments show that VERO outperforms prior methods, achieving an F1 score improvement of up to $6 . 0 7 \%$ on MALSC benchmark datasets.

Code — https://github.com/absdog/vero

![](images/69aa0c9e4dd435c0202d78362fa8a03007779dbe6ac7c2cda50f8594bd71b9d1.jpg)  
Figure 1: Illustrative example of VERO for LVLMs. The solid line represents the LVLM’s self-verification decision boundary, separating samples into “changed” (black circles) and “confirmed” (gray circles) classifications based on the model’s zero-shot and self-verification predictions on unlabeled data. The distance between circles and the boundary reflects the uncertainty level after self-verification. We focus on selecting samples within the low-uncertainty region of the “changed” class (highlighted in blue), as these are challenging and informative for fine-tuning the LVLM.

# Introduction

With the growing abundance of multimodal data shared on social media, multimodal aspect-level sentiment classification (MALSC) has gained more attention. MALSC aims to determine the sentiment polarity of an aspect based on the given text-image pair. Previous approaches to MALSC focused on enhancing model performance by leveraging large amounts of training data or incorporating additional auxiliary data (Khan and $\mathrm { F u } 2 0 2 1$ ; Ling, Yu, and Xia 2022; Yang, Xiao, and Du 2024). However, the collection and annotation of multimodal data for MALSC are inherently timeconsuming and labor-intensive. To address this challenge, many studies have focused on Few-shot MALSC (Yu and Zhang 2022; Yu, Zhang, and Li 2022; Yang et al. 2023b), aiming is to identify and select a few highly informative samples from a large pool of unlabeled data to optimize the model performance on the task.

Recent advancements in Few-shot MALSC have primarily focused on fine-tuning small-scale pretrained language models (PLMs), such as BERT (Devlin et al. 2019) and BART (Lewis et al. 2020). Meanwhile, Large VisionLanguage Models (LVLMs) demonstrate remarkable performance on the task using different prompt-based techniques, including zero-shot (Yang et al. 2024b), few-shot (Li et al. 2023; Ye et al. 2023), chain-of-thought (CoT) (Li et al. 2024) and most recently self-verification (Gero et al. 2023) - a better reasoning of CoT, where the LVLM performs a forward reasoning and backward verification to arrive at an answer. Then again, adapting LVLMs to unseen data through fine-tuning is crucial, yet this process on large-scale data is costly. Few-shot fine-tuning offers a promising alternative, allowing these models to quickly adapt to new domainspecific data while minimizing the resources and time typically required. However, limited studies have focused on few-shot fine-tuning of LVLMs for MALSC (Yang et al.

2024a).

To acquire samples to finetune LVLMs for the task, existing methods (Yu and Zhang 2022) rely on random sampling from a specific prior distribution (e.g., uniform or the distribution of the training dataset). The current state-of-theart (SOTA) approach, MultiPoint (Yang et al. 2023b) proposed consistently distributed sampling (CDS), where the acquisition function ensured the consistency of the sentiment distribution between the full training dataset and the sampled dataset. Though these sampling methods (or acquisition functions) can be directly applied to LVLMs, they may not guarantee that the selected samples are informative to the LVLMs. This is because these techniques do not consider the capability of the LVLM itself. Considering that LVLMs already possess some zero-shot capabilities for MALSC, samples selected by these prior methods (Yu and Zhang 2022; Yang et al. 2023b) may lead to limited training benefits.

To this end, we propose an acquisition function that searches the pool of unlabelled data for challenging examples. Specifically, our method VERO, selects unlabeled data in the pool where there is a divergence between the LVLM’s zero-shot predictions and subsequent self-verification predictions. Intuitively, the most informative samples are those that prompt the model to alter its classification during selfverification after an initial zero-shot prediction. Such samples likely contain complex features that cause the model to oscillate between different classes. As illustrated in Figure 1, we focus on selecting samples within the low-uncertainty region, shown in blue, because they present cases where selfverification reveals a confident shift in prediction. The fact that the LVLM becomes firmly convinced of a new class during re-evaluation indicates that these samples are challenging and crucial for refining the model’s understanding and enhancing its overall performance.

We validated our approach using the LVLM-based models, LLaVA-7b and LLaVA-13b on the Twitter-2015 and Twitter-2017 datasets, demonstrating that our method outperforms previous techniques in few-shot learning scenarios, with an F1 score improvement of up to $6 . 0 7 \%$ . Our analyses demonstrate that the samples selected by our method challenge LVLMs more effectively than current methods (Yu and Zhang 2022; Yu, Zhang, and Li 2022; Yang et al. 2023b), leading to significant performance gains.

Our contributions are the following:

# Related Work Multimodal Aspect-Level Sentiment Classification

• To the best of our knowledge, we are the first to successfully fine-tune a Large Vision Language Model to solve the few-shot MALSC task. • We propose VERO, a novel acquisition function that acquires challenging samples from a pool of unlabeled data for LVLM fine-tuning. VERO achieves this by leveraging the zero-shot and self-verification capability of LVLMs. • We conduct extensive experiments on two benchmark datasets across $1 \%$ and $7 \%$ few-shot settings, demonstrating the remarkable superiority of our approach.

With the proliferation of multimodal data disseminated on social media, multimodal aspect-level sentiment classification (MALSC) began to receive increasing attention. Research in MALSC could be mainly categorized into three research lines: cross-modal attention methods, image translation methods and small-scale PLMs. Approaches based on cross-modal attention focus on utilizing attention mechanisms to implicitly align and fuse the semantic information and emotion information in the two modalities (Xu, Mao, and Chen 2019; Yu and Jiang 2019; Xiao et al. 2023; Yang, Xiao, and Du 2024). In image translation methods (Khan and Fu 2021; Yang, Zhao, and Qin 2022a), images will be translated into textual descriptions, and then these descriptions are used as additional context for the input text. Small-scale PLMs aim at improving representation learning (Lu et al. 2019; Nguyen, Vu, and Nguyen 2020; Ling, Yu, and Xia 2022). However, these methods require a large amount of annotated data for model fine-tuning, which is time-consuming and labour intensive.

# Few-shot MALSC with PLMs

With the scaling of pre-trained language models (PLMs) from 110M parameters (Devlin et al. 2019) to over 500B parameters (Smith et al. 2022), the capabilities of these models have greatly improved. In recent studies, PLMs have been regarded as powerful tools for solving few-shot MALSC (Liu et al. 2023), which could be mainly categorized into two research lines: finetuning-based methods and in-context learning methods.

Finetuning-based Methods These models treat the classification task as a masked language modeling (MLM) task, where the model is fine-tuned with a set of prompts to guide its prediction by filling a special token, [MASK] (Gao, Fisch, and Chen 2021; Jian, Gao, and Vosoughi 2022; Hosseini-Asl, Liu, and Xiong 2022; Yu, Zhang, and Li 2022; Yu and Zhang 2022). Hosseini-Asl, Liu, and Xiong (2022) proposed a generative language model (GFSC) that reformulates the task as a language generation problem. Yang et al. (2023a) proposed a generative multimodal prompt (GMP) model for the joint multiple apsect-level sentiment analysis (JMALSA) task. Meanwhile, Yang et al. (2023b) proposed a unified multimodal prompt that allows for the joint processing of both text and image modalities in a coherent manner.

Despite the achievements of these methods, their limitations have become increasingly evident in the era of LVLMs. Firstly, they still persist in fine-tuning small-scale pretrained models, without recognizing the powerful image-text understanding capabilities of LVLMs (Yang et al. 2023a,b, 2024a). Secondly, it is important for few-shot MALSC to select a small set of samples that can maximize the fine-tuning benefits. These methods (Yu and Zhang 2022; Yu, Zhang, and Li 2022; Yang et al. 2023b) typically sample from training and development set randomly according to a specific distribution (e.g., uniform, the distribution of original training set), which does not guarantee that the samples are informative with respect to the capability of LVLMs.

In-context learning methods These approaches enhance the performance of LVLMs by incorporating demonstrations into prompts without the need for parameter updating (Li et al. 2023; Ye et al. 2023; Yang et al. 2024b). For instance, Yang et al. (2024b) explored the potential of using ChatGPT for In-Context Learning (ICL) on the MALSA task and enchanced the ICL framework’s performance in few-shot learning scenarios via an entity-aware contrastive learning method. Despite promising performances in extreme fewshot scenarios, the performance of the ICL method is often unstable, affected by factors such as instruction, the formatting of demonstrations, and the order of these examples.

# Active Learning

Active Learning (AL) (Settles 2009) focuses on selecting the most “informative” training samples to finetune a model. AL falls into three main categories: uncertainty, representativeness, and performance-based methods. Uncertainty-based methods target the most uncertain instances (Zhu et al. 2009; Yang et al. 2015; Raj and Bach 2022), using criteria like entropy and margin. Representativeness-based methods select samples best representing the input distribution (Huang, Jin, and Zhou 2014; Xie et al. 2022). Performance-based methods directly optimize informativeness via surrogates, considering the impact of revealing an instance’s label on future outcomes (Roy and McCallum 2001; Schein and Ungar 2007; Cai, Zhang, and Zhou 2013). Our method can be regarded as a an active learning approach, however it distinguishes itself from previous works by leveraging the inherent capability of the LVLM (e.g. zero-shot prediction, selfverification) for sample acquisition.

# Uncertainty Estimation

Uncertainty estimation in language models is significant for analyzing the potential erroneous behaviors of these models. To aggregate uncertainty information obtained at the token level, Manakul, Liusie, and Gales (2023) propose four different metrics, including the maximum or average likelihood and the maximum or average entropy. Additionally, the prediction uncertainty of large language models can also be estimated by sample-based methods (Huang et al. 2023) and perturbation-based methods (Meister et al. 2023; Huang et al. 2023).

# Methodology

We propose an acquisition function, called VERO, to select challenging samples from the pool of unlabeled data based on the LVLM’s zero-shot predictions and subsequent self-verification. In the following sections, we begin by formally defining the problem of few-shot MALSC. Next, we introduce the proposed VERO for sample acquisition. The overview of our approach is presented in Figure 2.

# Problem Statement

MALSC aims to classify the sentiment polarity $y$ of a specified aspect $a$ in a sentence-image pair, where $a$ is a marked phrase in $s$ and image $v$ assists classification. Typically, MALSC is treated as a three-class classification task where the polarity $y$ belongs to a predefined set of polarities, $y \ \in \ \{ \mathrm { N e g a t i v e , N e u t r a l , P o s i t i v e } \} .$ . In this paper, we focus on the few-shot MALSC task. Given a training dataset $\mathcal { D } _ { \mathrm { t r a i n } } = \{ ( s ^ { j } , a ^ { j } , v ^ { j } , y ^ { j } ) \} _ { j = 1 } ^ { N }$ , a subset of $K$ samples is selected for training, where $K \ll N$ . Specifically, the sentiment label $y ^ { j }$ are not available during selection, i.e., we optimize model performance while saving annotation costs.

# Verification and Zero-Shot Feedback Acquisition

Firstly, two multimodal prompts $\mathcal { P } _ { \mathrm { S C } }$ and $\mathcal { P } _ { \mathrm { S V } }$ are designed for sentiment classification (SC) and self-verification (SV) respectively. Each prompt consists of an instruction, an input and a output format. The instruction formally describes the task. The input presents the input sample for the task, while the output format formally describe the output structure. Examples of $\mathcal { P } _ { \mathrm { S C } }$ and $\mathcal { P } _ { \mathrm { S V } }$ are presented in Figure 2. Based on the constructed $\mathcal { P } _ { \mathrm { S C } }$ and $\mathcal { P } _ { \mathrm { S V } }$ , we perform zero-shot prediction and self-verification for sample acquisition, which involves three steps: zero-shot prediction, self-verification guided sample division, ranking by uncertainty.

Zero-Shot Prediction $\mathcal { D } _ { \mathrm { u n l a b e l e d } } = \{ ( s ^ { j } , a ^ { j } , v ^ { j } ) \} _ { j = 1 } ^ { N }$ denotes the unlabeled training dataset. For the $j$ -th sample in $\mathcal { D } _ { \mathrm { u n l a b e l e d } }$ , we perform zero-shot prediction as follows:

$$
\hat { y } ^ { j } = \mathrm { L V L M } ( \mathcal { P } _ { \mathrm { S C } } ^ { j } )
$$

where $\hat { y } ^ { j }$ denotes the sentiment prediction.

Self-Verification Guided Sample Division Subsequently, we perform a self-verification and divide samples into $\mathcal { D } _ { \mathrm { n o } }$ and $\mathcal { D } _ { \mathrm { y e s } }$ , as follows:

$$
\hat { c } ^ { j } = \mathrm { L V L M } ( \mathcal { P } _ { \mathrm { S V } } ^ { j } )
$$

$$
\mathcal { D } _ { \mathrm { n o } } = \{ ( s ^ { j } , a ^ { j } , v ^ { j } ) | \hat { c } ^ { j } = \mathtt { n o } \} _ { j = 1 } ^ { N }
$$

$$
\mathcal { D } _ { \mathrm { y e s } } = \{ ( s ^ { j } , a ^ { j } , v ^ { j } ) | \hat { c } ^ { j } = \mathtt { y e s } \} _ { j = 1 } ^ { N }
$$

Samples are assigned to $\mathcal { D } _ { \mathrm { n o } }$ if there is a divergence between their zero-shot predictions and subsequent selfverification outputs $( \hat { c } ^ { j } = \mathfrak { n } \circ \cdot$ ); otherwise, they are assigned to $\mathcal { D } _ { \mathrm { y e s } }$ $\hat { c } ^ { j } = \mathrm { y } \mathsf { e } \mathsf { s } )$ . Samples in $\mathcal { D } _ { \mathrm { n o } }$ are considered challenging since they cause the model to shift its prediction after re-evaluation via self-verification. Fine-tuning the model on these samples is crucial for improving its understanding and can lead to greater training benefits.

Ranking by Uncertainty To identify the samples where the LVLM becomes firmly convinced of a new prediction during re-evaluation, we estimate the uncertainty of selfverification output and focus on selecting samples within the low-uncertainty. The uncertainty for a sample is computed as follows:

$$
u ^ { j } = { \frac { 1 } { \operatorname* { m a x } ( P ( { \hat { c } } ^ { j } ) ) } }
$$

where $P ( \hat { c } ^ { j } )$ represents the probability distribution for the self-verification output $\hat { c } ^ { j }$ , and $\operatorname* { m a x } ( \dot { P } ( \hat { c } ^ { j } ) )$ represents the highest probability of $P ( \hat { c } ^ { j } )$ . We rank the samples by their uncertainty scores and select the top- $K$ samples with the lowest scores for fine-tuning.

# Multimodal Prompts Construction

# Input Example from Twitter-2017

Sentence: How jake paul is changing the influencer game.   
Aspect: jake paul.

# Example for Construction of $\mathcal { P } _ { S C }$

# Instruction:

Given a set of sentiment labels [positive, neutral, negative], your task is to determine the sentiment polarity associated with a specific aspect in the provided text. An image, supplied as additional context, will assist in inferring the sentiment towards the aspect.

![](images/ccca5dcb571424c69a92d4b6bf40881da7fb6669d6aa0e96b5d25a43bc5d0034.jpg)  
Input: Image:

Sentence:   
How jake paul is changing the   
influencer game. Aspect:   
jake paul   
Output Format:   
Select from [positive, neutral, negative]   
Assistant:

![](images/35f0c49a3bfc9ba2d9afed0afe2b1b6c919d129251d919d6d50c100dc79acd60.jpg)

zero-shot prediction and selfverification is consistent.

×

zero-shot prediction and selfverification is inconsistent.

国

Verification and Zero-Shot Feedback Acquisition Step 1: Zero-Shot Prediction p Sample #1 Positive prompt Sample #2 Neutral construction Sample.#.3. Positive Dunlabled LVLMs Sample #N Negative Example for Construction of Step 2: Self-verification Guided Sample Division Instruction: 0 Sample #1 Positive (Yes) pAlcecasoredjiundg teowthe tihmeargteheancdontcwleuesit,on cproonsmtrputction m Sample #23 PNoesuitriavle((No)) self-vresriuflitcsation is correct. LVLMs Sample #N Negative (Yes) Input: Sentence: Image: How jake paul is changing the Step 3: Ranking by Uncertainty influencer game. Sample #2 Neutral (No) #2 P(No) = [0.01, 0.2, 0.09, ...] #2 U(No) = 5.0 Aspect: Sample #3 Positive (No) #3 P(No) = [0.001, 0.1, 0.004, ...] #3 U(No) = 10.0 jake paul Sample #8MPoosistiitvivee((Noo) $\bullet \cdots \# \mathbb { 8 } \operatorname { P } ( \mathrm { N o } ) = [ 0 . 0 0 2 , \mathbf { 0 . 0 9 } , 0 . 0 3 , \ldots ] \bullet \cdots \quad \operatorname { U } ( \mathrm { N o } ) = 1 1 . 1$ $\mathbf { \Psi } = \mathbf { \Psi }$ Conclusion: Sample #1 Neutral (Yes) #1 P(Yes) = [0.02, 0.05, 0.08, ...] #1 U(Yes) = 12.5 The sentiment of jake paul is positive. Sample #4 Po ${ \mathrm { i t i v e ~ ( Y e s ) } } \bullet \cdots \# { \mathrm { 4 ~ P ( Y e s ) } } = \lceil 0 . 0 0 6 , 0 . 0 0 9 , \mathbf { 0 . 2 } , \ldots \mid \bullet \cdots \# { \mathrm { 4 ~ U ( Y e s ) } } = 5 . 0$ Output Format: Sample #5NPosiittiive $( \mathrm { Y e s } ) \bullet \dots \# 5 \mathrm { P } ( \mathrm { Y e s } ) = [ 0 . 0 1 2 , 0 . 0 5 , \mathbf { 0 . 1 7 } , \dots ] \bullet \dots \# 5 \mathrm { U } ( \mathrm { Y e s } ) = 5 . 8$ $( { \mathrm { Y e s } } ) ^ { \bullet \dots \sharp / \mathrm { N } } { \mathrm { P } } ( { \mathrm { Y e s } } ) = [ 0 . 0 9 , 0 . 0 7 , \mathbf { 0 . 1 } 3 , \dots ] \bullet \dots \sharp { \mathrm { N U } } ( { \mathrm { Y e s } } ) = 7 . 7 \quad$ rU Output Yes or No Dyes/no Probability Distribution UEnstciemrtatinotny Assistant: Top-(K-H) Top-H unlabeled training dataset. #4, #5, #N, #1 #2, #M, #3, #8 selected fine-tuning dataset, which is Easy samples Dyes Dft Challenging samples Dno assigned with ground-truth sentiment labels. LVLM Fine-tuning

Figure 2: Overview of our approach.

# Fine-tuning for Few-shot MALSC

The top- $\cdot K$ samples with the lowest scores are considered challenging, as their predictions diverge significantly between zero-shot and self-verification. However, to help the model maintain stable performance across a range of examples rather than focusing too heavily on challenging cases during few-shot fine-tuning, we include a small set of relatively easy samples from $\mathcal { D } _ { \mathrm { y e s } }$ to balance training. To achieve this, we introduce a hyperparameter $\lambda \in \ [ 0 , \bar { 1 } ]$ to quantify the proportion of samples selected from $\mathcal { D } _ { \mathrm { n o } }$ . Let $\bar { H } = \bar { \lfloor \lambda K \rfloor }$ be the number of samples selected from $\mathcal { D } _ { \mathrm { n o } }$ . The dataset for fine-tuning can be constructed as follows:

$$
\begin{array} { r l } & { \mathcal { D } _ { \mathrm { f t } } = \mathcal { D } ^ { C } \cup \mathcal { D } ^ { E } } \\ & { \mathcal { D } _ { \mathrm { f t } } = \{ ( x ^ { j } , y ^ { j } ) | x ^ { j } \in \mathrm { S o r t } ( \mathcal { D } _ { \mathrm { n o } } ) \} _ { j = 1 } ^ { H } } \\ & { \cup \{ ( x ^ { j } , y ^ { j } ) | x ^ { j } \in \mathrm { S o r t } ( \mathcal { D } _ { \mathrm { y e s } } ) \} _ { j = 1 } ^ { K - H } } \end{array}
$$

where $\mathcal { D } ^ { C } \subset \mathcal { D } _ { \mathrm { n o } }$ (Challenging samples) and $\mathcal { D } ^ { E } \subset \mathcal { D } _ { \mathrm { y e s } }$ (Easy samples); Sort(·) function ranks samples by selfverification uncertainty scores in ascending order.

Loss Function Finally, we fine-tune the LVLM on $\mathcal { D } _ { \mathrm { f t } }$ , where the sentiment label $y$ is transformed to the target sequence $\bar { y }$ to fit the generative nature of LVLM. For example, the sentiment label positive is formatted as The sentiment of {aspect} is positive. The target sequence is used to compute cross-entropy loss with the LVLM’s output, as follows:

$$
\mathcal { L } = \frac { 1 } { K \times M } \sum _ { j = 1 } ^ { K } \sum _ { i = 1 } ^ { M } \mathrm { C E } ( \hat { y } _ { i } ^ { j } , \bar { y } _ { i } ^ { j } )
$$

where CE( ) denotes the cross-entropy loss function, $K$ denotes the number of samples in $\mathcal { D } _ { \mathrm { f t } }$ and $M$ denotes the length of target sequence.

# Experiment

Setup Following the recent work (Yang et al. 2023b), we evaluate our approach on two benchmark datasets of MALSC: Twitter-2015 and Twitter-2017. To ensure a fair comparison, we directly utilize the preprocessed datasets provided by (Yang et al. 2023b). In the few-shot setting, $1 \%$ or $7 \%$ of samples are selected from the training dataset to fine-tune the models. The statistics of the two datasets are presented in Table 2. Implementation details and evaluation protocols are outlined in the accompanying code repository.

# Main Result

The performance comparison on few-shot MALSC is presented in Table 1. Firstly, we find that some Text-Only models even achieve promising performances, compared to the Text-Image baselines. For example, LM-SC shows competitive performance with UP-MPF while surpassing all previous Text-Image baselines, suggesting the importance of developing a strong image understanding module to effectively use image information. Secondly, we find that MultiPoint, current state-of-the-art for this task, achieves a large performance margin over previous baselines by leveraging consistently distributed sampling (CDS) for few-shot fine-tuning.

Table 1: Few-shot performance on the Twitter-2015 and Twitter-2017 datasets. For all datasets, the few-shot dataset represents $1 \%$ of the overall training data. The best performance is marked in bold. Our results are averaged over 10 runs with different random seeds, with standard deviation reported. Results for the compared models are retrieved from (Yang et al. 2023b).   

<html><body><table><tr><td rowspan="2">Modality</td><td rowspan="2">Model</td><td colspan="2">Twitter-2015</td><td colspan="2">Twitter-2017</td></tr><tr><td>Accuracy</td><td>Weighted-F1</td><td>Accuracy</td><td>Weighted-F1</td></tr><tr><td>Text-Only</td><td>RoBERTa (Liu et al. 2019)</td><td>55.58±4.13</td><td>52.32±2.28</td><td>48.22±2.95</td><td>46.37±3.17</td></tr><tr><td rowspan="6"></td><td>PT(Yang et al. 2023b)</td><td>61.97±3.15</td><td>60.11±3.38</td><td>58.77±3.70</td><td>57.85±3.63</td></tr><tr><td>LM-BFF(Gao,Fisch,and Chen 2021)</td><td>60.87±3.38</td><td>59.63±3.04</td><td>56.84±3.51</td><td>55.96±3.48</td></tr><tr><td>LM-SC (Jian, Gao,and Vosoughi 2022)</td><td>61.16±3.31</td><td>60.99±3.28</td><td>54.78±1.93</td><td>52.89±2.63</td></tr><tr><td>GFSC (Hosseini-Asl, Liu,and Xiong 2022)</td><td>52.77±0.38</td><td>52.01±0.56</td><td>54.43±2.47</td><td>53.15±2.70</td></tr><tr><td>MFN(Yang et al. 2023b)</td><td>55.86±1.66</td><td>52.81±1.45</td><td>50.91±2.86</td><td>49.20±3.05</td></tr><tr><td>CLMLF (Li et al. 2022)</td><td>56.97±2.08</td><td>52.04±2.35</td><td>49.63±2.40</td><td>45.72±2.17</td></tr><tr><td rowspan="10"></td><td>TomBERT(?)</td><td>55.95±5.17</td><td>43.25±0.06</td><td>47.47±2.26</td><td>36.93±5.89</td></tr><tr><td>EF-CapTrBERT (Khan and Fu 2021)</td><td>57.81±1.45</td><td>42.72±1.00</td><td>47.41±1.01</td><td>33.58±3.58</td></tr><tr><td>KEF (Ling, Yu,and Xia 2022)</td><td>57.58±2.04</td><td>43.09±0.25</td><td>45.74±0.78</td><td>31.29±2.39</td></tr><tr><td>FITE (Yang, Zhao,and Qin 2022b)</td><td>58.42±0.18</td><td>43.29±0.11</td><td>46.20±0.52</td><td>29.97±0.70</td></tr><tr><td>VLP-MABSA (Ling, Yu,and Xia 2022)</td><td>53.36±1.07</td><td>43.23±3.75</td><td>55.32±3.39</td><td>48.96±1.26</td></tr><tr><td>PVLM (Yu and Zhang 2022)</td><td>59.25±2.02</td><td>54.45±3.33</td><td>54.28±3.17</td><td>51.02±5.24</td></tr><tr><td>UP-MPF(Yu, Zhang,and Li 2022)</td><td>61.56±2.43</td><td>60.16±2.54</td><td>54.93±2.22</td><td>51.87±4.08</td></tr><tr><td>MultiPoint (Yang et al. 2023b)</td><td>67.33±1.07</td><td>66.61±1.36</td><td>61.88±2.56</td><td>61.23±2.58</td></tr><tr><td>LLaVA-7b</td><td>50.24</td><td>48.26</td><td>56.00</td><td>53.00</td></tr><tr><td>LLaVA-13b</td><td>57.86</td><td>57.04</td><td>57.86</td><td>54.87</td></tr><tr><td></td><td>VEROLLaVA-7b (ours) VEROLLaVA-13b (ours)</td><td>67.30±0.41 69.17±0.46</td><td>66.49±0.18 69.42±0.46 67.46±0.28</td><td>64.56±0.68</td><td>64.22±0.86</td></tr></table></body></html>

Table 2: Dataset Statistics, including counts of positive #POS, neutral #NEU, and negative #NEG instances, as well as the average number of aspects per sentence #ASP, and the average sentence length #LEN   

<html><body><table><tr><td></td><td colspan="3">Twitter-2015</td><td colspan="3">Twitter-2017</td></tr><tr><td></td><td>Train</td><td>Dev</td><td>Test</td><td>Train</td><td>Dev</td><td>Test</td></tr><tr><td>#POS #NEU</td><td>928 1883</td><td>303</td><td>317</td><td>1508</td><td>515</td><td>493</td></tr><tr><td>#NEG</td><td></td><td>679</td><td>607</td><td>1638</td><td>517</td><td>573</td></tr><tr><td>Total</td><td>368 3179</td><td>149 1122</td><td>113</td><td>416</td><td>144</td><td>168</td></tr><tr><td>#ASP</td><td>1.34</td><td></td><td>1037</td><td>3562</td><td>1176</td><td>1234</td></tr><tr><td>#Len</td><td>16.72</td><td>1.33 16.74</td><td>1.35 17.05</td><td>1.41 16.21</td><td>1.45 16.37</td><td>1.45 16.38</td></tr></table></body></html>

To establish a clear baseline for LVLM-based models, we experimented with the vanilla LLaVA-7b/13b models and observed that MultiPoint outperforms them by large margins. This highlights the importance of fine-tuning LVLMs to achieve competitive performance for the task, as these models are designed as general-purpose solvers. VERO, our finetuned LLaVa-7b/13b, achieves gains of at least 8 Acc and 11 Weighted-F1 points over LLaVA-7b/13b, suggesting its gains stem from fine-tuning. Comparing to MultiPoint, VEROLLaVA 7b achieves comparable performance on Twitter-2015 and outperforms it on Twitter-2017. $\mathrm { V E R O } _ { \mathrm { L L a V A - 1 3 b } }$ further sets a new SOTA, surpassing MultiPoint by $3 . 7 1 \%$ in accuracy and $4 . 4 4 \%$ in weighted F1, while demonstrating stability with low standard deviations.

Some recent works (e.g. GMP (Yang et al. 2023a)), perform this task on the $7 \%$ few-shot setting. Accordingly, we evaluate our model on this setting as well, as presented in Table 3. It can be observed that the VERO surpasses the GMP with an accuracy margin of $3 . 2 2 \%$ and $3 . 1 8 \%$ on Twitter2015 and Twitter-2017, further proving its superiority.

Table 3: Performance comparison in terms of Accuracy on the Twitter-2015 and Twitter-2017 datasets. For all datasets, the few-shot dataset represents $7 \%$ of the overall training data. Results for the compared models are retrieved from (Yang et al. 2023a).   

<html><body><table><tr><td>Model</td><td>Twitter-2015</td><td>Twitter-2017</td></tr><tr><td>LM-BFF LM-SC GFSC</td><td>64.87±0.40 65.47±1.74 60.75±1.07</td><td>52.08±0.54 57.51±2.95 61.72±0.16</td></tr><tr><td>TomBERT CapTrBERT KEF FITE VLP PVLM UP-MPF</td><td>61.78±3.27 58.76±0.25 55.81±3.74 63.11±0.53 59.34±1.35 64.54±1.81 63.71±3.62</td><td>59.97±2.30 56.48±1.61 46.50±0.08 60.89±1.40 60.24±1.61 61.45±2.31 62.02±0.40</td></tr><tr><td>GMP VERO</td><td>67.06±0.55 70.28±0.72</td><td>66.20±1.12 69.38±0.36</td></tr></table></body></html>

# Performance on Ablated Acquisition Functions Further Analysis

To verify the contribution of each component in our method, we compare the performances of different acquisition functions ablated from our approach. We categorize these acquisition functions into uncertainty-based methods which select samples based on the uncertainty scores of the LVLM’s zero-shot prediction or self-verification, and the diversitybased methods which select samples based on a prior distribution. The uncertainty-based methods include: (1) - $\cdot \mathcal { D } ^ { E }$ which forces the model to finetune on only challenging samples (2) “- $\cdot \mathcal { D } ^ { E }$ ,Ver” which additionally removes the selfverification from our model. Thus, the acquisition function selects samples based on the uncertainty scores of the zeroshot predictions. The diversity-based methods include: (3) “- $\mathcal { D } ^ { E }$ ,Unc” which reduces the model to randomly select challenging samples; (4) “- $\cdot \mathcal { D } ^ { E }$ ,Unc,Ver” which further reduces the model to follow the CDS method to randomly select samples from the training set. The results are presented in Table 4.

Table 4: Weighted-F1 performance under different acquisition functions ablated from our approach.   

<html><body><table><tr><td>Base</td><td>Model</td><td>Twitter-2015</td><td>Twitter-2017</td></tr><tr><td rowspan="4"></td><td colspan="2">Uncertainty-basedMethods</td></tr><tr><td>VERO 66.49±0.18 64.66±0.26 -DE</td><td>64.22±0.86 62.95±0.38</td></tr><tr><td>-DE,Ver 64.44±0.24 Diversity-basedMethods</td><td>62.80±0.59</td></tr><tr><td colspan="2">-DE,Unc 64.30±0.37 -DE,Unc,Ver 63.97±0.40</td></tr><tr><td rowspan="3"></td><td colspan="2">61.35±0.98 Uncertainty-basedMethods</td></tr><tr><td>VERO 69.22±0.46 -DE,Ver</td><td>67.30±0.29 65.73±0.53</td></tr><tr><td colspan="2">68.30±0.34 Diversity-basedMethods</td></tr><tr><td colspan="2">-DE,Unc -DE,Unc,Ver</td><td colspan="2">68.21±0.67 65.84±1.28 67.45±0.40 64.92±1.51</td></tr></table></body></html>

Firstly, we find that the uncertainty-based methods generally outperform the diversity-based methods, indicating the importance of considering the capability of the LVLM itself to select challenging samples. Secondly, we also find that the performance of our model drops when removing the $\mathcal { D } ^ { E }$ , demonstrating that easy samples improve the model’s robustness. We also observe that the “- $\mathbf { \nabla } \cdot \hat { \mathcal { D } } ^ { E ^ { , , } }$ model outperforms the “- $\mathcal { D } ^ { E }$ ,Ver” model, indicating the samples selected by self-verification provide greater training benefits than that by zero-shot prediction. The “- $\mathcal { D } ^ { E }$ ,Unc” model underperforms the “- $\mathring { \mathcal { D } } ^ { E }$ ” model, which is expected since ranking samples by their uncertainty scores results in more challenging samples than random selection. Meanwhile, the “- $\mathcal { D } ^ { \tilde { E } }$ ,Unc” model surpasses the “- $\mathcal { D } ^ { E }$ ,Unc,Ver” model, proving the effectiveness of selecting samples from the $\mathcal { D } _ { \mathrm { n o } }$ .

Constrained by computational resources, additional analyses are performed using the LLaVA-7b backbone.

Error Analysis To examine the training benefits of our approach, we experiment with our acquisition function and its two ablated variants: “- $\mathbf { \nabla } \cdot \mathcal { D } ^ { E ^ { , , } }$ and “- $\Dot { \mathcal { D } } ^ { E }$ ,Unc,Ver” (denoted as “CDS”). The error analysis is demonstrated by the confusion matrices in Figure 3.

Zero-Shot CDS -DE Ours   
I 260 47 10 241 75 1 173 144 0 176 139 2 50 279 222 106 196 387 24 94 499 14 97 479 31 30 12 19 82 11 73 29 5 87 21 4 66 43 T T 350   
GR 330 133 30 380 105 8 281 209 3 304 178 11 178 258 137 223 302 48 126 427 20 128 400 45 50 10 22 136 21 53 94 9 102 57 10 65 93 50 pos neu neg pos neu neg pos neu neg pos neu neg Prediction

Firstly, let us take a look at the results on the Twitter-2015 dataset. The LVLM’s zero-shot prediction shows strong performances on positive and negative sentiments while primarily failing on the neutral sentiment. We find the “CDS” model improves the performance on the neutral sentiment. However, it is still worse than the “- ${ \mathcal { D } } ^ { E : }$ ” model. This result proves the effectiveness of our acquisition function on enhancing the model’s performance on challenging samples. Despite the large improvement on the neutral sentiment, the “- $\mathbf { \nabla } \cdot \hat { \mathcal { D } ^ { E } }$ ” model’s performances on positive and negative sentiments drop compared to its zero-shot performances. This is due to focusing too heavily on challenging samples. By introducing a small set of samples in $\mathcal { D } _ { \mathrm { y e s } }$ to balance training, VERO improves in performances on positive and negative sentiments, albeit sacrificing a bit of performance on neutral sentiment. Similar model behaviours can be observed in the Twitter-2017 dataset as well.

Impact of Hyperparameter $\lambda$ The hyperparameter $\lambda$ denotes the proportion of samples selected from $\mathcal { D } _ { \mathrm { n o } }$ . Now, we vary the $\lambda$ from 1.0 to 0.0, to study its impact. $\lambda \ : = \ : 1 . 0$ means all samples are selected from $\mathcal { D } _ { \mathrm { n o } }$ , while $\lambda = 0 . 0$ means all samples are selected from $\mathcal { D } _ { \mathrm { y e s } }$ . The performance curves under different $\lambda$ are shown in Figure 4 and the sentiment distributions of selected samples under different $\lambda$ are shown in Figure 5.

Firstly, we find that the performance at $\lambda = 1 . 0$ significantly outperforms the performance at $\lambda \ : = \ : 0 . 0$ , indicating that learning from samples where the LVLM firmly convinces shift in predictions via self-verification brings greater training benefits, which supports our motivation. Secondly, we find that the model achieves the best performance at $\lambda = 0 . 9$ on both datasets, indicating the importance of introducing a small proportion of most confident samples in $\mathcal { D } _ { \mathrm { y e s } }$ . As observed in the “Error Analysis”, these samples help balance the training process, reducing the model’s tendency to overfit to challenging examples during fine-tuning.

![](images/fd36fa81abebfd5e2ca1c0db0b982087683c26fcb48b452887dd471a84f9e116.jpg)  
Figure 4: Weighted-F1 curves under different $\lambda$ on the Twitter-2015 (left) and Twitter-2017 (right) datasets.

![](images/12fe98235e5cd336e849394b94735b042e53923268a957af2ea104afb327fafc.jpg)  
Figure 5: Sentiment distributions of the selected samples under different $\lambda$ on the Twitter-2015 (left) and Twitter-2017 (right) datasets.

Turning to look at the sentiment distributions of selected samples under different $\lambda$ , we find that at $\lambda = 1 . 0$ the neutral samples account for the highest proportion, about 0.75, while both positive and negative samples account for a very low proportion on the Twitter-2015 dataset. This result indicates that our acquisition function is clever to select the challenging samples against the zero-shot capability of the LVLM, which is corresponding to our observation in the “Error Analysis”. As $\lambda$ decreases, the proportion of neutral samples drops while the proportion of positive samples rises. This is expected since the LVLM’s zero-shot performance on the positive sentiment is relatively higher. However, we find that the proportion of negative samples remains low as $\lambda$ varies. This is due to the very low proportion of negative samples in the entire training set. A similar behaviour can also be observed on the Twitter-2017 dataset, where the LVLM’s zero-shot performance on the positive sentiment is slightly lower than that on the Twitter-2015 dataset, resulting in a slightly higher proportion of positive samples at $\lambda = 1 . 0$ .

Effectiveness of Self-Verification for Sample Acquisition We select challenging samples based on the uncertainty of the LVLM’s self-verification, rather than its zero-shot prediction. However, selecting samples based on uncertainty of the model’s predictions is a conventional approach in existing active learning studies. For a further analysis, we present the sentiment proportion and error rate of samples selected by the two strategies in Table 5, where selecting samples based on the uncertainty of the LVLM’s self-verification and zero-shot prediction are denoted as “- $\mathbf { \nabla } \cdot \mathcal { D } ^ { E ^ { , , } }$ and “- $\cdot \mathcal { D } ^ { E }$ ,Ver” respectively.

Firstly, we find that the proportion of the neutral sentiment in selected samples by the “- $\mathbf { \nabla } \mathcal { D } ^ { E ^ { , , } }$ model is significantly higher than that by the “- $\mathcal { D } ^ { E }$ ,Ver” model on both datasets, while the proportions of neutral and positive sentiments for the “- $\mathcal { D } ^ { E }$ ,Ver” model are close on both datasets. It is evident that the distribution of samples selected by the “- $\begin{array} { r } { \mathcal { D } ^ { E } \dag , } \end{array}$ ” model aligns more closely with the model’s zero-shot performance, with the neutral sentiment accounting for the majority of error cases. Furthermore, we observe that the error rate of the LVLM’s zero-shot prediction on the samples selected by the “- $\mathbf { \nabla } \mathcal { D } ^ { E ^ { , , } }$ model is higher than that on the samples selected by the “- $\cdot \mathcal { D } ^ { E }$ ,Ver” model. This suggests that the self-verification mechanism is more effective at identifying challenging samples. We believe that self-verification represents a unique capability of LVLMs, distinct from zero-shot prediction. A majority of studies have also found that self-verification can effectively correct prediction errors. This ability offers more reliable cues for LVLMs to identify challenging samples, ultimately improving fine-tuning performance.

Table 5: The sentiment proportions of the selected samples for different models on the Twitter-2015 and Twitter-2017 datasets. “ER” denotes the error rate of the LVLM’s zeroshot prediction on the selected samples.   

<html><body><table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td colspan="2">Sentiment Proportion</td><td rowspan="2">ER</td></tr><tr><td>POS NEG</td><td>NEU</td></tr><tr><td>Twitter-15</td><td>-DE,Ver -DE</td><td>0.12 0.12 0.42 0.12</td><td>0.76 0.45</td><td>51.52 42.42</td></tr><tr><td>Twitter-17</td><td>-DE,Ver -DE</td><td>0.24 0.16 0.43 0.08</td><td>0.59 0.49</td><td>54.05 24.32</td></tr></table></body></html>

# Conclusion

In this paper, we solve the challenge of fine-tuning large vision-language models for few-shot multimodal aspectlevel sentiment classification by introducing VERO, a novel acquisition function. VERO identifies challenging samples from a pool of unlabeled data by leveraging the zero-shot and self-verification capabilities of LVLMs, ensuring that the selected samples significantly contribute to the model’s learning process. Our work also highlights the importance of strategically introducing easy samples, as a means to balance training and enhance the generalization of LVLMs. We experimented with VERO on two benchmark datasets and demonstrated that our approach consistently outperforms the recent competitive methods in both $1 \%$ and $7 \%$ few-shot settings. Further analyses demonstrate the advantage of VERO over existing sampling methods. The main limitation of our method is the underutilization of unlabeled data. Therefore, we plan to extend our approach to a semi-supervised framework in future work.