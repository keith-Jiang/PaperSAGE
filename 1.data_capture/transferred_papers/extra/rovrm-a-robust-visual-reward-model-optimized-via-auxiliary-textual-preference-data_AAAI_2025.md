# RoVRM: A Robust Visual Reward Model Optimized via Auxiliary Textual Preference Data

Chenglong Wang1\*, Yang $\mathbf { G a n } ^ { 1 * }$ , Yifu $\mathbf { H u o } ^ { 1 * }$ , Yongyu $\mathbf { M } \mathbf { u } ^ { 1 }$ , Murun Yang1, Qiaozhi $\mathbf { H e } ^ { 1 }$ , Tong Xiao1,2†, Chunliang Zhang1,2, Tongran $\mathbf { L i u } ^ { 3 }$ and Jingbo Zhu1,2

1 School of Computer Science and Engineering, Northeastern University, Shenyang, China 2 NiuTrans Research, Shenyang, China   
3 CAS Key Laboratory of Behavioral Science, Institute of Psychology, CAS, Beijing, China {clwang1119, zzhu8250}@gmail.com, {xiaotong, zhujingbo}@mail.neu.edu.cn

# Abstract

Large vision-language models (LVLMs) often fail to align with human preferences, leading to issues like generating misleading content without proper visual context (also known as hallucination). A promising solution to this problem is using human-preference alignment techniques, such as best-of$n$ sampling and reinforcement learning. However, these techniques face the difficulty arising from the scarcity of visual preference data, which is required to train a visual reward model (VRM). In this work, we continue the line of research. We present a Robust Visual Reward Model (RoVRM) which improves human-preference alignment for LVLMs. RoVRM leverages auxiliary textual preference data through a threephase progressive training and optimal transport-based preference data selection to effectively mitigate the scarcity of visual preference data. We experiment with RoVRM on the commonly used vision-language tasks based on the LLaVA1.5-7B and -13B models. Experimental results demonstrate that RoVRM consistently outperforms traditional VRMs. Furthermore, our three-phase progressive training and preference data selection approaches can yield consistent performance gains over ranking-based alignment techniques, such as direct preference optimization.

# Introduction

Large language models (LLMs) have demonstrated remarkable capabilities across various natural language processing tasks (Stiennon et al. 2020; Ouyang et al. 2022). Recent works tend to fine-tune LLMs using specialized visual instruction tuning datasets, leading to the emergence of powerful large vision-language models (LVLMs) (Liu et al. 2024a; Lin et al. 2024; Huang et al. 2024b). Despite these advancements, current LVLMs are not well-aligned with human preferences. A glaring problem is that LVLMs sometimes generate misleading content without anchoring to the given visual context (also known as hallucination) (Leng et al. 2024). For instance, as illustrated in Figure 1, an LVLM incorrectly identifies a “pitaya” in an image of mangosteens due to their visual similarity.

Two predominant research approaches aim to address this problem. The first approach focuses on generating richer and higher-quality visual instruction data (Li et al. 2023b; Liu et al. 2023, 2024c), i.e., annotating rich instruction samples on images of mangosteens to enable LVLMs to identify them more accurately. In contrast, a more sophisticated approach is applying human-preference alignment techniques, including best-of- $n$ sampling and reinforcement learning (RL), which can efficiently align models with human preferences on various tasks by optimizing against a reward model without instruction samples. However, applying these alignment techniques to LVLMs is not a low-hanging fruit. It typically faces the difficulty of training a visual reward model (VRM) due to the scarcity of high-quality visual preference data (Sun et al. 2023; Yu et al. 2024a; Zhou et al. 2024b).

This work is motivated by a simple idea: human preferences are well-captured by text and these preferences can be transferred across different modalities. In this way, we can make use of rich, high-quality textual preference data in training VRMs. Building on this idea, we present a Robust Visual Reward Model (RoVRM), which can improve human-preference alignment for LVLMs in two ways. For one, we propose a three-phase progressive training approach to gradually bridge the task and modality gaps between textual and visual preference data, which can take full advantage of auxiliary textual preference data to improve the robustness of RoVRM. Furthermore, considering the conflict in preferences (Coste et al. 2023; Eisenstein et al. 2023), leveraging textual preference data poses a problem: not all data is beneficial for training the RoVRM. Addressing this problem, we propose an optimal transport-based preference data selection approach. This approach can select textual preference data that better aligns with the visionlanguage task preferences, thereby improving the efficacy of the RoVRM training process. To the best of our knowledge, we are the first to investigate the integration of preferences from different modalities.

Through experiments on commonly used vision-language tasks, we aim to evaluate RoVRM using two humanpreference alignment techniques: best-of- $\boldsymbol { \cdot } \boldsymbol { n }$ sampling and RL. Our results demonstrate improved performance in each task when aligned with reward signals from RoVRM. Notably, when performing best-of- $\cdot n$ sampling on the LLaVA

1.5-7B model, RoVRM outperforms a traditional VRM by   
8.4 points on the LLaVA-Bench benchmark.

As another bonus, our three-phase progressive training and preference data selection can be seamlessly integrated with arbitrary ranking-based alignment techniques, such as direct preference optimization (DPO) (Rafailov et al. 2024), SimPO (Meng, Xia, and Chen 2024), and ORPO (Hong, Lee, and Thorne 2024). For instance, on the LLaVA-1.5-13B model, integrating with DPO results in an additional improvement of 17.82 points on the MM-Instruct benchmark compared to standard DPO.

Our code is publicly available\*. This version summarizes the key experimental setup and results, with further details provided in our arXiv submission†.

# Related Work

In recent years, LVLMs have served as the primary backbone for vision-language tasks (Achiam et al. 2023; AI 2023). Aligning LVLMs with human preferences is effective in gaining more performance (Liu et al. 2023; Wang et al. 2024b). However, in this process, they only used visual preference data and never leveraged the textual preference data that exists in abundance.

Large Vision-Language Models Inspired by the success of LLMs such as GPTs (Brown et al. 2020; Ouyang et al. 2022) and LLaMA (Touvron et al. 2023), researchers have been aiming to develop LVLMs. The basic idea is to augment LLMs with visual inputs (e.g., images) to provide an interface for vision-language tasks (Alayrac et al. 2022; Awadalla et al. 2023; Aiello et al. 2023). Recent works on LVLMs could be classified into two groups. The first group focused on integrating visual information into LLMs (Chen et al. 2023; Liu et al. 2024a; Wang et al. 2024c). For example, Liu et al. (2024b) constructed a large amount of visual instruction data to pre-train the visual projection layer. Lin et al. (2024) further investigated the effective pre-training design options to augment LVLMs. The second group that has attracted attention commonly aimed to improve the consistency of output text and visual content, particularly addressing the problem of hallucination (Zhou et al. 2023; Leng et al. 2024; Gunjal, Yin, and Bas 2024; Huang et al. 2024a; Favero et al. 2024). This work belongs to the latter, where our RoVRM can improve the consistency of output text and visual content.

Human-Preference Alignment for LVLMs Reinforcement learning with human feedback (RLHF) has been shown to effectively align LLM behaviors with human preferences (Stiennon et al. 2020; Ouyang et al. 2022). Several works have improved RLHF by using fine-grained reward models (Wu et al. 2024), reward model ensembles (Coste et al. 2023), and direct preference optimization objectives (Rafailov et al. 2024). Additionally, some works focused on generating large, high-quality textual preference datasets to further improve RLHF in LLMs (Cui et al. 2023; Dubois et al. 2024). In the context of LVLMs, existing works mainly focused on the adaptation of the human-preference alignment techniques (Sun et al. 2023; Li et al. 2023a; Yu et al. 2024a). A significant challenge here was the scarcity of visual preference data. To address this challenge, many efforts have been made to create visual preference data, including collecting human preferences (Sun et al. 2023), and acquiring preferences from a strong LVLM (Li et al. 2023a; Yu et al. 2024b). Different from these works, we investigate how to leverage rich, high-quality textual preference data to offset the scarcity of visual preference data.

# Our Method

We first review the preliminaries of the human-preference alignment training for language models. Then, we present the three-phase progressive training for use with RoVRM. Last, we introduce the proposed preference data selection.

# Preliminaries

Reinforcement Learning with Human Feedback RLHF is a key technique for aligning language models with human preferences. It typically consists of two main steps: 1) training a reward model (also known as preference model) from preference data, and 2) using an RL algorithm, such as PPO (Schulman et al. 2017), to maximize the reward. In step 1, we usually employ the Bradley-Terry model (Bradley and Terry 1952). When the preference data existed in a comparison pair, the loss function can be written as:

$$
\mathcal { L } _ { r e w a r d } = - \log ( \sigma ( r _ { \theta } ( x , y _ { w } ) - r _ { \theta } ( x , y _ { l } ) ) )
$$

where $\sigma$ is the Sigmoid activation function, $r ( \cdot )$ is a reward model and $\theta$ is its parameters. $y _ { w }$ and $y _ { l }$ are two different responses for the human prompt $x$ , where $y _ { w }$ is more preferred than $y _ { l }$ . When dealing with multiple responses more than two, we can induce $\mathcal { L } _ { r e w a r d }$ based on the more general Plackett-Luce model (Luce 2005):

$$
\mathcal { L } _ { r e w a r d } = - \sum _ { i = 1 } ^ { k } \log \frac { \exp { \left( r _ { \theta } ( x , y _ { i } ) \right) } } { \sum _ { j = i } ^ { k } \exp { \left( r _ { \theta } ( x , y _ { j } ) \right) } }
$$

where $k$ denotes the number of responses. These responses are ranked by the defined preferences: $( y _ { 1 } \succ \dots \succ y _ { k } | x )$ , where $y _ { 1 }$ is the best while $y _ { k }$ is the worst. In step 2, the reward signals produced by the trained reward model are instrumental in adjusting the parameters of the language models. Thus, the alignment of the language model is significantly influenced by how well the reward model is trained.

Direct Preference Optimization To bypass the complex RL procedure, Rafailov et al. (2024) proposed the direct preference optimization (DPO) which employs a reward model training objective to maximize rewards:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { D P O } } = - \log \sigma [ \beta \log ( \frac { p _ { \theta ^ { \prime } } \left( y _ { w } | x \right) } { p _ { \theta _ { o l d } ^ { \prime } } \left( y _ { w } | x \right) } ) } \\ & { \qquad - \beta \log ( \frac { p _ { \theta ^ { \prime } } \left( y _ { l } | x \right) } { p _ { \theta _ { o l d } ^ { \prime } } \left( y _ { l } | x \right) } ) ] } \end{array}
$$

where θ′ denotes the parameters of the language model, θ′old denotes the parameters of the language model trained via supervised fine-tuning, $\beta$ denotes a scaling factor, and $\sigma$ denotes a Sigmoid function.

Phase One: Pre-training with Textual Preference Data

![](images/c6a5d048ba0fed10f65dc3da8c262dfe0bc033bdc23e68564d14008a0814bd63.jpg)

P

Bridging Task Gap hase Two:  Fine-tuning with Image Caption-based Preference Data

![](images/8261e25dbdf712e52fcfb4fead7216a25074687c62fab20a17578568e610d6c9.jpg)

Phase Three:  Fine-tuning with Visual Preference Data

Bridging Modality Gap

![](images/8bc6f22cacc64133b79c27e296867b9b549ac7a17a194ada562fb27efb9970a0.jpg)

Instruction: Can you determine the missing number in the sequence: 2, 6, 14, 30, 62, __?

Chosen Response: The missing number in the sequence is 126. Rejected Response: The sequence is 6.

Instruction: Describe the image in detail. Image Caption: There are four mangosteens, one of which has been cut open. Chosen Response: There are several mangosteens on a wooden table. Rejected Response: This is a dark fruit, possibly a variety of pitaya.

![](images/79aa6c24f8e4410b1439379428ec017240a0e28d7264db9a8069ac3beb0d8238.jpg)

Instruction: Describe the image in detail.

Chosen Response: There are several mangosteens on a wooden table. Rejected Response: This is a dark fruit, possibly a variety of pitaya.

![](images/f8b367e449ea5982eddd92063f7343812cad8101c5532e4184e3e41878406150.jpg)  
Figure 1: We propose three-phase progressive training and optimal transport-based preference data selection approaches to train RoVRM. For three-phase progressive training, we take full advantage of textual preference data to compensate for the limited availability of visual preference data. Using this preference selection, samples for phases one and two are selected based on those selected for the subsequent phase. Green ✓denotes a selected sample, while red $\pmb { x }$ denotes one that is not selected.

Best-of- $\mathbf { \nabla } \cdot \boldsymbol { n }$ Sampling Best-of- $\boldsymbol { \cdot } \boldsymbol { n }$ sampling (also known as re-ranking) refers to reordering or reevaluating a set of candidate responses sampled from a trained model (Lee, Auli, and Ranzato 2021; Fernandes et al. 2022). Given a set $y$ of $n$ candidate responses for $x$ , we can also use the best-of- $n$ sampling approach to maximize the reward, thereby aligning the response with human preferences. Typically, we employ the reward model to score the candidate responses and select a final response that has a maximum reward score.

We can notice that when applying these alignment training methods to LVLMs, sufficient visual preference data is required either to train a VRM or to perform DPO training. However, in practice, visual preference data is often insufficient and expensive to acquire.

# A Robust Visual Reward Model

We aim to provide a RoVRM for human-preference alignment in LVLMs. The overview of training RoVRM is depicted in Figure 1. As shown in the figure, we present a three-phase progressive training and preference data selection to improve the robustness of RoVRM.

Three-Phase Progressive Training In response to the scarcity of visual preference data, we propose a three-phase progressive training approach that effectively solves this issue. Phase one is to conduct preference pre-training using a large amount of textual preference data. This phase can help our RoVRM to pre-learn general preferences. Ideally, the RoVRM would inherit these general preferences when processing vision-language tasks. However, this faces two serious obstacles: task gap and modality gap, which prevent these preferences from being directly applicable to visionlanguage tasks (see experiments in Figure 4). Here, we design phases two and three to bridge these gaps progressively. Phase two is to bridge the task gap by constructing vision-language preference data based on image captions and fine-tuning the RoVRM. Specifically, we use image captions to replace the images for visual preference data, i.e., changing the human prompt $x =$ [Instruction; Image] to $x { = } [$ [Instruction; Image Caption] in Eqs. 1 and 2. Building on phase two, phase three is to bridge the modality gap by using the visual preference data to continue finetuning the RoVRM with a visual projector. Compared to training a VRM directly with visual preference data, this three-phase training process incurs additional time costs due to an extra preference training session. However, it can leverage auxiliary textual preference data to improve robustness and respond to the scarcity of visual preference data. Furthermore, although pre-training followed by fine-tuning is widely used in machine learning (Devlin et al. 2019; Liu et al. 2019), our approach is the first to demonstrate the feasibility of optimizing a VRM through this paradigm.

Preference Data Selection Not all preference data aligns with the preferences used in subsequent phases, and conflicts may arise. Thus, during each training phase, we expect to employ samples that more closely align with the preferences contained in the data for the next phase. To achieve this, we propose an optimal transport-based preference data selection approach. We apply this approach to perform preference data selection for phases one and two, based on the preference data used in the next phase. For instance, in phase one, following Xia et al. (2024)’s work, we first extract gradient features for all samples in the textual preference dataset $\mathcal { D } _ { \mathrm { T } } = \{ s _ { 1 } ^ { t } , s _ { 2 } ^ { t } , \cdot \cdot \cdot , s _ { m } ^ { t } \}$ . Based on these features, we compute the distance score between each sample in $\mathcal { D } _ { \mathrm { T } }$ and the image caption-based preference dataset $\mathcal { D } _ { \mathrm { C } } = \{ s _ { 1 } ^ { c } , s _ { 2 } ^ { c } , \cdot \cdot \cdot , s _ { n } ^ { c } \}$ using optimal transport. The details are described as follows.

Gradient Feature. Xia et al. (2024) construct gradient features for each sample of general supervised fine-tuning data to select the data that more effectively improves the specific downstream task. Here, using these gradient features, we conduct the preference data selection. Specifically, we firstly use LoRA ( $\mathrm { H u }$ et al. 2022) to efficiently perform a warmup reward model training with a small subset of preference data ${ \mathcal { D } } _ { \mathrm { W a r m u p } }$ , where ${ \mathcal { D } } _ { \mathrm { W a r m u p } }$ is a subset extracted randomly from $\mathcal { D } _ { \mathrm { T } } \cup \mathcal { D } _ { \mathrm { C } }$ . Then, we extract the gradient features for each preference sample in $\mathcal { D } _ { \mathrm { T } }$ and $\mathcal { D } _ { \mathrm { C } }$ through the forward and backpropagating on the warmed-up reward model:

$$
g = \mathrm { R P } ( \nabla \mathcal { L } _ { r e w a r d } ( s ; \theta _ { w a r m u p } ) )
$$

where $g$ is the gradient feature of the preference sample $s$ and $\theta _ { w a r m u p }$ is the parameters of the warmed-up reward model. $\operatorname { R P } ( { \mathrm { \cdot } } )$ is a random projection (Xie, Li, and Xue 2017) that reduces the dimensionality of gradient features.

Optimal Transport-based Distance. Unlike the Xia et al. (2024) who use the cosine similarity to compute sample distance scores, we use optimal transport (Villani et al. 2009), endowed with the capability to compute the distance transferring an arbitrary data feature to a specific data feature (Gurumoorthy, Jawanpuria, and Mishra 2021; Kang et al. 2024). Our motivation is to gather preference data for easy integration into the next training phase. To reduce computational overhead, we select a representative subset $\mathcal { D } _ { \mathrm { S u b C } }$ from $\mathcal { D } _ { \mathrm { C } }$ . This subset approximates the distance computation for the entire dataset $\mathcal { D } _ { \mathrm { C } }$ when selecting samples from $\mathcal { D } _ { \mathrm { T } }$ . We define the distance score of $i$ -th sample in $\mathcal { D } _ { \mathrm { T } }$ by:

$$
c _ { i } = \frac { 1 } { | \mathscr { D } _ { \mathrm { S u b C } } | } \sum _ { j = 1 } ^ { | \mathscr { D } _ { \mathrm { S u b C } } | } \mathrm { O T } ( g _ { i } ^ { t } , g _ { j } ^ { c } )
$$

where $g _ { i } ^ { t }$ and $g _ { j } ^ { c }$ denote the gradient features for the preference samples $s _ { i } ^ { t }$ and $s _ { j } ^ { c }$ , respectively. $\mathrm { O T } ( \cdot )$ denotes the function of computing the transfer distance. Given gradient features $g _ { i } ^ { t } , \ \bar { g } _ { j } ^ { c }$ over a gradient space $\mathcal { Z }$ , the optimal transport-based transfer distance can be defined as:

$$
{ \mathrm { O T } } ( g _ { i } ^ { t } , g _ { j } ^ { c } ) : = \operatorname* { m i n } _ { \gamma \in \Gamma ( g _ { i } ^ { t } , g _ { j } ^ { c } ) } \int _ { \mathcal { Z } ^ { 2 } } C ( z , z ^ { \prime } ) d \gamma ( z , z ^ { \prime } )
$$

where $C ( \cdot )$ denotes a symmetric positive-definite cost function, and $\dot { \Gamma } ( g _ { i } ^ { t } , g _ { j } ^ { c } )$ denotes a collection of couplings between two gradients $g _ { i } ^ { t }$ and $g _ { j } ^ { c }$ . Here, we utilize $L _ { 2 }$ -norm as the cost function and define the sum of the solved $\gamma$ as the distance score. A lower distance score indicates that the textual preference sample has preferences more easily transferable to the vision-language task. Our implementation of optimal transport solvers is done using Python Optimal Transport $( \mathrm { P O T } ) ^ { \ddagger }$ . While optimal transport distance has been used in data selection before (Kang et al. 2024), this is the first application to preference data selection.

To ensure that the ultimate goal of selecting preference data is to transfer preferences from textual preference data to vision-language tasks, we start by selecting image captionbased preference data for phase two. Next, we choose the textual preference data for phase one based on the preference data selected in phase two.

# Experiments

# Experimental Setups

Datasets The datasets used in this work are as follows:

• Textual Preference Dataset: We used UltraFeedback (Cui et al. 2023), a large-scale, high-quality, and diversified preference dataset, as our textual preference dataset. It comprises $6 4 \mathrm { k }$ instructions, each with 4 responses, leading to over $3 4 0 \mathrm { k }$ comparison preference pairs.

• Image Caption-based Preference Dataset: We constructed an image caption-based preference dataset to bridge the task gap. Specifically, we employed GPT-4o-mini to generate detailed image captions that replace the visual content in our preference data. Note that when the image is present in the COCO caption dataset§, we used the human-annotated captions directly. • Visual Preference Dataset: We employed the visual preference dataset from RLAIF-V (Yu et al. 2024b), which consists of about 83k comparison preference pairs. To our knowledge, it is the largest scale open source preference dataset in computer vision. • RL Training: We sampled $5 0 \mathrm { k }$ instructions from LLaVAInstruct-150K (Liu et al. 2024b) for training.

Settings For training RoVRM, we used the LLaVA-1.5- 7B model to initialize the visual reward model. The learning rates for the three-phase progressive training were set to 2e-5 for phase one, and 1e-6 for phases two and three. For optimal transport-based preference data selection, we used $5 \mathrm { k }$ samples to warm up the VRM, consisting of $2 \mathrm { k }$ samples from the dataset to be selected and $3 \mathrm { k }$ samples from the target preference dataset. The representative subset size was set to $5 \mathrm { k }$ samples. For best-of- $n$ sampling and RL training, we employed the LLaVA-1.5-7B as the initial model. In the process of best-of- $n$ sampling, we set the sampling size to 8. We also tested other sampling sizes in Figure 5.

Evaluation We evaluated the RoVRM in two aspects: trustworthiness, which denotes the level of hallucination, and helpfulness, which reflects overall interaction capability. Trustworthiness was evaluated using two benchmarks: MMHal-Bench (Sun et al. 2023) and AMBER (Wang et al. 2023). GPT-4 was employed to evaluate the responselevel hallucination rate (HalRate) and informativeness score (Score) on the MMHalBench. We also provided the object coverage (Cover.) and hallucination rate metrics for AMBER. To assess helpfulness, we used two benchmarks: MM-Instruct (Liu et al. 2024c) and LLaVA-Bench (In-theWild) (Liu et al. 2024b). GPT-4, following the settings in $1 \mathrm { m m s - e v a l } ^ { \ P }$ , was used to score responses in LLaVABench. For MM-Instruct, responses from LLaVA-1.5-13B were used as a baseline, and we computed the win rate (WinRate) as per Liu et al. (2024c).

<html><body><table><tr><td rowspan="2">Method</td><td rowspan="2">#Param</td><td colspan="2">MMHalBench</td><td colspan="2">AMBER</td><td>LLaVA W</td><td>MMIns</td></tr><tr><td>Score ↑</td><td>HalRate↓</td><td>Cover. ↑</td><td>HalRate↓</td><td>Score ↑</td><td>WinRate ↑</td></tr><tr><td>Qwen-VL-Chat</td><td>10B</td><td>2.76</td><td>38.5</td><td>53.2</td><td>31.0</td><td>71.9</td><td>73.58</td></tr><tr><td>OmniLMM</td><td>12B</td><td>3.14</td><td>36.5</td><td></td><td></td><td>72.7</td><td></td></tr><tr><td>MiniGemini</td><td>34B</td><td>3.08</td><td>38.5</td><td></td><td></td><td>79.2</td><td></td></tr><tr><td>LLaVA-NeXT</td><td>34B</td><td>3.31</td><td>34.4</td><td>63.2</td><td>43.6</td><td>77.7</td><td>93.83</td></tr><tr><td>LURE</td><td>7B</td><td>1.64</td><td>60.4</td><td></td><td></td><td>36.9</td><td></td></tr><tr><td>HA-DPO</td><td>7B</td><td>1.98</td><td>60.4</td><td>49.5</td><td>29.1</td><td>60.3</td><td></td></tr><tr><td>VCD</td><td>7B</td><td>2.12</td><td>54.2</td><td>51.5</td><td>39.0</td><td>65.8</td><td>42.56</td></tr><tr><td>Silkie</td><td>10B</td><td>3.19</td><td>32.3</td><td>56.0</td><td>28.4</td><td>73.2</td><td>63.64</td></tr><tr><td>LLaVA-RLHF</td><td>13B</td><td>2.02</td><td>62.5</td><td>52.0</td><td>39.2</td><td>61.5</td><td>74.24</td></tr><tr><td colspan="8">Best-of-n Sampling</td></tr><tr><td>LLaVA-1.5-7B</td><td>7B</td><td>2.12</td><td>55.0</td><td>50.3</td><td>37.1</td><td>66.7</td><td>46.16</td></tr><tr><td>+VRM-Vanilla</td><td>7B</td><td>2.39</td><td>47.9</td><td>50.8</td><td>29.0</td><td>73.6</td><td>57.69</td></tr><tr><td>+RoVRM-Random</td><td>7B</td><td>2.52</td><td>43.8</td><td>51.7</td><td>26.9</td><td>77.2</td><td>58.49</td></tr><tr><td>+RoVRM</td><td>7B</td><td>2.68</td><td>40.6</td><td>53.2</td><td>23.9</td><td>82.0</td><td>61.91</td></tr><tr><td>LLaVA-1.5-13B</td><td>13B</td><td>2.30</td><td>53.8</td><td>50.6</td><td>37.2</td><td>75.6</td><td>50.00</td></tr><tr><td>+VRM-Vanilla</td><td>13B</td><td>2.41</td><td>51.0</td><td>51.4</td><td>26.6</td><td>84.0</td><td>73.08</td></tr><tr><td>+RoVRM-Random</td><td>13B</td><td>2.43</td><td>48.3</td><td>51.9</td><td>25.7</td><td>86.4</td><td>74.42</td></tr><tr><td>+RoVRM</td><td>13B</td><td>2.57</td><td>47.3</td><td>53.6</td><td>22.8</td><td>89.8</td><td>78.75</td></tr><tr><td colspan="8">Reinforcement Learning</td></tr><tr><td>LLaVA-1.5-7B</td><td>7B</td><td>2.12</td><td>55.0</td><td>50.3</td><td>37.1</td><td>66.7</td><td>46.16</td></tr><tr><td>+VRM-Vanilla</td><td>7B</td><td>2.17</td><td>53.2</td><td>49.1</td><td>29.1</td><td>72.8</td><td>51.11</td></tr><tr><td>+RoVRM-Random</td><td>7B</td><td>2.21</td><td>50.8</td><td>48.7</td><td>24.3</td><td>74.2</td><td>54.35</td></tr><tr><td>+RoVRM</td><td>7B</td><td>2.36</td><td>48.9</td><td>48.2</td><td>23.4</td><td>78.3</td><td>58.69</td></tr><tr><td>LLaVA-1.5-13B</td><td>13B</td><td>2.30</td><td>53.8</td><td>50.6</td><td>37.2</td><td>75.6</td><td>50.00</td></tr><tr><td>+VRM-Vanilla</td><td>13B</td><td>2.49</td><td>50.0</td><td>41.1</td><td>23.2</td><td>78.2</td><td>52.63</td></tr><tr><td>+RoVRM-Random</td><td>13B</td><td>2.34</td><td>47.9</td><td>48.6</td><td>21.0</td><td>79.5</td><td>60.53</td></tr><tr><td>+RoVRM</td><td>13B</td><td>2.57</td><td>43.8</td><td>47.7</td><td>19.5</td><td>81.7</td><td>65.79</td></tr></table></body></html>

Table 1: Experimental results on different vision-language tasks. The best results for each group are in bold.

Baselines Our baselines were the LLaVA-1.5-7B and - 13B models without human-preference alignment. We also compared with other general LVLMs, including Qwen-VLChat (Bai et al. 2023), OmniLMM (Hu et al. 2023), and MiniGemini (Li et al. 2024). Furthermore, we compared RoVRM with commonly used methods to solve the hallucination, including LURE (Zhou et al. 2023), HA-DPO (Zhao et al. 2023), VCD (Leng et al. 2024), Silkie (Li et al. 2023a), and LLaVA-RLHF (Sun et al. 2023). The traditional VRM training was also our baseline, where we optimized a VRM only using our visual preference dataset (VRM-Vanilla). To evaluate the effectiveness of optimal transport, we chose RoVRM-Random as a baseline, where we randomly selected samples during the preference data selection.

# Experimental Results

Results of Best-of- $\cdot n$ Sampling Table 1 summarizes the performance of our RoVRM on the best-of- $n$ sampling. On all vision-language tasks, RoVRM consistently outperforms the VRM-Vanilla which does not use textual preference data. For instance, when using the LLaVA-1.5-7B model, RoVRM can outperform VRM-Vanilla by 8.4 points on the LLaVA-Bench. We also observe this consistent phenomenon on the LLaVA-1.5-13B model. Moreover, from the results, we find that RoVRM significantly reduces visual hallucinations, e.g., lowering the hallucination rate by 13.2 points in the LLaVA-1.5-7B model. We attribute this improvement to the extensive use of textual preference data, which improves VRM’s capacity to evaluate facticity. Interestingly, we also find that RoVRM enables the LLaVA-1.5 models to outperform stronger LVLMs, with the LLaVA

Table 2: The suffixes “-One” and “-Two” denote the removal of phases one and two, respectively, in the three-phase progressive training approach. “w/o PDS” denotes that all data is used for each training phase without employing preference data selection. PDS: preference data selection; TPT: three-phase progressive training.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="2">AMBER</td><td>LLaVA W</td></tr><tr><td>Cover. ↑</td><td>HalRate↓</td><td>Score ↑</td></tr><tr><td>LLaVA-1.5-7B</td><td>50.3</td><td>37.1</td><td>66.7</td></tr><tr><td colspan="4">Best-of-n Sampling</td></tr><tr><td>RoVRM</td><td>53.2</td><td>23.9</td><td>82.0</td></tr><tr><td>W/o PDS</td><td>52.4</td><td>25.1</td><td>80.6</td></tr><tr><td>w/o TPT-One</td><td>51.0</td><td>26.7</td><td>71.3</td></tr><tr><td>w/o TPT-Two</td><td>51.8</td><td>24.9</td><td>78.0</td></tr><tr><td colspan="4">Reinforcement Learning</td></tr><tr><td>RoVRM</td><td>48.2</td><td>23.4</td><td>78.3</td></tr><tr><td>W/o PDS</td><td>46.2</td><td>32.2</td><td>75.2</td></tr><tr><td>w/o TPT-One</td><td>44.3</td><td>35.0</td><td>73.0</td></tr><tr><td>W/o TPT-Two</td><td>47.5</td><td>28.2</td><td>76.1</td></tr></table></body></html>

1.5-7B model even surpassing the LLaVA-1.5-13B model on most of the benchmarks, such as MMHalBench and LLaVABench. This finding shows a promising direction for achieving weak-to-strong generalization (Burns et al. 2023).

Results of Reinforcement Learning Compared to bestof- $\cdot n$ sampling, RL typically requires a more robust reward model: The reward model not only evaluates responses as “good” or “bad” but also provides an accuracy score margin between the responses (Zhou et al. 2024a). From the results, we find that RoVRM fulfills this requirement more effectively than VRM-Vanilla, resulting in improved RL training performance in LVLMs. For instance, in RL training on the LLaVA-1.5-7B model, RoVRM surpasses VRM-Vanilla by 7.58 points on MM-Instruct. This finding demonstrates that RoVRM is robust and can deliver high-quality reward signals across various alignment techniques. Additionally, we observe that RL training reduces hallucinations but slightly decreases the “Cover.” metric, which is consistent with the findings of Meng, Xia, and Chen (2024)’s work and DPO training in Table 3. We conjecture that preference alignment training may slightly hurt the instruction-following capability of LVLMs (Wang et al. 2024a).

Furthermore, compared to RoVRM-Random, RoVRM shows better performance across all benchmarks. This indicates that optimal transport-based preference data selection outperforms random selection. However, RoVRM-Random also significantly improves performance over VRM-Vanilla.

# Ablation Study

We present detailed ablation studies to investigate the effects of three-phase progressive training and our preference data selection approach. The experiments are conducted on the LLaVA-1.5-7B model and the impacts of removing each ap

Best-of- $n$ Sampling Reinforcement Learning 80.0 60.0 78.0 工 1 工   
746.0 工 王 工   
72.0 工 50.0 70.0 工 45.0 5k 10k 20k 40k 5k 10k 20k 40k LLaVAW MM-Instruct (a) Textual Preference Data 82.0 62.0 工 80.0   
78.0 1 60.0   
76.0 74.0 工 工 56.0 72.0 70.0 54.0 5k 10k 20k 40k 5k 10k 20k 40k LLaVAW MM-Instruct (b) Image Caption-based Preference Data

proach were thoroughly examined. Furthermore, we study the impact of eliminating the distinct designs of phases one and two. The results are summarized in Table 2. Through the results, we can see that three-phase progressive training significantly improves the performance of RoVRM in both best-of- $n$ sampling and RL. Notably, removing phase one leads to a substantial performance decline (e.g., a loss of 10.7 points on the LLaVA-Bench for best-of- $n$ sampling), highlighting the importance of textual preference data in training RoVRM. Likewise, removing image caption-based preference data also results in performance loss, indicating the need to address the task gap. Additionally, we see that using the preference data selection can train a better RoVRM. It shows the effectiveness of using optimal transport to conduct preference data selection.

# Analysis

Performance on Different Numbers of Selected Preference Samples We investigate the impact of different numbers of selected preference samples using a three-phase progressive training with LLaVA-Bench and MM-Instruct. We test sample sizes of 5k, 10k, $2 0 \mathrm { k }$ , and $4 0 \mathrm { k }$ , alongside $2 0 \mathrm { k }$ image caption-based preference samples (Figure 2(a)). Our results show that using $2 0 \mathrm { k }$ textual preference samples yields strong performance, even outperforming the 40k sample scenario. Consequently, we choose $2 0 \mathrm { k }$ textual preference samples for phase one to train our RoVRM. Similarly, we evalu

LLaVA-1.5-7B RL w/ RoVRM RL w/ VRM-Vanilla 60.0 80.0 50.0 75.0 70.0 45.0 65.0 100 300 500 700 900 100 300 500 700 900 Training Step Training Step

BoS w/ VRM-Vanilla RL w/ VRM-Vanilla BoS w/ RoVRM RL w/ RoVRM 60.0 85.0   
55.0 80.0   
4550.0 75.0 00 70.0 40.0 65.0 0k 10k 20k 30k 40k 0k 10k 20k 30k 40k Number of VPD Number of VPD

ate sample sizes of 5k, 10k, 20k, and $4 0 \mathrm { k }$ for phase two, i.e., image caption-based preference data selection (Figure 2(b)), identifying $1 0 \mathrm { k }$ as the optimal sample size.

Comparison of RL Training Process on Different VRMs Figure 3 illustrates the performance of the LLaVA-1.5- 7B model comparing RL training with VRM-Vanilla and RoVRM. The results show that RL training with RoVRM improves performance more effectively than VRM-Vanilla. Additionally, we observe that RoVRM can lead to a more stable RL training process by mitigating reward overoptimization (Gao, Schulman, and Hilton 2023).

Table 3: Performance on the direct preference optimization.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="2">AMBER</td><td>LLaVA W</td></tr><tr><td>Cover. ↑</td><td>HalRate↓</td><td>Score ↑</td></tr><tr><td>LLaVA-1.5-7B</td><td>50.3</td><td>37.1</td><td>66.7</td></tr><tr><td rowspan="2">+DPO +RoDPO</td><td>49.6</td><td>22.2</td><td>80.9</td></tr><tr><td>50.7</td><td>17.6</td><td>83.7</td></tr><tr><td>LLaVA-1.5-13B</td><td>50.6</td><td>37.2</td><td>75.6</td></tr><tr><td>+DPO</td><td>49.2</td><td>15.7</td><td>84.2</td></tr><tr><td>+RoDPO</td><td>49.8</td><td>12.8</td><td>86.4</td></tr></table></body></html>

Enabling Few-Shot Learning in VRM Figure 4 shows RoVRM’s performance with different numbers of visual preference data. Note that when the visual preference dataset is small (i.e., 1k, 5k, and 10k), we use the entire dataset without image caption-based preference data selection. From the results, we find that pre-training with textual preference data enables effective few-shot learning in VRM (Wang et al. 2020). Based on these textual preferences, the reward model quickly generalizes to vision-language tasks using only a few visual preference samples. Notably, using only 5k visual preference samples can achieve a performance comparable to that of VRM-Vanilla trained with 83k samples. However, while it is feasible to directly use a textual reward model (i.e., using 0k visual preference data) to optimize LVLM,

BoS w/ RoVRM BoS w/ VRM-Vanilla 60.0 55.0 85.0 45050.0 80.0 GU 75.0 35.0 70.0 30.0 4 8 16 32 4 8 16 32 Sampling Size Sampling Size the results are worse, particularly during RL training.

Integration with Direct Preference Optimisation Despite bypassing reward model training, direct preference optimization (DPO) still requires preference data to train the language model with a ranking-loss function. Consequently, DPO also faces the challenge of limited visual preference data in LVLMs. To address this, we propose a Robust DPO (namely RoDPO) by integrating our three-phase progressive training and preference data selection. Our experiments on the LLaVA-1.5-7B and -13B models show that RoDPO performs better than DPO, as summarized in Table 3.

Performance on Different Sampling Sizes We evaluate the performance of best-of- $n$ sampling with varying sample sizes using the LLaVA-1.5-7B model. Figure 5 presents a comparison of RoVRM and VRM-Vanilla on the MMHalBench (left) and LLaVA-Bench (right) benchmarks. The experimental results indicate that RoVRM consistently enhances performance across different sampling sizes, highlighting its improved robustness.

# Conclusion

In this paper, we focus on improving the human-preference alignment for LVLMs. We present a Robust Visual Reward Model (namely RoVRM) via three-phase progressive training and preference data selection approaches. Our extensive experiments demonstrate that our RoVRM significantly outperforms the traditional visual reward model.