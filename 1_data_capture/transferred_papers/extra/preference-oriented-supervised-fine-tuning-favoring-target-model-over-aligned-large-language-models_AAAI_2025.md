# Preference-Oriented Supervised Fine-Tuning: Favoring Target Model over Aligned Large Language Models

Yuchen Fan, Yuzhong Hong, Qiushi Wang, Junwei Bao\*, Hongfei Jiang, Yang Song

Zuoyebang Education Technology (Beijing) Co., Ltd {fanyuchen02, hongyuzhong, wangqiushi02, jianghongfei, songyang}@zuoyebang.com baojunwei001 $@$ gmail.com

# Abstract

Alignment, endowing a pre-trained Large language model (LLM) with the ability to follow instructions, is crucial for its real-world applications. Conventional supervised fine-tuning (SFT) methods formalize it as causal language modeling typically with a cross-entropy objective, requiring a large amount of high-quality instruction-response pairs. However, the quality of widely used SFT datasets can not be guaranteed due to the high cost and intensive labor for the creation and maintenance in practice. To overcome the limitations associated with the quality of SFT datasets, we introduce a novel preference-oriented supervised fine-tuning approach, namely PoFT. The intuition is to boost SFT by imposing a particular preference: favoring the target model over aligned $L L M s$ on the same SFT data. This preference encourages the target model to predict a higher likelihood than that predicted by the aligned LLMs, incorporating assessment information on data quality (i.e., predicted likelihood by the aligned LLMs) into the training process. Extensive experiments are conducted, and the results validate the effectiveness of the proposed method. PoFT achieves stable and consistent improvements over the SFT baselines across different training datasets and base models. Moreover, we prove that PoFT can be integrated with existing SFT data filtering methods to achieve better performance, and further improved by following preference optimization procedures, such as DPO.

# Code — https://github.com/Savannah120/alignmenthandbook-PoFT/

# 1 Introduction

Large language models(LLMs) such as ChatGPT (OpenAI et al. 2024) have exhibited successful and potent applications in comprehending human queries and delivering plausible responses. This ability has proven to be crucial in realworld applications, e.g. AI assistants and recommendation systems. To equip LLMs with this ability, the alignment methods are usually applied to pre-trained language models. Alignment enables pre-trained models to comprehend the context and generate responses suitable to human interactions. Typical alignment methods can be broadly categorized into two types: Supervised Fine-Tuning (SFT) and Preference Alignment (PA).

![](images/26e0d9be0d4cf9b89d369466df014d196f7b539dd20edd3b2605c9b37de2e58b.jpg)  
Figure 1: The overall modeling framework of PoFT. By leveraging the Bradley-Terry ranking objective, we impose a particular preference that favors the target model over the aligned LLMs on the same SFT data. Note that the preference score is generated based on the corresponding predicted likelihood.

Supervised fine-tuning (SFT) is an essential phase of alignment, wherein the task is framed as causal language modeling performed on a pre-trained language model with instruction-response data $\bar { \mathcal { D } } = \{ \langle x , y \rangle \}$ . Generally, it leverages the cross-entropy objective function in optimization, equipping the pre-trained language model with the ability to follow instructions and generate coherent sequences. Several studies (Schick and Schu¨tze 2021; Houlsby et al. 2019; Ivison et al. 2023) are dedicated to exploring SFT training strategies to enhance the alignment of LLMs. However, due to the intrinsic traits of modeling, the optimization process heavily depends on the availability of high-quality $\langle x , y \rangle$ data, which hinders its performance. Traditionally, the prevalent large-scale SFT datasets in earlier research, such as Alpaca (Taori et al. 2023) and ShareGPT (shareAI 2023), were mainly developed via AI distillation or human-andAI interaction. Assuring the quality of these datasets can be challenging, as the filtration and curation processes demand significant human resources and efforts.

Instead of solely aligning the instruction and responses, preference alignment (PA), such as InstructGPT (Ouyang et al. 2022) and Direct Preference Optimization (DPO) (Rafailov et al. 2023), optimizes the LLMs based on chosenrejected data $\langle x , y ^ { + } , y ^ { - } \rangle$ . These PA methods provide exceptional benefits in model alignment, enabling LLMs to align more accurately with AI/human preferences. In particular, DPO employs the Bradley-Terry (BT) ranking objective (Bradley and Terry 1952) in its optimization process to perform direct preference comparison.

Given the limitations of SFT in processing quality-limited data, we leverage the benefits of the BT preference model and incorporate it into the SFT framework, by proposing a Preference-oriented supervised Fine-Tuning method, called PoFT. Specifically, it applies the BT objective to different models by imposing a particular preference: favoring the target model over the aligned LLMs, given the same $\langle x , y \rangle$ data. Within this framework, the aligned LLMs act as baselines for the target model, prompting it to attain higher preference scores than that of the aligned LLMs on SFT data. Here, we assume these LLMs could discern data that contribute positively to model optimization, thereby providing valid data quality assessments, inspired by (Ngo et al. 2021). Moreover, we would like to emphasize that we use the BT model to rank models rather than to rank data. This means we are fundamentally not a PA approach but rather an SFT approach since we require only $\langle x , y \rangle$ and not $\langle x , y ^ { + } , y ^ { - } \rangle$ . For that matter, we show our approach is indeed orthogonal to PA since PoFT can be combined with PA methods to further enhance the overall alignment performance (e.g., first PoFT and then DPO).

Despite leveraging the preference modeling with BT, at its essence, PoFT remains faithful to the SFT paradigm, relying on instruction-response data. As an enhanced SFT method, PoFT’s objective offers a remarkable advantage over the conventional SFT objective cross-entropy (CE), i.e., PoFT is more stable and robust when training with quality-limited data. Specifically, the introduction of aligned LLMs provides quality assessments on each sample $\langle x , y \rangle$ , which decreases its sensitivity towards the data quality. In practice, by analyzing the gradient updates, we observe that PoFT assigns dynamic weights (namely coefficient defined in section 3) to different samples $\{ \langle \overset { \cdot } { x } , y \rangle \}$ by the aligned LLMs. These weights guide parameter optimization, reducing the negative effect of low-quality data. In contrast, the CE objective treats all the data equally, without differentiating data samples based on their quality, thus exposing it to vulnerabilities to low-quality data.

In summary, our contributions are three-fold:

• Innovative SFT Training Methodology With Preference Modeling. We present a novel method, called PoFT. This new methodology effortlessly integrates aligned LLMs for preference modeling - a fresh perspective that leads to a boost in the optimization process. • Analytical Insight into PoFT’s Stability. Through rigorous mathematical analysis, we provide theoretical explanations that shed light on the inherent characteristics of PoFT in gradient update. • Comprehensive Validation of Methodology. We validate the effectiveness of PoFT through extensive experiments on different base models, demonstrating that PoFT achieves superior performance over the CE objective across diverse training datasets. Our ablation studies indicate PoFT’s stability over increasing epochs and enhanced resilience to noise data. Impressively, our experiments prove that the integration of the PoFT and SFT filtering methods can lead to further performance enhancement. Moreover, the two-step training followed by DPO also shows promising alignment performance.

# 2 Related Work

# 2.1 Supervised Fine-Tuning

Enabling pre-trained language models to follow human instructions, supervised fine-tuning (SFT) is a way to align LLMs’ behavior with human desirability, by training on instruction-response data in a supervised fashion.

Dataset Construction Efforts have been made to construct diverse and complex training data, such as Orca (Mukherjee et al. 2023) and WizardLM (Xu et al. 2023). Wang et al. (2023) proposed a self-improvement pipeline, which enhances LLMs by using its own generations as a bootstrap. Rather than based on human-provided instructions, Li et al. (2024d) reversely constructed instructions from the web corpus via a back-translation model.

Data Filtering In addition to enhancing data complexity, some studies focus on data filtering to improve training efficiency (Chen et al. 2024a; Lu et al. 2024; Liu et al. 2024; Du, Zong, and Zhang 2023). Lu et al. (2024) trained a tagger based on semantics and intentions and regarded the number of tags as a complexity indicator for filtering. IFD, proposed by Li et al. (2024c), is a complexity metric that identifies the discrepancies between responses and the model’s generation capability. Liu et al. (2024) trained a scorer via ChatGPT to assess the complexity and quality of the data, thereby selecting “good” data. Li et al. (2024a) and Li et al. (2024b) leveraged a student model to select data for training a teacher model based on the IFD scores.

FT strategies Multiple works have explored efficient fine-tuning strategies to enhance the alignment process (Schick and Schu¨tze 2021; Houlsby et al. 2019; J. et al. 2021; Ivison et al. 2023). Schick and Schu¨tze (2021) converted the provided input into cloze-style statements, thereby facilitating language models to understand the tasks. Ivison et al. (2023) transformed the instructions and examples of a task into parameter-efficient modules through an extra text encoder. Different from these strategies, PoFT proposes a training objective by modeling preference between the target model and aligned LLMs, providing a fresh perspective to enhance the optimization process.

# 2.2 Preference Alignment

By aligning training objectives with human/AI preferences, RLHF/RLAIF are particularly useful in applications that require nuanced and context-aware decisions (Ouyang et al. 2022; OpenAI et al. 2024; Bai et al. 2022). A prominent preference alignment approach is Direct Preference Optimization (DPO) (Rafailov et al. 2023), which leverages Bradley-Terry (BT) ranking objective (Bradley and Terry 1952) to better prioritize actions based on perceived desirability. In general, the BT model estimates the probability of one item $i$ being chosen over another $j$ in a pairwise comparison, where the items are quantified with strength or quality parameters, denoted as $\lambda _ { i }$ and $\lambda _ { j }$ respectively, resulting in:

$$
\mathcal { P } ( i \succ j ) = \frac { \lambda _ { i } } { \lambda _ { i } + \lambda _ { j } } .
$$

As for DPO, it applies the BT objective to express preferences of the policy model for the chosen-rejected pairs $\langle x , y ^ { + } , y ^ { - } \rangle$ via their expected rewards. Therefore, the preference distribution can be written as:

$$
\begin{array} { l } { \displaystyle { \mathcal { P } \left( y ^ { + } \succ y ^ { - } \mid x \right) = \sigma ( ( r \left( x , y ^ { + } \right) - ( r \left( x , y ^ { - } \right) ) \right. } } \\ { \displaystyle { \phantom { \left. \sum _ { k } \int _ { } ^ { } } } = \frac { \exp { ( r \left( x , y ^ { + } \right) ) } } { \exp { \left( r \left( x , y ^ { + } \right) \right) } + \exp { \left( r \left( x , y ^ { - } \right) \right) } } , } \end{array}
$$

where $r \left( x , y \right)$ is a closed-form reward expression with the optimal policy in DPO’s context. Subsequently, more methods are proposed to improve the preference optimization process (Yuan et al. 2023; Dong et al. 2023; Song et al. 2024; Chen et al. 2024b).

# 3 Methodology

# 3.1 Preliminary

Typically, the cross-entropy(CE) objective for SFT training only minimizes the difference between predicted and true distributions, represented as

$$
L _ { \mathrm { C E } } = - \frac { 1 } { T _ { 0 } ( y ) } \log p _ { \theta } ( y | x ) ,
$$

where $T _ { 0 } ( y )$ refers to the length of $y$ tokenized by the target model $\theta$ . Its gradient is shown in Eq. 4.

$$
\nabla _ { \theta } L _ { \mathrm { C E } } = - \frac { 1 } { T _ { 0 } ( y ) } \frac { 1 } { p _ { \theta } ( y \mid x ) } \nabla p _ { \theta } ( y \mid x )
$$

# 3.2 PoFT: Preference-oriented Supervised Fine-Tuning

In this section, we introduce a novel preference-oriented fine-tuning objective that applies the Bradely-Terry model to perform preference modeling between the target model and aligned LLMs, namely PoFT. Given data $\{ x , y \} \sim \mathcal { D } _ { S F T }$ , it imposes a particular preference by prioritizing the target model over the aligned LLMs. Accordingly, the aligned LLMS acts as a reference point guiding the target model to generate higher preference scores. The preference score is generated based on the predicted likelihood, thus the one from aligned LLMs can be regarded as an indicator for estimating the data quality. Assigning such a preference could diversify the effects of the SFT data, emphasizing more on high-quality data in the optimization process. Note that the preferences are supposed to be generated by some reward model $r ^ { * } ( x , y )$ . Consequently, by applying the BT model, the preference distributions $\mathcal { P } ( \cdot )$ can be defined as:

$$
\begin{array} { r } { \mathcal { P } \left( r ^ { * } \left( x , y \right) \succ r _ { \mathrm { L L M s } } \left( x , y \right) \mid x , y \right) } \\ { = \frac { \exp \left( r ^ { * } \left( x , y \right) \right) } { \exp \left( r ^ { * } \left( x , y \right) \right) + \exp \left( r _ { \mathrm { L L M s } } \left( x , y \right) \right) } , } \\ { r _ { \mathrm { L L M s } } ( x , y ) = \mathbb { E } _ { \mathrm { L L M } \sim \mathcal { D } _ { \mathrm { L L M s } } } \left[ r _ { \mathrm { L L M } } ( x , y ) \right] . } \end{array}
$$

When accessing a static SFT dataset, a number of aligned LLMs (denoted as $\mathbf { L L M } _ { j } \in \mathcal { D } _ { \mathbf { L L M } }$ , $| { \mathcal { D } } | = M )$ , and a parameterized reward model $r _ { \theta } ( x , y )$ for $r ^ { * } ( x , y )$ , the training objective can be transformed into a binary classification problem via maximum likelihood:

$$
\begin{array} { l } { { \displaystyle { \cal L } _ { \mathrm { P o F T } } ( \theta ) } } \\ { { \displaystyle ~ = - \mathbb { E } _ { ( x , y ) \sim { \mathcal D } _ { \mathrm { S F T } } , r _ { \theta } ( x , y ) \succ r _ { \mathrm { L I M s } } ( x , y ) \sim { \mathcal P } ( \cdot ) } [ \log { \mathcal P } _ { \theta } ( \cdot ) ] } } \\ { { \displaystyle ~ \approx - \mathbb { E } _ { ( x , y ) \sim { \mathcal D } _ { \mathrm { S F T } } } [ \log \frac { \exp { \big ( r _ { \theta } ( x , y ) \big ) } } { \exp { \big ( r _ { \theta } ( x , y ) \big ) } + \exp { \big ( r _ { \mathrm { L I M s } } ( x , y ) \big ) } } ] } } \\ { { \displaystyle ~ = - \mathbb { E } _ { ( x , y ) \sim { \mathcal D } _ { \mathrm { S F T } } } [ \log \sigma ( \frac { 1 } { M } \sum _ { j = 1 } ^ { M } ( r _ { \theta } ( x , y ) - r _ { j } ( x , y ) ) ) ] , } } \end{array}
$$

where we first impose a particular preference $\mathcal { P } \longrightarrow 1$ (hence the $\approx$ ) and then parameterize the BT model (i.e., $\mathcal { P } _ { \theta } ( \cdot ) )$ ) using rewards defined as follows:

$$
\begin{array} { l } { { r _ { \theta } ( x , y ) = \displaystyle \frac { 1 } { T _ { 0 } ( y ) } \log p _ { \theta } ( y \mid x ) , } } \\ { { r _ { j } ( x , y ) = \displaystyle \frac { 1 } { T _ { j } ( y ) } \log p _ { j } ( y \mid x ) . } } \end{array}
$$

In our context, we leverage the logarithm of predicted likelihood $p ( y | x )$ with length normalization as the reward function to generate preference scores. Specifically, the logarithm of the predicted likelihood for the target model $\log p \theta ( y | x )$ and the $j$ -th aligned LLM $\log p _ { j } ( y | \bar { x } )$ are normalized by the corresponding length of the tokenized $y$ , i.e., $T _ { 0 } ( y )$ and $T _ { j } ( y )$ respectively. This preference score measures how likely a model would generate the response $y$ when given $x$ at a token level. Applying the length normalization effectively addresses issues related to tokenization mismatches. Moreover, as demonstrated in (Meng, Xia, and Chen 2024), length normalization can also mitigate the impact of sequence length on the reward.

$$
\nabla _ { \theta } L _ { \mathrm { P o F T } } = - \frac { 1 } { T _ { 0 } ( y ) } \frac { 1 } { p _ { \theta } \left( y \mid x \right) } \tau \nabla p _ { \theta } \left( y \mid x \right) ,
$$

where

$$
\tau = \frac { \Bigg ( \prod _ { j = 1 } ^ { M } p _ { j } ( y \mid x ) ^ { \frac { 1 } { T _ { j } ( y ) } } \Bigg ) ^ { \frac { 1 } { M } } } { \Bigg ( \prod _ { j = 1 } ^ { M } p _ { j } ( y \mid x ) ^ { \frac { 1 } { T _ { j } ( y ) } } \Bigg ) ^ { \frac { 1 } { M } } + p _ { \theta } ( y \mid x ) ^ { \frac { 1 } { T _ { 0 } ( y ) } } } .
$$

To delve deeper into the behavior of PoFT during optimization, we examine and present the gradients for CE and PoFT loss, shown in Eq.4 and Eq.7, respectively. By comparison, it can be observed that PoFT’s gradient contains an extra coefficient, which is outlined in Eq. 8. This coefficient indicates that the gradient is positively related to $p _ { j } ( y | x )$ , which indicates the assessment of $\langle x , y \rangle$ from the aligned LLMs. Intuitively, it allows for a more nuanced and dynamic optimization process, accounting for the unbalanced quality of the SFT datasets. Instead of assigning equal weights to all data, PoFT utilizes the aligned LLMs to direct optimization by diversifying the impacts of different samples on the gradient update. Accordingly, PoFT is proficient in alleviating the influence of lower-quality data, concentrating focus on data with a higher preference score. Thus, PoFT demonstrates its stability for the quality-limited data, compared to the conventional SFT methods.

# 4 Experiment

In this section, we present the main results of our experiments, highlighting the improvements achieved by PoFT across various datasets. Additionally, our ablation studies offer insights into the following aspects: (1) the effectiveness of PoFT on quality-limited data, (2) the comparison between PoFT and data filtering methods, and (3) the comparison between PoFT and data distillation from aligned LLMs.

# 4.1 Settings

• Training Data To align with Zephyr-7B-sft-full (Tunstall et al. 2023), we opt for UltraChat200k (Ding et al. 2023) as the primary training dataset for PoFT. Besides, we also employ the ShareGPT-Chinese-English90k dataset (shareAI 2023) and OpenHermes dataset (Teknium 2023), which encompasses 240k data pairs. As ShareGPT comprises parallel bilingual data, we exclusively utilize the English corpus for training purposes. Moreover, to examine PoFT’s compatibility with DPO, we introduce the UltraFeedback (Cui et al. 2023) dataset for two-step training.

• Benchmarks We evaluate models on the popular benchmark Huggingface Open LLM Leaderboard (Aidar Myrzakhan 2024), MT-Bench (Zheng et al. 2023) and AlpacaEval2.0 (Li et al. 2023). Open LLM Leaderboard covers a variety of tasks, enabling the assessment of specific capabilities of LLMs. Both MT Bench and AlpacaEval 2.0 applied GPT-4 as the judge model to assess the model performance.

• Model We choose Mistral-7B-v0.1 (Jiang et al. 2023) and Llama-3-8B (AI@Meta 2024) as backbones. For aligned LLMs, we adopt zephyr-7b-sft-full (Tunstall et al. 2023), Llama-3-8B-Instruct (AI $@$ Meta 2024), and Yi-6B-Chat (AI et al. 2024). Notably, Zephyr-7B-sft-full, derived from Mistral-7B-v0.1, trained on UltraChat200k.

# 4.2 Main Experiment

We adopt the base models trained on the cross-entropy (CE) objective as our baseline (i.e., SFT model) and investigate the effectiveness of PoFT under the same training settings. The experiments are conducted mainly on UltraChat200k, OpenHermes, and ShareGPT datasets.

Table 1 contains the experimental results of comparison between the models with different training objectives on the LLM Open Leaderboard. To ensure a fair evaluation, we report the results of the last epoch and the average scores across all training epochs after excluding the first epoch, which is typically considered unstable. Notably, as the same base model and datasets are used by Zephyr-7B-sft-full, by

1.75 UItraChat200k 1.50 OpenHermes 1.25 ShareGPT 01.0 0.50 0.25 0.004.0-3.5-3.0-2.5-20-1.5-10-0.5 0.0 Average Preference Score

adjusting hyper-parameters, it could achieve better performance (see the fourth row of Table 1).

Both scores on the Open LLM leaderboard show a consistent trend in which PoFT systematically outperforms the CE objectives across various training datasets on different base models. And the gap is more pronounced concerning OpenHermes, by 1.58 and 2.17 on Mistal-7B and Llama-3- 8B, respectively. Moreover, PoFT models have a comparable lower standard deviation than SFT models, indicating greater stability of PoFT across different training epochs. In terms of different evaluation datasets, PoFT models distinctly outperform SFT models on the GSM8k dataset.

The results for MT-Bench and AlpacaEval 2.0 echo the findings from the LLM Open Leaderboard, with remarkable improvements being made in OpenHermes, shown in Table 2. Nonetheless, the overall discrepancy between SFT and PoFT models is fairly minor. We attribute this to the evaluation perspectives of these benchmarks, such as helpfulness in human preference.

Since model performances on different training data are varied, we analyze the average preference scores distribution of aligned LLMs on three training datasets, depicted in Figure 2. It is observed that the distributions of UltraChat200k and ShareGPT are more concentrated, while the distribution of OpenHermes is wider and flatter in shape. This implies the discrepancy of gradients on OpenHermes is more diverse during the training process, thereby amplifying the difference in training performance between CE and PoFT objectives. Hence, we can hypothesize that PoFT is inclined to a certain type of data distribution. In other words, under this data distribution, our model can leverage its strengths more effectively than the SFT model. A comprehensive discussion regarding this observation is covered in section 4.3.

In addition to comparing our approach with SFT methods, we also investigate the compatibility between PoFT and DPO. The results in Table 3 demonstrate that integrating PoFT and DPO can yield superior performance across all benchmark tasks. It is worth noticing that this combined approach brings a significant improvement on the AlpacaEval benchmark, with the win rate surging to $2 7 . 8 3 \%$ , underscoring the effectiveness of PoFT-DPO synergy.

Table 1: Overall performance on LLM Open Leaderboard of Mistral-7B and Llama-3-8B training on UltraCha $2 0 0 \mathrm { k }$ , OpenHermes, and ShareGPT. The last three columns present the results of the last epoch and the average scores and standard deviation across all epochs, respectively. We also present the results of the aligned LLMs, where Zephyr†, Llama3-8B†, and $\mathrm { Y i } { - } 6 \mathbf { B } ^ { \dagger }$ stand for Zephyr-7B-sft-full, Llama-3-8B-Instruct, and Yi-6B-Chat respectively.   

<html><body><table><tr><td>Base</td><td>FT</td><td>Datasets</td><td>Arc</td><td>Truthful.</td><td>Wino.</td><td>GSM8k</td><td>HellaS.</td><td>MMLU</td><td>Overall</td><td>Avg.</td><td>Std.</td></tr><tr><td>Zephyrt</td><td></td><td>UltraChat</td><td>58.10</td><td>40.30</td><td>76.90</td><td>34.64</td><td>80.95</td><td>58.92</td><td>58.17</td><td></td><td></td></tr><tr><td>Llama3-8B†</td><td></td><td></td><td>62.03</td><td>51.64</td><td>75.30</td><td>75.44</td><td>78.78</td><td>65.75</td><td>68.16</td><td></td><td></td></tr><tr><td>Yi-6B†</td><td>1</td><td></td><td>57.51</td><td>50.01</td><td>71.98</td><td>40.63</td><td>78.48</td><td>63.17</td><td>60.30</td><td></td><td></td></tr><tr><td>Mistal-7B</td><td>SFT</td><td></td><td>63.31</td><td>49.13</td><td>78.77</td><td>42.53</td><td>83.79</td><td>62.04</td><td>63.26</td><td>63.34</td><td>0.09</td></tr><tr><td>Mistal-7B</td><td>PoFT</td><td>UltraChat</td><td>63.40</td><td>49.46</td><td>78.77</td><td>44.88</td><td>83.83</td><td>62.10</td><td>63.74↑0.48</td><td>63.710.38</td><td>0.04</td></tr><tr><td>Llama3-8B</td><td>SFT</td><td></td><td>60.84</td><td>54.97</td><td>78.30</td><td>53.22</td><td>81.91</td><td>65.03</td><td>65.71</td><td>65.65</td><td>0.06</td></tr><tr><td>Llama3-8B</td><td>PoFT</td><td></td><td>60.92</td><td>55.09</td><td>78.14</td><td>54.13</td><td>82.03</td><td>65.10</td><td>65.90个0.19</td><td>65.88↑0.23</td><td>0.10</td></tr><tr><td>Mistal-7B</td><td>SFT</td><td></td><td>62.54</td><td>51.45</td><td>78.14</td><td>37.98</td><td>82.18</td><td>59.34</td><td>61.94</td><td>60.86</td><td>1.62</td></tr><tr><td>Mistal-7B</td><td>PoFT</td><td>Open-</td><td>63.57</td><td>52.81</td><td>77.51</td><td>43.82</td><td>82.88</td><td>60.55</td><td>63.52个1.58</td><td>63.012.15</td><td>0.50</td></tr><tr><td>Llama3-8B</td><td>SFT</td><td>Hermes</td><td>59.22</td><td>56.84</td><td>73.28</td><td>47.31</td><td>80.34</td><td>61.38</td><td>63.06</td><td>64.25</td><td>0.81</td></tr><tr><td>Llama3-8B</td><td>PoFT</td><td></td><td>61.43</td><td>58.38</td><td>75.77</td><td>50.72</td><td>81.63</td><td>63.42</td><td>65.23个2.17</td><td>65.36个1.11</td><td>0.15</td></tr><tr><td>Mistal-7B</td><td>SFT</td><td></td><td>61.43</td><td>52.69</td><td>78.85</td><td>42.99</td><td>83.9</td><td>62.18</td><td>63.67</td><td>63.66</td><td>0.10</td></tr><tr><td>Mistal-7B</td><td>PoFT</td><td>ShareGPT</td><td>61.86</td><td>52.74</td><td>78.45</td><td>45.19</td><td>84.02</td><td>62.21</td><td>64.08↑0.41</td><td>64.00个0.34</td><td>0.10</td></tr><tr><td>Llama3-8B</td><td>SFT</td><td></td><td>58.28</td><td>54.93</td><td>77.82</td><td>54.59</td><td>81.71</td><td>65.33</td><td>65.44</td><td>65.34</td><td>0.11</td></tr><tr><td>Llama3-8B</td><td>PoFT</td><td></td><td>57.76</td><td>55.07</td><td>78.30</td><td>55.95</td><td>81.75</td><td>65.15</td><td>65.66个0.22</td><td>65.45个0.11</td><td>0.19</td></tr></table></body></html>

<html><body><table><tr><td rowspan="2">FT</td><td rowspan="2">Datasets</td><td colspan="2">MT-Bench</td><td colspan="2">AlpacaEval(%)</td></tr><tr><td>Last</td><td>Avg.</td><td>Last</td><td>Avg</td></tr><tr><td>Zephyr SFT</td><td>UltraChat</td><td>6.30 6.35</td><td></td><td>3.91</td><td>1</td></tr><tr><td>PoFT</td><td></td><td>6.52个0.17</td><td>6.05 6.12</td><td>3.98 4.1↑0.12</td><td>3.93 4.35</td></tr><tr><td>SFT PoFT</td><td>Open- Hermes</td><td>5.09 5.93↑0.84</td><td>5.16 5.89</td><td>4.86 5.96个1.1</td><td>3.99 4.57</td></tr><tr><td>SFT PoFT</td><td>ShareGPT</td><td>6.63 6.83↑0.20</td><td>6.44 6.60</td><td>2.44 2.61↑0.17</td><td>2.09 2.55</td></tr></table></body></html>

Table 2: Overall performance on MT-Bench and AlpacaEval 2.0 of Mistral-7B training on three datasets. The last two columns of each benchmark present the results of the last epoch and the average scores across all epochs, respectively. Specifically, the score for AlpacaEval 2.0 is the win rate $( \% )$ .

# 4.3 Ablation Study

Effectiveness on Quality-limited Data The CrossEntropy (CE) objective is vulnerable to poor data quality as it does not differentiate between high and low-quality data. In contrast, by integrating aligned LLMs, PoFT can diversify the impacts of data during the optimization process.

However, there is a disparity in the improvements of PoFT for different datasets. By observing the preference score distribution in Figure 2, we assume that this disparity could be attributed to the distribution of training data. Intuitively, when the distribution is highly concentrated, the gap between SFT and PoFT diminishes as the weights for different samples are less diverged. This leads us to our assumption that, upon training on a dataset with a more diverse preference score distribution, a more significant enhancement in PoFT could be observable over the SFT model.

To verify our assumption, we conduct experiments on the datasets Alpaca (Taori et al. 2023) and Dolly (Conover et al. 2023), which are regarded as quality-limited datasets (Li et al. 2024d; Lu et al. 2024). Note that we intentionally increase the number of training epochs to ten for a more nuanced observation of the effects over an extended period. Figure 3b and Figure 3c depict the performance of Mistral-7B models training with these two datasets respectively. During the initial epochs, there is a significant drop in both models. We attribute this to the significant percentage of noise data within the datasets. Nevertheless, PoFT is more robust, proven by the consistent improvement over subsequent epochs. Meanwhile, SFT models are underperformed, indicated by a decreasing trend.

To present the noise data more intuitively, we directly construct hand-crafted noise data to increase the data in the long-tail part of the preference score distribution. Utilizing OpenHermes as our source, we create a pair of inputs with a randomly matched output and simulate data corruption through the processes of character insertion, deletion, and modification, yielding $5 0 \mathrm { k }$ noise data. The newly created noise data is blended with the original data for training. Figure 3d presents the distribution of new training data.

Figure 3e elaborates the performance of Mistral-7B trained on the noise data. For comparison, we also display the performance of models trained on the original data under the same training settings. Overall, the PoFT models consistently surpass the performance of the SFT models regardless of the data settings. It is worth noting that the gap between the PoFT and SFT models is widened when trained with the noise data. As the training epoch increases, there is a remarkable drop in SFT models, particularly for the one with noise data. This indicates that SFT training is more likely

<html><body><table><tr><td rowspan="2">Model</td><td rowspan="2">Datasets</td><td rowspan="2">LLMOpenLeaderboard Avg. Std.</td><td rowspan="2"></td><td colspan="2">MT-Bench</td><td rowspan="2">AlpacaEval2.0 Avg.(%) Std.</td></tr><tr><td>Avg.</td><td>Std.</td></tr><tr><td>Zephyr+DPO</td><td rowspan="2">+UltraFeedback</td><td>62.62</td><td></td><td>7.11</td><td>19.01</td><td></td></tr><tr><td>SFT+DPO</td><td rowspan="2">65.18</td><td>0.32</td><td>6.84</td><td>0.25</td><td>25.09 1.72</td></tr><tr><td>PoFT+DPO</td><td>65.88 ↑0.70</td><td>0.12</td><td>7.04↑0.20</td><td>0.08 27.832.74</td><td>3.06</td></tr></table></body></html>

Table 3: Performance of two-step training models based on Mistral-7B. Specifically, the average score for AlpacaEval 2.0 is the average win rate $( \% )$ . For comparison, we also present the results of Zephyr-7b-beta, denoted as Zephyr+DPO.

![](images/57291994f4be2bdb0b907a18fb382fed2daa6c4fd6c39ce6be1130f72b8ec58c.jpg)  
Figure 3: Analysis and model performances on quality-limited data. (a) Preference score distributions of data-limited datasets - Alpaca and Dolly, compared to OpenHermes. Note that PDF stands for the probability density function. (c) Performances of PoFT and SFT models training with Alpaca. (d) Performances of PoFT and SFT models training with Dolly. (d) Preference score distributions of hand-crafted noise data on OpenHermes. The increase in the long-trail part indicates the distribution of the noise data. (e) Performances of PoFT and SFT models training with hand-crafted noise data.

to result in over-fitting, which is exacerbated by the noise in data. In contrast, PoFT shows impressive stability.

The studies above underscore the resilience of PoFT in dealing with various data qualities, which can be attributed to the preference scores from aligned LLMs. These scores help mitigate the negative effects of noisy data, emphasizing the higher-quality data during optimization, leading to a significant improvement.

PoFT v.s. Data Filtering When associating with the reward function, the coefficient in Eq.8 can be interpreted as:

$$
\frac { \exp ( \frac { 1 } { M } \sum _ { j = 1 } ^ { M } r _ { j } ( x , y ) ) } { \exp ( \frac { 1 } { M } \sum _ { j = 1 } ^ { M } r _ { j } ( x , y ) ) + \exp ( r _ { \theta } ( x , y ) ) } ,
$$

where $r _ { \theta } ( x , y )$ and $r _ { j } ( x , y )$ refer to the rewards (i.e., preference scores) of the target model and $j$ -th aligned LLM, respectively, and $M$ is the number of aligned LLMs. Intuitively, the preference scores assigned by aligned LLMs could directly guide the optimization process – higher scores increase gradient update weight. As $r _ { j } ( x , y )$ dynamically affects the importance of the samples in training, the PoFT objective can be regarded as a soft filtering approach.

To evaluate this implicit data-filtering mechanism, we apply preference scores to filter data directly. In our experiment, we first sort the data by the scores in descending order. Subsequently, we set thresholds to select varying percentages of data and train PoFT and SFT objectives accordingly.

Figure 4 demonstrates the model performances on the filtered data. In the initial stages, as the number of data increases, there is a positive trend on the SFT model, peaking at 40 percent. This performance even surpasses that of the model trained on the entire dataset. However, despite the continued increase in data volume, the performance begins to decline as more data of inferior quality are included. Interestingly, when trained on filtered data, the PoFT model can further enhance performance.

This steers us toward the hypothesis that combining PoFT and other filtering methods could further enhance overall performance. We assume that PoFT, when employed in conjunction with other filtering strategies, can deliver a multifaceted evaluation of data quality, resulting in a more comprehensive filtering process. Therefore, we conduct experiments on the widely recognized SFT-filtering techniques – IFD (Li et al. 2024c), Instag (Lu et al. 2024), and Deita (Liu et al. 2024). In detail, these filtering methods are applied to the OpenHermes dataset to filter out $20 \%$ of data. Subsequently, the Mistral-7B models are trained with CE and PoFT objectives on these data.

![](images/a2d5dc3e1373eb292d486706d29b54587560e87d89b568369e44841d620552c5.jpg)  
Figure 4: Performance of Mistral-7B trained with different percentages of data on Open LLM Leaderboard.

Table 4: Performance of Mistral-7B models trained with filtered data on Open LLM leaderboard. We present the overall results of the last epoch.   

<html><body><table><tr><td>Filtering method</td><td>FT</td><td>Overall</td></tr><tr><td>N/A Preference score</td><td>SFT PoFT SFT</td><td>61.94 63.52 62.14</td></tr><tr><td>IFD (Li et al. 2024c)</td><td>PoFT SFT PoFT</td><td>64.69 64.71 64.95</td></tr><tr><td>Instag (Lu et al. 2024)</td><td>SFT PoFT</td><td>64.04 64.27</td></tr><tr><td>Deita (Liu et al. 2024)</td><td>SFT PoFT</td><td>64.31 64.49</td></tr></table></body></html>

The overall performance is illustrated in Table 4. It is indisputable that these filtering methods significantly enhance the performance of SFT, even surpassing PoFT models with full data training. Nonetheless, the utility of these methods is not in contention with our approach. In fact, they can be seamlessly integrated with PoFT, yielding performance superior to applying either method in isolation.

In summary, the experiments confirm: (1) our reward function is effective since using preference scores for filtering allows the model to achieve superior performance on less amount of data; (2) PoFT is compatible with other data filtering methods, further enhancing the overall performance.

PoFT v.s. Data Distillation From Aligned LLMs The commonality between PoFT and data distillation is that they both leverage additional LLMs to provide information for model training. However, PoFT incorporates aligned LLMs to guide the gradient optimization process via preference modeling, while data distillation aims at transferring knowledge from the teacher models, rather than solving problems regarding the data quality.

Table 5: Performance of Mistral-7B models training with regenerated data on Open LLM Leaderboard. We present the results of the last epoch.   

<html><body><table><tr><td>FT</td><td>Regen-Model</td><td>Overall</td></tr><tr><td rowspan="3">SFT</td><td>N/A Llama-3-8B-instruct</td><td>61.94 63.16</td></tr><tr><td>Zephyr-7B-sft-full</td><td>62.30</td></tr><tr><td>Yi-6B-Chat N/A</td><td>60.18 63.52</td></tr></table></body></html>

To compare PoFT and data distillation methods, we employ aligned LLMs as teachers to create the responses of OpenHermes, resulting in a new training set. The experiments are conducted on these synthesized data with the CE objective. Intuitively, regenerating the responses is a more explicit way to amplify the effectiveness of aligned LLMs.

The results on the LLM Open Leaderboard are presented in Table 5. Surprisingly, directly replacing the original responses with synthesized data leads to performance degradation. The models trained on the regenerated data underperform PoFT models, performing even worse than the SFT model trained on the original data in some cases. The decrease is more remarkable in the teacher model Yi-6B-Chat.

To sum up, although directly applying aligned LLMs for data regeneration is a more straightforward way for incorporation, it could introduce variability and uncertainty, degrading the model performance. Hence, PoFT offers a more appropriate way of incorporation, efficiently taking advantage of those aligned LLMs through preference modeling.

# 5 Conclusion

In this paper, we present PoFT, a novel and effective preference-oriented SFT method by applying the BradleyTerry objective for modeling preferences between different models. Specifically, given the same SFT data, we intentionally define a preference: favoring the target model over aligned LLMs. This preference encourages the target model to generate higher preference scores when compared to the aligned LLMs. In essence, the aligned LLMs provide assessments of the data quality in the optimization process, varying the effects of SFT data. We conduct extensive experiments on diverse training datasets and different base models to verify the efficacy of PoFT compared to the baselines (the CE objective). Furthermore, we prove its stability towards noise data and validate the effectiveness of the designed objectives by conducting ablation studies on the reward functions and aligned LLMs. Furthermore, PoFT can be combined with other SFT Filtering methods to attain enhanced performance outcomes. Notably, integrating PoFT with DPO has the potential to yield even superior performance.

# Ethical Statement

This research exclusively employs methods and technologies within the field of Natural Language Processing (NLP). Throughout our experimentation, we strictly adhered to ethical guidelines and rules to ensure that no potential risks or unexpected consequences were caused. The data used in this research does not contain sensitive or offensive content. We aim to contribute positively to the NLP community and advance the technology.