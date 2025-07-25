# Interweaving Memories of a Siamese Large Language Model

Xin Song1, Zhikai Xue2, Guoxiu $\mathbf { H } \mathbf { e } ^ { 1 \ast }$ , Jiawei Liu3, Wei Lu3

1School of Economics and Management, East China Normal University 2Department of Computer Science, Worcester Polytechnic Institute 3School of Information Management, Wuhan University xsong2023 $@$ stu.ecnu.edu.cn, zxue1 $@$ wpi.edu, gxhe $@$ fem.ecnu.edu.cn, laujames2017, weilu @whu.edu.cn

# Abstract

Parameter-efficient fine-tuning (PEFT) methods optimize large language models (LLMs) by modifying or introducing a small number of parameters to enhance alignment with downstream tasks. However, they can result in catastrophic forgetting, where LLMs prioritize new knowledge at the expense of comprehensive world knowledge. A promising approach to mitigate this issue is to recall prior memories based on the original knowledge. To this end, we propose a model-agnostic PEFT framework, IMSM, which Interweaves Memories of a Siamese Large Language Model. Specifically, our siamese LLM is equipped with an existing PEFT method. Given an incoming query, it generates two distinct memories based on the pre-trained and fine-tuned parameters. IMSM then incorporates an interweaving mechanism that regulates the contributions of both original and enhanced memories when generating the next token. This framework is theoretically applicable to all open-source LLMs and existing PEFT methods. We conduct extensive experiments across various benchmark datasets, evaluating the performance of popular open-source LLMs using the proposed IMSM, in comparison to both classical and leading PEFT methods. Our findings indicate that IMSM maintains comparable time and space efficiency to backbone PEFT methods while significantly improving performance and effectively mitigating catastrophic forgetting.

Code — https://github.com/ECNU-Text-Computing/IMSM

# Introduction

Large Language Models (LLMs) have emerged as a significant breakthrough in natural language processing (NLP) (Bai et al. 2023; Touvron et al. 2023; OpenAI 2023; Zeng et al. 2023). With extensive world knowledge, they have exhibited remarkable zero-shot abilities across various tasks, such as language understanding (Achiam et al. 2023), logical reasoning (Kojima et al. 2022), etc. Nevertheless, one of the challenges with LLMs is their susceptibility to generating incorrect or unfaithful outputs, especially in unfamiliar domains (Sun et al. 2024; Lin et al. 2024). In-context Learning (ICL) (Brown et al. 2020; Huang and He 2024) mitigates this issue by incorporating typical examples into prompts, but does not inherently align LLMs with downstream objectives (Fu et al. 2023; Ding et al. 2023). Continued pretraining or supervised fine-tuning can better align parameters with tasks but is time-consuming and computationally intensive (VM et al. 2024; Han et al. 2024). To address this issue, parameter-efficient fine-tuning (PEFT) (Fu et al. 2023; Ding et al. 2023), as shown in Figure 1 (a), selectively finetunes a limited subset of parameters or adds new ones, while keeping the majority fixed.

![](images/425a5b65fc0891b74c83cd27505c6328c93c0e66adfc0936352e22d58c937153.jpg)  
Figure 1: Comparison between vanilla PEFT (a) and our proposed IMSM (b). Vanilla PEFT methods employ an LLM only once. The parameter distribution shift will cause the LLM to forget general knowledge. Instead, IMSM incorporates a siamese LLM, which can be regarded as two LLMs, sharing identical structure and pre-trained parameters. One remains frozen while the other is fine-tuned using an existing PEFT method. By flexibly recalling the memory of the original LLM, IMSM can improve fine-tuning performance and alleviate catastrophic forgetting.

However, refining the parameters of LLMs using target datasets enhances domain-specific knowledge but introduces biases related to that domain and task (Ren et al. 2024). This process can result in forgetting world knowledge and compromising general abilities, which is known as catastrophic forgetting (McCloskey and Cohen 1989; Kirkpatrick et al. 2017). Effective countermeasures typically encompass strategies for replaying pre-training data (Hayes et al. 2020; Kar et al. 2022) or constraints on updating parameters (Chen et al. 2020). Nevertheless, obtaining appropriate pre-training data for LLMs presents considerable challenges. Additionally, the vast range of tasks and the sheer number of parameters complicate efforts to balance integrating new knowledge with preserving existing capabilities. This complexity introduces uncontrollable variables during the fine-tuning process.

Instead, this paper highlights the significance of utilizing the original knowledge of LLM to improve fine-tuning effectiveness and reduce the risk of catastrophic forgetting, as depicted in Figure 1 (b). Generally, the hidden states of the final layer of an LLM can be conceptualized as a memory. It encapsulates the encoded input sequence along with the preceding tokens of the generated response. When the model processes a query, the new memory shaped by the updated knowledge from fine-tuning may diverge from its original understanding. Therefore, for tasks beyond the fine-tuning target, the predicted probability distribution of the next token from this new memory runs the risk of deviating from the typically accurate distribution. In such cases, recalling the original memory before predicting based on the new memory at each step can help prevent deviations in token predictions. This process is akin to opening a channel between two parallel worlds with different experiences, facilitating the fusion of memories of diverse values.

To this end, we propose a straightforward yet highly effective PEFT framework called IMSM, which Interweaves Memories of a Siamese Large Language Model. In particular, we employ a siamese LLM equipped with an existing PEFT method, like LoRA. Given a query, it generates two distinct final hidden states, i.e., memories, based on the original pre-trained parameters and the parameters equipped with the PEFT module. We then propose a query-aware gate as the memory interweaving mechanism to facilitate token generation at each step. During the training phase, the existing PEFT module effectively retains new knowledge for downstream tasks. Thus, the gating mechanism is capable of dynamically balancing between the original memory and the updated memory based on varying queries. During the inference stage, IMSM relies on the given query features, allowing it to flexibly meet the demands of original knowledge for both fine-tuned tasks and other tasks. This mechanism is key to improving performance on fine-tuning tasks and reducing catastrophic forgetting on other tasks. Theoretically, our proposed IMSM is a model-agnostic framework that allows for seamless integration across various open-source LLMs and existing PEFT methods.

We conduct extensive experiments to evaluate the performance of the proposed IMSM in comparison to classical PEFT methods, including LoRA (Hu et al. 2021), $( I A ) ^ { 3 }$ (Liu et al. 2022), and AdaLoRA (Zhang et al. 2023). We also compare with the state-of-the-art PEFT methods, LoRAMoE (Dou et al. 2023) and DoRA (Liu et al. 2024). Moreover, we employ popular LLMs such as ChatGLM3 (Du et al. 2021), Qwen1.5 (Bai et al. 2023), Llama2 (Touvron et al. 2023), and Llama3 (Touvron et al. 2023) as backbone LLMs and fine-tune them with PEFT on four benchmark datasets, e.g., MRPC (El-Said et al. 2015), CoLA (ElSaid et al. 2015), ROPES (Lin et al. 2019), and GSM8K (Cobbe et al. 2021). To evaluate the extent of catastrophic forgetting, we analyze the LLMs’ performance on other datasets before and after fine-tuning on a target dataset. The results demonstrate that IMSM achieves superior performance compared with vanilla PEFT methods. Notably, this improvement is attained without imposing too many additional trainable parameters. Furthermore, IMSM can effectively mitigate the issue of catastrophic forgetting.

Our main contributions are summarized as follows:

• To the best of our knowledge, this represents the first model-agnostic PEFT framework that employs a siamese LLM, effectively harnessing both its intrinsic world knowledge and the novel insights gained from PEFT.

• We propose a query-aware gate mechanism for interweaving memories, allowing flexible fusion of the last hidden states at each step.

Extensive experiments validate the effectiveness of our proposed IMSM, and the results have confirmed its exceptional performance in striking a balance between plasticity and stability.

# Related Work

We review three lines of related work: parameter-efficient fine-tuning methods, strategies for mitigating catastrophic in LLMs, and approaches employing logits arithmetic.

# Parameter-efficient Fine-Tuning

Traditional approaches to fully fine-tuning LLMs with billions of parameters suffer from significant difficulties in terms of training time, computational expense, and practical efficiency. To overcome these, parameter-efficient finetuning (PEFT) techniques have been developed, which focus on adjusting only a subset of the model weights while maintaining the rest unchanged (Mao et al. 2021).

PEFT methods can be categorized into four main types: additive, selective, re-parameterized, and hybrid approaches (Han et al. 2024). Additive methods, like prefix-tuning (Li and Liang 2021) and $( I A ) ^ { 3 }$ (Liu et al. 2022), introduce additional modules into the LLM layers. Selective methods, like BitFit (Zaken, Goldberg, and Ravfogel 2022), focus on selecting a small subset of parameters for fine-tuning. Reparameterized methods (Hu et al. 2021) use low-rank matrices to approximate the changing weights. Hybrid methods (Mao et al. 2022) are designed to combine different PEFT methods, offering a comprehensive solution.

Current PEFT methods enable domain alignment at the parameter level but can risk disrupting LLMs’ existing world knowledge and reasoning capabilities. Therefore, we propose IMSM, a model-agnostic framework for any opensource LLM and PEFT, which leverages the knowledge of the original LLM to enhance response accuracy by integrating the original memories generated from the same input.

# Catastrophic Forgetting in LLMs

LLMs face the challenge of catastrophic forgetting (McCloskey and Cohen 1989) during both pre-training and fine-tuning, where the model tends to forget previously acquired knowledge after being fine-tuned for downstream tasks. Catastrophic forgetting is the traditional challenge of continuous learning, leading to the development of various methods to address this issue in this scenario. Prominent strategies include techniques such as constrained parameter updates, data replay, and parameter isolation (Ke and Liu 2022).

Recently, empirical investigations have indicated that even advanced fine-tuning techniques like LoRA are not immune to catastrophic forgetting (Luo et al. 2023). To address the issue during PEFT, acquiring extensive pre-training data can be an expensive process. Moreover, it is challenging to ensure that the existing knowledge learned by the LLM is not overwritten at the parameter level. Correspondingly, this work suggests a novel approach to address this issue by utilizing the siamese model to directly recall the knowledge from the original LLM. By interweaving distinct memories for the same input, the responses can be generated with enhanced flexibility.

# Logits Arithmetic

Due to the limitations of individual LLM, model collaboration emerges as a promising prospect. Previous studies (Liu et al. 2021) have highlighted the effectiveness of ensembling logits from multiple LLMs in controlled text generation or reducing hallucinations. Contrastive decoding (O’Brien and Lewis 2023) is proposed to directly use the discrepancy in logits between robust and weaker models. Prefix-Adaptive Decoding (Pei, Yang, and Klein 2023) and Context-aware decoding (Shi et al. 2023) enhance the model’s faithfulness by amplifying output probability differences across prompts.

Logits are derived from the final hidden states of LLMs, which serve as internal representations or memories. Prior research (Azaria and Mitchell 2023; Duan, Yang, and Tam 2024) has shown that these hidden states encapsulate knowledge or information pertinent to factual judgments. Thus, the final hidden states, focused on generating the next token, reflect the LLMs’ comprehension of the input information.

Taking this as a starting point, and different from previous methods, we place the siamese LLM within the PEFT framework, enabling a learnable collaboration. This approach not only enhances the performance of PEFT in downstream tasks, but also retains extensive world knowledge and reasoning capabilities of the original LLM.

# Methodology

In this section, we introduce the proposed IMSM, a novel model-agnostic PEFT framework. As illustrated in Figure 2, the siamese LLM is enhanced with a PEFT module. We conceptualize this siamese LLM as comprising two LLMs: one retains the original pre-trained knowledge, while the other is adapted with knowledge specific to the downstream task. The input tokens and the previously generated response tokens are processed by the siamese LLM to create distinct memories, specifically the final hidden states. We then propose a query-aware gate mechanism to interweave these memory representations. The next token is generated from the intertwined memory.

In this framework, the parameters associated with PEFT and the parameters related to the gate mechanism will undergo fine-tuning, while the remaining parameters will be frozen. Our objective is to preserve the model’s inherent world knowledge and general reasoning capabilities while simultaneously improving its performance in both downstream and general tasks. IMSM has the potential to be applied to various open-source LLMs and PEFT methods.

![](images/d74d7d56696de4361bf4daf96ff5df626da540a407be4c023e29d502e0b7e85f.jpg)  
Figure 2: The overall architecture of IMSM, including a siamese LLM and an interweaving mechanism. Given the same input tokens, our siamese LLM produces memories with distinct values, which correspond to the two last hidden states. The generation of the next token relies on the updated memory through an interweaving mechanism. Trainable parameters are marked in red.

# Siamese Large Language Model

To facilitate description and enhance intuitive understanding, our Siamese LLM can be conceptualized as a LLM with dual types of parameters. When integrating PEFT methods into the siamese LLM, the model first retains a copy of the original knowledge. It then undergoes fine-tuning through the PEFT module using downstream data to better align with the new target objective. Taking LoRA as an exemplar PEFT method, we update the parameters $W$ by adding an adapter $\Delta W$ to the original parameters $W _ { o } \in \mathbb { R } ^ { d \times k }$ :

$$
W = W _ { 0 } + \Delta W = W _ { 0 } + B A
$$

where Wo remains frozen, while B Rd×r′ and $A \ \in$ Rr′×k denote trainable low-rank matrices. And r′ is significantly smaller than $d$ and $k$ . This configuration results in a substantial reduction in the number of trainable parameters. $W _ { o }$ and $W$ represent the parameters of the frozen LLM (i.e., original LLM) and tuned LLM, respectively.

# Memory Interweaving Mechanism

Throughout both the fine-tuning and inference processes, our siamese LLM generates distinct hidden states and corresponding logits. The last hidden states can serve as feature representations, encapsulating the LLM’s memory of the previous input tokens. We shall employ these two representations as vehicles to interweave memories derived from various experiences, and propose a gate-based interweaving mechanism for memories.

Let $\mathcal { M }$ represent the original LLM with parameters $W _ { o }$ , while $\mathcal { M } ^ { ' }$ denotes the fine-tuned LLM with parameters $W$ including partially tuned parameters $\Delta W$ .

At each time step $t$ , we can access the hidden states derived from $\mathcal { M }$ and $\mathcal { M } ^ { ' }$ , as well as the hidden states prior to $t$ . First, we compute the average of the query’s last layer hidden states in the Siamese LLM to obtain its dense representations, reflecting the model’s dual understanding of the input. Next, the hidden states for query understanding and next token prediction are concatenated along the feature dimension. Finally, we construct a linear layer with low-rank weight matrices at the top of the model to generate a gate, which determines the contribution of each LLM in the next step of logits generation process:

$$
\overline { { h } } _ { \mathcal { M } } ^ { q } = \frac { 1 } { T _ { i n } } \sum _ { t = 1 } ^ { T _ { i n } } { h _ { \mathcal { M } } ^ { t } } _ { \mathcal { M } }
$$

$$
\overline { { { h } } } _ { \mathcal { M } ^ { \prime } } ^ { q } = \frac { 1 } { T _ { i n } } \sum _ { t = 1 } ^ { T _ { i n } } { h } _ { \mathcal { M } ^ { \prime } } ^ { t }
$$

$$
\begin{array} { r l } & { g a t e = \mathrm { s i g m o i d } \left( f ( \overline { { h } } _ { \mathcal { M } } ^ { q } \oplus h _ { \mathcal { M } } ^ { t } \oplus h _ { \mathcal { M } ^ { \prime } } ^ { t } \oplus \overline { { h } } _ { \mathcal { M } ^ { \prime } } ^ { q } ) \right) } \\ & { = \mathrm { s i g m o i d } \left( ( \overline { { h } } _ { \mathcal { M } } ^ { q } \oplus h _ { \mathcal { M } } ^ { t } \oplus h _ { \mathcal { M } ^ { \prime } } ^ { t } \oplus \overline { { h } } _ { \mathcal { M } ^ { \prime } } ^ { q } ) \cdot W _ { A } \cdot W _ { B } \right) } \end{array}
$$

where $\boldsymbol { h } _ { \mathcal { M } } ^ { t }$ and $h _ { \mathcal { M ^ { \prime } } } ^ { t }$ represent the last layer hidden states of the original LLM $\mathcal { M }$ and the tuned LLM $\mathcal { M } ^ { ' }$ , respectively, for the $t$ -th token in the sequence. $T _ { i n }$ represents the input sequence length. $\overline { { { h } } } _ { \mathcal { M } } ^ { q }$ and $\bar { \overline { { { h } } } } _ { { \mathcal { M } } ^ { \prime } } ^ { q }$ represent the average of the query’s last layer hidden states. $\oplus$ denotes vector concatenation and $f$ represents the linear layer with low rank. Here, $\boldsymbol { h } _ { \mathcal { M } } ^ { t } , \boldsymbol { h } _ { \mathcal { M ^ { \prime } } } ^ { t } \in \mathbb { R } ^ { \bar { 1 } \times d }$ , where $d$ represents the dimension of the hidden state. $W _ { A } \in \mathbb { R } ^ { 4 d \times r }$ and $\boldsymbol { W } _ { B } \in \mathbb { R } ^ { r \times d }$ , where $r$ is a hyper-parameter. Notably, $r$ is significant smaller than $d$ , ensuring a limited number of parameters. Furthermore, the gate is obtained through sigmoid activation.

The updated last hidden state $h _ { \mathcal { N } } ^ { l }$ and the logits can be calculated by:

$$
\begin{array} { c } { { { \pmb h } _ { \mathcal { N } } ^ { t } = g { \pmb a t e } \circ { \pmb h } _ { \mathcal { M } } ^ { t } + ( 1 - g { \pmb a t e } ) \circ { \pmb h } _ { \mathcal { M } ^ { \prime } } ^ { t } } } \\ { { l o g i t s = { \pmb h } _ { \mathcal { N } } ^ { t } \cdot { \pmb W } _ { o u t } } } \end{array}
$$

where $\scriptscriptstyle \mathrm { ~ o ~ }$ denotes element-wise multiplication and $W _ { o u t }$ denotes the parameters of the linear layer of the siamese LLM that are utilized for generating the logits of the next token corresponding to the vocabulary size.

In the memory interweaving process, incorporating query information into the gating mechanism enables the model to better retain original knowledge and reasoning abilities. Unlike additive methods that lead to parameter sharing, concatenating hidden states facilitates finer-grained feature fusion. Generally, both averaging and maximization are valid strategies for handling the parallel memories of past query tokens. However, averaging to synthesize information offers a more comprehensive representation of the query.

# Training and Inference

The probability distribution of the next token can be calculated as follows:

$$
p ( y _ { t } | x , y _ { < t } ) = s o f t m a x ( l o g i t s )
$$

During the fine-tuning phase, we utilize cross-entropy as our loss function:

$$
\mathcal { L } ( \Delta W , W _ { A } , W _ { B } ) = - \sum _ { t = 1 } ^ { T _ { o u t } } \sum _ { y \in V } y _ { t } l o g p ( y _ { t } | x , y _ { < t } )
$$

where $T _ { o u t }$ is the length of the output sequence. Finally, we optimize the parameters $\Delta W$ , $W _ { A }$ , and $W _ { B }$ using the optimization algorithm like AdamW (Loshchilov and Hutter 2017). Consequently, the gate mechanism is capable of learning to execute flexible memory interweaving for diverse queries. This allows our proposed IMSM to mitigate the interference caused by extraneous new knowledge for tasks that fall outside the fine-tuning objective.

During the inference phase, we utilize a greedy search strategy to generate the output sequence. We select the token with the highest probability, informed by the updated memory, as the next generated token, add it to the input sequence, and then repeat this process until the complete sequence is generated.

# Experiments

# Datasets

We conduct experiments on four datasets: MRPC (El-Said et al. 2015), CoLA (El-Said et al. 2015), ROPES (Lin et al. 2019), and GSM8K (Cobbe et al. 2021), to evaluate the alignment capability of our IMSM. Following previous studies (Schick et al. 2024; Asai et al. 2023), we also employ MRPC (El-Said et al. 2015), WebQ (Bordes, Chopra, and Weston 2014), FreebaseQA (Jiang, Wu, and Jiang 2019), and MultiRC (Khashabi et al. 2018), to assess the abilities to retain general knowledge of LLM.

Distinct metrics are employed for different tasks: accuracy (Acc.) and F1 score for MRPC, Matthews correlation coefficient (MCC) for CoLA, Exact Match (EM) and F1 token overlap for ROPES, accuracy for GSM8K and MultiRC. Following Schick et al. (2024) and Asai et al. (2023), we evaluate the performance on WebQ and Freebase by examining whether the generated response contains the golden answer, rather than strictly requiring exact matching.

# Backbone LLMs

We employ four mainstream open-source LLMs, including ChatGLM3-6B (Zeng et al. 2023), Qwen1.5-4B (Bai et al. 2023), Llama2-7B (Touvron et al. 2023), and Llama3-8B (Touvron et al. 2023), as backbones of our siamese LLM. We also directly prompt these LLMs as baseline models.

<html><body><table><tr><td rowspan="2">Backbone</td><td rowspan="2">Method</td><td rowspan="2">Params</td><td colspan="2">MRPC</td><td>CoLA</td><td colspan="2">ROPES</td><td>GSM8K</td><td rowspan="2"></td></tr><tr><td>Acc.</td><td>F1</td><td>Mcc.</td><td>EM</td><td>F1</td><td>Acc. Average</td></tr><tr><td rowspan="5">ChatGLM3</td><td>Original</td><td></td><td>74.72</td><td>82.88</td><td>43.99</td><td>58.47</td><td>22.67</td><td>45.79</td><td>54.75</td></tr><tr><td>LoRA IMSM</td><td>3.899M 4.063M</td><td>86.67 87.13*</td><td>90.10 90.49*</td><td>63.56 65.21*</td><td>74.38 79.15*</td><td>79.41 81.28*</td><td>44.50 48.45*</td><td>73.10 75.29*</td></tr><tr><td>(IA)3 IMSM</td><td>0.513M 0.676M</td><td>86.96 88.70*</td><td>90.32 91.66*</td><td>62.12 64.53*</td><td>72.22 73.34*</td><td>76.36 77.56*</td><td>50.04 52.46*</td><td>73.00 74.71*</td></tr><tr><td>AdaLoRA IMSM</td><td>2.925M 3.089M</td><td>87.02 87.77*</td><td>90.35 90.81*</td><td>62.48 65.57*</td><td>75.14 78.50*</td><td>78.64 82.23*</td><td>49.36 50.80*</td><td>73.83 75.95*</td></tr><tr><td>Original</td><td></td><td>74.20</td><td>81.04</td><td>46.50</td><td>56.58</td><td>38.58</td><td>18.35</td><td>52.54</td></tr><tr><td rowspan="5">Qwen1.5</td><td>LoRA IMSM</td><td>3.277M 3.379M</td><td>86.41 87.25*</td><td>89.79</td><td>65.23</td><td>63.63</td><td>69.56</td><td>50.27</td><td>70.82</td></tr><tr><td>(IA)3</td><td>0.205M</td><td>86.84</td><td>90.48* 90.16</td><td>65.80* 64.96</td><td>65.64* 56.46</td><td>72.71* 64.35</td><td>52.69* 51.33</td><td>72.43* 69.02</td></tr><tr><td>IMSM AdaLoRA</td><td>0.307M 2.458M</td><td>87.13* 86.73</td><td>90.73* 90.15</td><td>69.13* 65.63</td><td>60.25* 64.81</td><td>67.62* 68.62</td><td>53.15* 52.01</td><td>71.34* 71.33</td></tr><tr><td>IMSM Original</td><td>2.560M</td><td>88.41* 68.17</td><td>91.32* 79.83</td><td>66.24* 37.96</td><td>67.18* 49.23</td><td>72.36* 13.95</td><td>51.86</td><td>72.90*</td></tr><tr><td>LoRA</td><td>6.816M</td><td>89.10</td><td>91.88</td><td>71.65</td><td>87.26</td><td>89.09</td><td>3.23 59.59</td><td>42.06 81.43</td></tr><tr><td rowspan="4">Llama3</td><td>IMSM (IA)3</td><td>6.980M</td><td>89.04</td><td>91.84</td><td>72.05*</td><td>87.85*</td><td>90.42*</td><td>61.26*</td><td>82.08*</td></tr><tr><td>IMSM</td><td>0.524M 0.688M</td><td>87.71 88.87*</td><td>90.78 91.77*</td><td>68.16 70.78*</td><td>78.55 81.58*</td><td>82.86 85.95*</td><td>62.62 61.33</td><td>78.45 80.05*</td></tr><tr><td>AdaLoRA IMSM</td><td>5.113M 5.276M</td><td>88.87 89.22*</td><td>91.72 91.95*</td><td>68.77 71.55*</td><td>87.21 87.80*</td><td>88.98 89.05*</td><td>55.42 59.74*</td><td>80.16 81.55*</td></tr></table></body></html>

Table 1: The overall comparison across four downstream tasks. The best results achieved using IMSM and its corresponding vanilla fine-tuning methods are highlighted in boldface. The improvements achieved by IMSM over all baselines are statistically significant, as measured by student’s t-test with a significance level of $p < 0 . 0 5$ .

# Baseline PEFT Methods

We utilize classical and state-of-the-art PEFT methods, like LoRA (Hu et al. 2021), $( I A ) ^ { 3 }$ (Liu et al. 2022), AdaLoRA (Zhang et al. 2023), and DoRA (Liu et al. 2024), to perform fine-tuning on the siamese LLM of our IMSM. In addition, we compare LoRAMoE (Dou et al. 2023), a plugin-based mixture of experts model designed for preventing catastrophic forgetting.

# Implementation Details

We utilize HuggingFace Transformers (Wolf et al. 2019) and PEFT (Mangrulkar et al. 2022) to perform our experiments. The fine-tuning procedure is executed on 8 NVIDIA A800 GPUs under a Linux system.

For LoRA, AdaLoRA, and DoRA, we employ AdamW as the optimizer with learning rates of 3 10−4, 2 10−3, and $1 \times 1 0 ^ { - 4 }$ , respectively, and a batch size of 16. The rank and alpha for LoRA are set to 16. For DoRA, we follow the authors’ recommendations, setting the rank and alpha to 16 and 32. For $( I A ) ^ { 3 }$ , we use Adafactor with a learning rate of $3 \times 1 0 ^ { - 3 }$ and a batch size of 8. All methods are trained for 3 epochs. For LoRAMoE, we use the original paper’s configuration. For a fair comparison, we set the configurations of the tuned target modules of IMSM to be exactly the same as vanilla PEFT. The gate rank $r$ of IMSM is set to 8.

# Performance Comparison

Table 1 presents the fine-tuning results of IMSM and baselines across four datasets. Generally, the fine-tuned LLMs outperform the solely prompted LLMs. This highlights the significance of parameter tuning compared with ICL for aligning LLMs to unfamiliar tasks. Moreover, our proposed method, IMSM, significantly surpasses all vanilla PEFT methods across all three LLMs.

The effect of parameter fine-tuning is influenced by multiple factors, including the quality and capabilities of the backbone LLM, and the adaptability of the employed PEFT method. As one of the most advanced open-source LLMs, the Llama3-based models achieve near-optimal performance. Furthermore, AdaLoRA dynamically adjusts the rank of low-rank matrices, highlighting its efficacy in aligning with downstream tasks.

IMSM, as a simple yet powerful model-agnostic PEFT framework, excels at integrating the stability of the original LLM with the adaptability of the tuned LLM. Additionally, it incorporates the gate mechanism that allows for selective control over whether the understanding is derived from the fixed or adjusted knowledge. Therefore, it leads to consistent improvement across various downstream tasks.

Table 2: Catastrophic forgetting validation between vanilla PEFT and IMSM using ChatGLM3-6B, fine-tuned on ROPES and evaluated on general knowledge benchmarks.   

<html><body><table><tr><td>Dataset ROPES</td><td colspan="5">WebQ MultiRC MRPC Freebase</td></tr><tr><td>ChatGLM3</td><td>58.47</td><td>25.49</td><td>71.74</td><td>74.72</td><td>36.81</td><td>Avg. 53.45</td></tr><tr><td>LoRA IMSM</td><td>74.38</td><td>20.47</td><td>66.67</td><td>70.32</td><td>38.61</td><td>54.09</td></tr><tr><td>(IA)3</td><td>79.15</td><td>21.11</td><td>71.47</td><td>71.83</td><td>38.01</td><td>56.31</td></tr><tr><td>IMSM</td><td>72.22 73.34</td><td>20.96 21.36</td><td>70.83 73.72</td><td>72.58 74.20</td><td>37.34 37.74</td><td>54.79 56.07</td></tr><tr><td>AdaLoRA</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>IMSM</td><td>75.14 78.50</td><td>19.98 22.88</td><td>67.63 73.72</td><td>70.20 74.96</td><td>35.99 38.41</td><td>53.79 57.69</td></tr></table></body></html>

Table 3: Evaluation of catastrophic forgetting between DoRA and its corresponding IMSM across different LLMs.   

<html><body><table><tr><td>Dataset</td><td>GSM8K WebQMultiRCMRPCFreebase</td><td></td><td></td><td>Avg.</td></tr><tr><td>ChatGLM3</td><td>45.79</td><td>25.49 71.74</td><td>74.72</td><td>36.81</td><td>50.91</td></tr><tr><td>DoRA</td><td>45.34</td><td>26.18 71.47</td><td>74.55</td><td>33.26</td><td>50.16</td></tr><tr><td>IMSM</td><td>48.75</td><td>25.49 75.00</td><td>75.19</td><td>38.89</td><td>52.66</td></tr><tr><td>Qwen1.5</td><td>18.35</td><td>32.33 66.99</td><td>74.20</td><td>47.12</td><td>|47.80</td></tr><tr><td>DoRA</td><td>51.33</td><td>27.51 66.03</td><td>73.91</td><td>41.19</td><td>51.99</td></tr><tr><td>IMSM</td><td>52.84</td><td>28.94 68.27</td><td>74.26</td><td>43.52</td><td>53.57</td></tr><tr><td>Llama3.1</td><td>33.06</td><td>41.78 54.17</td><td>42.43</td><td>76.35</td><td>49.56</td></tr><tr><td>DoRA</td><td>72.71</td><td>32.78</td><td>58.65 55.55</td><td>65.92</td><td>57.12</td></tr><tr><td>IMSM</td><td>73.77</td><td>40.31</td><td>66.90 47.94</td><td>75.45</td><td>60.87</td></tr></table></body></html>

# Catastrophic Forgetting Evaluation

Fine-tuning inevitably introduces a degree of forgetting. We fine-tune LLMs on one dataset and subsequently evaluate their performance on other general benchmarks.

Table 2 shows the results of fine-tuning ChatGLM on ROPES. For the three vanilla PEFT methods, they all prioritize enhancing adaptability while neglecting stability. After fine-tuning ChatGLM on ROPES with LoRA, $( I \dot { A } ) ^ { 3 }$ , and AdaLoRA, the accuracy on MultiRC drops from 71.74 to 66.67, 70.83, and 67.63, respectively. IMSM can effectively mitigate performance degradation. For example, IMSM improves overall performance by $4 . 1 0 \%$ compared with LoRA. Notably, the interweaving strategy of IMSM can even revive precise memories within the LLM, enhancing performance on non-target datasets.

We also compare IMSM with LoRAMoE and DoRA, which are state-of-the-art solutions for catastrophic forgetting and PEFT, respectively. For LoRAMoE, we follow the original setup using Llama2 as the backbone. Due to crashes when using DoRA for fine-tuning Llama3, we opt for Llama3.1-instruct instead. As illustrated in Figure 3, IMSM reaches or surpasses LoRAMoE on general benchmarks, offering an average boost of $1 . 8 5 \%$ and $8 . 7 7 \%$ in

80 书 Llama2 LoRA LoRAMoE IMSM 6 2 & T 5   
60 2 3 5T. 今 A 不。 66 6 6 16 8 8 50525 9 X OOS 5   
40   
20 2859   
0 ROPES WebQ MultiRC MRPC Freebase Avg.   
80 Llama2 LoRA LoRAMoE IMSM   
60 5 82 GO 6 01004 g 欢业 9 8 9 2   
20 2 3   
0 GSM8K WebQ MultiRC MRPC Freebase Avg.

overall performance. As shown in Table 3, whether on the target dataset GSM8K or general benchmarks, IMSM’s performance consistently outperforms the standalone DoRA.

The last hidden state serves as the memory for the LLMs’ understanding of the input sequence. By considering the original and updated memories, we ensure that the siamese LLM retains a balance between its prior knowledge and its adaptation to new data.

# Ablation Study

We conduct ablation tests to investigate how the query-aware gate, which serves as the memory interweaving mechanism, improves performance and assess its necessity. The results are shown in Table 4 and Figure 4. In the “w/o query” setting, the siamese LLM’s understanding of the query is excluded from the gate construction. In the “w/o gate” setting, the gate construction is omitted, and the two final hidden states of the siamese LLM are directly added in a 1:1 ratio.

As shown in Table 4, the query-aware gate mechanism brings the most significant improvement to downstream task performance in almost all settings. Figure 4 illustrates that directly combining memories and incorporating the original parameter memory bring performance gains on non-target datasets compared with vanilla PEFT. This highlights the effectiveness of recalling original knowledge through the final layer hidden state. With the introduction of a simple gate, the approach can dynamically balance the previous and the refreshed knowledge. The integration of the siamese LLM’s query feature further alleviates catastrophic forgetting.

# Space and Time Complexity

Compared to the PEFT method alone, IMSM maintains comparable space and time complexity. While the gate mechanism introduces additional trainable parameters, their impact is minimal compared with the original PEFT. We report the count of trainable parameters in Table 1, with a gate rank of 8 used across all IMSM experiments. During inference, the time required to interweave the two memories remains constant for each step, with the two memory

80 320 LoRA 80 欢8 (IA)3 80 AdaLoRA IMSM(w/o gate) 10联 IMSM(w/o gate) 8 201 8 IMSM(w/o gate) 66 6 IMSM(w/o query) IMSM(w/o query) 10. IMSM(w/o query)   
60 IMSM 60 IMSM 60 IMSM 3041 4 5080   
40 40 公 136.81. 40 2   
20 20 20 公 2 台留欢起 WebQ MultiRC MRPC Freebase Avg. WebQ MultiRC MRPC Freebase Avg. WebQ MultiRC MRPC Freebase Avg.

Table 4: The ablation test is performed on ChatGLM3-6B to compare its performance across various downstream tasks.   

<html><body><table><tr><td>Dataset</td><td>MRPC</td><td>CoLA</td><td>ROPES</td><td>Avg.</td></tr><tr><td>LoRA</td><td>86.67 90.10</td><td>63.56</td><td>74.38 79.41</td><td>78.82</td></tr><tr><td>IMSM</td><td>87.13 90.49</td><td>65.21</td><td>79.15 81.28</td><td>80.65</td></tr><tr><td>w/o query</td><td>87.62 90.81</td><td>64.92</td><td>77.43 80.38</td><td>80.23</td></tr><tr><td>w/o gate</td><td>87.01 90.31</td><td>63.74</td><td>76.54 80.12</td><td>79.54</td></tr><tr><td>(IA)3</td><td>86.96 90.32</td><td>62.12</td><td>72.22 76.36</td><td>77.60</td></tr><tr><td>IMSM</td><td>88.70 91.66</td><td>64.53</td><td>73.34 77.56</td><td>79.16</td></tr><tr><td>w/o query</td><td>87.77 91.04</td><td>63.49</td><td>72.93 76.73</td><td>78.39</td></tr><tr><td>w/o gate</td><td>87.04 90.48</td><td>62.94</td><td>71.86 76.77</td><td>77.82</td></tr><tr><td>AdaLoRA</td><td>87.02 90.35</td><td>62.48</td><td>75.14 78.64</td><td>78.73</td></tr><tr><td>IMSM</td><td>87.77 90.81</td><td>65.57</td><td>78.50 82.23</td><td>80.98</td></tr><tr><td>w/o query</td><td>87.59 90.68</td><td>65.19</td><td>75.41 79.56</td><td>79.69</td></tr><tr><td>w/o gate</td><td>87.30 90.59</td><td>64.94</td><td>76.60 80.07</td><td>79.90</td></tr></table></body></html>

streams being generated in parallel theoretically. Compared to vanilla PEFT, IMSM introduces only a lightweight gating mechanism overhead, maintaining equivalent time complexity. The inference speed, measured in tokens per second, is presented in Table 5. The extra memory usage is a necessary and acceptable trade-off to address catastrophic forgetting.

Table 5: The comparisons of inference speed (tokens/sec) between vanilla PEFT methods and IMSM.   

<html><body><table><tr><td>Method</td><td>ChatGLM3</td><td>Qwen1.5</td><td>Llama3</td></tr><tr><td>LoRA</td><td>31.30</td><td>22.86</td><td>22.98</td></tr><tr><td>IMSM</td><td>29.05</td><td>21.76</td><td>21.18</td></tr><tr><td>(IA)3</td><td>33.86</td><td>23.51</td><td>22.81</td></tr><tr><td>IMSM</td><td>30.40</td><td>22.10</td><td>20.51</td></tr><tr><td>AdaLoRA</td><td>30.85</td><td>21.68</td><td>21.49</td></tr><tr><td>IMSM</td><td>27.91</td><td>20.03</td><td>18.71</td></tr></table></body></html>

# Hyper-parameter Analysis

This section focuses on the sensitivity of the rank $r$ , a hyperparameter in the gate mechanism. Keeping other settings constant, we vary the rank among 4, 8, and 16. Please refer to Table 6 for a comprehensive representation. While increasing the rank adds more trainable parameters, it does not necessarily improve performance on the target dataset. The optimal rank varies depending on the specific PEFT used. For IMSM based on LoRA and $( I A ) ^ { 3 }$ , a rank of 8 is often optimal, while for AdaLoRA, 16 is usually the most beneficial choice. However, as the rank increases, the gate mechanism may better recall original memories, potentially reducing forgetting on non-target datasets more effectively.

Table 6: Efficiency on gate rank $r$ is fine-tuned on GSM8K using Llama3-8B, with effects on WebQ and Freebase.   

<html><body><table><tr><td>IMSMMethod Rank Params GSM8K|WebQ</td><td></td><td></td><td></td><td></td><td>Freebase</td></tr><tr><td rowspan="3">LoRA</td><td>4</td><td>6.898M</td><td>59.44</td><td>40.55</td><td>73.15</td></tr><tr><td>8</td><td>6.980M</td><td>61.26</td><td>40.90</td><td>73.72</td></tr><tr><td>16</td><td>7.143M</td><td>59.21</td><td>39.86</td><td>75.23</td></tr><tr><td rowspan="3">(IA)³</td><td>4</td><td>0.606M</td><td>58.98</td><td>37.80</td><td>74.60</td></tr><tr><td>8</td><td>0.688M</td><td>61.33</td><td>38.58</td><td>74.32</td></tr><tr><td>16</td><td>0.852M</td><td>61.26</td><td>38.98</td><td>75.90</td></tr><tr><td rowspan="3">AdaLoRA</td><td>4</td><td>5.194M</td><td>59.67</td><td>41.63</td><td>73.97</td></tr><tr><td>8</td><td>5.276M</td><td>59.74</td><td>40.94</td><td>74.28</td></tr><tr><td>16</td><td>5.440M</td><td>60.50</td><td>42.72</td><td>75.45</td></tr></table></body></html>

# Conclusion and Future Work

This paper presents a novel approach to fine-tuning LLMs, aiming to strike a delicate balance between plasticity and stability. Our proposed IMSM constitutes a model-agnostic framework applicable to any open-source LLMs, in conjunction with existing PEFT methods. Particularly, IMSM enables the interweaving of memories derived from the siamese LLM, facilitating collaboration between frozen knowledge and tuned knowledge. Extensive experiments substantiate that the interweaving mechanism significantly improves alignment performance on downstream tasks and alleviates catastrophic forgetting. Additional hyperparameter experiments further confirm the robustness of the proposed model.

In future research, we aim to explore more intricate memory fusion mechanisms within multiple Transformer layers. Besides, we intend to evaluate the performance of our model on more challenging scenarios of catastrophic forgetting.