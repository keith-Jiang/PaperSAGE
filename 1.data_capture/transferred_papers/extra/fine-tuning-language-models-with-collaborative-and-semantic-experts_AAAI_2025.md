# Fine-Tuning Language Models with Collaborative and Semantic Experts

Jiaxi $\mathbf { Y a n g } ^ { 1 , 2 , * , \dagger }$ †, Binyuan $\mathbf { H u i ^ { 4 , \dag } }$ , Min Yang1,3,‡, Jian Yang4, Lei Zhang1,2, Qiang $\mathbf { Q } \mathbf { u } ^ { 1 }$ , Junyang $\mathbf { L i n ^ { 4 , \ddagger } }$

1 Shenzhen Key Laboratory for High Performance Data Mining, Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences 2 University of Chinese Academy of Sciences 3 Shenzhen University of Advanced Technology 4 Alibaba Group {jx.yang, min.yang}@siat.ac.cn, binyuan.hby $@$ alibaba-inc.com

# Abstract

Recent advancements in large language models (LLMs) have broadened their application scope but revealed challenges in balancing capabilities across general knowledge, coding, and mathematics. To address this, we introduce a Collaborative and Semantic Experts (CoE) approach for supervised finetuning (SFT), which employs a two-phase training strategy. Initially, expert training fine-tunes the feed-forward network on specialized datasets, developing distinct experts in targeted domains. Subsequently, expert leveraging synthesizes these trained experts into a structured model with semantic guidance to activate specific experts, enhancing performance and interpretability. Evaluations on comprehensive benchmarks across MMLU, HumanEval, GSM8K, MT-Bench, and AlpacaEval confirm CoE’s efficacy, demonstrating improved performance and expert collaboration in diverse tasks, significantly outperforming traditional SFT methods.

![](images/7703874ab86ba14674b23a3943be9b34ccbb6aae9e9099dbf85ba2b6fe7c7902.jpg)  
Figure 1: Comparison of performance on MMLU, GSM8K, and HumanEval tasks for different SFT setups. LLaMa2- 7B-SFT means mixing all instruction data, whereas expert means training only on specialized data, e.g. Expert-Code only trains on code-related data.

# Introduction

Advancements in large language models (LLMs) like GPT4 (Achiam et al. 2023), PaLM-2 (Anil et al. 2023) and Claude (Anthropic 2023) have demonstrated impressive performance in versatile capabilities, e.g., knowledge understanding (OpenAI 2022; Ouyang et al. 2022), code generation (Roziere et al. 2023; Li et al. 2023), and mathematical reasoning (Shao et al. 2024; Azerbayev et al. 2023). These capabilities emerge from large-scale pre-training (Touvron et al. 2023; Bai et al. 2023) in conjunction with supervised fine-tuning (SFT) (Ouyang et al. 2022). Pre-training compresses data as much as possible by predicting the next token, facilitating the learning of knowledge. In contrast, SFT aligns the model using a limited but diverse set of instruction data pairs, ensuring that LLMs are helpful, honest, and harmless while activating specific capabilities.

Despite the broad range of the model’s capabilities, variations in data distribution and training strategies often result in strengths in different domain (Yue et al. 2023; Wei et al.

2023a; Singhal et al. 2023). Notably, achieving a balanced performance in general knowledge, coding, and mathematics remains an unresolved challenge (Xu et al. 2023b) for open models (Touvron et al. 2023; Roziere et al. 2023).

Recent efforts have focused on balancing the diverse capabilities of open LLMs. One approach involves enhancing weaker abilities through continued pre-training (Scialom, Chakrabarty, and Muresan 2022; Azerbayev et al. 2023; Xu et al. 2023b; Sukhbaatar et al. 2024). For example, Lemur (Xu et al. 2023b) aim to balance code and general ability by carefully mixing data. Although effective, the method incurs significant data costs and typically requires collecting a large volume of tokens to achieve the desired capabilities. An alternative strategy, Branch-TrainMix (Sukhbaatar et al. 2024) builds on continued pretraining by first developing branches skilled in multiple capabilities and then integrating them using a mixture-ofexperts (MoE) (Jacobs et al. 1991; Shazeer et al. 2017; Fedus, Zoph, and Shazeer 2022) approach. Some evidence suggests (Jiang et al. 2024) that MoE models trained in the standard way do not demonstrate domain specialization. In other words, it is entirely unclear what the experts are responsible for, which does not meet human expectations for structured and interpretable models.

Furthermore, we argue that prior works largely overlook necessity for eliciting balancing abilities at the SFT stage. Mixing instruction data from different domains without considering specific capabilities hinders the full utilization of the model’s pre-training potential. As shown in Figure 1, if the SFT data is grouped by capability, the direct mixing training approach cannot reach the level of independent training, indicating that the current SFT process indeed suffers from capability loss and conflict.

To this end, we propose a CoE (Collaborative and Semantic Experts) based SFT approach, consisting of two distinct phases. The first phase focuses on expert training, where only the feed-forward network (FFN) (Vaswani et al. 2017) is fine-tuned on specialized datasets. This phase produces experts in general knowledge, coding, and math. The second phase focuses on expert leveraging, where multiple experts are synthesized into a single semantically routed MoE model. In this stage, the experts are frozen, and the remaining parameters are trained. This alternating training approach allows the modules trained during the expert training and expert leveraging phases to be complementary, ensuring a seamless transition between the two phases. Besides, unlike vanilla mixture-of-expert models, $C o E$ activates specific experts based on semantically guided data labeling. This makes the model more structured and interpretable, since each expert is trained and utilized for specific capabilities.

We perform comprehensive evaluations on popular benchmarks, including general knowledge on MMLU (Hendrycks et al. 2020), coding on HumanEval (Chen et al. 2021), mathematics on GSM8K (Cobbe et al. 2021) and instruction following on MT-Bench (Zheng et al. 2024) and AlpacaEval (Dubois et al. 2024),. These evaluations demonstrate that $C o E$ achieves optimal performance matching each expert, confirming its efficacy. Remarkably, CoE exhibits superior collaborative capabilities in PoT (Gao et al. 2023) evaluations, where code experts assist in solving mathematical problems, outperforming traditional SFT methods. This synergy highlights the potential of expert collaboration in enhancing model performance across diverse tasks.

# Related Work

Numerous studies have explored the methods to balance diverse capabilities in LLMs. Xu et al. (2023b) attempts to enhance both natural language and coding capabilities, aiming to create versatile language agents with balanced proficiency. Similarly, Xie et al. (2022) proposes UnifiedSKG for structured knowledge grounding, effectively addressing various diverse tasks. FLAN-T5 extends instruction fine-tuning by scaling up the number of tasks and the model size, showcasing significant performance improvements. Dong et al. (2023) study examines how the composition and volume of data during SFT affect performance, finding that total data volume is more influential than data mix ratios. BranchTrain-Mix (BTX)(Sukhbaatar et al. 2024) aims at efficiently training LLMs with multiple domain expertise, which optimizes the accuracy-efficiency trade-off by integrating expert models through MoE layers using continual pre-training strategy. Besides, model merging represents a training-free approach to integrating the capabilities of various models. For example, Yu et al. (2023) proposed DARE, which employs a technique to sparsify delta parameters of multiple homologous models that have undergone supervised finetuning. Our work focuses on a nuanced integration of expert knowledge through structured MoE configurations on the SFT phase, ensuring not only balanced but also contextually optimized performance across diverse applications.

# Preliminaries

The standard Transformer (Vaswani et al. 2017) architecture comprises $L$ stacked blocks, each containing a self-attention module and a feedforward network (FFN), usually enhanced with Layer Normalization (Ba, Kiros, and Hinton 2016) and residual connections (He et al. 2016), for simplicity, we omit these in the formula. In the context of a Mixture-of-Experts (MoE) (Dai et al. 2024; Fedus, Zoph, and Shazeer 2022) setup within Transformers, these standard FFN layers in each block are replaced by MoE-structured FFN layers (Dai et al. 2024; Jiang et al. 2024). This adaptation involves multiple expert networks and a dynamic routing module that determines experts activation based on input tokens. The computation for the $l$ -th $( l \in [ 1 , N ] )$ layer of a Transformer with an MoE structure featuring $E$ prepared experts and a TopK routing, the tokens is processed as:

$$
\mathbf { z } _ { 1 : T } ^ { l } = \mathrm { A t t e n t i o n } ( \mathbf { h } _ { 1 : T } ^ { l - 1 } )
$$

where $\mathbf { h } _ { 1 : T } ^ { l - 1 }$ is the outputs from the $( l - 1 )$ -th layer and $\mathbf { z } _ { 1 : T } ^ { l }$ represents the outputs from the attention module. For each token $t \in [ 1 , T ]$ , the following computations are performed:

$$
\begin{array} { c } { \displaystyle \mathbf { h } _ { t } ^ { l } = \sum _ { i = 1 } ^ { E } \gamma _ { t , i } \cdot \mathrm { F F N } _ { i } ( \mathbf { z } _ { t } ^ { l } ) } \\ { \displaystyle \log \mathrm { i t s } _ { t } = \mathrm { G a t e } ( \mathbf { z } _ { t } ^ { l } ) } \\ { \displaystyle \gamma _ { t , i } = \mathbf { 1 } _ { \mathrm { T o p K } } ( i ) \cdot \mathrm { s o f t m a x } _ { i } ( \mathrm { l o g i t s } _ { t } ) } \end{array}
$$

where $\mathrm { F F N } _ { i }$ denotes the $i$ -th FFN module $( i \in [ 1 , E ] )$ , $\mathbf { h } _ { t } ^ { l }$ are the final outputs for token $t$ at that layer, Gate $\left( \mathbf { \bar { z } } _ { t } ^ { l } \right)$ is a linear module that outputs a vector of logits used to compute the routing weights $\gamma _ { t , i }$ and $\mathbf { 1 } _ { \mathrm { T o p K } } ( i )$ is an indicator function as:

$$
\mathbf { 1 } _ { \mathrm { T o p K } } ( i ) = \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f ~ } i \in \{ \mathrm { i n d i c e s ~ o f ~ t h e ~ t o p ~ } K \mathrm { ~ l o g i t s } \} } \\ { 0 } & { \mathrm { o t h e r w i s e . } } \end{array} \right.
$$

The softmax function normalizes the gate logits, but only the top $K$ experts, as determined by TopK operation, contribute to the output of the current layer for each token. This results in sparse activation, as only $K$ out of $E$ experts are activated for each token.

In the supervised fine-tuning process of the model, it is common to use a target-only next-token loss, denoted as $\mathcal { L } _ { \mathrm { S F T } }$ , to optimize the model’s performance.

# Methodology

Our methodology employs a systematic approach to develop a robust Mixture-of-Experts model, starting with labeling of a large-scale Supervised Fine-Tuning (SFT) dataset into predefined groups. This step is followed by a two-stage training process, Experts Training and Experts Leveraging.

![](images/4d192a32da17eac15e7b3d5ddcef73db66cff96fccb6aa4fdfe7c9200fc42a85.jpg)  
Figure 2: The two-stage training method. The first stage, Expert Training, involves fine-tuning the feedforward network (FFN) parameters for three capability groups: General, Math, and Coding. The second stage, Experts Leveraging, integrates these trained experts into a unified Mixture-of-Experts (MoE) model, utilizing a dynamic routing mechanism to optimize collaboration and performance across diverse tasks.

# Instruction Data Labeling

Accurate categorization into predefined capability groups is essential to enable targeted fine-tuning of experts. Inspired by (Lu et al. 2023), the foundation of our methodology involves a process of labeling instruction data crucial for the effective training of specialized models.

We aggregated large-scale SFT data from various source and systematically labeled each example by prompting GPT4 based on its primary category. This classification resulting in three categories: General, for broad knowledge instructions; Math, for instructions requiring mathematical reasoning; and Coding, for programming-related tasks.

This process led to the creation of three specialized SFT datasets: $\mathcal { D } _ { \mathrm { g e n e r a l } }$ , $\mathcal { D } _ { \mathrm { m a t h } }$ , and $\mathcal { D } _ { \mathrm { c o d i n g } }$ , each tailored to the expertise required for specific tasks. These datasets ensure that each expert is fine-tuned effectively, demonstrating high performance in their designated capability group.

and Coding.

We denote $\mathcal { M }$ as the entire set of parameters of our prepared base model and define $\mathcal { F }$ as the set of all feedforward network (FFN) (Vaswani et al. 2017) parameters, specifically, $\mathcal { F } = \left\{ \mathrm { F F N } _ { 1 } , \cdot \cdot \cdot , \mathrm { F F N } _ { N } \right\}$ . During this expert training phase, only the parameters in $\mathcal { F }$ are updated, while the rest of the parameters, $\mathcal { M } \backslash \mathcal { F }$ , remain frozen.

For better trace the updated FFN parameters uniquely associated with each capability group, we denote the $\mathcal { F } _ { \mathrm { g e n e r a l } }$ , ${ \mathcal { F } } _ { \mathrm { m a t h } }$ and $\mathcal { F } _ { \mathrm { c o d i n g } }$ represent the FFN parameters finetuned using $\mathcal { D } _ { \mathrm { g e n e r a l } }$ , ${ \mathcal { D } } _ { \mathrm { m a t h } }$ and $\mathcal { D } _ { \mathrm { c o d i n g } }$ , respectively.

Each set of FFN parameters undergoes SFT on its designated dataset. This ensures that the training is precisely tailored to enhance each expert’s abilities within its specific capability group. This specialized training not only prepares each expert for effective collaboration in the subsequent phase but also fosters a deep and robust understanding of its respective capability group.

# Experts Training

The initial phase of our methodology focuses on expert training, where each expert is fine-tuned to enhance its specialization within specific capability groups: General, Math,

# Experts Leveraging

Following the expert training phase, we move to the experts leveraging phase where the individually trained experts are synthesized into a unified Mixture-of-Experts (MoE)(Jiang et al. 2024; Dai et al. 2024) model, thus deriving our Collaborative and Semantic Experts architecture, i.e. the proposed $C o E$ model.

In this phase, the feedforward network (FFN) parameters fine-tuned for each capability group during the previous phase are frozen. Additionally, new parameters associated with the MoE router, denoted as $\mathcal { R }$ , are introduced to manage the routing of input tokens. We define the whole parameter set of $C o E$ as follows:

$$
\mathcal { M } _ { C o E } = \left( \mathcal { M } \setminus \mathcal { F } \right) \cup \mathcal { F } _ { \mathrm { g e n e r a l } } \cup \mathcal { F } _ { \mathrm { m a t h } } \cup \mathcal { F } _ { \mathrm { c o d i n g } } \cup \mathcal { R } .
$$

The remaining parameters, $\mathcal { M } \backslash \mathcal { F } \cup \mathcal { R }$ , are then fine-tuned using the union of the three SFT datasets: $\mathcal { D } _ { \mathrm { g e n e r a l } } \cup \mathcal { D } _ { \mathrm { m a t h } } \cup$ $\mathcal { D } _ { \mathrm { c o d i n g } }$ . Furthermore, to not only preserve the expertise inherited from each expert but also to enhance the MoE structure’s ability to harness collaborative interactions among the experts, we introduced a margin loss specifically designed for flexibility in the router’s decision-making process.

Let Logits $\in \mathbb { R } ^ { B \times T \times E }$ represent the logits output by the router for each token, where $B$ is the batch size, $T$ is the sequence length, and $E$ is the number of experts. Let $L \in \mathring { \mathbb { N } } ^ { B \times T }$ represent the indices of the semantically targeted experts, corresponding to the capability groups into which the samples have been classified. Since the label applies uniformly across all tokens in a sequence, it remains constant along the $T$ dimension for each example. Let $K$ be the number of top expert logits to consider.

1. First, we extract the targeted expert logits, where $b$ indexes the batch and $t$ indexes the position within the sequence:

$$
\begin{array} { r } { C _ { b , t } = \mathrm { L o g i t s } _ { b , t , L _ { b , t } } } \end{array}
$$

2. then determine the $\mathbf { k }$ -th highest logits for each token at each position in the batch:

$$
K _ { b , t } = \mathrm { T o p K } ( \mathrm { L o g i t s } _ { b , t , : } )
$$

3. thus we can compute the margin between the correct logits and the threshold logit:

$$
M _ { b , t } = C _ { b , t } - K _ { b , t }
$$

4. finally, the margin routing loss can be expressed as the average over all tokens and all examples:

$$
\mathcal { L } _ { \mathrm { r o u t e r } } = \frac { 1 } { B \cdot T } \sum _ { b = 1 } ^ { B } \sum _ { t = 1 } ^ { T } \operatorname* { m a x } ( 0 , - M _ { b , t } ) .
$$

By strategically freezing and integrating these specialized FFN parameters, our $C o E$ model ensures that each capability group’s expertise is effectively utilized and preserved, promoting robust performance and facilitating a seamless collaboration between the diverse expert modules.

# Experiments Instruction Tuning Datasets

Our initial methodology involved utilizing a large-scale dataset derived from TULU-v2 (Wang et al. 2023; Ivison et al. 2023), a comprehensive collection of instruction tuning datasets. We extracted samples from ShareGPT (Chiang et al. 2023), WizardLM (Xu et al. 2023a), CoT (Chung et al. 2024), FLAN (Chung et al. 2024), Open-Orca (Mukherjee et al. 2023; Lian et al. 2023), GPT4-Alpaca (Peng et al. 2023), and Open Assistant 1 (Ko¨pf et al. 2024). Each sample was labeled to categorize it into capability groups: General, Coding, or Math. To enhance the coding and math datasets, we incorporated additional samples from CodeAlpaca (Chaudhary 2023) and OSS-Instruct (Wei et al. 2023b) for coding, and the CoT partition from MAmmoTH (Yue et al. 2023) for math.

![](images/b2e79a477371c46e83a7b6727d85929e7c47d0e94807df7f81a71470c0608a11.jpg)  
Figure 3: Sankey diagram visualizing the distribution and flow of data from various SFT datasets into the three expertise categories.

# Evaluation Protocol

# Expertise Evaluation Datasets

General Capability We use the Massive Multitask Language Understanding dataset, MMLU (Hendrycks et al. 2020), to measure model’s general knowledge capabilities, which consists of a diverse array of topics.

Coding We employed the HumanEval (Chen et al. 2021) dataset, which consists of 164 programming problems described in natural language, along with corresponding test cases. The model should generate Python scripts that meets the requirements and passes these tests.

Math For mathematical proficiency, we used the Grade School Math (GSM8K)(Cobbe et al. 2021), particularly assessing problem-solving with the Chain-of-Thought (CoT) prompting method, which gauges step-by-step reasoning.

Experts Collaboration Evaluation We evaluated the model’s ability to integrate mathematical and coding skills using the Program-of-Thought (PoT)(Gao et al. 2023) accompany the GSM8K dataset. This method tests the model’s proficiency in solving math problems and generating executable Python scripts. The scripts are executed to assess accuracy and efficiency against expected outcomes. The PoT evaluation, absent from our training data, critically assesses the model’s combinatorial generalization and its capability to apply integrated skills to new tasks.

<html><body><table><tr><td>Model</td><td>MMLU 5-shot</td><td>GSM8K CoT, 8-shot</td><td>HumanEval Greedy, O-shot</td><td>GSM8K PoT,3-shot</td><td>Average All</td><td>Average W/o PoT</td></tr><tr><td>LLaMA2-7B-Base</td><td>45.67</td><td>14.40</td><td>12.80</td><td>17.89</td><td>22.69</td><td>24.29</td></tr><tr><td>LLaMA2-7B-SFT</td><td>49.91</td><td>40.33</td><td>19.51</td><td>31.69</td><td>35.36</td><td>36.58</td></tr><tr><td>Expert-General</td><td>51.26</td><td>14.71</td><td>6.10</td><td>19.79</td><td>22.97</td><td>24.02</td></tr><tr><td>Expert-Math</td><td>46.86</td><td>44.81</td><td>11.59</td><td>17.44</td><td>30.18</td><td>34.42</td></tr><tr><td>Expert-Coding</td><td>46.61</td><td>13.87</td><td>28.05</td><td>21.08</td><td>27.40</td><td>29.51</td></tr><tr><td>CoE-3E2A</td><td>50.47</td><td>44.73</td><td>26.22</td><td>44.81</td><td>41.56</td><td>40.47</td></tr></table></body></html>

Table 1: Performance of the base model (LLaMA2-7B-Base), the supervised fine-tuned model (LLaMA2-7B-SFT), individual expert models (Expert-General, Expert-Math, Expert-Coding), and CoE-3E2A across variouis datasets

Table 2: Performance of $C o E \mathrm { - } 3 E 2 A$ and other models on MT-Bench and AlpacaEval.   

<html><body><table><tr><td rowspan="2">Model</td><td colspan="3">MT-Bench</td><td rowspan="2">Alpaca Eval</td></tr><tr><td>T1</td><td>T2</td><td>Overall</td></tr><tr><td>LLaMA2-7B-SFT</td><td>5.91</td><td>5.28</td><td>5.59</td><td>62.69</td></tr><tr><td>Expert-General</td><td>6.11</td><td>5.43</td><td>5.76</td><td>64.88</td></tr><tr><td>Expert-Math</td><td>5.81</td><td>5.04</td><td>5.43</td><td>57.87</td></tr><tr><td>Expert-Coding</td><td>6.15</td><td>5.53</td><td>5.84</td><td>61.64</td></tr><tr><td>CoE-3E2A</td><td>6.63</td><td>6.09</td><td>6.37</td><td>73.01</td></tr></table></body></html>

Instruction Following Evaluations We evaluated the models’ ability to understand and respond to user inputs using two benchmarks. MT-Bench(Zheng et al. 2024) consists of 80 multi-turn questions, each involving a two-turn interaction, with GPT-4 (Achiam et al. 2023) scoring responses on a 0-10 scale to quantitatively assess conversational abilities. AlpacaEval(Dubois et al. 2024) measures the instruction-following ability by comparing their responses to those from text-davinci-003, using GPT-4 to determine a “win rate”.

# Implementation Details

We utilized LLaMA2-7B-Base (Touvron et al. 2023) for our experiments on 8 NVIDIA A100 GPUs, with training sequences limited to 2048 tokens using the ChatML formatting template(OpenAI 2022). Batch sizes were standardized at 8 per device to maintain consistency. Optimization was handled with the AdamW optimizer, starting with a learning rate warmup to $1 \times 1 0 ^ { - 5 }$ , and then adjusted down to $10 \%$ of its maximum via a cosine scheduler.

Expert Training Phase In the expert training phase, finetuning was concentrated on the feedforward network (FFN) components of the Transformer blocks, with other parameters frozen. The specialized models produced are termed Expert-General, Expert-Math, and Expert-Coding.

Expert Leveraging Phase During the expert leveraging phase, we integrated the trained experts using a Mixture-ofExperts (MoE) architecture. Each MoE block was equipped with a TopK routing module, initialized randomly, and set to Top-2 routing for token distribution among three experts. The synthesized model is named $\underline { { C o E ^ { - 3 } E 2 A } }$ , with “3E” representing the three experts and “2A” indicating Top-2 expert activation.

# Main Results

We conducted a comparative analysis to assess our expert training and leveraging processes. Models evaluated include the base model LLaMA2-7B-Base, a supervised fine-tuned model LLaMA2-7B-SFT across all capability groups, expert models Expert-General, Expert-Math, Expert-Coding, and our integrated MoE model, CoE-3E2A. Results are shown in Table 1 and Table 2, respectively.

Results on Expertise Evaluation Datasets Expert models, tailored to specific capability groups, consistently surpassed the SFT model, demonstrating an $11 \%$ improvement in GSM8K-CoT and a $43 \%$ in HumanEval, reinforcing the efficacy of specialized over mixed training.

Results on Collaboration Dataset Evaluation CoE3E2A achieved a 44.81 execution accuracy in GSM8K-PoT evaluations, highlighting its proficiency in integrating and generalizing across different expertise areas. This performance validates the MoE architecture’s capability to create a dynamic and strong model.

Results on MT-Bench and AlpacaEval In the GPT-4 based evaluations, CoE-3E2A notably outperforms other models, achieving a 6.37 overall score in MT-Bench and a $7 3 . 0 1 \%$ win rate in AlpacaEval. In MT-Bench, CoE-3E2A starts with a score of 6.63 and maintains a strong score of 6.09 in the second turn, demonstrating consistent contextual coherence. In AlpacaEval, it leads significantly, outperforming the SFT model by over $10 \%$ . These results highlight $C o E \mathrm { - } 3 E 2 A$ ’s adeptness at responding to complex user instructions. The robust performance across these benchmarks underscores the $C o E$ architecture’s effectiveness in applying

<html><body><table><tr><td>Model</td><td>MMLU</td><td>CGS.M-skot</td><td>Hreedy,O-salt</td><td>PGS.M-sKkt</td><td>Average</td><td>MT-Bench</td><td>AlpacaEval</td></tr><tr><td>CoE-3E2A</td><td>50.47</td><td>44.73</td><td>26.22</td><td>44.81</td><td>41.56</td><td>6.37</td><td>73.01</td></tr><tr><td>CoE-3E2A-full</td><td>50.40</td><td>44.73</td><td>25.00</td><td>44.88</td><td>41.25</td><td>5.95</td><td>70.20</td></tr><tr><td>CoE-3E2A-router</td><td>49.77</td><td>40.71</td><td>27.44</td><td>33.43</td><td>37.84</td><td>5.94</td><td>71.51</td></tr></table></body></html>

Table 3: Performance comparison of $C o E \mathrm { - } 3 E 2 A$ with FFN-excluded fine-tuning, CoE-3E2A-full with all parameters updated and CoE-3E2A-router with only router modules updated across various benchmarks.

expert knowledge, proving its suitability for handling complex, real-world scenarios.   
Table 4: Computational resource usage summary for various settings, presenting the total number of parameters (#Params), trainable parameters (#Trainable), and GPU memory consumption (GPU Mem) for each model, illustrating the impact of different parameter update strategies on resource efficiency.   

<html><body><table><tr><td>Model</td><td>#Params</td><td>#Trainable</td><td>GPU Mem</td></tr><tr><td>LLaMA2-7B-Base</td><td>6.74 B</td><td></td><td></td></tr><tr><td>LLaMA2-7B-SFT</td><td>6.74B</td><td>6.74B</td><td>39.88 GB</td></tr><tr><td>Expert</td><td>6.74 B</td><td>4.59 B</td><td>35.90 GB</td></tr><tr><td>CoE-3E2A CoE-3E2A-full</td><td>15.40 B</td><td>2.41 B 15.40 B</td><td>45.01 GB 56.30 GB</td></tr><tr><td>CoE-3E2A-router</td><td>15.40 B 15.40 B</td><td>0.0004 B</td><td>40.71 GB</td></tr></table></body></html>

# Ablations of Parameter Update Strategies

This section assesses the impact of different parameter update strategies on our MoE model’s performance during the expert leveraging phase. We conducted ablation studies to determine the optimal update strategy that enhances performance while maintaining the specialized knowledge from the expert training phase. We compared our FFN-excluded SFT model, CoE-3E2A, with two variants:

Full Parameter SFT Updates all parameters uniformly within the MoE model, termed CoE-3E2A-full.

Router-Only SFT Focuses updates solely on the MoE routing modules, creating CoE-3E2A-router.

Results, shown in Table 3, indicate that CoE-3E2A generally outperforms both variants by preserving FFN knowledge while dynamically updating other components. CoE3E2A-full shows a slight performance dip, suggesting that complete updates may dilute specialized capabilities. Meanwhile, CoE-3E2A-router, although performing similarly to CoE-3E2A-full, falls behind in tasks like GSM8K-PoT, highlighting challenges in fostering expert collaboration.

Table 4 outlines the computational resources for each model configuration, detailing total parameters, trainable parameters, and GPU memory consumption.1 LLaMA2-7BBase and LLaMA2-7B-SFT maintain $6 . 7 4 \mathrm { B }$ parameters. The

Expert models use $4 . 5 8 \mathrm { ~ B ~ }$ trainable parameters due to selective FFN-only updates. CoE-3E2A balances performance and resource use with 2.41 B trainable parameters and 45.01 GB of GPU memory. In contrast, CoE-3E2A-full updates all parameters, while CoE-3E2A-router, focusing solely on the routing mechanism, uses the least resources at 40.71 GB of GPU memory. These variations illustrate the efficiency gains from targeted updates, with CoE-3E2A and CoE-3E2A-router showing how focused enhancements can optimize performance and reduce operational costs.

# Ablations of Model Size

In this section, we investigate whether the CoE-3E2A model’s performance enhancements are due to expanded model scale or the strategic integration of expert knowledge. We conduct ablation studies comparing CoE-3E2A with variants that increase model scale but lack fine-tuned expertise:

Replicating MoE Structure To test the impact of the MoE architecture’s complexity, we developed $\underline { { M o E ^ { - 3 } E 2 A } }$ , expanding parameters by using three duplicates of the LLaMA2-7B-Base to initialize the experts, without specialized training. This model assesses whether increases in structure and size can contribute to performance gains.

Matching Activated Parameters We derived $\underline { { M o E ^ { - 2 E 2 A } } }$ model which matching CoE-3E2A’s activated parameters to evaluates whether merely expanding parameters can achieve the performance improvements observed.

As shown in Table 5, CoE-3E2A consistently outperforms both MoE-3E2A and MoE-2E2A on almost every evaluations, especially in the GSM8K-PoT and AlpacaEval, affirming the value of integrating specialized expert knowledge into the MoE architecture, which also suggests that simply increasing the number of model experts or parameters without targeted expertise does not consistently translate into better outcomes. Interestingly, the MoE-2E2A demonstrates a slight advantage over the MoE-3E2A in several benchmarks. This subtle performance discrepancy might be attributed to the more streamlined and efficient parameter usage in MoE-2E2A. With fewer experts to manage, MoE2E2A potentially benefits from reduced complexity in its routing processes, allowing for more effective utilization of its computational resources.

# Ablations of Routing Loss

In this section, we discuss the effectiveness of our routing loss, which is designed to enhance flexibility in the router’s

Table 5: Performance comparison of $C o E \mathrm { - } 3 E 2 A$ against MoE-3E2A and MoE-2E2A across several benchmarks, examining the influence of expanded model scale versus targeted expert integration.   

<html><body><table><tr><td>Model</td><td>MMLU 5-shot</td><td>GSM8K CoT, 8-shot</td><td>HumanEval Greedy, O-shot</td><td>GSM8K PoT,3-shot</td><td>Average</td><td>MT-Bench</td><td>AlpacaEval</td></tr><tr><td>CoE-3E2A</td><td>50.47</td><td>44.73</td><td>26.22</td><td>44.81</td><td>41.56</td><td>6.37</td><td>73.01</td></tr><tr><td>MoE-3E2A</td><td>50.30</td><td>41.70</td><td>25.61</td><td>32.37</td><td>37.50</td><td>6.07</td><td>67.60</td></tr><tr><td>MoE-2E2A</td><td>50.76</td><td>44.28</td><td>25.00</td><td>33.81</td><td>38.46</td><td>5.96</td><td>69.50</td></tr></table></body></html>

Table 6: Performance of CoE-3E2A with and without the routing loss.   

<html><body><table><tr><td></td><td>MMLU</td><td>GSM8K</td><td>HumanEval</td><td>Avg</td></tr><tr><td>CoE-3E2A</td><td>50.47</td><td>44.73</td><td>26.22</td><td>40.47</td></tr><tr><td>w/o Lrouter</td><td>50.06</td><td>41.17</td><td>24.39</td><td>38.54</td></tr></table></body></html>

Routing Decision: Coding+General Coding+Math General+Math   
Let's use python to solve math problems. Instruction   
Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast   
every morning and bakes muffins for her friends every day with four. She   
sells the remainder at the farmers' market daily for $\$ 2$ per fresh duck egg.   
How much in dollars does she make every day at the farmers' market?   
def solution(): Layer1 # Number of eggs Janet's ducks lay per day. eggs_per_day $= 1 6$ # Number of eggs Janet eats for breakfast every day. eggs_eaten $= 3$ # Number of eggs used in baking muffins for her friends every day. eggs_muffins $= 4$ # Number of eggs remaining after Janet eats some and bakes with some. eggs_remaining $\mathbf { \tau } = \mathbf { \tau }$ eggs_per_day - eggs_eaten - eggs_muffins # Price per egg at the farmers' market. price_per $\mathtt { \_ e g g } = 2$ # Calculate the total amount of money Janet makes daily. money_made $\mathbf { \tau } = \mathbf { \tau }$ eggs_remaining $\star$ price_per_egg # Return the total amount of money Janet makes daily at the market. result $\mathbf { \tau } = \mathbf { \tau }$ money_made return result   
def solution(): Layer8 # Number of eggs Janet's ducks lay per day. eggs_per_day = 16 # Number of eggs Janet eats for breakfast every day. eggs_eaten $\mathbf { \lambda } = \mathbf { \lambda } _ { 3 }$ # Number of eggs used in baking muffins for her friends every day. eggs_muffins $= 4$ # Number of eggs remaining after Janet eats some and bakes with some. eggs_remaining $\mathbf { \tau } = \mathbf { \tau }$ eggs_per_day - eggs_eaten - eggs_muffins # Price per egg at the farmers' market. price_per_egg $= 2$ # Calculate the total amount of money Janet makes daily. money_made $\mathbf { \tau } = \mathbf { \tau }$ eggs_remaining $\star$ price_per_egg # Return the total amount of money Janet makes daily at the market. result $\mathbf { \tau } = \mathbf { \tau }$ money_made return result   
def solution(): # Number of eggs Janet's ducks lay per day. Layer32 eggs_per_day $\mathbf { \tau } = \mathbf { \tau }$ 16 # Number of eggs Janet eats for breakfast every day. eggs_eaten $\mathbf { \lambda } = \mathbf { \lambda } _ { 3 }$ # Number of eggs used in baking muffins for her friends every day. eggs_muffins $\mathit { \Delta } = \mathit { \Delta } 4$ # Number of eggs remaining after Janet eats some and bakes with some. eggs_remaining $\mathbf { \tau } = \mathbf { \tau }$ eggs_per_day - eggs_eaten - eggs_muffins # Price per egg at the farmers' market. price_per_egg = 2 # Calculate the total amount of money Janet makes daily. money_made $\mathbf { \tau } = \mathbf { \tau }$ eggs_remaining \* price_per_egg # Return the total amount of money Janet makes daily at the market. result $\mathbf { \tau } = \mathbf { \tau }$ money_made return result

decision-making process. To validate its effectiveness, we compare our $C o E \mathrm { - } 3 E 2 A$ model against its variation trained without an additional routing loss function. The results for this compared model are presented as w/o ${ \underline { { \mathcal { L } } } } _ { \mathrm { r o u t e r } }$ .

As shown in Table 6, the configurations employing margin loss surpass those without it, indicating that a flexible approach to expert selection optimizes the use of diverse expertise. These results highlight that a well-crafted routing loss is crucial for maximizing the potential of MoE, particularly in tasks requiring collaboration among experts.

# Routing Analysis

In this section, we examine the routing decisions made by the CoE-3E2A model across different layers when processing a GSM8K-PoT question. We visualizing the routing decisions for each output token at layer 1, 8 and 32 by giving different background color for different activated experts combinations.

As shown in Figure 4, the routing decisions demonstrate a “convergence”-style behavior across various layers. Initially, at layer 1, there is frequent “routing switching” among different expert sets, indicating early exploration of expertise for optimal input interpretation. By layer 8, this behavior stabilizes, with longer sequences of tokens consistently routed to the same experts. This trend intensifies by layer 32, where almost all tokens are directed to a single expert combination, suggesting that as the model processes deeper, it identifies and commits to the most effective expert configuration for the task, enhancing both processing efficiency and output coherence.

# Conclusion

In this study, we introduced the CoE (Collaborative and Semantic Experts) framework, a novel approach that optimizes large language models through a two-phase supervised fine-tuning strategy, focusing on expert training and leveraging within a well-crafted Mixture-of-Experts (MoE) structure. This strategy includes efficiently performed selective parameter updates, making the two phases integrated seamlessly. Our evaluations across diverse benchmarks have demonstrated $C o E$ ’s effectiveness, showcasing superior problem-solving capabilities through experts collaboration. This framework not only enhances the interpretability and resource efficiency of large models but also sets a new standard for deploying sophisticated MoE model to address complex tasks with enhanced adaptability and performance.