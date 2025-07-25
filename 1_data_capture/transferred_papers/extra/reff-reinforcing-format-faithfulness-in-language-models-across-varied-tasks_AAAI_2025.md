# ReFF: Reinforcing Format Faithfulness in Language Models Across Varied Tasks

Jiashu Yao1, Heyan Huang1, Zeming $\mathbf { L i u } ^ { 2 }$ , Haoyu Wen1, Wei $\mathbf { S } \mathbf { u } ^ { 1 }$ , Boao Qian1, Yuhang Guo1\*

1School of Computer Science and Technology, Beijing Institute of Technology 2School of Computer Science and Engineering, Beihang University {yaojiashu, hhy63, haoyuwen, weisu, qianboao, guoyuhang}@bit.edu.cn, zmliu $@$ buaa.edu.cn

# Abstract

# FormatBench

Following formatting instructions to generate well-structured content is a fundamental yet often unmet capability for large language models (LLMs). To study this capability, which we refer to as format faithfulness, we present FORMATBENCH, a comprehensive format-related benchmark. Compared to previous format-related benchmarks, FORMATBENCH involves a greater variety of tasks in terms of application scenes (traditional NLP tasks, creative works, autonomous agency tasks), human-LLM interaction styles (single-turn instruction, multiturn chat), and format types (inclusion, wrapping, length, coding). Moreover, each task in FORMATBENCH is attached with a format checker program. Extensive experiments on the benchmark reveal that state-of-the-art open- and closed-source LLMs still suffer from severe deficiency in format faithfulness. By virtue of the decidable nature of formats, we propose to Reinforce Format Faithfulness (REFF) to help LLMs generate formatted output as instructed without compromising general quality. Without any annotated data, REFF can substantially improve the format faithfulness rate (e.g., from $2 1 . 6 \%$ in original LLaMA3 to $9 5 . 0 \%$ on caption segmentation task), while keep the general quality comparable (e.g., from 47.3 to 46.4 in F1 scores). Combined with labeled training data, REFF can simultaneously improve both format faithfulness (e.g., from $2 1 . 6 \%$ in original LLaMA3 to $7 5 . 5 \%$ ) and general quality (e.g., from 47.3 to 61.6 in F1 scores). We further offer an interpretability analysis to explain how REFF improves both format faithfulness and general quality.

# Code & Datasets — https://github.com/BITHLP/ReFF

# 1 Introduction

Recent years have witnessed a significant upsurge in the development and deployment of large language models (LLMs) (Brown et al. 2020; Touvron et al. 2023; Achiam et al. 2023). With their exceptional zero-shot and few-shot capabilities, LLMs have revolutionized the paradigm of language-related tasks, where a question can be understood and solved to the best without task-specific supervision (Radford et al. 2019).

The zero- and few-shot prompting paradigms have introduced a new problem in task solving procedure, namely, the specification of the output format. To elaborate, LLMs’ task

Perform NER tasks to annotate entities. Use tags '<PER>','</PER>' label person,organization,location,and miscellaneous entities. Apply a flat NER schema to avoid overlapping entities. # <few-shot examples> nput:Sarah baked delicious cookies yesterday.   
Original LLM Adapted LLM 电 。 Low High m Format Format Faithfulness 1 Faithfulness 个 1 format checker Reinforcing Format Faithfulness Well-Formatted Tag Mismatch   
<PER>Sarah</PER> <PER>Sarah</ORG>   
baked delicious baked delicious cookies cookies yesterday. yesterday

Figure 1: The overall framework of this work. The queries in FORMATBENCH are forwarded to an LLM to generate corresponding responses, whose format correctness are labelled by a format checker. The queries, generated responses, and the format labels are utilized in REFF process to iteratively obtain an adapted LLM with higher format faithfulness.

solving paradigm mandates that users must devise an output format and include it in the prompt as a request for LLMs to adhere to. The format specification holds significant importance in tasks of wide concern, for example:

• Various natural language processing (NLP) tasks, such as named entity recognition, text-to-data conversion, and syntactic parsing, hold rigorous format requirements. • Creative works, such as poems, intrinsically possess rigorous forms, including acrostic and numerous others. • LLM-based autonomous agents need strict format adherence to avoid system crashes or dangerous behaviors.

Table 1: Comparison between FORMATBENCH and previous format-related benchmarks. FORMATBENCH features a significantly larger test set, offering a greater variety of application scenes, human-computer interaction styles, and format requirement types. The definition of each category is described in detail in Section 3.1.   

<html><body><table><tr><td rowspan="2">Name</td><td rowspan="2"># Test</td><td colspan="3">Application Scene</td><td colspan="2">Interaction Style</td><td colspan="4">Format Type</td></tr><tr><td>TradNLP</td><td>Creat</td><td>Robot</td><td>Single</td><td>Multi</td><td>Include</td><td>Wrap</td><td>Length</td><td>Code</td></tr><tr><td>CHEM-*</td><td>148</td><td>X</td><td>X</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>√</td></tr><tr><td>S-BENCH</td><td>1,727</td><td>X</td><td></td><td>×</td><td>√</td><td>X</td><td>√</td><td>X</td><td>×</td><td></td></tr><tr><td>IFEVAL</td><td>541</td><td></td><td>√</td><td>X</td><td></td><td>X</td><td>√</td><td>√</td><td>√</td><td>X</td></tr><tr><td>FoFO</td><td>494</td><td>×</td><td></td><td>X</td><td>√</td><td>X</td><td></td><td>√</td><td>X</td><td>√</td></tr><tr><td>FORMATBENCH</td><td>24,483</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table></body></html>

To summarize, the ability to adhere to pre-defined format specifications is of utmost importance in the deployment of LLMs. This ability, which we refer to as format faithfulness, is a crucial aspect to consider in many real-world tasks.

However, there still exists two significant gaps in studies relating to format faithfulness. Firstly, current datasets related to formatting are primarily focused on one specific task, such as text-to-data (Tang et al. 2023), code generation (Skreta et al. 2023), one-turn instruction (Li et al. 2024), and specialty area documentation (Xia et al. 2024), rather than covering varied tasks. This narrow focus restricts the breadth and reliability of format faithfulness evaluation. Secondly, current adaptation approaches aimed at improving format faithfulness like prompt engineering (Skreta et al. 2023) and finetuning (Tang et al. 2023) neglect the decidable nature of format problems, i.e., whether a response adheres to format requirements can be assessed by a non-parameter format checker. This oversight can result in lower effectiveness, as demonstrated in the following experiments.

To address the gap in comprehensive benchmarks, we combine adaptation of existing datasets, online data collection, and manual data annotation, presenting FORMATBENCH. Compared to previous benchmarks, FORMATBENCH includes not only a significantly larger test set, but also a wider range of tasks in diverse application scenes, interaction styles, and format types. As a result, it obtains a comprehensive evaluation of the format faithfulness of LLMs. Extensive experiments on the benchmark reveal that FORMATBENCH poses significant challenges to even the most capable models with simple format requirements, such as selecting among admissible options in a multi-choice question.

To fill the gap in format adaptation approaches of neglecting format decidability, we propose Reinforcing Format Faithfulness (REFF), as is illustrated in Figure 1. REFF takes full advantage of the decidability of format by using a format checker to judge the format correctness of LLM generated content, and then utilizing the judged data in a reinforcement learning (RL) process to improve format faithfulness. Extensive experiments of REFF on FORMATBENCH yield highly favorable results. Without any annotated data, REFF can significantly improve the format faithfulness rate (e.g., from $2 1 . 6 \%$ in original LLaMA3 to $9 5 . 0 \%$ on caption segmentation task), while keep the general quality comparable (e.g., from 47.3 to 46.4 in F1 scores). Combined with labeled training data, REFF can simultaneously improve both format faithfulness (e.g., from $2 1 . 6 \%$ in original LLaMA3 to $7 5 . 5 \%$ ) and general quality (e.g., from 47.3 to 61.6 F1 scores).

We further combine analyses and examples to explain how REFF is able to obtain highly favorable results in terms of both format faithfulness and general quality. The discussion reveals that although often being consistent and aligned, format faithfulness and general quality of LLMs may also trade off as inversely correlated metrics. As a result, solely improving format faithfulness may cause LLMs generating wellformatted but semantically irrelevant content, while REFF can combine the best to two worlds by involving both metrics.

Our main contributions are summarized as follows.

• For a comprehensive evaluation of format faithfulness, we develop FORMATBENCH, which covers a variety of tasks. Experiments show FORMATBENCH is challenging for state-of-the-art LLMs.   
• We propose REFF by incorporating format checking in a reinforcement learning process. REFF is validated to be highly effective in improving format faithfulness with or without extra training data.   
• We offer an interpretability analysis to explain how REFF can simutaneously improve both format faithfulness and general quality.

# 2 Related Work

Format-Related LLM Benchmarks In recent years, there has been significant attention paid to benchmarks and evaluation metrics in language modeling fields. Several notable benchmarks have been developed to evaluate the holistic effectiveness of LLMs (Wang et al. 2018, 2019; Liang et al. 2022; Srivastava et al. 2023). Additionally, a few benchmarks have been proposed to evaluate format-related aspects. However, previous format-related benchmarks are task-specific, failing to provide a comprehensive evaluation of overall format faithfulness. For example, CHEM-\* (Skreta et al. 2023) exclusively addresses a domain-specific programming language, STRUC-BENCH (Tang et al. 2023) exclusively addresses text-to-table conversion, IFEVAL (Li et al. 2024) exclusively addresses single-turn instruction, and FOFO (Xia et al. 2024) exclusively addresses specific domain document generation. FORMATBENCH differs from these benchmarks as it covers a variety of tasks, as is shown in Table 1.

<html><body><table><tr><td>Task</td><td>Format Requirements</td><td>Bad Cases</td></tr><tr><td>NER</td><td>legal flat NER schema</td><td><PER>Sarah</ORG> baked delicious cookies yesterday.</td></tr><tr><td>CapSeg</td><td>≤ 42 chs perespe ine</td><td>Thesimeingkees></td></tr><tr><td>MTT</td><td>adherence to translation rules</td><td>src:Das Exanthem des M. Still ist ein Symptom von hoher Sensitivität. rule: "Exanthem” should be translated into "rash” The exanthema of Still's disease is a symptom of high sensitivity.</td></tr><tr><td>XDL</td><td>successful compilation</td><td><!-- a piece of XDL code that doesn'tpass compilation --></td></tr></table></body></html>

Table 2: Core format requirements for four tasks in FORMATBENCH, and examples of corresponding wrong responses. Parts of the response that do not meet the format requirements are shown by bolding.

Format-Related LLM Adaptations Before the era of LLMs, controllable text generation (CTG) has been proposed to steer a model to generate desired texts according to given control conditions (Prabhumoye, Black, and Salakhutdinov 2020; Zhang et al. 2023). However, CTG methods usually adopt specially designed finetuning schema or modify the sampling procedure in decoding steps (Miao et al. 2019; Qin et al. 2022; Kumar, Paria, and Tsvetkov 2022), which are intricate for LLMs. Recently, several works adopt prompt engineering (Skreta et al. 2023) or finetuning (Tang et al. 2023) to improve format following ability, but neglect the decidable nature of format problems. Unlike previous work, our proposed REFF is designed based on the decidability of formats, and significantly outperforms previous approaches with or without extra training data.

# 3 FORMATBENCH

FORMATBENCH is a collection of tasks with formatting requirements, as shown in the example in Figure 1. In this section, we will firstly introduce the variety that FORMATBENCH covers, then outline the benchmark construction, and finally define the metrics associated with the benchmark.

# 3.1 Variety

FORMATBENCH endeavors to conduct a comprehensive evaluation of format faithfulness. To this end, it covers a variety of application scenes, human-computer interaction styles, and format requirement types.

Application Scene (1) The rigorous output format is necessary for various traditional NLP tasks. (2) In currently prevalent creative tasks, users often devise a format and ask LLMs to follow. (3) Promising LLM-based autonomous agents need to adhere a pre-defined format to interact with the environment. FORMATBENCH combines the all three scenes.

Interaction Style Apart from traditional format specifications in (1) single-turn instructions, FORMATBENCH also involves evaluating format faithfulness in (2) multi-turn interactions, where an LLM iteratively receiving observations and choosing actions to meet the changing format requirements.

Format Types Inspired by previous works (Li et al. 2024; Xia et al. 2024), FORMATBENCH focuses on four aspects of format specifications, namely keyword inclusion, tag wrapping, length constraints, and coding. (1) Inclusion involves certain words to be included or excluded. (2) Wrapping involves enclosing a span of text with pre-defined tags or characters. (3) Length constrains the count of generated content, such as characters or sentences. (4) Coding here refers to more complex format structures that requires a compilers that synthesize the full text.

As is shown in Table 1, FORMATBENCH covers a variety of format-related tasks, and culminating a larger amount of test data compared to previous benchmarks.

# 3.2 Construction

Source Aiming at cover a variety of format-related tasks, the data construction in FORMATBENCH (Figure 2) involves adaptation of existing datasets, collection of web data, and manual annotation. The task descriptions, data annotation, and quality control are detailed in Appendix A.

Traditional NLP Multi-Choice Questions Extractive Question (MCQ) Answering (EQA) 1 5,452/500 1 NamedEntity Constituency Parsing Recognition (NER) (Parse) Creation Robotics Caption Segmentation FormattedTimeGeneration (CapSeg) (FTime) 229,703/542 TerminologyMachine Text Game Agent (Agent) Translation (MTT) 4,440/514 Acrostic Writing (AcroW) XDL Generation (XDL) Inclusion Length 5,452/500 Multi-Step Wrapping Coding # train /# test

Format For each task, we define the format requirements based on previous literature and rough consensus. Some examples and their corresponding cases that fail to satisfy them are listed in Table 2, and all specific format requirements are listed in Appendix A. Moreover, we construct a corresponding format checker for each task in FORMATBENCH. A format checker is a program, that given an input query and a response, determines whether the response adheres to the format requirements of the task. Formally, given a query $q$ and a generated response $r$ , the format checker is defined as:

$$
{ \mathcal { F } } ( q , r ) = \left\{ { \begin{array} { l l } { 1 \quad } & { \mathrm { i f ~ } r \mathrm { ~ f i t s ~ t h e ~ f o r m a t } , } \\ { - 1 \quad } & { \mathrm { o t h e r w i s e } . } \end{array} } \right.
$$

# 3.3 Metrics

We associate two metrics to FORMATBENCH, namely, format faithfulness rate and general quality. Format faithfulness rate evaluates to what extent can an LLM $\mathcal { M }$ follows the format specifications, by calculating the format checker pass-rate across the set all samples $D$ :

$$
F F R = \mathbb { E } _ { q \in D } [ \mathbb { 1 } \left( \mathcal { F } ( q , \mathcal { M } ( q ) ) = 1 \right) ] .
$$

General quality of the generated responses is also evaluated, as a response is considered effective only if it is both faithful in format and correct in content. Due to the heterogeneity among all tasks in FORMATBENCH, we respectively define the general quality metrics (e.g., BLEU, F1, accuracy) for each task, as described in Appendix A.

# 4 Reinforcing Format Faithfulness

# 4.1 Algorithm

Format problems have a decidable nature, where a format checker can easily discriminate whether generated texts adhere to format requirements or not. However, previous methods fail to fully take advantage of this feature.

We find that the format dicidability perfectly fits the reinforcement learning paradigm, where an environment provides rewards for the action given by an agent. To be specific, in a RL-based format faithfulness adaptation, LLMs can be viewed as agents that generating structured texts as actions, which are rewarded by an format checker environment.

In doing so, we proposed REFF to use reinforcement learning for format faithfulness adaptation by rewarding models for correct formats and penalizing incorrect ones.

<html><body><table><tr><td>Algorithm1:REFF</td></tr><tr><td>Input:querysetQ,formatcheckerF,LLMM,# epoch n Output: adapted LLM M'</td></tr><tr><td>1: LetM' ←M 2: for epoch in[1,2,..,n] do</td></tr><tr><td>3: for q in Q do 4: r←M'(q) / response generation</td></tr><tr><td>5: s← F(q,r) // format checking,s ∈{-1,1}</td></tr><tr><td>6: M' ← step(M',q,r,s) 7: end for // PPO stepping</td></tr><tr><td></td></tr><tr><td>8:end for 9:return M'</td></tr></table></body></html>

The process of REFF is shown in Algorithm 1, where the $s t e p ( )$ is the function of reinforcement learning from human feedback style (RLHF-style) stepping aimed at updating the

LLM given the action and the reward. The used RLHF-style loss function (Ziegler et al. 2019; Ouyang et al. 2022) differs from the original proximal policy optimization (PPO) (Schulman et al. 2017), in that it additionally adds a KullbackLeibler (KL) penalty from the original model to prevent the adapted model from shifting too far. The algorithm generally shares the same procedure with RLHF, except that the computation of rewards is different. Specifically, RLHF often uses a pre-trained reward model, while REFF relies on format checkers to compute the rewards. We offer a rigorous math representation of REFF algorithm in Appendix B.

# 4.2 Settings

Considering the data availability in various real-world scenarios, we set three settings for RL in REFF, whose accessible data groups (query set $Q$ in Algorithm 1) are list in Table 3.

Table 3: Data used for RL in three settings of REFF.   

<html><body><table><tr><td>Settings</td><td>Test Queries</td><td>Train Queries</td><td>Train Labels</td></tr><tr><td>REFF-tst</td><td>√</td><td>X</td><td>X</td></tr><tr><td>REFF-trn</td><td>×</td><td>√</td><td>X</td></tr><tr><td>REFF-trn-ft</td><td>×</td><td></td><td><</td></tr></table></body></html>

Test-Only REFF When there exists no extra training data, LLMs can use queries in the test set as the query set $Q$ . Notably, no label of the test set is available to the model in this setting. However, this setting only applies to the offline scenarios, where LLMs handle a batch of queries and generate all responses subsequently.

Train-Only REFF w./wo. Finetuning Train-only setting can be applied in an online scenario, where the queries are processed and responsed one by one, as the adaptation of LLMs only involves training queries as the query set $Q$ . Additionally, considering that a training set often includes both queries and labels, we further study a train-only with finetuning setting, where the reinforcement process is implemented after finetuning on the training set.

# 5 Experiments

# 5.1 Baselines

There are two groups of baselines we implement to compare with REFF, namely, refinement and finetuning.

Refinement There exists many works on prompt engineering about augmenting LLMs with internal reflections (Wei et al. 2022; Madaan et al. 2023) to refine their initial content. Among them, a recent paper focuses on generating wellstructured codes (Skreta et al. 2023). Inspired by this, we take refinement as a general prompt schema for improving format faithfulness on all tasks. Generally, an LLM iteratively polishes the output format according to error information from the format checker. Optionally, we further augment the refinement process with LLM internal thoughts following the CoT (Wei et al. 2022) and ReAct (Yao et al. 2023) prompting.

<html><body><table><tr><td>Models</td><td>1</td><td></td><td>Y</td><td>rse Par</td><td>CapSeg</td><td></td><td>公 AcroV</td><td>e</td><td>1 Age</td><td></td><td></td></tr><tr><td>GPT-3.5</td><td>99.0</td><td>89.7</td><td>95.3</td><td>36.2</td><td>45.8</td><td>56.0</td><td>44.5</td><td>95.4</td><td>71.0</td><td>5.5</td><td>avg. 63.8</td></tr><tr><td>LLaMA3</td><td>97.0</td><td>89.6</td><td>84.8</td><td>0.2</td><td>21.6</td><td>52.3</td><td>1.7</td><td>99.4</td><td>88.3</td><td>13.3</td><td>54.8</td></tr><tr><td>Gemma</td><td>98.0</td><td>90.0</td><td>82.0</td><td>5.5</td><td>28.2</td><td>50.9</td><td>2.0</td><td>98.8</td><td>91.4</td><td>0.0</td><td>54.7</td></tr><tr><td>Qwen1.5</td><td>96.6</td><td>91.3</td><td>71.9</td><td>4.6</td><td>24.7</td><td>55.3</td><td>0.9</td><td>99.0</td><td>88.1</td><td>10.3</td><td>54.3</td></tr><tr><td>Mistral</td><td>96.0</td><td>91.1</td><td>86.2</td><td>1.6</td><td>34.1</td><td>40.4</td><td>6.9</td><td>99.4</td><td>86.8</td><td>0.0</td><td>54.2</td></tr><tr><td>Mistral-inst</td><td>96.0</td><td>89.5</td><td>77.5</td><td>1.4</td><td>32.1</td><td>54.2</td><td>3.0</td><td>97.2</td><td>79.0</td><td>0.0</td><td>53.0</td></tr><tr><td>LLaMA2</td><td>97.4</td><td>86.9</td><td>83.9</td><td>0.3</td><td>25.5</td><td>39.9</td><td>0.1</td><td>99.8</td><td>73.7</td><td>8.9</td><td>51.6</td></tr><tr><td>LLaMA</td><td>89.6</td><td>88.5</td><td>74.1</td><td>0.3</td><td>22.1</td><td>29.7</td><td>0.0</td><td>72.8</td><td>81.7</td><td>42.4</td><td>50.1</td></tr><tr><td>Falcon</td><td>83.4</td><td>77.8</td><td>62.7</td><td>0.1</td><td>26.9</td><td>20.5</td><td>0.0</td><td>32.0</td><td>63.2</td><td>45.5</td><td>41.2</td></tr><tr><td>Falcon-inst</td><td>83.2</td><td>55.4</td><td>26.0</td><td>0.0</td><td>22.7</td><td>11.9</td><td>0.1</td><td>35.9</td><td>35.4</td><td>54.5</td><td>32.5</td></tr></table></body></html>

Table 4: Format faithfulness rate $( \% )$ of original models on FORMATBENCH.

Finetuning The abilities of LLMs can be further adapted according to specific goals by finetuning (Zhao et al. 2023). Specifically, recent work (Tang et al. 2023) finetunes LLMs to generate well-structured data on specific tasks. Inspired by these works, we propose to conduct finetuning on LLMs to improve the overall format faithfulness.

# 5.2 Experimental Setup

Models We conduct a comprehensive evaluation on format faithfulness with FORMATBENCH across many state-of-theart open-source LLMs sizing about 7B, including LLaMA7B, LLaMA-2-7B, LLaMA-3-8B, Qwen-1.5-7B, Falcon7B, Falcon-7B-Inst, Mistral-7B-v0.3, Mistral-7B-Inst-v0.3, Gemma-7B. We further compare these models to closedsource GPT-3.5 (gpt-3.5-turbo-instruct). Note that we only use instruction models, as all tasks in FORMATBENCH are instruction tasks, where instruction prompting styles are more suitable and flexible than chat ones. In adaptation experiments including refinement, finetuning, and REFF, we use LLaMA-3-8B as the base model, as it exhibits a favorable format faithfulness in the original model evaluation.

Adaptation Implementation We use trl (von Werra et al. 2020) library to implement the finetuning and the RLHF-style PPO of REFF. More information about the implementation of adaptation methods are detail in Appendix D.

Hyper-Parameters To ensure the robustness and reliability of the results, we try to use default and commonly-used hyper-parameters, and keep them consistent among different experiments. Here we list several key points, and the detailed hyper-parameters are outlined in Appendix D.

• In generation, we adopt greedy decoding in all experiments for a fair and efficient comparison. • We use LoRA (Hu et al. 2021) in all LLM adaptation experiments with a consistent configuration $r = 1 6$ . • In fintuning, we use a constant learning rate $2 e \mathrm { ~ - ~ } 5$ and train for 3 epochs with 256 instances per batch. • In reinforcement learning, we set target of KL divergency to be 6, use a constant learning rate $1 . 4 1 e \mathrm { ~ - ~ } 5$ , and train for 3 epochs with 32 instances per batch.

# 5.3 Original Model Results

We evaluate the models using the prompts in Appendix C.

Results The format faithfulness results of original LLMs on FORMATBENCH are presented in Table 4 (general quality in Appendix E). We find the benchmark to be both discriminating and challenging for LLMs. Firstly, it can be observed that stronger model like GPT-3.5 does exhibits better faithfulness to format, as its format faithfulness rates surpasses those of other smaller open-source models. Secondly, it is validated that format tasks are still highly challenging for even the most capable models.

Outliers Moreover, there are some intriguing exceptions found in the results, where smaller models like Falcon-inst demonstrate superior faithfulness compared to GPT-3.5 in XDL task $5 4 . 5 \%$ versus $5 . 5 \%$ ), as shown in Table 4. We try to explain this phenomenon in Section 6 by studying the relation between format faithfulness and general quality.

# 5.4 Adapted Model Results

In order to adapt LLMs to alleviate format unfaithfulness, we propose REFF by incorporating a format checker into a reinforcement learning process. We further offer three settings in Table 3, namely REFF-tst, REFF-trn, and REFF-trn-ft to cover different application scenarios.

We study the adaptation approaches in test-only setting with four tasks including NER, CapSeg, MTT, and XDL. The corresponding examples are listed in Table 2. These four tasks are chosen for two reasons, (1) they fully covers the application scenes and format types in our proposed taxonomy shown in Figure 2, and (2) their format requirement difficulties are moderate as shown in Table 4. In the train-only setting, we choose the NER and CapSeg tasks as the other two tasks are not attached with training data.

REFF-tst The results of REFF-tst and other baselines in test-only setting are shown in Table 5. Comparing REFF to other approaches, it is obvious that REFF can significantly improve the format faithfulness rate while keep the general quality comparable without any annotated data. Notably, the format faithfulness of REFF exceeds not only its LLaMA3 baselines, but also GPT-3.5, validating the high effectiveness of our proposed REFF approach. Moreover, comparing REFF trained on one specific task (REFF-tst-[task]) to that trained on mixed data (REFF-tst), we can find that the catastrophic forgetting phenomenon is not significant in format-related reinforcement learning, as the format faitfhfulness rate of one task doesn’t significantly drops when combined with data from other tasks.

Table 5: Format faithfulness rate $( \% )$ and general quality (F1 for NER and CapSeg, BLEU-4 for MTT) in the test-only setting. Best results among LLaMA3-based models are bolded. REFF-tst-[task] refers to the model adapting with exclusively corresponding dataset, while REFF-tst mixes and shuffles all four datasets. The asterisk symbol denotes refinement with internal thoughts. The general quality of XDL is not evaluated due to the need for a high level of expert knowledge, as detailed in Appendix A.   

<html><body><table><tr><td rowspan="2">Models</td><td colspan="4">Format Faifulness Rate (↑)</td><td colspan="3">General Quality (个)</td></tr><tr><td>NER</td><td>CapSeg</td><td>MTT</td><td>XDL</td><td>NER</td><td>CapSeg</td><td>MTT</td></tr><tr><td>GPT-3.5</td><td>95.3</td><td>45.8</td><td>56.0</td><td>5.5</td><td>94.3</td><td>40.6</td><td>30.9</td></tr><tr><td>+refine</td><td>96.1</td><td>62.5</td><td>73.1</td><td>5.9</td><td>94.3</td><td>42.0</td><td>31.7</td></tr><tr><td>+refine*</td><td>96.7</td><td>72.3</td><td>83.8</td><td>5.9</td><td>94.2</td><td>23.0</td><td>31.5</td></tr><tr><td>LLaMA3</td><td>84.8</td><td>21.6</td><td>52.3</td><td>13.3</td><td>88.3</td><td>47.3</td><td>32.2</td></tr><tr><td>+refine</td><td>85.0</td><td>21.8</td><td>61.8</td><td>13.3</td><td>88.3</td><td>47.2</td><td>17.7</td></tr><tr><td>+refine*</td><td>85.6</td><td>33.0</td><td>69.2</td><td>13.3</td><td>88.5</td><td>21.8</td><td>13.7</td></tr><tr><td>ReFF-tst-NER</td><td>96.7</td><td>20.7</td><td>58.5</td><td>17.6</td><td>91.4</td><td>47.8</td><td>33.0</td></tr><tr><td>ReFF-tst-CapSeg</td><td>87.6</td><td>95.0</td><td>52.0</td><td>15.0</td><td>87.9</td><td>46.4</td><td>31.3</td></tr><tr><td>ReFF-tst-MTT</td><td>88.8</td><td>21.6</td><td>98.2</td><td>13.0</td><td>88.4</td><td>47.6</td><td>31.0</td></tr><tr><td>ReFF-tst-XDL</td><td>86.2</td><td>21.4</td><td>51.0</td><td>52.6</td><td>89.0</td><td>47.3</td><td>31.9</td></tr><tr><td>ReFF-tst</td><td>99.7</td><td>100.0</td><td>97.2</td><td>14.8</td><td>93.1</td><td>40.9</td><td>35.3</td></tr></table></body></html>

REFF-trn When test data is not available beforehand, while there exists training data for adaptation, REFF-trn succeeds in improving format faithfulness to the extent that REFF-tst does, as is shown in Table 6. These results proves the format faithfulness improvement is robust, and does not result from overfitting to the test set.

REFF-trn-ft By combining reinforcement for improving format faithfulness and finetuning for improving general quality, REFF-trn-ft obtains highly favorable results on both metrics, as is shown in Table 6.

Table 6: Format faithfulness rate $( \% )$ and general quality (F1) in the train-only setting. Best results are bolded. The asterisk symbol denotes refinement with internal thoughts.   

<html><body><table><tr><td rowspan="2">Models</td><td colspan="2">FF Rate (↑)</td><td colspan="2">GQ (↑)</td></tr><tr><td>NER</td><td>CapSeg</td><td>NER</td><td>CapSeg</td></tr><tr><td>GPT-3.5</td><td>95.3</td><td>45.8</td><td>94.3</td><td>40.6</td></tr><tr><td>+refine</td><td>96.1</td><td>62.5</td><td>94.3</td><td>42.0</td></tr><tr><td>+refine*</td><td>96.7</td><td>72.3</td><td>94.2</td><td>23.0</td></tr><tr><td>LLaMA3</td><td>84.8</td><td>21.6</td><td>88.3</td><td>47.3</td></tr><tr><td>+ finetune</td><td>99.0</td><td>38.2</td><td>95.9</td><td>63.6</td></tr><tr><td>ReFF-trn</td><td>99.8</td><td>99.8</td><td>92.6</td><td>40.9</td></tr><tr><td>ReFF-trn-ft</td><td>99.2</td><td>75.5</td><td>95.2</td><td>61.6</td></tr></table></body></html>

Outliers Moreover, similar to the exceptions in the original model results, there also exists outliers in adapted model results. As shown in Table 5, the general quality drops drastically with refinement on LLaMA3 MTT task (from 32.2 to

![](images/8fe919f08941696142d0733f27f474e3fbaebf432736527d32cd267b08ad2a3b.jpg)  
Figure 3: Conceptual contour map of format faithfulness and general quality. Inner circles indicate higher scores for both metrics. Solely improving format faithfulness $\langle \mathbf { A } $ B) may result in an LLM with high format faithfulness but low general quality. REFF can get the best of two worlds by combining finetuning $\langle \mathbf { A } \to \mathbf { C } \rangle$ and reinforcement $( \mathbf { C }  \mathbf { D } )$ .

13.7 in BLEU-4), and REFF-tst-XDL obtains an unusually higher format faithfulness rate $( 5 2 . 6 \% )$ on XDL task than its counterpart trained with all four tasks $( 1 4 . 8 \% )$ . We will also give an explanation to the outliers in Section 6.

# 6 Analysis

We begin our analysis by considering the outliers in the results of experiments in Section 5, including:

• Outlier 1: Although considered being more capable, GPT3.5 does not perform as well as 7B open-source models in format faithfulness on XDL task (Table 4). • Outlier 2: When using refinement techniquesin LLaMA3, format faithfulness improves significantly, but general quality suffers from a drastic drop on MTT task (Table 5). • Outlier 3: REFF-tst-XDL significantly outperforms REFF-tst, its counterpart with similar training data, in format faithfulness rate on XDL task (Table 5).

![](images/3cfecda815b091e770a446ec670f9dc931e9e6c6add727e80ef8fd626cc21416.jpg)  
Dimethyl (2RS)-2-(N-(methyloxycarbonyl)amino)-6-oxopimelate ( $3 0 0 \mathrm { m g }$ $1 . 0 4 \mathrm { m m o l }$ ) was treated with a solution of CH2N2 in diethyl ether at $0 ^ { \circ } \mathsf { C }$ for 1h Solvent was removed in vacuo and the residue purified by flash chromatography $5 0 \%$ EtOAc in hexane, ${ \sf R f } 0 . 5$ ; $10 \%$ CH3CNin CH2Cl2 was also effective, Rf0.27)togiveDimethyl(2RS,6RS)-2-(N-(methyloxycarbonyl)amino)-6-epoxymethanopimelateasacolourlessoil(3.4mg $91 \%$ ).   
Figure 4: An instance in XDL task (top), the corresponding response of REFF-tst (left), and that of REFF-tst-XDL which obtains a higher format faithfulness rate on XDL task (right). REFF-tst-XDL generates syntactically correct but irrelevant code.

These exceptions have a same pattern, that is, the discrepancy between format faithfulness and general quality. More specifically, a model that is supposed to have higher quality may exhibits poorer format faithfulness. By analyzing the three outliers and the underlying shared pattern, we summarize the relation between format faithfulness and general quality, and then explain how our proposed REFF shows highly favorable results in both metrics.

In this section, we will first offer an insightful illustration to explain the discrepancy between format faithfulness and general quality, and then discuss a specific case of Outlier 3.

# 6.1 Illustration

Figure 3 illustrates the conceptual relation between format faithfulness and general quality. Although being consistent in most scenarios, format faithfulness and general quality are two different metrics, and sometimes may trade off as inversely correlated indicators.

An LLM can obtain an acceptable general quality, while failing to adhere format requirements, as point A shows. Meanwhile, an LLM can be completely faithful to format, while being poor in general quality, as point B shows. Solely adapting LLMs with the supervision of format faithfulness rates may guide them from the former situation (point A) to the latter one (point B).

The discussion above give all three exceptions an explanation. Point A explains Outlier 1, where strong LLMs like GPT-3.5 generate poor-structured texts. Point B explains Outlier 3, where an LLM may gain a significant format faithfulness improvement without higher general quality. The trace from Point A to Point B explains Outlier 2, where an LLM compromises general quality to improve format faithfulness.

Fortunately, experiments in Table 6 show that our proposed REFF is able to combine the best of both worlds. In the finetuning process in REFF, an LLM is firstly adapted to a position with high general quality (point A to C). In the following, it is futher adapted for better format faithfulness by reinforcement (point C to D). With the application of KL regularization term that prevents the model from shifting too far from the original parameters (point C) in reinforcement process, the LLM after reinforcement (point D) can avoid significant decrease in general quality.

# 6.2 Examples of XDL Task

To explain why in Outlier 3 the XDL task falls into the point B (high format faithfulness, low general quality), we take a closer look at the example in Figure 4. In the example, REFFtst-XDL sneakily passes the format checker by generating short and simple well-formatted code (high format faithfulness) that is irrelevant to the instruction (low general quality). The phenomenon is an typical instance of mode collapse in RLHF, characterized by a reduced diversity in produced samples (Casper et al. 2023). RL-based adaptation techniques that may suffer from the mode collapse still have some room for substantially improving the performance in XDL task, as they require the original LLM to produce a number of diverse correctly formatted responses for rewarding.

# 7 Conclusion

In this paper, we aim to conduct a comprehensive evaluation of format faithfulness and enhance it without compromising the general quality of LLMs. In doing so, we firstly propose FORMATBENCH, a format-related benchmark that covers a variety of tasks. FORMATBENCH is shown to be highly discriminating and challenging for state-of-the-arts LLMs. Subsequently, by utilizing the decidable nature of formats, we incorporate format checking procedures into reinforcement learning to propose REFF. Extensive experiments validate the high effectiveness of REFF in simultaneously enhancing both format faithfulness and general quality. Finally, we provide an interpretability analysis to elucidate the reasons behind REFF’s effectiveness by exploring the relationship between format faithfulness and general quality.