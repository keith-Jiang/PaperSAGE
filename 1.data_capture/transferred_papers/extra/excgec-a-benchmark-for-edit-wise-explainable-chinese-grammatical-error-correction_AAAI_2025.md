# EXCGEC: A Benchmark for Edit-Wise Explainable Chinese Grammatical Error Correction

Jingheng $\mathbf { Y } \mathbf { e } ^ { 1 \ast }$ , Shang ${ { \bf { Q } } { \bf { i n } } ^ { 1 * } }$ , Yinghui $\mathbf { L i } ^ { 1 }$ , Xuxin Cheng2, Libo $\mathbf { Q } \mathbf { i n } ^ { 3 }$ , Hai-Tao Zheng1†, Ying Shen4, Peng $\mathbf { X i n g ^ { 1 } }$ , Zishan ${ \bf X } { \bf u } ^ { 1 }$ , Guo Cheng1, Wenhao Jiang5†

1Tsinghua University, 2Peking University, 3Central South University, 4Sun Yat-Sen University, 5Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ) yejh22, qin-s23, liyinghu20 @mails.tsinghua.edu.cn

# Abstract

Existing studies explore the explainability of Grammatical Error Correction (GEC) in a limited scenario, where they ignore the interaction between corrections and explanations and have not established a corresponding comprehensive benchmark. To bridge the gap, this paper first introduces the task of EXplainable GEC (EXGEC), which focuses on the integral role of correction and explanation tasks. To facilitate the task, we propose EXCGEC, a tailored benchmark for Chinese EXGEC consisting of 8,216 explanation-augmented samples featuring the design of hybrid edit-wise explanations. We then benchmark several series of LLMs in multi-task learning settings, including post-explaining and pre-explaining. To promote the development of the task, we also build a comprehensive evaluation suite by leveraging existing automatic metrics and conducting human evaluation experiments to demonstrate the human consistency of the automatic metrics for free-text explanations. Our experiments reveal the effectiveness of evaluating free-text explanations using traditional metrics like METEOR and ROUGE, and the inferior performance of multi-task models compared to the pipeline solution, indicating its challenges to establish positive effects in learning both tasks.

# Code & Data — https://github.com/THUKElab/EXCGEC

(a) GEC Source Target Source (b) GEE Explanation Target Target (c) EXGEC Source Post-explaining 1 Pre-explaining Explanation Source: 接待的游客约在6000人次左右。 The number of tourists received is about 6000 people approximately. Target: 接待的游客约在6000人次。 The number of tourists received is about 6000 people. Error Type: 词语冗余 (Word Redundancy) Error Severity: 2 Error Description: 表达数量时，【约】和【左右】都含有大约的意思，同时使用 这两个词汇导致语义重复。使用一个即可清晰表达含义，所以应该删除【左右】。 When expressing quantities, both【 约 】and【 左 右 】have similar $\operatorname { \Pi } _ { 1 } ^ { 1 }$ meanings of "approximately". Using both of these words together $\begin{array} { c } { { 1 } } \\ { { 1 } } \end{array}$ results in semantic redundancy. Using only one of them is sufficient to $\begin{array} { c } { { 1 } } \\ { { 1 } } \end{array}$ convey the meaning clearly. Therefore, 【左右】should be deleted.

# Introduction

Despite the notable advancements in Grammatical Error Correction (GEC) (Bryant et al. 2023; Ye et al. 2023a; Li et al. 2025), there still exists a lack of profound examination into the explainability of GEC (Dwivedi et al. 2023), which is critical in educational scenarios for L2 (second language)- speakers (Wang et al. 2021). These mainstream users, who often face challenges in creating grammatically accurate and fluent texts, may be confused or even misguided if they are provided with limited access to only corrective texts. Therefore, augmenting the explainability of GEC is unquestionably beneficial for the progression of GEC as well as related fields, such as essay scoring (Stahl et al. 2024), intelligent tutoring systems (Montenegro-Rueda et al. 2023).

Figure 1: Task definitions of GEC, GEE, and EXGEC. We highlight 【evidence words】, correction , linguistic knowledge, error causes, and revision advice parts.

As illustrated in Figure 1, existing tasks like GEC and Grammatical Error Explanation (GEE) typically address either correction or explanation, ignoring the interaction between the two. To bridge the gap, we introduce the task of EXplainable Grammatical Error Correction (EXGEC). By integrating these two tasks, EXGEC enables systems to elucidate the linguistic knowledge and reasoning mechanism underlying predicted corrections, thereby achieving the best of both worlds. Additionally, EXGEC can function as a test bed for determining the explainable abilities of large language models (LLMs) and identifying any unintended biases and risks in educational scenarios.

To facilitate EXGEC, we present EXCGEC, a tailored benchmark for Chinese EXGEC, featuring the design of hybrid edit-wise explanations. Each explanation, based on a particular edit, consists of three elements: 1) Error types, which allow learners to absorb syntax and semantic knowledge in an inductive way (Fei et al. 2023). We establish a hierarchical and pragmatic two-tier taxonomy for Chinese grammatical errors. 2) Error severity levels ranging from 1 $\sim 5$ points, which are beneficial to prioritize core corrections. 3) Error descriptions, presented as the form of natural language explanation (Camburu et al. 2018; He et al. 2023), provide evidence words, relevant linguistic knowledge or syntax rules, error causes, and revision advice for edits. The edit-wise design provides more detailed and faithful guidance for learners, allowing them to comprehend each grammatical error committed. This is unlikely achievable for other designs such as example-based (Kaneko et al. 2022) or sentence-level explanations (Nagata et al. 2021).

Stimulated by the recent success of synthetic data generation (Shum, Diao, and Zhang 2023; Whitehouse, Choudhury, and Aji 2023), we employ a semi-automatic dataset construction solution to enhance efficiency, while minimising annotation costs. Initially, we synthesize the evaluation part of EXCGEC by prompting GPT-4 (Liu et al. 2024). Then we hire experienced annotators to filter out invalid data and concurrently provide a detailed analysis of the invalid data, ensuring the quality of our dataset (Ding et al. 2024). We finally obtain 8,216 clean explanation-augmented samples for benchmarking. Additionally, We utilize existing automatic metrics to evaluate the performance. Particularly for error descriptions, we conduct a human evaluation experiment to ascertain the correlation between the metrics and human judgements, thus demonstrating their effectiveness.

Based on the benchmark, we develop EXGEC multi-task baseline models that can perform both the correction and explanation tasks in either post-explaining (correct-thenexplain) or pre-explaining (explain-then-correct) sequences. Particularly, we design Correct-Then-Explain (COTE) decoding algorithm for post-explaining models. Benchmarking various series of open-source LLMs has yielded several intriguing findings. For example, post-explaining models display higher performance than pre-explaining models. However, both of them under-perform the pipeline solution. Moreover, COTE significantly enhances performance by alleviating the alignment workload for the LLMs. Our contributions in this paper are listed as follows:

• We introduce the EXGEC task and establish a corresponding benchmark consisting of a Chinese EXGEC dataset and a comprehensive set of metrics, contributing

to the stable development of the field of EXGEC. • We develop EXGEC baseline models and investigate the abilities of various LLMs using our proposed benchmark. • We conduct detailed analyses on our proposed dataset and baselines to gain further insights. Human evaluation experiments are also conducted to confirm the effectiveness of automatic metrics for error descriptions.

# Related Work

Explainable GEC. Exploration of explainable GEC has witnessed a paradigm shift from fine-tuning to prompting (Zhao et al. 2024). EXPECT (Fei et al. 2023) is an explainable GEC dataset annotated with evidence words and error types based on the standard GEC benchmark (Bryant et al. 2019). However, EXPECT falls short of flexibility due to the lack of natural language explanations. To fill the gap, Song et al. (2023) propose the task of grammatical error explanation. They observe that GPT-4 suffers from identifying and explaining errors with limited access to only parallel source-target pairs. To address this issue, they fine-tune an extra LLM as an edit extractor trained on synthesized data. On the other hand, a similar task called feedback comment generation, focuses on sentence-level explanations. However, it suffers from expensive costs associated with data annotation (Nagata, Inui, and Ishikawa 2020). Furthermore, it is explored with limited access to only a subset of English grammatical error types due to the complexity of the task (Nagata 2019). In conclusion, all these studies do not establish a comprehensive benchmark integrating both the tasks of GEC and GEE, and thus lack in-depth exploration in multi-task learning the both tasks. However, our work is the first to propose a systematic framework for EXCGEC.

Chinese GEC. The research on CGEC (Ye et al. 2023a; Ye, Li, and Zheng 2023) has also come a long way recently, along with a series of CGEC datasets (Zhao et al. 2018). Similar to those in English, Chinese grammatical errors can also be categorized into different error types. CLG (Ma et al. 2022) divides Chinese grammatical errors into 6 categories: Structural Confusion, Improper Logicality, Missing Component, Redundant Component, Improper Collocation, and Improper Word Order. However, the taxonomy of CLG is targeted toward grammatical errors made by native speakers and thereby can not cover those made by L2 speakers. To fill the gap, we design a two-tier hierarchical taxonomy, which is capable of covering most grammatical errors.

# Task Definition Grammatical Error Correction

GEC (Schneider and McCoy 1998) has been studied for decades, witnessing the shift from rule-based methods to LLM-based methods. Formally, given an ungrammatical text (source text) $X = \{ x _ { 1 } , x _ { 2 } , \cdot \cdot \cdot , x _ { T } \}$ , a GEC model is required to correct $X$ into a grammatically correct counterpart (target text) $Y = \{ y _ { 1 } , y _ { 2 } , \cdot \cdot \cdot , y _ { T ^ { \prime } } \}$ without changing the original semantic as far as possible. Typically, GEC is usually treated as a sequence-to-sequence $( { \mathrm { S e q } } 2 { \mathrm { S e q } } )$ task, the training objective of which is formulated as follows:

$$
\mathcal { L } _ { \mathrm { G E C } } = - \sum _ { t = 1 } ^ { T ^ { \prime } } \log P ( y _ { t } \mid Y _ { < t } , X ) .
$$

# Grammatical Error Explanation

GEE (Song et al. 2023) has received much attention recently and has been explored in several methodologies, including sentence-level explanation and edit-wise explanation. Since sentence-level explanations suffer from over-generalization and confusion especially when a sentence contains multiple grammatical errors, this work focuses solely on edit-wise explanations. Given a source text $X$ and its target counterpart $Y$ , the GEE model needs to explain each grammatical error $e _ { i }$ in $X$ . Specifically, GEE is typically solved in a two-step pipeline consisting of edit extraction and editwise explanation. 1) Edit extraction produces an edit set $E = \{ \stackrel { \textstyle - } { e _ { 1 } } , e _ { 2 } , \cdot \cdot \cdot , e _ { n } \}$ that represent grammatical errors in $X$ and also clarify the transformation from ungrammatical segments of $X$ to target segments of $Y$ . Typically, an edit contains four key elements: source position $s p$ , source content $s c$ , target position $t p$ , and target content tc. The process of edit extraction can be easily accomplished using alignment-based evaluation toolkits like ERRANT (Bryant, Felice, and Briscoe 2017) and CLEME (Ye et al. 2023b, 2024). 2) Edit-wise explanation generates a set of explanations $E ^ { \prime } ~ = ~ \{ e _ { 1 } ^ { \prime } , e _ { 2 } ^ { \prime } , \bar { { \cdot \cdot \cdot } } ~ , e _ { n } ^ { \prime } \}$ , with each explanation $e _ { i } ^ { \prime }$ corresponding to $e _ { i }$ , given $X$ and $Y$ . Although the design of explanation varies across related work (Song et al. 2023; Zhao et al. 2024), the typical training objective of GEE models is presented as follows:

$$
E = f ( X , Y ) ,
$$

$$
{ \mathcal { L } } _ { \mathrm { G E E } } = - \sum _ { i = 1 } ^ { n } \log P ( e _ { i } ^ { \prime } \mid X , Y , e _ { i } ) ,
$$

where $f : ( X , Y )  E = \{ ( s p _ { i } , s c _ { i } , t p _ { i } , t c _ { i } ) \} _ { i = 1 } ^ { n }$ is the edit extraction function used to extract edits of $X$ and $Y$ , and $n$ is the number of edits.

Existing studies (Song et al. 2023; Fei et al. 2023) focus on developing GEE models that can generate explanations. However, an extra GEC model is compulsory for GEE models to work, thus resulting in an issue of low efficiency.

# Explainable Grammatical Error Correction

To get rid of the drawbacks brought by the nature of GEE, we propose the EXGEC task which aims to perform both correction and explanation tasks simultaneously. The motivation for combining these two tasks majorly falls on two aspects. First, a branch of existing studies (Wiegreffe and Marasovic 2021; Hartmann and Sonntag 2022; Li et al. 2022, 2024) have demonstrated training with access to human explanations can improve model performance. It is also intuitive that either of the GEC and GEE tasks can mutually benefit from each other when training in a multi-task manner. Second, it is more time-saving and cost-efficient to deploy a single EXGEC model rather than two detached models in foreign language education platforms.

In this task, the only input element is an ungrammatical source text $X$ , and the EXGEC model learns to output both the grammatical target text $Y$ and explanations $E ^ { \prime }$ . Similar to GEE, EXGEC follows the edit-wise style of explanation, and it is categorized into two different settings by the order of correction and explanation tasks, with the basic scheme of multi-task learning.

Post-explaining. Models are trained first to generate target texts (Camburu et al. 2018), which allows the explanations to be explicitly conditioned on the target texts, thus ensuring high faithfulness of explanations towards the target texts. The training objective is as follows:

$$
{ \mathcal { L } } _ { \mathrm { p o s t } } = - \sum _ { t = 1 } ^ { T ^ { \prime } } \log P ( y _ { t } \mid Y _ { < t } , X ) - \sum _ { i = 1 } ^ { n } \log P ( e _ { i } ^ { \prime } \mid X , Y , e _ { i } ) .
$$

The inference of post-explaining models is as follows:

$$
\hat { Y } = \mathrm { E X G E C } _ { \mathrm { p o s t } } ( X ) ,
$$

$$
\hat { E } ^ { \prime } = \mathrm { E X G E C } _ { \mathrm { p o s t } } ( X , Y , f ( X , \hat { Y } ) ) .
$$

With the target texts generated ahead, post-explaining models can output explanations conditioned on the specific edits extracted by an aligning process, thus improving the accuracy and faithfulness of explanations.

Pre-explaining. This type of model is trained in converse order, whose mechanism is similar to the Chain of Thought (CoT) technique. Pre-explaining models are supposed to make full use of synthesized explanations to generate elaborated target texts. With minimal modification from Equation (4), the training objective of pre-explaining models is as follows:

$$
\mathcal { L } _ { \mathrm { p r e } } = - \sum _ { i = 1 } ^ { n } \log P ( e _ { i } ^ { \prime } \mid X ) - \sum _ { t = 1 } ^ { T ^ { \prime } } \log P ( y _ { t } \mid Y _ { < t } , X , E ^ { \prime } ) .
$$

Notably, pre-explaining models may struggle to generate well-formed edit-wise explanations due to the inaccessibility to the edit extraction function $f$ , which necessitates both the source and the target texts. Similarly, the inference of pre-explaining models is presented as follows:

$$
\begin{array} { r } { \hat { E } ^ { \prime } = \mathrm { E X G E C } _ { \mathrm { p r e } } ( X ) , \qquad } \\ { \hat { Y } = \mathrm { E X G E C } _ { \mathrm { p r e } } ( X , E ^ { \prime } ) . } \end{array}
$$

# EXCGEC Benchmark

To facilitate the development of EXGEC task, we construct EXCGEC, the first benchmark for explainable Chinese GEC particularly. As illustrated in Figure 2, we begin with the process of data curation, which consists of Explanation Design, Explanation Synthesizing, Explanation Refinement, and Analysis. Then we gain an in-depth understanding of GPT-4 (Achiam et al. 2023) by further analyzing the generated explanations, where we summarize common failure modes in invalid instances. Finally, we explain the evaluation for both the correction and the explanation tasks.

Dataset Curation Inference (b) Extract Edits Prompt: You are explaining grammatical errors. Source: 我希欢吃平果。 ： Source 1 S 血 Fine-tune Target: 我喜欢吃苹果。 EditE1x: t[r1,ac2]te希d→Ed[i1t,s2] 喜 3 Target Edits Labeled Data Clean Data Edit 2: [4, 5] 平 → [4, 5] 苹 （c) Explain Model Output ? Edit Extraction Correct Correction: 我喜欢吃苹果。 Edit 1: [1, 2] 希 → [1, 2] 喜 Error Type: Phonetic Confusion Error Text Extract Edit Error Severity: 3 Error Description Edit ↓ Explain Edit 2: [4, 5] 平 $ [ 4 ,$ 5] 苹 Error Type: Glyph Confusion Error Language Error Severity: 3 Error Description Learner

Table 1: Hierarchical taxonomy of grammatical error types.   

<html><body><table><tr><td>Major Type</td><td>Minor Type</td></tr><tr><td rowspan="3">Punctuation-level Error</td><td>标点冗余 (Punctuation Redundancy)</td></tr><tr><td>标点丢失 (Punctuation Missing)</td></tr><tr><td>标点误用(Punctuation Misuse)</td></tr><tr><td rowspan="5">Spelling-level Error</td><td>字音混淆错误 (Phonetic Confusion Error)</td></tr><tr><td>字形混淆错误(Glyph Confusion Error)</td></tr><tr><td>词内部字符异位错误</td></tr><tr><td>(Internal CharacterMisplacement Error)</td></tr><tr><td>命名实体拼写错误(Named Entity Misspelling)</td></tr><tr><td rowspan="3">Word-level Error</td><td>词语冗余 (Word Redundancy)</td></tr><tr><td>词语丢失(Word Missing)</td></tr><tr><td>词语误用(Word Misuse)</td></tr><tr><td rowspan="3">Sentence-levelError</td><td>词序不当 (Improper Word Order)</td></tr><tr><td>逻辑不通 (Illogicality)</td></tr><tr><td>句式杂糅 (Run-on Sentence)</td></tr><tr><td rowspan="3">Other Special Error</td><td>照应错误(Inconsistency Error)</td></tr><tr><td>歧义错误(Ambiguity Error)</td></tr><tr><td>语气不协调(Inconsistent Tone)</td></tr><tr><td></td><td>Other</td></tr></table></body></html>

# Explanation Design

In the pursuit of comprehensiveness and plausibility, we adopt a hybrid strategy for edit-wise explanations, where each edit is explained through three aspects, including error type labels, error severity levels, and free-text error descriptions. 1) Error type labels allow language learners to comprehend and inductively infer syntax and grammar rules. In particular, we employ a two-tier hierarchical taxonomy including 5 major types and 16 minor types shown in Table 1, inspired by authoritative linguistic books (Huang and Liao 2011; Shao 2016). Detailed descriptions of various error types are included in the supplementary materials. If an edit covers multiple error types, we select the one with the highest granule. 2) Error severity levels, ranging from 1 to 5 points, indicate the significance of a specific grammatical error. 3) Error descriptions are the most crucial and flexible element. These provide keywords, pertinent linguistic knowledge, causes of errors, and revision guidance in a free-text format. We stipulate well-defined error descriptions should meet three nonoverlapping principles: fluency, reasonability (making sense to humans), and faithfulness (targeted to a specific edit). To ensure reasonability and faithfulness, the error description must mostly conform to the syllogism form of deductive reasoning: [major premise: semantic rules and related knowledge], [minor premise: the reason for the error in the text], and [explain how to correct it]. Further, any evidence from the source $X$ must be enclosed within special markers 【 】. Similarly, correction content that occurs in the target sentence $Y$ must be enclosed within $\{ \}$ , as indicated in Figure 1.

# Explanation Synthesizing

Annotating high-quality explanations on a large scale poses a huge challenge to our benchmark construction. Hence, we leverage GPT-4 to synthesize edit-wise explanations efficiently. To achieve this, we first select 10,000 parallel samples across 6 existing benchmarks or datasets of Chinese GEC, including FCGEC (Xu et al. 2022), YACLC (Wang et al. 2021), MuCGEC (Zhang et al. 2022), NaCGEC (Ma et al. 2022), NLPCC (Zhao et al. 2018) and HSK (Zhang 2009). The details are listed in Table 2. We pick out only the samples with changed reference sentences to maximize training efficiency (Zhang et al. 2022). We select the reference sentence with the most edits as the target sentence if a sample is annotated with multiple reference sentences. Then, we prompt GPT-4 to generate edit-wise explanations following in-context learning. To ensure the faithfulness of the synthesized explanation, we first extract edits using the toolkit CLEME (Ye et al. 2023b). Inspired by Li et al. (2022), we then employ the Rationalization Prompting (RP) strategy, where we concatenate task definition, demonstrations, and a parallel sample $( X , Y )$ with extracted edits $E = \{ e _ { 1 } , e _ { 2 } , \hat { \cdot } \cdot \cdot , e _ { n } \}$ as the prompt. For each error type, we provide the definition, a suggested template of error description, and a demonstration. The prompt is listed in the

Table 2: Dataset statistics of the EXCGEC benchmark.   
Algorithm 1: COTE Decoding Algorithm   

<html><body><table><tr><td>Dataset</td><td>Sentences</td><td>Edits/Sent.</td><td>Chars/Sent.</td></tr><tr><td>FCGEC</td><td>41,340</td><td>1.0</td><td>53.1</td></tr><tr><td>YACLC-minimal-dev</td><td>1,839</td><td>2.9</td><td>25.9</td></tr><tr><td>MuCGEC-dev</td><td>1,137</td><td>3.2</td><td>38.5</td></tr><tr><td>NaCGEC-dev</td><td>500</td><td>1.1</td><td>56.2</td></tr><tr><td>NLPCC-test</td><td>2.000</td><td>2.0</td><td>29.7</td></tr><tr><td>HSK</td><td>156.870</td><td>1.4</td><td>27.2</td></tr><tr><td>EXCGEC(FCGEC)</td><td>2,308</td><td>1.1</td><td>55.1</td></tr><tr><td>EXCGEC(YACLC)</td><td>1,235</td><td>3.5</td><td>24.3</td></tr><tr><td>EXCGEC(MuCGEC-dev)</td><td>789</td><td>3.3</td><td>40.4</td></tr><tr><td>EXCGEC(NaCGEC-dev)</td><td>449</td><td>1.1</td><td>56.1</td></tr><tr><td>EXCGEC (NLPCC-test)</td><td>1,611</td><td>1.7</td><td>28.9</td></tr><tr><td>EXCGEC (HSK)</td><td>1,824</td><td>2.1</td><td>32.0</td></tr><tr><td>EXCGEC-train</td><td>5,966</td><td>2.0</td><td>38.7</td></tr><tr><td>EXCGEC-dev</td><td>750</td><td>2.0</td><td>38.9</td></tr><tr><td>EXCGEC-test</td><td>1,500</td><td>2.0</td><td>39.2</td></tr><tr><td>EXCGEC (all)</td><td>8,216</td><td>2.0</td><td>38.8</td></tr></table></body></html>

supplementary materials.

# Explanation Refinement and Analysis

Benefiting from the extensive knowledge acquired during the large-scale pre-training process, GPT-4 can generate fluent, reasonable, and plausible explanations in most cases, meeting the requirements with specified instructions. However, GPT-4 is not guaranteed to produce all high-quality explanations due to hallucination, and the patterns of those invalid explanations are referred to as failure modes. Therefore, we hired 12 native speakers, all of whom are Chinese post-graduated students specializing in Chinese linguistics, to screen out invalid explanations. Before formal annotation, we compile the annotation guidelines and all the annotators receive intensive training. Two authors of the paper, who are also in charge of compiling the annotation guidelines, have made sure that their annotation accuracies are over $90 \%$ on testing samples. We make sure that each formal sample is checked by at least two annotators. We finally obtained 8,216 clean samples out of 10,000 samples. We further investigate the failure modes of these invalid explanations, which are provided in the supplementary materials.

# Automatic Metrics

To promote the efficient development of EXGEC systems, we introduce a comprehensive suite of automatic metrics for both correction and explanation parts. Additionally, we conduct a human evaluation experiment in Section Analysis to demonstrate the alignment of the metrics used for assessing error descriptions with human judgments.

Correction. We employ CLEME (Ye et al. 2023b) and ChERRANT (Zhang et al. 2022) to evaluate the correction performance. Both are edit-based metrics that output $\mathrm { P / R / F _ { 0 . 5 } }$ scores, which have been proven reliable metrics for GEC on CoNLL-2014 (Ye et al. 2023b).

Explanation. Since an edit-wise explanation consists of three critical elements, we define respectively automatic

Input: Source text $X$ , a post-explaining model $\mathcal { M }$ , and the edit extraction function $f$ .   
Output: Target text $\hat { Y }$ , and explanations $\hat { E } ^ { \prime }$ .   
1: $\hat { Y } \gets \mathrm { B e a m S e a r c h } ( \mathcal { M } ( \mathrm { J s o n } ( X ) ) )$   
2: $\hat { E } ^ { \prime }  \varnothing$   
3: if ${ \hat { Y } } = X$ then   
4: return Yˆ , Eˆ′   
5: end if   
6: $\boldsymbol { E }  f ( X , \hat { Y } )$   
7: $\hat { E } ^ { \prime } \gets \mathrm { T o p - P } ( \mathcal { M } ( \mathrm { J s o n } ( X , Y , E ) ) )$   
8: return Yˆ , Eˆ′

metrics for them. 1) Accuracy and Macro-F1 scores are computed for error type clarification, following the conventional evaluation protocol of text clarification (Li et al. 2020). 2) We report the mean absolute error (MAE) to show the deviation of hypothesis error severity levels towards ground truth ones. 3) We employ various metrics for evaluating the freetext explanation descriptions considering both the reproductivity and efficiency, including BLEU (Papineni et al. 2002; Clinciu, Eshghi, and Hastie 2021), METEOR (Banerjee and Lavie 2005), ROUGE (Lin 2004).

# Method

Training. To streamline the training process covering all the tasks mentioned previously, we treat all of them as a unified Seq2Seq task. To achieve this, we linearize the data in the format of JSON (Gao et al. 2023). This structured approach simplifies the process of output parsing involving three elements of edit-wise explanations, and provides a consistent and controllable view to distinguish tasks, enabling the model to understand essential task elements and their relations. With this uniform format stipulation, we can train all models using the same smooth cross-entropy loss, regardless of the specific task.

Inference. For post-explaining EXGEC models, we design a specific Correct-Then-Explain decoding algorithm called COTE, which is presented in Algorithm 1. First, we employ the greedy beam search decoding strategy for the correction part, which is beneficial to relieve the overcorrection problem that is common in LLMs. Then, we apply CLEME to extract edits. Notably, we merge adjacent edits with a distance of less than 2 characters to avoid fragmented edits. Finally, we leverage the Top-p decoding strategy for generating explanations, encouraging diversified natural language explanations. It is worth noting that COTE is not accessible to pre-explaining models since the edit extraction tool necessitates both a source text and a target text.

# Experiments Experimental Settings

Backbones. We benchmark mainstream LLMs including Qwen-1.5 (Bai et al. 2023), Llama-3 (Touvron et al. 2023), and DeepSeek (Bi et al. 2024). For these LLMs, we experiment with their base and chat (or instruct) versions to investigate whether further alignment training benefits the task. All experimental results are averaged over three runs with different random seeds on EXCGEC-test in Table 2. More training details are reported in the supplementary materials.

Table 3: Main results of multi-task learning models. Results of post-explaining models are listed in the top block, while those f pre-explaining models are in the bottom block.   

<html><body><table><tr><td rowspan="2"></td><td rowspan="2">Model</td><td colspan="2">Correction↑</td><td colspan="8">Explanation</td></tr><tr><td>CLEME (P/R/F0.5)</td><td>ChERRANT(P/R/F0.5)</td><td>Hit个</td><td>Miss↓</td><td>Acc↑</td><td>F1↑ MAE↓</td><td>BLEU个</td><td></td><td>METEOR↑</td><td>ROUGE-(1/2/L)↑</td></tr><tr><td rowspan="6">Post</td><td>Qwen1.5-7B-base</td><td>26.00 / 26.54/26.10</td><td>33.87 / 20.16 /29.81</td><td>67.29</td><td>56.81</td><td>60.99</td><td>29.82</td><td>0.80</td><td>15.22</td><td>39.05</td><td>49.74/ 23.28/34.32</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>20.921/21.201/26.54</td><td>36.74/17.26/29.8</td><td>61.54</td><td>54.8</td><td>61.98</td><td>29.62</td><td>0.75</td><td>15.49</td><td>38.8</td><td>49.32/23.15/3524</td></tr><tr><td>Llama3-8B-instruct</td><td>21.33/26.05/22.14</td><td>29.00/19.40/26.39</td><td>61.40</td><td>55.71</td><td>59.16</td><td>25.63</td><td>0.88</td><td>14.70</td><td>36.89</td><td>49.41/23.54/34.87</td></tr><tr><td>DeepSeek-7B-base</td><td>26.21/7.00/16.92 ·······</td><td>36.00/7.04/19.75</td><td>69.92</td><td>85.39</td><td>60.64</td><td>26.47</td><td>0.79</td><td>15.07</td><td>38.05</td><td>50.19/24.10/34.90</td></tr><tr><td>DeepSeek-7B-chat</td><td>25.46 / 18.51 / 23.68</td><td>34.02 / 15.75 /27.62</td><td>67.52</td><td>66.64</td><td>58.11</td><td>24.45</td><td>0.84</td><td>13.94</td><td>36.97</td><td>48.66 /22.70 / 34.23</td></tr><tr><td rowspan="3">Pre</td><td>Qwen1.5-7B-chat</td><td>13.76 /13.42 / 13.69</td><td>19.27 / 9.93 / 16.22</td><td>29.49</td><td>80.24</td><td>23.35</td><td>8.22</td><td>1.17</td><td>7.75</td><td>27.67</td><td>40.47 /15.00 /28.20</td></tr><tr><td>Llama3-8B-instruct</td><td>7.12 / 11.17 /7.68</td><td>10.86/ 8.57/10.31</td><td>23.88</td><td>73.06</td><td>24.31</td><td>8.78</td><td>1.21</td><td>5.78</td><td>23.07</td><td>37.57 /13.47/27.19</td></tr><tr><td>DeepSeek-7B-chat</td><td>9.93 /8.26/9.55</td><td>14.28 /7.07/11.86</td><td>24.72</td><td>78.67</td><td>19.12</td><td>5.84</td><td>1.29</td><td>5.91</td><td>23.95</td><td>37.59 / 13.11/ 26.78</td></tr></table></body></html>

Table 4: Ground truth results of multi-task learning models. We report the explanation performance (right block) of postexplaining models conditioned on source texts and ground truth target texts. Contrarily, we report the correction performance (left block) of pre-explaining models conditioned on source sentences and ground truth explanations.   

<html><body><table><tr><td rowspan="2">Model</td><td colspan="2">Correction↑</td><td colspan="10">Explanation</td></tr><tr><td>CLEME (P/R/F0.5)</td><td>ChERRANT(P/R/F0.5)</td><td>Hit个</td><td>Miss↓</td><td>Acc↑</td><td>F1↑</td><td>MAE↓</td><td>BLEU个</td><td></td><td>METEOR↑</td><td>ROUGE- (1/2/L)</td></tr><tr><td>Qwen1.5-7B-chat</td><td>62.59 / 87.35 / 66.35</td><td>67.58 / 69.53 / 67.96</td><td>99.93</td><td>0.43</td><td>81.53</td><td>39.56</td><td></td><td>0.73</td><td>17.88</td><td>41.40</td><td>51.73 /28.81 /36.51</td></tr><tr><td>Llama3-8B-instruct</td><td>69.10 /90.90 / 72.58</td><td>73.75 / 74.37 / 73.87</td><td>99.63</td><td>1.67</td><td>85.99</td><td>41.84</td><td>0.78</td><td></td><td>20.73</td><td>42.98</td><td>54.60 / 29.64/40.04</td></tr><tr><td>DeepSeek-7B-chat</td><td>41.12 / 79.02 / 45.48</td><td>48.35 /53.20/49.25</td><td>99.93</td><td>0.40</td><td>81.17</td><td>35.93</td><td>0.74</td><td></td><td>19.57</td><td>42.32</td><td>53.12/ 28.03 / 38.59</td></tr></table></body></html>

Evaluation. We obtain the metric results using public toolkits including ROUGE (Lin 2004), NLTK (Bird and Loper 2004), and scikit-learn (Pedregosa et al. 2011). Particularly, we observe many hypothesis edits are not covered by the corresponding reference edits, making it impossible to subsequently evaluate the explanations for these edits. To address this, we introduce two extra indicators, namely Hit and Miss rates. A hypothesis edit overlapping with a reference edit is designated as a hit edit, while a reference edit without any match with hypothesis edits is deemed a miss edit. The hit rate is defined as the ratio of hit edits to all hypothesis edits, and the miss rate as the ratio of miss edits to all reference edits. Only the hit edits are used to calculate the evaluation outcomes for explanations.

# Results of Multi-task Models

Table 3 presents the main results of multi-task models.

Post-explaining models outperform pre-explaining models. Concerning the correction aspect, all post-explaining models consistently obtain higher $\mathrm { F _ { 0 . 5 } }$ scores than preexplaining models, regardless of the applied backbones. A similar pattern is observed in the explanation part, where all the pre-explaining models invariably underperform their post-explaining counterparts. This suggests complexity for LLMs to directly explain grammatical errors without auxiliary information like target sentences or extracted edits. And once pre-explaining models generate flawed explanations, the ensuing distraction impedes their ability to accurately correct the source text.

Chat models are superior to base models. For postexplaining models, we observe all chat or instruct models gain slightly higher $\mathrm { F _ { 0 . 5 } }$ correction scores, and they also marginally outperform their base version counterparts in the explanation task. It indicates that additional alignment training (Wang et al. 2023) can benefit the EXGEC task.

# Ground Truth Results

To examine the isolated performance of multi-task models, we introduce partial ground truth information in advance during the formal inference stage. This is achieved by preinserting ground truth corrections or explanations into the decoding phase prior to formal inference. Specifically, we utilize ground truth target texts for post-explaining and evaluate the performance of the explanation task. Conversely, we provide ground truth explanations for pre-explaining and assess the performance of the correction task. This approach enables a detailed analysis of each task’s performance under oracle conditions. The results, as depicted in Table 4, reveal that the incorporation of ground truth information significantly enhances performance. Notably, post-explanatory models equipped with ground truth corrections exhibit a marked improvement in explanatory performance across all LLMs. This observation extends to post-explanatory models with ground truth explanations, suggesting that previously generated low-quality content adversely affects subsequent generative processes.

<html><body><table><tr><td rowspan="2">Model</td><td colspan="2">Correction↑</td><td colspan="8">Explanation</td></tr><tr><td>CLEME (P/R/F0.5)</td><td>ChERRANT(P/R/F0.5)</td><td>Hit个</td><td>Miss↓</td><td>Acc↑</td><td>F1个</td><td>MAE↓</td><td>BLEU个</td><td>METEOR↑</td><td>ROUGE- (1/2/L)</td></tr><tr><td>Post-explaining</td><td>28.31 /21.21/ 26.54</td><td>36.74 / 17.26 /29.98</td><td>68.94</td><td>64.83</td><td>61.98</td><td>29.62</td><td>0.75</td><td>15.49</td><td>38.88</td><td>50.32/ 24.25 / 35.24</td></tr><tr><td>Pre-explaining</td><td>13.76 /13.42 /13.69</td><td>19.27 /9.93 /16.22</td><td>29.49</td><td>80.24</td><td>23.35</td><td>8.22</td><td>1.17</td><td>7.75</td><td>27.67</td><td>40.47 / 15.00 / 28.20</td></tr><tr><td>GEC-GEE Pipeline</td><td>32.45 /23.93/ 30.29</td><td>40.50 / 19.58 / 33.37</td><td>72.00</td><td>63.10</td><td>65.76</td><td>32.77</td><td>0.70</td><td>16.41</td><td>40.04</td><td>51.07 / 24.92 / 35.89</td></tr></table></body></html>

Table 5: Comparison of the multi-task solutions and the GEC-GEE pipeline solution based on Qwen1.5-7B-chat.

Table 6: Ablation results of COTE from the same model.   

<html><body><table><tr><td></td><td>Hit个</td><td>Miss↓</td><td>Acc↑</td><td>F1↑</td><td>MAE↓</td><td>ROUGE- (1/2/L)个</td></tr><tr><td>wCOTE</td><td>99.93</td><td>0.43</td><td>81.53</td><td>39.56</td><td>0.74</td><td>51.73/25.81/36.51</td></tr><tr><td>w/o COTE</td><td>49.64</td><td>54.01</td><td>42.51</td><td>17.77</td><td>0.93</td><td>46.35 / 19.34 /31.28</td></tr></table></body></html>

Table 7: Correlations between human judgements $\mathbf { \left. A _ { 1 } \right. }$ and $\mathbf { A } _ { 2 } { \mathrm { : } }$ ) and metrics results for error descriptions.   

<html><body><table><tr><td></td><td>Pearson</td><td>Spearson</td></tr><tr><td>Human v.s.BLEU</td><td>0.9222</td><td>0.6571</td></tr><tr><td>Human v.s.METEOR</td><td>0.9280</td><td>0.7714</td></tr><tr><td>Human v.s.ROUGE-1</td><td>0.9464</td><td>0.8286</td></tr><tr><td>Human v.s. ROUGE-2</td><td>0.9175</td><td>0.4857</td></tr><tr><td>Human V.s. ROUGE-L</td><td>0.9352</td><td>0.6571</td></tr><tr><td>A1 V.s. A2</td><td>0.9874</td><td>0.9429</td></tr></table></body></html>

# Comparison with Pipeline

We compare multi-task models and a GEC-GEE pipeline with COTE in Table 5. It indicates that the pipeline solution can improve both the correction and the explanation performance compared to multi-task models, highlighting the challenges of learning multi-task models for EXCGEC. However, adopting the pipeline solution requires heavy deployment and training costs. We speculate that LLMs with only 7B parameters cannot establish intimate interaction of correction and explanation tasks.

# Analysis

# Ablation Results

We conduct ablation studies on Qwen1.5-7B-chat to provide in-depth insights into post-explaining models. We also study the effect of model sizes and provide a case study for different LLMs in the supplementary materials.

Effect of COTE. We introduce COTE that provides gold alignment for post-explaining models, thus unburdening LLMs during the inference stage. The impact of COTE is quantitatively examined in this section. We provide the postexplaining model with ground truth target texts, which allows us to focus on the explanation performance. The results presented in Table 6 reveal a huge performance drop if we do not leverage COTE, especially the hit rate and the miss rate. This demonstrates the effectiveness of COTE.

# Human Evaluation for Error Descriptions

We adopt traditional metrics for assessing the quality of generated error descriptions mainly for their reproductivity and efficiency (Clinciu, Eshghi, and Hastie 2021). However, their reliability requires further validation. Therefore, this section attempts to demonstrate the suitability of these metrics through their corrections with human judgments. We assign two human annotators to score the error descriptions generated by all 6 post-explaining models, with the scoring scale from 0 to 100. For each sample, the annotators are instructed to concurrently evaluate all the error descriptions, referencing a gold explanation generated by GPT-4 to guarantee a rigorous and reliable assessment. Additional details are delineated in the supplementary materials.

We report Pearson and Spearson correlations between the metric results and the human judgments in Table 7. We observe the inter-annotator correlations are close to 1, meaning it is relatively easy to determine the quality of error descriptions for human annotators. Most metrics achieve moderate or high correlations with human judgments, which means that it is relatively reasonable to use simple n-gramsbased metrics to evaluate the quality of error descriptions efficiently. Among various metrics, ROUGE-1 achieves the highest correlations, followed by METEOR. All the introduced metrics show moderate or high correlations, indicating that it is advisable to employ them as proxies for human evaluation. We provide detailed annotation guidance and rating rules in the supplementary materials.

# Conclusion

We propose and formulate the task of EXGEC, establishing the interaction of correction and explanation tasks. To develop the task, we propose the EXCGEC benchmark, based on which we build baseline models. Extensive experiments and analyses reveal several challenges of the task. We hope this paper can serve as a starting point for future exploration.