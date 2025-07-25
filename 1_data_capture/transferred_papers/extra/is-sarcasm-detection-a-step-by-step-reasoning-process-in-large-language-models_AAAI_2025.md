# Is Sarcasm Detection A Step-by-Step Reasoning Process in Large Language Models?

Ben Yao1, Yazhou Zhang2,3\*, Qiuchi Li1\*, Jing Qin3

1University of Copenhagen 2Tianjin University 3The Hong Kong Polytechnic University yzhou zhang@tju.edu.cn, qiuchi.li@di.ku.dk

# Abstract

Elaborating a series of intermediate reasoning steps significantly improves the ability of large language models (LLMs) to solve complex problems, as such steps would evoke LLMs to think sequentially. However, human sarcasm understanding is often considered an intuitive and holistic cognitive process, in which various linguistic, contextual, and emotional cues are integrated to form a comprehensive understanding, in a way that does not necessarily follow a step-by-step fashion. To verify the validity of this argument, we introduce a new prompting framework (called SarcasmCue) containing four sub-methods, $v i z$ . chain of contradiction (CoC), graph of cues (GoC), bagging of cues (BoC) and tensor of cues (ToC), which elicits LLMs to detect human sarcasm by considering sequential and non-sequential prompting methods. Through a comprehensive empirical comparison on four benchmarks, we highlight three key findings: (1) CoC and GoC show superior performance with more advanced models like GPT4 and Claude 3.5, with an improvement of $3 . 5 \%$ . (2) ToC significantly outperforms other methods when smaller LLMs are evaluated, boosting the F1 score by $2 9 . 7 \%$ over the best baseline. (3) Our proposed framework consistently pushes the state-of-the-art (i.e., ToT) by $4 . 2 \%$ , $2 . 0 \%$ , $2 9 . 7 \%$ , and $5 8 . 2 \%$ in F1 scores across four datasets. This demonstrates the effectiveness and stability of the proposed framework.

# Code —

https://github.com/qiuchili/llm sarcasm detection.git

# Introduction

Recent large language models have demonstrated impressive performance across downstream natural language processing (NLP) tasks, in which “System 1” - the fast, unconscious, and intuitive tasks, e.g., sentiment classification, topic analysis, etc., have been argued to be successfully performed (Cui et al. 2024). Instead, increasing efforts have been devoted to the other class of tasks - “System $2 ^ { \circ }$ , which requires slow, deliberative and multi-steps thinking, such as logical, mathematical, and commonsense reasoning tasks (Wei et al. 2022). To improve the ability of LLMs to solve such complex problems, a popular paradigm is to decompose complex problems into a series of intermediate solution steps, and elicit LLMs to think step-by-step, such as chain of thought (CoT) (Wei et al. 2022), tree of thought (ToT) (Yao et al. 2024), graph of thought (GoT) (Besta et al. 2024), etc.

However, due to its inherent ambivalence and figurative nature, sarcasm detection is often considered a holistic and irrational cognitive process that does not conform to stepby-step logical reasoning. The main reasons are two fold: (1) sarcasm expression does not strictly conform to formal logical structures, such as the law of hypothetical syllogism (i.e.,if $\mathcal { A } \Rightarrow B$ and $\textstyle B \Rightarrow { \mathcal { C } }$ , then ${ \mathcal { A } } \Rightarrow { \mathcal { C } }$ ). For example, “Poor Alice has fallen for that stupid Bob; and that stupid Bob is head over heels for Claire; but don’t assume for $a$ second that Alice would like Claire”; (2) sarcasm judgment is often considered a fluid combination of various cues. Each cue holds equal importance, and there is no rigid sequence of steps among them. Hence, the main research question can be summarized as:

RQ: Is human sarcasm detection a step-by-step reasoning process?

To answer this question, we propose a theoretical framework, called SarcasmCue, based on the sequential and nonsequential prompting paradigms. It consists of four prompting methods, i.e., chain of contradiction $( C o C )$ , graph of cues $( G o C )$ , bagging of cues $( B o C )$ and tensor of cues (ToC). Each method has its own focus and advantages. In this work, cue is similar to thought, being a coherent language sequence related to linguistics, context, or emotion that serves as an intermediate indicator for identifying sarcasm, such as rhetorical devices or emotional words. More specifically,

• CoC. It harnesses the quintessential property of sarcasm (the contradiction between surface sentiment and true intention). It aims to: (1) identify the surface sentiment by extracting keywords, etc.; (2) deduce the true intention by scrutinizing rhetorical devices, etc.; and (3) determine the inconsistency between them. It is a typical linear structure. • GoC. Generalizing over CoC, GoC frames the problem of sarcasm detection as a search over a graph and treats various cues as nodes, with the relations across cues represented as edges. Unlike CoC and ToT, it goes beyond following a fixed hierarchy or linear reasoning path. Like CoC, GoC follow a step-by-step reasoning process.

• BoC. BoC is a bagging approach that constructs a pool of diverse cues and randomly samples multiple cue subsets. LLMs are employed to generate multiple predictions based on these subsets, and such predictions are aggregated to produce the final result. It is a set-based structure.

• ToC. ToC treats each type of cues (namely linguistic, contextual, and emotional cues) as an independent, orthogonal view for sarcasm understanding and constructs a multi-view representation through tensor product. It allows language models to leverage higher-order interactions among the cues. ToC can be visualized as a 3D volumetric structure. Like BoC, ToC views sarcasm detection as a non-step-by-step reasoning process.

These four methods evolve from linear to nonlinear structure, and from a single perspective to multiple perspectives. They together form a comprehensive theoretical framework (SarcasmCue). The diverse design of the methods makes our framework adaptive to various sarcasm detection scenarios.

We present empirical evaluations of the proposed prompting approaches across four benchmarks over 4 SOTA LLMs (i.e., GPT-4o, Claude 3.5 Sonnet, Llama 3-8B, Qwen 2-7B), and compare their results against 3 SOTA prompting approaches (i.e., standard IO prompting, CoT and ToT). Three key observations are highlighted: (1) When the base model is more advanced (such as GPT-4 and Claude 3.5 Sonnet), CoC and GoC show superior performance against the state-ofthe-art (SoTA) baseline with an improvement of $3 . 5 \%$ . (2) ToC achieves the best performance when smaller LLMs are evaluated. For example, in Llama 3-8B, ToC’s average F1 score of 65.24 represents a $2 9 . 7 \%$ improvement over the best baseline method, ToT. (3) Our proposed framework consistently pushes SoTA by $4 . 2 \%$ , $2 . 0 \%$ , $2 9 . 7 \%$ and $5 8 . 2 \%$ in F1 scores across four datasets. This demonstrates the effectiveness of the proposed framework. The main contributions are concluded as follows:

• Our work is the first to investigate the stepwise reasoning nature of sarcasm detection by using both sequential and non-sequential prompting methods. • We propose a new prompting framework that consists of four sub-methods, viz. CoC, GoC, BoC and ToC. • Comprehensive experiments over four datasets demonstrate the superiority of the proposed prompting framework.

# Related Work

# Chain-of-Thought Prompting

Inspired by the step-by-step thinking ability of humans, CoT prompting was proposed to “prompt” language models to produce intermediate reasoning steps. Wei et al. (2022) made a formal definition of CoT prompting in LLMs and proved its effectiveness by presenting empirical evaluations on arithmetic reasoning benchmarks. However, its performance hinged on the quality of manually crafted prompts. To fill this gap, Auto-CoT was proposed to automatically construct demonstrations with questions and reasoning chains (Zhang et al. 2022). Furthermore, Yao et al. (2024) introduced a non-chain prompting framework, namely ToT, which made LLMs consider multiple different reasoning paths to decide the next course of action. Beyond CoT and ToT approaches, Besta et al. (2024) modeled the information generated by an LLM as an arbitrary graph (i.e., GoT), where units of information were considered as vertices and the dependencies between these vertices were edges.

Table 1: Comparison of prompting methods.   

<html><body><table><tr><td rowspan="2">Scheme</td><td colspan="3">Seq?</td><td colspan="3">Non-Seq?</td></tr><tr><td>Chain?</td><td>Tree?</td><td>Graph?</td><td>Set?</td><td></td><td>Tensor?</td></tr><tr><td>10</td><td>×</td><td>×</td><td>×</td><td></td><td>×</td><td>×</td></tr><tr><td>CoT</td><td>四</td><td>×</td><td></td><td>×</td><td>×</td><td>×</td></tr><tr><td>ToT</td><td></td><td></td><td></td><td>×</td><td>×</td><td>×</td></tr><tr><td>GoT</td><td>□</td><td>□</td><td></td><td>□</td><td>×</td><td>×</td></tr><tr><td>SarcasmCue</td><td>□</td><td>□</td><td></td><td>□</td><td>□</td><td>□</td></tr></table></body></html>

However, all of them adopt the sequential decoding paradigm of “let LLMs think step by step”. Contrarily, it is argued that sarcasm judgment does not conform to step-bystep logical reasoning, and there is an urgent need to develop non-sequential prompting approaches.

# Sarcasm Detection

Sarcasm detection has evolved from early statistical learning based approaches to traditional neural methods, and further advanced to modern neural methods epitomized by Transformer models. In early stage, statistical learning based approaches mainly employ statistical learning techniques, e.g., SVM, NB, etc., to extract patterns and relationships within the data (Zhang et al. 2023). As deep learning based architectures have shown the superiority, numerous base neural networks, e.g., such as CNN (Jain, Kumar, and Garg 2020), LSTM (Ghosh, Fabbri, and Muresan 2018), GCN (Liang et al. 2022), etc., have been predominantly utilized during the middle stage of sarcasm detection research. Now, sarcasm detection research has stepped into the era of pretrained language models (PLMs). An increasing number of researchers are designing sophisticated PLM architectures to serve as encoders for obtaining effective text representations (Liu et al. 2023).

Different from them, we propose four prompting methods to make the first attempt to explore the potential of prompting LLMs in sarcasm detection.

# The Proposed Framework: SarcasmCue

The proposed SarcasmCue framework is illustrated in Fig. 1. We qualitatively compare SarcasmCue with other prompting approaches in Tab. 1. SarcasmCue is the only one to fully support chain-based, tree-based, graph-based, set-based and multidimensional array-based reasoning. It is also the only one that simultaneously supports both sequential and nonsequential prompting methods.

# Task Definition

Given the data set $\begin{array} { r l r } { \mathcal { D } } & { { } = } & { \{ ( \mathcal { X } , \mathcal { Y } ) \} } \end{array}$ , where $\begin{array} { r l } { \mathcal { X } } & { { } = } \end{array}$ $\{ x _ { 1 } , x _ { 2 } , \ldots , x _ { n } \}$ denotes the input text sequence and $\mathcal { V } =$ $\{ y _ { 1 } , y _ { 2 } , \ldots , y _ { n } \}$ denotes the output label sequence. We use ${ \mathcal { L } } _ { \theta }$ to represent a large language model with parameter $\theta$ . Our task is to leverage a collection of cues ${ \mathcal { C } } =$ $\{ c _ { 1 } , c _ { 2 } , . . . , c _ { k } \}$ to brige the input $\mathcal { X }$ and the output $y$ , where each cue $c _ { i }$ is a coherent language sequence that serves as an intermediate indicator toward identifying sarcasm.

![](images/67562ca3c2b16dff71b0022ec5f5ff6ee060d2ac9265ed6b24f3371223503cf9.jpg)  
Figure 1: An illustration of our SarcasmCue framework that consists of four prompting sub-methods.

# Chain of Contradiction

We capture the inherent paradoxical nature of sarcasm, which is the incongruity between the surface sentiment and the true intention, and propose chain of contradiction, a CoT-style paradigm that allows LLMs to decompose the problem of sarcasm detection into intermediate steps and solve each before making decision (Fig. 1 (a)). Each cue $c _ { k } \sim \mathcal { L } _ { \theta } ^ { C o C } \left( c _ { k } | \mathcal { X } , c _ { 1 } , c _ { 2 } , . . . , c _ { k - 1 } \right)$ is sampled sequentially, then the output $\mathcal { V } \sim \mathcal { L } _ { \theta } ^ { C o C } \left( \mathcal { V } | \mathcal { X } , c _ { 1 } , . . . , c _ { k } \right)$ . A specific instantiation of $\mathrm { C o C }$ involves three steps:

1. We first ask LLM to detect the surface sentiment via the following prompt $p _ { 1 }$ :

Given the input sentence $[ \mathcal { X } ]$ , what is the SURFACE sentiment, as indicated by clues such as keywords, sentimental phrases, emojis?

$c _ { 1 }$ is the output sequence, which can be formulated as $c _ { 1 } \sim$ $\mathcal { L } _ { \theta } ^ { C o C } \left( c _ { 1 } | \mathcal { \bar { X , } } p _ { 1 } \right)$ .

2. We thus ask LLM to carefully discover the true intention via the following prompt $p _ { 2 }$ :

Deduce what the sentence really means, namely the TRUE intention, by carefully checking any rhetorical devices, language style, unusual punctuations, common senses.

$c _ { 2 }$ is the output sequence, which can be formulated as $c _ { 2 } \sim$ LθCoC (c2|X , c1, p2).

3. Let LLM examine the consistency between surface sentiment and true intention and make the final prediction:

Based on Step 1 and Step 2, evaluate whether the surface sentiment aligns with the true intention. If they do not match, the sentence is probably ‘Sarcastic’. Otherwise, the sentence is ‘Not Sarcastic’. Return the label only.

Compared to CoT which prompts LLM to reason step-bystep in an open way, our CoC strategy provides specifically designed instructions for each step. Still, it presumes that the cues are linearly correlated, and detects human sarcasm through step-by-step reasoning.

# Graph of Cues

The linear structure of CoC restricts it to a single path of reasoning. To fill this gap, we introduce graph of cues, a graph based paradigm that allows LLMs to flexibly choose and weigh multiple cues, unconstrained by the need for unique predecessor nodes (Fig. 1 (b)). GoC frames the problem of sarcasm detection as a search over a graph, and is formulated as a tuple $( { \mathcal { M } } , { \mathcal { G } } , { \mathcal { E } } )$ , where $\mathcal { M }$ is the cue maker used to define what are the common cues, $\mathcal { G }$ is a graph of “sarcasm detection process”, $\mathcal { E }$ is cue evaluator used to determine which cues to keep selecting.

1. Cue maker. Human sarcasm judgment often relies on the combination and analysis of one or more cues to achieve an accurate understanding. Such cues can be broadly categorized into three types: linguistic cues, contextual cues and emotional cues. Linguistic cues refer to the linguistic features inherent in the text, including keywords, rhetorical devices, punctuation and language style. Contextual cues refer to the environment and background of the text, including topic, cultural background, common knowledge. Emotional cues denote the emotions implied in the text, including emotional words, special symbols (such as emojis) and emotional contrasts. Hence, GoC can obtain $4 + 3 + 3 = 1 0$ cues.

2. Graph construction. In $\mathcal { G } = ( V , E )$ , 10 cues are regarded as vertices, constituting the vertex set $V$ , the supplement relations across cues are regarded as edges. Given the cue $c _ { k }$ , the cue evaluator $\mathcal { E }$ considers cue $c _ { j }$ to provide the most complementary information to $c _ { k }$ , which would combine with $c _ { k }$ to facilitate a deep understanding of sarcasm.

3. Cue evaluator. We associate $\mathcal { G }$ with LLM detecting sarcasm process. To advance this process, the cue evaluator $\mathcal { E }$ assesses the current progress by asking the LLM whether the cumulative cues obtained thus far are sufficient to yield an accurate judgment. The search goes to an end if a positive answer is returned; otherwise, the detection process proceeds by instructing the LLM to determine which additional cues to select and in what order. In this work, an LLM will act as the cue evaluator, similar to ToT.

We employ a voting strategy to determine the most valuable cue for selection, by deliberately comparing multiple potential cue candidates in a voting prompt, such as:

Given an input text $\chi$ , the target is to accurately detect sarcasm. Now, we have collected the keyword information as the first step: $\{ k e y w o r d s \}$ , judge if this provides over $9 5 \%$ confidence for accurate detection. If so, output the result. Otherwise, from the remaining cues rhetorical devices, punctuation, $\left. \dots \right\}$ , vote the most valuable one to improve accuracy and confidence for the next step.

This step can be formulated as $\begin{array} { r l } { \mathcal { E } \left( \mathcal { L } _ { \theta } ^ { G o C } , c _ { j + 1 } \right) } & { { } \sim } \end{array}$ $V o t e \left\{ \mathcal { L } _ { \theta } ^ { G o C } \left( c _ { j + 1 } | \mathcal { X } , c _ { 1 , 2 , . . . , j } \right) \right\} _ { c _ { j + 1 } \in \left\{ c _ { j + 1 } , . . . , c _ { k } \right\} } .$ . Until the selected in a greedy fashion. Although GoC enables the exploration of many possible paths across the cue graph, its nature remains grounded in a step-by-step reasoning paradigm.

# Bagging of Cues

We relax the assumption that the cues are interrelated in detecting sarcasm. We introduce bagging of cues, a ensemble learning based paradigm that allows LLMs to independently consider varied combinations of cues without assuming a fixed order or dependency among them (Fig. 1 (c)).

BoC constructs a pool of the pre-defined 10 cues $\mathcal { C }$ . From this pool, $\tau$ subsets are obtained through $\tau$ random samplings, where each subset $\boldsymbol { \mathcal { S } } _ { t }$ consists of $q$ (i.e., $1 \leq q \leq 1 0 ,$ ) cues. BoC thus leverages LLMs to generate $\tau$ independent sarcasm predictions $\hat { y } _ { t }$ based on the cues of each subset. Finally, such predictions are aggregated using a majority voting mechanism to produce the final result. This approach embraces randomness in cue selection, enhancing the LLM’s ability to explore numerous potential paths. BoC consists of three key steps:

1. Cue subsets construction. A total of $\tau$ cue subsets $S _ { t \in [ 1 , 2 , . . . , T ] } = \{ c _ { t 1 } , c _ { t 2 } , . . . , c _ { t q } \}$ are created by randomly sampling without replacement from the complete pool of cues $\mathcal { C }$ . Each sampling is independent.

2. LLM prediction. For each subset $\boldsymbol { \mathcal { S } } _ { t }$ , a LLM $\mathcal { L } _ { \boldsymbol { \theta } } ^ { B o C }$ is used to independently make sarcasm prediction through the comprehensive analysis of the cues in the subset and the input text. This can be conceptually encapsulated as $\hat { y } _ { t } \sim$ $\mathcal { L } _ { \theta } ^ { \dot { B } o C } \left( \hat { y } _ { t } | S _ { t } , \mathcal { X } \right)$ .

3. Prediction aggregation. The predictions $\big \{ \hat { y } _ { 1 } , \hat { y } _ { 2 } , . . . , \hat { y } _ { T } \big \}$ are combined using majority voting to yield the final prediction $Y$ .

BoC does not follow the step-by-step reasoning paradigm for sarcasm detection.

# Tensor of Cues

$\mathbf { C o C }$ and GoC methods mainly handle low-order interactions between cues, while BoC assumes cues are independent. To capture high-order interactions among cues, we introduce tensor of cues, a stereo paradigm that allows LLMs to amalgamate three types of cues (viz. linguistic, contextual and emotional cues) into a high-dimensional representation. (Fig. 1 (d)).

ToC treats each type of cues as an independent, orthogonal view for sarcasm understanding, and constructs a multiview representation through the tensor product of such three types of cues. We first ask the LLM to extract linguistic, contextual, and emotional cues respectively via a simple prompt. For example:

Extract the linguistic cues from the input sentence for sarcasm detection, such as keywords, rhetorical devices, punctuation and language style.

We take the outputs of the LLM’s final hidden layer as the embeddings of the linguistic, contextual and emotional cues, and apply a tensor fusion mechanism to fuse the cues as additional inputs to the sarcasm detection prompt. Inspired by the success of tensor fusion network (TFN) for multimodal sentiment analysis (Zadeh et al. 2017), we apply token-wise tensor fusion to aggregate the cues. In particular, the embeddings are projected on a low-dimensional space via the fully-connected layers, i.e., $\vec { L i n } = \left( e _ { 1 } ^ { l } , e _ { 2 } ^ { l } , . . . , e _ { L } ^ { l } \right) ^ { T }$ , $\vec { C o n } = ( e _ { 1 } ^ { c } , e _ { 2 } ^ { c } , . . . , e _ { L } ^ { c } ) ^ { T }$ , $\vec { E } \vec { m } o = ( e _ { 1 } ^ { e } , e _ { 2 } ^ { e } , . . . , e _ { L } ^ { e } ) ^ { T }$ . Then, a tensor product is computed to combine the cues into a highdimensional representation $\mathcal { Z } = ( e _ { 1 } , e _ { 2 } , . . . , e _ { L } ) ^ { T }$ , where

$$
e _ { i } = \left[ \begin{array} { l }  e _ { i } ^ { l } \right] \otimes \left[ \begin{array} { l } { e _ { i } ^ { c } } \\ { 1 } \end{array} \right] \otimes \left[ \begin{array} { l } { e _ { i } ^ { e } } \\ { 1 } \end{array} \right] , \forall i \in [ 1 , 2 , . . . , L ] . \end{array}
$$

The additional value of 1 facilitates an explicit rendering of single-cue features and bi-cue interactions, leading to a comprehensive fusion of different cues encapsulated in each fused token $e _ { i } \in \mathcal { R } ^ { ( d _ { l } + 1 ) \times ( d _ { c } + 1 ) \times ( d _ { e } + 1 ) }$ . The values of $d _ { l }$ , $d _ { c }$ and $d _ { e }$ are delicately chosen such that the dimensionality of fused token is precisely $d ^ { 1 }$ . That enables an integration of the aggregated cues to the main prompt via:

Consider the information provided in the current cue above. Classify whether the input text is sarcastic or not. If you think the Input text is sarcastic, answer: yes. If you think the Input text is not sarcastic, answer: no.

The embedded prompt above is prepended with the aggregated cue sequence $\mathcal { Z }$ before fed to the LLM. As it is expected to output a single token of “yes” or “no” by design, we take the logit of the first generated token and decode the label accordingly as the output of ToC.

ToC facilitates deep interactions among these cues. Notably, as ToC manipulates cues on the vector level via neural structures, it requires access to the LLM structure and calls for supervised training on a collection of labeled samples. During training, the weights of the LLM are frozen, and the linear weights in $f _ { l i n } , f _ { c o n } , f _ { e m o }$ are updated as an adaptation of LLM to the task context.

# Experiments

# Experiment Setups

Datasets. Four benchmarking datasets are selected as the experimental beds, viz. IAC-V1 (Lukin and Walker 2013), IAC-V2 (Oraby et al. 2016), SemEval 2018 Task 3 (Van Hee, Lefever, and Hoste 2018) and MUStARD (Castro et al. 2019).

Baselines. A wide range of SOTA baselines are included for comparison. They are:

<html><body><table><tr><td rowspan="2">Paradigm</td><td rowspan="2">Method</td><td colspan="2">IAC-V1</td><td colspan="2">IAC-V2</td><td colspan="2">SemEval 2018</td><td colspan="2">MUStARD</td><td rowspan="2">Avg. of F1</td></tr><tr><td>Acc.</td><td>Ma-F1</td><td>Acc.</td><td>Ma-F1</td><td>Acc.</td><td>Ma-F1</td><td>Acc.</td><td>Ma-F1</td></tr><tr><td rowspan="6">GPT-40</td><td>I0</td><td>70.63</td><td>70.05</td><td>73.03</td><td>71.99</td><td>64.03</td><td>63.17</td><td>67.24</td><td>65.79</td><td>67.75</td></tr><tr><td>CoT</td><td>61.56</td><td>58.49</td><td>58.83</td><td>56.42</td><td>58.92</td><td>51.99</td><td>58.11</td><td>55.76</td><td>55.67</td></tr><tr><td>ToT</td><td>71.56</td><td>71.17</td><td>70.63</td><td>69.07</td><td>63.90</td><td>63.02</td><td>69.00</td><td>68.27</td><td>67.88</td></tr><tr><td>CoC(Ours)</td><td>72.19</td><td>71.52</td><td>73.36</td><td>72.31</td><td>70.79</td><td>70.60</td><td>69.42</td><td>68.48</td><td>70.73*</td></tr><tr><td>GoC (Ours)</td><td>65.00</td><td>62.91</td><td>64.97</td><td>61.30</td><td>74.03*</td><td>74.02</td><td>70.69*</td><td>69.91*</td><td>67.04</td></tr><tr><td>BoC (Ours)</td><td>68.75</td><td>67.36</td><td>71.35</td><td>69.39</td><td>62.12</td><td>61.85</td><td>69.42</td><td>68.45</td><td>66.76</td></tr><tr><td rowspan="6">Claude3.5Sonnet</td><td>10</td><td>66.56</td><td>66.54</td><td>76.78</td><td>76.62</td><td>75.13</td><td>75.11</td><td>74.78</td><td>74.78</td><td>73.26</td></tr><tr><td>CoT</td><td>71.25</td><td>71.14</td><td>74.66</td><td>74.10</td><td>71.56</td><td>71.47</td><td>73.62</td><td>73.53</td><td>72.56</td></tr><tr><td>ToT</td><td>63.44</td><td>62.48</td><td>71.88</td><td>71.74</td><td>68.62</td><td>68.61</td><td>58.84</td><td>54.46</td><td>64.32</td></tr><tr><td>CoC(Ours)</td><td>69.69</td><td>69.40</td><td>73.22</td><td>73.17</td><td>82.27*</td><td>82.23*</td><td>74.20</td><td>74.16</td><td>74.74*</td></tr><tr><td>GoC (Ours)</td><td>70.94</td><td>70.93</td><td>74.67</td><td>74.18</td><td>76.91</td><td>76.91</td><td>70.00</td><td>69.85</td><td>72.97</td></tr><tr><td>BoC (Ours)</td><td>66.88</td><td>66.40</td><td>73.61</td><td>72.82</td><td>70.28</td><td>70.07</td><td>72.61</td><td>71.93</td><td>70.31</td></tr><tr><td rowspan="7">Llama 3-8B</td><td>10</td><td>55.94</td><td>46.40</td><td>54.70</td><td>43.74</td><td>49.36</td><td>44.46</td><td>54.64</td><td>44.99</td><td>44.90</td></tr><tr><td>CoT</td><td>56.25</td><td>47.28</td><td>54.22</td><td>42.96</td><td>49.36</td><td>44.55</td><td>54.20</td><td>44.86</td><td>44.91</td></tr><tr><td>ToT</td><td>52.50</td><td>48.98</td><td>55.95</td><td>53.05</td><td>50.64</td><td>48.63</td><td>54.35</td><td>50.56</td><td>50.31</td></tr><tr><td>CoC(Ours)</td><td>56.25</td><td>46.95</td><td>54.03</td><td>42.60</td><td>49.23</td><td>44.36</td><td>54.93</td><td>45.66</td><td>44.89</td></tr><tr><td>GoC (Ours)</td><td>57.10</td><td>54.96</td><td>54.22</td><td>53.30</td><td>57.33</td><td>57.24</td><td>52.77</td><td>52.67</td><td>54.54</td></tr><tr><td>BoC (Ours)</td><td>62.50</td><td>59.28</td><td>62.57</td><td>58.11</td><td>65.94</td><td>65.50</td><td>59.71</td><td>56.70</td><td>59.90</td></tr><tr><td>ToC (Ours)</td><td>62.19</td><td>61.78</td><td>72.95*</td><td>72.94*</td><td>68.88*</td><td>68.21*</td><td>61.26*</td><td>58.03*</td><td>65.24*</td></tr><tr><td rowspan="7">Qwen 2-7B</td><td>10</td><td>56.56</td><td>49.32</td><td>51.82</td><td>38.57</td><td>45.15</td><td>38.83</td><td>54.78</td><td>46.17</td><td>43.22</td></tr><tr><td>CoT</td><td>54.69</td><td>46.53</td><td>52.88</td><td>40.12</td><td>43.24</td><td>35.79</td><td>54.93</td><td>45.81</td><td>42.06</td></tr><tr><td>ToT</td><td>53.44</td><td>43.71</td><td>50.29</td><td>39.62</td><td>44.26</td><td>38.12</td><td>52.90</td><td>44.60</td><td>41.51</td></tr><tr><td>CoC(Ours)</td><td>55.00</td><td>45.77</td><td>51.92</td><td>38.90</td><td>43.75</td><td>36.37</td><td>53.77</td><td>44.26</td><td>41.33</td></tr><tr><td>GoC (Ours)</td><td>55.00</td><td>47.35</td><td>53.45</td><td>42.25</td><td>45.03</td><td>38.17</td><td>54.49</td><td>47.49</td><td>43.82</td></tr><tr><td>BoC (Ours)</td><td>52.50</td><td>43.78</td><td>52.40</td><td>40.24</td><td>49.87</td><td>45.63</td><td>54.06</td><td>46.11</td><td>43.94</td></tr><tr><td>ToC (Ours)</td><td>71.56*</td><td>71.56*</td><td>72.33</td><td>71.76*</td><td>68.88</td><td>68.77</td><td>65.94*</td><td>61.46*</td><td>68.39*</td></tr></table></body></html>

Table 2: Performance on four datasets. For LLMs, all strategies are based on a zero-shot setting. Bold $^ +$ underline and underline indicate the best and second-best results for each dataset. $\clubsuit$ represents significance improvement over the best baseline via unpaired t-test $( \mathtt { p } < 0 . 0 5 )$ .

• Prompt tuning. (1) IO, (2) CoT (Wei et al. 2022) and (3) ToT (Yao et al. 2024) are three SOTA prompting approaches by leveraging advanced prompt approaches to enhance LLM’s performance.

• LLMs. We involve four general LLMs in the experiment, including (4) GPT-4o, (5) Claude 3.5 Sonnet, (6) Llama 3-8B and (7) Qwen 2-7B (Bai et al. 2023). The first two are non-open-source LLMs while the last two are open-source LLMs. All four LLMs are representative of the strongest capabilities of their kinds.

Implementation. We have implemented the prompting methods for GPT-4o, Claude 3.5 Sonnet, Llama 3-8B and Qwen2-7B. The GPT-4o and Claude 3.5 Sonnet methods are implemented with the respective official Python API library: openAI2 and anthropic3, while the LLaMA and Qwen methods are implemented based on the Hugging Face Transformers library4. For ToC, during training, the original LLM (LlaMA 3-8B and Qwen 2-7B) weights are frozen, while the projection layers are trainable ( $\operatorname { l r } = 0 . 0 0 0 1$ , epochs $= 2 0$ ).

# Main Results

We report both Accuracy and Macro-F1 scores for SarcasmCue and baselines in Table 2.

(1) SarcasmCue consistently outperforms SoTA prompting baselines. The proposed prompting strategies in the SarcasmCue framework achieve an overall superior performance compared to the baselines and consistently push the SoTA by $4 . 2 \%$ , $2 . 0 \%$ , $2 9 . 7 \%$ and $5 8 . 2 \%$ on F1 scores across four datasets. In particular, by explicitly designing the reasoning steps for sarcasm detection, CoC beats CoT by a tremendous margin on GPT-4o and Claude 3.5 Sonnet, whilst performing in par with CoT on Llama 3-8B and Qwen 2-7B. By pre-defining the set of cues in three main categories, GoC and BoC effectively guide LLMs to reason along correct paths, leading to more accurate judgments of sarcasm compared to the freestyle thinking in ToT. For example, the best proposed method, CoC (74.74), brings a $2 . 0 \%$ improvement over the best baseline method, IO (73.26). ToC achieves an effective tensor fusion of multi-aspect cues for sarcasm detection, significantly outperforming other baselines. For instance, it exhibits a $2 9 . 7 \%$ improvement over the best baseline method, ToT (50.31).

<html><body><table><tr><td>Method IAC-V1 IAC-V2 SemEval MUStARD</td></tr><tr><td>w/o Lin 68.41</td><td>75.62</td><td>77.42 69.66</td><td>Avg. of F1 72.78</td></tr><tr><td>w/o Emo</td><td>69.65 74.04</td><td>78.70</td><td>70.57</td><td>73.24</td></tr><tr><td>w/o Con</td><td>70.53</td><td>74.91</td><td>76.39 70.11</td><td>72.99</td></tr><tr><td>GoC w/o Lin</td><td>70.93 45.89</td><td>74.18 42.49</td><td>76.91 47.47</td><td>69.85 72.97 65.33 50.30</td></tr><tr><td>W/o Emo w/o Con BoC</td><td>58.00 61.71 66.40</td><td>56.99 63.70 72.82</td><td>56.81 68.84 69.53 74.80 70.07 71.93</td><td>60.16 67.44 70.31</td></tr><tr><td>w/o Lin w/o Emo w/o Con</td><td>45.79 48.60 52.51</td><td>51.90 49.40 53.69</td><td>56.01 46.84 52.38 45.12 52.14 48.28 52.67</td><td>50.14 48.88 51.66</td></tr><tr><td>GoC w/o Lin</td><td>54.96 52.71</td><td>53.30 57.51</td><td>57.24 57.53 53.06</td><td>54.54 55.20</td></tr><tr><td>w/o Emo</td><td>57.33</td><td>59.40 62.01</td><td>53.06</td><td>57.95</td></tr><tr><td>w/o Con</td><td>56.88</td><td>60.36</td><td>59.04 52.30</td><td>57.15</td></tr><tr><td>BoC</td><td>59.28</td><td>58.11</td><td>65.50</td><td>59.90</td></tr><tr><td></td><td></td><td></td><td></td><td>56.70</td></tr><tr><td>w/o Lin</td><td>53.31</td><td>67.05</td><td>59.20</td><td>48.05 56.90</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o Emo</td><td>57.42</td><td>67.08</td><td>64.01 52.89</td><td>60.35</td></tr><tr><td></td><td>55.26</td><td>71.78</td><td></td><td></td></tr><tr><td>w/o Con</td><td></td><td></td><td>63.93</td><td>52.48 60.86</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>65.24</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ToC</td><td></td><td>72.94</td><td>68.21</td><td>58.03</td></tr><tr><td></td><td>61.78</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table></body></html>

Table 3: Ablation study of BoC, GoC and ToC. All strategies are run on a zero-shot setting. The top part shows results for Claude 3.5 Sonnet, and the bottom part for Llama 3-8. The best results for each dataset are formatted in Bold $^ +$ underline.

(2) Sarcasm detection does not necessarily follow a step-by-step reasoning process. The comparison between sequential (CoT, CoC, GoC, ToT) and non-sequential (BoC, ToC) prompting strategies fails to provide clear empirical evidences on whether sarcasm detection follows a step-bystep reasoning process. Nevertheless, the results on Llama 3-8B are more indicative to GPT-4o and Claude 3.5 Sonnet, since the latter models have strong capabilities on their own (IO) and do not significantly benefit from any prompting strategies. For Llama 3-8B and Qwen 2-7B, non-sequential methods, particularly ToC, show superior performance. In Llama 3-8B, ToC achieves an average F1 score of $6 5 . 2 4 \%$ , which is $8 . 9 \%$ higher than the best sequential method (GoC at $5 4 . 5 4 \% \rangle$ ). The difference is even more pronounced on Qwen 2-7B. Furthermore, The McNemar’s test between CoC and BoC on the Llama 3-8B’s outputs exhibits $\chi ^ { 2 } =$ 117.00, $p \ < \ 0 . 0 5$ , suggesting that the BoC works significantly better than CoC. These result supports our hypothesize that sarcasm has a non-sequential nature.

# Ablation Study

Table 3 presents the result of ablation study. w/o Lin, w/o Emo, w/o Con refer to the method where linguistic, emotional and contextual cues are ablated, respectively. To avoid proactive extraction of ablated cues by an LLM, we explicitly “prompt away” the cues in the inputs. An example prompt could be “You can only use the emotional cues and contextual cues, and do not use any linguistic information here” for the w/o Lin case.

![](images/c8d84873f92f918992fdce76c836fd43e550a808b6b6ec9e4ac8c63d77f005b0.jpg)  
Figure 2: The average Macro-F1 across K-shots for the GPT-4o and Claude 3.5 Sonnet models.

The experiment results highlight the following conclusions: (a) the removal of any single type of cue leads to a noticeable drop in performance across all datasets, demonstrating the importance of each type of cue in sarcasm detection; (b) linguistic cues appear to have the most significant impact, as removing them leads to a noticeable decrease in performance across most settings; (c) the absence of contextual cues also affects the performance, but to a lesser extent compared to linguistic cues.

# Zero-Shot V/S Few-Shot Prompting

Since the above experiments are mainly based on a zeroshot setting, we are curious of whether the conclusions also apply in a few-shot scenario. Therefore, we perform fewshot experiments to evaluate whether the proposed SarcasmCue framework can perform better when a limited number of contextual examples are available. We plot the main results in Fig. 2, we randomly sample $k = \{ 0 , 1 , 5 , 1 0 \}$ examples from the training set.

As shown in the plot, the number of demonstrations has a significant impact on the results. For example, CoC appears sensitive to the initial introduction of demonstration examples with a slight descent in performance when only 1 example is provided. However, as the number of shots increases to 5 and 10, the performance progressively improves. This trend underscores the effectiveness of $\mathrm { C o C }$ in adapting and refining its approach with more examples. In contrast, BoC demonstrates a consistent improvement in performance as the number of shots increases.

Overall, these results demonstrate the robustness and adaptability of the SarcasmCue framework in zero-shot and few-shot scenarios. The framework can effectively utilize limited contextual examples to further improve sarcasm detection, making it suitable for applications where large annotated datasets are not readily available.

# Influences of LLM Scales

In an attempt to study the influence of different LLM scales, we evaluate the performance of sarcasm detection of Qwen and Llama of varying sizes, see Fig. 3.

![](images/ad0ee431ef894d0bb180be1ad5a522dc6b8892e7bc20204c3610d88936d13c78.jpg)  
Figure 3: The influence of model scale. The figures in the top and bottom correspond to Qwen and Llama models, respectively.

The key take-aways are two-fold. First, the efficacy of our prompting methods is amplified with increasing model scale. This aligns closely with the key findings of the CoT method (Wei et al. 2022). This occurs because when an LLM is sufficiently large, its capabilities for multi-hop reasoning and understanding language are significantly enhanced. Second, ToC exhibits high sensitivity to model scale, performing significantly better in larger models, making it particularly suitable for larger-scale applications. CoC and GoC demonstrate moderate sensitivity, indicating a balance between performance improvement and scalability. BoC offers robust performance even in smaller models, suggesting its utility in resource-constrained scenarios. Overall, our proposed framework has a high adaptability across various model scales by offering suitable methods.

# Error Analysis

Fig. 4 shows the error rates of failure cases in terms of false negative (FN) and false positive (FP) for all four prompting methods in SarcasmCue. CoC, GoC and BoC exhibit higher false positive rates, indicating an over-detection of sarcasm that could lead to the frequent misclassification of normal statements as sarcastic. In contrast, ToC exhibits the lowest overall error rate and the FP and FN rates are indeed much closer to each other, indicating a balanced performance in detecting both sarcastic and non-sarcastic texts. We further analyzed the common patterns among the over-detected examples, and found out that our methods are overly sensitive to certain cues commonly associated with sarcasm, such as negative information, exaggerated language, rhetorical devices, or harmful words. These insights highlight potential directions for future improvements in sarcasm detection methodologies. The higher false positive rates suggest a need for refining these methods to reduce over-sensitivity and improve discrimination between sarcastic and non-sarcastic texts.

# Extension to New Task

To evaluation the generalization capability of SarcasmCue, we apply it to another complex affection understanding task, humor detection. We compare our proposed SarcasmCue (where the backbone is GPT-4o) with two supervised PLMs (MFN (Hasan et al. 2021) and SVM $+$ BERT (Zhang et al. 2024)) on two benchmarking datasets, CMMA (Zhang et al. 2024) and UR-FUNNY-V2 (Hasan et al. 2019).

![](images/fd606ac8c3b6d0f9fa6ba4629ca71b1c756c740be28d2626add07acfe9148ec8.jpg)  
Figure 4: The average error rate of the four prompting methods.

Table 4: Performance on two humor detection datasets.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="3">CMMA UR-FUNNY-V2</td><td rowspan="2">Avg. of F1</td></tr><tr><td>Acc.</td><td>Ma-F1 Acc.</td><td>Ma-F1</td></tr><tr><td>MFN</td><td>、</td><td>64.44</td><td>64.12</td><td></td></tr><tr><td>SVM+BERT</td><td>55.23 54.08</td><td>69.62</td><td>69.27</td><td>61.68</td></tr><tr><td>CoC</td><td>78.14</td><td>58.60 64.08</td><td>60.13</td><td>65.24</td></tr><tr><td>GoC</td><td>79.60 57.42</td><td>64.89</td><td>61.65</td><td>65.89</td></tr><tr><td>BoC</td><td>75.81 58.58</td><td>68.71</td><td>66.83</td><td>67.48</td></tr></table></body></html>

As shown in Table 4, our methods (BoC and CoC) surpass the baseline on CMMA, whilst performing in par to the strongest baselines on the UR-FUNNY-V2 dataset. These results highlight the strong generalizability and versatility of our framework, confirming its potential utility across a wide range of affection understanding tasks.

# Conclusions

This work aims to study the stepwise reasoning nature of sarcasm detection, and introduces a prompting framework (called SarcasmCue) containing four sub-methods, viz. CoC, GoC, BoC and ToC. It elicits LLMs to detect human sarcasm by considering sequential and non-sequential prompting methods. Our comprehensive evaluations across multiple benchmarks and SoTA LLMs demonstrate that SarcasmCue outperforms traditional methods and pushes the state-of-the-art by $4 . 2 \%$ , $2 . 0 \%$ , $2 9 . 7 \%$ and $5 8 . 2 \%$ F1 scores across four datasets. Additionally, the performance of SarcasmCue on humor detection further validate its robustness and versatility.

Limitations. First, the ToC method demands extra computational resources due to its complex multi-view tensor structure. Second, aside from BoC, the effectiveness of the other three approaches in the SarcasmCue framework is largely dependent on the scale of LLM. Finally, this framework primarily focuses on text data while sarcasm detection often requires multi-modal analysis.

# Ethical Statement

We are committed to adhere to strict ethical standards, using open and fair datasets while recognizing the societal impact of sarcasm detection and promoting its responsible application.