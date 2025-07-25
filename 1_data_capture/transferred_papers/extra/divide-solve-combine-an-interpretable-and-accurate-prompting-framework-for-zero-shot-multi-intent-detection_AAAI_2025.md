# Divide-Solve-Combine: An Interpretable and Accurate Prompting Framework for Zero-shot Multi-Intent Detection

Libo $\mathbf { Q i n } ^ { 1 , 2 * }$ , Qiguang Chen3\*, Jingxuan Zhou1, Jin Wang4, Hao Fei5, Wanxiang Che3, Min Li1

1School of Computer Science and Engineering, Central South University, China 2Key Laboratory of Data Intelligence and Advanced Computing in Provincial Universities, Soochow University, China 3Research Center for Social Computing and Information Retrieval, Harbin Institute of Technology, China 4Yunnan University, China 5National University of Singapore, Singapore lbqin@csu.edu.cn, qgchen@ir.hit.edu.cn

# Abstract

Zero-shot multi-intent detection is capable of capturing multiple intents within a single utterance without any training data, which gains increasing attention. Building on the success of large language models (LLM), dominant approaches in the literature explore prompting techniques to enable zeroshot multi-intent detection. While significant advancements have been witnessed, the existing prompting approaches still face two major issues: lacking explicit reasoning and lacking interpretability. Therefore, in this paper, we introduce a Divide-Solve-Combine Prompting (DSCP) to address the above issues. Specifically, DSCP explicitly decomposes multi-intent detection into three components including (1) single-intent division prompting is utilized to decompose an input query into distinct sub-sentences, each containing a single intent; (2) intent-by-intent solution prompting is applied to solve each sub-sentence recurrently; and (3) multi-intent combination prompting is employed for combining each subsentence result to obtain the final multi-intent result. By decomposition, DSCP allows the model to track the explicit reasoning process and improve the interpretability. In addition, we propose an interactive divide-solve-combine prompting (Inter-DSCP) to naturally capture the interaction capabilities of large language models. Experimental results on two standard multi-intent benchmarks (i.e., MIXATIS and MIXSNIPS) reveal that both DSCP and Inter-DSCP obtain substantial improvements over baselines, achieving superior performance and higher interpretability.

Watch sports movie and play rock music Divide-Solve-Combine Prompting Single-Intent Part Part 2 Division Watch sports movie play rock music Intent-by-Intent Intent 1 Intent 2 Solution WatchMovie PlayMusic Multi-Intent Result 1 Result 2 Combination WatchMovie PlayMusic (a) Divide-Solve-Combine Prompting (DSCP) Watch sports movie and play rock music Single-Intent Division Prompting Single-Intent Part Part 2 陶 Division Watch sports movie play rock music A Intent-by-Intent Solution Prompting Intent-by-Intent Intent 1 Intent 2 WatchMovie PlayMusic A Solution Multi-Intent Combination Prompting Multi-Intent Result 1 Result 2 Combination WatchMovie PlayMusic A (b) Interactive Divide-Solve-Combine Prompting (Inter-DSCP)

# 1 Introduction

Intent detection plays a pivotal role in task-oriented dialog systems, which can be used to extract the intents of user queries for accurate system response generation (Tur and De Mori 2011; Qin et al. 2021c). In real-world scenarios, users often express multiple intents within a single utterance. Consequently, dominant approaches in the literature shift their focus from single intent detection to multi-intent detection (Qin et al. 2020; Zhu et al. 2023b).

With the revolution of the pre-trained models, remarkable success has been witnessed in the multi-intent detection. Specifically, recent multi-intent research can be broadly grouped into two main categories. The first category leverages slot filling for enhancing multi-intent detection through their mutual interaction (Gangadharaiah and Narayanaswamy 2019; Qin et al. 2021b; Song et al. 2022; Xing and Tsang 2022; Cheng, Yang, and Jia 2023). The second category seeks to improve multi-intent detection through intent semantic space optimization to enhance the performance (Wu, Su, and Juang 2021; Hou et al. 2021; Song, Huang, and Wang 2022; Vulic´ et al. 2022; Zhu et al. 2023a).

While remarkable success has been achieved, the current multi-intent detection approaches still follow the traditional training-test paradigm that requires a large amount of training data. This reliance on abundant data presents challenges for real-world applications, where sufficient data may not be readily available. Recently, the Large Language Models (LLMs) have been shown surprising performance in zeroshot settings that does not require model updates, which can greatly alleviate the difficulty of annotated data. Inspired by this, Pan et al. (2023) introduce a vanilla multi-intent detection prompting method (VMID) for zero-shot multi-intent detection, which prompts LLM to generate multi-intent results without any parameter tuning. Despite its simplicity, this method still faces two major drawbacks: (1) Lacking explicit reasoning process: VMID yields direct results without revealing any explicit reasoning process, limiting the performance (Wei et al. 2022); (2) Lacking interpretability: The whole zero-shot prompting prediction process in VMID is a black box for users, making it challenging to establish a clear connection between sentence spans and their corresponding intents (Jiang et al. 2023b).

Motivated by this, in this paper, we introduce a DivideSolve-Combine Prompting (DSCP) strategy to address the above issues. Specifically, as illustrated in Figure 1(a), DSCP consists of three components: (1) single-intent division prompting, (2) intent-by-intent solution prompting, and (3) multi-intent combination prompting. Concretely, given a user query, single-intent division prompting is first used to divide the utterance into spans where each span contains a single intent. Then, intent-by-intent solution prompting is applied to extract the intents from those spans. Furthermore, after obtaining all independent intent results, multi-intent combination prompting is required to combine all intents for the final multi-intent result. By explicitly emulating the multi-intent thought process, DSCP has the following advantages: (1) With the help of single-intent division prompting and intent-by-intent solution prompting, DSCP is able to explicitly solve zero-shot multi-intent SLU step by step, achieving the explicit reasoning process; (2) By decomposing the zero-shot multi-intent SLU into three sub-component solvers, DSCP enables us to analyze which span corresponds to each intent, helping us understand the reasons behind the results and thereby enhancing interpretability. In addition, we further propose an interactive divide-solve-combine prompting (Inter-DSCP) to naturally elicit the interaction capabilities of large language models (see Figure 1 (b)).

We conduct experiments on the two widely-recognized benchmarks, MIXATIS and MIXSNIPS. The experimental results demonstrate that DSCP achieves superior performance. Furthermore, the Inter-DSCP can bring further improvement by successfully capturing the interaction ability in LLM.

Our contributions can be summarized as follows:

• We introduce the Divide-Solve-Combine Prompting (DSCP) framework for zero-shot multi-intent detection, which can elicit the explicit reasoning process of multiintent detection in LLM and improve interpretability. • We further introduce an interactive divide-solve-combine prompting (Inter-DSCP), which is capable of naturally modeling the interaction ability of LLM to enhance performance.

• Experimental results on MIXATIS and MIXSNIPS demonstrate the effectiveness of DSCP and Inter-DSCP by achieving superior performance. Besides, extensive analysis reveals the superiority of our approaches.

To facilitate the reproducibility of our work, our code will be available at https://github.com/LightChen233/DSCP.

# 2 Divide-Solve-Combine Prompting

Unlike the vanilla multi-intent detection prompting, as shown in Figure 2, divide-solve-combine prompting explicitly decomposes the process into three components to solve it step by step, containing (1) Single-Intent Division Prompting (§2.1); (2) Intent-by-Intent Solution Prompting (§2.2) and Multi-intent Combination Prompting (§2.3).

# 2.1 Single-Intent Division Prompting

Single-Intent Division Prompting $( S I D P )$ requires the model to explicitly divide a sentence into multiple spans corresponding to different intents. Formally, $S I D P$ is shown as:

[Task Instruction $\tau ]$ : Assuming you are a professional multi-intent annotator, you need to label ... [Label Constraint $\mathcal { L } ^ { C ^ { \dagger } }$ ]: You need to select the intent of the sentence from the following intent list ... [Single-Intent Division $\mathcal { D } ]$ : Firstly, you need to divide the sentence into multiple parts that contain different intents;

Each part of the prompt is introduced as follows:

(1) Task Instruction $\tau$ describes the task requirements and definitions of multi-intent detection, aiming to clearly specify the task that the model needs to handle.   
(2) Label Constraint $\mathcal { L } ^ { C }$ contains all predefined label set $\mathcal { L }$ from the multi-intent detection task.   
(3) Single-Intent Division $\overline { { \mathcal { D } } }$ is provided to require models to split given input into a series of single-intent spans. In summary, the division process can be expressed as:

$$
\hat { \mathcal { Y } } ^ { D } = \underset { S } { \operatorname { a r g m a x } } \sum p ( s _ { 1 } , \ldots , s _ { n } | \mathcal { T } , \mathcal { L } ^ { C } , \mathcal { D } , \boldsymbol { \chi } ) ,
$$

where $\begin{array} { l l l } { { \mathcal S } } & { { = } } & { { \left\{ s _ { 1 } , s _ { 2 } , . . . , s _ { n } \right\} } } \end{array}$ denotes one of the possible span split divided from input $\chi$ , and $\begin{array} { r l } { \hat { \mathcal { V } } ^ { D } } & { { } = } \end{array}$ {yˆ1D, yˆ2D, . . . denotes the predicted split single-intent span set for next step.

# 2.2 Intent-by-Intent Solution Prompting

After obtaining the split single-intent span, we further introduce a Intent-by-Intent Solution Prompting $( I I S P )$ to detect intent on each span. Specifically, $I I S P$ can be defined as:

[Intent-by-Intent Solution Prompting $\boldsymbol { \mathcal { S } }$ ] : Secondly, you need to consider what intents each part contains;

(a) Divide-Solve-Combine Prompting (DSCP) Input sentence Watch sports movie and play rock music Divide-Solve-Combine Task Instruction (𝓣) : Assuming you are a professional multi-intent ... Label Constraint $( \mathcal { L } ^ { C } )$ ): You need to select the intent of the sentence from ... Part Part 2 Prompting Single-Intent Division $\mathbf { \Pi } ( \mathbf { \mathcal { D } } )$ ) : Firstly, you need to divide the sentence into .. Watch sports movie play rock music A Intent-by-Intent Solution (𝓢) : Secondly, you need to consider what intents each WaItncthenMto1vie PIlantyeMntus2ic Multi-intent Combination(𝓒) : Finally, you need to consider all the intents together; Result Result Regulation $( { \mathcal { R } } )$ : and output them in the form of ‘Result=INTENT1#. WatchMovie#PlayMusic   
(b)  Interactive Divide-Solve-Combine Prompting (Inter-DSCP) Input sentence Watch sports movie and play rock Task Instruction (𝓣) : Assuming you are a professional multi-intent ... TDuivrinde1 Label Constraint (𝓛C): You need to select the intent of the sentence from . Part 1 music Part 2 (𝓓) : Firstly, you need to divide the sentence into... 5 Single-Intent Division Watch sports movie play rock music Turn 2 Intent 1 Intent 2 Solve Intent-by-Intent Solution (𝓢) : Secondly, you need to consider what intents each WatchMovie PlayMusic Turn 3 Multi-intent Combination(𝓒) : Finally, you need to consider all the intents together; Result Combine Result Regulation (𝓡) : and output them in the form of ‘Result=INTENT1#... WatchMovie#PlayMusic

where Intent-by-Intent Solution Prompting $s$ requires models to predict the intents separately, described as:

$$
\hat { \mathcal { V } } ^ { S } = \underset { y ^ { S } } { \mathrm { a r g m a x } } p ( y _ { 1 } ^ { S } , \ldots , y _ { n } ^ { S } | \mathcal { H } ^ { D } , \hat { \mathcal { V } } ^ { D } , S ) ,
$$

where $y ^ { s }$ denotes one of the possible label lists for the corresponding span list, and $y _ { i } ^ { S } \in \mathcal { V } ^ { S }$ are all selected from the label set $\mathcal { L } . \mathcal { H } ^ { D }$ denotes the generation history which contains $\tau , \mathcal { L } ^ { C } , \mathcal { D } , \mathcal { X }$ in the $S I D P$ stage.

# 2.3 Multi-intent Combination Prompting

Furthermore, we provide Multi-Intent Combination Prompting (MICP) to combine all intents, which is defined as:

Multi-intent Combination Prompting $\mathcal { C }$ : Finally, you need to consider all the intents together. Result Regulation $\mathcal { R }$ : and output them in the form of ’Result=INTENT1#INTENT2...

Similarly, the prompt is introduced in detail as follows:

(1) Multi-intent Combination Prompting $\mathcal { C }$ requires the model to combine all intent results to obtain the final multi-intent result.

(2) Result Regulation $\mathcal { R }$ contains all label set $\mathcal { L }$ from the intent detection task.

Formally, the $M I C P$ is determined as:

$$
\hat { \mathcal { I } } = \underset { \mathbb { Z } _ { j } \in \mathbb { Z } \wedge \mathbb { Z } _ { j } \subseteq \mathcal { L } } { \mathrm { a r g m a x } } p ( \mathcal { I } | \mathcal { R } , \mathcal { C } , \hat { \mathcal { V } } ^ { S } , \mathcal { H } ^ { S } ) ,
$$

where $\hat { \boldsymbol { \mathcal { I } } }$ denotes the predicted multi-intent label results, and $\mathcal { H } ^ { S }$ denotes the generation history which contains $\mathcal { H } ^ { D } , \hat { \mathcal { V } } ^ { D }$ , $s$ in $I I S P$ stage.

# 2.4 Interactive Divide-Solve-Combine Prompting

With the emergence of LLM, their strong interactive capabilities have gathered considerable attention (Zheng et al. 2023). Inspired by this, as shown in Figure 2 (b), we introduce Interactive Divide-Solve-Combine Prompting (Inter-DSCP) to leverage the interactive potential of LLMs. Unlike DSCP that directly obtain results in a single dialogue turn, Inter-DSCP leverages three turns of dialogue to stimulate the interactive capabilities of LLMs.

Formally, in Inter-DSCP, the overall process can be formulated as follows:

Single-Intent Division Prompting In the first dialog interactive turn, the initial response can be mathematically represented as follows:

$$
\hat { \mathcal { Y } } _ { \mathrm { I n t e r } } ^ { D } = \underset { S ^ { p } } { \operatorname { a r g m a x } } \sum p ( s _ { 1 } ^ { p } , . . . , s _ { n } ^ { p } | \mathbb { U } \mathrm { s e r } ( S I D P ) ) ,
$$

where $\mathsf { U s e r } ( \cdot )$ signifies that the model receives the prompt in the “user” role.

Intent-by-Intent Solution Prompting In the second dialog interactive turn, the process of Intent-by-Intent Solution Prompting (IISP) is mathematically formulated as:

$$
\hat { \mathcal { Y } } _ { \mathtt { I n t e r } } ^ { S } = \underset { y ^ { S } } { \mathrm { a r g m a x } } p ( y ^ { S } | \tilde { \mathcal { H } } ^ { D } , \cup s \in \mathrm { \ r } ( I I S P ) ) ,
$$

where $\begin{array} { r l } { \mathcal { V } ^ { S } ~ = ~ } & { { } \{ y _ { 1 } ^ { S } , \cdot \cdot \cdot , y _ { n } ^ { S } \} } \end{array}$ . $\begin{array} { r l } { \tilde { \mathcal { H } } ^ { D } \ = \ } & { { } \{ \mathrm { U s e r } ( D ) , \bar { \mathrm { A s s } } } \end{array}$ $( \hat { \mathcal { V } } _ { \mathtt { I n t e r } } ^ { D } ) \}$ represents the generation history, and $\mathtt { A s s } ( \cdot )$ insistant” role.

Multi-intent Combination Prompting In the third dialog interactive turn, the prediction of intent in this framework can be defined as:

$$
\hat { \mathcal { I } } = \operatorname * { a r g m a x } _ { \mathcal { Z } _ { j } \in \mathcal { Z } \wedge \mathcal { T } _ { j } \in \mathcal { L } } p ( \mathcal { T } | \tilde { \mathcal { H } } ^ { S } , \cup \mathcal { s } \in \mathrm { r } ( M I C P ) ) ,
$$

Table 1: Main Results for Mistral-7B, GPT-3.5, PaLM-2, and GPT-4 on MIXSNIPS and MIXSNIPS test sets. Intent Acc. denotes the intent accuracy. All metrics are calculated based on OpenSLU framework (Qin et al. 2023b). The improvements over all baselines are statistically significant with $p < 0 . 0 5$ under t-test   

<html><body><table><tr><td rowspan="2">Model</td><td colspan="3">MIXSNIPS</td><td colspan="3">MIXATIS</td></tr><tr><td>Intent Acc.(%)</td><td>Macro F1(%)</td><td>Micro F1(%)</td><td>Intent Acc.(%)</td><td>Macro F1(%)</td><td>Micro F1(%)</td></tr><tr><td colspan="7">Mistral-7B (Jiang et al.2023a)</td></tr><tr><td>VMID (Pan et al. 2023)</td><td>32.87</td><td>66.35</td><td>67.48</td><td>5.97</td><td>39.46</td><td>37.88</td></tr><tr><td>Zero-CoT (Kojima et al. 2022)</td><td>32.98</td><td>69.61</td><td>69.79</td><td>8.19</td><td>43.59</td><td>42.40</td></tr><tr><td>Least-to-Most (Zhou et al.2022)</td><td>29.00</td><td>61.91</td><td>62.65</td><td>6.67</td><td>32.44</td><td>32.97</td></tr><tr><td>Plan-and-Solve(Wang etal.2023)</td><td>28.99</td><td>61.78</td><td>62.27</td><td>7.93</td><td>34.25</td><td>34.78</td></tr><tr><td>DSCP</td><td>36.65</td><td>69.44</td><td>71.28</td><td>11.21</td><td>42.20</td><td>41.81</td></tr><tr><td>Inter-DSCP</td><td>41.52</td><td>69.62</td><td>70.71</td><td>14.72</td><td>46.96</td><td>46.60</td></tr><tr><td colspan="7">GPT-3.5 (OpenAI 2022)</td></tr><tr><td>VMID (Pan et al.2023)</td><td>63.12</td><td>87.87</td><td>87.80</td><td>21.94</td><td>67.40</td><td>66.01</td></tr><tr><td>Zero-CoT (Kojima et al. 2022)</td><td>63.21</td><td>87.79</td><td>87.38</td><td>26.11</td><td>68.77</td><td>66.11</td></tr><tr><td>Least-to-Most (Zhou et al. 2022)</td><td>66.48</td><td>89.90</td><td>89.47</td><td>25.69</td><td>68.99</td><td>65.45</td></tr><tr><td>Plan-and-Solve (Wang et al. 2023)</td><td>69.12</td><td>90.35</td><td>90.04</td><td>27.08</td><td>65.32</td><td>62.59</td></tr><tr><td>DSCP</td><td>72.90</td><td>91.14</td><td>90.79</td><td>29.177</td><td>73.75</td><td>70.05</td></tr><tr><td>Inter-DSCP</td><td>74.90</td><td>91.50</td><td>91.30</td><td>41.11</td><td>75.04</td><td>75.41</td></tr><tr><td colspan="7">PaLM-2 (Anil et al. 2023)</td></tr><tr><td>VMID (Pan et al.2023)</td><td>79.82</td><td>92.32</td><td>92.96</td><td>26.25</td><td>65.57</td><td>66.15</td></tr><tr><td>Zero-CoT (Kojima et al. 2022)</td><td>79.95</td><td>93.07</td><td>92.88</td><td>26.39</td><td>65.94</td><td>65.74</td></tr><tr><td>Least-to-Most (Zhou et al.2022)</td><td>76.31</td><td>92.08</td><td>91.95</td><td>26.94</td><td>69.73</td><td>66.25</td></tr><tr><td>Plan-and-Solve (Wang et al.2023)</td><td>82.04</td><td>93.98</td><td>93.94</td><td>26.56</td><td>65.95</td><td>63.33</td></tr><tr><td>DSCP</td><td>83.95</td><td>94.18</td><td>94.16</td><td>28.19</td><td>65.99</td><td>67.24</td></tr><tr><td>Inter-DSCP</td><td>86.81</td><td>94.27</td><td>94.43</td><td>31.53</td><td>66.75</td><td>67.39</td></tr><tr><td colspan="7">GPT-4 (OpenAI 2023)</td></tr><tr><td>VMID (Pan et al.2023)</td><td>88.45</td><td>95.46</td><td>95.47</td><td>32.64</td><td>76.74</td><td>74.02</td></tr><tr><td>Zero-CoT (Kojima et al. 2022)</td><td>87.77</td><td>95.46</td><td>95.47</td><td>40.97</td><td>78.62</td><td>77.40</td></tr><tr><td>Least-to-Most (Zhou et al. 2022)</td><td>87.27</td><td>94.60</td><td>94.71</td><td>48.33</td><td>82.26</td><td>80.67</td></tr><tr><td>Plan-and-Solve (Wang et al.2023)</td><td>84.45</td><td>92.81</td><td>92.90</td><td>36.67</td><td>79.44</td><td>76.68</td></tr><tr><td>DSCP</td><td>89.68</td><td>95.69</td><td>95.80</td><td>50.69</td><td>84.57</td><td>81.97</td></tr><tr><td>Inter-DSCP</td><td>92.00</td><td>96.59</td><td>96.66</td><td>52.22</td><td>83.99</td><td>81.86</td></tr></table></body></html>

where $\begin{array} { r l r } { \mathcal { \tilde { H } } ^ { S } } & { { } \ = \ } & { \{ \mathrm { U s e r } ( S I D P ) , \mathbb { A } \mathrm { s } \mathrm { s } ( \hat { \mathcal { Y } } _ { \mathrm { I n t e r } } ^ { D } ) , \mathrm { U s } } \end{array}$ ser $( I I S P )$ , $\mathbb { A } s \mathbf { s } ( \hat { \mathscr { V } } _ { \mathrm { I n t e r } } ^ { S } ) \}$ captures the history of interactive prompts in a dialog format.

# 3 Experiments and Analysis

# 3.1 Implementation Settings

Following Qin et al. (2020); Cheng et al. (2024); Zhu et al. (2024b), we evaluate DSCP and Inter-DSCP on two widely used multi-intent benchmark: MIXATIS and MIXSNIPS (Qin et al. 2020). We follow Pan et al. (2023) to use similar regular expressions to extract multi-intent results to calculate the relevant metrics. The top- $\cdot \mathbf { p }$ parameter in all processes is selected from $\{ 0 . 9 5 , 1 \}$ . The temperatures for Single-Intent Division Prompting, Intent-by-Intent Solution Prompting and Multi-Intent Combination Prompting are selected from [0, 2].

# 3.2 Backbones and Baselines

We evaluate DSCP and Inter-DSCP on some representative LLMs, including: Mistral-7B (Jiang et al. 2023a); PaLM-2 (Anil et al. 2023); GPT-3.5 (OpenAI 2022) and

GPT-4 (OpenAI 2023) backbone. In addition, we adapt the following prompting baselines, including:

• Vanilla Multi-Intent Detection Prompting (VMID) (Pan et al. 2023) directly requires LLMs to output the corresponding multiple intents through vanilla prompting;   
• Zero-shot Chain-of-Thought Prompting $\scriptstyle ( \mathsf { Z e r o - C o T } )$ (Kojima et al. 2022) adds “Let’s think step-by-step!” to stimulate the LLMs’ thinking chain ability;   
• Least-to-Most Prompting (Least-to-Most) (Zhou et al. 2022) generates a series of sub-question and then solves them one-by-one;   
• Plan-and-Solve Prompting (Plan-and- Solve) (Wang et al. 2023) automatically makes a solution plan and then solves them.

# 3.3 Main Results

Following Qin et al. (2023b), we use Intent Accuracy, Micro F1, and Macro F1 for evaluating multi-intent detection performance. The main results are illustrated in Table 1. From the results, we have the following observations:

GPT-4 achieves the best results compared to other backbones. Among all backbones, GPT-4 has the best performance on the MIXATIS and MIXSNIPS benchmarks. Specifically, compared to other backbones, DSCP can be improved by at least $5 . 7 3 \%$ on Intent Acc, which shows that larger models and better training lead to better performance. DSCP achieves superior performance. DSCP surpasses all previous baselines on all backbones and achieves superior performance, while traditional CoT strategies fail in these benchmarks. Specifically, DSCP outperforms the Plan-and-Solve method by at least $1 . 9 1 \%$ and $1 . 6 3 \%$ on MIXSNIPS and MIXATIS, respectively, which shows the effectiveness of DSCP.

Table 2: Ablation experiments on MIXSNIPS and MIXATIS based on GPT-3.5 backbone.   

<html><body><table><tr><td>Model</td><td>Intent Acc.(%) Macro F1(%)</td><td>Micro F1(%)</td></tr><tr><td colspan="3">MIXSNIPS</td></tr><tr><td>W/o SIDP</td><td>70.44</td><td>90.22</td></tr><tr><td>w/o IISP</td><td>67.62</td><td>84.54</td></tr><tr><td>w/o MICP</td><td>66.07</td><td>87.26</td></tr><tr><td>DSCP</td><td>72.90 91.14</td><td>90.79</td></tr><tr><td colspan="3">MIXATIS</td></tr><tr><td>W/o SIDP</td><td>13.61</td><td>66.29 63.39</td></tr><tr><td>w/o IISP</td><td>24.72</td><td>72.42 70.00</td></tr><tr><td>w/o MICP</td><td>12.64</td><td>63.65</td></tr><tr><td>DSCP</td><td>29.17</td><td>68.25 73.75 70.05</td></tr></table></body></html>

Inter-DSCP brings further performance improvements. Inter-DSCP can further significantly improve performance. As illustrated in Table 1, Inter-DSCP shows superiority over DSCP across all backbones (with at least $1 . 5 3 \%$ improvements on Intent Acc.). We attribute it to the fact that utilizing the interaction ability of LLM can effectively boost zero-shot multi-intent detection.

# 3.4 Analysis

In this section, we conduct thorough analyses to better understand our approach by answering the following questions: (1) Does SIDP help utterance understanding? (2) Can IISP attain better intent detection performance? (3) What effects can MICP bring? (4) How DSCP performs on multiintent detection? (5) Is DSCP a robust method across different prompting? (6) Can DSCP be improved by few-shot demonstrations? (7) Is DSCP interpretable and user-friendly for humans? (8) Why DSCP works?

Answer 1: SIDP can boost the understanding of the user utterance. We investigate the effectiveness of singleintent division prompting (SIDP) by removing the $S I D P$ from DSCP and preserving the other prompting. As shown in Table 2 (w/o SIDP), we find that the performance on Intent Acc. drops significantly about $2 . 5 \%$ on MIXSNIPS. This is because missing single-intent division operations make it hard to effectively understand complex multiple intents in the utterance, which limits the performance.

Answer 2: IISP matters for intent detection on the divided span. This section explores the effectiveness of IISP by removing IISP and keeping other prompting unchanged. As shown in Table 2 (w/o IISP), we find the Intent Acc., Macro F1 , and Micro F1 exhibit significant decreases of over $5 \%$ . We attribute it to the fact that IISP can address simple single-intent detection more easily than directly performing multi-intent detection.

![](images/a706bbde54affec4932a400647bf20c07d9036ebc32dff366e0974f6e114f264.jpg)  
Figure 3: Performance differences under different number of intents on DSCP and $\mathtt { Z e r o \mathrm { - } C o T }$ based on GPT-3.5.

![](images/6db344cf833743683c20673001eec2cfeec511f6eb2c336b3695bdde3b19306e.jpg)  
Figure 4: The robust analysis across different prompts based on GPT-4 backbone.

Answer 3: MICP can eliminate redundancy and unreasonable intents. We verify the effectiveness of multiintent combination prompting (MICP) by removing the procedure of MICP. We still preserve “Result Regulation” prompting $( { \mathcal { R } } )$ for better result extraction. As illustrated in Table 2 (w/o MICP), when removing $M I C P$ , we observe that it drops by $6 . 8 3 \%$ on Intent Acc, $3 . 5 3 \%$ on Macro F1 and $3 . 5 3 \%$ on Micro F1 for MIXATIS. In addition, in our indepth exploration, we observe that MICP not only collects and regulates the output, but also eliminates redundancy and unreasonable intents, which enhances the performance.

Answer 4: DSCP can performs better on more intents. We further explore whether DSCP can perform better in multi-intent scenarios. Specifically, we conduct a grouping based on the number of intents in the multi-intent datasets on MIXSNIPS. From the Figure 3, we observe that as the (a) The artificial interpretability scores of different model outputs and the average time spent by humans in understanding these outputs.

![](images/619004d13c6cef01d078b6337d44c3e850941b5d45fab4489283c09563457ec6.jpg)  
Figure 5: The interpretability analysis across different prompts.

![](images/89e9965325a9cdf9f650df53500fee782f5291523ec36be7e77c91d87e0cf34f.jpg)  
(b) The explanatory analysis of why the DSCP surpasses Zero-CoT. We manually attribute the performance enhancement to three phases of DSCP.

![](images/97449219c935cb93b1853f7f6cb8716e3d017170dbbeee2a559bc088a8ffd6c8.jpg)  
Figure 6: Evaluation of GPT-3.5’s few-shot learning capability. The 3-shot examples are randomly chosen from the development dataset.

number of intents increased, the performance of DSCP improved. More importantly, the performance gap between DSCP and Zero-CoT also increased as the number of intents increased, which demonstrates that our model can attain better performance on complex multi-intent scenarios.

Answer 5: DSCP is robust across different prompting. To analyze the robustness of DSCP, we employ four synonymous prompts that convey the same meaning but use different expressions. Specifically, we leverage GPT-4 to generate three guiding prompts that are synonymous with the three prompt contents we have proposed. Figure 4 presents the performance of four distinct DSCP prompts. Across all metrics, the average performance of DSCP surpasses that of Least-to-Most prompting, which further verifies the robustness of DSCP.

Answer 6: DSCP can be well improved by few-shot demonstrations. To further analyze the model’s performance in a few-shot setting, we randomly select three samples from the validation set, and manually modify them to relevant natural language format. As shown in Figure 6, DSCP can significantly improve the effectiveness of GPT3.5 by few-shot learning, even exceeding the effect of GPT-4. Moreover, compared with the Plan-and-Solve model, the performance of the model is significantly improved, which further shows the effectiveness of DSCP.

Answer 7: DSCP is an interpretable and user-friendly framework To evaluate the interpretability and userfriendliness of DSCP, one must consider both the transparency of its underlying processes and the ease with which users can interact with it. To this end, we conduct a manual evaluation of the generated outputs, requiring evaluators to assign interpretability scores and record the time taken to understand the outputs and annotate the scores. As shown in Figure 5 (a), our framework achieves an interpretability score of 9.62 and shorter time spent in output understanding, significantly better than $\mathtt { Z e r o - C o T }$ . This is because the “Let’s think step-by-step!” approach fails to induce step-bystep reasoning in multi-intent tasks, often directly providing answers, which greatly suppresses the model’s interpretability. Furthermore, we require annotators to figure out which prompting strategies in DSCP lead to the enhanced predictions. As illustrated in Figure 5 (b), we can observe that all output can be explained by the relevant prompting strategies. And $67 \%$ of the examples resulted from the introduction of SIDP, which introduces more fine-grained interpretability.

Answer 8: Qualitative Analysis We conduct qualitative analysis to better understand DSCP by providing a case study comparing the outputs generated by the $\scriptstyle { \mathrm { Z e r o - C o T } }$ approach and our DSCP approach. As illustrated in Figure 7, we observe that Zero-CoT incorrectly predicts “SearchCreativeWork#RateBook” as “AddToPlaylist#GetWeather”. This is because Zero-CoT

Zero-shot Chain-of-Thought Divide-Solve-Combine Prompting u= [Task Instruction]: You need to annotate [Task Instruction]: Assuming you are … Step 1: Divide the sentence into multiple parts: some sentences … [Label Constraint]: [AddToPlaylist; …] Part 1: give a mortal flower a two… [Label Constraint]: [AddToPlaylist; …] [Single-Intent Division Prompting]: Part 2: tell me the actors … [Regulation]: You finally need … Firstly, you need to divide the Part 3: find worldly goods starting… [Given Input]: Here is the sentence: [Intent-by-Intent Solution Prompting]: Step 2: Determine the intents for each part: Here is the the sentence: I want to give Secondly, you need to consider … Part 1: RateBook …, a mortal flower a two, can you tell [Multi-Intent Combination Prompting]: Part 2: SearchScreeningEvent …, Let's think step by step! Finally, you need to consider … Part 3: SearchCreativeWork … [Given Input]: Here is the sentence: I Step 3: Combine all the intents: Intent=AddToPlaylist#GetWeather# want to give a mortal flower a two … Result=SearchCreativeWork#SearchScreeni SearchScreeningEvent ? ngEvent#RateBook (a) The reasoning result for Zero-CoT. (b) The reasoning result for DSCP.

makes it difficult to understand complex multi-intent detection. In contrast, DSCP can correctly predict the multiple intents. Unlike Zero-CoT, DSCP explicitly requires LLM to decompose the utterance first, then solve it intent-by-intent, and finally combine all intents, which not only reduces the difficulty of the task, but also improves the interpretability.

# 4 Related Work

# 4.1 Prompt Learning

Recent advancements in Large Language Models (LLMs) have significantly enhanced their performance in various NLP tasks (Chowdhery et al. 2022; OpenAI 2023). A key development is the improved chain-of-thought (CoT) reasoning capabilities (Wei et al. 2022; Kojima et al. 2022; Zhou et al. 2022; Zheng et al. 2023; Qin et al. 2023a; Chen et al. 2024a,b; Feng et al. 2024; Chu et al. 2023). Specifically, Wei et al. (2022) first highlight LLMs’ remarkable ability for multi-step chain-of-thought reasoning. Based upon this, Kojima et al. (2022) first introduce a novel and efficient trigger phrase, “Let’s think step by step!”, to activate zero-shot multi-step reasoning. Building on this, Zhou et al. (2022) propose the Least-to-Most to divide the hard request into simple requests and solve them one by one. Wang et al. (2023) develop the Plan-and-Solve method, which separates the process into planning and executing sub-tasks. Yao et al. (2023) and Hu et al. (2024) propose a tree-format thought path for better reasoning performance. Furthermore, Yang et al. (2024) introduce the concept of a meta-buffer to store a series of high-level cognitive templates for better reasoning, which are distilled from the problem-solving processes. Qin et al. (2023a) and Chen et al. (2024b) extend the vanilla CoT into cross-lingual and multi-step cross-modal scenarios.

In this work, we investigate the zero-shot multi-intent scenario. To this end, we introduce the DSCP to explicitly decouple the zero-shot multi-intent detection process and Inter-DSCP to further utilize the interactivity of LLMs.

# 4.2 Intent Detection

Intent detection is a fundamental component of task-oriented dialogue systems, as it enables the extraction of the user’s intent during interactions (Tur and De Mori 2011). Recent advancements in deep neural networks have led to diverse methodologies for single-intent detection (Goo et al. 2018; Qin et al. 2019, 2021a). However, these methods predominantly focus on single-intent scenarios, which limits their applicability in real-world environments. To overcome this, research has pivoted towards multi-intent SLU (Qin et al. 2020; Cheng et al. 2023; Qin et al. 2024). Gangadharaiah and Narayanaswamy (2019); Qin et al. (2020) pioneer joint modeling approaches for this domain and introduce MIXATIS and MIXSNIPS benchmarks for the multiintent SLU community. Based on this, Qin et al. (2021b) propose a non-autoregressive framework for faster speed. Additionally, Xing and Tsang (2022) introduce a heterogeneous semantics-label graph for multi-intent SLU. Song et al. (2022) and Pham, Tran, and Nguyen (2023) utilize the label knowledge between intents and slots. Recently, Pan et al. (2023); He and Garner (2023) and Zhu et al. (2024a) explore a vanilla prompt framework to solve zero-shot intent detection, which does not require any training data.

In contrast to their approaches, we introduce a dividesolve-combine prompting (DSCP) framework and an interactive version (Inter-DSCP), achieving to track the explicit reasoning process and improve the interpretability.

# 5 Conclusion

In this paper, we introduce a Divide-Solve-Combine Prompting (DSCP) for zero-shot multi-intent detection to enhance explicit reasoning and interpretability. Specifically, DSCP contains single-intent division prompting, intent-byintent solution prompting, and multi-intent combination prompting to first split an input utterance into multiple subsentences, solve the intents one-by-one, and finally combine the final multi-intents, respectively. Additionally, we further propose an Interactive Divide-Solve-Combine Prompting (Inter-DSCP) to capture the LLM interaction capabilities. Experiments on MIXATIS and MIXSNIPS show that DSCP and Inter-DSCP obtain promising performance.