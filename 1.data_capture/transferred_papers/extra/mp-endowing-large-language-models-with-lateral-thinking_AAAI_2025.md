# MP: Endowing Large Language Models with Lateral Thinking

Tian Bai 1, Yongwang Cao 1, Yan Ge 2, Haitao Yu 3\*

1College of Computer Science and Technology, Key Laboratory of Symbolic Computation and Knowledge Engineering, Ministry of Education, Jilin University 2Graduate School of Comprehensive Human Sciences, University of Tsukuba 3Institute of Library, Information and Media Science, University of Tsukuba baitian@jlu.edu.cn, caoyw $2 3 \textcircled { a }$ mails.jlu.edu.cn, $\mathrm { s } 2 3 3 0 5 3 9 @$ u.tsukuba.ac.jp, yuhaitao $@$ slis.tsukuba.ac.jp

# Abstract

The recent studies show that Large Language Models (LLMs) often fall short in tasks demanding creative, lateral thinking due to lacking a clear awareness of their own reasoning processes. To cope with this issue, we propose a novel metacognitive prompting method (titled as MP) by mimicking human metacognition. Through integrating metacognitive principles, MP endows LLMs with lateral thinking ability, thereby enhancing their abilities to strategize, monitor, and reflect on their responses when dealing with creative tasks. The experimental results with five base LLMs across three lateral thinking datasets demonstrate that: All LLMs armed with MP consistently outperform the representative baseline methods. For example, MP demonstrates superior performance over CoT prompting across Sentence Puzzle $( + 5 . 0 0 \% )$ , Word Puzzle $( + 1 0 . 0 7 \% )$ , BiRdQA $( + 6 . 4 8 \% )$ , and RiddleSense $( + 2 . 6 5 \% )$ with GPT-3.5-turbo model. In particular, the deployment of MP with GPT-4 achieves significant performance improvements that even surpass human performance on BRAINTEASER benchmark, demonstrating the transformative potential of MP in enhancing the creative problem-solving abilities of LLMs.

# Introduction

According to the study by Waks (1997), human reasoning processes comprise two types of thinking: vertical thinking and lateral thinking. Vertical thinking (also known as logical thinking) is a hierarchically ordered process based on rationality, logic, and rules, in which every single step has to be correct and justified before moving to subsequent stages, typically associated with the left-brain hemisphere. Take the question from the widely used dataset CommonsenseQA (Talmor et al. 2019) on commonsense reasoning for example, “Where would I not want a fox”, through the predatory relationship between foxes and hens, the answer hen-house can be derived. Lateral thinking (also known as thinking outside the box) uses an indirect and creative approach via reasoning that is not immediately obvious. It involves ideas that may not be obtainable using only traditional step-by-step logic. For example, given the question from the task of BRAINTEASER (Jiang et al. 2023), “What is the capital in London?”, a highly probable mistake is to think about what the capital of London is. Yet, the key is being able to think with an eye to the different meanings of capital, the puzzle can be readily solved.

The recent advancements in large language models (LLMs) (Devlin et al. 2018; Brown et al. 2020) have revolutionized prior artificial intelligence (AI) paradigms and shown remarkable performance in various fields (Min et al. 2023; Wang, Zhao, and Petzold 2023; Zhang, Ji, and Liu 2023; Zhang et al. 2024). For example, unlike earlier chatbots, ChatGPT (Achiam et al. 2023) can engage in coherent and contextually relevant conversations over multiple turns. Sora (OpenAI 2024), known as a text-to-video model, is capable of generating vivid and imaginative scenes based on textual descriptions. The aforementioned successes have heightened people’s expectations regarding AI systems’ capabilities towards human-like thinking. A group of representative studies are the prompting techniques detailed in the section of related work. For instance, chain of thought (CoT) prompting (Wei et al. 2022) enables models to solve problems by methodically breaking them down into sequential steps. Alternatively, Self-consistency (Wang et al. 2023) enhances performance by generating diverse reasoning paths and selecting the most coherent one. Despite the remarkable advancements, the recent studies (Jiang et al. 2023) show that most of the aforementioned efforts towards human-like thinking have significantly improved LLMs’ vertical thinking ability, leaving how to enhance LLMs’ lateral thinking ability unexplored. To cope with this problem, the BRAINTEASER benchmark (Jiang et al. 2023) was introduced to evaluate LLMs’ lateral thinking ability. Benefiting from advanced capabilities of GPT-4 (Achiam et al. 2023), some methods (Li et al. 2024; Monazzah and Feghhi 2024) based on GPT-4 have achieved state-of-the-art performance. Unfortunately, their effectiveness diminishes a lot when switching to other base LLMs such as the relatively smaller or weaker GPT-3.5. A key shortcoming is that these methods lack the ability to recognize errors that occur during the reasoning process when compared with human intelligence, preventing them from solving more complex problems.

When confronted with complex problems requiring lateral thinking, humans tend to reflect on their thinking processes, continuously adjusting and optimizing their strategies. This ability is directly aligned with metacognition

Cognitive Regulation Input Question Cognitive Cognitive Regulation for human Knowledge for LLMs Planning:selecting appknowledgeabout Planning: design ropriate strategies and allocating resources tasks appropriate strategies 2 Monitpring: eingad aek) performance use a given strategy Knywledwhabout Reateoning: conduct initial reasoning Evaluating:appraising Awarenessand Reflecting:assessing and products and regulatory 2 management of optimizing processes and processes of one's_learningi Response cognition Answer outcomes (a) Human Metacognition (b) LLMMetacognition

which refers to an individual’s awareness and regulation of their own cognitive processes and outcomes (Lai 2011; Schraw and Moshman 1995). Motivated by these insights, we explore how to incorporate metacognition into LLMs to enhance their lateral thinking ability. Metacognition can be simply defined as “thinking about thinking” or “cognition about cognition”. As shown in Figure 1, metacognition comprises two components: cognitive knowledge and cognitive regulation. Cognitive knowledge mainly includes an individual’s self-awareness of their own cognitive ability, knowledge about the target task, and knowledge about why and when to use a given strategy. Cognitive regulation refers to the reflection and control of one’s cognitive processes. Armed with metacognition, humans excel in tasks requiring lateral thinking through continuous self-reflection and regulation. Drawing inspiration from the human metacognitive process, we introduce metacognitive prompting (MP). It encompasses three main steps: 1) Initially, we design a strategy based on the specific characteristics of the task and relevant cognitive knowledge. 2) Subsequently, the strategy is integrated into the prompts, guiding the LLM to identify implicit information within the question and generate an initial answer. 3) Finally, the LLM is prompted to reflect on the previous reasoning process and reason to arrive at the final answer.

To evaluate the efficacy of MP, we conducted experiments on three datasets: BRAINTEASER (Jiang et al. 2023), BiRdQA (Zhang and Wan 2022), and RiddleSense (Lin et al. 2021), encompassing two primary task types involving lateral thinking: brain teasers and riddles. We selected several LLMs, including GPT-3.5-turbo, GPT-4 (Achiam et al. 2023), LLaMA3 (AI@Meta 2024), and Qwen (Team 2024). The experimental results indicate that MP outperforms existing prompting methods and achieves new state-of-the-art on dataset BRAINTEASER and BiRdQA, demonstrating that incorporating metacognition into prompting enhances LLMs’ lateral thinking ablity. The contributions of this paper are as follows:

• We propose a novel metacognitive prompting method by aligning with human-like metacognition, which endows LLMs with lateral thinking ability.

• We conducted a series of experiments on three datasets requiring lateral thinking. The results show that MP outperforms all strong baseline methods and achieves new state-of-the-art performance on the BRAINTEASER and BiRdQA datasets, demonstrating its effectiveness in enhancing the lateral thinking ability of LLMs. • The validity of MP was further confirmed by analyzing its key steps and error samples. Additionally, experimental results demonstrate that MP consistently outperforms all baseline methods across LLMs of varying scales.

# Related Work

In this section, we review the highly relevant studies by grouping them into three groups, namely in-context learning, prompting techniques for LLMs, and metacognition in LLMs.

In-context Learning With the increasing scale of language models, they have shown remarkable proficiency in in-context learning (ICL) (Dong et al. 2022). ICL allows LLMs to perform a wide range of tasks by simply providing a few examples and task descriptions, eliminating the need for gradient updates or additional training data. This ability to generalize across tasks in a text-generating manner has established ICL as a new paradigm in natural language understanding (NLU) (Brown et al. 2020; Kojima et al. 2022). However, merely increasing the model’s size has shown diminishing returns, prompting research to refine and expand ICL’s applications through better prompt design, example selection, and integration with other learning approaches (Wei et al. 2022; Chen et al. 2023).

Prompting Techniques for LLMs In addition to the prompting techniques concentrating on optimizing prompts for specific tasks, an increasing amount of research is choosing to integrate human-like thinking into prompt design. For example, CoT prompting (Wei et al. 2022) enables LLMs to solve problems step-by-step by providing a series of brief sentences that mimic the human reasoning process (Wei et al. 2022). Least-to-Most prompting breaks down a complex problem into a series of simpler sub-problems and then solves them in sequence (Zhou et al. 2023). Inferential $E x$ -

Q: What was the result of the Step 1 Strategy Formulating on Metacognitive Knowledge tiny lobster's math test? Based on task-relevant metacognitive knowledge $\mathbf { K _ { p } }$ :   
C: A:Very-bad. {...}, Ks: {...}, Kt: {...} and information T, generate B: Sea-plus. Ks strategies for solving task. C:Very-Good. Task information D:None of above. meta-knowledge 购 Strategy: These word puzzles may play on AI generated human- words include... Double Entendre, designed Homophones, ... and so on. You ne- Strategy: ed think out of the box and pay more OR attention to the implicit meaning. Step 2 Information Processing and Initial Reasoning   
Input: Promptidentify: Given a brain teaser and several Input: Promptreason: Given a brainteaser and four options, choose options, identify the type of wordplay or puzzle the most correct option and eplain why. being used and where does it happen? Wordplay: ... Strategy: Q&C: ... Q&C: Output: A: Thinking about the result of a math test for a tiny lobster is   
Output: A: The type of wordplay in this example is based on already an unusual concept, so we need to think creatively...By puns and double entendre. ...This is a clever way to considering the unique combination of elements in the create humor through language. wordplay, the most suitable and creative option is 'Sea-plus', making it the correct answer. Answer: 1: Sea-plus. Step 3 Deep Reflection and Further Reasoning   
Input: PromptRefl:Given a brainteaser and a context which contains two possible thinking, choose the most correct. C Context: Possible thinking1: The type of wordplay in this example is based on puns and double entendre. The question sets up the expectation for a math-related answer, but the answer is actually a play on words with "sea-plus" sounding like "C-plus." "Sea-plus" Possible thinking2: ...This play on words combines the idea of the tiny lobster being in the sea with the grade received on a math test. ...Answer: 1:Sea-plus. Q&C: ...   
Output: A: The question...is a math test specifically for a tiny lobster, so the answer must be related to both math and the sea. The "C-plus" pun here is on the typical grading system, where "Sea-plus" is a play on "C-plus" in terms of letter grades. ... Answer:1

clusion Prompting (IEP) (Tong et al. 2023) and process of elimination $( P o E )$ (Balepur, Palta, and Rudinger 2023) combine the principles of elimination and inference. Inspired by human cognitive processes, our approach explores the integration of metacognition into LLMs to enable their awareness of the reasoning process, thereby enhancing their lateral thinking ability.

Metacognition in LLMs To lay the groundwork for our framework, understanding the fundamental concept of metacognition is paramount. The term metacognition is borrowed from cognitive science which refers to an individual’s awareness and control of their own cognitive processes, abilities, and outcomes (Stuyck, Cleeremans, and Van den Bussche 2022; Kaadoud et al. 2022). There are two main aspects of the research on metacognition in LLMs. The first one is the utilization of metacognitive knowledge in large models. Didolkar et al. (2024) extract metacognitive knowledge from LLMs to solve the complex mathematical problem. The other one is to build an introspective system. Zhou et al. (2024) integrates metacognition into Retrieval-Augmented Generation (RAG) to solve multi-hop QA tasks and Wang and Zhao (2024) explores the significance of metacognition for natural language understanding (NLU). However, the application of metacognition in enhancing LLMs’ lateral thinking ability remains largely unexplored. Our approach seeks to bridge this gap by concentrating on two aspects: designing strategies using metacognitive knowledge and establishing mechanism that combine initial reasoning and reflection.

# Metacognitive Prompting

In this section, we detail our methodology for integrating metacognition into LLMs to enhance their lateral thinking ability.

# Task Description

The input consists of a question $\boldsymbol { Q }$ and several candidate options $\bar { C } = \{ c _ { i } \} _ { i = 1 } ^ { n }$ , where $n$ is the number of options. The goal for LLMs is to generate an answer $y$ that correctly identifies the most suitable option corresponding to the question. This can be represented as:

$$
y = \mathbf { L } \mathbf { L } \mathbf { M } _ { Q A } ( Q , C , { P r o m p t } ) ,
$$

$L L M _ { Q A }$ denotes the LLM concentrating on questionanswering tasks. P rompt refers to the additional contextual

or instructional information provided to the LLM to guide its response generation.

# Overview of MP

The overall framework of MP is depicted in Figure 2. MP primarily encompasses three main steps: (1) Planning; (2) Reasoning; (3) Reflecting. The following sections introduce the details of these three steps.

# Planning: Designing Appropriate Strategies

In metacognition, the concept of planning involves the selection of appropriate strategies to guide cognitive processes. In this step, we develop a specific strategy $S$ for each task, based on relevant cognitive knowledge. Specifically, we first summarized the cognitive knowledge $( K = \{ K _ { p } , K _ { s } , K _ { t } \} )$ essential for the task. For instance, brain teasers require the LLM to recognize potential pitfalls $( K _ { p } )$ , apply strategies such as breaking down the problem and considering multiple perspectives $( K _ { s } )$ , and understand that these brain teasers often contain metaphors and distractions, necessitating analysis of implied clues $( K _ { t } )$ . Subsequently, we manually designed a strategy based on cognitive knowledge to guide the reasoning processes, while also the considering AI-generated strategy. A sample strategy is shown in step 1 of Figure 2. In contrast to CoT and its variants, which primarily integrate thinking processes into demonstrations but often overlook relevant knowledge, our approach enables the model to not only infer the reasoning process from demonstrations but also acquire the knowledge and problem-solving suggestions embedded in the strategy.

# Reasoning: Conducting Initial Analysis

In this step, LLM integrates the strategy to conduct initial reasoning on the question and options. Questions such as brain teasers and riddles tend to play tricks on the problem description including metaphors, personification, hyperbole, puns, syntheses, and so on. Therefore, correctly identifying the type of tricks has become the key to solving this problem. This process can be represented as:

$$
W = L L M ( Q , C , S , P r o m p t _ { \mathrm { i d e n t i f y } } )
$$

P romptidentify directs the LLM to discern the wordplays $( W )$ within the problem, thereby providing clues for the subsequent reasoning. Following this, guided by P romptreason, LLM evaluates each option based on the strategy and the identified wordplay:

$$
y _ { p } = L L M ( Q , C , S , W , P r o m p t _ { \mathrm { r e a s o n } } )
$$

$y _ { p }$ includes the selected option and an explanation for the option.

# Reflecting: Assessing Thought Processes

Reflecting refers to assessing and optimizing the processes and outcomes of cognitive activities. As shown in Step 3 of Figure 2, we feed the previous reasoning process as the context for the original question into the LLM for evaluation and reflection. LLMs often struggle with lateral thinking tasks due to misleading information that hinders them from reaching the correct answer in a single attempt. Additionally, we observed that sometimes the text generated through in-context learning prompting may not accurately represent the model’s actual thought process. Inconsistencies between the final answer and the reasoning steps also occasionally occur, suggesting that even reasoning steps leading to an incorrect answer may contain useful information. To address these challenges, we thoroughly evaluate and reflect on the previous reasoning steps. This process can be represented as:

$$
y = L L M ( Q , C , W , y _ { p } , P r o m p t _ { \mathrm { R e f l e c t } } )
$$

where $y$ includes the selected option and an explanation for it. P romptReflect guides the model in evaluating and reflecting on the previous reasoning process to arrive at the final answer.

# Experiments

# Datasets and Evaluation Metric

To evaluate the effectiveness of our approach, we perform experiments on three datasets requiring lateral thinking:

BRAINTEASER A multiple-choice question answering task designed to test the model’s ability to exhibit lateral thinking and defy default commonsense associations.(Jiang et al. 2023) It has two different types of sub-tasks: Sentence Puzzle $( S P )$ and Word Puzzle (WP). Sentence Puzzle focuses on assessing the understanding of specific scenarios and Word Puzzle emphasizes understanding of word meanings and constructions.

BiRdQA A bilingual multiple-choice question answering dataset with 6614 English riddles and 8751 Chinese riddles (Zhang and Wan 2022). It only involves the part of English in our experiments. Riddles are commonly described with personification and metaphor and play tricks like a pun and misleading information. Each riddle has four distractors that are automatically generated at scale with minimal bias.

RiddleSense Another riddle-style multiple-choice question answering task (Lin et al. 2021). It requires complex commonsense reasoning abilities, an understanding of figurative language, and counterfactual reasoning skills.

As adopted in datasets BRAINTEASER (Jiang et al. 2023), BiRdQA (Zhang and Wan 2022), and RiddleSense (RS) (Lin et al. 2021), we evaluate the model performance with accuracy. Regarding the quality of explanations, we conduct human evaluations based on two criteria: Relevance and Usefulness.

# Baseline Methods

In this work, the following representative baseline methods are compared:

Standard prompting (STD) The approach by Brown et al. (2020) involves providing in-context exemplars of input-output pairs before generating a prediction for a testtime example.

Chain of thought prompting $( \mathbf { C o T } )$ The popular method by Wei et al. (2022) prompts LLMs to generate a series of brief explanations to answer a question step-by-step.

Table 1: The overall performance comparison based on four base LLMs. The best and second-best results are highlighted in bold and underlined, respectively. \* indicates p-value $< 0 . 0 5$ in the t-test.   

<html><body><table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="4">Dataset</td></tr><tr><td>Sentence Puzzle</td><td>Word Puzzle</td><td>BiRd</td><td>RiddlSense(Dev)</td></tr><tr><td rowspan="4">Llama3-8B</td><td>STD</td><td>46.67</td><td>55.21</td><td>55.48</td><td>59.84</td></tr><tr><td>APE</td><td>50.83</td><td>56.25</td><td>55.94</td><td>59.78</td></tr><tr><td>CoT</td><td>55.83</td><td>64.58</td><td>57.74</td><td>59.75</td></tr><tr><td>MP (ours)</td><td>57.78*</td><td>68.40*</td><td>59.04*</td><td>60.89*</td></tr><tr><td rowspan="3">Qwen1.5-14B</td><td>STD</td><td>51.67</td><td>39.58</td><td>65.55</td><td>66.01</td></tr><tr><td>APE</td><td>52.50</td><td>35.42</td><td>65.03</td><td>66.23</td></tr><tr><td>CoT</td><td>48.33</td><td>45.83</td><td>65.62</td><td>66.01</td></tr><tr><td></td><td>MP (ours)</td><td>57.29*</td><td>58.33*</td><td>67.26*</td><td>68.17*</td></tr><tr><td rowspan="3">Qwen1.5-110B</td><td>STD</td><td>61.67</td><td>68.75</td><td>73.97</td><td>74.67</td></tr><tr><td>APE</td><td>54.17</td><td>69.79</td><td>74.93</td><td>75.02</td></tr><tr><td>CoT</td><td>59.17</td><td>76.04</td><td>72.60</td><td>78.00</td></tr><tr><td rowspan="3"></td><td>MP (ours)</td><td>66.67*</td><td>85.42*</td><td>78.08*</td><td>80.20*</td></tr><tr><td>STD</td><td>60.83</td><td>67.71</td><td>69.86</td><td>77.33</td></tr><tr><td>APE</td><td>61.67</td><td>68.75</td><td>70.78</td><td>77.02</td></tr><tr><td rowspan="3">Qwen-max</td><td>CoT</td><td>60.83</td><td>71.88</td><td>73.97</td><td>79.00</td></tr><tr><td>MP (ours)</td><td>67.50*</td><td>84.36*</td><td>81.16*</td><td>82.08*</td></tr><tr><td>STD</td><td>62.50</td><td>76.04</td><td>67.12</td><td>77.47</td></tr><tr><td rowspan="3">GPT-3.5-turbo</td><td>APE</td><td>63.33</td><td>77.08</td><td>70.62</td><td>76.59</td></tr><tr><td>CoT</td><td>68.33</td><td>78.47</td><td>74.41</td><td>78.84</td></tr><tr><td>MP (ours)</td><td>73.33*</td><td>88.54*</td><td>80.89*</td><td>81.49*</td></tr><tr><td rowspan="2">GPT-4</td><td>(Li et al. 2024)</td><td>96.67</td><td>96.88</td><td></td><td></td></tr><tr><td>(Monazzah and Feghhi 2024)</td><td>86.67</td><td>97.92</td><td></td><td></td></tr><tr><td rowspan="2">Human</td><td>MP (ours)</td><td>98.33*</td><td>98.96*</td><td></td><td></td></tr><tr><td></td><td>91.98</td><td>91.67</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table></body></html>

Automatic Prompt Engineer (APE) The method by Zhou et al. (2022) enables automatic prompt generation and selection with LLMs.

State-of-the-art LLM prompting methods for BRAINTEASER Li et al. (2024) identified and categorized over 20 challenging training instances to include in an extended prompt. By employing an ensemble voting strategy, they achieved state-of-the-art performance on Sentence Puzzle with GPT-4. Monazzah and Feghhi (2024) tried ensemble and debate prompting engineering methods and achieved state-of-the-art performance on Word Puzzle.

# Experimental Setting

We conducted experiments using the closed-source GPT4, GPT-3.5-turbo and Qwen-max (Team 2024) models accessed via API invocations. Additionally, we evaluated the performance of three open-source models in our experiments: LLaMA3-8B (AI $@$ Meta 2024), Qwen1.5-14B, and Qwen1.5-110B (Team 2024). For all models, we utilized the default settings, including temperature, top k, and top p, to maintain consistency and reproducibility. In the context of the few-shot setting, the number of demonstrations utilized is consistently set to 4 across all three baseline methods as well as in our proposed approach.

# Results and Analysis

Table 1 shows the overall performance comparison among our proposed approach and the baseline methods based on four base LLMs. From Table 1, we can clearly observe that: (1) MP outperforms all baselines on all the 3 datasets. Across varying model sizes, all LLMs boosted with MP consistently outperform baseline methods. Specifically, MP demonstrates superior performance over CoT prompting across Sentence Puzzle $( + 5 . 0 0 \% )$ , Word Puzzle $( + 1 0 . 0 7 \% )$ , BiRdQA $( + 6 . 4 8 \% )$ , and RiddleSense $( + 2 . 6 5 \% )$ with GPT-3.5-turbo model. In particular, MP achieves new state-of-the-art results on the BiRdQA and BRAINTEASER datasets. (2) While CoT prompting generally outperforms standard prompting and APE, its improvements are often modest. In some cases, CoT even performs worse. A probable reason is that the effectiveness of CoT is closely tied to tasks that require successive reasoning steps, which are less applicable in lateral thinking tasks, leading to its diminished performance in such contexts.

Question Implicit Information Incorrect Examples Correct Examples A computer technician would The type of wordplay in this example is a pun. The question Where will a computer In this context, 'key' keep his keys in a designated is setting up an expectation for the answer to be a physical technician keep all his refers to the keys on a spot such as a drawer or location, but the answer is actually a play on the word "key," key? computer keyboard. hook. referring to computer keyboard keys. Why does a man say This brain teaser plays on the double meaning of "jump tjhuamt phihsigdhoegr ctohaulnda A house can not jump. Tdhoegajnusmwperdreovearlas tshmataltlhoer jhuigmhperotvhearna  aphoyusiscea.l" hWouhisle, itthemanyssweermlileiskeinththeedfoagctmthuast house? low-height house. houses cannot jump. $\boldsymbol { \mathsf { v } }$

![](images/a38171a42c8d4225f84d7ca7d76d81a398b8f0f03bd58f4a2664c7d01d009cc8.jpg)  
Figure 4: A comparison of CoT and MP in identifying key implicit information based on a manual analysis of all examples in BRAINTEASER. Y-axis shows the percentage of successfully identified instances out of the total examples.

# Ablation Study

To comprehensively investigate the effectiveness of incorporating metacognition into prompt design, we conduct additional experiments to analyze the roles of MP’s key steps and manually access the quality of the generated explanations.

# Impact of Different Strategies

In the step of planning, the objective of designing these strategies is to enable LLMs to fully leverage cognitive knowledge, enhancing their understanding of the target task and establishing a comprehensive approach before formal reasoning. To verify the contribution of strategies to MP’s performance, we integrated the strategies into demonstrations designed for few-shot learning (Brown et al. 2020) and evaluated their performance. As Table 2 shows, the performance of MP is already better than baselines just by adding the strategy. Initially, we manually designed strategies for different tasks and achieved promising performance. Subsequently, we prompt the model to automatically generate strategies, as defined by the following formula:

Figure 3: Two examples of success and failure in recognizing the wordplay.   
Table 2: Strategy1 and Strategy2 are designed manually and StrategyAI is generated by Qwen-max. The number of demonstrations is set to 4. The row of MP(w/o. Reflect) stands the performance of MP without the step Reflecting. All the experiments are conducted with GPT-3.5-turbo.   

<html><body><table><tr><td>Method</td><td>SP</td><td>WP</td><td>BirdQA</td><td>RS (Dev)</td></tr><tr><td>STD</td><td>62.50</td><td>76.04</td><td>67.12</td><td>77.47</td></tr><tr><td>APE</td><td>63.33</td><td>77.08</td><td>70.62</td><td>76.59</td></tr><tr><td>CoT</td><td>64.16</td><td>78.47</td><td>74.41</td><td>78.84</td></tr><tr><td>Strategy1</td><td>70.00</td><td>81.25</td><td>76.30</td><td>80.51</td></tr><tr><td>Strategy2</td><td>68.33</td><td>83.33</td><td>78.22</td><td>80.41</td></tr><tr><td>StrategyAI</td><td>71.67</td><td>84.38</td><td>77.12</td><td>80.71</td></tr><tr><td>Average</td><td>70.00</td><td>82.99</td><td>77.21</td><td>80.54</td></tr><tr><td>MP(w/o.Reflecting)</td><td>72.50</td><td>86.46</td><td>77.97</td><td>79.72</td></tr><tr><td>MP</td><td>73.33</td><td>88.54</td><td>80.89</td><td>81.49</td></tr></table></body></html>

$$
S = L L M _ { \mathrm { g e n } } ( K _ { p } , K _ { s } , K _ { t } , T )
$$

where T denotes task-specific information such as the inputoutput format of the task. We evaluate the performance of manually designed and AI-generated strategies. The results show that methods combining any of the three strategies outperform all the baselines. Notably, the strategy generated by Qwen-max (Team 2024) achieved the best performance across the three datasets.

# Effect of the Reasoning and the Reflecting Step

In our approach, the reasoning step is designed to make an initial attempt to understand the implicit meaning of the question before engaging in formal reasoning. As Figure 3 shows, correct examples generated through the reasoning step effectively identified the implicit information within the problem description, which is crucial for accurate problem-solving. Conversely, incorrect examples generated using CoT prompting were misled by superficial information, leading to vertical reasoning paths and finally incorrect answers. Subsequently, we assessed the outcomes of CoT

Error Type1: Misdirection Q: Who would serve as the team's captain if a crucial assistant captain: departed fromasporting organization? A: The first officer. B:The captain. C: The second officer. D: None of above.   
MP's Final Output The assistant captain's departure won't affect the captain.√But in sportingorganizations, the hierarchytypicallydictatesthat ifacrucial assistant captainwere to depart, the first officer would move up in rank to fill the vacant positionof assistantcaptain,leaving the original captain in their role. The second oficer would not automatically becometheteam'scaptain.Therefore,thefirstofficerwouldserveas   
the team'scaptain in this scenario.X (i) Misdirection error with MP   
Error Type2: Overthinking Q: How many times can you divide 10o by 10? A: Infinite time. B:Once. C: Two times. D: None of above.   
MP's Final Output While option1,"Once,"is theclosest toapossibleanswer from the given options,√the realityis thatyou can divide10oby10an infinite numberof times.Therefore,noneoftheprovidedoptionsaccurately representthetrueanswertothequestion.X (ii) Overthinking error with MP

Table 3: The results for the explanation quality analysis.   

<html><body><table><tr><td rowspan="2">Dataset</td><td colspan="3">Relevance(%)</td><td rowspan="2">Usefulness(%)</td></tr><tr><td>3</td><td>2</td><td>1 1</td></tr><tr><td>Sentence Puzzle</td><td>80</td><td>12</td><td>8</td><td>0 84 16</td></tr><tr><td>Word Puzzle</td><td>82</td><td>12</td><td>6 85</td><td>15</td></tr><tr><td>BiRdQA</td><td>84</td><td>9</td><td>7</td><td>89 11</td></tr><tr><td>RiddleSense</td><td>83</td><td>8</td><td>9</td><td>90 10</td></tr></table></body></html>

tions are highly relevant, while less than $9 \%$ are irrelevant on all the datasets.

and MP based on their effectiveness in recognizing key implicit information. This review revealed that MP recognized a greater amount of such implicit information, as depicted in Figure 4. Moreover, there is a significant disparity between the number of samples in which CoT successfully identified implicit information and those in which it provided a correct answer. We observed that CoT tends to evaluate the plausibility of options and selects those that appear reasonable without effectively solving the problem. In our MP method, the model initially discerns the true intention behind the question, thereby guaranteeing both the accuracy and interpretability of its responses.

Subsequently, we evaluated the usefulness of the information contained in the explanations, specifically assessing whether the evaluator could correctly answer the question and provide a reasonable interpretation based on the explanation generated by MP. Our results indicate that in over $84 \%$ of cases $90 \%$ for riddles), the evaluator was able to correctly derive the answer and offer a reasonable explanation.

# Quality of Explanation

MP prompts LLMs to generate an explanation alongside the final answer. To evaluate the quality of these explanations, we manually examined 300 random examples, selecting 100 examples from each dataset using MP with GPT-3.5-turbo. Each explanation was scored based on its relevance to the corresponding question, with scores ranging from 1 to 3, indicating unrelated, partially related, and highly related explanations, respectively (Trivedi et al. 2023; Yoran et al. 2023). Our findings indicate that over $80 \%$ of the explana

# Error Analysis

We manually analyzed 250 erroneous instances generated by GPT-3.5-turbo using MP(50 errors on BRAINTEASER and 100 errors on BiRdQA and RiddleSense with GPT-3.5- turbo) and identified two primary error types especially associated with MP as described in Figure 5. The first type is where the model gets misled by distracting information. For example, the question in Error Type 1 led the LLM to focus on the captaincy by emphasizing the departure of the assistant captain. This error is common in the BRAINTEASER which requires thinking outside the box and analyzing implied clues. The second one occurs when LLM overthinks the rationality of every option. This may lead to the negation of a previously generated correct answer in a subsequent step. Figure 5 shows the examples of these two types of errors. In addition to the types above, there are also errors associated with the shortcomings of LLM. For example, some LLMs, even as large as GPT-3.5 with 175B parameters, are not good at understanding word construction. As one output the letter ’S’ occupies the central position in the word ’Paris’ shows, the model fails to accurately distinguish the correct positioning of individual letters within words. Besides, models are often influenced by the cultural background, historical information, and other knowledge stored within them.

# Conclusion

In this work, we introduced metacognitive prompting (MP), which harnesses human meta-cognitive processes to endow large language models with lateral thinking ability. We assessed the effectiveness of MP across three established datasets focused on lateral thinking tasks, consistently observing superior performance compared to existing methodologies across all datasets. Additionally, we conducted an analysis to examine the influence of critical steps on MP’s performance and evaluated the explanations generated by MP, thereby enhancing our understanding of its role in guiding LLMs.