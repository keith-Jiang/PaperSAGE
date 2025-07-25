# TableBench: A Comprehensive and Complex Benchmark for Table Question Answering

Xianjie $\mathbf { W } \mathbf { u } ^ { 1 }$ , Jian Yang1 \*, Linzheng Chai1, Ge Zhang2, Jiaheng ${ { \bf { L i u } } ^ { 1 } }$ , Xeron $ { \mathbf { D } }  { \mathbf { u } } ^ { 2 }$ , Di Liang3, Daixin Shu1, Xianfu Cheng1, Tianzhen $\mathbf { S u n } ^ { 1 }$ , Tongliang $\mathbf { L i } ^ { 4 }$ , Zhoujun $\mathbf { L i } ^ { * }$ , Guanglin Niu1

1Beihang University 2M-A-P 3Fudan University 4Beijing Information Science and Technology University wuxianjie, jiaya, lizj @buaa.edu.cn

# Abstract

Recent advancements in large language models (LLMs) have markedly enhanced the interpretation and processing of tabular data, introducing previously unimaginable capabilities. Despite these achievements, LLMs still encounter significant challenges when applied in industrial scenarios, particularly due to the increased complexity of reasoning required with real-world tabular data, underscoring a notable disparity between academic benchmarks and practical applications. To address this discrepancy, we conduct a detailed investigation into the application of tabular data in industrial scenarios and propose a comprehensive and complex benchmark TableBench, including 18 fields within four major categories of table question answering (TableQA) capabilities. Furthermore, we introduce TABLELLM, trained on our meticulously constructed training set TableInstruct, achieving comparable performance with GPT-3.5. Massive experiments conducted on TableBench indicate that both open-source and proprietary LLMs still have significant room for improvement to meet real-world demands, where the most advanced model, GPT4, achieves only a modest score compared to humans.

# Code — https://github.com/TableBench/TableBench

# Introduction

Recent studies have shown the potential of large language models (LLMs) on tabular tasks such as table question answering (TableQA) (Zhu et al. 2021; Zhao et al. 2023; Hegselmann et al. 2023; Li et al. 2023b; Zhang et al. 2024b; Lu et al. 2024) by adopting in-context learning and structure-aware prompts (Singha et al. 2023), suggesting that a well-organized representation of tables improves the interpretation of tabular. Tai et al. (2023) notes that eliciting a step-by-step reasoning process from LLMs enhances their ability to comprehend and respond to tabular data queries. Furthermore, Zha et al. (2023) investigates the use of external interfaces for improved understanding of tabular data.

Traditionally, adapting language models for tabular data processing entailed modifying their architectures with specialized features such as position embeddings and attention

<html><body><table><tr><td>Year</td><td>Gallons Consumed</td><td>Fuel Expense</td><td>Average Price</td><td>Operating Expcentage</td><td>Available Seat Miles</td></tr><tr><td>2018</td><td>4,137</td><td>$9,307</td><td>$2.25</td><td>24%</td><td>67</td></tr><tr><td>2017</td><td>3,978</td><td>$6.913</td><td>$1.74</td><td>20%</td><td>66</td></tr><tr><td>2016</td><td>3,904</td><td>$5,813</td><td>$1.49</td><td>18%</td><td>65</td></tr></table></body></html>

自

# Multi-hop Fact Checking

# QUESTION:

Fact Checking

Does higher fuel consumption in 2017 and 2018 correspond to a higher fuel expenses in total?

# =

# Multi-hop Numerical Reasoning

# QUESTION:

Numerical Reasoning

How much overall operating expenses increased in 2018 compared to 2017?

M

# Trend Forecasting

# QUESTION:

Estimate what the total operating expense might be in 2019 based on available data.

Data Analysis

S

# Chart Generation

# QUESTION:

Visualization

Please create a line graph based on the years and the corresponding “operating expenses" data.

STEP-2: Self-Inspiration Question Annotation Seed Init golden questions Instruction: Refer to the 曲 with seed samples [Table] and [Type] description. & tCheedceksicfritphteio[nQaunedstisonv]alfiodl.low Q品 Select question as QGueosltdieon s few-shot example Quality Assurance ↓ Instruction: Generate a question × 18 seed task to referring to the [Table] below which generate question. Type: Multi-hop Numerical Reasoning meets the requirements in question [Type] a1ntdab1let,y1peqpuersttiaosnk [EFxeawm-sphleost:Example] LLM eQxupesntsieosn:inHcroewasemducfrhoomve2r0a1ll8otpoe2ra0t1in7?g Drop Q QSyunetshtieotincs Legend 黑 STEP-3: Self-consistency Answer Annotation Type DaWtTasQets STealbelcets TableBench Question ： 0 Sample TCoT Answer-1 VCohtectko 曲 SQA 8 曲 √ Human Table Answer TFaibnFQaAct Filtering Raw ？ SCoT Answer-2 × Check Steps Human LLM 0 Answer-3 Random Annotator Agent STEP-1: Table Collection PoT Select

mechanisms to grasp structural nuances of tables. However, the introduction of LLMs like GPT-4, GPT-3.5 (Brown et al. 2020; OpenAI et al. 2024), and PaLM2 (Anil et al. 2023) has heralded a new approach focused on the art of crafting precise, information-rich prompts that seamlessly integrate table data, coupled with leveraging external programming languages like SQL, Python, or other languages (Wang et al. 2024; Chai et al. 2024), which facilitates more sophisticated chain-of-thought (Wei et al. 2022) (CoT) reasoning processes across both proprietary and open-source LLM platforms, including Llama. Such advancements have propelled the fine-tuning of models for tabular data-specific tasks, showcased by initiatives like StructLM (Zhuang et al. 2024), enhancing capabilities in table structure recognition, fact verification, column type annotation, and beyond. However, the existing benchmark might not entirely resonate with the practical challenges, especially complex reasoning requirements encountered by professionals routinely navigating tabular data in real-world settings. Therefore, there is a huge need for creating a benchmark to bridge the gap between the industrial scenarios and the academic benchmark.

To better evaluate the capability of LLMs in Table QA, we introduce TableBench, a comprehensive and complex benchmark covering 18 subcategories within four major categories of TableQA abilities, as illustrated in Figure 1. First, We systematically analyze real-world challenges related to table applications and define task complexity based on the required number of reasoning steps. Based on the analysis, we introduce a rigorous annotation workflow, integrating manual and automated methods, to construct TableBench. Subsequently, We create a massively TableQA instruction corpora TableInstruct, covering three distinct reasoning methods. Textual chain-of-thought (TCoT) utilizes a textual reasoning approach, employing a series of inferential steps to deduce the final answer. Symbolic chain-ofthought (SCoT) adopts symbolic reasoning steps, leveraging programming commands to iteratively simulate and refine results through a Think then Code process. Conversely, program-of-thought (PoT) generates executable code, using lines of code as reasoning steps within a programming environment to derive the final result. Based on open-source models and TableInstruct, we propose TABLELLM as a strong baseline to explore the reasoning abilities of LLMs among tabular data, yielding comparable performance with GPT-3.5. Furthermore, we evaluate the performance of over 30 LLMs across these reasoning methods on TableBench, highlighting that both open-source and proprietary LLMs require substantial improvements to meet real-world demands. Notably, even the most advanced model, GPT-4, achieves only a modest score when compared to human performance.

The contributions are summarized as follows:

• We propose TableBench, a human-annotated comprehensive and complex TableQA benchmark comprising 886 samples across 18 fields, designed to facilitate factchecking, numerical reasoning, data analysis, and visualization tasks.   
• We introduce TableInstruct, a massive TableQA instruction corpus covering three distinct reasoning methods. TABLELLM, trained on TableInstruct, serves as a robust baseline for TableBench.   
• We systematically evaluate the interpretation and processing capabilities of more than 30 models on our crafted TableBench and create a leaderboard to evaluate

Table 1: Data statistics of TableBench   

<html><body><table><tr><td>Properties</td><td>Value</td></tr><tr><td colspan="2">Basic Insight</td></tr><tr><td rowspan="2">Unique Tables QuestionLength(Avg) Answer Length (Avg) Columns Per Table RowsPerTable RatioofNumerical Cells AverageReasoning Steps</td><td>3681 20.30 8.52</td></tr><tr><td>6.68 16.71 65.74% 6.26</td></tr><tr><td colspan="2">Question Categories</td></tr><tr><td rowspan="2">Fact Checking Numerical Reasoning</td><td>Match-Based Fact Checking</td></tr><tr><td>Multi-hop Fact Checking Arithmetic Calculation Comparison Aggregation</td></tr><tr><td>Data Analysis</td><td>Ranking Counting Time-based Calculation Multi-hop Numerical Reasoning Domain-Specific Descriptive Analysis Anomaly Detection Statistical Analysis Correlation Analysis Causal Analysis</td></tr><tr><td>Visualization</td><td>Trend Forecasting Impact Analysis Chart Generation</td></tr><tr><td>TableBench Size</td><td>886</td></tr><tr><td>TableInstruct Size</td><td>19,661</td></tr></table></body></html>

them on four main tasks. Notably, extensive experiments suggest that comprehensive and complex TableQA evaluation can realistically measure the gap between leading language models and human capabilities in real-world scenarios.

# Construction of TableBench

To bridge the gap between academic benchmarks and industrial scenarios, we comprehensively analyze tabular data applications in real-world contexts, categorizing these problems into four major categories and 18 specific subcategories. We define the complexity of these tasks based on the reasoning steps required for problem-solving and provide detailed guidelines for defining and decomposing these steps, which are rigorously followed during the annotation process. Additionally, we introduce an annotation framework that combines manual and automated methods to enhance annotation efficiency, as illustrated in Figure 2. Finally, we propose two high-quality corpora: TableBench, a comprehensive and complex benchmark consisting of 886 samples, and TableInstruct (20K samples in total), massive instruction corpora designed to instruct LLMs with various reasoning methods.

Competitions Motor Sports Economy Individual Sports Entertainment 18.10% Winter Sports Team Sports   
32.44% Statistics Politics Science Management 4.01% Geography 3.04% Athlete Recreational Miscellaneous   
eey 4250 Health CM 12O C 0 TErdauncsaptiortn /0 Infrastructure Financial Report

# Tabular Data Collection

We collect raw tabular data from existing datasets, including typical datasets such as WTQ (Pasupat and Liang 2015), SQA (Iyyer, Yih, and Chang 2017), TabFact (Nan et al. 2022), FeTaQA (Nan et al. 2022), FinQA (Chen et al. 2021c), AIT-QA (Katsis et al. 2022), etc. To align closely with the ”reasoning complexity of questions” dimension in real-world tabular problems, we do not specifically design for the complexity of the tables themselves, such as structural complexity or large-sized tables. Instead, we adopt a moderate complexity in tabular data. We select tables based on topics and size, ensuring each contains at least 8 rows and 5 columns. We focus on tables with significant numerical values to emphasize numerical reasoning, thereby ensuring depth in numerical computation reasoning. Ultimately, we collect 3681 tables covering 20 major topics: finance, competition, sports, science, etc.

# Question Annotation

We opt to manually construct a more complex set of questions to mitigate the data leak risk in LLMs rather than modi

Table 2: Comparison with existing datasets in categories.   

<html><body><table><tr><td>Dataset</td><td>Fact Checking</td><td>Numerical Reasoning</td><td>Data Analysis</td><td>Visulization</td></tr><tr><td>WTQ</td><td>√</td><td></td><td>X</td><td>X</td></tr><tr><td>SQA</td><td>√</td><td>X</td><td>X</td><td>X</td></tr><tr><td>TabFact</td><td>√</td><td>X</td><td>X</td><td>X</td></tr><tr><td>FeTaQA</td><td>√</td><td>X</td><td>X</td><td>X</td></tr><tr><td>FinQA</td><td>X</td><td>√</td><td>X</td><td>X</td></tr><tr><td>AIT-QA</td><td>X</td><td>√</td><td>X</td><td>X</td></tr><tr><td>WikiSQL</td><td></td><td>√</td><td>X</td><td>X</td></tr><tr><td>Spider</td><td>√</td><td>√</td><td>×</td><td>X</td></tr><tr><td>Bird</td><td>X</td><td>√</td><td>√</td><td>X</td></tr><tr><td>Text2Analysis</td><td>X</td><td>X</td><td>√</td><td>√</td></tr><tr><td>TableBench 一</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table></body></html>

CheckFiancgt 3.27 NReuamseorniicnagl 5.41 Reasoning Steps 4 Data 6.88 Analysis 2   
Visualization 7.29 0 0 2 4 6 WTQ SQA FeTaQA FinQA Spider BIRDTableBench   
a) Reasoning steps of various question categories in TableBench (b) Reasoning steps comparison across different datasets

fying existing datasets. We introduce a self-inspiration question generation mechanism to construct questions across different categories. Firstly, We meticulously craft one seed question and a detailed definition for each category, forming the initial question seed corpus. Subsequently, we incorporate these initial seed questions as examples into a meticulously designed prompt to guide GPT4-based agents in generating questions that adhere to specific category constraints. We limit the output to five questions in the initial rounds. These questions are manually annotated to identify new patterns and added to the seed corpus. We continuously select representative questions into the question seed corpus to promote benchmark qualities, eventually maintained at 50 questions, serving as the test set questions for TableBench. Upon reaching 50 questions per category, we conduct manual annotations on a sample basis $( 3 0 \% )$ , with the remaining questions validated by another GPT-4 agent through a question verification process, eventually serving as the questions for TableInstruct.

# Answer Annotation

We design a self-consistency mechanism for annotating answers based on a given table and question. During the answer generation phase, we utilize three LLM agents, each employing a distinct reasoning method (TCoT, SCoT, and PoT) to generate responses. We introduce a voting mechanism to assess the answers generated by the different agents. We preliminarily reserve the results if the voting system identifies a valid consistency among all agents. These preliminary results are then subjected to manual review and modification to produce the final answer and its associated reasoning details. Additionally, to minimize bias in answers generated by LLMs, we enforce a strict format for all answers, retaining only the essential and accurate content, thereby avoiding any preference for model-specific answer styles. For answers excluded due to inconsistencies, particularly those stemming from questions deemed too complex for LLMs to generate an adequate response, we randomly select $30 \%$ of the filtered data for manual annotation and subsequently incorporate them into the dataset. Notably, We manually annotate all answers in the TableBench with no omissions and carefully scrutinize each.

# Dataset Statistic

Topics TableBench primarily consists of numerical tables, with the largest portions derived from financial reports and data from competitive events, as illustrated in Figure 3,

Question Categories Drawing from real-world scenarios and user demands for tabular data, we devise four primary question categories: fact-checking, numerical reasoning, data analysis, and visualization, encompassing 18 subcategories, thoroughly illustrating the various challenges encountered in TableQA scenarios, as shown in Table 1. Compared to existing datasets, TableBench covers a wider range of question types, as shown in Table 2, with a particular focus on data analysis and chart generation capabilities, which are notably lacking in previous datasets.

Reasoning Steps We define the complexity of the dataset by calculating the number of reasoning steps required to solve the problem. Figure 4 illustrates that the overall complexity of the benchmark is significantly greater than that of existing datasets, especially concerning questions related to data analysis and visualization.

# TABLELLM

# Problem Definition

Table question answering (Table QA) can be formulated as follows: Given a semi-structured table $\tau$ , comprised of $\mathcal { R }$ rows and $\mathcal { C }$ columns, the objective is to generate an answer $\mathcal { A }$ to a question $\mathcal { Q }$ utilizing the information contained within $\tau$ , where $\mathcal { A }$ is a set of values or entities denoted as $\{ a _ { 1 } , a _ { 2 } , \ldots , a _ { k } \}$ , where ${ \boldsymbol k } \in  { \mathbb { N } } ^ { + }$ .

# Reasoning Methods

In-context learning (ICL) (Dong et al. 2022) refers to strategies that optimize input for LLMs $( \mathcal { M } )$ to generate practical outputs with a task-specific instruction $( { \mathcal { T } } )$ and a few output examples $( \mathcal { E } )$ . We introduce distinct reasoning methods to fully assess the reasoning capabilities of LLMs

Textual Chain-of-Thought (TCoT) TCoT (Wei et al. 2022) refers to a reasoning process in which LLMs incrementally derive a series of intermediate steps or sub-goals through textual prompts before generating the final answer. These intermediate steps constitute a ”thought chain” that ultimately leads the model to the correct outcome. Formally, the method is:

$$
\mathcal { M } ( \mathcal { T } , \mathcal { Q } , \mathcal { E } )  \{ r _ { 1 } , r _ { 2 } , \dotsc , r _ { k } , \mathcal { A } \}
$$

where $r _ { k }$ represents the $k$ -th reasoning step.

Symbolic Chain-of-Thought (SCoT) SCoT implements a methodology that utilizes Python-based instruction to facilitate logical reasoning, comprising three primary steps repeated until a definitive conclusion is derived: STEP-1: Analyzing the available information to determine the next move. STEP-2: Generating instructions using Python programming language commands. STEP-3: Simulating the outcomes by executing the instructions and analyzing the results. The entire steps can be formally framed as follows:

$$
\mathcal { M } ( \mathcal { T } , \mathcal { Q } , \mathcal { E } )  \{ ( \boldsymbol { r } _ { a _ { 1 } } , \boldsymbol { r } _ { p _ { 1 } } , \boldsymbol { r } _ { s _ { 1 } } ) , \ldots , ( \boldsymbol { r } _ { a _ { k } } , \boldsymbol { r } _ { p _ { k } } , \boldsymbol { r } _ { s _ { k } } ) , \mathcal { A } \}
$$

where $r _ { a _ { k } }$ is the analyzing step, $r _ { p _ { k } }$ is the program commands generating step, and $r _ { s _ { k } }$ is the result simulation step.

Program-of-Thoughts $( \mathbf { P o T } )$ PoT (Chen et al. 2022) offers a novel approach to numerical reasoning tasks by distinctly delineating computation from reasoning. PoT decomposes the problem into programming commands $\mathcal { P }$ and utilizes a language interpreter, like Python, to compile and execute the resultant code. In contrast to SCoT, PoT enhances reasoning capabilities by actually executing generated code $( \mathcal { P } )$ within a programming environment to output results, thereby implementing reasoning through structured code steps. The method can be formulated as:

$$
\mathcal { M } ( \mathcal { T } , \mathcal { Q } , \mathcal { E } )  \mathcal { P }  \mathcal { A }
$$

# Supervised Fine-Tuning

We train TABLELLM by fintuning all parameters of baseline LLMs to learn from the TableInstruct. The training objective $\mathcal { L } _ { a l l }$ can be described as:

$$
\mathcal { L } _ { a l l } = - \sum _ { n = 1 } ^ { N } \mathbb { E } _ { q ^ { R _ { n } } , a ^ { R _ { n } } \sim \{ D ^ { R _ { n } } \} _ { n = 1 } ^ { N } } \left[ \log P ( a ^ { R _ { n } } | q ^ { R _ { n } } ; \mathcal { M } ) \right]
$$

where $q ^ { R _ { n } }$ and $a ^ { R _ { n } }$ are the table-related question and answer from the dataset $D ^ { R _ { n } }$ of reasoning method $\scriptstyle { R _ { n } }$ , respectively. $N$ is the number of reasoning methods.

# Experiments Implementation Details

We meticulously design uniform style prompt templates to implement distinct reasoning methods to ensure the fairness of the evaluation. Furthermore, we impose formatting constraints on the outputs of LLMs and parse the final answers from the outputs to prevent any extraneous information from affecting the evaluation results. For open-source models, we operate within the transformer environment on multiple A100 GPUs. For proprietary models, we employ official APIs to interact with exclusive LLMs. We conduct supervised finetuning of various open-source LLMs on the designated training set (TableInstruct). We utilize a cosine annealing scheduler, setting the initial learning rate at 2e−5, and conduct training over three epochs. Optimization is performed using the Adam optimizer, with a batch size of 512 and a maximum sequence length of 4096.

# LLMs

We evaluate 34 models with sizes ranging from 7B to 110B parameters, including general/code LLMs, opensource/proprietary models, and SFT (Ouyang et al. 2022) models. For open-source LLMs, we evaluate on Llama2s (Touvron et al. 2023), Llama3s (Grattafiori et al. 2024), Llama3.1s, CodeLlamas (Rozie\`re et al. 2024), CodeQwen1.5-7B-Chat, Qwen1.5s (Bai et al. 2023), Qwen2s (Yang et al. 2024), Mistral-7B-Instructv0.2 (Jiang et al. 2023), Deepseek-Coders (Guo et al. 2024), StructLMs (Zhuang et al. 2024), MAP-Neo-7BInstruct (Zhang et al. 2024a), WizardLM-13B-V1.2 (Xu et al. 2023). For proprietary LLMs, we perform evaluation on GPTs (Brown et al. 2020; OpenAI et al. 2024) (GPT3.5-Turbo, GPT4-Turbo, GPT4-o), Qwen-Max (Yang et al. 2024), GLM-4 (GLM et al. 2024), Yi-Large (AI et al. 2024) and Deepseek models (DeepSeek-AI et al. 2024) (ChatV2, Coder-V2). Furthermore, we finetune TABLELLM based on CodeQwen-7B, DeepSeekCoder-7B, Llama3-8B, Llama3.1-8B, and Qwen2-7B to further explore the Table QA capabilities of LLMs.

# Automatic Evaluation Metrics

we adopt Rouge-L (Lin 2004) to assess the quality of the generated answers by measuring the n-gram overlap with reference answers. In the PoT method, we enforce a specific format for the executable code outputs and evaluate the final answer with the ROUGE-L metric, ensuring alignment with other reasoning methodologies. Specifically, in the task of chart generation, we parse and execute code derived from LLM responses and establish rigorous test cases to assess the accuracy of the generated charts, with a particular focus on the precision of y-axis fields, employing the pass $@ 1$ metric (Chen et al. 2021a) for evaluation.

# Main Results

Table 3 showcases the main results of over 30 advanced advanced LLMs on the TableBench. GPT-4 outperforms other models in numerous tasks, demonstrating superior performance across complex reasoning scenarios. Particularly in numerical computation and analytical tasks, GPT-4 maintains a commendable level of performance. TABLELLM finetuned on the open-source models with TableInstruct achieves a performance level comparable to GPT-3.5, significantly validating the effectiveness of our training data. Despite these advancements, humans still surpass all LLMs in these tasks. Nevertheless, certain advanced LLMs, especially those employing proprietary approaches, demonstrate potential in these scenarios. However, complex reasoning environments on tabular data still remain challenges.

Table 3: The main results of advanced LLMs on TableBench are presented alongside human performance. All methods involving code generation and computation, particularly in the chart generation task, execute code only once to derive the final answer. The overall results represent a weighted average of performance across different categories.   

<html><body><table><tr><td></td><td colspan="3">Fact Checking</td><td colspan="3"> Num-Reasoning</td><td colspan="3">Data Analysis</td><td colspan="3">Visualization</td><td></td><td>Overall</td><td></td></tr><tr><td></td><td>TCoT</td><td>SCoT</td><td>PoT</td><td>TCoT</td><td>SCoT</td><td>PoT</td><td>TCoT</td><td>SCoT</td><td>PoT</td><td>TCoT</td><td>SCoT</td><td>PoT</td><td>TCoT</td><td>SCoT</td><td>PoT</td></tr><tr><td>Human Performance</td><td></td><td>94.3</td><td></td><td></td><td>87.1</td><td></td><td></td><td>82.1</td><td></td><td></td><td>86.3</td><td></td><td></td><td>85.91</td><td></td></tr><tr><td colspan="10">Open-source In ContextLearning Methods</td><td colspan="7"></td></tr><tr><td>Llama2-7B</td><td>34.99</td><td>27.47</td><td>3.61</td><td>6.70</td><td>4.63</td><td>3.95</td><td>14.31</td><td>12.49</td><td>1.56</td><td>0.00</td><td>0.00</td><td>0.00</td><td>12.36</td><td>9.95</td><td>2.76</td></tr><tr><td>CodeLlama-7B</td><td>33.06</td><td>12.34</td><td>19.44</td><td>5.43</td><td>2.99</td><td>13.31</td><td>16.16</td><td>17.06</td><td>1.79</td><td>0.00</td><td>0.00</td><td>0.00</td><td>12.30</td><td>9.28</td><td>8.85</td></tr><tr><td>Gemma-7B</td><td>27.63</td><td>10.07</td><td>21.62</td><td>6.78</td><td>2.91</td><td>10.45</td><td>20.33</td><td>11.76</td><td>6.74</td><td>0.00</td><td>0.00</td><td>2.00</td><td>13.96</td><td>6.97</td><td>9.81</td></tr><tr><td>Mistral-7B</td><td>50.45</td><td>40.56</td><td>6.25</td><td>8.73</td><td>5.77</td><td>2.60</td><td>21.99</td><td>21.12</td><td>1.19</td><td>0.00</td><td>0.00</td><td>0.00</td><td>17.86</td><td>15.11</td><td>2.35</td></tr><tr><td>Deepseek-Coder-7B</td><td>22.92</td><td>27.48</td><td>48.98</td><td>6.45</td><td>5.61</td><td>34.66</td><td>18.73</td><td>20.72</td><td>18.17</td><td>8.00</td><td>18.00</td><td>18.00</td><td>13.10</td><td>14.58</td><td>28.89</td></tr><tr><td>CodeQwen1.5-7B</td><td>30.56</td><td>32.94</td><td>0.00</td><td>6.24</td><td>5.68</td><td>0.00</td><td>27.04</td><td>22.47</td><td>0.00</td><td>2.00</td><td>0.00</td><td>0.00</td><td>16.80</td><td>14.85</td><td>0.00</td></tr><tr><td>Qwen1.5-7B</td><td>56.08</td><td>53.53</td><td>39.23</td><td>11.30</td><td>10.99</td><td>20.40</td><td>24.77</td><td>22.96</td><td>7.66</td><td>0.00</td><td>0.00</td><td>0.00</td><td>20.70</td><td>19.65</td><td>16.29</td></tr><tr><td>Qwen2-7B</td><td>57.70</td><td>57.52</td><td>0.00</td><td>16.09</td><td>16.65</td><td>0.76</td><td>24.02</td><td>21.50</td><td>0.38</td><td>0.00</td><td>4.00</td><td>2.00</td><td>22.77</td><td>22.26</td><td>0.60</td></tr><tr><td>StructLM-7B</td><td>47.72</td><td>64.06</td><td>13.54</td><td>9.55</td><td>19.97</td><td>11.48</td><td>19.59</td><td>23.83</td><td>4.38</td><td>0.00</td><td>0.00</td><td>0.00</td><td>17.06</td><td>25.21</td><td>8.30</td></tr><tr><td>MAP-Neo-7B</td><td>32.70</td><td>33.22</td><td>0.00</td><td>7.23</td><td>6.46</td><td>0.00</td><td>21.85</td><td>14.38</td><td>0.44</td><td>0.00</td><td>0.00</td><td>4.00</td><td>15.26</td><td>12.03</td><td>0.40</td></tr><tr><td>Llama3-8B</td><td>38.32</td><td>72.53</td><td>13.94</td><td>22.02</td><td>17.33</td><td>19.50</td><td>30.15</td><td>30.75</td><td>9.31</td><td>0.00</td><td>0.00</td><td>10.00</td><td>25.71</td><td>27.59</td><td>14.43</td></tr><tr><td>Llama3.1-8B</td><td>47.89</td><td>36.29</td><td>30.38</td><td>11.26</td><td>13.77</td><td>17.24</td><td>15.78</td><td>14.82</td><td>8.86</td><td>8.00</td><td>0.00</td><td>8.00</td><td>16.76</td><td>15.81</td><td>14.88</td></tr><tr><td>Llama2-13B</td><td>48.47</td><td>32.69</td><td>3.03</td><td>15.83</td><td>6.79</td><td>4.48</td><td>22.04</td><td>17.16</td><td>3.19</td><td>0.00</td><td>0.00</td><td>0.00</td><td>20.86</td><td>13.25</td><td>3.61</td></tr><tr><td>StructLM-13B</td><td>26.28</td><td>64.49</td><td>1.04</td><td>12.30</td><td>17.38</td><td>0.00</td><td>20.70</td><td>18.41</td><td>0.28</td><td>0.00</td><td>0.00</td><td>0.00</td><td>16.35</td><td>21.94</td><td>0.21</td></tr><tr><td>WizardLM-13B</td><td>53.93</td><td>46.01</td><td>8.33</td><td>13.79</td><td>16.52</td><td>14.79</td><td>22.61</td><td>20.16</td><td>3.73</td><td>0.00</td><td>0.00</td><td>4.00</td><td>20.75</td><td>20.23</td><td>9.12</td></tr><tr><td>Qwen1.5-14B</td><td>40.83</td><td>61.92</td><td>44.38</td><td>10.29</td><td>15.01</td><td>28.20</td><td>22.99</td><td>29.24</td><td>10.33</td><td>2.00</td><td>8.00</td><td>2.00</td><td>18.03</td><td>25.14</td><td>21.48</td></tr><tr><td>Qwen1.5-32B</td><td>64.99</td><td>67.86</td><td>49.01</td><td>19.13</td><td>21.15</td><td>34.01</td><td>24.27</td><td>28.29</td><td>17.43</td><td>4.00</td><td>8.00</td><td>8.00</td><td>25.38</td><td>28.30</td><td>27.79</td></tr><tr><td>Deepseek-Coder-33B</td><td>48.27</td><td>54.34</td><td>33.12</td><td>9.41</td><td>12.69</td><td>32.60</td><td>9.09</td><td>21.70</td><td>19.97</td><td>0.00</td><td>0.00</td><td>24.00</td><td>13.01</td><td>19.92</td><td>27.20</td></tr><tr><td>CodeLlama-34B</td><td>64.39</td><td>58.28</td><td>5.90</td><td>13.10</td><td>13.30</td><td>4.20</td><td>19.23</td><td>15.28</td><td>0.53</td><td>0.00</td><td>0.00</td><td>2.00</td><td>20.24</td><td>18.19</td><td>2.88</td></tr><tr><td>StructLM-34B</td><td>19.10</td><td>30.21</td><td>27.74</td><td>15.36</td><td>9.03</td><td>14.45</td><td>20.74</td><td>17.92</td><td>5.38</td><td>0.00</td><td>0.00</td><td>2.00</td><td>16.93</td><td>14.37</td><td>11.61</td></tr><tr><td>Mixtral-8x7B</td><td>54.54</td><td>56.01</td><td>35.86</td><td>16.80</td><td>16.05</td><td>26.23</td><td>24.69</td><td>25.67</td><td>13.96</td><td>2.00</td><td>0.00</td><td>6.00</td><td>23.14</td><td>23.24</td><td>21.32</td></tr><tr><td>Qwen1.5-72B</td><td>71.27</td><td>67.03</td><td>33.16</td><td>19.01</td><td>16.68</td><td>20.85</td><td>26.63</td><td>27.33</td><td>13.03</td><td>2.00</td><td>8.00</td><td>14.00</td><td>26.66</td><td>25.80</td><td>18.65</td></tr><tr><td>Qwen2-72B</td><td>72.50</td><td>71.13</td><td>56.37</td><td>36.97</td><td>31.81</td><td>41.33</td><td>32.20</td><td>31.85</td><td>22.36</td><td>20.00</td><td>14.00</td><td>12.00</td><td>38.13</td><td></td><td>33.91</td></tr><tr><td>Qwen1.5-110B</td><td>74.87</td><td>69.80</td><td>53.55</td><td>29.81</td><td>23.33</td><td>36.83</td><td>27.34</td><td>29.32</td><td>18.38</td><td>14.29</td><td>12.00</td><td>24.00</td><td>32.81</td><td>35.14 30.10</td></table></body></html>

Category Analysis Experimental results in Table 3 reveal that most models perform commendably in fact-based reasoning tasks, indicating their proficiency in this area. However, challenges arise in numerical reasoning tasks due to the complexity of mathematical computations, especially complex calculations such as aggregation, which require multiple intermediate steps to reach the final answer. Data analysis tasks necessitate more intricate and comprehensive analytical skills, such as using correlation coefficients to analyze model relationships and employing linear regression functions to predict future trends, thereby imposing higher demands on the overall reasoning abilities of LLMs. The task of chart generation poses the greatest challenge, requiring significant coding skills and strict adherence to instructions. Notably, smaller-sized models exhibit significant deficiencies in chart generation tasks, highlighting their limitations in utilizing code to handle complex tasks.

Reasoning Methods Analysis As illustrated in Table 3, those methods incorporating reasoning steps demonstrate a clear advantage on TableBench compared to methods that derive conclusions directly. The TCoT method exhibits stable and superior performance across various dimensions. The PoT method delivers commendable results in purely numerical computations, particularly in chart generation, but falls short in textual reasoning. We investigate the factors contributing to the suboptimal performance of the PoT method and find that the code execution success rate constrains the performance, as we only conduct a single generation and execute the code without employing any strategy for code correction. Even for the best-performing GPT4- Turbo, the executable code ratio is only $78 . 6 7 \%$ . This indicates that the PoT method requires LLMs with significant code-generation capabilities and instruction-following ability. However, it also underscores the substantial potential of the PoT method. Conversely, the SCoT method adapts effectively in scenarios requiring a combination of numerical and textual reasoning, such as analytical tasks, achieving a balanced yet modest overall performance. The performance of SCoT falls short of expectations due to its reliance on simulated outcomes rather than executing actual code.

Table 4: We performed a consistency test of evaluation methods for advanced LLMs on TCoT performance   

<html><body><table><tr><td></td><td>Auto Metric</td><td>GPT-4 Eval</td><td>Human Eval</td></tr><tr><td>GPT-3.5-Turbo</td><td>30.87</td><td>32.84</td><td>34.12</td></tr><tr><td>Qwen-Max</td><td>34.29</td><td>36.12</td><td>37.12</td></tr><tr><td>Yi-Large</td><td>38.56</td><td>43.12</td><td>41.23</td></tr><tr><td>GLM-4</td><td>34.82</td><td>38.60</td><td>39.21</td></tr><tr><td>Deepseek-Chat-V2</td><td>47.24</td><td>48.31</td><td>50.12</td></tr><tr><td>Deepseek-Coder-V2</td><td>44.92</td><td>44.92</td><td>46.13</td></tr><tr><td>GPT-4-Turbo</td><td>51.30</td><td>52.82</td><td>54.02</td></tr><tr><td>GPT-40</td><td>50.53</td><td>54.18</td><td>53.19</td></tr><tr><td>PCC with Auto Metric</td><td>1.000</td><td>0.981</td><td>0.995</td></tr></table></body></html>

![](images/6e1674ed9ba99769806410c137354ca9a5a11fc186df0197639c77e4dee74130.jpg)  
Figure 5: The impact of the parsing ratio on the overall score, where the parsing ratio is defined as the proportion of responses generated by the LLM that can be successfully parsed according to predetermined instructions.

# Consistency of Evaluation Methods

Despite constraints imposed on the output format and the standardization of ground truth annotations, the ROUGE-L metric may not fully capture the real performance due to the inherent flexibility in the outputs of LLMs. Both GPT4 and human judgment are conducted, as shown in Table 4, to assess this potential bias. The Pearson Correlation Coefficient (Cohen et al. 2009) (PCC) is adopted to analyze the consistency across different evaluation methods. The results, as presented in the table, indicate a high level of agreement among these evaluating methods, demonstrating that the constraints are effective and our metric accurately reflects the real performance of LLMs on the TableBench.

# Further Analysis

# Instruct Following Analysis

We observe that the performance trends of small-size LLMs across different reasoning methods differ from those observed in large-size models in Table 3. In further analysis, we introduce a comparison with non-reasoning methods, specifically focusing on Direct Prompting (DP), which provides solutions directly without intermediate reasoning steps. We find that the non-reasoning method (DP) performs better on small-size LLMs than reasoning-based methods. As shown in Figure 5, most models exhibit good instructionfollowing capabilities with the DP method due to the simpler instructions to follow. Conversely, small-size LLMs perform significantly worse with the PoT method, mainly due to their insufficient code generation capabilities, resulting in a lower rate of executable code generation. Additionally, the iterative symbolic reasoning steps required by the SCoT method pose considerable challenges for small-scale models.

![](images/3af980df9e7dae067edaf02b7a010d824f9e123344e42fd85a5208ad4bfc0010.jpg)  
Figure 6: TableInstruct data efficient on TableLLMLlama3.1-8B

In comparison to the DP, SCoT, and TCoT methods in Figure 5, the data points on the left side of the quadratic curve show that at low parsing ratios, the overall score increases as the parsing ratio decreases, suggesting that certain models (e.g., StructLLM), possess strong table understanding capabilities but exhibit weaker instruction-following abilities. This may be attributed to differences in the instruction format during instruction tuning compared to the format we employ. The right side of the quadratic curve reveals that despite the strong instruction-following performance of the DP method, the non-reasoning DP method faces a clear performance ceiling. In contrast, reasoning-based methods show significant potential for improvement. The curve of the PoT highlights the substantial potential of the PoT to enhance the overall score by increasing the parsing rate.

# Data Efficiency of TableInstruct

In this section, we discuss the data efficiency of TableInstruct on the SFT process. We construct datasets of varying sizes by sampling from TableInstruct with sampling rates ranging from 0.2 to 0.6. Figure 6 visually depicts the relative performance at different sampling rates. Surprisingly, with only $60 \%$ of the samples, the model retains over $90 \%$ of the performance of the complete dataset. We observe that the Llama-3-8B model requires fewer than 4,000 samples to surpass the performance of Qwen1.5-70B on the dataset, demonstrating that the TableInstruct corpus significantly enhances tabular reasoning in smaller models. The full data provides the highest knowledge coverage, enabling the model to achieve optimal overall performance, comparable to GPT-3.5, with inference costs being only a fraction, indicating the high efficiency of TableInstruct.

# Related Work

Table QA (Mueller et al. 2019; Jin et al. 2022) has grown substantially, driven by the development of robust datasets that engage advanced algorithms in the tasks of semantic comprehension (Huang et al. 2024; Li et al. 2023c, 2024b; Bai et al. 2023; Yang et al. 2024; Li et al. 2022a, 2024c,a). These datasets function as significant milestones for enhancing table-centric semantic understanding. WTQ (Pasupat and Liang 2015), SQA (Iyyer, Yih, and Chang 2017), and TabFact (Chen et al. 2020) set the cornerstone for Table QA research. They furnish benchmarks founded on question-answer pairs predicated on HTML tables sourced from Wikipedia. However, these datasets rely heavily on specific cell content from the table to formulate answers, which can not fully represent the multi-dimensional queries posed in real-world scenarios.

Acknowledging this incongruity, some datasets have been introduced to bridge the gap. ToTTo (Parikh et al. 2020), OTTQA (Chen et al. 2021b), and FeTaQA (Nan et al. 2022) step into the fore by providing free-form QA datasets. These datasets challenge models to generate answers that go beyond the table’s explicit content, thereby enhancing model performance to align with the free-form nature of real-world questions. FinQA (Chen et al. 2021c) and AITQA (Katsis et al. 2022) lay emphasis on numeric-focused queries. These datasets predominantly target financial tables, suggesting complex reasoning challenges that necessitate models to not only interpret but also to compute and extract nuanced information precisely. Further diversifying the landscape, datasets such as WikiSQL (Zhong, Xiong, and Socher 2017), Spider (Yu et al. 2018), and Bird (Li et al. 2023a) introduce logical expressions as supervisory signals to train Table QA models, discreting reasoning capabilities through logic-based problem-solving. Despite the significant advancements made by LLMs in TableQA (Li et al. 2022b; Singha et al. 2023; Li et al. 2023b; Lei et al. 2023; He et al. 2023), there is still a critical need for benchmarks that reflect the reasoning complexity encountered in real-world tabular data scenarios. TableBench, a comprehensive and complex benchmark, incorporates real-world complexities into its evaluation scenarios, effectively addressing the limitations of existing benchmarks

# Conclusion

In this work, we introduce TableBench, a comprehensive and complex benchmark designed to evaluate a broad spectrum of tabular skills. It encompasses 886 questionanswer pairs across 18 distinct capabilities, significantly contributing to bridging the gap between academic benchmarks and real-world applications. We evaluate $^ { 3 0 + }$ models with various reasoning methods on TableBench and provide a training set TableInstruct that enables TABLELLM to achieve performance comparable to ChatGPT. Despite these advancements, even the most advanced model, GPT4, still lags significantly behind human performance on TableBench, underscoring the challenges of tabular tasks in real-world applications.

# Limitations

We acknowledge the following limitations of this study: (1) This paper mainly focuses on the reasoning complexity of table questions, which does not extensively explore the inherent complexities of the tables themselves. (2) Tabular data in image formats, which are also prevalent in real-world applications, are not discussed in this paper.