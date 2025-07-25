# Hierarchical Context Pruning: Optimizing Real-World Code Completion with Repository-Level Pretrained Code LLMs

Lei Zhang1,2, 3, Yunshui ${ \bf L i } ^ { 1 , 2 }$ , Jiaming ${ \bf L i } ^ { 1 , 2 }$ , Xiaobo $\mathbf { X _ { i a } } ^ { 4 , 5 }$ , Jiaxi Yang1,2 Run Luo1,2, Minzheng Wang2,7, Longze Chen1,2, Junhao $\mathbf { L i u } ^ { 6 }$ , Qiang ${ \bf Q } { \bf u } ^ { 1 }$ , Min Yang1,3\*

1Shenzhen Key Laboratory for High Performance Data Mining, Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences 2University of Chinese Academy of Sciences 3Key Laboratory of Intelligent Education Technology and Application of Zhejiang Province, Zhejiang Normal University 4School of Computing, National University of Singapore 5University of Science and Technology of China 6University of California, Irvine 7MAIS, Institute of Automation, Chinese Academy of Sciences {lei.zhang2, min.yang}@siat.ac.cn

# Abstract

Some of the latest released Code Large Language Models (Code LLMs) have been trained on repository-level code data, enabling them to perceive repository structures and utilize cross-file code information. This capability allows us to directly concatenate the content of repository code files in prompts to achieve repository-level code completion. However, in real development scenarios, directly concatenating all code repository files in a prompt can easily exceed the context window of Code LLMs, leading to a significant decline in completion performance. Additionally, overly long prompts can increase completion latency, negatively impacting the user experience. In this study, we conducted extensive experiments, including completion error analysis, topology dependency analysis, and cross-file content analysis, to investigate the factors affecting repository-level code completion. Based on the conclusions drawn from these preliminary experiments, we proposed a strategy called Hierarchical Context Pruning (HCP) to construct high-quality completion prompts. We applied the HCP to six Code LLMs and evaluated them on the CrossCodeEval dataset. The experimental results showed that, compared to previous methods, the prompts constructed using our HCP strategy achieved higher completion accuracy on five out of six Code LLMs. Additionally, the HCP managed to keep the prompt length around $8 \mathbf { k }$ tokens (whereas the full repository code is approximately 50k tokens), significantly improving completion throughput. Our code and data will be publicly available.

# Code — https://github.com/Hambaobao/HCP-Coder Extended version — https://arxiv.org/abs/2406.18294

# Introduction

Code completion tools powered by Code Large Language Models (Code LLMs) (Chen et al. 2021; Nijkamp et al. 2023b; Li et al. 2023; Fried et al. 2023; Allal et al. 2023), such as GitHub Copilot, have become integral to daily development workflows, significantly boosting developer productivity. As research on Code LLMs continues to advance (Bavarian et al. 2022; Sun et al. 2024), a new generation of models (Guo et al. 2024; Lozhkov et al. 2024; Team et al. 2024) trained on repository-level code data (RepoCode LLMs) has emerged. These models address the shortcomings of earlier file-level models, which struggled to comprehend repository structures and integrate code across multiple files during completion tasks. However, in realworld development scenarios, directly concatenating an entire code repository often exceeds the context window size of these Repo-Code LLMs, resulting in significant performance degradation and increased inference latency. Thus, effectively harnessing the capabilities of Repo-Code LLMs to integrate cross-file information and construct high-quality completion prompts within the model’s context window remains an open challenge for further research.

In this study, we first evaluated six Repo-Code LLMs on the CrossCodeEval (Ding et al. 2023) benchmark by constructing completion prompts through randomly concatenating all repository code. We conducted a detailed analysis and categorization of the erroneous completions produced by these models. The analysis revealed that at least $30 \%$ of the errors were caused by cross-file information issues, highlighting the importance of constructing high-quality repository-level completion prompts. Next, considering the causal architecture characteristics of Code LLMs, we performed a topological sort of code files based on their dependency relationships and conducted topological dependency analysis experiments. The results showed that maintaining the topological dependency order of code files within the prompt significantly improved the accuracy of code completions. Finally, we analyzed cross-file code content through experiments and found that global context information in cross-file code can be completely removed without affecting completion accuracy. Even when the specific implementations of all functions and class methods in cross-file code are removed, there is no significant reduction in completion accuracy.

Based on the results of these preliminary experiments, we proposed a strategy named Hierarchical Context Pruning (HCP) to construct high-quality completion prompts. The HCP models the code repository at the function level, retaining the topological dependencies between files while eliminating a large amount of irrelevant code content. In our experiments, the HCP successfully reduced the input from over 50,000 tokens to approximately 8,000 tokens, and significantly enhanced the accuracy of completions.

In summary, our contributions are threefold:

• Pioneering Studies Based on Repository-Aware Code LLMs: To the best of our knowledge, our research is pioneering in exploring how to develop high-quality completion prompts by leveraging the repository structural awareness of Repo-Code LLMs.   
• Thorough Preliminary Experiments: In this study, we first conducted extensive preliminary experiments, including baseline evaluation, completion error analysis, topological dependency analysis, and cross-file content analysis. The conclusions from these preliminary experiments collectively form the rationale behind the HCP.   
• Effective Method and Significant Improvements: Our proposed HCP strategy achieved better results on five out of six Code LLMs compared to previous baseline methods. These experimental results demonstrate the effectiveness of the HCP strategy. We will make our code and data publicly available.

# Experiments Setup Dataset & Evaluation Metrics

To assess the code completion performance of Code LLMs in real development scenarios, we utilized CrossCodeEval (Ding et al. 2023) as the evaluation dataset. The CrossCodeEval (Ding et al. 2023) benchmark provides test cases that require the use of cross-file code information for completion. Without loss of generality, in this study, we have chosen Python language as the primary language for our research.

We used the original data from CrossCodeEval, retaining the original repository structure. For each test case, we first identified the file for completion and the cursor’s position (the line and column where the completion occurs). We then removed the code after the cursor in that line to form authentic completion test cases. Ultimately, we obtained 2,655 real-world completion tests. Following the CrossCodeEval evaluation protocol, we evaluated the completion results using two metrics: Exact Match (EM) and Edit Similarity (ES). The results based on the Identifier Match (Identifier EM/Identifier F1) metrics can be found in the appendix.

# Models & Prompt Templates

We selected three series of Repo-Code LLMs for investigation: DeepSeek-Coder (Guo et al. 2024), Starcoder-2 (Lozhkov et al. 2024), and CodeGemma (Team et al. 2024). The specific prompt templates used by these Repo-Code LLMs are presented in the appendix of the extended version.

# Hardware & Hyperparameters

We conduct all the expiriments on NVIDIA A100 GPUs. We employ greedy decoding strategy for all the models, and set max new tokens to 32. The model max length of DeepseekCoder, Starcoder2 and CodeGemma is set to 16, 352, 16, 352 and 8, 160, respectively. All the prompts longer than the model max length are truncated from the left.

# Preliminary Studies

# Baseline Evaluation

Infile Only We initially evaluated the model’s completion ability using only information from the current file, with results presented in Table 1 under the Infile-Only row. The completion results are less than satisfactory. Even the bestperforming model achieved an accuracy of only about $30 \%$ .

Random-All Additionally, based on the pre-training data format of these Code LLMs, we constructed completion prompts by randomly concatenating all code files from the repository (truncating from the left to fit within the model’s context window). The evaluation results are shown in Table 1 under the Random-All row. The experimental results indicate that including information from other files in the prompt significantly improved completion accuracy (with the exception of DScoder-1.3B). The performance of the DScoder1.3B model progressively declines when the prompt includes multiple files, as can be seen in the experimental results in Appendix. We suspect this is likely due to the DScoder-1.3B model not being trained with a multi-file data format, rather than a reduction in the number of parameters.

RAG-Based Methods We also evaluated other retrievalaugmented generation (RAG) methods, including RAGBM25 (using BM25 as the relevance metric), RAG-OpenAI (using OpenAI text embedding similarity as the relevance metric), ReAcc (Lu et al. 2022), DraCo (Cheng, Wu, and Hu 2024), and Repofuse (Liang et al. 2024a). The experimental results show that, for these Repo-Code LLMs, the completion accuracy of prompts constructed by simply concatenating repository code files randomly is comparable to that of these RAG-based methods. This highlights the potential of Repo-Code LLMs in repository-level code completion tasks.

# Completion Error Analysis

To further investigate the issues of repository-level pretrained Code LLMs in real-world completion tasks, we sampled 200 error examples from each model’s Random-All evaluation results for error analysis. Ultimately, we categorized the issues present in these models into eight classes: Parameter Value Error, Non-existent Method Call, Improper Method Invocation, Missing Method Invocation, Redundant Content Generation, Partial Content Missing, Incorrect Content Generation, and Exact Match Error. We shows the error distribution statistics for six Repo-Code LLMs in the appendix of the extended version. We also provide examples of each type of error along with corresponding error analysis in the appendix.

<html><body><table><tr><td rowspan="2">XF-Context</td><td colspan="10">Baseline Evaluation</td></tr><tr><td colspan="2">DScoder-1.3B</td><td colspan="2">DScoder-6.7B</td><td colspan="2">Starcoder2-3B</td><td colspan="2">Starcoder2-7B</td><td colspan="2">CodeGemma-2B CodeGemma-7B</td></tr><tr><td></td><td>EM</td><td>ES</td><td>EM</td><td>ES EM</td><td>ES</td><td>EM</td><td>ES</td><td>EM</td><td>ES</td><td>EM ES</td></tr><tr><td>Infile-Only</td><td>16.72</td><td>56.58</td><td>28.14</td><td>68.36 21.92</td><td>61.49</td><td>22.98</td><td>63.58</td><td>20.64</td><td>56.26</td><td>30.58 70.36</td></tr><tr><td>Random-All</td><td>6.18</td><td>46.19</td><td>33.94 70.98</td><td>28.32</td><td>66.87</td><td>31.45</td><td>69.09</td><td>26.93</td><td>62.13 36.69</td><td>74.42</td></tr><tr><td>RAG-BM25</td><td>17.28</td><td>58.18</td><td>32.65 71.78</td><td>24.45</td><td>63.84</td><td>26.26</td><td>65.32</td><td>22.89</td><td>57.73 32.89</td><td>70.81</td></tr><tr><td>RAG-OpenAI</td><td>17.56</td><td>57.65</td><td>31.93 71.13</td><td>25.67</td><td>64.39</td><td>26.72</td><td>65.90</td><td>22.74</td><td>57.12 36.06</td><td>74.19</td></tr><tr><td>ReACC</td><td>16.08</td><td>55.51</td><td>32.36 70.60</td><td>24.79</td><td>63.59</td><td>26.54</td><td>64.39</td><td>22.80</td><td>56.56 35.67</td><td>73.47</td></tr><tr><td>DraCo</td><td>17.07</td><td>56.20</td><td>37.06 71.30</td><td>29.83</td><td>65.50</td><td>31.71</td><td>67.77</td><td>20.23</td><td>54.33 40.79</td><td>73.36</td></tr><tr><td>RepoFuse</td><td>22.59</td><td>68.85</td><td>27.92</td><td>73.09</td><td>-</td><td>-</td><td></td><td></td><td></td><td></td></tr></table></body></html>

![](images/ec0b1c4053c7344a89cfcfd8fe3c81a72886987bd223966397ea9b06e339a6f7.jpg)  
Table 1: The completion results of the baseline methods. EM denotes Exact Match, and ES denotes Edit Similarity.   
Figure 1: The distribution of tokenized prompt lengths in the CrossCodeEval benchmark. The $\mathbf { \boldsymbol { x } }$ -axis represents the dependent level, and the y-axis represents the number of tokens. denotes the median value of the tokenized prompt length. denotes the average value of the tokenized prompt length.

# Topological Dependency Analysis

Definition 1. (Dependency Level) Let $F$ denote a set of files in a code repository, and let $f \in F$ represent a specific file. We define the dependency levels as follows:

$$
\begin{array} { r l } & { I ( f ) = \{ g \mid g \mathrm { i s ~ i m p o r t e d ~ b y ~ } f \} } \\ & { D _ { 0 } ( f ) = \{ f \} } \\ & { D _ { i + 1 } ( f ) = D _ { i } ( f ) \cup I ( D _ { i } ( f ) ) } \end{array}
$$

We first identified the file requiring completion, then extracted all the import statements from the file with TreeSitter1, and used a breadth-first search (BFS) method to progressively add dependent files.

Figure 1 illustrates the growth in the number of dependent files (calculated by the length of the tokenized prompt) as the number of dependency layers increases. We used median and average as statistical measures and found that in the vast majority of cases, the number of dependent files for a single file increases slowly after reaching four layers of dependencies. This suggests that using four layers of dependencies is sufficient to cover most scenarios. We further define:

$$
D _ { \infty } ( f ) = D _ { 4 } ( f ) \cup \{ F \setminus D _ { 4 } ( f ) \}
$$

to represent the prompt including all files in the repository.

In Table 2, the D-level rows show the results of completion using cross-file information with different dependency levels. The results indicate that although the maximum dependency depth of most files reaches 4 levels, only the information provided by $D _ { 1 } ( f )$ files is the most useful. Furthermore, the effectiveness of using $D _ { \infty } ( f )$ surpasses that of Random-All, indicating that besides ${ \dot { D } } _ { 1 } ( f )$ files, there are many other useful files within the repository.

# Cross-File Content Analysis

Definition 2. (Pruning Level) We define the pruning levels into three categories:

• P-Level 0: No pruning is applied to the file content. • P-Level 1: All global context content is removed from the file. • P-Level 2: All global context content, function bodies and class method bodies are removed from the file.

Table 3 presents the results of completion using cross-file information with different pruning levels. We can see that the results of $P$ -level:1 outperform those of $P$ -level:0, indicating that the Global Context information from cross-file content has minimal impact on the completion of the current file. Additionally, the results of $P$ -level:2 are only slightly worse than those of $D _ { \infty } ( f )$ , and when combined with the information from $D _ { 1 } ( f )$ , they are almost equivalent to the results of $D _ { \infty } ( f )$ . This suggests that the specific implementations of most cross-file functions have minimal impact on the completion of the current file, and retaining only the function header information is sufficient.

Table 2: Comparison of completion results using different context dependency levels across six models. All the prompts is truncated to the max context window of the Code LLMs from the left. $\infty$ denotes the prompt including all files in the repository   

<html><body><table><tr><td rowspan="2">XF-Context</td><td colspan="10">Topological Dependency Analysis</td></tr><tr><td>DScoder-1.3B</td><td></td><td>DScoder-6.7B</td><td></td><td>Starcoder2-3B</td><td></td><td>Starcoder2-7B</td><td></td><td>CodeGemma-2B</td><td>CodeGemma-7B</td></tr><tr><td></td><td>EM</td><td>ES</td><td>EM</td><td>ES</td><td>EM</td><td>ES EM</td><td>ES</td><td>EM</td><td>ES</td><td>EM ES</td></tr><tr><td>D-Level: 1</td><td>15.44</td><td>55.03</td><td>33.03</td><td>70.77</td><td>26.18 64.15</td><td>28.51</td><td>66.91</td><td>24.37</td><td>58.79</td><td>34.65 73.01</td></tr><tr><td>D-Level: 2</td><td>13.63</td><td>53.45</td><td>33.56</td><td>70.74 26.70</td><td>64.58</td><td>29.45</td><td>67.03</td><td>25.31</td><td>59.27 35.67</td><td>73.26</td></tr><tr><td>D-Level: 3</td><td>13.26</td><td>53.17</td><td>33.07 70.51</td><td>26.82</td><td>64.56</td><td>29.23</td><td>67.01</td><td>25.35</td><td>59.30 35.93</td><td>73.34</td></tr><tr><td>D-Level: 4</td><td>13.37</td><td>53.20</td><td>33.22 70.57</td><td>26.59</td><td>64.46</td><td>29.53</td><td>67.07</td><td>25.54</td><td>59.42 36.12</td><td>73.54</td></tr><tr><td>D-Level: 00</td><td>5.76</td><td>46.22</td><td>35.29</td><td>71.51 30.43</td><td>67.34</td><td>33.03</td><td>69.57</td><td>29.08</td><td>62.91 39.32</td><td>75.35</td></tr></table></body></html>

<html><body><table><tr><td rowspan="2">XF-Context</td><td colspan="10">Cross-File Content Analysis</td></tr><tr><td colspan="2">DScoder-1.3B</td><td colspan="2">DScoder-6.7B</td><td colspan="2">Starcoder2-3B</td><td colspan="2">Starcoder2-7B</td><td colspan="2">CodeGemma-2B</td><td colspan="2">CodeGemma-7B</td></tr><tr><td></td><td>EM</td><td>ES</td><td>EM</td><td>ES</td><td>EM</td><td>ES</td><td>EM</td><td>ES</td><td>EM</td><td>ES</td><td>EM</td><td>ES</td></tr><tr><td>P-Level: 0</td><td>6.18</td><td>46.19</td><td>33.94</td><td>70.98</td><td>28.32</td><td>66.87</td><td>31.45</td><td>69.09</td><td>26.93</td><td>62.13</td><td>36.69</td><td>74.42</td></tr><tr><td>P-Level: 1</td><td>6.55</td><td>46.58</td><td>36.20</td><td>71.90</td><td>30.73</td><td>67.97</td><td>34.43</td><td>70.65</td><td>29.30</td><td>63.46</td><td>39.55</td><td>75.70</td></tr><tr><td>P-Level: 2</td><td>9.83</td><td>49.63</td><td>34.73</td><td>70.89</td><td>30.02</td><td>66.41</td><td>31.26</td><td>68.24</td><td>27.34</td><td>61.13</td><td>38.31</td><td>74.32</td></tr><tr><td>+ D-level:1</td><td>9.45</td><td>49.44</td><td>36.87</td><td>72.14</td><td>29.91</td><td>66.96</td><td>32.62</td><td>69.11</td><td>28.93</td><td>62.03</td><td>39.17</td><td>75.16</td></tr><tr><td>+ D-level:2</td><td>8.70</td><td>48.61</td><td>36.38</td><td>71.66</td><td>29.64</td><td>66.99</td><td>32.96</td><td>69.13</td><td>28.44</td><td>61.76</td><td>39.06</td><td>74.91</td></tr></table></body></html>

Table 3: The results of completion using cross-file information with different pruning levels. $+ \ D$ -level:x denotes the mode uses the cross-file information with dependency level x.

# Hierarchical Context Pruning

Based on the analysis results concerning the dependencies and content of the files, we attempt to construct a hierarchical context prompt based on the importance and relevance of the repository content. This approach aims to enhance the accuracy of code completion models while effectively reducing the length of the context. Figure 2 shows the specific process for constructing a hierarchical context prompt.

# Fine-grained Repository Modeling

In order to precisely control the content within the code repository, we employ Tree-Sitter to parse the files within the repository. We model the content using three types of nodes:

• Function Node: Represents a function or a class method within a code file. • Class Node: Represents a class in a code file, consisting of the class’s name, attributes, and Function Nodes.

• File Node: Represents a code file, comprising Nodes that represent the functions and classes within the file, along with global context information.

# Hierarchical Context

As shown in the top right of Figure 2, following the settings in Topological Dependency Analysis, we conduct a dependency analysis on the files in the repository. We perform a topological sort based on the dependency relationships, centering around the file currently being completed. According to the experimental results in Topological Dependency Analysis, only files at dependency level 1 significantly enhance completion accuracy. Therefore, we select files designated as $\bar { D _ { 1 } } ( f )$ to serve as dependency files. Ultimately, the files in the repository are categorized into three types: current file, dependency files, and other files. We will apply different strategies to optimize each type of file.

Current File. For the current file, any content within the file may be needed during completion, so we retain all content of the file and convert it into the Fill-in-the-middle (FIM) format.

Dependency Files. According to the experimental results in Cross-File Content Analysis, removing the global context across files does not affect the accuracy of completions. Therefore, for dependency files, we remove all global context from these files.

![](images/bd4b25bcc3f64f2bc2b5418aae8913780af0903cc15690d1c2e04d8d703da183.jpg)  
Figure 2: The framework of hierarchical context pruning for improving the performance of code large language models in real-world code completion tasks.

Other Files. We refer to files other than the current file and its direct dependency files, namely $\{ F \setminus D _ { 1 } ( f ) \} \setminus f \}$ , collectively as other files. For the content in other files, we remove all global context, and then we employ functionlevel sampling and pruning methods to optimize the content of these files.

# Function-level Sampling

In this study, we used OpenAI’s text-embedding $\mathbf { A P I } ^ { 2 }$ to embed each function (or class method) and query code snippet in the repository. We then used the pre-computed similarity of embeddings between the query and candidate functions (or class methods) as an indicator of relevance. We select the code from the current line of completion and the 10 lines before and after it as a query to find functions and class methods most relevant to the current completion content.

We implemented two sampling strategies (top- $\mathbf { \nabla } \cdot \mathbf { k }$ and topp) and designed distinct content pruning strategies for the functions (or class methods) sampled under each strategy, see Function-level Pruning.

# Function-level Pruning

According to the experimental results in Cross-File Content Analysis, the global context from all non-current files and most of the function bodies (or class method bodies) within the code repository can be pruned. Appropriately pruning low-relevance content can significantly reduce the length of the prompt input to the model.

Let $G$ denote the set of all functions and class methods in the repository, $F _ { k }$ represent the functions sampled using the top- $\mathbf { \nabla } \cdot \mathbf { k }$ strategy, and $F _ { p }$ represent the functions sampled using the top-p strategy:

$$
\begin{array} { l } { G _ { k } = \{ g \mid g \in \operatorname { T o p } _ { k } ( G ) \} } \\ { G _ { p } = \{ g \mid g \in \operatorname { T o p } _ { p } ( G ) \} } \end{array}
$$

where $G _ { k } \subseteq G _ { p }$ . Content from functions and class methods not within the set $G _ { k } \cup G _ { p }$ was completely pruned.

Top-k Context Pruning. For functions (or class methods) within the set $G _ { k }$ , we retained their entire content.

Top-p Context Pruning. For functions (or class methods) in the set $G _ { p }$ but not in $G _ { k }$ , we prune their implementations and retained only their function headers (or class method headers).

# File-level Relevance Ranking

Relevance Weighting Each function or class method in the repository is assigned a similarity score. We then apply different relevance weights to functions sampled using various sampling strategies, with highly relevant functions and class methods receiving higher weights. In this experiment, our specific setup is as follows:

Table 4: The results of completion using hierarchical context pruning with different top- $\mathbf { \nabla } \cdot \mathbf { k }$ values.   

<html><body><table><tr><td rowspan="2">XF-Context</td><td colspan="10">Hierarchical Context Pruning (Top-p: 1.0)</td></tr><tr><td colspan="2">DScoder-1.3B</td><td colspan="2">DScoder-6.7B</td><td colspan="2">Starcoder2-3B</td><td colspan="2">Starcoder2-7B</td><td colspan="2">CodeGemma-2B</td><td colspan="2">CodeGemma-7B</td></tr><tr><td></td><td>EM</td><td>ES</td><td>EM</td><td>ES</td><td>EM</td><td>ES</td><td>EM</td><td>ES</td><td>EM</td><td>ES</td><td>EM ES</td></tr><tr><td>Previous SOTA</td><td>22.59</td><td>68.85</td><td>37.06</td><td>73.09</td><td>29.83</td><td>66.87</td><td>31.71</td><td>69.09</td><td>26.93</td><td>62.13 40.79</td><td>74.42</td></tr><tr><td>Top-k: 0</td><td>9.45</td><td>49.44</td><td>36.87</td><td>72.14</td><td>29.91</td><td>66.96</td><td>32.62</td><td>69.11</td><td>28.93 62.03</td><td>39.17</td><td>75.16</td></tr><tr><td>Top-k: 5</td><td>9.64</td><td>49.78</td><td>39.74</td><td>73.90</td><td>32.68</td><td>69.05</td><td>35.76</td><td>71.41</td><td>31.26 63.74</td><td>42.44</td><td>76.95</td></tr><tr><td>Top-k: 10</td><td>9.91</td><td>49.85</td><td>40.30</td><td>74.56</td><td>34.15</td><td>69.37</td><td>36.47</td><td>71.50</td><td>31.82 64.34</td><td>42.63</td><td>77.35</td></tr></table></body></html>

<html><body><table><tr><td rowspan="2">XF-Context</td><td colspan="10">Hierarchical Context Pruning (Top-k: 5)</td></tr><tr><td colspan="2">DScoder-1.3B</td><td colspan="2">DScoder-6.7B</td><td colspan="2">Starcoder2-3B</td><td colspan="2">Starcoder2-7B</td><td colspan="2">CodeGemma-2B</td><td colspan="2">CodeGemma-7B</td></tr><tr><td></td><td>EM</td><td>ES</td><td>EM</td><td>ES</td><td>EM</td><td>ES</td><td>EM</td><td>ES</td><td>EM</td><td>ES</td><td>EM ES</td></tr><tr><td>Previous SOTA</td><td>22.59</td><td>68.85</td><td>37.06</td><td>73.09</td><td>29.83</td><td>66.87</td><td>31.71</td><td>69.09 26.93</td><td>62.13</td><td>40.79</td><td>74.42</td></tr><tr><td>Top-p: 0.1</td><td>14.27</td><td>53.94</td><td>37.85</td><td>73.11</td><td>32.99</td><td>68.75</td><td>34.16</td><td>70.43 29.19</td><td>62.09</td><td>40.98</td><td>76.26</td></tr><tr><td>Top-p: 0.2</td><td>13.52</td><td>53.20</td><td>38.04</td><td>73.13</td><td>33.15</td><td>68.59</td><td>34.84</td><td>70.40 29.72</td><td>62.32</td><td>40.94</td><td>76.25</td></tr><tr><td>Top-p: 0.3</td><td>12.88</td><td>52.60</td><td>38.49</td><td>73.19</td><td>32.84</td><td>68.31</td><td>35.22</td><td>70.64 30.13</td><td>62.77</td><td>41.21</td><td>76.20</td></tr></table></body></html>

Table 5: The results of completion using hierarchical context pruning with different top-p values.

$$
W ( g ) = { \left\{ \begin{array} { l l } { 1 . 0 , } & { \forall g \in G _ { k } } \\ { 0 . 5 , } & { \forall g \in G _ { p } \setminus G _ { k } } \\ { 0 . 0 , } & { \forall g \in G \setminus ( G _ { k } \cup G _ { p } ) } \end{array} \right. }
$$

where $G _ { k }$ and $G _ { p }$ represent the functions with the highest relevance scores sampled using the top- $\mathbf { \nabla } \cdot \mathbf { k }$ and top-p strategies, respectively.

Class-level Relevance The similarity of a class is defined as the weighted sum of its class methods:

$$
S ( C ) = \sum _ { g \in C } W ( g ) * S ( g )
$$

where, $C$ represents the class, $g$ represents the class method, and $S ( g )$ represents the similarity score of the class method.

File-level Relevance The similarity of a file is defined as the weighted sum of its functions and classes:

$$
S ( f ) = \sum _ { g \in \mathcal { G } } W ( g ) * S ( g ) + \sum _ { c \in \mathcal { C } } S ( c )
$$

where, $\mathcal { G }$ and $\mathcal { C }$ represent the set of functions and classes in the file, respectively.

Finally, we sort the files at the file-level according to the relevance score to determine their relative positions in the prompt.

# Experimental Results

Top-k and Top-p Analysis We initially fixed top-p at 1.0 and tested the impact of different top-k values on completion accuracy. Table 4 presents some of the experimental results, and we provides a more comprehensive results in the appendix of the extended version. We observed that increasing the top- $\mathbf { \nabla } \cdot \mathbf { k }$ value beyond 5 did not result in significant improvements in accuracy. Therefore, we conclude that a top- $\mathbf { \nabla } \cdot \mathbf { k }$ value of 5 is sufficient.

We further fixed the top- $\mathbf { \nabla } \cdot \mathbf { k }$ value at 5 and tested the impact of varying top-p values (ranging from 0.1 to 0.9) on completion accuracy. Partial experimental results are presented in Table 5, with more comprehensive results available in the appendix of the extended version. Our observations indicate that increasing the top-p value enhances completion accuracy; however, beyond a top-p value of 0.3, the improvement in accuracy slows considerably. Thus, we consider 0.3 to be a reasonable value.

Comparison with Random-All Figure 3 visually compares the Hierarchical Context Pruning (HCP) strategy (top$\mathrm { k } { = } 5$ , top- $\cdot { \mathrm { p } } { = } 0 . 3 { \mathrm { } } )$ ) with the method of randomly concatenating all repository code files across three dimensions: completion accuracy, throughput rate, and input length. The visualization shows that, compared to random concatenation, HCP significantly reduces input length (enhancing throughput) while improving the model’s completion accuracy.

# Related Work

# Code Large Language Models

General Code LLMs Building on the success of large language models, researchers have explored further pretraining these models on code data to enhance their performance on code-related tasks (Chen et al. 2021; Austin et al. 2021; Cassano et al. 2022). This has led to the development and release of several powerful code-focused large language models, such as the CodeX series (Chen et al.

Completion Accuracy Throughput Tokenized Prompt Length Random-All HCP Random-All HCP 1 Random-All HCP   
45.0 25.0 1581000 .0 = m 2   
36.0 20 14400 105 1702800 10,669 11,184 11,689 9,604 10,144 8,438 9,032 7,839 7,186   
18.0   
9.0 5 3600 0.0 0.0 DSC-13B DSC-6.7B SC2-3BSC2-7BCG-2BCG-7B 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Model Model Top-p

2021), CodeGen series (Nijkamp et al. 2023c), CodeT $^ { \cdot } 5 +$ (Wang et al. 2023), and Lemur (Xu et al. 2023). These models have demonstrated strong capabilities in handling coderelated tasks.

Infilling Code LLMs Infilling scenarios make up the majority of code completion tasks in the real world. Bavarian et al. (2022) demonstrated that pre-training Code LLMs with a certain proportion of fill-in-the-middle format code data enables these models to fill in missing code based on the surrounding context, without compromising their original leftto-right generation capabilities. Building on these findings, many subsequent Code LLMs, such as Incoder (Fried et al. 2023), SantaCoder (Allal et al. 2023), CodeGen2 (Nijkamp et al. 2023a), CodeLLama (Rozie\`re et al. 2024), Starcoder (Li et al. 2023; Lozhkov et al. 2024), DeepSeek-Coder (Guo et al. 2024) and StableCoder (Pinnaparaju et al. 2024), have been developed with enhanced infilling capabilities.

# Repository-Level Code Benchmarks

To better assess the repository-level code completion capabilities of Code LLMs, researchers have developed several comprehensive benchmarks. CrossCodeEval (Ding et al. 2023) introduced a multilingual cross-file code completion benchmark that underscores the importance of understanding cross-file context in real-world software development environments. RepoBench (Liu, Xu, and McAuley 2023) is a new benchmark specifically designed for evaluating code auto-completion systems at the repository level. It includes three interconnected evaluation tasks, code snippet retrieval, code completion, and end-to-end processing, emphasizing the critical role of multi-file context. Recently, additional repository-level code completion benchmarks have emerged, such as CoderEval (Zhang et al. 2024) and EvoCodeBench (Li et al. 2024).

# Repo-level Code Completion

Due to the lack of repository structure awareness in earlier Code LLMs, most research efforts (Shrivastava, Larochelle, and Tarlow 2023), such as ReAcc (Lu et al. 2022), RepoCoder (Zhang et al. 2023), RepoHyper (Phan et al. 2024) and RepoFuse (Liang et al. 2024b), adopted a retrievalaugmented generation approach. This approach involves retrieving relevant code snippets from the repository and concatenating them as comments at the beginning of the current file to leverage cross-file code information. Recent work has further considered additional factors to improve the repository-level completion accuracy of Code LLMs. For example, CoCoGen (Bi et al. 2024) incorporated compiler feedback, while DraCo (Cheng, Wu, and $\mathrm { H u } 2 0 2 4 )$ took data flow factors into account. However, most of these efforts have not utilized the Fill-in-the-Middle capability of Code LLMs. Additionally, since Repo-Code LLMs are relatively new, these works did not take advantage of the new capabilities offered by Repo-Code LLMs in their design. To the best of our knowledge, our research is pioneering in exploring how to develop high-quality completion prompts by leveraging the repository-awareness of Repo-Code LLMs.

# Conclusion

In this study, we conducted extensive experiments on six newly released Code LLMs that were trained on repositorylevel code data, and proposed an effective method for constructing high-quality repository-level code completion prompts. Our preliminary experimental results showed that maintaining the topological dependency order of code files in the prompt can improve completion accuracy, while removing global cross-file context and specific function implementations does not significantly reduce accuracy. Based on these findings, we proposed the Hierarchical Context Pruning (HCP) method to construct efficient repositorylevel code completion prompts. Compared to previous stateof-the-art methods, our HCP approach achieved better results on five out of six Code LLMs and effectively controlled input length, demonstrating the effectiveness of our method.