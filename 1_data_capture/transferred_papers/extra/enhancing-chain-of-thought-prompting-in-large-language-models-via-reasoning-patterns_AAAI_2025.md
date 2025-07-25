# Enhancing Chain of Thought Prompting in Large Language Models via Reasoning Patterns

Yufeng Zhang1,2, Xuepeng Wang1,3,\*, Lingxiang $\mathbf { W _ { u } } ^ { 1 , 3 , * }$ , Jinqiao Wang1,2,3

1Institute of Automation, Chinese Academy of Sciences, Beijing, China 2School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, China 3Wuhan AI Research, Wuhan, China {yufeng.zhang, xuepeng.wang} $@$ ia.ac.cn, {lingxiang.wu, jqwang}@nlpr.ia.ac.cn

# Abstract

Chain of Thought (CoT) prompting can encourage language models to engage in multi-step logical reasoning. The quality of the provided demonstrations significantly influences the success of downstream inference tasks. Current unsupervised CoT methods primarily select examples based on the semantics of the questions, which can introduce noise and lack interpretability. In this paper, we propose leveraging reasoning patterns to enhance CoT prompting effectiveness. Reasoning patterns represent the process by which language models arrive at their final results. By utilizing prior knowledge and prompt-based methods from large models, we first construct task-specific pattern sets. We then select diverse demonstrations based on different reasoning patterns. This approach not only mitigates the impact of noise but also provides explicit interpretability to help us understand the mechanisms of CoT. Extensive experiments demonstrate that our method is more robust and consistently leads to improvements across various reasoning tasks.

Q: Mark's father gave him \$85.Mark Q: Nancy has saved 4900 bought 10 books,each of which cost cents from selling \$5.How much money does Mark have lemonade. How many left? dollars does Nancy have? A: Let's think step by step.If Mark has A: Let's think step by step. \$85 and he buys 10 books at \$5 each, Nancy saved 490o cents, then he will spend a total of \$50 (10 x which means she saved \$5).So,after buying the books,Mark 4900 / 100 = 49 dollars. will have $\$ 35$ $\$ 85 -850$ left.The The answer is 49.   
answer is 35   
prompt 1 prompt2 Q: The value of a sport utility vehicle this year is 16,ooo dollars，which is O.8 of what its value was last year. How much is the value of the vehicle lastyear?   
A: Let's think step by step. question ！ ↓   
0.0   
The problem tells us that the value The problem tells us that ofthe sportutilityvehicle thisyear is the value of the sport utility 16,000 dollars,which is O.8 of what vehicle this year is \$16,000, it was last year.So,if we multiply which is O.8 times its value the value of the vehicle this year by last year.This means that 0.8,we get the value of the vehicle the value last year is lastyear $\mathbf { \tau } = \mathbf { \tau }$ 16,000 x 0.8 $\mathbf { \Psi } = \mathbf { \Psi }$ 12,800 \$16,000/0.8 $\mathbf { \tau } = \mathbf { \tau }$ \$20,000. dollars.The answer is 12,800. Theansweris\$20,000.

# Introduction

Large Language Models (LLMs) have demonstrated exceptional performance across a wide range of language tasks. In general question-answering tasks (Kwiatkowski et al. 2019), LLMs hold a distinct advantage over other language models due to their robust writing capabilities. However, when it comes to more advanced tasks such as logical reasoning, mathematical computation, and symbolic reasoning, LLMs often fall short (Qiao et al. 2023; Huang and Chang 2023).

One effective approach to addressing these challenges is Chain of Thought (CoT) prompting (Wei et al. 2022b). By providing several demonstration examples that include a problem, intermediate reasoning steps, and an answer, CoT prompting serves as a contextual guide for downstream tasks. This approach encourages LLMs to generate multistep logical reasoning, thereby maximizing the likelihood of producing more plausible answers. The advantage of this method lies in its simplicity and efficiency; unlike finetuning, it does not require extensive gradient updates or alter the model’s inherent capabilities. Instead, it acts as an external augmentation of knowledge. For different reasoning tasks, we can route the model to the appropriate context, and then easily switch the demonstration sets to activate the relevant knowledge and abilities in the corresponding domain.

However, we argue that existing unsupervised CoT prompting methods have two major shortcomings. First, there remains a significant gap between the selected demonstration sets and the reasoning targets. Although extensive research (Zhang et al. 2023; Levy, Bogin, and Berant 2023; Yang et al. 2023; Shum, Diao, and Zhang 2023a) has explored ways to provide CoT demonstrations to enhance LLMs’ reasoning capabilities, these methods largely rely on the semantic features of the problem or the answer. Such features introduce irrelevant noise on a global scale, which can obscure the logical information needed for reasoning. Consequently, the constructed demonstration sets do not effectively represent the domain-specific logical knowledge, and struggle to adequately trigger correct reasoning in LLMs. Second, some demonstration selection methods lack interpretability and scalability. These methods are primarily based on heuristic design (Wang et al. 2022; Zheng et al. 2023) or leverage the model itself to generate additional demonstrations (Zhong et al. 2024; Yasunaga et al. 2024). The demonstration sets chosen through these means inherently lack clear explanations, making it challenging to assess their effectiveness or determine the direction for further optimization. This limitation can be particularly problematic in scenarios where interpretability is crucial.

To better select a demonstration subset for a reasoning task, we believe that considering the logical patterns of reasoning is essential. Inspired by the work of (Min et al. 2022) and (Madaan, Hermann, and Yazdanbakhsh 2023), we observe that LLMs are more influenced by the templates and patterns in the context than by the correctness of the demonstrations themselves. Building on this insight, we investigate the selection of demonstrations based on Reasoning Patterns. This approach offers a dual benefit. First, it helps to eliminate bias introduced by irrelevant information, thereby reducing the gap between the demonstration set and the reasoning task. Second, it provides explicit interpretability, allowing us to gain a deeper understanding of how CoT prompting functions. This interpretability can also serve as a clue for attribution analysis and visualization.

In this work, we propose Pattern- $\mathbf { C o T } ^ { 1 }$ , a CoT demonstration selection method based on reasoning patterns. Unlike previous approaches that focus on overall semantics, our method targets finer-grained logical reasoning operations. For instance, in mathematical reasoning, addition and multiplication represent distinct operations, while multiple sequential operators may indicate more complex operational patterns, as shown in Figure 1. Inspired by recent studies (Yang et al. 2023), a diverse range of these patterns should be incorporated into CoT. Specifically, for a given reasoning task, we first obtain a set of seed demonstrations with rationale (intermediate reasoning steps). These examples can be sourced from the training set or generated using a zeroshot approach. We then obtain specific operation tokens tailored to different task types, which help us extract reasoning patterns from the rationales. Here, we incorporate prior knowledge and guide the LLMs in generating these operation tokens. Based on the extracted reasoning patterns, we apply clustering techniques to merge similar patterns and design metrics to automatically assess the number of demonstration categories. Finally, we select representative demonstrations from each category to enrich the diversity and construct context prompts for LLMs. Notably, by incorporating task-specific knowledge, our method improves interpretability and facilitates further scalability.

Our contributions can be summarized as follows:

• We introduce the use of diverse reasoning patterns to enhance CoT prompting and design a demonstration selection method to reduce the gap between the demonstration set and the task.   
• Our method strengthens the interpretability of CoT in unsupervised scenarios, and can be utilized for further attribution analysis.   
• Extensive experiments demonstrate that our method con

sistently enhances performance across multiple reasoning tasks and various models.

# Related Work

# Chain-of-Thought Prompting

Large language models have demonstrated significant ability in comprehending context and responding to prompts (Brown et al. 2020; Ouyang et al. 2022). Recent studies highlight that LLMs can achieve improved task completion without fine-tuning, particularly on reasoning tasks, when provided with few-shot demonstrations (Wei et al. 2022b). For instance, when presented with an example like $\boldsymbol { Q }$ : Mary has 9 yellow marbles. John has 3 yellow marbles. How many yellow marbles do they have in all? A: They have $9 + 3 = { \begin{array} { l } { } \end{array} }$ 12 yellow marbles. The answer is 12, LLMs are expected to emulate such a format, deconstruct the question, engage in multi-step reasoning, and refrain from generating random answers in subsequent tasks. This process is commonly referred to as chain-of-thought prompting or in-context learning (Wei et al. 2022a; Xie et al. 2022). However, implementing this practice often involves the manual design of prompts at a labour cost. Consequently, researchers are exploring more efficient example selection strategies to streamline this process.

# Demonstration Selection and Refinement

Several CoT studies are directed towards automating the generation of demonstrations, such as retrieval-based (Rubin, Herzig, and Berant 2022), zero-shot (Kojima et al. 2022), clustering-based (Zhang et al. 2023), and self-prompt (Shao et al. 2023; Yasunaga et al. 2024). However, many of these approaches encounter challenges in achieving performance comparable to Manual-CoT, primarily due to the absence of supervision in example selection. In another branch of research, efforts are focused on enhancing the quality of CoT demonstrations. They incorporate elements such as knowledge-infusion (Zhao et al. 2023; Weng et al. 2023; Li et al. 2024), self-consistency (Wang et al. 2023a), complexity-based (Fu et al. 2022), contrastive-based (Chia et al. 2023), and progressive-hint (Zheng et al. 2023). The primary goal of these strategies is to ensure that LLMs adhere to the correct prompt and avoid being misled.

# Role of In-Context Patterns

To understand the underlying mechanism of ICL, (Min et al. 2022) and (Madaan, Hermann, and Yazdanbakhsh 2023) employ counterfactual prompting methods. These methods involve substituting question-answer mapping, token distributions, answer patterns, and many other factors. Their findings consistently show that the correctness of examples is not the most crucial factor, but rather the distribution or pattern (e.g. equations, templates, sentence structure) of the examples. In this paper, we continue to uncover the power of CoT patterns and show how they can improve the reasoning process.

Dataset Downstream Task   
Q: Liam is 16 years old now.Two years ago Qi: Nancy has saved 4900 cents. Q1：..   
Q2:Melanie,Sally，andJessica each have.. A:.. saved4900/100=49 dollars... $\mathsf { Q } _ { 2 } \mathsf { : }$ ：Tomwasat the beach for5 days ..   
Qn: There were atotal of6 soccer gamesthis.. Az:.. a total of7×5=35 seashells. Question:   
LLMIO.O 工 Let's think step by step Thvauerastiitii Seed Demonstrations A Wwenicastyearowmuchis thevalueof the   
Qi:Liam is16 years old now.Two yearsago,   
Liam'sage was twice the age of Vince. How old   
is Vince now? K-Clustering Adaptive K xi LLMIO   
A: Let's think step by step.2 years ago,Liam's 。 1   
agewas twice theage of Vince.So,we can write Embeddings FinalAnswer   
！an equation based on this information:   
Liam.'wege 2 ytarst La $= 2$ x vine'sge years twice X = divide ÷ = + The problem tells us that the value of   
hisage 2 yearsago would have been $7 6 - 2 = 1 4$ - 香 the sport utility vehicle this year is   
Nqwt $1 4 = 2 1$ inctetisuarintgor Pattern Discovery TaskPatterns $\$ 16,000$ ,which is O.8 times its value   
solve for Vince'sage,we can divide both sides of last year.This means that the value   
theequation by2: $7 =$ Vince'sage2yearsago Prior Knowledge twice the age of last year is $\$ 16,000$ $0 . 8 = \$ 20,000$ -   
So,Vince is currently $7 + 2 = 9$ years old. divide both sides The answer is $\$ 20,000$ . Qn：. LLM Prompting 7+2=9 years old

# Methodology

We now explore the impact of diverse demonstration reasoning patterns on chain-of-thought prompting. According to (Min et al. 2022), the precision of demonstrations is not crucial when LLMs engage in ICL. Even if all the demonstrations provided are incorrect, it would only marginally impede performance. This aligns with the insight derived from Auto-CoT (Zhang et al. 2023): clustering zero-shot question-answer pairs without emphasizing accuracy can still yield valuable examples. Consequently, our focus shifts to a more nuanced factor - the underlying reasoning pattern that harbours more informative content (Madaan, Hermann, and Yazdanbakhsh 2023) - to evaluate its potential benefits for the CoT process. The entire process is summarized in Figure 2 and Algorithm 1.

# Seed Demonstration Collection

For a given task $Q = \{ q _ { 1 } , q _ { 2 } , . . . , q _ { N } \}$ with $N$ questions, we first need to obtain their rationales and answers $\{ q _ { i } , r _ { i } , a _ { i } \}$ that can be used as context for CoT prompting. For data from existing training sets, we can directly use the training data. However, in practical applications, complete training sets may not always be available. In such cases, we refer to methods like (Zhang et al. 2023; Shum, Diao, and Zhang 2023b) and leverage the zero-shot (Kojima et al. 2022) capabilities of LLMs to generate the corresponding rationales. It is important to note that we do not require the answers to be correct or labelled; our focus is on whether the generated rationales contain meaningful reasoning patterns.

# Pattern Discovery

Based on the rationale set $R a \ = \ \{ r _ { 1 } , r _ { 2 } , . . . , r _ { N } \}$ that we have obtained, we next identify the reasoning operations $T$ associated with the task. For tasks with a relatively limited action space, we can define reasoning operations using prior knowledge, as these operations represent the fundamental units of reasoning tasks. For example, in arithmetic problems, we refer to a glossary of possible operators from sources like Wikipedia2, including basic arithmetic operations, square roots, comparison symbols, etc. For tasks with less clearly defined operations, we adapt definitions from arithmetic problems to guide LLMs in generating the corresponding reasoning operations. We design the prompt as: ‘Similar to operators used in arithmetic such as $( + , - , ~ ^ { * } , / )$ , which operators do you think best represent the $I T A S K J !$ Example of [TASK]: ...’

For each rationale $r _ { i } \in R a$ , we extract the reasoning operation tokens or phrases $t _ { j } \in T$ to form its reasoning pattern:

$$
p _ { i } = f ( r _ { i } , T ) = \{ t _ { i 1 } , t _ { i 2 } , . . . , t _ { i j } \}
$$

where $f$ denotes the function used to extract the reasoning path. In this context, $p _ { i }$ represents how LLMs apply these operations step-by-step to reach the final result, and $t _ { i j }$ can repeated.

# Pattern Wise Demonstration Selection

Once we have identified the task-relevant patterns, we use them to select better demonstration sets. Following (Zhang et al. 2023), we cluster all the $p _ { i }$ patterns while preserving diversity. Although $p _ { i }$ is a simplified sequence of tokens, it still contains substantial semantic information that can be used to uncover underlying similarities. For instance, a sequence of addition operations is likely to be closer to a single addition operation than to a single multiplication operation. To leverage this, we use a language model to encode these patterns. We then apply the $k$ -means clustering algorithm to generate $k$ clusters and sample from each cluster:

$$
{ \widetilde { p } } _ { i } = { \mathrm { e n c o d e } } ( p _ { i } )
$$

Require: A set of task questions $Q$   
Ensure: Demonstration list $d = [ d _ { 1 } , d _ { 2 } , . . . , d _ { k } ]$   
1: Acquire operation token set $T$ with LLMs prompting or   
domain knowledge based on $Q$   
2: for $q _ { i } \in Q$ do   
3: Generate rationale $r _ { i }$ with Zero-Shot-CoT   
4: $p _ { i } = [ ]$   
5: for each token $t _ { i j } \in r _ { i }$ do   
6: if $t _ { i j } \in T$ then   
7: Update $p _ { i }$ with $t _ { i j }$   
8: end if   
9: end for   
10: ${ \widetilde { p } } _ { i } = \operatorname { e n c o d e } ( p _ { i } )$   
11: end for   
12: Select proper $k$   
13: Cluster all $[ \widetilde { p } _ { 1 } , \widetilde { p } _ { 2 } , . . . , \widetilde { p } _ { i } ]$ into $k$ clusters   
14: Sample $d = [ d _ { 1 } , d _ { 2 } , . . . , d _ { k } ]$ from each cluster   
15: return $d$

$$
c _ { 1 } , c _ { 2 } , . . . , c _ { k } = \mathrm { c l u s t e r } ( \widetilde { p } _ { 1 } , \widetilde { p } _ { 2 } , . . . , \widetilde { p } _ { i } )
$$

$$
d = \{ q _ { m } , r _ { m } , a _ { m } | \widetilde { p } _ { m } \in c _ { m } , m = 1 , 2 , . . . , k \}
$$

where $d$ denotes the demeonstration set, $c _ { k }$ denotes the $k$ - th cluster. Specifically, we use patterns primarily to select demonstrations rather than directly as context for downstream tasks. We utilize the original problem $q _ { k }$ and rationale $r _ { k }$ corresponding to the $p _ { k }$ patterns as the CoT input.

# Number of Demonstrations

Since previous methods lack knowledge-based guidance, the choice of $k$ is often based on heuristic values. However, having too many demonstrations does not proportionally enhance the performance (Wei et al. 2022b; Agarwal et al. 2024), while too few may fail to adequately capture the task’s characteristics. By incorporating reasoning operations, we can use the number of these operations to inform a more reasonable choice for $k$ :

$$
k = \lceil \frac { 1 } { 2 } \times n \times ( 1 + \log ( N ) ) \rceil
$$

where $n$ denotes the number of identified operations, and $\left\lceil \right\rceil$ represents the ceiling function that rounds up to the nearest integer. This formula empirically takes into account the impact of the number of operation types on the number of demonstrations and further adjusts based on the sample size.

# Experiments

In this section, our objective is to evaluate the effectiveness of our proposed method and answer the following research questions:

• RQ1: Does incorporating reasoning patterns enhance the effectiveness of CoT prompting?   
• RQ2: How do the reasoning patterns influence the outputs of LLMs?   
• RQ3: Is our method robust and scalable to other models?

Algorithm 1: Pattern-CoT Demonstration Selection   
Table 1: The number of samples and operation tokens.   

<html><body><table><tr><td>Dataset</td><td>Samples</td><td>Operation Tokens</td></tr><tr><td>GSM8K</td><td>1319</td><td>+,-,×,/ ‘more',‘less',‘twice',‘half'</td></tr><tr><td>AQuA</td><td>254</td><td>+,-,×,/,π,√x,xn,x°,log</td></tr><tr><td>MultiArith AddSub SingleEq</td><td>600 395 508</td><td>+,-,×,/</td></tr><tr><td>SVAMP Coin</td><td>1000 500</td><td>‘heads up',‘tails up'</td></tr><tr><td>Date</td><td>369</td><td>‘day',‘week', month',‘year' ‘yesterday',‘tomorrow'</td></tr></table></body></html>

# Experimental Setup

Datasets. We adopt eight representative datasets for our reasoning tasks: MultiArith (Roy and Roth 2015), GSM8K (Cobbe et al. 2021), AddSub (Hosseini et al. 2014), AQUARAT (Ling et al. 2017), SingleEq (Koncel-Kedziorski et al. 2015), SVAMP (Patel, Bhattamishra, and Goyal 2021), Coin-Flip (Wei et al. 2022b), and BIG-bench Date Understanding (Srivastava et al. 2023). They require certain reasoning steps and are commonly used for CoT method comparisons (Wei et al. 2022b; Kojima et al. 2022; Zhang et al. 2023; Wang et al. 2023b; Fu et al. 2022).

For tasks MultiArith, AddSub, SingleEq, and SVAMP, we define the set of operation tokens based on a glossary from Wikipedia, as the operations involved are relatively straightforward. For tasks GSM8K and AQUA, we expand the operation token vocabulary manually based on data distribution. For tasks Coin-Flip and BIG-bench Date Understanding, we prompt GPT-4 to generate the corresponding operation tokens. The specific details of the datasets can be found in Table 1.

Language Models. To facilitate subsequent interpretability analysis, we select open-source models as our reasoning engine. Specifically, we use models from the LLaMA-2 family due to their foundational logical reasoning capabilities and support for CoT prompting. These models are deployed on our local server, which is equipped with 8 RTX 3090 GPUs, each with 24GB of memory. Due to hardware constraints, we test only the 7B and 13B models. Experiments with larger models or those from other families are discussed in subsequent sections.

We use the inference functions of these models, and the process does not involve training or fine-tuning. Additionally, we set the hyperparameters with a temperature of 0.4 and top p of 0.9 to manage the model’s randomness $\mathrm { \Delta X u }$ et al. 2022). To maintain consistency with (Zhang et al. 2023), we use Sentence-BERT (Reimers and Gurevych 2019) as our encoder and select the ‘all-MiniLM-L6-v2’ model for semantic vector representation. This model has also been proven effective in our experiments.

Table 2: Accuracy $( \% )$ on eight reasoning datasets. We present the mean value obtained from five runs. \* denotes the situation where $k$ does not change, and results are copied from above. For the Random-CoT method, we report the best result since we are concerned about the potential of CoT. For the self-consistency method, we set the number of paths as 5 (Wang et al. 2023a).   

<html><body><table><tr><td colspan="2">LLaMA-2 Model</td><td>MultiArith</td><td>GSM8K</td><td>AddSub</td><td>AQuA</td><td>SingleEq</td><td>SVAMP</td><td>Coin</td><td>Date</td></tr><tr><td rowspan="7">7b-chat-hf</td><td>Zero-Shot-CoT</td><td>72.33</td><td>21.00</td><td>57.97</td><td>24.01</td><td>57.67</td><td>41.90</td><td>44.60</td><td>39.29</td></tr><tr><td>(+ SC)</td><td>79.83</td><td>27.14</td><td>62.78</td><td>21.65</td><td>68.11</td><td>47.60</td><td>52.80</td><td>40.37</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Rand-m-CoT</td><td>76.16</td><td>24.49</td><td>65.58</td><td>22.4</td><td>6.14</td><td>46.59</td><td>48.00</td><td>444</td></tr><tr><td>Auto-CoT-RA</td><td>74.83</td><td>26.76</td><td>63.29</td><td>23.80</td><td>66.92</td><td>45.19</td><td>48.00</td><td>43.08</td></tr><tr><td>Ours</td><td>79.66</td><td>27.45</td><td>65.06</td><td>28.34</td><td>71.85</td><td>48.50</td><td>59.40</td><td>45.79</td></tr><tr><td>Ours (Adaptive k)</td><td>79.66*</td><td>28.05</td><td>67.08</td><td>29.13</td><td>71.85*</td><td>48.50*</td><td>58.40</td><td>46.34</td></tr><tr><td rowspan="5">13b-chat-hf</td><td>Zero-Shot-CoT</td><td>77.50</td><td>34.49</td><td>60.75</td><td>15.74</td><td>69.29</td><td>49.40</td><td>47.40</td><td>46.07</td></tr><tr><td>Auto-CoT</td><td>82.16</td><td>36.77</td><td>63.03</td><td>25.19</td><td>70.67</td><td>55.50</td><td>54.20</td><td>53.93</td></tr><tr><td>Auto-CoT-RA</td><td>82.16</td><td>37.04</td><td>62.08</td><td>27.74</td><td>66.14</td><td>52.10</td><td>62.80</td><td>54.47</td></tr><tr><td>Ours</td><td>83.16</td><td>37.68</td><td>65.82</td><td>26.37</td><td>74.80</td><td>56.39</td><td>57.40</td><td>56.91</td></tr><tr><td>Ours (Adaptive k)</td><td>83.16*</td><td>38.44</td><td>64.81</td><td>31.49</td><td>74.80*</td><td>56.39*</td><td>67.80</td><td>60.97</td></tr></table></body></html>

Baselines. We primarily compare our methods with unsupervised methods including Zero-Shot-CoT (Kojima et al. 2022), Random-CoT, Auto-CoT (Zhang et al. 2023), and Self-Consistency (Wang et al. 2023a). Building on AutoCoT, we introduce an additional variant, Auto-CoT-RA, which replaces the original question embedding with the rationale embedding for clustering. The purpose of this modification is to investigate whether this subtle shift can implicitly uncover the underlying patterns in reasoning. Unless otherwise specified, our method uses the same $k$ value as the baseline in experiments. Additionally, we conduct experiments using our method with the adaptive $k$ value that we designed.

# Main Results (RQ1)

Table 2 presents the overall performance of various methods on the 7B and 13B models. Since our primary goal is to evaluate whether focusing on diverse patterns provides more benefit to reasoning than semantic information, we are not concerned with identifying which model achieves stateof-the-art performance. Based on these results, we make the following observations:

• Overall, our method consistently outperforms the baseline approaches. This stable improvement indicates that by introducing diverse reasoning patterns, we can identify more representative demonstration sets, where each example embodies a different reasoning strategy. Using these diverse examples as context for LLMs can further enhance their ability to solve downstream tasks. • We observe that for arithmetic problems with a limited set of operation tokens, such as MultiArith, AddSub, SingleEq, and SVAMP, our method achieves more significant improvements compared to methods based on semantic information. This suggests that the demonstration sets we construct can effectively cover the majority of reasoning paths, thereby providing comprehensive guidance for LLMs to select appropriate reasoning patterns.

![](images/f8922db1e7a36c7239ca09a820b19e200b3f6517f47055ff2893aded887462e1.jpg)  
Figure 3: Comparison of different operation sets.

• For datasets with a relatively broader action space, like GSM8K and $\mathrm { \ A O u A }$ , the improvements are less significant. This implies that a limited number of examples do not fully capture the diversity of reasoning patterns. However, when we recalculate the number of clusters using adaptive $k$ and expand the demonstration set, we observe additional gains on these two datasets. • Surprisingly, we find that for datasets like Coin and Date, where the operation patterns are not explicitly defined, our method actually lead to greater improvements. We hypothesize that this is because the questions in these datasets are quite similar, making it difficult to distinguish them based on semantic features alone. In contrast, leveraging reasoning patterns allows us to identify representative examples from a different dimension. Additionally, both of these datasets show further performance improvements when using adaptive $k$ .

There are several additional observations. For instance, in some cases, Auto-CoT-RA outperforms Auto-CoT, while in others it does not. This suggests that simply shifting from question semantics to rationale semantics does not necessarily narrow the gap between demonstrations and the reasoning task. Deeper reasoning patterns can still be obscured by irrelevant information. Moreover, in certain situations, using a random demonstration set can also surpass Auto-CoT, although this improvement is inconsistent. This indirectly highlights that other factors, such as underlying reasoning patterns, can influence the effectiveness of examples. Our method, in most cases, demonstrates a more stable ability to uncover these factors.

Table 3: Case study of Auto-CoT and our method for demonstration selection and downstream inference. The upper questions and answers are demonstrations constructed by two methods, and the lower part contains how LLMs solve the inference task.   

<html><body><table><tr><td>Auto-CoT</td><td>Ours Q: Tom was at the beach for 5 days and found 7 seashells</td></tr><tr><td>Q: Mark's father gave him $85.Mark bought 1O books,each of which cost $5.How much money does Mark have left? A: Let's think step by step.If Mark has $85 and he buys 10 books at $5 each, then he will spend a total of $50 (10 x $5). So,after buying the books,Mark will have $35 ($85 - $50) left. The answer is 35. Q:Mike has 48 books.Alyssa has 8 times more books than Mike.How many dozen books does Alyssa have? A:Let's think step by step.If Mike has 48 books,then Alyssa has 8 times that number = 8 x 48 = 384 books. So Alyssa has 384 books. The answer is 384.</td><td>every day. How many seashells did Tom find during the beach trip? A: Let's think step by step. If Tom found 7 seashells every day, then he found a total of 7 x 5 = 35 seashells during his entire beach trip. The answer is 35. Q:Nancy has saved 49oO cents from selling lemonade. How many dollars does Nancy have? A:Let's think step by step.Nancy saved 490O cents, which means she saved 4900/100 = 49 dollars. The an- swer is 49.</td></tr><tr><td>Q: The value of a sport utility vehicle this year is 16,OoO dollars ,which is O.8 of what its value was last year.How much is the value of the vehicle last year? A:Let's think step by step.The problem tells us that the value of the sport utility vehicle this year is 16,OoO dollars,which is O.8 of what it was last year. So, if we multiply the value of the vehicle this yearby O.8,we get thevalue of the vehiclelast year = 16,000 x 0.8 = 12,800 dollars. The answer is 12,800.</td><td>Q: The value of a sport utility vehicle this year is 16.000 dollars,which is O.8 of what its value was last year. How much is the value of the vehicle last year? A:Let's think step by step. The problem tells us that the value of the sport utility vehicle this year is $16,000, which is O.8 times its value last year. This means that the value last year is $16,000 / 0.8 = $20,000. The answer is $20,000.</td></tr></table></body></html>

Table 4: The number of demonstrations and their error rate for each dataset.   

<html><body><table><tr><td>Dataset</td><td>Demos</td><td>Incorrect</td><td>Error Rate</td></tr><tr><td>MultiArith</td><td>8</td><td>2</td><td>25.0%</td></tr><tr><td>GSM8K</td><td>8</td><td>5</td><td>62.5%</td></tr><tr><td>AddSub</td><td>8</td><td>3</td><td>37.5%</td></tr><tr><td>AQuA</td><td>4</td><td>4</td><td>100%</td></tr><tr><td>SingleEq</td><td>8</td><td>2</td><td>25.0%</td></tr><tr><td>SVAMP</td><td>8</td><td>6</td><td>75%</td></tr><tr><td>Coin</td><td>8</td><td>3</td><td>37.5%</td></tr><tr><td>Date</td><td>8</td><td>1</td><td>12.5%</td></tr></table></body></html>

# Impact of Operation Tokens (RQ1)

To further assess the impact of reasoning patterns, we conduct additional experiments. Given that GSM8K and AQuA datasets utilize additional operation tokens, we removed some of these tokens to determine their influence. Specifically, we categorize the expanded operation tokens into a basic operation subset, such as $\{ + , - , \times , / \}$ , similar to other arithmetic tasks, and the remaining tokens as supplementary subsets. These subsets represent only a portion of the reasoning patterns within these two datasets.

Figure 3 shows the results of using different subsets on the 7B model. The experimental results demonstrate that using operation subsets as reasoning pattern tokens can degrade overall performance. The primary reason for this is that these subsets do not sufficiently cover the task’s logical scope. It leads to a lack of diversity. However, when the full set of operations is utilized, a broader range of scenarios can be activated, allowing the model to better adapt to the task.

# Case Study (RQ2)

To gain a deeper understanding of CoT prompting, we perform a case study. Table 3 presents a typical instance analysis. We observe that Auto-CoT, due to its introduction of numerous irrelevant patterns, tends to distort the reasoning results of LLMs. In contrast, our method, which includes a diverse set of reasoning pattern templates, enables the model to generate correct responses.

# Feature Attribution (RQ2)

Following the previous case study, we seek to understand why different contextual reasoning patterns alter the output of LLMs. Specifically, we employ a perturbation-based feature attribution analysis method (Winter 2002) to aid in this understanding. Traditional attention-based analysis methods have been criticized for their inability to identify the most significant features (Wiegreffe and Pinter 2019; Zhao et al. 2024), which is why we turned to this perturbation-based approach. By masking portions of the input tokens, we recompute the generation probabilities for each output token to as

X48=30 x4=J 6、 4= 50(10x5 \$35(\$85 \$50 9+3 12 4900 /100 49 9-3=6 6、 6=0.3452 0.2580 0.5823 0.0161 -0.0472 _s -0.0756 0.0216 0.0065 0.1044 -0.0691  
=  
1 0.0811 -0.0066 -0.0037 0.3934 -0.0356 1 0.0198 -0.0041 -0.0987 0.0141 -0.0096 1.51.0  
6 0.0234 0.0145 0.0512 0.0243 0.0135 6 0.0001 0.0000 -0.0004 0.0000 -0.00001.0  
， 0.0049 -0.0019 -0.0196 -0.0211 -0.0011 ， 0.0017 -0.0041 -0.0351 -0.0060 -0.00700.5  
01 -0.0009 0.0008 -0.0015 -0.0010 -0.0000 01 0.0005 0.0006 -0.0048 -0.0003 -0.0002 -0.5  
01 0.0000 0.0000 -0.0000 0.0000 -0.0001 01 0.0000 -0.0000 -0.0001 0.0000 0.0000  
01 0.0001 0.0002 0.0002 -0.0001 -0.0002 0.0 01 0.0003 0.0005 -0.0004 0.0000 0.0001 0.0  
X 1.3714 0.6865 0.1771 0.6837 -0.3675 L 0.3820 0.0909 1.9074 0.6336 0.6902-0.50.0089 0.0134 0.0232 -0.0061 -0.0099 0.0458 0.0362 0.0022 0.0116 0.0091  
1 -0.5  
0- -0.0054 0.0008 0.0012 0.0025 0.0016 01 0.0005 0.0006 0.0000 0.0002 0.0002-1.0  
.1 0.0020 0.0022 0.0026 0.0026 -0.0011 0.0014 0.0003 0.0012 0.0001 -0.0002  
8- -0.0007 0.0015 0.0007 -0.0036 0.0009 -1.0 8- 0.0005 0.0003 0.0002 0.0001 0.0002 -1.5  
_= 0.0081 0.0104 0.0097 -0.0127 0.0004 0.0017 0.0026 0.0024 0.0007 0.0021

Figure 4: Visualization of token attribution for the case study. The left part stands for the score matrix of patterns from AutoCoT, and the right part stands for the score matrix from our method. The upper column denotes each individual prompt, and the row denotes the generated token sequence. Higher scores (positive) indicate that the input has a greater impact on the output.

Table 5: Result of GPT-3.5-turbo-0125 and Qwen-7b-chat model on different datasets.   

<html><body><table><tr><td colspan="2">Model</td><td>AddSub</td><td>AQuA</td><td>SingleEq</td></tr><tr><td rowspan="3">GPT-3.5</td><td>Zero-Shot</td><td>83.29</td><td>59.44</td><td>90.55</td></tr><tr><td>Auto-CoT</td><td>81.26</td><td>58.66</td><td>91.53</td></tr><tr><td>Ours</td><td>83.54</td><td>62.38</td><td>93.11</td></tr><tr><td rowspan="3">Qwen</td><td>Zero-Shot</td><td>54.93</td><td>35.03</td><td>69.07</td></tr><tr><td>Auto-CoT</td><td>62.53</td><td>30.31</td><td>80.31</td></tr><tr><td>Ours</td><td>67.59</td><td>33.46</td><td>82.08</td></tr></table></body></html>

sess the input’s attribution impact on these output tokens. We use Captum (Miglani et al. 2023) to achieve this visualization. Figure 4 presents the attribution analysis matrix for the case study. According to the visualization results, we find that when a particular pattern is overly dense in the examples, the model tends to activate related knowledge, which can lead to biased reasoning processes. Conversely, when these patterns are more diverse, the model is more likely to activate the correct reasoning pathways. Our method, by enhancing the diversity of patterns in the demonstrations, effectively reduces the distance to the reasoning task objectives.

# Error Robustness (RQ3)

It is worth mentioning that we do not impose supervision on the labels of the demonstrations. Therefore, we proceed to count the number of incorrect instances within the selected set, as shown in Table 4. It is intriguing to notice that the majority of our provided demonstrations are imperfect, with

AQuA even exhibiting a $100 \%$ error rate. This phenomenon suggests that LLMs struggle to discern incorrect instances from correct ones. Instead, they learn from how the example approaches problem-solving, which we refer to as ‘pattern’. Our method encourages LLMs to follow the most probable reasoning chain towards the final answer and thus leads to a significant improvement.

# Results on Other Models (RQ3)

To determine whether our method is applicable to different models, we test it on various LLM branches. Specifically, we select the GPT series to represent larger and more advanced models, and Qwen to represent multilingual models. For the sake of hardware resources and budget constraints, we experiment with the GPT-3.5-turbo and Qwen-7B models. Table 5 presents the performance of several methods on these models. Notably, the experiments show that Auto-CoT, in some cases, underperforms compared to direct answering on these models. We attribute this to the inherent noise in semantics-based methods. Our approach mitigates this noise, resulting in more consistent performance improvements.

# Conclusion

This paper aims to address the noise issue inherent in unsupervised semantic-based CoT methods and proposes a reasoning pattern-based approach for CoT demonstration selection. Our method explicitly enhances the interpretability of reasoning processes and illustrates how LLMs can be guided toward generating accurate answers. Extensive experiments validate the effectiveness, robustness, and compatibility of our approach.