# Unleashing the Potential of Large Language Models as Prompt Optimizers: Analogical Analysis with Gradient-based Model Optimizers

Xinyu Tang1\*, Xiaolei Wang1\*, Wayne Xin Zhao1†, Siyuan $\mathbf { L } \mathbf { u } ^ { 2 }$ , Yaliang $\mathbf { L i } ^ { 3 }$ , Ji-Rong Wen

1Gaoling School of Artificial Intelligence, Renmin University of China 2International School, Beijing University of Posts and Telecommunications 3Alibaba Group txy20010310@163.com, wxl1999 $@$ foxmail.com, batmanfly $@$ gmail.com

# Abstract

Automatic prompt optimization is an important approach to improving the performance of large language models (LLMs). Recent research demonstrates the potential of using LLMs as prompt optimizers, which can generate improved task prompts via iterative refinement. In this paper, we propose a novel perspective to investigate the design of LLM-based prompt optimizers, by drawing an analogy with gradient-based model optimizers. To connect these two approaches, we identify two pivotal factors in model parameter learning: update direction and update method. By systematically analyzing a rich set of improvement strategies on the two aspects, we further develop a capable Gradient-inspired LLM-based Prompt Optimizer called GPO. At each step, it first retrieves relevant prompts from the optimization trajectory as the update direction. Then, it utilizes the generation-based refinement strategy to perform the update, while controlling the edit distance through a cosinebased decay strategy. Extensive experiments demonstrate the effectiveness and efficiency of GPO. In particular, GPO brings an additional improvement of up to $56 . 8 \%$ on Big-Bench Hard and $6 2 . 6 \%$ on MMLU compared to baseline methods.

# Introduction

Nowadays, prompting has become the pivotal approach to unleashing the power of large language models (LLMs) (Zhao et al. 2023). However, prompt engineering is not easy and requires extensive trial-and-error efforts since LLMs are sensitive to prompts (Lu et al. 2022; Wang et al. 2024; Tang et al. 2024). Although general guidelines for high-quality prompts exist (Kojima et al. 2022; Amatriain 2024), they cannot always lead to optimal task performance.

To improve the task performance of LLMs, automatic prompt optimization has been proposed (Zhou et al. 2023). Early work either performs discrete optimization through methods like reinforcement learning (Zhang et al. 2023) or performs continuous optimization in the embedding space (Chen et al. 2023; Wang et al. 2022a, 2023). However, they often require access to the logits or internal states of LLMs, which is infeasible for those only accessible through APIs. In addition, they need to be specially trained for each task. Considering these issues, recent work proposes to model the optimization problem in natural language and using LLMs as prompt optimizers due to their strong understanding (Zhan et al. 2024) and generation capabilities (Schnabel and Neville 2024). In this approach, LLMs perform optimization with only outputs from the task model and quickly adapt to various tasks without training. However, such a method raises a new challenge for the design of meta-prompt, which is the prompt for LLMs to perform prompt optimization.

![](images/6804252c5172f1ac069e996be30f96361cb0485296eafbc0943c51cf8ba95570.jpg)  
Figure 1: Comparisons of GPO to existing LLM-based prompt optimizers in terms of effectiveness (Accuracy) and efficiency (improvement per dollar spent on API) on BBH.

To tackle this issue, we aim to investigate the design of meta-prompts. Existing methods for creating meta-prompts typically involve either manual human effort (Yang et al. 2023) or heuristic algorithms (Fernando et al. 2023). Despite the flexibility, these studies still lack principled guidelines about their designs. Our work is inspired by the great success of gradient-based optimizers in model optimization, which have been systemically studied in both theory and practice (Sun et al. 2020). Since both optimizers aim at enhancing model performance through iterative optimization, it is feasible to connect the two different approaches via analogical analysis. In this way, we can borrow the theoretical framework and extensive research of gradient-based model optimizers to enhance LLM-based prompt optimizers.

Therefore, in this paper, we propose a comprehensive analogy framework for the two key factors (i.e., update direction

Update direction LLM-based   
The current prompt is: {current prompt}     Score: {score} Meta-prompt prompt optimizer $\mathcal { M } _ { O }$   
Below are the previous prompts with their scores. The score ranges from 0 to 100,   
and higher scores indicate better quality.   
Prompt: {prompt} Score: {score} Evraelsualtison Task model   
Update method F(Mt(x;p),y) $\mathcal { M } _ { T }$   
You are allowed to change up to {number} words in the current prompt.   
Carefully analyze the previous prompts and their scores, and write a new improved Task prompts $p$   
prompt to replace <Prompt> in the task examples. Current information (Descent direction) Trajectory (Momentum) Edit distance (Learning rate) Refinement strategy (Descent) Optimal task prompt

and update method) in LLM-based optimizers. As illustrated in Table 1, for update direction, we analogize descent direction to the information of the current prompt and momentum to the information of the historical prompt, while for the update method, we analogize the learning rate to edit distance and gradient descent to the refinement strategy. Based on such a framework, we conduct systematic empirical studies on various implementations for each component and report their experimental results with detailed analysis.

Based on the findings from our systematic analysis, we further develop a capable Gradient-inspired LLM-based Prompt Optimizer called GPO, with the best implementation for each component. Figure 2 illustrates the overall framework of GPO. We evaluate its effectiveness across various LLMs, tasks, and evaluation settings. When using Llama-2-7b-chat as the task model, the prompts produced by GPO surpass the instruction “Let’s think step by step” by $1 8 . 5 \%$ on Big-Bench Hard (BBH) and $7 . 6 \%$ on MMLU. Furthermore, GPO produces an additional improvement of up to $56 . 8 \%$ on BBH and $6 2 . 6 \%$ on MMLU compared with baseline methods while using fewer tokens.

Our contributions can be summarized as follows: $\bullet$ To the best of our knowledge, this is the first time that a systematic analogy has been conducted for LLM-based prompt optimizers with gradient-based model optimizers. We conduct a comprehensive experimental analysis on the two key factors (i.e., update direction and update method) and report several key findings. Based on the findings of the systematic analysis, we develop a more effective and efficient LLM-based prompt optimizer, GPO, which surpasses competitive baseline methods across various LLMs, tasks, and evaluation settings while incurring lower costs.

# Related Work

Prompt Engineering and Optimization. Prompt engineering aims to find suitable prompts for LLMs to perform various tasks. To reduce human efforts, researchers have explored automatic prompt optimization, which can be categorized into continuous and discrete optimization methods. Discrete methods directly optimize the natural language prompts through methods like reinforcement learning (Deng et al. 2022; Zhang et al. 2023) and editing (Xu et al. 2022; Prasad et al. 2023). In contrast, continuous methods perform optimization in the embedding space of LLMs, allowing for optimization through gradient (Li and Liang 2021). We focus on discrete methods, especially LLM-based prompt optimizers.

LLM-based Prompt Optimizers. Due to the unprecedented capabilities of LLMs, recent work starts to utilize them as prompt optimizers. One line of work (Guo et al. 2023; Yang and Li 2023) combines LLMs with evolutionary algorithms to perform prompt optimization. Another line of work (Yang et al. 2023) aims to adapt concepts and techniques from gradient-based model optimizers (e.g., gradient (Pryzant et al. 2023) and momentum (Yang et al. 2023)) to LLM-based prompt optimizers. However, no comprehensive guidelines exist for using LLMs as prompt optimizers. We aim to tackle this with a systematic investigation, which is conducted by analogy with gradient-based model optimizers.

# Analogical Analysis

In this section, we present an analogical analysis between model optimization and prompt optimization to build connections and improve existing LLM-based prompt optimizers.

# Background

In this part, we first introduce the task definition of LLMbased prompt optimization, and then establish connections with gradient-based optimizers.

Task Formulation. Prompt optimization aims to find the optimal task prompt $p ^ { * }$ in the format of natural language that maximizes the performance on a specific task dataset $\mathcal { D }$ when using an LLM as the task model $\mathcal { M } _ { T }$ . To perform such optimization, our idea is to develop a prompt optimizer, which can be built upon some search algorithm (e.g., evolutionary algorithms (Guo et al. 2023)) or an LLM (Yang et al. 2023). In this paper, we focus on using an LLM as the prompt optimizer $\mathcal { M } _ { O }$ . Formally, the problem of prompt optimization can be formulated as:

Table 1: Analogy between glossaries in model optimizer and prompt optimizer.   

<html><body><table><tr><td>Factor</td><td>Gradient-based model optimizer</td><td>LLM-based prompt optimizer</td></tr><tr><td>Update direction</td><td>Descent direction Momentum</td><td>Current information Trajectory</td></tr><tr><td>Update</td><td>Learning rate</td><td>Edit distance</td></tr><tr><td>method</td><td>Descent</td><td>Refinement strategy</td></tr></table></body></html>

$$
p ^ { * } = \underset { p \sim \mathcal { M } _ { O } } { \arg \operatorname* { m a x } } ~ \mathbb { E } _ { \langle x , y \rangle \in \mathcal { D } } ~ \big [ F ( \mathcal { M } _ { T } ( x ; p ) , y ) \big ] ,
$$

where $p$ is the prompt generated by the prompt optimizer $\mathcal { M } _ { O }$ , $\boldsymbol { \mathcal { M } } _ { T } ( \boldsymbol { x } ; p )$ represents the output from the task model for input $x$ conditioned on the prompt $p$ , and the function $F ( \cdot )$ calculates the task performance based on some measurement.

Revisiting Gradient-based Optimizers. Similar to LLMbased prompt optimizer, gradient-based model optimizer aims to find the optimal values of model parameters that minimize the loss function. In the basic form of gradient descent (Boyd and Vandenberghe 2014), a single optimization step can be formulated as follows:

$$
\Theta _ { k + 1 } = \Theta _ { k } - \tau _ { k } g _ { k } ,
$$

where $\Theta _ { k }$ and $\Theta _ { k + 1 }$ are the values of model parameters at the last and current steps, $\tau _ { k }$ and $g _ { k }$ are the learning rate and gradient at the current step. Gradient descent can be improved by focusing on two elements in the formula: $\tau _ { k }$ and $g _ { k }$ . For $\tau _ { k }$ , learning rate schedulers (Gotmare et al. 2019) are proposed to dynamically adjust the learning rate. For $g _ { k }$ , the concept of momentum (Sutskever et al. 2013) is introduced to include historical gradients, and its computation can be expressed as follows: $\begin{array} { r } { v _ { k + 1 } = \beta v _ { k } + g _ { k } \stackrel {  } { = } \sum _ { i = 1 } ^ { k } \beta ^ { k - i } g _ { i } } \end{array}$ where $\beta$ represents the momentum coefficien .

Despite various gradient-based optimizers, they mainly model two key factors, namely update direction (e.g., gradient $g _ { k }$ ) and update method (e.g., subtract $\tau _ { k } g _ { k } .$ ). Our approach is inspired by the observation that existing LLM-based prompt optimization methods also implicitly employ the two aspects (see Table 1). However, existing work only initially explores the design of the two key factors, we aim to conduct more in-depth and systematic investigations from these two novel perspectives in the following subsection.

# Update Direction

The update direction refers to the adjustments based on the information of previous or current prompts to determine the best direction for improving them. We apply the descent direction and momentum concepts to design the meta-prompts.

Analogical Descent Direction. Descent direction determines the direction of parameter updates based on the model performance. We analogize two similar forms that determine how to improve new prompts according to the current information.

Prompt+Performance. One straightforward method is to include the last-round task prompt and the corresponding model performance into the meta-prompt for the LLM-based optimizer $\mathcal { M } _ { O }$ . It leverages the capacity of LLMs to reason about how to improve prompting optimization.

Prompt $^ +$ Performance+Reflection. Another way to solve the barrier of the descent direction is to leverage the reflection capability of LLMs (Pryzant et al. 2023). With the reflection mechanism, LLMs can generate feedback from past failures to improve performance. Such feedback can be seen as a form of “semantic” gradient signals.

Analogical Momentum. Inspired by the momentum in gradient descent, we aim to make LLMs aware of the prompts used in the optimization process and their corresponding results (i.e., descent direction), thereby achieving a more global update direction. A straightforward way is to directly include all intermediate results into the meta-prompt. However, it might be limited by the context length of LLMs and affected by the accumulated noise. To better utilize the optimization trajectory, we propose two alternative methods.

Summarization-based trajectory. One simple approach is to summarize the intermediate results from the optimization trajectory. Specifically, at each step, we use an instruction (i.e., “Your task is to summarize the problems in the previous prompt and the current prompt.”) to let the LLM perform summarization using the summary in the last step and the result in the current step.

Retrieval-based trajectory. Another approach is to select $k$ pieces of intermediate results from the optimization trajectory. Specifically, we consider three strategies: (1) Recency: selecting $k$ nearest intermediate results; (2) Relevance: selecting $k$ most relevant intermediate results, which are measured by the semantic similarity based on the BGE model (Xiao et al. 2023); (3) Importance: selecting $k$ most important intermediate results, which are measured by the performance.

# Update Method

The update method can refer to how the LLM utilizes such information to improve task prompts. Accordingly, we explore how to analogize the learning rate and the descent process into the design of meta-prompts.

Analogical Learning Rate. In the model optimization, the learning rate controls the extent of gradient updates at each step. Similarly, we aim to control the variation degree of prompt optimization. Specifically, we limit the number of words that can be modified in the meta-prompt (i.e., edit distance). Accordingly, we propose two methods of controlling edit distance as follows:

Decay-based constraint. To avoid overshooting the optimal solution, the decay strategy is proposed to gradually reduce the learning rate (Izmailov et al. 2018). Here, we borrow the idea of controlling the maximum edit distance and consider gradually reducing its value following either a linear or cosine curve. In particular, we reduce the constraint to approximately $20 \%$ of its maximum value until convergence.

<html><body><table><tr><td colspan="2">Prompt Optimizer</td><td></td><td colspan="2">GPT-3.5-turbo</td><td colspan="2">GPT-4</td><td colspan="2">GPT-4</td></tr><tr><td colspan="2">Task Model</td><td></td><td colspan="2">Llama-2-7b-chat</td><td colspan="2">Llama-2-7b-chat</td><td colspan="2">GPT-3.5-turbo</td></tr><tr><td colspan="2">Gradient</td><td></td><td>P+M</td><td>P+M+R</td><td>P+M</td><td>P+M+R</td><td>P+M</td><td>P+M+R</td></tr><tr><td rowspan="5">Momentum</td><td>None</td><td></td><td>41.07</td><td>40.34</td><td>40.32</td><td>39.56</td><td>64.76</td><td>64.55</td></tr><tr><td> Summarization</td><td></td><td>41.03</td><td>40.63</td><td>40.58</td><td>40.41</td><td>64.62</td><td>64.55</td></tr><tr><td>Recency</td><td></td><td>41.93</td><td>41.55</td><td>42.02</td><td>41.34</td><td>65.26</td><td>64.97</td></tr><tr><td>Relevance</td><td></td><td>42.53</td><td>40.81</td><td>42.89</td><td>41.97</td><td>65.87</td><td>65.39</td></tr><tr><td>Importance</td><td>41.84</td><td></td><td>39.73</td><td>41.73</td><td>41.04</td><td>65.26</td><td>65.11</td></tr></table></body></html>

Table 2: The performance comparison of various update directions based on different prompt optimizers and task models. “P” denotes prompt, “M” denotes performance, and “R” denotes reflection.

Warmup-based constraint. To avoid instability in the early stage of optimization, the warmup strategy is proposed to gradually increase the learning rate at the beginning (Goyal et al. 2017). Similarly, we adopt the widely used linear warmup schedule to gradually increase the constraint for the maximum edit distance in the initial $5 \%$ steps.

Analogical Gradient Descent. By analogy with the subtraction operation in gradient descent (i.e., $- \tau _ { k } g _ { k }$ in Eq. (2)), we introduce two methods to update the task prompt accordingly.

Editing-based refinement. The first method directly edits the last-round task prompt to improve performance. Specifically, we add an instruction (i.e., “Modify the current prompt”) into the meta-prompt, which requires the LLM to edit the task prompt from the last step according to the update direction. This method allows for effectively exploiting the existing prompt.

Generation-based refinement. In addition to direct edits, we can also leverage the in-context learning capability of LLMs to generate refined task prompts. Specifically, we present the information regarding the updated direction in the format of demonstrations. Then, We add an instruction (i.e., “Write a new improved prompt”) to let the LLM follow the demonstration to directly generate a new task prompt. Compared with the editing-based method, the generation-based approach explores a wider range of prompt variations.

# Empirical Experiments

In this part, we describe the experiment setting for our analogical analysis and detail our findings from the experiment.

Experiment Setup. We select a dataset from each type of task in BBH (Suzgun et al. 2023) to create a lite BBH benchmark for our analysis: i) Navigate (binary choice); ii) Movie Recommendation (multiple choice); iii) Object Counting (numeric response); iv) Word Sorting (free response). We adopt exact match as the metric for performance evaluation. We use three different model combinations of prompt optimizers and task models (i.e., GPT-3.5-turbo and Llama-2-7b-chat, GPT-4 and Llama-2-7b-chat, GPT-4 and GPT-3.5-turbo).

The optimization process lasts for at most 3 epochs, under which the task prompt can reach the plateau.

Findings for Update Direction. The results for the update direction are presented in Table 2. Here are the main findings:

Reflection Leads to Performance Drop. Regarding the analogy to descent direction, prompt+performance achieves better performance than prompt+performance $^ +$ reflection, which brings an improvement of up to $31 \%$ compared with the initial task prompt. The improvement brought by prompts can be attributed to their rich semantic information about the task, which can activate the task-specific knowledge of LLMs for optimization. In contrast, LLMs are limited in their capabilities of reflection (Huang et al. 2023), which may lead to inaccurate updates.

Prompt Optimizers can Learn More from Contextually Relevant Prompts. The analogical concept of momentum can further improve the performance. Among various designs, relevance-based trajectory emerges as the most effective one, which brings an additional $1 5 \%$ improvement. This can be attributed to the fact that LLMs can learn more from contextually relevant prompts, while it might be challenging for LLMs to fully understand the signal of recency or importance. By contrast, the summarization-based trajectory proves to be less helpful. One possible reason is that summarization only captures common aspects of the trajectory while neglecting details that may be crucial.

Findings for Update Method. To investigate the update method for prompt optimization, we conduct experiments using the best configuration found in the previous experiments. The results for the update method are presented in Table 3.

Generation-based Refinement is Better. In general, generation-based refinement outperforms editing-based refinement, which brings an improvement of up to $36 \%$ compared with the initial task prompt. This can be partly attributed to the significance of exploration in prompt optimization. Generation-based strategy is not confined to the current prompt, allowing the model to better leverage the in-context examples. Therefore, this approach can demonstrate great flexibility to enable LLMs to explore a larger search space.

• Decay Strategy is Helpful. Among various controlling methods for prompt variation, cosine decay-based constraint achieves the best performance, bringing an additional $10 \%$ improvement. However, unlike gradient-based model optimization, the warmup strategy does not yield improvement. This finding suggests that, at the early stage of prompt optimization, exploration plays a crucial role in conducting large-scale prompt searches, while stability becomes more important in the later stage for more refined adjustments.

Table 3: The performance comparison of various update methods based on different prompt optimizers and task models.   

<html><body><table><tr><td colspan="2">Prompt Optimizer</td><td colspan="2">GPT-3.5-turbo</td><td colspan="2">GPT-4</td><td colspan="2">GPT-4</td></tr><tr><td colspan="2">Task Model</td><td colspan="2">Llama-2-7b-chat</td><td colspan="2">Llama-2-7b-chat</td><td colspan="2">GPT-3.5-turbo</td></tr><tr><td colspan="2">Learning rate</td><td>Editing</td><td>Generation</td><td>Editing</td><td>Generation</td><td>Editing</td><td>Generation</td></tr><tr><td rowspan="7">Descent</td><td>No control</td><td>42.53</td><td>42.61</td><td>42.89</td><td>43.17</td><td>65.87</td><td>66.37</td></tr><tr><td>Fixed</td><td>42.91</td><td>43.09</td><td>43.38</td><td>43.66</td><td>66.48</td><td>66.91</td></tr><tr><td>+Warmup</td><td>41.76</td><td>40.08</td><td>42.53</td><td>42.95</td><td>65.79</td><td>65.52</td></tr><tr><td>Linear decay</td><td>42.68</td><td>42.86</td><td>43.55</td><td>44.03</td><td>66.56</td><td>67.10</td></tr><tr><td>+Warmup</td><td>41.47</td><td>41.12</td><td>41.47</td><td>42.91</td><td>66.03</td><td>66.18</td></tr><tr><td>Cosine decay</td><td>42.95</td><td>43.75</td><td>43.98</td><td>44.97</td><td>66.74</td><td>67.80</td></tr><tr><td>+ Warmup</td><td>40.19</td><td>41.29</td><td>42.68</td><td>43.13</td><td>65.94</td><td>66.37</td></tr></table></body></html>

Table 4: Comparisons of GPO with existing LLM-based prompt optimizers. “P” refers to prompt, “M” refers to performance, and “R” refers to reflection.   

<html><body><table><tr><td rowspan="2">Prompt optimizer</td><td colspan="2">Updatedirection</td><td colspan="2">Update method</td></tr><tr><td>Currect information</td><td>Trajectory</td><td>Edit distance</td><td>Refinement strategy</td></tr><tr><td>APE</td><td>P</td><td>None</td><td>None</td><td>Generation</td></tr><tr><td>APO</td><td>P+R</td><td>None</td><td>None</td><td>Editing</td></tr><tr><td>OPRO</td><td>P+M</td><td>Recency</td><td>None</td><td>Generation</td></tr><tr><td>PE2</td><td>P+M+R</td><td>Recency</td><td>Fixed</td><td>Generation</td></tr><tr><td>GPO</td><td>P+M</td><td>Relevance</td><td>Decay</td><td>Generation</td></tr></table></body></html>

# Method

In this section, we present our novel gradient-inspired LLMbased prompt optimizer called GPO, which leverages the insights gained from our systematic study. GPO adopts an iterative prompt optimization framework. At each step, it first retrieves relevant prompts from the optimization trajectory as the update direction. Then, it utilizes the generation-based refinement strategy to perform the update, while controlling the edit distance through a cosine-based decay strategy.

Iterative Prompt Optimization. GPO performs prompt optimization through a multi-step iterative process. At each step, it first generates multiple candidate task prompts based on a meta-prompt. The meta-prompt serves as the input that guides the LLM in optimizing its prompts. Subsequently, we select the task prompt with the best performance for the next iteration. This iterative process continues until either the performance of the task prompt reaches a plateau or the predefined maximum number of optimization steps is

reached.

The Design of Meta-Prompt. As the input to the LLM, our meta-prompt consists of two key components: update direction and update method.

Update direction. To determine the update direction, we leverage the retrieval-based optimization trajectory in Section . This trajectory consists of past task prompts, along with their model performance. They are selected using a relevance-based strategy and sorted in ascending order based on their similarity to the current prompt.

Update method. After the update direction is determined, we further employ the generation-based refinement strategy to update the task prompt. Specifically, we present the trajectory in the format of demonstrations in the meta-prompt. Then, the LLM can follow these demonstrations to generate a new task prompt via in-context learning. Additionally, we implement the cosine-based decay strategy to control the edit distance between task prompts at consecutive iterations, ensuring gradual and controllable changes.

Furthermore, we enhance the meta-prompt by incorporating a few task examples. These examples provide additional context to aid the LLM in understanding the task effectively.

Comparison of LLM-Based Prompt Optimizers. Existing LLM-based prompt optimizers can be divided into two main classes by their update direction. Some work (i.e., APO (Pryzant et al. 2023) and PE2 (Ye et al. 2023)) leverages the reflection capability of LLMs to produce textual “gradients” as the update direction, while others (i.e., OPRO (Yang et al. 2023) and APE (Zhou et al. 2023)) directly use task prompts as the update direction. Our approach is based on the systematic investigation of the update direction and the update method. In particular, we propose several novel designs for the meta-prompt: relevance-based trajectory as the update direction and decay-based constraint for edit distance in the update method. Table 4 presents a detailed comparison.

# Experiments

In this section, we first set up the experiments and then report the results and detailed analysis.

Table 5: Performance comparison using only the task prompts obtained from different methods. “Human.” and “Social.” denote the datasets classified as Humanities and Social Science in MMLU.   

<html><body><table><tr><td>Task</td><td colspan="2">Complex reasoning task</td><td colspan="5">Knowledge intensive task</td><td colspan="2">Common NLP task</td></tr><tr><td rowspan="2">Dataset</td><td rowspan="2">BBH</td><td rowspan="2">GSM8K</td><td colspan="5">MMLU</td><td rowspan="2">WSC</td><td rowspan="2">WebNLG</td></tr><tr><td>STEM</td><td>Human.</td><td>Social.</td><td>Other</td><td>Average</td></tr><tr><td>Empty</td><td>30.51</td><td>22.00</td><td>31.05</td><td>36.54</td><td>41.75</td><td>37.20</td><td>35.96</td><td>60.67</td><td>32.14</td></tr><tr><td>CoT</td><td>29.91</td><td>24.00</td><td>32.53</td><td>37.05</td><td>41.05</td><td>36.94</td><td>36.36</td><td>59.33</td><td>31.11</td></tr><tr><td>SGDM</td><td>33.30</td><td>27.33</td><td>32.88</td><td>38.36</td><td>41.88</td><td>38.02</td><td>37.20</td><td>64.00</td><td>38.01</td></tr><tr><td>APE</td><td>32.94</td><td>25.00</td><td>33.51</td><td>38.69</td><td>42.02</td><td>37.96</td><td>37.50</td><td>62.00</td><td>36.49</td></tr><tr><td>APO</td><td>32.97</td><td>25.33</td><td>33.17</td><td>37.94</td><td>44.94</td><td>38.23</td><td>37.71</td><td>62.00</td><td>34.92</td></tr><tr><td>OPRO</td><td>33.29</td><td>26.67</td><td>34.76</td><td>38.72</td><td>43.55</td><td>37.11</td><td>38.05</td><td>63.33</td><td>37.89</td></tr><tr><td>PE2</td><td>33.43</td><td>25.33</td><td>33.77</td><td>37.95</td><td>44.80</td><td>38.25</td><td>38.07</td><td>62.67</td><td>39.10</td></tr><tr><td>GPO</td><td>35.43</td><td>28.33</td><td>35.00</td><td>38.84</td><td>46.61</td><td>38.60</td><td>39.14</td><td>65.33</td><td>42.51</td></tr></table></body></html>

# Experimental Setup

Tasks and Datasets. We select datasets from three groups of tasks for the experiment: Big-Bench Hard (BBH) (Suzgun et al. 2023) and GSM8K (Cobbe et al. 2021) for complex reasoning tasks, MMLU (Hendrycks et al. 2021) for knowledgeintensive tasks, and WSC (Levesque, Davis, and Morgenstern 2012) and WebNLG (Gardent et al. 2017) for common NLP tasks. Due to computational limitations, we sample a subset from each dataset for the main experiment. In addition, we use the lite BBH benchmark for detailed analysis.

Baselines. We select several representative methods for comparison, including existing LLM-based prompt optimizers and one from gradient-based model optimizers. (1) SGDM (Sutskever et al. 2013) is a momentum-based model optimizer. We adapt it for prompt optimization using the summarization-based trajectory and the editing-based refinement strategy. (2) APE (Zhou et al. 2023) utilizes LLMs to generate semantically similar variants of task prompts and selects one with the best performance. (3) APO (Pryzant et al. 2023) first uses reflection to obtain the gradient and then edits the task prompt accordingly. (4) OPRO (Yang et al. 2023) incorporates historical prompts with their scores into the meta-prompt. (5) PE2 (Ye et al. 2023) adds historical prompts and reflection into the meta-prompt and controls the edit distance with a fixed constraint. In addition, we consider the baseline without an instruction (“Empty”) and the instruction “Let’ think step by step.” from chain-of-thought prompting (Kojima et al. 2022) for performance comparison.

Evaluation Metrics. We report the average accuracy of all the subtasks for BBH and MMLU following Suzgun et al. (2023) and Hendrycks et al. (2021), accuracy for GSM8K following Cobbe et al. (2021), ROUGE-L for WSC and WebNLG following Wang et al. (2022b).

Implementation Details. We use both the base model (i.e., Llama-2-7b and Llama-3-8b) and the instructiontuned models (i.e., Baichuan2-7b-chat, Llama-2-7b-chat, Llama-2-13b-chat, Llama-3-8b-instruct (Dubey et al. 2024), GPT-3.5-turbo, and GPT-4) as the task model. For the prompt optimizer, we use GPT-3.5-turbo and GPT-4. Unless otherwise specified, we use Llama $- 2 - 7 b$ -chat as task model and GPT-3.5-turbo as prompt optimizer throughout experiments. We repeat all the experiments three times and report the average of the results.

Table 6: Performance comparison under different evaluation settings. “I” denotes instruction and “D” denotes demonstration.   

<html><body><table><tr><td>Task model</td><td>Llama- 2-7b</td><td>Llama- 3-8b</td><td colspan="2">Llama-2- 7b-chat</td><td colspan="2">Llama-3- 8b-instruct</td></tr><tr><td>Setting</td><td colspan="2">I+D</td><td>I</td><td>I+D</td><td>I</td><td>I+D</td></tr><tr><td>Empty</td><td>40.28</td><td>60.42</td><td>32.29</td><td>36.63</td><td>48.26</td><td>54.86</td></tr><tr><td>CoT</td><td>36.46</td><td>51.39</td><td>31.25</td><td>34.20</td><td>52.43</td><td>54.34</td></tr><tr><td>SGDM</td><td>42.19</td><td>62.50</td><td>40.63</td><td>35.77</td><td>60.42</td><td>58.51</td></tr><tr><td>APE</td><td>42.54</td><td>61.28</td><td>42.01</td><td>36.29</td><td>59.43</td><td>58.51</td></tr><tr><td>APO</td><td>42.19</td><td>61.45</td><td>40.34</td><td>36.29</td><td>59.38</td><td>57.99</td></tr><tr><td>OPRO</td><td>42.02</td><td>62.68</td><td>42.14</td><td>36.46</td><td>62.33</td><td>59.02</td></tr><tr><td>PE2</td><td>42.88</td><td>63.20</td><td>42.01</td><td>36.81</td><td>62.67</td><td>58.68</td></tr><tr><td>GPO</td><td>45.48</td><td>65.11</td><td>43.75</td><td>38.02</td><td>63.89</td><td>61.11</td></tr></table></body></html>

# Main Results

Table 5 and 6 show the results of different methods for prompt optimization across various tasks and evaluation settings.

First, when only considering the task prompt, we can see that trajectory-based methods (i.e., SGDM, OPRO, PE2, and GPO) perform very well. One possible reason is that the trajectory helps the prompt optimizer pay more attention to the important information instead of the noise in the current step. Furthermore, our prompt optimizer GPO achieves the best performance across all tasks. Our relevance-based trajectory provides semantically similar demonstrations that can be effectively learned by the LLM, while the cosine-based decay strategy can control the optimization process in a fine-grained manner through edit distance.

Table 7: Performance of different prompt optimization methods with various models.   

<html><body><table><tr><td>Prompt optimizer</td><td colspan="5">GPT-3.5-turbo</td><td colspan="5">GPT-4</td></tr><tr><td>Task model</td><td>Baichuan2- 7b-chat</td><td>Llama-2- 7b-chat</td><td>Llama-2- 13b-chat</td><td>GPT-3.5- turbo</td><td>GPT-4</td><td>Baichuan2- 7b-chat</td><td>Llama-2- 7b-chat</td><td>Llama-2- 13b-chat</td><td>GPT-3.5- turbo</td><td>GPT-4</td></tr><tr><td>Empty</td><td>15.75</td><td>32.29</td><td>40.97</td><td>60.48</td><td>72.87</td><td>15.75</td><td>32.29</td><td>40.97</td><td>60.48</td><td>72.87</td></tr><tr><td>CoT</td><td>19.45</td><td>31.25</td><td>40.28</td><td>62.15</td><td>73.61</td><td>19.45</td><td>31.25</td><td>40.28</td><td>62.15</td><td>73.61</td></tr><tr><td>SGDM</td><td>20.87</td><td>40.63</td><td>42.24</td><td>64.18</td><td>75.23</td><td>21.85</td><td>40.41</td><td>42.03</td><td>64.55</td><td>75.82</td></tr><tr><td>APE</td><td>20.61</td><td>42.01</td><td>41.65</td><td>63.63</td><td>74.71</td><td>20.29</td><td>39.98</td><td>42.96</td><td>64.38</td><td>74.13</td></tr><tr><td>APO</td><td>19.97</td><td>40.34</td><td>41.79</td><td>63.96</td><td>74.13</td><td>22.09</td><td>39.56</td><td>43.67</td><td>64.55</td><td>74.56</td></tr><tr><td>OPRO</td><td>19.86</td><td>42.14</td><td>42.89</td><td>64.56</td><td>75.06</td><td>21.28</td><td>41.77</td><td>43.35</td><td>64.91</td><td>76.67</td></tr><tr><td>PE2</td><td>21.11</td><td>42.01</td><td>42.68</td><td>64.95</td><td>75.27</td><td>22.43</td><td>42.03</td><td>44.91</td><td>65.23</td><td>76.55</td></tr><tr><td>GPO</td><td>23.35</td><td>43.75</td><td>44.83</td><td>67.02</td><td>76.56</td><td>25.34</td><td>44.97</td><td>46.17</td><td>67.80</td><td>78.65</td></tr></table></body></html>

Second, under various evaluation settings for the lite BBH benchmark, it can be observed that GPO not only excels in the “Instruction” setting but also yields substantial gains in the “Instruction $+$ Demonstration” setting for both the base model and the instruction-tuned variant. Even in the scenario that is challenging for baselines (i.e., Llama-2-7b-chat with “Instruction $^ +$ Demonstration”), our approach still demonstrates strong improvement. This showcases the versatility of our approach in both zero-shot and few-shot evaluation settings.

# Detailed Analysis

Next, we conduct a detailed analysis of our prompt optimizer GPO from the following aspects.

The Impact of Model Selection. To confirm the effectiveness of GPO across different models, we explore the impact of different model combinations compared with other baseline methods. Table 7 presents the results on the lite BBH benchmark. In general, GPO demonstrates remarkable capabilities for prompt optimization across various scenarios, including strong-to-weak optimization, self-optimization, and weak-to-strong optimization. This indicates the versatility of our framework. Notably, GPT-4 can consistently find better task prompts than GPT-3.5-turbo, which suggests the need for a capable model as the prompt optimizer.

The Efficiency of Optimization. LLM-based prompt optimization requires multiple interactions with the LLM. In this part, we investigate the efficiency of LLM-based prompt optimizers by examining the optimization curve over the first 12 steps. Figure 3 shows that on the movie recommendation dataset, compared to other methods, GPO demonstrates rapid enhancement of the task prompt in the early stage, followed by steady and consistent performance improvement in the later stage of optimization. Since GPO only utilizes task prompts to derive the update direction and performs fine-grained control over the variation, it can achieve better performance with high efficiency.

![](images/895d5aad33ea1d14971b2c578bcbb3920acdd7448263eaf30f986a3b67d123fb.jpg)  
Figure 3: The efficiency of our approach GPO w.r.t. optimization steps.

# Conclusion and Discussion

In this paper, we conduct a systematic analogy between gradient-based model optimizers and LLM-based prompt optimizers. Based on existing work and our newly proposed approach, we conduct an experimental analysis of the two key factors (i.e., update direction and update method) to determine the best configuration. According to this configuration, we further propose a novel prompt optimization framework GPO. At each step, it retrieves relevant prompts from the optimization trajectory as the update direction. Then, it utilizes the generation-based refinement strategy to perform the update, while controlling the edit distance through a cosine-based decay strategy. Extensive experiments demonstrate the effectiveness and efficiency of GPO.

One limitation of our work is that we only draw an analogy with the most widely used gradient-based optimizers. More advanced model optimizers like Newton’s method and its application to meta-prompts remain to be investigated. Additionally, our approach relies on textual update directions, future research could explore more direct and fine-grained numerical update signal methods (Nie et al. 2024).