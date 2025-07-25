# Automated Creation of Reusable and Diverse Toolsets for Enhancing LLM Reasoning

Zhiyuan $\mathbf { M } \mathbf { a } ^ { 1 }$ , Zhenya Huang1,2\*, Jiayu ${ { \bf { L i u } } ^ { 1 } }$ , Minmao Wang3, Hongke Zhao3, Xin Li1,4

1State Key Laboratory of Cognitive Intelligence, University of Science and Technology of China 2Institute of Artificial Intelligence, Hefei Comprehensive National Science Center 3College of Management and Economics, Tianjin University 4iFLYTEK AI Research {zhyma, jy251198}@mail.ustc.edu.cn, {wmmmm, hongke}@tju.edu.cn, {huangzhy, leexin}@ustc.edu.cn

# Abstract

Augmenting large language models (LLMs) with tools significantly enhances their problem-solving potential across multifaceted tasks. However, current tools automatically created by LLMs often serve as a mere summary of specific problems or solutions, which face two main issues: 1) Low reusability: The tools are overly problem-specific and struggle to handle new problems. 2) Limited diversity: The toolsets are too narrow, limiting their application to address a broader range of different problems. In this paper, we propose the Knowledge-grounded Tool Creation with Evolution (KTCE) framework, which aims to craft reusable and comprehensive toolsets for LLMs in a two-stage process. In the first stage (Knowledge-based Tool Creation), we conceptualize tools as a form of executable domain knowledge and propose a problem-knowledge-tool paradigm. Specifically, we leverage LLMs to abstract “knowledge” from “problems” and create a three-layer knowledge tree of topics, concepts, and key points. This hierarchical structure serves as a foundation for inducing atomic “tools” from “knowledge”, grounding them in fundamental concepts and enhancing their usability. In the second stage (Tool Evolutionary Search), we evolve the toolsets through several actions including tool selection, mutation, and crossover. This stage mimics the biological evolution process, aiding toolsets in discovering new tools or updating existing ones, thereby increasing the diversity of the toolset. Experiments on challenging mathematical/tabular/scientific reasoning tasks demonstrate that our approach achieves substantial accuracy improvements ranging from $6 . 2 3 \%$ to $1 8 . 4 9 \%$ on average. Moreover, in-depth analyses reveal the superior characteristics of our toolkit, including high reusability, high diversity, and high generalizability on cross-data/LLM performance with low complexity.

# Code — https://github.com/zhymma/KTCE

# 1 Introduction

Equipping LLMs with tools enables them to accomplish tasks beyond their native capabilities, making it a promising approach for solving highly complex problems (Qin et al. 2024a; Qu et al. 2024; Gao et al. 2024; Zhao et al.

Question 1: On a merry-go-round, a horse at 24 feet makes 32 revolutions.   
How many revolutions for a horse at 8 feet to cover the same distance?   
Question 2: A person walks 300 meters in 10 minutes. How many meters   
will the person walk in 25 minutes if they maintain the same speed?   
(a) Existing Methods   
def cal_revolution(radius1, rev1, radius2): def cal_distance(time1,dis1,time2): rate $\mathbf { \Sigma } = \mathbf { \Sigma }$ radius1 / radius2 rate $\ c =$ time2 / time1 rev2 $\ c =$ rate \* rev1 dis2 $\mathbf { \Sigma } = \mathbf { \Sigma }$ rate \* dis1 日 return rev2 return dis2   
# Tool Calling 1 # Tool Calling 2 Temporary!   
result $\scriptstyle | = |$ cal_revolution(24,32,8) result2=cal_distance(10, 300, 25)   
(b) KTCE (Ours) Topic Concept Key Points Algebra Ratios Length ratio … different quantities   
def prop_area_len(ratio, is_area=False): # Tool Calling 1 #Calculates area/length proportional result1=32\*prop_area_len((24, 8)) ratio $\mathbf { \Sigma } = \mathbf { \Sigma }$ ratio[0] / ratio[1] G if isinstance(ratio, tuple) else ratio # Tool Calling 2 Reusable! return ratio\*\*2 if is_area else ratio result2 $scriptstyle : = 3 0 0 ^ { \mathrm { : } }$ \*prop_area_len((25, 10))

2024; Luo et al. 2024; Gong and Sun 2024). As a popular type of tool, Python functions have been widely investigated to solve computational problems. For example, using SymPy enables LLMs to perform symbolic computations (e.g., calculus) (Meurer et al. 2017), while using Pandas allows LLMs to perform table-related operations (e.g., data filtering and aggregation) (Lu et al. 2024). Building on these, researchers have explored automatically generating Python functions as tools (Wang et al. 2024b; Cai et al. 2024) and further enhancing LLMs’ capabilities through calling them.

However, as illustrated in Table 1, existing tools automatically created by LLMs still remain flawed in two key aspects. (Qian et al. 2023; Yuan et al. 2024; Wang, Neubig, and Fried 2024). Firstly, they are over problem-specific. For Question 1 (revolution proportional problem) in Figure 1(a), the generated cal revolution tool can effectively solve Question 1 but fails to address the similar Question 2 (distance proportional problem), despite the fact that both of them require the same proportional knowledge to solve. Secondly, the toolsets often lack diversity, resulting in limited problem coverage. For example, tools created by TROVE (Wang, Neubig, and Fried 2024) can only be applied to less than $1 5 \%$ of problems in the MATH dataset (Hendrycks et al. 2021), significantly restricting their applicability.

Table 1: Comparison of characteristics across different automatic tool creation methods.   

<html><body><table><tr><td>Method</td><td>Reusability</td><td>Scale</td><td>Structure</td><td>Diversity</td><td>Optimization Capability</td></tr><tr><td>CREATOR</td><td>Low</td><td></td><td></td><td></td><td></td></tr><tr><td>CRAFT</td><td>Low</td><td>Large</td><td>Disorganized</td><td>Low</td><td>Low</td></tr><tr><td>TroVE</td><td>Low</td><td>Small</td><td>Disorganized</td><td>Limited</td><td>Low</td></tr><tr><td>KTCE (Ours)</td><td>High</td><td>Large</td><td>Organized</td><td>High</td><td>High</td></tr></table></body></html>

Behind these phenomena, we believe these issues of low tool reusability and limited toolset diversity stem from two critical oversights in current methods. Firstly, they neglect the importance of abstract knowledge in tool development. The essence of tools is a highly condensed knowledge. Therefore, the lack of explicit connection between tools and knowledge results in generated tools struggling to solve new problems that share the same knowledge points. Secondly, they completely rely on the problem itself to build static tools, neglecting the combination and expansion of these tools. Ultimately, these tools are limited in functionality and hard to be applied on different problems.

To address these challenges, in this paper, we propose a Knowledge-grounded Tool Creation with Evolution (KTCE) framework. It consists of two stages to automatically create reusable and diverse toolsets for enhancing LLMs’ reasoning capabilities. In the first stage (Knowledge-based Tool Creation), we recognize that a tool is a form of executable, distilled abstract knowledge. While problems vary, the underlying knowledge is often interconnected (Liu et al. 2023b). Therefore, we establish a problem-knowledge-tool paradigm, which goes beyond existing methods by inducing tools based on the fundamental knowledge, rather than the problems themselves. Specifically, we automatically construct a hierarchical knowledge tree that contains three levels through knowledge extraction and clustering: Topic (e.g., “Algebra” in Figure 1(b)), Concept (e.g., “Ratios”), and Key Points (e.g., “Length ratio ...”). Based on this, we create atomic tools at Key Points level (e.g., prop area len) for each foundational Topic-Concept domain. In the second stage (Tool Evolutionary Search), we attempt to dynamically optimize the tools to further enhance the diversity of the toolset. We draw inspiration from Darwin’s theory of evolution (Fisher 1999; Zhao et al. 2022) and Genetic Programming (GP) (Koza 1994) to implement tool evolution using LLMs by simulating biological genetics. Specifically, we design several actions including tool selection, mutation, and crossover to evolve the toolsets. To guide the evolutionary direction, we additionally design multiperspective metrics and an optimization function to dynamically adapt KTCE’s toolset. Finally, we built a KTCEaugmented Agent for practical reasoning, which leverages the generated toolset to solve problems.

We conduct experiments on 9 datasets across three challenging reasoning tasks: mathematics MATH (Hendrycks et al. 2021), table question-answering TabMWP (Lu et al. 2023), and science problems SCIBENCH (Wang et al.

2024a). We observe that KTCE consistently outperformed competitive baselines, improving reasoning accuracy by $6 . 2 3 \%$ to $1 8 . 4 9 \%$ . More importantly, our KTCE substantially improved the coverage to over $60 \%$ (compared to previous methods’ $1 4 . 7 8 \%$ ) and tripled tool usage frequency, indicating the creation of a highly reusable and widely diverse toolset. Further experiments validated that our toolset in KTCE achieved improvements in multiple aspects, including reduced solution complexity, and robust generalization across various LLM scales, architectures, and datasets.

# 2 Related Work

LLM Reasoning. Reasoning with LLM is a crucial benchmark for AI’s problem-solving abilities and a key milestone towards AGI (Sun et al. 2024; Wu et al. 2024; Cheng et al. 2024; Zhang et al. 2024a). Recent LLM advancements have led to increased research in this area (Hong, Pang, and Zhang 2024; Xiao et al. 2024; Liu et al. 2024; Ni et al. 2024b,a; Zhang et al. 2024b; Wang et al. 2024c), with approaches falling into two categories. The first is language-based reasoning, which employs step-by-step intermediate steps and logical connections expressed in natural language (such as CoT (Wei et al. 2022), ToT (Yao et al. 2024) and GoT (Besta et al. 2024)) and DeAR (Xue et al. 2024), while the second is program-based reasoning, which harnesses the power of programming languages (Shi et al. 2024a), particularly Python, and code interpreters to perform precise computations and logical operations (such as PAL (Gao et al. 2023), PoT (Chen et al. 2023), ToRA (Gou et al. 2024), MAMMOTH (Yue et al. 2024)). Our research aims to enhance LLMs’ reasoning by constructing tools and integrating them into code generation, enabling LLMs to tackle complex reasoning problems with executable domain knowledge.

Tool Utilization and Tool Creation. Tool utilization has gained significant attention for efficiently expanding LLM capabilities (Qin et al. 2024a; Shi et al. 2024b; Gou et al. 2024; Ma et al. 2024; Qin et al. 2024b; Schick et al. 2024). Tool-augmented LLMs significantly expand their capabilities by enabling information retrieval, complex computations, environmental interactions, efficiency improvements, and transparent reasoning processes. These models show promise in tackling complex problems similarly to human problem-solving strategies (Liu et al. 2023a). Given the scarcity and high costs of manual tools, researchers are focusing on how to automate the creation of high-quality tools to better assist LLMs in problem-solving. As shown in Table 1, current research can be categorized into two approaches: 1) Ad-hoc Tool Creation: CREATOR (Qian et al.

2023) generates temporary Python function tools for each problem. While this approach is straightforward, it shows low reusability as each tool is tightly coupled with a specific problem. 2) Toolset Formation: This approach includes VOYAGER (Wang et al. 2023), which develops Java tools through Minecraft interactions; CRAFT (Yuan et al. 2024), which abstracts Python tools from specific problem solutions, forming a large but disorganized toolset through validation and deduplication processes; and TROVE (Wang, Neubig, and Fried 2024), which iteratively samples and selects superior Python tools for a small toolbox with limited diversity. These methods typically focus on immediate problem-solving while neglecting the fundamental aspects of tool quality - reusability, scale, structure, diversity, and optimizability (as shown in Table 1). In contrast, our twostage KTCE creates a reusable, widely applicable toolset that enhances LLM accuracy through tool invocation.

# 3 Knowledge-grounded Tool Creation with Evolution Framework

# 3.1 Preliminaries

Given a reasoning task with training dataset $D _ { t r a i n } \ =$ $\{ ( p _ { i } , s _ { i } ) \} _ { i = 1 } ^ { N }$ , where $p _ { i }$ and $s _ { i }$ represent input problems (e.g., “What is $\mathbf { { b } + c ? } ^ { , , }$ ) and corresponding solutions (e.g., “To convert this...”), KTCE aims to create a toolset $\tau ^ { * }$ (we consider the Python functions as tools) based on $D _ { t r a i n }$ that can support LLMs’ reasoning on testing set $D _ { t e s t }$ . During this process, we have two primary objectives:

• Tool Reusability: Our tools can solve classes of problems rather than specific instances, which can be assessed by the frequency of calling each tool in $T ^ { * }$ on $D _ { t e s t }$ . • Toolset Diversity: Our tools can cover a wide range of tasks, which can be reflected by the proportion of problems in $D _ { \mathrm { t e s t } }$ that can be solved by $T ^ { * }$ .

To achieve these objectives, KTCE employs a two-stage framework as shown in Figure 2 and formally presented in Algorithm 1. Firstly, the Knowledge-based Tool Creation stage addresses reusability by generating initial toolset $\tau$ from domain knowledge tree $K T$ (Figure2(a)). Secondly, the Tool Evolutionary Search stage enhances diversity by iteratively expanding and optimizing $\tau$ to produce the final toolset $\tau ^ { * }$ (Figure 2(b)). These two stages directly correspond to the processes described in Sections 3.2 and 3.3, respectively. Finally, we design a KTCE-augmented Agent (Section 3.4) that enhances LLM’s reasoning capabilities using $\tau ^ { * }$ . Detailed discussions of reusability and diversity metrics are provided in Section 4.1.

# 3.2 Stage 1: Knowledge-based Tool Creation

In this stage, we created the toolset $T$ through three main steps: Knowledge Extraction, Knowledge Clustering, and Tool Creation, as shown in Figure 2(a). Different from previous methods where tool is used as a summary of the problem or solution (Qian et al. 2023; Yuan et al. 2024), we think that tools are abstracted domain knowledge that encapsulate problem-solving methods, transforming concepts

Algorithm 1: The KTCE Framework   
Input: Dataset $D$ , toolset size $k$ , max iter $N$   
Output: Optimized toolset $\tau ^ { * }$   
1: Stage 1: Knowledge-based Tool Creation   
2: $K \gets$ ExtractKnowledge $( D _ { \mathrm { t r a i n } } )$   
3: $T , C , K P \gets$ ClusterKnowledge $( K )$   
4: $\tau  \{ \}$   
5: for $( t _ { i } , c _ { j } ) \in T \times C$ do   
6: $\begin{array} { r l } & { { \dot { F _ { i j } } } \gets \mathrm { G e n I n i t T o o l s } ( \mathrm { G e t K P } ( t _ { i } , c _ { j } , T \times C \times K P ) ) } \\ & { \mathcal { T } \gets \mathcal { T } \cup \{ ( t _ { i } , c _ { j } , F _ { i j } ) \} } \end{array}$   
7:   
8: end for   
9: Stage 2: Evolutionary Search   
10: for each $( t _ { i } , c _ { j } )$ pair in $\tau$ do   
11: Initialize $\bar { T _ { i j } } , \bar { \mathcal { L } } ^ { \mathrm { b e s t } } , \bar { T } _ { i j } ^ { \mathrm { b e s t } }$ and metrics   
12: to $N _ { \mathrm { m a x } }$ do   
13: $\mathcal { L } , \mathrm { T E S , T C , T A } \gets \mathrm { T o o l E v a l } ( \mathcal { T } _ { i j } , D _ { \mathrm { t r a i n } } ^ { i j } , k )$   
14: if $\mathcal { L } < \mathcal { L } ^ { \mathrm { b e s t } }$ then   
15: $\mathcal { L } ^ { \mathrm { b e s t } } , \mathcal { T } _ { i j } ^ { \mathrm { b e s t } }  \mathcal { L } , \mathcal { T } _ { i j }$   
16: end if   
17: $\mathcal { T } _ { i j }  \mathrm { E v o l u t i o n } ( \mathcal { T } _ { i j } , \mathrm { T E S } , \mathrm { T C } , \mathrm { T A } )$   
18: end for   
19: Update $\tau$ with $( t _ { i } , c _ { j } , T _ { i j } ^ { \mathrm { b e s t } } )$   
20: end for   
21: return $\tau ^ { * }  \tau$

into executable actions. Therefore, we establish a problemknowledge-tool paradigm to transform domain knowledge into atomic tools, which are Python functions representing indivisible steps in the reasoning process.

Specifically, we first construct a domain knowledge tree $K T$ through knowledge extraction. For each problemsolution pair $( p _ { i } , s _ { i } )$ in $D _ { \mathrm { t r a i n } }$ , we use LLMs to extract related knowledge in XML format, representing it as $K =$ $\{ ( t o p i c , c o n c e p t , k e y p o i n t s ) \}$ . For instance, a triplet might be (“Algebra”, “Quadratic Equations”, “Solving equations using ...”) as shown in Figure 2(a).

These extracted knowledge elements are then consolidated into a three-level domain knowledge tree $K T$ :

$$
K T = T \cup C \cup K P ,
$$

where $\textit { T } = \ \{ t _ { 1 } , t _ { 2 } , . . . , t _ { L } \}$ , $\begin{array} { l l l } { { C } } & { { = } } & { { \left\{ c _ { 1 } , c _ { 2 } , . . . , c _ { N } \right\} } } \end{array}$ , and $K P = \{ k p _ { 1 } , k \bar { p _ { 2 } } , . . . , k p _ { M } \}$ are sets of topics, concepts, and key points respectively. Using the BGE-M3 model (Chen et al. 2024) for semantic representations, we employ Kmeans clustering (Lloyd 1982) to group similar elements: Multiple concepts (e.g., “Quadratic Equations” and “Functions”) are clustered into a topic (e.g., “Algebra”), and multiple key points are clustered into a concept. For each topicconcept pair, we cluster all associated key points to identify distinct subtasks, with the optimal number of clusters determined using the elbow method. Each key point cluster within a topic-concept pair represents a distinct subtask. For example, under the topic-concept pair “Algebra - Quadratic Equations”, subtasks might include solving equations and finding the sum and difference of roots by Vieta’s formula, as shown in Figure 2(a).

Following the knowledge tree construction, we then

(a) Stage 1: Knowledge-based Tool Craft (b) Stage 2: Tool Evolutionary Search   
(QuestioTnr,aSionliuntigonD):ataset 邮 StruKcntoedwlDedogmeain ToInoiltsieatl𝑻 ToollEMveatlruica:ti on Tool Evolution Final ∗   
fr :n3f,…2oTrSrheaeoWvlhoqhuoluatruitsaoieidosnran)at\$s:t.8ibcfH+eo\$ce\$xtw?t^om2-a "{,"CToonpciecp"t:"“":C"A" laQgluceaubdldruraas"t, 代物 ToEoSlset Metric: Mutation ……   
$2 0 \mathrm { x } + 3 6 \mathrm { \mathbb { S } }$ ① Knowledge Extract ： Crossover   
Solution: To convert this to the tation i"cKCEeoynqcueatpito" "s"Q,uadrat   
form \$(x+b)^2+c\$ … \$b + c = -74\$. tic Represe Cluster i"cKEeyquations" "PKoienyts_"P:"o"iSnotlsv"e:e"aSxo²lv LLM-based Tool Evolution Knowledge Tree 𝑲𝑻 Knowledge ebya x…² "+}bx + c = 0  …… Selection Remove! Algebra Quadratic Equations 电 Initial Toolset 𝑻  $\mathbf { \sigma } = \mathbf { \sigma }$   Mutataion     
Math $\mathbf { \sigma } = \mathbf { \sigma }$ … ③ Tool Craft   Calculus  $( t _ { 0 } , c _ { 0 } )$ Initial Toolset 𝑻 (𝒕𝒊, 𝒄𝒋)  $\mathbf { \sigma } = \mathbf { \sigma }$   “Algebra - Quadratic Equations” “Calculus - Limits”     
      
  中  $\mathbf { \sigma } = \mathbf { \sigma }$    
…… ： …  Crossover Vieta’s … $\mathbf { \sigma } = \mathbf { \sigma }$  “Th   $= -$  $\mathsf { x } =$ 'x‘)  ……   “Algebra - Quadratic Equations”

prompt LLM to create atomic tools solving the subtasks in each topic-concept pair. As illustrated in Figure 2(a), this process generates functions such as vietas formulas from coeff (for computing the sum and product of roots). The result is an initial toolset $\tau$ corresponding to $K T$ :

$$
\mathcal { T } = \bigcup _ { ( t _ { i } , c _ { j } ) \in T \times C } \{ ( t _ { i } , c _ { j } , F _ { i j } ) \} ,
$$

where $F _ { i j } = \{ f _ { 1 } , f _ { 2 } , . . . , f _ { M _ { i j } } \}$ is the set of funcs for topic $t _ { i }$ and concept $c _ { j }$ . Each function $f _ { k } \in F _ { i j }$ corresponds to a subtask solution based on $\mathrm { k p } _ { k }$ in $K P$ for the topic-concept pair $( t _ { i } , c _ { j } )$ . This structure ensures $\tau$ directly reflects $K P$ ’s organization, with each topic-concept pair having its own function set, and each function corresponding to a subtask.

This approach facilitates efficient management of largescale tool collections while maintaining the inherent structure of domain knowledge. The structured toolset $\tau$ , with its hierarchical organization mirroring the knowledge tree $K T$ , encompasses various knowledge points within the task domain. By breaking down complex reasoning tasks into atomic tools based on key points, our method enables flexible problem-solving reasoning.

# 3.3 Stage 2: Tool Evolutionary Search

In this stage, we completed the adaptive optimization of the initial toolset $\tau$ through iterative Tool Evaluation and Tool Evolution, resulting in the final toolset $\tau ^ { * }$ , as shown in Figure 2(b). Although the initial toolset $\tau$ from Stage 1 reflects the structure of domain knowledge, it may still have some limitations: 1) Lack of comprehensiveness: It may miss specialized tools for subtasks absent in the training set, limiting system’s versatility. 2) Potential errors and inadequacies: Due to LLMs’ inherent uncertainty, generated tools may have flaws. For instance, Figure 2(b) shows the vietas formulas from coeff function missing the $^ {  } a \ = \ 0 ^ { \ ' }$ case. Such edge cases and boundary conditions can lead to incorrect reasoning results, resulting in unstable performance when LLMs handle complex reasoning tasks.

Given that these issues are inherent to static toolsets, dynamic optimization becomes crucial for improving tool effectiveness. However, a unique challenge arises as tools, unlike vectorized parameters in deep learning models, are Python functions that cannot compute gradients. Inspired by evolutionary biology and Genetic Programming (GP) (Koza 1994), we observe that mutation and crossover in biological evolution closely match our desired toolset updates.

Based on this insight, we propose an Evolutionary Search method to simulate this optimization process in a nongradient manner, as shown in Figure 2(b). The main objective of this stage is to further enhance the diversity of the toolset. Specifically, we conceptualize tool $f _ { k }$ in $\tau$ as individuals, toolset $\tau$ as the initial population, and the dataset $D _ { \mathrm { t r a i n } }$ as the environment. We leverage LLMs to simulate this evolutionary process to increase the diversity of the initial population and improve practical reasoning performance. We iteratively perform Tool Evaluation (simulate natural selection) and Tool Evolution (simulate selection, crossover, and mutation in evolution operations), with the former guiding the latter and controlling the iterative process. Finally, we can obtain the optimized final toolset $\tau ^ { * }$ .

Tool Evaluation. The Tool Evaluation phase assesses the performance of the population $\tau$ on the environment $D _ { \mathrm { t r a i n } }$ . This evaluation provides insights into the toolset’s performance, forming the foundation for the subsequent Tool Evolution phase. We use metrics and an optimization function that collectively reflect the adaptability of tools and toolset to the problem-solving environment, analogous to the fitness of individuals and populations in biological evolution.

To begin the evaluation process, for each $F _ { i j }$ with topicconcept pairs $( t _ { i } , c _ { j } )$ , we employ LLM to generate Python solutions by calling fk in Fij for sampled problems in Dtirjai By executing these solutions, we obtain the accuracy of each program and the usage and accuracy of each tool.

We then quantify performance using several metrics. For individual tools, we measure: 1) Tool Invocation Frequency $( \mathrm { T I F } _ { f _ { k } } )$ : represents the number of times $f _ { k }$ is called. 2) Tool Successful Invocation Frequency $( \mathrm { T S I F } _ { f _ { k } } )$ ): represents the number of times $f _ { k }$ is called and produces accurate results.

These metrics allow us to calculate a Tool Effectiveness Score $( \mathbf { T E S } _ { f _ { k } } )$ , defined as:

$$
\mathrm { T E S } _ { f _ { k } } = 1 - \frac { \mathrm { T S I F } _ { f _ { k } } } { \operatorname* { m a x } ( \mathrm { T I F } _ { f _ { k } } , 1 ) } ,
$$

where a lower score indicates better performance of $f _ { k }$

For the toolset level, we evaluate: 1) Toolset Coverage (TC): represents the proportion of problems that using $\tau$ to reason, i.e., the task coverage of $\tau$ for $D _ { \mathrm { t r a i n } } . 2 \rangle$ ) Task Accuracy (TA): measures the accuracy of reasoning using $\tau$ .

To guide the optimization process effectively, we introduce an optimization function $\mathcal { L }$ that integrates our three main objectives: tool reusability, toolset diversity, and ensuring reasoning accuracy. The function is defined as:

$$
\begin{array} { r l r } {  { \mathcal { L } = \alpha ( \sum _ { f _ { k } \in F _ { i j } } \mathrm { T E S } _ { f _ { k } } + \operatorname* { m a x } ( 0 , k - n ) ) } } \\ & { } & { + ( \beta ( 1 - \mathrm { T C } ) + \gamma ( 1 - \mathrm { T A } ) ) \times | D _ { \mathrm { t r a i n } } ^ { i j } | } \\ & { } & { + \delta ( n - k ) , } \end{array}
$$

where $n$ is the current toolset size, $k$ is the desired size, and $\alpha , \beta , \gamma , \delta$ are weight coefficients. $D _ { \mathrm { t r a i n } } ^ { i j }$ represents the subset of training dataset problems corresponding to the topicconcept pair $( t _ { i } , c _ { j } )$ .

By using these metrics and the comprehensive optimization function, we can effectively evaluate the performance of both individual tools and the entire toolset, providing crucial guidance for the subsequent Tool Evolution phase.

Tool Evolution. This phase leverages LLMs to implement three key mechanisms analogous to biological evolution (Fisher 1999; Koza 1994): selection, crossover, and mutation. These operations are applied to the toolset $\tau$ based on the metrics and feedback obtained during the Tool Evaluation phase. Crucially, each operation is executed through carefully crafted prompts to the LLM, enabling it to generate updated Python functions in the toolset. The bottom part of Figure 2(b) demonstrates examples of the three operations.

Selection: Mimicking natural selection where only the fittest individuals reproduce, we prompt the LLM to retain effective tools and remove ineffective ones by analyzing the metrics and runtime information. For example, the tool form quadratic from roots was rarely called during evaluation, resulting in a low T ES. This indicates its presence is more noise than signal, leading to its removal.

Mutation: To ensure the reasoning accuracy of $\tau$ , we prompt the LLM to modify the function code by adjusting parameters, expanding functionality, and improving error handling to enhance adaptability. These targeted changes help explore the search space more thoroughly. For instance, in the tool vietas formulas from coeff, error handling for edge cases was incorporated.

Crossover: To expand the diversity of $\tau$ , we generate new Python functions by combining features from two or more selected tools to create potentially more effective variants. We input the complete $F _ { i j }$ in $\tau$ , current TC scores, and uncovered subtasks, prompting the LLM to output new tools. For example, by combining two existing tools solve quadratic equation and vietas formulas from coeff, a more powerful new tool solve and verify quaratic can be created.

Throughout this evolutionary process, the optimization function $L$ in Eq. (4) serves as a guiding metric. The goal is to minimize $L$ by balancing tool usability, toolset coverage, and accuracy. If $L$ does not decrease, a rollback mechanism is implemented to revert ineffective changes. Additionally, we employ early stopping when $L$ shows no significant improvement over several iterations, ensuring computational efficiency. The process continues until either the stopping criterion is met or the maximum iterations are reached, resulting in the final optimized toolset $T ^ { * }$ .

Our Evolutionary Search stage effectively addresses the initial toolset’s limitations through LLM-driven iterative optimization. It systematically mirrors the biological evolution process to efficiently explore the solution space for enhancing both toolset diversity and reasoning accuracy. Moreover, this novel approach demonstrates the significant potential of combining LLM with evolutionary algorithms to create robust, adaptive, and self-improving agents.

# 3.4 KTCE-augmented Agent

After completing the two-stage process, we obtain the final optimized toolset $\tau ^ { * }$ . For practical reasoning, we develop a KTCE-augmented Agent (KA) to utilize tools from $\tau ^ { * }$ for generating code solutions for input problems. The KA’s process can be conceptualized as:

$$
A = { \mathcal { M } } ( p ; T ^ { * } , H )
$$

where $A$ is the final standardized answer, $p$ is the input problem, $T ^ { * }$ is the optimized toolset, $H$ represents historical usage and error experiences, and $\mathcal { M }$ is the KA’s problemsolving process.

Specifically, the KA operates through three key phases: Tool Retrieval, Solution Generation, and Result Formatting. In Tool Retrieval, it identifies the relevant $( t _ { i } , c _ { j } )$ pair to access a targeted toolset $F _ { i j }$ from $T ^ { * }$ , refined by historical data HH to ensure suitable tool selection. During Solution Generation, it leverages LLM to generate Python code that directly calls the retrieved tools, incorporating examples from HH through In-Context Learning (Dong et al. 2022). Finally, in Result Formatting, it executes the code to obtain intermediate results and transforms them into a standardized answer format $A$ for consistent evaluation. This structured approach enhances efficiency by directly accessing relevant tools without complex retrieval pipelines.

# 4 Experiments

# 4.1 Experimental Setup

Dataset and Evaluation We select three challenging reasoning tasks for evaluation.

• Mathematical Reasoning: We use the MATH dataset (Hendrycks et al. 2021) to test LLMs’ text-based numerical reasoning. It has 7,500 training and 5,000 test problems across 7 categories of competition-level questions. • Tabular Reasoning: The TabMWP dataset (Lu et al. 2023) is employed to assess LLMs’ capability in processing structured tabular data and performing reasoning calculations. It includes 38,431 problems and 37,644 tables, with a test set of 1,000 problems. • Scientific Reasoning: We utilize SCIBENCH dataset (Wang et al. 2024a) to examine numerical reasoning abilities in complex scientific contexts. This dataset contains 695 college-level problems from physics, chemistry, and mathematics. We randomly select 100 problems for testing and used the remainder for training.

Our evaluation incorporates a diverse range of answer types, including integers, multiple-choice questions, boolean values, mathematical expressions, and lists. This approach allows for a more comprehensive assessment of LLM reasoning capabilities across various problem formats.

We use answer accuracy (Acc) as the primary metric for evaluating model reasoning ability. Additionally, we employ task coverage (Cov), which describes the proportion of problems in the task that utilize tools. We also consider tool usage frequency (# Freq), which represents the average number of times each tool in the toolset is called. Finally, we measure toolset size (# T-size), which describes the number of functional tools in the toolset. These metrics collectively provide a comprehensive assessment of both the model’s performance and the effectiveness of the toolset.

Baselines We compare KTCE with baseline methods:

• Chain-of-Thought $( \mathbf { C o T } )$ (Wei et al. 2022): Uses LLM to generate step-by-step thinking processes and logical relationships in natural language without tools. • Program-of-Thought $( \mathbf { P o T } )$ (Chen et al. 2023): Utilize LLM to generate code to perform calculations and reasoning and run it through an external code interpreter. • Library (Meurer et al. 2017): Augments PoT with external Python library functions (e.g., SymPy, Scipy). • Wolfram: Enhances PoT by calling WolframAlpha API in Python for computational knowledge. • Creator (Qian et al. 2023): Employs temporary tool creation for each problem, then generate solution code. • Creator (SR) (Qian et al. 2023): Enhances Creator with self-refinement based on execution feedback. • CRAFT (Yuan et al. 2024): Implements toolset formation by abstracting, validating, and deduplicating tools. Uses LLM for query generation and tool retrieval.

• TROVE (Wang, Neubig, and Fried 2024): Utilizes multiple sampling for tool creation, generating various tools and solutions and form a compact toolbox.

Implementation We implement KTCE, and all baselines using GPT-3.5-Turbo for its cost-effectiveness, speed, and code generation capabilities. In Section 3.2, we extract max 3 knowledge triplets $K$ per problem. For Section 3.3, we sample up to 100 problems per topic-concept pair, with max 5 iterations. Early stopping is applied if L didn’t decrease for 3 consecutive iterations. Desired toolset size k is 10. For KA (in Section 3.4) and all baselines, to better utilize the tools, we allow up to three sampling attempts until the solution code successfully compiles.

# 4.2 Main Results

Table 2 presents the reasoning accuracy across all datasets. Overall, KTCE generates the most accurate solutions for three tasks, demonstrating the effectiveness of KTCEgenerated toolset and KA. For the seven types of MATH dataset, KTCE improves accuracy by $1 . 3 8 \%$ to $1 1 . 9 7 \%$ compared to TROVE, demonstrating how our knowledge-based tools can be effectively reused across different mathematical problem types. Additionally, it outperforms baseline using Python Library by up to a maximum of $1 0 . 2 \%$ . These results demonstrate KTCE’s superior capability in creating high-quality mathematical tools, enabling LLMs to effectively tackle competition-level problems.

In tabular reasoning, KTCE is the only method to exceed $90 \%$ accuracy. For the more challenging scientific reasoning domain, it improves LLM performance by $6 \%$ compared to PoT where most other methods underperform PoT. These results demonstrate KTCE’s unique advantage in leveraging domain knowledge - while tabular reasoning benefits from well-structured table operations, scientific reasoning showcases how KTCE effectively captures complex subject knowledge through our knowledge-grounded approach. This overcomes the limitation of previous methods in creating reusable tools across different tasks.

Building upon these observations, the experimental results in Table 3 further demonstrate KTCE’s capability in creating reusable and diverse tools. The high coverage rates $( 6 4 . 5 0 \%$ , $8 2 . 9 0 \%$ , and $6 2 . 0 0 \%$ across three datasets) and increased tool usage frequency validate that our knowledgebased tool creation (Stage 1) successfully addresses the reusability limitation of previous methods by grounding tools in fundamental domain knowledge. Meanwhile, the larger and more diverse toolset compared to CRAFT confirms that our evolutionary search (Stage 2) effectively expands tool diversity, overcoming the narrow toolset problem in existing approaches.

In summary, KTCE addresses the core challenges through two-stage framework: knowledge-based creation enables tool reusability by grounding in domain knowledge, and evolutionary search ensures toolset diversity through systematic optimization. These advantages provide a solid foundation for enhancing LLMs with high-quality tools.

Table 2: GPT-3.5-Turbo reasoning accuracy on three challenging reasoning datasets $( \% )$ . Bold values indicate the highest scores, underlined values indicate the second highest. ∗indicates statistical significance $( p < 0 . 0 5 )$ .   

<html><body><table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td colspan="7"></td><td rowspan="2">TabMWP</td><td rowspan="2">SCIBENCH</td></tr><tr><td>Alg</td><td>Count</td><td>Geo</td><td>MATH</td><td>Num</td><td>Pre.Alg</td><td>Pre.Cal</td></tr><tr><td rowspan="2">Basic</td><td>CoT</td><td>49.12</td><td>29.75</td><td>22.34</td><td>14.62</td><td>33.33</td><td>53.85</td><td>16.85</td><td>73.50</td><td>27.00</td></tr><tr><td>PoT</td><td>48.36</td><td>43.88</td><td>31.32</td><td>18.27</td><td>52.22</td><td>65.10</td><td>20.33</td><td>74.70</td><td>31.00</td></tr><tr><td rowspan="2">Tool-Aug.</td><td>Woiram</td><td></td><td></td><td></td><td></td><td>57.22</td><td>68.66</td><td>22.16</td><td></td><td></td></tr><tr><td></td><td>55.89</td><td>51.90</td><td>33.40</td><td>29.99</td><td></td><td></td><td></td><td>78.20</td><td>30.00</td></tr><tr><td rowspan="4">Tool Creation</td><td>Creator</td><td>34.29</td><td>43.04</td><td>25.47</td><td>24.81</td><td>38.52</td><td>43.86</td><td>21.06</td><td>84.90</td><td>27.00</td></tr><tr><td>Creator (SR)</td><td>50.38</td><td>48.73</td><td>28.18</td><td>28.90</td><td>48.70</td><td>62.34</td><td>19.78</td><td>87.80</td><td>32.00</td></tr><tr><td>CRAFT</td><td>53.33</td><td>42.62</td><td>22.96</td><td>25.25</td><td>41.67</td><td>69.35</td><td>20.51</td><td>77.00</td><td>28.00</td></tr><tr><td>TROVE</td><td>57.03</td><td>52.00</td><td>30.06</td><td>26.02</td><td>45.93</td><td>66.02</td><td>19.96</td><td>65.00</td><td>25.00</td></tr><tr><td>Ours</td><td>KTCE</td><td>69.00*</td><td>53.38*</td><td>40.29*</td><td>29.90*</td><td>57.96*</td><td>73.02*</td><td>31.68*</td><td>90.00*</td><td>37.00*</td></tr></table></body></html>

Table 3: Comparison of toolsets created by various methods.   

<html><body><table><tr><td>Method</td><td>Metric</td><td>MATH</td><td>TabMWP</td><td>SCIBENCH</td></tr><tr><td rowspan="2">CRAFT</td><td>Cov</td><td>19.10%</td><td>30.50%</td><td>6.00%</td></tr><tr><td>#T-size #Freq</td><td>1.02 936</td><td>1.69 180</td><td>0.27 22</td></tr><tr><td rowspan="2">TROVE</td><td>Cov</td><td>14.78%</td><td>55.00%</td><td>20.00%</td></tr><tr><td>#Freq #T-size</td><td>0.56</td><td>1.47</td><td>0.37</td></tr><tr><td rowspan="2">KTCE</td><td>Cov</td><td>1347 64.50 %</td><td>399 82.90 %</td><td>65</td></tr><tr><td>#Freq</td><td>3.10</td><td></td><td>62.00%</td></tr><tr><td rowspan="2"></td><td>#T-size</td><td></td><td>4.37</td><td>0.45</td></tr><tr><td></td><td>1317</td><td>222</td><td>199</td></tr></table></body></html>

Table 4: Ablation study results on the MATH dataset.   

<html><body><table><tr><td>Method</td><td>Acc</td><td>Cov</td><td>#Freq</td><td># T-size</td></tr><tr><td>KTCE</td><td>53.14%</td><td>64.50 %</td><td>3.10</td><td>1317</td></tr><tr><td>w/o Stage 1</td><td>50.16%</td><td>9.16%</td><td>5.99</td><td>78</td></tr><tr><td>w/o Stage 2</td><td>50.50%</td><td>57.92%</td><td>2.14</td><td>1612</td></tr><tr><td>w/o Sel</td><td>51.72%</td><td>65.10%</td><td>2.04</td><td>1858</td></tr><tr><td>w/o CO</td><td>50.62%</td><td>59.86%</td><td>2.85</td><td>1278</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o Mut</td><td>51.56%</td><td>64.42%</td><td>2.65</td><td>1402</td></tr></table></body></html>

# 4.3 Ablation Study

We conduct ablation experiments on the MATH dataset to evaluate each KTCE component. From Table 4, we observe that removing entire stages (“w/o Stage $1 ^ { \circ }$ and “w/o Stage $2 ^ { \overrightarrow { \mathbf { \alpha } } }$ ) decreases accuracy. Eliminating Stage 1 (Knowledgebased Tool Creation) drastically reduces the size and diversity of toolsets (only 78 tools with $9 . 1 6 \%$ coverage), indicating that the structured domain knowledge in Stage 1 is crucial for creating a comprehensive toolset. Removing Stage 2 (Tool Evolutionary Search) leads to decreased coverage and tool usage frequency despite an enlarged toolset, demonstrating that evolutionary optimization is necessary to ensure toolset effectiveness.

We then ablate the three operations in Stage 2: selection $( ^ { 6 6 } \mathrm { w } / \mathrm { o } ~ \mathrm { S e l } ^ { 3 } )$ , crossover $( ^ { 6 6 } \mathrm { w } / \mathrm { o } \mathrm { C O } ^ { 3 } )$ , and mutation (“w/o Mut”). Each operation proves essential for toolset quality:

Table 5: Complexity of Python solutions in the MATH dataset for different methods.   

<html><body><table><tr><td>Metric Method</td><td>CC↓</td><td>HV↓</td><td>HE↓</td></tr><tr><td>PoT PoT with Library</td><td>2.32</td><td>66.36</td><td>247.83</td></tr><tr><td></td><td>1.83</td><td>64.85</td><td>190.82</td></tr><tr><td>CRAFT</td><td>1.50</td><td>78.61</td><td>231.93</td></tr><tr><td>TROVE</td><td>1.57</td><td>82.79</td><td>235.88</td></tr><tr><td>KTCE</td><td>1.49</td><td>44.10</td><td>119.71</td></tr></table></body></html>

1) Removing selection leads to redundant, ineffective tools, shown by decreased accuracy despite more tools. 2) Without crossover, new tools can’t be generated through combination, limiting toolset diversity. 3) Removing mutation prevents necessary tool updates, reducing adaptation capability. These results demonstrate how each component contributes to KTCE’s success - Stage 1 ensures comprehensive tool creation through knowledge structuring, while Stage 2’s operations maintain toolset quality and diversity.

# 4.4 KTCE Analysis

Code Complexity Analysis Tools crucially allow LLMs to avoid generating complete solution code from scratch. To evaluate KTCE’s performance in further simplifying reasoning, we compare the complexity of solution codes generated by KTCE and baselines on the MATH dataset. We employ three established code evaluation metrics:

• Cyclomatic Complexity (CC): Measures linearly independent paths in the source code.   
• Halstead Volume (HV): Measures program size based on operands, operators, and length.   
• Halstead Effort (HE): Estimates cognitive load for designing, writing, and understanding the program (McCabe 1976; Halstead 1977).

As shown in Table 5, KTCE produces solutions with the lowest complexity across all metrics. The reduced Cyclomatic Complexity indicates simpler control flow structures, while lower Halstead Volume and Effort suggest more concise and comprehensible solutions. These improvements stem from KTCE’s ability to create reusable tools that encapsulate common problem-solving patterns, allowing LLMs to leverage accumulated knowledge rather than regenerating similar code repeatedly.

![](images/b0dc1bff1ad3ef87f2740055bb1f96ec008c1a1e7326a61f46bfbba8106f11fb.jpg)  
Figure 3: Comparison of Reasoning Accuracy of TROVE and KTCE w.r.t. Difficulty Level in the MATH dataset.

Table 6: Performance of KTCE toolset when generalized to different LLMs and datasets.   

<html><body><table><tr><td>Model</td><td>Dataset</td><td>Method</td><td>Accuracy</td></tr><tr><td>GPT-3.5-Turbo</td><td>MATH → GSM8K</td><td>PoT KTCE</td><td>78.62% 81.96 %</td></tr><tr><td>GPT-3.5-Turbo→ GPT-4o-Mini</td><td>MATH</td><td>PoT KTCE</td><td>69.94% 74.00%</td></tr><tr><td>GPT-3.5-Turbo -→ DeepSeek-Coder</td><td>SCIBENCH</td><td>PoT KTCE</td><td>30.00% 35.00 %</td></tr></table></body></html>

By building a comprehensive and reusable toolset through systematic knowledge extraction, KTCE effectively enables LLMs to focus on high-level reasoning rather than repetitive code generation. This knowledge-grounded approach significantly reduces solution complexity while substantially enhancing reasoning efficiency, as LLMs can concentrate on crucial problem-solving steps while reliably utilizing welltested tools for implementation.

Reasoning Across Difficulty Levels To assess KTCE’s robustness across difficulty levels, we analyze its performance on problems of varying complexity, as provided by the MATH dataset. Figure 3 shows the increase in correct reasoning instances (bars) and accuracy growth rate (line) relative to PoT. KTCE outperforms TROVE at all levels, with notable improvements on challenging problems. These results validate KTCE’s effectiveness in enhancing reasoning capabilities across difficulty levels, particularly excelling at complex problems through its well-designed toolset.

To explore KTCE’s generalizability, we apply the toolset created by GPT-3.5-Turbo across LLMs and datasets. From Table 6, we first observe that KTCE’s knowledge-grounded tools, created from MATH dataset, effectively transfer to GSM8K with a $3 . 3 4 \%$ accuracy improvement (Cobbe et al. 2021). This cross-dataset success validates that our knowledge-based tool creation approach successfully captures fundamental mathematical concepts, enabling tools to generalize across different mathematical tasks. Secondly, to evaluate the impact of the constructed toolset on different models, we provide GPT-3.5-Turbo’s tools to GPT-4oMini and DeepSeek-Coder on two datasets. When applied to GPT-4o-Mini on MATH, the accuracy improves by $4 . 0 6 \%$ , indicating potential across different model scales (OpenAI 2024). For DeepSeek-Coder on SCIBENCH, the accuracy rises by $5 . 0 0 \%$ . These findings demonstrate the cross-model robustness and generalizability of our approach.

These comprehensive results demonstrate KTCE’s unique ability to create a reusable and diverse toolset that significantly enhances reasoning across diverse LLMs and domains without fine-tuning, effectively highlighting its versatility and wide applicability.

# 5 Conclusion and Future Work

In this paper, we have introduced Knowledge-grounded Tool Creation with Evolution (KTCE), a two-stage framework for creating reusable and diverse toolsets for LLMs. By combining Knowledge-based Tool Creation with Tool Evolutionary Search, KTCE substantially improved tool reusability and diversity, breaking through the limitations of previous methods and laying the foundation for large-scale LLM tool utilization. Experiments across challenging reasoning tasks demonstrated its effectiveness, outperforming baselines in accuracy and tool quality. The resulting toolset enhanced LLMs’ problem-solving capabilities while reducing solution complexity. KTCE’s approach was analogous to learning explicit knowledge from tasks, preserving model integrity while enabling easy extension to different LLMs. Moreover, the tool evolution process enabled the combination of different tools, discovering connections between various knowledge domains. In the future, we plan to explore more advanced tools such as multi-modal and embodied AI tools, and design better agent workflows for tool utilization. These advancements will contribute to KTCE’s continued development in enhancing LLMs’ capabilities across diverse tasks.