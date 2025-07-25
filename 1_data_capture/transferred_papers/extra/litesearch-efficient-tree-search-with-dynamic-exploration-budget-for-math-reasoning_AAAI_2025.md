# LiteSearch: Efficient Tree Search with Dynamic Exploration Budget for Math Reasoning

Ante Wang1,3, Linfeng $\mathbf { S o n g ^ { 2 * } }$ , Ye Tian2, Baolin $\mathbf { P e n g } ^ { 2 }$ , Dian $\mathbf { Y } \mathbf { u } ^ { 2 }$ , Haitao $\mathbf { M } \mathbf { i } ^ { 2 }$ , Jinsong $\mathbf { S } \mathbf { u } ^ { 1 , 3 , 4 * }$ , Dong $\mathbf { Y } \mathbf { u } ^ { 2 }$

1School of Informatics, Xiamen University, China 2Tencent AI Lab, Bellevue, WA 3Shanghai Artificial Intelligence Laboratory, China   
4Key Laboratory of Multimedia Trusted Perception and Efficient Computing, Ministry of Education of China, Xiamen University, China   
wangante $@$ stu.xmu.edu.cn, lfsong $@$ global.tencent.com, jssu $@$ xmu.edu.cn

# Abstract

Recent research suggests that tree search algorithms (e.g. Monte Carlo Tree Search) can dramatically boost LLM performance on complex mathematical reasoning tasks. However, they often require more than 10 times the computational resources of greedy decoding due to wasteful search strategies, making them difficult to be deployed in practical applications. This study introduces a novel guided tree search algorithm with a goal-directed heuristic function and node-level exploration budget (maximum number of children) calculation to tackle this issue. By considering the search progress towards the final answer (history) and the guidance from a value network (future) trained without any step-wise annotations, our algorithm iteratively selects the most promising tree node before expanding it within the boundaries of the allocated computational budget. Experiments conducted on the GSM8K, TabMWP, and MATH datasets demonstrate that our method not only offers competitive performance but also enjoys significantly lower computational costs compared to baseline methods.

# 1 Introduction

Mathematical reasoning tasks (Amini et al. 2019; Cobbe et al. 2021; Hendrycks et al. 2021; Lu et al. 2022) have long been acknowledged as challenging. These tasks require transforming a question into a sequence of reasoning steps, which are subsequently executed to derive the correct answer. Recently, large language models (LLMs, Achiam et al. 2023; Touvron et al. 2023; Jiang et al. 2024) have demonstrated remarkable potential in addressing them. A pivotal approach is the employment of Chain-of-Thought (CoT) prompting (Wei et al. 2022; Kojima et al. 2022), which prompts LLMs to break down a question solution into a sequence of reasoning steps before reaching an answer.

Despite their impressive capabilities, LLMs still face challenges when tackling problems with increasing reasoning steps due to the nature of auto-regressive decoding. This can be analogous to the “System 1” mode of thought in psychology (Daniel 2017), which is characterized by fast, intuitive, but error-prone thinking. Much of recent work has focused on enhancing the “System 1” capability of LLMs by prompt-engineering, such as hierarchical prompting (Suzgun and Kalai 2024; Zeng et al. 2023) and automatic prompt refinement (Madaan et al. 2024; Yang et al. 2024; Zhou et al. 2024). On the other hand, growing research attention is being paid to promote the “System $2 ^ { \circ }$ mode of thought (Daniel 2017) for LLMs. It is characterized by deliberative thinking steps with back-and-forth refinements, which can be essential for solving complex math reasoning tasks. Particularly, prior efforts have studied enhancing LLMs both at inference time and through self-improvement using tree search algorithms, such as Breath-first Search (BFS, Khalifa et al. 2023; Yao et al. 2024; Xie et al. 2024; Zhu et al. 2024) and Monte Carlo Tree Search (MCTS, Feng et al. 2023; Tian et al. 2024; Zhang et al. 2024; Wang et al. 2024c).

![](images/0f2f69208af12adeacebfd13c528e924c9deee2bf7b02760935dfe5180d5499d.jpg)  
Figure 1: Comparison among ours and typical methods on GSM8K, where DFS, BFS, $\mathbf { M C T S ^ { 1 } }$ , and LiteSearch are only guided by the same value network. Hard Voting is also known as Self-Consistency (Wang et al. 2022) and Soft Voting additionally weights each trajectory using its value score. We measure the number of generated tokens (#Tokens $( k ) _ { , } ^ { \dag }$ ) by the LLM as computation costs.

However, these approaches often necessitate extensive manual labeling (Ma et al. 2023; Kang et al. 2024), making them difficult to be adopted in different settings. Moreover, they are computationally intensive, especially when tackling problems that require numerous logical steps (Xie et al. 2024). This is because these methods ineffectively manage the expansion budget (the number of nodes to expand) throughout the search process. As a typical example, BFS adopts a constant budget size throughout the search process, overlooking the fact that some tree nodes do not require much exploration. Some MCTS approaches (Tian et al. 2024) take adaptive budget sizes based on the importance of each node. Nevertheless, they still require a large number of simulations or rollouts for accurate statistics to make decisions, and they overlook other important information, such as the depth (progress) of each node. As a result, there is a pressing need to develop more efficient and adaptable methods for enhancing LLMs’ “System 2” reasoning capabilities to effectively handle complex reasoning tasks.

In this study, we introduce a guided tree search algorithm with dynamic node-level exploration budget calculation, aiming to maintain the performance at a moderate cost. Concretely, we utilize the value score from a value network (Tian et al. 2024) and the progress toward the final answer as heuristic guidance to select the most promising node from the search tree. Then, we expand this node within a dynamically computed budget size, effectively navigating the balance between exploration and exploitation for guided tree search. We continue iterating operations of selection and expansion until the resulting trajectory either meets the expected quality score or surpasses the maximum number of iterations. Notably, the computational budget for each node is inversely correlated to its value score. This is inspired by the observation that nodes with higher value scores are more likely to yield the correct solutions upon expansion, hence we allocate fewer computational resources for them to prevent unnecessary computation and vice versa. This not only promotes efficient exploitation, facilitating a faster convergence to the final answer, but also guarantees sufficient exploration to cover enough state space for maintaining performance.

We conduct experiments on the popular GSM8K (Cobbe et al. 2021), TabMWP (Lu et al. 2022), and MATH (Hendrycks et al. 2021). Results show that our method offer competitive performance but significantly less computation costs compared to other baselines. Detailed analyses confirm the usefulness of each component and provide more practical options for various settings. Additionally, we identify the limitations of this research line and suggest possible ways to tackle them.

# 2 Related Work

Thanks to the strong capabilities of LLMs, significant advancements have been made in mathematical reasoning tasks, surpassing traditional approaches that rely on semantic parsing (Matsuzaki et al. 2017; Hopkins et al. 2017) or Abstract Syntax Tree (AST) decoding (Li et al. 2019; Qin et al. 2021).

Some studies improved the reasoning capabilities of LLMs through further training. These efforts involve either manually annotating or automatically generating feasible and challenging problems to fine-tune LLMs (Luo et al. 2023; Yu et al. 2023; Liu and Yao 2024; Huang et al. 2024), as well as devising sophisticated techniques, such as reinforcement learning, for efficient training (Luo et al. 2023; Wang et al. 2023; Lightman et al. 2023; Chen et al. 2024).

Another line of research focused on inference-time improvement. Except for prompting engineering (Kojima et al. 2022; Liu et al. 2023) and the popular self-consistency (Wang et al. 2022, 2024a,b), most of these studies treat this task as a tree search problem and investigate various searching algorithms. Yao et al. (2024) were the first to introduce Tree-of-Thought (ToT), incorporating Depth-first Search (DFS) and Breath-first Search (BFS) to address reasoning problems. Some researchers (Khalifa et al. 2023; Zhu et al. 2024; Xie et al. 2024) applied step-wise Beam Search to math problems, which operates similarly to BFS under certain parameter conditions. To guide the search process, these studies either directly prompt LLMs to evaluate the quality of each step (Yao et al. 2024; Xie et al. 2024), or train a verifier on corresponding datasets to achieve better performance (Khalifa et al. 2023; Zhu et al. 2024).

Later studies delved into other sophisticated search algorithms, such as Monte Carlo Tree Search (MCTS, Tian et al. 2024; Zhang et al. 2024; Wang et al. 2024c), $\mathbf { A } ^ { * }$ (Ma et al. 2023), and Levin Tree Search (Kang et al. 2024). Nonetheless, these approaches necessitate more robust verifiers to steer the search procedure. Concretely, Tian et al. (2024) utilized a blend of the value function, Process-supervised Reward Model (PRM), and Outcome-supervised Reward Model (ORM). Ma et al. (2023) and Kang et al. (2024) train their PRM models on PRM800K (Lightman et al. 2023), which offers manual annotations for $8 0 0 k$ reasoning steps of problems from MATH (Hendrycks et al. 2021).

This study also follows the same research line, yet it concentrates on developing an efficient algorithm to decrease computation costs while maintaining performance. Besides, we employ a naive but more practical value network as the verifier, which is trained solely with the final answer labels as distant supervision.

# 3 Our Method

In this section, we first introduce LiteSearch (§3.1), an efficient tree search algorithm designed to maintain model performance at minimum computation costs. Then, we describe the value network (§3.2), which is easy to obtain and effectively guides the tree search process.

# 3.1 Guided Tree Search Algorithm

Taking the solving of each math reasoning question $q$ as a tree search problem, we initialize the root of the search tree with question $q$ , while the other tree nodes represent reasoning steps (e.g., $s _ { i } \big |$ ) generated by an LLM (denoted as

# Algorithm 1: LiteSearch

<html><body><table><tr><td>Input: question q Parameter:policyπ,valuenetworkv", thresholdε maximumiterationnumber N</td></tr><tr><td>Output: solution y 1:Initialize tree T with q as the root 2:i←O,y←null</td></tr><tr><td>3:whilei<Ndo 4: Select the node s' from T using Eq. 2</td></tr><tr><td>5: Expand s'to obtain itschild nodes C under</td></tr><tr><td>the budget constraint b computed by Eq.3 6: forc∈Cdo</td></tr><tr><td>7: s ← return_trajectory(T,c)</td></tr><tr><td>8: if is_terminal(s) and v"(s) >v"(y) then</td></tr><tr><td>9: y←s</td></tr><tr><td>10: end if</td></tr><tr><td>11: end for</td></tr><tr><td>12: if v"(y)>εthen</td></tr><tr><td>13: break</td></tr><tr><td></td></tr><tr><td>14: end if</td></tr><tr><td>15: i←i+1</td></tr><tr><td></td></tr><tr><td>16: end while 17: return y</td></tr></table></body></html>

policy $\pi$ ). Concretely, we treat an (incomplete) trajectory $q , s _ { 1 } , . . . , s _ { i }$ as the state $\mathbf { \boldsymbol { s } } _ { i }$ .2 Then, a next step can be sampled from the LLM which consumes $\mathbf { s } _ { i }$ as follows:

$$
s _ { i + 1 } \sim \pi ( \mathcal { D } , \mathbf { s } _ { i } ) ,
$$

where $\mathcal { D }$ is the in-context demonstrations consisting of question-solution pairs or task-specific instruction.

As shown in Alg. 1, our algorithm mainly comprises an iterative process of Selection and Expansion operations. During each iteration, we first select the most promising node, and then expand it within the constraints of the computational budget. Both operations are guided by a value network $\nu ^ { \pi }$ (§3.2). The algorithm terminates when the generated answers meet the expected value threshold $\varepsilon$ or the number of iterations reaches the limit $N$ .

Selection We mainly select the tree node with the highest value for expansion. Besides, we introduce a progress term, denoted as $p ( \mathbf { s } )$ , which quantifies the advancement of a state s towards the goal within the search trajectory. By incorporating this term, we prioritize the exploration of nodes that are expected to lead more rapidly toward the final answer. Formally, for each iteration, the next node to explore is selected via

$$
s ^ { \prime } = \operatorname* { m a x } _ { s _ { i } } ( \nu ^ { \pi } ( \mathbf { s } _ { i } ) + \lambda p ( \mathbf { s } _ { i } ) ) ,
$$

where $s ^ { \prime }$ denotes the selected node, and $\lambda$ is introduced to regulate the impact of the progress term.

However, it is non-trivial to estimate the progress of a state. To deal with this issue, we introduce an empirical approach based on the trajectory of greedy decoding. Specifically, we compute the progress term by comparing the number of tokens or steps from a given state to those of the corresponding greedy decoding. For example, when using the step number as the metric, the progress of a state with $d$ steps is $\operatorname* { m i n } ( d / \hat { d } , 1 )$ , where $\hat { d }$ denotes the total number of steps in the trajectory of greedy decoding.

Expansion During the expansion phase, we aim to balance the exploitation and exploration by effectively managing the computation budget allocated to the selected node. Intuitively, an appropriate budget size can promote efficient exploitation, facilitating a faster convergence to the final answer, while also guaranteeing sufficient exploration to cover enough state space for reducing uncertainty. In line with this spirit, we further explore two strategies preferring either exploitation or exploration: Incremental Expansion and Batch Expansion.

Budget Computaton We define the allocated budget for a node (corresponding to s) as the maximum number of its children, denoted as $b$ , which primarily depends on the value $\nu ^ { \pi } ( \mathbf { s } )$ and depth $d$ of the node:

$$
b = \operatorname* { m i n } \left( \lceil \frac { \log ( 1 - \epsilon ) } { d \log ( 1 - \nu ^ { \pi } ( \mathbf { s } ) ) } \rceil , B \right) ,
$$

where $B$ denotes the upper bound of the budget and $\epsilon$ is the expected accuracy, thus a larger $\epsilon$ (e.g., 0.95) encourages more conservative searching. Besides, we empirically employ the $^ { 1 / d }$ term, which fosters exploration at the start of searching but encourages exploitation with $d$ increasing to avoid search space explosion.

As the value scores of the preceding search steps usually suffer a larger variance due to the inefficient learning of delayed and sparse rewards (Sutton and Barto 2018), estimation of them is relatively not accurate enough. This inevitably influences the computation of suitable budget sizes. Therefore, we further propose to calibrate the value scores using the value of the corresponding trajectory from greedy decoding (denoted as $\hat { \nu }$ ), especially for the first few steps:

$$
\nu ^ { \prime } ( \mathbf { s } ) = \frac { \nu ^ { \pi } ( \mathbf { s } ) + \hat { \nu } / d } { 1 + 1 / d } ,
$$

where $\nu ^ { \prime } ( \mathbf { s } )$ represents the calibrated value after normalization. We add the $^ { 1 / d }$ term to mainly adjust the value scores of the first several steps.

Expansion Strategies We propose two expansion strategies that prioritize efficiency and performance, respectively.

• Incremental Expansion: This strategy incrementally expands one child node after another. If the budget allows, the same node can be reselected until the budget is fully utilized. This method tends to conserve computational resources by carefully managing the budget. • Batch Expansion: In contrast, this strategy consumes the entire budget allocated to the selected node during each iteration, resulting in the generation of multiple child nodes simultaneously. This method broadens the search space for subsequent iterations, potentially leading to the identification of superior nodes and enhancing overall performance.

# 3.2 Value Network

The value network $\nu ^ { \pi } ( \mathbf { s } )$ seeks to approximate the expected cumulative reward starting from state s and following a policy $\pi$ thereafter. This can be represented as $\begin{array} { r } { \nu ^ { \pi } ( \mathbf { s } ) \ = } \end{array}$ $\mathbb { E } _ { \boldsymbol { \pi } } \left[ r _ { t } \mid \mathbf { s } _ { t } = \mathbf { s } \right]$ , where $\boldsymbol { r } _ { t }$ is the discounted return starting from state $\mathbf { s } _ { t }$ .

Particularly, given a question $q$ and its correct answer $a$ from an expert demonstration dataset. Each trajectory with reasoning steps (e.g., $s _ { i } )$ ) and final predicted answer $\hat { a }$ is firstly sampled from the policy $\pi$ :

$$
s _ { 1 } , . . . , s _ { n } , \hat { a } \sim \pi ( \mathcal { D } , q ) .
$$

Then, we only take the answer correctness as distant supervision for each reasoning step to train the value network via a Mean Squared Error (MSE) loss:

$$
\mathcal { L } = ( \nu ^ { \pi } ( \mathbf { s } _ { i } ) - \mathbb { I } [ a = \hat { a } ] ) ^ { 2 } ,
$$

where $\mathbb { I } ( * )$ denotes an indicator function.

In this work, regardless of the policy used, we simply take Llama- $\scriptstyle 3 - 8 \mathbf { B } ^ { 3 }$ with a regressive head as our value network. This regressive head is a randomly initialized linear layer, which consumes the hidden state of the last input token and returns a scalar within [0, 1]:

$$
\nu ^ { \pi } ( \mathbf { s } _ { i } ) = { \mathrm { H e a d } } ( { \mathrm { L } } 1 { \mathrm { a m } } \alpha ( \mathbf { s } _ { i } ) [ - 1 ] ) .
$$

# 4 Experiment

# 4.1 Setup

Dataset We conduct experiments on three popular mathematical reasoning datasets:

• GSM8K (Cobbe et al. 2021): This dataset comprises 7,473 training and 1,319 testing grade school math word problems that take $2 \sim 8$ steps to solve. Solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations.   
• TabMWP (Lu et al. 2022): This dataset features 38,431 tabular math word problems, presented in either freetext or multiple-choice formats. We focus on the more general free-text category, consisting of 17,315 training questions and 1,000 randomly sampled test questions for evaluation.   
• MATH (Hendrycks et al. 2021): It consists of 12,500 challenging competition mathematics problems. Following previous work (Lightman et al. 2023), we test on a subset of 500 cases, MATH500.

Models and Hyperparameters For GSM8K and TabMWP, we employ Mixtral- $\mathbf { \delta } \cdot 8 \times 7 \mathbf { B }$ (Jiang et al. 2024) or Llama-3-8B as the policy model and train Llama-3-8B as the value network. Due to the difficulty of MATH, we adopt Llama-3-8B-Instruct, which has demonstrated exceptional mathematical abilities under system 1 mode after being fine-tuned on extensive post-training data. For the policy models, we adhere to the standard approach of utilizing 8 / 4 shots in-context learning for GSM8K / TabMWP, with a temperature of 0.6. We directly use the official 0-shot instruction for MATH testing, with a temperature of 0.8. By default, we set $N , B , \lambda , \varepsilon , \epsilon$ as 100, 10, 0, 0.8, and 0.9, respectively, and investigate other combinations in our analyses. For the value networks, we sample 8 trajectories per training instance, also with a temperature of 0.6. Then, we train the models for 1 epoch across all datasets, employing the AdamW optimizer (Loshchilov and Hutter 2017) with a learning rate of 5e-6 and a linear learning rate scheduler. Besides, we allocate $5 \%$ of the training instances as a development set to select the optimal checkpoints as value networks.

Evaluation Metrics We adopt answer Accuracy and the number of generated tokens (#Tokens $( k ) { \dot { } }$ ) as evaluation metrics for performance and cost, respectively. It should be noted that we do not take into account the cost of executing value networks following (Kang et al. 2024). This is because a value network only performs the regression task, which incurs significantly lower costs compared to the primary generation task. Besides, it can be deployed in parallel in practice.

Baselines We consider the following baselines:

• Greedy Decoding: It intuitively selects the most probable next token at each decoding step.   
• Hard $\mathbf { V o t i n g } @ K$ (Wang et al. 2022): Known as selfconsistency, which ensembles the answers from multiple sampled solutions as the final answer using majority voting. We sample $K = \{ 5 , 1 0 , 2 0 \}$ times with a temperature of 0.6.   
• ToT-DFS (Yao et al. 2024): We implement it by capitalizing on the guidance from our trained value network. Specifically, we prune a node if its value score falls below a threshold of 0.5 and limit the maximum number of children to 5 for preventing infinite loops.   
• ToT-BFS / BeamSearch (Khalifa et al. 2023; Yao et al. 2024; Xie et al. 2024; Zhu et al. 2024): These two methods work similarly for this task. Again leveraging our value networks, each node is expanded to have 5 child nodes, and only 5 nodes with the highest value scores at each depth are kept to avoid search space explosion.   
• Soft $\mathbf { V o t i n g } @ K$ : It is an enhancement over hard voting by utilizing our value networks. It softly ensembles the answers of different paths by taking their value scores as weights.

# 4.2 Main Results

Table 1 shows the main test results on GSM8K and TabMWP. We observe the following conclusions:

Value Guidance Boosts Model Performance In line with prior research (Wang et al. 2022), Hard Voting significantly improves Accuracy. However, its costs also proportionately increase with the growing of sampling size $K$ . With the guidance of our value networks, both Soft Voting and tree search algorithms can further enhance Accuracy without incurring additional costs. Besides, Soft Voting $@ 5$ consistently surpasses Hard Voting $\textcircled { a } 2 0$ , substantiating the effectiveness of verification as previously discussed in (Cobbe et al. 2021).

Table 1: Main test results, where Ours (Incremental) and Ours (Batch) denote LiteSearch equipped with Incremental and Batch Expansion strategies, respectively. For methods guided by our value networks, we emphasize the best results in bold and the second / third-best results with underlining.   

<html><body><table><tr><td colspan="2"></td><td colspan="2">GSM8K</td><td colspan="2">TabMWP</td></tr><tr><td colspan="2"></td><td>Accuracy ↑</td><td>#Tokens (k)↓</td><td>Accuracy ↑</td><td>#Tokens (k)↓</td></tr><tr><td rowspan="10">Mixtral-8×7B</td><td>Greedy Decoding</td><td>.607</td><td>0.14</td><td>.762</td><td>0.07</td></tr><tr><td>Hard Voting @ 5</td><td>.705</td><td>0.66</td><td>.761</td><td>0.37</td></tr><tr><td>Hard Voting @10</td><td>.740</td><td>1.32</td><td>.782</td><td>0.73</td></tr><tr><td>Hard Voting @ 20</td><td>.769</td><td>2.63</td><td>.796</td><td>1.46</td></tr><tr><td>ToT-DFS</td><td>.722</td><td>0.22</td><td>.822</td><td>0.16</td></tr><tr><td>ToT-BFS</td><td>.801</td><td>2.22</td><td>.861</td><td>1.45</td></tr><tr><td>Soft Voting @5</td><td>.779</td><td>0.66</td><td>.811</td><td>0.37</td></tr><tr><td>Soft Voting @10</td><td>.830</td><td>1.32</td><td>.832</td><td>0.73</td></tr><tr><td>Soft Voting @20</td><td>.843</td><td>2.63</td><td>.847</td><td>1.46</td></tr><tr><td>Ours (Incremental)</td><td>.797</td><td>0.41</td><td>.863</td><td>0.22</td></tr><tr><td rowspan="8"></td><td>Ours (Batch)</td><td>.823</td><td>0.55</td><td>.854</td><td>0.29</td></tr><tr><td>Greedy Decoding</td><td>.485</td><td>0.18</td><td>.659</td><td>0.08</td></tr><tr><td>Hard Voting @ 5</td><td>.572</td><td>0.57</td><td>.680</td><td>0.42</td></tr><tr><td>Hard Voting @ 20</td><td>.667</td><td>2.38</td><td>.698</td><td>1.68</td></tr><tr><td>ToT-DFS</td><td>.676</td><td>0.24</td><td>.704</td><td>0.19</td></tr><tr><td>ToT-BFS</td><td>.756</td><td>1.89</td><td>.787</td><td>1.35</td></tr><tr><td>Soft Voting @5</td><td>.689</td><td>0.57</td><td>.747</td><td>0.42</td></tr><tr><td>Soft Voting @20</td><td>.770</td><td>2.38</td><td>.796</td><td>1.68</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ours (Incremental)</td><td>.731</td><td>0.46</td><td>.779</td><td>0.27</td></tr><tr><td>Ours (Batch)</td><td>.757</td><td>0.59</td><td>.776</td><td>0.35</td></tr></table></body></html>

Table 2: Results on MATH500 using Llama-3-8B-Instruct.   

<html><body><table><tr><td></td><td>Accuracy ↑</td><td>#Tokens (k)↓</td></tr><tr><td>Greedy Decoding</td><td>.280</td><td>0.32</td></tr><tr><td>Hard Voting @ 5</td><td>.312</td><td>1.56</td></tr><tr><td>Hard Voting @ 10</td><td>.338</td><td>3.12</td></tr><tr><td>Soft Voting@5</td><td>.378</td><td>1.56</td></tr><tr><td>Soft Voting @10</td><td>.392</td><td>3.12</td></tr><tr><td>Ours (Incremental)</td><td>.420</td><td>2.36</td></tr><tr><td>Ours (Batch)</td><td>.412</td><td>2.43</td></tr></table></body></html>

Current Tree Search Algorithms Neglect the Performance-Cost Tradeoff Previous methods, ToTDFS and ToT-BFS, prefer different evaluation metrics. Among the value-guided approaches, ToT-DFS consistently has the lowest cost but achieves suboptimal performance. This is because ToT-DFS focuses mainly on pruning bad nodes and lacks the flexibility to select better nodes for further improvement. In contrast, ToT-BFS tackles this shortcoming of ToT-DFS by maintaining a branch of nodes with the highest values, thereby resulting in better performance. However, it also unnecessarily visits lots of nodes during the search, leading to significantly higher costs.

LiteSearch Maintains Performance and Decreases Cost By fully utilizing the guidance from value networks, our methods achieve the best tradeoff between performance and cost. Our approaches fall within the cost range of ToTDFS and Soft Voting $\textcircled { a } 5$ , yet yield significantly better performance. For the two expansion strategies, Ours (Incremental) saves nearly $20 \%$ of costs of Ours (Batch) and performs even better on TabMWP. However, Ours (Incremental) performs noticeably worse than Ours (Batch) on Accuracy on GSM8K, with a 2.6-point lower score. This is due to the Batch Expansion strategy providing a better comparison among nodes for selection by expanding more nodes each time. We extensively experiment on another more challenging MATH500 using Llama-3-8B-Instruct. The results in Table 2 again demonstrate the effectiveness of our methods.

Table 3: Ablation study on dynamic budgeting.   

<html><body><table><tr><td></td><td>Accuracy ↑ #Tokens (k)↓</td></tr><tr><td>Ours (Incremental) .797</td><td>0.41</td></tr><tr><td>=static budget</td><td>.779 0.67</td></tr><tr><td>w/o depth penalty</td><td>.780 0.43</td></tr><tr><td>w/o greedy value</td><td>.783 0.40</td></tr><tr><td>Ours (Batch)</td><td>.823 0.55</td></tr><tr><td>=static budget</td><td>.802 1.79</td></tr><tr><td>w/o depth penalty</td><td>.815 0.79</td></tr><tr><td>w/o greedyvalue</td><td>.806 0.62</td></tr></table></body></html>

# 4.3 Ablation Study and Analyses

Dynamic Budgeting Helps Both Performance and CostEfficiency We first study the effectiveness of the dynamic budget size $b$ , which is decided by Eq. 3. The following variants are considered: $( 1 ) \Rightarrow$ static budget: We directly set $b$ as $B$ , resulting in each node being expanded with a fixed budget size; (2) w/o depth penalty: We remove the $^ { 1 / d }$ term from Eq. 3, which previously penalized $b$ as the depth $d$ increased; (3) w/o greedy value: We do not consider Eq. 4 to calibrate value scores with greedy results.

![](images/ccea6a1a7ce7a7a5367483cdfa9097e67b94c0c9d6f5133da62d6060b988ce3c.jpg)  
Figure 2: Performance of Ours (Incremental) and Ours (Batch) on GSM8K when using different budget upperbounds $B$ , where $B = \{ 1 , 2 , 3 , 5 , 1 0 \}$ .

As shown in Table 3, we observe that dynamic budgeting helps in both Ours (Incremental) and Ours (Batch) by allowing them to maintain higher accuracy with fewer tokens compared to all other variants. Specifically, $\Rightarrow$ static budget severely hurts both performance and cost, particularly leading to 3 times computation costs when using Batch Expansion. w/o depth penalty and w/o greedy value perform competitively for Ours (Incremental), but still have considerable negative influence on Ours (Batch). These results highlight the importance of dynamic budgeting especially in scenarios where Batch Expansion is employed.

![](images/7627bc1e0f8a590a5c73006ea878ae0df4cf0926472e6652a418e6a61cf5cf8c.jpg)  
Figure 3: Performance of Ours (Incremental) on GSM8K when using step number and token number to estimate progress term $p ( \mathbf { s } )$ , where $\lambda = \{ 0 , 0 . 0 5 , 0 . 1 , 0 . 1 5 , 0 . 2 \}$ .

Influences of Budget Limitation Budget limitation $B$ decides the upperbound of budget size $b$ . As illustrated in Fig. 2, we observe a clear tradeoff between performance and cost. With the growth of $B$ , the computation cost also increases correspondingly because larger budget sizes are allocated to challenging states with lower value scores. Consequently, more problems are correctly solved due to more comprehensive searching. Regarding the two expansion strategies, Ours (Incremental) perform slightly better than Ours (Batch) with competitive accuracy but fewer costs when $\textit { B } \leq \ 3$ . This is because it may not use up all budgets when good nodes have been generated during incremental expansion. However, Ours (Batch) yields better accuracy by taking more costs when $B \ = \ \{ 5 , 1 0 \}$ because it fully utilizes allocated budgets, thus providing larger search space for better selection.

Influence of Progress Estimation We then investigate the choice of $p ( \mathbf { s } )$ and $\lambda$ in Eq. 2. We consider step number and token number against corresponding results of greedy decoding to estimate $p ( \mathbf { s } )$ . As depicted in Fig. 3, increasing $\lambda$ improves cost-efficiency by prioritizing nodes with faster progress at the risk of inaccuracy. Comparing step number and token number, the former is relatively better with a modest downward trend. By sacrificing 1.3 points in accuracy, utilizing step number and $\lambda = 0 . 1 5$ saves nearly $20 \%$ computational costs. In contrast, the efficacy of token number is unsatisfactory. This can be attributed to its higher degree of variability, thus yielding less precise estimates of progress terms.

![](images/0bc89b8b6b125c1261f7d7f8e10c14f24f675d73e68219792d1eeda9f02a4ce4.jpg)  
Figure 4: Cost of Ours (Incremental) and Ours (Batch) with the growth of difficulty on GSM8K. Difficulty is defined as $\cdot _ { 1 } - x ^ { \prime \prime }$ , where $x$ is the frequency of gold answer in 20 sampled paths.

Harder Problems are Allocated Larger Budgets Fig. 4 illustrates the correlation between cost and question difficulty. Inspired by (Wang et al. 2024b), we estimate the difficulty of a question by computing the frequency of the gold answer in multiple sampled paths after inversion $( ^ { 6 6 } 1 - x ^ { 3 7 } )$ . We observe that for easier questions, our methods cost competitively to Greedy Decoding. However, as the difficulty escalates, the cost of our methods also rises proportionately. Regarding our expansion strategies, Ours (Batch) consistently takes higher costs and the gap also widens with the difficulty increases.

Mixture-of-Domain Boosts Performance An important future direction is to construct a general value network that can address questions from different domains. To validate the potential of this direction, we conduct experiments us

![](images/595d842f07517ffd3fc5f2480aa281d0c0961f0cf3518d4ae6ff491bed1c58a6.jpg)  
Figure 5: Accuracy of Ours (Incremental) on TabMWP with value networks trained with Single- or Mixed-domain data, where we use full GSM8K and different ratios of TabMWP from $1 \%$ to $100 \%$ .   
Figure 6: A lemon picked example. We provide correct and wrong solutions with value scores for each step and error steps are marked.

Question:   
Josh decides to try flipping a house.He buys a house for $\$ 80,000$   
and then puts in $\$ 50,000$ in repairs.This increased the value of the   
house by $1 5 0 \%$ ，How much profit did he make?   
Correct Solution:   
Step 1.The value of the house increased by $1 5 0 \%$ ,so it increased by   
$1 5 0 \% * 8 0 \small { , } 0 0 0 = 1 2 0 \small { , } 0 0 0$ .(Value: 0.33)   
Step 2. Thus,the value of the house is now $8 0 , 0 0 0 + 1 2 0 , 0 0 0 =$   
200,000.(Value:0.49)   
Step 3.Josh put in $5 0 , 0 0 0 + 8 0 , 0 0 0 = 1 3 0 , 0 0 0$ in the house, so he   
madea profit of $2 0 0 , 0 0 0 - 1 3 0 , 0 0 0 = 7 0 , 0 0 0 .$ (Value: 0.97)   
Wrong Solution:   
Step 1.The total cost was $8 0 , 0 0 0 + 5 0 , 0 0 0 = 1 3 0 , 0 0 0$ (Value: 0.62)   
Step 2. The new value of the house was $1 5 0 \%$ of the cost or $1 . 5 ^ { * }$   
$1 3 0 , 0 0 0 = 1 9 5 , 0 0 0$ (Value: 0.31) $\times$   
Step 3.So the profit was $1 9 5 , 0 0 0 - 1 3 0 , 0 0 0 = 6 5 , 0 0 0$ (Value: 0.35) X

ing value networks trained with different ratios of TabMWP data and full GSM8K data. Despite the significant difference in question style and answer type, the results in Fig. 5 demonstrate that using a mixture of different domains helps improve search performance, especially when the targetdomain training instances are scarce (0.75 vs. 0.78 on Accuracy when using $1 \%$ TabMWP data). This highlights the effectiveness of building robust and stronger value networks by collecting various training instances. Further exploration in this direction will be pursued in future work.

Voting Helps when Values are Inaccurate Analyses above have shown the effectiveness of our method. However, though much more efficient, guided tree searches often cannot outperform Soft Voting on Accuracy. We first collect the questions that our approach fails to solve, yet are successfully addressed by Soft Voting $@ 2 0$ . Fig. 6 displays a lemon pick example. We observe that our value network can select the correct answer when the complete rationale is provided. However, the first two steps in the correct path are scored much lower than the first step of the wrong solution. This results in a reduced priority in exploring the node that is actually of higher quality. Besides, due to the imperfect value network, some incorrect paths may be erroneously scored higher than the correct ones. Guided tree search methods, which search for only one path as the final answer, inevitably fail in these instances. However, Soft Voting can mitigate this issue by leveraging the benefits of both majority voting and the value network. Consequently, even if the highest value is attained by an incorrect path, Soft Voting still has the potential to reach the correct answer with a higher frequency. As demonstrated in Table 4, the use of voting enables Soft Voting $\textcircled { a } 2 0$ to outperform $B e s t @ 2 0$ , highlighting the efficacy of voting in enhancing accuracy.

Table 4: Results of our methods enhanced by voting, where $B e s t @ 2 0$ is a variant of Soft Voting $@ 2 0$ , which only selects the best path with the highest value as the final output. For Ours $^ +$ Soft Voting, we discard results with values lower than $\alpha$ , and utilize Soft Voting $@ 2 0$ to solve them.   

<html><body><table><tr><td></td><td>Accuracy ↑</td><td>#Tokens (k)↓</td></tr><tr><td>Best@20</td><td>.833</td><td>2.63</td></tr><tr><td>Soft Voting @ 20</td><td>.843</td><td>2.63</td></tr><tr><td>Ours (Batch)</td><td>.823</td><td>0.55</td></tr><tr><td>+Soft Voting (α = 0.7)</td><td>.841</td><td>0.73</td></tr><tr><td>+Soft Voting (α = 0.8)</td><td>.843</td><td>0.79</td></tr><tr><td>+Soft Voting (α= 0.9)</td><td>.847</td><td>0.99</td></tr></table></body></html>

Inspired by these findings, we further investigate the improvement of our method using voting. Specifically, we discard the answers predicted by our method when their value scores fall below a threshold $\alpha$ . Generally, these predictions exhibit a higher error rate due to the correlation between value scores and correctness. Subsequently, we employ Soft Voting to address these unresolved questions. The results in Table 4 indicate that accuracy can be significantly improved by increasing $\alpha$ . However, the associated costs also rise substantially, albeit remaining lower than those of Soft Voting $\textcircled { a } 2 0$ .

# 5 Conclusion

In this work, we study guided tree search to address math problems, aiming to decrease the computation costs while maintaining the performance. Inspired by the theory of value function, we propose dynamic node selection and expansion strategies, which dynamically determine the priority of nodes to explore and manage the computational budget during expansion. Both procedures are guided by an easy-toimplement value network trained without step-wise supervision. Experiments show that our methods achieve competitive performance with typical baselines but significantly save computation costs. Ablation studies validate the effectiveness of each component, providing more feasible options for various practical scenarios. Besides, we identify the shortcomings of this research line, and provide a potential strategy for addressing these issues.