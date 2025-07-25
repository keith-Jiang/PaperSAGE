# What Are Step-Level Reward Models Rewarding? Counterintuitive Findings from MCTS-boosted Mathematical Reasoning

Yiran $\mathbf { M } \mathbf { a } ^ { 1 * }$ , Zui Chen2\*, Tianqiao Liu3, Mi Tian3, Zhuo Liu4, Zitao Liu5†, Weiqi Luo5,

1Zhejiang University, Hangzhou, China 2ShanghaiTech University, Shanghai, China 3TAL Education Group, Beijing, China 4University of Rochester, New York, USA 5Jinan University, Guangzhou, China mayiran $@$ zju.edu.cn, chenzui $@$ shanghaitech.edu.cn, liutianqiao1, tianmi @tal.com, zhuo.liu@rochester.edu, liuzitao, lwq @jnu.edu.cn

# Abstract

Step-level reward models (SRMs) can significantly enhance mathematical reasoning performance through process supervision or step-level preference alignment based on reinforcement learning. The performance of SRMs is pivotal, as they serve as critical guidelines, ensuring that each step in the reasoning process is aligned with desired outcomes. Recently, AlphaZero-like methods, where Monte Carlo Tree Search (MCTS) is employed for automatic step-level preference annotation, have proven particularly effective. However, the precise mechanisms behind the success of SRMs remain largely unexplored. To address this gap, this study delves into the counterintuitive aspects of SRMs, particularly focusing on MCTS-based approaches. Our findings reveal that the removal of natural language descriptions of thought processes has minimal impact on the efficacy of SRMs. Furthermore, we demonstrate that SRMs are adept at assessing the complex logical coherence present in mathematical language while having difficulty in natural language. These insights provide a nuanced understanding of the core elements that drive effective step-level reward modeling in mathematical reasoning. By shedding light on these mechanisms, this study offers valuable guidance for developing more efficient and streamlined SRMs, which can be achieved by focusing on the crucial parts of mathematical reasoning.

Math Problem: There are 6 students playing tennis and twice that number playing volleyball. There are 16 boys and 22 girls playing soccer. If each student only participates in one group, how many students are there in total? Thought: _ The problem involves calculating the total number of students by adding up the number of students in each group. Firstly, we need to find out how many students are playing volleyball. Since there are 6 students playing tennis, and it's stated that there are Step1 twice that number playing volleyball, we multiply 6 by 2. Math Expression: Students Playing Volleyball = 6 \* 2 = 12 Thought: Now you can answer the problem in this step. We also have the number of boys and girls playing soccer, which is 16 and 22 respectively. To find the total number of students, we add up the number of students in all three groups. Step2 Math Expression: Total Students $\mathbf { \sigma } = \mathbf { \sigma }$ Students Playing Tennis $^ +$ Students Playing Volleyball $^ +$ Boys Playing Soccer $^ +$ Girls Playing Soccer $= 6 +$ $1 2 + 1 6 + 2 2 = 5 6$ So, there are 56 students in total. The answer is 56.

# Introduction

Large Language Models (LLMs) have demonstrated their remarkable capabilities across a wide range of tasks, such as information extraction, natural language understanding, etc (Zhao et al. 2023), totally revolutionizing the deep learning community. Among these capabilities, reasoning stands out as a critical area of focus, especially mathematical reasoning, which needs to be further improved due to its complex nature. Numerous studies have shown that multi-step reasoning often facilitated through Chain-of-Thought (CoT)

prompting, can significantly enhance model performance on reasoning tasks (Zhou et al. 2023; Besta et al. 2024; Ding et al. 2023; Yao et al. 2024; Wang et al. 2022; Wei et al. 2022; Zheng et al. 2024; Li et al. 2024; Zhan et al. 2024).

Recently, guided tree-search methods further improved reasoning performance by exploring various reasoning paths through online simulation to identify the optimal solution paths (Hao et al. 2023, 2024; Feng et al. 2023). Although a better reasoning path leads to a better performance, the length of these reasoning chains leads to an exponential increase in the search space, resulting in substantial computational costs. Given the high expense of LLM inference, performing an online tree search for each reasoning problem introduces repeated and unnecessary overhead.

To address this issue, step-level reward models (SRM)

was proposed to improve search efficiency. Lightman et al. (2023) introduced the process reward model (PRM), which employs human-annotated step-level scores for reward modeling, and Ma et al. (2023) further demonstrated the effectiveness of SRMs in math reasoning and coding tasks. Then, Math-Shepherd (Wang et al. 2024), systematically generates step-level preference data through exhaustive reasoning process traversal to train reward models and reinforce the model’s capabilities. More recently, inspired by AlphaZero, Monte Carlo Tree Search (MCTS) (Xie et al. 2024; Chen et al. 2024a,b) was then used for collecting preferences more efficiently because of its capability of balancing exploration and exploitation. These trained SRMs can effectively enhance reasoning performance by either assisting step-level preference alignment with proximal policy optimization (PPO) during training stage or serving as step verifiers during inference stage.

Despite the significant achievements in mathematical reasoning performance achieved by the SRMs constructed by MCTS-based method, the exact workings of these reward models and what they are truly rewarding remain unclear. Brain and cognitive scientists have argued that diverse thinking and reasoning processes do not necessarily rely on natural language. (Fedorenko, Piantadosi, and Gibson 2024). A skilled human mathematician, for instance, can determine whether a mathematical expression is logically coherent and numerically correct without the participation of the natural language. Building on this idea, our research explores a similar hypothesis for LLMs: that natural language descriptions of thought processes are not essential for mathematical reasoning within these models. We suppose that LLMs can be trained to recognize preferences for mathematical language directly during problem-solving, without relying on natural language descriptions. This implies that LLMs might be capable of understanding and processing mathematical reasoning through the intrinsic structure of mathematical language, potentially leading to more efficient and focused training methods that bypass the need for natural language explanations. Furthermore, it is believed that incorrect solutions often arise from wrong mathematical calculations or logical errors (Zhang et al. 2024), with the latter being more challenging (Chen et al. 2024a). Therefore, we further investigate the effectiveness of SRMs in evaluating logical coherence in pure mathematical language, demonstrating that the improvements are not merely the result of encouraging correct calculations within a single step. Additionally, and somewhat surprisingly, we found that SRMs struggle to learn how to evaluate logical coherence in natural language. This will further support that natural language is not necessary for step-level reward modeling.

To investigate the respective roles of natural language and mathematical language in step-level reward modeling, we decompose each step of the reasoning path into two components: natural language descriptions of thought processes and math expressions (Figure 1). The ablation studies are conducted by selectively removing different parts from the inputs of the SRMs. This decomposition mirrors the human problem-solving process in mathematics, which typically involves an initial phase of thinking through the problem, followed by the execution of calculations based on that thought process. The thought processes include the strategy to be taken in that step, while the calculations are the executions of the thought processes. In other words, our decomposition aims to separate the natural language (composing the ‘thoughts’) from the mathematical expressions (contained in the execution of ‘thoughts’). This framework aims to foster a deeper understanding of the role of natural language for step-level reward modeling.

To summarize, our experiments support that SRMs appear to have some intrinsic affinity for mathematical expression, not natural language. Specifically, we propose the following key insights.

1. Natural language descriptions of thought processes are not necessary for successful step-level reward modeling. 2. SRMs not only promote accurate calculations within individual steps but also effectively assess the challenging logical coherence in mathematical language. 3. Assessing logical coherence in natural language is difficult, and SRMs often struggle with this task.

# Preliminaries

# Markov Decision Process

Definition A Markov Decision Process (MDP) is a mathematical framework used to model decision-making problems. This framework is fundamental for addressing a wide range of reinforcement learning (RL) problems where the outcomes are partially random and partially controllable. An MDP is defined by a tuple $( S , A , P , R , \gamma )$ , where:

• $S$ is the set of states.   
• $A$ is the set of actions.   
• $P$ is the transition probability function, $\textstyle P ( s _ { t + 1 } | s _ { t } , a _ { t } )$ , which defines the probability of transitioning to state $s _ { t + 1 }$ given the current state $s _ { t }$ and action $a _ { t }$ .   
• $R$ is the reward function, $R ( s _ { t } , a _ { t } , s _ { t + 1 } )$ , which defines the reward received after transitioning from state $s _ { t }$ to state $s _ { t + 1 }$ by taking action $a _ { t }$ .   
• $\gamma$ is the discount factor, which determines the importance of future rewards.

Bellman Expectation Equation For state value function $V ( s )$ , the Bellman Expectation Equation is:

$$
V ^ { \pi } ( s ) = \mathbb { E } _ { a \sim \pi ( \cdot | s ) } \left[ \mathbb { E } _ { s ^ { \prime } \sim P ( \cdot | s , a ) } \left[ R ( s , a , s ^ { \prime } ) + V ^ { \pi } ( s ^ { \prime } ) \right] \right]
$$

For state-action value function $Q ( s , a )$ , the Bellman Expectation is:

$$
Q ^ { \pi } ( s , a ) = \mathbb { E } _ { s ^ { \prime } \sim P ( \cdot \mid s , a ) } \left[ R ( s , a , s ^ { \prime } ) + \mathbb { E } _ { a ^ { \prime } \sim \pi ( \cdot \mid s ^ { \prime } ) } \left[ Q ^ { \pi } ( s ^ { \prime } , a ^ { \prime } ) \right] \right]
$$

Optimal Value Functions The optimal value functions are defined as:

$$
\begin{array} { c } { { V ^ { \ast } ( s ) = \displaystyle \operatorname* { m a x } _ { \pi } V _ { \pi } ( s ) } } \\ { { Q ^ { \ast } ( s , a ) = \displaystyle \operatorname* { m a x } _ { \pi } Q _ { \pi } ( s , a ) } } \end{array}
$$

Therefore, the relationship between the optimal value functions and the Bellman Optimality Equation is:

$$
V ^ { * } ( s ) = \operatorname* { m a x } _ { a } Q ^ { * } ( s , a )
$$

![](images/6778f96a5538f1259c110cbcad9b7d3e4256823ff7718ef0b4c93a9ec7d32a70.jpg)  
Figure 2: Illustration of the role of SRMs in mathematical reasoning and the SRMs with different input structures we investigate

# Setup

# LLM’s Math Reasoning as MDP: Our Definition

Figure 2 shows the mathematical reasoning process with each step decomposed into thought and math expressions. Specifically, our MDP definition is as follows:

$$
\mathrm { M D P } = ( S , A , P , R )
$$

where:

• State The state space $S$ consists of states defined as $s _ { i } =$ $( T _ { k } , E _ { k } ) _ { k = 0 } ^ { i }$ , representing a sequence of thoughts $T _ { k }$ and equations $E _ { k }$ up to step $i$ . • Action The action space $A$ consists of actions defined as $a _ { i } = T _ { i + 1 }$ , representing the natural language descriptions of the subsequent thought proposed by the LLM. • State Transition $P ( s _ { i + 1 } | s _ { i } , a _ { i } )$ is the state transition function, defining the probability of transitioning to state $s _ { i + 1 }$ from state $s _ { i }$ after taking action $a _ { i }$ . This function is implemented by the LLM generating the corresponding math expression $E _ { i + 1 }$ based on the next thought $a _ { i } = T _ { i + 1 }$ and the current state $s _ { i } = ( T _ { k } , E _ { k } ) _ { k = 0 } ^ { i }$ . • Reward Function $R ( s _ { i } , a _ { i } , s _ { i + 1 } )$ is the reward function, defining the immediate reward received after transitioning to state $s _ { i + 1 } = ( T _ { k } , E _ { k } ) _ { k = 0 } ^ { i + 1 }$ from state $s _ { i }$ by taking action $a _ { i }$ . We define the reward up to state $s _ { i + 1 }$ based on whether it can lead to the correct final answer:

$$
R ( s _ { i } , a _ { i } , s _ { i + 1 } ) = { \binom { 1 , \quad { \mathrm { f i n a l ~ a n s w e r ~ i s ~ c o r r e c t } } } { 0 , \quad { \mathrm { f i n a l ~ a n s w e r ~ i s ~ i n c o r r e c t } } } }
$$

Additionally, policy $\pi ( a _ { i } | s _ { i } )$ is implemented by the LLM generating the thought of the next step $a _ { i } ~ = ~ T _ { i + 1 }$ based on the current state $\mathbf { \bar { \rho } } _ { s _ { i } } = ( T _ { k } , E _ { k } ) _ { k = 0 } ^ { i }$ . According to Equation (1), the goal of an agent is to maximize $V _ { \pi } ( s _ { i } )$ or $Q _ { \pi } ( s _ { i } , a )$ by generating the correct thoughts $T$ in each step.

In summary, a language model plays a dual role in the MDP framework:

1. As an Agent The LLM is responsible for making decisions by selecting appropriate actions (next thoughts $T _ { i + 1 , }$ ) at each state, following the policy $\pi ( a _ { i } | s _ { i } )$ .   
2. As a World Model The LLM also acts as the world model $P ( s _ { i + 1 } | s _ { i } , a _ { i } )$ by predicting action outcomes (state transitions) using its internal knowledge and training data. It simulates the environment of mathematical reasoning by executing thought $T _ { i + 1 }$ through corresponding calculations, thus providing the prediction of new states si+1.

# MCTS for Step-Level Preference Collection

Understanding the natural correspondence between math reasoning and MDP, we can readily use MCTS for efficient step-level preference collection. The MCTS starts from a root node $s _ { 0 }$ , which is a math problem in mathematical reasoning tasks. Then, each new node corresponds to a state update. Each iteration of MCTS can be divided into four phases: Selection, Expansion, Rollout, and Backpropagation.

1. Selection. The selection phase in MCTS involves traversing the tree from the root node $s _ { 0 }$ (the initial math problem) to a leaf node using a selection policy. This policy, typically the Upper Confidence Bound for Trees (UCT) formula, balances exploration and exploitation. At node $s _ { i }$ , the next node is chosen by:

$$
s _ { i + 1 } ^ { * } = \arg \operatorname* { m a x } _ { s _ { i + 1 } } \left[ \frac { c ( s _ { i + 1 } ) } { N ( s _ { i + 1 } ) } + w _ { \exp } \cdot \sqrt { \frac { \log N ( s _ { i } ) } { N ( s _ { i + 1 } ) } } \right] ,
$$

where $c ( s _ { i + 1 } )$ is the correct counts, $N ( s _ { i } )$ and $N ( s _ { i + 1 } )$ are visit counts, and $w _ { \mathrm { e x p } }$ balances exploration and exploitation. This process continues until an unexplored node is found.

2. Expansion. Upon reaching a leaf node, $n$ new candidate actions (thoughts) $\{ a _ { i } ^ { j } \mid j = 1 , . . . , n \}$ are generated by the agent given the current state $s _ { i }$ . Given the candidate actions (thoughts), the world model will execute them through mathematical calculations, constructing the new candidate states $\{ s _ { i } ^ { j } | j = 1 , . . . , n \}$ . These candidate states are added as child nodes to the current node to expand the tree, allowing for a broader exploration of potential problem-solving paths.

3. Rollout. The rollout phase simulates the reasoning process from the newly expanded node to a terminal state or predefined maximum depth. The score of a node is then obtained according to Equation (3). This procedure estimates the scores of the new nodes according to the simulation results, informing the back-propagation phase.

4. Back-propagation. Results from the rollout are propagated back up the tree to update values and visit counts of each node. Starting from the final state, the effectiveness of the problem-solving process updates the value $V ( s )$ of each state. This procedure improves the selection policy for future iterations.

After completing MCTS, step-level preference pairs can be gathered by comparing the values of the nodes in each tree.

# Step-level Reward Modeling

After collecting all the preference pairs, step-level reward models can be constructed through contrastive learning. Based on our MDP definition, an SRM is regarded as the action-value function $Q ( s , a )$ or the value function $V ( s )$ . Specifically, we investigate different reward models for ablation studies, where reward models take different inputs to evaluate the ongoing reasoning process. Accordingly, we define four reward models (Figure 2-right) for the ablation study:

• Full-Context Step-level Reward Model (FC-SRM) This model takes both the thoughts and math expressions of the current state as input.

$$
V _ { 1 } ( s _ { i } ) = V _ { 1 } ( ( T _ { k } , E _ { k } ) _ { k = 0 } ^ { i } )
$$

• Math-Only Step-level Reward Model (MO-SRM) This model takes only the math expressions of the current state as input, excluding the natural language descriptions of thought processes.

$$
V _ { 2 } ( s _ { i } ) = V _ { 2 } ( ( E _ { k } ) _ { k = 0 } ^ { i } )
$$

• Single-Step Math-Only Step-level Reward Model (SSMO-SRM) This model takes only the newest math expression of the ongoing reasoning process as input, excluding the natural language and all the previous math expressions.

$$
V _ { 3 } ( s _ { i } ) = V _ { 3 } ( E _ { i } )
$$

• Next-Thought Step-level Reward Model (NT-SRM) This model takes both the thoughts and math expressions of the current state as input, and evaluates the next thought. According to our definition, the next thought is the action taken by the agent. Thus this reward model is the action-value function under our MDP definition of mathemetical reasoning.

$$
Q ( s _ { i } , a _ { i } ) = Q ( ( T _ { k } , E _ { k } ) _ { k = 0 } ^ { i } , T _ { i + 1 } )
$$

# Beam Search with Step-Level Reward Model

Given the SRMs trained on the preference data, it is commonly used for step-level preference alignment to update the policy. The purpose of this procedure is to generate the best action through the updated policy $\pi ^ { \prime }$ , thereby reducing the overhead caused by online MCTS. It is also possible to update the world model $P$ with these preference pairs as better accuracy indicates better mathematical performance.

# Algorithm 1: Beam Search Algorithm

， 1: Initialize beam $B \gets \{ s _ { 0 } \}$ 2: while $\boldsymbol { B }$ is not empty do 3: Initialize empty list $B _ { \mathrm { n e x t } }  \varnothing$ 4: for each state $s _ { i }$ in $\boldsymbol { B }$ do 5: Generate a set of candidate actions $\{ a _ { i } ^ { 1 } , a _ { i } ^ { 2 } , \ldots , a _ { i } ^ { c } \}$ based on $s _ { i }$ 6: for each action $a _ { i } ^ { j }$ in $\{ a _ { i } ^ { 1 } , a _ { i } ^ { 2 } , \ldots , a _ { i } ^ { c } \}$ do 7: Compute the next state $s _ { i + 1 } ^ { j }  P ( s _ { i + 1 } | s _ { i } , a _ { i } ^ { j } )$ 8: Evaluate the score of $s _ { i + 1 } ^ { j }$ + 9: Add sij+1 to Bnext 10: end for 11: end for 12: Sort $\scriptstyle B _ { \mathrm { n e x t } }$ by score and keep the top $B$ states 13: Update beam $B \gets \log B$ states from $\boldsymbol { B } _ { \mathrm { n e x t } }$ 14: end while 15: return the best state from the final beam

As this study focuses on the SRMs, our experiments will not include the preference alignment procedure. Instead, we can use the SRMs as the scoring function during beam search (BS) Algorithm 1 for simplification. This simplification excludes potential uncertainties in the alignment process, providing a more straightforward understanding of SRMs’ effectiveness. Notably, setting $B = 1$ makes BS effectively become greedy search (GS).

The greedy search can be regarded as a reasoning process supervised by an SRM (Figure 2-left). Indeed, with an infinite number of samples, the optimal actions and states identified through the policy $\pi$ and the world model $P$ will converge to the optimal actions and states similar to those generated by the optimal policy $\pi ^ { * }$ in Equation (1), respectively.

$$
\operatorname* { l i m } _ { n \to \infty } P ( \arg \operatorname* { m a x } _ { \{ a _ { t } \} _ { t = 0 } ^ { n } } Q ( s , a _ { t } ) = \arg \operatorname* { m a x } _ { a \in A _ { \pi } ( s ) } Q ( s , a ) ) = 1
$$

where $a _ { t } \sim \pi ( a | s )$ and $A _ { \pi } ( s )$ denotes the state space of actions generated by the policy $\pi$ given state $s$ . Similarly, for states, we also have

$$
\operatorname* { l i m } _ { n \to \infty } P ( \arg \operatorname* { m a x } _ { \{ s _ { t } ^ { \prime } \} _ { t = 0 } ^ { n } } V ( s _ { t } ^ { \prime } ) = \arg \operatorname* { m a x } _ { s ^ { \prime } \in S ( s , a ) } V ( s ^ { \prime } ) ) = 1
$$

where $s _ { t } \sim \mathbb { E } _ { a _ { t - 1 } \in \pi ( a | s _ { t - 1 } ) } P ( s | s _ { t - 1 } , a _ { t - 1 } ) .$ .

# Experiments Implementation Details

Datasets To construct step-level preference pairs through MCTS, we use the math problems and their corresponding final answers from the training data of GSM8K (Cobbe et al. 2021) and MATH (Hendrycks et al. 2021). The accuracies are evaluated on the test data.

<html><body><table><tr><td rowspan="2">Agent & World Model Llama-3-8B-Instruct</td><td rowspan="2">Historical Thoughts</td><td rowspan="2">Historical Equations Thoughts Equations</td><td rowspan="2">Next</td><td rowspan="2">Next</td><td colspan="2">Accuracy (Gain) %</td></tr><tr><td>GSM8K</td><td>MATH</td></tr><tr><td>Pass @1 (3-shots)</td><td></td><td></td><td></td><td></td><td>78.47 (+0.00)</td><td>31.16 (+0.00)</td></tr><tr><td>+GS w/ SRM (DeepSeek-Math-7B-Base)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Full-Context SRM</td><td>√</td><td><<x</td><td>√</td><td>√</td><td>86.20 (+7.73)</td><td>38.58 (+7.42)</td></tr><tr><td>Math-Only SRM</td><td>X</td><td></td><td>X</td><td></td><td>85.82 (+7.35)</td><td>39.64 (+8.48)</td></tr><tr><td>Single-Step Math-Only SRM</td><td>X</td><td></td><td>X</td><td></td><td>82.11 (+3.64)</td><td>37.46 (+6.30)</td></tr><tr><td>Next-Though SRM</td><td>√</td><td>√</td><td>√</td><td>X</td><td>79.38 (+0.91)</td><td>30.98 (-0.18)</td></tr><tr><td>+GS w/ SRM (Qwen2-7B)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Full-Context SRM</td><td>√</td><td>√</td><td>√</td><td>√</td><td>82.94 (+4.47)</td><td>35.58 (+4.42)</td></tr><tr><td>Math-Only SRM</td><td>X</td><td></td><td>X</td><td>√</td><td>83.78 (+5.31)</td><td>35.10 (+3.94)</td></tr><tr><td>Single-Step Math-Only SRM</td><td></td><td>×</td><td>X</td><td>√</td><td>81.65 (+3.18)</td><td>33.08 (+1.92)</td></tr><tr><td>Next-Though SRM</td><td>√</td><td>√</td><td>√</td><td>X</td><td>81.73 (+3.26)</td><td>31.40 (+0.24)</td></tr></table></body></html>

Table 1: SRMs act as step-level scoring functions during GS. Sample $c = 5$ candidates of the subsequent step at each node and use beam size $B = 1$ (greedy search). The agent and the environment model is Llama-3-8B-Instruct. The reward models are trained based on Deepseek-Math-7B-Base or Qwen2-7B.

Models The reasoning process is conducted by the dialogue between two LLMs. We use the Llama-3-8B-Instruct (Dubey et al. 2024) as both the agent and world model in MCTS because of its excellent ability to follow instructions.

Prompt One LLM (as agent) is instructed to generate natural language descriptions of thoughts, and the other (as world model) is instructed to execute the thoughts. For specific prompts, see Appendix.

Baseline We use Llama-3-8B-Instruct construct the ‘Pass $\mathbf { \Pi } ^ { \left( \alpha \right) } \mathbf { l } ^ { \prime }$ baseline based on our prompt with 3 shots.

MCTS for Step-Level Preference Collection The MCTS requires the agent sampling $n = 6$ candidate actions at each expansion phase and iterates 500 times on each problem to evaluate the quality of each node. Notably, to avoid the influence of the variation of answer format, we use a supervised fine-tuned (SFT) model based on DeepSeek-Math-7B-Base to assert the correctness of the solution after each rollout during the search. This model is also used in our evaluation pipeline. To strengthen the preferences, only the preference pairs whose difference of value is greater than 0.7 are assumed valid. For detailed hyperparameters, see Appendix.

Reward Training DeekSeek-Math-7B-Base (Shao et al. 2024) or Qwen2-7B (Yang et al. 2024) is used as the base model for SRM training. Each SRM is trained on two instances, with each instance equipped with 8 A800 GPUs. For detailed hyperparameters, see Appendix.

# Main Results

After collecting all the step-level preference pairs through MCTS, datasets are constructed for FC-SRM, MO-SRM, SSMO-SRM, and NT-SRM training by selecting the corresponding components in each piece of data. The training curves are shown in Figure 3. These SRMs are subsequently used as scoring functions in greedy search, the accuracy and absolute gains over baseline are reported in Table 1. The analyses will be included in the following sections.

# Do we really need natural language?

Intuitively, one might expect that natural language descriptions provide essential contextual information and aid SRMs’ cognitive understanding. The SRMs with different input formats: full-context (FC) and math-only (MO) are trained to investigate this aspect.

GSM8K MATH8025.050 T 40.0 宁 T37.5035.0 工 工32.530.0 1pass@1 CSw/SRM(FC) (FCI SRM(MO) pass@1 CSw/SRM(FC) RM(FC)SRM(MO)Setup SetupSRM BaseQwen2-7B deepseek-math-7b-base

Removing natural language has a minimal effect on steplevel reward modeling. FC-SRMs and MO-SRMs exhibit very similar performance in both preference prediction accuracy and greedy search, suggesting that successful step-level reward modeling is not contingent upon natural language descriptions, which is contrary to intuition. Even without the natural language descriptions of thoughts at each step, the

![](images/26ac6c8e3d6543d1521ea44a31f378ab50aeafe67cbfe564f839dc6af5033688.jpg)  
Figure 3: Effect of natural language descriptions and math expressions on step-level reward modeling. The agent and the environment model is Llama-3-8B-Instruct. The reward models are trained based on Qwen2-7B or Deepseek-Math-7B-Base. (Note that the ‘accuracy’ here is the accuracy of preference during reward training.)

MO-SRMs can still be successfully trained (Figure 3). Table 1 and Figure 4 further show the performance of these SRMs when used as scoring functions during greedy search. In setups such as MATH with DeekSeek-Math-7B-Base as the base model of SRM, the MO-SRM $( 3 9 . 6 4 \% )$ can even outperform the FC-SRM $( 3 8 . 5 8 \% )$ . We further conducted t-tests to provide a more detailed statistical comparison between the FC-SRMs and MO-SRMs across different datasets and base models. For the GSM8K dataset, the t-test results are $t = - 0 . 1 8$ , $p = 0 . 8 6$ for Qwen2-7B, and $t = - 0 . 1 4$ , $p = 0 . 8 9$ for deepseek-math-7b-base. For the MATH dataset, the results are $t = 0 . 7 9$ , $p = 0 . 4 4$ for Qwen2-7B, and $t = 0 . 7 7$ , $p = 0 . 4 5$ for deepseek-math-7bbase. In all cases, the $\mathsf { p }$ -values are greater than 0.05, indicating that the differences in performance between the FCSRM and MO-SRM are not statistically significant. These results support the conclusion that omitting natural language from the inputs of SRMs has negligible effects on the effectiveness of SRMs.

# Can SRMs evaluate logical coherence in math language?

The success of MCTS-based methods is attributed to the ability to avoid logical and numerical errors. It is commonly believed that logical errors are more difficult to evaluate, while MCTS-based methods are believed a competitive solution to this challenge by collecting such preferences. In this section, we investigate the role of natural language and mathematical language in assessing the logical coherence included in pure mathematical language by comparing SSMOSRM, MO-SRM, and NT-SRM.

Specifically, if the contextual information in the input of an SRM is useful, its performance should surpass that of SSMO-SRM, which takes only the current step as input. This ability is referred to as the model’s capacity to assess logical coherence, meaning it can determine whether a subsequent step logically follows from the information and conclusions derived in the previous context. The results are shown in Table 1.

LLMs can be trained to evaluate logical coherence in pure mathematical language. For DeepSeek-Math-7BBase, MO-SRM achieves an accuracy gain of $+ 7 . 3 5 \%$ on GSM8K and $+ 8 . 4 8 \%$ on MATH, which is higher than the gains $+ 3 . 6 4 \%$ and $6 . 3 0 \%$ observed for SSMO-SRM. Similarly, for Qwen2-7B, MO-SRM achieves an accuracy gain of $+ 5 . 3 1 \%$ on GSM8K and $+ 3 . 9 4 \%$ on MATH, higher than that of SSMO-SRM $+ 3 . 1 8 \%$ and $+ 1 . 9 2 \%$ . This substantial difference indicates that MO-SRM, which considers the full sequence of mathematical expressions, is effective at capturing logical coherence, rather than only focusing on the current step. This finding indicates that logical coherence in mathematical language can be assessed by LLMs as SRMs.

The SRMs have difficulties being trained to evaluate the logical coherence in the form of natural language. Based on our MDP definition, even after the mathematical expressions are stripped away from the current reasoning step, the natural language descriptions still include the details of the actions to be executed. In other words, the SRMs should be able to learn from these constructed preferences to identify which actions are useful for problem-solving. However, as shown in Figure 3, the dashed curves illustrate the challenges in training NT-SRMs, which were designed to evaluate the quality of the next thoughts. The training processes across various datasets and base models consistently demonstrate the difficulty in identifying preferences based solely on the descriptions of thoughts during reward training. The results presented in Table 1 further highlight the poor performance of NT-SRMs when used as scoring functions. These findings suggest that the implicit logic conveyed through natural language is difficult for LLMs to capture and evaluate effectively.

Additional Analysis   

<html><body><table><tr><td rowspan="2">Agent&WorldModel Llama-3-7OB-Instruct</td><td colspan="2">Accuracy (Gain) %</td></tr><tr><td>GSM8K</td><td>MATH</td></tr><tr><td>Pass @1 (3-shots)</td><td>90.37 (+0.00)</td><td>48.48</td></tr><tr><td>+GS /wMO-SRM1</td><td>92.95 (+2.58)</td><td>(+0.00) 54.12 (+5.64)</td></tr></table></body></html>

![](images/ebf6947619d4f2ed147f26178783d77320e2ac2b0212dc9508d1b520abd620ec.jpg)  
Figure 5: The performance of SRM is affected by the ability of the base model.

Supervising a larger model Despite being trained on preference data generated by a smaller model, the MO-SRM was able to effectively guide the reasoning process of a larger model and achieve substantial improvements $( + 2 . 5 8 \%$ on GSM8K and $+ 5 . 6 4 \%$ on MATH) (Table 2). This further illustrates the ability of the SRMs to focus exclusively on mathematical language.

Effect of base models for MO-SRM The choice of SRM base models impacts performance (Figure 5), while this effect doesn’t appear to be entirely related to the base model’s mathematical abilities. Despite its excellent mathematical capabilities, The surprising underperformance of Llama-3-8B compared to Llemma-7B (Azerbayev et al. 2023), Qwen2-7B, and DeekSeek-Math-7B-Base, suggests that factors beyond just original mathematical ability are at play. This might be due to the challenges in self-assessment or other reasons to be explored.

Table 2: Supervise a larger model (Llama-3-70B-Instruct).   
Table 3: Effect of $B$ and $c$ on beam search   

<html><body><table><tr><td rowspan="2">Agent&WorldModel Llama-3-8B-Instruct</td><td colspan="2">Accuracy</td></tr><tr><td>GSM8K</td><td>Accuracy</td></tr><tr><td>+BSw/MO-SRMl</td><td></td><td></td></tr><tr><td>B=1,c=5</td><td>85.82</td><td>39.64</td></tr><tr><td>B=1,c=10</td><td>85.90</td><td>40.06</td></tr><tr><td>B=3,c=10</td><td>88.17</td><td>40.24</td></tr></table></body></html>

Effect of $B$ and $c$ on beam search Increasing the beam size $B$ and the number of candidate count $c$ will slightly improve accuracy, but this improvement will eventually plateau, as shown in Table 3.

# Conclusion

Our investigation into the role of natural language and mathematical expressions in step-level reward modeling reveals that natural language descriptions are not essential for the success of these models. Through extensive experiments, we demonstrated that reward models operating solely on mathematical expressions perform comparably to those that incorporate both natural language and math. Furthermore, the difficulty in training models to evaluate the coherence of natural language thought processes underscores the challenges LLMs face in capturing implicit logical structures through language alone. We also found that the coherence of logical structure inherent in mathematical expressions can be assessed by SRMs trained based on LLMs. Given the overhead of obtaining step-level rewards, these findings offer new insights for developing more efficient and targeted reward models by isolating the most impactful components of mathematical reasoning steps.