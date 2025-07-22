# SkillTree: Explainable Skill-Based Deep Reinforcement Learning for Long-Horizon Control Tasks

Yongyan Wen1, Siyuan $\mathbf { L i } ^ { 1 \ast }$ , Rongchang $\mathbf { Z } \mathbf { u } \mathbf { o } ^ { 1 }$ , Lei $\mathbf { Y u a n } ^ { 2 , 3 , 4 }$ , Hangyu Mao5, Peng Liu1

1Faculty of Computing, Harbin Institute of Technology 2National Key Laboratory of Novel Software Technology, Nanjing University 3School of Artificial Intelligence, Nanjing University 4Polixir Technologies 5Kuaishou Technology 23B903027@stu.hit.edu.cn, siyuanli@hit.edu.cn

# Abstract

Deep reinforcement learning (DRL) has achieved remarkable success in various domains, yet its reliance on neural networks results in a lack of transparency, which limits its practical applications in safety-critical and human-agent interaction domains. Decision trees, known for their notable explainability, have emerged as a promising alternative to neural networks. However, decision trees often struggle in longhorizon continuous control tasks with high-dimensional observation space due to their limited expressiveness. To address this challenge, we propose SkillTree, a novel hierarchical framework that reduces the complex continuous action space of challenging control tasks into discrete skill space. By integrating the differentiable decision tree within the highlevel policy, SkillTree generates discrete skill embeddings that guide low-level policy execution. Furthermore, through distillation, we obtain a simplified decision tree model that improves performance while further reducing complexity. Experiment results validate SkillTree’s effectiveness across various robotic manipulation tasks, providing clear skill-level insights into the decision-making process. The proposed approach not only achieves performance comparable to neural network based methods in complex long-horizon control tasks but also significantly enhances the transparency and explainability of the decision-making process.

Code — https://github.com/YongyanWen/SkillTree

# Introduction

Deep Reinforcement learning (DRL) has been shown as a powerful framework for tackling complex decision-making tasks, achieving remarkable success in various domains such as games (Mnih et al. 2015; BAAI 2023), robotic manipulation (Akkaya et al. 2019; Jitosho et al. 2023), and visual navigation (Kulha´nek, Derner, and Babusˇka 2021). Despite these advancements, the black-box nature of neural networks poses significant challenges in understanding and trusting their decision making processes. This lack of transparency is particularly concerning in safety-sensitive and human-agent interaction applications, where understanding the rationale behind decisions is essential (Hickling et al. 2023). For example, in the autonomous driving domain, only if human users can understand the driving policies can they trustfully use them in their cars.

Explainable reinforcement learning (XRL) (Heuillet, Couthouis, and D´ıaz-Rodr´ıguez 2021; Hickling et al. 2023; Milani et al. 2024) aims to enhance the transparency and explainability of DRL models. XRL methods can be broadly categorized into inherently explainable models, where explainability is induced in the model training process, and post-hoc explanations, where models are explained after training. Existing works have shown that using a decision tree (DT) (Costa and Pedreira 2023) as the underlying model is effective (Frosst and Hinton 2017; Bastani, Pu, and SolarLezama 2018; Ding et al. 2020; Vasic´ et al. 2022). In a DT, observations are input at the root node, and decisions are determined by conditional branches leading to the leaf nodes, which provide the final outputs. This simple structure gives DTs high explainability, offering clear and straightforward decision paths. Unlike post-hoc methods that rely on external algorithms to interpret black-box models, DT-based methods embed transparency directly within the model. The DT-based XRL paradigm not only enhances clarity but also simplifies debugging and validation, as the rationale behind each decision is explicitly encoded within the tree.

Despite their advantages, DT-based methods are not suitable for the following challenging task settings. (a) Longhorizon tasks: The temporally-extended decision processes require large and complex trees, which are difficult to optimize (Liu et al. 2021). (b) High-dimensional state spaces: DTs often lack sufficient representational ability to effectively manage high-dimensional state spaces, leading to suboptimal performance in these environments (Bastani, Pu, and Solar-Lezama 2018; Ding et al. 2020). (c) Continuous action spaces: The limited number of leaf nodes constrains DTs’ ability to encode continuous control policies optimally, particularly in complex robotic tasks. These limitations restrict the applicability of DT-based RL methods in complex environments, highlighting the need for approaches that can manage high-dimensional spaces and long-horizon tasks.

To address the limitations of traditional DTs in handling long-horizon, high-dimensional, and continuous control tasks, we propose a novel framework called SkillTree. SkillTree introduces the concept of skills, which represent temporal abstractions of low-level actions in the context of RL. These skills simplify the decision-making process by breaking down long trajectories into manageable segments, enabling the agent to plan at a higher level, thereby reducing the complexity associated with extended tasks. However, the challenge of skill representation arises, as skills are often represented in continuous spaces, leading to difficulties in control and explainability. To overcome this, our framework leverages hierarchical structures and discrete skill representation learning. By regularizing the skill space into discrete units, we simplify policy learning for skill selection and enhance the explainability of the learned skills. Specifically, we employ decision trees to index and execute these discrete skills, combining the inherent explainability of decision trees with the flexibility of skill-based representations. This approach provides a robust solution for managing complex, long-horizon decision tasks while offering clear, skilllevel insights into the decision-making process. We summarize the main contributions of this paper as follows:

1. We propose SkillTree, a novel hierarchical, skill-based method for explainable reinforcement learning. To the best of our knowledge, it marks the first successful application of DT-based explainable method in long-horizon continuous control tasks.   
2. We introduce a method for discrete skill representation learning, which effectively reduces the skill space and improves the efficiency of skill-based policy learning.   
3. Experiment results across a variety of robotic manipulation tasks demonstrate that our method provides skilllevel explanations while achieving performance comparable to neural network based approaches.

# Related Work

# Explainable RL

Recently, explainability approaches in DRL have been broadly categorized into intrinsic and post-hoc explanations, based on the method of their generation. We briefly discuss these two classes of methods here. For a more detailed taxonomy and discussion of XRL, please refer to the following surveys: (Heuillet, Couthouis, and D´ıaz-Rodr´ıguez 2021; Hickling et al. 2023; Milani et al. 2024).

Intrinsic explanation methods are designed with explainability as a fundamental aspect, often incorporating simpler, more transparent models such as decision trees (Liu et al. 2019; Silva et al. 2020; Ding et al. 2020), linear models (Rajeswaran et al. 2017; Molnar, Casalicchio, and Bischl 2020; Wabartha and Pineau 2023) or symbolic rules (Lyu et al. 2019; Ma et al. 2021). By ensuring that the model itself is explainable, intrinsic methods offer real-time, straightforward explanations of the agent’s decisions. For instance, using a decision tree as the policy model enables immediate and clear understanding of the decision-making process.

Post-hoc methods apply explainability techniques after the RL model has been trained. These approaches do not alter the original model but instead seek to explain the decisions of more complex, often black-box models. Typical post-hoc methods include feature attribution methods like saliency maps (Greydanus et al. 2018; Anderson et al. 2019; Olson et al. 2021) and Shapley value (Zhang et al. 2021; Heuillet, Couthouis, and D´ıaz-Rodr´ıguez 2022), which highlight the most influential input features. Model distillation approximates the black-box model, either locally or globally, with explainable models like decision trees (Bastani, Pu, and Solar-Lezama 2018; Coppens et al. 2019; Bewley and Lawry 2021; Orfanos and Lelis 2023). Counterfactual explanations generate alternative scenarios by modifying input features to observe changes in the model’s output, offering a way to understand the sensitivity of decisions (Madumal et al. 2020; Olson et al. 2021). Some studies have explored identifying critical states crucial for achieving the final reward from a state-reward perspective (Guo et al. 2021; Cheng et al. 2024). Additionally, researchers have analyzed the impact of different training samples on policy training outcomes (Deshmukh et al. 2023). Lastly, rule extraction methods create human-readable rules that approximate the model’s behavior and enhance the transparency of the decision-making process (Hein, Udluft, and Runkler 2018; Landajuela et al. 2021).

# Decision Tree

Decision trees are widely used for their explainability and simplicity, often serving as function approximators in reinforcement learning. Classical decision tree algorithms like CART (Breiman et al. 1984) and C4.5 (Quinlan 1993) produce explainable surrogate policies but are limited in expressiveness and impractical for integration into DRL models. Approaches such as VIPER (Bastani, Pu, and Solar-Lezama 2018) attempt to distill neural network policies into verifiable decision tree by imitation learning. Recent advancements include rule-based node divisions (Dhebar and Deb 2020) and parametric differentiable decision tree, such as soft decision tree (Frosst and Hinton 2017) and differentiable decision tree for approximating Q-function or policy (Silva et al. 2020). While these methods improve expressiveness, they are generally constrained to simpler, lowdimensional environments. By contrast, our approach addresses the complexity of high-dimensional continuous control tasks by converting the action space into a skill space, thereby reducing complexity and providing a more effective and explainable solution for challenging tasks.

# Skills in Reinforcement Learning

Recent works have increasingly focused on learning and utilizing skills to improve agent efficiency and generalization (Shu, Xiong, and Socher 2018; Hausman et al. 2018; Shankar and Gupta 2020; Lynch et al. 2020). Skills can be extracted from the offline dataset (Shiarlis et al. 2018; Kipf et al. 2019; Pertsch, Lee, and Lim 2021; Pertsch et al. 2021; Shi, Lim, and Lee 2023) or manually defined (Lee, Yang, and Lim 2019; Dalal, Pathak, and Salakhutdinov 2021; BAAI 2023). Typically, these skills are represented as highdimensional, continuous latent variables, which limits their explainability. In contrast, we propose extracting a discrete skill representation from a task-agnostic dataset, making it suitable for DT-based policy learning.

# Preliminary

# Reinforcement Learning

In this paper, we are concerned with a finite Markov decision process (Sutton and Barto 2018), which can be represented as a tuple $M = ( S , A , R , \mathcal { P } , p _ { 0 } , \gamma , H )$ . Where $s$ is the state space, $\mathcal { A }$ is the action space, $\mathcal { S } \times \mathcal { A } \mapsto \mathbb { R }$ is the reward space, $\mathcal { P } : \mathcal { S } \times \mathcal { A } \mapsto \Delta ( \mathcal { S } )$ is the transfer function, $p _ { 0 } : \Delta ( S )$ is the initial state distribution, $\gamma \in ( 0 , 1 )$ is the discount factor, and $H$ is the episode length. The learning objective is to find the optimal policy $\pi$ maximizing the expected discounted return

$$
\operatorname* { m a x } \mathcal { I } ( \pi ) = \operatorname* { m a x } _ { \pi } \mathbb { E } _ { s _ { 0 } \sim p _ { 0 } , ( s _ { 0 } , a _ { 0 } , \dots , s _ { H } ) \sim \pi } \left[ \sum _ { t = 0 } ^ { H - 1 } \gamma ^ { t } \mathcal { R } ( s _ { t } , a _ { t } ) \right] .
$$

# Soft Decision Tree

The differentiable decision tree (Frosst and Hinton 2017) is a special type of decision tree that differs from the traditional ones like CART (Breiman et al. 1984) by using probabilistic decision boundaries instead of deterministic boundaries. This modification increases the flexibility of model by allowing for smoother transitions at each decision node, as shown in Figure 1. The differentiable soft decision tree is a complete binary tree in which each inner node can be represented by a learnable weight $\omega _ { j } ^ { i }$ and bias $\phi _ { j } ^ { i }$ . Here, $i$ and $j$ represent the layer index and the node index of that layer, respectively. By using the sigmoid function $\sigma ( \cdot )$ , the probability $\sigma ( \omega x + \phi )$ represents the transfer probability from that node to the left subtree, and $1 - \sigma ( \omega x + \phi )$ gives the transfer probability to the right subtree. This process allows the decision tree to provide a “soft” decision at each node for the current input.

Consider a decision tree with depth $d$ . The nodes in the tree are denoted by $n _ { u }$ , where $u = \bar { 2 } ^ { i } - 1 + j$ denotes the node index, and the decision path $P$ can be represented as

$$
P = \arg \operatorname* { m a x } _ { \{ u \} } \prod _ { i = 0 } ^ { d - 1 } \prod _ { j = 0 } ^ { 2 ^ { i } } p _ { \lfloor \frac { j } { 2 } \rfloor  j } ^ { i  i + 1 }
$$

where $\{ u \}$ denotes nodes on the decision path, and $p _ { \lfloor \frac { j } { 2 } \rfloor  j } ^ { i  i + 1 }$ denotes the probability of moving from node $n _ { 2 ^ { i } + \lfloor j / 2 \rfloor }$ to node $n _ { 2 ^ { i + 1 } + j }$ , the probability is calculated as

$$
p _ { \lfloor \frac { j } { 2 } \rfloor  j } ^ { i  i + 1 } = \{ \begin{array} { l l } { \sigma ( \omega _ { \lfloor \frac { j } { 2 } \rfloor } ^ { i } x + \phi _ { \lfloor \frac { j } { 2 } \rfloor } ^ { i } ) } & { \mathrm { i f ~ } j \mod 2 = 0 , } \\ { 1 - \sigma ( \omega _ { \lfloor \frac { j } { 2 } \rfloor } ^ { i } x + \phi _ { \lfloor \frac { j } { 2 } \rfloor } ^ { i } ) } & { \mathrm { o t h e r w i s e . } } \end{array} 
$$

With Equation 2 and 3, the tree can be traversed down to the leaf nodes. Leaf nodes are represented by parameter vectors $\mathbf { w } ^ { 2 ^ { d } \times K }$ , where $K$ is the number of output categories. The output of the leaf node $k$ is a categorical distribution given by softmax $\left( \mathbf { w } _ { k } \right)$ , which is independent of the input.

![](images/f6460a43fcd187af081f697a192c7ac8e922aa45b5de5bc893be16b4bbeb90fe.jpg)  
Figure 1: Comparison of the soft decision tree (left) and the hard decision tree (right).

The learnable parameters of both the leaf nodes and the inner nodes can be optimized using existing DRL algorithms, making the soft decision tree suitable as an explainable alternative model for DRL policy.

# Explainable RL with SkillTree

Our goal is to leverage discrete skill representations to learn a skill decision tree, enabling skill-level explainability. Overall, our approach is divided into three stages: (1) extracting the discrete skill embeddings from the offline dataset, (2) training an explainable DT-based skill policy by RL for downstream long-horizon tasks, and (3) distilling the trained policy into a simplified decision tree.

# Learning Discrete Skill Embeddings

Our goal is to obtain a fixed number of skill embeddings that contain a temporal abstraction of a sequence of actions. Skills represent action sequences of useful behaviors and are represented by $D$ -dimensional vectors. We assume an existing task-agnostic dataset $\boldsymbol { \mathcal { D } } \ = \ \{ \tau _ { i } \} , \tau _ { i } \ =$ $\{ s _ { 0 } , a _ { 0 } , \ldots , s _ { H _ { i } } , a _ { H _ { i } } \}$ , consists of $d$ trajectories of varying lengths, where each trajectory includes states $s _ { t } , \ldots , s _ { H _ { i } }$ and corresponding actions $a _ { t } , \dots , a _ { H _ { i } }$ . To regularize the skill embeddings, instead of learning within the continuous skill space, we employ a skill table $Z = \{ e _ { 1 } , e _ { 2 } , . . . , e _ { K } \}$ (i.e., codebook) that contains $D$ -dimensional vectors, each representing a learnable skill embedding.

To learn reusable skills from offline datasets and guide exploration efficiently, we modified VQ-VAE (Van Den Oord, Vinyals et al. 2017) to learn skill representations and skill prior, as illustrated in Figure 2. First, a demonstration trajectory is randomly divided into state-action segments of fixed length $h$ during the training. Each segment contains $h$ observation states $\mathbf { s } _ { t , h } = \boldsymbol { s } _ { t } , \boldsymbol { s } _ { t + 1 } , \ldots , \boldsymbol { s } _ { t + h - 1 }$ and the corresponding actions $\mathbf { a } _ { t , h } ~ = ~ a _ { t } , a _ { t + 1 } , \ldots , a _ { t + h - 1 }$ . The input to the encoder $q _ { \phi }$ is $\mathbf { s } _ { t , h }$ and $\mathbf { a } _ { t , h }$ . The output of the encoder is a $D$ -dimensional embedding $z _ { e }$ . Next, we select the embedding from the codebook that is nearest to $z _ { e }$ and its index $k \stackrel { \textstyle \cdot } { = } \arg \operatorname* { m i n } _ { i } \| z _ { e } - e _ { j } \| _ { 2 }$ , obtaining the regularized embedding $z _ { q } = e _ { k }$ . Finally, the low-level actual action $\boldsymbol { a } _ { t }$ is obtained through the state-conditioned decoder $\pi _ { l } ( s _ { t } , z _ { q } )$ , which serves as the low-level policy.

In addition, following the approach of (Pertsch, Lee, and Lim 2021), we introduce a skill prior to guide the highlevel policy. The skill prior $p ( k _ { t } | s _ { t } )$ predicts the skill distribution given the first state of the sequence in the offline data. This enhances exploration efficiency during high-level policy training by avoiding random skill sampling during warming up. Here, $s _ { t }$ represents the first state of the sequence. The skill prior is a $K$ -categorical distribution that matches the output of encoder. The objective of the skill prior is to fit the output distribution of encoder conditioned solely on the first state $s _ { t }$ . Since the encoder can only produce a fixed skill index $k$ , the skill prior aims to match a onehot distribution of that index, aligning its predictions with the encoder. The overall training objective is

![](images/b0001cc7c3274ebef71ed85d54d54b3d124f9132eea9e30a0b8a85c5de760489.jpg)  
Figure 2: Discrete skill embedding learning and downstream high-level DT policy learning. After completing the skill learning, we freeze the decoder and skill prior, and then proceed to finetune the codebook during the high-level policy learning.

$$
\begin{array} { l } { \displaystyle \mathcal { L } = \sum _ { t = 0 } ^ { h - 1 } { \| \pi _ { l } \left( s _ { t } , z _ { q } \right) - a _ { t } \| _ { 2 } ^ { 2 } } } \\ { \displaystyle \quad + \| \operatorname { s g } [ z _ { e } ] - e \| _ { 2 } ^ { 2 } + \beta \| z _ { e } - \operatorname { s g } [ e ] \| _ { 2 } ^ { 2 } } \\ { \displaystyle \quad - \sum _ { k = 1 } ^ { K } q _ { \phi } ( z = k | s _ { t , h } , a _ { t , h } ) \log p ( k _ { t } = k | s _ { t } ) } ,  \end{array}
$$

where $\mathrm { s g } [ \cdot ]$ is the stopgradient operator, $p ( k _ { t } = k | s _ { t } )$ is the probability that $k _ { t }$ is category $k$ and $\beta$ is the hyperparameter of the commitment loss term. Categorical distribution $q _ { \phi } ( z = k | \boldsymbol { s } _ { t , h } , \boldsymbol { a } _ { t , h } )$ probabilities are defined as one-hot as

$$
q _ { \phi } ( z = k | \mathbf { s } _ { t , h } , \mathbf { a } _ { t , h } ) = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { f o r ~ } } k = \arg \operatorname* { m i n } _ { j } \| z _ { e } - e _ { j } \| _ { 2 } , } \\ { 0 } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. }
$$

where $z _ { e }$ is the output of the encoder.

# Downstream RL Policy Learning

To reuse the learning skill embeddings and improve exploration efficiency in RL training, we utilize the skill prior to regularize the high-level policy in skill space. After pretraining the low-level policy and skill prior, the low-level policy $\bar { \pi } _ { l } ( s _ { t } , z _ { t } )$ and the skill prior $p ( k _ { t } | s _ { t } )$ are fixed. The high-level policy $\pi _ { h } ( k _ { t } | s _ { t } )$ is executed every $h$ -steps, and the output is the categorical distribution of the skill index $k _ { t }$ . To obtain the actual skill embedding, we sample an index $k _ { t }$ from the distribution and then query the codebook with $k _ { t }$ , i.e., $z _ { t } = Z [ k _ { t } ]$ . In each single time step, the low-level policy predicts the output action conditioned on the state and skill embedding. The low-level policy executes $h$ -steps and the state changes to $s _ { t + h }$ . We use $\mathcal { P } ^ { + } ( s _ { t + h } \vert s _ { t } , z _ { t } )$ to denote the state transition after $h$ steps (see line 3-6 in Algorithm 1). Next, in order to improve the training efficiency by utilizing the prior information obtained from previous skill training, we add an additional prior term as the learning objective:

$$
J ( \theta ) = \mathbb { E } _ { \pi _ { h } } \left[ \sum _ { t = 0 } ^ { H - 1 } \gamma ^ { t } r ^ { + } ( s _ { t } , z _ { t } ) - \alpha D _ { \mathrm { K L } } \left( \pi _ { h } ( k _ { t } | s _ { t } ) \| p ( k _ { t } | s _ { t } ) \right) \right] ,
$$

where $\begin{array} { r } { r ^ { + } ( s _ { t } , z _ { t } ) = \sum _ { i = t } ^ { t + h } \mathcal { R } ( s _ { i } , z _ { i } ) } \end{array}$ is the total sum of rewards, $\theta$ is the trainable parameter of policy, $\alpha$ is the hyperparameter of divergence term. To prevent the policy from overfitting the prior, following (Pertsch, Lee, and Lim 2021), $\alpha$ can be automatically tuned by defining a target divergence $\delta$ (see Algorithm 1 line 15).

Typically, the high-level policy is implemented as a neural network, which is a black box and lacks transparency. In order to achieve an explainable decision making process, we implement the high-level policy $\pi _ { h } ( k _ { t } | s _ { t } )$ using a differentiable soft decision tree instead, which can be optimized using existing backpropagation algorithm. Furthermore, the structure of the skill prior is identical to that of the highlevel policy, both of which are implemented using soft decision trees. To improve learning efficiency, the parameters of the skill prior are used to initialize the policy during the initialization of training. We modify SAC (Haarnoja et al. 2018) algorithm to learn the objective 6 and the process is summarized in Algorithm 1. See Appendix for more details.

Inputs: codebook $Z$ , target divergence $\delta$ , learning rates $\lambda _ { \theta } , \lambda _ { \psi } , \lambda _ { \alpha }$ , target update rate $\tau$ , DT skill prior $p ( k _ { t } | \bar { s } _ { t } )$ and discount factor .

1: Initialize replay buffer $\boldsymbol { B }$ , $d$ -depth high-level DT policy   
$\pi _ { \boldsymbol { \theta } } ( k _ { t } | \boldsymbol { s } _ { t } )$ , critic $Q _ { \psi } ( s _ { t } , z _ { t } )$ , target network $Q _ { \bar { \psi } } ( s _ { t } , z _ { t } )$   
2: for each iteration do   
3: for every $h$ environment steps do   
4: $\begin{array} { r l } & { k _ { t } \sim \pi _ { \theta } ( k _ { t } | s _ { t } ) } \\ & { z _ { t } = Z [ k _ { t } ] } \\ & { s _ { t ^ { \prime } } \sim \mathcal { P } ^ { + } ( s _ { t + h } | s _ { t } , z _ { t } ) } \\ & { \mathcal { B }  \mathcal { B } \cup \{ s _ { t } , z _ { t } , r _ { t } ^ { + } , s _ { t ^ { \prime } } \} } \end{array}$   
5:   
6:   
7:   
8: end for   
9: for each gradient step do   
10: $\begin{array} { r l } & { k _ { t ^ { \prime } } \sim \overline { { \pi } } _ { \theta } \big ( k _ { t ^ { \prime } } | s _ { t ^ { \prime } } \big ) \cdot } \\ & { z _ { t ^ { \prime } } = Z \big [ k _ { t } ^ { \prime } \big ] } \\ & { \bar { Q } = r _ { t } ^ { + } + \gamma \Big [ Q _ { \bar { \psi } } ( s _ { t ^ { \prime } } , z _ { t ^ { \prime } } ) + \alpha D _ { \mathrm { K L } } ( \pi _ { \theta } ( k _ { t ^ { \prime } } | s _ { t ^ { \prime } } ) \| p ( k _ { t ^ { \prime } } | s _ { t ^ { \prime } } ) ) \Big ] } \\ & { \theta \gets \theta - \lambda _ { \theta } \nabla _ { \theta } \left[ Q _ { \bar { \psi } } ( s _ { t } , z _ { t } ) - D _ { \mathrm { K L } } ( \pi _ { \theta } ( k _ { t } | s _ { t } ) \| p ( k _ { t } | s _ { t } ) ) \right] } \\ & { \psi \gets \psi - \lambda _ { \psi } \nabla _ { \psi } \left[ \frac { 1 } { 2 } \left( Q _ { \psi } \big ( s _ { t } | z _ { t } \big ) - \bar { Q } \right) ^ { 2 } \right] } \\ & { \underline { { \alpha } } \gets \alpha - \lambda _ { \alpha } \nabla _ { \alpha } \left[ \alpha \left( D _ { \mathrm { K L } } \big ( \pi _ { \theta } \big ( k _ { t } | s _ { t } \big ) \| p \big ( k _ { t } | s _ { t } \big ) \right) - \delta \big ) \right] } \\ & { \bar { \psi } \gets \tau \psi + ( 1 - \tau ) \bar { \psi } } \end{array}$   
11:   
12:   
13:   
14:   
15:   
16:   
17: end for   
18: end for   
19: return policy $\pi _ { \boldsymbol { \theta } } ( k _ { t } | \boldsymbol { s } _ { t } )$ , finetuned codebook $Z$

# Distilling the Decision Tree

After downstream RL training, we obtain a soft decision tree with skill-level explainability, where each decision node probabilistically selects child nodes until a skill choice is made. Generally, models with fewer parameters are easier to understand and accept. To further simplify the model structure, we distill the soft decision tree into a more straightforward hard decision tree through discretization. Previous methods select the feature with the highest weight at each node but it often obviously degrades tree performance (Ding et al. 2020). By framing the problem as an imitation learning task, we leverage the learned high-level policy as an expert policy. We sample trajectories from the environment to generate the dataset. Given the simplified skill space, we employ a low-depth CART to classify state-skill index pairs. This approach effectively manages the complexity of the tree while preserving performance, ensuring that the model remains both explainable and efficient.

# Experiments

In this section, we evaluate SkillTree’s effectiveness in achieving a balance between explainability and performance. We demonstrate that our method not only provides clear, skill-level explanations but also achieves performance on par with neural network based approaches.

# Tasks Setup

To evaluate the performance of our method in long-horizon sparse reward tasks, we chose the Franka Kitchen control tasks in D4RL (Fu et al. 2020), robotic arm manipulation task in CALVIN (Mees et al. 2022) and Office cleaning task (Pertsch et al. 2021), as illustrated in Figure 3.

![](images/ea128c6881fde643c45772e121456286e8f3e38ed28c8cfc4eddd89d0282f7aa.jpg)  
Figure 3: Four long-horizon sparse reward tasks to evaluate. (a) The robotic arm has to finish four subtasks in the correct order, i.e., Microwave - Kettle - Bottom Burner - Light (MKBL). (b) Similar to (a), but with different subtasks: Microwave - Ligt - Slide Cabinet - Hinge Cabinet (MLSH). (c) Finish subtasks in the correct order, i.e., Open Drawer - Turn on Lightbulb - Move Slider Left - Turn on LED. (d) In the office cleaning task, the robotic arm needs to pick up objects and place them in corresponding containers in sequence.

Kitchen The Franka Kitchen environment is a robotic arm controlled environment based on the mujoco implementation. The environment contains a 7-DOF robotic arm that is able to interact with other objects in the kitchen, such as a kettle and other objects. A total of 7 subtasks are included. The observation is 60-dimensional and contains information about joint positions as well as task parameters, and the action space is 9-dimensional with episode length of 280. The offset dataset contains 601 trajectories, and each trajectory completes a variety of tasks that may not be in the same order, but no more than 3 subtasks. We set two sets of different subtasks: MKBL and MLSH, as shown in Figure 3(a) and 3(b). The MLSH task is more difficult to learn due to the very low frequency of subtask transitions in the dataset.

CALVIN In CALVIN, a 7-DOF Franka Emika Panda robotic arm needs to be controlled to open a drawer, light a light bulb, move a slider to the left and light an LED in sequence. the observation space is 21-dimensional and contains both robotic arm states and object states with episode length of 360 steps. The dataset contains 1,239 trajectories

SkillTree (Ours) SPiRL VQ-SPiRL BC+SAC Kitchen MKBL Kitchen MLSH CALVIN Office 12 WWW 12 01234Average Subtasks 012Average Subtasks wwwaowm Ww 752 wwwWM 0 W 0.0 0.5 1.0 1.5 2.01e6 0.0 0.5 1.0 1.5 2.01e6 0 1 2 0.0 0.5 1.0 1.51e6 1e6 Environment Steps Environment Steps Environment Steps Environment Steps

Table 1: Average completed subtasks (ACS) and the number of leaf nodes (Leaf) across methods and domains.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="2">Kitchen MKBL</td><td colspan="2">Kitchen MLSH</td><td colspan="2">CALVIN</td><td colspan="2">Office</td></tr><tr><td>ACS</td><td>Leaf</td><td>ACS</td><td>Leaf</td><td>ACS</td><td>Leaf</td><td>ACS</td><td>Leaf</td></tr><tr><td>SPiRL</td><td>3.08 ± 0.46</td><td>N/A</td><td>2.82 ± 1.08</td><td>N/A</td><td>2.80 ± 1.42</td><td>N/A</td><td>1.48 ± 0.96</td><td>N/A</td></tr><tr><td>VQ-SPiRL</td><td>2.97 ± 0.41</td><td>N/A</td><td>2.04 ± 0.85</td><td>N/A</td><td>2.10 ± 0.81</td><td>N/A</td><td>1.69 ± 0.64</td><td>N/A</td></tr><tr><td>BC+SAC</td><td>0.00 ± 0.00</td><td>N/A</td><td>0.00 ±0.00</td><td>N/A</td><td>0.01 ± 0.10</td><td>N/A</td><td>0.00 ± 0.00</td><td>N/A</td></tr><tr><td>CART</td><td>2.11 ± 0.92</td><td>987</td><td>1.16 ± 0.78</td><td>981</td><td>1.02 ± 0.20</td><td>998</td><td>0.00 ± 0.00</td><td>949</td></tr><tr><td>CART+DC</td><td>1.32 ± 0.66</td><td>970</td><td>1.20 ± 0.88</td><td>985</td><td>1.49 ± 0.50</td><td>1006</td><td>0.00 ± 0.00</td><td>961</td></tr><tr><td>SkillTree</td><td>2.90 ± 0.42</td><td>64</td><td>2.58 ± 0.80</td><td>64</td><td>1.90 ± 0.90</td><td>64</td><td>2.00 ± 0.00</td><td>64</td></tr><tr><td>SkillTree (Distillation)</td><td>3.06 ± 0.65</td><td>64</td><td>2.52 ± 0.82</td><td>64</td><td>2.42 ± 0.49</td><td>64</td><td>2.00 ± 0.00</td><td>64</td></tr><tr><td>SkillTree (DC+D)</td><td>3.25 ± 0.70</td><td>64</td><td>2.62 ± 0.76</td><td>64</td><td>3.00 ± 0.00</td><td>64</td><td>2.00 ± 0.00</td><td>64</td></tr></table></body></html>

and a total of 34 subtasks. Since it contains a large number of subtasks, the average frequency of transitions between any subtasks in the dataset is very low, requiring more precise skill selection, which poses a greater challenge during the learning of the high-level policy.

Office For the Office environment, a 5-DOF WidowX robotic arm needs to be controlled to pick up multiple objects and place them in their corresponding containers with episode length of 350. We utilized a dataset collected by (Pertsch et al. 2021), which includes 2,331 trajectories. The state space is 97-dimensional and the action space is 8- dimensional. Due to the freedom of object manipulation in this environment, the task is more challenging to complete.

# Baselines

In the experiments, we compare our proposed method against several representative baselines to demonstrate its effectiveness and competitive performance with additional skill-level explainability.

• SPiRL (Pertsch, Lee, and Lim 2021): Learns continuous skills embeddings and a skill prior, and guides a highlevel policy using the learned prior. It serves as a competitive skill-based RL algorithm but lacks explainability. • VQ-SPiRL: Discretizes the skills in SPiRL but uses neural networks as the high-level policy model structure. • Behavioral Cloning $^ +$ Soft Actor-Critic $\mathrm { ( B C + S A C ) }$ : Trains a supervised behavioral cloning policy from the demonstration dataset and finetunes it on the downstream task using Soft Actor-Critic (Haarnoja et al. 2018). It serves as a general algorithm without leveraging skills.

• CART (Breiman et al. 1984): Imitation learning using CART with trained SPiRL agent as the teacher. • CART $+ \mathrm { D C }$ : CART is trained on the data generated by SPiRL after applying data cleaning.

For SkillTree, we set the depth of the tree to 6 in all domains to balance performance and explainability. The size of codebook $K$ is 16 for Kitchen and Office, and 8 for CALVIN. We set skill horizon $h$ to 10 for all experiments. For CART, we sample 1,000 trajectories on each domain to train and set the maximum depth to 10.

# Results

From the learning curves shown in Figure 4, it is evident that our method demonstrates comparable learning efficiency and asymptotic performance in all domains. This highlights the effectiveness of our discrete skill representation in handling complex, long-horizon tasks. $\mathrm { B C + S A C }$ failed to complete any subtasks because of the sparse reward. In the CALVIN and Office domain, our method exhibits a marked improvement in early learning efficiency compared to SPiRL and performs comparably to VQ-SPiRL. This advantage underscores the benefit of discrete skills in enhancing exploration efficiency, allowing our approach to quickly identify and leverage useful skills for target task. Through comparing to VQ-SPiRL, we validate that the decision tree can maintain performance while reducing model complexity in simplified discrete skill space.

In Table 1, we evaluate the performance of SkillTree after applying data cleaning (DC) and distillation on a sampled dataset. We consider two sampling methods: directly using

![](images/eeb01f00e27bdea637cbfb5a3f7982e22cdc4721ffc3b2cc8b26549a30f3d90f.jpg)  
Figure 5: Visualization of the SkillTree $( \mathrm { D C + D } ) ,$ ) with depth 3. qpos and $\mathsf { q p o s \mathrm { _ - O b j } }$ mean the position of the robotic arm and objects, respectively. $n$ denotes the number of stateskill pairs in the divided decision set.

SkillTree for distillation and applying data cleaning before distillation $( \mathrm { D C + D } ) ,$ ). For data cleaning, we retain trajectories that achieve at least two completed subtasks for Kitchen and CALVIN, and at least one for Office. In each environment, we sample 1000 trajectories, ensuring consistency across tasks. To balance performance and explainability, we set the tree depth to 6 across all domains.

The results show that SkillTree $\mathrm { D C + D }$ achieves the best performance in most domains, outperforming other baselines. Data cleaning significantly improves performance by filtering low-quality trajectories and emphasizing highreward ones. For instance, in the $\mathrm { D C + D }$ setting, the model demonstrates stronger performance while maintaining simplicity. In contrast, CART $+ \mathrm { D } { \mathsf { C } }$ also benefits from data cleaning but performs worse than SkillTree, likely due to its limited expressiveness even with excessive number of leaf nodes $( \sim 1 0 0 0 )$ , which reduce explainability. We also observe that the discrete skill representation (VQ-SPiRL vs. SPiRL) does not significantly enhance performance. This is expected, as the primary goal of discrete skill representation learning is to simplify the skill space, facilitating integration with a decision tree for enhanced policy transparency and explainability, rather than improving raw performance. In comparison, the combination of data cleaning and distillation not only simplifies the SkillTree model but also preserves critical decision-making capabilities, enabling superior performance across diverse tasks.

# Explainability

In Figure 5, we visualize the final decision tree obtained after distillation for the Kitchen MKBL task. This depth of decision tree is only three but can complete an average of 3.03 subtasks. The very few parameters make the decisionmaking process clear and easy to understand. Each node makes splits based on one of the observed features. In the

![](images/63e4d86fac5f3a18264e62397412f815522f83867170b270384857904b8387ba.jpg)  
Figure 6: We fix the skill output of high-level policy and evaluate the focus on subtasks of each skill in Kitchen MKBL task. We set the number of skills $K$ to 16, as shown in the $\mathbf { \boldsymbol { x } }$ -axis. The color means the success rate of each subtask.

![](images/0ef645f696c0b82fc692cb8116e0148962976a45f36b58ad11e328c73db2c112.jpg)  
Figure 7: Skill index output visualization of an episode in Kitchen MKBL task, which finished all 4 subtasks.

60-dimensional observations of the Kitchen domain, the last 30 dimensions are task parameters that are independent of the state and thus do not influence decision making. This property is reflected in the learned tree structure because the decision nodes use only the features in the first 30 dimensions (robot arm position and object position) for splitting.

To analyze the effects of each skill, we also evaluated the effectiveness of different skills, as shown in Figure 6. We fixed the skill output individually and evaluated the completion of subtasks. Each skill is capable of completing different subtasks with varying focus. SkillTree selects the appropriate skill based on the observed state, which are then executed by the low-level policy. Figure 7 illustrates the sequence of skill indices chosen in one episode, during which four subtasks were completed. It can be seen that the same skill was repeatedly selected at consecutive time steps to complete specific subtasks, indicating the reuse of particular skills. Moreover, repeated selection of the same skill indicates a relatively long-term and stable skill execution process. By isolating and testing each skill, we can clearly see which skills are most effective for particular subtasks. This makes it easier to understand the role and functionality of each skill within the decision making process. See Appendix for more experiment results as well as visualizations.

# Conclusions and Limitations

We proposed a novel hierarchical skill-based explainable reinforcement learning framework designed to tackle the challenges of long-hoziron decision making in continuous control tasks. By integrating discrete skill representations and

DT, our method offers a robust solution that enhances explainability without sacrificing performance. The proposed approach effectively regularizes the skill space, simplifies policy learning, and leverages the transparency of the DT. Experiment results demonstrate that our method not only achieves competitive performance compared to neural network based approaches but also provides valuable skill-level explanations. This work represents a important step towards more transparent and explainable RL approaches, paving the way for their broader application in complex real-world scenarios where understanding the decision making process is essential.

Despite the promising results demonstrated, several limitations remain. First, the decision tree high-level policy is not well-suited for image observation inputs. Second, the current approach relies on an offline, task-agnostic dataset for skill learning, which may not always be accessible in practical scenarios. Future work could address these challenges by extending the approach to handle image observations, potentially by incorporating explainable AI techniques into the high-level decision-making process. Additionally, investigating how to adapt SkillTree to multi-agent domains (Yuan et al. 2023) represents an interesting direction for future research.