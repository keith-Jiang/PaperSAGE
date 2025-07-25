# Alleviating Shifted Distribution in Human Preference Alignment through Meta-Learning

Shihan $\mathbf { D o u } ^ { 1 }$ , Yan ${ { \bf { L i u } } ^ { 1 } }$ , Enyu Zhou1, Songyang Gao1, Tianlong $\mathbf { L i } ^ { 1 }$ , Limao Xiong1, Xin Zhao2, Haoxiang $\mathbf { J i a } ^ { 3 }$ , Junjie $\mathbf { Y e } ^ { 1 }$ , Rui Zheng1, Tao $\mathbf { G u i } ^ { 4 * }$ , Qi Zhang1,5, Xuanjing Huang1,6∗

1School of Computer Science, Fudan University, Shanghai, China 2Ant Group, Shanghai, China 3School of Computer Science, Peking University, Beijing, China 4Institute of Modern Languages and Linguistics, Fudan University, Shanghai, China   
5Key Laboratory of Intelligent Information Processing, Fudan University, Shanghai, China   
6Shanghai Collaborative Innovation Center of Intelligent Visual Computing, Shanghai, China shihandou $@$ foxmail.com, tgui $@$ fudan.edu.cn

# Abstract

The capability of the reward model (RM) is crucial for the success of Reinforcement Learning from Human Feedback (RLHF) in aligning with human preferences. However, as training progresses, the output space distribution of the policy model shifts. The RM, initially trained on responses sampled from the output distribution of the early policy model, gradually loses its ability to distinguish between responses from the newly shifted distribution. This issue is further compounded when the RM, trained on a specific data distribution, struggles to generalize to examples outside of that distribution. These two issues can be united as a challenge posed by the shifted distribution of the environment. To surmount this challenge, we introduce MetaRM, a novel method leveraging meta-learning to adapt the RM to the shifted environment distribution. MetaRM optimizes the RM in an alternating way, by preserving both the preferences of the original preference pairs, as well as maximizing discrimination power over new examples of the shifted distribution. Extensive experiments demonstrate that MetaRM can iteratively enhance the performance of human preference alignment by improving the RM’s capacity to identify subtle differences in samples of shifted distributions.

# Introduction

Reinforcement learning from human feedback (RLHF) provides a pivotal technique to ensure that the behavior of AI systems aligns with the intentions of their designers and the expectations of users (Bai et al. 2022; Ouyang et al. 2022; Zheng et al. 2023b). RLHF is executed in two primary stages. The initial stage involves training a reward model using preference data, which is collected from a substantial number of crowdsource workers. The second stage entails the application of reinforcement learning (RL) to fine-tune the large language model (LLM), to maximize the reward. In this process, the reward model plays a pivotal role, as its performance significantly impacts the effectiveness of alignment with human preference (Eschmann 2021; Gao, Schulman, and Hilton 2022).

However, the reward model faces generalization challenges caused by the environment distribution shifts during the RL phase, as shown in Figure 1. Specifically, as the RL training progresses, the optimization of the policy model causes shifts in its output space distribution. Consequently, the reward model, initially trained on the preference pairs derived from the output distribution of the early policy model, gradually fails to distinguish between responses from the newly shifted distribution. This issue is also mirrored in out-of-distribution (OOD) scenarios. The reward model trained on data from a specific distribution struggles to identify subtle differences in OOD samples and has poor performance on such shifted distribution (Casper et al. 2023; Wulfe et al. 2022). Although researchers propose to iteratively annotate preference pairs and fine-tune the reward model to adapt it to the shifted environment (Touvron et al. 2023; DeepSeek-AI 2024), continuously collecting new data is resource and time-intensive. The approach of efficiently adapting the reward model to the shifted distribution remains insufficiently explored.

To solve this challenge, we introduce MetaRM, a novel approach that adapts the reward model to the shifted distribution and restores its distinguishing ability by using metalearning. MetaRM utilizes an alternating optimization way to train the reward model by minimizing the loss on the original preference pairs, particularly those data that can maximize the discrimination power to responses of the targetshifted distribution. In this way, we can bridge the gap between the preference data distribution and the target-shifted distribution of the environment. It ensures that the reward model not only performs well on the original preference distribution but also can distinguish the differences in samples of the target-shifted distribution. In terms of implementation, MetaRM can constantly enhance the performance of human preference alignment by iteratively adapting the reward model to the output distribution of the new policy model, achieving an iterative RLHF. Additionally, MetaRM also improves the capability of the reward model in OOD scenarios by training only on original preference pairs.

![](images/cd67f12a15e1d831fe6f6773a39c085d75ee088942e25ef280d394fd594aa5b4.jpg)  
Figure 1: (Upper) Variance of reward difference distribution. We randomly select 1,000 prompts in the training set and then sample two responses for each prompt from the output distribution of the policy model and compute the difference between the rewards over time. As training progresses, the distribution of output space shifts, leading the RM to gradually fail to distinguish between responses. (Bottom) Reward Difference Distributions. We sample two responses from a specific distribution for each prompt and obtain the difference between the rewards. For red and blue plots, the responses are sampled from the output distribution of the initial policy model and the latest policy model trained after PPO, respectively. The RM provide close rewards for the different responses in most queries. These indicate that the RM fails to capture subtle differences between responses under conditions of shifting environment distribution.

To evaluate the effectiveness of MetaRM, we conduct extensive experiments on the Anthropic’s HH-RLHF (Bai et al. 2022) and OpenAI’s summarization (Stiennon et al. 2020b) datasets. The experimental results demonstrate that MetaRM can constantly restore the reward model’s distinguishing ability to those responses sampled from the shifted distribution by iteratively training it on original preference data, achieving improvement of the LLM in 3 to 4 rounds. In addition, we also evaluate MetaRM in an OOD setting. The experimental results reveal that it outperforms other reward modeling approaches in LLM alignment by enhancing the ability to discriminate subtle differences in OOD samples. The main contributions of our paper are as follows:

• We introduce MetaRM, a novel method that adapts the reward model to the shifted distribution through meta

learning, to enhance its ability to distinguish responses sampled from the shifted distribution. • Extensive experiments show that MetaRM can iteratively improve the performance of LLM alignment by constantly training the reward model on the original preference pairs. • MetaRM also enhances the ability of the reward model trained only on specific distribution preference data to effectively discriminate OOD samples, without the need for labeling pairs data on the target distribution.

# Related Work

Reinforcement Learning from Human Feedback. Previous studies have demonstrated that RLHF (Bai et al. 2022; Ouyang et al. 2022) is a key component of training stateof-the-art LLMs, such as OpenAI’s GPT-4 (OpenAI 2023) and Meta’s Llama 2 (Touvron et al. 2023). Meanwhile, it also can improve various tasks, such as summarization (Stiennon et al. 2020a; Ziegler et al. 2019), dialogue (Bai et al. 2022), translation (Bahdanau et al. 2016), and make LLMs more helpful, honest, and harmless (3H) (Thoppilan et al. 2022; Ouyang et al. 2022). RLHF involves two main steps: first, using preference data collected from a large number of crowdsource workers to train a reward model. Secondly, using reinforcement learning methods to optimize the language model to maximize the reward. The reward model plays a crucial role in the RLHF process, so modeling a robust reward model is crucial for the RLHF (Rame´ et al. 2024; Lee et al. 2023).

Distribution Shift in Reward Models. Researchers have attempted to obtain a robust reward model by accurately modelling human preferences to boost the ability of the reward model and improve the performance of LLMs (Coste et al. 2023; Shen et al. 2023; Pace et al. 2024). Although these approaches can model reward models somewhat better, they are still suffering from the distribution shift in the RL training phase (Casper et al. 2023; Pikus et al. 2023). Casper et al. (2023) illustrates that distribution shifts can decrease the credibility of the reward model. Additionally, Krueger, Maharaj, and Leike (2020) analyses that samples with overestimated rewards will become gradually more, which may lead to stagnation in the RL training process. Rame´ et al. (2024) ensemble multiple reward models to mitigate the distribution shift and hence the reward overoptimization problem. Touvron et al. (2023) propose to iteratively collect preference pairs and fine-tune the reward model to adjust it to the new distribution. However, continuously collecting new data is resource and time-intensive. In contrast to these approaches, our method focuses on how to alleviate distribution shifts and align with out-of-distribution without labeling the data.

Meta-Learning. Meta-learning generally seeks to improve the models to adapt to new skills, unseen tasks, or new distributions (Finn, Abbeel, and Levine 2017; Li et al. 2019). With the advancement of LLMs, researchers have also introduced meta-learning into language models to enhance performance across various language-related tasks (Hospedales et al. 2021; Bansal et al. 2020; Min et al. 2021). Chen et al.

![](images/d3b1047ee0babb3238dfcd6165b1a15dc36b3c9d9022b9695c003ee438e4d061.jpg)  
Figure 2: The optimization process of MetaRM. MetaRM contains four simple steps: 1. Compute the difference loss on responses sampled from the shifted distribution. 2. Calculate the gradient of this loss wrt. the RM parameters $\theta _ { t }$ and adjust the parameters according to the ascent direction. 3. Compute the vanilla loss on the original preference pairs using the updated parameters ${ \boldsymbol { \theta } } _ { t } ^ { ' }$ . 4. Calculate the gradient of the vanilla loss wrt. ${ \boldsymbol { \theta } } _ { t } ^ { ' }$ and optimize the original parameters $\theta$ following the descent direction.

(2021) introduce meta-learning into in-context learning in language models, focusing on enhancing the adaptability of these models to new tasks with limited data. Jia (2024) efforts to train the reward model in multi-tasks by using meta-learning, to improve the model’s generalization ability. Dou, Yu, and Anastasopoulos (2019) explore meta-learning in low-resource natural language understanding tasks. Unlike these methods, our approach employs meta-learning to address distribution shift issues, enabling the reward model to distinguish out-of-distribution samples without the need for labeled data. Our proposed approach also can be utilized for iterative RLHF optimization.

# Method

In this section, we elaborate on the methodological details of MetaRM, and provide a detailed explanation of the optimization objective of our method.

# MetaRM

Our goal is that when the distribution of the environment shifts as the PPO training progresses, the reward model should still maintain the ability to distinguish new distribution responses, while modeling the human preference from original preference pairs. The key insight of MetaRM is that iteratively training the RM by minimizing the loss on the original preference pairs, particularly those pairs that can maximize the discrimination power over responses sampled from the shifted distribution. The optimization process of our proposed method MetaRM is shown in Figure 2.

The vanilla reward model is trained on a preference pairs dataset which contains comparisons between two responses under the same prompts (Bai et al. 2022; Ouyang et al.

# Algorithm 1: The optimization process of MetaRM.

Require: $\theta , \mathcal { D } , S , n , m$   
Require: $\eta , \alpha$   
1: for $t = 0 , \cdots , T - 1$ do   
2: Sample a mini-batch $X _ { t } = \{ ( x ^ { i } , y _ { w } ^ { i } , y _ { l } ^ { i } ) , 1 \le i \le n \}$ of size $n$ from the preference pairs dataset $\mathcal { D }$   
3: Sample a mini-batch $X _ { s } = \{ \bar { ( } x ^ { i } , s ^ { i } ) , 1 \leq i \leq m \}$ of size $m$ from the meta dataset $s$   
4: Compute the difference loss $\mathcal { I } _ { \theta } ( X _ { s } )$ with the parameters $\theta _ { t }$ on $X _ { s }$   
5: (Meta-process) Compute adapted parameters ${ \boldsymbol { \theta } } _ { t } ^ { ' }$ with gradient ascent: $\theta _ { t } ^ { ' }  \theta _ { t } + \eta \nabla _ { \theta } \mathcal { I } _ { \theta } ( X _ { s } )$   
6: Compute the vanilla loss $\mathcal { L } _ { \boldsymbol { \theta } ^ { \prime } } ( X _ { t } )$ with the parameters ${ \boldsymbol { \theta } } _ { t } ^ { ' }$ on $X _ { t }$   
7: (MetaRM-optimization) Update the parameters $\theta _ { t }$ with gradient descent: $\theta _ { t + 1 }  \theta _ { t } - \alpha \nabla _ { \theta ^ { \prime } } \mathcal { L } _ { \theta ^ { \prime } } ( X _ { t } )$

2022). Formally, for a given prompt $x$ inputted to the supervised fine-tuning (SFT) model $\pi ^ { \mathrm { S F T } } ( y | x )$ , the two responses generated by $\pi ^ { \bar { \mathrm { S F T } } }$ are denoted as $y _ { 1 }$ and $y _ { 2 }$ . The labeller provides a preference for these two responses $y _ { 1 }$ and $y _ { 2 }$ , denoted $y _ { w } \ \succ \ y _ { l }$ , where $y _ { w }$ is the response more consistent with prompt $x$ . Let the training dataset of the RM is $\mathcal { D } \ = \ \{ ( \bar { x ^ { i } } , y _ { w } ^ { i } , y _ { l } ^ { i } ) , 1 \ \leq \ i \ \leq \ N \}$ and $N$ is the number of preference pairs. The loss function of the vanilla reward model can be simplified as follows:

$$
\mathcal { L } _ { \theta } = - E _ { ( x , y _ { w } , y _ { l } ) \sim \mathcal { D } } [ \log \sigma ( r _ { \theta } ( x , y _ { w } ) - r _ { \theta } ( x , y _ { l } ) ) ] ,
$$

where $r _ { \theta }$ denotes the reward model which is often initialized from the SFT model $\pi ^ { \mathrm { S F T } }$ and $\theta$ is the parameters of the reward model $r _ { \theta }$ .

When putting reinforcement learning in the realm of large language models, the environment distribution and the output space distribution of the policy model $\pi ^ { \mathrm { R L } } ( y | x )$ are identical. It means that as $\pi ^ { \mathrm { R L } } ( \dot { y } | x )$ is optimized, the environment distribution shifts. We find that the RM fails to effectively distinguish between responses sampled from the same prompt in the shifted environment, as shown in Figure 1. To measure the reward model’s ability to distinguish the different responses under the same prompts, we define the difference loss function $\mathcal { I } _ { \boldsymbol { \theta } }$ of the reward model $r _ { \theta }$ . Formally, let $s = \{ s _ { i } , 1 \leq i \leq k \}$ be the sequence of responses generated multiple times by the policy model $\pi ^ { \mathrm { R L } } ( \dot { y } | x )$ under the same prompt $x$ , where $k$ denotes the number of responses. The difference function $\mathcal { J } _ { \boldsymbol { \theta } }$ can be written as follows:

$$
\mathcal { I } _ { \theta } = \frac { 2 } { k ^ { 2 } } \sum _ { i = 1 } ^ { k } \sum _ { j = i + 1 } ^ { k } \sigma ( | r _ { \theta } ( x , s _ { i } ) - r _ { \theta } ( x , s _ { j } ) | ) .
$$

It represents the degree of difference in the rewards given by $r _ { \theta }$ for responses s. When the environment distribution shifts, $\mathcal { I } _ { \boldsymbol { \theta } }$ tends to have a lower value. In contrast, a reward model with a higher loss value indicates that it has a remarkable ability to differentiate subtle differences in responses.

Inspired by meta-learning, to restore the reward model’s ability to distinguish responses sampled from a shifted distribution, we introduce an alternating optimization (i.e., meta-optimization and vanilla optimization) to iteratively adapt the RM to the new environment distribution. Our method can be summarised as the RM performs a metaprocess by maximizing the difference loss function $\mathcal { I } _ { \boldsymbol { \theta } }$ before the original gradient update. Let $\begin{array} { r } { S = \{ ( x ^ { i } , s ^ { i } ) , 1 \le i \le } \end{array}$ $M \}$ denotes the meta dataset sampled from a shifted distribution. The meta-process can be represented as updating parameters by a gradient ascent of the difference loss function $\mathcal { J } _ { \boldsymbol { \theta } }$ on a mini-batch $X _ { s }$ of the meta dataset $s$ . Formally, at step $t$ of the training phase, the parameters of the RM $r _ { \theta }$ are adjusted according to the ascent direction:

$$
\boldsymbol { \theta } _ { t } ^ { ' } = \boldsymbol { \theta } _ { t } + \eta \frac { \partial \mathcal { T } _ { \theta } ( \boldsymbol { X } _ { s } ) } { \partial \boldsymbol { \theta } } ,
$$

where $\eta$ controls the degree of learning differences between responses from the meta dataset $s$ . Subsequently, we compute the gradient of the vanilla loss function $\mathcal { L } _ { \boldsymbol { \theta } ^ { \prime } }$ wrt. the parameters $\boldsymbol { \theta } ^ { ' }$ of the RM on a mini-batch $X _ { t } ~ =$ $\{ ( x ^ { i } , y _ { w } ^ { i } , \bar { y } _ { l } ^ { i } ) , 1 \le i \le n \}$ of the original preference pairs dataset $\mathcal { D }$ , which can be represented as follows:

$$
\nabla \theta = \frac { \partial \mathcal { L } _ { \theta ^ { \prime } } ( X _ { t } ) } { \partial \theta ^ { \prime } } .
$$

The $x ^ { i }$ in each batch $X _ { s }$ of the meta dataset $s$ does not need to match the $x ^ { i }$ in $X _ { t }$ .

Note that the MetaRM-optimization using the gradient $\nabla \theta$ is performed over the RM parameters $\theta$ , whereas the objective ${ \mathcal { L } } _ { \theta }$ is computed using the updated RM parameters $\boldsymbol { \theta } ^ { ' }$ . Essentially, MetaRM seeks to learn more from these preference pairs, which can provide more information to differentiate between responses sampled from the shifted environment distribution. Formally, the MetaRM-optimization is performed via gradient descent, and the RM parameters $\theta$ are optimized as follows:

$$
\theta _ { t + 1 } = \theta _ { t } - \alpha \nabla \theta ,
$$

where $\alpha$ is the learning rate for vanilla optimization. The full algorithm is detailed in Algorithm 1.

# Analysis of Optimization Objective

To elucidate the aim of MetaRM, we derive the gradient $\nabla \theta$ (i.e., Equation 4) of optimizing the reward model $r _ { \theta }$ :

$$
\begin{array} { l } { \displaystyle \nabla \theta = \frac { \partial \mathcal { L } _ { \theta ^ { \prime } } ( X _ { t } ) } { \partial \theta ^ { \prime } } } \\ { \displaystyle \quad = \frac { \partial \mathcal { L } _ { \theta ^ { \prime } } ( X _ { t } ) } { \partial \theta } ( \frac { \partial \theta ^ { \prime } } { \partial \theta } ) ^ { - 1 } } \\ { \displaystyle \quad = \frac { \partial \mathcal { L } _ { \theta ^ { \prime } } ( X _ { t } ) } { \partial \theta } ( 1 + \eta \frac { \partial ^ { 2 } \mathcal { J } _ { \theta } ( X _ { s } ) } { \partial \theta ^ { 2 } } ) ^ { - 1 } } \end{array}
$$

where η ∂2Jθ(2Xs) )−1 is deterministic for Xt when the meta-dataset $s$ is sampled, so it can be considered as a constant. We then apply Taylor expansion to $\mathcal { L } _ { \boldsymbol { \theta } ^ { \prime } } ( X _ { t } )$ about point $\theta$ , which can be written as follows:

$$
\begin{array} { r l } & { \mathcal { L } _ { \theta ^ { \prime } } ( X _ { t } ) } \\ & { = \mathcal { L } _ { \theta } ( X _ { t } ) + \displaystyle \frac { \partial \mathcal { L } _ { \theta } ( X _ { t } ) } { \partial \theta } ( \theta ^ { ' } - \theta ) + o ( \theta ^ { ' } - \theta ) ^ { 2 } } \\ & { = \mathcal { L } _ { \theta } ( X _ { t } ) + \eta \displaystyle \frac { \partial \mathcal { L } _ { \theta } ( X _ { t } ) } { \partial \theta } \frac { \partial \mathcal { J } _ { \theta } ( X _ { s } ) } { \partial \theta } + o ( \theta ^ { ' } - \theta ) ^ { 2 } } \\ & { = \mathcal { L } _ { \theta } ( X _ { t } ) + \eta \displaystyle \sum _ { i = 1 } ^ { n } \frac { \partial \mathcal { L } _ { \theta } ( x _ { i } ) } { \partial \theta } \frac { \partial \mathcal { J } _ { \theta } ( X _ { s } ) } { \partial \theta } + o ( \theta ^ { ' } - \theta ) ^ { 2 } } \end{array}
$$

where $o$ is infinitesimals that can be ignored.

Substituting Equation 7 into Equation 4, we obtain the gradient $\nabla \theta$ :

$$
\nabla \theta \propto \frac { \partial } { \partial \theta } [ \mathcal { L } _ { \theta } ( X _ { t } ) + \sum _ { i = 1 } ^ { n } \frac { \partial \mathcal { L } _ { \theta } ( x _ { i } ) } { \partial \theta } \frac { \partial \mathcal { I } _ { \theta } ( X _ { s } ) } { \partial \theta } ] .
$$

Equation 8 suggests that MetaRM-optimization essentially adds a sum of dot products to the vanilla loss function. The dot product computes the similarity between the gradient directions of the meta loss $\mathcal { I } _ { \theta }$ wrt. $\theta$ and vanilla loss wrt. $\theta$ .

Specifically, when the direction of minimizing the vanilla loss on the preference pairs $X _ { t }$ and maximizing the difference between the rewards of the responses $X _ { s }$ are similar, the dot product of both is greater. In such instances, the gradient $\nabla \theta$ in the MetaRM-optimization is larger and the reward model $r _ { \theta }$ can learn more about these preference pairs. Conversely, if the gradients are in different directions, these preference pairs may not be more helpful in alleviating the environment distribution shift, so we downweight the degree of optimization on these data.

# Experiments Experimental Setup

In this work, we use Llama-2 (Touvron et al. 2023) with seven billion parameters as the base model for all experiments. To evaluate the effectiveness of our method in iterative RLHF optimization, we conduct experiments on the general dialogue task and the summarization task. In addition, we also evaluate our approach in an out-of-distribution setting to demonstrate MetaRM’s ability to differentiate subtle differences in OOD samples.

Generation Dialogue Task. Following Vicuna (Chiang et al. 2023), SFT dataset contains 52k multi-turn user-shared conversations from ShareGPT.com (ShareGPT 2023), including a variety of domains such as mathematics, knowledge querying, and coding. For Human preference data, we utilize Anthropic’s HH-RLHF (Bai et al. 2022), a comprehensive collection of human preference concerning AI assistant responses (Bai et al. 2022). It contains 161k training samples and 8,500 testing samples including helpfulness and harmlessness data.

Summarization Task. For SFT dataset, we use the Reddit TL;DR dataset (Vo¨lske et al. 2017) as the training dataset, which contains 123,169 Reddit posts paired with human-authored summaries. Human preference data is similar to the SFT dataset, which includes preference pairs posts. Each post is paired with two generated summaries, one of which is labeled as preferred by annotators (Stiennon et al. 2020a).

Out-of-Distribution Task. SFT dataset is the same as the dataset used in the generation dialogue task. For Human preference data, we use the Oasst1 dataset (Ko¨pf et al. 2024) as the helpfulness data of OOD task. This dataset is a human-annotated assistant-style conversation dataset including over $1 0 \mathrm { k }$ conversations (Ko¨pf et al. 2023). On the other hand, we use PKU-SafeRLHF (Dai et al. 2024) as the harmlessness data, which is a human-labelled dataset containing both performance and safety preferences.

Table 1: Main results on iterative RLHF optimization. We compare the win, tie, and lose ratios of RLHF by MetaRM in the different rounds against the SFT model under both GPT-4 and human evaluations. The results show the superior performance of LLM alignment by using MetaRM. It also highlights the consistency between human and GPT-4 evaluations.   

<html><body><table><tr><td rowspan="2">Dataset</td><td rowspan="2">Opponent vs SFT</td><td colspan="3">GPT-4</td><td colspan="3">Human</td></tr><tr><td>Win↑</td><td>Tie</td><td>Lose↓</td><td>Win↑</td><td>Tie</td><td>Lose↓</td></tr><tr><td rowspan="4">Anthropic-Harmless</td><td>Round 1</td><td>44</td><td>44</td><td>12</td><td>48</td><td>32</td><td>20</td></tr><tr><td>Round 2</td><td>65</td><td>31</td><td>4</td><td>63</td><td>28</td><td>9</td></tr><tr><td>Round 3</td><td>69</td><td>28</td><td>3</td><td>72</td><td>22</td><td>6</td></tr><tr><td>Round 4</td><td>64</td><td>31</td><td>5</td><td>68</td><td>27</td><td>5</td></tr><tr><td rowspan="4">Anthropic-Helpful</td><td>Round 1</td><td>39</td><td>52</td><td>9</td><td>44</td><td>39</td><td>17</td></tr><tr><td>Round 2</td><td>62</td><td>33</td><td>5</td><td>65</td><td>27</td><td>8</td></tr><tr><td>Round 3</td><td>73</td><td>23</td><td>4</td><td>69</td><td>29</td><td>2</td></tr><tr><td>Round 4</td><td>67</td><td>27</td><td>6</td><td>65</td><td>23</td><td>12</td></tr><tr><td rowspan="5">Summary</td><td>Round 1</td><td>51</td><td>11</td><td>38</td><td>54</td><td>16</td><td>30</td></tr><tr><td>Round 2</td><td>55</td><td>15</td><td>30</td><td>57</td><td>12</td><td>31</td></tr><tr><td>Round 3</td><td>67</td><td>14</td><td>19</td><td>63</td><td>15</td><td>22</td></tr><tr><td>Round 4</td><td>78</td><td>5</td><td>17</td><td>77</td><td>7</td><td>16</td></tr><tr><td>Round 5</td><td>72</td><td>8</td><td>20</td><td>69</td><td>12</td><td>19</td></tr></table></body></html>

Baselines. Our Baseline approaches include Supervised Fine-Tuning (SFT), Proximal Policy Optimization (PPO) (Schulman et al. 2017) in RLHF (Ouyang et al. 2022) and Direct Preference Optimization (DPO) (Rafailov et al. 2023). The detailed description is discussed in the supplementary material.

# Implementation Details

In the SFT phase, the learning rate is set to $2 e ^ { - 5 }$ , and we train two epochs with a linear decay to zero. We employ a warmup period of 0.3 epochs. The fine-tuning process was conducted on a single node with eight Nvidia A100-80G GPUs and the global batch size is set to 32. In the reward modelling phase, the learning rate is set to $5 e ^ { - 6 }$ , and the global batch size is set to 16 for both the vanilla training phase and the meta-process phase. The training epoch on original preference pair datasets is only one for our proposed method and all baselines. For each optimization round of MetaRM, the learning rates $\alpha$ and $\eta$ are both set to $5 e ^ { - 6 }$ . The meta dataset is constructed from the previous iteration round and for round 1, the responses are generated by the SFT model. We sample five responses for each prompt in the training dataset to compute Equation 2. In the PPO phase, the learning rate for the policy model and critic model is $5 e ^ { - 7 }$ and $1 . { \overset { - } { 5 } } e ^ { - 6 }$ . For each query, we collect 16 roll-out samples using nucleus sampling. the temperature, top-p and the repetition penalty in the sampling phase are set to 0.8, 0.9 and 1.1, respectively. We set the token-level KL penalty coefficient $\beta$ to 0.05 with a clip value of 0.8.

For the iterative RLHF process, we utilize the current policy model to sample multiple responses from the original prompt dataset to obtain the meta-data. For the OOD setting, the policy model is similarly employed to sample multiple responses from the OOD prompt dataset to obtain the meta-data.

# Metrics & Evaluation

Win rate. To evaluate the effectiveness of our method, we assess it by comparing its win rate with other baselines. Specifically, we randomly select 100 prompts from the test datasets and generate the responses from our method and baselines, respectively. We then provide these pairs of prompts and responses to human evaluators, asking them to determine which response is of higher quality, more useful, and harmless. During the entire evaluation process, the human evaluators are unaware of the responses’ sources. Additionally, some studies indicate that GPT-4’s evaluation of the responses aligns closely with that of human evaluators (Chang et al. 2023; Zheng et al. 2023a,c). So we also utilize GPT-4 to evaluate the performance of MetaRM against other baselines. The GPT-4 prompts for evaluation can be found in the supplementary material.

Diversity. To evaluate the diversity of prompts generated by LLMs, we employ the SelfBLEU (Zhu et al. 2018) score to evaluate diversity in the form of text and sentence embeddings to evaluate diversity in the semantics of text (Zhu et al. 2018; Reimers and Gurevych 2019). The mathematical forms of the two diversity metrics can be found in the supplementary material. Specifically, we calculate the average SelfBLEU scores using n-grams for $n \in \{ 2 , 3 , 4 , 5 \}$ and normalize both metrics, with lower values indicating greater diversity (Zhu et al. 2018). The metrics are computed based on all the test set data, and the diversity of responses is defined as the sum of these two diversity metrics.

# Main Results

Experimental results on iterative RLHF optimization. We iteratively optimize the LLM by recovering the reward model’s distinguishing ability through MetaRM without collecting extra preference pairs. We recorded the improvement achieved by our approach in each optimization round, in comparison to the SFT model, as written in Table 1. In addition, to more comprehensively demonstrate the superiority of our approach, we also compare the best round of MetaRM (i.e., rounds three and four in the generation dialogue task and the summarization task, respectively) against other widely used baselines including the vanilla PPO (Ouyang et al. 2022) and DPO (Rafailov et al. 2023), as shown in Table 2.

<html><body><table><tr><td rowspan="2">Dataset</td><td rowspan="2">Opponent</td><td colspan="3">GPT-4</td><td colspan="3">Human</td></tr><tr><td>Win↑</td><td>Tie</td><td>Lose↓</td><td>Win↑</td><td>Tie</td><td>Lose↓</td></tr><tr><td rowspan="3">Anthropic-Harmless</td><td>SFT</td><td>69 (68.3)</td><td>28 (27.0)</td><td>3 (4.7)</td><td>72 (71.3)</td><td>22 (21.7)</td><td>6 (7.0)</td></tr><tr><td>Vanilla PPO</td><td>54 (53.7)</td><td>31 (30.0)</td><td>15 (16.3)</td><td>58 (57.0)</td><td>24 (23.7)</td><td>18 (19.3)</td></tr><tr><td>DPO</td><td>49 (49.3)</td><td>16 (15.0)</td><td>35 (35.7)</td><td>53 (51.0)</td><td>14 (16.7)</td><td>33 (32.3)</td></tr><tr><td rowspan="3">Anthropic-Helpful</td><td>SFT</td><td>73 (73.3)</td><td>23 (21.7)</td><td>4 (5.0)</td><td>69 (68.3)</td><td>29 (25.0)</td><td>2 (6.7)</td></tr><tr><td>Vanilla PPO</td><td>65 (64.3)</td><td>30 (28.3)</td><td>5 (7.3)</td><td>67 (66.0)</td><td>28 (26.7)</td><td>5 (7.3)</td></tr><tr><td>DPO</td><td>58 (58.3)</td><td>35 (32.3)</td><td>7 (9.3)</td><td>56 (54.7)</td><td>34 (31.7)</td><td>10 (13.7)</td></tr><tr><td rowspan="3">Summary</td><td>SFT</td><td>78 (77.0)</td><td>5 (6.3)</td><td>17 (16.7)</td><td>77 (75.7)</td><td>7 (10.3)</td><td>16 (14.0)</td></tr><tr><td>Vanilla PPO</td><td>62 (61.7)</td><td>7 (9.0)</td><td>31 (29.3)</td><td>54 (55.0)</td><td>19 (16.3)</td><td>27 (28.7)</td></tr><tr><td>DPO</td><td>59 (59.7)</td><td>6 (10.7)</td><td>35 (29.7)</td><td>66 (64.0)</td><td>14 (15.0)</td><td>20 (21.0)</td></tr></table></body></html>

Table 2: The results compare RLHF by MetaRM against the SFT model and other popular alignment baselines. The values in parentheses indicate the average values under different learning rates $\eta$ in the sensitivity analysis experiment.

![](images/87d7a71b1284cca996d1c379eda8aed2b7ce0cbce8d26c1443ed991f5c93d7b7.jpg)  
Figure 3: The results on the out-of-distribution task compared to SFT and vanilla PPO. The results show that our method outperforms other baselines by adapting the reward model to the new distribution.

From the results of the two tables, we can observe that: (1) In each round, our proposed method can significantly improve the quality of responses compared to the SFT model, both for GPT-4 and human evaluation. This improvement was notable in the initial rounds of RLHF optimization, i.e., rounds one and two. (2) The results show a decline in the win rate in the fourth round of the dialogue generation task and the fifth round of the Summarization task. It indicates that the effectiveness of our approach has an upper limit, which varies depending on the task. (3) Our method significantly outperforms all other state-of-the-art baselines including the original RLHF and DPO, by iteratively training the language model without introducing extra preference pairs. (4) Evaluation by human evaluators aligns closely with GPT-4. Therefore, our primary reliance is placed upon the assessments from GPT-4 in subsequent experimental evaluation for saving time and resources.

Table 3: The diversity results compare RLHF by MetaRM against raw good responses and other baselines. Lower values indicate greater diversity.   

<html><body><table><tr><td>Methods</td><td>Anthropic- Anthropic- Harmless</td><td>Helpful</td><td>Summary</td></tr><tr><td>Raw data</td><td>0.14</td><td>0.09</td><td>0.07</td></tr><tr><td>SFT</td><td>0.07</td><td>0.04</td><td>0.06</td></tr><tr><td>Vanilla PPO</td><td>0.47</td><td>0.29</td><td>0.36</td></tr><tr><td>DPO</td><td>0.39</td><td>0.31</td><td>0.38</td></tr><tr><td>RLHFMetaRM</td><td>0.41</td><td>0.25</td><td>0.33</td></tr></table></body></html>

Experimental results on out-of-distribution task. We also apply MetaRM in an OOD setting to demonstrate its ability to adapt the reward model to a new out-ofdistribution. The experimental results are shown in Figure 3. The results reveal that MetaRM can enhance the performance of LLM alignment in the OOD task. MetaRM can increase the RM’s ability to identify subtle differences in responses sampled from OOD prompts to improve its performance in the RL training phase without extra preference data. The outstanding experimental results highlight the effectiveness and potential of our framework for LLM alignment in both ID and OOD scenarios.

Experimental results on diversity of responses. Recently, researchers found that the diversity of responses decreases during the alignment phase. To evaluate the response diversity of the LLM optimized by RLHF with MetaRM, we use two diversity metrics to access our proposed method and other alignment baselines on the test set, as shown in Table 3. Experimental results show that the diversity of responses generated by the SFT model is better than the diversity of good responses in the preference pairs. All LLM

0.67570050   
0.625   
0.567050 Vanilla PPO Ours (round ${ \tt = } 1$ ) Ours (round $^ { = 2 }$ ) Ours (round ${ \bf \alpha } = 3 { \bf \alpha }$ )   
0.550   
0.0 0.2 0.4 0.6 0.8 1.0 Epoch

Iterative RLHF Optimization Out-of-distribution Setting 10% MetaRM 20% 46% Vanilla RM 16% 12% 8% 2% 4% M 1 0 0.2    0.4    0.6    0.8   1.0 0 0.2    0.4    0.6    0.8   1.0 Reward Difference Reward Difference

alignment approaches reduce the diversity of responses, although they improve the helpfulness and the harmlessness of responses. On the other hand, compared to other alignment methods, our method can slightly increase the diversity of responses on helpfulness and summary tasks. The results indicate that our proposed method can improve the quality of responses iteratively, while slightly increasing the diversity of responses.

# Sensitivity Analysis

Compared to the vanilla reward modeling method, MetaRM introduces another hyper-parameter $\eta$ to control the degree of learning differences between samples of the meta dataset, as shown in Equation 3. To further evaluate the effectiveness of MetaRM, we analyze the hyper-parameter impact. Specifically, we set $\eta$ to $1 e ^ { - 6 }$ , $5 e ^ { - 6 }$ , and $\bar { 1 } e ^ { - 5 }$ , respectively, and fix other hyper-parameters to train the reward model. As with the previous experimental setting, we select the LLM of round three and round four in the generation dialogue task and the summarization task, respectively, and compare them with other alignment methods by GPT-4 and Human annotation. The average comparison results are shown in parentheses of Table 2. The results reveal that MetaRM can significantly improve the performance of LLM alignment across different experimental settings. Additionally, the performance of MetaRM does have little fluctuation across different degrees of learning differences in shifted distribution, which demonstrates the stability of our proposed method.

# Discussion

The Accuracy curves in the RM training phase. We record the reward model accuracy curves of the original RM training approach (i.e., as defined by Equation 1) and several training rounds of the MetaRM way during the training phase, as shown in Figure 4. Compared to the original RM training way, we can observe that the MetaRM does not affect the accuracy of the reward model on the valid set of the preference dataset, although we introduce an additional gradient ascent process on the meta dataset. This indicates that our method can enhance the reward model the capability of aligning with the new environment distribution while maintaining the ability to model human preferences through meta-learning. In addition, the trend of each round’s curve shows a high consistency which represents the reasonable and effectiveness of our proposed approach.

Reward Difference Distribution. We randomly select 1,000 prompts and plot the reward difference distribution of vanilla RM after PPO training and RM after MetaRM training, respectively, as shown in Figure 5. The reward difference means the absolute difference in rewards given by the RM for different responses under the same prompt. It means whether the reward model can capture the subtle differences between the samples in the new distribution. The results show that the difference generated by the reward model trained using the original RM way is centered in the range of zero to 0.2. On the contrary, the difference given by the RM trained using MetaRM exhibits lower peaks and greater dispersion. This indicates that our method significantly enhances the RM’s ability to distinguish responses sampled from a shifted environment distribution.

# Conclusion

In this paper, we introduce MetaRM, a method that adapts the reward model to the shifted environment distribution through meta-learning. MetaRM iteratively trains the RM in an alternating way, to maximise discrimination power over responses of the shifted distribution, while preserving its ability to modeling human preference from the original preference pairs. Extensive experiments show that MetaRM can constantly achieve an improvement of alignment within the iterative RLHF optimization, while enhancing RM’s capability of differentiating subtle differences in OOD samples.