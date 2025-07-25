# In-Dataset Trajectory Return Regularization for Offline Preference-based Reinforcement Learning

Songjun $\mathbf { T } \mathbf { u } ^ { 1 , 2 , 3 }$ , Jingbo Sun1,2,3, Qichao Zhang1,3\*, Yaocheng Zhang1,3, Jia Liu4, Ke Chen2, Dongbin Zhao1,2,3

1State Key Laboratory of Multimodal Artificial Intelligence Systems, CASIA, Beijing, China 2Peng Cheng Laboratory, Shenzhen, China   
3School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, China 4School of Mathematics and Statistics, Xi’an Jiaotong University, Xi’an, China tusongjun2023,zhangqichao2014 @ia.ac.cn

# Abstract

Offline preference-based reinforcement learning (PbRL) typically operates in two phases: first, use human preferences to learn a reward model and annotate rewards for a rewardfree offline dataset; second, learn a policy by optimizing the learned reward via offline RL. However, accurately modeling step-wise rewards from trajectory-level preference feedback presents inherent challenges. The reward bias introduced, particularly the overestimation of predicted rewards, leads to optimistic trajectory stitching, which undermines the pessimism mechanism critical to the offline RL phase. To address this challenge, we propose In-Dataset Trajectory Return Regularization (DTR) for offline PbRL, which leverages conditional sequence modeling to mitigate the risk of learning inaccurate trajectory stitching under reward bias. Specifically, DTR employs Decision Transformer and TD-Learning to strike a balance between maintaining fidelity to the behavior policy with high in-dataset trajectory returns and selecting optimal actions based on high reward labels. Additionally, we introduce an ensemble normalization technique that effectively integrates multiple reward models, balancing the tradeoff between reward differentiation and accuracy. Empirical evaluations on various benchmarks demonstrate the superiority of DTR over other state-of-the-art baselines.

Extended version — https://arxiv.org/html/2412.09104

# Introduction

Designing complex artificial rewards in reinforcement learning (RL) is challenging and time-consuming (Skalse et al. 2022; Wang et al. 2024a). Preference-based reinforcement learning (PbRL) addresses this by leveraging human feedback to guide policies, demonstrating success in aligning large language models (Ouyang et al. 2022) and robot control (Liang et al. 2022). Recently, considering the growing utilization of offline data in aiding policy optimization via offline RL (Fang et al. 2022; Chen, Li, and Zhao 2024), offline PbRL (Shin, Dragan, and Brown 2023) has gained attention. This approach involves training a reward model with limited human feedback, annotating rewards for a rewardfree offline dataset, and applying offline RL to learn policies.

Despite significant advancements in offline PbRL, learning an accurate step-wise reward model from trajectory-wise preference feedback remains inherently challenging due to limited feedback data (Zhang et al. 2023), credit assignment (Kim et al. 2023) and neural network approximation errors (Zhu, Jordan, and Jiao 2023). The introduced reward bias adds potential brittleness to the pipeline, leading to suboptimal performance (Yu et al. 2022; Hu et al. 2023). To mitigate this issue, some studies have aimed to enhance the robustness of the reward model (Shin, Dragan, and Brown 2023; Gao et al. 2024), yet ignoring the potential influence of reward bias in offline RL. Alternatively, approaches that bypass reward modeling and directly optimize policy using preference (An et al. 2023; Hejna et al. 2024) struggle to achieve out-of-distribution (OOD) generalization, limiting their ability to outperform the dataset (Xu et al. 2024).

For the policy learning, most of offline PbRL methods (Christiano et al. 2017; Yuan et al. 2024; Gao et al. 2024) apply the learned reward function directly to downstream TDLearning (TDL) based offline RL algorithms, such as CQL (Kumar et al. 2020) and IQL (Kostrikov, Nair, and Levine 2022). However, these TDL-based methods do not account for potential bias in the predicted rewards. The introduced reward bias, especially overestimated rewards, can lead to optimistic trajectory stitching and undermine the pessimism towards OOD state-action pairs in offline TDL algorithms (Yu et al. 2022). To minimize the impact of reward bias, it is necessary to consider being pessimistic about overestimated rewards during the offline policy learning phase (Zhan et al. 2024). Apart from TDL-based offline algorithms, another methodology, conditional sequence modeling (CSM) (Emmons et al. 2022) such as Decision Transformer (DT) (Chen et al. 2021), has not yet been investigated in offline PbRL. This type of imitation-based approach learns a maximum return policy with in-dataset trajectories by assigning appropriate trajectory reweighting. Compared to TDL, which extracts policy based on value functions, CSM extracts policy conditioned on in-dataset trajectory returns, thereby potentially constraints its trajectory stitching capability (Yamagata, Khalil, and Santos-Rodriguez 2023). Although a limited trajectory stitching ability may not be anticipated for offline RL with ground-truth (GT) rewards, this limitation can be considered as a pessimistic safeguard to alleviate the inaccurate trajectory stitching due to incorrect reward labels and maintain fidelity to the behavior policy with high indataset trajectory returns in offline PbRL. These properties trigger our further thought:

Is it possible to leverage the trajectory return-based CSM to mitigate inaccurate stitching of TDL for offline PbRL?

Building upon these insights, we propose In-Dataset Trajectory Return Regularization (DTR) for offline PbRL, which integrates CSM and TDL to achieve the offline RL policy based on in-dataset trajectory returns and annotated step-wise preference rewards. Specifically, DTR consists of the following three core components: (1) a DT (Chen et al. 2021) structure based policy is employed to associate the return-to-go (RTG) token with in-dataset individual trajectories, aintaining fidelity to the behavior policy with high trajectory-wise returns; (2) a TDL module aims to utilize $Q$ function to select optimal actions with high step-wise rewards and balance the limited trajectory stitching ability of DT; and (3) an ensemble normalization that effectively integrates multiple reward models to balance reward differentiation and inaccuracy.

By combining CSM and TDL dynamically, DTR mitigates the potential risk of reward bias in offline PbRL, leading to enhanced performance. Our contributions are summarized below:

• We propose DTR, a valid integration of CSM with DT-based regularization to address the impact of TDlearning stitching caused by reward bias in offline PbRL. • We introduce a dynamic coefficient in the policy loss to balance the conservatism and exploration, and an ensemble normalization method in reward labeling to balance reward differentiation and inaccuracy. • We prove that extracting policies from in-dataset trajectories leads to provable suboptimal bounds, resulting in enhanced performance in offline PbRL. • Our experiments on public datasets demonstrate the superior performance and significant potential of DTR.

# Related Works

Offline PbRL. To enhance the performance of offline PbRL, some works emphasize the importance of improving the robustness of reward model, such as improving the credit assignment of returns (Early et al. 2022; Kim et al. 2023; Verma and Metcalf 2024), utilizing data augmentation to augment the generalization of reward model (Hu et al. 2024b). Other approaches modify the Bradley-Terry model to optimize preferences directly and avoid reward modeling, such as DPPO (An et al. 2023) and CPL (Hejna et al. 2024). A related SOTA work is FTB (Zhang et al. 2023), which optimizes policies based on diffusion model and augmented trajectories without TDL. In contrast, our approach balances trajectory-wise DT and step-wise TD3 (Fujimoto, Hoof, and Meger 2018) to mitigate the risk of reward bias and leads to enhanced performance.

In theory, (Zhu, Jordan, and Jiao 2023) proves that the widely used maximum likelihood estimator (MLE) converges under the Bradley-Terry model in offline PbRL with the restriction to linear reward function. (Hu et al. 2023)

stresses that pessimism about overestimated rewards should be considered in offline RL. (Zhan et al. 2024) relaxes the assumption of linear reward and provides theoretical guarantees to general function approximation. Based on these results, we further provide theoretical suboptimal bound guarantees for offline PbRL under the estimated value function.

Improving the Stitching Ability of CSM. Due to the transformer architecture and the paradigm of supervised learning, CSM has limited trajectory stitching ability. Therefore, some works aggregate the input of the transformerbased policy or modify its structure to adapt to the characteristics of multimodal trajectories (Shang et al. 2022; Zeng et al. 2024; Kim et al. 2024). The alternative approach leverages $Q$ value to enhance trajectory stitching capability such as CGDT (Wang et al. 2024b), QDT (Yamagata, Khalil, and Santos-Rodriguez 2023), Q-Transformer (Chebotar et al. 2023) and recent QT (Hu et al. 2024a) for offline RL with GT rewards.

# Preliminaries

# Learning Rewards From Human Feedback

Following previous studies (Lee, Smith, and Abbeel 2021; Kim et al. 2023), we consider trajectories of length $H$ composed of states and actions, defined as $\begin{array} { r l } { \sigma } & { { } = } \end{array}$ $\left\{ s _ { k } , a _ { k } , \dotsc , s _ { k + H } , a _ { k + H } \right\}$ . The goal is to align human preference $y$ between pairs of trajectory segments $\sigma ^ { 0 }$ and $\sigma ^ { 1 }$ , where $y$ denotes a distribution indicating human preference, captured as $y \in \{ 1 , 0 , 0 . 5 \}$ . The preference label $y = 1$ indicates that $\sigma ^ { 0 }$ is preferred to $\sigma ^ { 1 }$ , namely, $\sigma ^ { 0 } \succ \sigma ^ { 1 }$ , $y = 0$ indicates $\sigma ^ { 1 } \succ \sigma ^ { \bar { 0 } }$ , and $y = 0 . 5$ indicates equal preference for both. The preference datasets are stored as triples, denoted as $\mathcal { D } _ { p r e f } \colon ( \sigma ^ { 0 } , \sigma ^ { 1 } , y )$ .

The Bradley-Terry model (Bradley and Terry 1952) is frequently employed to couple preferences with rewards. The preference predictor is defined as follows:

$$
P _ { \psi } [ \sigma ^ { 1 } \succ \sigma ^ { 0 } ] = \frac { \exp \left( \sum _ { t } \hat { r } _ { \psi } ( s _ { t } ^ { 1 } , a _ { t } ^ { 1 } ) \right) } { \sum _ { i \in \{ 0 , 1 \} } \exp \left( \sum _ { t } \hat { r } _ { \psi } ( s _ { t } ^ { i } , a _ { t } ^ { i } ) \right) }
$$

where $\hat { r } _ { \psi }$ is the reward model to be trained, and $\psi$ is its parameters. Subsequently, the reward function is optimized using the cross-entropy loss, incorporating the human groundtruth label $y$ and the preference predictor $P _ { \psi }$ :

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { C E } } = - \mathbb { E } _ { ( \sigma ^ { 0 } , \sigma ^ { 1 } , y ) \sim \mathcal { D } _ { p r e f } } \Big \{ ( 1 - y ) \log P _ { \psi } [ \sigma ^ { 0 } \succ \sigma ^ { 1 } ] } \\ & { \qquad \quad + y \log P _ { \psi } [ \sigma ^ { 1 } \succ \sigma ^ { 0 } ] \Big \} } \end{array}
$$

In the offline PbRL, we assume there exists a small dataset $\mathcal { D } _ { p r e f }$ with preference labels along with a much larger unlabeled dataset $\mathcal { D }$ without rewards or preference labels. We label the offline dataset $\mathcal { D }$ with estimated step-wise rewards $\hat { r }$ . Then we can obtain the trajectory dataset by calculating trajectory-wise RTG $\hat { R }$ for individual trajectories in $\mathcal { D }$ . The re-labeled dataset is used for downstream offline RL. The notation “in-dataset trajectory” indicates that the trajectory is in the offline dataset $\mathcal { D }$ .

𝒓𝒓𝟏𝟏𝟏𝟏 = 𝟐𝟐 𝒓𝒓𝟑𝟑𝟑𝟑 = 𝟐𝟐 In-Dataset Trajectory Start from $s _ { 1 }$ :   
𝒔𝒔𝟏𝟏 14=1 𝒔𝒔𝟑𝟑 -=2 𝒔𝒔𝟓𝟓 𝑠𝑠𝑠1, 𝑠𝑠𝑠3, 𝑠𝑠𝑠5 , 𝑅𝑅𝑅 = 24.0 ①𝐦𝐦𝐦𝐦𝐦𝐦 ෡𝑸𝑸 𝒔𝒔𝟏𝟏, 𝒂𝒂 : ൝ො𝒓ො𝒓𝒓 𝒓𝟏𝟏𝟏𝟏 + 𝑽෡𝑽෡𝑽 𝑽 𝒔𝒔 𝒔𝟑𝟑 = 𝟒𝟒 𝟒..  𝟎𝟓𝟎 𝟓 1 r45 (𝒔𝒔𝟏𝟏, 𝒔𝒔𝟒𝟒, 𝒔𝒔𝟓𝟓)   
𝒔𝒔𝟐𝟐 𝒔𝒔𝟒𝟒 𝒔𝒔𝟔𝟔 𝑠𝑠𝑠2, 𝑠𝑠𝑠4, 𝑠𝑠𝑠5 , 𝑅𝑅𝑅 = 43.5 TDL Agent ②𝐦𝐦𝐦𝐦𝐦𝐦 ෡𝑸𝑸(𝒔𝒔𝟒𝟒, 𝒂𝒂): ൝ො𝒓ො𝒓𝒓𝒓𝟒𝟒𝟒𝟒 + 𝑽෡𝑽𝑽 𝒔𝒔𝒔𝟓𝟓 = 𝟑𝟏𝟑𝟏. 𝟎𝟓𝟎𝟓 𝒓𝒓𝟐𝟐𝟐𝟐 = 𝟐𝟐. 𝟓𝟓 𝒓𝒓𝟒𝟒𝟒𝟒 = 𝟏𝟏 X ො𝒓𝒓𝟏𝟏𝟏𝟏 = 𝟐𝟐 ො𝒓𝒓𝟑𝟑𝟑𝟑 = 𝟐𝟐 In-Dataset Trajectory   
𝒔𝒔𝟏𝟏 A 𝒔𝒔𝟑𝟑 𝒔𝒔𝟓𝟓 𝑠𝑠𝑠1, 𝑠𝑠𝑠3, 𝑠𝑠𝑠5 , 𝑅෠𝑅𝑅 = 34.0 ①𝐦𝐦𝐦𝐦𝐦𝐦 ෡𝑹𝑹 𝒔𝒔𝟏𝟏, 𝒂𝒂 : ൝𝑹෡ 𝑹𝑹(𝒔𝒔𝒔𝟏𝟏, 𝒔𝒔𝒔𝟑𝟑, 𝒔𝒔𝒔𝟓𝟓) = 𝟒𝟑𝟒𝟑. 𝟎𝟎𝟎 14=15 =3 (𝒔𝒔𝟏𝟏, 𝒔𝒔𝟑𝟑, 𝒔𝒔𝟓𝟓) 45 Out-Dataset Trajectory ②𝐦𝐦𝐦𝐦𝐦𝐦 ෡𝑹𝑹 𝒔𝒔𝟑𝟑, 𝒂𝒂 : 𝑹෡𝑹 𝒔𝒔𝟑𝟑, 𝒔𝒔𝟓𝟓 = 𝟐𝟐. 𝟎𝟎   
𝒔𝒔𝟐𝟐 𝒔𝒔𝟒𝟒 𝒔𝒔𝟔𝟔 CSM Agent ො𝒓𝒓𝟐𝟐𝟐 𝟐 T = 𝟏𝟏. 𝟓𝟓 𝑠𝑠1, 𝑠𝑠4, 𝑠𝑠5 , 𝑅෠𝑅 = 4.5 𝒓𝟒𝟒𝟒

# Return-Based Conditional Sequence Modeling

RL is formulated as a Markov Decision Process (MDP) (Sutton 2018). A MDP is characterized by the tuple $M =$ $\langle S , A , P , r , \gamma \rangle$ , where $S$ is the state space, $A$ is the action space, $P : S \times A \times S \to \mathbb { R }$ is the transition probability distribution, $r : S  \mathbb { R }$ is the reward function, and $\gamma \in ( 0 , 1 )$ is the discount factor. The objective of RL is to determine an optimal policy $\pi$ that maximizes the expected cumulative reward: $\begin{array} { r } { \pi \stackrel { - } { = } \arg \operatorname* { m a x } _ { \pi } \mathbb { E } _ { s _ { 0 } , a _ { 0 } , \dots } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \bar { r } \left( s _ { t } \right) \right] } \end{array}$ .

Return-based CSM addresses sequential decision problems through an autoregressive generative model, thus avoiding value estimation (Brandfonbrener et al. 2022). A prominent example is DT (Chen et al. 2021), which considers a sequence of trajectories of length $H$ : $\begin{array} { r l } { \tau _ { t } } & { { } = } \end{array}$ $( R _ { t - H + 1 } , s _ { t - H + 1 } , a _ { t - H + 1 } , \cdot \cdot \cdot , R _ { t } , s _ { t } , a _ { t } )$ , where $R _ { t }$ denote−s return-t−o-go (RT−G): $\begin{array} { r } { R _ { t } = \sum _ { t ^ { \prime } = t } ^ { T } r _ { t ^ { \prime } } } \end{array}$ . During the training phase, the supervised loss between the target policy and behavior policy is computed using mean squared error (MSE) loss. During the inference phase, DT rolls out the estimated action $\hat { a } _ { t }$ with the provided RTG $\hat { R } _ { t }$ . Although CSM harnesses the power of large models for supervised learning, its limited trajectory stitching capability constrains the potential for improvement from suboptimal data.

# Methods

In this section, we first introduce the problem of reward bias in offline PbRL using a toy example and analyze the TDL and CSM methods in such scenarios to further emphasize our motivation. Then, we provide a detailed pipeline and the overall framework for the DTR method. Finally, we prove that extracting policies with in-dataset trajectory return regularization leads to provable suboptimal bounds.

# Rethinking Stitching in PbRL: A Toy Example

We use the deterministic MDP depicted in Figure 1 as an example. In this MDP, the agent starts from $s _ { 1 }$ or $s _ { 2 }$ , transitions through $s _ { 3 }$ or $s _ { 4 }$ , and finally reaches $s _ { 5 }$ or $s _ { 6 }$ . The GT reward for each state transition is denoted as $r$ . We estimate stepwise rewards $\hat { r }$ from trajectory-level pairwise preferences. Typically, the preference dataset $\mathcal { D } _ { p r e f }$ is gathered through a limited number of queries, making it difficult to cover all possible trajectories in the offline dataset $\mathcal { D }$ . Suppose $\mathcal { D } _ { p r e f }$ includes pairwise comparison labels for four trajectories and their intermediate segments:

$$
\mathcal { D } _ { p r e f } = \left( \begin{array} { l } { \left( s _ { 1 } , s _ { 3 } , s _ { 5 } \right) _ { r e d } , \quad \left( s _ { 1 } , s _ { 4 } , s _ { 6 } \right) _ { y e l l o w } , y = 1 } \\ { \left( s _ { 2 } , s _ { 4 } , s _ { 5 } \right) _ { g r e e n } , \left( s _ { 1 } , s _ { 4 } , s _ { 6 } \right) _ { y e l l o w } , y = 1 } \\ { \left( s _ { 2 } , s _ { 4 } , s _ { 5 } \right) _ { g r e e n } , \left( s _ { 1 } , s _ { 3 } , s _ { 5 } \right) _ { r e d } , \quad y = 1 } \\ { \left( s _ { 1 } , s _ { 3 } , s _ { 5 } \right) _ { r e d } , \quad \left( s _ { 2 } , s _ { 4 } , s _ { 6 } \right) _ { p u r p l e } , y = 1 } \end{array} \right)
$$

Here we use $( \cdot ) _ { c o l o r }$ to denote the trajectory and its color depicted in Figure 1. Suppose the trajectories in $\mathcal { D }$ are identical to those in $\mathcal { D } _ { p r e f }$ . There are three possible trajectories beginning from $s _ { 1 } \dot { : } \left( \dot { s } _ { 1 } , s _ { 3 } , s _ { 5 } \right)$ , $( s _ { 1 } , s _ { 4 } , s _ { 6 } )$ and $( s _ { 1 } , s _ { 4 } , s _ { 5 } )$ . The first two are intrinsic in $\mathcal { D }$ (in-dataset), while the last one is formed by stitching together two trajectory fragments (out-dataset). The trajectory with the highest GT return is $( s _ { 1 } , s _ { 3 } , s _ { 5 } )$ . However, due to the reward bias, we assume $\hat { r } _ { 4 5 } ~ = ~ 3$ is higher than the ground-truth $r _ { 4 5 } ~ = ~ 2$ . Consequently, the estimated return of the stitched trajectory $( s _ { 1 } , s _ { 4 } , s _ { 5 } )$ is higher than that of $( s _ { 1 } , s _ { 3 } , s _ { 5 } )$ . Such failures of estimated rewards are common when using a small-scale $\mathcal { D } _ { p r e f }$ to label a large-scale $\mathcal { D }$ .

We now scrutinize the performance of CSM and TDL methods under the reward bias. Starting from $s _ { 1 }$ , TDL chooses the stitched trajectory $( s _ { 1 } , s _ { 4 } , s _ { 5 } )$ that maximizes the expected cumulative reward. In contrast, CSM identifies the in-dataset trajectory that maximizes the RTG for $s _ { 1 }$ , considering only trajectories within $\mathcal { D }$ that contain $s _ { 1 }$ , specifically $( s _ { 1 } , s _ { 3 } , s _ { 5 } )$ and $( s _ { 1 } , s _ { 4 } , s _ { 6 } )$ . As a result, CSM opts for $s _ { 3 }$ over $s _ { 4 }$ , leading to $s _ { 5 }$ . To put it simply, CSM prefers conservative choices to learn behaviors under limited preference queries, seeking optimal actions that align closely with the behavior policy conditioned on RTG. Conversely, TDL may

# Ⅰ. Relabel Offline Dataset By Ensemble Normalization

Preference Dataset 𝓓𝓓𝒑𝒑𝒑𝒑𝒑𝒑𝒑 Trajectory Dataset $\pmb { \mathcal { D } }$ 𝓓 with Estimated Reward 2 Norm ො𝒓𝒓𝝍𝝍𝟏 Relabel   
& ： Mean 购购 Norm {… , 𝑹෡ 𝑹𝒊𝒊 , 𝒔𝒔𝒊𝒊, 𝒂𝒂𝒊𝒊, 𝒓ො 𝒓𝒊𝒊, … , 𝑹෡ 𝑹𝒕𝒕, 𝒔𝒔𝒕𝒕, 𝒂𝒂𝒕𝒕, 𝒓ො 𝒓𝒕𝒕} ො𝒓𝒓𝝍𝝍𝑵   
Ⅱ. Training  &       Ⅲ. Inference Sample   
Dynamic Train Q and Policy By Rollout   
$\begin{array} { r } { \mathcal { L } _ { \pi } = \mathcal L _ { D T } - \lambda ( \mathbf { s t e p } ) \sum _ { i = t - H + 1 } ^ { t } Q _ { \phi } ( s _ { i } , \widehat { \boldsymbol { a } } _ { i } ) } \end{array}$ 𝐬𝐬𝐬𝐬𝐬𝐬𝐬𝐬: +1 1 𝑹෡𝑹 𝒔𝒔𝒊𝒊 𝒂𝒂𝒊𝒊 𝑹෡𝑹𝒕𝒕 𝒔𝒔𝒕𝒕 𝒂𝒂𝒕𝒕 Policy Trajectory   
Autoregressive Inference 2 Update N Rollout ① Decision Transformer   
{ 𝑹෡𝑹𝟎𝟎 𝒔𝒔𝟎𝟎 ෝ𝒂𝒂𝟎𝟎 𝑹෡𝑹𝒕𝒕 𝒔𝒔𝒕𝒕 } × 𝑲𝑲 ③ ② D 𝒔ො𝒔 ෝ𝒂𝒂𝒊𝒊 𝒔ො𝒔𝒕𝒕 ෝ𝒂𝒂𝒕𝒕 Q Update Decision Transformer 十 𝑹෡𝑹𝒕𝒕+𝟏𝟏 } × 𝑲𝑲 由 𝒕𝒕−𝟏 ෝ𝒂𝒂𝟎𝟎 ෝ𝒂𝒂𝒕𝒕 } × 𝑲𝑲 𝐦𝐦𝐦𝐦𝐦𝐦ෝ𝒂𝒂𝒕𝒕∈[𝟏𝟏,𝑲𝑲]𝑸𝑸𝝓𝝓(𝒔𝒔𝒕𝒕, 𝒂ෝ𝒂𝒕𝒕) 𝑸෡ 𝑸(𝒔𝒔𝒊𝒊, 𝒂𝒂𝒊𝒊) 𝜸𝒋𝒋−𝒊𝒊𝒓ො𝒓𝒋𝒋 + 𝜸𝜸𝒕𝒕−𝒊𝒊min𝒊𝒊=𝟏𝟏,𝟐𝟐 𝑸𝑸𝝓𝝓′ (𝒔𝒔𝒕𝒕, ෝ𝒂𝒂𝒕𝒕)

![](images/38cf38946f213a1da83074116df03ea6560cab5a8374544c1820076760728c98.jpg)  
Figure 2: The overall framework DTR. Phase I: Utilize preference data $\mathcal { D } _ { p r e f }$ to learn the reward model, then label the offline dataset $\mathcal { D }$ by ensemble normalization. Phase II: Train the $Q$ function and policy network, comprising three components: trajectory rollout, $Q$ update, and policy update. Phase III: Autoregressively infer actions based on the target RTG and select the action with the highest $Q$ value.

fail due to optimistic trajectory stitching under reward bias. Therefore, we hope to balance conservatism and exploration by integrating CSM and TDL dynamically for offline PbRL.

# In-dataset Regularization: Training and Inference

In the offline policy training phase, we aim to utilize the DT and TD3 to achieve a balance between maintaining fidelity to the behavior policy with high in-dataset trajectory returns and selecting optimal actions with high reward labels. Specifically, we sample a mini-batch trajectories from $\mathcal { D }$ , and use DT as the policy network to rollout the estimated actions $\{ \hat { a } _ { i } \} _ { i = t - H + 1 } ^ { t }$ . Due to the powerful capability of autoregressive generative model, we also rollout estimated states $\{ \hat { s } _ { i } \} _ { i = t - H + 1 } ^ { t }$ to strengthen policy representation (Liu et al. 2024). Considering the trajectory $\begin{array} { r l } { \tau _ { t } } & { { } = } \end{array}$ $\{ \cdots , \hat { R } _ { i } , s _ { i } , a _ { i } , \cdots , \hat { R } _ { t } , s _ { t } , a _ { t } \}$ , the self-supervised loss of DT can be expressed as:

$$
\mathcal { L } _ { D T } = \mathbb { E } _ { \tau _ { t } \sim \mathcal { D } } \sum _ { i = t - H + 1 } ^ { t } \underbrace { \Vert \hat { s } _ { i } - s _ { i } \Vert ^ { 2 } } _ { \mathrm { A u x i l i a r y \ : L o s s } } + \underbrace { \Vert \hat { a } _ { i } - a _ { i } \Vert ^ { 2 } } _ { \mathrm { G o a l - B C \ : L o s s } }
$$

Additionally, we train the $Q$ function to select optimal actions with high reward labels. A natural idea is to use the estimated trajectory sequence for n-step TD bootstrapping. Following (Fujimoto, Hoof, and Meger 2018) and (Hu et al.

2024a), the $Q$ target and $Q$ loss are defined as follows:

$$
\hat { Q } ( s _ { i } , a _ { i } ) = \sum _ { j = i } ^ { t - 1 } \gamma ^ { j - i } \hat { r } _ { j } + \gamma ^ { t - i } \operatorname* { m i n } _ { i = 1 , 2 } Q _ { \phi _ { i } } ( s _ { t } , \hat { a } _ { t } )
$$

$$
\mathcal { L } _ { Q } = \mathbb { E } _ { \tau _ { t } \sim \mathcal { D } } \sum _ { i = t - H + 1 } ^ { t } \Big \Vert Q _ { \phi } ( s _ { i } , a _ { i } ) - \hat { Q } ( s _ { i } , a _ { i } ) \Big \Vert ^ { 2 }
$$

To prevent failures from incorrect trajectory stitching, as highlighted in the toy example, we dynamically integrate the policy gradient to train the policy network. Meanwhile, in Theorem 1, we will show that extracting policies from indataset trajectories leads to provable suboptimal bounds. In the early training stage, the policy loss ${ \mathcal { L } } _ { \pi }$ primarily consists of supervised loss $\mathcal { L } _ { D T }$ to ensure that the policy learns the trajectories of in-dataset optimal return and the corresponding $Q$ function. Subsequently, we gradually increase the proportion of policy gradient to facilitate trajectory stitching near the optimal range within the distribution. In summary, we minimize the following policy loss:

$$
\mathcal { L } _ { \pi } = \mathcal { L } _ { D T } - \lambda ( \operatorname { s t e p } ) \mathbb { E } _ { \tau _ { t } \sim \mathcal { D } } \sum _ { i = t - H + 1 } ^ { t } Q _ { \phi } ( s _ { i } , \hat { a } _ { i } )
$$

where $\lambda ( \mathrm { s t e p } )$ considers the normalized $Q$ value and increases linearly with the training steps:

$$
\lambda ( \mathsf { s t e p } ) = \frac { \eta } { \mathbb { E } _ { ( s , a ) \sim \tau _ { t } } | Q _ { \phi } ( s , a ) | } , \eta = \frac { \mathsf { s t e p } \times \eta _ { m a x } } { \mathsf { m a x \_ s t e p } }
$$

Different from TD3BC (Fujimoto and $\boldsymbol { \mathrm { G u } } 2 0 2 1 ,$ ) with the constant $\eta$ , our approach uses progressively increasing coefficients to maintain training stability. This follows a straightforward principle: “Learn to walk before you can run.”

After the training, we get a trained policy $\pi _ { \boldsymbol { \theta } }$ and $Q$ network $Q _ { \phi }$ . During the inference stage, we autoregressively rollout the action sequence almost following the setting of DT (Chen et al. 2021). The difference is that we cannot update the RTG with the rewards provided by the online environment. An alternative way is to replace the environment reward with a learned reward model:

$$
\hat { R } _ { t + 1 } = \hat { R } _ { t } - \hat { r } _ { t }
$$

However, the reward model has limited generalization for OOD state-action pairs, and the additional model leads to greater consumption of computing resources. Therefore, we propose a simple calculation: subtract a fixed value from the next timestep’s RTG $\hat { R } _ { t + 1 }$ until it is reduced to 0 at the end of the episode:

$$
\hat { R } _ { t + 1 } = \hat { R } _ { t } - \hat { R } _ { 0 } / \mathrm { m a x . t i m e s t e p }
$$

The reason for this simplification is that we normalize the rewards so that candidate states with high values tend to have similar rewards at each step. In implementation, we calculate $K$ initial RTGs as targets: $\{ \bar { \hat { R } } _ { 0 } ^ { i } \} _ { i = 1 } ^ { k } ~ =$ $\{ 0 . 5 , 0 . 7 5 , 1 , 1 . 5 , 2 \} \times \mathrm { { R e t u r n } _ { \operatorname* { m a x } } }$ , here we have $K = 5$ , and $\mathrm { R e t u r n } _ { \operatorname* { m a x } }$ is the maximum return in $\mathcal { D }$ . Through the parallel reasoning of Transformer, $K$ actions are derived, and the one with the highest $Q$ value is executed.

# Relabel Offline Data by Ensemble Normalization

During the annotation reward phase, we introduce a simple but effective normalization method to highlight reward differentiation, and further improve the performance of downstream policies. Specifically, we train $N$ -ensemble MLP reward models $\{ \hat { r } _ { \psi _ { i } } \} _ { i = 1 } ^ { N }$ from the offline preference dataset $\mathcal { D } _ { p r e f }$ with the loss function defined in Equation 2, then annotate the offline data $\mathcal { D }$ with predicted rewards.

We observe that the trained reward model labels different state-action pairs with only minor differences (as detailed in the Experiments Section and Figure 4a), and these indistinct reward signals may lead to exploration difficulties and low sample efficiency (Jin et al. 2020). While direct reward normalization can amplify these differences, it also increases uncertainty, resulting in inaccurate value estimation and high variance performance (Gao et al. 2024).

To balance these two aspects, we propose ensemble normalization, which first normalizes the estimates of each ensemble and then averages them:

$$
\boldsymbol { \hat { r } _ { \psi } } = \mathbf { M e a n } ( \mathbf { N o r m } ( \boldsymbol { \hat { r } _ { \psi _ { 1 } } } ) , \cdot \cdot \cdot , \mathbf { N o r m } ( \boldsymbol { \hat { r } _ { \psi _ { N } } } ) )
$$

Notably, ensemble normalization is a plug-and-play module that can enhance reward estimation without any modifications to reward training.

# Theoretical Analysis

Suppose the preference dataset $\mathcal { D } _ { p r e f }$ with $N _ { p }$ trajectories of length $H _ { p }$ and the reward-free offline dataset $\mathcal { D }$ with $N _ { o }$

trajectories of length $H _ { o }$ . Then, consider the following general offline PbRL algorithm:

# 1. Construct the Reward Confidence Set:

First, estimate the parameters $\boldsymbol { \widehat { \psi } } \in \mathbb { R } ^ { d }$ of reward model via MLE with Equation 2. Then cbonstruct a confidence set $\Psi ( \zeta )$ by selecting reward models that nearly maximize the log-likelihood of $\mathcal { D } _ { p r e f }$ to a slackness parameter $\zeta$ .

# 2. Bi-Level Optimization of Reward and Policy:

Identify the policy $\widehat { \pi }$ that maximizes the estimated policy value $\widehat { V }$ under the leasbt favorable reward model $\widetilde { \psi }$ with both $\mathcal { D } _ { p r e f }$ band $\mathcal { D }$ :

$$
\widetilde { \psi } = \arg \operatorname* { m i n } _ { \psi \in \Psi ( \zeta ) } \widehat { V } _ { \psi } - \mathbb { E } _ { \tau \sim \mu _ { r e f } } [ r _ { \psi } ( \tau ) ] , \widehat { \pi } = \arg \operatorname* { m a x } _ { \pi } \widehat { V } _ { \widetilde { \psi } }
$$

And we have the following theorem:

Theorem 1 (Informal) Suppose that: $( l )$ the dataset $\mathcal { D } _ { p r e f }$ and $\mathcal { D }$ have positive coverage coefficients $C _ { p } ^ { \dagger }$ and $C _ { o } ^ { \dagger }$ ; (2) the underlying MDP is a linear $M D P$ ; (3) the GT reward $\psi ^ { \star } \in \mathcal G _ { \psi }$ ; (4) $0 \leq r _ { \psi } ( \tau ) \leq r _ { \mathrm { m a x } }$ , and $\| \psi \| _ { 2 } ^ { 2 } \leq d$ for all $\psi \in$ $\mathcal { G } _ { \psi }$ and $\tau \in \mathcal { T }$ . and (5) $\widehat { \pi }$ is any mesurable function of the data $\mathcal { D } _ { p r e f }$ . Then with prbobability $1 - 2 \delta$ , the performance bound of the policy $\widehat { \pi }$ satisfies for all $s \in \mathcal S$ ,

$$
\begin{array} { r l } & { S u b O p t ( \widehat { \pi } ; s ) \leq \sqrt { \frac { c C _ { \psi } ^ { 2 } ( \mathcal { G } _ { \psi } , \pi ^ { \star } , \mu _ { r e f } ) \kappa ^ { 2 } \log ( \mathcal { N } _ { \mathcal { G } _ { \psi } } / N _ { p } \delta ) } { N _ { p } } } } \\ & { \phantom { x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x } + \frac { 2 c r _ { \operatorname* { m a x } } } { ( 1 - \gamma ) ^ { 2 } } \sqrt { \frac { d ^ { 3 } \xi _ { \delta } } { N _ { p } H _ { p } C _ { p } ^ { \dagger } + N _ { o } H _ { o } C _ { o } ^ { \dagger } } } \phantom { x x x x x x x x x x x x x x x x x } } \end{array}
$$

where $\xi _ { \delta } = \log \left( 4 d ( N _ { p } H _ { p } + N _ { o } H _ { o } ) / ( 1 - \gamma ) \delta \right)$ , and other variables involved are not related to with $N _ { p }$ or $N _ { o }$ . For detailed definitions, see Appendix A.

Remark 1 The theorem extends offline RL theory specifically to offline PbRL, leading to a theoretical upper bound for offline PbRL. In order to guarantee this bound, the learned policy $\widehat { \pi }$ is any measurable function of $\mathcal { D } _ { p r e f }$ should be satisfied. Inbother words, if $\widehat { \pi } \in \mathcal { D } _ { p r e f }$ , the assumption naturally holds. A relaxed condibtion is ${ \widehat { \pi } } \in { \mathcal { D } }$ , since trajectories in $\mathcal { D } _ { p r e f }$ are often sampled from $\mathcal { D }$ . This theoretical analysis guides us to make more use of in-dataset trajectories for policy optimization to ensure marginal performance improvements. Accordingly, our DTR method utilizes CSM for optimizing in-dataset trajectories to establish reliable performance bound, while employing TDL to enhance the utilization of out-dataset trajectories. We elaborate on this finding with experiments in Appendix E7.

# Experiments

In this section, we evaluate DTR and other baselines on various benchmarks. We aim to address the following questions: (1) Can DTR mitigate the risk of reward bias and lead to enhanced performance in offline PbRL? (2) What specific roles do the proposed modules play in the proposed DTR? (3) Can DTR still perform well with small amounts of preference feedback?

<html><body><table><tr><td>Dataset</td><td>Pb-IQL</td><td>Pb-TD3BC</td><td>DPPO</td><td>OPRL</td><td>FTB</td><td>Pb-DT*</td><td>Pb-QT*</td><td>DTR (Ours)*</td><td>DTR (Best)*</td></tr><tr><td>Walker2D-m</td><td>78.4</td><td>26.3</td><td>28.4</td><td>80.8</td><td>79.7</td><td>71.4 ± 5.1</td><td>80.1 ± 1.4</td><td>86.6 ± 2.8</td><td>88.3 ± 2.7</td></tr><tr><td>Walker2D-m-r</td><td>67.3</td><td>47.2</td><td>50.9</td><td>63.2</td><td>79.9</td><td>51.1 ± 10.5</td><td>79.3 ± 1.5</td><td>80.8 ± 2.8</td><td>84.4 ± 2.1</td></tr><tr><td>Walker2D-m-e</td><td>109.4</td><td>74.5</td><td>108.6</td><td>109.6</td><td>109.1</td><td>108.0 ± 0.2</td><td>109.5 ± 0.6</td><td>109.7 ± 0.3</td><td>111.1 ± 0.3</td></tr><tr><td>Hopper-m</td><td>50.8</td><td>48.0</td><td>44.0</td><td>59.8</td><td>61.9</td><td>49.8 ± 4.8</td><td>81.3 ± 8.8</td><td>90.7 ± 0.6</td><td>94.5 ± 0.4</td></tr><tr><td>Hopper-m-r</td><td>87.1</td><td>25.8</td><td>73.2</td><td>72.8</td><td>90.8</td><td>67.0 ± 8.2</td><td>84.5 ± 12.8</td><td>92.5 ± 0.9</td><td>96.0 ± 1.6</td></tr><tr><td>Hopper-m-e</td><td>94.3</td><td>97.4</td><td>107.2</td><td>81.4</td><td>110.0</td><td>111.3 ± 0.1</td><td>109.4 ± 1.8</td><td>109.5 ± 2.4</td><td>112.3 ± 0.3</td></tr><tr><td>Halfcheetah-m</td><td>43.3</td><td>34.8</td><td>38.5</td><td>47.5</td><td>35.1</td><td>42.5 ± 0.8</td><td>42.7 ± 0.3</td><td>43.6± 0.3</td><td>44.2 ± 0.3</td></tr><tr><td>Halfcheetah-m-r</td><td>38.0</td><td>38.9</td><td>40.8</td><td>42.3</td><td>39.0</td><td>37.6 ± 0.6</td><td>39.9 ± 0.7</td><td>40.6 ± 0.2</td><td>41.6 ± 0.2</td></tr><tr><td>Halfcheetah-m-e</td><td>91.0</td><td>73.8</td><td>92.6</td><td>87.7</td><td>91.3</td><td>68.7 ± 9.1</td><td>78.2 ± 11.6</td><td>91.9 ± 0.4</td><td>93.5 ± 0.3</td></tr><tr><td>Average</td><td>73.29</td><td>51.86</td><td>64.91</td><td>71.68</td><td>77.42</td><td>67.49</td><td>78.32</td><td>82.88</td><td>85.10</td></tr></table></body></html>

Table 1: The performance comparison of DTR and different baselines on Gym-MuJoCo locomotion. In the first column, -m, -mr, and -m-e are abbreviations for the medium, medium-replay, and medium-expert datasets, respectively. The results of $\mathrm { P b }$ -IQL and Pb-TD3BC are from Uni-RLHF benchmark (Yuan et al. 2024), the results of OPRL and FTB are from the experimental section of FTB (Zhang et al. 2023), the -m-r and -m-e results of DPPO are from (An et al. 2023), and the -m results are reproduced by ourselves. For the remaining methods with \*, we record the average normalized score of the 10 rollouts of the last checkpoint and run the experiment under 5 random seeds, and finally record the mean score and $\pm$ denotes the standard deviation. The bold font indicates the algorithm with the best performance, and the underline indicates the second one.

Table 2: The performance comparison on Adroit manipulation platform. In the first column, $- \mathbf { h }$ and -c are abbreviations for the human and clone datasets.   

<html><body><table><tr><td>Dataset</td><td>Pb-IQL</td><td>Pb-DT</td><td>Pb-QT</td><td>DTR (Ours)</td></tr><tr><td>Pen-h</td><td>99.8 ± 8.3</td><td>115.3 ± 2.0</td><td>111.2 ± 3.9</td><td>114.0 ± 9.6</td></tr><tr><td>Pen-c</td><td>93.7 ± 14.1</td><td>104.6 ± 13.4</td><td>69.1 ± 12.0</td><td>86.6± 6.4</td></tr><tr><td>Door-h</td><td>9.7 ± 1.4</td><td>14.2 ± 1.4</td><td>26.0 ± 5.0</td><td>33.3 ± 8.6</td></tr><tr><td>Door-c</td><td>2.2 ± 0.6</td><td>7.7 ± 0.9</td><td>10.7 ± 1.9</td><td>12.6 ± 0.6</td></tr><tr><td>Hammer-h</td><td>11.8 ± 2.0</td><td>2.0±0.3</td><td>15.0 ± 5.7</td><td>21.5 ± 6.9</td></tr><tr><td>Hammer-c</td><td>11.4 ± 2.7</td><td>4.0 ± 2.1</td><td>14.6 ± 1.5</td><td>26.7 ± 10.3</td></tr><tr><td>Average</td><td>38.10</td><td>41.29</td><td>41.10</td><td>49.11</td></tr></table></body></html>

Setup. We select three tasks from the Gym-MuJoCo locomotion suite (Brockman et al. 2016), and three tasks from the Adroit manipulation platform (Kumar 2016). We utilize the Uni-RLHF dataset (Yuan et al. 2024) as preference dataset $\mathcal { D } _ { p r e f }$ , which provides pairwise preference labels from crowdsourced human annotators. The offline dataset $\mathcal { D }$ is sourced from the D4RL benchmark (Fu et al. 2020).

Baselines. (1) Pb (Kim et al. 2023) including PT-IQL and $\mathrm { P b }$ -TD3BC, which perform IQL or TD3BC with preferencebased trained reward functions, respectively; (2) DPPO (An et al. 2023), which designs a novel policy scoring metric and optimize policies without reward modeling; (3) OPRL (Shin, Dragan, and Brown 2023), which performs IQL with ensemble-diversified reward functions; (4) FTB (Zhang et al. 2023), which generates better trajectories with higher preferences based on diffusion model. For a more intuitive comparison, we propose several variants: (5) $\mathrm { P b }$ -DT and $\mathrm { P b }$ -QT, which slightly modify CSM models DT (Chen et al. 2021) and QT (Hu et al. 2024a) for offline PbRL. The more implementation details are shown in Appendix D.

It is worth emphasizing that we provide a structural comparison of reward model between MLP and Transformer for offline PbRL. We find that that MLP-based reward model

Walker2d-medium Halfcheetah-medium-expert 80 MMWWMW 80 WW WwW 60 60 100 40 80 40 n=0 YMM -withoutrewardnorm 20 60 =inear(step) 20 with direct bewardnomorm 40 51015 0 0 10 20 30 40 50 0 10 20 30 40 50 Training steps (k) Training steps (k) (a) Ablation study on $\eta$ . (b) Ablation study on norm r.

has a more stable performance than the Transformer-based one. Please see Appendix E2 for the detailed analysis. Hence, our methods employ the MLP-based reward model.

# Q1: Can DTR Mitigate the Risk of Reward Bias and Lead to Enhanced Performance?

Gym-MuJoCo Locomotion Tasks. In the MuJoCo environment, we comprehensively compare DTR with the above baselines. The results are listed in Table 1. DTR outperforms all prior methods in most datasets. On average, DTR surpasses the best baseline FTB by a large margin with a minimum of $7 . 1 \%$ and even $1 1 . 0 \%$ in our best performance. Additionally, DTR exhibits significantly lower performance variance compared to $\mathrm { P b }$ -DT and $\mathrm { P b }$ -QT, which suffer from pronounced fluctuations. The superior performance reflects the importance of integrating CSM and TDL to mitigate the potential risk of reward bias in offline PbRL.

Adroit Manipulation Platform. We further evaluate DTR on the more challenging Adroit manipulation platform. We mainly compare DTR with $\mathrm { P b }$ -IQL since it performs best on this benchmark in previous work (Yuan et al. 2024). with an average score higher than IQL by $2 8 . 9 \%$ .

85 +0.95% +2.50%+5.82% 75 M 80 70 0 75- -12.59% 60 55 -2- 55- GT Direct_norm 50 -3 MLP Ensemble_norm 50 GT Rewards Estimated Rewards 45- I Pb-IQL  Pb-DT  Pb-QT DTR 0 20 40 60 80 100 Base Base $^ +$ Dynamicn+Norm_r 10% 30% 50% 70% 100% Timestep Base + Dynamic n Percentage of Preference Data (a) Compare of Reward Normalization. (b) Ablation with Ground-Truth Reward. (c) Scaling Trend for Preference Dataset.

Figure 4: Visual comparison of ablation experiments. (a) Changes in the reward values of trajectories labeled with different reward normalization methods in Halfcheetah-m-e; (b) The impact of the proposed module on the performance under the GT rewards and estimated rewards datasets. We record the average score across all 9 tasks of MuJoCo and ”Base” represents DTR without the dynamic coefficient and ensemble normalization; (c) The performance of different methods changes with the size of the preference dataset. We record the average score across 3 tasks of MuJoCo-Medium.

# Q2: What Specific Roles Do the Proposed Modules Play in DTR?

The Role of Dynamic Coefficient. To further elucidate the role of the proposed dynamic coefficients $\lambda ( \mathrm { s t e p } )$ in Equation 6 and 7. We study the performance difference compared to $\eta = 0$ and $\eta = \mathrm { c o n s t }$ , and present the training score curves in Figure 3a. The results of first $1 5 \mathrm { k \Omega }$ training steps highlight the superior training efficiency of dynamic coefficient, while the method with constant $\eta$ fluctuates and performs lower than the dynamic coefficient one.

Comparison of Reward Normalization. To explore the effectiveness of the proposed ensemble normalization of rewards, we present the training score curves for different normalization methods in Figure 3b. The curves demonstrate that the ensemble normalization method achieves superior convergence and reduced training fluctuations. Furthermore, we select the first 100 timesteps of the trajectory with the lowest GT return in Halfcheetah-m-e dataset and plot the changes in different estimated rewards in Figure 4a. Compared to GT rewards, the original rewards (MLP) have little differences across states, which is detrimental to downstream policy learning. Directly normalizing average rewards amplifies these differences but also exaggerates estimation errors, especially overestimation. Our method normalizes the estimates of each ensemble member individually before averaging them, thereby enhancing reward differences while mitigating estimation errors.

Overall Ablation Performance. Table 3 records the average scores of the above two ablation experiments, and more ablation results are given in Appendix E5.

Ablation with GT Reward. To explore the ability of DTR in the offline RL domain, we apply the same dynamic coefficient and normalization modules as DTR on the offline dataset with GT rewards and show the results in Figure 4b. With GT rewards, the method incorporating the dynamic coefficient $\eta$ performs closely to the oracle method. However, the performance significantly drops with reward normalization, likely due to inaccurate value estimation caused by the normalization of precise rewards. DTR, which employs preference learning to estimate rewards, achieves performance comparable to the oracle.

Table 3: The Gym-MoJoCo average score of DTR without dynamic coefficient $\mathbf { \eta } _ { \eta } = \mathrm { c o n s t } )$ and reward normalization.   

<html><body><table><tr><td>Gym-MuJoCo</td><td>DTR</td><td>-dynamic n</td><td>-reward_norm</td></tr><tr><td>Average</td><td>82.88 ± 1.20</td><td>79.63 ± 2.27 (-3.9%)</td><td>80.30 ±0.77 (-3.1%)</td></tr></table></body></html>

# Q3: Can DTR Still Perform Well With Small Amounts of Preference Feedback?

Human preference queries are often limited, so effective algorithms should align with human intentions using fewer queries. Consequently, we conduct experiments to evaluate the scalability of DTR and baselines with varying preference dataset scales in the MuJoCo-medium environment. We respectively train the reward model with portions of the queries from a dataset of 2000 preference pairs and then train downstream offline RL. The evaluation results are reported in Figure 4c. It shows that DTR can also achieve good performance when reducing the amount of preference data. indicating that finer reward estimation aids in enhancing the performance of the TDL module.

# Conclusion

In this work, we propose DTR, which dynamically combines DT and TDL and utilizes the in-dataset trajectory returns and ensemble reward normalization to alleviate the risk of reward overestimation in offline PbRL. We show the potential of CSM in offline PbRL, which may motivate further research on generative methods in this field and extend them to online settings. In the future, combining the bi-level optimization of the reward function and the policy will be an interesting topic.