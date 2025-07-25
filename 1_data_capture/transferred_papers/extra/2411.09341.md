# Approximated Variational Bayesian Inverse Reinforcement Learning for Large Language Model Alignment

Yuang Cai, Yuyu Yuan, Jinsheng Shi, Qinhong Lin

Beijing University of Posts and Telecommunications cyang,yuanyuyu,jinsheng,linqinhong @bupt.edu.cn

# Abstract

The alignment of large language models (LLMs) is crucial for generating helpful and harmless content. Existing approaches leverage preference-based human feedback data to learn the reward function and align the LLM with the feedback data. However, these approaches focus on modeling the reward difference between the chosen and rejected demonstrations, rather than directly modeling the true reward from each demonstration. Moreover, these approaches assume that the reward is only obtained at the end of the sentence, which overlooks the modeling of intermediate rewards. These issues lead to insufficient use of training signals in the feedback data, limiting the representation and generalization ability of the reward and potentially resulting in reward hacking. In this paper, we formulate LLM alignment as a Bayesian Inverse Reinforcement Learning (BIRL) problem and propose a novel training objective, Approximated Variational Alignment (AVA), to perform LLM alignment through Approximated Variational Reward Imitation Learning (AVRIL). The BIRL formulation facilitates intermediate reward modeling and direct reward modeling on each single demonstration, which enhances the utilization of training signals in the feedback data. Experiments show that AVA outperforms existing LLM alignment approaches in reward modeling, RL finetuning, and direct optimization.

# Introduction

Large language models (LLMs) trained on massive corpus encode a large amount of knowledge and demonstrate powerful linguistic and reasoning capabilities in various domains (OpenAI 2022; Achiam et al. 2023). However, due to the inevitable harmful and useless information in the training data, LLMs can potentially generate content inconsistent with human values or requirements (Holtzman et al. 2019; Zhang et al. 2019; Weidinger et al. 2021). LLM alignment is a prevalent and effective approach for LLMs to generate harmless and helpful content. The alignment task typically relies on human feedback data in the form of preferences, where each preference data consists of a chosen sentence and a rejected sentence, labeled by human annotators (Zopf 2018; Tay et al. 2020). Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) are two common approaches to align

LLMs with human feedback data (Shen et al. 2023). RLHF first performs reward modeling to learn a reward function from the feedback data and then fine-tunes the LLM policy to maximize the expected reward achieved by its generated content using Reinforcement Learning (RL) (Ouyang et al. 2022; Bai et al. 2022; Touvron et al. 2023; Yang et al. 2023). DPO formulates the reward modeling objective as a ranking objective based on the LLM policy, which facilitates the joint performance of reward modeling and LLM policy finetuning through a unified training objective (Yuan et al. 2023; Rafailov et al. 2024; Song et al. 2024).

The Inverse Reinforcement Learning (IRL) problem generally involves learning a reward model from observed demonstration data produced by a Markov Decision Process (MDP) $( \mathrm { N g }$ , Russell et al. 2000). Conversely, the Natural Language Generation (NLG) process can be viewed as an MDP where the generated sentences are considered demonstration data (Ranzato et al. 2016). Therefore, the alignment task performed by RLHF and DPO can be seen as addressing an IRL problem that infers the implicit reward function hidden in the preference-based human feedback data and learns the LLM policy either separately or jointly. However, existing RLHF and DPO alignment approaches only model the reward difference between the chosen and rejected demonstrations without explicitly modeling the true reward of every single sentence. This limitation means that the demonstration data is not fully utilized, which restricts the representation ability of the reward model and can lead to reward hacking (Skalse et al. 2022; Gao, Schulman, and Hilton 2023; Coste et al. 2023; Zhang et al. 2024). Additionally, current approaches generally model the end-toend sentence-level reward without considering the reward of intermediate states. Model generalization may be limited when confronted with data that have similar intermediate state distributions but different complete sentence distributions. It is more intuitive to model intermediate rewards since humans can not only provide overall feedback on the entire text but also explain which parts of the text influenced their feedback.

In this paper, we propose a novel LLM alignment training objective, Approximated Variational Alignment (AVA), based on Bayesian Inverse Reinforcement Learning (BIRL) (Ramachandran and Amir 2007). Specifically, we formulate the reward distribution as a posterior distribution conditioned on the demonstration data and perform Approximated Variational Reward Imitation Learning (AVRIL) (Chan and van der Schaar 2021) to jointly approximate the reward distribution (i.e., the reward model) and the demonstration likelihood (i.e., the policy). Unlike most previous LLM alignment approaches, which only model the reward difference between chosen and rejected demonstrations, AVA directly models the reward of every single demonstration through the AVRIL training objective, thereby making better use of the training signals from feedback data. Additionally, we do not adhere to the assumption that the reward is only obtained at the end of the sentence. Instead, we leverage the AVRIL training objective to model the intermediate reward conditioned on the intermediate demonstration data. To demonstrate flexibility, we use the AVA training objective on data in different formats through different pipelines.

Our work makes the following main contributions:

• We present a novel insight into LLM alignment by formulating the alignment task as a BIRL problem, which enhances the utilization of training signals and improves the representation and generalization ability of the LLM. • We demonstrate the flexibility of AVA by employing it for both reward modeling and direct optimization on either preference data or demonstration data. • We empirically show that AVA surpasses Bradley-Terry and Preference Transformer in reward modeling and downstream RL fine-tuning, and outperforms DPO and AfD in direct optimization, which indicates a reduction in the reward hacking issue and an improvement in representation and generalization ability.

# Related Work

LLM Alignment The Bradley-Terry model (Bradley and Terry 1952) formulates preference likelihood using the reward model and is widely adopted by RLHF alignment approaches for reward modeling. After reward modeling, the LLM is fine-tuned to maximize the expected reward achieved by LLM-generated content through downstream RL training (Ouyang et al. 2022; Bai et al. 2022; Touvron et al. 2023; Yang et al. 2023). A more concise approach for preference alignment is Direct Preference Optimization (DPO) (Rafailov et al. 2024), which denotes preference as the relative log-likelihood difference between the chosen sentence and the rejected sentence. DPO unifies reward modeling and LLM fine-tuning into a single process, facilitating LLM alignment with a simple classification loss. In addition to aligning LLMs with pairwise human preference data, some recent works also align LLMs with non-pairwise demonstration data. Sun and van der Schaar (2024) propose Alignment from Demonstrations (AfD), which leverages high-quality demonstration data to overcome challenges such as noisy labels and privacy concerns in preference datasets.

Intermediate Reward Modeling The above alignment approaches only model the end-to-end reward of a complete sentence, without considering the reward of intermediate states. This lack of intermediate reward modeling stems from the assumption that the reward is only achieved when the sentence is fully generated, regarding the Natural Language Generation (NLG) process as an MDP (Ranzato et al. 2016). To address this issue, we refer to related work on preference modeling in classic RL problems without the aforementioned assumption. Notably, the Preference Transformer (Kim et al. 2023) uses the attention weights computed by the Transformer architecture (Vaswani et al. 2017) to estimate the weighted non-Markovian reward of each intermediate state of the trajectory. The reward of a complete trajectory is then the weighted sum of all intermediate rewards. The preference between the chosen and rejected trajectories is formulated by their rewards and optimized through a contrastive training objective, similar to Bradley-Terry.

# Preliminaries

# MDP Formulation of NLG

At time step $t$ , the state is the previously generated tokens denoted as $\mathbf { y } _ { 1 : t } ~ = ~ ( y _ { 1 } , y _ { 2 } , \cdot \cdot \cdot , y _ { t } )$ , the action is the currently generated token $y _ { t + 1 }$ . Note that in auto-regressive decoding, the output tokens are time-shifted. The action space is the vocabulary $\nu$ containing all possible tokens. In the text generation setting, the state transition is deterministic, so we do not consider the transition probability function. The reward of taking action $y _ { t + 1 }$ under state $\mathbf { y } _ { 1 : t }$ is denoted as $R ( { \bf y } _ { 1 : t } , y _ { t + 1 } ) = \bar { R } ( { \bf y } _ { 1 : t + 1 } )$ , i.e., the reward can be the function of either the current state and the current action or merely the function of the next state due to the deterministic state transition. It is worth noting that for simplicity of denotation, we do not separately denote the prompt text and the response text but denote them as a whole sentence $\mathbf { y }$ . The separation of prompt and response is trivial during implementation. The policy can be denoted as $\pi _ { w } \big ( y _ { t + 1 } | \mathbf { y } _ { 1 : t } \big )$ , which is also the distribution of the language model parameterized by $w$ . For simplicity, we sometimes denote the policy as $\begin{array} { r } { \pi _ { w } ( \mathbf { y } ) = \prod _ { t = 1 } ^ { | \mathbf { y } | - 1 } \pi _ { w } ( y _ { t + 1 } | \mathbf { y } _ { 1 : t } ) } \end{array}$ , where $| \mathbf { y } |$ is the length of sequence $| \mathbf { y } |$ . Note that the accumulated product starts from $\pi _ { w } \big ( y _ { 2 } | \mathbf { y } _ { 1 : 1 } \big )$ instead of $\pi _ { w } ( y _ { 1 } )$ since we assume that all sequences start with a special token denoting the start of the sequence.

# Bayesian Inverse Reinforcement Learning

Inverse Reinforcement Learning (IRL) is the problem of extracting a reward function of a Markov Decision Process (MDP) given observed optimal behavior $\mathrm { N g }$ , Russell et al. 2000). Bayesian Inverse Reinforcement Learning (BIRL) regards the reward function $R$ as the hidden variable affecting and motivating the behavioral data $\tau$ . The objective of BIRL is to learn the posterior distribution $p ( R | \mathcal { T } )$ . Approximate Variational Reward Imitation Learning (AVRIL) (Chan and van der Schaar 2021) adopts variational inference to approximate the posterior distribution. Specifically, AVRIL employs a parameterized distribution $q _ { \phi }$ and minimizes the Kullback-Leibler (KL) divergence between $q _ { \phi }$ and the posterior distribution $p ( R | \mathcal { T } )$ , as shown in Eq. 1. This KL divergence is hard to compute since the posterior distribution is intractable. A common solution is to maximize the Evidence Lower Bound (ELBO), as shown in Eq. 2, where the second term is to minimize the KL divergence between $q _ { \phi }$ and the tractable prior distribution.

$$
\operatorname* { m i n } _ { \phi } D _ { \mathrm { K L } } [ q _ { \phi } ( R ) | | p ( R | T ) ]
$$

$$
\operatorname* { m a x } _ { \phi } \mathbb { E } _ { R \sim q _ { \phi } ( \cdot ) } [ \log p ( \mathcal { T } | R ) ] - D _ { \mathrm { K L } } [ q _ { \phi } ( R ) | | p ( R ) ]
$$

The first term of Eq. 2 is to maximize the log-likelihood of the observed optimal behaviors given any reward sampled from $q _ { \phi }$ . AVRIL denotes the action distribution as a Boltzmann policy, as shown in Eq. 3, where $Q _ { R } ^ { \pi _ { \mathcal { T } } }$ is the state-action value function following policy $\pi \tau$ under reward function $R$ . Intuitively, we can approximate the stateaction value using a Deep $\mathrm { \Delta Q }$ Network (DQN) (Mnih et al. 2013) $Q _ { \theta }$ parameterized by $\theta$ . An important problem is that in the RL setting, the reward function is fixed when optimizing $Q _ { \theta }$ . However, in the AVRIL setting, the reward function is also being optimized during the optimization of $Q _ { \theta }$ . The reward function and the state-action value function should satisfy $R ( s , a ) = \mathbb { E } _ { s ^ { \prime } \sim P ( \cdot \vert s , a ) , a ^ { \prime } \sim \pi ( \cdot \vert s ^ { \prime } ) } [ Q _ { R } ^ { \pi } ( s , a ) -$ $\gamma Q _ { R } ^ { \pi } ( s ^ { \prime } , a ^ { \prime } ) ]$ , $\forall s \in S , a \in A$ , i.e., the reward should equal the expectation of the TD error. By adding a penalty term forcing the TD error to follow the reward distribution, the final objective to be maximized is shown in Eq. 4, where $q _ { \phi } ( R | s , \bar { a } )$ denotes the distribution of reward values given the state $s$ and the action $a$ . In this way, the behavior is indirectly conditioned on the reward, which is consistent with the likelihood $p ( { \mathcal { T } } | R )$ in the ELBO (Eq. 2). Here, $B ( a | s ; Q _ { \theta } )$ is the Boltzmann policy upon the state-action value function $Q _ { \theta }$ parameterized by $\theta$ . The third term in the square brackets is to restrict the TD error to satisfy the constraint $R ( s , a ) = \mathbb { E } _ { s ^ { \prime } , a ^ { \prime } } [ Q _ { R } ^ { \pi } ( s , a ) - \gamma Q _ { R } ^ { \pi } ( s ^ { \prime } , a ^ { \prime } ) ]$ , $\forall s \in$ $s , a \in { \mathcal { A } }$ . $q _ { \phi } ( R | s , a )$ denotes the distribution of reward values given the state $s$ and the action $a$ .

$$
B ( a | s ; Q _ { R } ^ { \pi _ { T } } ) = \frac { \exp ( \beta Q _ { R } ^ { \pi _ { T } } ( s , a ) ) } { \sum _ { a ^ { \prime } \in \cal { A } } \exp ( \beta Q _ { R } ^ { \pi _ { T } } ( s , a ^ { \prime } ) ) }
$$

$$
\operatorname* { m a x } _ { \phi , \theta } \sum _ { ( s , a , s ^ { \prime } , a ^ { \prime } ) \in \mathcal { T } } \left[ \log B ( a | s ; Q _ { \theta } ) - D _ { \mathrm { K L } } \left[ q _ { \phi } ( \cdot | s , a ) | | p ( \cdot ) \right] \right]
$$

# Approximated Variational Alignment

In this section, we formulate the LLM alignment tasks as the BIRL problems and perform alignment with the Approximated Variational Alignment (AVA) training objectives. The AVA training objectives involve AVA from Demonstration (AVA-d) and AVA from Preference (AVA-p), both of which are BIRL training objectives based on the Approximated Variational Reward Imitation Learning (AVRIL) training objective (Chan and van der Schaar 2021). AVA-d is the implementation of the AVRIL training objective under the NLG setting, which learns on non-pairwise demonstration datasets. AVA-p is a contrastive variant of AVA-d, which learns on pairwise preference datasets.

# Alignment from Demonstration

We first consider the problem of aligning an LLM policy with the demonstration data $\mathcal { D }$ , where each sentence $\mathbf { y } \in { \mathcal { D } }$ is the ground-truth sentence. The alignment objective is to encourage the LLM policy to generate sentences like the demonstration data. Instead of building a direct training objective (e.g., supervised fine-tuning) to optimize the LLM policy, we focus on performing BIRL to learn a reward function from the demonstration data $\mathcal { D }$ , i.e., to learn the posterior $p ( R | \mathcal { D } )$ with a parameterized distribution $q _ { \phi } ( R )$ .

As illustrated in the preliminaries, the optimization of $q _ { \phi }$ can be achieved by maximizing the AVRIL training objective (Eq. 4), where each element $( s , a , s ^ { \prime } , a ^ { \prime } ) \ \in \ \mathcal { T }$ is a state-action quadruplet consisting of the current state $s$ , the current action $a$ , the next state $s ^ { \prime }$ and the next action $a ^ { \prime }$ . As for the Natural Language Generation (NLG) setting, at each time step $t$ , the current state is the current sub-sentence $\mathbf { y } _ { 1 : t }$ , the current action is the token-to-begenerated $y _ { t + 1 }$ , the next state is $\mathbf { y } _ { 1 : t + 1 }$ , the concatenation of $\mathbf { y } _ { 1 : t }$ and $y _ { t + 1 }$ , and the next action is $y _ { t + 2 }$ . By substituting the state-action quadruplet in Eq. 4 with the new quadruplet $\left( { \bf y } _ { 1 : t } , y _ { t + 1 } , { \bf y } _ { 1 : t + 1 } , y _ { t + 2 } \right)$ and rewrite the summation in timestep-wise form, we can obtain the AVRIL training objective applicable to the NLG setting, as shown in Eq. 5. We refer to this training objective as Approximated Variational Alignment from Demonstration (AVA-d), which is a variant of the AVRIL training objective in the NLG setting.

$$
\mathcal { F } _ { d } ( \mathcal { D } ) = \sum _ { \mathbf { y } \in \mathcal { D } } \sum _ { t = 1 } ^ { | \mathbf { y } | - 2 } \left[ \log B ( y _ { t + 1 } | \mathbf { y } _ { 1 : t } ; Q _ { \theta } ) - d _ { t } ( \phi ) \right]
$$

$$
d _ { t } ( \phi ) = D _ { \mathrm { K L } } \left[ q _ { \phi } ( \cdot | \mathbf { y } _ { 1 : t + 1 } ) | | p ( \cdot ) \right]
$$

$$
\delta _ { t } ( \theta ) = Q _ { \theta } ( \mathbf { y } _ { 1 : t } , y _ { t + 1 } ) - \gamma Q _ { \theta } ( \mathbf { y } _ { 1 : t + 1 } , y _ { t + 2 } )
$$

Here, $q _ { \phi } ( R | \mathbf { y } _ { 1 : t + 1 } )$ is the reward distribution of the subsequence $\mathbf { y } _ { 1 : t + 1 }$ . The Boltzmann policy $B ( y _ { t + 1 } | \mathbf { y } _ { 1 : t } ; Q _ { \theta } )$ built upon the $\mathrm { Q }$ -value model $Q _ { \theta }$ acts as the LLM policy for text generation. By maximizing $\mathcal { F } _ { d }$ , the $\mathrm { Q }$ -value model (i.e., the LLM policy) $Q _ { \theta }$ as well as the reward distribution $q _ { \phi }$ will be jointly optimized to be aligned with the demonstration dataset $\mathcal { D }$ .

Similar to the original AVRIL objective, the AVA-d objective consists of three sub-objectives: the log-likelihood maximization, the KL divergence minimization, and the TDerror constraint. The first objective trains the LLM policy to maximize the likelihood of the demonstration data, which is identical to supervised fine-tuning. The second objective is to ensure the reward distribution satisfies the prior distribution assumption. The third objective, TD-error constraint, distinguishes AVA-d from conventional supervised fine-tuning. With the constraint, the update of the $\mathrm { \Delta Q }$ -value model will not only increase the Q-value of the ground-truth token in demonstration data but also make the TD error of the Q-values 7 close to the reward obtained after generating the current token, which ensures the consistency between the reward and the policy.

# TQR Architecture

The original AVRIL adopts the architecture with a reward encoder and a Q-value decoder. To compute the AVA-d training objective and leverage the pre-trained weights of the backbone transformer model, we add a reward head and a Qvalue head at the top of the Transformer decoder, as shown in Fig. 1. We refer to this architecture as Transformer with Q-value and Reward Heads (TQR). The Q-value head takes the hidden states of the last decoder layer as input and outputs the Q-value of each action (i.e., token), as shown in Eq. 8. The reward is assumed to follow Gaussian distribution, and the reward head takes in the same hidden states and outputs the mean and standard deviation of the reward of each state, as shown In Eq. 9. Here, ${ \bf h } _ { t }$ is the hidden state vector of time step $t$ , $Q _ { \theta } ( \mathbf { y } _ { 1 : t } , \cdot ) \in \mathbb { R } ^ { | \nu | }$ is a vector whose $i$ -th element equals $Q _ { \theta } \big ( \mathbf { y } _ { 1 : t } , \boldsymbol { v } ^ { ( i ) } \big )$ , where $\boldsymbol { v } ^ { ( i ) }$ is the $i$ -th token in the vocabulary, and $\mu _ { t } , \sigma _ { t } \in \mathbb { R }$ are mean and standard deviation of reward $R \big ( \mathbf { y } _ { 1 : t + 1 } \big )$ at time step $t$ . Now we can compute the training objective in Eq. 5 based on the above outputs of the Q-value head and the reward head.

![](images/9722a11ba8c6962c41393d5f40ba8658d447005bd529f1c8127e40d30809a7c4.jpg)  
Figure 1: Overview of the TQR architecture.

$$
\begin{array} { r l } & { Q _ { \theta } ( \mathbf { y } _ { 1 : t } , \cdot ) = \mathrm { Q H e a d } ( \mathbf { h } _ { t } ; \theta ) , \forall t \in \{ 1 , \cdots , | \mathbf { y } | \} } \\ & { \quad \quad [ \mu _ { t } ; \sigma _ { t } ] = \mathrm { R H e a d } ( \mathbf { h } _ { t } ; \phi ) , \forall t \in \{ 1 , \cdots , | \mathbf { y } | \} } \\ & { \quad R ( \mathbf { y } _ { 1 : t + 1 } ) \sim q _ { \phi } ( R | \mathbf { y } _ { 1 : t + 1 } ) = \mathcal { N } ( R ; \mu _ { t } , \sigma _ { t } ) } \end{array}
$$

Inspired by preference transformer (Kim et al. 2023), we further compute a reward weight for each time step of reward based on attention weights, as shown in Eq. 11, where $\mathbf { q } _ { i }$ is the $i$ -th row of the query matrix of the attention mechanism, $\mathbf { k } _ { t ^ { \prime } }$ is the $t ^ { \prime }$ -th row of the key matrix. We then apply reward weights to the outputs of the Q-value head (Eq. 8) and reward head (Eq. 9). Specifically, we simply multiply the output of the $t$ -th position of the heads by the reward weight $\boldsymbol { w } _ { t }$ , as shown by the red arrows in Fig. 1.

$$
w _ { t } = \frac { 1 } { | \mathbf { y } | } \sum _ { i = 1 } ^ { | \mathbf { y } | } \mathrm { s o f t m a x } \left( \{ \mathbf { q } _ { i } \cdot \mathbf { k } _ { t ^ { \prime } } \} _ { t ^ { \prime } = 1 } ^ { | \mathbf { y } | } \right) _ { t }
$$

Besides using a randomly initialized Q-value head, we can also construct a pre-trained $\mathrm { \Delta Q }$ -value model from the pre-trained LLM policy. The Boltzmann policy formulates the action probability as the softmax function of Q-values. Inversely, we can also formulate the Q-value as the logsoftmax function of action probabilities, as shown in Eq. 12, where $\alpha$ is the temperature hyperparameter, $\pi _ { w }$ is the LLM policy parameterized by $w$ . Note that the log-softmax operation is a non-strict inversion of the softmax operation, which means we can tune $\alpha$ to find the best way to map token-level probabilities to token-level $\mathrm { Q }$ -values.

$$
Q _ { w } ( \mathbf { y } _ { 1 : t } , y _ { t + 1 } ) = \log \frac { \exp ( \alpha \pi _ { w } ( y _ { t + 1 } | \mathbf { y } _ { 1 : t } ) ) } { \sum _ { y ^ { \prime } \in \mathcal { V } } \exp ( \alpha \pi _ { w } ( y ^ { \prime } | \mathbf { y } _ { 1 : t } ) ) }
$$

By substituting with the above Q-value model, the AVA-d training objective can be denoted as Eq. 13, which facilitates us to initialize the Q-value model from a pre-trained LLM policy and adopt the AVA-d objective to fine-tune the LLM policy. The TD error can be denoted as Eq. 14.

$$
\mathcal { F } _ { d } ( \mathcal { D } ) = \sum _ { \mathbf { y } \in \mathcal { D } } \sum _ { t = 1 } ^ { | \mathbf { y } | - 2 } \bigg [ \int _ { - d _ { t } ( \phi ) + \lambda \log q _ { \phi } ( \delta _ { t } ( w ) | \mathbf { y } _ { 1 : t + 1 } ) } ^ { \beta \log \operatorname { s o f t m a x } ( \alpha \pi _ { w } ( y _ { t + 1 } | \mathbf { y } _ { 1 : t } ) ) \big ] } \bigg ]
$$

$$
\delta _ { t } ( w ) = \log \frac { \operatorname * { s o f t m a x } \bigl ( \alpha \pi _ { w } \bigl ( y _ { t + 1 } | \mathbf { y } _ { 1 : t } \bigr ) \bigr ) } { \operatorname * { s o f t m a x } \bigl ( \alpha \pi _ { w } \bigl ( y _ { t + 2 } | \mathbf { y } _ { 1 : t + 1 } \bigr ) \bigr ) ^ { \gamma } }
$$

# Alignment from Preference

We then consider the problem of aligning an LLM policy $\pi _ { w }$ with preference data $\mathcal { P }$ , where each data item $( \mathbf { y } ^ { + } , \mathbf { y } ^ { - } ) \in \mathcal { P }$ consists of the chosen sentence $\mathbf { y } ^ { + }$ and the rejected sentence $\mathbf { y } ^ { - }$ . We denote the set of all chosen sentences as $\mathcal { P } ^ { + } ~ = ~ \{ \mathbf { y } ^ { + } | ( \mathbf { y } ^ { + } , \mathbf { y } ^ { - } ) ~ \in ~ \mathcal { P } \}$ and the set of all rejected sentences as $\mathcal { P } ^ { - } \ = \ \{ \mathbf { y } ^ { - } | ( \mathbf { y } ^ { + } , \mathbf { y } ^ { - } ) \in \mathcal { P } \}$ . The alignment objective is to encourage the LLM policy to generate sentences like the chosen demonstrations ${ \mathcal { P } } ^ { + }$ while discouraging the LLM policy from generating sentences like the rejected demonstrations $\mathcal { P } ^ { - }$ .

Similar to the derivation of the AVA-d training objective, we first focus on performing BIRL to learn a reward function from the preference data $\mathcal { P }$ . We need to consider not only the chosen sentences as positive demonstrations but also the rejected sentences as negative demonstrations. We consider two posterior distributions, which are the reward conditioned on the chosen demonstrations $p ( R | \mathcal { P } ^ { + } )$ and the reward conditioned on demonstrations that differ from rejected demonstrations $p ( R | \overline { { \mathcal { P } ^ { - } } } )$ . Here, $\overline { { \mathcal { P } ^ { - } } }$ denotes demonstrations that differ from ${ \mathcal { P } } ^ { - }$ . Therefore, we define the training objective as Eq. 15, where the first term drives the reward distribution $q _ { \phi }$ close to rewards that motivate the positive behaviors ${ \mathcal { P } } ^ { + }$ , while the second term drives $q _ { \phi }$ close to rewards that motivate behaviors that differ from the negative demonstrations. We refer to the training objective as Contrastive Bayesian Inverse Reinforcement Learning (CBIRL).

$$
\operatorname* { m i n } _ { \phi } D _ { \mathrm { K L } } [ q _ { \phi } ( R ) | | p ( R | \mathcal { P } ^ { + } ) ] + D _ { \mathrm { K L } } [ q _ { \phi } ( R ) | | p ( R | \overline { { \mathcal { P } ^ { - } } } ) ]
$$

Unsurprisingly, the minimization of these two KL divergences is infeasible. We derive the equivalent ELBO objective, as shown in Eq. 16. The derivation is shown in the Technical Appendix.

$$
\begin{array} { r } { \underset { \phi } { \operatorname* { m a x } } [ \mathbb { E } _ { R \sim q _ { \phi } ( \cdot ) } [ \log p ( \mathcal { P } ^ { + } | R ) + \log [ 1 - p ( \mathcal { P } ^ { - } | R ) ] ] ] } \\ { - D _ { \mathrm { K L } } [ q _ { \phi } ( R ) | | p ( R ) ] ] } \end{array}
$$

Towards implementation, we need to further derive the ELBO objective as an approximated variational objective. Note that the main difference between the ELBO of CBIRL and the ELBO of conventional BIRL is the second optimization term in Eq. 16, which minimizes the log-likelihood of the negative demonstrations $\mathcal { P } ^ { - }$ . Therefore, the approximated variational objective also contains the minimization of the negative demonstrations, as shown in Eq. 17. We refer to this training objective as the Approximated Variational Alignment from Preference (AVA-p). By maximizing $\mathcal { F } _ { p } ( \mathcal { P } )$ , on one hand, the LLM policy $\pi _ { w }$ will be encouraged to generate sentences like ${ \mathcal { P } } ^ { + }$ and discouraged to generated sentences like $\mathcal { P } ^ { - }$ ; on the other hand, the policy and the reward will stay consistent under the TD-error constraint.

$$
\mathcal { F } _ { p } ( \mathcal { P } ) = \sum _ { \mathbf { y } ^ { + / - } \in \mathcal { P } } \sum _ { t } \left[ \begin{array} { c } { \beta \log \operatorname { s o f t m a x } \left( \alpha \pi _ { w } ( y _ { t + 1 } ^ { + } | \mathbf { y } _ { 1 : t } ^ { + } ) \right) } \\ { - \beta \log \operatorname { s o f t m a x } \left( \alpha \pi _ { w } ( y _ { t + 1 } ^ { - } | \mathbf { y } _ { 1 : t } ^ { - } ) \right) } \\ { - d _ { t } ( \phi ) + \lambda \log q _ { \phi } \left( \delta _ { t } ( w ) | \mathbf { y } _ { 1 : t + 1 } \right) } \end{array} \right]
$$

To ensure the reward difference between the chosen and rejected demonstrations, we adopt a more intuitive auxiliary training objective, the Contrastive Expected Return (CER) training objective, as shown in Eq. 18, which encourages the reward of the positive demonstrations to be higher than the reward of the negative demonstrations. Note that we only consider the reward of the last timestep in the CER objective. Although we model the intermediate rewards, we still assume that the reward of the last timestep is decisive for the overall expected return, since empirical practice and research (Geva et al. 2023; Hanna, Liu, and Variengien 2024) show that the last position of the Transformer gathers most of the knowledge.

$$
\mathcal { F } _ { c } ( \mathcal { P } ) = \sum _ { \mathbf { y } ^ { + } / - \in \mathcal { P } } \sigma \left[ \mathbb { E } _ { q _ { \phi } ( R | \mathbf { y } ^ { + } ) } [ R ] - \mathbb { E } _ { q _ { \phi } ( R | \mathbf { y } ^ { - } ) } [ R ] \right]
$$

# AVA Pipelines

The AVA training objectives facilitate the joint optimization of the reward function and the policy. Therefore, AVA can be leveraged for both reward modeling and direct optimization, which are two common pipelines in LLM alignment. Both pipelines have their advantages and disadvantages. The reward modeling pipeline can produce a lightweight and reusable reward function for downstream RL fine-tuning while it suffers from the high RL training cost. The direct optimization pipeline is more efficient than reward modeling with RL during training but cannot produce a lightweight reward function for other uses and may suffer from overfitting. The AVA pipeline is shown in Alg. 1. For direct optimization, the initial policy is the initial LLM policy $\pi _ { w ^ { ( 1 ) } }$ .

AVA for Reward Modeling For reward modeling, the initial policy $\pi _ { \theta ^ { ( 1 ) } }$ in Alg. 1 is the initial implicit policy $\pi _ { \psi ^ { ( 1 ) } }$ . In the TQR architecture, the reward function shares the same backbone model with the policy. Our purpose of reward modeling is to obtain an accurate and lightweight reward model. Therefore, we initialize the TQR architecture with a lightweight backbone model. In other words, we initialize the policy with a lightweight pre-trained language model $\pi _ { \psi ^ { ( 1 ) } }$ instead of a large language model. Meanwhile, the reward distribution is also initialized and denoted by $q _ { \phi ^ { ( 1 ) } }$ After the initialization, we leverage either AVA-d or AVAp training objectives to optimize the reward distribution according to the type of the dataset $\mathcal { D }$ . Note that the AVA training objectives require us to jointly train the reward function with the policy, although finally we only need the reward function. After reward modeling, we can leverage RL algorithms to fine-tune the LLM policy $\pi _ { w }$ to maximize the expected reward produced by the trained reward distribution $q _ { \phi }$ , as shown in Eq. 19.

Algorithm 1: The AVA pipeline.   

<html><body><table><tr><td></td><td colspan="3">Data: Dataset D,initial policy Tθ(1),initial reward distribution qφ(1), training epochs T Result: The trained reward distribution q(T)</td></tr><tr><td>2</td><td colspan="3">fori∈{1,.,T}do ifDisdemonstration dataset then</td></tr><tr><td>3</td><td colspan="3">(i+1)←(i)+V(i）Fa(D);</td></tr><tr><td>4</td><td colspan="3">0(i+1）←θ(i)+Vθ(i)Fd(D);</td></tr><tr><td>5</td><td colspan="3">else</td></tr><tr><td>6</td><td colspan="3">(i+1) ← Φ(i)+ V(i）Fp(D)+V(i)Fe(D);</td></tr><tr><td>7</td><td colspan="3">0(i+1）←0(i)+Vθ(）Fp(D)+Vθ(i）Fc(D)；</td></tr><tr><td>8 9</td><td colspan="3">end end</td></tr><tr><td colspan="3">0 return q(T),T(T)</td></tr><tr><td colspan="3"></td></tr></table></body></html>

$$
J ( w ) = \mathbb { E } _ { \mathbf { y } \sim \pi _ { w } ( \cdot ) } \left[ \sum _ { t = 1 } ^ { | \mathbf { y } | - 1 } \mathbb { E } _ { R \sim q _ { \phi } ( \cdot | \mathbf { y } _ { 1 : t + 1 } ) } [ R ] \right]
$$

AVA for Direct Optimization For direct optimization, the initial policy in Alg. 1 is the initial LLM policy $\pi _ { w ^ { ( 1 ) } }$ . In other words, we directly initialize the policy with the pretrained LLM $\pi _ { w ^ { ( 1 ) } }$ and leverage the AVA training objectives to jointly optimize the policy and the reward distribution $q _ { \phi ^ { ( 1 ) } }$ . After training, the LLM policy and the reward distribution are both aligned with the demonstration or preference dataset $\mathcal { D }$ .

# Experiment

# Experiment Setup

Datasets For preference datasets, we consider AnthropicHarmless, Anthropic-Helpful, and OpenAI-Summary and perform reward modeling, RL fine-tuning, and direct optimization on these datasets. For demonstration datasets, we consider Alpaca-GPT-4 and Math-GPT-4o and only perform direct optimization on these datasets.

Metrics For reward modeling, we evaluate the accuracy at which the reward of the chosen sentence is greater than that of the rejected sentence, as well as the win rates of the Bestof-N sampling (Stiennon et al. 2020; Nakano et al. 2021) results. The detailed calculation of win rate is in the Technical Appendix. For RL fine-tuning, we evaluate the win rates of the LLMs fine-tuned with different reward models (i.e., AVA-p/d and baselines). For direct optimization, we evaluate the win rates of LLMs fine-tuned with AVA-p/d against LLMs fine-tuned with baseline approaches.

Pre-trained Models For reward modeling, we initialize the implicit policy with GPT-2 (117M) and BART-base (140M) to see the reward modeling performance with different initializations. For RL fine-tuning and direct optimization, we initialize the LLM policy with Llama-2-7b-chat-hf. The reward models adopted in RL fine-tuning only involve those initialized with GPT-2.

Baselines For reward modeling, we adopt Bradley-Terry (Bradley and Terry 1952) and Preference Transformer (PrefTrans) (Kim et al. 2023) as baselines. For direct optimization from preference, we adopt DPO (Rafailov et al. 2024) as the baseline. For direct optimization from demonstration, we adopt AfD (Sun and van der Schaar 2024) as the baseline. Since AfD constructs preference data from demonstration data and relies on preference-based training objectives, we combine AfD with different preference-based training objectives. Specifically, for reward modeling, we construct AfD w/ Bradley-Terry, Afd w/ Pref-Trans, and AfD w/ AVAp. For direct optimization, we construct AfD w/ DPO. For win rate evaluations of aligned LLMs, we also adopt supervised fine-tuning (SFT) as the baseline.

Ablation Variants We construct the following variants of AVA-p and AVA-d training objectives for ablation studies:

• AVA-p/d w/o rwt: AVA-p/d without reward weighting, which removes the computation of reward weights and the weighted rewards from the TQR architecture.   
• AVA-p w/o neg: AVA-p without the negative demonstration, which removes the minimization of the likelihood of the negative demonstrations. Note that the objective does not completely degenerate into the AVA-d training objective since we still keep the CER auxiliary objective.   
• AVA-p w/o irl: AVA- $\cdot \mathbf { p }$ without inverse reinforcement learning, which removes the TD-error constraint and the reward prior assumption and only keeps the likelihood optimization, which can be regarded as contrastive supervised fine-tuning.   
• AVA-p w/o cer: AVA-p without CER auxiliary objective.   
• AVA-p/d w/o ptq: AVA-p/d without pre-trained Q-value head, which does not reuse the LM head of the pretrained policy as the Q-value head but initializes the Qvalue head from scratch.

Moreover, the prior reward distribution is assumed to be the standard Gaussian distribution. For detailed experiment setup, please refer to our code and the Experiment Details section of the Technical Appendix.

# Reward Modeling

Table 1 reports the reward accuracy of baseline and AVA training objectives. The results show that AVA-p surpasses Bradley-Terry and Pref-Trans in reward accuracy on all reported reward modeling tasks with different initial models and datasets. The ablation results further reveal that AVA-p achieves the highest reward accuracy on the greatest number of tasks compared to ablated training objectives, which suggests that removing any module from AVA-p diminishes the reward accuracy on most tasks. Furthermore, we consider the chosen half of the preference data as demonstration data and train the reward model on it using the AVA-d training objective. Surprisingly, AVA-d achieves the best performance on 2 out of 6 tasks, despite learning solely from the chosen demonstrations. We also evaluate the RewardBench score (Lambert et al. 2024) of the ensemble model over all 6 tasks for Bradley-Terry, Pref-Trans, and AVA-p, which achieve scores of 55.91, 57.05, and 59.84, respectively.

Table 1: Reward accuracy of AVA and baseline objectives.   

<html><body><table><tr><td rowspan="2"></td><td colspan="2">Harmless</td><td colspan="2">Helpful</td><td colspan="2">Summary</td></tr><tr><td>gpt2</td><td>bart</td><td>gpt2</td><td>bart</td><td>gpt2</td><td>bart</td></tr><tr><td>Baselines Bradley-Terry</td><td colspan="6">70.02 68.96 69.39 67.56 59.27 59.27</td></tr><tr><td>Pref-Trans Ours</td><td>70.26 71.32</td><td></td><td>71.37</td><td>72.37</td><td>59.3156.91</td><td></td></tr><tr><td>AVA-p</td><td colspan="6">70.27</td></tr><tr><td>AVA-p w/o rwt</td><td>70.06</td><td>72.30 70.73</td><td>72.37 69.81</td><td>74.84 69.32</td><td>61.79 60.55</td><td>64.31 58.89</td></tr><tr><td>AVA-p w/o neg</td><td>70.54</td><td>70.36 69.75</td><td></td><td>69.15</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>62.06</td><td>58.65</td></tr><tr><td>AVA-pw/o irl</td><td></td><td>69.48 67.46 68.87 65.38</td><td></td><td></td><td>58.96 58.46</td><td></td></tr><tr><td>AVA-p w/o cer</td><td></td><td>70.06 70.73 69.81</td><td></td><td>69.32</td><td>60.55</td><td>58.58</td></tr><tr><td>AVA-p w/o ptq</td><td>68.67</td><td>68.69</td><td>68.51</td><td>67.60</td><td>61.25</td><td>57.76</td></tr><tr><td>AVA-d</td><td>70.54</td><td>70.36</td><td>69.75</td><td>69.15</td><td>62.06</td><td>59.00</td></tr></table></body></html>

![](images/71f83077694a1871d3262746a0711e9fc6e832c0b466f32ae10c05a36521768c.jpg)  
Figure 2: Scalability of AVA-p and baseline objectives.

We demonstrate the scalability of AVA-p for reward modeling by performing the Helpful task using google flan-t5 models (Chung et al. 2024) with different sizes. Specifically, we perform reward modeling based on flan-t5-small, flant5-base, and flan-t5-large models using baseline and AVA- $\cdot \mathbf { p }$ objectives. The scalability result in Fig. 2 shows that AVAp scales better with the increase of model size compared to baseline objectives.

To further evaluate reward modeling performance, we employ Best-of-N (BoN) sampling. We evaluate the win rates of BoN w/ AVA-p against BoN w/ Bradley-Terry and BoN w/ Pref-Trans, where “BoN w/ xxx” means that the reward model used for BoN is trained with the “xxx” training objective. Additionally, we evaluate the win rate of BoN w/ AVA-p against the stochastic sampling results without BoN. Table 2 reports the win rates of the reward model trained with AVA-p against reward models trained with baseline objectives in BoN sampling. The results further demonstrate that AVA-p surpasses Bradley-Terry and Pref-Trans in reward modeling.

Table 2: Win rates of BoN with AVA-p reward model.   

<html><body><table><tr><td>Task</td><td>Opponent</td><td>Win↑ Tie</td><td>Lose↓</td></tr><tr><td rowspan="2"></td><td>Stochastic</td><td>43.0 17.4</td><td>39.6</td></tr><tr><td>Harmless BoNw/Bradley-Terry BoN w/Pref-Trans</td><td>28.8 42.6 35.6 31.9</td><td>28.6 32.5</td></tr><tr><td rowspan="2">Helpful</td><td>Stochastic</td><td>26.1 50.4</td><td>23.5</td></tr><tr><td>BoN w/Bradley-Terry</td><td>13.2 76.1</td><td>10.7</td></tr><tr><td rowspan="2"></td><td>BoN w/Pref-Trans</td><td>19.3 62.3</td><td>18.4</td></tr><tr><td>Stochastic</td><td>60.2 0.8</td><td>39.0</td></tr><tr><td rowspan="2"></td><td>Summary BoN w/ Bradley-Terry</td><td>34.6 34.5</td><td>30.9</td></tr><tr><td>BoN w/Pref-Trans</td><td>43.6 25.8</td><td>30.6</td></tr></table></body></html>

Table 3: Win rates of PPO with AVA-p reward model.   

<html><body><table><tr><td>Task</td><td>Opponent</td><td>Win↑</td><td>Tie</td><td>Lose↓</td></tr><tr><td rowspan="3"></td><td>SFT</td><td>42.5</td><td>23.4</td><td>34.1</td></tr><tr><td>HarmlessPPOw/Bradley-Terry</td><td>9.2 81.7</td><td></td><td>9.1</td></tr><tr><td>PPO w/Pref-Trans</td><td></td><td>9.0 83.2</td><td>7.8</td></tr><tr><td rowspan="3">Helpful</td><td>SFT</td><td>23.3 58.8</td><td></td><td>18.0</td></tr><tr><td>PPO w/Bradley-Terry</td><td></td><td>1.8 97.2</td><td>1.0</td></tr><tr><td>PPO w/Pref-Trans</td><td></td><td>2.6 95.8</td><td>1.6</td></tr><tr><td rowspan="3"></td><td>SFT</td><td>73.8</td><td>1.4</td><td>24.7</td></tr><tr><td>Summary PPO w/ Bradley-Terry</td><td>18.5</td><td>66.3</td><td>15.2</td></tr><tr><td>PPO w/Pref-Trans</td><td>33.9 34.6</td><td></td><td>31.5</td></tr></table></body></html>

# RL Fine-tuning

We adopt the PPO algorithm (Schulman et al. 2017) to finetune LLMs to maximize the reward produced by different reward models. We evaluate the win rates of PPO w/ AVAp against PPO w/ Bradley-Terry and PPO w/ Pref-Trans, where “PPO w/ xxx” means that the reward model used for PPO fine-tuning is trained with the “xxx” training objective. We also evaluate the win rate of PPO w/ AVA-p against supervised fine-tuning (SFT), where the LLM is fine-tuned on the chosen half of the preference data with supervised learning. The results in Table 3 show that AVA-p outperforms the baseline reward modeling objectives on all reported tasks in downstream RL fine-tuning of the LLM.

# Direct Optimization

From Preference We adopt AVA- $\cdot \mathbf { p }$ and DPO (Rafailov et al. 2024) to directly optimize the LLM from preference data and evaluate the win rates of AVA-p against DPO and

Table 4: Win rates of direct optimization with AVA-p.   

<html><body><table><tr><td>Task</td><td>Opponent</td><td>Win↑</td><td>Tie</td><td>Lose↓</td></tr><tr><td rowspan="2">Harmless</td><td>SFT</td><td>37.1</td><td>28.9</td><td>34.0</td></tr><tr><td>DPO</td><td>13.7</td><td>73.8</td><td>12.5</td></tr><tr><td rowspan="2">Helpful</td><td>SFT</td><td>22.5</td><td>59.6</td><td>17.9</td></tr><tr><td>DPO</td><td>14.4</td><td>72.4</td><td>13.2</td></tr><tr><td rowspan="2">Summary</td><td>SFT</td><td>59.0</td><td>7.3</td><td>33.7</td></tr><tr><td>DPO</td><td>44.9</td><td>11.0</td><td>44.1</td></tr></table></body></html>

Table 5: Win rates of direct optimization with AVA-d.   

<html><body><table><tr><td>Task</td><td>Opponent</td><td>Win↑</td><td>Tie</td><td>Lose↓</td></tr><tr><td rowspan="3">Alpaca</td><td>SFT</td><td>58.1</td><td>7.2</td><td>34.7</td></tr><tr><td>DPO w/ AfD</td><td>57.2</td><td>6.9</td><td>35.9</td></tr><tr><td>AVA-p w/ AfD</td><td>56.5</td><td>7.1</td><td>36.4</td></tr><tr><td rowspan="3">Math</td><td>SFT</td><td>47.0</td><td>9.7</td><td>43.3</td></tr><tr><td>DPO w/ AfD</td><td>44.3</td><td>11.4</td><td>44.3</td></tr><tr><td>AVA-p w/ AfD</td><td>45.4</td><td>11.4</td><td>43.1</td></tr></table></body></html>

SFT. The results in Table 4 show that AVA-p outperforms DPO in direct optimization from preference data.

From Demonstration We adopt AVA-d and AfD (Sun and van der Schaar 2024) to directly optimize the LLM from demonstration data. We evaluate the win rates of AVA-d against SFT, DPO w/ AfD, and AVA-p w/ AfD, where “xxx w/ AfD” means applying the “xxx” training objective on AfD-format data. The results in Table 5 show that AVAd outperforms the AfD approaches in direct optimization from demonstration data. Moreover, AVA-d is more trainingefficient since AfD requires supervised fine-tuning and sampling from LLM policies.

Limitation The direct optimization pipeline achieves little improvement over the DPO baseline in most tasks. Future work should be performed to address the limitations of the direct optimization pipeline.

# Conclusion

We present AVA, a flexible novel LLM alignment objective with enhanced capabilities. The flexibility of AVA is evident in two aspects. Firstly, AVA can utilize either preference data or demonstration data for alignment purposes. Secondly, AVA can be integrated into the reward modeling and RL fine-tuning pipeline or used to directly optimize the LLM. The representation and generalization capabilities of AVA are also evident in two aspects. Theoretically, AVA formulates reward modeling as a BIRL problem, facilitating both intermediate reward modeling and direct reward modeling on demonstration. Experimentally, AVA achieves superior reward accuracy in reward modeling tasks and higher win rates in RL fine-tuning and direct optimization of LLMs, which demonstrates the alleviation of the reward hacking issue and improved alignment performance.