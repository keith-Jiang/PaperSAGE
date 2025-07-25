# Self-Evolutionary Large Language Models Through Uncertainty-Enhanced Preference Optimization

Jianing Wang, Yang Zhou, Xiaocheng Zhang, Mengjiao Bao, Peng Yan\*

Meituan wangjianing16, yanpeng04 @meituan.com

# Abstract

Iterative preference optimization has recently become one of the de-facto training paradigms for large language models (LLMs), but the performance is still underwhelming due to too much noisy preference data yielded in the loop. To combat this issue, we present an Uncertainty-enhanced Preference Optimization (UPO) framework to make the LLM self-evolve with reliable feedback. The key idea is mitigating the noisy preference data derived from the current policy and reward models by performing pair-wise uncertainty estimation and judiciously reliable feedback sampling. To reach this goal, we thus introduce an estimator model, which incorporates Monte Carlo (MC) dropout in Bayesian neural network (BNN) to perform uncertainty estimation for the preference data derived from the LLM policy. Compared to the existing methods that directly filter generated responses based on the reward score, the estimator focuses on the model uncertainty in a pair-wise manner and effectively bypasses the confirmation bias problem of the reward model. Additionally, we also propose an uncertainty-enhanced self-evolution algorithm to improve the robustness of preference optimization and encourage the LLM to generate responses with both high reward and certainty. Extensive experiments over multiple benchmarks demonstrate that our framework substantially alleviates the noisy problem and improves the performance of iterative preference optimization.

# Introduction

Recently, the NLP community has witnessed the success of preference optimization for large language models (LLMs), which has become one of the significant ingredients of recent revolutions (Brown et al. 2020; OpenAI 2023; Tunstall et al. 2023; Zheng et al. 2023b). As a post-training process of LLM, preference optimization aims to align the LLM policy with the labeled human feedback or AI feedback data. Early approaches utilize reinforcement learning (RL) to train the LLM policy online based on the human feedback simulated by a tuned reward model, referred to as RLHF (Christiano et al. 2017; Lee, Smith, and Abbeel 2021; Ouyang et al. 2022). Besides, offline direct preference optimization (DPO) and some variants view LLM-as-judge (Yuan et al. 2024)

Preference Preference Preference Data Data Data T 1 Reward Estimator Reward Model Model Model LLM Policy LLM Policy LLM Policy (a) Offline Preference  (b) Iterative Preference  (c) Uncertainty-Enhanced Optimization Optimization Preference Optimization

and directly align the policy with feedback (Rafailov et al.   
2023; Ethayarajh et al. 2024).

Despite the success, these approaches relied on massive labeled preference data which requires tons of manpower and resources. To combat this issue, some recent researches introduce a novel iterative preference optimization (Pang et al. 2024; Chen et al. 2024; Kim et al. 2024; Xu et al. 2023; Rosset et al. 2024; Wu et al. 2024; Xie et al. 2024). As shown in Figure 1 (b), the offline methods can be iteratively applied similarly to the self-training procedure, where the previously trained policy generates new preference data which are then used to train the new policy. Generally, a reward model is also required in the iteration to simulate feedback for selfevolve (Xu et al. 2024; Tao et al. 2024).

However, we find one of the potential pitfalls in the iteration is that the reward model may assign unsuitable scores for the responses, leading to deriving multiple noisy preference pairs and hindering performance. This problem gets exaggerated when the interaction number increases (Han et al. 2018; Choi et al. 2024). Hence, the paramount challenge is meticulously selecting reliable preference data and making the preference optimization not distorted by noise. A simple solution is to choose one pair in which two responses ignifying a notable disparity in terms of the reward score (Pang et al. 2024). Yet, it can not bypass the confirmation bias problem (Andersen and Maalej 2022; Rizve et al. 2021; Wang et al. 2021) in the self-training-like paradigm.

To this end, we present an Uncertainty-enhanced Preference Optimization (UPO) framework to circumvent the noise problem. To elaborate, we introduce an estimator model that essentially performs a classification task to detect which response is more suitable for the query. As shown in Figure 1 (c), different from the existing reward model that can only assign a scalar score in the inference stage, it can be equipped with a Monte Carlo (MC) dropout technique, which is the approximation technique in Bayesian Neural Network (BNN) (Gal and Ghahramani 2016; Wang and Yeung 2016), to estimate the uncertainty of each preference pair (Wang et al. 2023a). Thus, a sampling signal based on the model certainty can be used to represent the reliability of the preference pair. To further improve the robustness of the iteration preference optimization, we additionally develop an uncertainty-enhanced self-evolution algorithm. Specifically, we first use the estimator certainty to split the generated preference data into reliable pairs and unreliable pairs, where reliable pairs can easily provide high-quality feedback and unreliable pairs are quite hard to express the preference. We thus integrate the uncertainty into DPO to encourage the LLM policy to know what generated pairs are reliable or unreliable feedback. Therefore, with the dual blessing of rewards and uncertainty, the new LLM policy can generate responses with both high reward and certainty.

We conduct extensive experiments on two universal NLP benchmarks (i.e., AlpacaEval 2.0 (Dubois et al. 2024) and MT-Bench (Zheng et al. 2023a)) and two mathematics reasoning tasks (i.e., GSM8K (Cobbe et al. 2021) and MATH (Hendrycks et al. 2021)), results demonstrate that our UPO framework substantially enhances the effectiveness of preference alignment, and achieves the best performance in auto evaluation 1.

# Preliminaries

We first introduce the background knowledge of the iteration preference optimization and Bayesian neural network.

# Preference Optimization

Suppose that the LLM policy is denoted as $\pi _ { \boldsymbol { \theta } }$ and it has been tuned after the pre-training and supervised fine-tuning (SFT) stage. The goal of preference optimization is to post-train the LLM policy on well-manual preference data. Formally, given a labeled preference data $\bar { \mathcal { D } } \ = \ \{ ( x , y _ { w } , y _ { l } ) \}$ which consists of multiple triples 2 conditioned by a prompt $x \in \mathcal { X }$ , a preferred response $y _ { w } \in \mathcal { V }$ as the winner (chosen) and a dispreferred response $y _ { l } \in \mathcal { V }$ as the loser (rejected). $\mathcal { X }$ and $y$ are respectively prompt and output distributions.

During the optimization, a series of methods leverage RLHF to process the feedback online. Generally, it requires a reward model pre-trained on the preference data through the Bradley-Terry model (Bradley and Terry 1952) as:

$$
p ( y _ { w } \succ y _ { l } ) = \frac { \exp { \left( r _ { \phi } ( x , y _ { w } ) \right) } } { \exp { \left( r _ { \phi } ( x , y _ { w } ) \right) } + \exp { \left( r _ { \phi } ( x , y _ { l } ) \right) } } ,
$$

where $r _ { \phi } ( x , y )$ is the reward model and outputs a scaler score as the reward of response $y$ towards the given prompt

$x$ . The parameters of $r _ { \phi } ( x , y )$ can be updated as the following maximum-likelihood objective:

$$
\mathcal { L } _ { r } ( \phi ) = - \mathbb { E } _ { ( x , y _ { w } , y _ { l } ) \sim \mathcal { D } } [ \log ( \sigma ( r _ { \phi } ( x , y _ { w } ) ) - \sigma ( r _ { \phi } ( x , y _ { l } ) ) ) ] ,
$$

where $\sigma ( \cdot )$ is the sigmoid function. When a pre-trained reward model is available, the LLM policy can be repetitively aligned to the new pairs derived from the reward model with a proximal policy optimization (PPO) algorithm:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { r l h f } } ( \theta ) = - \mathbb { E } _ { x \sim \mathcal { X } , y \sim \pi _ { \theta } ( y \vert x ) } [ r _ { \phi } ( x , y ) ] } \\ & { \quad \quad \quad + \beta \mathbb { E } _ { x \sim \mathcal { X } } [ \mathbf { K L } ( \pi _ { \theta } ( \cdot \vert x ) \vert \vert \pi _ { \mathrm { r e f } } ( \cdot \vert x ) ) ] , } \end{array}
$$

where $\beta \ > \ 0$ is the balance factor, the $\mathrm { K L }$ divergence $\mathrm { K L } ( \cdot | | \cdot )$ aims to maintain the original output distribution similar to the consistency regularization. $\pi _ { \mathrm { r e f } }$ is the reference model which shares the same parameters with $\pi _ { \boldsymbol { \theta } }$ but is frozen after the SFT stage.

In contrast to RLHF, DPO aims to follow the LLM-asjudge paradigm by directly optimizing the policy:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { d p o } } ( \theta ) = - \mathbb { E } _ { ( x , y _ { w } , y _ { l } ) \sim \mathcal { D } } \log \sigma ( \beta h _ { \pi _ { \mathrm { r e f } } } ^ { \pi _ { \theta } } ( x , y _ { w } , y _ { l } ) ) , } \end{array}
$$

where $h _ { \pi _ { \mathrm { r e f } } } ^ { \pi _ { \theta } } ( x , y _ { w } , y _ { l } )$ is the reward difference between prefered response and disprefered response:

$$
h _ { \pi _ { \mathrm { r e f } } } ^ { \pi _ { \theta } } ( x , y _ { w } , y _ { l } ) = \log \frac { \pi _ { \theta } ( y _ { w } | x ) } { \pi _ { \mathrm { r e f } } ( y _ { w } | x ) } - \log \frac { \pi _ { \theta } ( y _ { l } | x ) } { \pi _ { \mathrm { r e f } } ( y _ { l } | x ) } .
$$

# Bayesian Neural Network (BNN)

In the iteration procedure, the preference pairs derived from the reward model or LLM itself may contain noisy data and hinder the whole performance. We thus briefly describe the knowledge of BNN as the basic support for denoising. Concretely, suppose a neural model $f _ { \psi }$ can predict the preference, the vanilla BNN assumes a prior distribution over its model parameters $\psi$ . In other words, BNN averages over all the possible weights instead of directly optimizing for the weights (Mukherjee and Awadallah 2020). Given a labeled preference $\mathcal { D }$ , the parameter can be optimized by the posterior distribution $p ( \psi | \mathcal { D } )$ . During model inference, given one unlabeled triple $( \dot { x } , \dot { y } _ { w } , y _ { l } ) \in \mathcal { D } _ { u }$ where $\mathcal { D } _ { u }$ is the responses set generated by the LLM policy and reward model, the probability distribution can be formed as:

$$
p ( c | x , y _ { w } , y _ { l } ) = \int _ { \psi } p ( c | f _ { \psi } ( x , y _ { w } , y _ { l } ) ) p ( \psi | \mathcal { D } _ { u } ) d \psi ,
$$

where $c \in \{ 0 , 1 \}$ is the label represents $y _ { w } \succ y _ { l }$ is unsuitable or suitable. To make the equation tractable, we can find a surrogate tractable distribution $q ( \psi )$ based on a dropout distribution (Srivastava et al. 2014) that makes the model posterior easy to calculate. Thus, we can sample $T$ masked model weights $\{ \widetilde { \psi } _ { t } \} _ { t = 1 } ^ { T } \sim q ( \psi )$ from the current model. The approximate poste rior for each triple is:

$$
p ( c | x , y _ { 1 } , y _ { 2 } ) \approx \frac { 1 } { T } \sum _ { t = 1 } ^ { T } p ( c | f _ { \widetilde { \psi } _ { t } } ( x , y _ { 1 } , y _ { 2 } ) ) ,
$$

where $y _ { 1 } , y _ { 2 }$ are the responses.

(1) Fine-tuning (2) Generated Responses Rewarding (3) Reliable Preference Learning Prompt Set Pseudo Pairs Unreliable Pairs Reliable Pairs   
PrefeLraebnecledData Prompt: ... B C 自 B C A D UEncheartnacientdy Sampling 1.15 Self-Evolution 北 A Prompt √ √ B: 0.42   
Reward MC Dropout Certainty Reward Margin Reward   
Estimator B D √ Reward -0.13 ↑ √ ? √ Estimator × Policy Policy π(t-1) D: -0.21 Estimator (t-1) ...AB AC AD BC BD CD ... Policy T (t) Generating & Rewarding Permuting Uncertainty Estimating (4) Repeat Until Convergency

# Methodology

In this section, we develop an Uncertainty-enhanced Preference Optimization (UPO) framework illustrated in Figure 2, specialized for the improvement of the LLM selfevolve through iteration preference optimization paradigm. The framework consists of three main procedures, i.e., initial stage fine-tuning, generated responses rewarding, and reliable preference learning.

# Initial Stage Fine-tuning

In the initial stage, suppose that there is a supervised finetuned LLM $\pi _ { \mathrm { s f t } }$ and a corresponding labeled preference data $\mathcal { D } ^ { ( 0 ) }$ derived from human or AI feedback. We follow the previous works (Pang et al. 2024; Ouyang et al. 2022; Rafailov et al. 2023; Kim et al. 2024) to use the initialized preference data to train a reward model $r _ { \phi } ^ { ( 0 ) }$ based on the Bradley-Terry model in Eq. 1, and a weak LLM policy $\pi _ { \theta } ^ { ( 0 ) }$ optimized from $\pi _ { \mathrm { s f t } }$ via DPO in Eq. 4 .

In addition, we also develop an estimator which is essentially a binary classifier that detects whether a pair is suitable. Different from the reward model that only assigns a scaler score, the estimator model can provide the probability of the fact that the preferred response is better than the dispreferred one, and will be used for uncertainty estimation in the reliable preference learning stage. To train the model, we need to reform the existing preference data.

We first transform the original preference triple $( x , y _ { w } , y _ { l } ) ~ \in ~ \mathcal { D } ^ { ( 0 ) }$ into a unified prompt, and the template is denoted as $\mathcal { T } ( x , y _ { w } , y _ { l } )$ demonstrated in Appendix A. Therefore, we can construct a binary classification dataset to train an estimator model. To make the training easier, we directly choose the backbone from ⇡✓(0) and add an external classification head to project the last layer’s representations at the last token position into a binary space. The training objective is formulated as:

$$
\mathcal { L } _ { \mathrm { e s t } } ( \psi ) = - \mathbb { E } _ { ( x , y _ { w } , y _ { l } ) \sim \mathcal { D } ^ { ( 0 ) } } \log f _ { \psi } ( \mathcal { T } ( x , y _ { w } , y _ { l } ) ) .
$$

# Generated Responses Rewarding

The LLM policy will be iteratively updated with the coordination of reward and estimator models. For the $i$ -th iteration, we assume that the current LLM policy is $\pi _ { \theta } ^ { ( i - 1 ) }$ (i−1). In pursuit of obtaining more preference data to evolve the policy, we urge ⇡ $\pi _ { \theta } ^ { ( i - 1 ) }$ to generate multiple responses from new sampled prompts. Specifically, give a prompt $x \in \mathcal { X }$ , the corresponding responses can be represented as $\{ y _ { j } \} _ { j = 1 } ^ { N } \sim$ ⇡✓(i−1)(·|x), where N ≥ 4 is the number of responses. After that, the reward model $r _ { \phi } ^ { ( i - 1 ) }$ at the previous stage will be used to assign a scale score for each response. Hence, we can sort the responses with the reward score and obtain all permutations.

Considering that too many permutations of each prompt will affect the execution efficiency of the framework, we pre-screen these permutations by a simple heuristic rule: we remove the pair whose chosen response (i.e., winner $y _ { w }$ ) has a lower rank or rejected response (i.e., loser $y _ { l }$ ) has a higher rank. For example, if we get six responses in descending sort (has a total of 15 pairs) and the top three responses are viewed as higher rank, only no more than 9 pairs will be used, expediting the process of iteration procedure because fewer data need to be estimated in the next stage. At last, we denote the final generated permutations with the corresponding prompt as the pseudo preference pairs D(u .

# Reliable Preference Learning

In this stage, we aim to use the estimator model 4 to select reliable reference data based on uncertainty estimation.

Given an estimator model $f _ { \psi } ^ { ( i - 1 ) }$ and a pseudo preference data (ui generated by LLM policy and reward model. We assume that each preference triple is independent of another and can be measured individually. Specifically, we follow (Houlsby et al. 2011; Wang et al. 2023b) to leverage information gain of the model parameters to estimate how certain the estimator model is to the triple with respect to the true preference. Therefore, we can obtain the formulation:

$$
\begin{array} { r l } & { \mathbb { B } ( \tilde { c } _ { j } , \psi | \mathcal { T } _ { j } , \mathcal { D } _ { u } ^ { ( i ) } ) = \mathbb { H } ( \tilde { c } _ { j } | \mathcal { T } _ { j } , \mathcal { D } _ { u } ^ { ( i ) } ) - } \\ & { \qquad \quad \mathbb { E } _ { p ( \psi | \mathcal { D } _ { u } ^ { ( i ) } ) } [ \mathbb { H } ( \tilde { c } _ { j } | \mathcal { T } _ { j } , \psi ) ] , } \end{array}
$$

where $\mathbb { H } ( \cdot )$ is the entropy, $\mathcal { T } _ { j } ~ = ~ \mathcal { T } ( x _ { j } , y _ { w j } , y _ { l j } )$ is the input template of $j$ -th triple from $\mathcal { D } _ { u } ^ { ( i ) }$ . $\tilde { c } _ { j } ~ \in ~ \{ 0 , 1 \}$ denote the prediction of estimator model. $p ( \psi | \mathcal { D } _ { u } ^ { ( i ) } )$ is the posterior distribution. Through this information gain, we can find that a lower $\mathbb { B } ( \tilde { c } _ { j } , \psi | \mathcal { T } _ { j } , \mathcal { D } _ { u } ^ { ( i ) } )$ value means that the estimator model is more certain about the prediction, as higher certainty corresponds to lower information gain. In other words, the preference triples with higher certainty and is more reliable feedback towards the prompt.

For the implementation details, we use MC Dropout in BNN to estimate the information gain. Specifically, we open the dropout and repeat $T$ (default set as 10) times to get independent and identically distributed (i.i.d.) predictions:

$$
\begin{array} { r l } & { \displaystyle \hat { \mathbb { B } } ( \tilde { c } _ { j } , \psi | { \mathcal T } _ { j } , { \mathcal D } _ { u } ^ { ( i ) } ) = - \sum _ { c \in \{ 0 , 1 \} } ( \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \hat { p } _ { c } ^ { t } ) \log ( \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \hat { p } _ { c } ^ { t } ) } \\ & { \quad \quad \quad \quad \quad + \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sum _ { c \in \{ 0 , 1 \} } \hat { p } _ { c } ^ { t } \log ( \hat { p } _ { c } ^ { t } ) , } \end{array}
$$

where $\hat { p } _ { c } ^ { t } = p ( c | f _ { \widetilde { \psi } _ { t } } ( \mathcal { T } _ { j } ) )$ is the predict probability for the triple $( x _ { j } , y _ { w j } , y _ { l j } )$ derived from the $t$ -th masked model $\widetilde { \psi } _ { t } \sim q ( \dot { \psi } )$ .

# Uncertainty-Enhanced Self-Evolution

In the reliable preference learning stage, we also present an uncertainty-enhanced self-evolution algorithm to improve the robustness of LLM alignment. Based on the uncertainty estimation, we aspire for the LLM policy tune on the reliable preference data. So we define a sampling weight for each data. Given a preference data $\mathcal { D } _ { u } ^ { ( i ) }$ and each triple has a information gain value $\hat { \mathbb { B } } ( \tilde { c } _ { j } , \psi | \mathcal { T } _ { j } , \mathcal { D } _ { u } ^ { ( i ) } )$ , the sampling weight for the current iteration stage $i$ is defined as:

$$
\mathcal { P } _ { j } ^ { ( i ) } = \frac { ( 1 - \hat { \mathbb { B } } ( \tilde { c } _ { j } , \psi | \mathcal { T } _ { j } , \mathcal { D } _ { u } ^ { ( i ) } ) ) \mu } { \sum _ { k } ( 1 - \hat { \mathbb { B } } ( \tilde { c } _ { k } , \psi | \mathcal { T } _ { k } , \mathcal { D } _ { u } ^ { ( i ) } ) ) \mu } ,
$$

where $\mu > 0$ is the hyper-parameter, and $\mathcal { P } _ { j } ^ { ( i ) }$ is the probability that the preference triple $( x _ { j } , y _ { w j } , y _ { l j } )$ can be sampled as reliable data, i.e., $\begin{array} { r } { \sum _ { j } \mathcal { P } _ { j } ^ { ( i ) } = \dot { 1 } } \end{array}$ .

With the measure of the uncertainty-aware sampling weight, we rewrite the DPO 5 in Eq. 4 to make the LLM

capture two kinds of feedback: 1) what responses are better when given a prompt, and 2) what preference triples are better for the LLM to learn preference. Formally:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { u p o } } = - \mathbb { E } _ { ( x _ { j } , y _ { w j } , y _ { l j } ) \sim \mathcal { D } _ { u } ^ { ( i ) } } } \\ & { \left[ \left( 1 - \alpha _ { j } ^ { ( i ) } \right) \log \sigma ( \beta h _ { \pi _ { \theta } ^ { ( i - 1 ) } } ^ { \pi _ { \theta } ^ { ( i ) } } ) + \alpha _ { j } ^ { ( i ) } \log \sigma ( - \beta h _ { \pi _ { \theta } ^ { ( i - 1 ) } } ^ { \pi _ { \theta } ^ { ( i ) } } ) \right] , } \end{array}
$$

where $h _ { \pi _ { \theta } ^ { ( i - 1 ) } } ^ { \pi _ { \theta } ^ { ( i ) } }$ ⇡(✓i 1) is the reward margin and defined as:

$$
h _ { \pi _ { \theta } ^ { ( i - 1 ) } } ^ { \pi _ { \theta } ^ { ( i ) } } = \log \frac { \pi _ { \theta } ^ { ( i ) } ( y _ { w j } | x _ { j } ) } { \pi _ { \theta } ^ { ( i - 1 ) } ( y _ { w j } | x _ { j } ) } - \log \frac { \pi _ { \theta } ^ { ( i ) } ( y _ { l j } | x _ { j } ) } { \pi _ { \theta } ^ { ( i - 1 ) } ( y _ { l j } | x _ { j } ) } .
$$

We underscore that $0 \leq \alpha _ { j } \leq 1$ is the uncertainty-aware weight for the triple $( x _ { j } , y _ { w j } , y _ { l j } )$ and is used to balance two items in Eq. 12. In a nutshell, a lower $\alpha _ { j }$ value can encourage the LLM to focus on the given preference data. If the preference data is not reliable according to the uncertainty estimation, we not only expect to reduce the influence of this data but also let the LLM know that the pseudo-labeled preferred response is not suitable and needs to be reversed. So we follow the idea of label smoothing to design $\alpha _ { j }$ as:

$$
\alpha _ { j } = \frac { 1 } { \mathscr { P } _ { j } + 1 } .
$$

In addition, to improve the robustness of the iteration preference optimization, we follow (Pang et al. 2024) to add a negative log-likelihood loss for each preference triple as:

$$
\mathcal { L } _ { \mathtt { u p o + n l l } } = \mathcal { L } _ { \mathtt { u p o } } + \lambda \mathbb { E } _ { ( x _ { j } , y _ { w j } , y _ { l j } ) \sim \mathcal { D } ^ { ( i ) } } \frac { \log \pi _ { \theta } ^ { ( i ) } ( y _ { w j } | x _ { j } ) } { | r _ { \phi } ^ { ( i - 1 ) } ( x _ { j } , y _ { w j } ) | } ,
$$

where $\lambda > 0$ is the hyper-parameter. The whole algorithm is shown in Algorithm 1 in Appendix B.

# Experiments

In this section, we choose universal NLP and mathematics reasoning tasks to evaluate the UPO framework.

# Universal NLP Tasks

Following the practice in previous works, we validate the performance of LLM policy trained through the UPO framework over AlpacaEval 2.0 (Dubois et al. 2024) and MTBench (Zheng et al. 2023a). The benchmark of AlpacaEval 2.0 consists of 805 instructions and can be used to approximately head-to-head test the length-controlled (LC) weighted win rate of preference annotated by GPT-4. MTBench aims to evaluate the capability (scoring from 0 to 10) of the LLM policy to solve multiple basic problems such as writing, roleplay, reasoning, math, coding, extraction, stem, and humanities.

For the implementation setups, we choose zephyr-7b-sft-full (default as Zephyr-7B) as the backbone, which has been further instruction-tuned over UltraChat200K dataset from Mistral-7B (Jiang et al. 2023). The labeled preference data we used is UltraFeedback (Cui et al. 2023), which consists of 61K prompts post-processed by Tunstall et al. (2023) . We also select UltraChat200K as the prompt set. We repeatedly train three models (i.e., LLM policy, reward, and estimator) for three iterations. For the baselines, we choose SFT and DPO trained from Zephyr-7B to make a comparison. In addition, we also collect all cleaned preference data from the initial stage and three iterations and use DPO to train a model as UPO-Merge. More details of these benchmarks and hyper-parameters of each training iteration are listed in Appendix C.

Table 1: Main results derived from GPT-4 auto evaluation on AlpacaEval 2.0 (LC weighted win rate $\%$ compared with reference of GPT-4) and MT-Bench (absolute score).   

<html><body><table><tr><td>Models</td><td>Align</td><td>AlpacaEval 2.0</td><td>MT-bench</td></tr><tr><td>Mistral-7B</td><td>no</td><td>0.17</td><td>3.25</td></tr><tr><td>Alpaca-7B</td><td>no</td><td>5.88</td><td>5.81</td></tr><tr><td>Zephyr-7B-SFT</td><td>no</td><td>5.84</td><td>6.18</td></tr><tr><td>Zephyr-7B-DPO</td><td>yes</td><td>9.12</td><td>6.79</td></tr><tr><td>Zephyr-7B-UPO</td><td>yes</td><td>13.04</td><td>7.02</td></tr><tr><td>Zephyr-7B-UPO-Merge</td><td>yes</td><td>12.04</td><td>6.85</td></tr></table></body></html>

Main Results As shown in Table 1, the results of AlpacaEval 2.0 denote the win rate compared to the reference generated by GPT-4, and we can see that the LLM policy of Zephyr-UPO after three iterations achieves the best win rate against GPT-4 and improves by $7 . 2 0 \%$ and $3 . 9 2 \%$ over SFT and DPO, respectively. To further investigate the performance at each iteration compared to the baseline, we use GPT-4 to annotate the preference for each iteration and present in Table 2. The results suggest that the best performance can be achieved at the second iteration and improved by over $20 \%$ . It is noteworthy that the performance improvement does not rely on increasing response length, which indicates that our method can empower the output quality of LLM instead of outputting long text. For the benchmark of MT-Bench, we also use GPT-4 to annotate the average score of eight aspects and the results in Table 1 show that our method can obtain the highest score and improve the LLM policy from $6 . 7 9 \%$ to $7 . 0 2 \%$ .

In addition, by comparing the performance of UPOMerge with DPO and UPO, we can obtain the following suggestions: 1) the result of UPO-Merge is lower than UPO, which means that iterative evolution is more effective than single turn even though post-train with the same number of preference data, and 2) expending the preference data by self-generation manner can substantially enhance the LLM policy on universal NLP ability.

# Mathematics Reasoning

Apart from the universal generation, we also choose two widely-used GSM8K (Cobbe et al. 2021) and MATH (Hendrycks et al. 2021) to show the versatility of UPO on complex reasoning benchmarks. GSM8K consists of $8 . 5 \mathrm { K }$ high-quality linguistically diverse grade school math word problems and requires the LLM policy to multi-step reasoning capability, while MATH aims at featuring challenging competition math problems.

Table 2: Main results derived from GPT-4 auto evaluation (LC weighted win rate $\%$ ) of different iterations model from UPO over AlpacaEval 2.0 head-to-head comparison with responses of Zephyr-7B-SFT.   

<html><body><table><tr><td>Models</td><td>Align</td><td>Win Rate</td><td>Avg. Length</td></tr><tr><td>Zephyr-7B-SFT</td><td>no</td><td>50.00</td><td>1014</td></tr><tr><td>Zephyr-7B-DPO</td><td>yes</td><td>66.40</td><td>1298</td></tr><tr><td>Zephyr-7B-UPO-Iterl</td><td>yes</td><td>69.94</td><td>967</td></tr><tr><td>Zephyr-7B-UPO-Iter2</td><td>yes</td><td>71.53</td><td>1148</td></tr><tr><td>Zephyr-7B-UPO-Iter3</td><td>yes</td><td>70.21</td><td>1162</td></tr><tr><td>Zephyr-7B-UPO-Merge</td><td>yes</td><td>70.39</td><td>1200</td></tr></table></body></html>

Table 3: Main results (accuracy $\%$ ) on GSM8K and MATH benchmarks. † is trained by (Lai et al. 2024).   

<html><body><table><tr><td>Models</td><td>Align</td><td>GSM8K</td><td>MATH</td></tr><tr><td>GPT-4o-0513</td><td>yes</td><td>95.8</td><td>76.6</td></tr><tr><td>Claude-3-Opus Gemini-1.5-Pro (May)</td><td>yes yes</td><td>95.0 90.8</td><td>60.1 67.7</td></tr><tr><td>Qwen2-7B-Instruct</td><td>yes</td><td>82.3</td><td>49.6</td></tr><tr><td>Qwen2-7B-SFT†</td><td>no</td><td>88.2</td><td>54.8</td></tr><tr><td>Qwen2-7B-DPO</td><td>yes</td><td>88.3</td><td>55.0</td></tr><tr><td>Qwen2-7B-StepDPO+</td><td>yes</td><td>88.5</td><td>55.8</td></tr><tr><td>Qwen2-7B-UPO-Iter1</td><td>yes</td><td>88.5</td><td>55.4</td></tr><tr><td>Qwen2-7B-UPO-Iter2</td><td>yes</td><td>88.6</td><td>55.7</td></tr><tr><td>Qwen2-7B-UPO-Iter3</td><td>yes</td><td>88.4</td><td>55.6</td></tr><tr><td>Qwen2-7B-UPO-Merge</td><td>yes</td><td>88.4</td><td>55.6</td></tr><tr><td>Qwen2-7B-StepUPO-Iter1</td><td></td><td></td><td></td></tr><tr><td>Qwen2-7B-StepUPO-Iter2</td><td>yes</td><td>88.8</td><td>56.0</td></tr><tr><td></td><td>yes</td><td>88.9</td><td>56.3</td></tr><tr><td>Qwen2-7B-StepUPO-Iter3</td><td>yes</td><td>88.8</td><td>56.1</td></tr><tr><td>Qwen2-7B-StepUPO-Merge</td><td>yes</td><td>88.8</td><td>56.2</td></tr></table></body></html>

For the implementation, we choose MathInstruct (Yue et al. 2024) as the prompt set which focuses on the hybrid use of chain-of-thought (CoT) and program-of-thought (PoT) rationales. It contains 262K prompts that are compiled from 13 math rationale datasets. We remove GSM8K and MATH from it to prevent the data leak problem. We follow Lai et al. (2024) to use the technique of StepDPO to tune the LLM policy and the well-constructed fine-grained feedback data is Math-Step-DPO-10K which involves 10.8K prompts with both coarse-grained and fine-grained annotation towards the answers. We select $_ { \mathrm { Q w e n 2 - 7 B - S F T } }$ and Qwen2-7B-SFT-Step-DPO as our basic backbones and the initial LLM policy ⇡✓(0), respectively. The model trained based on our framework with DPO and StepDPO paradigms are respectively named as UPO and StepUPO. During the iteration, we do not filter the noisy data by directly matching the ground truth of each reasoning step or the final answer. In other words, we only leverage the uncertainty estimator to verify the reliable of each reasoning step, aiming to simulate the real scenario that solves the unseen question. More details of these benchmarks and training setups are shown in Appendix D.

Table 4: Ablation study at the first iteration over AlpacaEval 2.0 (LC weighted win rate $\%$ compared with GPT-4), MTBench (absolute score), GSM8K (accuracy $\%$ ) and MATH (accuracy $\%$ ).   

<html><body><table><tr><td>Models</td><td>AlpacaEval2.0 Zephyr-7B</td><td>MT-bench GSM8K</td><td>MATH Qwen2-7B</td></tr><tr><td>SFT</td><td>5.84</td><td>6.18 88.2</td><td>54.8</td></tr><tr><td>DPO／StepDPO</td><td>9.12</td><td>6.79</td><td>88.5 55.8</td></tr><tr><td>UPO ／StepUPO</td><td>13.04</td><td>7.02</td><td>88.9 56.3</td></tr><tr><td>w/o.Rule</td><td>13.01</td><td>7.01</td><td>88.8 56.1</td></tr><tr><td>w/o.Estimator</td><td>10.84</td><td>6.52</td><td>87.1 54.7</td></tr><tr><td>w/o.Weight α</td><td>12.70</td><td>6.94</td><td>88.0 55.8</td></tr><tr><td>W/o.NLL loss</td><td>12.39</td><td>6.92</td><td>87.9 55.7</td></tr></table></body></html>

Training Loss AlpacaEval2.0 0.8 DPO 12 UPO-iter1 UPO-iter2 R 6 0.6 UPO-iter3 10 ub UPO 0.4 8 UPO w/o.Weight α UPO W/o. NLL LOSS 6 UPO w/o.Estimator 0 500 1000 1500 SFT DPO UPO-iter1 iter2 iter3

Main Results The results are listed in Table 3 and we can obtain the following suggestions: 1) The LLM policy post-trained by DPO makes a marginal improvement, increasing from $8 8 . 2 \%$ and $5 4 . 8 \%$ to $8 8 . 3 \%$ and $5 5 . 0 \%$ , respectively. Yet, the improvement of StepDPO can achieve an obvious gain compared with the SFT model, indicating that LLM policy self-evolution can be better conducted with fine-grained feedback. 2) For each iteration, UPO and StepUPO can consistently achieve substantial improvements on GSM8K and MATH, respectively resulting in $8 8 . 9 \%$ and $56 . 3 \%$ accuracy metrics. 3) The result of UPO-Merge and StepUPO-Merge is similar to the performance at the third iteration, which conflicts with the findings in universal NLP tasks. We analyze that the task of mathematics reasoning highly relies on the cleaned preference data, yet the preference data after uncertainty estimation may still contain noisy fine-grained feedback and affect the performance inevitably.

# Further Analysis

# Ablation Study

To investigate the impact of different techniques used in UPO, we conduct the ablation study on all benchmarks to see the performance of different variants. Specifically, for benchmarks of AlpacaEval 2.0 and MT-Bench, we choose DPO as the main baseline and optimization paradigm, while the StepDPO paradigm will be used in GSM8K and MATH. We conduct the experiments at the first iteration. For the variants, w/o. Rule means directly choosing all permutations without any pre-screen processing. w/o. Estimator denotes that do not use uncertainty estimation and choose all generated preference data to train the LLM policy, which is the same as vanilla iterative preference optimization proposed by (Pang et al. 2024). w/o. Weight $\alpha$ represents only training the LLM policy on DPO or StepDPO without smoothing (i.e., $\alpha = 0$ ). w/o. NLL loss means removing the NLL loss by setting $\lambda \ = \ 0$ . Results demonstrated in Table 4 show that the performance will drop if the framework module is removed. Moreover, the use of robust techniques (i.e., uncertainty-enhanced weighting and the NLL loss) consistently contributes to the robustness improvement when training on pseudo preference data.

![](images/646b59ac4c337fdf97cbff87eace814f99334cf6446842cc12c5f447ef8bc4e2.jpg)  
Figure 3: The curve of training loss and LC win rate $( \% )$ on AlpacaEval 2.0 at each iteration.   
Figure 4: Performance of different iterations of UPO compared with SFT and DPO over MT-Bench.

# Effect of Uncertainty-Enhanced Self-evolution

We also explore how the Uncertainty-Enhanced Selfevolution algorithm empowers the LLM policy in the iteration preference optimization procedure. To ask this question, we choose the benchmarks of AlpacaEval 2.0 and MTBench to make a deep-seek. We first draw a training loss curve at the initial stage (DPO training) and each iteration in UPO when preference optimizing on UltraFeedback and newly generated preference data sampled from UltraChat200K. The curve presented in Figure 3 (left) demonstrates that iterative procedure advances the convergence which may contribute to the high performance.

To see the performance changes in different training stages, we also draw a curve to show the win rate increasing in Figure 3 (right) with multiple variants. The result suggests that UPO can substantially outperform vanilla preference optimization (e.g., DPO) in all iteration stages. It is worth noting that variant UPO w/o. Estimator has a bit of improvement compared to the DPO, indicating that many noisy pseudo-preference examples are used in the next iteration and make the iteration training useless. This finding reflects that considering noisy reduction and robustness in iteration preference optimization is significantly necessary.

# Capability Across Different Aspects in MT-Bench

To show the performance of the LLM policy tuned by the UPO framework, we perform task-wise deep analysis on MT-Bench and show the capability of eight aspects in Figure 4, including writing, roleplay, reasoning, math, coding, extracting, STEM, and humanities. Results show that UPO consistently enhances the generation of LLM policy on different aspects of basic problems. Notably, UPO can also realize an obvious improvement in complex tasks, such as reasoning, math, and coding.

![](images/4a38e8e29d802cf81d39b09eef28f463382bdab8de2a2ec4c63c001a2b0bc3d9.jpg)  
Figure 5: Noise rate $( \% )$ of different sampling strategies over multiple manual evaluation sets.

# Noisy Data Study

We end this section by investigating how the UPO framework realizes denoising during iteration preference optimization. We respectively sample 200 preference data from the validation set of UltraFeedback, AlpacaEval 2.0, and MATH-Step-DPO-10K to manually construct the evaluation set. In particular, for preference data from UltraFeedback and MATH-Step-DPO-10K, we directly use the label (which response is better) as the ground truth. For AlpacaEval 2.0, we use the reference generated from GPT-4 as the preferred response, while the dispreferred response is created by the SFT model. At each iteration, we present four different reliable data sampling strategies to select preference data to train the LLM policy after the rewarding process. 1) “Random” denotes randomly selecting from pseudo preference data; 2) “CB-RR” means Chosen response with Best reward and Rejected response with Random select from the rest lower reward, which is a similar strategy to UltraFeedback. 3) “Margin” denotes choosing only one preference data whose reward margin between chosen and rejected is the largest. 4) “Uncertainty” is our proposed method that uses the certainty weight to perform sampling.

Results demonstrated in Figure 5 indicate that considering the reward of the chosen response or reward margin is certainly effective to denoising, which has also been proven in some previous work (Pang et al. 2024). In addition, the results also showcase that leveraging uncertainty estimation can better reduce the noise rate by more than $20 \%$ , $10 \%$ , and $3 \%$ , respectively, indicating the effectiveness of UPO.

# Related Works Preference Optimization of LLMs

Large language models (LLMs), after undergoing extensive pre-training, may generate fabricated facts, biased content, or harmful text. To align these models with human values, fine-tuning language models to adhere to human preferences is an effective solution. Reinforcement Learning from Human Feedback (RLHF) (Stiennon et al. 2020; Ziegler et al. 2019) has emerged as a groundbreaking technique for aligning LLMs. By training a reward model on human feedback data and using Proximal Policy Optimization (PPO) (Schulman et al. 2017) to obtain the policy model for language generation, this approach has led to the development of powerful models such as GPT-4 (Achiam et al. 2023), Llama3 (Dubey et al. 2024), and Gemini (Team et al. 2023). Other methodologies such as DPO (Rafailov et al. 2024) and RRHF (Yuan et al. 2023), optimize language models directly on human feedback datasets. Nevertheless, to further improve performance, it becomes essential to conduct sampling using the model itself, necessitating the incorporation of an auxiliary reward model (RM) (Liu et al. 2023; Song et al. 2024; Wang et al. 2024a; Dong et al. 2023a).

# Iterative Preference Optimization

The optimization of preference datasets and preference models plays a significant role in the alignment of LLMs. Some works (Dong et al. 2023b; Wang et al. 2024b; Rame et al. 2024) employ fine-grained reward objectives and iteratively fine-tune large models for alignment. For example, IRPO (Pang et al. 2024), utilizes iterative DPO for optimization.(Yuan et al. 2024) directly explores a novel Self-Rewarding method for LLMs, which achieve selfimprovement by generating their rewards during training. (Fisch et al. 2024) proposes a reward model distillation algorithm to address the effectiveness and robustness in preference optimization. Similar to these works, we also focus on how to iteratively enhance the effectiveness of preferences and address the noise in the preference predictions by the reward model, aiming to improve the overall robustness of the alignment process.

# Conclusion

We propose an uncertainty-enhanced preference optimization framework to further boost the abilities of the selfevolution of LLMs. We develop an estimator model and let it cooperate with the reward model to provide high-quality preference data at each iteration stage. To reach this goal, we leverage the MC Dropout technique in BNN to perform uncertainty estimation, eliminating the potentially noisy data derived from the weak LLM policy. In addition, we also propose an uncertainty-enhanced self-evolution algorithm to improve the robustness of LLM when repeatedly updating parameters via DPO. We conduct extensive experiments on multiple universal NLP and mathematics reasoning tasks and the results indicate the effectiveness of our method. In the future, we aim to further improve the overall performance and adapt the framework to PPO and other LLMs.