# Behavior Preference Regression for Offline Reinforcement Learning

Padmanaba Srinivasan, William Knottenbelt

Department of Computing, Imperial College London ps3416, wjk @imperial.ac.uk

# Abstract

Offline reinforcement learning (RL) methods aim to learn optimal policies with access only to trajectories in a fixed dataset. Policy constraint methods formulate policy learning as an optimization problem that balances maximizing reward with minimizing deviation from the behavior policy. Closed form solutions to this problem can be derived as weighted behavioral cloning objectives that, in theory, must compute an intractable partition function. Reinforcement learning has gained popularity in language modeling to align models with human preferences; some recent works consider paired completions that are ranked by a preference model following which the likelihood of the preferred completion is directly increased. We adapt this approach of paired comparison. By reformulating the paired-sample optimization problem, we fit the maximum-mode of the Q function while maximizing behavioral consistency of policy actions. This yields our algorithm, Behavior Preference Regression for offline RL (BPR). We empirically evaluate BPR on the widely used D4RL Locomotion and Antmaze datasets, as well as the more challenging V-D4RL suite, which operates in image-based state spaces. BPR demonstrates state-of-the-art performance over all domains. Our on-policy experiments suggest that BPR takes advantage of the stability of on-policy value functions with minimal perceptible performance degradation on Locomotion datasets.

# Introduction

As reinforcement learning (RL) sees increasing application in a variety of fields, from control (Razzaghi et al. 2022) to language modeling (Christiano et al. 2017), it has also become increasingly data-hungry (Shalev-Shwartz, Shamir, and Shammah 2017). The need to acquire data through online interaction can make deep reinforcement learning infeasible in many domains. In response, one direction of research develops offline RL algorithms that aim to learn from a static dataset of pre-collected interactions (Lange, Gabel, and Riedmiller 2012).

Standard off-policy algorithms can be directly applied on offline datasets (Haarnoja et al. 2018; Gulcehre et al. 2020), though in practice the combined effect of off-policy learning, bootstrapping, and function approximation (Sutton and Barto 2018) introduces extrapolation error. The resulting distribution shift between the learned policy and behavior policy can cause training instability and subsequent failure when deployed in the real environment (Fujimoto, Meger, and Precup 2019).

Offline RL algorithms address the challenges of offline off-policy evaluation in one of three ways: 1) incorporating pessimism into value estimation, 2) imposing policy constraints or 3) avoiding off-policy evaluation altogether by learning an on-policy value function. Pessimism offers performance guarantees (Jin, Yang, and Wang 2021), policy constraints may make better use of the representational power of neural networks (Geng et al. 2022) and learning on-policy values is more stable and avoids the overestimation and iterative exploitation associated with off-policy evaluation (Brandfonbrener et al. 2021).

Another approach to learning aims to align policy rollouts with human preferences (Akrour, Schoenauer, and Sebag 2011; Cheng et al. 2011; Christiano et al. 2017). Preferencebased RL is popular in language modeling under the banner of RL from human feedback. Recent methods take models trained using supervised learning and finetune them using an offline dataset by directly increasing the likelihood of generating preferred sequences (Rafailov et al. 2024).

The principle of aligning policies with human preferences has been explored in offline RL (Kim et al. 2023; Rafailov et al. 2024; Hejna et al. 2023). While they aim to solve the same tasks, preference-based methods must either directly learn to generate aligned sequences (Kim et al. 2023) or must train a preference model (Rafailov et al. 2024; Hejna et al. 2023) on specially crafted datasets of human preferences. These methods typically eschew more traditional reward modeling (RM) and perform in-sample learning using pairs of trajectories.

Contributions Motivated by finetuning approaches to align language models (Rafailov et al. 2024; Gao et al. 2024), in this work we develop a policy objective for offline RL that directly learns the policy density: our algorithm performs Behavior Preference Regression for offline RL (BPR). We analyze BPR with respect to regularized value functions in the context of preference models to demonstrate theoretical performance improvement. Evaluation on D4RL (Fu et al. 2020) demonstrates that BPR achieves SOTA performance on Locomotion and Antmaze datasets. Additional tests on the image-based V-D4RL (Lu et al. 2022) tasks reveal that BPR is able to transition across modalities to achieve high performance in non-proprioceptive domains. In experiments with on-policy value functions, BPR outperforms competing methods by a substantial margin on four of six datasets. By incorporating more expressive ensembles of value functions, BPR improves performance substantially on tasks that typically require trajectory stitching.

# Related Work

Reinforcement learning aims to solve sequential decisionmaking tasks formulated as a Markov Decision Process (MDP), $\mathcal { M } \ = \ \{ S , \mathcal { A } , \mathcal { R } , P , p _ { 0 } , \gamma \}$ , where $s$ denotes the state space, $\mathcal { A }$ the action space, $\mathcal { R }$ a scalar reward function, $P$ the transition dynamics, $p _ { 0 }$ the initial state distribution, and $\gamma \in [ 0 , 1 )$ the discount factor. The goal of RL is to learn an optimal policy that executes actions such that it maximizes the expected discounted reward; for any policy $\pi$ we denote its return as $\begin{array} { r } { \eta ( \pi ) = \mathbb { E } _ { \tau \sim \rho _ { \pi } ( \tau ) } \left[ \sum _ { t = 0 } ^ { T } \dot { \gamma } ^ { t } \mathcal { R } ( s _ { t } , a _ { t } ) \right] } \end{array}$ where $\begin{array} { r } { \rho _ { \pi } ( \tau ) = p _ { 0 } ( s _ { 0 } ) \prod _ { t = 1 } ^ { T } \pi ( a _ { t } \vert s _ { t } ) P ( s _ { t + 1 } \vert s _ { t } , a _ { t } ) } \end{array}$ is a trajectory sampled under  policy $\pi$ (Sutton and Barto 2018).

# Offline Reinforcement Learning

Offline RL methods aim to maximize sample efficiency and learn optimal policies given only a static dataset of interactions $\dot { \mathcal { D } } = \{ s , a , r , s \} _ { n = 1 } ^ { \overline { { N } } }$ , which was produced by one or more unknown behavior policies of uncertain quality.

The tuples that form the dataset contain information that we are certain about. Actions beyond the support of the dataset are of unknown quality and lead to unknown trajectories. Generally, offline RL methods aim to train policies that maximize expected reward while remaining within the dataset support.

Off-Policy Methods A large body of offline RL methods adapt existing off-policy algorithms for the offline domain. Approaches can be classified into those that apply critic regularization to address overestimation and those that impose policy constraints to draw the current policy towards the dataset support.

Critic regularizers can explicitly reduce the values of OOD actions (Kumar et al. 2020; Kostrikov et al. 2021), thus shaping the Q function. This forces the policy to maximize Q values that are in-support. Regularizers can function implicitly by making use of the diversity-based-pessimism of large ensembles of value functions (An et al. 2021; Ghasemipour, Gu, and Nachum 2022). Ensembles condone some degree of OOD action selection which An et al. (2021) attribute to improving performance. Fu, Wu, and Boulet (2022) explore this further and find that relaxing constraints can improve performance in algorithms without large ensembles. Work by Ghasemipour, Gu, and Nachum (2022) suggests that large min-clipped ensembles may be redundant due to the collapse in independence of ensemble members.

Policy constraints aim to directly confine the actor to select in-support actions. These are typically formulated explicitly as divergence penalties (Wu, Tucker, and Nachum 2019; Fujimoto and Gu 2021), implicitly through weighted behavioral cloning (BC) (Wang et al. 2020; Peng et al. 2019; Nair et al. 2020) or by architecturally limiting the exploration afforded to the policy (Fujimoto, Meger, and Precup 2019).

On-Policy Methods Brandfonbrener et al. (2021) recognize off-policy evaluation as a source of instability in offline RL and instead learn an on-policy (Onestep) value function. The policy learned using this value function outperforms those learned via behavioral cloning and some offline off-policy methods. On-policy learning is extended by Kostrikov, Nair, and Levine (2021) and Garg et al. (2023) who attempt to approximate the in-sample maximum return by dataset trajectories which they use to train a weighted BC policy. Zhuang et al. (2023) adapt online, on-policy PPO (Schulman et al. 2017) for the offline setting and develop an algorithm that uses offline datasets with periodic online evaluation. This is not a fully offline RL algorithm and their own experiments show that without online evaluation to enable policy replacement, performance will degrade.

# Preference-Based Reinforcement Learning

Building on the ideas of Akrour, Schoenauer, and Sebag (2011) and Cheng et al. (2011), Christiano et al. (2017) suggest using preference-annotated data as reward signals to train language models that are better aligned with human values. Subsequent work has developed learning from preferences further (Kaufmann et al. 2023) with the notable $D i$ - rect Preference Optimization (DPO) (Rafailov et al. 2024) which finetunes a maximum likelihood trained policy on an offline dataset of paired preference annotated data by directly optimizing policy density as a proxy for the reward function.

In continuous-control offline RL, Kim et al. (2023) train a trajectory-producing policy on non-Markovian, preferencebased rewards. An et al. (2023) use a preference labeled dataset to train a preference model that is subsequently used to label preferred trajectories in an unlabeled dataset used for policy training. Using preference datasets, Hejna et al. (2023) directly train a policy (similar to DPO) as an optimal advantage function using preference data. Common themes of preference-based offline RL methods are the eschewing of traditional rewards for human-annotated data, and the requirement trajectories to be paired for preference learning which does not allow evaluation of OOD actions.

# Behavior Preference Regression

We consider the general, reverse KL-constrained problem:

$$
\begin{array} { r l } & { \pi _ { t + 1 } = \underset { \pi \in \Pi } { \arg \operatorname* { m a x } } \mathbb { E } _ { s \sim \mathcal { D } , a \sim \pi } [ f ( s , a ) } \\ & { \qquad - \left. \lambda D _ { \mathrm { K L } } ( \pi ( \cdot | s ) | | \pi _ { \mathrm { r e f } } ( \cdot | s ) ) \right] , } \end{array}
$$

where $\lambda \geq 0$ controls the tradeoff between remaining close to a distribution $\pi _ { \mathrm { r e f } }$ and maximizing some function $f ( \cdot , \cdot )$ .

The closed form solution to the optimization problem has been previously derived (Ziebart et al. 2008; Gru¨nwald and

Dawid 2004):

$$
\begin{array} { r l r } & { } & { \pi _ { t + 1 } = \pi _ { \mathrm { r e f } } ( a | s ) \exp ( \displaystyle \frac { 1 } { \lambda } f ( s , a ) ) \frac { 1 } { Z ( s ) } } \\ & { } & { Z ( s ) = \displaystyle \int _ { a \in \mathcal { A } } \pi _ { \mathrm { r e f } } ( a | s ) \exp ( \displaystyle \frac { 1 } { \lambda } f ( s , a ) ) d a , } \end{array}
$$

where $Z ( s )$ is the partition function.

Using the DPO trick (Rafailov et al. 2024), we can rearrange Equation 2 as:

$$
f ( s , a ) = \lambda \left( \log Z ( s ) + \log \frac { \pi _ { t + 1 } ( a | s ) } { \pi _ { \mathrm { r e f } } ( a | s ) } \right) ,
$$

following which using ranked, paired samples where $a _ { 1 } \succ$ $a _ { 2 }$ we can write:

$$
\begin{array} { r l } & { f ( s , a _ { 1 } ) - f ( s , a _ { 2 } ) = } \\ & { \quad \lambda \left( \log \frac { \pi _ { t + 1 } \left( a _ { 1 } | s \right) } { \pi _ { \mathrm { r e f } } \left( a _ { 1 } | s \right) } - \log \frac { \pi _ { t + 1 } \left( a _ { 2 } | s \right) } { \pi _ { \mathrm { r e f } } \left( a _ { 2 } | s \right) } \right) , } \end{array}
$$

which conveniently cancels out the partition function.

DPO takes the binary preference $a _ { 1 } ~ \succ ~ a _ { 2 }$ and passes the RHS through a Bradley-Terry preference model (Bradley and Terry 1952) to optimize for $a _ { 1 }$ . Consequently, DPO fails to capture how much more $a _ { 1 }$ is preferred to $a _ { 2 }$ . Gao et al. (2024) aim to directly learn the relative difference by solving the regression problem:

$$
\begin{array} { r l } & { \big [ ( f ( s , a _ { 1 } ) - f ( s , a _ { 2 } ) ) - } \\ & { \quad \lambda \left( \log \frac { \pi _ { t + 1 } ( a _ { 1 } | s ) } { \pi _ { \mathrm { r e f } } ( a _ { 1 } | s ) } - \log \frac { \pi _ { t + 1 } ( a _ { 2 } | s ) } { \pi _ { \mathrm { r e f } } ( a _ { 2 } | s ) } \right) \big ] ^ { 2 } . } \end{array}
$$

In this work, we focus on learning a policy by solving this relative regression problem.

# What do we Prefer in Offline RL?

Most policy constraint formulations typically choose $f ( s , \cdot ) \stackrel { - } { = } Q \dot { ( } s , \cdot )$ and $\pi _ { \mathrm { r e f } } ( \cdot | s ) = \hat { \pi } _ { \beta } ( \cdot | s )$ where $\hat { \pi } _ { \beta }$ is an empirical behavior policy. This follows the principle of maximizing reward while satisfying some constraint that must be carefully balanced by tuning $\lambda$ to curb the distribution shift (Brandfonbrener et al. 2021).

We propose an alternative optimization: we maximize behavioral consistency and reverse KL fit the (maximum) mode of the Q function – in preference terms, we fit a distribution of high-reward actions and regress toward actions with high likelihood under the behavior policy.

Selecting $\pi _ { \mathbf { r e f } }$ Soft Q-learning (Haarnoja et al. 2018) trains a maximum entropy Q function that can be written as an energy-based model (EBM) (Goodfellow, Bengio, and Courville 2016). We formulate πref(a|s) = expZ(Q(s,)a)) where $\begin{array} { r } { Z _ { Q } ( s ) = \int _ { \mathcal { A } } Q ( s , a ) d a } \end{array}$ is the partition function, which subsequently cancels out in the RHS of Equation 6. This allows us to directly optimize the soft actor–critic (SAC) policy objective (Haarnoja et al. 2018) without resorting to approximations of the entropy through a tanhtransformed Gaussian.

Selecting $f ( \cdot , \cdot )$ The true behavior policy is unknown and so we must make an empirical approximation. Prior methods typically learn explicit policies using behavioral cloning (Kostrikov et al. 2021; Wu, Tucker, and Nachum 2019; Zhuang et al. 2023). This can be limiting, as the number of behavior policy modes must be known beforehand. Implicit policies offer more flexible behavior models (Florence et al. 2022). We train an implicit behavior policy $\hat { \pi } _ { \beta }$ as an EBM that learns an energy function $E ( s , a ) \in \mathbb { R }$ . We recover an estimate of the explicit behavior policy using the Boltzmann distribution: $\begin{array} { r } { \hat { \pi } _ { \beta } \overset { - } { ( } a | s ) = \frac { \exp ( - \bar { E ( s , a ) ) } } { Z _ { E } ( s ) } } \end{array}$ where $\begin{array} { r } { Z _ { E } ( s ) = \int _ { A } \exp ( - E ( s , a ) ) d a } \end{array}$ is the EBM partition function.

Fortunately, $Z _ { E } ( s )$ also cancels out when using $f ( s , \cdot ) =$ $\log \hat { \pi } _ { \beta } ( \cdot | s )$ in the LHS of Equation 6 and we only need to compute $E ( s , a _ { 1 } )$ and $E ( s , a _ { 2 } )$ . Using an EBM behavior policy, we make no inductive bias with respect to the (multi)modality of the true behavior policy.

Combining everything, our policy optimization objective is:

$$
\begin{array} { r l r } {  { [ ( E ( s , a _ { 2 } ) ) - E ( s , a _ { 1 } ) ) - } } \\ & { } & { \quad \lambda ( \log \frac { \pi _ { t + 1 } ( a _ { 1 } | s ) } { \exp ( Q ( s , a _ { 1 } ) ) } - \log \frac { \pi _ { t + 1 } ( a _ { 2 } | s ) } { \exp ( Q ( s , a _ { 2 } ) ) } ) ] ^ { 2 } . } \end{array}
$$

Interpretation Learning $\pi ^ { * }$ requires a policy to select in-sample actions that also maximize expected reward. By selecting the regression target to be the difference $\log \hat { \pi } _ { \beta } ( a _ { 1 } | s ) - \log \hat { \pi } _ { \beta } ( a _ { 2 } | s )$ , we treat the behavior EBM as an expert preference model that communicates by how much $a _ { 1 } \ \succ \ a _ { 2 }$ . This differs from previous preference-based offline RL formulations that evaluate the preference by comparing discounted rewards over entire trajectories (produced by the behavior policy) for a pair of actions. Such rewardbased preference learning has been shown to be inconsistent with human-preference labels (Knox et al. 2022). Placing a support constraint on the policy towards high-reward modes in the soft Q function and combining this with offpolicy evaluation offers a far more flexible approach without the need for human-labeled preference datasets. Most importantly, we never need to compute any partition function $Z ( s ) { \dot { , } } Z _ { Q } ( s )$ or $Z _ { E } ( s )$ – past work has found that approximating partition functions, though technically correct, is deleterious to performance (Nair et al. 2020).

# Self-Play

Let $\mu _ { 1 } , \mu _ { 2 }$ be the sampling distributions for $a _ { 1 }$ and $a _ { 2 }$ , respectively. Offline preference-based methods use datasets that contain previously evaluated pairs of completions sampled from $\pi _ { \beta }$ (Kim et al. 2023; Rafailov et al. 2024; Hejna et al. 2023). In standard offline settings, samples are drawn from $\mathcal { D }$ or $\pi$ and in the paired setting this equates to using $\mu _ { 1 } = \pi _ { \beta } = \mathcal { D }$ and $\mu _ { 2 } = \pi$ (reference sampling). Recently, Swamy et al. (2024) prove that performing self-play with multiple samples drawn from $\pi$ itself results in stable learning with strong theoretical guarantees – this involves sampling a pair of actions from the current policy and querying a learned preference/reward model to optimize Equation 2. We use self-play to sample actions for policy optimization, hence $\mu _ { 1 } = \mu _ { 2 } = \pi$ . We compare reference sampling and self-play schemes in a toy bandit example in the Appendix.

# Analysis

Rearranging Equation 5 and inserting πref(·|s) = expZ(Q(s),·)) and $f ( s , \cdot ) = \log \pi _ { \beta } ( \cdot | s )$ , we obtain:

$$
\begin{array} { c l } { \displaystyle { \left( Q ( s , a _ { 1 } ) + \frac { 1 } { \lambda } \log \pi _ { \beta } ( a _ { 1 } | s ) \right) - } } \\ { \displaystyle { \left( Q ( s , a _ { 2 } ) + \frac { 1 } { \lambda } \log \pi _ { \beta } ( a _ { 2 } | s ) \right) } } \\ { \displaystyle { = \log \pi _ { t + 1 } ( a _ { 1 } | s ) - \log \pi _ { t + 1 } ( a _ { 2 } | s ) . } } \end{array}
$$

We explicitly cancel $Z _ { Q } ( s )$ but leave $Z _ { E } ( s )$ unfactorized for clarity.

We define $\begin{array} { r } { \tilde { Q } ( s , a ) \triangleq Q ( s , a ) + \frac { 1 } { \lambda } \log \hat { \pi } _ { \beta } ( a | s ) } \end{array}$ and notice that this is a variation of an implicit Q function popular in online RL (Vieillard et al. 2021; Peters, Mulling, and Altun 2010) and is exactly the Q function formulation used by Fisher-BRC when using $\lambda = 1 . 0$ (Kostrikov et al. 2021). We subsequently interpret that our policy regression objective is equivalent to fitting the policy to the implicit Q function.

We rewrite the LHS of Equation 8 as a soft preference function:

$$
\mathrm { P } \left( s , a _ { 1 } , a _ { 2 } \right) \triangleq \tilde { Q } ( s , a _ { 1 } ) - \tilde { Q } ( s , a _ { 2 } ) .
$$

Assumption 1 (Tuned Preference Function)

$$
\begin{array} { r } { P ( s , a _ { 1 } , a _ { 2 } ) \geq 0 \quad \forall a _ { 1 } , a _ { 2 } \in \mathcal { A } } \\ { w h e n \quad \pi _ { \beta } ( a _ { 1 } | s ) \geq \pi _ { \beta } ( a _ { 2 } | s ) . } \end{array}
$$

We assume that any action $a _ { 1 }$ with a higher likelihood under the behavior policy than $a _ { 2 }$ is preferred. In practice, this can be satisfied by tuning $\lambda$ .

For any policy $\pi$ , recall its return is given by $\eta ( \pi ) ~ =$ $\begin{array} { r } { \mathbb { E } _ { \tau \sim \rho _ { \pi } ( \tau ) } \left[ \sum _ { t = 0 } ^ { T } \gamma ^ { t } \mathcal { R } ( s _ { t } , a _ { t } ) \right] . } \end{array}$ The behavior policy used to produce the dataset is $\pi _ { \beta }$ and let the policy learned by optimizing using $\tilde { Q } ( s , a )$ be $\tilde { \pi }$ (i.e. the policy that maximizes soft preferences).

Proposition 1 (Perfect Preference Model) If the preference function $P ( s , a _ { 1 } , a _ { 2 } )$ is perfect i.e. $\tilde { Q } ^ { * } = \tilde { Q ^ { * } } + \bar { \pi } _ { \beta }$ is accurate, then the deterministic policies $\pi _ { \beta }$ and $\tilde { \pi }$ satisfy:

$$
\begin{array} { r l } & { \eta ( \tilde { \pi } ) - \eta ( \pi _ { \beta } ) } \\ & { \quad \approx \mathbb { E } _ { s \sim \mathcal { D } } \left[ \tilde { Q } ^ { * } ( s , \tilde { \pi } ( s ) ) - \tilde { Q } ^ { * } ( s , \pi _ { \beta } ( s ) ) \right] \geq 0 } \end{array}
$$

In practice, estimation is noisy. For $\tilde { Q }$ , this comes from two sources: errors are present in both Q function and behavior policy estimates. EBM approximation error has been studied by Florence et al. (2022) (Theorem 2) who prove that a Lipschitz-continuous EBM policy can exhibit arbitrarily small error.

The total variational distance between two value functions $Q _ { 1 } , Q _ { 2 }$ is: $D _ { \mathrm { T V } } ( Q _ { 1 } , Q _ { 2 } ) = \mathrm { m a x } _ { s \in { \cal S } } | Q _ { 1 } ( s , \pi ( s ) ) -$ $Q _ { 2 } ( s , \pi ( s ) ) |$ .

<html><body><table><tr><td>Algorithm 1:Policy improvement step. Comment NG de- notes steps where gradients do not have to be computed.</td></tr><tr><td>Require: Offline dataset D, pretrained EBM E(·, ·), training steps N Output: Trained policy π Let t = 0. for t=1 to N do Sample(s,a,r,s') ~ D Sample α1,α2 ~ π# NG Compute log π(a1|s),log π(a2|s) Compute E(s,a1) and E(s,a2) #NG Compute Q(s,a1) and Q(s,a2) #NG Update T using Equation 7. #Update critics</td></tr></table></body></html>

Proposition 2 (Noisy Preference Model) Consider the case where $\hat { \pi } _ { \beta }$ and $Q ^ { * }$ contain errors and produce the noisy $\tilde { Q } ^ { - }$ . Then $\forall \tilde { Q } ^ { - }$ where $D _ { T V } ( \tilde { Q } ^ { - } ( s , \tilde { \pi } ( s ) ) , Q ^ { * } ( s , \tilde { \pi } ( s ) ) ) \le \tilde { \epsilon }$ and $D _ { T V } ( \tilde { Q } ^ { - } ( s , \pi _ { \beta } ( s ) ) , Q ^ { * } ( s , \pi _ { \beta } ( s ) ) ) \leq \epsilon$ the following holds:

$$
\begin{array} { r l } & { \eta ( \tilde { \pi } ) - \eta ( \pi _ { \beta } ) } \\ & { \qquad \le \mathbb { E } _ { s \sim \mathcal { D } } \left[ \tilde { Q } ^ { - } ( s , \tilde { \pi } ( s ) ) - \tilde { Q } ^ { - } ( s , \pi _ { \beta } ( s ) ) \right] } \\ & { \qquad + 2 \rho _ { m a x } ( \tilde { \epsilon } + \epsilon ) } \\ & { \qquad w h e r e \quad \rho _ { m a x } = \operatorname* { s u p } \{ \rho _ { \pi _ { \beta } } ( s ) , s \in \mathcal { S } \} . } \end{array}
$$

We defer proofs to the Appendix.

The first term after the inequality is non-negative under Assumption 1 and the second term is present due to the modeling error of the estimated Q function and behavior policy. This can be reduced by using a more accurate function approximator.

# Implementation

Our actor–critic implementation follows a standard implementation of SAC (Haarnoja et al. 2018) with modifications to the policy improvement step. We illustrate our policy improvement in Algorithm 1 and provide additional implementation details in the Appendix.

The EBM approximation of $\pi _ { \beta }$ is trained prior to the main actor–critic training phase. We follow design decisions detailed in Florence et al. (2022), using spectral normalization (Miyato et al. 2018) and deep networks.

Summary of Hyperparameters In addition to the standard hyperparameters of SAC (clipped double-Q learning (Fujimoto, Hoof, and Meger 2018), entropy regularized offpolicy Q functions), our algorithm introduces the hyperparameter $\lambda$ , which controls the tradeoff between the KL constraint and maximizing behavioral consistency.

In general, we find that simply using $\lambda = 1 . 0$ works well across all tasks; our primary results use this hyperparameter value and we perform ablations to evaluate sensitivity in our experiments.

# Experiments

In this section, we evaluate empirically BPR and aim to answer the following questions:

• How well does BPR perform compared to state-of-the-art offline RL methods?   
• Does BPR perform well in tasks with visual state spaces?   
• Can Onestep-trained policies compete with off-policy offline RL?   
• How sensitive is BPR to values of $\lambda ?$

Experimental Setup In all BPR experiments, we report the normalized mean score with standard deviation on five seeds over 100 evaluations in Antmaze tasks and 10 in others. All scores are reported using the policy from the final checkpoint.

Baselines We compare results against the following, wellknown baselines: CQL (Kumar et al. 2020), IQL (Kostrikov, Nair, and Levine 2021) and $\mathrm { T D } 3 { + } \mathrm { B C }$ (Fujimoto and Gu 2021). We also include the recent offline RL algorithms: ReBRAC (Tarasov et al. 2023), XQL (Garg et al. 2023) and Diff-QL (Wang, Hunt, and Zhou 2022) (which replaces a Gaussian/deterministic policy with a Diffusion policy (Ho, Jain, and Abbeel 2020)). Of the latter three methods, both ReBRAC and XQL tune hyperparameters extensively for each dataset. In contrast, the older baselines, Diff-QL and our BPR find hyperparameters that generalize well across like-tasks (i.e. the same hyperparameters for all Locomotion tasks etc.).

For a more comprehensive comparison, we also include the preference-based offline RL methods: PT (Kim et al. 2023), OPPO (Kang et al. 2023) and DPPO (An et al. 2023).

# D4RL

We evaluate BPR on D4RL Locomotion and Antmaze datasets (Fu et al. 2020).

Locomotion The Locomotion datasets offer varying degrees of suboptimality, using mixtures of highly suboptimal trajectories $( - \tt r e p l a y )$ and optimal ones $( - \mathrm { e x p e r t } )$ . Table 1 shows BPR’s Locomotion scores. In general, all methods recover near-expert performance on any expert datasets. BPR greatly outscores all older baselines as well as preference-based algorithms. ReBRAC is highly tuned for each dataset and BPR, for the most part, scores similarly except for $\mathtt { h c - m }$ (where ReBRAC scores higher), and $\mathtt { w } \mathrm { - m }$ and $\mathtt { w } \mathrm { - m } \mathrm { - } \mathtt { r }$ , where BPR outperforms ReBRAC by a substantial margin.

Antmaze The Antmaze tasks are characterized by sparse reward schemes and suboptimal trajectories which necessitates off-policy evaluation (or IQL/XQL in-sample max estimation) to perform well. In the smaller mazes, BPR, ReBRAC and XQL perform similarly, though BPR is able to sustain high performance as the maze grows. Preferencebased PT does not perform well in larger mazes.

# V-D4RL

Most offline RL algorithms typically limit their evaluation to proprioceptive state spaces. V-D4RL (Lu et al. 2022) is a benchmarking suite that evaluates offline RL algorithms in visual state spaces on continuous control tasks with mixtures of trajectories similar to those found in D4RL Locomotion and based on the DMC environments (Yarats et al. 2021).

The V-D4RL paper provides scores for CQL, and behavioral cloning (BC) policies, as well as LOMPO (Rafailov et al. 2021) and a variant of DrQ (Yarats et al. 2021) with a behavioral cloning constraint. LOMPO and DrQ are designed specifically to learn from visual state spaces. We also include results for ReBRAC, which is again tuned for each dataset. We use V-D4RL environments without distractors following Tarasov et al. (2023).

We present V-D4RL results in Table 3. Generally, BC outperforms CQL – the standard offline RL baseline. ReBRAC, with the help of tuning, is able to slightly outperform BC. BPR consistently outperforms the image-adapted LOMPO and $\mathrm { D r Q + B C }$ , trading blows with ReBRAC on walker-walk and cheetah-run datasets and keeps pace with BC on the more difficult humanoid-walk tasks.

# Onestep Experiments

Off-policy evaluation can lead to querying and backing up of overestimated OOD actions that the policy can exploit, leading to instability. Onestep value functions are highly stable due to their on-policy nature (Brandfonbrener et al. 2021) and recent work by Eysenbach et al. (2023) shows equivalence between Onestep values and CQL-style critic regularization.

We evaluate how well BPR with a Onestep value function performs compared to the original Onestep RL (O-RL) algorithm (Brandfonbrener et al. 2021). We also include Locomotion results from CFPI (Li et al. 2023), which uses a first-order Taylor approximation as a linear approximation of the Q function, and trains a Onestep value function using distributional critics (Dabney et al. 2018a,b).

We report results on non-expert Locomotion datasets and the medium and large Antmaze datasets in Table 4. Both Onestep RL and CFPI perform similarly on Locomotion tasks. BPR matches their performance on two tasks and outperforms both by a large margin on four out of six Locomotion tasks.

Onestep RL performs poorly on the medium and large Antmaze tasks. In contrast, BPR is able to make significant progress in all these sparse reward tasks, falling slightly short of off-policy CQL (see Table 2).

Suboptimality in D4RL The similarity in performance between Onestep BPR and off-policy BPR in Locomotion tasks suggests that trajectories in these datasets may not be as suboptimal as originally thought (Fu et al. 2020). This explains the recent saturation in performance on Locomotion (Tarasov et al. 2023). Antmaze, while challengingly suboptimal, may be a poor evaluator of generalization (Rafailov et al. 2024). The performance of Onestep BPR indicates that this may be a pragmatic variant to select for application due to its improved stability.

Table 1: Normalized scores on D4RL Gym Locomotion datasets. All scores are taken from their respective original papers. hc, hp and w refer to halfcheetah, hopper and walker2d environments, respectively. Methods are grouped by: older baselines, newer offline RL baselines, preference-based offline RL methods followed by BPR. For XQL, we use the per-dataset tuned variant’s scores. We report SD for BPR and bold the top score and underline BPR scores when within 1 SD of the best.   

<html><body><table><tr><td>Dataset</td><td>CQL</td><td>IQL</td><td>TD3+BC</td><td>ReBRAC</td><td>XQL</td><td>Diff-QL</td><td>PT</td><td>OPPO</td><td>DPPO</td><td>BPR (ours)</td></tr><tr><td>hc-m</td><td>44.0</td><td>47.4</td><td>48.3</td><td>65.6</td><td>48.3</td><td>51.1</td><td></td><td>43.4</td><td></td><td>53.7 ± 1.4</td></tr><tr><td>hp-m</td><td>58.5</td><td>66.3</td><td>59.3</td><td>102.0</td><td>74.2</td><td>90.5</td><td></td><td>86.3</td><td></td><td>101.3 ± 1.1</td></tr><tr><td>w-m</td><td>72.5</td><td>78.3</td><td>83.7</td><td>82.5</td><td>84.2</td><td>87.0</td><td></td><td>85.0</td><td></td><td>91.1 ± 3.7</td></tr><tr><td>hc-m-r</td><td>45.5</td><td>42.2</td><td>44.6</td><td>51.0</td><td>45.2</td><td>47.8</td><td></td><td>39.8</td><td>40.8</td><td>50.9 ± 0.6</td></tr><tr><td>hp-m-r</td><td>95.0</td><td>94.7</td><td>60.9</td><td>98.1</td><td>100.7</td><td>101.3</td><td>84.5</td><td>88.9</td><td>73.2</td><td>102.0 ± 4.9</td></tr><tr><td>w-m-r</td><td>77.2</td><td>73.9</td><td>81.8</td><td>77.3</td><td>82.2</td><td>95.5</td><td>71.3</td><td>71.7</td><td>50.9</td><td>97.4 ± 2.7</td></tr><tr><td>hc-m-e</td><td>91.6</td><td>86.7</td><td>90.7</td><td>101.1</td><td>94.2</td><td>96.8</td><td></td><td>89.6</td><td>92.6</td><td>103.8 ± 4.3</td></tr><tr><td>h-m-e</td><td>105.4</td><td>91.5</td><td>98.0</td><td>107.0</td><td>111.2</td><td>111.1</td><td>69.0</td><td>108.0</td><td>107.2</td><td>110.9 ± 5.2</td></tr><tr><td>w-m-e</td><td>108.8</td><td>109.6</td><td>110.1</td><td>111.6</td><td>112.7</td><td>110.1</td><td>110.1</td><td>105.0</td><td>108.6</td><td>110.8 ± 0.2</td></tr></table></body></html>

Table 2: Normalized scores on D4RL Antmaze datasets. Methods are grouped by: older baselines, newer RM RL baselines, preference-based offline RL methods followed by BPR. For XQL, we use the per-dataset tuned variant’s scores. We report SD for BPR and bold the top score and underline BPR scores when within 1 SD of the best.   

<html><body><table><tr><td>Dataset</td><td>CQL</td><td>IQL</td><td>TD3+BC</td><td>ReBRAC</td><td>XQL</td><td>Diff-QL</td><td>PT</td><td>BPR (ours)</td></tr><tr><td>-umaze</td><td>74.0</td><td>87.5</td><td>78.6</td><td>97.8</td><td>93.8</td><td>93.4</td><td></td><td>95.6 ± 1.0</td></tr><tr><td>-umaze-d</td><td>84.0</td><td>62.2</td><td>71.4</td><td>88.3</td><td>82.0</td><td>66.2</td><td></td><td>89.1 ± 1.1</td></tr><tr><td>-medium-p</td><td>61.2</td><td>71.2</td><td>10.6</td><td>84.0</td><td>76.0</td><td>76.6</td><td>70.1</td><td>86.7 ± 3.7</td></tr><tr><td>-medium-d</td><td>53.7</td><td>70.0</td><td>3.0</td><td>76.3</td><td>73.6</td><td>78.6</td><td>65.3</td><td>82.9 ± 7.8</td></tr><tr><td>-large-p</td><td>15.8</td><td>39.6</td><td>0.2</td><td>60.4</td><td>46.5</td><td>46.4</td><td>42.4</td><td>70.3 ± 8.3</td></tr><tr><td>-large-d</td><td>14.9</td><td>47.5</td><td>0.0</td><td>54.4</td><td>49.0</td><td>56.6</td><td>19.6</td><td>72.1 ± 5.1</td></tr></table></body></html>

<html><body><table><tr><td>Dataset</td><td>BC</td><td>CQL</td><td>ReBRAC</td><td>LOMPO</td><td>DrQ+BC</td><td>BPR (ours)</td></tr><tr><td>ww-mixed</td><td>16.5 ± 4.3</td><td>11.4 ± 12.4</td><td>41.6 ± 8.0</td><td>34.7 ± 19.7</td><td>28.7 ± 6.9</td><td>45.0 ±11.2</td></tr><tr><td>ww-medium</td><td>40.9 ± 3.1</td><td>14.8 ± 16.1</td><td>52.5 ± 3.2</td><td>43.9 ± 11.1</td><td>46.8 ± 2.3</td><td>50.7 ± 4.1</td></tr><tr><td>ww-medexp</td><td>47.7 ± 3.9</td><td>56.4 ± 38.4</td><td>92.7 ± 1.3</td><td>39.2 ± 19.5</td><td>86.4 ± 5.6</td><td>97.4 ± 1.9</td></tr><tr><td>cr-mixed</td><td>25.0 ± 3.6</td><td>10.7 ± 12.8</td><td>46.8 ± 0.7</td><td>36.3 ± 15.6</td><td>44.8 ± 3.6</td><td>45.0 ± 3.1</td></tr><tr><td>cr-medium</td><td>51.6 ± 1.4</td><td>40.9 ± 5.1</td><td>58.3 ± 11.7</td><td>16.4 ± 18.3</td><td>50.6 ± 8.2</td><td>55.3 ± 1.2</td></tr><tr><td>cr-medexp</td><td>57.5 ± 6.3</td><td>20.9 ± 5.5</td><td>58.3 ± 11.7</td><td>11.9 ± 1.9</td><td>50.6 ± 8.2</td><td>62.7 ± 8.5</td></tr><tr><td>hw-mixed</td><td>18.8 ± 4.2</td><td>0.1±0.0</td><td>16.0 ± 2.7</td><td>0.2±0.0</td><td>15.9 ± 3.8</td><td>18.3 ± 1.9</td></tr><tr><td>hw-medium</td><td>13.5 ± 4.1</td><td>0.1 ± 0.0</td><td>9.0 ± 2.3</td><td>0.1±0.0</td><td>6.2 ± 2.4</td><td>9.0±0.8</td></tr><tr><td>hw-medexp</td><td>17.2 ± 4.7</td><td>0.1 ± 0.0</td><td>7.8 ± 2.4</td><td>0.2 ±0.0</td><td>7.0 ± 2.3</td><td>13.3 ± 4.4</td></tr></table></body></html>

Table 3: Normalized scores on V-D4RL tasks. ww, cr and hw refer to walker-walk, cheetah-run and humanoid-walk environments, respectively. Methods are grouped by: BC, offline RL baselines, RL algorithms adapted for visual state spaces followed by BPR. We report 1 SD for all methods and bold the top score and underline BPR scores when within 1 SD of the best.

<html><body><table><tr><td>Dataset</td><td>O-RL</td><td>CFPI</td><td>Onestep BPR</td></tr><tr><td>hc-m</td><td>55.6</td><td>51.1</td><td>52.0 ± 0.8</td></tr><tr><td>hp-m</td><td>83.3</td><td>86.8</td><td>96.4 ± 0.4</td></tr><tr><td>W-m</td><td>85.6</td><td>88.3</td><td>89.7 ± 1.3</td></tr><tr><td>hc-m-r</td><td>41.4</td><td>44.5</td><td>51.0 ± 0.4</td></tr><tr><td>h-m-r</td><td>71.0</td><td>93.6</td><td>99.1 ± 2.3</td></tr><tr><td>w-m-r</td><td>71.6</td><td>78.2</td><td>92.0 ± 0.8</td></tr><tr><td>amaze-m-p</td><td>0.3</td><td>-</td><td>52.7 ± 10.3</td></tr><tr><td>amaze-m-d</td><td>0.0</td><td></td><td>40.0 ± 7.8</td></tr><tr><td>amaze-l-p</td><td>0.0</td><td></td><td>10.4 ± 2.9</td></tr><tr><td>amaze-l-d</td><td>0.0</td><td></td><td>12.7 ± 1.6</td></tr></table></body></html>

Table 4: Scores for Onestep BPR with Onestep RL and Onestep CFPI. We evaluate on non-expert Locomotion and medium and large Antmaze (amaze) datasets. The authors of CFPI do not report Onestep results for Antmaze. We report 1 SD for BPR and bold the top score and underline BPR scores when within 1 SD of the best.

More Expressive Onestep Value Functions O-RL uses a single Q function and samples actions to estimate statevalue to compute advantage. CFPI trains two distributional critics and ses the min-clipped value estimate during bootstrapping. Onestep BPR trains two regular, min-clipped critics. Diversity can collapse in ensembles with shared targets. We investigate whether diversity at the cost of pessimism can improve performance; we experiment with Onestep, independent 4-critic ensembles to estimate the $\mathrm { ~ Q ~ }$ value lower confidence bound (Ghasemipour, Gu, and Nachum 2022):

$$
Q _ { \mathrm { L C B } } ( s , a ) = \mathbb { E } ^ { \mathrm { e n s } } \left[ Q _ { i } ( s , a ) \right] - \omega \mathbb { V } ^ { \mathrm { e n s } } \left[ Q _ { i } ( s , a ) \right] ,
$$

where $\mathbb { E } ^ { \mathrm { e n s } }$ and $\mathbb { V } ^ { \mathrm { e n s } }$ indicate mean and variance over the ensemble of $\mathrm { \Delta Q }$ functions and $\omega$ is a parameter that controls the degree of pessimism. We use $\omega = 2 . 0$ in all experiments.

Compared to Onestep BPR, Ensemble BPR sees performance improvements of at least 10 points on each dataset on the medium and large Antmaze datasets. Detailed perdataset scores and implementation information can be found in the Appendix.

# Ablations

Recall that $\lambda$ controls the tradeoff between maximizing behavioral consistency and fitting the Q function in Equation 1. We examine sensitivity to $\lambda$ for off-policy BPR in a series of ablation experiments in the D4RL Locomotion tasks.

Sensitivity to $\lambda$ varies between datasets, with little performance variation on halfcheetah-medium and halfcheetah-medium-replay. In other datasets, using $\lambda ~ = ~ 0 . 5$ or $\lambda ~ = ~ 2 . 0$ sees performance decline. Our choice of $\lambda \ : = \ : 1 . 0$ generalizes well over all datasets and usually outperforms $\lambda = 1 . 5$ . We provide detailed ablation results in the Appendix.

# Discussion

Performance Our key contribution in this work is the development of a policy objective that reduces policy improvement to a regression problem. Off-policy BPR results in

D4RL Locomotion datasets are on par with current SOTA and BPR outperforms RL baselines in 5 out of 6 Antmaze datasets and 6 out of 9 V-D4RL datasets. Our Onestep experiments show that Onestep BPR outperforms Onestep RL in 9 out of 10 tasks and CFPI in all Locomotion tasks. BPR requires minimal tuning to achieve high performance – all our results are produced using $\lambda = 1 . 0$ .

Density Estimation Employing estimates of the behavior policy is common in many offline RL algorithms. Most prior works use explicit density estimates using Gaussian policies, mixture density networks (Bishop 1994) or VAEs (Kingma and Welling 2013). If the modality of the behavior policy is known, the first two methods can be used in BPR. VAEs are unsuitable as density estimation requires sampling.

The function $f ( \cdot , \cdot )$ does not need to be a density estimate. Another natural choice for $f ( \cdot , \cdot )$ is a discriminator (Goodfellow et al. 2014) that replaces a density estimate with an adversarial critic trained concurrently. This offers more choice of the exact $f$ -divergence to minimize at the cost of increased training instability (Jolicoeur-Martineau 2020).

Critic Ensembles Our ensemble experiments imply that Onestep-trained policies might perform better than prior work reports. The optimistic pessimism of $Q _ { \mathrm { L C B } }$ ensembles could enable algorithms to learn better policies while still enjoying the stability of on-policy evaluation.

Limitations EBMs can be difficult and computationally expensive to train. As a consequence of the Manifold hypothesis, they may also generalize poorly (Bengio, Courville, and Vincent 2013), though all models capable of multimodal learning suffer from their own slew of problems (Goodfellow, Bengio, and Courville 2016). Advancements in methodology have improved the stability of training and quality of models (Du and Mordatch 2019). Both prior work (Florence et al. 2022) and the results of our experiments suggest that EBMs are well-suited for offline RL.

# Conclusion

In this paper, we introduce Behavior Preference Regression (BPR). Our method formulates a reframed, paired-sample policy objective that directly trains a policy likelihood to be behaviorally consistent and maximize reward, using leastsquares regression. Though our method is motivated by finetuning approaches in language models, it is extensible to offline RL. We validate our algorithm on datasets with a variety of task types and reward schemes that offer both proprioceptive and image-based state spaces. BPR consistently outperforms prior RM-based approaches and preference-based ones by a substantial margin.

Additional experiments evaluating Onestep BPR demonstrate that our algorithm can learn policies that outperform previous Onestep methods. Furthermore, with more expressive Onestep value functions, BPR makes headway on the challenging Antmaze tasks that typically demand off-policy evaluation.

Future work should further review the viability of Onestep ensembles and look to adapt paired completion approaches for offline continuous control.