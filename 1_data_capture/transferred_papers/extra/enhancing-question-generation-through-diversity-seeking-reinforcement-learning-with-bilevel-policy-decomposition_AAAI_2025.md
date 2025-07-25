# Enhancing Question Generation through Diversity-Seeking Reinforcement Learning with Bilevel Policy Decomposition

Tianyu Ren, Hui Wang\*, Karen Rafferty

School of Electronics, Electrical Engineering and Computer Science, Queen’s University Belfast, United Kingdom tren01, h.wang, k.rafferty @qub.ac.uk

# Abstract

Recent advancements in question generation (QG) have been significantly propelled by reinforcement learning (RL). Although extensive reward models have been designed to capture the attributes of ideal questions, their associated learning challenges, particularly in sample efficiency and diversity, remain underexplored. This paper introduces a bilevel policy decomposition (BPD) framework and a diversity-seeking RL (DSRL) objective to address these issues. The BPD framework utilizes two cascading policies to divide QG into two more manageable sub-tasks: answer-centric summary generation and summary-augmented QG, facilitating exploration and accelerating policy learning. Concurrently, the DSRL objective preserves the inherent diversity of QG by ensuring the bilevel policies align probabilistically with their reward models rather than merely maximizing returns. Our integrated approach, named BPD-DSRL, demonstrates superior performance over existing baselines on multiple question quality and diversity metrics across various QG benchmarks.

Code and other supplementary material — https://github.com/Tianyu-Ren/BPD-DSRL

Context with highlighted answer: # Questions (Rewards > 0.75) He seemed to be a quick student, adopting new W/ BPD   
tseuchnaolsosgiiegseanwda rifdaeraesfrtohmat  the  eCnhcionuenste.red, W/O BPD Question by vanilla supervised model:   
Who did Chopin learn siege warfare from?   
Rationale & Question by our method: # He adopting new technologies and ideas   
such as siege warfare from the Chinese. 0 Where did he learn about siege warfare? 0 Training Steps 50 (a) Rewards vs. Likelihoods Context with highlighted answer:   
In January 1943, at the age of 86, Tesla died   
alone in room 3327 of the New Yorker Hotel.   
Diverse Rationales & Questions:   
Tesla died at the age of 86.   
How old was Tesla when he died?   
Tesla died at the age of 86. Our Method   
What age was Tesla when he died? REINFORCE -3 Tesla died at the age of 86 in 1943.   
How old was Tesla when he died in 1943? -4 Log Rewards -7 (b)

# Introduction

Question Generation (QG) aims to generate questions from a given reading passage and answer pair. As a core task within question answering (QA), QG offers a wide range of practical benefits, such as data augmentation for QA (Shakeri et al. 2020; Yu et al. 2024), dialogue system development (Ling et al. 2020) and assessment generation for educational purposes (Zhao et al. 2022; Yoon and Bak 2023).

QG has advanced significantly with the advent of pretrained language models (PLMs) (Radford et al. 2019) and robust QG datasets (Rajpurkar et al. 2016). Although these PLMs can be effectively adapted for QG through supervised fine-tuning (SFT), they are still prone to generating unfaithful or hallucinated content (Gou et al. 2023; Xia et al. 2023) (see Figure 1 (a), left part). This issue primarily arises from their use of maximum likelihood estimation, which penalizes deviations from the ground truth but does not always promote alignment or faithfulness (Christiano et al. 2017;

Xie et al. 2020; Ouyang et al. 2022; Tian et al. 2024). To address this, some studies (Chen, Wu, and Zaki 2023; Hong and Liu 2024) have employed reinforcement learning (RL) (Sutton and Barto 2018) to optimize QG through reward models that explicitly capture desirable question attributes. We refer to this line of research as RLQG.

Ideal questions should possess a range of desirable attributes, such as relevance to context, fluency, and answerability. Previous RLQG work has focused on crafting novel QG-specific reward models to ensure these qualities (Xie et al. 2020; Gaur et al. 2022; Chen, Wu, and Zaki 2023). While various robust reward models have been discovered, effectively aligning the QG policy with them remains an underexplored challenge. Existing RLQG methods typically frame QG as a one-step generation process (Hong and Liu 2024). However, like many other intricate tasks such as mathematical problem solving (Wei et al. 2022), the sourcetarget mapping in QG is highly implicit and requires deductive reasoning steps (Zhao et al. 2022; Xia et al. 2023). Directly modeling this complex mapping within one single policy presents substantial exploration challenges in pursuit of high-return outcomes, resulting in low sample efficiency and finally a sub-optimal QG policy (Yan et al. 2024; Zhou et al. 2024), as shown in the right part of Figure 1 (a).

Additionally, an inherent property of QG, namely diversity, is frequently neglected in existing RLQG research. Most prior efforts aim for a QG policy that can generate the optimal question with maximum expected returns (Chen, Wu, and Zaki 2023; Hong and Liu 2024). However, QG inherently has a one-to-many nature (Shen et al. 2019), where many questions with high returns are all acceptable for one passage-answer pair (Figure 1 (b)). Focusing solely on the single best question while discarding the full range of acceptable alternatives conflicts with this inherent property, compromising the effectiveness of the resulting QG policies in downstream applications such as data augmentation for QA (Sultan et al. 2020; Yu et al. 2024).

In light of the above challenges, this study aims to advance RLQG by enhancing sample efficiency and promoting diversity. To improve sample efficiency during RLQG, we first formulate RLQG as a bilevel optimization problem (Colson, Marcotte, and Savard 2007) and introduce a bilevel policy decomposition framework that hierarchically segments the overall QG policy into a high-level rationale policy and a low-level action policy. The rationale policy directs the agent to generate answer-centric summaries from the input, maintaining a working memory to provide clues for the action policy to generate questions. This proposed method divides QG into two more manageable and coherent sub-tasks, i.e., answer-centric summary generation and summary-augmented QG, thereby simplifying the exploration process and accelerating policy learning.

To maintain the one-to-many nature of RLQG, we present a diversity-seeking RL objective inspired by Generative Flow Networks (GFlowNets) (Bengio et al. 2021, 2023) and Soft Q-learning (Haarnoja et al. 2017). This diversityseeking RL objective shifts attention from the maximumreward state to the entire reward distribution, guiding the policy to match the reward model in probability rather than to find a configuration that maximizes rewards or returns (Malkin et al. 2022; Hu et al. 2024). In other words, the optimal policy under this objective is defined to generate sequences with probabilities proportional to their rewards, thereby balancing optimality and diversity.

In summary, our main contributions are the following: (i) We identify the sample efficiency issue in existing RLQG work. To handle this, we propose a bilevel policy decomposition framework to facilitate the exploration of high-return outcomes, thus improving sample efficiency. (ii) We identify the diversity issue in current RLQG studies and introduce a new RL objective that models the full diversity of the reward distribution rather than the maximal-reward state to maintain

QG diversity. (iii) Our method sets a new state-of-the-art on multiple QG benchmarks, surpassing previous RLQG methods in both question quality and diversity metrics.

# Related Work

Reinforcement Learning for Question Generation. Previous RLQG research has primarily focused on reward engineering to enhance QG quality. Some investigations emphasize the similarity between the generated questions and the references, using lexical metrics (Chen, Wu, and Zaki 2023) or semantic metrics (Gaur et al. 2022; Zhang and Bansal 2019) as the reward model. However, these reference-based rewards may restrict exploration in policy gradient methods as they actually encourage agents to mimic the reference questions. To this end, some studies have begun to explore reference-free rewards (Ramnath et al. 2024) to reflect QG quality, such as answerability, fluency, and context relevance (Xie et al. 2020; Hong and Liu 2024). While numerous reward signals for QG quality have been studied, their effective deployment in RL has received considerably less attention. This paper aims to advance this research area by refining the policy learning process and ensuring the developed RL policy can maintain the one-to-many nature of QG.

Diverse Question Generation Modeling. Diversity is inherent to QG and holds value for many downstream tasks (Sultan et al. 2020). Early attempts formulate diverse QG modeling as a variational inference problem (Shen et al. 2019; Cho, Seo, and Hajishirzi 2019; Wang et al. 2020). They incorporate a latent variable for content selection and maximize the evidence lower bound to find the best candidate distribution. Similarly, Narayan et al. (2022) introduce a diverse sampling method which first utilizes nucleus sampling (Holtzman et al. 2020) to sample a chain-of-entities and uses beam search (Freitag and Al-Onaizan 2017) to find questions with high likelihood conditioned on the entities. Some other work also seeks to boost question diversity by recursive generation (Yoon and Bak 2023) and retrievalaugmented style transfer (Gou et al. 2023). Different from previous work, we focus on extracting complex and diverse behaviors from the language model itself using RL, with no additional efforts for crafting proposal distributions or retrieving external question templates.

# Methodology

# Preliminaries and Notions

We follow previous RLQG studies to consider the policy gradient framework of RL and adopt the Markov Decision Process (MDP) as the mathematical model (Ramamurthy et al. 2023). A standard RLQG procedure can be viewed as an MDP $\langle S , \mathcal { Q } , \mathcal { P } , \mathcal { R } , \mathcal { T } \rangle$ using a finite vocabulary $\nu$ in a sparse reward environment. An episode in the MDP starts with a reading passage and answer pair $( x , a )$ which is used as the initial state $s _ { 0 } \in \mathcal { S }$ , where $s$ is the state space. At each time step $t$ , the agent follows a policy $\pi : \mathcal { S } \overset { \cdot } { \times } \mathcal { Q } \mapsto [ 0 , 1 ]$ , which is generally represented by a parameterized function (e.g., a neural network) $\pi _ { \boldsymbol { \theta } }$ , to sample a question token $q _ { t } \in \mathcal { Q }$ from vocabulary $\nu$ . The transition function $\mathcal { P } : \mathcal { S } \times \mathcal { Q } \mapsto \bigtriangleup ( \mathcal { S } )$ deterministically appends $q _ { t }$ to the end of the state $s _ { t - 1 } = ( x , a , q _ { < t } )$ . This process continues until $t$ exceeds the time horizon $\tau$ or an end-ofsentence (EOS) token is generated, yielding a question sequence $q = ( q _ { 0 } , \dots , q _ { T } )$ . At the end of an episode, a reward model $\mathcal { R } : \mathcal { S } \times \mathcal { Q } \mapsto \mathbb { R } ^ { 1 }$ assigns a scalar reward value to the last generated token. A typical goal in previous RLQG work is to find the parameters $\theta$ of the policy that maximizes the expected rewards $J ( \pi _ { \theta } ) = \mathbb { E } _ { s \sim \rho ^ { \pi } , q \sim \pi _ { \theta } } [ \mathcal { R } ( s , q ) ]$ .

# Bilevel Policy Decomposition

Directly learning a QG policy $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { q } | \boldsymbol { x } , \boldsymbol { a } )$ through reward optimization is challenging due to the implicit mapping $( x , a ) \mapsto q$ , which hinders the exploration of high-return questions and impedes sample efficiency. To address this, we introduce a bilevel policy decomposition (BPD) framework, which segments $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { q } | \boldsymbol { x } , \boldsymbol { a } )$ into a high-level rationale policy $\pi _ { \boldsymbol { \theta } } ( d | \boldsymbol { x } , \boldsymbol { a } )$ and a low-level action policy $\pi _ { \boldsymbol { \theta } } ( q | d , \boldsymbol { x } , \boldsymbol { a } )$ :

$$
\pi _ { \boldsymbol { \theta } } ( q \vert x , a ) = \sum _ { d } \pi _ { \boldsymbol { \theta } } ( q , d \vert x , a ) = \underbrace { \pi _ { \boldsymbol { \theta } } ( d \vert x , a ) } _ { \mathrm { r a t i o n a l e p o l i c y } } \underbrace { \pi _ { \boldsymbol { \theta } } ( q \vert x , a , d ) } _ { \mathrm { a c t i o n p o l i c y } } .
$$

Both policies are optimized with policy gradients to achieve a simpler sub-task and finally work synergistically for QG. Specifically, the rationale policy $\pi _ { \boldsymbol { \theta } } ( d | \boldsymbol { x } , \boldsymbol { a } )$ directs the agent to generate a rationale $d$ (i.e., an answer-centric summary; see Figure 1 (a)). It maintains a working memory that captures process reward signals $r _ { d }$ and provides clues for future decision-making; the action policy $\pi _ { \boldsymbol { \theta } } ( q | d , \boldsymbol { x } , \boldsymbol { a } )$ subsequently leverages the rationale and the initial input to generate questions, aiming to align these questions with the outcome reward signals $r _ { q }$ .

The idea of BPD is: by breaking down an intricate task into several coherent and simpler sub-tasks, the overall problem becomes more approachable, allowing for more efficient exploration of high-return outcomes (Yan et al. 2024; Zhou et al. 2024; Yao et al. 2023; Zhou 2023). First, we use a rationale policy to solve the task $( x , a ) \mapsto d$ for the essential source-target transformation, which intuitively presents less ambiguity compared to $( x , a ) \mapsto q$ . Subsequently, $d$ serves as an intermediate representation to augment the context to the action policy, which makes the exploration of good questions more explicit and easier, thereby improving the overall sample efficiency during RL.

# Diversity-Seeking RL Objective

Building on our BPD framework, we aim to develop rationale and action policies that can sample a diverse set of high-return solutions. This objective recognizes the inherent one-to-many nature of QG, which contrasts previous RLQG work that focuses on a single return-maximizing question. To achieve this, our approach is to develop a learning objective that directly transforms the reward model into a generative policy, such that the likelihood of generating sequences becomes proportional to their rewards. Formally, given the process reward model $r _ { d } ( x , a , d )$ and the outcome reward model $r _ { q } ( x , a , q )$ , the optimal rationale policy and the optimal action policy can be defined as:

$$
\pi _ { \theta } ^ { * } ( d | x , a ) \propto r _ { d } ( x , a , d ) = \frac { r _ { d } ( x , a , d ) } { \sum _ { d } r _ { d } ( x , a , d ) } ,
$$

$$
\pi _ { \theta } ^ { * } ( q | x , a , d ) \propto r _ { q } ( x , a , q ) = \frac { r _ { q } ( x , a , q ) } { \sum _ { q } r _ { q } ( x , a , q ) } ,
$$

where $\textstyle \sum _ { d } r _ { d } ( x , a , d )$ and $\textstyle \sum _ { q } r _ { q } ( x , a , q )$ are the partition functions that turn the unnormalized measures of reward models into probability distributions.

Using the multiplication rule, we can further derive the overall optimal policy from Eq. (2) and (3):

$$
\pi _ { \boldsymbol { \theta } } ^ { * } ( \boldsymbol { q } , d | x , a ) = \frac { r _ { d } ( x , a , d ) r _ { q } ( x , a , q ) } { \sum _ { d } r _ { d } ( x , a , d ) \sum _ { q } r _ { q } ( x , a , q ) } .
$$

We cast the optimal policy search problem to a minimization problem. Suppose there is a bilevel decomposed policy $\dot { \pi _ { \boldsymbol { \theta } } } ( q , d | x , a )$ with parameters $\theta$ and a regressor $Z _ { \mu } ( \dot { x } , a )$ with parameters $\mu$ which takes a $( x , a )$ pair as input and estimates $\begin{array} { r } { \sum _ { d } r _ { d } ( x , a , d ) \sum _ { q } r _ { q } ( x , a , q ) } \end{array}$ , the diversity-seeking RL (DSRL) objective is defined using $L _ { 2 }$ loss on a log-scale:

$$
\mathcal { L } _ { \mathrm { D S R L } } ( d , q ) = [ \log \pi _ { \theta } ( q , d | x , a ) - \log \frac { r _ { d } ( x , a , d ) r _ { q } ( x , a , q ) } { Z _ { \mu } ( x , a ) } ] ^ { 2 } .
$$

Eq. (5) reveals an intriguing connection to the Trajectory Balance (TB) objective (Malkin et al. 2022) for GFlowNets training. It can be treated as a special case of TB where the policy is conditional on an input variable and samples actions autoregressively. If one policy can satisfy the constraint in Eq. (4), it is easy to find $\mathcal { L } _ { \mathrm { D S R L } } ( d , q ) = 0$ for any trajectory $( d , q )$ it samples. Conversely, if all trajectories can lead Eq. (5) to zero, the regarding policy then satisfies Eq. (4). Detailed proof can be found in (Malkin et al. 2022), which guarantees the correctness of the DSRL objective.

# Implementation

Integrating the above two components, we develop the BPDDSRL method. The implementation of BPD-DSRL is outlined in Figure 2, which will be detailed as follows.

# Bilevel Policy Warm-up with SFT

Given the vast action space during language modeling, directly applying the proposed on-policy RL to PLMs is notably inefficient (Guo et al. 2022). Therefore, we first use SFT to initialize the rationale and action policy from PLMs, with the aim to refine the search domain during RL:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S F T } } = \mathbb { E } _ { ( d , q , x , a ) \sim D } [ - \log \pi _ { \theta } ( d , q | x , a ) ] , } \end{array}
$$

where $D$ is the QG dataset which contains quadruples $( d , q , x , a )$ , with $d$ denoting the answer-centric summaries.

A major challenge in this setup is the effective labeling of $d$ . Rather than generating $d$ from $( x , a )$ in a prior manner by text summarization models or manual annotation, we employ a question conversion model (Chen, Choi, and Durrett 2021) to inversely synthesize $d$ from $( q , a )$ . This model takes a $( q , a )$ pair as input and converts it to a declarative statement, serving as the summary of the reading passage. Annotating $d$ in this way not only maintains its semantic integrity as the answer-centric summary but also better ensures the coherence between it and the corresponding question.

Step 1: Supervised Fine-tuning Step 2: Improving Question Generation via Diversity-Seeking Reinforcement Learning Hm Step 2.2 Score the rationale and the question using two reward models % Sampled Reading Passage: Rationale Policy Output: D NLI Model It is a replica of the frotto at Lourdes, France The Virgin Mary reputedly appeared in Bilevel Policies Outcome Reward Model where the Virgin Mary reputedly appeared Lourdes, France to Saint Bernadette Soubirous. to Saint Bernadette Soubirous in 1858. ⇡✓(q, d|x, a) rq(x, a, q) , pφ(q, a|x) Sampled Answer: Entailment Reward $p _ { \mathrm { N L I } } ( \mathrm { e n t a i l m e n t } | x , d )$ Saint Bernadette Soubirous Action Policy Output: √ (q, d, x, a) (q, x, a) Process Reward Rationale Policy Output: Who did the Virgin Mary reputedly appear to QG Training dataset $r _ { d } ( x , a , d )$ The Virgin Mary reputedly appeared in in Loudes France? Reading Passage x: It is a replica of the frotto at Lourdes, France Lourdes, France to Saint Bernadette Fluency Reward  pφ(q|x) |q| SwahienrteBtehrenaVdiregtiten SMoaurbyi roeupsutiend1ly85a8p.peared to SAocutiboirnouPso.licy Output: Answerability Reward  pφ(a x, q) |a| Answer a: Outcome Reward Who did the Virgin Mary reputedly SRaitnitonBaelrenad (eattnesSwoeurbcireonutrsic summary) qStue $r _ { q } ( x , a , q )$ eprastisoanpgapele-easrntasonwidenrLpoauirdses FSranmcpel?e (d, q) Step 2.3 Optimizing writdh(xD, Sa,RdL)rOq(bxje, cat,iqv)e 2 LQouersdtieosnFrqa:nce to Saint Bernadette Soubirous. Sample (x, a) 会 Zµ(x, a) ⇡ rd(x, a, d) Zµ(rxq, (ax), a, q) appear in 1858 in Lourdes France. Behavior Polices E(d,q) ⇡[r✓,µLDSRL(d, q)|x, a]

# Reward Model

Process Reward Model. The process reward model offers supervisory signals to guide the rationale policy in generating effective intermediate representations to support the action policy. Consider ideal answer-centric summaries should be textually entailed by the corresponding reading passages to ensure the consistency of subsequent QG, we use natural language inference (NLI) models for the textual entailment task (Bowman et al. 2015) as the process reward model:

$$
r _ { d } ( x , a , d ) \triangleq p _ { \mathrm { N L I } } ( \mathrm { e n t a i l m e n t } | x , d ) ,
$$

where $p _ { \mathrm { N L I } }$ is the NLI model that takes $x$ as premise and $d$ as hypothesis, producing a entailment probability for $d$ which we treat as the process reward score. In practice, we use one off-the-shelf NLI model from (Nie et al. 2020) as the process reward model due to its superior performance.

Outcome Reward Model. The outcome reward model guides the action policy toward generating high-quality questions. Following practices in (Xie et al. 2020), we assess question quality using two key metrics: fluency and answerability. Specifically, given two language models $p _ { \mathrm { f l u } } ( q | x )$ and $p _ { \mathrm { a n s } } ( a | q , x )$ , the fluency and answerability scores of an observed question are determined by the geometric mean of the likelihoods produced by these models. Combining these two measures, the outcome reward model is defined as:

$$
r _ { q } ( x , a , q ) \triangleq p _ { \mathtt { H u } } ( q | x ) ^ { \frac { 1 } { | q | } } p _ { \mathtt { a n s } } ( a | q , x ) ^ { \frac { 1 } { | a | } } ,
$$

where $\left| \cdot \right|$ indicates the valid number of tokens.

By observing the left-to-right nature of autoregressive language models, we can exploit both fluency and answerability scores from a unified model $p _ { \phi } ( q , a | x ) \stackrel {  } { = } p _ { \phi } ( q | x ) p _ { \phi } ( a | q , \dot { x ) }$ to lower implementation cost. In practice, we learn this reward model with the following multi-task SFT objective:

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } } _ { \mathrm { r e w a r d } } ^ { q } = - { \mathbb { E } } _ { ( q , x , a ) \sim D } [ \frac { \beta } { | q | } \sum _ { i = 1 } ^ { | q | } \log p _ { \phi } ( q _ { i } | x , q _ { < i } ) } } \\ { { \displaystyle ~ + \frac { 1 - \beta } { | a | } \sum _ { m = 1 } ^ { | a | } \log p _ { \phi } ( a _ { m } | x , q , a _ { < m } ) ] } , } \end{array}
$$

where $\beta$ is the coefficient which controls the loss contribution. Note that when $\beta$ equals to q |+q|a , the above objective degenerates to the single-task one which maximizes the loglikelihood of the continual sequence $( q , a )$ .

# Optimize the Policy Against the Reward

Putting the supervised bilevel policies and two reward models in a bandit environment which presents a random $( x , a )$ pair from the training dataset and expects $( d , q )$ as responses, we further fine-tune the bilevel policies using the following RL objective based on Eq. (5):

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { D S R L } } ( d , q ) = [ \log \pi _ { \theta } ( d \vert x , a ) + \log \pi _ { \theta } ( q \vert d , x , a ) + \log Z _ { \mu } ( x , a ) } \\ & { - \log p _ { \phi } ( q \vert x ) ^ { \frac { 1 } { \vert q \vert } } p _ { \phi } ( a \vert q , x ) ^ { \frac { 1 } { \vert a \vert } } - \log p _ { \mathrm { N L I } } ( \mathrm { e n t a i l m e n t } \vert x , d ) ] ^ { 2 } , \quad } \end{array}
$$

with stochastic gradient:

$$
\begin{array} { r } { \mathbb { E } _ { ( d , q ) \sim \pi } [ \nabla _ { \theta , \mu } \mathcal { L } _ { \mathrm { D S R L } } ( d , q ) | x , a ] , } \end{array}
$$

where $Z _ { \mu } ( x , a )$ is an encoder-only regressor initialized from the supervised bilevel policies $\pi _ { \theta } ^ { \mathrm { S F T } } ( q , d | x , a )$ .

As the state space for language modeling is combinatorially large, it is important to have a training policy that can efficiently explore it. We follow the same settings described in (Hu et al. 2024), using trajectories sampled from three different sources to optimize Eq. (10): (1) the current policy $\pi _ { \boldsymbol { \theta } }$ , (2) a tempered version of $\pi _ { \boldsymbol { \theta } }$ , and (3) a replay buffer which stores past experiences credited with high rewards. The pseudo-code for BPD-DSRL training and further technical specifics are provided in the supplementary material.

Table 1: Statistics of the selected benchmarks. SQuAD 1.1 / 1 and SQuAD 1.1 / 2 are two different splits of SQuAD 1.1 from (Zhou et al. 2017) and (Du, Shao, and Cardie 2017).   

<html><body><table><tr><td>Dataset</td><td>Train</td><td>Validation</td><td>Test</td></tr><tr><td>SQuAD1.1/1</td><td>86635</td><td>8965</td><td>8964</td></tr><tr><td>SQuAD 1.1/ 2</td><td>75722</td><td>10570</td><td>11877</td></tr><tr><td>NewsQA</td><td>92549</td><td>5166</td><td>5126</td></tr></table></body></html>

# Experiments

# Experimental Setup

Benchmarks. Following previous work (Gou et al. 2023; Narayan et al. 2022; Wang et al. 2020), we conduct experiments on two QG datasets: SQuAD 1.1 (Rajpurkar et al. 2016) and NewsQA (Trischler et al. 2017). Due to the inaccessibility of SQuAD 1.1’s test set, we use two popular splits of it from (Zhou et al. 2017) and (Du, Shao, and Cardie 2017), which we will further refer to SQuAD $1 . 1 ~ / ~ 1 ~ \$ and SQuAD 1.1 / 2 respectively. As for NewsQA, we apply necessary truncation of its reading passage to comply with the input length constraints of PLMs (e.g., 512 tokens). Statistics about these benchmarks are presented in Table 1.

Metrics. For every testing example $( q _ { n } , x _ { n } , a _ { n } ) _ { n = 1 } ^ { N }$ n=1, we generate $K$ questions from QG models and adopt the following three BLEU-based metrics (Papineni et al. 2002) to evaluate the quality and the diversity of them:

• Top-1 metric $( \uparrow )$ : This metric evaluates the consistency between model-generated (candidate) questions with the highest confidence and the reference questions $q _ { n }$ . For every test example, it keeps the candidate question $\hat { q } _ { n } ^ { * }$ that has the highest likelihood, i.e., $\hat { q } _ { n } ^ { * } =$ arg $\begin{array} { r } { \operatorname* { m a x } _ { \hat { q } _ { n } ^ { k } } p _ { \mathrm { Q G } } ( \hat { q } _ { n } ^ { k } , d _ { n } ^ { k } | x _ { n } , \check { a _ { n } } ) } \end{array}$ , and calculate the corpuslevel BLEnU score usin|g $\{ ( q _ { n } , \hat { q } _ { n } ^ { * } ) \} _ { n = 1 } ^ { N }$ . • Oracle metric $( \uparrow )$ : This metric evaluates the upper bound of the consistency between the reference and candidate questions. The only difference between this metric and the Top-1 metric lies in the selection strategy of $\hat { q } _ { n } ^ { * }$ . Here, $\hat { q } _ { n } ^ { * }$ is the candidate question that achieves the highest sentence-level BLEU score with $q _ { n }$ . • Self metric ( ): This metric evaluates the average diversity of candidate questions. It pair-wisely calculates the sentence-level BLEU score among the $K$ candidate questions for every testing example. Finally, it averages the $N$ self-metric value to indicate the capability of QG models to generate diverse questions.

In our experiments, we follow (Gou et al. 2023) to set $K = 5$ and report BLEU-4 calculated by SacreBLEU (Post 2018).

Baselines. We compare the proposed BPD-DSRL method with state-of-the-art QG approaches to validate its effectiveness. The baselines can be grouped into three categories:

• Cross entropy based QG (CEQG). We include Composition (Narayan et al. 2022) and RAST (Gou et al. 2023) as our CEQG baselines. They can be treated as extensions of traditional supervised QG methods, where Composition adds a prefix entity sampling task before QG and

RAST combines QG with retrieval augmented generation (RAG) (Lewis et al. 2020) to boost diversity.

• Large language models (LLMs). We include two opensourced LLMs: LLaMA3-8B-instruct and Mistral-7Binstruct-v0.3 (Jiang et al. 2023) as our LLM baselines. We report their zero-shot and three-shot performance in the main experiments. Prompts and demonstrations are presented in the supplementary material. • RLQG. REINFORCE (Williams 1992) and Proximal Policy Optimization (PPO) (Schulman et al. 2017) are two representative RL algorithms used by previous RLQG work (Hong and Liu 2024; Chen, Wu, and Zaki 2023; Gaur et al. 2022). We implement them with BPD as the RLQG baselines. Specific implementation details are presented in the supplementary material.

Implementation Details. All of our QG models and outcome reward models start from the pre-trained checkpoints of T5-large (Raffel et al. 2020). We use consistent hyperparameter configurations across all three datasets during training (SFT warm-up and RL) and inference. The implementation of them is detailed in the supplementary material.

# Main Results and Analysis

Table 2 presents the comparative results on three widelyused QG benchmarks. Regarding QG quality (consistency with the reference), our BPD-DSRL consistently surpasses all the baselines on the Top-1 metric, and secures the best average rank (1.13 compared to the runner-up’s 1.67) on the Oracle metric. As for diversity, our method outperforms all non-RAG approaches on the Self metric, including Composition (Narayan et al. 2022), LLMs, and RLQG baselines.

More specifically, it is clear that the RLQG group generally achieves better performance on Top-1 and Oracle compared to the CEQG baselines (Narayan et al. 2022; Gou et al. 2023). This finding verifies our motivation for using RL to improve QG quality. While RAST (Gou et al. 2023) offers enhanced diversity, its reliance on RAG may incur significant costs regarding memory usage and inference latency. In fact, RAST provides a stand-alone RAG framework that could potentially be integrated into our method to further enrich QG diversity. However, consider the primary focus of this paper is on RLQG and the high implementation demands of RAST, we reserve its integration for future work.

Within the RLQG group, our method generally outperforms the baselines on QG quality and diversity metrics across all test datasets, yielding the most favorable overall performance. We conduct further analysis in the next section using advanced metrics for QG quality and diversity to better understand their model behaviors.

As for LLMs, our method significantly outperforms zeroshot and few-shot configurations of LLaMA-3-8B-Instruct and Mistral-7B-Instruct-v0.3, even with about $10 \%$ parameters. This makes our method a more resource-efficient choice for QG-specific scenarios such as data augmentation.

# Ablation Studies

To further validate our approach, we conduct a series of ablation studies on our BPD framework and DSRL objective.

Table 2: Comparison of different QG methods. Here, LLaMA3† and Mistral† respectively indicate the 8B-Instruct version and the 7B-Instruct- $\mathbf { \sigma } \cdot \mathbf { v } 0 . 3$ version of them. The up-arrow $\uparrow$ means higher value is better and the down-arrow $\downarrow$ means lower value is better. Experimental results of Composition (Narayan et al. 2021) is reevaluated by (Gou et al. 2023).   

<html><body><table><tr><td rowspan="2">Group</td><td rowspan="2">Model</td><td rowspan="2">Year</td><td colspan="3">Top-1TQuAD1171 Self1</td><td colspan="3">Top-1 SQuAD1.1/2 Sef↓</td><td colspan="3"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>Top-1个</td><td>NrwsQ</td><td>Self↓</td></tr><tr><td rowspan="2">CEQG</td><td></td><td></td><td>16.50</td><td>25.70</td><td>58.99</td><td>15.94</td><td>24.9</td><td>60.05</td><td></td><td></td><td></td></tr><tr><td>CoRAsition</td><td>2022</td><td></td><td></td><td></td><td></td><td></td><td></td><td>11.02</td><td>16.26</td><td>23.16</td></tr><tr><td rowspan="4">LLMs</td><td>LLaMA3† 3-shot</td><td>2024</td><td>12.19</td><td>20.45</td><td>75.14</td><td>12.20</td><td>20.55</td><td>77.87</td><td>4.53</td><td>8.57</td><td>74.36</td></tr><tr><td>LLaMA3+ 0-shot</td><td>2024</td><td>10.83</td><td>19.19</td><td>73.16</td><td>10.42</td><td>18.86</td><td>73.87</td><td>3.91</td><td>8.41</td><td>69.23</td></tr><tr><td>Mistral† 3-shot</td><td>2023</td><td>11.02</td><td>16.47</td><td>71.50</td><td>11.80</td><td>17.27</td><td>74.92</td><td>4.41</td><td>7.31</td><td>71.11</td></tr><tr><td>Mistral+ O-shot</td><td>2023</td><td>10.74</td><td>16.55</td><td>69.88</td><td>10.33</td><td>15.99</td><td>69.92</td><td>4.38</td><td>7.49</td><td>68.83</td></tr><tr><td rowspan="3">RLQG</td><td>BPD-REINFORCE</td><td>-</td><td>19.82</td><td>25.65</td><td>67.79</td><td>18.93</td><td>25.16</td><td>71.71</td><td>14.11</td><td>20.30</td><td>63.63</td></tr><tr><td>BPD-PPO</td><td></td><td>18.99</td><td>26.31</td><td>56.85</td><td>18.47</td><td>26.72</td><td>61.91</td><td>13.92</td><td>21.09</td><td>55.44</td></tr><tr><td>BPD-DSRL</td><td>Ours</td><td>19.92</td><td>26.58</td><td>55.28</td><td>19.66</td><td>27.02</td><td>58.63</td><td>14.13</td><td>20.69</td><td>49.19</td></tr></table></body></html>

Table 3: Ablation studies on the effectiveness of BPD and DSRL. In the DSRL column, $r _ { d }$ and $r _ { q }$ respectively indicate whether we use the process or outcome reward models during training. Note that for baselines trained without BPD, the process reward model is defined by $p _ { \mathrm { N L I } }$ (entailment $| x \rrangle$ $, f ( q , a ) )$ ), where $f ( \cdot )$ is the question conversion model (Chen, Choi, and Durrett 2021).   

<html><body><table><tr><td rowspan="2">BPD</td><td colspan="2">DSRL</td><td colspan="3">SQuAD 1.1/1</td><td colspan="3">SQuAD 1.1/2</td><td colspan="3">NewsQA</td></tr><tr><td>rd</td><td>rq</td><td>Top-1个</td><td>Oracle↑</td><td>Self↓</td><td>Top-1个</td><td>Oracle↑</td><td>Self↓</td><td>Top-1个</td><td>Oracle↑</td><td>Self↓</td></tr><tr><td></td><td>-</td><td>-</td><td>18.52</td><td>25.12</td><td>48.44</td><td>18.45</td><td>25.89</td><td>51.17</td><td>10.20</td><td>16.79</td><td>31.89</td></tr><tr><td>-</td><td>√</td><td></td><td>18.55</td><td>25.19</td><td>48.40</td><td>18.53</td><td>26.01</td><td>51.84</td><td>11.73</td><td>18.00</td><td>41.87</td></tr><tr><td></td><td>-</td><td>√</td><td>17.97</td><td>24.71</td><td>43.04</td><td>17.86</td><td>25.36</td><td>45.42</td><td>10.27</td><td>15.73</td><td>30.89</td></tr><tr><td>1</td><td>√</td><td>√</td><td>18.56</td><td>25.30</td><td>48.51</td><td>18.59</td><td>25.98</td><td>51.57</td><td>12.47</td><td>18.74</td><td>43.43</td></tr><tr><td>√</td><td></td><td></td><td>19.41</td><td>25.91</td><td>49.85</td><td>18.85</td><td>26.52</td><td>52.88</td><td>12.27</td><td>18.52</td><td>39.74</td></tr><tr><td>√</td><td>√</td><td>1</td><td>19.45</td><td>25.99</td><td>53.40</td><td>19.05</td><td>26.43</td><td>57.35</td><td>13.72</td><td>19.90</td><td>47.04</td></tr><tr><td>√</td><td>-</td><td>√</td><td>19.64</td><td>26.08</td><td>51.18</td><td>19.30</td><td>27.02</td><td>53.77</td><td>13.18</td><td>20.25</td><td>44.49</td></tr><tr><td>√</td><td>√</td><td>√</td><td>19.92</td><td>26.58</td><td>55.28</td><td>19.66</td><td>27.02</td><td>58.63</td><td>14.13</td><td>20.69</td><td>49.19</td></tr></table></body></html>

Effectiveness of BPD for QG. We first perform an ablation study on the BPD framework to see its effectiveness for the task of QG. To achieve this, we train an SFT baseline $\pi ^ { \mathrm { S F T } } ( q | x , a )$ , which directly models $( x , a ) \mapsto q$ , and compare it with our SFT initialization $\pi ^ { \mathrm { S F T } } ( q , d | x , a )$ . Presented in the first and the fifth row in Table 3, $\pi ^ { \mathrm { S F T } } ( q , d | x , a )$ can outperform its ablated version $\pi ^ { \mathrm { S F T } } ( q | x , a )$ on QG quality metrics across all three datasets. These empirical results are consistent with our previous hypothesis that BPD can make QG more tractable. Since the prefix rationales may constrain the searching space for questions, the QG diversity may get influenced. However, on the two test splits of SQuAD 1.1, we find the diversity cost is generally on an acceptable level.

Effectiveness of BPD for RLQG. We proceed to evaluate how BPD improves sample efficiency in DSRL. Specifically, we train an RLQG baseline from the above supervised model $\pi ^ { \mathrm { S F T } } ( q | x , a ) $ using DSRL (referred to as DSRLonly) under the same training configurations as BPD-DSRL. The evaluation results for BPD-DSRL and DSRL-only are presented in Table 3. Additionally, after each RL training step, we sample 16 questions (rationales) from the current passage-answer pair using both BPD-DSRL and DSRLonly. The cumulative averages of the process and outcome rewards are documented in Figure 3. As shown in the fourth row and the last row in Table 3, within the same time horizon, BPD can bring substantial absolute improvement in quality metrics compared to its ablated version. By respectively comparing BPD-DSRL and DSRL-only with their

SFT initialization, we also notice a great relative enhancement. Taking SQuAD 1.1 / 1 for example, BPD-DSRL gains $2 . 6 1 \%$ and $2 . 5 8 \%$ improvement on Top-1 and Oracle metrics compared to its SFT start point, while the values for DSRLonly are $0 . 2 1 \%$ and $0 . 7 2 \%$ . Figure 3 more vividly illustrates the enhanced sample efficiency achieved by BPD, where BPD-DSRL achieves higher cumulative values of both process and outcome rewards compared to DSRL-only.

Effectiveness of DSRL and Reward Models. We further ablate the reward models to validate their effectiveness and explore whether the proposed DSRL can well leverage them. Starting from $\pi ^ { \mathrm { S F T } } { \dot { ( q | x , a ) } }$ and $\pi ^ { \mathrm { S F T } } ( q , d | x , a )$ , we use DSRL to train four RLQG baselines with either the process reward model or the outcome reward model. As shown in the second to third and the sixth to seventh rows in Table 3, DSRL with both separate reward models can generally bring improvement to the SFT initialization. Combining these two reward models together (the fourth row and the last row), the performance on quality metrics can be further improved. Such experimental results demonstrate the effectiveness of DSRL and indicate the compatibility of the employed process and outcome reward models.

# Further Analysis

In the previous section, we follow prior work to report experimental results by BLEU-based metrics. While these metrics provide initial insights, they may not adequately capture

On SQuAD 1.1 / 1 On SQuAD 1.1 / 2 On NewsQA Cumulative Mean Rewards020406080 (W/ BPD) Cumulative Mean Rewards020406080 60 r (W/O BPD) (W/ BPD) Cumulative Mean Rewards q (W/O BPD) 40 20 0 0 20 40 60 80 100 0 20 40 60 80 100 0 20 40 60 80 100 Time Horzion Time Horzion Time Horzion

<html><body><table><tr><td rowspan="2">RLQG</td><td colspan="4">SQuAD 1.1 /1</td><td colspan="4">SQuAD1.1/2</td><td colspan="4">NewsQA</td></tr><tr><td>HR↓</td><td>ANS↑</td><td>ACC↑</td><td>SBS↓</td><td>HR↓</td><td>ANS↑</td><td>ACC↑</td><td>SBS↓</td><td>HR↓</td><td>ANS↑</td><td>ACC↑</td><td>SBS↓</td></tr><tr><td>BPD-REINFORCE</td><td>4.48</td><td>90.12</td><td>82.66</td><td>94.84</td><td>7.94</td><td>94.34</td><td>86.86</td><td>95.41</td><td>2.64</td><td>91.34</td><td>65.04</td><td>94.41</td></tr><tr><td>BPD-PPO</td><td>7.46</td><td>84.04</td><td>80.17</td><td>93.84</td><td>6.62</td><td>93.89</td><td>84.51</td><td>94.83</td><td>1.97</td><td>90.30</td><td>61.22</td><td>93.94</td></tr><tr><td>BPD-DSRL (Ours)</td><td>7.09</td><td>86.39</td><td>80.93</td><td>93.48</td><td>6.05</td><td>94.99</td><td>85.98</td><td>94.07</td><td>2.62</td><td>91.55</td><td>64.63</td><td>92.65</td></tr></table></body></html>

Table 4: Further analysis on RLQG methods. HR: Hallucination Rate based on Spacy. ANS: Answerability Rate based on GPT-3.5. ACC: Accuracy of GPT-3.5 among questions it judges as answerable. SBS: Self metric based on BertScore.

more nuanced aspects of the generated questions. Focusing on RLQG methods, we conduct a further analysis using metrics beyond the lexical level to gain a more holistic understanding of QG quality and diversity.

Moreover, we also enlist three annotators to critically evaluate the performance of all RLQG methods on QG quality and QG diversity, following similar human evaluation procedures in (Xia et al. 2023; Gou et al. 2023). Detailed results can be found in the supplementary material.

# Hallucination and Answerability

Hallucination and unanswerability have significant impacts on user experience and are crucial for assessing the quality of generated questions (Xie et al. 2020; Narayan et al. 2022). Traditional corpus-level metrics such as BLEU, however, may not be able to effectively measure these attributes (Dale et al. 2023). To this end, we here quantify both indicators to provide deeper insights into QG quality.

To assess hallucination, we employ SpaCy to extract named entities from generated questions and verify their presence in the corresponding reading passages. We calculate and report the average proportion of questions containing hallucinated entities $( \mathrm { H R } \% )$ . As for answerability, we follow a common practice (Liu, Huang, and Chang 2023; Mohammadshahi et al. 2023) to deploy a QA model to determine if a generated question is answerable. For precision and cost-effectiveness, we utilize GPT-3.5 (Turbo-0125) in a zero-shot setting as the QA model. We present both the average proportion of answerable questions $( \mathrm { A N S \% } )$ and the model accuracy $( \mathrm { A C C \% } )$ of these questions.

Table 4 presents the experimental results for the above three metrics. Generally, the RLQG approaches exhibit comparable performance in terms of faithfulness $( \mathrm { H R } \% )$ and answerability $( \mathrm { A N S \% } )$ . A notable observation is that questions generated by the BPD-REINFORCE baseline are most often solvable by GPT-3.5 (highest $\mathrm { A C C \% }$ ). Our human evaluations indicate this could result from the highly extractive nature of questions generated by BPD-REINFORCE, which may suggest the policy is over-optimized.

# Semantic Diversity

Given that the Self-BLEU metric in the main experiments only measures n-gram diversity, we here replace BLEU with BERTScore (Zhang et al. 2020) in the Self metric to provide a more comprehensive analysis of QG diversity. BERTScore evaluates the cosine similarity between embeddings generated by PLMs, offering deeper insights into the semantic diversity of the generated questions. For this analysis, we employ the official BERTScore package in its default configuration and report the resulting BERTScore F1 values.

Table 4 outlines the relevant experimental results. Generally, the outcomes from Self-BERTScore (SBS) are consistent with those obtained from Self-BLEU. Our BPDDSRL continues to achieve the best performance in diversity among the RLQG group while the BPD-REINFORCE baseline tends to yield the most deterministic policy.

# Conclusion

In this paper, we present BPD-DSRL, a bilevel policy decomposition framework and a diversity-seeking reinforcement learning objective, to improve RLQG sample efficiency and preserve QG diversity. Compared to existing RLQG methods, policies developed by BPD-DSRL yield more diverse questions while achieving superior performance across different quality metrics within the same time horizon, setting a new state-of-the-art on three widely-used QG benchmarks. Comprehensive ablation studies, as well as supplementary quantitative and qualitative analyses, further validate the effectiveness of BPD-DSRL in improving sample efficiency and fostering QG diversity.