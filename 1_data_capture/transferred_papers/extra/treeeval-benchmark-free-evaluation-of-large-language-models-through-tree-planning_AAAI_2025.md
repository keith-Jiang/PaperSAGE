# TreeEval: Benchmark-Free Evaluation of Large Language Models through Tree Planning

Xiang $\mathbf { L i } ^ { 1 }$ , Yunshi Lan1\*, Chao Yang2

1East China Normal University 2Shanghai AI Laboratory xiang.li@stu.ecnu.edu.cn, yslan@dase.ecnu.edu.cn, yangchao@pjlab.org.cn

# Abstract

Recently, numerous new benchmarks have been established to evaluate the performance of large language models (LLMs) via either computing a holistic score or employing another LLM as a judge. However, these approaches suffer from data leakage due to the open access of the benchmark and inflexible evaluation process. To address this issue, we introduce TreeEval, a benchmark-free evaluation method for LLMs that let a high-performance LLM host an irreproducible evaluation session and essentially avoids the data leakage. Moreover, this LLM performs as an examiner to raise up a series of questions under a topic with a tree planing strategy, which considers the current evaluation status to decide the next question generation and ensures the completeness and efficiency of the evaluation process. We evaluate 6 models of different parameter sizes, including 7B, 13B, and 33B, and ultimately achieved the highest correlation coefficient with AlpacaEval2.0 using only around 45 questions. We also conduct more analysis to show the robustness and reliability of TreeEval.

# Code — https://github.com/Ashura5/TreeEval

# Introduction

The recent surge in Large Language Models (LLMs) has been significant, transitioning from closed-source (OpenAI 2023; Team 2023a) to open-source (Touvron et al. 2023; et al. 2023a; Jiang et al. 2023) models. Various Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF) techniques have been proposed to further enhance the performance of LLMs (Taori et al. 2023; Chiang et al. 2023; Bai et al. 2022; Ouyang et al. 2022; Tunstall et al. 2023a). These LLMs demonstrate capabilities to address diverse tasks and are widely utilized in both academic and industrial fields. While human evaluation is intuitive for assessing the performance of LLMs, it is timeconsuming and susceptible to unexpected bias (Zheng et al.

2023b; Wang et al. 2024). Thus, investigating automatic evaluation approaches for LLMs becomes crucial.

To date, numerous automatic evaluation methods have been proposed. One approach involves annotating benchmark datasets, such as MMLU and BBH (Hendrycks et al. 2021; Suzgun et al. 2022), to test various capabilities of an LLM. The performance is assessed by checking the overlap between annotated answers and generated answers, producing a holistic score to indicate the LLM’s performance. We refer to this category of evaluation methods as the benchmark paradigm. However, the holistic score can be inflexible for measuring the quality of LLM outputs since token mismatches do not necessarily indicate incorrect answers.

With the advent of high-performance LLMs, another approach leverages them to simulate human evaluation. This involves providing the evaluated LLM with predefined benchmark questions and using another LLM, such as GPT4, to judge its responses (Zheng et al. $2 0 2 3 \mathrm { a }$ ; Li et al. 2023b; Bai et al. 2023; Wang et al. 2023a; Zhang et al. 2023b; Wang et al. 2023c; Li et al. 2023a; Zhu, Wang, and Wang 2023). We refer to this category of evaluation methods as the LLM-as-judge paradigm. However, this evaluation approach can also introduce additional biases, including positional bias (Wang et al. 2023a), verbosity bias (Saito et al. 2023), and style bias (Wu and Aji 2023). Positional bias refers to the tendency to assign higher scores to answers based on their specific positions. Verbosity bias indicates that large language models often prefer more verbose answers, even if these longer responses are not necessarily of higher quality than shorter ones. Style bias manifests in the inclination of large language models to favor answers that match their own generated style, such as giving lower scores to correct responses with spelling errors, since LLMs rarely produce content with spelling mistakes.

Despite enabling automatic evaluation with standard pipelines, both the benchmark and LLM-as-judge paradigms face significant data leakage issues. The extensive training data used in LLM development, considered a valuable asset by many closed and even open-source models, can easily lead to benchmark data leakage, severely biasing evaluation results (Zhou et al. 2023b). To solve this issue, we propose a novel evaluation paradigm, which takes an LLM as an examiner to raise questions. The examiner should produce different evaluation session for each time which makes it hard to duplicate the evaluation questions and protect the evaluation benchmark from disclosure for fine-tuning and pre-training an LLM deliberately. However, simply adopting an LLM as examiner would lead to arbitrary evaluation question generation without a goal. Designing such a benchmark-free evaluation method need take the following aspects into consideration: (1) Similar as the question in a benchmark (Taori et al. 2023; Zheng et al. 2023a), the generated questions should be derived from certain topics, which ensures the scope of the evaluation. (2) Drawing inspiration from the interview, within a topic, the examiner should generate a line of questions that are diverse to cover different knowledge rather than producing a single question. (3) The generation procedure should be flexible enough to generate mutually connected questions and control the difficulty level of these questions. When the current line of question cannot distinguish two LLMs, more difficult questions should be raised up. Otherwise, the evaluation could be terminated immediately.

![](images/ef7d7c440b4f6ea64fc709a4fd7eaddf4fa85123af29f8c031a84a58b382e2d1.jpg)  
Figure 1: Comparison of TreeEval with existing evaluation paradigms.

To this end, we propose TreeEval, which is a benchmarkfree evaluation of the knowledge implication and questionanswering capabilities of LLM through tree planning. The line of questions within a topic for evaluation are organized in a tree, where each node contains a question. In the process of constructing a tree, we repeatedly revisit the status of the current tree and generate the next node until the tree is enough to differentiate two LLMs. The difference between our evaluation method and previous paradigms can be found in Figure 1. To verify the effect of our method, we evaluate multiple LLMs. The results demonstrate that our method shows similar ranking as AlpacaEval2.0 in LLM-as-judge paradigm with only 45 questions in average for each round of evaluation. Further analysis shows our advantages in measuring fine-grained capabilities and conducting robust comparison for LLMs.

Our contributions are summarized as follows:

• We introduce a novel evaluation paradigm, TreeEval, which allows for efficient and comprehensive evaluation of LLMs, inherently preventing data leakage issues. • TreeEval has advantage in distinguishing two LLMs with similar performance by constructing a deeper tree, which extends the evaluation process to obtain more stable and accurate assessment results. • We compare with a set of automatic evaluation baselines, and find that our TreeEval achieves the highest correlation coefficient with AlpacaEval2.0.

# Related Work

# Methods of LLM Evaluation

Due to the explosive growth and rapid update of LLMs, a significant challenge is to conduct accurate and comprehensive evaluation for them (Chang et al. 2023). Early studies leverage open-ended question answering datasets and math word problems as the evaluation benchmarks (Touvron et al. 2023; et al. 2023b; Chen et al. 2024) to evaluate the commonsense knowledge and reasoning capabilities of LLMs. Subsequently, more benchmark datasets like MMLU (Hendrycks et al. 2021), AGIEval (Zhong et al. 2023), IFEval (Zhou et al. 2023a) have been elaborately designed to gauge diverse abilities of LLMs. Some studies (Wang et al. 2023a,a; Saha et al. 2023) go beyond standard evaluation metrics. They evaluate the quality and accuracy of predicted results through human annotation, which is able to provide a more comprehensive feedback. With the emergence of high-performance LLMs like GPT-4 (OpenAI 2023), Gemini Pro (Team 2023a), more recent studies start to utilize them to simulate the human evaluation process. In this realm, PandaLM (Wang et al. 2023c) strives to provide reproducible and automated comparison between various LLMs by training a LLM as the judge. GPTScore (Fu et al. 2023) and G-Eval (Liu et al. 2023) utilize GPT-3 and GPT-4 as the judge to evaluate the LLMs with incorporation of in-context learning and chain-of-thought strategies. The above methods rely heavily on a well-organized benchmark dataset. However, there have been some recent works focusing on data leakage of LLM reviews. (Zhu et al. 2024) proposed a method based on DAG to dynamically generate samples to evaluate LLM reasoning capabilities during the evaluation process. And our method is benchmark-free and has LLMs performing as the examiner to evaluate other models’ knowledge entailment and question answering capabilities.

# Data Leakage of LLM Evaluation

As the number of benchmarks for language model evaluation increases, data leakage emerges as an inevitable concern. However, there appear to be a limited number of studies addressing this issue. Sainz et al.(2023) propose a method to detect data breaches in closed-source LLMs, based on the premise that LLMs can recall training data and tend to reproduce similar content. Zhou et al.(2023b) conduct qualitative analysis of the impact of data leakage, which suggests that a data breach in one benchmark significantly enhances the LLM’s performance on that specific benchmark while diminishing its capabilities on other uncompromised benchmarks. Yang et al.(2023) propose a more accurate approach which employs an LLM detector with top- $\mathbf { \nabla } \cdot \mathbf { k }$ closest training data to determine if they match the test data. In contrast to these methods, which develop additional models for detecting data leakage during LLM evaluation with given benchmark datasets, our proposed method introduces a novel paradigm for LLM evaluation. It not only ensures the high quality of test questions but also inherently avoids data leakage.

# Methodology

# Overall Architecture

Figure 2 shows the overall structure of TreeEval. TreeEval organizes evaluations in a tree format, using components such as an Examiner, a Judge, and an Eval Controller. After the tree is built, an Aggregator compiles the scores. This framework allows for benchmark-free evaluation of LLMs through tree planning. Here’s how it works:

1. Session Setup: For each evaluation session, we choose two LLMs and start with an initial topic.   
2. Question Generation: The Examiner generates questions within this topic.   
3. Response Collection: These questions are sent to the LLMs, and their responses are collected.   
4. Response Evaluation: The Judge compares the responses and decides the winner for each question.   
5. Evaluation Control: If the responses are closely matched, the Eval Controller deepens the question. If a clear winner is found, the process moves to a new question. This follows a breadth-first search strategy, ensuring diverse and reliable questions.   
6. Score Aggregation: Finally, the Aggregator compiles the scores from all nodes in the tree to produce a comprehensive evaluation score.

TreeEval uses a tree structure to evaluate LLMs, minimizing the number of questions needed. Questions are generated automatically, preventing benchmark leakage. The root node starts with the session’s topic, and each node represents a question within that topic. Connections between nodes show how questions evolve. Deeper nodes indicate more similar abilities between the LLMs. Sibling nodes, derived from the same parent, cover different subtopics of the same main topic.

# TreeEval Modules

In this section, we provide more details of the components of the TreeEval and illustrate how to construct a tree for evaluation via these components.

Examiner. The examiner is a LLM-based module, which takes charge of generating exam questions that are able to cover diverse topics. Following (Bai et al. 2023), we predefine a set of topics as the scope of evaluation.

As the initialization of an evaluation session, we randomly sample a topic from the pre-defined topic set, which is denoted as $\mathcal { F C } _ { \mathrm { p r e - d e f i n e } }$ . Given a topic, the examiner is requested to craft a question that related to it via a prompt with the consideration of the coherence to the topic and the required format of the question. The detailed instruction is displayed in (Appendix Prompt for Examiner).

Once the session begins, we organize the follow-up questions in a tree structure. For simplify, we generally denote the follow-up topic at the $t$ -th time step as $C _ { t }$ . And the above procedure can be presented as:

$$
Q _ { t } = { \mathrm { E x a m i n e r } } ( C _ { t } ) .
$$

Subsequently, $Q _ { t }$ is utilized as the question to test the LLMs under review.

Judge. Previous studies (Wang et al. 2023a) conduct pairwise comparison and identify the superior responses among two evaluated LLMs, which has advantage in providing more nuanced assessment. Following these studies, we consult a pair of LLMs with the same question. The detailed instruction is displayed in (Appendix Prompt for Judge). After the responses have been produced via the LLMs, another LLM performs as the judge to the responses.

To ensure the reliability of the judge, we further conduct exchange evaluation, that is to switch the order of the responses. This procedure can be denoted as:

$$
\begin{array} { r } { S _ { t } ^ { 1 } = \mathrm { J u d g e } ( Q _ { t } , A _ { t } ^ { 1 } , A _ { t } ^ { 2 } ) ; } \\ { S _ { t } ^ { 2 } = \mathrm { J u d g e } ( Q _ { t } , A _ { t } ^ { 2 } , A _ { t } ^ { 1 } ) , } \end{array}
$$

where $A _ { t } ^ { 1 } , A _ { t } ^ { 2 }$ denote the responses from the pair of LLMs for $Q _ { t }$ . Each output judges the winner is $A _ { t } ^ { 1 }$ or $A _ { t } ^ { 2 }$ or a tie exits. If there is an agreement for $S _ { t } ^ { 1 }$ and $S _ { t } ^ { 2 }$ , We assign 2 score to the winner and 0 score to the loser to form $S _ { t }$ . Otherwise, we assign 1 to each model as $S _ { t }$ .

As the evaluation proceeds, we maintain a memory to record the history of the session, including the initial topic, historical questions as well as responses from the two evaluated LLMs. After $Q _ { t }$ has been responded, the history at the $t$ -th time step in the evaluation session can be denoted as $\mathcal { M } _ { t } = \{ C _ { 0 } , \bar { Q } _ { 0 } , A _ { 0 } ^ { 1 } , A _ { 0 } ^ { 2 } , . . . , C _ { t } , Q _ { t } , A _ { t } ^ { 1 } , A _ { t } ^ { 2 } \}$ . To involve the coherence of the flowing conversation and raise up rational follow-up questions, we prompt the examiner with the consideration of the history.

Eval Controller. The evaluation controller takes charge of the process of tree planning. Arbitrary generation of questions result in unorganized evaluation of LLMs with repeated questions and limited topics. To ensure the relevance and diversity of the generated questions, we have the following consideration: (1) To simulate the real-world interview of a certain subject, where the questions in an examination are mutually connected, we assume the generated followup question should be closely linked to its previous question via topics. For example, in Figure 2, inheriting from the root topic “technology and communication”, we can raise a

question sampled 𝑄0 𝐶0: Technology and Communication Examiner 小 topics 𝑆0: (0, 0) LLM1 vs LLM2 cqaunedsitidoantes ConEtvraol ler 𝑄1 𝑆𝐶:1(:1,AI1) 𝑄2 𝑆𝐶:2(:15,G1) ? 1 个 𝑄3 𝐶3: 𝑄4 𝐶4: 𝑄5 𝐶5: Human 𝑄6 𝐶6: 1 response1 score AI Ethics Accessibility Tools Machine Interaction applications i response2 Judge 𝑆3: (2, 0) 𝑆4: (1, 1) 𝑆5: (2, 0) 𝑆6: (0, 2) 1 TreeEval system Score Aggregator 𝑄7 𝐶7: Enhancing Acc𝑆ess:i(b2il,i t0y)in Communication

# Algorithm 1: Procedure of TreeEval

<html><body><table><tr><td colspan="2">1: Input FCpre-define;</td></tr><tr><td colspan="2">2:Initial t←O;Mt←@ 3:while Termination strategy is not satisfied do</td></tr><tr><td></td><td></td></tr><tr><td>4:</td><td>for Ct ∈ FCparent or Co EFCpre-define do Qt ← Examiner(Ct)</td></tr><tr><td>5: 6:</td><td></td></tr><tr><td></td><td>Qt ← arg maxqi∈Qt(Sim(Qi,Ct) - maxQk∈Mt Sim(Qi,Qk))</td></tr><tr><td>7: 8:</td><td>A¹,A² ← LLMs(Qt) St =Judge(Qt,A¹,A²)</td></tr><tr><td>9:</td><td>Mt ←MtU{Ct,Qt,A,A²}</td></tr><tr><td>10:</td><td></td></tr><tr><td>11:</td><td>FCt ←NER(A¹) UNER(A²) FCt←</td></tr><tr><td>12:</td><td>while|FCt|<k do</td></tr><tr><td>13:</td><td></td></tr><tr><td>14:</td><td>Ci ← arg maxoi∈Fct(Sim(C²,Ct))</td></tr><tr><td>15:</td><td>FCt ←FCtU{Ci} FCt←FCt\C</td></tr><tr><td>16:</td><td>forC ∈FCt do</td></tr><tr><td>17:</td><td>Sim(C𝑖,Ct) ← Sim(C,Ct)-Sim(C²,C²)</td></tr><tr><td>18:</td><td></td></tr><tr><td></td><td>t←t+1</td></tr></table></body></html>

question on $\cdot 5 G "$ that is relevant to the root topic and goes deeper. (2) The generated questions should not be repeated in the existing questions and we should ensure the diverse knowledge covered by the tree. For example, in Figure 2, under the topic $^ { \bullet } A I ^ { \prime }$ , we can come up with distinct but related sub-topics as siblings such as “AI Ethics”, “Accessibility Tools” and “Human Machine Interaction”.

Inspired by the Tree-of-Thought (Long 2023), where a controller produces the next thought step, we let Eval Controller arrange the follow-up evaluation according to $\mathcal { M } _ { t }$ . On the one hand, it prepares the follow-up topics $\mathcal { F } \mathcal { C } _ { t }$ based on $\{ C _ { t } , A _ { t } ^ { 1 } , A _ { t } ^ { 2 } \} \in \mathbf { \dot { \mathcal { M } } } _ { t } ^ { }$ for any of its child nodes in advance. On the other hand, it determines $Q _ { t + 1 }$ based on the $\mathcal { F } \mathcal { C } _ { t }$ and $\{ Q _ { 1 } , Q _ { 2 } , . . . , Q _ { t } \} \in \mathcal { M } _ { t }$ if the $t$ -th node is the parent node at $t + 1$ time step. We next describe the above two steps in detail:

vious question: $\mathcal { F C } _ { t } \sim \mathrm { N E R } ( A _ { t } ) ^ { 1 }$ . This works better when the Named Entity Recognition (NER) tool is built upon a LLM as some relevant entities could be revised via the model instead of solely being extracted (Wang et al. 2023b).

We sample candidate topics from both $A _ { t } ^ { 1 }$ and $A _ { t } ^ { 2 }$ then merge them together, which results in a set of candidate topics $\tilde { \mathcal { F } } \mathcal { C } _ { t }$ as the follow-up topics of $t$ -th node. However, this may produce some candidates that are repeated. To avoid this, we first measure the similarity between $C _ { t } ^ { i } \in \tilde { \mathcal { F } } \mathcal { C } _ { t }$ and $C _ { t }$ by computing the Cosine Similarity of their encoded vector representation (Zhang et al. 2023a), which is denoted as $\bar { \mathrm { S i m } } ( C _ { t } ^ { i } , C _ { t } )$ . Then, we iteratively push out $C _ { t } ^ { i }$ with the largest score. Next, we update the similarity scores of the rest topic $C _ { t } ^ { j }$ by subtracting the • Step One: Sample topics from the responses of the presimilarity score of $C _ { t } ^ { j } \in \tilde { \mathcal { F } } \mathcal { C } _ { t } \setminus C _ { t } ^ { i }$ and $C _ { t } ^ { i }$ , which is to decrease the possibility of retrieving similar topics. This procedure continues until we have pushed out $k$ topics as $\mathcal { F } \mathcal { C } _ { t }$ for the follow-up question generation.

• Step Two: If the question at $( t + 1 )$ -th time step is the child node of the node at $t$ -th time step, we generate questions based on the sampled topic via $Q _ { t + 1 } ^ { i } \sim$ Examiner $( C _ { t + 1 } )$ , where $C _ { t + 1 } \in \mathcal { F C } _ { t }$ . This could form a candidate question set $\tilde { \mathcal Q } _ { t + 1 }$ . Still, to avoid repetition of the generated questions and ensure a broad spectrum of inquiry questions, we conduct ranking for the candidate questions. Specifically, we measure the similarity between $Q _ { t + 1 } ^ { i } \in \tilde { \mathcal { Q } } _ { t + 1 }$ and $C _ { t + 1 }$ via Cosine Similarity. Then we push out $Q _ { t + 1 } ^ { i }$ with the largest similarity score of $\mathrm { S i m } ( Q _ { t + 1 } ^ { i } , C _ { t + 1 } )$ and the least similarity score of arg ${ \mathrm { m i n } } _ { Q _ { k } \in { \mathcal { M } } _ { t } } \operatorname { S i m } ( Q _ { t + 1 } ^ { i } , Q _ { k } )$ .

Termination Strategy. We use the following criteria to stop generating questions for a topic:

• Distinctive Question: If a question distinguishes the capabilities of the two LLMs or if there is no tie, we terminate further exploration of the current node. • Dominant Sibling: After generating sibling nodes for a parent, if most of the sibling nodes yield the same result, we stop searching further for these siblings. • Maximum Depth: A maximum depth $T$ is set for the tree search. Once reached, we terminate the search for the current topic.

The search stops when all nodes meet these conditions. The full process is outlined in Algorithm 1.

# Score Aggregator

After we have constructed the multiple trees across $\mathcal { F C } _ { \mathrm { p r e - d e f i n e } }$ , where the nodes in each tree implies the win-rate between two LLMs under review towards a specific topic. To yield a final win-rate result, we aggregate the scores of these constructed trees. However, it is irrational to consider all the nodes in a tree equally due to their different features and result scores. Specifically, we take the following aspects of $t$ -th node in a tree into account when we aggregate their scores:

• Distance to the root node. Based on the principle of an evaluation session, a longer distance to the root node indicates a more intensive competition between the evaluated LLMs and the more important the node is. This suggests that the winner only has a marginal advantage over the other one. Therefore, we define one aspect of an important node as $\begin{array} { r } { w _ { t } ^ { \mathrm { r o o t } } = \frac { 1 } { d } } \end{array}$ , where $d$ is the distance from the $t$ -th node to the root node in a tree. • Origin of the topic. As the topic is derived from the responses in its parent node, a node inherited the topic generated from responses of the losing LLM is more important considering it is more likely to balance the situation. Hence, we define one aspect of an important node as:

$$
w _ { t } ^ { \mathrm { t o p i c } } = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { T o p i c ~ o r i g i n a t e d ~ f r o m ~ t h e ~ l o s e r } } } \\ { 0 . 5 } & { { \mathrm { O t h e r w i s e } } } \end{array} \right. }
$$

• Variance of the sibling nodes. The disagreement of the evaluation of the sibling node may implicit a potential randomness derived from the topic. So we define the sibling consensus as:

$$
w _ { t } ^ { \mathrm { t o p i c } } = \frac { 1 } { \sigma ^ { 2 } + 1 } ,
$$

where $\sigma$ is the variance of the score of its sibling nodes.

Considering the above aspects, we compute the final importance weights of $t$ -th node as:

$$
w _ { t } = w _ { t } ^ { \mathrm { r o o t } ^ { \alpha } } \cdot w _ { t } ^ { \mathrm { t o p i c } ^ { \beta } } \cdot w _ { t } ^ { \mathrm { s i b l i n g } ^ { \gamma } } ,
$$

where $\alpha , \beta$ , and $\gamma$ are hyper-parameters indicating the relative importance of these aspects. As a result, we sum up the $\boldsymbol { w } _ { t }$ multiplying with the win-rate of an LLM and devide the total evaluation questions to obtain its final scores:

$$
S = \frac { 1 } { N } \sum _ { \substack { i \mathrm { \tiny ~ \cdot t h ~ T r e e ~ f r o m ~ } \mathcal { F } \mathcal { C } _ { \mathrm { p r e - d e f i n e } } ; } } \sum _ { \substack { t \mathrm { \tiny ~ t h ~ n o d e ~ i n ~ } i \mathrm { \tiny ~ \cdot t h ~ T r e e } } } w _ { t } \cdot S _ { t } ,
$$

where $N$ is the sum of node weights in the evaluation session and $S$ is normalized.

# Experiments

# Experimental Setup

Evaluated LLMs. We evaluated the following open-source LLMs, including two 7B models, two 13B models, and two 33B models. These models are either derived from LLaMA (Touvron et al. 2023; et al. 2023a) or trained from scratch using the LLaMA architecture, and some show similar performance according to the open-source LLM leaderboard2.

• Yi-34B-Chat (AI 2024) is a product from 01.AI, built on a large-scale multilingual dataset.   
• Xwin-LM-13B-V0.1 (Team 2023b) is based on LLaMA2-13B and tuned through SFT and RLHF.   
• Mistral-7B-Instruct-v0.2 (Jiang et al. 2023) is tuned on the Mistral-7B model, built with the LLaMA architecture.   
• Vicuna-33B-v1.3 (Zheng et al. 2023a) originates from LLaMA-33B and is fine-tuned using dialogues from ShareGPT.   
• WizardLM-13B-V1.2 (Xu et al. 2023) is based on LLaMA2-13B and fine-tuned with enhanced instruction data using Evol-Instruct.   
• Zephyr-7B-beta (Tunstall et al. 2023b) is derived from Mistral-7B and aligned using SFT and DPO methods.

Comparable Evaluation Methods. We compare TreeEval with several existing methods, including:

• Benchmark Paradigm: – MMLU (Hendrycks et al. 2021) – Big-Bench Hard (BBH) (Suzgun et al. 2022)

2https://tatsu-lab.github.io/alpaca eval/ • LLMs as Judges: – AlpacaEval and AlpacaEval2.0 (Li et al. 2023b) – MT-Bench (Zheng et al. 2023a)

Table 1: Comparison of LLMs across various evaluation methods. $\stackrel { \cdot } { \star } \stackrel { \cdot } { }$ denotes we re-implement MMLU and BBH benchmarks (Chia et al. 2023), calculating results in both 5-shot and 3-shot contexts. “ ” denotes we directly take results from the respective leader-boards from MT-bench, AlpacaEval, and AlpacaEval2.0. “#Q” denotes the number of questions used for evaluation. We report the correlation of rankings obtained through different methods with those from AlpacaEval2.0, using $\tau$ for the Kendall correlation coefficient (KENDALL 1938) and $\rho$ for the Spearman correlation coefficient (Spearman 1904).   

<html><body><table><tr><td>LLMs</td><td>MMLU*</td><td>BBH*</td><td>AlpacaEvalt</td><td>MT-bench†</td><td>AlpacaEval2.0+</td><td colspan="2">TreeEval(Ours)</td></tr><tr><td></td><td>Acc</td><td>Acc</td><td>Win-Rate</td><td>score</td><td>一 Win-Rate</td><td>#Q</td><td>Score(var)</td></tr><tr><td>Mistral-7B-Instruct-v0.2</td><td>70.6</td><td>46.4</td><td>92.78</td><td>8.30</td><td>14.72</td><td>1</td><td>2.50(0.000)</td></tr><tr><td>Yi-34B-Chat</td><td>73.46</td><td>71.74</td><td>94.08</td><td>8.65</td><td>27.19</td><td>31.67</td><td>3.48(0.011)</td></tr><tr><td>xwinlm-13b-v0.1</td><td>56.6</td><td>37.58</td><td>91.76</td><td>7.34</td><td>17.43</td><td>62.33</td><td>2.67(0.000)</td></tr><tr><td>WizardLM-13B-V1.2</td><td>52.7</td><td>40.12</td><td>89.17</td><td>7.2</td><td>12.03</td><td>44.67</td><td>1.10(0.070)</td></tr><tr><td>zephyr-7b-beta</td><td>61.4</td><td>42.72</td><td>90.60</td><td>7.34</td><td>一 10.99</td><td>45.67</td><td>2.19(0.003)</td></tr><tr><td>Vicuna-33b-v1.3</td><td>59.2</td><td>52.0</td><td>88.99</td><td>一 7.12</td><td>12.71</td><td>41.33</td><td>1.61(0.044)</td></tr><tr><td>Average #Q↓</td><td>14,079</td><td>6,511</td><td>804</td><td>一 80</td><td>804</td><td>45.1</td><td>1</td></tr><tr><td>p↑</td><td>0.43</td><td>0.37</td><td>0.71</td><td>0.61</td><td>1.0</td><td>1</td><td>0.83</td></tr><tr><td>T个</td><td>0.33</td><td>0.33</td><td>0.47</td><td>一 0.41</td><td>1.0</td><td>1</td><td>0.73</td></tr></table></body></html>

AlpacaEval and AlpacaEval2.0 use ChatGPT as the judge for single-turn interactions, while MT-Bench focuses on multi-turn dialogues.

Implementation Details. We use $\mathtt { G P T - 4 - 0 6 1 3 }$ as the examiner, deployed with FastChat (Zheng et al. 2023a), with a temperature of 1 for varied question generation. We set $T$ and $k$ to 3, and $\alpha , ~ \beta$ , and $\gamma$ to 1, 1, and 0.4, respectively. To ensure stability, we repeat experiments three times, average the scores, and report the variance. Mistral-7B-Instruct $- \mathtt { v } 0 . 2$ is used as the reference model for pairwise comparison, given its moderate performance on public leaderboards.

# Performance of TreeEval

We present the performance of TreeEval in Table 1, with the following key observations:

• High Correlation: Among all comparable evaluation methods, our method achieves the highest correlation with AlpacaEval2.0 rankings in both $\rho$ and $\tau$ . This high consistency demonstrates the reliability of our method, as AlpacaEval2.0 is a recognized LLM evaluation leaderboard. • Evaluation Efficiency: Our method completes the evaluation with an average of only 45 questions, while other methods require significantly more. This shows that our approach is efficient in evaluating LLMs. • Reference Comparison: Using Mistral-7B-Instruct-v0.2 as the reference, we observe that the larger the gap between the evaluated LLM and the reference, the fewer the test questions generated. This indicates that tree planning effectively aligns with our expected goal.

Further pairwise correlation analysis in the appendix confirms that TreeEval has high correlation with AlpacaEval2.0.

# Further Analysis

We further analyze to verify TreeEval’s effect.3

More powerful models. To demonstrate the performance of our approach in comparison with more powerful models, Table 2 presents the results of using Yi-34B-Chat as the baseline. We selected some of the most advanced open-source models currently available for testing against Yi-34B-Chat. Our TreeEval achieved performance closest to that of AlpacaEval2. More model results can be found in the (appendix More model results).

Pairwise Comparison for Different Model Pairs. We vary the reference model for pairwise comparison, with results shown in Table 3. Selecting an appropriate baseline model is crucial for our evaluation strategy. We chose Mistral-7B-Instruct $- \mathtt { v } 0 . 2$ as the baseline, as it offers a balanced performance for fair comparisons. Interestingly, even a randomly selected baseline model yields rankings similar to those from a thorough pairwise evaluation. This suggests that an initial ranking can be set with a random baseline, then refined efficiently using bubble sort. Since the initial order closely matches the final ranking, the refinement process has an $O ( n )$ complexity, improving both precision and efficiency.

Ablation Studies. As we can see in Table 4, changing BFS search to DFS search dramatically increases the number of questions but decreases the performance. This is because DFS search generates the child node first rather than the sibling node such that the influence of sibling node will be neglected in both question generation and termination identification procedures. Removing step one, which indicates skip the topic generation step, decreases the performance. This indicates the significant role of identifying the topic for question generation. When we iteratively remove the scores in aggregator, we observe general performance drop on $\tau$ . This indicates that all the scores in the aggregator are important in producing a comprehensive score.

Table 2: Results of More Powerful Models. Given the strong capabilities of the evaluated models, we use Yi-34B-Chat as the baseline and exclude the AlpacaEval benchmark, which is relatively simple for these models.   

<html><body><table><tr><td>LLMs</td><td>MMLU*</td><td>BBH*</td><td>MT-bench†</td><td>AlpacaEval2.0+</td><td colspan="2">TreeEval(Ours)</td></tr><tr><td></td><td>Acc</td><td>Acc</td><td>score</td><td>Win-Rate</td><td>#Q</td><td>Score(Var)</td></tr><tr><td>Yi-34B-Chat</td><td>73.5</td><td>71.7</td><td>8.65</td><td>27.1</td><td>1</td><td>2.50(0.000)</td></tr><tr><td>Qwen1.5-110B-Chat</td><td>80.4</td><td>74.8</td><td>8.88</td><td>33.7</td><td>42.67</td><td>4.03(0.110)</td></tr><tr><td>Meta-Llama-3-70B-Instruct</td><td>82.0</td><td>81.3</td><td>8.92</td><td>34.4</td><td>36.33</td><td>3.82(0.128)</td></tr><tr><td>Qwen1.5-72B-Chat</td><td>75.6</td><td>65.5</td><td>8.61</td><td>36.6</td><td>31.33</td><td>3.45(0.089)</td></tr><tr><td>Mixtral-8x7B-Instruct-v0.1</td><td>70.6</td><td>57.3</td><td>8.30</td><td>23.7</td><td>44.67</td><td>2.02(0.027)</td></tr><tr><td>vicuna-33b-v1.3</td><td>59.2</td><td>52.0</td><td>7.12</td><td>12.7</td><td>21.33</td><td>0.35(0.033)</td></tr><tr><td>p↑</td><td>0.82</td><td>0.71</td><td>0.71</td><td>1.0</td><td>1</td><td>0.94</td></tr><tr><td>T↑</td><td>0.73</td><td>0.60</td><td>0.60</td><td>1.0</td><td>1</td><td>0.86</td></tr></table></body></html>

<html><body><table><tr><td>Model</td><td>Yi-3atB</td><td>-1winv01</td><td>-IMisrral-76.2</td><td>-vic-v1.3</td><td>-izBav1.2</td><td>zephera</td></tr><tr><td>Yi-34B-Chat</td><td>1</td><td>1.88(0.400)</td><td>1.52(0.010)</td><td>2.1(0.070)</td><td>1.21(0.076)</td><td>1.75(0.143)</td></tr><tr><td>Xwin-LM-13B-V0.1</td><td>3.12(0.400)</td><td>1</td><td>2.33(0.000)</td><td>1.53(0.403)</td><td>1.57(0.109)</td><td>2.41(0.000)</td></tr><tr><td>Mistral-7B-Instruct-v0.2</td><td>3.48(0.010)</td><td>2.67(0.000)</td><td>1</td><td>1.61(0.044)</td><td>1.10(0.070)</td><td>2.19(0.003)</td></tr><tr><td>vicuna-33b-v1.3</td><td>2.9(0.070)</td><td>3.47(0.403)</td><td>3.39(0.044)</td><td>1</td><td>2.01(0.374)</td><td>3.7(0.071)</td></tr><tr><td>WizardLM-13B-V1.2</td><td>3.79(0.076)</td><td>3.43(0.109)</td><td>3.90(0.070)</td><td>2.99(0.374)</td><td>1</td><td>3.94(0.069)</td></tr><tr><td>zephyr-7b-beta</td><td>3.25(0.143)</td><td>2.59(0.000)</td><td>2.81(0.003)</td><td>1.3(0.071)</td><td>1.06(0.069)</td><td>1</td></tr></table></body></html>

Table 3: Our result for each model pairs. The elements in this table represent the scores obtained by comparing models using treeEval, with the column model being compared against the row model.

Table 4: Ablation study on TreeEval.   

<html><body><table><tr><td>Methods</td><td>#Q</td><td>p</td><td>T</td></tr><tr><td>TreeEval</td><td>45.1</td><td>0.83</td><td>0.73</td></tr><tr><td>BFS →DFS</td><td>149.4</td><td>0.37</td><td>0.33</td></tr><tr><td>w/o Step One</td><td>49.3</td><td>0.31</td><td>0.2</td></tr><tr><td>w/o wroot</td><td>45.1</td><td>0.77</td><td>0.6</td></tr><tr><td>w/o wtopic</td><td>45.1</td><td>0.77</td><td>0.6</td></tr><tr><td>w/o wsibling</td><td>45.1</td><td>0.71</td><td>0.47</td></tr></table></body></html>

Case studies are presented in (Appendix Case Studies).

# Conclusions

In this paper, we introduce TreeEval, a benchmark-free evaluation approach for LLMs with tree planning, which automatically controls the evaluation process with tree planning. We experimentally verify that TreeEval can not only produce reliable evaluation results without data leakage but also enhance discrimination between similarly performing LLMs.

# Limitations

Using LLMs like GPT-4 as judges introduces potential data leakage risks due to biases in their pre-training data. This can be mitigated by selecting neutral evaluators independent of the assessed models’ training data or randomly rotating evaluators to reduce bias.

While GPT-4 is a powerful examiner, it has limitations, particularly in areas outside its expertise. This can be addressed by providing more contextual guidance during evaluations. In the future, training specialized evaluators to extract questions from document repositories and assess comprehension could ensure more accurate, domain-specific evaluations.

# Ethics Statement

Although we prioritize the security of the LLMs we use during evaluations, striving to employ aligned LLMs with higher safety standards, and endeavor to ensure that LLM outputs adhere to ethical and legal requirements, limitations arising from model size and probabilistic generation paradigms may lead to various unexpected outputs. These could include questions or responses containing biases, discrimination, or other harmful content. Please refrain from disseminating such content.