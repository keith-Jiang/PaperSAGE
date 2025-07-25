# MaFeRw: Query Rewriting with Multi-Aspect Feedbacks for Retrieval-Augmented Large Language Models

Yujing Wang1,2, Hainan Zhang1,2\*, Liang Pang4, Binghui Guo2, Hongwei Zheng3, Zhiming Zheng1,2

1Beijing Advanced Innovation Center for Future Blockchain and Privacy Computing 2 School of Artificial Intelligence, Beihang University, China 3Beijing Academy of Blockchain and Edge Computing, China   
4Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China {wangyujing, zhanghainan}@buaa.edu.cn

# Abstract

In a real-world RAG system, the current query often involves spoken ellipses and ambiguous references from dialogue contexts, necessitating query rewriting to better describe user’s information needs. However, traditional contextbased rewriting has minimal enhancement on downstream generation tasks due to the lengthy process from query rewriting to response generation. Some researchers try to utilize reinforcement learning with generation feedback to assist the rewriter, but this sparse rewards provide little guidance in most cases, leading to unstable training and generation results. We find that user’s needs are also reflected in the gold document, retrieved documents and ground truth. Therefore, by feeding back these multi-aspect dense rewards to query rewriting, more stable and satisfactory responses can be achieved. In this paper, we propose a novel query rewriting method MaFeRw, which improves RAG performance by integrating multi-aspect feedback from both the retrieval process and generated results. Specifically, we first use manual data to train a T5 model for the rewriter initialization. Next, we design three metrics as reinforcement learning feedback: the similarity between the rewritten query and the gold document, the ranking metrics, and ROUGE between the generation and the ground truth. Inspired by RLAIF, we train three kinds of reward models for the above metrics to achieve more efficient training. Finally, we combine the scores of these reward models as feedback, and use PPO algorithm to explore the optimal query rewriting strategy. Experimental results on two conversational RAG datasets demonstrate that MaFeRw achieves superior generation metrics and more stable training compared to baselines.

What is San Juan's history? In 1509, Juan Ponce de Leon moved to a site 四 which was called at that time Puerto Rico. When did the name get added? In 1521, the name San Juan was added, and 四 the newer settlement was given its formal name of San Juan Bautista de Puerto Rico.   
①Rewrite How popular in the present? 四 main spot for ... tourism. How far from Puerto Rico? T5 Rewriter How far from puerto rico from Puerto Rico reward MaFeRw How far is Old San Juan from Puerto Rico?   
N   
GR ...Old San Juan is located on..., about 35 miles Gold Doc $( 5 6 ~ \mathsf { k m } )$ ) from the east end of Puerto Rico... sim (1)  ... Old San Juan is located on...about 35 miles $( 5 6 ~ \mathsf { k m } )$ from the east end of Puerto Rico.. Retrieved (2) ...the settlement was moved to a site which Docs was called at the time "Puerto Rico"... rank (3)  ...Every alleyway in Old San Juan offers a   
③Generate glimpse into a vibrant past... Old San Juan is located on a small and narrow Ground island which lies along the north coast, about 35 Truth miles from the east end of Puerto Rico. Generated Old San Juan is located on a small island, about ROUGE Results 35 miles from the east end of Puerto Rico.

# Code — https://github.com/TAP-LLM/MaFeRw

# Introduction

Retrieval-augmented generation (RAG) effectively addresses the issue of factual inaccuracy in large language models (LLMs) by integrating relevant retrieved information. It is widely used in various fields such as QA systems (Du and Ji 2022; Siriwardhana et al. 2023; Sharma et al. 2024), content generation (Zhao et al. 2024; Su et al.

2024; Wang et al. 2023), and virtual agents (Yang, Yue, and He 2023; Schick et al. 2023; Zhang 2023). Real-world RAG systems often rely on multi-turn dialogues, where queries include spoken ellipses and ambiguous references from dialogue contexts (Mao et al. 2023b), making it challenging for RAG systems to accurately understand user intent (Ye et al. 2024). In light of this challenge, effective query rewriting is necessary to better describe user information needs and ensure the accuracy of retrieval and generation in RAG.

Currently, RAG mainly performs context-based query rewriting to improve the quality of response generation, but the resulting enhancement is minimal due to the lengthy process from query rewriting to response generation. Although context-based query rewriting can directly affect the retrieval results, this impact is difficult to transmit to the final generated results through this loose-coupling retrieval and generation framework of RAG. As shown in Figure 1, the T5 rewriter (Lin et al. 2020) trained on context-based rewrite dataset fails to effectively capture the user’s intent. To address this issue, Ma et al. try to use reinforcement learning (RL) with generation feedback to assist the rewriter. They optimize the rewriter with policy gradient and use the accuracy and ’hit’ rate of RAG inference’s answers to form the reward. However, this reward is sparse and provides little guidance in most cases, resulting in unstable training and generation results of their model. Additionally, it is inefficient to directly use the full RAG inference to obtain RL feedback. Because RL typically requires a large number of training steps to achieve good results, and each inference in RAG takes a long time. Thus, if a stable and efficient method for training the rewriter can be designed, it will undoubtedly further enhance the effectiveness of RAG.

We find that user information needs are also reflected in the gold document, retrieved documents and ground truth, which allows for the design of multi-aspect rewards to provide more dense feedback signals. As illustrated in Figure 1, the current query ’How far from Puerto Rico’ omits ’Old San Juan’ from the previous context. The user intent ’Old San Juan’s distance from Puerto Rico’ is evident not only in the context but also in the gold document, retrieved documents, and ground truth. If query rewriting can leverage this information, it can better assist the model in generating correct answers. Therefore, by feeding multi-aspect dense rewards from the gold document, ranking measures, and generation metrics back to rewriting, more stable and satisfactory responses can be achieved.

In this paper, we propose MaFeRw, a novel query Rewriting method, to improve RAG performance by integrating Multi-aspect Feedback from both the retrieval process and generated results. Specifically, we first use manual data to train a T5 model (Raffel et al. 2020) as the rewriter’s initialization. Next, we design three metrics as reinforcement learning feedback: the similarity between the rewritten query and the gold document, the ranking metrics of similarity between retrieved documents and the ground truth, and ROUGE between the generated response and the ground truth. Inspired by RLAIF (Lee et al. 2023), to achieve more efficient training, we collect datasets based on these metrics and train three kinds of reward models for them. Additionally, we use ROUGE scores of model-rewritten queries and manual-rewritten queries as the fourth feedback to measure the rewriter’s performance. Finally, we combine scores of three reward models and the rewritten ROUGE as feedback, and use PPO algorithm (Schulman et al. 2017; Ziegler et al. 2020) to explore the optimal query rewriting strategy.

Experimental results on two conversational RAG datasets demonstrate that MaFeRw achieves superior generation metrics compared to baselines. Further analysis shows that multi-aspect dense rewards provide a more stable training process and generation results than single reward, validating the stability and transferability of MaFeRw.

The innovations of this paper are as follows:

• We find that dense reward feedback can lead to more stable and satisfactory generation results than single reward, and design four rewards from multiple aspects, such as gold document, retrieved documents and ground truth. • To achieve more efficient RL training, we collect the feedback signals and train corresponding reward models, rather than using full RAG inference like the baseline. • Experimental results on two RAG datasets demonstrate that MaFeRw achieves superior generation metrics and more stable training than baselines.

# Related Work

In dialogue systems, user utterances often contain omissions and ambiguous references (Fang et al. 2022; Zhang et al. 2019). Therefore, a rewriting model is needed to resolve these ambiguities in the current query and recover missing elements (e.g., anaphora) from the context to better articulate the user’s information needs. Current research on query rewriting mainly focuses on conversational search tasks, conversational QA tasks, and RAG tasks.

In conversational search tasks, researchers try to explore effective query rewriting methods to enhance the retrieval process. Some studies focus on extracting user intents from the dialogue context. For example, Lin et al. use human rewrites as labels to train a sequence-to-sequence model as the rewriter. Qian and Dou propose query rewriting explicitly highlights relevant terms in the query context, facilitating contextual modeling and thereby improving the quality of the learned contextual query embeddings. In (Ye et al. 2023), LLM is used as the rewriter and is prompted to complete a ”rewrite-then-edit” process based on the conversation history. Others incorporate retrieval feedback to enhance rewriting for retrieval tasks. Mo et al. and Mao et al. both take the feature similarity between the query and the target passage into the optimization objective when training the rewriting model and Wu et al. use the retrieval effect as a reward for reinforcement learning. The rewritten queries are encouraged to achieve better retrieval performance.

In conversational QA tasks, mainstream query rewriting studies (Kim et al. 2021; Chen et al. 2022) focus on feeding the generated results back into the rewriting model to enhance answer accuracy. Kim et al. train the QA model by regularizing the consistency of generated answers so that the answers predicted by the model according to the original questions were similar to the answers generated according to the rewritten questions. Chen et al. use the effect of generated answers as a reward for reinforcement learning, to improve the effect of rewriting on the final generated results.

In conversational RAG task, Ma et al. firstly create a pseudo-dataset for supervised warm-up training, then further optimize the rewriter with a policy gradient RL framework, using the accuracy and ’hit’ rate of LLMs’ answers to form the reward.

However, for RAG tasks, focusing solely on the retrieval process results in rewrites unsuitable for prompting LLMs during generation. Conversely, considering only the feedback from the final generated results can lead to unstable training of the rewriting model due to the sparse reward signal. Therefore, we propose MaFeRw, which combines multi-aspect reward feedbacks to produce more stable and satisfactory responses.

# Problem Definition

In RAG, each dialogue history $h ^ { k }$ contains a sequence of (query, answer) pairs $h ^ { k } = ( \bar { q } ^ { 1 } , a ^ { 1 } , . . . , q ^ { k - 1 } , a ^ { k - \bar { 1 } } )$ , where $q ^ { i }$ and $a ^ { i }$ denote the query and the system generation of the $i$ -th turn. A conversational query $q ^ { i }$ can be elliptical and ambiguous. Given the dialogue history $h ^ { k }$ and the current query $q ^ { k }$ , a query rewriting model $R _ { \theta }$ with parameters $\theta$ rewrites $q ^ { k }$ to $\grave { q } _ { R } ^ { k }$ :

$$
q _ { R } ^ { k } = R _ { \theta } ( h ^ { k } , q ^ { k } ) .
$$

The rewritten query $q _ { R } ^ { k }$ is then fed into an off-the-shelf RAG system’s retriever to search for relevant documents $D ^ { k } = \left\{ d _ { 1 } ^ { k } , . . . , d _ { l } ^ { k } \right\}$ , where $d _ { i } ^ { k }$ denotes the $i$ -th retrieved document of the $k ^ { t h }$ turn, $l$ is the number of retrieved documents. The current gold document is represented as $d _ { + } ^ { k }$ . Subsequently, the rewritten query $q _ { R } ^ { k }$ , the retrieved documents $D ^ { k }$ , and the $( q ^ { k - 1 } , a ^ { k - 1 } )$ pair from the previous turn are combined as input to prompt the LLM to generate more satisfactory results. The $\mathring { ( } q ^ { k - 1 } , a ^ { k - 1 } )$ pair is used to instruct the LLM to mimic the style of the dataset by explicitly prompting the LLM to ’follow the style of the example responses in the context $( q ^ { k - 1 } , a ^ { k - 1 } ) ^ { \prime }$ . In this paper, we aim to train the query rewriting model $R _ { \theta }$ to make the rewritten query $q _ { R } ^ { k }$ better describe user needs and ensure the accuracy of retrieval and generation in RAG.

# Approach

# Framework Overview

For MaFeRw, the rewriter is trained to transform the current query $q ^ { k }$ into $q _ { R } ^ { k }$ utilizing the dialogue history $h ^ { k }$ , which enables the RAG system to generate more satisfactory results. Firstly, we use manual rewriting data to train a T5 model as the rewriter’s initialization. Secondly, we design three metrics as RL feedback: the similarity between the rewritten query $q _ { R } ^ { k }$ and the gold document $d _ { + } ^ { \check { k } }$ , the ranking metrics of similarity between retrieved documents $D ^ { k }$ and the ground truth $G ^ { k }$ , and ROUGE between the generation $g ^ { k }$ and the ground truth $G ^ { k }$ . Inspired by RLAIF, we train three kinds of reward models for the above metrics on three corresponding paired data sets, as shown in Figure 2a. Considering the obvious improvement of manual rewriting in the generation, ROUGE between manual rewrites and model rewrites is used as another feedback. Finally, we combine scores of these reward models and the rewritten ROUGE to feedback PPO training, exploring the optimal query rewriting strategy, as shown in Figure 2b.

# Rewriter Initialization

To equip the rewriter with fundamental rewriting capabilities, we train the pre-trained T5-base model on datasets containing manually rewritten queries. The query rewriter con

current query dialogue history current query dialogue history <s> <s> ... rewriter rewriter .</s> 牛 </s>   
rmewodareld rewritten query rewritten query ra d similarity </s> positive retrieved manual rewrite doc docs   
reward ranking metric   
model $r _ { \psi } ^ { D }$ reward reward reward ground generated model model model metric truth result r d rD ra G mq   
reward   
model ROUGE $r _ { \psi } ^ { G }$ reward (a) (b)

catenates the current query $q ^ { k }$ with the dialogue history $h ^ { k }$ as input. Similar to the approach used by (Wu et al. 2022), a separator token ”[SEP]” is inserted between each interaction, and dialogue turns are connected in reverse order, as

$$
I = [ \mathrm { C L S } ] q ^ { k } [ \mathrm { S E P } ] a ^ { k - 1 } [ \mathrm { S E P } ] q ^ { k - 1 } \cdot \cdot \cdot q ^ { 1 } [ \mathrm { S E P } ] .
$$

When $q _ { + }$ is a manual-rewritten query, the objection of initialization is to optimize parameters $\theta$ of the rewriter $R _ { \theta }$ by minimizing the cross-entropy loss between the model’s prediction $q _ { R }$ and $q _ { + }$ :

$$
\begin{array} { r } { \mathcal { L } _ { i n i t } = - y _ { q _ { + } } \log y _ { q _ { R } } , } \end{array}
$$

where $y _ { q _ { + } }$ is the one-hot vector of $q _ { + }$ and $y _ { q _ { R } }$ is the distribution over tokens in $q _ { R }$ predicted by the rewriter. The rewriter after initialization is represented as $R _ { \theta } ^ { 0 }$ , which defines a probabilistic policy for this task.

# Multi-aspect Feedbacks

Since user information needs are also reflected in gold documents, retrieved documents, and ground truth, we design three metrics as RL feedback. For simplicity, we omit the superscript $k$ of $q ^ { k } , q _ { R } ^ { k } , d _ { + } ^ { k } , D ^ { k } , G ^ { k }$ and $g ^ { k }$ in the rest of the paper if not specified.

(1) The similarity between $q _ { R }$ and $d _ { + }$ In RAG, the retriever converts the query and documents into dense vectors, calculates similarity scores between the query vector and document vectors, and retrieves the top- $\mathbf { \nabla } \cdot \mathbf { \vec { l } }$ most relevant documents. To ensure the rewritten query $q _ { R }$ retrieves the gold document $d _ { + }$ more effectively, the cosine similarity (CS) between the dense vector $v _ { q _ { R } }$ of $q _ { R }$ and the dense vector $v _ { d _ { + } }$ of $d _ { + }$ is used as a metric $m _ { d _ { + } }$ which provides retrieval feedback to the rewriter, i.e., $m _ { d _ { + } } ^ { \textrm { \tiny { ( m _ { d _ { + } } ) } } } = \mathrm { C S } ( v _ { q _ { R } } , v _ { d _ { + } } )$ .

(2) The ranking metric of similarity between $D$ and $G$ Since user needs are embedded in the ground truth $G$ , and the order of retrieved documents $D$ impacts generation quality, we hope the rewritten query can assist the retrieval model in prioritizing the document that exhibits high semantic relevance to the ground truth. Therefore, we propose a ranking metric $m _ { D }$ using $G$ and $D = \{ d _ { 1 } , \cdot \cdot \cdot , d _ { l } \}$ . Specifically, we transform the ground truth $G$ into a dense vector $v _ { G }$ using the retrieval embedding model. Then the CS value is calculated between $v _ { G }$ and dense vectors of retrieved documents $\{ v _ { d _ { i } } \} _ { i = 1 } ^ { l }$ . For each $d _ { i }$ , let $1 / i$ be its rank score. The metric $m _ { D }$ is computed by summing the product of each document’s rank score and its CS value with $v _ { G }$ , formulated as $m _ { D } = \Sigma _ { i = 1 } ^ { l } 1 / i \cdot \mathrm { C S } ( v _ { G } , v _ { d _ { i } } )$ . A higher $m _ { D }$ indicates that the $d _ { i }$ with greater similarity to $G$ is ranked higher, thereby facilitating the generation of better responses.

(3) ROUGE between $g$ and $G$ To ensure that the generated output $g$ produced by RAG closely aligns with the ground truth $G$ , we use ROUGE score between the generation $g$ and the ground truth $G$ as the metric $m _ { G }$ .

(4) ROUGE between $q _ { R }$ and $q _ { + }$ Considering the obvious improvement of manual rewrites in the generation, we also use ROUGE between the manual rewrite $q _ { + }$ and model rewrite $q _ { R }$ as a reference metric $m _ { q }$ .

# Reward Model

The end-to-end RL approach without a reward model demands extensive training time due to the prolonged inference time of RAG and the large number of training steps required. Incorporating the complete RAG process into RL led to training durations of several days, severely limiting tuning and flexibility in later stages. To address this issue, reward models were employed to evaluate rewritten queries, effectively decoupling the evaluation from RAG process and enabling more efficient training. Inspired by RLAIF, we train three types of reward models for the aforementioned metrics on three corresponding paired datasets. The reward model is based on the T5-base model with a value head, possessing significantly fewer parameters than the generation model Llama2-13b. During RL training, the reward is derived from scores of reward models rather than using the complete RAG process, thereby reducing RAG inference time.

Specifically, for the given dialogue history $h$ and the current query $q$ , we pair the corresponding manual rewrite $q _ { m }$ , rewrite $q _ { T 5 }$ generated by the initialized rewriter, and rewrite $q _ { s }$ obtained through sampling. All rewrites are then put into the RAG system to calculate values of $m _ { d _ { + } } , m _ { D }$ and $m _ { G }$ . For each metric, we compare rewrites in every pair and label the one with the higher metric value as ’chosen’ and the other as ’rejected’, creating a paired dataset of (chosen, rejected) rewrites. Taking the paired dataset of metric $m _ { G }$ as an example, if $m _ { G } ( q _ { m } ) > m _ { G } ( q _ { T 5 } ) > m _ { G } ( q _ { s } )$ , then for the dialogue history $h$ and current query $q$ , three pairs of data are collected:

$$
\begin{array} { r l r } {  { \{ ( c h o s e n : [ h + q , q _ { m } ] , r e j e c t e d : [ h + q , q _ { T 5 } ] ) , } } \\ { \quad ( c h o s e n : [ h + q , q _ { m } ] , r e j e c t e d : [ h + q , q _ { s } ] ) , } & { } \\ & { } & { ( c h o s e n : [ h + q , q _ { T 5 } ] , r e j e c t e d : [ h + q , q _ { s } ] ) \} , } \end{array}
$$

here $\vec { \mathbf { \nabla } } \vec { \mathbf { \nabla } } \cdot \vec { \mathbf { \nabla } } \vec { \mathbf { \nabla } } \cdot \vec { \mathbf { \nabla } } \vec { \mathbf { \nabla } } \cdot \vec { \mathbf { \nabla } } \vec { \mathbf { \nabla } } \cdot \vec { \mathbf { \nabla } } \cdot \vec { \mathbf { \nabla } } \cdot \vec { \mathbf { \nabla } } \cdot \vec { \mathbf { \nabla } } \cdot \vec { \mathbf { \nabla } } \cdot \vec { \mathbf { \nabla } } \cdot \vec { \mathbf { \nabla } } \cdot \vec { \mathbf { \nabla } } \cdot \vec { \mathbf { \nabla } } \cdot \vec { \mathbf { \nabla } } \cdot \vec { \mathbf { \nabla } } \cdot \vec { \mathbf { \nabla } } \cdot \vec { \mathbf { \nabla } } \cdot \vec { \mathbf { \nabla } } \cdot \vec { \mathbf { \nabla } } \cdot \vec { \mathbf { \nabla } } \cdot \vec { \mathbf { \nabla } } \cdot \vec { \nabla } \cdot \vec { \mathbf { \nabla \nabla } } \cdot \vec { \nabla } \cdot \vec \nabla \nabla \cdot \nabla \nabla \nabla \cdot \nabla \nabla \nabla \cdot \nabla \nabla \nabla \nabla \cdot \nabla \nabla \nabla \nabla \$ denotes string concatenation. When training the reward model, we adopt a method similar to Wang et al.. By following the Bradley-Terry model (Bradley and Terry 1952), we formulate a preference distribution by employing the reward model $r _ { \psi }$ as outlined below:

$$
P _ { \psi } ( q _ { c } \succ q _ { r } | h + q ) = \sigma ( r _ { \psi } ( h + q , q _ { c } ) - r _ { \psi } ( h + q , q _ { r } ) ) ,
$$

where $\sigma$ is the logistic function, $q _ { c }$ and $q _ { r }$ represent the chosen and rejected rewrite, respectively. This problem can be treated as a binary classification task, resulting in a negative log-likelihood loss function:

$$
\mathcal { L } _ { r m } = - \mathbb { E } _ { D _ { r m } } [ \log \sigma ( r _ { \psi } ( h + q , q _ { c } ) - r _ { \psi } ( h + q , q _ { r } ) ) ] ,
$$

where $D _ { r m }$ is the paired dataset.

In this work, we use the initialized rewriter to initialize $r _ { \psi }$ . Additionally, we incorporate an extra linear layer on top of the final transformer layer to generate a scalar prediction representing the reward value. Let $r _ { \psi } ^ { d _ { + } }$ r , $r _ { \psi } ^ { D }$ and $r _ { \psi } ^ { G }$ denote the reward models associated with metrics $m _ { d _ { + } }$ , $m _ { D }$ and $m _ { G }$ , respectively. Given the dialogue history $h$ , the current query $q$ , and the rewritten query $q _ { R }$ , the corresponding rewards can be obtained as $r _ { \psi } ^ { d _ { + } } ( h + q , q _ { R } )$ , $r _ { \psi } ^ { D } ( h + q , q _ { R } )$ and $r _ { \psi } ^ { G } ( h + q , q _ { R } )$ , abbreviated as $r _ { d _ { + } } ( q _ { R } )$ , $r _ { D } ( q _ { R } )$ and $r _ { G } ( q _ { R } )$ . To facilitate subsequent aggregation, we scale the scores of reward models to [0, 1].

# RL Training

To further train the rewriter, we employ a policy gradient RL framework. In the context of RL, the optimization of the rewriter can be viewed as a Markov Decision Process (MDP) represented by the 5-tuple $\langle S , A , P , r , \gamma \rangle$ . Here, the state space $S$ is a finite set constrained by the vocabulary and the sequence length. The action space $A$ comprises the available vocabulary and the transition probability $P$ is determined by the policy network, specifically the rewriter model $R _ { \theta } ^ { R L }$ . The reward function $r$ provides the reward value based on the current state, and $\gamma$ is the discount factor. For each step $t$ , the current state $s ^ { t }$ comprises the dialogue history $h$ , the current query $q$ , and the already generated rewrite $q _ { R , [ < t ] }$ . The action $a ^ { t }$ is the generation of the next token based on the current state. When the generation process stops, the reward for this episode is calculated using the reward function.

In the RL stage, for the dialogue history $h$ , the current query $q$ , and the rewritten query $q _ { R }$ , we combine scores of these reward models and the metric $m _ { q }$ as the feedback $r _ { R L }$ for RL training. This is expressed as:

$$
\begin{array} { r } { r _ { R L } ( q _ { R } ) = \lambda _ { 1 } r _ { d _ { + } } ( q _ { R } ) + \lambda _ { 2 } r _ { D } ( q _ { R } ) } \\ { + \lambda _ { 3 } r _ { G } ( q _ { R } ) + \mu m _ { q } ( q _ { R } ) , } \end{array}
$$

where $\lambda _ { 1 } , \lambda _ { 2 }$ and $\lambda _ { 3 }$ are hyperparameters that control the relative importance of each reward model, and $\mu$ is a hyperparameter that controls the weight of the metric $m _ { q }$ .

Then we use PPO algorithm to explore the optimal query rewriting strategy $R _ { \theta } ^ { R \bar { L } }$ . The reward objective we need to maximize in policy optimization is:

$$
r _ { t o t a l } = r _ { R L } ( q _ { R } ) - \eta \mathrm { K L } ( R _ { \theta } ^ { R L } | | R _ { \theta } ^ { 0 } ) ,
$$

where $\eta$ is a coefficient governing the magnitude of $\mathrm { K L }$ penalty, the initial policy model is the initialized rewriter.

# Experiments Experimental Settings

Datasets We conduct main experiments on two multi-turn dialogue RAG datasets, including QReCC (Anantha et al. 2020) and TopiOCQA (Adlakha et al. 2022). And conduct the transferability experiment on the ${ \tt W S D M @ 2 4 }$ MultiDoc QA dataset 1. QReCC contains $1 4 \mathrm { k }$ conversations with 81k question-answer pairs, built on questions from TREC CAsT (Dalton, Xiong, and Callan 2020), QuAC (Choi et al. 2018), and Google Natural Questions (Kwiatkowski et al. 2019). TopiOCQA is an open-domain conversational dataset with topic switches based on Wikipedia, containing 3,920 conversations with information-seeking questions and freeform answers. $\mathrm { w s D M @ 2 4 }$ Multi-Doc QA dataset comprises multi-turn question-answering dialogues and corresponding documents from Xiaohongshu.

Evaluation Metrics To evaluate the generated results of RAG, we use four standard evaluation metrics: ROUGE-1, ROUGE-L (Lin 2004), BLEU (Papineni et al. 2002), and METEOR (Banerjee and Lavie 2005). ROUGE prioritizes recall and $\mathfrak { n }$ -gram overlap, BLEU balances precision and brevity, and METEOR emphasizes semantic similarity. In addition, we utilize MRR (Craswell 2009) to evaluate the retrieval results, as it measures the system’s effectiveness in placing relevant items at the top of the ranked list.

Baselines We compare MaFeRw with four conversational RAG methods: raw RAG (Lewis et al. 2021), T5-base rewriter (Lin et al. 2020), RL-base rewriter (Ma et al. 2023), and ConvGQR (Mo et al. 2023). Raw RAG is the baseline model that performs retrieval and response generation without query rewriting, using the concatenation of dialogue history and the current query. T5-base rewriter uses manual rewrites to train a T5 model for query rewriting based on dialogue history. RL-base rewriter utilizes RL with generation feedback to assist the training of the rewriter. ConvGQR generates potential answers for query expansion and incorporates similarity between the rewrite and gold document into the rewriter’s optimization objective.

Implementation Details In this work, we train the query rewriter based on the pre-trained T5-base model. The dense retrieval component in the RAG system is constructed using FAISS (Johnson, Douze, and Jegou 2021) and the pretrained embedding model msmarco-roberta-base-ance-firstp (Reimers and Gurevych 2019). The generative model is the Llama-2-13b-chat model (Touvron et al. 2023). After initialization of the rewriter, a value head is added to the rewriter to initialize reward models. Details on hyperparameter determination are provided in the Appendix2.

# Main Results

We demonstrate experimental results on two datasets.

Accuracy of Reward Models It is crucial to train reward models to accurately score a given rewritten query with dialogue history and correctly categorize it as either ”chosen”

Table 1: The accuracy of reward models.   

<html><body><table><tr><td></td><td>rd+</td><td>rD</td><td>TG</td></tr><tr><td>QReCC</td><td>0.8927</td><td>0.8534</td><td>0.7802</td></tr><tr><td>TopiOCQA</td><td>0.8362</td><td>0.7303</td><td>0.7277</td></tr></table></body></html>

Table 2: The comparison of MaFeRw and the baselines on QReCC and TopiOCQA datasets.   

<html><body><table><tr><td>Method</td><td>ROUGE-1</td><td>ROUGE-L BLEU</td><td>METEOR</td><td>MRR</td></tr><tr><td>rawRAG</td><td>31.98</td><td>26.44</td><td>5.776 35.10</td><td>0.4729</td></tr><tr><td>T5-base</td><td>35.73</td><td>30.50</td><td>11.07 38.74</td><td>0.4884</td></tr><tr><td>RL-base</td><td>35.47</td><td>29.92</td><td>11.27 38.93</td><td>0.4742</td></tr><tr><td>ConvGQR</td><td>34.41</td><td>28.33</td><td>10.39 37.53</td><td>0.4974</td></tr><tr><td>MaFeRw</td><td>37.05</td><td>31.31</td><td>12.20 40.76</td><td>0.5032</td></tr><tr><td>Manual</td><td>39.73</td><td>34.07</td><td>13.54 43.85</td><td>0.5649</td></tr><tr><td></td><td></td><td>(a) QReCC</td><td></td><td></td></tr><tr><td>Method</td><td>ROUGE-1</td><td>ROUGE-L BLEU</td><td>METEOR</td><td>MRR</td></tr><tr><td>rawRAG</td><td>20.76</td><td>17.70 2.295</td><td>25.48</td><td>0.2664</td></tr><tr><td>T5-base</td><td>21.46</td><td>19.12 3.854</td><td>25.92</td><td></td></tr><tr><td>RL-base</td><td>22.94</td><td>20.46 4.993</td><td></td><td>0.3015</td></tr><tr><td>ConvGQR</td><td>17.05</td><td>15.11 3.503</td><td>27.83 22.16</td><td>0.3717</td></tr><tr><td>MaFeRw</td><td>23.97</td><td>21.38 5.496</td><td>29.51</td><td>0.3081 0.3802</td></tr><tr><td>Manual</td><td>24.72</td><td>21.95</td><td>5.367 30.00</td><td>0.4001</td></tr></table></body></html>

(b) TopiOCQA

or ”rejected”. Table 1 gives information about the accuracy of trained metric-specific reward models on their respective test sets. On both QReCC and TopiOCQA datasets, classification accuracies consistently exceeded $70 \%$ for all reward models, ensuring the success of subsequent PPO training based on these reward models.

RAG Performance Table 2 presents a comparison of our method and the baseline approaches in terms of retrieval and generation performance. Compared to the baselines, MaFeRw demonstrates significantly improved retrieval and generation performance on both datasets. MaFeRw achieves 5.07 and 3.19 ROUGE-1 higher than raw RAG on QReCC and TopiOCQA, respectively. Moreover, in both datasets, the improvement in MRR score is accompanied by an improvement in generation metrics.

Compared to QReCC dataset, TopiOCQA dataset features topic shifts within the dialogues. Notably, on TopiOCQA, the rewritten queries show a more significant improvement in MRR compared to using raw RAG with the dialogue history directly for retrieval. MRR of Long Rewriter improves by $6 . 4 1 \%$ on QReCC compared to raw RAG, and by $4 2 . 6 6 \%$ on TopiOCQA. Even T5-base rewriter achieves a more pronounced MRR improvement on TopiOCQA compared to QReCC. However, due to the topic shifts in TopiOCQA, while ConvGQR’s query expansion based on generated answers improves retrieval quality, it fails to effectively enhance generation quality. These findings suggest that query rewriting can effectively filter out redundant information, thereby significantly enhancing retrieval performance when dealing with dialogues involving topic shifts.

<html><body><table><tr><td>ROUGE-1</td><td>ROUGE-2</td><td>ROUGE-L</td><td>MRR</td></tr><tr><td>md+</td><td>36.10 20.70</td><td>30.22</td><td>0.4863</td></tr><tr><td>mD</td><td>35.69 20.40</td><td>29.96</td><td>0.4924</td></tr><tr><td>mG</td><td>36.69 21.59</td><td>31.03</td><td>0.4937</td></tr><tr><td>mq</td><td>36.88 22.06</td><td>31.22</td><td>0.4897</td></tr><tr><td>MaFeRw</td><td>37.04 21.84</td><td>31.31</td><td>0.5032</td></tr><tr><td>T5-base rewriter</td><td>35.73 20.02</td><td>30.50</td><td>0.4884</td></tr><tr><td colspan="4">(a) QReCC</td></tr><tr><td>ROUGE-1</td><td>ROUGE-2</td><td>ROUGE-L</td><td>MRR</td></tr><tr><td>md+</td><td>21.27 10.24</td><td>18.78</td><td>0.3619</td></tr><tr><td>mD</td><td>21.86 10.41</td><td>19.24</td><td>0.3693</td></tr><tr><td>mG</td><td>23.14 11.21</td><td>20.55</td><td>0.3570</td></tr><tr><td>mq</td><td>22.92 10.96</td><td>20.23</td><td>0.3723</td></tr><tr><td>MaFeRw</td><td>23.97 12.31</td><td>21.38</td><td>0.3802</td></tr><tr><td>T5-base rewriter</td><td>21.46 9.874</td><td>19.12</td><td>0.3015</td></tr></table></body></html>

(b) TopiOCQA

Table 3: The impact of four different metrics on the RL training of the query rewriter.

The dialogues in QReCC come from various sources, resulting in more diverse question-and-answer styles than those in TopiOCQA. As a result, RL-base rewriter struggles to adapt to these stylistic variations in QReCC dataset, leading to lower performance metrics compared to T5-base rewriter, and revealing its instability in training and outcomes. These results verify that more satisfactory responses can be obtained by feeding back the results of the remote retrieval and generation modules to query rewriting.

Additionally, it can be observed that the RAG performance with manual rewrites remains consistently strong across these two datasets. This observation underlies our decision to incorporate ROUGE between the rewriter output and the manual rewrite in the RL feedback.

# Ablation Study

We conduct RL training on both datasets using feedback from a single reward model or metric and compare results with those of T5-base rewriter. Table 3 demonstrates the impact of four different metrics on rewriter’s RL training.

(1) Impact of the metric $m _ { d _ { + } }$ For QReCC, we can be see that when only the reward model $r _ { d _ { + } }$ is used, ROUGE-1 and ROUGE-2 scores are higher than those of T5-base rewriter but MRR is lower. For TopiOCQA, MRR is much higher than that of T5-base rewriter. This indicates that using $m _ { d _ { + } }$ as feedback can, to some extent, guide RL optimization to achieve better retrieval and encourage LLM to improve generation quality, although this promoting effect is not stable.

(2) Impact of the metric $m _ { D }$ Using only the reward model $r _ { D }$ yields a better MRR than T5-base rewriter but lower ROUGEs on QReCC. For TopiOCQA, both ROUGE and MRR of $r _ { D }$ are higher than those of T5-base rewriter. This confirms that $m _ { D }$ helps the retriever more effectively find relevant documents, but this benefit does not translate to the generative model, thus failing to produce responses that better meet user needs.

![](images/32ca1db7d6bcb5762a94408419bff9bbfaf9c0f11329af4a12dee1df4ef3edcf.jpg)  
Figure 3: The changes on ROUGE-1 and MRR as training iterations increase when rewrites are applied to RAG.

(3) Impact of the metric $m _ { G }$ For both datasets, when only the reward model $r _ { G }$ is used, ROUGE and MRR scores improve compared to T5-base rewriter. On TopiOCQA, ROUGEs show a clear advantage, even though MRR of $r _ { G }$ is smaller than those of $r _ { d _ { + } }$ and $r _ { D }$ . This confirms the positive impact of $m _ { G }$ on RAG quality and highlights the necessity of conjunction with other metrics to achieve better results.

(4) Impact of the metric $m _ { q }$ When only the metric $m _ { q }$ is used, both ROUGEs and MRR scores show improvement over T5-base rewriter. On QReCC dataset, the improvement in ROUGE-1 score is notable. This suggests that when facing multi-source, diverse-style conversational data like QReCC, the metric $m _ { q }$ can guide RL optimization to achieve better RAG performance.

These results indicate that individual metrics can improve certain aspects of RAG performance, but their performance varies across different datasets, which is insufficient to stably support RL training. Therefore, combining multiple metrics is necessary to achieve better overall results.

# Analysis

To investigate whether using scores output by reward models and $m _ { q }$ as RL feedback can ensure stable training results, we conduct experiments on QReCC. We use either a single metric or reward model to guide the rewriter’s RL training and test the retrieval and generation performance of rewrites after 1000, 1500, 2000, and 2500 iterations. Figures 3a and 3b show the changes in ROUGE-1 and MRR scores as the number of iterations increases, respectively.

It can be observed that when training the rewriter using feedback from the reward model $r _ { d _ { + } }$ or $r _ { D }$ , MRR of the rewritten queries in RAG gradually increases, while ROUGE-1 shows slight decreases or fluctuations. This indicates that these two types of reward models can reliably guide the rewriter to optimize towards better retrieval performance, but are insufficient to consistently constrain the rewriter to improve generation results. When utilizing the reward model $r _ { G }$ for training feedback, although there are fluctuations in ROUGE-1 and MRR metrics, there is an overall upward trend. This trend validates that the reward model can provide positive guidance to the rewriter. When directly using $m _ { q }$ for feedback, ROUGE-1 exhibits a stable increase while MRR shows some variability. This indicates that the metric can, to some extent, consistently guide the rewriter towards optimizing in a manner beneficial for RAG generation. Furthermore, Figure 3 shows the changes in ROUGE-1 and MRR as iterations increase after training the rewriter with synthetic feedback. Both metrics show a more stable upward trend, further validating MaFeRw’s stability.

# Case 1:   
Figure 4: The case study comparing with two baselines.   

<html><body><table><tr><td>Context (QReCC): q1:What are the 2 most abundant elements in the earth's crust? a1: Oxygen and Silicon are the most abundant elements in the .. q2:How much is Oxygen? a2: Oxygen makes up 46.6% of the Earth's crust.</td></tr><tr><td>q4:What's the third most abundant? a4:Aluminum is the third most abundant element in the earth's crust. Current Query:What’s the atomic number? T5-base:What's the atomic number for silicon?(MRR:0.0) RL-base:What's the atomic number for silicon? (MRR: 0.0)</td></tr><tr><td>MaFeRw:What'stheatomicnumberforaluminum?(MRR:1.0) Generated Responce: T5-base: The atomic number for silicon is 14.</td></tr><tr><td>RL-base:The atomic number for silicon is 14. MaFeRw:The atomic numberforaluminum is 13. Ground Truth: The atomic number of aluminum is 13.</td></tr><tr><td># Case 2: Context (QReCC): qi:Who won the Tour de France in 1985? α1:Bernard Hinault won the Tour de France in 1985.</td></tr><tr><td>q2:Who won in 2019? a2:Egan Bernal from Team Ineos q3:Who was the youngest winner? α3: The youngest Tour de France winner was Cornet Henri, who was 19 years old. Current Query: Which year? T5-base:Which year did Cornet Henri win the Tour de France in 1985?</td></tr><tr><td>(MRR: 0.5) RL-base:Which year did Cornet Henri win the 1985 Tour de France?</td></tr><tr><td>(MRR: 0.5) MaFeRw:Which year did Cornet Henri win the Tour de France? (MRR: 0.5) Generated Responce: T5-base: Cornet Henri did not win the Tour de France in 1985. RL-base:Cornet Henri did not win the Tour de France in 1985.The Tour de France was first held in 1903. MaFeRw:Cornet Henri won the Tour de France in1904. Ground Truth: 1904 Tour de France.</td></tr></table></body></html>

Based on the above results, it can be analyzed that, by training on paired datasets, reward models can capture a more essential correspondence between rewriting based on dialogue history and each metric. Therefore, RL training with the feedback given by reward models can obtain more stable and effective results than directly training with the whole process of RAG.

Table 4: The generation performance on the WSDM@24 Multi-Doc QA task.   

<html><body><table><tr><td>Method</td><td>ROUGE-1</td><td>ROUGE-L</td><td>BLEU METEOR</td></tr><tr><td>rawRAG</td><td>38.79</td><td>23.95</td><td>5.989 24.35</td></tr><tr><td>T5-base</td><td>39.80</td><td>24.74</td><td>7.227 25.61</td></tr><tr><td>MaFeRw</td><td>41.01</td><td>25.86</td><td>8.226 26.46</td></tr></table></body></html>

# Case Study

We present a case study and a comparison with two other rewrite baselines in Figure 4 to illustrate more intuitively the improvement of MaFeRw in capturing users’ needs for multi-turn dialogue RAG.

For the first case, according to the dialogue history, the real need of the current query is to ask for ’the atomic number of aluminum’. However, because ’ silicon’ is also mentioned in the dialogue history, the two baselines are disturbed and add ’silicon’ mistakenly. As a result, the error retrieval and generation are triggered. In the second case, both baselines incorrectly add ’in $1 9 8 5 '$ to rewritten queries. Although the correct document can still be retrieved, the LLM is unable to give a correct reply based on the rewritten query due to the interference of the information ’1985’.

These results verify that MaFeRw can accurately capture the anaphora relationship between the current query and the dialogue history, especially when other interfering words are mentioned in the dialogue history.

# Transferability

To evaluate the transferability of MaFeRw, we test its performance on the $\mathrm { w s D M @ 2 4 }$ Multi-Doc QA dataset by using the rewriting model trained on QReCC to rewrite the current query, as shown in Table 4. This allows us to assess its effectiveness in handling multi-document conversational QA tasks. It can be observed that compared to the T5-base rewriter, MaFeRw achieves better generation performance on this task. This validates that the rewriter trained using our method possesses generalization capability across different conversational tasks.

# Conclusion

This paper introduces MaFeRw, a novel query rewriting method that enhances RAG performance by incorporating multi-aspect feedbacks. We find that dense reward feedback can lead to more stable and satisfactory generation results than single reward, and design four rewards from multiple aspects, such as gold document, retrieved documents and ground truth. The scores of these reward models are combined to optimize the rewriter using PPO algorithm. Experimental results on two conversational RAG datasets show MaFeRw outperforms baselines in generation metrics, and further analysis validates MaFeRw’s stability and transferability. Future work involves document re-ranking to better match retrieved documents with contextual constraints and prompt reconstruction to capture users’ complex intents in multi-turn dialogues.