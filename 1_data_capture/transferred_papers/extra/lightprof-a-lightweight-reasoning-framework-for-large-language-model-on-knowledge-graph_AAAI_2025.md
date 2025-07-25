# LightPROF: A Lightweight Reasoning Framework for Large Language Model on Knowledge Graph

Tu $\mathbf { A 0 } ^ { 1 * }$ , Yanhua $\mathbf { Y } \mathbf { u } ^ { 1 \dag \ast }$ , Yuling Wang2‡, Yang Deng3, Zirui Guo1, Liang Pang5, Pinghui Wang6, Tat-Seng Chua4, Xiao Zhang1, Zhen Cai1

1Beijing University of Posts and Telecommunications, China 2Hangzhou Dianzi University, China 3Singapore Management University, Singapore 4National University of Singapore, Singapore 5Institute of Computing Technology, Chinese Academy of Sciences, China 6Xi’an Jiaotong University, China {aotu bupt, yuyanhua, zrguo, xiao20010420, caizhen}@bupt.edu.cn, wangyl0612 $@$ hdu.edu.cn, pangliang $@$ ict.ac.cn, phwang $@$ mail.xjtu.edu.cn, dcscts@nus.edu.sg

# Abstract

Large Language Models (LLMs) have impressive capabilities in text understanding and zero-shot reasoning. However, delays in knowledge updates may cause them to reason incorrectly or produce harmful results. Knowledge Graphs (KGs) provide rich and reliable contextual information for the reasoning process of LLMs by structurally organizing and connecting a wide range of entities and relations. Existing KG-based LLM reasoning methods only inject KGs’ knowledge into prompts in a textual form, ignoring its structural information. Moreover, they mostly rely on close-source models or open-source models with large parameters, which poses challenges to high resource consumption. To address this, we propose a novel Lightweight and efficient Prompt learning-ReasOning Framework for KGQA (LightPROF), which leverages the full potential of LLMs to tackle complex reasoning tasks in a parameter-efficient manner. Specifically, LightPROF follows a “Retrieve-Embed-Reason” process, first accurately, and stably retrieving the corresponding reasoning graph from the KG through retrieval module. Next, through a Transformer-based Knowledge Adapter, it finely extracts and integrates factual and structural information from the KG, then maps this information to the LLM’s token embedding space, creating an LLM-friendly prompt to be used by the LLM for the final reasoning. Additionally, LightPROF only requires training Knowledge Adapter and can be compatible with any open-source LLM. Extensive experiments on two public KGQA benchmarks demonstrate that LightPROF achieves superior performance with small-scale LLMs. Furthermore, LightPROF shows significant advantages in terms of input token count and reasoning time.

# Introduction

With the emergence of more Large Language Models (LLMs), their continuously improving performance has brought substantial innovations to the field of Natural Language Processing (NLP) (Zhao et al. 2023; Touvron et al. 2023; Achiam et al. 2023; Team et al. 2023; GLM et al. 2024). The “emergent abilities” displayed under extensive training data and vast parameters allow LLMs to excel in complex zero-shot tasks (Wei et al. 2022a). Despite their effectiveness, LLMs often struggle with knowledge-intensive tasks due to limited task-specific prior knowledge and understanding capabilities (Sun et al. 2024). Additionally, the costly and time-consuming training process of LLMs presents considerable challenges in continuously updating and maintaining their knowledge bases.

To address the aforementioned challenges, it is crucial to enable LLMs to access a reliable and continuously updated knowledge base to support more accurate and interpretable reasoning (Pan et al. 2024). Knowledge Graphs (KGs) are ideally suited for this purpose, as they offer a structured semantic framework that delivers both accessible and timely information. Knowledge Graph Question Answering (KGQA), as a common knowledge-intensive task, existing work has explored methods for integrating LLMs with KGs to conduct KGQA reasoning (Jiang et al. 2023; Wu et al. 2023; Baek, Aji, and Saffari 2023; Wen, Wang, and Sun 2023; Sun et al. 2024; Guo et al. 2024). Broadly speaking, current KG-empowered LLM reasoning primarily involves retrieving information from KGs and incorporating the results into LLM input prompts, leveraging the LLMs reasoning capabilities to address questions.

While LLMs reasoning on KGs holds great promise, several challenges remain: Firstly, the content of KGs is often represented directly as extensive textual content, which fails to effectively convey the rich logical relationships within their graph structure that are crucial for reasoning. In previous work, the content of KGs was presented in input prompts as multidimensional lists or in natural language form, making it difficult to clearly express the complex relationships and hierarchical structures within them. Secondly, retrieval and reasoning on KGs demand a high number of LLM calls and substantial LLM reasoning power. Previous work used an iterative approach starting from the question entity, gradually obtaining information for reasoning. This increased the number of LLM calls, sacrificed reasoning efficiency, and diminished feasibility. The textual content describing KGs is vast, requiring not only a larger context window but also a more powerful LLM to ensure that no information is missed while avoiding the generation of incorrect answers in the redundant context.

In response to these challenges, we propose a RetrieveEmbed-Reason framework for LLMs, which is a novel Lightweight and efficient Prompt learning-ReasOning Framework called LightPROF, designed to provide smallscale LLMs with stable retrieval and efficient reasoning capabilities. The framework is structured around three core components: the Retrieval, Embedding, and Reasoning modules. The Retrieval module utilizes relation as the fundamental retrieval unit and limits the retrieval scope based on the question’s semantics to obtain the reasoning graph needed to answer the question. This approach not only boosts the accuracy and stability of retrieval but also considerably narrows the search space and reduces the need for frequent LLM invocations. Next, the Embedding module introduces a small and refined Transformer-based Knowledge Adapter that extracts and integrates the textual and structural information from the reasoning graph, generating representations perfectly suited for the LLM. This module offers an efficient and streamlined way of encoding information, addressing potential ambiguity and information redundancy while reducing the required input token count and context window size, resulting in a more accurate and efficient reasoning process. Finally, The Reasoning module combines the embedded representation vectors with carefully designed natural language prompts, allowing the LLM to derive the final answer. This design allows LightPROF to seamlessly support any open-source LLM and various KGs, requiring only the tuning of the Knowledge Adapter during training, without needing to update the costly and time-consuming LLM. Our contributions are summarized as follows:

• To the best of our knowledge, it is the first framework that transforms both the textual content and graph structure of KGs into embeddings used to prompt LLMs. • We propose LightPROF, a lightweight and efficient prompt-learning reasoning framework that provides small-scale LLMs with stable retrieval and efficient reasoning capabilities, requiring far fewer training parameters compared to the LLM itself. • Extensive experiments conducted on two KGQA datasets demonstrate the superiority of our proposed LightPROF, surpassing methods that use large-scale LLMs (such as LLaMa-2-70B, ChatGPT). Further analysis shows that LightPROF has significant efficiency advantages in terms of input token count and reasoning time.

# Related Work

LLM Prompt Engineering. In expanding the capabilities of LLMs, prompt engineering has become a crucial technology. It maximizes the performance of LLMs across different applications and research domains by designing special task instructions (i.e., prompts) without altering model parameters (Sahoo et al. 2024; Saravia 2022). Many studies have been proposed on prompt engineering, spanning from zero-shot prompts (Radford et al. 2019) and few-shot prompts (Brown et al. 2020) to Chain-of-Thought (CoT) (Wei et al. 2022b) and its derivatives such as Tree-ofThoughts (ToT) (Yao et al. 2024; Long 2023) and Graph-ofThoughts (GoT) (Besta et al. 2024). Additionally, to address the issues of poor robustness and weak expressiveness in discrete prompts, many studies have explored soft prompts (Li and Liang 2021; Liu, Lee, and Yih 2022; Chen et al. 2024; Perozzi et al. 2024), demonstrating their effectiveness and feasibility in various NLP tasks and structured data representations. Proficiency in prompt engineering can enhance the understanding of the strengths and weaknesses of LLMs.

KG-based LLM Reasoning. KGs store a vast amount of explicit and structured knowledge that can effectively enhance the knowledge awareness of LLMs (Pan et al. 2024). Therefore, researchs have been conducted on using KGs to enhance LLMs’ pre-training and generation techniques. Compared to natural language, KGs have clearer structured logic, which can better guide reasoning. Many studies use factual triples from KGs to construct corpora and employ various pre-training tasks to enhance the capabilities of LLMs (Zhang et al. 2023b; Dong et al. 2023a; Yu et al. 2022; Sun et al. 2021). However, this approach causes KGs to lose their advantages of interpretability and dynamism, and may also face catastrophic forgetting issues during the training process (Hu et al. 2023).

Therefore, constructing LLM prompts using factual information from KGs is a more flexible, convenient, and secure solution, and our method belongs to this kind of approach. For example, KAPING (Baek, Aji, and Saffari 2023) retrieves factual knowledge from KGs based on the semantic similarity of the question, adds it to the question as a prompt, and then uses the LLM to generate answers. KGGPT (Kim et al. 2023) uses LLMs to perform reasoning on KG data through three steps: sentence segmentation, graph inference, and reasoning. StructGPT (Jiang et al. 2023) constructs an specialized interface for KG and proposed an Iterative Reading and Reasoning (IRR) framework for LLMs to solve KG-based tasks using this interface. ToG (Sun et al. 2024) utilizes LLMs to iteratively perform beam search on KGs, discovering reasoning paths and returning the most probable reasoning results. KnowledgeNavigator (Guo et al. 2024) enhances LLM reasoning by more efficiently and accurately retrieving external knowledge from KGs. While the aforementioned methods have demonstrated commendable performance, they uniformly represent KGs in natural language, which can introduce information redundancy and confusion, ultimately leading to incorrect reasoning.

# Preliminaries

Knowledge Graph (KG) is a data structure that stores a vast quantity of knowledge in the form of triples: $\mathcal { G } = \{ ( \bar { h , } r , t ) | \bar { h } , t \in \mathcal { E } , r \in \bar { \mathcal { R } } \}$ ,where $\mathcal { E }$ and $\mathcal { R }$ denote the set of entities and relations, respectively. A triple $\langle h , r , t \rangle$ represents the existence of a relation $r$ between the head

Stage1: Reasoning Graph Retrieval   
B Question: Which company was founded by the person who wrote the book'The Lean Startup'? 0 TheLean Rritsgnig rarries Knowledge Startup Graph founded founded IMVU Stock Exchange Long-Term Stage2:Knowledge Embedding Knowledge Reasoning Graph Soft Prompt The Ltuan witeaed EricRieounded Knowlere IMVU Long-Term Stock Exchange   
Knowledge Stage3:Knowledge PromptsMixedReasoning   
Soft Prompt Hard Prompt Answer Based on the reasoning Θ graph,pleaseanswer LLM Long-Mer'Stock the given question: <graph> Exchange Semantic Extraction   
?Which company was foundedby theperson Question_hop:2-hop whowrote the book‘TheLeanStartup'? Anchor Entity:The Lean Startup RelationRetrieval 2-hop KG-Subgraph Relation Links 2-hop Relation link1:["written_by","live_in"] -hop Extraction link2:["written_by","founded"] Knowledge Graph link3:["published_by","part_of"] link4:.... :anchorentity relation   
Reasoning Graph Sampling RelationRanking Top-kRelation Links Basedonthequestion,asess link2:["written_by","founded"] therelevanceof thefollowing link4:["published_by","part_of"] relation linksandrank them. ReasoningGraph Here is thesorted list with IMVU corresponding scores: EricRies 1 ["written_by","founded"]-9/10 The Lean LTSE 2. ["published_by","part_of']-5/10 Startup 3. ["written_by","live_in"]-0/10 4.

entity $h$ and the tail entity $t$

Anchor Entities are a set of entities: $B = \{ b _ { 1 } , b _ { 2 } , \dots , b _ { K } \}$ that are referenced in the KG-based question, where $b _ { k } \in \mathcal { E }$ denotes the $k$ -th entity in the question $q$ .

Relation Link is a sequence of relations: $\begin{array} { r l } { l } & { { } = } \end{array}$ $\{ r _ { 1 } , r _ { 2 } , \dots , r _ { J } \}$ , initiated by an anchor entity for J hop exploration, where $r _ { j } \in \mathcal { R }$ denotes the $j$ -th relation in the relation link.

Reasoning Path represents a concrete example of the relation link $l$ within the KG of anchor entity $b _ { 1 } ~ \in ~ B$ : $R _ { l } = \{ b _ { 1 } , r _ { 1 } , e _ { 1 } , r _ { 2 } , . . . , r _ { M } , e _ { M } \}$ , where $r _ { m } ~ \in ~ l$ and $\begin{array} { l l l } { e _ { m } } & { \in } & { \mathcal { E } } \end{array}$ denote the $m$ -th relation and entity in $R _ { l }$ , respectively.

# Methodology

We design the LightPROF framework, which achieves efficient complex KG problem reasoning under small-scale LLMs through precise retrieval and fine-grained structured data processing capabilities. As shown in Figure 1, our proposed Retrieve-Embed-Reason framework contains three stages: Reasoning Graph Retrieval, Knowledge Embedding, and Knowledge Prompts Mixed Reasoning.

# Stage1: Reasoning Graph Retrieval

For the complex multi-hop KGQA task, the question “How to efficiently, accurately, and stably retrieve information from a KG based on a question?” is paramount. To address this critical issue, we devide the retrieval module into three steps: semantic extraction, relation retrieval, and reasoning graph sampling, as depicted in Figure 2.

Semantic Extraction. For a given question $q$ , our goal is to extract relevant semantics (i.e., the number of hops $h _ { q }$ and anchor entities $B$ ) from the KG to narrow the retrieval scope while preserving the essential reasoning knowledge. This approach enables the retrieval and construction of a highly relevant and precise reasoning graph (Guo et al. 2024). Specifically, we fine-tune a pre-trained language model (PLM), such as BERT, to learn the number of hops $h _ { q }$ in KG required for reasoning, based on the semantic vector $V _ { q }$ of the query $q . H$ is the maximum number of hops in the dataset, which can be framed as a classification task:

$$
V _ { q } = \mathrm { P L M } ( q )
$$

$$
h _ { q } = \arg \operatorname* { m a x } _ { h } P ( h | V _ { q } ) , h = 1 , 2 , \ldots , H .
$$

Relation Retrieval. Relations in KGs describe the specific connections between two entities, providing semantic clarity for their interactions and substantially enriching the information content of KGs. Many studies currently utilize semantically rich relation links for KG reasoning tasks (Xiong, Hoang, and Wang 2017; Xu et al. 2022; Dong et al. 2023b). More crucially, relations in KGs demonstrate more stability and intuitiveness compared to the continuously changing and complex entities (Cai et al. 2023). To gather as much relevant knowledge as possible, we adopt a search for relation links in the KG based on anchor entities $B$ and the predicted hop $h _ { q }$ . Specifically, the model first selects an anchor entity and then employs a constrained breadth-first search (BFS) with a depth limit of $h _ { q }$ . This process is designed to collect all relation links originating from the anchor entity $B$ and extending up to a predetermined length of $h _ { q }$ .

Reasoning Graph Sampling. First, the retrieved relation links are fed into a LLM. Subsequently, the LLM calculates scores and ranks them according to their semantic relevance to the question $q$ . Then, we select the top- $k$ relevant links. Finally, we sample in KG based on the selected relation links, extracting multiple reasoning paths $\{ R _ { 1 } , R _ { 2 } , \ldots , R _ { N } \}$ to construct a refined reasoning graph, denoted as $G _ { R }$ .

# Stage2: Knowledge Embedding

KGs typically encompass a rich array of complex structural information, including subgraph structures, relational patterns, and the relative relation between entities (Zhang et al. 2023a). Such structural information is essential for LLMs to gain a deep understanding of KGs. However, the natural language expression of KG structural information contains redundancy and confusion, which cannot directly reveal its inherent nature, thus impeding LLMs from effectively utilizing this information.

![](images/7344df39067c835d3ba2da0bb59ba2f507ff3b5435c735aad1e177c51c6191aa.jpg)  
Figure 3: Illustration of the Knowledge Adapter and the schematic representation of its crucial components.

To address the aforementioned challenge, as inspired by (Chen et al. 2024; Perozzi et al. 2024), we propose a refined and compact Knowledge Adapter that can encode textual information in the reasoning graph while extracting its structural information, as illustrated in Figure 3. By combining textual information with structural details at a fine granularity, Knowledge Adapter aids the model in deeply comprehending the knowledge within the reasoning graph, enabling more precise reasoning.

Specifically, we assume that the reasoning graph $G _ { R } = { }$ $\{ R _ { n } \} _ { n = 1 } ^ { N }$ is composed of $N$ reasoning paths, each of which is decomposed into a set of triples $\mathcal { T } ^ { n } = \{ ( h _ { i } ^ { n } , r _ { i } ^ { n } , t _ { i } ^ { n } ) | i \in$ $[ 1 , h _ { q } ] \}$ , where $h _ { q }$ is the number of reasoning hops. Subsequently, Embed $\bar { ( \cdot ) }$ , i.e., BERT, is used to obtain the relational embedding $e _ { i } ^ { r }$ for each triple:

$$
\mathbf { e } _ { i } ^ { r } = { \mathrm { E m b e d } } ( r _ { i } ^ { n } ) .
$$

We can obtain the entity embeddings $e _ { i } ^ { h } , e _ { i } ^ { t }$ in the same way. Next, we aim to capture both the local and global interactions between each entity and relation. We first use StructEmb( ) to encode the local structural information $\mathbf { s } _ { i }$ of $i$ -th triple in ${ \mathcal { T } } ^ { n }$ . Then, a linear layer Linear $( \cdot )$ is used to aggregate the global structural information $\mathbf { z } ^ { s }$ from the entire reasoning path $\textstyle R _ { n }$ :

$$
\begin{array} { r } { { \bf s } _ { i } = \mathrm { S t r u c t E m b } ( { \bf e } _ { i } ^ { h } , { \bf e } _ { i } ^ { r } , { \bf e } _ { i } ^ { t } ) , } \\ { { \bf z } ^ { s } = \mathrm { L i n e a r } ( { \bf s } _ { 1 } , { \bf s } _ { 2 } , . . . , { \bf s } _ { h _ { q } } ) . } \end{array}
$$

Additionally, to capture the textual information of the reasoning path $\scriptstyle { R _ { n } }$ , we use $\mathrm { F u s i o n } ( \cdot )$ to combine the text-level information of all entities and relations in $R _ { n }$ . We first obtain the combined text representation $\mathbf { z } ^ { t _ { h } }$ of all head entities

as follows:

$$
\begin{array} { r } { { \bf z } ^ { t _ { h } } = \mathrm { F u s i o n } ( { \bf e } _ { 1 } ^ { h } , \ldots , { \bf e } _ { h _ { q } } ^ { h } ) . } \end{array}
$$

Then, the combined text representations of relations $\mathbf { z } ^ { t _ { r } }$ and tail entities $\mathbf { z } ^ { t _ { t } }$ can be obtained in the same way. Afterwards, these vectors are consolidated into a single vector $\mathbf { z } ^ { t }$ to represent the comprehensive textual information of the entire reasoning path $\scriptstyle { R _ { n } }$ :

$$
\begin{array} { r } { { \bf z } ^ { t } = f _ { c } ( { \bf z } ^ { t _ { h } } , { \bf z } ^ { t _ { r } } , { \bf z } ^ { t _ { t } } ) , } \end{array}
$$

where $f _ { c } ( \cdot )$ is the consolidation function. While $f _ { c } ( \cdot )$ can be complex neural networks or language models, to preserve the semantic integrity of the text and reduce the model’s training complexity, we use a simple concatenation operation to form a composite vector that encapsulates all the textual information of the entire reasoning path.

Finally, we use KnowledgeEncoder $( \cdot )$ to seamlessly integrate the obtained comprehensive textual information $\mathbf { z } ^ { t }$ and global structural information $\mathbf { z } ^ { s }$ , deriving a fused representation of the reasoning path, as shown in Figure 3:

$$
\mathbf { z } ^ { f } = { \mathrm { K n o w l e d g e E n c o d e r } } ( [ \mathbf { z } ^ { t } , \mathbf { z } ^ { s } ] )
$$

In this way, the Knowledge Encoder can effectively encode each reasoning path in the reasoning graph into a single token, significantly improving the token utilisation efficiency of the LLM and enhancing the representational capacity of the reasoning paths. During the encoding process, the Knowledge Encoder captures not only rich textual information from the reasoning graph but also crucial structural information. Since the fused information $\mathbf { z } ^ { f }$ contains both textual and structural elements, the model can more fully understand the meaning embedded in each reasoning path during inference. This multidimensional information representation enhances the model’s sensitivity to context , facilitating more effective deep semantic analysis and reasoning. Consequently, this information integration allows the model to more accurately capture the complex interactions between semantics and structure, thereby enhancing the accuracy and depth of reasoning.

By aggregating all paths {Rn}nN= , we obtain the representational sequence [zf1 , zf2 , . . . , zfN ] of the reasoning graph $G _ { R }$ . Before inputting the sequence into the LLM, a dimension transformation is necessary. Due to the differences between the embedding space of the Knowledge Encoder and the input space of the LLM, directly using these tokens would be ineffective. Therefore, we develop a trainable projector $\Phi ( \cdot )$ , which maps these tokens into the token embedding space of the LLM. As a result, this process generates an input sequence suitable for the LLM, which we refer to as the knowledge soft prompt $\displaystyle p _ { \mathrm { s } }$ :

$$
p _ { \mathrm { s } } = \Phi ( [ \mathbf { z } _ { 1 } ^ { f } , \mathbf { z } _ { 2 } ^ { f } , \ldots , \mathbf { z } _ { N } ^ { f } ] ) .
$$

Here we set $\Phi ( \cdot )$ as a two-layer multilayer perceptron. Following the aforementioned process, the Knowledge Adapter is able to encode the textual representation of the reasoning graph into the corresponding knowledge soft prompt. Importantly, all parameters of this adapter are derived from the parameters of the Knowledge Encoder and Projector, which are the only components requiring tuning during the LightPROF training process.

# Stage3: Knowledge Prompts Mixed Reasoning

LLMs have acquired extensive knowledge through broad training on large corpora. However, despite their proficiency in general knowledge, LLMs show notable deficiencies in processing specialized knowledge, complex long logic chains, and multi-hop knowledge reasoning, which mainly stem from the limitations of their pre-training data. Additionally, although the knowledge base of LLMs can be expanded through retraining, this method is usually costly and time-consuming (Sun et al. 2024). More seriously, retraining may lead to catastrophic forgetting of existing knowledge in the model (Zhang et al. 2024). Thus, this presents certain challenges in keeping LLMs’ knowledge up-to-date. To avoid the aforementioned challenges, we freeze the parameters of the LLM during the LightPROF training process and use a combination of soft prompts and hard prompts to guide the model to answer questions more precisely and efficiently, which can be seen in Figure 1.

Specifically, the input to the LLM is organized in a chat format, where instructions and questions are combined using carefully designed text templates, which we call hard prompts. During the encoding phase of the LLM, we insert the knowledge soft prompt, representing the reasoning graph, into specific locations of the hard prompt to effectively inject external knowledge, as shown in Figure 1. This approach allows the LLM to autonomously and accurately answer questions based on the given input content without the need for parameter updates. By this method, we not only maintain the stability of the model but also enhance its performance and efficiency within specific knowledge domains.

The training objective of LightPROF is to maximize the likelihood of generating correct answers $\mathcal { A }$ for all samples in the dataset $\mathcal { D }$ . This can be compatible with the task of nexttoken prediction, a fundamental method for training generative models. The training goal can be articulated as:

$$
\arg \operatorname* { m a x } _ { \mathcal { A } } P _ { \mathrm { l m } } ( \mathcal { A } | p _ { \mathrm { p } } ) = \sum ^ { \mathcal { D } } \sum _ { t = 1 } ^ { | A | } \log P _ { \mathrm { l m } } ( a _ { t } | a _ { 1 : t - 1 } , p _ { \mathrm { h } } , p _ { \mathrm { s } } ) ,
$$

where $p _ { \mathrm { p } }$ is the input sequence that includes both hard prompt $p _ { \mathrm { h } }$ and soft prompt $\displaystyle p _ { \mathrm { s } }$ , and $a _ { t } ( t = 1 , 2 , \dots , | \mathcal { A } | )$ is the $t$ -th token of the output sequence. Notably, when $t = 1$ , $a _ { 1 : t - 1 }$ is the model’s beginning-of-sequence (BOS) token.

# Experiments

In this experiment, we will thoroughly discuss the following questions. Q1: How significantly can LightPROF enhance LLMs’ performance in KGQA tasks? Q2: Can LightPROF be integrated with different LLM backbones to enhance performance? Q3: Can LightPROF achieve efficient input and stable output with small-scale LLMs?

# Datasets

We train and evaluate LightPROF’s multi-hop reasoning capabilities on two public datasets based on the Freebase knowledge graph (Bollacker et al. 2008): WebQuestionsSP(WebQSP) (Yih et al. 2016) and ComplexWebQuestions(CWQ) (Talmor and Berant 2018). Based on previous works, we utilize match accuracy $( \mathrm { H i t s } @ 1 )$ to evaluate whether the model’s top-1 answer is correct.

• WebQSP is a benchmark with fewer questions but a larger knowledge graph, consisting of 4,737 questions. Each question includes a topic entity, a reasoning chain, and a SPARQL query to find the answer. The answer entity requires up to 2-hop reasoning on the Freebase. • CWQ is a benchmark specifically designed for complex knowledge graph question answering research. It includes 34,689 question-answer pairs, built upon the WebQSP dataset. It involves automatically creating more complex SPARQL queries and generating corresponding natural language questions, thereby creating a wide and diverse range of question types. These questions require up to 4-hop reasoning on Freebase.

# Baselines

We consider three types of baseline methods: full finetuning methods, vanilla LLM methods, and $\mathrm { L L M + K G s }$ methods. The full fine-tuning methods include KV-Mem (Miller et al. 2016), EmbedKGQA (Saxena, Tripathi, and Talukdar 2020), TransferNet (Shi et al. 2021), NSM (He et al. 2021), KGT5 (Saxena, Kochsiek, and Gemulla 2022), GraftNet (Sun et al. 2018), PullNet (Sun, Bedrax-Weiss, and Cohen 2019), UniKGQA (Jiang et al. 2022). Vanilla LLM methods include LLaMa series models (Touvron et al. 2023). LLM $\cdot +$ KGs methods include StructGPT (Jiang et al. 2023), ToG (Sun et al. 2024), KnowledgeNavigator (Guo et al. 2024), AgentBench (Liu et al. 2024). Notably, to ensure fair comparisons, the LLM+KGs methods we select do not involve fine-tuning the LLMs, i.e., all of them are zeroshot methods without any training of the LLM.

# Implementation

To demonstrate the plug-and-play convenience and parameter efficiency of LightPROF, we conduct experiments on two small-scale language models in the LLaMa series: LLaMa7B-chat (Touvron et al. 2023) and LLaMa-8B-Instruct1. The model was optimized over one training epoch with a batch size of 4. The initial learning rate was set at 2e-3, adjusted using a cosine annealing schedule to enhance the model’s learning efficiency during training. All experiments are conducted using the PyTorch toolkit on NVIDIA A800 GPU.

The Knowledge Encoder module is based on the BERT model. The module includes a two-layer MLP Projector that maps dimensions to the LLM’s input dimension.

# Q1: Performance Comparison

Main Result. We evaluate LightPROF against three categories of baseline methods: full fine-tuning, vanilla LLM, and LLM+KGs approaches. As illustrated in Table 1, LightPROF not only excels in simple questions but also demonstrates high performance in scenarios requiring deep reasoning and complex query handling. Specifically, LightPROF significantly surpasses the state-of-the-art model on the WebQSP dataset $8 3 . 7 \%$ vs. $7 5 . 1 \%$ ) and also excels on the more complex CWQ dataset $( 5 9 . 3 \%$ vs. $5 7 . 6 \%$ ). These outcomes validate our framework’s excellent capability in addressing KGQA tasks, emphasizing LightPROF’s efficacy in managing multi-hop and complex challenges.

Compared to vanilla LLMs and $\mathrm { L L M + K G s }$ methods that utilize plain text prompts, LightPROF’s significant improvement indicates that soft prompts produced by the Knowledge Adapter can effectively encapsulate more complex structural knowledge than discrete text, being concise, informative, and highly expressive, thus enhancing LLM’s understanding of KG information. It is noteworthy that our framework outperforms other large-scale models in all experimental conditions. For example, our framework excels, particularly in reasoning through complex problems, compared to ToG (Sun et al. 2024) with LLaMa2-70B-Chat and StructGPT (Jiang et al. 2023) with ChatGPT. Additionally, even with the smaller LLaMa2-7b version, our framework competes effectively with other large-scale models, underscoring the efficiency and optimization of our framework’s design.

<html><body><table><tr><td>Methods</td><td>WebQSP</td><td>CWQ</td></tr><tr><td>KV-Mem</td><td>46.7</td><td>18.4</td></tr><tr><td>EmbedKGQA</td><td>66.6</td><td>45.9</td></tr><tr><td>NSM KGT5</td><td>68.7</td><td>47.6</td></tr><tr><td></td><td>56.1</td><td>36.5</td></tr><tr><td>GraftNet</td><td>66.4</td><td></td></tr><tr><td>PullNet TransferNet</td><td>68.1</td><td>1</td></tr><tr><td>UniKGQA</td><td>71.4 75.1</td><td>48.6 50.7</td></tr><tr><td>LLaMa2-7B-Chat</td><td></td><td></td></tr><tr><td>LLaMa2-70B-Chat</td><td>61.4 57.4</td><td>31.5 39.1</td></tr><tr><td>ToG (LLaMa2-70B)</td><td>68.9</td><td></td></tr><tr><td>StructGPT(ChatGPT)</td><td>72.6</td><td>57.6</td></tr><tr><td>AgentBench</td><td>47.8</td><td>54.3</td></tr><tr><td>KnowledgeNavigator(LLaMa2-</td><td>71.8</td><td>24.8</td></tr><tr><td>70B)</td><td></td><td></td></tr><tr><td>LightPROF (LLaMa3-8B)</td><td>83.8</td><td>59.3</td></tr><tr><td>LightPROF (LLaMa2-7B)</td><td>71.2</td><td>48.5</td></tr></table></body></html>

Table 1: Performance comparison of LightPROF with baselines on the two datasets. Bold and underlined typefaces indicate optimal and sub-optimal methods, respectively.

Table 2: Model ablation study of our LightPROF framework.   

<html><body><table><tr><td>Methods</td><td>WebQSP</td><td>CWQ</td></tr><tr><td>LightPROF</td><td>83.77</td><td>59.26</td></tr><tr><td>LightPROF w/o Struct</td><td>82.36</td><td>58.05</td></tr><tr><td>LightPROFw/o Train</td><td>80.37</td><td>55.63</td></tr><tr><td>LightPROFw/RandomRetrieve</td><td>53.44</td><td>46.84</td></tr></table></body></html>

Ablation Study. An ablation study is performed on LightPROF to investigate the specific effects of the Knowledge

Adapter on KGQA task performance. We examine three variants: (1) w/o Struct, removing the structural information included in the knowledge embedding process, (2) w/o Train, without training the Knowledge Encoder, and (3) w/ Random Retrieve, randomly retrieve reasoning paths from KGs. The results are displayed in Table 2.

The results indicate that the integration of structural information is crucial for the model’s understanding and handling of entities and relationships in complex queries. The incorporation of structural information significantly enhances the model’s utilization efficiency of data in the knowledge graph. Continuous training of the Knowledge Encoder is also essential for enhancing the model’s comprehension and generation of knowledge representations. This training process notably improves the model’s capability to encode complex structural knowledge, allowing it to more accurately respond to queries rooted in deep knowledge. Moreover, randomly retrieved reasoning paths can cause significant damage to performance, highlighting the importance of an accurate and stable retrieval module.

Additionally, we explore different structural encoders. The structural encoder used in our framework encodes triples as Head $\mathrm { ( H ) + }$ Relation (R) - Tail (T). Results in Table 3 show that the performance of the $\mathrm { H } { + } \mathrm { R } { + } \mathrm { T }$ encoding method slightly declines due to its inability to distinguish the order of the triples, e.g., the structural information derived from (Eric Ries, founded, IMVU) and (IMVU, founded, Eric Ries) is identical, reducing the model’s capacity to understand structural information. In contrast, LightPROF can better capture structural information within the reasoning graph and integrate it at a finer granularity, enhancing the model’s understanding, particularly in scenarios involving complex structured data reasoning.

Table 3: Performance impact of different structure encoder in LightPROF.   

<html><body><table><tr><td>Methods</td><td>WebQSP</td><td>CWQ</td></tr><tr><td>LightPROF(H+R+T)</td><td>83.68</td><td>58.32</td></tr><tr><td>LightPROF(H+R-T)</td><td>83.77</td><td>59.26</td></tr></table></body></html>

# Q2: Plug-and-Play

For our framework, any open-source LLM capable of accepting token embedding inputs is suitable. In this section, we evaluate the effectiveness of integrating different LLMs within LightPROF. As illustrated in Table 5, the results demonstrate that the LightPROF framework significantly enhances the performance of integrated LLMs, regardless of the baseline performance of the original models. LightPROF enhances the model’s capability to address complex KG questions through effective integration and optimization of structured data. This plug-and-play integration strategy does not require costly fine-tuning of LLMs, making it particularly suitable for quickly enhancing existing models’ performance on KGQA task.

Table 4: Case Study of LightPROF and StructGPT on the WebQSP Dataset.   

<html><body><table><tr><td>Question</td><td>what drugs lindsay lohan abuse?</td></tr><tr><td>Answer</td><td>["Alcoholic beverage”,"Cocaine”]</td></tr><tr><td>StructGPT</td><td>The relevant relation:celebrities.celebrity.substance_abuse_problems The possible constraints: celebrities.substance_abuse_problem.substance: Alcoholic beverage The final answers: Alcoholic beverage</td></tr><tr><td>LightPROF</td><td>Number of Hops: 2 Relation Links: ['base.popstra.celebrity.substance_abuse',‘base.popstra.substance_abuse.substance'] - 9/10 ['base.popstra.celebrity.substance_abuse',‘base.popstra.substance_abuse.abuser'] - 9/10 Based on the knowledge graphs, please answer the given question. Please keep the ans wer as simple as possble and return all the possible answers as a list. knowledge graphs: < graph > ["Cocaine”,"Alcoholic beverage"]</td></tr></table></body></html>

Table 5: The performance of integrating various LLMs into the LightPROF framework.   

<html><body><table><tr><td>Methods</td><td>WebQSP</td><td>CWQ</td></tr><tr><td>Llama2-7b</td><td>61.36</td><td>31.49</td></tr><tr><td>LightPROF (Llama2-7b)</td><td>71.19</td><td>48.48</td></tr><tr><td>Llama3-8b</td><td>66.83</td><td>48.87</td></tr><tr><td>LightPROF (Llama3-8b)</td><td>83.77</td><td>59.26</td></tr></table></body></html>

# Q3: Efficient Input and Stable Output

Efficiency Results. A series of efficiency tests are conducted to compare the performance of LightPROF and StructGPT (Jiang et al. 2023) when processing the WebQSP dataset. Specifically, the models’ runtime, the total number of input tokens, and the average Number of tokens Per Request (NPR) are measured, with results presented in Table 6. The table shows that LightPROF is more time-efficient when processing the same dataset, with a $30 \%$ reduction in time cost (1:11:49 vs. 1:42:12). Regarding the total number of input tokens, LightPROF and StructGPT show a significant difference (365,380 vs. 24,750,610), demonstrating that LightPROF is more economical in input processing, reducing token usage by approximately $98 \%$ . Furthermore, LightPROF’s NPR value is 224, significantly lower than StructGPT’s 6400. This comparison further highlights LightPROF’s advantage in the number of tokens needed per request, showcasing its more precise and resource-efficient handling of each request, validating LightPROF’s effectiveness when integrating small-scale LLMs.

<html><body><table><tr><td>Methods</td><td>TimeCost</td><td>TokenUsed</td><td>NPR</td></tr><tr><td>LightPROF</td><td>1:11:49</td><td>365,380</td><td>224</td></tr><tr><td>StructGPT</td><td>1:42:12</td><td>24,750,610</td><td>6400</td></tr></table></body></html>

Table 6: Efficiency performance of LightPROF and StructGPT on Llama-3-8b. NPR represents the average number of tokens per request.

Case Study. As shown in Table 4, we validate LightPROF’s efficient input and stable output capabilities when using small-scale LLMs by comparing its performance with StructGPT to answer complex queries about Lindsay Lohan’s drug abuse. The results show that LightPROF not only accurately identify and comprehensively answer the query, but also demonstrate deeper reasoning pathways and overall scoring. In contrast, although StructGPT handled the relevant questions, it failed to fully capture all related answers. Interestingly, we found that LightPROF can consistently generate output that includes only the answers and uses fewer input tokens and less reasoning time. This suggests that LightPROF can effectively integrate and precisely output complex information from knowledge graphs, demonstrating its reliability and practicality in efficiently and accurately handling complex KGQA tasks.

# Conclusion

In this paper, we introduce the LightPROF framework, which accurately retrieves and efficiently encodes KGs to enhance LLM reasoning. To effectively narrow the retrieval scope, LightPROF incrementally samples the KG using stable relationships as units. To achieve efficient reasoning on LLMs with fewer parameters, we develop a delicate Knowledge Adapter that can effectively parse graph structures and perform fine-grained information integration, thus condensing the reasoning graph into a smaller number of tokens and achieving comprehensive alignment with the LLM’s input space through the Projector. Experimental results show that our framework outperforms other baseline methods, particularly those involving large-scale language models. In comparison to other methods based exclusively on text, our knowledge soft prompts integrate a more comprehensive range of structural and textual information, making them more easily understood by LLMs. In future work, we plan to explore 1) KG encoders with stronger generalization and compatibility, and design an encoder that can be applied to unseen KG data without retraining. 2) A unified cross-modal encoder capable of encoding multimodal KGs.