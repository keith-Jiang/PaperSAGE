# Fast Think-on-Graph: Wider, Deeper and Faster Reasoning of Large Language Model on Knowledge Graph

Xujian Liang1,2, Zhaoquan $\mathbf { G u } ^ { 2 , 3 * }$

1School of Cyberspace Security, Beijing University Of Posts And Telecommunications, China 2Department of New Networks, Peng Cheng Laboratory, Shenzhen, China 3School of Computer Science and Technology, Harbin Institute of Technology (Shenzhen), Shenzhen, China xjliang $@$ bupt.edu.cn, guzhaoquan $@$ hit.edu.cn

# Abstract

Graph Retrieval Augmented Generation (GRAG) is a novel paradigm that takes the naive RAG system a step further by integrating graph information, such as knowledge graph (KGs), into large-scale language models (LLMs) to mitigate hallucination. However, existing GRAG still encounter limitations: 1) simple paradigms usually fail with the complex problems due to the narrow and shallow correlations capture from KGs 2) methods of strong coupling with KGs tend to be high computation cost and time consuming if the graph is dense. In this paper, we propose the Fast Think-on-Graph (FastToG), an innovative paradigm for enabling LLMs to think “community by community” within KGs. To do this, FastToG employs community detection for deeper correlation capture and two stages community pruning - coarse and fine pruning for faster retrieval. Furthermore, we also develop two Community-to-Text methods to convert the graph structure of communities into textual form for better understanding by LLMs. Experimental results demonstrate the effectiveness of FastToG, showcasing higher accuracy, faster reasoning, and better explainability compared to the previous works.

Code — https://github.com/dosonleung/FastToG

# Introduction

Retrieval-Augmented Generation (RAG) (Lewis et al. 2020) is a cutting-edge technique for large-scale language models (LLMs) to generate accurate, reasonable, and explainable responses. The early RAG, known as Naive RAG (Gao et al. 2023), mostly works by indexing the documents into manageable chunks, retrieving relevant context based on vector similarity (Karpukhin et al. 2020) to a user query, integrating the context into prompts, and generating the response via LLMs. As the paradigm is simple and highly efficient for the basic tasks, Naive RAG is widely adopted in various applications e.g. Q&A (Chan et al. 2024), Recommendation System (Deldjoo et al. 2024), Dialogue System (Ni et al. 2023), etc. However, there remain such as low precision as the ambiguity or bias (Thurnbauer et al. 2023) exist in the embedding model, low recall when dealing with complex queries, and lack of explainability as the queries and documents are embedded in high-dimensional vector spaces.

The Graph-based RAG (GRAG) is widely considered an advanced RAG by incorporating KGs as an external source for LLMs. Employing various graph algorithms within a graph database, GRAG empowers LLMs for complex operations (Liang et al. 2024) such as BFS and DFS querying, path exploration, and even community detection, etc, providing LLMs a wider and deeper understanding of the relations within the data, making it possible to execute sophisticated reasoning and the generate more precise responses. In this paper, GRAG is categorized into n-w GRAG and n-d GRAG (Fig. 1) based on the breadth and depth of retrieval. Particularly, there are still quite a few 1-w 1-d Graph RAG researches (Andrus et al. 2022; Baek, Aji, and Saffari 2023; Li et al. 2023). Similar to the “Retrieve one Read One” paradigm of Naive RAG, most of the 1-w 1-d research focuses on single entity or one-hop relationship queries, thereby inheriting the shortcomings of Naive RAG. To overcome these, researches (Jiang et al. 2023; Modarressi et al. 2023; Edge et al. 2024; Wang et al. 2024), categorized as nw GRAG, aim to expand the context window of retrieval. Noteworthy works within the n-w GRAG category focus on treating the densely interconnected groups of nodes as the basic units for retrieval and reasoning. These groups are partitioned from the graph by community detection, allowing them to furnish LLMs with more contextual information compared to single node or random clustering methods. For example, given the query concerning the “climate of the area where Pennsylvania Convention Center is located”, triples such as (Pennsylvania Convention Center, located in, Market East Section),...,(Pennsylvania, climate, humid subtropical) would be sufficient for answering as the entity Pennsylvania Convention Center of the query is linked to the relevant entity humid subtropical over a short distance (n-hop). Despite its simplicity, the retrieval may falter if the distance is too long between the entities (Fig 1 a,b). n-d GRAG (Sun et al. 2023; Wang et al. 2022a) are ideal for this predicament by broadening the depth of the context window. Analogous to “Let’s think step by step” (Kojima et al. 2022), n-d GRAG takes entities and relations as intermediate steps in the chain of thought (CoT) to guide the generation of LLMs. As they are tightly coupled with KGs, these paradigms exhibit higher recall in retrieval. However, since LLMs are required to “think” at each step, longer chains result in heightened computational costs (Fig. 1, c). On the other hand, if

Question: What kindofclothingshouldIbring alongifIhead tothearea around the Pennsylvania Convention Centerin spring?   
1-w 1-d GraphRAG (a) Response: Sorry, based on my query n-w 1-d GraphRAG (b) Response: Sorrybased onmy query result from the knowledge base,l result from the knowledge base,I can cannotaveeruinstisice etract eiyntncent notaeutsind Retrieve Answer RetrieveNodes Answer Pennsylvania Pennveivana cloth ？ PeneyivaniaConientiarket Convention Penslnetonthas Center PromptEastectionaet Wison Bronterated by Prompt copamkee &Company .·.   
1-WndGaptRAG (c) Themosti h related toPennsylvaina artrerisis answering the question.Looking heading to Pennsylvaina Convention Center,consider Section. for triples related to Pennsylvania Enough Retrieve Node [Retrieve Node too much "think" Retrieve Node Infomation Combine Peoneneaoia isLocatedln Market ast isLocatedIn isCityOf Pennsylvania climate Suburmidcal ? .......... "think" "answer" n-w n-d GraphRAG (d) The most relevant one is (Pennsylvaina Convention Center, isLocatedln, In the area with humid subtropical in spring,lightweight Looking for triples related Pennsylvania State). Information not enough for answering the question. and breathable clothing are recommended.Thus,if you're Looking for the communities related to thecommunity Pennsylvania. heading to Pennsylvaina Convention Center,consider Retrieve Community 1 Enough RetrieveNode (Community:_Pennsylvania_State) Infomation ↑ Combine Pennsylvania LLMs inner knowledge: Convention isLocatedlni Market East isLocatedIn isCityOf Pennsylvania climate subtropical higltomildraturswitheiia

the graph is dense, the multitude of paths between the source and target entities can become exceedingly vast, leading to a decreased recall of entity retrieval.

Building on the consideration of n-w or n-d methods, we propose a n-d n-w Graph RAG paradigm: Fast Thinkon-Graph (FastToG). The key idea of FastToG is to guide the LLMs to think “community by community”. As illustrated in Fig. 1, though the path between Pennsylvania Convention Center and humid subtropical may be lengthy, it notably shortens when nodes are grouped. Accordingly, FastToG regards communities as the steps in the chain of thought, enabling LLMs to “think” wider, deeper, and faster. Concretely, FastToG leverages community detection on the graph to build the reasoning chains. Considering the time complexity of community detection, we introduce Local Community Search (LCS), aiming to literately detect the communities in a local scope. Given potential graph density concerns, LCS incorporates two stages of community pruning: modularity-based coarse pruning and LLMs-based fine pruning, aiming to enhance the efficiency and accuracy of retrieval. Furthermore, as the language models are trained on textual data, graph structures are incompatible with the format of input. In light of this, We explore two methods to convert community into text: Triple2Text and Graph2Text, which aim to provide better inputs for LLMs.

We conducted experiments on real-world datasets, and FastToG exhibited the following advantages:

1. Higher Accuracy: FastToG demonstrate its significant enhancement on the accuracy of LLMs generated content compared with the previous methods.

2. Faster Reasoning: Our experiments also show that community-based reasoning can notably shorten the reasoning chains, reducing the number of calls to the LLMs.   
3. Better Explainability: The case study indicates that FastToG not only simplifies the retrieval for LLMs but also enhances the explainability for users.

# Related Work

# Algorithms of Community Detection

Community detection algorithms are used to grouping or partitioning nodes, as well as their tendency to strengthen or separate apart. These algorithms can be categorized into agglomerative and divisive types.

Agglomerative: These algorithms iteratively group smaller communities until a stop condition is met. One prominent method is hierarchical clustering, which progressively groups nodes or small communities into larger ones by similarity. This down-top grouping procedure is intuitive but computationally expensive, making it impractical for large-scale networks. Since the determination to stop the clustering is important, Louvain (Blondel et al. 2008) and Leiden (Traag, Waltman, and Van Eck 2019) algorithms introduce the modularity benefit function as a measure of community quality. Based on modularity, the optimization procedure can be explicitly stopped as the global modularity can no longer be improved given perturbations.

![](images/8b5714216f475a473524b4f3f1f01e7916d25c6c251e874e2c88470de401dcf3.jpg)  
Figure 2: A general schema of the FastToG paradigm.

Divisive: In contrast, Divisive methods follow the upbottom process of iteratively partitioning the original graph into small substructures. For example, the Girvan-Newman (GN) algorithm (Girvan and Newman 2002) removes edges with high betweenness centrality to foster more connected components as communities. Spectral clustering leverages the spectrum(eigenvalues) to embed nodes to $\mathbf { k }$ -dimensional vector space and group them based on the clustering method e.g. k-means.

# Graph-based Retrieval Augmented Generation

Graph RAG empowers LLMs to capture and utilize the structure and semantics of KGs, enhancing the accuracy of response and explainability of reasoning process. According to the stage at which KGs participate in, GRAG are categorized into Pretrain/Fine-tuning based and Prompt-guided based methods.

Pretrain/Fine-tuning based: These methods enable language models to learn KGs by Input augmentation or model enhancement. (Xie et al. 2022) linearize graph structures to fine-tune language models, whereas K-BERT expands input structures with KGs information. (Zhang et al. 2019) embed the entities by incorporating knowledge encodes. (Wang et al. 2020) infuses knowledge into pretrained models via transformer adapters, and (Yamada et al. 2020) enhances transformers with an entity-aware self-attention mechanism. While these methods effectively integrate KGs into language models, they require additional supervised data, leading to increasing costs as the model scaling increases.

Prompt-guided based As the training/fine-tuning process is often expensive and time-consuming, recent RAG systems focus on prompt-guided methods. Based on the width and depth of retrieval, these works can be categorized into 1-w 1-d, n-w, and n-d methods. Examples of 1-w 1- d methods include Cok (Li et al. 2023), KAPING (Baek, Aji, and Saffari 2023) etc. Cok enhances the flexibility of queries with SPARQL. KAPING vectorizes triples for efficient retrieval. Representative works of n-w includes KGP (Wang et al. 2024), StructGPT (Jiang et al. 2023), RETLLMs (Modarressi et al. 2023) and GraphRAG (Edge et al. 2024) etc. While the first three all randomly or fully capture the adjacent nodes for retrieval purposes, GraphRAG stands out by utilizing community detection to optimize retrieval for text summarization. KP-PLM (Wang et al. 2022a) and ToG (Sun et al. 2023) are the typical works of the n-d paradigm. KP-PLM firstly designs the $d > 1$ entity-relation chains for information retrieval. ToG, particularly relevant to our research, introduces enhancements such as pruning and beam search within chains via LLMs.

# The Method

# Overview

We begin by introducing the basic framework of Fast think on graph (FastToG). Inspired by the work (Kojima et al. 2022) “Let’s think step by step”, the basic idea of FastToG is enabling large models to “think” community by community on the KGs. Consequently, the core component of the FastToG (see our repository for the pseudocode) are the $W$ reasoning chains $P = [ p _ { 1 } , p _ { 2 } , . . . , p _ { W } ]$ , where each chain $p _ { i } \subseteq P$ is a linked list of communities i.e., $p _ { i } = [ c _ { 0 } ^ { i } , c _ { 1 } ^ { i } , . . . , c _ { n - 1 } ^ { \hat { i } } ]$ .

The FastToG framework comprises two main phases: the initial phase and the reasoning phase. The objective of the initial phase is to determine the start communities $c _ { 0 }$ and the header community (Fig. 2a) $c _ { 0 } ^ { i }$ for each reasoning chain $p _ { i }$ . FastToG prompts LLMs to extract the subject entities of to query $x$ as a single-node community $c _ { 0 }$ . After that, FastToG employs Local Community Search (LCS), which involves two key components: Community Detection on Subgraph (Fig. 2b) and Community Pruning (Fig. 2c), to identify neighbor communities with highest potential for solving the query $x$ , serving as the head community $c _ { 0 } ^ { i }$ for each reasoning chain $p _ { i }$ .

Once the head communities are determined, the algorithm enters into the reasoning phase. For each $p _ { i }$ , FastToG continues to leverage LCS for local community detection and pruning, with the selected community being added to $p _ { i }$ as the newest element, termed pruning. After the update of all reasoning chains, Community2Text methods like Graph2Text (G2T) or Triple2Text (T2T) are utilized to convert all communities within chains into textual form as the input content of LLMs, which is called reasoning (Fig. 2d). If the gathered reasoning chains are deemed adequate by the LLMs for generating the answer, the algorithm will be terminated and return the answer. To mitigate time consumption, if no answer is returned within $D _ { m a x }$ iterations of reasoning, the algorithm will be degraded into the methods of answering the query by inner knowledge of LLMs e.g. IO/CoT/CoT-SC. We will illustrate all the details in the following sections.

# Local Community Search

In this subsection, we focus on the LCS, which consist of there main part: community detection on the subgraph, pruning methods of coarse-pruning and fine-pruning.

Community Detection on Subgraph Due to the vast number of nodes and relations in the KGs, as well as the dynamic of queries, it is impossible to partition the KGs into communities entirely and frequently. To solve this, we propose community detection based on the local subgraphs for each pruning. Given KGs $\Omega$ , start community $c _ { 0 }$ or header community $c _ { 0 } ^ { i }$ , FastToG firstly retrieves the subgraph $g \ ( g \subset \Omega )$ within $n$ hops from $c _ { 0 }$ or $c _ { 0 } ^ { i }$ . Considering the rapid growth of the number of neighbors in the dense graph and the degradation of semantics as the nodes move further apart, the algorithm randomly samples neighbor nodes at different hops with exponential decay $\rho$ . The probability of selecting each node $x$ at $n$ -hop from $c _ { 0 }$ or $c _ { 1 } ^ { i }$ is given by:

$$
P r ( x = 1 ) = \rho ^ { n - 1 }
$$

Once the local subgraph $g$ is retrieved, the next step is to partition it into communities. Community detection algorithms are utilized to reveal the closely connected groups of nodes (referred to as communities) of the graph. This process enables the graph to be partitioned into subgroups with tight internal connectivity and sparse external connections, aiding LLMs in uncovering implicit patterns within the graph structure. In this study, we consider Louvain (Blondel et al. 2008), Girvan–Newman algorithm (Girvan and Newman 2002), Agglomerative Hierarchical Clustering, and Spectral Clustering as representative algorithms. To prevent the retrieval of repeated communities, the algorithm ensures that new communities discovered do not exist in the historical community set during each community detection iteration, and adds them to the historical community set after each pruning step.

The oversize communities may contain more redundant nodes, leading to the increased noise, prolonged community detection time, and reduced explainability. Thus, we introduce a constraint on the maximum size of community within the community detection. To do this, all detection algorithms will be backtracked when the stop condition is met. At each iteration from the end to the beginning of backtracking, the algorithm verifies whether the size of each community meets the condition $\mathrm { s i z e } ( c _ { i } ) < = M$ (hyperparameter). Finally, the algorithm returns the partition status that first satisfies the size constraint.

Modularity-based Coarse-Pruning When dealing with too many candidate communities, existing methods relying on Large Language Models (LLMs) for community selection encounter difficulties, resulting in either high time consumption or selection bias (Zheng et al. 2023). To tackle this, current methods frequently incorporate random sampling to diminish the number of candidate communities or nodes. However, these methods inevitably ignore the network structure of the graph, resulting in the loss of structural information from the candidate communities.

Based on the above, we propose modularity-based coarsepruning. The modularity (Blondel et al. 2008) of a partitioned subgraph $g$ is:

$$
Q = \frac { 1 } { 2 m } \sum _ { i j } [ A _ { i j } - \frac { k _ { i } k _ { j } } { 2 m } ] \delta ( c _ { i } , c _ { j } )
$$

Where $m$ is the number of edges, $A _ { i j }$ is an element in the adjacency matrix $A , k _ { i }$ and $k _ { j }$ are the degrees of nodes $i$ and $j$ , respectively. The function $\delta ( c _ { i } , c _ { j } ) = 1 \quad$ indicates that nodes $i$ and $j$ belong to the same community, otherwise, it returns 0. In the weighted graph, $m$ is the sum of graph weights, $\begin{array} { r } { \frac { 1 } { 2 m } \sum _ { i j } A _ { i j } \delta \big ( c _ { i } , c _ { j } \big ) } \end{array}$ represent the average weight of the given coPmmunity, while $\frac { k _ { i } k _ { j } } { 2 m } \delta ( c _ { i } , c _ { j } )$ denotes the average weight of the given community under random conditions. The larger the difference between them, the higher the quality of community partitioning. For simplicity, we do not consider the weights or directions of edges in our work. The expression1 of modularity can be rewritten as:

$$
\begin{array} { l } { { \displaystyle Q = \frac { 1 } { 2 m } [ \sum _ { i j } A _ { i j } - \frac { \sum _ { i } k _ { i } \sum _ { j } k _ { j } } { 2 m } ] \delta ( c _ { i } , c _ { j } ) } } \\ { { \displaystyle ~ = \frac { 1 } { 2 m } \sum _ { c } [ \sum \mathrm { i n } - \frac { ( \sum \mathrm { t o t } ) ^ { 2 } } { 2 m } ] } } \end{array}
$$

where $\textstyle \sum$ in is the number of edges in community $c$ and $\textstyle \sum$ tot is  he number of edges connected to $c$ . Thus, the modularity of community $c$ can be:

$$
Q ( c ) = \sum \mathrm { i n } - { \frac { ( \sum \mathrm { t o t } ) ^ { 2 } } { 2 m } }
$$

Upon calculating the modularity of each candidate community $c \subseteq C$ , the communities $C ^ { ' }$ with low modularity will be pruned, while the remaining will serve as the refined set of candidate communities for the next stage.

$$
C ^ { ' } : = \arg \log \mathrm { k } _ { c \subseteq C } Q ( c )
$$

LLMs-based Fine-Pruning After coarse pruning, a smaller set of candidate communities $\boldsymbol { C } ^ { ' }$ that are more compact in structure is identified. Subsequently, FastToG prompts the LLMs to make the final selection C′′:

$$
C ^ { ' \prime } = \mathrm { f n e \mathrm { . p r u n i n g } } ( x , C ^ { ' } , \Pi , k )
$$

where $x$ is the query, $\Pi$ is the instance of LLMs, $k = 1$ or $W$ is for the single or multiple choice.

To simplify the pruning process, FastToG no longer considers scoring-based pruning (Sun et al. 2023). Instead, it prompts the LLMs to directly choose either the best community or top $W$ communities. In the initial phase, LLMs are guided through multiple-choice prompts to retrieve $W$ communities, which will act as the header communities $c _ { 0 } ^ { 1 } , . . . , c _ { 0 } ^ { W }$ for the reasoning chains $p _ { 1 } , . . . , p _ { W }$ , respectively. During the reasoning phase, single-choice prompts are employed for each reasoning chain to recall the best community $c _ { j } ^ { i }$ , which is then appended to its chain $p _ { i }$ . Not that the length of each chain $p$ may not be the same if LLMs insist that none of the candidate communities are relevant to the query. In such cases, the exploration of such chains will be discontinued.

# Reasoning

Once of all the reasoning chains $p _ { i } \subset P$ are updated, LLMs will be prompted to integrate and generate the answers from all the chains. The returned results could be a clear answer if the chains are adequate for answering, or “Unknown” if not. In cases where “Unknown” is returned, the algorithm proceeds to the next iteration until either reaching the maximum depth $D = D _ { m a x }$ or obtaining a definitive answer. If the maximum iterations are exhausted without a conclusive response, FastToG will be degraded to generate the answer by the inner knowledge of LLMs itself. Consistent with ToG (Sun et al. 2023), the entire process of FastToG paradigm involves 1 round of pruning and reasoning in the initial phase, and $2 W D _ { m a x }$ pruning and $D _ { m a x }$ reasoning in the second. Consequently, the worst condition needs $2 W D _ { m a x } + D _ { m a x } + 2$ calls to the LLMs.

# Community-to-Text

Knowledge graphs are organized and stored in the form of RDF triples. Thus, a community also consisted of triples e.g. $c =$ [(Philadelphia, isCityOf, Pennsylvania), (Pennsylvania, climate, Humid Subtropical)]. To input this structure into LLMs, it needs to be converted into text format. To do this, we propose two methods: Triple-to-Text (T2T) and Graphto-Text (G2T). For T2T, triples are directly converted into text by rule-based scripts e.g. $\mathrm { T } 2 \mathrm { T } ( c ) =$ “Philadelphia isCityOf Pennsylvania, Pennsylvania climate Humid Subtropical”. For G2T, triple will be converted into human language like $\mathbf { G } 2 \mathbf { T } ( c ) =$ “Philadelphia, located in the state of Pennsylvania, features a Humid Subtropical climate.”

T2T may result in redundancy. For instance, a T2T result of [(Allentown, isCityOf, Pennsylvania), (New Castle, isCityof, Pennsylvania), (Philadelphia, isCityOf, Pennsylvania)] is “Allentown isCityOf Pennsylvania, New Castle isCityof Pennsylvania, Philadelphia isCityOf Pennsylvania”, which can be summarized as: “Allentown, New Castle, and Philadelphia are the cities of Pennsylvania”. Therefore, G2T also undertakes the role of the text summary. To do this, we fine-tune the smaller language models (like T5-base, 220M) on the outputs from T2T for the conversion.

Since G2T leverages a base model with less parameters compared to existing LLMs (e.g., llama-3-8b has 36 times more parameters than T5-base), the impact of time efficiency is tiny. Apart from intra-community relationships, there also exist inter-community paths such as $c _ { 1 } - E _ { 1 , 2 } - c _ { 2 }$ . Therefore, it is necessary to perform text transform on the $\boldsymbol { E } _ { 1 , 2 }$ . This study excludes candidate communities with distance larger than 1 hop from the current community, meaning the paths between the current community and candidate communities do not contain any intermediate nodes.

# Experiments

# Experimental Setup

Dataset and Evaluation Metric We evaluated FastToG on 6 real-world datasets, which include datasets of multihop KBQA: CWQ (Talmor and Berant 2018), WebQSP (Yih et al. 2016), and QALD (Perevalov et al. 2022), slot filling: Zero-Shot RE (abbr. ZSRE) (Petroni et al. 2021) and TREx (Elsahar et al. 2018), and common-sense reasoning Creak (Onoe et al. 2021). To ensure comparability with prior works (Li et al. 2023; Sun et al. 2023; Edge et al. 2024), we employed exact match $( \mathrm { h i t } @ 1 )$ as the evaluation metric. Considering the computational cost and time consumption, we randomly sampled 1k examples from the datasets containing more than a thousand data. Furthermore, the initial entities provided by (Sun et al. 2023) were used as the starting community for the evaluations.

Language Model To reduce the computational cost, we employ two LLMs: gpt- $4 \mathrm { { o } - m i \mathrm { { n i } } ^ { 1 } }$ and Llama-3-70b-instruct (Dubey et al. 2024) without quantization. To keep the pruning steady, the temperature is configured to 0.4 during pruning and 0.1 during reasoning. Considering the text descriptions of community structure are longer, the maximum length of output is increased to 1024.

Graph2Text Model The Graph2Text module is developed by fine-tuning T5-base (Raffel et al. 2020) using dataset WebNLG (Auer et al. 2007) and synthetic data generated by GPT-4. To build the synthetic data, we prompt GPT-4 to generate text descriptions of given communities, which are the byproduct of the experiments.

Knowledge Graph We utilize Wikidata (Vrandeˇcic´ and Kr¨otzsch 2014), which is a free and structured knowledge base, as source of KGs. The version of Wikidata is 20240101 and only the triples in English are considered. Following extraction, 84.6 million entities and 303.6 million relationships are stored in $\mathrm { N e o } 4 \mathrm { j } ^ { 2 }$ .

# Performance on Accuracy

Accuracy is one of the most important criteria for RAG systems. In this experiment, we evaluate FastToG and compared methods on the datasets and settings mentioned above.

Compared Methods We consider two categories of comparative methods: Inner-knowledge-based or KGs-retrievalbased methods. The former methods include: 1) IO prompt (Brown et al. 2020): prompting the model to directly generate the result. 2) CoT (Wei et al. 2022): guiding the model to “think” step by step before answering. 3) CoT-SC (Wang et al. 2022b): ensembling all the CoTs to obtain more consistent predictions. The KGs-retrieval-based methods include: 1) 1-d 1-w methods, which represent the methods like CoK (Li et al. 2023), KAPING (Baek, Aji, and Saffari 2023), etc. 2) 1-d n-w methods for KGP (Wang et al. 2024) 3. n-d 1- w method for ToG (Sun et al. 2023). To keep consistency with previous works, we keep a $W = 3$ for the number of chains and $D _ { m a x } = 5$ for the maximum iterations. For n-w methods like 1-d n-w, Ours(t2t), and Ours(g2t), the maximum size of community is all set at 4 and the algorithm for community detection are Louvain Method.

Table 1: Accuracy $( \% )$ for different datasets by gpt-4o-mini.   

<html><body><table><tr><td>Method</td><td>CWQ</td><td>WebQSP</td><td>QALD</td><td>ZSRE</td><td>TREx</td><td>Creak</td></tr><tr><td colspan="7">Inner-knowledge based Methods</td></tr><tr><td>10</td><td>31.2</td><td>49.6</td><td>38.6</td><td>26.4</td><td>46.4</td><td>90.2</td></tr><tr><td>CoT</td><td>35.1</td><td>60.8</td><td>51.8</td><td>35.6</td><td>52.0</td><td>94.6</td></tr><tr><td>CoT-SC</td><td>36.3</td><td>61.2</td><td>52.4</td><td>35.8</td><td>52.0</td><td>95.0</td></tr><tr><td colspan="7">KGs-retrieval based Methods</td></tr><tr><td>1-d 1-w</td><td>35.5</td><td>59.2</td><td>50.7</td><td>39.4</td><td>56.1</td><td>92.0</td></tr><tr><td>1-d n-w</td><td>42.3</td><td>64.4</td><td>54.8</td><td>46.1</td><td>58.8</td><td>92.8</td></tr><tr><td>n-d 1-w</td><td>42.9</td><td>63.6</td><td>54.9</td><td>54.0</td><td>64.2</td><td>95.4</td></tr><tr><td>Ours(t2t)</td><td>43.8</td><td>65.2</td><td>56.1</td><td>54.4</td><td>67.3</td><td>95.6</td></tr><tr><td>Ours(g2t)</td><td>45.0</td><td>65.8</td><td>55.9</td><td>54.2</td><td>68.6</td><td>96.0</td></tr></table></body></html>

Table 2: Accuracy $( \% )$ for different datasets by llama-3-70b.   

<html><body><table><tr><td>Method</td><td>CWQ</td><td>WebQSP</td><td>QALD</td><td>ZSRE</td><td>TREx</td><td>Creak</td></tr><tr><td colspan="7">Inner-knowledge based Methods</td></tr><tr><td>10</td><td>32.9</td><td>55.8</td><td>44.3</td><td>30.7</td><td>52.8</td><td>91.0</td></tr><tr><td>CoT</td><td>34.2</td><td>58.8</td><td>45.0</td><td>32.6</td><td>55.6</td><td>91.2</td></tr><tr><td>CoT-SC</td><td>34.8</td><td>60.6</td><td>46.8</td><td>34.9</td><td>57.0</td><td>91.8</td></tr><tr><td colspan="7">KGs-retrieval based Methods</td></tr><tr><td>1-d 1-w</td><td>35.0</td><td>57.6</td><td>44.4</td><td>34.4</td><td>56.0</td><td>91.3</td></tr><tr><td>1-d n-w</td><td>39.8</td><td>64.0</td><td>46.6</td><td>57.1</td><td>60.4</td><td>91.9</td></tr><tr><td>n-d 1-w</td><td>40.3</td><td>62.4</td><td>51.6</td><td>64.8</td><td>61.2</td><td>93.3</td></tr><tr><td>Ours(t2t)</td><td>42.1</td><td>65.9</td><td>54.9</td><td>67.7</td><td>63.5</td><td>92.2</td></tr><tr><td>Ours(g2t)</td><td>46.2</td><td>66.4</td><td>54.3</td><td>67.9</td><td>64.7</td><td>94.5</td></tr></table></body></html>

Compared Result Tab. 1 and Tab. 2 respectively display the accuracy achieved by gpt-4o-mini and llama3-70binstruct across the datasets. Overall, FastToG, which includes t2t and $\mathrm { g 2 t }$ mode, outperforms all previous methods. In particular, Ours(g2t) surpasses n-d 1-w (ToG) by $4 . 4 \%$ in Tab. 1 and $5 . 9 \%$ in Tab. 2.

For two community-to-text conversions, $\mathtt { g 2 t }$ methods show higher accuracy on most datasets, aligning well with the idea for our proposed Graph2Text in the previous section. However, we note that the improvement of $\mathtt { g 2 t }$ is tiny (mostly $< 1 \%$ ), and even slightly underperforms to t2t on QALD dataset. To find out the explanation for these counterintuitive results, we carried out extensive checks (see our repository for details) and found that the hallucination from the base model of Graph2Text is the main reason for $\mathtt { g 2 t }$ falling short of t2t.

# Performance on Efficiency

Efficiency is another key criterion of our work. To verify it, we consider the number of calls to the LLMs as the evaluation criterion. As outlined in Section Reasoning, for each question, the number of calls to the LLMs is estimated as $2 W D + D + 2$ . Since all the settings of $n - d$ methods have the same W, we only need to compare $D$ to for the evaluation as efficiency $\epsilon \propto D$ . Methods IO, CoT, CoT-SC, and 1-d are not considered because the value of D is fixed at 1.

![](images/6afd9183f2ef58aee3683e673f14bfb6c6130ea04e9b506dce4a2974131a987d.jpg)  
Figure 3: Average Depth versus Max size of community

![](images/341a7af9ce7df7c85d074f3d98461a46985c5d22394386f8674b027472e31a09.jpg)  
Figure 4: Accuracy versus Max size of community

![](images/25e79b88b6b4f4c37e616422b7cacbee4e765fe326fca96c02960b5f6801a836.jpg)  
Figure 5: Accuracy $( \% )$ of two pruning methods

Fig. 3 illustrates the relationship between the max size of the community and average depth on the CWQ and WebQSP datasets, where $M a x S i z e ~ = ~ 1$ refers to the previous n-d 1-w works like ToG. It is evident that FastToG $( M a x S i z e > 1 )$ ) outperforms the n-d 1-w methods. Even with the communities with $M a x S i z e = 2$ , there can be a significant reduction in the AvgDepth of the reasoning chains. For example, the AvgDepth on CWQ is reduced by approximately 0.2 and 0.4 for $\mathfrak { t } 2 \mathfrak { t } / \mathfrak { g } 2 \mathfrak { t }$ modes, respectively. On WebQSP, these reductions increase to about 1.0 for both modes. Moreover, as the $M a x S i z e$ increases, the decrease in depth becomes less substantial.

Table 3: Accuracy $( \% )$ of Community Detection Algorithms   

<html><body><table><tr><td colspan="2"></td><td>Rand</td><td>Louvain</td><td>GN</td><td>Hier</td><td>Spectral</td></tr><tr><td>CwQ</td><td>t2t g2t</td><td>40.4 42.6</td><td>43.8 45.0</td><td>44.0 45.7</td><td>44.5 46.0</td><td>44.0 45.3</td></tr><tr><td>WebQSP</td><td>t2t g2t</td><td>60.6 62.5</td><td>65.2 65.8</td><td>66.8 66.9</td><td>66.0 66.1</td><td>65.1 66.7</td></tr></table></body></html>

# Ablation Study

Trade-off between Accuracy and Efficiency While larger communities can reduce the average depth of reasoning chains, such large communities may also bring more noise, potentially negatively impacting the reasoning effectiveness of LLMs. Fig. 4 illustrates the relationship between the MaxSize and accuracy. Overall, most of the cases achieve higher accuracy when $M a x S i z e$ is set to 4. However, at $M a x S i z e = 8$ , results show a decrease in accuracy. Therefore, setting a larger size of the community does not necessarily result in higher gain from accuracy.

Comparison on Pruning Methods To validate the effectiveness of our proposed Modularity-based Coarse Pruning, we compared it with Random Pruning, a method widely used by previous works. Fig. 5 depicts the accuracy comparison between Modularity-based coarse pruning and random pruning on the CWQ and WebQSP datasets for 2 modes, with all cases using $M a x S i z e = 4$ . Overall, Modularitybased Coarse Pruning outperforms Random Pruning in all cases. Particularly, we observed that cases based on $\mathrm { g 2 t }$ mode are more sensitive in modularity-based pruning, indicating that communities of densely connected structure are preferable for the conversion between graphs and text.

Comparison on different Community Detection Community detection is a crucial step in FastToG. Tab. 3 compares the impact of different detection algorithms — including Louvain, Girvan-Newman (abbr. GN), hierarchical clustering (abbr. Hier), and Spectral Clustering (abbr. Spectral) on accuracy. Additionally, we consider random community detection (abbr. Rand), which randomly partitions nodes into different groups. Comparing the 4 non-random algorithms, the impact of different community detection on FastToG is very small $( < ~ 1 \%$ on average). On the other hand, when comparing algorithms between random and non-random, the latter outperforms the former $( > 3 \%$ on average), demonstrating that community detection does help FastToG.

# Case Study

For the query “Of the 7 countries in Central America, which consider Spanish an official language?” from dataset CWQ, we visualize the retrieval process of FastToG. Fig. 6 displays a snapshot of LLMs pruning with start community Central America. Note that the node Central America has 267 onehop neighbors, making it hard to visualize. Tab. 4 shows the corresponding Graph2Text output of the communities or nodes. The left column shows part of the pruning with communities as the units to be selected, while the right column shows the nodes. As we can see, community as the basic unity for pruning greatly reduces the number of unit. In addition, Community as the unit can increase the likelihood of solving this problem. For example, comparing two option A, the community associated with the option A in left column has a path connecting to the entity Spanish. Thus, we can sure that EI Salvador is one of the answers. In contrast, when using Node as the unit, it requires more exploration to reach this entity. Hence, “think” on the community not only simplifies the pruning process for LLMs but also enhances clarity for users in understanding the reasoning process.

![](images/f3c0c0cff12a52c48d22ce6ffaeb983a0a50fa12e5c848500c3ead671b8937eb.jpg)  
Figure 6: Visualization of the retrieval by FastToG

Table 4: Think on Community versus Think on Node   

<html><body><table><tr><td>Think on Community</td><td>Think on Node</td></tr><tr><td>A.The Sonsonate Department is located in El Salvador,which is part of Central America. Spanish is the language used in the Sonsonate Department..</td><td>A. El Salvador is belong to Central America</td></tr><tr><td>B.The Nuttall Encyclopida describes Mexico City as a city in Central America, which is part of North America. The Centralist Republic of Mexico ..</td><td>B.Nueva especie de Phrynus is main subject in Central America.</td></tr><tr><td>C.The first record of genus Piaroa Villarreal, Giupponi & Tourinho,2008,is from Central America. Carlos Vquez,a Panamanian author, has written about..</td><td>.. more than 20 options Z. The South and Central Ameri- can Club is held in Central America.</td></tr></table></body></html>

# Conclusion

We introduced a novel GraphRAG paradigm - FastToG. By enabling large models to think “community by community” on the knowledge graphs, FastToG not only improves the accuracy of answers generated by LLMs but also enhances the efficiency of the RAG system. We also identified areas for further improvement, such as incorporating semantics and syntax with community detection, introducing multi-level hierarchical communities to enhance retrieval efficiency, and developing better community-to-text conversion.