# A Systematic Exploration of Knowledge Graph Alignment with Large Language Models in Retrieval Augmented Generation

Shiyu Tian1, Shuyue $\mathbf { X _ { i n g } } ^ { 1 }$ , Xingrui $\mathbf { L i } ^ { 1 }$ , Yangyang Luo1, Caixia Yuan1, Wei Chen2, Huixing Jiang2\*, Xiaojie Wang1

1Beijing University of Posts and Telecommunications $^ 2 \mathrm { L I }$ Auto Inc. {tiansy, xsy84160158, liangy298, luoyangyang, yuancx, xjwang}@bupt.edu.cn jianghuixing, chenwei10 @lixiang.com

# Abstract

Retrieval Augmented Generation (RAG) with Knowledge Graphs (KGs) is an effective way to enhance Large Language Models (LLMs). Due to the natural discrepancy between structured KGs and sequential LLMs, KGs must be linearized to text before being inputted into LLMs, leading to the problem of KG Alignment with LLMs (KGA). However, recent $\mathtt { K G } \mathtt { + R A G }$ methods only consider KGA as a simple step without comprehensive and in-depth explorations, leaving three essential problems unclear: (1) What are the factors and their effects in KGA? (2) How do LLMs understand KGs? (3) How to improve $\mathtt { K G } \mathtt { + R A G }$ by KGA? To fill this gap, we conduct systematic explorations on KGA, where we first define the problem of KGA and subdivide it into the graph transformation phase (graph-to-graph) and the linearization phase (graph-to-text). In the graph transformation phase, we study graph features at the node, edge, and full graph levels from low to high granularity. In the linearization phase, we study factors on formats, orders, and templates from structural to token levels. We conduct substantial experiments on 15 typical LLMs and three common datasets. Our main findings include: (1) The centrality of the KG affects the final generation; formats have the greatest impact on KGA; orders are model-dependent, without an optimal order adapting for all models; the templates with special token separators are better. (2) LLMs understand KGs by a unique mechanism, different from processing natural sentences, and separators play an important role. (3) We achieved $7 . 3 \%$ average performance improvements on four common LLMs on the KGQA task by combining the optimal factors to enhance KGA.

# 1 Introduction

Large language models (LLMs) (Zhao et al. 2023b) have demonstrated superior capabilities and generalizability across various tasks (Bang et al. 2023; Ye et al. 2023) and are regarded as a step toward realizing Artificial General Intelligence (AGI) (Bubeck et al. 2023; Xi et al. 2023). However, recent studies reveal that LLMs lack expertise and up-to-date knowledge (Schick et al. 2023; Peng et al. 2023), suffer from hallucination (Rawte, Sheth, and Das 2023), especially in knowledge-intensive tasks (Bang et al. 2023).

male JeremyBieber 58.1 55.4 50.5 16.2 41.9 son   
gender father flat triples (Justin Bieber, father, Jeremy Bieber) (JeremyBieber,son, Jaxon Bieber) $\textcircled{2}$   
path-based $$ father $$ JeremyBieber .→son → Jaxon Bieber $\textcircled{3}$ KG to text Justin Bieber is a male and his father is Jeremy Bieber. Jeremy Bieber hasasonnamed Jaxon $\textcircled{4}$ digraphG { KG to code $$ male[label $\mathbf { \tau } = \mathbf { \tau }$ "gender"] Jeremy Bieber $$ Jaxon Bieber[ labe $\mathbf { \tau } = \mathbf { \tau }$ "son"] }

To address these issues, the current mainstream approach builds Retrieval-Augmented Generation (RAG) systems that enable LLMs to access external knowledge to improve generation quality (Gao et al. 2024).

Knowledge Graphs (KGs) are one of the most commonly used external knowledge sources and have unique advantages: (1) They are structured and easy to retrieve (Ji et al. 2022). (2) They represent the core semantics of knowledge in the form of triple, which is aligned with human cognition (Zhao et al. 2022; Lin et al. 2024). (3) They support symbolic reasoning, which can give human-readable reasoning paths for interoperability (Abu-Salih 2021). Thus, incorporating KGs into LLMs by RAG $( \mathsf { K G } { + } \mathsf { R A G } )$ is a promising way to elevate LLMs to the next level (Pan et al. 2024).

Since there is a natural discrepancy between structural KGs and sequential LLMs, KGs must be linearized into text to be fed into LLMs, which raises a unique problem:

KG Alignment with LLMs (KGA), i.e., how to input KGs into LLM. Although recent KG+RAG studies have explored some factors in KGA, such as formats (Zhang et al. 2024; Wu et al. 2023), orders (Li et al. 2023b; LUO et al. 2024a), and templates (Jiang et al. 2023b; Wen, Wang, and Sun 2024), they only consider KGA as a simple step without systematic and in-depth studies. However, we find KGA is critical and indispensable, directly affecting the efficiency of utilizing the retrieved knowledge and the quality of the final generation (see Figure 1 as an example). Moreover, a deficient KGA can even corrupt the original capabilities of LLMs, leading to knowledge conflict and worse responses (Longpre et al. 2021; Zhou et al. 2023).

Although previous works have extensively studied the $\mathtt { K G } \mathtt { + R A G }$ , three key questions remain unclear: (1) What are the influencing factors and their effects in KGA? (2) How do LLMs understand KGs, and what is the underlying mechanism? (3) How to use KGA to improve KG+RAG?

To address the above problems, we systematically explore KGA. We subdivide KGA into two phases (see Figure 2): (1) the graph transformation phase, where we perform graph-tograph transformation by modifying the retrieved sub-KG at the graph level; (2) the linearization phase, in which the subKG is linearized to achieve graph-to-text conversion. Then, we examine the influencing factors at each phase. In the graph transformation phase, we explore 81 graph features, according to Graph Theory (West et al. 2001), from node, edge, and full graph granularities to study their impact on the final generation and identify the most crucial features by feature analysis methods. In the linearization phase, we investigate the performance of 13 formats, 13 orders, and 14 templates across 15 models, 3 datasets, and 2 enhancement tricks (few-shot and fine-tuning), providing systematic and generalizable results. We completed more than 3,500 experiments covering almost all influencing factors in KGA and identified the effects of each kind of factor. Based on these results, we draw conclusions from the perspective of interpretability and usability. For interpretability, we uncover a distinct mechanism for LLMs to process KGs, which varies from processing general sentences. For usability, we find and utilize the combinability between different factors to enhance the KGA, which can effectively improve the performance of $\mathbf { K G } { + } \mathbf { R A G }$ .

Our contributions are summarized as follows:

• We first define and emphasize the problem of KG Alignment with LLMs (KGA), which is a critical and indispensable part of $\mathrm { K G + R A G }$ , and conduct systematic experiments to fill this gap.   
• We identify key factors and their effects in KGA, including (1) the centrality of the sub-KG affects the final generation, (2) formats are the most influential factor in KGA, (3) orders are model-dependent, without one that is optimal for all models, and (4) templates with special token separators work better.   
• We find LLMs process KGs through a unique mechanism (circuits) by performing mechanistic interpretable analysis, where separators play an important role.   
• Based on our findings, enhancing the KGA can effec

tively improve the average performance of four typical LLMs on the $\scriptstyle \mathrm { K G + R A G }$ task by $7 . 3 \%$ .

# 2 Related Work

# 2.1 $\mathbf { K G + R A G }$ with LLMs

Based on the way of fusing KGs, recent $\mathtt { K G } \mathtt { + R A G }$ studies can be categorized into two types: (1) explicit fusion, where KGs are converted to text and concatenated into prompt; (2) implicit fusion, where additional KG encoding modules are added to the LLMs so that the KGs can be directly inputted into the LLMs.

Most of the current methods are explicit fusion, where they usually focus on designing the retrieval methods (Baek, Aji, and Saffari 2023; Li et al. 2023a; Jiang et al. 2023b; LUO et al. 2024b; Zhang et al. 2024; Xiong, Bao, and Zhao 2024) and the overall pipeline (or LLM agents) (Li et al. 2023b; Wu et al. 2023; Sun et al. 2024; Wen, Wang, and Sun 2024; Sanmartin 2024; Dong et al. 2024). Although these methods achieve good results, they ignore the importance of KGA without considering the structural characteristics of KG itself, resulting in non-optimum performance.

Some studies try to integrate KGs directly into LLMs by designing additional KG encoding modules. Such as GNP (Tian et al. 2024b), KG-Adapter (Tian et al. 2024a) and GRetriever (He et al. 2024). They design different KG encoders and use LoRa or other methods to fine-tune the LLM and KG encoders simultaneously. Although these methods bypass the KGA process by utilizing KG encoders to input the KG directly, they need training LLMs and KG encoders, resulting in high costs. Furthermore, since the KG representations and the textual representations exist in different vector spaces, suffering from the heterogeneous representation problem (Lin et al. 2019; Sun et al. 2022), causing them less effective than the explicit fusion methods.

# 2.2 Mechanistic Interpretability

Mechanistic interpretability is a sub-field of LLMs interpretability that understands language models by investigating individual neurons and especially their connections in terms of circuits (Zhao et al. 2023a). A foundation work conceptualizes the operation of transformers in a new way, providing a mathematical framework for transformer circuits, using that they find “induction heads” that can explain in-context learning in small models (Elhage et al. 2021). Another work (Olsson et al. 2022) validates if “induction heads” still hold true on much more complex state-of-art models. Then, many of the following works try to discover more circuits for different tasks, like “indirect object identification” (Wang et al. 2023) and “greater-than” (Hanna, Liu, and Variengien 2023). Besides, some works are trying to explain different behaviors in LLMs. For example, Chughtai, Chan, and Nanda (2023) study the universality hypothesis by presenting a novel algorithm, Nanda et al. (2023) investigate the recently-discovered phenomenon of “grokking”, Meng et al. (2022) locate factual knowledge in LMs, and Todd et al. (2024) discover task vectors. Recently, new theories and analytical tools have been proposed that greatly advance the field (Nanda and Bloom 2022; Ferrando and Voita

(a) The KG+RAG Process Datasets Pleasestructirnbased Tricks Question: ①CWQ②WQSP on the given KG. Oew-shot Who is the brother of Justin Bieber? Retrieval KGA Prompt: KG imaleJ genderfather son Jeremy Bieber! gender mal Jreemy eer fT u Wiesnre on the given KG. Jaxon Bieber. (Jeremy Bieber,son,Jaxon Bieber) LLMs retrieved modified Linearized KG sub-KG (G) sub-KG (G')   
(b) Factors Graph TransformationPhase Lingarizaton Phase (c) Interpretability from low to high granularity from structural to token level conclude ↓ 1 ↓ 1 ? Node Edge Graph Formats Orders Templates Models + 1 1 1 1   
营 丰 E 号 0 福   
C 三 ④is triad 三 B-8B 三 m di $\mathbf { \Sigma } = \mathbf { \Sigma }$ directed graph ①travel: BFS (undi) ③<head><relation><tail> <sep> ③ChatGPT ■enhance text pahse undi $\mathbf { \sigma } = \mathbf { \sigma }$ undirected graph②travel: DFS (undi) LhMrMtRO=>①(h,r,t), GPT-40 conclude Q3: How to improve KG-RAG Q1: What are the factors and their effects in KGA? by KGA?

2024; Tufanov et al. 2024). Although many phenomena have been explained, there is still a long way to go in fully understanding LLMs. There are no interpretability studies for how LLMs understand KGs. Hence, we take the first step to fill this gap by analyzing the information flow routes of KGs in LLMs.

# 3 KG Alignment with LLMs

# 3.1 Definitions

First, we will define Knowledge Graph (KG) and KG Alignment with LLMs (KGA).

Definition 1. Knowledge Graph. Given an entity set $\boldsymbol { E } ~ = ~ \{ e _ { 1 } , . . . , e _ { N } \}$ and a relation set $\boldsymbol { R } \ = \ \{ r _ { 1 } , . . . , r _ { M } \}$ . A knowledge graph is a graph constructed by $E$ and $R$ : $G = \{ E , R \}$ . If there is a directed edge $r _ { i j }$ from $e _ { i }$ to $e _ { j }$ , then it can construct a triple: $T _ { m } = T _ { i j } = ( { \bar { e } } _ { i } , r _ { i j } , e _ { j } ) , m { \bar { \leq } }$ $M$ . KG can also be defined as a set of triples $\begin{array} { r l } { G } & { { } = } \end{array}$ $\{ T _ { 1 } , T _ { 2 } , . . . , T _ { k } , . . . , T _ { M } \}$ , where each $T _ { k }$ is a triple.

Definition 2. KGA. KGA is the process of converting structural KGs into sequential texts while maximizing model performance. We subdivide it into the graph transformation phase and the linearization phase. Previous studies only examined the linearization without considering that LLMs may also have preferences for certain graph features, i.e., some graph features of the sub-KG will affect the final generation.

Therefore, we introduce the graph transformation phase to study the impact of graph features and how to better align sub-KG with LLMs by graph-level modifications.

The first phase is to modify the graph at the graph level:

$$
G ^ { \prime } = f _ { G } ( G )
$$

where $f _ { G }$ are graph operations, such as adding reverse edges or changing the graph type. In the second phase, the updated graph $G ^ { \prime }$ is converted into text:

$$
T e x t = f _ { T } ( G ^ { \prime } ) = f _ { T } ( \bigcup _ { k } ^ { M } ( T _ { k } ) ) = f _ { T } ( \bigcup _ { i , j } ( e _ { i } , r _ { i j } , e _ { j } ) )
$$

$$
= f _ { f } ( f _ { o } ( \bigcup _ { i , j } f _ { t } ( e _ { i } , r _ { i j } , e _ { j } ) ) )
$$

where $f _ { T } = \{ f _ { f } , f _ { o } , f _ { t } \}$ is a set of methods that transforms triples into texts, $f _ { f } , f _ { o } , f _ { i }$ are three specific methods in formats, orders, and templates. So, the target of KGA can be defined as below:

$$
\underset { f _ { G } , f _ { T } } { \arg \operatorname* { m a x } } l o g P _ { \theta } ( y | Q , f _ { T } ( f _ { G } ( G _ { q } ) ) )
$$

where $Q$ is a user question, $y$ is the true answer, $G _ { q }$ is a subKG related to $Q , \theta$ is a frozen LLM, and the target of KGA is to find two best operations $f _ { G }$ and $f _ { T }$ to maximize the probability of generating the correct answer.

# 3.2 Studied Factors

After defining the KGA, we examine the influencing factors and their effects at each phase. See Figure 2 for our studied factors.

Graph Transformation Phase We aim to examine how graph features impact the final generated results, determine their relative importance, and how to use them to guide the performance improvement of $\mathbf { K G } { + } \mathbf { R A G }$ .

Based on Graph Theory (West et al. 2001), a graph has features from three dimensions: node, edge, and full graph. Accordingly, we selected 25 node features, 8 edge features, and 48 graph features to study. We only list some typical features here (see Appendix for details of all features).

• Node Features (25): degree centrality, laplacian centrality, average neighbor degree, eccentricity, pagerank, etc. • Edge Features (8): betweenness centrality, preferential attachment, jaccard coefficient, etc. • Graph Features (48): vertex cover size, minimum cut, global reaching centrality, s-metric, non-randomness, dominating set num, etc.

Linearization Phase In the linearization phase, we focus on three kinds of factors from structural level to token level: (1) formats, (2) orders, and (3) templates. The formats determine the overall structure of the linearized text, and the orders determine the positional relationships between the triples, while the templates determine the specific representation of each triple. Thus, these three categories of factors encompass the complete process of KGA in the linearization phase.

The formats determine what form of text the sub-KG will be transformed into (see Figure 1 as an example). Flat triples are the most common way to convert KGs to text, which fills the entities and relations of the triples into a pre-defined template. The path-based formats fill the retrieved paths (i.e., multi-hop paths from entities in the question to entities in the possible answers) into path-style templates, and we explore three algorithms to retrieve paths: simple path, Dijkstra short path, and Bellman short path. KG-to-Text is a unique format that designs and trains models to convert KGs to natural language sentences, where we use MVP (Tang et al. 2023), the SOTA KG-to-Text method. Graph description language (GDL) is a special language that can convert a graph into code-like formal text, and we test the two most commonly used GDLs: DOT (Gansner, Koutsofios, and North 2006), and GML (Brandes et al. 2013).

Orders decide the positional relations between different triples, often used in RAG systems’ post-retrieval process (i.e., re-ranking). We study two kinds of order methods: semantic similarity-based and graph travel-based. For the semantic similarity-based methods, we try two sentence embedding models (all-mpnet-base-v2 1 and BGE-M3 (Chen et al. 2024)) and four rank strategies (ascending or descending order based on similarity to the question or answer), totaling eight combinations. For the graph travel-based methods, we test two graph traversal algorithms, BFS and DFS, combined with directed and undirected graphs, resulting in four combinations.

Templates determine the specific representation of each triple transformed into text. Basically, a template is a set of separators, and we define them as left separator (L), right separator (R), middle separator (M), and outer separator (O) based on their position in the template. Formally, a template is $\mathbf { L } h \mathbf { M } r \mathbf { M } t \mathbf { \bar { R 0 } }$ . where $h , r , t$ are the head entity, relation, and tail entity from a triple, $\mathbf { L } , \mathbf { M } , \mathbf { R } , \mathbf { O }$ are separators in the template, which could be various punctuation marks or special tokens. We explore 14 templates used in previous methods.

# 4 Experimental Setup

# 4.1 Datasets

We use three commonly used KGQA datasets: GraphextQA (Shen et al. 2023), ComplexWebQuestions(CWQ) (Talmor and Berant 2018) and WebQuestionsSP(WQSP) (Yih et al. 2016). More details are in the Appendix.

# 4.2 Models

We use 15 models total from 125M to $^ { 1 7 5 \mathrm { B } + }$ , including: Llama-2-7b-base, Llama-2-7b-chat, Llama-2-13b-chat, Llama-3-8b-base, Llama-3-8b-instruct (Touvron et al. 2023; Meta 2024); Mistral-7b-v0.1, Mistral-7b-instruct (Jiang et al. 2023a); Phi-3-mini-128k-instruct (Abdin et al. 2024); opt- $1 2 5 \mathrm { m }$ , opt- $3 5 0 \mathrm { m }$ , opt-1.3b, opt-2.7b, opt-6.7b (Zhang et al. 2022); ChatGPT, GPT-4o (OpenAI 2021).

# 4.3 Enhancement Tricks

We test the performance of different factors under few-shot (Brown et al. 2020) and fine-tuning by LoRa (Hu et al. 2022) as two enhancement tricks for generalizability.

# 4.4 Evaluation Metrics

We follow previous work (Jiang et al. 2023b) to use Hits $@ 1$ to evaluate the correctness of the prediction, which assesses whether the predicted answer is correct or not, suitable for evaluating LLMs (Adlakha et al. 2024).

Rank-biased Overlap (RBO) (Webber, Moffat, and Zobel 2010) measures the similarity between incomplete rankings and weight high ranks more heavily than low ranks. We use it to measure the consistency of different factors across settings to indicate generalizability.

# 4.5 Implementation Details

For the graph transformation phase, we use the Python library NetworkX (Hagberg, Schult, and Swart 2008) to compute the graph features2, and use seven feature analysis methods: Permutation Importance, feature importance in Random Forests and XGBoost, Pearson Correlation, Recursive Feature Elimination, PCA, and ANOVA to calculate the importance scores for each graph feature.

80 80 80 星 60 60 60 E 40 40 40 H 20 20 上 20 14 89 888088 上 1489 88 88 88 8890 8989 46 45 89 78 80 75 78 78 78 77 77 80 80 888477727472665161593939 38 77 78 747676747679807778 37 777875 72 73767577 7676 77 71 73 7467676761746968414141 72 74 69 74 71 69 74 777873 72 7474 2 727174 74 7474 475 7078 707978665371394949 70 77 68 733 70 70 7677 77 74 28 765 734 745 723 787273 74 717078 62 63 63 69 70 69 59 61 53 41 70 69 70 69 69 71 2 30 67 69 70 70 66 65 70 67 66 70 69 68 7265646466 19 16 0 28 2828 66 64 70 63 66 69 6560 59 64 64 33 32 68 54 73 63 668265 74 52 64 61 54 56 52 84 76 74 63 56 9 39 0 26 25 77 73 63 74 24 55 553 59 61 60 220 21 6 63 63 64 62 66 62 64 62 64 62 63 59 53 6061 32 1528 55 54 55 54 53 57 55 2 5 64 67 63 57 54 40 57 58 56 6 6968 61616155 8 3 53 59 55 5957 53525046 46 13 14 68 41 53 287233 554958424746 42 48 34353428232318 15 15 3936 36 3837 3435 25 4.84.7 83 7841 7643463645 24 353434 44 2 9 9 9 35 40373937353533 24 24 4.1 4.2 56 62 26 42 4669 54 37503644413836 106 31919192212 150 9 9 9 16171817 18 4.54.6 64 697976356186.97.71311136.6 (a) 1  π 4 1112221 28 3 8 106 c 39364636393855374 1 18 1 1 (e)8 14 47 94 13 15 2236 36 19 40353218 25 GR RBO=0.73 g RBO=0.82 RBO=0.60 tge 86102 7 9 5 3 114120 910 4 81211 05 3 2 6 71281113 5 6410903 507912431121 8106 3762501111248109 91003 5118461212137 GR RBO=0.65 RBO=0.70 2 RBO=0.60 -1.5 CWphextQA data -1 GraphextQA WQS -1 WanextQA data WQSP 610 812 2 11 0 4 10 12 2 5 12 13 70636941414163616059625358 GraQA697065646463666466596032 33 GraQA 6463646263656363626459666262 53 534954535337363331293021 WQSP 43 4238363737353733323237 36 WQSP 3838373837363837373538353535 wQ 53534253535321221919171716 CWQ 37 3522212021201919 16152120 CWQ2121212121202021212122171919 (b) 5 7 910FormatiDs4 111 122 3 (d) 1 351 11 OrderI1264 8910 (f) 54 3 20 10Template1s87 121113

For the linearization phase, we use vLLM (Kwon et al. 2023) with a temperature of 0 and a top-p of 0.25 for inference, a greedy decoding strategy, and a zero-shot prompt for most experiments. We extract the model generation’s first and last lines to compute the metrics. We conduct case studies of interpretability analysis using Information Flow Routes (Ferrando and Voita 2024), one of the mechanistic interpretability tools for explaining LLMs’ behavior. For ChatGPT and GPT-4o, we random sample $10 \%$ data to test.

See the Appendix for more details.

# 5 Results and Discussions

# 5.1 Effects of Factors

Graph Features Table 1 shows the graph features with the top-10 importance scores averaged over seven feature analysis methods. We find that the most important features are mostly associated with centrality, which measures the importance (or how “central” a node is in the graph) of various nodes in a graph. This may be because nodes with higher centrality appear in more triples and recur more often in the linearized text. Based on that, we design two methods to improve the centrality of a sub-KG: adding a virtual global node and adding reverse edges. Surprisingly, both methods improve the final performance on different models (in Table 2), which suggests that the graph features of sub-KG can affect the generation results. However, few existing KG-RAG methods have considered graph features and added them to the system design, so there is still a lot of potential for improvement in existing methods.

Table 1: Top-10 graph features, avg, std, and range representing the mean, standard deviation, and range of their rankings among seven feature analysis methods.   

<html><body><table><tr><td>Graph Features</td><td>Avg</td><td>Std</td><td>Range</td></tr><tr><td>degree_centrality</td><td>42.2</td><td>13.8</td><td>69.1</td></tr><tr><td>betweenness_centrality</td><td>45.8</td><td>11.4</td><td>69.8</td></tr><tr><td>average_neighbor_degree</td><td>46.4</td><td>11.2</td><td>64.4</td></tr><tr><td>information_centrality</td><td>48.5</td><td>11.6</td><td>69.1</td></tr><tr><td>current_flow_betweenness_centrality</td><td>49.6</td><td>15.0</td><td>75.6</td></tr><tr><td>global_reaching_centrality</td><td>49.8</td><td>11.5</td><td>74.1</td></tr><tr><td>closeness_centrality</td><td>50.6</td><td>13.1</td><td>70.7</td></tr><tr><td>katz_centrality</td><td>52.3</td><td>13.5</td><td>64.2</td></tr><tr><td>eccentricity</td><td>52.3</td><td>16.1</td><td>90.9</td></tr><tr><td>non_randomness</td><td>52.8</td><td>11.7</td><td>73.7</td></tr></table></body></html>

Formats In Figure 3(a), the flat triple methods $( \mathrm { i d } { = } 1 2 , 1 1 , 0 )$ ) achieve the best average performance since they keep all information from sub-KG. Path-based approaches using different path algorithms obtained similar scores $( \mathrm { i d } { = } 9 , 5 , 7 )$ , suggesting that these algorithms are not vital influencing factors, but the graph type (directed or undirected) displays a significant influence $\scriptstyle \mathrm { i d } = 8 , 1 0 , 6 )$ . KG-to-Text methods $( \mathrm { i d } { = } 1 , 2 )$ ) are not as good as in previous work (Wu et al. 2023) because the KG-to-Text model (MVP) is often missing information or even generating non-existent information, implying that this approach needs training on specific datasets with low generalizability. GDL methods $( \mathrm { i d } { = } 4 , 3 )$ are ineffective on small LLMs as they struggle to understand formal code but perform comparably to flat triples on larger LLMs such as Chatgpt and GPT-4o.

![](images/d18623e74984c7345fad5a2b537c6cc7023ed44510df3b160dc3a77307e4a6aa.jpg)  
Figure 4: Case studies: information flow routes of opt- $1 2 5 \mathrm { m }$ by different inputs to show how the model processes various factors. Nodes are representations in residual streams. Edges mean via residual (gray), attention (green), or FFN (pink).

We compare the information flow of different formats to understand why flat triples perform better. As shown in Figure 4(a),(d), their information flow is quite distinct, and there is a solid connection between the separators (Figure 4(a)), which “guide” the direction of information flow.

For generalizability, the average RBO on three datasets is 0.65, meaning an optimal method has a roughly $65 \%$ probability of keeping optimal on another dataset, and the RBO of different tricks is 0.73, suggesting the linearization formats are generalizable across datasets and tricks (Figure 3(b)).

Orders Orders are strongly related to models’ preferences without an optimal order adapts all models (Figure 3(c)) and show great generalizability across datasets and tricks (Figure 3(d)) with 0.82 and 0.7 RBO scores. However, previous studies usually rank triples in descending order based on similarity with questions, which is unsuitable for all LLMs. The orders are model-related due to similar information flow across different orders for the same model (Figure 4(a)(b)).

Templates There is a 10.7-point difference between the average score of the best and worst templates, suggesting that templates are an important consideration, which is overlooked in previous studies (Figure 3(e)). We find the templates with special tokens perform better $( \mathrm { i d } { = } 8 \mathord { \sim } 1 3$ ) since the information flow of these special tokens shows a BOS feature (Ferrando and Voita 2024; Cancedda 2024), where the residual stream is not activated in the middle layers (Figure 4(c)). This BOS feature benefits the model’s understanding of the KG by summarizing previous information as a relayer without adding their own linguistic content since memories are stored in the middle layer FFN (Geva et al. 2021). The RBO on different datasets and tricks are both 0.6, showing decent generalizability (Figure 3(f)).

# 5.2 Interpretability: How LLMs understand KGs?

We find that separators play an important role in the LLMs’ understanding of KG. To further investigate, we conduct quantitative and qualitative experiments. We use t-SNE (Van der Maaten and Hinton 2008) to search patterns in the information flows of the LLM and whether there are mechanisms associated with separator types.

From Figure 5, the separators (L, R, and M/O) are clustered in different groups (in most cases, M and O are the same token), while the different types of entities (Eh, Er, and Et) are not well separated, which suggests that the model has special information flow routes in processing these separators with completely different mechanisms than handling entities in triples.

![](images/aab32a3b603547203d96b9d67fc53b2244ce348f5715d94e3331eb22f9a06de9.jpg)  
Figure 5: t-SNE of vectorized contribution weights on 100 random samples, colored by token types. L, R, O, and M are separators. Eh, Et and Er are entities and relations in triples.

![](images/190fdbc5c1747b65756155d33d48b1fecb20d3a1a87af53318eaf045ca8d0266.jpg)  
Figure 6: The normalized contribution weights of attention heads and FFN in each layer, colored by different input token types.

Following previous studies to find circuits (Ferrando and Voita 2024), we calculate the contribution weights of attention heads and FFN in each layer. In Figure 6(a), we find the separators, entities, and words activate different heads, suggesting that the model can distinguish and process them with different components. Notably, the FFN weights vary considerably, with entities and separators having low weights while words have high weights in the middle layer, indicating that the model utilizes different mechanisms for processing general sentences and KGs. Moreover, different types of separators also activate different heads, and some heads are activated by more than one type of separator (Figure 6(b)). This suggests that there are specialized heads within the model to handle different types of separators. We also note that these heads usually appear in the lower layers of the model, where the model mainly processes inputs and extracts information. The FFN block exhibits high activation in the bottom and top layers and low in the middle layer, which is consistent with the case in Figure 4(c), suggesting that some separators have the BOS feature for summarizing previous information rather than introducing new information.

# 5.3 Usability: How to use KGA to improve $\mathbf { K G + R A G ? }$

This section studies how to use KGA to improve the $\mathtt { K G } \mathtt { + R A G }$ performance. According to our findings in the graph transformation phase, the centrality of sub-KG is strongly correlated to the generation performance. As shown in Table 2, improving the centrality by adding reverse edges and a virtual global node can improve the performance, showing the effectiveness of the graph transformation phase and graph features. However, many methods overlook this phase in the design of RAG systems.

We find the factors in the linearization phase are combinable, which means using two optimal methods simultaneously usually remains optimal (more in Appendix). Based on this property, we locate and combine the optimal method for each factor, which considerably improves the performance of the $\mathrm { K G + R A G }$ task (Table 2).

Table 2: Results of the four models with different combinations of factors on the GraphextQA test set. BF, BT, and BO are the best formats, templates, and orders for each model.   

<html><body><table><tr><td rowspan="2">Combinations</td><td colspan="4">Models</td></tr><tr><td>Llama-2-7b</td><td>Llama-3-8b</td><td>ChatGPT</td><td>GPT-40</td></tr><tr><td>base</td><td>72.25</td><td>77.40</td><td>76.12</td><td>87.54</td></tr><tr><td>w/reverse_edges</td><td>68.24</td><td>88.20</td><td>77.51</td><td>88.93</td></tr><tr><td>w/ global_node</td><td>72.73</td><td>84.01</td><td>78.55</td><td>87.54</td></tr><tr><td>w/BF</td><td>72.25</td><td>77.40</td><td>79.58</td><td>87.89</td></tr><tr><td>w/BO</td><td>78.41</td><td>78.20</td><td>79.93</td><td>89.62</td></tr><tr><td>W/BT</td><td>78.24</td><td>82.32</td><td>77.85</td><td>88.93</td></tr><tr><td>W/BF+BO</td><td>78.24</td><td>78.20</td><td>75.09</td><td>91.00</td></tr><tr><td>W/BF+BT</td><td>78.24</td><td>82.32</td><td>79.58</td><td>88.58</td></tr><tr><td>W/BO+BT</td><td>81.90</td><td>82.80</td><td>75.77</td><td>88.58</td></tr><tr><td>W/BF+BO+BT</td><td>83.22</td><td>82.80</td><td>75.09</td><td>91.35</td></tr></table></body></html>

# 6 Conclusion

In this paper, we highlight the problem of KG Alignment with LLMs (KGA) and systematically explore the KGA as a critical part of $\mathrm { K G + R A G }$ . We subdivide it into graph transformation and linearization phases, exploring 81 graph features, 13 formats, 13 orders, and 14 templates across 15 models, 3 datasets, and 2 enhancement tricks, totaling more than 3,500 experiments. From these results, we identify key factors in KGA and their effects. For interpretability, we find a unique mechanism for LLMs to understand KGs. For usability, we show that optimizing KGA effectively improves the performance of $\mathtt { K G } \mathtt { + R A G }$ by $7 . 3 \%$ on average. We believe that our work not only provides an indepth and systematic study of the KGA problem to fill the gap in $\mathtt { K G } \mathtt { + R A G }$ but also opens up new directions for enlightening more powerful $\mathtt { K G } \mathtt { + R A G }$ designs.