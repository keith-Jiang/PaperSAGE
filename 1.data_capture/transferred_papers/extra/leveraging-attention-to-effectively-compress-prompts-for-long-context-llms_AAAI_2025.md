# Leveraging Attention to Effectively Compress Prompts for Long-Context LLMs

Yunlong Zhao1, 2\*, Haoran $\mathbf { W } \mathbf { u } ^ { 1 , 3 * }$ , $\mathbf { B o X u } ^ { 1 , 2 }$ 2

1The Key Laboratory of Cognition and Decision Intelligence for Complex Systems, Institute of Automation, Chinese Academy of Sciences, Beijing, China 2School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, China 3Nanjing Artificial Intelligence Research of IA, Nanjing, China {zhaoyunlong2020, wuhaoran2018, xubo}@ia.ac.cn

# Abstract

Prompt compression is increasingly studied for its potential to reduce computational costs and alleviate the burden on language models when processing lengthy prompts. Prior research has assessed token retention and removal by calculating information entropy. However, prompt compression encounters two significant challenges: (1) Information entropy, while widely used, may not be the optimal compression metric; and (2) The semantic significance of tokens is contextdependent, which renders independent token retention decisions inadequate.

We posit that the solution to these challenges lies in the intrinsic mechanism of language models. Large language models (LLMs) exhibit robust contextual processing capabilities, with recent studies on their internal dynamics revealing that the attention mechanism plays a crucial role in modeling how LLMs leverage long contexts. Building on this insight, we introduce AttnComp, a novel approach that exploits the attention mechanism within language models to guide prompt compression. Our method employs causal cross-attention from the query to the context to evaluate the significance of each token, and we develop a graph-based algorithm to efficiently cluster tokens into semantic units, thus mitigating the issue of independent dependencies.

We conduct experiments on datasets for retrieval-augmented generation and multiple long tasks involving single or multidocument QA. Our proposed method, AttnComp, outperforms previous baselines and validates the contributions of our components through analytical experiments. Compared to other methods that use a causal LM for prompt compression, our approach results in shorter latency and improved performance.

# Introduction

Thanks to advanced prompting techniques, large language models (LLMs) such as ChatGPT have realized notable achievements in domains like Retrieval-Augmented Generation (RAG) (Lewis et al. 2020), Agent (Park et al. 2023), and In-Context Learning (ICL) (Dong et al. 2022). Nonetheless, these advancements present challenges to the longcontext capabilities of LLMs. Recently, the development of prompt compression methods has garnered considerable interest for their potential to enhance the efficiency of context management, reduce computational and economic burdens, and minimize interference in LLMs by removing superfluous content.

Some studies have attempted to compress prompts through retrieval (Xu et al. 2024) or summary generation (Xu, Shi, and Choi 2023) methods, but these approaches each have their limitations, such as coarse retrieval granularity or latency issues with generation methods. Recently, research based information entropy theory has garnered widespread attention, with representative works including the Selective-Context (Xu, Shi, and Choi 2023) and LLMLingua series (Jiang et al. 2023a). These studies utilize a small causal model to calculate the information entropy metric, specifically the PPL, to evaluate the importance of tokens within a prompt and prune it accordingly. However, these current methods face the following issues:

• Information entropy is an empirical metric that assumes redundancy in natural language, which is not always optimal.   
• A fundamental challenge in prompt compression is the assumption of strong independence underlying the process. The algorithm independently evaluates each token, deciding whether to retain or remove it, without accounting for its impact on the comprehensive semantics of the remaining prompt.

Given the challenges faced by prompt compression, two important research questions urgently need to be addressed:

Q1: How can we derive a better metric to measure the importance of information in the context?

Q2: How do we address the independence assumption in prompt compression, ensuring that removing tokens from the prompt does not affect the remaining semantics? We equate this problem to determining which tokens should be collectively considered for removal.

Salvation lies within. We contend that the solutions to the outlined questions can be derived from the intrinsic attention mechanisms of LLMs. Recent research has delved into the attention mechanism of LLMs, elucidating their core operational principles and offering vital insights into their functionalities (Wu et al. 2024). Given their robust capacity to manipulate contextual information, we assert that all requisite attributes for effective long-text compression are innately integrated within these attention mechanisms. Consequently, we introduce AttnComp, a straightforward and efficacious approach for prompt compression that exploits the native attention capabilities of LLMs.

For Question 1, we propose using query-guided crossattention as a metric for evaluating token importance. We utilize causal cross-attention from the query to the context to assess the significance of each token within it. Compared to perplexity (PPL), attention captures richer patterns, enabling more precise identification of question-related, finegrained semantic information. Initially, we identify the retrieval heads in the small causal LLM that integrates contextual information and then apply a maximum strategy to consolidate the importance scores. To address Question 2, we redefine the problem as minimizing semantic dependency in token grouping. We hypothesize that attention values effectively capture the degree of semantic dependency between tokens. Building on the premise that attention values reflect semantic dependencies, we develop a graph-based algorithm to efficiently group tokens into semantic units. These semantic units provide a more cohesive representation of consistent meanings, enabling our compression algorithm to operate effectively at this level, thereby mitigating challenges posed by the independence assumption.

We conduct experiments on the synthetic long-context dataset and the real-world long-context datasets from LongBench. The results show that our approach surpasses previous prompt compression baselines. Furthermore, our experimental analysis validates the effectiveness of the components we proposed.

# Problem Formulation

Given an LLM input with an augmented prompt $\textbf { \textit { x } } =$ $( \pmb { x } _ { d o c } , \pmb { x } _ { q u e r y } )$ , the prompt compression system aims to compress $\pmb { x } _ { d o c }$ to reduce the prompt length while retaining key context information, ensuring it can effectively respond to xquery.

The objective of a prompt compression system can be formulated as:

$$
\operatorname* { m i n } _ { \widetilde { \pmb { x } } } \mathrm { D } \left( L L M \left( \widetilde { \pmb { y } } \mid \widetilde { \pmb { x } } \right) , L L M \left( \pmb { y } \mid \pmb { x } \right) \right) ,
$$

where $\widetilde { \pmb { x } }$ denotes the compressed prompt, which is a subsequenceeof original prompt $\pmb { x } . \widetilde { \pmb { y } }$ and $_ y$ are the outputs generated by the LLM based on $\widetilde { \pmb { x } }$ aend $\scriptstyle { \mathbf { { \boldsymbol { x } } } }$ , respectively. $D ( ; )$ is a distance measure between tweo distributions. In this work, we focus on compressing the document $x _ { d o c }$ , which occupies the largest portion of the prompt. The compressed prompt should be concise to maximize efficiency while remaining informative and faithful to the retrieved evidence documents.

# Method: AttnComp

The series of works by LLMLingua introduces a coarse-tofine framework. The coarse-grained compression primarily retrieves the chunked context, while the fine-grained compression focuses on pruning tokens with low information

Algorithm 1: Pseudo code of our AttnComp

Input: A small language model $M$ , the original context $S _ { c o n t e x t }$ , the query $S _ { q u e r y }$ , the windows size $w$ .

1: Optional: Perform coarse-grained compression using an external retriever $r$ .   
2: if length $( S _ { c o n t e x t } ) > w$ then   
3: Split $S _ { \mathrm { c o n t e x t } }$ into a list of chunks $\mathbb { S } _ { \mathrm { o r i } }$ , where each chunk’s length does not exceed $w$ .   
4: else   
5: $\mathbb { S } _ { o r i } = \{ S _ { c o n t e x t } \}$   
6: end if   
7: Calculate the filtering ratio $p$ based on the compression constraint.   
8: for $C \in \mathbb { S } _ { o r i }$ do   
9: CA, $\mathbf { S } \mathbf { A } = M ( S _ { q u e r y } , C )$   
10: Derive the importance score for each token $t _ { i }$ based on CA.   
11: Get the maximum spanning tree $\mathcal { T } = \mathrm { F i n d M S T } ( \mathcal { G } )$ , where $\mathcal { G }$ is the graph with a weight matrix SA.   
12: Get the semantic units $U _ { 1 } , \dots , U _ { k }$ by applying the community detection algorithm to the subgraph $\tau$ .   
13: Assign an importance score to each semantic unit $U _ { k }$ .   
14: Filter out $p$ percent of the semantic units based on their importance scores.

15: end for

Output: The compressed context.

content. Our method AttnComp emphasizes fine-grained compression and seamlessly integrates with coarse-grained compression. Our contributions are twofold: we propose an cross-attention-based compression metric and introduce a semantic unit identification method with self-attention. Figure 1 illustrates our framework, and Algorithm 1 details the main process of our approach.

# Attention Extraction

In long-context scenarios, prompts typically consist of two parts: the query and the augmented documents or demonstrations, represented as $x = ( x _ { q u e r y } , x _ { d o c } )$ . The key information within the context is typically query-aware, and we leverage a small language model to capture this relationship effectively. Since we are using a causal language model, we append the query at the end of the document to ensure that the query has access to the document’s information. We then extract attention as follows:

$$
\begin{array} { c } { { \bf { A t t n } } = \mathrm { C a s u a l L M } \left( x _ { d o c } , x _ { q u e r y } \right) } \\ { { \bf { C A } } = { \bf { A t t n } } _ { 1 : c , n } } \\ { { \bf { S A } } = { \bf { A t t n } } _ { 1 : c , 1 : c } } \end{array}
$$

where $c$ is the document length, and $n$ is the total prompt length. CA (Query-guided Cross-Attention) denotes the attention from the last token in query to the tokens in documents. SA (Self-Attention) denotes the attention between tokens in the document. CA and SA are employed in two key components of our algorithm: deriving the compression metric and identifying semantic units, respectively.

Semantic Unit Identification Filter units with Compressed Prompt low score The Nobel Prize in Physics is an 一 annual award given by the Royal Swedish Academy of Sciences for those who have made the most Build maximum Spanning Tree Community Detection outstanding contributions to mankind in the field of physics. Assign importance score It is one of the five Nobel Prizes established by the will of Alfred ↑ Merge 1N9o0b1e,lin…1…895 and awarded since 了了 647 tokens Self-Attention Query-guided Cross-Attention Context ? Query ↓ Small Casual Language Model S Target Large Language Model ↑ ↓ The Nobel Prize in Physics is an annual award given by the ……. Question: who got the first nobel prize in physics ? Answer: Wilhelm Conrad Röntgen Context (\~3000 tokens) Query Answer

# Cross-Attention as the Compression Metric

How can we derive a more robust compression metric for evaluating token importance within context? Compared to previous metrics based on information entropy, we believe that the attention mechanism may offer a superior alternative. Recent research (Wu et al. 2024) has identified that certain attention heads, termed retrieval heads, become active in LLMs during context processing. This finding not only sheds light on the underlying mechanisms through which LLMs leverage context but also opens up a new avenue for enhancing our prompt compression method by disentangling the contextual capabilities of LLMs. As a result, we propose using the CA as a compression metric to decide whether to retain or discard tokens in context.

Different attention heads in LLM exhibit unique attention distributions, each reflecting distinct patterns of information utilization. To calculate the final importance score, we apply a max strategy across all attention heads. Building on insights from research on retrieval heads, we first identify the retrieval heads within the LLM that are most relevant to contextual information and select the top 20 to derive the compression metric. Formally, the important score of token at index $t$ are defined as follows:

$$
s c o r e ( t ) = \operatorname* { m a x } _ { h \in H } \left\{ \mathbf { C } \mathbf { A } _ { t } ^ { h } \right\}
$$

where $H$ represents the set of retrieval heads, $\mathbf { C A } _ { t } ^ { h }$ represents the cross-attention scores of attention head $h$ at index $t$ . We adopt a max aggregation strategy for attention scores, as we observe that it more effectively preserves the information integration patterns of the retrieval head compared to averaging.

We utilize a percentile-based filtering method to adaptively select the most informative content. Specifically, we rank the importance scores and filter out the bottom $p \%$ of tokens. The value of $p$ is determined by the target compression constraint. The remaining tokens are then combined to form the compressed prompt

# Semantic Unit Identification with Self-Attention

As previously mentioned, prompt compression faces the challenge of the independence assumption. To further address this challenge, we propose a Semantic Unit Identification algorithm. This approach aims to ensure that tokens within the same unit exhibit strong semantic dependencies, while different units have weak semantic dependencies. Consequently, we can independently assess whether to retain or remove each unit.

Our approach is based on the hypothesis that the attention distribution within language models inherently encodes a mechanism for segmenting semantic units. Specifically, the self-attention between tokens captures their conditional dependencies, where tokens with strong mutual attention are semantically related and should be grouped into the same semantic unit. Conversely, tokens with weaker mutual attention exhibit lower semantic relatedness and should be assigned to different semantic units. Consequently, the task can be formally defined as finding an optimal token grouping that maximizes attention values within groups while minimizing attention values across groups.

Let $S$ represent the set of all tokens, $S = \{ t _ { 1 } , t _ { 2 } , \ldots , t _ { n } \}$ , where $n$ is the number of tokens. Let $\mathbf { S A } _ { p q }$ denote the causal self-attention value between token $t _ { p }$ and token $t _ { q }$ . $U _ { 1 } , U _ { 2 } , \dots , U _ { k }$ represent the $k$ semantic units, which are groups of tokens, where each $U _ { i } \subseteq S$ , and $U _ { 1 } \cup U _ { 2 } \cup \cdot \cdot \cdot \cup$

$U _ { k } = S$ . The problem can be formulated as:

$$
\operatorname* { m a x } _ { \{ U _ { 1 } , U _ { 2 } , \ldots , U _ { k } \} } \sum _ { i = 1 } ^ { k } A _ { \mathrm { i n t r a ~ } } ( U _ { i } ) - \lambda \sum _ { 1 \leq i < j \leq k } A _ { \mathrm { i n t e r ~ } } ( U _ { i } , U _ { j } )
$$

where $\begin{array} { r c l } { A _ { \mathrm { i n t r a } } ( U _ { i } ) } & { = } & { \sum _ { t _ { p } , t _ { q } \in U _ { i } } \mathbf { S } \mathbf { A } _ { p q } } \end{array}$ represents the total attention value within group $U _ { i }$ , and $A _ { \mathrm { i n t e r } } ( U _ { i } , U _ { j } ) =$ tp Ui,tq Uj SApq represents the total attention value between group $U _ { i }$ and group $U _ { j }$ . $\lambda$ is a weighting hyperparameter.

However, this problem presents a highly complex combinatorial optimization challenge. Given the stringent time constraints of our compression algorithm, direct computation is impractical. Therefore, we propose a heuristic method grounded in graph theory to address this problem. We first construct a maximum spanning tree to extract the core semantic structure of the prompt, then apply a community detection algorithm to partition the tokens.

We represent all tokens $S$ as vertices $V$ in a fully connected, undirected graph, with the SA matrix serving as the edge weight matrix between the vertices. We represent this graph as $\mathcal { G } = ( V , w )$ . Note that we apply the max strategy across all heads of SA to obtain scalar-weighted edges. To simplify the problem, we disregard edge direction, considering the attention from token $t _ { i }$ to token $t _ { j }$ as equivalent to the weight between them.

$$
w _ { i j } = w _ { j i } = \mathbf { S } \mathbf { A } _ { i j } { \mathrm { ~ w h e r e ~ } } i > j
$$

The maximum spanning tree (MST) is defined as the spanning tree with the greatest possible total weight compared to any other spanning tree. In a connected graph, a spanning tree is a subgraph that includes all vertices without forming any cycles. The maximum spanning tree has been employed to efficiently solve dependency parsing tasks (McDonald et al. 2005; Stanojevic´ and Cohen 2021). Constructing the maximum spanning tree is consistent with the objective of our optimization problem.

$$
\mathcal { T } = \mathrm { F i n d M S T } ( \mathcal { G } )
$$

Leveraging the MST, we transform the simple sequential links of tokens in the prompt into a tree-like topological structure, which highlights the semantic connections between tokens. Within this graphical structure, our goal is to identify clusters of closely linked tokens, treating them as distinct semantic units. To segment the MST, we employ the Louvain algorithm (Blondel et al. 2008), a community detection method that uncovers highly modular community structures in large networks:

$$
U _ { 1 } , U _ { 2 } , \dots , U _ { K } = \operatorname { L o u v a i n } ( T )
$$

The importance score of a semantic unit is calculated by averaging the importance scores of all tokens within it:

$$
S ( U _ { k } ) = \frac { 1 } { | U _ { k } | } \sum _ { t _ { i } \in U _ { k } } s c o r e ( i )
$$

We also apply a percentile-based filtering method. For the retained semantic units, the tokens are combined in their original sequence to form the final compressed prompt.

# Experiments

In this section, we describe the experiments conducted to evaluate the effectiveness of our proposed approach.

# Datasets and Evaluation Metric

Our experiments explore synthesized long-context scenarios within the retrieval-augmented generation setting, as well as general long-context scenarios. We use the Natural Question dataset (Kwiatkowski et al. 2019) and datasets from LongBench (Bai et al. 2023), respectively.

Natural Questions (NQ) (Kwiatkowski et al. 2019) is a classic dataset for open-domain question answering. We use a retrieval-augmented setup provided by (Liu et al. 2024). In this setup, each question is paired with 20 documents in the initial prompt, one of which contains the correct answer. The benchmark tests five positions for the ground truth document: 1st, 5th, 10th, 15th, and 20th. We evaluate using the accuracy metric as described by (Liu et al. 2024).

LongBench (Bai et al. 2023) is a benchmark designed to assess the long-context capabilities of LLMs. From this benchmark, we select datasets for single-document and multi-document question answering, including Qasper, MultiFieldQA, NarrativeQA, Musique, HotpotQA, and 2WikiMultiQA. The context lengths in these datasets range from $4 \mathrm { k \Omega }$ to 18k. We utilize the benchmark’s provided metrics and scripts for our evaluation.

# Implementation Details

In this paper, we validate our method on both open-source and commercial models. Following previous work, we use GPT-3.5-Turbo-0613 and LongChat-13B-16k on the NQ dataset, and GPT-3.5-turbo-16k and LongChat-v1.5-7B-32k on LongBench datasets. By default, we also employ the open-source model Llama-2-7B (Touvron et al. 2023) as the small LM for extracting attention, consistent with prior work. Taking into account the latency and Llama2’s 4k window limitation, we set the compression algorithm’s processing window to 2k for extremely long prompts. Data exceeding this window is chunked and processed over multiple passes. We implement our approach using PyTorch and Huggingface’s Transformers (Wolf et al. 2019). Due to the slow execution speed in Python, we implement Prim’s algorithm $( \mathrm { P r i m } ~ 1 9 5 7 )$ in $^ { C + + }$ and access it as a dynamic library. For the Louvain algorithm, we utilize the implementation provided by the Python package community louvain.

Following a similar approach to previous compressionbased methods, we first apply coarse-grained compression similar to Cond.PPL, to achieve a specific compression ratio. Then, we refine the results through fine-grained compression to meet the final constraints. For the NQ dataset, we initially achieve a compression ratio of over $4 \mathrm { x }$ through coarse-grained compression, followed by fine-grained adjustments to satisfy the constraints. Similarly, for the LongBench dataset, consistent with previous work, we set a target compression limit of 2,000 tokens. We first apply coarsegrained compression to reduce the length to 4,000 tokens and then use a $50 \%$ fine-grained filtering process to meet the final compression requirement.

Table 1: Performance of different methods on NaturalQuestions.† indicates the method using coarse-grained compression (i.e., Cond.PPL). The baseline results are directly cited from (Jiang et al. 2023b) and (Pan et al. 2024). For the results under reorder, we apply the reordering strategy, while for the 1st to 20th positions, the documents remain in their original order.   

<html><body><table><tr><td rowspan="2">Methods</td><td colspan="6">GPT3.5-Turbo</td><td colspan="6">LongChat-13b</td><td colspan="2">Length</td></tr><tr><td>1st</td><td>5th</td><td>10th</td><td>15th</td><td>20th</td><td>Reorder</td><td>1st</td><td>5th</td><td>10th</td><td>15th</td><td>20th</td><td>Reorder</td><td>Tokens</td><td>1/ T</td></tr><tr><td colspan="9">Retrieval-basedMethods</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SBERT</td><td>66.9</td><td>61.1</td><td>59.0</td><td>61.2</td><td>60.3</td><td>64.4</td><td>62.6</td><td>56.6</td><td>53.9</td><td>55.0</td><td>59.1</td><td>59.1</td><td>808</td><td>3.6x</td></tr><tr><td>OpenAI</td><td>63.8</td><td>64.6</td><td>65.4</td><td>64.1</td><td>63.7</td><td>63.7</td><td>61.2</td><td>56.0</td><td>55.1</td><td>54.4</td><td>55.8</td><td>58.8</td><td>804</td><td>3.7x</td></tr><tr><td>Cond.PPL *</td><td>71.1</td><td>70.7</td><td>69.3</td><td>68.7</td><td>68.5</td><td>71.5</td><td>67.8</td><td>59.4</td><td>57.7</td><td>57.7</td><td>58.6</td><td>64.0</td><td>807</td><td>3.7x</td></tr><tr><td colspan="9">Compression-based Methods</td><td colspan="5"></td></tr><tr><td>Selective-Context</td><td>31.4</td><td>19.5</td><td>24.7</td><td>24.1</td><td>43.8</td><td></td><td>38.2</td><td>17.2</td><td>15.9</td><td></td><td></td><td></td><td>791</td><td>3.7x</td></tr><tr><td>LLMLingua</td><td>25.5</td><td>27.5</td><td>23.5</td><td>26.5</td><td>30.0</td><td>27.0</td><td>32.1</td><td>30.8</td><td>29.9</td><td>16.0 28.9</td><td>27.3 32.4</td><td>30.5</td><td>775</td><td>3.8x</td></tr><tr><td>LLMLingua-2</td><td>48.6</td><td>44.5</td><td>43.6</td><td>40.9</td><td>39.9</td><td>46.2</td><td></td><td></td><td>1</td><td>-</td><td>1</td><td></td><td>748</td><td>3.9x</td></tr><tr><td>LLMLingua-2†</td><td>74.0</td><td>70.4</td><td>67.0</td><td>66.9</td><td>65.3</td><td>71.9</td><td></td><td>1</td><td>1</td><td>1</td><td></td><td>1</td><td>739</td><td>3.9x</td></tr><tr><td>LongLLMLingua†</td><td>75.0</td><td>71.8</td><td>71.2</td><td>71.2</td><td>74.7</td><td>75.5</td><td>68.7</td><td>60.5</td><td>59.3</td><td>58.3</td><td>61.3</td><td>66.7</td><td>748</td><td>3.9x</td></tr><tr><td>AttnComp</td><td>76.6</td><td>73.3</td><td>73.1</td><td>72.9</td><td>75.0</td><td>76.8</td><td>70.1</td><td>62.1</td><td>61.6</td><td>61.2</td><td>61.3</td><td>68.6</td><td>647</td><td>4.5x</td></tr><tr><td>Original Prompt</td><td>75.7</td><td>57.3</td><td>54.1</td><td>55.4</td><td>63.1</td><td></td><td>68.6</td><td>57.4</td><td>55.3</td><td>52.5</td><td>55.0</td><td></td><td>2.946</td><td>1</td></tr><tr><td>Zero-shot</td><td colspan="5"></td><td></td><td colspan="4">35.0</td><td></td><td></td><td>15 196x</td></tr></table></body></html>

Table 2: Ablation study.   

<html><body><table><tr><td></td><td>Acc 1/T</td></tr><tr><td>AttnComp</td><td>68.6 4.5x</td></tr><tr><td>- w/o Semantic Units</td><td>67.1 4.5x</td></tr><tr><td>- w/Phrase</td><td>67.4 4.6x</td></tr><tr><td>- w/o RetrievalHeads - W/ PPL</td><td>68.2 4.5x</td></tr><tr><td>- W/Iterative Token-level Compression</td><td>61.5 4.5x 67.5 4.5x</td></tr><tr><td>- w/Mistral-7B-v0.2</td><td>68.8 4.5x</td></tr></table></body></html>

# Baselines

Our baselines include retrieval-based methods and compression-based methods.

(i) Retrieval-based Methods. Retrieval-based methods use a retriever to rank documents based on their relevance to the question. They discard sentences or paragraphs with low relevance until the compression constraint is met while preserving the original document order. We select the following retrievers: SentenceBERT (Reimers and Gurevych 2020), OpenAI Embedding, and Cond.PPL (Jiang et al. 2023b) to measure the association between the query and the documents.

(i) Compression-based Methods. We compare our method with state-of-art methods for prompt compression.

• Selective-Context (Li et al. 2023) is the first work to discuss context compression, which prunes redundant lex

ical units by estimating self-information through a language model.   
• LLMLingua (Jiang et al. 2023a) proposes a coarse-tofine approach to manage compression ratio constraints. It employs iterative token-level prompt compression and utilizes perplexity as the compression metric.   
• LongLLMLingua (Jiang et al. 2023b) is an improved version based on LLMLingua. It uses conditional perplexity for coarse-grained compression of the context and contrastive perplexity for fine-grained compression.   
• LLMLingua-2 (Jiang et al. 2023a) defines prompt compression as a token classification task (i.e., preserve or discard). It is a BERT-based model trained on a dataset collected from GPT.

# Main Results

Tables 1 and 3 compare the performance of our method in both RAG settings and general long-context scenarios. The following observations and conclusions can be drawn:

(1) On the NQ dataset under RAG settings, our method outperforms previous baselines in both performance and compression rate. This improvement is evident not only in the open-source LongChat model but also in commercial closed-source models, underscoring the effectiveness of our approach. Moreover, on datasets without reranking strategies, our method also shows improvement, partially mitigating the ’lost-in-the-middle’ phenomenon. (2) On the LongBench dataset, our method significantly enhances performance in most tasks on the open-source LongChat-v1.5- 7B-32k model, surpassing the baseline. For the commercial model, our method also achieves overall performance improvement. However, due to the strong contextual capabilities of GPT, there is a slight performance decline in the Qasper and NarrativeQA tasks.

<html><body><table><tr><td>Model</td><td>Methods</td><td>Qasper</td><td>MultiFieldQA</td><td>NarrativeQA</td><td>MuSiQue</td><td>HotpotQA</td><td>2WikiMultihopQA</td><td>AVG</td></tr><tr><td rowspan="5">LongChat-v1.5-7B-32k</td><td>Original Prompt</td><td>27.7</td><td>41.4</td><td>16.9</td><td>9.7</td><td>31.5</td><td>20.6</td><td>24.6</td></tr><tr><td>LLMLingua</td><td>23.3</td><td>34.8</td><td>14.3</td><td>9.0</td><td>26.2</td><td>20.2</td><td>21.3</td></tr><tr><td>LLMLingua2</td><td>26.2</td><td>32.7</td><td>8.5</td><td>9.3</td><td>24.9</td><td>26.6</td><td>21.4</td></tr><tr><td>LongLLMLingua</td><td>27.0</td><td>39.9</td><td>15.5</td><td>14.7</td><td>33.2</td><td>22.4</td><td>25.5</td></tr><tr><td>AttnComp</td><td>30.4</td><td>43.8</td><td>15.8</td><td>19.8</td><td>40.0</td><td>27.6</td><td>29.6</td></tr><tr><td rowspan="5">GPT3.5-Turbo</td><td>Original Prompt</td><td>43.3</td><td>52.3</td><td>23.6</td><td>26.9</td><td>51.6</td><td>37.7</td><td>39.2</td></tr><tr><td>LLMLingua</td><td>25.9</td><td>35.7</td><td>12.4</td><td>12.8</td><td>38.5</td><td>35.3</td><td>26.8</td></tr><tr><td>LLMLingua2</td><td>37.0</td><td>42.5</td><td>14.7</td><td>22.1</td><td>44.3</td><td>42.2</td><td>33.8</td></tr><tr><td>LongLLMLingua</td><td>37.1</td><td>48.9</td><td>17.2</td><td>30.0</td><td>48.8</td><td>48.9</td><td>38.5</td></tr><tr><td>AttnComp</td><td>38.8</td><td>55.1</td><td>21.5</td><td>30.2</td><td>58.4</td><td>42.7</td><td>41.1</td></tr></table></body></html>

Table 3: Performance of different methods on LongBench. The target compression constraint is 2000 tokens.The original prompt results are directly cited from (Bai et al. 2023).

![](images/40380210c9a98ec883d5935c45b7cc1954938ac7276b287f19b135f8c49c2692.jpg)  
Figure 2: The effect of filtration rate across different tokenlevel compression metrics.

# Analysis

We conduct additional analysis experiments to better demonstrate the effectiveness of our method. All experiments are carried out on the NQ dataset, using LongChat13b as the target LLM.

# Ablation Study

To further validate the contributions of the various components in our method, we conduct ablation experiments on the following model variants: (1) Ours w/o Semantic Units, which compresses the prompt at the token level; (2) Ours w/ Phrase Units, which employs spaCy to merge tokens into phrase units and compresses the prompt at the phrase level; (3) Ours w/ Retrieval Heads, which uses all attention heads instead of only retrieval heads; (4) Ours w/ PPL, which uses PPL as the compression metric; (5) Ours w/ Iterative Tokenlevel Compression, which uses iterative compression as in LongLLingua; (6) Ours w/ Mistral-7B-v0.2, which uses another recent small LM for fine-grained compression.

Table 2 presents the results of the ablation study, revealing that the removal of any individual component results in a performance decline. These findings highlight the effectiveness of our proposed approach, which involves using attention as a compression metric and introducing semantic units.

The results of variant (6) further demonstrate that AttnComp proves effective when combined with different small LMs, and pairing it with more advanced models could lead to additional performance improvements.

# Does Cross-Attention Serve as an Effective Compression Metric?

To further assess the effectiveness of our proposed crossattention as a compression metric, we systematically compare the performance of various compression metrics across different filtering rates. Figure 2 shows the performance curves at these filtering rates. Our findings indicate that the cross-attention-based compression method not only enhances performance at lower filtering ratios but also significantly outperforms PPL and iterative compression methods at higher compression rates, thereby validating crossattention as an effective compression metric. Notably, as the filtering ratio increases, our approach maintains over $90 \%$ of the original performance, even when the majority of the context is filtered out, whereas PPL and ITPC suffer from significant performance degradation. This underscores the superior efficacy of our method, particularly in fine-grained compression scenarios.

# Does AttnComp Identify Good Semantic Units?

To evaluate whether our algorithm effectively identifies meaningful semantic units, we examine how well the optimization problem defined in Equation 6 is solved. Table 4 shows the objective function values, where ’Intra Attention’ refers to attention scores within a semantic unit, and ’Inter Attention’ denotes the scores between different semantic units. For comparison, we employ a random partitioning method that produces the same number of semantic units as our approach. Our results indicate that the intra-unit attention scores identified by our algorithm are eight times

Table 4: Analysis of objective function values.   

<html><body><table><tr><td></td><td>Intra Attention</td><td>Inter Attention</td></tr><tr><td>RandomUnits</td><td>359</td><td>11543</td></tr><tr><td>Semantic Units</td><td>2881</td><td>9049</td></tr></table></body></html>

The first Nobel Prize in Physics was awarded in 1[1]901 to Wilhelm Conrad Röntgen, of Germany, [2]who received 150,782 SEK, which is equal to 7,731,004 SEK in December 2007.[3]John Bardeen is the only laureate to win the prize twice—in 1956 and 1972. Maria Skłodowska[4]-Curie also won two Nobel Prizes, for physics in 1903 and chemistry in 1911.[5]William Lawrence Bragg was, until October 2014, the youngest ever Nobel laureate; he won the prize in 1915 at the age of 25…

Figure 3: A Case of semantic units. Each color represents a semantic unit.   
Table 5: Efficiency analysis results.   

<html><body><table><tr><td></td><td>Raw</td><td>LongLLMLingua (4.6x)</td><td>Ours (4.5x/9.4x)</td></tr><tr><td>Latency</td><td>6.31</td><td>1.72</td><td>1.59 /1.32</td></tr><tr><td>Acc</td><td>55.3</td><td>67.1</td><td>68.6/64.3</td></tr></table></body></html>

higher than those generated by the random method, demonstrating that our approach successfully addresses the optimization problem.

Additionally, we manually examine the semantic units identified by our method. Figure 3 illustrates an example, demonstrating that the semantic units produced by our algorithm typically represent complete semantic structures, such as full phrases, clauses, or sentences.

# Latency Evaluation

Table 5 shows the overall latency results (in seconds per example), including compression and response generation latencies. Unlike LongLLMLingua, our method uses a single LLM call without iterative compression and reduces compression latency. At the same performance level, our method demonstrates higher overall inference efficiency. With the higher compression rate, it retains sufficient information for comparable performance while minimizing latency. However, it is important to note that when the prompt is particularly long, the compression efficiency of our method tends to decrease, especially during the semantic unit identification stage. There is still room for further optimization in our algorithm’s efficiency.

# Related Work

# Long Context for LLM

Long context modeling is a fundamental challenge for large language models based on Transformer. Many efforts focus on improving the model itself, such as improving the attention mechanism (Sun et al. 2023), expanding the window length of LLMs through continue pre-training (Xiong et al. 2024), or improving positional encoding (Chen et al. 2023a). Unlike these approaches, which aim to extend the window length of the LLM itself, our work takes a different direction by compressing the long context or prompt.

# Prompt Compression

Prompt compression techniques had already been explored in the era of BERT-scale (Devlin 2018) language models (Goyal et al. 2020; Kim and Cho 2021; Modarressi, Mohebbi, and Pilehvar 2022). With the widespread success of large generative language models (Raffel et al. 2020; Brown et al. 2020) across various tasks (Zhao et al. 2024), prompt compression has garnered significant attention and can broadly be categorized into two main approaches: blackbox compression and white-box compression. White-box compression focuses on compressing the context into summary vectors (Mu, Li, and Goodman 2024; Chevalier et al. 2023; Ge et al. 2024). However, this line of research requires the target LLM to be a model with accessible parameters and is highly task-specific. On the other hand, black-box compression (Li et al. 2023; Jiang et al. 2023a; Pan et al. 2024) typically relies on information entropy theory, using a small language model to evaluate the significance of each token within the original prompt and subsequently removing those deemed less important.

# Attention in LLMs

As the core mechanism of Transformer, the attention mechanism (Vaswani et al. 2017) has been extensively studied. The prevailing view is that while the FFN layer (Dai et al. 2022) stores knowledge, attention is where the algorithm is implemented. Some work has analyzed the role of attention heads in LLMs, identifying certain retrieval heads (Wu et al. 2024) and revealing the intrinsic mechanisms by which LLMs utilize context information. Prior to LLMs, research also examined the role of attention in natural language generation (Lu et al. 2022) and phrase tagging (Gu et al. 2021). In the field of retrieval-augmented generation, many studies have built upon earlier research in ODQA that utilized attention for retrieval (Lee et al. 2022) or token elimination (Berchansky et al. 2023). Recent studies have focused on using state-ofthe-art large models, such as GPT-4, to generate complete semantic units, namely propositions (Chen et al. 2023b). In contrast, our research leverages the inherent properties of attention mechanisms to achieve more efficient segmentation of these semantic units.

# Conclusion

In this paper, we introduce AttnComp, a method that leverages the built-in attention mechanisms of LLMs for prompt compression. This approach employs cross-attention to derive a more effective compression metric and incorporates graph-based algorithms to identify semantic units, addressing the independence assumption inherent in prompt compression. We validated the effectiveness of this method on RAG and other common long-text tasks. Even at high filtering rates, AttnComp retains most of its performance while significantly reducing costs and inference latency. Furthermore, this method paves the way for new avenues in utilizing the contextual capabilities of LLMs, offering valuable insights for the deeper understanding and application of LLMs.