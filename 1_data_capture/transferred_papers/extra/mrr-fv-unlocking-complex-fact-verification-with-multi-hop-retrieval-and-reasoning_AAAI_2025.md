# MRR-FV: Unlocking Complex Fact Verification with Multi-Hop Retrieval and Reasoning

Liwen Zheng1, Chaozhuo $\mathbf { L i } ^ { 1 * }$ , Litian Zhang2, Haoran Jia1, Senzhang Wang3, Zheng Liu4, Xi Zhang

1Key Laboratory of Trustworthy Distributed Computing and Service (MoE), Beijing University of Posts and Telecommunications, China 2Beijing University of Aeronautics and Astronautics, Beijing 100191, China 3Central South University, China 4BAAI, China {zhenglw, lichaozhuo} $@$ bupt.edu.cn, litianzhang $@$ buaa.edu.cn, jiahaoran $@$ bupt.edu.cn, szwang@csu.edu.cn, zhengliu $1 0 2 6 @$ gmail.com, zhangx $@$ bupt.edu.cn

# Abstract

The pervasive spread of misinformation on social networks highlights the critical necessity for effective fact verification systems. Traditional approaches primarily focus on pairwise correlations between claims and evidence, often neglecting comprehensive multi-hop retrieval and reasoning, which results in suboptimal performance when dealing with complex claims. In this paper, we propose MRR-FV, a generative retrieval-enhanced model designed to address the novel challenge of Multi-hop Retrieval and Reasoning for Fact Verification, which integrates two core modules: Generative Multi-hop Retriever and the Hierarchical Interaction Reasoner. MRR-FV utilizes an autoregressive model for iterative multi-hop evidence retrieval, complemented by a pre-trained compressor to address the challenge of intention shift across retrieval hops. For claim verification, we propose a hierarchical interaction reasoner that conducts intra-sentence reasoning to capture long-term semantic dependencies and intersentence reasoning across multi-hop evidence subgraphs to reveal complex evidence interactions. Experimental evaluations on the FEVER and HOVER datasets demonstrate the superior performance of our model in both claim verification and evidence retrieval tasks.

Claim: Mickey Rourke appeared in a superhero film based on a Marvel Comics character. Themostnotableandsuccessful superhero films of the present day are filmsset in the Marvel Comics characters, such as the Iron Man One-hop series. Retriever In the superhero film, the villain's × plotthreatens the entireuniverse,and only a team of Marvel Comics characters can stop him. The most notable and successful Multi-hop Retriever 1-hop superhero films of the present day are filmsset in the Marvel Comics characters, such as theIron Man) series. Since then,MickeyRourke has appeared in several commercially Multi-hop successfulfilms including the 2010 Retriever 2-hop filmsIron ManI and The Expendables and the 2011 film Immortals.

# Introduction

Fact verification (FV) aims to leverage credible evidence to autonomously assess the veracity of textual statements, which helps combat the proliferation of misinformation and enhance the reliability and credibility of social media (Guo, Schlichtkrull, and Vlachos 2022; Zhang et al. 2025). Existing FV models usually follow a two-phase paradigm consisting of evidence retrieval and claim verification (Hu et al. 2023). Evidence retrieval focuses on precisely identifying critical evidential sentences within a vast corpus (Chen et al. 2022a). In the claim verification phase, the semantic interactions between the claim and the retrieved evidence are organized into sequences or graphs, which are then jointly analyzed to assess their authenticity (Zeng, Abumansour, and Zubiaga 2021).

Evidence retrieval serves as the cornerstone of FV systems, offering a robust and essential foundation for the subsequent claim verification (Samarinas, Hsu, and Lee 2021). Most retrieval modules adhere to a one-hop retrieval paradigm, where the semantic similarity between each candidate evidence and the claim is assessed independently. This approach focuses solely on the semantic relationship within individual claim-evidence pairs, overlooking the potential interactions between the claim and a wider set of evidence (Li et al. 2021). Consequently, this limitation may result in sub-optimal performance when addressing complex claims that necessitate support from multiple pieces of evidence. As depicted in Figure 1, the one-hop retriever identifies two evidence that are semantically aligned with the claim. However, both pieces of evidence fail to capture the critical information concerning “Mickey Rourke”, rendering the combination of these independently retrieved evidence insufficient for verifying the claim.

Moving beyond the traditional one-hop retrieval strategy, we emphasize fact verification enhanced by multi-hop evidence retrieval, where evidence is selected not only based on the claim but also by leveraging information accumulated from previous hops (Liao et al. 2023; Chen et al. 2024). As illustrated in Figure1, the two-hop evidence is selected by considering both the input claim and the first-hop evidence. While the two-hop evidence may be less directly aligned with the claim, it incorporates crucial information (e.g., “Iron Man”) from the one-hop evidence, thereby providing complementary details that contribute to a more comprehensive set of evidence to support the claim.

Few efforts have explored integrating multi-hop evidence retrieval into FV systems (Subramanian and Lee 2020). The common approach involves iterative dense retrieval, where evidence from previous hops is concatenated with the current query for subsequent retrievals (Zhu et al. 2023; Zhang et al. 2023). However, as more evidence is added, irrelevant information may accumulate, causing the final query to drift semantically from the claim, which may undermine verification accuracy (Zhang et al. 2024b). To address the limitations, our motivation lies in designing generative paradigm for multi-hop evidence retrieval. Unlike traditional discriminative methods, generative approaches can dynamically generate next-hop evidence in an autoregressive manner, introducing constraints to reduce intention shift instead of merely concatenating prior evidence with the original claim.

The generative multi-hop retrieval mechanism effectively aggregates comprehensive evidence across multiple hops. Nevertheless, the intricate interrelationships among the retrieved evidence introduce substantial complexity to the ensuing multi-hop reasoning process within the claim verification phase (Zhang et al. 2024c,a). Existing approaches to multi-hop reasoning predominantly employ either sequencebased or graph-based architectures (Guo, Schlichtkrull, and Vlachos 2022). Sequence-based methods are particularly effective in capturing local contextual information (Zhao et al. 2021), while graph-based methods excel in handling multiple evidence nodes and their complex interconnections, facilitating the synthesis of global information (Lan et al. 2024; Li et al. 2017). Considering the complementary roles of local and global information in multi-hop reasoning (Zhang, Zhang, and Pan 2022), it is imperative to develop a unified reasoning module capable of capturing both hierarchical semantics and multi-hop structural features.

In this paper, we propose a novel MRR-FV model to alleviate the problem of Multi-hop Retrieval and Reasoning for Fact Verification. In the evidence retrieval phase, the autoregressive generative model BART (Lewis et al. 2020) is employed as the backbone, supervised and optimized through joint training with the claim verification module. To address intention shift in multi-hop retrieval, we propose a novel query compressor that condenses the claim and historical evidence into a query. To enhance the compressor’s generation capability, we introduce a novel pre-training strategy that integrates the powerful generative capacity of large language models (LLMs) into smaller language models (SLMs). The compressed query is then input into the constrained generative retriever to generate evidence from the candidate pool. For claim verification, we propose a hierarchical interaction reasoner to simultaneously encode intra-sentence reasoning and inter-sentence correlations, integrating long-term semantic dependencies and interactions among multi-hop evidence within a unified reasoning framework. Experimental results on two datasets demonstrate the superiority of our approach. The main contributions of this work include:

• We investigate the novel problems of intention shift in multi-hop retrieval and explicit multi-hop reasoning in verification by modeling deep interactive features between evidence throughout the retrieval and verification. • We further propose MRR-FV, a generative retrievalenhanced model with two key modules: the Generative Multi-hop Retriever for constructing a complete and coherent evidence set, and the Hierarchical Interaction Reasoner, which sequentially extracts critical clues from intra and inter-sentence to facilitate claim verification. • We conduct extensive experiments on both the FEVER and HOVER datasets, demonstrating superior performance and underscoring the effectiveness of MRR-FV.

# Related Work

Fact verification involves assessing the veracity of a given claim by validating it against reliable evidence (Guo, Schlichtkrull, and Vlachos 2022). At the evidence retrieval stage, one-hop methods account for the relationship between claim and evidence (Fajcik, Motlicek, and $\mathrm { S m r z } \ 2 0 2 3 \mathrm { a } )$ . However, their lack of interactive capabilities during retrieval limits the effectiveness (Liao et al. 2023). Multi-hop retrieval mechanisms address this limitation by enabling iterative retrieval and integration of multiple pieces of evidence, thereby enhancing the system’s capacity to manage complex claims (Xiao et al. 2024). However, current multihop methods remain constrained by their heavy reliance on hyperlinks (Subramanian and Lee 2020) and are prone to intention shift, limiting their overall effectiveness (Khattab, Potts, and Zaharia 2021).

During the stage of claim verification, existing approaches predominantly emphasize claim verification through the exploration of fine-grained reasoning methods based on sequential (Wu, Wang, and Zhao 2024) or graph (Xu et al. 2022; Zhang, Zhang, and Zhou 2024) structures to integrate existing evidence. However, the absence of a unified reasoning framework capable of concurrently modeling both intrasentence contextual features and inter-sentence structural relationships presents significant challenges in addressing the complex interactions inherent in multi-hop evidence. Recent advancements utilize the reasoning capabilities of large language models, achieving notable performance improvements (Zhang and Gao 2023; Yue et al. 2024; Liu et al.

2024). Nevertheless, these approaches often demand substantial resource integration and tend to neglect fine-grained evidence retrieval, limiting their overall effectiveness.

# Problem Definition

Fact verification evaluates the truthfulness of a claim $c$ based on an evidence set $E = \{ e _ { 1 } , e _ { 2 } , . . . , e _ { n } \}$ sourced from a large corpus of documents $D$ to either support or refute the claim, in which $e _ { i }$ denotes a single evidence. The annotation $y$ can be categorized as “Supports” or “Refutes” and occasionally as “Not Enough Information”. This process requires not only a precise understanding of the relationship between the claim and the evidence but also efficient retrieval of relevant information from a vast evidence collection.

# Methodology

As illustrated in Figure 2, the proposed MRR-FV model consists of two modules: the Generative Multi-hop Retriever, which adeptly retrieves comprehensive evidence by modeling semantic correlations across evidence, and the Hierarchical Interaction Reasoner, which captures long-term dependencies within individual sentences as well as multihop interactions across multiple sentences.

# Generative Multi-hop Retriever

Given the claim $c$ , the generative multi-hop retriever aims to iteratively select a set of evidence $E _ { 1 : n } = \{ e _ { 1 } , e _ { 2 } , . . . , e _ { n } \}$ using generative models. Specifically, at each hop $t$ , the query compressor encodes the sequence formed by concatenating the claim $c$ with the previously retrieved evidence $E _ { 1 : t - 1 }$ , generating a compressed query $Q _ { t }$ that maintains semantic consistency with the original claim. The constrained generative retriever then utilizes a seq2seq model to retrieve the next-hop evidence $\boldsymbol { e } _ { t }$ from the corpus based on $Q _ { t }$ . This iterative process entails leveraging historical evidence to refine subsequent retrievals at each step, ultimately synthesizing a comprehensive chain of evidence to facilitate robust claim verification.

Query Compressor During retrieval, directly concatenating the claim with multi-hop evidence can make input queries excessively long, diminishing the claim’s semantic integrity and impairing evidence retrieval accuracy. To address this, we introduce a query compressor that condenses the extended text while retaining the core semantics and relationships between the claim and evidence.

At the $t$ -th hop, as illustrated in Figure 2(a), the query compressor utilizes the original claim $c$ and evidence retrieved in the previous $t - 1$ hops, denoted as $E _ { 1 : t - 1 } ~ =$ $\{ e _ { 1 } , e _ { 2 } , \ldots , e _ { t - 1 } \}$ , as input, and generates the compressed query $Q _ { t }$ for the $t$ -th evidence retrieval. Specifically, given the input query $I _ { t } = \{ c ; e _ { 1 } ; e _ { 2 } ; . . . ; e _ { t - 1 } \}$ , where $[ ; ]$ denotes the concatenation operation, the encoder first processes $I _ { t }$ into a contextual representation $h _ { t } = \operatorname { E n c o d e r } _ { C } ( I _ { t } )$ . At each decoding step $j$ , the decoder generates the $j$ -th token $q _ { t , j }$ of $Q _ { t }$ based on $h _ { t }$ and the previously generated tokens $q _ { t , < j }$ :

$$
\mathbb { P } ( q _ { t , j } \mid q _ { t , < j } , h _ { t } ) = { \mathrm { D e c o d e r } } _ { C } ( q _ { t , < j } , h _ { t } ) , \quad j \leq L ,
$$

where $\mathrm { ~ L ~ }$ is the pre-defined maximum compression length.

Enhanced Pre-training of the Query Compressor To mitigate intention shift and construct a more compact framework, we leverage the powerful inductive capabilities of LLMs to pre-train the query compressor, as depicted in Figure 2(b). (1) Distillation to endow the query compressor with compression ability. Given the claim $c$ and the evidence list $E$ , we ask the LLM to generate the compressed text $Q ^ { \prime }$ , which serves as the learning target for the query compressor given $c$ and $E$ . (2) Alignment to endow the query compressor with intention consistency. Given $c$ and $E$ , we use an LLM to assess the matching degree between $N$ compressed queries and $c$ , designating the highest-scoring query as the positive sample $Q ^ { + }$ and the lowest-scoring queries as negatives $\{ Q _ { 1 } ^ { - } , Q _ { 2 } ^ { - } , \dots , Q _ { n } ^ { - } \}$ . Subsequently, we apply an InfoNCE loss $L _ { a }$ to align the compressed queries with $Q ^ { + }$ :

$$
L _ { a } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \left( \frac { \exp \left( \frac { f ( Q _ { i } ) \cdot f ( Q ^ { + } ) } { \tau } \right) } { \sum _ { j = 1 } ^ { n } \exp \left( \frac { f ( Q _ { i } ) \cdot f ( Q _ { j } ) } { \tau } \right) } \right)
$$

where $f ( \cdot )$ is the encoder and $\tau$ is the temperature coefficient. The pre-training process ensures that compressed queries align strongly with the original claim, optimizing compression to mitigate intention shift and enhance fact verification.

Constrained Generative Retriever During the retrieval process, the interaction between evidence is enhanced by employing an autoregressive generation approach. At each step, the generative retriever takes the original query and the historical evidence as input, using a seq2seq model to generate the subsequent evidence.

Initially, the query compressor compresses the input sequence $I _ { t }$ into the compressed query $Q _ { t }$ . During the generation process, the compressed query $Q _ { t }$ is encoded into the contextual representation $h _ { t } ^ { q } = \operatorname { E n c o d e r } _ { G } ( Q _ { t } )$ . Subsequently, the $t$ -th evidence is progressively generated. At each decoding time step $j$ , Decoder $G$ produces the current token $\boldsymbol { e } _ { t , j }$ based on the previously generated tokens $\boldsymbol { e } _ { t , < j }$ and $h _ { t } ^ { q }$ :

$$
\mathbb { P } ( e _ { t , j } \mid e _ { t , < j } , h _ { t } ^ { q } ) = \operatorname { D e c o d e r } _ { G } ( e _ { t , < j } , h _ { t } ^ { q } ) .
$$

The overall generation process of the $t$ -th evidence $e _ { t }$ can be expressed as:

$$
\mathbb { P } ( e _ { t } | h _ { t } ^ { q } ) = \prod _ { j = 1 } ^ { L } P ( e _ { t , j } | e _ { t , < j } , h _ { t } ^ { q } ) ,
$$

where $L$ represents the length of $\boldsymbol { e } _ { t }$ , and $\mathbb { P } ( e _ { t } | h _ { t } ^ { q } )$ denotes the probability distribution of generating $e _ { t }$ given the input contextual representation $h _ { t } ^ { q }$ .

Moreover, our goal is to retrieve crucial evidence from the candidate corpus to support claim verification. However, allowing the generation model to select any token from the entire vocabulary at each decoding step may generate evidence that doesn’t match valid candidates (Cao et al. 2021). To prevent this issue, we employ a constrained generation strategy restricting the model to decode only valid tokens.

The prefix tree, or trie, is a tree-like data structure used to store strings, with each node representing a character’s token id. The path from the node to any node represents a string or its prefix. We store the candidate evidence in a prefix tree to restrict allowed tokens at each decoding step. The prefix tree-based constraint significantly reduces the number of candidate tokens, thereby enhancing generation efficiency.

![](images/96762bb9cc9d2cbc54578d436bf9a5d9c5e52897ab9d9bbc4398e99e3e5a9745.jpg)  
Figure 2: Framework of the proposed MRR-FV model.

# Hierarchical Interaction Reasoner

To model the clue chain in the multi-hop evidence for claim verification, we conduct reasoning on both sequence and graph structures within a unified framework. This process is divided into two main components: Intra-sentence Reasoning and Inter-sentence Reasoning.

Intra-sentence Reasoning We conduct intra-sentence reasoning on the sequence of each claim-evidence pair to initialize the contextual features. As the number of retrieving hops increases, the distance between the retrieved evidence and the original claim expands. Traditional transformers often blur global information when processing long sequences due to dispersed context. To address this issue, we propose utilizing the Recurrent Memory Transformer (Bulatov, Kuratov, and Burtsev 2022), which incorporates memory tokens to facilitate the transmission and retention of global information across multiple segments, thereby enhancing the capture of global context in long sequences.

As depicted in Figure 2(c), suppose $X$ is the sequence of a claim-evidence pair, which is split into $S$ segments, i.e., $X = [ X _ { 1 } , X _ { 2 } , \ldots , X _ { S } ]$ . For effective long-term dependency modeling, the $\tau$ -th segment $X _ { \tau }$ is augmented with special memory tokens $H _ { \tau } ^ { \mathrm { m e m } }$ , and the augmented sequence is then processed with a standard Transformer. Specifically, the memory tokens are appended to both the beginning and end of the segment tokens’ representations $H _ { \tau } ^ { 0 }$ :

$$
\begin{array} { r l } & { \tilde { H } _ { \tau } ^ { 0 } = [ H _ { \tau } ^ { \mathrm { m e m } } \circ H _ { \tau } ^ { 0 } \circ H _ { \tau } ^ { \mathrm { m e m } } ] , } \\ & { \quad \quad \bar { H } _ { \tau } ^ { N } = \mathrm { T r a n s f o r m e r } ( \tilde { H } _ { \tau } ^ { 0 } ) , } \\ & { \quad \quad [ H ^ { \mathrm { r e a d } } \circ H _ { \tau } ^ { N } \circ H _ { \tau } ^ { \mathrm { w r i t e } } ] : = \bar { H } _ { \tau } ^ { N } , } \end{array}
$$

where $N$ represents the number of Transformer layers. Additionally, to enable the sequence representation to attend to the memory states produced in this segment, the starting group of tokens in $\tilde { H } _ { \tau } ^ { \bar { N } }$ functions as the read memory $H _ { \tau } ^ { \mathrm { r e a d } }$ . To attend to all current segment tokens and update representation stored in the memory, the ending group of tokens in $\tilde { H } _ { \tau } ^ { N }$ works as the write memory $H _ { \tau } ^ { \mathrm { w r i t e } }$ , containing updated memory tokens for the $\tau$ -th segment.

The $( \tau + 1 )$ -th segment of the input sequence is processed in order. To establish a recurrent connection between segments, the memory tokens output from the current segment are transferred to the input of the subsequent segment:

$$
\begin{array} { r } { \begin{array} { r l } & { H _ { \tau + 1 } ^ { \mathrm { m e m } } : = H _ { \tau } ^ { \mathrm { w r i t e } } , } \\ & { \tilde { H } _ { \tau + 1 } ^ { 0 } = \left[ H _ { \tau + 1 } ^ { \mathrm { m e m } } \circ H _ { \tau + 1 } ^ { 0 } \circ H _ { \tau + 1 } ^ { \mathrm { m e m } } \right] . } \end{array} } \end{array}
$$

Finally, the sentence embedding $H$ can be obtained by combining the read memory outputs of each segment. This recurrent memory process efficiently models long-term dependencies and ensures precise information transmission.

Inter-sentence Reasoning As illustrated in Figure 2(d), to capture the semantic connections between the claim and evidence, we construct the evidence graph based on the retrieval path, denoting the $t _ { \mathrm { t h } }$ hop evidence graph as $G ^ { t }$ . To mitigate the limitations of sparse graph structures on node feature learning, we leverage latent edge learning to infer potential connections, enhancing the representation of claim-evidence pairs (Zheng et al. 2022; Li et al. 2018). Following the network’s homogeneity principle, where similar nodes are more likely to establish mutual connections, we infer latent edges by computing feature similarity between nodes.

Supposing $H ^ { t } = [ h _ { 1 } ^ { t } , \dots , h _ { t - 1 } ^ { \dot { t } } , h _ { t } ^ { t } ]$ is the initial representation of $G ^ { t }$ , where $h _ { i } ^ { t }$ denotes the $i _ { \mathrm { t h } }$ nodes in $G ^ { t }$ . To integrate semantic and structural features in latent edge learning, we first employ graph convolutional networks to derive the structural feature $h _ { i } ^ { t - \mathrm { G } }$ . This is then combined with the initial semantic representation $h _ { i } ^ { t }$ to calculate the similarity between nodes, which is subsequently used to construct the latent edge matrix:

$$
h _ { i } ^ { t - \mathrm { s i m } } = \frac { h _ { i } ^ { t } + h _ { i } ^ { t - \mathrm { G } } } { 2 } .
$$

Subsequently, we compute the cosine similarity $\beta _ { i j } ^ { t }$ between each pair of nodes and construct latent edges $e _ { i j } ^ { t }$ using a predetermined similarity threshold $\gamma$ :

$$
\beta _ { i j } ^ { t } = \frac { h _ { i } ^ { t - \mathrm { s i m } } \cdot h _ { j } ^ { t - \mathrm { s i m } } } { \| h _ { i } ^ { t - \mathrm { s i m } } \| \| h _ { j } ^ { t - \mathrm { s i m } } \| } ,
$$

$$
e _ { i j } ^ { t } = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } \beta _ { i j } ^ { t } > \gamma } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. } .
$$

To account for the dynamic nature of node relationships under different connections, we employ a hop-specific trainable weight matrix $\boldsymbol { W } _ { R } ^ { t }$ , enabling the calculation of the edge weight matrix $E ^ { t }$ for the $t$ -hop evidence graph, where $e _ { i j } ^ { t }$ represents the relationship value between node $i$ and node $j$ :

$$
e _ { i j } ^ { t } = \mathrm { L e a k y R e L U } \left( a ^ { \top } [ W _ { R } ^ { t } h _ { i } ^ { t } \ | | \ W _ { R } ^ { t } h _ { j } ^ { t } ] \right) ,
$$

Furthermore, shorter graph distances indicate stronger correlations between nodes. Thus, we introduce graph-relative positions as an attention bias during aggregation. This bias is incorporated into the attention mechanism based on the softmax function, thereby enhancing attention between neighboring evidence nodes. The output representation of $G ^ { t }$ , denoted as $\tilde { H } ^ { t } = S ^ { t } H ^ { t }$ , where $s _ { i j } ^ { t }$ in $S ^ { t }$ is defined as:

$$
s _ { i j } ^ { t } = \frac { \exp ( e _ { i j } ^ { t } - m _ { G } | d _ { i j } | ) } { \sum _ { k = 1 } ^ { n } \exp ( e _ { i k } ^ { t } - m _ { G } | d _ { i k } | ) } ,
$$

where $d _ { i j }$ denotes the graph-relative distance between evidence nodes $\mathbf { \chi } _ { i }$ and $j$ , and $m _ { G }$ represents the fixed slope for a specific head. To stabilize the learning process, the aforementioned mechanism can be extended to a multi-head format with $h$ heads. After concatenating the outputs of each head, the final graph representation $\tilde { H } ^ { t }$ can be obtained through a normalization layer:

$$
\tilde { H } ^ { t } = \mathrm { N o r m } ( \mathrm { c o n c a t } ( \tilde { H } _ { ( 1 ) } ^ { t } , \tilde { H } _ { ( 2 ) } ^ { t } , \dots , \tilde { H } _ { ( h ) } ^ { t } ) ) .
$$

Subsequently, we derive the final fusion representation $F$ by concatenating the multi-hop representations:

$$
F ^ { t } = \mathrm { c o n c a t } ( \tilde { H } ^ { 1 } , \tilde { H } ^ { 2 } , \cdot \cdot \cdot , \tilde { H } ^ { t } ) .
$$

Ultimately, we feed the resulting representation $F ^ { t }$ into MLP to obtain the predicted label.

# Joint Training Mechanism

To oversee the intermediate stages of multi-hop retrieval, we verify the claim following each retrieval step to assess the current evidence chain. We compute the verification loss $L _ { v }$ using the verification outcomes, and proceed with joint training of the retriever and reasoner through gradient back propagation. Furthermore, we calculate the retrieval loss $L _ { r }$ by assessing whether the retrieved evidence is contained within the gold evidence set, offering additional supervision to enhance the effectiveness of the retrieval process.

# Experiments

# Experimental Setup

This section describes the dataset, evaluation metrics, and baselines of our experiments.

Datasets We conduct our evaluations using the large-scale dataset FEVER (Thorne et al. 2018) and HOVER (Jiang et al. 2020), a multi-hop fact-verification dataset. FEVER comprises 185,455 annotated claims alongside 5,416,537 Wikipedia documents. HOVER includes claims that necessitate integration and reasoning across multiple Wikipedia articles, with claims verifiable in 2-4 hops. Besides, we use the dev set of HOVER for evaluation since the test sets are not publicly released. All claims in both FEVER and HOVER are classified by annotators as Supports, Refutes, or Not Enough Info.

Evaluation Metrics Following previous studies, we employ Label Accuracy (LA) and the FEVER score as evaluation metrics for claim verification on the FEVER dataset (Hanselowski et al. 2018; Liu et al. 2020). LA is a widely applicable metric, and the FEVER score considers label accuracy contingent upon the provision of at least one complete set of golden evidence (Thorne et al. 2018). Additionally, for evidence retrieval, given the gold evidence, we utilize Precision, Recall, and F1-score to assess retrieval performance. For HOVER, we use Macro-F1 scores for claim verification, and F1-score for evidence retrieval.

Baselines As illustrated in Table 1, we compare our proposed method with several advanced baselines to validate effectiveness and superiority. The other baselines treats the two stages independently. BERT Concat (Liu et al. 2020), GAT (Liu et al. 2020), and GEAR (Zhou et al. 2019) modify the ESIM (Chen et al. 2017) model to compute the relevance score between the evidence and the claim for evidence retrieval. KGAT (Liu et al. 2020), DREAM (Zhong et al. 2020), TARSA (Si et al. 2021), Proofver (Krishna, Riedel, and Vlachos 2022), and EvidenceNet (Chen et al. 2022b) utilize BERT-based models trained with a one-hop ranking loss. HESM (Subramanian and Lee 2020) employs a multihop evidence retrieval method that is hyperlink-dependent, and GERE is a generative evidence retrieval method. During the reasoning phase, BERT Concat employs a BERT-based model, while Proofver utilizes a seq2seq model to generate natural logic-based inferences as proofs. The remaining methods adopt graph-based approaches.

As depicted in Table 2, we design three types of baselines: BERT-FC (Soleimani, Monz, and Worring 2020) and LisT5 (Jiang, Pradeep, and Lin 2021) are pretrained models, RoBERTa-NLI (Nie et al. 2020), DeBERTaV3- NLI (He, Gao, and Chen 2023) and MULTIVERS (Wadden et al. 2022) are fact verification fine-tuned models, ProgramFC (Pan et al. 2023) and FOLK (Wang and Shu 2023) are LLM-enhanced models.

Table 1: Overall performance on FEVER. Bold indicates the best result, while underline denotes the second best.   

<html><body><table><tr><td rowspan="2">Models</td><td colspan="2">Dev</td><td colspan="2">Test</td></tr><tr><td>LA</td><td>FEVER</td><td>LA</td><td>FEVER</td></tr><tr><td>BERTConcat</td><td>73.67</td><td>68.89</td><td>71.01</td><td>65.64</td></tr><tr><td>GAT</td><td>76.13</td><td>71.04</td><td>72.03</td><td>67.56</td></tr><tr><td>GEAR</td><td>74.84</td><td>70.69</td><td>71.60</td><td>67.10</td></tr><tr><td>HESM</td><td>75.77</td><td>73.44</td><td>74.64</td><td>71.48</td></tr><tr><td>KGAT</td><td>78.29</td><td>76.11</td><td>74.07</td><td>70.38</td></tr><tr><td>DREAM</td><td>79.16</td><td></td><td>76.85</td><td>70.60</td></tr><tr><td>KGAT+GERE</td><td>79.44</td><td>77.38</td><td>75.24</td><td>71.17</td></tr><tr><td>TARSA</td><td>81.24</td><td>77.96</td><td>73.97</td><td>70.70</td></tr><tr><td>Proofver</td><td>80.74</td><td>79.07</td><td>79.47</td><td>76.82</td></tr><tr><td>EvidenceNet</td><td>81.46</td><td>78.29</td><td>76.95</td><td>73.78</td></tr><tr><td>TwoWingOS</td><td>-</td><td>-</td><td>75.99</td><td>54.33</td></tr><tr><td>KGAT+FER</td><td>79.02</td><td>76.59</td><td>73.34</td><td>69.61</td></tr><tr><td>CD</td><td>80.80</td><td>78.00</td><td>79.30</td><td>76.50</td></tr><tr><td>MRR-FV</td><td>82.98</td><td>79.83</td><td>80.83</td><td>78.25</td></tr></table></body></html>

Table 2: Overall performance on HOVER.   

<html><body><table><tr><td></td><td>Models</td><td>2-hop</td><td>3-hop</td><td>4-hop</td></tr><tr><td rowspan="2">I</td><td>BERT-FC</td><td>50.68</td><td>49.86</td><td>48.57</td></tr><tr><td>LisT5</td><td>52.56</td><td>51.89</td><td>50.46</td></tr><tr><td rowspan="3">II</td><td>RoBERTa-NLI</td><td>63.62</td><td>53.99</td><td>52.40</td></tr><tr><td>DeBERTaV3-NLI</td><td>68.72</td><td>60.76</td><td>56.00</td></tr><tr><td>MULTIVERS</td><td>60.17</td><td>52.55</td><td>51.86</td></tr><tr><td rowspan="2">ⅢI</td><td>FOLK</td><td>66.26</td><td>54.80</td><td>60.35</td></tr><tr><td>ProgramFC</td><td>70.30</td><td>63.43</td><td>57.74</td></tr><tr><td rowspan="2"></td><td>MRR-FV</td><td>71.66</td><td></td><td></td></tr><tr><td></td><td></td><td>65.09</td><td>62.72</td></tr></table></body></html>

# Overall Performance

Table 1 presents the performance results of our proposed model MRR-FV on the FEVER dataset, compared to the baselines for fact verification. Overall, MRR-FV demonstrates superior performance across all metrics, underscoring its effectiveness and robustness. It outperforms HESM, indicating that our multi-hop retrieval mechanism can efficiently gather evidence for claim verification without relying on hyperlinks. When compared to the generative retrieval method GERE, MRR-FV shows significant improvement, highlighting the benefits of multi-hop and joint training mechanisms in enhancing the overall performance of fact verification systems. Furthermore, although TwoWingOS, FER, and CD (Fajcik, Motlicek, and $\mathrm { S m r z } \ 2 0 2 3 \mathrm { b }$ ) utilize the claim verification results to jointly optimize the evidence retriever, they still rely on one-hop methods for evidence retrieval, which fail to capture the interrelationships between evidence pieces during the retrieval process. This limitation is likely the primary reason for their inferior performance compared to MRR-FV.

Table 3: Retrieval performance comparison on FEVER.   

<html><body><table><tr><td rowspan="2">Models</td><td colspan="3">Dev</td><td colspan="3">Test</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>TF-IDF</td><td></td><td>-</td><td>17.20</td><td>11.28</td><td>47.87</td><td>18.26</td></tr><tr><td>ESIM</td><td>24.08</td><td>86.72</td><td>37.69</td><td>23.51</td><td>84.66</td><td>36.80</td></tr><tr><td>BERT</td><td>27.29</td><td>94.37</td><td>42.34</td><td>25.21</td><td>87.47</td><td>39.14</td></tr><tr><td>XLNet</td><td>26.60</td><td>87.33</td><td>40.79</td><td>25.55</td><td>85.34</td><td>39.33</td></tr><tr><td>RoBERTa</td><td>26.67</td><td>87.64</td><td>40.90</td><td>25.63</td><td>85.57</td><td>39.45</td></tr><tr><td>GERE</td><td>58.43</td><td>79.61</td><td>67.40</td><td>54.30</td><td>77.16</td><td>63.74</td></tr><tr><td>DQN</td><td>54.75</td><td>79.92</td><td>64.98</td><td>52.24</td><td>77.93</td><td>62.55</td></tr><tr><td>MRR-FV</td><td>60.37</td><td>87.79</td><td>71.54</td><td>58.30</td><td>87.82</td><td>70.08</td></tr></table></body></html>

Table 4: Retrieval performance comparison on HOVER.   

<html><body><table><tr><td></td><td>Models</td><td>2-hop</td><td>3-hop</td><td>4-hop</td></tr><tr><td rowspan="3">One-hop</td><td>TF-IDF+BERT</td><td>57.2</td><td>49.8</td><td>45.0</td></tr><tr><td>Oracle+BERT</td><td>68.3</td><td>71.5</td><td>76.4</td></tr><tr><td>CD</td><td>81.3</td><td>80.1</td><td>78.1</td></tr><tr><td rowspan="2">Multi-hop</td><td>Baleen</td><td>81.2</td><td>82.5</td><td>80.0</td></tr><tr><td>GMR</td><td>81.9</td><td>82.2</td><td>80.2</td></tr><tr><td></td><td>MRR-FV</td><td>82.4</td><td>83.6</td><td>81.2</td></tr></table></body></html>

As depicted in Table 2, following ProgramFC, we divide the HOVER validation set into three subsets based on the number of hops required to verify the claim. On the dev set of HOVER, MRR-FV outperforms the SOTA baseline by $1 . 9 \%$ , $2 . 6 \%$ , and $3 . 9 \%$ on the two-hop, three-hop, and fourhop subsets, respectively. This demonstrates that MRR-FV is more effective for verifying deeper claims.

# Performance on Evidence Retrieval

Following previous works (Chen et al. 2022a), we select several representative one-hop evidence retrieval methods as baselines on the FEVER dataset, including TFIDF (Thorne et al. 2018), ESIM (Hanselowski et al. 2018), BERT (Liu et al. 2020), XLNet (Zhong et al. 2020), and RoBERTa (Zhong et al. 2020). Additionally, GERE (Chen et al. 2022a) is a generative evidence retrieval method, and DQN (Wan et al. 2021) employs a reinforcement learningbased approach to identify precise evidence. In Table 3, we see that MRR-FV’s evidence retrieving results mirror its superiority, with large performance gains across the board against the baseline model.

The baselines on the HOVER dataset are divided into two categories: one-hop and multi-hop baselines. As illustrated in Table 4, MRR-FV is capable of retrieving more precise evidence compared to the baselines. Additionally, traditional one-hop methods perform significantly worse than multihop methods (Lee et al. 2022), since most claims require synthesizing information from multiple evidence for accurate verification. CD outperforms the other two single-hop baselines as it employs a joint framework.

Table 5: Performance on intention shift mitigation. The evaluation metric is the cosine similarity score between the query at different hop and the original claim   

<html><body><table><tr><td>Models</td><td>1-hop</td><td>2-hop</td><td>3-hop</td></tr><tr><td>Traditional multi-hop</td><td>0.85</td><td>0.76</td><td>0.71</td></tr><tr><td>Compressor enhanced multi-hop</td><td>0.85</td><td>0.81</td><td>0.79</td></tr></table></body></html>

# Performance on Intention Shift Mitigation

We assess the effectiveness of our proposed method in mitigating the intention shift by evaluating the semantic similarity between query of each hop and the original claim. We benchmark our approach against the “Traditional Multihop” baseline, where the query is a concatenation of the original claim and previously retrieved evidence. As depicted in Table 5, as the number of hops increases, the matching degree between the queries of “Traditional Multihop” and the initial claim rapidly decreases. This observation aligns with our expectation that concatenating evidence leads to progressively longer queries, which inevitably introduces more noise. In contrast, we leverage a pre-trained query compressor to condense the concatenated sequence while maintaining its semantic integrity. This approach leads to a more gradual decline in the matching degree, indicating that our method can partially mitigate the intention shift.

# Ablation Study

As illustrated in Table 6, we design ablation studies to verify the effectiveness of core modules. Specifically, we independently remove or replace a module and observe the impact on the final verification performance.

-w/o Compressor signifies using the evidence retrieved from the previous hops to enhance the claim, without any compression. -w/o Pre-training eliminates the pre-training process of the query compressor. -w/o Multi-hop retrieves evidence in a single step, without considering the interactions between the evidence during the retrieval process. The evidence retriever of -w/o Generative Retriever is replaced by a traditional dense retrieval. -w/o Intra-sentence Reasoning eliminates the component that employs the recurrent memory transformer for reasoning within the query, thus neglecting the modeling of long-term dependencies within sentences. -w/o Inter-sentence Reasoning processes the claim and evidence directly in a sequential manner, thereby disregarding the modeling of interactions between sentences.

The experimental results show that removing or replacing any module leads to a decline in overall performance, indicating the effectiveness of these modules and their contributes to the enhancement of the overall performance.

# Hyperparameter Sensitivity Analysis

As depicted in Figure 3, Length and Hops dictate the maximum length of the compressed query and the pre-set number of hops for multi-hop retrieval, respectively. The experimental results in Figure 3(a) suggest that both excessively long and overly short compressed text can lead to a decline in performance. Overly long compressed text may not effectively mitigate the dilution of key information in the claim, while excessively short compressed text may fail to preserve the completeness of the evidence. As depicted in Figure 3(b), the evaluation metric on the HOVER dataset is the average Macro-F1 across the three subsets. Similarly, an excessively high or low number of hops can also lead to a decline in performance. Moreover, since the complexity of claims varies across different datasets, each dataset corresponds to a different optimal number of hops. In this regard, our experimental results align with this hypothesis.

Table 6: Performance of ablation study.   

<html><body><table><tr><td></td><td colspan="2">Dev</td><td colspan="2">Test</td></tr><tr><td>Models</td><td>LA</td><td>FEVER</td><td>LA</td><td>FEVER</td></tr><tr><td>-w/o Compressor</td><td>81.94</td><td>79.47</td><td>80.06</td><td>77.58</td></tr><tr><td>-w/o Pre-training</td><td>82.27</td><td>79.62</td><td>80.21</td><td>77.87</td></tr><tr><td>-w/o Multi-hop</td><td>81.72</td><td>79.32</td><td>79.69</td><td>77.02</td></tr><tr><td>-w/o Generative Retriever</td><td>82.13</td><td>79.68</td><td>80.27</td><td>77.82</td></tr><tr><td>-W/o Intra-sentence Reasoning</td><td>82.49</td><td>79.57</td><td>80.43</td><td>78.15</td></tr><tr><td>-W/o Inter-sentence Reasoning</td><td>81.86</td><td>79.36</td><td>79.83</td><td>77.37</td></tr><tr><td>MRR-FV</td><td>82.98</td><td>79.83</td><td>80.83</td><td>78.25</td></tr></table></body></html>

![](images/dafd6c22ece99e10e33ba0e3d58efd1c4211ab2e40a422479d4d7b826c6aeb24.jpg)  
Figure 3: Hyperparameter sensitivity analysis.

# Conclusion

This paper investigates the novel problem of Multi-hop Retrieval and Reasoning for Fact Verification(MRR-FV), and further propose a generative retrieval enhanced model. We leverage a the Generative Multi-hop Retriever to construct a comprehensive evidence chain by integrating inter-evidence interactions during retrieval, enhanced by a query compressor to mitigate intention shift throughout the multi-hop process. At the reasoning stage, the Hierarchical Interaction Reasoner integrates sequence-based and graph-based approaches, enabling thorough analysis of local inter-sentence correlations alongside global multi-hop features and interrelationships. Finally, experiments on the FEVER and HOVER datasets underscore the superior performance of this method, demonstrating its efficacy in tackling complex fact verification tasks through comprehensive multi-hop evidence retrieval and reasoning.