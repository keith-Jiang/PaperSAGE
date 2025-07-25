# Language Models of Code are Few-Shot Planners and Reasoners for Multi-Document Summarization with Attribution

Abhilash Nandy1\*, Sambaran Bandyopadhyay2

1Indian Institute of Technology Kharagpur, India 2Adobe Research nandyabhilash@gmail.com, samb.bandyo@gmail.com

# Abstract

Document summarization has greatly benefited from advances in large language models (LLMs). In real-world situations, summaries often need to be generated from multiple documents with diverse sources and authors, lacking a clear information flow. Naively concatenating these documents and generating a summary can lead to poorly structured narratives and redundancy. Additionally, attributing each part of the generated summary to a specific source is crucial for reliability. In this study, we address multi-document summarization with attribution using our proposed solution MiDAS$P R o$ , consisting of three stages: (i) Planning the hierarchical organization of source documents, (ii) Reasoning by generating relevant entities/topics, and (iii) Summary Generation. We treat the first two sub-problems as a code completion task for LLMs. By incorporating well-selected in-context learning examples through a graph attention network, LLMs effectively generate plans and reason topics for a document collection. Experiments on summarizing scientific articles from public datasets show that our approach outperforms state-ofthe-art baselines in both automated and human evaluations.

# 1 Introduction

The exploration of text-to-text generation within the NLP community has evolved through a variety of methodologies, experiencing considerable progress (Zhang et al. 2020; Song et al. 2018; Wiseman, Shieber, and Rush 2018; Zhang et al. 2020). In recent years, the exponential growth of digital information has led to an increased demand for automated methods to summarize this vast amounts of data (ElKassas et al. 2021). Emergence of Large Language Models (LLMs) enables effective and efficient summarization of huge amount of text data (Grail, Perez, and Gaussier 2021; Jin et al. 2024). In real-world situations, text may be derived from a variety of sources, each with its distinct characteristics. In these instances, a user might have a particular intent in mind, such as a specific topic, and desire to create a coherent summary that encompasses relevant information from those diverse sources in relation to the intended topic.

With ever-increasing context windows, LLMs could be leveraged for the task of multi-document summarization.

However, such works (Kurisinkel and Chen 2023; Xiao et al. 2022; Wang et al. 2023; Shi et al. 2023) are few in number, and suffer from drawbacks such as need for expensive finetuning and lack of a source document attribution (citation) module, which is essential to make the summary more reliable, transparent, and interpretable. Also, several methods augment annotated data/metadata as training inputs to the collection of documents to be summarized, such as graphs (Zhang et al. 2023; Pasunuru et al. 2021), document timestamps (Song et al. 2024; Chen et al. 2023b, 2019), which are difficult to obtain. Hence, there is a lack of an LLMbased multi-document attribution-inclusive summarization framework to generate intent-based summaries with negligible training cost and annotations.

In this paper, we propose MiDAS-PRo (Multi-Document Attribution-inclusive Summarization via Planning-cumReasoning) to mitigate aforementioned issues. MiDAS-PRo uses a Code-LLM to plan the layout of the attribution of source documents in the final summary (where attribution is the citations to the source documents) and to reason by generating important domain-specific entities/- topics corresponding to the citations. Finally, we utilize this planning-cum-reasoning information to enhance the attribution-inclusive summary generation capability of an LLM of interest. The planning-cum-reasoning and the summarization modules are all carried in an in-context learning setting, thus requiring no LLM training. The In-Context Examples having the most similar attribution/reference layout to the test example in their summaries are selected by training a 2-layer Graph Attention Network (GAT), which incurs minimal training cost and time. MiDAS-PRo Framework is depicted in Fig. 1. MiDAS-PRo provides a sizeable improvement in summarization metrics and a massive improvement in clustering-based metrics when applied across different LLMs and varying number of in-context examples.

We make the following contributions in this paper - (1) Code LLMs are Few-Shot Planners and Reasoners for Multi-Document Summarization. To that effect, we propose MiDAS-PRo for multi-document summarization, where planning-cum-reasoning is carried out in 2 stages - Hierarchical Reference Layout Tree (HRLT) Generation, followed by Sentence-Wise Entity Generation (2) We introduce several clustering-based metrics to evaluate the attribution of references in the generated summary (3) Strategically organizing the references within the summary in MiDAS-PRo clearly indicates which specific source documents contribute to different parts of the summary, thus enhancing transparency and interpretability (4) To make the in-content examples more relevant, we utilize a novel GATbased In-Context Example Selection Method for the proposed MiDAS-PRo (5) We also manually curate MiDAS, a subset of 96 examples from the BigSurvey (LIU et al. 2022) dataset with reference attribution for evaluating MiDASPRo.

# 2 Related Works

Multi-Document Summarization. PRIMERA (Xiao et al. 2022) introduces a novel pre-training objective specifically tailored for multi-document summarization, enabling the model to effectively connect and aggregate information across multiple documents. By leveraging efficient encoderdecoder transformers, PRIMERA simplifies the processing of concatenated input documents. Graph-based models (Liao, Lebanoff, and Liu 2018; Li et al. 2020; Pasunuru et al. 2021) have also been used for multi-document summarization. However, such models frequently depend on auxiliary information, such as AMR (Abstract Meaning Representation) or discourse structure, to construct an input graph. This reliance on additional models reduces their generalizability. Code-based reasoning and planning using LLMs. Using code for reasoning and planning has shown promising results. For instance, code-based reasoning using the ”Program of Thoughts” (PoT) approach (Chen et al. 2023a) enables LLMs to solve math problems by generating code to express complex reasoning procedures, which is then executed by a program interpreter, effectively separating reasoning from computation. Similarly, Gao et al. (2023) introduces Program-Aided Language models (PAL), which utilize LLMs to convert natural language problems into executable code, offloading the computation to a Python interpreter. This method enhances accuracy in several mathematical, symbolic, and algorithmic reasoning tasks. Madaan et al. (2022) demonstrates that framing structured commonsense reasoning tasks as code generation problems enables pre-trained code LLMs to outperform natural language models, even when the tasks don’t involve code, across three varied structured reasoning tasks.

# 3 Problem Statement and Datasets

Given a collection of $n$ documents, which are paper abstracts of interest - $( R _ { 1 } ^ { a } , R _ { 2 } ^ { a } , . . . , R _ { n } ^ { a } )$ , and a query intent $Q ^ { I }$ , the objective is to generate the target multi-document abstractive summary $S ^ { T }$ . Note that in our setting, $S ^ { T }$ contains citations to $R _ { 1 } ^ { a } , R _ { 2 } ^ { a } , . . . , R _ { n } ^ { a }$ .

# 3.1 Datasets

Multi-XScience. The Multi-XScience dataset (Lu, Dong, and Charlin 2020) is a large-scale dataset created to facilitate the task of multi-document summarization, particularly in the context of scientific articles. This dataset combines information from two primary sources: arXiv.org and the Microsoft Academic Graph (MAG). The construction of Multi

XScience involves multiple stages to ensure its robustness and utility. Initially, the LaTeX source of approximately 1.3 million arXiv papers is cleaned. These papers are then aligned with their references in MAG through a series of heuristics, followed by five iterations of data cleaning interspersed with human verification. Here, $Q ^ { I }$ is the abstract of a query paper, and $S ^ { T }$ is the related work paragraph of the query paper. The dataset consists of 30, 369 training, 5, 066 validation, and 5, 093 test instances. On average, documents in the dataset have a length of 778.08 tokens, summaries are 116.44 tokens long, and each document has 4.42 references. Multi-XScience stands out for its high degree of abstractiveness, reflected in the significant proportion of novel n-grams in its target summaries.

MiDAS. MiDAS (Multi-Document Attribution-inclusive Summarization) is a subset of BigSurvey-MDS dataset (LIU et al. 2022), where we annotate additional reference attribution information and paper metadata that is not present in BigSurvey-MDS dataset. BigSurvey-MDS is a large-scale dataset designed for the task of summarizing numerous academic papers on a single research topic. For MiDAS, $Q ^ { I }$ is the title of a survey research paper (annotated by us), and $S ^ { T }$ is the introduction paragraph of the survey paper. BigSurvey-MDS is constructed using 4, 478 survey papers from arXiv.org and their associated reference papers’ abstracts. The survey papers are selected for their comprehensive coverage of various research topics. The entire dataset is divided into training, validation, and test sets in the ratio of 8:1:1. BigSurvey-MDS has an average input document length of 11, 893.1 words and 450.1 sentences. The target summaries average 1, 051.7 words and 38.8 sentences. MiDAS samples 40 examples from the training and 56 examples from the test sets of BigSurvey-MDS.

# 4 Proposed Approach

Our proposed approach MiDAS-PRo introduces a novel planning-cum-reasoning module using pre-trained Code LLMs in an in-context learning setting, which is carried out in 2 stages - (1) Hierarchical Reference Layout Tree (HRLT) Generation, where relative locations of references to source documents in the final summary are predicted to produce a hierarchical plan of the final summary. (2) Sentence-wise Entity Generation, where entities relevant to each sentence in the final summary are predicted to act as a reasoning scaffold for the final summary. (as shown in Fig. 1). The HRLT and sentence-wise entities so generated are added to a summarization-specific natural language prompt that is passed to a pre-trained natural language LLM to generate the final summary. Also, we use a novel GAT-based in-context example selection method (see Section 4.4).

# 4.1 Planning - HRLT Generation

To generate the final summary in a well-organized fashion, we first attempt at generating a layout of the references that are to appear in the final summary. The layout is hierarchical in nature, where we first to try to predict as to which references will be cited adjacent to each other/will appear in the same citation bracket, followed by predicting which

】 M INTENT INTENT INTENT IN-CONTEXT HOSE SDOMNE GAT-BASED FPEFONSGE INTENTiaspehesebyan ICE (IN-CONTEXT DOCS. ROOT E SELECTION SUMMARY ...@cite_1_4_2...@cite_3_5..SENTENCE pool of utterances.. LEVEL Abstract of @cite_1 @ite4ii One of the key tasks for analyzing LEVEL conversational data issegmentingit 1 Aetof2 into coherent topicsegments.However, Ge LEAES   
SOURCE rft   
DTES) Prn Abstract of @cite_4 # Extracting topics/entities relevant to each srci Identifying influential speakers in multi- REASON sentence containing citations indebates.Additionally，@cite_2 focuseson party conversations has been the focus visualizing sentiment dynamics from Twitter ofresearchincommunicati. Abstract of @cite_5 @cite_5highlighttheroleof socialmedia in tet deuanalism" ouralisdtutf

references will be cited in the same sentence. Such a hierarchical layout ensures that there is - (1) logical flow and association of content corresponding to the references (2) no citation in the summary that is not present in the set of source documents. We frame our problem of HRLT generation as a code completion problem similar to Madaan et al. (2022), which solves structured commonsense reasoning tasks via code completion1. The desired format of the generated code is shown in Fig. 1, where references within the same citation bracket are grouped together into lists to form the cit brack dictionary. These groups are then further grouped based on whether the citations appear in the same sentence or not, thus populating the sentence dictionary. This code generation is carried out in an in-context learning setting (Brown et al. 2020) (described in Section 4.4) so that this desired format is obtained.

# 4.2 Reasoning - Sentence-Wise Entity Generation

Berezin and Batura (2022) has shown that named entitybased supervision improves abstractive text summarization performance. Similarly, we extract entities as a reasoning aid to improve the final summary so obtained. This is also carried out in an in-context learning setting (described in Section 4.4). For the in-context learning samples, we extract scientific entities from sentences in the ground truth summary containing references using scispaCy2(Neumann et al. 2019). After extracting the entities, we convert it into the format of a code (as shown in Fig. 1), where we populate the dictionary topics sentence-wise. We name the dictionary as topics due to the entities extracted being topics relevant to the sentences in the final summary.

All in all, we generate the code for populating citation bracket. sentence, and topics dictionaries in an in-context learning setting. These dictionaries would serve as planning, reasoning-based information when generating the summary, which is elaborated in Section 4.3. Structure of the 1-shot prompt is shown in Prompt 1.

# 4.3 Generating the Final Summary

To generate the final summary, the code (generated in case of the test samples and extracted in case of the in-context samples) is converted into natural language. This planning and reasoning information is appended to the intent and source documents in the in-context examples (which contain ground truth summaries at each example’s end) and the test example at hand to create the complete input prompt (see the prompt structure in Prompt 2).

# 4.4 In-Context Example Selection using Graph-Attention Network (GAT)

The LLM is provided with task-specific examples and instructions within the same session, without updating its parameters, thus enriching the output given by the LLM. This means that the examples used in the ICL prompt should be relevant and helpful to the test example for which the task is to be solved by the LLM. For this, we use an vector embedding-based similarity approach to pick examples from a training set that have the most similar embeddings to that of the test example. We introduce a novel method to learn a single embedding to represent the collection of documents to be summarized.

Algorithm 1: Structure of 1-shot Prompt used Planning and   
Reasoning in MiDAS-PRo for Multi-XScience 1 ##Example 1:   
2 def main(): 3 # Given is a dictionary of paper abstracts cited as references in the related work of a query paper, and query paper abstract4 reference_paper_abstracts $\mathbf { \Sigma } = \mathbf { \Sigma }$ dict() 5 reference_paper_abstracts["@cite_30" ]="<@cite_30 ABSTRACT>" 6 reference_paper_abstracts["@cite_19" ]="<@cite_19 ABSTRACT>" 7 query_paper_abstract $\mathbf { \Sigma } = \mathbf { \Sigma }$ "<ABSTRACT OF QUERY RESEARCH PAPER>" 8 def hierarchical_clustering(): 9 # Hierarchical Clustering of references within Related Work Section of query paper   
10 cit_brack $\mathbf { \Sigma } = \mathbf { \Sigma }$ {} # contains lists as values that show how references are grouped within same citation bracket in Related Work of query paper   
11 sentence $\mathbf { \Sigma } = \mathbf { \Sigma }$ {} # contains lists, each list contains references in a sentence in Related Work of query paper   
12 topics $\mathbf { \Sigma } = \mathbf { \Sigma }$ {} # contains lists as values, each list contains topics/ entities relevant to a sentence   
13 cit_brack["@cite_19_30"] $\mathbf { \Sigma } = \mathbf { \Sigma }$ ["@cite_19 ", "@cite_30"]   
14 sentence["@cite_19_30"] $\mathbf { \Sigma } = \mathbf { \Sigma }$ [cit_brack[ "@cite_19_30"]]   
15 topics["@cite_19_30"] $\mathbf { \Sigma } = \mathbf { \Sigma }$ ["automobiles ", "economic models"]   
16 ##Example 2:   
17 def main():   
18 # Given is a dictionary of paper abstracts cited as references in the related work of a query paper, and query paper abstract  
19 reference_paper_abstracts $\mathbf { \Sigma } = \mathbf { \Sigma }$ dict()   
20 reference_paper_abstracts["@cite_9"] $\ c =$ "<@cite_9 ABSTRACT>"   
21 reference_paper_abstracts["@cite_15" ]="<@cite_15 ABSTRACT>"   
22 query_paper_abstract $\mathbf { \Sigma } = \mathbf { \Sigma }$ "<ABSTRACT OF QUERY RESEARCH PAPER>"   
23 def hierarchical_clustering():   
24 # Hierarchical Clustering of references within Related Work Section of query paper   
25 cit_brack $\mathbf { \Sigma } = \mathbf { \Sigma }$ {} # contains lists as values that show how references are grouped within same citation bracket in Related Work of query paper   
26 sentence $\mathbf { \Sigma } = \mathbf { \Sigma }$ {} # contains lists, each list contains references in a sentence in Related Work of query paper   
27 topics $\mathbf { \Sigma } = \mathbf { \Sigma }$ {} # contains lists as values, each list contains topics/ entities relevant to a sentence

# Algorithm 2: Structure of 1-shot Prompt used for generating summary in MiDAS-PRo for Multi-XScience

1 ##Example 1:   
2 Paper Abstracts:   
3 @cite_30 : <ABSTRACT OF @cite_30 PAPER>   
4 @cite_19 : <ABSTRACT OF @cite_19 PAPER>   
5 Query Paper Abstract:   
6 <ABSTRACT OF QUERY RESEARCH PAPER>   
7 Generate related work section.   
8 @cite_19, @cite_30 are in same citation bracket (i.e., they are right next to each other) within Related work. @cite_19 , @cite_30 are in same sentence within Related work - contains following entities - <TOPICS IN SENTENCE>.   
9 Answer: <RELATED WORK>   
10 ##Example 2:   
11 Paper Abstracts:   
12 @cite_9 : <ABSTRACT OF @cite_9 PAPER>   
13 @cite_15 : <ABSTRACT OF @cite_15 PAPER>   
14 Query Paper Abstract:   
15 <ABSTRACT OF QUERY RESEARCH PAPER>   
16 Generate related work section.   
17 @cite_9, @cite_15 are in same citation bracket (i.e., they are right next to each other) within Related work. @cite_9, @cite_15 are in same sentence within Related work - contains following entities - <TOPICS IN SENTENCE>.   
18 Answer:

First, we create a star graph $\mathcal { G }$ , where the central node $\boldsymbol { v } _ { 0 }$ (connected to all other nodes of the graph) represents the intent, and the other nodes $v _ { 1 } , v _ { 2 } , \dotsc , v _ { N }$ . . , vN represent the N source documents. Let $h _ { i }$ denote the embedding of node $v _ { i }$ for $i = 0$ , 1, . , $N$ . We initialize the central node embedding $h _ { 0 }$ with the intent representation and other node embeddings $h _ { i }$ (for $i = 1$ , . . . , $N$ ) with embeddings of the respective source documents using Sentence-BERT (Reimers and Gurevych 2019) transformer encoder. We then implement a Graph Attention Network (GAT) (Velicˇkovic´ et al. 2018) to update the central node embedding $h _ { 0 }$ . The attention mechanism assigns weights $\alpha _ { 0 i }$ to each edge connecting central node $\scriptstyle v _ { 0 }$ with a source document node $\boldsymbol { v } _ { i }$ as follows:

$$
\alpha _ { 0 i } = \frac { e x p \big ( L e a k y R e L U \left( a ^ { T } \left[ W h _ { 0 } | | W h _ { i } \right] \right) \big ) } { \sum _ { j = 1 } ^ { N } e x p \big ( L e a k y R e L U \left( a ^ { T } \left[ W h _ { 0 } | | W h _ { j } \right] \right) \big ) }
$$

, where $a$ is the attention vector, $W$ is the weight matrix, and $| |$ denotes concatenation.

The central node embedding $h _ { 0 } ^ { \prime }$ is then updated as: $h _ { 0 } ^ { \prime } =$ $\begin{array} { r } { \sigma \left( \sum _ { i = 1 } ^ { N } \alpha _ { 0 i } W h _ { i } \right) } \end{array}$ , where $\sigma$ is a non-linear sigmoid activation function. $h _ { 0 } ^ { \prime }$ encapsulates both the overall content of the source documents and the intent.

Fig. 2 shows the training pipeline for training the GAT. Pairs of document collections are sampled from the training set. For each pair of collections $A$ and $B$ , the collection embeddings $\bar { h _ { 0 } ^ { \prime A } }$ and $h _ { 0 } ^ { \prime B }$ are generated by passing the corresponding star graphs $\mathring { \mathcal { G } } ^ { A }$ and $\mathbf { \bar { \boldsymbol { g } } } ^ { B }$ as inputs to the GAT

NORMALIZED TREE EDIT DISTANCE GROUNDTRUTH SUMMARY 1 GROUNDTRUTH ed SUMMARY2 during political events have been discussed The role of social media in journalism and inseveral studies that investigate the the evolution of topic dynamics in political dynamics of political debates and their debates areemphasizedb @cite_2 effectsonmedia coverage.For instance, @cite_3,and @cite_4.@cite_1 highlights @cite_1，@cite_4 introducetheSITS the intricate relationship between political hierarchicalBayesianmodel,Aditionally, communication, media coverage, and public @cite_2，@cite_3emphasizes visualizing perception, providing a foundation for our sentiment trends on Twitter during political exploration of the factors that influence how debates, offering a valuable perspective on |media select political speech.   
HIERARCHICAL lpubicreiDOCUMNENT DOCUMENT HIERARCHICAL