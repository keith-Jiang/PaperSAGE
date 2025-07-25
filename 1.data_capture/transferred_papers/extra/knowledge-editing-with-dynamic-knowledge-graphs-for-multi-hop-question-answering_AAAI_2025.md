# Knowledge Editing with Dynamic Knowledge Graphs for Multi-Hop Question Answering

Yifan $\mathbf { L u } ^ { 1 }$ , Yigeng Zhou1, $\mathbf { J i n g \mathbf { L i } ^ { * } }$ , Yequan Wang2, Xuebo ${ { \bf { L i u } } ^ { 1 } }$ , Daojing $\mathbf { H e } ^ { 1 }$ , Fangming $\mathbf { L i u } ^ { 3 }$ , Min Zhang

1Harbin Institute of Technology, Shenzhen, China 2Beijing Academy of Artificial Intelligence, Beijing, China 3Pengcheng Laboratory, Shenzhen, China lu.yifan $@$ foxmail.com, jingli.phd $@$ hotmail.com

# Abstract

Multi-hop question answering (MHQA) poses a significant challenge for large language models (LLMs) due to the extensive knowledge demands involved. Knowledge editing, which aims to precisely modify the LLMs to incorporate specific knowledge without negatively impacting other unrelated knowledge, offers a potential solution for addressing MHQA challenges with LLMs. However, current solutions struggle to effectively resolve issues of knowledge conflicts. Most parameter-preserving editing methods are hindered by inaccurate retrieval and overlook secondary editing issues, which can introduce noise into the reasoning process of LLMs. In this paper, we introduce KEDKG, a novel knowledge editing method that leverages a dynamic knowledge graph for MHQA, designed to ensure the reliability of answers. KEDKG involves two primary steps: dynamic knowledge graph construction and knowledge graph augmented generation. Initially, KEDKG autonomously constructs a dynamic knowledge graph to store revised information while resolving potential knowledge conflicts. Subsequently, it employs a fine-grained retrieval strategy coupled with an entity and relation detector to enhance the accuracy of graph retrieval for LLM generation. Experimental results on benchmarks show that KEDKG surpasses previous state-of-the-art models, delivering more accurate and reliable answers in environments with dynamic information.

uestion: On which continent will the next Olympic Games take place?

Outdated Answer: The next Olympic Games will be held in Tokyo, Japan. Japan is in Asia.

![](images/fc09c47f3ca2cf926e5d89e41d580087717f73d937529d9ce9b4bb1002e58091.jpg)

(a) An example of outdated knowledge stored in an LLM

Edited Memory: The next Olympic Games will be held in Paris.

![](images/b5dce11322fb7850288bc9848dd446d9ca65b8352eddf04af940bfe5d2a72bb6.jpg)

![](images/9933242c6a4460204a13a1a8961c90750f2166d20f301cdb3f57aafade7aef71.jpg)

Answer: The next Olympic Games will be held in Paris, France. France is in Europe.

(b)  A successful knowledge editing example under MHQA

Edited Memory: The next Olympic Games will be held in Paris.   
[NEW FACT] Los Angeles is the next Olympic host.

![](images/ed661d51eab7b5c931eec059b4f6bc4866aa7c446d8f6cc0540f18be5cb7a98a.jpg)

X

Answer: The two pieces of knowledge contradict each other…

(c)  A failed knowledge editing example under secondary editing

Figure 1: Examples of knowledge editing for MHQA. (a) an example of outdated information stored in LLMs. (b) a successful update with a parameter-preserving editing method. (c) a failure occurring during secondary editing.

# Introduction

Large language models (LLMs) have gained widespread adoption due to their advanced language understanding and reasoning capabilities (Zhao et al. 2023; Huang and Chang 2023). However, as the world changes and information becomes outdated, the datasets or corpora used to pre-train LLMs may no longer be relevant, potentially leading to unreliable outcomes in certain question-and-answer applications. To circumvent the substantial time and costs associated with retraining LLMs, researchers have shifted focus to knowledge editing (Zhang et al. 2024; Wang et al. 2023). The goal of knowledge editing is to precisely modify and update the knowledge within LLMs in a cost-effective manner, thereby enabling them to deliver dependable answers.

For LLMs, multi-hop question answering (MHQA) is a challenging task that requires high-order comprehension and reasoning abilities (Mavi, Jangra, and Jatowt 2022). In particular, knowledge editing in MHQA is often vulnerable due to the cascade effect. For example, as shown in Figure 1.(a) with the multi-hop question: “On which continent will the next Olympic Games take place?”, if the information about the next Olympic host is updated from Tokyo to Paris, then the answer to the multi-hop question should also be updated to reflect Paris’ location. Currently, approaches to knowledge editing for MHQA can be categorized into two main types: (1) Parameter-based editing methods: These methods directly adjust the parameters of the LLM to change its output (Zhu et al. 2020; Meng et al. 2022; De Cao, Aziz, and Titov 2021; Meng et al. 2023). While effective for singlehop questions, recent studies have shown that they face challenges with MHQA, including issues like catastrophic forgetting and degradation of the original model’s performance (Gu et al. 2024b). (2) Parameter-preserving editing methods: These methods retain the original model’s parameters and alter its behavior by adding extra modules (Mitchell et al. 2022; Huang et al. 2023; Zhong et al. 2023; Gu et al. 2024a; Cheng et al. 2024). A typical strategy involves storing the edited fact directly in memory and using retrievalenhanced techniques to prompt the LLM to modify its responses. Figure 1.(b) illustrates a successful application of this approach in MHQA.

However, existing approaches exhibit two primary limitations. First, the retrieval process often relies on semantic similarity matching, which can be inadequate and overlook information pertinent to the original question. In multihop questions, only the initial entity is known, requiring the LLM to infer intermediate entities. Second, as illustrated in Figure 1.(c), edited facts may require updates over time, a process we refer to as secondary editing. Directly storing secondarily edited knowledge in memory can conflict with previously edited facts, leading to interference in the retrieval process and errors during the LLM’s inference phase.

To address the aforementioned limitations, we propose a novel method called KEDKG: Knowledge Editing with Dynamic Knowledge Graphs for MHQA. Inspired by the evolving knowledge storage and convenient modification features of knowledge graphs, KEDKG initially converts edited knowledge into structured triples to establish a knowledge graph. This knowledge graph can dynamically expand, modify, or delete information as the world’s knowledge evolves. Utilizing this dynamic knowledge graph, KEDKG tackles the issue of secondary editing that may lead to potential conflicts, accurately providing the necessary edited knowledge for the LLM. When handling multi-hop questions, inspired by the chain-of-thought (CoT) approach (Wei et al. 2022), KEDKG use a fine-tuned LLM to decompose a multi-hop question into multiple sub-questions. For each sub-question, KEDKG perform fine-grained retrieval within the newly created knowledge graph and use detectors to filter the results, ensuring both the speed and accuracy of retrieval. Extensive experiments demonstrate that KEDKG surpasses existing state-of-the-art knowledge editing methods. We summarize the key contributions of our work as follows:

• To the best of our knowledge, this is the first approach to automatically construct dynamic knowledge graphs specifically for knowledge editing, and to investigate the knowledge conflict issue of secondary editing. • We propose a novel knowledge editing method, named KEDKG, for MHQA tasks that leverages knowledge graphs to resolve knowledge conflicts in secondary editing. KEDKG employs a fine-grained retrieval and filtering strategy to improve the accuracy of the retrieval process. Additionally, the method is lightweight and applicable with all open-source and black-box LLMs. • We conduct extensive experiments across various LLMs and datasets to validate the effectiveness and usability of KEDKG. Our empirical results and analysis demonstrate that KEDKG significantly outperforms the advanced existing baselines, achieving superior performance.

# Related Work Parameter-Preserving Editing Methods

Parameter-preserving editing methods do not modify the original model’s parameters. Instead, they introduce additional modules (such as neurons, retrievers, etc.) to influence the model’s behavior. SERAC (Mitchell et al. 2022) introduces a scope classifier to determine if an input request requires editing and employs a counterfactual model to handle requests. T-patcher (Huang et al. 2023) and CaliNet (Dong et al. 2022) incorporate additional trainable parameters in feed-forward layers of PLMs to provide edited knowledge. IKE (Zheng et al. 2023) utilizes in-context learning based on demonstration storage to prompt LLMs to edit knowledge with predefined templates. MeLLo (Zhong et al. 2023) is a simple editing method for multi-hop QA tasks, storing all edited facts externally while iteratively prompting LLMs to generate answers consistent with the edited facts. PokeMQA (Gu et al. 2024a) improves retrieval and model answer accuracy by dividing the task into 2 steps: question decomposition and decoupled self-checking. TEMPLEMQA (Cheng et al. 2024) directly generates multi-hop reasoning paths and constructs temporal graphs to enhance support for handling temporal multi-hop QA tasks. Unlike existing solutions, KEDKG constructs a structured knowledge graph to store the edited knowledge and supports dynamic updates to the edited knowledge.

# Parameter-Based Editing Methods

Parameter-based editing methods directly modify some or all parameters to incorporate edited knowledge into the model. Current methods can be categorized into three types: fine-tuning, meta-learning, and locate-then-edit methods. Traditional fine-tuning methods (Chen et al. 2020; Zhu et al. 2020) can enable the model to learn new knowledge, but it suffers from catastrophic forgetting. Meta-learning methods use a hyperparameter network to learn to modify model weights. For example, MEND (De Cao, Aziz, and Titov 2021) learns to transform the gradient of fine-tuned LLMs through low-rank decomposition of the gradient, but this approach incurs additional training overhead. Locate-thenedit methods directly identify specific parameters within the model and modify them. For instance, ROME (Meng et al. 2022) uses causal mediation analysis to locate the editing region and modifies the feed-forward layer parameters in that region to update knowledge. However, this method does not perform well for complex multi-hop QA problems. Recent research (Gu et al. 2024b) indicates that parameterbased editing methods are likely to significantly impact the model’s original performance.

# Preliminaries

Notations. We define each piece of factual knowledge embedded in the LLM as a triple $f = ( s , r , o ) \in F$ , where $F$ represents the collection of all facts. Each triple $f$ consists of a subject $s$ , a relation $r$ and an object $o$ . Edited knowledge shares the same subject and relation as the original knowledge, and the object is updated from $o$ to $o ^ { * }$ , denoted as $e =$ $( s , r , o \to o ^ { * } ) ,$ ). For example, a possible edit could be (Brazil,

Step 1: Dynamic Knowledge Step 2: Knowledge Graph Graph Construction Augmented Generation   
2=   
王 Multi-hop Question Edited Facts Relation Fine-tuned LLM for cEiltlize Kneomf pCerro iast a. iTshRe cahutahrdorDoafwMkiisnesr.y ExMtroadcteilon tQh:eWauhtahtoirs ot fh “eMniastei royn”a?lity of Question Decomposition … O   
窗 Detector Subquestions Edited Fact Triplets Entity Q1: Who is the author of Knowledge Graph detector “Misery”? Ellie Kemper citizen of Croatia citizen of Misery author Richard Dawkins Q72077 Q224 Relation Q2: What is the nationality Richard Dawkins notable work Misery Q970867 detector of [ENT]? author notable work 中 Q596874 Q44461 A1: Richard Dawkins 克 Base LLM for Entity Linker citizen of …… Q18 A2: [NOT FOUND] Question Answering Q72077 Q224 author S notable work Conflict all the subquestions solved A1: Richard Dawkins. WIKIDATA Q596874 ·· Q44461 Detection Output: British A2: British. Replace [ENT]

continent, South America $\mathrm {  \mathrm { A s i a } } _ { \mathrm { \ell } } ^ { \mathrm { \bullet } }$ . A knowledge editing operation can be represented as $E = \{ e _ { 1 } , e _ { 2 } , . . . , e _ { m } \}$ , potentially encompassing multiple edited triples of knowledge.

Multi-hop QA under knowledge editing. Given a multihop question $Q$ , answering $Q$ requires sequentially querying and retrieving multiple facts. Following the order in which these facts are retrieved, these facts can be represented a chain of facts $C = [ ( s _ { 1 } , r _ { 1 } , o _ { 1 } ) , . . . , ( s _ { n } , r _ { n } , \bar { o _ { n } } ) ]$ , where $s _ { i + 1 } ~ = ~ o _ { i }$ and $o _ { n }$ is the final answer to the question $Q$ . If we replace old knowledge in the chain of facts with new knowledge $( s _ { i } , r _ { i } , o _ { i } ^ { * } )$ , the cascading effect inherent in multi-hop question tasks affects the entire chain. Consequently, the final reasoning chain becomes ${ \cal C } ^ { * } \ =$ $[ ( s _ { 1 } , r _ { 1 } , o _ { 1 } ) , . . . , ( s _ { i } , r _ { i } , o _ { i } ^ { * } ) , . . . , ( s _ { n } ^ { * } , r _ { n } , o _ { n } ^ { * } ) ]$ , where ${ o } _ { n } ^ { * }$ is the updated answer. It should be noted that any number of old knowledge triples can be replaced in the chain of facts. The goal of the knowledge editing task for MHQA can be outlined as follows, given an editing set $E$ and a language model $M$ , generate a conditionally edited language model $M ^ { * }$ . For each multi-hop question influenced by the editing operations in $E$ , $M ^ { * }$ should be able to infer the correct answer $\boldsymbol { o } _ { n } ^ { * }$ . Meanwhile, the reasoning chain needs to align with $C ^ { * }$ , which we call the golden path for question $Q$ . KEDKG freezes the parameters of $M$ and utilizes a dynamic knowledge graph $\mathcal { G }$ generated by the editing set $E$ as an external knowledge base to guide $M$ in generating the answer.

# Methodology

KEDKG is a lightweight memory-based knowledge editing approach that stores edited knowledge in the form of a knowledge graph. This facilitates subsequent retrieval and enables dynamic updates to the graph in response to realworld knowledge changes. Additionally, we decouple multihop question decomposition from the original model to preserve its performance, handling this decomposition by finetuning an additional language model. As illustrated in the Figure 2, KEDKG consists of two main steps: (1) dynamic knowledge graph construction, and (2) knowledge graph augmented generation.

# Dynamic Knowledge Graph Construction

Edited knowledge triples extraction. Given a set of plain texts $T = \{ t _ { 1 } , t _ { 2 } , . . . , t _ { n } \}$ that represent edited knowledge in natural language form, KEDKG employs a relation extraction model to transform these natural texts into structured triples. This process can be described as follows:

$$
E _ { i } = \{ e _ { 1 } ^ { i } , e _ { 2 } ^ { i } , . . . , e _ { m } ^ { i } \} = R e t r i e v e r ( t _ { i } ) , i \in [ 1 , n ]
$$

where Retriever is the relation extraction model.

Previous approaches typically stored natural texts directly in memory, retrieving the most relevant knowledge based on semantic similarity to the question. However, this method is often influenced by syntactic structures and a variety of comparable words, which significantly affects accuracy. By pre-extracting relations from natural texts, we can more effectively capture the relationships between entities. As shown in Figure 2, from the sentence “The author of Misery is Richard Dawkins”, we can extract two pieces of edited knowledge: (Misery, author, Richard Dawkins) and (Richard Dawkins, notable work, Misery), the latter of which is often overlooked by conventional editing methods.

Entity linking. To store the edited knowledge in triple form into the knowledge graph, we note that different entity names might refer to the same entity (e.g., United States and U.S. denote the same entity). We link the extracted entities to Wikidata and store their corresponding node IDs and aliases in the knowledge graph.

Conflict detection and modification. Traditional knowledge editing methods do not account for the necessity of updating edited knowledge as the world evolves, a process we refer as secondary editing. Secondary editing can introduce conflicting knowledge which, if stored directly in memory, negatively impacts subsequent retrieval and is difficult to detect. The utilization of knowledge graphs can simplify this process. For a new piece of edited knowledge $e _ { n e w } \ \stackrel { \cdot } { = } \ ( s , \stackrel { \cdot } { r } , o _ { n e w } ^ { * } )$ , we locate the old edited knowledge eold = (s, r, oo∗ld) based on the subject s and relation r, then remove it and add $e _ { n e w }$ to the knowledge graph. If $e _ { o l d }$ is not found, this indicates that the knowledge has not undergone secondary editing, and $e _ { n e w }$ can be added directly.

# Knowledge Graph Augmented Generation

Unlike the MeLLo (Zhong et al. 2023) approach, which iteratively decomposes multi-hop questions, we propose an approach that utilizes a template $P _ { d i v i d e }$ to guide a trained LLM to decompose a multi-hop question $Q$ into multiple sub-questions in a single step:

$$
\{ q _ { 1 } , q _ { 2 } , . . . , q _ { n } \} = L L M ( P _ { d i v i d e } ( Q ) )
$$

where $P _ { d i v i d e } ( Q )$ is the prompt obtained by filling the multihop question $Q$ into the template $P _ { d i v i d e }$ . Except for the first sub-question $q _ { 1 }$ , which contains the subject $s$ from $Q$ , the subjects in other sub-questions are replaced by a special marker $[ E N T ]$ . We believe this can effectively reduce the burden on LLMs in understanding contexts during iteration, thereby improving the accuracy of question decomposition.

To search for relevant edited knowledge in the knowledge graph, we employ a fine-grained retrieval scheme and filter the retrieval results using entity and relation detectors. When processing each generated sub-question $q _ { i }$ , we first use a relation extraction model to generate a set of possible entities $K = \{ k _ { 1 } , . . . k _ { n } \}$ . Then, we use an entity detector $g _ { \phi } ( q _ { i } , k _ { j } ) \to [ 0 , 1 ]$ to predict the probability that entity $k _ { j }$ is the subject of $q _ { i }$ . We select $s = \arg \operatorname* { m a x } _ { k _ { i } \in K } g _ { \phi } ( q _ { i } , k _ { j } )$ as the subject and link it to the knowledge graph. If the subject does not exist in the knowledge graph, it indicates that no information related to the subject needs to be modified, and we directly call the original LLM for output. Otherwise, we can easily obtain the set of relations $R = \{ r _ { 1 } , . . . , r _ { m } \}$ related to the entity and use a relation detector $g _ { \psi } ( q _ { i } , r _ { j } )  [ 0 , 1 ]$ to detect the probability of the relation $r _ { j }$ being relevant to the subject in $q$ . If the highest probability $p$ exceeds a threshold $\alpha$ , which is set to 0.5 in our experiments, we can retrieve the corresponding fact triple $( s , r , o ^ { * } )$ and use $o ^ { * }$ as the retrieval answer. Otherwise, we directly call the original LLM for the answer. The final answer $a _ { i }$ to the sub-question $q _ { i }$ can be expressed using the formula:

$$
a _ { i } = \left\{ \begin{array} { l l } { L L M ( P _ { r e t r i e v e } ( q _ { i } , o ^ { * } ) ) , } & { \mathrm { i f } p > \alpha } \\ { L L M ( P _ { a n s w e r } ( q _ { i } ) ) , } & { \mathrm { i f } s \notin G \mathrm { o r } p < = \alpha } \end{array} \right.
$$

$$
o ^ { * } = \mathcal { G } ( s , \arg \operatorname* { m a x } _ { r _ { j } \in R } { g _ { \psi } ( q _ { i } , r _ { j } ) } )
$$

where $P _ { a n s w e r }$ is the template guiding the original LLM in generating answers, $P _ { r e t r i e v e }$ is a template that prompts the LLM to refine the response based on the retrieved entities, and $\mathcal { G }$ is the knowledge graph constructed in Step 1.

Entity detector and relation detector can be implemented using any text classification model. After obtaining the answer $a _ { i }$ to $q _ { i }$ , we use $a _ { i }$ as the subject to fill in the special marker in the next sub-question $q _ { i + 1 }$ . This process is iteratively repeated until all sub-questions are resolved, culminating in the output of the final answer.

# Training Objectives

Question decomposition module. This module is designed to enable the LLM to learn both explicit and implicit atomic sentence meanings and the relationships between sentences. It enhances the LLM’s ability to decompose a complex question into multiple sub-questions. This objective focuses on training the LLM to predict subsqequent tokens based on previous tokens. Specifically, given the predefined prompt template $\pmb { p }$ and question $\pmb q$ , the objective function for generating the sub-questions sequence $\mathbf { \bar { \rho } } o = [ o _ { 1 } , . . . , o _ { T } ]$ is outlined as follows:

$$
\mathcal { L } _ { \mathrm { d e c } } ( \theta ) = - \sum _ { t = 1 } ^ { T } \log p _ { \theta } \big ( o _ { t } | o _ { < t } , p , q )
$$

where $\theta$ represents the parameters of the model, and the output sequence $o = [ q _ { 1 } , \dot { . . . } , q _ { H } ]$ is composed of multiple subclauses derived from decomposition. $q _ { i }$ represents each individual sub-question, and $H$ represents the number of subquestions.

Entity detector and relation detector. Specifically, given the input pair $( q _ { i } , k _ { j } )$ or $( q _ { i } , r _ { j } )$ , the training objective of the entity detector $g _ { \phi }$ and relation detector $g _ { \psi }$ is to minimize the average binary cross-entropy loss over the training dataset:

$$
\mathcal { L } _ { \mathrm { d e t } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } [ y _ { i } \log ( p _ { i } ) + ( 1 - y _ { i } ) \log ( 1 - p _ { i } ) ]
$$

where $N$ denotes the number of training samples. $y _ { i }$ represents the true label of the $i$ -th sample, and $p _ { i }$ represents the probability that the $i$ -th sample is predicted to be a subject or relation in sub-question $q _ { i }$ . For more details about training, please refer to Appendix C and D.

# Experiments Experimental Settings

Datasets. We evaluate KEDKG using the MQuAKE dataset. MQuAKE is a knowledge editing benchmark for multi-hop QA, comprising MQuAKE-CF based on counterfactual editing and MQuAKE-T based on temporal knowledge updates. We use MQuAKE-CF as the training set, which contains 9,218 data points, and MQuAKE-CF-3k as the test set, which includes 3,000 data points. It is important to note that there are no overlapping data points between these two sets. These datasets include numerous $k$ -hop questions $( k \in \{ 2 , 3 , 4 \} )$ , with each question corresponding to $n$ edits $( n \in [ 1 , k ] )$ . See Appendix A for more details.

Table 1: The experimental results on MQUAKE-CF-3K and MQUAKE-T benchmarks. The best results are indicated in bold, and the second-best results are underlined. Results marked with $*$ indicate they are sourced from (Gu et al. 2024a), and results marked with $\dagger$ indicate they are sourced from (Cheng et al. 2024). The notation $^ { 6 6 } k$ edited” refers to the size of the edit batch is $k$ . “COT” implies that the current method employs chain-of-thought prompting, otherwise, a decomposition prompt is used. The evaluation metrics are multi-hop accuracy (M-Acc.) and hop-wise answering accuracy (H-Acc.). “OOM” denotes that the method does not work due to high GPU memory usage.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="5">MQUAKE-CF-3K</td><td colspan="4">MQUAKE-T</td></tr><tr><td colspan="2">1 edited</td><td colspan="2">100 edited</td><td colspan="2">All edited</td><td colspan="2">1 edited</td><td colspan="2">All edited</td></tr><tr><td>M-Acc.</td><td></td><td>H-Acc.</td><td>M-Acc. H-Acc.</td><td>M-Acc.</td><td>H-Acc.</td><td>M-Acc.</td><td>H-Acc.</td><td>M-Acc.</td><td>H-Acc.</td></tr><tr><td colspan="10">LLaMa 2-7B</td></tr><tr><td>FTcoT*</td><td>22.30</td><td></td><td>2.13</td><td></td><td>OOM</td><td></td><td>47.32</td><td></td><td>3.75</td><td></td></tr><tr><td>FT*</td><td>28.20</td><td>7.30</td><td>2.37</td><td>0.03</td><td>0OM</td><td>0OM</td><td>56.48</td><td>33.89</td><td>1.02</td><td>0.37</td></tr><tr><td>ROMEcOT*</td><td>11.17</td><td></td><td>2.87</td><td></td><td>2.77</td><td></td><td>28.96</td><td></td><td>14.40</td><td></td></tr><tr><td>ROME*</td><td>13.13</td><td>5.37</td><td>3.50</td><td>0.03</td><td>3.63</td><td>0.10</td><td>24.89</td><td>17.99</td><td>1.71</td><td>0.32</td></tr><tr><td>MEMITcOT*</td><td>11.83</td><td></td><td>9.23</td><td></td><td>5.57</td><td></td><td>36.88</td><td></td><td>31.58</td><td></td></tr><tr><td>MEMIT*</td><td>14.97</td><td>6.43</td><td>9.40</td><td>2.47</td><td>2.30</td><td>0.37</td><td>30.89</td><td>23.98</td><td>25.21</td><td>20.13</td></tr><tr><td>MeLLot</td><td>33.57</td><td>9.90</td><td>20.00</td><td>10.07</td><td>17.33</td><td>9.90</td><td>65.78</td><td>55.27</td><td>57.69</td><td>44.55</td></tr><tr><td>PokeMQA*</td><td>44.13</td><td>30.60</td><td>37.33</td><td>27.83</td><td>32.83</td><td>23.87</td><td>75.43</td><td>60.44</td><td>74.36</td><td>60.22</td></tr><tr><td>KEDKG (Ours)</td><td>66.80</td><td>63.67</td><td>58.50</td><td>55.37 Vicuna-7B</td><td>48.30</td><td>43.90</td><td>73.13</td><td>69.06</td><td>71.15</td><td>66.76</td></tr><tr><td colspan="9"></td></tr><tr><td>MeLLo*</td><td>30.70</td><td>20.84</td><td>24.75</td><td>12.25</td><td>22.35</td><td>10.18</td><td>60.72</td><td>48.55</td><td>51.55</td><td>42.97</td></tr><tr><td>PokeMQA*</td><td>45.83</td><td>34.80</td><td>38.77</td><td>31.23</td><td>31.63</td><td>25.30</td><td>74.57</td><td>55.19</td><td>73.07</td><td>55.09</td></tr><tr><td>KEDKG (Ours)</td><td>68.60</td><td>65.13</td><td>62.43</td><td>58.20</td><td>51.10</td><td>44.67</td><td>71.90</td><td>67.23</td><td>74.68</td><td>66.54</td></tr><tr><td colspan="9">GPT-3.5-turbo-instruct</td></tr><tr><td>MeLLo*</td><td>57.43</td><td>28.80</td><td>40.87</td><td>28.13</td><td>35.27</td><td>25.30</td><td>88.12</td><td>52.84</td><td>74.57</td><td>53.53</td></tr><tr><td>PokeMQA*</td><td>67.27</td><td>56.37</td><td>56.00</td><td>49.63</td><td>45.87</td><td>39.77</td><td>76.98</td><td>68.09</td><td>78.16</td><td>67.88</td></tr><tr><td>KEDKG (Ours)</td><td>68.00</td><td>65.33</td><td>59.50</td><td>56.80</td><td>49.10</td><td>43.17</td><td>78.75</td><td>76.18</td><td>77.19</td><td>73.77</td></tr></table></body></html>

Baselines. We compare KEDKG with common knowledge editing methods, including parameter-preserving and parameter-based editing methods. The parameter-preserving editing methods include MeLLo (Zhong et al. 2023) and PokeMQA (Gu et al. 2024a), while the parameter-based editing methods include FT (Zhu et al. 2020), ROME (Meng et al. 2022), and MEMIT (Meng et al. 2023). We also report the performance of these methods under chain-of-thought (CoT) and question decomposition (QD) prompts. See Appendix B for more details.

Evaluation metrics. Following previous work (Zhong et al. 2023; Gu et al. 2024a), we use Multi-hop Accuracy (MAcc) and Hop-wise Answering Accuracy (H-Acc) as evaluation metrics. Multi-hop accuracy measures the accuracy of the LLM in answering multi-hop questions. However, in some cases, the LLM might produce an incorrect reasoning process but still arrive at the correct answer. Hop-wise answering accuracy measures the accuracy where every answer in the reasoning path is correct, avoiding such interference. This is the primary metric we focus on to assess the model’s ability to use edited knowledge. For all metrics, higher values indicate better performance.

Experimental setup. We train an entity detector and a relation detector based on the DistilBERT (Sanh 2019) model and fine-tune the Llama 2-7B model for the question decomposition task. In addition, we use REBEL (Cabot and Navigli 2021) as our relation extraction model and spacy entity linker as entity linking model. It is important to note that there is no overlap between the training and test datasets we use. We conducted training on multiple base models, including mainstream open-source models Llama 2-7B (Touvron et al. 2023) and Vicuna-7B (Zheng et al. 2024), as well as the black-box model GPT-3.5-turbo-instruct. Parameterpreserving editing methods are applicable to all models, while parameter-based editing methods are only applicable to open-source models. Therefore, we only reported the experimental results of parameter-based editing methods on open-source models. All our experiments are carried out on a NVIDIA 8 A800-SXM4-80G machine.

# Main Results

Table 1 shows the experimental results on MQUAKE datasets. We can draw the following observations:

(1) KEDKG demonstrates outstanding performance across all metrics, notably surpassing all baselines in the H-Acc metric. Utilizing Llama 2-7B as the base model,

Table 2: Ablation results of KEDKG and its variants in terms of Multi-hop Accuracy (M-Acc.) and Hop-wise Answering Accuracy (H-Acc.) on MQUAKE datasets. “CDM” refers to the Conflict Detection and Modification module.   

<html><body><table><tr><td rowspan="3">9 g4</td><td rowspan="3"></td><td rowspan="3">CDM</td><td colspan="6">MQUAKE-CF-3k</td><td colspan="4">MQUAKE-T</td></tr><tr><td colspan="2">1 edited</td><td colspan="2">100 edited</td><td colspan="2">All edited</td><td colspan="2">1 edited</td><td colspan="2">All edited</td></tr><tr><td>M-Acc.</td><td>H-Acc.</td><td>M-Acc.</td><td>H-Acc.</td><td>M-Acc.</td><td>H-Acc.</td><td>M-Acc.</td><td>H-Acc.</td><td>M-Acc.</td><td>H-Acc.</td></tr><tr><td>1</td><td>√</td><td>√</td><td>35.23</td><td>31.63</td><td>31.17</td><td>26.93</td><td>23.10</td><td>18.87</td><td>54.39</td><td>50.05</td><td>53.80</td><td>48.50</td></tr><tr><td>√</td><td>1</td><td>√</td><td>20.03</td><td>16.03</td><td>19.13</td><td>15.10</td><td>16.80</td><td>12.57</td><td>54.98</td><td>52.19</td><td>56.48</td><td>52.78</td></tr><tr><td>√</td><td>√</td><td></td><td>66.30</td><td>63.43</td><td>58.43</td><td>54.60</td><td>41.10</td><td>36.83</td><td>73.13</td><td>68.74</td><td>71.15</td><td>63.81</td></tr><tr><td>√</td><td>√</td><td>√</td><td>66.80</td><td>63.67</td><td>58.50</td><td>55.37</td><td>48.30</td><td>43.90</td><td>73.13</td><td>69.06</td><td>71.15</td><td>66.76</td></tr></table></body></html>

KEDKG achieves an improvement of $1 0 8 . 1 \%$ in 1 edit batch, $9 6 . 2 \%$ in 100 edit batches, and $8 3 . 9 \%$ in all edit batches compared to the best previous baseline. This robust performance underscores KEDKG’s capability to effectively answer multi-hop questions under conditions of knowledge editing and adapt seamlessly to varying numbers of edits.

(2) Notably, other methods exhibit a significant discrepancy between M-Acc and H-Acc, indicating that their reasoning processes do not align with the task logic presented in the prompt, which can be seen as a mismatch in question decomposition. In contrast, KEDKG demonstrates a smaller difference between M-Acc and H-Acc, primarily because we fine-tune the model specifically on the question decomposition task, decoupling it from the original model and reducing the burden of understanding complex prompts. This results in more reliable reasoning outcomes.

(3) Interestingly, on the MQUAKE-CF- $3 \mathrm { k }$ benchmark, KEDKG’s performance with a 7B parameter open-source base model matches or even surpasses that of GPT-3.5- turbo-instruct. This comparison reveals that the knowledge capability of the 7B parameter model is on par with that of the black-box model, though the latter demonstrates a stronger ability to understand and follow human instructions. KEDKG effectively bridges this gap, enhancing the 7B parameter model’s ability to adeptly handle multi-hop QA problems under conditions of knowledge editing.

(4) We also observe that all baselines perform best under the 1 edit batch setting, with performance declining as the edit batch size increases, a trend that is particularly pronounced in the MQUAKE-CF- $3 \mathbf { k }$ dataset. This phenomenon can be attributed to two main factors: first, conducting retrieval within large batches of edits is a challenging task, especially when the entities and relationships involved are very similar. Second, in large batch edits, the edited knowledge in different cases within the MQUAKE-CF-3k dataset may conflict with each other. Thanks to the integration of a conflict detector and the application of dynamic knowledge graphs, we have effectively improved KEDKG’s performance in scenarios involving large batch edits. Even under the all edit batch setting, KEDKG can still achieve an accuracy rate close to $50 \%$ .

# Ablation Study

We conduct ablation experiments with the base model Llama 2-7B on the MQUAKE-CF-3k and MQUAKE-T datasets to analyze the impact of $g _ { \phi } , g _ { \psi }$ , and conflict detection modifi

![](images/580286c5406a42dd5437fe6ed42de4e8da22984ab51462f6abc9933e2fbd3df5.jpg)  
Figure 3: Multi-hop Accuracy and Hop-wise Answering Accuracy results on MQUAKE-CF-3K, utilizing different knowledge editing methods. The experiments are conducted on Llama 2-7B and the edit batch size is 1.

cation (CDM) modules.

The trained entity detector $g _ { \phi }$ and relation detector $g _ { \psi }$ have proven to be effective. As shown in Table 2, replacing $g _ { \phi }$ or $g _ { \phi }$ with the semantic similarity model, allMiniLM-L6-v2 from Huggingface (Wolf et al. 2020), resulted in a decrease in answer accuracy across all experimental settings. The most significant performance decline was observed when the relation detector was replaced, with a $3 1 . 5 \%$ drop in Multi-hop Accuracy under the “all edited” setting. This decline highlights the limitations of relying solely on semantic similarity to select relations from the knowledge graph, as a specific relation may have various expressions, making such an approach unreliable.

Table 3: A case study of KEDKG solving one 3-hop question in MQUAKE-CF-3K. Yellow parts are generated by the finetuned LLM for question decomposition. Green parts are generated by the base LLM for question answering, and blue parts are answers retrieved in the knowledge graph.   

<html><body><table><tr><td></td><td colspan="4">Edited Knowledge: Association football was created in the country of Brazil. Brazil is located in the continent of Africa.</td><td colspan="4"></td></tr><tr><td></td><td></td><td></td><td>Knowledge Graph: (Association football, country of origin, Brazil); (Brazil， sport,</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Association football)； (Brazil，continent，Africa)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>Question: What is the continent of origin for the sport associated with Watford F.C.?</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>Subquestions: 1. Which sport is Watford F.C. associated with? 2. Which country was [ENT]</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>created in? 3. Which continent is [ENT] located in?</td><td></td><td></td></tr><tr><td></td><td></td><td>Retrieved answer:[None]</td><td>Thought 1:Which sport is Watford F.C.associated with?</td></tr><tr><td></td><td>Generated answer: Watford F.C. is associated with Association Football (Soccer).</td></tr><tr><td></td><td>Thought 2: Which country was Association Football (Soccer) created in? Retrieved answer: Brazil.</td></tr></table></body></html>

The CDM module effectively enhances performance in batch editing. The MQUAKE-CF dataset includes some conflicting edits, providing an opportunity to evaluate KEDKG’s conflict resolution capabilities. As shown in Table 2, the CDM module has no significant effect in small batch editing, which is intuitive since there are rarely conflicting edits in small batches. However, as the number of edits increases, the probability of conflicts also rises. Removing the CDM module led to a $7 . 2 \%$ drop in Multihop Accuracy. This decrease is attributed to the presence of conflicting knowledge triples within the knowledge graph, which disrupt the retrieval process. Introducing the CDM module effectively minimizes this interference and facilitates the execution of secondary edits. On the MQUAKET dataset, we observe that applying the CDM module does not significantly impact the results. This is intuitive because MQUAKE-T itself does not contain conflicting edits.

# Analysis

KEDKG excels in 3-hop and 4-hop question answering tasks. As shown in Figure 3, both KEDKG and PokeMQA perform well in two-hop questions. However, in 3-hop and 4-hop questions, KEDKG significantly outperforms PokeMQA, almost doubling the performance metrics. In the most complex 4-hop questions, KEDKG achieves over $50 \%$ Multi-hop Accuracy, while the accuracy of other methods remains below $2 5 \%$ . This superior performance is attributed to our training on the decomposition task, which effectively minimizes the likelihood of errors or hallucinations in the LLM during the decomposition process. Additionally, the fine-grained retrieval based on the knowledge graph significantly improves the accuracy of answers at each hop.

Meanwhile, we have also observed that parameter-based editing methods generally underperform compared to parameter-preserving editing methods in multi-hop QA tasks, exhibiting notably low Hop-wise Answering Accuracy. This suggests that current parameter-based editing methods still struggle to enable LLMs to flexibly utilize edited knowledge during the reasoning process.

# Case Study

We conduct a case study as presented in Table 3. KEDKG first extracts edited triples from the given edited knowledge. After passing through the entity linking and conflict detection modules, these triples are stored in the knowledge graph $\mathcal { G }$ . Note that Table 3 only shows the edited knowledge relevant to this case. In large-scale editing, there are many instances of edited knowledge and conflicting edits.

In the knowledge graph-based QA phase, the fine-tuned LLM used for question decomposition breaks down the multi-hop question into three sub-questions, where the subjects of the latter two sub-questions are replaced with special markers. Subsequently, the entity detector and relation detector are applied sequentially to retrieve answers for the three sub-questions. For the first sub-question, since no relevant facts about Watford $F C .$ . are found in $\mathcal { G }$ , the question is directly handed to the base LLM to produce an answer, and the entity in the answer is then filled into the next subquestion. For the latter two questions, KEDKG successfully retrieves the edited facts from $\mathcal { G }$ and prompts the LLM to provide answers, ultimately yielding the correct answer.

# Conclusion

In this paper, we propose a novel knowledge editing method, KEDKG for MHQA. KEDKG constructs a dynamic knowledge graph to store edited knowledge and to handle conflicting edits. Additionally, KEDKG employs a fine-grained retrieval scheme to fetch edited knowledge from the knowledge graph, guiding the model to modify its answers. Extensive experiments on the MQUAKE dataset demonstrate that KEDKG outperforms previous knowledge editing methods.