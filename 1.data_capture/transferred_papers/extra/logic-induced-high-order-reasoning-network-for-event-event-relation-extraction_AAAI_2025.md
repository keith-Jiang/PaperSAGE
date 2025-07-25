# Logic Induced High-Order Reasoning Network for Event-Event Relation Extraction

Peixin Huang1, Xiang Zhao2, Minghao $\mathbf { H } \mathbf { u } ^ { 3 }$ , Zhen $\mathbf { T a n } ^ { 1 * }$ , Weidong Xiao1

1National Key Laboratory of Information Systems Engineering, National University of Defense Technology, China 2Laboratory for Big Data and Decision, National University of Defense Technology, China 3Information Research Center of Military Science, China {huangpeixin15, xiangzhao, tanzhen08a, wdxiao}@nudt.edu.cn, huminghao16@gmail.com

# Abstract

To understand a document with multiple events, event-event relation extraction (ERE) emerges as a crucial task, aiming to discern how natural events temporally or structurally associate with each other. To achieve this goal, our work addresses the problems of temporal event relation extraction (TRE) and subevent relation extraction (SRE). The latest methods for such problems have commonly built document-level event graphs for global reasoning across sentences. However, the edges between events are usually derived from external tools heuristically, which are not always reliable and may introduce noise. Moreover, they are not capable of preserving logical constraints among event relations, e.g., coreference constraint, symmetry constraint and conjunction constraint. These constraints guarantee coherence between different relation types, enabling the generation of a unified event evolution graph. In this work, we propose a novel method named LogicERE, which performs high-order event relation reasoning through modeling logic constraints. Specifically, different from conventional event graphs, we design a logic constraint induced graph (LCG) without any external tools. LCG involves event nodes where the interactions among them can model the coreference constraint, and event pairs nodes where the interactions among them can retain the symmetry constraint and conjunction constraint. Then we perform high-order reasoning on LCG with relational graph transformer to obtain enhanced event and event pair embeddings. Finally, we further incorporate logic constraint information via a joint logic learning module. Extensive experiments demonstrate the effectiveness of the proposed method with state-of-the-art performance on benchmark datasets.

# Extended version — https://arxiv.org/abs/2412.14688

# Introduction

Interpreting news messages involves identifying how natural events temporally or structurally associate with each other from news documents, i.e., extracting event temporal relations and subevent relations. Through this process, one can induce event evolution graphs that arrange multiple-granularity events with temporal relations and subevent relations interacting among them. The event evolution graphs built through event-event relation extraction (ERE) are important aids for future event forecasting (Chaturvedi, Peng, and Roth 2017) and hot event tracking (Zhuang, Fei, and Hu 2023b). As shown in Figure 1, ERE aims to induce such an event evolution graph, in which the event mention storm involves more fine-grained subevent mentions, i.e., killed, died and canceled. Some of those mentions follow temporal order, e.g., died happens BEFORE canceled. Generally, predicting the relations between diverse events within the same document, such that these predictions are coherent and consistent with the document, is a challenging task (Xiang and Wang 2019).

Recently, significant research efforts have been devoted to several ERE tasks, such as event temporal relation extraction (TRE) (Zhou et al. 2021; Wang, Li, and $\mathrm { X u } 2 0 2 2$ ; Tan, Pergola, and He 2023) and subevent relation extraction (SRE) (Man et al. 2022; Hwang et al. 2022). Nonetheless, ERE is still challenging because most event relations lack explicit clue words such as before and contain in natural languages, especially when events scatter in a document. Accordingly, a few previous methods attempt to build a documentlevel event graph to assist in the cross-sentence inference, where the nodes are events, and the edges are designed with linguistic/discourse relations of event pairs (Zhuang, Hu, and Zhao 2023; Li and Geng 2024). Despite the success, these methods face two major issues. First, they are not capable of preserving logic constraints among relations, such as transitivity, during training time (Roth and Yih 2004). Second, the edges heuristically from external tools may introduce noise and cause exhaustive extraction (Phu and Nguyen 2021).

For the first issue, Wang et al. (2020) propose a constrained learning framework which enforces logic coherence amongst the predicted relation types through extra differentiable objectives. However, since the coherence is enforced in a soft manner, there is still room for improving coherent predictions. In this work, we show that it is possible to enforce coherence in a much stronger manner by feeding logic constraints into the event graphs. For the second issue, Chen et al. (2022) propose an event pair centered causality identification model which takes event pairs as nodes and relations of event relations as edges, refraining the external tools and enabling the causal transitivity reasoning. However, some useful prior relation constraints such as coreference are discarded. Moreover, we observe logical property information loss from the document to graph, which should be complied in global inference for

On Tuesday, there was a typhoon-strength (𝒆𝟏:storm) in   
Japan. One man got ( $\underline { { e _ { 2 } } }$ :killed) and thousands of people   
were left stranded. Police said an 81-year-old man ( $\stackrel { \cdot } { \underline { { e } } _ { 3 } }$ :died)   
in central Toyama when the wind blew over a shed, trapping   
him underneath. Later this afternoon, with the agency   
warning of possible tornadoes, Japan Airlines $\underline { { e _ { 4 } } }$ :canceled)   
230 domestic flights, $\underline { { e _ { 5 } } }$ :affecting) 31,600 passengers. Parent-Child 𝒆𝟏:storm Parent-Child Parent-Child 𝒆 :killed Coreference 𝒆𝟑:died Before 𝒆𝟒:canceled Before Before 𝒆𝟓:affecting Parent-Child Coref. constraint 𝑟(𝒆𝟏, 𝒆𝟐) ⟺ 𝑟(𝒆𝟏, 𝒆𝟑) Symmetry constraint 𝑟(𝒆𝟏, 𝒆𝟐) ⟺ 𝑟ҧ (𝒆𝟐, 𝒆𝟏) 𝑟1(𝒆𝟏, 𝒆𝟐) ∧ 𝑟2(𝒆𝟐, 𝒆𝟑) ⇒ 𝑟3(𝒆𝟏, 𝒆𝟑) Conjunct. constraint 𝑟1(𝒆𝟏, 𝒆𝟐) ∧ 𝑟2(𝒆𝟐, 𝒆𝟑) ⇒ ¬𝑟4(𝒆𝟏, 𝒆𝟑)

two aspects. Firstly, our proposed LCG consolidates both event centered and event pair centered graphs, so that it can reason over not only coreference property among events, but also high-order symmetry and conjunction properties among event pairs. Specifically, LCG defines two types of nodes, i.e., event nodes and event pair nodes. Accordingly, there are three types of edges: (1) Event-event edge for prior event relations (e.g., coreference), which retains the coreference constraint and model the information flow among event nodes. (2) Event pair-event pair edge for two event pairs sharing at least one event, which keeps the symmetry and conjunction constraints, as well as captures the interactions among event pair nodes. (3) Event-event pair edge for an event pair and its corresponding two events, which models the information flow from event nodes to event pair nodes. As LCG preserves these inherent properties of event relations, we can get enhanced event and event pair embeddings through high-order reasoning over it. Secondly, inspired by the logic-driven framework of Li et al. (2019), we soften the logical properties through differentiable functions so as to incorporate them into multi-task learning objectives. The joint logic learning module enforces our model towards consistency with logic constraints across both TRE and SRE tasks. It is also a natural way to combine the supervision signals coming from two different tasks.

ERE. We summarize these properties implied in the temporal and subevent relations as three logical constraints: (1) Coreference constraint: Considering the example in Figure 1, given that $e _ { 2 }$ :killed is BEFORE $e _ { 5 }$ :affecting and $e _ { 3 }$ :died is COREFERENCED to $e _ { 2 }$ :killed, $e _ { 3 }$ :died should be BEFORE $e _ { 5 }$ :affecting. (2) Symmetry constraint: As shown in Figure 1, $e _ { 1 }$ :storm is a PARENT of $e _ { 2 }$ :killed, indicting $e _ { 2 }$ :killed is a CHILD of $e _ { 1 }$ :storm. (3) Conjunction constraint: From Figure 1, if $e _ { 1 }$ :storm is a PARENT of $e _ { 3 }$ :died and $e _ { 3 }$ :died is BEFORE $e _ { 4 }$ :canceled, the learning process should enforce $e _ { 1 }$ :storm is a PARENT of $e _ { 4 }$ :canceled by considering the conjunctive logic on both temporal and subevent relations. These logical constraints depict the mutuality among event relations of TRE and SRE, enabling the generation of a unified event evolution graph. While previous researches center on preserving these properties with post-learning inference or differentiable loss functions (Wang et al. 2020), there is no effective way to endow the graph model with these logical constraints for global reasoning.

In this paper, we consider the above constraints and propose a novel ERE model, logic induced high-order reasoning network (LogicERE). Our intuition is to feed the logic constraints into the event graphs for high-order event relation reasoning and prediction. Specifically, we first build a logic constraint induced graph (LCG) that models these logical constraints through interactions between events and among event pairs. Then, we encode the heterogeneous LCG with relational graph transformer. This enables the model to effectively reasoning over remote event pairs while maintaining inherent logic properties of event relations. Finally, we introduce a joint logic learning module and design two logic constraint learning objectives to further regularize the model towards consistency on logic constraints.

LogicERE models these high-order logical constraints in

# Related Work

# Temporal Relation Extraction

Early studies usually employ statistical models combined with handcrafted features to extract temporal relations (Yoshikawa et al. 2009; Mani et al. 2006). These methods are of high computational complexity and are difficult to transfer to other types of relations.

Recently, with the rise of pre-trained language models, various new strategies have been applied to TRE task. To effectively model the long texts, some studies incorporate syntax information such as semantic trees or abstract meaning representation (AMR) to capture remote dependencies (Venkatachalam, Mutharaju, and Bhatia 2021; Zhang, Ning, and Huang 2022). Others construct global event graphs to enable the information flow among long-range events (Liu et al. 2021; Fei et al. 2022b). For example, Zhang, Ning, and Huang (2022) design a syntax-guided graph transformer to explore the temporal clues. Liu et al. (2021) build uncertaintyguided graphs to order the temporal events. To deepen the models’ understanding of events, some studies propose to induce external knowledge (Han, Zhou, and Peng 2020; Tan, Pergola, and He 2023). Han, Zhou, and Peng (2020) propose an end-to-end neural model that incorporates domain knowledge from the TimeBank corpus (Pustejovsky et al. 2003). Some multi-task strategies are also employed for this task (Huang et al. 2023; Knez and Zitnik 2024). These strategies can facilitate the models’ learning of complementary information from other tasks.

# Subevent Relation Extraction

In earlier studies, machine learning algorithms are utilized to identify the internal structure of events (Fei et al. 2022a; Glavas et al. 2014). Afterwards, the introduction of deep learning has led to new advances of this task. Zhou et al. (2020) adopt the multi-task learning strategy and utilize duration prediction as auxiliary task. Man et al. (2022) argue that some context sentences in documents can facilitate the recognition of subevent relations. Thus, they adopt the reinforcement learning algorithm to select informative sentences from documents to provide supplementary information for this task. Hwang et al. (2022) enforce constraint strategies into probabilistic box embedding to maintain the unique properties of subevent relations.

To sum up, existing work only focuses on either TRE or SRE, only a few studies seek to resolve both two tasks. Zhuang, Fei, and Hu (2023b); Zhuang, Hu, and Zhao (2023) adopt the dependency parser and hard pruning strategies to acquire syntactic dependency graphs that are task-aware. Zhuang, Fei, and Hu (2023a) propose to use knowledge from event ontologies as additional prompts to compensate for the lack of event-related knowledge. Li and Geng (2024) introduce the features of event argument and structure to obtain graph-enhanced event embeddings. However, the above works regard event relation extraction as a multi-class classification task, and do not guarantee any coherence between different relation types, such as symmetry and transitivity. Different from these works, our LogicERE guarantee coherence through a high-order reasoning graph which is embedded with three essential logical constraints, as well as joint logic learning.

# Model

This paper focuses on the task of event-event relation extraction (ERE). The input document $\mathcal { D }$ is represented as a sequence of $n$ tokens $\mathcal { D } ~ = ~ [ x _ { 1 } , x _ { 2 } , . . . , x _ { n } ]$ . Each document contains a set of annotated event triggers (the most representative tokens for each event) $\mathcal { E } _ { \mathcal { D } } = \{ e _ { 1 } , e _ { 2 } , . . . , e _ { k } \}$ The goal of ERE is to extract the multi-faceted event relations from the document. Particularly, we focus on two types of relations, i.e., Temporal and Subevent, corresponding to the label sets $\mathcal { R } _ { T e m p }$ which contains BEFORE, AFTER, EQUAL, VAGUE, and $\mathcal { R } _ { S u b }$ which contains PARENTCHILD, CHILD-PARENT, COREF, NOREL respectively. Note that each event pair is annotated with one relation type from either $\mathcal { R } _ { T e m p }$ or $\mathcal { R } _ { S u b }$ , as the labels within two sets are mutually exclusive.

There are four major parts in our LogicERE model: (1) Sequence Encoder, which encodes event context in input document, (2) Logic Constraint Induced Graph, which build a event graph preserving logic constraints, (3) High-Order Reasoning Network on LCG, which performs high-order reasoning with relational graph transformer to obtain enhanced event and event pair embeddings, and (4) Joint Logic Learning, which further incorporates logic properties through well-designed learning objectives.

# Sequence Encoder

To obtain the event representations and contextualized embeddings of the input document $\mathbf { \mathcal { D } } = [ x _ { t } ] _ { t = 1 } ^ { n }$ (can be of any length $n$ ), we leverage pre-trained RoBERTa (Liu et al. 2019) as a base encoder. We add special tokens ”[CLS]”

and ”[SEP]” at the start and end of $\mathcal { D }$ , and insert ”<t>” and $\overrightarrow { } < / t > \overrightarrow { }$ at the start and end of all the events to mark event positions (Chen et al. 2022). Thus, we have:

$$
h _ { 1 } , h _ { 2 } , . . . , h _ { n ^ { \prime } } = \mathrm { E n c o d e r } ( [ x _ { 1 } , x _ { 2 } , . . . , x _ { n ^ { \prime } } ] )
$$

where $h _ { i } \in \mathbb { R } ^ { d }$ is the embedding of token $x _ { i }$ . We employ the embeddings of token ”[CLS]” and $\because t > "$ to represent the document and the events respectively. If the document’s length exceeds the limits of RoBERTa, we adopt the dynamic window mechanism to segment $\mathcal { D }$ into several overlapping spans with specific step size, and input them into the encoder separately. Then, we average all the embeddings of ”[CLS]” and $\because t > 3$ of different spans to obtain the document embedding $h _ { [ \mathsf { C L S } ] } \in \mathbb { R } ^ { d }$ or each event embedding $h _ { e _ { i } } \ \in \ \mathbb { R } ^ { d }$ , respectively.

# Logic Constraint Induced Graph

Considering the logic constraints of event relation is based on the motivation that such logic properties comprehensively define the varied interactions among those events and relations. In this section, we construct a logic constraint induced graph (LCG) which can preserve these logic properties through unifying both event centered and event pair centered graphs. Specifically, given all the events of document $\mathcal { D }$ , LCG is formulated as $\mathcal { G } = \{ \mathcal { V } , \mathcal { C } \}$ , where $\nu$ represents the nodes and $\mathcal { C }$ represents the edges in the graph. We highlight the following differences of $\mathcal { G }$ from previous event graphs and event pair graphs.

First, there are two types of nodes in $\nu$ , i.e., the event nodes $\nu _ { e }$ and the event pair nodes $\nu _ { e p }$ . Each node in $\nu _ { e p }$ refers to a different pair of events from $\mathcal { D }$ . Instead of merely using events or event pairs as nodes, LCG preserves both of them, enabling high-order interactions through edges.

Second, for edges $\mathcal { C }$ , instead of using all edges between any two nodes, we design three types of edges following the logic constraints: (1) Event-event edges $\mathcal { C } _ { e e }$ for two events that are co-referenced , which is motivated by the coreference constraint in Introduction. These edges are optional. $\mathcal { C } _ { e e }$ contributes to event relation reasoning as co-referenced events are expected to share the same relations with other events. Meanwhile, no additional relations exist between co-referenced events. (2) Event pair-event pair edges $\mathcal { C } _ { p p }$ for two event pairs that share at least one event ,which is motivated by the symmetry constraint and conjunction constraint in Introduction. Particularly, for the TRE task, symmetry constraint exists in a pair of reciprocal relations BEFORE and AFTER, as well as two reflexive ones EQUAL and VAGUE. Similarly, the SRE task includes reciprocal relations PARENT-CHILD and CHILD-PARENT as well as reflexive ones COREF and NOREL. The conjunction constraint enables the relation transitivity in a single task, and unifies the ordered nature of TRE and the topological nature of SRE (Wang et al. 2020). (3) Event-event pair edges $\mathcal { C } _ { e p }$ for an event pair and its corresponding events. We design $\mathcal { C } _ { e p }$ to bridge the information flow between events and event pairs.

# High-Order Reasoning Network on LCG

We perform high-order reasoning on LCG, which takes the relation heterogeneity into account and captures diversified high-order interactions within events and event pairs.

Initial Node Embeddings. For global inference, we firstly initialize node embeddings. Formally, for the event node $e _ { i } \in \mathcal { V } _ { e }$ , we take the contextualized event embeddings from the sequence encoder for initialization:

$$
v _ { e _ { i } } ^ { ( 0 ) } = h _ { e _ { i } } \mathbf { W } _ { n }
$$

where 0 indicates the initial state and $\mathbf { W } _ { n } \ \in \ \mathbb { R } ^ { d \times 2 d }$ is a learnable weight matrix.

For the event pair node $e _ { i , j } \in \mathcal { V } _ { e p }$ , we concatenate two corresponding event embeddings:

$$
v _ { e _ { i , j } } ^ { ( 0 ) } = [ h _ { e _ { i } } | | h _ { e _ { j } } ]
$$

Node Embedding Update. Then, we adopt relational graph transformer (Bi et al. 2024) to enhance the node features with the relational information from neighbor nodes. Each layer $l$ is similar to the transformer architecture. It takes a set of node embeddings ${ \bf V } ^ { ( l ) } \in \mathbb { R } ^ { N \times d _ { i n } }$ as input, and outputs a new set of node embeddings V(l+1) ∈ RN×dout, where $N = | \mathcal { V } _ { e } | + | \mathcal { V } _ { e p } |$ is the number of nodes in LCG, $d _ { i n }$ and $d _ { o u t }$ are the dimensions of input and output embeddings.

In each layer, to integrate information from each neighbor, we adopt a shared self-attention mechanism (Vaswani et al. 2017) to calculate the attention score:

$$
\begin{array} { r } { \alpha _ { i j } = \mathrm { s o f t m a x } ( c o _ { i j } ) } \\ { c o _ { i j } = \frac { ( v _ { i } \mathbf { W } _ { q } ) ( v _ { j } \mathbf { W } _ { k } ) ^ { \mathrm { T } } } { \sqrt { d _ { k } } } } \end{array}
$$

where $N _ { i }$ is the first order neighbor set of node $i$ , $c o _ { i j }$ measures the importance of neighbor $j$ t $\mathbf { \sigma } _ { 0 } \mathbf { \cdot } \mathbf { W } _ { q } , \mathbf { W } _ { k } \in \mathbb { R } ^ { d _ { i n } \times d _ { k } }$ are learnable matrices, $d _ { k }$ is a scaling factor to assign lower attention weights to uninformative nodes.

Then we aggregate relational knowledge from the neighborhood information with weighted linear combination of the embeddings:

$$
v _ { i } ^ { ( l + 1 ) } = \sum _ { j \in N _ { i } } \alpha _ { i j } ^ { ( l ) } ( v _ { j } ^ { ( l ) } \mathbf { W } _ { v } ^ { ( l ) } )
$$

where $\mathbf W _ { v } ^ { ( l ) } \in \mathbb R ^ { d _ { i n } \times d _ { k } }$ is a learnable matrix. We also adopt multi-head attention to attend to information from multiple attention heads. Thus, the output of the $l$ -th layer for node $i$ is:

$$
v _ { i } ^ { ( l + 1 ) } = ( \left| \right| _ { c = 1 } ^ { C } \sum _ { j \in N _ { i } } \alpha _ { i j } ^ { ( l ) } ( v _ { j } ^ { ( l ) } \mathbf { W } _ { v } ^ { ( l ) } ) ) \mathbf { W } _ { o } ^ { ( l ) }
$$

where $C$ is the number of attention head and $\mathbf { W } _ { o } ^ { ( l ) } ~ \in \$ RCdk×dout is a learnable matrix.

Measure Edge Heterogeneity. It is intuitive that three types of edges in LCG contributes differently to ERE. Thus we propose to measure the edge heterogeneity and incorporate the edge features into node embeddings. Specifically, for each edge type in LCG, we learn a scalar:

$$
\beta _ { t } = r _ { t } \mathbf { W } _ { r }
$$

where $1 \leq t \leq T$ , $T$ is the number of edge types, $\boldsymbol { r } _ { t } \in \mathbb { R } ^ { 1 \times d }$ denotes the edge features specific to the edge type, ${ \bf W } _ { r } \in \{ 0 , \frac { \partial } { \partial r } \in \}$ $\mathbb { R } ^ { d \times 1 }$ is a learnable matrix. $\boldsymbol { r } _ { t }$ will be randomly initialized. Then we incorporate $\beta _ { t }$ as the attention bias into the attention score to adjust the interaction strength between two adjacent nodes:

$$
\widetilde { \alpha } _ { i j } = \mathrm { s o f t m a x } ( \beta _ { t } + c o _ { i j } )
$$

As the result, tehe final updated node embeddings considering the edge heterogeneity is:

$$
\widetilde { v } _ { i } ^ { ( l + 1 ) } = \big ( \big | \big | \sum _ { c = 1 } ^ { C } \widetilde { \alpha } _ { i j } ^ { ( l ) } ( v _ { j } ^ { ( l ) } \mathbf { W } _ { v } ^ { ( l ) } ) \big ) \mathbf { W } _ { o } ^ { ( l ) }
$$

By stacking multiple layers, the reasoning network could reach high-order interaction and maintain logic properties.

Learning and Classification. To predict whether there is the temporal or subevent relation between events $e _ { i }$ and $e _ { j }$ , we concatenate the embeddings of ”[CLS]”, $e _ { i }$ , $\boldsymbol { e } _ { j }$ and the corresponding event pair as the logic enhanced representation. Thus, the probability distribution of the relation can be obtained through linear classification:

$$
p _ { e _ { i , j } } = \mathrm { s o f t m a x } ( [ h _ { [ \mathrm { C L S } ] } | | \widetilde { v } _ { i } | | \widetilde { v } _ { j } | | \widetilde { v } _ { i , j } ] \mathbf { W } _ { p } )
$$

where $| |$ denotes concatenation and $\mathbf { W } _ { p }$ is a learnable matrix. For training, we adopt cross-entropy as the loss function:

$$
\mathcal { L } _ { 1 } = - \sum _ { e _ { i } , e _ { j } \in \mathcal { E } _ { \mathcal { D } } } ( 1 - y _ { e _ { i , j } } ) \mathrm { l o g } ( 1 - p _ { e _ { i , j } } ) + y _ { e _ { i , j } } \mathrm { l o g } ( p _ { e _ { i , j } } )
$$

where $y _ { e _ { i , j } }$ denotes the golden label.

# Joint Logic Learning

Inspired by the logic-driven framework for consistency of Li et al. (2019), we further design two learning objectives by directly transforming the logical constraints into differentiable loss functions.

Symmetry Constraint. Symmetry constraints indicate the event pair with flipping orders will have the reversed relation, the logical formula can be written as:

$$
\underset { e _ { i } , e _ { j } \in \mathcal { E } _ { D } , r \in \mathcal { R } _ { s y m } } { \bigwedge } r ( e _ { i } , e _ { j } )  \bar { r } ( e _ { j } , e _ { i } )
$$

where $\mathcal { R } _ { s y m }$ indicates the set of relations enforcing the symmetry constraint. We use the product $\mathbf { t }$ -norm and transformation to the negative log space and obtain the symmetry loss:

$$
\mathcal { L } _ { s y m } = \sum _ { e _ { i } , e _ { j } \in \mathcal { E } _ { \mathcal { D } } } | \log ( p _ { e _ { i , j } } ) - \log ( \overline { { p } } _ { e _ { j , i } } ) |
$$

Conjunction Constraint. Conjunctive constraint are applicable to any three related events $\boldsymbol { e } _ { i } , \boldsymbol { e } _ { j }$ and $\boldsymbol { e } _ { k }$ . It contributes to the joint learning of TRE and SRE. The conjunction constraint enforces the following logical formulas:

$$
\underset { r _ { 1 } , r _ { 2 } \in \mathcal { R } , r _ { 3 } \in D e ( r _ { 1 } , r _ { 2 } ) } { \bigwedge } r _ { 1 } ( e _ { i } , e _ { j } ) \wedge r _ { 2 } ( e _ { j } , e _ { k } )  r _ { 3 } ( e _ { i } , e _ { k } )
$$

$$
\underset { { r _ { 1 } , r _ { 2 } \in \mathscr { R } , r _ { 4 } \notin { \mathscr { L } } _ { \mathscr { D } } } } { \bigwedge } r _ { 1 } ( e _ { i } , e _ { j } ) \wedge r _ { 2 } ( e _ { j } , e _ { k } )  \neg { r _ { 4 } ( e _ { i } , e _ { k } ) }
$$

where $D e ( r _ { 1 } , r _ { 2 } )$ is a set composed of all relations from $\mathcal { R }$ that do not conflict with $r _ { 1 }$ and $r _ { 2 }$ .

Similarly, the loss function specific to conjunction constraint is:

$$
\begin{array} { r l } & { \mathcal { L } _ { c o n j } = \displaystyle \sum _ { e _ { i } , e _ { j } , e _ { k } \in \mathcal { E } _ { \mathcal { D } } } | \mathcal { L } _ { c _ { 1 } } | + \displaystyle \sum _ { e _ { i } , e _ { j } , e _ { k } \in \mathcal { E } _ { \mathcal { D } } } | \mathcal { L } _ { c _ { 2 } } | } \\ & { \qquad \mathcal { L } _ { c _ { 1 } } = \log ( p _ { e _ { i , j } } ) + \log ( p _ { e _ { j , k } } ) - \log ( p _ { e _ { i , k } } ) } \\ & { \mathcal { L } _ { c _ { 2 } } = \log ( p _ { e _ { i , j } } ) + \log ( p _ { e _ { j , k } } ) - \log ( 1 - p _ { e _ { i , k } } ) } \end{array}
$$

The final loss function combines the above logic learning and event relation learning objectives, where $\gamma$ are nonnegative coefficients to control the influence of each loss term:

$$
\mathcal { L } = \mathcal { L } _ { 1 } + \gamma _ { s y m } \mathcal { L } _ { s y m } + \gamma _ { c o n j } \mathcal { L } _ { c o n j }
$$

# Experiments

# Datasets and Metrics

We evaluate logicERE on four widely used datasets. MATRES (Ning, Wu, and Roth 2018) and TCR (Ning et al. 2018) are used to test the performance of TRE. HiEve (Glavas et al. 2014) is used for SRE. MAVEN-ERE (Wang et al. 2022) is used to test the joint learning performance. MATRES is a new dataset and mainly annotates four temporal relationships, i.e., BEFORE, AFTER, EQUAL and VAGUE. TCR is a small-scale dataset which has the same annotation scheme as MATRES. We only apply it to the testing phase. HiEve is a news corpus and annotates four subevent relationships, i.e., PARENT-CHILD, CHILD-PARENT, COREF and NOREL. MAVEN-ERE is a unified large-scale dataset that annotates data for event coreference, temporal, causal, and subevent relations. We only experiment on the event temporal and subevent relations. For TRE, it defines six relationships. To be consistent with our framework, we only consider type BEFORE and SIMULTANEOUS, and we manually annotate reflexive relationships AFTER and VAGUE, respectively. For SRE, it defines one relationships SUBEVENT and we manually annotate corresponding reflexive relationships SUPEREVENT. Note that HiEve and MAVEN-ERE provide ground-truth event coreference annotations, but MATRES does not. We follow Chen et al. (2023) and perform pretraining on MAVEN-ERE, and then use the pre-trained model to extract coreference data for MATRES. After the preprocessing steps, we add event-event edges $\mathcal { C } _ { e e }$ to MATRES. For compatible comparison, we utilize the same data splits as in prior work for the considered datasets. We briefly summarize the data statistics for the above datasets in Table 1.

We adopt the standard micro-averaged Precision (P), Recall (R) and F1-scores (F1) as evaluation metrics. All the results are the average of five trials of different random seeds in each experiment.

# Parameter Settings

Our implementation uses HuggingFace Transformers (Wolf et al. 2020) and PyTorch (Paszke et al. 2019). We employ RoBERTa-base (Liu et al. 2019) as the document encoder. As for the input of the encoder, we set the dynamic window size to 256, and divide documents into several overlapping windows with a step size 32. We use AdamW (Loshchilov and Hutter 2019) optimizer and learning rate is set to 2e-5. We adopt layer normalization (Ba, Kiros, and Hinton 2016) and dropout (Srivastava et al. 2014) between the high-order reasoning network layers. We perform early stopping and tune the hyper-parameters by grid search on the development set: heads $\mathbf { \bar { \Gamma } } _ { C } \in \{ 1 , 2 , 4 , 8 \}$ , dropout rate $\in \{ 0 . 1 , 0 . 2 , 0 . 3 \}$ and loss coefficients $\gamma _ { s y m } , \gamma _ { c o n j } \in \{ 0 . 1 , 0 . 2 , 0 . 4 , 0 . 6 \}$ .

Table 1: Data statistics for dataset MATRES, TCR, HiEve and MAVEN-ERE (TRE/SRE).   

<html><body><table><tr><td colspan="2">Dataset</td><td>Train</td><td>Dev</td><td>Test</td></tr><tr><td>MATRES</td><td>Document Event pairs</td><td>260 10,888</td><td>21 1,852</td><td>20 840</td></tr><tr><td>TCR</td><td>Document Event pairs</td><td>-</td><td></td><td>25 2,646</td></tr><tr><td>HiEve</td><td>Document Event pairs</td><td>80 35,001</td><td>-</td><td>20 7,093</td></tr><tr><td>MAVEN-ERE (TRE)</td><td>Document Event pairs</td><td>2,913 792,445</td><td>710 188,928</td><td>857 234,844</td></tr><tr><td>MAVEN-ERE (SRE)</td><td>Document Event pairs</td><td>2,913 9,193</td><td>710 2,826</td><td>857 3.822</td></tr></table></body></html>

# Baselines

We adopt the following state-of-the-art models for the MATRES, TCR, HiEve and MAVEN-ERE datasets respectively.

For the MATRES dataset, we consider the following baselines. (1) TEMPROB+ILP (Ning, Subramanian, and Roth 2019) incorporates temporal commonsense knowledge and integer linear programming; (2) Hierarchical (Adhikari et al. 2019) uses RoBERTa to encode different chunks of the document, and sets an additional BILSTM model to aggregate representations; (3) Joint Constrain (Wang et al. 2020) enforces consistency with joint constrained learning objectives; (4) Self-Training (Ballesteros et al. 2020) relys on multitask and self-training techniques; (5) Vanilla Classifier (Wen and Ji 2021) is a common event relation classifier based on the RoBERTa; (6) Relative Time (Wen and Ji 2021) is a stack propagation framework employing relative time prediction as hints; (7) Probabilistic Box (Hwang et al. 2022) uses probabilistic boxes as implicit constraints to improve the consistency; (8) TGAGCN (Zhuang, Hu, and Zhao 2023) is based on event-specific syntactic dependency graph; (9) OntoEnhance (Zhuang, Fei, and Hu 2023a) fuses semantic information from event ontologies to enhance event representation; (10) SDLG (Zhuang, Fei, and Hu 2023b) builds syntax-based dynamic latent graph for event relation reasoning.

For TCR, the following baselines are included in our comparison. (1) LSTM+knowledge (Ning, Subramanian, and Roth 2019) is a variant of TEMPROB $+$ ILP that incorporates temporal commonsense knowledge in LSTM; (2) HGRU (Tan, Pergola, and He 2021) maps event embeddings to hyperbolic space; (3) Poincare´ Embeddings (Tan, Pergola, and He 2021) learns rich event representations in hyperbolic spaces; (4) We also compare with TEMPROB+ILP, Vanilla

Table 2: Model performance on test data of MATRES for temporal relation extraction.   

<html><body><table><tr><td>Model</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>TEMPROB+ILP(2019) Hierarchical (2019) Joint Constrain (2020) Self-Training (2020) Vanilla Classifier (2021) Relative Time (2021) Probabilistic Box (2022) TGAGCN (2023) OntoEnhance (2023)</td><td>71.3 74.2 73.4 78.1 78.4 81.1 79.0</td><td>82.1 83.1 85.0 82.5 85.2 83.0 86.5</td><td>76.3 78.4 78.3 81.6 80.2 81.7 77.1 82.0 82.6</td></tr><tr><td>SDLG (2023) LogicERE (ours)</td><td>82.0 82.9</td><td>84.2 84.5</td><td>83.1 83.7</td></tr></table></body></html>

# Classifier, TGAGCN, OntoEnhance and SDLG.

For the HiEve dataset, we choose the following baselines. (1)StructLR (Glavas et al. 2014) is a supervised classifier combining event, bag-of-words, location, and syntactic features; (2) TACOLM (Zhou et al. 2020) uses duration prediction to assist SRE; (3) Similarly, we also compare with Joint Constrain, Vanilla Classifier, Hierarchical, TGAGCN, OntoEnhance and SDLG.

For MAVEN-ERE, we select RoBERTa (Liu et al. 2019) as the main baseline, which adopts RoBERTa as the document encoder and obtains event embeddings for pair-wise classification. According to whether multiple event relations are trained simultaneously, we consider two settings, i.e., (1) $\mathbf { R o B E R T a } _ { s p l i t }$ and (2) $\mathbf { R o B E R T a } _ { j o i n t }$ for comparisons. We also compare with (3) GraphEREsplit and (4) $\mathbf { G r a p h E R E } _ { j o i n t }$ (Li and Geng 2024), which adopt graphenhanced event embeddings for pair-wise classification. Besides, for each tasks, we explore related baselines in our experiments: (5) DocTime (Mathur et al. 2022) constructs temporal dependency graph for TRE; (6) MultiFeatures (Aldawsari and Finlayson 2019) employs discourse and narrative features for SRE.

# Comparison

Table 2, 3, 4 and 5 show the performance of the models on four datasets. Here, the performance for the models in previous work is inherited from the original papers.

TRE Evaluation. From the results in Table 2 and 3 for temporal relation extraction, we can draw the following observations: 1) LogicERE significantly outperforms previous models (with $p < 0 . 0 5 )$ on both datasets, which demonstrates its effectiveness for TRE. 2) Compared with other methods considering logic coherence (i.e., Joint Constrain and Probabilistic Box), LogicERE gains at least $5 . 4 \%$ F1 scores improvements on MATRES. This indicates that LogicERE is more effective in incorporating logic constraints in graphenhanced event embeddings, which preserves coherence of temporal relations and enriches the features in event embeddings. 3) LogicERE surpasses the graph-based model (i.e., TGAGCN, OntoEnhance and SDLG) by at least $0 . 6 \%$ and $1 . 1 \%$ F1 scores improvements on MATRES and TCR respectively. It shows the advantages of logic constraint induced graph for TRE. Importantly, LogicERE can accomplish state-of-the-art performance for TRE without any external knowledge. This is different from recent work that requires additional resources to secure good performance, such as ontology knowledge in OntoEnhance or dependency parsing in TGAGCN and SDLG.

Table 3: Model performance on test set of TCR for temporal relation extraction.   

<html><body><table><tr><td>Model</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>LSTM+knowledge (2019) TEMPROB+ILP (2019) Vanilla Classifier (2021) HGRU (2021) Poincaré Embeddings (2021) TGAGCN (2023)</td><td>79.3 89.2 88.3 85.0 89.2</td><td>76.9 76.7 79.0 86.0 84.3</td><td>78.1 78.6 82.5 83.5 85.5 86.7</td></tr><tr><td>OntoEnhance (2023) SDLG (2023)</td><td>89.6 88.3</td><td>84.3 87.0</td><td>86.8 87.6</td></tr><tr><td>LogicERE (ours)</td><td>90.8</td><td>86.7</td><td>88.7</td></tr></table></body></html>

Table 4: Model performance on test set of HiEve for subevent relation extraction. We focus on the performance for PARENT-CHILD (PC), CHILD-PARENT (CP), and their micro-average (Avg.).   

<html><body><table><tr><td rowspan="2">Model</td><td colspan="3">F1</td></tr><tr><td>PC</td><td>CP</td><td>Avg.</td></tr><tr><td>StructLR (2014) Hierarchical (2019) TACOLM (2020) Joint Constrain (2020) Vanilla Classifier (2021) TGAGCN (2023) OntoEnhance (2023)</td><td>52.2 63.7 48.5 62.5 62.8 65.2 66.7</td><td>63.4 57.1 49.4 56.4 52.3 56.0 57.3</td><td>57.7 60.4 48.9 59.5 57.5 60.6 62.0</td></tr><tr><td>SDLG (2023) LogicERE(ours)</td><td>64.9 68.3</td><td>58.5 62.5</td><td>61.7 65.4</td></tr></table></body></html>

SRE Evaluation. The results in Table 4 for subevent relation extraction exhibit similar observation: 1) LogicERE can effectively deal with SRE task, achieving a significant advantage over all methods (with $p < 0 . 0 5 \$ ). The reason may be that our model can benefit from high-order reasoning of subevent relations through logic induced reasoning network and can get logic-enhanced event embeddings for SRE. 2) Consistent with experimental results on MATRES and TCR, LogicERE still shows significant advantages over other logic-enhanced models (i.e., Joint Constrain) and graph-based models (i.e., TGAGCN, OntoEnhance and SDLG). This verifies the effectiveness of our model for SRE.

Joint Learning Evaluation. MAVEN-ERE contains annotations for both TRE and SRE. We experiment on it to evaluate the ability of LogicERE in jointly learning multiple event relations. From Table 5, we observe that: 1) The proposed Logic $\mathrm { E R E } _ { j o i n t }$ surpasses all baselines in F1 scores for both event relations. Compared with the best existing methods for each task, our model improves by $0 . 3 \%$ in TRE and $3 . 0 \%$ in SRE. 2) LogicERE has a considerable improvement in Precision. Logic $\mathrm { E R E } _ { j o i n t }$ gains on average $1 . 4 \%$ Precision improvement on the two tasks $( \mathrm { L o g i c E R E } _ { s p l i t }$ for $1 . 6 \% )$ ). This indicates that we incorporate logic constraints into LCG which enriches the features in event embeddings through high-order reasoning, and improves the performance in Precision. 3) The models in joint setting outperform corresponding ones in split setting. Meanwhile, the improvement from Logic $\mathrm { E R E } _ { s p l i t }$ to $\mathrm { L o g i c E R E } _ { j o i n t }$ are $0 . 9 \%$ and $4 . 6 \%$ on TRE and SRE respectively, indicting that learning from other types of relations boost the ERE performance. 4) SRE is a challenging task with almost all the models exhibiting less than $3 0 \%$ in F1 Score. However, Logic $\mathrm { E R E } _ { j o i n t }$ has the most significant advantage of $3 . 0 \%$ in F1 Score over baselines. The possible reason is that the global consistence ensured by LCG and joint logic learning naturally makes up for the weak supervision signals for SRE.

Table 5: Model performance on test set of MAVEN-ERE for temporal relation extraction and subevent relation extraction, split/joint stands for separately or jointly training TRE and SRE tasks.   

<html><body><table><tr><td rowspan="2">Model</td><td colspan="3">TRE</td><td colspan="3">SRE</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>DocTime (2022) MultiFeatures (2019)</td><td>54.4</td><td>53.0 1</td><td>53.7 1</td><td>16.4</td><td>19.9</td><td>18.0</td></tr><tr><td>RoBERTasplit (2024) RoBERTajoint (2024)</td><td>53.1 50.7</td><td>53.3 56.7</td><td>53.2 53.5</td><td>27.3 19.9</td><td>20.9 24.2</td><td>23.7 21.8</td></tr></table></body></html>

In general, the experimental results show that LogicERE can effectively combine the supervision signals from two tasks, assisting in the comprehension of both temporal relation and subevent relation.

# Ablation Study

We then conduct ablation study to elucidate the effectiveness of main components of our model. Likewise, we only present the results on MAVEN-ERE. In particular, we consider the following internal baselines. (1) w/o edge heterogeneity does not consider the edge heterogeneity, and thus the scalar $\beta _ { t }$ is removed from Eq. 9 in the relational knowledge aggregation process. (2) w/o coreference removes $\mathcal { C } _ { e e }$ from LCG and do not use ground-truth coreference annotations as training labels. (3) w/o event-event pair edges removes $\mathcal { C } _ { e p }$ from LCG. (4) w/o symmetry objective removes $\mathcal { L } _ { s y m }$ from Eq. 20. (5) w/o conjunction objective removes $\mathcal { L } _ { c o n j }$ from Eq. 20. (6) w/o joint logic learning removes the joint logic learning objective and adopt $\mathcal { L } _ { 1 }$ as the loss function.

Results are shown in Table 6. We can observe that: 1) Our full model significantly outperforms all internal baselines on MAVEN-ERE. Compared to LogicERE, w/o. edge heterogeneity drops $0 . 6 \%$ and $0 . 8 \%$ F1 scores for TRE and SRE respectively, validating the necessity of capturing the semantic information of different types of edges in LCG. 2) w/o. event-event pair edges shows huge performance drop. This result demonstrates that the deep information interaction between the event and the event pair contributes to more informative event embeddings, impelling better event pair relation inference. 3) Experimental results of w/o symmetry objective, w/o conjunction objective and w/o joint logic learning demonstrate that, after removing either of the two, the F1 scores go down. It indicates that these two kinds of logic constraints both contribute to our model. Simultaneously using two kinds of knowledge further improves the overall performance.

Table 6: Ablation results (F1 scores) on MAVEN-ERE.   

<html><body><table><tr><td>Model</td><td>TRE</td><td>SRE</td></tr><tr><td>LogicEREjoint (full)</td><td>55.0</td><td>30.3</td></tr><tr><td>w/o edge heterogeneity w/o coreference w/o event-event pair edges w/o symmetry objective w/o conjunction objective</td><td>54.4 (-0.6) 54.7 (-0.3) 53.5 (-1.5) 54.8 (-0.2) 54.5 (-0.5) 54.1 (-0.9)</td><td>29.5 (-0.8) 30.0 (-0.3) 28.8 (-1.5) 30.0 (-0.3) 29.7 (-0.6)</td></tr></table></body></html>

# Conclusion

We present a novel logic induced high-order reasoning network (LogicERE) to enhance the event relation reasoning with logic constraints. We first design a logic constraint induced graph (LCG) which contains interactions between events and among event pairs. Then we encode the heterogeneous LCG for high-order event relation reasoning while maintaining inherent logic properties of event relations. Finally, we further incorporate logic constraints with joint logic learning. Extensive experiments show that LogicERE can effectively utilize logic properties to enhance the event and event pair embeddings, and achieve state-of-the-art performance for both TRE and SRE. The joint learning evaluation reveals that LogicERE can effectively maintain global consistency of two types relations, assisting in the comprehension of both temporal and subevent relation.