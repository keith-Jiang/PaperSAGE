# Adaptive Prompting for Continual Relation Extraction: A Within-Task Variance Perspective

Minh Le 1\*†, Tien Ngoc Luu 2†, An Nguyen The 3†, Thanh-Thien Le 1∗, Trang Nguyen 1∗, Tung Thanh Nguyen 4, Linh Ngo Van 2‡, Thien Huu Nguyen 5

1VinAI Research 2Hanoi University of Science and Technology 3FPT Software AI Center 4Moreh Inc. 5University of Oregon, Eugene, Oregon, USA v.minhld12 $@$ vinai.io, ngoc.lt204595 $@$ sis.hust.edu.vn, annt68 $@$ fpt.com, v.thienlt3 $@$ vinai.io, v.trangnvt2 $@$ vinai.io, tung.nguyen $@$ moreh.com.vn, linhnv $@$ soict.hust.edu.vn, thien $@$ cs.uoregon.edu

# Abstract

To address catastrophic forgetting in Continual Relation Extraction (CRE), many current approaches rely on memory buffers to rehearse previously learned knowledge while acquiring new tasks. Recently, prompt-based methods have emerged as potent alternatives to rehearsal-based strategies, demonstrating strong empirical performance. However, upon analyzing existing prompt-based approaches for CRE, we identified several critical limitations, such as inaccurate prompt selection, inadequate mechanisms for mitigating forgetting in shared parameters, and suboptimal handling of cross-task and within-task variances. To overcome these challenges, we draw inspiration from the relationship between prefix-tuning and mixture of experts, proposing a novel approach that employs a prompt pool for each task, capturing variations within each task while enhancing cross-task variances. Furthermore, we incorporate a generative model to consolidate prior knowledge within shared parameters, eliminating the need for explicit data storage. Extensive experiments validate the efficacy of our approach, demonstrating superior performance over state-of-the-art prompt-based and rehearsal-free methods in continual relation extraction.

# 1 Introduction

Continual Relation Extraction (CRE) involves classifying semantic relationships between entities in text while adapting to an expanding set of relation types. Previous CRE approaches (Zhao et al. 2022; Nguyen et al. 2023; Le et al. 2024d) have successfully addressed the challenge of learning new relations without sacrificing accuracy on previously learned ones by employing memory-based techniques (Shin et al. 2017; Chaudhry et al. 2019). These methods utilize a rehearsal mechanism supported by a memory buffer, enabling the model to revisit and consolidate knowledge of prior relations while learning new tasks, thereby reducing catastrophic forgetting. Nonetheless, concerns regarding data storage and privacy have prompted the research community to investigate alternative strategies for CRE (Ke and Liu 2022).

To address these limitations, recent advances in Continual Learning (CL) have introduced innovative prompt-based methods (Wang et al. 2022b,a, 2023a; Tran et al. 2024a). Unlike memory-based approaches, these methods eliminate the need for rehearsal, focusing instead on the dynamic insertion of auxiliary parameters, known as prompts, during training. These prompts are adaptable to specific tasks, enabling continual learning without the necessity of data replay. However, our analysis identifies several inherent weaknesses in these prompt-based approaches. Firstly, they lack robust mechanisms to prevent forgetting in shared components, such as the shared Prompt Pool (Wang et al. 2022b), the task-agnostic General Prompt (G-Prompt) (Wang et al. 2022a), or the shared MLP classifier, which can lead to potential performance degradation. Secondly, task-specific prompt-based approaches (Wang et al. 2022a, 2023a) are prone to inaccuracies in prompt selection, leading to a mismatch between training and testing prompts. Lastly, these methods demonstrate limited optimization in managing both cross-task and within-task variance. For instance, Wang et al. (2022b) employ a common prompt pool, where instances from different tasks may frequently share one or more prompts, thereby reducing cross-task variance. This issue becomes particularly prominent in CRE, where instances from different relation classes might frequently have very similar contexts, as shown in the example below:

"[X] is a professor at [Z university]."   
• "[X] is advised by a professor at [Z university]."

In alignment with these approaches, Le et al. (2024a) explores the relationship between Prefix-tuning (Li and Liang 2021), a widely used technique for implementing prompts,

Our method: Diagram Key   
Representation Generator   
$^ +$ Prompt Pool Learning Trainable   
Representation Mini Batching Generator of past tasks Relation Classifier Query Prompt   
Task K Pool Pk

and Mixture of Experts (MoE) models (Jacobs et al. 1991; Jordan and Jacobs 1994). The study demonstrates that selfattention can be seen as embodying multiple MoE models, and that implementing prefix-tuning is analogous to adding new prefix experts to these pre-trained MoE models to fine-tune their representations. Building on this foundation, we introduce a novel prompting method, WAVE-CRE (Within-Task Variance Awareness for Continual Relation Extraction), designed to address the limitations highlighted earlier, as illustrated in Figure 1. During training, each task is allocated a dedicated Prompt Pool (Wang et al. 2022b), fostering cross-task divergence while capturing intra-task variations. To mitigate catastrophic forgetting during the shared parameter learning process between tasks, we employ a generative model that generates latent data representations for replay. Unlike generating natural language text, learning the underlying distribution and generating continuous latent representations is significantly more feasible. For prompt pool selection, a separate generative model generates uninstructed representations, which are then utilized to train the task predictor. Extensive experiments demonstrate that our method surpasses state-of-the-art prompt-based and rehearsal-free baselines. Therefore, our contributions are as follows:

• We reveal limitations of current prompt-based approaches, including inaccurate prompt selection, inadequate strategies for mitigating forgetting in shared parameters, and suboptimal handling of cross-task and within-task variances.   
• To overcome these challenges, we propose a novel prompting method, WAVE-CRE, which leverages taskspecific prompt pools and generative models for latent representation.   
• Our extensive experimental evaluation demonstrates that WAVE-CRE significantly outperforms state-of-the-art prompt-based and rehearsal-free baselines.

# 2 Background

# 2.1 Continual Relation Extraction

Continual Relation Extraction (CRE) is a subfield of continual learning (Hai et al. 2024; Phan et al. 2022; Van et al. 2022) and continual information extraction (Le et al. 2024c; Dao et al. 2024; Tran et al. 2024b). It involves training a model sequentially on a series of tasks $\{ \mathcal { T } _ { 1 } , \mathcal { T } _ { 2 } , \ldots , \mathcal { T } _ { T } \}$ , with each task $\mathcal { T } _ { t }$ associated with a training dataset $\mathcal { D } _ { t }$ and a corresponding set of relations $\mathcal { R } _ { t }$ . Similar to traditional task $\mathcal { T } _ { t }$ nsists of $\mathcal { N } _ { t }$ labeled sample $\mathcal { D } _ { t } = \{ ( \boldsymbol { x } _ { i } ^ { t } , y _ { i } ^ { t } ) \} _ { i = 1 } ^ { \mathcal { N } _ { t } }$ $\mathbf { \boldsymbol { x } } _ { i } ^ { t }$ $y _ { i } ^ { t }$ bel from the relation set $\mathcal { R } _ { t }$ . The primary objective in CRE is to train a model that can effectively learn from new tasks while preserving its performance on previously acquired tasks. Upon completing the $t$ -th task, the model should be able to accurately identify the relation of an entity pair from the cumulative relation set $\begin{array} { r } { \hat { \mathcal { R } } _ { t } = \bigcup _ { i = 1 } ^ { t } \mathcal { R } _ { i } } \end{array}$ . Most existing methods in CRE rely on the use of  memory buffer to store samples of previously encountered relations, which raises significant concerns regarding memory and privacy.

# 2.2 Mixture of Experts

An MoE model consists of $N$ expert networks, denoted by $f _ { i } : \mathbb { R } ^ { D }  \mathbb { R } ^ { D _ { v } }$ for $i = 1 , \ldots , N$ , and a gating function, $G :$ $\dot { \mathbb { R } } ^ { D } \to \mathbb { R } ^ { N }$ , which dynamically determines the contribution of each expert for a given input $\scriptstyle { \mathbf { { \vec { x } } } }$ . The gating function is based on learned score functions, $s _ { i } : \mathbb { R } ^ { D } \xrightarrow [ ] { } \mathbb { R }$ , associated with each expert, resulting in the following formulation:

$$
\mathbf { y } = \sum _ { j = 1 } ^ { N } G ( \pmb { x } ) _ { j } \cdot f _ { j } ( \pmb { x } ) = \sum _ { j = 1 } ^ { N } \frac { \exp { ( s _ { j } ( \pmb { x } ) ) } } { \sum _ { \ell = 1 } ^ { N } \exp { ( s _ { \ell } ( \pmb { x } ) ) } } \cdot f _ { j } ( \pmb { x } ) ,
$$

where $G ( { \pmb x } ) = \mathrm { s o f t m a x } ( s _ { 1 } ( { \pmb x } ) , \ldots , s _ { N } ( { \pmb x } ) )$ . Shazeer et al. (2017) introduced Sparse Mixture of Experts (SMoE) architecture as an efficient method to scale up MoE models. This is achieved by utilizing a sparse gating function TopK, which selects only the $K$ experts with the largest affinity scores $s _ { j } ( \pmb { x } )$ . The TopK function is defined as:

$$
\begin{array} { r l r } & { \mathrm { T o p K } \left( \pmb { v } , \boldsymbol { K } \right) _ { i } } & \\ & { = \left. \begin{array} { l } { \pmb { v } _ { i } , \quad \mathrm { ~ i f ~ } \pmb { v } _ { i } \mathrm { ~ i s ~ i n ~ t h e ~ } K \mathrm { ~ l a r g e s t ~ e l e m e n t s ~ o f ~ } \pmb { v } } \\ { - \infty , \quad \quad \quad \quad \mathrm { ~ o t h e r w i s e } . } \end{array} \right. } \end{array}
$$

Subsequently, the selected experts independently calculate their outputs, and these are linearly combined using their corresponding affinity scores to produce the final prediction:

$$
\mathbf { y } = \sum _ { j = 1 } ^ { N } \mathrm { s o f t m a x } \left( \mathrm { T o p K } \left( s ( \pmb { x } ) , K \right) \right) _ { j } \cdot f _ { j } ( \pmb { x } ) ,
$$

where $s ( \pmb { x } ) = ( s _ { 1 } ( \pmb { x } ) , \dots , s _ { N } ( \pmb { x } ) )$ . MoE has gained significant attention for its flexibility and adaptability in fields like large language models (Du et al. 2022; Zhou et al. 2023), computer vision (Riquelme et al. 2021), and multitask learning (Ma et al. 2018).

# 2.3 Prompt-based Methods

Recently, parameter-efficient fine-tuning techniques like Prompt-tuning (Lester, Al-Rfou, and Constant 2021) and Prefix-tuning (Liu et al. 2022; Le et al. 2024b) have become prominent for fine-tuning pre-trained models on downstream tasks. This study focuses on prefix-tuning for prompt implementation, where prompts are passed to several Multihead Self-attention (MSA) layers in the pre-trained transformer encoder. Let $\mathbf { X } = \left[ \pmb { x } _ { 1 } ^ { \top } , \ldots , \pmb { x } _ { N } \right] ^ { \top ^ { \bot } } \in \ \mathbb { R } ^ { N \times D }$ represent the input matrix, where $\mathbf { \boldsymbol { x } } _ { i }$ is the embedding of the $i$ -th token, and $D$ is the embedding dimension. The output of an MSA layer is:

$$
\begin{array} { r l } & { \mathrm { M S A } ( \mathbf { X } ) = \mathrm { C o n c a t } ( h _ { 1 } , . . . , h _ { m } ) W _ { O } , } \\ & { h _ { i } = \mathrm { A t t e n t i o n } ( \mathbf { X } W _ { i } ^ { Q } , \mathbf { X } W _ { i } ^ { K } , \mathbf { X } W _ { i } ^ { V } ) , } \end{array}
$$

for $i = 1 , \ldots , m$ , where $W _ { O } , W _ { i } ^ { Q } , W _ { i } ^ { K }$ , and $W _ { i } ^ { V }$ are projection matrices, and $m$ is the number of heads. In prefixtuning, a prompt $\boldsymbol { P } \in \mathbb { R } ^ { L _ { p } \times D }$ of length $L _ { p }$ is divided into $P _ { k } , P _ { v } \in \mathbb R ^ { \frac { L _ { p } } { 2 } \times D }$ . Each head $h _ { i }$ calculation is modified as:

$$
\hat { h } _ { i } = \mathrm { A t t e n t i o n } ( \mathbf { X } W _ { i } ^ { Q } , [ P _ { k } ; \mathbf { X } ] W _ { i } ^ { K } , [ P _ { v } ; \mathbf { X } ] W _ { i } ^ { V } ) ,
$$

where $[ \cdot ; \cdot ]$ denotes the concatenation operation along the sequence length dimension.

Recent research by Le et al. (2024a) has demonstrated that self-attention can be interpreted as a specialized architecture comprising multiple MoE models. The study further suggests that prefix-tuning functions as a mechanism to introduce new experts into these MoE models. Specifically, consider the $l$ -th head in the MSA layer, with output $h _ { l } =$ $[ h _ { l , 1 } , \ldots , h _ { l , N } ] ^ { \top } \ \in \ \mathbb { R } ^ { N \times D _ { v } }$ . Let $\tilde { \mathbf { X } } = [ \pmb { x } _ { 1 } ^ { \top } , \dots , \pmb { x } _ { N } ^ { \top } ] ^ { \top } \in$ $\mathbb { R } ^ { N \cdot D }$ represent the concatenation of all input token embeddings. We define $N$ experts $f _ { j } : \mathbb { R } ^ { N \cdot D }  \bar { \mathbb { R } } ^ { D _ { v } }$ and $N$ gating functions $G _ { i } : \mathbb { R } ^ { N \cdot D }  \mathbb { R } ^ { N }$ with input $\tilde { \mathbf { X } }$ as follows:

$$
\begin{array} { r l } & { f _ { j } ( \tilde { \mathbf { X } } ) = { W _ { l } ^ { V } } ^ { \top } E _ { j } \tilde { \mathbf { X } } = { W _ { l } ^ { V } } ^ { \top } \mathbf { \Phi } \mathbf { x } _ { j } , } \\ & { G _ { i } ( \tilde { \mathbf { X } } ) = \mathrm { s o f t m a x } ( s _ { i , 1 } ( \tilde { \mathbf { X } } ) , \ldots , s _ { i , N } ( \tilde { \mathbf { X } } ) ) , } \\ & { s _ { i , j } ( \tilde { \mathbf { X } } ) = \frac { \tilde { \mathbf { X } } ^ { \top } E _ { i } ^ { \top } W _ { l } ^ { Q } W _ { l } ^ { K } ^ { \top } E _ { j } \tilde { \mathbf { X } } } { \sqrt { D _ { v } } } , } \end{array}
$$

for $i , j = 1 , \dots , N$ , where $E _ { i } \in \mathbb { R } ^ { D \times N \cdot D }$ are matrices such that $E _ { i } \tilde { \mathbf { X } } = \pmb { x } _ { i }$ , and $\begin{array} { r } { D _ { v } = { \frac { D } { m } } } \end{array}$ is the key dimension. Then, from equation (4), the output of the $l$ -th head can be expressed as:

$$
\begin{array} { l } { { \displaystyle h _ { l , i } = \sum _ { j = 1 } ^ { N } G _ { i } ( { \tilde { \bf X } } ) _ { j } \cdot f _ { j } ( { \tilde { \bf X } } ) } } \\ { { \displaystyle \quad = \sum _ { j = 1 } ^ { N } \frac { \exp \Big ( s _ { i , j } ( { \tilde { \bf X } } ) \Big ) } { \sum _ { \ell = 1 } ^ { N } \exp \Big ( s _ { i , \ell } ( { \tilde { \bf X } } ) \Big ) } \cdot f _ { j } ( { \tilde { \bf X } } ) } , } \end{array}
$$

for $i ~ = ~ 1 , \dots , N$ . From equation (8), we observe that each head $h _ { l }$ in the MSA layer includes $N$ MoE models $h _ { l , 1 } , \ldots , h _ { l , N }$ . This structure is similar to the Multi-gate Mixture of Experts (Ma et al. 2018), where multiple MoE models leverage the same set of expert networks but employ independent gating functions. Extending this concept, Le et al. (2024a) proposes that prefix-tuning can be viewed as a method for incorporating new experts into these MoE models. New prefix experts and score functions can be defined as:

$$
f _ { N + j } ( \tilde { \mathbf { X } } ) = { W _ { l } ^ { V } } ^ { \top } \mathbf { p } _ { j } ^ { v } ,
$$

$$
s _ { i , N + j } ( \tilde { \mathbf { X } } ) = \frac { \tilde { \mathbf { X } } ^ { \top } E _ { i } ^ { \top } W _ { l } ^ { Q } W _ { l } ^ { K ^ { \top } } \pmb { p } _ { j } ^ { k } } { \sqrt { D _ { v } } } ,
$$

for $\begin{array} { r c l } { i } & { = } & { 1 , \dots , N } \end{array}$ and $\begin{array} { r c l } { \textit { j } } & { = } & { 1 , \ldots , L } \end{array}$ , where $\begin{array} { r l } { P _ { k } } & { { } = } \end{array}$ $[ \pmb { p } _ { 1 } ^ { k } , \dots , \pmb { p } _ { L } ^ { k } ] ^ { \top }$ , $P _ { v } \ = \ [ \pmb { p } _ { 1 } ^ { v } , \ldots , \pmb { p } _ { L } ^ { v } ] ^ { \top }$ , and $\begin{array} { r } { L \ = \ \frac { L _ { p } } { 2 } } \end{array}$ . From equation (5), the output of the $l$ -th head can be written as $\hat { \boldsymbol { h } _ { l } } = [ \hat { h } _ { l , 1 } , \ldots , \hat { h } _ { l , N } ] ^ { \top } \in \mathbb { R } ^ { N \times D _ { v } }$ , where

$$
\begin{array} { r l } & { \hat { h } _ { l , i } = \displaystyle \sum _ { j = 1 } ^ { N } \frac { \exp \left( s _ { i , j } ( \tilde { \mathbf { X } } ) \right) } { \sum _ { k = 1 } ^ { N + L } \exp \left( s _ { i , k } ( \tilde { \mathbf { X } } ) \right) } f _ { j } ( \tilde { \mathbf { X } } ) } \\ & { \quad \quad + \displaystyle \sum _ { j ^ { \prime } = 1 } ^ { L } \frac { \exp \left( s _ { i , N + j ^ { \prime } } ( \tilde { \mathbf { X } } ) \right) } { \sum _ { k = 1 } ^ { N + L } \exp \left( s _ { i , k } ( \tilde { \mathbf { X } } ) \right) } f _ { N + j ^ { \prime } } ( \tilde { \mathbf { X } } ) , } \end{array}
$$

for $i = 1 , \ldots , N$ . These new experts, $f _ { N + 1 } , \dots , f _ { N + L }$ , collaborate with the pre-trained experts $f _ { 1 } , \ldots , f _ { N }$ to adapt the model for downstream tasks. Several recent methods have effectively integrated prompting techniques with pre-trained transformer encoders, yielding notable results as demonstrated by L2P (Wang et al. 2022b), DualPrompt (Wang et al. 2022a), and HiDe-Prompt (Wang et al. 2023a).

# 3 Methodology

Our overall approach, depicted in Figure 1, involves three main stages: (1) Prompt pool learning for a new task; (2) Generative models; and (3) Training the task predictor and relation classifier.

# 3.1 Task-specific Prompt Pool

In our approach to CRE, we adopt a frozen BERT model (Devlin et al. 2019) as the pre-trained transformer encoder, maintaining consistency with prior studies (Xia et al. 2023; Zhao, Cui, and Hu 2023).

Previous approaches, such as HiDe-Prompt (Wang et al. 2023a), utilize a single prompt per task. This strategy can be compared to utilizing a fixed set of experts for every instance within a given task. However, as detailed in equation (6) and equation (9), the prefix experts encoded in these prompts are considerably simpler, functioning as offset vectors rather than pre-trained experts, which are linear functions of the input. This inherent simplicity implies that a fixed set of prefix experts may lack the necessary flexibility to effectively capture the full range of task variations. To address this limitation, we extend the concept of the prompt pool introduced in L2P (Wang et al. 2022b) by proposing a task-specific prompt pool. For each task $t$ , we introduce a prompt pool $\mathbf { P } _ { t }$ :

$$
\mathbf { P } _ { t } = \{ ( \pmb { k } _ { 1 } ^ { ( t ) } , P _ { 1 } ^ { ( t ) } ) , ( \pmb { k } _ { 2 } ^ { ( t ) } , P _ { 2 } ^ { ( t ) } ) , . . . , ( \pmb { k } _ { M } ^ { ( t ) } , P _ { M } ^ { ( t ) } ) \} ,
$$

where $M$ represents the number of prompts. Each prompt $P _ { i } ^ { ( t ) }$ is associated with a learnable key $\pmb { k } _ { i } ^ { ( t ) } ~ \in ~ \mathbb { R } ^ { D }$ . To facilitate prompt selection, we adopt the same key-query mechanism described in L2P. Specifically, given an input sentence ${ \pmb x }$ , it is first encoded using a pre-trained BERT model to generate a query vector $q ( { \pmb x } )$ . A scoring function $\gamma : \mathbb { R } ^ { D } \times \mathbf { \bar { R } } ^ { D } \to \mathbb { R }$ then evaluates the match between the query vector and each prompt key (e.g., using cosine similarity). The top $K$ most relevant prompts are selected by optimizing the following objective:

![](images/5b27890a4d0ca3c5d5b0f6524034499cf2405860ab4cc71389642b5b7083159e.jpg)  
Figure 2: Data Flow Diagram: Initially, the task predictor predicts the task identity of the input $\scriptstyle { \pmb x }$ , enabling the selection of the corresponding prompt pool. Subsequently, the input $\scriptstyle { \mathbf { { \vec { x } } } }$ queries this prompt pool to identify prompts whose corresponding keys are closest to the query $q ( { \pmb x } )$ . The chosen prompt is then prepended to the embedded input $\pmb { x } _ { e }$ , creating the prompted input $\scriptstyle { \pmb { x } } _ { p }$ . The combined $\scriptstyle { \pmb { x } } _ { p }$ is fed into the BERT Encoder, where the two embeddings corresponding to the positions of the entities $E _ { 1 }$ and $E _ { 2 }$ are concatenated. Finally, the resulting concatenated embedding is passed to the relation classifier, which predicts the relation label $y$ of the input $\scriptstyle { \mathbf { { \vec { x } } } }$ .

$$
K _ { x } = \underset { S \subseteq \{ 1 , \dots , M \} : | S | = K } { \mathrm { a r g m i n } } \sum _ { s \in S } \gamma ( q ( \pmb { x } ) , \pmb { k } _ { s } ^ { ( t ) } ) ,
$$

where $K _ { x }$ denotes a subset of the top- $K$ keys that are specifically chosen for $\scriptstyle { \mathbf { { \vec { x } } } }$ .

We strategically set the number of experts within a prompt to one, with $L _ { p } \ = \ 2$ , resulting in $\begin{array} { r } { L = \frac { L _ { p } } { 2 } = 1 } \end{array}$ Lp = 1. Using prompts of a larger length would be akin to incorporating more experts per prompt, all of which would share a common prompt key within the prompt pool. However, our approach not only reduces memory costs but also enhances flexibility in selecting experts during training and testing. This configuration allows each expert to adapt to different inputs, providing a more versatile assignment compared to methods that use multiple experts per prompt.

Our proposed architecture enables the assignment of different sets of prompts or experts to specific regions of the input data, guided by the contextual query feature $q ( { \pmb x } )$ . This design allows each prompt within the pool to selectively focus on the relevant information and patterns necessary for optimal performance in distinct areas of the input domain. As a result, the model is capable of capturing within-task variations effectively. Furthermore, by utilizing a task-specific prompt pool, we reduce the need for parameter sharing, which helps maximize cross-task variance.

Relationship with Sparse Mixture of Experts. Our proposed task-specific prompt pool shares certain similarities with the SMoE architecture in Section 2.2. Specifically, from equation (12), we denote the experts encoded in the prompt pool $\mathbf { P } _ { t }$ as $f _ { N + 1 } ^ { ( t ) } , \ldots , f _ { N + M } ^ { ( t ) }$ f (Nt)+M . Unlike the pre-trained experts $f _ { 1 } , \ldots , f _ { N }$ , which are selected by default, we employ sparse selection exclusively for these newly introduced prefix experts. As illustrated in equation (11), each head in the MSA layer when applying prefix-tuning encompasses $N$ MoE models $\hat { h } _ { l , 1 } , \dots , \hat { h } _ { l , N }$ . The standard approach involves applying the TopK function to each of these $N$ models individually, necessitating the computation of all $N \times M$ score functions $s _ { i , N + j } ( \tilde { \mathbf { X } } )$ for $i = 1 , \ldots , N$ and $j = 1 , \dots , M$ . This results in a distinct set of prefix experts selected for each model. Conversely, our strategy leverages the same set of $K$ new experts across all $N$ MoE models using auxiliary score functions defined as:

$$
\begin{array} { r } { \hat { s } _ { i , N + j } ( \tilde { \mathbf { X } } ) = \gamma ( q ( \pmb { x } ) , \pmb { k } _ { j } ^ { ( t ) } ) , } \end{array}
$$

where $i = 1 , \ldots , N$ and $j = 1 , \dots , M$ . This approach only requires the computation of $M$ score functions, as the computation of $\hat { s } _ { i , N + j } ( \tilde { \mathbf { X } } )$ only depends solely on $k _ { j _ { - } } ^ { ( t ) }$ , thereby enabling the efficient and effective selection of $K$ experts from the prompt pool. Although the computation of $q ( { \pmb x } )$ might appear to be an added expense, this value is already calculated for the task predictor in Section 3.3 during both the training and testing phases. Therefore, the computation of $q ( { \pmb x } )$ can be reused, incurring no additional cost in the prompt selection process.

Optimization Objective. For each new task $\mathcal { T } _ { t }$ , a new prompt pool $\mathbf { P } _ { t }$ is created. During each training step, following the aforementioned strategy, $K$ prompts are selected, and the corresponding prompted embedding feature, denoted as $\scriptstyle { \pmb { x } } _ { p }$ , is inputted to the pre-trained transformer encoder $f _ { r }$ and the final classifier $g _ { \phi }$ , which is parameterized by $\phi$ . The objective can be summarized as follows:

$$
\operatorname* { m i n } _ { \mathbf { P } _ { t } , \phi } \mathcal { L } \big ( g _ { \phi } ( f _ { r } ( x _ { p } ) ) , y \big ) + \lambda \sum _ { s _ { i } \in K _ { x } } \gamma ( q ( x ) , k _ { s _ { i } } ^ { ( t ) } ) ,
$$

where $K _ { x }$ is obtained with equation (13), $\gamma$ denotes the cosine similarity function, and $\lambda$ is a hyperparameter. The first term in equation (15) employs the classification loss, while the second term minimizes the distance between prompt keys and the corresponding query features. Note that only the prompt parameters in $\mathbf { P } _ { t }$ and the final classifier $g _ { \phi }$ are learned during training task $t$ . The pre-trained BERT model and previous pools $\mathbf { P } _ { 1 } , \ldots , \mathbf { P } _ { t - 1 }$ remain frozen.

# 3.2 Generative Models for Relation Representation

To effectively retain knowledge acquired from prior tasks, we utilize a generative model that captures the distributions of observed relations, enabling the replay of relation samples. For each relation $\boldsymbol { r } \in \hat { \mathcal { R } } _ { t }$ , we maintain a distribution $\mathbf { \bar { G } } _ { z } ^ { r } \sim \mathcal { N } ( \pmb { \mu } _ { z } ^ { r } , \pmb { \Sigma } _ { z } ^ { r } )$ . This distribution is obtained by fitting a Gaussian distribution to the set ${ \mathcal D } _ { z } ^ { r } = \{ z ^ { r } = \dot { f } _ { r } ( { \bf x } _ { p } ^ { r } ) \}$ , which represents the prompted representation of the input $\pmb { x } ^ { r }$ for relation $r$ :

$$
\pmb { \mu } _ { z } ^ { r } = \sum _ { z ^ { r } } \frac { z ^ { r } } { | \mathscr { D } _ { z } ^ { r } | } , \ \Sigma _ { z } ^ { r } = \sum _ { z ^ { r } } \frac { ( z ^ { r } - \pmb { \mu } _ { z } ^ { r } ) ( z ^ { r } - \pmb { \mu } _ { z } ^ { r } ) ^ { \top } } { | \mathscr { D } _ { z } ^ { r } | } .
$$

Similarly, the distribution of the query corresponding to each relation denoted as $\mathcal { D } _ { q } ^ { r } = \{ \pmb q ^ { r } = q ( \pmb x ^ { r } ) \}$ , is stored as ${ \bf G } _ { q } ^ { r } \sim$ $\mathcal { N } ( \mu _ { q } ^ { r } , \Sigma _ { q } ^ { r } )$ :

$$
\pmb { \mu } _ { q } ^ { r } = \sum _ { \pmb { q } ^ { r } } \frac { \pmb { q } ^ { r } } { | \mathscr { D } _ { q } ^ { r } | } , \ \Sigma _ { q } ^ { r } = \sum _ { \pmb { q } ^ { r } } \frac { ( \pmb { q } ^ { r } - \pmb { \mu } _ { q } ^ { r } ) ( \pmb { q } ^ { r } - \pmb { \mu } _ { q } ^ { r } ) ^ { \top } } { | \mathscr { D } _ { q } ^ { r } | } .
$$

This approach ensures that we have a generative model for each relation, capable of reconstructing the corresponding prompted and query representations for all previously observed relations without storing any instance-specific data. The choice of a Gaussian distribution is motivated by its memory efficiency, as it requires storing only the mean vectors and covariance matrices, thereby minimizing the overall memory footprint. Future works may explore alternative generative models.

# 3.3 Task Predictor and Relation Classifier

Each task is associated with its own prompt pool, designed to adapt and learn from the samples specific to the task. However, at test time, it becomes crucial to identify which prompt pool corresponds to a new, unseen sample. To tackle this challenge, we introduce a task predictor, denoted as $\hat { g } _ { \psi }$ . This predictor is a feed-forward MLP with an output dimension matching the total number of relations encountered thus far (i.e., $| \hat { \mathcal { R } } _ { t } | )$ . Once trained, $\hat { g } _ { \psi }$ is capable of predicting the task identity, thereby facilitating the selection of the appropriate prompt pool during testing.

To train the task predictor, we utilize $\mathbf { G } _ { q } ^ { r }$ described in Section 3.2 to generate a representation set $\pmb q ^ { \hat { r } }$ for each relation

Input: Training $t$ -th dataset $\mathcal { D } _ { t }$ , current relation set $\mathcal { R } _ { t }$   
Output: Prompt pool $\mathbf { P } _ { t }$ , task predictor $\psi$ , and relation clas  
sifier $\phi$   
1: Randomly initialize $\mathbf { P } _ { t }$   
2: for $e _ { i d } \gets 1$ to training epoch do   
3: for batch $\pmb { x } _ { B } \in \mathcal { D } _ { t }$ do   
4: Update $\mathbf { P } _ { t }$ and $g _ { \phi }$ on $\scriptstyle { \pmb { x } } _ { B }$ via equation (15)   
5: end for   
6: end for   
7: Update $\hat { \mathcal { R } } _ { t } \gets \hat { \mathcal { R } } _ { t - 1 } \cup \mathcal { R } _ { t }$   
8: for each $\boldsymbol { r } \in \mathcal { R } _ { t }$ do   
9: $\mathcal { D } _ { q } ^ { r }  \emptyset , \mathcal { D } _ { z } ^ { r }  \emptyset$   
10: for batch $\pmb { x } _ { B } \in \mathcal { D } _ { t } ^ { r }$ do   
11: Update $\mathcal { D } _ { q } ^ { r } , \mathcal { D } _ { z } ^ { r }$   
12: end for   
13: Fit $\mathbf { G } _ { z } ^ { r }$ to $\mathcal { D } _ { z } ^ { r }$ via equation (16)   
14: Fit $\mathbf { G } _ { q } ^ { r }$ to $\mathcal { D } _ { q } ^ { r }$ via equation (17)   
15: end for   
16: Train the task predictor $\psi$ via equation (18)   
17: Train the relation classifier $\phi$ via equation (19)   
18: return $\mathbf { P } _ { t } , \phi , \psi$

$\boldsymbol { r } \in \hat { \mathcal { R } } _ { t }$ . The task predictor is trained with the cross-entropy loss function defined as:

$$
\mathcal { L } ( \psi ) = \sum _ { r \in \hat { \mathcal { R } } _ { t } } \sum _ { \pmb { q } \sim \mathbf { G } _ { q } ^ { r } } - \log \frac { \exp ( \hat { g } _ { \psi } ( \pmb { q } ) [ r ] ) } { \sum _ { r ^ { \prime } \in \hat { \mathcal { R } } _ { t } } \exp ( \hat { g } _ { \psi } ( \pmb { q } ) [ r ^ { \prime } ] ) } .
$$

While our approach shares similarities with HiDe-Prompt (Wang et al. 2023a) in utilizing an additional MLP head, a key distinction lies in how relations are treated. HiDePrompt categorizes all relations within a task as a single class in the cross-entropy loss. This strategy can be suboptimal, as the resulting classes may lack semantic significance and their meaning can depend on the sequence in which tasks are presented during training.

In a manner analogous to the training of the task predictor, we train the relation classifier $g _ { \phi }$ using $\mathbf { G } _ { z } ^ { r }$ with the same cross-entropy loss function defined as follows:

$$
\mathcal { L } ( \phi ) = \sum _ { r \in \hat { \mathcal { R } } _ { t } } \sum _ { z \sim \mathbf { G } _ { z } ^ { r } } - \log \frac { \exp ( g _ { \phi } ( z ) [ r ] ) } { \sum _ { r ^ { \prime } \in \hat { \mathcal { R } } _ { t } } \exp ( g _ { \phi } ( z ) [ r ^ { \prime } ] ) } .
$$

This approach helps to mitigate catastrophic forgetting in the shared classification head without requiring the storage of samples from previous relations. For a detailed overview of the training process, please refer to Algorithm 1. The data flow diagram illustrating the inference process is provided in Figure 2.

# 4 Experiments

# 4.1 Experimental Settings

Datasets. To evaluate the effectiveness of WAVE-CRE and the baseline models, we utilize two popular datasets:

• FewRel (Han et al. 2018) contains 80 relation types with a total of 56,000 samples. Following the configurations

<html><body><table><tr><td colspan="10">FewRel</td></tr><tr><td>Model</td><td>T1</td><td>T</td><td>T</td><td>T4</td><td>T5</td><td>T6</td><td>T7 T</td><td>T9</td><td>T10</td></tr><tr><td>EA-EMR</td><td>89.0</td><td>69.0</td><td>59.1</td><td>54.2</td><td>47.8 46.1</td><td>43.1</td><td>40.7</td><td>38.6</td><td>35.2</td></tr><tr><td>RP-CRE</td><td>97.9</td><td>92.7</td><td>91.6</td><td>89.2</td><td>88.4</td><td>86.8 85.1</td><td>84.1</td><td>82.2</td><td>81.5</td></tr><tr><td>CRL</td><td>98.2</td><td>94.6</td><td>92.5</td><td>90.5</td><td>89.4</td><td>87.9 86.9</td><td>85.6</td><td>84.5</td><td>83.1</td></tr><tr><td>CRE-DAS</td><td>98.1</td><td>95.8</td><td>93.6</td><td>91.9</td><td>91.1</td><td>89.4 88.1</td><td>86.9</td><td>85.6</td><td>84.2</td></tr><tr><td>CDec+ACA</td><td>98.4</td><td>95.4</td><td>93.2</td><td>92.1</td><td>91.0</td><td>89.7 88.3</td><td>87.4</td><td>86.4</td><td>84.8</td></tr><tr><td>L2P</td><td>97.4</td><td>90.8</td><td>83.6</td><td>76.5</td><td>68.9</td><td>64.1</td><td>61.0 57.4</td><td>50.1</td><td>44.6</td></tr><tr><td>EPI</td><td>98.3</td><td>89.9</td><td>84.0</td><td>79.9</td><td>76.5</td><td>73.1 70.1</td><td>67.0</td><td>64.5</td><td>61.8</td></tr><tr><td>HiDe-Prompt</td><td>95.5</td><td>89.4</td><td>86.0</td><td>85.7</td><td>87.8</td><td>84.2 75.9</td><td>75.1</td><td>70.3</td><td>67.2</td></tr><tr><td>WAVE-CRE</td><td>97.9</td><td>95.5</td><td>93.6</td><td>92.4</td><td>91.1</td><td>90.2 88.7</td><td>87.6</td><td>86.5</td><td>85.0</td></tr><tr><td colspan="10">TACRED</td></tr><tr><td>Model</td><td>T1</td><td>T</td><td>T3</td><td>T4</td><td>T5</td><td>T6</td><td>T7</td><td>T8</td><td>T9 T10</td></tr><tr><td>EA-EMR</td><td>47.5</td><td>40.1</td><td>38.3</td><td>29.9</td><td>24</td><td>27.3 26.9</td><td>25.8</td><td>22.9</td><td>19.8</td></tr><tr><td>RP-CRE</td><td>97.6</td><td>90.6</td><td>86.1</td><td>82.4</td><td>79.8</td><td>77.2 75.1</td><td>73.7</td><td>72.4</td><td>72.4</td></tr><tr><td>CRL</td><td>97.7</td><td>93.2</td><td>89.8</td><td>84.7</td><td>84.1</td><td>81.3</td><td>80.2 79.1</td><td>79.0</td><td>78.0</td></tr><tr><td>CRE-DAS</td><td>97.7</td><td>94.3</td><td>92.3</td><td>88.4</td><td>86.6</td><td>84.5 82.2</td><td>81.1</td><td>80.1</td><td>79.1</td></tr><tr><td>CDec+ACA</td><td>97.7</td><td>92.8</td><td>91.0</td><td>86.7</td><td>85.2</td><td>82.9 80.8</td><td>80.2</td><td>78.8</td><td>78.6</td></tr><tr><td>L2P</td><td>96.9</td><td>88.2</td><td>73.8</td><td>68.6</td><td>66.3</td><td>63.1</td><td>60.4 59.1</td><td>56.8</td><td>54.8</td></tr><tr><td>EPI</td><td>97.5</td><td>90.7</td><td>82.7</td><td>76.7</td><td>74.0</td><td>72.3 68.2</td><td>66.5</td><td>65.1</td><td>63.4</td></tr><tr><td>HiDe-Prompt</td><td>97.3</td><td>92.8</td><td>86.2</td><td>82.6</td><td>80.6</td><td>80.4</td><td>75.8</td><td>73.7 72.9</td><td>72.6</td></tr><tr><td>WAVE-CRE</td><td>98.4</td><td>94.3</td><td>91.6</td><td>87.8</td><td>85.7</td><td>83.5</td><td>81.3</td><td>80.4 79.5</td><td>78.7</td></tr></table></body></html>

Table 1: Average accuracy $( \% )$ of all methods across learning stages for FewRel and TACRED dataset. The best accuracy score under the rehearsal-free and rehearsal-based setting are in bold and underlined, respectively.   

<html><body><table><tr><td colspan="10">Task-IncrementalLearning-TACRED</td></tr><tr><td>Model</td><td>T1</td><td>T</td><td>T</td><td>T4</td><td>T5</td><td>T6</td><td>T7</td><td>T</td><td>T9</td><td>T10</td></tr><tr><td>WAVE-CRE</td><td>98.4</td><td>95.5</td><td>94.2</td><td>94.1</td><td>92.7</td><td>89.9</td><td>88.3</td><td>87.6</td><td>86.5</td><td>85.2</td></tr><tr><td>w/o Prompt Pool</td><td>96.8</td><td>94.0</td><td>92.9</td><td>91.5</td><td>90.6</td><td>88.0</td><td>86.1</td><td>84.9</td><td>84.4</td><td>83.4</td></tr></table></body></html>

Table 2: Detailed analysis of WAVE-CRE with task-specific prompt pool in the task-incremental learning scenario of TACRED We report the average accuracy across different stages. The best accuracy scores are in bold.

outlined in Wang et al. (2019), we split it into 10 nonoverlapping sub-datasets.

• TACRED (Zhang et al. 2017) consists of 42 relations and 106,264 samples. We adopt the experimental settings proposed by Cui et al. (2021) to partition the dataset into 10 distinct sub-datasets.

Baselines. We compare WAVE-CRE with recent rehearsal-free and prompt-based continual learning methods including L2P (Wang et al. 2022b), HiDe-Prompt (Wang et al. 2023a), and EPI (Wang et al. 2023b). As these methods were originally designed for computer vision, we re-implemented them for CRE using BERT (Devlin et al. 2019) as the encoder. Additionally, we compare our method with rehearsal-based CRE baselines including EA-EMR (Wang et al. 2019), RP-CRE (Cui et al. 2021), CRL (Zhao et al. 2022), CRE-DAS (Zhao, Cui, and Hu 2023), and CDec+ACA (Xia et al. 2023).

Implementation Details. In this work, we used a single NVIDIA A100 for all methods. We tune the hyperparameters for the proposed model using random search. We maintained a consistent size for the prompt pool $M$ across all tasks. For baselines, we follow the identical experimental settings employed by Zhao et al. (2022) to ensure fair comparisons. Our proposed model has in total 114M parameters. Since we froze the BERT model, the number of learnable parameters is thus only 3.8M. Training on the FewRel dataset took approximately 7 hours, while for the TACRED dataset, it took approximately 3 hours using our method.

Evaluation Metrics. We use the same performance measures (mean accuracy on 5 different random seeds) as in prior work (Zhao et al. 2022) for fair comparison.

# 4.2 Main Results

Table 1 summarizes the performance of all methods on FewRel and TACRED datasets. We begin by comparing WAVE-CRE with rehearsal-free and prompt-based methods for CRE. Notably, among all rehearsal-free methods, WAVE-CRE consistently outperforms across different stages of training on both datasets. Particularly on the last task $\mathcal { T } _ { 1 0 }$ , L2P and EPI exhibit substantially lower performance, with a gap of up to $1 5 \%$ in final average accuracy compared to our method. This substantial difference underscores the limitations of these existing approaches in addressing catastrophic forgetting across diverse domains. While HiDe-Prompt shows some improvements, it still experiences performance losses of over $1 5 \%$ and $6 \%$ on FewRel and TACRED, respectively. These losses can be attributed to insufficient task-identity inference techniques, as discussed in Section 4.3.

Furthermore, we evaluate WAVE-CRE against recent suc

<html><body><table><tr><td colspan="10">TaskIncremental Learning-TACRED</td></tr><tr><td>L</td><td>K</td><td>T1</td><td>T</td><td>T</td><td>T4</td><td>T5</td><td>T6</td><td>T7</td><td>T</td><td>T9</td><td>T10</td></tr><tr><td>8</td><td>1</td><td>97.2</td><td>94.8</td><td>93.1</td><td>92.3</td><td>90.5</td><td>87.8</td><td>84.6</td><td>84.1</td><td>83.4</td><td>84.2</td></tr><tr><td>4</td><td>2</td><td>97.7</td><td>95.5</td><td>94.9</td><td>91.2</td><td>90.7</td><td>88.0</td><td>86.5</td><td>86.3</td><td>85.3</td><td>84.1</td></tr><tr><td>2</td><td>4</td><td>96.5</td><td>95.1</td><td>93.2</td><td>92.1</td><td>91.2</td><td>88.9</td><td>85.3</td><td>85.2</td><td>84.6</td><td>84.0</td></tr><tr><td>1</td><td>8</td><td>98.4</td><td>95.5</td><td>94.2</td><td>94.1</td><td>92.7</td><td>89.9</td><td>88.3</td><td>87.6</td><td>86.5</td><td>85.2</td></tr></table></body></html>

Table 3: Detailed analysis of the impact of the number of experts within a prompt. We report the average accuracy across different stages on TACRED in the task incremental learning scenario. The best accuracy scores are in bold.

<html><body><table><tr><td colspan="10">FewRel</td></tr><tr><td>Model</td><td>T1</td><td>T2</td><td>T3</td><td>T4</td><td>T5</td><td>T6</td><td>T7</td><td>T8</td><td>T</td><td>T10</td></tr><tr><td>EPI</td><td>64.5</td><td>62.8</td><td>64.4</td><td>64.9</td><td>64.5</td><td>64.4</td><td>64.4</td><td>59.1</td><td>61.1</td><td>56.6</td></tr><tr><td>HiDe-Prompt</td><td>76.7</td><td>81.7</td><td>80.6</td><td>80.6</td><td>80.2</td><td>78.9</td><td>77.8</td><td>83.1</td><td>80.3</td><td>81.0</td></tr><tr><td>WAVE-CRE</td><td>88.5</td><td>86.2</td><td>86.9</td><td>86.4</td><td>85.2</td><td>86.9</td><td>87.7</td><td>86.0</td><td>85.4</td><td>82.5</td></tr><tr><td colspan="9"></td></tr><tr><td>Model</td><td>T1</td><td>T2</td><td>T</td><td>TACRED T4</td><td>T5</td><td>T6</td><td>T7</td><td>T</td><td>T9</td><td>T10</td></tr><tr><td>EPI</td><td>64.2</td><td>58.8</td><td>66.2</td><td>55.3</td><td>64.0</td><td>62.1</td><td>67.8</td><td>62.0</td><td>62.4</td><td>62.5</td></tr><tr><td>HiDe-Prompt</td><td>70.2</td><td>72.0</td><td>73.8</td><td>75.9</td><td>68.6</td><td>79.8</td><td>75.2</td><td>71.2</td><td>68.5</td><td>64.9</td></tr><tr><td>WAVE-CRE</td><td>83.8</td><td>79.5</td><td>84.2</td><td>76.2</td><td>80.4</td><td>74.8</td><td>79.7</td><td>69.3</td><td>83.7</td><td>81.5</td></tr></table></body></html>

cessful CRE methods, all of which are rehearsal-based. Remarkably, without retaining training data or directly finetuning BERT, WAVE-CRE achieves results nearly equivalent to these state-of-the-art baselines on both datasets. Notably, in the FewRel dataset, our method surpasses the latest rehearsal-based methods on the last task. This highlights the significance and effectiveness of our approach.

# 4.3 Detailed Analysis

Task-specific Prompt Pool. To illustrate the efficacy of task-specific prompt pools in capturing within-task variations compared to a single prompt approach, we conducted experiments in the task incremental learning setting (van de Ven, Tuytelaars, and Tolias 2022). In this setting, task identities are provided, ensuring the prompt pool remains identical during training and testing. Table 2 compares models trained with and without the task-specific prompt pool. Here, “w/o Prompt Pool” represents using only a single taskspecific prompt per task. Our method, WAVE-CRE, trained with the prompt pool, shows a $1 . 8 \%$ improvement on the final task, demonstrating its effectiveness in enhancing the model’s ability to capture within-task variations.

Number of Experts per Prompt $L$ . Similarly, we investigate the effect of different values of $L$ in the task incremental learning scenario. The number of prompts selected, $K$ , was chosen to ensure a fair comparison by keeping the total number of experts across experiments equal. The results are presented in Table 3. As discussed in Section 3.1, setting $L = 1$ allows for flexibility in expert selection and increases the model’s expressiveness, as each expert has its own key. This is empirically shown, as our proposal achieves the best performance among different values of $L$ .

Detailed Analysis of Task Predictor. We evaluate our task-ID prediction technique against the baselines EPI and

HiDe-Prompt, with results summarized in Table 4. EPI uses BERT for task identification and Mahalanobis distance to select task-specific parameters, assuming pre-trained representations are well-separated—an unrealistic assumption given their generic origin, leading to poor prediction accuracy. In contrast, our method trains a dedicated task predictor on synthesized query representations, significantly improving accuracy. HiDe-Prompt adopts a similar strategy and improves over EPI but groups all relations within a task as a single class, which can be suboptimal (Section 3.3). WAVE-CRE addresses this by using a task predictor with output dimensions corresponding to the number of relations, achieving up to $10 \%$ improvements on both datasets, showcasing our strategy’s effectiveness.

# 5 Conclusion

In this work, we propose a novel framework called WAVECRE for rehearsal-free continual relation extraction. Our contributions focus on generating representations for replay, precise task prediction, and optimizing within-task and cross-task variability via prompting. These strategies address limitations of current state-of-the-art prompt-based baselines for continual learning. Through extensive benchmarks, we show our model consistently outperforms existing rehearsal-free methods and achieves competitive results with advanced CRE methods. While our methods mitigate catastrophic forgetting, challenges remain. Retaining knowledge of past tasks is difficult, as seen in prior works. Despite improvements using prompt pools for task-specific knowledge, forgetfulness occurs when pools are not properly utilized during testing. Additionally, while prompt pools enhance expressiveness, the current prefix-tuning experts are relatively simple. Future work could explore more complex expert designs to improve the model.