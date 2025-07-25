# Prototype-Guided Multimodal Relation Extraction based on Entity Attributes

Zefan Zhang 1, Weiqi Zhang 1, Yanhui Li 1, Tian Bai 1\*

1College of Computer Science and Technology, Key Laboratory of Symbolic Computation and Knowledge Engineering, Ministry of Education, Jilin University zefan23@mails.jlu.edu.cn, zwq23@mails.jlu.edu.cn, yanhui23@mails.jlu.edu.cn, baitian@jlu.edu.cn

# Abstract

Multimodal Relation Extraction (MRE) aims to predict relations between head and tail entities based on the context of sentence-image pairs. Most existing MRE methods progressively incorporate textual and visual inputs to dominate the learning process, assuming both contribute significantly to the task. However, the diverse visual appearances and text with ambiguous semantics contain less-informative contexts for the corresponding relation. To tackle these challenges, we highlight the importance of semantically invariant entity attributes that encompass fine-grained categories. Towards this, we propose a novel Prototype-Guided Multimodal Relation Extraction (PG-MRE) framework based on Entity Attributes. Specifically, we first generate detailed entity explanations using Large Language Models (LLMs) to supplement the attribute semantics. Then, the Attribute Prototype Module (APM) refines attribute categories and condenses scattered entity attribute features into cluster-level prototypes. Furthermore, prototype-aligned attribute features guide diverse visual appearance features to produce compact and distinctive multimodal representations in the Relation Prototype Module (RPM). Extensive experiments demonstrate that our method gains superior relation classification capability (especially in scenarios involving various unseen entities), achieving new state-of-the-art performances on MNRE dataset.

# Introduction

Multimodal Relation Extraction (MRE) (Zheng et al. 2021a,b) aims to identify the relation between the given entities based on image and text. It is a fundamental task in the interaction field of computer vision (Wu et al. 2024a,b; Liu et al. 2022; Yao, Li, and Xiao 2024; Yao et al. 2024) and natural language processing (Zhang, Ji, and Liu 2023; Zhang et al. 2024). Multimodal relation extraction aims to enhance computer understanding of textual relations by integrating visual data.

Previous research has primarily focused on identifying relations between entities from textual contexts (Hu et al. 2020b,a, 2021, 2022b). Other studies (Zheng et al. 2021a,b) have emphasized multimodal entity alignment, which involves locating textual entities within visual features to better understand their relations. Despite their remarkable efforts (Chen et al. $2 0 2 2 \mathrm { a }$ ; Wang et al. 2022b; Hu et al. 2023b), current research still faces several challenges:

![](images/35de1e85f850b5cbc539af4cb9dfb42e91cbee42db664baf5fcec1740815037b.jpg)  
Figure 1: (a) shows the incomprehensible text $( 1 7 . 0 4 \% )$ . (b), (c) and (d) describe the high intra-class variation $( 1 2 . 0 8 \% )$ and inter-class similarity $( 4 3 . 0 6 \% )$ in the diverse visual appearance. All ratios are the results we annotated on the MNRE test dataset.

1. Less-informative input contexts. In textual contexts, the semantics of input text is often too obscure. Entities may incorporate novel vocabularies unseen in both the training corpus and the pre-trained language model’s vocabulary, as the “LMX” shown in Figure 1 (a). In visual contexts, entities exhibit diverse visual appearances. As the example shown in Figure 1 (c), images depicting the “peer” relation often feature two or more people. However, there is an interclass similarity in relations, as seen in Figure 1 (b), where “member of” and “peer” relations have similar visual representations. Additionally, there is significant visual variation within the same class, as demonstrated in Figure 1 (c) and (d), which shows a racing scene that differs greatly from typical “peer” scenarios. Both diverse visual appearances and text with ambiguous semantics contain less-informative contexts for the corresponding relation and restrict extracting compact and distinctive multimodal representations. TMR (Zheng et al. 2023a) transfers knowledge in generative diffusion models as back-translation for MRE. CMG (Wu et al. 2023a) proposes to perform topic modeling over the input image and text, incorporating latent multimodal topic features to enrich the contexts. However, both text and image inputs often contain obscure semantics, which limits their performance in knowledge inheriting. Our key intuition is that semantically invariant entity attributes with fine-grained categories are more reliable than visual and textual contexts in multimodal representations.

2. Coarse-granularity attribute categories. Welldefined and comprehensive categories of entity attributes can facilitate more accurate entity relation classification. For example, there is only a single “nationality” relation between “person” and “country”. However, the current dataset has limited categories of entity attributes (including person, organization, localization, and miscellaneous), often using “miscellaneous” to cover a large portion of entity attributes (such as country, religion, movies...). These coarse-granularity attribute categories cannot alleviate the large intra-class variations within some categories which limits the effect of attribute information.

In light of these challenges, we propose a PrototypeGuided Multimodal Relation Extraction (PG-MRE) framework. Specifically, to address the insufficient understanding of entity attributes in the input contexts, we propose to introduce detailed entity explanations by Large Language Models (LLMs). LLMs could provide explanations for entities with ambiguous semantics based on their vast reserve of commonsense knowledge, as the explanation shown in Figure 1 (a). Then, we introduce an Attribute Prototype Module (APM) to refine the existing attribute categories and extract attribute category prototypes from detailed entity explanations for inheriting fine-granularity attributes. APM captures the essential characteristics of the attribute and stores them in a prototype bank. Moreover, during multimodal fusion, attribute features guide the diverse visual features and are fused into compact and distinguishable representations by the Relation Prototype Module (RPM). Specifically, RPM captures the representative embeddings for each relation as relation prototypes and stores them in the corresponding prototype bank. During the inference phase, attribute and relation prototypes will be utilized to aid in classification.

In summary, our contributions are shown as follows:

• We propose a novel Prototype-Guided Multimodal Relation Extraction (PG-MRE) framework based on Entity Attributes to handle more general and flexible scenes. • We introduce the Attribute Prototype Module which could expand the attribute categories and condense the scattered attribute semantics of entities to cluster-level prototypes based on the detailed entity explanations generated by LLMs. The Relation Prototype Module could produce compact and distinctive relation representations based on attribute features.

• We evaluate our methods on the MNRE dataset and demonstrate their superiority and generalization compared to previous state-of-the-art baselines.

# Related Work Multimodal Relation Extraction

As one of the key sub-tasks in the information extraction track (Wang et al. 2022a; Zheng et al. 2024; Lyu et al. 2023), Relation Extraction (RE) (Liu et al. 2019; Chen et al. 2022b) has recently garnered significant attention. Previous studies primarily focus on extracting relations from a single text modality (Hu et al. 2023c, 2022a). Recognizing that visual features from images can provide additional reasoning clues, researchers have proposed Multimodal Relation Extraction (MRE), which has subsequently gained more attention. Recent studies on MRE aim to leverage relevant images to enhance relation extraction.

In the early stages, numerous works (Lu et al. 2018; Moon, Neves, and Carvalho 2018; Zhang et al. 2018) propose encoding text using RNNs and images using CNNs, then establishing implicit interactions between the two modalities. Other works (Yu et al. 2020; Zhang et al. 2021) propose leveraging region-based image features to represent objects in images, exploiting fine-grained semantic correspondences based on Transformer. IFAformer (Li et al. 2023) uses a fine-grained multimodal alignment approach with Transformer, aligning visual and textual objects in the representation space. MoRe (Wang et al. 2022b) proposes retrieving textual evidence from a knowledge base constructed using Wikipedia. However, most methods overlook the issue of interference from irrelevant objects in images.

For multimodal alignment, RpBERT (Sun et al. 2021) learns a text-image similarity score to filter out irrelevant visual representations. HVPNeT (Chen et al. 2022a) proposes a visual prefix-guided fusion mechanism to remove irrelevant objects. For more fine-grained alignment, PROMU (Hu et al. 2023a) proposes entity-object and relation-image alignment pretraining tasks to improve MRE performance. HVFormer (Liu et al. 2024a) utilizes a novel two-stage hierarchical visual context fusion Transformer incorporating the mixture of multimodal experts framework to effectively represent and integrate hierarchical visual features into textual semantic representations. EEGA (Yuan et al. 2023) is an edge-enhanced graph alignment network to enhance the MRE task by aligning nodes and edges in the cross-graph.

Despite continuous progress, these methods overlook the paramount significance of the entity attributes and still face challenges due to the inaccurate semantic understanding of the contexts with ambiguous semantics.

# Prototype Learning

Prototype Learning is a typical few-shot learning approach and has achieved success in recognizing classes with few training examples. Researchers (Chen et al. 2024; Zheng et al. 2023b) propose to model entities/predicates with prototype-aligned compact and distinctive representations and thereby establish matching between entity pairs and predicates in a common embedding space for relation recognition. CRUP (Liu et al. 2024b) stores the mean vectors of representations belonging to new entity types as their prototypes and updates existing prototypes belonging to old types only based on representations of the current batch. The prototypes will be used for the nearest class mean classification. Researchers (Cheng et al. 2023) construct a sentence-wise prototype memory bank, enabling the network to focus on low-level visual and high-level clinical linguistic features.

![](images/35f5087ec8a8be92ef320d4d2f53ce6f2fc59a8bf677e5ae5df8d90f12d6f759.jpg)  
Figure 2: The framework of Prototype-Guided Multimodal Relation Extraction (PG-MRE).

In multimodal relation extraction, more accurate and sufficient entity attributes will provide great assistance for relation extraction. Prototype learning, with its characteristics of summarizing inter-class features (both textual and visual), is very suitable for learning entity attributes and relations in MRE. Therefore, we attempt for the first time to introduce prototype learning into MRE.

# Method

# Overview

Our Prototype-Guided Multimodal Relation Extraction (PGMRE) framework is illustrated in Figure 2. Most previous methods fail to effectively utilize discriminative entity attributes and cannot handle contexts with ambiguous semantics. Our work addresses this problem in Multimodal Relation Extraction (MRE). Given an image-query pair, we first use Large Language Models (LLMs) to generate detailed entity explanations, supplementing the entity semantics. The Attribute Prototype Module then condenses these scattered entity semantics into cluster-level prototypes. Through the Relation Prototype Module, the attribute features and variable visual features are clustered into more compact and distinctive multimodal representations. Within the Attribute

Prototype Module and Relation Prototype Module, we calculate the average features corresponding to each prototype and update the prototype features with momentum in each training iteration. During the test phase, we assign a label to the current features based on the nearest prototypes. These approaches ensure robust and discriminative classification, effectively harnessing the rich semantics and visual cues encapsulated within the prototypes.

# Representation of Visual and Textual Features

Visual Features The image contains several visual objects linked to entities in the text, playing a significant role in aligning multimodal entities. Hence, we utilize object-level visual data provided by other works (Chen et al. 2022a). Additionally, global image features can convey abstract concepts for the whole relation between entities.

We begin by using the visual grounding toolkit to extract local visual objects with the top $m$ salience (Zhang et al. 2021; Yang et al. 2019). Following other works (Chen et al. 2022a), we rescale both the original image and the object images to $2 2 4 \times 2 2 4$ pixels, referring to these as the original images $\nu$ and the object images $\mathcal { O }$ .

As illustrated in Figure 2, for the multimodal relation extraction task, we input the original images $\nu$ and object images $\mathcal { O }$ into the Vision Transformer (Han et al. 2022) for encoding. This process obtains the original image features $F _ { V }$ and the object image features $F _ { O }$ . We define the image feature $F _ { I }$ to represent these uniformly, as shown in Eq.(1):

$$
F _ { I } = \left\{ F _ { V } , F _ { O } \right\} .
$$

Textual Features Since there are often contexts with ambiguous semantics and unseen entities in the training set, the model cannot accurately capture their semantics for relation prediction, such as “Hola (head entity) amigos. Messerschmitt KR (tail entity) 175 bubblecar (1953).” Therefore, we introduce large language models with a robust repository of commonsense knowledge to provide accurate explanations for the head and tail entities, $\mathbf { \bar { \Psi } } C ^ { H } , C ^ { T }$ . We define the explanations $C$ to represent these uniformly, as follows,

$$
{ \cal C } = \{ { \cal C } ^ { H } , { \cal C } ^ { T } \} .
$$

The model can learn global semantics and achieve relation-level alignment with sentence-level features. Additionally, we leverage word-level features to ensure the model selects the correct features between different modalities. Hence, we leverage BERT (Devlin et al. 2018) to encode the input text $T$ and entity explanations $C$ . Then, we could obtain sentence-level features $T _ { S }$ , $C _ { S }$ and word-level features $T _ { W }$ , $C _ { W }$ where the sentence-level features are the pooler output of the word-level features.

# Attribute Prototype Module

Current attribute categories are coarse-granularity and large intra-class variations within the same categories limit the effect of attribute information. Therefore, this module aims to capture the cluster-level attribute prototypes based on detailed entity explanations generated by LLMs.

We first define the attribute prototype bank as follows:

$$
{ \mathcal { P } } ^ { a } = \{ p _ { i } ^ { a } | i = 1 . . . k \} ,
$$

where $k$ is the number of attribute prototypes and each prototype is an embedding vector with $d$ dimension. Then we concatenate the sentence-level head and tail entity explanation features and calculate the distance between textual features and prototypes,

$$
C _ { a l l } = M L P ( C _ { S } ^ { H } \odot C _ { S } ^ { T } ) ,
$$

$$
C _ { a l l } { } ^ { \prime } = p _ { j } ^ { a } + C _ { a l l } ,
$$

$$
j = m i n d i s t ( C _ { a l l } , \mathcal { P } ^ { a } ) ,
$$

where $\odot$ denotes concatenation and $M L P$ is the MLP (Multi-Layer Perceptron) layer. $j$ is the assigned index and $d i s t ( \cdot )$ is the euclidean distance. Here, the prototypes are discrete. The nearest neighbor searching method is not differentiable and can not be optimized with gradient backpropagation. Therefore, we follow (van den Oord, Vinyals, and kavukcuoglu 2017) and adopt their optimizing method by using the stopping gradient strategy. Specifically, it can be optimized as follows:

$$
C _ { a l l } { ' } = s g [ p _ { j } - C _ { a l l } ] + C _ { a l l } ,
$$

where $s g [ \cdot ]$ is the operation that stops the gradient on the parameters. The gradient for previous parameters is transmitted by the auxiliary term $C _ { a l l }$ , which makes the prototype bank trainable. In addition, we used momentum average optimization for prototype clustering. When the prototypes possess a non-zero count of features, the averages of those features are refreshed and serve as the updated prototypes. In general, this Attribute Prototype Module utilizes mini-batch deep learning clustering techniques to aggregate features that share similar attribute semantics.

# Relation Prototype Module

The previous Attribute Prototype Module focuses on extracting textual entity attribute features. In this module, we focus on using the attribute features to guide the visual features and fusing them into compact and distinctive representations.

The relation-level multimodal fusion result encounters significant intra-class variation and severe inter-class similarity in previous studies. Subsequently, we establish multimodal features as relation-centric prototypes based on attribute features and use the key entity attribute features obtained from the Attribute Prototype Module to effectively guide the model to precisely comprehend and represent the intricate relations between entities. Specifically, we first use the cross-attention method to align and fuse entity attribute features with visual features, as follows:

$$
C r o s s A t t \left( \mathcal { Q } , \mathcal { K } , \mathcal { V } \right) = s o f t m a x \left( \frac { Q \mathcal { K } } { \sqrt { d } } \right) \mathcal { V } ,
$$

$$
R _ { V } = C r o s s A t t \left( F _ { V } , C _ { a l l } ^ { \prime } , C _ { a l l } ^ { \prime } \right) ,
$$

$$
{ \cal { R } } _ { O } = C r o s s A t t \left( F _ { O } , C _ { a l l } ^ { \prime } , C _ { a l l } ^ { \prime } \right) ,
$$

$$
R = R _ { V } \odot R _ { O } .
$$

The $R$ denotes the dual-level multimodal fused result. It then transposes to the relation prototypes to capture compact relation-level features.

$$
\begin{array} { c } { { \mathcal { P } ^ { r } = \{ p _ { i } ^ { r } | i = 1 . . . t \} , } } \\ { { R ^ { \prime } = p _ { q } ^ { r } + R , } } \\ { { q = m i n d i s t ( R , \mathcal { P } ^ { r } ) , } } \end{array}
$$

where $t$ is the number of relation prototypes. Finally, we perform residual links with the input visual features.

$$
\begin{array} { c } { { R _ { V } } ^ { \prime } = R _ { V } + R ^ { \prime } , } \\ { { R _ { O } } ^ { \prime } = R _ { O } + R ^ { \prime } . } \end{array}
$$

# Fusion Module

We first perform cross-attention calculations on the relation prototype enhanced visual features ${ \mathit { R } } _ { V } ^ { \prime }$ and ${ { R } _ { O } } ^ { \prime }$ with the word-level text feature $T _ { W }$ as follows:

$$
M _ { V } = C r o s s A t t \left( R _ { V } { } ^ { \prime } , T _ { W } , T _ { W } \right) ,
$$

$$
M _ { O } = C r o s s A t t \left( R o ^ { \prime } , T _ { W } , T _ { W } \right) .
$$

Then, We add the residual link with $M _ { V }$ and $M _ { O }$ . Then fuse these two-level features and obtain $M$ ,

$$
M = M _ { V } + { R _ { V } } ^ { \prime } + M _ { O } + { R _ { O } } ^ { \prime } .
$$

Finally, we concatenate $M$ and sentence-level text feature $T _ { S }$ through the MLP layers $M L P _ { f i n a l }$ and obtain the final fusion feature $F _ { f i n a l }$ :

$$
F _ { f i n a l } = M L P _ { f i n a l } \left( M \odot T _ { S } \right) .
$$

# Classifier

The primary objective of the multimodal relation extraction task is to accurately predict the relation $r$ that exists between the head and tail entities, based on the given set of labels $L$ . To achieve this, we employ a dedicated [CLS] (Classification) head, which serves as a pivotal point for aggregating the probability distribution over the set of relation labels $L$ with the softmax function. Finally, we calculate the RE loss with the cross-entropy loss function:

$$
p ( r | X ) = s o f t m a x ( X ) ,
$$

Table 2: Ablation study on MNRE testing set. Base means baseline, Ent means incorporating the detailed entity explanations. RPM means the Relation Prototype Module, and APM denotes the Attribute Prototype Module.   

<html><body><table><tr><td>Methods</td><td>Accuracy Precision</td><td></td><td>Recall F1</td></tr><tr><td>Text Models</td><td></td><td></td><td></td></tr><tr><td>BERT (2019)</td><td>74.42</td><td>58.58</td><td>60.25 59.40</td></tr><tr><td>PCNN (2015)</td><td>73.15</td><td>62.85</td><td>49.69 55.49</td></tr><tr><td>MTB (2019)</td><td>75.69</td><td>64.46</td><td>57.81 60.86</td></tr><tr><td>Text+Image Models</td><td></td><td></td><td></td></tr><tr><td>MoRe (2022) MEGA (2021)</td><td>79.87 80.05</td><td>65.25</td><td>67.32 66.27 68.44 66.41</td></tr><tr><td>IFAformer (2023)</td><td>92.38</td><td>64.51 82.59</td><td>80.78 81.67</td></tr><tr><td>HVPNeT (2022)</td><td>92.52</td><td>83.64</td><td>80.78 81.85</td></tr><tr><td>TSVFN (2023)</td><td>92.67</td><td>85.16</td><td>82.07 83.02</td></tr><tr><td>MMIB (2024)</td><td></td><td></td><td>82.97 83.23</td></tr><tr><td></td><td>-</td><td>83.49</td><td></td></tr><tr><td>HVFormer (2024) MRE-ISE (2023)</td><td>-</td><td>84.14</td><td>82.65 83.39</td></tr><tr><td></td><td>94.06</td><td>84.69</td><td>83.38 84.03</td></tr><tr><td>MRE (2023)</td><td>93.54</td><td>85.03</td><td>84.25 84.64</td></tr><tr><td>PROMU (2023)</td><td></td><td>84.95</td><td>85.76 84.86</td></tr><tr><td>TMR(2023)</td><td>-</td><td>90.48</td><td>87.66 89.05</td></tr><tr><td>PG-MRE (ours)</td><td>96.34</td><td>92.41</td><td>91.25 91.82</td></tr></table></body></html>

$$
\mathcal { L } r e = - \sum _ { i = 1 } ^ { n } \log \big ( p \left( r | F _ { f i n a l } \right) \big ) .
$$

# Experiments

Table 1: Accuracy $( \% )$ comparison on MNRE testing set.   

<html><body><table><tr><td>Methods</td><td>Accuracy Precision Recall</td><td>F1</td></tr><tr><td>Base</td><td>94.61</td><td>90.08 86.56 88.29</td></tr><tr><td>Base+Ent</td><td>95.85</td><td>91.11 89.69 90.39</td></tr><tr><td>Base+Ent+APM</td><td>95.97</td><td>91.57 90.00 90.78</td></tr><tr><td>Base+Ent+RPM</td><td>95.97</td><td>91.22 90.9491.08</td></tr><tr><td>Base+Ent+RPM+APM</td><td>96.34</td><td>92.41 91.25 91.82</td></tr></table></body></html>

In this section, we conduct comprehensive experiments on the MNRE dataset. Our overarching goal is to answer the following research questions:

• RQ1: How does our model compare with the state-of-theart MRE approaches on quantitative results? • RQ2: How much do various components of our model contribute to its overall performance? • RQ3: How does the proposed model compare against existing MRE methods in some special scenarios? • RQ4: How does the proposed model compare to the stateof-the-art MRE methods on visual results?

![](images/8e7564570a061df70e802a40bb1e08ace6f3bea5a658f17ef9399ba940069c05.jpg)  
Figure 3: Ablation study of the explanations generated by different LLMs.

# Datasets

We evaluate the model on MNRE dataset (Zheng et al. 2021a), which contains 15485 entity pairs, 9,201 text-image pairs, and 23 relation types. Successful extraction of the relation between two entities is achieved when the predicted relation type matches the ground truth. To assess the performance, we employ Accuracy, Precision, Recall, and F1 score as our evaluation metrics.

# Implementation Details

For the text encoder of our model, we leverage the BERTBase default tokenizer with a max length of 128 to preprocess data. For the vision encoder, we leverage the vision Transformer (Han et al. 2022) to encode the original images and object images. The batch size is 16 and the learning rate is 2e-5. We train the model for 8 epochs. Moreover, the dimension of the hidden states $d$ is set to 768. we set the entity explanation length to 40 and uniformly use a bert-baseuncased encoding. All optimizations are performed with the AdamW optimizer with a linear warmup of learning rate over the first $10 \%$ of gradient updates to a maximum value, then linear decay over the remainder of the training. And weight decay on all non-biased parameters is set to 0.01. We set the number of image objects $l$ to 3.

![](images/dfaff4683458b99f9da09b7159d0aa6dc86b8c7932afe7c4950250184148a72d.jpg)  
Figure 4: Ablation study of the different number of prototypes in APM and RPM.

# Baselines

We compare our model with the following baselines for a comprehensive comparison.

BERT (Devlin et al. 2018), PCNN (Zeng et al. 2015) and MTB (Soares et al. 2019) are Text-based RE methods. MoRe (Wang et al. 2022b) injects knowledge-aware information into multimodal studies using multimodal retrieval. MEGA (Zheng et al. 2021a), IFAformer (Li et al. 2023), and MMIB (Cui et al. 2024) are multimodal alignment based models. HVPNeT (Chen et al. 2022a) treats visual representations as visual prefixes that can be inserted to guide textual representations of error-insensitive prediction decisions. TSVFN (Zhao, Gao, and Guo 2023) combines the powerful modeling capabilities of graph neural networks and Transformer networks to fully fuse critical information between visual and textual modalities. MRE (Hu et al. 2023b) uses cross-modal retrieval for obtaining multimodal evidence to improve prediction accuracy and synthesize visual and textual information for relational reasoning. MREISE (Wu et al. 2023b) introduces a novel idea of simultaneous information subtraction and addition for multimodal relation extraction. PROMU (Hu et al. 2023a) enables the extraction of self-supervised signals from massive unlabeled image-caption pairs to pretrain multimodal fusion modules. TMR (Zheng et al. 2023a) implements multimodal versions of back-translation and high-resource bridging, which provide a multi-view to the misalignment between modalities. HVFormer (Liu et al. 2024a) proposes a novel two-stage hierarchical visual context fusion Transformer incorporating the mixture of multimodal experts framework.

# Quantitative Comparison with State-of-the-art Methods (RQ1)

The main results are shown in Table. 1. It could be seen that our model achieves the SOTA results.

Obviously, our method significantly outperforms text-based approaches. In comparison with multimodal alignment-based methods (MEGA (Zheng et al. 2021a), IFAformer (Li et al. 2023), and MMIB (Cui et al. 2024) ), our method also achieves superior performance, which indicates that previous multimodal alignment strategies overlook the entities with ambiguous semantics. The current state-of-the-art (SOTA) model is TMR (Zheng et al. 2023a), which leverages the diffusion strategy to generate a large amount of additional image data to enhance the understanding of relations between entities. However, due to the crucial guiding role of text in multimodal alignment, the neglect of entity attributes still renders its performance inferior to ours.

Table 3: Comparison results between HVPNeT (Chen et al. 2022a), TMR (Zheng et al. 2023a), and our PG-MRE on MNRE US and MNRE UT.   

<html><body><table><tr><td rowspan="2">Methods</td><td>Accuracy (%)</td></tr><tr><td>MNRE_US MNRE_UT</td></tr><tr><td>HVPNet (2022)</td><td>87.43 86.94</td></tr><tr><td>TMR (2023)</td><td>93.71 94.49</td></tr><tr><td>PG-MRE (ours)</td><td>95.81 96.00</td></tr></table></body></html>

Therefore, the results indicate that our method provides more accurate entity semantics and clusters more efficient attribute and relation prototypes. PG-MRE achieves stable and excellent results (Accuracy $+ 2 . 2 8 \%$ , Precision $+ 1 . 9 3 \%$ , Recall $+ 3 . 5 9 \%$ , F $1 + 2 . 7 7 \% ,$ ) which also indicates that our PG-MRE has advantages in identifying diverse visual appearances and contexts with ambiguous semantics.

# Ablation Study (RQ2)

In this section, we conduct extensive experiments with our model to analyze the effectiveness of each component.

Explanations generated by different LLMs In Table 2, comparing the results “Base” and “Base+Ent”, we could find that the detailed entity explanations lead to a steady improvement (Accuracy $+ 1 . 2 4 \%$ , Precision $+ 1 . 0 3 \%$ , Recall $+ 3 . 1 3 \%$ , F1 $+ 2 . 1 \% )$ . Among them, the metric of Recall is improved significantly, which also means that entity explanations play an important role in obscure context understanding. Additionally, we evaluate the generated explanation’s quality and find that only $3 . 3 4 \%$ of the explanations contain noisy tokens, with minimal impact on the model.

In Figure 3, we conduct an ablation study over the entity explanations generated by different Large Language Models (LLMs), such as GPT-3.5, GPT-4o mini, Llama3-70B (Touvron et al. 2023), and Qwen1.5-1.8B (Bai et al. 2023). Good explanations require a vast reserve of commonsense knowledge. The results show that current LLMs can provide sufficient explanations, such as “MLB, or Major League Baseball, is a professional baseball organization...”. Their performances fluctuate within the normal range, and the model with the explanations generated by GPT-4o mini achieves the current best performance.

Attribute Prototype Module In Table 2, comparing the results “Base+Ent” and “Base+Ent+APM”, we could find that the APM can boost model performance. It indicates that the APM could assist the model enriches the attribute cate

Case 1 Case 2 Case 3 Text: $@$ _PLICE: A lot of people don’t know "My Text: @paddypope:When Arsenal go 4th but our Text: @CLS_Rob: Was looking for a title for my @CAPhys Prince" from Mr. Bones is John Diggle in Arrow. next game is City... talk when I spotted this on the @CanLightSource control room. Head : Mr.Bones Head : Arsenal Head : @CAPhys FV 王 五 Attribute: PER Attribute: ORG Attribute: ORG Tail : Arrow Tail : City Tail : @CanLightSource Attribute: ORG Attribute: ORG Attribute: ORG Head Entity Explanation: Mr. Bones is a Head Entity Explanation: The entity "Arsenal" Head Entity Explanation: The entity "@CAPhys" is fictional character who is depicted as ...... refers to the professional football club ...... likely a name given to a person or organization ...... Tail Entity Explanation: Arrow is an American Tail Entity Explanation: ...... Manchester City Tail Entity Explanation: @CanLightSource" crime drama television series set in the ...... (also an English Premier League club) . .... entity represents a specific platform  ...... Gold Relation: per/org/member_of Gold Relation: org/org/subsidiary Gold Relation: /org/org/alternate_names HVPNeT: per/per/peer × per/loc/place_of_residence × None × TMR: per/per/alternate_names × per/loc/place_of_residence × None × PG-MRE: per/org/member_of √ org/org/subsidiary V org/org/alternate_names √

gories, and the cluster-level attribute prototypes facilitate the accurate reasoning process.

In addition, in Figure 4, we conduct an ablation study over the attribute prototype number, and it could be found that 23 is the best number.

Relation Prototype Module In Table 2, comparing the results “Base+Ent”, “Base+Ent+APM” and “Base+Ent $^ +$ RPM”, we could find that the model with RPM has a steady improvement. However, the lack of the APM results in not concise and accurate attribute semantics, thereby interfering with the RPM. Similarly, not incorporating the RPM after the APM leads to interference from related relation features during multimodal fusion. The model achieves the best performance only when both work together. In addition, in Figure 4, we also conduct an ablation study over the relation prototype number. It could be found that 23 is the best number which is the same as the number of relation type.

# Comparison Results in Some Special Scenarios (RQ3)

To test the robustness and the generalization of our model, we experiment with it in some special scenarios.

In the test set of the MNRE dataset, we select samples containing entities not seen in the training set as unseen samples, splitting them as a subset called MNRE US. The results show that $76 . 8 9 \%$ (1241/1614) of the samples in the test set contained entities not seen in the training set. Additionally, we used BERT as the text encoder, but the dataset included words outside BERT’s vocabulary that could not be tokenized. We split test set samples containing these untokenizable entities as a subset called MNRE UT. The results showed that $6 9 . 7 6 \%$ (1126/1614) of the samples in the test set contained untokenizable entities.

We test HVPNeT, TMR, and our PG-MRE on these two data subsets. As shown in the Table 3, the results indicate that PG-MRE consistently improves accuracy (MNRE US $+ 2 . 1 0 \%$ , MNRE UT $+ 1 . 5 1 \%$ ). These experimental results demonstrate that our method maintains stable performance on unseen and untokenizable entities.

# Case Analysis (RQ4)

We conduct a case study of PG-MRE with the current typical works (HVPNeT (Chen et al. 2022a) and TMR (Zheng et al. 2023a)), as shown in Figure 5.

In case 1, HVPNeT (Chen et al. 2022a) and TMR (Zheng et al. 2023a) can not understand the “Arrow”, they may focus on the image for supplement information. The image describes two men, and HVPNeT tends to choose “peer”. However, our PG-MRE provides an accurate explanation of “Arrow” and chooses the right relation.

Similarly, in case 2, the entity of “City” has uncommon semantics. Its real meaning is “Manchester City which is an English Premier League club”. However, TMR and HVPNeT tend to think it is “LOC” and choose the wrong relation without explanations. In case 3, these entities are unseen and untokenizable. Their ambiguous contexts make it difficult to determine the relations between entities directly. After obtaining explanations for key entities, further summarizing and clustering their attribute information based on the critical words (“organization”, “platform”) enables our model to understand and classify the right relations accurately.

# Conclusion

In this work, we propose a Prototype-Guided Multimodal Relation Extraction (PG-MRE) network. We emphasize the significant importance of entity attributes, and we incorporate detailed entity explanations for understanding diverse visual appearances and text with ambiguous semantics. Then, we utilize the Attribute Prototype Module to expand the attribute categories and condense the scattered attribute semantics of entities to cluster-level prototypes. The Relation Prototype Module is used to produce compact and distinctive entity relation representations based on the attribute features. Extensive experiments show that our model could achieve the state-of-the-art (SOTA) result. In the future, we intend to explore the Retrieval Augmented Generation (RAG) multimodal relation extraction models to provide more real-time entity explanations. We are also interested in adapting our approach to more general scenarios and open-vocabulary scenarios.