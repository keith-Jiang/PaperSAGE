# Multi-Grained Query-Guided Set Prediction Network for Grounded Multimodal Named Entity Recognition

Jielong Tang1, Zhenxing Wang4, Ziyang $\mathbf { G o n g } ^ { 2 }$ , Jianxing $\mathbf { Y } \mathbf { u } ^ { 1 , 5 }$ , Xiangwei $\mathbf { Z } \mathbf { h } \mathbf { u } ^ { 3 }$ , and Jian $\mathbf { Y i n } ^ { 1 * }$

1School of Artificial Intelligence, Sun Yat-sen University, Zhuhai, China 2School of Atmospheric Sciences, Sun Yat-sen University, Zhuhai, China 3School of Electronics and Communication Engineering, Sun Yat-sen University, Guangzhou, China 4State Key Laboratory of Intelligent Game, Institute of Software, Chinese Academy of Sciences, Beijing, China 5Pazhou Lab, Guangzhou, 510330, China {tangjlong3, gongzy23}@mail2.sysu.edu.cn, {issjyin, yujx26, zhuxw666} $@$ mail.sysu.edu.cn, wangzhenxing $@$ iscas.ac.cn

# Abstract

Grounded Multimodal Named Entity Recognition (GMNER) is an emerging information extraction (IE) task, aiming to simultaneously extract entity spans, types, and corresponding visual regions of entities from given sentence-image pairs data. Recent unified methods employing machine reading comprehension or sequence generation-based frameworks show limitations in this difficult task. The former, utilizing human-designed type queries, struggles to differentiate ambiguous entities, such as Jordan (Person) and offWhite x Jordan (Shoes). The latter, following the one-byone decoding order, suffers from exposure bias issues. We maintain that these works misunderstand the relationships of multimodal entities. To tackle these, we propose a novel unified framework named Multi-grained Query-guided Set Prediction Network (MQSPN) to learn appropriate relationships at intra-entity and inter-entity levels. Specifically, MQSPN explicitly aligns textual entities with visual regions by employing a set of learnable queries to strengthen intraentity connections. Based on distinct intra-entity modeling, MQSPN reformulates GMNER as a set prediction, guiding models to establish appropriate inter-entity relationships from a optimal global matching perspective. Additionally, we incorporate a query-guided Fusion Net (QFNet) as a glue network to boost better alignment of two-level relationships. Extensive experiments demonstrate that our approach achieves state-of-the-art performances in widely used benchmarks.

Code — https://github.com/tangjielong928/mqspn

# Introduction

To effectively comprehend and manage vast amounts of multimodal content from social media, recent research (Yu et al. 2023a) proposes a nascent multimodal information extraction task named Grounded Multimodal Named Entity Recognition (GMNER). It aims to extract multimodal entity information, including entity spans, types, and corresponding regions of entities, from image-text pairs. Prior studies decompose the GMNER task into several subtasks such as

Multimodal Named Entity Recognition (Zhang et al. 2021a) and Visual Grounding (Redmon and Farhadi 2018), adopting the pipeline approach to solve it, which leads to serious error propagation. To address this problem, recent research paradigm has transferred to detect span-type-region triplets with unified model by formulating GMNER as machine reading comprehension (MRC) (Jia et al. 2023) or sequence generation (Yu et al. 2023a; Wang et al. 2023).

Despite their remarkable performance, limitations still exist. The MRC-based frameworks utilize human-designed type-specific queries as prior instructions to simultaneously guide entity recognition and entity grounding, struggling to distinguish different ambiguous entities. For example in Figure 1 (a), with the input of multiple fixed person queries like ”Please extract person: People’s name...”, the model incorrectly detects off-White x Jordan (Shoes) as Jordan (Person) and assigns the wrong region originally belonging to other entity (Kevin Durant). On the other hand, sequence generation-based methods suffer from exposure bias. They autoregressively decode span-type-region triples one by one in predefined sequence order, resulting in the prediction of off-White x Jordan region highly sensitive to errors in preceding Kevin Durant detection in Figure 1 (b).

In our perspective, the essence of these errors is the inappropriate modeling of multimodal entity relationships. (1) Manually building specific query for each entity is laborintensive and impractical. Existing MRC-based frameworks overlook the distinctions of intra-entity connection due to their reliance on duplicate and untrainable type queries that cannot distinguish intrinsic features of ambiguous entities. (2) Sequence generation-based methods excessively rely on inter-entity relationships between different multimodal entities elements, where the current output is vulnerable to previous predictions. Therefore, we propose a novel unified framework named MQSPN, mainly consisting of learnable Multi-grained Query Set (MQS) and Multimodal Set Prediction Network (MSP) to focus on modeling appropriate relationships at intra-entity and inter-entity levels.

We maintain that modeling intra-entity connections is foundational; only when a model is capable of distinguishing individual entities can it further model inter-entity re

Human- Framework (a) Multiple Times Inference   
Designed NAN QLuOerCy (Oracle Arena,  [LOC],  None)   
Query QPuEeRry (Kevin Durant, [PER],   
Text Input: PER (Jordan,  [PER],   
Kevin Durant Query   
enters Oracle (b) Inference One by One   
AOreffn-aWwheiaterinxg 11 (Oracle Arena,  [LOC],  None)   
Jordan Exposure (Kevin Durant,  [PER],   
Image Input: Bias (Off-White x Jordan, [SHO], None) (c) One Time Inference G Query 1 (Oracle Arena,  [LOC], None) Query 2 (Kevin Durant,  [PER],   
LQeuaernyaSbelet Query u (Off-White x Jordan, [SHO], $\surd$   
Entity Class: [LOC] - Location [PER] - Person [SHO] - Shoes

MSP to filter this noisy information, thereby boosting better alignment of two-level relationships. Unlike direct fusion methods (Zhang et al. 2021b; Wu et al. 2023), QFNet employs queries as intermediaries to facilitate the separate integration of textual and visual region representations. Our contributions could be summarized as follows:

• We delve into the essence of existing unified GMNER methods’ weaknesses from a new perspective, two-level relationships (intra-entity and inter-entity), and propose a unified framework MQSPN to adaptively learn intraentity relationships and establish inter-entity relationships from global optimal matching view. • To the best of our knowledge, our MSP is the first attempt to apply the set prediction paradigm to the GMNER task. • Extensive experiments on two Twitter benchmarks illustrate that our method outperforms existing state-of-theart (SOTA) methods. The ablation study also validates the effectiveness of each designed module.

lationships effectively. Hence, we first propose MQS to adaptively learn intra-entity connections. Instead of humandesigned queries in the MRC-based framework, MQS adopts a set of learnable queries (Li et al. 2023; Gong et al. 2024) (denoted as entity-grained query) to perform joint learning in span localization, region grounding, and entity classification for different entities, which enforces queries to learn distinguishable features and automatically establish explicit intra-entity connections. However, simply learnable queries are insufficient to detect regions and spans due to the lack of semantics information. To tackle this, we feed a prompt with masked type statement into vanilla BERT to build type-grained queries with type-specific semantics. Finally, each multi-grained query is constructed by integrating a learnable entity-grained query with a type-grained query.

Based on distinct intra-entity modeling, we further apply MSP to explore suitable inter-entity relationships. Different from previous sequence generation-based methods, MSP reformulates GMNER as set predictions (Tan et al. 2021; Shen et al. 2022). As shown in Figure 1 (c), with one-time input of learnable MQS, MSP parallelly predicts a set of multimodal entities in the non-autoregressive manner without the need for a preceding sequence. The training objective of MSP is to find the optimal bipartite matching with minimal global matching cost, which can be efficiently solved by the off-the-shelf Hungarian Algorithm (Kuhn 1955). In this manner, the inference of MSP will not depend on redundant dependencies dictated by a predefined decoding order, thereby guiding models to establish suitable inter-entity relationships from a global matching perspective.

Besides, since directly fusing textual features with irrelevant visual features will impair model performance (Chen et al. 2021), we further propose a QFNet between MQS and

# Related Work

Grounded Multimodal Named Entity Recognition. Previous multimodal named entity recognition (MNER) models (Yu et al. 2020; Chen et al. 2022a; Zhang et al. 2021a) primarily focused on how to utilize visual information to assist textual models in entity extraction. Grounded MNER (GMNER) (Yu et al. 2023a) is proposed to additionally output the bounding box coordinates of named entities within the image, which has great potential in various downstream tasks, such as knowledge base construction (Liu et al. 2019) and QA systems (Yu et al. 2021; Yu, Zha, and Yin 2019; Yu et al. 2023b). The taxonomy of previous GMNER works encompasses two branches, namely pipeline manner and unified manner. Pipeline methods (Li et al. 2024a; Ok et al. 2024) decompose the GMNER task into several subtasks like MNER, entity entailment and entity grounding. To tackle the error propagation issue, unified methods formulate GMNER as an end-to-end machine reading comprehension (Jia et al. 2023, 2022) or sequence generation task (Wang et al. 2023; Yu et al. 2023a; Li et al. 2024b). Different from them, our MQSPN reformulate GMNER as set prediction to learn appropriate intra-entity and inter-entity relationships.

Set Prediction. Set prediction is a well-known machine learning technique where the goal is to predict an unordered set of elements. It is widely used in various applications such as object detection (Carion et al. 2020). In the field of NER, Seq2Set (Tan et al. 2021) first reformulates the nested NER task as set prediction to eliminate order bias from the autoregressive model. Subsequently, PIQN (Shen et al. 2022) associates each entity with a learnable instance query to capture the location and type semantics. Recently, DQSetGen (Chen et al. 2024) extended set prediction to other sequence labeling tasks such as slot filling and part-of-speech tagging. Different from these methods, we introduce the MQSPN, which exploits the set prediction paradigm in a new GMNER area to model the two-level relationship of multimodal entities.

Visual Grounding. Visual Grounding (VG) aims to detect the most relevant visual region based on a natural language query, i.e., phrase, category, or sentence. Most ex

H Class-agnostic Multimodal Set Prediction (MSP) RPN EVnicsoiodner PredicCtiRonMHead Pr 0ediction Set{   0,    2,  [PER], Image Input 1 {   3,    5,  [PER], } DoVtnihakeltidor rTmrOeurebmtiápngbaeanftdore QuMerulytiS-getr a(iMneQdS) F(uQsiFoNneNt)et PredictEioCn Head HAlugnograitrhiamn u-2 { 14,  17,  [LOC], None } eTsrtuatmepi'ns PMalarm-aB-eLacgho, EnTceoxdter SBL u-1 None } 价 Florida... Prediction Head Bipartite Text Input Matching Ground Truth Set   
(b) Multi-grained Query Set (MQS) (c) Query-guided Fusion Net (QFNet)   
Person is an entity type about [MASK]. LEQ ×  u ... ... ...   
LOorgcatniioznatiisoanniseantniteyntiytpyetaybpeouatb[oMutA[SMKA].SK]. 油 Transformer SimAilgagrietgy-atowrare Block Similarity-aware Aggregator Transformer Block Tanh+FC O ↑ 4 4 → ? FC Query Generator Type-grained Y copy ×  d ... Prefix Cross Softmax 华 Type-grained 里 Integration Attention 出 0 Query ① □□□□□ □ Learnable Visual Features Query Features Textual Features Entity-grained Query   
CRM Candidate Regions EC Entity SBL Span Boundary Freezing FC Fully Connected E Element-wise Dot Matching Classification Localization Layer Addition Product

isting works can be divided into two branches. The first branch utilizes the one-stage detector, such as DETR (Carion et al. 2020) and YOLO (Redmon and Farhadi 2018), to directly output object bounding boxes in an end-to-end manner. The second branch first generates candidate regions with some region proposal methods, such as Region Proposal Network (RPN) (Girshick 2015) and selective search (Uijlings et al. 2013), and then selects the best-matching region based on language query. In this work, we follow the two-stage paradigm to construct an entity grounding model.

# Our Method

# Overview

As illustrated in Figure 2, we present a set prediction-based method named MQSPN with four different components, consisting of the Feature Extraction Module, Multi-grained Query Set (MQS), Query-guided Fusion Net (QFNet) and Multimodal Set Prediction Network (MSP). The objective of our MQSPN is to predict a set of multimodal quadruples which can be represented as:

$$
Y = \{ \left( Y _ { 1 } ^ { s } , Y _ { 1 } ^ { e } , Y _ { 1 } ^ { t } , Y _ { 1 } ^ { r } \right) , \dots , \left( Y _ { m } ^ { s } , Y _ { m } ^ { e } , Y _ { m } ^ { t } , Y _ { m } ^ { r } \right) \} ,
$$

where $( Y _ { i } ^ { s } , Y _ { i } ^ { e } , Y _ { i } ^ { t } , Y _ { i } ^ { r } )$ denote the $i$ -th quadruple, ${ Y } _ { i } ^ { s } \in$ $[ 0 , n - 1 ]$ and $\bar { Y } _ { i } ^ { e } \in [ 0 , n - 1 ]$ are the start and end boundary indices of the $i$ -th target entity span. $Y _ { i } ^ { t }$ refers to its corresponding entity type, and $Y _ { i } ^ { r }$ denotes grounded region. Note that if the target entity cannot be grounded in the given image, $Y _ { i } ^ { r }$ is $N o n e$ ; otherwise, $Y _ { i } ^ { r }$ consists of a 4-D spatial feature including the top-left and bottom-right coordinates of the grounded region.

# Feature Extraction Module

Text Representation. Given the input sentence $X$ , textual encoder BERT (Devlin et al. 2019) is used to tokenize it into a sequence of word embeddings $\begin{array} { r l } { H _ { T } } & { { } = } \end{array}$ $( [ C L S ] , e _ { 1 } , . . . , e _ { n } , [ \bar { S E P } ] )$ , where $\boldsymbol { e } _ { i } \in \mathbb { R } ^ { h }$ , $h$ is the hidden dimension, $\left[ C L S \right]$ and $[ S E P ]$ are special tokens of the beginning and end positions in word embeddings.

Visual Representation. Given the input image $I$ , we utilize $\mathrm { V i n V L }$ (Zhang et al. 2021c) as the class-agnostic region proposal network (RPN) to obtain all candidate regions. Following the work of (Yu et al. 2023a), we also retain the top- $k$ region proposals as our candidate regions, using the ViT-B/32 (Dosovitskiy et al. 2020) from pre-training CLIP (Radford et al. 2021) as vision encoder. The initial visual representation for the candidate regions are denoted as $\mathbf { V } = \mathbf { \bar { \{ v } }  _ { 1 } , \ldots . . . , \mathbf { v } _ { k } \}$ . To match those entities that are ungroundable in the images, we construct a special visual token embedding $\mathbf { v } _ { [ u g ] }$ by feeding a blank image $\tau$ into vision encoder. Finally, $\mathbf { v } _ { [ u g ] }$ and $\mathbf { V }$ are concatenated to serve as the final visual representation $H _ { V } \in \mathbb { R } ^ { ( k + 1 ) \times h }$ , where $k$ is the number of candidate regions.

# Multi-grained Query Set Construction

Previous manually constructed query statements struggle to learn distinguishable features for different entities, hindering intra-entity connections modeling. In this section, we propose a learnable multi-grained query set to overcome this.

Prompt-based Type-grained Query Generator. The Entity type can provide effective information for entity span extraction and candidate region matching. We designed a prompt template: P rompt $= [ T Y P E ]$ is an entity type about $[ M A { \bar { S } } K ]$ , where $[ T Y P { \bar { E } } ]$ refers to the entity type name, such as Person, Location, Organization, and Others. Then the prompt template is fed into a vanilla BERT model. The type-grained query embedding is calculated as the output embedding of the $[ M A S K ]$ position:

$$
H _ { Q } ^ { o } = B E R T \left( P r o m p t \right) _ { [ \mathrm { M A S K } ] }
$$

where $H _ { Q } ^ { o } \in \mathbb { R } ^ { p \times h }$ is the type-grained query embedding, $p$ denotes the number of entity types.

Learnable Entity-grained Query Set. Entity-grained queries are randomly initialized as learnable input embeddings $H _ { Q } ^ { e } \in \mathbb { R } ^ { u \times h }$ . During training, the entity-grained semantics and corresponding relationships between candidate regions and entity spans can be learned automatically by these embeddings. To ensure that type-grained and entitygrained query embeddings have the same dimensions, we replicated the former $d$ times. Then the multi-grained query embedding is given by:

$$
H _ { Q } = H _ { Q } ^ { e } \oplus \left[ H _ { Q } ^ { o } \right] ^ { d }
$$

where $H _ { Q } \in \mathbb { R } ^ { u \times h }$ refers to the multi-grained queries set, $p < u$ , and $d = u / p$ . $u$ is the number of queries, we use the token-wise addition operation $\oplus$ to fuse the multi-grained queries. $[ \cdot ] ^ { d }$ denotes repeating $d$ times.

# Query-guided Fusion Net

Previous multimodal fusion approaches suffered from direct fusion of textual and visual information due to semantic discrepancies and noisy information between multimodal data. Different from them, we use queries as intermediaries to guide the integration of textual representations and visual region representations respectively. This module includes three interaction mechanisms.

Query-text Cross-attention Interaction. As shown in Figure 2 (b), the query set $H _ { Q }$ and the textual sequence $H _ { T }$ are fed into the transformer-based architecture (Vaswani et al. 2017). The cross-attention mechanism is used to fuse these unimodal features.

Query-region Prefix Integration. Following prefix tuning (Li and Liang 2021) and its successful applications (Chen et al. 2022a,b) in multimodal fusion, we propose query-region prefix integration to reduce the semantic bias between multimodal data. Details of the Query-region Prefix Integration are provided in the Appendix.

Similarity-aware Aggregator. To mitigate the noise caused by misaligned candidate regions and entity spans, we propose a similarity-aware aggregator to learn fine-grained token-wise alignment between query tokens and regional features/textual tokens. We denote the query set, textual, and visual features as $\tilde { H } _ { Q } , \tilde { H } _ { T }$ , and $\tilde { H } _ { V }$ , respectively. We compute the token-wise similarity matrix of the $i$ -th query token

as follows:

$$
\alpha _ { j } ^ { \phi } = \frac { \exp { ( \tilde { H } _ { \phi _ { j } } \cdot \tilde { H } _ { Q } ) } } { \sum _ { j } \exp { ( \tilde { H } _ { \phi _ { j } } \cdot \tilde { H } _ { Q } ) } }
$$

where $\phi \in \{ V , T \}$ represents the visual or textual feature. $\tilde { H } _ { \phi _ { j } }$ refers to the representation of the $j$ -th visual regions or textual tokens. Finally, the fine-grained fusion for integrating similarity-aware visual or textual hidden states into the query hidden states can be represented as:

$$
\mathrm { F } ( \tilde { H } _ { Q } ) = \mathrm { T a n h } ( \tilde { H } _ { Q } W _ { 1 } + \sum _ { \phi } \sum _ { j } \lambda _ { \phi } \alpha _ { j } ^ { \phi } \tilde { H } _ { \phi _ { j } } ) W _ { 2 } + { \bf b }
$$

where $\lambda _ { \phi }$ is the trade-off coefficient to balance the visual similarity and textual similarity $\sum _ { \phi } \lambda _ { \phi } = 1 . \stackrel { \cdot } { V }$ W1 and $W _ { 2 }$ are linear transformations parameters and $\mathbf { b }$ is the bias term.

# Multimodal Set Prediction Network

Instead of relying on the predefined decoding order in previous methods, we propose a Multimodal Set Prediction Network to maintain suitable inter-entity relationships in a global matching perspective.

Span Boundary Localization. Given the output textual representation $\hat { H } _ { T } \in \mathbb { R } ^ { n \times h }$ and the query set representation $\hat { H _ { Q } } \in \mathbb { R } ^ { u \times h }$ from QFNet, we first expand their dimensions as $\mathsf { \bar { H } } _ { T } \in \mathbb { R } ^ { 1 \times n \times h }$ and $\hat { H } _ { Q } \mathrm { ~ \in ~ } \mathbb { R } ^ { u \times 1 \times h }$ , and then integrate them into a joint representation:

$$
\mathrm { S } ^ { \mathrm { b } } = \mathrm { R e L U } ( \hat { H } _ { Q } W _ { Q } ^ { b } + \hat { H } _ { T } W _ { T } ^ { b } )
$$

where ${ \bf S } ^ { \mathrm { b } } \in \mathbb { R } ^ { u \times n \times h }$ is the joint representation for span boundary localization, $W _ { Q } ^ { b }$ and $W _ { T } ^ { b } \in \mathbb { R } ^ { h \times h }$ are learnable parameters. Thus the probability matrix of each textual token being a start index can be calculated as below:

$$
P ^ { s } = \mathrm { s i g m o i d } ( \mathrm { S } _ { b } W _ { s } ^ { b } )
$$

where $P ^ { s } \in \mathbb { R } ^ { u \times n }$ , $W _ { s } ^ { b }$ is learnable parameter. Similarly, we can simply replace $W _ { s } ^ { b }$ with a new parameter $W _ { e } ^ { b }$ to obtain the probability matrix of the end index $P ^ { e } \in \mathbb { R } ^ { u \times n }$ .

Candidate Regions Matching. Given the output visual region representation $\hat { H } _ { V } ~ \in ~ \mathbb { R } ^ { ( k + 1 ) \times h }$ and the query set representation $\hat { H } _ { Q } \in \mathbb { R } ^ { u \times h }$ , the candidate regions matching task is quite similar to boundary localization, which also uses a query set to predict the corresponding visual index in the candidate regions proposed by the class-agnostic RPN. The process can be formalized as follows:

$$
\begin{array} { r l } & { \mathrm { S } ^ { \mathrm { r } } = \mathrm { R e L U } ( \hat { H } _ { Q } W _ { Q } ^ { r } + \hat { H } _ { V } W _ { V } ^ { r } ) , } \\ & { P ^ { r } = \mathrm { s i g m o i d } ( \mathrm { S } ^ { \mathrm { r } } W _ { s } ^ { r } ) } \end{array}
$$

where $P ^ { r } \in \mathbb { R } ^ { u \times ( k + 1 ) }$ represents the matching probability matrix. $W _ { s } ^ { r } , W _ { Q } ^ { r }$ and $\boldsymbol { W _ { V } ^ { r } }$ are learnable parameters.

Entity Classification. Since the query set has been endowed with type-level information during its construction, the entity classification task can be regarded as an existence detection for type-grained queries. Besides, considering that boundary localization and region-matching information are useful for query existence detection, we concatenate them with queries. The representation of query existence detection can be formalized as:

Table 1: Performance comparison of different competitive approaches on Twitter-GMNER datasets. Bold represents the optimal result, and underlined represents the suboptimal result. For the baseline methods, $\clubsuit$ indicates that the results are reproduced according to the corresponding papers, and others are from (Yu et al. 2023a).   

<html><body><table><tr><td rowspan="2">Category</td><td rowspan="2">Methods</td><td colspan="3">GMNER</td><td colspan="3">MNER</td><td colspan="3">EEG</td></tr><tr><td>Pre.</td><td>Rec.</td><td>F1</td><td>Pre.</td><td>Rec.</td><td>F1</td><td>Pre.</td><td>Rec.</td><td>F1</td></tr><tr><td rowspan="6">Pipeline Methods</td><td>GVATT-RCNN-EVG</td><td>49.36</td><td>47.80</td><td>48.57</td><td>78.21</td><td>74.39</td><td>76.26</td><td>54.19</td><td>52.48</td><td>53.32</td></tr><tr><td>UMT-RCNN-EVG</td><td>49.16</td><td>51.48</td><td>50.29</td><td>77.89</td><td>79.28</td><td>78.58</td><td>53.55</td><td>56.08</td><td>54.78</td></tr><tr><td>UMT-VinVL-EVG</td><td>50.15</td><td>52.52</td><td>51.31</td><td>77.89</td><td>79.28</td><td>78.58</td><td>54.35</td><td>56.91</td><td>55.60</td></tr><tr><td>UMGF-VinVL-EVG</td><td>51.62</td><td>51.72</td><td>51.67</td><td>79.02</td><td>78.64</td><td>78.83</td><td>55.68</td><td>55.80</td><td>55.74</td></tr><tr><td>ITA-VinVL-EVG</td><td>52.37</td><td>50.77</td><td>51.56</td><td>80.40</td><td>78.37</td><td>79.37</td><td>56.57</td><td>54.84</td><td>55.69</td></tr><tr><td>BARTMNER-VinVL-EVG</td><td>52.47</td><td>52.43</td><td>52.45</td><td>80.65</td><td>80.14</td><td>80.39</td><td>55.68</td><td>55.63</td><td>55.66</td></tr><tr><td rowspan="4">Unified Methods</td><td>MNER-QG$</td><td>53.02</td><td>54.84</td><td>53.91</td><td>78.16</td><td>78.59</td><td>78.37</td><td>58.48</td><td>56.59</td><td>57.52</td></tr><tr><td>H-Index</td><td>56.16</td><td>56.67</td><td>56.41</td><td>79.37</td><td>80.10</td><td>79.73</td><td>60.90</td><td>61.46</td><td>61.18</td></tr><tr><td>TIGER+</td><td>55.84</td><td>57.45</td><td>56.63</td><td>79.88</td><td>80.70</td><td>80.28</td><td>60.72</td><td>61.81</td><td>61.26</td></tr><tr><td>MQSPN (Ours)</td><td>59.03</td><td>58.49</td><td>58.76</td><td>81.22</td><td>79.66</td><td>80.43</td><td>61.86</td><td>62.94</td><td>62.40</td></tr></table></body></html>

$$
\begin{array} { r } { \mathrm { S } ^ { \mathrm { c } } = \mathrm { R e L U } \left( \left[ \hat { H } _ { Q } W _ { Q } ^ { c } ; P ^ { s } \hat { H } _ { T } ; P ^ { e } \hat { H } _ { T } ; P ^ { r } \hat { H } _ { V } \right] \right) } \end{array}
$$

$$
P ^ { c } = \mathrm { s i g m o i d } ( \mathrm { S } ^ { c } W _ { s } ^ { c } )
$$

where $P ^ { c } \in \mathbb { R } ^ { u }$ denotes the existence probability of typegrained query. $W _ { Q } ^ { c }$ and $W _ { s } ^ { c }$ are learnable parameters. Finally, the multimodal entity quadruple predicted by the $i$ -th query can be represented as $\bar { \hat { Y } _ { i } } = ( \bar { \hat { Y } _ { i } ^ { s } } , \bar { \hat { Y } _ { i } ^ { e } } , \hat { Y } _ { i } ^ { t } , \hat { Y } _ { i } ^ { r } )$ , and the predicted quadruple set is $\hat { Y } = \{ \hat { Y } _ { i } \} _ { i = 1 } ^ { u }$ .

Bipartite Matching Loss. Following previous works (Shen et al. 2022; Tan et al. 2021; Chen et al. 2024), we construct a loss function based on optimal bipartite matching. However, since the number of queries $u$ is greater than the total quantity of gold samples $m$ , we define a label $\varnothing$ for those unassignable predictions and then add the label $\varnothing$ to the gold set $Y$ , replicating $Y$ until it reaches the size of $u$ . Our training objective is to find the optimal assignment $\pi$ that minimizes the global cost. Searching for a permutation of $u$ elements $\pi \in \mathbb { O } _ { u }$ can be formalized as follows:

$$
\begin{array} { l } { { \displaystyle { \hat { \pi } = \arg \operatorname* { m i n } _ { \pi \in \mathbb { Q } _ { u } } \sum _ { i } ^ { u } \mathcal { L } _ { \mathrm { c o s t } } \left( Y _ { i } , \hat { Y } _ { \pi ( i ) } \right) } , } } \\ { { \medskip \mathcal { L } _ { \mathrm { c o s t } } \left( Y _ { i } , \hat { Y } _ { \pi ( i ) } \right) = - 1 _ { \left\{ Y _ { i } ^ { t } \neq \infty \right\} } \left[ p _ { \pi ( i ) } ^ { c } \left( Y _ { i } ^ { t } \right) \right. } } \\ { { \displaystyle \left. + p _ { \pi ( i ) } ^ { s } \left( Y _ { i } ^ { s } \right) + p _ { \pi ( i ) } ^ { e } \left( Y _ { i } ^ { e } \right) + p _ { \pi ( i ) } ^ { r } \left( Y _ { i } ^ { r } \right) \right] } . }  \end{array}
$$

where $\mathcal { L } _ { \mathrm { c o s t } } ( \cdot )$ denotes the pair matching cost between the gold quadruple $Y _ { i }$ and a prediction with index $\pi ( i ) . 1 _ { \{ \delta \} }$ represents the indicator function that takes 1 when $\delta$ is true and 0 otherwise. The optimal assignment $\pi$ can then be efficiently solved by the off-the-shelf Hungarian Algorithm (Kuhn 1955). To jointly train the model, the final loss function is defined as follows:

$$
\begin{array} { r l } { \mathcal { L } ( Y , \hat { Y } ) = \displaystyle \sum _ { i = 1 } ^ { u } \Big \{ - \log p _ { \hat { \pi } ( i ) } ^ { c } ( Y _ { i } ^ { t } ) } & { } \\ { + 1 _ { \{ Y _ { i } ^ { t } \neq \mathcal { B } \} } \left[ - \log p _ { \hat { \pi } ( i ) } ^ { s } ( Y _ { i } ^ { s } ) \right. } & { } \\ { \left. - \log p _ { \hat { \pi } ( i ) } ^ { e } ( Y _ { i } ^ { e } ) - \log p _ { \hat { \pi } ( i ) } ^ { r } ( Y _ { i } ^ { r } ) \right] \Big \} . } \end{array}
$$

Table 2: F1 scores of previous unified methods and our MQSPN on Twitter-FMNERG datasets for each fine-grained entity type in GMNER task.   

<html><body><table><tr><td>Entity Type</td><td>H-Index</td><td>TIGER</td><td>MQSPN</td></tr><tr><td>Person</td><td>45.13</td><td>43.78</td><td>48.64</td></tr><tr><td>Location</td><td>62.33</td><td>67.69</td><td>71.03</td></tr><tr><td>Building</td><td>32.88</td><td>40.00</td><td>44.78</td></tr><tr><td>Organization</td><td>46.68</td><td>46.75</td><td>49.03</td></tr><tr><td>Product</td><td>28.19</td><td>27.38</td><td>29.46</td></tr><tr><td>Art</td><td>38.89</td><td>43.27</td><td>42.84</td></tr><tr><td>Event</td><td>45.56</td><td>48.39</td><td>51.88</td></tr><tr><td>Other</td><td>41.81</td><td>48.28</td><td>43.49</td></tr><tr><td>Au</td><td>46.55</td><td>47.20</td><td>48.57</td></tr></table></body></html>

# Experiments

# Experiment Settings

GMNER Datasets. We conduct experiments on the TwitterGMNER (Yu et al. 2023a) and Twitter-FMNERG (Wang et al. 2023). Please refer to Appendix for detailed information about these datasets and evaluation metrics.

Implementation Details All experiments are implemented on 4 NVIDIA RTX3090 GPUs with Pytorch 1.9.1. For a fair comparison, we use the pre-trained BERT-based model1 as the textual encoder, ViT-B/32 from pre-training $\mathrm { C L I P } ^ { 2 }$ as the visual encoder, and ${ \mathrm { V i n V L } } ^ { 3 }$ as a class-agnostic RPN. For model’s efficiency, we freeze the visual encoder and RPN and assign the layer of the QFNet as $L = 3$ . We initialize the learnable query part with a normal distribution $\mathcal { N } ( 0 , 0 . 0 2 )$ . For training, we set the batch size to 16, the learning rate to $2 \times 1 0 ^ { - 5 }$ , and the training epoch to 50. Our model uses an Adam optimizer with a linear warmup of ratio 0.05. To allow the multi-grained queries to learn query semantics initially, we first train the model for 5 epochs with the freezing parameters of the pre-training model. Please refer to Appendix for baseline methods.

Table 3: Ablation study of each component on overall F1 score of Twitter-GMNER and Twitter-FMNERG.   

<html><body><table><tr><td>Module</td><td>Settings</td><td>GMNER</td><td>FMNERG</td></tr><tr><td>MQSPN</td><td></td><td>58.76</td><td>48.57</td></tr><tr><td>MQS</td><td>w/o PTQ w/o LEQ</td><td>56.95 (↓1.81) 56.02 (↓2.74)</td><td>47.23 (↓1.34) 45.39(↓3.18)</td></tr><tr><td>QFNet</td><td>w/o QCT w/o QPI W/o SAG</td><td>57.21 (↓1.55) 57.86 (↓0.90) 58.13(↓0.63)</td><td>47.61(↓0.96) 48.33(↓0.24) 47.96(↓0.61)</td></tr><tr><td>MSP</td><td>w/o BML</td><td>56.58 (↓2.18)</td><td>46.81(↓1.76)</td></tr></table></body></html>

# Overall Performance

Performance on GMNER, MNER, and EEG. Following (Yu et al. 2023a), we also report two subtasks of GMNER, i.e., Multimodal Named Entity Recognition (MNER) and Entity Extraction & Grounding (EEG). MNER aims to identify entity span and entity type pairs, while EEG aims to extract entity span and entity region pairs. Table 1 shows the performance comparison of our method with competitive baseline models on Twitter-GMNER benchmarks.

First, unified models are significantly superior to pipeline methods due to the joint training of MNER and EEG to mitigate error propagation. Second, MQSPN significantly outperforms the MRC-based method MNER-QG by $+ \dot { 4 } . 8 5 \%$ F1 scores. Furthermore, compared with the previous stateof-the-art (SOTA) generative model TIGER, our method MQSPN exhibits superior performance, achieving $+ 0 . 1 5 \%$ , $+ 1 . 1 4 \%$ , and $+ 2 . 1 3 \%$ F1 scores improvements on the MNER, EEG, and GMNER tasks, respectively.

We attribute the performance improvements of MQSPN to the following reasons: (1) Compared with the sequence generation-based approach H-Index and TIGER, MQSPN eliminates the dependencies on predefined decoding order and establishes suitable inter-entity relationships from a global matching view. (2) Compared with the MRC-based approach MNER-QG, MQSPN can learn fine-grained entity semantics and model distinct intra-entity connections between regions and entities.

Performance on fine-grained GMNER. We validate the fine-grained GMNER ability of different methods on Twitter-FMNERG datasets, and the experimental results are shown in Table 2. It reveals that our proposed MQSPN achieves the best results on all fine-grained entity types except Other and $A r t$ among the state-of-the-art models. Specifically, we achieve a great improvement of $+ 3 . 5 1 \%$ , $\dot { + } 3 . 3 4 \%$ , $+ 4 . 7 8 \%$ , $+ 2 . 2 8 \%$ , $+ 1 . 2 \hat { 7 } \%$ , $+ 3 . 4 9 \%$ , and $+ 1 . 3 7 \%$ F1 scores on the Person, Location, Building, Organization, Product, Event, and $A l l$ entity types respectively. The experimental results demonstrate that through robust modeling of intra-entity and inter-entity relationships, MQSPN exhibits superior capability in fine-grained multimodal entity understanding.

![](images/3da68f2bfcaa4cc342bea7c4ac651dd86246d0ff8d9425e29d9526252a5ed091.jpg)  
Figure 3: Analysis of candidate visual regions number $k$ for H-Index, TIGER, and our MQSPN.   
Figure 4: Analysis of the multi-grained queries quantity $u$ on GMNER, MNER and EEG tasks.

GMNER MNER EEG   
589.84 80.8 63.0   
578.062 80.0 62.4 61.8 79.2 61.2 778.64 60.06 25 50 75 25 50 75 25 50 75 The number of queries u

# Ablation Study

Ablation setting. To verify the effectiveness of each designed component in MQSPN, we conduct a comprehensive ablation study on the Twitter-GMNER and TwitterFMNERG dataset: For Multi-grained Query Set (MQS), (1) w/o PTQ: we remove the prompt-based type-grained part of the query set. (2) w/o LEQ: we replace the learnable queries part with human-designed queries (the same as MNER-QG query construction). For Query-guided Fusion Net (QFNet), (3) w/o QCT: we encode sentences and the query set using the original BERT without query-text crossattention. (4) w/o QPI: we eliminate the query-region prefix integration. (5) w/o SAG: we eliminate the similarity-aware aggregator. For Multimodal Set Prediction (MSP), (6) $w / o$ BML: we replace the bipartite matching loss with a joint cross-entropy loss in the fixed permutation of the entities. The experimental results are shown in Table 3.

The effectiveness of Multi-grained Query Set. In Table 3, we observe a clear F1 scores drop in model performance ( $1 . 8 1 \%$ in Twitter-GMNER and $1 . 3 4 \%$ in TwitterFMNERG) without the prompt-based type-grained queries. This indicates that integrating type-grained information into the query can enhance model performance. Furthermore, adding an additional learnable entity-grained part to the query substantially improves model performance: in the Twitter-GMNER and Twitter-FMNERG datasets, the F1

EEG: 广 Ground Truth w/o LEQ w/o BML MQSPN (Blackhawks,  Sport Team) (Blackhawks Bradon Saad, Person) (Blackhawks,  Other) (Blackhawks,  Sport Team)   
MNER: (Bradon Saad, Person) (Detroit,  Other) × (Bradon Saad, Person) (Bradon Saad, Person) (Detroit,  Sport Team) (Detroit s Johan Franzen, Person) (Detroit,  Other) (Detroit,  Sport Team) (Johan Franzen, Person) (Johan Franzen, Person) (Johan Franzen, Person)

scores increased by $+ 2 . 7 4 \%$ and $+ 3 . 1 8 \%$ , respectively. The main reason is that these learnable vectors can effectively learn the entity-grained semantics and model intraentity connections between entity spans and regions, thereby boosting model performance.

The effectiveness of Query-guided Fusion Net. In Table 3, the model without three fusion modules achieves consistent drops. We find the average F1 scores of these three modules decreased by $1 . 0 3 \%$ and $0 . 6 0 \%$ on TwitterGMNER and Twitter-FMNERG, respectively. Experimental results demonstrate that the QFNet can fuse information across different modalities and filter the noise introduced by irrelevant visual regions, which is crucial for improving the performance of two-level relationship modeling.

The effectiveness of Bipartite Matching Loss. In Table 3, excluding the bipartite matching loss from the model leads to an F1 scores decline of $2 . 1 8 \%$ and $1 . 7 6 \%$ in TwitterGMNER and Twitter-FMNERG datasets. This illustrates that the bipartite matching loss contributes to model performance through modeling inter-entity relationships in global matching perspective.

# Discussion and Analysis

Sensitivity Analysis of Irrelevant Visual Regions. The performance of EEG is largely determined by the ground truth coverage of the top- $k$ regions proposed by the classagnostic RPN. However, a higher value of $k$ implies the introduction of more irrelevant visual regions. To delve into the model’s error tolerance to noisy visual regions, we conduct experiments on H-Index, TIGER, and MQSPN under different values of $k$ . As shown in Figure 3, we can observe that the F1 scores of these three methods first increase, and then decrease with the augment of $k$ . Compared with H-Index and TIGER, our proposed MQSPN has better and more stable performance on both EEG and GMNER tasks when $k$ becomes larger. This demonstrates MQSPN’s insensitivity to irrelevant visual regions, indicating its excellent error tolerance and robustness of intra-entity connection.

Analysis of query quantity. To explore the impact of multi-grained query quantity (i.e., $u$ ) on GMNER and its 2 subtasks, we report the F1 scores of Twitter-GMNER in Figure 4 by tuning $u$ from 10 to 90. We observe that as the number of queries increases from 10 to 60, the model’s performance gradually improves. These results indicate that query quantity plays a crucial role in intra-entity connection learning. However, a query quantity exceeding 60 does not lead to better performance. There is an optimal number of queries for MQSPN. In our experiments, we find $u = 6 0$ in the Twitter-GMNER dataset can achieve the optimal results.

Case Study. We conduct a comprehensive case study in different ablation setting. As shown in Figure 5, when we replace learnable queries with human-designed queries in MQS (w/o LEQ setting), the model cannot differentiate Bradon Saad and Johan Franzen regions, and incorrectly localize Blackhawks and Detroit as Person spans, highlighting deficiencies in intra-entity learning for ambiguous entities. After integrating learnable MQS, the model precisely aligns textual spans with corresponding regions for Person entities. However, using MQS without MSP (w/o BML) leads to confusion in mapping proper inter-entity relationships, resulting in incorrect classifications for Blackhawks and Detroit and more erroneous span-type-region triplet predictions. After combining MQS and MSP, MQSPN effectively differentiates entities and makes accurate predictions, demonstrating the effect of intra-entity and inter-entity modeling.

# Conclusion

In this paper, we propose a novel method named MQSPN to model appropriate intra-entity and inter-entity relationships for the GMNER task. To achieve this, we propose a learnable Multi-grained Queries Set (MQS) to adaptively learn explicit intra-entity connections. Besides, we reformulate GMNER as a multimodal set prediction (MSP) to model inter-entity relationships from an optimal matching view. Experimental results demonstrate that MQSPN achieves state-of-the-art performance on GMNER and its 2 subtasks across two Twitter benchmarks.