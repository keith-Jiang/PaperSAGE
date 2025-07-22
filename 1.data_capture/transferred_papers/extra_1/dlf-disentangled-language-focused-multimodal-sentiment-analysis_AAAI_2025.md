# DLF: Disentangled-Language-Focused Multimodal Sentiment Analysis

Pan Wang1\*, Qiang Zhou1, Yawen $\mathbf { W } \mathbf { u } ^ { 1 }$ , Tianlong Chen2, Jingtong Hu1

1University of Pittsburgh 2UNC Chapel Hill {pan.wang, qiang.zhou, yawen.wu, jthu}@pitt.edu, tianlong@cs.unc.edu

# Abstract

Multimodal Sentiment Analysis (MSA) leverages heterogeneous modalities, such as language, vision, and audio, to enhance the understanding of human sentiment. While existing models often focus on extracting shared information across modalities or directly fusing heterogeneous modalities, such approaches can introduce redundancy and conflicts due to equal treatment of all modalities and the mutual transfer of information between modality pairs. To address these issues, we propose a Disentangled-Language-Focused (DLF) multimodal representation learning framework, which incorporates a feature disentanglement module to separate modalityshared and modality-specific information. To further reduce redundancy and enhance language-targeted features, four geometric measures are introduced to refine the disentanglement process. A Language-Focused Attractor (LFA) is further developed to strengthen language representation by leveraging complementary modality-specific information through a language-guided cross-attention mechanism. The framework also employs hierarchical predictions to improve overall accuracy. Extensive experiments on two popular MSA datasets, CMU-MOSI and CMU-MOSEI, demonstrate the significant performance gains achieved by the proposed DLF framework. Comprehensive ablation studies further validate the effectiveness of the feature disentanglement module, languagefocused attractor, and hierarchical predictions.

Code — https://github.com/pwang322/DLF.

# Introduction

With the rapid development of social media, multimodal interaction has become increasingly popular, which attracts many researchers to transfer uni-modal learning to multimodal learning tasks (Awal et al. 2024; Guan et al. 2024; Xu, Zhu, and Clifton 2023). One of the most significant subfields is multimodal sentiment analysis (MSA) (Geetha et al. 2024; Yang et al. 2022a). MSA aims to perceive human sentiment through multiple heterogeneous modalities, such as language, vision, and audio, playing a crucial role in many applications including cognitive psychology, scenario understanding, and mental health (Ali and Hughes 2023; Ezzameli and Mahersia 2023; Yang et al. 2023a). Compared with unimodal solutions, MSA often presents a more robust performance by leveraging complementary information from different modalities. How to effectively learn essential representations without redundant and conflicting information from multiple heterogeneous modalities, however, remains an open question in the academic field, especially in multimodal learning communities.

![](images/34a0bf1f1a080be387043b6169affc60aefd3a048502a8f0e520c08aa843128c.jpg)  
Figure 1: Task pipeline of the Multimodal Sentiment Analysis, and varied performance of different modalities.

In recent years, researchers have shown an increased interest in MSA. Many multimodal models have been proposed to facilitate MSA, and they can be categorized into two groups: representation learning-oriented methods (Guo et al. 2022; Hazarika, Zimmermann, and Poria 2020; Sun et al. 2023; Yang et al. 2022c) and multimodal fusionoriented methods (Zhang et al. 2023; Huang et al. 2020; Yang et al. 2022b; Tsai et al. 2019; Lv et al. 2021; Rahman et al. 2020). The former primarily aims to acquire an advanced semantic understanding of various modalities enriched with diverse clues of human sentiments, resulting in more powerful human sentiment encoders. Conversely, the latter emphasizes designing sophisticated fusion strategies at various levels, including feature-level, decision-level, and model-level fusion, to derive unified representations from multimodal data. It is worth noting that the fundamental aspect of MSA lies in learning and integrating multimodal representations, where the goal is to accurately process and integrate various modal inputs to discern sentiments from the underlying data. Although current leading methods in MSA (Hazarika, Zimmermann, and Poria 2020; Tsai et al. 2019; Zadeh et al. 2017; Yu et al. 2021) have shown considerable progress, the inherent disparities across diverse modalities continue to present challenges, complicating the development of stable and effective multimodal representation. For MSA, as shown in Figure 1, existing works and our ablation study (see Table 2) have shown that language, vision, and audio sources contribute differently to the overall prediction performance (Pham et al. 2019; Tsai et al. 2019; Kim and Park 2023; Li et al. 2024a; Lei et al. 2023), which indicates the big distribution gap among different modalities hinders the final performance.

To mitigate the distribution gap among heterogeneous modalities, as shown in Figure 1, knowledge distillationbased methods, such as cross-modal and graph distillation, are introduced to transfer reliable information between different modalities (Gupta, Hoffman, and Malik 2016; Guo et al. 2020; Aslam et al. 2023; Kim and Kang 2022; Hazarika, Zimmermann, and Poria 2020; Li, Wang, and Cui 2023; Tsai et al. 2019). Cross-modal distillation typically leverages the stronger Language modality to teach weaker modalities (Vision and Audio), while graph distillation completely performs bidirectional information transfer between all modality pairs. However, it is important to note that conventional distillation is inherently asymmetric—it is effective when transferring information from one modality to another, but the benefits of the reciprocal process are unclear. This asymmetry can lead to redundant or even conflicting information in cross-modal and graph distillation, ultimately limiting overall performance.

To this end, we critically reconsider the characteristics illustrated in Figure 1: Why focus solely on bridging the gap between different modalities, rather than strategically enhancing the strengths of the dominant one? However, directly enhancing the dominant modality while treating all modalities equally and employing bidirectional information transfer across all modality pairs often introduces redundancy and conflicts (Hazarika, Zimmermann, and Poria 2020), thereby reducing overall performance. In contrast, our work strategically leverages a pivotal characteristic of MSA: language has been empirically recognized as the dominant modality (Tsai et al. 2019). Building on this insight, we intend to develop a novel Language-Focused Attractor (LFA), a targeted enhancement scheme designed to transfer complementary information exclusively to the dominant language modality, which consolidates information through pathways such as Video $$ Language, Audio $$ Language, and Language $$ Language, resulting in effectively minimizing redundancy and conflicting information and improving overall MSA accuracy.

To achieve this, we propose a Disentangled-LanguageFocused (DLF) multimodal representation learning framework to fully exploit the potential of language-dominant MSA. The framework follows a structured pipeline: feature extraction, disentanglement, enhancement, fusion, and prediction. To specifically address the issues of redundancy and conflicting information to facilitate language-targeted feature enhancement, DLF introduces four geometric measures as regularization terms in the total loss function, effectively refining shared and specific spaces both separately and jointly. Within the modality-specific space, we further develop the LFA to enhance language representation by attracting complementary information from other modalities. This process is guided by a Language-Query-based multimodal cross-attention mechanism, ensuring precise and targeted feature enhancement between heterogeneous modality pairs ( $\boldsymbol { X } \to ]$ Language, where $X$ refers to Language, Video, or Audio). Finally, the enhanced shared and specific features are fused, followed by hierarchical predictions to further improve overall prediction accuracy.

Our main contributions can be summarized as follows:

• Proposed Framework: In this study, we propose a   
Disentangled-Language-Focused (DLF) multimodal representation learning framework to promote MSA tasks. The DLF framework presents a structured pipeline: feature extraction, disentanglement, enhancement, fusion, and prediction.   
• Language-Focused Attractor (LFA): We develop the   
LFA to fully harness the potential of the dominant language modality within the modality-specific space. The   
LFA exploits the language-guided multimodal crossattention mechanisms to achieve a targeted feature en  
hancement ( $X $ Language).   
• Hierarchical Predictions: We devise hierarchical predictions to leverage the pre-fused and post-fused features, improving the total MSA accuracy. Comprehensive ablation studies further validate the effectiveness of each component in the DLF framework.

# Related Work

# Multimodal Sentiment Analysis

Multimodal Sentiment Analysis (MSA) integrates information from diverse modalities, such as language, video, and audio (Ali and Hughes 2023; Ezzameli and Mahersia 2023). Mainstream methods can be categorized into representation learning-oriented (Guo et al. 2022; Sun et al. 2023; Yang et al. 2022c) and fusion-oriented approaches (Zhang et al. 2023; Huang et al. 2020; Tsai et al. 2019). Representation methods like (Guo et al. 2022), enhance cross-modal interactions, while fusion techniques, including Transformerbased model (Huang et al. 2020), focus on combining features effectively. Despite progress, performance disparities among modalities hinder the overall prediction accuracy. To this end, distillation-based strategies were introduced to bridge the gap. For instance, Kim and Kang (2022) proposed cross-modal distillation between textual and auditory modalities to enhance emotion classification granularity. To dynamically adapt to distillation, ternary-symmetric architectures (MulT) (Tsai et al. 2019) or graph distillation units (Zadeh et al. 2018b; Li, Wang, and Cui 2023) were introduced, representing modalities as vertices and their interactions as edges. Recent approaches also leverage large multimodal language models for flexible interactions (Wu et al.

![](images/fa6ea4467ea0ebd3ec8da1b9ad82c8b8c6f4d388b04e6f04f0ed13577f2f0e6b.jpg)  
Figure 2: Overview of the proposed DLF framework. The framework follows a pipeline of feature extraction, disentanglement, enhancement, fusion, and prediction, featuring three core components: the feature disentanglement module, the LanguageFocused Attractor (LFA), and hierarchical predictions (including shared prediction, specific prediction, and final prediction).

2023) and integrate contextual knowledge to boost predictions (Wang et al. 2024). However, previous paradigms treat different modalities equally, easily causing redundant and conflicting information. Our proposed Language-Focused Attractor (LFA) directs knowledge transfer to the dominant language, mitigating redundant and conflicting information and enhancing overall performance.

# Disentangled Multimodal Representation Learning

Disentangled multimodal representation learning aims to separate distinct factors of variation across multiple modalities (e.g., vision, language, audio) into independent subspaces (Yang et al. 2023b; Li et al. 2024b; Yang et al. 2023a). Tsai et al. (2018) introduced the concept of factorized representations for multimodal data, designed to disentangle modality-specific and shared components, successfully isolating audio and visual features in audiovisual speech recognition. Building on this, Hazarika, Zimmermann, and Poria (2020) proposed the MISA framework, which projects each modality into common and private feature spaces, reducing inter-modality disparities while enhancing representation diversity. Further advancements including Yang et al. (2022a,c), employed metric learning and adversarial learning to construct modality-invariant and modalityspecific subspaces, significantly improving multimodal fusion. Li, Wang, and Cui (2023) developed a decoupled multimodal distillation (DMD) approach to address distribution gaps between modalities. However, previous methods uniformly treat all modalities regarding the purpose of disentanglement. In our DLF, disentanglement aims to facilitate language-targeted feature enhancement in LFA, thus, we introduce four measures as regularization terms, carefully designed to reinforce disentanglement by jointly and independently optimizing shared and specific spaces.

# Proposed Approach

Preliminaries. As shown in Figure 2, the task of MSA aims to predict the sentiment intensity or label of given multimodal inputs. In DLF, three modalities are concurrently considered, such as Language $( L )$ , Vision $( V )$ , and Audio $( A )$ , represented as 2D tensors $\hat { X } _ { m } \in \mathop { R ^ { N _ { m } \times d _ { m } } }$ , where $N _ { m }$ is the sequence length, $d _ { m }$ is the embedding dimension, and $m \in \{ \bar { L } , V , A \}$ means different modalities.

# Overview

The framework of the proposed DLF is illustrated in Figure 2. It adopts a structured pipeline comprising feature extraction, disentanglement, enhancement, fusion, and prediction. The framework integrates three core components: the feature disentanglement module, the Language-Focused Attractor (LFA), and hierarchical predictions (shared, specific, and final predictions). DLF decomposes multimodal features into modality-shared and modality-specific spaces to minimize redundancy and conflicts among heterogeneous modalities. To reinforce this decoupling, four geometric measures are incorporated into the total loss as regularization terms. Additionally, the LFA is designed to leverage the dominant language modality by integrating complementary information from other modalities, thereby enhancing language representation. Finally, hierarchical predictions are performed to boost overall MSA performance. The details are as follows:

# Feature Disentanglement Module

To reduce redundant and conflicting information, the proposed DLF framework utilizes a shared encoder and three modality-specific encoders to decompose multimodal information into modality-shared and modality-specific feature spaces, denoted as $S h ^ { m }$ and $S p ^ { m }$ , respectively, where $m \in \{ V , L , A \}$ . Formally, the shared and specific encoders are defined as:

$$
\begin{array} { r } { { S h ^ { m } = E _ { m } ^ { S h } ( \hat { X } _ { m } ) , } } \\ { { S p ^ { m } = E _ { m } ^ { S p } ( \hat { X } _ { m } ) , } } \end{array}
$$

where $E _ { m } ^ { S h }$ and $E _ { m } ^ { S p }$ represent the shared and specific encoders, respectively. In this work, both encoders are implemented as cascaded Transformer layers.

For effective disentanglement, DLF incorporates the regularization effect of carefully designed regularization terms. While classical approaches often employ distribution similarity measures such as KL-Divergence along the hidden dimensions (Kim and Mnih 2018), we adopt four geometric measures based on Euclidean distances and cosine similarity due to their intuitive nature and computational efficiency.

After initial disentanglement, DLF concatenates $S h ^ { m }$ and $S p ^ { m }$ for each modality and reconstruct the multimodal input $\hat { X } _ { m }$ , resulting $\hat { X } _ { m } ^ { \prime }$ by decoding the fused features $[ S h ^ { m } \oplus S p ^ { m } ]$ . This process can be formulated as follows:

$$
\hat { X } _ { m } ^ { \prime } = D _ { m } ( [ S h ^ { m } \oplus S p ^ { m } ] ) ,
$$

where $D _ { m }$ is the 1D convolution decoder, and $\oplus$ is the concatenation operation. The discrepancy between the low-level features $\hat { X } _ { m }$ and the generated features $\hat { X } _ { m } ^ { \prime }$ , called the reconstruction loss $L _ { r }$ , can serve as a regularization term contributing to the feature disentanglement module:

$$
L _ { r } = | | \hat { X } _ { m } - \hat { X } _ { m } ^ { \prime } | | ^ { 2 } .
$$

Furthermore, the modality-specific reconstruction process can be formulated as:

$$
S p ^ { m } { } ^ { \prime } = E _ { m } ^ { S p } ( \hat { X } _ { m } ^ { \prime } ) ,
$$

where $S p ^ { m \prime }$ is estimated modality-specific features from the modality-specific reconstruction process. Naturally, the discrepancy between original modality-specific features $S p ^ { m }$ and the estimated ones $S p ^ { m ^ { \prime } }$ can be regarded as the specific loss $L _ { s }$ :

$$
L _ { s } = | | S p ^ { m } - S p ^ { m \prime } | | ^ { 2 } .
$$

Although the reconstruction loss $L _ { r }$ and specific loss $L _ { s }$ contribute to the decoupling process, their effectiveness in achieving robust disentanglement remains limited. This limitation arises from the potential sub-optimal performance of the shared encoder during the initial training phase, especially when compared to the modality-specific encoder. Such a disparity, if left unaddressed, is exacerbated by these loss functions, causing an increasing divergence between the two encoders as training progresses. Therefore, we incorporate a modified triplet loss (Schroff, Kalenichenko, and Philbin 2015) to enhance the performance of the modalityshared encoder. The triplet loss is defined as:

$$
L _ { m } = \frac { 1 } { \left| T \right| } \operatorname* { m a x } \left( 0 , d \left( S , P \right) - d \left( S , N \right) + \mu \right) ,
$$

where $S$ represents a sampled modality in the modalityshared space, $P$ denotes the positive sample corresponding to the representation of the same sentiment across different modalities, and $N$ refers to the negative sample representing distinct sentiments within the same modality. $T$ is the total number of positive and negative samples, $d ( \cdot , \cdot )$ computes the cosine similarity between two feature vectors, and $\mu$ represents a distance margin.

The aforementioned losses, $L _ { r } , L _ { s }$ , and $L _ { m }$ , regulate the shared and specific features to ensure they focus on their respective objectives. To further refine the decoupling between these two spaces, a soft orthogonality loss, $L _ { o }$ , is introduced to minimize redundancy and conflicts between shared and specific multimodal features. It is defined as:

$$
L _ { o } = O ( S h ^ { m } , S p ^ { m } ) ,
$$

where $O ( \cdot , \cdot )$ represents a non-negative counterpart of cosine similarity, promoting orthogonality between the two feature spaces.

Eventually, the four geometric-measure-based regularization terms, addressing both inter- and intra-decoupled spaces, are combined to form the decoupling loss:

$$
L _ { d } = \sum _ { k \in \{ r , s , m , o \} } \lambda _ { k } L _ { k } ,
$$

where $\lambda _ { k }$ are weighting coefficients for the individual regularization terms, providing a flexible mechanism to balance their contributions and calibrate the model effectively.

# Language-Focused Attractor (LFA)

Unlike conventional feature enhancement methods that aim to bridge modality gaps through cross-modal and graph distillation (Gupta, Hoffman, and Malik 2016; Guo et al. 2020; Aslam et al. 2023; Li, Wang, and Cui 2023), we propose the LFA in the modality-specific space after feature decoupling. The detailed structure of LFA is depicted in Figure 3.

In the LFA, decoupled modality-specific features $S p ^ { m }$ (where $m \in \{ L , V , \bar { A } \}$ , representing language, vision, and audio) are first processed through positional embedding and dropout, then fed into Multimodal Transformer layers. The core operation within these layers is the Multimodal Cross-Attention (MCA) mechanism. LFA performs three branches of MCA, including one self-attention and two cross-attention mechanisms, all centered on the language modality as the Query $( Q _ { L } )$ . This setup allows the language modality to attract complementary information from other modality-specific features $S p ^ { m }$ , where $m \in \{ L , V , A \}$ . The corresponding Key-Value pairs are defined as $( K _ { m } , V _ { m } )$ . The MCA operation is mathematically expressed as:

$$
\begin{array} { r l } & { \quad \mathrm { M C A } ( Q _ { L } , K _ { m } , V _ { m } ) } \\ & { = s o f t m a x \left( \frac { Q _ { L } K _ { m } ^ { T } } { \sqrt { d } } \right) V _ { m } } \\ & { = s o f t m a x \left( \frac { ( S p ^ { L } ) W _ { Q _ { L } } ( S p ^ { m } ) W _ { K _ { m } } ^ { T } } { \sqrt { d } } \right) ( S p ^ { m } ) W _ { V _ { m } } } \end{array}
$$

where $m \in \{ L , V , A \}$ , softmax represents normalized attention score between $Q _ { L }$ and $K _ { m }$ , $W _ { Q _ { L } }$ and $W _ { K _ { m } }$ are learnable parameters, $d$ indicates the dimension of $Q _ { L }$ and

Positional Embedding LFA D T 1 1 L Add&Norm Add & Norm Add & Norm ↑ FFN FFN FFN ↑ 1 Add&Norm Add & Norm Add & Norm ↑ Dropout Dropout Dropout CrossAttention Self-attention Cross Attention Ky V ? QKV QL ↑↑KA VA↑ □ 不 ↑ ↑ SpecificFeatures Specific Features Specific Features

is defined as:

$$
\begin{array} { r } { h _ { m } ^ { n + 1 } = L a y e r N o r m ( h _ { m } ^ { n } + D r o p ( \mathbf { M } \mathbf { C } \mathbf { A } ( h _ { m } ^ { n } ) ) ) } \\ { h _ { m } ^ { o } = L a y e r N o r m ( h _ { m } ^ { n + 1 } + F F N ( h _ { m } ^ { n + 1 } ) ) \quad } \end{array}
$$

where $m \in \{ L , V , A \}$ , $D r o p ( \cdot )$ denotes the Dropout operation, and $F F N ( \cdot )$ represents a feed-forward module. Here, $h _ { m } ^ { n }$ and $h _ { m } ^ { n + 1 }$ are the input and intermediate features, respectively, while $h _ { m } ^ { o }$ is the output of a Multimodal Transformer layer. As illustrated in Figure 3, cascaded Multimodal Transformers are employed to enhance modality-specific features.

$$
L _ { f } = \frac { 1 } { N _ { d } } \sum _ { n = 0 } ^ { N _ { d } } \lvert \hat { y } _ { n } - y _ { n } \rvert ,
$$

The LFA effectively leverages modality-specific features from three modalities, aligning them with the language modality to strengthen multimodal representation. As shown in Figure 2, the enhanced features are projected into higherlevel specific features, denoted as $H S p ^ { m }$ $( m \in \{ L , V , A \} )$ ), and then integrated with enhanced shared features. Additionally, these features are processed by modality-specific predictors for the specific prediction.

$K _ { m }$ . Consequently, the language-focused feature enhancement in the Multimodal Transformer is defined as:

where $y _ { n }$ is the MSA label, $N _ { d }$ is the number of samples. Unlike traditional MSA learning, which only involves a single output loss $L _ { f }$ , the proposed DLF explores hierarchical predictions considering modality-shared loss $L _ { S h }$ , modality-specific loss $L _ { S p ^ { m } }$ , and the output loss $L _ { f }$ concurrently. The total MSA learning loss is thus expressed as:

Multimodal Fusion. As illustrated in Figure 2, modalityshared features are first processed through a unified Transformer layer, followed by two fully connected layers, and then projected into higher-level shared features, denoted as $H S h$ . Finally, the multimodal fusion layer combines $H S h$ with $H S p ^ { m }$ to form the final multimodal features:

$$
L _ { M S A } = \sum _ { l \in \{ f , S h , S p ^ { m } \} } \beta _ { l } L _ { l } ,
$$

$$
\begin{array} { c } { { F ( H S h , H S p ^ { m } ) } } \\ { { = C o n c a t ( H S p ^ { L } , H S p ^ { V } , H S p ^ { A } , H S h ) , } } \end{array}
$$

where Concat denotes the concatenation operation.

# Hierarchical Predictions

where $m \ \in \ \{ L , V , A \}$ , $\beta _ { l }$ are weighting coefficients that control the relative importance of different losses.

As shown in Figure 2, a classifier can predict the MSA output $\hat { y }$ after multimodal fusion. The final MSA output loss $L _ { f }$

Overall Learning Objective. The proposed DLF framework integrates the decoupling loss $L _ { d }$ and the total MSA learning loss $L _ { M S A }$ to form the overall learning objective:

$$
{ \cal L } _ { D L F } = { \cal L } _ { d } + { \cal L } _ { M S A } .
$$

# Experiments Datasets and Evaluation Metrics

We evaluate DLF on two widely used datasets: CMU Multimodal Sentiment Intensity (MOSI) (Zadeh et al. 2016) and CMU Multimodal Opinion Sentiment and Emotion Intensity (MOSEI) (Zadeh et al. 2018b).

MOSI. The MOSI dataset comprises 2,199 monologue video clips, with audio and visual features extracted at 12.5 $\mathrm { H z }$ and $1 5 \mathrm { H z }$ , respectively. The dataset is divided into 1,284 training, 229 validation, and 686 test samples.

MOSEI. The MOSEI dataset, significantly larger, consists of 22,856 movie review video clips sourced from YouTube. Features are extracted at $2 0 \ : \mathrm { H z }$ for audio and 15 $\mathrm { H z }$ for visual modalities. The dataset is split into 16,326 training samples, 1,871 validation samples, and 4,659 test samples. For both datasets, each video clip is annotated with a sentiment score ranging from -3 to 3, representing a spectrum from highly negative to highly positive sentiment.

Evaluation Metrics. Consistent with established practices in previous studies (Liang et al. 2021; Lv et al. 2021; Mao et al. 2022), the performance of MSA is evaluated using multiple metrics: 7-class accuracy (Acc-7), 5-class accuracy (Acc-5), binary accuracy (Acc-2), F1 score, correlation between model predictions and human annotations (Corr), and mean absolute error (MAE). These metrics collectively offer a comprehensive assessment of DLF’s effectiveness across various sentiment analysis tasks.

# Implementation Details

In this study, we align our methodology with previous works (Hazarika, Zimmermann, and Poria 2020; Mao et al. 2022) by utilizing the BERT-base-uncased model (Devlin et al. 2018) to extract unimodal linguistic features. This process generates word representations with a 768-dimensional hidden state. For visual data, DLF employs the Facet framework (Baltrusˇaitis, Robinson, and Morency 2016) to encode each video frame, focusing on 35 distinct facial action units as detailed in (Li et al. 2019). For audio processing, we utilize the COVAREP framework (Degottex et al. 2014), which produces 74-dimensional audio features. Our experiments are implemented using the PyTorch framework and executed on one NVIDIA V100 GPU with 32GB of memory. The model is trained with a batch size of 16 and optimized using an initial learning rate of 1e-4. Early stopping with a patience of 10 epochs is applied to ensure convergence.

<html><body><table><tr><td rowspan="2">Method</td><td colspan="6">CMU-MOSI</td><td colspan="6">CMU-MOSEI</td></tr><tr><td>Acc-7(↑) Acc-5(↑) Acc-2(↑) F1(↑) Corr(↑) MAE(↓)</td><td></td><td></td><td></td><td></td><td></td><td></td><td>Acc-7(↑) Acc-5(↑) Acc-2(↑) F1(↑) Corr(↑) MAE(↓)</td><td></td><td></td><td></td><td></td></tr><tr><td>TFN*</td><td>34.90</td><td>39.39†</td><td>80.08</td><td>80.07</td><td>0.698</td><td>0.901</td><td>50.20</td><td>53.10†</td><td>82.50</td><td>82.10</td><td>0.700</td><td>0.593</td></tr><tr><td>LMF*</td><td>33.20</td><td>38.13†</td><td>82.50</td><td>82.40</td><td>0.695</td><td>0.917</td><td>48.00</td><td>52.90+</td><td>82.00</td><td>82.10</td><td>0.677</td><td>0.623</td></tr><tr><td>EF-LSTM†</td><td>35.39</td><td>40.15</td><td>78.48</td><td>78.51</td><td>0.669</td><td>0.949</td><td>50.01</td><td>51.16</td><td>80.79</td><td>80.67</td><td>0.683</td><td>0.601</td></tr><tr><td>LF-DNN†</td><td>34.52</td><td>38.05</td><td>78.63</td><td>78.63</td><td>0.658</td><td>0.955</td><td>50.83</td><td>51.97</td><td>82.74</td><td>82.52</td><td>0.709</td><td>0.580</td></tr><tr><td>MFN†</td><td>35.83</td><td>40.47</td><td>78.87</td><td>78.90</td><td>0.670</td><td>0.927</td><td>51.34</td><td>52.76</td><td>82.85</td><td>82.85</td><td>0.718</td><td>0.575</td></tr><tr><td>Graph-MFN†</td><td>34.64</td><td>38.63</td><td>78.35</td><td>78.35</td><td>0.649</td><td>0.956</td><td>51.37</td><td>52.69</td><td>83.48</td><td>83.43</td><td>0.713</td><td>0.575</td></tr><tr><td>MulT</td><td>40.00</td><td>42.68†</td><td>83.00</td><td>82.00</td><td>0.698</td><td>0.871</td><td>51.80</td><td>54.18†</td><td>82.50</td><td>82.30</td><td>0.703</td><td>0.580</td></tr><tr><td>PMR</td><td>40.60</td><td></td><td>83.60</td><td>83.60</td><td></td><td></td><td>52.50</td><td></td><td>83.60</td><td>83.40</td><td></td><td></td></tr><tr><td>MISA†</td><td>41.37</td><td>47.08</td><td>83.54</td><td>83.58</td><td>0.778</td><td>0.777</td><td>52.05</td><td>53.63</td><td>84.67</td><td>84.66</td><td>0.752</td><td>0.558</td></tr><tr><td>MAG-BERT</td><td>43.62</td><td></td><td>84.43</td><td>84.61</td><td>0.781</td><td>0.727</td><td>52.67</td><td></td><td>84.82</td><td>84.71</td><td>0.755</td><td>0.543</td></tr><tr><td>DMD**</td><td>46.06</td><td></td><td>83.23</td><td>83.29</td><td></td><td>0.752</td><td>52.78</td><td></td><td>84.62</td><td>84.62</td><td></td><td>0.543</td></tr><tr><td>DLF (Ours)</td><td>47.08</td><td>52.33</td><td>85.06</td><td>85.04</td><td>0.781</td><td>0.731</td><td>53.90</td><td>55.70</td><td>85.42</td><td>85.27</td><td>0.764</td><td>0.536</td></tr></table></body></html>

Table 1: Comparision on MOSI and MOSEI. Bold is the best. Note: † represents the result from THUIAR’s GitHub page (Thuiar 2024), ∗ represents the result from (Hazarika, Zimmermann, and Poria 2020), - represents the result from the original paper is not provided, and ∗∗ represents reproduced results from public code with hyper-parameters provided in the original paper.

# Main Results

Baselines. We compare the DLF against eleven leading MSA methods on both benchmarks, including EF-LSTM (Williams et al. 2018b), LF-DNN (Williams et al. 2018a), TFN (Zadeh et al. 2017), LMF (Liu et al. 2018), MFN (Zadeh et al. 2018a), Graph-MFN (Zadeh et al. 2018b), MulT (Tsai et al. 2019), PMR (Lv et al. 2021), MISA (Hazarika, Zimmermann, and Poria 2020), MAG-BERT (Rahman et al. 2020), and DMD (Li, Wang, and Cui 2023).

Performance Comparison. Comparative results, as reported in Table 1, demonstrate that our proposed DLF exhibits superior performance on almost all metrics for both benchmarks. Particularly, we have the following key observations. Compared to decoupled-feature-based MSA methods like MISA (Hazarika, Zimmermann, and Poria 2020), MulT (Tsai et al. 2019), and DMD (Li, Wang, and Cui 2023), the proposed DLF, especially the LFA, captures effective intermodality dynamics and further improves the multimodal representation capability by enhancing the dominant language in the specific subspace. Compared to methods that leverage multimodal transformers to learn crossmodal interactions and fusion such as LMF (Liu et al. 2018), MFN (Zadeh et al. 2018a), and PMR (Lv et al. 2021), our proposed method learns effective multimodal representations in the disentangled subspaces and further facilitates the overall prediction performance using hierarchical predictions which utilizes both pre-fused and post-fused features.

# Ablation Study

We conduct extensive ablation studies to thoroughly examine the impact of various modality combinations, different regularization strategies, and critical components including the Feature Disentanglement Module (FDM), LanguageFocused Attractor (LFA), and Hierarchical Predictions (HP).

Various Modality Combinations. As shown in Table 2, all analysis are conducted on the MOSI dataset. We first present the performance of each unimodality, it is easy to notice that language modality $( L )$ serves as the dominant one. In the bi-modalities case, we consider both $( L , A )$ and $( L , V )$ pairs in our DLF framework, the results not only demonstrate the performance improvement, especially the finegrained classification, through two modalities but also showcase that language modality attracts useful information from vision $( V )$ or audio $( A )$ modality by LFA to enhance the multimodal representation capability. Furthermore, the trimodalities DLF consistently outperforms the bi-modalities DLF on all metrics, which indicates that each modality provides a unique contribution and multimodal learning can effectively improve the MSA performance by reasonably exploiting the information from different modalities.

Different Regularization. We remove each loss to verify the importance of different regularization terms. When removing the soft orthogonality loss $L _ { o }$ , the DLF learns decoupled features under the constraints that focus on separate subspaces. The worst performance suggests the importance of the soft orthogonality loss considering the shared and specific subspaces jointly in the feature disentanglement module. Meanwhile, we also notice that the modified triplet loss $L _ { m }$ in the shared subspace improves the overall performance, which indicates the importance of $L _ { m }$ in learning shared features in the shared subspace. Besides, we observe that both reconstruction loss $L _ { r }$ and specific loss $L _ { s }$ contribute to the model’s performance. This is because these two losses ensure feature consistency during disentanglement.

Critical Components. To verify the effectiveness of different components of DLF, we remove each critical component separately. The removal of LFA, replaced by three separate Query components like MulT (Tsai et al. 2019), leads to a remarkable decrease in overall MSA performance. This demonstrates that the LFA is a straightforward and effective component, mitigating the potential redundancies or conflicts in traditional cross-attention mechanisms. Moreover, when deactivating the FDM, the performance also becomes inferior to that of DLF, which shows the effectiveness of the designed FDM and further indicates the redundant and conflicting information limits the MSA performance. With a similar phenomenon, subtracting the HP module (which just remains the final output loss function) decreases the overall performance again, revealing the value of both the pre-fused and post-fused features.

Table 2: Results of ablation studies on the MOSI benchmark.   

<html><body><table><tr><td>Method Acc-7 (%)</td><td>Acc-2 (%)</td><td>F1 (%)</td><td>MAE(↓)</td></tr><tr><td>DLF (Ours)</td><td>47.08</td><td>85.06 85.04</td><td>0.731</td></tr><tr><td></td><td>DifferentModalities</td><td></td><td></td></tr><tr><td>only A</td><td>15.31 42.84</td><td>26.64</td><td>1.453 1.455</td></tr><tr><td>only V only L</td><td>15.01 43.29 45.63 84.45</td><td>29.73 84.38</td><td>0.752</td></tr><tr><td>L&A</td><td>45.77 83.84</td><td>83.88</td><td>0.741</td></tr><tr><td>L&V</td><td>46.65 83.08</td><td>83.13</td><td>0.745</td></tr><tr><td></td><td>Different Regularization</td><td></td><td></td></tr><tr><td>w/o Lr</td><td>45.92 84.67</td><td>84.59</td><td>0.734</td></tr><tr><td>w/o Ls</td><td>45.36 84.60</td><td>84.56</td><td>0.740</td></tr><tr><td>w/o Lm</td><td>45.77 83.99</td><td>83.97</td><td>0.735</td></tr><tr><td>w/o Lo</td><td>45.77 83.08</td><td>83.16</td><td>0.738</td></tr><tr><td></td><td>Different Components</td><td></td><td></td></tr><tr><td>w/o FDM</td><td>45.92</td><td>84.60 84.58</td><td>0.739</td></tr><tr><td>W/o LFA</td><td>42.71</td><td>83.84 83.85</td><td>0.767</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>W/o HP</td><td>42.42 84.76</td><td>84.74</td><td>0.761</td></tr></table></body></html>

# Further Analysis

To further study the impact of sentiment granularity on MSA, we present the confusion matrix and corresponding accuracy of each sentiment for the MOSI benchmark. As shown in Figure 4, we observe that most sentiment classes have similar accuracy above $4 0 \%$ . However, “HN” and “HP”, especially “HN”, have the worst performance, limiting the overall MSA performance. Furthermore, when we dive into the confusion matrix, it can be noticed that the samples for “HN” and “HP” are relatively less than other sentiments, which strongly indicates that the long-tailed distribution of the data limits the overall MSA performance, which can be studied in the future.

To better understand the effectiveness of our method, we visualize the distribution of fused multimodal representations. As depicted in Figure 5, compared to DMD, which is a decoupled-multimodal-distillation strategy, our proposed DLF shows superior performance in separating different sentiments. This is mainly due to that the LFA mitigates redundant and conflicting information during multimodal interaction compared to the interaction between random pairs.

![](images/1b8f675aba0748d4e87f059fb6c512d5d1fdbb01eb26c215535ca991084c3ab2.jpg)  
Figure 4: Left: Confusion matrix on MOSI. Right: Corresponding accuracy for each sentiment. HN: Highly Negative; N: Negative; WN: Weakly Negative; NT: Neutral; WP: Weak Positive; P: Positive; HP: Highly Positive.

![](images/398a9e82f79539ecc42eb384704fd174a1f37283ce8e7bbb97d948e3feb3fa27.jpg)  
Figure 5: Visualization of the fused multimodal representations. HN: Highly Negative; N: Negative; WN: Weakly Negative; NT: Neutral; WP: Weak Positive; P: Positive; HP: Highly Positive.

# Conclusion

In this paper, we propose the DLF framework to improve the MSA performance. DLF yields powerful multimodal representations by following the pipeline of feature extraction, disentanglement, enhancement, fusion, and prediction, mainly benefiting from the feature disentanglement module, language-focused attractor, and hierarchical predictions. Extensive results verify the superiority of DLF by comparisons with eleven baselines and comprehensive ablation studies.

Broad impacts. (i) This study demonstrates its potential to exploit the imbalanced capabilities of various modalities in multimodal learning, thereby setting a new benchmark in this field. (ii) The proposed LFA facilitates the generalization of our method to other multimodal scenarios by changing the dominant modality. Limitation and future work. Our method only considers the scenarios of complete modalities. When facing missing modalities, the feature disentanglement and enhancement modules are potentially limited.

# Acknowledgments

This work is supported in part by NIH R01EB033387, NSF CNS-2122320, and NSF CCF-2324937. Tianlong Chen is supported by NIH OT2OD038045-01 and UNC SDSS Seed Grant.