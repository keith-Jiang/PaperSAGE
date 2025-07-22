# Towards Multimodal Sentiment Analysis via Hierarchical Correlation Modeling with Semantic Distribution Constraints

Qinfu $\mathbf { X } \mathbf { u } ^ { 1 }$ , Yiwei Wei2\*, Chunlei $\mathbf { W } \mathbf { u } ^ { 1 }$ , Leiquan Wang1, Shaozu Yuan3, Jie $\mathbf { W } \mathbf { u } ^ { 1 }$ , Jing $\mathbf { L u } ^ { 1 }$ , Hengyang Zhou2

1Qingdao Institute of Software, College of Computer Science and Technology, China University of Petroleum (East China) 2China University of Petroleum (Beijing) at Karamay, 3 JD AI Research xqfupc $ @ 1 6 3 . \mathrm { c o m }$ , weiyiwei $@$ cupk.edu.cn

# Abstract

Sentiment analysis is rapidly advancing by utilizing various data modalities (e.g., text, video, and audio). However, most existing techniques only learn the atomic-level features that reflect strong correlations, while ignoring more complex compositions in multimodal data. Moreover, they also neglected the incongruity in semantic distribution among modalities. In light of this, we introduce a novel Hierarchical Correlation Modeling Network (HCMNet), which enhances the multimodal sentiment analysis by exploring both the atomic-level correlations based on dynamic attention reasoning and the composition-level correlations through topological graph reasoning. In addition, we also alleviate the impact of distributional inconsistencies between modalities from both atomic-level and composition-level perspectives. Specifically, we first design an atomic-level contrastive loss that constrains the semantic distribution across modalities to mitigate the atomic-level inconsistency. Then, we design a graph optimal transport module that integrates transport flows with different graphs to constrain the composition-level semantic distribution, thus reducing the inconsistency of compositional nodes. Experiments on three public benchmark datasets have demonstrated the superiority of the proposed model over the state-of-the-art methods.

# Introduction

Multimodal Sentiment Analysis (MSA), a challenging but significant research topic, has gained increasing attention and more scientific efforts owing to its facility to convey emotions and views of individuals (Veltmeijer, Gerritsen, and Hindriks 2021; Zhang, He, and Lu 2019; Song et al. 2021). It aims to learn emotional information from mixed data containing multiple modalities (e.g., text, video, and audio) and make judgments based on psychological categorization. Current methods (Hu et al. 2021; Ma et al. 2022; Hu et al. 2022; Shi and Huang 2023; Tu et al. 2024) that aim to learn efficient emotional representations for MSA heavily rely on the hypothesis that different modalities exhibit evident correlations and design diversified fusion methods, which facilitates the cross-modal representation for final prediction. Some of the previous approaches (Hu et al. 2021; Chudasama et al. 2022; Shi and Huang 2023) employ advanced attention mechanisms to integrate strong correlations for emotion modeling. Other studies (Hu et al. 2022; Yang et al. 2023) focus on designing unified solutions for dual granularity emotion recognition. With the equipment of fine-grained multimodal content modeling, they have constantly promoted correlation mining for MSA task.

![](images/cfd3c0bbe8c043d570f21a6456f18749f098843b0675a1ffe9117383187ebd69.jpg)  
Figure 1: Illustration of hierarchical correlation modeling and the semantic distributions. Up: shows the complexity of multimodal correlations. Previous MSA methods tend to learn atomic-level correlations between modalities, ignoring the composition-level modeling that aims to capture weak but vital correlations. Down: shows the semantic distribution of a case obtained by unimodal classification, which indicates the incongruity between unimodal semantics.

Although promising, they still suffer from two limitations. On the one hand, most of the existing methods only consider the atomic-level correlations between different modalities (Hu et al. 2021; Chudasama et al. 2022; Yang et al. 2023) and ignore the importance of multi-granularity alignments (e.g., granularity such as frames, and relations between video frames or audio segments), which have been proven to be effective in other related multi-modal tasks, such as cross-modal retrieval (Li et al. 2021) and imagesentence matching (Xu et al. 2020; Liu et al. 2020). The hierarchical structures of both texts and other modalities advocate for weak correlation modeling. By exploring compositional semantics, helps to identify more vital but implicit correlations, e.g., correlations between an unaligned pair of entities and a group of video frames. On the other hand, multi-modal data inherently exhibits complex interrelations, with inconsistent semantic distributions (shown in Figure 1) among different modalities, thereby resulting in differences in sentiment congruity in both atomic-level and composition-level perspectives. Unfortunately, previous works overlooked the harm brought by such inconsistency for achieving better multimodal sentiment analysis.

To tackle these limitations, in this paper, we propose a novel Hierarchical Correlation Modeling Network (HCMNet) for multi-modal sentiment analysis. Specifically, our proposed method takes both atomic-level correlations between independent video frames, audio segments, and text tokens, as well as composition-level correlations considering spatial and semantic dependencies to explore weak dependency signals. To obtain atomic-level correlations, we design a dynamic attention reasoning method to align different modalities into the same space and compute the similarity score for each token-utterance-segment pair via inner products. Next, we obtain composition-level correlations based on the treated features of the text, audio, and video modalities acquired in the previous step. Concretely, we introduce a topological graph reasoning strategy, which constructs three uni-modal graphs using semantic dependencies among words and spatial dependencies among video utterances or audio segments to capture composition-level features for each modality using graph convolutional networks.

More importantly, we also mitigate the impact of distributional inconsistencies between modalities from both atomiclevel and composition-level perspectives. From the atomiclevel perspective, we design an atomic-level contrastive loss that empowers the model to learn robust class-relevant features in atomic-level feature space and alleviate the adverse effect of distributional inconsistency. As for compositionlevel, we propose a semantic optimal transport module to integrate transport flows with video, audio, and text graphs to constrain the composition-level incongruities between modalities. Specifically, we first utilize an optimal transport kernel to redefine the alignment problem across different modality pairs, eliminating the distributional gap between modalities by computing an informative cost matrix between video, audio, and text graphs. Then, we acquire optimal transportation plans, which are used for assigning source values to target distribution at minimum total cost. By doing so, it can learn strong cross-modal distributional consistency in composition-level features.

We validated our HCMNet on several benchmarks including CMU-MOSEI, IEMOCAP, and MELD over several models. Experiments demonstrate the effectiveness and universality of our approach, and extensive analyses provide insights into when and how our method works. In summary, the main contributions of this work are as follows:

• To the best of our knowledge, we are the first to exploit hierarchical semantic correlations between textual and visual modalities to jointly model the atomic-level and composition-level correlations for MSA task.

• We also mitigate the impact of distributional inconsistencies between modalities from both atomic-level and composition-level perspectives via contrastive learning and optimal transport learning.   
• At the same time, the universality of our HCMNet method also provides the possibility to extend it to other multimodal understanding tasks.

# Related Work

Multimodal Sentiment Analysis. Most MSA solutions adopt two different paradigms to understand multimodal emotion content. First, some of them paid more attention to designing advanced transformer architectures to capture the emotion dependencies across different modalities. UniMSE (Hu et al. 2022) proposed to obtain multimodal features that are fused by integrating audio and vision representations into a language model. MVN (Ma et al. 2022) proposed a multi-view network to explore both word-level and utterance-level emotion information. i-Code (Yang et al. 2023) designed an integrative and composable multimodal learning framework for triple-modal learning. MultiEMO (Shi and Huang 2023) integrated multimodal cues by capturing cross-modal mapping relationships. The second group is graph-based methods. MMGCN (Hu et al. 2021) leveraged both multimodal information and long-distance contexts for efficient emotion learning. AdaIGN (Tu et al. 2024) designed a graph interaction method to balance intra- and inter-speaker context dependencies for MSA task. However, most of them only consider atomic information to model the correlations contained in different modalities. Therefore, our work aims to emphasize the importance of more complex compositional information and the necessity of a combination of atomic-level and composition-level correlations.

Multimodal Fusion Methods. Early methods can be broadly categorized into two groups: aggregation-based methods (Hazirbas et al. 2017; Zeng et al. 2019; Valada, Mohan, and Burgard 2020; Colombo et al. 2021; Song et al. 2020) and methods that employ Optimal Transport (OT) (Chen et al. 2020; Pramanick, Roy, and Patel 2021; Zhou, Fang, and Feng 2023; Xu and Chen 2023). In the former, separate representations are learned for each modality, and these learned representations from different modalities are directly aggregated. However, these approaches lack effective inter-modal communication. These methods overlook the intra-modal characteristics by simply aligning distributions (Song et al. 2020). While some approaches (Ju et al. 2021; Han, Chen, and Poria 2021) attempted to combine both aggregation and alignment, they often require intricate hierarchical design, which can introduce additional computational costs and engineering complexity. OT-based works aim to achieve balanced feature alignment using the optimal transport method. CMOT (Zhou, Fang, and Feng 2023) conducted cross-modal mixup via optimal transport. OT-Coattn $\mathrm { \Delta X u }$ and Chen 2023) designed OT-based co-attention for structural interactions in survival prediction tasks. However, it is still a challenge to apply those methods for correlation learning. In this work, we propose a Graph-based optimal transport module to assist the model in learning semantic

HCMNet Atomic-level Correlation Modeling Atomic-level DC √   
Video EFxetratauctroer Video Features AMFttuoesnditouilnoen Product Video-Audio Lv2t Similarity e Emotion Aggregator 有德 P²at Pat Similarity Lv2a Similarity 国 Cross Entropy Loss Class-aware   
Text Inner PMt Atomic-level Contrastive C V Fusion Q Product PA Loss Text Features Semantic Transportation Composition-level Correlation Modeling PC Lce   
Audio Extractor Video Graph Graph Convolutional ? Gu -→ G'u 9 GrPaapihr-Fwuiseion Optimal ... [pp pgpε] D Guidance Learning Objectives Audio Features -》 9t -→ g't 9 ， Lacl = Lv2t +Lv2a+La2t Video nodes $\oplus$ Summation Text Graph 》 ga → G'a 9t -Ht2 ? L_Lacl+Lce/lCacl/Lcell ATeuxdtionondoedses $\otimes$ $\circledcirc$ ICnonecratpernoadtiuoctn Composition-level DC 9aHt2a Cost Matrices Fusion Audio Graph

interactions and facilitate cross-modal communication.

# Method

# Preliminaries

Multimodal sentiment analysis aims to identify the specific emotion category to which a given sample (comprising video, text, and audio components) belongs. Formally given a multimodal feature vectors ${ \bf { X _ { i } } } = \{ { \bf { X _ { i } ^ { v } } } , { \bf { X _ { i } ^ { t } } } , { \bf { X _ { i } ^ { a } } } \}$ , where $v , t , a$ denote video, text, and audio modality, respectively, the goal of multimodal sentiment analysis is to predict the emotion label $y _ { i } ^ { g t }$ of the input feature vectors.

# Feature Extraction

Given an input that contains three modalities $( \mathbf { X _ { i } ^ { v } } , \mathbf { X _ { i } ^ { t } } , \mathbf { X _ { i } ^ { a } } )$ , we first employ the pre-trained RoBERTa (Liu et al. 2019) to produce a feature representation for each word token, denoted as $\mathbf { I _ { i } ^ { T } } = [ t _ { 1 } , t _ { 2 } , . . . , t _ { n } ]$ , where $n$ is the number of word tokens and $\mathbf { I _ { i } ^ { T } } \in \mathbb { R } ^ { n \times d _ { t } }$ . To learn contextual information, we follow previous work (Hu et al. 2022) to concatenate the current utterance with its former and latter 2-turn utterances. For video modality, we follow the previous insights (Shi and Huang 2023; Hu et al. 2022) and extract video visual features $\mathbf { \bar { I } _ { i } ^ { V } } \in \mathbb { R } ^ { T \times d _ { v } }$ by employing the pre-trained efficientNet (Tan and Le 2019). Moreover, we feed treated features to additional Multilayer Perceptron (MLP) to model the frame importance and relationships contained in videos. As for the audio modality, we extract Mel-spectrogram sequential vectors by utilizing librosa toolkit 1 and fully-connected layers as audio features IA Rk×da.

# Atomic-level Correlation Modeling

To model the atomic-level correlations, we first encode each modality, mapping them into the same semantic space, and learn atomic-level correlation by dynamic attention reasoning method. After that, we introduce atomic-level contrastive learning to constrain the inconsistent distributions of each modality. The corresponding details are shown in the top branch of Figure 2.

For modality encoding, we employ the pre-trained videotext-audio CLIP (Guzhov et al. 2022) to encode the different unimodal features with modality-specific heads. Specifically, we use pre-trained CLIP (Radford et al. 2021) for visual and textual encoding, as well as pre-trained ESResNeXt (Guzhov et al. 2022) for audio encoding. Therefore, we can obtain the refined unimodal features. Moreover, we employ an attention Module for text features and a cross-frame Fusion Module for video and audio features to learn the longterm dependency in each modality.

$$
\begin{array} { r l } & { \mathcal { T } ^ { i } = \mathbf { W _ { t } } ( \mathcal { F } _ { d i r e c t } ^ { t } ( [ \mathbf { I _ { i } ^ { T } } \oplus \alpha _ { \mathbf { t } } ] ) + \mathbf { b _ { t } } } \\ & { \mathcal { V } ^ { i } = \mathbf { W _ { v } } ( \mathcal { F } _ { c r o s s } ^ { v } ( [ \mathbf { I _ { i } ^ { v } } \oplus \beta _ { \mathbf { v } } ] ) + \mathbf { b _ { v } } } \\ & { \mathcal { A } ^ { i } = \mathbf { W _ { a } } ( \mathcal { F } _ { c r o s s } ^ { a } ( [ \mathbf { I _ { i } ^ { a } } \oplus \beta _ { \mathbf { a } } ] ) + \mathbf { b _ { a } } } \end{array}
$$

where $W _ { * }$ and $b _ { * }$ are the learnable matrices, $\alpha$ and $\beta _ { * }$ are the [CLS] vector. After learning the long-term dependencies, we can get the final video features $\mathcal { V } ^ { i } \stackrel {  } { = } \{ \mathcal { V } _ { ( 1 ) } ^ { i } , \dot { \mathcal { V } } _ { ( 2 ) } ^ { i } , . . . , \mathcal { V } _ { ( M ) } ^ { i } \}$ , text features $\mathcal { T } ^ { i } = \{ \mathcal { T } _ { ( 1 ) } ^ { i } , \mathcal { T } _ { ( 2 ) } ^ { i } , . . . , \mathcal { T } _ { ( M ) } ^ { i } \}$ and audio features $\mathcal { A } ^ { i } = \{ \mathcal { A } _ { ( 1 ) } ^ { i } , \mathcal { A } _ { ( 2 ) } ^ { i } , . . . , \mathcal { A } _ { ( M ) } ^ { i } \}$ , respectively.

To model the atomic-level correlation of $\mathbf { V _ { i } } , \mathbf { T _ { i } }$ and $\mathbf { A _ { i } }$ , we propose a dynamic attention reasoning process to apply cross-attention mechanisms to dynamically integrate different modalities, which are defined as:

$$
\mathbf { h } _ { i } ^ { m } = s o f t m a x \bigl ( \frac { I _ { i } ^ { m } \mathcal { T } _ { i } ^ { \top } } { \sqrt { d _ { c t n } } } \bigr ) \mathcal { T } _ { i } + \lambda \alpha _ { t } \bigl ( \beta _ { v } + \beta _ { a } \bigr )
$$

where $I _ { i } ^ { m }$ denotes input modality and $m \in \{ v , a \}$ . $h _ { i } ^ { m } \in$ $\{ 1 , 2 , . . . , M _ { c t n } \}$ is the i-th attention head, and $M _ { c t n }$ is hyperparameter of the number of cross-attention network. $\lambda$ is the CLS coefficient. Therefore, the atomic-level features are:

$$
\left\{ \begin{array} { l l } { \widetilde { T _ { i } ^ { c } } = \{ \mathcal { T } _ { ( l ) } ^ { c } \} _ { l = 1 } ^ { M } } \\ { \widetilde { \mathcal { V } _ { i } ^ { c } } = \mathcal { V } _ { i } ^ { c } + P r o j ( \{ \mathbf { h } _ { i } ^ { v } \} } \\ { \widetilde { A _ { i } ^ { c } } = \mathcal { A } _ { i } ^ { c } + P r o j ( \{ \mathbf { h } _ { i } ^ { a } \} } \end{array} \right.
$$

Then, the video and audio features are individually fused with text features using the element-wise inner product, yielding the final atomic-level representation $\mathsf { \bar { P } } ^ { a t } ] \in \mathbb { R } ^ { 2 M \times d _ { m } }$ , where $\mathcal { P } ^ { v t } \ \stackrel {  } { = } \ \{ \mathcal { P } _ { ( 1 ) } ^ { v t } , \mathcal { P } _ { ( 2 ) } ^ { v t } , . . . , \mathcal { \dot { P } } _ { ( M ) } ^ { v t } , \}$ $\mathcal { P } ^ { A } = \left[ \mathcal { P } ^ { v t } \oplus \right]$ $\mathcal { P } ^ { a t } = \{ \mathcal { P } _ { ( 1 ) } ^ { a t } , \mathcal { P } _ { ( 2 ) } ^ { a t } , . . . , \mathcal { P } _ { ( M ) } ^ { a t } , \} \in \mathbb { R } ^ { M \times \dot { d } _ { m } }$ .

$$
\mathcal { P } ^ { m ^ { \prime } } = \{ \sum _ { l = 1 } ^ { M } I _ { ( l ) } ^ { m ^ { \prime } } \mathcal { T } _ { ( i ) } ^ { c } \} _ { i = 1 } ^ { M }
$$

where $m ^ { \prime } \in \{ v t , a t \}$ and $I _ { i } ^ { m ^ { \prime } } \in \{ \widetilde { \mathcal { V } } _ { i } ^ { c } , \widetilde { \mathcal { A } } _ { i } ^ { c } \} ,$ $\mathcal { P } ^ { m ^ { \prime } }$ denotes the final atomic-level representations.

Atomic-level Distribution Constraint. Most previous works obtain multimodal features by directly fusing different unimodal representations. However, due to the semantic gaps among diverse modalities, a straightforward fusion approach may potentially elevate the vagueness of emotional information. Therefore, we propose to use contrastive learning for feature semantic distribution constraint, named Atomic-level Contrastive Learning (ACL). For the multimodal samples that have the same emotion class, we hope that they are in the same semantic space, and vice versa. Therefore, our ACL method can perform a unified embedding process with a contrastive learning manner, reducing the distributional inconsistency at the atomic level. Ultimately, given the encoded features $\widetilde { \mathcal { V } _ { i } ^ { c } }$ , $\widetilde { \mathcal { T } _ { i } ^ { c } }$ , and $\widetilde { \mathcal { A } } _ { i } ^ { c }$ , we can define the cosine similarity loss funcftion $\ell _ { A C L }$ beftween video, text, and audio representations as below:

$$
\mathcal { L } _ { a c l } = \underbrace { \frac { < \widetilde { \mathcal { V } _ { i } ^ { c } } , \widetilde { \mathcal { T } _ { i } ^ { c } } > } { \| \widetilde { \mathcal { V } _ { i } ^ { c } } \| \| \widetilde { \mathcal { T } _ { i } ^ { c } } \| } } _ { \mathcal { L } _ { v 2 t } } + \underbrace { \frac { < \widetilde { \mathcal { V } _ { i } ^ { c } } , \widetilde { \mathcal { A } _ { i } ^ { c } } > } { \| \mathcal { V } _ { i } ^ { c } \| \sqrt { \mathcal { A } _ { i } ^ { c } } \| } } _ { \mathcal { L } _ { v 2 a } } + \underbrace { \frac { < \widetilde { \mathcal { A } _ { i } ^ { c } } , \widetilde { \mathcal { T } _ { i } ^ { c } } > } { \| \widetilde { \mathcal { A } _ { i } ^ { c } } \| \| \widetilde { \mathcal { T } _ { i } ^ { c } } \| } } _ { \mathcal { L } _ { a 2 t } }
$$

Therefore, as is defined in $\mathcal { L } _ { a c l }$ , for a multimodal pair, the small similarity means the features should be accordingly pushed away. Otherwise, they should be pulled close.

# Composition-level Correlation Modeling

The composition-level correlation modeling considers the more complex structure of different modalities. To achieve that, we introduce topological graph reasoning, which utilizes topology graph structures in each modality to capture inter-modal correspondence most related to sentiment. Specifically, we first construct the topology graph structures for the textual, visual, and audio modalities. Then, we model the three different graphs with graph convolution network (GCN) (Kipf and Welling 2016). The details are shown in the bottom branch of Figure 2. Specifically, for the text graph $\mathcal { G } _ { t } = ( \nu _ { \mathbf { t } } , \delta _ { \mathbf { t } } )$ , we consider tokens in the text features as graph nodes $\nu _ { \mathrm { t } }$ and employ dependency relations between words extracted by spaCy2 as graph edges $\delta _ { \mathbf { t } } \ \in \ \mathbb { R } ^ { M \times M }$ , which have been emphasized to be resultful for various graph learning process. For constructing video graphs $\mathcal { G } _ { v } \doteq ( \bar { \mathcal { V } } _ { \bf v } , \delta _ { \bf v } )$ , we build edges between each utterance according to the cosine similarity of representations. As is defined in Equation 8, if the cosine similarity $\sigma _ { i j }$ between two utterances is greater than the threshold $\eta$ , we establish an edge between the two utterances. For audio graph learning, we follow a simple utterance-to-node transformation, where $M$ utterances that are short and overlapping segments of the audio signal are the $M$ nodes in an audio graph $\mathcal { G } _ { a } = ( \nu _ { \mathbf { a } } , \delta _ { \mathbf { a } } )$ . We simply utilize the line graph edge definition to construct the audio graph, which connects each utterance with the sequence order.

$$
\delta _ { v } ~ = ~ \left\{ \begin{array} { l l } { \sigma _ { i j } } & { , i f ~ \sigma _ { i j } ~ > ~ \eta } \\ { 0 } & { , o t h e r w i s e } \end{array} \right.
$$

Consequently, given a set of graphs $( \mathcal { G } _ { t } , \mathcal { G } _ { v }$ , and $\mathcal { G } _ { a }$ ), we resort to the graph convolution network (GCN) to mine the inherent relationship and learn the multi-modal composition-level semantics. We aim to learn the node features with their neighborhoods for fine-grained semantic mixing. The detailed formulas are defined below:

$$
\left\{ \begin{array} { l l } { { \displaystyle { \gamma } _ { \bf t } ^ { { \bf k } ^ { \prime } } ~ = ~ \mathrm { R e } L U \left( \tilde { \delta } _ { \bf t } { \bf \gamma } \rangle _ { t } ^ { k - 1 } { \bf W } _ { \bf t } ^ { { \bf k } } + { \bf b } _ { \bf t } ^ { { \bf k } } \right) } } \\ { { \displaystyle { \gamma } _ { \bf v } ^ { { \bf k } ^ { \prime } } ~ = ~ \mathrm { R e } L U \left( \tilde { \delta } _ { \bf v } { \bf \gamma } \rangle _ { v } ^ { k - 1 } { \bf W } _ { \bf v } ^ { { \bf k } } + { \bf b } _ { \bf v } ^ { { \bf k } } \right) } } \\ { { \displaystyle { \gamma } _ { \bf a } ^ { { \bf k } ^ { \prime } } ~ = ~ \mathrm { R e } L U \left( \tilde { \delta } _ { \bf a } { \bf \gamma } \rangle _ { a } ^ { k - 1 } { \bf W } _ { \bf a } ^ { { \bf k } } + { \bf b } _ { \bf a } ^ { { \bf k } } \right) } } \end{array} \right.
$$

$$
\tilde { \delta } _ { m } ~ = ~ \left( \bf { D } _ { m } \right) ^ { - \frac { 1 } { 2 } } \delta _ { m } \left( \bf { D } _ { m } \right) ^ { - \frac { 1 } { 2 } }
$$

where $\tilde { \delta } _ { m }$ is the normalized symmetric adjacency matrix, $D _ { m }$ is the degree matrix of adjacency matrix $\delta _ { m }$ , $\dot { \mathcal { V } } _ { t } ^ { k ^ { \prime } }$ is the $\boldsymbol { k } ^ { t h }$ process of GCNs. $k \in [ 1 , M _ { g c n } ]$ , $m \in \{ t , v , a \}$ denotes the different modalities. $\mathbf { W _ { m } ^ { 1 } } \in \mathbb { R } ^ { d _ { h } \times d _ { h } }$ is the weight matrix and ${ \bf b _ { m } ^ { l } } \in \mathbb { R } ^ { d _ { h } }$ is the bias matrix. $l$ is the hyperparameter of the number of GCN layers.

Composition-level Distribution Constraint. Next, we design the composition-level optimal transport module to learn the semantic distribution for video-text and audio-text groups, respectively. Different from the previous works $\mathrm { \Delta X u }$ and Chen 2023; Zhou, Fang, and Feng 2023), it is necessary to model the correlation between different pairs of modality. Therefore, we propose to learn multimodal descriptors by integrating video-audio graphs with text graphs under the guidance of transferred cost matrices obtained in the optimal transport process. Specifically, we first feed the final outputs of the GCN layers, $\mathcal { G } _ { m } = \{ g _ { m } ^ { ( 1 ) } , g _ { m } ^ { ( 2 ) } , . . . , g _ { m } ^ { ( M ) } \}$ into the optimal transport module. Formally, the optimal transport is conducted on audio-to-text and video-to-text and is defined by the discrete Kantorovich formulation to search the optimal semantic flows $\mathcal { H } _ { m ^ { \prime } }$ between graph $\mathcal { G } _ { v } \ \in \ \mathbb { R } ^ { M \times d _ { m } ^ { \star } }$ , $\mathcal { G } _ { a } \in \mathbb { R } ^ { M \times d _ { m } }$ and $\mathcal { G } _ { t } \in \mathbb { R } ^ { M \times d _ { m } }$ :

$$
{ \mathcal W } ( { \mathcal G } _ { m ^ { \prime } } , { \mathcal G } _ { t } ) = \operatorname* { m i n } _ { \substack { \mathcal { H } _ { m ^ { \prime } } \in \prod ( \mu _ { m ^ { \prime } } , \mu _ { t } ) } } < { \mathcal H } _ { m ^ { \prime } } , { \mathcal C } _ { m ^ { \prime } } >
$$

$$
\mathcal { G } _ { m ^ { \prime } } ^ { ' } = \mathcal { G } _ { m ^ { \prime } } + \gamma _ { m ^ { \prime } } \mathcal { H } _ { m ^ { \prime } } ^ { \top } \mathcal { G } _ { m ^ { \prime } }
$$

where $m ^ { \prime } \in \{ v , a \}$ and $\mathcal { C } _ { m ^ { \prime } }$ denote the cost matrices and are defined with Euclidean distance that measures the distance of local pair-wise instances of $\mathcal { G } _ { m } . \mu _ { m ^ { \prime } }$ and $\mu _ { t }$ are the marginal distributions. $\gamma$ is the adaptive graph coefficient.

After obtaining the transport flows, the updated graphs can be defined as Equation 12. To reduce modality complexity and obtain semantic-consistent multimodal features, we feed the processed graph representations into a Pair-wise Graph Fusion module, an $M _ { c }$ -layer attention network, where $M _ { c }$ is the hyper-parameter. This allows us to learn the semantic dependencies between visual-acoustical and textual features. We concate visual graph $\mathcal { G } _ { v } ^ { ' }$ and audio graph $\mathcal { G } _ { a } ^ { ' }$ as query and key, and text graph as value for attention score calculation, which can be defined as the following equation.

$$
\mathcal { G } _ { v a } = s o f t m a x ( \frac { \mathbf { Q _ { v + a } K _ { v + a } ^ { \top } } } { \sqrt { d _ { a t t n } } } ) \mathbf { V _ { t } }
$$

$$
\mathcal { P } ^ { u } = \mathbf { W _ { 1 } } ( \mathbf { W _ { 2 } } \mathcal { G } _ { v a } + \mathbf { b _ { 2 } } ) + \mathbf { b _ { 1 } }
$$

where $\bf { W _ { 1 } } , \psi \bf { W _ { 2 } }$ are the weight parameters of feedforward layers, and $\bf { b _ { 1 } }$ and $\mathbf { b _ { 2 } }$ are the bias parameters. $\mathbf { Q _ { v + a } } , \mathbf { K _ { v + a } } = \mathbf { W _ { q } } [ \mathcal { G } _ { v } ^ { ' } \oplus \mathcal { G } _ { a } ^ { ' } ]$ and $\mathbf { V _ { t } } = W _ { v } \mathcal { G } _ { t }$ .

We fuse the visual-acoustical features $\mathcal { P } ^ { u } \in \mathbb { R } ^ { M \times d _ { m } }$ with transferred cost matrix $\mathcal { C } ^ { \prime } \in \mathbb { R } ^ { M \times M }$ to generate the final multimodal descripors ${ \mathcal { P } } ^ { C }$ . We conduct element-wise addition fusion to calculate the transferred cost matrix: ${ \mathcal { C } } ^ { \prime } =$ $\mathcal { C } _ { v } + \mathcal { C } _ { a }$ . The intention is to retrieve the crucial representations in inter- and intra-graph. The formula is as follows:

$$
\mathcal { P } ^ { C } = \mathcal { C } ^ { \prime } \odot \mathcal { P } ^ { u } \in \mathbb { R } ^ { M \times d _ { m } }
$$

where $\odot$ denotes vector inner-product and $M$ is the number of visual-acoustical features, $d _ { m }$ is the dimension of $\mathcal { P } ^ { u }$ .

# Learning Objectives

To perform emotion analysis with dual-level features, we concatenate both the atomic- and composition-level features into the final feature representations for prediction. The Emotion Classifier contains a Class-aware Cross-Entropy Loss function $\mathcal { L } _ { c c e }$ between multimodal representations and emotional labels as is defined below:

$$
\mathcal { L } _ { c c e } = - \sum \log \mathbf { P } _ { i } \left( y = i | [ \mathcal { P } ^ { A } \oplus \mathcal { P } ^ { C } ] \right) + \xi | | \theta | | _ { 2 } ^ { 2 }
$$

As for the learning objective, we introduce both the Atomic-level Contrastive Learning Loss $\mathcal { L } _ { A C L }$ in Equation 7 and Class-aware Cross-Entropy Loss $\mathcal { L } _ { c c e }$ in Equation 16 for correlation learning. To balance the difference in the algebraic scale of the two losses, we adopt an adaptive loss formula. The detailed formula is as follows:

$$
\mathcal { L } = \mathcal { L } _ { a c l } + \mathcal { L } _ { c c e } / \parallel \mathcal { L } _ { a c l } / \mathcal { L } _ { c c e } \parallel
$$

where the $| | \cdot | |$ represents the truncated gradient operator, which calculates the adaptive balance coefficient of losses.

<html><body><table><tr><td>Dataset</td><td>Train</td><td>Valid</td><td>Test</td><td>All</td></tr><tr><td>MELD</td><td>9989</td><td>1108</td><td>2610</td><td>13707</td></tr><tr><td>IEMOCAP</td><td>5354</td><td>528</td><td>1650</td><td>7532</td></tr><tr><td>CMU-MOSEI</td><td>16326</td><td>1871</td><td>4659</td><td>22856</td></tr></table></body></html>

Table 1: The details of CMU-MOSEI, MELD, and IEMOCAP, including data splitting details.

# Experimental Setup

# Experimental Settings

Datasets. We assess the performance of our method on multimodal sentiment analysis benchmark datasets, including IEMOCAP (Busso et al. 2008) and MELD (Poria et al. 2018) and CMU-MOSEI (Zadeh et al. 2018) datasets. The statistics are reported in Table 1. Both of them are multimodal datasets with textual, visual, and acoustic modalities.

Implementation Details. For a fair comparison, we follow previous works (Hu et al. 2022; Ma et al. 2022) to pre-process the datasets and use the same dataset split. We employ ResNet-50 CLIP (Radford et al. 2021) for the visual and textual head, and ESResNeXt initialized on pretrained datasets for the acoustic head. The number of training epochs is 100. We set the batch size as 64 for three datasets. We utilize AdamW as the optimizer with an initial learning rate of $2 \times 1 0 ^ { - 4 }$ . The dropout rate is set to 0.1 to avoid overfitting. The number of GCN layers is set to 4 as default. We use an 8-layer PGF module for graph fusion. All the experiments are carried out on NVIDIA RTX3090 GPUs. We use the weighted-F1 (w-F1) score as evaluation metrics for IEMOCAP and MELD datasets. For the CMUMOSEI dataset, we adopt mean absolute error (MAE), Pearson correlation (Corr), accuracy (Acc), and F1-score as metrics. More details about datasets are provided in Appendix.

# Baseline Models

To validate the effectiveness of HCMNet, we compared it with several state-of-the-art baselines. DialogueRNN (Majumder et al. 2019) and DialogueGCN (Ghosal et al. 2019) are dialogue-based models. They learn context information using recurrent networks and directed graphs. ICCN (Sun et al. 2020) learns correlations between three modalities via deep canonical correlation analysis. IterativeERC (Lu et al. 2020) enhances emotion interactions by using predicted emotion labels. MMIM (Han, Chen, and Poria 2021) hierarchically maximizes the Mutual Information in unimodal input pairs for the MSA task. MMGCN (Hu et al. 2021) leverages both multimodal information and long-distance contexts for efficient emotion learning. UniMSE (Hu et al. 2022) obtains multimodal features that are fused by integrating audio and vision representations into language models. i-Code (Yang et al. 2023) designs an integrative and composable multimodal learning framework for triple-modal learning. MultiEMO (Shi and Huang 2023) integrates multimodal cues by capturing cross-modal mapping relationships. AdaIGN (Tu et al. 2024) proposes a new adaptive graph learning for cross-modal interaction.

<html><body><table><tr><td rowspan="2">Methods</td><td colspan="7">IEMOCAP</td><td colspan="8">MELD</td></tr><tr><td>Happy</td><td>Sad</td><td>Neutral</td><td>Anger</td><td>Excited</td><td>Frustrated</td><td>w-F1</td><td>Neutral</td><td>Surprise</td><td>Fear</td><td>Sad</td><td>Joy</td><td>Disgust</td><td>Angry</td><td>w-F1</td></tr><tr><td>DialogueRNN</td><td>33.18</td><td>78.80</td><td>59.21</td><td>65.28</td><td>71.86</td><td>58.91</td><td>62.75</td><td>76.23</td><td>49.59</td><td>0.00</td><td>26.33</td><td>54.55</td><td>0.81</td><td>46.76</td><td>58.73</td></tr><tr><td>DialogueGCN</td><td>51.87</td><td>76.76</td><td>56.76</td><td>62.26</td><td>72.71</td><td>58.04</td><td>63.16</td><td>76.02</td><td>46.37</td><td>0.98</td><td>24.32</td><td>53.62</td><td>1.22</td><td>43.03</td><td>57.52</td></tr><tr><td>IterativeERC</td><td>53.17</td><td>77.19</td><td>61.31</td><td>61.45</td><td>69.23</td><td>60.92</td><td>64.37</td><td>77.52</td><td>53.65</td><td>3.31</td><td>23.62</td><td>56.63</td><td>19.38</td><td>48.88</td><td>60.72</td></tr><tr><td>MMGCN</td><td>42.34</td><td>78.67</td><td>61.73</td><td>69.00</td><td>74.33</td><td>62.32</td><td>66.22</td><td>-</td><td>-</td><td></td><td></td><td>-</td><td></td><td>-</td><td>58.65</td></tr><tr><td>UniMSE</td><td></td><td>-</td><td></td><td>=</td><td></td><td>-</td><td>70.66</td><td>-</td><td></td><td></td><td>-</td><td></td><td>-</td><td>-</td><td>65.51</td></tr><tr><td>MultiEMO</td><td>65.77</td><td>85.49</td><td>67.08</td><td>69.88</td><td>77.31</td><td>70.98</td><td>72.84</td><td>79.95</td><td>60.98</td><td>29.67</td><td>41.51</td><td>62.82</td><td>36.75</td><td>54.41</td><td>66.74</td></tr><tr><td>AdaIGN</td><td>53.04</td><td>81.47</td><td>71.26</td><td>65.87</td><td>76.34</td><td>67.79</td><td>70.74</td><td>1</td><td></td><td>-</td><td>-</td><td></td><td>-</td><td>1</td><td>1</td></tr><tr><td>HCMNet</td><td>67.86</td><td>86.37</td><td>69.74</td><td>72.38</td><td>78.19</td><td>71.33</td><td>74.62</td><td>82.09</td><td>63.87</td><td>30.31</td><td>45.86</td><td>64.77</td><td>39.07</td><td>57.15</td><td>68.94</td></tr></table></body></html>

Table 2: Results on IEMOCAP and MELD. The best and secondary performances are in bold and underlined, respectively.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="5">CMU-MOSEI</td></tr><tr><td>MAE↓</td><td>Corr ↑</td><td>Acc-7个</td><td>Acc-2↑</td><td>F1↑</td></tr><tr><td>ICCN</td><td>0.565</td><td>0.704</td><td>51.60</td><td>84.20</td><td>84.20</td></tr><tr><td>MMIM</td><td>0.526</td><td>0.772</td><td>54.24</td><td>85.97</td><td>85.94</td></tr><tr><td>UniMSE</td><td>0.523</td><td>0.773</td><td>54.39</td><td>87.50</td><td>87.46</td></tr><tr><td>i-Code</td><td>0.502</td><td>0.811</td><td>50.80</td><td>87.50</td><td>87.40</td></tr><tr><td>HCMNet</td><td>0.489</td><td>0.819</td><td>54.97</td><td>87.63</td><td>87.56</td></tr></table></body></html>

Table 3: Results on MOSEI dataset. The best and secondary performances are in bold and underlined, respectively.

# Results and Analysis

# Main Results

We compare our HCMNet model with the existing methods. To verify the capability of multi-granularity multimodal sentiment representation, we design experiments carried out on IEMOCAP and MELD datasets that have fine-grained sentiment categories for multimodal sentiment analysis. Further, we conduct both 7-class and 2-class sentiment analysis experiments on the CMU-MOSEI dataset. Table 2 and Table 3 show the comparison results on the three datasets. We can obtain the following conclusions: (1) Our method achieves the best performance across three datasets, demonstrating the effectiveness of learning hierarchical correlations with distribution constraints for MSA task. Specifically, HCMNet obtains $2 . 4 4 \%$ and $3 . 3 0 \%$ overall Weighted-F1 improvement compared with MultiEmo (Shi and Huang 2023) in IEMOCAP and MELD datasets, respectively. (2) Table 3 shows the detailed performance comparison on the CMU-MOSEI dataset. We note that our method achieves mediocre accuracy and F1-score improvement, which indicates that there is potential space for HCMNet to optimize the emotion representation process for the CMU-MOSEI dataset. (3) The F1 score of MELD dataset is significantly lower than the F1-score on IEMOCAP dataset. The possible reason is that IEMOCAP is collected from well-designed utterances that have selected scripts with clear emotional content.

# Ablation Study

To investigate the effectiveness of the different components and settings, we introduce several variants of our method for comparison on IEMOCAP and MELD datasets.

(1) Analysis of model components. We compared HCMNet with the following derivations. ACM-only contains a single atomic-level correlation modeling module. CCMonly, which mainly utilizes graph learning and compositionlevel optimal transport to obtain multimodal features. w/o ACL means we remove the Atomic-level Contrastive Learning from the ACM module, which aims to explore the impact of atomic-level distribution constraint. w/o graph learning: we replace graph learning with untreated features gained from the ACM module directly. To testify to the effectiveness of optimal transport in composition-level correlation modeling, we remove the composition-level distribution constraint, which is w/o CDC. The results are shown in Table 4 and we make the following observations: First, compared with the variant that only uses the CCM module, ACM-only model which utilizes atomic-level correlation modeling and distribution constraint for MSA task gains more positive results on both IEMOCAP and MELD datasets. Therefore, fine-grained contrastive learning and element-wise fusion help a lot when learning the clarity correlations. Second, Atomic-level Contrastive Learning (ACL) plays an important role in catching the semantics that are exploited for particle semantic aligning in subsequent features representation process. Third, it is more befitting to employ an efficient graph learning approach to obtain structural semantic information. We can infer that compositionlevel distribution constraint balances different semantic distributions between different modality pairs.

Table 4: Experiment results of ablation study. We compare different variants on IEMOCAP and MELD datasets.   

<html><body><table><tr><td rowspan="2">Variant Model</td><td>IEMOCAP</td><td>MELD</td></tr><tr><td>ACC w-F1</td><td>ACC w-F1</td></tr><tr><td>HCMNet (ours)</td><td>73.86 74.62</td><td>68.31 68.94</td></tr><tr><td>ACM-only</td><td>70.13 70.25</td><td>67.97 68.36</td></tr><tr><td>CCM-only</td><td>68.55 68.76</td><td>67.24 67.31</td></tr><tr><td>w/o ACL w/o graph learning</td><td>69.71 70.22</td><td>67.08 67.45</td></tr><tr><td>w/o CDC</td><td>69.82 70.18 68.83 69.52</td><td>67.33 67.74 66.67 67.49</td></tr></table></body></html>

(2) Analysis of different modality settings. We conduct an ablation experiment with four modality settings on IEMOCAP and MELD datasets, which gives us more insights and inspiration about the MSA task and the attribute of the dataset. Table 5 shows the experiment result. We can infer the following conclusions. First, performances on both two datasets vary with different modality settings. As shown in

Table 5: Ablation study of HCMNet with different modalities on IEMOCAP and MELD datasets. T, V and A represent textual, visual and acoustic modalities, respectively.   

<html><body><table><tr><td rowspan="2">Modality</td><td colspan="2">IEMOCAP</td><td colspan="2">MELD</td></tr><tr><td>ACC</td><td>W-F1</td><td>ACC</td><td>w-F1</td></tr><tr><td>A+T</td><td>71.20</td><td>71.71</td><td>65.09</td><td>65.12</td></tr><tr><td>V+T</td><td>72.61</td><td>73.22</td><td>66.98</td><td>67.19</td></tr><tr><td>V+A</td><td>71.96</td><td>72.07</td><td>65.42</td><td>66.48</td></tr><tr><td>V+A+T</td><td>73.86</td><td>74.62</td><td>68.31</td><td>68.94</td></tr></table></body></html>

![](images/672c6dc913488bbe5c2b52c3ddbaf5f24d0d6fc883abe29861b927e1f465052d.jpg)  
Figure 3: Parameter sensitive analysis of the number of GCN (left) and PGF layers (right).   
Figure 5: Case study of IEMOCAP dataset. We provide a sample to demonstrate the effectiveness of our method.

Table 5, experiment items with textual modality generally gain satisfactory scores, indicating that dialogue content is more important in multimodal emotion prediction. Specifically, compared with $\mathbf { V } { + } \mathbf { A }$ , the combination of $\mathbf { V + A + T }$ achieves $2 . 6 4 \%$ and $4 . 0 3 \%$ weighted-F1 increase on the two datasets, respectively. Moreover, for two-modality settings, it can be observed that $\mathbf { V } { + } \mathbf { T }$ is more promising in emotion semantic modeling compared to others.

# Parameter Sensitivity Analysis

We evaluate the effect of the number of GCN and PGF layers. As is shown in Figure 3. We can observe that as the number of GCN and PGF layers increases, the model’s performance improves steadily until reaching its peak, after which it starts to decline. Therefore, we can infer that: (1) Our model works best when the layers parameter is set to 4 for GCN and 8 for the PGF module. (2) It is ideal to choose fewer GCN layers and more PGF layers for better performance, indicating that graph learning used at the top of the module is less demanding than semantic-level graph fusion.

# Qualitative Analysis

Visualization analysis. In Figure 4, we visualize the latent features obtained from HCMNet to explore how it works for MSA task. (1) The results reveal that the HCMNet is capable of capturing discriminative features for different categories. The distinct feature clusters for each category indicate that our model learns to highlight and differentiate between specific multimodal features associated with different emotion semantics. (2) Classes with CDC tend to exhibit closer proximity in the latent space, reflecting the model’s ability to align multimodal features with emotional semantic relationships. As shown in Figure 4 (a), many outlier features spread across different feature spaces, thus disturbing the discrim

![](images/2ecc1c7b7a862093256dd34a9f5d1086211957074b5677f66bb3b5209ddb0aee.jpg)  
Figure 4: t-SNE visualization of IEMOCAP dataset. Different colored dots represent samples with different categories.

Raw Sample Annotations Frustrated Neutral Neutral （ （ （ Ground Truth: Frustrated Predictions UniMSE Neutral $\textcircled{8}$ Okay. But I didn't tell you to get in this line MultiEMO $\nRightarrow$ Neutral $\bullet$ if you are filling out this particular form. HCMNet Frustrated

inability of emotion analysis models. When we employ the CDC module in HCMNet, as shown in Figure 4 (b), different emotion features maintain sufficient distances from each other, which provides beneficial partition boundaries.

Case Study. To further demonstrate the effectiveness of our method, we visualize an example of MSA results on the IEMOCAP dataset in Figure 5. Note that the connections in the figure indicates the composition-level correlations of each modality. From the case, we can see two people talking face to face, with confused and frustrated emotions. Compared with previous work UniMSE (Hu et al. 2022) and MultiEMO (Shi and Huang 2023), our method considers both atomic- and composition-level information, thus giving accurate prediction. With hierarchical correlation modeling and semantic distribution constraints, our model can obtain more discriminative features for correlation modeling.

# Conclusion And Future Work

We present a Hierarchical Correlation Modeling Network, enhancing the multimodal sentiment analysis by exploring both atomic- and composition-level correlations. Besides, we propose to perform distribution constraints for dual-level features. Experiments on three public benchmark datasets have demonstrated the superiority of the HCMNet model over previous methods. In future work, we will incorporate contextual knowledge from large vision-language models, thereby better learning emotional correlations.

# Acknowledgments

This work is partially supported by the grants from the Natural Science Foundation of Shandong Province (ZR2024MF145, ZR2024ZD20), and the National Natural Science Foundation of China (62072469).