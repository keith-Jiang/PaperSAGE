# Dual-View Learning for Conversational Emotion Recognition Through Context and Emotion-Shift Modeling

Xupeng Zha, Huan Zhao\*, Guanghui Ye, Zixing Zhang

College of Computer Science and Electronic Engineering, Hunan University, China zhaxupeng, hzhao, yghui, zixingzhang @hnu.edu.cn

# Abstract

Conversational Emotion Recognition (CER) has recently been explored through conversational context modeling to learn the emotion distribution, i.e., the likelihood over emotion categories associated with each utterance. While these methods have shown promising results in emotion classification, they often focus on the interactions between utterances (utterance-view) and overlook shifts in the speaker’s emotions (emotion-view). This emphasis on homogeneous view modeling limits their overall effectiveness. To address this limitation, we propose DVL-CER, a novel Dual-View Learning approach for CER. DVL-CER integrates both the utterance-view and emotion-view using two projection heads, enabling cross-view projection of emotion distributions. Our approach offers several key advantages: (1) We introduce an emotion-view that captures shifts in a speaker’s emotions from initial to subsequent states within a conversation. This view enriches the conversation modeling and supports seamless integration with various CER baseline models. (2) Our dual-view projection learning strategy flexibly balances consistency and independence between the two heterogeneous views, promoting view-specific adaptation learning and incorporating the emotion verification capability within CER. We validate DVL-CER through extensive experiments on two widely-used datasets, IEMOCAP and EmoryNLP. The results demonstrate that DVL-CER achieves state-of-the-art performance, delivering robust and high-quality emotion distributions compared with existing CER methods and other dualview learning strategies.

# 1 Introduction

Conversational Emotion Recognition (CER) is a crucial task in natural language processing that involves identifying the emotion expressed in each utterance within a dialogue. With the increasing prevalence of conversational data and the growing demand for empathetic AI systems, CER has gained significant importance. This research is essential for applications, such as social media opinion analysis (Kumar, Dogra, and Dabas 2015) and emotion-aware chatbots (Chatterjee et al. 2019), highlighting its importance in understanding and recognizing emotional states in conversations.

电 呂 View 1 View 2 View 1 View Network 1 Network 2 Network 1 Network 2 tion (a) Existing dual-view (b) Our dual-view projection learning methods. learning method.

Current CER research focuses heavily on predicting the emotion of each utterance by considering both intra- and inter-speaker emotion dependencies within the conversational context. Common methods employ Recurrent Neural Networks (RNNs) (Majumder et al. 2019; Ghosal et al. 2020; Hu, Wei, and Huai 2021) or Graph Neural Networks (GNNs) (Ghosal et al. 2019; Li et al. 2021; Shen et al. 2021b) to model these dependencies, transforming the semantic content of utterances into emotion representations for learning emotion distributions, i.e., the likelihood over emotion categories for each utterance. Despite some successes, these methods face challenges, including interaction discord and transformation inconsistency. These issues stem primarily from two factors: i) natural language semantics in utterance embeddings do not adequately encode affective meaning due to the distributional hypothesis (Faruqui et al. 2015; Babanejad et al. 2024); and ii) the learned emotion distributions, though effective for emotion prediction, may lack sufficient sensitivity to capture subtle emotional nuances (Felbo et al. 2017; Poria et al. 2020). Consequently, the current focus on context modeling from an utterance perspective limits the potential to capture emotion dependencies.

To overcome these challenges, previous efforts have introduced external knowledge to enhance emotion comprehension in models. For instance, methods like KET (Zhong, Wang, and Miao 2019), which incorporates the NRC VAD emotion lexicon (Mohammad 2018), and COSMIC (Ghosal et al. 2020), which fine-tunes a pre-trained RoBERTa model (Liu et al. 2019) for emotion label prediction, aim to embed affective meaning into utterance embeddings. On the other hand, approaches like SKAIG (Li et al. 2021), COSMIC, and CauAIN (Zhao, Zhao, and Lu 2022) leverage commonsense knowledge from COMET (Bosselut et al. 2019) to mediate interactions between utterances, thereby improving information transfer and the model’s expressive capabilities. However, despite these advances, existing studies have not fully considered and explored emotion dependencies from an emotion-centric perspective. Thus, it is necessary to investigate emotion view modeling to capture these dependencies and interactions more effectively.

Emotion shift, a concept extensively studied in the CER literature, is defined in psychology as the “evolution of emotional experience during exposure to a media message” (Nabi and Green 2015). Current research (Ghosal et al. 2021; Yang et al. 2022; Gao et al. 2022; Bansal et al. 2022; Li et al. 2024) focuses primarily on emotion label shifts rather than on the underlying emotion dynamics. Yang et al. (2022) and Gao et al. (2022) investigate the consistency of emotion representation across consecutive utterances by the same speaker, while other studies (Bansal et al. 2022; Li et al. 2024) explore the probability of emotion label shifts between consecutive or any two utterances, regardless of speaker identity. However, these studies typically address only emotion variables and binary classification problems, neglecting the media message—the sequence of utterances between two consecutive emotions from the same speaker, which limits understanding of emotion shifts. To address this gap, we propose an emotion shift modeling approach that integrates the initial emotion with the utterance sequence to predict future emotions, aiming to explicitly reveal intraspeaker emotion transfer and interactions.

Building on both the conversational context network (utterance-view) and the emotion shift network (emotionview), we propose a hypothesis that the emotion distributions learned from these two views are shared in CER. Under this hypothesis of emotion distribution consistency, we introduce a dual-view projection learning strategy, where two projection heads predict the emotion distributions of one view from another view. This strategy ensures consistency between views by aligning cross-view emotion distributions while preserving view-specific independence. We call the resulting framework DVL-CER, a Dual-View Learning approach for CER, which, to the best of our knowledge, is the first to model an emotion view and to explore category distribution consistency under dual/multi-view supervision learning. This approach distinguishes our work from existing CER solutions, which rely solely on homogeneous utterance-view modeling and representation-level view sharing, as illustrated in Figure 1. Additionally, our framework is agnostic to the conversational context network and can be seamlessly integrated with it.

Our main contributions can be summarized as follows:

• We introduce DVL-CER, a novel dual-view learning framework for CER that enables desirable sharing of emotion distributions between a conversational context network for utterance-view modeling and an emotion shift network for emotion-view modeling through two

projection heads. • DVL-CER exhibits strong generalization ability, allowing seamless integration with various CER baselines. • Our dual-view projection learning strategy effectively balances the trade-off between consistency and independence across two heterogeneous views. • Extensive experiments on the IEMOCAP and EmoryNLP benchmarks show that DVL-CER produces more robust and higher-quality emotion distributions compared with existing CER methods and other dual-view learning strategies, achieving state-of-the-art results.

# 2 Related Works

From the perspective of model architecture, CER methods can be broadly categorized into two groups: single-view and multi-view CER. Multi-view CER methods employ multiple networks, each tailored to a specific interaction perspective, allowing for fine-grained modeling of conversational context. In contrast, single-view CER methods utilize a single network to model the interaction relationships considered within a conversation, simplifying the learning setting.

Single-View CER. Single-view CER methods often employ GNN-based frameworks, representing the conversation as a graph with utterances as nodes, speaker dependencies as edges, and dependency types as edge attributes. Notable examples include DialogueGCN (Ghosal et al. 2019), which models both intra- and inter-speaker dependencies; SKAIG (Li et al. 2021), which enhances edge representations with psychological knowledge; and DialogXL (Shen et al. 2021a), which uses a self-attention mechanism to capture crucial dependencies between speakers.

Multi-View CER. Multi-view CER methods leverage multiple networks to model different interaction perspectives of conversational context. While commonly applied in multimodal CER (Zadeh et al. 2018; Meng et al. 2024), where each modality corresponds to a distinct perspective, multiview learning is also present in textual CER. For example, HMVDM (Ruan et al. 2022) introduces a hierarchical structure to capture token-level and utterance-level dependencies, and MVN (Ma et al. 2022) proposes a dualview learning model that combines an attention mechanism for word-level dependencies with a bidirectional gated recurrent unit for utterance-level dependencies. These methods, despite being termed as “multi-view learning,” share similarities with CER approaches that employ multiple networks for modeling different dependencies or perspectives. Examples include DialogueRNN (Majumder et al. 2019), which models speaker, context, and emotion states with three GRUs; COSMIC (Ghosal et al. 2020), which refines speaker states—context, internal, external, intent, and emotion—using five GRUs; DialogueCRN (Hu, Wei, and Huai 2021), which captures situation-level and speaker-level context with two LSTMs; and DualRAN (Li, Wang, and Zeng 2024), which employs a dual-stream recurrence-attention network to extract contextual information. These methods are, therefore, categorized as multi-view learning methods.

As alluded to earlier, both single-view and multi-view CER methods emphasize conversational context modeling from an utterance perspective. In contrast, this paper introduces the first emotion-view network, focusing on modeling emotion shifts and integrating existing CER networks into a novel heterogeneous dual-view CER framework.

![](images/ee8826dfe5b032f414d070b6567bcd0d3cd0d5b5a572d77163155feda241f139.jpg)  
Figure 2: Illustration of the proposed DLV-CER approach. This approach employs a base network—either based on recurrence or graph networks—to model conversational context (utterance-view) and a Transformer-based network to capture emotion shifts (emotion-view), with both networks aimed at learning emotion distributions. Two projection heads then map these emotion distributions from one view to another, optimized using the Mean Squared Error (MSE) loss function. At the end of training, the emotion-view network and two projection heads are discarded, leaving the utterance-view network and its learned emotion distribution $e _ { \theta }$ for emotion classification. MSE loss functions, indicated by dotted lines of the same colors, are computed.

Multi-View Learning Strategies. In recent decades, multiview learning (Sun 2013; Yan et al. 2021) has gained considerable attention in machine learning and computer vision, inspiring many promising algorithms. Prominent among these are multi-view alignment learning (Radford et al. 2021), multi-view feature aggregation learning (Yang et al. 2018), and multi-view subspace learning (Andrew et al. 2013; Xue et al. 2019). The latter approach, which searches for two projections to filter out view-specific independence while mapping views onto a common low-dimensional subspace to maximize correlation, is particularly relevant to our work. In contrast to this filtering mechanism, our proposed multiview projection learning strategy aims to find two projections that map one view onto the space of another, maximizing correlation while leaving view-specific independence. Furthermore, rather than conventional representation-level sharing, our method strives to a direct sharing that targets task-relevant category distributions.

# 3 Methodology

# 3.1 Task Definition

We define the task of CER as follows: Given a conversation consisting of $N$ consecutive utterances $\begin{array} { r c l } { U } & { = } & { \left[ u _ { 1 } , u _ { 2 } , \dots , u _ { N } \right] } \end{array}$ spoken by $M$ speakers $\begin{array} { r l } { S } & { { } = } \end{array}$ $[ s _ { 1 } , s _ { 2 } , \ldots , s _ { M } ]$ , the objective is to learn the emotion distribution $e _ { i }$ for each utterance $u _ { i }$ , with the corresponding emotion label $y _ { i } \in \{ y _ { 1 } , y _ { 2 } , . . . , y _ { C } \}$ serving as supervision.

In line with prior work by Ghosal et al. (2020), we use the RoBERTa Large model (Liu et al. 2019) to extract an utterance embedding $h _ { i } \in \mathbb { R } ^ { d }$ for each utterance $u _ { i }$ .

The proposed DVL-CER framework comprises three main components: an utterance-view network for modeling conversational context, an emotion-view network for modeling emotion shifts, and two projectors that facilitate crossview correlations, as illustrated in Figure 2.

# 3.2 Conversational Context Modeling

Traditional CER methods often model conversational context from an utterance view to address the task defined earlier. These methods typically employ RNNs or GNNs to transform the utterance embeddings $H ~ \in ~ \mathbb { R } ^ { N \times d }$ into emotion representations $r _ { \theta }$ . These representations are subsequently processed by a Feed-Forward Layer (FFL) to predict emotion distributions $e _ { \theta }$ , guided by the corresponding emotion labels. The process is formalized as:

$$
e _ { \theta } = \mathrm { F F L } ( \mathrm { X } \mathrm { - N N s } ( H ) ) ,
$$

where $e _ { \theta , i } \in \mathbb { R } ^ { | C | }$ represents the predicted emotion distribution for utterance $u _ { i }$ , and $\mathrm { \Delta X }$ -NNs refers to either RNNs or GNNs. The conversational context network is trained using a Cross-Entropy (CE) loss function:

$$
\mathcal { L } _ { C C M } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { c = 1 } ^ { | C | } y _ { i } ^ { c } \log ( \mathrm { s o f t m a x } ( e _ { \theta , i } ) ) ,
$$

where $y _ { i } ^ { c }$ is a one-hot vector for the emo-tion label of utterance $u _ { i }$ , and $c$ denotes the dimension of each emotion label.

Since the subsequent sections focus on the final emotion distribution $e _ { \theta }$ without directly involving the conversational context network, our framework accommodates various choices for the conversational context network without imposing any constraints.

# 3.3 Emotion Shift Modeling

While emotion distributions derived from conversational context modeling are often sufficient for emotion classification, they may lack explicit interactions between emotions, leading to potential inconsistencies and biases that affect prediction accuracy. To address this, we propose emotion shift modeling, which explicitly captures these interactions.

Given a conversation segment $\bar { U _ { w } } = [ u _ { i } , \dotsc , u _ { i + w - 1 } ] \in$ $U$ and an emotion distribution $e _ { \theta , i }$ from the conversational context network, where $u _ { i }$ and $u _ { i + w - 1 }$ are spoken by the same speaker, i.e., $s _ { i } ~ = ~ s _ { i + w - 1 }$ , and the intermediate utterances $[ u _ { i + 1 } , \dotsc , u _ { i + w - 2 } ]$ are from others, with $w$ representing the emotion shift span (i.e., the number of utterances between the speaker’s initial and future emotion), we treat $e _ { \theta , i }$ as the initial emotion and $U _ { w }$ as the medium through which the emotion shift occurs. A Transformer-based model is used to predict the future emotion label $y _ { i + w - 1 }$ .

First, the emotion distribution $e _ { \theta , i }$ is embedded into a vector $h _ { \theta , i } ^ { e }$ of dimension $\mathbb { R } ^ { d }$ through two fully connected layers:

$$
h _ { \theta , i } ^ { e } = W _ { 2 } ( W _ { 1 } e _ { \theta , i } + b _ { 1 } ) + b _ { 2 } .
$$

Here, $W _ { 1 } , W _ { 2 } , b _ { 1 }$ , and $b _ { 2 }$ are trainable parameters.

Next, a special token, [CLS], is prepended to the sequence $\left[ h _ { \theta , i } ^ { e } , h _ { i } , \ldots , h _ { i + w - 1 } \right]$ to serve as an aggregator for both the initial emotion and the medium, forming an input sequence $z _ { i } ~ = ~ [ h _ { [ \mathrm { C L S } ] } , h _ { \theta , i } ^ { e } , h _ { i } , \dots , h _ { i + w - 1 } ] ~ \in ~ \mathbb { R } ^ { ( w + 2 ) \times d }$ . To account for the temporal nature of the emotion shift, position embeddings $p _ { t } \ \in \ \mathbb { R } ^ { ( w + 2 ) \times d }$ , encoded by sine and cosine functions, are added to the input sequence:

$$
z _ { i } ^ { 0 } = z _ { i } + p _ { t } .
$$

The input sequence $z _ { i } ^ { 0 } ~ \in ~ \mathbb { R } ^ { ( w + 2 ) \times d }$ is then processed through a standard Transformer architecture with $L$ layers. The $l$ -th Transformer block is defined as follows:

$$
\begin{array} { r l } & { \tilde { g } _ { i } ^ { l } = \mathbf { M } \mathbf { U } \mathbf { L } \mathbf { T } \mathbf { I } \mathbf { A } \mathbf { T } \mathbf { T } \mathbf { N } ( z _ { i } ^ { l - 1 } ) , } \\ & { g _ { i } ^ { l } = \mathbf { L } \mathbf { A } \mathbf { Y } \mathbf { E } \mathbf { R } \mathbf { N O R } \mathbf { M } ( \tilde { g } _ { i } ^ { l } + z _ { i } ^ { l - 1 } ) , } \\ & { \tilde { z } _ { i } ^ { l } = \mathbf { F } \mathbf { F } \mathbf { N } ( g _ { i } ^ { l } ) , } \\ & { z _ { i } ^ { l } = \mathbf { L } \mathbf { A } \mathbf { Y } \mathbf { E } \mathbf { R } \mathbf { N O R } \mathbf { M } ( \tilde { z } _ { i } ^ { l } + g _ { i } ^ { l } ) , } \end{array}
$$

where MULTIATTN represents the multi-head self-attention mechanism, LAYERNORM refers to layer normalization, and FFN is a two-layer feed-forward neural network with ReLU activation. After passing through $L$ layers, the hidden state $z _ { i , \mathrm { [ C L S ] } } ^ { L } \in \mathbb { R } ^ { d }$ of the [CLS] token from the final layer is used as the future emotion representation $r _ { \psi , i + w - 1 }$ .

Finally, these emotion representations $r _ { \psi }$ are encoded into an emotion distributions $\boldsymbol { e } _ { \psi }$ through a Feed-Forward Layer:

$$
\begin{array} { r } { e _ { \psi } = \mathrm { F F L } ( r _ { \psi } ) . } \end{array}
$$

The loss function for emotion shift modeling is as follows:

$$
\mathcal { L } _ { E S M } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { e = 1 } ^ { | C | } y _ { i } ^ { e } \log ( \mathrm { s o f t m a x } ( e _ { \psi , i } ) ) .
$$

In our approach, we use the emotion distribution $e _ { \theta , i }$ as the initial emotion instead of the emotion label $y _ { i }$ , forming an implicit emotion validation chain, which we will discuss later. Empirical evidence indicates that this chain enhances model performance. Moreover, leveraging labeled data in single-view learning helps guide representations and distributions toward task-relevant features (Zhai et al. 2019; Henaff 2020).

# 3.4 Dual-View Projection Learning

Our objective is to develop robust emotion distributions that can effectively classify emotions across conversations while capturing shifts within individual emotions. This task is challenging due to the significant discrepancy between the heterogeneous views involved. A straightforward solution is to use existing multi-view learning strategies that aggregate information from multiple views by maximizing correlation or independence. However, these strategies, as demonstrated in Section 4.4, are effective but generally optimized for representation rather than category distribution, leading to suboptimal results. To address this limitation, we propose a dual-view projection learning strategy tailored to emotion distributions in CER. Inspired by contrastive learning (Chen et al. 2020; Grill et al. 2020), our approach predicts different views of the same utterance from one another, enabling the sharing of learned emotion distributions between views.

Given two emotion distributions, $e _ { \theta }$ and $\boldsymbol { e } _ { \psi }$ , derived from the utterance and emotion views, respectively, we recognize that these distributions exist in distinct, heterogeneous spaces. To bridge this gap, we employ two non-linear projectors, $f _ { \theta }$ and $f _ { \psi }$ , to directly learn the output of one view from the other. Consistency between views and projections is then calculated within their respective spaces. The viewsharing objective is defined using the Mean Squared Error (MSE) loss function:

$$
\begin{array} { r } { \mathcal { L } _ { M S E } ^ { \theta } = | | e _ { \theta } - f _ { \psi } ( e _ { \psi } ) | | _ { 2 } ^ { 2 } , } \\ { \mathcal { L } _ { M S E } ^ { \psi } = | | e _ { \psi } - f _ { \theta } ( e _ { \theta } ) | | _ { 2 } ^ { 2 } . } \end{array}
$$

These projectors provide information from other views and translate the individually training process within each view’s space into a process of mutual learning and interaction. This approach maximizes cross-view correlations without sacrificing the independence of each view.

Emotion Validation Chain. Our framework further benefits from the emotion shift network, which inputs $e _ { \theta }$ and utilizes the projector $f _ { \psi }$ to output $e _ { \theta }$ . This creates an implicit emotion validation chain, forming a feedback loop that injects the model’s ability to validate $e _ { \theta }$ .

Why Emotion Distribution Instead of Emotion Representation? The distinction lies in information density. Emotion representations carry more detailed information than emotion distributions, making cross-view prediction more challenging. Additionally, the continuous updating of initial emotions can lead to instability and complicate future emotion predictions. Therefore, our approach formulates tasks involving emotion distributions: (1) The FFL compresses high-dimensional emotion representations into task-specific low-dimensional emotion distributions, akin to knowledge distillation (Gou et al. 2021), where logits act as carriers of view information. (2) Small deviations in emotion distribution vectors (e.g., six classes in IEMOCAP) do not result in significant projection losses, contributing to training stability. (3) Unlike unsupervised and semi-supervised representation learning, which attempts to learn a generalizable representation, our fully supervised approach directly learns a class distribution tailored to the classification task.

Table 1: Statistics of the datasets.   

<html><body><table><tr><td rowspan="2">Datasets</td><td colspan="3">#Dialogues</td><td colspan="3">#Utterances</td><td rowspan="2">#Classes</td></tr><tr><td>train</td><td>val</td><td>test</td><td>train</td><td>val</td><td>test</td></tr><tr><td>IEMOCAP</td><td>120</td><td></td><td>31</td><td>4,810</td><td>1,000</td><td>1,623</td><td>6</td></tr><tr><td>EmoryNLP</td><td>713</td><td>99</td><td>85</td><td>9.934</td><td>1,344</td><td>1,328</td><td>7</td></tr></table></body></html>

# 3.5 Training and Inference

Training. We train the DVL-CER approach using a combination of cross-entropy losses $\mathcal { L } _ { C C M }$ and $\mathcal { L } _ { E S M }$ for singleview emotion classification, aLnd MSE l oLsses $\mathcal { L } _ { M S E } ^ { \theta }$ and $\mathcal { L } _ { M S E } ^ { \psi }$ for cross-view correlation:

$$
\mathcal { L } = \mathcal { L } _ { C C M } + \mathcal { L } _ { E S M } + \lambda _ { 1 } \mathcal { L } _ { M S E } ^ { \theta } + \lambda _ { 2 } \mathcal { L } _ { M S E } ^ { \psi } ,
$$

where $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are hyperparameters that balance the tradeoff between independence $\ L _ { C C M }$ and $\mathcal { L } _ { E S M }$ ) and correlation $( \mathcal { L } _ { M S E } ^ { \theta }$ and $\mathcal { L } _ { M S E } ^ { \psi } )$ .

Inference. The resulting emotion distributions from both views are expected to aggregate all information and achieve high distribution consistency (see Section 4.6), despite residing in different feature spaces. For inference, we retain only the conversational context network and its corresponding emotion distribution $e _ { \theta }$ , ensuring comparability with existing CER baselines and addressing two key questions:

Question I: Is the emotion shift modeling scheme reasonable and effective?   
Question II: Does the proposed dual-view projection learning strategy effectively integrate information from other views, enhancing singleview learning and its emotion distributions?

# 4 Experiments

# 4.1 Experiment Setup

Datasets and Evaluation Metric. We evaluate our DVLCER approach using two benchmark datasets: IEMOCAP (Busso et al. 2008) and EmoryNLP (Zahiri and Choi 2018). Detailed descriptions of these datasets are provided in Appendix $\mathbf { A } ^ { 1 }$ , and the statistics are presented in Table 1.

Following recent studies (Ghosal et al. 2020; Li et al. 2021), we utilize only the textual modality for experiments and adopt the weighted-F1 metric for evaluation. Additionally, we perform a paired t-test $( \mathsf { K i m } 2 0 1 5 )$ to assess the statistical significance of performance improvements.

Baseline Models and Implementation Details. To comprehensively evaluate DVL-CER, we compare it with state-ofthe-art baselines, covering both single-view and multi-view

Table 2: Performance comparison of various CER methods on IEMOCAP and EmoryNLP. †Results are taken from Hu et al. (2023). ‡Results are from our replication.   

<html><body><table><tr><td>Method IEMOCAP EmoryNLP</td></tr><tr><td>Single-viewmethods DialogueGCN (Ghosal et al.2019) 64.18 RGAT (Ishiwatari et al. 2020) 65.22 34.42</td></tr><tr><td>SKAIG (Li et al. 2021) 66.96 38.88 DialogXL(Shen etal.2021a) 65.94 34.73 DAG-ERC-W (Shen etal.2021b) 68.03 39.02 CoMPM (Lee and Lee 2022) 69.46 38.93 VAE (Ong et al. 2022) 68.23 CoG-BART (Li, Yan,and Qiu 2022) 66.18 39.04 SUNET (Song et al. 2023) 68.96 39.89 DAG-ERC-X (Quan et al. 2023) 68.50 39.19</td></tr><tr><td>denoiseGNN(Gan et al. 2024) 69.70 39.70 Multi-viewmethods DialogueRNN† (Majumder et al. 2019) 64.65 37.54 COSMIC (Ghosal et al. 2020) 65.28 38.11 DialogueCRN† (Hu,Wei,and Huai 2021) 67.53 38.79 MVN (Ma et al. 2022) 65.99 COSMIC+HCL(Yang etal.2022) 66.23 38.96 HMVDM (Ruan et al. 2022) 67.96 38.46</td></tr><tr><td>SACL-LSTM (Hu et al.2023) 69.22 39.65 DualRAN(Li,Wang,and Zeng 2024) 69.17 39.18</td></tr><tr><td>DialogueCRN* 69.01 38.97 DVL-CER 70.11 (↑ 1.10) 39.92 (↑ 0.95)</td></tr></table></body></html>

CER methods (see Appendix B for a complete list). DialogueCRN serves as the backbone of the conversational context network in DVL-CER, except where noted otherwise. The reported results are averaged over 20 random runs on the test set to ensure reliable performance evaluation. Additional experimental details are available in Appendix C.

# 4.2 Comparison with State of the Art

Table 2 compares the performance of DVL-CER with the baselines. The results indicate that: (1) DVL-CER comparatively outperforms all baselines across both datasets, demonstrating its effectiveness in CER. By leveraging information from both utterance and emotion views, our approach generates high-quality emotion distributions, leading to more accurate emotion recognition. (2) Compared with the default DialogueCRN backbone, DVL-CER improves performance, suggesting that the dual-view projection heads effectively transfer valuable information from the emotion shift network to the conversational context network (addressing Questions I and II). Overall, DVL-CER achieves leading performance on both datasets, highlighting the advantages of our dual-view learning framework.

# 4.3 Applying DVL-CER to Different CER Baselines

The proposed DVL-CER is a flexible, context networkagnostic framework that can be integrated with various CER baseline models. To validate its effectiveness and generalizability, we apply DVL-CER to several prominent CER models: DialogueRNN, COSMIC, DAG-ERC-W, RGAT, and SACL-LSTM, which represent different mainstream approaches to conversational context modeling. As shown in Table 3, DVL-CER consistently provides significant improvements over both single-view and multi-view baselines. This demonstrates that dual heterogeneous view learning enhances the accuracy and robustness of emotion distributions (addressing Questions I and $\mathbf { I I }$ ).

Table 3: Performance of the proposed DVL-CER framework when integrated with other CER baselines across two benchmarks. ∗Results show significant test $p$ -value $< 0 . 0 5$ compared to their respective baseline models.   

<html><body><table><tr><td>Backbone</td><td>DVL-CER</td><td>IEMOCAP</td><td>EmoryNLP</td></tr><tr><td>DialogueRNN DialogueRNN</td><td>√</td><td>66.03 67.77* (↑1.74)</td><td>37.73 38.30* (↑0.57)</td></tr><tr><td>COSMIC COSMIC</td><td>√</td><td>67.45 68.96* (↑1.51)</td><td>38.67 39.20* (↑0.53)</td></tr><tr><td>RGAT RGAT</td><td>√</td><td>65.83 66.72* (↑0.89)</td><td>37.53 38.43*(↑0.90)</td></tr><tr><td>DAG-ERC-W DAG-ERC-W</td><td>√</td><td>65.81 66.22* (个0.41)</td><td>38.34 39.20* (↑0.86)</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>69.40</td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>SACL-LSTM</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>39.22</td></tr><tr><td>SACL-LSTM</td><td>√</td><td>70.67* (↑1.27)</td><td>39.49 (↑0.27)</td></tr></table></body></html>

# 4.4 Comparison with Other Dual-View Learning Strategies

A key contribution of DVL-CER is its dual-view projection learning strategy. To assess whether the emotion view is beneficial across CER domains or specific to this strategy, we replace the dual-view projection learning strategy with other dual-view learning strategies, including dual-view feature aggregation learning, dual-view subspace learning, and dual-view alignment learning. These strategies integrate utterance and emotion views differently (see Appendix D for architectural details and Section 4.6 for theoretical analysis).

Our results on the IEMOCAP dataset, shown in the second column of Table 4, reveal that: (1) All dual-view learning strategies benefit from the emotion view, surpassing the DialogueCRN baseline and indicating successful emotion shift modeling (addressing Question I). (2) Among these strategies, our dual-view projection learning achieves the highest performance in DVL-CER. Similar results are noted on the EmoryNLP dataset, as detailed in Appendix.

# 4.5 Ablation Study

We conduct an ablation study on DVL-CER to gain insights into its behavior and performance. Table 5 presents the impact of various architectural components on model performance, and Table 6 explores the influence of different initial emotion settings and the application of the dual-view projection learning strategy to different interaction objects. In addition, Figure 3 shows the effect of varying emotion shift spans.

Detailed Ablation on Projectors. The ablation study on the two projectors, shown in the third row of Table 5, indicates that removing either projector significantly hinders view interaction, leading to a notable decline in performance.

The importance of ReLU in Projectors. Introducing the ReLU non-linear activation function in the projectors enhances the interaction between heterogeneous views, as shown in the fourth row of Table 5.

Impact of Supervision in Emotion View. The fifth row of Table 5 highlights the critical role of the loss $\mathcal { L } _ { M S E } ^ { \psi }$ in model training. Without emotion-view supervision, the model’s performance deteriorates, consistent with previous studies (Zhai et al. 2019; Henaff 2020).

Impact of Emotion Verification Chain. The sixth row of Table 5 indicates that verifying the emotion distributions predicted by the backbone substantially improves performance, indicating effective emotion connections in our emotion shift (addressing Question I).

Emotion Distribution versus Emotion Representation. Table 6 illustrates that emotion representations from two views are difficult to project and learn from each other, as discussed in Section 3.4.

Choice of Initial Emotions. To assess the role of initial emotions in emotion shift, we test two alternatives: a real but less informative one-hot label vector, and a zero vector that zeros out the initial emotion variable while maintaining consistency in the framework structure.

Results on the IEMOCAP dataset, as shown in Table 6, indicate that initializing with the emotion distribution is better than with the one-hot vector, and much better than with the zero vector (addressing Question I). Similar observations are noted on the EmoryNLP dataset, where the difference between the one-hot and zero vectors is minimal, primarily due to disruptions in the emotion verification chain.

Effect of Emotion Shift Span. We experiment with different emotion shift spans to explore the optimal threshold. As shown in Figure 3, the emotion shift modeling consistently provides valuable information to the context network across varying span settings (addressing Question I). The best performance for both datasets is achieved with an emotion shift span of $w = 4$ , which is used as the default in our study.

# 4.6 Understanding the Learned Emotion Distributions

Finally, we explore the unique properties of our dual-view projection learning strategy compared to other dual-view learning strategies. As prerequisites, we discuss two important attributes: consistency and independence $\mathrm { S u n } \ 2 0 1 3 )$ . We use the similarity metric $\mathrm { S i m } ( \pmb { u } , \mathbf { \bar { v } } ) = \pmb { u } ^ { \top } \pmb { v } / ( \| \pmb { u } \| \| \pmb { v } \| )$ to evaluate the similarity among the two views and their projections. Consistency correlates positively with similarity, while independence correlates negatively. Theoretical and mathematical insights are as follows:

• Dual-view feature aggregation learning adds emotion distributions from both views for emotion prediction, maximizing independence for complementarity. Here, $\mathrm { S i m } ( e _ { \theta } , e _ { \psi } )$ shows the highest independence. • Dual-view subspace learning use two projections to map both views into a common subspace where independence is filtered out and cross-view correlation is maximized. Hence, $\sin ( e _ { \theta } , e _ { \psi } )$ before projections indicates

Table 4: Comparison of different dual-view learning strategies applied to DVL-CER on the IEMOCAP dataset. Symbol “✕” ndicates that the strategy cannot handle quantization.   

<html><body><table><tr><td>Strategy</td><td>IEMOCAP|Sim(eθ,fφ(e))|</td><td></td><td></td><td></td><td>Sim(fe(eθ),eψ)|Sim(eθ,ep)| Sim(fθ(eθ),f(e))</td></tr><tr><td>Dual-view feature aggregation learning</td><td>69.73</td><td>× 一</td><td>×</td><td>0.6837</td><td>×</td></tr><tr><td>Dual-view subspace learning</td><td>69.55</td><td>0.0962</td><td>0.1918</td><td>0.8563</td><td>0.8803</td></tr><tr><td>Dual-view alignment learning</td><td>69.19</td><td>×</td><td>×</td><td>0.9667</td><td>×</td></tr><tr><td>Dual-view projection learning</td><td>70.11</td><td>0.9629</td><td>0.8858</td><td>0.7270</td><td>0.6435</td></tr></table></body></html>

Table 5: Component-wise ablation study with Projectors and Softmax layers.   

<html><body><table><tr><td colspan="2">Projector</td><td rowspan="2"></td><td colspan="2">Softmax</td><td rowspan="2">IEMOCAP</td><td rowspan="2">EmoryNLP</td><td rowspan="2">Comment</td></tr><tr><td>fe f</td><td>ReLU</td><td>softmax0</td><td>softmaxy</td></tr><tr><td></td><td></td><td></td><td>√</td><td></td><td>69.01</td><td>38.97</td><td>Loss: LcCM (Baseline)</td></tr><tr><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>69.70</td><td>39.68</td><td></td></tr><tr><td>√</td><td></td><td>√</td><td>√</td><td>√</td><td>69.30</td><td>39.42</td><td>Los:CCM+CES+XCsE</td></tr><tr><td>√</td><td>√</td><td></td><td>√</td><td>√</td><td>69.48</td><td>39.66</td><td>Los CCM+LESM +XCMs+LMSE</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td>69.63</td><td>39.65</td><td>Los:CCCM +CMSE+XCMSE</td></tr><tr><td></td><td>√</td><td>√</td><td>√</td><td></td><td>69.54</td><td>39.36</td><td>Loss: CcM + X2CMSE</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>70.11</td><td>39.92</td><td> Loss: LcCM +LESm + X1LMSE + X2LMSE</td></tr></table></body></html>

<html><body><table><tr><td>Interaction object</td><td>Initial emotion</td><td>IEMOCAP</td><td>EmoryNLP</td></tr><tr><td>representation r</td><td></td><td>68.47</td><td>38.50</td></tr><tr><td>distribution e</td><td></td><td>70.11</td><td>39.92</td></tr><tr><td></td><td>zero vector</td><td>68.55</td><td>39.75</td></tr><tr><td></td><td>one-hot vector</td><td>69.60</td><td>39.79</td></tr><tr><td></td><td>eθ</td><td>70.11</td><td>39.92</td></tr></table></body></html>

Table 6: Impact of dual-view projection learning strategy on interaction objects and initial emotions in emotion shifts.

independence, while $\mathrm { S i m } ( f _ { \theta } ( e _ { \theta } ) , f _ { \psi } ( e _ { \psi } ) )$ after projections indicates consistency. • Dual-view alignment learning aligns the emotion distributions of both views to maximize correlation, resulting in the highest consistency with $\sin ( e _ { \theta } , e _ { \psi } )$ . • Dual-view projection learning transforms one view’s emotion distribution to match the other’s. Specifically, $\mathrm { S i m } ( f _ { \theta } ( e _ { \theta } ) , e _ { \psi } )$ and $\mathrm { S i m } ( e _ { \theta } , f _ { \psi } ( e _ { \psi } ) )$ measure the consistency between the two views in different view spaces, while $\sin ( e _ { \theta } , e _ { \psi } )$ represents their independence.

Quantitative results on the IEMOCAP dataset, shown in Table 4, demonstrate that our dual-view projection learning strategy achieves a consistency level $( \mathrm { S i m } ( e _ { \theta } , f _ { \psi } ( e _ { \psi } ) ) )$ similar to dual-view alignment learning $( \mathrm { { S i m } } ( e _ { \theta } , e _ { \psi } ) )$ , while preserving independence $( \sin ( e _ { \theta } , e _ { \psi } ) )$ comparable to dualview feature aggregation learning $( \mathrm { \dot { S } i m } ( e _ { \theta } , e _ { \psi } ) )$ . This better balance between consistency and independence across the utterance and emotion views makes our strategy outperform dual-view subspace learning and achieve state-of-theart performance, as elaborated in Appendix E.

weighted-F1 69.9 69.6 39.50 39.75 69.3 8 39.25 W 69.0 39.00 2 3 4 5 6 2 3 4 5 6 emotion shift span emotion shift span (a) IEMOCAP (b) EmoryNLP

# 5 Conclusions

In this work, we present a dual-view learning framework that integrates utterance and emotion views to generate robust emotion distributions for CER. Through comprehensive empirical comparisons with existing CER and dualview learning methods, we derive two main conclusions: (1) The emotion shift modeling approach is both effective and well-founded, as it successfully establishes connections between emotions. (2) The dual-view projection learning strategy effectively integrates information from multiple views, enhancing single-view learning process while maintaining a balance between cross-view consistency and view-specific independence. Future research could explore extending the emotion shift modeling to multi-modal CER. Additionally, the dual-view projection learning strategy shows potential for applications in multi-view supervised learning tasks.