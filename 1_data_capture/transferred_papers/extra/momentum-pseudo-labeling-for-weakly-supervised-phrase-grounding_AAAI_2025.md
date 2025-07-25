# Momentum Pseudo-Labeling for Weakly Supervised Phrase Grounding

Dongdong Kuang1, Richong Zhang1,3\*, Zhijie Nie1,4, Junfan Chen1,2, Jaein Kim1

1 CCSE, School of Computer Science and Engineering, Beihang University, Beijing, China 2 School of Software, Beihang University, Beijing, China 3 Zhongguancun Laboratory, Beijing, China 4 Shen Yuan Honors College, Beihang University, Beijing, China {kuangdd, zhangrc, niezj, chenjf} $@$ act.buaa.edu.cn, jaein $@$ buaa.edu.cn

# Abstract

Weakly supervised phrase grounding tasks aim to learn alignments between phrases and regions with coarse imagecaption match information. One branch of previous methods established pseudo-label relationships between phrases and regions based on the Expectation-Maximization (EM) algorithm combined with contrastive learning. However, adopting a simplified batch-level local update (partial) of pseudolabels in E-step is sub-optimal, while extending it to global update requires inefficiently numerous computations. In addition, their failure to consider potential false negative examples in contrastive loss negatively impacts the effectiveness of M-step optimization. To address these issues, we propose a Momentum Pseudo Labeling (MPL) method, which efficiently uses a momentum model to synchronize global pseudo-label updates on the fly with model parameter updating. Additionally, we explore potential relationships between phrases and regions from non-matching image-caption pairs and convert these false negative examples to positive ones in contrastive learning. Our approach achieved SOTA performance on 3 commonly used grounding datasets for weakly supervised phrase grounding tasks.

#1:Ayoung girl with #2:A young girl in a 2 a blue shirt is eating formal dress and foodoffofatable. heels Pseudo-label generationqxy :y-th phrase in the x-th caption (a) Previous Method EncoderModel AlignmentProbability ×N M-Step: E-Step: Negatiye samples   
Update Parameter Update Matrix 000 Pseudo-LabelMartix q11 2 qi Global (N/Bupdates Nexttime SlowUpdate Local (One update) needed) Supervision (b) MPL (Ours) FalseNegativecorrection ×Ni Encoder Model Momentum Momentum Pseudo-Label Update 0 0 q11 P 中 qi Momentum Model Next time   
PseudoLabeling Global (One update) Supervision

# Introduction

Phrase grounding is a crucial task in multimodal learning, involving the identification of specific regions within an image that corresponds to a given textual description (Liu and Hockenmaier 2019). This task has significant practical applications, including visual question answering (Chen, Anjum, and Gurari 2022) and cross-modal regions retrieval $\operatorname { L i } ^ { * }$ et al. 2022). Existing fully supervised methods (Huang et al. 2021; Zhang et al. 2022) depend heavily on extensive annotations of fine-grained object detection bounding boxes, which provide precise locations of objects within an image to achieve high performance. However, obtaining such detailed annotations is costly. In contrast, matched imagecaption pairs, where each image is paired with a descriptive caption, are more readily available and do not require detailed spatial information about object locations. This ease of availability leads to growing interest in weakly supervised phrase grounding (WSPG), which aims to learn the correspondence between phrase and image regions without the need for extensive annotations.

Recent works in WSPG (Gupta et al. 2020; Wang et al. 2020; Chen et al. 2022; Rigoni et al. 2023; Zhang, Wang, and Liu 2023) have utilized image-caption level or phraseregion level contrastive losses based on image-caption supervision to align the representations of phrases and regions. However, under a weakly supervised setting, it is challenging to accurately identify positive samples during contrastive learning. To address this difficulty, these methods establish pseudo-label relationships between phrases and regions within matched image-caption pairs based on the Expectation-Maximization (EM) algorithm.

However, existing methods encounter two main challenges. First, they may encounter efficiency problems in Estep computation. As shown in Figure 1 (a), to quickly update the posterior probabilities of the latent variable, they adopt a simplified batch-level local update of pseudo-labels in E-step, usually on a batch of image-caption examples and fail to update the out-of-batch examples. This updating strategy is sub-optimal because a typical EM optimization requires a global update that needs to compute posterior probabilities for all examples in the whole dataset. We refer to this as the Slow Pseudo-label Update Issue. The necessity of such global updates is verified by our empirical study (Figure 4, Table 2), which shows that, with the same number of updates, global update achieves significant improvements compared with local update. Attempting to extend existing methods to global update requires more computation.

Second, the effectiveness of M-step optimization of existing methods may be affected by potential false negative examples in contrastive learning. Existing methods focus solely on the matching relationships between phrases and regions within matched image-caption pairs, without considering potential matching relationships from other imagecaption pairs. This approach treats all regions in nonmatching images as negative samples.

As shown in the top of Figure 1, both pairs of example image-captions feature the phrase a young girl and visually similar regions depicting a girl. However, simply treating the phrase-region pairs in non-matching image-caption pairs as negative examples may lead to the issue of false negatives, thereby impacting the effectiveness of contrastive learning and the consistency performance of grounding in M-step optimization. We refer to this as the False Negative Impact.

To address these challenges, we first propose to introduce a Momentum Pseudo Labeling (MPL) framework to globally update pseudo-labels in E-step computation to address the Slow Pseudo-label Update Issue. As shown in Figure 1 (b), instead of performing a local update on a batch of examples, MPL performs global pseudo-label computation by only one update to the momentum model parameters on the fly. In this way, the pseudo-labels for any examples in the next M-step can be easily accessed from the updated momentum model. Furthermore, within the MPL framework, we model the relationships between phrases and regions in non-matching images, exploring potential connections between region-phrase pairs across unmatched image-caption pairs to mitigate the False Negative Impact.

In summary, our key contributions are:

• We introduce a new pseudo-label updating strategy, Momentum Pseudo Labeling (MPL), which enables us to efficiently perform global updates of pseudo-labels. • We consider potential phrase-region relationships in nonmatching image-caption pairs and treat the false negative examples as pseudo-positive samples to improve contrastive learning in MPL. • Experimental results and intensive analysis on three commonly used phrase grounding datasets demonstrate the efficiency and effectiveness of our MPL approach.

# Problem Analysis

In this section, we introduce the problem setup of WSPG and revisit the previous methods (Wang et al. 2020; Chen et al. 2022; Zhang, Wang, and Liu 2023) from the perspective of the Expectation-Maximization algorithm (Moon 1996).

![](images/f4196b8c31c005871977a5a5e1ef1cfecce80117bbb01cab81e55fbd82a25360.jpg)  
Figure 2: Diagram of the Phrase-Region Dual-Encoder

# Problem Formulation

Consider an image-caption pair, $( I _ { i } , T _ { i } )$ . Let $Q ( T _ { i } )$ be the set of phrases in the caption $T _ { i }$ , and let $K ( I _ { i } )$ denote the set of regions extracted from the image $I _ { i }$ , namely $Q ( T _ { i } )$ and $K ( \bar { I _ { i } } )$ are the partitions of the caption $T _ { i }$ and the image $I _ { i }$ , respectively. The goal of phrase grounding is to locate a region $k \in K ( I _ { i } )$ in the image $I _ { i }$ for each phrase $q \in Q ( T _ { i } )$ from the corresponding caption $T _ { i }$ , such that the phrase refers to an object within that region.

For convenience, we may also use $q$ and $k$ to refer to the index of the corresponding phrase in the caption and the region in the image, respectively.

# EM Algorithm Perspective

In the weakly supervised scenario, the target regions in the matched image for the phrase $q$ are ambiguous. Specifically, for each $q$ , we introduce a variable $z$ to represent the correspondence between phrase $q$ and regions $k \in K ( I _ { i } )$ . Since $z$ is usually unobservable, we treat it as a latent variable. The set of latent variables for a given $q$ is defined as:

$$
Z = \{ z _ { q k } | k \in K ( I _ { i } ) \} .
$$

Moreover, the EM algorithm is particularly effective for estimating model parameters involving such latent variables.

EM Algorithm The EM algorithm is iterative and includes two steps: the $\mathrm { ^ E }$ -step and the M-step. In the $\mathrm { E }$ -step of $t$ -th iteration, the algorithm estimates the posterior probabilities of the latent variables $Z$ given the current parameter estimates $\theta ^ { ( t - 1 ) }$ and the observed data (e.g., image-caption pairs), denoted as $X$ . In the M-step, the algorithm then maximizes the expected log-likelihood (also known as the $\mathbf { Q } .$ - function) to update the parameters, yielding $\theta ^ { t }$ :

$$
\boldsymbol { \theta } ^ { t } = \arg \operatorname* { m a x } _ { \boldsymbol { \theta } } \underbrace { \int _ { Z } \log P ( \boldsymbol { X } , Z | \boldsymbol { \theta } ) \cdot P ( Z | \boldsymbol { X } , \boldsymbol { \theta } ^ { ( t - 1 ) } ) d Z } _ { \mathbf { Q } \cdot \mathrm { f u n c t i o n } } ,
$$

where $P ( X , Z | \theta )$ represents the joint probability of the observed variables $X$ and the latent variables $Z$ and $P ( Z | X , \theta ^ { ( t - 1 ) } )$ is the posterior probability of the latent variables. For simplicity, given a phrase $q$ , we denote $P ( Z | X , \theta ^ { ( t - 1 ) } )$ sceourdroe-slpaobneld.ing to region $k$ as $\bar { \pi } _ { q k } ^ { ( t - 1 ) }$ , and re

Revisiting from an EM Algorithm Perspective Existing contrastive learning-based methods (Wang et al. 2020; Chen et al. 2022; Zhang, Wang, and Liu 2023) can be summarized as implementing the E-step by estimating the pseudo-labels $\pi _ { q k }$ based on the current model parameters $\theta ^ { ( t - 1 ) }$ and a batch of $B$ image-caption pairs data $X \ =$ $\{ ( Q ( T _ { i } ) , K ( I _ { i } ) ) \} _ { i = 1 } ^ { B }$ . In the M-step, the model parameters $\theta ^ { t }$ are obtained by applying a single step of stochastic gradient descent (SGD).

More precisely, in existing methods, the posterior probability of latent variables is updated according to the following equation:

$$
\pi _ { q k } ^ { t } = \lambda \cdot \pi _ { q k } ^ { ( t - 1 ) } + ( 1 - \lambda ) \cdot s _ { q k } ,
$$

where πq(tk−1) is the posterior probability of latent variables from the last iteration, and $s _ { q k }$ is computed as

$$
s _ { q k } = \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f } \ k = \ \underset { 1 \leq c \leq | K ( I _ { i } ) | } { \arg \operatorname* { m a x } } \ \langle \pmb { q } , \pmb { k } _ { c } \rangle , } \\ { 0 } & { \mathrm { o t h e r w i s e } , } \end{array} \right.
$$

where $\pmb q$ and $\pmb { k } _ { c }$ represent the output features from a dual encoder as shown in Figure 2 of the phrase $q$ in $T _ { i }$ and $c$ -th region in $I _ { i }$ , respectively, with respect to current model parameters $\theta ^ { ( t - 1 ) } . \left. . , . \right.$ denotes the dot product operation.

As concrete examples, in the context of MAF from Wang et al. (2020) and CC from Zhang, Wang, and Liu (2023), their $\mathbf { Q }$ -function can be represented by the image-caption level cross-entropy, as given below:

$$
\sum _ { i = 1 } ^ { B } \frac { 1 } { B } \cdot \log \frac { \underset { q \in Q ( T _ { i } ) } { \sum } \underset { k ^ { + } \in K ( I _ { i } ) } { \sum } \pi _ { q k ^ { + } } \cdot \langle \pmb { q } , \pmb { k } ^ { + } \rangle / | Q ( T _ { i } ) | ) } { \underset { j = 1 } { \overset { B } { \sum } } \exp ( \underset { q \in Q ( T _ { i } ) } { \sum } \underset { k \in K ( I _ { j } ) } { \sum } \langle \pmb { q } , \pmb { k } \rangle / | Q ( T _ { i } ) | ) } .
$$

Similarly, for CLEM as discussed in Chen et al. (2022), the pseudo-label estimation formula during the E-step is identical to that of MAF and CC. However, it replaces the $\mathbf { Q }$ -function with the alignment term from the InfoNCE loss:

$$
\sum _ { i = 1 } ^ { B } \sum _ { q \in Q ( T _ { i } ) } \sum _ { k ^ { + } \in K ( I _ { i } ) } \frac { \pi _ { q k ^ { + } } } { B \cdot | Q ( T _ { i } ) | } \cdot \log \frac { \exp ( \langle q , k ^ { + } \rangle ) } { \sum _ { k \in K ( \mathfrak { I } ) } \exp ( \langle q , k \rangle ) } ,
$$

where $K ( \tau )$ in the denominator represents the set of regions from all images in the batch.

# Motivating Factors

Efficiency in E-step Computation: Ideally, the E-step in the EM algorithm requires updating the posterior probabilities of pseudo-labels $\pi _ { q k }$ for all $N$ samples $( Q ( T _ { i } ) , K ( I _ { i } ) )$ in the dataset. However, this is usually computationally expensive because $N$ is very large in practice, often exceeding $1 0 K$ . Therefore, as demonstrated in Equation (5) and (6), to quickly update the posterior probabilities $\pi _ { q k }$ existing works adopt a local update strategy in the Estep. Specifically, they only update $\pi _ { q k }$ of a mini-batch of $( Q ( T _ { i } ) , K ( I _ { i } ) )$ samples and ignore the update of pseudolabels outside of that batch. This strategy may lead to suboptimal estimation of the posterior probabilities $\pi _ { q k }$ and degrade the EM performance under for Slow Pseudo-label

Update Issue, as evidenced in Figure 4. However, applying global updates to all samples $( Q \bar { ( \cal { T } _ { i } ) } , K ( \cal { I } _ { i } ) )$ with existing methods is inefficient, requiring at least $N / B$ updates.

Effectiveness in M-step Optimization: Existing methods transform the maximization of the $\mathbf { \overline { { Q } } } .$ -function in M-step into minimizing a contrastive loss, where the positive and negative examples are determined by the pseudo-labels estimated with $\pi _ { q k }$ . Therefore, treating one of these similar regions as a positive sample and the others as negatives can potentially confuse the model (Huynh et al. 2022), thereby influencing the parameter updates in the M-step and the latent variable estimation in the next E-step.

# Methodology

We introduce our MPL-based weakly supervised contrastive learning framework, as illustrated in Figure 3. Our framework is structured into three main components: the Dual Encoder, the Momentum Pseudo Labeling Module, and the Pseudo-label Guided Contrastive Loss. The Dual Encoder comprises text and image encoders responsible for encoding regions and phrases. The MPL module ensures stable and efficient pseudo-label updates leveraging the momentum model. Finally, the Pseudo-label Guided Contrastive Loss incorporates strategies to handle false negative region samples, improving the model’s consistency.

# Dual Encoder

Our framework utilizes a dual encoder architecture similar to previous work (Gupta et al. 2020; Wang et al. 2020; Chen et al. 2022). This dual encoder consists of a text encoder and an image encoder, as shown in Figure 2. In the following section, we will refer to the dual encoder as the base model, corresponding to the momentum model.

Text Encoder Our text encoder, similar to the one used in Chen et al. (2022), uses pre-trained GloVe embeddings (Pennington et al. 2014) to obtain word vector $\textbf { \em w }$ for each word. Given a phrase $q$ , the representation of a phrase is computed by summing the hidden states that constitute it:

$$
\begin{array} { r } { \{ \pmb { h _ { i } } \} _ { i = 1 } ^ { n _ { q } } = F _ { t } ( \{ \pmb { w _ { i } } \} _ { i = 1 } ^ { n _ { q } } ) , } \\ { \pmb { q } = W _ { q } ( \displaystyle \sum _ { i = 1 } ^ { n _ { q } } \pmb { h _ { i } } / \sigma ) , } \end{array}
$$

where $F _ { t }$ denotes one layer of LSTM network, $\mathbf { \delta } _ { h _ { i } }$ represents the $i$ -th output hidden states, $W _ { q }$ is a linear mapping, $\sigma$ is a hyperparameter scaling the text representation, and $n _ { q }$ is the number of words in the current phrase.

Image Encoder Our image encoder consists of two functional components. The first is a frozen pre-trained object detector responsible for identifying salient regions in the image and extracting visual features from these regions. The second component focuses on establishing relationships between these regions and mapping their features across spaces. For instance, given an image $I$ , the object detector $f _ { D }$ outputs $m$ pairs $( l _ { i } , v _ { i } )$ , where $l _ { i }$ represents the probable object label of the $i$ -th region, and $\boldsymbol { v } _ { i }$ denotes the highdimensional features output from the object detector. Formally, $\{ ( l _ { i } , v _ { i } ) \} _ { i = 1 } ^ { m } = f _ { D } \mathbf { \bar { ( } } I )$ .

Region Labels Dual Encoder Contrastive Learning Module False Negative Detection 𝒍 23 Model Image Text Region Features Cross-Modal Logits Input Encoder 𝒑13 𝒑23 Cy 队 中   
𝒗13RO𝒗I1F3eat·u·re𝒗s23 Phrase Features 𝑝11 品 EMA Cross-Modal 𝑝12 中 X   
Cr 𝑝11 𝑝12 𝑝13 Momentum Similarity 𝑝13 False ROI Features Image Text Negative Similarity 𝑝21 𝑝22 𝑝23 Encoder KL-Divergence   
Object Detector 𝑝11:a young girl I T in #1 caption stop gradient Momentum False Negative False Negative #1:A young girl Pseudo Labeling Elimination Conversion with a blue shirt is eating food off 𝒌෩ 𝒌෩ ： Pseudo Labels CVL M CvlaN CNLCAN I2 T2 of a countertop. M_Region Features Calculation 𝑝11 0 0 0 𝑝11 0 0 𝑝11 0 0 #2:A young girl 𝑝12 0 0 0 𝑝12 0 0 𝑝12 0 0 in a formal dress ෥𝒑23 𝑝13 0 0 0 𝑝13 0 0 𝑝13 0 0 and heels. M_Phrase Features Original Pseudo-Labels Pseudo-Labels after FNE Pseudo-Labels after FNC

In the remaining part of the image encoder, similar to (Gupta et al. 2020; Chen et al. 2022), we use the outputs $l _ { i }$ to incorporate prior knowledge from object detectors into the region features. We utilize transformer encoder (Vaswani et al. 2017) to establish relationships among different region features, enhancing the regional perception capabilities within each image, as illustrated in the following equations:

$$
\begin{array} { r } { \hat { \pmb { k } } _ { i } = W _ { k } ( v _ { i } ) + l _ { i } , } \\ { \{ \pmb { k } _ { i } \} _ { i = 1 } ^ { m } = E ( \{ \hat { \pmb { k } } _ { i } \} _ { i = 1 } ^ { m } ) , } \end{array}
$$

where $W _ { k }$ is a linear projection applied to each region’s visual features, $\mathbf { \xi } _ { l _ { i } }$ represents the word embedding corresponding to the object label of each region $k$ , and $E$ denotes the transformer encoder operation.

# Momentum Pseudo Labeling Module

As mentioned earlier, previous methods, from the perspective of the EM algorithm, update pseudo-labels only within the current batch during the E-step due to efficiency considerations. To tackle this limitation, we propose the Momentum Pseudo Labeling (MPL) method. MPL uses a momentum model to accumulate updates to the dual encoder parameters during the M-step. This approach computes pseudolabels for different batches in a more stable and smooth manner, providing a cost-effective and efficient way to approximate a global update of pseudo-labels.

Notably, our MPL method distinguishes itself from previous methods such as (Li et al. 2020; He et al. 2020; Li et al. 2021). Specifically, Li et al. (2020) apply a momentum model during the E-step to globally maintain certain cluster centers for latent variables, while He et al. (2020) utilize the momentum model to maintain queues of negative examples. Li et al. (2021) use a momentum model to compute pseudo-labels, reducing noise in the actual imagecaption labels. Our approach utilizes the momentum model to compute pseudo-labels of phrase-region pairs to guide the parameter update of the base model.

Pseudo Labels Calculation Unlike ALBEF (Li et al. 2021), we only have image-caption level supervision in the current weakly supervised scenario, but finer-grained phrase-region alignment is needed. Under this constraint, we use the momentum model to compute pseudo-labels for phrase-region matches in matched image-caption pairs, serving as fine-grained supervision. These pseudo-labels guide the contrastive learning module in our framework, detailed in the next section. Given a phrase $q \ \in \ Q ( T _ { i } )$ and all regions within the batch $k ^ { + } \in \bar { K } ( \bar { \mathbf { I } } )$ , pseudo-labels are computed as follows:

$$
\pi _ { q k ^ { + } } = \left\{ \begin{array} { c l } { 0 } & { \mathrm { i f ~ } k ^ { + } \in K ( \mathtt { I } ) \backslash K ( I _ { i } ) , } \\ { \frac { \exp ( \langle \tilde { q } , \tilde { k } ^ { + } \rangle / \tau _ { E } ) } { \sum _ { k \in K ( I _ { i } ) } \exp ( \langle \tilde { q } , \tilde { k } \rangle / \tau _ { E } ) } } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

Where the pseudo-label $\pi _ { q k ^ { + } }$ represents the correspondence probability between $q$ and $k ^ { + }$ . The terms $\tilde { \pmb q }$ and $\tilde { k }$ denote the phrase and region features output by the momentum model, respectively. Additionally, $\tau _ { E }$ is a temperature parameter. Here we simply set the matching probability of phrases and regions across different image-caption pairs to 0.

Momentum Model Update The update strategy for the momentum model follows a similar approach to that used in MoCo (He et al. 2020):

$$
\tilde { \theta } ^ { t } = \gamma \cdot \tilde { \theta } ^ { ( t - 1 ) } + ( 1 - \gamma ) \cdot \theta ^ { t } ,
$$

where $\theta$ and $\tilde { \theta }$ represent the parameters of the base model and the momentum model, respectively, with $t$ indicating a specific time step. $\gamma$ is the momentum coefficient of the model. The initialization of the momentum model is identical to the base model initialization.

It is worth noting that our momentum model provides more consistent feature output for pseudo-label calculation by leveraging exponential moving averaging (EMA) and benefits from its low update overhead. Specifically, after each update of the base model parameters, the synchronously updated momentum model is used to calculate the pseudo-labels for the positive pairs in the next minibatch of image-caption pairs. This can be viewed as implicitly completing the global update of pseudo-labels in the Estep after the M-step.

# Pseudo-label Guided Contrastive Loss

Unlike traditional binary relationships in conventional image-text contrastive learning, our approach uses pseudolabels between phrases and regions provided by the momentum model to establish positive relationships, which we term as pseudo-label guided contrastive learning. While $\pi _ { q k ^ { + } }$ is calculated using our MPL method, the contrastive loss for a phrase $q$ can be expressed in the form of a KL divergence:

$$
\mathcal { L } = - \sum _ { k ^ { + } \in K \left( \mathbb { I } \right) } \left( \pi _ { q k ^ { + } } \cdot \log \underbrace { \frac { \exp ( \langle \pmb { q } , \pmb { k } ^ { + } \rangle / \tau ) } { \sum _ { k \in K \left( \mathbb { I } \right) } \exp \left( \langle \pmb { q } , \pmb { k } \rangle / \tau \right) } } _ { \xi _ { q k ^ { + } } } - \pi _ { q k ^ { + } } \cdot \log \pi _ { q k ^ { + } } \right) .
$$

Where $\xi _ { q k ^ { + } }$ represents the matching probability between the phrase $q$ and the region $k ^ { + }$ as computed by the base model, since the last term is gradient-free, we will omit it in the following loss function.

About False Negative Impact mentioned earlier, we highlight how current methods (Wang et al. 2020; Chen et al. 2022) based on contrastive learning overlook the issue of false negatives under weakly supervised settings, potentially affecting the consistency of grounding. Considering the possible presence of false negatives, we propose two strategies to build connections between phrases and regions in nonmatching image-caption pairs.

False Negative Elimination To mitigate the impact of false negative samples under a weak supervision setting, we retrieve potential false negative samples based on the similarity of regional features in the visual modality, as shown in Figure 3. When a given phrase $q \in Q ( T _ { i } )$ , This set of potential false negatives can be represented as:

$$
\begin{array} { r l } & { { \mathcal { F } _ { q } } = \{ k ^ { \prime } \mid k ^ { \prime } \in K ( \mathbb { I } ) \setminus K ( I _ { i } ) , \exists k ^ { + } \in K ( I _ { i } ) } \\ & { \qquad \mathrm { s . t . } \cos ( v _ { k ^ { \prime } } , v _ { k ^ { + } } ) > \phi \} , } \end{array}
$$

where $v$ represents the high-dimensional visual features outputted by the detector, $\cos ( . , . )$ denotes cosine similarity, and $\phi$ is the similarity threshold used to retrieve false negatives.

We attempt to ignore these potential false negative region samples in the loss calculation. We define the filtered set of remaining regions $K ^ { e } ( \mathbb { I } ) \ = \ K ( \mathbb { I } ) \setminus \mathcal { F } _ { q }$ . The modified pseudo-label calculation is as follows:

$$
\pi _ { q k ^ { + } } ^ { e } = \left\{ \begin{array} { c l } { 0 } & { \mathrm { i f } \ k ^ { + } \mathrm { \in } \mathcal { K } ^ { e } ( \mathbb { I } ) \backslash K ( I _ { i } ) , } \\ { \frac { \exp \left( \left. \tilde { q } , \tilde { k } ^ { + } \right. / \tau _ { E } \right) } { \sum _ { k \in K ( I _ { i } ) } \exp \left( \left. \tilde { q } , \tilde { k } \right. / \tau _ { E } \right) } } & { \mathrm { o t h e r w i s e . } } \end{array} \right.
$$

The loss function is also modified as

$$
\mathcal { L } _ { q } ^ { e } = - \sum _ { k ^ { + } \in K ^ { e } ( \mathbb { I } ) } \pi _ { q k ^ { + } } ^ { e } \cdot \log \frac { \exp ( \langle q , k ^ { + } \rangle / \tau ) } { \sum _ { k \in K ^ { e } ( \mathbb { I } ) } \exp ( \langle q , k \rangle / \tau ) } .
$$

By designing the loss in this manner, we adjust the negative sampling under weakly supervised contrastive learning. This approach is referred to as False Negative Elimination.

False Negative Conversion To further investigate the role of the false negative examples detected and explore the potential relationships of phrase-region pairs from nonmatching image-caption pairs, we design to convert region $k ^ { \prime }$ within $\mathcal { F } _ { q }$ into potential positive sample to strengthen the grounding consistency of model.

We utilize the regions in $\mathcal { F } _ { q }$ to expand $K ( I _ { i } )$ , defining a new expanded set of regions $\mathring { \mathcal { K } } ^ { c } ( I _ { i } ) \stackrel { \cdot } { = } K ( I _ { i } ) \cup \mathcal { F } _ { q }$ . Given a phrase $\bar { \boldsymbol { q } } \in Q ( T _ { i } )$ , the calculation of the pseudo-labels is:

$$
\pi _ { q k ^ { + } } ^ { c } = \left\{ \begin{array} { c l } { 0 } & { \mathrm { i f ~ } k ^ { + } \in K ( \mathtt { I } ) \backslash K ^ { c } ( I _ { i } ) , } \\ { \frac { \exp ( \langle \tilde { q } , \tilde { k } ^ { + } \rangle / \tau _ { E } ) } { \sum _ { k \in K ^ { c } ( I _ { i } ) } \exp \big ( \langle \tilde { q } , \tilde { k } \rangle / \tau _ { E } \big ) } } & { \mathrm { o t h e r w i s e . } } \end{array} \right.
$$

After converting false negatives, the contrastive learning loss can be expressed as:

$$
\mathcal { L } _ { q } ^ { c } = - \sum _ { k ^ { + } \in K \left( \mathbb { I } \right) } \pi _ { q k ^ { + } } ^ { c } \cdot \log \frac { \exp ( \langle q , k ^ { + } \rangle / \tau ) } { \sum _ { k \in K \left( \mathbb { I } \right) } \exp ( \langle q , k \rangle / \tau ) } .
$$

To recap, within our framework, we designed the False Negative Elimination to modify the original negative sample sampling method, using the False Negative Conversion to transform negative samples into potential positive samples. These two methods are both used to mitigate the impact of potential false negatives on contrastive learning in the current weakly supervised setting. For clarity, we refer to our method incorporating the FNC strategy as MPL.

# Experiments

Datasets and Metric Our main experimental results are derived from benchmarks on three publicly used datasets for phase grounding task, including the Flickr30k Entities (Plummer et al. 2015), RefCOCO and $\operatorname { R e f C O C O + }$ (Kazemzadeh et al. 2014; Yu et al. 2016). For the RefCO$\mathrm { C O } / +$ dataset, we employ the UNC split (Yu et al. 2016), dividing both datasets into four parts: train, validation, testA, and testB. We evaluate our method using the IoU $\textcircled { a } \mathbf { 0 . 5 }$ as utilized in previous works (Wang et al. 2020; Jin et al. 2023).

Implementation Details Following prior work, we extracted regions and their features using Faster R-CNN (Ren et al. 2016), which is pre-trained on Visual Genome (Krishna et al. 2017). For the Flickr30k Entities dataset, we used the image features as in MAF (Wang et al. 2020), while for $\operatorname { R e f C O C O } / +$ datasets, we used our image features aligned with CLEM (Chen et al. 2022). Regarding the hyperparameter settings: the momentum update coefficient $\gamma$ is set to 0.99. For FNE, the threshold $\phi$ is set to 0.85, and for FNC, $\phi$ is set to 0.95. For more training and evaluation details, please refer to our supplementary materials. Our code can be accessed via https://github.com/Kuangdd01/MPL.

<html><body><table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td rowspan="2">LM</td><td rowspan="2">Proposals</td><td rowspan="2">Flickr30k</td><td colspan="2">RefCOCO</td><td colspan="2">RefCOCO+</td></tr><tr><td>TestA</td><td>TestB</td><td>TestA</td><td>TestB</td></tr><tr><td>ARN (Liu et al. 2019)</td><td>RN101</td><td>LSTM</td><td>Faster-RCNN</td><td>1</td><td>35.27</td><td>36.47</td><td>34.40</td><td>36.12</td></tr><tr><td>W-visualBERT (Dou et al.2021)</td><td>RNXT152</td><td>VL-BERT</td><td>Faster-RCNN(VG)</td><td>62.10</td><td></td><td>，</td><td>47.89</td><td>38.20</td></tr><tr><td>Pseudo-Q (Jiang et al. 2022)</td><td>RN101</td><td>BERT</td><td>Faster-RCNN (VG)</td><td>60.41</td><td>58.25</td><td>54.13</td><td>45.06</td><td>32.13</td></tr><tr><td>CPL (Liu et al. 2023)</td><td>RN101</td><td>BERT</td><td>Faster-RCNN (VG)</td><td>63.87</td><td>69.77</td><td>63.44</td><td>55.30</td><td>45.52</td></tr><tr><td>CCL (Zhang et al. 2020)</td><td>RN101</td><td>GRU</td><td>Faster-RCNN</td><td>1</td><td>37.64</td><td>32.59</td><td>36.91</td><td>33.56</td></tr><tr><td>InfoGround (Gupta et al. 2020)</td><td>RN101</td><td>BERT</td><td>Faster-RCNN (VG)</td><td>51.67</td><td></td><td></td><td>=</td><td></td></tr><tr><td>MAF (Wang et al. 2020)</td><td>RN101</td><td>GloVe</td><td>Faster-RCNN (VG)</td><td>61.43</td><td>51.76†</td><td>34.86†</td><td>32.20+</td><td>38.27†</td></tr><tr><td>KD+CL (Wang et al. 2021)</td><td>RN101</td><td>LSTM</td><td>Faster-RCNN (OI)</td><td>53.10</td><td>1</td><td></td><td>1</td><td>1</td></tr><tr><td>CLEM (Chen et al. 2022)</td><td>RN101</td><td>GloVe</td><td>Faster-RCNN (VG)</td><td>63.05</td><td>66.63†</td><td>54.60t</td><td>59.51</td><td>43.46</td></tr><tr><td>RefCLIP (Jin et al. 2023)</td><td>Darknet-53</td><td>GRU</td><td>Y0LOv3(VG)</td><td></td><td>58.58</td><td>57.13</td><td>40.45</td><td>38.86</td></tr><tr><td>MPL_FNc (ours)</td><td>RN101</td><td>GloVe</td><td>Faster-RCNN (VG)</td><td>64.15</td><td>70.19</td><td>55.74</td><td>63.59</td><td>45.20</td></tr></table></body></html>

Table 1: Comparison of three mainstream WSPG methods across three datasets. The top section pertains to methods based on modal reconstruction. The middle gray section represents methods that enhance weak supervision through additional data and model knowledge, while the bottom section and our method employ contrastive learning. (†) denotes our reproduction under identical settings. (VG) (CC) (OI) denote the object detector pre-trained on Visual Genome, MSCOCO, and OpenImage dataset.

ZZ Timer Cost 59.4 216.061.0 ZZTime Cost 12   
10² 60 10² Accuracy 49.7 46.3   
101 40 101 36.3   
100 100   
10-1 2 0.5 10-11 2 N Local UpdateMPL Global Update Local UpdateMPL Global Update (a) Flickr30K Entities (b) RefCOCO+

Table 2: The impact of different strategies for pseudo-label accuracy on the training set.   

<html><body><table><tr><td>Update Method</td><td>Flickr30k</td><td>RefCOCO</td><td>RefCOCO+</td></tr><tr><td>Local Update</td><td>62.01</td><td>61.27</td><td>53.85</td></tr><tr><td>Global Update</td><td>62.32</td><td>63.57</td><td>54.49</td></tr><tr><td>MPL (ours)</td><td>62.44</td><td>63.22</td><td>55.81</td></tr></table></body></html>

# Main Result

As shown in Table 1, we demonstrate the top-1 accuracy of our method on three datasets. Our method outperforms others that are also based on contrastive learning and show comparable performance to methods (Jiang et al. 2022; Liu et al. 2023) that require additional pre-training and knowledge of multimodal models. Specifically, it outperformed the previous SOTA, CLEM (Chen et al. 2022), which was based on weakly-supervised contrastive learning, by $1 . 1 \%$ on the Flickr30k, $3 . 6 \% / 1 . 1 \%$ on the RefCOCO testA/testB, and $4 . 1 \% / 1 . 7 \%$ on the $\operatorname { R e f C O C O + }$ testA/testB.

# Ablation Study

Comparison of Pseudo-label Updates To demonstrate the advantages of our MPL method in pseudo-label updating, we designed comparative experiments comparing Local pseudo-label updating, Global pseudo-label updating, and our methods, as shown in Figure 4. Both the Local and Global updates of pseudo-labels follow Equation (3) with settings consistent with those used in Chen et al. (2022). The difference is that the global update refreshes the pseudolabels across the entire training dataset after a batch-based parameter update, whereas the local update only updates the pseudo-labels within the current batch. To ensure fairness, three models were trained for the same number of steps with the same scale of trainable parameters, differing only in the form of pseudo-label updates.

![](images/6f618012050a23f755301623a44820e7dad987bc4adfcdf25d7e3d327f440195.jpg)  
Figure 4: Comparison of performance and time cost between different pseudo-label update strategies on the validation set.   
Figure 5: The impact of different similarity threshold values on the effectiveness of FNE and FNC. Accuracy refers to the performance on the validation set across different datasets.

From Figure 4, methods that update pseudo-labels globally yield better results but incur greater time costs. Conversely, local pseudo-label updating methods reduce time costs during training but suffer from slower pseudo-label updates, leading to poorer performance under insufficient training steps. MPL enables the modeling of global pseudolabels without significant additional time costs, achieving faster convergence and effectiveness comparable to global updating methods. In Table 2, we compared the pseudo-label accuracy at the end of model training under different update strategies on three datasets. Our method outperforms the Local Update approach regarding pseudo-label accuracy, providing better supervision for contrastive learning.

Table 3: Ablation study on EMA and False Negatives handling strategies. “FNE” stands for False Negative Elimination, and “FNC” denotes False Negative Conversion.   

<html><body><table><tr><td colspan="3">Strategy</td><td colspan="2">Flickr30k</td><td colspan="3">RefCOCO</td></tr><tr><td>EMA</td><td>FNE</td><td>FNC</td><td>val</td><td>test</td><td>val</td><td>testA</td><td>testB</td></tr><tr><td>-</td><td>-</td><td></td><td>61.59</td><td>62.47</td><td>62.07</td><td>68.31</td><td>54.80</td></tr><tr><td></td><td>√</td><td>1</td><td>61.18</td><td>62.45</td><td>62.69</td><td>68.41</td><td>55.23</td></tr><tr><td>1</td><td></td><td>√</td><td>62.09</td><td>63.72</td><td>63.66</td><td>69.68</td><td>55.07</td></tr><tr><td>√</td><td>1</td><td></td><td>62.25</td><td>64.15</td><td>63.82</td><td>69.77</td><td>55.53</td></tr><tr><td>√</td><td>√</td><td></td><td>62.35</td><td>64.02</td><td>63.45</td><td>70.05</td><td>56.11</td></tr><tr><td>√</td><td></td><td>√</td><td>62.53</td><td>64.15</td><td>64.07</td><td>70.19</td><td>55.74</td></tr></table></body></html>

Ablation of Weakly Supervised Training We further examined the effect of EMA on the pseudo labeling and false negative handling strategies within the MPL framework with results shown in Table 3. The findings demonstrate that EMA helps improve model performance, confirming the effectiveness of EMA for the pseudo-labeling module. Additionally, converting false negatives within batches can also enhance model performance consistently.

To further explore the effect of the similarity thresholds $\phi$ on FNE and FNC, we recorded the performance of MPL with different threshold values as shown in Figure 5.

Ablation of momentum Referring to the settings of the momentum coefficient in MoCo (He et al. 2020), we considered the following momentum values $\gamma$ as shown in Table 4. The results represent the accuracy of the model on the Flickr30K validation set under different momentum values.

Table 4: The impact of the momentum coefficient on model performance on the Flickr30k validation set.   

<html><body><table><tr><td>momentum y</td><td>0</td><td>0.9</td><td>0.99</td><td>0.999</td></tr><tr><td>Accuracy(%)</td><td>61.59</td><td>62.22</td><td>62.25</td><td>62.20</td></tr></table></body></html>

# Case Study

To more vividly demonstrate the effectiveness of our method in the pseudo-labeling and prediction, we visualized some of the pseudo-labels during the training phase. The results in the upper part of Figure 6 are from CLEM, while the lower part shows the results from our method. As shown in Figure 6, the left images depicts the top three confidence target regions for a phrase during training; the right images show the predictions of our model. The dark blue box represents the golden box (invisible during training). The orange box in the images on the right represents prediction results. Where Orange, yellow, and sky blue boxes indicate regions with the top 1, 2, and 3 confidence levels, respectively.

# Related Work

Weakly Supervised Phrase Grounding In WSPG tasks, previous studies have been divided into three primary classifications. The first strategy employs a multimodal maskreconstruction loss to enhance the model’s capacity in comprehending intricate connections between images and captions (Li et al. 2019, 2021; Dou et al. 2021; Zhao et al. 2023). The second strategy incorporates knowledge distillation from multimodal models like BLIP (Li et al. 2022) to utilize their captioning abilities and transform weakly supervised assignments into fully supervised ones (Jiang et al. 2022; Liu et al. 2023) or use the attention-based heatmaps generated by multimodal models to achieve weakly-supervised grounding (Shaharabany, Tewel, and Wolf 2022; Shaharabany and Wolf 2023; Lin et al. 2024). The final approach implements the Expectation-Maximization (EM) algorithm (Moon 1996) to allot and regularly update pseudo-labels for phrase-region pairings, contributing to contrastive learning.

![](images/6bc49e4a7a3b9b8009cba286d7fadd473f02a14d089530240a73792547043308.jpg)  
Figure 6: Visualization of pseudo-labels (left) on the Flickr30k and prediction (right) on the RefCOCO+.

False Negative Detection The effectiveness of contrastive learning is limited by negative example sampling, making false negative detection crucial for self-supervised representation learning and cross-modal retrieval. Huynh et al. (2022) applied strategies to filter similar images and reduce false negatives in visual representation learning. In weakly supervised visual-audio tasks, Sun et al. (2023) utilized a uni-modal similarity matrix to mitigate the influence of false negatives and enhance true negatives to improve visualaudio alignment. Some works (Li et al. 2023, 2024) attempt to improve the image-text retrieval performance by correcting false negatives and selecting negative examples.

# Conclusion

We introduce a novel method called Momentum Pseudo Labeling (MPL) that leverages a momentum model to compute pseudo-labels. Building on this foundation, we explore and model the relationships between phrases and regions in both matching and non-matching image-caption pairs. Empirical experiments demonstrate that our MPL method provides more effective guidance during the training stages.