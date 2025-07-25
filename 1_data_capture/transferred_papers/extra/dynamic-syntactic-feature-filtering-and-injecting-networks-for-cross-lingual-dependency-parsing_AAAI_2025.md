# Dynamic Syntactic Feature Filtering and Injecting Networks for Cross-lingual Dependency Parsing

Jianjian Liu, Zhengtao Yu, Ying $\mathbf { L i ^ { * } }$ , Yuxin Huang, Shengxiang Gao

Yunnan Provincial Key Laboratory of Artificial Intelligence, Faculty of Information Engineering and Automation, Kunming University of Science and Technology, Kunming 650000, China jjliu_nj $@$ foxmail.com, ztyu@hotmail.com, {yingli_hlt, gaoshengxiang.yn} $@$ foxmail.com, huangyuxin2004@163.com

# Abstract

Pre-trained language models enhanced parsers have achieved outstanding performance in rich-resource languages. Crosslingual dependency parsing aims to learn useful knowledge from high-resource languages to alleviate data scarcity in low-resource languages. However, effectively reducing the syntactic structure distributional bias and excavating the commonalities among languages is the key challenge for crosslingual dependency parsing. To address this issue, we propose novel dynamic syntactic feature filtering and injecting networks based on the typical shared-private model that employs one shared and two private encoders to separate source and target language features. Concretely, a Language-Specific Filtering Network (LSFN) on private encoders emphasizes helpful information and ignores the irrelevant or harmful parts of it from the source language. Meanwhile, a LanguageInvariant Injecting Network (LIIN) on the shared encoder integrates the advantages of BiLSTM and improved Transformer encoders to transcend language boundaries, thus amplifying syntactic commonalities across languages. Experiments on seven benchmark datasets show that our model achieves an average absolute gain of $1 . 8 4 ~ \mathrm { U A S }$ and 3.43 LAS compared with the shared-private model. Comparative experiments validate that both LSFN and LIIN components are complementary in transferring beneficial knowledge from source to target languages. Detailed analyses highlight that our model can effectively capture linguistic commonalities and mitigate the effect of distributional bias, showcasing its robustness and efficacy.

# Code — https://github.com/Flamelunar/crosslingualDP

# Introduction

The purpose of dependency parsing is to identify and describe the syntactic and grammatical relationships between input words via a dependency tree. Figure 1 depicts a dependency tree, where a dependency arc from the head word “hợp đồng (contract)” to the modifier word “vô hiệu (invalid)” with relation label “amod” indicates that “vô hiệu (invalid)” serves as an adjective to modify “hợp đồng (contract)”. These dependency trees reveal the syntax information via a hierarchical structure which is easily injected into various artificial intelligence models, i.e., neural machine translation (Yamin, Sarno, and Tambunan 2024), grammatical error correction (Tang, Qu, and Wu 2024), and question answering (Hou et al. 2024).

![](images/d52e67142ded53829fe45429fd5b9a618ae72f1a4c0ec3d7f65d3ce7f65eb7c1.jpg)  
Figure 1: Examples of the dependency trees, with the Chinese on the top and the Vietnamese on the bottom.

Recent researchers have focused on integrating strong representations of pre-trained language models into dependency parsers, enhancing the parsing performance significantly (Ross, Cai, and Min 2020; Yao, Xue, and Min 2022; Nishida and Matsumoto 2022a; Gu et al. 2024). In its early stages, Dozat and Manning (2017) propose a BiAffine parser which employs GloVe word embeddings (Pennington, Socher, and Manning 2014) as its inputs and utilizes a multi-layer BiLSTM to encode contextual information, achieving excellent performances on multiple languages. Li et al. (2019a) improve the BiAffine parser by integrating the representations of ELMo (Peters et al. 2018) and BERT (Devlin et al. 2019). Nguyen (2020) utilize the fine-tuned BERT model and extra POS tagging task to improve the Vietnamese parsing performance. However, the effectiveness of these models is still limited for low-resource languages due to insufficient training data (Rotman and Reichart 2019; Wang et al. 2020; Effland and Collins 2023).

To mitigate the scarce corpus of low-resource languages, cross-lingual dependency parsing has gained more attention, which mainly transfers useful knowledge from rich-resource languages to enhance parsing accuracy in low-resource languages (Schuster et al. 2019; Lauscher et al. 2020; Ansell et al. 2021). Sun, Li, and Zhao (2023) employ a self-training strategy to construct pseudo corpus and a multi-task framework to capture invariant linguistic features, thus improving the parsing performance of low-resource languages. Choudhary and O’riordan (2023) incorporate linguistic typological knowledge into a multi-task learning framework to enhance cross-lingual knowledge transfer. However, these approaches often focus on capturing linguistic invariant features, ignoring language distributional bias and in-depth commonalities. As illustrated in Figure 1, both the target language Vietnamese and the source language Chinese share a "subject-predicate-object" main syntactic structure. But as shown by the red arrow, the Vietnamese usually adopt a postmodifiers pattern while the Chinese use a pre-modifiers pattern. Therefore, it is extremely important and challenging to excavate in-depth linguistic commonalities and filter beneficial knowledge from the source language (Yuan, Jiang, and Tu 2019; Sun, Li, and Zhao 2023; Huang et al. 2024; Sherborne, Hosking, and Lapata 2023).

To address these challenges, we propose novel dynamic syntactic feature filtering and injecting networks for crosslingual dependency parsing. We first utilize the traditional shared-private model to yield original language-specific and language-invariant features by separated BiLSTM encoders. Then, a Language-Specific Filtering Network (LSFN) is applied in private encoders to emphasize useful source linguistic features and ignore detrimental ones. Simultaneously, we exploit a Language-Invariant Injecting Network (LIIN) on the shared encoder to enhance syntactic commonalities across languages by integrating the advantages of BiLSTM and improved Transformer encoders. Finally, we substitute the Multi-Layer Perception (MLP) with the KolmogorovArnold Network (KAN) (Liu et al. 2024) to enrich syntax features more flexibly and dynamically. Experiments on seven benchmark datasets show that our model achieves average improvements of $1 . 8 4 / 3 . 4 3 \$ points in the UAS/LAS scores compared with the strong shared-private model, leading to new state-of-the-art results on all datasets. Comparison experiments demonstrate that LSFN can filter beneficial information from the source language to decrease language distributional bias and LIIN is helpful in excavating language-invariant features. Detailed analyses further prove these components complement each other with light parameters, greatly improving the cross-lingual dependency parsing performance.

# Related Works

Cross-lingual dependency parsing leverages syntactic information from rich-resource languages to enhance parsing accuracy in low-resource languages (Zhang 2020; Xu et al. 2022; Zhao et al. 2024). Early approaches predominantly utilize projection-based techniques and annotation transfer (Xiao and Guo 2015). Xiao and Guo (2014) and Guo et al. (2015) employ distributed representations to map lexical features across languages, facilitating the capture of linguistic structures. Similarly, Tiedemann and Agic´ (2016) and

Lacroix et al. (2016) introduce several parsers from partially annotated data through annotation projection. Despite their innovation, these methods are limited by the quality and availability of parallel data and the syntactic divergences between languages.

The emergence of Transformer architectures (Vaswani et al. 2017) and pre-trained language models (PLMs) like BERT (Devlin et al. 2019), XLM-RoBERTa (Conneau et al. 2020), and BART (Lewis et al. 2020) revolutionize crosslingual dependency parsing. Researchers usually extract robust contextual representations from these PLMs to enhance traditional parsers or fine-tune PLMs for adapting parsing tasks. Kumar et al. (2022) use word-to-word dependency tagging features from BERT to enhance the malt parser, tackling data imbalance and consequently improving parsing results. Choenni, Garrette, and Shutova (2023) finetune mBERT assisted with language-specific subnetworks for controlling cross-lingual parameter sharing, improving low-resource languages parsing accuracy. However, the improvement in syntactic analysis for low-resource languages is still limited, since these PLMs are initially trained on a small-scale low-resource context (Haque, Liu, and Way 2021; Min et al. 2023).

Recent advancements in cross-lingual dependency parsing emphasize the transfer of explicit linguistic typology knowledge to reduce the interference of language differences and enhance generalization for low-resource languages (Choudhary and O’riordan 2023; Danilova and Stymne 2024; Kunz and Holmstro¨m 2024). Choudhary and O’riordan (2023) utilize multi-task Learning to transfer linguistic typology knowledge from the source language to enhance the target language parsing performance. Danilova and Stymne (2024) use topic modeling to assist cross-genre transfer and gain promising parsing performance. Despite prior researches achieving good results, excavating the implicit commonalities across different languages and mitigating distributional deviation still hinder current cross-lingual dependency parsing. Motivated by Wu et al. (2021) and Li et al. (2022), we propose dynamic syntactic feature filtering and injecting networks to filter beneficial information from the source language and extract deeper syntactic commonalities across different languages.

# Our Approach

The typical shared-private model for cross-lingual dependency parsing employs a shared encoder to extract languageinvariant features and multiple private encoders to capture language-specific features (Nishida and Matsumoto 2022b). However, this model treats all features equally and fails to construct linguistic differences and comprehensive linguistic links. Inspired by Li et al. (2022) and Gu et al. (2024), we propose dynamic syntactic feature filtering and injecting networks based on the shared-private model. On the one hand, a Language-Specific Filtering Network (LSFN) is applied to private encoders to emphasize useful source linguistic features and ignore irrelevant or detrimental ones. On the other hand, a Language-Invariant Injecting Network (LIIN) is exploited on the shared encoder to enhance syntactic commonalities across languages. Figure 2 presents the overall architecture of our proposed model, which is organized into three components, i.e., Input, Cross-Lingual Encoder, and Decoder.

![](images/4766ae28493ceffa426ef6b99f3cc890e671d3e79d22b300b59454569bb0c1f2.jpg)  
Figure 2: The architecture of our model, where “LSFN” and “LIIN” are the Language-Specific Filtering Network and the Language-Invariant Injecting Network. “KAN” stands for the Kolmogorov-Arnold Network. $\cdot { { L } _ { 1 } } ^ { \prime \prime }$ and $^ { \cdot 6 } L _ { 2 } ^ { , 9 }$ means layer numbers. Dashed arrows indicate that the LSFN optimizes the private BiLSTM parameters. Solid lines in blue and red represent data flows from the source and target languages, while black solid lines represent shared data flows.

# Input Component

Given a sentence $w _ { 1 } , w _ { 2 } , \ldots , w _ { n }$ either in source or target language, the input layer converts them into highdimensional vectors $\pmb { x } _ { 1 } , \pmb { x } _ { 2 } , \ldots , \pmb { x } _ { n }$ . Different from the traditional shared-private model, we leverage the multilanguage pre-trained language model (XLM-RoBERTa)1 to enhance the representation capability of word vectors. As illustrated in Equation 1, each word vector $\mathbf { \boldsymbol { x } } _ { i }$ comprises its word representation and corresponding character representation $\mathbf { w o r d } _ { i } ^ { \mathrm { c h a r } }$ . The word representation is the combination of the XLM-RoBERTa output $\mathbf { r e p } _ { i } ^ { \mathrm { X L M - R } }$ and a randomly initialized word embedding $\mathbf { e m b } _ { i } ^ { \mathsf { w o r d } }$ . The character representation $\mathbf { w o r d } _ { i } ^ { \mathrm { c h a r } }$ is produced by a Char-BiLSTM network, which utilizes a one-layer BiLSTM to encode the characters of each word $w _ { i }$ and merges the hidden vectors from two directions (Lample et al. 2016).

$$
\pmb { x } _ { i } = ( \mathbf { r e p } ^ { \mathrm { X L M - R } _ { i } } + \mathbf { e m b } ^ { \mathrm { w o r d } _ { i } } ) \oplus \mathbf { w o r d } ^ { \mathrm { c h a r } _ { i } }
$$

# Cross-Lingual Encoder

The original shared-private model utilizes one shared and two private three-layer BiLSTMs to encode languageinvariant and language-specific features. However, these

BiLSTMs treat all syntactic features equally, thus limiting their ability to excavate the generality and discrepancy between source and target languages. To construct an in-depth relationship with the source language, we design a languagespecific filtering network on private BiLSTMs to dynamically emphasize helpful source features and ignore the harmful ones. Meanwhile, we exploit a language-invariant injecting network on the shared BiLSTM to extract in-depth commonalities.

Language-Specific Filtering Network (LSFN). In the shared-private model, the input sentence from the source or target language is fed into its private BiLSTM to obtain language-specific contextual representation $\pmb { c } _ { i } ^ { s r c }$ or $c _ { i } ^ { t g t }$

$$
\begin{array} { r } { \pmb { c } _ { i } ^ { s r c } = \mathrm { B i L S T M } ^ { s r c } ( \pmb { x } _ { i } ^ { s r c } , \theta _ { \mathrm { B i L S T M } ^ { s r c } } ) } \\ { \pmb { c } _ { i } ^ { t g t } = \mathrm { B i L S T M } ^ { t g t } ( \pmb { x } _ { i } ^ { t g t } , \theta _ { \mathrm { B i L S T M } ^ { t g t } } ) } \end{array}
$$

Considering some source language-specific syntactic features may benefit the target language dependency parsing, we design a language-specific filtering network on private BiLSTMs to construct the in-depth relationship between source and target representations. Motivated by Li, Li, and Zhang (2022), we minimize the $\mathcal { L } _ { 2 }$ distance between outputs of source and target private BiLSTMs to transfer useful information from source to target language, thus enriching the target language feature space.

$$
\mathcal { L } _ { 2 } = \| f _ { \theta } ( \pmb { t } _ { i } ^ { m } ) - \pmb { s } _ { i } ^ { n } \| _ { 2 } ^ { 2 }
$$

where $f _ { \theta } ( \cdot )$ denotes a linear transformation, $\mathbf { \Delta } _ { s _ { i } ^ { n } } ^ { n }$ is the $n ^ { t h }$ -layer BiLSTMsrc output, and $\pmb { t } _ { i } ^ { m }$ is the $m ^ { t h }$ -layer BiLSTM $t g t$ output.

Concretely, each input sentence of the target language is first fed into both source and target private BiLSTMs to acquire target language-specific representation $\mathbf { \Delta } _ { t _ { i } }$ and source representation $\mathbf { \boldsymbol { s } } _ { i }$ . Then, our filtering network is applied to the private BiLSTMs to screen out harmful source information and emphasize useful ones by mimicking the welltrained source representations. For each filtering pair $( n , m )$ between their private BiLSTMs are assigned learnable layer filtering weights $\mathbf { W } ^ { n , m }$ to determine the migration impact from BiLSTMsrc to $\mathrm { B i L S T M } ^ { t g t }$ . Moreover, our model learns element filtering weight Edn,m to filter useful elements from source representations.

$$
\begin{array} { r l } & { \mathbf { W } ^ { n , m } = q _ { \phi } ^ { n , m } ( \pmb { s } _ { i } ^ { n } ) } \\ & { \mathbf { E } _ { d } ^ { n , m } = \operatorname { s o f t m a x } \left( p _ { \phi } ^ { n , m } ( \pmb { s } _ { i } ^ { n } ) \right) _ { d } } \end{array}
$$

where $q _ { \phi } ^ { n , m }$ and $p _ { \phi } ^ { n , m }$ are a nonlinear and a linear transformation respectively. $\mathbf { E } _ { d } ^ { n , m }$ is the non-negative weight of the $d$ -th element in $( n , m ) . n , m \in \{ 1 , 2 , 3 \}$ are the layer numbers of source and target BiLSTMs. Once the optimal layer matching weights are learned, element filtering weights are optimized. The filtering loss for the pair $( n , m )$ is defined as follows,

$$
\mathcal { L } ^ { f i l } = \frac { 1 } { K D } \sum _ { n , m } \mathbf { W } ^ { n , m } \sum _ { d = 1 } ^ { D } \mathbf { E } _ { d } ^ { n , m } \left( f _ { \theta } ( \pmb { t } _ { i } ^ { m } ) _ { d } - ( \pmb { s } _ { i } ^ { n } ) _ { d } \right) ^ { 2 }
$$

where $D$ is the output dimension of BiLSTMs, and $K =$ $3 \times 3 = 9$ denotes the total filtering pairs.

Language-Invariant Injecting Network (LIIN). In the shared-private model, sentences from both source or target languages are encoded by shared BiLSTM to obtain language-invariant contextual representation $c _ { i } ^ { s h a }$ . Considering self-attention is more suitable for capturing longdistance dependencies due to its capability of directly building connections between distant word pairs, while BiLSTM may fade the long-distance information in the encoding process, we design a language-invariant injecting network below the shared BiLSTM to compensate for the above shortcomings. Concretely, each LIIN layer is composed of a multi-head self-attention and a Mamba sub-layer. First, the multi-head attention layer computes pairwise correlations between word vectors, thus generating attention scores that weight contributions based on their relevance. Meanwhile, self-attention enables parallel processing across multiple subspaces, thus capturing syntactic knowledge of diverse aspects and intricate patterns within the input data. The formulas are defined as follows,

$$
h e a d _ { i } = \mathrm { S o f t m a x } \left( \frac { \mathbf { Q } _ { i } \mathbf { K } _ { i } ^ { T } } { \sqrt { d _ { k } } } \right) \mathbf { V } _ { i }
$$

$$
\mathbf { a } _ { i } = ( h e a d _ { 1 } , \ldots , h e a d _ { n } ) \mathbf { W } ^ { 0 }
$$

where $\mathbf { Q } _ { i } , \mathbf { K } _ { i } , \mathbf { V } _ { i }$ are transformations of each word vector $\mathbf { \boldsymbol { x } } _ { i }$ by learnable matrices $\mathbf { W } ^ { Q } , \mathbf { W } ^ { K } , \mathbf { W } ^ { V } .$ $\mathbf { W } ^ { 0 }$ is a model parameter. Second, the Mamba sub-layer addresses limitations of static self-attention mechanisms by dynamically adjusting positional weights $\mathbf { A }$ and $\mathbf { B }$ , which are generated based on the relevance of words $\mathbf { \alpha } _ { \mathbf { \beta } } \mathbf { \alpha } _ { \mathbf { \beta } } \mathbf { \alpha } _ { \mathbf { \beta } } \mathbf { \alpha } _ { \mathbf { \beta } } \mathbf { \alpha } _ { \mathbf { \beta } } \mathbf { \alpha } _ { \mathbf { \beta } } \mathrm { ~ } \mathbf { \alpha } _ { \mathbf { \beta } } \mathrm { ~ } \mathbf { \alpha } _ { \mathbf { \beta } } \mathrm { ~ } \mathbf { \alpha } _ { \mathbf { \beta } } \mathrm { ~ } \mathrm { ~ } \mathbf { \alpha } _ { \mathbf { \beta } } \mathrm { ~ } \mathrm { ~ } \mathbf { \alpha } _ { \mathbf { \beta } } \mathrm { ~ \mathrm { ~ } \ \ } \mathbf { \alpha } _ { \mathbf { \beta } } \mathrm { ~ \mathrm { ~ } \ \ } \mathrm { ~ \ \ } \mathbf { \alpha } _ { \mathbf { \beta } } \mathrm _ { \mathrm { ~ \alpha } \mathbf { \beta } }$ and their corresponding state vector $\mathbf { \delta } _ { h _ { i } }$ . This adaptability enhances computational efficiency and improves the parsing of complex syntactic structures. The formulas are as follows,

$$
\begin{array} { r } { \hat { h } _ { i } = \mathbf { A } ( \mathbf { B } \pmb { a } _ { i - 1 } ) + \mathbf { B } \pmb { a } _ { i } } \\ { \pmb { h } _ { i } = \mathbf { C } \hat { \pmb { h } } _ { i } } \end{array}
$$

where $\hat { h } _ { 1 } = \mathbf { B } a _ { 1 }$ . A, B, and $\mathbf { C }$ are adaptive weight matrices tailored to syntactic positions. Next, vectors $h _ { 1 } , h _ { 2 } , \ldots , h _ { n }$ are fed into the shared BiLSTM to learn contextual word information.

$$
\pmb { c } _ { i } ^ { s h a } = \mathrm { B i L S T M } ^ { s h a } ( \pmb { h } _ { i } , \theta _ { \mathrm { B i L S T M } ^ { s h a } } )
$$

where $\theta _ { \mathrm { B i L S T M } }$ is the shared BiLSTM parameters. Finally, we obtain the cross-lingual encoder output $\boldsymbol { c } _ { i }$ by adding $\pmb { c } _ { i } ^ { s r c }$ and $c _ { i } ^ { s h a }$ for source language data, or $c _ { i } ^ { t g t }$ and $c _ { i } ^ { s h a }$ for target language data.

$$
\pmb { c } _ { i } = \left\{ \begin{array} { l l } { \pmb { c } _ { i } ^ { s h a } + \pmb { c } _ { i } ^ { s r c } , l = s r c } \\ { \pmb { c } _ { i } ^ { s h a } + \pmb { c } _ { i } ^ { t g t } , l = t g t } \end{array} \right.
$$

where $l$ is the language type of current data.

# Decoder Component

Kolmogorov-Arnold Network (KAN). Unlike the sharedprivate model, we substitute the traditional Multi-Layer Perception (MLP) with KAN (Liu et al. 2024). The MLP aims to downscale contextualized word representations and capture syntactic features. However, its fixed activation function may limit the expressiveness and interpretability of syntax information extraction. To address this issue, we replace it with the KAN, which uses learnable nonlinear activation functions to replace MLP weights, thus enriching syntactic features flexibly. In general, the KAN takes contextualized word representation $\boldsymbol { c } _ { i }$ as input and obtain its low-dimension head representations $( \pmb { k } _ { i } ^ { h } \ \pmb { \mathcal { E } } \ \pmb { k } _ { i } ^ { h ^ { \prime } } )$ and modifier representations $( \hat { \mathbf { k } _ { i } ^ { d } } \& \mathbf { k } _ { i } ^ { d ^ { \prime } } )$ .

$$
\begin{array} { r l r } & { } & { { \pmb k } _ { i } ^ { h } , { \pmb k } _ { i } ^ { d } , { \pmb k } _ { i } ^ { h ^ { \prime } } , { \pmb k } _ { i } ^ { d ^ { \prime } } = { \pmb { \mathrm { K A N } } } _ { h } ( { \pmb c } _ { i } ) , { \pmb \mathrm { K A N } } _ { d } ( { \pmb c } _ { i } ) , } \\ & { } & { { \pmb \mathrm { K A N } } _ { h ^ { \prime } } ( { \pmb c } _ { i } ) , { \pmb \mathrm { K A N } } _ { d ^ { \prime } } ( { \pmb c } _ { i } ) } \end{array}
$$

The formula of one-layer KAN is defined as follows,

$$
\begin{array} { l } { { \displaystyle { k _ { i , b } = \sum _ { a = 1 } ^ { n _ { \mathrm { i n } } } \sum _ { i = 1 } ^ { n } \psi _ { i , a , b } ( { \bf c } _ { i , a } ) } } } \\ { { \displaystyle { k _ { i } = ( k _ { i , 1 } , \cdot \cdot \cdot \ , k _ { i , b } , \cdot \cdot \cdot k _ { i , n _ { \mathrm { o u t } } } ) } } } \end{array}
$$

where $n$ is the word number of the entered sentences. $n _ { \mathrm { i n } }$ and $n _ { \mathrm { o u t } }$ are the input and output dimensions of KAN, respectively. $\boldsymbol { c } _ { i , a }$ means the $a$ -th dimension of input representation $\mathbf { \Lambda } _ { c _ { i } . \ k _ { i , b } }$ means the $b$ -th dimension of output representation $\boldsymbol { k } _ { i }$ . $\psi _ { i , a , b }$ represents a parameterized learnable nonlinear activation function that establishes the syntactic association between the $a$ -th dimension of all input word vectors and the $b .$ -th dimension of each corresponding output word vector. Finally, each output vector $\mathbf { \Delta } _ { k _ { i } }$ is obtained by concatenating all dimension representations $\boldsymbol { k } _ { i , b }$ where $\bar { b \in \{ 1 , 2 , \dots , n _ { \mathrm { o u t } } \} }$ .

Algorithm 1: Cross-lingual Training Procedure.   
Input: Source language data $S$ , target language data $T$ Parameter: Loss weight $\alpha$ , training iterations $k$ .   
Output: Result of parsing   
1: Initialize $i t e r = 0$   
2: while iter ${ \bf \Psi } = k { \bf \Psi }$ or convergence do   
3: Select mini-batch $x$ alternately from $S$ or $T$   
4: if $x \in S$ then   
5: Compute parser loss ${ \mathcal { L } } = { \mathcal { L } } ^ { p a r }$   
6: Update parser, LIIN parameters by minimizing $\mathcal { L }$ 7: else if $x \in T$ then   
8: Compute final loss $\mathcal { L } = \mathcal { L } ^ { p a r } + \alpha \mathcal { L } ^ { f i l }$   
9: Update all parameters by minimizing $\mathcal { L }$   
10: end if   
11: $i t e r + = 1$   
12: end while

BiAffine Layer. The dependency arc score between the modifier word $w _ { j }$ and its head word $w _ { i }$ is $\sec ( i  j )$ which is computed by a BiAffine operation. Simultaneously, the score of dependency label score $( i  j )$ is calculated by another separated BiAffine operation as equation 12.

$$
\mathrm { s c o r e } ( i \gets j ) = \left[ \begin{array} { c } { { \pmb { k } _ { i } ^ { d } } } \\ { { 1 } } \end{array} \right] ^ { \mathrm { T } } { \mathbf { U } _ { 1 } \pmb { k } _ { j } ^ { h } }
$$

$$
\mathrm { s c o r e } ( i \ :  \ : j ) = k _ { j } ^ { h ^ { \prime } } \mathbf { U } _ { 2 } \ : k _ { i } ^ { d ^ { \prime } } + ( k _ { j } ^ { h ^ { \prime } } \oplus k _ { i } ^ { d ^ { \prime } } ) \mathbf { U } _ { 3 } + b
$$

where $\mathbf { U } _ { 1 } , \mathbf { U } _ { 2 }$ , $\mathbf { U } _ { 3 }$ , and $b$ are parameters. $l$ denotes the relation label. The Maximum Spanning Tree (MST) algorithm is used to find the highest-score tree as the final parsing result.

Parser Loss. For each position $i$ , if the gold-standard head word $w _ { i }$ modifies word $w _ { j }$ with relation label $l$ , the parsing loss is computed as follows,

$$
\begin{array} { c } { { { \mathcal { L } } ^ { p a r } = - \log \displaystyle \frac { e ^ { \mathrm { s c o r e } ( i \cdot \cdot - j ) } } { \displaystyle \sum _ { 0 \leq k \leq n , k \neq i } e ^ { \mathrm { s c o r e } ( i \cdot  k ) } } } } \\ { { - \log \displaystyle \frac { e ^ { \mathrm { s c o r e } ( i \cdot ^ { \angle } { j } ) } } { \sum _ { l ^ { \prime } \in L } e ^ { \mathrm { s c o r e } ( i \cdot ^ { \ell ^ { \prime } } { j } ) } } } } \end{array}
$$

where $L$ refers to the collection of all dependency labels.

# Cross-lingual Training

In this work, we propose a cross-lingual training strategy to take advantage of both source and target languages as Algorithm 1. We sample mini-batch $x$ from source or target language data alternately. If $x$ belongs to the source language $S$ , we only update the parameters of the shared-private parser and LIIN by minimizing parsing loss. While $x$ comes from the target language $T$ , we update all parameters by minimizing parsing and filtering loss. Finally, we iteratively train all data until it converges or stops prematurely.

# Experiments Experimental Setups

Datasets. We conduct experiments on seven low-resource languages, i.e., Vietnamese (vi), Wolof (wo), Coptic (cop),

Table 1: Dataset statistics in sentence number.   

<html><body><table><tr><td>Dataset</td><td>Train</td><td>Dev</td><td>Test</td><td>All</td></tr><tr><td>English (EWT)</td><td>12,544</td><td>2,001</td><td>2,077</td><td>16,622</td></tr><tr><td>Chinese (GSDSimp)</td><td>3,997</td><td>500</td><td>500</td><td>4,997</td></tr><tr><td>Vietnamese (VTB)</td><td>1,400</td><td>1,123</td><td>800</td><td>3,323</td></tr><tr><td>Wolof (wTB)</td><td>1,188</td><td>449</td><td>470</td><td>2,107</td></tr><tr><td>Coptic (Scriptorium)</td><td>1,419</td><td>381</td><td>403</td><td>2,203</td></tr><tr><td>Maltese (MUDT)</td><td>1,123</td><td>433</td><td>518</td><td>2.074</td></tr><tr><td>Tamil (TTB)</td><td>400</td><td>80</td><td>120</td><td>600</td></tr><tr><td>Uyghur (UDT)</td><td>1,656</td><td>900</td><td>900</td><td>3,456</td></tr><tr><td>Thai (TUD)</td><td>2.902</td><td>362</td><td>363</td><td>3,627</td></tr></table></body></html>

Maltese (mt), Tamil (ta), Uyghur (ug), and Thai (th) where six languages are sourced from the Universal Dependencies (UD) v2.13 treebank2, while the Thai dataset comes from the TUD corpus3. According to the similar language family, we select Chinese as the source language for Vietnamese, Tamil, Uyghur, and Thai, while English is the source language for Wolof, Coptic, and Maltese. Detailed dataset information is presented in Table 1.

Evaluation. We utilize Labeled Attachment Score (LAS) and Unlabeled Attachment Score (UAS) as evaluation metrics (Hajic et al. 2009). All models are trained with no more than 1, 000 iterations, and their performances are evaluated on the development dataset after each iteration to guide the model selection. Model training is stopped if the peak performance does not increase for 20 consecutive iterations.

Hyper-parameter choices. We follow the most hyperparameter settings of Li et al. (2019a), including MLP and BiAffine dimensions and learning rates. In addition, attention and Mamba components have 3 layers with a 0.5 dropout rate. The KAN uses a 0.33 dropout rate and its basic activation is initialized with the SiLU function. The loss weight $\alpha$ for the LSFN is set as 1.

Baselines. We reproduce the following baseline models for our comparative experiments.

• Fully shared method (FulSha). Peng, Thomson, and Smith (2017) utilize the fully shared encoder parameters to construct the commonality cross three dependency graph formalisms, thus enhancing heterogeneous dependency parsing performance. Here, we directly train the BiAffine parser with both source and target language data, which treats training data equally and shares all parameters between different languages.

• Language embedding method (LanEmb). Li et al. (2019b) use domain embeddings as an extra input to indicate the domain type of each word, which is proved effective for cross-domain dependency parsing. Motivated by this work, we also leverage 8 dimension language embeddings to guide the model to identify language types.

• Shared-private method (ShaPri). Wu et al. (2021) introduce a text-centred shared-private framework to capture shared semantic features and distinguish private ones across modalities. Inspired by this work, we exploit a shared-private framework to capture language-invariant and language-specific features via separated BiLSTMs.

Table 2: Main results of seven languages on the test dataset.   

<html><body><table><tr><td></td><td colspan="2">vi</td><td colspan="2">wo</td><td colspan="2">cop</td><td colspan="2">mt</td><td colspan="2">ta</td><td colspan="2">ug</td><td colspan="2"></td><td colspan="2">avg.</td></tr><tr><td>Model</td><td>UAS</td><td>LAS</td><td>UAS</td><td>LAS</td><td>UAS</td><td>LAS</td><td>UAS LAS</td><td></td><td>UAS LAS</td><td></td><td>UAS LAS</td><td></td><td>UAS</td><td>LAS</td><td>UAS</td><td>LAS</td></tr><tr><td></td><td colspan="10">Results of previous works</td><td colspan="7"></td></tr><tr><td>UDPipe(2019)</td><td>70.38 62.56</td><td></td><td></td><td>=</td><td>85.58 80.97</td><td></td><td></td><td>1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>74.11 66.37 78.46 67.09 82.34 75.28 78.17 70.45</td></tr><tr><td>UDify(2019)</td><td>74.11 66.00</td><td></td><td></td><td></td><td>27.5810.82 83.07 75.56</td><td></td><td></td><td></td><td>1</td><td>1</td><td>65.89 48.80</td><td></td><td></td><td>1</td><td></td><td>62.66 50.45</td></tr><tr><td>MBERT(2022)</td><td></td><td></td><td></td><td>72.94</td><td></td><td>82.11</td><td></td><td>72.69</td><td></td><td>54.94</td><td>、</td><td>42.97</td><td></td><td>1</td><td>1</td><td>65.13</td></tr><tr><td>ESR(2023)</td><td></td><td>60.80</td><td></td><td>73.30</td><td></td><td></td><td></td><td>74.20</td><td></td><td>66.40</td><td>1</td><td>39.20</td><td></td><td>1</td><td></td><td>62.78</td></tr><tr><td></td><td colspan="10">Compare with baseline models</td><td colspan="3"></td><td colspan="3"></td></tr><tr><td>FulSha</td><td></td><td>78.94 62.53 81.99 74.25 85.60 79.28 81.61 72.79 77.23 63.15 78.40 63.45 83.12 70.99 80.98 69.49</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LanEmb</td><td></td><td>79.28 63.52 82.47 75.18 85.52 79.14 81.74 73.01 78.18 64.25 78.56 63.77 83.08 70.96 81.26 69.98</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ShaPri</td><td>78.61 63.36 82.18 74.95 88.29 83.83 81.31 73.46 76.77 63.70 77.82 63.50 83.12 71.18 81.16 70.57</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Our</td><td></td><td>80.03 66.75 83.20 76.34 89.95 86.32 83.28 76.19 79.09 69.18 79.98 67.67 85.45 75.52 83.00 74.00</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table></body></html>

# Experimental Results

Table 2 presents the final results of seven languages on the test dataset. First, we can see that the “LanEmb” model outperforms the “FulSha” model, indicating that extra language embeddings can help the parser effectively distinguish the source and target languages. Second, compared with “FulSha” and “LanEmb”, the “ShaPri” model further improves the parsing accuracy, demonstrating that separated features can initially construct the commonalities and differences between different languages. Finally, our model achieves the best performance among all strong baselines, illustrating its effectiveness in meticulously capturing invariant syntactic features and distinguishing linguistic structural differences.

We also compare our model with several previous works. Straka, Straková, and Hajic (2019) present a UDpipe framework, which is jointly trained with tokenization, POS tagging, and dependency parsing sub-tasks on multiple languages to enhance parsing accuracy. Then, Kondratyuk and Straka (2019) propose a UDify framework which fine-tunes a multilingual BERT model in 104 languages to enhance parsing accuracy. Moreover, Gessler and Zeldes (2022) employ a vocabulary expansion method and fine-tune the BERT for parsing. Lastly, Effland and Collins (2023) adopt an expected statistic regularization that utilizes low-order multi-task structural statistics to shape model distributions to improve dependency parsing performance. Compared with these models, our approach achieves superior performance with only a single source language, highlighting its efficiency and robustness.

Table 3: Ablation study on Vietnamese dev dataset.   

<html><body><table><tr><td>Model</td><td>Parameter (M)</td><td>Time (s)</td><td>UAS</td><td>LAS</td></tr><tr><td>Our model</td><td>310.4</td><td>81</td><td>76.53</td><td>61.65</td></tr><tr><td>w/o LSFN</td><td>304.9</td><td>69</td><td>75.58</td><td>60.90</td></tr><tr><td>w/o LIIN</td><td>306.5</td><td>79</td><td>75.81</td><td>60.30</td></tr><tr><td>w/o Two</td><td>301.0</td><td>67</td><td>75.04</td><td>59.52</td></tr></table></body></html>

# Ablation Study

Table 3 shows the ablation study results on dev data. First, although our model introduces 9.4 million additional parameters, it only increases training time by 14 seconds and achieves a $2 . 1 3 ~ \mathrm { L A S }$ improvement, proving the efficiency and lightweight of our model. Second, we can see that removing the LSFN (“w/o LSFN”) or LIIN (“w/o LIIN”) component leads to an obvious decrease in parsing performance, demonstrating that each module is crucial for mitigating conflicts from direct language transfer and effectively helps our model to learn language commonalities and differences. Then, removing the LIIN and LSFN components simultaneously reduces dependency parsing accuracy significantly, indicating they are complementary and can benefit from each other. Finally, these observations emphasise the importance of retaining language-invariant features and reinforce the need for strategic filtering of source languagespecific features.

# Error Analysis

Sentence lengths. Figure 3 illustrates the LAS across various sentence lengths on Vietnamese dev data. All models perform better on shorter sentences. For sentences under 10 words, the LAS ranges from 67 to 70. However, there is a significant decline of over 13 points for sentences around 60 words, highlighting the increased parsing difficulty with longer sentences. The “ShaPri” model consistently records the lowest scores across all length categories. In contrast, the “ShaPri” model with LSFN and LIIN achieves higher scores across all lengths, indicating their effectiveness in learning cross-lingual commonalities and filtering out irrelevant or harmful source language information. Finally, our model achieves the highest scores in all sentence lengths, demonstrating that our model greatly enhances short- and long-range syntactic dependency parsing capabilities.

![](images/60c7cf3593d936c365d0bf18cff4b534e345ff954b5f9901b385d8baa9aed4e1.jpg)  
Figure 3: LAS regarding diverse sentence lengths.

![](images/88af50ab4174fe915f8534e2963221cad3a7b2db63e62efc2a621d2b9cbb3db4.jpg)  
Figure 4: LAS curves regarding dependency distances.

![](images/7c4606076c9456778ce7b911ed2e393f4528d34c13605836ded90a03a81f6757.jpg)  
Figure 5: Dependency label distributional biases analysis.

Dependency Distances. Figure 4 presents LAS based on absolute dependency distances between head and modifier words on Vietnamese dev data. The “ShaPri” model consistently performs the lowest across most distances. On the contrary, “w/o LSFN” and “w/o LIIN” models acquire higher accuracies across most distances, suggesting their superior capability to capture language-invariant features and filter out language-specific noise. Our model shows significant improvements across all distances, highlighting the parsing ability of our model on both short- and long-range dependency distances.

Distributional biases. Figure 5 illustrates the dependency label distributions across various data sources, where the percentage is a certain label number divided by the total label number. First, the Chinese and Vietnamese golden UD test data, denoted by the solid blue and red curves respectively, exhibit notable distributional differences. For example, the “nmod” label percentage is 14.1 in Chinese versus 4.9 in Vietnamese. Then, the “ShaPri” model predicted label distribution for Vietnamese deviates substantially from the Vietnamese golden data distribution, instead approximating the Chinese golden data distribution. This is attributable to excessive transfer stemming from the resource imbalance between Chinese and Vietnamese data. Finally, our proposed method effectively mitigates this excessive transfer, resulting in a predicted Vietnamese label distribution that more closely aligns with the true Vietnamese distribution, thus demonstrating the validity and robustness of our model.

Table 4: Comparative experiments between MLP and KAN on Vietnamese dev dataset.   

<html><body><table><tr><td>Model</td><td colspan="2">Models withMLP</td><td colspan="2">Models with KAN</td></tr><tr><td></td><td>UAS</td><td>LAS</td><td>UAS</td><td>LAS</td></tr><tr><td>FulSha</td><td>75.04</td><td>57.63</td><td>74.83</td><td>57.89</td></tr><tr><td>ShaPri</td><td>75.48</td><td>59.24</td><td>75.04</td><td>59.52</td></tr><tr><td>Our</td><td>76.02</td><td>61.23</td><td>76.53</td><td>61.65</td></tr></table></body></html>

# Effect on Different Syntax Extraction Strategies

Table 4 compares two different syntax extraction strategies on three models. First, KAN consistently achieves higher LAS scores than MLP across all models, indicating that KAN can more effectively capture complex syntactic features and patterns. Then, “FulSha” and “ShaPri” with KAN slightly improve the UAS scores, possibly since KAN associates each output word with all input words which may yield a small amount of noise or interference. Finally, our model with KAN achieves the best performance, further demonstrating that our model can filter out irrelevant and harmful information to help KAN extract syntax features more accurately. In a word, compared with MLP, KAN can extract syntax features more flexibly and comprehensively.

# Conclusion

In this work, we propose dynamic syntactic feature filtering and injecting networks for cross-lingual dependency parsing, where the LSFN emphasizes helpful information from the source language and LIIN excavates commonalities across different languages simultaneously. Experiments on seven benchmark datasets exhibit that our efficient yet lightweight model consistently outperforms all strong baselines, leading to new state-of-the-art results on all datasets. Detailed comparative experiments confirm that both LSFN and LIIN can effectively transfer valuable knowledge from the source language to the target language and benefit from each other. Further analysis demonstrates that our model has outstanding capability to capture long sentences and remote dependencies. In addition, the comparison between MLP and KAN verifies that KAN can extract syntax features more flexibly and comprehensively.