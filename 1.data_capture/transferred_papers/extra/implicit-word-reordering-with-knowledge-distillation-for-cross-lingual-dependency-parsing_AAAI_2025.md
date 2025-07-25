# Implicit Word Reordering with Knowledge Distillation for Cross-Lingual Dependency Parsing

Zhuoran Li, Chunming Hu, Junfan Chen, Zhijun Chen, Richong Zhang

SKLSDE, Beihang University, Beijing, China lizhuoranget, hucm, zhijunchen $@$ buaa.edu.cn, chenjf, zhangrc @act.buaa.edu.cn

# Abstract

Word order difference between source and target languages is a major obstacle to cross-lingual transfer, especially in the dependency parsing task. Current works are mostly based on order-agnostic models or word reordering to mitigate this problem. However, such methods either do not leverage grammatical information naturally contained in word order or are computationally expensive as the permutation space grows exponentially with the sentence length. Moreover, the reordered source sentence with an unnatural word order may be a form of noising that harms the model learning. To this end, we propose an Implicit Word Reordering framework with Knowledge Distillation (IWR-KD). This framework is inspired by that deep networks are good at learning feature linearization corresponding to meaningful data transformation, e.g. word reordering. To realize this idea, we introduce a knowledge distillation framework composed of a wordreordering teacher model and a dependency parsing student model. We verify our proposed method on Universal Dependency Treebanks across 31 different languages and show it outperforms a series of competitors, together with experimental analysis to illustrate how our method works towards training a robust parser.

# Introduction

Dependency parsing is a fundamental task which aims to extract the low-level grammatical relationships between words in a sentence (Mcdonald 2006; Nivre 2008; Kiperwasser and Goldberg 2016), such as subject-verb relationships. Recently, cross-lingual dependency parsing has attracted considerable attention from academic and industrial communities, for which a parser is trained on a source language and directly applied in the target language of interest. Multilingual pre-trained language models (mPLMs) demonstrate exceptional performance in cross-lingual dependency parsing (Devlin et al. 2019; Conneau et al. 2020). However, these mPLMs inevitably encode word order features to model contextual representations, referred to as order-sensitive (Ahmad et al. 2019; Liu et al. 2021). Since word order is inherently different across languages, there is a risk of over-fitting into the word order of the source language that could hurt the performance in the target languages.

![](images/4a1c0fe4ed7c643e86a88682c9ed8d5d38377326a5dbd0b3b97c55c6f2abd5d0.jpg)  
Figure 1: Comparison between different methods for Word Ordering Difference in cross-lingual dependency parsing. (a) Removing word order information. (b) Permuting the words in a source sentence to resemble the word order of a target language. (c) Our method adapts the word order in the feature space. Red arrows indicate reordering steps.

Therefore, a lot of works pay attention to the word order difference problem in cross-lingual dependency parsing. The methods mainly fall into two categories: First, as shown in Figure 1 (a), order-agnostic based methods, which utilize order-free encoders, e.g. Self-Attention (Ahmad et al. 2019), Frozen Position Embedding (Liu et al. 2021), Bagof-Words (Ji et al. 2021), to maintain robust performance to the change of word order. Second, as shown in Figure 1 (b), word reordering-based methods (Rasooli and Collins 2019; Liu et al. 2020; Arviv et al. 2023), which generally first generate a large set of new sentences syntactically similar to the target language via rearranging words in the source sentence, and then select generated high-quality sentences to train a target-language dependency parser. Since these word reordering methods essentially rearrange words in the source language, we call this category of methods as explicit word

reordering (EWR).

However, there are limitations to the previous methods. The order-agnostic methods could weaken its representation capability due to the lack of word order information, resulting in a drop in dependency parsing. The EWR methods can be computationally expensive, as the permutation space grows exponentially with the sentence length. Moreover, the explicit word rearrangement may introduce linguistic adversity, i.e. unnatural word order in the source language, and therefore can be seen as a form of noising that harms the model learning (Wei et al. 2021; Arviv et al. 2023).

To address the limitations, we explore a new attempt, implicit word reordering and propose an Implicit Word Reordering method with Knowledge Distillation (IWR-KD) for cross-lingual dependency parsing. Motivated by that deep learning are surprisingly good at features linearization (Bengio et al. 2013; Upchurch et al. 2017; Wang et al. 2019), IWR-KD learns to implicitly adapt the word order relationship in the word representation space rather than really permuting the words given a sentence. Figure 1 (c) shows the differences between the proposed IWR-KD method and the typical order-agnostic and EWR-based methods.

Specifically, we train a word order model using the targetlanguage part-of-speech (POS) data1, which is much easier to annotate than the parsing tree and often reflects the syntactic structure of a language (Liu et al. 2020). This trained target-language word order prediction model is then used as a teacher model to decide the new order between two POS tags. Finally, we train a student model to mimic the order prediction of the teacher and the dependency parsing using the labelled source language training data. Once the student model is ready, it can not only on-the-fly generate representations that correspond to the target-language word order relationship, but also preserve the essential linguistic structure of the original input, thus avoiding the mentioned limitations of previous works.

In summary, the contributions of this paper are as follows:

• We propose an implicit word reordering method for cross-lingual dependency parsing, addressing the limitations of previous approaches related to word order representations and high computational costs. • We introduce a word reordering teacher model to effectively incorporate the target word order knowledge into the dependency parsing student model. • We conduct extensive experiments demonstrating that IWR-KD outperforms several competing baselines on Universal Dependency Treebanks across 31 languages.

# Preliminaries

In this section, we first briefly review the cross-lingual dependency parsing task, backbone and prior word reordering work. These are the foundations applied to our approach.

root obl nsubj obj case ? 1 Mary prepared dinner for us   
PROPN VERB NOUN ADP PRON (a) Original English sentence. nsubj root case obl obj ? Mary for us prepared dinner   
PROPN ADP PRON VERB NOUN   
(b) Explicit Estonian-specific reordered sentence

Task Description Dependency parsing is the task of creating the dependency tree for an input sentence, which is a directed graph and defines the grammatical relation between dependent words (e.g. Mary) and their heads (e.g. prepared), as shown in Figure 2 (a). The goal of cross-lingual dependency parsing is to train a parser on the source language and perform well on an unseen target language. Throughout this work, we systematically study the transferability of the proposed implicit word reordering between source and target languages with different word order distances.

Backbone: Biaffine Dependency Parser Following (Ahmad et al. 2019; Wu and Dredze 2019; Arviv et al. 2023), we adopt the graph-based bi-affine dependency parsing model (Dozat and Manning 2018) as the backbone of our parsers, which is composed of four layers, i.e., embedding layer, BiLSTM layer, MLP layer and scorer layer, as shown in the right side of Figure 3. Following (Ahmad et al. 2019; Liu et al. 2021), all the parsers take words as well as their gold part-of-speech (POS) tags as input. Formally, given the input sequence $s$ with $L$ words $\{ w _ { 1 } , w _ { 2 } , . . . , w _ { L } \}$ and its POS tags $\{ p _ { 1 } , p _ { 2 } , . . . , p _ { L } \}$ , the embedding layer creates a sequence of input embeddings $e _ { 1 : L }$ in which each $e _ { i }$ is a concatenation of its word embedding $( e _ { w _ { i } } )$ and POS embedding $( e _ { p _ { i } } )$ . The POS embedding is trained from scratch, while the word embedding is initialized with the pre-trained multilingual language model, such as mBERT (Devlin et al. 2019).

$$
e _ { i } = e _ { w _ { i } } \oplus e _ { p _ { i } }
$$

To further introduce the contextual information, we then encode each input embedding by a multilayer bidirectional LSTM:

$$
r _ { i } = \mathrm { B i L S T M } ( e _ { 1 : L } , i )
$$

Two dimension-reducing MLPs are then used to specialise each recurrent representation $r _ { i }$ into head-word and dependent-word representations for both the edge and the label prediction.

$$
\begin{array} { r l } & { { h } _ { i } ^ { \mathrm { { e d g e - h e a d } } } = \mathrm { { M L P } } ^ { \mathrm { { e d g e - h e a d } } } ( r _ { i } ) } \\ & { ~ h _ { i } ^ { \mathrm { { e d g e - d e p } } } = { \mathrm { M L P } } ^ { \mathrm { { e d g e - d e p } } } ( r _ { i } ) } \\ & { { h } _ { i } ^ { \mathrm { { l a b e l - h e a d } } } = \mathrm { { M L P } } ^ { \mathrm { { l a b e l - h e a d } } } ( r _ { i } ) } \end{array}
$$

WOKD Loss Edge Loss Label Loss mimic Orders Orders Orders R Orders Edges Labels   
Scorer Score Score 0 公 Scorer PROPN Mary   
MLP ！ 8 NVOEURNB predpianrnedr 8 88 88 MLP 8 ： ADP for 88 ： 8 PRON us   
Transformer 8m m BiLSTM   
Embeddings ： Teacher Student B8 日8 Embeddings 4   
Input (Bag…oVfEPROBS ta…gs) PROPN VERB … PRON Mary prepared … us Mary PR(OSePqNuencial…words adnindnPerOSPRtaOgsN) Input

$$
h _ { i } ^ { \mathrm { l a b e l - d e p } } = \mathrm { M L P } ^ { \mathrm { l a b e l - d e p } } ( r _ { i } )
$$

Following (Dozat and Manning 2018), we then use biaffine classifiers to compute the edge attention scores and label attention scores. These scores can be decoded into a graph by keeping only edges that received a positive score.

$$
s _ { i \to j } ^ { \mathrm { e d g e } } = h _ { j } ^ { \mathrm { e d g e - d e p } } U ^ { \mathrm { e d g e } } h _ { i } ^ { \mathrm { e d g e - h e a d ( T ) } } + b ^ { \mathrm { e d g e } }
$$

$$
s _ { i  j } ^ { \mathrm { l a b e l } } = h _ { j } ^ { \mathrm { l a b e l - d e p } } U ^ { \mathrm { l a b e l } } h _ { i } ^ { \mathrm { l a b e l - h e a d ( T ) } } + b ^ { \mathrm { l a b e l } }
$$

$$
\hat { y } _ { i  j } ^ { \mathrm { e d g e } } = \{ s _ { i  j } \geq 0 \}
$$

$$
\hat { y } _ { i  j } ^ { \mathrm { l a b e l } } = \arg \operatorname* { m a x } s _ { i  j } ^ { \mathrm { l a b e l } }
$$

where $U$ and $b$ are linear transformation and bias term, respectively. For each position pair $i , j$ , a binary cross-entropy loss is used for the existence of edge $i  j$ , and a crossentropy loss is used for the labels of gold edges.

$$
\mathcal { L } ^ { \mathrm { e d g e } } = \mathrm { B C E L o s s } ( y _ { i  j } ^ { \mathrm { e d g e } } , \hat { y } _ { i  j } ^ { \mathrm { e d g e } } )
$$

$$
\mathcal { L } ^ { \mathrm { l a b e l } } = \mathrm { C E L o s s } ( y _ { i  j } ^ { \mathrm { l a b e l } } , \hat { y } _ { i  j } ^ { \mathrm { l a b e l } } )
$$

The training object for parsing is the summing of above two losses.

$$
\mathcal { L } = \mathcal { L } ^ { \mathrm { e d g e } } + \lambda _ { 1 } \mathcal { L } ^ { \mathrm { l a b e l } }
$$

Prior Explicit Word Reordering Given a sentence $s =$ $\{ w _ { 1 } , w _ { 2 } , . . . , w _ { L } \}$ in the source language, prior explicit word reordering (EWR) works (Liu et al. 2021; Arviv et al. 2023; Rasooli and Collins 2019) aim to permute the words in it to syntactically more similar to the order of the target language. Then the reordered sentences, denoted as $s ^ { \prime } =$ $\{ w _ { 1 } ^ { \prime } , w _ { 2 } ^ { \prime } , . . . , w _ { L } ^ { \prime } \}$ , as shown in Figure 2 (b), used to train the cross-lingual dependency parser.

# Method: Implicit Word Reordering

To mitigate the shortcomings of the existing EWR approach, we propose an Implicit Word Reordering algorithm via Knowledge Distillation (IWR-KD). Unlike conventional EWR where words really are rearranged, our IWR-KD adapts word order relations in the feature space and integrates this procedure into the training of cross-lingual dependency parser.

The overall structure of the proposed IWR-KD is shown in Figure 3, which consists of two important components: a word reordering teacher aims to decide the new word order given the source input, and a dependency parsing student not only learns dependency parsing from the labelled source input but also mimics target word order predictions of the teacher.

# Word Reordering Teacher Training

The goal of the word reordering teacher model is to decide the word order of each dependency according to the POS tags of the dependent words and their heads. Given the target language sentence $\{ w _ { 1 } , w _ { 2 } , . . . , w _ { L } \}$ and their POS tags $\overline { { \{ p _ { 1 } , p _ { 2 } , . . . , p _ { L } \} } }$ , the following steps are applied to train the word reordering teacher:

1. Head-Words Finding. For each word $w _ { i }$ in the sentence, we predict its head word $w _ { j }$ using a pre-trained parser in the source language.

2. Training Instances Extraction. We create a training instance for each pairwise POS tags $< \ p _ { i } , p _ { j } >$ of dependent-head words $< w _ { i } , w _ { j } >$ . The word order label $y _ { i  j } ^ { o r d e r } \in \{ 0 , 1 \}$ is made based on their positions in the ori→ginal sentence, where $y _ { i  j } ^ { o r d e r } = 0$ indicates the POS tcagt $p _ { i }$ iosnotnhtehle frtigohftitosf a→idn $p _ { j }$ ,eworhiglie $y _ { i  j } ^ { o r d e r } = 1$ .indi$p _ { i }$ $p _ { j }$

3. Teacher Training. As shown in the left-side in Figure 3, the teacher network consists of four layers. Given a sequence of POS tag $\{ p _ { 1 } , p _ { 2 } , . . . , p _ { L } \}$ , we first obtain its POS embeddings $\{ e _ { p _ { 1 } } , e _ { p _ { 2 } } , . . . , e _ { p _ { L } } \}$ . We only use the universal POS tags to discard the word form discrepancy. We then introduce contextual information by an order-free Transformer layer, which could improve cross-lingual generalization compared with the BiLSTM layer.

$$
z _ { p _ { i } } = \mathrm { T r a n s f o r m e r } ( e _ { p _ { i } } )
$$

We obtain the head and dependent representation for each $z _ { p _ { i } }$ through two dimension-reducing MLPs.

$$
h _ { i } ^ { \mathrm { o r d e r - h e a d } } = \mathrm { M L P } ^ { \mathrm { o r d e r - h e a d } } ( z _ { p _ { i } } )
$$

$$
h _ { i } ^ { \mathrm { o r d e r - d e p } } = \mathrm { M L P } ^ { \mathrm { o r d e r - d e p } } ( z _ { p _ { i } } )
$$

We compute word order for the given pairwise POS tags having head-dependent edges. A binary cross-entropy loss is used for the word order learning.

$$
\begin{array} { r } { \hat { s } _ { i  j } ^ { \mathrm { o r d e r - t e a } } = h _ { j } ^ { \mathrm { o r d e r - d e p } } U ^ { \mathrm { o r d e r } } h _ { i } ^ { \mathrm { o r d e r - h e a d ( T ) } } + b _ { \quad ( 1 7 } ^ { \mathrm { o r d e r } } } \\ { \hat { y } _ { i  j } ^ { \mathrm { o r d e r - t e a } } = \arg \operatorname* { m a x } s _ { i  j } ^ { \mathrm { o r d e r - t e a } } \quad \quad ( 1 8 } \end{array}
$$

$$
\mathcal { L } ^ { \mathrm { o r d e r } } = \mathrm { B C E L o s s } ( y _ { i  j } ^ { \mathrm { o r d e r } } , \hat { y } _ { i  j } ^ { \mathrm { o r d e r - t e a } } )
$$

After the teacher has well-trained in the target language, we take the teacher as guidance to train the dependency parsing student.

# Dependency Parsing Student Learning

The target of the student model is to parse a sentence based on the transformed features corresponding to the reordered source sentence. As illustrated in the right side of Figure 3, we implement the student model by adding a word order learning network based on the backbone. The student model is trained to mimic the prediction probability distribution of word order generated by the teacher model on the POS tags in the target language. This process aims to transfer word order knowledge from the teacher model to the student model while allowing the student model to leverage language-specific word ordering knowledge available in the target-language POS tags.

Here, the student model consists of four layers. Specifically, given a source input sentence $s$ with $L$ words $\{ w _ { 1 } , w _ { 2 } , . . . , w _ { L } \}$ and its POS (part-of-speech) tags $\{ p _ { 1 } , p _ { 2 } , . . . , p _ { L } \}$ . We first obtain its representation sequence $\{ r _ { 1 } , r _ { 2 } , . . . , r _ { L } \}$ through the embedding layer and BiLSTM layer, as shown in Equation 1-2.

We then use Equation 3-6 and 15-16 to specialize each recurrent representation $\boldsymbol { r } _ { i }$ into head-word $h _ { 1 : L } ^ { \mathit { \hat { h e a d } } }$ and rdedpuecntidoen -MwLorPds $h _ { 1 : L } ^ { d e p }$ dregper,elsaebnetlatainodnswuosridngotrdweor dpirmedeincstion-, respectively. In addition to calculating the edge prediction and label prediction according to the Equation 7-10, we also calculate the word order prediction as follows:

$$
s _ { i  j } ^ { \mathrm { o r d e r - s t u } } = h _ { j } ^ { \mathrm { o r d e r - d e p } } U ^ { \mathrm { o r d e r } } h _ { i } ^ { \mathrm { o r d e r - h e a d ( T ) } } + b ^ { \mathrm { o r d e r } }
$$

$$
\hat { y } _ { i \to j } ^ { \mathrm { o r d e r - s t u } } = \arg \operatorname* { m a x } s _ { i \to j } ^ { \mathrm { o r d e r - s t u } }
$$

The student model learns dependency parsing by the following two losses:

$$
\mathcal { L } ^ { \mathrm { e d g e } } = \mathrm { B C E L o s s } ( y _ { i  j } ^ { \mathrm { e d g e } } , \hat { y } _ { i  j } ^ { \mathrm { e d g e - s t u } } )
$$

$$
\mathcal { L } ^ { \mathrm { l a b e l } } = \mathrm { C E L o s s } ( y _ { i  j } ^ { \mathrm { l a b e l } } , \hat { y } _ { i  j } ^ { \mathrm { l a b e l - s t u } } )
$$

The word order knowledge distillation learning loss is formulated as the mean squared error loss:

$$
\mathcal { L } ^ { \mathrm { o r d e r } } = \mathrm { M S E L o s s } ( \hat { y } _ { i  j } ^ { \mathrm { o r d e r - t e a } } , \hat { y } _ { i  j } ^ { \mathrm { o r d e r - s t u } } )
$$

And the whole student training loss is the summation of three losses:

$$
{ \mathcal { L } } = { \mathcal { L } } ^ { \mathrm { e d g e } } + \lambda _ { 1 } { \mathcal { L } } ^ { \mathrm { l a b e l } } + \lambda _ { 2 } { \mathcal { L } } ^ { \mathrm { o r d e r } }
$$

# Experiments

In this section, we conduct extensive experiments on 31 languages across a broad spectrum of language families to validate the effectiveness and reasonableness of our proposed IWR-KD method for cross-lingual dependency parsing.

# Setup

Datasets Following the setup of (Ahmad et al. 2019), we conduct experiments on Universal Dependencies (UD) Treebanks (v2.14) (Zeman et al. 2024), in which 31 different languages are selected for evaluation. In our main experiments, we take English as the source language and 30 other languages as the target ones. We only use the source language for both training and hyper-parameter tuning.

Performance Metric The evaluation metrics are unlabeled attachment score (UAS) and labeled attachment score (LAS). Each experiment is conducted in three runs with different random seeds and the average scores are reported.

Implement Details We employ mBERT (Devlin et al. 2019) to derive the cross-lingual word embeddings. Since the mBERT embeddings are subword-level, we follow previous work (Wu and Dredze 2019; Liu et al. 2021) in taking the first subword as word-level embeddings. The maximum subword sequence length is 512. We train the POS embeddings from scratch and set the dimension size of POS embeddings as 50. The batch size is set to 32. We use Adam (Kingma and Ba 2015) to train models with $\beta _ { 1 }$ of 0.9, $\beta _ { 2 }$ of 0.9 and $L 2$ weight decay of 1e-5. The model is trained for 50 epochs with the learning rate of 1e-5 for mBERT and 3e-5 for other network layers. We choose the best hyperparameters according to the development set in the source language. We empirically set $\lambda _ { 1 } = 1 , \lambda _ { 2 } = 0 . 0 0 1$ . We implement our method using PyTorch 1.8.0 based on Hugging Face transformer library

To quantify the word order distance between two languages, we following (Ahmad et al. 2019) select the 52 most frequently occurring dependency triples (Dep $\curvearrowleft$ Head, Label) across 31 languages, then concatenate relative frequency of the left-direction (dependent word before its head) for all triples in each language as the word order feature. We use Manhattan distance as our word order distance. Due to differences in the versions of the datasets used, our word order distance does not strictly align with previous word order distance (Ahmad et al. 2019) 3.

Baseline Parsers We compare our IWR-KD with several competing baselines involving word order learning as follows. SelfAtt-Direct (Ahmad et al. 2019) adopts the selfattention based order-free encoder for cross-lingual parsing. mBERT-Direct (Wu and Dredze 2019) is fine-tuned by adding the graph-based biaffine dependency parser on top of it. Frozen $P E$ (Liu et al. 2021) freezes the positional embedding in mBERT during the fine-tuning. Subtree-EWR (Arviv et al. 2023) explicitly reorders words through subtreeaware constraints for cross-lingual parsing. WOL is a variant of ours, which learns the word order knowledge in the source language with an auxiliary loss. For fair comparison, all methods are implemented with the same datasets and training configurations. We report the results of SelfAttDirect in the original paper and re-implement other methods. Note that we do not compare ours with methods that use extra resources, e.g., SubDP (Shi, Gimpel, and Livescu 2022) utilizes translation and word alignment to augment data, SFDP (Sun, Li, and Zhao 2023) employs large amounts of unlabelled data to improve the performance.

Table 1: Cross-lingual dependency parsing results by language $( \mathrm { U A S \% / L A S \% }$ ). We order the languages by order typology distances (Ahmad et al. 2019) to English. We use ” ” to indicate results below mBERT and use underlined text to highlight the best performance.   

<html><body><table><tr><td>Lang.</td><td>Dist.</td><td>SelfAttn</td><td>mBERT</td><td>Frozen PE</td><td>Subtree-EWR</td><td>WOL</td><td>IWR-KD (ours)</td></tr><tr><td>en</td><td>0.00</td><td>90.4/88.4</td><td>92.4/90.3</td><td>92.2/90.3↓</td><td>92.4/90.3</td><td>93.0/91.5</td><td>92.5/90.5</td></tr><tr><td>no</td><td>0.06</td><td>80.8/72.8</td><td>86.7/77.4</td><td>87.0/78.0</td><td>84.5/76.6↓</td><td>88.3/78.8</td><td>87.9/78.8</td></tr><tr><td>SV</td><td>0.07</td><td>81.0/73.2</td><td>84.3/76.7</td><td>84.2/76.4↓</td><td>81.8/74.8 ↓</td><td>84.9/77.3</td><td>85.3/78.1</td></tr><tr><td>fr</td><td>0.09</td><td>77.9/72.8</td><td>83.4/71.7</td><td>83.4/72.0</td><td>84.7/73.5</td><td>84.4/72.5</td><td>85.2/73.4</td></tr><tr><td>pt</td><td>0.09</td><td>76.6/67.8</td><td>81.2/73.0</td><td>80.4/72.4↓</td><td>81.4/73.0</td><td>82.6/74.6</td><td>83.1/75.4</td></tr><tr><td>da</td><td>0.10</td><td>76.6/67.9</td><td>81.7/72.6</td><td>81.3/72.8 ↓</td><td>80.0/71.6↓</td><td>82.7/73.2</td><td>83.0/73.8</td></tr><tr><td>es</td><td>0.12</td><td>74.5/66.4</td><td>80.5/71.1</td><td>79.1/70.1↓</td><td>79.8/70.3↓</td><td>81.6/72.1</td><td>82.3/72.7</td></tr><tr><td>it</td><td>0.12</td><td>80.8/75.8</td><td>84.3/77.7</td><td>83.6/77.2↓</td><td>84.9/78.9</td><td>85.6/79.5</td><td>85.9/79.5</td></tr><tr><td>ca</td><td>0.13</td><td>73.8/65.1</td><td>80.5/70.4</td><td>79.1/69.4↓</td><td>81.6/72.1</td><td>81.5/71.5</td><td>82.3/71.9</td></tr><tr><td>hr</td><td>0.13</td><td>61.9/52.9</td><td>78.0/64.4</td><td>77.7/64.2↓</td><td>77.7/65.3</td><td>77.0/63.1 ↓</td><td>78.2/65.0</td></tr><tr><td>pl</td><td>0.13</td><td>74.6/62.2</td><td>88.4/75.7</td><td>87.2/75.4↓</td><td>88.3/76.8</td><td>88.8/75.9</td><td>88.8/76.6</td></tr><tr><td>sl</td><td>0.13</td><td>68.2/56.5</td><td>79.2/66.2</td><td>77.5/64.9↓</td><td>77.1/64.4↓</td><td>79.2/65.0↓</td><td>79.7/67.2</td></tr><tr><td>uk</td><td>0.13</td><td>60.1/52.3</td><td>73.9/63.0</td><td>75.8/64.5</td><td>73.8/62.3 ↓</td><td>75.8/63.8</td><td>77.1/65.4</td></tr><tr><td>bg</td><td>0.14</td><td>79.4/68.2</td><td>85.3/73.9</td><td>85.6/74.4</td><td>76.8/66.7↓</td><td>87.0/74.8</td><td>86.3/74.4</td></tr><tr><td>CS</td><td>0.14</td><td>63.1/53.8</td><td>78.3/64.2</td><td>77.7/64.0↓</td><td>78.4/65.0</td><td>77.5/63.0 ↓</td><td>78.5/64.5</td></tr><tr><td>de</td><td>0.14</td><td>71.3/61.6</td><td>77.9/69.3</td><td>78.7/70.3</td><td>80.8/73.0</td><td>78.7/68.6</td><td>79.9/70.4</td></tr><tr><td>he</td><td>0.14</td><td>55.3/48.0</td><td>68.0/53.8</td><td>68.3/54.2</td><td>69.7/55.1</td><td>68.3/54.5</td><td>70.6/55.0</td></tr><tr><td>nl</td><td>0.14</td><td>68.6/60.3</td><td>79.7/71.3</td><td>79.4/71.2 ↓</td><td>79.8/71.9</td><td>79.3/70.6 ↓</td><td>80.7/72.2</td></tr><tr><td>ru</td><td>0.14</td><td>60.6/51.6</td><td>77.3/65.8</td><td>76.0/64.9↓</td><td>75.6/65.6↓</td><td>77.3/65.9</td><td>77.8/66.1</td></tr><tr><td>ro</td><td>0.15</td><td>65.1/54.1</td><td>78.2/64.6</td><td>78.5/66.0</td><td>79.1/66.0</td><td>79.2/65.2</td><td>76.1/63.2 ↓</td></tr><tr><td>id</td><td>0.17</td><td>49.2/43.5</td><td>60.4/48.9</td><td>62.1/50.3</td><td>70.5/58.4</td><td>60.3/49.1</td><td>62.2/50.7</td></tr><tr><td>sk</td><td>0.17</td><td>66.7/58.2</td><td>82.9/69.4</td><td>83.4/70.9</td><td>82.4/69.8↓</td><td>81.8/67.8 ↓</td><td>84.5/70.6</td></tr><tr><td>lv</td><td>0.18</td><td>70.8/49.3</td><td>78.2/56.9</td><td>78.0/56.9↓</td><td>72.3/54.1 ↓</td><td>78.8/56.9</td><td>79.2/57.8</td></tr><tr><td>et</td><td>0.20</td><td>65.7/44.9</td><td>73.7/52.2</td><td>73.8/52.4</td><td>71.0/51.7 ↓</td><td>74.3/52.2</td><td>74.7/53.0</td></tr><tr><td>fi</td><td>0.20</td><td>66.3/48.7</td><td>74.5/53.7</td><td>75.4/54.9</td><td>71.1/53.5 ↓</td><td>76.1/55.5</td><td>75.9/55.4</td></tr><tr><td>zh</td><td>0.23</td><td>42.5/25.1</td><td>55.7/37.5</td><td>56.0/37.6</td><td>60.0/40.7</td><td>57.2/37.8</td><td>55.9/37.8</td></tr><tr><td>ar</td><td>0.26</td><td>38.1/28.0</td><td>44.8/33.1</td><td>47.1/35.5</td><td>54.1/41.1</td><td>46.5/34.9</td><td>48.5/36.2</td></tr><tr><td>la</td><td>0.28</td><td>48.0/35.2</td><td>57.8/41.4</td><td>57.0/40.6↓</td><td>57.9/42.5</td><td>57.0/40.9↓</td><td>56.6/40.8 ↓</td></tr><tr><td>ko</td><td>0.33</td><td>34.5/16.4</td><td>44.8/25.1</td><td>45.0/25.1</td><td>47.9/29.7</td><td>43.8/23.1↓</td><td>47.7/26.6</td></tr><tr><td>hi</td><td>0.40</td><td>35.5/26.5</td><td>41.4/26.8</td><td>42.5/28.0</td><td>41.0/27.0↓</td><td>40.5/27.1 ↓</td><td>43.3/28.2</td></tr><tr><td>ja</td><td>0.49</td><td>28.2/20.9</td><td>25.2/16.6</td><td>27.1/17.8</td><td>26.4/18.4</td><td>25.5/16.6</td><td>28.8/20.6</td></tr><tr><td>AVG</td><td></td><td>64.1/53.8</td><td>72.9/60.5</td><td>72.9/60.7</td><td>73.0/61.3</td><td>73.4/60.8</td><td>74.1/61.7</td></tr><tr><td>±STD</td><td>0.17</td><td>/</td><td>± 0.4/0.1</td><td>± 0.3/0.2</td><td>±0.2/0.2</td><td>± 0.3/0.4</td><td>± 0.3/0.2</td></tr></table></body></html>

Table 2: Ablation study (averaged results over all languages).   

<html><body><table><tr><td>Model</td><td>UAS%/LAS%</td></tr><tr><td>IWR-KD (Ours)</td><td>74.1/61.7</td></tr><tr><td>w/Pseudo-Labelling</td><td>73.5-0.6/61.3-0.4</td></tr><tr><td>w/ Silver POS Tags</td><td>73.4-0.7/60.9-0.8</td></tr><tr><td>w/o FT</td><td>72.6-1.5/60.3-1.4</td></tr></table></body></html>

# Main Results

Table 2 presents the results on the test sets. The languages are ordered by their order typology distance to English. From the experimental results, we can make the following observations. (1) Our proposed IWR-KD method achieves state-of-the-art performance on most languages and average performance across all languages, demonstrating the effectiveness and generalization of IWR-KD. (2) The ordersensitive direct transfer method (mBERT) significantly surpasses the traditional order-agnostic self-attention method (SelfAttn) by effectively capturing richer contextual information, including word order. (3) The performance of the mBERT-based word order-agnostic model (Frozen PE) declines relative to direct transfer (mBERT) in both the source language and similar languages, such as English (en) and Norwegian (no), due to underfitting caused by the frozen position representation. (4) The explicit word reordering method (Subtree-EWR) substantially declines in some languages, such as Slovenian (sl) and Finnish (fi), due to unnatural sentence rearrangements that introduce noise detrimental to model learning. Note that the EWR method indirectly accesses the target language dependency annotation information (Arviv et al. 2023), so it can achieve high performance in some languages. While IWR-KD relies solely on low-cost target language POS tags, it outperforms SubtreeWR in average performance. (5) WOL performs better in languages closer to English, while mBERT outperforms it in languages that are more ”distant” from English. This indicates capturing word order is critical in cross-lingual dependency parsing.

Table 3: Case study on Implicit Word Reordering. Word order frequency indicates the relative frequency of the left directio (dependent before its head). The GREEN (RED) highlight indicates a head (dependent) word.   

<html><body><table><tr><td>(Dep Head,Label)</td><td>Word Order Frequency</td><td colspan="2">Source/Target Examples</td><td></td></tr><tr><td>(PRON VERB,obj)</td><td>EN: 0.1351 EN: 0WR-KD: 0.9348</td><td>EN:Recently I'm having trouble trainingVERB</td><td>head</td><td>dep himPRON</td></tr><tr><td>(NOUN←VERB,obl)</td><td>EN: 0.0996 EN: 0WR-KD: 0.5055</td><td>head EN: ... be causingvERB</td><td>dep us trouble for yearsNoUN</td><td>to come.</td></tr><tr><td>(NOUN ← NOUN, nmod)</td><td>EN: 0.0056 EN+IWR-KD: 0.4867 ET: 0.8296</td><td>EN: Hopetheisusefulgistfortheilhe. ET: .. tanapievaNoUN</td><td></td><td></td></tr></table></body></html>

# Ablation Study

To verify the effect of each part of our method, we do experiments with the following variants of our IWR-KD. (1) w/ pseudo-labelling, which converts the soft word order probability distribution into hard labels to guide the student model training. Hard labels result in performance degradation, highlighting that our word order distillation loss conveys richer word order knowledge than hard labels. (2) w/ Silver UPOS, which employs the Stanza 4 tool to annotate POS tags to simulate a more realistic application scenario. In this case, our method can improve the performance of the direct transfer method (mBERT in Table 2). (3) $w / o F T$ , which eliminates the fine-tuning of mPLMs to make the model more lightweight. However, performance drops significantly without fine-tuning, indicating that fine-tuning enables the encoder to effectively adapt to the specific task.

# Case Study

In this section, we conduct a series of case studies to bring up insights into why the proposed IWR-KD works.

The IWR-KD method can help the parser correctly identify dependency labels by exploiting the learned order relationship between dependent words and their heads. Specifically, if the model can align the word order relationships of the source language to the target language, misidentified dependency labels in the target language may be corrected. As shown in Table ??, in the first example (PRON $\curvearrowleft$ VERB, obj), the relative frequency of PRON before VERB differs greatly between the source and target languages, which leads to migration failure. Our model’s prediction of the relative frequency of PRON before VERB in the source language is close to that of the target language. At the same time, our model learned that their label is obj on the source language training set, and then the model can correctly predict PRON and PRON in the target language. The dependency tag of VERB is obj. Two additional examples present the same results using different languages.

# Performance versus Word Order Distance

As shown in the upper subfigure of Figure 4, transfer ability decreases as the word order distance increases from left to right, indicating that word order is a crucial factor for transfer performance (Perf.). The Pearson correlation coefficient between word order distance and transfer performance is - 0.8803 (p-value $= 1 \mathrm { e } { - } 1 0 \$ ), demonstrating a strong negative correlation. As shown in the lower subfigure of Figure 4, the reduction (Redn.) in word order distance positively influences performance improvement (Impr.). The Pearson correlation coefficient is 0.4308 (p-value $= 0 . 0 1 7 5$ ), indicating a strong positive correlation.

# Effect of Different Teacher Models

This section investigates the effect of the different word reordering teacher models. We compare three learning methods: (1) Rand: Randomly learning the word order relationship between pairs of words. (2) Heur: Heuristically learning the ordering relationships of pairs belonging to 52 selected dependency triples. (3) Our: Based on a pre-trained parser in the source language, we predict the head of each word and learn their ordering relationships

![](images/239b2d2edc5047966502158de0b6de8101f1dbf258c89fdd7a4a483b725ca94a.jpg)  
Figure 4: Word order distance and performance. Languages ( $\mathbf { \boldsymbol { x } }$ -axis) are sorted by their order typology distances (Ahmad et al. 2019) to English from left to right.

![](images/7f82fa24d94ed8ee68622dfa30df8b7481a3b47111e6a6e03dd8dc913783a767.jpg)  
Figure 5: Word order distance predicted by different word reordering teachers. The green bar indicates original word order distance between English and the target language.

As shown in Figure 5, we calculate the word order distance predicted by three different teacher models on six different target languages. Three languages, i.e. no, sv, da, are closer to the source language, while the other three languages, i.e. zh, ko, ja, are far away from the source language. It can be seen that our transfer-based teacher model can effectively reduce the word order distance between the source language and target languages. We notice that our method fails to reduce the word order distance between the source language and zh (Chinese), but it improves the transfer performance as in Table 2. Through more thorough observation, we find that our method reduces the word order difference on some dependency triples. For example, for the word order frequency of (NOUN $\curvearrowleft$ NOUN, nmod), EN: 0.0056, ZH: 0.9980, EN+IWR-KD: 0.5413. We conjecture that different dependencies have different effects on transfer performance, which may be one of our future research directions.

# Related Work

Word order refers to the arrangement of words in a sentence or phrase to convey meaning in a particular language. In the cross-lingual community, word order is widely explored from two perspectives: order-agnostic methods (Ahmad et al. 2019; Liu et al. 2021; Ding, Wang, and Tao 2020; Chen, Zhang, and Fu 2023; Si et al. 2019; Hessel and Schofield 2021) and word reordering methods (Rasooli and Collins 2019; Ji et al. 2021; Liu et al. 2020; Chen et al. 2019; Goyal and Durrett 2020; Al-Negheimish, Madhyastha, and Russo 2023; Pham et al. 2021). The former argues that word order encoding is a risk for cross-lingual transfer, as the model often fits in the language-specific order. Reducing the word order information fitted into the models can improve the cross-lingual adaptation performance in position representation (Ding, Wang, and Tao 2020) and dependency parsing (Ahmad et al. 2019; Liu et al. 2021). The latter is dedicated to reordering the word from one language to another. For example, Arviv et al. (Arviv et al. 2023) achieves better cross-lingual transfer results by rearranging the words in the source language to meet the word order in the target language conditioned on the syntactic constraints.

# Conclusion

In this paper, we explore a new attempt, implicit word reordering, and propose an implicit word reordering method with knowledge distillation (IWR-KD), which uses the word reordering model as a teacher to guide the student model to simultaneously learn target-language word order knowledge and source-language dependency parsing knowledge on the labeled data of the source language. IWR-KD can effectively capture vital word order information and does not require the actual generation and selection of reordered sentences, which addresses the limitations of previous order-agnostic and explicit word reordering methods. We conduct extensive experiments on 31 different languages on Universal Dependency Treebanks, which showed that IWR-KD outperforms multiple competitive methods.