# Quantum-inspired Non-homologous Representation Constraint Mechanism for Long-tail Senses of Word Sense Disambiguation

Junwei Zhang, Xiaolin Li\*

Hangzhou Institute of Medicine, Chinese Academy of Sciences, Hangzhou, Zhejiang, China zhangjunwei $@$ him.cas.cn, xiaolinli@ieee.org

# Abstract

Word Sense Disambiguation (WSD) aims to determine the meaning of target words according to the given context. The recognition of high-frequency senses has reached expectations, and the current research focus is mainly on lowfrequency senses, namely Long-tail Senses (LTSs). One of the challenges in long-tail WSD is to obtain clear and distinguishable definition representations based on limited word sense definitions. Researchers try to mine word sense definition information from data from different sources to enhance the representations. Inspired by quantum theory, this paper provides a constraint mechanism for representations under non-homogeneous data to leverage the geometric relationship in its Hilbert space to constrain the value range of parameters, thereby alleviating the dependence on big data and improving the accuracy of representations. We theoretically analyze the feasibility of the constraint mechanism, and verify the WSD system based on this mechanism on the standard evaluation framework, constructed LTS datasets and cross-lingual datasets. Experimental results demonstrate the effectiveness of the scheme and achieve competitive performance.

# Introduction

Word Sense Disambiguation (WSD) aims to determine the meaning of target words according to the given context, which belongs to the basic research topic in the field of natural language processing (Bevilacqua et al. 2021; Navigli 2009). The accuracy of WSD is of great significance and value to downstream tasks (Kaddoura, Ahmed, and D. 2022), such as machine translation (Campolungo et al. 2022), information retrieval (Abderrahim and Abderrahim 2022), sentiment analysis (Farooq et al. 2015), etc. In addition, WSD, like text classification, is often used as a training ground for new models and methods.

WSD systems have reached expectations for the recognition of high-frequency senses (that is, commonly used word senses), and the current research focus is mainly on the recognition of low-frequency senses (that is, rarely used word senses), also known as Long-Tail Senses (LTSs) (Su et al. 2022; Zhang et al. 2022b; Chen, Zhang, and He 2022; Du et al. 2021; Blevins and Zettlemoyer 2020). The root of the difficulty in long-tail WSD lies in:

• LTSs are rarely used, which leads to insufficient training samples;   
• LTSs lack clear and distinguishable definitions, which leads to inaccurate representation of word sense definitions.

Among them, training samples refer to the application scenarios of word senses, such as example sentences given in dictionaries, which provide the context information of target words; word sense definitions refer to the descriptions of word senses, such as glosses given in dictionaries, which provide word sense label information. In the process of WSD, the two jointly affect the accuracy of word sense recognition. There are many methods to deal with the lack of training samples of LTSs, and the solution is similar to the classic few-shot task, so this paper focuses on improving the inaccurate representation of word sense definitions.

Kumar et al. (Kumar et al. 2019) constructed word sense definition representations in a continuous space to constrain unclear representations through clear ones. Blevins et al. (Blevins and Zettlemoyer 2020) leveraged the semantic knowledge in the training samples to enhance the word sense definition representations through the joint training of dual encoders (that is, the target word encoder and the definition encoder). Zhang et al. (Zhang et al. 2022a) realized the enhancement of word sense definition representations by integrating example sentences and definitions of related word senses in WordNet. It is not difficult to see from the relevant research in recent years that the effective means of strengthening word sense definition representations is to integrate external knowledge and impose constraint mechanisms.

Inspired by quantum theory (Nielsen and Chuang 2002), this paper proposes a quantum-like model that can simultaneously expand external knowledge and enforce constraints. More importantly, the constraint mechanism not only provide constraints in continuous space, but also implement geometric constraints in Hilbert space for representations of non-homogeneous data. Specifically, firstly, we map homologous or non-homologous data into corresponding embeddings through classical language models, such as BERT (Devlin et al. 2019); secondly, impose normalization constraints on the embeddings to make them quantum states; thirdly, combine the states into a superposition state; finally, the quantum measurement operation is implemented to obtain the result.

Note that homologous data refers to data generated by the same generator (that is, has the same structure and obeys similar statistical laws); otherwise, it is non-homologous data. The quantum-like model exhibits spatial geometry constraints only when the generated representations come from non-homogeneous data. If the model is used on homogeneous data, it can only provide representation combinations and continuous space constraints.

Our contributions can be summarized as follows:

• A quantum-like model that can simultaneously provide representation combinations, continuous space constraints, and spatial geometry constraints is proposed, inspired by quantum theory;   
• A theoretical analysis of why the geometric constraint mechanism can be helpful for few-shot learning tasks is given;   
• Finally, the WSD system based on the quantum-like model is verified on the standard evaluation framework, constructed LTS datasets and cross-lingual datasets; the experimental results prove the effectiveness of the quantum-like model and achieve state-of-the-art performance.

# Related Work

# Long-tail WSD

The recognition process of WSD is a matching process between the target word embedding and the text embedding of word sense definitions, so the solutions for long-tail WSD can be roughly divided into two categories, enhancing the representation of target words and improving the representation of word sense definitions. The research focus of this paper is on improving the representation of word sense definitions.

Huang et al. (Huang et al. 2019) first proposed to train the representation of word sense labels through the word sense definitions in the dictionary. Subsequently, Blevins et al. (Blevins and Zettlemoyer 2020) enhanced the representation of word sense definitions through the semantic information in the training samples; Yap et al. (Yap, Koh, and Chng 2020) and Zhang et al. (Zhang, He, and Guo 2021) borrowed the example sentences from the dictionary; Scarlini et al. (Scarlini, Pasini, and Navigli 2020a) used multilingual knowledge. Kumar et al. (Kumar et al. 2019) leveraged continuous space constraints to enhance unclear representations from clear representations of word sense definitions. In addition, Kumar et al. (Kumar et al. 2019) and Holla et al. (Holla et al. 2020) also transferred solutions from the fields of few-shot learning and meta learning.

It is not difficult to see from the related work in recent years that integrating external resources and imposing constraint mechanisms are the mainstream methods to deal with long-tail WSD. The work in this paper attempts to achieve the above two functions in one model, and provides spatial geometry constraints for representations under non-homogeneous data.

# Quantum-inspired Models for WSD

Quantum-inspired models refer to learning models based on quantum theory (Nielsen and Chuang 2002) or quantum cognition (Busemeyer and Bruza 2012). Since the success of the quantum language model proposed by Basile et al. (Basile and Tamburini 2017), the intersection of quantum theory and natural language processing has gradually become a research hotspot (Li et al. 2016; Xie et al. 2015).

In the field of WSD, related research is still in the early stages of exploration. The model QWSD proposed by Tamburini et al. (Tamburini 2019) is the first of its kind. Although QWSD is very simple in model design, its advantage is that it does not require a long training process. In addition, the WSD system proposed by Zhang et al. (Zhang et al. 2022b) uses the disentanglement representation inspired by quantum theory.

Our work also leverages the mathematical form of quantum theory, namely quantum probability theory, to realize that the constructed WSD model can combat the data sparsity problem faced by long-tail WSD.

# Theoretical Analysis

# Preliminaries

Before analyzing the constraint mechanism and presenting the quantum-like model, the necessary preliminaries of Quantum Probability Theory (QPT) is given. QPT is a more general probability theory, which is backward compatible with Classical Probability Theory (CPT), namely Kolmogorov probability theory. QPT, like CPT, can be used as a modeling tool for information systems. See Ref. (Nielsen and Chuang 2002) for more details.

Quantum Events: QPT assigns probabilities to events like CPT, but the difference is that events in quantum probability are described by subspaces in Hilbert space $\mathcal { H } \in \mathbb { C } ^ { n }$ , while events in classical probability are described by sets.

Quantum States: Quantum states, also called quantum systems, are described as complex vectors $| \psi \rangle \in { \mathcal { H } }$ in Hilbert space using the Dirac1 notation. Quantum superposition states refer to quantum states in which multiple states are superimposed at the same time. Its formalization is defined as

$$
| \psi \rangle = \varepsilon _ { 1 } | e _ { 1 } \rangle + \varepsilon _ { 2 } | e _ { 2 } \rangle + \ldots + \varepsilon _ { i } | e _ { i } \rangle + \ldots
$$

where $\varepsilon _ { i }$ is called the probability amplitude, $\varepsilon _ { i } = \langle e _ { i } | \psi \rangle \in$ $\mathbb { C } , \sum _ { i } | \varepsilon _ { i } | ^ { 2 } = 1$ , and $\left| e _ { i } \right.$ is the basis state of $\mathcal { H }$ . In general, superposition states can also be composed of other superposition states,

$$
| \Psi \rangle = \varepsilon _ { 1 } | \psi _ { 1 } \rangle + \varepsilon _ { 2 } | \psi _ { 2 } \rangle + \ldots + \varepsilon _ { i } | \psi _ { i } \rangle + \ldots
$$

Quantum Measurements: The mainstream quantum measurement methods include general measurement, projection measurement and POVM measurement. Among

them, general measurement is often used in the field of machine learning, so only general measurement is presented here.

General measurement is described by a set of measurement operators $\{ M _ { m } \}$ , and they satisfy completeness,

$$
\sum _ { m } M _ { m } ^ { \dagger } M _ { m } = I ,
$$

where $m$ refers to a possible measurement result in the experiment, and $I$ refers to the identity matrix. The quantum system is in $| \psi \rangle$ before being measured, and the probability of the possible result $m$ is

$$
p ( m ) { = } p ( M _ { m } ; | \psi \rangle ) { = } \langle \psi | M _ { m } ^ { \dagger } M _ { m } | \psi \rangle = \| M _ { m } | \psi \rangle \| ^ { 2 } ;
$$

after being measured, the quantum system is in

$$
| \psi ^ { \prime } \rangle = \frac { M _ { m } | \psi \rangle } { \sqrt { \langle \psi | M _ { m } ^ { \dagger } M _ { m } | \psi \rangle } } .
$$

From the completeness of the measurement operators, it can be deduced that

$$
\sum _ { m } p ( m ) = \sum _ { m } \langle \psi | M _ { m } ^ { \dagger } M _ { m } | \psi \rangle = 1 .
$$

# Theoretical Analysis for Quantum-inspired Constraint Mechanism

In the field of quantum information processing, the characteristic representations from data are described as quantum states, multi-source data are often integrated as quantum superposition states, and possible results are described by measurement operators (Nielsen and Chuang 2002; Zhang et al. 2020, 2021, 2022c,d).

Accordingly, the quantum states $\left| \psi _ { A } \right.$ and $\left| \psi _ { B } \right.$ obtained from non-homologous data, say $A$ and $B$ , can be constructed as the superposition state,

$$
| \Psi _ { A B } \rangle = \alpha | \psi _ { A } \rangle + \beta | \psi _ { B } \rangle ,
$$

where $\alpha$ , $\beta \in \mathbb { R }$ are the probability amplitudes of constructing the superposition state, and ${ \dot { \alpha } } ^ { 2 } + { \dot { \beta } } ^ { 2 } = 1$ . In specific tasks, it can be a superposition state composed of multiple quantum states, or a probability amplitude in the form of complex numbers. The measurement operators corresponding to the possible results can be described by a concrete basis state $\left| e _ { i } \right.$ , $M _ { m } = | e _ { i } \rangle \langle e _ { i } |$ , or by a quantum state $\mathinner { | { \phi } \rangle }$ obtained from the label data $\phi$ ,

$$
M _ { m } = | \phi \rangle \langle \phi | .
$$

The probability of the possible result $m$ is

$$
\begin{array} { r l r } { p ( M _ { m } ; | \Psi _ { A B } \rangle ) = \| M _ { m } | \Psi _ { A B } \rangle \| ^ { 2 } } & { { \scriptstyle ( 9 } } \\ { \quad } & { { \scriptstyle = \| M _ { m } ( \alpha | \psi _ { A } \rangle + \beta | \psi _ { B } \rangle ) \| ^ { 2 } } } \\ { \quad } & { { \scriptstyle = \| \alpha M _ { m } | \psi _ { A } \rangle + \beta M _ { m } | \psi _ { B } \rangle \| ^ { 2 } } } \\ { \quad } & { { \scriptstyle = \| \alpha M _ { m } | \psi _ { A } \rangle \| ^ { 2 } + \| \beta M _ { m } | \psi _ { B } \rangle \| ^ { 2 } + I n t , } } \end{array}
$$

where the interference term $I n t$ is

$$
\begin{array} { r } { I n t = 2 \alpha \beta \cos ( \theta ) | \langle \psi _ { A } | M _ { m } | \psi _ { B } \rangle | } \\ { = 2 \alpha \beta \cos ( \theta ) | \langle \psi _ { A } | \phi \rangle \langle \phi | \psi _ { B } \rangle | } \end{array}
$$

![](images/74adaf8992506e9ef853d18df5155965200c2f517811b791cfb7096c631045c7.jpg)  
Figure 1: Schematic illustration of the geometric relationship between quantum states in Hilbert space revealed by the interference term.

and $\theta$ is the phase angle between quantum states.

The interference term is a unique feature derived from quantum probability, which reveals the geometric relationship of quantum states in Hilbert space, as shown in Fig. 1. $\alpha$ and $\beta$ in the interference item are the parameters for constructing the superposition state, which can be regarded as weights. $| \langle \psi _ { A } | \bar { \phi } \rangle |$ and $\left| \langle \phi | \psi _ { B } \rangle | \right.$ describe the side lengths of the two sides of the triangle in the illustration, and $\cos ( \theta )$ is the degree of an angle of the triangle. They jointly describe the area of the triangle, so the interference term itself can be considered as a description of the geometric relationship of the quantum state in Hilbert space.

In the learning model constructed in the above form, the interference term is used as a constraint item of the loss function to realize the spatial geometry constraint between the quantum states (that is, the representation obtained from the label data and the representations obtained from the nonhomologous data). Compared with the traditional loss function with no constraint term, the loss function with the spatial geometry constraints can limit the value range of the features in the representation and alleviate the dependence on the large amount of data in the representation learning process. In fact, it is not difficult to understand that the constraint item describing the geometric relationship limits the positional relationship of the representations in space, which can naturally improve the difficulty of representation learning compared to unconstrained representations.

# Methodology

# Word Sense Disambiguation

The WSD task can be formalized as a mapping function from the word embedding of the target word in the disambiguated text, $V ^ { t a r g e t } \mathbf { \Psi } \in \mathbb { R } ^ { 1 \times n }$ , to the text embedding of word sense definitions in the dictionary, $V _ { k } ^ { d e f i n i t i o n s } \in$ R1×n,

$$
\mathcal { F } : V ^ { t a r g e t } \to V _ { k } ^ { d e f i n i t i o n s }
$$

where $V _ { k } ^ { d e f i n i t i o n s }$ refers to the $k$ -th word sense in the candidate list corresponding to the target word (Bevilacqua et al.

![](images/6253f630b67b6a14916a4a85eb50c65019ece27b3588d9525a5a859c26ae70d5.jpg)  
Figure 2: The architecture of QiWSD: a word sense disambiguation system with a traditional recognition method for highfrequency senses and a quantum recognition method for long-tail senses that can integrate non-homologous data using the quantum-like model.

2021; Navigli 2009; Zhang et al. 2024; Zhang, He, and Guo   
2023).

# Quantum-like Model with Quantum-inspired Constraint Mechanism

The core part of the quantum-like model with quantuminspired non-homologous representation constraint mechanism is a quantum measurement operation, that is,

$$
p ( M _ { m } ; | \Psi \rangle ) = \| M _ { m } | \Psi \rangle \| ^ { 2 } ,
$$

in which the important components are the superposition state $\left| \Psi \right.$ and the measurement operator $M _ { m }$ .

A superposition state can be composed of any number of quantum states $| \psi _ { i } \rangle$ ,

$$
| \Psi \rangle = \varepsilon _ { 1 } | \psi _ { 1 } \rangle + \varepsilon _ { 2 } | \psi _ { 2 } \rangle + \ldots + \varepsilon _ { i } | \psi _ { i } \rangle + \ldots ;
$$

the quantum states can be obtained by imposing a normalized function of the sum of squares on general representations $V _ { i }$ ,

$$
| \psi _ { i } \rangle = S S N ( V _ { i } ) = \frac { V _ { i } } { \sqrt { \| V _ { i } \| _ { 2 } } } .
$$

The representations used to construct the superposition state can be obtained from homologous or non-homologous data. However, the representations obtained based on homologous data do not have the spatial geometry constraints pointed out in this paper, because the representations obtained from homologous data will eventually be merged into one. It should be noted that the quantum-like model is also valuable for homogeneous data, which is equivalent to a model with data integration capabilities.

A measurement operator is constructed from a quantum state $\mathinner { | { \phi } \rangle }$ ,

$$
M _ { m } = | \phi \rangle \langle \phi | ;
$$

the quantum state can also be obtained by applying $S S N ( \cdot )$ to a general representation $V _ { m }$ ,

$$
\vert \phi \rangle = S S N ( V _ { m } ) .
$$

# QiWSD: Quantum-inspired WSD System

In this section, we apply the quantum-like model to build a WSD system to verify whether the spatial geometry constraint mechanism applicable to non-homologous data can improve the inaccurate representation of word sense definitions faced by long-tail WSD tasks. The quantum-inspired WSD system is called QiWSD, and its model structure is shown in Fig. 2.

BiWSD consists of two parts, the traditional recognition method for high-frequency senses and the quantum recognition method enhanced by non-homologous data for LTSs. The traditional recognition method has been verified to be effective for high-frequency WSD, and this part is added to make the overall WSD system take into account highfrequency senses. The quantum recognition method leverages the spatial geometry constraint mechanism proposed in this paper, and this part is added to make the overall WSD system take into account LTSs.

The traditional recognition method uses two pretrained language models BERT (Devlin et al. 2019) as encoders (namely target word encoder and word sense definition encoder) to obtain the word embedding of the target word in the disambiguated text $W ^ { t e x t }$ ,

$$
V ^ { t a r g e t } = B E R T _ { T a r g e t } ( W ^ { t e x t } ) ,
$$

and the text embedding of the word sense definitions provided by the glosses $W ^ { \bar { g } l o s s e s }$ in the dictionary,

$$
V _ { k } ^ { d e f i n i t i o n s } = B E R T _ { G l o s s e s } ( W _ { k } ^ { g l o s s e s } ) .
$$

According to the specification of the BERT model, the vector corresponding to the target word in the disambiguated text is used as the word embedding of the target word output by the target word encoder; the vector corresponding to the start token “[CLS]” in the gloss is used as the text embedding of the word sense definition output by the word sense definition encoder.

Finally, the inner product of the target word embedding $V ^ { t a r g e t }$ and word sense definition embeddings $V _ { k } ^ { d e }$ finitions is calculated separately to obtain the similarity score of each word sense under the traditional recognition method,

$$
S c o r e _ { k } ^ { T r a d i t i o n a l } = V ^ { t a r g e t } \odot V _ { k } ^ { d e f i n i t i o n s } .
$$

The quantum recognition method uses the target word embedding and the word sense definition embeddings respectively output by the target word encoder and word sense definition encoder. Furthermore, example sentences of word sense definitions from the dictionary are integrated for word sense definition embeddings using the quantuminspired constraint mechanism proposed in this paper.

Similarly, a pre-trained language model BERT is used as a word sense example encoder to obtain the text embedding of example sentences W examples,

$$
V _ { k } ^ { e x a m p l e s } = B E R T _ { E x a m p l e s } ( W _ { k } ^ { e x a m p l e s } ) .
$$

Since the target word exists in the example sentence, the vector corresponding to the target word in the example sentence is used as the text embedding output by the word sense example encoder.

Next, the quantum recognition method is implemented using the quantum-like model with the quantum-inspired constraint mechanism:

$V _ { k } ^ { d e f i n i t i o n s }$ and $V _ { k } ^ { e x a m p l e s }$ are constructed as quantum states by the normalization function $S S N ( \cdot )$ ,

$$
| \psi _ { k } ^ { d e f i n i t i o n s } \rangle = S S N ( V _ { k } ^ { d e f i n i t i o n s } )
$$

and

$$
| \psi _ { k } ^ { e x a m p l e s } \rangle = S S N ( V _ { k } ^ { e x a m p l e s } ) ;
$$

• they are constructed as a superposition state by Eq. (13),

$$
| \Psi _ { k } ^ { d e f + e x m } \rangle = \varepsilon _ { k } ^ { 1 } | \psi _ { k } ^ { d e f i n i t i o n s } \rangle + \varepsilon _ { k } ^ { 2 } | \psi _ { k } ^ { e x a m p l e s } \rangle
$$

where $\varepsilon _ { k } ^ { 1 } = \sin ( \varepsilon _ { k } )$ , $\varepsilon _ { k } ^ { 2 } = \cos ( \varepsilon _ { k } )$ , and $\textstyle \varepsilon _ { k } \in \mathbb { R }$ is obtained by Vkdefinitions a nd $V _ { k } ^ { e x a m p l e s }$ through a linear layer of the neural network;

• $V ^ { t a r g e t }$ is constructed as a measurement operator by the normalization function $S S N ( \cdot )$ and Eq. (15),

$$
| \phi ^ { t a r g e t } \rangle = S S N ( V ^ { t a r g e t } )
$$

and

$$
M _ { + } ^ { t a r g e t } = | \phi ^ { t a r g e t } \rangle \langle \phi ^ { t a r g e t } |
$$

where $\cdot _ { + } , \cdot _ { }$ refers to the observations that the target word belongs to the corresponding word sense definition;

• finally, the similarity score of each word sense under the quantum recognition method is calculated through the quantum-like model Eq. (12),

$$
S c o r e _ { k } ^ { Q u a n t u m } = p ( M _ { + } ^ { t a r g e t } ; | \Psi _ { k } ^ { d e f + e x m } \rangle ) .
$$

# Model Training

We train QiWSD by optimizing the similarity scores of the word senses obtained by traditional and quantum recognition methods,

$$
S c o r e _ { k } = a \cdot S c o r e _ { k } ^ { T r a d i t i o n a l } + b \cdot S c o r e _ { k } ^ { Q u a n t u m } ,
$$

through cross-entropy loss,

$$
\begin{array} { l } { { \displaystyle L o s s ( S c o r e , i n d e x ) } } \\ { ~ = - \log \left( \frac { \exp ( S c o r e ^ { [ i n d e x ] } ) } { \sum _ { i = 1 } \exp ( S c o r e ^ { [ i ] } ) } \right) } \\ { ~ = - S c o r e ^ { [ i n d e x ] } + \log \sum _ { i = 1 } \exp ( S c o r e ^ { [ i ] } ) , } \end{array}
$$

where $a , b \in \mathbb { R }$ are the weights of each recognition method, $S c o r e = [ S c o r e _ { 1 } , S c o r e _ { 2 } , . . . , S c o r e _ { k } , . . . ]$ , and index is the index of the candidate list of the word senses.

# Experiments

The following questions will be answered through experimental analysis:

• Whether the WSD system based on the quantum-like model can effectively improve the overall performance of the system is verified by standard and data-enhanced evaluation experiments;   
• Whether the spatial geometry constraint mechanism for non-homologous data can effectively enhance the recognition ability of LTSs is verified by ablation experiments;   
• Whether the system has effective generalization performance for other languages is verified by the latest crosslingual datasets.

# Datasets and Model Settings

Datasets: The standard evaluation experiments of QiWSD are constructed by using the WSD evaluation framework proposed by Navigli et al. (Navigli, CamachoCollados, and Raganato 2017); the data-enhanced evaluation experiments are constructed by adding the training set $\mathrm { \Delta W N G T ^ { 2 } }$ . The training set is $\mathrm { S e m } \dot { \mathrm { C o r } } ^ { 3 }$ , the development set is SemEval-07 (SE7; (Pradhan et al. 2007)), and the test sets include Senseval-2 (SE2; (Edmonds and Cotton 2001)), Senseval-3 (SE3; (Snyder and Palmer 2004)), SemEval-13 (SE13; (Navigli, Jurgens, and Vannella 2013)), SemEval-15 (SE15; (Moro and Navigli 2015)), and the concatenation of all test sets (ALL).

Table 1: F1-score $( \% )$ on the English all-words WSD task. The comparison systems are divided into two groups: those under the standard evaluation experiments (i.e., using only SemCor) and those under the data-enhanced evaluation experiments (i.e., using SemCor and WNGT). SOTA performance is underlined compared to $\mathrm { Q i W S D _ { b a s e } }$ and bold compared to $\mathrm { Q i W S D _ { l a r g e } }$ .   

<html><body><table><tr><td rowspan="2">WSD Systems</td><td>Deyset</td><td colspan="4">SETest SE13</td><td colspan="5">NoConcaenationdf.all est setALL</td></tr><tr><td></td><td>SE2</td><td></td><td></td><td>SE15</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Standard Evaluation Experiments:</td></tr><tr><td>EWISE (ACL; Kumar et al., (2019))</td><td>67.3</td><td>73.8</td><td>71.1</td><td>69.4</td><td>74.5</td><td>74.0</td><td>60.2</td><td>78.0</td><td>82.1</td><td>71.8</td></tr><tr><td>LMMS (ACL; Loureiro et al., (2019))</td><td>68.1</td><td>76.3</td><td>75.6</td><td>75.1</td><td>77.0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>75.4</td></tr><tr><td>SREF (EMNLP;Wang et al.,(2020))</td><td>72.1</td><td>78.6</td><td>76.6</td><td>78.0</td><td>80.5</td><td>80.6</td><td>66.5</td><td>82.6</td><td>84.4</td><td>77.8</td></tr><tr><td>ARES(EMNLP;Scarlinietal.,(2020b))</td><td>71.0</td><td>78.0</td><td>77.1</td><td>77.3</td><td>83.2</td><td>80.6</td><td>68.3</td><td>80.5</td><td>83.5</td><td>77.9</td></tr><tr><td>BEM(ACL;Blevins et al., (2020))</td><td>74.5</td><td>79.4</td><td>77.4</td><td>79.7</td><td>81.7</td><td>81.4</td><td>68.5</td><td>83.0</td><td>87.9</td><td>79.0</td></tr><tr><td>EWISER (ACL; Bevilacqua et al., (2020))</td><td>71.0</td><td>78.9</td><td>78.4</td><td>78.9</td><td>79.3</td><td>81.7</td><td>66.3</td><td>81.2</td><td>85.8</td><td>78.3</td></tr><tr><td>SyntagRank (ACL; Scozzafava et al.,(2020))</td><td>59.3</td><td>71.6</td><td>72.0</td><td>72.2</td><td>75.8</td><td></td><td>一</td><td></td><td></td><td>71.2</td></tr><tr><td>COF(EMNLP;Wang et al.,(2021))</td><td>69.2</td><td>76.0</td><td>74.2</td><td>78.2</td><td>80.9</td><td>80.6</td><td>61.4</td><td>80.5</td><td>81.8</td><td>76.3</td></tr><tr><td>ESR (EMNLP; Song et al., (2021))</td><td>75.4</td><td>80.6</td><td>78.2</td><td>79.8</td><td></td><td>82.5</td><td>69.5</td><td>82.5</td><td>87.3</td><td></td></tr><tr><td>Z-Reweighting (ACL; Su et al.,(2022))</td><td>71.9</td><td>79.6</td><td>76.5</td><td>78.9</td><td>82.8 82.5</td><td>1</td><td>1</td><td>一</td><td>1</td><td>79.8 78.6</td></tr><tr><td colspan="9"></td></tr><tr><td>Quantum-inspired Systems QWSD (RANLP; Tamburini, (2019))</td><td>1</td><td>70.5</td><td>69.8</td><td>69.8</td><td></td><td>73.6</td><td>54.4</td><td>77.0</td><td>80.6</td><td></td></tr><tr><td>DRWSD (CIKM; Zhang et al., (2022b))</td><td>74.7</td><td>80.8</td><td>78.0</td><td>80.0</td><td>73.4</td><td>82.7</td><td>69.5</td><td>82.9</td><td></td><td>70.6</td></tr><tr><td>QiWSDbase</td><td>74.8</td><td>81.0</td><td>79.3</td><td>80.8</td><td>82.7</td><td></td><td>71.5</td><td></td><td>86.6</td><td>80.4</td></tr><tr><td>QiWSDlarge</td><td>75.2</td><td>82.5</td><td>80.5</td><td>81.2</td><td>82.7 83.2</td><td>83.7 84.2</td><td>71.7</td><td>82.8 83.0</td><td>87.6 87.7</td><td>80.8</td></tr><tr><td colspan="9">Data-enhanced Evaluation Experiments:</td><td>81.8</td></tr><tr><td>SparseLMMS (EMNLP;Berend,(2020))</td><td>73.0</td><td>79.6</td><td>77.3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EWISER (ACL; Bevilacqua et al., (2020))</td><td>75.2</td><td>80.8</td><td>79.0</td><td>79.4 80.7</td><td>81.3</td><td>81.7</td><td>1 66.3</td><td></td><td>1</td><td>78.8</td></tr><tr><td>ESR (EMNLP; Song et al., (2021))</td><td>77.4</td><td>81.4</td><td>78.0</td><td>81.5</td><td>81.8 83.9</td><td>83.1</td><td>71.1</td><td>81.2 83.6</td><td>85.8</td><td>80.1</td></tr><tr><td>QiWSDbase</td><td>75.0</td><td>81.8</td><td>79.5</td><td>81.0</td><td>83.0</td><td>84.0</td><td>72.2</td><td>82.8</td><td>87.5</td><td>80.7</td></tr><tr><td>QiWSDlarge</td><td>75.4</td><td>83.3</td><td>80.8</td><td>81.5</td><td>83.9</td><td>85.3</td><td>73.0</td><td>83.0</td><td>87.6 87.7</td><td>81.1 82.1</td></tr></table></body></html>

Note that the candidate list of word senses includes all word sense definitions in WordNet 3.0. For the case where there are multiple word sense example sentences, the first one is selected by default; for the case where there is no word sense example sentence, the training text is used instead. Evaluation metrics and other unlisted information are subject to the settings of the evaluation framework.

Model Settings: The hardware platform deployed by QiWSD is Ubuntu 18.04, which installs six Tesla P40 GPUs. The development platform is Python 3.6, and the learning framework is Pytorch 1.8. WordNet 3.0 is provided by NLTK 3.5. Versions bert-base-uncased and bertlarge-uncased of BERT are provided by Transformers 4.5. QiWSD based on bert-base-uncased and bert-largeuncased are called $\mathrm { Q i W S D _ { b a s e } }$ and $\mathrm { Q i W S D _ { l a r g e } }$ respectively. Learning rate, epoch and batch size of QiWSD are $\{ 1 e { \ - } 5 , 5 e { - } 6 , 1 e { - } 6 \}$ , 20 and 4 respectively. Other hyperparameters not listed will be given in the published code.

# Baselines

The comparison systems of the standard evaluation experiments select the related work of the past four years as the baselines, including EWISE (Kumar et al. 2019) and LMMS (Loureiro and Jorge 2019) in 2019, SREF (Wang and Wang 2020), ARES (Scarlini, Pasini, and Navigli 2020b),

BEM (Blevins and Zettlemoyer 2020), EWISER (Bevilacqua and Navigli 2020) and SyntagRank (Scozzafava et al. 2020) in 2020, COF (Wang, Zhang, and Wang 2021) and ESR (Song et al. 2021) in 2021, Z-Reweighting (Su et al. 2022) in 2022. In addition, QWSD (Tamburini 2019) and DRWSD (Zhang et al. 2022b), which is also based on quantum theory, is selected. The comparison systems of the data-enhanced evaluation experiments include SparseLMMS (Berend 2020), EWISER (Bevilacqua and Navigli 2020) and ESR (Song et al. 2021). The experimental results of the above systems are all taken from the data published in the original papers.

# Results and Analysis

The results of the standard and data-enhanced evaluation experiments are shown in Tab. 1.

From the perspective of overall performance, QiWSD outperforms the comparison systems in the standard evaluation experiments; QiWSD partially outperforms the comparison systems in the data-enhanced evaluation experiments. The reason for this phenomenon is that the training samples of LTSs are scarce under the standard experiment setting, but a certain number of training samples are provided under the data-enhanced experiment setting. It is conceivable that improving the weak position of LTSs by increasing the amount of data is directly effective for improving the recognition of LTSs. From the gap between the result values, QiWSD does not have a big gap with the comparison systems. The reason is that the number of LTSs is relatively small, and it is difficult to have a significant gap.

Table 2: Experimental results of the ablation experiments under the original (namely SemCor) and LTS datasets.   

<html><body><table><tr><td rowspan="2">Models</td><td>Dev set</td><td colspan="5">Test sets</td></tr><tr><td>SE7</td><td>SE2</td><td>SE3</td><td>SE13</td><td>SE15</td><td>ALL</td></tr><tr><td colspan="7">Dataset: SemCor</td></tr><tr><td>QiWSDbase</td><td>74.8</td><td>81.0</td><td>79.3</td><td>80.8</td><td>82.7</td><td>80.8</td></tr><tr><td>QiWSDbase</td><td>71.1</td><td>77.7</td><td>75.3</td><td>76.3</td><td>78.0</td><td>77.7</td></tr><tr><td colspan="7">Dataset: LTS</td></tr><tr><td>QiWSDbase</td><td>51.0</td><td>52.3</td><td>48.6</td><td>50.1</td><td>51.5</td><td>49.3</td></tr><tr><td>QiWSDbase</td><td>33.3</td><td>37.7</td><td>35.0</td><td>35.9</td><td>37.6</td><td>34.9</td></tr></table></body></html>

From the perspective of detailed performance, the result on the development set is suboptimal and the result on the test set ALL is optimal, indicating that QiWSD does not overfit the training data and has good generalization ability. The poor performance on the test sets Adj. and $A d \nu .$ is due to the fact that there are relatively few LTSs in adjectives and adverbs, so QiWSD, which has the ability to recognize LTSs, cannot give full play to its advantages.

It should be emphasized that compared with the quantuminspired systems QWSD and DRWSD, QiWSD is superior to the comparison systems in various indicators, indicating that it has certain competitiveness.

# Ablation Study under LTS Datasets

Datasets: The datasets of the standard evaluation experiment setting and the constructed LTS datasets are used to carry out the ablation study. LTSs in the training set, development set and test sets of standard evaluation experiments are extracted and constructed as corresponding datasets. We refer to the word senses with less than three samples as LTSs.

Model Settings: Based on $\mathrm { Q i W S D _ { b a s e } }$ , the model that deletes the component of the quantum recognition method is called $\mathrm { Q i W S D _ { b a s e } ^ { - } }$ ; the model that retains the component of the quantum recognition method is called $\mathrm { Q i W S D _ { b a s e } ^ { + } }$ , which is the original model. Other information not listed remains the same as the above model settings.

Result Analysis: The experimental results are shown in Tab. 2, and the analysis is as follows:

• On the original datasets, the results of $\mathrm { Q i W S D _ { b a s e } ^ { - } }$ are obviously lower than those of $\mathrm { Q i W S D _ { b a s e } ^ { + } }$ , which shows that the quantum recognition method is valuable and helpful to the overall performance of the WSD system. The reason for the gap of 3-4 percentage points in the result values is that the proportion of LTSs is relatively small. • On the LTS datasets, the results of $\mathrm { Q i W S D _ { b a s e } ^ { - } }$ are also significantly lower than those of $\mathrm { Q i W S D _ { b a s e } ^ { + } }$ , indicating that for the recognition of LTSs, the role of the quantum recognition method is significant. This is corroborated by a gap in the result values of around 15 percentage points.

100 XLMR-Base QiWSDbase   
90 7928.6 82.3 283.7. 82717680.0 QiWSDbase   
80 7675.8   
70 7782.5 7272.5 rerral 51.052.6 49.8   
50 143.5.3 47.8 41.1 43.043.3   
40   
30 5g! vgar atale h oati anis Jutc Est onia enc icis T Mgari 公 中 tor rean Sovenia st C

# Experiments under Cross-Lingual Datasets

The generalization ability of QiWSD in other languages is verified under the latest cross-lingual datasets4 proposed by Pasini et al. (Pasini, Raganato, and Navigli 2021). The performance in minority languages can also reflect the role of the quantum-like model from the side. The experimental models are $\mathrm { Q i W S D _ { b a s e } ^ { + } }$ and $\mathrm { Q i W S D _ { b a s e } ^ { - } }$ proposed by ablation experiments. The comparison model, XLMR-Base (Conneau et al. 2020), is the model used in the original paper. The encoders of the model are implemented by bertbase-multilingual-cased of BERT. Note that since the crosslingual datasets are constructed based on BabelNet5, the glosses and example sentences in this section are from BabelNet.

The experimental results are shown in Fig. 3. From the overall performance, $\mathrm { Q i W S D _ { b a s e } ^ { + } }$ is better than XLMR-Base to a certain extent and $\mathrm { Q i W S D _ { b a s e } ^ { + } }$ is definitely better than $\mathrm { Q i W S D _ { b a s e } ^ { - } }$ , which shows that QiWSD has a certain generalization ability.

# Conclusions

For long-tail WSD, the word sense definitions are limited, and it is difficult to obtain clear and easily distinguishable representations. Researchers propose to expand multisource data to deal with it, but the unavoidable assumption is homogeneous data. This paper proposes a quantum-like model that simultaneously fuses representations obtained from homologous and non-homologous data, which means that the model has a stronger tolerance for multi-source data. The WSD system constructed based on the quantum-like model is verified under the WSD evaluation framework, the constructed LTS datasets and the cross-lingual datasets, and the experimental results show its effectiveness.

In future work, its internal mechanism and applicable fields will be further clarified. At the same time, systems based on the quantum-like model are constructed and verified on other tasks.