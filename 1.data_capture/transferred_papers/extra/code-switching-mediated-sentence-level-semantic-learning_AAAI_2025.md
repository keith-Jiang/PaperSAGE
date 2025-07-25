# Code-switching Mediated Sentence-level Semantic Learning

Shuai Zhang1, Jiangyan $\mathbf { X _ { i } ^ { \bullet 1 } }$ , Zhengqi Wen1, Jianhua Tao 1\*, Feihu Che 1\*, Jinyang $\mathbf { W } \mathbf { u } ^ { 1 }$ , Ruibo Fu2

1Department of Automation & BNRist, Tsinghua University 2Institute of Automation, Chinese Academy of Sciences {zhang shuai, yijy, zqwen, jhtao, qkr, wu-jy23}@mail.tsinghua.edu.cn, ruibo.fu@nlpr.ia.ac.cn

# Abstract

Code-switching is a linguistic phenomenon in which different languages are used interactively during conversation. It poses significant performance challenges to natural language processing (NLP) tasks due to the often monolingual nature of the underlying system. We focus on sentence-level semantic associations between the different code-switching expressions. And we propose an innovative task-free semantic learning method based on the semantic property. Specifically, there are many different ways of languages switching for a sentence with the same meaning. We refine this into a semantic computational method by designing the loss of semantic invariant constraint during the model optimization. In this work, we conduct thorough experiments on speech recognition, speech translation, and language modeling tasks. The experimental results fully demonstrate that the proposed method can widely improve the performance of code-switching related tasks.

# Introduction

Code-switching is a common linguistic phenomenon in which several languages are used interactively during conversation (Poplack 1981). The number of multilingual speakers far outnumbers monolingual speakers in the worldwide population (Tucker 2003; Winata et al. 2021). Codeswitching expressions are widely used in a variety of scenarios, including and not limited to daily conversations, classroom teaching, conferences, social media, etc. It is a strong incentive to develop technologies that can handle codeswitching efficiently. However, progress in this area has been limited, primarily since code-switching typically occurs during informal expressions, such as spoken language, where real-time data collection is difficult (Sitaram et al. 2019; Jose et al. 2020; Doruz et al. 2021). The significant increase of smart speech devices has alleviated this problem, while new technologies are developed to fulfill users’ needs for multilingual interaction.

Code-switching computation research focuses on two main areas: speech processing and text processing. Automatic speech recognition (ASR) and speech synthesis are the most widely researched tasks in speech processing (Graves et al. 2006; Graves, Mohamed, and Hinton 2013; Jia et al. 2019; Sperber et al. 2019; Be´rard et al. 2016), and the most interesting tasks for researchers in text processing are language identification, sentiment analysis, and language modeling, with some other research in text classification, question answering, and sequence labeling (Solorio et al. 2014; Molina et al. 2019; Patra, Das, and Das 2018; Qin et al. 2020; Zheng et al. 2021). This strong correlation between task and venue shows that the speech-processing and textprocessing communities remain somewhat fragmented and tend to work in isolation from each other. Such research isolation limits the study of code-switching and hinders the ability to draw more generalized task-independent insights (Napoli et al. 2014).

![](images/5f7fc630a020a24019874295e44d18210894a612dc53f6322167daa11c20d516.jpg)  
(a) code-switching formal space (b) sentence-level semantic space   
Figure 1: Schematic illustration of the correspondence between code-switching expressions and semantics. (a) represents the code-switching formal space, where each point represents a language mixing modality and similar expressions are distributed in adjacent regions. (b) represents the sentence-level semantic space, each dot denotes a sentence semantic, with correspondences to the many different forms of the left diagram.

This study aims to explore the semantic association across code-switching expressions and develop a novel generalized semantic learning method for various tasks. Specifically, the process of code-switching occurs with a certain degree of randomness, influenced by linguistic rules, social psychology, and other factors (Poplack 1981). There are multiple legitimate textual candidates for each position in a sentence, leading to many different expression forms of the same sentence-level semantic. As shown in Figure 1, each dot in the left graph represents a code-switching formal, and multiple dots together correspond to the same semantic in the right semantic space. We refine this into an explicit semantic computational method by designing the loss of semantic invariant metrics across sentences. To verify the effectiveness and generalization of the proposed method, we conduct experiments on typical research tasks in the fields of speech and text processing, including ASR, automatic speech translation (AST), and language modeling. We also explore the capability boundaries of the proposed method by examining the generative capabilities of large language models (LLM) for code-switching. The results indicate a significant improvement in performance for each task. Additionally, we demonstrate the effectiveness of the proposed method in semantic modeling through visual qualitative analysis of the samples. This indicates that the methodology presented in this paper can provide a novel perspectives on the study of code-switching computational methods, which can benefit a wide range of related research tasks.

Our main contributions are summarized as follows:

• We utilize the semantic properties of code-switching to achieve a task-free approach to semantic learning, with implications for related research in both speech and text domains.   
• We use code-switching as a mediator to design taskrelated prompts for efficient unified modeling of ASR and AST tasks.   
• Detailed experimental results on a variety of different tasks with careful analysis prove that our method significantly outperforms the baseline model and some existing methods in terms of performance and semantic modeling ability.

# Related Work

ASR refers to the transcription of code-switched speech into corresponding text, and AST refers to the direct translation of speech into another language. Recently, the end-to-end model has attracted attention in the two fields for its extremely simplified architecture without complicated pipeline systems (Graves et al. 2006; Graves, Mohamed, and Hinton 2013; Jia et al. 2019; Sperber et al. 2019; Be´rard et al. 2016). Some work applies multi-task learning to train AST and ASR task jointly (Weiss et al. 2017; Anastasopoulos and Chiang 2018; Berard et al. 2018; Vydana et al. 2021; Nakayama et al. 2019). Some work uses semantic information to improve the quality of AST (Dong et al. 2021a,b). The semantic information usually comes from two aspects, one is pre-trained models (Dong et al. 2021a), such as BERT (Kenton and Toutanova 2019), and the other is from acoustic features (Dong et al. 2021a). However, these methods do not establish semantic associations between ASR and AST, making it difficult to achieve efficient unified modeling and limiting further performance improvements.

# Semantic Information for Code-switching

Most research on extracting semantic information for codeswitching involves transforming linguistic theories into computable forms to improve code-switching-related tasks (Qin et al. 2020; Zheng et al. 2021; Li and Fung 2013, 2014). Another approach is to use code-switching as a data augmentation method to enhance the performance of the multilingual model (Qin et al. 2020). However, this approach does not explicitly model the semantic relationships between code-switching expressions. (Zheng et al. 2021) notes semantic associations, however, it is still considered as a data augmentation method to enhance text tasks that do not involve cross-modal code-switching task. It lacks systematic observation and validation of code-switching research.

# Methodology

# Semantic Invariance Constraint

We illustrate the principle of semantic invariance constraint based on ASR and AST tasks. As shown in Figure 2, for a code-switching speech input, three kinds of texts are constructed, corresponding to English AST task, codeswitching ASR task, and Chinese AST task. Despite the different format, the three target text clearly express the same semantic. This work extracts semantic representations and implements semantic constraints by measuring the invariance between semantic.

Two methods are employed to extract sentence-level semantic vector, one is performing an average pooling operation on the decoder contextual vectors to obtain the corresponding semantic representation. Another is to add a special symbol [CLS] in the sentence and integrate the semantic information of the whole sentence through the attention mechanism. For example, ${ < } E N G { > }$ you are so cute $\left[ C L S \right]$ .

After obtaining the sentence-level semantic representation of different tasks, we measure the semantic distance between tasks according to the semantic invariance. In order not to lose generality, we describe our approach using the case of CS and the two corresponding monolinguals. The semantic invariance loss can be expressed as follows,

$$
\mathcal { L } _ { s i l } ( \theta ) = \mathcal { D } ( \theta ; { \bf s } _ { A } , { \bf s } _ { E } ) + \mathcal { D } ( \theta ; { \bf s } _ { A } , { \bf s } _ { C } ) + \mathcal { D } ( \theta ; { \bf s } _ { C } , { \bf s } _ { E } )
$$

where $\mathcal { L } _ { s i l }$ refers to the total semantic invariance loss, $\theta$ refers to the model parameter, $\mathcal { D }$ refers to the distance calculator between semantic representation, $\mathbf { s } _ { A } , \mathbf { s } _ { E } , \mathbf { s } _ { C }$ refer to semantic vectors of ASR, English AST, Chinese AST respectively.

# Model Details

Problem Formulation The data used in this paper contain speech-transcription-translations quadruples, denoted as $\begin{array} { r } { S { \bf \bar { \Psi } } = \left( { \bf x } , { \bf z } , { \bf e } , { \bf c } \right) } \end{array}$ . Specially, $\textbf { x } = ~ ( x _ { 1 } , . . . , x _ { T _ { x } } )$ , $\textbf { z } =$ $\big ( z _ { 1 } , . . . , z _ { T _ { z } } \big )$ , $\mathbf { e } = ( e _ { 1 } , . . . , e _ { T _ { e } } )$ , $\mathbf { c } = ( c _ { 1 } , . . . , c _ { T _ { c } } )$ represent the acoustic features sequence, the corresponding transcription, the translation of English and the translation of Chinese respectively. And the $T _ { x }$ is the frame number of the speech sequence. The $T _ { z } , T _ { e } , T _ { c }$ are the lengths of the above three target sequences. The goal is to model the all three target sequences simultaneously $( \mathbf { z } , \mathbf { e } , \mathbf { c } )$ based on the acoustic features $\mathbf { x }$ .

![](images/5aec31bd631f3acdee41e1bc47dbc4745f97f4ae03d5d32d688659b0edce006c.jpg)  
Figure 2: The model architecture of unified ASR and AST task learning. The left part is the acoustic encoder which takes acoustic features as input. Its main component is a convolution enhanced transformer structure for efficient encoding of acoustic feature. The right part is the multi-task decoder. It receives text input for the three tasks and extracts information from the acoustic encoder while modeling ASR and AST.

Model Components In this section, we illustrate the structure of our model and how it deal with three different tasks simultaneously. As shown in Figure 2, the overall architecture of the model consists of two modules: a) an acoustic encoder network that encodes the speech features sequence into a high-level hidden representation; b) a multimask decoder receives text input for the three tasks and extracts information from the acoustic encoder while modeling ASR and AST. One can freely choose the structure of the encoder and decoder, such as transformer network, recurrent neural network, convolution network, and so on. We adopt transformer as the backbone network. It is now the state-ofthe-art model in the translation task, and it also shows excellent performance in the ASR field. For details of the model, please refer to (Gulati et al. 2020).

Acoustic Encoder. The acoustic encoder receives the input of low-level acoustic features and outputs the highlevel hidden representation. It is based on the conformer, a convolution-augmented transformer structure. Since the number of acoustic feature frames is much larger than the length of the corresponding text, the down-sampling technique is essential. We adopt the 2D CNN layer to produce the down-sampled acoustic hidden representation. After a linear layer, the positional encoding is used to attend relative positions. Then a stack of $N _ { e }$ conformer blocks is used to get the final encoded representation. Each conformer module mainly includes three modules, which are multihead self-attention module, convolution module, and feedforward module in sequence. Compared with the classic transformer structure, it adds a convolution module to extract local information in acoustic encoding.

Multi-Task Decoder. For the decoder, a learnable word embedding and positional encoding are applied to the target sequence. Then a stack of $N _ { d }$ decoder blocks is subsequent. The decoder mainly consists of three parts: multihead self-attention, multi-head cross-attention, and feed forward network. The multi-head self-attention is used to encode multi-task input text to obtain high-dimensional encoding representation. The multi-head cross-attention takes the high-dimensional representation as the query vectors and performs cross-attention computation on the output vectors of the acoustic encoder to get the contextual vectors. For the self-attention, the query, key, and value are the target text embedding. For the multi-head cross attention, the key and value come from the encoder outputs and the query comes from the previous sub-block outputs. The feed-forward network performs further encoding on the context vectors, followed by dimensional transformation and softmax to get the final decoder output.

# Loss Function

The loss function consists of three parts, including connectionist temporal classification (CTC) loss (Graves et al. 2006), cross-entropy loss, and semantic invariance loss. The cross-entropy loss is the sum of the losses for the three tasks.

$$
\mathcal { L } _ { c e } ( \boldsymbol { \theta } ; \mathbf { x } , \mathbf { z } , \mathbf { e } , \mathbf { c } ) = \mathcal { L } _ { c e } ( \boldsymbol { \theta } ; \mathbf { x } , \mathbf { z } ) + \mathcal { L } _ { c e } ( \boldsymbol { \theta } ; \mathbf { x } , \mathbf { e } ) + \mathcal { L } _ { c e } ( \boldsymbol { \theta } ; \mathbf { x } , \mathbf { c } )
$$

In this paper, we use Kullback-Leibler divergence (KL) and mean squared error (MSE) to measure the similarity between semantic vectors. Therefore, the overall loss function for end-to-end multi-task training is the weighted sum for the above three parts:

$$
\mathcal { L } _ { a l l } ( \theta ) = \alpha \mathcal { L } _ { c e } ( \theta ) + ( 1 - \alpha ) \mathcal { L } _ { C T C } ( \theta ) + \beta \mathcal { L } _ { s i l } ( \theta ) \mathrm { ~ , ~ }
$$

where the $\alpha$ is hyper-parameters to balance the cross entropy loss $\mathcal { L } _ { c e } ( \theta )$ and the CTC loss ${ \mathcal { L } } _ { C T C } ( \theta )$ . The hyperparameter $\beta$ is used to adjust the weight of the semantic invariance loss $\mathcal { L } _ { s i l } ( \theta )$ in the total loss.

CTC Auxiliary Module CTC is an alignment-free object function for sequence-to-sequence modeling. It counts all possible output sequence forms corresponding to the input sequence based on the idea of dynamic programming. CTC loss is often used as an auxiliary loss for speech translation tasks.

The loss function directly maximizes the probabilities of the correct label.

$$
P ( \mathbf { z } | \mathbf { x } ) = \sum _ { \pi \in \mathcal { B } ^ { - 1 } ( \mathbf { z } ) } P ( \pi | \mathbf { x } ) = \sum _ { \pi } \prod _ { t = 1 } ^ { T } P ( \pi _ { t } | \mathbf { x } )
$$

where $\mathbf { T }$ is frame length of input sequence and $\mathbf { B }$ is a manyto-one mapping $\mathbf { B } : \mathbf { Z } \cup \{ b l a n k \}  \mathbf { Z } . \mathbf { \mu }$ $\mathbf { Z }$ is the label unit set. $\mathbf { B }$ indicates the label sequence $\mathbf { y }$ and its corresponding set of CTC paths $\pi$ . The mapping is by inserting an blank between each label unit in y. $\bar { P } ( \bar { \pi } _ { t } | \mathbf { x } )$ is estimated from the neural network taking the feature sequence $\mathbf { x }$ as the input. With the conditional independent assumption, $P ( \pi | \mathbf x )$ can be decomposed into a product of posterior $P ( \pi _ { t } | \mathbf { x } )$ in each frame $\mathbf { t }$ . Finally, the CTC loss used in this work is defined as

$$
\mathcal { L } _ { C T C } ( \boldsymbol { \theta } ; \boldsymbol { x } , z ) = - l o g P ( \mathbf { z } | \mathbf { x } )
$$

A linear layer is used to transform the output of the acoustic encoder to the appropriate dimension, and then a softmax layer is used for probability normalization. The computation of the CTC loss is performed using the transformed sequence. After training, the probability values of non-blank units are concentrated in a few spikes, as shown in Figure 2. In this work, we only compute the CTC loss for the task of speech recognition.

# Training and Inference

During the training process, the ASR and AST tasks are carried out simultaneously. A batch of training data is randomly sampled, consisting of acoustic features and their corresponding target texts. The forward calculation process is completed for each target text in the batch, followed by uniform gradient back-propagation and parameter updates. To differentiate between tasks, a task ID is added to the beginning of each target text. As shown in Figure 2, the task IDs ${ < A S R > }$ , ${ < } E N G { > }$ , ${ \angle C H N > }$ refer to the ASR task, English translation task, and Chinese translation task. These IDs are essential for the unified training of different tasks, as they can bias the same model for different tasks.

Table 1: Code-switching audio data distribution information in each dataset. CS stands for code-switch and Mono for monolingual.   

<html><body><table><tr><td>data</td><td>split type</td><td>hours</td><td>language</td></tr><tr><td rowspan="3">ASRU2019</td><td>Train CS</td><td>200</td><td rowspan="3">Mandarin-English</td></tr><tr><td>Dev</td><td>CS 20</td></tr><tr><td>Test</td><td>CS 20</td></tr><tr><td rowspan="3">Fisher</td><td>Train</td><td>Mno 13.28</td><td rowspan="3">English-Spanish</td></tr><tr><td>Dev</td><td>Mono 1.45</td></tr><tr><td>Test</td><td>CS 1.63</td></tr></table></body></html>

In the inference process, only these three task IDs need to be provided to decode the three target texts simultaneously. The decoding process is made via auto-regressive forms which is same as ordinary end-to-end ASR.

Code-switching Capabilities for LLM The powerful generative capabilities of LLM have recently been impressive, and we utilize LLM to explore the boundaries of the validity of our approach. Specifically, we fine-tune the LLM using code-switching text data, adding semantic invariant loss in the process. The fine-tune data passes through the text corresponding to the speech data used in this work. Specifically, the input is monolingual text and the output is codeswitching text. The quality of the code-switching data generated by the LLM was evaluated to judge the effectiveness of our method.

# Experiments

# Data

We conduct our experiments on three popular publicly available datasets, including the ASRU 2019 Mandarin-English code-switching challenge dataset (Shi, Feng, and Xie 2020), Fisher dataset (Cieri, Miller, and Walker 2004) and TED English-Chinese dataset (Liu et al. 2019). The ASRU 2019 dataset is designed for code-switching ASR task. Although the Fisher dataset is not a code-switching focused dataset, it contains a large amount of (annotated) code-switching utterances. Fortunately, the dataset has a corresponding annotated English translation. The Fisher data consists of three evaluation sets (Dev/Dev2/Test) that together contain approximately a thousand instances of code-switching with corresponding translations in monolingual English. We therefore combined all the code-switching data from the three evaluation sets as a test set. Statistical information on the code-switching dataset is shown in Table 1. However the first two datasets are designed for ASR task and are less often used for AST tasks. Therefore to better compare with other methods on the AST, we conduct experiments on the public TED English-Chinese speech translation dataset. It contains 528 hours of English speech and corresponding annotated Chinese translations.

# Data Preprocessing

In this paper, the input acoustic features of the encoder network are a 40-dimensional filter bank with $2 5 \mathrm { m s }$ windowing and 10ms frameshift, which are extended with mean and variance normalization. For all ASR transcription, we remove punctuation and lowercase all English words to keep more consistent with the output of ASR. For the ASRU2019 code-switching challenge dataset, we first use Llama 3 70B to get the corresponding Chinese translation text and then perform a sample-by-sample manual check for corrections. For the Chinese translation, we segment the sentence into characters. We keep about 3500 characters as the modeling units. For the Fisher data, the sentences are encoded using the BPE method, with a shared vocabulary of 2000 subwords. For the TED English-Chinese dataset, the processing method of ASR transcription and translation text is similar to the previous method.

# Experimental Results Evaluation Metrics

For the code-switching ASR task, we use a mix error rate (MER) to evaluate the experimental results of our methods. The MER is defined as the word error rate (WER) for English and the character error rate (CER) for Mandarin. For the English ASR task, the WER is used as the evaluation index. For the Chinese and English translation tasks, we report case-insensitive BLEU (Papineni et al. 2002) scores $\mathtt { B L E U } _ { c i }$ and character-level BLEU scores $\mathtt { B L E U } _ { c l }$ respectively.

# Experimental Details

All of the models are implemented based on transformer architecture. For the input acoustic features, two $3 { * 3 }$ 2D CNN down-sampling layers with stride 2 are used. The dimension of the subsequent linear layer is 512. Relative position encoding is used to model position information. The attention dimensions of the encoder and decoder are both 512 and the number of the head is 4. The dimension of position-wise feed-forward networks is 1024. The number of acoustic encoder blocks and decoder blocks are 12 and 6 respectively. To avoid over-fitting, the unified label smoothing technique is used, and the parameter is set to 0.1. SpecAugment with frequency masking $\mathrm { F } { = } 3 0$ , ${ \mathrm { m F } } { = } 2$ ) and time masking $\scriptstyle ( \mathrm { T } = 4 0$ , $\mathrm { m T } { = } 2$ ) is used to improve the performance of the models (Park et al. 2019). Meanwhile, we set the residual dropout as 0.1, where the residual dropout is applied to each sub-block before adding the residual information. We use Adam optimizer with $\bar { \beta _ { 1 } } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 8 , \epsilon = 1 e ^ { - 8 }$ on 4 NVIDIA A100 GPUs. The batch size is set to 128 during the training process. The learning rate is set by a warm-up strategy. We perform decoding using beam search with a beam size of 10.

# Hyper-parameter Selection

There are two hyper-parameters in Equation (4), which are used to balance the weights of the cross-entropy loss, the CTC loss, and the semantic invariance loss. First, reasonable hyper-parameters are determined based on the ASRU 2019 dataset for subsequent experiments. As shown in Table 2, when $\alpha$ is set to 0.7 and $\beta$ is set to 0.1, both ASR and

Table 2: Effects of hyper-parameters in loss.   

<html><body><table><tr><td rowspan="2">a</td><td rowspan="2">β</td><td>ASRU2019Dev</td></tr><tr><td>MER(↓) BLEUct (↑)</td></tr><tr><td>0.7</td><td>0.1</td><td>10.55 24.71</td></tr><tr><td>0.7</td><td>0.05</td><td>10.87 23.85</td></tr><tr><td>0.7</td><td>0.01</td><td>10.96 23.70</td></tr><tr><td>0.8</td><td>0.1</td><td>11.01 24.05</td></tr><tr><td>0.8</td><td>0.05</td><td>11.32 24.11</td></tr><tr><td>0.8</td><td>0.01</td><td>10.79 24.14</td></tr></table></body></html>

Table 3: Results of ASR and AST on Fisher test set. The semantic extraction method and the similarity calculation metric are abbreviated SEM and SCM, respectively.   

<html><body><table><tr><td>model</td><td>SEM</td><td>SCM</td><td>WER(↓)</td><td>BLEUci(↑)</td></tr><tr><td>Pretrained Multi-task (weller et al)</td><td>1 1 1</td><td>1 1</td><td>30.21 30.57</td><td>25.31 25.83</td></tr><tr><td>proposed</td><td>[CLS] [CLS] ave_pool ave_pool</td><td>1 KL MSE KL</td><td>30.00 28.33 28.72 28.51 28.49</td><td>25.60 26.88 27.02 26.85</td></tr></table></body></html>

AST tasks can achieve satisfactory results. All subsequent experiments use these parameter settings.

# Favorable Effects on Code-switching ASR

The method is first evaluated on the imbalanced codeswitching dataset. Two baselines are used: the first pretrains the AST model using ASR data and then finetunes the model on AST data. The second baseline is a multi-task learning model where the ASR and AST models are jointly trained with independent decoders and a shared acoustic encoder. There are two methods for semantic extraction: the $\left[ C L S \right]$ method, which is similar to using the BERT model for classification, and the average pooling method ave pooling. For the semantic similarity calculation, we use $K L$ and $M S E$ . The results show that there is not much difference between the different calculation methods. To be precise, the performance is relatively better when using the $\left[ C L S \right]$ and the $M S E$ at the same time. Based on these two methods, we conduct the following ASR experiments.

Upon further analysis of the experimental results presented in Table 4, it can be observed that the ASR and AST tasks mutually reinforce each other. This mutual promotion can be attributed to the fact that both tasks share the same semantic space and our unified modeling approach better satisfies this condition than pre-training and multi-task training. Subsequent ablation experiments demonstrate that our method continues to outperform several baseline models even after removing the semantic invariance loss. This shows that our implicit semantic modeling scheme can enhance the performance of both ASR and AST at the same time.

To enhance the method’s credibility, we compare the performance of code-switching ASR with other existing research results. This comparison is conducted under the same training data conditions, using only code-switching data from the dataset. The results in Table 3 and Table 4 demonstrate that our method outperforms others under the same training data conditions. As shown in Table 5, we achieve the state-of-the-art recognition performance on the TED dataset.

Table 4: Results of ASR and AST on ASRU2019 code-switching test and dev sets. Unless otherwise noted, Dev and Test in all tables below belong to this dataset. The semantic extraction method and the similarity calculation metric are abbreviated SEM and SCM, respectively.   

<html><body><table><tr><td rowspan="2">model</td><td rowspan="2">SEM</td><td rowspan="2">SCM</td><td colspan="2">Dev</td><td colspan="2">Test</td></tr><tr><td>MER(↓)</td><td>BLEUct (↑)</td><td>MER(↓)</td><td>BLEUc (↑)</td></tr><tr><td>Pretrained</td><td>1</td><td>1</td><td>11.53</td><td>76.39</td><td>11.25</td><td>77.11</td></tr><tr><td>Multi-task</td><td>1</td><td>1</td><td>11.31</td><td>78.73</td><td>11.01</td><td>78.98</td></tr><tr><td>(Lu et al. 2020)</td><td>1</td><td>1</td><td>1</td><td>1</td><td>11.84</td><td>-</td></tr><tr><td>(Zhang et al. 2021b)</td><td>1</td><td>1</td><td>11.21</td><td>1</td><td>10.51</td><td>1</td></tr><tr><td>(Zhang et al. 2021a)</td><td>1</td><td>1</td><td>12.67</td><td>1</td><td>11.94</td><td>1</td></tr><tr><td>(Yan et al. 2021)</td><td>二</td><td>二</td><td>1</td><td>二</td><td>11.1</td><td>1</td></tr><tr><td rowspan="4">proposed</td><td>[CLS]</td><td>KL</td><td>10.76</td><td>81.72</td><td>10.53</td><td>82.31</td></tr><tr><td>[CLS]</td><td>MSE</td><td>10.55</td><td>81.42</td><td>10.37</td><td>82.61</td></tr><tr><td>ave_pool</td><td>KL</td><td>10.91</td><td>81.21</td><td>10.61</td><td>82.43</td></tr><tr><td>ave_pool</td><td>MSE</td><td>10.78</td><td>81.30</td><td>10.51</td><td>82.52</td></tr></table></body></html>

Table 5: Results of ASR and AST on TED English-Chinese test set.   

<html><body><table><tr><td>model</td><td>SEM</td><td>SCM</td><td>Enc Pretrain</td><td>Dec Pretrain</td><td>WER(↓)</td><td>BLEUct(1)</td></tr><tr><td>Transformer+pretrain (Liu et al. 2019)</td><td></td><td>1</td><td></td><td></td><td>1</td><td>16.80</td></tr><tr><td>+ knowledge distillation (Liu et al. 2019)</td><td>1</td><td>1</td><td></td><td></td><td>1</td><td>19.55</td></tr><tr><td>Multi-task+pretrain (Inaguma et al. 2019)</td><td>1</td><td>1</td><td></td><td>X</td><td>1</td><td>20.45</td></tr><tr><td>Interactive decoding(Liu et al. 2020)</td><td>1</td><td>1</td><td>×</td><td>×</td><td>13.38</td><td>21.68</td></tr><tr><td>COSTT without pretraining(Dong et al. 2021a)</td><td>1</td><td>1</td><td>X</td><td>X</td><td></td><td>21.12</td></tr><tr><td rowspan="4">proposed methods</td><td>[CLS]</td><td>KL</td><td>X</td><td>X</td><td>11.35</td><td>22.11</td></tr><tr><td>[CLS]</td><td>MSE</td><td>X</td><td>X</td><td>11.19</td><td>21.50</td></tr><tr><td>ave_pool</td><td>KL</td><td></td><td>X</td><td>12.12</td><td>21.32</td></tr><tr><td>ave_pool</td><td>MSE</td><td></td><td>×</td><td>12.05</td><td>21.36</td></tr></table></body></html>

# Favorable Effects on Code-switching AST

The semantic enhancement for speech translation is even more obvious. As shown in Table 4, our method has a significant improvement over the strong baseline. And the high $\mathtt { B L E U } _ { c l }$ score for the Chinese translation task is due to the dominance of Chinese data. This indicates that the task of Chinese translation is easier, which is closely related to Chinese speech recognition. The consistency improvement of our approach can be seen in Table 3. The difference is that the metrics are a bit lower compared to Table 4, which is due to the more balanced language distribution of this dataset, which enhances the difficulty of the speech translation task. Overall, our proposed method outperforms the baseline model in all metrics.

As the above two code-switching datasets are intended for ASR, it is not possible to compare the performance of AST with other existing methods. Therefore, we conducted experiments on the TED English-Chinese dataset, which is designed for AST. Table 5 presents a comparison with existing studies. Only a few research works provide error rate metrics for ASR, and we achieve a relative performance improvement of $1 6 . 3 7 \%$ . Our method achieves better performance than other methods on the AST and the experimental results demonstrate its effectiveness.

# Code-switching Generation of LLM

We evaluate the proposed method using two LLM at different scales, llama2-7b and llama2-13b (Touvron and et al 2023). The LLM is fine-tuned using code-switching text data, with the addition of semantic invariant loss. One hundred pieces of code-switching data were generated based on the one hundred monolingual data prompts provided. The validity of the data was determined through manual evaluation. Table 7 demonstrates that the LLM’s ability to generate legitimate code-switching data is weak. However, finetuning the code-switching data can effectively improve this ability. Our method is equally effective on the LLM.

![](images/691bfbbb4e92bb8c92a586c2bf51040ce2c2e1276d8fdaa2379b8a729593d923.jpg)  
Figure 3: Sample visualizations of different methods. From left to right the pretrained model, the multi-task model, and our model. No code-switching refers to samples from TED English-Chinese dataset.

Table 6: Ablation experimental results of semantic invariance loss. (with/without)   

<html><body><table><tr><td>Metrics</td><td>Dev of ASRU2019</td><td>Testof ASRU2019</td><td>Test of Fisher</td><td>Test set of TED</td></tr><tr><td>MER(↓)</td><td>10.55/10.91</td><td>10.37/10.77</td><td>28.72/29.33</td><td>11.19/12.32</td></tr><tr><td>BLEUci(↑)</td><td>/</td><td></td><td>27.02/25.56</td><td>-/</td></tr><tr><td>BLEUct (↑)</td><td>81.42/80.37</td><td>82.61/80.81</td><td>1/1</td><td>21.50/21.07</td></tr></table></body></html>

Table 7: Qualification rate $( \% )$ of the code-switching data generated by LLM.   

<html><body><table><tr><td>model</td><td>llama2-7B</td><td>llama2-13B</td></tr><tr><td>No finetune</td><td>23%</td><td>44%</td></tr><tr><td>supervised finetune</td><td>28%</td><td>53%</td></tr><tr><td>proposed</td><td>33%</td><td>59%</td></tr></table></body></html>

# Effect of Semantic Invariance Loss

To assess the impact of semantic invariance loss in our approach, we conducted ablation experiments to analyze the results. Table 6 presents the experimental outcomes with and without semantic invariance loss. Overall, the inclusion of semantic invariance loss is beneficial for both ASR and AST tasks. Notably, this loss has a greater impact on AST than ASR, possibly due to the greater importance of semantic information in AST tasks. Furthermore, our proposed unified modeling approach achieves superior performance compared to baseline methods, even without the semantic invariance loss. The results obtained from the TED EnglishChinese dataset also demonstrate highly competitive performance when compared to other existing methods.

# Semantic Visualization

To demonstrate more intuitively the semantic modeling capabilities of our method, we visualize word embedding representations in different languages. We use the t-SNE toolkit (Van der Maaten and Hinton 2008) to realize the dimension reduction operation of word embedding. Obviously, it can be seen that the semantic distribution of the pre-trained model is very chaotic due to the lack of semantic modeling constraints. The semantic distribution of the multitask model is relatively regular, but most of the word pairs are still far apart. In Figure 3(c), we can intuitively observe that the distance between the synonym pairs is closer. Our method can effectively learn semantic information by sharing the semantic space and losing the semantic invariance of multi-tasks.

To explore the role of Chinese-English code-switching data in semantic modeling, we select word pairs from the TED English-Chinese dataset, which does not contain codeswitching data, and visualize them in Figure 3(d). It can be observed that their semantic distribution is relatively regular, but the distribution between synonym pairs is more scattered compared to Figure 3(c). This suggests that codeswitching data plays a facilitating role in semantic modeling. This may be due to the co-occurrence of Chinese and English in the same sentence in the code-switching data. This co-occurrence makes the code-switching data closer to the Chinese and English monolingual data and acts as an intermediate bridge connecting the monolingual data.

# Conclusion

In this paper, we focus on exploring the sentence-level semantic associations between different code-switching expressions. We propose a task-free semantic learning method based on this analysis. The model can learn the common semantic information from different tasks by sharing semantic space. We refine this into a semantic computational method by designing the loss of semantic invariant metrics across sentences. Experiments are conducted on tasks such as language modeling, ASR, and AST. The results indicate a significant improvement in performance for each task. This suggests that semantic constraint is a widely applicable method in the context of code-switching.