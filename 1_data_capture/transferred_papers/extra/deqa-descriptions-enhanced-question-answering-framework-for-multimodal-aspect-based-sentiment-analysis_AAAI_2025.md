# DEQA: Descriptions Enhanced Question-Answering Framework for Multimodal Aspect-Based Sentiment Analysis

Zhixin Han1, Mengting $\mathbf { H } \mathbf { u } ^ { 1 * }$ , Yinhao Bai2, Xunzhi Wang1, Bitong Luo1

1College of Software, Nankai University $\bar { } ^ { 2 } \mathbf { J } \bar { \mathbf { D } }$ AI Research, Beijing, China zhixinhan $@$ mail.nankai.edu.cn, mthu $@$ nankai.edu.cn, yinhaobai68 $@$ gmail.com, xunzhi $@$ mail.nankai.edu.cn, luobitong $@$ mail.nankai.edu.cn

# Abstract

Multimodal aspect-based sentiment analysis (MABSA) integrates text and images to perform fine-grained sentiment analysis on specific aspects, enhancing the understanding of user opinions in various applications. Existing methods use modality alignment for information interaction and fusion between images and text, but an inherent gap between these two modalities necessitates a more direct bridging mechanism to effectively connect image understanding with text content. For this, we propose the Descriptions Enhanced Question-Answering Framework (DEQA), which generates descriptions of images using GPT-4, leveraging the multimodal large language model to provide more direct semantic context of images. In DEQA, to help the model better understand the task’s purpose, we frame MABSA as a multiturn question-answering problem to add semantic guidance and hints. We input text, image, and description into separate experts in various combinations, allowing each expert to focus on different features and thereby improving the comprehensive utilization of input information. By integrating these expert outputs within a multi-turn question-answering format, we employ a multi-expert ensemble decision-making approach to produce the final prediction results. Experimental results on two widely-used datasets demonstrate that our method achieves state-of-the-art performance. Furthermore, our framework substantially outperforms GPT-4o and other multimodal large language models, showcasing its superior effectiveness in multimodal sentiment analysis.

# Introduction

Multimodal aspect-based sentiment analysis (MABSA) (Ju et al. 2021) is an advanced field at the intersection of natural language processing and computer vision, integrating text and images to perform fine-grained sentiment analysis on specific aspects, enhancing the understanding of user opinions in various applications. As shown in Figure 1, MABSA involves extracting all aspect terms from image-text pairs and predicting their sentiment polarities. It includes two subtasks: multimodal aspect term extraction (MATE) (Wu et al. 2020a), which identifies all aspect terms in the sentence prompted by the associated image; and multimodal aspect sentiment classification (MASC) (Yu and Jiang 2019),

Our next show ! We are raising money for 4 year old Slaters Stem Cell therapy ! Join us in Riverside , CA ! Sat May 16 .

<html><body><table><tr><td>Aspect term</td><td>Slaters</td><td>Stem Cell</td><td>Riverside</td><td>CA ********</td></tr><tr><td>Sentiment polarity</td><td>Positive</td><td>Negative</td><td>Neutral</td><td>Neutral</td></tr></table></body></html>

![](images/c088686abfd4f5132ed7563371f137d948eaccba7ac8703e0b70dc3bbec49491.jpg)  
GPT-4: The image shows a young boy with a joyful expression on his face. He has short, spiked hair and is wearing a red t-shirt. The boy is seated in a wheelchair, which is equipped with a headrest and straps that appear to be securing him safely in the chair. The background is nondescript, suggesting the photo may have been taken indoors, possibly in a room or hallway. The boy's smile and the twinkle in his eyes convey a sense of happiness and playfulness.

Figure 1: A dataset entry from Twitter2015 (Yu and Jiang 2019). MABSA contains two subtasks, MATE and MASC.

which determines the sentiment polarity of each aspect. MABSA requires the utilization of both image and text information, take Figure 1 as an example, where solely based on the text, people might assume the sentiment polarity of Slaters to be negative because he is ill and requires treatment. However, upon observing the image, it becomes clear that the young boy is smiling. In this case, we can infer that the sentiment polarity of Slaters is positive.

Image and text, as two distinct modalities, exhibit significant differences in the types of information they convey and the forms in which they are expressed, each carrying heterogeneous information. Some methods focus on fusing these two modalities, while others advocate for modality alignment as the main means to effectively integrate the heterogeneous information, thereby improving performance. Ling, Yu, and Xia (2022) introduce three task-specific pre-training tasks to identify fine-grained aspect, opinions, and their cross-modal alignments. Zhou et al. (2023) introduce aspectaware attention module $( \mathbb { A } ^ { 3 } \mathbb { M } )$ for semantically fine-grained image-text alignment. Furthermore, to achieve better alignment, many cutting-edge methodologies adopt a pre-training followed by fine-tuning approach. However, despite the advancements in modality alignment, the inherent gap between image and text modalities still exists, necessitating a more direct bridging mechanism to effectively connect image understanding with text content. What is more, the pre-training process consumes a significant amount of time and computational resources.

For this, we propose the Descriptions Enhanced Question-Answering Framework (DEQA), which generates descriptions of images using GPT-4, leveraging the multimodal large language model to provide more direct semantic context of images. Unlike previous methods, our method does not require pre-training, significantly reducing the demand for computational resources. Furthermore, we find that the descriptions not only contain purely visual information but also incorporate related knowledge, commonsense, and appropriate inferences made by GPT-4 based on the image content. As illustrated in Figure 1, in the description, the underlined portions are supplementary knowledge and inferences made by GPT-4, and some of them can aid our model in making predictions. Terms joyful and convey a sense of happiness and playfulness suggest that the sentiment polarity of the aspect term Slaters is likely to be positive.

We design two sub-models to separately handle the MATE and MASC tasks, and then sequentially connect the inputs and outputs of these two sub-models to complete the MABSA task. Each sub-model contains three experts, with each dedicated to processing one of the following scenarios: text-only input, text and image input, and text and descriptions input, allowing each expert to specialize in different features. This allows us to comprehensively utilize these three types of information. Furthermore, to help the model better understand the task’s purpose and context, we frame MABSA as a multi-turn question-answering problem to add semantic guidance and hints. By integrating expert outputs within this question-answering format, we employ multiexpert ensemble decision-making approaches to produce the final prediction results. Experimental results on two widelyused datasets demonstrate that our method achieves stateof-the-art performance. What is more, our framework substantially outperforms GPT-4o and other multimodal large language models, showcasing its superior effectiveness in multimodal sentiment analysis.

In summary, our contributions are as follows:

• We introduce the DEQA for MABSA. This framework leverages the capabilities of GPT-4 to generate descriptions from images. DEQA integrates queries, transforming MABSA into the structured question-answering format. We input text, image, and description into separate experts in various combinations, allowing each expert to focus on different features. By integrating these expert outputs within the multi-turn question-answering format, we employ multi-expert ensemble decision-making approaches to produce the final prediction results.

• Our method demonstrates state-of-the-art performance on two widely-used datasets, namely, Twitter2015 and Twitter2017 (Yu and Jiang 2019).

• We evaluate $\mathtt { g p t - 4 o - 2 0 2 4 - 0 5 - 1 3 }$ on the MABSA task, and compare our model with it and other commonly used multimodal large language models. The results demonstrate that DEQA significantly outperforms GPT-4o as well as other multimodal large language models.

# Related Work

Given the prevalence of multimodal data on the Internet, information from the visual modality can be utilized to provide complementary sentiment signals to text features (Zhang, Wang, and Liu 2018). Thus, MABSA and its subtasks are widely studied. Next, we will introduce some classic and cutting-edge methods that also serve as the baselines for comparison.

# Aspect-Based Sentiment Analysis

Unlike MABSA, aspect-based sentiment analysis (Pontiki et al. 2014) relies solely on the text modality. Existing methods primarily focus on capturing the structural information within the text, aiming to extract richer semantic details from its structure and relationships. Hu et al. (2019) propose a span-based extract-then-classify framework and denote it as SPAN, Chen, Tian, and Song (2020) propose directional graph convolutional networks (D-GCN), and Yan et al. (2021) exploit the pre-training sequence-to-sequence model BART (Lewis et al. 2019) to solve all aspect-based sentiment analysis subtasks in an end-to-end framework. We compare these three text-based methods with our multimodal approach on MABSA to highlight the advantages of “multimodal” and our method.

# Multimodal Aspect Term Extraction

MATE aims to extract all aspect terms from the given textimage pair. To address this task, Wu et al. (2020b) propose a region-aware alignment network (RAN). OCSGA (Wu et al. 2020c) and UMT (Yu et al. 2020) are originally proposed for multimodal named entity recognition (Moon, Neves, and Carvalho 2018). However, due to the high degree of similarity between multimodal named entity recognition and the MATE task, these methods can also be effectively adapted for MATE.

# Multimodal Aspect Sentiment Classification

Given an aspect term, MASC aims to identify the corresponding sentiment polarity from the text-image pair. Given that the aspect term is already known, some existing methods focus on aligning the specific aspect term with the image to selectively extract the image region information relevant to the aspect term. Yu, Jiang, and Xia (2020) propose an entity-sensitive attention and fusion network (ESAFN), Yu and Jiang (2019) propose the target-oriented multimodal BERT (TomBERT), and Khan and Fu (2021) introduce a two-stream model that translates images in input space and leverages this translation to construct an auxiliary sentence that provides multimodal information.

# Multimodal Aspect-based Sentiment Analysis

MABSA aims to extract all aspect terms from the image-text pair and predict their sentiment polarities. Ju et al. (2021) carries out the MABSA task by combining, transferring, and modifying existing models that are not originally designed for MABSA. Specifically, Ju et al. (2021) implement two pipeline approaches upon two representative studies of MATE and MASC and three collapsed tagging approaches. 1) UMT $+$ TomBERT. 2) OCSGA+TomBERT. 3) UMT-collapsed. 4) OCSGA-collapsed. 5) RpBERT.

The following five models are specifically designed for MABSA and all its subtasks. Ju et al. (2021) are the first to jointly perform MATE and MASC, proposing a multi-modal joint learning approach, namely JML. Ling, Yu, and Xia (2022) propose a task-specific vision-language pre-training framework for MABSA (VLP-MABSA). Yang, Na, and Yu (2022) propose a multi-task learning framework named cross-modal multitask transformer (CMMT). Zhou et al. (2023) propose an aspect-oriented method (AoM) to detect aspect-relevant semantic and sentiment information. Peng et al. (2024) propose a framework called DQPSA, which contains a prompt as dual query module and an energy-based pairwise expert module.

# Methodology

# Framework Overview

Given a tweet that contains an image $I$ and a sentence $S$ , DEQA aims to identify the set of pairs $\{ ( a _ { 1 } , s _ { 1 } ) , ( a _ { 2 } , s _ { 2 } ) , \ldots , ( a _ { i } , s _ { i } ) , \ldots \}$ . Here, $( a _ { i } , s _ { i } )$ represents a pair of (aspect term, sentiment polarity), and $s _ { i }$ belongs to the set $\{ p o s i t i v e , n e u t r a l , n e g a t i v e \}$ .

As shown in Figure 2, to formalize MABSA as multipleinstance question-answering tasks, we construct three types of queries for a pair, including an aspect extraction query $Q ^ { e }$ , an aspect validation query $Q ^ { v }$ , and a sentiment classification query $Q ^ { c }$ . Concretely, at first, the aspect extraction query $Q ^ { e }$ aims to extract the aspect term $a _ { i }$ from the sentence $S$ . Then, given the aspect term $a _ { i }$ , the aspect validation query $Q ^ { v }$ is designed to validate the accuracy of $a _ { i }$ . Finally, the sentiment classification query $Q ^ { c }$ aims to predict the sentiment polarity $s _ { i }$ for $a _ { i }$ . Additionally, by expressing questions in natural language, semantic hints are provided to the model, aiding in a better understanding of the tasks’ objectives.

It is worth noting that, in Figure 2, we introduce two special tokens, <target> and </target>, to mark $a _ { i }$ , emphasizing and differentiating this aspect term. We find that some sentences contain multiple instances of the same aspect term, and marking them helps to avoid ambiguity. For instance, in the Twitter2015 dataset, there is a sentence, $R T$ @SoSingaporean: What people from other countries do at IKEA VS What I do at IKEA #sosingaporean, where IKEA appears twice.

Our model contains two sub-models, one for MATE, and the other for MASC. In each sub-model, there are three experts: the text-only expert, the text and description expert, and the text and vision expert, each responsible for processing different combinations of modality inputs. Additionally, each sub-model includes a decision ensemble that integrates the outputs from the various experts to determine the final prediction. Finally, the output of the sub-model for MATE serves as the input for the sub-model for MASC, and by combining the outputs of these two sub-models, the final prediction for the MABSA task is obtained.

# Sub-model for MATE

The sub-model starts with the text-only aspect extraction expert, which utilizes the aspect extraction query $Q ^ { e }$ to extract the aspect term $a _ { i }$ using a pre-trained language model combined with the BIO tagging scheme (Huang, Xu, and Yu 2015) and CRF (Lafferty et al. 2001). Following this, the text and description aspect validation expert incorporates both text and description to validate the extracted aspect terms with the help of the aspect validation query $Q ^ { v }$ . Meanwhile, the text and vision aspect validation expert utilizes $Q ^ { v }$ and integrates visual information with text to confirm the correctness of aspect terms. Finally, the aspect extraction decision ensemble combines predictions from two validation experts, filtering the extracted aspect terms to determine final predictions.

Text-Only Aspect Extraction Expert We construct the input as shown in Figure 2, and feed it into DeBERTa (He et al. 2021) to obtain a representation for each token. These token representations are then passed through a fully connected layer to produce $\mathbf { X }$ , which is used for predicting the labels according to the BIO tagging scheme. Subsequently, we use CRF to compute the conditional probability of the label sequence y (the gold labels) given $\mathbf { X }$ :

$$
P \left( \mathbf { y } \mid \mathbf { X } \right) = { \frac { \exp \left( \operatorname { s c o r e } \left( \mathbf { X } , \mathbf { y } \right) \right) } { \sum _ { \mathbf { y } ^ { \prime } \in { \mathcal { y } } } \exp \left( \operatorname { s c o r e } \left( \mathbf { X } , \mathbf { y } ^ { \prime } \right) \right) } }
$$

where $y$ represents the set of all possible label sequences, and score $( \cdot )$ is the score function (Lafferty et al. 2001) that assigns a score to the label sequence $\mathbf { y }$ based on X. The loss function is the negative log-likelihood of this conditional probability:

$$
{ \mathcal { L } } _ { \mathrm { t } } ^ { a } = - \log P \left( \mathbf { y } \mid \mathbf { X } \right)
$$

Text and Description Aspect Validation Expert We construct the input as shown in Figure 2, and feed it into DeBERTa to obtain the representation for each token. To ensure that the model effectively utilizes the functionality and role of the added special tokens, we deviate from the common practice of selecting the ${ < s > }$ token. Instead, we select the representation of the first <target $>$ token and pass it through a fully connected layer to predict whether the aspect term is correct. The output of the fully connected layer is then applied to a softmax function (Bridle 1989) to obtain the predicted probabilities $\mathbf { P } _ { \mathrm { d } } ^ { a }$ . Finally, we compute the cross-entropy loss (Rumelhart, Hinton, and Williams 1986):

$$
\mathcal { L } _ { \mathrm { d } } ^ { a } = - \sum _ { i = 1 } ^ { N } y _ { i } \log ( \hat { y } _ { i } )
$$

where $N$ is the number of classes (i.e., correct or incorrect), $y _ { i }$ is the true label for class $i$ , and $\hat { y } _ { i }$ is the predicted probability for class $i$ .

Text and Vision Aspect Validation Expert As shown in Figure 2, we feed the aspect validation query $Q ^ { v }$ into the CLIP (Radford et al. 2021) text encoder to obtain the representation $\mathbf { t }$ of the entire text. We feed the image $I$ into the CLIP vision encoder to obtain the representation

𝑆: Our next show ! We are raising money for 4 year old <s> 𝑆 </s </s> <s> 𝑄௩ </s> 𝐷 </s> 𝑄௩ Sub-model for MATE   
SlaatteMrasy S1t6em. Cell therapy ! Join us in Riverside , CA ! 口 Text-Only Aspect Text and Description Aspect Text and Vision Aspect   
𝐷: The image shows a young boy with a joyful expression Extraction Expert Validation Expert Validation Expert   
on his face. He has short, spiked hair and is wearing a   
red t-shirt. The boy is seated in a wheelchair, which (money，Slaters) Pୢ௔ P୴௔   
is equipped with a headrest and straps that appear to   
be securing him safely in the chair. The background is   
nondescript, suggesting the photo may have been taken Aspect Extraction Decision Ensemble   
iandootrhse, pwoisnskilbelyi inh as reoyoems ocro hvaelylwaays.e Tshee bfoyh'asp simnielses ↓   
and playfulness. (money ，Slaters ) ↓   
𝐼: <s> </s> 𝑄௖ Sub-model for MASC   
$Q ^ { e }$ : What aspect terms? Text-Only Sentiment Text and Description Sentiment Text and Vision Sentiment Classification Expert Classification Expert Classification Expert   
$Q ^ { c }$ : What is the sentiment polarity of the <target> $a _ { i }$ 几   
</target> in the sentence “𝑆”? P୲௖ Pୢ௖ P୴௖   
$Q ^ { v }$ : Is <target> $a _ { i }$ </target> the aspect term of the sentence Sentiment Classification Decision Ensemble   
“𝑆”? ↓ (Slaters，positive)

$\mathbf { V } = [ \mathbf { p } _ { 1 } , \mathbf { p } _ { 2 } , \ldots , \mathbf { p } _ { n } ]$ for patch sequence. We employ an attention mechanism (Vaswani et al. 2017) to achieve crossmodal alignment between the text and image patches, using t as the query and $\mathbf { p } _ { i }$ as the key and value. This approach ultimately yields an aligned representation of the question and the image:

$$
\mathbf { g } = \mathbf { C } \mathbf { A } \left( \mathbf { t } , \mathbf { V } \right)
$$

where $\mathrm { C A } \left( \cdot \right)$ denotes multi-head cross attention (Chen, Fan, and Panda 2021). Unlike conventional methods, we utilize multimodal factorized bilinear pooling (MFB) (Yu et al. 2017) as the attention scoring function. Subsequently, we convert the original text representation:

$$
\mathbf { t } ^ { \prime } = \mathrm { R e L U } \left( \mathbf { W } \cdot \mathrm { D r o p o u t } \left( \mathbf { t } , 0 . 5 \right) + \mathbf { b } \right)
$$

where $\mathbf { W }$ is the weight matrix, $\mathbf { b }$ is the bias vector, and Dropout $( \mathbf { t } , 0 . 5 )$ applies dropout (Srivastava et al. 2014) with a probability of 0.5 to the original text representation. The function ReLU $( \cdot )$ is the rectified linear unit activation function (Krizhevsky, Sutskever, and Hinton 2012). We then employ MFB again to fuse $\mathbf { t } ^ { \prime }$ and $\mathbf { g }$ , obtaining the final fused representation:

$$
\mathbf { f } = \mathbf { M F B } \left( \mathbf { t } ^ { \prime } , \mathbf { g } \right)
$$

We pass f through a fully connected layer to predict whether the corresponding aspect term is correct. The output of the fully connected layer is then passed through a softmax function to obtain the predicted probabilities $\mathbf { P } _ { \mathrm { ~ v ~ } } ^ { a }$ . Finally, we derive the cross-entropy loss ${ \mathcal { L } } _ { \mathrm { v } } ^ { a }$ .

Aspect Extraction Decision Ensemble We combine $\mathbf { P } _ { \mathrm { d } } ^ { a }$ with $\mathbf { P } _ { \mathrm { ~ v ~ } } ^ { a }$ , by element-wise addition and normalization to obtain the final probability distribution for each label:

$$
\mathbf { P } ^ { a } = \frac { \mathbf { P } _ { \mathrm { d } } ^ { a } + \mathbf { P } _ { \mathrm { v } } ^ { a } } { \Vert \mathbf { P } _ { \mathrm { d } } ^ { a } + \mathbf { P } _ { \mathrm { v } } ^ { a } \Vert _ { 1 } }
$$

Then, we use $\mathbf { P } ^ { a }$ to validate the extracted aspect terms.

To jointly train each expert in MATE sub-model and make them mutually beneficial, we sum the loss functions of the different experts to form the overall loss objective of the MATE sub-model:

$$
\mathscr { L } ^ { a } = \mathcal { L } _ { \mathrm { t } } ^ { a } + \mathcal { L } _ { \mathrm { d } } ^ { a } + \mathcal { L } _ { \mathrm { v } } ^ { a }
$$

# Sub-model for MASC

This sub-model predicts the sentiment polarity for each aspect term $a _ { i }$ determined by the sub-model for MATE. The text-only sentiment classification expert uses a pre-trained language model to predict the sentiment polarity. Meanwhile, the text and description sentiment classification expert refines sentiment polarity predictions by incorporating text and description, and the text and vision sentiment classification expert integrates visual data with text to make predictions. Finally, the sentiment classification decision ensemble aggregates outputs from the three experts to determine the final sentiment polarity for each aspect term.

Text-Only Sentiment Classification Expert We construct the input as shown in Figure 2, and feed it into DeBERTa to obtain the representation for each token. Then, we use the first <target> token to predict the sentiment polarity, and obtain the predicted probabilities $\mathbf { P } _ { \mathrm { ~ t ~ } } ^ { c }$ . Finally, we derive the cross-entropy loss $\mathcal { L } _ { \mathrm { t } } ^ { c }$ .

Table 1: Results of different methods for MATE. $F _ { 1 }$ denotes Micro-F1. † denotes the results from (Ju et al. 2021). The best results are bold-typed and the second best ones are underlined.   

<html><body><table><tr><td rowspan="2">Methods</td><td colspan="3">Twitter2015</td><td colspan="2">Twitter2017</td></tr><tr><td>P</td><td>R F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>RAN+</td><td>80.5</td><td>81.5 81.0</td><td>90.7</td><td>90.0</td><td>90.3</td></tr><tr><td>UMT+</td><td>77.8</td><td>81.7 79.7</td><td>86.7</td><td>86.8</td><td>86.7</td></tr><tr><td>OCSGA+</td><td>81.7</td><td>82.1</td><td>81.9</td><td>90.2 90.7</td><td>90.4</td></tr><tr><td>JML</td><td>83.6</td><td>81.2</td><td>82.4</td><td>92.0 90.7</td><td>91.4</td></tr><tr><td>VLP-MABSA</td><td>83.6</td><td>87.9</td><td>85.7</td><td>90.8 92.6</td><td>91.7</td></tr><tr><td>CMMT</td><td>83.9</td><td>88.1</td><td>85.9</td><td>92.2 93.9</td><td>93.1</td></tr><tr><td>AoM</td><td>84.6</td><td>87.9</td><td>86.2</td><td>91.8 92.8</td><td>92.3</td></tr><tr><td>DQPSA</td><td>88.3</td><td>87.1</td><td>87.7</td><td>95.1 93.5</td><td>94.3</td></tr><tr><td>DEQA</td><td>86.6</td><td>89.5</td><td>88.0</td><td>93.8 95.1</td><td>94.4</td></tr></table></body></html>

Text and Description Sentiment Classification Expert We construct the input as shown in Figure 2, and feed it into DeBERTa to obtain the representation for each token. Then, we use the first <target $>$ token to predict the sentiment polarity, and obtain the predicted probabilities $\mathbf { P } _ { \mathrm { d } } ^ { c }$ . Finally, we derive the cross-entropy loss $\mathcal { L } _ { \mathrm { d } } ^ { c }$ .

Text and Vision Sentiment Classification Expert As shown in Figure 2, we feed the sentiment classification query $Q ^ { c }$ and the image $I$ into CLIP. Subsequently, we adopt the same approach as used in the text and vision aspect validation expert to obtain $\mathbf { P } _ { \mathrm { ~ v ~ } } ^ { c }$ . Finally, we derive the cross-entropy loss $\mathcal { L } _ { \mathrm { v } } ^ { c }$ .

Sentiment Classification Decision Ensemble Similar to the aspect extraction decision ensemble, we obtain the final probability distribution:

$$
\mathbf { P } ^ { c } = \frac { \mathbf { P } _ { \mathrm { d } } ^ { c } + \mathbf { P } _ { \mathrm { v } } ^ { c } + \mathbf { P } _ { \mathrm { t } } ^ { c } } { \| \mathbf { P } _ { \mathrm { d } } ^ { c } + \mathbf { P } _ { \mathrm { v } } ^ { c } + \mathbf { P } _ { \mathrm { t } } ^ { c } \| _ { 1 } }
$$

Unlike the training approach used for the sub-model for MATE, we have trained each MASC expert individually. Specifically, we employ three separate losses— $\cdot \mathcal { L } _ { \mathrm { t } } ^ { c }$ , $\mathcal { L } _ { \mathrm { d } } ^ { c }$ and $\mathcal { L } _ { \mathrm { v } } ^ { c }$ —to guide the training. In the sub-model for MATE, three experts have a strong sequential and logical relationship among them, which indicates significant mutual influence. Therefore, joint training can effectively account for the interactions among these experts. Conversely, in the submodel for MASC, the relationships among the three experts are relatively independent. Thus, training each expert separately and performing individual hyperparameter tuning for each one are more beneficial.

# Experiments

# Datasets

Following previous studies, we use two widely adopted benchmarks: Twitter2015 and Twitter $2 0 1 7 ^ { 1 }$ to evaluate DEQA.

# Implementation Details

We use gpt-4-vision-preview to generate descriptions of the images. Our code and data2 provide implementation details, including pre-trained models, training details, and training durations across two datasets.

We train our model using an NVIDIA RTX A6000 GPU and implement an early stopping strategy (Prechelt 1998) with a patience of 3 epochs and a threshold of 0.01 to prevent overfitting. The AdamW optimizer (Loshchilov and Hutter 2019) is utilized for training, with a weight decay (Krogh and Hertz 1991) of 0.01. Additionally, we employ a linear learning rate scheduler with a warmup ratio (He et al. 2015) of 0.1 to adjust the learning rate throughout the training process.

In the MASC sub-model, we first train the text-only sentiment classification expert. Subsequently, the fine-tuned weights of this expert are used to initialize the text and description sentiment classification expert. For the two text and vision experts, the factor dimension of MFB is set to 1 for attention scoring and 8 for fusion. During the training process, we freeze the CLIP image encoder.

# Evaluation Metrics

In continuation of prior studies, we assess the performance of our model on the MATE and MABSA tasks using Precision $( P )$ , Recall $( R )$ , and Micro-F1 $( F _ { 1 } )$ scores. For the MASC task, we report both Accuracy $( A c c )$ and Macro-F1 $( F _ { 1 } )$ scores.

# Results

Performance on MATE Table 1 presents the comparative results for MATE. Our method surpasses the secondbest model on both Twitter2015 and Twitter2017 datasets. Although our method offers only a slight advantage over DQPSA, it is important to note that DQPSA relies on computationally intensive pre-training.

Performance on MASC Table 2 presents the comparative results for MASC. Our method surpasses the secondbest models on Twitter2015. However, the performance of DEQA on the Twitter2017 dataset is not particularly outstanding. In terms of F1 score, our method exceeds AoM by only $0 . 1 \%$ ; meanwhile, in terms of accuracy, it falls behind AoM by $0 . 6 \%$ . Peng et al. (2024) point out that Twitter2017 contains a significant number of unresolvable and unidentifiable symbols, including emojis commonly used on Twitter, which are unknown to the DeBERTa model we used. Given this, we believe our proposed method remains effective for MASC.

Performance on MABSA Table 3 presents the comparative results for MABSA. Our method surpasses the second-best model on both Twitter2015 and Twitter2017 datasets. This demonstrates that our model has successfully achieved state-of-the-art performance. Furthermore, compared to text-based models, DEQA exhibits significantly better performance. Additionally, we construct an end-to-end version of DEQA for comparison. Specifically, we use deberta-v3-large for the text modality and clip-vit-large-patch14-336 for the image modality, with cross-attention as the fusion method.

Table 2: Results of different methods for MASC. $F _ { 1 }$ denotes Macro-F1. † denotes the use of Micro-F1 as the evaluation metric. The best results are bold-typed and the second best ones are underlined.   

<html><body><table><tr><td rowspan="2">Methods</td><td colspan="2">Twitter2015</td><td colspan="2">Twitter2017</td></tr><tr><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td></tr><tr><td>TomBERT</td><td>77.2</td><td>71.8</td><td>70.5</td><td>68.0</td></tr><tr><td>ESAFN</td><td>73.4</td><td>67.4</td><td>67.8</td><td>64.2</td></tr><tr><td>CapTrBERT</td><td>78.0</td><td>73.2</td><td>72.3</td><td>70.2</td></tr><tr><td>JML</td><td>78.7</td><td></td><td>72.7</td><td></td></tr><tr><td>VLP-MABSA</td><td>78.6</td><td>73.8</td><td>73.8</td><td>71.8</td></tr><tr><td>CMMT</td><td>77.9</td><td></td><td>73.8</td><td>=</td></tr><tr><td>AoM</td><td>80.2</td><td>75.9</td><td>76.4</td><td>75.0</td></tr><tr><td>DQPSA</td><td>81.1</td><td>81.1†</td><td>75.0</td><td>75.0+</td></tr><tr><td>DEQA</td><td>82.1</td><td>77.6</td><td>75.8</td><td>75.1</td></tr></table></body></html>

# Ablation Study

W/o Aspect Validation Query After removing aspect validation query $Q ^ { v }$ , as shown in Table 4, a decline in the F1 score is observed for the MATE task across both datasets. Specifically, for the Twitter2015 dataset, precision decreases by $1 . 4 \%$ , while recall increases by $0 . 2 \%$ . The increase in recall indicates more correct aspect terms being identified, specifically an increase in true positives and a decrease in false negatives. However, despite the increase in true positives, the precision shows a larger decrease, suggesting an increase in false positives. Based on the above reasoning, we can conclude that, on the one hand, $Q ^ { v }$ may lead to fewer correct aspect terms being identified, indicating that $Q ^ { v }$ might misclassify some correctly identified aspect terms as incorrect. On the other hand, $Q ^ { v }$ can result in a reduction in false positives, implying that $Q ^ { v }$ effectively filters out some incorrect predictions. Overall, when removing $Q ^ { v }$ , the larger decrease in precision compared to the smaller increase in recall suggests that the misclassification of correct predictions by $Q ^ { v }$ is less significant than its role in filtering out incorrect predictions. For the Twitter2017 dataset, both precision and recall decrease, but the drop in precision $( 1 . 5 \% )$ is greater than the drop in recall $( 0 . 4 \% )$ , showing a similar trend to that observed in the Twitter2015 dataset.

W/o Semantic Hints We remove semantic hints from all queries. Specifically, we completely remove $Q ^ { e }$ . We transform $Q ^ { v }$ into <target> $a _ { i }$ </target> of the sentence “ $S ^ { \prime \prime }$ , and $Q ^ { c }$ into <target> $a _ { i }$ </target $>$ in the sentence “ $S ^ { \prime \prime 3 }$ . As shown in Table 4, the model’s performance on the MATE task experience a slight decline, while its performance on the MASC and MABSA tasks see a noticeable drop. This suggests that semantic hints are important for MABSA and MASC but have a relatively minor effect on the MATE task.

Table 3: Results of different methods for the MABSA task. $F _ { 1 }$ denotes Micro-F1. † denotes the results from (Ju et al. 2021); ‡ denotes the results from (Ling, Yu, and Xia 2022). The best results are bold-typed and the second best ones are underlined; the best results for text-based methods are italicized.   

<html><body><table><tr><td rowspan="2">Methods</td><td colspan="2">Twitter2015</td><td colspan="3">Twitter2017</td></tr><tr><td>P</td><td>R F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>SPAN+ D-GCN+</td><td>53.7 58.3 62.9</td><td>53.9 53.8 58.8 59.4 65.0 63.9</td><td>59.6 64.2 65.2</td><td>61.7 64.1 65.6</td><td>60.6 64.1 65.4</td></tr><tr><td>UMT+TomBERT† OCSGA+TomBERTt BART*</td><td>58.4 61.7</td><td>61.3 63.4</td><td>59.8 62.3 62.5 63.4</td><td>62.4 64.0</td><td>62.4 63.7</td></tr><tr><td>OCSGA-collapset UMT-collapse†</td><td>63.1 60.4</td><td>63.7 63.2 61.6 61.0</td><td>63.5 60.0</td><td>63.5 61.7</td><td>63.5 60.8</td></tr><tr><td>RpBERT†</td><td>49.3 46.9</td><td>48.0</td><td>57.0</td><td></td><td>56.2</td></tr><tr><td>JML</td><td>65.0</td><td></td><td></td><td>55.4</td><td></td></tr><tr><td>VLP-MABSA</td><td></td><td>63.2 64.1</td><td>66.5</td><td>65.5</td><td>66.0</td></tr><tr><td></td><td>65.1</td><td>68.3 66.6</td><td>66.9</td><td>69.2</td><td>68.0</td></tr><tr><td>CMMT</td><td>64.6</td><td>68.7 66.5</td><td>67.6</td><td>69.4</td><td>68.5</td></tr><tr><td>AoM</td><td>67.9 69.3</td><td>68.6</td><td>68.4</td><td>71.0</td><td>69.7</td></tr><tr><td>DQPSA</td><td>71.7 72.0</td><td>71.9</td><td>71.1</td><td>70.2</td><td>70.6</td></tr><tr><td>DEQA (end-to-end)</td><td>64.0 64.7</td><td>64.4</td><td>64.3</td><td></td><td>64.1</td></tr><tr><td></td><td></td><td></td><td></td><td>63.9</td><td></td></tr><tr><td>DEQA</td><td>71.4</td><td>73.9 72.7</td><td>71.4</td><td>72.4</td><td>71.9</td></tr></table></body></html>

W/o Special Tokens We remove the special tokens <target $>$ and </target $>$ from $S$ and replace those not in $S$ with quotation marks. Additionally, following common practice, we select the representation of the beginning token ${ < } s >$ . As shown in Table 4, the experimental results indicate that the special tokens we introduced play an important role.

W/o Description After removing descriptions, we observe a decline in model performance across all three tasks, as shown in Table 4. This indicates that descriptions are beneficial to MABSA.

W/o Vision We remove images, remaining descriptions. The experimental results, as shown in Table 4, indicate that removing images leads to declines in model performance. This suggests that descriptions cannot fully substitute for the images. While descriptions provide accurate and comprehensive summaries of the image content, they still fail to capture all the information that images offer, particularly detailed information. We observe that the performance degradation caused by the absence of descriptions is generally greater than that caused by the absence of visual modality. This indicates that descriptions are more important than images.

Sentiment Classification Decision Ensemble We evaluate several alternative decision methods. However, none outperforms the Sentiment Classification Decision Ensemble. Table 7 are results comparing different strategies on MASC.

<html><body><table><tr><td rowspan="2">Tasks</td><td rowspan="2">Methods</td><td colspan="4">Twitter2015</td><td colspan="4">Twitter2017</td></tr><tr><td>Acc</td><td>P</td><td>R</td><td>F1</td><td>Acc</td><td>P</td><td>R</td><td>F1</td></tr><tr><td rowspan="6">MATE</td><td>DEQA</td><td></td><td>86.6</td><td>89.5</td><td>88.0</td><td></td><td>93.8</td><td>95.1</td><td>94.4</td></tr><tr><td>w/o Aspect validation query</td><td></td><td>85.2</td><td>89.7</td><td>87.4</td><td></td><td>92.3 94.7</td><td></td><td>93.4</td></tr><tr><td>w/o Semantic hints</td><td></td><td>86.9</td><td>88.2</td><td>87.6</td><td></td><td>94.2 94.7</td><td></td><td>94.4</td></tr><tr><td>w/o Special tokens</td><td></td><td>86.9</td><td>87.7</td><td>87.3</td><td></td><td>93.2</td><td>95.1</td><td>94.1</td></tr><tr><td>w/o Description</td><td></td><td>86.5</td><td>88.8</td><td>87.6</td><td></td><td>93.4</td><td>93.8</td><td>93.6</td></tr><tr><td>w/o Vision</td><td></td><td>87.1</td><td>88.0</td><td>87.5</td><td></td><td>93.8</td><td>94.2</td><td>94.0</td></tr><tr><td rowspan="5">MASC</td><td>DEQA</td><td>82.1</td><td></td><td></td><td>77.6</td><td>75.8</td><td></td><td></td><td>75.1</td></tr><tr><td>w/o Semantic hints</td><td>80.3</td><td></td><td></td><td>76.1</td><td>74.2</td><td></td><td></td><td>73.4</td></tr><tr><td>w/o Special tokens</td><td>79.1</td><td></td><td></td><td>74.4</td><td>74.1</td><td></td><td></td><td>73.1</td></tr><tr><td>w/o Description</td><td>79.8</td><td></td><td></td><td>74.4</td><td>72.2</td><td></td><td></td><td>71.0</td></tr><tr><td>w/o Vision</td><td>80.9</td><td></td><td></td><td>77.0</td><td>75.7</td><td></td><td>-</td><td>74.7</td></tr><tr><td rowspan="5">MABSA</td><td>DEQA</td><td></td><td>71.4</td><td>73.9</td><td>72.7</td><td></td><td>71.4</td><td>72.4</td><td>71.9</td></tr><tr><td>w/o Semantic hints</td><td></td><td>69.7</td><td>70.8</td><td>70.2</td><td></td><td>70.2</td><td>70.5</td><td>70.3</td></tr><tr><td>w/o Special tokens</td><td></td><td>69.7</td><td>70.4</td><td>70.1</td><td></td><td></td><td></td><td>69.170.6 69.8</td></tr><tr><td>w/o Description</td><td></td><td>68.9</td><td>70.8</td><td>69.9</td><td></td><td>68.0 68.3</td><td></td><td>68.2</td></tr><tr><td>w/o Vision</td><td></td><td>71.0</td><td>71.7</td><td>71.3</td><td></td><td>71.1</td><td>71.5</td><td>71.3</td></tr></table></body></html>

Table 4: Results of ablation study. For the MATE and MABSA tasks, $F _ { 1 }$ denotes Micro-F1, whereas for the MASC task, $F _ { 1 }$ denotes Macro-F1.

Table 5: Results of different (multimodal) large language models for the MASC task. We use Accuracy as the evaluation metric. All results are from (Yang et al. 2024). The best results are bold-typed and the second best ones are underlined.   

<html><body><table><tr><td>Large Models</td><td>Twitter2015</td><td>Twitter2017</td></tr><tr><td>ChatGPT-3.5 LLaMA2-13B</td><td>65.5 60.4</td><td>60.0 48.5</td></tr><tr><td>Mixtral-AWQ</td><td>55.5</td><td>60.2</td></tr><tr><td>GPT-4V</td><td>53.9</td><td>60.2</td></tr><tr><td>Claude3-V</td><td>38.5</td><td>54.5</td></tr><tr><td>Gemini-V</td><td>54.5</td><td>59.3</td></tr><tr><td>LLaVA-v1.6-13B</td><td>58.7</td><td>56.1</td></tr><tr><td>Fuyu-8B</td><td>58.8</td><td></td></tr><tr><td></td><td></td><td>50.8</td></tr><tr><td>Qwen-VL-Chat</td><td>65.5</td><td>59.7</td></tr><tr><td>DEQA</td><td>82.1</td><td>75.8</td></tr></table></body></html>

<html><body><table><tr><td rowspan="2">Models</td><td colspan="3">Twitter2015</td><td colspan="3">Twitter2017</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>DEQA gpt-4o-2024-05-13</td><td>23.7 71.4</td><td>28.5 73.9</td><td>25.9 72.7</td><td>25.2 71.4</td><td>26.3 72.4</td><td>25.7 71.9</td></tr></table></body></html>

Table 6: Results of different models for MABSA. $F _ { 1 }$ denotes Micro-F1.

# Performance Compared to Large Models

We compare DEQA on the MASC task with several (multimodal) large language models. The results are presented in Table 5. It can be observed that our DEQA significantly outperforms all the (multimodal) large language models, including GPT-4V, LLaMA2-13B (Touvron et al. 2023),

Table 7: Results of different decision methods for MASC. $F _ { 1 }$ denotes Macro-F1. MF refers to Maximization Fusion, MLR to Multi-response Linear Regression (Ting and Witten 1999), PoE to Product of Experts (Hinton 1990), MLP to Multilayer Perceptron fusion, PV to Plurality Voting, and SCDE to Sentiment Classification Decision Ensemble.   

<html><body><table><tr><td rowspan="2">Methods</td><td colspan="4">Twitter2015 Twitter2017</td></tr><tr><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td></tr><tr><td>MF</td><td>79.6</td><td>74.3</td><td>74.1</td><td>73.3</td></tr><tr><td>MLR</td><td>81.2</td><td>76.0</td><td>74.6</td><td>73.5</td></tr><tr><td>PoE</td><td>81.2</td><td>76.7</td><td>75.6</td><td>74.9</td></tr><tr><td>MLP</td><td>81.9</td><td>77.4</td><td>75.0</td><td>74.3</td></tr><tr><td>PV</td><td>82.1</td><td>77.6</td><td>75.8</td><td>74.8</td></tr><tr><td>SCDE</td><td>82.1</td><td>77.6</td><td>75.8</td><td>75.1</td></tr></table></body></html>

Mixtral-AWQ (Egiazarian et al. 2024) and Gemini-V (Qi et al. 2023). We also evaluate $\mathtt { g p t - 4 o - } 2 0 2 4 - 0 5 - 1 3$ on MABSA. The results are presented in Table 6. It can be observed that its performance is relatively poor compared to our method.

# Conclusion

In this paper, we propose DEQA for MABSA and its subtasks, addressing the challenge of bridging the gap between text and visual modalities without pre-training. In DEQA, we frame MABSA as a multi-turn question-answering problem, where text, image, and description are input into separate experts in various combinations. By integrating these expert outputs within a multi-expert ensemble decisionmaking approach, our method generates the final predictions, achieving state-of-the-art performance. Furthermore, our framework substantially outperforms GPT-4o and other (multimodal) large language models.