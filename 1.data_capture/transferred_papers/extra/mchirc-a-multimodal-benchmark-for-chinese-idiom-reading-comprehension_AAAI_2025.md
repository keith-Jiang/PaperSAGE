# MChIRC: A Multimodal Benchmark for Chinese Idiom Reading Comprehension

Tongguan Wang\*1, 2, 3, 4, Mingmin $\mathbf { W _ { u } } ^ { * 4 }$ , Guixin $\mathbf { S u } ^ { 4 }$ , Dongyu $\mathbf { S u } ^ { 4 }$ , Yuxue Hu1, 2, 3, 4, Zhongqiang Huang4, Ying Sha†1, 2, 3, 4

1Key Laboratory of Smart Farming for Agricultural Animals, Wuhan, China 2Engineering Research Center of Intelligent Technology for Agriculture, Ministry of Education, Wuhan, China 3Hubei Engineering Technology Research Center of Agricultural Big Data, Wuhan, China 4College of Informatics, Huazhong Agricultural University, Wuhan, China {wang tg, wmm nlp, cometsue, su dy, hzq} $@$ webmail.hzau.edu.cn, {hyx,shaying}@mail.hzau.edu.cn

# Abstract

The performance of various tasks of natural language processing has greatly improved with the emergence of large language modelSst.uHdoenwtever, there is still much room for improvement in understanDding certain specific linguistic phenomena, such as Chinese idioms, which are usually composed of four characters. Chinese idioms are difficult to understand due to semantic gaps between their literal and actual meanings. Researchers have proposed the Chinese idiom reading comprehension task to exAamine the ability of large language models to re＋present and understand Chinese idioms. The task requires choosing the correct Chinese idiom from a list of candidates to complete the sentence. The current research mainly focuses on text-based idiom comprehension. Nevertheless, there are many idiom application scenarios that combine images and text, and we believe that the corresponding images are beneficial for the model’s understanding of the idioms. Therefore, to address the above problems, we first construct a largescale Multimodal Chinese Idiom Reading Comprehension dataset (MChIRC), which contains a total of 44,433 imagetext pairs covering 2,926 idioms. Then, we propose a DualContrastive Idiom Graph Network (DCIGN), which employs a dual-contrastive learning module to align the text and image features corresponding to the same Chinese idiom at both coarse and fine levels, while utilizing a graph structure to capture the semantic relationships between idiom candidates. Finally, we use a cross-attention module to fuse multimodal features with graph features of candidate idioms to predict correct answers. The authoritativeness of MChIRC and the effe绝ct处ive逢ne生ss：of在D最C危IG险N的ar时e d候e得mo到ns生tra路ted through a variety of e(xRpesrciumeednftrso, mwhdiecshpepraotvioidn:esDaesncerwibebsenbcehinmgark for the multimgiovdean aCwhianyeoseutiadtiothme rmeoasdt ndgancgoermopursethiemnes)ion task.

# 到I生nt路roduction

Chinese idioms are a special form of linguistic expression, usually consisting of four Chinese characters, such as “绝 处逢生 (Rescued from desperation)”, as shown in Figure 1. The challenge in understanding Chinese idioms is the inconsistency between the literal meaning and the metaphorical meaning, which usually requires some background knowledge, such as history. Therefore, obtaining accurate representation and understanding of idioms is crucial for downstream tasks such as machine translation (Li et al. 2024), image generation (Ali et al. 2024), and image-text matching (Yosef, Bitton, and Shahaf 2023).

![](images/af29fdaf9553c6fa80467eeabf1890d059fb10a0bcd4aa76e33bec235c8a7dfb.jpg)  
Figure 1: An intriguing phenomenon is observed during the classroom instruction of idioms.

Therefore, Zheng, Huang, and Sun (2019) proposed a cloze task for idioms and established the widely used Chinese idiom completion dataset, ChID. The cloze task involving idioms typically requires the model to select the most suitable idioms from the set of candidate idioms to be placed in the blanks of the sentence, which requires the model to maximize the understanding of the special meanings of different idioms in different contexts.

The current methods are mainly based on deep learning models and have achieved certain performances. Tan and Jiang (2021) proposed a BERT-based dual embedding model and a two-stage model with context pooling and finetuning to improve the accuracy of idiom prediction. Wang et al. (2020) introduced an attribute focus mechanism to correct idiomatic misuse. Sha et al. (2023) proposed a promptbased representation individual enhancement approach to learn idiomatic metaphorical meaning. Wu et al. (2024a) addressed the problem of inconsistency between idiomatic metaphor and context through multi-semantic connectivity and contrastive learning, achieving state-of-the-art performance with their method in the ChID dataset. However, all these existing methods consider only textual features. The rapid development of social media has led to a diverse range of data sources for idioms. How to effectively utilize heterogeneous information such as images corresponding to idioms to assist the model in accurately understanding the id

<html><body><table><tr><td rowspan="2">Dataset</td><td rowspan="2">Sentence</td><td rowspan="2">Number of idioms</td><td rowspan="2">Average words</td><td colspan="2">TexModaImage</td></tr><tr><td></td><td></td></tr><tr><td>People Daily (Cui et al. 2016)</td><td>876,710</td><td>248,160</td><td>39</td><td>√</td><td>X</td></tr><tr><td>Children's Fairy Tale (Cui et al.2016)</td><td>3,599</td><td></td><td>20</td><td>√</td><td>X</td></tr><tr><td>CMRC-2017 (Cui et al. 2018)</td><td>364,295</td><td>94,352</td><td>17.4</td><td>√</td><td>X</td></tr><tr><td>CIBB (Shao et al. 2018)</td><td>1,194</td><td>50</td><td>9.81</td><td>√</td><td>X</td></tr><tr><td>CCT (Jiang et al. 2018)</td><td>108,987</td><td>7,395</td><td>27.37</td><td>√</td><td>X</td></tr><tr><td>ChID(Zheng,Huang,and Sun 2019)</td><td>580,807</td><td>3,848</td><td>100</td><td>√</td><td>X</td></tr><tr><td>ChIdSyn (Tanand Jiang 2021)</td><td>21,000</td><td>8,125</td><td>97.66</td><td>√</td><td>X</td></tr><tr><td>CIP (Qiang et al. 2023)</td><td>115,529</td><td>8,421</td><td>36.55</td><td>√</td><td>X</td></tr><tr><td>CIDT (Wu et al. 2024a)</td><td>3,600</td><td>3,848</td><td>22.12</td><td>√</td><td>X</td></tr><tr><td>MChIRC(Ours)</td><td>44,433</td><td>2,926</td><td>15.42</td><td>√</td><td>√</td></tr></table></body></html>

Table 1: Summary of datasets of Chinese idioms datasets.

ioms is an urgent issue to be considered.

In reality, images and words related to idioms often appear simultaneously, which facilitates people’s understanding of idioms. In a real teaching case as shown in Figure 1, if only words are used to describe the meaning of the idiom “绝处逢生” (Rescued from desperation), students may lack intuitive feelings to understand “what is the most dangerous situation?” and “how to get out?”. If a picture is added, and the picture shows “a tree growing on the edge of a cliff”, it will be easier for students to understand the meaning and application scenarios of this idiom. Accordingly, these students are more likely to get an “A” in the test (Wong et al. 2010; Pintado and Fajardo 2021; Zhang 2021). Inspired by this phenomenon, the Chinese idiom reading comprehension task must also consider multimodal scenarios. In addition, the context of Chinese idioms in existing datasets is generally longer, while the context of Chinese idioms in actual scenarios is often shorter.

To address the above issues, we build, to the best of our knowledge, the first multimodal Chinese idiom reading comprehension dataset (MChIRC1). We crawl numerous images from Baidu2 and Sogou3. After manual annotation, we collect 44,433 image-text pairs covering 2,926 idioms. The average length of sentences in the dataset is 15.42, which is more consistent with the context length of Chinese idioms in actual scenarios (Wang et al. 2021). The content of the MChIRC dataset covers multiple fields such as news, comics, animation, and product information.

Based on the constructed multimodal dataset, we propose a dual-contrastive idiom graph network (DCIGN), which employs a dual-contrastive learning module to align the text and image features corresponding to the same Chinese idiom at both coarse and fine levels, thus improving the model’s ability to recognize positive and negative idiom text-image pairs. In addition, the graph structure is utilized to learn the semantic relationship between the candidate idioms and further distinguish the nuances between different candidate idioms. Finally, we use a cross-attention module to fuse multimodal features with graph features of candidate idioms to predict correct answers. Extensive experiments conducted on the MChIRC dataset demonstrate the effectiveness of our proposed method, achieving an average accuracy of $73 \%$ in the four test sets.

The main contributions are as follows:

• We construct the first multimodal Chinese idiom dataset, MChIRC, which is more aligned with real-world scenarios. The average length of the sentences is only 15.42. • We propose a dual-contrastive idiom graph network, DCIGN, which employs a dual-contrastive learning module to align the text and image features corresponding to the same Chinese idiom at both coarse and fine levels, while a graph structure is utilized to learn the semantic relationships among idiom candidates. • Experimental results demonstrate that DCIGN achieves promising performance on the MChIRC dataset, which proves the rational utilization of visual information and provides a novel benchmark for Chinese idiom research.

# Related Work

# Datasets

As a unique linguistic phenomenon in Chinese, Chinese idioms have attracted numerous scholars to conduct extensive research. We summarize and analyze the Chinese idiom datasets in recent years, as presented in Table 1. The Chinese idiom dataset is currently primarily applied to four types of tasks, which are used for the Chinese idiom reading comprehension task (Cui et al. 2016, 2018; Jiang et al. 2018; Zheng, Huang, and Sun 2019; Wu et al. 2024a), the Chinese idiom translation task (Shao et al. 2018), the Chinese idiom embedding task (Qiang et al. 2023), and the Chinese idiom rewriting task (Tan and Jiang 2021). In summary, these Chinese idiom datasets have provided abundant resources for the research of Chinese idioms.

However, we find that the average words of sentences in these datasets are usually large, which usually means that they contain ample contextual information. In the real-world application of Chinese idioms, news headlines often use concise sentences that include idioms to attract attention, which means that context information for Chinese idioms is sparse, making the prediction of correct idioms more challenging (Knietaite et al. 2024). It is particularly noteworthy that these datasets only utilize textual information, neglecting the potential of image information to assist in task learning, as demonstrated by other multimodal datasets (Jin et al. 2017; Cai, Cai, and Wan 2019; Qi et al. 2023).

<html><body><table><tr><td>Text&Candidate</td><td>Image</td></tr><tr><td>#idiom#的树，生命真的是强大啊！ #idiom# trees,life is really powerful!</td><td rowspan="3"></td></tr><tr><td>NO.1否极泰来Adversity leads to prosperity</td></tr><tr><td>NO.2 引而不发 Draw the bow without shooting</td></tr><tr><td>NO.3 峰回路转 The twists and turns</td><td></td></tr><tr><td>NO.4 颠扑不破 Indisputable</td><td></td></tr><tr><td>NO.5 绝处逢生Rescued from desperation√</td><td></td></tr><tr><td>NO.6 指手划脚 Point and gesture with one's fingers</td><td rowspan="2">正确选项是NO.5 Thecorrect choice isNO.5</td></tr><tr><td>NO.7起死回生 Bring the dying back to life</td></tr></table></body></html>

# Chinese Idiom Reading Comprehension

The Chinese idiom comprehension task aims to make models understand the exact meaning of idioms. Cui et al. (2016, 2018) used the complete representation of the queries and the “Overly Attentive Reader” to solve reading comprehension tasks. Jiang et al. (2018) used background knowledge to improve the accuracy of the cloze task. Zheng, Huang, and Sun (2019) proposed three baselines on the ChID dataset: LM, AR, and SAR. Subsequent studies, such as those using attention mechanisms by Long et al. (2020) and Wang et al. (2020), a two-stage model by Tan and Jiang (2021), and interpretative approaches by Dai et al. (2023) and Sha et al. (2023), have significantly improved machine understanding of Chinese idioms. In addition, Wu et al. (2024a) addressed the problem of inconsistency between idiom literal and metaphorical meanings through multi-semantic contrastive learning, achieving state-of-the-art performance on the ChID dataset with an accuracy of $9 6 . 8 \%$ on the test set.

Nevertheless, to the best of our knowledge, there is a gap in the study of multimodal Chinese idioms. To address this, we construct MChIRC, a more relevant and challenging multimodal Chinese idioms dataset tailored for real-world scenarios, with an average sentence length of only 15.42 words. We also propose a dual-contrastive idiom graph network, DCIGN, which integrates both text and image features for the first time to tackle the Chinese idiom reading comprehension task. Our method provides a novel perspective for research on Chinese idioms.

# Task Definition

The objective of the multimodal Chinese idiom cloze task is to select the most appropriate idiom from a set of seven candidates, based on a given text segment and an accompanying image. The text is denoted as $T = \{ w _ { 1 } , w _ { 2 } , . . \nonumber$ ., $\mathsf { \bar { [ } } M \mathsf { \bar { A } } \bar { S K } ] , . . . , \mathsf { w } _ { i } , { w } _ { n } \}$ , where each character $w _ { i }$ represents a Chinese character, and the position of the idiom is marked with $[ M A S K ]$ . The image is denoted as $I = \{ i m a g e \}$ . The candidate idiom set is denoted as $C = \{ c _ { 1 } , . . . , c _ { i } , . . . , c _ { k } \}$ , comprises six distractor idioms and one correct idiom. We present a data sample in Figure 2. The text reads “#idiom#的 树, 生命真的是强大啊! (#idiom# trees, life is really powerful!)”. The image visually depicts a tree growing on cliffs. Therefore, in this example, the correct option for “#idiom#” can be identified as “NO.5 绝处逢生 (NO.5 Rescued from desperation)”, which perfectly captures the essence of the text and image.

# DCIGN Model

# Model Overview

We propose a dual-contrastive idiom graph network (DCIGN) for multimodal Chinese idiom reading. As shown in Figure 3, DCIGN consists of five main components: Unimodal Feature Extraction, Fine-Grained Feature Extraction, Dual-Contrastive Learning from Coarse to Fine, Graph Relationship Modeling for Idiom Candidates, and a Predict Module.

# Unimodal Feature Extraction

The input text for the network is denoted as $T = \{ w _ { 1 } , w _ { 2 }$ , ..., $[ M A S K ]$ , ..., $\left. w _ { i } , w _ { n } \right\}$ , while the input image for the network is a corresponding image of the idiom, denoted as $I = \{ i m a g e \}$ . To obtain the most primitive features from different modalities, similar to most multimodal approaches (Huang et al. 2023), we utilize a pre-trained BERT4 to extract textual features and a pre-trained DeiT5 to extract image features. The formulations are detailed as follows:

$$
\begin{array} { r } { T _ { o } = B E R T ( [ C L S ] , w _ { 1 } , w _ { 2 } , \dots , [ M A S K ] , } \\ { \dots , w _ { i } , w _ { n } , [ S E P ] ) , } \end{array}
$$

$$
V _ { o } = D e i T ( I ) ,
$$

where $T _ { o }$ represents the extracted primitive text features. $V _ { o }$ represents the extracted primitive image features. The special marks $[ C L S ]$ and $[ S E P ]$ are boundary marks used to guide and terminate the input. The $[ M A S K ]$ refers to idioms that are masked out.

# Fine-grained Feature Extraction

Due to the presence of redundant or interfering information in the originally extracted features, the model’s ability to understand text or image information is affected to some extent. Therefore, we use self-attention to refine the key information in the extracted primitive features and ignore the information that is not important for understanding idioms, respectively. The formulations are detailed as follows:

$$
\begin{array} { l } { { \tilde { T } _ { o } = L N \left( S A \left( Q _ { T _ { o } } , K _ { T _ { o } } , V _ { T _ { o } } \right) + T _ { o } \right) , } } \\ { { \tilde { V } _ { o } = L N \left( S A \left( Q _ { V _ { o } } , K _ { V _ { o } } , V _ { V _ { o } } \right) + V _ { o } \right) , } } \end{array}
$$

where $\tilde { T } _ { o }$ represents refined text features. $\tilde { V } _ { o }$ represents refined image features. $L N$ stands for Layer Normalization, $S A$ stands for Self-Attention.

Then, we fuse the extracted refined features with the extracted original features to obtain the final multimodal features $F _ { f u s e }$ as follows:

$$
F _ { f u s e } = T _ { o } + \tilde { T } _ { o } + V _ { o } + \tilde { V } _ { o } .
$$

![](images/dbcbccd7d8770c187cd5c7262a980a23dfce7a4076d3e9f4c7cef7a81c7072e2.jpg)  
Figure 3: Dual-Contrastive Idiom Graph Network for Multimodal Chinese Idiom Reading.

# Dual-Contrastive Learning from Coarse to Fine

Contrastive learning has been achieved with impressive results across tasks (Zhang et al. 2024; Su et al. 2024; Hu et al. 2024). The primitive and refined features focus on different regions in the text and images, potentially causing finegrained errors despite coarse-grained alignment. Therefore, to improve the alignment of the model at different levels of modal features, we use a dual-contrastive learning module to align multimodal features.

Firstly, we use coarse contrastive learning (CCL) to align the primitive text and image features. Coarse contrastive learning helps the model align features that are roughly different. Then, to further refine and optimize the feature representation and enhance the model’s ability to distinguish subtle semantic differences, we use fine contrastive learning (FCL) to process the finely extracted text and image features. Based on coarse contrastive learning, fine contrastive learning further adjusts and optimizes the layout of the feature space, so that the text-image features of the same idiom are more precisely aligned, while the text-image features of different idioms are more clearly distinguished.

$$
\mathcal { L } _ { c c l } = - \sum _ { i = 1 } ^ { N } l o g \frac { e ^ { s i m \left( T _ { 0 } ^ { i } , V _ { 0 } ^ { i } \right) / \tau } } { \sum _ { j = 1 } ^ { N } e ^ { s i m \left( T _ { 0 } ^ { i } , V _ { 0 } ^ { j } \right) / \tau } } ,
$$

$$
\mathcal { L } _ { f c l } = - \sum _ { i = 1 } ^ { N } l o g \frac { e ^ { s i m \left( \tilde { T } _ { 0 } ^ { i } , \tilde { V } _ { 0 } ^ { i } \right) / \tau } } { \sum _ { j = 1 } ^ { N } e ^ { \sin \left( \tilde { T } _ { 0 } ^ { i } , \tilde { V } _ { 0 } ^ { j } \right) / \tau } } ,
$$

where $N$ represents the batch size, $( T _ { 0 } ^ { i } , V _ { 0 } ^ { i } )$ stands for the i-th sample comes from two different modal features. sim is the cosine similarity. $\tau$ is the temperature hyper-parameter.

Coarse contrastive learning and fine contrastive learning collaborate with each other, which not only aligns the text image features of the same idiom at two levels but also enhances the model’s sensitivity to small variations, enabling it to make more accurate and fine-grained judgments when faced with complex and subtle multimodal idiom reading comprehension cloze tasks.

# Graph Relationship Modeling for Idiom

We also use pre-trained BERT to extract the features of the idioms in the candidate set. The formulations are as follows:

$$
C _ { k } = B E R T ( I d i o m _ { k } ) ,
$$

where $k$ stands for the $\mathbf { k }$ -th idiom, $k \in [ 1 , 7 ]$ .

After obtaining the feature representation of each candidate idiom $C _ { k }$ , we concatenate these features to get the feature representation $C$ of the candidate idiom.

$$
C = C o n c a t ( C _ { 1 } , C _ { 2 } , \dots , C _ { k } ) .
$$

The idioms among the candidates exhibit semantic associations. Taking the idiom candidate set in the test set as an example, it comprises three idioms most similar to the correct option and three idioms randomly selected. We consider each candidate idiom as a node and the relationship between them as edges. The representation of candidate idioms is not solely determined by their individual characteristics, it can also aggregate information from adjacent idiom nodes through a graph structure, thus obtaining a richer and more distinctive representation. Therefore, we introduce a graph structure to capture the semantic representations between idioms. The feature update process for each node $c$ at layer $l$ is as follows:

$$
h _ { c } ^ { ( l + 1 ) } = R e L U \left( \sum _ { u \in \mathcal { N } ( c ) } \frac { 1 } { p _ { u c } } W ^ { ( l ) } h _ { u } ^ { ( l ) } \right) ,
$$

$\textcircled{1}$ Data collection ④ Keywords: 水滴石穿 (Constant dripping wears away a stone) ££ 3 77 Final data Text Image Supplement (image.baidu.com) (pic.sogou.com) (pic.sogou.com)   
② ③ + 山 & + (image.Mbaidnu.com) Data cleaning Data screening Data supplement

where $h _ { u } ^ { ( l ) }$ represents the feature vector of the neighbors $u$ of idiom node $c$ at layer l. $\mathcal { N } ( c )$ denotes the set of neighboring idiom nodes of idiom node $c$ . $p _ { u c }$ is the normalization coefficient, which is used to control the influence of the features of neighboring idiom nodes on the features of the current idiom node. $W ^ { ( l ) }$ is the weight matrix at layer $l$ .

After the graph convolution layer, we concatenate the representations of each node to obtain the overall representation of the candidate idiom, denoted by $h _ { v }$ .

# Prediction Module

In order to capture the intention and context of the query more accurately. We use fused multimodal features as query $Q$ and candidate idioms that have been through the graph structure as key $K$ , value $V$ . The two features are fused using the cross-attention module, and the final representation of the fusion feature $\tilde { M }$ is obtained.

$$
\tilde { M } = C A ( Q _ { F _ { f u s e } } , K _ { h _ { v } } , V _ { h _ { v } } ) .
$$

We feed the fusion feature representation $\tilde { M }$ into a linear layer, which then undergoes a softmax function for probabilistic prediction of the seven idioms in the candidate set.

$$
P ( I d i o m _ { k } | T , I ) = S o f t m a x ( W \tilde { M } + b ) ,
$$

where $I d i o m _ { k }$ stands for the $\mathbf { k }$ -th idiom in the candidate set.

We minimize the cross-entropy loss function to calculate the difference between the predicted probability distribution and the true distribution for the correct idiom, as follows:

$$
\mathcal { L } _ { c e } = - \sum _ { k = 1 } ^ { 7 } C _ { g } l o g ( P ( I d i o m _ { k } | T , I ) ) ,
$$

where $C _ { g }$ stands for the one-hot label distribution for the correct idiom.

Using joint dual-contrastive learning loss, we end up with a total loss as shown as follows:

$$
\mathcal { L } = \mathcal { L } _ { c e } + \alpha \mathcal { L } _ { c c l } + \beta \mathcal { L } _ { f c l } ,
$$

where $\alpha + \beta = 1$ , $\alpha$ and $\beta$ are the weights of the coarse contrastive learning loss and the fine contrastive learning loss, respectively.

<html><body><table><tr><td></td><td>Train</td><td>Dev</td><td>Test/Sim</td><td>Out</td><td>Total</td></tr><tr><td>Idiom</td><td>2,926</td><td>2,456</td><td>2,881</td><td>2,926</td><td>2,926</td></tr><tr><td>Pair</td><td>29,054</td><td>4,151</td><td>8,302</td><td>2,926</td><td>44,433</td></tr></table></body></html>

Table 2: Division of the MChIRC dataset. The data in the Test set and the Sim set are identical except for the different composition of the set of candidate idioms.

# Experiments

# Datasets

We present an overview of the construction process for the MChIRC dataset, as illustrated in Figure 4. In general terms, the process is primarily divided into four stages: data collection, data cleaning, data screening, and data supplement. All data are used for scientific research only. We promise that the data collected will not be used for non-commercial or profit-making purposes and strictly respect this dataset’s copyright.

We evaluate all methods using the MChIRC multimodal idiom dataset. Initially, we extract one sample from each idiom as the Out set, which is the most challenging. The remaining samples are divided into a training, validation, and test set in a 7: 1: 2 ratio using stratified sampling to ensure as much balance as possible between interclass and intraclass data. To further test the robustness of the models, we construct the Sim set following Zheng, Huang, and Sun (2019). The set of candidate idioms for the Sim set consists of the six idioms that are semantically most similar to the correct idiom. The final dividend dataset is shown in Table 2.

# Baselines

We compare six text modality methods, three image modality methods, and four classic multimodal methods as baselines for comparison. These methods are described in detail below.

• BERT-WWM (Cui et al. 2021): An upgraded version of the BERT model that uses whole word masking.   
• RoBERTa (Cui et al. 2021): An improved BERT employs dynamic masking, more pre-training data, and extended training time.   
• MacBERT (Cui et al. 2021): An improved version of BERT, specifically designed for Chinese, introduces the pre-training task of MLM as correction (Mac).   
• PRIEM (Sha et al. 2023): Fusing the definitions of idioms through the prompt method and then using orthogonal projection to distinguish idioms’ representations.   
• RISCF (Wu et al. 2024b): The accuracy of idiom completion tests has been enhanced through semantic contrast learning and an anti-interference cross-attention module.   
• MSCLM (Wu et al. 2024a): The model addresses the issues of metaphorical inconsistency and contextual inconsistency in idioms. By using metaphor contrastive learning and multi-semantic cross-attention modules.   
• VGG-16 (Simonyan and Zisserman 2015): Enhancing model representation by stacking multiple smaller convolutional and pooling layers.   
• CLIP (Radford et al. 2021): A multimodal pretraining model designed to combine text and image information for robust cross-modal understanding and generation.   
• DeiT (Touvron et al. 2021): Utilizing distillation to train small models by transferring knowledge from large pretrained models, reducing model complexity and computational cost.   
• MSCA (Huang et al. 2023): A multimodal stack crossattention network for better alignment and fusion of multimodal token-level text and visual features for the multimodal fake news detection task.   
• Multi-view CLIP (Qin et al. 2023): A framework that is capable of leveraging multi-grained cues from multiple perspectives for the multimodal sarcasm detection task.   
• DivE (Kim, Kim, and Kwak 2023): A cross-modal retrieval approach utilizing smoothed chamfer similarity and using ensemble prediction modules.   
• CLTL (Wang and Markov 2024): A method that won first place for hate speech target detection in the Multimodal Hate Speech Event Detection Challenge 2024.

<html><body><table><tr><td>Modality</td><td>Method</td><td>Dev-Acc</td><td>Test-Acc</td><td>Sim-Acc</td><td>Out-Acc</td><td>Avg-Acc</td></tr><tr><td rowspan="6">Text</td><td>BERT-WWM (Cui et al. 2021)</td><td>63.43</td><td>60.40</td><td>59.50</td><td>49.79</td><td>58.28</td></tr><tr><td>RoBERTa (Cui et al. 2021)</td><td>64.49</td><td>61.85</td><td>60.50</td><td>50.72</td><td>59.39</td></tr><tr><td></td><td></td><td></td><td></td><td>50.3</td><td>59.05</td></tr><tr><td>mPRBERT (Cauietal.2021)</td><td>64.27</td><td>64.4</td><td>60.30</td><td></td><td></td></tr><tr><td>RISCF(Wu et al. 2024b)</td><td>67.02</td><td>66.12</td><td>66.65</td><td>46.82</td><td>61.65</td></tr><tr><td>MSCLM (Wu et al. 2024a)</td><td>71.19</td><td>69.56</td><td>67.89</td><td>51.95</td><td>65.15</td></tr><tr><td rowspan="3">Image</td><td>VGG-16</td><td>45.70</td><td>44.05</td><td>45.27</td><td>33.36</td><td>42.10</td></tr><tr><td>CLIP (Radford et al. 2021)</td><td>40.66</td><td>38.63</td><td>42.57</td><td>26.35</td><td>37.05</td></tr><tr><td>DeiT (Touvron et al. 2021)</td><td>55.46</td><td>54.01</td><td>54.81</td><td>40.64</td><td>51.23</td></tr><tr><td rowspan="5">Multimodal</td><td>MSCAt (Huang et al. 2023)</td><td>68.63</td><td>69.15</td><td>68.01</td><td>55.04</td><td>65.21</td></tr><tr><td>Multi-view CLIPt (Qin et al. 2023)</td><td>59.05</td><td>61.05</td><td>51.04</td><td>49.24</td><td>55.10</td></tr><tr><td>DivEt (Kim,Kim,and Kwak 2023)</td><td>74.24</td><td>71.56</td><td>66.71</td><td>56.03</td><td>67.13</td></tr><tr><td>CLTL† (Wang and Markov 2024)</td><td>65.55</td><td>64.05</td><td>60.24</td><td>50.75</td><td>60.15</td></tr><tr><td>DCIGN (Ours)</td><td>77.26</td><td>77.16</td><td>77.34</td><td>60.25</td><td>73.00</td></tr></table></body></html>

Table 3: Comparison results $( \% )$ with baseline models on the MChIRC dataset. “ ” stands for rerunning these methods.

# Evaluation Metrics

As in previous work, we also use accuracy as an evaluation metric for the model, i.e., the proportion of test samples in which the idioms selected by the model correspond to the correct idioms. Moreover, we add Avg-Acc, which represents the average accuracy of all previous test sets, to evaluate the overall performance of the model.

# Results and Analysis Comparison with Baselines

Table 3 presents the results of the comparison of our method with different baselines on the MChIRC dataset. Compared to unimodal Chinese idiom reading comprehension methods, DCIGN achieves the best accuracy on Dev, Test, Sim, and Out sets, with the highest average accuracy. In particular, our method surpasses the SOTA method by Wu et al. (2024a) in the ChID dataset. This not only indicates the challenging nature of the MChIRC dataset but also demonstrates the potential of image features to provide valuable information for the task of comprehension of Chinese idioms. Compared to the four methods used in other multimodal tasks, our method still performs optimally, demonstrating the unique applicability of our proposed DCIGN for multimodal Chinese idiom reading comprehension. At the same time, we find that the results of all methods on the Out set are less accurate than on the other test sets, which highlights the greater challenge posed by the Out set.

Table 4: Comparison results of the MLLMs with 500 samples in the Out set, “B” stands for billion.   

<html><body><table><tr><td>Method</td><td>Parameters</td><td>Acc</td><td>Team</td></tr><tr><td>GLM4V</td><td>9B</td><td>0.528</td><td>ZhipuAI</td></tr><tr><td>CogVLM2</td><td>8B</td><td>0.498</td><td>ZhipuAI</td></tr><tr><td>InternVL2-Pro</td><td>1</td><td>0.518</td><td>OpenGVLab</td></tr><tr><td>Qwen1.5</td><td>110B</td><td>0.526</td><td>Aliyun</td></tr><tr><td>GPT-40</td><td>175B</td><td>0.538</td><td>OpenAI</td></tr><tr><td>Ours</td><td></td><td>0.544</td><td></td></tr></table></body></html>

# Comparison with MLLMs

We compare five advanced multimodal large language models, selected from the top 10 on the OpenCompass6, $\mathrm { G L M } 4 \mathrm { V } ^ { 7 }$ (GLM et al. 2024), $\mathbf { \dot { C } } \mathbf { o g } \mathbf { V } \mathbf { L M } 2 ^ { 8 }$ , InternVL2-Pro9, Qwen1. $5 ^ { 1 0 }$ (Team 2024), GPT- $\bar { 4 0 } ^ { 1 1 }$ , respectively. For cost considerations, we only test 500 data from the Out set. The comparison results are shown in Table 4. It can be observed that compared to MLLMs, DCIGN still exhibits strong competitiveness. This indicates that (1) there is still room for improvement in multimodal large language models’ understanding of Chinese idioms in multimodal forms; (2) the model we proposed is applicable to the task of understanding multimodal Chinese idioms, providing a benchmark method for this task’s first introduction.

Table 5: Comparison of different component ablations on the MChIRC dataset. $T _ { S A }$ and $V _ { S A }$ represent the self-attention of text and image modalities, respectively.   

<html><body><table><tr><td>Method</td><td>Dev</td><td>Test</td><td>Sim</td><td>Out</td><td>Avg</td></tr><tr><td>w/o GCN</td><td>55.91</td><td>54.19</td><td>58.64</td><td>37.12</td><td>51.47</td></tr><tr><td>w/o Lccl</td><td>38.01</td><td>35.27</td><td>40.79</td><td>22.21</td><td>34.07</td></tr><tr><td>w/o Lfcl</td><td>70.25</td><td>69.42</td><td>72.08</td><td>51.95</td><td>65.93</td></tr><tr><td>w/o TsA</td><td>69.60</td><td>67.53</td><td>68.66</td><td>49.45</td><td>63.81</td></tr><tr><td>w/o VsA</td><td>69.24</td><td>68.08</td><td>69.89</td><td>49.15</td><td>64.09</td></tr><tr><td>w/o Lccl and L fcl</td><td>37.89</td><td>35.76</td><td>41.07</td><td>21.91</td><td>34.16</td></tr><tr><td>W/o TsA and Vs A</td><td>38.52</td><td>36.22</td><td>40.98</td><td>22.11</td><td>34.46</td></tr><tr><td>DCIGN (Ours)</td><td>77.26</td><td>77.16</td><td>77.34</td><td>60.25</td><td>73.00</td></tr></table></body></html>

![](images/3e3d0f00233b8eaa82a67203fe894af873a46b61c3d480596e118a2e716e7577.jpg)  
Figure 5: Visual comparison presentation after contrastive contrast learning and fine contrastive learning.

# Ablation Study

We conduct an ablation study on various components of DCIGN to validate the rationality of the network architecture. As shown in Table 5, when we individually remove $\mathcal { L } _ { f c l }$ , $T _ { S A }$ , and $V _ { S A }$ , the model’s accuracy experienced a slight decrease. However, when we individually remove GCN, the coarse contrastive learning loss $\mathcal { L } _ { c c l }$ , the dual-contrastive learning loss, and the dual-modality self-attention, the model’s accuracy significantly decreased. This demonstrates the effectiveness of each component of DCIGN working in concert to accomplish the task of reading comprehension for Chinese idioms.

# Hyperparameter Analysis

We perform a hyperparameter analysis of the $\alpha$ and $\beta$ before $\mathcal { L } _ { c c l }$ and $\mathcal { L } _ { f c l }$ , and the experimental results are shown in Table 6. We find that the average accuracy of the model is optimized when $\alpha { = } 0 . 4$ , $\beta { = } 0 . 6$ . We also find that the highest average accuracy is observed when $\alpha$ is higher and $\beta$ is lower in the combination of $\alpha$ and $\beta$ , e.g., $\scriptstyle \alpha = 0 . 9$ , $\beta { = } 0 . 1$ . On the contrary, when $\alpha$ is low and $\beta$ is high, e.g., $\alpha { = } 0 . 1$ , $\beta { = } 0 . 9$ , relatively low average accuracy is observed. This indicates that coarse contrastive learning loss accounts for a higher impact in the network optimization process, followed by fine contrastive learning loss.

Table 6: Hyperparameter analysis of $\alpha$ and $\beta$ .   

<html><body><table><tr><td>α,β</td><td>Dev</td><td>Test Sim</td><td>Out</td><td>Avg</td></tr><tr><td>α=0.1,β=0.9</td><td>75.50 75.32</td><td>76.11</td><td>58.13</td><td>71.27</td></tr><tr><td>α=0.2,β=0.8</td><td>77.40 76.16</td><td>76.74</td><td>60.15</td><td>72.61</td></tr><tr><td>α=0.3,β=0.7</td><td>75.26 74.60</td><td>75.48</td><td>56.87</td><td>70.55</td></tr><tr><td>Q=0.4, β=0.6</td><td>77.26 77.16</td><td>77.34</td><td>60.25</td><td>73.00</td></tr><tr><td>α=0.5, β=0.5</td><td>75.60 74.28</td><td>75.33</td><td>57.96</td><td>70.79</td></tr><tr><td>α=0.6, β=0.4</td><td>76.49 76.04</td><td>77.01</td><td>58.58</td><td>72.12</td></tr><tr><td>α=0.7,β=0.3</td><td>75.79 74.01</td><td>74.96</td><td>58.37</td><td>70.78</td></tr><tr><td>α=0.8,β=0.2</td><td>75.48 74.39</td><td>75.48</td><td>57.48</td><td>70.71</td></tr><tr><td>α=0.9, β=0.1</td><td>78.27 76.39</td><td>76.61</td><td>60.18</td><td>72.86</td></tr></table></body></html>

# Visualization

We conduct a visual comparison of text and images before and after applying coarse contrastive learning and refined contrastive learning, as shown in Figure 5. Before selfattention, coarse contrastive learning could only roughly align the text with the corresponding content in the image, as shown on the left side of Figure 5, where only the “trees” in red is aligned with the root of the tree in the image. After self-attention, fine contrastive learning allows for more precise alignment of text to image content. This refinement allows the visual presentation to focus on specific details, such as on the right side of Figure 5, where the textual analysis emphasizes the keywords “tree”, “life”, and “strength”, which are aligned to the roots of the tree, thus effectively conveying the image of a tree thriving on the edge of a cliff. The combination of these textual and visual features reflects the indomitable vitality subtly, thus encapsulating more profoundly the essence of the idiom “Rescued from desperation”. It can be concluded that we perform dual-contrastive learning on the features of text and images both before and after self-attention, thereby further enhancing the model’s ability to semantically associate text-image pairs.

# Conclusion

The current Chinese idiom reading comprehension methods focus mainly on text. In real-world scenarios, more multimodal forms occur simultaneously and the text length is shorter. Therefore, to address the above problems, we first construct a large-scale multimodal Chinese idiom reading comprehension dataset, MChIRC. Then, we propose a dualcontrastive idiom graph network, DCIGN. Extensive experimental results show that DCIGN can better understand the meaning of Chinese idioms with the help of their corresponding images. We believe that DCIGN, as a new benchmark for Chinese idiom reading comprehension, can provide a fresh perspective for future research.

# Ethical Statement

The list of Chinese idioms used in the MChIRC dataset is from Zheng, Huang, and Sun (2019). All text-image pairs are sourced from two platforms Baidu and Sogou. We guarantee that all data are used for scientific research only. We promise that the data collected will not be used for noncommercial or profit-making purposes and strictly respect this dataset’s copyright.