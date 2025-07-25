# Leveraging the Dual Capabilities of LLM: LLM-Enhanced Text Mapping Model for Personality Detection

Weihong $\mathbf { B i } ^ { 1 , 2 * }$ , Feifei ${ \bf K o u } ^ { 1 , 2 * \dagger }$ , Lei $\mathbf { S h i } ^ { 3 , 4 }$ , Yawen $\mathbf { L i } ^ { 5 }$ , Haisheng $\mathbf { L i } ^ { 6 }$ , Jinpeng Chen1,7, Mingying $\mathbf { X } \mathbf { u } ^ { 8 }$

1School of Computer Science (National Pilot School of Software Engineering), BUPT, Beijing, 100876, China 2 Key Laboratory of Trustworthy Distributed Computing and Service, BUPT, Ministry of Education, Beijing, 100876, China 3State Key Laboratory of Media Convergence and Communication, CUC, Beijing, 100024, China 4State Key Laboratory of Intelligent Game, Yangtze River Delta Research Institute of NPU, Taicang 215400, China 5School of Economics and Management, BUPT, Beijing, 100876, China 6Beijing Technology and Business University, Beijing, 100048, China 7Xiangjiang Laboratory, Changsha, 410205, China 8North China University of Technology, Beijing, 100144, China bwh2023140721 $@$ bupt.edu.cn; koufeife $\operatorname { i 0 0 0 } \ @$ bupt.edu.cn; leiky shi $@$ cuc.edu.cn; warmly0716@126.com; lihsh $@$ btbu.edu.cn; jpchen $@$ bupt.edu.cn; xumingying $@$ ncut.edu.cn

# Abstract

Personality detection aims to deduce a user’s personality from their published posts. The goal of this task is to map posts to specific personality types. Existing methods encode post information to obtain user vectors, which are then mapped to personality labels. However, existing methods face two main issues: first, only using small models makes it hard to accurately extract semantic features from multiple long documents. Second, the relationship between user vectors and personality labels is not fully considered. To address the issue of poor user representation, we utilize the text embedding capabilities of LLM. To solve the problem of insufficient consideration of the relationship between user vectors and personality labels, we leverage the text generation capabilities of LLM. Therefore, we propose the LLM-Enhanced Text Mapping Model (ETM) for Personality Detection. The model applies LLM’s text embedding capability to enhance user vector representations. Additionally, it uses LLM’s text generation capability to create multi-perspective interpretations of the labels, which are then used within a contrastive learning framework to strengthen the mapping of these vectors to personality labels. Experimental results show that our model achieves state-of-the-art performance on benchmark datasets.

Code — https://github.com/BUPT-SN/ETM (Myers 1987). User posts on social media can be used to detect their MBTI personality type. Therefore, the goal of this task is to map posts to specific personality types, as shown in Figure 1(a). Using deep learning methods to solve personality detection task is generally effective. This approach uses a data-driven method to extract text features from posts, fuses them into a user vector, and then maps the user vector to personality labels, as depicted in Figure 1(b). The previous methods (Keh, Cheng et al. 2019; Jiang, Zhang, and Choi 2020) use BERT (Devlin et al. 2018) to encode each individual post or to encode a concatenated sequence of multiple posts from a user. Some alternative methods (Yang et al. 2021b, 2023a; Zhu et al. 2022) use BERT to obtain semantic features and then apply graph neural networks to enhance the representation of individual posts, followed by average pooling to create a user vector. Other methods (Yang et al. 2023a; Zhu et al. 2022; Zhang et al. 2023) manually extract psychological and statistical features, combine them with text features into a user vector. Previous methods face two main issues. On one hand, these methods use BERT to extract text features, but this small model encoder cannot effectively represent multiple long posts. On the other hand, they fail to capture the relationship between user representation vectors and personality labels. These issues collectively lead to poor performance.

# Introduction

Personality plays a crucial role in understanding the relationship between individual behaviors and mental activities (Kernberg 2016). It can also serve as a valuable tool for guiding personal growth and career choices. The MBTI is a widely used system for classifying personality, dividing individuals into sixteen categories based on four dimensions

Recently, large language models (LLM) have demonstrated powerful text understanding and generation capabilities, surpassing small models in tasks like translation and question answering (Wang et al. 2023), but perform poorly on classification tasks. Some research (Yang et al. 2023b) has designed prompts based on psychological questionnaires to leverage the capabilities of LLM for personality detection. This approach still underperforms compared to finetuning small models, showing that using only LLM with prompts might not be an effective method for personality detection. Some studies (Hu et al. 2024) have used LLM for data augmentation from an emotional perspective, but the results show that this approach does not significantly im

Mapping →{E/I,N/S,F/T,P/J} <personalitylabels> (a)Personality detection task IMapping   
面 encoder- classify {E/I,N/S,F/T,P/J} <personalitylabels> <uservector>   
(b) Traditional deep learning methods for personality detection [Mapping classify i<personality labels> encoderadd {E/I,N/S,F/T,P/J} embedding extraction <uservector> enhance   
o LLM'sembeddingcapabilities LLM's generation capabilities (c) Our method

prove performance, indicating that the potential of LLM in personality detection tasks has not been fully realized. Previous research (Freestone and Santu 2024) shows that its distinct pre-training tasks enable it to cluster semantically related words more effectively than traditional encoding models. Therefore, we considered using LLM’s distinct text embedding capabilities, along with its widely adopted text generation abilities, to enhance the mapping process from user posts to personality labels, as illustrated in figure 1(c).

In this paper, we introduce a LLM-Enhanced Text Mapping Model for Personality Detection. We use a small model to encode individual posts and apply average pooling to create an initial user representation. These posts are then merged into one long post, and a lightweight language model generates its embedded representation. To better integrate the semantics of the two models, we add a crossattention mechanism to integrate the user and long post representations. We also use a powerful language model’s text generation capabilities to analyze personality labels from three aspects: personality definition, thematic tendency, and text expression. Additionally, we employ contrastive learning to enhance the correlation between documents and labels. The contributions can be summarized as follows:

• We propose an LLM-Enhanced Text Mapping Model for Personality Detection that effectively achieves the goal of accurately mapping posts to specific personality types.   
• Our model effectively uses the text embedding capability of LLM to enhance the mapping process of multiple long posts to user vectors. The text generation capability of LLM is used to enhance the mapping process of user vectors to personality labels.   
• According to the test results on the benchmark dataset, our model outperforms other existing personality detection models.

# Related Work Traditional and Deep Learning Methods in Personality Detection

For the personality detection task, both traditional methods and deep learning methods aim to better represent the user vector, and then map the user vector to the MBTI labels. Traditional methods extract static features from texts. Machine learning models such as Support Vector Machines (Cui and Qi 2017) and XGBoost (Tadesse et al. 2018) were then used to fit the mapping of posts to MBTI personalities. Traditional machine learning methods rely too much on hand-extracted features, leading to subpar performance. Deep learning methods initially use some feature extraction models, such as LSTM, hierarchical DNNs with AttRCNN, and GRUs with attention mechanisms (Tandera et al. 2017; Xue et al. 2018; Lynn, Balasubramanian, and Schwartz 2020) used a data-driven approach to model the mapping of posts to MBTI personality. Some pre-trained models such as BERT (Devlin et al. 2018) show unique advantages in text feature extraction tasks. Some studies (Keh, Cheng et al. 2019; Jiang, Zhang, and Choi 2020)use text pretraining models to extract features from posts and map the user vectors to MBTI labels. Other studies aim to enhance the representation of individual posts by constructing relationships between posts, thereby improving the overall user vector representation. Transformer MD (Yang et al. 2021a) leverages Transformer XL’s memory to store posts and uses attention mechanisms to capture and fuse relationships between them. Some studies (Yang et al. 2021b; Zhu et al. 2022; Yang et al. 2023a) use psychological statistical feature similarity or post embedding similarity to measure whether there is correlation between posts, and then use graph neural networks to build the topology of post sets and carry out feature fusion between posts. Deep learning methods that rely only on small model semantic encoders to extract semantic features from multiple long texts often produce low-quality user vectors, and the relationship between these vectors and MBTI labels is not fully established.

# Large Language Models in Personality Detection

Large Language Models (LLM) are used to solve some natural language tasks due to their powerful in-context learning(ICL) ability and extensive knowledge reserve (Brown et al. 2020; Rae et al. 2021; Thoppilan et al. 2022; Chowdhery et al. 2023; Achiam et al. 2023). Some new methods use LLM to help solve personality detection tasks. One way to use LLM to solve complex problems is to break down a complex problem into several small problems and then guide LLM to solve these problems in turn (Wei et al. 2022). Wei et al. (Yang et al. 2023b) adapted a psychological questionnaire into multiple questions, directing language models to answer based on post content and then drawing conclusions from the answers obtained in each round. The method using language models for personality detection relies heavily on prompt construction, leading to suboptimal results. Using LLM for data augmentation from an emotional perspective is also a way to leverage LLM. Hu et al. (Hu et al. 2024)used LLM to enhance posts from three perspectives, semantic, semantic and semantic, and then used comparative learning to establish the relationship between enhanced posts and initial posts. This method only uses the text generation capability of LLM, resulting in little improvement of the mapping process of user posts to personality labels in the process of this method.

# Approach

In this section, we first define the problem of personality detection, then conduct preliminary explorations using the method of fine-tuning lightweight LLM, and finally provide detailed descriptions of the two enhancement modules in our ETM model.

# Problem Definition

The personality detection task involves mapping posts to the personality types. Each posts, denoted as $\begin{array} { r l } { P } & { { } = } \end{array}$ $\{ p _ { 1 } , p _ { 2 } , \ldots , p _ { n } \}$ , consists of $M$ tokens per post, represented as ${ p _ { i } } = [ w _ { i 1 } , w _ { i 2 } , \dots , w _ { i M } ]$ . The personality types is formalized as $Y = ( y _ { 1 } , y _ { 2 } , y _ { 3 } , y _ { 4 } )$ , with each component $\mathbf { \mu } _ { y _ { i } }$ taking a value of 0 or 1. This task aims to establish a mapping from $P$ to $Y$ .

# Exploring Lightweight Large Language Models for Personality Detection

Lightweight Large Language Models retain the essential capabilities of traditional language models while being optimized for task-specific training (Yang et al. 2023c). In our experiments, we initially explored the use of lightweight LLM for personality detection through zero-shot learning, binary classification fine-tuning, and sixteen-class finetuning. first, we establish a baseline by using a lightweight LLM with zero-shot learning. Then, we explore two approaches: fine-tuning for sixteen-class classification and fine-tuning for binary classification. Although these methods outperform direct zero-shot learning, they still do not reach the baseline performance achieved with fine-tuned small models. The experimental results can be seen in the “Performance of Lightweight LLM” section of the Experiments. The extensive pre-training of lightweight large language models on varied datasets has often reduced their effectiveness when fine-tuning for specific text classification tasks. In response, we explored the use of LLMs to address the challenges of poor user vector representation and the inadequate relationship between user vectors and personality labels typically encountered with small models in personality detection. This strategic application of LLM allowed our approach to ultimately achieve state-of-the-art performance.

# An Overview of Our ETM Model Architecture

In this paper, we propose an LLM-Enhanced Text Mapping Model for Personality Detection. As the overall architecture shows in Figure 2. We first use BERT to encode each post individually, then apply average pooling to generate the initial user vectors. These posts are then combined into a single text. To enhance its text embedding, we apply a lightweight large language model, capturing the text representation from the model’s unique perspective. This vector is subsequently refined using a cross-attention mechanism to enhance its representational accuracy. Then, we leverage the text generation capabilities of powerful LLM to interpret MBTI labels from three dimensions: personality definition, thematic tendency, and text expression. This interpretation is encoded using a small model. Finally, a contrastive learning framework strengthens the mapping from user vectors to MBTI labels.

# Lightweight LLM Enhance User Posts to Vector Representation

We followed the previous approach (Keh, Cheng et al. 2019; Hu et al. 2024) to get the initial representation of the user vector, and then used the lightweight LLM to get a new perspective on the user representation. Then, using the crossattention mechanism, the user representation from a new perspective is used to strengthen the initial representation, so as to obtain an enhanced user vector representation.

For a post $p _ { i }$ in a post set $P$ , we use the BERT model as a text encoder to encode $p _ { i }$ , and then use the token at the [CLS] position as the feature representation of the $p _ { i }$ post to denote $\boldsymbol { h } _ { i }$ .

$$
h _ { i } = \operatorname { E n c o d e r } ( p _ { i } )
$$

Multiple long documents generated by a user, totaling $N$ , are encoded separately, and then an initial user vector $u$ is obtained by means of average pooling:

$$
\pmb { u } = \mathrm { m e a n } ( [ h _ { 1 } , h _ { 2 } , \dots , h _ { N } ] )
$$

To enhance the semantic representation beyond BERT, we leverage lightweight LLM to process concatenated user posts as extended contexts. Using the Llama3 model (Zhang et al. 2024), we extract embeddings from the concatenated texts:

$$
P _ { l o n g } = [ p _ { 1 } | p _ { 2 } | \dots | p _ { N } ]
$$

where $|$ denotes the concatenation operator.

Based on previous findings (Geva et al. 2020), which indicate that deeper insights are derived from upper layers of transformer-based models, we utilize the embeddings from the last $d _ { m }$ layer. To synthesize a comprehensive user vector $U _ { l l a m a }$ , we perform average pooling across and within these layers, capturing richer semantic details from the lightweight LLM.

$$
[ h _ { 0 } , h _ { 1 } , \dots , h _ { L } ] _ { n } = \operatorname { L l a m a } ( P _ { l o n g } , n )
$$

$$
U _ { n } = \operatorname* { m e a n } ( [ h _ { 0 } , h _ { 1 } , . . . , h _ { L } ] _ { n } )
$$

$U _ { l l a m a } = \mathrm { m e a n } ( [ U _ { 3 3 - d _ { m } } , U _ { 3 4 - d _ { m } } , \ldots , U _ { 3 2 } ] )$ (6) where $n$ denotes the $n$ -th layer and $\scriptstyle { \pmb { L } }$ represents the numbe of tokens after text embedding.

To enhance the initial user vector with additional insights from the lightweight LLM, we employed a cross-attention mechanism for effective vector fusion. To match the differing token dimensions of the BERT and Llama3 models, we utilized transformation matrices $W _ { o 1 }$ and $W _ { o 2 }$ . The vectors were then mapped to the $Q , \kappa$ , and $V$ spaces of the attention mechanism using three fully connected layers: $W _ { Q }$ , $W _ { K }$ , and $W _ { V }$ :

$$
\begin{array} { c } { Q = \boldsymbol { { u } } W _ { Q } } \\ { K = \left( U _ { \mathrm { { l l a m a } } } W _ { o 1 } \right) W _ { K } } \\ { V = \left( U _ { \mathrm { { l l a m a } } } W _ { o 2 } \right) W _ { V } } \end{array}
$$

The cross-attention mechanism generates an $\mathbf { \nabla } H _ { l l a m a }$ vector that incorporates insights from a new perspective. This vector is then fused with the initial user vector, resulting in

LLM Improve User Vector to MBTI contrastive learning Label Mapping Non-target labels Ist Ij.minimize agreement   
S Prompt ENFP 216 Text Encoder og Target labels lj+ , maximize agreement →Lcl   
Past split Post 11 Text Encoder Mean 000 WQ S 1 H Add OOOOHP concat WO   
.. 1 Mean 0 Ullama “ CrosAttention C Representation Llama3 Embeddings Frozen Params

the final enhanced user vector $H _ { p }$ . Here, $d _ { k }$ denotes the dimension of the $\pmb { K }$ space.

$$
\begin{array} { c } { { H _ { l l a m a } = \mathrm { s o f t m a x } \left( \displaystyle \frac { Q K } { \sqrt { d _ { k } } } \right) V } } \\ { { H _ { P } = H _ { l l a m a } + u } } \end{array}
$$

# Powerful LLM Improve User Vector to MBTI Label Mapping

MBTI labels represent specific personality traits, each with its inherent meaning (Myers 1987). By positioning the semantic encoding of user posts closer to their corresponding labels and farther from unrelated labels, we can improve the accuracy of subsequent classification tasks. Therefore, we plan to use the advanced model GPT-4 (Achiam et al. 2023) to interpret MBTI personality labels from three dimensions: type definition, thematic inclination, and mode of expression. Following this, we will employ the contrastive learning framework proposed by Chen (Chen et al. 2020) to establish the relationship between the semantics of the post collection and the label meaning.

To reduce potential hallucinations (Huang et al. 2023) by GPT-4, we provide it with official interpretations of the 16 MBTI personality labels as part of the prompt. Due to space constraints, we replace the full MBTI personality interpretations with personality authoritative interpretations and the specific social network context with description of the social media contex $\}$ in the prompt. The prompt is as follows:

The official interpretations of the 16 personality types is: {personality authoritative interpretations}. Based on the following social media context: description of the social media context}, generate discussion content interpretations for the 16 MBTI personality types on a social platform. Each interpretation should include the type’s definition, thematic tendencies, and typical modes of expression.

First, obtain detailed interpretations of the MBTI personality types, then encode these interpretations using a text encoder, and finally apply the max pooling method to maximize the retention of MBTI personality interpretations:

$$
[ h _ { j _ { 0 } } , h _ { j _ { 1 } } , \dots , h _ { j _ { L } } ] = \mathrm { E n c o d e r } ( l _ { j } )
$$

$$
\begin{array} { r } { l _ { j _ { + } } = \mathrm { M a x P o o l i n g } ( [ h _ { j _ { + 0 } } , h _ { j _ { + 1 } } \dots , h _ { j _ { + L } } ] ) } \\ { l _ { j _ { - } } = \mathrm { M a x P o o l i n g } ( [ h _ { j _ { - 0 } } , h _ { j _ { - 1 } } \dots , h _ { j _ { - L } } ] ) } \end{array}
$$

where $\iota _ { j _ { + } }$ corresponds to the label associated with the post collection, and $\ l _ { j _ { - } }$ corresponds to the remaining labels that do not pair with the post collection. Here, $L$ represents the number of tokens after text embedding.

The contrastive loss $L _ { c l }$ is defined as:

$$
L _ { c l } = - \log \frac { e ^ { \sin ( { \bf u } , l _ { j _ { + } } ) / \tau } } { e ^ { \sin ( { \bf u } , l _ { j _ { + } } ) / \tau } + \sum _ { j = 1 } ^ { 1 5 } e ^ { \sin ( { \bf u } , l _ { j _ { - } } ) / \tau } }
$$

where $\tau$ is a temperature parameter that adjusts the sensitivity of similarity scores.

# Joint Learning

We employ the focal loss function (Lin et al. 2017) to better target hard-to-classify samples and lessen focus on simpler ones. A linear layer maps the user vector to match the dimensions of MBTI labels Y, and softmax converts these to probabilities:

$$
y = \operatorname { s o f t m a x } ( H _ { P } W _ { u } + b _ { u } )
$$

where $W _ { u }$ represents the weight matrix, and $\pmb { b } _ { \pmb { u } }$ is the bias term used in the softmax calculation.

The focal loss is defined as:

$$
L _ { \mathcal { \boldsymbol { H } } } = \frac { 1 } { V } \sum _ { i = 1 } ^ { V } \sum _ { j = 1 } ^ { T } [ - \alpha ( 1 - p ( \hat { y } _ { j i } | \theta ) ) ^ { \gamma } y _ { j i } \log p ( \hat { y } _ { j i } | \theta ) ]
$$

where $\alpha$ is a weighting factor to balance the importance of different classes, and $\gamma$ adjusts the rate at which easy examples are down-weighted. The variable $V$ denotes the total number of training samples, and $y$ corresponds to the actual label for each dimension of personality traits. $p ( \hat { y } | \theta )$ indicates the predicted probability for each dimension, computed based on model parameters $\theta$ . Here, $T$ represents the number of personality dimensions, typically $T = 4$ .

Furthermore, we combine the focal loss with the contrastive loss:

$$
L = L _ { f l } + \lambda L _ { c l }
$$

where $\lambda$ is a hyperparameter adjusting the relative importance of each loss in the objective function.

# Experiments

# Datasets

Considering the datasets employed in prior research (Hu et al. 2024; Yang et al. 2023a, 2021a,b), we also choose to use the Kaggle1 and Pandora2 MBTI personality datasets for our experiments. The Kaggle dataset comes from PersonalityCafe3 and includes over 8,600 entries. Each entry lists an individual’s four-letter MBTI type and excerpts from their 50 most recent posts. The Pandora dataset is from Reddit4 and features posts from 9,067 users with self-reported MBTI types. The number of posts varies, with each user having from dozens to hundreds of posts. To prevent information leaks, words related to personality label are replaced with $\langle \mathrm { m a s k } \rangle$ (Yang et al. 2023a). We also adopt their data partitioning strategy, splitting the datasets into training, validation, and testing sets using a 60-20-20 ratio. Performance is evaluated using the Macro-F1 metric.

Table 1 presents the distribution of MBTI types and the number of analyzed posts for each dataset. While the Kaggle dataset retains its original label distribution, the Pandora dataset, due to its more pronounced class imbalance, employs undersampling strategies during the training, validation, and testing phases to ensure balanced labels along each MBTI dimension. With these adjustments, our model achieves exceptional performance on the Pandora dataset while maintaining stable, state-of-the-art results on the Kaggle dataset. These findings confirm that our model design consistently demonstrates robust generalization and highlevel performance under varying distributional conditions.

Table 1: Statistics of the Kaggle and Pandora datasets.   

<html><body><table><tr><td>Dataset</td><td>Types</td><td>Train</td><td>Validation</td><td>Test</td></tr><tr><td rowspan="4">Kaggle</td><td>I/E</td><td>4032/1173</td><td>1330/405</td><td>1314/421</td></tr><tr><td>S/N</td><td>724/4481</td><td>230/1505</td><td>243/1492</td></tr><tr><td>T/F</td><td>2388/2817</td><td>802/933</td><td>791/944</td></tr><tr><td>P/J</td><td>3160/2045</td><td>1007/728</td><td>1074/661</td></tr><tr><td rowspan="5">Pandora</td><td>I/E</td><td>4314/1126</td><td>1425/388</td><td>1403/411</td></tr><tr><td>S/N</td><td>621/4819</td><td>202/1611</td><td>205/1609</td></tr><tr><td>T/F</td><td>3527/1913</td><td>1160/653</td><td>1164/650</td></tr><tr><td>P/J</td><td></td><td></td><td></td></tr><tr><td></td><td>3211/2229</td><td>1064/749</td><td>1035/779</td></tr></table></body></html>

# Baselines

SVM (Cui and Qi 2017) and XGBoost (Tadesse et al. 2018): This method combines all user posts into one long document, extracts features using a bag-of-words model, and processes the data with classification algorithms like SVM or XGBoost.

BiLSTM (Tandera et al. 2017): This method uses a BiLSTM architecture with average pooling to merge post embeddings into a single representation for personality prediction.

BERTconcat (Jiang, Zhang, and Choi 2020): This method concatenates a user’s posts into one long post, extracts features using the BERT model, and maps these features to personality labels through fully connected layers.

BERTmean (Keh, Cheng et al. 2019): This approach encodes posts using the BERT model, applies average pooling to create user feature representation, and maps these features to personality labels via fully connected layers.

AttRCNN (Xue et al. 2018): This method uses a hierarchical deep neural network that combines an AttRCNN structure with an Inception variant to extract deep semantic features from social network texts. These features are then combined with statistical linguistic features and fed into regression algorithms.

AttnSeq (Lynn, Balasubramanian, and Schwartz 2020): This method uses a hierarchical attention mechanism to process posts, applying word-level and message-level attentions for personality prediction.

Transformer-MD (Yang et al. 2021a): This method uses a Multi-Document Transformer architecture to encode posts without order bias, utilizing memory tokens with shared position embeddings. This allows dynamic access to information across posts, creating a coherent personality profile across multiple documents.

TrigNet (Yang et al. 2021b): This method uses a psycholinguistic tripartite graph network, which combines a BERTbased initializer with a graph attention mechanism to integrate psycholinguistic knowledge for text-based personality detection.

D-DGCN (Yang et al. 2023a): This method employs a Dynamic Deep Graph Convolutional Network to detect personality traits from social media posts. It constructs graphs dynamically, with posts as nodes using multi-hop connectivity and deep graph convolutional layers, reducing biases from post order.

TAE (Hu et al. 2024): This method combines LLM-based text augmentation with a small model to improve personality detection. It uses LLM to generate augmented posts focusing on semantic, sentiment, and linguistic aspects, enhancing data and personality label representations.

# Implementation Details

Our deep learning models are developed using PyTorch (Paszke et al. 2017), utilizing AdamW (Loshchilov and Hutter 2017) as the optimizer. The learning rate is set to $3 \times 1 0 ^ { - 5 }$ . We conducted our experiments on a setup with an NVIDIA A6000 GPU. We use BERT-base-uncased as the text encoder and Meta-Llama-3-8B-Instruct as the lightweight LLM text embedding extraction tool. The model setup specifies a batch size of 4, with the temperature parameter $( \tau )$ maintained at 0.07 and the trade-off parameter $( \lambda )$ at 1. Each post is limited to 128 tokens. Dataset limits are set to 50 posts for Kaggle and 100 posts for Pandora.

We use $\mathsf { G P T } { - } 4 ^ { 5 }$ for interpreting MBTI labels and Meta-Llama-3-8B-Instruct for evaluating the performance of fine-tuned lightweight models on personality detection task. The fine-tuning process was conducted under the LoRA framework (Hu et al. 2021). We selected a rank of 16 and conducted the fine-tuning over a duration of 5 epochs. The learning rate was set at 0.0001, and the training was carried out with a batch size of 8.

Figure 3: Prompts for fine-tuning lightweight LLM.   

<html><body><table><tr><td colspan="3">Posts:Iam all about the sacrament of reconciliation(confession)x85...</td></tr><tr><td></td><td colspan="3">Please determine the author’s MBTI type based on the following text. Please answer the MBTI type directly with out any other explanation. MBTI has four letters,the first letter is Eor I, the second letter is Nor S, the third letter is TorF,and the fourth letter is JorP.The above is the content of the posts.Your response must be one of the sixteen MBTI types.Here are the sixteen types enumerated: ISTJ,ISFJ, INFJ,INTJ, ISTP,ISFP,</td></tr><tr><td>Output</td><td colspan="3">INFP,INP,EP,EP,ENFP,ENP,EJ,EFJ,ENFJ,J. INFP</td></tr><tr><td></td><td colspan="3">(a)PromptTemplate forLlama3MBTI16-TypeClassification</td></tr><tr><td colspan="3">Propt: Brove od he oxtprverphd Wouldyouifer th athor</td><td>Output: I</td></tr><tr><td colspan="3">Prompt: Considering the details in the text paragraph, do you</td><td>Output:N</td></tr><tr><td colspan="3">think the author relies more on sensing (S) or intuition (N)? Prompt: From the given text paragraph, does it appear that the</td><td>Output: F</td></tr><tr><td colspan="3">author makes decisions based on thinking(T) or feeling (F)? Prompt: Agangzigthe texivig Paph, would yousay the auhor</td><td>Output: P</td></tr></table></body></html>

(b)Prompt Template forLlama3 MBTIBinary Classification

# Performance of Lightweight LLM

In the task of fine-tuning lightweight LLM for personality detection, it can be understood as involving a 16-category classification for MBTI personality types, with the prompt used shown in Figure 3 (a). Alternatively, for binary classification of each MBTI dimension, the prompt used is shown in Figure 3 (b). Zero-shot methods are also used as a baseline experiment to verify whether fine-tuning is effective.

The experimental results, as shown in Table 2, indicate that the fine-tuning performs better than the zero-shot approach but is still significantly lower than the baseline using the small BERT model. The results from experiments using fine-tuned lightweight LLM, alongside insights from previous work (Hu et al. 2024) on employing ChatGPT for personality detection, indicate that relying only on LLM for personality detection can lead to poor performance.

Table 2: Performance comparison on Kaggle dataset.   

<html><body><table><tr><td rowspan="2">Methods</td><td colspan="5">Kaggle</td></tr><tr><td>I/E</td><td>S/N</td><td>T/F</td><td>P/J</td><td>Avg</td></tr><tr><td>Llama3+ zero-shot</td><td>43.63</td><td>48.66</td><td>47.55</td><td>49.27</td><td>47.28</td></tr><tr><td>Llama3+ FT (2) Llama3+ FT(16)</td><td>48.75 48.80</td><td>46.25 49.26</td><td>48.54 47.55</td><td>48.62 50.26</td><td>48.04 48.97</td></tr><tr><td>BERT_mean</td><td>64.05</td><td>57.82</td><td>77.06</td><td>65.25</td><td>66.04</td></tr><tr><td>ETM(our)</td><td>68.97</td><td>71.21</td><td>86.19</td><td>84.78</td><td>77.79</td></tr></table></body></html>

# Overall Results

Table 3 shows that our ETM method surpasses all existing baseline models on the benchmark dataset in terms of Macro-F1 scores. Specifically, our ETM model secures performance gains of $9 . 7 8 \%$ and $1 1 . 5 1 \%$ over TrigNet, and $9 . 0 3 \%$ and $7 . 1 2 \%$ over D-DGCN on the benchmark datasets by effectively utilizing concatenated post representations, a feature that previous methods overlooked by ignoring post order. Additionally,Our ETM model outperforms the BERT concat method by $2 8 . 3 5 \%$ and $22 \%$ , addressing its limitations in context length and truncation, thanks to the lightweight LLM’s improved capacity for handling extended context. Our model outperforms TAE by $7 . 9 4 \%$ and $4 . 3 1 \%$ on the benchmark datasets. Although TAE uses personality label semantics to produce soft labels, it doesn’t fully leverage this information, resulting in only minor improvements in ablation studies. In contrast, our model fully interprets label information through LLM and integrates it into a contrastive learning framework, significantly refining the relationship between user vectors and the meanings of the 16 personality labels. Overall, our model achieves outstanding performance due to two key enhancements. Firstly, the integration of a lightweight LLM significantly boosts the small model encoder’s capability to process extended texts and deliver distinct semantic insights. Secondly, we employ a powerful LLM to generate multi-dimensional interpretations of personality labels, which are effectively incorporated within a contrastive learning framework.

# Ablation Study

To evaluate the significance of each component within our ETM model, we conducted an ablation study using the Kaggle dataset, as shown in Table 4. By employing a lightweight LLM to enhance user representation, removing this enhancement led to an $8 . 5 \%$ decrease in overall performance, confirming the significance of the diverse embedding semantics of lightweight LLM as a complement to textual features. Furthermore, we utilized the powerful generative model to interpret personality labels and employed a contrastive learning framework to establish the relationship between user vectors and label vectors. The removal of this component resulted in a performance decrease of $6 . 9 7 \%$ , highlighting the value of using multi-dimensional personality labels in the mapping process. We interpret personality labels from three perspectives: type definitions, thematic inclinations, and modes of expression. Removing these interpretations individually led to performance decreases of $1 . 9 5 \%$ , $1 . 6 3 \%$ , and $0 . 9 8 \%$ respectively, showing that type definitions have the greatest impact on performance among the multi-dimensional interpretations. When both key components are removed, the performance deteriorates by $12 . 9 9 \%$ , further validating the architecture’s efficacy in personality detection. These results demonstrate that our model effectively leverages the text embedding and generation capabilities of LLMs to significantly enhance the mapping process from user posts to personality labels.

# Impact of Llama3 Layer Selection

The Llama3 model is used as a crucial component to obtain representations of user posts from a lightweight LLM perspective, thereby enhancing the user vector representations. Given that the deeper layers of the transformer architecture can capture richer semantic information, we set up a comparative experiment by selecting the last $d _ { m }$ embedding layers of Llama3 for average pooling between layers. As shown in

Table 3: Performance comparison on Kaggle and Pandora datasets.   

<html><body><table><tr><td rowspan="2">Methods</td><td colspan="5">Kaggle</td><td colspan="5">Pandora</td></tr><tr><td>I/E</td><td>S/N</td><td>T/F</td><td>P/J</td><td>Avg</td><td>I/E</td><td>S/N</td><td>T/F</td><td>P/J</td><td>Avg</td></tr><tr><td>SVM(Cui and Qi 2017)</td><td>53.34</td><td>47.75</td><td>76.72</td><td>63.03</td><td>60.21</td><td>44.74</td><td>46.92</td><td>64.62</td><td>56.32</td><td>53.15</td></tr><tr><td>XGBoost (Tadesse et al. 2018)</td><td>56.67</td><td>52.85</td><td>75.42</td><td>65.94</td><td>62.72</td><td>45.99</td><td>48.93</td><td>63.51</td><td>55.55</td><td>53.50</td></tr><tr><td>BiLSTM(Tandera et al.2017)</td><td>57.82</td><td>57.87</td><td>69.97</td><td>57.01</td><td>60.67</td><td>48.01</td><td>52.01</td><td>63.48</td><td>56.21</td><td>54.93</td></tr><tr><td>BERT_concat (Jiang,Zhang,and Choi 2020)</td><td>58.33</td><td>53.88</td><td>69.36</td><td>60.88</td><td>60.61</td><td>54.22</td><td>49.15</td><td>58.31</td><td>53.14</td><td>53.91</td></tr><tr><td>BERT_mean (Keh, Cheng et al. 2019)</td><td>64.05</td><td>57.82</td><td>77.06</td><td>65.25</td><td>66.04</td><td>56.60</td><td>48.71</td><td>64.70</td><td>56.07</td><td>56.52</td></tr><tr><td>AttRCNN (Xue et al. 2018)</td><td>59.74</td><td>64.08</td><td>78.77</td><td>66.44</td><td>67.25</td><td>48.55</td><td>56.19</td><td>64.39</td><td>57.26</td><td>56.60</td></tr><tr><td>AttnSeq (Lynn,Balasubramanian,and Schwartz 2020)</td><td>65.43</td><td>62.15</td><td>78.05</td><td>63.92</td><td>67.39</td><td>56.98</td><td>54.78</td><td>60.95</td><td>54.81</td><td>56.88</td></tr><tr><td>Transformer-MD (Yang etal. 2021a)</td><td>66.08</td><td>69.10</td><td>79.19</td><td>67.50</td><td>70.47</td><td>55.26</td><td>58.77</td><td>69.26</td><td>60.90</td><td>61.05</td></tr><tr><td>TrigNet (Yang etal.2021b)</td><td>69.54</td><td>67.17</td><td>79.06</td><td>67.69</td><td>70.86</td><td>56.69</td><td>55.57</td><td>66.38</td><td>57.27</td><td>58.98</td></tr><tr><td>D-DGCN (Yang et al. 2023a)</td><td>68.41</td><td>65.66</td><td>79.56</td><td>67.22</td><td>70.21</td><td>61.55</td><td>55.46</td><td>71.07</td><td>59.96</td><td>62.01</td></tr><tr><td>D-DGCN+Co (Yang et al.2023a)</td><td>69.52</td><td>67.19</td><td>80.53</td><td>68.16</td><td>71.35</td><td>59.98</td><td>55.52</td><td>70.53</td><td>59.56</td><td>61.40</td></tr><tr><td>TAE (Hu et al. 2024)</td><td>70.90</td><td>66.21</td><td>81.17</td><td>70.20</td><td>72.07</td><td>62.57</td><td>61.01</td><td>69.28</td><td>59.34</td><td>63.05</td></tr><tr><td>ETM (our)</td><td>68.97</td><td>71.21</td><td>86.19</td><td>84.78</td><td>77.79</td><td>68.57</td><td>64.91</td><td>66.07</td><td>63.53</td><td>65.77</td></tr></table></body></html>

<html><body><table><tr><td rowspan="2">Methods</td><td colspan="5">Kaggle</td></tr><tr><td>I/E</td><td>S/N</td><td>T/F</td><td>P/J</td><td>Avg</td></tr><tr><td>ETMw/o llama3-boost</td><td>61.10</td><td>56.98</td><td>84.12</td><td>82.51</td><td>71.18</td></tr><tr><td>ETMw/o gpt4-cl</td><td>57.50</td><td>65.31</td><td>83.00</td><td>83.68</td><td>72.37</td></tr><tr><td>ETMw/o definition</td><td>67.95</td><td>67.54</td><td>85.17</td><td>84.42</td><td>76.27</td></tr><tr><td>ETMw/o tendencies</td><td>69.48</td><td>68.31</td><td>85.74</td><td>82.54</td><td>76.52</td></tr><tr><td>ETMw/o expression</td><td>68.28</td><td>69.58</td><td>85.24</td><td>85.03</td><td>77.03</td></tr><tr><td>ETMw/o all</td><td>55.31</td><td>49.00</td><td>82.92</td><td>83.50</td><td>67.68</td></tr><tr><td>ETM (our)</td><td>68.97</td><td>71.21</td><td>86.19</td><td>84.78</td><td>77.79</td></tr></table></body></html>

Table 4: Results of ablation study on Macro-F1 on the Kaggle dataset.

Table 5, choosing the last five embedding layers of Llama3 is more effective in helping the small model enhance the representation of user vectors.   
Table 5: Performance of Selecting Llama’s Last $d _ { m }$ Layer Embeddings.   

<html><body><table><tr><td rowspan="2">Methods</td><td colspan="5">Kaggle</td></tr><tr><td>I/E</td><td>S/N</td><td>T/F</td><td>P/J</td><td>Avg</td></tr><tr><td>中d</td><td>68.19</td><td>68.82</td><td>82.65</td><td>85.09</td><td>76.19</td></tr><tr><td></td><td>69.40</td><td>68.07</td><td>83.48</td><td>85.16</td><td>76.53</td></tr><tr><td></td><td>67.97</td><td>67.52</td><td>85.65</td><td>80.51</td><td>75.41</td></tr><tr><td>12</td><td>68.51</td><td>68.71</td><td>85.73</td><td>82.45</td><td>76.35</td></tr><tr><td>Pdm=1,2,3,4,5 (Llama3,dm)</td><td>68.97</td><td>71.21</td><td>86.19</td><td>84.78</td><td>77.79</td></tr></table></body></html>

# Effect of Trade-Off Parameter

We tested various $\lambda$ in the ETM, from $\{ 0 . 5 , 1 , 1 . 5 , 2 , 2 . 5 , 3 ,$ $3 . 5 , 4 , 4 . 5 \}$ . Figure 4 demonstrates that the model’s performance is optimal at $\lambda = 1$ across the benchmark datasets, with a decline in performance as $\lambda$ exceeds this value. This indicates that a moderate increase in $\lambda$ improves the mapping from user vectors to personality labels by via contrastive learning. However, excessively high $\lambda$ values cause the focus loss to become increasingly insignificant in the optimization objective, resulting in poorer performance. Therefore, $\lambda = 1$ is identified as the optimal setting.

![](images/17db35735dfb1c24e01e39d9926dfc1ad63200ff1ac19218adc80ca7bab93b5f.jpg)  
Figure 4: Performance curves for different trade-off parameter.

# Conclusion

In this paper, we propose the LLM-Enhanced Text Mapping Model for Personality Detection, which achieves the goal of accurately mapping posts to specific personality types. Our method leverages the text embedding and text generation capabilities of LLM to address the issues of poor user vector representation and the insufficient relationship between user vectors and personality labels in small model-based personality detection. Firstly, we employ lightweight LLM text embeddings for concatenated documents, enhanced by a crossattention mechanism to improve user vector accuracy. Secondly, we use a powerful LLM to deliver multidimensional explanations of personality labels. This is integrated with a contrastive learning framework that better maps text to labels, enhancing the process. Our model outperforms the best existing baseline methods on benchmark datasets, achieving improvements of $7 . 9 4 \%$ and $4 . 3 1 \%$ . In future work, we plan to build a knowledge graph focused on emotion theory and psychology to enhance text-based emotion recognition with LLMs, improving our personality detection model.