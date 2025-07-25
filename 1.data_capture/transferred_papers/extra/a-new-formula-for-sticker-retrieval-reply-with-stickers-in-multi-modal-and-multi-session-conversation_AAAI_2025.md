# A New Formula for Sticker Retrieval: Reply with Stickers in Multi-Modal and Multi-Session Conversation

Bingbing Wang1,2\*, Yiming $\mathbf { D } \mathbf { u } ^ { 3 * }$ , Bin Liang3†, Zhixin Bai1, Min Yang4, Baojun Wang5, Kam-Fai Wong3, Ruifeng $\mathbf { X } \mathbf { u } ^ { 1 , 2 , 6 \dagger }$

1Harbin Institute of Technology, Shenzhen, China 2Guangdong Provincial Key Laboratory of Novel Security Intelligence Technologies, Shenzhen, China 3The Chinese University of Hong Kong, Hong Kong, China 4Shenzhen Institutes of Advanced Technology Chinese Academy of Sciences, Shenzhen, China 5Huawei Noah’s Ark Lab, Shenzhen, China 6Peng Cheng Laboratory, Shenzhen, China bingbing.wang, baizhixin @stu.hit.edu.cn, ydu@se.cuhk.edu.hk, bin.liang@cuhk.edu.hk, xuruifeng@hit.edu.cn

# Abstract

Stickers are widely used in online chatting, which can vividly express someone’s intention, emotion, or attitude. Existing conversation research typically retrieves stickers based on a single session or the previous textual information, which can not adapt to the multi-modal and multi-session nature of the real-world conversation. To this end, we introduce MultiChat, a new dataset for sticker retrieval facing the multimodal and multi-session conversation, comprising 1,542 sessions, featuring 50,192 utterances and 2,182 stickers. Based on the created dataset, we propose a novel Intent-Guided Sticker Retrieval (IGSR) framework that retrieves stickers for multi-modal and multi-session conversation history drawing support from intent learning. Specifically, we introduce sticker attributes to better leverage the sticker information in multi-modal conversation, which are incorporated with utterances to construct a memory bank. Further, we extract relevant memories for the current conversation from the memory bank to identify the intent of the current conversation, and then retrieve a sticker to respond guided by the intent. Extensive experiments on our MultiChat dataset reveal the robustness and effectiveness of our IGSR approach in multi-session, multi-modal scenarios.

# Introduction

With the advent of instant messaging applications, stickers have become popular in online chats (Zhang et al. 2024). On the one hand, several works on stickers mainly concentrate on sentiment analysis (Liu, Zhang, and Yang 2022; Ge et al. 2022; Zhao et al. 2023). In contrast, stickers present unique advantages in fostering a vibrant and innovative atmosphere within conversations due to their visual nature (Nilasari, Sudipa, and Sukarini 2018; Albar 2018). Therefore, integrating automatic sticker replies based on previous conversations into dialogue systems can make interactions more engaging.

Current Session Historical Session We're still not clear about this 品 ？ User5 User25 Confused +1,I forgot what we 品 What's going on?   
I Userl0 need to do for this group project. User22 0 Trim loose threads. User22   
User5 0 Fix computers. Computer Networks, we 0 User5 need two more people! Is User29 Computer Networks,we need Q anyonestillwithoutateam? one more person,one more! User29 Is anyone still without a team? 品 ？ 0 It'sabitawkward. User29 User29 embarrassment 煎 煎 只 品 = User29! User29！ 贝 Sadness joy embarrassment embarrassment X X Retrieved by the current session and Retrieved by the current session historical session

Recent research in sticker retrieval has focused on using stickers to respond to text-based contexts or a single conversation session, enhancing the expression of emotions, attitudes, and opinions (Laddha et al. 2020; Gao et al. 2020; Fei et al. 2021a; Zhang et al. 2022, 2024). However, they only focus on the textual content of the current session when retrieving a sticker for a response, ignoring the stickers and previous conversation sessions. Figure 1 shows a conversation with multi-modal and multi-session information in the real-world scenario. This example illustrates two issues not considered in existing sticker retrieval tasks: 1) Combining information from historical sessions is necessary to understand that the current session discusses “course team formation not yet successful”; 2) Learning from the stickers in historical sessions is needed to determine a sticker with the expression “embarrassment” to respond to the current session. Furthermore, stickers are utilized to respond to the previous conversation intent, expressing a user’s objective for the current utterance within a dialogue session (Shi et al. 2019). As illustrated in Figure 1, User 22 employs a “question mark” sticker to convey the intent of seeking clarification.

To address these above issues, we create a new sticker retrieval dataset with intent labels called MultiChat, to encompass more comprehensive real-world scenarios. MultiChat is a multi-modal and multi-session dataset specifically created for open-domain conversation. It contains 1,542 Chinese conversation sessions, featuring 50,192 utterances and 2,182 stickers. To enhance sticker retrieval towards multimodal and multi-session conversations, we introduce intent information and further propose a novel Intent-Guided Sticker Retrieval framework, to make full use of the multimodal information in the multi-session conversation for sticker retrieval based on the understanding of stickers’ intents, named IGSR. To be specific, we first define six attributes, which are combined with the historical conversational utterances and fed into the LLM for constructing a memory bank. Afterward, relevant memories are retrieved from a memory bank based on the current session via OpenAI’s LLM-based embedding model (text-embedding-ada003). Furthermore, a pre-trained Vision Language Model (VLM) (Radford et al. 2021a) plays a central role in our framework, serving as the foundation for both the text and image encoders. The VLM enables the model to concurrently derive user intents and retrieve stickers through a multi-task learning approach. For intent derivation, relevant memories and the current session are fed into the text encoder to obtain memory and context representations. These representations are concatenated to form an intent representation, which is then used by a classifier to predict user intent. For sticker retrieval, sticker representations are generated via the image encoder. These representations, combined with the derived intent representation, are used to retrieve the most suitable sticker. The main contributions of our work can be summarized as follows:

• We create MultiChat, a new dataset designed to facilitate the research of sticker retrieval towards multi-modal and multi-session conversations in social media.   
• We propose a novel IGSR framework for sticker retrieval, in which a multi-modal history modeling strategy and a multi-task learning scheme are devised to retrieve the most appropriate stickers for response based on the learning of the sticker’s intent.   
• Experimental results on our MultiChat dataset demonstrate that our proposed IGSR framework outperforms the baseline models.

# Related Work

# Sticker Dataset

Stickers have gained substantial attention in recent years (Zhang et al. 2024), particularly within the domain of multimodal sentiment analysis, where researchers have developed diverse datasets (Liu, Zhang, and Yang 2022; Zhao et al. 2023). Due to their visual nature, stickers offer unique advantages in enhancing conversational dynamics. This insight has prompted researchers to explore context-based sticker retrieval, shifting the focus from simply expressing sentiment to strategically using stickers based on conversational cues. Fei et al. (2021a) presented a meme-incorporated open-domain conversation task with a dataset including $4 5 \mathrm { k }$ Chinese conversations and $6 0 6 \mathrm { k }$ utterances. Additionally, Ge et al. (2022) introduced a Chinese multi-modal dataset specifically designed for sentiment analysis, comprising 28k text-sticker pairs and $1 . 5 \mathrm { k }$ annotated samples. In these datasets, stickers primarily serve as supplements or responses to the text-based context. However, the conversation context is inherently multi-modal and multi-session in the real world. To address this limitation, this paper creates a comprehensive dataset for sticker retrieval in conversations.

# Multi-modal Conversation Method

Several multi-modal studies aim to enhance the efficacy of conversational agents by enriching textual responses with associative vision elements (Huang et al. 2024; Zhang et al. 2024; Maharana et al. 2024). In contemporary social media interactions, using stickers as replies has become commonplace, resulting in a growing body of work focused on sticker retrieval to assist users in selecting the appropriate sticker for responses. Gao et al. (2020) introduced a sticker response selector model that utilizes a convolutional sticker image encoder paired with a self-attention multi-turn dialogue encoder. This model employs a deep interaction network for detailed matching and a fusion network to determine the final matching score. Fei et al. (2021a) presented the Meme Incorporated Open-domain Dialogue (MOD) task, which seamlessly integrates text generation and internet meme prediction into a single sequence generation process. While these methods match conversation contexts with stickers, they fail to model the multi-modal context and the relationships between different sessions. This underscores the need for approaches that better handle the complexities of multimodal, multi-session conversations.

# MultiChat Dataset

# Data Preparation

We curated our dataset from the popular social platform WeChat1, which features a diverse range of conversations and stickers in both individual and group chats. We specifically chose five active chat groups with engaged participants and collected their conversations. These groups engage in open-domain discussions, resulting in a varied and extensive use of stickers.

We established rigorous guidelines and policies for data preprocessing. To safeguard user privacy, all personal information (such as real names, ages, addresses, etc.) is removed, and user IDs are anonymized. Furthermore, any content containing offensive, or insulting language is excluded.

Table 1: Statistics of MultiChat. Avg. represents average.   

<html><body><table><tr><td>DatasetStatistics</td><td>Train</td><td>Valid</td><td>Test</td></tr><tr><td># sessions # samples # utterances # stickers</td><td>1,120 3,447 30,092 1,295</td><td>238 1,290 9,711 637</td><td>184 1,114 10,389 658</td></tr><tr><td>#users Max.utterances ina session Avg.utterances in a session Avg.users in a conversation</td><td>71 428 26.87 5.13</td><td>60 423 40.80 6.56</td><td>63 436 56.46 7.14</td></tr></table></body></html>

The entire chat content is segmented into distinct conversations to maintain the integrity and independence of each conversation. Following this framework, we systematically examine each sticker in the chat history, capturing its associated context to ensure that each sticker is linked to a corresponding conversation context.

# Data Annotation

We recruited five experienced researchers, each with over three years of expertise in multi-modal learning, to serve as annotators. Their tasks were to 1) assess the appropriateness of stickers and 2) categorize each sticker with style and intent tags for every conversation. To enhance the selection of stickers for replies, recognizing sentiment, emotion, and intent is crucial. Therefore, inspired by (Aman and Szpakowicz 2008) and incorporating the labels from GoEmotions (Demszky et al. 2020), we applied these intent tags to capture the diverse and complex nature of conversational expressions.

# Quality Assessment

To assess inter-annotator agreement, we use Cohen’s Kappa Statistic (Cohen 1960). The average Cohen’s Kappa scores for annotator pairs evaluating style and intent in the MultiChat dataset are 0.919 and 0.832, respectively. These substantial Kappa scores demonstrate strong agreement among the annotations, indicating the reliability of the annotations. After the initial annotation, we conducted a post-annotation review process. This involved a detailed examination of randomly selected samples from the dataset by senior researchers to ensure consistency and correctness in annotations, providing another layer of quality.

# Dataset Statistics

Table 1 provides a detailed overview of the dataset statistics. In total, there are 1,542 conversation sessions which contain 50,192 utterances, and 2,182 stickers. In this paper, we split each session into multiple samples based on the stickers. For instance, consider a session containing $m$ utterances $C = \{ u _ { 1 } , u _ { 2 } , \dots , u _ { m } \}$ , where each $u$ can be either text or a sticker. If $u _ { 4 }$ , $u _ { 1 0 }$ , and $u _ { 1 5 }$ are stickers, the three samples are formed by $\{ u _ { 1 } , u _ { 2 } , \ldots , u _ { 4 } \}$ , $\{ u _ { 1 } , u _ { 2 } , \ldots , u _ { 1 0 } \}$ , and $\{ u _ { 1 } , u _ { 2 } , \ldots , u _ { 1 5 } \}$ , respectively. After dividing, there are 4,851 samples. The ratio of the training set, validation set, and test set is approximately 3:1:1. Each conversation session includes 41.38 utterances on average. The average number of users who participate in a conversation is 6.28.

# Methodology

In this section, we introduce our novel IGSR framework for multi-modal multi-session sticker retrieval in detail. Each conversation is denoted as $\textit { C } = \{ H _ { n } , D _ { m } , v _ { m } \}$ , where $H _ { n } = \left\{ H _ { 1 } , H _ { 2 } , . . . , H _ { n } \right\}$ indicates $n$ past sessions. Each session includes multiple utterances or stickers among speakers. ${ \cal D } _ { m } ~ = ~ \{ ( s _ { 1 } , \bar { u } _ { 1 } ) , ( s _ { 2 } , u _ { 2 } ) , . . . , ( s _ { m } , u _ { m } ) \}$ denotes the context of the current session at $m$ step, where $s$ and $u$ denote the speaker and the utterance/sticker from the corresponding speaker. $v _ { m }$ is the ground truth sticker to $D _ { m }$ with history session $H$ .

To select an appropriate sticker $v$ from the sticker set $V$ or a conversation based on past and current sessions, we propose a pipeline framework, IGSR, to deal with the sticker retrieval task, as shown in Figure 2, which consists of three primary components: 1) Multi-modal History Modeling: This component aggregates multi-modal historical data, including attributes using a Large Language Model (LLM) to create a memory bank. 2) Intent Derivation: This component integrates relevant memory from the memory bank with the current session to predict the user’s intent. 3) Sticker Retrieval: This final component selects a sticker based on the derived intent and sticker representations.

# Multi-modal History Modeling

History session is crucial in conversation which can help in understanding the context and maintaining the topic. However, existing conversation systems primarily concentrate on the single-modal text from history sessions or even overlook the history sessions entirely. Recognizing that real-world dialogues convey emotions and intentions in a multi-modal manner, we propose to model the multi-modal history.

To better capture the richness of historical interactions, for each sticker in the history session, we design six attributes to represent the key information and reduce unnecessary interference from irrelevant information for the learning of stickers, i.e. intent $L _ { I }$ , style $L _ { S }$ , gesture $L _ { G }$ , posture $L _ { P }$ , facial expression $L _ { F }$ , and verbal $L _ { V }$ . The intent and style attributes are provided from our dataset. For the other four attributes, we use Qwen-VL (Bai et al. 2023a), a Multi-modal Large Language Model (MLLM) to produce attribute-aware sticker descriptions based on the designed prompts $\{ A _ { G } , A _ { P } , A _ { F } , A _ { V } \}$ :

$$
\begin{array} { r l } {  { \{ L _ { G } , L _ { P } , L _ { F } , L _ { V } \} } \quad } & { { } } \\ & { = \mathbf { M L L M } ( \{ A _ { G } , A _ { P } , A _ { F } , A _ { V } \} ) } \end{array}
$$

Through several turns of interactions, we use system prompts such as, “This is a sticker used in conversation. Please provide several keywords to describe the gesture, posture, facial expression, and verbal content,” to leverage the LLM’s ability to generate these descriptive attributes for each sticker. Then, we feed the history sessions, including utterances and the attributes of the stickers, in chronological order into the LLM to generate a summary, which is then stored in the memory bank.

![](images/8f517ea2be354b2d7de3799ee96b49dd1de3e019a3ebf0f4f2e9827afa0fda2a.jpg)  
Figure 2: Illustration of our IGSR framework comprising multi-modal history modeling, intent derivation, and sticker retrieval.

$$
M _ { i } = \operatorname { L L M } ( D _ { i } , P )
$$

where $M _ { i }$ indicates multiple sentences that summarise the crucial information based on the current session, and $P$ denotes the prompt of LLM for memory generation: “Your goal is to summarize the session $I D _ { i } J ^ { \prime }$ . This operation is repeated $K$ times until the session ends, resulting in the final memory bank $M _ { K }$ .

# Intent Derivation

To capture the precise intent during the conversation, we incorporate as many relevant historical memories as possible to capture intent-related features. We then combine the current session with a designed encoder to predict the intent.

Similar to the process in multi-modal history modeling, we use attributes to represent stickers in the current session. Then, the current session and relevant memories are fed into the text encoder to derive summary and context representations. Using text-davinci-003 embeddings for semantic retrieval, we extract the top- $. \mathrm { \Delta N }$ relevant memories $M _ { r } = \{ M _ { 1 } , . . . , M _ { k } \}$ from the memory bank, which stores summaries of historical sessions based on the summary of the current session $D _ { i }$ .

$$
\begin{array} { c } { { R _ { m } = f _ { t e x t } ( M _ { r } ) } } \\ { { R _ { c } = f _ { t e x t } ( D _ { i } ) } } \end{array}
$$

where the $f _ { t e x t }$ represents the text encoder. The intent representation $R _ { I }$ is obtained by concatenating memory representation $R _ { m }$ and context representation $R _ { c }$ . The combined representation is then fed into a classifier, consisting of a linear layer for dimensionality reduction followed by a softmax function to produce the probability distribution for each intent category. We train the model using the standard gradient descent algorithm, minimizing the cross-entropy loss:

$$
\mathcal { L } _ { I } = - \sum _ { j = 1 } ^ { N } y _ { I } ^ { j } \log \hat { y } _ { I } ^ { j } + \lambda _ { I } \vert \vert \Theta _ { I } \vert \vert ^ { 2 }
$$

where $\sigma$ denotes the softmax function. $W _ { I } \in \mathbb { R } ^ { d _ { r } \times d _ { r } }$ is the learnable parameter and $b _ { I }$ is the bias training alone with the model. $d _ { r }$ is the dimension of intent representation. $y _ { I }$ and $\hat { y } _ { I }$ represent the ground truth and predicted label distribution of the intent. $\Theta _ { I }$ represents all the learnable parameters of the model, and $\lambda _ { I }$ denotes the coefficient for L2 regularization.

# Sticker Retrieval

To obtain the sticker representation, we apply the pre-trained vision transformer from CLIP model as the image encoder.

$$
R _ { v } = f _ { i m a g e } ( v )
$$

where $R _ { v }$ is the sticker representation and $f _ { i m a g e }$ denotes the image encoder.

Loss Function. During the training stage, we follow previous contrastive learning methods (Radford et al. 2021b) and utilize the InfoNCE loss (Oord, Li, and Vinyals 2018) to train our framework. Given a batch of $\beta$ intent-sticker representation pairs $r _ { j } , v _ { j } , j = 1 ^ { \beta }$ as training data, where $r _ { j } \in R _ { I }$ and $v _ { j } \in R _ { v }$ , we calculate the text-to-image contrastive loss $\mathcal { L } _ { v 2 t }$ and the image-to-text contrastive loss $\mathcal { L } _ { t 2 v }$ as follows:

$$
\begin{array} { r l } & { \mathcal { L } _ { v 2 t } = - \log \frac { \exp ( f _ { s i m } ( v _ { j } , r _ { j } ) / \tau ) } { \sum _ { k = 1 } ^ { \beta } \exp ( f _ { s i m } ( v _ { j } , r _ { k } ) / \tau ) } } \\ & { \mathcal { L } _ { t 2 v } = - \log \frac { \exp ( f _ { s i m } ( r _ { j } , v _ { j } ) / \tau ) } { \sum _ { k = 1 } ^ { \beta } \exp ( f _ { s i m } ( r _ { j } , v _ { k } ) / \tau ) } } \end{array}
$$

where $f _ { s i m } ( \boldsymbol { r } _ { j } , \boldsymbol { v } _ { j } )$ denotes the cosine similarity and $\tau \in$ $\mathbb { R } ^ { + }$ is a temperature factor. $\beta$ represents the batch size. The total loss of our approach is:

$$
\widehat { y } _ { I } = \sigma ( W _ { I } R _ { I } + b _ { I } ) , \mathrm { w h e r e } R _ { I } = R _ { m } \oplus R _ { c }
$$

$$
\mathcal { L } = \mathcal { L } _ { v 2 t } + \mathcal { L } _ { t 2 v } + \mathcal { L } _ { I }
$$

<html><body><table><tr><td>Modality</td><td>Model</td><td>P@1</td><td>P@3</td><td>P@5</td><td>mAP</td><td>GPT-4</td><td>HumanEvaluation</td></tr><tr><td rowspan="4">Text Modality</td><td>Baichuan2</td><td>7.00</td><td>18.22</td><td>25.58</td><td>13.57</td><td>38.60</td><td>48.25</td></tr><tr><td>Qwem1.5</td><td>8.08</td><td>21.36</td><td>31.42</td><td>15.96</td><td>44.74</td><td>36.84</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ChatGLM3</td><td>14.09</td><td>18.85</td><td>23.16</td><td>16.97</td><td>42.98</td><td>31.58</td></tr><tr><td rowspan="8">Multi-modality</td><td>MOD</td><td>3.23</td><td>4.85</td><td>6.46</td><td>5.21</td><td>21.93</td><td>51.75</td></tr><tr><td>SRS</td><td>3.77</td><td>6.82</td><td>9.34</td><td>7.09</td><td>24.56</td><td>52.63</td></tr><tr><td>IRRA</td><td>5.57</td><td>9.69</td><td>14.90</td><td>14.55</td><td>25.44</td><td>55.26</td></tr><tr><td>LGUR</td><td>8.26</td><td>14.81</td><td>17.68</td><td>15.23</td><td>21.93</td><td>58.77</td></tr><tr><td>PCME</td><td>9.34</td><td>21.72</td><td>28.73</td><td>16.93</td><td>35.09</td><td>64.04</td></tr><tr><td>CLIP</td><td>9.25</td><td>22.62</td><td>28.19</td><td>16.57</td><td>35.97</td><td>64.04</td></tr><tr><td>Qwen-VL</td><td>9.25</td><td>24.69</td><td>48.38</td><td>14.73</td><td>39.47</td><td>62.28</td></tr><tr><td>GPT-4</td><td>9.96</td><td>24.42</td><td>34.83</td><td>17.85</td><td>44.74</td><td>57.90</td></tr><tr><td rowspan="2"></td><td>LLaVA</td><td>10.23</td><td>28.10</td><td>44.08</td><td>13.74</td><td>30.70</td><td>61.40</td></tr><tr><td>IGSR</td><td>14.45*</td><td>34.02*</td><td>54.94*</td><td>19.39*</td><td>64.04*</td><td>67.54*</td></tr></table></body></html>

Table 2: Main results $( \% )$ of various methods. Bold indicates that our method surpasses other models. The results with ∗ indicate the significance tests of our IGSR over other baseline models (with $p$ -value $< 0 . 0 5$ ).

# Experiment

# Experimental Settings

Implement details. We apply GPT-4 (Achiam et al. 2023) to construct the memory bank in multi-modal history modeling and utilize the Qwen-VL (Bai et al. 2023a) to generate four attributes of stickers. We extract the top 5 relevant memories based on text-davinci-003 embeddings. The CLIP text encoder and image encoder are employed. We use a batch size of 2 and employ the Adam optimizer (Kingma and Ba 2014) for training. The learning rate is set to $1 \times 1 0 ^ { - 4 }$ . All experiments are conducted at Tesla $\mathrm { V } 1 0 0 \mathrm { s } ^ { 2 }$ .

Evaluation metrics. In our experiments, we utilize four evaluation metrics: $\mathrm { P } @ \mathrm { N }$ , mAP, GPT-4, and human evaluation. $\mathrm { P } @ \mathrm { N }$ measures the precision of the top $\mathbf { N }$ predictions, focusing on $\mathrm { P } @ 1 , \mathrm { P } @ 3 .$ and $\operatorname { P } @ 5$ . A result is correct if the retrieved sticker matches the intention label of the ground truth sticker, acknowledging that multiple stickers can appropriately respond to the same conversation. Mean average precision (mAP) is used as a widely accepted metric for evaluating retrieval accuracy (Lin et al. 2014). We utilize both GPT-4 and human evaluation to ensure a comprehensive assessment of models’ performance. Specifically, we randomly sample around $10 \%$ of the test cases (114 samples) and ask both GPT-4 and human evaluators to rate the quality on a scale of $\{ 0 , 1 \}$ , focusing on the background consistency and relevance of the stickers. This dual approach allows us to capture both automated and nuanced human perspectives.

# Baselines

To assess the performance of our model, we compare the proposed IGSR against several baseline methods, including existing text-based models and multi-modal models. (1) Text-based models: Baichuan2 (Yang et al. 2023), Llama3 (Touvron et al. 2023), ChatGLM3 (Du et al. 2022), Qwen1.5 (Bai et al. 2023b). Multi-modal models: IRRA (Jiang and Ye 2023), PCME (Chun et al. 2021), MOD (Fei et al.

Table 3: Experimental results $( \% )$ of ablation study. w/o mem and w/o int mean without memory and intent. “Human” represents the human evaluation.   

<html><body><table><tr><td>Model</td><td>P@1</td><td>P@3</td><td>P@5</td><td>mAP</td><td>GPT-4</td><td>Human</td></tr><tr><td>IGSR</td><td>14.45</td><td>34.02</td><td>54.94</td><td>19.39</td><td>64.04</td><td>67.54</td></tr><tr><td>w/o mem</td><td>7.36</td><td>25.40</td><td>42.46</td><td>18.87</td><td>29.38</td><td>56.01</td></tr><tr><td>w/o int</td><td>2.69</td><td>32.23</td><td>53.32</td><td>17.24</td><td>26.88</td><td>52.50</td></tr></table></body></html>

2021b), SRS (Gao et al. 2020), LGUR (Shao et al. 2022), CLIP (Radford et al. 2021a), LLaVA (Liu et al. 2024), Qwen-VL (Bai et al. 2023b), GPT-4 (Achiam et al. 2023). Notably, MOD and SRS are two sticker retrieval approaches.

For LLMs including Baichuan2, Qwen1.5 Llama3, ChatGLM3, Qwen-VL, GPT-4, LLaVA, we first design a text response generation response prompt that integrates the relevant summary with the current session to generate responses for each session. We then retrieve the appropriate sticker based on the generated response and the sticker attributes using BM25 (Robertson et al. 1995). In text-based LLMs such as Baichuan2, Qwen1.5, Llama3, and ChatGLM3, sticker attributes are derived using Qwen-VL. For multi-modal LLMs like Qwen-VL, GPT-4, and LLaVA, these attributes are directly obtained by the models themselves. We also utilize OpenAI’s LLM-based embedding model as a retriever and design a sticker intent prediction prompt for LLM to generate sticker intents or descriptions instead of responses for retrieval, but this approach results in decreased performance.

# Main Results

We compare the performance of our IGSR with baselines across various evaluation metrics, as shown in Table 2. IGSR consistently outperforms all baseline models, demonstrating its effectiveness in multi-modal, multi-session sticker retrieval. We observe improved results in Top N precision as N increases, since a larger N allows for a greater number of results, expanding the scope of potential matches and enhancing the likelihood of finding relevant labels.

Table 4: Experimental results $( \% )$ of effect of memory. / splits the results using memory and without using.   

<html><body><table><tr><td rowspan="2">Model</td><td>P@1</td><td>P@3</td><td>P@5</td><td>mAP</td></tr><tr><td>w/wo</td><td>w/wo</td><td>w/wo</td><td>w/wo</td></tr><tr><td>Baichuan2</td><td>7.0/7.7</td><td>18.2/19.6</td><td>25.6/29.7</td><td>13.6/15.1</td></tr><tr><td>Qwen1.5</td><td>8.1/8.9</td><td>21.4/22.2</td><td>31.4/32.9</td><td>16.0/16.7</td></tr><tr><td>Llama3</td><td>8.5/7.9</td><td>21.4/20.7</td><td>33.0/31.0 16.5/15.6</td><td></td></tr><tr><td>ChatGLM3</td><td>14.1/13.5</td><td></td><td>18.9/22.0 23.2/27.5</td><td>17.0/18.3</td></tr><tr><td>Qwen-VL</td><td>9.2/9.3</td><td>24.7/25.5</td><td>48.4/47.2</td><td>14.7/14.5</td></tr><tr><td>LLaVA</td><td>10.2/8.3</td><td>28.1/25.4</td><td>44.1/42.3</td><td>13.7/13.9</td></tr></table></body></html>

Text-based models, predominantly implemented by LLMs, outperform some multi-modal models. This superior performance is due to LLMs’ extensive parameterization and sophisticated network architectures, which significantly enhance their ability to understand and generate intricate language and image descriptions. For the metric $\mathbf { P } \ @ 1$ , ChatGLM3 performs the best, while for $\mathrm { P } @ 5$ , Qwen-VL shows the best performance, reaching $4 8 . 3 8 4 \%$ . All baseline models significantly underperform compared to our IGSR. This further highlights the effectiveness of our approach in intent derivation, underscoring the pivotal role of intention as a key bridging element in the process.

Multi-modal models include both small models (e.g., PCME, MOD, IRRA, LGUR, and CLIP) and large models (e.g., Qwen-VL, LLaVA). Small models primarily focus on capturing semantic relationships between textual and visual content but are not specifically designed for sticker retrieval scenarios, leading to inferior performance compared to our proposed framework. Interestingly, as a sticker retrieval model, MOD exhibits overall inferior performance compared to most multi-modal methods. This disparity can be attributed to MOD’s design, which targets retrieving suitable stickers from a limited set of similar candidates, thus emphasizing the distinction of local information among similar sticker expressions. As large models, Qwen-VL and LLaVA perform better than smaller models but still fall short of our approach, which leverages relevant memory, significantly enhancing performance. Across the $\mathbf { P } \ @ 1$ , $\operatorname { P } \ @ 3$ , and $\operatorname { P } @ 5$ metrics, our method achieves minimum improvements of $4 . 2 2 9 \%$ , $5 . 8 8 2 \%$ , and $6 . 5 5 3 \%$ , respectively. In summary, while large multi-modal models outperform smaller ones in sticker retrieval tasks, our IGSR framework surpasses both, demonstrating its exceptional robustness and effectiveness in handling complex multi-modal scenarios.

Furthermore, the results of GPT-4 and human assessments are not entirely consistent. For the text-based method, GPT4’s scores are higher compared to the multi-modal method, whereas human evaluations show the opposite trend. Additionally, human evaluations overall are higher than those of GPT-4. This discrepancy may stem from differing evaluation criteria: GPT-4 focuses on the alignment of image and text features, while human evaluators consider the conversational context and potential emotional nuances.

<html><body><table><tr><td>Char.Intent Style</td><td>P@1 P@3</td><td>P@5</td><td>mAP</td></tr><tr><td>√</td><td>7.54 23.79 9.96 26.30</td><td>33.12 44.25 52.96</td><td>12.73 19.21</td></tr><tr><td>√</td><td>10.95</td><td>32.68</td><td>20.58</td></tr><tr><td></td><td>5.30</td><td>23.34 44.43</td><td>19.67</td></tr><tr><td>√</td><td>9.61</td><td>27.02 43.45</td><td>20.41</td></tr><tr><td>√</td><td>7.81</td><td></td><td></td></tr><tr><td></td><td>V</td><td>24.42 39.95</td><td>18.98</td></tr><tr><td>√</td><td>√ 9.87</td><td></td><td></td></tr><tr><td>√ √</td><td>√ 14.45 34.02</td><td>28.28 54.94</td><td>46.95 18.32</td></tr></table></body></html>

![](images/3d9f78d413d4a3a9ed9b016b9a55fdebcb9621e9f2b933323d641de851b3b2b5.jpg)  
Table 5: Experimental results $( \% )$ of different attributes. $\surd$ represents the used attribute.   
Figure 3: Performance of our approach across all metrics when varying the number of utterances.

# Ablation Study

We also perform an ablation study on the use of memory and intent with evaluation results shown in Table 3. All ablation models perform worse than the complete model under all metrics, demonstrating the necessity of each component in our approach. Notably, removing relevant memory (“w/o memory”) leads to considerable performance degradation, underscoring the importance of summary in understanding the conversation context. Moreover, the removal of intent (“w/o intent”) significantly degrades performance, especially in the metric of $\mathbf { P } \ @ 1$ , indicating that the intent prediction during the model training improves the learning of sticker representation across different sticker properties.

Effect of Relevant Memory. Based on the results of the ablation study, we further explore the role of memory in sticker retrieval. Specifically, we design a prompt to generate the response without relevant memory. The comparative results are shown in Table 4, where the results with and without memory are separated by a slash. We observe that without using relevant memory, Baichuan2, Qwen1.5, ChatGLM3, and Qwen-VL show an increase in performance. This may be because incorporating relevant memory results in input sequences that are too long, causing LLMs to struggle with processing lengthy inputs effectively, which in turn impacts their performance. Notably, the performance of Llama3 and LLaVA improves when relevant memory is utilized, suggesting that these models have enhanced capabilities for processing long text inputs.

Effect of Attributes. In the process of multi-modal history modeling, six attributes are utilized to represent the key

Myopli User 10 User 29How about you book aseat next to them? User 6 Most if Hupleliorhe Shen User29That way you won't fel lonely. I'm in an internet café with five people. User 13 User 5 User 6 T It's tough. Can't even slack off. User 10 After all, their houses are there. User 24 I'malreadyslackingoff. User 13 No need to trouble myself. They always sell User5 Envying woman who have her own PC. User10 out the second row in every show. User 6 What's Huicheng? Im really not familiar   
User 5 User 29 Ihadhesaeocebtgot with it. is User 10 User 6 were from Huicheng in TikTok. I saw most people supporting strict policies   
User5 Ichaedstsi User 29 Trestissl Goi afi User1   
User 5 Ground Truth Prediction Prediction Ground Truth Prediction Prediction User 2 User 6 Ground Truth Prediction Prediction G T 品 Y 73 oo (a) (b) (c)

information of the sticker. This section examines the effectiveness of different attributes. We consider posture, gesture, and verbal expression as the main sticker characteristics generated by the LLM (referred to as “Char.”) and list all the scenarios in Table 5. It can be seen that using all the attributes gains the best performance. Conversely, the exclusion of attributes results in the lowest performance. Using only the intent to represent the sticker achieves performance close to using all attributes, with the lowest difference being just $1 . 3 4 6 \%$ at $\mathrm { P } @ 3$ . In contrast, using only the style attribute yields the worst performance. This indicates that the intent provided in our dataset better captures the sticker’s function, while the “Character” attribute provides supplementary details. These three types of attributes complement each other, and using all attributes together more comprehensively represents the crucial information of the sticker.

Effect of Max Number of Utterances. In a conversation, the context of the current session can provide valuable information for the response. This section explores and analyzes the effect of varying the maximum number of utterances per session on the performance of our IGSR framework. We conducted experiments with maximum values ranging from 50 to 450, and the results are illustrated in Figure 3. The maximum lengths of utterances in the training, validation, and test phases are 428, 423, and 436, respectively. Therefore, setting the maximum number of utterances to 450 means we do not restrict the utterances in the current session. As shown in Figure 3, we observe that using all utterances yields the best performance, while limiting the number to 50 results in the worst performance. This suggests that insufficient context and data hinder accurate predictions, highlighting the importance of utilizing a more extensive range of utterances for better model performance.

# Case Study

Figure 4 showcases various interactive cases retrieved by our approach. These examples highlight the IGSR framework’s ability to enhance communication with expressive and contextually relevant stickers. From Figure 4(a), we observe that the expression of sadness can be conveyed through both facial expressions and actions. Our proposed method effectively learns information from various expressions or actions, allowing for appropriate sticker responses. In addition, a significant challenge in sticker retrieval is its diversity in styles. Our model, incorporating intent information, reduces interference from different styles, thereby enabling more precise localization of useful sticker information. As shown in example Figure 4(b), our method retrieves stickers that encompass both realistic and cartoon styles.

Unlike traditional dialogue systems, this study collected open-domain group chat data, which exhibits characteristics of multi-party, multi-modal, multi-turn conversations. Therefore, complex conversational dynamics and interaction patterns may occur during chat dialogues. As seen in Figure 4(c), the content spoken by User $\boldsymbol { { \mathit { 1 } } }$ is not strongly related to the previous context, thus affecting the final sticker prediction results. Consequently, in future research, combining user information could be explored to further enhance the performance of sticker retrieval.

# Conclusion

We create a new dataset for multi-modal multi-session sticker retrieval, called MultiChat. Unlike previous studies that retrieve stickers based on the current session, our new dataset can cover more realistic scenarios. Based on our created dataset, we propose IGSR, a framework for sticker retrieval in conversation. In which, a multi-modal history modeling strategy is designed for memory bank construction, and a multi-tasking framework is employed to simultaneously derive intents and retrieve stickers. Extensive experiments on our MultiChat dataset highlight the importance of intent and demonstrate that our proposed approach effectively utilizes memory, achieving exceptional performance in multi-modal, multi-session sticker retrieval.