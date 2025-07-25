# SimRP: Syntactic and Semantic Similarity Retrieval Prompting Enhances Aspect Sentiment Quad Prediction

Zhongquan Jian1,2, Yanhao Chen3, Jiajian $\mathbf { L i } ^ { 1 }$ , Shaopan Wang1,2, Xiangjian Zeng4, Junfeng Yao3,2,1,6, Xinying $\mathbf { A } \mathbf { n } ^ { 5 , * }$ , Qingqiang Wu3,2,1,6,\*

1Institute of Artificial Intelligence, Xiamen University, Xiamen, China 2School of Informatics, Xiamen University, Xiamen, China 3School of Film, Xiamen University, Xiamen, China 4School of Journalism and Communication, Xiamen University, Xiamen, China 5Institute of Medical Information, Chinese Academy of Medical Sciences, Beijing, China 6Xiamen Key Laboratory of Intelligent Storage and Computing, School of Informatics, Xiamen University {jianzq,cyhao,shaopanw,lijiajian}@stu.xmu.edu.cn, $\{ \mathrm { x j z e n g , y a o 0 0 1 0 } \}$ @xmu.edu.cn, an.xinying $@$ imicams.ac.cn, wuqq $@$ xmu.edu.cn

# Abstract

Aspect Sentiment Quad Prediction (ASQP) is the most complex subtask of Aspect-based Sentiment Analysis (ABSA), aiming to predict all sentiment quadruples within the given sentence. Due to the complexity of sentence syntaxes and the diversity of sentiment expressions, generative methods gradually become the mainstream approach in ASQP. However, existing generative models are constrained in the effectiveness of demonstrations. Semantically similar demonstrations help in judging sentiment categories and polarities but may confuse the model in recognizing aspect and opinion terms, which are more related to sentence syntaxes. To this end, we first develop Syn2Vec, a method for calculating syntactic vectors to support the retrieval of syntactically similar demonstrations. Then, we propose Syntactic and Semantic Similarity Retrieval Prompting (SimRP) to construct effective prompts by retrieving the most related demonstrations that are syntactically and semantically similar. With these related demonstrations, pre-trained generative models, especially Large Language Models (LLMs), can fully release their potential to recognize sentiment quadruples. Extensive experiments in Supervised Fine-Tuning (SFT) and Incontext Learning (ICL) paradigms demonstrate the effectiveness of SimRP. Furthermore, we find that LLMs’ capabilities in ASQP are severely underestimated by biased data annotations and the exact matching metric. We propose a novel constituent subtree-based fuzzy metric for more accurate and rational quadruple recognition.

Code — https://github.com/jian-projects/simrp

# Introduction

Aspect-based Sentiment Analysis (ABSA) is a crucial task in Natural Language Processing (NLP) and real-world applications, aiming to enable machines to understand human concerns and opinions expressed in text (Liu et al. 2020; Brauwers and Frasincar 2022). ABSA has consistently attracted increasing attention in the Pre-trained Language Model (PLM) era, as well as in the Large Language

S In the end you PP NP VP end up withafair √ ? tab and nothing Syn2Vec IN NP PRP VBP PRT PP 1 DTNN RP IN NP Y but a great time !!! Constituent Retrieve NP cc NP Tree ↓ NP NP CC 1 NP DT JJ NN NP √ CC NP VBG DT JJ NN NP PP DT JJ NN DT JJ NN IN NP Syntaxsimilarsentence: DT NN A great choice at any cost and a great deal !

Model (LLM) era, owing to its wide range of applications, including product reviews, social media analysis, and customer feedback analysis (Zhang et al. 2024b). The most representative and challenging subtask in ABSA is Aspect Sentiment Quad Prediction (ASQP), which aims to predict all sentiment quadruples within the given sentence (Zhang et al. 2023, 2024a), each consisting of aspect term, opinion term, aspect category, and sentiment polarity, denoted as $( a , o , c , s ) { \bar { } }$ . As depicted in Figure 1, given a review sentence ”In the end you end up with a fair tab and nothing but a great time !!!”, ASQP requires predicting two containing sentiment quadruples: (Null, fair, restaurant price, positive) and (Null, great, restaurant general, positive), where Null represents an implicit aspect term in this sentence.

Due to the complexity of ASQP in recognizing all sentiment quadruples, generative models, like T5 (Raffel et al. 2020) and GPT (Brown et al. 2020), with seq2seq paradigm have been mainstream approaches and achieved promising results (Peper and Wang 2022). Initially, studies show that generation objectives greatly influenced the extraction of sentiment quadruples. Representatively, GAS (Zhang et al. 2021b) explored the influences of annotation-style and extraction-style generation objectives, demonstrating the superiority of the extraction-style mode in ASQP. Another method is to paraphrase sentiment quadruples into natural sentences (Zhang et al. 2021a; Hu et al. 2022), which follows attributes of generative models, therefore achieving outstanding performance. Subsequently, MvP (Gou, Guo, and Yang 2023) further explored the impact of quadruple generation order on model performance and proposed voting on the most reasonable quadruples generated in different orders, significantly enhancing predictive capability.

With the rise of In-context Learning (ICL) (Dong et al. 2022; Xu et al. 2024), researchers find that model input also plays a crucial role in activating the reasoning capabilities of PLM and LLM (Zhao et al. 2023). Furthermore, adding detailed instructions and demonstrations to task prompts can significantly activate the model performance on downstream tasks, whether in zero-shot inference or supervised training (Yang et al. 2024). For instance, Sun et al. (2024) showed that integrating enriched domain insights from the knowledge base enhances the input, improving model performance. Recently, Instruct-ABSA (Scaria et al. 2024) proposed an instruction learning paradigm that can extend the input with several positive, negative, and neutral demonstrations, and instruction tune the backbone model to address the ABSA subtasks, yielding significant performance improvements. Consequently, the following research has begun to investigate optimal demonstration selection. Numerous studies (Liu et al. 2022; Min et al. 2022) found that choosing demonstrations semantically and label-wise closer to the actual input is more effective.

In the deep learning era, semantic similarity retrieval has been successfully applied to various NLP and computer vision tasks, demonstrating superior performance in acquiring valuable information relevant to the target task. For the ASQP task, semantically similar demonstrations provide similar sentiment insights to aid in judging sentiment categories and polarities (Zhang et al. 2023). However, recognizing aspect and opinion terms is more closely tied to sentence syntax, emphasizing the need for syntactically similar demonstrations, as they offer explicit syntactic cues for identifying these terms. As the example shown in Figure 1, the target sentence has the same constituent subtree (CsT) as the retrieval sentence, allowing quadruples in the target sentence to be intuitively recognized by referencing the quadruple recognition pattern of syntactically similar demonstrations. Therefore, with the syntactically and semantically similar demonstrations, the generative models, especially LLMs, can fully release their potential in recognizing sentiment quadruples, and thus improve the performance of ASQP.

To this end, we first develop Syn2Vec, a syntactic vector calculation method that allows representing sentence syntaxes as vectors to support the retrieval of syntactically similar demonstrations. Syn2Vec is built upon the vocabulary of CsTs, meaning that syntactic vectors are word vectors derived from CsTs present in sentences. Thus, syntactic similarity between sentences can be measured by comparing their syntactic vectors, for example, using Mean Square Error (MSE). Building on Syn2Vec, we introduce Syntactic and Semantic Similarity Retrieval Prompting (SimRP), which constructs effective prompts by retrieving demonstrations that are both syntactically and semantically similar. Specifically, we first use Syn2Vec to retrieve syntactically similar demonstrations from the training set, then use Sentence-BERT model (Reimers and Gurevych 2019) to select the most semantically similar ones from this subset. To verify the effectiveness of the proposed SimRP, we conduct extensive experiments in Supervised Fine-Tuning (SFT) and ICL paradigms on two public ASQP datasets, where the ICL paradigm is implemented under zero-shot and few-shot settings based on GPT-4 and Llama-3 models. In addition, we propose a novel CsT-based fuzzy metric to evaluate the accuracy of quadruple recognition, which can better reflect the potential of LLMs in ASQP. In summary, the main contributions of this work are as follows:

• We propose SimRP to address the ASQP task by constructing effective prompts with syntactic and semantically similar demonstrations, which can effectively activate the reasoning capabilities of PLMs and LLMs. • We developed a novel Syn2Vec to represent sentence syntaxes as vectors to enable the retrieval of syntactically similar demonstrations, after which Sentence-BERT is used to select the most semantically similar ones. • Experiments in SFT and ICL paradigms demonstrate the effectiveness of the proposed SimRP, achieving state-ofthe-art results on two public ASQP datasets. The proposed CsT-based fuzzy evaluation metric verifies the potential of LLMs in ASQP.

# Methodology

# Definition

Given a review sentence $x$ , ASQP aims to identify all aspectbased sentiment quadruples $\{ ( a , o , c , s ) \}$ , where each corresponds to the aspect term, opinion term, aspect category, and sentiment polarity, respectively. Generally, the aspect term $a$ and opinion term $o$ are typically text spans in $x$ , and $a$ can also be represented by a specific tag Null if it is not explicitly mentioned in $x$ , i.e., $a \in U _ { x } \cup$ Null and $o \in U _ { x }$ , where $U _ { x }$ is the set of possible continuous spans of $x$ . The aspect category $c$ falls into a predefined category set $U _ { c }$ , and the sentiment polarity $s$ belongs to one of the sentiment class $U _ { s } \in \{ P o { \bar { s i } } t i \nu e , { \dot { N } } e u t r a l , { \dot { N e g } } a t i \nu e \}$ .

# Motivation

With the emergence of PLMs and LLMs, generative models have been widely applied to NLP tasks, including the ASQP task. The goal of generative models is to maximize the likelihood of the next token prediction, formulated as:

$$
\hat { y } _ { t } = p ( y _ { t } \mid p r o m p t _ { x } , y _ { < t } )
$$

where promptx denotes the input sequence that is usually composed of task instruction, demonstrations, and the input sentence $x . y$ denotes the target sequence, and $y _ { < t }$ represents the previous tokens of $y$ before the $t$ -th token. The templates of promptx and $y$ for the ASQP task are introduced in the section of Implementation Details. In our work, we focus on the construction of promptx, which plays a crucial role in activating the reasoning capabilities of generative models, especially for LLMs. More specifically, we attempt to seek suitable types of demonstrations for the ASQP task.

1: Initial vocabulary $V = \emptyset$ . ▷ initialize the vocabulary $V$   
2: for Each sentence $x$ in the Training set $T$ do   
3: $x _ { q }$ denotes the arbitrary word of $a$ or $o$ in $x$ .   
4: Parse $x$ to obtain its CsTs: spa $C y ( x )$ .   
5: for $c s t$ in set $( s p a C y ( x ) )$ do   
6: if $x _ { q }$ in cst & cst not in $V$ then   
7: $\dot { F } ( c s t ) = 0$ . $D$ initialize the frequency of cst   
8: $V = { \dot { V } } \cup \{ c s t \}$ . $D$ add $c s t$ to $V$   
9: end if   
10: Increment $F ( c s t )$ . $D$ count the frequency of $c s t$   
11: end for   
12: end for   
13: Sort $V$ by $F$ in descending order.   
14: Calculate the IDF value of each $c s t$ in $V : I D F ( c s t )$ .   
15: return $V$ .

Retrieval Augmentation Generation (RAG) (Ram et al. 2023; Zhao et al. 2024) is a widely used method for enhancing the effectiveness of generative models by retrieving useful knowledge from external sources or training data (Wang et al. 2022). In the ASQP task, aspect and opinion terms are closely related to sentence syntax. Existing semantic-based retrieval methods (Karpukhin et al. 2020) are not well-suited for retrieving demonstrations with similar syntaxes. To address this gap, we introduce Syn2Vec, a method that extracts syntactic information by representing sentence syntax as vectors. Using Syn2Vec, syntactically similar demonstrations can be retrieved from the training set, similar to how semantically similar demonstrations are retrieved.

# Syn2Vec

Dense retrieval (Karpukhin et al. 2020; Zhao et al. 2024) excels in retrieving similar text from knowledge sources. In this section, we introduce Syn2Vec, a method that encodes syntactic information in sentences as vectors, enabling dense retrieval and then obtaining syntactically similar demonstrations. In a nutshell, Syn2Vec is a function that maps a sentence $x$ to its syntactic vector $v _ { x }$ , represented as:

$$
v _ { x } = \mathrm { S y n 2 V e c } ( x )
$$

where $v _ { x } \in \mathbb { R } ^ { d }$ is a sparse vector with $d$ as the dimension.

In NLP, sentence syntax is often represented by its constituent tree (CT), extracted using tools like spaCy or Stanford NLP. Building on the accurate parsing of CTs, we developed Syn2Vec to represent sentence syntaxes as word vectors based on the vocabulary of CsTs. As outlined in Algorithm 1, the vocabulary construction process is as follows: (1) For a sentence $x$ , we first utilize the spaCy tool to parse its CT, then extract all CsTs using regex matching. This process is represented as $s p a C y ( x )$ , for simplicity. As the example illustrated in Figure 1, a non-leaf node and its descendant nodes form a CsT, e.g., (NP DT JJ NN). (line 4)

(2) We compile all CsTs from training set sentences into the CsT set. For ASQP, we filter CsTs to retain only those aiding aspect-opinion pair extraction, forming a refined CsT vocabulary $V$ . This is done simply by checking whether an aspect or opinion term appears within each CsT. (lines 2-12)

(3) Generally, CsT frequencies indicate how commonly syntax patterns extract aspect-opinion pairs, $i . e .$ , a sentence with a frequent CsT is more likely to yield these pairs. Hence, we sort CsTs by frequency and calculate the Inverse Document Frequency (IDF) of each CsT as its weight. (lines 13-14)

$$
I D F ( c s t ) = \log { ( N / F ( c s t ) ) }
$$

where $N$ is the total number of sentences, and $F ( c s t )$ is the number of sentences containing $c s t$ . The intuition behind using IDF values to weight CsTs is that less frequent CsTs are more informative for distinguishing sentence syntax.

Finally, the word vector of $v _ { x }$ is represented as follows:

$$
v _ { x } = \{ I D F ( V _ { i } ) * C o u n t ( s p a C y ( x ) , V _ { i } ) \} _ { i = 1 } ^ { d }
$$

where $V _ { i }$ is the $i$ -th $\mathrm { C s T }$ in $V$ , and $C o u n t ( )$ is use to count the frequency of $V _ { i }$ in $s p a C y ( x )$ .

# SimRP

Our goal is to retrieve the most effective demonstrations, offering both syntactic and semantic references to enlighten the generative model to generate intact and correct sentiment quadruples. With the developed $\mathrm { S y n 2 V e c }$ , sentences can be represented as syntactic vectors. Due to the sparsity of syntactic vectors, we employ MSE to measure the syntactic similarity between sentences, and retrieve syntactically similar demonstrations for $x$ from the training set $T$ .

$$
\operatorname { S y n S i m } ( x , T ) = \arg \operatorname* { m a x } _ { x _ { i } \in T } \| x ^ { s y n } - x _ { i } ^ { s y n } \| _ { 2 }
$$

where $x ^ { s y n } = { \mathrm { S y n } } 2 { \mathrm { V e c } } ( x )$ represents the syntactic vector of $x$ . $\| * \| _ { 2 }$ denotes the calculation of Euclidean distance, known as the L2 norm. We pick top- $k$ similar samples as syntactically similar demonstrations, denoted as $T ^ { \mathrm { S y n } }$ . Subsequently, we rank demonstrations in $T ^ { \mathrm { S y n } }$ according to their semantic similarity to the input sentence $x$ .

$$
\mathrm { S e m S i m } ( x , T ^ { \mathrm { S y n } } ) = \arg \operatorname* { m a x } _ { x _ { i } \in T ^ { \mathrm { S y n } } } \frac { x ^ { s e m } \cdot x _ { i } ^ { s e m } } { \| x ^ { s e m } \| \| x _ { i } ^ { s e m } \| }
$$

where $x ^ { s e m } = \mathrm { { } } S$ entence-BERT $( x )$ represents the semantic vector of $x$ . We further select top- $k ^ { \prime }$ demonstrations $T _ { \mathrm { s e m } } ^ { \mathrm { S y n } }$ for prompt construction. Hence, the task prompt is achieved:

$$
\mathit { p r o m p t } _ { x } = \{ \mathit { I n s t r u c t i o n } , T _ { \mathrm { S e m } } ^ { \mathrm { S y n } } , x \}
$$

where Instruction refers to the task instruction tailored for the downstream task.

# Experimental Setup

# Datasets

Experiments are carried out on two widely used ASQP datasets, i.e., Rest15 and Rest16, with their statistics shown in Table 1. These datasets are initially constructed based on the SemEval task (Pontiki et al. 2015, 2016) and have undergone multiple annotations (Peng et al. 2020a; Wan et al. 2020). Zhang et al. (2021a) aligned these datasets and served as the standard datasets for the ASQP task finally. Each sentence contains multiple sentiment quadruples, and a predicted quadruple is correct only if all its elements exactly match the gold quadruple. Following previous works (Zhang et al. 2021b; Gou, Guo, and Yang 2023), the F1 score is used as the main evaluation metric in our experiments.

Table 1: Dataset statistics for Rest15 and Rest16. $N$ denotes the number of sentences. Positive, Neutral, and Negative represent the number of quadruples for each sentiment.   

<html><body><table><tr><td rowspan="2">Dataset</td><td colspan="2">Rest15</td><td colspan="2">Rest16</td></tr><tr><td>Train</td><td>Valid Test</td><td>Train Valid</td><td>Test</td></tr><tr><td>N</td><td>834 209</td><td>537</td><td>1264 316</td><td>544</td></tr><tr><td rowspan="3">Positive Neutral</td><td>1005 252</td><td>453</td><td>1369 341</td><td>583</td></tr><tr><td>34 14</td><td>37</td><td>62 23</td><td>40</td></tr><tr><td>315 81</td><td>305</td><td>558 143</td><td>176</td></tr><tr><td>Negative Total</td><td>1354</td><td>347 795</td><td>1989</td><td>507 799</td></tr></table></body></html>

# Implementation Details

We evaluate the proposed SimRP in two paradigms: SFT and ICL. In the SFT paradigm, we fine-tune the PLM using labeled data, while in ICL, we leverage LLMs to perform the ASQP task in zero-shot and few-shot settings. The dimension of the syntactic vector $d$ is set to $2 ^ { 8 }$ in our experiments.

Supervised Fine-Tuning (SFT) We employ the T5-large1 pre-trained model as the backbone. Greedy search decoding is used by default. During the model training, the max epoch number is set to 10, and the batch size is set to 4 for all experiments. AdamW with an initial learning rate of 8e-5 is used as the optimizer, and linear scheduling is applied to adjust the learning rate. Setting $k = 1 0$ in the data preparation stage, i.e., we retrieve the top 10 syntactically similar demonstrations from the training set for each sentence. The numbers of concatenated demonstrations $k ^ { \prime }$ are set to 5 and 1 for rest15 and rest16, respectively. To increase the diversity of training samples, we randomly pick $k ^ { \prime }$ out of $k$ demonstrations during the training stage. While in the testing stage, we concatenated the top $k ^ { \prime }$ similar demonstrations to construct the prompt. Following the successful application of Gou, Guo, and Yang (2023), the target sequence is linearized as $y = [ A ] a [ O ] o [ C ] c [ S ] s [ S S E P ] . . . ,$ , where [A], $[ O ]$ , $\big [ C \big ]$ , and $[ S ]$ are special tokens for differentiating aspect, opinion, category, and sentiment, respectively, and multiple quadruples are concatenated using the special token $[ S S E P ]$ . SFT directly optimizes model parameters to map the input sentence to the target sequence. Therefore, we simply concatenate the input sentence with demonstrations to construct the prompt, omitting any instructional description. Some examples of prompts are shown in the Appendix. We employ the standard cross entropy loss to optimize the parameters of the generative model:

$$
\mathcal { L } = \sum _ { t = 1 } ^ { n } y _ { t } \mathrm { l o g } ( \hat { y } _ { t } )
$$

where $n$ is the length of the target sequence $y , y _ { t }$ is a one-hot vector representing the ground truth of $t$ -th word in $y$ , and $\hat { y } _ { t }$ denotes the corresponding predicted probability distribution.

Recent advanced ASQP models are compared to evaluate our proposed SimRP, including GAS (Zhang et al. 2021b),

Paraphrase (Zhang et al. 2021a), DLO/ILO (Hu et al. 2022), MVP (Gou, Guo, and Yang 2023), GenDa (Wang et al. 2023a), MUL (Hu et al. 2023), OTCL (Li et al. 2024), and IVLS (Nie et al. 2024), where OTCL and IVLS are nongenerative methods, and the others are generative methods.

In-context Learning (ICL) ICL is a crucial technique for LLMs to tackle complex tasks based on a few annotated demonstrations without additional training or gradient updates. Our proposed SimRP emphasizes demonstration selection for ICL, making it particularly well-suited for using LLMs to tackle the ASQP task. Following the instruction template in Zhang et al. (2024b), we construct the prompt for LLMs by the proposed SimRP. Some cases of prompts are shown in the Appendix due to space limitations. Experiments are conducted based on GPT- $4 0 ^ { 2 }$ (i.e., GPT-4omini and GPT-4o) and Llama- $3 . 1 ^ { 3 }$ (i.e., Llama-3.1-8B and Llama-3.1-70B) models under zero-shot and few-shot settings. In the ICL paradigm, $k$ is also set to 10, and the most similar demonstrations are used to construct the prompt.

Compared methods including: ThoR (Fei et al. 2023) designed a three-step prompting principle to induce the implicit aspect, opinion, and sentiment polarity step by step. LLMs for SA (Zhang et al. 2024b) is a comprehensive study of sentiment analysis tasks using LLMs, where samples are randomly added with diverse demonstrations.

# Results and Analysis in the SFT Paradigm Main Results

We execute experiments three times with fixed seeds of [2024, 2025, 2026], and report the mean values in Table 2. The best results are marked in bold and the second-best underlined. Overall, our approach greatly outperforms existing methods, with an average F1 score improvement of $2 . 2 0 \%$ .

Additionally, we have the following observations. (1) ASQP remains challenging for generative models, with conversational non-generative models (e.g., OTCL and IVLS) generally outperforming them, particularly IVLS, which performs best on both datasets. (2) Generative models possess great potential for improvement, as evidenced by the substantial performance gains achieved by MvP, which optimize the output views of different quadruple-generation orders. (3) Based on the achievements of existing generative methods, SimRP focuses on the optimization of input sequences, providing more syntactically and semantically similar demonstrations for the input, which significantly activates the reasoning capabilities of generative models. (4) The performance of generative models for ASQP is underestimated, the more flexible and comprehensive CsT-based fuzzy metric is better suited for evaluation.

# Impacts of Demonstration Number

In this section, we investigate the impacts of integrated demonstrations in the SFT paradigm, experimental results are illustrated in Figure 2. It can be found that demonstrations play a crucial role in enhancing the performance of generative models, especially on Rest15. However, more demonstrations seem useless, as the number of demonstrations increases, the results on Rest15 no longer enhanced, while on Rest16 significantly declined. The potential reason is that retrieved demonstrations have been exposed to the model during training, and the model has learned the mapping from inputs to outputs after fine-tuning. In the inference stage, the model may be confused by more retrieved demonstrations, leading to a performance decline.

<html><body><table><tr><td rowspan="2">Methods</td><td colspan="3">Rest15</td><td colspan="3">Rest16</td><td rowspan="2">Avg.(F1)</td></tr><tr><td>Pre</td><td>Rec</td><td>F1</td><td>Pre</td><td>Rec</td><td>F1</td></tr><tr><td>GASt (Zhang et al. 2021b)</td><td>45.31</td><td>46.70</td><td>45.98</td><td>54.54</td><td>57.62</td><td>56.04</td><td>51.01</td></tr><tr><td>Paraphrase (Zhang et al. 2021a)</td><td>46.16</td><td>47.72</td><td>46.93</td><td>56.63</td><td>59.30</td><td>57.93</td><td>52.43</td></tr><tr><td>DLO (Hu et al. 2022)</td><td>47.08</td><td>49.33</td><td>48.18</td><td>57.92</td><td>61.80</td><td>59.79</td><td>53.99</td></tr><tr><td>ILO (Hu et al. 2022)</td><td>47.08</td><td>50.38</td><td>49.05</td><td>57.58</td><td>61.17</td><td>59.32</td><td>54.19</td></tr><tr><td>MvP (Gou, Guo,and Yang 2023)</td><td></td><td></td><td>51.04</td><td></td><td></td><td>60.39</td><td>55.72</td></tr><tr><td>GenDA(Wang et al. 2023a)</td><td>49.74</td><td>50.29</td><td>50.01</td><td>60.08</td><td>61.70</td><td>60.88</td><td>55.45</td></tr><tr><td>MUL (Hu et al. 2023)</td><td>49.12</td><td>50.39</td><td>49.75</td><td>59.24</td><td>61.75</td><td>60.47</td><td>55.11</td></tr><tr><td>OTCL (Li et al. 2024)</td><td>47.86</td><td>50.77</td><td>49.27</td><td>58.31</td><td>62.02</td><td>60.11</td><td>54.69</td></tr><tr><td>IVLS (Nie et al. 2024)</td><td>54.46</td><td>48.53</td><td>51.28</td><td>62.69</td><td>59.75</td><td>61.04</td><td>56.16</td></tr><tr><td>SimRP (Ours)</td><td>53.12</td><td>53.50</td><td>53.30↑2.02</td><td>62.74</td><td>64.12</td><td>63.42个2.38</td><td>58.36↑2.20</td></tr><tr><td>usingCsT-based fuzzymetric</td><td>56.97</td><td>57.61</td><td>57.29</td><td>68.37</td><td>68.46</td><td>68.42</td><td>63.00</td></tr></table></body></html>

Table 2: Comparison results in terms of precision (Pre, $\%$ ), recall (Rec, $\%$ ) and F1 score (F1, $\%$ ), with the best in bold and the second best underlined. † denotes the results quoted from Wang et al. (2023a), the others are cited from their original papers.

![](images/d3289ee8ea8bb112cfa776ccb63ca63f988e586bde7a8d30fe71ad8a1ab21e36.jpg)  
Figure 2: Impact of demonstration number $k ^ { \prime }$ .

Table 3: Ablation studies in the SFT paradigm.   

<html><body><table><tr><td>Methods</td><td>Rest15</td><td>Rest16</td><td>Avg.(F1)</td></tr><tr><td>SimRP</td><td>53.30</td><td>63.42</td><td>58.36</td></tr><tr><td>- [Syn.]</td><td>52.58↓0.72</td><td>62.97↓0.45</td><td>57.78↓0.58</td></tr><tr><td>- [Sem.]</td><td>52.44↓0.86</td><td>62.83↓0.59</td><td>57.64↓0.72</td></tr><tr><td>- [Syn., Sem.]</td><td>51.00↓2.30</td><td>62.18↓1.24</td><td>56.59↓1.77</td></tr></table></body></html>

# Ablation Study

Furthermore, we conducted ablation studies to evaluate the roles of various types of retrieval demonstrations. Experimental results are tabulated in Table 3, where ”-[Syn.]” denotes the removal of SynSim, namely using semantically similar demonstrations only. Similarly, ”-[Sem.]” denotes the removal of SemSim, namely using syntactically similar demonstrations only. ”-[Syn., Sem.]” means that demonstrations are randomly selected from the training set.

From Table 3, both syntactically and semantically similar demonstrations contribute to performance improvement, with semantically similar ones playing a more crucial role, as removing them causes a greater decline.

# Results and Analysis in the ICL Paradigm Main Results

Table 4 reports the experimental results of using LLMs to address the ASQP task. Experiments are executed under zero-shot and few-shot settings. We can observe that: (1) Longitudinally, as the backbone model size increases, LLM performance in ASQP improves significantly, e.g., Llama3.1-70B and GPT-4o outperform their lower scaled versions greatly. GPT-4o achieves the best results among all LLMs under different settings. (2) Transversely, as the number of demonstrations increases, the accuracy of quadruple generation improves across all models, except for lower scaled models in ”LLMs for SA”, whose performance tends to decrease with a single demonstration. (3) Comparison among these three methods, SimRP surpasses the other methods in all settings, especially equipped large-scale models as the backbone. Demonstrating the superiority of syntactically and semantically demonstrations for the ASQP task.

# Ablation Study

Similarly, we execute ablation studies to investigate the roles of different types of demonstrations in the ICL paradigm. Experimental results of GPT-4o are tabulated in Table 5, where ”Syn.” and ”Sem.” share the same meanings as those in Table 3. It’s noted that we have the same observations as in the SFT paradigm, where semantically similar demonstrations are more important than syntactically similar ones. Syntactically retrieval demonstrations also play a crucial role in enhancing the performance of LLMs, as removing ”Syn.” leads to a significant performance decline, especially under the 1-shot setting. These observations emphasize the importance of syntactically and semantically similar demonstrations in activating LLMs’ reasoning in ASQP.

Table 4: Experimental results of LLMs under zero-shot and few-shot settings.   

<html><body><table><tr><td rowspan="2">Methods</td><td rowspan="2">Backbone</td><td colspan="4">Rest15</td><td colspan="4">Rest16</td></tr><tr><td>O-shot</td><td>1-shot</td><td>5-shot</td><td>10-shot</td><td>0-shot</td><td>1-shot</td><td>5-shot</td><td>10-shot</td></tr><tr><td rowspan="4">THOR (Fei et al. 2023)</td><td>Llama-3.1-8B</td><td>7.88</td><td>8.02</td><td>8.65</td><td>10.01</td><td>9.44</td><td>10.78</td><td>11.37</td><td>11.95</td></tr><tr><td>Llama-3.1-70B</td><td>17.62</td><td>22.49</td><td>26.78</td><td>31.47</td><td>27.61</td><td>30.04</td><td>34.34</td><td>35.91</td></tr><tr><td>GPT-4o-mini</td><td>18.18</td><td>21.52</td><td>24.29</td><td>29.36</td><td>29.83</td><td>31.15</td><td>32.84</td><td>34.05</td></tr><tr><td>GPT-40</td><td>30.79</td><td>33.37</td><td>35.12</td><td>36.81</td><td>37.23</td><td>38.92</td><td>42.05</td><td>43.57</td></tr><tr><td rowspan="4">LLMs for SA (Zhang et al. 2024b)</td><td>Llama-3.1-8B</td><td>8.61</td><td>8.82</td><td>8.66</td><td>10.69</td><td>11.43</td><td>9.99</td><td>11.67</td><td>12.59</td></tr><tr><td>Llama-3.1-70B</td><td>18.20</td><td>20.79</td><td>27.81</td><td>32.84</td><td>29.01</td><td>30.99</td><td>35.04</td><td>37.10</td></tr><tr><td>GPT-4o-mini</td><td>19.80</td><td>18.92</td><td>24.93</td><td>30.11</td><td>31.43</td><td>29.42</td><td>33.84</td><td>35.45</td></tr><tr><td>GPT-40</td><td>34.56</td><td>35.62</td><td>36.29</td><td>37.08</td><td>39.15</td><td>39.61</td><td>43.36</td><td>45.00</td></tr><tr><td rowspan="4">SimRP (Ours)</td><td>Llama-3.1-8B</td><td>8.88</td><td>9.26</td><td>14.13</td><td>15.26</td><td>12.55</td><td>15.46</td><td>16.71</td><td>21.60</td></tr><tr><td>Llama-3.1-70B</td><td>18.12</td><td>26.83</td><td>35.56</td><td>38.19</td><td>28.71</td><td>33.73</td><td>40.91</td><td>43.81</td></tr><tr><td>GPT-40-mini</td><td>19.47</td><td>26.46</td><td>33.33</td><td>35.55</td><td>31.25</td><td>39.93</td><td>40.44</td><td>43.13</td></tr><tr><td>GPT-40</td><td>34.44</td><td>37.88</td><td>41.08</td><td>43.17</td><td>39.33</td><td>45.50</td><td>48.62</td><td>49.74</td></tr></table></body></html>

Table 5: Ablation studies in the ICL paradigm.   

<html><body><table><tr><td rowspan="2">Methods</td><td colspan="2">Rest15</td><td colspan="2">Rest16</td></tr><tr><td>1-shot</td><td>10-shot</td><td>1-shot</td><td>10-shot</td></tr><tr><td>SimRP</td><td>37.88</td><td>43.17</td><td>45.50</td><td>49.74</td></tr><tr><td>-[Syn.]</td><td>36.48</td><td>42.57</td><td>44.27</td><td>49.39</td></tr><tr><td>-[Sem.]</td><td>35.45</td><td>40.52</td><td>42.56</td><td>48.15</td></tr><tr><td>-[Syn., Sem.]</td><td>35.65</td><td>36.82</td><td>38.03</td><td>44.64</td></tr></table></body></html>

![](images/71b6bc0242325a132e3e2e3d087a9388a5195a9b6b81977e4deb49ea69e97cb4.jpg)  
Figure 3: Influences of different syntactic vector dimensions.

# Influence of $d$ in Syn2Vec

As illustrated in Equation (4), the dimension of the syntactic vector $d$ is a crucial hyperparameter for accurately representing sentence syntax, which further influences the retrieval of syntactically similar demonstrations. Hence, we conduct the parameter searching experiments to acquire the optimal $d$ . Figure 3 presents the experimental results of GPT-4o on Rest15 under the 10-shot setting, with $d$ varying from $2 ^ { 5 }$ to $2 ^ { 1 1 }$ . The model’s performance clearly improves as $d$ increases, peaking at $d = 2 ^ { 8 }$ , after which it declines.

# Error Analysis

In this section, we conduct an error analysis to investigate the capability of LLM in ASQP. Table 6 presents several examples from Rest15, where ID denotes the sample index, and predictions are generated by GPT-4o under the 10-shot setting. From Table 6, we can observe that LLM correctly identifies ”wings” in the 4th case but is marked incorrect for omitting ”with chimichurri”. The last four cases show similar situations where LLM generates correct but more detailed opinion terms. It’s worth noting that the 29th case is misannotated, where the opinion term should be ”wasn’t a fan” rather than ”fan” due to the negative sentiment. These observations highlight the rationality of LLMs’ predictions and demonstrate that existing exact-matching metrics fail to fully reflect the model’s performance.

Generally, detailed opinion words, co-occurred under a CsT, are rational and reasonable in the ASQP task. Hence, we propose a CsT-based fuzzy matching metric. We first extend the ground truth of aspect and opinion terms according to their CsTs. As the example illustrated in Figure 1, the labeled opinion term ”fair” is extended to ”a fair tab” according to the CsT of (NP DT JJ NN). A predicted quadruple is considered correct if the aspect or opinion terms are included in the extended ground truth. More examples are provided in the Appendix. The performance of GPT-4o is re-evaluated using the CsT-based fuzzy metric, and the results are presented in Table 7. $\ " \mathrm { A d d _ { \pm w } } \ '$ in the table represents the simple fuzzy metric that extends the ground truth by adding $w$ words forward and backward, respectively. Thus, $\mathrm { \ " { A d d } _ { \pm 0 } \mathrm { ? } }$ is the original exact matching metric. For a fair comparison, we report the average number of words added by the CsTbased fuzzy metric, shown at subscripts of results.

From Table 7, it can be found that the proposed CsTbased fuzzy metric significantly outperforms the other fuzzy matching metrics, with fewer adding words and higher F1 scores. In addition, using the CsT-based fuzzy metric, GPT4o under the 10-shot setting has achieved performance comparable to fine-tuned generative models on Rest15.

# Related Work

# Aspect-based Sentiment Analysis

ABSA is a fine-grained sentiment analysis task and has evolved into a series of subtasks, including Aspect-level sentiment Classification (ALSC) (Tang, Qin, and Liu 2016), Aspect Opinion Pair Extraction (AOPE) (Zhao et al. 2020), Aspect Sentiment Triplet Extraction (ASTE) (Peng et al. 2020b), and ASQP (Zhang et al. 2021b). ASTE and ASQP, being more challenging compound tasks, have gained increased attention in recent years, proposing various endto-end methods to address them. These methods include span-based methods (Cai, Xia, and Yu 2021; Xu, Chia, and Bing 2021), table-filling methods (Wu et al. 2020; Chen et al. 2022), machine reading comprehension-based methods (Mao et al. 2021; Chen et al. 2021), and generative methods (Zhang et al. 2021b; Gao et al. 2022; Gou, Guo, and Yang 2023). Among them, generative methods have gradually become the mainstream due to their implementation simplicity, task universality and ability to exploit rich label information.

<html><body><table><tr><td>ID</td><td>Sentence</td><td>Ground Truth</td><td>Predictions</td></tr><tr><td>4</td><td>Guacamole at pacifico is yummy, asare thewingswith chimichurri.</td><td>(Guacal positive, FQ)</td><td>(Guacamole, yummy, positive, FQ) (wings, yummy, positive, FQ)</td></tr><tr><td>6</td><td>In the end you end up with a fair tab and nothing but a great time !!</td><td>(Null, fair, positive,RP) (Null, great, positive,RG)</td><td>(Null,a fair tab,positive,RP) (Null, a great time,positive,RG)</td></tr><tr><td>29</td><td>Also, I personally wasn't a fan of the portobello and asparagus mole.</td><td>(portobello and asparagus mole, fan,negative, FQ)</td><td>(portobello and asparagus mole, wasn't a fan, negative, FQ)</td></tr><tr><td>37</td><td>I'd definitely go back again.</td><td>(Null,go back, positive,RG)</td><td>(Null, go back again, positive, RG)</td></tr><tr><td>48</td><td>She just nodded and walked off.</td><td>(Null, walked off, negative,SG)</td><td>(Null, nodded and walked off, negative, SG)</td></tr></table></body></html>

Table 6: Error analysis on the Rest15 dataset. FQ, RP, RG, FQ, and SG are abbreviations of different categories.

Table 7: Experimental results by fuzzy matching metrics.   

<html><body><table><tr><td rowspan="2">Metrics</td><td colspan="2">Rest15</td><td colspan="2">Rest16</td></tr><tr><td>1-shot</td><td>10-shot</td><td>1-shot</td><td>10-shot</td></tr><tr><td>Add±0</td><td>37.88</td><td>43.17</td><td>45.50</td><td>49.74</td></tr><tr><td>Add±1</td><td>44.28</td><td>51.49</td><td>52.63</td><td>56.49</td></tr><tr><td>Add±2</td><td>47.76</td><td>55.31</td><td>55.46</td><td>60.03</td></tr><tr><td>Add±3</td><td>49.39</td><td>57.67</td><td>57.50</td><td>61.52</td></tr><tr><td>Add±4</td><td>51.13</td><td>59.13</td><td>58.97</td><td>62.78</td></tr><tr><td>CsT.</td><td>53.46+3.0 -1.8</td><td>59.25+2.9 -2.0</td><td>59.20+3.0 -3.1</td><td>63.12+2.8 -2.6</td></tr></table></body></html>

The core idea of generative ASQP is to transform sentiment elements into a label sequence and then use the seq2seq paradigm to learn the matching relationships between the input text and the label sequence. For instance, Zhang et al. (2021a) introduced a paraphrase modeling framework, transforming the quadruple prediction task into a text generation task by using paraphrase sentences as target sequences. MVP (Gou, Guo, and Yang 2023) explored the impact of quadruple generation order on the model’s performance and proposed to vote the most reasonable quadruples generated in different orders, significantly enhancing the model’s predictive capability.

# In-Context Learning for ASQP

ICL is one of the key techniques in which LLMs can tackle complex tasks based on a few annotated demonstrations without additional training or gradient updates (Zhao et al. 2023). Hence, demonstrations in ICL play the most crucial role in activating the reasoning capabilities of LLMs. Numerous studies focused on how to provide better demonstrations for LLMs. Representatively, Liu et al. (2022) found that examples closely related to the target data in the embedding space yield better results. Building on this idea, Wang et al. (2022) proposed enhancing inputs by retrieving similar examples from the training set using BM25.

Owing to the developments LLM and ICL, addressing complex ABSA subtasks, such as ASQP, has become more feasible (Wang et al. 2023b). Zhong et al. (2023) observed that the zero-shot performance of LLMs is comparable to fine-tuned BERT models on the sentiment analysis tasks. Zhang et al. (2024b) further demonstrated that LLMs can match the performance of smaller models specifically trained for simple sentiment analysis tasks.

# Conclusion

In this paper, we present Syn2Vec, which encodes sentence syntax as sparse vectors and uses MSE to compute syntactic similarity, allowing the retrieval of syntactically similar demonstrations. Subsequently, Sentence-BERT is utilized to select the semantically similar ones as the most related demonstrations. With these syntactically and semantically similar demonstrations, the proposed SimRP significantly enhances the reasoning capabilities of generative models. Experimental results on the standard ASQP datasets demonstrate that SimRP outperforms existing methods, both in SFT and ICL paradigms. Furthermore, we find that LLMs performance is significantly underestimated by the exact matching evaluation metric, as their generated results are logical but may differ from annotated labels, which are often subjective and non-unique. Therefore, we propose a CsTbased fuzzy metric to better evaluate the performance of LLMs in the ASQP task. Comparison results demonstrate the superiority of the proposed CsT-based fuzzy metric.