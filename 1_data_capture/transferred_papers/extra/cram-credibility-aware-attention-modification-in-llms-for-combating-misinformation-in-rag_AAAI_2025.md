# CrAM: Credibility-Aware Attention Modification in LLMs for Combating Misinformation in RAG

Boyi Deng1, Wenjie $\mathbf { W a n g ^ { 2 * } }$ , Fengbin $\mathbf { Z } \mathbf { h } \mathbf { u } ^ { 2 * }$ , Qifan Wang3, Fuli Feng1

1University of Science and Technology of China, 2National University of Singapore, 3Meta AI dengboyi $@$ mail.ustc.edu.cn, wqfcr $@$ fb.com,   
wenjiewang96, zhfengbin, fulifeng93 $@$ gmail.com

# Abstract

Retrieval-Augmented Generation (RAG) can alleviate hallucinations of Large Language Models (LLMs) by referencing external documents. However, the misinformation in external documents may mislead LLMs’ generation. To address this issue, we explore the task of “credibility-aware RAG”, in which LLMs automatically adjust the influence of retrieved documents based on their credibility scores to counteract misinformation. To this end, we introduce a plug-and-play method named Credibility-aware Attention Modification (CrAM). CrAM identifies influential attention heads in LLMs and adjusts their attention weights based on the credibility of the documents, thereby reducing the impact of low-credibility documents. Experiments on Natual Questions and TriviaQA using Llama2-13B, Llama3-8B, and Qwen1.5-7B show that CrAM improves the RAG performance of LLMs against misinformation pollution by over $20 \%$ , even surpassing supervised fine-tuning methods.

Question (Q): RAG   
Who won the first Nobel Prize   
in Physics？ Q Answer: D1 LLM Albert   
Document 1 (D1): Einstein. The first Nobel Prize in Physics D2   
was awarded to physicist   
Wilhelm Röntgen.   
Credibility score (S1): 0.8 Q credibility-aware RAG D1   
Document 2 (D2): Answer: Actually, it is Albert Einstein S1 LLM Wilhelm who won the first Nobel Prize Röntgen. in Physics. D2   
Credibility score (S2): 0.1   
S2

Figure 1: A comparison between RAG and credibility-aware RAG. Credibility-aware RAG considers credibility to reduce the impact of low-credibility documents.

# 1 Introduction

Retrieval-Augmented Generation (RAG) (Gao et al. 2024; Zhu et al. 2021) is a representative approach to mitigate hallucination issues of Large Language Models (LLMs) (Zhang et al. 2023) by retrieving and referencing relevant documents from an external corpus. Despite its effectiveness, most RAG works overlook a crucial issue: misinformation pollution in the external corpus (Pan et al. 2023b; Dufour et al. 2024). The maliciously generated misinformation may mislead LLMs to produce unfaithful responses. For instance, Microsoft’s Bing can be misled by misinformation on the internet to generate incorrect information for Bing users (Vincent 2023). Besides, Pan et al. (2023b) and Pan et al. (2023a) demonstrated that inserting LLM-generated misinformation into the RAG corpus can significantly degrade LLMs’ performance. Therefore, addressing the misinformation pollution for RAG is essential.

A straightforward and common idea to address this misinformation pollution issue is misinformation detection and filtering. Extensive misinformation detection works focus on measuring the credibility of documents, i.e., the probability of the document not containing misinformation. And these works have achieve significant results (Kaliyar, Goswami, and Narang 2021; Pelrine et al. 2023; Quelle and Bovet 2024; Li et al. 2024). Once we obtain the credibility of each retrieved document, we can exclude those with credibility below a certain threshold before using them in RAG. However, directly discarding certain documents may result in the loss of relevant and important information, leading to performance degradation (Yoran et al. 2024)1. Therefore, given the remarkable advancements in the measurement of credibility scores and the relatively underdeveloped mechanisms for utilizing these scores, it is essential to explore how these scores can be effectively utilized by LLMs, assuming that high-quality credibility scores are accessible.

To achieve this, we focus on a task named “credibilityaware RAG” as shown in Figure 1. Specifically, given a user query $x$ with a list of relevant documents $\begin{array} { l l l } { { \mathcal { D } } } & { { = } } & { { \{ \bar { d } _ { 1 } , \bar { d } _ { 2 } , . . . , d _ { n } \} } } \end{array}$ and $\mathcal { D }$ ’s credibility scores $\begin{array} { r l } { S } & { { } = } \end{array}$ $\{ s _ { 1 } , s _ { 2 } , . . . , s _ { n } \}$ , credibility-aware RAG requests LLMs to automatically adjust the influence of documents in $\mathcal { D }$ on the generated output $y$ based on their credibility scores in $s$ . Initial attempts on credibility-aware RAG adopted supervised fine-tuning (SFT) to teach LLMs to distinguish the importance of different documents in the prompt by their credibility scores (Hong et al. 2024; Pan et al. 2024). However, SFT requires additional computational resources and welldesigned training data, which limits the application scenarios. Therefore, we explore non-SFT method for LLMs to attain credibility-aware RAG.

Given that the attention mechanism serves as the central component for adjusting the significance of various input data, we consider manipulating attention weights of LLMs to achieve credibility-aware RAG. In particular, we adjust attention weights according to credibility scores in the inference stage of LLMs. In this way, we can regulate LLMs to pay less “attention” to less credible documents by decreasing the corresponding attention weights. Moreover, previous studies (Clark et al. 2019; Elhage et al. 2021; Voita et al. 2019) have indicated that different attention heads exhibit distinct patterns and functions, resulting in varying impacts on LLMs’ outputs. In this context, the key lies in identifying a subset of influential attention heads for attention weight modification.

In this work, we propose a plug-and-play method named Credibility-aware Attention Modification (CrAM), which identifies the influential attention heads and then modifies their attention weights w.r.t. different document tokens to reduce the impact of low-credibility documents. Specifically, 1) influential head identification: we select top-ranked attention heads according to an extended causal tracing method (Meng et al. 2022) that estimates the contribution of each attention head to generating incorrect answers over a small dataset. 2) Attention weight modification: we scale down the attention weights of the retrieved documents based on their normalized credibility scores.

We conduct extensive experiments on two open-domain Question Answering (QA) datasets, Natual Questions (NQ) (Kwiatkowski et al. 2019) and TriviaQA (Joshi et al. 2017), using three open-source LLMs: Llama2-13B (Touvron et al. 2023), Llama3-8B (Meta 2024), and Qwen1.5-7B (Bai et al. 2023). The results show that CrAM significantly alleviates the influence of misinformation documents on RAG, in terms of both ideal credibility scores and GPT-generated credibility scores. It is worth noting that CrAM even outperforms the SFT-based method CAG (Pan et al. 2024) in most scenarios, demonstrating the superiority of CrAM. We release our code at https://github.com/Aatrox103/CrAM.

In summary, our main contributions are:

• We explore the task of credibility-aware RAG without fine-tuning LLMs to alleviate the misinformation pollution issue.   
• We develop a plug-and-play method, CrAM, which identifies influential attention heads and modifies their attention weights to equip LLMs with credibility-aware RAG capabilities.   
• We conduct extensive experiments with two QA datasets on three LLMs using ideal credibility scores and GPTgenerated credibility scores, validating the superiority of CrAM.

# 2 Credibility-Aware RAG

Given a user query $x$ , RAG retrieves a set of documents $\mathcal { D } =$ $\{ d _ { 1 } , d _ { 2 } , \dots , d _ { n } \}$ relevant to $x$ through a retriever (Gao et al. 2024). Then the relevant documents $\mathcal { D }$ are evaluated by a credibility estimator2, obtaining their credibility scores ${ \boldsymbol { S } } =$ $\{ s _ { 1 } , s _ { 2 } , \ldots , s _ { n } \}$ , which represents the probability of each document not containing misinformation.

Credibility-Aware RAG. Given an LLM $L$ , a user query $x$ , and relevant documents $\mathcal { D }$ associated with credibility scores $s$ , the objective of credibility-aware RAG is to enable LLMs to automatically adjust the influence of these documents on the generated output $y$ based on their credibility scores $s$ . This can be formally defined as:

$$
\operatorname* { m a x } \mathrm { ~ M e t r i c } ( \operatorname { C o m b i n e } ( L , x , \mathcal { D } , \mathcal { S } ) ) ,
$$

where Combine $( \cdot )$ represents the method or mechanism to integrate credibility scores into the generation process of $L$ . For example, Pan et al. (2024) employ SFT to finetune LLMs to capture the credibility difference of documents more effectively, denoted as Combine $( L , x , \mathcal { D } , \mathcal { S } ) =$ $L _ { S F T } ( x , \mathcal { D } , S )$ . Additionally, Metric( ) is a function that assesses whether documents with different credibility scores have varying impacts on the output of $L$ . Indeed, we can utilize the performance of generating factual answers to measure Metric( ). For instance, we use the accuracy of QA tasks to approximate $\operatorname { M e t r i c } ( \cdot )$ in this work. The rationality is that if the impact of low-credibility documents decreases, the accuracy of QA tasks should increase accordingly.

# 3 CrAM

CrAM first identifies influential attention heads, and then modifies the attention weights of these identified heads to reduce the impact of low-credibility documents as shown in Figure 2. Since influential attention heads identification process involves attention weight modification, we first explain the procedure of attention weight modification in Section 3.1, and then describe influential attention heads identification in Section 3.2. Finally, we summarize the overall CrAM workflow in Section 3.3.

# 3.1 Attention Weight Modification

As defined in Section 2, the objective of credibility-aware RAG is to reduce the impact of low-credibility documents on the generated output of LLMs. Intuitively, it requires LLMs to pay less “attention” to low-credibility documents. To this end, a natural approach is scaling down the corresponding attention weights of low-credibility documents.

For RAG, a user query $x$ and a set of relevant documents $\mathcal { D } = \{ d _ { 1 } , d _ { 2 } , \dots , d _ { n } \}$ should be concatenated and tokenized into a token sequence $\mathcal { T } ( x , \mathcal { D } ) = \{ t _ { 1 } , t _ { 2 } , . . . , t _ { m } \}$ , where $t _ { k }$ denotes the $k$ -th token. Given the credibility scores for each document $\boldsymbol { S } = \{ s _ { 1 } , s _ { 2 } , . . . , s _ { n } \}$ , the normalized credibility score for token $t _ { k }$ can be calculated as follows:

$$
\bar { s } _ { k } = \left\{ \begin{array} { l l } { \frac { s _ { i } - \operatorname* { m i n } ( S ) } { \operatorname* { m a x } ( S ) - \operatorname* { m i n } ( S ) } } & { { \mathrm { i f ~ } } t _ { k } { \mathrm { ~ b e l o n g s ~ t o ~ } } d _ { i } } \\ { 1 } & { { \mathrm { o t h e r w i s e } } } \end{array} , \right.
$$

![](images/9a4137dea139ec1259f1ac66736c653a210ca86370d4418e64c572603eb918d1.jpg)  
Figure 2: Illustration of CrAM. Compared to RAG, CrAM first identifies influential attention heads and then modifies thei attention weights based on the credibility scores of each document.

where $s _ { i }$ is subtracted by $\operatorname* { m i n } ( S )$ , and then scaled down by $1 / ( \operatorname* { m a x } ( S ) - \operatorname* { m i n } ( S ) )$ to ensure all credibility scores are normalized to $[ 0 , 1 ]$ . Besides, we define $\overline { { \bf s } } = [ \bar { s } _ { 1 } , \ldots , \bar { s } _ { m } ] \in$ $\mathbb { R } ^ { 1 \times m }$ as the normalized credibility scores of the whole token sequence $\tau ( x , \mathcal { D } )$ .

For each attention head $h$ in LLM, ${ \bf A } _ { h }$ represents its attention weights matrix3. Let $( { \bf A } _ { h } ) _ { k }$ represent the $k$ -th row vector4 of ${ \bf A } _ { h }$ , we can obtain the modified attention weight matrix $\mathbf { A } _ { h } ^ { * }$ by element-wise multiplying s¯ as follows:

$$
\begin{array} { r } { ( \mathbf { A } _ { h } ) _ { k } ^ { * } = \operatorname { N o r m } ( ( \mathbf { A } _ { h } ) _ { k } \odot \overline { { \mathbf { s } } } ) , k \in \{ 1 , \ldots , m \} , } \end{array}
$$

where $\odot$ denotes the element-wise multiplication of vectors. The Norm function refers to $\ell _ { 1 }$ normalization, which ensures that the attention weights sum to one.

# 3.2 Influential Head Identification

Previous works Clark et al. (2019); Elhage et al. (2021); Voita et al. (2019) have found that different attention heads exhibit various patterns and functions, leading to different impacts on LLMs’ output. As such, we hypothesize that some attention heads have a larger impact on using misinformation documents to generate incorrect answers. Previously, causal tracing (Meng et al. 2022) has been developed to quantify the contribution of each hidden state towards generating given answers. The contribution is measured by adding noises to each hidden state to compare the changes in the generation probability of the given answer. In light of this, CrAM revises causal tracing to evaluate the contribution of attention heads instead of hidden states. Utilizing attention weight modification, as detailed in Section 3.1, CrAM estimates the change in probability of generating incorrect answers to determine the contribution of each attention head. Thereafter, CrAM ranks all attention heads by contributions and identifies influential ones.

Specifically, the contribution of one attention head $h$ can be obtained as follows:

• Given an LLM $L$ , a user query $x$ , a set of relevant documents $\mathcal { D } = \{ d _ { m i s } , d _ { 1 } , d _ { 2 } , \ldots , d _ { n } \}$ with one misinformation document $d _ { m i s }$ , and an incorrect answer $a _ { w r o n g }$ to $x$ that is supported by $d _ { m i s }$ , we first calculate the generation probability of $a _ { w r o n g }$ with $x$ and $\mathcal { D }$ by $L$ . Formally, we have:

$$
P _ { 0 } = P _ { L } ( a _ { w r o n g } \mid x , \mathcal { D } ) .
$$

• Next, we modify a specific attention head as described in Section 3.1 by using the credibility scores $\begin{array} { r l } { S } & { { } = } \end{array}$ $\{ 0 , 1 , 1 , \ldots , 1 \}$ of $\mathcal { D }$ and recalculate the generation probability of awrong:

$$
P _ { 1 } = P _ { L _ { h } ^ { * } } ( a _ { w r o n g } \mid x , \mathcal { D } ) ,
$$

where $L _ { h } ^ { * }$ denotes the LLM $L$ whose attention weight matrix of the attention head $h$ is modified according to Equation (1).

• Finally, we quantify the contribution of head $h$ towards generating the incorrect answer, $a . k . a .$ . the indirect effect (IE) (Meng et al. 2022):

$$
\mathrm { I E } _ { h } = P _ { 0 } - P _ { 1 } ,
$$

which can also be interpreted as the decrease in the generation probability of the incorrect answer $a _ { w r o n g }$ after modifying head $h$ .

To improve the robustness of the contribution estimation, we utilize a small dataset $\{ ( x , a _ { w r o n g } , D , S ) , \ldots \}$ that do not overlap with the test data to compute the average IE for each attention head (refer to Section 4.3 for robustness analysis). Thereafter, we can calculate IEs for all the attention heads and rank them to select the top-ranked ones with larger IEs for attention weight modification.

# 3.3 CrAM Workflow

The CrAM workflow is summarized as follows:

• First, we use a small dataset with misinformationpolluted documents to calculate the average IE for each attention head in an LLM as described in Section 3.2. Then, we rank all attention heads by their IEs in descending order and select the top-ranked heads as influential attention heads. • Given any user query, along with the relevant documents and credibility scores, we modify the attention weights of influential attention heads using the method described in Section 3.1 to obtain the final answer, thereby significantly reducing the impact of low-credibility documents.

# 4 Experiments

# 4.1 Experimental Settings

Datasets, LLMs and Metrics. We conduct experiments over the Natural Questions (NQ) (Kwiatkowski et al. 2019) and TriviaQA (Joshi et al. 2017) datasets with three LLMs, i.e. Llama2-13B (Touvron et al. 2023), Llama3-8B (Meta 2024), and Qwen1.5-7B (Bai et al. 2023). We adopt Exact Match (EM) and F1 score as evaluation metrics, which are widely used in the QA setting (Karpukhin et al. 2020; Rajpurkar et al. 2016; Chen et al. 2017).

Document Preparation. We prepare both high-credibility and low-credibility documents (i.e., with misinformation) associated with the questions for evaluating the proposed method. 1) High-credibility documents are collected by retrieving the most relevant documents from the external corpus for each question. Specifically, we first employ $\mathtt { b g e - l a r g e - e n - v 1 . 5 ^ { 5 } }$ to obtain a set of candidates from the Wikipedia dump on December 30, 2018 (Karpukhin et al. 2020). Then, we apply bge-reranker-large6 to rank the retrieved candidates and select the top four documents. 2) Low-credibility documents are generated via prompting LLMs (i.e., gpt-3.5-turbo-0125), with misinformation included, similar to the practice in previous works (Pan et al. 2023a,b, 2024; Hong et al. 2024; Chen and Shu 2024). Specifically, given a question, we instruct the LLM to generate a news-style piece containing misinformation that supports an incorrect answer, which is regarded as one lowcredibility document for the question. For each question, we collect three distinct low-credibility documents, all supporting the same incorrect answer. The prompts can be found in Appendix H.

In our implementation, we combine generated lowcredibility documents with retrieved high-credibility documents as input for the LLM. This approach avoids injecting low-credibility documents directly into the corpus, which can lead to inputs that are either overwhelmed by misinformation or completely devoid of it. In contrast, our method provides greater control, enabling us to effectively evaluate the impact of varying amounts of low-credibility documents on the LLM’s performance.

Credibility Scores Generation. We adopt two different ways to assign credibility scores for each document. 1) Ideal Setting. After obtaining the high-credibility and lowcredibility documents, we assign a score of 10 to each highcredibility document and a score of 1 to each low-credibility document. 2) GPT Setting. We employ GPT (i.e., gpt-3.5- turbo-0125) to directly generate the credibility score for each document. The prompts and the distribution of GPTgenerated scores for all documents are provided in Figure 20 and Appendix C.

Compared Methods. We compare our CrAM model with four types of methods: 1) Naive RAG. The Naive RAG follows the standard RAG pipeline without any mechanisms against misinformation. 2) Prompt Based. This method directly informs the LLM of the credibility score via prompts, feeding the score and documents into the LLM without additional training. 3) Exclusion. This method excludes the documents with credibility scores below a threshold. This method will not be compared under the ideal setting due to the binary value of the ideal credibility score. 4) CAG. This method is proposed by Pan et al. (2024), which directly incorporates credibility scores and documents into prompts to fine-tune an LLM (i.e., Llama2-13B) to lift its understanding capabilities. Among them, Naive RAG, Prompt Based, and Exclusion are non-SFT methods, while CAG is an SFTbased method.

Hyperparameters. Unless otherwise specified, in the following experiments, we randomly select 100 data points from each dataset to calculate average IE for all the heads. And we use another validation set of 100 data points from each dataset to determine how many top-ranked heads should be included in the final modified set.

# 4.2 Main Results

Comparison with Non-SFT Methods. We first compare our CrAM model with Non-SFT methods, i.e., Naive RAG, Prompt Based, and Exclusion. Table 1 and Table 2 show the experimental results in the Ideal and GPT settings respectively. We make the following observations. 1) Table 1 demonstrates that our CrAM method significantly outperforms all compared methods across all three LLMs: Qwen1.5-7B, LLama2-13B, and LLama3-8B, on both NQ and TriviaQA datasets in the setting of $4 \checkmark + 1 \pmb { x }$ (i.e., four high-credibility documents plus one low-credibility document). For instance, our CrAM model surpasses the secondbest method, i.e. Prompt Based, by $2 5 . 5 \%$ , $3 1 . 9 0 \%$ and $1 0 . 9 \%$ on Qwen1.5-7B, Llama2-13B and Llama3-8B in terms of EM on TriviaQA, demonstrating remarkable per

<html><body><table><tr><td rowspan="2">Model</td><td rowspan="2">In-context corpus</td><td rowspan="2">Method</td><td colspan="2">NQ</td><td colspan="2">TriviaQA</td></tr><tr><td>EM</td><td>F1 score</td><td>EM</td><td>F1 score</td></tr><tr><td rowspan="4">Qwen1.5-7B</td><td>0</td><td>Naive LLM</td><td>7.20</td><td>16.41</td><td>28.00</td><td>38.23</td></tr><tr><td>4</td><td>Naive RAG</td><td>27.60</td><td>39.08</td><td>55.30</td><td>66.85</td></tr><tr><td rowspan="2">4√+1x</td><td>Naive RAG</td><td>10.50</td><td>20.71</td><td>25.00</td><td>35.63</td></tr><tr><td>Prompt Based</td><td>12.20</td><td>22.26</td><td>27.40</td><td>37.98</td></tr><tr><td rowspan="5">Llama2-13B</td><td>0</td><td>CrAM Naive LLM</td><td>29.10 (+16.90)</td><td>41.02 (+18.76)</td><td>52.90 (+25.50)</td><td>64.16 (+26.18)</td></tr><tr><td>4</td><td>Naive RAG</td><td>20.30 28.90</td><td>28.59 39.98</td><td>50.40 62.50</td><td>57.56 71.03</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">4√+1x</td><td>Naive RAG</td><td>11.90</td><td>19.97</td><td>28.00</td><td>36.22</td></tr><tr><td>Prompt Based CrAM</td><td>12.50 33.60 (+21.10)</td><td>22.94 44.62 (+21.68)</td><td>23.10 59.90 (+31.90)</td><td>32.70 67.11 (+30.89)</td></tr><tr><td rowspan="4">Llama3-8B</td><td>0</td><td>Naive LLM</td><td>20.60</td><td>30.58</td><td>55.70</td><td>62.67</td></tr><tr><td>4√</td><td>Naive RAG</td><td>33.10</td><td>45.66</td><td>64.30</td><td>73.68</td></tr><tr><td rowspan="2">4√+1 x</td><td></td><td>16.00</td><td>26.16</td><td>36.80</td><td></td></tr><tr><td>Naive RAG Prompt Based</td><td>29.90</td><td>39.69</td><td>53.50</td><td>47.09</td></tr><tr><td></td><td></td><td>CrAM</td><td>36.90 (+7.00)</td><td>48.45 (+8.76)</td><td>64.40 (+10.90)</td><td>63.01 73.49 (+10.48)</td></tr></table></body></html>

Table 1: Main results under ideal setting. $0 \checkmark$ indicates no document and the model directly prompted, $4 \times$ indicates all four documents retrieved from the Wikipedia dump, and $4 \checkmark + 1 \pmb { x }$ indicates four high-credibility documents (i.e., retrieved from external corpus) plus one low-credibility document (i.e., containing misinformation). In the $4 \checkmark + 1 \pmb { x }$ setting, the best performance is highlighted in bold. And the red part indicates the difference between CrAM and second best performance.

F formance gains. 2) With GPT-generated credibility scores, our CrAM model also outperforms all compared methods on all three LLMs over both NQ and TriviaQA datasets, as shown in Table 2, further highlighting its effectiveness. 3) Interestingly, we find that our CrAM model with $4 \checkmark + 1 \pmb { x }$ sometimes even outperforms the Naive RAG with $4 \checkmark$ under ideal setting. This is likely because our generated misinformation includes both affirmations of incorrect information and denials of correct information, e.g.“The first person to win the Nobel Prize in Physics was not Roentgen, but Einstein.” This allows LLMs to reuse the correct information denied by the misinformation. To further validate this hypothesis, we conduct additional experiments and present the findings in Appendix F.

Comparison with SFT-based Method. For a fair comparison, we only compare our Llama2-13B based CrAM model with CAG-13B, because CAG-13B is trained on Llama2- 13B. Moreover, to verify the robustness of our CrAM model,

M

we perform comparisons using different numbers of lowcredibility documents. As shown in Figure 3, our CrAM model consistently outperforms the CAG-13B model remarkably in terms of F1 score when the number of lowcredibility documents ranges from 1 to 3. The results further prove the effectiveness of our CrAM model.

# 4.3 In-Depth Analysis

Effect of Number of Low-credibility Documents. In the following, we analyze the effect of varying the number of low-credibility documents fed into the LLM. We conduct experiments using Llama3-8B on the NQ dataset. Specifically, we vary the number of low-credibility documents from 1 to 3 while keeping the number of high-credibility documents constant, i.e., 4. We present the experimental results in Figure 4. From the figure, we make the following observations. 1) Our CrAM model consistently outperforms the compared models when changing the number of low-credibility documents from 1 to 3 in both ideal and GPT settings. 2) Compa

<html><body><table><tr><td rowspan="2">Model</td><td rowspan="2">In-context corpus</td><td rowspan="2">Method</td><td colspan="2">NQ</td><td colspan="2">TriviaQA</td></tr><tr><td>EM</td><td>F1 score</td><td>EM</td><td>F1 score</td></tr><tr><td rowspan="4">Qwen1.5-7B</td><td>0</td><td>Naive LLM Naive RAG</td><td>7.20 27.60</td><td>16.41 39.08</td><td>28.00 55.30</td><td>38.23 66.85</td></tr><tr><td>4√</td><td>Naive RAG</td><td>10.50</td><td>20.71</td><td>25.00</td><td></td></tr><tr><td rowspan="2">4√+1x</td><td></td><td></td><td></td><td></td><td>35.63</td></tr><tr><td>Promupt nased CrAM</td><td>12.50</td><td>22.8</td><td>29.70</td><td>40.18</td></tr><tr><td rowspan="5">Llama2-13B</td><td>0</td><td>Naive LLM</td><td>23.10 (+1.50) 20.30</td><td>34.84 (+2.28) 28.59</td><td>52.10 (+2.60) 50.40</td><td>63.76 (+2.73) 57.56</td></tr><tr><td>4</td><td>Naive RAG</td><td>28.90</td><td>39.98</td><td>62.50</td><td>71.03</td></tr><tr><td></td><td>Naive RAG</td><td>11.90</td><td>19.97</td><td>28.00</td><td>36.22</td></tr><tr><td>4√+1x</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td> Prompt Based</td><td>21.20</td><td>21.60</td><td>20.50</td><td>30.09</td></tr><tr><td rowspan="5">Llama3-8B</td><td>0</td><td>CrAM Naive LLM</td><td>25.10 (+1.40) 20.60</td><td>35.56 (+1.56) 30.58</td><td>56.20 (+1.80) 55.70</td><td>64.03 (+1.66) 62.67</td></tr><tr><td>4√</td><td>Naive RAG</td><td>33.10</td><td>45.66</td><td>64.30</td><td>73.68</td></tr><tr><td></td><td></td><td>16.00</td><td>26.16</td><td></td><td></td></tr><tr><td></td><td>Naive RAG</td><td></td><td></td><td>36.80</td><td>47.09</td></tr><tr><td>4√+1x CrAM</td><td> Prompt Based</td><td>24.20 30.70 (+4.10)</td><td>34.140 41.71 (+3.27)</td><td>49.50</td><td>58.39</td></tr></table></body></html>

Table 2: Main results under GPT setting. $0 ~ \checkmark$ indicates no document and the model directly prompted, $4 \checkmark$ indicates all four documents retrieved from the Wikipedia dump, and $4 \checkmark + 1 \pmb { x }$ indicates four high-credibility documents (i.e., retrieved from external corpus) plus one low-credibility document (i.e., containing misinformation). In the $4 \checkmark + 1 \pmb { x }$ setting, the best performance is highlighted in bold. The red part indicates the improvement of our CrAM compared to the second-best model.

![](images/ff1db17db3edb1e838d14f96a5de45a7f0a8d23364094ae4c9d3a4ba56d68872.jpg)  
Figure 5: Performance on NQ and TriviaQA regarding the dataset size for determining the influential attention head changes.

rably, our CrAM model exhibits much smaller performance drops compared to other models when increasing the number of low-credibility documents. These results demonstrate the robustness of our proposed model to the varying number of low-credibility documents.

Effect of Dataset Size on Attention Heads Selection. As we described in Section 3.3, we randomly select 100 data points from each dataset to identify the influential attention heads. In the following, we vary the number of data points used for selecting these influential attention heads to analyze its impact on model performance. The experimental results are presented in Figure 5. Despite fluctuations in performance along with the changing dataset size, the variations are not substantial on both NQ and TriviaQA datasets, with a maximum difference of $4 \%$ in terms of EM. The results indicate that the number of data points has a minor impact on the final model performance.

![](images/93a9f3cf6367269c64d90f9e0f82776093da55183bb7c8bbcce8e9c564397f28.jpg)  
Figure 6: Performance on NQ in ideal setting regarding the varying number of selected attention heads.

![](images/a3a7bf5ff552747dc1eea2ebebe54bad8f9f20623940353613e4c86d29271db2.jpg)  
Figure 7: Density distribution of IE of all the attention heads in Llama3-8B.

Analysis on Number of Selected Attention Heads. In the following, we analyze the performance change when we adjust the number of selected attention heads. We present the results in Figure 6. We observe a sharp drop in model performance when the number of selected attention heads is near either 0 or the maximum number of heads, i.e., 1024; comparably, it has a minor effect when the number of selected attention heads falls into the range of values in between. To investigate the underlying reasons, we further analyze the IE’s density distribution using Llama3-8B, as shown in Figure 7. We find that the IE density distribution approximates a normal distribution centered around 0, with the majority of values concentrated near 0. It indicates that most attention heads have minor impact on model performance, and only when the attention heads with IE values far from zero, either positive or negative, are selected, the model performance will be affected significantly.

Table 3: Results of ablation study under ideal setting with 4 $\checkmark + 1 \pmb { x }$ (i.e., four high-credibility documents plus one lowcredibility document).   

<html><body><table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td>NQ</td><td>TriviaQA</td></tr><tr><td>EM</td><td>EM</td></tr><tr><td rowspan="3">Qwen1.5-7B</td><td>CrAM</td><td>29.10</td><td>52.90</td></tr><tr><td>CrAM-all</td><td>27.20 (-1.90)</td><td>50.60 (-2.30)</td></tr><tr><td>Naive RAG</td><td>10.50 (-18.60)</td><td>25.00 (-27.90)</td></tr><tr><td rowspan="3">Llama2-13B</td><td>CrAM</td><td>33.60</td><td>59.90</td></tr><tr><td>CrAM-all</td><td>29.50 (-4.10)</td><td>59.50 (-0.40)</td></tr><tr><td>Naive RAG</td><td>11.90 (-21.70)</td><td>28.00 (-27.90)</td></tr><tr><td rowspan="3">Llama3-8B</td><td>CrAM</td><td>36.90</td><td>64.40</td></tr><tr><td>CrAM-all</td><td>22.40 (-14.50)</td><td>51.50 (-12.90)</td></tr><tr><td>Naive RAG</td><td>16.00 (-20.90)</td><td>36.80 (-27.60)</td></tr></table></body></html>

Ablation Study To better understand the rationality of our model design, we conduct ablation study and present the results in Table 3. First, we remove the selection of influential attention heads and apply attention weight modification on all attention heads in LLMs, and denote this variant model as CrAM-all. As shown in Table 3, we observe that the performance of the CrAM-all model has noticeable drops on all three LLMs. Among them, Llama3-8B based CrAM has the largest decrease on both NQ and TriviaQA, i.e., $1 4 . 5 \%$ and $1 2 . 9 \%$ . This indicates the necessity of identifying the influential attention heads before modifying the attention weights.

If we disable the attention weight modification mechanism in our model, it becomes the Naive RAG method. Table 3 shows that this results in a remarkable performance drop on all three LLMs compared to the CrAM model. For instance, the performance of all three LLMs decreases more than $2 7 . 5 \%$ on TriviaQA dataset. These results verify that it is necessary to modify the attention weight and meanwhile take into account the credibility scores of the documents.

# 5 Related Work

Misinformation Detection. Misinformation detection aims to identify false or misleading information from various data sources (Guo et al. 2019; Kaliyar and Singh 2019; Vaibhav, Mandyam, and Hovy 2019; Huang et al. 2024). It can be categorized into non-LLM-based methods and LLM-based methods. Non-LLM methods often involve training models to identify misinformation (Vaibhav,

Mandyam, and Hovy 2019; Kaliyar, Goswami, and Narang 2021; Liu, Wang, and Li 2023; Goonathilake and Kumara 2020). For example, Kaliyar, Goswami, and Narang (2021) utilize BERT (Devlin et al. 2019) to score the credibility of documents, while Vaibhav, Mandyam, and Hovy (2019) use a graph neural network for misinformation detection. Comparably, LLM-based methods typically use LLMs without additional training (Pelrine et al. 2023; Quelle and Bovet 2024; Caramancion 2023; Hoes, Altay, and Bermeo 2023). For instance, Pelrine et al. (2023) adopt GPT-4 (OpenAI et al. 2024) for document credibility scoring, while Quelle and Bovet (2024) employ an LLM agent (Xi et al. 2023) for iterative verification of document credibility. In this study, we employ LLMs to obtain the credibility score for each document similar to the previous LLM-based methods (Pelrine et al. 2023; Hoes, Altay, and Bermeo 2023).

Combating Misinformation in RAG. RetrievalAugmented Generation (RAG) enhance LLMs by retrieving relevant documents from external corpus (Lewis et al. 2020; Izacard and Grave 2021; Cai et al. 2024). However, prior works (Zou et al. 2024; Pan et al. 2023b,a) find that RAG is vulnerable to misinformation in its corpus, leading to undesired results. To combat misinformation in RAG, lots of studies have been conducted. For example, CAR (Weller et al. 2024) adopt a query augmentation scheme to retrieve a larger set of documents first and then apply a voting mechanism to mitigate the impact of misinformation. RobustRAG (Xiang et al. 2024) obtains the LLM response for each document independently and aggregates these responses through keyword-based and decoding-based algorithms to generate the final result. Hong et al. (2024) and Pan et al. (2024) assign each retrieved document a credibility score and fine-tune LLMs with the documents and their scores, enabling the LLMs to leverage these credibility scores when generating. $\mathrm { C D ^ { 2 } }$ Jin et al. (2024) train two LLMs to generate truthful answers and misleading answers respectively to make it better distinguish the conflict information. However, CAR (Weller et al. 2024) and RobustRAG (Xiang et al. 2024) require multiple rounds of model inference, leading to inefficiency. The methods proposed by Hong et al. (2024), Pan et al. (2024), and Jin et al. (2024) require fine-tuning LLMs, which demands additional computational resources and well-designed training data, thereby limiting their application scenarios.

# 6 Conclusion

This work introduces CrAM, a plug-and-play method that enables RAG to automatically adjust the influence of retrieved documents on the output of LLMs based on document credibility. CrAM first identifies influential attention heads and then adjusts the attention weights of identified attention heads according to the credibility score of documents, regulating LLMs to pay less attention to the lowcredibility documents. Empirical experiments demonstrate that, compared to vanilla RAG, CrAM improves EM performance by over $20 \%$ on two datasets and even outperforms the baseline with SFT, demonstrating CrAM’s efficiency.