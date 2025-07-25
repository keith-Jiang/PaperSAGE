# Retrieval-Augmented Visual Question Answering via Built-in Autoregressive Search Engines

Xinwei Long1, Zhiyuan $\mathbf { M } \mathbf { a } ^ { 1 }$ , Ermo Hua1, Kaiyan Zhang1, Biqing ${ \bf Q } ^ { \mathrm { i } ^ { 2 } }$ , Bowen Zhou1

1Department of Electronic Engineering, Tsinghua University 2Shanghai Artificial Intelligence Laboratory longxw $2 2 @$ mails.tsinghua.edu.cn

# Abstract

Retrieval-augmented generation (RAG) has emerged to address the knowledge-intensive visual question answering (VQA) task. Current methods mainly employ separate retrieval and generation modules to acquire external knowledge and generate answers, respectively. We propose ReAuSE, an alternative to the previous RAG model for the knowledgebased VQA task, which seamlessly integrates knowledge retriever into the generative multi-modal large language model, serving as a built-in search engine. Specifically, our model functions both as a generative retriever and an accurate answer generator. It not only helps retrieve documents from the knowledge base by producing identifiers for each document, but it also answers visual questions based on the retrieved documents. Furthermore, we also propose a reinforced retrieval calibration module from relevance feedback to improve retrieval performance and align with the preferences for accurate answer generation. Extensive experiments on two representative OKVQA and A-OKVQA datasets demonstrate significant improvements ranging from $2 . 9 \%$ to $9 . 6 \%$ across all evaluation metrics when compared to strong baselines.

# Introduction

The Visual Question Answering (VQA) task aims to answer questions based on a user-provided image, which has received significant attention from CV and NLP community (Antol et al. 2015; Hu et al. 2017; Shen et al. 2023b; Sun et al. 2024; Zhu et al. 2024). Early VQA methods (Mascharka et al. 2018; Gao et al. 2019) mainly focus on understanding visual elements within the image. Recently, the research trend of VQA has shifted towards knowledgeintensive scenarios (Shah et al. 2019), requiring the incorporation of external knowledge and joint reasoning over multimodal content to generate accurate answers. However, existing methods generally face challenges in effectively acquiring relevant information from large-scale knowledge bases using multi-modal queries (Lin et al. 2023).

Retrieval-augmented generation (RAG) (Chan et al. 2024; Chen et al. 2024) has recently emerged as a promising approach for knowledge-based visual question answering (KBVQA) tasks (Gao et al. 2022; Lin and Byrne 2022a;

![](images/f48dbd3672c91cd307874ebce736f81004d96323f35621db0465486839bfe7f7.jpg)  
Figure 1: Comparing with the Paradigm of Previous Knowledge-Based VQA Methods.

Chen et al. 2022). RAG-based approaches typically consist of two separate stages: retrieval and generation. In the first retrieval stage, these methods usually integrate multiple discriminative retrievers, each designed for specific purposes such as image-to-text or text-to-text retrieval. Afterward, in the second answer generation stage, these methods typically use generative multi-modal large language models (MLLM) to produce the final result. Despite achieving success in some benchmarks (Marino et al. 2019; Schwenk et al. 2022), this workflow still encounters several limitations. 1) Current methods sequentially invoke models in the pipeline for feature engineering, retrieval, and answer generation, requiring the integration of multiple heterogeneous models. 2) Moreover, these methods typically combine generative answer generators with discriminative retrievers. The disparate model architectures make it challenging for retrievers to further optimize their performance based on the feedback from the answer generator. Consequently, the research question arises: how can we integrate knowledge retrieval and answer generation into a homogeneous generative model?

To address the above issue, we propose ReAuSE, a novel Retrieval-augmented framework with built-in Autoregressive Search Engines for knowledge-based VQA tasks, which seamlessly integrates knowledge retrieval into the generative MLLM. ReAuSE takes advantage of the fact that MLLMs can serve as virtual knowledge warehouses (Pan et al. 2024), recognizing the documents that a multi-modal query can be linked to. Therefore, ReAuSE abandons the discriminative retrieval paradigm that computing the similarity between the query and document one by one, whereas directly generates the document identifier in an autoregressive manner, where each identifier corresponds to a document within the knowledge base. We define the document identifiers as a sequence of tokens that appears at least once within a document in the knowledge base, thus enabling effective and efficient mapping to the document. Subsequently, we propose a reinforced retrieval calibration method based on relevance feedback to further enhance retrieval performance. To collect relevance preference data, we employ a MLLM as a reward model, which inputs sampled documents and questions into this model and assesses document relevance based on the VQA scores (Antol et al. 2015) of the generated answers. To align with relevance preference, we employ a direct preference optimization (DPO) algorithm (Rafailov et al. 2023) to further refine the generative retrieval model. In the answer generation stage, we input the retrieved documents one by one, and the model obtains the final prediction based on the joint probability of retrieval and answer generation.

We conduct primary experiments on two representative knowledge-based VQA benchmarks, OKVQA and AOKVQA. The experimental results show significant improvements of $2 . 9 \% - 9 . 6 \%$ across all metrics compared to strong baselines. Additionally, we perform knowledge retrieval experiments on three datasets to further validate the performance of the generative knowledge retrievers. Our model consistently outperforms other discriminative knowledge retrievers and the improvements become more apparent when applied to large-scale knowledge bases. This outcome illustrates our model’s capability to retrieve knowledge from large-scale knowledge sources. The code will be available at https://github.com/xinwei666/ReAuSE

# Related Work

Traditional Visual Question Answering (VQA) tasks (Johnson et al. 2017; Mishra et al. 2019), which focus on answering questions related to visual elements (e.g., simple counting, visual attributes), have been extensively studied. Several studies (Marino et al. 2019) have revealed that over $78 \%$ of questions can be answered by people under ten years old, indicating that traditional VQA tasks require little background knowledge to answer a vast majority of questions.

Knowledge-based VQA. To assess models’ capacity to leverage world knowledge instead of relying solely on input data, knowledge-based VQA tasks have emerged, such as OKVQA (Marino et al. 2019), and A-OKVQA (Schwenk et al. 2022). OKVQA and A-OKVQA datasets pose challenges in acquiring the necessary knowledge from an outside source and performing reasoning over multi-modal contexts and knowledge. Recently, Infoseek (Chen et al. 2023d) has been proposed, featuring visual questions about detailed properties of factual knowledge in Wikipedia. The above datasets all highlight the importance of retrieving knowledge from external sources and underscore that current state-ofthe-art methods still have significant room for improvement in this task.

Existing approaches have been proposed to incorporate knowledge in two ways to address knowledge-based VQA tasks. One line of research (Xenos et al. 2023a; Chen et al. 2023e; Gui et al. 2021) leverages implicit knowledge from LLMs. This approach involves converting images into text or directly feeding multi-modal contexts into LLMs (e.g. GPT-3 (Brown et al. 2020), GPT-4V (Achiam et al. 2023), etc.) to generate text that serves as augmented knowledge, but hallucinated information produced by LLMs poses risks to the overall pipeline. Another research direction (Lin et al. 2022; Hao et al. 2024a; Lin et al. 2023) aims to retrieve explicit knowledge from structured or unstructured KB. This approach, known as retrieval augmentation, often uses offthe-shelf tools to generate visual tags and captions, thereby boosting the performance of knowledge retrievers. Several studies (Gao et al. 2022; Hu et al. 2023b) have tried to combine both ways by simply using the results of LLMs and retrievers but led to limited improvements over baselines.

Knowledge Retrieval. As a crucial component of retrievalaugmented approaches, knowledge retrievers face challenges in handling multi-modal queries (Luo et al. 2021, 2023; Shen et al. 2023a). Several methods (Lin and Byrne 2022b,a; Gao et al. 2022), which employ separate text-totext and image-to-text retrievers, struggle to capture crossmodal interactions. To bridge this gap, Reviz (Luo et al. 2023) leverages visual-language models to unify the encoding of image and text queries, and FMLR (Lin et al. 2023) proposes a fine-grained late-interaction framework to fuse cross-modal features at the token level. PreFLMR (Lin et al. 2024) explores scaling laws for knowledge retrieval based on the FLMR model. Although these methods achieve improvements over previous approaches, they require training on large-scale datasets containing millions of image-text pairs, which incurs high computational costs.

Recently, some studies (Bevilacqua et al. 2022; Ziems et al. 2023; Li et al. 2023, 2024a; Long et al. 2024b; Jain, Soares, and Kwiatkowski 2024) have introduced generative pipelines in information retrieval tasks, instead of discriminative retrievers. These methods (Tay et al. 2022) are based on the assumption that all documents are memorized by generative language models, and the language model directly generates the identifiers of relevant documents based on the query. While prior research (Li et al. 2024b; Long et al. 2024a) has investigated generative retrieval for multi-modal tasks, such methods have demonstrated only marginal gains over traditional methods when applied to general tasks. Different from them, we are the first work to seamlessly integrate generative retrieval and retrieval-augmented VQA tasks, and use the feedback from the QA module to enhance the retrieval performance, thereby achieving better retrieval and QA results simultaneously.

# Methodology

We introduce ReAuSE, a Retrieval-Augmented framework utilizing built-in Autoregressive Search Engines tailored for

Name the type of plant this is? ### Instruction: ...... respond to the question Question-Image Pair + Search-specific Prompt Question-Image Answer-specific with tokens from the Pair Prompt Wikipedia passages ... Sampled Docs and Docids ### Input:{Question} ### IRmesapgoe:nse: {} cDa73a4si5a9nfsicpuesciceas.r.i.- pDa8l9m99t7re..e.sparient.e..r aDsi7a3n45sp9e..c.iefiscoufs cpalraincta..i.s   
Question-Image Pair Search-specific Prompt MLLM MLLM MLLM   
Serving as a built-in Autoregressive Search Engine Serving as a built-in Autoregressive Search Engine Serving as a Retrieval-augmented Generator   
Constraints Generative Loss VQA RM: SearLcohs-sDPO   
KB 8 Document Identifiers EM  RM: palm X ficus Sim RM:   
Get_Docs (·) 司 tDre8e9s9a9r7e .a..mpaoingteerxpoatlicm.. 四>酉 Relevance Feedback Pine ferns

knowledge-based VQA tasks. ReAuSE is designed as a unified model to facilitate both effective knowledge retrieval and question-answering tasks.

# Problem Formulation

Formally, let $\textit { \textbf { D } } = \{ D _ { 1 } , . . . , D _ { k } \}$ denotes a knowledge base used for the knowledge-based VQA task, $\begin{array} { r l } { D _ { i } } & { { } = } \end{array}$ $\{ d _ { 1 } , . . . , d _ { | D | } \}$ denotes a document with its title and textual contexts, and $R _ { i } = \{ r _ { 1 } , r _ { 2 } , . . . , r _ { | R | } \}$ denotes an identifier of the document $D _ { i }$ . Given a multi-modal query $X$ , the generative knowledge retrieval can be formulated as a $\tt S e q 2 S e q$ task, as Eq. 1,

$$
\mathcal { P } ( R _ { i } | \boldsymbol { X } ) = \prod _ { j = 1 } \mathcal { P } ( r _ { j } | \boldsymbol { r } _ { < j } , \boldsymbol { X } , \boldsymbol { \Theta } ) .
$$

where $\mathcal { P }$ denotes the standard auto-regressive language modeling probability and $\Theta$ are the paramters of our model. During inference, the model employs a constrained strategy to guide the decoder in generating valid identifiers, which maintains a deterministic mapping relationship $\varphi$ between identifier and document, as Eq. 2,

$$
\varphi : R _ { i } \to D _ { i } , { \mathrm { w h e r e ~ } } D _ { i } \in { \mathcal { D } } .
$$

Finally, we obtain a subset $\hat { D } = \{ D _ { 1 } , . . . , D _ { | K | } \}$ from $\mathcal { D }$ to improve answer generation. The overall likelihood of generating the answer $Y$ is given by Eq. 3,

$$
\mathcal { P } ( \boldsymbol { Y } | \boldsymbol { X } ) = \sum _ { D _ { i } \in \hat { D } } \underbrace { \mathcal { P } ( R _ { i } | \boldsymbol { X } ) } _ { r e t r i e v a l } \cdot \underbrace { \mathcal { P } ( \boldsymbol { Y } | \boldsymbol { X } , D _ { i } ) } _ { g e n e r a t i o n } .
$$

# Built-in Autoregressive Search Engines

We introduce a novel autoregressive search engine for knowledge-based VQA tasks to facilitate retrieval from external knowledge bases. The autoregressive search engine leverages a generative architecture similar to that of common multimodal large language models, instead of discriminative models, enabling its seamless integration and functioning as a built-in module.

Given a multi-modal input $X = \{ Q , V \}$ , the autoregressive search engine aims to generate the relevant identifier directly in a seq2seq manner as Eq. 1. For example, Fig. 2 shows how our model generates the corresponding identifier for a document related to the “palm tree” based on the input image and question. To achieve such a generative retriever, we mainly elaborate on the three aspects as follows:

Document Identifier. Based on the assumption in (Pan et al. 2024) that large language models are aware of the content within each document, we define any document’s identifier as subsequences that appear only in that specific document. Unlike the one-to-one relationship in DSI (Tay et al. 2022), We assign more than one identifier to each document, as long as these identifiers are unique for this document. Consequently, our model does not require additional memory steps as in existing studies (Tay et al. 2022; Li et al. 2024b) to associate documents with identifiers.

Supervised Fine-tuning teaches our model to generate relevant identifiers based on the autoregressive probability for each given multi-modal query. To sample the most relevant sub-sequences from the given ground-truth document as identifiers, we employ a large language model (Touvron et al. 2023) as an extractive summarizer, which uses a fixedlength original text to answer a given question. Later, we filter the obtained set of identifiers and select the identifier containing the most answer keywords as the target identifier. Note that our model is model-agnostic, allowing it to be applied to any generative multi-modal large language model. The generative loss function can be formalized as maximizing the likelihood of the target identifier using the teacher forcing strategy, as Eq. 4.

$$
\mathcal { L } _ { \mathit { r e t r i e v a l } } = \sum _ { j = 1 } \log \mathcal { P } ( r _ { j } | \boldsymbol { r } _ { < j } , X ) .
$$

To avoid overfitting and catastrophic forgetting, we freeze all the parameters of the MLLM and adopt the Low-Rank Adaptation (LoRA) method (Hu et al. 2021) to efficiently fine-tune our model, with only the parameters of LoRA being updated.

Constrained Decoding and FM-Index. A valid identifier is defined as a generated sequence that appears at least once within a document in the knowledge base, ensuring that each generated identifier can be directly linked to a specific document. To help the model generate valid identifiers during inference, we implement a beam decoding strategy constrained by knowledge bases.

Specifically, we use the previously generated sequence $R _ { i } ^ { t - 1 } = \{ r _ { 1 } , . . . , r _ { t - 1 } \}$ as the prefix condition to search for all matching strings in the knowledge base. We then extract the subsequent tokens from these strings to form a feasible token set $s$ . The model’s next token, $\boldsymbol { r } _ { t }$ , is restricted to selection from $s$ , guaranteeing that all generated sequences exist within the knowledge base. To support fast substring search, we utilize an FM-Index database (Ferragina and Manzini 2000; Bevilacqua et al. 2022) to store the knowledge base. FM-Index is an efficient indexing structure tailored for substring search. The time complexity for obtaining the next allowed token is nearly $\mathcal { O } ( V )$ , where $V$ is the vocabulary size, independent of the size of the knowledge base.

# Reinforced Retrieval Calibration via Relevance Feedback

Despite teaching our model through supervised fine-tuning to generate relevant document identifiers based on user queries, the retrieved documents exhibit varying degrees of relevance. Even when documents are provided, the QA model may struggle to provide accurate responses. Optimally, the generative retriever should retrieve documents that: (1) strongly correlate with the multi-modal query, and (2) minimize extraneous content. Consequently, it is essential to further improve retrieval performance through feedback from the QA model.

As the first step towards this goal, we sample a set of identifiers $\{ R _ { 1 } , . . . , R _ { k } \}$ for each $X$ using the generative retriever $\pi _ { s f t }$ that has been supervised fine-tuned. Then, we score the collected samples by evaluating their relevance from three aspects:

• Contributions to VQA performance. A document is deemed relevant if a model can produce the correct answer using it. To evaluate this relevance, we employ an MLLM that has not been fine-tuned on downstream data as the reward model, with the VQA score serving as the reward value $v _ { v q a } \in [ 0 , 1 ]$ .   
• Keyword Hit Count. If an identifier includes keywords from the answer set, it is likely to be relevant. To quantify this relevance, we employ an exact matching function as the reward function, with matching signals serving as the reward values $v _ { h i t } \in \{ 0 , 1 \}$ .   
• Semantic Similarity. Higher semantic similarity between an identifier and a document indicates that the identifier better represents the document’s semantics, thereby suggesting a lower presence of irrelevant content within the document. To measure this relevance, we use the BERT model to calculate the cosine similarity between identifiers and documents as the reward values $v _ { s i m } \in [ 0 , 1 ]$ .

The overall reward can be obtained by taking a weighted sum of the scores from different aspects. Then, we build a triplet $< X , R ^ { + } , R ^ { - } >$ for each $X$ by treating the identifiers with the highest/lowest reward as positive/negative samples, respectively. Using the triplets reflecting the QA model’s preference, the retriever can be further aligned by preference-based reinforcement learning. As one of the typical methods, direct preference optimization (DPO) (Rafailov et al. 2023) is widely used for its efficiency and effectiveness. Therefore, we employ the DPO loss to further optimize our autoregressive knowledge retriever as Eq. 5,

$$
\mathcal { L } _ { d p o } = - \mathrm { l o g } \sigma \Bigg ( \beta \mathrm { l o g } \frac { \pi _ { \Theta } ( R ^ { + } | X ) \pi _ { s f t } ( R ^ { - } | X ) } { \pi _ { s f t } ( R ^ { + } | X ) \pi _ { \Theta } ( R ^ { - } | X ) } \Bigg ) .
$$

where $\pi _ { s f t }$ is the original model used as reference, and $\pi _ { \Theta }$ is the model being optimized. As before, we only update the parameters of LoRA.

# Answer Generation

Utilizing built-in autoregressive knowledge retrievers, we extract the top-K relevant documents from extensive knowledge bases to serve as external knowledge. For our answer generation model, we employ a model architecture homologous to that of the retrieval module. As illustrated in Fig. 2, we construct a prompt template, filling the slots with the image, question, and each retrieved document. The multimodal contexts are then fed into the model, and the training loss of the answer generation follows that of the generative retrieval model, as Eq. 6,

$$
\mathcal { L } _ { g e n } = \sum _ { j = 1 } \log \mathcal { P } ( y _ { j } | \pmb { y } _ { < j } , X , D _ { i } ) .
$$

where $y _ { j }$ denotes the $j - t h$ token of the ground-truth answer $Y$ . As before, we freeze all the parameters of the MLLM, but introduce another LoRA, and only update the parameters of this new LoRA.

$$
\begin{array} { r l } & { \hat { Y } , \hat { D } = \underset { Y , D _ { i } } { \mathrm { a r g m a x } } \mathcal { P } ( Y , D _ { i } | X ) } \\ & { \quad \quad = \underset { Y , D _ { i } } { \mathrm { a r g m a x } } \mathcal { P } ( Y | X , D _ { i } ) \cdot \mathcal { P } ( R _ { i } | X ) . } \end{array}
$$

During inference, We use the same MLLM and parameters for both the retrieval and answer generation stages, except for the two LoRA adapters. After retrieving the relevant document set, we switched to the LoRA adapter for answer generation, and obtain the final prediction through the joint probability of retrieval and answer generation, as Eq. 7.

# Experiments

# Experiment Setup

Datasets and Knowledge Bases. We focus on the knowledge-based VQA benchmarks, OKVQA (Marino et al. 2019) and A-OKVQA (Schwenk et al. 2022). Previous work provided two retrieval corpora, GS112K (Luo et al. 2021) and Wiki21M (Karpukhin et al. 2020), for the OKVQA dataset. GS112K contains 112K passages collected through Google Search, while Wiki21M is a subset of Wikipedia, containing 21M Wikipedia entries. Moreover, we also conduct retrieval experiments on these two corpora and introduce a new information-seeking dataset, InfoSeek (Chen et al. 2023d), to evaluate the model’s retrieval performance. Since InfoSeek’s KB is not publicly available, we use the KB provided by PreFLMR (Lin et al. 2024) and follow the same experimental setup.

Evaluation Metrics. We strictly follow the settings of the original papers, using the corresponding metrics for each dataset. For the OKVQA dataset and the “direct answer” setting of the A-OKVQA dataset, we use the VQA score to evaluate the model’s performance. For the “multi-choice” setting of the A-OKVQA dataset, we use accuracy for evaluation. To evaluate the performance of knowledge retrieval, we use the Pseudo-relevance Recall ${ \mathfrak { Q } } \mathrm { K }$ (PRR ${ \mathfrak { Q } } \mathbf { K } )$ (Luo et al. 2021), consistent with the baselines.

Baselines. We adopt several baseline methods for comparison, categorized as follows: 1) multi-modal large language models: LLaVA-13B (Liu et al. 2023), PALmE-562B (Chen et al. 2023c), and GPT-4V (Achiam et al. 2023). 2) knowledge-enhanced methods via GPT3/4 APIs: Prophet (Shao et al. 2023), Promptcap (Hu et al. 2023a), FillingGap (Wang et al. 2023) and REVIVE (Lin et al. 2022). 3) retrieval-augmented methods: TwO (Si et al. 2023), ReVeaL (Hu et al. 2023b), GeMKR (Long et al. 2024a), and FLMR (Lin et al. 2023). For the A-OKVQA dataset, we also add the advanced GPV2 (Schwenk et al. 2022), SimVQA (Xenos et al. 2023b), Cola-FT( $\mathrm { 1 1 B } { + } 3 \mathrm { B } )$ (Chen et al. 2023b) and CKR-VQA (Hao et al. 2024a) as baselines.

Implementation Details. Our framework is modelagnostic. In our main experiments, we utilize MiniGPT4- v2-7B as the base model, which employ ViT-L/14 from pretrained CLIP as the image encoder and LLaMa-v2-7B (Touvron et al. 2023) as the text encoder. We freeze all parameters of the MLLM, allowing updates only to the LoRA parameters. We use the same MLLM in the three stages but apply two sets of LoRA parameters to optimize the model respectively: one for retrieval and alignment, and the other for answer generation. Our model is implemented in PyTorch, utilizing version 0.3.0 of the PEFT library, which supports efficient switching between two LoRA adapters during inference. Similar to baselines, we use image captions as features to enhance the model’s performance. Each training stage is performed on four NVIDIA A6000 48G GPUs and completed within three hours.

Table 1: Performance on the OKVQA benchmark. PPR@K applies only to RAG baselines; “-” denotes inapplicability or unavailable results. $^ { 6 6 } { * } ^ { 7 3 }$ indicates the results we reproduced using the official code and the same answer generator as our model.1   

<html><body><table><tr><td>Model</td><td>PRR@K</td><td>Score</td></tr><tr><td colspan="3">Multi-modal Large Language Models</td></tr><tr><td>LLaVA-13B (Liu et al. 2023)</td><td></td><td>61.9</td></tr><tr><td>Minigpt4-v2-7B (Chen et al. 2023a)</td><td></td><td>57.8</td></tr><tr><td>Minigpt4-v2-7B(FT) (Chen et al.2023a)</td><td></td><td>61.9</td></tr><tr><td>PaLM-E-562B (Driess etal.2023)</td><td></td><td>66.1</td></tr><tr><td>GPT-4V (Achiam etal.2023)</td><td></td><td>64.3</td></tr><tr><td colspan="3">Knowledge-enhancedMethodsviaGPT-3/4vAPIs</td></tr><tr><td>ReVIVE (Lin et al.2022)</td><td></td><td>58.0</td></tr><tr><td>Prophet (Shao et al.2023)</td><td></td><td>61.1</td></tr><tr><td>Promptcap (Hu et al.2023a)</td><td></td><td>60.4</td></tr><tr><td>FillingGap (Wang et al. 2023)</td><td></td><td>61.3</td></tr><tr><td>MM-Reasoner (Khademi et al. 2023)</td><td></td><td>60.8</td></tr><tr><td colspan="3">Retrieval-augmentedGenerationMethods</td></tr><tr><td>TRiG (Gao et al. 2022)</td><td>45.8</td><td>50.5</td></tr><tr><td>RA-VQA (Lin and Byrne 2022a)</td><td>82.8</td><td>54.5</td></tr><tr><td>TwO (Si et al. 2023)</td><td></td><td>56.7</td></tr><tr><td>ReVeaL (Hu etal.2023b)</td><td></td><td>59.1</td></tr><tr><td>FLMR (Lin et al. 2023)</td><td>89.3</td><td>62.1</td></tr><tr><td>FLMR(Lin et al. 2023) *</td><td>88.3</td><td>62.7</td></tr><tr><td>KSVQA (Hao et al. 2024b)</td><td></td><td>62.8</td></tr><tr><td>GeMKR (Long et al. 2024a) *</td><td>78.6</td><td>61.8</td></tr><tr><td>ReAuSE (Ours)</td><td>92.6</td><td>65.7</td></tr></table></body></html>

# Main Results

We compare our ReAuSE with the aforementioned baselines for knowledge-based VQA tasks in Tab. 1 and Tab. 2. The experimental results illustrate that ReAuSE achieves significant improvements over the competitive baselines on the challenging OKVQA and A-OKVQA datasets.

From Tab. 1, we can observe that ReAuSE outperforms the competitive baseline FLMR on both retrieval and VQA metrics, which consistently demonstrates the effectiveness of our method in integrating both knowledge retrieval and answer generation into a unified multi-modal large language model framework. ReAuSE achieves an advanced VQA score on OKVQA when compared to models with similar parameter scales, surpassing the previous best retrievalaugmented method by more than $2 . 9 \%$ and outperforming methods that use LLM-APIs for knowledge enhancement by $4 . 6 \%$ . Moreover, our method exceeds GPT-4V by $1 . 4 5 \%$ in VQA score. Even compared with the closed-source PALME-562B, which is over 80 times larger than ours, our method is only $0 . 5 \%$ behind.

Table 2: Performance on the A-OK-VQA benchmark.   

<html><body><table><tr><td colspan="3">Multi-Choice Models</td><td colspan="2">Direct-Answer</td></tr><tr><td></td><td>val</td><td>test</td><td>val</td><td>test</td></tr><tr><td>LLaVA-1.5-7B</td><td>77.1</td><td>74.5</td><td>63.7</td><td>58.6</td></tr><tr><td>InstructBLIP-7B(FT)</td><td>73.0</td><td>71.1</td><td>62.4</td><td>58.7</td></tr><tr><td>Minigpt4-v2-7B(FT)</td><td>-</td><td>1</td><td>61.3</td><td>-</td></tr><tr><td>GPV-2</td><td>60.3</td><td>53.7</td><td>48.6</td><td>40.7</td></tr><tr><td>PromptCap</td><td>73.2</td><td>73.1</td><td>56.3</td><td>59.6</td></tr><tr><td>Prophet</td><td>76.4</td><td>73.6</td><td>58.2</td><td>55.7</td></tr><tr><td>FillingGap</td><td></td><td></td><td>59.8</td><td>-</td></tr><tr><td>SimVQA</td><td></td><td></td><td>58.6</td><td>57.5</td></tr><tr><td>REVEAL</td><td></td><td></td><td>52.2</td><td></td></tr><tr><td>Cola-FT</td><td>78.1</td><td>76.7</td><td>-</td><td></td></tr><tr><td>CKR-VQA</td><td>76.2</td><td>75.4</td><td>58.1</td><td>60.1</td></tr><tr><td>ReAuSE (Ours)</td><td>85.0</td><td>80.3</td><td>67.7</td><td>65.8</td></tr></table></body></html>

Table 3: Ablation Studies. w/o denotes “without”.   

<html><body><table><tr><td>Ablation Setting</td><td>PRRecall@5</td><td>Score</td></tr><tr><td>Full Model (Ours)</td><td>92.6</td><td>65.7</td></tr><tr><td>w/o Search Engines</td><td>-</td><td>61.9</td></tr><tr><td>w/o Fine-tuning Search Engine</td><td>33.1</td><td>61.6</td></tr><tr><td>w/o Constrained Decoding</td><td>1</td><td>63.2</td></tr><tr><td>w/o Retrieval Calibration</td><td>88.7</td><td>62.5</td></tr><tr><td>w/o VQAReward Model</td><td>91.0</td><td>63.3</td></tr><tr><td>w/o EMReward Func.</td><td>89.9</td><td>64.5</td></tr><tr><td>w/o Sim.RewardModel</td><td>91.7</td><td>65.3</td></tr></table></body></html>

The OKVQA benchmark poses a challenging issue of retrieving relevant knowledge from extensive knowledge bases or directly generating useful information about multimodal contexts. Despite using GPT-3 or GPT-4V to acquire knowledge or directly adopting GPT-3 as the backbone, MM-Reasoner and FillingGap fail to achieve obvious improvements compared to retrieval-augmented methods. In contrast, retrieval-augmented methods, such as FLMR and KSVQA, achieve better VQA performance by incorporating manually designed feature engineering and integrating multiple retrievers and selectors.

From Tab. 2, ReAuSE demonstrates more significant performance improvements on A-OKVQA, with accuracy and VQA scores increasing by $4 . 9 \%$ to $9 . 6 \%$ compared to baselines of similar parameter scales2. Our approach demonstrates consistent improvements, which can be attributed to two key factors. First, we leverage large language models as virtual knowledge bases by replacing traditional discriminative pipelines with generative retrievers. Second, we implement reinforced retrieval calibration to align the search engine with the answer generator, enabling the retriever to incorporate relevance feedback for refinement, thereby yielding more relevant results. In the following sections, we will examine the performance of the autoregressive search engine and analyze the impact of search results on the answer generation process.

# Ablation Study

We conduct a series of ablation studies by gradually removing each module of our framework and the corresponding results are presented in Tab. 3.

To evaluate the impact of retrieval augmentation, we first remove the built-in autoregressive search engine, using the MLLM as an answer generator without access to external knowledge. This operation results in a $3 . 8 \%$ decrease in the VQA score, indicating that external knowledge retrieval is crucial for knowledge-based VQA tasks. Next, if we do not supervised fine-tune the MLLMs, it cannot effectively serve as a generative search engine to retrieve knowledge from the KB. Moreover, we disable the constrained decoding strategy, allowing the MLLM to generate image-related knowledge without restrictions. However, since this freely generated content cannot be linked to the document in the KB, it is used directly as external knowledge to support the answer generation process. This approach leads to a $2 . 5 \%$ decrease in the VQA score, likely due to the MLLM producing erroneous or hallucinated information, which results in inaccurate outputs from the answer generator.

To evaluate the effectiveness of the Reinforced Retrieval Calibration (RRC) module, we employ the generative search engine after supervised fine-tuning but remove the reinforced calibration module. We observe a $3 . 9 \%$ decrease in retrieval performance, which is slightly below that of the strongest baseline, FLMR. This suggests that the autoregressive retriever can be further optimized through the RRC module by leveraging relevance feedback from reward models. Furthermore, we disable each reward model to assess its effectiveness. We find that the VQA reward model enables the generative retriever to retrieve documents that align with the answer generator’s preferences, thereby improving VQA performance. Conversely, the EM reward model ensures that the generated identifiers include answer keywords, leading to enhanced retrieval performance.

# Effects of Retrieval Performance

To assess our model’s capability in retrieving knowledge from large-scale knowledge bases, we conduct experiments on the OKVQA dataset using two retrieval corpora: Google Search (GS112K) (Luo et al. 2021) and Wikipedia (Wiki21M) (Karpukhin et al. 2020), with knowledge bases ranging in size from 112K to 21M documents. Additionally, we introduce a new dataset, Infoseek (Chen et al. 2023d), consisting of 100K documents, which poses challenges for visual entity retrieval. As shown in Tab. 4, our proposed approach consistently outperforms the leading state-of-the-art baselines FLMR and Pre-FLMR across all evaluated metrics. Specifically, our model outperforms FLMR by $3 . 3 \%$ in PRRecall $\textcircled { a } 5$ on the GS112K corpus. This improvement explains why our answer generation model surpasses the FLMR model by $3 . 6 \%$ in the VQA score, as shown in Tab. 1. Moreover, our method outperforms FLMR by $1 0 . 6 \%$ on the Infoseek dataset and surpasses PreFLMR by $1 . 7 \%$ , indicating the effectiveness of ReAuSE in handling visual entity retrieval tasks.

We observe a significant performance drop of over $20 \%$ in the FLMR model when applied to the Wiki21M corpus, while our model exhibits only a $4 . 6 \%$ decrease. This indicates that our model demonstrates stronger generalization capabilities for retrieving from large-scale corpora. This can be attributed to the advantages of generative search engines, which generate document identifiers through tokenlevel search rather than relying on one-to-one matches at the document level (i.e., document-level search). Although the number of documents increases, the size of the token set (i.e., vocabulary) does not expand proportionally. Consequently, generative search engines are less affected by the scale of the knowledge base, whereas the performance of discriminative methods degrades as the corpus size increases.

Table 4: Retrieval Performance on Three Retrieval Corpora.   

<html><body><table><tr><td rowspan="2">#</td><td rowspan="2">Retrievers</td><td colspan="2">PRReCaKVQA- PR112al1@ 10</td><td colspan="2">PRReCKVQA- PR1MI1@ 10</td><td rowspan="2">IPfReck-100K</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>DPR (Karpukhin et al. 2020)</td><td>83.4</td><td>90.3</td><td>66.9</td><td>76.4</td><td></td></tr><tr><td>2</td><td>RA-VQA (Lin and Byrne 2022a)</td><td>82.8</td><td>89.0</td><td>1</td><td>=</td><td>-</td></tr><tr><td>3</td><td>ReViz-ICT (Luo et al. 2023)</td><td>73.4</td><td>83.2</td><td>61.9</td><td>72.6</td><td></td></tr><tr><td>4</td><td>GeMKR (Long et al. 2024a)</td><td>78.6</td><td>86.2</td><td>70.8</td><td>79.1</td><td>48.9</td></tr><tr><td>5</td><td>FLMR (Lin et al. 2023)</td><td>89.3</td><td>94.0</td><td>68.1</td><td>78.0</td><td>47.1</td></tr><tr><td>6</td><td>Pre-FLMR (Lin et al. 2024)</td><td>-</td><td></td><td>68.6</td><td>-</td><td>57.8</td></tr><tr><td>7</td><td>ReAuSE (Ours)</td><td>92.6</td><td>95.8</td><td>88.0</td><td>91.3</td><td>59.5</td></tr></table></body></html>

Figure 3: Case Studies. The highlighted text represents the document identifier generated by our model.   

<html><body><table><tr><td>Image & Question</td><td>What flavor is this</td><td></td><td>What type of plane isWhat is the person in</td></tr><tr><td>Docids</td><td>pastry? strawberry flavor that the 747-400 is a is so delicious</td><td>that? proven performer</td><td>the photo wearing? a wetsuit is a garment</td></tr><tr><td>Docs & Docids</td><td>slightly strawberry flavorthatissodeli- ciousandit couldn't</td><td>... the 747-400 is a proven performerwith worn to provide</td><td>worn awetsuitisagarment</td></tr><tr><td></td><td>bed it cr to make.</td><td>high reliability and incorporates major...</td><td>thermal protection while wet ...</td></tr><tr><td>GT Ans Ours</td><td>strawberry strawberry</td><td>passenger passenger</td><td>wetsuit wetsuit</td></tr></table></body></html>

Table 5: The Retrieval Time in the Inference Stage.   

<html><body><table><tr><td>Models</td><td>TopK</td><td>GS112K</td><td>Wiki21M</td></tr><tr><td>DPR</td><td>5</td><td>370.4ms</td><td>518.4ms</td></tr><tr><td>FLMR</td><td>5</td><td>758.4ms</td><td></td></tr><tr><td>ReAuSE (Ours)</td><td>5</td><td>751.3ms</td><td>962.1ms</td></tr><tr><td>ReAuSE(Ours)</td><td>10</td><td>1023.3ms</td><td>1273.2.1ms</td></tr></table></body></html>

generation and at least one traditional retriever for retrieval. ReAuSE uses LoRA fine-tuning, requiring 9K (OKVQA) to 17K (A-OKVQA) training data to train $0 . 4 9 \%$ of the parameters, with both SFT and RRC completing within 3 hours across 4 GPUs. In contrast, traditional retrievers such as PreFLMR and ReViz-ICT necessitate additional millions of data for full-scale fine-tuning.

We record the inference time of ReAuSE, FLMR, and the most efficient baseline, DPR to provide a qualitative result. As shown in Tab. 5, ReAuSE has comparable efficiency to traditional retrievers. ReAuSE generates top-K document identifiers with $l$ tokens for each query through a $l .$ -steps decoding $( l = 1 0 )$ ). In contrast, for each query, traditional retrievers need to calculate the similarity with many documents. When TopK $\scriptstyle : = 5$ , our model is only 440 milliseconds slower than DPR. Considering the nearly $20 \%$ performance improvement, we argue that such latency is acceptable.

# Case Study

As illustrated in Fig. 3, ReAuSE accurately generates the correct answers for all three samples. ReAuSE directly generates document identifiers associated with the image-text pairs using its built-in search engine. Each document identifier is a sequence of tokens representing a document, and it can be linked to a corresponding document that potentially contains information to answer the given question. What’s more, we observe that all generated document identifiers contain answer keywords, suggesting that generated document identifiers are highly relevant to the question.

# Conclusion

In this paper, we introduce ReAuSE, a novel KBVQA approach by integrating knowledge retrieval and generation within a unified generative multi-modal large language model (MLLM) framework. Extensive experimental results have shown that ReAuSE consistently outperforms existing methods, achieving significant improvements across various evaluation metrics on two benchmarks. Future work will focus on extending the application of ReAuSE to domains such as biomedicine and education (Gao et al. 2021).

# Efficiency

ReAuSE requires fewer resources than other retrievalaugmented baselines. ReAuSE unifies three stages into a single MLLM, whereas other baselines require an MLLM for