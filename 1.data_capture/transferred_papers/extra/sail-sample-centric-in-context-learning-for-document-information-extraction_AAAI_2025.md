# SAIL: Sample-Centric In-Context Learning for Document Information Extraction

Jinyu Zhang1\*, Zhiyuan $\mathbf { Y o u } ^ { 2 * }$ , Jize Wang1, Xinyi Le1†

1Shanghai Jiao Tong University 2The Chinese University of Hong Kong {zhang jinyu, jizewang2000, lexinyi} $@$ sjtu.edu.cn, zhiyuanyou@foxmail.com

# Abstract

Document Information Extraction (DIE) aims to extract structured information from Visually Rich Documents (VRDs). Previous full-training approaches have demonstrated strong performance but may struggle with generalization to unseen data. In contrast, training-free methods leverage powerful pre-trained models like Large Language Models (LLMs) to address various downstream tasks with only a few examples. Nonetheless, training-free methods for DIE encounter two primary challenges: (1) understanding the complex relationship between layout and textual elements in VRDs, and (2) providing accurate guidance to pre-trained models. To address these challenges, we propose SAmplecentric In-context Learning (SAIL). SAIL introduces a finegrained entity-level textual similarity to facilitate in-depth text analysis by LLMs and incorporates layout similarity to enhance the analysis of layouts in VRDs. Moreover, SAIL formulates a unified In-Context Learning (ICL) prompt template for various sample-centric examples, enabling tailored prompts that deliver precise guidance to pre-trained models for each sample. Extensive experiments on FUNSD, CORD, and SROIE benchmarks with various base models (e.g., LLMs) indicate that our SAIL outperforms training-free baselines, even closer to the full-training methods, showing the superiority and generalization of our method.

(a) Test sample 05-THEE3. 國 P.500.00 2005- HEESEGCH 9.500.00 2005- HESEDOHO   
Total Ttem □ Total Iten Total Iten:   
[Total, 9.500.00 Total. 9.500.00 Total. 9.500.00   
Cashored 50N1.00 Cashored Tenderedt   
Layout similarity Entity-level text similarity Document-level text similarity   
(b) Examples □ □ 2005-CHEESE J0HN oes □ □ □ □ Total Iter 4 55,454   
1 國 Tendered: otat 60.999 identification of menu   
GR C005H E P.500.09 2005-CHEESEJOHR 9.50, .500.00 nuanitmperiocfemoef nmuenu Y -" quantity of menu  total price of menu   
画     tporticalecofumnteonfuqauftaenrt itdyiscount amount of price paid in cash GPT-4o SAIL(ours) Label description

Code — https://github.com/sky-goldfish/SAIL

# 1 Introduction

Document Information Extraction (DIE) focuses on extracting structured information from Visually Rich Documents (VRDs) such as receipts, forms, and invoices (Park et al. 2019; Huang et al. 2019; Jaume, Ekenel, and Thiran 2019). Previous works, including LayoutLMv3 (Huang et al. 2022), primarily concentrate on full-training methodologies that demand extensive task-specific labeled data. While these models have achieved notable success on the trained dataset, they often struggle to generalize effectively to unseen data, especially when the test data distribution significantly diverges from that of the training data. To address this challenge, training-free DIE methods (He et al. 2023) leverage powerful pre-trained models like Large Language Models (LLMs) that can generalize to unseen data given only a few examples, and thus begin to attract more research interests.

One of the primary challenges in the training-free DIE task is understanding the complex relationship between the document layout and its textual entities using only a few examples. VRDs possess discrete textual elements alongside flexible, inherently structured layouts, complicating the establishment of relationships between textual entities and the extraction of implicit layout information. Even the advanced multi-modal LLMs like GPT-4o (OpenAI 2023b) demonstrate limited effectiveness in performing DIE task. As illustrated in Figure 1(c), GPT-4o misidentifies three entity texts and labels three entity texts incorrectly, highlighting the challenges inherent in the training-free DIE task.

Another significant challenge is providing a clear and effective guidance to pre-trained models (e.g., LLMs). Although these models possess extensive knowledge and capabilities, they necessitate appropriate instructions for optimal performance on specific downstream tasks. Recent research has incorporated In-Context Learning (ICL) within LLMs to enhance the DIE performance (He et al. 2023). This approach involves selecting a few textually similar examples and carefully crafting the in-context prompts with diverse demonstrations for the entire dataset. While this method shows promising results in GPT-3.5 (Brown et al. 2020), the fixed in-context examples fail to effectively guide different LLMs, leading to a significant performance decline when transitioning across different LLMs, as detailed in Table 1.

To address these challenges, we propose a SAmplecentric In-context Learning (SAIL) method. Our method follows two core principles: (a) To enhance LLMs’ understanding of the complex interplay between layout and text within VRDs, the provided prompts must analyze the question from different angles in depth. (b) To ensure precise guidance, it is essential to develop a customized prompt for each test sample. Regarding the first principle, previous methods (He et al. 2023) only adopted rough document-level textual similarity for example selection, which inadequately supports LLMs in understanding textual information in lengthy documents. Consequently, we propose a refined entity-level text similarity for in-depth text analysis. Additionally, we incorporate layout similarity to identify examples that enjoy similar layouts, facilitating LLMs in comprehending complex layout information in VRDs. The three distinct examples are illustrated in figure 1(b). For the second principle, we select distinct examples for each test sample and integrate them into a unified prompt template with clear instructions to devise a tailored sample-centric in-context prompt.

Equipped with these designs, our proposed SAIL demonstrates versatility across various LLMs on multiple benchmarks. SAIL not only stably surpasses all training-free baselines, but even achieves comparable performance to many fully-trained models when implemented with GPT-4. Overall, our main contributions can be summarized as follows:

• We introduce layout similarity and entity-level text similarity, each highlighting unique facets of VRDs, resulting in a thorough and in-depth analysis of VRDs. • To form sample-centric in-context prompts, we propose a unified ICL prompt template applicable to various examples. With clear instructions, LLMs enhance their attention to specific information in the examples. • We conduct extensive experiments on multiple benchmarks including FUNSD, CORD, and SROIE with various base LLMs. Our SAIL achieves superior performance than training-free baselines, even closer to the performance of full-training methods.

# 2 Related Works

Document Information Extraction (DIE). Traditional DIE methods primarily rely on extensive datasets for model pretraining and subsequent fine-tuning on downstream tasks. These methods can be classified into four main categories. The first category consists of grid-based methods (Katti et al. 2018; Zhao et al. 2019; Denk and Reisswig 2019; Kerroumi, Sayem, and Shabou 2021), which encode each document page as a two-dimensional character grid of characters to preserve the document’s layout. The second category, graph-based methods, utilizes Graph Convolutional Networks (GCN) (Qian et al. 2019; Liu et al. 2019) or Graph Neural Networks (GNN) (Tang et al. 2021) for DIE. The third category encompasses transformer-based (Vaswani et al. 2017) methods. Traditional methods design small models in specialized fields. Some methods integrate text semantics and layout modality for model pre-training (Li et al. 2021; Hong et al. 2022; Wang, Jin, and Ding 2022; Wang et al. 2023b), while other methods jointly leverage text, layout, and image modality to enhance document understanding (Da et al. 2023; Xu et al. 2020, 2021; Huang et al. 2022). A recent trend has seen numerous studies employing LLMs’ advanced language capabilities. (Wang et al. 2023a; Perot et al. 2023; Lu et al. 2024; Li et al. 2024; Luo et al. 2024; Fujitake 2024). In contrast to the categories above that necessitate OCR for text and box recognition, the final category aims to bypass the OCR process and establish end-toend models (Wang et al. 2021; Kim et al. 2022; Liu et al. 2024b; Mao et al. 2024; Hu et al. 2024; Abramovich et al. 2024). Despite the notable performance of many methods, they demand retraining for specific downstream tasks.

In-Context Learning (ICL). Brown et al. (2020) discovered that pre-trained LLMs can address unseen tasks using only a few examples without weight updates through ICL. From then on, ICL has been widely adopted in question answering (Yang et al. 2022; Liu et al. 2023; Wang et al. 2024), multi-modal named entity recognition (Cai et al. 2023), and dialogue improvement (Meade et al. 2023; Hu et al. 2022).

ICL-based DIE. ICL presents a viable approach for performing the DIE task with minimal examples. ICLD3IE (He et al. 2023), the first work to construct ICL prompts for DIE, utilizes diverse demonstrations through examples selected via text semantic search. Nonetheless, ICLD3IE exhibits limited generalization capabilities to novel LLMs, primarily due to its reliance on fixed examples and handcrafted prompts. Our method clearly distinguishes itself from this work. First, we dynamically select unique examples for each test sample, in contrast to ICL-D3IE’s fixed examples. Second, we employ a unified template to construct prompts that can be generalized to various LLMs, while ICL-D3IE adopts specifically designed prompts that are less adaptable to new models. Third, we demonstrate that relying solely on document-level text similarity is inadequate for identifying optimal examples, and thus introduce layout similarity and entity-level text similarity for enhanced performance. With these designs, our method achieves better results than ICL-D3IE across various base LLMs.

# 3 Methods

# 3.1 Problem Formulation

Training-free DIE leverages pre-trained models (e.g., LLMs) to extract specified categories of text information (e.g., company, address, and date (Huang et al. 2019)) from VRDs. Specifically, given a document image $I$ , the goal is to label all entities within $I$ . First, entity texts $\mathbf { \mathit { \check { T } } } = \{ t _ { 1 } , t _ { 2 } , . . . , t _ { n _ { \mathrm { e } } } \}$ and their corresponding boxes $B \ =$

Training dataset T: entity text Examples selection Sample-centric  prompt o:xenstity-level text embedding Training sample a)LaCbeal:ndidate labels illustration $( C _ { \mathrm { c l } } )$ $E _ { \mathrm { e n t } }$ $E _ { \mathrm { d o c } }$ : document-level text embedding Test sample MENU.NM : name of menu 𝐼ሚ: layout image MENU.NUM : identification of menu 三 b) Entity-level text demonstrations $\cdot ( c _ { \mathrm { e t } } )$ filter Sentence- 𝐸ent × 𝑚f Entity level Textueanltliytiesismilar {etnetxit:y":BCMbHEkANBNUe.GnNEgMl2}7N,a5s0i"0,"B,Boxo:x[:9[353132340665734],265], OCR BERT join T × 𝑚d 𝐸doc × 𝑚d c) Layout demonstrations (𝐶l) Document:{text:"1 x",Box:[10 18 54 31],entity:MENU.CNT} $\pmb { { \cal B } } \times m _ { \mathrm { d } }$ 𝐼ሚ × 𝑚d Layout similar These are the information extracted from the document… Layout image documents -BTashed" MonEtNhUe.pCrNovTi"d,e"dMiEnfNoUr.mNatMi"o,n,a nwde"cMaEnNsUe.ePtRhIaCt:E" entities are generally located in the same row, with the quantity, menu name, construction and price listed together. B 𝐼ሚ d) Document-level text demonstrations $\left. C _ { \mathrm { d t } } \right. _ { \mathrm { \ell } } ^ { \mathrm { \ell } }$ T OCR join 𝐸doc Texdtoucaullymesintmsilar 3AQ:7:{]t}e…xt,:"w1"h",aBtoaxr:[e[3t3he1l1a4b47e7 3s31f]o,}re{tnehtixettys::"eCMtoEekxNtesU(?.LC)"N,BT}o x:[98 10 228 1 SentenceBERT filter $E _ { \mathrm { e n t } } \times n _ { \mathrm { f } }$ Document level e) Test question 𝜑 𝑇, 𝐵 Q:{text:"BASO KUAH",Box:[152 430 317 455]}…, what are the labels for these texts? infer {text:“BASO KUAH”,Box:[152 430 317 455],entity:MENU.NM}… Test sample

$\{ b _ { 1 } , b _ { 2 } , . . . , b _ { n _ { \mathrm { e } } } \}$ are recognized from $I$ by an OCR system, where $n _ { \mathrm { e } }$ is the total number of entities in the document image. To effectively utilize LLMs, in-context prompts $C$ are designed to convey the extraction intention. For ICLbased DIE, $C$ is constructed by selecting several examples demonstrating how to solve DIE tasks. With these in-context prompts as illustrations, LLMs are tasked with generating labels $Y _ { \mathrm { p r e d } }$ for all detected entities. The process is achieved by maximizing the conditional probability $P ( { Y \vert { T } , { B } } )$ while incorporating the prompts $C$ as an additional condition:

$$
P ( Y | T , B ) = \frac { 1 } { n _ { \mathrm { e } } } \sum _ { k = 1 } ^ { n _ { \mathrm { e } } } P _ { \mathrm { L M } } ( l _ { k } | C , \varphi ( T , B ) ) ,
$$

where $P _ { L M }$ is the conditional probability predicted by the LLMs, and $\varphi$ denotes the operation of converting the entity texts and boxes into a textual format suitable for LLMs’ input. In training-free DIE, the construction of effective incontext prompt $C$ is crucial, which is the primary focus of this work. Finally, the predicted labels $Y _ { \mathrm { p r e d } }$ are evaluated using F1 scores against the ground truth labels $Y _ { \mathrm { g t } }$ .

# 3.2 Overview Framework

To maximize $P ( { Y \vert { T } , { B } } )$ with the in-context prompt $C$ , we propose SAIL, a sample-centric in-context prompt construction method for DIE. SAIL focuses on designing $C$ for individual samples by automatically selecting tailored layout examples, document-level text similarity examples, and entitylevel text similarity examples based on the test sample, subsequently leveraging these examples to generate $C$ .

The overall architecture, illustrated in Figure 2, comprises five steps. Firstly, the test document image and $m$ training document images are processed through OCR to extract entity texts $T$ and boxes $B$ . Secondly, $T$ are transformed into entity-level text embeddings $E _ { \mathrm { e n t } }$ and document-level text embeddings $E _ { \mathrm { d o c } }$ . $B$ are used to construct layout image $\tilde { I }$ . Thirdly, $E _ { \mathrm { e n t } }$ , $\tilde { I }$ and $E _ { \mathrm { d o c } }$ are used to select textually similar entities, layout similar documents, and textually similar documents for the test sample. Then, these selections are substituted into the prompt template to form a tailored incontext prompt $C$ . Finally, LLM performs inference with $C$ and question $\varphi ( T , B )$ to generate predicted labels $Y _ { \mathrm { p r e d } }$ .

# 3.3 Document-Level Text Similarity Examples

To improve the capability of ICL, we employ text semantic search to select the nearest training document examples for a given test sample (Liu et al. 2022). The entity texts $T$ extracted from a document image are concatenated into a single sentence and encoded with Sentence-BERT (Reimers and Gurevych 2019), resulting in a text semantic embedding $E _ { \mathrm { d o c } }$ for the document. We determine the nearest training examples by computing the document-level text similarity $T _ { \mathrm { s i m \_ d o c } }$ between the test embedding $E _ { \mathrm { d o c } } ^ { \mathrm { t e s t } }$ and $m$ training embeddings $E _ { \mathrm { d o c } } ^ { \mathrm { t r a i n } }$ using the cosine similarity score:

$$
T _ { \mathrm { s i m \_ d o c } } = \frac { E _ { \mathrm { d o c } } ^ { \mathrm { t e s t } } \cdot E _ { \mathrm { d o c } } ^ { \mathrm { t r a i n } } } { | | E _ { \mathrm { d o c } } ^ { \mathrm { t e s t } } | | \ | | E _ { \mathrm { d o c } } ^ { \mathrm { t r a i n } } | | } .
$$

# 3.4 Entity-Level Text Similarity Examples

The document-level text similarity $T _ { \mathrm { s i m \_ d o c } }$ between a lengthy text document and the found text-similar documents is notably low. To facilitate LLMs in generating text with more relevant examples for learning, we propose entity-level text similarity examples, as shown in Figure 2.

![](images/dc3d5c72692f35e4cc1258a50c4b89adb7f62d5846f2d67ef8f3a9f9844c11b9.jpg)  
Figure 3: Illustration of layout similarity evaluation, including drawing boxes onto a blank image, cropping and resizing to form layout image, and comparing layout images.

Entity texts $T ~ = ~ \{ t _ { 1 } , t _ { 2 } , . . . , t _ { n _ { \mathrm { e } } } \}$ recognized by OCR are filtered to exclude texts consisting solely of numbers, which provide minimal semantic content. Subsequently, the filtered $m _ { \mathrm { f } }$ training entity texts and $n _ { \mathrm { f } }$ test entity texts are encoded using Sentence-BERT to derive the semantic embedding $E _ { \mathrm { e n t } }$ . The entity-level text similarity $T _ { \mathrm { s i m \_ e n t } }$ is computed from the semantic embedding $E _ { \mathrm { e n t } }$ by employing the cosine similarity score, defined as follows:

$$
T _ { \mathrm { s i m \_ e n t } } = \frac { E _ { \mathrm { e n t } } ^ { \mathrm { t e s t } } \cdot E _ { \mathrm { e n t } } ^ { \mathrm { t r a i n } } } { | | E _ { \mathrm { e n t } } ^ { \mathrm { t e s t } } | | \ | | E _ { \mathrm { e n t } } ^ { \mathrm { t r a i n } } | | } .
$$

We select $n _ { \mathrm { s } }$ textually similar entities for each test entity by nearest neighbor search and obtain $n _ { \mathrm { f } } \times n _ { \mathrm { s } }$ examples.

# 3.5 Layout Similarity Examples

To identify documents with similar layouts, we introduce a layout similarity assessment methodology, illustrated in Figure 3. Firstly, all $b _ { i }$ from boxes $B = \{ b _ { 1 } , \stackrel { \cdot } { b } _ { 2 } , . . . , b _ { n _ { \mathrm { e } } } \}$ are rendered as black rectangles on a blank image. Subsequently, we define the information area as the minimal region that contains all entity texts and crop the layout image to maintain a 10-pixel margin between the information area and the image borders. Next, we standardize the layout image dimensions through resizing. Finally, we select $n _ { \mathrm { s } }$ layout similar documents by calculating the layout similarity $L _ { \mathrm { s i m } }$ between the training layout image $\tilde { I } ^ { \mathrm { t r a i n } }$ and the test layout image $\tilde { I } ^ { \mathrm { t e s t } }$ using Mean Square Error (MSE) loss:

$$
L _ { \mathrm { s i m } } = \frac { 1 } { \mathrm { M S E } } = \frac { n _ { \mathrm { l } } } { ( U - V ) ^ { \mathrm { T } } ( U - V ) } ,
$$

where $U , V$ are the pixel matrix of $\tilde { I } ^ { \mathrm { t r a i n } }$ and $\tilde { I } ^ { \mathrm { t e s t } }$ , and $n _ { \mathrm { l } }$ is the total number of pixels in the layout image.

Moreover, to enhance the understanding of layouts by LLMs, we substitute the boxes from the cropped image $B ^ { \prime }$ for all documents in the prompt instead of using $B$ .

# 3.6 Sample-Centric ICL Prompt Template

To construct $C$ for an individual test sample, we propose an adaptive sample-specific ICL prompt template. The template is comprised of 5 parts: candidate labels illustration

$C _ { \mathrm { c l } }$ , entity-level text demonstrations $C _ { \mathrm { e t } }$ , layout demonstrations $C _ { 1 }$ , document-level text demonstrations $C _ { \mathrm { d t } }$ and test question $\varphi ( T , B )$ , as shown on the right of Figure 2.

Candidate labels illustration $C _ { \mathrm { c l } }$ enumerates all potential labels for the DIE task. For abbreviated labels, a corresponding natural language description is appended.

Entity-level text demonstrations $C _ { \mathrm { e t } }$ present textually similar entities. The prompt $p _ { \mathrm { e } }$ “Sample text and corresponding label:” in conjunction with the labels of the selected $n _ { \mathrm { s } }$ textually similar entity examples $Y _ { \mathrm { e t } }$ , formulates the entity-level text similarity demonstrations:

$$
C _ { \mathrm { e t } } = { \mathrm { C O N C A T } } [ p _ { \mathrm { e } } , Y _ { \mathrm { e t } } ] .
$$

Layout demonstrations $C _ { 1 }$ aim to facilitate LLMs in analyzing the layout of the test document. After obtaining $n _ { \mathrm { s } }$ layout similar documents, we introduce a layout analysis step. This step enables LLMs to comprehend the overall document structure and the relationship between layout and label selection. The layout analysis prompt $\displaystyle p _ { \mathrm { a } }$ is defined as: “These are the information extracted from the document through OCR, and the Box is the position of the text in the document. Please analyze where each label is generally located in the document.”, which can apply to any dataset. The labels of layout-similar documents $Y _ { \mathrm { l } }$ are input into LLMs together with $\boldsymbol { p _ { \mathrm { a } } }$ , allowing LLMs to analyze the layout information in layout-similar documents by themselves. The resulting output from the LLM is denoted as $\textstyle A _ { \mathrm { l } }$ . A layout similarity demonstration $C _ { 1 }$ is formulated as follows:

$$
C _ { 1 } = \mathrm { C O N C A T } [ Y _ { \mathrm { l } } , p _ { \mathrm { a } } , A _ { \mathrm { l } } ] .
$$

Document-level text demonstrations $C _ { \mathrm { d t } }$ showcase textually similar documents in question-answer format, guiding LLMs to produce answers in a specific format. The textually similar documents $X _ { \mathrm { d t } }$ , the ground truth answer $ { Y _ { \mathrm { d t } } }$ and the DIE instruction $p _ { \mathrm { q } }$ such as “What are the labels for these texts?” form the Document-level text demonstration prompt:

$$
C _ { \mathrm { d t } } = \mathrm { C O N C A T } [ X _ { \mathrm { d t } } , p _ { \mathrm { q } } , Y _ { \mathrm { d t } } ] .
$$

Finally, the test question $\varphi ( T , B )$ for the test sample is:

$$
\varphi ( T , B ) = \mathrm { C O N C A T } [ T , B ^ { \prime } , p _ { \mathrm { q } } ] .
$$

# 3.7 Inference

After selecting a diverse set of examples, ICL prompts facilitate LLMs in generating entity labels $Y _ { \mathrm { p r e d } }$ . This process is mathematically represented as follows:

$$
P ( Y | T , B ) = \frac { 1 } { n _ { \mathrm { e } } } \sum _ { k = 1 } ^ { n _ { \mathrm { e } } } P _ { \mathrm { L M } } ( l _ { k } | C _ { \mathrm { c l } } , C _ { \mathrm { e t } } , C _ { 1 } , C _ { \mathrm { d t } } , \varphi ( T , B ) ) .
$$

Subsequently, entity labels $Y _ { \mathrm { p r e d } }$ are extracted from the generated output. We assess the accuracy of $Y _ { \mathrm { p r e d } }$ against the ground truth labels $Y _ { \mathrm { g t } }$ utilizing the F1 score.

# 4 Experiments

# 4.1 Datasets, Metrics, and Details

FUNSD (Jaume, Ekenel, and Thiran 2019) is a dataset for understanding the content of tables in scanned documents.

It contains 149 tables and 7,411 entities in the training set, and 50 tables and 2,332 entities in the test set. In the DIE task, the candidate labels of the FUNSD dataset include “Header”, “Question”, “Answer”, and “Other”.

SROIE (Huang et al. 2019) is another scanned receipt understanding dataset, containing 626 receipts in the training set and 347 in the test set. The DIE task needs to extract “company”, “date”, “address”, and “total” information.

CORD (Park et al. 2019) is a receipt understanding dataset that contains 800 training data, 100 test data, and 100 validation data. This dataset features 30 detailed and hierarchical labels, much more than the above two datasets.

Metrics. Following previous works (He et al. 2023), we adopt entity-level F1 score, precision and recall as metrics.

Details. We evaluate our method using three LLMs: the open-source ChatGLM3 (THUDM 2023) and the closedsource GPT-3.5 (OpenAI 2023a) and GPT-4 (OpenAI 2023b). Specifically, we use the $\mathtt { c h a t g l m 3 - 6 b - 3 2 k }$ version for ChatGLM3, gpt-3.5-turbo API version for GPT-3.5, and $A P ^ { \mathrm { { c } - 4 0 } }$ API version for GPT-4. For GPT3.5 and GPT-4o, we set the temperature parameter to 0 to enhance the reproducibility. In the case of GPT-4o, we only provide text prompts as input, while also testing its multimodal capabilities by providing document images and clear task instructions. In our experiments, for each test document, we select four textually similar documents and four layout-similar documents as examples due to the limitation of prompt token number. Furthermore, for each filtered test entity, we choose four textually similar entity examples.

# 4.2 Results on DIE Benchmarks

Baselines. We compare our SAIL against baseline models including BERT (Devlin et al. 2019), LiLT (Wang, Jin, and Ding 2022), BROS (Hong et al. 2022), XYLayoutLM (Gu et al. 2022), LayoutLM (Gu et al. 2022), LayoutLMv2 (Xu et al. 2021), and LayoutLMv3 (Huang et al. 2022) in both full-training and few-shot settings. We borrow their metrics from (He et al. 2023). Training-free methods including standard ICL and ICL-D3IE (He et al. 2023) are also compared. ICL-D3IE only reports the performance of standard ICL and ICL-D3IE with GPT-3.5, so we evaluate their performance with GPT-4 and ChatGLM3 using their official repositories.

Quantitative results are presented in Table 1. First, overall, our method stably outperforms ICL-D3IE across different LLMs on all datasets. Second, when switching the LLM from GPT-3.5 to ChatGLM3, the performance drop of ICL-D3IE is significantly larger than our SAIL (e.g., $- 7 3 . 8 \%$ $\nu s . - 1 2 . 7 3 \%$ in SROIE), demonstrating that our method has better robustness and adaptability to various LLMs. Third, the performance of ICL-D3IE degrades slightly when transitioning from GPT-3.5 to the more advanced GPT-4 on FUNSD and SROIE datasets, further indicating its incompatibility with new LLMs. However, in all datasets, our method achieves better performance on more advanced GPT-4 than on GPT-3.5, which is intuitive and reasonable. These results demonstrate the advantages of our method.

Qualitative results are illustrated in Figure 4. ICL-D3IE incorrectly predicts the entities on the three left green boxes as “answer”, while our SAIL accurately identifies them as “question”. This indicates that fixed examples in ICL-D3IE are insufficient to guide LLMs in effectively learning the relationship between discrete texts, highlighting the importance of selecting diverse examples for each test sample.

Table 1: Quantitative results with F1 metric. Our SAIL stably surpasses baselines across various base LLMs.   

<html><body><table><tr><td>Setting</td><td colspan="2">Methods BERTBASE</td><td>FUNSD CORD SROIE</td><td>90.99</td></tr><tr><td>Full-Training</td><td colspan="2">XYLayoutLMBASE LayoutLMBASE LayoutLMv2BASE LayoutLMv3BASE LiLTBASE BROSBASE</td><td>60.26 89.68 88.41 96.07 83.05 95.73 83.35 94.45 79.27 91.06 82.76 94.95 90.29 96.56</td><td>94.68 95.48 95.74 94.38 96.25 96.89</td></tr><tr><td>Few-Shot</td><td colspan="2">BERTBASE XYLayoutLMBASE LayoutLMBASE LayoutLMv2BASE LayoutLMv3BASE LiLTBASE BROSBASE</td><td>38.76 38.88 54.88 69.12 59.46 72.78 65.44 69.16 32.49 40.19 71.42 65.71 70.67 70.13</td><td>38.76 84.03 76.78 75.66 76.79 81.81 79.13</td></tr><tr><td rowspan="3">Training-Free</td><td>ChatGLM3</td><td>ICL-D3IE SAIL (ours) Standard ICL</td><td>40.93 35.90 58.24 72.76</td><td>67.30 81.37 36.44 18.83 83.04 85.03 68.34 82.11</td></tr><tr><td>GPT-3.5</td><td>ICL-D3IE SAIL (ours)</td><td>83.66 87.13 83.48 95.80</td><td>92.63 97.76</td></tr><tr><td>GPT-4</td><td>Standard ICL ICL-D3IE SAIL (ours)</td><td>75.15 90.22 78.94 87.47 84.67 96.41</td><td>96.00 89.23 98.18</td></tr></table></body></html>

# 4.3 Comparison with Multi-modal LLMs

Baselines. Recent years have witnessed the rapid development of multi-modal LLMs (MLLMs) represented by GPT4o (OpenAI 2023b). To further validate the effectiveness of our method, we also compare our SAIL with MLLMs including open-source LLaVA-1.5 (Liu et al. 2024a) and proprietary GPT-4o. We provide these MLLMs with explicit and detailed instructions to inform the task definition.

Quantitative results are provided in Table 2. The opensource LLaVA exhibits limited DIE capabilities, resulting in a low F1 score (e.g., $0 . 7 \%$ in FUNSD). The proprietary GPT-4o significantly outperforms LLaVA $( 5 0 . 7 2 \%$ vs $0 . 7 \%$ in FUNSD), yet still falls short when compared to specialized DIE methods. Therefore, despite their rapid evolution, MLLMs still underperform in the DIE task, highlighting the importance and contribution of our proposed work.

# 4.4 Ablation Studies

Effect of Adaptive Example. We assess the influence of adaptive examples by employing both fixed and adaptive examples to construct in-context prompts within the same prompt template. The base LLM is selected as GPT-3.5, and the results are illustrated in Table 3. The utilization of adaptive examples results in superior F1 scores, confirming the

Table 2: Performance comparison with multi-modal LLMs. Multi-modal LLMs even powerful GPT-4o still struggle with DIE tasks and our method significantly surpasses GPT-4o and LLaVA-v1.5-7B.   

<html><body><table><tr><td rowspan="2">Methods</td><td colspan="3">SROIE</td><td colspan="3">CORD</td><td colspan="3">FUNSD</td></tr><tr><td>F1</td><td>Precision</td><td>Recall</td><td>F1</td><td>Precision</td><td>Recall</td><td>F1</td><td>Precision</td><td>Recall</td></tr><tr><td>GPT-40</td><td>47.49</td><td>46.77</td><td>48.24</td><td>71.53</td><td>82.96</td><td>62.87</td><td>50.72</td><td>73.01</td><td>38.85</td></tr><tr><td>LLaVA-v1.5-7B</td><td>2.32</td><td>5.49</td><td>1.47</td><td>8.85</td><td>67.39</td><td>4.74</td><td>0.70</td><td>61.54</td><td>0.35</td></tr><tr><td>SAIL (ours)</td><td>98.18</td><td>97.72</td><td>98.64</td><td>96.41</td><td>96.41</td><td>96.41</td><td>84.67</td><td>84.67</td><td>84.67</td></tr></table></body></html>

Table 3: Ablation study of adaptive examples with F1 metric. Adaptive examples is superior than fixed examples.   

<html><body><table><tr><td>Setting</td><td>FUNSD</td><td>CORD</td><td>SROIE</td></tr><tr><td>Fixed example</td><td>74.23</td><td>82.35</td><td>91.08</td></tr><tr><td>Adaptive example</td><td>83.48</td><td>95.80</td><td>97.76</td></tr><tr><td>△</td><td>一 9.25</td><td>13.45</td><td>6.68</td></tr></table></body></html>

Question:{text:"# CASES",Box:[307 426 359 439]} {text:"KENT III K. S.", Box:[306 454 379 465]}......{text:"1", Box:[429 482 440 495]}……, What are the labels for these texts?   
(a) ICL-D3IE (b) Ours CASES CASES KENT IIIK.S. KENT IIIK.S. KENTII100 KENTII100 TRUEK.S. TRUEK.S.   
Answer: {text:"# Answer: {text:"#   
CASES",Box:[307 426 359 CASES",Box:[307 426 359   
439],entity:header} 439],entity:header}   
{text:"KENT III K. {text:"KENT III K.   
S.",Box:[306 454 379 S.",Box:[306 454 379   
465],entity:answer}  465],entity:question}   
{text:"KENT III 100",Box:[306 {text:"KENT III 100",Box:[306   
466 376 479],entity:answer} 466 376 479],entity:question}   
{text:"TRUE K. S.",Box:[306 {text:"TRUE K. S.",Box:[306   
480 363 493],entity:answer} 480 363 493],entity:question}

effectiveness of our method. Among the three datasets, the performance improvement with adaptive examples is most pronounced in the CORD dataset $( 1 3 . 4 5 \% )$ . Note that the CORD dataset contains 30 labels, much more complex than the other two datasets with only four labels. This suggests that sample-centric examples could more effectively guide the LLMs to comprehend the layout and text information especially in complex situations.

Effect of Different Examples. We conduct ablation experiments using GPT-3.5 to evaluate the influence of different examples, as shown in Table 5. In #0, where none of the three examples are available, we employ fixed random examples to instruct the LLM to generate answers in a specific format, simplifying the label extraction. The highest F1 score is observed when document-level text similarity examples, layout examples, and entity-level text similarity examples $( \# 4 )$ are used, validating the efficiency of the three examples. The addition of layout examples (#1 vs. #2) or entity-level text similarity examples (#1 vs. #3) to documentlevel text similarity examples results in superior F1 scores.

Table 4: Performance comparison of the example order in the prompt with F1 metric in the CORD dataset.   

<html><body><table><tr><td>1 Layout /</td><td rowspan="2">Ascending</td><td rowspan="2">Descending</td></tr><tr><td>Text</td></tr><tr><td>Ascending</td><td>95.19</td><td>94.73</td></tr><tr><td>Descending</td><td>94.73</td><td>95.80</td></tr></table></body></html>

<html><body><table><tr><td>#</td><td colspan="3">Similar FUNSD CORD SROIE Text-Doc.Layout Text-Ent.</td></tr><tr><td></td><td></td><td></td><td>69.87 83.04</td><td>95.08</td></tr><tr><td>0 1</td><td>√</td><td>69.60</td><td>92.13</td><td>96.38</td></tr><tr><td>2</td><td>√ √</td><td>73.13</td><td>92.97</td><td>97.24</td></tr><tr><td>3</td><td>√</td><td>√</td><td>81.67 92.51</td><td>97.13</td></tr><tr><td>4</td><td>√ √</td><td>√</td><td>83.48 95.80</td><td>97.76</td></tr></table></body></html>

![](images/7f713d280be10559beeef51b90222e45c3aa478e21564ef35fff26bd80cba51c.jpg)  
Figure 4: Case study on performance comparison of (a) ICLD3IE and (b) our SAIL. ICL-D3IE wrongly predicts the three green boxes on the left as “answer”. In contrast, our proposed SAIL correctly predicts them as “question”.   
Table 5: Ablation study of various similarity with F1 metric. Text-Doc., Layout, & Text-Ent. mean textual similar documents, layout similar documents, & textual similar entities.   
Figure 5: Ablation study of layout analysis. “w/o LA” means without adding layout analysis. Adding layout analysis achieves higher F1 scores across all three datasets.

For the long text FUNSD dataset, the F1 score with document-level text similarity examples is even lower than fixed examples (#0 vs. #1). This could be attributed to the inherent randomness of LLM generation, but it also signi

Question: {text:"Bubur Polos $^ +$ Telur",Box:[14 10 444 43]}{text:"13.000",Box:[285 42 419 73]}{text:"13.000",Box:[591 47 731 74]}……, What are the labels for these texts? (a1) Without layout similar examples Answer: {text:"Bubur Polos $^ +$ Telur",Box:[14 10 444 MENU.NM 43],entity:MENU.NM}{text:"13.000",Box:[285 42 419 MENU.PRICE MENU.PRICE 73]1,e7n4ti]t,ye:nMtitEy:NMU.EPNRUI.CPER}I{tCeEx}t:"…13….000",Box:[591 47 (a2) With layout similar examples Answer: {text:"Bubur Polos $^ +$ Telur",Box:[14 10 444 MENuNo1os+Telur MENU.PRICE 473],entity:MENU.NUNMI}T{tPexRtI:"C1E3}.{0te0x0t":,"B1o3.x0:[0208"5,B4o2x:4[5191 (a) MENU.UNITPTICE 47 731 74],entity:MENU.PRICE}…… Question: {text:"COMPOUND SENSITIVE TO",Box:[14 630 104 642]}{text:"☐ AIR",Box:[33 656 56 665]}{text:"☐ HEAT",Box:[69 654 98 665]}……, What are the labels for these texts? (b1) Without entity-level text similar examples Answer: {text:"COMPOUND SENSITIVE TO",Box: COUPOUNO SENSITVETO [14 630 104 642],entity:question}{text:"☐ AIR",Box: [33 656 56 665],entity:answer}{text:"☐ HEAT",Box: HEAT OMOISTURE OIHEA [69 654 98 665],entity:answer}  (b2) With entity-level text similar examples Answer: {text:"COMPOUND SENSITIVE TO",Box: COUPOUND SENSIETO [14 630 104 642],entity:header}{text:"☐ AIR",Box: [33 656 56 665],entity:question}{text:"☐ HEAT",Box: A HEAT MOISTURE OTHEA [69 654 98 665],entity:question} (b)

fies that in lengthy documents, document-level text similarity examples do not provide effective guidance for the LLM. In the FUNSD dataset, adding entity-level text similarity examples ( $1 0 . 3 5 \%$ , #2 vs. #4) is much superior than adding layout similarity examples ( $1 . 8 1 \%$ , #3 vs. #4), suggesting that entity-level text similarity examples are more important for lengthy documents. For the CORD and SROIE datasets, removing layout similarity examples (#3 vs. #4) causes a greater F1 score decrease than omitting entity-level text similarity examples (#2 vs. #4), indicating the higher significance of layout information for these two datasets.

Effect of Example Order. We perform experiments on the CORD dataset using GPT-3.5 to test the effect of example order, as detailed in Table 4. When layout-similar and text-similar document examples are arranged in a consistent order based on their similarity, F1 scores tend to be higher. This phenomenon may result from improved attention allocation within the LLM due to the consistent ordering. Furthermore, the highest F1 scores are observed when layoutsimilar and text-similar examples are sorted from high to low similarity concerning the test sample. This suggests that the LLM can capitalize on the information presented first.

Effect of Layout Analysis. Our methodology requires the LLM to perform layout analysis on our searched layoutsimilar examples. To assess the impact of layout analysis, we conduct comparative experiments with / without the layout analysis on the FUNSD, CORD, and SROIE datasets using the GPT-3.5. As illustrated in Figure 5, the results indicate that F1 scores are consistently higher when incorporating layout analysis compared to only using layout-similar examples across all datasets. This suggests that layout analysis is able to enhance the LLM’s comprehension of layout.

Case Study. Figure 6(a) illustrates a comparison from the CORD dataset regarding the inclusion of layout demonstrations in the prompt. Using the prompt without layout similar demonstrations, the LLM predicts two “ $1 3 . 0 0 0 ^ { \circ }$ both as “MENU.PRICE”, while our SAIL distinguishes the left “13.000” as “MENU.UNITPRICE” and the right “13.000” as “MENU.PRICE”. This outcome underscores the necessity of incorporating layout demonstrations for LLMs to grasp document structure effectively. Figure 6(b) showcases a comparison from the FUNSD dataset about the addition of entity-level text demonstrations in the prompt. Upon omitting these demonstrations, the LLM mistakenly predicts “COMPOUND SENSITIVE TO” as “question” and incorrectly classifies the four subsequent entities as “answer”. Although this prediction makes sense in terms of layout, it fails to correspond with the textual context, highlighting the critical role of entity-level text similarity examples.

# 5 Conclusion

In this work, we propose SAIL, a sample-centric ICL method for training-free DIE task. Our SAIL leverages layout similarity and entity-level text similarity in combination with a unified prompt template, constructing tailored prompts for each test sample, showcasing superiority over baselines on three DIE benchmarks with different LLMs.