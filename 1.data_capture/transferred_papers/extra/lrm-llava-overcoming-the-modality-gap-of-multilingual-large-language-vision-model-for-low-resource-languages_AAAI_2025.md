# LRM-LLaVA: Overcoming the Modality Gap of Multilingual Large Language-Vision Model for Low-Resource Languages

Junchen Li1\*, Qing Yang1\*, Bojian Jiang1, Shaolin $\mathbf { Z } \mathbf { h } \mathbf { u } ^ { 2 \dag }$ ,Qingxuan Sun1

1Du Xiaoman Finance, Beijing, China 2 Tianjin University, Tianjin, China {lijunchen, yangqing, jiangbojian, sunqingxuan}@duxiaoman.com zhushaolin@tju.edu.cn

# Abstract

Multilingual large language-vision models (LVLMs), which understand and generate both text and images across multiple languages, have achieved remarkable performance on English-centric multimodal generation tasks. However, their performance on non-English tasks has been underwhelming. One major challenge with multilingual LVLMs is the modality gap between visual inputs and multilingual textual inputs/outputs due to the lack of high-quality multilingual training data. In this paper, we propose LRM-LLaVA, a multilingual large language-vision model designed for low-resource languages to overcome the modality gap. It is composed of four components: a visual encoder, a multilingual large language model, a vision-text representation projector, and a cross-modal regularizer. Both the projector and regularizer aim at reducing the modality gap and improving multilingual performance. To train LRM-LLaVA, we employ a two-stage training strategy including pre-training and instruction finetuning. Meanwhile, we construct a multilingual visual question answering dataset based on English open-source datasets and adopt multiple task instructions. To evaluate the performance of LVLMs across various languages, we construct four multilingual benchmarks for 10 languages, based on English open-source benchmarks. Experimental results show that LRM-LLaVA achieves competitive performance compared to other multilingual LVLMs of similar parameters.

![](images/13f71ad7cb8c330d30959459600a66f0d31a163ab68bcac81aa69bcbeb2d11cd.jpg)  
Figure 1: Performance of various large language-vision models on multilingual MME $\mathrm { F u }$ et al. 2023) benchmark.

et al. 2024). However, their performance on non-English tasks has been underwhelming as shown in Figure 1. This disparity is due to the limited availability of high-quality multilingual training data for low-resource languages.

# Introduction

Multilingual large language-vision models (LVLMs), which receive images and multilingual text as input and then conduct multi-round dialogues or provide corresponding text reasoning (Zhou et al. 2024), have recently attracted significant interest (You et al. 2023; Ying et al. 2024). Recent mainstream multilingual LVLMs are mainly trained on English image-text pairs to achieve semantic alignment between text and images, e.g., BLIP-2 (Li et al. 2023b), LLaVA (Liu et al. 2024b), and LLaVAR (Zhang et al. 2023). Those multilingual LVLMs have achieved remarkable performance on English multimodal generation tasks, such as image captioning (Wang et al. 2024), visual question answering (Liu et al. 2024c), and image-text retrieval (Zhu

As a cross-modal task, a major challenge of multilingual LVLMs for low-resource languages is the representation discrepancy across the textual and visual modality. This discrepancy arises because the modalities have distinct semantic spaces, and aligning them requires learning complex relationships between visual features and textual descriptions. The lack of sufficient text-image pairs for non-English languages makes it incredibly difficult for multilingual LVLMs to effectively bridge this gap (Zhu et al. 2023b).

Previous efforts for multilingual LVLMs for non-English languages, e.g., the method presented in Li et al. (2023d), use contrastive learning to align visual and textual representations by pulling positive pairs (text and image depicting the same concept) closer while pushing negative pairs (text and image depicting different concepts) further apart. Such methods may not be able to sufficiently leverage crossmodal knowledge as they require careful selection of training data and techniques to minimize the discrepancy across the textual and visual modalities and do not explicitly deal with the modality gap issue for low-resource languages.

To mitigate this problem, we propose a multilingual LVLMs framework which can bridge the modality gap between non-English languages and images, and based on which we train LRM-LLaVA. LRM-LLaVA is composed of four components: a visual encoder ViT-L/14 (Radford et al. 2021), a backbone multilingual large language model Vicuna-13B (Chiang et al. 2023), a vision-text representation projector, and a cross-modal regularizer. We build a multilingual visual question answering dataset including $4 . 8 \mathbf { M }$ image-text pairs based on English open-source datasets and construct multiple monolingual or bilingual task instructions. We use a two-stage strategy to train LRMLLaVA. The first pre-training stage aims to achieve lowcost alignment of multilingual features and visual features through a multilingual task instruction. The second finetuning stage aims to improve the multilingual instruction following ability through three multilingual task instructions. The regularizer is also optimized to force LRM-LLaVA to generate the same representations for the same input in different modalities. We construct four LVLMs multilingual benchmarks in 10 languages based on four English opensource benchmarks and evaluate their reliability. Experimental results show that LRM-LLaVA achieves competitive performance compared to other multilingual LVLMs of similar parameters and substantially improves multilingual imagetext understanding without compromising English ability.

To summarize, our contributions are as follows:

• We propose a multilingual LVLMs framework for lowresource languages. We build a multilingual visual question answering dataset based on English open-source datasets and construct multiple task instructions to improve the multilingual ability of LVLMs. • We build four multilingual benchmarks for 10 languages and evaluate their reliability, based on which we benchmark the mainstream LVLMs. • Based on LVLMs multilingual framework, we train LRM-LLaVA on a 13B-parameter large language model and a 0.6B-parameter visual encoder, which significantly outperforms other LVLMs of similar parameters.

# Related Work

The emergence of LVLMs has revolutionized multimodal understanding, enabling natural interactions between users and systems through images and text. Generally speaking, the architecture of LVLMs connects visual encoders such as CLIP (Radford et al. 2021) and SigLIP (Zhai et al. 2023) with large language models such as Vicuna (Chiang et al. 2023), LLaMA (Touvron et al. 2023), and Qwen (Bai et al. 2023a) through a cross-modal connection layer. Models like BLIP-2 (Li et al. 2023b), LLaVA (Liu et al. 2024b), and ShareGPT4V (Chen et al. 2023) have demonstrated the potential of LVLMs for a wide range of applications.

However, extending the capabilities of LVLMs to encompass multiple languages, especially for low-resource languages, presents significant challenges. The scarcity of high-quality, multilingual training data, coupled with the inherent difficulty of aligning visual and textual representations across different languages, has hindered the development of effective multilingual LVLMs. Existing research has explored various strategies to address this issue. MURAL (Cadene et al. 2019) and Pali (Chen et al. 2022) leverage dual encoders for parallel translation prediction and train large-scale multilingual models, respectively. M3P (Yu et al. 2024) and UC2 (Zhou et al. 2021) employ data augmentation techniques, such as translation and data construction, to expand the training data available for low-resource languages. While these methods show promise, they often face challenges in generating high-quality, contextually relevant data, especially for languages with limited resources. Some approaches (Bellagente et al. 2024) further prove the powerful language capabilities of multilingual large language models can be effectively migrated to various downstream tasks without relying on additional multilingual downstream training data. However, these methods may not be sufficient to bridge the modality gap effectively, particularly for lowresource languages, as they often require careful selection of training data and techniques to minimize the discrepancy across the textual and visual modalities.

Our work combines instruction data construction (Zhu et al. 2023a; Chen et al. 2024, 2023) and multilingual improvement using monolingual and bilingual data (Li et al. 2023d; Nguyen et al. 2024). We construct multilingual training data containing multiple monolingual and bilingual task instructions and use a cross-modal regularizer to make the model generate consistent representations for the same input across different modalities. This novel approach, which we term LRM-LLaVA, effectively addresses the key challenges of data scarcity and modality gap.

# Methodology

This section introduces the model architecture of LRMLLaVA and the two-stage pre-training strategy that leverages monolingual and bilingual task instructions on a constructed multilingual dataset to reduce the modality gap.

# Model Architecture

The architecture of LRM-LLaVA, illustrated in Figure 2, is designed to effectively bridge the modality gap between visual and multilingual textual information. It comprises four key components: (1) Visual Encoder: pretrained ViT-L/14 (Radford et al. 2021) is employed to extract rich visual features from input images. (2) Multilingual Large Language Model: Vicuna-13B (Chiang et al. 2023) fine-tuned on LLaMA (Touvron et al. 2023), is responsible for processing textual information and generating coherent and relevant responses. (3) Vision-Text Representation Projector: A two-layer multi-layer perception (MLP) projector serves as a crucial bridge to align visual features and multilingual features. (4) Cross-Modal Regularizer: A cross-modal regularizer is incorporated into the architecture to further mitigate the modality gap between the multilingual large language model and the visual encoder. This regularizer operates by leveraging multiple task instructions to encourage the model to learn consistent representations for the same input across different modalities.

![](images/b854e1cceb3a494e0e49a2e53c7e7f71a425fb7cc9f057b49b83d334556bb348.jpg)  
Figure 2: The two-stage training and model architecture of the LRM-LLaVA with the visual encoder, large language model, vision-text representation projector and cross-modal regularizer.

# Two-Stage Training

To effectively train LRM-LLaVA and achieve robust crossmodal alignment, we employ a two-stage training strategy, inspired by recent advancements (Liu et al. 2024b).

For pre-training, the emphasis is on aligning the visual features and the multilingual features to establish a shared semantic space. We freeze the parameters of both the visual encoder and the large language model, and use a multilingual task instruction to train the parameters of the projector. The training data consists of images, multilingual questions, and image descriptions. This approach can achieve low-cost alignment of visual and multilingual features. Specifically, for an image $X _ { v }$ and a text sequence $X _ { q }$ of length $L$ , we compute the probability of the target answers $X _ { a } ^ { p }$ by:

$$
p ( X _ { a } ^ { p } \mid X _ { v } , X _ { q } ) = \prod _ { i = 1 } ^ { L } p _ { \theta 1 } ( x _ { i } \mid X _ { v } , X _ { q , < i } , X _ { a , < i } ^ { p } )
$$

where $\theta 1$ are the trainable parameters in the pre-training stage. $X _ { q , < i } , X _ { a , < i } ^ { p }$ are the input sequence and answer tokens before the current prediction token $x _ { i }$ .

For instruction fine-tuning, the emphasis is on improving the instruction following capabilities. We only freeze the parameters of the visual encoder and train the large language model and the projector with three multilingual task instructions based on various types of training data to improve the instruction following ability. This diverse training data exposes the model to a wide range of instructions and prompts, enhancing its ability to generalize to unseen scenarios. Specifically, for an image $X _ { v }$ and a text sequence $X _ { q }$ of length $L$ , we compute the probability of the target answers $X _ { a } ^ { f }$ by:

$$
p ( X _ { a } ^ { f } \mid X _ { v } , X _ { q } ) = \prod _ { i = 1 } ^ { L } p _ { \theta 2 } ( x _ { i } \mid X _ { v } , X _ { q , < i } , X _ { a , < i } ^ { f } )
$$

where $\theta 2$ are the trainable parameters in the fine-tuning stage. Xq,<i and Xaf, <i are the input sequence and answer tokens before the current prediction token $x _ { i }$ .

# Vision-Text Representation Projector

The visual-text representation projector plays a critical role by bridging the modality gap between visual and multilingual information. This projector, implemented as a twolayer multi-layer perceptron (MLP), converts the visual features extracted by the visual encoder into semantic features of the multilingual language model. Its concise architecture allows low-cost training and provides an efficient and effective mechanism for aligning the visual features and the multilingual features. Specifically, given an image $X _ { v }$ , the projector first extracts visual features using the visual encoder, denoted as $Z _ { v }$ :

$$
Z _ { v } = g _ { \phi } ( X _ { v } )
$$

where $g _ { \phi }$ represents the visual encoder with parameters $\phi$ . These visual features are then passed through the two-layer MLP projector, parameterized by weight matrices $W _ { 1 }$ and $W _ { 2 }$ and bias vectors $b _ { 1 }$ and $b _ { 2 }$ , to produce the aligned multilingual features $M _ { v }$ :

$$
H _ { v } = \sigma ( W _ { 1 } \cdot Z _ { v } + b _ { 1 } )
$$

$$
M _ { v } = W _ { 2 } \cdot H _ { v } + b _ { 2 }
$$

where $\sigma$ represents a non-linear activation function. During the two-stage training process, the projector’s parameters $( W _ { 1 } , W _ { 2 } , b _ { 1 } , b _ { 2 } )$ are continuously updated to optimize the alignment between visual and multilingual features.

# Cross-Modal Regularizer

The core of successful LRM-LLaVA training lies in effectively regularizing the interaction between visual encoder and multilingual language model. The regularizer is crucial to minimizing the modality gap between visual and multilingual textual information. In LRM-LLaVA, we achieve this through carefully designed three task instructions including monolingual vision question answering (MVQA), non-English vision question answering with English as the pivot (EP-VQA), and bilingual vision question answering (BVQA). Among them, only MVQA is used in the pretraining stage, while all instructions are used in the finetuning stage. These instructions, individually and collectively, contribute to enhancing the multilingual capabilities by promoting robust cross-modal alignment.

![](images/f4092034b90a19a7b4a56be90ee6f2f2a050bc1149df2de65b9ae5780ef97690.jpg)  
Figure 3: Illustration of how the multilingual task instructions enhance the alignment of image, non-English human question and assistant answer. (a) (b), (c), and (d) respectively represent the alignment for ordinary LVLMs, MVQA instruction, EP-VQA instruction, and BVQA instruction. Among them, black arrows indicate strong alignment, while gray arrows indicate weak alignment. It can be observed that these three multilingual training tasks enhance alignment ability through different paths.

MVQA: MVQA task instruction focuses on solidifying the ability to understand and respond to visual questions within individual languages. The model is trained on image-question-answer triplets in the same language, learning to map visual information to language-specific questionanswer pairs. This process strengthens the connection between visual information and its corresponding linguistic expressions within each language. For an image $X _ { v }$ , the user question $X _ { \cdot } ^ { l a }$ , and assistant answer $X _ { a } ^ { l a }$ in language $l a$ , the loss function for MVQA can be expressed as:

$$
\mathcal { L } _ { M V Q A } = - \sum _ { i = 1 } ^ { | X _ { a } ^ { l a } | } p ( x _ { a _ { i } } | X _ { v } , X _ { q } ^ { l a } ; \theta ) \log q ( x _ { a _ { i } } )
$$

where $p ( x _ { a _ { i } } | X _ { v } , X _ { q } ^ { l a } ; \theta )$ represents the one-hot encoding of the ground truth answer token, $q ( x _ { a _ { i } } )$ is the predicted probability distribution over the answer vocabulary for token $x _ { a _ { i } }$ , and $\theta$ represents the model parameters.

EP-VQA: EP-VQA task instruction leverages better English proficiency to promote cross-lingual understanding. By introducing English as a pivot language, we bridge the gap between non-English languages and English. This instruction is also part of the text input, which first asks the model to translate non-English questions into English questions, and then combines the non-English and translated questions with visual input to generate non-English answers. This multistep process prompts the model in a step-by-step guided manner, improving translation skills and cross-lingual reasoning because the model must synthesize information from both languages in the context of the visual scene. For an image $X _ { v }$ , the user question $X _ { q } ^ { l a }$ in non-English language $l a$ the EP-VQA instruction $X _ { e }$ , and assistant answer $X _ { a }$ , the loss function for MVQA can be expressed as:

$$
\mathcal { L } _ { E P - V Q A } = - \sum _ { i = 1 } ^ { | X _ { a } | } p ( x _ { a _ { i } } | X _ { v } , X _ { q } ^ { l a } , X _ { e } ; \theta ) \log q ( x _ { a _ { i } } )
$$

where $X _ { a }$ includes non-English answer and English translation of non-English question.

BVQA: BVQA task instruction causes the model to have a bilingual question and answer conversation based on visual information. This instruction requires the model to answer questions in one language with answers in another language, and ensures that one language is English. This forces the model to learn direct mappings between visual representations and linguistic expressions in two languages simultaneously. The objective function of BVQA is similar to MVQA, but operates on cross-lingual question-answer pairs, ensuring that the model learns accurate translation while maintaining visual context. For an image $X _ { v }$ , the user question $X _ { a } ^ { l a 1 }$ in language la1, and assistant answer $X _ { a } ^ { l a 2 }$ in language $l a \overset { \triangledown } { 2 }$ , the loss function for MVQA can be expressed as:

$$
\mathcal { L } _ { B V Q A } = - \sum _ { i = 1 } ^ { | X _ { a } ^ { l a 2 } | } p ( x _ { a _ { i } } | X _ { v } , X _ { q } ^ { l a 1 } ; \theta ) \log q ( x _ { a _ { i } } )
$$

LVLMs use the ability of the large language model to align images, questions, and answers in various languages. However, since the training dataset is mainly in English, the model forms a strong alignment relationship between images, English questions, and English answers, while it forms a weak alignment relationship for images, questions, and answers in other non-English languages. Figure 3 shows how MVQA, EP-VQA, and BVQA improve the alignment relationship between images and non-English through different strategic paths, either directly or indirectly with English as the pivot. Each task instruction contributes a unique facet to the overall learning process, encouraging the model to develop a rich understanding of cross-lingual relationships, grounded in visual information. This multi task instruction strategy, coupled with LRM-LLaVA’s carefully designed architecture, significantly enhances its multilingual ability.

# Experiments

# Datasets and Benchmarks

Due to the lack of multilingual LVLMs datasets, the twostage training data and benchmarks are constructed from open-source English datasets. For training data, we introduce more than 120M English pre-trained image-text alignment dataset ShareGPT4V-PT (Chen et al. 2023) in pretraining and more than 120M English instruction fine-tuning datasets including LLaVA-1.5 mixtures (Liu et al. 2024a), ChartQA (Masry et al. 2022), GeoQA (Chen et al. 2021), DocVQA (Mathew, Karatzas, and Jawahar 2021), DVQA (Kafle et al. 2018), and AI2D (Kembhavi et al. 2016) in finetuning to ensure diversity and comprehensiveness. Furthermore, based on these English training data, we construct the final multilingual training data containing three task instructions by translating from English to non-English languages. For benchmarks, we select four English benchmarks including MME (Fu et al. 2023), MMBench (Liu et al. 2023), POPE (Li et al. 2023c), and SEED-Bench (Li et al. 2023a). Their questions and answers about the images are relatively short, which effectively reduces the information errors or distortions caused by translation.

<html><body><table><tr><td rowspan="2">Model</td><td colspan="10">Languages</td></tr><tr><td>En</td><td>Cs</td><td>De</td><td>Fr</td><td>Hr</td><td>It</td><td>Ro</td><td>Ru</td><td>Sp</td><td>Zh</td></tr><tr><td colspan="10">Multilingual MME (Fu et al. 2023)</td></tr><tr><td>InstructBLIP</td><td>1237.5</td><td>879.8</td><td>1034.0</td><td>1107.1</td><td>933.0</td><td>1095.2</td><td>1092.2</td><td>977.1</td><td>1155.3</td><td>975.3</td></tr><tr><td>LLaVA-v1.5</td><td>1529.4</td><td>1074.7</td><td>1211.2</td><td>1235.0</td><td>945.1</td><td>1072.5</td><td>1210.4</td><td>902.9</td><td>999.1</td><td>1230.1</td></tr><tr><td>LVIS-4V</td><td>1574.9</td><td>1157.8</td><td>1263.2</td><td>1336.5</td><td>805.8</td><td>1238.2</td><td>1288.6</td><td>826.1</td><td>1145.5</td><td>1209.3</td></tr><tr><td>ShareGPT4V</td><td>1599.9</td><td>1272.9</td><td>1401.0</td><td>1365.6</td><td>1035.6</td><td>1240.6</td><td>1296.4</td><td>1246.0</td><td>1178.7</td><td>1329.0</td></tr><tr><td>Qwen-VL-Chat</td><td>1549.0</td><td>784.1</td><td>1107.0</td><td>1427.7</td><td>1037.3</td><td>1317.7</td><td>1235.2</td><td>1239.9</td><td>1418.5</td><td>1209.9</td></tr><tr><td>LRM-LLaVA</td><td>1534.3</td><td>1338.9</td><td>1386.7</td><td>1437.4</td><td>1422.3</td><td>1406.3</td><td>1441.5</td><td>1305.6</td><td>1454.2</td><td>1358.3</td></tr><tr><td colspan="10">Multilingual POPE (Li et al. 2023c)</td></tr><tr><td>InstructBLIP</td><td>84.8</td><td>67.2</td><td>81.8</td><td>81.5</td><td>40.6</td><td>81.1</td><td>78.1</td><td>79.4</td><td>81.2</td><td>84.3</td></tr><tr><td>LLaVA-v1.5</td><td>86.0</td><td>76.1</td><td>83.9</td><td>77.9</td><td>44.8</td><td>76.2</td><td>77.5</td><td></td><td>75.1</td><td>83.7</td></tr><tr><td>LVIS-4V</td><td>85.6</td><td>74.2</td><td>84.4</td><td>80.6</td><td>40.6</td><td>71.2</td><td>78.0</td><td>83.6 83.6</td><td>72.9</td><td>84.3</td></tr><tr><td>ShareGPT4V</td><td>86.4</td><td>64.6</td><td>81.4</td><td>77.4</td><td>43.1</td><td>69.2</td><td>79.1</td><td>84.4</td><td>71.9</td><td>85.0</td></tr><tr><td>LRM-LLaVA</td><td>86.6</td><td>86.8</td><td>87.0</td><td>86.9</td><td>86.0</td><td>87.4</td><td>83.3</td><td>87.4</td><td>84.6</td><td>85.1</td></tr><tr><td colspan="10">Multilingual SEED-Bench (Image) (Li et al. 2023a)</td></tr><tr><td>LLaVA-v1.5</td><td>68.2</td><td>63.6</td><td>66.8</td><td>67.1</td><td>62.5</td><td></td><td></td><td></td><td></td><td>66.6</td></tr><tr><td>LVIS-4V</td><td>69.0</td><td>64.3</td><td>67.4</td><td>67.5</td><td></td><td>66.6</td><td>64.1</td><td>66.7</td><td>67.0 67.9</td><td>67.0</td></tr><tr><td>ShareGPT4V</td><td>70.8</td><td>66.1</td><td>69.6</td><td></td><td>63.0</td><td>67.2</td><td>64.2</td><td>67.4</td><td></td><td></td></tr><tr><td>LRM-LLaVA</td><td>70.0</td><td>68.7</td><td>69.1</td><td>68.8 69.1</td><td>65.0 68.6</td><td>68.4 69.1</td><td>66.4 68.9</td><td>69.2 69.4</td><td>70.3 69.1</td><td>68.1 68.5</td></tr><tr><td colspan="10">Multilingual MMBench (Liu et al. 2023)</td></tr><tr><td>LLaVA-v1.5</td><td>69.5</td><td>59.8</td><td>62.5</td><td>62.1</td><td>58.2</td><td>59.9</td><td>62.4</td><td>59.6</td><td>61.9</td><td>65.3</td></tr><tr><td>LVIS-Instruct4V</td><td>68.4</td><td>60.3</td><td>62.8</td><td>64.8</td><td>57.5</td><td>62.8</td><td>62.1</td><td>59.9</td><td>62.4</td><td>64.7</td></tr><tr><td>ShareGPT4V</td><td>69.8</td><td>61.0</td><td>64.6</td><td>63.3</td><td>59.1</td><td>63.1</td><td>62.7</td><td>60.4</td><td>62.6</td><td>65.1</td></tr><tr><td>LRM-LLaVA</td><td>70.1</td><td>66.1</td><td>66.3</td><td>66.1</td><td>65.1</td><td>65.8</td><td>65.5</td><td>66.1</td><td>66.4</td><td>67.2</td></tr></table></body></html>

Table 1: Evaluation results on our proposed 4 multilingual benchmarks among similar parameters LVLMs. Model names are InstructBLIP (Dai et al. 2023), LLaVA-1.5 (Liu et al. 2024a), LVIS-Instruct4V (Wang et al. 2023), ShareGPT4V (Chen et al. 2023), and Qwen-VL (Bai et al. 2023b). Bold numbers are the best results.

Table 2: Average BLEU scores of Google Translate and GPT-4o on non-English languages for different benchmarks.   

<html><body><table><tr><td>Benchmarks Avg BLEU</td><td>MME 57.34</td><td>MMB 54.23</td><td>SEED 58.55</td><td>POPE 56.19</td></tr></table></body></html>

We select Google Translate, which has the best performance in BLEU scores (Goyal et al. 2022) based on the analysis in Zhu et al. (2023c). In particular, we train directly on English data for tasks such as mathematical computations due to the unreliability of translating these tasks. For the training languages, we consider the diversity of language families and language sizes under limited resources, and finally select 10 languages, including English (En), Chinese (Zh), Italian (It), French $( \mathrm { F r } )$ , German (De), Spanish (Sp), Russian $\mathbf { \left( R u \right) }$ , Romanian (Ro), Czech $\mathrm { ( C s ) }$ , and Croatian $( \mathrm { H r } )$ . We sample data from these languages and integrate them into the final training data. The final data contains more than 240M pre-training data and more than 240M instruction fine-tuning data, of which English accounts for $50 \%$ and the 9 non-English languages each account for $5 . 6 \%$ .

# Implementation Details

LRM-LLaVA follows a two-stage training process on the 13B-parameter large language model Vicuna-13B (Chiang et al. 2023) and the 0.6B-parameter visual encoder ViT-L/14 (Radford et al. 2021), and it takes 144 hours to complete all the training on $8 ^ { * } \mathrm { A } 8 0 0$ GPUs. We pre-train LRM-LLaVA on the proposed 240M multilingual pre-training image-text pairs for 1 epoch with a learning rate of 1e-3 and a batch size of 24 per GPU. Also, we fine-tune LRM-LLaVA on the proposed 240M multilingual instruction fine-tuning image-text pairs for 1 epoch with a learning rate of 2e-5 and a batch size of 32 per GPU. In both stages of training, we use a max context length of 2048 and adopt the same response formatting prompts as LLaVA-1.5.

<html><body><table><tr><td>User: PopiSte prosim obsah tohoto obrazku v Ceském jazyce. (English translation: Please describe the content of this image in Czech.)</td><td>LRM-LLaVA (English translation of Czech answer) : There are two elephants in the picture. The first elephant, which occupies the upper right, is only partially visible, mainly its rear,and appears to be standingoff-camera. The second elephant, which is the main focus of the image,is positioned in the middle and appears to be standing up or moving in shallow water.</td><td>LLaVA-1.5 (English translation of Czech answer) : In the picture there is a large elephant that is in the mud-eating water. The elephant has its trunk extended above the water and its head under water. There are also several eggs around the elephant that are scattered in the mud eater water. The image is created digitally, which is evident in its quality and detail.</td></tr></table></body></html>

Figure 4: An example of LRM-LLaVA and LLaVA-1.5 using Czech questions and answers on the same image. Due to readability and limited space, we only show the English translations of the Czech answers. Compared with LLaVA-1.5, LRM-LLaVA’s description of the image is more accurate, including the number, location and behavior of the elephants.

Table 3: Analysis results of our proposed method on multilingual MMBench. Avg represents the average score of 9 non-English languages. Bold numbers are the best results.   

<html><body><table><tr><td>Method</td><td colspan="2">Languages</td></tr><tr><td>w/o All Task Instructions</td><td>En 70.2</td><td>Avg</td></tr><tr><td>W/o MVQA Instruction (Pre-training)</td><td>69.8</td><td>61.6</td></tr><tr><td>w/o MVQA Instruction (Fine-tuning)</td><td>69.6</td><td>65.0</td></tr><tr><td>w/oEP-VQAInstruction</td><td>70.0</td><td>63.8</td></tr><tr><td>W/o BVQA Instruction</td><td></td><td>63.9</td></tr><tr><td></td><td>69.6</td><td>65.2</td></tr><tr><td>LRM-LLaVA</td><td>70.1</td><td>66.1</td></tr></table></body></html>

# Main Results

We compare the performance of multiple LVLMs with 7B parameters and 13B parameters on our proposed four multilingual benchmarks and present the results in Table 1. Compared with previous LVLMs, LRM-LLaVA outperforms competitors especially in non English languages.

Notably, previous LVLMs are far less capable of understanding non-English, especially in low-resource languages such as Croatian and Czech. LRM-LLaVA uses the same large language model as some of the LVLMs such as LLaVA-1.5 (Liu et al. 2024a) and LVIS-Instruct4V (Wang et al. 2023). However, with our proposed multilingual training framework, the performance of LRM-LLaVA in nonEnglish languages significantly improves and is close to that in English. In particular, although the training data of LRMLLaVA includes non-English languages, LRM-LLaVA performs well on English and even achieves the best results on multilingual MMBench and multilingual POPE. For qualitative analysis, Figure 4 shows an example of LRM-LLaVA and LLaVA-1.5 using Czech questions and answers on the same image. It can be seen that LRM-LLaVA’s description of the image is more detailed and accurate. The comprehensive results indicate that our proposed method can significantly improve the multilingual ability of LVLMs.

![](images/77a7ecc0153a92804fef1153de8c9d3cd6347e1e1f7b957ba89c223d1088777e.jpg)  
Figure 5: The red line represents the distribution of visual features through projector, while the blue line represents the distribution of multilingual features. (a) corresponds to LRM-LLaVA, (b) corresponds to LLaVA.

# Analysis and Discussion

# Analysis on Benchmark Translation Results

To verify the reliability of the translated multilingual benchmarks, we use Google Translate and GPT-4o (Achiam et al. 2023) to translate the English benchmarks including MMBench, MME, POPE, and SEED-Bench into 9 non-English languages. Based on the two translation results, we calculate the average BLEU scores of the 9 non-English languages to quantify the similarity of the two translation results. Table 2 shows the specific results. The experimental results show that Google Translate and GPT-4o demonstrate good synergy in the translation results of the four benchmarks, and their BLEU scores have good performance. This is because the questions and answers of the multiple-choice questions and true-or-false questions in the four benchmarks are relatively short, making the translation results more reliable.

# Ablation Study on Task Instructions

To better understand the contribution of the three multilingual task instructions, we conduct an ablation study on LRM-LLaVA as shown in Table 3.

Firstly, removing all task instructions reduces the average score of non-English languages by 4.5, but the English score remains the same as LRM-LLaVA. This shows our proposed method can significantly improve multilingual ability without reducing English ability. Secondly, removing MVQA in the pre-training stage, and removing BVQA in the fine-tuning stage will reduce 1.1 and 0.9 points respectively, while removing MVQA and EP-VQA in the finetuning stage will reduce 2.3 and 2.2 points respectively. This is because MVQA and EP-VQA in the fine-tuning stage change the parameters of the large language model and their alignment paths are more brief. Nevertheless, MVQA in the pre-training stage and BVQA in the fine-tuning stage are also necessary, and play roles in different alignment paths.

Table 4: Multilingual performance of different LLMs on multilingual MMBench with and without our approach. Avg represents the average score of 9 non-English languages.   

<html><body><table><tr><td>Language Model</td><td>Method</td><td>Languages En</td><td>Avg</td></tr><tr><td>Vicuna-13B</td><td>w/o All Instructions W/ All Instructions</td><td>70.2 70.1</td><td>61.6 66.1</td></tr><tr><td>Vicuna-7B</td><td>w/o All Instructions w/All Instructions</td><td>69.4 69.2</td><td>60.9 64.7</td></tr><tr><td>LLaMA2-13B</td><td>w/oAllInstructions w/ All Instructions</td><td>65.2 65.5</td><td>58.7 62.1</td></tr><tr><td>LLaMA3-8B</td><td>w/oAllInstructions w/All Instructions</td><td>69.7 69.4</td><td>62.3 66.0</td></tr></table></body></html>

# Analysis on Vision-Text Modality Alignment

As shown in Figure 5, we visualize the distribution of the multilingual features and visual features of LRM-LLaVA and LLaVA to examine the aligning effectiveness of our method. Specifically, we average the sequential representations of the image features through the projector and the multilingual features in 5,000 data points, and then apply the T-SNE dimensionality reduction algorithm to reduce the hidden dimensions to 2 dimensions. We plot the kernel density estimation distribution based on the reduced 2- dimensional representations of the two features. Our method significantly enhances the modality alignment of multilingual and visual features in the semantic space.

# Analysis on the Large Language Models

Table 4 shows the effect of our proposed multilingual training framework on different large language models. Firstly, to observe the impact of parameter scale, we use the same series of large language models, Vicuna-7B and Vicuna-13B, to train LVLMs with English training data and multilingual training data in the same process. With our method, the 13Bparameter model achieves higher multilingual performance than the 7B model. Moreover, compared with the model trained with English data, the improvement in multilingual performance of the 13B-parameter model is also higher than that of the 7B-parameter model. This shows that larger-scale models have better multilingual understanding, can more easily achieve alignment between multilingual question answering and visual modalities, and achieve better returns.

Secondly, to verify our method does not depend on a specific large language model, we select LLaMA2-13B (Touvron et al. 2023) and LLaMA3-8B (Dubey et al. 2024) as language models to train multilingual LVLMs. Experimental results show that our method achieves improvements on different large language models and exhibits the same trend.

![](images/b8d2665b456b7086f614c897c1be2cc6f60deb6b7f538609aafdb4e2927077d0.jpg)  
Figure 6: We mix $40 \%$ , $70 \%$ , $100 \%$ , and $1 5 0 \%$ of nonEnglish data into the English data respectively for training, and score them in 10 languages on multilingual MMBench.

# Analysis on the Size of Synthesized Data

To explore the reasonable size of synthetic non-English data, we add different proportions of non-English data to the English fine-tuning data and evaluate their scores in 10 languages on multilingual MMbench. As shown in Figure 6, we mix $40 \%$ , $70 \%$ , $100 \%$ , and $1 5 0 \%$ of non-English data respectively. Experimental results show that mixing $100 \%$ non-English data significantly improves the multilingual performance compared with $40 \%$ and $70 \%$ mixtures. However, when the proportion increases to $1 5 0 \%$ , the nonEnglish performance does not improve significantly, but the English performance decrease significantly. For the final training data, we choose to use a $100 \%$ non-English data integration ratio in both pre-training and instruction finetuning stages to improve multilingual capabilities as much as possible without affecting English capabilities.

# Conclusion

This paper proposes a multilingual LVLMs training framework for low-resource languages. Based on this, we train LRM-LLaVA on 10 languages, which mainly solves the modality gap between visual features and multilingual features. The multilingual training framework consists of four components, including a visual encoder, a backbone multilingual large language model, a vision-text representation projector, and a cross-modal regularizer. LRM-LLaVA uses a two-stage training approach for model training and visiontext alignment. Due to the lack of non-English data, we translate English training data into non-English data and construct three monolingual or bilingual task instructions. To evaluate the multilingual capabilities of LVLMs, we translate four English benchmarks to obtain multilingual benchmarks and verify their reliability. Experimental results show that LRM-LLaVA achieves competitive multilingual performance compared to other multilingual LVLMs of similar parameters.