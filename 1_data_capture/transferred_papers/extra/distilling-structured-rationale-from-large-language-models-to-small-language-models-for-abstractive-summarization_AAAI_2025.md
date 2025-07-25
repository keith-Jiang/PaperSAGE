# Distilling Structured Rationale from Large Language Models to Small Language Models for Abstractive Summarization

Linyong Wang1,2, Lianwei $\mathbf { W _ { u } } ^ { 1 , 2 * }$ , Shaoqi $\mathbf { S o n g ^ { 1 } }$ , Yaxiong Wang3, Cuiyun Gao4, Kang Wang1

1ASGO, School of Computer Science, Northwestern Polytechnical University, Xi’an, China 2Research & Development Institute of Northwestern Polytechnical University in Shenzhen, China 3School of Computer Science and Information Engineering, Hefei University of Technology, Hefei, China. 4Harbin Institute of Technology, Shenzhen, China linyongwang $@$ mail.nwpu.edu.cn, wlw $@$ nwpu.edu.cn, songshaoqi $@$ mail.nwpu.edu.cn, wangyx15 $@$ stu.xjtu.edu.cn, gaocuiyun $@$ hit.edu.cn, $\mathrm { w k 0 } \textcircled { < } \textcircled { \omega }$ mail.nwpu.edu.cn

# Abstract

Large Language Models (LLMs) have permeated various Natural Language Processing (NLP) tasks. For the summarization tasks, LLMs can generate well-structured rationales, which consist of Essential Aspects (EA), Associated Sentences (AS) and Triple Entity Relations (TER). These rationales guide smaller models $( \leq 1 \mathrm { B } )$ to produce better summaries. However, their high deployment costs $( \geq 7 0 \mathbf { B } )$ ), such as substantial storage space and high computing requirements, limit their utilization in resource-constrained environments. Furthermore, effectively distilling these structured rationales from LLMs into Small Language Models (SLMs) models remains a challenge. To address this, we propose the LLM-based Structured Rationale-guided Multi-view Weak-gated Fusion framework (LSR-MWF). The framework initially employs LLMs to dig structural rationales from a document, considering multiple viewpoints such as EA, AS, and TER. Then, it develop a multistep summary generation evaluation strategy to select highquality structured rationales. Subsequently, it aligns with these rationales using additional modules organized in a hierarchical structure. Finally, the framework integrates the features output by these modules with original abstractive model through a weak-gated mechanism. Experimental results on two publicly available CNN/DailyMail and XSum datasets show that our method improves the performance of the abstractive model, outperforming baselines by $1 1 . 2 \%$ and $5 . 8 \%$ , respectively. In addition, our method improves the interpretability of summary generation from the viewpoints of EA, AS and TER.

# Code — https://github.com/Wangdoudou8/LSR-MWF

# Introduction

Large language models (LLMs), such as GPT-series (Ouyang et al. 2022; Achiam et al. 2023), Llama (Touvron et al. 2023), PaLM (Chowdhery et al. 2023), and Chinchilla (Hoffmann et al. 2022), have permeated every aspect of natural language processing (NLP), encompassing tasks like questionanswering (QA) systems (Longpre et al. 2023), etc. For summarization tasks, LLMs can generate high-quality structured rationales from a document, which consist of Essential Aspects (EA), Associated Sentences (AS), and Triple Entity

![](images/0bfd33af15753d816a58811177bf7ddd754c92956b7a911e74f1d8a160eacedf.jpg)  
Figure 1: A simple demonstration of structured rationales distillation of documents via LLMs, including Essential Aspects Extraction (EAE), Associated Sentences Extraction (ASE) and Triples Entity Relations Generation (TERG).

Relations (TER), as shown in Figure 1. These rationales function similarly to Chain-of-Thought (CoT) (Wang, Zhang, and Wang 2023; Wen et al. 2023b,a) in guiding Small Language Models (SLMs) to produce better summaries. However, deploying LLMs $( \geq 7 0 \mathbf { B } )$ requires substantial computational resources, limiting their utility in resource-constrained environments $( \leq 1 \mathbf { B } )$ (Strubell, Ganesh, and McCallum 2020). Furthermore, effectively distilling these structured rationales from LLMs to SLMs remains a significant challenge due to the lack of careful rationale extraction, rationale selection, or training strategies (Pham et al. 2023; Jiang et al. 2024; Liu et al. 2024a).

Abstractive summarization is a pivotal task that condenses lengthy texts into concise, informative summaries (Radev, Hovy, and McKeown 2002). BART (Lewis et al. 2019) and PEGASUS (Zhang et al. 2020) are two prominent pre-trained Seq2Seq language models widely used in academic research. GSum (Dou et al. 2020), built upon BART, further enhances its performance by incorporating guidance from an extractive summarizer. Recently, SeqCo (Xu et al. 2022) emerged as a method that harnesses contrastive learning to bolster the effectiveness of BART in abstractive summarization. However, existing methods for summarization largely focus on the overall content of the source document, neglecting to explore its hierarchical structured information. This often results in coarse-grained, poor-quality summaries (Gekhman et al. 2023; Liu et al. 2023). Recognizing the hierarchical nature of source documents, recent research has indicated the potential of LLMs in generating structured rationales by extracting core themes from documents (Min et al. 2023;

Jiang et al. 2024).

To impart the structured abstractive summarization capabilities of LLMs to SLMs, with the aim of enhancing both performance and interpretability, we introduce a novel framework, LLM-based Structured Rationale-guided Multi-view Weak-gated Fusion framework (LSR-MWF), which not only makes the process of generating summaries clearer but also provides deeper insights into the content. Our work encompasses three main components: 1) Exploring structured rationales of documents by LLMs, 2) Selecting the best rationales, and 3) Training the local small model. First, we design three sub-tasks for LLMs to “dig” EA, AS and TER (three types of gems) of a document based on the LLM-based Structured Rationale-guided sub-framework (LSR). Simultaneously, these “gems” compose the structured rationales, offering insights into the document from three distinct viewpoints. Secondly, To ensure the overall quality of structured rationales, we adopt a multi-step summary generation evaluation strategy to select high-quality structured rationales for subsequent local small model training. Finally, we train our local small model using the Multi-view Weak-gated Fusion sub-framework (MWF). Specifically, we add three hierarchically structured modules to gradually align features of EA, AS, and TER. These features are then controlled through the weak-gated mechanism to fuse the outputs of the abstractive model.

The main contributions of this paper are as follows:

• Effective Generation of Structured Rationales in LLMs: We propose an innovative method, LLM-based Structured Rationale-guided sub-framework (LSR), which harnesses the extraction and generation capabilities of LLMs to progressively extract or generate EA, AS, and TER of documents.   
• Effective Knowledge Distillation of Structured Rationales to SLMs: We ensure that SLMs can fully absorb and utilize these structured rationales. This is achieved by adopting the Multi-view Weak-gated Fusion subframework (MWF), significantly enhancing the performance of SLMs in summarization tasks.   
• Experimental Validation of Method Superiority and Interpretability: Experimental results demonstrate that our proposed method significantly outperforms baselines on two public standard datasets, CNN/DailyMail and XSum. Finally, the entire framework, LSR-MWF, is innovative as it bridges the gap between LLMs and SLMs, enabling SLMs to inherit the structured abstractive summarization capabilities of LLMs while maintaining high performance and interpretability by showing specific cases and visualizations.

# Related Work

# LLMs for Abstractive Summarization

With technological advancements, numerous LLMs capable of performing summarization tasks have emerged, such as ChatGPT, GPT-4 (Achiam et al. 2023), and PaLM (Anil et al. 2023). These models, trained on vast amounts of text corpus with billions of parameters, exhibit exceptional performance in abstractive summarization tasks. Notably, their performance can be further enhanced when guided through step-by-step reasoning (Wei et al. 2022; Liu et al. 2024b). However, despite their impressive capabilities, the substantial resource requirements of these LLMs pose a challenge to their widespread adoption. Additionally, when utilizing LLM-as-a-service APIs, data privacy concerns cannot be overlooked, particularly when handling sensitive information. This underscores the importance of running SLMs locally. To leverage the powerful reasoning capabilities of LLMs in abstractive summarization, Wang et al. (2021) ingeniously utilized these models to enhance the quality of tags generated for headlines, while Jiang et al. (2024) harnessed aspect-triple rationales generated by LLMs to improve the summary quality of SLMs. Despite these advancements, existing methods still fail to fully transfer the comprehensive extraction and generation capabilities of LLMs to SLMs.

# Knowledge Distillation and Interpretability in Abstractive Summarization

Knowledge distillation techniques (Hinton, Vinyals, and Dean 2015; Kim and Rush 2016; Guo et al. 2023) aim to extract specialized knowledge from larger models to tailor smaller models for specific tasks. This technology has found wide application in various domains (Shleifer and Rush 2020; Avram et al. 2021; Zhou, Xu, and McAuley 2021; Jiao et al. 2019; Jia et al. 2024). For abstractive summarization tasks, Jia et al. (2020) and Liu, Yang, and Chen (2024) focused on extractive and abstractive summarization, respectively, both leveraging knowledge distillation techniques to enhance summary generation quality. However, their approaches lack effective visualization, leaving interpretability as an area for further improvement. As the complexity of deep neural networks increases, the interpretability of models becomes increasingly crucial. To enhance model interpretability, researchers have begun exploring rationales generation techniques (Ho, Schmid, and Yun 2022; Hsieh et al. 2023; Li and Chaturvedi 2024; Wu et al. 2023b,a; Jiang et al. 2024). For abstractive summarization, the creation of rationales not only enhances model interpretability but also provides insights for keypoint generation, contributing to the production of higher-quality and more structured summaries. Recent work has leveraged the structured rationales generated by LLMs to enhance the performance and transparency of smaller summarization models. For instance, Jiang et al. (2024) utilized rationales and summaries generated by LLMs to train a smaller model through a multi-round curriculum learning approach. Similarly, other researchers, such as Ho, Schmid, and Yun (2022) employed the reasoning samples produced by LLMs to fine-tune smaller models, achieving substantial improvements in reasoning capabilities and transparency across various tasks. Even with these advancements, the comprehensive extraction and generation capabilities of LLMs in abstractive summarization remain underexplored. To further explore the limits of LLMs’ reasoning abilities, this paper refines the rationale generation method and introduces the LSR-MWF to more comprehensively absorb and utilize these structured information.

Step 1: LLM-based Structured Rationale Digging Step 3: Multi-view Weak-gated Fusion Small Model Training   
Given a document and its summary, do the following tasks:   
Task 1: Extract one essential aspect for each summary sentence. Abstractive Model   
Task 2: For the essential aspect, extract one associated sentence from the document.   
Task 3: For the associated sentence, retrieve detailed triples of entities and relations Document Encoder Decoder   
in the format [subject entity # relation # object entity, ...]. Prompt Document GrSounmdm-Taryuth Θ 油 ? LLM Cross- 曲 Wake- Wake- Wake  
structured rationales Attn EA Feature gated gated gated Aspects Seentnetnecnecses Relations CrossSelf-Attn Attn AS Feature Step 2: Best Rationale Selection   
Given a document and its rationale for summarization. Self-Attn Cross- 中□   
sTehneternatcieosnfaolre scsoentiailnas (p1e)ctsh;e(3e)stshentriapleas opfecnttsitfioers saundmrmelartyi;o(n2s)fotrhesuamssmoacriya.ted Attn TER Feature   
Task 4: For each "[aspects] | [associated sentences] | [subject entity # relation # object   
entity, ...]", generate a multiple sub-summary.   
Task 5: Generate a totalPsruomparty in one sentence with the multiple subm-sulmmarietso.l A S R AA E Cosine loss EA Feature MoEdAule Document ? LLM S S E Cosine loss A S R AS Module   
AAA SSSS RRR Select-Rouge A S R A S R Best Structured A S R RR E Co sine loss AS Feature 曲 MToEduRle A S R Rationales TER Feature

# Methodology

# Overview

We introduce the LSR-MWF. The overall architecture consists of three main components, as shown in Figure 2. It outlines the entire process, including LLM-based Structured Rationale Digging, Best Rationale Selection, and Multi-view Weak-gated Fusion Small Model Training. This method transfers document summarization capabilities from LLMs $\scriptstyle ( \geq 7 0 \mathbf { B } )$ to local SLMs $( \leq 1 \mathrm { B } )$ . We elaborate on each process of the overall method as follows.

# Step 1: LLM-based Structured Rationale Digging

Based on LLM-based Structured Rationale-guided subframework (LSR), we leverage the powerful extraction and generation capabilities of LLMs, along with tailored prompt templates, to conduct an extensive exploration of the latent reasoning process, termed “structured rationale”, between the source document and its corresponding ground-truth summary. This exploration progresses through the hierarchical digging of three distinct viewpoints: Essential Aspects (EA), Associated Sentences (AS), and Triple Entity Relations (TER), which are guided by the requirements of the subsequent three tasks. To facilitate further detailed discussion, we outline some important concepts below.

Gem 1: Essential Aspect (EA) Defined as the topic words $a _ { 1 \sim n }$ extracted from the source document that correspond to each summary sentence.

Gem 2: Associated Sentence (AS) Defined as the most relevant sentences $s _ { 1 \sim n }$ in the document corresponding to the EA.

Gem 3: Triples Entity Relation (TER) Defined as the structured triple entity relations r1∗ m $r _ { 1 \sim m } ^ { \ast } = \langle s | r | o \rangle _ { 1 \sim m } , ( m \geq$ $n$ ) extracted from the AS, including the subjects $s _ { 1 \sim m }$ , relations $r _ { 1 \sim m }$ and objects $o _ { 1 \sim m }$ .

Task 1: EA Extraction This task is defined as extracting the EA in the document $D$ and its corresponding ground-truth summary $S ^ { * }$ , to ultimately obtain a set $A =$ $\mathbf { \bar { \{ } }  a _ { 1 } , a _ { 2 } , \ldots , a _ { n } \}$ . Assuming $\lvert S ^ { * } \rvert$ to represent the number of sentences in $S ^ { * }$ , the formula is given by:

$$
a _ { i } \sim P \left( A | D , S ^ { * } \right) = \prod _ { i = 1 } ^ { | S ^ { * } | } p \left( a _ { i } | D , s _ { i } ^ { * } , a ^ { < i } \right)
$$

Task 2: AS Extraction Given the extracted EA, this task is defined as extracting the AS from the document $D$ to form

a set $S = \{ s _ { 1 } , s _ { 2 } , \cdot \cdot \cdot , s _ { n } \}$ , where:

$$
s _ { i } \sim P \left( S | A , D , S ^ { * } \right) = \prod _ { i = 1 } ^ { | S ^ { * } | } p \left( s _ { i } | a _ { i } , D , s _ { i } ^ { * } , s ^ { < i } \right)
$$

Task 3: TER Extraction Based on the extracted AS, this task is defined as extracting the corresponding TER from each $s _ { i }$ to obtain $R = \{ r _ { 1 } ^ { * } , r _ { 2 } ^ { * } , \cdot \cdot \cdot , r _ { m } ^ { * } \}$ , where:

$$
\begin{array} { l } { { \displaystyle r _ { i } ^ { * } = \left\{ \langle s _ { j } | r _ { j } | o _ { j } \rangle , \cdots \right\} \sim P \left( R | S , D , S ^ { * } \right) , } } \\ { { \displaystyle P \left( R | S , D , S ^ { * } \right) = \prod _ { i = 1 } ^ { | S | } p \left( r _ { i } | s _ { i } , D , s _ { i } ^ { * } , r ^ { * , < i } \right) } } \end{array}
$$

# Step 2: Best Rationale Selection

For each training sample, we set the temperature parameter $\tau$ of LLMs to 0 to ensure the uniqueness of the extracted structured rationales and generate multiple sub-summaries $S ^ { \mathrm { m u l } }$ and an total summary $S ^ { \mathrm { t o l } }$ using a multi-step summary generation evaluation strategy. This strategy aligns with the requirements of the following two tasks.

Task 4: Multiple Sub-summaries Generation Given a document $D$ , we utilize LLMs again to generate a summary for each structured rationale, referred to as a sub-summary. Ultimately, we obtain a set of multiple sub-summaries $S ^ { \mathrm { m u l } }$ . The structured rationales $R ^ { * } = \{ ( \bar { A } _ { i } , S _ { i } , R _ { i } ) \} _ { i = 1 } ^ { n }$ , the corresponding formula is:

$$
s _ { i } ^ { \mathrm { m u l } } \sim P \left( S ^ { \mathrm { m u l } } | D , R ^ { * } \right) = \prod _ { i = 1 } ^ { | R ^ { * } | } p \left( s _ { i } ^ { \mathrm { m u l } } | D , r _ { i } ^ { * } , s ^ { \mathrm { m u l } , < i } \right)
$$

Task 5: Total Summary Generation Based on the already generated multiple sub-summaries $S ^ { \mathrm { m u l } }$ , we further compress them using LLMs to obtain a shorter total summary $S ^ { \mathrm { t o l } }$ . The formula is as follows:

$$
s _ { i } ^ { \mathrm { { t o l } } } \sim P \left( S ^ { \mathrm { { t o l } } } | S ^ { \mathrm { { m u l } } } , D \right) = \prod _ { i = 1 } ^ { | R ^ { * } | } p \left( s _ { i } ^ { \mathrm { { t o l } } } | s _ { i } ^ { \mathrm { { m u l } } } , D , s ^ { \mathrm { { t o l } } , < i } \right)
$$

Then, we calculate the $\mathsf { R O U G E } _ { \mathrm { N } }$ scores for $S ^ { \mathrm { m u l } }$ and $S ^ { \mathrm { t o l } }$ . Following the evaluation method adopted by Liu et al. (2022); Zhang et al. (2013) during abstractive model validation, we use Eq. (7) and Eq. (8) respectively for CNNDM (Hermann et al. 2015; Nallapati et al. 2016) and XSum (Narayan, Cohen, and Lapata 2018) datasets to calculate the quality scores of the structured rationales. We discard the training examples with low scores and ultimately obtain two new datasets, as shown in Table 1.

$$
\mathrm { R O U G E _ { N } } = \frac { \sum _ { S \in { \cal R e f } } \sum _ { g r a m _ { s } \in S } C o u n t _ { m a t c h } ( g r a m _ { s } ) } { \sum _ { S \in { \cal R e f } } \sum _ { g r a m _ { s } \in S } C o u n t ( g r a m _ { s } ) }
$$

$$
S c o r e _ { 1 } = 1 - \frac { \mathrm { R O U G E _ { 1 } \times R O U G E _ { 2 } } } { \mathrm { R O U G E _ { 1 } + R O U G E _ { 2 } } }
$$

$$
S c o r e _ { 2 } = 1 - \frac { \mathrm { R O U G E _ { 1 } + R O U G E _ { 2 } + R O U G E _ { 3 } } } { 3 }
$$

# Step 3: Multi-view Weak-gated Fusion Small Model Training

In this subsection, we comprehensively train our local SLM using the Multi-view weak-gated Fusion sub-framework (MWF). This framework aligns with structured rationales from LLMs by utilizing additional modules organized in a hierarchical structure. Subsequently, it integrates the features output by these modules with the original abstractive model through a weak-gated mechanism. For simplicity, we use $\langle A \rangle$ , $\bar { \langle S \rangle }$ and $\langle R \rangle$ to represent the features of EA, AS, and TER, respectively. The ultimate goals of this framework are twofold:

Multi-view Hierarchical Aligning of Structured Rationales. We construct three hierarchically structured modules for three viewpoints of structured rationales: the essential aspects module, the associated sentences module, and the triple entity relations module. All modules are based on the Transformer architecture (Vaswani et al. 2017). The input to all modules is the same source document $D$ . After passing through a shared embedding layer (omitted in the Figure 2 for simplicity), each module processes the input $D$ through its own self-attention layer to enrich the semantic content. To ensure that the semantic features $A _ { \mathrm { o u t } }$ , $S _ { \mathrm { o u t } }$ and $R _ { \mathrm { { o u t } } }$ output by these modules align closely with $\langle A \rangle$ , $\langle S \rangle$ and $\langle R \rangle$ , we previously encode $\langle A \rangle$ , $\langle S \rangle$ and $\langle R \rangle$ using the encoder of the abstractive model. Next, we apply average pooling to the encoded $\langle A \rangle , \langle S \rangle$ and $\langle R \rangle$ . Finally, we compute the cosine similarity between the likewise average-pooled semantic features $^ { * } \mathrm { ‰ }$ , $S _ { \mathrm { o u t } }$ , $R _ { \mathrm { o u t } } { \ ' } ^ { * }$ and $^ { \prime } \langle A \rangle , \langle S \rangle , \langle R \rangle ^ { \prime }$ using the formula Cosin $\begin{array} { r } { { e l o s s } = s i m  \boldsymbol x , \boldsymbol y  = ( x \boldsymbol y ) / ( | | x | | | \boldsymbol y | | ) } \end{array}$ , such as $\mathcal { L } _ { \mathrm { E A } } = s i m \left. A _ { \mathrm { o u t } } , \left. A \right. \right.$ .

Features Fusion through Weak-gated Mechanism. To dynamically adjust the fusion degree of various features extracted from different viewpoints at each decoding layer, based on the current context. After $A _ { \mathrm { o u t } } , S _ { \mathrm { o u t } }$ and $R _ { \mathrm { o u t } }$ enter their respective weak-gated networks, each is copied $L$ times, where $L$ represents the number of layers in the decoder of the abstractive model. Once the output $X ^ { e n }$ from the encoder in the abstractive model is obtained, during the decoding phase, the following operations are performed at each layer of the decoder:

$$
X _ { i + 1 } ^ { d e } = \mathrm { { M u l H e a d } } \left( W _ { i } ^ { Q } X _ { i } ^ { d e } , W _ { i } ^ { K } X _ { \mathrm { { n e w } } , i } ^ { e n } , W _ { i } ^ { V } X _ { \mathrm { { n e w } } , i } ^ { e n } \right)
$$

Here, $X _ { \mathrm { n e w } , i } ^ { e n }$ is the result of hierarchically features fusion between the output of the abstractive model’s encoder and structured rationales, and where $i \in L$ :

$$
X _ { \mathrm { n e w } , i } ^ { e n } = X _ { i } ^ { e n } + g _ { i } ^ { A } \cdot A _ { \mathrm { o u t } , i } + g _ { i } ^ { S } \cdot S _ { \mathrm { o u t } , i } + g _ { i } ^ { R } \cdot R _ { \mathrm { o u t } , i }
$$

Here, $g _ { i } ^ { A }$ represents the weak-gated unit at the $i$ -th layer specifically designed to incorporate semantic features related to $\langle A \rangle$ . Its value range is $[ 0 , \bar { 1 } ]$ . It is a continuous value that can be adaptively updated during training. The responsibility of $g _ { i } ^ { S }$ and $\mathring { g } _ { i } ^ { R }$ is similar to that of $\mathbf { \bar { \mathbf { \xi } } } _ { g _ { i } ^ { A } }$ . Different from previous work, which uses a fixed ReLU activation function as a sturdy gate (Yao et al. 2020; Sun, Ren, and Xie 2024), we treat the gate as a kind of adaptively learned weight network parameter, thus called weak-gated unit. By observing the values of weakgated units at different layers, we can gain insight into the model’s dependence on distinct features at different decoding stages, which aids in deeper understanding of the abstractive model’s decision-making process and working mechanism.

<html><body><table><tr><td colspan="3"># Examples</td><td colspan="2"># Avg Words</td></tr><tr><td>Datasets</td><td>Train</td><td>Valid Test</td><td>Doc.</td><td>Sum.</td></tr><tr><td>CNNDM</td><td>287K</td><td>13K</td><td>11K 791.6</td><td>55.6</td></tr><tr><td>CNNDM*</td><td>203K</td><td>13K</td><td>11K 773.2</td><td>57.8</td></tr><tr><td>XSum</td><td>203K</td><td>11K</td><td>11K 429.2</td><td>23.3</td></tr><tr><td>XSum*</td><td>126K</td><td>11K</td><td>11K 457.6</td><td>25.5</td></tr></table></body></html>

Table 1: Datasets Statistics. “\*” represents the datasets processed through steps 1 and 2.

Training Objective of Loss Function To preserve and enhance the generative capabilities of the abstractive model, we adopt a combined loss function that integrates the sequencelevel cosine loss $( \mathcal { L } _ { \mathrm { E A } } + \mathcal { L } _ { \mathrm { A S } } + \mathcal { L } _ { \mathrm { T E R } } )$ with the token-leve cross-entropy loss $\mathcal { L } _ { c r o s s - e n t r o p y }$ . Our composite loss function is formulated as follows:

$$
\mathcal { L } = \gamma _ { 1 } \mathcal { L } _ { c r o s s - e n t r o p y } + \gamma _ { 2 } ( \mathcal { L } _ { \mathrm { E A } } + \mathcal { L } _ { \mathrm { A S } } + \mathcal { L } _ { \mathrm { T E R } } )
$$

Here, $\gamma _ { 1 }$ and $\gamma _ { 2 }$ are two hyper-parameters. Notably, the sequence-level cosine loss effectively complements the tokenlevel cross-entropy loss. This is because the cosine loss captures the overall structural similarity, while the cross-entropy loss acts as a normalization mechanism, ensuring that the model can assign a balanced probability distribution across the entire sequence.

# Experiments Datasets and Metrics

We conduct experiments on two widely-used abstractive summarization datasets: CNN/DailyMail1 (CNNDM) (Hermann et al. 2015; Nallapati et al. 2016) and $\mathrm { \Delta } \mathrm { X S u m } ^ { 2 }$ (Narayan, Cohen, and Lapata 2018). These datasets differ in text length and level of abstraction, allowing us to demonstrate the generalization ability of our method. For original datasets, we first perform preprocessing, which includes removing empty items from documents or summaries. The documentsummary pairs are then filtered through two processing steps, where the threshold for $S c o r e _ { 1 }$ is set to 85 and the threshold for $S c o r e _ { 2 }$ is set to 65. The dataset sizes before and after processing for CNNDM and XSum are shown in Table 1. We use ROUGE (Lin 2004) to measure the quality of abstracts in our results, specifically reporting F1 scores of ROUGE-1 (R-1), ROUGE-2 (R-2), and ROUGE-L (R-L) between ground-truth summaries and the generated abstracts. Additionally, we use BERTScore (BS) to measure semantic similarity between the generated summary and the reference summary.

# Baselines

We choose a variety of strong-performing baseline models for comparison, including BERTSumAbs (Liu and Lapata 2019), T5 (Raffel et al. 2020), BART (Lewis et al. 2019), PEGASUS (Zhang et al. 2020), GSum (Dou et al. 2020), BigBird (Zaheer et al. 2020), SimCLS (Liu and Liu 2021), SeqCo (Xu et al. 2022), GLM (Du et al. 2021), BRIO (Liu et al. 2022), GPT-3.5 (Ouyang et al. 2022), and TriSum (Jiang et al. 2024).

# Setup

We used Llama3- ${ \bf 7 0 B ^ { 3 } }$ as our LLM and BART-large from Hugging Face (Wolf et al. 2020) as our origin abstractive model. The overall parameter number of LSRMWF is $4 3 9 \mathbf { M } ( \leq 1 \mathbf { B } )$ . All experiments are conducted on 2 NVIDIA RTX A6000 GPUs. We employ the Adam optimizer (Kingma and Ba 2014) with learning rate scheduling, where the learning rate $l r$ is calculated as $2 \times$ $1 0 ^ { - 3 } \mathrm { m i n } \left( \mathrm { s t e p } ^ { - 0 . 5 } \right.$ , step $\cdot \cdot$ warmup−1.5 , where step representing the number of update steps, and warmup set to 10000. For CNNDM, the initial weak-gated units are all set to 0.02, $\gamma _ { 1 } = 0 . 6$ and $\gamma _ { 2 } = 0 . 4$ . For XSum, the initial weak-gated units are all set to 0.01, $\gamma _ { 1 } = 0 . 7$ and $\gamma _ { 2 } = 0 . 3$ .

# Results

The results are shown in Table 2. When utilizing structured rationales, Llama3-70B shows excellent summarization ability, outperforming all baselines, which indicates that structured rationales contribute to generating higher quality summaries. Furthermore, LSR-MWF outperforms many models across both datasets, highlighting its strength and adaptability. More specifically, LSR-MWF outperforms the original $\mathbf { B A R T _ { l a r g e } }$ by $1 1 . 2 \%$ and $5 . 8 \%$ on the two datasets, respectively. Therefore, LSR-MWF not only retains the ability of the abstractive model but also improves the quality of summary generation. It is worth noting that LSR-MWF performs better than the model TriSum, which also utilizes structured rationales, and this illustrates the effectiveness of our refined structured rationales and model design.

Ablation Study Table 3 examines the impact of removing different “gems” of structured rationales and their corresponding module on model performance. The results reveal that for both CNNDM and XSum datasets, removing TER (relation-level information) has a significant effect on model performance. Specifically, the model’s performance degrades most when TER is absent, indicating its crucial role in summary generation. Additionally, we observe that relying solely on TER doesn’t yield as good results as using AS (sentencelevel information) alone. We assume that this is because TER is extracted from sentences, and without the support of AS, the model struggles to effectively utilize TER for CNNDM. Conversely, for XSum, TER alone demonstrates greater benefit when compared to AS alone. This may be attributed to the higher abstract level of summaries in XSum. To sum up, this multi-view constraint and hierarchically structured modules aid the abstractive model in making more accurate inferences and generations in complex contexts.

<html><body><table><tr><td></td><td colspan="4">CNNDM</td><td colspan="4">XSum</td></tr><tr><td>Model</td><td>R-1</td><td>R-2</td><td>R-L</td><td>BS</td><td>R-1</td><td>R-2</td><td>R-L</td><td>BS</td></tr><tr><td>BERTSumAbs (Liu and Lapata 2019)</td><td>41.18</td><td>18.73</td><td>37.22</td><td>0.8576</td><td>38.81</td><td>16.48</td><td>31.00</td><td>0.8723</td></tr><tr><td>T5Large (Raffel et al. 2020)</td><td>42.42</td><td>20.78</td><td>39.93</td><td>0.8722</td><td>40.12</td><td>17.23</td><td>32.34</td><td>0.9073</td></tr><tr><td>BARTLarge (Lewis et al. 2019)</td><td>44.01</td><td>21.12</td><td>40.58</td><td>0.8798</td><td>45.42</td><td>22.31</td><td>37.28</td><td>0.9162</td></tr><tr><td>PEGASUS (Zhang et al. 2020)</td><td>44.23</td><td>21.57</td><td>41.30</td><td>0.8737</td><td>46.71</td><td>24.38</td><td>38.89</td><td>0.9190</td></tr><tr><td>GSum (Dou et al. 2020)</td><td>45.52</td><td>22.32</td><td>42.13</td><td>0.8783</td><td>45.12</td><td>21.53</td><td>36.55</td><td>0.9123</td></tr><tr><td>BigBirdLarge (Zaheer et al. 2020)</td><td>43.83</td><td>21.12</td><td>40.74</td><td>0.8803</td><td>47.13</td><td>24.06</td><td>38.77</td><td>0.9197</td></tr><tr><td>SimCLS (Liu and Liu 2021)</td><td>45.57</td><td>21.91</td><td>41.02</td><td>0.8828</td><td>46.59</td><td>24.20</td><td>39.11</td><td>0.9078</td></tr><tr><td>SeqCo (Xu et al. 2022)</td><td>45.02</td><td>21.79</td><td>41.81</td><td>0.8747</td><td>45.59</td><td>22.38</td><td>37.02</td><td>0.9135</td></tr><tr><td>GLMRoBERTa (Du et al. 2021)</td><td>43.82</td><td>20.97</td><td>40.45</td><td>0.8733</td><td>45.51</td><td>23.48</td><td>37.33</td><td>0.8855</td></tr><tr><td>BRIO-Mul (Liu et al. 2022)</td><td>47.63</td><td>23.53</td><td>44.49</td><td>0.8874</td><td>47.10</td><td>24.52</td><td>39.15</td><td>0.9238</td></tr><tr><td>GPT-3.5zero-shot (Ouyang et al. 2022)</td><td>37.42</td><td>13.78</td><td>29.10</td><td>0.8770</td><td>26.63</td><td>06.71</td><td>18.78</td><td>0.8767</td></tr><tr><td>TriSum (Jiang et al. 2024)</td><td>45.72</td><td>22.70</td><td>41.93</td><td>0.8850</td><td>47.33</td><td>24.39</td><td>39.01</td><td>0.9217</td></tr><tr><td>GPT-3.5TriSum (Jiang et al. 2024)</td><td>46.68</td><td>23.48</td><td>40.73</td><td>0.8920</td><td>34.44</td><td>12.61</td><td>28.43</td><td>0.8925</td></tr><tr><td>Llama-3-70B zero-shot (Touvron et al. 2023)</td><td>38.56</td><td>14.69</td><td>30.78</td><td>0.8795</td><td>33.42</td><td>11.66</td><td>26.73</td><td>0.8916</td></tr><tr><td>Llama-3-7OB w/ structured rationale</td><td>50.85</td><td>25.27</td><td>46.42</td><td>0.9106</td><td>49.74</td><td>27.41</td><td>40.54</td><td>0.9325</td></tr><tr><td>LSR-MWF(≤1B)</td><td>48.54</td><td>23.91</td><td>45.10</td><td>0.8977</td><td>47.47</td><td>24.32</td><td>39.28</td><td>0.9257</td></tr></table></body></html>

Table 2: Performance comparison of ROUGE and BERTScore scores on CNN/DailyMail and XSum datasets. We highlight the top-2 results in bold font. Our backbone model $\mathbf { B A R T _ { L a r g e } }$ is colored gray for reference.

Table 3: Ablation study of LSR-MWF. “w/o” means without. “EA” means essential aspects, “AS” means associated sentences, and “TER” means triple entity relations.   

<html><body><table><tr><td colspan="4">CNNDM</td><td colspan="3">XSum</td></tr><tr><td>Model</td><td>R-1</td><td>R-2</td><td>R-L</td><td>R-1</td><td>R-2</td><td>R-L</td></tr><tr><td>LSR-MWF</td><td>48.54</td><td>23.91</td><td>45.10</td><td>47.47</td><td>24.32</td><td>39.28</td></tr><tr><td>W/o EA</td><td>48.08</td><td>23.13</td><td>44.29</td><td>47.34</td><td>24.01</td><td>38.97</td></tr><tr><td>w/o AS</td><td>47.55</td><td>22.87 43.78</td><td></td><td></td><td>46.93 23.58 38.66</td><td></td></tr><tr><td>W/o TER</td><td>46.37</td><td></td><td>22.73 42.22</td><td>45.22</td><td>22.95 37.39</td><td></td></tr><tr><td>w/oAS&TER</td><td>44.58</td><td>21.37</td><td>41.06</td><td>45.09</td><td>22.37 36.89</td><td></td></tr><tr><td>W/oEA&TER</td><td>45.22</td><td>21.89</td><td>41.39</td><td>44.73</td><td>22.25</td><td>36.78</td></tr><tr><td>w/oEA&AS</td><td>44.76</td><td>21.58</td><td>41.15</td><td>46.69</td><td>23.46</td><td>38.43</td></tr></table></body></html>

# Analysis

Superiority of Weak-gated Mechanism To observe from Table 4, the initialization weights of weak-gated units should not be excessively large or overly small. Furthermore, we found that when weights of weak-gated units are set to a fixed value and not updated during training, the performance of the model is significantly reduced. This suggests that in the process of integrating structured rationales into the abstractive model, the weights of weak-gated units adaptively and dynamically change to effectively fuse with the features of different decoder layers to improve model’s performance.

Visualization of Weak-gated Mechanism According to Figure 4, we further illustrate the dynamic changes of the weights of weak-gated units during training for LSR-MWF with three types of “gems” inputs. For the CNNDM dataset, it can be observed that the weight fluctuation of the weak-gated units corresponding to the first six decoder layers is relatively small, maintaining a value close to 0.02. However, the fluctuations in the last 6 layers are much larger. Additionally, we note that the weak-gated weights of TER change more rapidly. This may be due to the critical nature of TER, which is highly required by different decoder layers, necessitating quick adaptation of its weights. For XSum, it is evident that only the weak-gated units in the last four layers are active. Notably, the weak-gated units in the final layer exhibit rapid fluctuations across the three layers of EA, AS, and TER. This suggests that the information from the last decoder layer of the abstractive model may be important, requiring the weights to rise rapidly to adapt.

Table 4: Analytical experiments on initialization weights of weak-gatd units on CNNDM. “\*” represents that the weights of weak-gated unit are fixed.   

<html><body><table><tr><td>Gate Init.</td><td>R-1</td><td>R-2</td><td>R-L</td><td>BS</td></tr><tr><td>0.001</td><td>47.21</td><td>22.76</td><td>43.92</td><td>0.8825</td></tr><tr><td>0.015</td><td>47.74</td><td>23.32</td><td>44.38</td><td>0.8851</td></tr><tr><td>0.020</td><td>48.54</td><td>23.91</td><td>45.10</td><td>0.8977</td></tr><tr><td>0.020*</td><td>46.60</td><td>20.33</td><td>42.83</td><td>0.8785</td></tr><tr><td>0.025</td><td>48.02</td><td>23.53</td><td>44.76</td><td>0.8943</td></tr><tr><td>0.100</td><td>15.02</td><td>5.44</td><td>14.72</td><td>0.8513</td></tr></table></body></html>

Study of Sequence-level and Token-level Loss Figure 5 indicates that the optimal ratio of $\gamma _ { 1 } { : } \gamma _ { 2 }$ for the CNNDM dataset is 6:4, while the best ratio for the XSum dataset is 7:3. This suggests that CNNDM prefers sequence-level cosine loss, whereas

Article: (CNN) -- There’s some magic coming to a British stage. Author J.K. Rowling has announced she is developing a play based on her "Harry Potter" stories. According to her website, Rowling is working in collaboration with award-winning producers Sonia Friedman and Colin Callender on the project. "Over the years I have received countless approaches about turning Harry Potter into a theatrical production, but Sonia and Colin’s vision was the only one that really made sense to me, and which had the sensitivity, intensity and intimacy I thought appropriate for bringing Harry's story to the stage," Rowling said in a statement. "After a year in gestation it is exciting to see this project moving on to the next phase. I’d like to thank Warner Bros. for their continuing support in this project." Warner Bros. is owned by CNN’s parent company, Time Warner. Rowling will reportedly be a producer of the play and work with a writer, but she will not be writing the play. The story will follow Potter in his early years as an orphan. Directors and writers for the play, which will go into development in 2014, are currently being considered.   
Ground Truth summary: J.K. Rowling Structured Rationales:   
is developing a "Harry Potter" play. The Harry Potter Play | Author J.K. Rowling has announced she is developing a play based on her "Harry story will follow Potter in his early years   
Potter" stories. | <J.K. Rowling # is developing # Harry Potter play>, <J.K. Rowling # has announced as an orphan. The play will go into   
# play>   
development in 2014.   
Harry Potter Story | The story will follow Potter in his early years as an orphan. | <The story # will Bart summary: J.K. Rowling is devel- follow # Potter>, <The story # will follow # orphan>   
oping a play based on her "Harry Potter" Play Development | The play, which will go into development in 2014. | <The play # will go into stories. The story will follow Potter in his   
development # 2014>   
early years as an orphan. Rowling will   
reportedly be a producer of the play and LSR-MWF summary:   
work with a writer. Directors and writers J.K. Rowling is developing a play based on her "Harry Potter" stories, which will follow Harry Potter for the play are currently being considered. in her early years as an orphan and is set to go into development in 2014.

Figure 3: An example of CNNDM. EA, AS and TER are respectively wrapped in red, yellow and purple boxes, like the document of Figure 1. For the three summaries in the figure, the semantically identical parts are colored with the same color.

CNNDM-EA 123456 XSum-EA   
0.02008 0.00125   
0.02004 0.00115   
0.02000 7 0.00105 8   
0.01996 9 0.00095 10   
0.01992 11 0.00085   
0 0.5k 1k 1.5k 2k 2.5k 3k 3.5k 12 0 0.5k 1k 1.5k 2k 2.5k 3k 3.5k CNNDM-AS 1 XSum-AS   
0.02008 2 3 4 0 5 0.00115   
N 67 0.00105   
0.01996 89 10   
U 0 0.5k 1k 1.5k 2k 2.5k 3k 3.5k 112 0 0.5k 1k 1.5k m2-k 2.R5k 3k 3.5k   
0.02006 CNNDM-TER 1 XSu   
0.02004 23 0.00125   
0.02002 4567   
0.02000 0.00115   
0.01998 0.00105   
0.01996 891 0.00095   
0.01994   
0.01992 11 0.00085 0 0.5k 1k 1.5k 2k 2.5k 3k 3.5k -120 0.5k 1k 1.5k 2k 2.5k 3k 3.5k Step Step CNNDM R-1 XSum R-1   
48.753 48.2948.11 48.14 48.35 47.53 47.16 47.1347.4747.18 47.3147.18 47.1547.26 48.0348.06   
47.9 47.91 46.9   
47.7 46.7   
47.5 9:1 8:2 7:3 6:4 5:5 4:6 3:7 2:8 1:9 46.5 9:1 8:2 7:3 6:4 5:5 4:6 3:7 2:8 1:9 CNNDM R-2 XSum R-2   
24.0 23.91 24.5 24.32   
23.86 23.6623.48 23.51 23.40 23.43 23.72 24.31 23.92 23.89 23.96 23.9724.0 2 23.9524.09   
23.4 23.37 23.28 23.9   
23.2 23.7   
23.0 23.5   
22.8 23.3 9:1 8:2 7:3 6:4 5:5 4:6 3:7 2:8 1:9 9:1 8:2 7:3 6:4 5:5 4:6 3:7 2:8 1:9

and its consequences. The AS consists of the most relevant and direct statements from the documents. The TER further refines these statements to clarify the relationship between the entities involved in AS. This technique ensures the completeness of the summary and improves clarity, allowing the reader to follow the content of the summary back to its main aspects and detailed triples for a deeper understanding of the summarization process.

XSum prefers token-level loss. This difference may be attributed to the higher level of abstraction in the summaries in XSum.

Case Study Figure 3 compares a summary of a CNNDM article discussing the upcoming production of J.K. Rowling’s Harry Potter script. On the one hand, BART’s summary specifies the characters and storyline in detail, it omits some key information such as the release event and time. On the other hand, the structured rationales of our method are hierarchically progressive, starting with EA, progressing to AS, and finally to TER. The EA presents a high-level overview of the event

# Conclusion

In this work, we propose a distillation method, LSR-MWF, which leverages LLMs and employs specific strategies to obtain high-quality structured rationales. These rationales are then used to hierarchically guide SLMs. Our method bridges the gap between LLMs and SLMs, enabling SLMs to inherit the structured abstractive summarization capabilities of LLMs while maintaining high performance and interpretability. We believe that the model’s performance can be further enhanced by refining and extending the structured rationales, such as oppositional or anaphora relations.