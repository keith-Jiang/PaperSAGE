# Small Language Model Makes an Effective Long Text Extractor

Yelin Chen1\*†, Fanjin Zhang2\*‡, Jie Tang2‡

1School of Computer Science and Technology, Xinjiang University, Urumqi 830049, China 2Department of Computer Science and Technology, Tsinghua University, Beijing 100084, China ylin $@$ stu.xju.edu.cn, {fanjinz, jietang}@tsinghua.edu.cn

# Abstract

Named Entity Recognition (NER) is a fundamental problem in natural language processing (NLP). However, the task of extracting longer entity spans (e.g., awards) from extended texts (e.g., homepages) is barely explored. Current NER methods predominantly fall into two categories: span-based methods and generation-based methods. Span-based methods require the enumeration of all possible token-pair spans, followed by classification on each span, resulting in substantial redundant computations and excessive GPU memory usage. In contrast, generation-based methods involve prompting or fine-tuning large language models (LLMs) to adapt to downstream NER tasks. However, these methods struggle with the accurate generation of longer spans and often incur significant time costs for effective fine-tuning. To address these challenges, this paper introduces a lightweight span-based NER method called SeNER, which incorporates a bidirectional arrow attention mechanism coupled with LogNScaling on the [CLS] token to embed long texts effectively, and comprises a novel bidirectional sliding-window plus-shaped attention (BiSPA) mechanism to reduce redundant candidate token-pair spans significantly and model interactions between token-pair spans simultaneously. Extensive experiments demonstrate that our method achieves state-ofthe-art extraction accuracy on three long NER datasets and is capable of extracting entities from long texts in a GPUmemory-friendly manner.

# Code — https://github.com/THUDM/scholarprofiling/tree/main/sener

# Introduction

Named entity recognition (NER), a fundamental task in information extraction (IE), aims to identify spans indicating specific types of entities. It serves as the foundation for numerous downstream tasks, including relation extraction (Miwa and Bansal 2016), knowledge graph construction $\mathrm { \Delta X u }$ et al. 2017), and question answering (Molla´, Van Zaanen, and Smith 2006).

# Dr. Ware's profile snippet:

Prior to joining the UMMS faculty , Dr. Ware founded QualityMetric Incorporated and served as its CEO and Chairman for more than 10 years . . . where he led the development of health status measures used in the Health Insurance Experiment . He has published more than 400 peer-reviewed articles , …… was the first recipient of the International Society for Pharmacoeconomics and Outcomes Research ( ISPOR ) Avedis Donabedian Outcomes Research Lifetime Achievement Award Work Experience Gender Award

Despite extensive studies, existing NER research rarely focuses on extracting named entities from long texts, a common real-world scenario such as extracting author attributes from homepages and identifying “methods” and “problems” in academic papers. For example, in Figure 1, “work experience” is a long entity block while “award” is a long entity, posing greater challenges for the NER task. We also extend the input length of a NER method to extract short entities in academic papers, as shown in Figure 2a, suggesting that long input length brings clear benefits to extract entities more precisely due to the perception of longer contexts.

Traditional approaches treat the NER task as a sequence labeling task, assigning a single label to each token, exemplified by the BIOES format. However, these methods are inadequate for recognizing nested entities. To address this issue, later efforts typically employ span-based methods (Su et al. 2022; Yan et al. 2023b), which consider all possible token-pair spans and classify each span. These methods achieve satisfactory accuracy on sentence-based NER tasks but struggle to identify entity blocks across sentences or extract entities from long texts due to substantial redundant computations and GPU memory usage resulting from $\mathcal { O } ( L ^ { 2 } )$ computation complexity based on the tokenpair span tensor, where $L$ is the input length.

Recently, large language models (LLMs) demonstrate remarkable performance on a spectrum of natural language understanding and generation tasks (Zhao et al. 2023). However, LLMs still fall short and do not align well with information extraction tasks (Qi et al. 2024). On the ScholarXL dataset (Zhang et al. 2024), extracting author attributes via prompting GPT-4o (Achiam et al. 2023) by providing (a) Impact of input length for our model SeNER regarding extraction performance on the SciREX dataset $( \% )$ .

![](images/6e37e28f2e714ff791b35b2617bc985d0d51b1602a1f684971d54fe2eec71e30.jpg)

![](images/28e7de5217026a7c86e166f4d2e97293de526e8a930d25717d6eb4a5867dc662.jpg)  
(b) Model parameters of span-based and generation-based methods vs. F1 score on the Scholar-XL dataset.

Figure 2: Performance of entity recognition with respect to input length and the number of model parameters of NER methods.

5 similar demonstrations only achieves a $2 1 . 8 7 \%$ F1 score, as shown in Figure 2b. Fine-tuning LLMs to extract entities from long texts is also feasible (Sainz et al. 2023; Qi et al. 2024), but it incurs significant time costs for training and inference compared to span-based methods, without guaranteeing high accuracy.

Therefore, we aim to recognize named entities from long texts in a GPU-memory-friendly way without compromising accuracy. Given these limitations, we propose SeNER, a lightweight span-based method to extract entities from long texts. The main idea of SeNER is to reduce the redundant computations during the encoding and extraction processes of long texts. SeNER presents two core components that have an edge over existing span-based NER methods.

• To encode long texts effectively and efficiently, we employ a bidirectional arrow attention mechanism that encodes both local and global contexts simultaneously. To overcome the entropy instability issue of input texts of varied length, we apply LogN-Scaling $\left( \operatorname { S u } 2 0 2 1 \right)$ on the [CLS] token to keep the entropy of attention scores stable. • To reduce superfluous span-based computation and model interactions between token-pair spans, we propose a novel bidirectional sliding-window plus attention (BiSPA) mechanism to efficiently compute horizontal and vertical attention on focused spans.

To enhance the robustness and generalization of our model, we employ the whole word masking strategy (Cui et al. 2021) and the LoRA (Hu et al. 2021) technique during training. Extensive experimental results on three datasets highlight the superiority of our proposed method. SeNER achieves state-of-the-art accuracy while maintaining relatively small model parameters, as depicted in Figure 2b. Additionally, under the same hardware and configuration, our model is capable of handling texts 6 times longer than previous advanced span-based NER methods.

# Related Work

NER methods are generally categorized into span-based methods, generation-based methods, and other methods.

# Span-based Methods

Span-based methods (Li et al. 2022; Yuan et al. 2022; Su et al. 2022; Zhu and Li 2022) reframe the NER task as a token-pair span classification task. They identify spans based on start and end positions, enumerate all possible candidate spans in a sentence, and perform classification. Most existing methods focus on obtaining high-quality span representations and modeling interactions between spans. CNNNER (Yan et al. 2023a) utilizes Convolutional Neural Networks (CNNs) to model spatial relations in the token-pair span tensor. UTC-IE (Yan et al. 2023b) further incorporates axis-aware interaction with plus-shaped self-attention for the token-pair span tensor on top of CNN-NER. These methods offer parallel extraction, simple decoding, and advantages in handling nested entity recognition, leading to widespread use and excellent performance. However, calculating all span representations and aggregating interactions between token-pair spans requires substantial computational resources, which limits their effectiveness for long texts.

# Generation-based Methods

Generation-based methods extract entities from text in an end-to-end manner, where the generated sequence can be text (Lu et al. 2022; Jiang et al. 2024), entity pointers (Yan et al. 2021), or code (Sainz et al. 2023). With the rise of large language models (LLMs), such methods (Wang et al. 2023a; Xie et al. 2023; Ashok and Lipton 2023) achieve good performance with only a few examples due to their generalization abilities. Some methods (Wang et al. 2023b; Dagdelen et al. 2024) enhance general extraction capabilities by using powerful LLMs, high-quality data, diverse extraction tasks, and comprehensive prior knowledge. GoLLIE (Sainz et al. 2023) ensures adherence to annotation guidelines through strategies such as class order shuffling, class dropout, guideline paraphrasing, representative candidate sampling, and class name masking. ADELIE (Qi et al. 2024) performs instruction tuning on a high-quality alignment corpus and further optimizes it with a Direct Preference Optimization (DPO) objective. However, compared to span-based methods, these methods often require significant computational resources and may perform poorly in generating accurate longish entities from long texts. The construction of instructions and use of examples can compress input text length, leading to low text utilization. Additionally, autoregressive generation can result in long decoding times.

# Other Methods

In addition to the two main paradigms, there are a few other types of methods. Some methods (Ma and Hovy 2016; Yan et al. 2019; Strakova´, Straka, and Hajiˇc 2019) model the NER task as a sequence labeling task. However, these methods struggle with nested entities. Some methods (Li et al. 2019; Tan et al. 2021; Shen et al. 2022) use two independent multi-layer perceptrons (MLPs) to predict the start and end positions of entities separately, which can lead to errors due to treating the entity as separate modules. Some approaches (Lou, Yang, and Tu 2022; Yang and $ { \mathrm { T u } } \ 2 0 2 2 )$ employ hypergraphs to represent spans, but their decoding processes is complex.

# Problem Definition

In this section, we introduce the problem formulation of named entity recognition from long texts.

Problem 1 Named Entity Recognition from Long Texts (Long NER). Given a long input text, the goal is to extract different types of named entities or entity blocks that mark their start and end positions in the text. Note that the input length can exceed 1,000 tokens and the entity length can exceed 100 tokens in our problem.

Taking scholar profiling (Schiaffino and Amandi 2009; Gu et al. 2018) as an example, “birth place” is a kind of entity, while “work experience” often appears as an entity block that involves multiple segments.

# Method

As previously discussed, conventional NER methods fall into two main categories: span-based and generation-based. For NER in long texts, span-based methods need to model interactions between token-pair spans, which incurs substantial GPU memory and computation. In contrast, generationbased methods, commonly based on LLMs, are arduous to generate longish entity spans accurately.

In response to these limitations, we propose a lightweight span-based NER model, SeNER, that efficiently encodes long input texts and models token-pair spans interactions. First, we employ a pre-trained language model (PLM) with a arrow attention mechanism to encode long inputs efficiently. To alleviate entropy instability resulting from varied input lengths, we apply LogN-Scaling (Su 2021) to the [CLS] token. Next, we leverage a Biaffine model (Dozat and Manning 2017) to obtain the hidden representation of each tokenpair span. Then, we present the token-pair span interaction module, where we propose a novel BiSPA mechanism to significantly reduce redundant candidate token pairs and model interactions between token pairs simultaneously. Finally, we introduce the training strategy and prediction method. An overview of our model is shown in Figure 3.

![](images/9607aa21da2b9e9bda8d4aa6d19c60e62482de39a3445985d24f69da46020488.jpg)  
Figure 3: An overview of the SeNER model.

![](images/2e9c88452096f962a7ea85ca0a599ac489cda10e9954f9389f4e1d4cfc03b149.jpg)  
Figure 4: Illustration of arrow attention, full attention, and sliding window attention.

# Long Input Encoding

Given a piece of text, we pass it into a PLM to obtain its contextual vector representation.

$$
H = [ h _ { 1 } , h _ { 2 } , . . . , h _ { L } ] = \mathrm { P L M } \left( [ x _ { 1 } , x _ { 2 } , . . . , x _ { L } ] \right)
$$

where $H \in \mathbb { R } ^ { L \times d }$ , $L$ is the input length, and $d$ is the output dimension of the PLM.

Traditional NER methods utilize PLMs with full bidirectional attention, incuring a large amount of GPU memory footprint and computation for long texts. Moreover, full attention for long texts is often unnecessary since distant tokens are usually semantically unrelated. In light of this, a straightforward idea is to use sliding window attention (SWA) (Beltagy, Peters, and Cohan 2020; Zaheer et al. 2020), which adopts a fixed window, say $w$ , so that each token attends to $w$ tokens to its left and $w$ tokens to its right. However, SWA ignores the global context, impairing the ability of the Transformer layers to acquire a comprehensive understanding of the entire input text.

To this end, we propose an Arrow Attention mechanism, where the [CLS] token uses global attention while other tokens use local sliding window attention, as illustrated in Figure 4. Arrow Attention strikes a balance between global and local attention. Compared to the computational complexity of $\mathcal { O } ( L ^ { 2 } )$ for the full attention, arrow attention only requires $\mathcal { O } ( w L )$ . Furthermore, the global information captured by the [CLS] token supplements the knowledge of SWA, enhancing the representation of each token and mitigating the information loss caused by the fixed receptive field. Thus, the [CLS] token acts as an attention sink (Xiao et al. 2023) that balances the weights of global and local contexts.

However, varying text lengths can cause entropy instability for the [CLS] token, where the scale of attention scores can change significantly. In this regard, we employ a LogNScaling technique on the [CLS] token to stabilize the entropy of attention scores. Specifically, LogN-Scaling is defined as follows:

$$
\begin{array} { r } { H _ { [ \mathbb { C } \mathbb { L } \mathrm { S } ] } ^ { t } = \mathrm { A t t n } _ { \mathrm { s } } \left( H _ { [ \mathbb { C } \mathbb { L } \mathrm { S } ] } ^ { t - 1 } W ^ { Q } , H ^ { t - 1 } W ^ { K } , H ^ { t - 1 } W ^ { V } \right) } \\ { \mathrm { A t t n } _ { \mathrm { s } } \left( Q , K , V \right) = \mathrm { s o f t m a x } \left( \frac { \log _ { 5 1 2 } L } { \sqrt { d } } Q K ^ { \top } \right) V } \end{array}
$$

where $\mathsf { A t t } \mathsf { n } _ { s }$ is the scaled attention, $H ^ { t }$ is the hidden representation of the $t$ -th Transformer layer, and $W ^ { Q } , \dot { W } ^ { K } , W ^ { V } \in \mathbb { R } ^ { d \times d }$ are projection matrices.

Note that LogN-Scaling is commonly used for length extrapolation in LLMs and imposed on all input tokens. Here we utilize LogN-Scaling solely on the [CLS] token to improve the stability and robustness of our model.

# Biaffine Model

Subsequently, the hidden representation $H$ is fed into a Biaffine model to extract features for each candidate span.

$$
\begin{array} { r } { H ^ { s } , H ^ { e } = \mathbf { M } \mathbf { L } \mathbf { P } _ { \mathrm { s t a r t } } \left( H \right) , \mathbf { M } \mathbf { L } \mathbf { P } _ { \mathrm { e n d } } \left( H \right) } \\ { S _ { i , j } = ( H _ { i } ^ { s } ) ^ { \top } W _ { 1 } H _ { j } ^ { e } + W _ { 2 } ( H _ { i } ^ { s } \oplus H _ { j } ^ { e } ) + b } \end{array}
$$

where $\mathbf { M L P _ { \mathrm { s t a r t } } }$ and $\mathbf { M L P _ { e n d } }$ are multi-layer perceptrons, $H ^ { s } / H ^ { e } \in \mathbb { R } ^ { L \times d }$ are hidden start/end embeddings, $W _ { 1 } \ \in$ $\mathbb { R } ^ { d \times c \times d }$ , $W _ { 2 } \in \mathbb { R } ^ { c \times 2 d }$ , $b \in \mathbb { R } ^ { c }$ , and $c$ is the output dim - sion of the Biaffine model. The symbol $\oplus$ represents the concatenation operation. $S \in \mathbb { R } ^ { L \times \mathbf { \bar { L } } \times c }$ , called token-pair span tensor, denotes the hidden representation of each candidate span. For example, $S _ { i , j }$ represents the features of $[ x _ { i } , . . . , x _ { j } ]$ .

# Token-Pair Span Interaction Module

Note that the token-pair span tensor $S$ considers each possible candidate span. However, for long input texts, it is unnecessary to consider every candidate span, especially for extremely long spans. Additionally, the GPU memory occupied by tensor $S$ increases quadratically with the input length $L$ . In light of this, we propose preserving only the hidden features of spans whose lengths do not exceed $w ^ { \prime }$ as shown in Figure 5. Thus, S is compressed to Sh ∈ RL×w′×c.

Previous studies (Yan et al. 2023a,b) show that modeling the interactions between token pairs, such as plus-shaped and local interaction, should be helpful. Plus-shaped attention applies the self-attention mechanism horizontally and vertically. However, plus-shaped attention cannot be performed directly on the compressed hidden feature tensor $S _ { h }$ since either the original horizontal or vertical dimension is disrupted. Therefore, we propose a novel bidirectional sliding-window plus attention (BiSPA) mechanism to perform plus-shaped attention on the compressed $S _ { h }$ .

![](images/f5dac8024042ecedaede3c72a81918db8f91dd51eabe9b2bb913ddd319059afe.jpg)  
Figure 5: Diagram of the transformation for the token-pair span tensors in BiSPA mechanism.

Specifically, we first compute the horizontal self-attention on $S _ { h }$ , as shown in the top middle of Figure 5. Next, we propose a transformation method, that transforms the top left matrix $S$ to the bottom middle matrix $S _ { v }$ , and then compute the vertical self-attention based on $S _ { v }$ . Finally, we concatenate the horizontal and vertical attention matrices and feed them into an MLP to aggregate plus-shaped perceptual information. Notably, the computational complexity of the BiSPA mechanism is reduced from $\mathcal { O } ( L ^ { 3 } )$ to $\mathcal { O } ( L \times ( w ^ { ' } ) ^ { 2 } )$ , optimizing the training efficiency significantly.

$$
\begin{array} { r } { Z _ { i , : } ^ { h / v } = \mathrm { A t t n } \left( S _ { i , : } ^ { h / v } W _ { h / v } ^ { Q } , S _ { i , : } ^ { h / v } W _ { h / v } ^ { K } , S _ { i , : } ^ { h / v } W _ { h / v } ^ { V } \right) } \\ { \mathrm { A t t n } \left( Q , K , V \right) = \mathrm { s o f t m a x } \left( \frac { Q K ^ { T } } { \sqrt { c } } \right) V } \end{array}
$$

$$
S ^ { \prime } = \mathbf { M } \mathbf { L } \mathbf { P } \left( Z ^ { h } \oplus Z ^ { v } \right)
$$

where $W _ { h } ^ { Q } , W _ { h } ^ { K } , W _ { h } ^ { V } , \ W _ { v } ^ { Q } , W _ { v } ^ { K } , W _ { v } ^ { V } \ \in \ \mathbb { R } ^ { c \times c }$ , $Z ^ { h } / Z ^ { v }$ is intermediate representation after horizontal/vertical selfattention, and $S ^ { \prime } \stackrel { - } { \in } \mathbb { R } ^ { L \times w ^ { \prime } \times c }$ is the token-pair span feature after BiSPA mechanism.

The BiSPA mechanism endows the model with the capacity to perceive horizontal and vertical directions. We further use two types of position embeddings to enhance the sense of distances between token pairs and the area the token pair locates (Yan et al. 2023b). (1) Rotary Position Embedding (RoPE) (Su et al. 2024) encodes the relative distance between token pairs, which is used for both horizontal and vertical self-attention. (2) Matrix Position Embedding indicates whether each entry in $S ^ { \prime }$ is the original upper or lower triangles, which adds to $S _ { h }$ and $S _ { v }$ .

After the BiSPA mechanism, we employ CNN with kernel size $3 \times 3$ on $S ^ { ' }$ to model the local interactions between token-pair spans.

$$
S ^ { \prime \prime } = \operatorname { R e c o v e r } ( \operatorname { C o n v } \left( \sigma \left( \operatorname { C o n v } \left( S ^ { \prime } \right) \right) \right) )
$$

where $S ^ { \prime \prime } \in \mathbb { R } ^ { L \times L \times c }$ is recovered to the square size, and $\sigma$ is the activation function.

We name the module encompassing the BiSPA mechanism and the convolutional module as the BiSPA Transformer block. The BiSPA Transformer blocks will be repeatedly used to ensure full interaction between token pairs.

# Training and Prediction

We utilize MLP layers to transform the output of the final BiSPA Transformer block into output scores. We use binary cross-entropy as the loss function.

$$
\widehat { Y } = \mathbf { M } \mathbf { L } \mathbf { P } \left( S ^ { \prime \prime } + S \right)
$$

$$
\begin{array} { r l } & { \mathcal { L } = - \displaystyle \sum _ { i , j = 1 } ^ { L } \sum _ { r = 1 } ^ { R } \left( Y _ { i , j } ^ { r } l o g \left( \widehat { Y } _ { i , j } ^ { r } \right) \right. } \\ & { \left. \mathrm { ~ \ ~ \ } + \left( 1 - Y _ { i , j } ^ { r } \right) l o g \left( 1 - \widehat { Y } _ { i , j } ^ { r } \right) \right) } \end{array}
$$

where $\widehat { Y } \in \mathbb { R } ^ { L \times L \times R }$ represents the scores of candidate entities, abnd $R$ is the number of entity types.

To improve the robustness and generalization of our model, we employ the whole word masking strategy (Cui et al. 2021) during training and utilize LoRA (Hu et al. 2021) technique to train the PLM parameters.

During prediction, our model uses the average of the upper triangular and lower triangular values as the final prediction score, as follows:

$$
P _ { i , j } ^ { r } = \frac { \left( \widehat { Y } _ { i , j } ^ { r } + \widehat { Y } _ { j , i } ^ { r } \right) } { 2 } , i \leq j
$$

All text spans that satisfy $P _ { i , j } ^ { r } > 0$ are outputted. If the boundaries of multiple candidate spans conflict, the span with the highest prediction score is selected.

# Experiment

# Datasets

We conduct experiments on three NER datasets: ScholarXL (Zhang et al. 2024), SciREX (Jain et al. 2020), and Profiling-07 (Tang, Zhang, and Yao 2007; Tang et al. 2008). The statistics of all datasets are detailed in Table 1. As shown in Table 1, the input lengths and entity lengths of the three datasets are longer than those of traditional named entity recognition datasets, presenting greater challenges.

# Baselines

We compare our model with several recent NER methods:

Span-based Methods: CNN-NER (Yan et al. 2023a): is a span-based method that utilizes Convolutional Neural Networks (CNN) to model local spatial correlations between spans. UTC-IE (Yan et al. 2023b): models axis-aware interaction with plus-shaped self-attention and local interaction with CNN on top of the token-pair span tensor.

Others Methods: DiffusionNER (Shen et al. 2023): formulates the NER task as a boundary-denoising diffusion process and thus generates named entities from noisy spans.

Table 1: Statistics of the datasets (in words).   

<html><body><table><tr><td></td><td>Scholar-XL</td><td>SciREX</td><td>Profiling-07</td></tr><tr><td>Input avg. len.</td><td>433.42</td><td>5678.29</td><td>785.09</td></tr><tr><td>Input max.len.</td><td>692</td><td>13731</td><td>17382</td></tr><tr><td>Input num.</td><td>2099</td><td>438</td><td>1446</td></tr><tr><td>Entity num.</td><td>20994</td><td>156931</td><td>17416</td></tr><tr><td>Entity type</td><td>12</td><td>4</td><td>13</td></tr><tr><td>Entity avg. len.</td><td>12.45</td><td>2.28</td><td>8.88</td></tr><tr><td>Entity max. len.</td><td>480</td><td>18</td><td>307</td></tr></table></body></html>

Generation-based Methods: UIE (Lu et al. 2022): uniformly encodes different extraction structures via a structured extraction language, adaptively generates target extractions, and captures the common IE abilities via a large-scale pre-trained text-to-structure model. InstructUIE (Wang et al. 2023b): leverages natural language instructions and instruction tuning to guide large language models for IE tasks. GOLLIE (Sainz et al. 2023): is based on Code-Llama (Roziere et al. 2023) and fine-tunes the foundation model to adhere to specific annotation guidelines. ADELIE (Qi et al. 2024): builds a high-quality instruction tuning dataset and utilizes supervised fine-tuning (SFT) followed by direct preference optimization (DPO). ToNER (Jiang et al. 2024): firstly employs an entity type matching model to discover the entity types that are most likely to appear in the sentence, and then adds multiple binary classification tasks to fine-tune the encoder in the generative model. GPT-4o (Achiam et al. 2023): employs the gpt-4o-2024-08-06 API, utilizing a 5-shot in-context learning approach to enhance performance. Claude-3.5 (Anthropic 2024): uses the claude-3-5-sonnet-20241022 API, also adopting 5-shot in-context learning.

# Experimental Setup

All experiments are conducted on an 8-card 80G Nvidia A100 server. The entire text is used for the Scholar-XL dataset, while the other two datasets are truncated to 5120 using a sliding window approach, as a trade-off due to limited GPU memory. For prediction, the prediction of the text segment is mapped to the starting/ending position of the original text. Hyper-parameters are selected based on the F1 score on the validation set. For each experiment, we run 3 times with different random seeds and report the average results. We choose DeBERTa-V3-large (He, Gao, and Chen 2023) as the PLM for span-based methods and DiffusionNER. We use AdamW (Loshchilov, Hutter et al. 2017) optimizer with a weight decay of $1 e ^ { - 2 }$ . The unilateral window sizes of the arrow attention and BiSPA mechanism are both set to 128. We only use low-rank adaptation on the $Q$ and $V$ matrix of the self-attention mechanism with a rank of 8.

# Evaluation Metrics

We report the micro-F1 score for all attributes. An entity is considered correct only if both the entity type and the entity span are predicted correctly. Precision (P) is the portion of correctly predicted spans over predicted spans, while Recall (R) is the portion of correctly predicted spans over groundtruth entity spans.

# Main Results

Table 2 provides a holistic comparison of different NER methods on three datasets. Generally speaking, span-based methods (CNN-NER, UTC-IE, and our model SeNER) outperform other types of NER methods.

Generation-based methods utilize generation loss to finetune the foundation model to adapt to the long NER task, achieving unfavorable performance. UIE outperforms InstructUIE, possibly because UIE defines a structured extraction language that suits the long NER problem better than naively performing instruction tuning. GOLLIE and ADELIE achieve similar performance, except for Profiling07 dataset, which is due to the fact that this dataset is sliced and diced with a high number of empty data and thus makes GOLLIE overfit these empty examples. ToNER obtains unsatisfactory performance, possibly since the twostage framework leads to error propagation and the usage of small language models for generation limits its potential. GPT-4o and Claude-3.5-sonnet are less effective, suggesting that the proprietary model prompting does not perform the long text NER task well.

The span-based NER methods (CNN-NER, UTC-IE, and SeNER) outperform other types of NER methods, including Diffusion-NER. DiffusionNER is a diffusion-based method that recovers the boundaries of the entities from a fixed amount of Gaussian noise and it is hard to recover longish entities from long texts. CNN-NER models fine-grained span interactions via CNN, achieving decent extraction performance. UTC-IE further improves CNN-NER by introducing plus-shaped attention on the token-pair span tensor, achieving consistent outperformance over CNN-NER.

Our model SeNER exhibits noticeable improvements or is on par with the best baseline, suggesting that with the design of arrow attention coupled with LogN-Scaling on the [CLS] in the PLM encoder, as well as the BiSPA mechanism on the token-pair span tensor, our model is capable of saving computation and memory resources without degrading the extraction accuracy. In addition, longer text with more focused attention can effectively help the model understand the semantic information of the text in more detail and extract the corresponding entities.

# Ablation Study

Table 3 presents a justification for the effectiveness of each component in our model. Removing either the arrow attention or BiSPA mechanism results in a decrease in model performance on Scholar-XL and Out-of-Memory (OOM) errors on SciREX and Profiling-07. It indicates that both modules effectively reduce explicit memory usage, enabling the model to handle longer texts and thereby improving overall performance. Specifically, BiSPA significantly reduces compute and memory footprint by reducing negative samples. In contrast, the arrow attention has a limited ability to reduce memory usage for short text on the Scholar-XL dataset. Substituting the arrow attention with sliding window attention (SWA) leads to a significant performance drop, highlighting

Gender Work Exp. 83.96 Highest Edu. 20.86 80.03 Education Exp. 51.48 14.78 81.74 93.26Birthday 38.65 8.7 76.10 76.19 90 25.83 86.74 Social 35.3 21.78 N/A 46.12 49.8 53.48 Position Service 11.08 71.26 19.66 27.5134.05 41 74.09 HoTniotlreary 28.24 32.22 43.11 76.92Birth Place 48.17 36.93 45.03 Interest Institution 62.29 Award CNN-NER UTC-IE InstructUIE GOLLIE ADELIE SeNER

the necessity of imposing attention scores on the [CLS] token to absorb global contextual information. Adding LogNScaling consistently improves the performance, thereby enhancing model stability and robustness. Although removing LoRA does not cause OOM errors, the F1 score decreases across all datasets to some extent, demonstrating that LoRA can effectively reduce training parameters and prevent overfitting. Whole Word Masking (WWM) increases the diversity of input texts, thus improving the generalization capacity of the model.

# Detailed Analysis for Entity Types

In this subsection, we focus on comparing the performance of our method with span-based methods (CNN-NER and UTC-IE) and LLM-based methods (InstructUIE, GOLLIE, and ADELIE) across entities of varying lengths and types.

The results on the Scholar-XL dataset are depicted in Figure 6, with the average length of the entity types increasing clockwise from “Gender” to “Work Experience”. Generative methods, leveraging the powerful capabilities of LLMs, achieve superior performance in extracting “Gender” and “Birth Place” types. However, for other types of entities, span-based methods demonstrate consistent superiority. Our model SeNER outperforms CNN-NER and UTC-IE in most types of entities, with particularly notable improvements for longish entities. Specifically, for “Social Service”, our method achieves an improvement of $6 . 3 8 \%$ over CNN-NER and $4 . 1 4 \%$ over UTC-IE, respectively. The performance of SeNER for entity types “Education Experience” and “Work Experience” falls behind the leading ones a little, indicating that the approximation strategy in our model inevitably loses some information, especially on very long entities.

Table 2: Main results on three long NER datasets $( \% )$ . The best results are boldfaced and the second best results are underlined   

<html><body><table><tr><td rowspan="2">Method type</td><td rowspan="2">Method</td><td colspan="3">Scholar-XL</td><td colspan="3">SciREX</td><td colspan="3">Profiling-07</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td rowspan="7">Generation-based Methods</td><td>UIE ToNER</td><td>43.32</td><td>36.80</td><td>39.80</td><td>65.88</td><td>56.44</td><td>60.80</td><td>65.92</td><td>57.51</td><td>61.43</td></tr><tr><td></td><td>40.08</td><td>29.48</td><td>33.97</td><td>57.43</td><td>31.56</td><td>40.73</td><td>48.80</td><td>41.21</td><td>44.68</td></tr><tr><td>InstructUIE</td><td>34.63</td><td>36.50</td><td>35.54</td><td>56.31</td><td>54.60</td><td>55.44</td><td>59.09</td><td>63.19</td><td>61.07</td></tr><tr><td>GOLLIE</td><td>43.74</td><td>40.88</td><td>42.26</td><td>71.56</td><td>71.50</td><td>71.53</td><td>64.51</td><td>9.46</td><td>16.50</td></tr><tr><td>ADELIE</td><td>45.60</td><td>39.05</td><td>42.07</td><td>70.10</td><td>71.84</td><td>70.96</td><td>65.75</td><td>49.53</td><td>56.50</td></tr><tr><td>Claude-3.5</td><td>17.15</td><td>30.16</td><td>21.87</td><td>57.78</td><td>7.97</td><td>14.01</td><td>34.40</td><td>43.07</td><td>38.25</td></tr><tr><td>GPT-40</td><td>18.24</td><td>27.31</td><td>21.87</td><td>40.44</td><td>7.69</td><td>12.92</td><td>36.96</td><td>43.73</td><td>40.06</td></tr><tr><td rowspan="2">Others Methods Span-based</td><td>DiffusionNER</td><td>55.33</td><td>29.87</td><td>38.80</td><td>77.11</td><td>62.36</td><td>68.96</td><td>70.51</td><td>44.16</td><td>54.31</td></tr><tr><td>CNN-NER</td><td>50.92</td><td>44.72</td><td>47.59</td><td>72.13</td><td>74.56</td><td>73.32</td><td>69.19</td><td>62.79</td><td>65.56</td></tr><tr><td rowspan="2">Methods</td><td>UTC-IE</td><td>53.17</td><td>46.01</td><td>49.10</td><td>71.90</td><td>75.09</td><td>73.42</td><td>69.79</td><td>65.28</td><td>67.43</td></tr><tr><td>SeNER (Ours)</td><td>57.41</td><td>46.80</td><td>51.56</td><td>72.89</td><td>76.17</td><td>74.49</td><td>67.52</td><td>67.17</td><td>67.34</td></tr></table></body></html>

Table 3: Ablation studies on three long NER datasets. Mem means memory usage (GB), SWA denotes sliding window attention and WWM is whole word masking.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="2">Scholar-XL</td><td colspan="2">SciREX</td><td colspan="2">Profiling-07</td></tr><tr><td>F1</td><td>Mem</td><td>F1</td><td>Mem</td><td>F1</td><td>Mem</td></tr><tr><td>SeNER w/o Arrow w/o LogN w SWA</td><td>51.56 51.34 50.78</td><td>16.95 16.94 17.18</td><td>74.49</td><td>69.23 OOM</td><td>67.34 67.11</td><td>63.36 OOM</td></tr></table></body></html>

# Analysis for Maximum Input Length

We examine the maximum input length supported by training each NER method on a single Nvidia A100 with a batch size of 1, as shown in Figure 7. Generation-based methods often employ various lightweight strategies, such as quantization, FlashAttention (Dao et al. 2022), and Zero Redundancy Optimizer (ZeRO) (Rajbhandari et al. 2020) enabling models like GOLLIE and ADELIE to handle long texts. In contrast, span-based methods need to model token-pair span tensor, resulting in supporting shorter input length. Our method, SeNER, demonstrates substantial improvements over CNN-NER and UTC-IE, supporting input lengths that are 3 times and 6 times longer, respectively.

# Efficiency Performance for Inference Time

Figure 7 also displays the inference time of span-based methods and LLM-based methods. It can be observed that LLM-based methods lead to 10 times longer inference time than span-based methods. Our method SeNER achieves a similar inference time compared with CNN-NER, achieving significantly better extraction accuracy simultaneously. SeNER can save $2 0 \%$ inference time compared with UTCIE and encode longer input texts, still maintaining state-ofthe-art extraction accuracy.

![](images/43f024dd74dfc1ab1a25dacb1ce7c0ff7c085842f44aadb8f61e5723c94e9667.jpg)  
Figure 7: Blue bar: Maximum input length comparison of different methods (k). Orange bar: Inference time comparison of different methods (second). Both are conducted on the longest SciREX dataset.

# Conclusion

In this paper, we tackle the problem of extracting entities from long texts, a less explored area in Named Entity Recognition (NER). Current span-based and generationbased NER methods face issues such as computational inefficiency and memory overhead in span enumeration, along with inaccuracy and time costs in text generation. To address these challenges, we introduce SeNER, a lightweight spanbased approach that featuring a bidirectional arrow attention mechanism and LogN-Scaling for effective long-text embedding. Additionally, we propose a bidirectional slidingwindow plus-shaped attention (BiSPA) mechanism that significantly reduces redundant candidate token-pair spans and models their interactions. Extensive experiments show that SeNER achieves state-of-the-art accuracy in extracting entities from long texts across three NER datasets, while maintaining GPU-memory efficiency. Our innovations in arrow attention and the BiSPA mechanism have the potential to advance future research in information extraction tasks.