# Multi-Level Optimal Transport for Universal Cross-Tokenizer Knowledge Distillation on Language Models

Xiao $\mathbf { C u i ^ { \mathrm { 1 * } } }$ , Mo Zhu2\*, Yulei $\mathbf { Q } \mathbf { i n } ^ { 3 }$ , Liang $\mathbf { X _ { i e } } ^ { 2 , 4 }$ , Wengang Zhou1, Houqiang Li1†

1University of Science and Technology of China 2Zhejiang University 3Tencent YouTu Lab 4Zhejiang University of Technology cuixiao $@$ mail.ustc.edu.cn, $\{ \mathrm { z h w g } , \mathrm { l i h q } \} @ \cdot$ ustc.edu.cn, mozhu $@$ zju.edu.cn, yuleiqin@tencent.com, lilydedbb@gmail.com

# Abstract

Knowledge distillation (KD) has become a prevalent technique for compressing large language models (LLMs). Existing KD methods are constrained by the need for identical tokenizers (i.e., vocabularies) between teacher and student models, limiting their versatility in handling LLMs of different architecture families. In this paper, we introduce the Multi-Level Optimal Transport (MultiLevelOT), a novel approach that advances the optimal transport for universal cross-tokenizer knowledge distillation. Our method aligns the logit distributions of the teacher and the student at both token and sequence levels using diverse cost matrices, eliminating the need for dimensional or token-by-token correspondence. At the token level, MultiLevelOT integrates both global and local information by jointly optimizing all tokens within a sequence to enhance robustness. At the sequence level, we efficiently capture complex distribution structures of logits via the Sinkhorn distance, which approximates the Wasserstein distance for divergence measures. Extensive experiments on tasks such as extractive QA, generative QA, and summarization demonstrate that the MultiLevelOT outperforms state-ofthe-art cross-tokenizer KD methods under various settings. Our approach is robust to different student and teacher models across model families, architectures, and parameter sizes.

# Introduction

Large language models (LLMs) such as LLaMA (Touvron et al. 2023a,b; Meta 2024), Mistral (Jiang et al. 2023) and Qwen (Bai et al. 2023; Yang et al. 2024) have set stateof-the-art (SOTA) records on various natural language processing (NLP) tasks. While the scaling laws of LLMs have driven the development of larger models with billions of parameters, their substantial sizes pose significant challenges to deployment under resource-constrained environments. To address this issue, knowledge distillation (KD) has emerged as a cost-efficient technique for its ability to distill smaller models that maintain competitive performance.

Cross-tokenizer knowledge distillation (CTKD) refers to the process of transferring knowledge between models that use different tokenizers (see Figure 1). It is crucial to ensure compatibility for applications such as multi-teacher knowledge transfer, where the student model learns from multiple teacher models with potentially different tokenization schemes. However, most existing KD methods rely on divergence measures such as Kullback–Leibler (KL) divergence (Hinton, Vinyals, and Dean 2015; Park, Kim, and Yang 2021; Agarwal et al. 2024; Wu et al. 2024), reverse KL (RKL) divergence (Tu et al. 2020; Gu et al. 2023b), and Jensen–Shannon (JS) divergence (Wen et al. 2023; Yin et al. 2020; Fang et al. 2021). These measures require a strict point-by-point correspondence across dimensions between the student and teacher, necessitating the use of the same tokenizer and consistent vocabularies, which limits their applicability when different tokenizers are involved.

![](images/44ee92b70da847e41070b8414c2922eeda0659c551e3140a5493b19683a36647.jpg)  
Figure 1: An illustration of vocabulary mismatch resulting from cross-tokenizer discrepancies. Unlike strict token-wise distillation methods that may lead to token misalignment, we employ sequence-level and sequence-aware token-level optimal transport to facilitate effective knowledge transfer.

Very few studies notice such deficiency in directly applying existing KD techniques on LLMs, for the simple reason that most KD methods are developed for few mainstream open-source models. ULD (Boizard et al. 2024), the first attempt ever to tackle this issue, aligns the distributions of individual tokens between the teacher and the student using token-wise optimal transport (OT). However, ULD focuses solely on the internal information of individual tokens without considering the global context for robust matching. Additionally, its reliance on zero padding introduces noise and hinders the effective use of logarithmic cost matrices. DSKD (Zhang et al. 2024b), another token-wise alignment method, tries to transform the hidden states of one model to the space of another one bidirectionally via learnable projectors. Despite its efforts in alignment for a unified output space, DSKD fails to effectively leverage the distribution information as the transformed distribution often exhibits low accuracy. Also, although these methods avoid strict dimensional correspondence, they assume a rigid token-by-token correspondence, which is often not the case in practice.

To address these shortcomings, we propose the MultiLevel Optimal Transport (MultiLevelOT) for crosstokenizer knowledge distillation on LLMs. Our method comprehensively measures the discrepancy between teacher and student logit distributions by calculating the optimal transport distance both within and across tokens in each sequence. Such a dual-level approach ensures that both token-level and sequence-level relationships are incorporated into the distillation process, effectively eliminating the need for dimensional or token-by-token correspondence.

At the token level, we jointly optimize all tokens within a sequence by minimizing token-level discrepancies within the context of the entire sequence. This is achieved by applying a sequence-level ranking process, which enables the same optimal transport plan for all tokens and effectively selects the important dimensions. To eliminate noise from redundant dimensions, we truncate the logits, focusing only on the most impactful logit dimensions for each sequence. This truncation ensures that the teacher and student logits share a common support size, making each dimension meaningful and applicable for a logarithmic form cost matrix. To capture both the fine-grained, token-wise nuances and the holistic, sequence-scale context view, we employ two types of cost matrices: one in the form of absolute difference and the other in the form of logarithm-based likelihood difference. The absolute difference cost matrix captures the direct discrepancies in logits, providing a straightforward and interpretable measure of distance. Conversely, the logarithmic cost matrix accounts for the relative differences to offer a more nuanced and scalable measure. It is particularly effective in handling logits with a wide range of magnitudes.

At the sequence level, which has not been considered in previous studies, we utilize token-to-token OT distances to construct the sequence-level cost matrix. Since optimal transport automatically finds the corresponding relationships between tokens, this is particularly crucial for addressing token order misalignment caused by varying tokenization of long words across different tokenizers. Unlike token-level transport, which deals with individual logit values, sequence-level transport requires calculating the optimal transport between vectors of tokens. Given the computational intensity of directly computing the Wasserstein distance for this purpose, we employ the Sinkhorn distance as an efficient approximation. This approach retains the benefits of the Wasserstein distance while significantly reducing computational complexity. Importantly, we achieve all the improvements without introducing additional modules or modifying output formats specific to NLP tasks.

Extensive experiments are conducted in view of 1) comparability, 2) validity, and 3) generalizability. For comparability, we test our method on different tasks under both labeled and unlabeled distillation settings. Our method consistently outperforms the state-of-the-art CTKD methods. For validity, we provide a comprehensive analysis through ablation studies and hyper-parameter tuning, which corroborate the effectiveness of each component. For generalizability, the proposed method is validated on different students across families, architectures, and sizes. We also experiment with diverse teachers to demonstrate its robustness across various model choices. In summary, our contributions are:

• We propose the MultiLevelOT, a cross-tokenizer knowledge distillation approach that leverages both sequenceaware token-level and sequence-level optimal transport for comprehensive distribution matching. • We enhance the robustness of our method by jointly optimizing all tokens and using varied cost matrices, effectively capturing both global and local information. • We demonstrate the superiority of MultiLevelOT over existing methods through extensive experiments, validating its comparability, validity, and generalizability.

# Related Work

# Knowledge Distillation

Knowledge distillation (KD) is proposed to transfer the intrinsic knowledge from a teacher model to a student model by approximating the teacher’s soft targets, such as output logits and intermediate representations. Cross-Tokenizer KD extends this traditional framework to scenarios involving different tokenizers, each with distinct vocabularies, which is crucial for LLM distillation. Various KD methods have been explored, ranging from logit-based distillation to representation-based distillation. These methods typically employ divergence measures like KL divergence (Hinton, Vinyals, and Dean 2015; Agarwal et al. 2024; Wu et al. 2024; Zhou, Xu, and McAuley 2022; Zhang et al. 2023; Liu et al. 2022), RKL (Tu et al. 2020; Gu et al. 2023b; Ko et al. 2024), and JS divergence (Wen et al. 2023; Yin et al. 2020; Fang et al. 2021). These measures compute discrepancies on each dimension, requiring a one-to-one correspondence between teacher and student logit dimensions. SinKD (Cui et al. 2024b,a) addresses the limitations of these traditional measures by using the Sinkhorn distance. However, its approach still requires dimensional correspondence in the cost matrix. In cross-tokenizer distillation, such dimensional correspondence is absent, making these methods inapplicable.

To overcome this challenge, both ULD (Boizard et al. 2024) and DSKD (Zhang et al. 2024b) propose promising solutions for token-wise alignment. ULD measures tokenwise OT distance between the logits of the student and teacher models, eliminating the dependency on dimensional correspondence. DSKD attempts to transform the hidden states of one model to that of another by training projectors, but the transformed distribution often exhibits low accuracy. Comparatively, the proposed method differs in the following aspects: 1) ULD only considers local information while neglecting global distributional properties. Its padding approach, more like an ad-hoc brutal tactic, limits it to a singular cost matrix. In contrast, we stem from the tokenlevel and sequence-level perspectives and deduce different forms of cost matrices for lexical and semantic alignment. 2) While DSKD relies on traditional divergence measures, which suffer from issues like mode-averaging and modecollapsing (Cui et al. 2024b), we employ the Sinkhorn distance to fully capture the geometric characteristics of logit distributions. In addition, we do not explicitly enforce crossmodel space mapping because such dual-space projection lacks semantic interpretability and thereafter hinders sequence comprehension. 3) Both ULD and DSKD assume a rigid token-by-token correspondence, which is often impractical. Our approach uses sequence-level OT, which automatically identifies corresponding relationships between tokens, thereby eliminating the need for strict token correspondence.

![](images/571f60405d551a13b7cb7b364aa3b7f2f6540549eee846466bf314e094d91942.jpg)  
Figure 2: Illustration of our pipeline. MultiLevelOT computes sequence-aware token-level and sequence-level optimal transport distances between the output logits of the teacher and student models. This approach effectively transfers local and global information within the logits distribution, accommodating vocabulary differences and enabling cross-tokenizer distillation.

# Optimal Transport

Optimal transport (OT) theory offers a robust mathematical framework for comparing probability distributions by calculating the minimal cost required to transform one distribution into another. The Wasserstein distance, a pivotal concept in OT, quantifies this cost and excels in capturing the geometric structure of distributions (Villani and Villani 2009; Zhang, Liu, and Tao 2021). This metric has been instrumental across various domains, including causal discovery (Wei et al. 2022; Weilin et al. 2023), image generation (Arjovsky, Chintala, and Bottou 2017; Gulrajani et al. 2017; Peyre´, Cuturi et al. 2019), unsupervised learning (Gu et al. $2 0 2 3 \mathrm { a }$ ; Chen et al. 2022; He et al. 2022), and reinforcement learning (Du et al. 2023; Lan et al. 2023; Zhang et al. 2024a).

While the Wasserstein distance may be simplified in some low-dimensional cases, it can be computationally intensive in other scenarios. To address this, the Sinkhorn distance has been proposed as an approximation, which introduces an entropy regularization term to the OT problem, making it more tractable (Cuturi 2013). This approach has demonstrated success in diverse applications such as machine translation (Li, Unanue, and Piccardi 2023), domain adaptation (Nguyen and Luu 2022; Xu et al. 2023), classification (Liu et al. 2023), and teacher model selection (Lu,

Ye, and Zhan 2022; Bhardwaj, Vaidya, and Poria 2021).

Our approach employs both token-level and sequencelevel OT for cross-tokenizer knowledge distillation. This dual-level OT captures global and local information, enhancing geometry information transfer and model efficacy.

# Methods

# Problem Statement

Given a sample $\mathbf { x }$ and its ground-truth label $\mathbf { y }$ , the output logits with softmax activation $\sigma _ { \tau }$ from the teacher $f _ { \mathbf { T } }$ and the student $f _ { \mathbf { S } }$ are $\mathbf { t } \in \mathbb { R } ^ { T \times m }$ and $\mathbf { s } \in \mathbb { R } ^ { T \times n }$ , respectively:

$$
\mathbf { t } = \sigma _ { \tau } ( f _ { \mathbf { T } } ( \mathbf { x } ) ) , \quad \mathbf { s } = \sigma _ { \tau } ( f _ { \mathbf { S } } ( \mathbf { x } ) ) ,
$$

where $\tau$ represents the temperature parameter, $m$ and $n$ denote the dimensions of the teacher and student output vocabularies, respectively, and $T$ is the total number of tokens in the generated sequence. We denote the $i$ -th dimension of the teacher and student logits for the $t$ -th token as $\mathbf { t } _ { i } ( t )$ and ${ \bf s } _ { i } ( t )$ , respectively. Our objective is to minimize the optimal transport distance between the distributions of the teacher’s and student’s outputs for knowledge transfer. In scenarios where the ground-truth label is unavailable, we use teachergenerated text as a substitute.

# Reconstructing optimal transport in ULD

ULD (Boizard et al. 2024) leverages OT to address the challenge of cross-tokenizer knowledge distillation. To ensure equal support size between the teacher and student distribution spaces, ULD pads the smaller vocabulary with zero values, matching the larger size $\mathrm { m a x } ( m , n )$ . The ULD loss is then computed by summing the token-wise Wasserstein distances. The OT distance for the $t$ -th token is defined as:

$$
\operatorname* { m i n } _ { \mathbf { P } ( t ) } \sum _ { i = 1 } ^ { \operatorname* { m a x } ( m , n ) } \sum _ { j = 1 } ^ { \operatorname* { m a x } ( m , n ) } { \mathbf { P } } _ { i j } ( t ) { \mathbf { C } } _ { i j } ( t ) ,
$$

where $\mathbf { P }$ is the optimal transport matrix and $\mathbf { C }$ is the cost matrix. ULD asserts that each transport cost is equal to 1 and applies the following constraints on $\mathbf { P }$ :

$$
\sum _ { i } \mathbf { P } _ { i j } ( t ) = \mathbf { s } _ { j } ( t ) \quad \forall j , t , \sum _ { j } \mathbf { P } _ { i j } ( t ) = \mathbf { t } _ { i } ( t ) \quad \forall i , t .
$$

However, the original formulation lacks flexibility. We propose a more adaptable reformulation by setting $\begin{array} { l l } { \mathbf { C } _ { i j } } & { = } \end{array}$ $\bar { \vert } \mathbf { t } _ { i } ( t ) - \mathbf { s } _ { j } ( t ) \vert$ and using these constraints:

$$
\sum _ { i } \mathbf { P } _ { i j } ( t ) = 1 \quad \forall j , t , \quad \quad \sum _ { j } \mathbf { P } _ { i j } ( t ) = 1 \quad \forall i , t .
$$

Both formulations yield the same optimal transport distance:

$$
\mathcal { L } _ { \mathrm { U L D } } = \sum _ { t = 1 } ^ { T } \sum _ { i = 1 } ^ { \operatorname* { m a x } ( m , n ) } \left| \mathbf { t } _ { \mathrm { T R } , i } ( t ) - \mathbf { s } _ { \mathrm { T R } , i } ( t ) \right| ,
$$

where ${ \bf s } _ { \mathrm { T R } } ( t )$ and ${ \bf t } _ { \mathrm { T R } } ( t )$ are the token-wise ranked logits of the student and teacher, respectively:

$$
\begin{array} { r l } & { { \bf s } _ { \mathrm { T R } } ( t ) = { \bf s } \left[ \operatorname { a r g s o r t } \left( { \bf s } ( t ) , \mathrm { d e s c e n d i n g } \right) \right] } \\ & { { \bf t } _ { \mathrm { T R } } ( t ) = { \bf t } \left[ \operatorname { a r g s o r t } \left( { \bf t } ( t ) , \mathrm { d e s c e n d i n g } \right) \right] . } \end{array}
$$

By reconstructing equivalent optimal transport problems, we can design various cost matrices and extend token-wise optimal transport distance to multi-level optimal transport.

# Multi-Level Optimal Transport

Instead of considering each token independently, our method jointly optimizes all tokens within a sequence through sequence-aware multi-level OT, effectively aligning the distributions of teacher and student output logits. The primary objective is to minimize the sum of token-level and sequence-level costs using an optimal transport plan $\mathbf { P }$ :

$$
\underset { \mathbf { P } } { \operatorname* { m i n } } \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { n } \mathbf { P } _ { i j } \sum _ { t = 1 } ^ { T } \mathbf { C } _ { i j } ^ { t o k } ( t ) + \underset { \mathbf { P } } { \operatorname* { m i n } } \sum _ { i = 1 } ^ { T } \sum _ { j = 1 } ^ { T } \mathbf { P } _ { i j } \mathbf { C } _ { i j } ^ { s e q } ,
$$

where $\mathbf { C } ^ { t o k }$ and ${ \bf C } ^ { s e q }$ represent the token-level and sequence-level cost matrices, respectively. Specific mathematical formulations will be detailed in subsequent paragraphs. The optimization is subject to the constraints:

$$
\sum _ { i } \mathbf { P } _ { i j } = 1 \quad \forall j , \qquad \sum _ { j } \mathbf { P } _ { i j } = 1 \quad \forall i .
$$

We model the token-level cost using both absolute difference and logarithmic forms, while the sequence-level cost is captured through the optimal transport distance between tokens. For token-level alignment, our optimization strategy integrates both global and local information by considering the entire sequence within the optimal transport process. The full pipeline is illustrated in Figure 2.

Holistic Absolute Difference Loss We define the first token-level cost matrix $\mathbf { C } _ { i j } ^ { t o k } ( t )$ using the absolute difference between logits: $\mathbf { C } _ { i j } ^ { t o k } ( t ) = | \mathbf { t } _ { i } ( t ) - \mathbf { s } _ { j } ( t ) |$ , so that the

Wasserstein distance can be obtained by solving this optimization problem:

$$
\operatorname* { m i n } _ { \mathbf { P } } \sum _ { t = 1 } ^ { T } \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { n } \mathbf { P } _ { i j } \left| \mathbf { t } _ { i } ( t ) - \mathbf { s } _ { j } ( t ) \right| .
$$

While ULD employs a separate optimal transport matrix for each token, leading to inconsistent dimensional relationship, our approach ensures robustness by performing sequencelevel ranking across all logits within a sequence. This allows us to use a single optimal transport matrix for all tokens, ensuring consistent dimensional ordering within each token $t$ . Our sequence-level ranking process is defined as follow:

$$
{ \mathbf t } _ { \mathrm { S R } } = { \mathbf t } \left[ \mathrm { a r g s o r t } \left( \sum _ { t = 1 } ^ { T } { \mathbf t } ( t ) , \mathrm { d e s c e n d i n g } \right) \right] , \quad { \mathbf s } _ { \mathrm { S R } } = { \mathbf Q } { \mathbf s } ,
$$

where $\mathbf { Q } = \mathbf { Q } ^ { * }$ is a permutation matrix used to match the dimensions of s with the corresponding dimensions of $\mathbf { t } _ { \mathrm { S R } }$ at the sequence level, satisfying:

$$
{ \bf Q } ^ { * } = \underset { { \bf Q } } { \operatorname { \arg m i n } } \sum _ { t = 1 } ^ { T } \sum _ { i = 1 } ^ { m } \left| { \bf t } _ { \mathrm { S R } , i } ( t ) - [ { \bf Q } { \bf s } ( t ) ] _ { i } \right| .
$$

To ensure the consistency of the support size, allow for a logarithmic cost matrix, prevent mode-averaging, and reduce noise from unlikely words, we conduct the top- $\mathbf { \nabla } \cdot \mathbf { k }$ truncation as follows:

$$
\begin{array} { r } { \mathbf { s } _ { \mathrm { { S R } , \mathrm { T r } } } ( t ) = \mathbf { s } _ { \mathrm { { S R } } } ( t ) [ : k ] , \quad \mathbf { t } _ { \mathrm { { S R } , \mathrm { T r } } } ( t ) = \mathbf { t } _ { \mathrm { { S R } } } ( t ) [ : k ] , } \end{array}
$$

where $[ \colon k ]$ denotes the slicing operation for choosing the top- $\mathbf { \nabla } \cdot \mathbf { k }$ elements of the vector. Then the optimization problem can be reformulated as:

$$
\operatorname* { m i n } _ { \mathbf { P } } \sum _ { t = 1 } ^ { T } \sum _ { i = 1 } ^ { k } \sum _ { j = 1 } ^ { k } \mathbf { P } _ { i j } \left| \mathbf { t } _ { \mathrm { S R } , \mathrm { T r } , i } ( t ) - \mathbf { s } _ { \mathrm { S R } , \mathrm { T r } , j } ( t ) \right| .
$$

The optimal transport matrix to the above Eq. (13) is $\mathbf { P ^ { * } } =$ $\mathbf { P } ^ { \mathrm { H A D } }$ , where $\mathbf { P } _ { i j } ^ { \mathrm { H A D } }$ is 1 only when $i = j$ , and 0 otherwise. The absolute difference loss, representing the solution to this optimization problem, is then computed as:

$$
\mathcal { L } _ { \mathrm { H A D } } = \sum _ { t = 1 } ^ { T } \sum _ { i = 1 } ^ { k } \left| \mathbf { t } _ { \mathrm { S R } , \mathrm { T r } , i } ( t ) - \mathbf { s } _ { \mathrm { S R } , \mathrm { T r } , i } ( t ) \right| .
$$

In the following text, all instances of $\mathbf { t } ^ { k }$ and $\mathbf { s } ^ { k }$ refer to $\mathbf { t } _ { \mathrm { S R , T r } }$ and $\mathbf { s } _ { \mathrm { S R , T r } }$ , respectively.

Sequential Logarithmic Loss For the token-level cost matrix, in addition to the absolute difference, we also incorporate a logarithmic form: ${ \bf C } _ { i j } ^ { t o k } ( t ) = - { \bf t } _ { i } ( t ) \log { { \bf s } _ { j } ( t ) }$ . We apply the previously mentioned top- $\mathbf { \nabla } \cdot \mathbf { k }$ truncation, which ensures that no zero-value elements are present in the student logits, thus making this cost matrix meaningful and effective. Given that each dimension is equally important, the optimization problem for computing the Wasserstein distance can be formulated in a sequence-level ranked order:

$$
\operatorname* { m i n } _ { \mathbf { P } } \sum _ { t = 1 } ^ { T } \sum _ { i = 1 } ^ { k } \sum _ { j = 1 } ^ { k } - \mathbf { P } _ { i j } \mathbf { t } _ { i } ^ { k } ( t ) \log \mathbf { s } _ { i } ^ { k } ( t ) .
$$

The optimization objective is minimized by the sequential transfer between logit dimensions, making the optimal transport matrix $\mathbf { P } ^ { \mathrm { S L } }$ equivalent to $\mathbf { P } ^ { \mathrm { H A D } }$ . Consequently, the loss function is defined as:

$$
\mathcal { L } _ { \mathrm { { S L } } } = - \sum _ { t = 1 } ^ { T } \sum _ { i = 1 } ^ { k } \mathbf { t } _ { i } ^ { k } ( t ) \log \mathbf { s } _ { i } ^ { k } ( t ) .
$$

Sinkhorn Distance Loss We employ the optimal transport distance between tokens to measure pairwise differences between the $i$ -th and $j$ -th tokens in a sequence, constructing the sequence-level cost matrix $\mathbf { C } \in \mathbf { \dot { \mathbb { R } } } ^ { T \times T }$ with entries $\begin{array} { r } { \mathbf { \bar { C } } _ { i j } ^ { s e q } = \sum _ { l = 1 } ^ { k } \sum _ { q = 1 } ^ { k } \mathbf { P } _ { l q } ^ { \mathrm { H A D } } \left| \mathbf { t } _ { l } ^ { k } ( i ) - \mathbf { s } _ { q } ^ { k } ( j ) \right| , } \end{array}$ . Following SinKD (Cu  et al. 2024b,a), we use Sinkhorn distance as an efficient approximation for Wasserstein distance, retaining its benefits while significantly reducing computational costs for online distillation. The Sinkhorn distance is based on the relaxed formulation of an OT plan with entropy regularization. The OT plan $\mathbf { P } ^ { \lambda }$ is obtained by minimizing:

$$
\mathbf { P } ^ { \lambda } = \underset { \mathbf { P } } { \operatorname { a r g m i n } } \sum _ { i = 1 } ^ { T } \sum _ { j = 1 } ^ { T } \mathbf { P } _ { i j } \mathbf { C } _ { i j } - \lambda h \left( \mathbf { P } \right) ,
$$

where $h ( \mathbf { P } )$ is the entropy of the matrix $\mathbf { P }$ , $\lambda > 0$ is the entropy regularization weight. To solve this iteratively, we construct the kernel matrix $\mathbf { K } ^ { 0 } \in \mathbb { R } ^ { T \times T }$ by applying the Gaussian kernel to $\mathbf { C }$ with the parameter $\lambda$ :

$$
\mathbf { K } ^ { 0 } = \exp \left( - { \frac { \mathbf { C } } { \lambda } } \right) .
$$

The OT plan $\mathbf { P } ^ { \lambda }$ is then derived through sequence-level Sinkhorn normalization, using iterative updates on $\mathbf { K }$ :

$$
\widehat { \mathbf { K } } ^ { i } \gets \mathbf { K } ^ { i - 1 } \oslash \left( \mathbf { K } ^ { i - 1 } \mathbf { 1 } _ { b } \mathbf { 1 } _ { b } ^ { \top } \right) , \mathbf { K } ^ { i } \gets \widehat { \mathbf { K } } ^ { i } \oslash \left( \mathbf { 1 } _ { b } \mathbf { 1 } _ { b } ^ { \top } \widehat { \mathbf { K } } ^ { i } \right) .
$$

For simplicity, irrelevant constants are excluded from the equations. After a pre-determined number of iterations $N$ , the OT matrix is obtained as $\mathbf { P } ^ { \lambda } = \mathbf { K } ^ { N }$ . The sequence-level optimal transport distance loss is then computed as:

$$
\mathcal { L } _ { \mathrm { S D } } = \left. \mathbf { P } ^ { \lambda } , \mathbf { C } \right. = \sum _ { i = 1 } ^ { T } \sum _ { j = 1 } ^ { T } \mathbf { K } _ { i , j } ^ { N } \mathbf { C } _ { i , j } .
$$

Total Loss We combine the Cross-Entropy (CE) loss with the weighted holistic absolute difference loss, sequential logarithmic loss, and Sinkhorn distance loss for distillation. For a sequence of $T$ tokens, the total loss is defined as:

$$
\mathcal { L } = \sum _ { t = 1 } ^ { T } \mathcal { L } _ { \mathrm { C E } } \left( \mathbf { y } ( t ) , \mathbf { s } ( t ) \right) + \alpha ( \mathcal { L } _ { \mathrm { H A D } } + \beta \mathcal { L } _ { \mathrm { S L } } + \gamma \mathcal { L } _ { \mathrm { S D } } ) ,
$$

where $\alpha , \beta$ and $\gamma$ are weights for each loss component.

# Experiments

# Experimental Settings

Datasets. We evaluate our method on three representative tasks: an extractive QA task (QED) (Lamm et al. 2021), a generative QA task (FairytaleQA) $\mathrm { \Delta X u }$ et al. 2022), and a summarization task (DIALOGSum) (Chen et al. 2021). For evaluation, we use the F1 score for the QED and the RougeLSum (Giarelis, Mastrokostas, and Karacapilidis 2023) for others. More details are given in the appendix.

Table 1: Performance of the students in labeled distillation. Both the teacher and ground-truth provide supervision.   

<html><body><table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td rowspan="2">QED (F1)</td><td rowspan="2">FairytaleQA</td><td rowspan="2">DIALOGSum</td></tr><tr><td>(Rouge-LSum) (Rouge-LSum)</td></tr><tr><td>LLaMA2-7B</td><td>Few-Shot 61.68</td><td></td><td>50.90</td><td>37.75</td></tr><tr><td rowspan="6">OPT-350M</td><td>Origin</td><td>12.46</td><td>11.16</td><td>14.02</td></tr><tr><td>SFT</td><td>55.71</td><td>46.04</td><td>35.59</td></tr><tr><td>SeqKD</td><td>49.61</td><td>39.19</td><td>30.71</td></tr><tr><td>MinED</td><td>56.03</td><td>46.11</td><td>35.82</td></tr><tr><td>ULD</td><td>56.76</td><td>45.82</td><td>36.05</td></tr><tr><td>Ours</td><td>58.97</td><td>46.96</td><td>37.61</td></tr><tr><td rowspan="6">Pythia-410M</td><td>Origin</td><td>22.87</td><td>15.14</td><td>4.41</td></tr><tr><td>SFT</td><td>59.03</td><td>47.23</td><td>36.06</td></tr><tr><td>SeqKD</td><td>51.12</td><td>39.78</td><td>31.57</td></tr><tr><td>MinED</td><td>59.21</td><td>47.31</td><td>35.97</td></tr><tr><td>ULD</td><td>59.71</td><td>47.81</td><td>36.07</td></tr><tr><td>Ours</td><td>61.79</td><td>49.10</td><td>37.45</td></tr><tr><td rowspan="6">Bloomz-560M</td><td>Origin</td><td>47.67</td><td>43.47</td><td>11.82</td></tr><tr><td>SFT</td><td>60.48</td><td>49.07</td><td>36.52</td></tr><tr><td>SeqKD</td><td>52.33</td><td>45.68</td><td>31.83</td></tr><tr><td>MinED</td><td>60.52</td><td>49.10</td><td>36.39</td></tr><tr><td>ULD</td><td>61.22</td><td>49.87</td><td>36.40</td></tr><tr><td>Ours</td><td>62.58</td><td>50.94</td><td>37.68</td></tr><tr><td rowspan="6">Average</td><td>Origin</td><td>27.67</td><td>23.25</td><td>10.08</td></tr><tr><td>SFT</td><td>58.41</td><td>47.45</td><td>36.05</td></tr><tr><td>SeqKD</td><td>50.99</td><td>41.55</td><td>31.37</td></tr><tr><td>MinED</td><td>58.58</td><td>47.47</td><td>36.06</td></tr><tr><td>ULD</td><td>59.30</td><td>47.83</td><td>36.17</td></tr><tr><td>Ours</td><td>60.99</td><td>49.00</td><td>37.58</td></tr></table></body></html>

Table 2: Performance of the students in unlabeled distillation. The ground-truth is unavailable for supervision.   

<html><body><table><tr><td>Model</td><td>Method QED (F1)</td><td>FairytaleQA (Rouge-LSum) (Rouge-LSum)</td><td>DIALOGSum</td></tr><tr><td rowspan="2">LLaMA2-7B</td><td>Few-Shot 61.68</td><td>50.90</td><td>37.75</td></tr><tr><td>Origin 12.46</td><td>11.16</td><td>14.02</td></tr><tr><td rowspan="3">OPT-350M</td><td>Raw Text 49.61</td><td>39.19</td><td>30.71</td></tr><tr><td>ULD</td><td>50.71 39.86</td><td>32.03</td></tr><tr><td>Ours</td><td>51.96 40.68</td><td>36.88</td></tr><tr><td rowspan="4">Pythia-410M</td><td>Origin</td><td>22.87</td><td>15.14 4.41</td></tr><tr><td>Raw Text 51.12</td><td></td><td>39.78 31.57</td></tr><tr><td>ULD</td><td>52.09</td><td>40.69 34.15</td></tr><tr><td>Ours</td><td>53.56 41.28</td><td>36.52</td></tr><tr><td rowspan="4">Bloomz-560M</td><td>Origin</td><td>47.67 43.47</td><td>11.82</td></tr><tr><td>Raw Text 52.33</td><td></td><td>45.68 31.83</td></tr><tr><td>ULD</td><td>53.02 46.72</td><td>34.21</td></tr><tr><td>Ours</td><td>54.15 47.88</td><td>37.10</td></tr><tr><td rowspan="4">Average</td><td>Origin</td><td>27.67 23.25</td><td>10.08</td></tr><tr><td></td><td></td><td>41.55</td></tr><tr><td>RawLText 50.99</td><td></td><td>31.37</td></tr><tr><td>Ours</td><td>53.22</td><td>43.28</td></tr></table></body></html>

Implementation details. We use four advanced teacher models: LLaMA2 7B Chat (Touvron et al. 2023b), Mistral3 7B Instruct (Jiang et al. 2023), Qwen 7B Chat (Bai et al. 2023) and LLaMA3 8B Instruct (Meta 2024). These models are chosen for their proficiency in few-shot learning and their unique vocabulary coverage (Brown et al. 2020). For student models, we use a range of LLMs from various families and architectures, including OPT 350M (Zhang et al. 2022), Pythia 160M, Pythia 410M, Pythia 1B (Biderman et al. 2023), Bloomz 560M (Muennighoff et al. 2023), and mT0 300M (Muennighoff et al. 2023), initializing them with their pretrained weights. Following ULD (Boizard et al. 2024), we set the learning rate $l r ~ = ~ 1 e ~ - ~ 6$ , $\alpha \ : = \ : 0 . 1 5$ , $\beta = 0 . 1$ . Additionally, we empirically set $\gamma = 0 . 1$ , $\tau _ { \mathrm { S L } } = 1$ , $\tau _ { \mathrm { S D } } = 2$ , $\lambda = 0 . 1$ , $N = 2 0$ and $k = 5 0$ . Discussions on the effects of key factors $N$ , and $k$ are presented later. Although further tuning may enhance performance, we maintain a consistent set of hyper-parameters across all tasks to underscore the robustness of our approach.

Baselines. Our experiments involve two settings: labeled distillation and unlabeled distillation. Labeled distillation, commonly used in most distillation studies, involves supervision with ground-truth labels. In contrast, unlabeled distillation relies solely on the generated texts from the teacher as pseudo-targets (Boizard et al. 2024). For labeled distillation, we compare our approach against the following baselines: Supervised Fine-Tuning (SFT), Sequence-level KD (SeqKD) (Kim and Rush 2016), MinED (Wan et al. 2024), and ULD (Boizard et al. 2024). SeqKD can be interpreted as a form of supervised fine-tuning using the teacher’s outputs, deriving knowledge exclusively from the teacher model. MinED, which can align the logits using dynamic programming , is also included in our comparison. For unlabeled distillation, we follow the ULD to adopt the same baselines. In both settings, we use the official code and default hyperparameters for each baseline to ensure a fair comparison. We exclude DSKD (Zhang et al. 2024b) from our comparison as it introduces additional modules whose increased learnable parameters may cause unfair comparison.

# Results and Discussions

Comparison with SOTA. Results on labeled distillation and unlabeled distillation are presented in Table 1 and Table 2, respectively. MultiLevelOT consistently outperforms all baseline methods across all datasets and student models.

Table 3: Ablation Study on QED across three students.   

<html><body><table><tr><td>CE</td><td>AD</td><td>TR</td><td>SR</td><td>Tr</td><td>SL</td><td>SD</td><td>OPT</td><td>Pythia</td><td>Bloomz</td></tr><tr><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td><td>55.71</td><td>59.03</td><td>60.48</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td><td>56.76</td><td>59.71</td><td>61.22</td></tr><tr><td>√</td><td>√</td><td></td><td>√</td><td></td><td></td><td></td><td>58.02</td><td>60.18</td><td>61.56</td></tr><tr><td>√</td><td>√</td><td></td><td>√</td><td>√</td><td></td><td></td><td>58.01</td><td>60.22</td><td>61.58</td></tr><tr><td>√</td><td>√</td><td></td><td>√</td><td>√</td><td>√</td><td></td><td>58.17</td><td>61.10</td><td>61.87</td></tr><tr><td>√</td><td>√</td><td></td><td>√</td><td>√</td><td></td><td><</td><td>58.15</td><td>61.20</td><td>61.90</td></tr><tr><td>√</td><td></td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>58.23</td><td>61.17</td><td>61.80</td></tr><tr><td>√</td><td>√</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>58.97</td><td>61.79</td><td>62.58</td></tr></table></body></html>

Table 4: Comparison of token-level and sequence-level Sinkhorn distance loss on QED across three students.   

<html><body><table><tr><td></td><td>OPT</td><td>Pythia</td><td>Bloomz</td></tr><tr><td>w/o SD loss</td><td>58.17</td><td>61.10</td><td>61.87</td></tr><tr><td>W token-level SD loss</td><td>58.32</td><td>61.22</td><td>61.95</td></tr><tr><td>w sequence-level SD loss</td><td>58.97</td><td>61.79</td><td>62.58</td></tr></table></body></html>

QED FairytaleQA 55.0 M 52.5   
0850.0 S47.5 Raw Text Raw Text 45.0   
L ULD ULD   
F42.5 40.0 Ours Ours 160M 410M 1B 160M 410M 1B Model Size Model Size

Notably, compared with ULD (Boizard et al. 2024), MultiLevelOT reduces the performance gap between the student and the teacher by over $71 \%$ in the QED task on labeled distillation. This improvement highlights the effectiveness of MultiLevelOT in bridging the performance gap by transferring sequence-level and sequence-aware token-level knowledge from the teacher to the student. The superior performance of our approach is also attributed to the well-rounded design of the cost matrix. By employing diverse cost matrices, we facilitate effective geometry distribution information extraction and enhance the knowledge transfer process.

Each components plays its role in MultiLevelOT. The ablation study on the QED task, as shown in Table 3, demonstrates the critical role of each component in the MultiLevelOT framework. The baseline model, utilizing only cross-entropy (CE) loss, corresponds to the standard SFT. Adding the absolute difference (AD) and token-wise ranking (TR), as in ULD, provides a reference for improvement. However, the key advancements come from our proposed components. Integrating sequence-level ranking (SR) and truncation (Tr) with AD results in the Holistic Absolute Difference Loss, which shows significant gains by capturing both global and local geometrical information. Incorporating the Sequential Logarithmic Loss (SL) further boosts performance, highlighting the value of various cost matrices in capturing different aspects of the distribution. Finally, integrating the Sinkhorn Distance Loss (SD) results in the best performance, underlining the necessity of sequencelevel knowledge for effective knowledge transfer.

<html><body><table><tr><td>Method</td><td>QED (F1)</td><td>FairtaleQA (Rouge-LSUM)</td><td>DIALOGSum (Rouge-LSUM)</td></tr><tr><td>Raw Labels</td><td>34.96</td><td>29.73</td><td>28.88</td></tr><tr><td>ULD</td><td>37.25</td><td>31.52</td><td>30.04</td></tr><tr><td>Ours</td><td>41.37</td><td>34.01</td><td>33.01</td></tr></table></body></html>

Table 5: Generalizability of MultiLevelOT in student architecture. Teacher: LLaMA, student: mT0-300M.

Table 6: Generalizability of MultiLevelOT across different teacher models on QED. Student : OPT-350M.   

<html><body><table><tr><td>Method</td><td>LLaMA2</td><td>Mistral3</td><td>Qwen</td><td>LLaMA3</td></tr><tr><td>Teacher</td><td>61.68</td><td>64.03</td><td>62.16</td><td>65.96</td></tr><tr><td>Raw Text</td><td>49.61</td><td>51.24</td><td>51.21</td><td>51.91</td></tr><tr><td>ULD</td><td>50.71</td><td>52.08</td><td>52.89</td><td>52.81</td></tr><tr><td>Ours</td><td>51.96</td><td>52.96</td><td>53.99</td><td>54.38</td></tr></table></body></html>

Table 7: Effect of $N$ on QED.   

<html><body><table><tr><td>Number of Iterations N</td><td>5</td><td>10</td><td>20</td><td>50</td><td>100</td></tr><tr><td>OPT-350M</td><td>58.26</td><td>58.52</td><td>58.97</td><td>59.02</td><td>58.99</td></tr><tr><td>Pythia-410M</td><td>60.56</td><td>61.24</td><td>61.79</td><td>61.76</td><td>61.78</td></tr></table></body></html>

Sequence-level Sinkhorn distance excels token-level Sinkhorn distance. Table 4 demonstrates that sequencelevel Sinkhorn distance outperforms token-level distance across all student models. The sequence-level approach captures the geometric properties of logit distributions more comprehensively, providing a robust framework for understanding global contextual relationships among tokens. In contrast, while token-level distance, akin to a Holistic Absolute Difference Loss with an added entropy term, enhances robustness and mitigates sparsity, it fails to fully encapsulate the overarching patterns of entire sentences.

MultiLevelOT generalizes well on student LLMs across scales. We evaluate the impact of student LLMs’ sizes on the efficacy of MultiLevelOT through a detailed analysis in an unlabeled distillation context. Using two diverse tasks, QED (Lamm et al. 2021) and FairytaleQA (Xu et al. 2022), as illustrated in Figure 3, we observe that MultiLevelOT consistently enhances the performance of student models across various scales. This improvement substantiates MultiLevelOT’s advanced capability to effectively utilize optimal transport for knowledge distillation, clearly outperforming the ULD method (Boizard et al. 2024).

Generalization of MultiLevelOT across student architectures. Since MultiLevelOT relies solely on logits in the distillation process, it can be applied to any architecture. In addition to decoder-only models, we also test it on the encoder-decoder model mT0 (Muennighoff et al. 2023). Results in Table 5 reveal significant performance enhancements, underscoring MultiLevelOT’s flexibility and effectiveness across various architectural frameworks.

Table 8: Effect of $k$ on QED.   

<html><body><table><tr><td>Truncation Threshold k</td><td>5</td><td>20</td><td>50</td><td>100</td><td>1000</td></tr><tr><td>OPT-350M</td><td>58.54</td><td>58.84</td><td>58.97</td><td>58.78</td><td>58.42</td></tr><tr><td>Pythia-410M</td><td>61.42</td><td>61.50</td><td>61.79</td><td>61.40</td><td>61.32</td></tr></table></body></html>

Generalization of MultiLevelOT across teacher LLMs. An extensive evaluation of MultiLevelOT’s performance with varying teacher LLMs is conducted, employing models including LLaMA2 7B Chat (Touvron et al. 2023a), Mistral 7B Instruct (Jiang et al. 2023), Qwen 7B (Bai et al. 2023), and LLaMA3 8B Chat (Meta 2024). As shown in Table 6, MultiLevelOT consistently outshines its counterparts. This highlights MultiLevelOT’s robust capacity to leverage the distinct advantages of various teacher models.

$N$ as the number of Sinkhorn iterations. We analyze the impact of varying the number of Sinkhorn iterations $( N )$ on model performance, as summarized in Table 7. Increasing $N$ to 20 led to substantial improvements in F1 scores for both OPT-350M (58.97) and Pythia (61.79), underscoring the importance of adequate iterations for achieving convergence. Beyond this point, however, raising $N$ to 50 yields negligible performance gains, indicating a saturation threshold where additional iterations do not contribute further. This suggests that while sufficient iterations are necessary for convergence, excessive iterations offer diminishing returns and unnecessarily increase computational costs.

$k$ as the number of truncation threshold. Table 8 illustrates the effect of the truncation threshold $( k )$ on knowledge distillation for two student models, OPT-350M and Pythia410M. Our findings demonstrate that $k = 5 0$ is optimal for both models on the QED dataset. A smaller $k$ insufficiently captures the full sentence structure, weakening the Sinkhorn distance’s ability to model high-dimensional geometric information, and thus limiting the student model’s capacity to mimic the teacher’s logit distribution. Conversely, a larger $k$ introduces too many near-zero logit elements, adding noise and causing mode-averaging, which impairs the student’s ability to distinguish critical information.

# Conclusion

We propose MultiLevelOT for cross-tokenizer knowledge distillation that leverages both sequence-aware token-level and sequence-level optimal transport. Our method incorporates diverse cost matrices, using joint token optimization and Sinkhorn distance to provide a robust and comprehensive framework for KD. Extensive experiments demonstrate that MultiLevelOT consistently outperforms state-ofthe-art cross-tokenizer KD methods across various NLP tasks. Moreover, our approach proves robust across different student model families, architectures, sizes, and teacher models, showcasing its versatility and broad applicability.

Broader Impact It is prospective to use our method for multi-teacher knowledge transfer, integrating knowledge from multiple teachers to enhance model performance. Additionally, MultiLevelOT may be suitable for cross-language and multi-modal knowledge transfer, enabling robust alignment across different languages and data modalities.