# Seeing Your Speech Style: A Novel Zero-Shot Identity-Disentanglement Face-based Voice Conversion

Yan Rong, Li Liu\*

The Hong Kong University of Science and Technology (Guangzhou) yrong854@connect.hkust-gz.edu.cn, avrillliu@hkust-gz.edu.cn

# Abstract

Face-based Voice Conversion (FVC) is a novel task that leverages facial images to generate the target speaker’s voice style. Previous work has two shortcomings: (1) suffering from obtaining facial embeddings that are well-aligned with the speaker’s voice identity information, and (2) inadequacy in decoupling content and speaker identity information from the audio input. To address these issues, we present a novel FVC method, Identity-Disentanglement Face-based Voice Conversion (ID-FaceVC), which overcomes the above two limitations. More precisely, we propose an IdentityAware Query-based Contrastive Learning (IAQ-CL) module to extract speaker-specific facial features, and a Mutual Information-based Dual Decoupling (MIDD) module to purify content features from audio, ensuring clear and high-quality voice conversion. Besides, unlike prior works, our method can accept either audio or text inputs, offering controllable speech generation with adjustable emotional tone and speed. Extensive experiments demonstrate that IDFaceVC achieves state-of-the-art performance across various metrics, with qualitative and user study results confirming its effectiveness in naturalness, similarity, and diversity.

![](images/ef095c9f442c95ef71dbd90199694e81a3912cfbc366805d2042736b2fc9b3b8.jpg)  
Figure 1: (a) Traditional voice conversion (VC) paradigm. (b) Our novel ZS-FVC paradigm, which accepts either audio or text as input and allows control over the emotional tone and speed of the generated speech.

# Project website — https://id-facevc.github.io Extended version — https://arxiv.org/pdf/2409.00700

# Introduction

Voice Conversion (VC) (Choi, Lee, and Lee 2024; Yao et al. 2024) aims to change the speaker identity in speech from a source speaker to that of a target speaker, while preserving the linguistic content. However, audio from the target speaker is not always available in some scenarios (e.g., digital humans, historical figures). Instead, some studies have explored an alternative approach by generating the identity information of unseen speakers’ voices from their facial images (Mavica and Barenholtz 2013; Smith et al. 2016), known as Zero-Shot Face-based Voice Conversion (ZSFVC). Recently, this has become a promising research topic with potential applications in various scenarios, such as generating voices that match character appearances in automated film dubbing (Cong et al. 2024) and personalized virtual assistants (Park et al. 2024).

In the literature, great progress in this domain has been achieved by prior work (Goto et al. 2020; Lu et al. 2021; Sheng et al. 2023; Weng, Shuai, and Cheng 2023). The fundamental challenge is to accurately map identity information between faces and voices. Specifically, this involves (1) acquisition of facial embeddings that are well-aligned with the speaker’s voice identity, and (2) decoupling of content and speaker identity information from the audio input.

For the first challenge, the current state-of-the-art (SOTA) work FVMVC (Sheng et al. 2023) used FaceNet (Schroff, Kalenichenko, and Philbin 2015) to extract general facial features and mapped them through a memory net. Another SOTA work, SP-FaceVC (Weng, Shuai, and Cheng 2023) averaged all frames to achieve consistent facial embeddings. However, these methods focus on general facial features rather than speaker-specific features, which include substantial non-specific, identity-irrelevant information (e.g., facial expressions, head angles, background). As a result, the models become highly dependent on the training data and lack the ability to locate unique voice characteristics among different speakers, leading to the production of general voices. For the second challenge, FVMVC attempted feature decoupling through a mixed supervision strategy, relying heavily on the quality and scope of the supervision voices. However, in practical scenarios, it is often difficult to acquire adequate and balanced supervision, leading to suboptimal decoupling performance. SP-FaceVC used a low-pass filtering strategy in data pre-processing to eliminate high-frequency elements from audio signals, aiming to reduce style features linked to the speaker identity. Despite its simplicity, this hard filtering approach risks indiscriminately filtering out some key voice details, thereby affecting the naturalness and expressiveness of the synthesized voice and potentially introducing noise and other artifacts.

Table 1: Comparison of input ways and controllability with previous studies.   

<html><body><table><tr><td>Methods</td><td>Input</td><td>Controllability</td></tr><tr><td>Face2Speech (Goto et al. 2020)</td><td>Text</td><td>X</td></tr><tr><td>FaceVC (Lu et al.2021)</td><td>Audio</td><td>X</td></tr><tr><td>FVMVC (Sheng et al. 2023)</td><td>Audio</td><td>X</td></tr><tr><td>SP-FaceVC(Weng,Shuai,and Cheng 2023)</td><td>Audio</td><td>X</td></tr><tr><td>FaceTTS (Lee,Chung,and Chung 2023)</td><td>Text</td><td></td></tr><tr><td>Ours (ID-FaceVC)</td><td>Audio /Text</td><td>√</td></tr></table></body></html>

To address the above two challenges, we introduce a novel zero-shot Identity-Disentanglement Face-based Voice Conversion (ID-FaceVC) method. For the first challenge, instead of adopting static encoding methods that generalize facial features, we design an Identity-Aware Query-based Contrastive Learning (IAQ-CL) module to precisely extract the most identity-relevant facial features. Specifically, we propose a Self-Adaptive Face-Prompted QFormer (SAFPQ), which employs a set of learnable self-adaptive face prompts to query identity-relevant facial features from a frozen Contrastive Language-Image Pretraining (CLIP) visual encoder (Radford et al. 2021). Indeed, the SAFPQ functions as an information bottleneck, efficiently filters and maps facial features to produce speech-relevant facial features, which are then subjected to contrastive learning with identity features extracted from audio.

For the second challenge, rather than using implicit supervision or hard filters, we design a novel Mutual Informationbased Dual Decoupling (MIDD) module to purify the extracted content features. This module decomposes speech into subspaces representing different attributes and minimizes the overlapping information between speaker identity and content features through Mutual Information (MI) constraints. Additionally, inspired by (Peng et al. 2023; Park et al. 2024), we implement the fine-grained speaker identity supervision to fully leverage speaker identity information, compelling the model to learn the subtle distinctions between different speakers and preventing model collapse.

In addition, previous approaches that employed the target speaker’s voice as input (Lu et al. 2021; Sheng et al. 2023; Weng, Shuai, and Cheng 2023), as depicted in Table 1, suffer from limitations in practical applications due to the occasional unavailability of the reference audio. Some existing works have utilized text as the input for speech generation (Goto et al. 2020; Lee, Chung, and Chung 2023), but their outputs lack the flexibility to manipulate speech style and often produce speech in a “machine” manner. In this work, we first incorporate text as an alternative modality during the inference stage and introduce a style-controllable strategy that allows for control over the emotion and speed of the generated speech, thereby enabling the generation of natural, rhythmical, and controllable speech from text.

In summary, the main contributions of this work are as follows.

• A novel paradigm named ID-FaceVC is proposed for zero-shot face-based voice conversion that can accept either audio or text as input, allowing control over the emotional tone and speed of the generated speech. To the best of our knowledge, this is the first attempt to explore dualinput controllable face-based voice conversion. • We design an IAQ-CL module, containing a new SelfAdaptive Face-Prompted QFormer to query facial features most relevant to speaker identity and forces the model to learn the subtle differences between speakers. • We propose an effective mutual information-based MIDD module to completely decouple content and speaker identity from audio features. • Extensive experimental results demonstrate that our method achieves SOTA performance across multiple metrics. Qualitative and user study results further validate the effectiveness of the proposed model in terms of naturalness, similarity, and diversity.

# Related Work

# Evidence of Face-Voice Correlation

Facial and vocal characteristics are closely linked to individual identity. Studies have demonstrated a natural synergy between these features, collectively providing concordant source identity information (Smith et al. 2016). Features of a voice can be inferred from facial structures (Krauss, Freyberg, and Morsella 2002; Mavica and Barenholtz 2013). For example, vocal pitch and intonation may be associated with facial features like jaw width and eyebrow density, which together create a distinct identity signature. Recently, several studies have exploited the strong similarity between voice and face for novel applications, such as reconstructing a speaker’s face from their voice (Wang et al. 2023; Oh et al. 2019; Wen, Raj, and Singh 2019; Duarte et al. 2019). Our research explores the inverse of this process, generating diverse vocal styles from various facial images.

# Face-based Voice Conversion

Prior research has validated the potential for synthesizing speech from facial features. Face2Speech (Goto et al. 2020) pioneered this field with a three-stage training strategy and a supervised generalized end-to-end loss to generate speech that reflects speaker facial characteristics. Building on this foundation, subsequent works proposed more adaptable loss functions (Wang et al. 2022) and more sophisticated network designs (Lee, Chung, and Chung 2023) to enhance the quality of the synthesized speech. These methodologies typically employ text as the input to avoid entanglement issues. FaceVC (Lu et al. 2021) developed a three

1 Ep: Pitch Encoder Econ : Content Encoder : Face-Identity Latent Space Espk : Speaker Encoder Ef :Face Encoder : Speech-Identity Latent Space Lrec D : Audio Decoder F-V : Face-Voice : Content Latent Space D → 1 SAFPQ : Self-Adaptive Face-Prompted QFormer :Pitch Latent Space F-V Mapping Contrastive learning Mutual Information learning Lid-s Lid- Fspk 工 □□□□□ Fquery 1 Fcon Adprer er LM1 SAPO iLcon SAPO 00 青 Lcon 十 LMI ▲ □ 2 → ↑ ↑ ↑ cross attn. Ep Econ 厂 Espk E □ 1 a ↑q Adapter Ff self attn. self attn. ↑ ↑ → k↑q v ↑k 9 正老書 MIDD N IAQ-CL\ Fspk Fcon Speech A Mel-spec A Face Image A Self-adaptive face prompts

stage model that leverages a bottleneck adjustment strategy and a straightforward MSE loss to extract necessary content embeddings from audio. However, this model struggles to capture the complex mappings between speech and facial domains, often defaulting to predicting an “average voice” across variations, making it unsuitable for zero-shot applications. The most advanced approaches in this field, FVMVC (Sheng et al. 2023) and SP-FaceVC (Weng, Shuai, and Cheng 2023), improved upon FaceVC through memorybased feature mapping and rigorous data preprocessing.

Nevertheless, these methods still have considerable potential for improvement in achieving well-aligned facial embeddings with speech and effectively decoupling content from speaker identity in audio features.

# Our Method

Our proposed ID-FaceVC employs an end-to-end training approach. It comprises three main components: IDAware Query-based Contrastive Learning module, Mutual Information-based Dual Decoupling module, and Alternative Text-Input with Style Control module.

# ID-Aware Query-based Contrastive Learning

We design the IAQ-CL module to extract facial features that are well-aligned with the speaker’s voice identity. This module includes Self-Adaptive Face-Prompted QFormer and face-related speaker identity supervision.

Self-Adaptive Face-Prompted QFormer. Considering the inherent limitation of CNN-based architectures in handling the diversity of facial features, we instead employ a frozen CLIP visual encoder (Radford et al. 2021) to extract features from facial images. For the frame-wise visual embeddings extracted by the CLIP visual encoder, we compute the arithmetic mean to obtain average frame-level facial features, rather than randomly selecting a single frame as the facial embedding, to reduce potential sampling bias.

Due to the high-dimensional and redundant nature of CLIP visual features, facial embeddings contain abundant information, including facial expressions, head poses, and backgrounds, with only a small portion related to the speaker’s style. Therefore, we propose the SAFPQ to filter the most speech-relevant features, as illustrated in Figure 2. Unlike the vanilla Query Transformer (QFormer) (Li et al. 2023), our model better integrates identity information from both face and voice domains, resulting in a more cohesive representation. The SAFPQ functions as an information bottleneck, filtering out redundant facial features while emphasizing those crucial for speech. In the inference stage, the self-adaptive face prompts retrieves identity-relevant facial features from input facial embeddings, facilitating the prediction of the speaker’s style from unseen facial images.

To be specific, we initialize a set of learnable self-adaptive face prompts. The most informative prompts are highlighted through a self-attention mechanism that integrates the information from a global perspective. Subsequently, the face prompts interact with facial embeddings via cross-attention to retrieve features relevant to the identity information. Finally, a fully connected layer fuses these retrieved features. The process are defined as follows:

$$
A _ { s e l f } = \mathrm { s o f t m a x } \left( \frac { \mathbf { Q } W _ { q } ^ { \mathrm { s e l f } } \left( \mathbf { Q } W _ { k } ^ { \mathrm { s e l f } } \right) ^ { T } } { \sqrt { d _ { k } } } \right) \mathbf { Q } W _ { v } ^ { \mathrm { s e l f } } \ : ,
$$

$$
A _ { c r o s s } = \mathrm { s o f t m a x } \left( \frac { A _ { s e l f } W _ { q } ^ { \mathrm { c r o s s } } \left( F _ { f } W _ { k } ^ { \mathrm { c r o s s } } \right) ^ { T } } { \sqrt { d _ { k } } } \right) F _ { f } W _ { v } ^ { \mathrm { c r o s s } } ,
$$

$$
F _ { q u e r y } = \mathrm { F F N } ( A _ { c r o s s } ) ,
$$

where $W _ { q _ { . } } ^ { \mathrm { s e l f } } , W _ { k _ { . } } ^ { \mathrm { s e l f } } , W _ { v } ^ { \mathrm { s e l f } }$ are the learnable weights for the self-attention, ankd $W _ { q } ^ { \mathrm { c r o s s } } , W _ { k } ^ { \mathrm { c r o s s } } , W _ { v } ^ { \mathrm { c r o s s } }$ are the learnable weights for the cross-attention. $Q$ is the self-adaptive face prompts, $d _ { k }$ is the dimension of the key, $F _ { f }$ is the facial embedding extracted by the CLIP, and $F _ { q u e r y }$ represents the final queried facial features.

![](images/975968684bf507f5ed3177d40060ab5b2dc71546ee19819f424e96b2f35acb30.jpg)  
Figure 3: The inference stage of ID-FaceVC. Text is introduced as an alternative modality to produce natural, rhythmic, and controllable speech.

Recall that our objective is to extract features highly relevant to the speaker’s identity, ensuring that the retrieved facial embeddings closely match the style features in speech. We employ contrastive learning to measure the distance between these two features, thereby optimizing the selfadaptive face prompts. This encourages the speech style and facial embeddings from the same speaker to be as similar as possible, while those from different speakers are distinctly separated. The formulation for this process is as follows:

$$
L _ { c o n } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } y _ { i , j } \log \left( \frac { \exp ( \sin ( i , j ) / \tau ) } { \sum _ { k = 1 } ^ { N } \exp ( \sin ( i , k ) / \tau ) } \right) ,
$$

where $N$ is the number of samples in a batch, $\tau$ is a temperature hyperparameter, $i$ is a fixed index for facial embeddings, $j$ and $k$ are indices for speech embeddings, and $y _ { i , j }$ is an indicator function. If samples $i$ and $j$ belong to the same speaker, then $y _ { i , j } = 1$ ; otherwise, $y _ { i , j } = 0$ .

Face-related Speaker-identity Supervision. To distinguish key facial features between different speakers, inspired by (Peng et al. 2023), we design a fine-grained speaker identity supervision mechanism to enhance our model’s capability. This supervision ensures that facial features should maintain consistency for the same speaker while exhibiting distinctiveness between different speakers. The formulation can be expressed as:

$$
L _ { i d - f } = - \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \sum _ { i = 1 } ^ { C } t _ { n i } \cdot \log \left( p _ { n i } \right) ,
$$

where $C$ represents the number of distinct speakers and $t _ { n i }$ denotes the one-hot encoded target label for the $n$ -th sample (where $t _ { n i } = 1$ if the sample belongs to the $i$ -th speaker, otherwise $t _ { n i } = 0$ ). $p _ { n i }$ is the softmax probabilities that the $n$ -th sample’s $F _ { q u e r y }$ belong to the $i$ -th speaker.

During the inference, ID-FaceVC is able to generate consistent style speech for different facial images of the same speaker and diverse style speech for different speakers.

# Mutual Information-based Dual Decoupling

We propose the MIDD module to achieve precise representation of different disentangled latent spaces and purification of speech content information. It includes Disentangled Latent Space and Mutual Information-based Decoupling.

Disentangled Latent Space. The core of achieving robust content representation is the removal of non-content-related features. A natural idea is to decompose speech into distinct subspaces that represent various attributes. Thus, we use two different encoders to separately extract a compact speaker style code $F _ { s p k }$ and a continuous content code $F _ { c o n }$ from the mel spectrogram. Specifically, to fully leverage the powerful representational capabilities of large models, we employ the Contrastive Language-Audio Pretraining (CLAP) (Wu et al. 2023) audio encoder as our speaker encoder. As depicted in Figure 2, the features obtained from the CLAP are processed through SAFPQ, following the same procedure as facial embedding handling but without the crossattention module. For the content encoder, we adopt vector quantization (Bitton, Esling, and Harada 2020) and contrastive predictive coding (Oord, Li, and Vinyals 2018) techniques, commonly used in voice conversion tasks, to extract the content embeddings $F _ { c o n }$ .

Mutual Information-based Decoupling. Given the diversity of speech styles, merely constructing two separate latent spaces may not ensure sufficient feature decoupling. Previous decoupling methods, such as inter-speaker supervision (Schroff, Kalenichenko, and Philbin 2015), still result in certain overlaps among features. To address this issue, we utilize MI, which can measure the overall dependency between variables and capture both linear and non-linear relationships (Veyrat-Charvillon and Standaert 2009), as a metric to evaluate the correlation between speaker embeddings and content embeddings extracted from speech. However, due to the high dimensionality and unknown distributions of the variables, directly calculating probability distributions in MI is impractical. To solve this, we employ a variational upper bound technique (Cheng et al. 2020), to establish parameterized conditional distributions, which aids in controlling the minimization process by estimating the upper bound of MI:

$$
L _ { M I } = \frac { 1 } { N ^ { 2 } } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \log \frac { q \left( F _ { \mathrm { c o n } , i } \mid F _ { \mathrm { s p k } , i } \right) } { q \left( F _ { \mathrm { c o n } , j } \mid F _ { \mathrm { s p k } , i } \right) } ,
$$

where $F _ { \mathrm { s p k } , i }$ denotes the speaker style embeddings for the $i$ -th sample, $F _ { \mathrm { c o n } , i }$ and $F _ { \mathrm { c o n } , j }$ represent the content embeddings for the $i$ -th and $j$ -th samples, respectively.

By minimizing the overlap information between the extracted style features and content features, we successfully establish a speaker style space related to identity and a content space associated with semantics.

In addition, similar to Eq. (5), we apply speech-related speaker-identity supervision $L _ { i d - s }$ to the style features extracted from speech. In this context, $p _ { n i }$ in represents the softmax probability that the speaker feature $F _ { s p k }$ of the $n$ - th sample belongs to the $i$ -th speaker, while other variables remain consistent with Eq. (5). The joint speaker-identity supervision for both facial and speech features enforces the model to recognize the consistency within the same speaker’s identity and the diversity across different speakers’ identities. These two loss functions prevent the generation of overly similar outputs, thereby protecting against mode collapse.

Table 2: Comparison with SOTA methods. Best performances are highlighted in bold, while second-best are underlined.   

<html><body><table><tr><td rowspan="2">Input</td><td rowspan="2">Methods</td><td colspan="3">Naturalness</td><td>Similarity</td><td colspan="2">Consistency & Diversity</td></tr><tr><td>UTMOS ↑</td><td>WER↓</td><td>CER↓</td><td>SECS ↑</td><td>SEC↑</td><td>SED↓</td></tr><tr><td rowspan="4">Audio Input</td><td>FaceVC (Lu et al. 2021)</td><td>2.155</td><td>16.67%</td><td>10.79%</td><td>0.702</td><td>0.986</td><td>0.965</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>0.988</td><td></td></tr><tr><td>SP-FaceVCVc(sShuait alCheng 2023)</td><td>1.833</td><td>29.17%</td><td>19.67%</td><td>0.723</td><td></td><td>0.91</td></tr><tr><td>Ours (ID-FaceVC)</td><td>3.286</td><td>12.11%</td><td>7.86%</td><td>0.713</td><td>0.988</td><td>0.832</td></tr><tr><td rowspan="2">Text Input</td><td>FaceTS (Ler ChD-Fand Chung 2023)</td><td>2.104</td><td>14.31%</td><td></td><td>0.701</td><td>0.987</td><td>0.912</td></tr><tr><td></td><td></td><td></td><td>8.70%</td><td></td><td></td><td></td></tr></table></body></html>

# Alternative Text-Input with Style Control

In addition to audio inputs, text serves as a more flexible modality in practical applications because it does not require prior recording of a source speaker’s speech. In this work, as illustrated in Figure 3, we introduce text as an alternative option to specify the content of generated audio, which broadens the applicability and accessibility of our framework.

The transition from text to audio often results in monotonous narrations due to the absence of references for emotion, accent, and rhythm. To address this issue, inspired by the OpenVoice (Qin et al. 2023), we develop a style control strategy that uses the base speaker TTS as a bridge to generate an intermediate single-speaker audio. This audio can be flexibly manipulated in terms of speed and emotion through style control parameters. The choice of base speaker TTS is flexible, allowing for either a single-speaker or multi-speaker TTS, as the timbre produced by the TTS is not our focus. In this task, we select the VITS (Kim, Kong, and Son 2021) model as the base speaker TTS, which accepts both text and style control inputs. The audio generated through this process then serves as the source speaker audio input into our network, where a content encoder extracts the speech content, and a pitch encoder captures the pitch information. Together with the speaker style information inferred from unseen speaker facial images, we generate the final audio output.

In the inference, when using text as input, the flexible control of speech and the injection of timbre inference are separated, allowing for a straightforward and training-free implementation of face-based controllable voice conversion.

# Training Loss

We utilize L2 loss to evaluate the quality of the reconstructed mel spectrograms. The formula for this is as follows:

$$
L _ { r e c } = \| M e l - \hat { M e } l \| _ { 2 } ^ { 2 } ,
$$

where $\mathit { M e l }$ and $\boldsymbol { M } \boldsymbol { e l }$ represent the Mel spectrogram input to the network and the Mel spectrogram reconstructed by the model, respectively. Additionally, we follow the training setup described in FVMVC (Sheng et al. 2023), incorporating both inter-speaker supervision loss and face-voice mapping loss, collectively referred to as $L _ { F }$ .

The total training loss is defined as follows: $\mathcal { L } = L _ { r e c } + \lambda _ { 1 } L _ { c o n } + \lambda _ { 2 } L _ { M I } + \lambda _ { 3 } L _ { i d - f } + \lambda _ { 4 } L _ { i d - s } + \lambda _ { 5 } L _ { F } ,$ (8) where $\lambda _ { 1 }$ is the weight of $L _ { c o n }$ (in Eq. (4)), $\lambda _ { 2 }$ is the weight of $L _ { M I }$ (in Eq. (6)), $\lambda _ { 3 }$ is the weight of $L _ { i d - f }$ (in Eq. (5)), $\lambda _ { 4 }$ is the weight of $L _ { i d - s }$ , and $\lambda _ { 5 }$ is the weight of $L _ { F }$ .

# Experiment and Result Experimental Setup

Datasets. To the best of our knowledge, current ZS-FVC methods utilized the LRS3 (Afouras, Chung, and Zisserman 2018) dataset, which comprises over 400 hours of TED talks collected from YouTube, for training. For a fair comparison, we follow the same dataset setup. More precisely, we selected the paired data from the top 200 speakers by video count, resulting in 11,430 videos for training and 5,173 videos for validation. For testing, we randomly selected 16 previously unseen speakers, including 8 target speakers (4 male, 4 female) and 8 source speakers (4 male, 4 female).

Implementation Details. We employ the MTCNN (Zhang et al. 2016) to detect and align faces in each video frame. Facial features are extracted using the ViT-B/32 from CLIP, with outputs from the penultimate layer utilized to enhance generalization over the final layer. Audio is extracted from video clips via FFmpeg (Yamamoto, Song, and $\mathrm { K i m } \ 2 0 2 0 )$ , and the HTSAT-base from CLAP serves as the speaker feature extractor. Training is conducted on a single Nvidia-A800 GPU with a batch size of 256 for 2000 epochs. F-V mapping is a memory-based feature mapping module, following the setup of FVMVC (Sheng et al. 2023). For the vocoder, we utilize a pretrained ParallelWaveGAN (Yamamoto, Song, and $\mathrm { K i m } \ 2 0 2 0 )$ . Loss weights specified in Eq. (8) are set at $\lambda _ { 1 } ~ = ~ 0 . 1$ , $\lambda _ { 2 } ~ = ~ 0 . 0 1$ , $\lambda _ { 3 } ~ = ~ 0 . 1$ , $\lambda _ { 4 } = 0 . 1$ , and $\lambda _ { 5 } = 1$ .

# Evaluation Metrics

Subjective Metrics. We evaluate the Mean Opinion Score (MOS) for speech naturalness (nMOS) and speaker similarity (sMOS) with ratings from 8 listeners. Ratings are assigned using a five-point scale: $1 { = } \mathrm { B a d }$ , ${ \tt 2 } \mathrm { = } { \tt P o o r }$ , $ 3 { = } \mathrm { F a i r } .$ $4 { = } \mathrm { G o o d }$ , $5 { = }$ Excellent. Additionally, we conduct two preference tests to assess the alignment between generated speech and facial images: (1) selecting the generated speech that best matches a given face, and (2) selecting the facial image that best matches the style of a given generated speech.

![](images/9a7802169d308a5078b916830292fbab6797b778504afc6e6146742cf1d1499a.jpg)  
Figure 4: The t-SNE visualization of speaker embeddings form generated speech. Each point represents a voice sample, with nearby images showing the faces of the speakers.

Objective Metrics. We utilize the UTMOS (Saeki et al. 2022) to evaluate the overall quality of the generated speech, serving as an objective alternative to the nMOS. The robustness and content consistency of the generated speech are quantified using the word error rate (WER) and character error rate (CER), which are calculated using the Whisper (Radford et al. 2023). Additionally, speaker embeddings extracted via Resemblyzer are used to compute the speaker encoder cosine similarity (SECS) between the generated speech and the true style speech of the same speaker. Due to the absence of the paired speech data, SECS does not ensure content consistency in a pair, but the relative SECS scores across models indicate the ability to accurately map facial images to speaker styles. Moreover, we have developed metrics for speaker embedding consistency (SEC), which measure the uniformity of speech outputs from different facial angles of the same speaker, and speaker embedding diversity (SED), which assess the variability among speech outputs from different speakers.

# Quantitative Result and Analysis

Comparison with SOTA Methods. We evaluate our method against four recent face-to-speech generation methods, as outlined in Table 2. Among these, FaceVC (Lu et al. 2021), SP-FaceVC (Weng, Shuai, and Cheng 2023), and FVMVC (Sheng et al. 2023) control speech content using audio from the source speaker, while FaceTTS (Lee, Chung, and Chung 2023) uses text as input. Our approach exhibits a notable improvement on the UTMOS metric, indicating enhanced audio quality. Notably, our method achieves lower WER and CER, benefiting from the efficient content refinement implemented by MIDD. Although our method slightly trails SP-FaceVC in terms of the SECS metric, it surpasses all other methods evaluated. It is important to highlight that SP-FaceVC scores 0.912 on the SED metric, indicating a high level of timbral convergence among different speakers, which suggests a tendency toward generating an “average” timbre. In contrast, our method demonstrates superior performance on the SED metric, effectively capturing the most task-relevant facial features. Additionally, our results on the SEC metric highlight the robustness of our method to variations in facial embeddings.

Table 3: Results of ablation studies on different model components. Best performances are highlighted in bold, while second-best performances are underlined.   

<html><body><table><tr><td>IAQ-CL</td><td>MIDD</td><td>Lid-f</td><td>Lid-s</td><td>UTMOS ↑</td><td>WER↓</td><td>CER↓</td><td>SECS ↑</td></tr><tr><td></td><td></td><td></td><td></td><td>2.945</td><td>15.29%</td><td>10.04%</td><td>0.693</td></tr><tr><td>√</td><td></td><td></td><td></td><td>3.227</td><td>18.04%</td><td>11.57%</td><td>0.726</td></tr><tr><td>√</td><td>√</td><td></td><td></td><td>3.266</td><td>12.70%</td><td>8.63%</td><td>0.709</td></tr><tr><td>√</td><td></td><td>/</td><td></td><td>3.236</td><td>12.25%</td><td>7.97%</td><td>0.709</td></tr><tr><td>√</td><td>√</td><td></td><td>√</td><td>3.221</td><td>12.01%</td><td>8.01%</td><td>0.712</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>3.286</td><td>12.11%</td><td>7.86%</td><td>0.713</td></tr></table></body></html>

Ablation Studies. We investigate the impact of different model components on ID-FaceVC by conducting the following ablation studies: (1) w/o IAQ-CL: Randomly selects a facial frame from a video and uses FaceNet to generate a 512-dimensional vector, which is then fused through selfattention mechanisms and linear layers. (2) w/o MIDD: Directly extracts speaker embeddings by the Resemblyzer and maps these features to face embeddings using self-attention mechanisms and linear layers. (3) w/o $L _ { i d - f }$ and (4) w/o $L _ { i d - s }$ : Omits the corresponding loss function. Experimental results are shown in Table 3.

In contrast to using static encoders for direct facial feature extraction, the IAQ-CL module significantly improves voice generation quality and face-to-voice mapping by effectively capturing facial features relevant to speaking styles. The MIDD module efficiently purifies the extracted content information, enhancing the clarity of the generated speech. Although there is a slight reduction in the SECS, this likely results from a trade-off with some intonation-related style features while preserving semantic content. This focus on content plays a crucial role in the clear expression of voice content. Additionally, supervision based on both facial and voice characteristics of speakers further strengthens the model’s ability to distinguish critical features, thus improving generalization across different speakers.

# Qualitative Result and Analysis

Visualization of Controllable Speech. For text-based input, we visualize the Mel spectrograms under various emotional states and speaking speeds, as depicted in Appendix Figure 2. In the “whispering” state, the generated audio exhibits a more dispersed energy pattern with an increase in high-frequency components due to the incomplete vibration of vocal cords typical in whispering. In contrast, in the “angry” state, the speaker’s voice shows greater fluctuations and intensity, with a quicker frequency and broader dynamic range. As the speaking speed increases, the spectral energy distribution becomes more compact, reducing the intervals between syllables. Conversely, when the speed decreases, the energy distribution expands, and syllables lengthen. These observations demonstrate that ID-FaceVC performs well in controlling different emotions and speaking speeds.

![](images/59fa3847d007eb04a647c5e24c6cd013fcbf8c5085abe3c34838d1cea5343d67.jpg)  
Figure 5: User study results for naturalness and similarity metrics. Ours (T) and Ours (A) represent ID-FaceVC with text and audio inputs, respectively.

Style Manipulation. We interpolate facial embeddings from two different speakers to generate various voice outputs, as shown in Appendix Figure 3. As the facial embeddings transition from female to male, the fundamental frequency of the generated voice gradually decreases, and the harmonic distribution becomes denser. The voices in the intermediate transition phase not only retain high-frequency harmonic features typical of female voices but also incorporate low-frequency characteristics of male voices, illustrating a smooth transition in voice characteristics from female to male. This demonstrates our model’s ability to precisely control voice output based on varying facial features.

Distribution of Speaker Embedding. For the generated speech, we use the Resemblyzer to extract speaker embeddings and visualize them using t-SNE, as depicted in Figure 4. Voice samples generated from the same facial image form tight clusters, indicating that our model successfully maps unique vocal styles to different faces. Notably, embeddings for speakers of different genders display distinct distributions, with those of the same gender and similar ages showing closely matched speaker embeddings. This demonstrates our model’s capability to effectively capture the most speech-relevant features from facial images.

Visualization of Different Face Angles. We randomly selected two speakers and three facial images of each, captured from various angles, to perform ZS-FVC, as shown in Appendix Figure 1. Regardless of the facial expressions and angles, the voices generated by the model remained consistent across different images of the same speaker. This consistency is attributed to the model’s ability to effectively align identity-related features in the faces with style-related features in the voice, demonstrating robustness to camera positions, backgrounds, and other noise.

![](images/dc46f8d0b93a5c6c4338c6e03ccb218fcb33379507e9abbbf2d824eb3a576c54.jpg)  
Figure 6: Results of the preference test. (a) Given a facial image, preference results for choosing the more matching speech between outputs from FVMVC and ID-FaceVC. (b) Accuracy of selecting the more matching facial image given the model-generated speech.

# User Study

We evaluated the naturalness and similarity of generated speech through a user study involving eight experts. Each expert rated 27 sets of audio samples, with each set containing six comparative groups. As depicted in Figure 5, our method consistently outperforms current SOTA approaches in both naturalness and similarity, while also exhibiting smaller confidence intervals. These findings demonstrate that ID-FaceVC reliably produces high-quality outputs with enhanced stability.

We further validate our model’s ability to map facial features to speech characteristics through preference tests. To increase the challenge of the experiment, we conduct gender-matched tests, selecting face and audio samples from individuals of the same gender. As depicted in Figure 6, in the face-based preference test, $5 7 . 1 4 \%$ of evaluators believe that ID-FaceVC produces results that better match the given facial images. In the voice-based preference test, evaluators correctly identify the match $2 2 . 9 \%$ more often when the speech is generated by ID-FaceVC rather than FVMVC, demonstrating that the speech generated by IDFaceVC more accurately aligns with the corresponding facial images.

# Conclusion

In this work, we introduce a novel ID-FaceVC framework, effectively generating speech that aligns with facial identity features. Our framework includes the IAQ-CL and MIDD modules to precisely map facial features to speech. Additionally, we incorporate text as an alternative modality for controlling speech content and employ a style controllable strategy that ensures speech generated from text is natural, rhythmic, and controllable. Both quantitative and qualitative experiments validate the overall effectiveness of our framework and the individual modules. Future work aims to expand beyond audio generation to include expressive facial animations, transitioning from merely “audible” to “both audible and visible.”