# Multi-modal and Multi-scale Spatial Environment Understanding for Immersive Visual Text-to-Speech

Rui Liu1\*, Shuwei $\mathbf { H } \mathbf { e } ^ { 1 }$ , Yifan $\mathbf { H } \mathbf { u } ^ { 1 }$ , Haizhou Li2, 3

1Inner Mongolia University, China 2Shenzhen Research Institute of Big Data, School of Data Science, The Chinese University of Hong Kong, Shenzhen, China 3Department of Electrical and Computer Engineering, National University of Singapore, Singapore liurui imu $@$ 163.com, shuwei he $\textcircled { a } 1 6 3 . \mathrm { c o m }$ , hyfwalker $\textcircled { a } 1 6 3 . \mathrm { c o m }$ , haizhouli $@$ cuhk.edu.cn

# Abstract

Visual Text-to-Speech (VTTS) aims to take the environmental image as the prompt to synthesize the reverberant speech for the spoken content. The challenge of this task lies in understanding the spatial environment from the image. Many attempts have been made to extract global spatial visual information from the RGB space of an spatial image. However, local and depth image information are crucial for understanding the spatial environment, which previous works have ignored. To address the issues, we propose a novel multi-modal and multi-scale spatial environment understanding scheme to achieve immersive VTTS, termed $\mathbf { M } ^ { 2 } \mathbf { S } \mathbf { E }$ -VTTS. The multimodal aims to take both the RGB and Depth spaces of the spatial image to learn more comprehensive spatial information, and the multi-scale seeks to model the local and global spatial knowledge simultaneously. Specifically, we first split the RGB and Depth images into patches and adopt the Gemini-generated environment captions to guide the local spatial understanding. After that, the multi-modal and multiscale features are integrated by the local-aware global spatial understanding. In this way, $\mathrm { \dot { M } ^ { 2 } S E }$ -VTTS effectively models the interactions between local and global spatial contexts in the multi-modal spatial environment. Objective and subjective evaluations suggest that our model outperforms the advanced baselines in environmental speech generation.

Code and Audio Samples — https://github.com/AI-S2-Lab/M2SE-VTTS

# Introduction

Visual Text-to-Speech (VTTS) aims to leverage the environmental image as the prompt to generate the reverberant speech that corresponds to the spoken content. With the advancement of human-computer interaction, VTTS has become integral to intelligent systems and plays an important role in fields such as augmented reality (AR) and virtual reality (VR) (Liu et al. 2023b).

Unlike acoustic matching tasks that transform input speech to match the environmental conditions of a reference source (Chen et al. 2022; Liu et al. $2 0 2 3 \mathrm { a }$ ; Somayazulu, Chen, and Grauman 2024; Im and Nam 2024), VTTS seeks to synthesize speech with the environmental characteristics of the reference based on given textual content (He, Liu, and Li 2024). For example, Lee et al. (2024) utilizes the pretrained CLAP model to map a textual or audio description into an environmental feature vector that controls the reverberation aspects of the generated audio. Tan, Zhang, and Lee (2022) design an environment embedding extractor that learns environmental features from the reference speech. In more recent studies, Liu et al. (2023b) propose a visual-text encoder based on the transformer to learn global spatial visual information from the RGB image. Building on these advancements, this paper focuses on employing visual information as the cue to generate reverberation audio for the targeted scene.

However, previous VTTS methods have not fully understood the spatial environment, due to the neglect of local and depth image information. For example, the local elements in the spatial environment can directly influence the reverberation style. Specifically, the hard surfaces such as tables reflect sound waves, while softer materials like carpets absorb them, directly affecting the audio’s authenticity and naturalness (Chen et al. 2023, 2022; Liu et al. 2023b). In addition, the Depth space of the image contains positional relationships within the spatial environment (Majumder et al. 2022; Chen et al. 2023), such as the arrangement of objects, the position of the speaker and the room geometry. Therefore, it is crucial for VTTS systems to accurately capture the local and depth spatial environment information simultaneously.

To address the issues, we propose a novel multi-modal and multi-scale spatial environment understanding scheme to achieve immersive VTTS, termed $\mathbf { M } ^ { 2 } \mathbf { S } \mathbf { E }$ -VTTS. The multi-modal aims to take both the RGB and Depth spaces of the spatial image to learn more comprehensive spatial information, such as the speaker’s location and the positions of key objects that influence sound absorption and reflection. The multi-scale seeks to model the impact of local and global spatial knowledge on reverberation. Specifically, we first split the RGB and Depth images into patches following the visual transformer strategy (Dosovitskiy et al. 2021). In addition, we adopt the Gemini-generated (Team et al. 2024) environment captions to guide the local spatial understanding based on an identification mechanism. After that, the local-aware global spatial understanding takes the multimodal and multi-scale features as input and progressively integrates spatial environment knowledge. In this way, $\mathbf { M } ^ { \mathrm { 2 } } \mathbf { S } \mathbf { E } .$ -

VTTS effectively models the interactions between local and global spatial contexts among the multi-modal spatial environment. The main contributions of this paper include:

• We propose a novel multi-modal and multi-scale spatial environment understanding framework, termed $\mathrm { \bf { M } ^ { 2 } { S E } } .$ - VTTS, that leverages both the RGB and Depth information to enhance the synthesis of immersive reverberation speech.   
• Our approach comprehensively integrates both local and global spatial elements, providing a more comprehensive understanding of the spatial environment, which is crucial for accurately modeling environmental reverberation.   
• Objective and subjective experimental evaluations demonstrate that our model significantly outperforms all existing state-of-the-art benchmarks in generating environmental speech.

# Related Works Spatial Environment Understanding

Spatial environment understanding plays a crucial role in spatial cognition, particularly in complex three-dimensional scenes where accurate spatial comprehension is essential for applications such as robotic navigation, augmented reality, and autonomous driving. In the visual domain, researchers often employ multi-modal and multi-scale approaches to capture and analyze spatial information more comprehensively (Chen et al. 2020; Guo et al. 2022; Jain et al. 2023; Jiang et al. 2024; Xu et al. 2024; Wang et al. 2024). For instance, Cheng et al. (2024a) enhances Vision Language Models (VLMs) by introducing a data curation pipeline and a plugin module that improves the understanding of 3D spatial relationships by integrating depth information and learning regional representations from 3D scene graphs. Fu et al. (2024) further extends the capabilities of language models by incorporating 3D visual data, improving the embodied agents’ reasoning and decision-making abilities in interactive 3D environments. This approach is particularly effective in tasks such as dense annotation and interactive planning. Similarly, Cheng et al. (2024b) addresses the challenge of depth estimation in autonomous driving by proposing a method that fuses single-view and multi-view depth estimates, showing robust performance in scenarios with sparse textures and dynamic objects. These studies demonstrate that multi-modal and multi-scale methods are pivotal in advancing the comprehensive understanding of spatial environments, as they effectively integrate information from various sources to enhance spatial reasoning in complex environments.

While these works have made significant strides in improving spatial understanding in VLMs, they primarily focus on extracting global spatial information, often overlooking the importance of local and depth information. Our work differs from these approaches in several key aspects: (1) We focus on the Visual VTTS task, rather than solely on visual spatial reasoning; (2) Unlike the previous works, we emphasize the integration of both local and depth image information, in addition to global spatial data from RGB images, to achieve a more holistic understanding of the spatial environment. These differences allow our approach to better address the unique challenges of the VTTS task and excel in complex spatial environments.

# LLM-based Image Understanding

In recent years, leveraging Large Language Models (LLMs) for image understanding has emerged as a significant research focus in the fields of computer vision and natural language processing. By integrating the powerful natural language processing capabilities of LLMs with visual information, researchers have developed multi-modal large language models (MLLMs) to tackle complex tasks such as visual question answering, image captioning, and image comprehension (Zhang et al. 2024; Zhu, Wei, and Lu 2024). For example, Karthik et al. (2024) pioneered the combination of pre-trained vision encoders with language models, utilizing a Perceiver Resampler module to extract features from images for generating textual descriptions, thus achieving cross-modal image-text alignment. Building on this, Swetha et al. (2024) introduced a Querying Transformer (Q-Former) that extracts the most relevant visual features through crossmodal fusion, enhancing the model’s visual understanding capabilities. Chowdhury et al. (2024) further advanced these efforts by incorporating spatial coordinate information, thereby improving multi-modal models’ abilities in object localization and visual reasoning. However, these approaches largely focus on capturing global visual features, which, while effective in many cases, show limitations when dealing with tasks that require fine-grained visual understanding. To address this challenge, Swetha et al. (2024) proposed a novel approach that combines contrastive learning (CL) with masked image modeling (MIM), integrating features from both CLIP-ViT (Radford et al. 2021) and MAEViT vision encoders. X-Former uses a dual cross-attention mechanism to align visual and language features, demonstrating superior performance in fine-grained visual tasks, such as object counting and fine-grained category recognition. In contrast, traditional multi-modal models often struggle with these tasks due to their limited ability to capture local details effectively.

While these methods have significantly advanced the capabilities of multi-modal visual understanding, they still face limitations when applied to specific spatial environment perception tasks. Our work introduces a novel multimodal and multi-scale spatial environment understanding scheme, designed to overcome the shortcomings of existing models in capturing the local and depth information. Unlike previous approaches, our method integrates both RGB and depth information and utilizes Gemini-generated environmental captions to guide local spatial understanding. By fusing multi-modal and multi-scale features, our method provides a more comprehensive modeling of spatial environments, offering robust support for VTTS tasks, particularly in understanding the spatial layout and environmental characteristics of complex scenes.

2. Local Spatial Understanding 3. Local-aware Global Spatial Understanding 4. Speech Generation   
DReGpBthPatactchhEmbbeeddidningg CLS ★ \*\* CLS 招谐 Topk Depth Q HD Semantic- HG 2 SRelgeicotnors AtDtenptihon K,V AGtDtueinpdteihodn Adaptor & Denoiser FR share HTOpK FS Q ④ 4 Hv Visual-text Encoder RGB Image Patchification Image Encoder CLS Q FARA FP Topk RGB Q Local-Aware H SeGmuiadnetidc- Embedding Y L DRetgeicotnosr AttReGntBion K,V AttReGntBion HG   
Gemini Pro Vison Spatial Environment Spatial Environment Y CLIP-ViT F HTOPK K,v Phonemization Prompt Captions Text Encoder CLS Spoken Content

# Methodology

As shown in the pipeline of Fig. 1, the proposed $\mathbf { M } ^ { 2 } \mathbf { S } \mathbf { E } .$ - VTTS consists of four components: 1) Multi-modal Features Extraction; 2) Local Spatial Understanding; 3) Localaware Global Spatial Understanding and 4) Speech Generation. As mentioned previously, multi-modal features, including the RGB and Depth space representation of an image, can provide more comprehensive information about the spatial environment. To understand the interactions between local and global spatial contexts, the multi-modal and multiscale knowledge is integrated by local-aware global spatial understanding. The following subsections provide detailed descriptions of the design and training processes for these components.

# Multi-modal Features Extraction

Given the RGB and Depth image pairs of the spatial environment $\{ \gamma _ { R } , \gamma _ { D } \}$ , we first partition them into $M$ patches. In addition, we employ the image encoder of a pre-trained CLIP (Radford et al. 2021) model with frozen parameters to extract patch-level features as $\mathcal { F } _ { P } ^ { R }$ , $\mathcal { F } _ { P } ^ { D } \in \mathcal { R } ^ { \dot { M } \times D }$ from each of $\nu _ { R }$ and $\nu _ { D }$ , where $D$ denotes the dimensionality of the features and $M$ indicates the number of patches per image. As illustrated in Fig 1, a special $\left[ C L S \right]$ token is used at the beginning of the first patch to represent the global-level features as $\check { \mathcal { F } } _ { G } ^ { R }$ , $\mathcal { F } _ { G } ^ { D } \in \mathcal { R } ^ { \mathrm { i } \times D }$ .

# Local Spatial Understanding

As shown in the second panel of Fig. 1, Local Spatial Understanding consists of three parts: 1) LLM-based Spatial Semantic Understanding, leveraging Gemini’s powerful multimodal understanding capabilities to accurately convert complex visual scenes into semantic information; 2) ${ \mathrm { T o p } } _ { k }$ RGB Regions Detector, guided by environmental captions to identify crucial semantic information of the RGB space of the image; and 3) ${ \mathrm { T o p } } _ { k }$ Depth Regions Selector, selecting important semantic information of the Depth space of the image.

LLM-based Spatial Semantic Understanding To capture rich spatial information, including the spatial positions of objects, their arrangement, and the overall scene structure, we utilize Gemini’s advanced multi-modal understanding capabilities to convert the complex visual data into the structured caption. This approach enables us to accurately extract and represent the spatial semantics embedded within the image.

First of all, the spatial environment captions are generated using the Gemini Pro Vision, which is a multi-modal large language model configured with its default settings. The prompt designed for Gemini is as follows: “Observe this panoramic image and briefly describe its content. Identify the objects in the image in one to two sentences, focusing only on key information and avoiding descriptive words.” After analysis by Gemini, the spatial environment in Fig. 1 is described as follows: “The image shows a spacious, circular room with a blue and white color scheme. It features a dining table with chairs, a kitchenette, a bedroom area with a bed, and a person standing in the center of the room.” In the end, the caption $\mathcal { C }$ is tokenized into $N$ individual words, represented as $\mathcal { C } = \{ c _ { n } \} _ { n = 1 } ^ { N }$ . And each word $c _ { n }$ is represented as a fixed-length vector using word embeddings, which are input into the text encoder of a pre-trained CLIP model to obtain spatial semantic features $\star _ { S } ^ { C }$ , where $\mathcal { F } _ { S } ^ { C } \in \mathcal { R } ^ { 1 \times D }$ . It is important to note that the $\left[ C L \breve { S } \right]$ token is used to aggregate and represent the overall semantic information of the entire input text, with this embedding vector serving as the primary representation of the text when aligning with image features.

$\mathbf { T o p } _ { k }$ RGB Regions Detector Our goal is to identify and focus on the image regions that significantly influence sound propagation and reflection characteristics, enabling more accurate simulation of the reflection and absorption effects of different materials and surfaces, thereby making the generated speech more natural and realistic.

To begin with, we apply the spatial attention to $\mathcal { F } _ { P } ^ { R }$ and $\mathcal { F } _ { S } ^ { C }$ after using a linear projection layer, which is formalized as:

$$
\hat { \mathcal { F } } _ { P } ^ { R } , \mathcal { A } _ { P } ^ { R } = M u l t i H e a d ( \mathcal { F } _ { S } ^ { C } , \mathcal { F } _ { P } ^ { R } , \mathcal { F } _ { P } ^ { R } ) ,
$$

where $\hat { \mathcal { F } } _ { P } ^ { R }$ represents the updated features from $\mathcal { F } _ { P } ^ { R }$ , and $\ v { A } _ { P } ^ { R }$ denotes the average attention weights across all heads, with $\mathcal { A } _ { P } ^ { R } \in ( 0 , 1 ) ^ { M }$ . Inspired by SRSM (Li, Hou, and $\mathrm { H u } \ 2 0 2 3 \$ ), after that, we introduce a detection operation, denoted as $\Phi _ { L S U }$ to identify the patches with the highest ${ \mathrm { T o p } } _ { k }$ attention weights and their indices:

$$
\mathcal { H } _ { T o p _ { k } } ^ { R } , \Omega _ { R } = \Phi _ { L S U } ( \hat { \mathcal { F } } _ { P } ^ { R } , \mathcal { A } _ { P } ^ { R } , T o p _ { k } ) ,
$$

where HTR op $\mathcal { H } _ { T o p _ { k } } ^ { R } \in \mathcal { R } ^ { T o p _ { k } \times D }$ represents the detected local features of the RGB space, and $\Omega _ { R } \in \{ 0 , 1 , \ldots , M \} ^ { T o p _ { k } }$ represents the indices corresponding to the highest $\mathrm { T o p } _ { k }$ weights in $\mathbf { \mathcal { A } } _ { P } ^ { R }$ .

$\mathbf { T o p } _ { k }$ Depth Regions Selector This module aims to capture the relative distances of key objects, their arrangement, and the geometric layout of the room within the spatial environment, and to accurately simulate sound propagation and reflection, thereby generating reverberation that more closely aligns with the actual physical space. This module implements a selection attention-based strategy similar to ΦLSU.

Specifically, we take the indices $\Omega _ { R }$ from $\Phi _ { L S U }$ to select the corresponding crucial patch-level depth features. This approach is based on the following three key considerations: 1) the CLIP is pre-trained using RGB images paired with text, resulting in a stronger correlation between RGB and textual data compared to Depth information; 2) by maintaining consistent patch indices across both RGB and Depth modalities, we ensure spatial coherence, allowing the model to accurately align and integrate features from the same spatial locations; and 3) this alignment further prevents potential information redundancy or conflicts between the modalities, ensuring that the model is better equipped to precisely capture and utilize complementary features from both RGB and Depth data. This process can be formulated as:

Local-aware RGB/Depth Attention This section aims to understand how local spatial details, such as the position and material of key objects, interact within the overall spatial layout and to comprehend the spatial relationships across different scales in the scene, thereby enabling the generation of reverberation that more accurately reflects the actual

For the RGB image, given its $\mathcal { H } _ { T o p _ { k } } ^ { R }$ and $\mathcal { F } _ { G } ^ { R }$ , we perform the Local-aware RGB Attention to model the interactions between the local and global spatial knowledge of the RGB space after using a linear projection layer, which is formulated as follows:

$$
\mathcal { H } _ { L } ^ { R } = M u l t i H e a d ( \mathcal { H } _ { T o p _ { k } } ^ { R } , \mathcal { F } _ { G } ^ { R } , \mathcal { F } _ { G } ^ { R } ) ,
$$

where $\mathcal { H } _ { L } ^ { R } \in \mathcal { R } ^ { T o p _ { k } \times D }$ is updated from $\mathcal { F } _ { G } ^ { R }$ .

For tHhLe ∈DeRpth image, giving its HTDF oGpk and $\mathcal { F } _ { G } ^ { D }$ , the Local-aware Depth Attention adopts a similar strategy, which is formulated as follows:

$$
\mathcal { H } _ { L } ^ { D } = M u l t i H e a d ( \mathcal { H } _ { T o p _ { k } } ^ { D } , \mathcal { F } _ { G } ^ { D } , \mathcal { F } _ { G } ^ { D } ) ,
$$

where $\mathcal { H } _ { L } ^ { D } \in \mathcal { R } ^ { T o p _ { k } \times D }$ is updated from $\mathcal { F } _ { G } ^ { D }$ .

Semantic-Guided RGB/Depth Attention To deepen our understanding of the complex relationships between spatial contexts across different scales and to enhance the model’s performance in the multi-modal environment, we further employ a semantic-guided attention mechanism to achieve a more advanced fusion of local and global spatial features.

For the RGB image, given its $\mathcal { H } _ { L } ^ { R }$ and $\mathcal { F } _ { S } ^ { \dot { C } }$ , we adopt the Semantic-Guided RGB Attention to attain an advanced understanding between the local and global spatial contexts following a linear projection layer, which is formulated as follows:

$$
\mathcal { H } _ { G } ^ { R } = M u l t i H e a d ( \mathcal { F } _ { S } ^ { C } , \mathcal { H } _ { L } ^ { R } , \mathcal { H } _ { L } ^ { R } ) ,
$$

where $\mathcal { H } _ { G } ^ { R } \in \mathcal { R } ^ { 1 \times D }$ is updated from $\mathcal { F } _ { S } ^ { C }$

For the Depth image, the Semantic-Guided Depth Attention employs a similar method to learn an advanced understanding of the Depth space, which is formulated as follows:

$$
\mathcal { H } _ { G } ^ { D } = M u l t i H e a d ( \mathcal { F } _ { S } ^ { C } , \mathcal { H } _ { L } ^ { D } , \mathcal { H } _ { L } ^ { D } )
$$

Eventually, we integrate the multi-modal and multi-scale features to derive a comprehensive representation of the spatial environment, which is formulated as follows:

$$
\mathcal { H } _ { V } = \lambda _ { 1 } \mathcal { H } _ { G } ^ { R } + \lambda _ { 2 } \mathcal { H } _ { G } ^ { D } ,
$$

$$
\mathcal { H } _ { T o p _ { k } } ^ { D } = \Psi _ { L S U } ( \mathcal { F } _ { P } ^ { D } , \Omega _ { R } ) ,
$$

where $\mathcal { H } _ { T o p _ { k } } ^ { D }$ represents the selected local features of the Depth space,kand HTD opk $\mathcal { H } _ { T o p _ { k } } ^ { D } \in \mathcal { R } ^ { T o p _ { k } \times D }$

# Local-aware Global Spatial Understanding

As shown in the third panel of Fig. 1, Local-aware Global Spatial Understanding aims to effectively model the interactions between local semantics and the global spatial context, which consists of two parts: 1) the Local-aware RGB/Depth Attention, which focuses on learning the interactions between local details and global spatial features, and 2) the Semantic-Guided RGB/Depth Attention, which enhances the understanding of spatial contexts by integrating semantic information with the local-aware global features.

where the weights, $\lambda _ { 1 }$ and $\lambda _ { 2 }$ , are both set to 0.5.

# Speech Generation

As illustrated in Fig. 1, we adopt ViT-TTS as the backbone for our TTS system. To begin with, the phoneme embeddings and visual features are converted into hidden sequences. In addition, the variance adaptor predicts the duration of each hidden sequence to regulate the length of the hidden sequences to match that of speech frames. After that, different variances like pitch and speaker embedding are incorporated into hidden sequences following Ren et al. (2021). Furthermore, the spectrogram denoiser iteratively refines the length-regulated hidden states into melspectrograms. In the end, the BigVGAN (Lee et al. 2022) transforms mel-spectrograms into waveform. For more details, please refer to the ViT-TTS (Liu et al. 2023b).

# Experiments and Results

# Dataset

We employ the SoundSpaces-Speech dataset (Chen et al. 2023), which is developed on the SoundSpaces platform using real-world 3D scans to simulate environmental audio. To enhance the dataset, we refine it following the approach described in Chen et al. (2022); Liu et al. (2023b). Specifically, we exclude out-of-view samples and divide the remaining data into two subsets: test-unseen and testseen. The test-unseen subset includes room acoustics derived from novel images, while the test-seen subset contains scenes previously observed during training. The dataset consists of 28,853 training samples, 1,441 validation samples, and 1,489 testing samples. Each sample includes clean text, reverberation audio, and panoramic camera RGB-D images. To preprocess the text, we convert the sequences into phoneme sequences using an open-source grapheme-tophoneme tool 1.

Following common practices (Ren et al. 2019; Huang et al. 2022; Liu et al. 2024b,a), we preprocess the speech data in three steps. First, we extract spectrograms with an FFT size of 1024, a hop size of 256, and a window size of 1024 samples. Next, we convert the spectrogram into a mel-spectrogram with 80 frequency bins. Finally, we extract the F0 (fundamental frequency) from the raw waveform using Parselmouth 2. These preprocessing steps ensure consistency with prior work and prepare the data for subsequent modeling.

# Implementation Details

For the visual modality, we utilize the pre-trained CLIP-ViTL/14 as the visual feature extractor. This model generates 768-dimensional feature vectors at both global and patch levels for each visual snippet. These visual features undergo a linear transformation and are subsequently aligned with the 512-dimensional hidden space of the phoneme embeddings. The phoneme vocabulary consists of 74 distinct phonemes. The cross-modal fusion module employs two attention heads, while all other attention mechanisms use four heads each. The patch number, ${ \mathrm { T o p } } _ { k }$ , is set to 140. The configuration of other encoder parameters follows the implementation in ViT-TTS. In the denoiser module, we use five transformer layers with a hidden size of 384 and 12 heads. Each transformer block functions as the identity, with $T$ set to 100 and $\beta$ values increasing linearly from $\dot { \beta } _ { 1 } ~ = ~ 1 0 ^ { - 4 }$ to $\beta _ { T } = 0 . 0 6$ . This configuration facilitates effective noise reduction and enhances the quality of the generated outputs.

The training process consists of two stages. In the pretraining stage, we adopt the encoder pre-training strategy from ViT-TTS, training the encoder for $1 2 0 \mathrm { k }$ steps until convergence. In the main training stage, the $\mathbf { M } ^ { 2 }$ SE-VTTS model is trained on a single NVIDIA A800 GPU with a batch size of 48 sentences, extending over 160k steps until convergence. During inference, we use a pre-trained BigVGAN as the vocoder to transform the generated mel-spectrograms into waveforms. Further details on the model configuration and implementation are provided in Appendix A 3.

# Evaluation Metrics

We measure the sample quality of the generated waveform using both objective metrics and subjective indicators. The objective metrics are designed to evaluate various aspects of waveform quality by comparing the ground-truth audio with the generated samples. Following the common practice of Liu et al. (2022); Huang et al. (2022), we randomly select 50 samples from the test set for objective evaluation. We provide three main metrics: (1) Perceptual Quality: This is assessed by human listeners using the Mean Opinion Score (MOS). A panel of listeners evaluates the audio’s quality, naturalness, and its congruence with the accompanying image. Ratings are assigned on a scale from 1 (poor) to 5 (excellent). The final MOS is the average of these ratings. (2) Room Acoustics (RT60 Error): RT60 measures the reverberation time in seconds for an audio signal to decay by $6 0 ~ \mathrm { d B }$ , which is a standard metric for characterizing room acoustics. To calculate the RT60 Error (RTE), we estimate the RT60 values from the magnitude spectrograms of the output audio, using a pre-trained RT60 estimator provided by Chen et al. (2022). (3) Mel Cepstral Distortion (MCD): MCD quantifies the spectral distance between the synthesized and reference mel-spectrogram features. It is widely used as an objective measure of audio quality, particularly in tasks involving speech synthesis. Lower MCD values indicate higher spectral similarity between the generated and ground-truth audio.

Each of these metrics provides a distinct perspective on the quality of the generated waveform, allowing for a comprehensive evaluation of the system’s performance.

# Baselines

To demonstrate the effectiveness of our $\mathbf { M } ^ { 2 }$ SE-VTTS, we compare it against five baseline systems:

• ProDiff (Huang et al. 2022): This first baseline is a progressive fast diffusion model designed for high-quality speech synthesis, where the input is text and the model directly predicts clean mel-spectrograms, significantly reducing the required sampling iterations. • DiffSpeech (Liu et al. 2022): This method is a TTS model that employs a diffusion probabilistic approach, where the input is text and the model iteratively converts noise into mel-spectrograms conditioned on the text. • VoiceLDM (Lee et al. 2024): The third system is a TTS model that uses text as its primary input, effectively capturing global environmental context from descriptive prompts to generate audio that aligns with both the content and the overarching situational description. Given the differences in environmental text descriptions between the training datasets—where the original dataset primarily describes the type of environment, while ours emphasizes the specific components and their spatial relationships—we choose to concentrate on the model’s novel method of leveraging textual descriptions to guide the synthesis of reverberation speech during code reproduction.

<html><body><table><tr><td rowspan="2"></td><td colspan="2">Test-Unseen</td><td rowspan="2">MCD (↓)</td><td colspan="3">Test-Seen</td></tr><tr><td>MOS (↑)</td><td>RTE (↓)</td><td>MOS (↑)</td><td>RTE (↓)</td><td>MCD (↓)</td></tr><tr><td>GT</td><td>4.353± 0.023</td><td>/</td><td>/</td><td>4.348 ± 0.022</td><td></td><td>/</td></tr><tr><td>GT(voc.)</td><td>4.149 ± 0.027</td><td>0.0080</td><td>1.4600</td><td>4.149 ± 0.023</td><td>0.0060</td><td>1.4600</td></tr><tr><td>ProDiff (Huang et al. 2022)</td><td>3.550± 0.023</td><td>0.1341</td><td>4.7689</td><td>3.647± 0.023</td><td>0.1243</td><td>4.6711</td></tr><tr><td>DiffSpeech (Liu etal. 2022)</td><td>3.649 ± 0.022</td><td>0.1193</td><td>4.7923</td><td>3.675 ± 0.011</td><td>0.1034</td><td>4.6630</td></tr><tr><td>VoiceLDM (Lee et al. 2024)</td><td>3.702 ± 0.020</td><td>0.0825</td><td>4.8952</td><td>3.702 ± 0.025</td><td>0.0714</td><td>4.6572</td></tr><tr><td>ViT-TTS-ResNet18 (Liu etal.2023b)</td><td>3.700 ± 0.025</td><td>0.0759</td><td>4.5933</td><td>3.804 ± 0.022</td><td>0.0677</td><td>4.5535</td></tr><tr><td>ViT-TTS-CLIP (Liu et al. 2023b)</td><td>3.651 ± 0.023</td><td>0.0772</td><td>4.5871</td><td>3.746 ± 0.023</td><td>0.0678</td><td>4.5385</td></tr><tr><td>M²SE-VTTS</td><td>3.849 ± 0.025</td><td>0.0744</td><td>4.4215</td><td>3.939 ± 0.022</td><td>0.0642</td><td>4.3809</td></tr></table></body></html>

Table 1: Comparison with baselines on the SoundSpaces-Speech for Seen and Unseen scenarios. Subjective (with $9 5 \%$ confidence interval) and objective results with the different systems.

• ViT-TTS-ResNet18 (Liu et al. 2023b): The fourth baseline is a VTTS model that takes both text and environmental images as inputs, leveraging ResNet18 (He et al. 2016) to extract global visual features from the image to enhance audio generation by capturing the room’s acoustic characteristics.

• ViT-TTS-CLIP (Liu et al. 2023b): The last system is also ViT-TTS, which utilizes CLIP-ViT as a global RGB feature extractor.

# Main Results

As shown in Table 1, the performance of the $\mathbf { M } ^ { 2 }$ SE-VTTS model on the test-unseen set is generally lower than that on the test-seen set, largely due to the presence of scenarios not encountered during training. Nevertheless, our model consistently outperforms all baseline systems across both sets, achieving the best results in RTE (0.0744), MCD (4.4215), and MOS $( 3 . 8 4 9 \pm 0 . 0 2 5 )$ . These results demonstrate that our model is capable of synthesizing immersive reverberant speech. In addition, our model outperformed TTS diffusion models, such as DiffSpeech and ProDiff, across all metrics, notably in RTE. This indicates that traditional TTS models struggle to understand spatial environment information, focusing instead on audio content, pitch, and energy. To address this limitation, our multi-modal scheme learns more comprehensive spatial information. Furthermore, comparison with voiceLDM highlights the advantages of the multi-modal spatial cues and Gemini-based spatial environment understanding. Although voiceLDM takes environmental context descriptions as prompts to synthesize environmental audio, its choice of the spatial prompt and lack of a spatial semantic understanding strategy result in worse performance in predicting the correct reverberation and synthesizing high-quality audio with perceptual accuracy. Finally, ViT-TTS, which uses ResNet18 for global visual feature extraction, and ViT-TTS-CLIP, which employs CLIPViT, both outperform other baseline models. However, compared to our proposed model, both ViT-TTS and ViT-TTSCLIP showed inferior performance in both test-unseen and test-seen environments. This suggests that our accurate modeling of the interaction between crucial local regions and the global context is effective, achieved by integrating knowledge gained from local spatial understanding.

Table 2: Ablation study results. The results of $\mathbf { M } ^ { 2 }$ SE-VTTS are sourced from Table 1.   

<html><body><table><tr><td>System</td><td>MOS (↑)</td><td>RTE (↓)</td><td>MCD (↓)</td></tr><tr><td>GT(voc.)</td><td>4.149 ± 0.027</td><td>0.0080</td><td>1.4600</td></tr><tr><td>W/o RGB</td><td>3.716 ± 0.049</td><td>0.0985</td><td>4.6378</td></tr><tr><td>w/o Depth</td><td>3.753 ± 0.022</td><td>0.0957</td><td>4.6808</td></tr><tr><td>w/o LLM</td><td>3.749 ± 0.026</td><td>0.0881</td><td>4.6121</td></tr><tr><td>w/o LSU</td><td>3.753 ± 0.025</td><td>0.0984</td><td>4.6238</td></tr><tr><td>W/o LGSU-L</td><td>3.698 ± 0.043</td><td>0.1011</td><td>4.6939</td></tr><tr><td>W/o LGSU-G</td><td>3.703 ± 0.046</td><td>0.1039</td><td>4.7706</td></tr><tr><td>M²SE-VTTS</td><td>3.849 ± 0.025</td><td>0.0744</td><td>4.4215</td></tr></table></body></html>

In conclusion, our comprehensive evaluation results demonstrate the effectiveness of our proposed scheme in generating reverberant speech that matches the target environment.

# Ablation Results

To evaluate the individual effects of several key techniques on the Test-Unseen set in our model, including the RGB space (RGB), the Depth space (Depth), the Gemini-based spatial semantic understanding (LLM), the local spatial understanding (LSU), local-aware interactions (LGSU-L), and global knowledge interactions (LGSU-G), we remove these components to build various systems. A series of ablation experiments were conducted, and the subjective and objective results are shown in Table 2.

We find that removing different types of modality information (w/o RGB and w/o Depth) in the Multi-modal Features Extraction led to a decrease in performance across most objective metrics, and the subjective MOS scores also dropped. This suggests that our multi-modal strategy can learn more comprehensive spatial information and enhance the expressiveness of reverberation.

In addition, to validate the Gemini-based spatial semantic understanding (LLM), we remove this component (w/o LLM). As shown in Table 2, the removal of the semantic

4.65 0.0925 4.60 Shared Indindices 0.0900 M 0.0875 4.55 0.0850 0 4.50 E 0.0825 4.45 R 0.0800 0.0775 4.40 0.0750 Shared Indices 4.35 0.0725 Unshared Indices 0.0700 20 40 60 80100 120140 160 180 200 220 240 20 40 60 80100120140160180 200 220 240 (a)Selected $\mathrm { T o p } _ { k }$ Patches (b) Selected $\mathrm { T o p } _ { k }$ Patches

understanding component led to a reduction in all subjective and objective metrics. This demonstrates that spatial images, when analyzed by Gemini for semantic understanding, enable our model to achieve a more accurate representation of reverberation.

Furthermore, we further explored the ${ \mathrm { T o p } } _ { k }$ region selection for RGB/Depth (w/o LSU). Omitting these critical regions leads to a decrease in both subjective and objective metrics. This suggests that by identifying important semantic information, the model can accurately understand the spatial environment and improve the style and quality of the reverberation.

Finally, we remove local semantics (w/o LGSU-L) and global context (w/o LGSU-G) in the Local-aware Global Spatial Understanding component. This removal results in decreased performance across both subjective and objective metrics, underscoring the efficacy of our multi-modal and multi-scale approach in modeling the interplay between local and global spatial contexts for reverberation.

# $\mathbf { T o p } _ { k }$ Index Sharing Comparative Study

To evaluate the effectiveness of selecting depth features using shared ${ \mathrm { T o p } } _ { k }$ indices from the RGB image and to compare this approach with independent semantic-guided methods, we focus on the efficacy of index-sharing and the impact of different ${ \mathrm { T o p } } _ { k }$ values. Specifically, we design experiments to compare two feature selection strategies: shared ${ \mathrm { T o p } } _ { k }$ indices and unshared ${ \mathrm { T o p } } _ { k }$ indices.

In the shared indices strategy, ${ \mathrm { T o p } } _ { k }$ critical regions from RGB images guide depth feature selection. In contrast, the unshared $\mathrm { T o p } _ { k }$ indices strategy independently selects features from RGB and depth images based on their respective semantic information. Various $\mathrm { T o p } _ { k }$ values (e.g., 20, 40, ..., 240) are tested to observe their effects on performance, evaluated using two objective metrics, with lower values indicating better performance.

The experimental results demonstrate that the shared ${ \mathrm { T o p } } _ { k }$ strategy consistently outperforms the unshared ${ \mathrm { T o p } } _ { k }$ strategy for all tested values of ${ \mathrm { T o p } } _ { k }$ . This approach yields lower values for the objective metrics, indicating more natural audio generation that better aligns with the environment. As the ${ \mathrm { T o p } } _ { k }$ value increases, performance improves as more comprehensive spatial information is captured, though it plateaus or declines slightly after reaching a threshold (e.g., 140).

The comparison of strategies confirms that shared ${ \mathrm { T o p } } _ { k }$ indices from RGB more effectively capture critical spatial information, leading to more realistic and environmentconsistent audio. This suggests that the shared index strategy is superior for depth feature selection in multi-modal tasks, providing guidance for improving feature selection in future multi-modal systems. Further refinement of this index-sharing approach is recommended to maximize performance.

# Conclusion

This paper introduces $\mathbf { M } ^ { \mathrm { 2 } } \mathbf { S } \mathbf { E }$ -VTTS, an innovative multimodal and multi-scale approach for Visual Text-to-Speech (VTTS) synthesis. Our method addresses the limitations of previous VTTS systems, such as their limited spatial understanding, by incorporating both RGB and depth images to achieve a comprehensive representation of the spatial environment. This comprehensive spatial representation includes modeling both local and global spatial contexts, which are crucial for capturing the environmental nuances influencing speech reverberation. By combining local spatial understanding guided by the environment caption and local-aware global spatial modeling, $\mathbf { M } ^ { 2 }$ SE-VTTS effectively captures the interactions between different spatial scales, which are essential for accurate reverberation modeling. Evaluations demonstrate that our model consistently outperforms stateof-the-art benchmarks in generating environmental speech, establishing a new standard for environmental speech synthesis in VTTS.

Despite its advances, the $\mathbf { M } ^ { 2 }$ SE-VTTS framework has limitations, such as increased computational complexity from multi-modal and multi-scale feature integration, potentially hindering real-time applications. Additionally, the model’s performance is inconsistent in unseen environments, highlighting the need for improved generalization. Future research should focus on optimizing computational efficiency and enhancing the model’s adaptability to unseen spatial contexts.