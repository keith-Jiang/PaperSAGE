# FaceSpeak: Expressive and High-Quality Speech Synthesis from Human Portraits of Different Styles

Tian-Hao Zhang1,\*, Jiawei Zhang1, Jun Wang2, Xinyuan Qian1,†, Xu-Cheng Yin1

1School of Computer and Communication Engineering, University of Science and Technology Beijing, Beijing, China 2Tencent AI Lab, Shenzhen, China tianhaozhang@xs.ustb.edu.cn, jiaweizhang@xs.ustb.edu.cn

# Abstract

Humans can perceive speakers’ characteristics (e.g., identity, gender, personality and emotion) by their appearance, which are generally aligned to their voice style. Recently, visiondriven Text-to-speech (TTS) scholars grounded their investigations on real-person faces, thereby restricting effective speech synthesis from applying to vast potential usage scenarios with diverse characters and image styles. To solve this issue, we introduce a novel FaceSpeak approach. It extracts salient identity characteristics and emotional representations from a wide variety of image styles. Meanwhile, it mitigates the extraneous information (e.g., background, clothing, and hair color, etc.), resulting in synthesized speech closely aligned with a character’s persona. Furthermore, to overcome the scarcity of multi-modal TTS data, we have devised an innovative dataset, namely Expressive Multi-Modal TTS $\left( \mathrm { E M ^ { 2 } T T S } \right)$ ), which is diligently curated and annotated to facilitate research in this domain. The experimental results demonstrate our proposed FaceSpeak can generate portraitaligned voice with satisfactory naturalness and quality.

Demos — https://facespeak.github.io

# Introduction

Human voices contain munificent information in aspects such as age (Grzybowska and Kacprzak 2016; Singh et al. 2016), gender (Li et al. 2019), emotional nuances (Wang and Tashev 2017; Zhang, Wu, and Schuller 2019), physical fitness (Verde et al. 2021), and speaker identity (Deaton 2010; Ravanelli and Bengio 2018). These vocal characteristics are intrinsically related to an individual’s physical and psychological makeup (Xu et al. 2024; Hardcastle, Laver, and Gibbon 2012), offering a unique profile of the speaker. For example, the emotional content conveyed through speech is often mirrored in facial expressions. This inherent correlation between voice and image sparked research exploration in various fields, including emotion recognition (Zhou et al. 2021; Lei and Cao 2023; Zhang et al. 2021), speaker verification (Qian, Chen, and Wang 2021; Nawaz et al. 2021), face-speech retrieval (Li et al. 2023a), and speech separation (Gao and Grauman 2021; Lee et al. 2021).

![](images/d5775a72b5fce15c44cd384c769ad77a2492ae494df61964658d84a2ae974e01.jpg)  
Figure 1: Our proposed multi-modal speech synthesis framework, namely FaceSpeak, which performs expressive and high-quality speech synthesis, given image prompt of different styles and the content text (Note: image-speech data from various characters are encoded with distinct color codecs).

In recent years, growing research interest has focused on the controllability of synthesized speeches. To this end, one solution is to introduce an auxiliary reference input to model the style features of speech. For example, PromptTTS (Guo et al. 2023) and InstructTTS (Yang et al. 2023) leverage textual descriptions to control the speech synthesis style. In contrast, the input text description still needs human crafting efforts and expertise, while some individuals may struggle to accurately express their intended synthesis goals. Other works employ vision as a reference. For example, visualTTS (Lu et al. 2022) generates temporal synchronized speech sequences for visual dubbing, and MM-TTS (Guan et al. 2024) transfers multi-modal prompts (i.e., text, human face, pre-recorded speech) into a unified style representation to control the generation process. Despite both works using images (real human faces) as the reference, thus cannot adapt to nonphotorealistic portraits which are widespread in digital assistants, video games, and virtual reality scenarios.

Although previous attempts have been made to generate speech based on visual cues, they have faced notable limitations. One such limitation is the predominant reliance on real-face datasets, which lack diversity in image styles necessary for comprehensive speech synthesis guidance. This significantly restricts the potential applications of synthesizing speech from portrait images. Additionally, prior methods often employ entangled embeddings of facial images to guide speech synthesis, potentially introducing extraneous information that hampers performance. Moreover, relying on entangled visual features can further constrain the flexibility of the synthesis system, as the synthesized speech in this case can only be controlled by a single image. However, decoupling identity and emotion features can enable the control of speech synthesis by using different images providing identity and emotion information separately, greatly increasing the diversity and flexibility of multi-modal TTS.

In this paper, we tackle the aforementioned issues through a novel multi-modal speech synthesis process. As shown in Fig. 1, given the text of the content and the images in different styles (e.g., fantasy art and cartoon), our aim is to generate high-quality vivid human speech that is aligned with the characteristics indicated by vision. Our key contributions are summarized as follows.

1. We introduce $\mathrm { E M ^ { 2 } T T S }$ , a pioneering multi-style, multimodal TTS dataset. It is designed and re-annotated through a collaborative multi-agent framework which leverages chatGPT1 for crafting intermediate textual descriptions, PhotoMaker (Li et al. 2023b) for generating human portraits from text, and DALL-E to establish multi-modal coherence. It provides large-scale and diverse style images that enable the training model to generate high-quality and image-coherent speech, thus facilitates the state-of-the-art (SOTA) multi-modal TTS development.

2. We propose a novel speech synthesis method given human portrait prompt, namely FaceSpeak, to generate speech that is aligned with the characteristics indicated by the visual input. To our best knowledge, this is the first multi-modal expressive speech synthesis work that allows input any-style images. In particular, we disentangle the identity and expression features from facial images, ensuring that the synthesized audio aligns with the speaker’s characteristics, while mitigating the impact of irrelevant factors in the images and enhancing the flexibility and diversity of synthesis systems.

3. Extensive experiments demonstrate that our proposed FaceSpeak can synthesize image-aligned, high-quality, diverse, and expressive human speech. Its superior performance is also validated through the numerous subjective and objective evaluations.

# Related Work

Existing TTS works utilizing prompts leverage a multitude of modalities, including reference speech, textual descriptions, and human faces.

Speech prompt: Traditional TTS system extracts features from a reference speech to obtain the desired voice with unique vocal characteristics. For example, MetaStyleSpeech (Min et al. 2021), which is built on FastSpeech2 (Ren et al. 2020), fine-tunes the gain and bias of textual input based on stylistic elements extracted from a speech reference, facilitating effective style-transferred speech synthesis. YourTTS (Casanova et al. 2022), based on

VITS, proposes modifications for zero-shot multi-speaker and multilingual training, resulting in good speaker similarity and speech quality. GenerSpeech (Huang et al. 2022) proposes a multi-level style adapter and a generalizable content adapter to efficiently model style information. MegaTTS (Jiang et al. 2023) employs various techniques (e.g., VQ-GAN, codec-LM) to extract different speech attributes (e.g., content, timbre, prosody, and phase), leading to successful speech disentanglement. Despite the impressive results, they are still limited by the availability of pre-recorded reference speech with a clean background. Moreover, the synthesized speech is often limited by the intrinsic attributes of the reference speech.

Text prompt: In contrast to traditional TTS systems that require users to have acoustic knowledge to understand style elements such as prosody and pitch, the use of text prompts is more user-friendly as text descriptions offer a more intuitive and natural means of expressing speech style. For example, PromptTTS (Guo et al. 2023) utilizes the BERT model as a style encoder and a transformer-based content encoder to extract the corresponding representations from the text prompt to achieve voice control. InstructTTS (Yang et al. 2023) proposes a novel three-stage training procedure to obtain a robust sentence embedding model that can effectively capture semantic information from style prompts and control the speaking style in the generated speech. Sally (Ji et al. 2024) employs an autoregressive codec-LM as a style encoder and a non-autoregressive codec-LM as a decoder to generate acoustic tokens across varying granularities, capitalizing on the robust coding capabilities of language models. Promptspeaker (Zhang et al. 2023) integrates the Glow model to create an invertible mapping between semantic and speaker representations, thereby enabling text-driven TTS. Despite promising results, these works still rely on human effort to provide detailed text descriptions, which may be unavailable at scenarios requiring rapid content creation.

Image prompt: Image prompt-based TTS allows a more comprehensive and expressive synthesis process, as the visual context can provide additional information and nuances that enhance the overall quality and authenticity of the generated speech. For example, VisualTTS (Lu et al. 2022) pioneers the use of silent pre-recorded videos as conditioned inputs to not only generate human-like speech but also achieve precise lip-speech synchronization. Similarly, Imaginary Voice (Lee, Chung et al. 2023) introduces a face-styled diffusion TTS model within a unified framework which designs a speaker feature binding loss to enforce similarity between the generated and real speech segments in the speaker embedding space. To be noted, MMTTS (Guan et al. 2024) proposes an aligned multi-modal prompt encoder that embeds three different modalities into a unified style space. Despite allowing any modality input, it limits speech generation to real faces, overlooking the potential influence of images with diverse styles.

Summary: Considering the aforementioned limitations, in this paper, we aim to generate expressive and high-quality human speech associated with characters from input images of various styles. Henceforth, we introduce a portrait-speech pairing dataset comprising multi-style images. Using this dataset, we develop FaceSpeak, enabling the model to be generalized across various image styles. In particular, our FaceSpeak framework achieves enhanced accuracy and flexibility in controlling visual features for synthesized speech, which is achieved through the deliberate decoupling of identity and emotion information within the visual features of the portrait.

![](images/97f7a83d99c0866b761b65131dd4b34a7c0b467af2c1c6bc4825838cec5c1ebb.jpg)  
Figure 2: Our image generation pipeline of 1) $\mathrm { E M ^ { 2 } }$ TTS-MEAD subset (top): we specify the desired output style and transfer the real human image to images of different styles using PhotoMaker. 2) $\bar { \mathrm { E M } } ^ { 2 }$ TTS-ESD-EmovDB subset (bottom): we use a human expert to label the character factors for chatGPT to create the descriptive text, which is utilized by DALL-E-3 to produce images that are highly aligned with the specified parameters.

# Proposed EM2TTS Dataset

The widely-used TTS datasets, such as LibriTTS (Zen et al. 2019) and VCTK (Yamagishi et al. 2019), predominantly exhibit a single-modal nature, comprising solely audio recordings without corresponding textual or visual labels. Existing multi-modal TTS datasets, including ESD (Zhou et al. 2022), EmovDB2, and Expresso (Nguyen et al. 2023), lack visual labels for character profiles. While IEMOCAP (Busso et al. 2008), MEAD (Wang et al. 2020), CMUMOSEI (Zadeh et al. 2018) and RAVDESS (Livingstone and Russo 2018) datasets provide labels with limited aspects such as emotion and facial expressions, no existing TTS datasets provide face data with diverse image styles.

To address the above limitations, we re-design and annotate an expansive multi-modal TTS dataset, termed $\mathrm { E M ^ { 2 } T T S }$ . It is enriched with diverse text descriptions and a wide range of facial imagery styles. Due to the distinct data characteristics and varying levels of annotation completion, we have delineated the dataset into two distinct emotional subsets. Please see Appendix for more details.

# $\mathbf { E M } ^ { 2 }$ TTS-MEAD

For the MEAD dataset (Wang et al. 2020) with real human face recordings, we initially applied a random selection strategy to extract video frames. Then, following steps are designed to proceed the data: 1) Automatic text generation: we generate corresponding text labels that encompass gender, emotion and its intensity, utilizing the raw data from the MEAD dataset. In particular, our methodology draws from the MMTTS framework (Guan et al. 2024), which correlates emotion intensity levels with specific degree words, as shown in Figure 2. 2) Image style transfer: we leverage the innovative image style conversion model, PhotoMaker (Li et al. 2023b), which excels in producing a variety of portrait styles representing the same individual, guided by character images and textual prompts. For each speaker exhibiting varying emotion intensities, we have created four styles of images (i.e., fantasy art, cinematic, neonpunk, and line art), enriching the visual diversity of the dataset.

Nevertheless, a significant challenge arises in discerning the correlation between a speaker’s appearance and their timbre, particularly with hard samples. These cases involve a mismatch between the physical attributes and the vocal characteristics of the speaker (e.g., a physically strong person’s voice may sound similar to that of a slim woman). Consequently, models trained exclusively on $\mathrm { E M ^ { 2 } T T S ^ { _ { - } } }$ MEAD face difficulties in aligning with intuitive expectations of human perception.

# $\mathbf { E M } ^ { 2 }$ TTS-ESD-EmovDB

The unimodal emotional speech datasets ESD and EmovDB lack accompanying speaker images, thus performing style transfer based on real human face is unfeasible. Therefore, we design the next steps to process them: 1) Manual annotation: we explore a human expert to label age, gender, and characteristics by listening to each speech data; 2) Text expansion: we use the Large Language Model (LLM) model e.g., ChatGPT to expand the label words into texts with varying contents but similar meanings; 3) Text-driven image generation: the enriched texts were then fed into DALL-E-3, a text-to-image model capable of generating a multitude of images in distinct styles. We assigned these generated images to the corresponding speeches, resulting in style-free images that emphasize the character’s emotions, surpassing the limitations of the recorded images.

# Proposed Method

Let us denote $\mathbf { I } _ { i }$ , $\mathbf { x } _ { i }$ and $\mathbf { t } _ { i }$ as the visual image of arbitrary style, the corresponding voice signal, and the co-speech content of character $i$ , respectively. As depicted in Fig. 3, our proposed FaceSpeak algorithm aims to generate a speech waveform $\mathbf { s } _ { i }$ corresponding to the imagined voices of characters given the input text $\mathbf { t } _ { i }$ and images of different styles (either real $\mathbf { I } _ { i } ^ { R }$ or generated $\mathbf { I } _ { i } ^ { G }$ ).

The proposed FaceSpeak consists of two sub-modules: 1) Multi-style image feature decomposition module that receives different styles of portrait images for feature extraction and decouples emotion and speaker information. With this module, we can get the disentangled identity and emotion embeddings extracted from the portrait visual features; 2) Expressive TTS module receives the identity and emotion embeddings as control vectors for generating high-quality speech that matches the portrait images. Detailed descriptions are given below.

# Multi-Style Image Feature Disentanglement

To enhance the coherence of identity and emotion between the input image and the synthesized speech, it is crucial to mitigate the influence of extraneous visual elements (e.g., background, clothing, distracting objects) in the extracted visual features. The FaRL model (Zheng et al. 2022), leveraging the multi-modal pre-trained CLIP model on largescale datasets of face images and correlated text, ensures the extraction of predominantly face-related visual features with robust generalization capabilities. Thus, we apply it on real images and corresponding multi-style images to extract high-dimensional intermediate visual representations:

$$
\mathbf { e } _ { i } = \mathrm { F a R L } ( \mathbf { I _ { i } } )
$$

where $\mathbf { e } _ { i } \ \in \ \mathbf { R } ^ { 5 1 2 }$ contains both the emotion and speaker information of the portrait.

Our subsequent objective is to decouple the emotion and the speaker information within $\mathbf { e } _ { i }$ . Let us define the Identity Adapter Module (IAM) and the Expression Adapter Module (EAM) to learn the mapping from $\mathbf { e } _ { i }$ to the identity embedding $\alpha _ { i }$ and emotion embedding $\beta _ { i }$ , respectively:

$$
\begin{array} { r } { \pmb { \alpha _ { i } } = \mathrm { I A M } ( \pmb { \mathrm { e } } _ { i } ) = \mathrm { F C } \left( \mathrm { G e L U } ( \mathrm { F C } ( \pmb { \mathrm { e } } _ { i } ) ) \right) } \\ { \pmb { \beta _ { i } } = \mathrm { E A M } ( \pmb { \mathrm { e } } _ { i } ) = \mathrm { F C } \left( \mathrm { G e L U } ( \mathrm { F C } ( \pmb { \mathrm { e } } _ { i } ) ) \right) } \end{array}
$$

Subsequently, we used the emotion classification model following $\beta _ { i }$ to bias the $\beta _ { i }$ features toward emotion characteristics. Furthermore, we incorporated the emotion classification model after introducing Gradient Reverse Layer (GRL) following $\alpha _ { i }$ , which aims to minimize the sentiment information retained in $\alpha _ { i }$ . We utilize the Cross-Entropy loss to constrain the two emotion classification models, formulated as follows:

$$
\begin{array} { r l } & { \mathcal { L } _ { e m o } = \mathrm { C r o s s E n t r o p y } ( \mathrm { C L S } ( \beta _ { i } ) , \mathrm { L } _ { e } ) } \\ & { \quad \mathcal { L } _ { g r l } = \mathrm { C r o s s E n t r o p y } ( \mathrm { G R L } ( \mathrm { C L S } ( \alpha _ { i } ) ) , \mathrm { L } _ { e } ) } \end{array}
$$

where CLS represents the classification layer, and $\mathrm { L } _ { e }$ denotes the emotion categorization label of the input image $\mathbf { I _ { i } }$ . GRL inverts the sign of the incoming gradient during the back-propagation phase. By this strategic reversal, IAM learns to remove or minimize features that are correlated with emotion, emphasizing the identity aspects of the input.

To enhance the decoupling of identity embedding $\alpha _ { i }$ and emotion embedding $\beta _ { i }$ , we propose a feature decoupling approach that leverages Mutual information (MI) minimization. This strategy effectively reduces the correlation between identity and emotion representations, enabling a more robust and accurate analysis of each aspect.

Mutual information based decoupling: MI is a fundamental concept in information theory that quantifies the statistical dependence between two random variables $V _ { 1 }$ and $V _ { 2 }$ , calculated as:

$$
I ( V _ { 1 } ; V _ { 2 } ) = \sum _ { v _ { 1 } \in V _ { 1 } } \sum _ { v _ { 2 } \in V _ { 2 } } p ( v _ { 1 } , v _ { 2 } ) \log \left( { \frac { p ( v _ { 1 } , v _ { 2 } ) } { p ( v _ { 1 } ) p ( v _ { 2 } ) } } \right)
$$

where $p ( v _ { 1 } , v _ { 2 } )$ is the joint probability distribution between $v _ { 1 }$ and $v _ { 2 }$ , while $p ( v _ { 1 } )$ and $p ( v _ { 2 } )$ are their marginals. However, it is still a challenge to obtain a differentiable and scalable MI estimation. In this work, we use vCLUB (Cheng et al. 2020), an extension of CLUB, to estimate an upper bound on MI, which allows efficient estimation and optimization of MI when only sample data are available and a probability distribution is not directly available. Given the sample pairs of identity embedding and emotion embedding $\{ ( \boldsymbol { \alpha _ { i } } , \boldsymbol { \beta _ { i } } ) \} _ { i = 1 } ^ { N }$ , where $N$ denotes the number of samples, the MI can be computed as:

$$
\mathcal { L } _ { m i } = \frac { 1 } { N ^ { 2 } } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \left[ \log q _ { \theta } ( \beta _ { i } | \alpha _ { i } ) - \log q _ { \theta } ( \beta _ { j } | \alpha _ { i } ) \right]
$$

where $q _ { \theta } ( \beta _ { i } | \alpha _ { i } )$ is a variational approximation which can make vCLUB holds a MI upper bound or become a reliable MI estimator. At each iteration during the training stage, we first obtain a batch of samples $\{ ( \bar { \alpha } _ { i } , \beta _ { i } ) \}$ from IAM and EAM, then update the variational approximation $q _ { \theta } ( \beta _ { i } | \alpha _ { i } )$ by maximizing the log-likelihood ${ \mathcal { L } } _ { \theta } ~ =$ $\begin{array} { r } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log q _ { \theta } ( \beta _ { i } | \alpha _ { i } ) } \end{array}$ . The updated $q _ { \theta } ( \beta _ { i } | \alpha _ { i } )$ can be used to calculate the vCLUB estimator. Finally, we sum the decoupled identity embedding and the emotion embedding obtained as the ultimate control embedding $p _ { i }$ for speech synthesis.

![](images/9471cce122af59b392c69d561c6e60714c2554aaf848fef51f6bde1c2f26813d.jpg)  
Figure 3: Block diagram of our proposed FaceSpeak which generates speech given the input text $\mathbf { t } _ { i }$ and images of different styles (either real $\mathbf { I } _ { i } ^ { \breve { R } }$ or generated $\bar { \mathbf { I } } _ { i } ^ { G }$ ). It consists of two sub-modules: multi-style image feature disentanglement (yellow region) and expressive TTS (gray region).

# Expressive TTS

We use VITS2 (Kong et al. 2023), one of the SOTA TTS models, as our speech synthesis backbone. As shown in Figure 3, The VITS2 model consists of a Posterior Encoder and a Text Encoder that generate posterior distribution and prior distribution based on the input speech and text, respectively; a transformer-based Flow module to refine the latent representation produced by the posterior encoder; a Monotonic Alignment Search (MAS) module to estimate an alignment between input text and target speech; a Duration Predictor module to predict the duration of each phoneme; a Decoder to reconstructs the speech from the latent representation generated by the posterior encoder. In particular, we injected the control embedding from the portrait images into the Posterior Encoder, Decoder, Flow module, and Duration Predictor to generate the speech corresponding to the portrait images, which are highlighted in green in Figure 3. In the training stage, identity embedding $\alpha _ { i }$ and emotion embedding $\beta _ { i }$ are decoupled from the same image and the final loss can be expressed as:

$$
\mathcal { L } = \mathcal { L } _ { v i t s } + \lambda _ { 1 } \mathcal { L } _ { m i } + \lambda _ { 2 } \mathcal { L } _ { e m o } + \lambda _ { 3 } \mathcal { L } _ { g r l }
$$

where $\lambda _ { 1 } , \lambda _ { 2 }$ and $\lambda _ { 3 }$ are the hyper-parameters to balance the individual losses. During inference, $\alpha _ { i }$ and $\beta _ { i }$ can come

from the same image or be provided by different images separately, and the inference process of Expressive TTS is consistent with work (Kong et al. 2023).

# Experiments Dataset and Experimental Setup

In the evaluation of our method, we conduct separate experiments on intra-domain and out-of-domain data, respectively. We use $\mathrm { E M ^ { 2 } }$ TTS-MEAD as the intra-domain experimental data, while for the out-of-domain evaluation, we perform different settings for real portrait scenes and multi-style virtual portrait scenes, respectively. For the real portrait scenes, we follow the MMTTS’s setup which uses the face images in Oulu-CASIA dataset and transcriptions in LibriTTS, while for the multi-style virtual portrait scenario, we use a different image generation API from the training set to generate new test data based on $\mathrm { E M ^ { 2 } }$ TTS-ESD. Specifically, we compare our method with the following system:1) GT: Ground Truth. 2) VITS2: A multispeaker TTS baseline. 3) MMTTS: A style transfer TTS system using prompt including image. 4) MM-StyleSpeech: Same as MMTTS using StyleSpeech as backbone. The proposed FaceSpeak is trained for 150K iterations using the Adam optimizer on NVIDIA GeForce RTX 3090 GPUs. Detailed training parameters and network configuration can be found in the Appendix.

# Synthetic Quality on Real Portraits

We first evaluate the quality of the speech synthesized by FaceSpeak given a real portrait as the prompt. We conduct a Mean Opinion Score (MOS) (Sisman and Yamagishi 2021)

Table 1: MOS results with $9 5 \%$ confidence interval (N-: naturalness; IS-: identity similarity; ES-: emotion similarity; -: info mation not applicable.)   

<html><body><table><tr><td>Method</td><td>NMOS ↑</td><td>Intra-domain ISMOS ↑</td><td>ESMOS ↑</td><td>NMOS ↑</td><td>Out-of-domain ISMOS ↑</td><td>ESMOS ↑</td></tr><tr><td>GT</td><td>4.42 ± 0.02</td><td></td><td>4.52 ± 0.03</td><td></td><td></td><td></td></tr><tr><td>VITS2</td><td>3.55 ± 0.06</td><td>3.68 ± 0.07</td><td>3.38 ± 0.13</td><td>3.42 ± 0.05</td><td>3.56 ± 0.10</td><td>3.31 ± 0.09</td></tr><tr><td>MM-StyleSpeech</td><td>3.58 ± 0.08</td><td>3.64 ± 0.04</td><td>3.89 ± 0.11</td><td>3.23 ± 0.08</td><td>3.61 ± 0.07</td><td>3.78 ±0.08</td></tr><tr><td>MM-TTS</td><td>3.94 ± 0.05</td><td>3.82 ±0.08</td><td>4.08 ± 0.08</td><td>3.41 ± 0.06</td><td>3.68 ± 0.04</td><td>3.91 ± 0.05</td></tr><tr><td>FaceSpeak</td><td>4.13 ± 0.04</td><td>3.97 ± 0.07</td><td>4.36 ± 0.05</td><td>4.28 ± 0.05</td><td>3.77 ± 0.09</td><td>3.98 ± 0.07</td></tr></table></body></html>

<html><body><table><tr><td rowspan="3">Method</td><td colspan="4">Intra-domain</td><td colspan="4">Out-of-domain</td></tr><tr><td rowspan="2">7-point score ↑</td><td colspan="3">B Preference %) </td><td rowspan="2">7-point score ↑</td><td colspan="3">B Preference(%) </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MM-StyleSpeech</td><td>1.25 ± 0.08</td><td>28</td><td>22</td><td>50</td><td>1.97 ± 0.12</td><td>10</td><td>15</td><td>75</td></tr><tr><td>MM-TTS</td><td>1.03 ± 0.06</td><td>33</td><td>14</td><td>53</td><td>1.42 ± 0.09</td><td>16</td><td>24</td><td>60</td></tr></table></body></html>

Table 2: AXY preference test results. B, E and O respect the preference rate for baseline model, equivalent and our model, respectively.

with $9 5 \ \%$ confidence intervals to assess speech quality. Naturalness-MOS (NMOS) and Similarity-MOS (SMOS) evaluate the speech naturalness and image-speech similarity. Specifically, we generate 50 speech samples for each model, which are rated by 20 volunteers on a scale of 1 to 5, with higher scores indicating better results ( ). We also perform an AXY preference test (Skerry-Ryan and Battenberg 2018) to verify the style transfer effect based on image prompt. In the test, “A” is the reference speech that is stylistically consistent with the image prompt, “X” and “Y” are the speech generated by the compared model or our proposed FaceSpeak. The participants decide whether the speech style of “X” or “Y” is closer to that of image prompt where the scale range of [-3,3] indicating “X” is closer to “Y” is closer. Table 1 displays the subjective results including NMOS, ISMOS and ESMOS. Our FaceSpeak achieves better results on both speech naturalness and style similarity, showing the effectiveness of DSP module on speaker representation extraction by using IAM and EAM. As shown in Table 2, the results of AXY test indicate that listeners prefer FaceSpeak synthesis against the compared models. The generated data and the method significantly improves the style extraction ability, allowing an arbitrary reference sample to guide the stylistic synthesis of arbitrary content text.

As shown in the Table 3, we further objectively measure the quality of speech through MCD (Kubichek 1993), emotion and gender classification accuracy, and speaker similarity. MCD measures the spectral distance between the reference and synthesized speech and FaceSpeak achieved a MCD result of 3.32. We measure speaker similarity (SS) between two speech samples in Resemblyzer 3, our result is 0.95. A hubert-based pre-trained model 4 is used for gender classification $( \operatorname { A c c } _ { g e n } )$ and emotion2vec is used to predict the emotion category $( \operatorname { A c c } _ { e m o } )$ of speech. On the intradomain data, we obtained $\operatorname { A c c } _ { g e n }$ for $9 9 . 4 0 \%$ and $\operatorname { A c c } _ { e m o }$ for $6 0 . 9 2 \%$ . For out-of-domain data, the results for $\operatorname { A c c } _ { g e n }$ and $\operatorname { \bf A c c } _ { e m o }$ are $9 2 . 4 2 \%$ and $3 1 . 3 2 \%$ , respectively.

![](images/16030ae6e31accabc2a1ddc0a6a7e3134dce05ee64eca8e4e74e13d25e5dd412.jpg)  
Figure 4: Visualization of emotion embeddings (colors index emotions).

# Synthetic Quality on Multi-Style Virtual Portraits

In evaluating the quality of FaceSpeak’s speech synthesis based on multi-style virtual portraits, the models we compare are whether or not trained with the virtual portrait images of our proposed multi-style dataset $\mathrm { E M ^ { 2 } T \bar { T } S }$ , respectively. Since models trained with $\mathrm { E M ^ { 2 } T T S }$ will have “seen” multi-style portraits in the domain, our evaluation will only be performed on out-of-domain multi-style portraits. As illustrated in Table 4, the model trained with the virtual portrait images performs better in all evaluation metrics, confirming the effectiveness of our methods.

# Results of Decoupled Identity and Emotion Information

TSNE results of decoupled features: Fig. 4 (a) visualizes the embeddings extracted from FaRL in emotion, which are randomly distributed. By applying the EAM module, as shown in Fig. 4 (b), the learned embeddings with the same

Table 3: Subjective results on real portraits controlled speech synthesis.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="4">Intro-domain</td><td colspan="2">Out-of-domain</td></tr><tr><td>MCD↓</td><td>ACCemo ↑</td><td>ACCgen ↑</td><td>SS↑</td><td>ACCemo ↑</td><td>ACCgen ↑</td></tr><tr><td>GT</td><td>1</td><td>84.54</td><td>100.00</td><td>1</td><td>-</td><td>=</td></tr><tr><td>FaceSpeak</td><td>3.32</td><td>60.92</td><td>99.40</td><td>0.95</td><td>31.32</td><td>92.42</td></tr></table></body></html>

<html><body><table><tr><td>Method</td><td>NMOS ↑</td><td>ISMOS ↑</td><td>ESMOS ↑</td><td>ACCemo ↑</td><td>ACCgen↑</td></tr><tr><td>w/o EM²TTS</td><td>4.31 ± 0.05</td><td>3.88 ±0.09</td><td>4.02 ± 0.06</td><td>18.34</td><td>84.24</td></tr><tr><td>w/EM²TTS</td><td>4.38 ± 0.06</td><td>4.06 ± 0.08</td><td>4.47 ± 0.04</td><td>26.56</td><td>92.22</td></tr></table></body></html>

Table 4: Objective and Subjective results on out-of-domain multi-style portraits controlled.

![](images/4efaa01eb7f819a793fae9ab9aae34c97eecd74192cdbe43878a4b5648b37a9b.jpg)  
Figure 5: Visualization of identity embeddings (colors index identities; M: male; F: female).   
Figure 6: The accuracy in matching the synthesized speech to the emotion and identity styles when they are controlled separately by different images.

emotion are more clustered, while the others are more discriminative. Fig. 5 (a) shows the embeddings extracted from FaRL in identity, which are also randomly distributed. By applying the IAM module, from Fig. 5 (b) to (d), it is observed that embeddings from different speakers are more distinguished when using both the GRL and vCLUB strategies. These intuitively verify the effectiveness of our decoupled identity and emotion characteristics.

Speech synthesis controlled by combined portraits: By decoupling identity and emotion information in an image, we can use different image combinations to control the synthesized speech. As shown in Figure 6, we define $\mathbf { X }$ as the image that provides identity embedding and $\mathbf { Y }$ as the image that provides emotion embedding, ensuring that $\mathbf { X }$ and $\mathbf { Y }$ have different genders and emotions. We let listeners discriminate the synthesized speech by deciding whether it is matched to the $\mathbf { X }$ -image or $\mathbf { Y }$ -image in terms of identity and emotion, respectively. As a result, $9 8 . 6 \%$ of the speech are determined to correctly match the X image on identity, while $9 2 . 1 \%$ of the speech are determined to correctly match the $\mathbf { Y }$ image on emotion, which proves that our proposed FaceSpeak speech synthesis system can reliably control the synthesis by combining different images, greatly improving the diversity and flexibility.

X Identity 98.6% Y Identity   
X Emotion 92.1% Y Emotion X: Images provided as Identity Prompt Y: Images provided as Emotion Prompt

duced an innovative $\mathrm { E M ^ { 2 } T T S }$ dataset, which is meticulously curated and annotated to support and advance research in this emerging field. Additionally, novely methods are proposed for decoupling emotional and speaker-specific features, enhancing both the adaptability and fidelity of our system. Experimental results confirm that FaceSpeak can generate high-quality, natural-sounding speech that authentically align with the visual attributes of the character.

Looking ahead, we aspire to broaden the diversity of speaker categories within the Facespeak system, integrating a wider array of emotions, roles, and other dynamic attributes. By leveraging larger, more comprehensive datasets, we aim to advance the system’s development, enhancing its adaptability and versatility.

# Conclusion

In this paper, we introduce FaceSpeak, a pioneering approach for multi-modal speech synthesis which extracts key identity and emotional cues from diverse character images to drive the TTS module to synthesize the corresponding speech. To tackle the problem of data scarcity, we intro