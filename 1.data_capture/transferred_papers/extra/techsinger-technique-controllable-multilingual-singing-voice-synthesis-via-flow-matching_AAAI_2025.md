# TechSinger: Technique Controllable Multilingual Singing Voice Synthesis via Flow Matching

Wenxiang Guo, Yu Zhang, Changhao Pan, Rongjie Huang, Li Tang, Ruiqi Li, Zhiqing Hong, Yongqi Wang, Zhou Zhao\*

Zhejiang University guowx314,yuzhang34,panch,zhaozhou @zju.edu.cn

# Abstract

Singing voice synthesis has made remarkable progress in generating natural and high-quality voices. However, existing methods rarely provide precise control over vocal techniques such as intensity, mixed voice, falsetto, bubble, and breathy tones, thus limiting the expressive potential of synthetic voices. We introduce TechSinger, an advanced system for controllable singing voice synthesis that supports five languages and seven vocal techniques. TechSinger leverages a flow-matching-based generative model to produce singing voices with enhanced expressive control over various techniques. To enhance the diversity of training data, we develop a technique detection model that automatically annotates datasets with phoneme-level technique labels. Additionally, our prompt-based technique prediction model enables users to specify desired vocal attributes through natural language, offering fine-grained control over the synthesized singing. Experimental results demonstrate that TechSinger significantly enhances the expressiveness and realism of synthetic singing voices, outperforming existing methods in terms of audio quality and technique-specific control.

# Code — https://github.com/gwx314/TechSinger Demo — https://tech-singer.github.io

# Introduction

Singing voice synthesis (SVS) aims to produce high-fidelity vocal performances that capture the nuances of human singing, including pitch, pronunciation, emotional expression, and vocal techniques. This field has attracted considerable attention due to its potential to revolutionize music creation and expand the boundaries of artistic expression. In recent years, rapid advancements in deep learning and generative models have driven substantial progress in singing voice synthesis (Resna and Rajan 2023; Liu et al. 2022; Huang et al. 2022; Kim et al. 2023; Hong et al. 2023).

As singing voice synthesis technology advances, realworld applications, such as personalized virtual singers, content creation for multimedia platforms, and music production tools, highlight the growing need for controllable singing synthesis systems. However, challenges remain in achieving fine-grained control over specific vocal techniques during synthesis. Techniques like vibrato, breathy, and other stylistic nuances require precise manipulation to elevate the artistic expressiveness of synthesized singing voices. While recent algorithms have enabled accurate reproduction of acoustic features like pitch and timbre (Kumar et al. 2021), further advancements are needed to integrate detailed control over vocal techniques. This capability is essential for meeting the personalized and creative demands of modern music production, offering artists and creators more expressive and versatile tools for their work.

Although the task of technique-controllable singing voice synthesis holds great promise to revolutionize how we create and interact with vocal performances, it faces several significant challenges: 1) Most existing SVS datasets, like M4Singer (Zhang et al. 2022a) and OpenCPOP (Wang et al. 2022), focus on basic features such as pitch and emotion but lack detailed annotations for singing techniques. Although Gtsinger (Zhang et al. 2024c) provides a dataset with several technique annotations, such datasets are still relatively rare. The absence of annotations for techniques limits models’ ability to perform singing techniques. 2) Achieving finegrained control over various singing techniques remains a core challenge. While many studies have advanced expressive singing voice synthesis by controlling features like intensity, vibrato, and breathy, they still face limitations in finely controlling multiple complex vocal techniques. Precisely modeling and reproducing various techniques while maintaining natural pitch and timbre variation is a current research focus. 3) Utilizing the prompts for more convenient and intuitive control of singing voice synthesis based on fine-grained phoneme-level annotations is an innovative research direction (Wang et al. 2024). The prompt mechanism allows users to instruct the model on the desired singing style and techniques using natural language, lowering the technical barrier and enhancing user experience. However, designing effective prompt representations, training models to understand and respond to these prompts, and achieving flexible technique control while ensuring high-quality generated singing voices require further research and practice.

To address these challenges, we employ various strategies. Firstly, we tackle the scarcity of technique-annotated datasets by training a technique detector to automatically annotate technique information in open-source singing voice data. Secondly, we introduce the first flow-matchingbased singing voice synthesis, enabling fine-grained control of multiple singing techniques and enhancing generated singing voices’ realism and artistic expressiveness. To accurately model the complex relationship between pitch variations and technique expressions, we also use a flowmatching strategy to predict pitch. Lastly, we leverage pretrained language models GPT-4o to construct comprehensible prompts and train a technique predictor, allowing users to easily specify desired singing styles and techniques through natural language input, thereby simplifying the operational process, enhancing user experience, and further promoting the development of personalized and customized music creation. TechSinger achieves the best results, with subjective MOS $3 . 8 9 \ / \ 4 . 1 0$ in terms of the quality and technique-expressive of the singing voice generation.

In summary, this paper makes the following significant contributions to the field of singing voice synthesis:

• We introduce TechSinger, the first multi-lingual singing voice synthesis model via flow matching that achieves fine-grained control over multiple techniques. • To tackle the challenge of limited technique-annotated datasets, we develop an automatic technique detector for annotating singing techniques in open-source data. • We unveil the Flow Matching Pitch Predictor (FMPP) and the Classifier-Free Guidance Flow Matching MelSpectrogram Postnet (CFGFMP) to improve quality. • We leverage GPT-4o to create a prompt-based singing dataset and, based on this dataset, propose a technique predictor that allows for controlling singing techniques through natural language prompts. • Experiments show that our model excels in generating high-quality, technique-controlled singing voices.

# Related Works

# Singing Voice Synthesis

Singing Voice Synthesis (SVS) has advanced significantly with deep learning, aiming to generate high-quality singing from musical scores and lyrics. Early models like XiaoiceSing (Lu et al. 2020) and DeepSinger (Ren et al. 2020b) utilize non-autoregressive and feed-forward transformers to synthesize singing voice. VISinger (Zhang et al. 2022b) employs the VITS (Kim, Kong, and Son 2021) architecture for end-to-end SVS. GANs have also been used for high-fidelity voice synthesis (Wu and Luan 2020; Huang et al. 2022), and DiffSinger (Liu et al. 2022) introduces diffusion for improved mel-spectrogram generation. Despite these advancements, precise control over singing techniques remains a challenge, which is essential for enhancing artistic expressiveness. Controllable SVS focuses on managing aspects like timbre, emotion, style, and techniques. Existing works often target specific controls, such as Muse-SVS (Kim et al. 2023) for pitch and emotion, StyleSinger (Zhang et al. 2024a) and TCSinger (Zhang et al. 2024b) for style transfer, and models for vibrato control (Liu et al. 2021; Song et al. 2022; Ikemiya, Itoyama, and Okuno 2014). However, we advance technique controllable SVS by enabling control over seven techniques across five languages.

# Prompt-guided Voice Generation

In terms of voice generation, previous controls rely on texts, scores, and feature labels. Prompt-based control is emerging as a simpler, more intuitive alternative and has achieved great success in text, image, and audio generation tasks (Brown et al. 2020; Ramesh et al. 2021; Kreuk et al. 2022) In speech generation, PromptTTS (Guo et al. 2023) and InstructTTS (Yang et al. 2023) use text descriptions to guide synthesis, offering precise control over style and content. In singing voice generation, Prompt-Singer (Wang et al. 2024) uses natural language prompts to control attributes like the singer’s gender and volume but lacks advanced technique control. This paper addresses this gap by integrating multiple techniques into prompt-based control, allowing for more sophisticated and expressive singing voice generation.

# Flow Matching Generative Models

Flow matching (Lipman et al. 2022) is an advanced generative modeling technique that optimizes the mapping between noise distributions and data samples by ensuring a smooth transport path, reducing sampling complexity. It has significantly improved audio generation tasks. Voicebox (Le et al. 2024) uses flow matching for high-quality text-tospeech synthesis, noise removal, and content editing. Audiobox (Vyas et al. 2023) leverages flow matching to enhance multi-modal audio generation with better controllability and efficiency. Matcha-TTS (Mehta et al. 2024) applies optimal-transport conditional flow matching for highquality, fast, and memory-efficient text-to-speech synthesis. VoiceFlow (Guo et al. 2024) utilizes rectified flow matching to generate superior mel-spectrograms with fewer steps. Inspired by these successes, we use flow matching for controllable singing voice synthesis to boost quality and efficiency.

# Preliminary: Rectified Flow Matching

Firstly, we introduce the preliminaries of the flow matching generative model (Liu, Gong et al. 2022). When constructing a generative model, the true data distribution is $q ( x _ { 1 } )$ which we can sample, but whose density function is inaccessible. Suppose there is a probability path $p _ { t } ( x _ { t } )$ , where $x _ { 0 } \sim p _ { 0 } ( x )$ is a known simple distribution (such as a standard Gaussian distribution), and $x _ { 1 } \sim p _ { 1 } ( x )$ approximates the realistic data distribution. The goal of flow matching is to directly model this probability path, which can be expressed in the form of an ordinary differential equation (ODE):

$$
\mathrm { d } x = u ( x , t ) \mathrm { d } t , t \in [ 0 , 1 ] ,
$$

where $u$ represents the target vector field, and $t$ represents the time position. If the vector field $u$ is known, we can obtain the realistic data through reverse steps. We can regress the vector field $u$ using a vector field estimator $v ( \cdot )$ with the flow matching objective:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F M } } ( \theta ) = \mathbb { E } _ { t , p _ { t } ( x ) } \left\| v ( x , t ; \theta ) - u ( x , t ) \right\| ^ { 2 } , } \end{array}
$$

where $p _ { t } ( x )$ is the distribution of $x$ at timestep $t$ . To guide the regression by incorporating a condition $c$ , we can use the conditional flow matching objective (Lipman et al. 2022):

$$
\mathcal { L } _ { \mathrm { C F M } } ( \theta ) = \mathbb { E } _ { t , p _ { 1 } ( x _ { 1 } ) , p _ { t } ( x | x _ { 1 } ) } \left\| v ( x , t | c ; \theta ) - u ( x , t | x _ { 1 } , c ) \right\| _ { - } ^ { 2 } ,
$$

Flow matching proposes using a straight path to transform from noise to data. We adopt the linear interpolation schedule between the data $x _ { 1 }$ and a Gaussian noise sample $x _ { 0 }$ to get the sample $x _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 }$ . Therefore, the conditional vector field is $u ( x , t \vert x _ { 1 } , c ) = x _ { 1 } - x _ { 0 }$ , and the rectified flow matching (RFM) loss used in gradient descent is:

$$
\left\| v ( x , t | c ; \theta ) - ( x _ { 1 } - x _ { 0 } ) \right\| ^ { 2 } ,
$$

If the vector field $u$ can be obtained, we can generate realistic data by propagating sampled Gaussian noise through various ODE solvers at discrete time steps. A common approach for the reverse flow is the Euler ODE:

$$
x _ { t + \epsilon } = x + \epsilon v ( x , t | c ; \theta ) .
$$

where $\epsilon$ is the step size. In this work, we use the notes, lyrics, and technique as condition $c$ , while the data $x _ { 1 }$ is fundamental frequencies (F0) or mel-spectrograms.

# TechSinger

In this section, we outline the overall framework of TechSinger, followed by detailed descriptions of its key components, including the flow matching pitch predictor, classifier-free flow matching postnet, technique detector, and technique predictor. We conclude with an explanation of TechSinger’s two-stage training and inference process.

# Overview

The architecture of TechSinger is illustrated in Figure 1. Initially, the phoneme encoder processes the lyrics while the note encoder captures the musical rhythm by encoding note pitches, note durations, and note types. Technique information is provided by encoding a sequence of techniques, and for more precise control over the singing style, a technique predictor is utilized, which generates corresponding technique sequences from the natural language prompt. The technique embeddings, along with the musical information, are then used to predict durations and extend to produce frame-level intermediate features $E _ { p }$ . The flow matchingbased model employs $E _ { p }$ as the condition to generate fundamental frequencies (F0). Subsequently, the coarse mel decoder predicts coarse mel-spectrograms. Finally, the flow matching-based postnet refines these predictions to generate high-quality mel-spectrograms. The process concludes with the use of HiFi-GAN vocoder (Kong, Kim, and Bae 2020), which converts the mel-spectrograms into audio signals.

# Flow Matching Pitch Predictor

Reconstructing fundamental frequencies (F0) using only L1 loss makes it difficult to model the complex mapping between different techniques and F0. To precisely model the pitch contour variations across different techniques, we introduce the Flow Matching Pitch Predictor (FMPP). The fundamental frequency (F0) can be regarded as onedimensional continuous data. The corresponding condition $c$ is the combination features $E _ { p }$ of the music score and technique sequence, and the sampled $x _ { 1 }$ is the F0 extracted by open-source tool RMVPE (Wei et al. 2023) as the target $f 0 _ { g }$ .

Inspired by Lipman et al. (2022), we perform linear interpolation between a F0 sample $x _ { 1 } = f 0 _ { g }$ and Gaussian noise $x _ { 0 }$ to create a conditional probability path $x _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 }$ . We then use the vector field estimator $v _ { p }$ to predict the vector field and train it using the $L _ { p f l o w }$ loss:

$$
\operatorname* { m i n } _ { \theta } \mathbb { E } _ { t , p _ { 1 } ( x _ { 1 } | c ) , p _ { 0 } ( x _ { 0 } ) } \left\| v _ { p } ( x , t | c ; \theta ) - ( x _ { 1 } - x _ { 0 } ) \right\| ^ { 2 }
$$

# CFG Flow Matching Postnet

During the first stage, the mel-spectrogram decoder primarily leverages simple losses (e.g., L1 or L2) to reconstruct the generated mel-spectrograms. Following FastSpeech2 (Ren et al. 2020a), we combine pitch and technique features as inputs and employ stacked FFT (Feed Forward Transformer) blocks with L2 loss for generation training:

$$
L _ { m e l } = \left\| m e l _ { p } - m e l _ { g } \right\| ^ { 2 } ,
$$

However, the generator optimized under the assumption of an unimodal distribution yields mel-spectrograms that lack naturalness and diversity. To further enhance the quality and expressiveness of the mel-spectrograms, we adopt the CFG flow matching mel postnet (CFGFMP). In this work, we utilize the coarsely generated mel-spectrograms $m e l _ { p }$ and the combined pitch and technique features $E _ { m }$ as conditioning information $c$ to guide the training and generation of optimized mel-spectrograms $m e l _ { g }$ . The $L _ { m f l o w }$ loss is analogous to the $L _ { p f l o w }$ loss, as shown in equation 6.

For the reverse process, we randomly sample noise and use the Euler solver to generate samples. To further control the quality of the generated singing voice and its alignment with the intended technique, we implement the classifierfree guidance (CFG) strategy. Specifically, we introduce an unconditional label 2 alongside the conditional labels $\{ 0 , 1 \}$ . During the first two stages, we randomly drop the technique labels for entire phrases or partial phonemes at a rate of 0.1. During sampling, we modify the vector field as follows:

$$
v _ { \mathrm { C F G } } ( x , t | c ; \theta ) = \gamma v _ { m } ( x , t | c ; \theta ) + ( 1 - \gamma ) v _ { m } ( x , t | \mathcal { D } ; \theta ) ,
$$

where $\gamma$ is the classifier free guidance scale. Additionally, since the technique detector output contains errors, this random drop approach ensures the generative model doesn’t blindly trust the labels, to enhance the robustness of the model. For the pseudo-code of the algorithm, please refer to Algorithm 1 and Algorithm 2 provided in Appendix B.1.

# Technique Predictor

For controllable singing synthesis, such as timbre and emotion, many approaches use deterministic labels or corresponding audio to control the generation (Liu et al. 2022; Zhang et al. 2024a). We use natural language as a more intuitive and convenient means to control singing techniques.

However, open-source datasets don’t provide corresponding prompts for each sample. Therefore, we devise a method to generate descriptions. Unlike Prompt-Singer (Wang et al. 2024), which focuses on simple controls like gender, vocal range, and volume, we need to control the singing techniques. We incorporate the singer’s identity (e.g., Alto,

![](images/4e8e5427f8b4daac256f32ec391df4f840ec06bccbe351612e52be2eaa1f192a.jpg)  
Figure 1: The overall architecture of TechSinger. In Figure (a), the technique predictor can predict technique sequences with natural language prompts. The flow matching pitch predictor (FMPP) conditions on the expanded input encoding $E _ { p }$ to generate the F0 sequences. The mel decoder generates the coarse mel-spectrogram. The vector field estimator infers the vector field $v _ { m }$ . In Figure (b), $v _ { m }$ is used to flow the standard Gaussian noise into a fine mel-spectrogram via an ODE solver. In Figure (c), the input of the technique predictor is prompt, note, and lyrics. The text encoder is a pre-trained language model.

Tenor), singing techniques, and language into prompt statements to annotate each sample. First, we collect the singer identity information and the global technique labels from the dataset. Then, we use GPT-4o to generate synonyms for each singer’s identity and singing technique. We create over 60 prompt templates, each containing placeholders for the song’s global technique label, language, and identity. We randomly select these templates and fill in the corresponding synonyms of techniques, identities, and languages to form prompt descriptions for each item. We provide the prompt templates and keywords in the appendix A.1.

As shown in Figure 1(c), our technique predictor comprises two components: a frozen natural language encoder for extracting semantic features and a technique decoder. For the natural language encoder, we evaluate both BERT (Devlin et al. 2018) and FLAN-T5 (Chung et al. 2022) encoders. For the technique decoder, we inject semantic conditions through cross-attention transformers, allowing the model to integrate linguistic cues more effectively. Finally, several classification heads are added to perform multi-task, multilabel classification for different techniques. Singing techniques are classified into three categories: mixed-falsetto and intensity, and four binary categories: breathy, bubble, vibrato, and pharyngeal. The glissando technique can be identified from the music score by determining if a word corresponds to multiple notes.The $L _ { \mathrm { t e c h } }$ classification loss is:

$$
\begin{array} { r l } & { { \cal L } _ { \mathrm { C E } } ^ { ( i ) } = - \displaystyle \sum _ { k = 1 } ^ { 3 } y _ { k } ^ { ( i ) } \log ( p _ { k } ^ { ( i ) } ) } \\ & { { \cal L } _ { \mathrm { B C E } } ^ { ( j ) } = - \left[ y ^ { ( j ) } \log ( p ^ { ( j ) } ) + ( 1 - y ^ { ( j ) } ) \log ( 1 - p ^ { ( j ) } ) \right] } \end{array}
$$

$$
L _ { \mathrm { t e c h } } = \sum _ { i = 1 } ^ { 2 } L _ { \mathrm { C E } } ^ { ( i ) } + \sum _ { j = 1 } ^ { 4 } L _ { \mathrm { B C E } } ^ { ( j ) }
$$

where L(i) represents the cross-entropy loss for the $i$ -th three-class technique group, and L(BjC)E represents the binary cross entropy loss for the $j$ -th binary technique group.

# Technique Detector

Due to the scarcity of technique-labeled singing voice synthesis datasets and the cost and complexity of annotating, we train a singing technique detector to obtain phone-level technique labels. We can also annotate the glissando technique sequence by the same rule as the technique predictor.

As shown in Figure 2, we start by extracting features from the audio, including the mel-spectrogram, fundamental frequency (F0), and other variances features (e.g., energy, and breathiness). These features are encoded and combined as the input feature. We then pass them through a U-Net architecture to extract frame-level intermediate features. To capture the high-level audio features, we utilize the Squeezeformer (Kim et al. 2022) network, one of the most popular ASR models. Inspired by ROSVOT (Li et al. 2024), rather than just using simple averaging or median operations to obtain phoneme-level audio features, we employ a weight prediction average approach. Suppose the frame-level output features are $\mathbf { \bar { \boldsymbol { E } } } _ { f } \in \mathbb { R } ^ { T \times C }$ , where $T$ is the number of frames and $C$ is the number of channels. We predict weights $W _ { f } = \sigma ( E _ { f } W _ { \mathrm { \scriptscriptstyle A } } )$ using a linear layer and the sigmoid operation, where $W _ { \mathrm { A } } \in \mathbb { R } ^ { C \times N }$ , $N$ is the number of heads, and $W _ { f } \in \mathbb { R } ^ { T \times N }$ . We then apply the weights to element-wise multiply $E _ { f }$ to obtain weighted features $E _ { w f } = E _ { f } \odot W _ { f }$ . Assume that phone $i$ corresponds to a sequence starting from frame $j$ with a length of $k$ . we perform a weighted average method across the frame-level embeddings to obtain the final phoneme-level features $E _ { w p }$ :

![](images/3231f03836f7d70eeca6fd61c6b7ed9453900d5b26fdbd4d78118fa1619bcead.jpg)  
Figure 2: The architecture of the technique detector.

$$
E _ { w p } ^ { i } = { \frac { \sum _ { t = 1 } ^ { k } E _ { w f } ^ { i + j + t } } { \sum _ { t = 1 } ^ { k } W _ { f } ^ { i + j + t } } }
$$

where $E _ { w p } \in \mathbb { R } ^ { L \times C \times N }$ , $L$ is the length of phones. Next, we average different heads to get the final phoneme-level features $\mathbf { \Psi } _ { z } ^ { } \in \mathbb { R } ^ { L \times C }$ . Finally, we also use cross-entropy (CE) loss $L _ { p }$ to optimize the multi-task, multi-label technique classification task like the technique predictor.

# Training and Inference Procedures

The training process of TechSinger comprises two stages. During the first stage, we optimize the entire model, excluding the post-processing flow-matching network, and use gradient descent to minimize the $L _ { 1 }$ loss:

$$
L _ { 1 } = L _ { p f l o w } + L _ { m e l } + L _ { d u r }
$$

where $L _ { p f l o w }$ , $L _ { m e l }$ , and $\ b { L _ { d u r } }$ represent the F0 flow matching, mel-spectrogram, and duration losses, respectively. During the second stage, we freeze the components trained in the first phase and optimize the classifier-free flow matching postnet $( L _ { m f l o w } )$ using adding feature $E _ { m }$ of the predicted fundamental frequency, coarse mel-spectrogram, and technique encoding as the condition. During the inference generation process, we can get the technique sequence based on input or prompt statements, which are then combined with lyrics and notes to generate a coarse mel-spectrogram. Subsequently, the flow-matching network refines this coarse mel-spectrogram to produce the final output.

# Experiments Experimental Setup

Dataset and Process Current singing synthesis datasets typically lack the diverse and detailed technique labels necessary for training high-quality models. We use the GTSinger dataset (Zhang et al. 2024c), focusing on its Chinese, English, Spanish, German, and French subsets. Additionally, we collect and annotate a 30-hour Chinese dataset with two singers and four technique annotations (e.g., intensity, mixed-falsetto, breathy, bubble) at the phone and sentence levels. Additionally, to further expand the dataset, we use a trained technique predictor and glissando judgment rule to annotate the M4Singer dataset at the phoneme level, which is used under the CC BY-NC-SA 4.0 license. Finally, we randomly select 804 segments covering different singers and techniques as a test set. The audio used for training has a sample rate of $4 8 ~ \mathrm { k H z }$ , with a window size of 1024, a hop size of 256, and $8 0 \ \mathrm { m e l }$ bins for the extracted mel-spectrograms. Chinese lyrics are phonemicized with pypinyin, English lyrics follow the ARPA standard, while Spanish, German, and French lyrics are phonemicized according to the Montreal Forced Aligner (MFA) standard.

Implementation Details In this experiment, the number of training steps for the F0 and Mel vector field estimator is 100 steps. Their architectures are based on non-causal WaveNet architecture (van den Oord et al. 2016). The number of the technique detector Squeezeformer layers and the technique predictor Transformer layers are both 2. In the first stage, training is performed for $2 0 0 \mathrm { k }$ steps with an NVIDIA 2080 Ti GPU, and in the second stage, for 120k steps. We train the technique detector and predictor for $1 2 0 \mathrm { k }$ and $8 0 \mathrm { k }$ steps. Further details are provided in the appendix B.2.

Evaluation Details For technique-controllable SVS experiments, we use both subjective and objective evaluation metrics. For objective evaluation, we use F0 Frame Error (FFE) to assess the accuracy of F0 prediction and Mean Cepstral Distortion (MCD) to measure the quality of the melspectrograms. For subjective evaluation, we use MOS-Q to assess the quality and naturalness of the audio and MOS-C to evaluate the expressiveness of the technique control. We use objective metrics precision, recall, F1, and accuracy to evaluate the technique predictor and the technique detector. More details are provided in the appendix D.2.

Baseline Models In this section, we compare our approach with state-of-the-art singing voice synthesis models. However, due to the limitations of current datasets, existing singing voice synthesis models are unable to control the techniques of the generation singing audio. Therefore, we augment these baseline systems with a phoneme-level technique embedding layer to enable technique control. The baseline systems we compared are as follows: 1) GT: The ground truth audio sample; 2) GT (vocoder): The original audio is converted to mel-spectrograms and then synthesized back to audio using the HiFi-GAN vocoder; 3) DiffSinger (Liu et al. 2022): A diffusion-based singing voice synthesis model; 4) VISinger2 (Zhang et al. 2022c): An end-to-end high-fidelity singing voice synthesis model; 5) StyleSinger (Zhang et al. 2024a): A style-controllable singing voice synthesis system; 6) TechSinger: The foundational singing voice synthesis system proposed in this paper.

<html><body><table><tr><td>Method</td><td>MOS-Q ↑</td><td>MOS-C ↑</td><td>FFE↓</td><td>MCD↓</td></tr><tr><td>Refernece</td><td>4.54 ± 0.05</td><td></td><td></td><td></td></tr><tr><td>Reference (vocoder)</td><td>4.15 ± 0.06</td><td>4.30 ± 0.09</td><td>0.034</td><td>0.919</td></tr><tr><td>DiffSinger</td><td>3.59 ± 0.07</td><td>3.84 ± 0.08</td><td>0.255</td><td>3.897</td></tr><tr><td>VISinger2</td><td>3.52 ± 0.05</td><td>3.85 ± 0.11</td><td>0.296</td><td>3.944</td></tr><tr><td>StyleSinger</td><td>3.69 ± 0.09</td><td>3.93 ± 0.08</td><td>0.328</td><td>3.981</td></tr><tr><td>TechSinger (ours)</td><td>3.89 ± 0.07</td><td>4.10 ± 0.08</td><td>0.245</td><td>3.823</td></tr></table></body></html>

Table 1: Technique controllable singing voice synthesis performance comparison with different systems. We employ MOS-Q and MOS-C for subjective measurement and use FFE and MCD for objective measurement.

![](images/4e8e18fe33fda863c5de2d70a5fef1ace05c4d7cad5019c0e1bccad35e065631.jpg)  
Figure 3: Visualization of the mel-spectrograms and pitch contour of the ground-truth and results of different SVS systems.

# Main Results

Singing Voice Synthesis As shown in the Table 1, we can draw the following conclusions: (1) In terms of objective metrics, our FFE and MCD values are the lowest, which demonstrates that our TechSinger, through flow matching strategies, can better model pitch and mel-spectrograms under different singing techniques. (2) On the subjective metric MOS-Q, our TechSinger shows higher quality than other baseline models, indicating that our model generates audio with superior quality. Similarly, on the subjective metric MOS-C, our model also outperforms other models, proving that our generation model can faithfully generate corresponding singing voices based on technique conditions. This can be observed from Figure 3, where the F0 generated by our model exhibits more variation and details compared to the relatively flat F0 of other models. Additionally, our melspectrogram is closer to the ground truth mel-spectrograms, showcasing rich details in frequency bins between adjacent harmonics and high-frequency components. The above results demonstrate that our controllable singing voice generation model surpasses other models in terms of both quality and expressiveness in controlling technique generation.

Furthermore, to examine the technique controllability of our model, we present mel-spectrograms and F0 results for the same segments under different technique conditions. As shown in Figure 4, Figure (a) represents the control group without any technique, and Figure (b) displays the result for the bubble, showing more pronounced changes in F0 and mel-spectrograms with a stuttering effect, effectively reflecting the ”cry-like” tone. Figure (c) shows the strong intensity, which appears brighter compared to the control group, enhancing the resonance and intensity of the singing. Figure (d) is the breathy tone result, where harmonics are less distinct and there is more noise, due to the vocal cords not fully closing as air passes through them, causing the breathy sound. From the figures, it is evident that our generated melspectrograms can accurately understand and generate features corresponding to different techniques. More visualization results can be found in the Appendix D.3

Table 2: The quality and relevance to the technique controllablity via different controlling strategies.   

<html><body><table><tr><td>Method</td><td>MOS-Q ↑</td><td>MOS-C↑</td></tr><tr><td>TechSinger(GT)</td><td>3.89±0.07</td><td>4.10±0.08</td></tr><tr><td>TechSinger(Rand)</td><td>3.78 ±0.05</td><td>3.76 ± 0.08</td></tr><tr><td>TechSinger(Prompt)</td><td>3.85 ± 0.05</td><td>4.04 ± 0.07</td></tr></table></body></html>

Table 3: Objective metrics for different text representations, including precision, recall, F1-score, and accuracy.   

<html><body><table><tr><td>Method</td><td>Precision</td><td>Recall</td><td>F1</td><td>Acc</td></tr><tr><td>bert-base-uncased</td><td>0.819</td><td>0.811</td><td>0.807</td><td>0.845</td></tr><tr><td>bert-large-uncased</td><td>0.809</td><td>0.789</td><td>0.786</td><td>0.827</td></tr><tr><td>flan-t5-small</td><td>0.814</td><td>0.808</td><td>0.802</td><td>0.837</td></tr><tr><td>flan-t5-base</td><td>0.828</td><td>0.826</td><td>0.817</td><td>0.851</td></tr><tr><td>flan-t5-large</td><td>0.825</td><td>0.836</td><td>0.818</td><td>0.846</td></tr></table></body></html>

Technique Predictor We employ different text encoders to encode prompts, incorporating their embeddings into the technique sequence prediction through a cross-attention mechanism, with the results shown in Table 3. Overall, the FLAN-T5 model’s performance tends to improve with the increasing size of the encoder. The choice of encoder also has an impact, with FLAN-T5 generally outperforming

![](images/a58604e8c75e1b809b17762073b4fa1440ccaaa3da86be20a1b76dab22c06bfc.jpg)  
Figure 4: Visualization of the mel-spectrogram results generated by TechSinger under different techniques. The red box contains the fundamental pitch, and the yellow box contains the details of harmonics.

Table 4: Ablation experiments for the technique detector.   

<html><body><table><tr><td>Setting</td><td>Precision 个</td><td>Recall 个</td><td>F1↑</td><td>Acc ↑</td></tr><tr><td>whole</td><td>0.815</td><td>0.761</td><td>0.770</td><td>0.833</td></tr><tr><td>ConvUnet</td><td>0.759</td><td>0.726</td><td>0.742</td><td>0.783</td></tr><tr><td>Average</td><td>0.807</td><td>0.756</td><td>0.763</td><td>0.831</td></tr></table></body></html>

BERT. Based on these observations, we select the FLANT5-Large model for the subsequent experiments. More results can be found in the Appendix A.2

To validate the effectiveness of the technique predictor, we compare several different methods of providing techniques for generating results. Among them, TechSinger (GT) represents the results obtained from the annotated technique sequences, TechSinger (Prompt) represents the results predicted by our predictor based on prompts, and TechSinger (Random) represents the results when no techniques are provided and the model generates them automatically. From Table 2, we can see that the mean opinion scores for quality (MOS-Q) and mean opinion scores for controllability (MOS-C) indicate that the ”Prompt” strategy significantly outperforms the ”Random” results and are very close to the ”GT” effect. This demonstrates that our singing voice synthesis model can achieve controllable technique generation through the natural language. Additionally, we can manually adjust the predicted sequences to control the technique used in the generation of singing voices further.

# Ablation Study

Technique Detector As shown in Table 4, we conduct ablation experiments on the methods used in our technique detector to prove their effectiveness. We evaluate the results using objective metrics—precision, recall, F1 score, and accuracy—on six techniques other than glissando, which can be determined by rule-based judgment. By comparing these, we find that the whole technique detector achieves the highest scores across all metrics. Specifically, we replace the Squeezeformer structure with convolution and the multihead weight prediction method with averaging, conducting separate experiments for each. From the table, we can see that the full skill detector outperforms in all metrics, with an F1 score improvement of $0 . 5 \%$ over convolution and

Table 5: Ablation experiments for technique controllable singing voice synthesis with different settings.   

<html><body><table><tr><td>Setting</td><td>CMOSQ ↑</td><td>CMOSC↑</td><td>FFE↓</td></tr><tr><td>TechSinger</td><td>0.00</td><td>0.00</td><td>0.2448</td></tr><tr><td>w/o Pitch</td><td>-0.25</td><td>-0.23</td><td>0.2537</td></tr><tr><td>w/o Postnet</td><td>-0.33</td><td>-0.27</td><td>0.2680</td></tr><tr><td>W/o CFG</td><td>-0.10</td><td>-0.18</td><td>0.2453</td></tr></table></body></html>

$2 . 8 \%$ over averaging, thus validating the effectiveness of the Squeezeformer and the multi-head weight prediction. For more detailed objective metric results of the individual techniques, please refer to Appendix C.

Singing Voice Synthesis As depicted in Table 5, in this experiment, we compare the results using CMOSQ, CMOSC, and FFE. As shown in the first two rows of the table, when we remove the flow-matching pitch predictor, both the F0 prediction accuracy and the quality of the generated audio decline, making it difficult to control the techniques effectively. Comparing the first and third rows, we observe a noticeable decrease in the quality of the synthesized singing when the postnet is omitted. By contrasting the first and fourth rows, we demonstrate that the classifier-free guidance strategy enhances the quality of the generated singing.

# Conclusion

In this paper, we introduce TechSinger, the first multilingual, multi-technique controllable singing synthesis system built upon the flow-matching framework. We train a technique detector to effectively annotate and expand the dataset. To model the fundamental frequencies with high precision, we develop a Flow Matching Pitch Predictor (FMPP), which captures the nuances of diverse vocal techniques. Additionally, we employ Classifier-free Guidance Flow Matching Mel Postnet (CFGFMP) to refine the coarse mel-spectrograms into fine-grained representations, leading to more technique-controllable and expressive singing voice synthesis. Moreover, we train a prompt-based technique predictor to enable more intuitive interaction for controlling the singing techniques during synthesis. Extensive experiments demonstrate that our model can generate high-quality, expressive, and technique-controllable singing voices.

# Ethical Statement

TechSinger’s ability to synthesize singing voices with controllable techniques raises concerns about potential unfair competition and the possible displacement of professional singers in the music industry. Furthermore, its application in the entertainment sector, including short videos and other multimedia content, could lead to copyright issues. To address these concerns, we will implement restrictions on our code and models to prevent unauthorized use, ensuring that TechSinger is deployed ethically and responsibly.