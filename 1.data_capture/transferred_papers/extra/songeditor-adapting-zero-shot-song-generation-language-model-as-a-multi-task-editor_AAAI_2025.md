# SongEditor: Adapting Zero-Shot Song Generation Language Model as a Multi-Task Editor

Chenyu Yang1, Shuai Wang1,3, Hangting Chen\*2 , Jianwei $\mathbf { Y } \mathbf { u } ^ { * 2 }$ , Wei $\mathbf { T a n } ^ { 2 }$ , Rongzhi $\mathbf { G } \mathbf { u } ^ { 2 }$ , Yaoxun $\mathbf { X } \mathbf { u } ^ { 4 }$ , Yizhi Zhou6, Haina $\mathbf { Z } \mathbf { h } \mathbf { u } ^ { 5 }$ , Haizhou $\mathbf { L i } ^ { 1 }$ , 3

1The Chinese University of Hong Kong, Shenzhen (CUHK-Shenzhen), Shenzhen, China 2Tencent AI Lab 3Shenzhen Research Institute of Big Data 4Tsinghua University $^ 5 \mathrm { X }$ -LANCE Lab, Shanghai Jiao Tong University 6National Key Laboratory of Novel Software Technology, Nanjing University chenyuyang2@link.cuhk.edu.cn, erichtchen $@$ tencent.com, tomasyu@foxmail.com

# Abstract

The emergence of novel generative modeling paradigms, particularly audio language models, has significantly advanced the field of song generation. Although state-of-the-art models are capable of synthesizing both vocals and accompaniment tracks up to several minutes long concurrently, research about partial adjustments or editing of existing songs is still underexplored, which allows for more flexible and effective production. In this paper, we present SongEditor, the first song editing paradigm that introduces the editing capabilities into language-modeling song generation approaches, facilitating both segment-wise and track-wise modifications. SongEditor offers the flexibility to adjust lyrics, vocals, and accompaniments, as well as synthesizing songs from scratch. The core components of SongEditor include a music tokenizer, an autoregressive language model, and a diffusion generator, enabling generating an entire section, masked lyrics, or even separated vocals and background music. Extensive experiments demonstrate that the proposed SongEditor achieves exceptional performance in end-to-end song editing, as evidenced by both objective and subjective metrics.

Demo — https://cypress-yang.github.io/SongEditor demo/

# Introduction

In recent years, applications for generating vocal music have achieved remarkable results. Compared with speech or accompanied singing voice, a song typically contains multiple sections interspersed with music-only segments like prelude, interlude, and postlude, which requires strict consistency and contextual coherence. Therefore, an ideal song generation system must be both structure-guided and capable of handling long content.

However, part of the generated song might be unsatisfactory in practice, but regenerating the entire sequence takes much time and computing resources, and potentially leads to further problems. Therefore, song editing is crucial as it allows for the refinement and enhancement of musical compositions. Moreover, there are instances where only a specific track, such as vocal or accompaniment, needs to be changed. How to rewrite the designated part of a song without destroying its contextual consistency and overall musicality, is a significant challenge for AI song composition.

![](images/52bfeb4ba9d58e5e78d3977f2a050f872ecc3efe5f70b653c7305d5256dce431.jpg)  
Figure 1: SongEditor supports various song generation and editing tasks, involving creating complete songs from scratch and Infilling Editing. Accomp-to-song and vocal-tosong mean generating full songs based on partial input conditions such as accompaniments or vocals, respectively.

Currently, there are relatively few studies on end-to-end song generation in the academic field. State-of-the-art song generation platforms like $\mathrm { S u n o ^ { 1 } }$ are generally for commercial use and not open-source. Jukebox (Dhariwal et al. 2020) can synthesize songs in specific styles and genres, but it lacks more refined controls such as structural adjustments or specific instructions. To date, these approaches have primarily focused on pure generation and cannot be directly applied to editing scenarios.

Ding et al. (2024) proposed SongComposer, which supports various tasks like lyric-to-melody and melody-to-lyric, potentially allowing for some degree of song editing. However, SongComposer decomposes the song into lyrics and musical notes (MIDI), retaining no information about vocals such as timbre. Additionally, there is an unavoidable distortion between the original song and the extracted MIDI files during data preprocessing.

In this paper, we propose SongEditor, a general paradigm that enables partial editing in existing language-modelbased song generation approaches. Since few techniques of song generation model are currently available, we first implement SongLM, a fundamental zero-shot song generation framework that can generate song sections up to two minutes long based on structure-guided lyrics and a 10-second excerpt from a reference song as an acoustic prompt. Subsequently, we make modifications in the architecture design and model training process, enabling the system to perform editing based on different types of contexts.

As shown in figure 1, in addition to generating songs from scratch, SongEditor can also regenerate specific periods and separate vocals or accompaniments. Worth noting is that the above two capabilities can be performed at the same time, allowing for the editing of even a short segment of vocals or accompaniment. The main contributions of this paper are summarized as follows:

• We propose SongEditor, a language model-based generation approach that handles long-content generation guided by user-defined lyrics and structures.   
• SongEditor is among the first works designed not only to generate songs from scratch, but also to conduct sentence-level editing at specific segment (segment-wise) and complete vocal or accompaniment tracks when given the rest (track-wise).   
• A context-free editing method is employed that can effectively reduce the reliance on contextual lyrics and proves to be more suitable for long-content editing. Additionally, considering the complex composition of accompaniment, a series of measures are taken to ensure the consistency of editing segments and the naturalness of transitions.

We evaluated our model using both objective metrics and human evaluations. Results show that our model can generate songs with superior musicality and excellent consistency.

# Related Work

# Zero-Shot Speech Synthesis & Editing

Zero-shot Text-to-Speech (TTS) enables generating speech of unseen speakers based on their enrolled recordings. The current mainstream generation methods can be divided into language-modeling approaches and diffusion approaches. The former (Wang et al. 2023a; Du et al. $2 0 2 4 6$ ; Borsos et al. 2023) proposes to generate discrete codec tokens (De´fossez et al. 2022) autoregressive, while the latter (Shen et al. 2023; Chen et al. 2024b; Yang et al. 2024; Vyas et al. 2023; Wang et al. 2023b) leverages diffusion models to generate internal continuous features.

Moreover, some approaches have extensively explored for the task of speech editing. For instance, VoiceCraft (Peng et al. 2024) proposes a token rearrangement method for the autoregressive generation procedure in neural codec language models. Apart from this, diffusion models (Vyas et al. 2023; Wang et al. 2023b; Du et al. 2024a) naturally possess the infilling editing ability in a non-autoregressive manner. These approaches have achieved a high quality of speech resynthesis. However, since the speech utterances used in these studies are relatively short and simple, they cannot be directly employed for song editing.

# Music Generation

Recently, music generation has gained substantial attention from the research community. Compared to speech, music typically has a greater degree of variation in pitch, melody, tempo, and timbre. To facilitate user-friendly conditional generation without expert acknowledges, text-tomusic has become exceptionally popular. For instance, MusicGen (Copet et al. 2023) and Mustango (Melechovsky et al. 2024) employ the T5 (Raffel et al. 2020) and FlanT5 (Chung et al. 2024) text encoder to process the natural language prompts, and is capable of generating controllable music. Since the paired music and caption data is hard to collect, some approaches (Agostinelli et al. 2023; Lam et al. 2023; Chen et al. 2024a; Evans et al. 2024) leverage crossmodal alignment models (Huang et al. 2022; Elizalde et al. 2023; Wu et al. 2023) to project both music and text into the same embedding space instead. The condition vector derived from music can be used to train generative models in a selfsupervised manner, while the vector from text is used for practical synthesis purposes. Some approaches incorporate musical signals, such as chromagrams (Copet et al. 2023) or beats (Lin et al. 2023) as auxiliary conditions. This additional information assists in making the generation process more controllable.

Several intriguing studies have also focused on music editing. For instance, Music ControlNet (Wu et al. 2024a) receives multiple types of musical signals to achieve aimevarying controls. MusicMagus (Zhang et al. 2024) aims to partially edit specific attributes of generated music by comparing the differences between old and new prompts. Meanwhile, VampNet (Garcia et al. 2023) is designed to complete the masked regions in acoustic token sequences with a nonautoregressive transformer.

# Singing Voice Synthesis

Singing Voice Synthesis (SVS) aims to generate vocals based on both lyrics and musical scores. Benefiting from note pitches and durations, the SVS process becomes more controllable and easier to converge.

Conventional SVS systems primarily focus on incorporating phoneme-level melody controls into the generated acoustic representations. For instance, VISinger (Zhang et al. 2022a,b) adopts a methodology similar to VITS (Kim, Kong, and Son 2021), which learns the alignments between acoustic and textual units. TokSing (Wu et al. 2024b) discretizes raw audio using multiple self-supervised learning models and further enhances melody information during both the training and inference phases. Recent models (Liu et al. 2022; Hwang, Lee, and Lee 2025; He et al. 2023) have demonstrated promising performance for highquality singing voice generation. HiddenSinger (Hwang, Lee, and Lee 2025), for instance, utilizes the latent diffusion model (Rombach et al. 2022) to synthesize regularized intermediate codec representations. Meanwhile, RMSSinger (He et al. 2023) proposes a DDPM-based method (Ho, Jain, and Abbeel 2020) that synthesizes voice using word-level coarse-grained musical scores, alleviating the heavy demand for manual annotations in real-world scenarios.

![](images/b7bf76425dded4e888ec0ff705affd9ca90f72ea6372808772fd02c5c6a38773.jpg)  
and generator.   
track-wise editing (optional).   
Figure 2: The architecture of the proposed SongEditor framework. We train the DiT and RVQ jointly first and then the semantic language model. The multi-source encoder is exclusively used for track-wise editing.

# Base Model: SongLM

Before introducing the editing framework, we would first like to demonstrate the base system, SongLM, for our editing paradigm. Figure 2 presents the architectural framework of SongLM, which accepts multiple lyric sentences and a 10-second acoustic prompt as inputs. The primary components of this framework include a semantic tokenizer, a language model, and a diffusion-based generator. The semantic tokenizer compresses audio waveforms including the acoustic prompt, context, and separated tracks, into discrete semantic tokens. Subsequently, the language model generates semantic tokens in an autoregressive manner. The final waveform is reconstructed from the output sequence by the diffusion generator.

The training process is divided into two distinct phases. Initially, the tokenizer and diffusion generator are trained concurrently. Subsequently, the language model is trained using tokens generated by the tokenizer. During the second phase, both the tokenizer and the diffusion generator remain in a frozen state.

Lyric preprocessor The original data pairs only contain plain text and waveform $\{ ( \breve { X _ { i } } , W _ { i } ) \} _ { i = 1 } ^ { L }$ , where $L$ is the number of lyrics, $X _ { i }$ is the $i$ -th sentence and $W _ { i }$ is the corresponding waveform. It is worth noting that both waveforms and lyrics are consecutive, so they can be concatenated in sequence and $W _ { i }$ may be empty at non-vocal sections. However, relying solely on the text of lyrics is not enough for the complicated song generation task. In order to incorporate the structure information into the lyrics, a structure detector (Kim and $\Nu \mathrm { a m } \ 2 0 2 3 ,$ ) is applied. The category of structure indicators is shown in Appendix A.

Semantic Tokenizer Neural codec tokens have been widely used by previous music generation systems (Huang et al. 2022). However, we found that the quantized semantic tokens (Borsos et al. 2023; Agostinelli et al. 2023) derived from self-supervised training models exhibit a higher compression rate. In this paper, we propose a tokenizer consisting of two distinct branches with different pretrained encoders: MERT (Li et al. 2024) and HuBERT (Hsu et al. 2021). The former focuses on accompaniment while the latter is for vocal. Two Residual Vector Quantizers (RVQ) (Zeghidour et al. 2021; Kumar et al. 2023) are appended to the end of each branch. And each quantizer has two layers, resulting in $K = 4$ tokens in each frame. The semantic token extraction can be formulated as:

$$
Y = { \mathrm { R V Q } } \left( { \mathrm { M E R T } } ( W ) \right) \oplus { \mathrm { R V Q } } \left( { \mathrm { H u B E R T } } ( W ) \right) ,
$$

where $Y ~ \in ~ \mathbb { Z } ^ { T \times K }$ is the discrete token sequence and $T$ is the sequence length. $W$ is the input waveform. $\oplus$ denotes the frame-by-frame concatenation. During the training phase, only the RVQ layers are updated using a commitment loss (van den Oord, Vinyals, and kavukcuoglu 2017):

$$
\mathbb { L } _ { r v q } = \sum ^ { K } \left( | | s g ( e _ { k } ) - z _ { k } | | _ { 2 } ^ { 2 } + | | e _ { k } - s g ( z _ { k } ) | | _ { 2 } ^ { 2 } \right) ,
$$

where $s g ( \cdot )$ is the stop-gradient operation, $z _ { k }$ is the input latent vector and $\boldsymbol { e } _ { k }$ is the nearest codebook entry.

Decoder-only Language Model The condition tensors and semantic token sequence are first concatenated together and then fed into a decoder-only transformer. Each transformer layer contains a causal self-attention mechanism with Rotary Position Embeddings (RoPE) (Su et al. 2024) and a feed-forward block.

Diffusion Generator We utilize the Latent Diffusion Model (LDM)(Rombach et al. 2022) as the generator, comprising a diffusion model, a variational autoencoder (VAE)(Kingma and Welling 2013), and a vocoder. We replace the conventional U-Net (Ronneberger, Fischer, and

Brox 2015) backbone with DiT (Peebles and Xie 2023), which conditions on semantic tokens and applies forward and reverse processes on the latent vectors produced by VAE (van den Oord, Vinyals, and kavukcuoglu 2017). The VAE then converts latent vectors into a Mel spectrogram, which is subsequently transformed into a waveform by the HifiGAN (Kong, Kim, and Bae 2020) vocoder.

# SongEditor

In this section, we will present SongEditor, a novel framework that integrates editing capabilities into a base model. SongEditor enables two types of song editing: segmentwise, which modifies the vocals and accompaniment within a song segment simultaneously, and track-wise, which allows independent editing of the vocals or accompaniment. Importantly, these approaches can be combined, empowering the flexibility to edit song segments on a single track or holistically. In the following, we will detail how SongEditor implements these editing functionalities.

# Long-Content Segment-Wise Editing

For segment-wise editing, we first discuss leveraging contextual information for the segment being edited. Then, we introduce a rearrangement operation similar to VoiceCraft (Peng et al. 2024), applied to the semantic token sequence. To achieve a more natural transition effect, we propose a force-smoothing strategy during training and scorebased candidate selection during inference.

Context Selection The waveform segment to be edited is defined as $W _ { [ L _ { A } : L _ { B } ] } = [ W _ { L _ { A } } , W _ { L _ { A } + 1 } , \cdot \cdot \cdot , W _ { L _ { B } } ]$ , where $1 \leq L _ { A } \leq \dot { L } _ { B } \leq L$ . Here, $L _ { A }$ and $L _ { B }$ denote the start and end sentences. It is worth noting that $L _ { A }$ and $L _ { B }$ may be equal to 1 or $L$ , allowing SongEditor to also perform the continuation or generation task. During the training phase, the preceding and following contexts $y _ { [ 1 : A ) }$ and $y _ { ( B : T ] }$ are directly cut from the token sequence $Y$ , where $A$ and $\dot { B }$ are the start and end indexes of frames. During inference, these contexts are extracted by the semantic tokenizer from the waveforms $W _ { [ 1 : L _ { A } ) }$ and $W _ { ( L _ { B } : L ] }$ respectively, with an overlap of 1 second to avoid edge effects. The overlapped area is subsequently removed after tokenization. For the lyrics input, only the edited sentences are retained, represented as $X = [ \overbar { X } _ { L _ { A } } , X _ { L _ { A } + 1 } , \cdot \cdot \cdot , X _ { L _ { B } } ]$ , which we refer to as a lyric context-free strategy. The choice of not including the lyric context is based on two observations: 1) The contextual connection between lyric sentences is not particularly tight. 2) A context-free approach can achieve more precise control over resynthesized vocals, as contextual lyrics can sometimes lead to incorrect vocal generation.

Additionally, we have observed that the context-free strategy is more suitable for practical applications, such as consecutive generation for songs that exceed the maximum length of training samples. By eliminating the need for context lyrics, the language model can automatically generate a much longer song one step at a time (Agostinelli et al. 2023), without requiring human interception and annotation of the preceding context at each step. This approach enhances both efficiency and usability in practical scenarios.

Rearrangement Assuming the original sequence consists of three segments: $Y = [ y _ { [ 1 : A ) } , y _ { [ A : B ] } , y _ { ( B : T ] } ]$ , during the training phase, the editing segment is moved to the end of the sequence, and a $\langle \mathsf { S E P } \rangle$ token is appended at the end of each segment. Thus, the rearranged sequence should be $Y ^ { r e } = [ y _ { [ 1 : A ) } , y _ { < s > } , y _ { ( B : T ] } , y _ { < s > } , y _ { [ A : B ] } , y _ { < s > } ] ,$ , where $y _ { i } ~ = ~ ( y _ { i , k } ) _ { k = 1 } ^ { K }$ and $y _ { < s > }$ denotes the embedding of the $\langle \mathsf { S E P } \rangle$ token.

A delayed pattern (Copet et al. 2023) is further applied to rearrange the $K$ RVQ tokens. Tokens of the $k$ -th quantizer are moved $( k - 1 )$ timesteps backward and extra $( K - 1 )$ empty frames are appended to each segment in order to prevent overlap of quantizers. Then tokens at the same timestep are stacked together. Consequently, the final length of the language model input should be $( T + 3 K )$ . Because the lyrics of contexts have been discarded, only the training loss of editing segment is calculated as $\mathbb { L } _ { c e } = - \log \left( P _ { \theta } ^ { [ A , B ] } \right)$ , where

$$
\begin{array} { r l r } {  { P _ { \theta } ^ { [ A , B ] } = P _ { \theta } ( y _ { [ A : B ] } \vert y _ { [ 1 : A ) } , y _ { ( B : T ] } , \boldsymbol { X } , Y ^ { s t y } ) } } \\ & { } & { = \prod _ { i = A } ^ { B } \prod _ { i = A } ^ { K } p _ { \theta } ( y _ { i , k } \vert \forall y _ { m , n } , \boldsymbol { X } , Y ^ { s t y } ) , } \end{array}
$$

if $m \in ( 0 , A ) \cup ( B , T )$ or $n + m < i + k$ .

Force-Smoothing Training Due to the presence of melody and rhythm, listeners tend to be more sensitive and strict about interruptions in music compared to speech. However, ensuring smoothness is more challenging for music editing, especially at the endpoint of the editing segment. To tackle this issue, we propose a force-smoothing strategy. During the training phase, the model is enforced to predict additional $\lambda$ frames with a probability of 0.1 for each step even after the editing segment has ended. These frames are directly copied from the beginning of the following contbeyx $y _ { \left( B : T \right] }$ a.nDdutrhineg $P _ { \theta } ^ { [ A , B ] }$ nfcoer, tlhoessmcoaldceullawtilolnfiris rgerpeleadcieldy $P _ { \theta } ^ { [ A , B + \lambda ] }$ determine whether to stop prediction at the current step. As long as the probability of $\left. \cos \right.$ is the largest, the model will immediately output $\left. \cos \right.$ token. Otherwise, a top- $\mathbf { \nabla } \cdot \mathbf { k }$ sampling method will be employed to select the next token randomly from the remaining options. this sampling strategy can effectively alleviate the over-writing issue caused by force-smoothing during training.

Score-Based Candidate Selection To achieve better transitions, a score-based candidate selection (Jiang et al. 2023) is applied during inference. Specifically, we first generate an initial candidate $\hat { y } _ { [ A : B ] } ^ { 1 }$ and resynthesize the last 3 seconds for $( N - 1 )$ times, resulting in a set of $N$ candidates ${ \hat { Y } } _ { [ A : B ] } ~ = ~ \{ { \hat { y } } _ { [ A : B ] } ^ { 1 } , { \hat { y } } _ { [ A : B ] } ^ { 2 } , \cdot \cdot \cdot , { \hat { y } } _ { [ A : B ] } ^ { N } \}$ . Each candidate is used as a new prefix to predict the subsequent $\lambda$ frames. The score of each candidate can be represented as the loglikelihood of $y _ { \left( B : B + \lambda \right] }$ . The candidate with the highest score is ultimately selected, which can be formulated as:

$$
\begin{array} { r } { y = \arg \operatorname* { m a x } _ { \hat { y } \in \hat { Y } _ { [ A : B ] } } P _ { \theta } \left( y _ { ( B : B + \lambda ] } | y _ { [ 1 : A ) } , \hat { y } , X , Y ^ { s t y } \right) . } \end{array}
$$

# Multi-Source Track-Wise Editing

Furthermore, we explore the integration of vocal and accompaniment completion into the model. As depicted in Figure 2, SongEditor utilizes either the separated vocal or accompaniment track as auxiliary context to complete the other. The provided track is processed through a gated multisource encoder and incorporated into the language model. Details of this process are introduced below.

Source Separation The Band-Split RNN (BS-RNN) (Yu et al. 2023; Luo and $\mathrm { Y u } ~ 2 0 2 3 )$ is adopted as source separation module for track-wise editing. BS-RNN splits the spectrogram into multiple subbands and performs bidirectional LSTM layers across both subbands and frames. Given that music typically has a higher sample rate and a broader frequency range, BS-RNN is exceptionally suitable for music source separation. We leverage BS-RNN to separate vocals from the mixture, leaving the remaining as accompaniments.

During the training phase, a white noise $\epsilon \sim \mathbb { N } ( 0 , \sigma ^ { 2 } )$ is added to the separated vocals in order to prevent the potential leakage of remaining music components after separation (Donahue et al. 2023). In this paper, $\sigma$ is set to 0.01, resulting in a signal-to-noise ratio (SNR) of 40dB.

Multi-Source Encoder As shown in Figure 2, to be compatible with various conditions, both vocal and accompaniment share the same tokenizer and embedding layers. The token embedding of each frame can be represented as:

$$
c _ { i } = \Bigg \{ \begin{array} { c c } { \sum ^ { K } c _ { i , k } + t } & { t \in \{ t ^ { M } , t ^ { V } \} } \\ { t } & { t = t ^ { \emptyset } . } \end{array}
$$

where $t ^ { M } , t ^ { V }$ , and $t ^ { \circ }$ are special embeddings indicating the type of source, respectively accompaniment, vocal, and none. $\boldsymbol { c } _ { i , k } \in \mathbb { R } ^ { D }$ is the quantizer embedding and $D$ is the dimension of embedding vectors. If one source is provided, the sequence should be $\mathbf { \boldsymbol { c } } = ( c _ { t } ) _ { t = 1 } ^ { T } \in \mathbb { R } ^ { T \times D }$ ; otherwise, $c \in \mathbb { R } ^ { \mathrm { 1 } \times D }$ . The sequence is passed hrough a multi-layer transformer encoder with RoPE and injected into each transformer decoder layer via cross-attention.

# Experimental Settings

# Datasets

For the training of SongEditor, a large-scale dataset with approximately 700K songs was used, which sums up to 50K hours. A data cleaning pipeline similar to (Yu et al. 2024) was adopted to filter out the low-quality samples and correct the timestamp of lyrics in the preprocessing process. BSRNN is initially employed to extract vocals. Pyannote (Bredin 2023) and WhisperX (Bain et al. 2023) toolkits are then leveraged for voice activity detection (VAD) and singing voice recognition (SVR) respectively. Lyrics with serious mismatches will be discarded. The time boundary will be corrected based on the VAD results and two adjacent lyric sentences with a short gap will be merged.

For objective evaluation, we assembled a test set of 200 randomly selected samples. Each sample ranges from 90 to 120 seconds in length and contains at least one complete verse and chorus. A 10-second prompt is extracted from other parts of the song. For the song editing task, we randomly mask a region, which may be located in the middle, beginning, or end of the song. For subjective evaluation, we randomly select 15 samples with manual correction of annotations, and two recordings are generated based on each sample, which are then assessed by expert musicians.

Table 1: Evaluation results of baseline and our proposed method for song generation.   

<html><body><table><tr><td>Method</td><td>PER(%)↓</td><td>FAD↓</td><td>Musicality↑</td><td>Quality↑</td></tr><tr><td>GT(restore)</td><td>5.51</td><td>0.86</td><td>3.63</td><td>3.59</td></tr><tr><td>SongLM</td><td>20.63</td><td>1.99</td><td>2.95</td><td>3.06</td></tr><tr><td>SongEditor</td><td>18.33</td><td>2.24</td><td>3.02</td><td>3.19</td></tr></table></body></html>

# Evaluation Metrics

To conduct a comprehensive comparison, we evaluated three different models: SongLM, SongEditor, and SongEditor+. SongLM serves as our baseline and can only generate songs from scratch. SongEditor supports segment-wise editing, while SongEditor $^ +$ is capable of both segment- and trackwise editing. Details of them are presented in Appendix B.

The proposed models are evaluated in terms of both objective and subjective metrics. For objective evaluation, both Phoneme Error Rate (PER) and Fre´chet Audio Distance (FAD) (Kilgour et al. 2019) are employed. To calculate PER, the vocal is first extracted by BSRNN. Then the Whisperlarge-v2 is utilized for speech recognition. All punctuation and structure tokens in the lyrics have been removed. The FAD score is computed based on the internal representation from the last layer of MERT-95M, and the fadtk2 toolkit is used for calculating statistics and comparison. For editing tasks, only the editing segments are taken into consideration.

For human evaluation, we conduct a mean opinion score (MOS) listening test. Specifically, we employ 30 listeners to rate each audio sample on a scale from 1 to 5 across five aspects: musicality, quality, coherence, smoothness, and intelligibility. Coherence measures the degree of consistency between the editing part and context. Smoothness refers to the naturalness of transitions. These two scores are only considered for the segment-wise editing task.

In our experiments, we observe that tracks restored from the original audio sometimes significantly reduce the performance of the source separation module, resulting in an abnormally high PER. To adjust the quality of vocals more precisely, we also ask the listeners to assess whether the generated vocals sound clear and match the given lyrics (intelligibility) for a track-wise editing task.

# Results and Analysis

# Song Generation

Table 1 compares the performance of our proposed methods to the baseline for lyric-to-song generation. Additionally, we demonstrate the results of directly restoring audio from ground truth semantic tokens (”restore”), which represents the upper bound of this framework. Incorporating segment-wise editing capability does not degrade the performance of SongLM. Instead, a slight improvement can be observed in terms of subjective MOS scores and PER.

<html><body><table><tr><td rowspan="2">Method</td><td rowspan="2">Context-Free</td><td colspan="2">Objective</td><td colspan="4">Subjective (MOS)</td></tr><tr><td>PER(%)↓</td><td>FAD↓</td><td>Musicality↑</td><td>Quality↑</td><td>Coherence↑</td><td>Smoothness↑</td></tr><tr><td>VoiceCraft (Peng et al. 2024)</td><td>×</td><td>35.23</td><td>1.34</td><td>3.03</td><td>3.17</td><td>3.48</td><td>3.63</td></tr><tr><td>SongEditor</td><td>√</td><td>20.82</td><td>1.30</td><td>3.25</td><td>3.39</td><td>3.82</td><td>3.53</td></tr><tr><td>-cs</td><td>√</td><td>=</td><td></td><td></td><td></td><td></td><td>3.50</td></tr><tr><td>- CS&FS</td><td>√</td><td>25.68</td><td>1.37</td><td>3.13</td><td>3.23</td><td>3.54</td><td>3.43</td></tr></table></body></html>

Table 2: Evaluation results of SongEditor for segment-wise editing. “CS” represents the score-based candidate selection. Since it only affects the local smoothness, other metrics are dismissed. “FS” represents force-smoothing.

# Segment-Wise Editing

Table 2 reports the result of SongEditor for segment-wise editing. To demonstrate the advantage of the context-free strategy used in our model, we adopt VoiceCraft, a contextbased approach for speech editing as our baseline. VoiceCraft takes the complete lyrics of both contexts and modified sentences as input. During training, the start and end points are randomly selected. The proposed SongEditor significantly outperforms the baseline in PER. We further find that sentence-level mistakes, such as repeating or missing, are more likely to happen around the editing segments. We believe these mistakes are caused by the failure to align the boundary of context audio and lyrics. In most of the subjective metrics, especially the coherence score, SongEditor also surpasses the baseline, which proves that the contextfree strategy is more suitable for such a long-content editing task. One notable observation is that the smoothness score deteriorates for our proposed model. We attribute this decline to the potential inaccuracies in the annotation of temporal boundaries within the training data.

An ablation study was conducted to verify the effectiveness of force-smoothing training. We retrained SongEditor for the same number of steps without force-smoothing. As shown in Table 2, all metrics declined to varying degrees. This may suggest that force-smoothing training not only enables the model to produce smoother transitions but also helps it learn long-term dependencies in the following context. Additionally, with the score-based candidate selection during inference, SongEditor can achieve a higher smoothness score even without further training.

Figure 3 compares the Mel spectrograms of the transition areas. We randomly generated several samples with the same context. Since the autoregressive generation progresses from left to right, the transition from the preceding context to the edited segment is relatively straightforward, so we primarily focus on transitions from generated to subsequent audio. A 10-second audio clip is captured around the end point of each sample. As shown in Figure 3, after applying forcesmoothing and candidate selection, the Mel spectrograms of the generated samples appear smoother, with high-energy bands becoming more continuous. In contrast, distinct discontinuities or mismatches can be observed at the transition points without force-smoothing.

![](images/f7945ed39adde0652a117629b5a8a61c411576cd9599e44ff7fec247183398c7.jpg)  
Figure 3: Mel spectrograms of transitions. The center of the red box is the transition point. The left half is generated while the right half is restored from ground truth.

# Track-Wise Editing

Moreover, we evaluate SongEditor with cross-attention layers on the track-wise editing task. In this section, we primarily use the Intelligibility MOS score to assess the correctness of the generated vocals, as combining restored and generated tracks can result in abnormal PER outcomes. Our experiments investigate the influence of different source tracks by testing various configurations: vocals only (vocal-to-song), accompaniment only (accomp-to-song), and neither of them. Track-wise information is integrated into both the generation and segment-wise editing tasks.

As shown in Table 3, both vocals and accompaniments boost performance in terms of musicality and FAD scores. The vocal track plays a dominant role in track-wise editing, which is expected since vocals often serve as the backbone of a song, directly conveying crucial elements such as rhythm and emotion. Additionally, listeners naturally focus more on vocals. Conversely, generating vocals from accompaniments presents greater challenges, as the model must maintain precise rhythm synchronization with accompaniments, resulting in a slight decrease in the intelligibility score as a trade-off. Notably, the FAD score for vocal-tosong generation from scratch is considerably high (2.71) but

<html><body><table><tr><td colspan="2">input</td><td rowspan="2">w/ context</td><td colspan="2">Objective</td><td colspan="5">Subjective (MOS)</td></tr><tr><td>VA</td><td></td><td>PER(%)↓</td><td>FAD↓</td><td>Musicality↑</td><td>Quality↑</td><td>Coherence↑</td><td>Smoothness↑</td><td>Intelligibility↑</td></tr><tr><td></td><td>×</td><td>×</td><td>1</td><td>2.71</td><td>3.38</td><td>3.54</td><td></td><td></td><td>4.35</td></tr><tr><td></td><td>×<</td><td>×</td><td>1</td><td>1.29</td><td>2.83</td><td>2.90</td><td></td><td></td><td>3.74</td></tr><tr><td>×</td><td>×</td><td>×</td><td>19.30</td><td>2.61</td><td>2.68</td><td>2.92</td><td>1</td><td>1</td><td>3.98</td></tr><tr><td>√</td><td>×</td><td>√</td><td>1</td><td>1.13</td><td>3.56</td><td>3.66</td><td>4.19</td><td>3.99</td><td>4.33</td></tr><tr><td></td><td>×<</td><td>√</td><td>1</td><td>1.17</td><td>3.21</td><td>3.06</td><td>3.48</td><td>3.72</td><td>3.53</td></tr><tr><td>×</td><td>×</td><td>√</td><td>23.72</td><td>1.32</td><td>2.93</td><td>3.06</td><td>3.33</td><td>3.47</td><td>3.57</td></tr></table></body></html>

Table 3: Performance of SongEditor+ with different source tracks. “V” is the abbreviation of vocal and “A” is accompaniment The upper part compares the performance for pure generation, while the lower part introduces additional contextual information

![](images/64ce5c41cca30eb6f17915d19d4f7a52a884f72860bf2960623806266631371a.jpg)  
Figure 4: An example of track-wise editing. The Mel spectrogram above corresponds to the separated accompaniment, while the below corresponds to the vocal. Since the spectrogram of accompaniment is more complex and difficult to identify, its chroma change trend is plotted (red line).

drops significantly when contextual information is provided (1.13). This suggests that the model can learn more detailed accompaniment features when contexts are available.

Figure 4 illustrates the result of music separation for both reference and generated audio. Our goal is to study the consistency of the regenerated audio with the original source input. Specifically, we first separate the reference audio into vocals and accompaniment, then use SongEditor+ to independently complete each track. As shown in the figure, the generated vocal spectrogram closely resembles the input in the vocal-to-song task. Conversely, for accomp-to-song, while the restored chromagram generally remains consistent, some differences can be observed in both the chromagram and spectrogram. We believe that vocals are relatively independent and easier to compress, while accompaniments often rely on vocals and contain more detailed information.

# Consecutive and Multi-Singer Story Mode

As highlighted in Section 4, the context-free strategy demonstrates significant advantages in practical applications. In this section, we explore generating full songs longer than two minutes through an iterative round-by-round process, referred to as Story Mode, following Agostinelli et al. (2023). The complete lyrics are pre-divided into several sections. During each round, only the lyrics of the current section are provided, while a fixed-length token sequence is automatically extracted from the previous output as the prefix. No manual intervention is required throughout the process.

Specifically, we adopt two task settings: full song generation and multi-singer story mode. The former aims to generate a full song from intro to outro, without changing the style prompt. This task focuses on the coherence across rounds, and we advance with a stride of 60 seconds. For the latter, we use two different style prompts, typically one for male and one for female. Verse and chorus sections are generated with alternate prompts. In order to prevent the emergence of vocal in the prefix, a short instrumental segment is generated at the end of each round and the stride length is set to 5 seconds. Experiments demonstrate that our model can generate full songs with smooth transitions and coherent content. By slightly modifying the task settings, it can also achieve multi-singer generation with variational style prompts. Audio samples are presented in the demo page.

# Limitations and Ethic Discussion

Limitations: SongEditor currently regenerates specified sentences without providing additional control over the editing segment. Furthermore, there is no explicit decoupling of tracks at the semantic level, which could enhance interpretability and improve the distinction.

Ethics: SongEditor is an innovative tool that helps either a professional musician or a passionate enthusiast to create their own songs. However, we also fully acknowledge the potential ethical risks. We ensure that our training data does not infringe on any copyright, and only public-domain melodies are used for inference.