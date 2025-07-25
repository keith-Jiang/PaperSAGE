# StableVC: Style Controllable Zero-Shot Voice Conversion with Conditional Flow Matching

Jixun $\mathbf { Y a o } ^ { 1 }$ , Yang Yuguang2, Yu Pan2, Ziqian $\mathbf { N i n g ^ { 1 } }$ , Jianhao $\mathbf { Y e } ^ { 2 }$ , Hongbin Zhou2, Lei Xie1\*

1 Audio, Speech and Language Processing Group (ASLP $@$ NPU), Northwestern Polytechnical University, China 2Ximalaya Inc, China {yaojx,ningziqian,lxie} $@$ mail.nwpu.edu.cn, {yuguang.yang,yu.pan,jianhao.ye,hongbin.zhou}@ximalaya.com

# Abstract

Zero-shot voice conversion (VC) aims to transfer the timbre from the source speaker to an arbitrary unseen speaker while preserving the original linguistic content. Despite recent advancements in zero-shot VC using language modelbased or diffusion-based approaches, several challenges remain: 1) current approaches primarily focus on adapting timbre from unseen speakers and are unable to transfer style and timbre to different unseen speakers independently; 2) these approaches often suffer from slower inference speeds due to the autoregressive modeling methods or the need for numerous sampling steps; 3) the quality and similarity of the converted samples are still not fully satisfactory. To address these challenges, we propose a Style controllable zero-shot VC approach named StableVC, which aims to transfer timbre and style from source speech to different unseen target speakers. Specifically, we decompose speech into linguistic content, timbre, and style, and then employ a conditional flow matching module to reconstruct the high-quality melspectrogram based on these decomposed features. To effectively capture timbre and style in a zero-shot manner, we introduce a novel dual attention mechanism with an adaptive gate, rather than using conventional feature concatenation. With this non-autoregressive design, StableVC can efficiently capture the intricate timbre and style from different unseen speakers and generate high-quality speech significantly faster than real-time. Experiments demonstrate that our proposed StableVC outperforms state-of-the-art baseline systems in zero-shot VC and achieves flexible control over timbre and style from different unseen speakers. Moreover, StableVC offers approximately $2 5 \times$ and $1 . 6 5 \times$ faster sampling compared to autoregressive and diffusion-based baselines.

# Introduction

Zero-shot voice conversion (VC) aims to transfer the timbre from the source speaker to an arbitrary unseen speaker while keeping the linguistic content unchanged. Currently, numerous impressive zero-shot VC techniques demonstrate remarkable efficacy in converting realistic and natural sound samples, finding crucial applications in both professional audiobook production and entertainment short videos (Popov et al. 2021; Ning et al. 2023b; Yao et al. 2023; Li et al. 2024;

![](images/d82665ff95f809a40e4a0e4e5b333327337e07fe3e833bdf3bb3fd261c0ea6a4.jpg)  
Figure 1: The concept of style-controllable zero-shot voice conversion. It aims to build a VC system capable of adapting timbre to unseen speakers and transferring the style to another unseen speaker. Here, “unseen” refers to speakers not present in the training set.

Choi, Lee, and Lee 2024). However, these approaches primarily focus on adapting timbre to unseen speakers while often overlooking style attributes. Style refers to how a person expresses themselves, including pitch variation, speaking rate, intonation, and emotion. It can vary significantly across different contexts to enhance speech expressiveness. Additionally, most current approaches come at a cost, such as more complicated training setups and computationally expensive autoregressive formulations.

Inspired by the recent successes in the zero-shot capabilities of large-scale language models like GPT (Radford et al. 2019), a similar approach is widely believed to achieve comparable results in zero-shot speech generation. Notably, recent works such as LM-VC (Wang et al. 2023b) can synthesize high-quality personalized speech with a 3-second enrolled recording from an unseen speaker. However, the style information (such as speaking rate, tone, etc.) is also derived from the 3-second prompt and cannot be independently controlled. An ideal approach would allow for the simultaneous conversion of both timbre and style from the source speech to various arbitrary unseen speakers, enabling free combination, as illustrated in Figure 1. Although some works (Yao et al. 2024; Li, Han, and Mesgarani 2023b; Yuan et al. 2021) have investigated stylistic voice conversion, most focus on many-to-many scenarios or directly serve timbre as a particular style. Converting both style and timbre to different unseen speakers simultaneously remains a significant challenge in the field.

Inference speed is another challenge of current zero-shot VC approaches. Although conventional VC frameworks can achieve faster speed (Qian et al. 2019; Wang et al. 2021; Ning et al. 2023a; Casanova et al. 2022), the quality and similarity in zero-shot scenarios are unsatisfactory. Recent typical works in zero-shot VC can be divided into two categories: large-language model-based (Wang et al. 2023a,b) and diffusion-based approaches (Popov et al. 2021; Choi, Lee, and Lee 2024). The large-language model-based approach employs a neural audio codec to encode continuous speech waveforms into discrete tokens and training models using a ”textless NLP” approach (Borsos et al. 2023). It demonstrates an impressive ability for in-context learning and performs well in zero-shot speech generation tasks. However, autoregressive token prediction inherently has a slow inference speed, and the discrete token only contains compressed information of the original waveform, hindering its performance on tasks that require high quality and expression (Lee et al. 2024). On the other hand, the diffusion-based approach is also present in many state-ofthe-art speech and audio generation models (Shen et al. 2023; Liu et al. 2023) but requires multiple reverse steps during inference, which also be computationally intensive.

To achieve style-controllable VC and tackle inference speed problems, we propose StableVC, a style-controllable zero-shot VC model that is fast and efficient. StableVC first disentangles speech into content, timbre, and style, and then employs a flow matching generative module to reconstruct high-quality mel-spectrograms. The flow matching module consists of multiple diffusion transformer (DiT) blocks and is trained to predict a vector field, which efficiently models the probabilistic distribution of the target mel-spectrograms. For attribute disentanglement, we use a pre-trained selfsupervised model with K-means clustering to extract linguistic content, while a factorized codec is used to extract style representation. Instead of using conventional concatenation for timbre and style modeling, we propose a novel dual attention mechanism with an adaptive gate control to capture timbre and style information effectively. An adaptive style gate and strong timbre prior information are introduced to ensure the stability of timbre and style modeling. Therefore, we can convert timbre to the target unseen speaker while flexibly controlling style using another speaker’s reference. Audio samples can be found in demo pages 1. The main contributions of this study can be summarized:

• We propose StableVC, a novel style-controllable zeroshot voice conversion approach. To the authors’ knowledge, this is the first approach that can independently convert the timbre and style to different unseen speakers, enabling any combination of timbre and style transfer. • We introduce a conditional flow matching module for probability-density path prediction conditioned on the timbre and style, significantly improving synthesis speed and sample quality compared to conventional diffusionbased and language model-based approaches.

• We propose a dual attention mechanism with adaptive gate control, called DualAGC, in the flow matching module to capture distinct timbre and style. An adaptive style gate and timbre prior information are incorporated to ensure the stability of timbre and style modeling.

# Related Work

# Zero-shot Voice Conversion

A popular paradigm for zero-shot VC involves tokenizing speech waveforms into discrete semantic and acoustic tokens using a self-supervised learning (SSL) model and a neural audio codec, respectively. For example, LMVC (Wang et al. 2023b) employs a two-stage language modeling approach: first generating coarse acoustic tokens to recover the source linguistic content and target speaker’s timbre, and then reconstructing fine acoustic details as converted speech. Similarly, several zero-shot VC frameworks (Yang et al. 2024; Li et al. 2024; Hussain et al. 2023; Wang et al. 2022) use semantic representations extracted from SSL models to disentangle the linguistic content. GR0 (Wang et al. 2024) uses a generative SSL framework to jointly learn a global speaker embedding and a zeroshot voice converter, achieving zero-shot VC. Another approach (Luo and Dixon 2024) decouples speech into local and global representations and employs dropout with multiplicative Gaussian noise to apply an information bottleneck for timbre disentanglement. Although these approaches achieve zero-shot VC, they mainly focus on capturing the timbre and overlook the style information.

Several works have dedicated significant efforts to exploring the simultaneous modeling of fundamental frequency (F0) and timbre (Choi, Lee, and Lee 2024). For example, DVQVC (Li, Li, and Li 2023) and SLMGAN (Li, Han, and Mesgarani 2023a) concatenate F0 with linguistic content and speaker timbre to reconstruct the target speech. Diff-HierVC (Choi, Lee, and Lee 2023) introduces a hierarchical system to generate F0 with the target voice style and convert the speech based on the generated F0. Additionally, VoiceShopVC (Anastassiou et al. 2024) employs a conditional diffusion backbone model with optional normalizing flow-based modules to achieve speaker attribute editing. However, these works primarily focus on F0 not style and scenarios are many-to-many, which can not adapt timbre and style from different unseen speakers.

# Flow Matching

Flow matching generative models (Lipman et al. 2022) estimate the vector field of the transport probability path from noise to the target distribution. These models learn the transport path using an ordinary differential equation (ODE) to find a straight path that connects the noise and target samples, thereby reducing transport costs and requiring fewer sampling steps. Compared to conventional denoising diffusion models like DDPM (Ho, Jain, and Abbeel 2020), flow matching provides more stable training and superior inference speed. This technique has shown excellent performance in accelerating image generation (Esser et al. 2024).

![](images/3859fbf54b5241dc94f2cfccac2b4a91c6f8a2d449c8431219945b3afe7658e6.jpg)  
Figure 2: The overall framework of StableVC includes three feature extractors for style, linguistic content, and mel-spectrogram extraction. It also incorporates a content module and a duration module to re-predict the duration based on different styles and timbres, and a flow matching module generates high-quality speech at speeds significantly faster than real-time.

In speech generation, Voicebox (Le et al. 2024) leverages flow matching to build a large-scale, text-conditioned speech generative model. Its successor, Matcha-TTS (Mehta et al. 2024), adopts an encoder-decoder architecture and utilizes optimal transport flow matching for model training. VoiceFlow (Guo et al. 2024) treats the generation of mel-spectrograms as an ordinary differential equation conditioned on text inputs, incorporating rectified flow matching to boost sampling efficiency. Despite these advancements, flow matching has not yet been explored in zero-shot VC to enhance converted quality and inference speed.

# StableVC

# Overview

We aim to provide flexible style control capabilities in zeroshot VC with high quality and efficiency. The overall framework of StableVC is illustrated in Figure 2. First, we extract linguistic content, style representation, and mel-spectrogram from the waveform. To disentangle linguistic content, we use a pre-trained WavLM model2 and apply K-means clustering to extract discrete tokens (Chen et al. 2022), setting the number of K-means clusters to 1024. Additionally, we deduplicate adjacent tokens and replace them with the corresponding K-means embeddings. This process is the same as described in Yao et al. 2024 and has demonstrated its effectiveness in linguistic content extraction. The purpose of deduplication is to re-predict the duration for each token, allowing the duration of the same linguistic content to adapt to different timbre and styles. For style representation, we use a factorized codec3 as style extractor. This model can extract disentangled subspace representations of speech style. For timbre representation, we extract the mel-spectrogram from multiple reference speeches by the same speaker as the source speech. The details of timbre and style modeling will be discussed in the following section.

The disentangled features are encoded through the content module, which consists of multiple DiT blocks. This is followed by the duration module, which predicts the duration and aligns the hidden representations to the frame level. The output from the duration module is transformed into melspectrograms using the flow matching module, which comprises multiple DiT blocks with timestep fusion. Style and timbre are modeled by the DualAGC within each DiT block. Finally, the generated mel-spectrograms are reconstructed into waveforms using an independently trained vocoder.

# DualAGC

The widely used DiT block employs adaptive layer norm or cross-attention to append external conditions, typically including only timesteps and one instructive condition (Peebles and Xie 2023). In this section, we introduce DualAGC, which incorporates a dual attention approach to capture style information and speaker timbre simultaneously. To improve the stability of timbre modeling, we introduce timbre prior information in the dual attention mechanism. Additionally, an adaptive gate mechanism is employed to gradually integrate style information into the content and timbre. The detail of DualAGC in the DiT block is shown in Figure 3.

Suppose the input of the DiT block is $c$ and the outputs of the style encoder and mel extractor are $s$ and $p$ , respectively. We add a FiLM layer (Perez et al. 2018) before the DiT block to apply an affine feature-wise transformation conditioned on timestep $t$ . The quantized style embeddings extracted from the style extractor are passed through the style encoder, which consists of an average pooling layer and several convolutional blocks. The style encoder compresses the quantized embedding four times in the time dimension and captures the correlation of style information (Jiang et al. 2024). $p$ is the mel-spectrogram extracted from multiple reference speech, while $\bar { c _ { q } }$ represents the hidden representation obtained by query projection and query-key normaliza

tion (Henry et al. 2020).

Timbre Attention: Our objective is to extract finegrained information from reference speech that represents the speaker’s timbre. We employ $\bar { c _ { q } }$ as the attention query and use the mel-spectrogram extracted from multiple references as the value of timbre attention. The cross-attention mechanism is agnostic to input positions, effectively resembling a temporal shuffling of the mel-spectrogram. This process preserves a significant amount of speaker information while minimizing other details, such as linguistic content, enabling the cross-attention mechanism to focus on learning and capturing speaker timbre from the reference speech (Li et al. 2024). To improve the stability of timbre modeling, we extract a global speaker embedding from a pre-trained speaker verification model and concatenate it with the melspectrogram as the attention key. The extracted speaker embedding serves as an instructive timbre prior, guiding the timbre attention to capture timbre-related information.

![](images/dd3a88a41c5dc1aeb1d421998bcb899995d7b5ca159249afffe2c1dbed9a5963.jpg)  
Figure 3: Details of DualAGC in the DiT block.

Style Attention: The compressed style representations $s$ serve as both the key and value in style attention, with $\bar { c _ { q } }$ as the attention query. To gradually inject style information into linguistic content and timbre, we propose an adaptive gating mechanism and employ a zero-initialized learnable parameter $\alpha$ as the gate to control the injection process. Given the style keys $s _ { k }$ and values $\boldsymbol { s } _ { v }$ with timbre keys $p _ { k }$ and values $p _ { v }$ , the final output $O$ of DualAGC is formulated as:

$$
O = \tau ( \frac { \bar { c _ { q } } ( \mathrm { v p } \oplus \bar { p _ { k } } ) ^ { T } } { \sqrt { d } } ) p _ { v } + \mathrm { t a n h } ( \alpha ) \tau ( \frac { \bar { c _ { q } } \bar { s _ { k } } ^ { T } } { \sqrt { d } } ) s _ { v } ,
$$

where $\bar { s _ { k } }$ and $\bar { p _ { k } }$ stand for applying query-key norm, $\tau$ is the softmax function. $\oplus$ and vp represent concatenation and the global speaker embedding extracted from the pre-trained speaker verification model. The style injection process allows for adaptive style modeling without impacting the final performance of timbre modeling.

# Conditional Flow Matching

In this section, we first introduce the probability-density path generated by a vector field and then lead into the training objective used in our flow matching module.

Flow matching (Lipman et al. 2022; Tong et al. 2023) is a method used to fit the time-dependent probability path between the target distribution $p _ { 1 } ( x )$ and a standard distribution $p _ { 0 } ( x )$ . It is closely related to continuous normalizing flows but is trained more efficiently in a simulation-free fashion. The flow $\phi : [ 0 , 1 ] \times \mathbb { R } ^ { d }  \mathbf { \bar { \mathbb { R } } } ^ { d }$ is defined as the mapping between two density functions using the ODE:

$$
\frac { d } { d t } \phi _ { t } ( x ) = v _ { t } \left( \phi _ { t } ( x ) \right) ; \quad \phi _ { 0 } ( x ) = x
$$

where $\boldsymbol { v } _ { t } ( \boldsymbol { x } )$ represents the time-dependent vector field and is also a learnable component. To efficiently sample the target distribution in fewer steps, we employ conditional flow matching with optimal transport as specified in Tong et al. 2023. Since it is difficult to determine the marginal flow in practice, we formulate it by marginalizing over multiple conditional flows as follows:

$$
\phi _ { t , x _ { 1 } } ( x ) = \sigma _ { t } \left( x _ { 1 } \right) x + \mu _ { t } \left( x _ { 1 } \right) ,
$$

where $\sigma _ { t } ( x )$ and $\mu _ { t } ( x )$ are time-conditional affine transformations used to parameterize the transformation between distributions $p _ { 1 } ( x )$ and $p _ { 0 } ( x )$ . For the unknown distribution $q ( x )$ over our training data, we define $p _ { 1 } ( x )$ as the approximation of $q ( x )$ by perturbing individual samples with small amounts of white noise with $\sigma _ { \mathrm { m i n } } = 0 . 0 0 0 1$ .

Therefore, we can specify our trajectories with simple linear trajectories as follows:

$$
\mu _ { t } ( x ) = t x _ { 1 } , \sigma _ { t } ( x ) = 1 - ( 1 - \sigma _ { \operatorname* { m i n } } ) t .
$$

The final training objective of the vector field using the conditional flow matching is denoted as :

$$
\mathcal { L } _ { \mathrm { C F M } } = \mathbb { E } \left. v _ { t } \left( \phi _ { t , x _ { 1 } } \left( x _ { 0 } \right) ; h \right) - \left( x _ { 1 } - \left( 1 - \sigma _ { \operatorname* { m i n } } \right) x _ { 0 } \right) \right. ^ { 2 }
$$

where $h$ is the conditional set containing the output of the duration module and the style and timbre representations.

Conditional flow matching encourages simpler and straighter trajectories between source and target distributions without the need for additional distillation. We sample from standard Gaussian distribution $p _ { 0 } ( x )$ as the initial condition at $t = 0$ . Using 10 Euler steps, we approximate the solution to the ODE and efficiently generate samples that match the target distribution.

# Training Objectives

In practice, even though the style representation extracted from the factored codec primarily contains style-related information, there may still be potential timbre leakage. To mitigate this risk and achieve better disentanglement between style and timbre, we employ an adversarial classifier with a gradient reversal layer (GRL) (Ganin and Lempitsky 2015) to eliminate potential timbre information in the output of the style encoder. We first average the output of the style encoder into a fixed-dimensional global representation and employ an additional speaker classifier to predict its speaker identity. The GRL loss can be denoted as follows:

$$
\mathcal { L } _ { \mathrm { G R L } } = \mathbb { E } [ - l o g ( C _ { \theta } ( I \mid \operatorname { a v g } ( s ) ) ) ] ,
$$

where $C _ { \theta }$ and $I$ represent the speaker classifier and speaker identity label. The gradients are reversed to optimize the style encoder to eliminate potential timbre information.

The duration module is trained to re-predict the duration (deduplicated length in the content extractor) conditioned on the style and timbre representations. This enables the generated waveform to adapt different durations based on the different timbre and style. The training objective ${ \mathcal { L } } _ { \mathrm { d u r } }$ for the duration module is to minimize the mean squared error with respect to the log-scale predicted duration and the groundtruth duration. The overall training loss of StableVC is:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { C F M } } + \mathcal { L } _ { \mathrm { d u r } } + \lambda \mathcal { L } _ { \mathrm { G R L } } , } \end{array}
$$

where $\lambda$ is the hyper-parameter used to balance the loss term and we set $\lambda = 0 . 1$ .

# Experimental Setup

# Configuration

We train StableVC on 8 NVIDIA 3090 GPUs for 800K iterations with a total batch size of 128 and the AdamW optimizer is used with a learning rate of 0.0001. During inference, we sample the target mel-spectrograms using 10 Euler steps in the flow matching module with a guidance scale of 1. The mel-spectrograms are reconstructed to waveform by Hifi-GAN vocoder (Kong, Kim, and Bae 2020).

# Datsets

We conduct our experiments on the LibriLight dataset (Kahn et al. 2020), which consists of 60k hours of speech data. For model training, we use samples longer than 5 seconds and filter out low-quality samples using DNSMOS P.808 scores (Reddy, Gopal, and Cutler 2022), resulting in a $2 0 \mathrm { k }$ hours subset. For evaluation, we use the VCTK corpus (Veaux et al. 2019) and ESD corpus (Zhou et al. 2022), ensuring no speaker or style overlap with the training data. The timbre references are selected from the VCTK dataset, consisting of 10 male and 10 female speakers. The style references are selected from the ESD dataset and we chose two high-intensity samples for each of five different styles, resulting in 200 reference samples.

# Baseline Systems

We conduct a comparative analysis of the performance in zero-shot voice conversion between our proposed StableVC approach and several baseline systems, encompassing the following system: 1) StyleVC (Hwang, Lee, and Lee 2022): a style voice conversion system that employs adversarial style generalization; 2) LM-VC (Wang et al. 2023b): a twostage language model based approach for zero-shot voice conversion; 3) VALLE-VC (Wang et al. 2023a), a language model based speech synthesis approach where we replace the original phoneme input with the same content representation in StableVC to enable voice conversion capabilities; 4) $\mathrm { N S 2 V C ^ { 4 } }$ , a voice conversion version of NaturalSpeech2 (Shen et al. 2023); 5) DDDM-VC (Choi, Lee, and Lee 2024), a high-quality zero-shot voice conversion system based on decoupled denoising diffusion models; 6) SEFVC (Li et al. 2024), a speaker embedding free zero-shot voice conversion model.

# Evaluation Metrics

Subjective Metrics: We employ the naturalness mean opinion score (nMOS) to evaluate the naturalness of the generated samples. Additionally, we use two similarity mean opinion scores (sMOS-s and sMOS-p) to evaluate style similarity and timbre similarity, respectively.

Objective Metrics: For objective evaluation, we employ cosine distance (SECS) for speaker similarity, word error rate (WER) for robustness and MOS predicted by neural network (UTMOS) for speech quality. In specific, 1) SECS: we employ the WavLM-TDCNN speaker verification model5 to evaluate speaker similarity between generated samples and the target speaker reference; 2) WER: we use a pretrained CTC-based ASR model6 to transcribe the generated speech and compare with ground-truth transcription; 3) UTMOS (Saeki et al. 2022): a MOS prediction system that ranked first in the VoiceMOS Challenge $2 0 2 2 ^ { 7 }$ . It is used to estimate the speech quality of the generated samples.

For inference latency, we calculate the real-time factor (RTF) on a single NVIDIA 3090 GPU to compare latency across systems. Additionally, we compute the number of model parameters (#Param.) for reference. Since all baseline systems use SSL models to extract representations and employ a vocoder or codec to reconstruct the waveform, we only count the parameters of the acoustic model. We employ two pitch-related metrics for the evaluation of style transfer: Root Mean Squared Error (RMSE) and Pearson correlation (Corr) (Sedgwick 2012). These metrics are widely used in the evaluation of stylistic VC. Since the sequences between the reference and the generated samples are not aligned, we perform Dynamic Time Warping (Mu¨ller 2007) to align the sequences before comparison.

<html><body><table><tr><td></td><td>nMOS ↑</td><td>sMOS-p ↑</td><td>UTMOS↑</td><td>WER↓</td><td>SECS↑</td><td>RTF↓</td><td>#Param.</td></tr><tr><td>GT</td><td>4.33±0.04</td><td>1</td><td>4.24</td><td>1.61</td><td></td><td></td><td>1</td></tr><tr><td>LMVC</td><td>3.01±0.05</td><td>3.17±0.07</td><td>3.32</td><td>4.17</td><td>0.61</td><td>3.891</td><td>305M</td></tr><tr><td>VALLE-VC</td><td>3.56±0.05</td><td>3.65±0.04</td><td>3.63</td><td>3.08</td><td>0.55</td><td>3.944</td><td>302M</td></tr><tr><td>StyleVC</td><td>3.24±0.07</td><td>3.21±0.07</td><td>3.41</td><td>5.21</td><td>0.43</td><td>0.075</td><td>31M</td></tr><tr><td>NS2VC</td><td>3.32±0.05</td><td>3.16±0.05</td><td>3.49</td><td>4.88</td><td>0.44</td><td>0.337</td><td>435M</td></tr><tr><td>DDDM-VC</td><td>3.67±0.07</td><td>3.61±0.06</td><td>3.75</td><td>3.07</td><td>0.51</td><td>0.287</td><td>66M</td></tr><tr><td>SEF-VC</td><td>3.63±0.04</td><td>3.72±0.05</td><td>3.59</td><td>2.89</td><td>0.53</td><td>0.168</td><td>260M</td></tr><tr><td>StableVC</td><td>3.96±0.04</td><td>4.04±0.05</td><td>4.12</td><td>2.03</td><td>0.67</td><td>0.146</td><td>166M</td></tr></table></body></html>

Table 1: The subjective and objective evaluation results for StableVC and the baseline systems in zero-shot voice conversion.   
All subjective metrics are computed with $9 5 \%$ confidence intervals and “GT” refers to ground truth samples.

# Experimental Results

# Experimental Results on Zero-shot VC

In this subsection, we first evaluate the performance in zeroshot voice conversion and compare StableVC with baselines in terms of speech naturalness, quality, speaker similarity, robustness, and inference latency. The evaluation results of both subjective and objective metrics are shown in Table 1.

For the naturalness of the converted samples, StableVC outperforms the baseline systems by a substantial margin, achieving nMOS scores closest to the ground truth samples. Regarding speech quality, StableVC attains the highest UTMOS score of 4.12, surpassing all baseline systems and showing only a slight decline compared to the ground truth. These results demonstrate that samples converted by our proposed StableVC are both natural and high-quality.

The speaker similarity results show that: 1) StableVC achieves an sMOS-p score of 4.05 and an SECS of 0.67, reflecting the high speaker similarity of the generated sample; 2) most baseline systems only obtain around 0.5 of SECS, which is significantly lower than StableVC, while the sMOS-p for baseline systems also reflects a similar trend. These findings highlight the superiority of our approach in timbre modeling and also confirm that using attention mechanisms to capture fine-grained timbre details leads to a substantial enhancement in speaker similarity.

We evaluate the robustness of StableVC in zero-shot VC by computing the WER of converted speech samples, as shown in Table 1. The results show that: 1) StableVC achieves a 2.03 WER, only about 0.4 higher than the ground truth samples, proving its high robustness and intelligibility; 2) StableVC outperforms other baselines by a considerable margin, demonstrating the superior robustness of StableVC.

Table 2: Objective metrics comparison of zero-shot style transfer between StableVC and baseline systems.   

<html><body><table><tr><td>Model</td><td>Corr↑</td><td>RMSE↓</td><td>UTMOS ↑</td><td>WER↓</td><td>SECS ↑</td></tr><tr><td>StyleVC</td><td>0.71</td><td>14.98</td><td>3.63</td><td>3.54</td><td>0.21</td></tr><tr><td>NS2VC</td><td>0.67</td><td>17.19</td><td>3.57</td><td>5.31</td><td>0.47</td></tr><tr><td>DDDM-VC</td><td>0.69</td><td>15.68</td><td>3.71</td><td>3.56</td><td>0.50</td></tr><tr><td>StableVC</td><td>0.75</td><td>12.87</td><td>4.06</td><td>2.12</td><td>0.64</td></tr></table></body></html>

To compare inference latency, we measure the RTF against various baseline systems and count the total trainable parameters of each system for reference. The results show that StableVC achieves approximately a $2 5 . 3 \times$ speedup over the language model-based approach and about a $1 . 6 5 \times$ speedup over the denoising diffusion-based approach, while consistently surpassing these baseline systems in all subjective and objective metrics. This demonstrates that StableVC is both effective and efficient. Although StyleVC has the lowest RTF and smallest model parameters, its performance on each metric is not ideal. Moreover, unlike the autoregressive baseline, where generation time is proportional to the source speech duration, the non-autoregressive StableVC maintains similar latency regardless of the length. These results highlight the efficiency of StableVC in zeroshot voice conversion, offering a significant advantage in terms of both speed and performance.

# Experimental Results on Style Transfer

In this subsection, we evaluate the performance of zeroshot style transfer, where the timbre and style prompts are from different unseen speakers. Since some baseline systems do not support both timbre and style prompts, we select StyleVC, NS2VC, and DDDM-VC as baseline systems. For these systems, we use the F0 extracted from the style reference as the style prompt and compare these baselines with StableVC using subjective and objective metrics.

![](images/f1d6c2e8810fda02ed622dcf2dce7db0d0d79edb9083949b111b7a2338e4da8e.jpg)  
Figure 4: Violin plots for timbre and style similarity of speech generated by baseline systems and StableVC.

Table 2 shows the objective metrics comparison between StableVC and baseline systems. We have the following observations: 1) StableVC outperforms baseline systems across all measured metrics, including pitch-related metrics, speech quality, and speaker similarity; 2) StyleVC achieves comparable results in style transfer but struggles with effectively transferring the target timbre. The lower speaker timbre similarity in the converted samples of StyleVC is primarily due to timbre leakage during the style modeling process. This leakage causes the style features to carry some timbre information, thereby compromising the accuracy of the timbre conversion.

Furthermore, we visualize the subjective evaluation results of timbre similarity and style similarity using violin plots, as shown in Figure 4. These plots indicate that the median values for both style similarity and speaker similarity from StableVC are higher than those of the three baseline systems. Additionally, the overall distribution of subjective scores for our proposed StableVC is superior to the baseline systems. Notably, the speaker similarity scores for StyleVC are significantly lower in subjective evaluations.

The subjective evaluation results are consistent with the objective evaluation results. These results demonstrate the superior performance of StableVC in accurately transferring both timbre and style, achieving higher speaker similarity, lower WER, and more accurate style representation compared to the baseline systems. These metrics highlight StableVC’s capability to independently and accurately convert timbre and style from different unseen speakers, leading to high-quality, intelligible, and expressive conversion results.

Table 3: Effect of Euler sample steps $\mathbf { N }$ on flow matching module. Objective metrics improve rapidly in the first 5 steps and continued qualitative improvements up to 20 steps.   

<html><body><table><tr><td>Model</td><td>N</td><td>UTMOS ↑</td><td>WER↓</td><td>SECS↑</td><td>RTF↓</td></tr><tr><td rowspan="5">StableVC</td><td>1</td><td>3.42</td><td>3.87</td><td>0.54</td><td>0.031</td></tr><tr><td>2</td><td>3.49</td><td>3.94</td><td>0.55</td><td>0.043</td></tr><tr><td>5</td><td>3.82</td><td>2.98</td><td>0.61</td><td>0.068</td></tr><tr><td>10</td><td>4.12</td><td>2.03</td><td>0.67</td><td>0.146</td></tr><tr><td>20</td><td>4.15</td><td>2.07</td><td>0.66</td><td>0.215</td></tr></table></body></html>

# Ablation Study

We conduct ablation studies to verify the following: 1) the effect of using different Euler sampling steps on the quality and similarity of the converted samples; 2) the effectiveness of each method in timbre and style modeling, including the introduction of the timbre prior information and multiple reference speech in timbre attention, the adaptive gate in style attention, the GRL loss, and the deduplication for repredicting duration. These studies aim to evaluate the contributions of each component to the overall performance.

We compare the performance by using different Euler steps to evaluate the trade-offs between efficiency and quality, as shown in Table 3. By analyzing these results, we observe that StableVC achieves low WER even with very few Euler steps, indicating that the sample generated by the flow matching module can be accurately recognized by the ASR model, demonstrating high robustness and intelligibility. Evaluation results stabilize after approximately 5 Euler steps, with better audio quality achieved around 10 steps and only slight improvement observed at 20 steps. Although the

RTF increases with the number of sampling steps, the RTF at 10 Euler steps remains competitive.   

<html><body><table><tr><td>Model</td><td>Corr↑</td><td>RMSE↓</td><td>UTMOS↑</td><td>WER↓</td><td>SECS↑</td></tr><tr><td>StableVC</td><td>0.75</td><td>12.87</td><td>4.06</td><td>2.12</td><td>0.64</td></tr><tr><td>w/o vp&multi</td><td>0.59</td><td>18.78</td><td>3.51</td><td>22.16</td><td>0.41</td></tr><tr><td>w/oa</td><td>0.73</td><td>12.85</td><td>4.01</td><td>2.21</td><td>0.36</td></tr><tr><td>W/o LGRL</td><td>0.68</td><td>14.61</td><td>3.62</td><td>2.63</td><td>0.51</td></tr><tr><td>w/o dur</td><td>0.66</td><td>16.51</td><td>4.05</td><td>2.27</td><td>0.63</td></tr></table></body></html>

Table 4: The ablation study results of removing each component in timbre and style modeling.

To verify the effectiveness of each method in timbre and style modeling, we conduct the following ablation studies: 1) only use the source utterance as reference at the training stage and do not concatenate with timbre prior vp, denoted as w/o vp&multi; 2) remove the adaptive gate in style attention, denoted as w/o $\alpha$ ; 3) remove the GRL loss, denoted as w/o $\mathcal { L } _ { \bf G R L } ; 4 )$ ) remove the deduplication in linguistic content extraction and duration modeling, denoted as w/o dur.

As shown in Table 4, we observe significant performance degradation across all metrics when the multiple reference and timbre prior are removed, particularly in the WER results. The primary reason is that the timbre attention mechanism captures irrelevant linguistic information from the reference speech, leading to poor intelligibility in the final converted samples. A noticeable decline in SECS is observed when the adaptive gate is removed. This indicates that without the adaptive gate, the style attention mechanism captures extraneous information from the style prompt, weakening the relationship between the attention score and style information, resulting in lower speaker similarity. These findings suggest that the adaptive gate plays a critical role in controlling style injection without compromising timbre similarity, even when style representations from unseen speakers are injected. When the GRL loss is removed, both style and timbre metrics degrade. This demonstrates that the GRL loss aids in disentangling timbre and style information, thereby enhancing the final timbre and style similarity of the generated samples. Additionally, a decline in style-related metrics is observed when deduplication and duration modeling are removed. By comparing the performance of these ablated models with the full StableVC model, we can evaluate the contribution of each component to the system’s overall ability to control timbre and style information.

# Conclusion

In this study, we propose StableVC to flexibly and efficiently convert style and timbre of source speech to different unseen target speakers. StableVC consists of 1) three feature extractors to disentangle source speech into linguistic content, style, and timbre, and 2) a conditional flow matching module to reconstruct the target mel-spectrogram in a nonautoregressive manner. We propose a novel dual attention mechanism with an adaptive gate in the DiT block to effectively capture distinctive timbre and style information. Experimental results demonstrate that StableVC outperforms all zero-shot VC baseline systems in both subjective and objective metrics. We also show that StableVC enables flexible conversion of style and timbre in a zero-shot manner.