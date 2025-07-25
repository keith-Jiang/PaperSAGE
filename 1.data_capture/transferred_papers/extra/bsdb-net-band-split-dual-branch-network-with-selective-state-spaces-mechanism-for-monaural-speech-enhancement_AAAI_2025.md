# BSDB-Net: Band-Split Dual-Branch Network with Selective State Spaces Mechanism for Monaural Speech Enhancement

Cunhang Fan1, Enrui Liu1, Andong $\mathbf { L i } ^ { 2 , 3 * }$ , Jianhua $\mathbf { T a 0 } ^ { 4 }$ , Jian Zhou1, Jiahao $\mathbf { L i } ^ { 1 }$ , Chengshi Zheng2,3, Zhao Lv1\*

1Anhui Province Key Laboratory of Multimodal Cognitive Computation, School of Computer Science and Technology, Anhui University, Hefei, China 2Key Laboratory of Noise and Vibration Research, Institute of Acoustics Chinese Academy of Sciences, Beijing, China 3University of Chinese Academy of Sciences, Beijing, China 4Department of Automation, Tsinghua University, Beijing, China cunhang.fan@ahu.edu.cn, $\mathord { \mathfrak { c } } 2 3 2 0 1 0 9 0 \emptyset$ stu.ahu.edu.cn, liandong $@$ mail.ioa.ac.cn, jhtao $@$ nlpr.ia.ac.cn, jzhou@ahu.edu.cn, e23301209@stu.ahu.edu.cn, cszheng $@$ mail.ioa.ac.cn, kjlz@ahu.edu.cn

# Abstract

Although the complex spectrum-based speech enhancement (SE) methods have achieved significant performance, coupling amplitude and phase can lead to a compensation effect, where amplitude information is sacrificed to compensate for the phase that is harmful to SE. In addition, to further improve the performance of SE, many modules are stacked onto SE, resulting in increased model complexity that limits the application of SE. To address these problems, we proposed a dual-path network based on compressed frequency using Mamba. First, we extract amplitude and phase information through parallel dual branches. This approach leverages structured complex spectra to implicitly capture phase information and solves the compensation effect by decoupling amplitude and phase, and the network incorporates an interaction module to suppress unnecessary parts and recover missing components from the other branch. Second, to reduce network complexity, the network introduces a band-split strategy to compress the frequency dimension. To further reduce complexity while maintaining good performance, we designed a Mamba-based module that models the time and frequency dimensions under linear complexity. Finally, compared to baselines, our model achieves an average 8.3 times reduction in computational complexity while maintaining superior performance. Furthermore, it achieves a 25 times reduction in complexity compared to transformer-based models.

# Introduction

In the realm of audio signal processing, speech enhancement (SE) is regarded as a fundamental technique to recover the clean speech from noisy environments. The degradation of speech quality by background noise is not only perceptually bothersome but also significantly impairs the performance of automatic speech recognition (ASR) (Chan 2016; Liu et al. 2018). Also, SE is indispensable in smart devices, vehicular systems, and home automation. With the burgeoning prevalence of online conferencing, the demand for realtime SE solutions has surged, underscoring the necessity for techniques that are both effective and computationally efficient (Pandey and Wang 2019).

Existing speech enhancement methods can be roughly categorized into two classes, namely in the time domain (Luo and Mesgarani 2019; Yin et al. 2020) and in the time-frequency (T-F) domain. This paper mainly focuses on the latter. Initially, SE methods focused on magnitude spectrum enhancement with the phase kept unaltered (Tan and Wang 2019). Subsequent researches have revealed the pivotal role of phase (Yin et al. 2020), which inspire many studies to model complex spectra to better recover phase information. Masking-based methods, such as the complex ratio mask (Williamson, Wang, and Wang 2015), have been recognized for their ability to modulate both the real (R) and imaginary (I) components of the noisy complex spectra, surpassing the performance of traditional masks like ideal binary mask (Divenyi 2004) and ideal ratio mask (Hummersone, Stokes, and Brookes 2014). Besides, complex spectral mapping (CSM) has also been introduced to reconstruct the target RI components directly (Tan and Wang 2019). However, this direct mapping can lead to a compensation effect between phase and magnitude (Wang, Wichern, and Le Roux 2021). Recent advancements have seen the application of decoupling concepts in audio processing (Alex et al. 2024). Some multi-stage methods have been proposed to address SE problems. These methods decouple the original mapping problem into two stages: predicting the magnitude spectrum at first and refining the complex spectrum through residual learning in the second stage. This approach partially mitigates the implicit compensation effect between phase and magnitude (Li et al. 2021a). Nonetheless, the sequential nature of these methods can be limiting, as the performance of later stages is heavily dependent on the output of earlier stages (Li et al. 2022b).

Despite the promising results of previous works, their high computational complexity can impede the practical application of SE models. First, as a front-end task, the impact of SE on downstream tasks, such as ASR, should be considered. Moreover, effective deployment in edge or resource-constrained environments, like online meetings and real-time communications, necessitates extremely lowcomplexity SE methods. Strategies such as frequency band division (Yu and Luo 2023), have been introduced to reduce frequency modeling overhead. Additionally, the computational complexity of sequence modeling can be substantial, with some transformer-based models reaching hundreds of gigaflops $\mathrm { ( G / s ) }$ due to their quadratic attention mechanisms (Lu, Ai, and Ling 2023). Recently, State Space Models (Hamilton 1994), such as HIPPO (Gu et al. 2020) and the Structured State Space Model (Gu, Goel, and Re´ 2021), have demonstrated promising performance with reduced complexity. The Selective SSM (Gu and Dao 2023), in particular, enables the establishment of long-range dependencies with linear computational complexity, making it suitable for sequence modeling.

In light of these challenges, we propose the Band-Split Dual-branch Network (BSDB-Net) for the monaural SE task. This approach introduces a decomposition strategy to enhance the magnitude and complex spectra in parallel. Specifically, we devised the Magnitude Enhancement Network (MEN) to suppress the noise components without interfering with phase information, and the Complex Spectral Enhancement Network (CEN) to restore phase information implicitly, complementing the magnitude features extracted by MEN and mitigating the compensation effect. Besides, to effective decrease the overhead for frequency modeling, a frequency band-split strategy is adopted to compress the frequency axis. To better decrease the complexity for sequence modeling, the Mamba structure is introduced to reduce complexity in temporal and frequency sequential relations.

The major contributions of this paper are summarized as follows:

• We propose a dual-branch parallel speech enhancement network that implicitly extracts phase information while ensuring the independence of amplitude information. • We adopt a frequency band-splitting strategy and Mamba-based sequence modeling modules, significantly reducing computational complexity. • Comprehensive experiments on two public datasets demonstrate that BSDB-Net achieves an average 8.3 times reduction in computational complexity while maintaining comparable performance.

# Related Work

Multi-stage Methods: Due to the lack of prior information, the performance of the single-stage SE pipeline is often heavily limited in complicated acoustic scenarios. In contrast, for the multi-stage pipeline, the original mapping problem is usually decomposed into several separate subtasks to enable the learning progressively (Li et al. 2022a). In DTLN (Westhausen and Meyer 2020), the authors proposed a stacked dual signal transformation network. In FullSubNet (Hao et al. 2021), the idea of combing sub-band and full-band was proposed to restore the spectrum. CompNet (Fan et al. 2023) combined the time domain and T-F domain, using a cross-domain complementary approach to optimize the speech enhancement network. CTS-Net (Li et al. 2021a) utilized a two-stage paradigm to supplement phase information on the basis of extracting the magnitude spectrum. TaylorNet (Li et al. 2022a) proposed an end-to-end framework to simulate the 0th-order and high-order items of Taylor-unfolding. FAF-Net (Yue et al. 2022) proposed reference-based speech enhancement via a feature alignment and fusion network.

Sequence Modeling: The exploration of long and short-term temporal dependencies within speech signals prompted the adoption of Recurrent Neural Networks (RNNs (Zaremba, Sutskever, and Vinyals 2014)) to better capture contextual relations. To address the issue of gradient explosion, Long Short-Term Memory (LSTM (Chen et al. 2016)) networks were introduced. The temporal convolutional module (TCM) (Bai, Kolter, and Koltun 2018) introduced in TCM was found to be more effective in time series modeling than LSTM. Recently, a compressed version of TCM called the squeezed temporal convolutional module (S-TCM) (Li et al. 2021a) was proposed. Transformerbased models (Xu, Tu, and Yang 2023) have emerged as a promising alternative due to their superior capability in modeling long-range dependencies using self-attention mechanisms (Yu et al. 2022). Due to its linear complexity, Mamba (Li and Chen 2024) is considered a promising alternative to transformer in sequence modeling.

# Proposed Architecture

The received noisy speech in the short-time Fourier transform (STFT) domain can be presented as,

$$
Y ( t , f ) = X ( t , f ) + N ( t , f )
$$

where $\{ Y , X , N \}$ denote the mixture, clean and noise, respectively. $t \in \mathsf { \Gamma } \{ 1 , \cdots , T \}$ is the time frame index, and $f \in \{ 1 , \cdots , F \}$ is the frequency index.

The proposed BSDS framework is shown in Figure 1. The noisy magnitude and complex spectra are first decoupled. By separately band-splitting and feature encoding, they are converted into abstract representations for magnitude and phase, respectively. Then, stacked T-F Mamba-blocks are adopted for effectively modeling along the time and frequency axes. Subsequently, the representations from two streams are fused. The segmented frequency bands are then merged using a Mask-Decoder module to obtain the estimated complex spectrum.

# Band-Split and Mask-Decoder

As shown in Figure 2, the complex and magnitude spectra are first compressed into lower-resolution bands using the Band-Split module ( $\mathrm { Y u }$ and Luo 2023). To be specific, the noisy input spectrum $X$ is segmented into a sequence of non-overlapping frequency bands $\left\{ A _ { i } \right\} _ { i = 1 } ^ { K }$ , each of which is individually projected to yield the embedding of dimension $N$ . Subsequently, the $K$ bands are stacked to obtain a 3-D tensor $Z$ . The process can be formulated into

$$
A _ { i } \in \left\{ A _ { 1 } , A _ { 2 } , . . . , A _ { K } \right\} , A _ { i } \in \mathbb { R } ^ { T \times F _ { i } }
$$

$$
\boldsymbol { X _ { i } } = F C ( L N ( A _ { i } ) ) , \boldsymbol { X _ { i } } \in \mathbb { R } ^ { T \times N }
$$

$$
Z = C o n c a t \left( X _ { 1 } , X _ { 2 } , . . . , X _ { K } \right) \in \mathbb { R } ^ { K \times T \times N }
$$

![](images/4556a1f7035d1cc396fab5011478fe16fba9ed9d737d696c1c53a86cedb9eddf.jpg)  
Figure 1: Overall architecture of the proposed BSDB-Net consists of three main components. The first part includes the BandSplit module for frequency band segmentation and the Mask-Decoder module for generating masks used in band synthesis. The second part features a dual-branch enhancement network: the MEN branch suppresses noise in the magnitude spectrum roughly, while the CEN branch primarily estimates complex spectra to capture phase characteristics. The third part involves the Mamba-block module designed for sequence modeling.

![](images/fd10c3cd85e28fa9f10eae94c43644cc481483a5f08b9a16142d5a9b9bccce1d.jpg)  
Figure 2: (a) The Band-Split module divides frequency bands for input into the modeling module. (b) The Mask-Decoder module synthesizes frequency bands postmodeling to generate masks.

where $A _ { i }$ represents the $\mathbf { \chi } _ { i }$ frequency bands to be split, $\{ F C , L N \}$ denote the linear layer and layer normalization, respectively. Concat $( \cdot )$ denotes the concatenation operation.

For the Band-Merge module, let us denote the input as $B \in \mathbb { R } ^ { K \times T \times N }$ . Similarly, for the frequency band feature $B _ { i } \in \mathbb { R } ^ { T \times N }$ , where $i \in \mathsf { \Gamma } \{ 1 , \cdots , K \}$ denotes the band index, it is processed through separate layers to obtain the decoded target feature. After all the bands are processed, they are concatenated along the frequency axis to form the output $M$ , whose formulation can be given by

$$
M _ { i } = G L U ( T a n h ( F C ( L N ( B _ { i } ) ) ) )
$$

$$
M = M e r g e ( M _ { 1 } , M _ { 2 } , . . . , M _ { K } ) , M \in \mathbb { R } ^ { F \times T }
$$

where $B _ { i }$ denotes the $i$ th band feature, $\{ T a n h , G L U \}$ denote the Tanh activation function and gated linear unit, re

spectively. Merge (·) is the concatenation operation along the frequency axis.

# Dual-Branch

After the band split module, the magnitude-oriented and complex-oriented features are fed into the MEN and CEN branches for modeling, respectively.

For both branches, the data first pass through the encoder layer. Instead of using typical encoder layers with downsampling operations, here we coarsely model the features without frequency downsampling to mitigate the possible information loss:

$$
\begin{array} { r } { C _ { m } = P R e L U ( L N ( C o n v 2 d ( I n t e r ( Z _ { m a g } , Z _ { r i } ) ) ) ) } \\ { C _ { r i } = P R e L U ( L N ( C o n v 2 d ( I n t e r ( Z _ { r i } , Z _ { m a g } ) ) ) ) } \end{array}
$$

where $Z _ { m a g }$ , $Z _ { r i }$ represent the magnitude and complex features after Band-Split module, $C _ { m }$ , $C _ { r i }$ denote the output of magnitude and complex encoders, respectively. Inter $( a , b )$ denotes the interaction module between the features $a$ and $b$ .

As shown in Figure 1, the feature processed by the encoder is passed to the sequence modeling using the MambaBlock. After that, a mask is generated via the F1E-Mask layer. The F1E Mask refers to the extraction of features that include amplitude and the generation of a mask. It includes a dilated block layer to expand the model’s receptive field, followed by the PReLU activation function and layer normalization. Finally, the input data to the MEN and the estimated mask are multiplied to obtain the filtered feature of the MEN branch. The whole process can be given by

$$
D = P R e L U ( L N ( D C o n v ( M a m ( C _ { m a g } ) ) ) )
$$

$$
D _ { 1 } = S i g m o i d ( C o n v ( D ) )
$$

$$
D _ { 2 } = T a n h ( C o n v ( D ) )
$$

$$
D _ { m e g } = D _ { 1 } \otimes D _ { 2 } , M E N _ { o u t } = A _ { m a g } \otimes D _ { m e g }
$$

![](images/0a3e81a449315a1f4345cf1518e573b1c4e7bcaa61fd1d022104e6911ffe0dc6.jpg)  
Figure 3: The Mamba-Block: It is primarily divided into temporal modeling and frequency modeling. (a) The proposed unidirectional Mamba module. (b) The proposed bidirectional Mamba module.

where $\otimes$ denotes the elementary multiplication operation. $\left\{ M a m \left( \cdot \right) , D C o n v \left( \cdot \right) \right\}$ represent the Mamba-Block and the Dilated-Block, respectively. $M E N _ { o u t }$ represents the output of MEN branch. The CEN branch processes data in a similar way to the MEN branch, with the only difference lying in the final module. The F2E-Real and F2E-Imag modules refer to the implicit extraction of phase information by obtaining complex spectral features:

$$
E = M a m ( C _ { r i } )
$$

$$
E _ { r } = P R e L U ( L N ( D C o n v ( E ) ) )
$$

$$
E _ { i } = P R e L U ( L N ( D C o n v ( E ) ) )
$$

$$
C E N _ { o u t } = E _ { r } \oplus E _ { i }
$$

where $\oplus$ denotes the elementary sum operation. $C E N _ { o u t }$ represents the output of CEN branch, and $E _ { r } , E _ { i }$ represent the real and imaginary parts outputted by different decoder modules, respectively.

# Mamba-Block

As shown in Figure 3, the Mamba-Block module is used for sequence modeling in BSDB-Net. The corresponding equation is as follows:

$$
h _ { n } = \overline { { \mathbf { A } } } h _ { n - 1 } + \overline { { \mathbf { B } } } x _ { n }
$$

$$
y _ { n } = \mathbf { C } h _ { n }
$$

where $\overline { { \mathbf { A } } }$ and $\overline { { \mathbf { B } } }$ represents the parameters of the discretization matrix. The process of discretization converts continuous parameters $( \Delta , \mathbf { A } , \mathbf { B } )$ into discrete ones $( \overline { { \mathbf { A } } } , \overline { { \mathbf { B } } } )$ , enabling the model to handle discrete data effectively. Mamba incorporates Selective SSMS into an H3 structure.

First, the Mamba-Block receives data from both the MEN and CEN branches enters the interaction layer for fusion. After passing through the LN layer, the dimension is transposed from $\bar { Z } \in \mathbb { R } ^ { \bar { B } \times T \times N \times F }$ to $Z \in \mathbb { R } ^ { ( B \times T ) \times N \times F }$ to better model the frequency dimension. Subsequently, the data is sent into the F Mamba module for sequence modeling, as illustrated in Figure 3(b). This module achieves bidirectional modeling, the forward and backward data are sent separately into the Mamba module:

$$
X _ { i n } = I n t e r ( X _ { r i } , X _ { m a g } ) , F _ { i n } = T r a n ( L N ( X _ { i n } ) )
$$

$$
F _ { o u t } = X _ { i n } \oplus D e c o n v ( F \_ M a ( U n f o l d ( F _ { i n } ) ) )
$$

where $X _ { r i }$ , $X _ { m a g }$ represent the input of Mamba-Block in the Dual-branch. Inter represents interaction layer. F Ma as shown in Figure 3(b) represents sequence modeling.

The modeling for the time dimension is similar to that for the frequency dimension. However, the key difference is that unidirectional Mamba is employed for time modeling since this model operates under a causal framework. Firstly, the data will be Transpose $Z \in \mathbb { R } ^ { ( B \times K ) \times N \times T }$ . Subsequently, it will go through the T-Mamba module as shown in Figure 3(a) for sequence modeling. Once the modeling is completed, the data will pass through a deconvolution layer and be Transposed into Z RB×T ×N×F :

$$
T _ { i n } = U n f o l d ( P a d ( T r a n ( F _ { o u t } ) ) )
$$

$$
T _ { o u t } = T r a n ( D e c o n v ( T - M a ( T _ { i n } ) ) )
$$

$$
M a m b a _ { o u t } = F _ { o u t } \oplus T _ { o u t }
$$

where T Ma as shown in Figure 3(a) represents sequence modeling, $M a m b a _ { o u t }$ represents the output of MambaBlock. Trans represents the tensor dimension reshaping.

# Interaction Module

We achieve information interaction through the interaction module. This module first concatenates $I n p u t _ { 1 }$ and $I n p u t _ { 2 }$ using a Cat layer, then passes them through a Conv2d layer and a LayerNorm layer, followed by generating a mask using a Sigmoid activation function. After that, it multiplies the result with $I n p u t _ { 1 }$ to output the final data:

$$
I n p u t = C a t ( I n p u t _ { 1 } , I n p u t _ { 2 } )
$$

$$
M a s k = S i g m o i d ( L N ( C o n v 2 d ( I n p u t ) ) )
$$

$$
O u t p u t = I n p u t _ { 1 } + ( I n p u t _ { 2 } \otimes M a s k )
$$

# Experimental Setup

# Datasets

We selected the WSJ0-SI84 and the DNS-Challenge noise dataset to create synthetic data to evaluate our model and conduct ablation experiments. Subsequently, we compared our model with others using a widely-used dataset, VoiceBank $^ +$ Demand.

WSJ0-SI84 $\cdot +$ DNS-Challenge: WSJ0-SI84 (Paul and Baker 1992) consists of 7138 clean speech samples from 83 speakers. From these, 5428 and 957 utterances from 77 speakers are randomly selected for the training and validation sets. To construct ”noisy-clean” training pairs, approximately 20,000 types of noise from the DNSChallenge (Reddy et al. 2020) dataset’s noise library are randomly selected and concatenated, resulting in a total duration of approximately 55 hours. The training set and validation set of the dataset are synthesized by the author

Table 1: The objective is to compare the effects of different models on PESQ, ESTOI, and SI-SDR metrics for Set-A and Set-B in an unseen speaker test set.   

<html><body><table><tr><td colspan="2">SMRtrics)</td><td rowspan="2">Feat.</td><td rowspan="2"></td><td colspan="4">PESQ5</td><td colspan="4">EST0I%)</td><td colspan="4">SI-SDR(dB)</td></tr><tr><td colspan="2"></td><td>-5</td><td></td><td></td><td>Avg.</td><td></td><td></td><td></td><td>Avg.</td><td>-5</td><td></td><td></td><td>Avg.</td></tr><tr><td>A-10S</td><td>Noisy ConvTasNet DPRNN DDAEC LSTM GCRN</td><td>1 Wave Wave Wave Mg RI RI RI</td><td>1.54 2.11 2.17 2.27 1.97 2.02 1.90 2.20</td><td>1.86 2.54 2.60 2.79 27 2.55 2.46</td><td>2.17 2.88 2.96 3.16 2.7 2.92</td><td>1.85 2.52 2.57 2.74 24 2.50 2.40</td><td>-5 29.25 60.06 61.74 63.12 49.33 56.44</td><td>43.11 73.80 74.74 76.65 64.24 72.83 50.98 68.06</td><td>78.73</td><td>57.53 82.90 83.53 84.73 74. 82.08</td><td>43.30 72.25 73.34 74.83 6.82 70.45 65.92</td><td>-5.00 6.56 6.88 7.22 2 5.36 4.17</td><td>0.00 10.43 10.60 11.23 6.5 9.72</td><td>5.00 13.63 13.82 14.15 10.9 12.67</td><td>0.00 10.21 10.43 10.87 6 9.25</td></tr><tr><td></td><td>BSDB(ours) Noisy ConvTasNet DPRNN DDAEC LSRN. GCRN</td><td>M+RI M+RI 1 Wave Wave Wave Mag RI</td><td>2.32 2.36 2.43 1.74 2.57 2.66 2.83 23 2.55 2.47</td><td>2.79 2.85 2.92 2.04 2.90 2.98 3.17 28 2.94</td><td>3.22 2.26 2.41 3.21 3.27 3.43 3.21</td><td>2.81 2.87 2.06 2.89 2.97 3.15 262 2.90</td><td>65.84 66.27 44.59 73.00 74.23 75.57 64.27</td><td>78.13 78.19 57.38 81.79 82.44 83.65 74.51</td><td>85.79 86.66 69.45 87.90 88.32 89.03 81.76</td><td>76.59 77.04 57.14 80.90 81.66 82.75 73.516</td><td></td><td>7.36 7.42 -5.00 10.25 10.49 10.91 6.04</td><td>11.23 11.34 0.00 13.18 13.37 13.67 90.3</td><td>14.31 14.39 5.00 16.07 16.17 16.16 13.8</td><td>10.97 11.05 0.00 13.17 13.34 13.58 8.0.9</td></tr></table></body></html>

Table 2: The aim is to compare different models and the impact of varying input dimensions and module stacking times in Mamba on PESQ and Computational Complexity. (Here, $^ { \cdot \cdot } 6 4 ^ { \cdot } 4 ^ { \cdot \cdot }$ represents an input dimension of 64 with modules stacked four times, and so forth.)   

<html><body><table><tr><td>Modle</td><td>Cau.</td><td>PESQ</td><td>MACs</td></tr><tr><td>LSTM</td><td>√</td><td>2.37</td><td>3.69 G/s</td></tr><tr><td>CRN</td><td>√</td><td>2.45</td><td>2.54 G/s</td></tr><tr><td>GCRN</td><td>√</td><td>2.55</td><td>2.40 G/s</td></tr><tr><td>FullSubNet</td><td>√</td><td>2.64</td><td>29.83 G/s</td></tr><tr><td>CTSNet</td><td>√</td><td>2.79</td><td>5.48 G/s</td></tr><tr><td>ConvTasNet</td><td>√</td><td>2.54</td><td>5.22 G/s</td></tr><tr><td>DPRNN</td><td>√</td><td>2.60</td><td>8.47 G/s 36.85 G/s</td></tr><tr><td>DDAEC</td><td>√</td><td>2.79</td><td></td></tr><tr><td>DBTNet</td><td>X</td><td>3.18</td><td>42.64 G/s</td></tr><tr><td>GaGNet</td><td>√</td><td>2.85</td><td>2.81 G/s</td></tr><tr><td>64-4(our)</td><td>√</td><td>2.62</td><td>0.88 G/s</td></tr><tr><td>64-6(our)</td><td>√</td><td>2.70</td><td>0.98 G/s</td></tr><tr><td>128-4(our)</td><td>√</td><td>2.78</td><td>1.34 G/s</td></tr><tr><td>128-6(our)</td><td>√</td><td>2.92</td><td>1.68 G/s</td></tr><tr><td></td><td></td><td></td><td>1.87 G/s</td></tr><tr><td>256-2(our)</td><td>√</td><td>2.74</td><td></td></tr><tr><td>256-4(our)</td><td>√</td><td>2.83</td><td>3.06 G/s</td></tr><tr><td>256-6(our)</td><td>√</td><td>2.95</td><td>4.26 G/S</td></tr></table></body></html>

Li (Li et al. 2021a), and the Set-A and Set-B are consistent with the literature (Li et al. 2022b).

VoiceBank+Demand: VoiceBank (Veaux, Yamagishi, and King 2013) consists of 30 speakers, with 28 speakers used for the training set and the remaining two speakers used for testing. The training (Botinhao et al. 2016) set includes 11,572 “noisy-clean” pairs, mixed with 10 types of noise (8 from the Demand (Thiemann, Ito, and Vincent 2013) noise database and two from artificial noise), this is a public dataset in SE, and this paper uses the same dataset as mentioned above.

# Implementation Setup

All speech signals in the training set are sampled at $1 6 \mathrm { k H z }$ . The speech signals are framed using a $2 0 ~ \mathrm { { m s } }$ Hann window with $50 \%$ overlap between frames. Transforming these framed signals into the time-frequency domain involves using a 320-point FFT. Following findings from the literature (Li et al. 2021b), we use a power spectral density compression strategy, where the compression coefficient is set to 0.5, denoted as $\lvert X \rvert _ { 0 . 5 } , \lvert S \rvert _ { 0 . 5 }$ . The Adam optimizer is utilized with parameters $\beta = 0 . 9$ and $\beta = 0 . 9 9 9$ . The learning rate is initialized to 5e-4, and if the validation loss does not decrease for two consecutive evaluations, the learning rate is halved.

# Baseline Models

On the WSJ0-SI84 dataset, a total of 9 baseline methods were selected for comparison with the proposed model. ConvTasNet, DPRNN (Luo, Chen, and Yoshioka 2020), and

DDAEC (Pandey and Wang 2020) are all time-domain SE models. LSTM (Chen et al. 2016), CRN (Tan and Wang 2018), GCRN (Tan and Wang 2019), DCCRN (Hu et al. 2020), FullSubNet (Hao et al. 2021), CTSNet (Li et al. 2021a), and GaGNet (Li et al. 2022b) are models in the timefrequency domain. Among them, GCRN and DCCRN were developed based on CRN. FullSubNet introduces full-band modeling and sub-band modeling. CTSNet and GaGNet are respectively parallel and serial two-stage amplitude-phase decoupling models.

On the VoiceBank $^ { + }$ Demand dataset, a total of 13 baseline methods were selected for comparison with the proposed model. SEGAN (Pascual, Bonafonte, and Serra 2017), MMSEGAN (Soni, Shah, and Patil 2018), MetricGAN (Fu et al. 2019), and SRTNET (Qiu et al. 2023) are generative models of speech enhancement (SE). Wavenet (Oord et al. 2016) operates as a time-domain model. PHASEN (Yin et al. 2020), MHSASPK (Koizumi et al. 2020), DCRNN, TSTNN (Wang, He, and Zhu 2021), S4NDUNet (P-J et al. 2023), FDFnet (Zhang, Zou, and Zhu 2024), CSTnet, and GaGnet are all models in the time-frequency domain, with the latter five being multi-stage SE models. CompNet (Fan et al. 2023) is a model that spans both the time-domain and time-frequency domain.

# Loss Function

Our BSDB-Net employs a dual-branch approach for speech enhancement, utilizing the “RI+Mag” loss function to supervise the optimization of both phase and magnitude components simultaneously:

$$
\mathcal { L } _ { R I } = \left. \tilde { S } _ { r } - S _ { r } \right. _ { F } ^ { 2 } + \left. \tilde { S } _ { i } - S _ { i } \right. _ { F } ^ { 2 }
$$

$$
\mathcal { L } _ { M a g } = \left. \sqrt { | \tilde { S } _ { r } | ^ { 2 } + | \tilde { S } _ { i } | ^ { 2 } } + \sqrt { | S _ { r } | ^ { 2 } + | S _ { i } | ^ { 2 } } \right. _ { F } ^ { 2 }
$$

$$
\mathcal { L } = \beta \mathcal { L } _ { R I } + ( 1 - \beta ) \mathcal { L } _ { M a g }
$$

where $\left\| . \right\| _ { F }$ represents Frobenius norm, and $\beta$ is empirically set to 0.5.

# Evaluation Metrics

Multiple objective metrics are adopted, including narrowband (NB) and wide-band (WB) perceptual evaluation speech quality (PESQ) (Rix et al. 2001) for speech quality, short-time objective intelligibility (STOI) and its extended version ESTOI (Jensen and Taal 2016) for intelligibility, SISDR (Hu and Loizou 2007) for speech distortion, and MOS (CSIG, CBAK, COVL) (Hu and Loizou 2007) for speech quality.

# Results and Analysis

# Ablation Study

We conducted ablation experiments on the WSJ0 dataset, which cover the following three aspects: (1) Whether the dual-path structure is effective; (2) How many layers of Mamba-based modules should be stacked and how many hidden layers should be fed into Mamba to achieve the best effect; (3) Whether using Mamba as a sequence modeling model is effective.

Table 3: Compare the effects of enhancing magnitude spectrum only(BSDB-MEN), enhancing complex spectrum only(BSDB-CEN), and enhancing both magnitude and complex spectra in parallel(BSDB-DB) on PESQ, ESTOI, and SDR.   

<html><body><table><tr><td>Model</td><td>PESQ</td><td>ESTOI</td><td>SI-SDR(dB)</td></tr><tr><td>BSDB-MEN</td><td>2.67</td><td>71.25</td><td>9.84</td></tr><tr><td>BSDB-CEN</td><td>2.76</td><td>74.32</td><td>10.37</td></tr><tr><td>BSDB-DB</td><td>2.92</td><td>78.19</td><td>11.34</td></tr></table></body></html>

Effect of Dual-Branch Model: Here we are primarily investigating the effectiveness of the dual-branch structure. Initially, we removed the CEN branch to train the MEN single branch, and subsequently removed MEN while retaining the CEN branch for training. As shown in Table 3, the dualpath structure outperforms the single-path structure across all metrics. This means that the coordinated efforts of CEN and MEN can improve the quality of the target speech. CEN filters out primary noise for rough estimation, while MEN continuously supplements speech information, thereby enhancing the overall performance of the system.

The Number of Layers and Hidden Layers of MambaBlock: The dimensions of the input data to Mamba have an impact on the model’s performance. Additionally, stacking Mamba-Block layers can further enhance model performance but also increases complexity. As shown in Table 2, we selected Mamba input dimensions of 64, 128, and 256, and stacked modules 2, 4, and 6 times in different combinations. From our findings, when we stack sequence modeling modules and increase the number of Mamba hidden layers, performance will improve and the increase in depth has a more significant effect than the increase in breadth, using a combination like 128-6 for Mamba already achieves a balance between performance and complexity. Additionally, the current SE models are all at the complexity level of $\mathrm { G b / s }$ , while our model further compresses the complexity to the level of Mb/s without a significant decrease in performance.

Effect of Mamba-Block: Our model aims to ensure performance while compressing model complexity, benefiting from advancements in Selective State Spaces. To demonstrate the effectiveness of Mamba-Block, we replaced it with LSTM and Transformer while keeping other factors constant, as shown in Table 5 Our results indicate that our model maintains optimal performance with significantly reduced complexity, validating the applicability and feasibility of Mamba for our task. Although the performance improvement over the Transformer is relatively minor, the significant reduction in complexity is the reason why our model opts for Mamba for sequence modeling.

# Model Complexity Comparison

As shown in Table 2, we evaluated the complexity of our proposed model and other baseline models on the WSJ0- SI84 dataset. It is worth noting that all input samples were set to one second of audio, ensuring fairness in our experiments. The model selected from the previous ablation experiments has an average computational complexity about 8 times lower than the baseline and has good performance. BSDB-Net exhibits lower complexity along with excellent performance. In terms of performance, our model is only slightly behind DBTNet. The reasons for this can be analyzed as follows: First, our model is a causal model while DBTNet is a non-causal model. Second, DBTNet has been continuously increasing its model complexity to achieve SOTA performance. As can be seen from Table 2, its model complexity is about 20 times that of our model, yet the performance has not been significantly improved. When we further compress the complexity to the level of M/s, proving the effectiveness of our network structure.

Table 4: Comparison was conducted with other state-ofthe-art methods, including both time-domain and timefrequency domain approaches. “-” indicates where results were not provided in the original text.   

<html><body><table><tr><td>Modle</td><td colspan="4">Year|PESQ-WB STOI% CSIG CBAK COVL</td></tr><tr><td>Noisy</td><td>、</td><td>1.97</td><td>92.1 3.35</td><td>2.44</td><td>2.63</td></tr><tr><td>SEGAN</td><td>2017</td><td>2.16</td><td>92.5 3.48</td><td>2.94</td><td>2.80</td></tr><tr><td>MMSEGAN</td><td>2018</td><td>2.53</td><td>93 3.8</td><td>3.12</td><td>3.14</td></tr><tr><td>Wavenet</td><td>2018</td><td></td><td>1 3.62</td><td>3.32</td><td>2.98</td></tr><tr><td>MetricGAN</td><td>2019</td><td>2.86</td><td>3.99</td><td>3.18</td><td>3.42</td></tr><tr><td>DCCRN</td><td>2020</td><td>2.68</td><td>93.7 3.88</td><td>3.18</td><td>3.27</td></tr><tr><td>PHASEN</td><td>2020</td><td>2.99</td><td>4.21</td><td>3.55</td><td>3.62</td></tr><tr><td>MHSA-SPK</td><td>2020</td><td>2.99</td><td>1</td><td>4.15 3.42</td><td>3.53</td></tr><tr><td>TSTNN</td><td>2021</td><td>2.96</td><td>95</td><td>4.17 3.53</td><td>3.49</td></tr><tr><td>CTS-Net</td><td>2022</td><td>2.92</td><td>1 4.25</td><td>3.46</td><td>3.59</td></tr><tr><td>GaGnet</td><td>2022</td><td>2.94</td><td>94.7 4.26</td><td>3.45</td><td>3.59</td></tr><tr><td>SRTNET</td><td>2023</td><td>2.69</td><td></td><td>4.12 3.19</td><td>3.39</td></tr><tr><td>CompNet</td><td>2023</td><td>2.90</td><td></td><td>4.16 3.37</td><td>3.53</td></tr><tr><td>FDFNet</td><td>2024</td><td>3.05</td><td></td><td>4.23 3.55</td><td>3.65</td></tr><tr><td>S4DSE</td><td>2024</td><td>2.55</td><td>1</td><td>3.94 3.00</td><td>3.32</td></tr><tr><td colspan="2">BSDBNet(256)|2024 BSDBNet(128) 2024</td><td>3.11 3.07</td><td>95</td><td>4.33</td><td>3.58</td><td>3.73</td></tr></table></body></html>

# Comparisons with Baselines on WSJ0-SI84 Corpus

As shown in Table 1, the objective metrics results for the proposed method and baseline models on the WSJ0-SI84 dataset include PESQ, ESTOI, and SI-SDR. From the table, we can draw the following conclusions: Firstly, it is evident that models based on complex spectra generally outperform those based solely on magnitude spectra. For instance, models like GCRN and DCCRN consistently outperform CRN and LSTM across all metrics. This indicates that incorporating phase information in addition to amplitude recovery can significantly enhance both speech quality and intelligibility. Secondly, multi-stage models that simultaneously consider magnitude and complex spectra outperform models that focus solely on a single spectrum type or operate in the time domain. Models like CSTNet and GaGNet show superior performance compared to others, suggesting that parallel optimization of amplitude and complex spectra can effectively leverage phase information to generate higher-quality speech. Lastly, our model achieves state-ofthe-art (SOTA) results across all metrics. The reasons for this can be analyzed as follows: BSDB-NET employs a parallel two-stage model, which is better at decoupling amplitude and phase information to address compensation effects compared to other models; A sequence modeling module based on Mamba is constructed, demonstrating good performance at a linear complexity.

Table 5: Compare the impact of replacing the sequence modeling module with LSTM, Transformer, and Mamba on PESQ, Computational Complexity, and Parameters.   

<html><body><table><tr><td>Model</td><td>PESQ MACs</td><td>Parameters</td></tr><tr><td>Ours-LSTM</td><td>2.82 11.05G/s</td><td>14.71M</td></tr><tr><td>Ours-Transformer</td><td>2.90 28.06G/s</td><td>14.06M</td></tr><tr><td>Ours-Mamba(ours)</td><td>2.92 1.68G/s</td><td>9.78M</td></tr></table></body></html>

# Comparisons with Baselines on VoiceBank $^ +$ Demand

In addition to the WSJ0-SI84 corpus, we also conducted experiments on another public benchmark, VoiceBank $^ +$ Demand. BSDB-Net was compared with other baselines, and our model achieved superior results across all metrics. As evident from Table 4, the baseline models achieved average improvements of 0.32, $1 . 5 \%$ , 0.31, 0.17, and 0.35 in PESQ, STOI, CSIG, CBAK, and COVL, respectively. This indicates that the methodologies proposed by our model adeptly address the issues presented in the background. The main purpose of experimenting on this public dataset is to fairly demonstrate that our model can maintain competitive performance while significantly reducing complexity, thereby better proving that the network has indeed effectively solved the proposed problem.

# Conclusion

In this paper, we propose a Band-Split Dual-branch Network based on Selective State Spaces. This network reduces complexity while estimating speech spectra through a complementary mechanism. Specifically, we divide the network into a magnitude enhancement net (MEN) and a complex spectral enhancement net (CEN), which jointly filter noise components while continuously refining and supplementing speech spectrum information. To further reduce model complexity, we introduce a band-splitting strategy in each branch. Additionally, we incorporate Mamba for sequence modeling, ensuring model performance while reducing complexity. In future work, we intend to further enhance model performance and decrease the computational complexity. Besides, we intend to apply the proposed method to more tasks, like multi-channel speech enhancement, dereverberation, and target extraction. Mamba is still far from mature in deployment, which renders it necessary for further optimization.