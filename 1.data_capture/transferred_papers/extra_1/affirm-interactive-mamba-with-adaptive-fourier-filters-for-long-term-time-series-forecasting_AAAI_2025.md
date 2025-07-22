# Affirm: Interactive Mamba with Adaptive Fourier Filters for Long-term Time Series Forecasting

Yuhan $\mathbf { W _ { u } } ^ { * }$ , Xiyu Meng\*, Huajin Hu, Junru Zhang, Yabo Dong†, Dongming Lu

College of Computer Science and Technology, Zhejiang University, Hangzhou, 310027, China {wuyuhan, mengxiyu, huajinhu, junruzhang, dongyb, ldm}@zju.edu.cn

# Abstract

In long-term series forecasting (LTSF), it is imperative for models to adeptly discern and distill from historical time series data to forecast future states. Although Transformerbased models excel at capturing long-term dependencies in LTSF, their practical use is limited by issues like computational inefficiency, noise sensitivity, and overfitting on smaller datasets. Therefore, we introduce a novel time series lightweight interactive Mamba with an adaptive Fourier filter model (Affirm). Specifically, (i) we propose an adaptive Fourier filter block. This neural operator employs Fourier analysis to refine feature representation, reduces noise with learnable adaptive thresholds, and captures inter-frequency interactions using global and local semantic adaptive Fourier filters via element-wise multiplication. (ii) A dual interactive Mamba block is introduced to facilitate efficient intra-modal interactions at different granularities, capturing more detailed local features and broad global contextual information, providing a more comprehensive representation for LTSF. Extensive experiments on multiple benchmarks demonstrate that Affirm consistently outperforms existing SOTA methods, offering a superior balance of accuracy and efficiency, making it ideal for various challenging scenarios with noise levels and data sizes. The codes and data are publicly available at https://github.com/zjuml/Affirm.

# Introduction

Time series data, collected daily and continuously by IoT sensors and wearable devices, is inherently sequential and time-dependent. Time series forecasting (TSF) predicts future values based on historical observations and is widely applied in finance, meteorology, healthcare, useful life, and transportation (Huang, Chen, and Qiao 2024; Wu et al. 2024b; Wang et al. 2024d; Guo et al. 2024; Long et al. 2024).

Nowadays, deep learning models for TSF are primarily in four families: Transformer, MLP, CNN, and RNN. Transformers are the mainstream due to their ability to capture long-term dependencies through self-attention, such as iTransformer (Liu et al. 2024b) and PatchTST (Nie et al. 2023). However, they struggle with small datasets, easily falling into overfitting and computational inefficiencies due to large parameter sizes and quadratic complexity. Besides, some studies also questioned the effectiveness of Transformers, revealing the permutation invariant nature in attention may compromise temporal information (Zeng et al. 2023). Their experiments show that a simple linear layer can surprisingly outperform complex Transformers in TSF. Yet, linear models have their issues: struggling with complex, noisy data, and failing to capture long-term correlations. Similarly, CNNs are limited by small receptive fields, hampering effectiveness for long-term dependency (Zeng et al. 2024). RNNs, like LSTM and GRU, address long-term dependency but suffer from computational inefficiency and lack of parallelization, leading to slower training and inference.

Recently, state-space-based models (SSMs) (Gu et al. 2020) have garnered attention for their potential in sequence modeling. SSMs excel with lengthy sequences by utilizing ordinary differential equations to dynamically evolve states over time. It incorporates hidden-attention mechanisms and context-aware selectivity with linear complexity, making them ideal for time-series analysis (Cai et al. 2024). Mamba (Gu and Dao 2023), building on SSM and S4, has become powerful in formal language learning (Park et al. 2024), visual representation (Zhu et al. 2024), and image haze removal (Zheng and Wu 2024). Mamba advances SSM capabilities with a selective scanning mechanism that tailors parameters to inputs, enhancing feature compression and information extraction. It also employs a unique hardware-aware algorithm for better parallel processing, offering faster inference and scalability than Transformers (Wang et al. 2024b; Zhu et al. 2024). We replaced the Transformer block with Mamba in PatchTST, and compared it to SOTA iTransformer and PatchTST on various datasets, as shown in Figure 1 and Table 1, revealing that Mamba’s performance varies with data frequency. It performs comparably on short-frequency datasets like weather (10 minutes) and ETTm1 (15 minutes) but underperforms on longer-frequency datasets like ETTh2 (hourly) and Exchange (daily). This suggests Mamba struggles with time variations at lower frequencies, prompting us to consider how to enhance Mamba’s performance in TSF. Clearly, improvements can be made by better learning short-term and long-term dependencies in time series data.

To this end, we introduce Affirm (Adaptive fourier filter interactive mamba), a lightweight time series model that scales like Transformers but replaces the self-attention mechanism with the proposed Adaptive Fourier Filter Block (AFFB). Inspired by the convolution theorem (Rabiner and Gold 1975; Huang et al. 2023), AFFB leverages the mathematical equivalence between time domain convolution and the element-wise product in the frequency domain. In this neural operator, Affirm replaces the self-attention layer with four key steps: (i) Fourier transform: convert signals from time domain to frequency domain. (ii) adaptive thresholding: attenuate high or low frequencies using learnable threshold, a strategy to reduce noise and improve signal clarity. (iii) adaptive filtering: apply learnable global and local filters for adaptive filtering across all frequencies via the element-wise product, capturing both long-term and shortterm interactions similar to circular convolution. (iv) inverse Fourier transform: inverse frequency back to the time domain. We replace the traditional feedforward network with the proposed Interactive Dual Mamba Block (IDMB), where Mamba with dual causal convolutional kernel sizes promotes interactive learning to extract temporal patterns. We also incorporate self-supervised pre-training to boost performance, especially on large datasets.

<html><body><table><tr><td rowspan="2">Models</td><td colspan="2">Mamba</td><td colspan="2">PatchTST</td><td colspan="2">iTransformer</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>Weather</td><td>0.346</td><td>0.342</td><td>0.329</td><td>0.338</td><td>0.358</td><td>0.349</td></tr><tr><td>ETTm1</td><td>0.462</td><td>0.446</td><td>0.422</td><td>0.423</td><td>0.491</td><td>0.459</td></tr><tr><td>ETTh2</td><td>0.432</td><td>0.447</td><td>0.395</td><td>0.427</td><td>0.427</td><td>0.445</td></tr><tr><td>Exchange</td><td>0.913</td><td>0.722</td><td>0.901</td><td>0.714</td><td>0.847</td><td>0.691</td></tr></table></body></html>

Table 1: Comparison between vanilla Mamba and Transformer-based architectures for forecasting task.

Affirm is lightweight, reducing complexity from $O ( N ^ { 2 } )$ to $O ( N l o g N )$ using the computationally efficient Fast Fourier Transform (FFT), making it more efficient than selfattention. In summary, our contributions are as follows:

• We introduce Affirm, a generalized lightweight time series forecasting model that uses an interactive Mamba mechanism and three adaptive Fourier filters to capture both long-term and short-term relationships in data.   
• We develop an Adaptive Frequency Filtering Block that utilizes Fourier transforms, local and global filters to effectively encompass all frequencies. The block adaptively reduces noise with adaptive appropriate thresholds. Additionally, we introduce the Interactive Dual Mamba Block, which excels in learning complex temporal and spatial features, enhancing adaptability and generalization across various domains.   
• Extensive experiments demonstrate that Affirm surpasses SOTA methods, proving its effectiveness and superiority in time series forecasting.

# Related Work

![](images/e12cedb5559c2f259ed2a310a5a10d78d8804f1423cb4d843591f162f8b5f40b.jpg)  
Figure 1: GPU usage and Training speed comparisons.

# State Space Models

State-space models (SSMs) effectively tackle long-range dependencies but are constrained by high computational and memory demands of state representations (Gu et al. 2021). To mitigate this, researchers have developed several variants, such as S4 (Gu, Goel, and Re´ 2021), S5 (Smith, Warrington, and Linderman 2022), and SSDNet (Lin, Koprinska, and Rana 2021). Mamba (Gu and Dao 2023) stands out by implementing an S4-based data-dependent selection mechanism to filter inputs and using hardware-aware algorithms for parallel processing. Its successful application in computer vision (Tang et al. 2024), recommendations (Liu et al. 2024a), and graphs (Wang et al. 2024a) demonstrates its effectiveness and adaptability. Moreover, it achieves lineartime efficiency with long sequences and outperforms Transformers in benchmark evaluations, positioning it as a potent alternative for time series (Cai et al. 2024). For example, MambaTS (Cai et al. 2024) uses time-varying scanning to arrange historical information and solve the scan order sensitivity issue. TimeMachine (Ahamed and Cheng 2024) proposes a multiscale Mamba to address channel mixing and independence. MambaMixer (Behrouz, Santacatterina, and Zabih 2024) uses bi-directional blocks for inter- and intrasequence analysis. To further enhance its ability in TSF, we introduce IDMB to learn both local and global features.

# Frequency-Aware Time Series

Frequency analysis is a staple in traditional signal processing, offering valuable insights into frequency representation learning (Wu et al. 2024a; Liu et al. 2023). For instance, Autoformer (Wu et al. 2021) replaces self-attention with an FFTbased auto-correlation mechanism, while FEDformer (Zhou et al. 2022b) enhances long-term periodic patterns using a DFT-based attention mechanism. Similarly, FiLM (Zhou et al. 2022a) leverages Fourier analysis to filter noise while preserving historical data. FreTS (Yi et al. 2024) captures channel and time dependencies by analyzing both real and imaginary frequency components, and FITS (Xu, Zeng, and $\mathrm { X u } \ 2 0 2 3 ,$ uses rFFT and low-pass filters for compact representation. TSLANet (Eldele et al. 2024) employs adaptive spectral blocks to effectively capture time series features.

However, many of these methods rely on feature engineering for cycle selection, focusing on dominant cycles and harmonics, which may hinder representation learning and lead to inefficiency or overfitting (Xu, Zeng, and $\mathtt { X u } 2 0 2 3 ,$ ). This paper handles this inspired by frequency filters in digital image processing and computer vision (Pitas 2000; Huang et al. 2023), where learnable Fourier filters facilitate mutual information learning, improve semantic adaptability, and reduce computational costs and parameters.

# Method

# Preliminaries

Discrete Fourier Transform. The discrete nature of DFT aligns seamlessly with digital processing, enabling efficient numerical calculations with ${ \bar { O ( N \log N ) } }$ complexity. It converts a sequence of $N$ complex numbers $x [ n ]$ , where $0 \leq \mathtt { n }$ $\leq N - 1$ , from the time domain to the frequency domain through a 1D transformation.

$$
X [ k ] = \sum _ { n = 0 } ^ { N - 1 } x [ n ] e ^ { - j ( 2 \pi / N ) k n } : = \sum _ { n = 0 } ^ { N - 1 } x [ n ] W _ { N } ^ { k n }
$$

where $\mathrm { j }$ is the imaginary unit with $W _ { N } \ = \ e ^ { - j ( 2 \pi / N ) }$ . It transforms a sequence $x [ n ]$ into its frequency spectrum $X [ k ]$ with frequencies $\omega _ { k } = \dot { 2 \pi } k / N$ , where are periodic for length $N$ . Thus, only the first $N$ points are considered.

DFT is a bijective transformation, enabling exact recovery of the original sequence $x [ n ]$ via the Inverse DFT (IDFT):

$$
x [ n ] = { \frac { 1 } { N } } \sum _ { k = 0 } ^ { N - 1 } X [ k ] e ^ { j ( 2 \pi / N ) k n }
$$

State Space Models. SSMs function as linear timeinvariant (LTI) systems that map continuous input signals $x ( t ) \mapsto y ( t )$ via a hidden state $\bar { h } ( t )$ . This state space captures the evolution of the state over time and is typically described using ordinary differential equations as follows:

$$
h ^ { \prime } ( t ) = \mathbf { A } h ( t ) + \mathbf { B } x ( t ) , y ( t ) = \mathbf { C } h ( t ) .
$$

where $\begin{array} { r } { h ^ { \prime } ( t ) = \frac { d h ( t ) } { d t } } \end{array}$ dh( , A denotes state evolution, B and C are projection parameters.

S4 and Mamba are discrete SSMs, employing a timescale parameter $\Delta$ and discretization methods like Euler’s or Zero-Order Hold (ZOH) to derive their discrete-time matrices $\overline { { \mathbf { A } } }$ and $\overline { { \mathbf { B } } }$ from the continuous-time matrices A and B:

$$
{ \overline { { \mathbf { A } } } } = \exp ( \Delta \mathbf { A } ) , { \overline { { \mathbf { B } } } } = ( \Delta \mathbf { A } ) ^ { - 1 } ( \exp ( \Delta \mathbf { A } ) - \mathbf { I } ) \cdot \Delta \mathbf { B }
$$

Then, SSM can discretize continuous signals into sequences with a time interval of $\Delta$ :

$$
h _ { t } = \overline { { \mathbf { A } } } h _ { t - 1 } + \overline { { \mathbf { B } } } x _ { t } , y _ { t } = \mathbf { C } h _ { t } .
$$

Here, $h _ { t }$ and $x _ { t }$ denote the state vector and input vector.

Finally, the model computes the output by a global convolution for training as the following:

$$
\overline { { { K } } } = ( C \overline { { { B } } } , C \overline { { { A B } } } , . . . , C \overline { { { A } } } ^ { M - 1 } \overline { { { B } } } ) , y = x * \overline { { { K } } } .
$$

# Overall Architecture

Our framework integrates two new components: the Adaptive Frequency Filtering Block (AFFB) and the Interactive Dual-Mamba Block (IDMB), illustrated in Figure 2. AFFB employs Fourier analysis to transform the time series data to the frequency domain, where it applies adaptive thresholding to attenuate high-frequency and low-frequency noises through learnable Low Pass Filtering (LPF) and High Pass Filtering (HPF), highlighting relevant spectral features and compacting model size. Then the combined signals are captured through a learnable filter using a $1 \times 1$ convolutional (linear) layer, a ReLU function, and another linear layer. After processing, the IFFT reconstructs the temporal representations with reduced noise and enhanced features. IDMB interactively refines local and global features with different kernel sizes and dropout mechanisms, enhancing adaptability to temporal dynamics in TSF. Together, these components effectively balance local and global temporal features for robust time series analysis.

# Embedding Layer

Given an input time series $X \in R ^ { C \times L }$ , where $C$ is the channel number and $L$ is the sequence length, $X$ is divided into $N$ non-overlapping patches $\{ P _ { 1 } , P _ { 2 } , . . . , P _ { N } \}$ , each of length $p$ , totally patches $\bar { P _ { i } } \in R ^ { C * p }$ . Each patch is mapped to a new dimension $d$ , resulting in $P _ { i }  P _ { i } ^ { \acute { \prime } } \in R ^ { C \times d }$ . Position embeddings $E _ { i }$ are added to each patch to preserve temporal order, giving $X _ { P E _ { i } } = P _ { i } ^ { \prime } + E _ { i }$ . The complete set of latent patches is $X _ { P E } = X _ { P E _ { 1 } } , X _ { P E _ { 2 } } , . . . , X _ { P E _ { N } }$ , with the positional embeddings enhancing temporal correlation capture.

# Adaptive Fourier Filter Block

High frequencies are often discarded as noise, and the denoised signal is obtained by reconstructing only the lowfrequency components. This approach has drawbacks: (i) it ignores some useful information in high frequencies, and (ii) it may reconstruct weak noise in low frequencies as part of the output, leading to reduced signal-to-noise ratios and increased errors (Li, Bu, and Yang 2023). Inspired by (Rao et al. 2021; Huang et al. 2023), we propose the AFF Block, which aims to learn spatial information through global circular convolution operations.

Fast Fourier Transformations. For a given $X _ { P E }$ , its representation is calculated as:

$$
X _ { F } = \mathcal { F } [ X _ { P E } ] \in \mathcal { C } ^ { C \times L _ { F } ^ { \prime } }
$$

Here, $\mathcal { F } [ \cdot ]$ denotes the 1D FFT operation, and $L _ { F } ^ { \prime }$ is the sequence length after Fourier transformation.

Adaptive Low Pass Filter. High-frequency components often indicate rapid fluctuations or deviations from the underlying trend, making the signal more random and harder to interpret (Eldele et al. 2024). Removing high-frequency noise can help to learn the trend and periodic patterns, which is crucial in TSF. Besides, this operation can simplify the model, accelerate training, and lower computational costs. Thus, we propose an adaptive low-pass filter with a learnable threshold that dynamically adjusts the filtering degree to suppress irrelevant noise while preserving vital information, resulting in a more concise frequency representation.

The trainable threshold $\theta _ { h i g h }$ is adaptively tuned to match the frequency characteristics, optimized through backpropagation to effectively distinguish valuable information from high-frequency noise and eliminate components exceeding the threshold. The formula is as follows:

![](images/a06f883a190a25facb998998740a56b01c678c66c96bc930bc2fc072b93d4a1b.jpg)  
Figure 2: The structure of our proposed Affirm.

$$
X _ { F i l t e r } ^ { h i g h } = X _ { F } \odot ( | F | \leq \theta _ { h i g h } )
$$

where $\odot$ denotes element-wise multiplication (the Hadamard product), and $| F | \le \theta _ { h i g h }$ represents a binary mask where frequencies below the threshold $\theta _ { h i g h }$ are retained, while others are filtered out.

Adaptive High Pass Filter. Low-frequency components generally encapsulate the underlying trends and cycles in time series data. However, they may still contain noise, particularly systematic errors from data collection or processing, such as sensor calibration issues, algorithmic biases, and data entry mistakes (Li, Bu, and Yang 2023). Consequently, we implemented an adaptive low-frequency denoising branch, which is applied after handling high-frequency noise. This method offers several benefits: (1) It removes unnecessary noise, allowing the model to focus on crucial frequency components, and enhancing generalization. (2) It helps capture essential features, reducing overfitting and improving performance. (3) It improves the model’s ability to handle non-smooth data, stabilizing the data structure. The process with a learnable threshold $\theta _ { l o w }$ is:

$$
X _ { F i l t e r } ^ { l o w } = X _ { F } \odot ( | F | > \theta _ { l o w } )
$$

Similarly, $| F | > \theta _ { l o w }$ means frequencies above the threshold $\theta _ { l o w }$ are retained, while others are discarded.

Learnable Linear Filters. After adaptive filtering in the frequency domain, Affirm employs three learnable linear filters: a global filter $\mathop { \mathcal { M } } ( X _ { F } ) _ { G }$ from the original frequency $X _ { F }$ ; a local filter $\mathcal { M } ( X _ { F } ^ { h i g h } ) _ { L }$ from high frequency; and a local filter $\mathcal { M } ( X _ { F } ^ { l o w } ) _ { L }$ from low frequency. Each filter is tailored to match the corresponding frequency features $X _ { F }$ , $X _ { F } ^ { h i g h }$ , and $X _ { F } ^ { l o w }$ . The process is as follows:

$$
X _ { G } = \mathcal { M } ( X _ { F } ) _ { G } \odot X _ { F }
$$

$$
X _ { L } ^ { h i g h } = \mathcal { M } ( X _ { F } ^ { h i g h } ) _ { L } \odot X _ { F } ^ { h i g h }
$$

$$
X _ { L } ^ { l o w } = \mathcal { M } ( X _ { F } ^ { l o w } ) _ { L } \odot X _ { F } ^ { l o w }
$$

As shown in Figure 2, to make the network as lightweight as possible, $M ( \cdot )$ is implemented by a $1 * 1$ convolutional (linear) layer, followed by ReLU, and another linear layer. The three filtered spectra are then integrated to capture the full spectral feature details, calculated as:

$$
X _ { m i x e d } = X _ { G } + X _ { L } ^ { h i g h } + X _ { L } ^ { l o w }
$$

The $X _ { m i x e d }$ denotes the global and local adaptive frequency mixing of $X _ { F }$ . The multiplication operation is mathematically equivalent to the dynamic circular convolutional process.

Inverse Fourier Transform. We further use IFFT to transfer the mixed frequency back to the time domain:

$$
X _ { T } = \mathcal { F } ^ { - 1 } [ X _ { m i x e d } ] \in \mathcal { R } ^ { C \times d }
$$

where $\mathcal { F } ^ { - 1 } ( \cdot )$ represents IFFT, ensuring the combined features remain consistent with the original time series data.

# Interactive Dual Mamba Block

The original Mamba structure is shown in the bottom right of Figure 2. Apply LayerNorm first, then process the features through two parallel branches: the left SSM branch for sequence modeling and the right for a gated nonlinear layer.

$$
h _ { t } = S S M ( C o n v ( L i n e a r ( X _ { T } ) ) ) + \sigma ( L i n e a r ( X _ { T } ) )
$$

We design an IDMB with causal convolutions of different kernel sizes to process $X _ { T }$ after AFFB. $X _ { T }$ is passed through these convolutions with LayerNorm to fully extract features, capturing both local features and long-term dependencies through parallel interactions. Specifically, the first convolution, using a $2 \mathrm { x } 2$ depth-wise filter $f _ { 2 * 2 } \colon h _ { 2 * 2 } =$ $L N ( S S M ( \sigma ( C o n v 1 ( L i n e a r ( X _ { T } ) ) ) ) ) \}$ ), captures fine-grained local patterns, while the second, with a $4 \mathrm { x } 4$ depth-wise filter $f _ { 4 * 4 } \colon h _ { 4 * 4 } \ = \ L N ( S S M ( \sigma ( C o n v 2 ( L i n e a r ( X _ { T } ) ) ) ) ) \mathrm { , }$ ), identifies broader dependencies. $\mathrm { C o n v 1 } ( \odot )$ and $\mathbf { C o n v } 2 ( \odot )$ are 1Dconvolution layers. IDMB allows each layer to modulate the other, enhancing comprehensive feature extraction by promoting interaction between different scales, computed as:

$$
\mathcal { H } _ { 1 } = \sigma ( h _ { 2 * 2 } ) \odot h _ { 4 * 4 } \odot \sigma ( L i n e a r ( X _ { T } ) )
$$

$$
\mathcal { H } _ { 2 } = \sigma ( h _ { 4 \ast 4 } ) \odot h _ { 2 \ast 2 } \odot \sigma ( L i n e a r ( X _ { T } ) )
$$

$\mathcal { H } _ { 1 }$ and $\mathcal { H } _ { 2 }$ are then added and passed through a linear convolution layer Conv3 $\left( \bullet \right)$ tailored to specific task:

$$
\mathcal { O } _ { I D M B } = C o n v 3 ( \mathcal { H } _ { 1 } + \mathcal { H } _ { 2 } )
$$

where $O _ { I D M B }$ is the output features.

# Self-Supervised Pretraining

Self-supervised learning utilizes data’s inherent structure to extract valuable information without needing external labels, which is vital in scenarios where data acquisition is costly or annotations are scarce (Nie et al. 2023). Inspired by methods in computer vision and NLP, we introduced a selfsupervised pre-training phase for Affirm using a masked autoencoder approach for time series data (He et al. 2022). By partially masking input series, Affirm is trained to detect hidden patterns and reconstruct complete information, enhancing its robustness, generalization, and feature extraction. Affirm focuses on broader data patches rather than single-point masking, encouraging thorough sequence analysis and optimizing patch reconstruction through MSE minimization.

# Experiments

# Experimental Setup

Datasets. To validate our model, we evaluate Affirm on 8 benchmarks: 4 ETT datasets (ETTh1, ETTh2, ETTm1, ETTm2), Electricity (ECL), Exchange, Traffic, and Weather (Wu et al. 2021).

Baselines. We assessed Affirm against ten baselines, including: 1) Transformers: iTransformer (Liu et al. 2024b), PatchTST (Nie et al. 2023), Crossformer (Zhang and Yan 2023); 2) MLPs: Dlinear (Zeng et al. 2023), Rlinear (Li et al. 2023), TimeMixer (Wang et al. 2024c); 3) CNNs: TSLANet (Eldele et al. 2024); 4) Mambas: DTMamba (Wu, Gong, and Zhang 2024); 5) and LLMs: Time-LLM (Jin et al. 2023), GPT4TS (Zhou et al. 2023).

Implementation Details. We conduct all experiments on an NVIDIA GeForce RTX 4090 Ti GPU with 64-bit Linux 5.15.0-56-generic, with 60/20/20 train/validation/test split for ETTs, and 70/10/20 for other datasets. Similar to (Zhou et al. 2023) settings, we use the look-back window of 336 for ETTs, 96 for exchange, 512 for Traffic and Weather, and 96 for Electricity. All datasets are normalized during training (Kim et al. 2021). For baselines, we report their best results if setups match; otherwise, we rerun their code.

# Results

As shown in Table 2, the Time-LLM model excels by leveraging the robust llama-7B model, effectively capturing complex patterns. Affirm outshines in almost all benchmarks except Time-LLM, achieving second-best performances on seven out of eight datasets and improving MSE $4 . 0 \%$ and $6 . 9 \%$ over SOTA PatchTST in ETTh1 and ECL (avg), confirming its robustness in diverse scenarios. Affirm and TSLANet both use the neural operator, with Affirm achieving superior MSE improvements of $1 . 6 \%$ and $3 . 5 \%$ on ETTm2 and Exchange (avg), indicating its effectiveness in learning long-term dependencies. Furthermore, results also show Affirm excels beyond specialized Transformers and MLPs. These models, such as iTransformer and Dlinear, perform competitive performance in certain cases but fall short in others. GPT4TS also showcases the capability of GPT models in forecasting by achieving the secondbest in some datasets. Despite Time-LLM’s slightly better performance, it requires much higher computational resources. For instance, on the ETTh1 dataset, Affirm presents a nearly equivalent performance to Time-LLM with an MSE of 0.411 against Time-LLM’s 0.408 but uses far less computational costs— $9 . 7 8 \mathrm { e } + 0 7$ FLOPS against Time-LLM’s $7 . 3 \mathrm { e } + 1 2$ , highlights Affirm’s efficient balance between performance and computational demand.

# Ablation Study

Various Variants of Affirm. To assess the impact of Affirm’s various components, we conduct ablation studies as detailed in Table 3 . Removing the IDMB (i.e., w/o IDMB) and AFFB (i.e., w/o AFFB) significantly impairs performance, with AFFB’s absence resulting in more pronounced declines. Specifically, eliminating IDMB decreased MSE by $2 . 6 \%$ and $5 . 4 \%$ and MAE by $2 . 5 \%$ and $4 . 4 \%$ on ETTh1 and Weather. The removal of AFFB led to larger reductions in MSE by $5 . 1 \%$ and $5 . 8 \%$ , emphasizing its importance in feature extraction and denoising. We also investigate the effects of omitting local adaptive components within AFFB—LPF and HPF—on noise filtering. Removing both adversely impacted performance, especially the LPF, indicating that high-frequency noise more severely affects outcomes. A scenario without both local filters (i.e., $\scriptstyle { \cal N } / 0 \mathrm { L P F + H P F } ,$ ) showed deteriorating performance over time with only global filtering, confirming the essential role of local filters in noise management. Similarly, the importance of pretraining is confirmed, as its absence marginally reduces the model’s predictive performance.

Varying Look-back Window. In principle, a longer lookback window should improve prediction accuracy by expanding the receptive field, but this is not typically observed in most Transformer-based models (Zeng et al. 2023). As in Figure 3, these models including TSLANet, show limited benefits from extended look-back periods, indicating a weak grasp of temporal information. In contrast, the MSE of PatchTST and our Affirm decreases as the look-back window lengthens, with Affirm consistently outperforming PatchTST across all settings, demonstrating its superior ability in both short-term and long-term prediction tasks.

<html><body><table><tr><td colspan="2">Methods</td><td></td><td>Affirm</td><td>DTMamba</td><td>TSLANet</td><td>Time-LLM</td><td>GPT4TS</td><td>iTransformer</td><td>PatchTST</td><td>Crossformer</td><td>Rlinear</td><td>Dlinear</td><td>TimeMixer</td></tr><tr><td colspan="2">Metrics</td><td colspan="8">MSE MAE|MSE MAE|MSE MAE|MSE MAE|MSE MAE|MSE</td><td>MAE|MSE</td><td>MAE|MSE MAE|MSE MAE|MSE</td><td></td><td>MAE|MSE MAE</td><td></td></tr><tr><td></td><td>96 192</td><td>10.363 0.408</td><td>0.392 0.421</td><td>0.3860.399|0.370 0.4260.424</td><td>0.412</td><td>0.394|0.362 0.398</td><td>0.392</td><td>0.376 0.397</td><td>0.3860.405</td><td>0.382 0.401</td><td>0.4230.448|0.3860.395|0.375</td><td></td><td></td><td>0.399|0.3750.400</td></tr><tr><td></td><td>6 720</td><td>0.424 0.450</td><td>0.426 0.453</td><td>0.480 0.4500.399</td><td>0.417 0.416</td><td>0.430</td><td>0.418 0.427 0.4420.433</td><td>0.4160.4180.4410.436 0.487</td><td>0.428 0.458 0.451</td><td>0.425 0.436</td><td>0.4710.4740.437 0.570 0.5460.479</td><td>0.424 0.446 0.439</td><td>0.405 0.416 0.443</td><td>0.429 0.421 0.4840.458</td></tr><tr><td></td><td>Avg.</td><td></td><td>0.423 |0.444 0.435|0.413</td><td>0.484 0.470|0.472</td><td></td><td>0.475|0.442</td><td>0.457</td><td>0.4770.456</td><td>0.503 0.491</td><td>0.452 0.459</td><td>0.6530.6210.4810.4700.472</td><td></td><td>0.490</td><td>0.498 0.482</td></tr><tr><td></td><td>96</td><td>10.411 0.276</td><td></td><td>0.343|0.290 0.340|0.280</td><td>0.341</td><td>0.426 |0.408 0.268</td><td>0.328</td><td>0.423 |0.428 0.426 |0.454 0.448|0.428 0.2850.3420.2970.349</td><td></td><td>0.285</td><td>0.430 |0.529 0.522|0.446 0.434|0.423 0.340|0.745 0.584|0.288 0.338|0.289</td><td></td><td></td><td>0.437 |0.447 0.440</td></tr><tr><td></td><td></td><td></td><td>0.373</td><td>0.366 0.4090.37</td><td></td><td>0.375 0.329</td><td>0.375</td><td>0.354 0.489</td><td>0.388 0.40</td><td>0.356</td><td>0.386107 0510750260</td><td></td><td></td><td>0.353|0.289 0.341</td></tr><tr><td></td><td></td><td>Avg.</td><td>0.391 0.331 0.381</td><td>0.429 0.416 0.437</td><td>0.404</td><td>0.440 0.372</td><td>0.420</td><td>0.4060.441</td><td>0.427 0.445</td><td>0.395 0.427</td><td></td><td>1.1040.7630.420 0.440|0.605</td><td>0.551</td><td>0.4180.372 0.44 0.412 0.434</td></tr><tr><td></td><td></td><td>96 192</td><td>0.285 0.323</td><td>[0.363 0.344|0.325 0.360|0.289</td><td>0.395|0.333</td><td>0.383|0.334 0.349|0.272</td><td>0.383|0.355 0.334</td><td>0.395 0.2920.346|0.334</td><td>|0.383 0.407 0.368</td><td>0.347 0.291</td><td>0.387|0.942 0.684|0.374 0.399|0.431 0.340|0.4040.426|0.3550.376|0.299</td><td></td><td>0.447|0.364</td><td>0.395</td></tr><tr><td></td><td></td><td>336 720</td><td>0.351 0.384 0.418</td><td>0.3650.375 0.3860.328 0.3960.4050.355 0.414|0.454 0.442|0.421</td><td>0.370 0.389 0.425</td><td>0.310 0.352 0.383</td><td>0.358 0.384 0.411</td><td>0.3320.3720.377 0.391 0.3660.394 0.4170.421</td><td>0.4260.420 0.491 0.459</td><td>0.328 0.365 0.389 0.422 0.423</td><td>0.3650.4500.4510.3910.392 0.6660.5890.2870.4500.425</td><td>0.5320.5150.4240.415</td><td>0.343 0.335 0.369 0.386</td><td>0.3200.357 0.3650.361 0.381 0.3900.404</td></tr><tr><td></td><td></td><td>Avg.10.344 96</td><td>10.167 0.260</td><td>0.377|0.388 0.399|0.348 0.177</td><td>0.259|0.169</td><td>0.383|0.329 0.259|0.161</td><td>0.253</td><td>0.372|0.352 0.383 |0.407 0.173 0.262</td><td>0.180 0.264</td><td>0.410 |0.352 0.254</td><td>0.379|0.513 0.495|0.414 0.408|0.357</td><td></td><td>0.421</td><td>0.454 0.441 0.379 |0.381 0.395</td></tr><tr><td></td><td></td><td>192 720</td><td>8.22 0.296 0.351 0.377</td><td>0.340 0.300224</td><td>0.297</td><td>0.27 0.380 0.352</td><td>0.223</td><td>0.290.3410311 0.3409</td><td></td><td>0.169 0.230 0.294</td><td>0.287 0.366|0.182 0494054206034024</td><td>0.265</td><td>0.167 0.260</td><td>10.175 0.258 0.3430.237 0.3490</td></tr><tr><td></td><td>96</td><td>Avg.</td><td>10.252</td><td>0.395 0.3940.354 0.315|0.281 0.325|0.256</td><td></td><td>0.316 |0.251</td><td>0.379</td><td>0.3780.401 0.313|0.267 0.326|0.288 0.332|0.264</td><td>0.4120.407</td><td>0.378 0.386</td><td>1.730 0.316|0.757 0.611|0.286 0.327|0.267</td><td>1.042|0.407 0.398</td><td>0.397 0.421</td><td>0.391 0.396 0.332|0.275 0.323</td></tr><tr><td></td><td></td><td>192 0.146 336 0.162</td><td>[0.129 0.223 0.239 0.252</td><td>0.1660.256|0.136 0.178 0.268</td><td>0.229 0.152 0.244 0.168</td><td>0.131 0.152 0.262 0.160</td><td>0.224 0.241 0.248</td><td>10.1390.238|0.1480.240 0.153 0.251</td><td>0.162 0.253</td><td>0.138 0.149 0.243</td><td>0.230|0.219 0.314|0.201 0.281|0.140 0.231 0.3220.201</td><td>0.283</td><td>0.153 0.249</td><td>0.237|0.1530.247 0.166 0.256</td></tr><tr><td></td><td>720</td><td>0.191 Avg.|0.157</td><td>0.288</td><td>0.1970.289 0.243 0.326|0.205</td><td>0.293</td><td>0.192 0.257|0.158</td><td></td><td>0.1690.2660.178 0.298|0.206 0.297</td><td>0.269 0.225 0.317</td><td>0.169 0.262 0.211 0.299</td><td>0.2460.3370.2150.298 0.280 0.3630.257 0.331</td><td></td><td>0.169 0.267 0.203 0.301</td><td>0.1850.277 0.225 0.310</td></tr><tr><td></td><td>96 192</td><td>0.080 0.169</td><td>0.198 0.296</td><td>0.250 |0.196 0.285|0.165 0.0830.201 0.083</td><td>0.201 0.177 0.299</td><td>0.123 0.224</td><td>0.251 0.344 0.171</td><td>0.252 |0.167 0.263 |0.178 10.0820.199|0.0860.206</td><td>0.270 |0.167</td><td>0.088</td><td>0.259 |0.244 0.334|0.219 0.298|0.166 0.205|0.256 0.367|0.093 0.217</td><td>0.081</td><td></td><td>0.264|0.182 0.272</td></tr><tr><td></td><td>336 720</td><td>0.325 0.852</td><td>0.411 0.690</td><td>0.1730.295 0.3460.427 0.331 0.868 0.6980.888</td><td>0.417 0.739</td><td>0.377 1.018</td><td>0.451 0.354 0.771</td><td>0.293 0.428 0.877 0.704|0.847</td><td>0.177 0.299 0.331 0.417 0.691</td><td>0.176 0.299 0.301 0.397 0.901 0.714</td><td>0.4700.5090.184 0.307 1.268 0.8830.351 1.767 1.0680.886 0.714</td><td>0.4320.305 0.643</td><td>0.157 0.293 0.414</td><td>0.203|0.091 0.215 0.1970.318 0.4160.472</td></tr><tr><td></td><td>96</td><td>Avg.10.356 0.364</td><td>0.255 0.487</td><td>0.399|0.368 0.405|0.370 0.317|0.372</td><td>0.414|0.436 0.261|0.362</td><td></td><td>0.454 |0.371 0.248</td><td>0.406 |0.360</td><td>0.403|0.367</td><td></td><td>0.404 |0.940 0.707|0.379 0.418|0.297</td><td></td><td>0.601 0.378</td><td>0.968 0.725 0.418 0.433</td></tr><tr><td></td><td>192 336</td><td>0.381 0.392 0.433</td><td>0.262 0.268</td><td>0.4980.3250.388 0.5110.334 0.394</td><td>0.2660.374 0.269</td><td>0.385 0.430</td><td>0.247 0.271 0.288</td><td>0.3880.282 0.4070.290 0.4120.294</td><td>0.3950.268 0.4170.276 0.433 0.283</td><td>0.401 0.267 0.406 0.268 0.421 0.277</td><td>10.5220.290|0.6490.389 0.5580.3050.609 0.3690.436</td><td>0.5300.2930.6010.3660.423</td><td>0.410 0.282 0.287</td><td>0.462 0.285 0.4730.296</td></tr><tr><td></td><td>720 Avg.</td><td>0.392</td><td>0.268|0.507</td><td>0.2900.5330.3260.430 0.326| 0.396</td><td>0.289 0.271 |0.388</td><td></td><td></td><td>0.450 0.3120.467 0.295 |0.428</td><td>0.302</td><td>0.452 0.297</td><td>0.589 0.3280.647 0.387</td><td>0.466</td><td>0.296</td><td>0.4980.296 0.3150.506 0.313</td></tr><tr><td></td><td>96</td><td>0.146</td><td>0.196 0.171</td><td></td><td></td><td></td><td>0.264</td><td>|0.414 0.212</td><td>0.214</td><td>0.282|0.420 0.204</td><td>0.277 |0.550 0.403|0.627</td><td>0.378|0.434</td><td>0.295|0.484</td><td>0.297</td></tr><tr><td></td><td></td><td>0.321</td><td>0.194 0.239</td><td>0.218|0.148 0.220 0.257</td><td>0.197 0.293 0.242 0.349 0.3460.325</td><td>0.147 0.282 0.337 0.304 0.264 |0.225 0.257</td><td>0.201 0.239 0.316</td><td>0.162 0.204 0.248 0.3260.337</td><td>0.174 0.2210.254 0.358 0.349</td><td>0.160 0.329 0.338</td><td>02402572</td><td>0.1580.230|0.192 0.232</td><td>0.176 0.237</td><td>0.1630.209 0.3190.258 0.250</td></tr></table></body></html>

Table 2: Multivariate long-term series forecasting results on different prediction lengths $\in \{ 9 6 , 1 9 2 , 3 3 6 , 7 2 0 \}$ . A lower value indicates better performance. Bold: best, underlined: second best.

Table 3: Ablation study of each component in Affirm.   

<html><body><table><tr><td rowspan="2">Variants</td><td colspan="3">ETTh1 (avg.) Weather (avg.)</td></tr><tr><td>MSE</td><td>MAE MSE</td><td>MAE</td></tr><tr><td>W/o IDMB</td><td>0.423</td><td>0.434 0.239</td><td>0.273</td></tr><tr><td>W/o AFFB</td><td>0.434</td><td>0.435 0.240</td><td>0.275</td></tr><tr><td>W/o LPF</td><td>0.421 0.432</td><td>0.236</td><td>0.271</td></tr><tr><td>W/o HPF</td><td>0.417</td><td>0.430 0.229</td><td>0.265</td></tr><tr><td>W/o LPF+HPF</td><td>0.422</td><td>0.433 0.237</td><td>0.272</td></tr><tr><td>w/o pretraining</td><td>0.415</td><td>0.426 0.228</td><td>0.263</td></tr><tr><td>Affirm</td><td>0.411</td><td>0.423</td><td>0.226 0.261</td></tr></table></body></html>

# Efficiency of Adaptive Filter in Noise Reduction

We further explore the effectiveness of adaptive filters (LPF, HPF) in reducing noise, as illustrated in Figure 4. Specifically, it depicts how the Transformer, PatchTST, Affirm, and Affirm w/o filters perform under varying degrees of Gaussian noise. While the Transformer’s performance declines sharply with increased noise, PatchTST is more sensitive to noise than ETTh1 in large Weather, and both Affirm variants exhibit stable and robust behavior, especially Affirm (with filters). This indicates the significant advantage of adaptive filters in managing noisy environments. Notably, Affirm w/o adaptive filters still outperforms Transformer and PatchTST, emphasizing the benefits of IDMB and global filters.

# Scaling Efficiency

To evaluate Affirm’s scalability, we compared it against one of the best-performing Transformers in TSF, i.e., PatchTST (Nie et al. 2023) on ETTm1 with varying sizes and layers. Figure 5 shows that Affirm consistently outperforms PatchTST across all conditions. As data size increases, the performance of both models slightly decreases with layers increase, but PatchTST’s drop is more pronounced. Notably, on a smaller dataset $( 1 \% )$ , Affirm remains stable despite increased layers, while PatchTST’s curve suffers a marked upswing, indicating a tendency toward overfitting or inefficiency as model complexity rises with limited data. This

+Transformer → Informer Fedformer + Affirm +Autoformer \*PatchTST + TSLANet Electricity (96) Traffic (96) Weather (96) 1.3 0.5 0.7   
GPE 0.3 0.8 0.5 0.3 0.1 0.3 0.1 24 4896 192 336 720 24 48 96 192 336 720 24 48 96 192336720 720 720 720 1.1 1.2 1.3   
PE 0.6 0.8 0.7 0.1 0.3 0.2 2448 96 192 336 720 24 48 96 192336720 2448 96 192336720 Look back Look back Look_back

Weather ETTh1 1.0 0.45 0.9 0.40 Transformer 0.8 Transformer 0.30 E 0.35 Affirmw/oFilter PatchTST 0.7 PatchTST Affirmw/o Filter 0.25 Affirm 0.6 Affirm 0.20 0.5 0.15 0.4 0.0 0.5 1.01.5 2.0 2.5 0.0 0.5 1.0 1.5 2.0 2.5 NR NR

may stem from PatchTST’s inherent design, which might yield diminishing returns as model depth increases, posing optimization challenges that impede the learning of effective features. In contrast, Affirm’s performance remains steady or improves with larger datasets and increased layer counts, showcasing its capacity to effectively utilize extensive data.

![](images/97442170ce02b9e4dfcd516d57a0bd6bdc04fb0e52036cd5a51e5a76e6f16ab1.jpg)  
Figure 3: MSE on 3 large datasets.   
Figure 4: Effectiveness of Adaptive Filter in noise reduction.   
Figure 5: Scaling comparison of Affirm vs. PatchTST across different layer counts and data percentages on ETTm1.

# Complexity Analysis

We assessed Affirm’s complexity against various TSF models on the ECL dataset, focusing on training parameters, MACs, and MSE over a 96 look-back window and a 720 prediction length. Table 4 highlights Affirm’s efficiency and superior performance. Models like Transformer, Informer, Autoformer, FEDformer, and FiLM have parameter counts ranging from 13.61M to 20.68M and MACs between 3.93G to 5.97G, yet showing higher MSEs compared to Affirm. In contrast, against lightweight PatchTST and TSLANet, Affirm aligns in parameter size but significantly decreases MACs by $6 6 . 1 \%$ and $7 7 . 4 \%$ , and MSE by $1 2 . 9 \%$ and $9 . 6 \%$ , respectively. Remarkably, Affirm slashes both parameters and MACs by over $9 9 \%$ compared to TimesNet, which has a parameter count of 301.7M and MACs of 1226.49G and reduces MSE by $1 4 . 3 \%$ . This drastic cut in computational demand, coupled with improved performance, positions Affirm as a lightweight yet powerful alternative.

Table 4: Number of training parameters, MACs, and MSE of TSF models under look-back window $^ { = 9 6 }$ and forecasting horizon $\scriptstyle 1 = 7 2 0$ on the large Electricity dataset.   

<html><body><table><tr><td>Method</td><td>MACs</td><td>Parameters</td><td>MSE</td></tr><tr><td>Transformer</td><td>4.03G</td><td>13.61M</td><td>0.491</td></tr><tr><td>Informer</td><td>3.93G</td><td>14.38M</td><td>0.399</td></tr><tr><td>Autoformer</td><td>4.41G</td><td>14.91M</td><td>0.412</td></tr><tr><td>FEDfomrer</td><td>4.41G</td><td>20.68M</td><td>0.264</td></tr><tr><td>PatchTST</td><td>5.07G</td><td>1.5M</td><td>0.248</td></tr><tr><td>FiLM</td><td>5.97G</td><td>14.91M</td><td>0.268</td></tr><tr><td>TimesNet</td><td>1226.49G</td><td>301.7M</td><td>0.255</td></tr><tr><td>TSLANet</td><td>7.6G</td><td>1.4M</td><td>0.239</td></tr><tr><td>Affirm</td><td>1.72G</td><td>1.7M</td><td>0.216</td></tr></table></body></html>

# Sensitivity Analysis

Affirm involves several hyperparameters requiring careful tuning for optimal performance. We conducted a sensitivity analysis of mask ratio and dropout on ETTh1 and Weather. Table 5 reveals that the best performance is achieved with a mask ratio of 0.4. Reasonable parameters can improve generalization and enhance robustness to adapt to unstable and data loss scenarios.

Table 5: Sensitivity experiments of mask ratio.   

<html><body><table><tr><td>mask_ratio</td><td>p=0.01</td><td>p=0.1</td><td>p=0.2</td><td>p=0.4</td><td>p=0.5</td><td>p=0.6</td></tr><tr><td>ETTh1</td><td>0.371</td><td>0.369</td><td>0.366</td><td>0.363</td><td>0.365</td><td>0.367</td></tr><tr><td>Weather</td><td>0.159</td><td>0.152</td><td>0.150</td><td>0.146</td><td>0.149</td><td>0.151</td></tr></table></body></html>

# Conclusion

We introduce Affirm, a novel lightweight framework for time series forecasting, with an innovative combination of Mamba with adaptive Fourier filters as a potent alternative to Transformers. It introduces a dual interactive Mamba and uses dropout regularization to improve parameter selectivity and prevent overfitting. Besides, it integrates adaptive global and local Fourier filters in the frequency domain, where the local neural operator utilizes learnable thresholds to filter noise. Experiments demonstrate Affirm’s ability to achieve SOTA ability and efficiency trade-offs in TSF, particularly under noisy conditions and various data sizes. Complexity analysis confirms Affirm’s superiority with significantly lower computational costs. Moreover, our in-depth layerwise scaling analysis reveals that Affirm outperforms Transformers on smaller datasets and exhibits enhanced scalability with increased layers, especially in larger datasets. Affirm opens up new avenues for TSF as a foundation model.

# Acknowledgments

We thank the anonymous reviewers for their helpful feedbacks. The work is supported by Zhejiang Provincial Science and Technology Plan Project (No. 2023C03183), Key Scientific Research Base for Digital Conservation of Cave Temples (Zhejiang University).