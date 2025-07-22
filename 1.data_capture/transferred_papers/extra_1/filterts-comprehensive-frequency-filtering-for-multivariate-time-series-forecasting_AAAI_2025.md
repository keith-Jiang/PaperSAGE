# FilterTS: Comprehensive Frequency Filtering for Multivariate Time Series Forecasting

Yulong Wang1,2, Yushuo Liu1,2, Xiaoyi Duan1,2, Kai Wang1,2,\*

1College of Computer Science, Nankai University 2Tianjin Media Computing Center yl.wang, yushuo.liu, xyd @mail.nankai.edu.cn, wangk@nankai.edu.cn

# Abstract

Multivariate time series forecasting is crucial across various industries, where accurate extraction of complex periodic and trend components can significantly enhance prediction performance. However, existing models often struggle to capture these intricate patterns. To address these challenges, we propose FilterTS, a novel forecasting model that utilizes specialized filtering techniques based on the frequency domain. FilterTS introduces a Dynamic Cross-Variable Filtering Module, a key innovation that dynamically leverages other variables as filters to extract and reinforce shared variable frequency components across variables in multivariate time series. Additionally, a Static Global Filtering Module captures stable frequency components, identified throughout the entire training set. Moreover, the model is built in the frequency domain, converting time-domain convolutions into frequencydomain multiplicative operations to enhance computational efficiency. Extensive experimental results on eight real-world datasets have demonstrated that FilterTS significantly outperforms existing methods in terms of prediction accuracy and computational efficiency.

code — https://github.com/wyl010607/FilterTS

Daily   
Monthly Non-ral Frequ Frey eney Twelve-hourly Eight-hourly .Six-hourly   
TtTTttttTt+\*\*\*\*! oo090000+ Dooooo0oo@e Frequency (a) HUFL Spectrum Plot on Training Set □ MUFL Syneced Frequencies □ irrmlPmmr:imel.rnrm 中 Frequency Frequency   
(b) Spectrum Plot in First and Second Months of Training Set

# Introduction

Multivariate time series forecasting is crucial across various domains such as finance(Olivares et al. 2023), transportation(Bui, Cho, and Yi 2022), and energy(Zhou et al. 2021), where the data often exhibits complex temporal characteristics like periodic changes and trends(Qiu et al. 2024).

Traditional time series analysis methods predominantly focus on time-domain analysis (Zhou et al. 2021; Nie et al. 2023; Liu et al. 2024), yet they often fall short in capturing periodic information directly(Wang et al. 2023). Recent studies have increasingly applied frequency domain analysis to time series, which transforms time series into the frequency space, allowing distinct frequency components to be clearly separated (Xu, Zeng, and Xu 2024; Yi et al. 2024). Despite these advancements, these methods sometimes fail to effectively differentiate the importance of various frequency components, treating all components equally, which might lead to model overfitting while neglecting crucial a priori periodic information and inter-variable dependencies.

To address these challenges, an effective strategy is the selective extraction and emphasis of frequency components that are most predictive of future observations. We categorize features in time series into (a) stable frequency components and (b) variable frequency components based on their behavior in the frequency domain. Stable components, such as natural cycles (daily, weekly, monthly) and frequencies associated with specific business processes, often appear as dominant frequencies in the series, as determined through frequency domain analysis on training data, depicted in Figure 1a. Variable components, whose characteristics may change over time due to environmental or external factors, are often elusive with static filters. Capturing these dynamically changing frequency components effectively remains a research gap. We hypothesize, based on experimental observations, that these variable frequency components are not only present within individual variables but may also be shared across different variables in a multivariate time series (Zhao and Shen 2024). Specifically, there may be synchronization in frequency and intensity changes among these variables, as shown in Figure 1b, revealing underlying inter-variable connections crucial for enhancing multivariate forecasting models.

Based on the foregoing analysis, we introduce FilterTS, an innovative multivariate time series forecasting model that enhances and precisely extracts frequency components through carefully designed filtering modules. The model operates in the frequency domain, converting time-domain convolution to frequency-domain multiplication to improve computational efficiency. FilterTS utilizes two types of filtering modules:

The Static Global Filtering Module are constructed by performing frequency analysis on the entire training set, building band-pass filters at frequencies corresponding to components with relatively high amplitudes, thereby capturing the dominant stable frequency components. The Dynamic Cross-variable Filtering Module, on the other hand, treats each variable as a filter for the others, dynamically extracting shared frequency components across variables within each lookback window. This approach enhances the capture of variable frequency components. The output sequences from these filters are then merged by a complex matrix.

The experimental results detailed in this paper substantiate the exceptional performance of the FilterTS model across eight real datasets, FilterTS demonstrates superior forecasting accuracy and computational efficiency compared to existing state-of-the-art methods. The main contributions of this paper are summarized as follows:

• We introduce FilterTS, a novel multivariate time series forecasting model leveraging filters to enhance frequency component extraction and improve prediction accuracy. • We develop the Static Global Filtering Module, designed to capture stable periodic components, and the Dynamic Cross-Variable Filtering Module, which dynamically extracts and emphasizes significant frequency components shared across variables. • Our empirical evaluation across eight real-world datasets demonstrates that FilterTS surpasses existing methods in forecasting accuracy and computational efficiency.

# Related Work

# Multivariate Time Series Forecasting Models

Multivariate Time Series (MTS) involves a set of simultaneously sampled time series data. Commonly, MTS forecasting models attempt to capture dependencies among variables, employing methods such as MLPs (Shao et al. 2022; Ekambaram et al. 2023; Zeng et al. 2023; Zhang and Yan 2023; Huang et al. 2024), CNNs (Wu et al. 2023; Luo and Wang 2024), GNNs (Wu et al. 2020; Cao et al. 2020; Cai et al. 2024), and Transformers (Zhou et al. 2021; Liu et al. 2024) to learn these relationships. However, empirical evidence suggests that models that do not explicitly model these dependencies can still achieve strong performance (Nie et al. 2023; Zhou et al. 2023; Xu, Zeng, and $\mathrm { X u } ~ 2 0 2 4 )$ . This may be attributed to the tendency of models that explicitly account for inter-variable dependencies to overfit complex variable interactions when the available prior knowledge is insufficient (Nie et al. 2023).

# Frequency-Domain Enhanced Time Series Forecasting Models

Recent developments in time series forecasting have increasingly leveraged frequency-domain analysis to improve predictive accuracy. This approach focuses on extracting and utilizing periodic and global dependencies that are often more discernible in the frequency domain than in the time domain. FEDformer (Zhou et al. 2022) revolutionizes forecasting by applying self-attention directly in the frequency domain, shifting the focus from time-domain dynamics to spectral characteristics. FreTS (Yi et al. 2024) optimizes predictions by utilizing frequency-domain MLPs to capture static frequency features. FITS (Xu, Zeng, and $\mathrm { X u } ~ 2 0 2 4 )$ 1 simplifies models by selectively filtering out high-frequency noise, maintaining only essential low-frequency information. Additionally, TSLANet (Eldele et al. 2024) incorporates adaptive high-frequency filtering techniques to mitigate noise. Despite these technological advancements, a prevalent limitation within these models is their primary focus on static frequency features, which neglects the dynamic changes among variables within the frequency domain.

# Preliminary

# Problem Formulation

Given a multivariate time series $X \in \mathbb { R } ^ { N \times L }$ , where $N$ denotes the number of variables and $L$ the lookback window length, the task of multivariate time series forecasting is to predict the values in the future forecasting window. Specifically, for each variable $i \in \{ 1 , . . . , N \}$ , the series is represented as $X _ { i } = [ x _ { i , 1 } , \ldots , \dot { x _ { i , L } } ] ^ { T }$ . The objective is to estimate the future values $\hat { X } \in \mathbb { R } ^ { N \times F }$ , where $F$ denotes the length of the forecasting window. The forecast values for each variable are given by $\hat { X } _ { i } = [ \hat { x } _ { i , L + 1 } , \dotsc , \hat { x } _ { i , L + F } ] ^ { T }$ for $i \in \{ 1 , \ldots , N \}$ .

# Fast Fourier Transform (FFT)

The Fast Fourier Transform (FFT)(Duhamel and Vetterli 1990) is an efficient algorithm for computing the Discrete Fourier Transform (DFT) of a sequence. Consider a univariate time series $\boldsymbol { z } _ { t } \in \mathbb { R } ^ { L }$ , where $t$ represents the time index, and $L$ is the length of the sequence. The FFT transforms this time series from the time domain into its frequency domain representation $\mathcal { Z } _ { f } \in \mathbb { C } ^ { L }$ , where $f$ represents the frequency index. The transformation is expressed as:

$$
\mathcal { Z } _ { f } = \sum _ { t = 0 } ^ { L - 1 } z _ { t } \cdot e ^ { - j \frac { 2 \pi } { L } f t } , \quad f = 0 , 1 , \ldots , L - 1
$$

where $j$ is the imaginary unit. The frequency domain representation $\mathcal { Z } _ { f }$ can be decomposed into its real and imagi

(b) For Each Variable i : Output Xˆ ∈ N F (d)Real Part Re( Σ ) i ∈D tl.l1  ∈N Frequency to Time Concat Projection   
 → ∈N D As Filters L Loetl.. ②ILale NLayer ⅹ LayerNorm  Σ∈ N D  Σ ∈ N D Xˆ ∈ N F  ∈D Imag Part Im( Σ ) Time Domain   ∈ N D   
lowR-eamoplviteude Decompose Aggregate  ∈ N D  ∈ N D (c) For Each Variable iS:tatic Filters   
(a) Cross-Variable Dynamic Global Static  ∈ K D i ∈K   
W FFT L Padding 0 D Filter Filter Top K W .i...tit...  ∈ N D ..l.... × iΩ ∈T L MwMw Time RIN Frequency TimEe tmobFerdedqinugency Jel.tll  K D i ∈D Time Domain Frequency Domain Input X ∈ N L  ∈D Decompose Aggregate

nary components, denoted as $\operatorname { R e } ( \mathcal { Z } _ { f } )$ and $\operatorname { I m } ( { \mathcal { Z } } _ { f } )$ , respectively. The magnitude (or amplitude) of $\mathcal { Z } _ { f }$ , representing the strength of the frequency components, is defined as:

$$
| \mathcal { Z } _ { f } | = \sqrt { \mathrm { R e } ( \mathcal { Z } _ { f } ) ^ { 2 } + \mathrm { I m } ( \mathcal { Z } _ { f } ) ^ { 2 } }
$$

# Method

# Overall Structure

Figure 2 illustrates the comprehensive architecture of the FilterTS model, comprised of four main components: the Time to Frequency Embedding Module (T2FEmbed), the Dynamic Cross-Variable Filtering Module (DCFilter), the Static Global Filtering Module (SGFilter), and the Frequency to Time Projection Layer (F2TOut).

We first transform the given multivariate time series $X \in$ $\mathbb { R } ^ { N \times L }$ into a frequency domain representation $\chi$ to facilitate subsequent filtering operations:

$$
\mathcal { X } = \mathrm { T } 2 \mathrm { F E m b e d } ( X )
$$

The resulting $\boldsymbol { \mathcal { X } } \in \mathbb { C } ^ { N \times D }$ represents the data in a frequency domain format optimized for neural network processing, where $\mathbb { C }$ denotes the complex domain and $D$ represents the dimension of hidden layers.

The frequency domain representation $\chi$ is then concurrently fed into two distinct pathways:

$$
\begin{array} { r } { \mathcal { O } = \mathrm { D C F i l t e r } ( \mathcal { X } ) } \\ { \mathcal { P } = \mathrm { S G F i l t e r } ( \mathcal { X } ) } \end{array}
$$

where DCFilter $\left. { \cdot } \right.$ and SGFilter $( \cdot )$ denote the Dynamic Cross-Variable Filtering Module and Static Global Filtering Module, respectively. Both $\mathcal { O }$ and $\mathcal { P }$ are frequency domain representations in $\mathbf { \dot { \mathbb { C } } } ^ { N \times D }$ , corresponding to the crossvariable and stable filtering outputs.

Subsequently, the outputs from the embedding and both filtering modules are aggregated:

$$
\mathcal { X } ^ { \Sigma } = \mathrm { L a y e r N o r m } ( \alpha \mathcal { X } + \beta \mathcal { O } + \gamma \mathcal { P } )
$$

where $\alpha , \beta$ , and $\gamma$ are weighting coefficients optimized during training. We apply LayerNorm to stabilize the training process.

This process can be repeated for $e$ layers, where equations (4) to (6) are repeated $e$ times. In each iteration, the output $\Dot { \mathcal { X } } ^ { \Sigma }$ from the previous layer serves as the input $\chi$ in equation (4) for the next layer.

Finally, the aggregated frequency domain representation $\chi ^ { \check { \Sigma } }$ is fed into the Frequency-to-Time Projection LayerF $2 \mathrm { T O u t } ( \cdot )$ , which converts the frequency domain back into the time domain, generating forecasts $\hat { X } \in \mathbb { R } ^ { N \times F }$ for $F$ future steps:

$$
\hat { X } = \Gamma 2 \mathrm { T O u t } ( \mathcal { X } ^ { \Sigma } )
$$

# Time to Frequency Embedding

The Time-to-Frequency Embedding Module converts the time-domain input series into the frequency domain, leveraging the equivalence between time-domain convolution and frequency-domain multiplication to enhance computational efficiency in subsequent filtering stages.

Firstly, we apply instance normalization to each time series $X _ { i } \ \in \mathbb { R } ^ { L }$ to mitigate the effects of varying data distributions (Kim et al. 2022):

$$
\tilde { X } _ { i } = \frac { X _ { i } - \mu ( X _ { i } ) } { \sigma ( X _ { i } ) + \epsilon }
$$

where $\mu ( X _ { i } )$ and $\sigma ( X _ { i } )$ are the mean and standard deviation of each individual time series, and $\epsilon$ is a small constant to ensure numerical stability.

Secondly, the Fast Fourier Transform (FFT) is applied to convert the normalized time series into the frequency domain efficiently:

$$
\mathcal { X } _ { i } ^ { + } = \mathrm { F F T } \left( \tilde { X } _ { i } \parallel \mathsf { z e r o s } ( L ) \right)
$$

In this formulation, $\tilde { X } _ { i }$ is zero-padded by appending $L$ zeros, increasing the sequence length to $2 L$ . This ensures that the frequency domain multiplication corresponds to linear convolution in the time domain, rather than circular convolution. The FFT then transforms this extended sequence into a complex number spectrum $\mathcal { X } _ { i } ^ { + } \in \mathbb { C } ^ { 2 L }$ . For an in-depth introduction to FFT and extended-length effects, please see the Appendix.

Finally, to match the required model dimensionality $D$ , while maintaining fidelity to the frequency domain representation, Fourier interpolation is performed:

$$
\begin{array} { r } { \mathcal { X } _ { i , j } = \left\{ \begin{array} { c l } { \mathcal { X } _ { i , j } ^ { + } } & { \mathrm { i f ~ } j < L + 1 } \\ { 0 } & { \mathrm { i f ~ } j \ge L + 1 } \end{array} \right. \quad \mathrm { f o r ~ } j = 1 , \dots , D } \end{array}
$$

Given the conjugate symmetry of the FFT for real-valued signals, only the first $L + 1$ components contain unique information. Therefore, this operation either extends the truncated frequency spectrum by zero-padding if $D$ exceeds $L + 1$ or reduces it by selecting only the first $D$ lowfrequency components.

# Dynamic Cross-Variable Filtering Module

Time Series as Filter The foundational theory of utilizing time series as finite impulse response (FIR) filters is grounded in the signal processing principle that the filtering operation can be represented as convolution in the time domain or multiplication in the frequency domain. This section discusses the transformation of a time series into an FIR filter and its practical implications.

An FIR filter is characterized by a finite sequence of $N$ coefficients $h [ m ]$ , defining the filter’s impulse response in the time domain:

$$
y [ n ] = \sum _ { m = 0 } ^ { M - 1 } h [ m ] \cdot x [ n - m ]
$$

The output signal $y [ n ]$ is determined by convolving the input signal $x [ n ]$ with the filter coefficients $h [ m ]$ , where $M$ indicates the memory of the filter.

Consider two time series, each of length $L$ : one representing the input signal $x = [ x _ { 1 } , \dots , x _ { L } ]$ and the other acting as the filter $h = [ h _ { 1 } , \ldots , h _ { L } ]$ . The filtered output signal $y$ of the FIR filter is obtained by convolving these two sequences:

$$
y _ { n } = \sum _ { m = 1 } ^ { L } h _ { m } \cdot x _ { n - m } , \quad { \mathrm { f o r } } n = 1 , \dots , L
$$

where $y _ { n }$ denotes the filtered time series at time $n$ .

The convolution theorem facilitates this operation by transforming the time-domain convolution into a frequencydomain multiplication:

$$
Y _ { f } = X _ { f } \cdot H _ { f } , \quad { \mathrm { f o r } } \ f = 1 , \dots , L
$$

where $Y _ { f } , X _ { f }$ , and $H _ { f }$ are the discrete Fourier transforms of $\mathbf { y } , \mathbf { x }$ and $\mathbf { h }$ at frequency $f$ , respectively.

This principle enables significant frequency components in $X _ { f }$ and $H _ { f }$ to produce pronounced energy in $Y _ { f }$ , thereby accentuating the frequencies common to both sequences. This characteristic is particularly advantageous for the analysis of interactions within the frequency domain.

Generation of Dynamic Cross-Variable Filters Building on the concept of utilizing time series as filters, we leverage the frequency domain representation $ { \mathcal { X } } \in \mathbb { C } ^ { N \times D }$ to generate dynamic filters. In this context, dynamic refers to the construction of filters that adapt within each look-back window, capturing key frequency components that are shared across variables.

Dynamic filters are derived by evaluating the amplitude of each frequency component. To focus on significant frequencies and reduce the influence of noise, components below a certain amplitude threshold are discarded. This threshold is dynamically set based on the amplitude distribution of each variable’s frequency components:

$$
\tau _ { i } = \mathrm { q u a n t i l e } ( \left| \mathcal { X } _ { i } \right| , \alpha )
$$

where $\tau _ { i }$ represents the $\alpha$ -quantile, defining the minimum amplitude threshold for the $i$ -th variable to consider a frequency component significant.

The dynamic filter for each frequency component $\mathcal { H } _ { i , f }$ is then defined as:

$$
\mathcal { H } _ { i , f } = \left\{ { \begin{array} { l l } { \mathcal { X } _ { i , f } } & { \mathrm { i f ~ } | \mathcal { X } _ { i , f } | > \tau _ { i } } \\ { 0 } & { \mathrm { o t h e r w i s e } } \end{array} } \right.
$$

where $\vert \mathcal { X } _ { i , f } \vert$ representing the magnitude of the $f$ -th frequency component for the $i$ -th variable. The resulting dynamic filters, which capture key frequency components of each variable’s real-time input, are combined to filter each variable’s original series to enhance shared frequency components across variables. These dynamic filters are assembled into a matrix $\mathcal { H } \ = \ \left[ \mathcal { H } _ { 1 } , \ldots , \mathcal { H } _ { N } \right]$ of dimensions CN×D.

Filtering Decomposition and Aggregation After obtaining the dynamic cross-variable filters $\mathcal { H }$ , we return to the processing of the input time series. Upon adjusting the amplitude and phase of the frequency domain representations using the scaling matrix $\mathcal { A } ^ { o ^ { \bullet } } \in \mathring { \mathbb { C } } ^ { N \times D }$ , we obtain the adjusted frequency domain representation $\mathcal { X } ^ { o }$ :

$$
\mathcal { X } ^ { o } = \mathcal { X } \odot \mathcal { A } ^ { o }
$$

For each variable $i$ and each dynamic filter $k$ , the adjusted representation is multiplied by the respective filter to obtain:

$$
\mathcal { X } _ { i , k } ^ { \mathcal { H } } = \mathcal { X } _ { i } ^ { o } \odot H _ { k } ^ { * }
$$

where $\odot$ denotes the element-wise multiplication. $H _ { k } ^ { * }$ is the conjugate of the $k$ -th dynamic filter. Here, $\mathcal { X } _ { i , k } ^ { \mathcal { H } } \in \mathbb { C } ^ { D }$ represents the subsequence generated by filtering the time series $i$ using the dynamic filter derived from time series $k$ .

Next, aggregation of these filtered subsequences is performed using a complex weight matrix $\mathcal { W } \in \overline { { \mathbb { C } ^ { N \times N } } }$ , which is processed through modified softmax and ReLU functions to form a sparse matrix:

$$
\mathcal { W } ^ { * } = \operatorname { s o f t m a x } ( \operatorname { R e L U } ( \mathcal { W } ) )
$$

The weighted aggregation for each variable $i$ is then computed as follows:

$$
\mathcal { O } _ { i } = \sum _ { k = 1 } ^ { N } \mathcal { X } _ { i , k } ^ { \mathcal { H } } \cdot W _ { i , k } ^ { * }
$$

$$
\mathcal { O } = [ \mathcal { O } _ { 1 } , \ldots , \mathcal { O } _ { N } ]
$$

Here, $\mathcal { O } \in \mathbb { C } ^ { N \times D }$ represents the output of the Dynamic Cross-Variable Filtering Module. It enhances shared frequency components across variables, allowing the model to effectively capture and utilize critical inter-variable interactions through dynamic filtering.

# Static Global Filtering Module

The Static Global Filtering Module identifies dominant stable frequency components across the entire training dataset. By constructing band-pass filters targeting the top $K$ highamplitude frequencies, it extracts key frequency representations from each input sequence during filtering.

Generation of Static Global Filters Let the training dataset $X ^ { \Omega } ~ \in ~ \mathbb { R } ^ { N \times T }$ consist of multivariate time series, where $N$ is the number of variables and $T$ is the length of each sequence. The Fourier transform is applied to obtain the global frequency domain representation $\mathbf { \bar { \mathcal { X } } } ^ { \Omega } \in \mathbb { C } ^ { N \times T }$ :

$$
\chi ^ { \Omega } = \operatorname { F F T } ( X ^ { \Omega } )
$$

Since the model’s input sequence $X \in \mathbb { R } ^ { N \times L }$ has a lookback window length $L$ , the frequency resolution of $\mathcal { X } ^ { \Omega }$ (of length $T$ ) is higher than that of the input sequence. To align the frequency resolutions, we perform down-sampling on $\mathcal { X } ^ { \Omega }$ to reduce its length from $T$ to $L$ . For each variable $i$ , the down-sampled frequency representation is obtained by summing the magnitudes of frequency components over nonoverlapping windows $\kappa$ of size $T / \dot { L }$ :

$$
\tilde { \chi } _ { i , f } ^ { \Omega } = \sum _ { m = \kappa \times f } ^ { \kappa \times ( f + 1 ) - 1 } \chi _ { i , m } ^ { \Omega }
$$

where $\tilde { \mathcal { X } } _ { i , f } ^ { \Omega }$ represents the downsampled magnitude of frequency component $f$ for variable $i$ , with $i \in \{ 1 , \ldots , N \}$ and $f \in \{ 1 , \ldots , L \}$ .

For each variable $\tilde { \mathcal { X } } _ { i } ^ { \Omega }$ , the top $K$ frequency components are selected based on their magnitudes. Let $\{ f _ { i , 1 } ^ { * } , f _ { i , 2 } ^ { * } , . . . , f _ { i , K } ^ { * } \}$ denote the indices of these top $K$ frequencies for variable $i$ . For each selected frequency $f _ { i , s } ^ { * }$ , define a band-pass filter $\mathcal { Z } _ { i , s }$ as:

$$
\mathcal { Z } _ { i , s , f ^ { \prime } } = \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f ~ } f ^ { \prime } \in [ f _ { i , s } ^ { * } - \Delta f , f _ { i , s } ^ { * } + \Delta f ] } \\ { 0 } & { \mathrm { o t h e r w i s e } } \end{array} \right.
$$

where $\Delta f$ represents the half bandwidth of the filter, accounting for minor deviations around the dominant frequency, and $f ^ { \prime } \in \{ 1 , \ldots , D \}$ to match the dimensionality of the input frequency domain representation $\boldsymbol { \mathcal { X } } _ { i } \in { \mathbb { C } } ^ { D }$ , as both the filter $\mathcal { Z } _ { i , s }$ have the same dimensions $\mathbb { C } ^ { D }$ .

These static filters of variable $i$ are assembled into a matrix $\mathcal { Z } _ { i } = [ \mathcal { Z } _ { i , 1 } , \ldots , \mathcal { Z } _ { i , K } ]$ of dimensions $\mathbb { C } ^ { K \times D }$ .

Filtering Decomposition and Aggregation The frequency domain representation of the model’s multivariate input $\mathbf { \bar { \mathcal { X } } } \in \mathbb { C } ^ { N \times D ^ { \mathbf { i } } }$ , obtained through Fourier transform, is first adjusted in amplitude and phase using an amplitude scaling matrix p CN×D:

$$
\chi ^ { p } = \chi \odot \mathcal { A } ^ { p e n d }
$$

For each variable $i$ and each static filter $s$ in $\mathcal { Z } _ { i }$ , the adjusted frequency representation is filtered by multiplying it with the corresponding static filter:

$$
\mathcal { X } _ { i , k } ^ { \mathcal { Z } } = \mathcal { X } _ { i } ^ { p } \odot \mathcal { Z } _ { i , s }
$$

where $\odot$ denotes element-wise multiplication, and $\mathcal { Z } _ { i , s }$ is the static filter $s$ for variable $i$ .

To aggregate the filtered representations, a complex weight matrix $\boldsymbol { \mathcal { V } } \in \mathbb { C } ^ { N \times K }$ is applied, which is first processed through a modified softmax and ReLU function to enforce sparsity:

$$
\mathcal { V } ^ { * } = \operatorname { s o f t m a x } ( \operatorname { R e L U } ( \mathcal { V } ) )
$$

The weighted aggregation for each variable is then computed through matrix multiplication:

$$
\mathcal { P } _ { i } = \sum _ { s = 1 } ^ { K } \mathcal { X } _ { i , k } ^ { \mathcal { Z } } \cdot V _ { i , s } ^ { * }
$$

Finally, the static filtered sequences for all variables are assembled into the output matrix $\mathcal { P } = [ \mathcal { P } _ { 1 } , \ldots , \mathcal { P } _ { N } ]$ as the static global filtering module’s output. This sequence of operations enhances static frequency components within the data, effectively leveraging the inherent static frequencies of the sequences for robust feature extraction.

# Frequency to Time Projection

Finally to transform the frequency domain representation $\chi ^ { \Sigma }$ back into the time domain, we first apply a linear transformation to handle the real and imaginary components of the frequency domain data:

$$
\operatorname { R e } ( \hat { \mathcal { X } } ) = \operatorname { R e } ( \mathcal { X } ^ { \Sigma } ) \cdot \mathbf { U } ^ { \mathrm { R e } } - \operatorname { I m } ( \mathcal { X } ^ { \Sigma } ) \cdot \mathbf { U } ^ { \mathrm { I m } }
$$

$$
\operatorname { I m } ( \hat { \mathcal { X } } ) = \operatorname { R e } ( \mathcal { X } ^ { \Sigma } ) \cdot \mathbf { U } ^ { \mathrm { I m } } + \operatorname { I m } ( \mathcal { X } ^ { \Sigma } ) \cdot \mathbf { U } ^ { \mathrm { R e } }
$$

Here, $\mathbf { U } ^ { \mathrm { R e } }$ and $\mathbf { U } ^ { \mathrm { I m } }$ represent the weight matrices for the real and imaginary components, respectively, in the space $\mathbb { R } ^ { D \times D }$ . This dual transformation preserves the properties of the complex numbers, such as phase relationships and amplitude variations. Subsequently, the real and imaginary parts are concatenated and then linearly transformed to map the combined complex data into the predicted time series output:

$$
\begin{array} { l } { { \displaystyle { X ^ { \parallel } = [ \mathbf { R e } ( \hat { \mathcal { X } } ) \mid \mid \mathbf { I m } ( \hat { \mathcal { X } } ) ] } } } \\ { { \displaystyle ~ \hat { X } = X ^ { \parallel } \cdot { \bf Q } } } \end{array}
$$

where $\mathbf { Q }$ is the final linear transformation matrix with dimensions $\mathbb { R } ^ { 2 D \times F }$ , and $\hat { X }$ is the final predicted time series in D×F .

Table 1: Results of the multivariate long-term time series forecasting task, evaluated using MSE and MAE (lower is better). The lookback window for all models was set to 96. The best results are highlighted in bold, while the second-best results are underlined.   

<html><body><table><tr><td colspan="2">Models</td><td colspan="2">Ours</td><td colspan="2">TimeMixer</td><td colspan="2">iTransformer</td><td colspan="2">PatchTST</td><td colspan="2">Crossformer</td><td colspan="2">TimesNet</td><td colspan="2">FreTS</td></tr><tr><td>Metric</td><td>96</td><td>MSE</td><td>MAE 0.360</td><td>MSE 0.333</td><td>MAE 0.371</td><td>MSE 0.334</td><td>MAE 0.368</td><td>MSE 0.329</td><td>MAE 0.367</td><td>MSE 0.404</td><td>MAE 0.426</td><td>MSE 0.338</td><td>MAE 0.375</td><td>MSE MAE 0.335 0.371</td></tr><tr><td></td><td>192 720 Avg 96</td><td>0.321 0.363 0.462 0.385 0.172</td><td>0.382 0.438 0.396 0.255</td><td>0.367 0.460 0.389 0.174</td><td>0.386 0.445 0.403 0.258</td><td>0.377 0.491 0.407 0.180</td><td>0.391 0.459 0.410 0.264</td><td>0.399 0.454 0.387 0.175</td><td>0.385 0.439 0.400 0.259</td><td>0.450 0.666 0.513 0.287</td><td>0.451 0.589 0.495 0.366</td><td>0.374 0.478 0.400 0.187</td><td>0.387 0.377 0.450 0.483 0.406 0.402 0.267 0.181</td><td></td><td>0.398 0.461 0.411 0.269</td></tr><tr><td></td><td>192 336 720 Avg 96</td><td>0.237 0.299 0.397 0.276 0.374</td><td>0.299 0.398 0.394 0.321 0.391</td><td>0.239 0.296 0.393 0.276 0.384</td><td>0.302 0.340 0.397 0.324 0.398</td><td>0.250 0.311 0.412 0.288 0.386</td><td>0.309 0.348 0.407 0.332 0.405</td><td>0.241 0.305 0.402 0.281 0.414</td><td>0.302 0.343 0.400 0.326 0.419</td><td>0.414 0.597 1.730 0.757 0.423</td><td>0.492 0.542 1.042 0.611 0.448</td><td>0.249 0.321 0.408 0.291 0.384</td><td>0.309 0.351 0.403 0.333 0.402</td><td>0.249 0.340 0.449 0.305 0.390</td><td>0.322 0.382 0.455 0.357 0.404</td></tr><tr><td></td><td>192 720 Avg 96</td><td>0.424 0.470 0.433 0.290</td><td>0.441 0.466 0.430 0.338</td><td>0.439 0.502 0.453 0.293</td><td>0.455 0.482 0.441 0.344</td><td>0.441 0.503 0.454 0.297</td><td>0.436 0.491 0.448 0.349</td><td>0.461 0.500 0.469 0.302</td><td>0.4465 0.488 0.455 0.348</td><td>0.471 0.653 0.529 0.745</td><td>0.44 0.621 0.522 0.584</td><td>0.496 0.521 0.458 0.340</td><td>0.429 0.500 0.450 0.374</td><td>0.448 0.559 0.475 0.317</td><td>0.439 0.535 0.462 0.373</td></tr><tr><td>TLLE</td><td>192 720 Avg</td><td>0.374 0.418 0.372</td><td>0.390 0.437 0.396</td><td>0.376 0.440 0.381 0.157</td><td>0.396 0.452 0.406 0.250</td><td>0.380 0.427 0.383 0.148</td><td>0.402 0.445 0.407 0.240</td><td>0.4288 0.431 0.387 0.181</td><td>0.400 0.446 0.407 0.270</td><td>1.877 1.104 0.942</td><td>0.656 0.763 0.684</td><td>0.402 0.462 0.414</td><td>0.414 0.468 0.427</td><td>0.427 0.684 0.489</td><td>0.442 0.591 0.478</td></tr><tr><td></td><td>9 336 720 Avg</td><td>0.1613 0.180 0.224 0.180</td><td>0.246 0.274 0.311 0.271</td><td>0.189 0.233 0.189</td><td>0.282 0.316 0.278</td><td>0.178 0.225 0.178</td><td>0.269 0.317 0.270</td><td>0.204 0.246 0.205</td><td>0.293 0.324 0.290</td><td>0.219 0.246 0.280 0.244</td><td>0.314 0.337 0.363 0.334</td><td>0.168 0.198 0.220 0.193</td><td>0.272 0.300 0.320 0.295</td><td>0.177 0.198 0.234 0.255</td><td>0.264 0.288 0.322 0.300</td></tr><tr><td></td><td>92 336 720 Avg</td><td>0.081 0.321 0.837 0.352</td><td>0.19 0.409 0.688 0.397</td><td>0.187 0.333 0.912 0.378 0.485</td><td>0.205 0.417 0.719 0.410 0.323</td><td>0.177 0.331 0.847 0.360 0.395</td><td>0.206 0.417 0.691 0.403 0.268</td><td>0.178 0.301 0.901 0.367 0.462</td><td>0.205 0.397 0.714 0.404 0.295</td><td>0.256 1.268 1.767 0.940 0.522</td><td>0.367 0.883 1.068 0.707 0.290</td><td>0.107 0.367 0.964 0.416 0.593</td><td>0.234 0.448 0.746 0.443</td><td>0.085 0.471 0.858 0.399 0.512</td><td>0.312 0.508 0.695 0.432 0.328</td></tr><tr><td></td><td>96 192 720 96 Avg</td><td>0.448 0.452 0.508 0.471 0.162</td><td>0.309 0.307 0.332 0.315 0.207</td><td>0.48 0.549 0.507 0.166</td><td>0.322 0.335 0.325 0.213</td><td>0.417 0.467 0.428 0.174</td><td>0.276 0.302 0.282 0.214</td><td>0.466 0.514 0.481 0.177</td><td>0.296 0.322 0.304 0.218</td><td>0.530 0.589 0.550 0.158</td><td>0.293 0.328 0.304 0.230</td><td>0.617 0.640 0.620</td><td>0.321 0.336 0.350 0.336</td><td>0.527 0.562 0.526 0.182</td><td>0.325 0.348 0.333 0.236</td></tr><tr><td></td><td>192 336 720 Avg</td><td>0.209 0.263 0.344 0.244</td><td>0.252 0.292 0.344 0.274</td><td>0.209 0.264 0.342 0.245</td><td>0.251 0.293 0.343 0.275</td><td>0.221 0.278 0.358 0.258</td><td>0.254 0.296 0.349 0.278</td><td>0.225 0.278 0.354 0.259</td><td>0.259 0.297 0.348 0.281</td><td>0.206 0.272 0.398 0.259</td><td>0.277 0.335 0.418 0.315</td><td>0.172 0.219 0.280 0.365 0.259</td><td>0.220 0.261 0.306 0.359 0.287</td><td>0.219 0.270 0.348 0.255</td><td>0.217 0.313 0.380 0.300</td></tr><tr><td colspan="2">1st Count</td><td>23</td><td>28</td><td>5</td><td>2</td><td>一 9</td><td>9</td><td>2</td><td>1</td><td>2</td><td>0</td><td>1</td><td>0</td><td>0 0</td></tr> colspan="15"></td></tr></table></body></html>

# Experiments

# Experimental Details

Datasets Following (Zhou et al. 2021) and (Nie et al. 2023), we evaluate our proposed model on eight widelyrecognized multivariate time series forecasting datasets, spanning diverse domains such as energy, economics, transportation, and climate. Specifically, these datasets include the four ETT datasets (ETTh1, ETTh2, ETTm1, ETTm2), Electricity, Exchange Rate, Traffic, and Weather.

Baselines We select several state-of-the-art multivariate time series forecasting models as baselines, including

TimeMixer (Wang et al. 2024), iTransformer (Liu et al. 2024), PatchTST (Nie et al. 2023), Crossformer (Zhang and Yan 2023), and TimesNet (Wu et al. 2023). Additionally, we incorporate FreTS (Yi et al. 2024), a novel model built upon frequency-domain.

Setup All experiments were conducted on an NVIDIA GeForce RTX 4090 24GB GPU. We adopted a consistent experimental setup identical to that of iTransformer (Liu et al. 2024) to ensure a fair comparison. Specifically, the lookback length for all models was fixed at 96, with prediction lengths set to $F \in \{ 9 6 , 1 9 2 , 3 3 6 , 7 2 0 \}$ , Mean Squared Error (MSE)

was used as the loss function.

For detailed information on datasets, model descriptions, hyperparameter settings, and other experimental implementation specifics, please refer to the Appendix.

# Main Results

Table 1 presents the predictive performance of FilterTS across eight multivariate long-term time series forecasting datasets, demonstrating superior accuracy over current state-of-the-art models in most cases. Specifically, averaging MSE across all prediction lengths, FilterTS achieved the best performance on 6 out of the 8 datasets and secured the second-best performance on the remaining 2 datasets.

The slightly weaker performance of FilterTS on the Traffic and ECL datasets can be attributed to the high number of variables in these datasets, which presents challenges for the static weight matrix in capturing complex inter-variable relationships, particularly under nonlinear conditions. In contrast, models like iTransformer, which utilize a sophisticated attention mechanism, dynamically adjust weights to better handle complex, multivariate interactions.

Despite these challenges, FilterTS reduced the average MSE by $4 . 2 4 \%$ compared to PatchTST, a representative of channel-independent models, indicating that effectively leveraging inter-variable information can significantly enhance model performance. Additionally, FilterTS outperformed models that capture inter-variable relationships using MLP (TimeMixer) and attention mechanisms (iTransformer), reducing average MSE by $3 . 6 9 \%$ and $2 . 0 2 \%$ , respectively. This suggests that the method of extracting and integrating shared frequency components across variables through filtering mechanisms is more effective than simpler fusion strategies. Furthermore, FilterTS consistently led FreTS across all tasks, underscoring that its performance benefits derive not solely from its construction in the frequency domain, but from its effective filtering strategies.

# Model Analysis

Ablation Study To validate the effectiveness of the FilterTS module design, we conducted ablation studies on three datasets. Specifically, we evaluated the following variants: w/o-SGF: The Static Global Filtering Module was removed. w/o-DCF: The Dynamic Cross-Variable Filtering Module was removed. w/o-SGF&DCF: Both modules removed. reMLP: The Dynamic Cross-Variable Filtering Module was replaced with a simple MLP to capture inter-variable relationships. re-Attn: The Dynamic Cross-Variable Filtering Module was replaced with an attention mechanism to capture inter-variable relationships.

Table 2 shows the results of the ablation study. We observe that removing the Static Global Filtering Module results in performance degradation of $5 \%$ , highlighting the importance of capturing dominant stable frequency components within the time series for long-term forecasting. Similarly, the removal of the Dynamic Cross-Variable Filtering Module leads to a performance drop of $2 \%$ , indicating the benefits of effectively leveraging inter-variable relationships.

Furthermore, replacing the Dynamic Cross-Variable Filtering Module with either an MLP or an attention mechanism resulted in inferior performance. This suggests that selectively extracting shared frequency components across variables, rather than simply fusing inter-variable information, is a more effective approach for modeling inter-variable relationships in multivariate time series forecasting.

Table 2: Ablation analysis of the FilterTS model, averaged across all prediction lengths for each dataset.   

<html><body><table><tr><td rowspan="2">Dataset</td><td colspan="2">ETTm1</td><td colspan="2">Weather</td><td colspan="2">Electricity</td></tr><tr><td>Metric</td><td>MSE MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>FilterTS</td><td>0.385</td><td>0.396</td><td>0.244</td><td>0.274</td><td>0.180</td><td>0.271</td></tr><tr><td>W/o-SGF</td><td>0.392</td><td>0.399</td><td>0.250</td><td>0.276</td><td>0.197</td><td>0.282</td></tr><tr><td>W/o-DCF</td><td>0.390</td><td>0.400</td><td>0.250</td><td>0.278</td><td>0.184</td><td>0.274</td></tr><tr><td>w/o-SGF&DCF</td><td>0.405</td><td>0.407</td><td>0.267</td><td>0.286</td><td>0.208</td><td>0.286</td></tr><tr><td>re-MLP</td><td>0.387</td><td>0.397</td><td>0.248</td><td>0.274</td><td>0.204</td><td>0.297</td></tr><tr><td>re-Attn</td><td>0.392</td><td>0.398</td><td>0.254</td><td>0.281</td><td>0.181</td><td>0.273</td></tr></table></body></html>

![](images/fb30355f0e901dc159fd58b980f648d5cf2ab0e121b0d367bcb08760a6c90ed1.jpg)  
Figure 3: Performance Analysis of FilterTS: Assessing MSE, Training Time, and Memory Usage, evaluated on the Weather Dataset with a $9 6 { \mathrm { - I n } } / 3 3 6 .$ -Out Setup.

# Model Efficiency

We compare the FilterTS model against other state-of-theart models in terms of forecasting accuracy, memory usage, and training speed. The results, as shown in Figure 3, indicate that FilterTS outperforms the comparison models in predictive performance, achieves lower memory consumption, and faster training speed.

# Conclusion

This paper introduced FilterTS, a novel multivariate time series forecasting model that enhances prediction accuracy and computational efficiency through comprehensive filtering techniques in the frequency domain. By incorporating both Static Global and Dynamic Cross-Variable Filtering Modules, FilterTS effectively captures essential frequency components. Our extensive experiments on eight real-world datasets demonstrate that FilterTS outperforms existing state-of-the-art methods in forecasting accuracy and efficiency. These results underscore the benefits of targeted frequency analysis in time series forecasting, suggesting that refining frequency components can significantly advance predictive capabilities.

# Acknowledgments

This work was supported in part by the Natural Science Foundation of Tianjin of China under Grant No. 21JCZDJC00740.