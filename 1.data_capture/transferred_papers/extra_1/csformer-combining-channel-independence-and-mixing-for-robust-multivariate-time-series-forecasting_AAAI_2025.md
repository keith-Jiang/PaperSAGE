# CSformer: Combining Channel Independence and Mixing for Robust Multivariate Time Series Forecasting

Haoxin Wang1, Yipeng $\mathbf { M } \mathbf { o } ^ { 1 }$ , Kunlan Xiang2, Nan $\mathbf { Y _ { i n } } ^ { 1 }$ , Honghe Dai1 Bixiong $\mathbf { L i } ^ { 3 }$ , Songhai $\mathbf { F a n } ^ { 4 }$ , Site $\mathbf { M _ { 0 } } ^ { 1 * }$

1College of Electrical Engineering, Sichuan University   
2University of Electronic Science and Technology of China   
3College of Architecture and Environment, Sichuan University 4State Grid Sichuan Electric Power Research Institute whx1122, moyipeng @stu.scu.edu.cn, mosite $@ 12 6 . \mathrm { c o m }$

# Abstract

In the domain of multivariate time series analysis, the concept of channel independence has been increasingly adopted, demonstrating excellent performance due to its ability to eliminate noise and the influence of irrelevant variables. However, such a concept often simplifies the complex interactions among channels, potentially leading to information loss. To address this challenge, we propose a strategy of channel independence followed by mixing. Based on this strategy, we introduce CSformer, a novel framework featuring a twostage multiheaded self-attention mechanism. This mechanism is designed to extract and integrate both Channel-specific and Sequence-specific information. Distinctively, CSformer employs parameter sharing to enhance the cooperative effects between these two types of information. Moreover, our framework effectively incorporates sequence and channel adapters, significantly improving the model’s ability to identify important information across various dimensions. Extensive experiments on several real-world datasets demonstrate that CSformer achieves state-of-the-art results in terms of overall performance.

# Introduction

Time series forecasting is vital in areas such as traffic management (Cirstea et al. 2022; Yin and Shang 2016; Qin et al. 2023), power systems (Stefenon et al. 2023; Wang et al. 2023; Mo et al. 2024), and healthcare (Ahmed, Lin, and Srivastava 2023; Alshanbari et al. 2023; Sen 2022). However, it faces significant challenges due to the complex long-term dependencies and variable interrelations inherent in time series data. These difficulties have propelled multivariate time series forecasting (MTSF) to the forefront of research in these domains.

Recently, many deep learning models have been applied to MTSF, especially Transformer-based models, such as Informer (Li, Hui, and Zhang 2021), Autoformer (Wu et al. 2021), and Fedformer (Zhou et al. 2022). These models have demonstrated improved predictive performance by refining the attention mechanism. However, recent research has raised questions regarding the suitability of Transformer models for MTSF task, proposing a straightforward linear model, DLinear (Zeng et al. 2023), that surpasses traditional Transformer models. Given the Transformer’s success in other domains (NLP, CV, etc.) (Devlin et al. 2018; Brown et al. 2020; Dosovitskiy et al. 2021), researchers have shifted their focus from modifying the attention structure within Transformers to altering the input data. iTransformer (Liu et al. 2023) treats sequences from each channel as a single token, embedding them along the sequence dimension and applying attention across the channel dimension. PatchTST (Nie et al. 2023) uses a channel independent approach. It divides the input data into patches before feeding it into the Transformer and then embedding each patch as a token. These approaches have reinstated Transformer methods to a prominent position. We summarize the Transformerbased model based on the channel processing approach. As shown in Figure 1, traditional models like Informer and Autoformer (Li, Hui, and Zhang 2021; Wu et al. 2021) directly embed variables, leading to channel mixing and potential noise introduction, especially from varied sensors and distributions. To eliminate these defects, PatchTST (Nie et al. 2023) adopts a channel-independent approach, risking information loss due to reduced physical intuitiveness. These observations suggest that neither direct channel independence nor mixing is optimal. To address this, hybrid methods like iTransformer (Liu et al. 2023) combine sequence embedding with channel-wise attention, offering a balanced approach. However, embedding methods of iTransformer may lead to erasure of inter-sequence correlations and treat the extraction of sequence information and channel information as separate non-interactive processes.

![](images/77447789cbda1709d073794a97546562489a37a89389d3de27d93eef8be62d49.jpg)  
Figure 1: Transformer-based models categorized by channel independence and channel mixing.

![](images/f53de7e6c0b585db389ac62586825f00659f547846fafcba75238ac40619889e.jpg)  
Figure 2: Structural Comparison of Transformer (top left), iTransformer (top right), and proposed CSformer (bottom): In the illustration, we compare the architectures of Transformer (top left) and iTransformer (top right) with the proposed CSformer (bottom). While Transformer and iTransformer employ attention mechanisms separately in the sequence and channel dimensions, CSformer diverges by embedding sequences into a high-dimensional space. Consequently, CSformer performs attention independently in both channel and sequence dimensions.

Drawing on the aforementioned motivation, we believe an effective combination of channel independence and channel mixing is crucial for mining more robust features in multivariate time-series data. In this paper, We propose CSformer, a model adept at extracting and combining sequence and channel information without modifying Transformer’s attention mechanism. As depicted in Figure 2, our approach introduces a dimensionality-augmented embedding technique that enhances the sequence dimensionality while preserving the original data’s integrity. We then use a unified multiheaded self-attention (MSA) mechanism to apply different attentions on sequence and channel dimensions respectively, employing a shared-parameter approach to realize the interplay of information from different dimensions. After each MSA mechanism, we include an adapter to ensure that different features are extracted by the two-stage self-attention mechanism. The primary contributions of our work are outlined as follows:

• To enhance the extraction capabilities of channel and sequence information, we propose CSformer, a two-stage attention Transformer model. This model efficiently captures the sequential and channel information of multivariate time series while minimizing the increase in model parameters, enabling effective information interaction.

• CSformer achieves state-of-the-art (SOTA) performance on diverse real-world datasets, covering domains such as electricity and weather. Extensive ablation studies further validate the model’s design rationality. • We propose a new training strategy for MTSF: channel independence followed by channel mixing, providing a new insight for future research.

# Related Work

Time series forecasting models are broadly classified into statistical and deep learning models. Statistical models, such as ARIMA (Bartholomew 1971) and Exponential Smoothing (Hyndman et al. 2008), are noted for their simplicity and efficacy in capturing temporal dynamics, but mainly focus on univariate forecasting, neglecting additional covariates. Conversely, deep learning models, including RNNbased (Medsker and Jain 2001; Graves and Graves 2012; Dey and Salem 2017; Mo et al. 2023), CNN-based (Bai, Kolter, and Koltun 2018; Wang et al. 2022; Wu et al. 2023), and Transformer-based (Li, Hui, and Zhang 2021; Wu et al. 2021; Zhou et al. 2022), have been prominent in MTSF for their comprehensive capabilities. They aim to improve prediction accuracy by integrating temporal and interchannel information. Recent studies, however, indicate focusing solely on sequence information may be more effective. Therefore, deep learning models in MTSF can be categorized as channel-independent or channel-mixing based on their handling of inter-channel information.

Channel-independent Models The channel-independent models process each time series channel (variable) independently, disregarding any interdependencies or interactions among different channels. Intuitively, this simplistic approach may not yield optimal results in certain contexts due to its failure to consider the potentially complex interrelationships within time series data. Despite this, significant performance enhancements have been observed with such models. DLinear (Zeng et al. 2023), a proponent of channel-independent architectures, employs a singular linear model along the sequence dimension and surpasses the performance of all contemporary state-of-the-art Transformer-based models. PatchTST (Nie et al. 2023), another exemplar of channel-independent models, initially divides a single time series into segments, treats each segment as a token, and then processes these segments through a transformer model to extract temporal information.

Channel-mixing Models The channel-mixing models are characterized by approaches in which the analysis of one time series channel is influenced by other channels. These models integrate interdependencies among multiple time series variables, facilitating richer and more accurate representations of data patterns and trends. Previous research (Li et al. 2019; Li, Hui, and Zhang 2021; Wu et al. 2021; Zhou et al. 2022) mainly concentrated on reducing the computational complexity of the attention mechanism but addressed inter-channel relationships only through basic embedding operations. This approach may not fully capture mutual information, and at times, it might inadvertently introduce noise. To address this limitation, Crossformer (Zhang and Yan 2023) introduces a two-stage attention layer that leverages both cross-dimensional and cross-temporal dependencies. Similarly, TSMixer (Chen et al. 2023) applies linear layers alternately across time and channel dimensions, thus enabling the extraction of temporal and cross-variate information. In a more recent development, iTransformer (Liu et al. 2023) conducts embedding operations along the time dimension and then utilizes the attention mechanism to extract information between variables, leading to notable performance enhancements.

![](images/51ccd67a75e46a6d0e01879c637890a17f3d6a0b8bcdada54a1c16d320a13964.jpg)  
Figure 3: We present the overall framework of CSformer (d). Initially, the input sequence undergoes a dimensional expansion operation before embedding (a). This dimensional transformation allows the standard MSA (b) to be adapted separately for channels and sequences (c). Note that Channel-MSA and Sequence-MSA share weights but are applied to different input dimensions.

As highlighted in PatchTST (Nie et al. 2023), improperly handling cross-dependency extraction can lead to the introduction of noise. Our objective with CSformer is to devise a method that effectively utilizes cross-variable information while concurrently adequately extracting temporal data. Additionally, considering that advancements in the attention mechanism have not substantially enhanced efficiency in practical applications (Zeng et al. 2023), this paper maintains adherence to the vanilla attention mechanism architecture.

# CSformer

In this section, we present our novel model, the CSformer, which is capable of concurrently learning channel and sequence information. Figure 3 illustrates the overall architecture of CSformer. We first enhance multivariate time series data using dimension-augmented embedding. Then, we apply a two-stage MSA mechanism to extract channel and sequence features, sharing parameters to facilitate interaction between dimensions. Finally, we introduce channel/ sequence adapters for each output to strengthen their capability to learn information across different dimensions.

# Preliminaries

In the context of multivariate time series forecasting, the historical sequence of inputs is denoted as $\mathrm { ~ \bf ~ X ~ } =$ $\{ \mathbf { x } _ { 1 } , \mathbf { \eta } _ { \cdot } \mathbf { \cdot } \mathbf { \cdot } , \mathbf { x } _ { L } \} \ \in \ \dot { \mathbb { R } } ^ { N \times L }$ , where $N$ represents the number of variables (channels) and $L$ represents the length of the historical sequence. Let $\mathbf { x } _ { i } \in \mathbb { R } ^ { N }$ represent the values of the various variables at the $i$ -th time point. Let $\mathbf { X } ^ { ( k ) } \in \mathbb { R } ^ { L }$ denote the sequence of the $k$ -th variable. Assuming our prediction horizon is $T$ , the predicted result is denoted as $\hat { \mathbf { X } } = \{ \mathbf { x } _ { L + 1 } , \dotsc , \mathbf { x } _ { L + T } \} \in \mathbb { R } ^ { N \times T }$ . Considering a model $\mathbf { f } _ { \theta }$ , where $\theta$ represents the model parameters, the overall process of multivariate time series forecasting can be abstracted as $\mathbf { f } _ { \theta } ( \mathbf { X } ) \to { \hat { \mathbf { X } } }$ .

# Reversible Instance Normalization

In practical scenarios, time-series data distributions often shift due to external factors, leading to data distribution bias. To mitigate this, Reversible Instance Normalization (ReVIN) (Kim et al. 2021) has been introduced. As delineated in Equation 1, ReVIN normalizes each input variable’s sequence, mapping it to an alternative distribution. This process artificially minimizes distributional disparities among univariate variables in the original data. Subsequently, the original mean, variance, and learned parameters are reinstated in the output predictions.

$$
\operatorname { R e V I N } ( \mathbf { X } ) = \left\{ \gamma _ { k } { \frac { \mathbf { X } ^ { ( k ) } - \operatorname { M e a n } ( \mathbf { X } ^ { ( k ) } ) } { \sqrt { \operatorname { V a r } ( \mathbf { X } ^ { ( k ) } ) + \varepsilon } } } + \left. \quad \beta _ { k } \right| k = 1 , \cdots , N \right\} ,
$$

where $\varepsilon$ refers to an infinitesimal quantity that prevents the denominator from being 0. $\beta \in \dot { \mathbb { R } } ^ { N }$ is the parameter for performing the normalization and $\boldsymbol { \gamma } \in \mathbb { R } ^ { N }$ is the learnable parameter for performing the affine transformation.

# Dimension-augmented Embedding

PatchTST (Nie et al. 2023) segments input sequences into temporal blocks, treating each as a Transformer token. TimesNet (Wu et al. 2023) uses a fast Fourier transform (FFT) to discern sequence cycles, reshaping them into 2D vectors for CNN processing. While these methods increase sequence dimensionality, they risk information loss. To address this issue, we introduce a direct sequence embedding approach, which is inspired by word embedding techniques (Mikolov et al. 2013; Yi et al. 2023). To maintain input data integrity, we first apply a dimension-augmentation operation: $\mathbf { \bar { X } } \ \in \ \mathbb { R } ^ { N \times L } \ \overset { \circ \cdot \mathbf { \bar { \Sigma } } } {  } \ \mathbf { \bar { X } } \ \in \ \mathbb { R } ^ { N \times L \times 1 }$ . Then, we multiply the augmented sequence element-wise with a learnable vector ν R1×D, producing the embedded output $\mathbf { H } \in \mathbb { R } ^ { N \times L \times D } = \mathbf { X } \times \boldsymbol { \nu }$ . This method facilitates dimensionality enhancement and embedding without distorting the original input’s intrinsic information, setting the stage for the subsequent two-stage MSA detailed later.

# Two-stage MSA

The CSformer consists of $M$ blocks. Due to the dynamic weighting characteristics of the self-attention mechanism, each block employs a two-stage MSA using shared parameters. A brief exposition of this characteristic is as follows: with input data $\mathbf { \bar { H } } \in \mathbb { R } ^ { S \times D }$ , where $S$ represents $N$ or $L$ , we define $\mathbf { Q } , \mathbf { K } \in \mathbb { R } ^ { S \times D _ { k } }$ as query and key. The attention score $\mathbf { A } \in \mathbb { R } ^ { S \times S }$ can be derived as Softmax $( \mathbf { Q K } ^ { \top } / \sqrt { D _ { k } } )$ , where $\mathbf { Q } = \mathbf { H } \mathbf { W } _ { \mathbf { Q } }$ and $\mathbf { K } = \mathbf { H } \mathbf { W } _ { \mathbf { K } }$ , with $\mathbf { W _ { Q } }$ , ${ \bf W _ { K } } \in \mathbf { \Omega }$ $\mathbb { R } ^ { D \times D _ { k } }$ as projection matrices of query and key. By the associative law of matrix multiplication, A can be written as $\mathrm { S o f t m a x } ( { \bf H W H ^ { \top } } / \sqrt { D _ { k } } )$ , where $\mathbf { W } = \mathbf { W _ { Q } } \mathbf { W _ { K } } ^ { \top }$ . Since $\mathbf { W } , D _ { k }$ are fixed values, A dynamically adapts to changes in input data $\mathbf { H }$ . This characteristic implies that with varying dimensional inputs, the attention scores adaptively adjust, even under a shared parameter method. Following the application of MSA, an adapter is incorporated to optimize the discriminative learning of both channel and sequence information. Detailed discussions of the MSA and adapter components are outlined in subsequent sections.

Channel MSA The channel MSA in the first stage is similar to the iTransformer (Liu et al. 2023), both employing MSA along the channel dimension. However, a distinction arises in the treatment of the time series, while the iTransformer considers the entire time series as a single token, herein, we apply channel-wise attention at each time step to discern inter-channel dependencies. Let Hc ∈ RL×N×D denote the input subjected to the channel-wise MSA. The output post-attention $\mathbf { Z } _ { c } \in \mathbb { R } ^ { L \times N \times D }$ is formulated as:

$$
\begin{array} { r } { \mathbf { Z } _ { c } = \operatorname { M S A } ( \mathbf { H } _ { c } ) . } \end{array}
$$

The shared parameters between channel and sequence MSAs in CSformer pose challenges in learning across two dimensions simultaneously. Inspired by fine-tuning technique in NLP’s large-scale model adaptation (Houlsby et al. 2019; Hu et al. 2022), we integrate adapter technology (Houlsby et al. 2019). This comprises two fully connected layers interspersed with an activation function. The first layer downscales input features, followed by activation, and the second layer reverts the representation to its original dimensionality. This iterative process enables the model to capture channel representations more effectively. Thus, after channel information extraction, the model output is delineated as follows:

$$
\mathbf { A } _ { c } = \mathrm { A d a p t e r } ( \mathrm { N o r m } ( \mathbf { Z } _ { c } ) ) + \mathbf { H } _ { c } ,
$$

where $\mathbf { A } _ { c } \in \mathbb { R } ^ { L \times N \times D }$ denotes the output after channel information extraction, with Norm indicating the normalized MSA output via batch normalization. To mitigate gradient challenges, the adapter’s output is additively fused with $\mathbf { H } _ { c }$ , enhancing overall framework stability.

Sequence MSA Transformer (Vaswani et al. 2017) and iTransformer (Liu et al. 2023) models traditionally concentrate on sequence or channel dimensions. Crossformer (Zhang and Yan 2023) introduces a novel approach, applying attention first to the sequence, then the channel dimension. However, its independent two-stage mechanism may result in each MSA only being able to focus on specific aspects of information, reducing the ability to fuse information.

To address this challenge, we present a novel method: the reuse of parameters derived from MSA applied along the channel dimension for sequence modeling. Specifically, the output $\mathbf { A } _ { c } \in \mathbb { R } ^ { L \times N \times D }$ from the channel MSA undergoes a reshape operation to seamlessly transition into the input $\mathbf { H } _ { s } \in \dot { \mathbb { R } } ^ { N \times \dot { L } \times D }$ for the subsequent sequence MSA. This is designed to establish the interactions between channel and sequence representations and to extract the implicit associations between channels and sequences.

When the input ${ \bf { H } } _ { s }$ is fed into the sequence MSA, the output value $\mathbf { Z } _ { s }$ can be formulated as:

$$
\begin{array} { r } { \mathbf { Z } _ { s } = \operatorname { M S A } ( \mathbf { H } _ { s } ) . } \end{array}
$$

The MSA in this study is a shared layer, distinctively applied across different input dimensions. This efficient operation substantially enhances both sequence and channel modeling capabilities while maintaining the number of parameters.

In line with the channel MSA, a subsequent adapter is introduced post-reused MSA layer for sequence feature accommodation. The architecture of this sequence adapter is analogous to that of the channel adapter, described as follows:

$$
\mathbf { A } _ { s } = \operatorname { A d a p t e r } ( \operatorname { N o r m } ( \mathbf { Z } _ { s } ) ) + \mathbf { H } _ { s } ,
$$

where $\mathbf { A } _ { s } \in \mathbb { R } ^ { N \times L \times D }$ denotes the output values of the sequence adapter. The integration of this approach is pivotal for imparting distinctiveness to sequence and channel features. The model can attain refined differentiation by utilizing a lightweight adapter, bolstering functionality without substantially increasing complexity.

Table 1: Multivariate forecasting results. We compare extensive competitive models under different prediction lengths following the setting of iTransformer. The input sequence length is set to 96 for all baselines. Results are averaged from all prediction lengths. The best results are in bold and the second best are underlined.   

<html><body><table><tr><td>Models</td><td>CSformeriTransformer</td><td>PatchTST Crossformer</td><td>SCINet</td><td>TiDE</td><td>TimesNet</td><td>DLinear</td><td>FEDformer</td><td>Stationary</td><td></td><td>Autoformer</td><td>Informer</td></tr><tr><td>Metric</td><td>MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAEMSE MAE MSE MAE MSE MAE MSE MAEMSE MAE</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ETTm1</td><td>0.3850.400.4070.410.3870.400.5130.496041904190.4000604300704850.4104804520.410.4560.5880.170.96074</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ETTm2</td><td>02820.310.8.6.58.35..56407</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ETTh1</td><td>0.4290.4320.4540.447046904540.5290.5220.5410.5070458045004560.4520.7470.647040600.57005370496047405</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ETTh2</td><td>0.364 0.3920.3830.407</td><td>0.3870.4070.942 0.6840.9540.7230.6110.500.4140.4270.5590.5150.4370.4490.5260.5160.4500.4594.4311.729</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Electricity</td><td>0.176 0.2700.1780.270</td><td>0.2160.3040.2440.3340.2680.3650.2510.3440.1920.2950.2120.3000.2140.3270.1930.2960.2270.3380.3110.397</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Solar Energy</td><td>023002700360069254070.136.8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Weather</td><td>0.250 0.2800.258 0.2790.259 0.2810.259 0.3150.292 0.3630.271 0.3200.259 0.287</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.265 0.3170.309 0.3600.288 0.3140.338 0.3820.634 0.548</td><td></td></tr><tr><td>1st Count</td><td>6 4 0</td><td>3 1 2</td><td>0 0</td><td>0 0</td><td>0</td><td>0 0</td><td>0</td><td>0 0</td><td>0 0</td><td>0 0</td><td>0 0</td></tr></table></body></html>

# Prediction

After two-stage MSA, we reduce the data’s complexity by squeezing the sequence into a lower dimension, forming $\bar { \mathbf Z } \in \mathbb R ^ { N \times ( L \ast D ) }$ . This step smartly packs the features, keeping the important information. Next, we use a linear layer for the final prediction, getting prediction result Xˆ RN×T .

# Experiments

In this section, a wide range of experiments was meticulously conducted to evaluate the CSformer model. The experiments were designed with a detailed examination of the rationality of the model’s architecture. This thorough evaluation aimed not only to examine the model’s performance across various dimensions but also to gain deeper insights into the principles behind its design.

# Datasets

To evaluate the performance of CSformer, we employed commonly used datasets for various domains in the multivariate long-term forecasting tasks, including ETT (ETTh1, ETTh2, ETTm1, ETTm2), Weather, Electricity, and Solar Energy.

# Baseline

We have selected a multitude of advanced deep learning models that have achieved SOTA performance in multivariate time series forecasting. Our choice of Transformer-based models includes the Informer (Li, Hui, and Zhang 2021), Autoformer (Wu et al. 2021), Stationary (Liu et al. 2022b), FEDformer (Zhou et al. 2022), Crossformer (Zhang and Yan 2023), PatchTST (Nie et al. 2023), and the leading model, iTransformer (Liu et al. 2023). Notably, Crossformer (Zhang and Yan 2023) shares conceptual similarities with our proposed CSformer. Furthermore, recent advancements in MLP-based models have yielded promising predictive outcomes. In this context, we have specifically selected representative models such as DLinear (Zeng et al. 2023) and TiDE (Das et al. 2023) for our comparative evaluation. Additionally, for a more comprehensive comparison, we also selected TCN-based models, which include SCINet (Liu et al. 2022a) and TimesNet (Wu et al. 2023).

# Experimental Setting

We adhered to the experimental configurations employed in iTransformer (Liu et al. 2023). For all datasets, the lookback window length $L$ is set to 96, and the prediction length is designated as $\bar { T } \in \{ 9 6 , 1 9 2 , 3 3 6 , 7 2 0 \}$ . All experiments were conducted on an NVIDIA A40 GPU, utilizing the PyTorch (Paszke et al. 2019) framework. The mean squared error (MSE) serves as the chosen loss function, with both MSE and mean absolute error (MAE) employed as evaluation metrics to gauge the efficacy of the prediction results.

# Main Results

As shown in Table 1, CSformer achieves the best overall performance across all datasets, ranking first in 6 MSE and 4 MAE metrics. This dominance underscores its superiority. Notably, compared to the latest SOTA model, iTransformer (Liu et al. 2023), CSformer further explores Transformer potential, achieving additional gains. It also significantly outperforms Crossformer (Zhang and Yan 2023), likely due to the latter’s lack of interaction in crosssequence and cross-channel extraction. Moreover, CSformer surpasses the channel-independent PatchTST (Nie et al. 2023), emphasizing the importance of effectively capturing inter-variable dependencies.

![](images/53e9d0b1d0659921d5be45cbb642792849e2644b259d2be3475baa44865df3ad.jpg)  
Figure 4: Visualization of input-96-predict-96 results on the ETTm2 dataset.

# Ablation Study

Two-stage MSA To evaluate the importance of each MSA component in the two-stage MSA framework, we removed one MSA (including its adapter) for comparative analysis. As evidenced by Table 2, CSformer demonstrates superior overall performance. Notably, retaining only the sequence MSA also yields satisfactory results, attributable to the resulting channel independence.

Table 2: Ablation of two-stage MSA on three real-world datasets with MSE and MAE metrics.   

<html><body><table><tr><td rowspan="2">Datasets Metric</td><td>ETTh1</td><td>Weather</td><td>Electricity</td></tr><tr><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td></tr><tr><td>w/o Channel MSA</td><td>0.431 0.433</td><td>0.248 0.278</td><td>0.197 0.282</td></tr><tr><td>w/o Sequence MSA</td><td>0.443 0.432</td><td>0.259 0.272</td><td>0.200 0.286</td></tr><tr><td>CSformer</td><td>0.429 0.432</td><td>0.250 0.280</td><td>0.176 0.270</td></tr></table></body></html>

Comparing parameter sharing between channel MSA and sequence MSA with their non-shared counterparts reveals that shared parameters generally yield better prediction outcomes (see Table 3). This improvement is attributed to the interaction between channel and sequence information. Although the performance of the non-shared method does not degrade significantly compared to the performance of the parameter-sharing method, it significantly increases the number of parameters in the model, which may lead to parameter redundancy. Therefore, parameter sharing is a better choice.

Table 3: Ablation of parameter sharing on three real-world datasets with MSE and MAE metrics.   

<html><body><table><tr><td rowspan="2">Datasets Metric</td><td>ETTh1</td><td>Weather</td><td>Electricity</td></tr><tr><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td></tr><tr><td>w/o Parameter Sharing</td><td>0.440 0.438</td><td>0.256 0.282</td><td>0.189 0.280</td></tr><tr><td>Parameter Sharing</td><td>0.429 0.432</td><td>0.250 0.280</td><td>0.176 0.270</td></tr></table></body></html>

Finally, we thoroughly tested the influence of varying the order of application between sequence MSA and channel MSA on the predictive performance of our model.

As shown in Table 4, the experimental results suggests a superior efficacy when channel MSA precedes sequence MSA $( \mathbf { C }  \mathbf { S } )$ ). This ordered approach facilitates the initial establishment of inter-channel interactions and connections, thereby providing a comprehensive base of information. Such information is invaluable in laying down a relevant understanding of channel dynamics. This initial comprehension of inter-channel relationships significantly enriches the subsequent analysis of temporal evolution of these variables in sequence MSA. This insight highlights the critical nature of the order of operations in MSA applications, especially in the context of enhancing predictive accuracy in complex datasets.

Table 4: Ablation of two-stage MSA’s order on three realworld datasets with MSE and MAE metrics.   

<html><body><table><tr><td>Datasets</td><td colspan="2">ETTh1</td><td colspan="2">Weather</td></tr><tr><td>Metric</td><td>MSE MAE</td><td>MSE</td><td>MAE MSE</td><td>Electricity MAE</td></tr><tr><td>S→C</td><td>0.443 0.442</td><td>0.250</td><td>0.280 0.178</td><td>0.274</td></tr><tr><td>C→S</td><td>0.429 0.432</td><td>0.250</td><td>0.280 0.176</td><td>0.270</td></tr></table></body></html>

Adapter Then, we analyze the impact of dual adapters on model performance. Table 5 reports results after removing adapters from the architecture, revealing suboptimal performance. This decline stems from the inherent dissimilarity between sequence and variable information, whose interaction is crucial, especially for datasets with many variables like Electricity. The absence of an adapter significantly reduces predictive accuracy, highlighting its role in integrating diverse data characteristics for improved performance.

Table 5: Ablation of adapter on three real-world datasets with MSE and MAE metrics.   

<html><body><table><tr><td rowspan="2">Datasets Metric</td><td colspan="2">ETTh1 Weather</td><td colspan="2">Electricity</td></tr><tr><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td></td></tr><tr><td>w/o all Adapter</td><td>0.434 0.438</td><td>0.2520.282</td><td>0.192 0.284</td><td></td></tr><tr><td>w/o Channel Adapter</td><td>0.4360.436</td><td>0.250 0.280</td><td></td><td>0.201 0.291</td></tr><tr><td>w/o Sequence Adapter|0.429 0.434</td><td></td><td>0.250 0.281</td><td></td><td>0.192 0.284</td></tr><tr><td>CSformer</td><td>0.429 0.432</td><td>0.250 0.280</td><td></td><td>0.176 0.270</td></tr></table></body></html>

# Model Analysis

Correlation Analysis We show the results of the correlation visualization in the Weather dataset in Figure 5. Initially, the blocks exhibit exploratory, uniform attention distribution. However, as the model deepens, attention becomes more focused on specific channels and sequence dimensions, indicating a growing learning of inter-channel and temporal dependencies. Particularly in the final blocks, heightened attention concentration reveals the model’s ability to identify key features and moments. This highlights the cross-dimensional information integration facilitated by shared parameters, crucial for understanding the intrinsic representation mechanisms of deep learning models in multivariate time series analysis.

![](images/192b83552f4927b10c5d7cb7cc48099fa15e4572675f5f5377aa87ca5652b48a.jpg)  
Figure 4 shows that CSformer outperforms other models in prediction accuracy, stability, detail capture, and trend tracking, likely due to its superior integration of features across sequences and channels.   
Figure 5: A case visualization of two-stage attention.

Patch Embedding vs Dimension-augmented Embedding Dimension-augmented embedding enhances the dimensional representation of sequence data without compromising original information. This approach not only preserves data integrity but also offers processing flexibility, including the option of channel independence or mixing. Such augmentation, a vital preprocessing step, ensures data compatibility with advanced models while enriching its representation to uncover complex patterns and dependencies.

To validate the efficacy of Dimension-augmented Embedding (DE), we replaced the Patch Embedding (PE) in PatchTST (Nie et al. 2023) with Dimension-augmented Embedding and conducted comparisons across three datasets. Figure 6 compares the performance. It can be seen that our proposed embedding method is generalized and effective.

![](images/1bc9322469be30498d807e0b574b23017082e0e408fe0a7fe4dc906d8e7402f9.jpg)  
Figure 6: Performance of Patch Embedding and Dimensionaugmented Embedding.

Generalization The Two-stage MSA in CSformer enhances the integration of sequence and channel information, enabling the model to learn inherent correlations and improve generalization. To assess this, we trained the model on one dataset and tested it on another, using both samefrequency (e.g., $\mathrm { E T T h } 2  \mathrm { E T T h } 1 \rangle$ ) and different-frequency (e.g., $\exists \mathrm { T T h 1 } \to \mathrm { E T T m 1 }$ ) datasets.

The experimental results, as shown in Table 6, demonstrate that CSformer outperforms other models in various scenarios. We observed that iTransformer has the poorest generalization performance, which can be attributed to its embedding method in the sequence dimension cannot learn robust representations. When cross-domain data distribution changes, the parameters learned in the source domain become inapplicable. The poor performance of PatchTST can be attributed to its use of a channel-independent method, which fails to integrate hidden correlation between channels and sequences. Therefore, the CSformer model exhibits outstanding generalization capabilities.

Table 6: Comparison of generalization capabilities between CSformer and other Transformer-based models. “Dataset A $$ Dataset B” indicates training and validation on the training and validation sets of Dataset A, followed by testing on the test set of Dataset B.   

<html><body><table><tr><td>Dataset</td><td colspan="2">ETTh2→ETTh1</td></tr><tr><td>Horizon</td><td>96 192 336 720</td><td>96 192 336 720</td></tr><tr><td>PatchTST</td><td>0.491 0.529 0.555 0.627</td><td>0.761 0.788 0.7770.790</td></tr><tr><td>CSformer 0.4440.484</td><td>iTransformer0.880 0.920 0.924 0.919</td><td>0.9230.938 0.9380.945</td></tr></table></body></html>

Training Strategy In our earlier discussion, we highlighted the benefits of a training strategy that initially processes channels independently (CI) and then mixes them. This method prevents noise introduced by premature channel mixing (CM) and overcomes the limitations of prolonged independence impeding inter-channel information exchange. To investigate this hypothesis, we conducted an experiment with PatchTST (Nie et al. 2023), modifying the dimension of its attention mechanism. Specifically, we shifted the focus of the attention mechanism from the sequence dimension to the channel dimension to facilitate channel mixing. Table 7 indicates that a reasonable balance of channel independence and mixing enhances predictive performance in multivariate time series forecasting, significantly boosting modeling capabilities. This also provides a new perspective for future research: how to effectively combine channel independence and mixing?

Table 7: A comprehensive performance comparison between channel independence and channel mixing is presented. The results represent the average values across all four prediction lengths.   

<html><body><table><tr><td rowspan="2">Datasets Metric</td><td colspan="2">ETTh1</td><td colspan="2">Weather</td><td colspan="2">Electricity</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>CI</td><td>0.469</td><td>0.454</td><td>0.259</td><td>0.281</td><td>0.216</td><td>0.304</td></tr><tr><td>CM</td><td>0.442</td><td>0.431</td><td>0.248</td><td>0.278</td><td>0.179</td><td>0.273</td></tr><tr><td>Promotion</td><td>5.8%</td><td>5.1%</td><td>4.2%</td><td>1.1%</td><td>17.1%</td><td>10.2%</td></tr></table></body></html>

# Conclusion and Future Work

In this paper, we summarized the channel-independent and mixing models and show that channel-independent followed by channel mixing is a superior strategy. Based on this, we introduced CSformer, which is an architecture for a twostage attention mechanism. It balances channel independence and channel mixing to robustly model multivariate time series. In future work, we will investigate more rational ways of combining channel independence and channel mixing and reducing their computational demand.

# Acknowledgments

This work is supported by Science and Technology Project of State Grid Corporation of China (Research on Transmission Line Damage Detection and Safety Risk Assessment Technology for Earthquakes and Secondary Disasters, No. 521999230017).