# Efficiently Enhancing Long-term Series Forecasting via Ultra-long Lookback Windows

Suxin Tong1,2, Jingling Yuan1,2\*

1School of Computer Science and Artificial Intelligence, Wuhan University of Technology, Wuhan 430070, China 2Hubei Key Laboratory of Transportation Internet of Things, Wuhan University of Technology, Wuhan 430070, China suxin tong $@$ whut.edu.cn, yjl $@$ whut.edu.cn

# Abstract

Long-term series forecasting aims to predict future data over long horizons based on historical information. However, existing methods struggle to effectively utilize long lookback windows due to overfitting, computational resource constraints, or information extraction challenges, thereby limiting them to using limited lookback windows for predicting long-term future series. To address these issues, this paper introduces the Input Refinement and Prediction Auxiliary (IRPA) framework, a lightweight model consisting of four linear layers designed to extract key information from ultra-long lookback windows to enhance limited lookback windows and assist prediction processes. IRPA comprises an Input Refinement Module (IRM) and a Prediction Auxiliary Module (PAM), each constructed from two linear layer sub-modules. The IRM performs effective decomposition and patching of ultra-long series, refining seasonal and trend features to increase the information density in limited lookback windows and mitigate overfitting and parameter inflation. The PAM extracts historical similarities and seasonal patterns from ultra-long lookback windows to significantly improve prediction accuracy. IRPA substantially extends the utilization of lookback windows, offering a lightweight and efficient solution with broad applicability. Experimental results on eight datasets show IRPA reduces the Mean Squared Error (MSE) by an average of $1 6 . 1 \%$ for various models.

# Code — https://github.com/SuxinTong/IRPA

# Introduction

Long-term series forecasting, as a pivotal branch within the domain of time series analysis, plays an indispensable role across an array of critical fields. These include traffic flow prediction (Jiang et al. 2023), meteorological forecasting (Huang et al. 2023), and recommendation systems (Chen et al. 2024a), etc. Accurate long-term predictions not only arm policymakers with reliable data for decision-making but also facilitate long-term planning and enable early warning systems (Wu et al. 2021; Zhou et al. 2021; Cao et al. 2023). In recent years, the swift advancements in deep learning technologies have led to the emergence of various neural network-based time series forecasting models (Rasul et al.

![](images/790ef35b7bfee15b0cc675f0f43c76df4c181b0db336fc69e6f408fb15ab6f2f.jpg)  
Figure 1: Existing models show overfitting and resource limitations with increasing lookback window length, while IRPA effectively mitigates these issues in the Weather dataset at prediction length $k { = } 7 2 0$ .

2021; Nie et al. 2023; Zeng et al. 2023; Liu et al. 2024), which have demonstrated superior performance.

Despite these advancements, the effective utilization of long lookback windows in long-term series forecasting remains a significant challenge. Long lookback windows contain rich information about long-term trends, complex seasonal patterns, and rare events that can be crucial for accurate long-term predictions (Jia et al. 2024). However, leveraging this extensive historical data introduces several challenges, as shown in Figure 1:

1. Overfitting Issues: When processing long lookback windows, models often tend to overfit, capturing noise within the data and leading to unstable forecasting.

2. Computational Resource Constraints: Expanding historical lookback windows significantly increases computational costs and memory requirements, potentially exceeding practical resource limitations.

3. Information Extraction Challenges: Effectively distilling relevant information from long lookback windows poses a significant challenge. Models need to identify and leverage key patterns and trends while disregarding irrelevant or redundant information.

To address these challenges, we propose an innovative solution centered on refining relevant information from ultralong lookback windows. Our approach ensures that even limited historical windows capture richer content and leverage similar patterns within the historical data to enhance predictive accuracy. Compared to directly utilizing the entire ultra-long lookback windows, this approach effectively avoids overfitting due to excessive noise and prevents parameter proliferation. The motivation behind this approach lies in the fact that ultra-long lookback windows contain rich information about long-term trends and periodicities, while the most recent limited lookback window immediately preceding the future series often captures the most relevant temporal features and patterns for the prediction target period. If the association between these two can be effectively computed and historical information extracted to assist predictions, it can significantly enhance the accuracy and stability of forecasts.

Founded on this motivation, we craft a lightweight and universally applicable framework, the IRPA (Input Refinement and Prediction Auxiliary) framework. Comprising an Input Refinement Module (IRM) and a Prediction Auxiliary Module (PAM), this framework utilizes only four linear layers to facilitate efficient information extraction and predictive enhancement. Specifically, the IRM performs seasonaltrend decomposition and patching, leveraging the periodic and repetitive characteristics of seasonal components to guide the refinement of seasonal and trend features. This increases the information density in limited lookback windows and significantly mitigates parameter inflation. The PAM extracts historical similarities and seasonal patterns from ultralong lookback windows, providing additional reference information for the prediction process and thus boosting the accuracy of predictions.

Our contributions are summarized as follows:

• We introduce the IRPA framework, an efficient plug-andplay solution that refines ultra-long historical time series for improved long-term series forecasting. • We design an Input Refinement Module that efficiently extracts seasonal and trend-related information from ultra-long lookback windows, enhancing the information density in limited lookback windows and mitigating overfitting and parameter inflation. • We propose a Prediction Auxiliary Module that leverages historical similarities and seasonal patterns to offer extra predictive guidance, improving forecasting outcomes. • Experimental results demonstrate IRPA’s effectiveness in enhancing various time series forecasting models, reducing MSE by $9 . 7 \%$ on iTransformer, $1 5 . 0 \%$ on PatchTST, $1 4 . 4 \%$ on TimesNet, and $2 5 . 2 \%$ on DLinear.

# Related Work

# Time Series Forecasting

Time series forecasting involves predicting future values using historical data. Traditional time series forecasting methods primarily rely on statistical models (Farlie 1964; Watson 1994; Makridakis and Hibon 1997). In recent years, machine learning methods (Kan et al. 2022; Sun and Yu 2023), especially deep learning models, have demonstrated formidable performance in time series forecasting tasks. Recurrent Neural Networks (RNNs) and their variants (Hochreiter and Schmidhuber 1997; Wen et al. 2017; Baytas et al. 2017; Cao et al. 2018; Salinas et al. 2020) were among the early deep learning approaches for handling sequential data, though they struggle with capturing long-term dependencies in long sequences. Convolutional Neural Networks (CNNs) (van den Oord et al. 2016; Bai, Kolter, and Koltun 2018; Chen et al. 2020; Wang et al. 2023) have also been applied to time series analysis, and they capture long-range temporal correlations through causal and dilated convolutions. Recently, the Transformer architecture (Vaswani et al. 2017; Lim et al. 2021; Yang et al. 2022) has excelled in capturing long-term dependencies through its self-attention mechanism. Enhanced attention models (Zhou et al. 2021; Wu et al. 2021; Zhou et al. 2022) improve sequence handling but are relatively computationally intensive. Multilayer Perceptrons (MLPs), as a simple and efficient neural network structure, have demonstrated significant potential in time series forecasting. Recent research (Oreshkin et al. 2020; Challu et al. 2023; Olivares et al. 2023) has captured temporal patterns through stacked fully connected layers and residual branches, while DLinear and NLinear (Zeng et al. 2023) have exhibited remarkable performance in time series forecasting due to their lightweight and effective structures. This paper proposes a new framework based on the MLP architecture, aiming to effectively address the time series forecasting challenge with long lookback windows, leveraging the simplicity and computational efficiency of MLPs while enhancing their ability to model long-term dependencies.

# Time Series Forecasting with Long Lookback Windows

Time series forecasting with long lookback windows involves using an extended period of historical data to predict future data points. Addressing this problem is critical for capturing long-term dependencies and intricate temporal patterns. However, handling long sequence data presents challenges such as high computational complexity, substantial memory consumption, and susceptibility to overfitting. Current solutions primarily focus on two directions: one approach entails segmenting the time series into overlapping or non-overlapping patches, as done in (Nie et al. 2023; Zhang and Yan 2023; Chen et al. 2024b), which utilize selfattention mechanisms to capture long-term dependencies. The other approach involves directly leveraging the entire time series to capture long-term dependencies, with models like iTransformer (Liu et al. 2024) introducing inverted dimensions for series embeddings, and another work (Wang et al. 2024b) employing both patch-level and series-level representations to capture endogenous and exogenous variables. Notwithstanding the progress made, these solutions could still face constraints like high computational complexity and overfitting tendencies. To tackle these challenges, this paper employs seasonal-trend decomposition, patching, and similarity calculation to efficiently and effectively extract information from long lookback windows.

![](images/8c479a9df763f3def77c27d8c2a9d5c8f599411b4b26a983304bae5a9f5709e3.jpg)  
Figure 2: The overall structure of IRPA. (a) Trend Refinement Module. (b) Seasonal Refinement Module. (c) Historical Similarity Auxiliary Module. (d) Historical Seasonal Pattern Prediction Auxiliary Module.

# Methodology

# Problem Definition

Define the input historical time series data as $\begin{array} { r l } { \boldsymbol { X } } & { { } = } \end{array}$ $\{ \pmb { x } _ { 1 } , \ \pmb { x } _ { 2 } , \ \dots , \ \hat { \pmb { x } } _ { l } \} \ \in \ \mathbb { R } ^ { b \times n \times l }$ , where $b$ is the batch size, $l$ is the input lookback window length, and $n$ is the number of variables. Our objective is to refine the historical time series of length $l$ into a shorter time series of length $r$ $( 5 r \leq l )$ , denoted as x′ , x′ , . . $\{ \pmb { x } _ { 1 } ^ { ' } , \pmb { x } _ { 2 } ^ { ' } , \dots , \pmb { x } _ { r } ^ { ' } \} \in \mathbb { R } ^ { b \times n \times r }$ , and then predict the future $k$ timesteps ${ \pmb Y } = \{ { \pmb x } _ { l + 1 } , ~ { \pmb x } _ { l + 2 } , ~ . ~ . ~ . ~ , { \pmb x } _ { l + k } \} \in$ Rb×n×k based on the refined series. Formally, it can be expressed as $Y = f (  { \boldsymbol { X } } ; \theta )$ , where $\boldsymbol { Y }$ is the predicted future data, $f$ is the prediction model, and $\theta$ represents the model parameters.

# Overall Structure

Figure 2 shows the overall structure of IRPA, which consists of two key components: the Input Refinement Module (IRM) and the Prediction Auxiliary Module (PAM). The IRM is responsible for refining key features from the input data, while the PAM provides additional auxiliary information to improve prediction accuracy. Specifically, the Input Refinement Module includes seasonal and trend refinement modules; the input time series is first decomposed into trend and seasonal components, then these components are patched and input into the seasonal and trend refinement modules, which refine the components by identifying similar seasonal patterns in the long historical data. The Prediction

Auxiliary Module includes historical similarity and historical seasonal pattern prediction auxiliary modules. The prediction auxiliary module identifies similar historical patches and their subsequent seasonal information from the ultralong lookback window to provide additional reference information for the forecasting process. In the following sections, we will detail the structure and function of these two modules.

# Input Refinement Module

The Input Refinement Module focuses on extracting critical information from raw input data and refining it to enhance prediction accuracy and efficiency. To effectively extract key information, we decompose the input time series data into seasonal and trend components and refine them separately. This decomposition reveals the underlying features of the data, allowing the model to focus on these distinct aspects. Moreover, the effectiveness of seasonal-trend decomposition has been demonstrated in numerous works (Wu et al. 2021; Zeng et al. 2023; Wang et al. 2023). Therefore, decomposing the input into seasonal and trend components for separate refinement can enhance the model’s accuracy and stability. For the input time series data $\boldsymbol { X } \in \mathbb { R } ^ { b \times n \times l }$ , it first undergoes a normalization process (Liu et al. 2022b), where $X$ is subtracted by the mean and divided by the variance vector for all channels, which helps improve the model’s stability and performance. We then decompose $X$ into seasonal component $\boldsymbol { s }$ and trend component $_ { T }$ as follows:

$$
\pmb { T } = \mathrm { A v g P o o l } ( \mathrm { P a d d i n g } ( \pmb { X } ) ) , \pmb { S } = \pmb { X } - \pmb { T } ,
$$

where AvgPool denotes the moving average operation, and padding is applied to maintain a consistent series length. Subsequently, we perform the patching operation (Nie et al. 2023) on both the decomposed seasonal and trend components and the undecomposed series. The operation segments each of these components into patches of a specified length $r$ with a stride of $r / 2$ , and the resulting $m$ patches are reshaped. This patching process effectively divides the extended series into multiple shorter patches, facilitating subsequent similarity computations and information extraction. The formula for this process is articulated as:

$$
S _ { p } = \operatorname { P a t c h i n g } ( S ) , T _ { p } = \operatorname { P a t c h i n g } ( T ) ,
$$

$$
X _ { p } = \operatorname { P a t c h i n g } ( X ) ,
$$

where $S _ { p } , T _ { p } , X _ { p } \in \mathbb { R } ^ { b \cdot n \times m \times r }$ represent the patched seasonal, trend, and undecomposed series, respectively.

Seasonal Refinement Module In the Seasonal Refinement Module, we choose to identify similar seasonal components from the extensive historical data and use these to guide the refinement of both seasonal and trend components. This approach is due to the cyclical and repetitive nature of seasonal components, which typically recur at fixed intervals, making it easier to find similar seasonal patterns within historical data. In contrast, trend components are often significantly influenced by external factors such as policy changes or climatic fluctuations. Directly searching for similar trend components may introduce unnecessary noise, reducing the model’s predictive accuracy. To precisely locate similar historical patterns, we employ the Hellinger distance (Hellinger 1909) to quantify patch similarity, given its well-defined range and stability. Additionally, by sharing computational results across modules, we optimize efficiency. The formula is given by:

$$
H = \sqrt { 1 - \sum _ { i = 0 } ^ { r - 1 } \sqrt { \phi ( S _ { p } [ : , : , i ] ) \odot \phi ( S _ { p } [ : , - 1 , i ] ) } } ,
$$

where $\pmb { H } \in \mathbb { R } ^ { b \cdot n \times m }$ represents the computed seasonal similarity matrix, $\phi$ denotes the softmax activation function, and $\odot$ indicates the element-wise product. Selecting the last patch as the baseline for similarity calculation is primarily due to its proximity to the future series to be predicted, thus containing the temporal features and patterns most relevant to the target prediction period. This design not only effectively guides the refinement process of the historical series but also better captures the latest changes and trends that may impact future predictions.

We then select the most similar seasonal patch to guide the refinement of the seasonal component, leveraging the strong regularity of seasonal patterns to ensure stability and accuracy. Additionally, we incorporate the first patch as an anchor point, enabling the model to better capture the overall temporal evolution patterns from the start to the end of the series. The formula is as follows:

$S _ { r } ^ { ' } = ( \mathrm { G a t h e r } ( S _ { p } , \operatorname { a r g m a x } ( H ) ) + S _ { p } [ : , 0 , : ] ) \odot \sigma ( W _ { h } ) ,$ (4)

where $W _ { h } \in \mathbb { R } ^ { b \cdot n \times r }$ denotes a sigmoid-gated weight matrix, employed to dynamically integrate the gathered seasonal component with the last seasonal patch. Subsequently, we apply a Reshape operation followed by a linear transformation to derive the refined seasonal output:

$$
S _ { r } = \mathrm { R e s h a p e } ( S _ { r } ^ { ' } + S _ { p } [ : , - 1 , : ] \odot ( 1 - \sigma ( W _ { h } ) ) ) W _ { s } + b _ { s } ,
$$

where $S _ { r } \in \mathbb { R } ^ { b \times n \times k }$ is the refined seasonality component, $W _ { s } \in \mathbb { R } ^ { b \times r \times k }$ is the weight matrix, and ${ \pmb b } _ { s }$ is the bias. $\sigma$ denotes the sigmoid activation function.

Trend Refinement Module For trend refinement, we gather the top three most similar trend patches: this approach is adopted because trend components may be influenced by various factors, and choosing multiple similar trend patches can capture more trend variation information, reducing the potential impact of anomalies or noise from a single component. Averaging multiple similar components also minimizes the uncertainty of individual trend components. The calculation of similar trend item $\pmb { T } _ { r } ^ { ' } \in \mathbb { R } ^ { b \cdot n \times r }$ is given by:

$$
\begin{array} { r } { \pmb { T } _ { r } ^ { ' } = \mathrm { M e a n } ( \sigma ( \mathrm { G a t h e r } ( \pmb { S } _ { p } , \mathrm { t o p k } ( \pmb { H } , 3 ) ) ) ) \odot \sigma ( \pmb { W } _ { r } ) , } \end{array}
$$

Similarly to the seasonality refinement, the refined trend component is computed as:

$\begin{array} { r } { \pmb { T } _ { r } = \mathrm { R e s h a p e } ( \pmb { T } _ { r } ^ { ' } + \pmb { T } _ { p } [ : , - 1 , : ] \odot ( 1 - \sigma ( \pmb { W } _ { r } ) ) ) \pmb { W } _ { t } + \pmb { b } _ { t } , } \end{array}$ (7) where $\pmb { T } _ { r } \in \mathbb { R } ^ { b \times n \times k }$ is the refined trend component, $W _ { r } \in$ $\mathbb { R } ^ { b \cdot n \times r }$ , $W _ { t } \in \mathbb { R } ^ { b \times r \times k }$ are weight matrices, $\mathbf { \delta } _ { b _ { t } }$ is the bias.

# Prediction Auxiliary Module

The Prediction Auxiliary Module aims to provide additional information to support the forecasting process, thereby improving prediction accuracy. The module consists of two key parts: the Historical Similarity Auxiliary Module and the Historical Seasonal Pattern Prediction Auxiliary Module.

Historical Similarity Auxiliary Module We opt to use the undecomposed original historical series to support predictions rather than the decomposed seasonal or trend historical series. This choice is based on the consideration that undecomposed similar items encapsulate the combined information of seasonality and trends, providing more references for the predictive process. Certain patterns may exist in the interaction between seasonality and trends; using the original series can capture these complex relationships. In contrast, using only decomposed seasonal and trend components might overlook some critical information. The calculation of historical similar item $\boldsymbol { X } _ { s } ^ { ' } \in \mathbb { R } ^ { b \cdot n \times 4 \times r }$ is:

$$
X _ { s } ^ { ' } = \operatorname { C o n c a t } ( \operatorname { G a t h e r } ( X _ { p } , \operatorname { t o p k } ( H , 3 ) ) , X _ { p } [ : , - 1 , : ] ) .
$$

Then, we perform a Reshape operation and linear transformation on the historical similar item:

$$
X _ { s } = \operatorname { R e s h a p e } ( X _ { s } ^ { ' } ) W _ { x } + b _ { x } ,
$$

where the result of the historical similarity auxiliary $X _ { s } \in$ $\mathbb { R } ^ { b \times n \times k }$ , $W _ { x } \in \mathbb { R } ^ { b \times { 4 r \times k } }$ is the weight matrix, and ${ \pmb b } _ { x }$ represents the bias.

Historical Seasonal Pattern Prediction Auxiliary Module This module identifies the most similar seasonal patterns in historical data and uses their subsequent patterns to guide future predictions. We focus on seasonality components rather than trend components or the undecomposed original historical series because seasonality components have inherent periodicity and repeatability. This approach provides reliable and regular reference information while avoiding the introduction of uncertain trend information that could mislead predictions. To ensure that the selected similar seasonal patterns cover a sufficient subsequent time span, we define a threshold $q { = } \lceil k / r \rceil$ , where $k$ is the prediction length, ensuring that $q \cdot r \geq k$ , meaning that the selected historical patterns include at least one complete subsequent development to guide future predictions. Thus, the most similar seasonal pattern auxiliary indices $S _ { a } ^ { ' } \in \mathbb { R } ^ { b \cdot n \times q }$ are calculated as:

$$
S _ { a } ^ { ' } { = } \mathrm { a r g m a x } ( H [ : , : ( m - q ) ] ) + [ 1 , 2 , \ldots , q ] .
$$

Then, the subsequent $k$ values of the most similar seasonal pattern are utilized to provide auxiliary prediction information:

$$
S _ { a } = ( \operatorname { R e s h a p e } ( \operatorname { G a t h e r } ( S _ { p } , S _ { a } ^ { ' } ) ) [ : , : , : k ] ) \boldsymbol { W } _ { a } + \boldsymbol { b } _ { a } ,
$$

where $S _ { a } \in \mathbb { R } ^ { b \times n \times k }$ represents the auxiliary prediction information $W _ { a } \in \mathbb { R } ^ { b \times k \times k }$ is the weight matrix, and $b _ { a }$ denotes the bias.

The prediction result $Y \in \mathbb { R } ^ { b \times n \times k }$ is the sum of the outputs from the four modules, then de-normalized to obtain the final prediction result. This combined prediction method aims to leverage the diverse features and information extracted by each module, potentially producing more accurate and robust prediction results. The formula is as follows:

$$
Y = S _ { r } + { \pmb T } _ { r } + { \pmb X } _ { s } + { \pmb S } _ { a } .
$$

# Experiments

# Experimental Settings

Datasets To comprehensively evaluate the performance of the proposed IRPA framework, we have selected eight openly accessible datasets that are widely utilized in the realm of time series forecasting research, including Weather, Solar, ECL, Traffic, and four ETT datasets (Liu et al. 2024). They exhibit significant diversity in terms of temporal granularity, the number of variables, and data attributes, presenting a thorough and challenging testing environment for our model. Detailed statistics are in Table 1.

Baselines To thoroughly assess the effectiveness of the IRPA framework, we selected a suite of state-of-the-art time series forecasting models as baselines for comparison. These include Transformer-based models: iTransformer (Liu et al. 2024), PatchTST (Nie et al. 2023), Autoformer (Wu et al. 2021), and CNN-based models: TimesNet (Wu et al. 2023), SCINet (Liu et al. 2022a), as well as the linear model DLinear (Zeng et al. 2023).

Implementation Details All experiments were implemented using PyTorch 2.1.2 (Paszke et al. 2019) and conducted in a Linux environment equipped with two NVIDIA Tesla T4 16GB GPUs. We used L2 loss and the

Table 1: Statistics of the experimental datasets.   

<html><body><table><tr><td>Datasets</td><td>Features</td><td>Timesteps</td><td>Frequency</td></tr><tr><td>ETTh1&ETTh2</td><td>7</td><td>17,420</td><td>1 Hour</td></tr><tr><td>ETTm1&ETTm2</td><td>7</td><td>69,680</td><td>15 Minutes</td></tr><tr><td>Weather</td><td>21</td><td>52,696</td><td>10 Minutes</td></tr><tr><td>Solar</td><td>137</td><td>52,560</td><td>10 Minutes</td></tr><tr><td>ECL</td><td>321</td><td>26,304</td><td>1 Hour</td></tr><tr><td>Traffic</td><td>862</td><td>17,544</td><td>1 Hour</td></tr></table></body></html>

Adam (Kingma and Ba 2015) optimizer for model training, with an initial learning rate of 0.01. Each model was trained for 10 epochs with a batch size of 32. For the IRPA framework, the input series length was set to $\{ 4 8 0 , 1 9 2 0 , 3 6 0 0 \}$ based on the dataset’s time sampling frequency. The series refinement length and patch length were both fixed at 96. For all baseline models, the input series length was set to 96. The prediction series length was set to $\left\{ 9 6 , 1 9 2 , 3 3 6 , 7 2 0 \right\}$ , covering various forecasting ranges. The data partitioning and baseline model experimental setup followed the configurations in (Liu et al. 2024), and the code utilized the implementation provided by (Wang et al. 2024a). To enhance reproducibility, we fixed random seeds. MSE and Mean Absolute Error (MAE) were used as primary evaluation metrics, providing insights into both variance and magnitude of prediction errors.

# Main Results

Forecasting results Table 2 meticulously illustrates the long-term series forecasting performance of IRPA compared to six baseline models across eight datasets. The results indicate that IRPA achieved the best results in 54 out of 64 long-term forecasting tasks, significantly outperforming other models. Compared to the previous state-of-the-art method, iTransformer, IRPA reduced the average MSE by $9 . 4 \%$ . IRPA demonstrated outstanding performance across datasets with various characteristics, showcasing its strong generalization ability and adaptability. This suggests that the enhancement of input feature quality is as crucial as model improvements. As the prediction length increases, IRPA’s advantage over baseline models becomes more pronounced, particularly in 720-time-step forecasts. This is primarily due to IRPA’s series refinement mechanism, which more effectively captures historical long-term information for time series forecasting.

Framework generality To validate the generality of our framework, we applied the IRPA to baseline time series prediction models and analyzed the performance improvement. Specifically, the extended input series was refined by the Input Refinement Module before being fed into the subsequent baseline model, and the model’s prediction output was augmented by adding the Prediction Auxiliary Module. As shown in Table 3, IRPA reduced the MSE by $9 . 7 \%$ on iTransformer, $1 5 . 0 \%$ on PatchTST, $1 4 . 4 \%$ on TimesNet, and $2 5 . 2 \%$ on DLinear, surpassing previous state-of-the-art levels. Overall, IRPA demonstrates strong generality and effectiveness across different models and datasets.

<html><body><table><tr><td rowspan="2" colspan="2">Model Metric</td><td colspan="2">IRPA</td><td colspan="2">iTransformer</td><td colspan="2">PatchTST</td><td colspan="2">TimesNet</td><td colspan="2">DLinear</td><td colspan="2">SCINet</td><td colspan="2">Autoformer</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE 0.386</td><td>MAE 0.405</td><td>MSE</td><td>MAE 0.419</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td></td><td>96 192 336 720</td><td>0.359 0.391 0.418 0.454</td><td>0.386 0.408 0.428 0.464</td><td>0.441 0.487 0.503</td><td>0.436 0.458 0.491</td><td>0.414 0.460 0.501 0.500</td><td>0.445 0.466 0.488</td><td>0.384 0.436 0.491 0.521</td><td>0.402 0.429 0.469 0.500</td><td>0.386 0.437 0.481 0.519</td><td>0.400 0.432 0.459 0.516</td><td>0.654 0.719 0.778 0.836</td><td>0.599 0.631 0.659 0.699</td><td>0.449 0.500 0.521 0.514</td><td>0.459 0.482 0.496 0.512</td></tr><tr><td></td><td>96 336 720</td><td>0.370 0.345 0.393</td><td>0.3324 0.395 0.432</td><td>0.297 0.428 0.427</td><td>0.349 0.432 0.445</td><td>0.302 0.426 0.431</td><td>0.348 0.433 0.446</td><td>0.3402 0.452 0.462</td><td>0.374 0.452 0.468</td><td>0.33 0.594 0.831</td><td>0.387 0.541 0.657</td><td>0.707 1.000 1.249</td><td>0.621 0.744 0.838</td><td>0.346 0.482 0.515</td><td>0.388 0.486 0.511</td></tr><tr><td></td><td>9 336 720</td><td>0.307 0.347 0.395</td><td>0.359 0.380 0.414</td><td>0.334 0.426 0.491</td><td>0.38 0.420 0.459</td><td>0.329 0.399 0.454</td><td>0.36 0.410 0.439</td><td>0.334 0.410 0.478</td><td>0.375 0.411 0.450</td><td>0.345 0.413 0.474</td><td>0.372 0.413 0.453</td><td>0.418 0.490 0.595</td><td>0.438 0.485 0.550</td><td>0.55 0.621 0.671</td><td>0.475 0.537 0.561</td></tr><tr><td></td><td>96 336 720 96</td><td>0.162 0.264 0.317 0.156</td><td>0.255 0.330 0.368 0.214</td><td>0.180 0.311 0.412 0.174</td><td>0.264 0.348 0.407 0.214</td><td>0.175 0.305 0.402 0.177</td><td>0.252 0.343 0.400 0.218 0.259</td><td>0.187 0.321 0.408 0.172 0.219</td><td>0.267 0.351 0.403 0.220</td><td>0.193 0.369 0.554 0.196</td><td>0.292 0.427 0.522 0.255</td><td>0.286 0.637 0.960 0.221</td><td>0.377 0.591 0.735 0.306</td><td>0.255 0.339 0.433 0.266</td><td>0.339 0.372 0.432 0.336</td></tr><tr><td></td><td>192 336 720 96 192</td><td>0.235 0.285 0.196 0.216</td><td>0.253 0.287 0.326 0.278 0.292</td><td>0.221 0.278 0.358 0.203 0.233</td><td>0.254 0.296 0.349 0.237 0.261</td><td>0.225 0.278 0.354 0.234 0.267</td><td>0.297 0.348 0.286 0.310</td><td>0.280 0.365 0.250 0.296</td><td>0.261 0.306 0.359 0.292 0.318</td><td>0.237 0.283 0.345 0.290 0.320</td><td>0.296 0.335 0.381 0.378 0.398</td><td>0.261 0.309 0.377 0.237 0.280</td><td>0.340 0.378 0.427 0.344 0.380</td><td>0.307 0.359 0.419 0.884 0.834</td><td>0.367 0.395 0.428 0.711 0.692</td></tr><tr><td></td><td>336 720 96 192 336</td><td>0.235 0.237 0.143 0.156 0.171</td><td>0.303 0.295 0.240 0.250 0.265</td><td>0.248 0.249 0.148 0.162 0.178</td><td>0.273 0.275 0.240 0.253 0.269</td><td>0.290 0.289 0.195 0.199 0.215</td><td>0.315 0.317 0.285 0.289 0.305</td><td>0.319 0.338 0.168 0.184 0.198</td><td>0.330 0.337 0.272 0.289 0.300</td><td>0.353 0.356 0.197 0.196 0.209</td><td>0.415 0.413 0.282 0.285 0.301</td><td>0.304 0.308 0.247 0.257 0.269</td><td>0.389 0.388 0.345 0.355 0.369</td><td>0.941 0.882 0.201 0.222 0.231</td><td>0.723 0.717 0.317 0.334 0.338</td></tr><tr><td></td><td>720 96 192 336 720</td><td>0.205 0.429 0.436 0.438 0.454</td><td>0.293 0.296 0.296 0.296 0.302</td><td>0.225 0.395 0.417 0.433 0.467</td><td>0.317 0.268 0.276 0.283 0.302</td><td>0.256 0.544 0.540 0.551 0.586</td><td>0.337 0.359 0.354 0.358 0.375</td><td>0.220 0.593 0.617 0.629 0.640</td><td>0.320 0.321 0.336 0.336 0.350</td><td>0.245 0.650 0.598 0.605 0.645</td><td>0.333 0.396 0.370 0.373 0.394</td><td>0.299 0.788 0.789 0.797 0.841</td><td>0.390 0.499 0.505 0.508 0.523</td><td>0.254 0.613 0.616 0.622 0.660</td><td>0.361 0.388 0.382 0.337 0.408</td></tr></table></body></html>

Table 2: Forecasting results with different input series lengths $l$ (96 for baselines, $\{ 4 8 0 , 1 9 2 0 , 3 6 0 0 \}$ for IRPA and then refined length to $r { = } 9 6$ ), with prediction length $k \in \{ 9 6 , 1 9 2 , 3 3 6 , 7 2 0 \}$ . The best results are in bold, the second-best are underlined.

# Model Analysis

Ablation Study We conducted extensive ablation studies on IRPA to investigate the impact of its various components on prediction performance. The results in Table 4 show that removing the Seasonal Refinement Module (w/o SRM), Trend Refinement Module (w/o TRM), Historical Similarity Auxiliary Module (w/o HSAM), and Historical Seasonal Pattern Prediction Auxiliary Module (w/o HSPPAM) leads to a decrease in model performance. This indicates that capturing the seasonality and long-term trends of the data is crucial for improving predictive accuracy, particularly for data with evident long-term variations, such as weather patterns or electrical load. Utilizing historical similar patterns and seasonal patterns can also improve predictive accuracy. These results validate the effectiveness of the IRPA framework design and demonstrate that the model can effectively leverage various patterns and features within time series data to enhance predictive performance.

Efficiency analysis To comprehensively evaluate the efficiency of the IRPA framework, we conducted a thorough analysis across four key dimensions: model parameter count, memory footprint, training time per iteration, and prediction performance. In our experimental setup, the lookback window length for baseline models was set to 96. The “+lookback” designation indicates an extension of the lookback window length to 3600, while $ { \mathrm { \cdot } } _ {  { \mathbf { + } }  { \mathrm { I R P A } } ^ {  { \prime }  { \prime } } }$ represents the integration of the IRPA framework into the base model with the same lookback window length. As shown in Figure 1 and Table 5, the IRPA framework demonstrates significant computational efficiency advantages compared to simply increasing the lookback window of baseline models. IRPA achieved an average reduction in MSE of at least $2 0 . 2 2 \%$ while simultaneously reducing the model’s parameter count by $6 9 . 2 5 \%$ . This not only effectively mitigates overfitting issues but also successfully addresses challenges related to resource constraints, as evidenced by the resolution of Out-ofMemory (OOM) problems encountered with the PatchTST model. For time complexity, from Equation 3, the complexity is $\begin{array} { r } { O ( m r ) = O ( \frac { 2 \dot { l } } { r } r ) \dot { = } O ( l ) } \end{array}$ . With fully connected layers, IRPA maintains ${ \bf \dot { \boldsymbol { O } } } ( l )$ complexity.

Table 3: The MSE improvement of baseline models with the IRPA framework. Results are averaged from all prediction lengths. Avg means further averaged by subsets.   

<html><body><table><tr><td>Datasets</td><td>ETT-Avg</td><td>Weather</td><td>Solar</td><td>ECL</td><td>Traffic</td></tr><tr><td>iTransformer +IRPA</td><td>0.383 0.339</td><td>0.258 0.226</td><td>0.233 0.207</td><td>0.178 0.168</td><td>0.428 0.416</td></tr><tr><td>Promotion</td><td>11.5%</td><td>12.4%</td><td>11.2%</td><td>5.6%</td><td>2.8%</td></tr><tr><td>PatchTST +IRPA</td><td>0.381</td><td>0.259</td><td>0.270</td><td>0.216</td><td>0.555</td></tr><tr><td></td><td>0.337</td><td>0.222</td><td>0.209</td><td>0.174</td><td>0.457</td></tr><tr><td>Promotion</td><td>11.5%</td><td>14.3%</td><td>22.6%</td><td>19.4%</td><td>17.7%</td></tr><tr><td>TimesNet</td><td>0.391</td><td>0.259</td><td>0.301</td><td></td><td>0.620</td></tr><tr><td>+IRPA</td><td>0.348</td><td>0.241</td><td></td><td>0.193</td><td></td></tr><tr><td>Promotion</td><td>11.0%</td><td></td><td>0.204</td><td>0.182</td><td>0.459</td></tr><tr><td></td><td></td><td>6.9%</td><td>32.2%</td><td>5.7%</td><td>26.0%</td></tr><tr><td>DLinear</td><td>0.442</td><td>0.265</td><td>0.330</td><td>0.212</td><td>0.625</td></tr><tr><td>+IRPA</td><td>0.330</td><td>0.219</td><td>0.221</td><td>0.169</td><td>0.439</td></tr><tr><td>Promotion</td><td>25.3%</td><td>17.4%</td><td>33.0%</td><td>20.3%</td><td>29.8%</td></tr></table></body></html>

Table 4: Average results from all prediction lengths of the ablation study conducted on three different time sampling frequency datasets.   

<html><body><table><tr><td>Datasets Metric</td><td>ETTm2</td><td>Weather</td><td>ECL</td></tr><tr><td>IRPA</td><td>MSE MAE 0.239 0.312</td><td>MSE MAE</td><td>MSE MAE 0.169 0.262</td></tr><tr><td>w/o SRM</td><td>0.242</td><td>0.219 0.270 0.221 0.276</td><td></td></tr><tr><td>w/o TRM</td><td>0.315 0.243 0.313</td><td>0.229 0.279</td><td>0.169 0.262</td></tr><tr><td>w/o HSAM</td><td></td><td></td><td>0.176 0.273</td></tr><tr><td></td><td>0.242 0.313</td><td>0.220 0.274</td><td>0.170 0.264</td></tr><tr><td>w/oHSPPAM</td><td>0.240 0.314</td><td>0.224 0.279</td><td>0.176 0.269</td></tr></table></body></html>

Varying Lookback Window We explored the influence of varying historical lookback window lengths on the forecasting efficacy of the IRPA model across six datasets, revealing a systematic relationship between sampling frequency and optimal lookback window length. Figure 3 shows that 60-minute interval datasets (ETTh1, ECL) perform best with a 480 length, 15-minute interval datasets (ETTm1, ETTm2) with a 1920 length, and 10-minute interval datasets (Weather, Solar) with a 2880/3600 length. The ratios between sampling frequencies $6 0 / 1 5 = 4$ , $6 0 / 1 0 = 6 \$ ) correspond to the ratios of optimal lookback windows (1920/480 $= 4$ $2 8 8 0 / 4 8 0 = 6 )$ ), suggesting a predictable scaling principle for window size selection in time series forecasting.

Table 5: Static and runtime metrics of IRPA and other baseline models on the Weather Dataset with a prediction length $k { = } 7 2 0$ .   

<html><body><table><tr><td>Model</td><td>Parameters</td><td>Memory</td><td>Train. Time</td><td>MSE</td></tr><tr><td rowspan="3">iTransformer +lookback +IRPA</td><td>5.15M</td><td>389.6MB</td><td>32.5ms</td><td>0.358</td></tr><tr><td>6.95M</td><td>590.6MB</td><td>40.6ms</td><td>0.414</td></tr><tr><td>5.97M</td><td>495.6MB</td><td>43.3ms</td><td>0.292</td></tr><tr><td rowspan="3">PatchTST + lookback +IRPA</td><td>10.74M</td><td>3642.0MB</td><td>128.3ms</td><td>0.354</td></tr><tr><td>172.20M</td><td>OOM</td><td></td><td></td></tr><tr><td>11.55M</td><td>3723.4MB</td><td>133.9ms</td><td>0.289</td></tr><tr><td rowspan="3">TimesNet + lookback +IRPA</td><td>1.25M</td><td>2148.9MB</td><td>894.9ms</td><td>0.365</td></tr><tr><td>16.73M</td><td>4420.4MB</td><td>1388.3ms</td><td>0.538</td></tr><tr><td>2.07M</td><td>2549.8MB</td><td>748.5ms</td><td>0.332</td></tr><tr><td rowspan="3">DLinear + lookback +IRPA</td><td>139.68K</td><td>72.4MB</td><td>8.3ms</td><td>0.345</td></tr><tr><td>5.19M</td><td>270.6MB</td><td>20.9ms</td><td>0.328</td></tr><tr><td>936.04K</td><td>199.4MB</td><td>22.3ms</td><td>0.285</td></tr></table></body></html>

![](images/914fcd83e83ffd55e38aaa17fa51096e0c012438b39f1b1cf88bca5f6b7990ef.jpg)  
Figure 3: MSE results of IRPA with varying lookback window lengths at prediction length $k { = } 7 2 0$ .

# Conclusion

In this paper, we introduce the IRPA framework, aimed at addressing critical challenges when using long lookback windows in long-term series forecasting, such as overfitting, inadequate information capture, and increasing demands on computational resources. IRPA innovatively refines inputs and auxiliary processes via specialized modules that efficiently extract crucial information from ultra-long lookback windows, thereby enhancing the quality of limited lookback windows and supporting the forecasting process. Experimental results indicate that IRPA significantly boosts the performance of various baseline models with only a minor increase in parameter count. This lightweight and generic framework provides a novel solution for long-term series forecasting, showcasing substantial advantages in enhancing predictive accuracy and computational efficiency.

# Acknowledgments

This work was supported by the National Natural Science Foundation of China (No.62472332, No.62276196), the “Open Bidding for Selecting the Best Candidates” Project of Wuhan East Lake High-Tech Development Zone (No.2024KJB322), and the Hubei Provincial International Science and Technology Cooperation Project (No.2024EHA031).