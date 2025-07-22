# HyperMixer: Specializable Hypergraph Channel Mixing for Long-term Multivariate Time Series Forecasting

Changyuan Tian1,2,3,4,\*, Zhicong $\mathbf { L u } ^ { 1 , 2 , 3 , 4 , * }$ , Zequn Zhang1,† Heming Yang1,2,3,4, Wei Cao1, Zhi Guo1,2, Xian $\mathbf { S u n } ^ { 1 , 2 , 3 , 4 }$ , Li Jin1,2,†

1Aerospace Information Research Institute, Chinese Academy of Sciences 2Key Laboratory of Target Cognition and Application Technology (TCAT) 3University of Chinese Academy of Sciences 4School of Electronic, Electrical and Communication Engineering, University of Chinese Academy of Sciences tianchangyuan21 $@$ mails.ucas.edu.cn, zqzhang1 $@$ mail.ie.ac.cn, jinlimails $@$ gmail.com

# Abstract

Long-term Multivariate Time Series (LMTS) forecasting aims to predict extended future trends based on channelinterrelated historical data. Considering the elusive channel correlations, most existing methods compromise by treating channels as independent or tentatively modeling pairwise channel interactions, making it challenging to handle the characteristics of both higher-order interactions and time variation in channel correlations. In this paper, we propose HyperMixer, a novel specializable hypergraph channel mixing plugin which introduces versatile hypergraph structures to capture group channel interactions and time-varying patterns for long-term multivariate time series forecasting. Specifically, to encode the higher-order channel interactions, we structure multiple channels into a hypergraph, achieving a two-phase message-passing mechanism: channel-to-group and group-to-channel. Moreover, the functionally specializable hypergraph structures are presented to boost the capability of hypergraph to capture the time-varying patterns across periods, further refining modeling of channel correlations. Extensive experimental results on seven available benchmark datasets demonstrate the effectiveness and generalization of our plugin in LMTS forecasting. The visual analysis further illustrates that HyperMixer with specializable hypergraphs tailors channel interactions specific to certain periods.

# Introduction

Long-term Multivariate Time Series (LMTS) forecasting aims to predict extended future trends of multiple interrelated variables (a.k.a, channels). It has long been studied and applied across diverse fields like traffic, weather, and energy. The channels within multivariate time series potentially denotes to monitoring variables in real systems (e.g., traffic sensors, power system indicators). Therefore, it is significant for LMTS to model intricate correlations entailed in these channels, followed by making effective predictions.

Considering the elusive channel correlations, most existing methods compromise through two main streams: (1) channel-independent (Zeng et al. 2023; Nie et al. 2023;

![](images/84fd4588f557092a3287e8e1cb8a9cab13463340a4455afc52d9b8e0779f2bde.jpg)  
Figure 1: Comparison of previous channel modeling methods (gray pathways) versus our method (colored pathways) in traffic scenarios. Previous methods focus on pairwise channel interactions and expect to model the correlations across different time periods using a general network. Our method targets higher-order channel interactions and models time-varying patterns via multiple specialized networks.

Wang et al. 2024): treating channels as independent and processing each channel separately, (2) channel-mixing (Zhou et al. 2021; Wu et al. 2021; Zhang and Yan 2023): tentatively modeling pairwise channel interactions by well-designed architectures, such as cross-channel attention and graph neural networks. Despite the temporary progress, they fail to completely unleash the power of channel correlations, making it challenging to handle the characteristics of both higherorder interactions and time variation in channel correlations.

The higher-order interaction refers to complex correlations among more than two channels within LMTS, reflecting obvious group behaviors. Taking the weekday traffic patterns illustrated in Figure 1 as an example, the traffic flow of road D is affected by the flow of road A flowing into road C, collectively forming a traffic route with group interaction behavior: roads A, D, and part of C involved in this group display consistent traffic flow trends. This informative higher-order interaction undoubtedly facilitates predicting the traffic flow of roads A, C, and D. However, it is either ignored by channel-independent methods or reduced to multi-hop pairwise interactions (i.e., roads A to C and roads C to D) by existing channel-mixing methods, which is not conducive to LMTS. In addition, the time-varying characteristic refers to the channel correlations exhibiting various association patterns over different time periods. As illustrated in Figure 1, the traffic route (A, C, D) on weekday shifts to the router (A, C, E) on weekends. This demands the model to capture versatile time-varying patterns across periods. Nevertheless, it is not afforded by current inflexible methods which are forced to balance the modeling of pairwise interactions: roads $\mathbf { C }$ to $\mathrm { ~ D ~ }$ and roads C to E, leading to a generalized compromise across all time periods. This is illustrated by the gray dashed line in Figure 1, indicating the ambiguous correlation between C and D. With the above consideration, it is crucial to capture the higher-order interactions and time-varying patterns for modeling channel correlations.

In this paper, we propose HyperMixer, a novel specializable hypergraph channel mixing plugin which introduces versatile hypergraph structures to capture group channel interactions and time-varying patterns for long-term multivariate time series forecasting. HyperMixer comprises two core components: a Hypergraph Chanel Mixer (HCM) and a Specializable Hypergraph Structure Learning module $\mathrm { ( H S ^ { \bar { 2 } } L ) }$ . Specifically, to encode the higher-order interactions, we structure multiple channels into a channel hypergraph, where each channel serves as a node and hyperedges connect multiple nodes simultaneously, forming channel groups. HCM is introduced to encode the channel hypergraph by conducting two-phase message-passing: channelto-group and group-to-channel. To capture the time-varying patterns, inspired by the specialization concept in the mixture of experts (Jacobs et al. 1991), we devise $\mathrm { H S ^ { 2 } L }$ to develop specialized hypergraph structures tailored to specific periods, utilizing a router layer and multiple experts. Ultimately, inserting the plugin HyperMixer into existing representative methods incorporates channel correlations featured by higher-order interactions and time-varying characteristics, improving the predictive performance for longterm multivariate time series. The main contributions of this paper are summarized as follows:

• We propose HyperMixer, a novel specializable hypergraph channel mixing plugin which introduces versatile hypergraph structures to capture group channel interactions and time-varying patterns for long-term multivariate time series forecasting.   
• To encode the higher-order channel interactions, we structure multiple channels into a hypergraph, achieving a two-phase message-passing mechanism: channel-togroup and group-to-channel. To capture the time-varying patterns across periods, the functionally specializable hypergraph structures are presented to boost the capability of hypergraph.   
• Extensive experimental results on seven available benchmark datasets demonstrate the effectiveness and generalization of our plugin in LMTS forecasting. The visual analysis further illustrates that HyperMixer with specializable hypergraphs tailors channel interactions specific to certain periods.

# Related Work

# Long-Term Multivariate Time Series Forecasting

Deep learning-based approaches dominate recent research, introducing a variety of designs suitable for Long-term multivariate time series (LMTS) forecasting (Zheng et al. 2020; Zhou et al. 2022; Nie et al. 2023). Recurrent Neural Networks (RNNs) (Che et al. 2018; Salinas et al. 2020) and Temporal Convolutional Networks (TCNs) (Lea et al. 2017) are utilized for modeling time series data to capture temporal dependencies. However, they encounter challenges when modeling long-term dependencies. Graph Neural Networks (GNNs) (Kipf and Welling 2017; Velicˇkovic´ et al. 2018) have also gained prominence due to their ability to explicitly capture spatiotemporal correlation in LMTS (Diao et al. 2019; Cao et al. 2020; Cai et al. 2024). However, these methods suffer from over-smoothing and overfitting, and are also limited to pairwise interactions (Lambiotte, Rosvall, and Scholtes 2019; Battiston et al. 2020). Recently, linear-based LMTS forecasting methods (Zeng et al. 2023; Wang et al. 2024; Huang et al. 2024) have achieved surprisingly strong performance with streamlined architectures. For example, LTSF-Linear (Zeng et al. 2023) is proposed to surpass previous complex methods through simple one-layer linear network. On the other hand, a series of transformer-based time series methods have been proposed (Wu et al. 2021; Zhou et al. 2021; Nie et al. 2023; Liu et al. 2024), notably advancing the development of LMTS forecasting. Representative PatchTST (Nie et al. 2023) employs subseries-level embedding techniques and the channel independence assumption to significantly improve the accuracy of LMTS forecasting. Despite significant progress, most current methods prioritize capturing temporal dependencies, underestimating the channel modeling in LMTS.

# Channel Modeling in LMTS Forecasting

Current methods can be divided into channel-independent methods and channel-mixed methods. Channel-independent methods assume that channels are independent and process each channel separately, thereby simplifying the task. However, they fail to exploit the potential benefits of channel correlations. In contrast, channel-mixed methods expect to capture the channel correlations through well-designed architectures. For example, methods based on GNN (Cao et al. 2020; Deng and Hooi 2021; Cai et al. 2024) organize channels into a graph, where edges facilitate message passing between pairs of channels. Patch-based methods (Zhang and Yan 2023; Huang et al. 2024) decompose each time series of the LMTS into multiple patches, enabling the exploration of dependencies between patches across different channels using cross-channel attention or multi-layer perceptrons (MLPs). Additionally, several techniques have been developed to explicitly represent specific channel relationships. For instance, channel clustering (Chen et al. 2024) is proposed to captures channel similarity by minimizing intra-cluster series distance, while LIFT (Zhao and Shen 2024) identifies lead-lag relationships between channels by analyzing cross-correlation coefficients, which indicate the degree of lead and lag between two channels. Despite advancements in capturing channel correlations through various designs, these methods primarily focus on pairwise interactions and attempt to capture interaction patterns across different periods using a shared, inflexible network. This approach presents difficulties in addressing the characteristics of both higher-order interactions and the time-varying patterns inherent in channel correlations.

![](images/553b50b8ede054bd2825a1cc60b04ceedb9398e0148fe6b25c10e0e7d4a78b8c.jpg)  
Figure 2: The left panel illustrates the process of representing multiple channels as a hypergraph. Five channels, involving two types of group interactions, are naturally represented as a hypergraph with five nodes and two hyperedges. The right panel presents the pipeline of HyperMixer.

# Proposed Method

The problem of LMTS forecasting is formulated as: Let $\mathbf { X } \in \mathbf { \mathbb { R } } ^ { L \times N }$ represent the historical data, where $L$ denotes the number of historical time steps and $N$ denotes the number of channels. The objective of LMTS forecasting is to predict a matrix $\hat { Y } \in \mathbb { R } ^ { \mathit { \hat { T } } \times N }$ , where $T$ represents the prediction length, i.e., the number of future time steps to be forecasted.

Our HyperMixer is illustrated in Figure 2, which involves channel hypergraph construction, specializable hypergraph structure learning, and a hypergraph channel mixing.

# Representing Multiple Channels as a Hypergraph

Compared to ordinary graphs, hypergraphs offer the advantage of hyperedges that can connect more than two nodes simultaneously, enabling them to naturally represent complex higher-order interactions (Feng et al. 2019; Jiang et al. 2019). A hypergraph is typically represented as $\mathcal { H } = \bar { ( \mathcal { V } , \mathcal { E } ) }$ , where $\nu$ denotes the set of nodes and $\mathcal { E }$ represents the set of hyperedges, with each hyperedge $e \in \mathcal { E }$ is a non-empty subset of nodes. The hypergraph structure can be futher formalized using an incidence matrix $\mathbf { H } \in \mathbb { R } _ { \geq 0 } ^ { | \mathcal { V } | \times | \mathcal { E } | }$ , where each element ${ \bf H } ( v , e )$ indicates the degree to which node $v$ is associated with hyperedge $e$ .

To model potential higher-order interactions among multiple channels, we organize them into a channel hypergraph, $\bar { \mathcal { H } } _ { C } = ( \mathcal { C } , \mathcal { E } _ { C } )$ . Here, $\mathcal { C }$ denotes the set of channels in the LMTS, and $\mathcal { E } _ { C }$ represents the hyperedges, with a hyperparameter $m$ as the total number of hyperedges. Figure 2 further illustrates this construction process.

# Specializable Hypergraph Structure Learning

After constructing the channel hypergraph $\mathcal { H } _ { C }$ , Specializable Hypergraph Structure Learning $\mathrm { ( \bar { H } S ^ { \bar { 2 } } L ) }$ develops a hypergraph structure $\mathbf { H } _ { C }$ for $\mathcal { H } _ { C }$ , tailored to specific periods by using a router layer and multiple hypergraph structure learning experts.

The processing workflow of $\mathrm { H S ^ { 2 } L }$ is as follows.

Channel Projection. First, we obtain the initial representations of channels in the feature space through a channel projection layer which maps the original time series $\mathbf { X }$ to a set of vectors. Formally,

$$
\mathbf { Z } _ { c } = { \mathrm { C h a n n e l P r o j } } ( \mathbf { X } ) ,
$$

here, $\mathbf { Z } _ { C } \in \mathbb { R } ^ { N \times d _ { c } }$ represents the channel representation, where $N$ is the number of channels, and $d _ { c }$ denotes the dimensionality of the channel representation. ChannelProj(·) is implemented using a one-layer linear transformation.

Router. The router is designed to identify features of input time series data and direct it to the appropriate expert.

For the channel representations $\dot { \mathbf Z } _ { C } ~ \in ~ \mathbb R ^ { N \times d _ { c } }$ , we first derive a compact representation vector $\mathbf { z } _ { c } \in \mathbb { R } ^ { d _ { c } }$ using an average pooling operation:

$$
\begin{array} { r } { { \bf z } _ { c } = \mathrm { A v e r a g e P o o l } ( { \bf Z } _ { C } ) . } \end{array}
$$

This representation vector is then fed into the router, which includes a simple linear layer. The output is normalized with the Softmax function to compute scores for each expert:

$$
\mathbf { s } = \mathrm { S o f t m a x } \big ( \mathrm { R o u t e r } ( \mathbf { z } _ { c } ) \big ) ,
$$

where $\mathbf { s } \in \mathbb { R } ^ { K }$ represents the score distribution across the $K$ experts.

Finally, a top- $\mathbf { \nabla } \cdot k$ activation mechanism is employed to select the top $k$ experts with the highest scores for the input data.

Hypergraph Structure Learning Expert. Upon assignment by the router, each sample is directed to a specific expert $E$ , which is responsible for learning a hypergraph structure $\mathbf { H } _ { C } ^ { E }$ based on the sample’s channel representation $\mathbf { Z } _ { c }$ . This process is formally defined as follows:

$$
\begin{array} { r } { { \bf H } _ { C } ^ { E } = \mathrm { E x p e r t } ( { \bf Z } _ { C } ) , } \end{array}
$$

where $\mathbf { H } _ { C } ^ { E } \in \mathbb { R } ^ { N \times m }$ denotes the hypergraph structure learned by expert $E$ . Here, $m$ is a hyperparameter that specifies the number of hyperedges in the channel hypergraph. Each hypergraph structure learning expert $E$ is implemented as a single-layer linear transformation followed by a ReLU activation function.

In more detail, HyperMixer includes one expert shared across all LMTS data and $K$ additional experts that can be selectively activated.

Learned Hypergraph Structures. The hypergraph structure output by the $\mathrm { H } \bar { \mathrm { S } } ^ { 2 } \mathrm { L }$ consists of two parts. The first part is the hypergraph structure learned by a shared hypergraph structure learning expert, denoted as $\mathbf { H } _ { S }$ . The second part is the combination of the outputs from the top- $k$ experts.

Let $\{ E _ { i _ { 1 } } , E _ { i _ { 2 } } , . . . , E _ { i _ { k } } \}$ be the $k$ experts selected for the LMTS data sample $i$ in a batch by the top- $\mathbf { \nabla } \cdot k$ selection mechanism. Each expert $E _ { i _ { j } }$ outputs a hypergraph structure $\mathbf { H } _ { C } ^ { i _ { j } }$ , where $j = 1 , 2 , \dots , k$ . The final hypergraph structure $\mathbf { H } _ { C }$ is represented as the weighted sum of the shared expert’s output and these experts’ outputs:

$$
\mathbf { H } _ { C } = \mathbf { H } _ { S } + \sum _ { j = 1 } ^ { k } s _ { i _ { j } } \mathbf { H } _ { C } ^ { i _ { j } } ,
$$

where $s _ { i _ { j } }$ is the score corresponding to expert $E _ { i _ { j } }$ , which is the $i _ { j }$ th element in the expert score vector s.

# Hypergraph Chanel Mixer

The Hypergraph Channel Mixer (HCM), implemented by a hypergraph neural network (Feng et al. 2019; Jiang et al. 2019), aims to encode the channel hypergraph $\mathcal { H } _ { C }$ and its structure $\mathbf { H } _ { C }$ , effectively mapping the nodes of the hypergraph into a $d$ -dimensional vector space. This encoding captures complex channel correlations by leveraging higherorder interactions inherent in the hypergraph structure.

To achieve this, HCM employs a two-phase messagepassing process as illustrated in Figure 2. In the first phase, messages are transmitted from nodes to hyperedges (i.e., groups), allowing for the aggregation of collective node information. In the second phase, messages are sent from hyperedges back to nodes, thereby updating and refining the node representations.

From Node to Hyperedge. First, HCM performs message passing from nodes to hyperedges. To obtain the messages passed from nodes to hyperedges, we employ a single-layer nonlinear transformation to convert the original node representations. Specifically,

$$
{ \bf Z } _ { { \mathcal { C } } \to { \mathcal { E } } _ { C } } = \mathrm { R e L U } ( { \bf W } _ { 1 } { \bf Z } _ { C } ) ,
$$

where $\mathbf { W } _ { 1 } ~ \in ~ \mathbb { R } ^ { d _ { c } \times d _ { c } }$ denotes the learnable parameters, and $\mathbf { Z } _ { \mathcal { C }  \mathcal { E } _ { C } } \in \mathbb { R } ^ { N \times d _ { c } }$ represents the messages passed from nodes to hyperedges.

Next, we aggregate the messages from the nodes to obtain the hyperedge representations. This process is expressed as:

$$
\mathbf { Z } _ { \mathcal { E } _ { C } } = \mathbf { D } _ { \mathcal { E } _ { C } } ^ { - 1 } \mathbf { H } _ { C } ^ { T } \mathbf { Z } _ { \mathcal { C }  \mathcal { E } _ { C } } ,
$$

where $\mathbf { D } _ { \mathcal { E } _ { C } } ^ { - 1 } \in \mathbb { R } ^ { m \times m }$ represents the degree matrix of the hyperedges, which is a diagonal matrix with $\mathbf { D } _ { \mathcal { E } _ { C } } ^ { - 1 } ( e , e ) =$ $\begin{array} { r } { ( \sum _ { c \in \mathcal { C } } \mathbf { H } _ { C } ( c , e ) ) ^ { - 1 } } \end{array}$ ; $\mathbf { H } _ { C } ^ { T } \in \mathbb { R } ^ { m \times N }$ denotes the transpose of the hypergraph structure, and $\mathbf { Z } _ { \mathcal { E } _ { C } } \in \mathbb { R } ^ { m \times d _ { c } }$ represents the hyperedge representations.

From Hyperedge to Node. After obtaining hyperedge representations, HCM performs message passing from hyperedges to nodes. Specifically, to obtain the message from hyperedge $e$ to node $c$ , HCM first performs a concatenation operation and then uses a single-layer nonlinear transformation to convert the concatenated vector:

$$
\begin{array} { r } { \mathbf { z } _ { e  c } = \operatorname { R e L U } ( [ \mathbf { Z } _ { C } ( c ) , \mathbf { Z } _ { \mathcal { E } _ { C } } ( e ) ] \mathbf { W } _ { 2 } ) , } \end{array}
$$

where $[ \cdot , \cdot ]$ denotes the concatenation operation of vectors, $\mathbf { W } _ { 2 } \in \mathbf { \bar { R } } ^ { 2 \bar { d } _ { c } \times d _ { c } }$ represents the learnable weight parameters, and $\mathbf { z } _ { e  c } \in \mathbb { R } ^ { d _ { c } }$ represents the message from hyperedge $e$ to node $c$ . $\mathbf { Z } _ { C } ( c )$ denotes the representation vector of node $c$ and ${ \bf Z } _ { { \scriptscriptstyle \mathcal E } _ { C } } ( e )$ denotes the representation vector of hyperedge $e$ .

Next, we need to aggregate messages from all hyperedges to update the representation of node $c$ . This can be represented as:

$$
{ \bf Z } _ { C } ^ { \prime } ( c ) = \sum _ { e \in { \mathcal { E } } _ { C } } { \bf H } _ { C } ( c , e ) { \bf z } _ { e  c } ,
$$

where $\mathbf { Z } _ { C } ^ { \prime } ( c )$ represents the new representation of node $c$ .

To maintain the stability of node representations, the aggregated node representation is normalized as follows:

$$
\mathbf { Z } _ { C } ^ { \prime \prime } = \mathbf { D } _ { C } ^ { - 1 } \mathbf { Z } _ { C } ^ { \prime } ,
$$

here, $\mathbf { D } _ { C }$ represents the degree matrix of nodes, which is a diagonal matrix with diagonal elements representing the degree of each node, defined as $\begin{array} { r } { \mathbf { D } _ { C } ( i , i ) = \dot { \sum _ { e \in \mathcal { E } } } \mathbf { H } ( i , e ) } \end{array}$ .

Update Channel Representation. The updated channel representation is obtained as follows:

$$
\bar { \mathbf Z } _ { c } = \mathbf Z _ { c } + \mathbf Z _ { c } ^ { \prime \prime } .
$$

Table 1: The statistics of datasets.   

<html><body><table><tr><td>Dataset</td><td># Channels</td><td># Timesteps</td><td>Frequency</td></tr><tr><td>ETTh1</td><td>7</td><td>17,420</td><td>1 hour</td></tr><tr><td>ETTh2</td><td>7</td><td>17,420</td><td>1 hour</td></tr><tr><td>ETTm1</td><td>7</td><td>69,680</td><td>15 min</td></tr><tr><td>ETTm2</td><td>7</td><td>69,680</td><td>15 min</td></tr><tr><td>Weather</td><td>21</td><td>52.696</td><td>10 min</td></tr><tr><td>Electricity</td><td>321</td><td>26,304</td><td>1 hour</td></tr><tr><td>Traffic</td><td>862</td><td>17,544</td><td>1 hour</td></tr></table></body></html>

This enhanced channel representation is then utilized to refine the original multivariate time series representation, which is specifically expressed as:

$$
\mathbf { X } ^ { \prime } = \hat { \mathbf { X } } + \bar { \mathbf { Z } } _ { c } \mathbf { W } _ { o } ,
$$

where $\mathbf { W } _ { o } \in \mathbb { R } ^ { d _ { c } \times L }$ are learnable parameters.

The representation of the multivariate time series data improved by HyperMixer can be fed into various backbones, such as TimeMixer (Wang et al. 2024), PatchTST (Nie et al. 2023), and DLiner (Zeng et al. 2023), to enhance predictive performance.

# Experiment

# Datasets

We select seven mainstream long-term multivariate time series datasets, including ETT (ETTh1, ETTh2, ETTm1, ETTm2) (Zhou et al. 2021), Traffic (Lai et al. 2018), Weather (Zeng et al. 2023), and Electricity (Wu et al. 2020). Table 1 presents the statistics of the seven datasets.

# Experimental Details

The proposed HyperMixer serves as a channel mixing plugin that can be incorporated into a variety of existing LMTS forecasting backbones. To validate the enhancement brought by HyperMixer, we select four current state-ofthe-art LMTS forecasting models as our backbones. These include the linear-based methods TimeMixer(Wang et al. 2024) and DLinear(Zeng et al. 2023), the transformer-based method PatchTST(Nie et al. 2023), and the convolutionbased method TimesNet(Wu et al. 2023). Additionally, we compare our approach with another competitive channel modeling method, CCM(Chen et al. 2024), which is based on a channel clustering strategy. For a fair comparison, the optimal hyperparameter settings provided in the official implementations of the backbones are used for both the backbones and the HyperMixer-enhanced models. For CCM, we use the experimental results provided in the original paper for PatchTST, DLinear, and TimesNet, while the results for the recently proposed TimeMixer are based on our own reproduction.

For our proposed HyperMixer, the number of experts $T$ is explored within the range of [4, 12], with the number of activated experts, $k$ , set to 2. The number of hyper-edges, $m$ , is searched within the range of [4, 10]. Mean Squared Error (MSE) and Mean Absolute Error (MAE) are used as metrics for evaluating predictive performance. For all datasets in our experiments, the prediction lengths are set to 96, 192, 336, $7 2 0 \}$ . We use the L2 loss function for training. All experiments are implemented using PyTorch and executed on an NVIDIA A800-80GB GPU.

![](images/f381cc99351f4660eb4d62e918a408ad9bc739689e5a21edf837f4d03017e960.jpg)  
Figure 3: t-SNE visualization of channel representations on the ETTh2 Dataset.

# Main Results

Table 2 presents the experimental results of HyperMixer across four representative backbones and seven datasets. The results clearly demonstrate that HyperMixer significantly enhances the performance of the backbones, achieving an average performance boost of $3 . 4 6 \%$ . Notably, models equipped with HyperMixer achieve the best performance in most cases, outperforming both the base models and those equipped with CCM. In details, ETTh1 is more challenging than ETTm1 due to its larger sampling intervals and resulting more pronounced fluctuations. HyperMixer’s substantial improvements on ETTh1 highlight the importance of channel correlations in predicting unstable trends, with similar patterns seen in ETTm2 and ETTh2. Furthermore, in datasets like Weather, Electricity, and Traffic—characterized by a higher number of channels—HyperMixer achieves even greater average improvements compared to other ETT datasets with fewer channels. This is attributed to its powerful capability in modeling complex channel correlations.

# Qualitative Visualization

Channel Representation Visualization. To understand why HyperMixer works, we further conduct a visualization analysis of HyperMixer on the ETTh2 dataset, which includes 7 channels. Figures 3a and Figure 3b respectively present the t-SNE visualizations of channel representations learned by the base model DLinear and the DLinear model equipped with HyperMixer. A noticeable transformation occurs with HyperMixer: previously dispersed channel representations become more tightly grouped and concentrated within the feature space. To further illustrate HyperMixer’s capability in identifying channel group behavior, we visualize the 7 channels in Figure 3c. This figure shows the time series data over 200 time steps from the ETTh2 dataset. Channels 1, 2, 3, and 4, which exhibit similar behaviors, are closely aligned in the feature space as demonstrated in Figure 3b. Likewise, channels 5 and 6 show similar clustering patterns. In contrast, channel 7, which displays unique behavior, stands apart with an independent feature representation in Figure 3b. These visualizations demonstrate that HyperMixer with specializable hypergraphs excels in identifying and encoding intricate channel correlations, resulting in more expressive and informative channel representations.

Table 2: Forecasting results, averaged over four prediction lengths: $\{ 9 6 , 1 9 2 , 3 3 6 , 7 2 0 \}$ . A smaller MSE or MAE signifies a more accurate prediction. A bold result signifies the best performance, while an underlined result indicates the second-best performance. The look-up window length $L$ aligns with each backbone’s official setting: 512 for PatchTST, 336 for DLinear, and 96 for both TimeMixer and TimesNet. For HyperMixer, the number of hyperedges is set to 10 and the number of experts to 4 for the ETTh2, ETTm1 and Traffic datasets, while 8 for others. Imp represents the average percentage improvement achieved by HyperMixer in the MSE and MAE metrics across all backbones.   

<html><body><table><tr><td>Model</td><td>Metric</td><td>ETTh1</td><td>ETTh2</td><td>ETTm1</td><td>ETTm2</td><td>Weather</td><td>Electricity</td><td>Traffic</td></tr><tr><td>PatchTST</td><td>MSE</td><td>0.417</td><td>0.332</td><td>0.354</td><td>0.270</td><td>0.227</td><td>0.167</td><td>0.396</td></tr><tr><td></td><td>MAE MSE</td><td>0.431 0.412</td><td>0.383</td><td>0.385 0.353</td><td>0.329 0.262</td><td>0.264</td><td>0.260</td><td>0.267 0.389</td></tr><tr><td rowspan="2">+CCM</td><td>MAE</td><td>0.430</td><td>0.330 0.372</td><td>0.381</td><td>0.322</td><td>0.225 0.263</td><td>0.167 0.261</td><td>0.259</td></tr><tr><td></td><td></td><td>0.327</td><td>0.346</td><td>0.256</td><td></td><td></td><td>0.381</td></tr><tr><td rowspan="2">+HyperMixer</td><td>MSE MAE</td><td>0.408 0.430</td><td>0.379</td><td>0.380</td><td>0.315</td><td>0.223 0.260</td><td>0.162 0.255</td><td>0.262</td></tr><tr><td></td><td></td><td>0.387</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TimeMixer</td><td>MSE MAE</td><td>0.463 0.446</td><td>0.409</td><td>0.385 0.400</td><td>0.276 0.323</td><td>0.245 0.275</td><td>0.182 0.273</td><td>0.485 0.298</td></tr><tr><td rowspan="2">+CCM</td><td>MSE</td><td>0.455</td><td>0.383</td><td>0.388</td><td>0.277</td><td>0.246</td><td>0.189</td><td>0.486</td></tr><tr><td>MAE</td><td>0.443</td><td>0.408</td><td>0.400</td><td>0.323</td><td>0.274</td><td>0.277</td><td>0.304</td></tr><tr><td rowspan="2">+HyperMixer</td><td>MSE</td><td>0.450</td><td>0.376</td><td>0.386</td><td>0.276</td><td>0.243</td><td></td><td></td></tr><tr><td>MAE</td><td>0.440</td><td>0.403</td><td>0.399</td><td>0.323</td><td>0.274</td><td>0.171 0.264</td><td>0.464 0.294</td></tr><tr><td>DLinear</td><td>MSE</td><td>0.433</td><td>0.431</td><td>0.359</td><td>0.265</td><td>0.246</td><td>0.166</td><td>0.434</td></tr><tr><td rowspan="2"></td><td>MAE</td><td>0.445</td><td>0.444</td><td>0.381</td><td>0.329</td><td>0.299</td><td>0.263</td><td>0.295</td></tr><tr><td>MSE</td><td>0.423</td><td>0.400</td><td>0.355</td><td>0.289</td><td>0.255</td><td>0.173</td><td>0.435</td></tr><tr><td rowspan="2">+CCM</td><td>MAE</td><td>0.437</td><td>0.428</td><td>0.378</td><td>0.349</td><td>0.303</td><td>0.275</td><td>0.296</td></tr><tr><td>MSE</td><td>0.414</td><td>0.349</td><td>0.357</td><td>0.261</td><td>0.228</td><td>0.165</td><td>0.421</td></tr><tr><td rowspan="2">+HyperMixer</td><td>MAE</td><td>0.425</td><td>0.395</td><td>0.375</td><td>0.316</td><td>0.263</td><td>0.257</td><td>0.297</td></tr><tr><td>MSE</td><td>0.458</td><td>0.414</td><td>0.400</td><td>0.291</td><td></td><td></td><td></td></tr><tr><td rowspan="2">TimesNet</td><td>MAE</td><td>0.450</td><td>0.427</td><td>0.406</td><td>0.333</td><td>0.259 0.287</td><td>0.193 0.295</td><td>0.620 0.336</td></tr><tr><td>MSE</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">+CCM</td><td>MAE</td><td>0.454 0.445</td><td>0.411 0.422</td><td>0.399 0.405</td><td>0.288 0.330</td><td>0.256 0.281</td><td>0.179 0.279</td><td>0.571 0.339</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">+HyperMixer</td><td>MSE</td><td>0.447</td><td>0.383 0.406</td><td>0.393 0.403</td><td>0.286 0.326</td><td>0.249 0.277</td><td>0.178 0.273</td><td>0.540 0.312</td></tr><tr><td>MAE</td><td>0.439</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Imp (%)</td><td></td><td>2.54%</td><td>6.15%</td><td>1.02%</td><td>2.37%</td><td>3.88%</td><td>4.02%</td><td>4.20%</td></tr></table></body></html>

Visualization of the $\mathbf { H S ^ { 2 } L }$ . To analyze the behavior of $\mathrm { H S ^ { 2 } L }$ , we present the distinguishable periods identified by the router in $\mathrm { H S ^ { 2 } L }$ and the specialized hypergraph structures that correspond to these periods. Each time series data depicted in Figure 4a is generated by averaging multiple instances assigned by the router to a specific expert, serving as a representative of all data handled by that expert. From Figure 4a , we can observe that the router can identifying distinct periods in the LMTS data and assigning them to the appropriate experts for processing. Specifically, experts 1 and 2 are responsible for handling LTMS data with downward trends, whereas experts 3 and 4 deal with upward trends. Moreover, although both experts 1 and 2 process data with downward trends, their focal points differ: expert 1 processes data that end in a trough position, whereas expert 2 processes data that end in a peak position, which imply opposite future trend changes. Experts 3 and 4 exhibit similar distinctions. Figure 4b illustrates the normalized hypergraph structures output by the four experts; larger values

![](images/6af511a74237e6575ea48a99acef9c4f14da22bde40e0414c54f533b26ef2799.jpg)  
(a) The distinguishable periods identified by the router in $\mathrm { H S ^ { 2 } L }$

Expert 1 Expert 2 Expert 3 Expert 4 0 0.12 0.27 0.27 0.17 0.44 0.34 0.39 1.0 1 0.28 0.47 0.49 0.85 0.98 -0.8 0.18 0.42 0.42 0.28 0.50 1.00 0.27 0.43 0.46 0.81 0.87 -0.6 -0.4 0.04 0.04 0.03 0.01 0.09 0.05 0.14 0.29 -0.2 51 0.25 0.32 0.26 0.26 0.44 0.74 6 0.31 0.38 0.33 0.25 0.24 0.24 -0.0 6 i 0 1 0 0 hyperedges (b) Specialized hypergraphs with 2 hyperedges

indicate stronger higher-order channel interactions. It is observed that the hypergraph structure values for experts 3 and 4, who handle more volatile data, are generally higher. This may suggest that when LMTS data changes more dramatically (e.g., rapid increases), accurate predictions rely more heavily on the information provided by channel correlation.

# Ablation Study

To validate the effectiveness of HyperMixer components, we conduct ablation experiments using TimesNet as the base model. We incrementally integrate HCM and $\mathrm { H S ^ { 2 } L }$ to assess their contributions across ETTh1, ETTm2, and Weather datasets. Observations from Table 3 are as follows. First, starting from the base model, the variant with HCM and the variant with $\mathrm { H C M + H S ^ { 2 } L }$ (i.e., TimesNet+HyperMixer) show progressively better predictive performance, highlighting each component’s effectiveness. Secondly, HCM notably contributes to datasets with more channels like Weather, indicating its ability to capture complex correlations. Thirdly, $\mathrm { H S ^ { \bar { 2 } } L }$ yields greater gains on datasets like Weather $( l . 6 4 \% )$ and ETTm2 $( \boldsymbol { I } . \boldsymbol { I 0 \% } )$ , which contain richer time-varying patterns due to their minute-level sampling frequency. This suggests that the $\mathrm { H S ^ { 2 } L }$ can effectively capture the time-varying patterns in LMTS data, thus strengthening the model’s predictive capability.

Table 3: Ablation experiment based on the base model TimesNet. Gain represents the average improvement percentage over the previous setup for MSE and MAE metrics.   

<html><body><table><tr><td>Model</td><td>Metric</td><td>ETTh1</td><td>ETTm2</td><td>Weather</td></tr><tr><td rowspan="2">TimesNet</td><td>MSE</td><td>0.458</td><td>0.291</td><td>0.259</td></tr><tr><td>MAE</td><td>0.450</td><td>0.333</td><td>0.287</td></tr><tr><td rowspan="3">+HCM</td><td>MSE</td><td>0.451</td><td>0.290</td><td>0.254</td></tr><tr><td>MAE</td><td>0.440</td><td>0.329</td><td>0.281</td></tr><tr><td>Gain (%)</td><td>1.87%</td><td>0.75%</td><td>1.88%</td></tr><tr><td rowspan="3">+HCM+HS²L</td><td>MSE</td><td>0.447</td><td>0.286</td><td>0.249</td></tr><tr><td>MAE</td><td>0.439</td><td>0.326</td><td>0.277</td></tr><tr><td>Gain (%)</td><td>0.61%</td><td>1.10%</td><td>1.64%</td></tr></table></body></html>

![](images/9f83f8a829f64a3c8d8b1f870972cb7df77f87afff47568f3ff98c3b7f02a6f4.jpg)  
Figure 4: Visualization of the $\mathrm { H S ^ { 2 } L }$ using the DLinear base model on the ETTh2 dataset   
Figure 5: Hyperparameter analysis on the weather dataset with a prediction length of 96 across different bachbones: PatchTST, TimeMixer, DLinear, and TimesNet

# Hyperparameter Analysis

To explore the impact of two important hyperparameters: the number of experts $K$ and the number of hyperedges $m$ , we conduct a hyperparameter analysis. As shown in Figure 5a, with an increase in the number of experts, the performance of all models enhanced by HyperMixer initially improves and then deteriorates. This suggests that establishing an appropriate number of experts to avoid redundancy is necessary, with a range of 2 to 10 being advisable. Similarly, Figure 5b shows that choosing 6 to 14 hyperedges is a suitable starting point.

# Conclusion

In this paper, we propose HyperMixer, a novel channel mixing plugin with specializable hypergraphs, to effectively manage the multiple interrelated channels in LMTS forecasting. Empowered by hypergraph channel mixing and period-oriented hypergraph structure specialization, HyperMixer excels in channel modeling by encoding high-order interactions and capturing time-varying patterns. Moreover, HyperMixer can incorporate this channel modeling capability into arbitrary forecasting models as a plugin. Extensive experiments demonstrate the superiority of HyperMixer in LMTS forecasting. Further visualization and ablation studies are presented to provide insights for our HyperMixer.