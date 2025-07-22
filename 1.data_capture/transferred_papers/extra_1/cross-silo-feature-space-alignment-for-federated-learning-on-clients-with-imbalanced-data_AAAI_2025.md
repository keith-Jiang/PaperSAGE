# Cross-Silo Feature Space Alignment for Federated Learning on Clients with Imbalanced Data

Zhuang $\mathbf { Q } \mathbf { i } ^ { 1 }$ , Lei Meng1,2\*, Zhaochuan $\mathbf { L i } ^ { 3 }$ , Han $\mathbf { H } \mathbf { u } ^ { 4 }$ , Xiangxu Meng

1School of Software, Shandong University, China 2Shandong Research Institute of Industrial Technology, China 3Inspur, China 4School of information and Electronics, Beijing Institute of Technology, China z qi@mail.sdu.edu.cn, lmeng $@$ sdu.edu.cn, lizhaoch $@$ inspur.com, hhu $@$ bit.edu.cn, mxx $@$ sdu.edu.cn

# Abstract

Data imbalance across clients in federated learning often leads to different local feature space partitions, harming the global model’s generalization ability. Existing methods either employ knowledge distillation to guide consistent local training or performs procedures to calibrate local models before aggregation. However, they overlook the ill-posed model aggregation caused by imbalanced representation learning. To address this issue, this paper presents a cross-silo feature space alignment method (FedFSA), which learns a unified feature space for clients to bridge inconsistency. Specifically, FedFSA consists of two modules, where the in-silo prototypical space learning (ISPSL) module uses predefined text embeddings to regularize representation learning, which can improve the distinguishability of representations on imbalanced data. Subsequently, it introduces a variance transfer approach to construct the prototypical space, which aids in calibrating minority classes feature distribution and provides necessary information for the cross-silo feature space alignment (CSFSA) module. Moreover, the CSFSA module utilizes augmented features learned from the ISPSL module to learn a generalized mapping and align these features from different sources into a common space, which mitigates the negative impact caused by imbalanced factors. Experimental results verified that FedFSA improves the consistency between diverse spaces on imbalanced data, which results in superior performance compared to existing methods.

# Introduction

Federated learning enables collaborative modeling with imbalanced data from various sources, which shares model parameters instead of raw data between data sources and the server (Hu et al. 2024b; Liu et al. 2023; Cai et al. 2024b; Qi et al. 2022; Kairouz et al. 2021). This significantly improves the effective utilization of isolated data, enabling them to contribute to cooperative decision-making and learn a generalized model (Cai et al. 2024a; Wang et al. 2023a; Meng et al. 2024; Wang et al. 2024a). However, existing studies show that the data heterogeneity between clients could lead to a decrease in the effectiveness of collaborative modeling (Qi et al. 2024a; Shi et al. 2023; Wen et al. 2023; Qi et al. 2024b). This is mainly because learning a consistent feature

In-Silo Prototypical Space Learning Cross-Silo Feature Space Alignment Client Prototypical Multi-Source Relearned Common Prototypes Space Prototypical Spaces Feature Space Majority to MinorityVariance Transfer Heterologous Space Alignment Predicts □ Shared Decision Boundaries Local Prototypes Augmented Features Variance

space becomes challenging when dealing with data that is imbalanced within clients and has inconsistent distribution across clients, which makes it difficult to integrate learners with inconsistent objectives into a remarkable model.

To alleviate issues of class imbalance, existing methods can roughly be categorized into two types. The former approach typically employs knowledge distillation to guide local model learning on the client-side, which aims to transfer global knowledge to client models and leverage regularization techniques to guide them in learning consistent representations of data (MOON (Li, He, and Song 2021), Fedproc (Mu et al. 2023), FedNTD (Lee et al. 2022) and FPL (Huang et al. 2023)). For instance, Fedproc and FPL construct prototypical representation for each class of samples and employ it to facilitate the feature space alignment across clients. The latter method usually involves model calibration, including global classifier fine-tuning and projection head retraining, which is aimed at mitigating bias issues introduced by the weighted averaging of local models (CReFF (Shang et al. 2022), CLIP2FL (Shi et al. 2024) and FedCSPC (Qi et al. 2023)), where CReFF and CLIP2FL focus on refining the global classifier to enhance its robustness with varied feature environments. Notably, these strategies have shown promising results in classes with a majority of samples. However, data imbalance typically results in poor representation learning for minority classes in clients. Additionally, they directly use the intermediate features output by clients to retrain the model, but the inherent differences between features from different sources limit their effectiveness.

To address this problem, this paper presents a cross-silo feature space alignment method, termed FedFSA, which constructs a unified space to align features from diverse sources to mitigate the negative impact of data imbalance. As depicted in Figure 1, FedFSA includes two main modules, including the in-silo prototypical space learning (ISPSL) module and the cross-silo feature space alignment (CSFSA) module. Specifically, the ISPSL module leverages pre-defined feature sapce learned from pretrained CLIP to guide the local representation learning, which is conducive to enhance the discriminability of the minority class representation in imbalanced data. Moreover, the ISPSL module introduces the variance transfer technique that leverage the diversity of samples in majority classes to expand and construct the prototypical space of minority classes. This provides meaningful and privacy-preserving information for the cross-silo feature space alignment (CSFSA) module. Subsequently, the CSFSA module maps features from diverse spaces into a unified space to further reduce feature distribution discrepancies between data sources caused by class imbalanced and emphasizes the contribution of each feature by weighting them based on their attention scores, effectively mitigating the interference of outliers.

Experiments were conducted on three datasets, including performance comparison, ablation study, in-depth analysis, case study and error analysis of FedFSA. The results validate that FedFSA can promote precise cross-silo feature alignment on imbalanced data. Furthermore, the error analysis can offer valuable insights to guide future refinements. In summary, the main contributions of this paper include:

• To alleviate the negative impact of data imbalance in federated learning, this study proposes a cross-silo feature space alignment method (FedFSA). To the best of our knowledge, FedFSA is the first method to align feature spaces from different sources on imbalanced data.   
• This study proposes a model-agnostic framework, which can integrate various client-based methods. It mitigates the impact of imbalanced data by learning a shared feature space for different clients.   
• Experimental findings have revealed that aligning the feature spaces of different clients can benefit the retrained model, which avoids the impact of inherent differences between client feature spaces. This provides a feasible approach for future research.

# Related Work Methods Based on Knowledge Distillation

Knowledge distillation methods aim to guide clients to learn consistent knowledge, which mitigates data imbalance in federated learning. Typically, they entail the use of extra information as a regularizer to regulate updates locally (Li, He, and Song 2021; Tan et al. 2022; Huang et al. 2023; Yu et al. 2021; Ye et al. 2023; Li et al. 2024b; Ren et al. 2024). Within the context, regularization have played a significant role (Wu et al. 2023). For example, MOON employs contrastive regularization to penalize inconsistencies between local and global feature spaces (Li, He, and Song 2021). FedProc (Mu et al. 2023), FedProto (Tan et al. 2022) and FPL (Huang et al. 2023) construct prototypes for each class based on data representations to represent the center of within-class representations. It guides the local training process by constraining the representations of all clients to converge towards these prototypes. Moreover, guiding the calibration of the feature space with a classifier is also an effective strategy, such as FedETF employs a fixed simplex equiangular tight frame classifier to encourage all clients in learning a unified and optimal feature representation (Li et al. 2023). AdressIM infers the global data distribution and mitigates global imbalance by using a ratio-weighted approach (Wang et al. 2021). Despite the positive outcomes of these methods, further exploration is needed for imbalanced data, as imbalances typically accumulate errors across training iterations.

# Methods Based on Model Calibration

Different from knowledge distillation, model calibration methods concentrate on making improvements on the server side, which re-trains global model to alleviate class imbalance issues. These methods along this line including global classifier calibration (Luo et al. 2021; Shang et al. 2022; Zeng et al. 2023; Shi et al. 2024), projection head retraining (Qi et al. 2023), and global model fine-tuning (Zhang et al. 2022; Hu et al. 2024a,c). They both hope to obtain a generalized model to fit all data from various sources. For instance, CCVR fuses the mean and variance of sample features obtained from client and employs a gaussian model to generate virtual features for retraining the global classifier (Luo et al. 2021). CReFF (Shang et al. 2022) and CLIP2FL (Shi et al. 2024) generate a series of federated features with gradients consistent with real data to fine-tune the classifier. FedFTG transfers knowledge from local to global models by exploring input spaces with a generator to fine-tune the entire global model (Zhang et al. 2022). From these analysis, their performance is closely tied to the quality of local feature information. However, both of them overlook the interference of local imbalance.

# Problem Formulation

Federated learning systems typically utilize multiple data sources to collaboratively build the global model. It contains $K$ data sources, $\boldsymbol { S } = \{ \boldsymbol { \bar { S } } _ { 1 } , . . . , \boldsymbol { S } _ { K } \}$ , and a central sever $S$ . The source $S _ { k }$ utilizes its private data $D ^ { k } \ : = \ : \{ ( X ^ { k } , Y ^ { k } ) \}$ to optimize the model $M _ { k }$ with the objective $\ell _ { k } ( \theta _ { k } ; D ^ { k } )$ , where $\theta _ { k }$ is the parameter of the model $M _ { k }$ . And the server $S$ aggregates the parameters of all locally learned models $\{ \theta _ { k } | k = 1 , . . . , K \}$ to obtain global parameters, i.e., $\theta _ { g } = $ $\textstyle \sum _ { k = 1 } ^ { K } \alpha _ { k } \theta _ { k }$ , where where $\alpha _ { k } = | D ^ { k } | / \bigcup _ { k = 1 } ^ { K } | D ^ { k } |$ .

By comparison, FedFSA introduces the in-silo feature space reconstruction (ISPSL) module and the cross-silo feature space alignment (CSFSA) module, where the ISPSL module improves representation learning by aligning image representations $f _ { i }$ with label text embeddings $U \ =$ $\{ \bar { u } _ { 1 } , . . . , u _ { C } \}$ (where $C$ is the number of classes) learned from the pre-trained CLIP model, i.e., $f _ { i } \mapsto u _ { i }$ . Meanwhile, the ISPSL module uses clustering to learn the cluster pro

In-Silo Prototypical Space Learning Cross-Silo Feature Space Alignment   
DL…Dogaobgels Textual Embedd𝓛i𝓛n𝑫g𝑪s𝑪 Local Local Spaces Local Consistency Matching 𝓛𝑪𝒍𝒊𝒆𝒏𝒕 心 C   
Cat M() Global 𝓛𝑨 𝓛𝑬   
D 𝑀𝐾(∙) 𝓛𝑬𝑴 ⨁ Projector 𝓛𝑳𝑪𝑴 𝐶𝑔(∙)   
Images Local Encoder Features Predicts Labels ： ： 0.3 0.8 0.6 𝓛𝑾𝑪𝑬   
FeFateuarteuspraecespace ProPtrotoytyppe  lLeearanrinigng SpSapcaceeCReocnosntsrtruction ⨁ o O 0 𝐻𝑔(∙) Global Classifier Predicts Labels   
∴ sterin 0..88 ： N 。 o O + VTVTrarararinaisnafsecnfrecere 。 0.3 Q 。 .40.4 GMaoudseslian Mg（) 0.8 0.4 0.3 𝓛𝑨 𝓛𝑬 Model O 。 。 ® 。 Q： Global Encoder Complementary Consistency Matching Clients Fill color denotes classes. Server Outline color denotes clients.

totype $\mu _ { k } ^ { i }$ , cluster variance $\Sigma _ { k } ^ { i }$ , and attention score $s _ { k } ^ { i }$ . Furthermore, the ISPSL module generates augmented features $f _ { a }$ based on the variance $\Sigma$ $( { \bar { \mu } } _ { k } ^ { i } \oplus \mathcal { N } ( 0 , { \bar { \Sigma } } ) \mapsto f _ { a } )$ , where $\mathcal { N } ( \cdot )$ is a Gaussian model. Subsequently, the CSFSA module leverages these augmented features to relearn the generalized projection $H _ { g } ( \cdot )$ and classifier $C _ { g } ( \cdot )$ on the server, i.e., $H _ { g } ( \mu _ { k } ^ { i } ) \approx H _ { g } ( \mu _ { K } ^ { i } ) .$ . And, FedFSA optimizes $H _ { g } ( \cdot )$ and $C _ { g } ( \cdot )$ based on alignment and classification status.

# Approach

This study proposes a cross-silo feature space alignment method (FedFSA) in federated learning, which alleviates the issue of ill-posed aggregation caused by imbalanced data across clients. As shown in Figure 2, FedFSA includes two modules: the in-silo prototypical space learning (ISPSL) module and the cross-silo feature space alignment (CSFSA) module. Specifically, the ISPSL module transfers variance knowledge from majority to minority class to calibrate feature distribution and provides feature information to the CSFSA module while preserving privacy. The CSFSA module aligns feature spaces from different sources to bridge feature gaps between clients, which forms a generalized model.

# In-Silo Prototypical Space Learning (ISPSL)

The ISPSL module aims to enhance representation learning and construct prototypical spaces on imbalanced data, which mitigates the impact of imbalances and provides image features for the CSFSA module, while preserving data privacy. However, data imbalance leads to a decline in the discriminability of minority class features, which compromises the effectiveness of cross-silo feature space alignment. To address this issue, the ISPSL module designs two processes: text-enhanced representation learning and variance transfer based space construction.

Text-Enhanced Representation Learning (TERL). To enhance the discriminability of minority class features, the TERL module uses a predefined space from pre-trained

CLIP (Radford et al. 2021) to regularize representation learning. Specifically, it employs supervised prototypical contrastive learning to align image feature $f _ { k } ^ { c }$ with textual embedding $u _ { c }$ in client $k$ . The loss $\mathcal { L } _ { D C }$ is defined as:

$$
\mathcal { L } _ { D C } = - \frac { 1 } { N _ { k } } \sum _ { i = 1 } ^ { N _ { k } } \log \frac { \sum _ { c = 1 } ^ { C } \mathbb { 1 } _ { y _ { k } ^ { c } = c } \exp ( f _ { k } ^ { c } \cdot u _ { c } / \tau ) } { \sum _ { c = 1 } ^ { C } \exp ( f _ { k } ^ { c } \cdot u _ { c } / \tau ) } ,
$$

where $u _ { c } ~ = ~ C L I P _ { \mathrm { t e x t } } ($ (’a photo of [the name of class c]’), $C L I P _ { \mathrm { t e x t } } ( \cdot )$ is the text encoder. $C$ denotes the quantity of classes. $y _ { k } ^ { c }$ is the label of $f _ { k } ^ { c }$ , $\mathbb { 1 } _ { \mathrm { T r u e } } = 1$ and $\mathbb { 1 } _ { \mathrm { F a l s e } } = 0$ , $N _ { k }$ is the number of training data of client $k$ . Meanwhile, the TERL module uses an empirical loss to ensure the discriminative capability of the model, i.e.,

$$
\begin{array} { r } { \mathcal { L } _ { E M } = - \frac { 1 } { N _ { k } } \sum _ { i = 1 } ^ { N _ { k } } \left( \sum _ { c = 1 } ^ { C } y _ { c } z _ { c } + \sum _ { c = 1 } ^ { C } y _ { c } \log \left( \sum _ { j = 1 } ^ { C } e ^ { z _ { j } } \right) \right) , } \end{array}
$$

where $z _ { c }$ represents the $\mathrm { ~  ~ c ~ }$ -th element in the model output vector. $y _ { c }$ is the ground-truth of image.

Variance Transfer based Space Construction (VTSC). To provide features for the CSFSA module in a privacypreserving manner, the VTSC module constructs the prototypical space. It sends augmented features to the server instead of the original features, which distinguishes it from existing methods (Chen et al. 2024; Yang et al. 2024b). Despite efforts to improve feature learning, intra-class variations and inter-class overlap create noise that disrupts prototype modeling. Therefore, the VTSC module first employs clustering (Meng, Tan, and Miao 2019; Meng, Tan, and Wunsch 2015; Qi et al. 2023) to mine patterns in the latent space within each class and evaluates the importance of each prototype,

$$
\begin{array} { r } { v _ { j } ^ { 1 } , . . . , v _ { j } ^ { N _ { v } } = \mathrm { c l u s t e r i n g } ( M _ { k } ( D _ { j } ^ { k } ) , N _ { v } ) , } \end{array}
$$

where $M _ { k } ( \cdot )$ is local model with frozen parameters and $D _ { j } ^ { k }$ denotes data of class $j$ in client $k$ . $N _ { v }$ is the number of clusters. Subsequently, the VSTR module calculates the mean $\mu _ { j } ^ { N _ { v } }$ and variance $\Sigma _ { j } ^ { N _ { v } }$ of features within clusters $\boldsymbol { v } _ { j } ^ { N _ { v } }$ , i.e.,

$$
\mu _ { j } ^ { N _ { v } } = \frac { 1 } { N } \Sigma _ { i = 1 } ^ { | \upsilon _ { j } ^ { N _ { v } } | } f _ { i } \in \upsilon _ { j } ^ { N _ { v } } ,
$$

$$
\begin{array} { r } { \Sigma _ { j } ^ { N _ { v } } = \frac { 1 } { | v _ { j } ^ { N _ { v } } | - 1 } { \sum _ { i = 1 } ^ { | v _ { j } ^ { N _ { v } } | } } \Big ( f _ { i } - \mu _ { j } ^ { N _ { v } } \Big ) \Big ( f _ { i } - \mu _ { j } ^ { N _ { v } } \Big ) ^ { T } , } \end{array}
$$

where $| \cdot |$ denotes the size of cluster. To reduce interference of outlier in model calibration, the VSTR module evaluates cluster significance using three factors: cluster size $( \rho )$ , compactness $( \sigma )$ , and minimum distance to other cluster centers $( \xi )$ . For cluster $\boldsymbol { v } _ { j } ^ { t }$ , $\rho _ { j } ^ { t } = | v _ { j } ^ { t } |$ , $\begin{array} { r } { \sigma _ { j } ^ { t } = \frac { 1 } { n } \sum _ { i = 1 } ^ { | \upsilon _ { j } ^ { t } | } \left\| f _ { i } - \mu _ { j } ^ { t } \right\| _ { 2 } . } \end{array}$ , $\xi _ { j } ^ { t } = \lvert { m i n } \{ \rvert \lvert { \mu - \mu _ { j } ^ { t } } \rvert \rvert _ { 2 } \}$ , where $\mu$ is the cluster
center o
f a different cla
ss than $v _ { j } ^ { \bar { t } } , f _ { i }$ is a feature of cluster $\boldsymbol { v } _ { j } ^ { t }$ . Inherently, the larger, more compact clusters farther from others are more important. Therefore, the importance score of cluster υjt is stj = ρj×tξj .

Moreover, data imbalance prevents minority class samples from adequately covering the underlying distribution, which is shown in long-tailed learning (Li et al. 2024a, 2021). Consequently, the VTSC module transfer variance from majority to minority classes, which calibrates the feature distribution on imbalanced data. Specifically, the VTSC module uses a Gaussian model $\mathcal { N } ( \cdot )$ to generate augmented features based on variance (Lindsay 1995; Nie et al. 2016). It fuses the variance $\Sigma _ { m a j }$ of a randomly selected majority class and other variances to transfer distribution knowledge to other features $\mu$ in a client, i.e.,

$$
\left\{ f _ { a } ^ { j } \right\} = \{ \mu + \Delta _ { j } \ | \ \Delta _ { j } \in \mathcal { N } \left( 0 , \Sigma _ { f u s e } \right) , j = 1 , . . . , J \} ,
$$

where $\mu$ is a local prototype. $J$ is the number of augmented features $\{ f _ { a } ^ { j } \} , \Sigma _ { f u s e } = ( \bar { 1 } - \kappa ) * \Sigma + \kappa * \Sigma _ { m a j }$ denotes fused variance. These augmented features share the same scores as their corresponding prototypes. And the VTSC module transmits all augmented features, their corresponding scores, and the local model from the client to the CSFSA module.

# Cross-Silo Feature Space Alignment (CSFSA)

The CSFSA module aims to learn a generalized model that fits the data from all clients. However, the inherent feature differences between different sources severely limit the performance of the retrained model. To address this problem, the CSFSA module employs cross-silo feature space alignment to map features from different sources into a unified space, which is used to bridge inconsistency between clients caused by data imbalance.

Specifically, the CSFSA module uses features reconstructed from the VTSC module to learn a generalized projection $H _ { g } ( \cdot )$ and classifier $C _ { g } ( \cdot )$ , which enables global model to realize the unified feature learning for samples with the same label across data sources. Specifically, it employs a dual-tiered regularization to refine the representation learning, including the Local Consistency Matching and the Complementary Consistency Matching.

For the Local Consistency Matching, it applies consistency in relationships between representations across different clients to guide learning process, which is expressed by

$$
\mathcal { L } _ { A } ( h _ { k } ^ { c _ { 1 } } , h _ { k } ^ { c _ { 2 } } , h _ { k } ^ { c _ { 3 } } ) = \| \mathcal { L } ( h _ { k } ^ { c _ { 1 } } , h _ { k } ^ { c _ { 2 } } , h _ { k } ^ { c _ { 3 } } ) - \mathcal { L } ( u ^ { c _ { 1 } } , u ^ { c _ { 2 } } , u ^ { c _ { 3 } } ) \| _ { 2 } ,
$$

where $h _ { k } ^ { c _ { i } } \ = \ H _ { g } ( \mu _ { k } ^ { c _ { i } } + \triangle )$ , if $\mu _ { k } ^ { c _ { i } }$ is a local prototype of class $c _ { i }$ in the client $k$ $, ~ \triangle ~ = ~ 0$ ; if $\mu _ { k } ^ { c _ { i } }$ is an aug

<html><body><table><tr><td>Datasets</td><td>#Class</td><td>#Training</td><td>#Testing</td><td>#Image Size</td></tr><tr><td>CIFAR10</td><td>10</td><td>50000</td><td>10000</td><td>32 * 32</td></tr><tr><td>CIFAR100</td><td>100</td><td>50000</td><td>10000</td><td>32 * 32</td></tr><tr><td>TinyImagenet</td><td>200</td><td>100000</td><td>10000</td><td>64 * 64</td></tr></table></body></html>

Table 1: Statistics of CIFAR10, CIFAR100 and TinyImagenet datasets used in the experiment.

mented feature, $\triangle \ = \ \mathcal { N } ( 0 , \Sigma _ { f u s e } )$ . $\begin{array} { r l } { \mathcal { L } ( h _ { k } ^ { c _ { 1 } } , h _ { k } ^ { c _ { 2 } } , h _ { k } ^ { c _ { 3 } } ) } & { { } = } \end{array}$   
$\left. \frac { h _ { k } ^ { c _ { 1 } } - h _ { k } ^ { c _ { 2 } } } { \left\| h _ { k } ^ { c _ { 1 } } - h _ { k } ^ { c _ { 2 } } \right\| _ { 2 } } \right.$ $\frac { h _ { k } ^ { c _ { 3 } } - h _ { k } ^ { c _ { 2 } } } { \left\| h _ { k } ^ { c _ { 3 } } - h _ { k } ^ { c _ { 2 } } \right\| _ { 2 } } \right. . \left.$ ·⟩ denotes dot product.

Meanwhile, to achieve rapid alignment within limited training epochs, distance-based consistency constraints are also applied, i.e.,

$\mathcal { L } _ { E } \big ( h _ { k } ^ { c _ { 1 } } , h _ { k } ^ { c _ { 2 } } \big ) = \big \| d i s t ( h _ { k } ^ { c _ { 1 } } , h _ { k } ^ { c _ { 2 } } ) - d i s t ( u ^ { c _ { 1 } } , u ^ { c _ { 2 } } \big ) \big ) \big \| _ { 2 } ,$ (8) where $d i s t ( \cdot )$ is an Euclidean distance. Overall, the local matching loss is defined by

$$
\begin{array} { r } { \mathcal { L } _ { L C M } = \sum _ { c _ { i } \in \cup _ { c = 1 } ^ { K } , k \in \cup _ { k = 1 } ^ { K } } ( \mathcal { L } _ { A } ( h _ { k } ^ { c _ { 1 } } , h _ { k } ^ { c _ { 1 } } , h _ { k } ^ { c _ { 1 } } ) + \mathcal { L } _ { E } ( h _ { k } ^ { c _ { 1 } } , h _ { k } ^ { c _ { 2 } } ) ) . } \end{array}
$$

For Complementary Consistency Matching, it uses complementary features from diverse sources to help the model learn consistent attributes across clients, which enables the model to transcend the limitations of a single perspective,

$$
\begin{array} { r } { \mathcal { L } _ { C C M } = \sum _ { c _ { i } \in \cup _ { c = 1 } ^ { K } , k _ { i } \in \cup _ { k = 1 } ^ { K } } ^ { k _ { 1 } \neq k _ { 2 } \neq k _ { 3 } } ( \mathcal { L } _ { A } ( h _ { k _ { 1 } } ^ { c _ { 1 } } , h _ { k _ { 2 } } ^ { c _ { 2 } } , h _ { k _ { 3 } } ^ { c _ { 3 } } ) + \mathcal { L } _ { E } ( h _ { k _ { 1 } } ^ { c _ { 1 } } , h _ { k _ { 2 } } ^ { c _ { 2 } } ) ) . } \end{array}
$$

Moreover, to enhance robustness and maintain decision boundaries, the CSFSA module uses importance scores $s _ { i }$ learned from clients to downweight lower-quality features, leading to a weighted supervised classification loss, i.e.,

$$
\begin{array} { r } { \mathcal { L } _ { W C E } = - \sum _ { i = 1 } ^ { N } s _ { i } \left( y _ { i } \log ( p _ { i } ) + ( 1 - y _ { i } ) \log ( 1 - p _ { i } ) \right) . } \end{array}
$$

# Training Strategies

FedFSA obtains the final model through the training of local client models and the calibration of the global model. It has following training strategies. First, FedFSA aims to calibrate local distributions on the client side, its optimization objective loss is defined as

$$
\mathcal { L } _ { C l i e n t } = \mathcal { L } _ { E M } + \alpha \mathcal { L } _ { D C } .
$$

Furthermore, FedFSA further minimizes distribution discrepancies across different spaces on the server side, the loss for optimization is characterized as

$\mathcal { L } _ { S e r v e r } = \mathcal { L } _ { W C E } + \eta ( \mathcal { L } _ { L C M } + \mathcal { L } _ { C C M } ) ,$ where $\alpha$ and $\eta$ are weighted parameters.

# Experiments

# Experiment Settings

Datasets. Following existing studies (Li, He, and Song 2021; Luo et al. 2021; Mu et al. 2023), experiments were conducted on three datasets, including CIFAR10 (Krizhevsky, Hinton et al. 2009), CIFAR100 (Krizhevsky, Hinton et al. 2009) and TinyImageNet (Le and Yang 2015) to validate the effectiveness of the FedFSA. The statistical details are presented in the Table 1. And the dataset is partitioned using the Dirichlet distribution with $\beta = 0 . 5$ .

Table 2: Performance comparison between FedFSA with existing methods on CIFAR10, CIFAR100 and TinyImagenet datasets.   

<html><body><table><tr><td colspan="2">Methods</td><td colspan="2">CIFAR10</td><td colspan="2">CIFAR100</td><td colspan="2">TinyImagenet</td></tr><tr><td>Base</td><td>FedAvg (AISTATS'17)</td><td>K=5 70.85</td><td>K=10 68.24</td><td>K=5</td><td>K=10</td><td>K=5</td><td>K=10</td></tr><tr><td></td><td>MOON (CVPR'21)</td><td>71.43</td><td>69.44</td><td>60.67 61.54</td><td>57.58 58.82</td><td>49.58 50.12</td><td>46.12 47.38</td></tr><tr><td rowspan="4">Methods based on Knowledge Distillation</td><td>FedProc (FGCS'23)</td><td>72.64</td><td>69.85</td><td>62.04</td><td>59.32</td><td>50.23</td><td>47.79</td></tr><tr><td>FedDeccor (ICLR'23)</td><td>72.11</td><td>70.21</td><td>61.59</td><td>59.24</td><td>49.75</td><td>47.63</td></tr><tr><td>FedETF (ICCV'23)</td><td>73.03</td><td>70.79</td><td>62.36</td><td>60.45</td><td>50.46</td><td>48.25</td></tr><tr><td>FedRCL (CVPR'24)</td><td>71.54</td><td>69.25</td><td>61.48</td><td>58.67</td><td>50.45</td><td>47.46</td></tr><tr><td rowspan="4">Methods based on Model Calibration</td><td>CCVR (NeurIPS'21)</td><td>71.25</td><td>69.67</td><td>60.67</td><td>58.59</td><td>49.67</td><td>46.23</td></tr><tr><td>FedCSPC (MM'23)</td><td>73.24</td><td>70.85</td><td>62.87</td><td>60.88</td><td>50.31</td><td>48.12</td></tr><tr><td>CLIP2FL (AAAI'24)</td><td>72.89</td><td>70.49</td><td>63.27</td><td>61.05</td><td>50.74</td><td>48.26</td></tr><tr><td>FedFSAFedAug (Ours)</td><td>74.45</td><td>72.35</td><td>64.48</td><td>62.41</td><td>51.05</td><td>48.73</td></tr><tr><td></td><td>FedFSAFedETF (Ours)</td><td>75.15</td><td>72.53</td><td>64.23</td><td>62.58</td><td>51.42</td><td>49.16</td></tr></table></body></html>

Evaluation Measures. Following existing studies (Li, He, and Song 2021; Mu et al. 2023), this study employs the Top1 Accuracy to evaluate the performance of methods, i.e.,

$$
\mathrm { A c c u r a c y } ~ = N _ { c o r r e c t } / N _ { t o t a l }
$$

where $N _ { c o r r e c t }$ , $N _ { t o t a l }$ are the number of correct predictions and total samples, respectively.

Network Architecture. Following existing studies (Li, He, and Song 2021; Mu et al. 2023), the network setup includes an image encoder, a projection head with a 2-layer MLP, and a classifier with a single-layer fully-connected network. We use a CNN with two 5x5 convolutional layers, $2 \mathrm { x } 2$ max pooling, and two ReLU-activated fully-connected layers as the encoder on CIFAR10 and use a ResNet18 encoder on other datasets, omitting its last fully-connected layer.

Implementation Details. Following existing studies (Li, He, and Song 2021; Mu et al. 2023), we set clients size $K =$ 5 and $K = 1 0$ in cross-silo settings, the local training epochs $E = 1 0$ , the batch size $B = 6 4$ , the communication round $T = 1 0 0$ for CIFAR10 and CIFAR100 datasets, $T = 5 0$ for TinyImagenet dataset, the learning rate $l r = 0 . 0 1$ and the weight decay $w d \ = \ 1 e \mathrm { ~ - ~ } 0 5$ in the SGD optimizer. The weighted parameter $\alpha = \{ 0 . 1 , 0 . 5 , 1 , 5 \}$ , the temperature $\tau = 0 . 5$ , the number of clusters $N _ { v } = \{ 1 , 2 , 3 \}$ . The weighted parameter $\eta = \{ 0 . 0 1 , 0 . 1 , 1 \}$ , $\kappa = \{ 0 . 3 , 0 . 5 , 0 . 7 \}$ , the number of augmented features $\textit { J } = \ \{ 1 , 2 , 4 , 8 \}$ . For other methods, we tuned their hyper-parameters by referring to corresponding papers for fair comparison.

# Performance Comparison

We compare FedFSA1 with nine state-of-the-art methods, including FedAvg (McMahan et al. 2017), MOON (Li, He, and Song 2021), CCVR (Luo et al. 2021), FedProc (Mu et al. 2023), FedDecorr (Shi et al. 2023), FedETF (Li et al. 2023), FedRCL (Seo et al. 2024), FedCSPC (Qi et al. 2023) and CLIP2FL (Shi et al. 2024). The following results can be observed from Table 2.

• FedFSA is a general framework that can combine various knowledge distillation based approaches, such as

FedAvg and FedETF, to bring them performance gains, which showcases its model-agnostic capability.

• Model calibration-based methods typically outperform knowledge distillation-based methods, as demonstrated by FedCSPC, CLIP2FL and FedFSA. This is because they all endeavor to utilize information from multiple sources to train a generalized model. • FedETF employs a unified simplex equiangular tight frame classifier often results in better outcomes than methods based on data-driven knowledge (FedProc, FedNTD, MOON). This may be due to they avoid issues of poor knowledge quality caused by data disparities and inherent limitations of the models themselves. • With an increase in the number of data sources, there is often a decline in the performance. This results from the amplified disparities across data distributions. FedFSA retains its superiority in performance, fully demonstrating the efficacy of its calibration mechanism.

# Ablation Study

This section explores the effectiveness of FedFSA’s components with $K = 5$ and $K \ : = \ : 1 0$ clients, and a Dirichlet parameter $\beta = 0 . 5$ . The results are shown in Table 3.

• The Text-Enhanced Representation Learning (TERL) module plays a crucial role, contributing an average performance gain of $1 . 2 \%$ to the baseline method, which verifies that providing unified guidance to different clients aids in enhancing their collaborative outcomes. • The collaboration between the Cross-Silo Feature Space Alignment (CSFSA) and TERL modules has resulted in a significant improvement in accuracy. This enhancement has provided an approximate $3 \%$ increase for the baseline methods across all cases. • The CSFSA module alone can also produce good results, as it mitigates the impact of outliers in a weighted manner compared to existing methods.

# In-depth Analysis

Robustness of FedFSA on Hyperparameters. This section evaluates the robustness of FedFSA in different hyperparameters. We select the $N _ { v }$ , weight parameters $\alpha$ , $\eta$ and $\kappa$ from $\{ 1 , 2 , 3 \}$ , $\{ 0 . 1 , 0 . 5 , 1 , 5 \}$ , $\{ 0 . 0 1 , 0 . 1 , 1 \}$ and $\{ 0 , 0 . 3 , 0 . 5 , 0 . 7 \}$ . As shown in Figure 3, FedFSA consistently outperforms FedAvg across various scenarios and demonstrates insensitivity to hyperparameter variations over a wide range, indicating its strong robustness in hyperparameter selection. Additionally, the model performs best with 2 clusters, as a single cluster may miss intra-class variability, while too many clusters could dilute key features and focus on noise. For $\alpha$ and $\eta$ , the model performs best when $\alpha = 2$ and $\eta = 0 . 1$ . This is because lower values of $\alpha$ or $\eta$ might result in the model assigning too little weight to key features, while higher values could lead to over-reliance on certain specific features, ignoring other valuable information. Notably, fusing an appropriate level of variance knowledge is beneficial, but excessive fusion may lead to inter-class feature overlap, introducing noise and resulting in degraded performance.

![](images/e289126b903cc64a190eefb38f6172bc0792dfd0505354c2ebcb45c395146469.jpg)

Figure 3: The impact of hyperparameters on performance.   
Table 3: Ablation study on the effectiveness of main modules of FedFSA on the CIFAR10 and CIFAR100 datasets.   

<html><body><table><tr><td></td><td colspan="2">CIFAR10</td><td colspan="2">CIFAR100</td></tr><tr><td></td><td>K=5</td><td>K=10</td><td>K=5</td><td>K=10</td></tr><tr><td>Base</td><td>70.85</td><td>68.24</td><td>60.67</td><td>57.58</td></tr><tr><td>+TERL</td><td>72.06</td><td>70.01</td><td>62.53</td><td>59.46</td></tr><tr><td>+TERL+CSFSA</td><td>73.21</td><td>71.02</td><td>63.26</td><td>61.03</td></tr><tr><td>+VTSC+CSFSA</td><td>73.63</td><td>71.21</td><td>63.55</td><td>61.14</td></tr><tr><td>+TERL+VTSC+CSFSA</td><td>74.45</td><td>72.35</td><td>64.48</td><td>62.41</td></tr></table></body></html>

The Effect of the Number of Augmented Features on Performance. This section discusses the effect of augmented feature numbers on calibration results. We adjust $N _ { a u g }$ from $\{ 0 , 1 , 2 , 4 , 8 \}$ with $N _ { v } = 2$ . $N _ { a u g } = 0$ means training with only local prototypes. Figure 4 shows the results. Increasing the number of augmented features generally improves performance by enriching the feature space and simulating real distributions, which helps prevent overfitting. Even a few augmented samples can boost performance by about $3 \%$ . However, performance on CIFAR100 declines when $N _ { v } = 8$ due to fewer samples per class and overlapping distributions, highlighting the importance of effective feature learning in complex tasks.

# Case Study

The Impact of Text-Enhanced Representation Learning on Feature Learning. This section evaluates the impact of Text-Enhanced Representation Learning on feature learning, prototype modeling, and test performance. We selected two clients with different data distributions and used t-SNE to visualize feature distribution for two classes in both training and testing sets. As shown in Figure 5, FedProc and FedFSA learned more discriminative representations compared to FedAvg, especially for majority classes (e.g., the sandy brown class). However, FedProc struggled with classes with few samples due to error accumulation across training rounds. FedFSA uses consistent features to guide local training, ensuring similar representations for shared classes, which helps it outperform other methods. Additionally, FedFSA evaluates prototype significance, assigning low weights to prototypes in overlapping regions of different classes, aiding the CSFSA module in reducing outlier interference.

![](images/606896d90a064669b439b550b5a066bae1eff955e08d930973f657b8e3bbbd1d.jpg)  
Figure 4: The effect of the number of augmented features $N _ { f } ~ = ~ \{ 0 , 1 , 2 , 4 , 8 \}$ on performance of FedFSA on CIFAR10 with different number of clients $K = \{ 5 , 1 0 \}$ .

![](images/cd07aa2ade3f43a64c5e80b6dabc1efdf45f6ab1d6f712095dcb126982a4325c.jpg)  
Figure 5: Local feature distributions learned by FedAvg, FedProc and FedFSA on CIFAR10 training and testing set.

Visualization Analysis of Cross-Silo Feature Space Alignment. In this section, we randomly selected two clients and two shared categories (birds with fewer samples and airplanes with more). Figure 6 shows the representation distributions, the CKA similarity (Kornblith et al. 2019; Gao et al. 2024; Liu et al. 2022), and model performance learned from FedCSPC and FedFSA methods. Results indicate that FedFSA learns more compact and discriminative representations within and between classes than FedCSPC. Additionally, FedFSA reduces feature space heterogeneity across clients even before calibration, aiding cross-source feature alignment. This improvement is also reflected in CKA similarity. Conversely, FedCSPC aligns the airplane class better than the bird class due to limited representation quality from minority samples. This is due to poor representation quality from minority samples hindering feature alignment. Furthermore, feature heterogeneity may decrease collaborative performance (see Figure 6(a)). And model calibration

Client 1 Bird 0.8 0.8   
0.76 0   
ient 2 irp 国 0.61 0   
ent 2 Bird   
Before 0.4 Before 0.4   
Calibration 0 0.52 Calibration 0 0.69   
airplane bird 0.0 airplane bird 0.0   
Client 2 Client 2   
Performance A𝐵i𝑖r𝑟p𝑑𝑐𝑐𝑙𝑖𝑙𝑖𝑒𝑒𝑛𝑡𝑡11: 0.52715 𝐵A𝑖i𝑟r𝑑p𝑐𝑙𝑖𝑒𝑛𝑡2 : 0.5335 𝐵A𝑖i𝑟r𝑑p𝑔𝑙𝑙𝑜𝑏𝑎𝑙𝑙: 0.52805 Performance A𝐵i𝑖r𝑟p𝑑𝑐𝑐𝑙𝑖𝑙𝑖𝑒𝑒𝑛𝑡𝑡11: 0.6320 𝐵A𝑖i𝑟r𝑑p𝑐𝑙𝑖𝑒𝑛𝑡2 : 0.3595𝐵A𝑖i𝑟r𝑑p𝑔𝑙𝑙𝑜𝑏𝑎𝑙𝑙: 0.647350   
0.74 0 0.8 0.81 0 0.8   
After 0.4 After U 0.4   
Calibration 0 0.54 Calibration 0 0.77   
0.0 0.0   
airplane Client 2 bird 安 airplane Client 2 bird   
Performance A𝐵i𝑖r𝑟p𝑑𝑐𝑐𝑙𝑖𝑙𝑖𝑒𝑒𝑛𝑡𝑡11 : 0.52935 𝐵A𝑖i𝑟r𝑑p𝑐𝑙𝑖𝑒𝑛𝑡2: 0.53405 𝐵A𝑖i𝑟r𝑑p𝑔𝑙𝑙𝑜𝑏𝑎𝑙𝑙: 0.63150 Performance A𝐵i𝑖r𝑟p𝑑𝑐𝑐𝑙𝑖𝑙𝑖𝑒𝑒𝑛𝑡𝑡11: 0.64805 𝐵A𝑖i𝑟r𝑑p𝑐𝑙𝑖𝑒𝑛𝑡2 : 0.643250 𝐵A𝑖i𝑟r𝑑p𝑔𝑙𝑙𝑜𝑏𝑎𝑙𝑙: 0.74155   
(a) FedCSPC (b) FedFSA

![](images/bfd502f87567a0ba758b484dd813ffff5c7f4b669f61f2ade3d77937b7a49453.jpg)  
Figure 6: Comparison of representation learning across clients between FedCSPC and FedFSA. FedFSA improves the effectiveness of cross-silo feature alignment for classes with minority samples, and enhances the performance of the global model.   
Figure 7: Error analysis. (a) FedFSA improves the local feature learning and the generalization of global model. (b) FedFSA can calibrate attention towards minority class sample, improving the performance of aggregation. (c) FedFSA failed due to unreliable local learning. (d) FedFSA narrows the gap between actual outcomes and top-1 predictions.

typically enhances the local models’ personalized capability, which leverages other clients’ knowledge to compensate for shortcomings.

Error analysis. This section uses GradCAM (Selvaraju, Cogswell, and et al 2017; Meng et al. 2019) to analyze FedFSA. As shown in Figure 7(a), ISRC and CSCRL enhance the base method by leveraging more samples, which strengthens discrimination ability through improved representation learning and cross-source feature alignment. Conversely, Figure 7(b) indicates that limited samples may cause base model failure, reducing collaboration effectiveness. TERL aids feature learning, enabling CSFSA to correct errors, but calibration may still fail (Figure 7(c)), as models struggle to focus on target regions despite accurate predictions. Figure 7(d) shows these methods often mispredict classes with few samples. The CSFSA module helps the model focus better, reducing prediction errors. These findings validate the impact of data imbalance in federated learning and validate the proposed framework’s effectiveness.

# Conclusion

To address the issue of ill-posed aggregation caused by data imbalance, this paper proposes a method (FedFSA) for aligning feature spaces across silos. FedFSA introduces the variance transfer technique to construct the prototypical space, which calibrates feature distribution of minority classes. Moreover, FedFSA aligns feature spaces from different sources to bridge inconsistency, fitting data from all clients to obtain a generalized model. Experimental results show that aligning the feature spaces of different clients can improve the performance of the retrained model.

Despite FedFSA mitigates the impact of data imbalance, there are still some directions worth exploring. Firstly, stronger strategies for representation alignment and causal discovery to enhance collaborative modeling (Chen et al. 2023a,b; Wang et al. 2022b,a; Lin et al. 2020; Yang et al. 2024a). Secondly, it makes sense to extend this to more challenging tasks, such as video classification (Wang et al. 2023b, 2024b) and recommendation systems (Ma et al. 2023; Meng et al. 2020).

# Acknowledgments

This work is supported in part by the Shandong Province Excellent Young Scientists Fund Program (Overseas) (Grant no. 2022HWYQ-048), the TaiShan Scholars Program (Grant no. tsqn202211289), the National Natural Science Foundation of China (NSFC) Joint Fund Key Project (Grant No. U2336211).