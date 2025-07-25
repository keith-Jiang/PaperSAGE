# Int\*-Match: Balancing Intra-Class Compactness and Inter-Class Discrepancy for Semi-Supervised Speaker Recognition

Xingmei Wang, Jinghan Liu, Jiaxiang Meng\*, Boquan Li, Zijian Liu

College of Computer Science and Technology, Harbin Engineering University, Harbin 150001, China {wangxingmei, liujinghan, mjxwjy, liboquan, 32185246}@hrbeu.edu.cn

# Abstract

Open-set speaker recognition is to identify whether the voices are from the same speaker. One challenge of speaker recognition is collecting large amounts of high-quality data. Based on the promising results of image classification, one intuitively feasible solution is semi-supervised learning (SSL) which uses confidence thresholds to assign pseudo labels for unlabeled data. However, we empirically demonstrated that applying SSL methods to speaker recognition is nontrivial. These methods focus solely on inter-class discrepancy as thresholds to select pseudo labels, overlooking intraclass compactness, which is particularly important for openset speaker recognition tasks. Motivated by this, we propose Int\*-Match, a semi-supervised speaker recognition method selecting reliable pseudo labels with intra-class compactness and inter-class discrepancy for speaker recognition. In particular, we use the inter-class discrepancy of labeled data as the threshold for pseudo-label selection and adjust the threshold based on the intra-class compactness of the pseudo labels dynamically and adaptively. Our systematic experiments demonstrate the superiority of $\mathbf { I n t ^ { * } }$ -Match, presenting an outstanding Equal Error Rate (EER) of $1 . 0 0 \%$ on the VoxCeleb1 original test set, which is merely $0 . 0 6 \%$ below the performance achieved by fully supervised learning.

# Code — https://github.com/LiuJinghan2001/IntMatch

# 1 Introduction

Speaker recognition is to recognize the identity of a speaker based on voices (Kinnunen and Li 2010; Lee et al. 2011). Due to unique pronunciation organs and speaking styles, including vocal tract shapes, larynx sizes, accents, and rhythm, each speaker possesses a distinctive voice, akin to fingerprints, enabling speaker identification (Bai and Zhang 2021; Irum and Salman 2019). Based on this characteristic, speaker recognition often performs open-set recognition tasks, where the cosine similarity between two non-enrolled category samples is evaluated to determine whether they belong to the same speaker. In this paper, we focus on studying the open-set speaker recognition task.

Utilizing extensive high-quality labeled data, existing methods normalize speaker embeddings onto a hypersphere and utilize margins to explicit decision boundaries between different classes (Xiang et al. 2019; Wang et al. 2018b; Deng et al. 2019). This approach optimizes cosine distance during training, enhancing inter-class discrepancy and intra-class compactness. However, obtaining high-quality labeled data is challenging, and the scarcity of such data leads to generalization issues (Ying 2019). Therefore, the lack of highquality labeled data is still the main challenge faced by existing efforts.

To address such issues, semi-supervised learning (SSL) (Zhu and Goldberg 2022) leverages abundant unlabeled data alongside a small portion of labeled ones during training, emerging as a reliable alternative to supervised learning. Specifically, existing SSL methods achieve satisfactory performance in image classification tasks, especially those state-of-the-art (SOTA) ones (Sohn et al. 2020; Zhang et al. 2021; Chen et al. 2023) assign pseudo labels (Lee et al. 2013) for unlabeled data based on confidence thresholds. The produced pseudo-label data plays the role of labeled ones to complete model training. The success of SSL indicates a prospective direction of devising effective semi-supervised speaker recognition models, towards addressing the labeled data lack issue.

We have empirically evaluated the SOTA threshold-based SSL methods (in Section 4.2) on the speaker recognition tasks, and find (1) these methods show limited success, (2) the utilization rate of pseudo labels is limited. That is, if the quality (correctness) of pseudo labels is over-focused, the quantity of the selected pseudo-label data is not enough to complete the recognition task. Moreover, in speaker recognition, enhancing both intra-class compactness and inter-class discrepancy is crucial for effectively performing open-set recognition tasks. However, we find that existing thresholdbased SSL methods select pseudo labels based solely on inter-class discrepancy as thresholds, neglecting intra-class compactness. Based on such intuitions, in this work, we propose an effective SSL method $\mathbf { I n t ^ { * } }$ -Match) for speaker recognition, towards selecting both high-quality and highquantity pseudo labels for unlabeled data with intra-class compactness and inter-class discrepancy. To be specific, our contributions in this work mainly include:

• We propose Int\*-Match, an SSL method to balance the quality and quantity of the selected pseudo labels for speaker recognition. The proposed approach takes both intra-class compactness and inter-class discrepancy into consideration which offers the potential of achieving better pseudo-data quality and labeled-data efficiency.

• Our systematic experiments demonstrate the excellent speaker recognition performance of Int\*-Match, which achieves the best EER of $1 . 0 0 \%$ on the VoxCeleb1 original test set that outperforms other baseline methods and is approximate to fully supervised learning.

# 2 Preliminary

In this section, we present the preparing knowledge and the motivations of this work, including the inter-class discrepancy and intra-class compactness in speaker recognition and the limitations of existing threshold-based SSL methods.

# 2.1 Preparing Knowledge

Let $\mathcal { D } _ { L } = \left\{ x _ { i } ^ { l } , y _ { i } \right\} _ { i = 1 } ^ { N _ { L } }$ and $\mathcal { D } _ { U } = \{ x _ { i } ^ { u } \} _ { i = 1 } ^ { N _ { U } }$ denote the labeled and unlabeled datasets, respectively, where $x _ { i } ^ { l }$ and $x _ { i } ^ { u }$ are the labeled and unlabeled training samples, and $y _ { i }$ is the corresponding ground-truth label for labeled data. We use $N _ { L }$ and $N _ { U }$ to represent the number of training samples in $\mathcal { D } _ { L }$ and ${ \mathcal { D } } _ { U }$ , respectively. For the labeled data, let $z _ { i } ^ { l } \in \mathbb { R } ^ { d }$ denote the $i$ -th speaker embedding of the input utterance $x _ { i } ^ { l }$ with data augmentation. For the most widely used classification loss, the softmax loss is calculated as follows:

$$
\mathcal { L } _ { s o f t m a x } ^ { l } = - \log \frac { e ^ { ( W _ { y _ { i } } ^ { T } z _ { i } ^ { l } + b _ { y _ { i } } ) } } { \sum _ { j = 1 } ^ { N } e ^ { ( W _ { j } ^ { T } z _ { i } ^ { l } + b _ { j } ) } } ,
$$

where $W _ { j }$ is the $j$ -th column of the last FC layer weight matrix, $W \in \mathbb { R } ^ { d \times N }$ , and $N$ is the number of class. $b _ { j } \in \mathbb { R } ^ { N }$ denotes the bias term, which is simplified to 0 in most cases.

While the softmax-based cross-entropy loss is effective for closed-set classification problems like image classification, where all possible classes are known during training, it fails to produce sufficiently discriminative embeddings for open-set recognition tasks (Deng et al. 2019). To handle this challenge, speaker recognition usually adopts cosine distance to estimate the similarities between pairs of speaker embeddings.

To directly optimize the cosine distance during training, $W$ and $z _ { i } ^ { l }$ are $l _ { 2 }$ -normalized and rescaled to $s$ , distributing the embeddings on a hypersphere with a radius of $s$ (Wang et al. 2017, 2018a; Deng et al. 2019):

$$
\mathcal { L } _ { N o r m } ^ { l } = - \log \frac { e ^ { s \cos \theta _ { i , y _ { i } } } } { \sum _ { j = 1 } ^ { N } e ^ { s \cos \theta _ { i , j } } } ,
$$

where $\delta$ denotes random augmentation, and $\cos _ { \delta } \theta _ { i , j }$ denotes the cosine similarity between the $i$ -th speaker embedding and $W _ { j } , j \in [ 1 , N ]$ .

To enhance intra-class compactness and inter-class discrepancy, existing methods apply an additive angular margin penalty to enforce a margin between the decision boundaries of different class centers. For the classic AAM-softmax (Xiang et al. 2019), the supervised loss is defined as follows:

$$
\mathcal { L } _ { A A M } ^ { l } = - \log \frac { e ^ { s \cos _ { \delta } ( \theta _ { i , y _ { i } } + m ) } } { e ^ { s \cos _ { \delta } ( \theta _ { i , y _ { i } } + m ) } + \sum _ { j = 1 , j \neq y _ { i } } ^ { N } e ^ { s \cos _ { \delta } \theta _ { i , j } } } .
$$

0.10.20.30.40.5Intra-Class Compactness 12.802468E-4 0.12345 right pl 2.02468E-4 ground-truth rwirgohnt gplpl   
0.0 Intra-Class Compactness ground-truth   
Inter-Class Discrepancy wrong pl 1.6E-4   
0k 10k 20k 30k 0k 10k 20k 30k Iter. Iter.   
(a) supervised learning (b) semi-supervised learning

For unlabeled data, where the ground-truth labels are unknown, we cannot directly compute the loss using Equation (3). Instead, threshold-based SSL methods first predict pseudo labels for the unlabeled data through the cosine prediction distribution, $\cos \theta _ { i , j } , j ~ \in ~ [ 1 , N ]$ . $\begin{array} { r l } { \hat { y _ { i } } } & { { } = } \end{array}$ arg $\operatorname* { m a x } ( \cos \theta ) _ { i , j } , j \ \in \ [ 1 , N ]$ represents the pseudo label of no-augmented unlabeled data. Subsequently, thresholdbased methods select pseudo labels by applying a threshold $\tau$ . For data with pseudo labels whose predicted scores exceed the threshold, we calculate unsupervised loss between strongly-augmented unlabeled data and its predicted pseudo label:

$$
\mathcal { L } _ { A A M } ^ { u } = - \beta _ { i } \log \frac { e ^ { s \cos _ { \phi } ( \theta _ { i , \hat { y _ { i } } } + m ) } } { e ^ { s \cos _ { \phi } ( \theta _ { i , \hat { y _ { i } } } + m ) } + \sum _ { j = 1 , j \neq \hat { y _ { i } } } ^ { N } e ^ { s \cos _ { \phi } \theta _ { i , j } } } ,
$$

where $\beta _ { i } ~ = ~ \mathbb { I } ( s o f t m a x ( \cos \theta ) _ { i , \hat { y } _ { i } } > \tau )$ , and $\phi$ denotes strong augmentation. The SSL models ultimately optimize the sum of supervised loss and unsupervised loss to complete the training of a batch.

# 2.2 Intra-Class Compactness and Inter-Class Discrepancy in Speaker Recognition

In supervised learning, margin-based softmax loss can enforce both intra-class compactness and inter-class discrepancy, as shown in Figure 1 (a). However, in SSL, prediction errors in pseudo labels are common, especially during the early stages of training. This leads to unreliable inter-class discrepancy and intra-class compactness. Therefore, it motivates us to re-examine these two properties.

Intuitively, intra-class compactness represents the angle between embedding $z _ { i }$ and the class center $W _ { y _ { i } }$ , which can be expressed through $\cos \theta _ { i , y _ { i } }$ . For a compact intra-class distribution, $z _ { i }$ should be clustered tightly around $W _ { y _ { i } }$ , exhibiting high cosine similarity. Therefore, for a batch of data, intra-class compactness can be defined as the average cosine similarity between the embeddings and their class centers:

$$
\gamma _ { \mathrm { { i n t r a } } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \cos \theta _ { i , y _ { i } } ,
$$

where $B$ denotes the batch size during training.

Inter-class discrepancy represents the differences in cosine similarity between $z _ { i }$ and different class centers $W _ { j }$ .

![](images/dc6ce94bf8d547303e47c0a0c9f262db26830584a2aab5245d14566fbd79b646.jpg)  
Figure 2: Overview of our method. During training, a batch of labeled data, no-augmented and strongly-augmented unlabeled data are given at the same time. $\mathrm { I n t ^ { * } }$ -Match selects reliable pseudo labels using the inter-class threshold (blue box) and restricts their intra-class compactness using the intra-class threshold (green box).

For a distribution with strong inter-class discrepancy, the cosine similarity between $z _ { i }$ and the corresponding class center $W _ { y _ { i } }$ should be significantly higher than with other class centers. Based on the above analysis, the softmax function can provide a more intuitive reflection of inter-class discrepancy. Therefore, for a batch of data, the inter-class discrepancy can be defined as the average of the cosine similarities between the $z _ { i }$ and the corresponding class center $W _ { y _ { i } }$ , scaled by the softmax function:

$$
\gamma _ { \mathrm { { i n t e r } } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } { s o f t m a x ( \cos \theta ) _ { i , y _ { i } } } .
$$

To obtain a more stable estimation, we aggregate $\gamma _ { \mathrm { i n t r a } }$ and $\gamma _ { \mathrm { i n t e r } }$ by employing Exponential Moving Average (EMA) with a momentum factor $m$ over previous batches:

$$
\begin{array} { r } { \gamma _ { \mathrm { i n t r a } _ { t } } = m \gamma _ { \mathrm { i n t r a } _ { t - 1 } } + ( 1 - m ) \gamma _ { \mathrm { i n t r a } } , } \\ { \gamma _ { \mathrm { i n t e r } _ { t } } = m \gamma _ { \mathrm { i n t e r } _ { t - 1 } } + ( 1 - m ) \gamma _ { \mathrm { i n t e r } } . } \end{array}
$$

# 2.3 Limitations of Threshold-based SSL Methods

Threshold-based SSL methods use fixed or dynamic thresholds to select pseudo labels. For example, the classic FlexMatch (Zhang et al. 2021) assigns different confidence thresholds to different classes based on their learning difficulties. The threshold is applied to the predicted scores of pseudo labels, usually normalized by the softmax function. Therefore, it only acts as a filter for inter-class discrepancy.

Figure 1 (b) shows the inter-class discrepancy and intraclass compactness of pseudo labels (pl) and ground-truth labels. It is observed that in the early stages of training, regardless of the accuracy of pseudo labels, the data exhibits high inter-class discrepancy and intra-class compactness within the assigned classes. This impedes the embeddings from forming a discriminative distribution with their ground-truth class vectors, ultimately affecting the performance of openset speaker recognition. We have visualized the embedding distributions of different methods in Section 4.2.

# 3 Methodology

In this section, we present our method Int\*-Match designed to address the limitations of threshold-based SSL methods by balancing intra-class compactness and inter-class discrepancy, as shown in Figure 2. Int\*-Match adaptively obtain inter-class and intra-class thresholds from labeled data. The inter-class threshold is used to select reliable pseudo labels, while the intra-class threshold evaluates their compactness. When their intra-class compactness exceeds the intra-class threshold, the inter-class threshold adaptively decreases to capture more pseudo labels, and the intra-class threshold adaptively increases to enhance compactness.

# 3.1 Inter-Class and Intra-Class Thresholds

In this section, we introduce inter-class and intra-class thresholds to balance intra-class compactness and inter-class discrepancy.

Similar to threshold-based SSL methods, the inter-class threshold $\tau _ { \mathrm { i n t e r } }$ is used to select pseudo labels. We refine the definition of $\gamma _ { \mathrm { i n t e r } }$ in Equation (6) by considering only correctly predicted labeled data to obtain a reliable $\tau _ { \mathrm { i n t e r } }$ :

$$
\gamma _ { \mathbf { i n t e r } } ^ { r i g h t } = \frac { 1 } { M } \sum _ { i = 1 } ^ { B } \zeta _ { i } \frac { e ^ { \cos _ { \delta } \theta _ { i , y _ { i } } } } { \sum _ { j = 1 } ^ { N } e ^ { \cos _ { \delta } \theta _ { i , j } } } ,
$$

where $\zeta _ { i } = \mathbb { I } ( \arg \operatorname* { m a x } ( \cos _ { \delta } \theta ) _ { i , j } = y _ { i } ) , j \in [ 1 , N ] . \ M$ is the number of correctly predicted labeled data in the batch, and $M$ must be greater than 0 when calculating the threshold for this batch. When computing $\tau _ { \mathrm { i n t e r } }$ at iteration $t$ , it is set to γinter t.

The intra-class threshold $\tau _ { \mathrm { i n t r a } }$ is used to control the intra-class compactness of pseudo labels. It is defined as the average of the maximum intra-class compactness of labeled data within each class. When computing $\tau _ { \mathrm { i n t r a } }$ at iteration $t$ , it is set to $\gamma _ { \mathbf { i n t r a } t } ^ { m a x }$ :

$$
\gamma _ { \mathbf { i n t r a } t } ^ { m a x } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \operatorname* { m a x } _ { ( 1 \leq i \leq t \times B ) } \bigl ( \mathbb { I } _ { ( j = y _ { i } ) } \cos _ { \delta } { \theta } _ { i , y _ { i } } \bigr ) .
$$

# 3.2 Int\*-Match

Int\*-Match utilizes the intra-class threshold to regulate the inter-class threshold, thereby balancing the inter-class discrepancy and intra-class compactness of the pseudo labels.

To achieve this, we first set a fixed intra-class threshold $\tau$ and initially rely solely on supervised learning to obtain a reliable inter-class threshold.

Subsequently, $\tau _ { \mathrm { i n t e r } }$ is used to select pseudo labels and compute the unsupervised loss like Equation (4). When the intra-class compactness of the selected pseudo labels exceeds $\tau _ { \mathrm { i n t r a } }$ , it indicates that the selected pseudo labels have achieved strong inter-class discrepancy and intra-class compactness. At this point, $\tau _ { \mathrm { i n t e r } }$ dynamically decreases to capture more pseudo labels, while $\tau _ { \mathrm { i n t r a } }$ dynamically increases to further enhance compactness:

$$
\begin{array} { r } { \tau _ { \mathrm { i n t e r } } = \tau _ { \mathrm { i n t e r } } - \left( \tau _ { \mathrm { i n t e r } } - \gamma _ { \mathrm { i n t e r } t } ^ { u s } \right) \times \alpha _ { t } , } \\ { \tau _ { \mathrm { i n t r a } } = \tau _ { \mathrm { i n t r a } } + ( \gamma _ { \mathrm { i n t r a } t } ^ { m a x } - \tau _ { \mathrm { i n t r a } } ) \times \alpha _ { t } , } \end{array}
$$

where $\gamma _ { \mathrm { i n t e r } t } ^ { u s }$ denotes the inter-class discrepancy of the unselected pseudo labels. $\alpha _ { t }$ is an adaptive scaling parameter, defined as the maximum value between the quantity of selected pseudo labels $q _ { t } \in [ 0 , 1 ]$ and their intra-class compactness γisntrat:

$$
\alpha _ { t } = \operatorname* { m a x } ( q _ { t } , \gamma _ { \mathbf { i n t r a } { t } } ^ { s } ) .
$$

During the early stages of training, $\tau _ { \mathrm { i n t e r } }$ is used to select fewer but reliable pseudo labels, so $\gamma _ { \mathbf { i n t r a } t } ^ { s }$ is primarily responsible for updating. As $\tau _ { \mathrm { i n t r a } }$ increases and $\tau _ { \mathrm { i n t e r } }$ decreases, $q _ { t }$ gradually assumes the role of updating.

Through Int\*-Match, the network can learn highquantity and high-quality pseudo labels by balancing their inter-class discrepancy and intra-class compactness.

# 4 Experiment

In this section, we perform systematical experiments to evaluate our proposed Int\*-Match. We first present the experimental setup, and further analyze our comparative experiments as well as ablation results respectively.

# 4.1 Experimental Setup

We start with presenting the datasets, implementations, baselines, and evaluation protocols of our experiments.

Dataset. We use the most typical VoxCeleb2 (Chung, Nagrani, and Zisserman 2018) for training, which comprises 1,092,009 utterances contributed by 5,994 speakers. On the other hand, we use the Original, Extended, and Hard VoxCeleb1 test sets (Nagrani, Chung, and Zisserman 2017; Nagrani et al. 2020) for evaluation. We follow the settings to threshold-based SSL methods (Zhang et al. 2021), selecting 2, 4, 10, and 20 utterances per class as labeled data, with the remaining data used as unlabeled data. It is worth noting that choosing 20 utterances per class represents $11 \%$ of the training dataset. Furthermore, additional experiments are conducted in Table 2 by selecting $20 \%$ , $30 \%$ , $40 \%$ , and $50 \%$ of utterances from each class proportionally, enabling comparisons with fully supervised learning. What’s more, we use MUSAN (Snyder, Chen, and Povey 2015) and RIR (Ko et al. 2017) datasets for data augmentation.

Implementation. For a fair comparison, we adopt an identical training strategy across the SSL methods: we employ the popular speaker recognition model ECAPATDNN (Desplanques, Thienpondt, and Demuynck 2020) as the network with a channel size of 1024, and the input is an 80-dimensional logarithmic mel spectrum extracted from 2-second speech segments. Meanwhile, the output is a 192- dimensional speaker embedding. The labeled batch size is set to 150, and the unlabeled batch size is the same, with a total training step of $5 6 0 \mathrm { k }$ . The network parameters are optimized by Adam optimizer (Kingma and Ba 2015), where the initial learning rate is set to 0.001, which decreases $3 \%$ in every $^ { 7 \mathrm { k } }$ iterations, roughly one epoch of unlabeled data. We use AAM-softmax loss as the loss function, with the margin as 0.2 and the scale as 30. We assess the maximum inter-class discrepancy per batch on VoxCeleb2 under fully supervised learning. The average of these values increases to $3 . 8 4 \times 1 0 ^ { - 4 }$ and stabilizes. Consequently, the confidence threshold for threshold-based SSL methods is set to $3 . 4 6 \times 1 0 ^ { - 4 }$ $( 3 . 8 4 \times 1 0 ^ { - 4 } \times 0 . 9 )$ ).

To improve data diversity, we apply strong and random augmentation techniques similar to those used for XVectors (Snyder et al. 2018), except that the unlabeled data used for predicting pseudo labels is not subjected to augmentation. The difference between strong and random augmentation is that random augmentation may apply no augmentation with some probability.

For Int\*-Match, we set $m$ and $\tau$ to 0.999 and 0.65, respectivly. Experiments with different $\tau$ will be shown in the ablation study (Section 4.3).

Baseline method. We first apply multiple SOTA threshold-based SSL methods in image classification as baselines, including Pseudo label (Lee et al. 2013), FixMatch (Sohn et al. 2020), FlexMatch (Zhang et al. 2021), Dash (Xu et al. 2021), FreeMatch (Wang et al. 2023), and SoftMatch (Chen et al. 2023). These methods rely on threshold-based pseudo labeling and achieve excellent performance in image classification.

Additionally, we refer to the results of SOTA SSL methods in speaker recognition, including GCL (Inoue and Goto 2020), GCN (Tong et al. 2022) and GLL (Wang et al. 2024).

Evaluation protocol. We evaluate all methods using equal error rate (EER) and minimum detection cost function (minDCF, set $P _ { t a r g e t } = 0 . 0 5$ , which are common metrics to evaluate the performance of speaker recognition (Kinnunen and Li 2010). Furthermore, we conduct a qualitative analysis to compare the quality and quantity of pseudo labels selected by SSL methods, providing further evidence of the efficacy of our approach. The quantity of pseudo labels shows the proportion selected by the inter-class threshold relative to the total unlabeled data. The quality of pseudo labels reflects the proportion of correctly predicted pseudo labels among those selected.

Table 1: Performance in $\mathrm { E E R } ( \% )$ and minDCF of SOTA methods and proposed Int\*-Match on the VoxCeleb1 test sets. The experimental setups of GCL and GCN differ from ours: GCL selected 899 speakers ( $1 5 \%$ of VoxCeleb2) as labeled data, while GCN used the VoxCeleb1 dev set (equivalent to $14 \%$ of VoxCeleb2) as labeled data. Both methods involve more labeled data than our setting of selecting 20 samples per class ( $11 \%$ of VoxCeleb2).   

<html><body><table><tr><td>Evaluation set</td><td></td><td colspan="4">VoxCeleb1-O</td><td colspan="4">VoxCeleb1-E</td><td colspan="4">VoxCeleb1-H</td></tr><tr><td>#Label</td><td></td><td>2</td><td>4</td><td>10</td><td>20</td><td>2</td><td>4</td><td>10</td><td>20</td><td>2</td><td>4</td><td>10</td><td>20</td></tr><tr><td rowspan="9">FixMatch (Sohn et al.2020)</td><td rowspan="9">Supervised learning Pseudo label (Lee et al.2013)</td><td rowspan="9">8.68 14.43 8.86</td><td rowspan="9">5.98 10.56</td><td rowspan="9">3.45 10.45</td><td rowspan="9">2.14 4.24</td><td rowspan="9">9.10 14.91</td><td rowspan="9">6.24</td><td colspan="7">3.86</td><td>6.23 4.25</td></tr><tr><td></td><td>11.09</td><td>2.36</td><td>13.34 19.10</td><td>9.74 14.77</td><td></td><td>6.50</td></tr><tr><td>3.38</td><td></td><td>6.36</td><td>10.64 3.44</td><td>4.08 2.32</td><td>13.23</td><td>9.84</td><td>13.84 5.86</td><td>4.14</td></tr><tr><td>6.23 1.52</td><td>2.22 1.38</td><td>9.11 2.06 8.65</td><td>1.74</td><td>1.69</td><td>1.57</td><td>3.62</td><td>3.15</td><td>3.02 2.91</td></tr><tr><td>4.08</td><td>1.44 2.51 2.20</td><td>1.91</td><td>4.40</td><td>2.68</td><td>2.09</td><td>12.70</td><td>4.64</td><td>3.76</td></tr><tr><td>8.25 FreeMatch (Wang et al.2023) 2.55 2.00</td><td>2.49</td><td>1.79</td><td>2.90 2.37</td><td>2.69</td><td>2.27 1.86</td><td>5.06</td><td>7.10 4.74</td><td>4.06</td></tr><tr><td>SoftMatch (Chen et al.2023)</td><td>1.95</td><td>1.87</td><td>1.50</td><td>2.14</td><td>2.06</td><td>1.75 4.14</td><td>3.85</td><td>3.71</td><td>3.12</td></tr><tr><td>GCL (Inoue and Goto 2020)</td><td>1.30</td><td>6.01</td><td></td><td></td><td>1</td><td></td><td></td></tr><tr><td colspan="10">GLL (Wang et al. 2024)</td></tr><tr><td colspan="10"></td><td colspan="7"></td></tr><tr><td colspan="7">Int*-Match</td><td colspan="7"></td></tr><tr><td>Supervised learning</td><td colspan="10">1.45</td><td colspan="3">3.44 3.16</td><td colspan="3">2.99 2.81</td></tr><tr><td rowspan="7"></td><td></td><td colspan="10">0.476</td><td colspan="7"></td></tr><tr><td colspan="10"></td><td colspan="7">0.153 0.613 0.496</td></tr><tr><td colspan="10">Pseudo label (Lee et al. 2013)</td><td colspan="7">0.410</td></tr><tr><td colspan="10">FixMatch (Sohn et al.2020)</td><td colspan="7">0.711 0.595 0.149</td></tr><tr><td colspan="10">FlexMatch (Zhang et al.2021)</td><td colspan="7">0.616 0.499 0.180</td></tr><tr><td colspan="10">Dash (Xu et al. 2021)</td><td colspan="7">0.101 0.212 0.188</td></tr><tr><td colspan="10">FreeMatch (Wang et al.2023)</td><td colspan="7">0.135 0.591 0.379</td></tr><tr><td colspan="10">SoftMatch (Chen et al.2023) Int*-Match</td><td colspan="7">0.117 0.284 0.269</td></tr></table></body></html>

0.2468 1.0 FixMatch 0.8 FixMatch FlexMatch 0.6 FlexMatch FreeMatch FreeMatch SDoafsthMatch Int\*-Match 0.02 Int\*-Match SDoafsthMatch 200k 400k 600k 0k 200k 400k 600k Iter. Iter. (a) Quality with 20 labels per class (b) Quantity with 20 labels per class 01.80 1.0 FlexMatch 0.8 FlexMatch 0.46 0.6 0.6 0.6 0.625 0.625 0.02 0.6575 0.7 0.02 0.6575 0.7 200k 400k 600k 200k 400k 600k Iter. Iter. (c) Quality with 2 labels per class (d) Quantity with 2 labels per class

# 4.2 Comparative Experiment

In the following, we evaluate and compare the performance of Int\*-Match as well as other baseline methods.

Comparison with baselines. In Table 1, we compare various SOTA methods as baselines with our proposed method. First, it is observed that Int\*-Match almost outperforms the SOTA threshold-based SSL methods in terms of EER and minDCF on VoxCeleb1. Second, compared to the SOTA methods in semi-supervised speaker recognition, Int\*-Match still shows better performance with fewer labels. Specifically, with 20 labels per class $3 \%$ fewer labeled data compared to GCN), Int\*-Match achieves an EER of $1 . 2 8 \%$ , which is comparable to GCN’s $1 . 3 0 \%$ on VoxCeleb1-O.

Comparison with fully supervised learning. In Table 2, we use ECAPA-TDNN as the backbone model and compare SoftMatch, FlexMatch, and Int\*-Match to supervised learning and fully supervised learning. Int\*-Match achieves the best performance, narrowing the average gap to $0 . 1 1 \% / 0 . 6 \%$ in EER/minDCF with fully supervised learning.

Qualitative analysis. We provide a qualitative comparison of the threshold-based SSL methods with 20 labels per class, as shown in Figure 3 (a) and (b). It is first observed that Int\*-Match consistently obtains high-quality pseudo labels with accuracy close to $100 \%$ across the training. Moreover, compared to FlexMatch, our method adjusts interclass thresholds through intra-class compactness, which allows for a more reasonable selection of high-quality and high-quantity pseudo labels than strategies based on learning difficulty, thus enhancing the discriminative power of the speaker embeddings.

Table 2: Performance in $\mathrm { E E R } ( \% )$ and minDCF of different SSL methods and fully supervised learning on the VoxCeleb1 tes sets. The performance of fully supervised learning is based on our settings and may differ slightly from the original paper.   

<html><body><table><tr><td>Evaluation set</td><td></td><td colspan="4">VoxCeleb1-O</td><td colspan="4">VoxCeleb1-E</td><td colspan="4">VoxCeleb1-H</td></tr><tr><td colspan="2">#Label</td><td>20%</td><td>30%</td><td>40%</td><td>50%</td><td>20%</td><td>30%</td><td>40%</td><td>50%</td><td>20%</td><td>30%</td><td>40%</td><td>50%</td></tr><tr><td rowspan="5">EEE</td><td>Supervised learning</td><td>1.64</td><td>1.42</td><td>1.40</td><td>1.33</td><td>1.78</td><td>1.61</td><td>1.56</td><td>1.55</td><td>3.21</td><td>2.93</td><td>2.84</td><td>2.76</td></tr><tr><td>SoftMatch (Chen et al. 2023)</td><td>1.66</td><td>1.59</td><td>1.86</td><td>1.75</td><td>1.79</td><td>1.79</td><td>2.00</td><td>1.86</td><td>3.18</td><td>3.12</td><td>3.47</td><td>3.20</td></tr><tr><td>FlexMatch (Zhang et al. 2021)</td><td>1.29</td><td>1.27</td><td>1.21</td><td>1.24</td><td>1.49</td><td>1.52</td><td>1.39</td><td>1.44</td><td>2.72</td><td>2.72</td><td>2.58</td><td>2.59</td></tr><tr><td>Int*-Match</td><td>1.22</td><td>1.08</td><td>1.00</td><td>1.10</td><td>1.44</td><td>1.36</td><td>1.33</td><td>1.34</td><td>2.62</td><td>2.51</td><td>2.44</td><td>2.48</td></tr><tr><td>Fully supervised learning</td><td colspan="4">0.94</td><td colspan="4">1.22</td><td colspan="4">2.29</td></tr><tr><td rowspan="5"></td><td>Supervised learning</td><td>0.116</td><td>0.099</td><td>0.096</td><td>0.095</td><td>0.115</td><td>0.104</td><td>0.102</td><td>0.098</td><td>0.192</td><td>0.179</td><td>0.173</td><td>0.168</td></tr><tr><td>SoftMatch (Chen et al.2023)</td><td>0.110</td><td>0.110</td><td>0.130</td><td>0.112</td><td>0.115</td><td>0.113</td><td>0.127</td><td>0.118</td><td>0.188</td><td>0.184</td><td>0.202</td><td>0.188</td></tr><tr><td>FlexMatch (Zhang et al. 2021)</td><td>0.087</td><td>0.097</td><td>0.080</td><td>0.088</td><td>0.096</td><td>0.097</td><td>0.091</td><td>0.092</td><td>0.165</td><td>0.169</td><td>0.157</td><td>0.159</td></tr><tr><td>Int*-Match</td><td>0.086</td><td>0.081</td><td>0.075</td><td>0.079</td><td>0.093</td><td>0.088</td><td>0.085</td><td>0.087</td><td>0.164</td><td>0.156</td><td>0.150</td><td>0.153</td></tr><tr><td>Fully supervised learning</td><td colspan="4">0.070</td><td colspan="4">0.080</td><td colspan="4">0.143</td></tr></table></body></html>

Visualization of intra-class compactness and interclass discrepancy. To differentiate from the speakers in the training set, we use data from ten classes in VoxCeleb1 to visualize embedding distributions for different methods, as shown in Figure 4. This helps us evaluate the intraclass compactness and inter-class discrepancy of speaker embeddings for open-set speaker recognition. The figure shows that Int\*-Match and fully supervised learning exhibit strong intra-class compactness and inter-class discrepancy for unknown classes, outperforming FlexMatch. Int\*- Match achieves high-quality embedding distributions by balancing these two properties with limited labeled data.

Table 3: Performance in $\mathrm { E E R } ( \% )$ on the VoxCeleb1 test sets with different $\tau$ for ablation study.   

<html><body><table><tr><td rowspan="2">T</td><td colspan="5">VoxCeleb1-O VoxCeleb1-E VoxCeleb1-H</td></tr><tr><td>2</td><td>10</td><td>2 10</td><td>2</td><td>10</td></tr><tr><td>0.6</td><td>1.55</td><td>1.47</td><td>1.93</td><td>1.61 3.50</td><td>2.98</td></tr><tr><td>0.625</td><td>1.56</td><td>1.40</td><td>2.00 1.65</td><td>3.62</td><td>2.99</td></tr><tr><td>0.65</td><td>1.45</td><td>1.38</td><td>1.89</td><td>1.63 3.44</td><td>2.99</td></tr><tr><td>0.675</td><td>1.55</td><td>1.48</td><td>1.93 1.73</td><td>3.53</td><td>3.17</td></tr><tr><td>0.7</td><td>1.71</td><td>1.60</td><td>2.05</td><td>1.85 3.72</td><td>3.32</td></tr></table></body></html>

# 4.3 Ablation Study

We perform ablation studies to explore the effect of different initial intra-class thresholds $\tau$ . First, as shown in Table 3, the best performance in most settings is achieved when $\tau$ is set to 0.65. Performance remains relatively robust across most values of $\tau$ . Second, as shown in Figure 3 (c) and (d), as the initial intra-class threshold decreases, the inter-class threshold progressively lowers the quality of selected pseudo labels, causing the method to perform similarly to FlexMatch. Conversely, when the initial intra-class threshold is too high, the inter-class threshold does not adequately select a sufficient quantity of pseudo labels. The suitable range for selecting $\tau$ can be refined by analyzing the cosine values at the model’s decision boundary.

![](images/c9a5415aaa0fb018fb9afaed35d00207f63d93026043c4635a6d6d57a4552d7a.jpg)  
Figure 4: Visualization of embedding distributions for 10 classes from VoxCeleb1 using t-SNE for (a) Supervised learning, (b) FlexMatch, (c) Int\*-Match with 2 labels per class, and (d) Fully supervised learning.

# 5 Conclusion

This work proposes a novel semi-supervised speaker recognition method, Int\*-Match, designed to leverage inter-class discrepancy and intra-class compactness to select highquality and high-quantity pseudo labels. It utilizes the interclass threshold to select high-quality pseudo labels while employing the intra-class threshold to enhance the compactness of selected pseudo labels and increase their quantity. Experimental results show that Int\*-Match is superior to the SOTA methods and obtains reliable pseudo labels by balancing intra-class compactness and inter-class discrepancy.

# 6 Related Work

In this section, we review related work around speaker recognition and semi-supervised learning.

# 6.1 Speaker Recognition

Currently, deep learning has significantly advanced the field of speaker recognition by effectively extracting highly abstract embedding features from utterances (Snyder et al. 2018; Mun et al. 2020; Inoue and Goto 2020), surpassing traditional methods. We categorize deep learning-based speaker recognition research into three groups, from the perspective of labeled data requirements, i.e., (1) supervised learning-based, (2) self-supervised learning-based, and (3) SSL-based methods.

First, based on enough labeled data, supervised learningbased methods have consistently demonstrated improved performance in speaker recognition over the past few years. In (Variani et al. 2014), a DNN is trained at the frame level to classify speakers, and utilized to extract speaker-specific features for enrollment. Snyder et al. (Snyder et al. 2018) propose to use a time-delay neural network (TDNN) to extract the frame-level features from utterances. Then, a temporal aggregation layer is proposed to aggregate the framelevel features into fixed-length utterances-level representations. Some FC layers are next used to produce classification results. Among existing supervised learning-based methods, TDNN (Desplanques, Thienpondt, and Demuynck 2020; Liu et al. 2022; Mun et al. 2023; Thienpondt, Desplanques, and Demuynck 2021) and ResNet (Zeng et al. 2022; Zhou, Zhao, and $\mathrm { W u } 2 0 2 1$ ) report the best performance.

Second, self-supervised learning methods utilize extensive amounts of unlabeled data, circumventing the limitation posed by the need for labeled data. Given the absence of manual annotations, a common approach in speaker recognition involves employing contrastive learning to acquire meaningful speech representations (Zhang, Zou, and Wang 2021; Zhang and Yu 2022; Sang et al. 2022). For example, Kang et al. (Kang et al. 2022) propose an Augmentation Adversarial Training (AAT) strategy, enabling the network to be speaker-discriminative while remaining invariant to the applied augmentations. Cai et al. (Cai, Wang, and Li 2021) incorporate a second-stage contrastive learning model that generates pseudo labels for unlabeled data using clustering algorithms. Tao et al. (Tao et al. 2022) present a loss-gated learning strategy within a two-stage framework, selecting reliable pseudo labels with a fixed threshold for each iteration. Han et al. (Han, Chen, and Qian 2022, 2024) utilize a Gaussian mixture model to fit data loss, which assigns dynamic thresholds to select pseudo labels.

Third, SSL methods utilize a small amount of labeled data along with a substantial pool of unlabeled data for model training. For example, Inoue et al. (Inoue and Goto 2020) introduce a contrastive learning framework utilizing Generalized Contrastive Loss (GCL) for text-independent speaker verification. Kreyssig et al. (Kreyssig and Woodland 2020) present a Cosine-Distance Virtual Adversarial Training (CD-VAT) approach to tackle speaker recognition tasks. Chen et al. (Chen, Ravichandran, and Stolcke 2021) propose an SSL approach integrating label propagation, primarily focusing on enhancing recognition performance through label inference. Tong et al. (Tong et al. 2022) perform speaker recognition tasks based on the graph convolutional network (GCN), leveraging pseudo-label clustering for unlabeled data. Wang et al. (Wang et al. 2024) propose a twostage SSL framework, which uses a Gated Label Learning (GLL) strategy to select reliable pseudo-label data.

# 6.2 SSL in Image Classification

We further explore SSL, referring to its application in image classification domains. Specifically, existing work is mainly divided into three groups, i.e., consistency regularizationbased, entropy-minimization-based, and holistic methods.

First, consistency regularization performs SSL by encouraging a model to provide consistent predictions for perturbed versions of the same input data, even when only a small portion of the data is labeled. For example, Laine et al. (Laine and Aila 2017) introduce two notable approaches: Π-Model and Temporal Ensembling, which utilize consistency regularization to promote consistent predictions across variations of input data. Tarvainen et al. (Tarvainen and Valpola 2017) propose Mean Teacher that minimizes the divergence between different augmented outputs as well as quickly integrates information by using a teacher model to mitigate confirmation bias. Miyato et al. (Miyato et al. 2018) propose Virtual Adversarial Training (VAT) that achieves consistency regularization by focusing on perturbations that have the most significant impact on the model’s predictions.

Second, entropy minimization minimizes the entropy of the prediction function, encouraging models to provide confident predictions. For example, Lee et al. (Lee et al. 2013) propose to leverage the class with the highest prediction probability from unlabeled data as pseudo labels, which enhances learning by incorporating confident predictions from unlabeled datasets. Rizve et al. (Rizve et al. 2021) introduce an Uncertainty-aware Pseudo-label Selection (UPS) framework to refine pseudo labels by reducing noise, thereby improving the utilization rate of unlabeled data. Pham et al. (Pham et al. 2021) propose a meta pseudo labels approach, which employs a teacher network to guide the learning of a student network, and optimizes learning progress based on the performance feedback of the student.

Third, holistic methods that integrate consistency regularization and entropy minimization have demonstrated notable performance improvements. Sohn et al. (Sohn et al. 2020) propose FixMatch, which calculates the loss between pseudo labels of weakly-augmented samples that exceed a specified confidence threshold and their strongly augmented counterparts. Zhang et al. (Zhang et al. 2021) introduce FlexMatch, which considers variations in learning difficulties among different classes and implements a Curriculum Pseudo Label (CPL) method to encourage the selection of pseudo labels. Chen et al. (Chen et al. 2023) propose SoftMatch that employs a truncated Gaussian function to assign weights to unlabeled data, providing a balanced learning approach. Wang et al. (Wang et al. 2023) propose the selfadaptive threshold to adjust the threshold in a self-adaptive manner according to the model’s learning status.