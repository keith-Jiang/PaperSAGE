# Ambiguous Instance-Aware Contrastive Network with Multi-Level Matching for Multi-View Document Clustering

Zhenqiu Shu, Teng Sun, Yunwei Luo , Zhengtao Yu\*

Faculty of Information Engineering and Automation, Kunming University of Science and Technology shuzhenqiu $@$ 163.com, st251100@163.com, lywinaaa $@$ gmail.com, ztyu@hotmail.com

# Abstract

Multi-view document clustering (MvDC) aims to improve the accuracy and robustness of clustering by fully considering the complementarity of different views. However, in real-world clustering applications, most existing works suffer from the following challenges: 1) They primarily align multi-view data based on a single perspective, such as features and classes, thus ignoring the diversity and comprehensiveness of representations. 2) They treat each instance equally in cross-view contrastive learning without considering ambiguous ones, which weakens the model’s discriminative ability. To address these problems, we propose an ambiguous instance-aware contrastive network with multi-level matching (AICN-MLM) for MvDC tasks. This model contains two key modules: a multi-level matching module and an ambiguous instance-aware contrastive learning module. The former attempts to align multi-view data from different perspectives, including features, pseudo-labels, and prototypes. The latter dynamically adjusts instance weights through a weight modulation function to highlight ambiguous instance pairs. Thus, our proposed method can effectively explore the consistency of multi-view document data and focus on ambiguous instances to enhance the model’s discriminative ability. Extensive experimental results on several multi-view document datasets verify the effectiveness of our proposed method.

Traditional MVC methods (Wang et al. 2022b; Li et al. 2023; Shu et al. 2023) usually use shallow machine learning techniques to cluster multi-view data. For example, matrix factorization-based MVC methods (Zhao, Ding, and Fu 2017; Wang, Zhang, and Gao 2018) learn latent representations by decomposing the original multi-view data matrix. Graph-based MVC methods (Li et al. 2021; Huang et al. 2021) exploit the structural information of multi-view data by learning graph representations and then performing spectral clustering algorithms. Subspace-based MVC methods (Li et al. $2 0 1 9 \mathrm { a }$ ; Yin, Wu, and Wang 2015) aim to learn a low-dimensional subspace from original high-dimensional data that can better reveal the intrinsic structure. However, due to the limited representation ability of shallow learning methods, they usually perform poorly in complex scenarios such as multilingual document clustering.

# Introduction

In this era of rapid information development, data often originates from various sources, such as news organizations reporting the same event in different languages. Many studies treat each language document as a separate view, utilizing multi-view analysis techniques to explore semantic information from multilingual documents. Compared with singleview data, multi-view data offers a more comprehensive understanding and usually achieves better results in downstream tasks. Multi-view clustering (MVC) (Chao, Sun, and Bi 2021) attempts to reveal hidden data patterns and group structures by integrating information from multi-view data. Existing MVC methods are divided into two offers: traditional MVC methods and deep MVC methods.

Recently, deep MVC (DMVC) technology has attracted increasing attention due to the excellent feature representation ability of deep neural networks (DNNs). DMVC methods based on shared representation learning (Du et al. 2021; Xu et al. 2021a) learn nonlinear feature representations between different views by using autoencoders while maintaining consistency in the latent space through parameter sharing and reconstruction loss. DMVC methods based on self-supervised learning (Xu et al. 2022a; Xia et al. 2021) usually design self-supervised losses to optimize the autoencoder network jointly. GCN-based DMVC methods (Zhao, Yang, and Nie 2023; Cheng et al. 2021) use GCN to automatically learn node feature representations, thus effectively capturing high-order structural relationships between multiview data. GAN-based DMVC algorithms (Li et al. 2019b; Shu et al. 2024) try to align the feature distributions of multiview data through generators and discriminators. Most existing DMVC methods utilize contrastive learning techniques to explore the consistency of multi-view data by maximizing the similarity between positive pairs and minimizing it between negative pairs in the latent space (Xu et al. 2022b; Chen et al. 2023). However, they treat all positive and negative instance pairs equally, without considering that some instance pairs are easier to classify while others are more challenging and ambiguous.

Although the effectiveness of these techniques has been demonstrated in several real-world applications, they suffer from the following challenges: 1) The distribution of multiview data may be biased due to its multi-source attributes. Moreover, most existing methods only align multi-view data from the instance or cluster level. Therefore, they cannot provide a comprehensive alignment strategy across different views; 2) Previous contrastive learning approaches treat all instances equally and overlook ambiguous instances that are difficult to classify, thus limiting the model’s discriminative ability.

To address the above issues, we propose a novel multi-view document clustering method, called ambiguous instance-aware contrastive network with multi-level matching (AICN-MLM). It includes three main modules: a multi-view data reconstruction module, a multi-level matching module, and an ambiguous instance-aware contrastive learning module. Specifically, the multi-view data reconstruction module aims to learn suitable feature representations for clustering from the original multi-view data. The multi-level matching module consists of multi-view similarity distribution matching (SDM) and cross-view prototype matching (CVPM). The SDM seeks to align multi-view data by minimizing the difference between the normalized multi-view similarity score distribution and the normalized pseudo-label matching distribution. The CVPM establishes the corresponding relationship by minimizing the JensenShannon divergence (JS) between the prototypes of multiview data. Additionally, the ambiguous instance-aware contrastive learning module uses an adaptive instance weighting function to dynamically adjust the weights of instance pairs according to their confidence levels. Therefore, it can pay more attention to ambiguous instance pairs and significantly improve the network’s discriminative ability. Extensive experimental results on several datasets have shown the advantage of the proposed method in MvDC tasks.

The main contributions of this paper are summarized as follows:

In our proposed method, we introduce a novel ambiguous instance-aware contrastive network that dynamically adjusts instance weights to emphasize ambiguous pairs, thereby enhancing the network’s discriminative ability. We propose a multi-level matching strategy that aligns the feature of multi-view document data with pseudolabel distributions and view prototype sets, effectively exploring the consistency of multi-view document data. Extensive experiments are conducted on seven selfconstructed multilingual document datasets and one public multilingual document dataset, demonstrating the superiority of our proposed method in multi-view document clustering.

# Related Work Deep Multi-view Clustering

In the past few years, DMVC has become an essential technique in multi-view clustering tasks, effectively extracting feature information from high-dimensional and complex multi-view data due to the powerful feature representation ability of DNNs. Among them, autoencoders are widely used to learn latent representations of multi-view data. Li et al. utilized autoencoders to learn latent shared representations between different views and adopted adversarial training to capture the latent distribution of multiview data (Li et al. 2019b). Kusner et al. extracted the feature information of the multi-view data using variational autoencoders to model the data distribution and learn robust representations effectively (Kusner, Paige, and Herna´ndezLobato 2017). Xu et al. learned to disentangle view-shared and view-specific visual representations and then achieved multi-view data clustering tasks (Xu et al. 2021b). Another type of DMVC method has recently utilized GCN to model the structural relationship between different views. Cheng et al. introduced a multi-view attribute graph convolutional network (MAGCN) to cluster multi-view attribute graph data (Cheng et al. 2021). Zhou et al. adopted adversarial learning and attention mechanisms to align latent feature distributions and quantify the importance of different modalities (Zhou and Shen 2020). Wang et al. combined adversarial training and adaptive fusion techniques to extract consistent latent representations from multi-view data (Wang et al. 2022a). However, existing DMVC methods only align multiview data from a single perspective, ignoring the diversity and complementarity of multi-view data.

# Contrastive Learning

Contrastive learning (Chuang et al. 2020; Lin et al. 2021; Xu et al. 2022b; Yang et al. 2023) has been widely applied in deep clustering and representation learning. Its core goal is to maximize the similarity between positive instance pairs, minimize the similarity between negative instance pairs, and enhance the consistency between different views by aligning the encoded representations.

In recent years, various studies have explored different contrastive learning frameworks in MVC tasks $\mathrm { \Delta X u }$ et al. 2022b; Chen et al. 2023). Yang et al. proposed a dual contrastive calibration mechanism to maintain the consistency of similar but different instances in cross-view scenarios (Yang et al. 2023). Yan et al. attempted to learn consensus and view-specific representations from multiple views through global and cross-view feature aggregation, and then used structure-guided contrastive learning to align the feature representations of each view (Yan et al. 2023). These studies investigate the application of contrastive learning to enhance multi-view clustering performance. However, they treat all positive and negative instance pairs equally and neglect ambiguous ones, resulting in suboptimal performance improvement.

# Method

# Network Architecture

As shown in Figure 1, the proposed AIACN-MLM method consists of three modules: a multi-view data reconstruction module, a multi-level matching module, and an ambiguous instance-aware contrastive learning module. Next, we will provide a detailed introduction to these three modules.

# Multi-view Data Reconstruction

Since multi-view data usually contain redundant information and random noise that can adversely affect clustering tasks, autoencoders are commonly used to learn salient representations from original multi-view data. Specifically, we denote the encoder and decoder of the $\boldsymbol { v }$ -th view as $f ^ { v } ( x _ { i } ^ { v } ; \theta ^ { v } )$ and $g ^ { v } ( x _ { i } ^ { v } ; \varphi ^ { v } )$ , respectively, where $\theta ^ { v }$ and $\varphi ^ { v }$ represent the network parameters corresponding to the encoder and decoder, respectively. Thus, the low-dimensional feature learned from multi-view data using the model’s encoder can be represented as follows:

![](images/6c26c1ee491b31b9218e0afd2727ac419430fbb91f847803c293560c0e02f3d5.jpg)  
Figure 1: The overall structure of the proposed AICN-MLM method.

$$
h _ { i } ^ { v } = f ^ { v } ( x _ { i } ^ { v } ; \theta ^ { v } ) ,
$$

where $h _ { i } ^ { v }$ is the embedded feature of $x _ { i } ^ { v }$ . The decoder reconstructs the instance by low-dimensional feature representation $h _ { i } ^ { v }$ . Therefore, multi-view data reconstruction using the decoder $g ^ { v } ( h _ { i } ^ { v } ; \phi ^ { v } )$ can be expressed as follows:

$$
\hat { x } _ { i } ^ { v } = g ^ { v } ( h _ { i } ^ { v } ; \phi ^ { v } ) .
$$

where $\hat { x } _ { i } ^ { v }$ denotes the reconstructed instance. The reconstruction loss from input $X ^ { v }$ to output ${ \hat { X } } ^ { v }$ is denoted as $\mathcal { L } _ { r e c }$ . Thus the reconstruction objective loss of all views is formulated as follows:

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \boldsymbol { r e c } } = \sum _ { v = 1 } ^ { V } \Big \| \boldsymbol { X ^ { v } } - \hat { \boldsymbol { X } } ^ { v } \Big \| _ { F } ^ { 2 } } \\ { \displaystyle \qquad = \sum _ { v = 1 } ^ { V } \sum _ { i = 1 } ^ { N } \| \boldsymbol { x _ { i } ^ { v } } - { \boldsymbol g ^ { v } } ( f ^ { v } ( \boldsymbol { x _ { i } ^ { v } } ; { \boldsymbol { \theta ^ { v } } } ) ; { \boldsymbol { \phi ^ { v } } } ) \| _ { 2 } ^ { 2 } , } \end{array}
$$

where $N$ denotes the number of instances in a batch, and $V$ represents the number of views. By optimizing Eq.(3), we can obtain low-dimensional feature representations of multiview data for clustering tasks.

# Multi-level Matching

To explore the consistency of multi-view data, we design a multi-level matching strategy in the proposed method. It includes the multi-view similarity distribution matching module (SDM) and the cross-view prototype matching module (CVPM). The SDM module aligns feature representations from different views by minimizing the Maximum Mean Discrepancy (MMD) between the cosine similarity distributions and pseudo label distributions. The CVPM module establishes prototype correspondences by optimizing the Jensen-Shannon divergence (JS) between cross-view prototypes.

In SDM, we initially stack a single-layer linear MLP on the low-level features $h ^ { v }$ to obtain the high-level features $z ^ { v }$ , denoted as zv = F ({hv}vV=1; WH), where WH represents a set of learnable parameters. Next, we calculate the cosine similarity matrix $s$ of the high-level features $z ^ { v }$ of multiview data. Then, the high-level features $z ^ { v }$ of each view are concatenated to obtain the common features $z$ . We perform $k$ -means clustering on $z$ to obtain the pseudo-labels. Finally, we minimize the MMD to align the view similarity distribution and the pseudo-labels distribution.

Given $N$ multi-view instance pairs, the high-level features of the $i$ -th instance in the $m$ -th view are denoted as $z _ { i } ^ { m }$ . Then i.e. we construct a set of representation pairs of different views, $\{ ( z _ { i } ^ { m } , z _ { j } ^ { n } ) , y _ { i , j } \} _ { i , j = 1 } ^ { N }$ , where $y _ { i , j }$ is the matching label. Specifically, $y _ { i , j }$ indicates whether the -th instance and the $j$ -th instance belong to the same category. If $y _ { i , j } = 1$ , then $( z _ { i } ^ { m } , z _ { i } ^ { n } )$ is a matching pair of the same class, while $y _ { i , j } = 0$ represents an unmatched pair. The probability of a matching pair can be calculated using the following softmax function:

$$
p _ { i , j } = \frac { \exp ( s i m ( z _ { i } ^ { m } , z _ { j } ^ { n } ) / \tau ) } { \sum _ { i , j = 1 } ^ { N } \exp ( s i m ( z _ { i } ^ { m } , z _ { j } ^ { n } ) / \tau ) } ,
$$

where $\begin{array} { r } { s i m ( z _ { i } ^ { m } , z _ { j } ^ { n } ) = \frac { z _ { i } ^ { m \top } z _ { j } ^ { n } } { \| z _ { i } ^ { m } \| \| z _ { j } ^ { n } \| } } \end{array}$ represents the cosine sim1   
ilarity between $z _ { i } ^ { m }$ and $z _ { j } ^ { n }$ , and $\tau$ is the temperature hyper  
parameter that controls the sharpness of the probability dis  
tribution.

The loss of the SDM between the $m$ -th view and the $n$ -th view in a batch is calculated as follows:

$$
\begin{array} { r l r } {  { \mathcal { L } _ { s d m } = \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } d _ { \mathcal { H } } ( q _ { i , j } ; p _ { i , j } ) } } \\ & { } & { = \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \big \| \mathbb { E } _ { q _ { i , j } } [ \varphi ( q _ { i , j } ) ] - \mathbb { E } _ { p _ { i , j } } [ \varphi ( p _ { i , j } ) ] \big \| _ { \mathcal { H } } ^ { 2 } , ~ } \end{array}
$$

where qi,j = $\begin{array} { r } { q _ { i , j } ~ = ~ \frac { y _ { i , j } } { \sum _ { i , j = 1 } ^ { N } y _ { i , j } } } \end{array}$ is the predict match probability. Here, $\mathcal { H }$ represents the Reproducing Kernel Hilbert Space (RKHS), and $\varphi ( \cdot )$ represents the mapping function that projects the original feature to RKHS, such as the Gaussian kernel function.

We assume that the cluster distributions of two related views should be closer to each other. To this end, we introduce a prototype matrix $C = [ c _ { 1 } , c _ { 2 } , \dotsc , c _ { k } ] \in \mathbb { R } ^ { D \times K }$ , where $K$ is the number of clusters, and $D$ is the dimension of the embedding across all views. Each $c _ { k }$ represents a trainable prototype vector, indicating the center of the corresponding cluster. In practice, we use a linear layer to learn the prototype matrix $C$ .

The cross-view prototype matching module uses JensenShannon (JS) divergence to measure the pairwise prototype differences between cross-views in the feature space. Specifically, the loss function of the cross-view prototype matching can be defined as follows:

$$
\begin{array} { r l r } & { } & { \mathcal { L } _ { c v p m } = \displaystyle { \frac { 1 } { 2 } \sum _ { k = 1 } ^ { K } p ( C _ { k } ^ { m } ) \log \left( \frac { 2 * p ( C _ { k } ^ { m } ) } { p ( C _ { k } ^ { m } ) + p ( C _ { k } ^ { n } ) } \right) } + } \\ & { } & { \displaystyle { \frac { 1 } { 2 } \sum _ { k = 1 } ^ { K } p ( C _ { k } ^ { n } ) \log \left( \frac { 2 * p ( C _ { k } ^ { n } ) } { p ( C _ { k } ^ { n } ) + p ( C _ { k } ^ { m } ) } \right) , } } \end{array}
$$

where $p ( C _ { k } ^ { m } )$ and $p ( C _ { k } ^ { n } )$ represent the probability distribution of the $k$ -th prototype in the $m$ -th view and the $n$ -th view, respectively.

By matching the prototype-to-prototype correspondence between each pair of views, this module calibrates the relationship between prototypes in different views, thereby solving the prototype-shift problem and further improving clustering performance.

# Ambiguous Instance Aware Contrastive Learning

Traditional contrastive learning methods minimize InfoNCE loss by pulling together instances of the same category from different views while pushing dissimilar instances apart.

However, these methods treat ambiguous and clear instance pairs equally, which can limit the discriminative ability of the network. To address this issue, we propose a weight adjustment function $\mathcal { M }$ to dynamically adjust the weights of instance pairs during training. Thus, we first calculate the distance from the instance to the cluster center, and then obtain the confidence score. The top $\lambda$ instances, based on this score, form the high-confidence instance set $H \in \mathbb { R } ^ { M }$ . Here, $\lambda$ is the confidence hyperparameter, and $M$ is the number of high-confidence instances. We then derive the pair pseudolabels $Q \in \mathbb { R } ^ { K \times N }$ from the instance pseudo-labels $P$ as follows:

$$
Q _ { i j } = \left\{ \begin{array} { l l } { 1 } & { \quad P _ { i } = P _ { j } , } \\ { 0 } & { \quad P _ { i } \neq P _ { j } . } \end{array} \right.
$$

Based on the similarity function $S$ and the pair pseudo-labels $Q$ , the weight adjustment function $\mathcal { M }$ is formulated as follows:

$$
\mathcal { M } ( z _ { i } ^ { m } , z _ { j } ^ { n } ) = \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f ~ } i , j \notin H , } \\ { | Q _ { i j } - N o r m ( S ( z _ { i } ^ { m } , z _ { j } ^ { n } ) ) | ^ { \gamma } } & { \mathrm { o t h e r w i s e , ~ } } \end{array} \right.
$$

where $z _ { i } ^ { m }$ represents the $i$ -th instance of the $m$ -th view, $\gamma$ is the focusing factor, and Norm represents the Min-Max normalization. $S ( z _ { i } ^ { m } , z _ { j } ^ { n } )$ represents the cosine similarity between the $i$ -th instance in $m$ -th view and the $j$ -th instance in $n$ -th view. In Eq.(8), when the confidence of the instance is low, we keep the initial setting in the InfoNCE loss. When the instance has high confidence, the instance weight is modulated by the pseudo information and instance similarity. $\mathcal { M }$ can increase the weight of ambiguous instances while reducing the weight of clear instances.

Specifically, when the $i$ -th and $j$ -th instances are identified as a positive pair $( Q _ { i j } = 1 )$ , their possibility of being classified into the same category increases with their similarity. Therefore, $\mathcal { M }$ increases the weight of positive pairs with less similarity (ambiguous instances) and decreases the weight of positive pairs with greater similarity (clear instances). The ambiguous instance-aware contrastive loss for the $i$ -th instance of the $m$ -th view is given as follows:

$$
\begin{array} { r l } & { \mathcal { L } ( z _ { i } ^ { m } ) = - \log } \\ & { \frac { \sum _ { i = 1 } ^ { N } e ^ { \mathcal { M } ( z _ { i } ^ { m } , z _ { i } ^ { n } ) \cdot S ( z _ { i } ^ { m } , z _ { i } ^ { n } ) } } { \sum _ { j = 1 } ^ { N } ( e ^ { \mathcal { M } ( z _ { i } ^ { m } , z _ { j } ^ { m } ) \cdot S ( z _ { i } ^ { m } , z _ { j } ^ { m } ) } + e ^ { \mathcal { M } ( z _ { i } ^ { m } , z _ { j } ^ { n } ) \cdot S ( z _ { i } ^ { m } , z _ { j } ^ { n } ) } ) } . } \end{array}
$$

In contrast to traditional InfoNCE loss, our weight modulation function $\mathcal { M }$ increases the weight of ambiguous instance pairs and reduces the weight of clear instance pairs. The overall loss formula is given as follows:

$$
\mathcal { L } _ { a i c l } = \frac { 1 } { N } \frac { 1 } { V } \sum _ { m = 1 } ^ { V } \sum _ { i = 1 } ^ { N } \mathcal { L } ( z _ { i } ^ { m } ) .
$$

This ambiguous instance-aware contrastive loss can guide the network to pay more attention to ambiguous instances, thereby enhancing the model’s discriminative ability.

# Loss Function

By integrating the reconstruction loss $\mathcal { L } _ { r e c }$ , the similarity distribution matching loss $\mathcal { L } _ { s d m }$ , the cross-view prototype

matching loss $\mathcal { L } _ { c v p m }$ and the ambiguous instance-aware contrastive loss $\mathcal { L } _ { a i c l }$ , the total loss function of the proposed method is formulated as follows:

$$
\mathcal { L } = \mathcal { L } _ { r e c } + \mathcal { L } _ { a i c l } + \alpha \mathcal { L } _ { c v p m } + \beta \mathcal { L } _ { s d m } ,
$$

where $\alpha$ and $\beta$ are the trade-off hyperparameters.

# Experiments

In this section, we conducted extensive experiments to evaluate the effectiveness of the proposed method in MVC tasks.

# Datasets

The experiments were carried out using two datasets: a selfconstructed multilingual document dataset (KUST) and a public multilingual document dataset (Reuters).

The KUST multi-view document dataset includes the following seven subsets: (1) KUST-ETD has 10,000 instances spanning 10 themes, with English and Thai document views. (2) KUST-CTD consists of 10,000 instances from 10 themes, each with views of Chinese and Thai documents. (3) KUST-CVD comprises 10,000 instances with Chinese and Vietnamese documents, divided into 10 distinct classes. (4) KUST-CED contains 10,000 instances with Chinese and English documents, categorized into 10 groups. (5) KUST-CBD includes 10,000 samples from 10 themes with views of Chinese and Burmese documents. (6) KUST-BTD includes 10,000 instances with Burmese and Thai documents, across 10 themes. (7) KUST-CLD consists of 10,000 instances from 10 themes with Chinese and Laos document views.

The Reuters dataset contains 9379 samples from six classes, with English and French views projected into a 10- dimensional space using a standard autoencoder, following the method described by Yang et al. (Yang et al. 2022).

# Comparison Methods And Evaluation Measures

To verify the superiority of our proposed method, we conducted a comprehensive comparison with several state-ofthe-art MVC methods, such as DEMVC (Xu et al. 2021a); SDMVC (Xu et al. 2022a); DSMVC (Tang and Liu 2022); MFLVC (Xu et al. 2022b); FastMICE (Huang, Wang, and Lai 2023); GCFAgg (Yan et al. 2023); DealMVC (Yang et al. 2023); CVCL (Chen et al. 2023); DIVIDE (Lu et al. 2024); ICMVC (Chao, Jiang, and Chu 2024); and MAGA (Bian et al. 2024).

To evaluate clustering performance, we employed four widely accepted metrics: accuracy (ACC), normalized mutual information (NMI), adjusted Rand index (ARI), and purity (PUR). Generally, higher values in these metrics indicate better clustering performance.

# Implementation Details

All samples were reshaped into vectors, and then the fully connected (FC) autoencoders with a similar architecture were used to extract low-dimensional features $h ^ { v }$ of multiview document data. Specifically, the structure of the encoders was given as follows: Input $\mathrm { \cdot \ F C 5 0 0 - F C 5 0 0 - }$

FC 2000 FC 512, and the decoders mirrored with the encoder. The following settings were consistent across all experimental datasets: The ReLU activation function was applied to all layers except for the output layer. Adam optimizer was used with a default learning rate of 0.0003. The confidence parameter $\lambda$ was set to 0.9, the focusing factor $\gamma$ to 1, and the temperature parameter $\tau$ to 0.02. For the KUSTCB and KUST-CL datasets, we fixed the parameters $\alpha$ and $\beta$ to 0.001 and 1, respectively. For other datasets, $\alpha$ and $\beta$ were fixed to 0.001 and 0.01. To ensure fairness in comparing results, we ran all the methods five times and reported their average clustering results. The clustering experiments were performed on an Ubuntu computer with an NVIDIA GeForce RTX 3090 GPU (24.0GB memory size).

# Experimental Results And Analysis

In our experiments, we compared our proposed method with several state-of-the-art methods with different metrics. Tables 1 and 2 show the experimental results of various methods on eight datasets. It is worth noting that the best value of the clustering result was highlighted in bold format. It can be seen that our proposed approach outperforms other stateof-the-art methods across various datasets.

On the one hand, our proposed method adopts a multilevel matching strategy to explore the consistency of multiview data from different perspectives, such as features, pseudo-labels, and prototypes. By minimizing the MMD between the cosine similarity distribution of views and the pseudo-label distribution in the feature space, we can more effectively align the representation of each view. Furthermore, our cross-view prototype matching aligns prototype matrices across views, addressing the prototype offset issue and significantly improving clustering performance.

On the other hand, we introduce an ambiguous instanceaware strategy in contrastive learning. Our instance weight adjustment function dynamically increases the weights of ambiguous instance pairs while reducing the weights of clear instance pairs. It effectively distinguishes positive instance pairs with low similarity and negative instance pairs with high similarity, providing comprehensive training for all instance pairs. As a result, this significantly enhances the network’s discriminative ability, leading to better clustering performance.

# Ablation Study

In this subsection, we conducted ablation experiments to assess the contribution of each component in the proposed method under the same experimental settings. Specifically, we constructed six variants of our proposed method: (1) Excluding the similarity distribution matching module, called AICN-MLM (w/o SDM); (2) Removing the cross-view prototype matching module, termed AICN-MLM (w/o CVPM); (3) Eliminating the ambiguous instance aware contrastive learning module, labeled AICN-MLM (w/o AICL); (4) Excluding the ambiguous instance-aware component within the AICL module, referred to as AICN-MLM (w/o AI); (5) Using KL divergence to replace the MMD in the SDM module, referred to as AICN-MLM (w/ KL in SDM); (6) Replacing JS divergence with KL divergence in the CVPM module, referred to as AICN-MLM (w/ KL in CVPM); (7) Replacing the AICL module with the PSCL contrastive loss, called AICN-MLM (w/ PSCL).

Table 1: The clustering performances of different MVC methods on the KUST-ETD, KUST-CTD, KUST-CVD, and Reuter datasets.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="4">KUST-ETD</td><td colspan="4">KUST-CTD</td><td colspan="4">KUST-CVD</td><td colspan="4">Reuters</td></tr><tr><td>ACC</td><td>NMI</td><td>ARI</td><td>PUR</td><td>ACC</td><td>NMI</td><td>ARI</td><td>PUR</td><td>ACC</td><td>NMI</td><td>ARI</td><td>PUR</td><td>ACC</td><td>NMI</td><td>ARI</td><td>PUR</td></tr><tr><td>DEMVC (2021)</td><td>67.19</td><td>67.24</td><td>53.86</td><td>68.07</td><td>52.93</td><td>59.13</td><td>40.48</td><td>55.28</td><td>52.83</td><td>56.80</td><td>37.15</td><td>53.28</td><td>50.98</td><td>30.91</td><td>25.21</td><td>55.37</td></tr><tr><td>SDMVC (2022)</td><td>62.87</td><td>70.68</td><td>50.16</td><td>69.79</td><td>56.03</td><td>66.75</td><td>44.91</td><td>66.38</td><td>45.88</td><td>59.65</td><td>35.47</td><td>53.25</td><td>45.38</td><td>21.84</td><td>18.19</td><td>51.47</td></tr><tr><td>DSMVC (2022)</td><td>48.92</td><td>46.16</td><td>35.35</td><td>53.95</td><td>43.52</td><td>41.85</td><td>26.89</td><td>53.03</td><td>41.41</td><td>42.38</td><td>26.19</td><td>50.95</td><td>45.87</td><td>20.49</td><td>20.25</td><td>50.85</td></tr><tr><td>MFLVC (2022)</td><td>75.38</td><td>82.92</td><td>68.80</td><td>82.30</td><td>72.56</td><td>83.94</td><td>69.25</td><td>82.48</td><td>67.63</td><td>73.12</td><td>53.78</td><td>74.55</td><td>53.62</td><td>35.25</td><td>28.22</td><td>59.36</td></tr><tr><td>FastMICE (2023)</td><td>60.11</td><td>71.83</td><td>52.18</td><td>73.34</td><td>55.29</td><td>67.89</td><td>46.86</td><td>68.74</td><td>56.99</td><td>61.59</td><td>40.91</td><td>64.38</td><td>38.73</td><td>19.17</td><td>13.28</td><td>49.79</td></tr><tr><td>GCFAgg (2023)</td><td>74.04</td><td>81.60</td><td>66.32</td><td>80.96</td><td>73.00</td><td>81.13</td><td>65.95</td><td>79.92</td><td>62.41</td><td>74.86</td><td>52.14</td><td>74.87</td><td>55.84</td><td>37.41</td><td>29.06</td><td>59.10</td></tr><tr><td>DealMVC (2023)</td><td>74.12</td><td>78.58</td><td>66.59</td><td>74.43</td><td>74.12</td><td>78.58</td><td>66.59</td><td>74.43</td><td>71.15</td><td>76.71</td><td>57.43</td><td>78.67</td><td>55.29</td><td>42.45</td><td>31.37</td><td>62.60</td></tr><tr><td>CVCL (2023)</td><td>79.32</td><td>83.36</td><td>75.51</td><td>83.34</td><td>77.05</td><td>81.10</td><td>66.31</td><td>82.58</td><td>58.17</td><td>72.37</td><td>52.44</td><td>65.37</td><td>55.64</td><td>31.14</td><td>26.53</td><td>57.35</td></tr><tr><td>DIVIDE (2024)</td><td>55.64</td><td>70.85</td><td>41.73</td><td>79.36</td><td>70.52</td><td>78.63</td><td>62.35</td><td>77.44</td><td>58.82</td><td>65.70</td><td>52.92</td><td>69.11</td><td>50.95</td><td>34.82</td><td>27.66</td><td>59.72</td></tr><tr><td>ICMVC (2024)</td><td>69.45</td><td>78.09</td><td>61.36</td><td>78.38</td><td>56.24</td><td>64.66</td><td>42.72</td><td>66.17</td><td>63.52</td><td>74.84</td><td>56.01</td><td>72.44</td><td>51.54</td><td>36.15</td><td>28.06</td><td>59.88</td></tr><tr><td>MAGA (2024)</td><td>72.42</td><td>79.98</td><td>68.56</td><td>78.06</td><td>76.13</td><td>81.34</td><td>68.94</td><td>83.05</td><td>64.44</td><td>73.33</td><td>54.99</td><td>71.68</td><td>51.07</td><td>31.97</td><td>26.62</td><td>56.38</td></tr><tr><td>AICN-MLM(Ours)</td><td>89.08</td><td>86.21</td><td>80.13</td><td>89.08</td><td>77.30</td><td>84.31</td><td>70.47</td><td>83.58</td><td>72.56</td><td>79.98</td><td>64.59</td><td>79.48</td><td>59.03</td><td>42.50</td><td>33.57</td><td>63.01</td></tr></table></body></html>

Table 2: The clustering performances of different MVC methods on the KUST-CED, KUST-CBD, KUST-BTD, and KUST CLD datasets.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="4">KUST-CED</td><td colspan="4">KUST-CBD</td><td colspan="4">KUST-BTD</td><td colspan="4">KUST-CLD</td></tr><tr><td>ACC</td><td>NMI</td><td>ARI</td><td>PUR</td><td>ACC</td><td>NMI</td><td>ARI</td><td>PUR</td><td>ACC</td><td>NMI</td><td>ARI</td><td>PUR</td><td>ACC</td><td>NMI</td><td>ARI</td><td>PUR</td></tr><tr><td>DEMVC (2021)</td><td>55.67</td><td>61.25</td><td>41.76</td><td>61.43</td><td>57.22</td><td>61.49</td><td>40.75</td><td>60.53</td><td>61.39</td><td>64.39</td><td>51.93</td><td>63.08</td><td>52.23</td><td>57.56</td><td>36.54</td><td>54.88</td></tr><tr><td>SDMVC (2022)</td><td>50.44</td><td>59.81</td><td>39.19</td><td>57.96</td><td>60.46</td><td>67.75</td><td>46.85</td><td>67.35</td><td>57.60</td><td>70.42</td><td>49.20</td><td>68.43</td><td>56.08</td><td>67.47</td><td>44.85</td><td>64.66</td></tr><tr><td>DSMVC (2022)</td><td>49.02</td><td>48.11</td><td>36.24</td><td>52.80</td><td>42.94</td><td>34.89</td><td>26.85</td><td>44.96</td><td>47.75</td><td>42.05</td><td>32.08</td><td>52.56</td><td>43.03</td><td>34.58</td><td>20.15</td><td>44.24</td></tr><tr><td>MFLVC (2022)</td><td>69.75</td><td>77.80</td><td>61.84</td><td>79.14</td><td>71.35</td><td>78.86</td><td>59.10</td><td>78.27</td><td>70.83</td><td>78.61</td><td>61.61</td><td>77.75</td><td>69.14</td><td>79.99</td><td>62.63</td><td>76.85</td></tr><tr><td>FastMICE (2023)</td><td>62.81</td><td>69.34</td><td>48.93</td><td>68.57</td><td>58.43</td><td>67.05</td><td>48.62</td><td>67.72</td><td>57.96</td><td>69.92</td><td>49.73</td><td>70.83</td><td>61.58</td><td>71.65</td><td>51.51</td><td>72.18</td></tr><tr><td>GCFAgg (2023)</td><td>71.96</td><td>78.53</td><td>59.76</td><td>77.40</td><td>67.90</td><td>81.35</td><td>63.78</td><td>79.27</td><td>72.49</td><td>80.39</td><td>65.10</td><td>79.41</td><td>69.88</td><td>78.46</td><td>61.95</td><td>76.80</td></tr><tr><td>DealMVC (2023)</td><td>65.47</td><td>73.93</td><td>52.54</td><td>73.29</td><td>71.60</td><td>73.90</td><td>62.02</td><td>73.88</td><td>68.24</td><td>69.00</td><td>58.81</td><td>68.24</td><td>60.81</td><td>70.53</td><td>52.88</td><td>67.95</td></tr><tr><td>CVCL (2023)</td><td>63.66</td><td>72.97</td><td>51.43</td><td>71.91</td><td>60.22</td><td>74.37</td><td>53.88</td><td>69.30</td><td>73.54</td><td>77.87</td><td>64.10</td><td>78.87</td><td>62.81</td><td>73.58</td><td>69.72</td><td>53.23</td></tr><tr><td>DIVIDE (2024)</td><td>63.96</td><td>71.25</td><td>52.07</td><td>72.10</td><td>68.95</td><td>76.16</td><td>58.42</td><td>75.87</td><td>55.14</td><td>68.82</td><td>41.05</td><td>78.31</td><td>54.54</td><td>64.02</td><td>36.63</td><td>63.91</td></tr><tr><td>ICMVC (2024)</td><td>66.71</td><td>75.15</td><td>57.06</td><td>74.52</td><td>59.68</td><td>76.07</td><td>58.49</td><td>79.21</td><td>70.86</td><td>79.12</td><td>62.60</td><td>77.78</td><td>61.94</td><td>73.07</td><td>52.59</td><td>68.86</td></tr><tr><td>MAGA (2024)</td><td>64.98</td><td>74.12</td><td>56.83</td><td>71.38</td><td>63.66</td><td>71.61</td><td>52.90</td><td>70.40</td><td>78.67</td><td>81.98</td><td>74.55</td><td>82.47</td><td>61.23</td><td>72.51</td><td>52.72</td><td>68.25</td></tr><tr><td>AICN-MLM(Ours)</td><td>76.22</td><td>82.64</td><td>64.54</td><td>82.67</td><td>72.73</td><td>81.36</td><td>63.85</td><td>79.65</td><td>83.14</td><td>84.68</td><td>77.18</td><td>83.53</td><td>74.21</td><td>82.40</td><td>67.17</td><td>81.13</td></tr></table></body></html>

<html><body><table><tr><td rowspan="2">Model Setting</td><td colspan="4">KUST-ETD</td><td colspan="4">KUST-CTD</td><td colspan="4">KUST-CVD</td><td colspan="4">Reuters</td></tr><tr><td>ACC</td><td>NMI</td><td>ARI</td><td>PUR</td><td>ACC</td><td>NMI</td><td>ARI</td><td>PUR</td><td>ACC</td><td>NMI</td><td>ARI</td><td>PUR</td><td>ACC</td><td>NMI</td><td>ARI</td><td>PUR</td></tr><tr><td>AICN-MLM(w/o SDM)</td><td>83.76</td><td>84.84</td><td>76.68</td><td>84.18</td><td>70.25</td><td>79.36</td><td>62.35</td><td>77.33</td><td>71.15</td><td>79.82</td><td>63.92</td><td>78.07</td><td>58.94</td><td>42.44</td><td>33.42</td><td>62.92</td></tr><tr><td>AICN-MLM(w/o CVPM)</td><td></td><td>84.2285.80</td><td>78.10</td><td>84.78</td><td>75.27</td><td>83.19</td><td>70.03</td><td>82.57</td><td>69.60</td><td>77.55</td><td>60.40</td><td>76.51</td><td>57.63</td><td>41.25</td><td>32.13</td><td>62.52</td></tr><tr><td>AICN-MLM (w/o AICL)</td><td></td><td>28.3618.78</td><td>11.70</td><td>33.34</td><td>29.32</td><td>15.13</td><td>12.37</td><td>28.77</td><td>28.95</td><td>21.35</td><td>11.94</td><td>31.41</td><td>35.29</td><td>8.65</td><td>7.57</td><td>39.25</td></tr><tr><td>AICN-MLM (w/o AI)</td><td>83.15</td><td>84.75</td><td>76.30</td><td>83.49</td><td>75.78</td><td>83.62</td><td>69.86</td><td>82.77</td><td>72.24</td><td>77.68</td><td>55.67</td><td>79.16</td><td>43.15</td><td>32.68</td><td>18.45</td><td>53.26</td></tr><tr><td>AICN-MLM(w/KL in SDM)</td><td>83.09</td><td>84.45</td><td>76.30</td><td>83.61</td><td>74.78</td><td>81.59</td><td>68.25</td><td>81.33</td><td>71.16</td><td>79.78</td><td>63.13</td><td>78.08</td><td>53.12</td><td>39.52</td><td>29.35</td><td>61.11</td></tr><tr><td>AICN-MLM(w/KL in CVPM)</td><td>78.93</td><td>80.07</td><td>70.98</td><td>78.93</td><td>62.86</td><td>73.64</td><td>56.25</td><td>70.01</td><td>56.56</td><td>67.57</td><td>43.98</td><td>59.01</td><td>53.18</td><td>39.73</td><td>29.31</td><td>61.98</td></tr><tr><td>AICN-MLM(w/PSCL)</td><td>58.39</td><td>72.27</td><td>41.34</td><td>70.58</td><td>69.87</td><td>78.35</td><td>62.87</td><td>77.86</td><td>61.54</td><td>69.37</td><td>32.94</td><td>68.46</td><td>47.61</td><td>38.34</td><td>24.47</td><td>57.71</td></tr><tr><td>AICN-MLM(Ours)</td><td></td><td>89.08 86.21</td><td>80.13</td><td>89.08</td><td>77.30</td><td>84.31</td><td>70.47</td><td>83.58</td><td>72.56</td><td>79.98</td><td>64.59</td><td>79.48</td><td>59.03</td><td>42.50</td><td>33.57</td><td>63.01</td></tr></table></body></html>

Table 3: The ablation study of our method on the KUST-ETD, KUST-CTD, KUST-CVD, and Reuters datasets.

![](images/01802db04770b1e3464fc3efc9adf4c0e793dfa5b680b796dc5018142e32e454.jpg)  
Figure 2: The clustering performances and training loss of our approach on the KUST-ETD and KUST-CTD datasets.

Table 3 shows the ablation results of our proposed method across four different datasets. It can be seen that removing any component from our method or substituting our proposed modules with alternative ones significantly degrades clustering performance. This shows that each component of our proposed method plays a crucial role in enhancing performance in real-world clustering applications.

# Convergence Analysis

In this experiment, we analyzed the convergence rate of the proposed model in clustering. Figure 2 illustrates the clustering performance and training loss of our proposed model on the KUST-ETD and KUST-CTD datasets. It can be seen that the loss of the model steadily decreases and eventually converges to a stable value as training epochs increase. Meanwhile, the clustering performance gradually improves and eventually stabilizes. These convergence results demonstrate the reliability and effectiveness of the proposed method in the MVDC tasks.

# Parameter Sensitivity Analysis

In this subsection, we comprehensively investigated the impact of hyperparameters $\alpha$ and $\beta$ on the performance of our proposed method. Using a grid search strategy, we tested various combinations of these hyperparameters to analyze their sensitivity and impact on clustering performance.

The range of the hyperparameters $\alpha$ and $\beta$ was set from 0.001 to 10. Figure 3 shows the experimental results of the proposed method on the KUST-ETD dataset with different combinations of hyperparameters. The results show that our method can achieve stable clustering performance across a wide parameter range. This demonstrates that the proposed method can be easily applied to various practical problems.

# Visualization

To visually assess the effectiveness of our proposed model, we adopted the $t$ -SNE method to display the latent features generated by the clustering layer. Figure 4 shows the visualizations of the raw features and the learned features on the KUST-ETD and KUST-CTD datasets. The results indicate that the clustering structure of the learned features becomes clearer, demonstrating the effectiveness of our proposed model.

![](images/e2d7bef66e0b6d04e4ad534a4588daf47e686b54e4b7bedc4f32b0d9f1bc75cd.jpg)  
Figure 3: The sensitivity analysis of the hyperparameters $\alpha$ and $\beta$ on the KUST-ETD and KUST-CTD datasets.

![](images/766b96d5c4ad73fe4ff397b8b47a17ef20054da0495e6d30dffb73ed07211303.jpg)  
Figure 4: The $t$ -SNE visualization results of the proposed method on the KUST-ETD dataset.

# Conclusion

In this paper, we propose a novel method, called ambiguous instance-aware contrastive network with multi-level matching (AICN-MLM), for MvDC tasks. First, we design a multi-level matching strategy from multiple perspectives to more effectively align multi-view features. Additionally, we introduce an ambiguous instance-aware contrastive learning module that adopts an instance weight adjustment function to dynamically increase the weight of ambiguous instances while decreasing the weight of clear instances. This guides the network to focus on ambiguous instances and enhances its discriminative ability, overcoming the limitation of classic contrastive learning that treats all instances equally. Extensive experiments on eight multi-view document datasets demonstrate the superior performance of our proposed method in real-world clustering tasks.