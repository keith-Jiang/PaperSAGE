# Revisiting Interpolation for Noisy Label Correction

Yuanzhuo $\mathbf { X } \mathbf { u } ^ { 1 }$ , Xiaoguang $\mathbf { N i u } ^ { 1 * }$ , Jie $\mathbf { Y a n g ^ { 1 } }$ , Ruiyi $\mathbf { S } \mathbf { u } ^ { 1 }$ , Jian Zhang1, Shubo Liu1, Steve Drew2

1School of Computer Science, Wuhan University, China 2 Department of Electrical and Software Engineering, University of Calgary, Canada {xyzxyz, xgniu, csyangjie, ruiyisu, jzhang, liu.shubo}@whu.edu.cn, steve.drew $@$ ucalgary.ca

# Abstract

Label correction methods are popular for their simple architecture in learning with noisy labels. However, they suffer severely from false label correction and achieve subpar performance compared with state-of-the-art methods. In this paper, we revisit the label correction methods through theoretical analysis of gradient scaling and demonstrate that the sample-wise dynamic and class-wise uniformity of interpolation weight prevents memorization of the mislabeled samples. We then propose DULC, a simple yet effective label correction method that uses the normalized Jensen-Shannon divergence (JSD) metric as the interpolation weight to promote sample-wise dynamic and class-wise uniformity. Additionally, we provide theoretical evidence that sharpening predictions in label correction facilitates the memorization of true class, and we achieve it by employing the augmentation strategy along with the sharpening function. Extensive experiments on CIFAR-10, CIFAR-100, TinyImageNet, WebVision and Clothing1M datasets demonstrate substantial improvements over state-of-the-art methods.

Code — https://github.com/kovelxyz/DULC.

# Introduction

Deep neural networks (DNNs) have proven effective in various tasks (He et al. 2016; Song, Kim, and Lee 2019; Wang et al. 2021; Srinivas et al. 2021; Song et al. 2021). The effectiveness relies heavily on the collection of datasets with high-quality annotations. However, collections of datasets and manual annotations are challenging and expensive. As an alternative, most large-scale datasets focus on opensource data that can be automatically annotated by inexpensive strategies, such as adopting web crawling and leveraging search engines (Le and Yang 2015; Li et al. 2017). These alternative methods inevitably introduce numerous noisy samples. Prior art (Arpit et al. 2017) has revealed that deep networks suffer from dramatic degradation in the generalization due to the tendency to overfit to noisy labels.

To tackle this problem, numerous methods (Arpit et al.   
2017; Han et al. 2018; Li, Socher, and Hoi 2020; Wei et al.   
2020; Li et al. 2022; Lu and He 2022; Karim et al. 2022;

Liu, Cheng, and Zhang 2023; Wei et al. 2023; Zhang et al. 2021) have been proposed for learning with noisy labels. These approaches focus on label correction. Part representative group of methods (Patrini et al. 2017; Hendrycks et al. 2018; Liu, Cheng, and Zhang 2023) propose to reverses noisy labels to clean ones with estimation of the noise transition matrix, which is challenging for high numbers of classes and in high noise scenarios. Another group of labelcorrection-based methods (Reed et al. 2014; Arazo et al. 2019; Lu and He 2022) propose to generate soft targets by performing a convex combination of noisy labels and predictions according to interpolation weight. The core of these methods lies in the construction of the weight for interpolation. Bootstrapping (Reed et al. 2014) employs a static weight without accounting for sample differences. In subsequent works (Arazo et al. 2019; Lu and He 2022), dynamic weights are introduced to evaluate different samples. These dynamic weights are usually formulated based on loss criteria (e.g., CE loss) and, as a result, may be non-uniform by differences in the distribution of losses between easy and hard classes (Karim et al. 2022).

The design of interpolation weights lacking dynamic and class uniformity is susceptible to false corrections as the hard samples and classes are less likely to be corrected. Consequently, label correction methods have gradually lost their competitiveness against semi-supervised methods relying on the clean sample selection (Li, Socher, and Hoi 2020; Karim et al. 2022; Lu and He 2022; Li et al. 2022; Hu et al. 2023; Feng, Ren, and Xie 2023). Then the questions naturally arise: Can carefully designed dynamic weights rejuvenate interpolation schemes?

In this paper, we first revisit the interpolation scheme in label correction from the perspective of gradient scaling. The core idea behind gradient scaling is to promote the memorization of true classes and diminish the impact of mislabeled samples on gradients. The theoretical analysis demonstrates that the interpolation weight should adhere to two properties in order to prevent memorization of the mislabeled samples: i) sample-wise dynamic that indicates the weight is dynamic for each sample and aligned with its label cleanliness; ii) the class-wise uniformity, emphasizing that the weight of samples from different classes should be aligned to reduce the inconsistent gradients caused by class difficulty. Besides, we provide theoretical evidence to demonstrate that sharpening predictions in label correction can further facilitate the memorization of true classes.

We then propose a Dynamic and Uniform Label Correction (DULC) method which enjoys simplicity and effectiveness. Specifically, we measure the Jensen-Shannon divergence (JSD) between the predictions and noisy labels as the interpolation weight and demonstrate its sample-wise dynamic and alignment with label cleanliness. Then, we adopt max-min normalization on JSDs within the classes to promote class-wise uniformity. We finally sharpen the prediction for the combination with noisy labels by employing the augmentation strategy and sharpening function. DULC outperforms state-of-the-art (SOTA) selection-based semisupervised methods with much lower complexity under various noise settings, even in the presence of very high label noise (see Table 1). Our contributions are summarized as follows:

• We are the first to revisit the interpolation scheme for label correction from the perspective of gradient scaling. We provide theoretical evidence for two properties of an ideal interpolation weight: sample-wise dynamic and class-wise uniformity. Besides, we theoretically ensure the effectiveness of prediction sharpening in label correction.   
• We propose a simple yet effective label correction method named DULC, utilizing the normalized JensenShannon divergence (JSD) to measure the interpolation weight in label correction, ensuring sample-wise dynamic and class-wise uniformity. Furthermore, we sharpen the prediction through the augmentation and sharpening function.   
• By providing comprehensive experimental results, we show that DULC, with a much simpler architecture, significantly outperforms SOTA methods on both simulated and real-world noisy datasets. Furthermore, extensive ablation studies are conducted to validate the effectiveness of different components in DULC.

# Related Work

A variety of methods have been proposed to improve the robustness of DNNs on noisy datasets. Here, we mainly introduce label correction relevant to our work and sample selection, which becomes the SOTA baseline.

Label correction is mainly based on noise transition matrices or model predictions. The former category of methods (Patrini et al. 2017; Hendrycks et al. 2018; Liu, Cheng, and Zhang 2023) try to estimate the transition matrix from noisy labels to clean labels but are often limited in high noise ratios. The latter category of methods (Tanaka et al. 2018; Zhang et al. 2021; Zheng, Awadallah, and Dumais 2021; Reed et al. 2014; Arazo et al. 2019; Lu and He 2022) gradually adjusts the assigned label based on the model’s prediction. Bootstrapping (Reed et al. 2014) proposes to generate the new labels by convexly combining model predictions and assigned labels with fixed weights. M-correction (Arazo et al. 2019) uses instead dynamic weights defined in terms of the sample’s training loss values. Follow-up work (Lu and

He 2022) proposes to use of the ensemble prediction of multiple epochs to avoid the possible bias of correction. However, although dynamic weight design (Arazo et al. 2019; Lu and He 2022) makes sense compared to fixed weight (Reed et al. 2014), these methods still lack the careful design of weights without consideration of the class-wise uniformity, which we discuss later.

Sample selection identifies the noisy samples, e.g., using a small-loss selection to separate them from the clean ones. Early works (Han et al. 2018; Wei et al. 2020; Yao et al. 2021; Xu et al. 2023) perform small loss selection to filter out clean samples with a known noise ratio and train on them. Follow-up methods (Li, Socher, and Hoi 2020; Karim et al. 2022; Li et al. 2022; Hu et al. 2023; Feng, Ren, and Xie 2023; Zhang et al. 2024; Wang, Fu, and Sun 2024) remove the dependence on the noise prior and design a more precise division scheme to divide the dataset into clean and noisy subsets. The clean set is typically used for conventional supervised learning and the noisy samples are treated as unlabeled data for semi-supervised learning (Berthelot et al. 2019). Crosssplit (Kim et al. 2023) does not separate clean and noisy samples, but randomly divides the dataset and still performs semi-supervised training. To prevent overfitting to noisy samples, the co-training strategy of peer networks is usually applied (Li, Socher, and Hoi 2020; Karim et al. 2022; Hu et al. 2023; Kim et al. 2023). Label correction can be leveraged in sample selection methods (Li, Socher, and Hoi 2020; Karim et al. 2022; Kim et al. 2023) to alleviate the repercussions of incorrect selection. However, it is not deployed to the entire dataset but exclusively to a subset.

Other deep learning methods including: 1)regularization (Zhang et al. 2017; Liu et al. 2020); 2)robust loss (Lu, Bo, and He 2022; Wei et al. 2023); 3)contrastive learning (Kim et al. 2021; Ortego et al. 2021); 4)representation learning (Iscen et al. 2022; Tu et al. 2023). Compared with them, label correction methods exhibit a simpler structure that is efficient and easier to deploy.

Our objective is to elevate label correction to the level of competitiveness seen in SOTA methods relying on sample selection, making it not only simple but also effective. We formulate sample-wise dynamic and class-wise uniformity for interpolation weight from the perspective of gradient scaling and then propose our DULC that ensures them.

# Preliminaries

Classification with Noisy Labels Consider the $K$ -class classification task in the noisy-label scenario, the ground truth (clean) label $y$ is unobservable. We only have a noisy training set $\mathcal { D } \ = \ \{ { \pmb x } _ { i } , \tilde { y } _ { i } \} _ { i = 1 } ^ { N }$ , where $\pmb { x } _ { i }$ is an input and $\tilde { y } _ { i } \in \tilde { \{ 1 , \cdot \cdot \cdot , K \} }$ is the corresponding noisy label. We denote $\tilde { y } _ { i } \in \{ 0 , 1 \} ^ { K }$ as one-hot vector of noisy label $\tilde { y } _ { i }$ . A DNN ${ \mathcal { N } } _ { \theta }$ maps an input $\mathbf { \boldsymbol { x } } _ { i }$ to a K-dimensional logits $z _ { i }$ and then feeds the logits to a softmax function to obtain the predictions $\pmb { p } _ { i }$ of the conditional probability of each class. $\theta$ denotes the parameters of the DNN and $\dot { z _ { i } } \in \mathbb { R } ^ { K \times 1 }$ denotes the logits. We have $z _ { i } = \mathcal { N } _ { \theta } ( \pmb { x } _ { i } )$ and ${ \pmb p } _ { i } = \mathrm { s o f t m a x } ( \pmb z _ { i } )$ . Our task is to obtain a classifier that is robust to label noise without knowing joint probability distribution $P ( \pmb { x } , y )$ .

Early Learning Phenomenon When training DNNs with the typical cross-entropy (CE) loss in noisy-label scenarios, it has been observed that the DNNs preferentially fit easy (clean) samples before overfitting hard (noisy) samples (Arpit et al. 2017). Since the memorization of DNNs has a preference for easy (clean) samples, the predictive power of a sample’s representation aligns with its label cleanliness in the early training stage.

Label Correction Methods Label-correction methods utilize the early learning phenomenon as the model tends to generate clean predictions for each sample. They typically try to generate soft targets by interpolating between the noisy labels and model prediction for each sample $\pmb { x } _ { i }$ by:

$$
\pmb { \hat { y } _ { i } } = \alpha _ { i } \pmb { \hat { p } _ { i } } + ( 1 - \alpha _ { i } ) \pmb { \tilde { y } _ { i } }
$$

where $\alpha _ { i } \in [ 0 , 1 ]$ is the interpolation weights. $\hat { p } _ { i }$ is obtained by performing certain operations (e.g., copy or ensemble) on $\pmb { p } _ { i }$ and its gradient is typically frozen. Thus the empirical training cross-entropy loss becomes:

$$
\mathcal { L } _ { c e } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \ell _ { c e } ( \hat { \pmb { y } } _ { i } , \pmb { p } _ { i } ) = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \hat { \pmb { y } } _ { i } ^ { \top } \log \left( \pmb { p } _ { i } \right)
$$

The key to the label correction methods lies in the design of interpolation weights $\alpha$ in Equation (1).

Table 1: Performance under extreme label noise on CIFAR10 and CIFAR100. $( ^ { * } )$ denotes the results we obtain by rerunning their publicly available code.   

<html><body><table><tr><td>Dataset</td><td>CIFAR-10</td><td></td><td>CIFAR-100</td></tr><tr><td>Noise ratio</td><td>92%</td><td>95%</td><td>92% 95%</td></tr><tr><td>UNICON</td><td>90.08</td><td>85.94</td><td>32.24* 19.37*</td></tr><tr><td>DULC</td><td>92.94</td><td>92.04</td><td>45.32 25.61</td></tr></table></body></html>

# Revisit the Interpolation via Gradients

In this section, we introduce a gradient analysis of Eq.(2) to motivate our scheme of interpolation weight. Despite simplifying the actual model and training process, the analysis leads to some interesting implications and provides insight into how the interpolation weight should be set.

For clarity of explanation, we denote the true label of sample $\scriptstyle { \mathbf { { \vec { x } } } }$ as $y \in \mathsf { \bar { \{ 1 , \cdots , K \} } }$ . We then denote the distribution over ground-truth labels for sample $\scriptstyle { \mathbf { { \vec { x } } } }$ as $q ( y | \mathbf { x } )$ , and $\begin{array} { r } { \sum _ { k = 1 } ^ { K } q ( k | \pmb { x } ) = 1 } \end{array}$ . Similarly, the prediction probability sPdefined as $p ( k | \pmb { x } )$ and $\begin{array} { r } { \sum _ { k = 1 } ^ { K } p ( k | \pmb { x } ) = 1 } \end{array}$ . In the case of a single ground-truth label $y$ , we have $q ( y | \pmb { x } ) = 1$ and $q ( k | \pmb { x } ) \bar { = } 0$ for all $k \neq y$ . For notation simplicity, we denote $p _ { j } , q _ { j } , q _ { y }$ as abbreviations for $p ( j | \pmb { x } ) , q ( j | \pmb { x } )$ and $q ( y | \mathbf { x } )$ , where $j$ represents $j$ -th entry. Based on the early learning phenomenon, We assume that samples tend to have a higher posterior probability $p ( y | \mathbf { \boldsymbol { x } } )$ of ground-truth labels in the early training stage. We then derive the following theorem and the proof is offered in Appendix A.1:

Theorem 0.1. Given the cross-entropy loss $\mathcal { L } _ { c e }$ in Eq.( 2), we rewrite the sample-wise loss $\begin{array} { r } { \ell _ { c e } = - \sum _ { k = 1 } ^ { K } ( \alpha \hat { p } _ { k } + ( 1 - } \end{array}$ $\alpha ) q _ { k } ) \log p _ { k }$ . Its gradient with respect to $z _ { j }$ is

$$
\frac { \partial \ell _ { c e } } { \partial z _ { j } } = \left\{ \begin{array} { l l } { \alpha ( p _ { j } - \hat { p } _ { j } ) + ( 1 - \alpha ) ( p _ { j } - 1 ) , } & { q _ { j } = 1 } \\ { \alpha ( p _ { j } - \hat { p } _ { j } ) + ( 1 - \alpha ) p _ { j } , } & { q _ { j } = 0 } \end{array} \right.
$$

where $\alpha \in [ 0 , 1 ]$ and $z _ { j }$ is the $j$ -th entry of logits $z$ .

Theorem 0.1 indicates that learning on true class persists when training with Eq.(2) and the gradient of two terms in $\ell _ { c e }$ is scaled by a positive multiplier term $\alpha$ . In addition, Theorem 0.1 has the following interpretations:

• Sharpening on ${ \pmb p } _ { i }$ . The gradient term $p _ { j } - { \hat { p } } _ { j }$ in Eq.(3a) and Eq.(3b) is independent of $q _ { j }$ and becomes 0 if $\hat { p } _ { j }$ is a copy of $p _ { j }$ . Simply setting this term to 0 is not advisable as we notice that the true class typically has a higher posterior probability (i.e., $p _ { y } > p _ { j }$ for $j \neq y$ ). This fact inspires us to apply sharpening operation on ${ \pmb p } _ { i }$ , making $\hat { p } _ { y } > p _ { y }$ and $\hat { p } _ { j } < p _ { j }$ for $j \neq y$ . Thus, the gradient of $p _ { j } - { \hat { p } } _ { j }$ is negative in true class and positive in other class, effectively promoting memorization of the true classes.

• Sample-wise dynamic of $\alpha$ . For the samples with true class $j$ , the gradient term $p _ { j } \mathrm { ~ - ~ } 1$ of clean samples in Eq.(3a) tend to vanish after the early learning stage, causing mislabeled samples in Eq.(3b) to dominate the gradient. The multiplier $1 - \alpha$ should be sample-wise dynamic and we expect $\alpha$ to reflect the label cleanliness of sample $\mathbf { \delta } _ { \mathbf { \boldsymbol { x } } _ { i } }$ under prediction ${ \mathbf { } } p _ { i }$ $\mathbf { \psi } _ { { \mathbf { \psi } } _ { i } ; \textit { \textbf { \em } } \alpha } \to \mathrm { ~ 0 ~ }$ for clean samples and $\alpha  1$ for mislabeled samples. Thus by multiplying $1 - \alpha$ , it counteracts the effect of gradient dominating by mislabeled samples. For the samples that $j$ is not the true class, the gradient term $p _ { j } - 1$ in Eq.(3a) is negative, and $p _ { j }$ in Eq.(3b) is positive. Multiplying the dynamic $1 - \alpha$ effectively reduces the magnitudes of coefficients on mislabeled samples, thereby mitigating their impact on the gradient.

• Class-wise uniformity of $\alpha$ . It should also be considered that the distribution of prediction $p _ { j }$ over different classes is uneven. Higher prediction probability $p _ { j }$ tends to be skewed towards easier classes, as clean samples from hard classes (e.g.,cats and dogs in CIFAR10) may not have been memorized yet (Karim et al. 2022). In addition to the inherent bias in gradients across classes in Eq.(3a) and Eq.(3b), the non-uniformity of predictions $p _ { j }$ leads to the inter-class bias of $\alpha$ because the dynamic $\alpha$ is related to $p _ { j }$ , which further exacerbates the bias of gradients on different classes. Therefore, we recommend incorporating a mechanism of class-wise uniformity in $\alpha$ to align the gradient scaling of different classes.

Despite their importance, the dynamic and uniformity have hardly been considered or substantiated by theoretical analysis in previous methods. Bootstrapping (Reed et al. 2014) overlooks the sample-wise dynamic and applies the static weight (e.g., $\alpha = 0 . 6 \ '$ ) to all samples indiscriminately. M-correction (Arazo et al. 2019) makes $\alpha$ dynamic by BMM estimation on standard CE losses of all samples without consideration of class-wise uniformity. Besides, SELC (Lu and He 2022) assigns the same weight to all samples within one epoch, emphasizing accurate ensemble predictions by aggregating predictions over multiple epochs using exponential moving averages, while overlooking sample-wise dynamic.

We propose DULC to achieve both sample-wise dynamic and class-wise uniformity, which is a simple and effective label correction method. Details of DULC are presented in the following section.

# Our Algorithm: DULC

# Augmentation and Sharpening

As the discussion for Theorem 0.1, the prediction ${ \pmb p } _ { i }$ should be sharpened to promote memorization of the true classes. In other words, the prediction $\hat { { p } _ { i } }$ involved in label correction should be more confident than the prediction ${ \pmb p } _ { i }$ in gradient backpropagation. In this paper, We adopt two mechanisms to achieve this.

Firstly, we utilize the “Weak and Strong Augmentation” strategy to alleviate this problem, which is widely used in semi-supervised learning task (Berthelot et al. 2019; Sohn et al. 2020) and label noise learning (LNL) methods (Li, Socher, and Hoi 2020; Kim et al. 2023). Trivially, we generate strong and weak augmentation sets of the entire dataset, denoted $\mathcal { D } _ { s }$ and $\mathcal { D } _ { w }$ respectively. DULC exploits predictions on weak augmentation set $\mathcal { D } _ { w }$ for label correction and train the network on strong augmentation set $\mathcal { D } _ { s }$ . The network produces more confident predictions on the weak augmented set than on the strong augmented set (Cubuk et al. 2018), resulting in sharpening predictions. Besides, we apply a sharpening function with temperature coefficient $T$ on the prediction ${ \pmb p } _ { i }$ on $\mathcal { D } _ { w }$ directly reduce its temperature:

$$
\hat { p } _ { j } = { p _ { j } } ^ { \frac { 1 } { T } } \bigg / \sum _ { k = 1 } ^ { K } p _ { k } { ^ { \frac { 1 } { T } } } , \mathrm { f o r } j = 1 , 2 , \cdots , K .
$$

# JSD metric for Sample-wise Dynamic

In DULC, we seek to utilize a new metric to dynamize $\alpha$ . Jo-SRC (Yao et al. 2021) and UNICON (Karim et al. 2022) propose to adopt the Jensen-Shannon divergence (JSD) to quantify the difference between prediction probability distribution $\pmb { p } _ { i }$ and the noisy labels distribution $\tilde { y } _ { i }$ . JSD is naturally bounded in $[ 0 , 1 ]$ , and we derive the following Theorem to prove that JSD can serve as an ideal metric of $\alpha$ :

Theorem 0.2. Given the noisy label of sample $\scriptstyle { \mathbf { { \vec { x } } } }$ as $\tilde { y } \in$ $\{ 1 , \cdots , K \}$ . We denote the prediction and one-hot label as $\pmb { p }$ and $\tilde { \pmb { y } } \in \{ 0 , 1 \} ^ { K }$ , respectively. Then the $J S$ -divergence between the $\pmb { p }$ and $\tilde { y }$ becomes:

$$
\mathrm { J S D } ( \tilde { y } , p ) = \frac { 1 } { 2 } p _ { \tilde { y } } \log p _ { \tilde { y } } - \frac { 1 } { 2 } ( 1 + p _ { \tilde { y } } ) \log ( 1 + p _ { \tilde { y } } ) + 1
$$

Where $p _ { \tilde { y } }$ is the abbreviation of $p ( \tilde { y } \vert x )$ .

Theorem.0.2 shows that in a single-classification scenario, the JSD value between $\tilde { y }$ and $\pmb { p }$ is monotonically decreasing on $p _ { \tilde { y } }$ . As the network tends to have a larger posterior probability on true class y $, p _ { \tilde { y } }  1$ or 0 indicates that the sample $\scriptstyle { \mathbf { { \vec { x } } } }$ is more likely to be clean (i.e., $\tilde { y } = y$ ) or mislabeled (i.e., $\tilde { y } \ne y$ ). To verify the effectiveness of JSD metric in discriminating the mislabeled samples from clean samples, we empirically analyze the JSD value distribution of clean and mislabeled samples with symmetric noise. Figure 1 shows the results on CIFAR-10. We observe that the JSD values of clean samples exhibit a peak close to 0, while the mislabeled samples are mostly significantly approaching 1, verifying the effectiveness of JSD metric. Thus the JSD metric satisfies the sample-wise dynamic of $\alpha$ we discussed before. We then perform label correction by linearly combining the noisy label $\tilde { \pmb { y } } _ { i }$ with the sharpening prediction $\hat { \pmb { p } } _ { i }$ , guided by $\alpha _ { i } = \mathrm { J S D } \left( \tilde { \pmb { y } } _ { i } , \pmb { p } _ { i } \right)$ ,

![](images/1572943624cf84d609b6627cfffa969950d4b1285fdc7023b0b253f5d1339910.jpg)  
Figure 1: The distribution of interpolation weight $\alpha$ on CIFAR-10 with Sym- $50 \%$ and $\mathrm { S y m - 9 0 \% }$ label noise using PreAct ResNet-18, and the $\alpha$ is achieved throught normalized JSD metric.

$$
\pmb { \hat { y } } _ { i } = \mathrm { J S D } \left( \tilde { \pmb { y } } _ { i } , \pmb { p } _ { i } \right) \hat { \pmb { p } } _ { i } + \left( 1 - \mathrm { J S D } \left( \tilde { \pmb { y } } _ { i } , \pmb { p } _ { i } \right) \right) \tilde { \pmb { y } } _ { i }
$$

# JSD Normalization for Class-wise Uniformity

DULC utilizes $\alpha _ { i } = \mathrm { J S D } \left( \tilde { \pmb { y } } _ { i } , \pmb { p } _ { i } \right)$ and ensures the samplewise dynamic in label correction. However, as we analyzed before, the standard JSD still metric lacks class-wise uniformity as it relies on the posterior probability of noisy class $\tilde { y } _ { i }$ in Eq.(6). Figure 2(a) shows that the JSD metric exhibits a broader range in easy classes (e.g., class 8 and 9) compared to difficult ones (e.g., class 2,3 and 5). To this end, we employ the normalization on standard JSD to align the range of JSD values of each class.

At the beginning of each training epoch, we compute the maximum and minimum JSD values within a class $c$ , which can be expressed as,

$$
\begin{array} { r l } & { \mathrm { J S D } _ { c } ^ { \operatorname* { m a x } } = \underset { \{ i | \tilde { \pmb { y } } _ { i } = c \} } { \operatorname* { m a x } } \mathrm { J S D } ( \tilde { \pmb { y } } _ { i } , \pmb { p } _ { i } ) } \\ & { \mathrm { J S D } _ { c } ^ { \operatorname* { m i n } } = \underset { \{ i | \tilde { \pmb { y } } _ { i } = c \} } { \operatorname* { m i n } } \mathrm { J S D } ( \tilde { \pmb { y } } _ { i } , \pmb { p } _ { i } ) } \end{array}
$$

Then we perform min-max normalization for each sample $( \pmb { x } _ { i } , \tilde { \pmb { y } } _ { i } )$ with Equation (7):

$$
\mathrm { J S D } _ { \mathrm { n o r m } } ( \tilde { y } _ { i } , { p } _ { i } ) = \frac { \mathrm { J S D } ( \tilde { y } _ { i } , { p } _ { i } ) - \mathrm { J S D } _ { \tilde { y } _ { i } } ^ { \mathrm { m i n } } } { \mathrm { J S D } _ { \tilde { y } _ { i } } ^ { \mathrm { m a x } } - \mathrm { J S D } _ { \tilde { y } _ { i } } ^ { \mathrm { m i n } } }
$$

Through min-max normalization, we align the interpolation weight $\alpha _ { i }$ of the samples in each class to [0,1]. Nevertheless, the interpolation weight $\alpha _ { i }$ for samples from hard classes (especially in high noise ratios) should not be set as high as 1, as their predictions ${ \pmb p } _ { i }$ are not entirely reliable in the early stages. We further adopt a linear decay strategy to constrain

0.7 1.0 0.6 0.8 0.5 0.46 correct   
0.4   
S incorrect   
0.3 noise fitting 0.2 0.2 0.1 0.0 0.0 0 1 2 3 4 5 6 7 8 9 0 50 100 150 200 250 300 Class Epoch (a) JSD distribution (b) The memorization of DULC

the JSD values to a smaller range in the early stage, $[ a , b ]$ (e.g., [0.2,0.8]). The final label correction function in Equation (1) becomes:

$$
\hat { \pmb { y } } _ { i } = \alpha _ { i } \pmb { p } _ { i } + \left( 1 - \alpha _ { i } \right) \tilde { \pmb { y } } _ { i }
$$

$$
\alpha _ { i } = ( 1 - \beta ) \mathrm { J S D } _ { \mathrm { n o r m } } \left( \tilde { y } _ { i } , p _ { i } \right) + \gamma \beta
$$

where $\gamma$ is the scaling factor and $\beta$ is the coefficient that decay linearly with current training epochs $t$ as follow:

$$
\beta _ { t } = \mathbb { I } _ { \{ t \leq N \} } a \left( 1 - \frac { t } { N } \right)
$$

The linear decay strategy allows us to maintain a certain proportion of prediction weight in the early stages. We perform ablation experiments and provide more details in the Appendix B.4. we further combines MixUp augmentation (Zhang et al. 2017) and contrastive learning loss $\mathcal { L } _ { \mathrm { c } }$ (Karim et al. 2022) to mitigate noisy label memorization. The overall training objective is expressed as

$$
\mathcal { L } = \mathcal { L } _ { c e } + \lambda \mathcal { L } _ { \mathrm { c } }
$$

where $\lambda$ is the contrastive loss coefficient and $\mathcal { L } _ { c e }$ is the cross-entropy loss in Eq.2.

# Experiment

# Datasets and Implementation Datails

Extensive experiments are conducted on three manually corrupted datasets with different noisy types (i.e. CIFAR-10/100 (Krizhevsky, Hinton et al. 2009) and TinyImageNet (Le and Yang 2015)) and two real-world noisy datasets (i.e., WebVision (Li et al. 2017) and Clothing1M (Xiao et al. 2015)), to demonstrate the effectiveness of DULC. Both CIFAR-10 and CIFAR-100 contain 50K training images and 10K test images. Tiny-ImageNet is a subset of the ImageNet, featuring 200 classes, each with 500 images, totaling 100K images at a size of $6 4 \times 6 4$ . WebVision comprises 2.4 million images sourced from Flickr and Google, categorized into the same 1,000 classes as ImageNet ILSVRC12. Consistent with previous studies (Li, Socher, and Hoi 2020; Karim et al. 2022), we utilize the initial 50 classes from the Google image subset as the training data. Clothing1M is an unbalanced real-world noisy dataset that contains 1M images with about $3 8 . 4 6 \%$ noisy labels for training and 10K images with clean labels for testing, and its most populated class contains almost 5 times more instances than the smallest one. For CIFAR-10/100, we conduct two types of commonly simulated noisy labels: symmetric noise (Patrini et al. 2017) rates of $20 \%$ , $50 \%$ , $80 \%$ , and $9 0 \% ^ { 1 }$ and asymmetric noise (Li et al. 2019) rates of $10 \%$ , $30 \%$ , and $40 \%$ . Symmetric noise is generated by uniformly flipping the label to the opposite class. Asymmetric noise simulates fine-grained classification, with label flipping limited to similar classes (e.g., $\mathrm { d o g }  \mathrm { c a t } \rrangle$ . For CIFAR100, label flips are applied within each class to transition to the next one within the super-classes. For Tiny-ImageNet, we consider the symmetric noise rates of $20 \%$ and $50 \%$ .

We use the PreAct ResNet-18 (He et al. 2016) architecture for CIFAR10/100 and TinyImageNet in line with other methods. For WebVision and Clothing1M, we take a ResNet50 (He et al. 2016) instead of a more complex InceptionResNetV2 network (Szegedy et al. 2017) in other methods. To obtain strongly augmentated images, we follow the Auto-augment policy described in (Cubuk et al. 2018). For CIFAR10, CIFAR100 and TinyImageNet, we apply CIFAR10-Policy. For WebVision, we use ImageNetPolicy. Additional details of training and parameter setting, as well as more experimental results and discussions, can be found in the Appendix.

# Results

CIFAR-10/100: Tabel 2 shows the average test accuracies for CIFAR-10 and CIFAR-100. DULC consistently outperforms the baseline methods in a wide range of noisy ratios. We observe that DULC achieves significant improvements in all noise ratios except for CIFAR-100 with symmetric$5 0 \%$ and asymmetric- $40 \%$ noise ratios. For the exception, one possible explanation could be that CIFAR-100 has fewer single-class samples (i.e., 500), and the equal number of noisy and clean samples has a higher tolerance for biased division of the dataset, thus bringing greater benefits to the sample selection methods. In particular, we achieve a larger improvement over the SOTA for asymmetric noise. In this more challenging noise ratio, the class labels of the dataset become unbalanced and more consistent with the real scenario. Additionally, following UNICON (Karim et al. 2022), we perform a T-SNE visual comparison on the features learned by the classifier in Appendix C.2. The results show that the features learned by DULC are not more discriminative than UNICON but still promise the SOTA effect. This is due to the smooth transition between interpolated classes, and we provide more discussion in Appendix C.2. It is worth mentioning that DULC can be used as a MixUplike scheme to benefit the standard classifier, as DULC can achieve higher accuracy than standard CE on clean datasets (a.k.a, with 0 noise ratio), which we conduct more experiments and discussions in the Appendix B.5.

Table 2: Comparison with state-of-the-art methods in test accuracy $( \% )$ on CIFAR-10 and CIFAR-100 with symmetric noise The best scores are boldfaced, and the second best ones are underlined.   

<html><body><table><tr><td>Dataset</td><td colspan="6">CIFAR-10</td><td colspan="6"></td></tr><tr><td>Noise type</td><td></td><td colspan="2">Symmetric</td><td colspan="3"></td><td colspan="3">Symmetric</td><td colspan="3">Asymmetric</td></tr><tr><td>Methods/Noise ratio</td><td>20</td><td>50</td><td>80</td><td>90</td><td>10 30</td><td>40</td><td>20</td><td>50</td><td>80</td><td>90</td><td>10</td><td>30 40</td></tr><tr><td>Standard CE</td><td>86.8</td><td>79.4</td><td>62.9</td><td>42.7</td><td>88.8</td><td>81.7 76.1</td><td>62.0</td><td>46.7</td><td>19.9</td><td>10.1</td><td>68.1 53.3</td><td>44.5</td></tr><tr><td>MixUp (2017)</td><td>95.6</td><td>87.1</td><td>71.6</td><td>52.2 93.3</td><td>83.3</td><td>77.7</td><td>67.8</td><td>57.3</td><td>30.8</td><td>14.6</td><td>72.4 57.6</td><td>48.1</td></tr><tr><td>ELR (2020)</td><td>95.8</td><td>94.8 93.3</td><td>78.7</td><td>95.4</td><td>94.7</td><td>93.0</td><td>77.6</td><td>73.6</td><td>60.8 33.4</td><td>77.3</td><td>74.6</td><td>73.2</td></tr><tr><td>DivideMix (2020)</td><td>96.1</td><td>94.6</td><td>93.2</td><td>76.0 93.8</td><td>92.5</td><td>91.7</td><td>77.3</td><td>74.6</td><td>60.2</td><td>31.5 71.6</td><td>69.5</td><td>55.1</td></tr><tr><td>JPL (2021)</td><td>93.5</td><td>90.2</td><td>35.7</td><td>23.4 94.2</td><td>92.5</td><td>90.7</td><td>70.9</td><td>67.7</td><td>17.8</td><td>12.8 72.0</td><td>68.1</td><td>59.5</td></tr><tr><td>MOIT (2021)</td><td>94.1</td><td>91.1</td><td>75.8 70.1</td><td>94.2</td><td>94.1</td><td>93.2</td><td>75.9</td><td>70.1</td><td>51.4</td><td>24.5 77.4</td><td>75.1</td><td>74.0</td></tr><tr><td>Sel-CL (2022)</td><td>95.5</td><td>93.9 89.2</td><td>81.9</td><td>95.6</td><td>95.2</td><td>93.4</td><td>76.5</td><td>72.4</td><td>59.6</td><td>48.8 78.7</td><td>76.4</td><td>74.2</td></tr><tr><td>UNICON (2022)</td><td>96.0</td><td>95.6</td><td>93.9</td><td>90.8 95.3</td><td>94.8</td><td>94.1</td><td>78.9</td><td>77.6</td><td>63.9</td><td>44.8 78.2</td><td>75.6</td><td>74.8</td></tr><tr><td>MILD (2023)</td><td>93.0</td><td>88.7 79.1</td><td></td><td></td><td>1</td><td>89.8</td><td>67.3</td><td>36.0</td><td></td><td></td><td>69.9</td><td>1</td></tr><tr><td>OT-Filter (2023)</td><td>96.0</td><td>95.3 94.0</td><td>90.5</td><td></td><td></td><td>95.1</td><td>76.7</td><td>73.8</td><td>61.8</td><td>43.8</td><td></td><td>76.6</td></tr><tr><td>HMW (2024)</td><td>93.5</td><td>95.2</td><td>93.7</td><td>90.7</td><td>93.5 94.7</td><td>93.7</td><td>76.6</td><td>75.8</td><td>63.4</td><td>43.4</td><td>76.7 76.3</td><td>72.1</td></tr><tr><td>K-SPR (2024)</td><td>95.4</td><td></td><td>84.6</td><td></td><td>- 94.5</td><td>93.6</td><td>77.5</td><td>-</td><td>30.5</td><td>1</td><td>76.3</td><td>73.9</td></tr><tr><td>Bootstrapping (2014)</td><td>86.8</td><td>79.8</td><td>63.3</td><td>42.9</td><td>1</td><td>1</td><td>62.1</td><td>46.6</td><td>19.9</td><td>10.2</td><td></td><td></td></tr><tr><td>M-correction (2019)</td><td>94.0</td><td>92.0</td><td>86.8</td><td>69.1</td><td>89.6 92.2</td><td>91.2</td><td>73.9</td><td>66.1</td><td>48.2</td><td>24.3 67.1</td><td>1 58.6</td><td>47.4</td></tr><tr><td>SELC (2022)</td><td>95.0</td><td></td><td>78.6</td><td></td><td></td><td>92.9</td><td>76.4</td><td>1</td><td>37.2</td><td></td><td>1</td><td>73.6</td></tr><tr><td>DULC (ours)</td><td>96.6</td><td>96.0</td><td>95.0</td><td>93.5</td><td>96.7 95.5</td><td>95.2</td><td>79.4</td><td>76.4</td><td>67.7</td><td>52.8</td><td>79.2 77.8</td><td>75.8</td></tr></table></body></html>

Table 3: Test accuracies $( \% )$ on Tiny-ImageNet dataset under symmetric noise settings. We report the results for other methods directly from (Karim et al. 2022) with the Best and the average (Avg.) test accuracy $( \% )$ over the last 10 epochs.   

<html><body><table><tr><td>Noise (%)</td><td colspan="2">20</td><td colspan="2">50</td></tr><tr><td>Method</td><td>Best</td><td>Avg.</td><td>Best</td><td>Avg.</td></tr><tr><td>Standard CE F-correction (2017) MentorNet (2018) Co-teaching+ (2019) M-correction (2019)</td><td>35.8 44.5 45.7 48.2 57.2</td><td>35.6 44.4 45.5 47.7 56.6</td><td>19.8 33.1 35.8 41.8 51.6</td><td>19.6 32.8 35.5 41.2 51.3</td></tr><tr><td>NCT (2021) OT-Filter (2023) UNICON (2022) DULC (ours)</td><td>58.0 58.1 59.2 58.9</td><td>57.2 57.7 58.4 58.5</td><td>47.8 50.9 52.7</td><td>47.4 50.1 52.4</td></tr></table></body></html>

TinyImageNet: We conduct experiments in $20 \%$ and $5 0 \%$ symmetric noise ratios. Table 3 presents the performance comparison of DULC and other methods. We count the test accuracy both with the best and average accuracy over the last 10 epochs. The results show that we achieve the best or suboptimal performance on all noise ratios. For the suboptimality of DULC (decreased by the average of $0 . 3 \% )$ , our analysis is that the TinyImageNet dataset has more categories and fewer samples per class (500), which leads to more serious prediction confusion in the early stages of the network, resulting in increased errors in label correction. In this scenario, the selection-based method (i.e., UNICON) can alleviate the memorization of noisy labels relatively well.

Table 4: The results on WebVision and ILSVRC12. All methods are trained on WebVision while evaluated on both Webvsion and ILSVRC12 validation set.   

<html><body><table><tr><td>Dataset</td><td colspan="2">WebVision</td><td colspan="2">ILSVRC12</td></tr><tr><td>Method</td><td>Topl</td><td>Top5</td><td>Topl</td><td>Top5</td></tr><tr><td>MentorNet (2018) Co-Teaching (2018) Iterative-CV (2018) DivideMix (2020) ELR(2020)</td><td>63.0 63.6 65.2 77.3 77.8 78.8</td><td>81.4 85.2 85.3 91.6 91.7 93.4 75.3</td><td>57.8 61.5 61.6 75.2 70.3</td><td>79.9 84.7 85.0 90.8 89.8</td></tr><tr><td>RCAL+ (2023) HMW (2024)</td><td>79.6 78.0</td><td>93.4 93.1</td><td>76.3 71.9</td><td>93.7 93.7 92.2</td></tr><tr><td>K-SPR (2024) DULC (ours)</td><td>78.0 79.9</td><td>92.3 93.7</td><td>74.7 76.9</td><td>92.9 93.9</td></tr></table></body></html>

WebVision: We present our experimental results on this dataset in Table 4. All comparison methods use InceptionResNetV2 as the backbone, while DULC adopts a simpler network ResNet50. The results demonstrate Top-1 and Top5 test accuracy on WebVision and ILSVRC12. Despite WebVision’s increased complexity as a real-world dataset, we outperform all baselines, achieving $0 . 3 \%$ (top-1) and $0 . 2 \%$ (top-5) improvement over the suboptimal approach. With a simpler ResNet50 backbone, DULC attains superior performance compared to other methods, affirming the effectiveness of our design.

Clothing1M Tabel 6 presents performance comparison on this real world noisy labeled dataset. We achieve $0 . 1 1 \%$ performance improvement over UNICON (Karim et al. 2022). Clothing1M dataset is unbalanced with greater challenge. The superior result demonstrates that the design of the two criteria to enhancing the robustness of the model for unbalanced datasets.

<html><body><table><tr><td>Noise type</td><td colspan="4">Symmetric</td><td colspan="3">Asymmetric</td></tr><tr><td>Noise ratio</td><td>20%</td><td>50%</td><td>80%</td><td>90%</td><td>10%</td><td>30%</td><td>40%</td></tr><tr><td>DULC w/o weak&strong Aug.</td><td>95.6±0.05</td><td>94.8±0.14</td><td>93.8±1.89</td><td>82.3±2.64</td><td>95.9±0.06</td><td>94.6±0.21</td><td>90.4±0.78</td></tr><tr><td>DULC w/o sharpening (T=1)</td><td>96.2±0.03</td><td>95.8±0.12</td><td>86.7±0.18</td><td>74.7±2.12</td><td>96.5±0.06</td><td>94.8±0.23</td><td>93.4±0.78</td></tr><tr><td>DULC w/o JSD metric</td><td>95.9±0.06</td><td>95.2±0.15</td><td>78.6±2.64</td><td>60.7±3.69</td><td>90.3±0.16</td><td>94.6±0.23</td><td>93.1±0.23</td></tr><tr><td>DULC w/o JSD normalization</td><td>95.2±0.06</td><td>90.9±0.04</td><td>80.3±0.14</td><td>24.6±0.09</td><td>96.3±0.11</td><td>92.3±0.12</td><td>91.5±2.55</td></tr><tr><td>DULC w/o linear decay</td><td>96.5±0.02</td><td>95.9±0.04</td><td>94.8±0.3</td><td>93.6±0.23</td><td>96.6±0.17</td><td>92.9±0.37</td><td>92.2±0.84</td></tr><tr><td>DULC</td><td>96.6±0.02</td><td>96.0±0.04</td><td>95.0±0.03</td><td>93.5±0.24</td><td>96.7±0.16</td><td>95.5±0.34</td><td>95.2±0.67</td></tr></table></body></html>

Table 5: Ablation study for DULC on CIFAR-10: Test accuracy $( \% )$ of different noise ratios. The best scores are boldfaced, and the second best ones are underlined.

<html><body><table><tr><td>Method</td><td>Accuracy(%)</td></tr><tr><td>Cross-Entropy JPL (Kim et al. 2021)</td><td>69.21</td></tr><tr><td>DivideMix (Li, Socher,and Hoi 2020)</td><td>74.15</td></tr><tr><td>ELR (Liu et al. 2020)</td><td>74.76 74.81</td></tr><tr><td>SELC (Lu and He 2022)</td><td>74.01</td></tr><tr><td>UNICON(Karim et al. 2022)</td><td>74.98</td></tr><tr><td>OT-Filter (Feng,Ren,and Xie 2023)</td><td>74.50</td></tr><tr><td>DISC (Li et al. 2023)</td><td>74.79</td></tr><tr><td>DULC (ours)</td><td>75.09</td></tr></table></body></html>

Table 6: Test accuracies $( \% )$ on Clothing1M dataset.

# Ablation Study and Discussions

To study the impact of each component in DULC, we use CIFAR-10 for the ablation studies. Here, we mainly perform studies on the following components: weak and strong augmentation; dynamic JSD metric, JSD normalization and linear decay. For contrastive learning, we examine and discuss them in Appendix B.6. Table 5 shows the contribution of each component into DUCL.

Discussion of augmentation and sharpening. We first study the effect of augmentation and the sharpening function as they both sharpen the prediction according to our analysis. For augmentation, we perform both label correction and training on weakly augmented datasets as ablation. The result shows a degradation of overall performance. We can observe a $1 1 . 2 \%$ drop (from $9 3 . 5 \%$ to $8 2 . 3 \%$ ) in the case of symmetric- $90 \%$ -noise and a $4 . 8 \%$ drop (from $9 5 . 2 ~ \%$ to $9 0 . 4 \%$ in the case of asymmetric-40-noise. For sharpening, we set $T = 1$ to remove. The result shows a significant performance decline, particularly at symmetric-90-noise (from $9 3 . 5 \%$ to $7 4 . 7 \%$ ). We analyze the reason that in high-noise scenarios, the model needs to focus more on learning true classes (i.e., the gradient term $p _ { j } - { \hat { p } } _ { j }$ in Theorem 0.1) rather than suppressing memorization of mislabeled samples.

Discussion of dynamic JSD metric. The $\alpha$ weight of label correction in DULC adjusts dynamically according to the memorization (JSD metric) of each sample. we use the ensemble strategy in SELC for ablation research, where $\begin{array} { r } { \pmb { t } _ { [ k ] } = \alpha ^ { k } \pmb { \hat { y } } + \sum _ { j = 1 } ^ { k } ( 1 - \alpha ) \alpha ^ { k - j } \pmb { p } _ { [ j ] } } \end{array}$ and $\alpha$ is fixed (e.g., 0.9). The resu ts showed similar results to SELC. We observe severe performance degradation of $3 2 . 8 \%$ in the case of symmetric-90-noise. In high noise ratios, increased differences between samples highlight the deficiency of a static weight design, compromising class-wise uniformity and adversely affecting label correction.

Discussion of JSD normalization. We perform label correction by Equation (6) without normalizing the JSD. The result shows that under high noise ratios (symmetric- $90 \%$ - noise and asymmetric- $45 \%$ -noise), the performance drops by as much as $6 8 . 9 \%$ . The significant decline in performance is a result of heightened non-uniformity among classes in high noise ratios. In this case, hard classes become even harder to rectify, leading to a notable rise in the count of false label corrections. Subsequently, these inaccurate corrections frequently spill over to affect other related classes and create a ripple effect. In contrast, DULC can perform successful label correction and avoids the memorization of noisy labels (see Figure 2(b)).

Discussion of linear decay. We fix $\scriptstyle { \beta = 0 }$ in Equation (10) to remove the linear decay. The results show varied performance degradations, especially severe under asymmetric noise. Specifically, we observe a $3 \%$ drop in asymmetric$40 \%$ -noise. This is because the class imbalance under this noise ratio leads to lower prediction accuracy for samples in difficult classes. Still, their weights are amplified due to JSD normalization, resulting in false correction. Linear relaxation enables label correction to retain more information about assigned labels during the early stage of low prediction accuracy. We provide more discussion in Appendix B.4.

# Conclusion

In this paper, we revisit the interpolation scheme for label correction from the perspective of gradient scaling and provide theoretical evidence for two properties of an ideal interpolation weight: sample-wise dynamic and class-wise uniformity. We then use normalized JSD metric as the interpolation weight to meet the two properties and propose a simple yet effective label correction method DULC. Besides, we theoretically ensure the effectiveness of prediction sharpening in label correction and and successfully implemented it. We demonstrate the SOTA performance of DULC with extensive experiments on multiple noisy datasets. Furthermore, we conduct an ablation study to illustrate the individual contributions of each component to DULC.