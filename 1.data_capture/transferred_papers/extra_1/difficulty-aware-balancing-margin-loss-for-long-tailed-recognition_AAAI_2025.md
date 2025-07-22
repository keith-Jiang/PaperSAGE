# Difficulty-aware Balancing Margin Loss for Long-tailed Recognition

Minseok Son\*, Inyong Koo\*, Jinyoung Park, Changick Kim

Korea Advanced Institute of Science and Technology {ksos104, iykoo010, jinyoungpark, changick}@kaist.ac.kr

# Abstract

When trained with severely imbalanced data, deep neural networks often struggle to accurately recognize classes with only a few samples. Previous studies in long-tailed recognition have attempted to rebalance biased learning using known sample distributions, primarily addressing different classification difficulties at the class level. However, these approaches often overlook the instance difficulty variation within each class. In this paper, we propose a difficulty-aware balancing margin (DBM) loss, which considers both class imbalance and instance difficulty. DBM loss comprises two components: a class-wise margin to mitigate learning bias caused by imbalanced class frequencies, and an instance-wise margin assigned to hard positive samples based on their individual difficulty. DBM loss improves class discriminativity by assigning larger margins to more difficult samples. Our method seamlessly combines with existing approaches and consistently improves performance across various long-tailed recognition benchmarks.

Code — https://github.com/quotation2520/dbm ltr

# Introduction

In recent decades, deep neural networks have demonstrated remarkable success in image recognition tasks (Simonyan and Zisserman 2014; He et al. 2016; Szegedy et al. 2015), largely due to the availability of large-scale datasets like ImageNet (Deng et al. 2009). However, real-world datasets often exhibit an imbalanced distribution, known as a longtailed distribution, wherein a few ‘head’ classes contain a large number of samples, while numerous other classes, referred to as ‘tail’ classes, contain significantly fewer samples. This imbalance presents significant challenges: deep learning models, predominantly trained on the abundant majority classes, struggle to effectively learn features for the minority classes. As a result, models tend to underperform on these underrepresented classes, compromising their overall accuracy.

Addressing class imbalance has been a focal point in longtailed recognition (LTR) research. Existing methods have employed various strategies to rebalance the influence of different classes. Re-sampling techniques, such as oversampling (Byrd and Lipton 2019) and undersampling (Drummond, Holte et al. 2003), adjust the occurrence of class samples to create a more balanced training set. Re-weighting approaches (Cui et al. 2019; Menon et al. 2021; Ren et al. 2020) modify class weights or logit values to emphasize learning from difficult minority classes. For instance, the label-distribution-aware margin (LDAM) loss (Cao et al. 2019) introduces larger margins for minority classes to counteract the bias towards majority classes. Despite these advances, many methods focus primarily on class-level imbalance and often overlook variations in difficulty among individual samples within each class. This oversight can lead to suboptimal performance on challenging instances, even within well-represented classes.

![](images/3df98dff4521d7f4e04f0732fef8fcdc1017702a3771744e7f4856397f55259a.jpg)  
Figure 1: Overview of our method. The model is trained to align samples within decision boundaries defined by adaptive margins. (1) Hard positive samples. Misclassified samples identified during training are labeled as hard positive samples. (2) Class-wise margins. Larger margins are assigned to minority classes to ensure sufficient separation from majority classes. (3) Instance-wise Margins. We propose to apply adaptive margins to hard positive samples, considering both class frequency and sample difficulty.

To address this gap, we propose a novel Difficulty-aware Balancing Margin (DBM) loss that considers both instancelevel difficulty and class imbalance. Unlike previous methods that primarily address class-level bias, DBM loss incorporates two key components: a class-wise margin to mitigate imbalance in class frequencies and an instance-wise margin that adapts to the difficulty of individual samples. By assigning additional margins to hard positive samples, our approach enhances class discriminability even more.

Figure 1 illustrates the overview of our method. Here, we consider a binary classification problem where class A has more samples than class B. The decision boundary determined by the classifier is denoted by the black line in Fig. 1(1). The misclassified samples, indicated by a twoline border, are identified as hard positive samples. Our margin loss assigns a tighter decision boundary to each sample, aiming to bring sample features closer to their class centers. First, a class-wise margin of varying sizes is applied to each sample based on its class frequency. As a result, different decision boundaries, $b _ { A }$ and $b _ { B }$ , are defined as shown in Fig. 1(2), with the minority class experiencing a larger displacement in its decision boundary compared to the majority class. For hard positive samples, an additional instance-wise margin is applied. The final decision boundaries for the hard positive samples $\boldsymbol { A 1 }$ , $A 2$ , and $B 1$ in Fig. 1(3) are denoted as $b _ { A 1 }$ , $b _ { A 2 }$ and $b _ { B 1 }$ , respectively. Given that $\boldsymbol { A 1 }$ exhibits a greater angular distance from the class center compared to $A 2$ , $\boldsymbol { A 1 }$ is assigned a larger instance-wise margin, leading to a more shift in $b _ { A 1 }$ relative to $b _ { A 2 }$ . This leads to a higher loss value for difficult samples, encouraging a denser feature distribution within each class.

Our method integrates seamlessly with existing LTR techniques with negligible computational impact and demonstrates consistent performance improvements across multiple benchmarks, including CIFAR-10-LT, CIFAR-100-LT (Cao et al. 2019; Kang et al. 2020b), ImageNet-LT (Liu et al. 2019c), and iNaturalist2018 (Van Horn et al. 2018). Extensive experiments validate our design choices and showcase the effectiveness and robustness of our method.

The main contributions of this paper are summarized as follows:

• We propose the difficulty-aware balancing margin (DBM) loss, which effectively balances learning bias due to class imbalance and sample-level difficulty variation within a class. • Our DBM loss is compatible with various existing longtailed recognition techniques, and incurs no significant additional computational overhead. • When combined with state-of-the-art methods, our approach demonstrates competitive performance on major long-tailed recognition benchmarks.

# Related Work

# Long-tailed Recognition

Long-tailed recognition (LTR) has been extensively explored through multiple perspectives. Conventional approaches focus on rebalancing the bias introduced by imbalanced class influence during training, aiming to mitigate performance degradation for minority classes. Re-sampling methods (Buda, Maki, and Mazurowski 2018; He and Garcia 2009) address the class imbalance by either undersampling majority classes (Drummond, Holte et al. 2003; Tahir, Kittler, and Yan 2012) or oversampling minority classes (Byrd and Lipton 2019; Ando and Huang 2017). Re-weighting methods (Cui et al. 2019; Cao et al. 2019; Ren et al. 2020) propose class-discriminative losses to emphasize the relative contribution of minority classes. Logit compensation methods (Menon et al. 2021; Li, Cheung, and Lu 2022; Ren et al. 2020; Wang et al. 2023, 2024) adaptively adjust logit values based on prior knowledge of the sample distribution for balancing.

Another line of LTR research focuses on enhancing the robustness of representation learning to reduce model bias. Cao et al. (2019) demonstrated that applying class rebalancing methods in the later stages of training can be more effective than conventional one-stage methods. Kang et al. (2020b) proposed decoupling the training of the feature extractor from the classifier, which inspired later twostage approaches (Zhou et al. 2020; Zhong et al. 2021). Augmentation-based methods (Li et al. 2021; Park et al. 2022; Ahn, Ko, and Yun 2023) aim to improve the sample diversity for tail classes. Inspired by the robust feature representation learned through self-supervision (He et al. 2020; Chen et al. 2020), variants of supervised contrastive learning (Khosla et al. 2020) methods have been introduced to LTR (Wang et al. 2021a; Kang et al. 2020a; Li et al. 2022b; Cui et al. 2021; Zhu et al. 2022). Suh and Seo (2023) integrated contrastive learning with logit compensation by introducing a Gaussian mixture likelihood loss, aiming to maximize mutual information between latent features and the ground truth labels. They employed a teacher-student strategy to generate contrast samples using a pre-trained teacher encoder. Ensemble-based methods (Wang et al. 2021b; Cai, Wang, and Hwang 2021; Li et al. 2022a; Tao et al. 2023) exploit the complementary knowledge from multiple experts through various incorporation methods, such as routing (Wang et al. 2021b) and distillation (Li et al. 2022a).

Most LTR studies assume that the tail classes are inherently more difficult to learn and therefore assign more weights to less frequent classes. However, some recent works (Zhao et al. 2022; Sinha and Ohashi 2023) observed that actual class-specific performance does not always correlate with class frequency. In response, they tried to consider classification difficulty in addition to sample distribution for re-weighting. We share a similar motivation and introduce an adaptive margin loss that makes instance-level adjustments based on the angular distance between the positive class center and the sample feature.

# Margin Loss

Large-margin softmax loss (L-Softmax) (Liu et al. 2016) was introduced to enhance feature discrimination by encouraging intra-class compactness and inter-class separability in the embedding space. In the domain of facial recognition, margin losses have been further explored in angular space, utilizing a cosine classifier (Liu et al. 2017; Wang et al. 2018; Deng et al. 2019). These approaches aim to improve discriminativity by optimizing the angular separation between class centers.

Challenges arising from class imbalance have also been addressed within margin-based frameworks. For example, face recognition methods such as fair loss (Liu et al. 2019a) and AdaptiveFace (Liu et al. 2019b), and label-distributionaware margin (LDAM) loss (Cao et al. 2019) for LTR adaptively adjust class-wise margin values or sampling frequencies to mitigate bias. LDAM loss assigns larger margins to minority classes by explicitly incorporating class distribution priors, which helps counteract the imbalance. However, LDAM loss applies a uniform margin to all samples within a class, without accounting for variations in sample difficulty. In contrast, we propose a difficulty-aware balancing margin (DBM) loss, which introduces the consideration of instance difficulty to assign even larger margins to challenging samples. By adapting the margin based on the angular distance between the positive class center and the sample feature, DBM loss provides a more refined approach to margin adjustment, effectively addressing both class imbalance and individual sample difficulty.

# Proposed Method

# Preliminaries

Loss functions for Long-tailed Recognition. The crossentropy loss with softmax function is defined as:

$$
L _ { \mathrm { C E } } = - \log \frac { e ^ { \psi _ { y } ( x ) } } { \sum _ { i } e ^ { \psi _ { i } ( x ) } } .
$$

Here, $\psi _ { i } ( x )$ represents the logit function of the $i$ -th class for sample $x$ , which belongs to the class of index $y$ . For models that utilize a linear classifier, the logit function is given by $\psi _ { i } ( x ) = W _ { i } ^ { \top } f ( x ) + b _ { i }$ , where $f ( x )$ denotes the feature representation of sample $x$ , and $W _ { i }$ and $b _ { i }$ represent the weight and bias of the linear classifier for the $i$ -th class, respectively. Alternatively, a cosine classifier embeds features and class centers in an L2-normalized space, with logits determined by the angular distance between sample features and class centers. Specifically,

$$
\psi _ { i } ( x ) = s \frac { W _ { i } ^ { \top } f ( x ) } { \| W _ { i } \| \| f ( x ) \| } = s \cos { \theta _ { i } } ,
$$

where $s$ is the scaling factor and $\theta _ { i }$ denotes the angular distance between $W _ { i }$ and $f ( x )$ .

In long-tailed recognition (LTR), re-weighting methods address class imbalance by incorporating class frequency $n _ { i }$ into the loss functions. Variants of cross-entropy loss include the class-balanced (CB) loss (Cui et al. 2019) and balanced softmax (BS) (Ren et al. 2020). The class balanced loss $L _ { \mathrm { C B } }$ is formulated as:

$$
L _ { \mathrm { C B } } = - \frac { 1 - \beta } { 1 - \beta ^ { n _ { y } } } \log \frac { e ^ { \psi _ { y } ( x ) } } { \sum _ { i } e ^ { \psi _ { i } ( x ) } } ,
$$

introducing a class-wise weight determined by the effective number of samples given a hyperparameter $\beta$ . The balanced softmax loss $L _ { \mathrm { B S } }$ is:

$$
{ \cal L } _ { \mathrm { B S } } = - \log \frac { e ^ { \psi _ { y } ( x ) + \log p _ { y } } } { \sum _ { i } e ^ { \psi _ { i } ( x ) + \log p _ { i } } } ,
$$

<html><body><table><tr><td>Methods</td><td>m(0y)</td></tr><tr><td>SphereFace (Liu et al.2017)</td><td>cos (mθy)</td></tr><tr><td>CosFace (Wang et al. 2018)</td><td>cos 0y-m</td></tr><tr><td>ArcFace (Deng et al. 2019)</td><td>cos (0y + m)</td></tr><tr><td>LDAM (Cao et al. 2019)</td><td>-1/4 cos0y-mny</td></tr></table></body></html>

Table 1: $\psi _ { y } ^ { m } ( \theta _ { y } )$ used in different margin losses.

where $p _ { i }$ represents the sample proportion of the $i$ -th class over all classes, i.e., $p _ { i } = n _ { i } \bar { / \sum _ { j } n _ { j } }$ . The balanced softmax loss is widely adopted in later LTR studies, such as balanced contrastive learning (BCL) (Zhu et al. 2022) and nested collaborative learning (NCL) (Li et al. 2022a).

Margin-based Variants of Cross-entropy Loss. Margin losses introduce a specialized logit function associated with a margin for the positive class. A margin-based crossentropy loss $L _ { m }$ can be generally formulated as:

$$
L _ { m } = - \log \frac { e ^ { s \psi _ { y } ^ { m } ( \theta _ { y } ) } } { e ^ { s \psi _ { y } ^ { m } ( \theta _ { y } ) } + \sum _ { i \neq y } e ^ { s \cos \theta _ { i } } } ,
$$

where $s \psi _ { y } ^ { m } ( \theta _ { y } )$ denotes the logit function for the positive class incorporating the margin. If $\psi _ { y } ^ { m } ( \theta _ { y } )$ adopts no margin, i.e., $\psi _ { y } ^ { m } ( \theta _ { y } ) = \cos \theta _ { y }$ , $L _ { m }$ is equivalent to $L _ { \mathrm { C E } }$ .

Table 1 provides a summary of various margin-based loss functions and their respective logit formulations. Traditional margin losses (Liu et al. 2017; Wang et al. 2018; Deng et al. 2019) apply a constant margin for all classes. CosFace (Wang et al. 2018) applies a margin to the measured cosine similarity, while ArcFace (Deng et al. 2019) directly adjusts the angular distance. LDAM loss (Cao et al. 2019) follows a similar formulation to CosFace, subtracting a margin that varies with class frequency from the cosine similarity to address the class imbalance problem.

# Difficulty-aware Balancing Margin Loss

Our difficulty-aware balancing margin (DBM) loss comprises two components: a class-wise margin and an instancewise margin. By integrating these two elements, we address both the bias from class imbalance and the variation in instance difficulty within a class. Following prior works (Xiao et al. 2022; Li et al. 2024), we apply the instance-wise margin specifically to hard positive samples. Figure 2 illustrates the margins determined by class frequency and angular distance. Detailed mathematical descriptions of each component are provided below.

Class-wise Margin. The class-wise margin $m _ { C }$ is defined as:

$$
m _ { C } = K \rho _ { y } ^ { - \tau } .
$$

Here, $\rho _ { y } = n _ { y } / n _ { \mathrm { m i n } }$ represents the ratio of the number of samples in class $y$ to the number in the least frequent class. The parameter $\tau$ controls the extent of the margin difference across classes, while $K$ scales the margin. As illustrated in Fig. 2a, $m _ { C }$ is solely based on the class frequency ratio $\rho _ { y }$ . By scaling inversely with $\rho _ { y }$ , minority classes receive a larger margin compared to majority classes, ensuring the least frequent class obtains the maximum margin of $K$ . This helps mitigate performance degradation for minority classes. We have found that setting $\tau = 1$ is effective for our approach.

![](images/50f9cc8af118d29b0f4a01e2583574c07c7675625663cb4ec05b25748f39ae4a.jpg)  
Figure 2: Margins for $K = 0 . 1$ and $\tau = 1$ . Less frequent classes have larger class-wise margin, and more difficult samples have larger instance-wise margin.

Instance-wise Margin. The instance-wise margin addresses varying sample-level difficulties. Samples with lower positive logit values are more prone to misclassification. For our cosine classifier, difficult samples are those whose feature representations are farther from the positive class center in the hypersphere. We quantify the instance difficulty $d _ { I }$ via following equation:

$$
d _ { I } = \frac { 1 - \cos \theta _ { y } } { 2 } .
$$

Here, $d _ { I }$ is determined by the angular distance between the feature representation of the sample and the positive class center $\theta _ { y }$ . A sample with its feature representation exactly at the class center has $d _ { I } = 0$ , while a sample with the feature representation at the maximum distance $\dot { \theta } _ { y } = \pi )$ ) has $d _ { I } =$ 1.

The instance-wise margin $m _ { I }$ is given by:

$$
m _ { I } = m _ { C } \cdot d _ { I } .
$$

As illustrated in Fig. 2b, this margin is determined by both $\rho _ { y }$ and $\theta _ { y }$ , encouraging difficult and less-frequent samples to move more aggressively towards the positive class center.

Loss formulation. Our DBM loss modifies the angular distance by incorporating both margins, similar to the ArcFace approach. Specifically, our logit function for the positive class is formulated as:

$$
s \psi _ { y } ^ { d b m } ( \theta _ { y } ) = s \cos ( \theta _ { y } + m _ { C } + \mathbb { 1 } [ \mathrm { a r g m i n } ( \{ \theta _ { i } \} _ { i = 1 } ^ { N } ) \neq y ] m _ { I } ) ,
$$

where $\mathbb { 1 } [ \cdot ]$ is an indicator function for applying the instancewise margin only to hard positive samples. By substituting this logit function into Eq. (5), we derive the difficulty-aware balancing margin cross-entropy (DBM-CE) loss.

The DBM loss can be easily integrated with various existing LTR methods. For example, it can be combined with the class-balanced loss introduced in Eq. (3) as follows:

$$
{ \cal L } _ { \mathrm { D B M - C B } } = - \frac { 1 - \beta } { 1 - \beta ^ { n _ { y } } } \log \frac { e ^ { s \psi _ { y } ^ { d b m } ( \theta _ { y } ) } } { e ^ { s \psi _ { y } ^ { d b m } ( \theta _ { y } ) } + \sum _ { i \neq y } e ^ { s \cos \theta _ { i } } } .
$$

Similarly, DBM-BS can be derived as:

$$
L _ { \mathrm { D B M - B S } } = - \log \frac { e ^ { s \psi _ { y } ^ { d b m } ( \theta _ { y } ) + \log p _ { y } } } { e ^ { s \psi _ { y } ^ { d b m } ( \theta _ { y } ) + \log p _ { y } } + \sum _ { i \ne y } e ^ { s \cos \theta _ { i } + \log p _ { i } } } ,
$$

reformulating the original balanced softmax loss described in Eq. (4). Note that our method requires adjusting the classifier from a linear to a cosine classifier.

Moreover, Our method is highly versatile and can be incorporated with a range of other LTR techniques. We demonstrate this versatility with various configurations of our method, including DBM-DRW, DBM-BCL, DBMGML, and DBM-NCL. DRW, or deferred re-weighting (Cao et al. 2019), integrates class-balanced loss into the training process at a later stage, allowing DBM-DRW to be implemented by applying $L _ { \mathrm { D B M - C E } }$ and $L _ { \mathrm { D B M - C B } }$ sequentially according to the scheduling policy. Similarly, methods like BCL (Zhu et al. 2022), GML (Suh and Seo 2023) and NCL (Li et al. 2022a), which originally use balanced softmax loss, can incorporate our approach by substituting the classification loss with LDBM-BS.

The integration of DBM loss into existing models does not incur significant additional computational complexity. The class-wise margin $m _ { C }$ is determined in advance based on the known sample distribution, ensuring that this computation does not affect the training time. The instance-wise margin $m _ { I }$ is computed during the logit calculation, leveraging the angular distance $\theta _ { y }$ that is already part of the model’s forward pass. This design ensures that DBM can be incorporated into existing frameworks without introducing substantial overhead.

# Experiments

# Datasets

To evaluate the performance of our proposed method, we conducted experiments on four benchmark long-tailed datasets. The imbalance factor $\rho$ of each dataset is defined as the ratio of training instances between the largest and smallest classes, i.e., $\rho = n _ { \mathrm { m a x } } / n _ { m i n }$ , following previous works (Cao et al. 2019; Kang et al. 2020b).

Long-tailed CIFAR-10 and CIFAR-100. We sampled long-tailed CIFAR datasets from the original CIFAR-10 and CIFAR-100 (Krizhevsky, Hinton et al. 2009) datasets with imbalance factors of 10, 50, and 100 using an exponential down-sampling profile outlined in (Cao et al. 2019; Cui et al. 2019). Evaluations were performed on the original balanced test sets.

ImageNet-LT. ImageNet-LT (Liu et al. 2019c) is a longtailed version of ImageNet-1K (Deng et al. 2009), sampled from a Pareto distribution with $\alpha = 6$ . It comprises 1,000 categories and 115.8K training images, with an imbalanced factor of $\rho = 1 2 8 0 / 5$ .

<html><body><table><tr><td rowspan="3">Method</td><td colspan="3">CIFAR-10-LT</td><td colspan="6">CIFAR-100-LT</td></tr><tr><td colspan="3">Imb.Factor</td><td colspan="3">Imb. Factor</td><td colspan="3">Statistics (IF 100)</td></tr><tr><td>100</td><td>50</td><td>10</td><td>100</td><td>50</td><td>10</td><td>Many</td><td>Med.</td><td>Few</td></tr><tr><td>CE</td><td>78.48</td><td>82.73</td><td>89.91</td><td>44.60</td><td>48.75</td><td>61.98</td><td>73.03</td><td>45.37</td><td>10.53</td></tr><tr><td>LDAM (Cao et al. 2019)</td><td>79.92</td><td>83.84</td><td>90.54</td><td>45.25</td><td>50.16</td><td>62.86</td><td>75.31</td><td>44.00</td><td>11.63</td></tr><tr><td>DBM-CE</td><td>80.84</td><td>84.12</td><td>90.95</td><td>46.53</td><td>51.13</td><td>63.18</td><td>73.89</td><td>46.17</td><td>15.03</td></tr><tr><td>CE-DRW (Cao etal.2019) LDAM-DRW (Cao et al. 2019)</td><td>82.24 82.60</td><td>85.05 85.36</td><td>90.94 91.22</td><td>48.28 48.99</td><td>53.89 54.27</td><td>64.25 64.58</td><td>65.89 66.09</td><td>50.74 50.83</td><td>24.87 26.90</td></tr><tr><td>DBM-DRW</td><td>82.82</td><td>85.83</td><td>91.55</td><td>49.41</td><td>54.69</td><td>64.75</td><td>63.23</td><td>52.66</td><td>29.50</td></tr><tr><td>BS (Ren et al. 2020)</td><td>83.57</td><td>86.45</td><td>91.26</td><td>49.35</td><td>54.79</td><td>63.93</td><td>65.77</td><td>50.14</td><td>29.27</td></tr><tr><td>DBM-BS</td><td>84.60</td><td>87.06</td><td>91.42</td><td>51.30</td><td>55.84</td><td>65.22</td><td>67.29</td><td>50.80</td><td>33.23</td></tr><tr><td>BCL (Zhu et al. 2022)</td><td>82.95</td><td>86.76</td><td>91.57</td><td>50.23</td><td></td><td></td><td>67.14</td><td></td><td>29.23</td></tr><tr><td>DBM-BCL</td><td>84.60</td><td>87.16</td><td>91.69</td><td>51.66</td><td>55.35 55.98</td><td>64.98 65.25</td><td>67.91</td><td>51.31 51.91</td><td>32.40</td></tr><tr><td>GML (Suh and Seo 2023)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DBM-GML</td><td>85.19 85.30</td><td>88.07 88.35</td><td>92.11 92.59</td><td>53.12 53.70</td><td>58.17 58.41</td><td>66.93 67.15</td><td>71.60 72.34</td><td>54.57 54.89</td><td>28.20 30.57</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>NCL (Li et al. 2022a) DBM-NCL</td><td>87.37 87.53</td><td>89.89</td><td>93.15 93.19</td><td>56.68 57.48</td><td>61.65</td><td>69.46 69.75</td><td>73.94 71.49</td><td>56.97 59.06</td><td>36.20</td></tr><tr><td></td><td></td><td>89.90</td><td></td><td></td><td>62.01</td><td></td><td></td><td></td><td>39.30</td></tr></table></body></html>

Table 2: Top-1 accuracy $( \% )$ of ResNet-32 on CIFAR-10-LT and CIFAR-100-LT with the imbalance factor (IF) of 100, 50, and 10.

iNaturalist2018. The iNaturalist2018 dataset (Van Horn et al. 2018) is a large-scale real-world dataset that features a highly long-tailed distribution with an imbalance factor of $\rho = 1 0 0 0 / 2$ . It includes approximately 437K training images and 24.4K validation images gathered from 8,142 finegrained species classes in the wild.

# Implementation Details

For the CIFAR-10-LT and CIFAR-100-LT datasets, we integrated our method with several existing approaches including:

(1) vanilla cross-entropy (CE)   
(2) CE-DRW (Cao et al. 2019), a two-stage training method applying CB loss (Cui et al. 2019).   
(3) BS (Ren et al. 2020), a re-weighting method.   
(4) BCL (Zhu et al. 2022), a supervised contrastive learningbased method.   
(5) GML (Suh and Seo 2023), a mutual information maximization method.   
(6) NCL (Li et al. 2022a), an ensemble-based method.

We ensured a fair comparison by evaluating our models under identical experimental conditions. All models utilized ResNet-32 (He et al. 2016) as the backbone network, while ResNet56 was employed as the teacher network for GML. The SGD optimizer with a momentum of 0.9 and weight decay of $2 { \sqrt { \times 1 0 ^ { - 4 } } }$ was employed, along with a learning rate warm-up for the first five epochs and a cosine annealing scheduler for gradual decay. Data augmentation strategies included Cutout (DeVries and Taylor 2017) and AutoAugment (Cubuk et al. 2019). For BCL, we used an initial learning rate of 0.15 and a batch size of 256. For all other methods, we used an initial learning rate of 0.1 and a batch size of 64. Training was conducted for 200 epochs for most methods, except for NCL, which was trained for 400 epochs. In the case of DRW, class-balanced loss is introduced after 160 epochs. We used a scaling factor $s = 3 2$ for all our experiments, and tuned the hyperparameter for margin scaling $K$ within the range 0.1 to 0.3, adjusting it based on datasets and baselines.

For larger datasets, our method was integrated into BCL and GML. NCL was excluded from this comparison due to its extensive training requirements of 400 epochs. For ImageNet-LT, we utilized ResNet-50 and ResNeXt-50 (Xie et al. 2017) as backbones and trained them for 90 epochs. For iNaturalist2018, we employed ResNet-50 and trained for 100 epochs. In both benchmarks, we set the scaling factor $s$ to 30 and the margin scaling hyperparameter $K$ to 0.1. Further details are in the supplementary materials.

# Experimental Results

Long-tailed CIFAR. Table 2 presents the experimental results for CIFAR-10-LT and CIFAR-100-LT. For CIFAR100-LT with an imbalance factor of 100, we report the accuracy across three groups of classes: ‘Many $( > 1 0 0$ shots),’ ‘Medium $2 0 { \sim } 1 0 0$ shots),’ and ‘Few ( $\phantom { - } < 2 0$ shots).’ To ensure fairness, we have reproduced the performance of each previous method and provided these results in the corresponding cells. Methods incorporating DBM loss are highlighted in gray.

The results demonstrate that DBM consistently provides a significant performance improvement over baseline methods. When applied to CE and CE-DRW, our method achieves superior enhancement compared to LDAM and LDAM-DRW, which solely introduces a class-wise margin.

Table 3: Top-1 accuracy $( \% )$ of ResNet-50 and ResNeXt50 on ImageNet-LT. $\dagger$ and $^ { \ddagger }$ denotes results borrowed from Kang et al. (2020b) and Park et al. (2022), respectively.   

<html><body><table><tr><td>Method CE†</td><td>R50 41.6</td><td>RX50</td></tr><tr><td>T-norm (Kang et al. 2020b) cRT (Kang et al.2020b) LWS (Kang et al. 2020b)</td><td>46.7 47.3 47.7 49.8</td><td>44.4 49.4 49.6 49.9</td></tr><tr><td>LDAM-DRW‡ (Cao et al. 2019) CE-DRW‡ (Cao et al. 2019) BS‡ (Ren et al. 2020) ALALoss (Zhao et al. 2022) DisAlign (Zhang et al. 2021)</td><td>50.1 50.9 52.4 52.9</td><td>53.3</td></tr><tr><td>Difficulty-Net(Sinha and Ohashi 2023) RIDE (3 experts) (Wang et al. 2021b)</td><td>54.0 54.9</td><td>53.4</td></tr><tr><td>BCL (Zhu et al. 2022)</td><td></td><td>56.4</td></tr><tr><td></td><td>56.0</td><td>56.7</td></tr><tr><td>GML (Suh and Seo 2023)</td><td></td><td></td></tr><tr><td></td><td></td><td>58.3</td></tr><tr><td></td><td></td><td></td></tr><tr><td>DBM-BCL</td><td>56.3</td><td>57.4</td></tr><tr><td>DBM-GML</td><td>57.4</td><td>58.6</td></tr></table></body></html>

Notably, DBM-BS surpasses BCL, indicating a substantial performance boost without the additional complexity introduced by BCL’s contrastive learning branch. Although some algorithms show a slight decrease in accuracy for the ‘Many’ group compared to the baseline, our method achieves a notable increase in accuracy for the ‘Medium’ and ‘Few’ groups, demonstrating its effectiveness in mitigating performance bias.

ImageNet-LT and iNaturalist2018. Table 3 shows the performance of DBM-BCL and DBM-GML compared to the existing methods on the ImageNet-LT dataset. We report overall accuracy using ResNet-50 and ResNeXt-50 backbones. For a fair comparison, we evaluated our method against existing works that reported the performance after 90 epochs of training. DBM-BCL outperforms the baseline BCL, with an overall accuracy improvements of $0 . 3 \% \mathrm { p }$ and $0 . 7 \% \mathrm { p }$ for the ResNet-50 and ResNeXt-50 backbones, respectively. Although GML did not report results for the ResNet-50 model, DBM-GML demonstrates an improved performance of $0 . 3 \% \mathrm { p }$ for the ResNeXt-50 backbone.

Table 4 displays the performance comparisons on the iNaturalist2018 dataset. We report overall accuracy and the accuracy of ‘Many,’ ‘Medium,’ and ‘Few’ groups in our experiment. To ensure a fair comparison, we excluded methods that involve extensive additional training (Cui et al. 2021; Li et al. 2022a). Since BCL and GML did not report accuracy for each group, we re-implemented their results using their official code. DBM-BCL and DBM-GML achieve improvements of $0 . 9 \% \mathrm { p }$ and $0 . 8 \% \mathrm { p }$ in overall accuracy, respectively, surpassing the performances of existing methods.

# Analysis

In this section, we analyze the components of the DBM loss to evaluate their contributions to performance improvement. We also investigate the impact of different hyperparameters on the method’s effectiveness. Additionally, we illustrate how the introduced margin enhances intra-class compactness and inter-class separability, thus improving classification performance. All experiments for analysis were conducted on CIFAR-100-LT with an imbalance factor of 100.

Table 4: Top-1 accuracy $( \% )$ of ResNet-50 on iNaturalist2018. † and ‡ denotes results borrowed from Zhou et al. (2020) and Ahn, Ko, and Yun (2023), respectively. ⋆ denotes reproduced results with the official code. RIDE (2 experts) (Wang et al. 2021b) was trained for 100 epochs.   

<html><body><table><tr><td>Methods CEt</td><td>Many Med.</td><td>Few</td><td>All</td></tr><tr><td>T-norm (Kang et al.2020b) cRT (Kang et al. 2020b) LWS (Kang et al. 2020b) LDAM-DRW† (Cao et al. 2019) CE-DRW* (Cao et al. 2019) BS‡ (Ren et al. 2020) DisAlign (Zhang et al. 2021)</td><td>73.9 63.5 65.6 65.3 69.0 66.0 65.0 66.3 68.2 67.3 65.5 67.5 61.6 70.8</td><td>55.5 65.9 63.2 65.5 66.4 67.5 69.9</td><td>61.0 65.6 65.2 65.9 66.1 67.0 67.2 69.5</td></tr><tr><td>RIDE (Wang et al. 2021b) BCL*(Zhu et al. 2022) GML* (Suh and Seo 2023)</td><td>70.2 71.3 68.2 71.3 70.7 72.3</td><td>71.7 71.3 73.6</td><td>71.4 71.0</td></tr><tr><td>DBM-BCL DBM-GML</td><td>65.6 71.8 66.9 71.9</td><td>71.5 73.8</td><td>71.2 71.9 72.0</td></tr></table></body></html>

Table 5: Ablation study for the components of DBM loss. ‘Cosine’ denotes replacing the linear classifier with cosine classifier. $m _ { C }$ and $m _ { I }$ denote class-wise and instance-wise margin, respectively. P and HP represent the cases where instance-wise margin is applied to all positive samples and hard positive samples, respectively.   

<html><body><table><tr><td>Cosine</td><td>mc</td><td>mI</td><td>CE</td><td>BS</td></tr><tr><td></td><td></td><td></td><td>44.60</td><td>49.35</td></tr><tr><td>√</td><td></td><td></td><td>44.29</td><td>49.84</td></tr><tr><td>√</td><td>√</td><td></td><td>45.85</td><td>50.61</td></tr><tr><td>√</td><td>√</td><td>P</td><td>46.38</td><td>50.93</td></tr><tr><td>√</td><td>√</td><td>HP(ours)</td><td>46.53</td><td>51.30</td></tr></table></body></html>

Component Analysis. Table 5 presents the results of our ablation study, which examines the impact of class-wise and instance-wise margins. We integrated these components into two baseline methods: CE and BS (Ren et al. 2020). Our findings reveal that using a cosine classifier alone does not significantly improve performance. However, incorporating a class-wise margin leads to notable gains. Adding the instance-wise margin results in an additional performance increase of approximately $0 . 7 \% \mathrm { p }$ for both loss functions.

We also observed differences in performance based on the way the instance-wise margin is applied. Specifically, the ‘hard positive (HP)’ strategy, where the margin is applied only to misclassified positive samples, yields better results compared to the ‘positive (P)’ strategy, which applies the margin to all positive samples. This indicates that focusing on the difficulty of hard positive samples only rather than all positive samples improves performance more effectively.

![](images/a76691b363947d82b52d853f75fbe68f464ebe50d71887ed29446ca21eeca4ec.jpg)  
Figure 3: Analysis for effects of hyperparameters $\tau$ and $K$ . For all cases, DBM-BS outperforms the baseline BS (Ren et al. 2020) performance of $4 9 . 3 5 \%$ .

Impacts of Hyperparameters. Figure 3 illustrates the effects of various hyperparameters on the performance of DBM-BS. We analyze the impact of $\tau$ and $K$ , which are critical parameters in our method. The results show that while variations in these hyperparameters cause slight performance differences, DBM consistently outperforms the baseline across all settings.

Based on our observation, we fixed $\tau$ at 1.0 throughout all experiments on the long-tailed benchmarks. The optimal value for $K$ varies depending on the method, but setting $K = 0 . 1$ generally yields satisfactory results.

Intra-class Compactness and Inter-class Separability. We apply instance-wise margins to bring hard positive samples closer to their respective class centers, aiming to enhance intra-class compactness. Figure 4 compares the distribution of angular distances between sample features and their positive class centers for the ‘Many’, ‘Medium’, and ‘Few’ groups in BS and DBM-BS. DBM-BS shows a reduction in the mean angular distance of approximately $1 0 ^ { \circ }$ across all groups, indicating enhanced intra-class compactness. This suggests that DBM improves the alignment of sample features with their respective class centers, which may contribute to better performance in classification tasks.

To evaluate inter-class separability, we use the Fisher criterion from Fisher’s linear discriminant analysis (LDA) (Fisher 1936) as a metric to measure the distance between feature distributions of different classes. LDA aims to find a projection vector $W$ that maximizes the separation between classes by projecting the data onto a new axis where the classes are most distinguishable. The Fisher criterion is used to determine the optimal $W$ that maximizes the ratio of the between-class variance to the within-class variance.

The Fisher criterion is defined as:

$$
J ( W _ { i j } ) = \frac { ( \mu _ { i } - \mu _ { j } ) ^ { 2 } } { \sigma _ { i } ^ { 2 } + \sigma _ { j } ^ { 2 } } ,
$$

where $W$ is the projection vector, and $\mu _ { k }$ and $\sigma _ { k } ^ { 2 }$ denote the mean and variance of the projected feature distribution for the $k$ -th class, respectively. The objective is to find $W _ { i j }$ such that the means of the projected classes $\mu _ { i }$ and $\mu _ { j }$ are as far apart as possible while the variances $\sigma _ { i } ^ { 2 }$ and $\sigma _ { j } ^ { 2 }$ are minimized. A higher value of the Fisher criterion $\check { J ( W _ { i j } ) }$ indicates greater separability between the two classes.

![](images/de07766c7029f187a67e009628f7bbae392e3f64304619f1ed0d040b78d80a93.jpg)  
Figure 4: Comparison of BS (Ren et al. 2020) and DBM-BS of the distribution of angular distance between sample features and their positive class centers for ‘Many’, ‘Medium’, and ‘Few’ groups. Dashed horizontal lines denote the quartiles.

Table 6: Analysis for inter-class separability. A larger value indicates better separability.   

<html><body><table><tr><td>Method</td><td>Many</td><td>Med.</td><td>Few</td><td>All</td></tr><tr><td>BS (Ren et al. 2020)</td><td>6.02</td><td>5.98</td><td>5.65</td><td>5.89</td></tr><tr><td>DBM-BS</td><td>6.21</td><td>6.26</td><td>5.95</td><td>6.15</td></tr></table></body></html>

After calculating the optimal projection vectors for all class pairs, we define the separability of a class $S _ { i }$ as:

$$
S _ { i } = \frac { 1 } { C - 1 } \sum _ { j = 1 , j \neq i } ^ { C } J ( W _ { i j } )
$$

where $C$ is the number of classes. Table 6 presents the separability for ‘Many,’ ‘Medium,’ ‘Few,’ and ‘All’ groups. Our observations confirm that DBM enhances inter-class separability across all groups, leading to improved overall classification performance.

# Conclusion

In this work, we propose a difficulty-aware balancing margin (DBM) loss, a novel approach designed to address classlevel imbalance and instance-level difficulty variations in long-tailed datasets. The DBM loss incorporates a classwise margin to mitigate the performance degradation caused by class imbalance and a instance-wise margin to enhance class discriminability by more effectively aligning misclassified samples with their corresponding class centers. Our method integrates effortlessly with existing long-tailed recognition techniques and consistently improves performance across benchmarks. We comprehensively evaluated our method on the long-tailed CIFAR, ImageNet-LT, and iNaturalist2018 datasets, and demonstrated its effectiveness through extensive experiments.

# Acknowledgments

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korean government (MSIT) (NRF-2018R1A5A7025409).