# Pre-Trained Vision-Language Models as Noisy Partial Annotators

Qian-Wei Wang1,2, Yuqiu $\mathbf { X _ { i } ^ { \bullet } } ^ { 1 } $ , Letian Zhang1, Zimo Liu2, Shu-Tao Xia1,2

1Tsinghua Shenzhen International Graduate School, Tsinghua University 2Pengcheng Laboratory {wanggw21, jieyq22, zlt23}@mails.tsinghua.edu.cn, liuzm@pcl.ac.cn, xiast@sz.tsinghua.edu.cn

# Abstract

In noisy partial label learning, each training sample is associated with a set of candidate labels, and the ground-truth label may be contained within this set. With the emergence of powerful pre-trained vision-language models, e.g. CLIP, it is natural to consider using these models to automatically label training samples instead of relying on laborious manual annotation. In this paper, we investigate the pipeline of learning with CLIP annotated noisy partial labels and propose a novel collaborative consistency regularization method, in which we simultaneously train two neural networks, which collaboratively purify training labels for each other, called CoPseudo-Labeling, and perform consistency regularization between label and representation levels. For instance-dependent noise that embodies the underlying patterns of the pre-trained model, our method employs multiple mechanisms to avoid overfitting to noisy annotations, effectively mines information from potentially noisy sample set while iteratively optimizing both representations and pseudo-labels during the training process. Comparison experiments with various kinds of annotations and weakly supervised methods, as well as other pre-trained model application methods demonstrates the effectiveness of method and the feasibility of incorporating weakly supervised learning into the distillation of pretrained models.

Code — https://github.com/Portugas0807/Co-Reg

# Introduction

Partial label learning (PLL) (Lv et al. 2020; Zhang and Yu 2015; Lyu et al. 2021; Wang et al. 2022a; Wu, Wang, and Zhang 2022), a type of weakly supervised learning (WSL), deals with the classification problem where each training example is associated with multiple candidate labels, called partial label, among which only one label is the groundtruth. Due to the fundamental assumption that the groundtruth label must be within the candidate label set, which cannot be satisfied in many scenarios and thus affects the practical applicability of PLL, recent years have seen a growing focus on noisy partial label learning (NPLL) (Qiao et al. 2023; Shi et al. 2023b,a; Wang et al. 2022b). In NPLL, the aforementioned assumption is relaxed, allowing the groundtruth labels of some noisy samples to appear outside their candidate label sets.

Previous research on NPLL typically assumes that the partial annotations of training dataset originate from manually labelling manners. With the emergence of powerful pretrained vision-language models, e.g. CLIP (Radford et al. 2021), which learns from massive ”image-natural language” pairs and can generalize to unseen tasks by leveraging textual categorical description, it is natural to consider using these models to automatically label training samples instead of relying on laborious manual annotation.

In this paper, we investigate the pipeline of employing pre-trained vision-language model CLIP to annotate images of downstream tasks with partial labels, and then training a specialized model using these partial labels. Here, each prompt template corresponds to a classification result of a sample, and the classification results from multiple prompt templates collectively form the candidate label set. Compared to simulated settings such as random noise (also called symmetric noise), which are typically experimented on by previous NPLL research, training on datasets annotated by CLIP becomes significantly more challenging. This is because the noise here is instance-dependent and follows the underlying patterns of the pre-trained model. Specifically, even if a dataset contains a large amount of random noise, since the model cannot learn any effective patterns from it, the algorithm can easily identify that those noisy labels conflict with the current model’s output and consequently correct them by assigning pseudo-labels. However, when dealing with noise from a pre-trained model, the algorithm can easily misinterpret noisy labels as high-confidence true labels, since they may align with the patterns learned from the pre-trained model’s annotations.

To address this issue, we propose a novel Collaborative consistency Regularization (Co-Reg) method. Our method simultaneously trains two neural networks, which collaboratively purify training labels for each other, called CoPseudo-Labeling, and performs consistency regularization between label and representation levels. In Co-PseudoLabeling, we train two neural networks simultaneously and each network divides the pre-trained model’s annotated partial labels into valid and noisy, which are then provided to the other network for training. This approach effectively reduces the confirmation bias that can arise from mimicking the pre-trained model. For samples identified as having noisy partial labels, we regard them as unlabeled samples and aggregate the outputs from multiple data-augmented variants predicted by both neural networks for consistency regularization. Utilizing, rather than discarding, the unlabeled split is crucial for the model to achieve performance improvements, since it largely corrects the errors made by the original pre-trained model on the downstream task. When performing self-training, our method not only uses the obtained pseudo-label as the training objective for the output class distribution but also maintains a representation prototype for each class and then calculates the prototypical similarity distribution of the sample representation and aligns it with the pseudo-label for additional calibration. Furthermore, our method introduces a noise-tolerant supervised contrastive learning module, aimed at enabling the model to learn more discriminative representations, which enhances the advantage of the specialized small model over the general pre-trained model in specific tasks.

Our main contributions can be summarized as:

• To the best of our knowledge, we are the first to investigate the NPLL problem in the context of learning with pre-trained model annotation and propose a novel collaborative consistency regularization algorithm. • We conduct comparative experiments with various kinds of annotations and WSL methods, as well as other pretrained model application methods including zero-shot learning, knowledge distillation, and prompt learning. • We discuss the comparison of introducing WSL with other pre-trained model application paradigms and demonstrate the feasibility of WSL and NPLL on image classification tasks. This inspiration can be further extended to various types of tasks, pre-trained models, and WSL scenarios.

# Related Work

# Noisy Partial Label Learning

PLL (Zhang and Yu 2015; Feng and An 2018; Wang, Li, and Zhou 2019; Wang et al. 2023) is an important branch of weakly-supervised learning, in which each training sample is associated with multiple candidate labels, among which only one is valid. The main difficulty of this problem is recognizing the ground-truth label among other associated false-positive candidates, which is called ”disambiguation” of the partial label. In recent years, data-augmentation-based consistency regularization has been widely applied to PLL, leading to the emergence of a series of impressive methods (Wu, Wang, and Zhang 2022; Wang et al. 2022a; Xia et al. 2023). They achieved performances with only minor differences compared to fully supervised training counterpart, even when the training labels contained a large proportion of false-positives.

However, traditional PLL imposes an overly strict assumption that the ground-truth labels of all samples must be included within the their candidate label sets, which can hardly be met in scenarios such as crowdsourcing and learning with pre-trained model annotation. As a result, there has been a growing tendency to study a more practical extension of PLL, known as NPLL (Lian et al. 2022; Wang et al. 2022b; Xu et al. 2024; Shi et al. 2023b,a; Qiao et al. 2023). NPLL allows the existence of noises of partial labels, i.e. the ground-truth label is not any of the candidate labels. Previous methods usually design specific mechanisms to handle noisy partial samples simultaneously with partial label disambiguation. Some methods (Qiao et al. 2023; Lian et al. 2022; Shi et al. 2023a) incorporate non-candidate labels with high predicted probabilities into the candidate label sets during training. Shi et al. (2023b) divide the samples based on whether they contain noise. Xu et al. (Xu et al. 2024) assign a certain probability to non-candidate labels in the training objective.

However, previous NPLL methods primarily address simulated settings such as random noise and show unsatisfactory performances in the scenario of learning with pretrained model annotations. Xu et al. (Xu et al. 2021) study instance-dependent partial labels by using a neural network trained on ground-truth labels to obtain the probability of flipping false-positive candidate labels. However, this scenario remains a simulated crowdsourcing setting and has not been extended to NPLL.

# Applying CLIP to Downstream Tasks

With the invention of effective multi-modal deep frameworks and training algorithms, as well as significant improvements in computing power, pre-trained visionlanguage models (Li et al. 2022; Bao et al. 2022; Achiam et al. 2023) have demonstrated impressive capabilities on a wide range of tasks. Taking CLIP (Contrastive LanguageImage Pre-Training) (Radford et al. 2021) as an example, it models unified representations of images and natural language by learning massive ”image-text” pairs. During pretraining, semantically related pairs of images and texts are encoded by the image encoder and the text encoder to obtain aligned representations. Because CLIP learns from natural language descriptions, it can leverage the rich contextual information provided by language to understand and categorize new images without having been explicitly trained on them. When presented with a new downstream task, CLIP can interpret the category descriptions which are then matched by the encoded input images in the shared embedding space.

There are two main issues when directly applying CLIP to downstream image classification tasks: 1. For tasks where the input image domain differs significantly from the general domain, CLIP often fails to accurately match the input images with the corresponding category descriptions, resulting in poor performances; 2. In scenarios with limited computational resources, the overhead of executing large model inference may be unsustainable. To enhance the performance and efficiency of pre-trained vision-language models, several adaptation and distillation techniques have been explored.

One approach is prompt learning (Zhou et al. 2022b,a; Khattak et al. 2023), which involves fine-tuning the input textual or visual embeddings of vision-language models. This approach can be seen as an extension of manual prompt engineering, which designing appropriate prompts for better aligning downstream images and textual descriptions by domain experts. Since prompt learning only fine-tunes the embedding vectors while keeping the text and image encoder frozen, it requires only few-shot training samples.

Similarly, Adapter (Houlsby et al. 2019; Gao et al. 2024) has been proposed as a lightweight adaptation mechanism for pre-trained models. Adapters allow the model to learn task-specific information while retaining the majority of its pre-trained parameters. This approach offers a more efficient alternative to fully fine-tuning, enabling quick adaptation to new tasks without the need of large amount of data and significant computational overhead.

Knowledge distillation is a technique where a smaller, student model is trained to replicate the behavior of a larger, teacher model. This approach aims to transfer knowledge from the large, often cumbersome models to more compact and efficient versions, maintaining high performance while reducing computational requirements. (Hinton, Vinyals, and Dean 2015) laid the groundwork for this technique, demonstrating that distillation can significantly compress models without substantial loss in accuracy.

# CLIP Annotated Partial Labels

This section introduces how we use CLIP to annotate downstream image datasets. We use a collection of prompt templates, denoted as $\{ \mathcal { T } _ { 1 } ( \cdot ) , \mathcal { T } _ { 2 } ( \cdot ) \ldots \mathcal { T } _ { d } ( \cdot ) \}$ , and combine them with the class names of images to form the textual input. The template here is like: ”a photo of a $\{ \}$ .”, where $\{ \}$ is replaced with class names. We denote the class names as $\{ n _ { 1 } , \bar { n } _ { 2 } , \dots , n _ { C } \}$ , and denote the combination of $j$ -th class and $i$ -th template as $\mathcal { T } _ { i } ( n _ { j } )$ .

For each template $\mathscr { T } _ { i } ( \cdot )$ , we combine it with all class names and then take them as the input of CLIP text encoder and obtain $C$ textual representations $\{ t _ { 1 } , t _ { 2 } , \ldots , t _ { C } \}$ , where $t _ { j } = \mathrm { T e x t E n c o d e r } ( \mathcal { T } _ { i } ( n _ { j } ) )$ . Input the training image into the CLIP Image Encoder and obtain image representation $r$ . Then, we can predict the probabilities of the image belonging to different classes under this prompt template as $p _ { i } = { \mathrm { s o f t m a x } } ( r t _ { 1 } , r t _ { 2 } , \dots , r t _ { C } )$ .

We then obtain the numerical label from each predicted probabilities $\hat { p } _ { i } = \arg \operatorname* { m a x } p _ { i }$ , and deem each prompt template’s label as a candidate and form the partial label $Y =$ $\mathbf { \bar { \Psi } } ( y _ { 1 } , y _ { 2 } , \ldots , y _ { C } ) \in \{ 0 , 1 \} ^ { C }$ , in which $y _ { j } = 1$ if $j \in S$ and $y _ { j } ~ = ~ 0$ if $j \not \in \ S$ , and $S$ denotes the candidate label set formed by numerical label $\hat { p } _ { i }$ .

In experiments, we also compare the methods that annotate noisy single-labels by averaging the predicted probabilities $p _ { i }$ of all prompt templates and then train on these labels with corresponding algorithms. We found that annotating partial labels achieves better results, especially under extreme circumstances when most prompt templates fail to provide satisfactory predictions. And at this time, as long as one prompt template makes a correct prediction, the prediction will be included in the candidate label set and the difficult of the algorithm to recognize it as the correct label with the help of consistency regularization is greatly decreased. This is very helpful when the characteristic of downstream task is unknown and prompt engineering can hardly be performed.

# Methodology Supervised Training

To avoid overfitting to the pre-trained model’s outputs, as is common in traditional unsupervised knowledge distillation, our method applies supervised training on the pre-trained model’s annotations for only a few warm-up epochs. We adopt the partial cross-entropy loss as the supervised loss, as well as the negative entropy to prevent from overfitting.

$$
\begin{array} { c } { { L _ { s u p } = - \log { \displaystyle \sum _ { j = 1 } ^ { C } } y _ { j } f _ { j } ( x ) , } } \\ { { L _ { n e g } = \displaystyle \sum _ { j = 1 } ^ { C } f _ { j } ( x ) \log f _ { j } ( x ) , } } \\ { { L _ { w a r m } = L _ { s u p } + L _ { n e g } . } } \end{array}
$$

Afterward, our method stops supervised learning and relies solely on consistency regularization.

# Co-Pseudo-Labeling

Our method simultaneously trains two neural networks (denoted as $f ( x ; \theta _ { 1 } )$ and $f ( x ; { \dot { \theta } } _ { 2 } ) )$ , which collaboratively purify training labels for each other and obtain the pseudo-labels. The advantage of this approach is that it can effectively reduce the confirmation bias that can arise from mimicking the pre-trained model annotation comparing to the usage of selfgenerated pseudo-labels. In the following, we take the example of using $f ( x ; \theta _ { 1 } )$ to provide pseudo-labels for $f ( x ; \theta _ { 2 } )$ .

Firstly, we attempt to divide the provided partial labels as valid or noisy, i.e., whether the ground-truth labels are in the candidate label sets. Drawing inspiration from the minimalloss criterion (Arazo et al. 2019; Chen et al. 2019) which assumes that noise-free samples are easier to learn. We speculate that if the partial label of the current sample is valid, the model warm-up trained using supervised loss can predict the sample to a category within its candidate label set with a higher probability. Our method adopts two types of data augmentation (Sohn et al. 2020; Berthelot et al. 2020), i.e., weak data augmentation $\mathrm { A u g } _ { w } ( \cdot )$ and strong data augmentation $\mathrm { A u g } _ { s } ( \cdot )$ . Details are shown in the Appendix.

Specifically, we calculate the following division loss over the predicted probabilities of all samples with weak data augmentation $\left\{ L _ { d i v } ( \boldsymbol { \mathrm { A u g } } _ { w } ( x ^ { i } ) ; \boldsymbol { \theta } _ { 1 } ) \right\} _ { i = 1 } ^ { N ^ { \bullet } }$ .

$$
L _ { d i v } ( x ; \theta ) = - \log f _ { j } ( x ; \theta ) , j = \underset { j \in \mathcal { V } , y _ { j } = 1 } { \arg \operatorname* { m a x } } f _ { j } ( x ; \theta ) ,
$$

where $f _ { j } ( x ; \theta )$ indicates the predicted probability on the $j$ -th category of neural network with parameter $\theta , \mathcal { D }$ represents the label space and $N$ is total number of training samples. Then, we use a two-component Gaussian mixture model (GMM)(Permuter, Francos, and Jermyn 2006) to fit the above losses to classify the whole training set into a partial split whose annotated partial labels are assumed to be valid with a probability $\hat { w } ^ { i }$ , and an unlabeled split whose annotated partial labels are assumed to be nonvalid and discarded. We use $\mathcal { P } = \{ ( x ^ { i } , p ^ { i } ) \}$ to denote the partial split, and $\mathcal { U } \ : = \ : \{ x ^ { i } \}$ to denote the unlabeled split.

$p ^ { i } \ = \ ( p _ { 1 } ^ { i } , p _ { 2 } ^ { i } , . . . , p _ { C } ^ { i } )$ is calculated by re-scaling the predicted label distribution of $x ^ { i }$ with Eq.5, which eliminates the probabilities outside the candidate label set.

$$
p _ { j } ^ { i } = \frac { y _ { j } ^ { i } \cdot f _ { j } ( \mathsf { A u g } _ { w } ( x ^ { i } ) ; \theta _ { 1 } ) } { \sum _ { k = 1 } ^ { C } y _ { k } ^ { i } f _ { k } ( \mathsf { A u g } _ { w } ( x ^ { i } ) ; \theta _ { 1 } ) } , \quad \mathrm { f o r } j = 1 , 2 , \ldots , C .
$$

For more robust and generalizable pseudo-labels, we combine the predicted label distributions from both networks to obtain the fused pseudo-labels $p ^ { \prime }$ for training. We combine $p ^ { i }$ with the average predicted probabilities of $K$ weakly-augmented inputs from $f ( x ; \theta _ { 2 } )$ . The confidences of the validity of the partial labels for the samples in the partial split, i.e. $\hat { w } ^ { i }$ , are taken as the fusion weights of the predicted label distributions from $f ( x ; \theta _ { 1 } )$ . For samples in the unlabeled split, we combine the average predicted probabilities from both networks and set both weights to 0.5. The fusion process can be expressed as:

$$
p ^ { ' i } = \left\{ \begin{array} { l l } { w ^ { i } \cdot p ^ { i } + ( 1 - w ^ { i } ) \cdot \bar { p } _ { 2 } ^ { i } , } & { \mathrm { i f ~ } x ^ { i } \in \mathcal { P } ; } \\ { ( \bar { p } _ { 1 } ^ { i } + \bar { p } _ { 2 } ^ { i } ) / 2 , } & { \mathrm { i f ~ } x ^ { i } \in \mathcal { U } . } \end{array} \right.
$$

Here, $\begin{array} { r } { \bar { p } _ { 1 ( 2 ) } ^ { i } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } f ( \operatorname { A u g } _ { w } ( x ^ { i } ) ; \theta _ { 1 ( 2 ) } ) } \end{array}$ represents the average predicted probabilities of $K$ weakly-augmented inputs of one of the two networks. Ablation experiments show that the exploitation of the unlabeled split is of crucial importance for achieving performance improvements.

Finally, the pseudo-labels are sharpened with a temperature of $T$ to obtain more discriminative label distributions,

$$
\tilde { p } _ { j } ^ { i } = \frac { ( p _ { j } ^ { ' i } ) ^ { 1 / T } } { \sum _ { k = 1 } ^ { C } ( p _ { k } ^ { ' i } ) ^ { 1 / T } } , \quad \mathrm { f o r ~ } j = 1 , 2 , \dots , C .
$$

# Self-Training

We perform self-training between the obtained pseudolabels and predicted label distribution as well as the pseudolabels and prototypical similarity distribution. Firstly, we use the pseudo-labels as the training objective of the predicted class distribution. We choose the cross-entropy loss as the training objective on $\mathcal { P }$ and the mean square error on $\mathcal { U }$ due to its noisy-tolerant property:

$$
L _ { \mathrm { c r } } ( \boldsymbol { x } ; \theta ) = \left\{ \begin{array} { l l } { - \sum _ { j = 1 } ^ { C } \tilde { p } _ { j } \log f _ { j } ( \boldsymbol { x } ; \theta ) , } & { \mathrm { i f } \boldsymbol { x } \in \mathcal { P } ; } \\ { \sum _ { j = 1 } ^ { C } ( \tilde { p } _ { j } - f _ { j } ( \boldsymbol { x } ; \theta ) ) ^ { 2 } , } & { \mathrm { i f } \boldsymbol { x } \in \mathcal { U } . } \end{array} \right.
$$

Secondly, we adopt the prototypical similarity alignment which enforces the consistency between the pseudo-labels and the sample representation. We project the output representations of image $x ^ { i }$ of the two neural networks to a shared embedding space through a two-layer MLP with L2 normalization, respectively (See Fig.1), obtaining $z ^ { i } =$ $g ( f ( \mathrm { A u g } _ { w } ( x ^ { i } ) ; \bar { \theta } ) )$ , in which $g ( \cdot )$ represents the MLP projector and $\theta$ represents the neural network parameters $\theta$ , excluding the last fully-connected layer. During the training process, we maintain a cluster center for each category in the shared representation space that represents the representation of the category, called ”prototype”, denoted by $\mathbf { \bar { \{ } }  o _ { j } \} _ { j = 1 } ^ { C }$ . We believe that different data augmentation variants of the

Strongly-augmented P Net1' Proj1'(.) Ⅲ- Net1 Proj1(-) Z1 Noisy Supervised Contrastive Weakly-augmented 國 Update Primtypical Alignment Weakly-augmented 4 Net2 Proj2(-) Z2 Noisy Supervised Contrastive Net2' Proj2'-) Strongly-augmented

same sample should maintain consistent distributions between label space and representation space. Specifically, our method calculate the similarity distribution over the representation of current image and class prototypes as $s ^ { i } \ =$ softmax $( z ^ { i } o _ { 1 } , z ^ { i } o _ { 2 } , \dots , z ^ { \bar { i } } o _ { C } )$ , which is then aligned to the pseudo-label distribution $\tilde { p ^ { i } }$ . We choose KL-Divergence for samples in partial split $\mathcal { P }$ , which have a much higher pseudoaccuracy and mean square error for samples in unlabeled split $\mathcal { U }$ . The loss functions for prototypical similarity alignment can be written as:

$$
\begin{array} { r } { L _ { \mathrm { p r o t } } ( x ; \theta ) = \left\{ \begin{array} { l l } { \sum _ { j = 1 } ^ { C } \tilde { p } _ { j } ^ { T ^ { \prime } } ~ \log ( \tilde { p } _ { j } ^ { T ^ { \prime } } / s _ { j } ^ { T ^ { \prime } } ) , } & { \mathrm { i f ~ } x \in \mathcal { P } ; } \\ { \sum _ { j = 1 } ^ { C } ( \tilde { p } _ { j } - s _ { j } ) ^ { 2 } , } & { \mathrm { i f ~ } x \in \mathcal { U } . } \end{array} \right. } \end{array}
$$

The class prototypes are momentum updated during training with Eq.10.

$$
o _ { j } = \gamma o _ { j } + ( 1 - \gamma ) z ^ { i } , \quad j = \underset { j \in \mathcal { Y } } { \arg \operatorname* { m a x } } \tilde { p } _ { j }
$$

# Noisy Contrastive Learning

To further exploit from the data distribution property of downstream unlabeled images while enhancing the model’s representation ability, we employ the noisy supervised contrastive learning.

In addition, we utilize contrastive learning to pull together representations of samples from the same class while pushing apart those from different classes, enabling the model to encode more discriminative features on downstream data. In implementation, we adopt the MoCo (He et al. 2020) framework, in which a large-size ”first-in-first-out” queue of image representations encoded by the momentum updated copy of our model is maintained. We select positive and negative set for the current image representation from the representation queue by their pseudo-labels and optimize the following noisy-tolerant contrastive loss:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { n c o n t } } ( x ; \theta ) = - \displaystyle \frac { 1 } { | P ( x ) | } \sum _ { z _ { + } \in P ( x ) } } \\ & { \log \displaystyle \frac { \exp ( z ^ { \top } z _ { + } / T ^ { \prime \prime } ) } { z _ { + } \in P ( x ) } \exp ( z ^ { \top } z _ { + } / T ^ { \prime \prime } ) + \displaystyle \sum _ { z _ { - } \in N ( x ) } \exp ( z ^ { \top } z _ { - } / T ^ { \prime \prime } ) } \end{array} ,
$$

Table 1: Accuracy comparisons of all comparing methods on CLIP annotated datasets, best performances in bold. Zeroshot(train) and Zero-shot(test) separately indicate the performance of zero-shot learning in train and test dataset splits.   

<html><body><table><tr><td>Methods</td><td>CIFAR-10</td><td>CIFAR-100</td><td>SVHN</td><td>FMNIST</td><td>EuroSAT</td><td>FER2013</td><td>GTSRB</td></tr><tr><td>Zero-Shot(train)</td><td>88.40%</td><td>61.83%</td><td>9.34%</td><td>62.57%</td><td>32.26%</td><td>41.47%</td><td>24.87%</td></tr><tr><td>Zero-Shot*(train)</td><td>89.09%</td><td>62.75%</td><td>9.33%</td><td>65.81%</td><td>30.61%</td><td>46.55%</td><td>25.53%</td></tr><tr><td>Zero-Shot(test)</td><td>88.51%</td><td>61.55%</td><td>8.63%</td><td>61.46%</td><td>31.49%</td><td>42.07%</td><td>25.14%</td></tr><tr><td>Zero-Shot*(test)</td><td>89.01%</td><td>62.74%</td><td>8.82%</td><td>65.14%</td><td>30.78%</td><td>46.72%</td><td>25.57%</td></tr><tr><td>KD</td><td>87.39%</td><td>56.31%</td><td>8.01%</td><td>65.91%</td><td>35.24%</td><td>44.89%</td><td>25.70%</td></tr><tr><td>KD*</td><td>87.74%</td><td>56.80%</td><td>8.21%</td><td>67.60%</td><td>34.13%</td><td>46.43%</td><td>26.98%</td></tr><tr><td>DivideMix</td><td>93.32%</td><td>65.76%</td><td>16.46%</td><td>71.60%</td><td>42.41%</td><td>44.31%</td><td>30.07%</td></tr><tr><td>DivideMix*</td><td>93.83%</td><td>66.03%</td><td>17.16%</td><td>74.21%</td><td>37.74%</td><td>47.42%</td><td>32.69%</td></tr><tr><td>CR-DPLL</td><td>84.20%</td><td>60.05%</td><td>6.82%</td><td>71.27%</td><td>8.85%</td><td>16.75%</td><td>27.16%</td></tr><tr><td>ALIM-Onehot</td><td>93.18%</td><td>64.60%</td><td>17.06%</td><td>72.42%</td><td>34.57%</td><td>52.77%</td><td>31.06%</td></tr><tr><td>ALIM-Scale</td><td>93.59%</td><td>64.61%</td><td>20.66%</td><td>72.36%</td><td>37.11%</td><td>52.51%</td><td>31.81%</td></tr><tr><td>Co-Reg</td><td>94.06%</td><td>71.04%</td><td>46.57%</td><td>76.28%</td><td>65.54%</td><td>50.26%</td><td>41.18%</td></tr><tr><td>CoOp-2</td><td>74.25%</td><td>46.08%</td><td>15.10%</td><td>70.42%</td><td>53.74%</td><td>33.33%</td><td>22.53%</td></tr><tr><td>CoOp-4</td><td>75.19%</td><td>46.14%</td><td>20.90%</td><td>72.77%</td><td>60.44%</td><td>39.81%</td><td>20.01%</td></tr><tr><td>CoOp-8</td><td>75.66%</td><td>48.41%</td><td>28.18%</td><td>76.10%</td><td>68.52%</td><td>46.22%</td><td>21.02%</td></tr><tr><td>CoOp-16</td><td>75.00%</td><td>52.18%</td><td>27.14%</td><td>78.99%</td><td>75.78%</td><td>46.81%</td><td>25.55%</td></tr></table></body></html>

where $P ( x )$ and $N ( x )$ separately denote the set of selected positive and negative examples for image $x$ , $T ^ { \prime \prime } \geq 0$ is the temperature. We treat $\boldsymbol { x } \in \mathcal { P }$ as confident samples and $x \in \mathcal { U }$ as lacking of confidence and applying the selection strategy for $P ( x )$ and $N ( x )$ in (Wang et al. 2024).

# Experiments

# Experimental Setup

We conduct experiments on several image classification benchmarks: CIFAR-10, CIFAR-100 (Krizhevsky, Hinton et al. 2009), SVHN (Goodfellow et al. 2013a), FashionMNIST (Xiao, Rasul, and Vollgraf 2017), EuroSAT (Helber et al. 2019), FER2013 (Goodfellow et al. 2013b) and GTSRB (Houben et al. 2013). We annotate the images of these datasets with CLIP ViT-B/32 in the manner of partial label or single label following Section 3. For CIFAR10, CIFAR-100, Fashion-MNIST and EuroSAT, we use the class descriptions from PyTorch; and for SVHN, FER2013 and GTSRB, their class descriptions are manually assigned and are the same for all comparison methods. The prompt templates we used are listed in Appendix.

We compare the performances of our method with various types of WSL methods under partial and single label annotations: DivideMix (Li, Socher, and Hoi 2020), CRDPLL (Wu, Wang, and Zhang 2022), ALIM-Onehot and ALIM-Scale (Xu et al. 2024), in which CR-DPLL, ALIMOnehot and ALIM-Scale learn from partial label annotations and DivideMix learn from single label annotations. Also, we compare with other pre-trained application paradigms: zeroshot, unsupervised knowledge distillation (KD) and fewshot fine-tuning. Zero-shot, KD and WSL require no human labeling while few-shot fine-tuning is performed with 2, 4, 8, and 16 labeled samples per class. We choose prompt learning method CoOp (Zhou et al. 2022b) as the few-shot finetuning comparing method. For zero-shot learning, KD and DivideMix, we record the performances with single prompt template ”a photo of a $\{ \} ^ { \bar { \mathfrak { s } } }$ as well as using the average of predicted probabilities of multiple prompt templates for fair comparison (superscript with asterisk).

The average amount of candidate labels per training sample and the proportions of ground-truth label being outside of the candidate sets, i.e. $\eta$ , is recorded for partially annotated datasets in Table 2.

We use the PreAct ResNet-18 (He et al. 2016) as the backbone for all comparing methods. The training batch-size is 256, and the number of warm up and total epochs are chosen from 50 or 100 and 100 or 800, respectively. The number of weakly-augmented inputs for co-pseudo-labeling is $K = 2$ and the sharpening temperature is $T = 0 . 5$ . The dimension of projected representations is 128, and the length of MoCo queue is 8192. The loss weights are $\lambda _ { 1 } = 0 . 1$ , $\lambda _ { 2 } = 0 . 1$ The experiments are all carried on NVIDIA 3090 GPUs.

# Main Results

Our method improves performance over zero-shot learning on all experimental datasets and outperforms $\mathbf { C o O p }$ on five of the seven datasets, even though CoOp uses human annotated task-relevant labels. On the other two datasets, $\mathbf { C o O p }$ performs worse than our method at 2 and 4 shots, while achieves better performances at 8 and 16 shots.

Among the comparing paradigms, only KD and WSL can obtain smaller-size deployment model, which is very flexible when facing application scenarios with restrained inference resources. However, student models distilled from the supervision of pre-trained teacher can hardly outperforms the teacher models without task-related training examples. Our method outperforms unsupervised KD on all experimental datasets.

<html><body><table><tr><td>Datasets</td><td>CIFAR-10</td><td>CIFAR-100</td><td>SVHN</td><td>FMNIST</td><td>EuroSAT</td><td>FER2013</td><td>GTSRB</td></tr><tr><td>Avg. Candi</td><td>1.39</td><td>2.36</td><td>2.41</td><td>1.58</td><td>3.26</td><td>2.38</td><td>2.84</td></tr><tr><td>m</td><td>4.79%</td><td>21.50%</td><td>61.61%</td><td>22.35%</td><td>32.89%</td><td>21.49%</td><td>58.22%</td></tr></table></body></html>

Table 2: The average number of candidate labels associated with each sample in partially annotated datasets, as well as the proportion of noisy partial labels.   
Table 3: Accuracy comparisons on synthetic NPLL datasets, best performances in bold.   

<html><body><table><tr><td rowspan="2">Methods</td><td colspan="3">q = 0.01</td><td colspan="3">q= 0.03</td><td colspan="3">q = 0.05</td></tr><tr><td>η=0.1</td><td>η=0.2</td><td>η=0.3</td><td>n=0.1</td><td>η=0.2</td><td>n= 0.3</td><td>η=0.1</td><td>η=0.2</td><td>n= 0.3</td></tr><tr><td>CC</td><td>53.63%</td><td>48.84%</td><td>45.50%</td><td>51.85%</td><td>47.48%</td><td>43.37%</td><td>50.64%</td><td>45.87%</td><td>40.87%</td></tr><tr><td>RC</td><td>52.73%</td><td>48.59%</td><td>45.77%</td><td>52.15%</td><td>48.25%</td><td>43.92%</td><td>46.62%</td><td>45.46%</td><td>40.31%</td></tr><tr><td>LWC</td><td>53.16%</td><td>48.64%</td><td>45.51%</td><td>51.69%</td><td>47.60%</td><td>43.39%</td><td>50.55%</td><td>45.85%</td><td>39.83%</td></tr><tr><td>LWS</td><td>56.05%</td><td>50.66%</td><td>45.71%</td><td>53.59%</td><td>48.28%</td><td>42.20%</td><td>45.46%</td><td>39.63%</td><td>33.60%</td></tr><tr><td>PiCO</td><td>68.27%</td><td>62.24%</td><td>58.97%</td><td>67.38%</td><td>62.01%</td><td>58.64%</td><td>67.52%</td><td>61.52%</td><td>58.18%</td></tr><tr><td>CR-DPLL</td><td>68.12%</td><td>65.32%</td><td>62.94%</td><td>67.53%</td><td>64.29%</td><td>61.79%</td><td>67.17%</td><td>64.11%</td><td>61.03%</td></tr><tr><td>PiCO+</td><td>75.04%</td><td>74.31%</td><td>71.79%</td><td>74.68%</td><td>73.65%</td><td>69.97%</td><td>73.06%</td><td>71.37%</td><td>67.56%</td></tr><tr><td>IRNet</td><td>71.17%</td><td>70.10%</td><td>68.77%</td><td>71.01%</td><td>70.15%</td><td>68.18%</td><td>70.73%</td><td>69.33%</td><td>68.09%</td></tr><tr><td>ALIM-Scale</td><td>77.37%</td><td>76.81%</td><td>76.45%</td><td>77.60%</td><td>76.63%</td><td>75.92%</td><td>76.86%</td><td>76.44%</td><td>75.67%</td></tr><tr><td>ALIM-Onehot</td><td>76.52%</td><td>76.55%</td><td>76.09%</td><td>77.27%</td><td>76.29%</td><td>75.29%</td><td>76.87%</td><td>75.23%</td><td>74.49%</td></tr><tr><td>Co-Reg</td><td>78.13%</td><td>78.01%</td><td>77.20%</td><td>77.16%</td><td>76.85%</td><td>75.71%</td><td>76.30%</td><td>74.91%</td><td>73.45%</td></tr></table></body></html>

Table 4: Ablation experiments on different degenerations of Co-Reg.   

<html><body><table><tr><td>w/o Co-PL</td><td>w/ SupCont</td><td>w/o proto</td><td>w/o U</td><td>Co-Reg</td></tr><tr><td>68.40%</td><td>67.96%</td><td>69.81%</td><td>65.32%</td><td>71.04%</td></tr></table></body></html>

Our method outperforms other WSL methods on all datasets except one, on which ALIM-Onehot performs the best. The results demonstrate the effectiveness of our method and partial label annotation. CLIP’s performance on the SVHN dataset is extremely poor. It predicts almost all samples as ”number $0 ^ { , , }$ , making it impossible to obtain an effective model using ordinary training algorithms. However, though most prompt templates fail, one of them can achieve an accuracy rate over $2 5 \%$ and the correctly predicted labels are included into the candidate label sets, which facilitates the training of our method.

# Synthetic Datasets

We also conduct the experiments on synthetic datasets of CIFAR-100, following the generation process used by the previous method (Xu et al. 2024). First, we generate partially labeled datasets by flipping negative labels $\bar { y } \ne y$ to false positive labels with a probability $q = P ( { \bar { y } } \in Y | { \bar { y } } \neq$ $y$ ). Then, we generate noisy partially labeled datasets by randomly substituting the ground-truth label with a noncandidate label with a probability $\eta = P ( y \not \in Y )$ for each sample. We choose the noise level $\eta$ from $\{ 0 . 1 , \dot { 0 } . 2 , 0 . 3 \}$ , and consider $q \in \{ 0 . 0 1 , 0 . 0 3 , 0 . 0 5 \}$ for CIFAR-100.

We compare our method with ten PLL and NPLL methods, i.e. CC (Feng et al. 2020), RC (Feng et al. 2020), LWC (Wen et al. 2021), LWS (Wen et al. 2021), PiCO (Wang et al.

2022a), CR-DPLL (Wu, Wang, and Zhang 2022), ${ \mathrm { P i C O } } +$ (Wang et al. 2022b), IRNet (Lian et al. 2022), ALIM-Scale and ALIM-Onehot (Xu et al. 2024).

On five of the nine subtasks, our method achieves the best performances, while on the remaining subtasks, ALIMOnehot or ALIM-Scale achieves the best performances (See Table 3). It is worth noting that our method is not designed for synthetic datasets, but still achieves good performance. It can be clearly seen that our method has more advantages when $q$ is small. This is because there are usually relatively few candidate labels associated with each sample on the dataset annotated by the pre-trained model.

# Ablations

We conduct experiments on four degenerations of our method to demonstrate the effectiveness of our proposed modules, which are: 1. w/o Co-PL: replaces the collaborative pseudo-labeling mechanism to performing pseudolabeling with their own prediction; 2. w/ SupCont: replace noisy supervised contrastive learning to traditional supervised contrastive learning; 3. w/o proto: does not perform prototypical similarity alignment; 4. w/o U: discarding unlabeled set $\mathcal { U }$ during training. It can be seen that, all modules contribute positively to the performance of our method.

# Discussion and Limitation

In this section, we briefly discuss incorporating weakly supervised learning into distillation from pre-trained models and compare this approach with other mainstream paradigms of applying pre-trained models to downstream tasks.

<html><body><table><tr><td>Paradigms</td><td>Samples</td><td>Human Annotations</td><td>Inference Size</td><td>Perf. Improvements</td></tr><tr><td>Zero-Shot</td><td>×</td><td>×</td><td>1</td><td></td></tr><tr><td>PromptLearning ／Adapter</td><td>few</td><td>few</td><td>increase sightly</td><td>√</td></tr><tr><td>KDunsup</td><td>√</td><td>×</td><td>small</td><td>×</td></tr><tr><td>KDsup</td><td>√</td><td>√</td><td>small</td><td>√</td></tr><tr><td>Fully Fine-Tuning</td><td>√</td><td>√</td><td></td><td>√</td></tr><tr><td>WSL</td><td>√</td><td>×</td><td>small</td><td>√</td></tr></table></body></html>

Table 5: Comparison among different pre-trained model application paradigms. Zero-Shot means directly performing zero-shot inference on untrained tasks. Prompt Learning fine-tunes a parameterized prompt, which is concatenated with text or image inputs, using few-shot downstream samples and Adapter adds a small number of trainable parameters to the neural network. $\mathrm { K D } _ { s u p }$ and ${ \mathrm { K D } } _ { u n s u p }$ represent supervised and unsupervised knowledge distillation, i.e. with or without task labels, respectively. ”-” means remaining the same with original model.

# Advantages of Incorporating Weakly Supervised Learning

Just like what this paper does, we can use pre-trained models as weak annotators to annotate unlabeled samples of downstream task with weak labels, and then formalize this task as a specific type of weakly supervised learning and design corresponding algorithms to address it.

Table 5 compares different pre-trained model application paradigms. It is evident that WSL is the only one that can achieve performance improvements over the original model without using additional manual annotations. Meanwhile, by retraining specialized small models on the downstream samples, the inference model size are significantly reduced. Additionally, due to the fact that few-shot fine-tuning techniques (e.g., prompt learning and adaptors) only add a small number of trainable parameters to the pre-trained models, their performance improvements are usually limited.

It is worth noting that the main difference from traditional unsupervised knowledge distillation is that this approach formalizes the downstream task as a specific weakly supervised learning problem and employs corresponding algorithms and can achieve significantly better performances on many scenarios compared to the pre-trained model. In contrast, knowledge distillation uses the output class distributions of the pre-trained model as training target, aiming to transfer the knowledge from a large model to a smaller specialized model and often does not achieve performance improvements over the pre-trained model.

# Limitation

Nevertheless, there are two main limitations associated with using weakly supervised learning. First, in downstream tasks where the training images are relatively similar to the general domain images used for pre-training the large model, such as ImageNet, specialized models trained through weakly supervised learning often fail to surpass the performance of the original pre-trained model. In these tasks, the large model is already highly effective, and zero-shot inference typically yields satisfactory results. Our method should primarily be applied to tasks where the image domain significantly differs from the general domain, and where the pre-trained model does not perform well.

![](images/bd22cb3b1f1ff16b64d1da59a92bfd7cfb96bd86271ddcd51622ad0d8e49529e.jpg)  
Figure 2: Accuracy on CIFAR-100 using different numbers of unlabeled samples.

Second, applying weakly supervised learning requires a large number of downstream unlabeled samples, which we assume are readily available. Here we conduct experiments on CIFAR-100 using different numbers of unlabeled samples and observe the performance of our method to explore its dependence on the number of unlabeled samples. As shown in Fig.2, our method requires 20,000 unlabeled samples (i.e., 200 per class) on CIFAR-100 to exceed the performance of CLIP zero-shot; when using the complete dataset, our method only drops $3 \%$ in performance compared to fully supervised training.

# Conclusion

Against the backdrop of significant advances in pre-trained large models, this paper makes a modest attempt to explore whether WSL can shift from primarily focusing on learning from manually annotated weak labels to using weak labels provided by pre-trained models and also examines whether this approach offers advantages compared to other pre-trained model application paradigms. In some image classification tasks, we use CLIP to annotate partial labels for images and designed NPLL algorithms for training, achieving effective results.