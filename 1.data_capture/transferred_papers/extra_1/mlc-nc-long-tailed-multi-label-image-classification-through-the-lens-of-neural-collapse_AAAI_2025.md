# MLC-NC: Long-Tailed Multi-Label Image Classification Through the Lens of Neural Collapse

Zijian Tao1,2, Shao-Yuan Li1,2,3\*, Wenhai Wan4, Jinpeng Zheng1,2, Jia-Yao Chen1,2, Yuchen $\mathbf { L i } ^ { 5 }$ , Sheng-Jun Huang1,2, Songcan Chen1,2

1MIIT Key Laboratory of Pattern Analysis and Machine Intelligence   
2College of Computer Science and Technology, Nanjing University of Aeronautics and Astronautics 3State Key Laboratory for Novel Software Technology, Nanjing University   
4School of Computer Science and Technology, Huazhong University of Science and Technology 5College of Computer and Software, Hohai University

# Abstract

Long-tailed (LT) data distribution is common in multilabel image classification (MLC) and can significantly impact the performance of classification models. One reason is the challenge of learning unbiased instance representations (i.e. features) for imbalanced datasets. Additionally, the co-occurrence of head/tail classes within the same instance, along with complex label dependencies, introduces further challenges. In this work, we delve into this problem through the lens of neural collapse (NC). NC refers to a phenomenon where the last-layer features and classifier of a deep neural network model exhibit a simplex Equiangular Tight Frame (ETF) structure during its terminal training phase. This structure creates an optimal linearly separable state. However, this phenomenon typically occurs in balanced datasets but rarely applies to the typical imbalanced problem. To induce NC properties under Long-tailed multilabel classification (LT-MLC) conditions, we propose an approach named MLC-NC, which aims to learn high-quality data representations and improve the model’s generalization ability. Specifically, MLC-NC accounts for the fact that different labels correspond to different feature parts located in images. MLC-NC extracts class-wise features from each instance through a cross-attention mechanism. To guide the features toward the ETF structure, we introduce visual-semantic feature alignment with a fixed ETF structured label embedding, which helps to learn evenly distributed class centers. To reduce within-class feature variation, we introduce collapse calibration within a lower-dimensional feature space. To mitigate classification bias, we concatenate features and feed them into a binarized fixed ETF classifier. As an orthogonal approach to existing methods, MLC-NC can be seamlessly integrated into various frameworks. Extensive experiments on widely-used benchmarks demonstrate the effectiveness of our method.

# Introduction

Multi-label image classification involves identifying and predicting a comprehensive set of labels that correspond to the various objects, attributes, or actions present within an image(Zhang and Zhou 2013; Everingham et al. 2015). This field has seen a surge in research on developing advanced algorithms such as probabilistic graphical models(Chen et al. 2019), new deep model architectures(Vaswani 2017), and novel loss functions(Ridnik et al. 2021; Kobayashi 2023). Despite these advancements, the success of these methods is often constrained by the prevailing yet unrealistic assumption: all categories appear with comparable numbers of instances. This textbook scenario glosses over the intricate long-tailed data distribution challenge inherent to practical multi-label classification scenarios(Reed 2001; Zhang et al. 2023).

![](images/468c2f049e2170d27452e739726b73ae51d04f89c27f093ab8de247670db06db.jpg)  
Figure 1: An illustrative example and the challenges of the LT-MLC problem on the VOC-LT dataset(Wu et al. 2020).

Figure 1 plots an illustrative example and the challenges of the LT-MLC problem on the VOC-LT dataset(Wu et al. 2020). It is characterized by: 1) Long-Tailed Class Distribution of Categories: From the class aspect, we can see a significant disparity in the prevalence of categories, with head classes (e.g., ’person’, ’chair’, ’car’) enjoying abundant instances, in the meanwhile a large number of tail classes (e.g., ’aeroplane’, ’sheep’, ’cow’) are underrepresented with rare instances. It’s widely acknowledged in the literature that training on such datasets can lead the model to be biased towards overfitting the head classes and underfitting the tail classes, significantly undermining the model’s generalization ability(He and Ma 2013; Veit et al.

2017). 2) Co-occurrence of Head and Tail Classes: From an instance perspective, each image can be annotated with a diverse array of labels, ranging from frequently assigned head classes to rarely assigned tail classes. For example, images (a) and (c) are tagged with head classes such as ’person’ and ’car,’ alongside tail classes like ’horse’ and ’sheep.’ This co-occurrence further complicates the learning of tail classes, as the dominance of head classes can suppress the extraction of features for tail classes. Notably, in VOC-LT, the co-occurrence rate between head and tail classes reaches $100 \%$ , while in COCO-LT, it is $9 9 . 2 6 \%$ . 3) Complex Correlations of Labels: Coupled with the reality that individual images may correspond to multiple labels, the labels may have complex dependencies, see $\{$ ’person’ ride on ’horse’ in the image (a), and ’cat’ sit on ’sofa’ in the image (b). This layer of complexity necessitates the model to account for both the interdependencies and the individualities of labels.

Due to the presence of multiple classes per instance, the common learning base of extracting one feature vector for each instance in the single-label field is certainly insufficient to capture the complex correlations among labels in multi-label datasets(Cao et al. 2019; Zhou et al. 2020). Additionally, strategies such as resampling and reweighting the instances, and those advanced representation learning and novel loss functions that rely on them are inevitably influenced by the co-occurrence of head and tail classes, leading models to focusing more on head classes and neglect learning the tail classes(Zhang et al. 2017; Hou et al. 2023; Wu et al. 2020; Guo and Wang 2021; Zhang et al. 2023). How to improve the discriminative ability of the learning model on LT-MLC without the interference of multiple label cooccurrence is of considerable significance.

Fortunately, recent studies on the neural collapse (NC) phenomenon have opened a chance for us. The NC phenomenon was first observed on balanced single-label multiclass datasets during the terminal phase of training deep classification models(Papyan, Han, and Donoho 2020): (i) the variability of features within every class collapses to zero, (ii) the set of feature means form a simplex equiangular tight frame (ETF), and (iii) the last layer classifiers collapse to the feature means and forms the same simplex ETF upon some scaling. These properties in turn facilitate an optimal linear separable state for classification. More characteristics include the global optimal property(Zhu et al. 2021) and generalization ability(Galanti, Gyo¨rgy, and Hutter 2021) were subsequently found for NC.

Whereas this phenomenon holds only for balanced datasets. On imbalanced datasets, it was identified that as the imbalance level increases, the learned representations and the classifier weights on minority classes will become indistinguishable(Fang et al. 2021). The absence of NC property is believed to explain the model performance degradation on imbalanced data partially. Subsequently, a few promising attempts to induce NC properties in the single-label imbalanced field were made. (Yang et al. 2022; Li et al. 2023b) proposed training a neural network with a fixed ETF classifier and proved that it naturally leads to optimal ETF structured features under certain assumptions. (Liu et al. 2023)

explicitly proposed two feature regularization terms to learn high-quality representation.

Building on the above neural collapse insights, we propose to introduce the NC properties to LT-MLC to improve model generalization ability. With the reasonable assumption that different labels correspond to distinct locations in each image, we extract class-wise features on each instance and leverage the semantic information of labels to guide feature learning. For each instance, we first use label embeddings with a fixed ETF structure to extract class-wise features for each category. By aligning these class-wise features with the corresponding label embeddings, we ensure that features from different classes on each instance are maximally distinguishable, ultimately forming an ETF structure. We further project these features into a lower-dimensional space and perform collapse calibration to reduce the withinclass feature variance. Finally, the resulting features are concatenated and fed forward into a fixed ETF classifier to obtain the model’s predictions. Extensive experiments provide strong evidence for the effectiveness of our method.

It’s worth noting that recent work(Li et al. 2023a) generalized the concept of NC to multi-label learning. It proved a generalized NC phenomenon, i.e., the means of features for instance with multiple labels are the scaled averages of means for their single-label counterparts. However, it requires balanced training instances within the same class multiplicity. In summary, our key contributions are:

• To the best of our knowledge, this is the first work to explore LT-MLC with a neural-collapse-inspired approach. It opens the chance to fundamentally solve the LT-MLC problem without the interference of the label co-occurrence and interdependency challenges. • We propose a novel MLC-NC approach to learn ETF structured class-wise features. It aligns the features with a fixed ETF label embedding to enforce evenly distributed class centers, and reduces within-class feature variation through collapse calibration in a lower dimension space. • MLC-NC outperforms strong baselines and achieves state-of-the-art results on the extensive benchmark datasets including COCO-LT, VOC-LT and VG200.

# Related Works

Long-tailed Classification Methods in literature mitigate long-tail by following aspects: resampling techniques(Chawla et al. 2002; Shen, Lin, and Huang 2016), re-weighted loss functions(Zhang et al. 2017; Cui et al. 2019) and specialized architectures(Liu et al. 2019; Zhou et al. 2020). MiSLAS(Zhong et al. 2021) explored methods to enhance model calibration by addressing the overconfidence issue typically seen in imbalanced data distributions. CSA(Shi et al. 2023) analyzed the effectiveness of re-sampling techniques in addressing class imbalance in long-tailed learning. SBCL(Hou et al. 2023) introduced a subclass-balancing approach to contrastive learning, significantly improving the representation of minority classes.

MLC ML-GCN(Chen et al. 2019) is a graph-based model that constructs relationships based on multi-label associations. Focal Loss(Lin et al. 2017) addresses the issue of class imbalance by assigning different weights to instances, thereby focusing the model on learning from difficult examples. ASL(Ridnik et al. 2021) dealt with the imbalance between positive and negative instances within labels by setting thresholds to reject incorrect label annotations. TW(Kobayashi 2023) proposed a novel loss function to cope with intra-class imbalance.

Long-tailed MLC DB(Wu et al. 2020) firstly addresses long-tailed MLC by introducing a co-occurence based loss function. It assigned balanced weights to instances to address the imbalance caused by label co-occurrence and incorporated a regularization term to mitigate the overemphasis on negative labels. URS(Guo and Wang 2021) proposed a dual-branch network architecture using uniform and rebalanced sampling as inputs for the branches and introduced a consistency loss to ensure effective learning. PLM(Duarte, Rawat, and Shah 2021) proposed randomly masking certain labels during loss computation to balance the classes. While mask training can alleviate class imbalance, it also results in the loss of useful learning information, preventing optimal performance. MFM(Zhang et al. 2023) decoupled the $\gamma$ in focal loss and proposes a Multi-Focal Modifier to increase the model’s attention to tail positive instances.

Neural Collapse (Papyan, Han, and Donoho 2020) first investigated the NC phenomenon that occurs during the terminal phase of training deep neural networks. Subsequent research(Tirer and Bruna 2022; Ji et al. 2021) focused on uncovering the underlying mechanisms of NC and identifying the conditions under which it occurs.(Yang et al. 2022) questioned the necessity of learning a linear classifier at the end of a deep neural network when its ETF structure is known. It proposed training a neural network with a fixed ETF classifier and demonstrated that it naturally leads to NC for feature representations.(Liu et al. 2023) proposed two explicit feature regularization terms to learn high-quality representation for class-imbalanced data.(Li et al. 2023b) applied NC to federated learning, and utilized a synthetic and fixed ETF classifier to address the data heterogeneity across different clients. These studies are for sing-label tasks. Recently, (Li et al. 2023a) generalized the concept of NC to the special balanced multi-label classification scenario, where the training instances within the same multiplicity are required to be balanced. It proved that a generalized NC phenomenon holds with the “pick-all-label” formulation where the means of features for instance with multiple labels are the scaled averages of means for their single-label counterparts.

As we discussed in the introduced section, mostly relying on some implicit instance-level resampling/reweighting strategies, exiting LT-MLC learning techniques were inevitably interfered with by the label co-occurrence and interdependency. In this paper, we fundamentally avoid this problem through the lens of NC by learning class-wise features with the optimal ETF structure.

# Preliminaries

We introduce a multi-label dataset with long-tailed distribution $D = \{ ( x _ { 1 } , y _ { 1 } ) , ( x _ { 2 } , y _ { 2 } ) , . . . . _ { . . } ( x _ { N } , y _ { N } ) \}$ , $N$ denotes the number of{inst1anc1es, $\boldsymbol { x } _ { i } \in \dot { \mathbb { R } } ^ { W \times H \times 3 }$ Nis thNe $i$ }-th input image, and $y _ { i }$ represents the label vector corresponding to the $i$ -th image. $\boldsymbol { y _ { i } ^ { \prime } } = [ y _ { i } ^ { 1 } , y _ { i } ^ { 2 } , \ldots , y _ { i } ^ { C } ] \in [ 0 , 1 ]$ represents the label vector of the $i$ -th instance. $C$ is the number of classes in the dataset. $y _ { i } ^ { c } = 1$ indicates that the $i$ -th instance includes class $c$ . Otherwiise, it does not include class $c$ . Let $\begin{array} { r } { n _ { c } = \sum _ { i = 1 } ^ { N } y _ { i } ^ { c } } \end{array}$ represent the number of training instances that inc ude class $C$ . In the long-tailed distribution, the number of instances for head classes is significantly larger than those of tail classes.

We first give the concept of a simplex ETF in the context of neural collapse.

Definition 1. Simplex Equiangular Tight Frame (ETF) ${ \bf V } = [ { \bf v } _ { 1 } , \cdot \cdot \cdot , { \bf v } _ { C } ] \in \mathbb { R } ^ { \hat { d } \times C }$ is composed of a set of vectors $\mathbf { v } _ { i } \in \mathbb { R } ^ { d }$ , for $i \in [ C ]$ and $d \geq C - 1$ . $\mathbf { V }$ is called $a$ simplex equiangular tight frame $i f$ :

$$
\mathbf { V } = { \sqrt { \frac { C } { C - 1 } } } \mathbf { U } \left( \mathbf { I } _ { C } - { \frac { 1 } { C } } \mathbf { J } _ { C } \right) .
$$

Here ${ \bf U } = [ { \bf u } _ { 1 } , { \bf u } _ { 2 } , \ldots , { \bf u } _ { C } ] \in \mathbb { R } ^ { d \times C }$ is an orthogonal matrix satisfying $\mathbf { U } ^ { T } \mathbf { U } = \mathbf { I } _ { C }$ , $\mathbf { I } _ { C } \in \mathbb { R } ^ { C \times C }$ is the identity matrix, and $\mathbf { j } _ { C } \in \mathbb { R } ^ { C \times C }$ is an all nes matrix. All vectors in a simplex ETF have equal $\ell _ { 2 }$ norm and identical pair-wise angles, with a cosine value of $- { \frac { 1 } { C - 1 } }$ . Hence,

$$
\mathbf { v } _ { i } ^ { T } \mathbf { v } _ { j } = \left\{ { \begin{array} { l l } { - { \frac { 1 } { C - 1 } } } & { { \mathrm { i f ~ } } i \neq j , } \\ { 1 } & { { \mathrm { i f ~ } } i = j . } \end{array} } \right.
$$

The pair-wise angle $- { \frac { 1 } { C - 1 } }$ is the maximal equiangular separation of $C$ vectors in $\mathbb { R } ^ { d }$ .

Then, we specify three fundamental characteristics of the neural collapse (NC) phenomenon below.

• Variability Collapse (NC1): The variability within the last-layer activations for instances within the same class collapses to zero, meaning the activations converge to their class means.   
• ETF Convergence (NC2): The class means collapse to the vertices of a simplex ETF, which is a highly symmetric geometric structure i.e. $\begin{array} { r } { \tilde { \bf f } _ { i } \cdot \tilde { \bf f } _ { j }  - \frac { 1 } { C - 1 } } \end{array}$ , $\forall i , j \in$ $\big [ C \big ]$ , $i \neq j$ , $\tilde { \mathbf { f } } _ { c }$ is the feature prototype of class $c$ .   
• Self-Duality (NC3): Up to rescaling, the last-layer classifiers also collapse to the class means, leading to a selfdual configuration where classifiers align with the class means which means that the classifier vectors collapse to the same simplex ETF i.e. $\tilde { \mathbf { v } } _ { i } \cdot \tilde { \mathbf { v } } _ { j }  - \frac { 1 } { C - 1 }$ , $\forall i , j \in$ $[ C ] , \ i \neq j$ where $\begin{array} { r } { \mathbf { v } _ { c } = \frac { \mathbf { v } _ { c } } { \| \mathbf { v } _ { c } \| } } \end{array}$ , $\forall c \in [ C ]$ , $\mathbf { v } _ { c }$ i−s the classifier vector of ETF classifier.

# Method

NC tells us the optimal structure (i.e. simplex ETF) of classifiers and feature prototypes in a perfect training setting. It inspires us to induce the NC properties in LT-MLC to improve the discriminative ability of the learning model. Concretely, we propose MLC-NC. As elaborated in Figure 2, MLC-NC consists of three major components: ETF Label Embedding Guided Feature Learning to learn distinct between-class features, feature projection and collapse calibration to reduce within-class feature variation, and binarized fixed ETF classifier to maximally separate the pairwise angles of all classes. In the following, we elaborate on the details.

![](images/1f7c44fd1e9ba256170db09a530c49a1b8b6d44c59a8309c42d958fe10dbab63.jpg)  
Figure 2: Overall structure of MLC-NC. MLC-NC consists of 3 major components: ETF Label Embedding Guided Feature Learning, Feature Projection and Collapse Calibration, and Binarized Fixed ETF Classifier.

# ETF Label Embedding Guided Feature Learning

In the long-tail multi-label domain, different labels correspond to features located in different parts of the instance. Therefore, it is crucial to consider spatial information during feature extraction, as different classes emphasize different locations. Hence, we need to extract corresponding features for each class, as detailed below.

Firstly, for an input instance $x _ { i }$ , we utilize a feature extractor backbone $\bar { G } ( \cdot )$ , e.g., ResNet50, to extract features $z _ { i } ~ = ~ G ( x _ { i } )$ , where $\bar { z } _ { i } ~ \in ~ \mathbb { R } ^ { w \times h \times c h }$ . $w , h$ , and ch represent the width, height, and number of channels of the features, respectively. To obtain features from different spatial positions, as shown in Figure 2, we divide $z _ { i }$ spatially into $\mathsf { \bar { \{ } }  z _ { i } ^ { 1 } , z _ { i } ^ { 2 } , \ldots . z _ { i } ^ { w \times h } \}$ , where $\boldsymbol { z } _ { i } ^ { k } \in \mathbb { R } ^ { 1 \times 1 \times c h }$ .

Secondly, accounting for the different spatial information of features for various labels, we employ label embedding and feature embedding to compute similarity scores for reweighting the features. We synthesize a simplex ETF label embedding ${ \mathbf l } = \{ { \mathbf l } _ { 1 } , { \mathbf l } _ { 2 } , \ldots , \dot { \mathbf l _ { C } } \} \in \mathbb { R } ^ { d \times C }$ by Eq. (1), where $d$ is the dimension of the label embedding and $d = c h$ . $C$ is the number of labels. The feature positional weight of label embedding ${ \mathbf { l } } _ { c }$ on feature embedding $\mathbf { z } _ { i } ^ { k }$ is computed as:

$$
\sigma ( \mathbf { z } _ { i } ^ { k } , \mathbf { l } _ { c } ) = \frac { \exp ( \sin ( \mathbf { z } _ { i } ^ { k } , \mathbf { l } _ { c } ) ) } { \sum _ { j = 1 } ^ { h \times w } \exp ( \sin ( \mathbf { z } _ { i } ^ { j } , \mathbf { l } _ { c } ) ) } ,
$$

with

$$
\sin ( \mathbf { z } _ { i } ^ { k } , \mathbf { l } _ { c } ) = \frac { \mathbf { z } _ { i } ^ { k } \cdot \mathbf { l } _ { c } } { \| \mathbf { z } _ { i } ^ { k } \| \| \mathbf { l } _ { c } \| } .
$$

$\sin ( \mathbf { z } _ { i } ^ { k } , \mathbf { l } _ { c } )$ is the cosine similarity function. The higher the value of $\sigma ( \cdot )$ , the more attention the label pays to the features at that position. Therefore, the feature corresponding to label $c$ in the $i$ -th instance $\mathbf { f } _ { i } ^ { c }$ is:

$$
\mathbf { f } _ { i } ^ { c } = \sum _ { k = 1 } ^ { h \times w } \sigma ( \mathbf { z } _ { i } ^ { k } , \mathbf { l } _ { c } ) \cdot \mathbf { z } _ { i } ^ { k } .
$$

Finally, to align the class-wise features with the label embeddings and ensure that the class-wise features also exhibit the ETF structure, we propose the visual-semantic feature alignment loss $\mathcal { L } _ { \mathrm { F L A } }$ :

$$
\begin{array} { r l } & { { \mathcal { L } } _ { \mathrm { F L A } } = \displaystyle - \frac { 1 } { N \cdot C } \sum _ { i = 1 } ^ { N } \sum _ { c = 1 } ^ { C } \left( y _ { i } ^ { c } \log \left( \frac { 1 + \sin ( \mathbf { f } _ { i } ^ { c } , \mathbf { l } _ { c } ) } { 2 } \right) + \right. } \\ & { \left. \left( 1 - y _ { i } ^ { c } \right) \log \left( 1 - \frac { C - 1 } { C } \left| \frac { 1 } { C - 1 } + \sin ( \mathbf { f } _ { i } ^ { c } , \mathbf { l } _ { c } ) \right| \right) \right) . } \end{array}
$$

When $y _ { i } ^ { c } = 1$ , through $\mathcal { L } _ { \mathrm { F L A } }$ , we align $\mathbf { f } _ { i } ^ { c }$ and ${ \mathbf { l } } _ { c }$ to maximize their cosine similarity. Conversely, when $y _ { i } ^ { c } = 0$ , we aim to minimize the similarity between the instance’s feature and the current label embedding, making the cosine value reflect the ETF structure as $- \frac { \mathbf { \bar { 1 } } } { C - 1 }$ . For example, we assume that instance $i$ contains label $c _ { 1 }$ , and instance $j$ contains label $c _ { 2 }$ and $c _ { 1 } \neq c _ { 2 }$ . Through $\mathcal { L } _ { \mathrm { F L A } }$ , $\sin ( \mathbf { f } _ { i } ^ { c _ { 1 } } , \mathbf { l } _ { c _ { 1 } } ) \approx$ 1 and sim(fj , lc2 ) ≈ 1. Therefore, f  ic1 $\begin{array} { r l r } { \frac { { \bf f } _ { i } ^ { c _ { 1 } } } { \| { \bf f } _ { i } ^ { c _ { 1 } } \| } \approx } & { { } \frac { { \bf l } _ { c _ { 1 } } } { \| { \bf l } _ { c _ { 1 } } \| } } & { } \end{array}$ and $\begin{array} { r } { \frac { \mathbf { f } _ { j } ^ { c _ { 2 } } } { \| \mathbf { f } _ { j } ^ { c _ { 2 } } \| } \approx \frac { \mathbf { l } _ { c _ { 2 } } } { \| \mathbf { l } _ { c _ { 2 } } \| } } \end{array}$ Given $\begin{array} { r } { \frac { \mathbf { l } _ { c _ { 1 } } } { \| \mathbf { l } _ { c _ { 1 } } \| } \cdot \frac { \mathbf { l } _ { c _ { 2 } } } { \| \mathbf { l } _ { c _ { 2 } } \| } = - \frac { 1 } { C - 1 } } \end{array}$ − C1 1 , it follows that $\frac { \mathbf { f } _ { i } ^ { c _ { 1 } } } { \| \mathbf { f } _ { i } ^ { c _ { 1 } } \| } \cdot \frac { \mathbf { f } _ { j } ^ { c _ { 2 } } } { \| \mathbf { f } _ { j } ^ { c _ { 2 } } \| } \approx - \frac { 1 } { C - 1 }$ 1 . In other words, in each instance, the class feature for $c _ { 1 }$ and $c _ { 2 }$ forms an ETF structure.

# Feature Projection and Collapse Calibration

According to NC1, in an ideal training scenario, within-class features should collapse to the corresponding class centers. However, in real-world scenarios, such as in the VOC and COCO datasets, the features of the same label in different instances may vary. For example, instances labeled ”bird” may contain seagulls in some cases and sparrows in others. Consequently, it becomes challenging to unify the extracted features $\mathbf { f } _ { i } ^ { c }$ in high-dimensional scenarios. Collapsing features based on NC1 in such cases may interfere with the feature extractor. Therefore, it is necessary to project the features into a lower dimension and normalize them:

$$
\hat { \mu } _ { i } ^ { c } = g ( \mathbf { p } _ { \mathbf { c } } ; \mathbf { f _ { i } ^ { c } } ) , \quad \mu _ { i } ^ { c } = \frac { \hat { \mu } _ { i } ^ { c } } { \lVert \hat { \mu } _ { i } ^ { c } \rVert _ { 2 } } .
$$

Here $g ( \cdot )$ is the linear projection function with $\mathbf { p _ { c } }$ denoting the parameters of the projection layer for the feature of class c. $\boldsymbol { \mu } _ { i } ^ { c } \in \mathbb { R } ^ { p \times C }$ is the normalized result of the lowdimensional feature vector $\hat { \mu } _ { i } ^ { c }$ obtained after projection. $p$ is the dimension of the feature vector after projection.

We note that the projection layer is essential in collapse calibration for the following reasons: (i) If the last layer of the feature extractor employs non-linear activation, e.g., ReLU, the raw feature $\mathbf { f } _ { c }$ will be sparse with zeros. This leads to features easily orthogonal to each other, making it difficult to collapse into the ETF structure. (ii) Highdimensional features of the same label may contain different information due to the difference between samples. By projecting them into a lower dimension, we refine the features and reduce their variability. This allows us to better leverage the principles of NC1 to optimize the features.

In the previous subsection, although we extract class-wise features for each class from an instance, these features all originate from the same instance. This results in a correlation between the class-wise features extracted from the same instance and this correlation limits our ability to collapse $\mu _ { i } ^ { c }$ to the class center $\mu ^ { c }$ . To address this, we adopt a contrastive learning approach, allowing each $\mu _ { i } ^ { c }$ to collapse to the class feature center while distinguishing it from the feature vectors of other class centers. In the $t$ -th epoch, we define the class center feature of class $c$ as $\mu _ { t } ^ { c }$ :

$$
\mu _ { t } ^ { c } = \frac { 1 } { n _ { c } } \sum _ { i = 1 } ^ { N } \mathbb { 1 } ( y _ { i } ^ { c } = 1 ) \mu _ { i , t } ^ { c } .
$$

$n _ { c }$ is the number of instances that contain label $c$ . $\mu _ { i , t } ^ { c }$ represents $\mu _ { i } ^ { c }$ in $t$ -th epoch. During training, we calculate the mean feature for each class in every epoch and use it as the class center for the next epoch. Inspired by contrastive learning, to encourage features to collapse into their respective class prototypes while remaining distant from the prototypes of other classes, we propose a loss function $\mathcal { L } _ { \mathrm { F P C } }$ based on feature projection and collapse calibration. At the $t$ -th epoch, $\mathcal { L } _ { \mathrm { F P C } }$ is defined as:

$$
\mathcal { L } _ { \mathrm { F P C } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { c = 1 } ^ { C } \Im \left( y _ { i } ^ { c } = 1 \right) \log \left( \frac { \exp ( S _ { i } ^ { c , c } / \tau _ { 1 } ) } { \sum _ { k = 1 } ^ { C } \exp ( S _ { i } ^ { c , k } / \tau _ { 2 } ) } \right) .
$$

Here Sic,k = $S _ { i } ^ { c , k } = \sin ( \mu _ { i , t } ^ { c } , \mu _ { t - 1 } ^ { k } )$ is the cosine similarity between the feature of class $c$ for instance $i$ in current t-th epoch and the class center feature of class $k$ computed in last epoch. $\tau _ { 1 }$ and $\tau _ { 2 }$ are temperature parameters used to control the degree of collapse, with a larger(smaller) value results in a higher degree of collapse. As shown in Figure 2, through $\mathcal { L } _ { \mathrm { F P C } }$ , we can learn more robust feature information for different classes and maximize the distinction of different classes.

# Binarized Fixed ETF Classifier

To minimize the impact of imbalance on the classifier, we employ the binarized fixed ETF classifier for multi-label datasets. We first synthesize a simplex ETF classifier $V _ { \mathrm { E T F } } =$ $\{ v _ { 1 } , v _ { 2 } , \ldots , v _ { C } \} \stackrel { } { \in } \mathbb { R } ^ { D \times C }$ by Eq. (1), where ${ \mathcal { D } } = p \times C$ . Extensive previous research has demonstrated the effectiveness of the ETF classifier when dealing with imbalanced datasets(Zhu et al. 2021; Galanti, Gyo¨rgy, and Hutter 2021). To capture the complete information of image features and avoid the classifier bias influenced by the bias in feature extraction across different classes, we concatenate the projected class-wise features $\mu _ { i } ^ { c }$ into $\mu _ { i }$ as the final feature. Then, we compute the inner product of the projected feature vector and the classifier vector to obtain the logit:

$$
h _ { i } ^ { c } = \gamma \mu _ { i } \cdot v _ { c } ,
$$

where $\mu _ { i } = c o n c a t ( \mu _ { i } ^ { 1 } , \mu _ { i } ^ { 2 } , \ldots , \mu _ { i } ^ { C } )$ , $h _ { i } ^ { c }$ is the logit of class c of i-th instance. We also introduce a learnable temperature scalar $\gamma$ to scale the results. Subsequently, the logit $h _ { i } ^ { c }$ for each class is fed into the sigmoid function to obtain the binary classification prediction.

Recently, (Kobayashi 2023) proposed a novel loss function to address the issue of imbalance between positive and negative labels in MLC. It approaches the problem from both an instance-wise way and a class-wise way, aiming to increase the margin between the logits of positive and negative labels. This enhancement improves the model’s attention to positive labels and suppression of negative labels. The specific loss function is as follows:

$$
\begin{array} { r } { \ell _ { i } = \mathrm { s o f t p l u s } \left[ \log \displaystyle \sum _ { c | y _ { i } ^ { c } = 0 } e ^ { h _ { i } ^ { c } } + T \log \displaystyle \sum _ { c | y _ { i } ^ { c } = 1 } e ^ { - \frac { h _ { i } ^ { c } } { T } } \right] , } \\ { \ell ^ { c } = \mathrm { s o f t p l u s } \left[ \log \displaystyle \sum _ { i | y _ { i } ^ { c } = 0 } e ^ { h _ { i } ^ { c } } + T \log \displaystyle \sum _ { j | y _ { j } ^ { c } = 1 } e ^ { - \frac { h _ { j } ^ { c } } { T } } \right] . } \end{array}
$$

The final two-way multi-label loss function $\mathcal { L } _ { T W }$ is defined as follows:

$$
\mathcal { L } _ { T W } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \ell _ { i } \left( \{ x _ { i } , y _ { i } ^ { c } \} _ { c = 1 } ^ { C } \right) + \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \ell ^ { c } \left( \{ x _ { i } , y _ { i } ^ { c } \} _ { i = 1 } ^ { M } \right) .
$$

Here $M$ is the number of instances in a batch and $T$ is the temperature parameter. In this paper, we use the two-way loss to handle the imbalance between positive and negative instances within labels.Thus, the final loss function $\mathcal { L }$ is:

$$
\begin{array} { r } { \mathcal { L } = \alpha \mathcal { L } _ { \mathrm { F L A } } + \beta \mathcal { L } _ { \mathrm { F P C } } + \mathcal { L } _ { \mathrm { T W } } . } \end{array}
$$

The pseudocode for MLC-NC is provided in Appendix A.

<html><body><table><tr><td rowspan="2">Category</td><td rowspan="2">Methods</td><td colspan="4">COCO-LT</td><td colspan="4">VOC-LT</td></tr><tr><td>Total</td><td>Head</td><td>Medium</td><td>Tail</td><td>Total</td><td>Head</td><td>Medium</td><td>Tail</td></tr><tr><td>MLC</td><td>ML-GCN Focal Loss ASL</td><td>44.24 49.46 54.35</td><td>44.04 49.80 50.59</td><td>48.36 54.77 58.76</td><td>38.96 42.14 51.82</td><td>68.92 73.88 78.31</td><td>70.14 69.41 71.12</td><td>76.41 81.43 84.95</td><td>62.39 71.56 78.71</td></tr><tr><td>LT-SLC</td><td>RS RW OLTR LDAM CB Focal BBN</td><td>46.97 42.27 45.83 40.53 49.06 50.00</td><td>47.58 48.62 47.45 48.77 47.91 49.79</td><td>50.55 45.80 50.63 48.38 53.01 53.99</td><td>41.70 32.02 38.05 22.92 44.85 44.91</td><td>75.38 74.70 71.02 70.33 75.24 73.37</td><td>70.95 67.58 70.31 68.73 70.30 71.31</td><td>82.94 82.81 79.80 80.38 83.53 81.76</td><td>73.05 73.96 64.96 69.09 72.74 68.62</td></tr><tr><td>LT-MLC</td><td>DB DB-Focal URS MFM MLC-NC</td><td>52.53 53.55 56.90 55.25 60.52</td><td>50.25 51.13 54.13 48.71 49.69</td><td>56.33 57.05 60.59 58.24 64.94</td><td>49.54 51.06 54.47 57.08 64.21</td><td>78.65 78.94 81.44 79.64 84.37</td><td>73.16 73.22 75.68 66.32 72.75</td><td>84.11 84.18 85.53 84.69 88.15</td><td>78.66 79.30 82.69 85.83 90.31</td></tr></table></body></html>

Table 1: Performance $( \mathrm { m A P \% } )$ ) comparison on COCO-LT, VOC-LT. The best and second-best performances are highlighted in bold and underline notes.

Table 2: Performance $( \mathrm { m A P \% } )$ ) comparison on imbalanced VG200.   

<html><body><table><tr><td>Methods</td><td>Total</td><td>G-mAP</td><td>Head</td><td>Medium</td><td>Tail</td></tr><tr><td>BCE DB DB-Focal ASL URS</td><td>24.74 30.83 30.51 26.75 10.23 29.43</td><td>38.39 44.69 44.46 42.99 17.8 44.51 50.07</td><td>39.73 47.07 47.65 42.54 24.28 44.12 49.78</td><td>22.70 29.07 28.58 24.73 8.73 27.54 35.13</td><td>19.26 22.80 22.36 20.38 3.17 23.52</td></tr></table></body></html>

# Experiment

Dataset: We analyze and conduct experiments on two artificially created long-tailed multi-label image classification datasets VOC-LT and COCO-LT following(Wu et al. 2020). Besides, we verify the universality of the proposed approach on one real-world multi-label dataset VG200 with milder imbalance distribution(Krishna et al. 2017).

Comparison Methods: To objectively evaluate MLC-NC, we compare it against methods from three scenarios. Classical Deep Multi-Label Methods: We employ ML-GCN, Focal Loss, and ASL. Classical Long-Tailed Single-Label Methods: We employ Empirical Risk Minimization(ERM) / Re-Sampling (RS) / Re-Weighting (RW)(Shen, Lin, and Huang 2016), OLTR(Liu et al. 2019), LDAM(Cao et al. 2019), CB Focal(Cui et al. 2019), BBN(Zhou et al. 2020). Long-Tailed Multi-Label Methods: We employ DB/DBFocal(Wu et al. 2020), URS(Guo and Wang 2021) and MFM(Zhang et al. 2023).

Training Setup: We use a ResNet50 pre-trained on ImageNet as the feature extractor. In $\mathcal { L } _ { \mathrm { F P C } }$ , $\tau _ { 1 }$ and $\tau _ { 2 }$ are set to $0 . 5 . \ \alpha$ and $\beta$ are set to 0.5 and 0.2, The dimension $d$ of the feature projection is set to 20. We evaluate mean average precision (mAP) across all classes, averaging the results over three runs for all methods.

# Results

Tables 1 and 2 present the experimental results on COCOLT, VOC-LT, and VG200. Our MLC-NC demonstrates significant performance improvements over the second-best baseline, especially in the medium and tail classes, and achieves the best overall total performance. For instance, on COCO-LT, our method achieves a medium mAP of 64.94 compared to 60.59 (URS), a tail mAP of 64.21 compared to 57.08 (MFM), and a total mAP of 60.52 compared to 56.90 (URS). Similarly, on VOC-LT, our method achieves a medium mAP of 88.15 compared to 85.53 (URS), a tail mAP of 90.31 compared to 85.85 (MFM), and a total mAP of 84.37 compared to 81.44 (URS).

URS enhances head class learning by enforcing consistency between resampled and non-resampled samples, leading to better performance in head classes, but highly inferior on medium and tail classes. LT-SLC methods emphasize tail class learning in single-label tasks but neglect label dependency and head-tail co-occurrence in long-tail multilabel scenarios, resulting in weaker performance.

Methods such as MFM and DB rely on co-occurrence matrices to shift the model’s focus towards tail classes, which inherently unable to improve the model’s learning of tail class features. In contrast, our approach learns class-wise optimal features with the guidance of ETF structured label embeddings, which fundamentally allows for a clearer representation of tail class features without the interference by the head-tail label co-occurrence.

On VG200, we additionally analyze the Global mAP (GmAP) from an instance-wise perspective. MLC-NC demonstrates superior performance under all scenarios, especially in tail classes, showing our robustness to milder imbalanced datasets.

Table 3: Ablation study on VOC-LT dataset.   

<html><body><table><tr><td>EGFL</td><td>FPC</td><td>BEC</td><td>Total</td><td>Head</td><td>Medium</td><td>Tail</td></tr><tr><td>1</td><td>√</td><td>√</td><td>82.13</td><td>72.47</td><td>87.99</td><td>84.98</td></tr><tr><td>√</td><td>1</td><td>√</td><td>83.43</td><td>71.12</td><td>87.88</td><td>89.34</td></tr><tr><td>√</td><td>√</td><td></td><td>79.59</td><td>66.40</td><td>85.64</td><td>84.96</td></tr><tr><td>√</td><td>√</td><td>√</td><td>84.37</td><td>72.75</td><td>88.15</td><td>90.31</td></tr></table></body></html>

Table 4: Performance $( \mathrm { m A P \% } )$ ) comparison of EGFL using different embedding methods on VOC-LT dataset.   

<html><body><table><tr><td>EGFL</td><td>Total</td><td>Head</td><td>Medium</td><td>Tail</td></tr><tr><td>Trainable embedding</td><td>83.92</td><td>71.99</td><td>87.57</td><td>90.14</td></tr><tr><td>Glove embedding</td><td>83.74</td><td>71.89</td><td>87.74</td><td>89.65</td></tr></table></body></html>

# Ablation Studies

We first conduct ablation studies on the three key components of our method: ETF label embedding Guided Feature Learning (EGFL) through not using label embedding to guide feature learning(w/o $\mathcal { L } _ { \mathrm { F L A } } )$ ; Feature Projection and Collapse Calibration (FPC) by not projecting and calibrating the feature(w/o $\mathcal { L } _ { \mathrm { F P C } } )$ ); Binarized Fixed ETF Classifier (BEC) by using randomly initialized trainable classifier and not concatenating features. As illustrated in Table 3, the performances on tail classes drop sharply without EGFL, which plays an important role in robust feature extraction. FPCC helps to refine class features. BEC significantly mitigates the impact of head and tail classes on the classification bias.

We then ablate the EGFL by using two other types of label embeddings: randomly initialized trainable label embedding, and fixed Glove label embedding(Pennington, Socher, and Manning 2014), as shown in table 4. The Glove embedding performs the worst for at least two reasons: firstly, it requires strict adherence to pre-trained Glove initialization, which does not cover all VOC labels, e.g., ”diningtable” and ”pottedplant”; besides, its fixed embedding dimension (300) constrains the feature dimensions to be the same. Compared with trainable embedding, our fixed ETF label embedding is optimal in both least training complexity and performance.

![](images/325f171da78cc9d40db0523fba2eca372730cb0dd4a69590074779aa7cc535ce.jpg)  
Figure 3: Pair-wise angle degree on VOC-LT of different classes feature center. The bigger the angle is, the easier for the model to differentiate classes. The optimal ETF pairwise angle for 20 classes is arccos $\left( { \frac { - 1 } { 1 9 } } \right) \approx 9 3 ^ { \circ }$ .

![](images/ab08e8589e1040b6a4341502a680a67c0b0de6e809e362104ddd261dd0949153.jpg)  
Figure 4: The effect of projection dimension on VOC-LT.

Effect of projection dimension Figure 4 illustrates the mAP performance across different projected feature dimensions on VOC-LT. The chart has two y-axes: the left y-axis represents the Total mAP, while the right y-axis shows the mAP for the Head, Medium, and Tail categories. The $\mathbf { \nabla } _ { \mathbf { X } } .$ -axis indicates the feature projection dimensions, ranging from 20 to 768, with 768 representing the original, unprojected features. It can be seen as the projection dimension increases, the overall mAP of the model tends to decline. The head classes which consist of instances with greater within-class diversity are more impacted, for which reducing features to lower dimensions is more helpful to enhance the model’s ability to aggregate features of same classes.

# Futher In-depth Analysis

Geometric structure of features In Figure 3, we plot the pair-wise angles of the centered feature means of different classes on VOC-LT guided by randomly initialized trainable label embedding and our fixed ETF label embedding. The larger the angle value, the more spread out the features, which makes it easier for the model to distinguish different class features. Our fixed ETF label embedding achieves much larger pair-wise angles, around $7 8 ^ { \circ }$ compared with the $4 3 ^ { \circ }$ of trainable label embedding.

# Conclusion

We address the LT-MLC problem by introducing neural collapse (NC). Our method, MLC-NC, uses fixed Equiangular Tight Frame (ETF) label embeddings and collapse calibration to optimize class-wise feature learning, enhancing discrimination across all classes while handling head-tail label co-occurrence and inter-dependency. Additionally, we employ Binarized Fixed ETF Classifier and concatenated features to mitigate classification bias. Extensive experiments have confirmed MLC-NC’s effectiveness.