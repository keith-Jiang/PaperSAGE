# Multi-label Self Knowledge Distillation

Xucong Wang 1, Pengkun Wang 12\*, Shurui Zhang 3, Miao Fang 3, Yang Wang 12\*

1University of Science and Technology of China, Hefei 230236, China   
2Suzhou Institute for Advanced Research, University of Science and Technology of China, Suzhou 215123, China 3Northeastern University at Qinhuangdao, Qinhuangdao 066000, China   
xuco $@$ mail.ustc.edu.cn, {pengkun, angyan}@ustc.edu.cn, $2 0 2 3 1 5 2 7 9 @$ stu.neuq.edu.cn, fangmiao $@$ neuq.edu.cn

# Abstract

Self-Knowledge Distillation (SKD) leverages the student’s own knowledge to create a virtual teacher for distillation when the pre-trained bulky teacher is not available. Whilst existing SKD approaches demonstrate gorgeous efficiency in single-label learning, to directly apply them to multi-label learning would suffer from dramatic degradation due to the following inherent imbalance: targets with unified labels but multifarious visual scales are crammed into one image, resulting in biased learning of major targets and disequilibrium of precision-recall. To address this issue, this paper proposes a novel SKD method for multi-label learning named Multi-label Self-knowledge Distillation (MSKD), incorporating three Spatial Decoupling mechanisms (i.e. Locality-SD (L-SD), Reconstruction-SD (R-SD), and Step-SD (S-SD)). LSD exploits relational dark knowledge from regional outputs to amplify the model’s perception of visual details. R-SD reconstructs global semantics by integrating regional outputs from local patches and leverages it to guide the model. SSD aligns outputs of the same input at different steps, aiming to find a synthetical optimizing direction and avoid the overconfidence. In addition, MSKD combines our tailored loss named MBD for balanced distillation. Exhaustive experiments demonstrate that MSKD not only outperforms previous approaches but also effectively mitigates biased learning and equips the model with more robustness.

Code — https://github.com/asaxuc/MSKD

# Introduction

Multi-Label Learning (MLL) has exemplified eye-catching applications in various downstream tasks, like action recognition (Zhang et al. 2020b), recommendation (Schultheis et al. 2022; Zhang et al. 2020a), and user profiling (Wang et al. 2021). Different from Single-Label Learning (SLL), the major challenge of MLL lies in learning severe noninjective mapping between visual targets and labels, which means the model is required to perceive all targets within an image equally, regardless of their cardinality, size, or location. While existing MLL models which propose to model visual-label correlations (Li et al. 2023; Wang et al. 2020;

![](images/04fc34791163953006d2cb03b261aa8ec7adcf2c0970d92211493b359bdf0dc8.jpg)  
Figure 1: Top-4 predictions of ResNet34 trained w w/o our MSKD, ranked in descending order. MSKD endows the model with significant distinguishing capability and robustness to space-intricately distributed samples.

Chen et al. 2019) or design class-specific decoders (Liu et al. 2021; Ridnik et al. 2023) have made remarkable success, most of them are costly and struggled to reach prevailing lightweight demand on both efficiency and accuracy. In this case, Knowledge Distillation (Hinton, Vinyals, and Dean 2015), the latest benchmark of model compression which asks a lightweight student to mimic its pre-trained bulky teacher to receive comparable performance, is widely applied to MLL and exemplifies prominent efficiency. (Yang et al. 2023a; Xu et al. 2022; Song et al. 2021).

However, the fact is that pre-trained teachers aren’t always accessible in latency-sensitive circumstances. In this case, SKD offers a competent solution by allowing students to directly learn from their self-teachers. Notably SKD is a non-trivial problem since no explicit peer model is provided. Existing SKD approaches dig out self-teacher from various layers (Yang et al. 2023b), samples (Yang et al. 2022), and training stages (Yang et al. 2019; Kim et al. 2021) as the selfteacher to form knowledge transferring, which has shown prominent efficiency in SLL. However, they perform unsatisfyingly in MLL due to the tough awareness of inherent imbalance, especially when faced with sophisticated backbones or datasets. Recently, a few SKD methods attempted to approach MLL but work poorly and are restrictive. For example, (Song et al. 2021) proposed the Uncertainty Distillation (UD) scheme to avoid over-training on difficult labels; MulSupCon (Zhang and $\mathrm { \sf W u } 2 0 2 4 ,$ proposed a supervised contrastive learning for MLL which selects positive sets based on their overlapping degree with the anchor. However, both of them primarily leverage the coarse regularization over the entire image semantics and show highly limited performance, where further consideration of the inherent imbalance problem of MLL is neglected.

Therefore, an urgent problem to be solved is: How to design an exactly efficient and generic SKD for MLL, which could alleviate the inherent imbalance? We answer the question by proposing our Multi-label Self-Knowledge Distillation (MSKD), incorporating three decoupling mechanisms attached with our tailored distillation for MLL. First, we extract randomly cropped and resized regions from original image. Intuitively, outputs of these regions contain more limited but specific semantics which is often overlooked while processing the whole image. Our first mechanism named L-SD then lies in utilizing them to amplify the corresponding features in the overall image. Moreover, we leverage these outputs to reversely identify the uniqueness of each patch and generate corresponding pseudo labels, then employ nonparametric graph propagation on them to capture spatial correlations for semantic integration. Accordingly we formulate our R-SD between integrated patches and vanilla output logits. Another mechanism S-SD requires the model to find a synthetical optimizing direction by aligning it’s outputs at near training steps, which would not only mitigate overfitting tendencies towards fitful difficult or miss-labeled targets but also enhance model’s stability.

In addition, we design a balanced distillation function tailored for MLL named MBD. Inspired by ASL (Ridnik et al. 2021), MBD employs reformulated softmax and dynamic KL-Divergence to mitigate the imbalance distillation between positive and negative logits.

Comprehensive experiments exemplify MSKD’s state-ofart performance. Figure 1 provides an intuitive visualization of MSKD’s effects, indicating that MSKD enhances the model’s ability to discriminate imbalanced samples and strengthens decision boundaries. Our contributions are then summarized as follows:

• New insight: To the best of our knowledge, our work is the first study to expand SKD methods to MLL and proposes a tailored benchmark SKD for it named MSKD. • New advisable distillation framework: MSKD combines three kinds of spatial decoupling mechanisms to address inherent imbalance problems of MLL, supported by our tailored distillation loss named MBD. • Compelling empirical results: we conduct exhaustive tests on multiple benchmark datasets. Results show our MSKD’s state-of-art performance as well as superior robustness under sophisticated circumstances.

# Related Works

# Multi-Label Learning

Multi-Label Learning (MLL) has attracted prevalent interest from research communities due to its widespread application in real-world scenarios. Existing MLL works can be roughly divided into the following categories: Loss Rebalancing (LB), Relation Modeling (RM), and Class Specific Decoding (CSC). LB-based approaches (Lin 2017) endeavor to mitigate the biased label supervision resulted from unmatching cardinality of positive and negative labels in the dataset. For example, ASL (Ridnik et al. 2021) punishes easy negative learning and emphasizes positive learning by introducing flexible exponent hyper-params and special threshold for negative probabilities. RM-based SKD utilizes multi-source prior information like the co-occurrence matrix, label embedding, or knowledge graphs (Lee et al. 2018) to assist the modeling of deep semantics across labels. Typically, ML-GCN (Chen et al. 2019) extracts label embedding and condition probability as the node features and edges separately and constructs a labeled graph over the image. Then multiple GCN layers are employed to model the label relation, and outputs are fused with visual features to generate integrated prediction. PatchCT (Li et al. 2023) wisely utilizes the optimal transportation theory to learn the visual-language interactions. CSC-based SKD aims at designing class-specific decoder architectures for better modeling the distinguished class semantics. For example, Query2Label (Liu et al. 2021) adapts DETR (Carion et al. 2020) from object detection and relies on distinguished learnable queries to perform class-specific prediction.

# Self-Knowledge Distillation

Firstly introduced by Geoffrey Hinton, Knowledge Distillation (KD) aims to transfer the abundant knowledge of a bulky teacher into a lightweight student which is more suitable for real-time applications. However, access to a pretrained teacher model is often impractical, thus it’s vital to dig out a self-teacher to guide the model in such circumstances, which is Self-Knowledge Distillation (SKD). As a general semi-supervised method, SKD mainly consists of the following categories: Architecture Modification (AM), Consistency Regularization (CM), and Label Smoothing (LS). AM-based SKD employs auxiliary modules to draw extra updating flows, such as BYOT (Zhang et al. 2019) which employs an auxiliary classifier for each block, and USKD (Yang et al. 2023b) which introduces an extra classifier for the middle layer to regularize the whole model. CM-based SKD focuses on improving the model’s robustness in multiple dimensions, for example, CS-KD (Yun et al. 2020) endeavors to close the logits between the same class. PS-KD (Kim et al. 2021) and DLB (Shen et al. 2022) try to keep the model’s robustness in different training stages. In a more general perspective, several contrastive learning approaches would be also treated as CM if the label supervision is applied. LS is regarded as a specific SKD category where smoothed labels could be viewed as the virtual teacher. For example, Zipf’s LS (Liang et al. 2022) employs Zipf’s distribution to guide the model.

While numerous SKD methods have been proposed, few have effectively addressed the specific challenges of MLL. Reluctantly, (Song et al. 2021) proposes an uncertaintybased self-distillation scheme (UD), but limited progress is achieved due to undistinguished classifiers, and its cumbersome training pipeline deviates efficiency principle of SKD. (Pan et al. 2022) introduces a self-distillation scheme between visual encoder and label encoder, however, it’s largely limited to specific backbones and prior knowledge. Others like MulSupCon (Zhang and ${ \mathrm { W u ~ } } 2 0 2 4 _ { , }$ ) devise a supervised contrastive learning mechanism that imposes dynamical weights based on the label overlap between two samples. However, the essential bottleneck of MLL concerning the aforementioned imbalance problem has not been considered. To the best of our knowledge, we first try to directly utilize SKD theory to handle the inherent imbalance in MLL and craft a tailored and pioneering SKD method for it.

# Methodology

Preliminary. Firstly we define some notations for a $C$ - class MLL task. Given batch of samples $\begin{array} { r l } { ( \boldsymbol { \mathcal { X } } , \boldsymbol { \mathcal { Y } } ) } & { { } = } \end{array}$ $[ ( \pmb { x } _ { 1 } , \pmb { y } _ { 1 } ) , ( \pmb { x } _ { 2 } , \pmb { y } _ { 2 } ) , \cdot \cdot \cdot , ( \pmb { x } _ { i } , \pmb { y } _ { i } ) , \cdot \cdot \cdot , ( \pmb { x } _ { B } , \pmb { y } _ { B } ) ]$ , where $B$ denotes the batch size and $\pmb { y } _ { i } \in \{ 0 , 1 \} ^ { C }$ denote the label, where each element indicates whether the corresponding target exists in image $\mathbf { \delta } _ { \mathbf { \boldsymbol { x } } _ { i } }$ or not. We denote the positive label set of $\mathbf { \boldsymbol { x } } _ { i }$ as $\mathcal { M } _ { i }$ $( { \mathcal { M } } _ { i } \subset y _ { i } )$ in extra. For any feature extractor $h ( ; \phi )$ and classifier $d ( ; \rho )$ , we denote ${ \bf f } _ { i } = h ( { \bf x } _ { i } ; \phi )$ , $\begin{array} { r } { \mathbf { q } _ { i } \ = \ d ( f _ { i } ; \pmb { \rho } ) ) } \end{array}$ . The optimizer endeavors to discover the optimal $\tilde { \phi }$ and $\tilde { \rho }$ that minimize the expected Binary CrossEntropy loss on $( \mathcal { X } , \mathcal { Y } )$ :

$$
\mathcal { L } _ { \mathrm { B C E } } = \sum _ { ( \boldsymbol { x } _ { i } , \boldsymbol { y } _ { i } ) } \boldsymbol { y } _ { i } \log ( \sigma ( \boldsymbol { q } _ { i } ) ) + ( 1 - \boldsymbol { y } _ { i } ) \log ( 1 - \sigma ( \boldsymbol { q } _ { i } ) ) ,
$$

where $\sigma$ is the sigmoid function. Next, we introduce our tailored SKD for dealing with the inherent imbalance problem.

# Multi-label Self Knowledge Distillation

The general framework of MSKD is illustrated in Figure 2. It incorporates three SD mechanisms, namely Locality-SD, Reconstruction-SD, and Step-SD, arranged in three flows of a single training step $t$ .

Locality-SD. We propose to randomly crop $S$ patches (each patch is denoted as $\mathcal { O } _ { i } ^ { s }$ ) from the original image $\pmb { x } _ { i }$ and resize them to magnify the details of the image and dilate the influence of large visual targets. Cropped patches are then fed into $h ( ; \phi ^ { t } )$ to generate regional feature maps $f _ { i } ^ { s }$ and output logits $\mathbf { \Delta } _ { o _ { i } ^ { s } } ^ { s }$ , i.e $\mathbf { \mathscr { f } } _ { i } ^ { s } = h ( \mathscr { O } _ { i } ^ { s } ; \boldsymbol { \phi } ^ { t } )$ , and $o _ { i } ^ { s } = d ( f _ { i } ^ { s } ; \pmb { \rho } ^ { t } )$ . We suggest treating these regional semantics as the selfteacher and leveraging them to amplify the corresponding regions of the original feature map. To achieve this, we firstly employ Region-of-Interest Pooling to obtain the feature map regions corresponding to $\mathcal { O } _ { i } ^ { s }$ in the original feature map $f _ { i } ^ { s }$ , then use classifier $\bar { d } ( ; \bar { \rho } ^ { t } )$ to generate outputs:

$$
r _ { i } ^ { s } = d ( \mathrm { R O I } ( { \pmb f } _ { i } ; \mathcal { O } _ { i } ^ { s } ) ; { \pmb \rho } ^ { t } ) .
$$

A naive idea to form regularization is directly aligning $\boldsymbol { r } _ { i } ^ { s }$ with $\mathbf { \sigma } _ { o _ { i } ^ { s } } ^ { s }$ . However, it would incur excessive learning of large targets and offset the locality regularization brought by patches. Instead, we conduct relation distillation between them to utilize the batch-wise dark knowledge and region-wise dark knowledge. Given $\pmb { r } \in R ^ { B \times S \times C }$ and $o \in$ RB×S×C, we firstly calculate inter-patch and inter-batch similarities (i.e., $\mathrm { s i m } _ { p } ( \cdot )$ and $\mathrm { s i m } _ { b } ( \cdot ) )$ for both of them:

$$
\begin{array} { r l } & { ( \operatorname* { s i m } _ { \mathrm { p } } ( \pmb { r } _ { i } ) ) _ { j k } = | | \pmb { r } _ { i } ^ { j } - \pmb { r } _ { i } ^ { k } | | _ { 2 } , } \\ & { ( \operatorname* { s i m } _ { \mathrm { b } } ( \pmb { r } ) ) _ { j k } = | | \mathrm { a v g } _ { s } ( \pmb { r } _ { j } ^ { s } ) - \mathrm { a v g } _ { s } ( \pmb { r } _ { k } ^ { s } ) | | _ { 2 } , } \end{array}
$$

where $\mathrm { s i m _ { p } } ( { \pmb r } _ { i } ) \in R ^ { S \times S }$ , $\mathrm { s i m } _ { \mathrm { b } } ( { \pmb r } ) \in R ^ { N \times N }$ . $( \cdot ) _ { j k }$ denotes the $( j , k )$ element; $| | \cdot | | _ { 2 }$ denotes normalized 2-D Euclidean distance. Notably, $\boldsymbol { r } _ { i } ^ { s }$ is averaged when calculating $\mathrm { s i m } _ { \mathrm { b } } ( \pmb { r } )$ since separate alignment of the same locations in different samples is not expected. The same operation in Eq. 3 is conducted on $\mathbf { \sigma } _ { o _ { i } }$ . To distill instance-wise and patch-wise dark knowledge from the self-teacher, we employ Huber-Loss like (Yang et al. 2023a) as follows:

$$
\begin{array} { l l l } { \displaystyle \mathcal { L } _ { \mathrm { L S D } } = \frac { 1 } { B } ( \sum _ { i } \mathrm { H u b e r L o s s } ( \operatorname* { s i m } _ { \mathrm { p } } ( \pmb { r } _ { i } ) , \operatorname* { s i m } _ { \mathrm { p } } ( \mathrm { s g } ( \pmb { o } _ { i } ) ) ) ) } \\ { \displaystyle ~ + \mathrm { H u b e r L o s s } ( \operatorname { s i m } _ { \mathrm { b } } ( \pmb { r } ) , \operatorname { s i m } _ { \mathrm { b } } ( \mathrm { s g } ( \pmb { o } ) ) ) , } \end{array}
$$

where $\operatorname { s g } ( \cdot )$ means stop-gradient.

Reconstruction-SD. For certain, resized random patches magnify visually subtle targets and provide different perspectives of viewing image semantics. With this regard, we seek to synthetically utilize them to endow the model with more sensibility to finer details. To be specific, we propose a graph-based reconstruction module named Graph Propagation (GP), to generate dynamic weights for different regional logits and reconstruct global semantics for regularization. For a random patch $\mathcal { O } _ { i } ^ { s }$ of image $\pmb { x } _ { i }$ , use $\mathbf { \sigma } _ { o _ { i } ^ { s } } ^ { s }$ to denote its output logits, GP firstly formulates pseudo label $\pmb { u } _ { i } ^ { s }$ with:

$$
( \pmb { u } _ { i } ^ { s } ) _ { j } = \left\{ \begin{array} { l l } { 1 , } & { i , j , s \in \mathrm { a r g m a x } ( \{ o _ { i j } ^ { s } - q _ { i j } \} , \beta ) } \\ { - 1 , } & { i , j , s \in \mathrm { a r g m i n } ( \{ o _ { i j } ^ { s } - q _ { i j } \} , \beta ) } \\ { 0 , } & { e l s e } \end{array} \right.
$$

where $o _ { i j } ^ { s }$ denotes the $j$ -th class in $\mathbf { \Delta } _ { o _ { i } ^ { s } } ^ { s }$ , the same meaning for $j$ in $q _ { i j }$ . Note that, the argmax operation is batch-wise and $\beta$ is set to $| \pmb { y } |$ in default. Eq. 5 highlights top- $\beta$ classes whose prediction probabilities show the largest difference with that of the overall image, i.e. ‘overachiever’ classes, as representatives of the patch. Intuitively, these pseudo labels indicate the distinctiveness and specificity of patches with respect to the original image in certain training stages: positive pseudo labels exemplify more equal ‘existing’ uniqueness of targets in the patch since the imbalance effect is dilated by the random cropping; Negative pseudo labels indicate the peculiarity of ‘non-existing’ by minus value, which would be transferred to the general ”existing” of corresponding classes in other patches after normalization. As a result, the pseudo labels are believed to indicate reliability of every class of each logit in representing a patch or being a self-teacher, and are more likely to concentrate on visually subtle targets.

Next, we consider forming graph propagation on these pseudo labels to further obtain relation-aware representations. Specifically, $\mathbf { \sigma } _ { o _ { i } ^ { s } } ^ { s }$ is selected to construct edges for its abundant spatial dark knowledge. We refer to Geom-GCN (Pei et al. 2020) and construct edge weights with normalized 2-D Euclidean distance:

![](images/e57d2fae24d01d4fbf0184a3a60d5d134aa681eff27659e2914639f909583149.jpg)  
Figure 2: Overview of MSKD. $\phi ^ { t }$ represents parameters of the feature encoder in step $t$ . $\rho ^ { t }$ represents the parameters of the classifier in step $t$ . The slash $( / )$ over arrows means stop-gradient. Label supervision is omitted.

$$
( A _ { i } ^ { s } ) _ { j k } = \frac { | | o _ { i } ^ { j } - o _ { i } ^ { k } | | _ { 2 } } { \operatorname* { m a x } \{ | | o _ { i } ^ { j } - o _ { i } ^ { k } | | _ { 2 } ; ( j , k ) \in B ^ { 2 } \} } .
$$

then a nonparametric propagation is employed to endow pseudo labels with deep spatial correlation:

$$
u _ { i } ^ { s } \gets ( D _ { i } ^ { s } ) ^ { - \frac { 1 } { 2 } } ( A _ { i } ^ { s } + I ) ( D _ { i } ^ { s } ) ^ { - \frac { 1 } { 2 } } { \boldsymbol u } _ { i } ^ { s } ,
$$

where $I$ is identity matrix, and $\begin{array} { r } { D _ { i } ^ { s } = \mathrm { d i a g } [ \sum _ { k } ( A _ { i } ^ { s } ) _ { : , k } + } \end{array}$ $( I ) _ { : , k } ]$ is the diagonal degree matrix of $A _ { i } ^ { s }$ . Initial feature $\pmb { u } _ { i } ^ { s }$ is updated $\tau$ times based on Eq. 7 to capture deep representations (denoted as $\overline { { e } } _ { i } ^ { s }$ ). Finally, $\overline { { e } } _ { i } ^ { s }$ is normalized with Softmax, and then weighs the average of regional logits $\mathbf { \sigma } _ { o _ { i } ^ { s } } ^ { s }$ . The averaged regional logits reconstruct the global semantics from a finer locality perspective which the model fails to capture, hence our R-SD leverages it to guide the model:

$$
\mathcal { L } _ { \mathrm { R S D } } ^ { ( i ) } = \mathcal { L } _ { \mathrm { M B D } } \big ( \sum _ { s } \big ( \mathrm { s o f t m a x } ( \overline { { e } } _ { i } ^ { s } ) \odot \mathrm { s g } ( o _ { i } ^ { s } ) \big ) , q _ { i } \big ) ,
$$

where $\odot$ is hadamard product, $\mathcal { L } _ { \mathrm { M B D } }$ is our proposed MBD loss for more balanced distillation and will be introduced in the next subsection. Notably, softmax applied on $\overline { { e } } _ { i } ^ { s }$ makes classes that frequently appear in different patches to obtain averaged weights, enabling the awareness of multifarious disjoint patterns and improving the model’s robustness.

Step-SD. Since the inherent imbalance problem of MLL gives a natural optimization challenge, we would like to find an optimized direction for each step and avoid models from being severely influenced by fitful difficult or miss-labeled samples. In this case, our S-SD takes parameters $\phi ^ { t - 1 }$ of the model $h$ in last step $t - 1$ , and feed current input $\mathbf { \delta } _ { \mathbf { \boldsymbol { x } } _ { i } }$ into both $h ( ; \phi ^ { t - 1 } )$ and $\bar { h } ( ; \phi ^ { t } )$ . Denote their output logits as $\pmb q ^ { t - 1 }$ and $\boldsymbol q ^ { t }$ respectively, S-SD hopes that the model would find an integrated optimization direction by shortening the discrepancy between $\pmb q _ { i } ^ { t - 1 }$ and $\pmb q _ { i } ^ { t }$ :

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S S D } } ^ { ( i ) } = \mathcal { L } _ { \mathrm { M B D } } ( \pmb { q } _ { i } ^ { t } , \mathrm { s g } ( \pmb { q } _ { i } ^ { t - 1 } ) ) . } \end{array}
$$

here $\mathcal { L } _ { \mathrm { M B D } }$ is also employed to avoid the overwhelming distillation on negative logits.

# Balanced Distillation Loss for MLL: MBD

Theoretical analysis. Let’s begin with the limitations of employing softmax-based KL-Div in MLL. Using $\tilde { \pmb { p } } _ { i }$ and $\tilde { \pmb q } _ { i }$ to denote the prediction distribution of teacher and student after softmax, $\tilde { q } _ { i t }$ to denote the random $t$ -th term of $\tilde { \pmb q } _ { i }$ , then:

$$
\mathcal { L } _ { \mathrm { s f x + K L } } ^ { ( i ) } = \sum _ { j } ^ { C } \tilde { p } _ { i j } \log ( \frac { \tilde { p } _ { i j } } { \tilde { q } _ { i j } } ) .
$$

by taking the gradient of L(sfi)x+ $\mathcal { L } _ { \mathrm { s f x + K L } } ^ { ( i ) }$ with respect to $\tilde { q } _ { i t }$ we get the following:

$$
\nabla _ { \tilde { q } _ { i t } } \mathcal { L } _ { \mathrm { s f x + K L } } ^ { ( i ) } = - \frac { \tilde { p } _ { i t } } { \tilde { q } _ { i t } } + ( \sum _ { j \neq t } \tilde { p } _ { i j } \frac { 1 } { \tilde { q } _ { i t } \cdot \sum _ { j \neq t } e ^ { q _ { i j } } } ) .
$$

after reformulation, we get:

$$
\nabla _ { \tilde { q } _ { i t } } \mathcal { L } _ { \mathrm { s f x + K L } } ^ { ( i ) } = ( - \tilde { p } _ { i t } + \frac { \sum _ { j \neq t } \tilde { p _ { i j } } } { \sum _ { j \neq t } e ^ { q _ { i j } } } ) \frac { 1 } { \tilde { q } _ { i t } } .
$$

without loss of generality, assume $\begin{array} { r } { \sum _ { j \neq t } e ^ { q _ { i j } } \gg \sum _ { j \neq t } \tilde { p } _ { i j } } \end{array}$ . Due to the existence of multi-labels, the discrepancy between $\tilde { p } _ { i t }$ and $\textstyle \sum _ { j \neq t } { \tilde { p } } _ { i j }$ largely shrinks, which means the optimization s ategy would push both positive and negative logits to the same direction. This will lead to two issues:

• Negative logits would be largely mis-leaded; • Positive and negative distillation is indistinguishable, resulting in overwhelming learning of the latter one and making the model conservative to positive prediction.

To deal with above issues, we propose MBD which simultaneously adopts reformulated softmax and reformulated KL-Div Loss for balanced distillation.

Reformulated Softmax (RS). To solve issue $\jmath$ , our Reformulated Softmax (RS) borrows the calibration branch from (Song et al. 2021) and applies softmax on multiple one-versus-negative combinations, to highlight the distinctive group knowledge of the positive logits $\cdot \tilde { q } _ { i j }$ is re-used):

$$
\tilde { q } _ { i j } = \mathrm { R S } ( q _ { i j } ) = \frac { 1 } { \lvert \mathcal { M } _ { i } \rvert } \sum _ { t \in \mathcal { M } _ { i } } \frac { e ^ { q _ { i j } } } { e ^ { q _ { i t } } + \sum _ { k \not \in \mathcal { M } _ { i } } e ^ { q _ { i k } } } ,
$$

where $\lvert \mathcal { M } _ { i } \rvert$ denotes number of elements in $\mathcal { M } _ { i }$ . RS generates probabilities for one-positive-all-negative combinations rather than directly take all positive logits into consideration, which substantially sharpen the discrepancy between positive and negative distillation and avoids the misleading of corrupted $\tilde { p } _ { i t }$ . From another perspective, RS avoids the mutual influence between optimization of different positive logits and unleashes the distinctiveness and equality of each one, since the maximation of posterior distribution no longer contradicts with the simultaneous propulsion of all positive logits, where more pareto optimality could be discovered.

Reformulated Distillation $\mathbf { \left( R e D \right) }$ . To deal with issue 2, we rectify the KL-Divergence with separate coefficients respectively for emphasizing positive distillation and punishing negative ones. Reusing $\tilde { \mathbf { \pmb { p } } } _ { i }$ , $\tilde { \pmb q } _ { i }$ in the former subsection, our Reformulated Distillation (ReD) would be written as:

$$
\mathcal { L } _ { \mathrm { R e D } } = \sum _ { j \in \mathcal { M } _ { i } } w _ { i j } ^ { + } \tilde { p _ { i j } } \log ( \frac { \tilde { p } _ { i j } } { \tilde { q } _ { i j } } ) + \sum _ { j \not \in \mathcal { M } _ { i } } w _ { i j } ^ { - } \tilde { p } _ { i j } \log ( \frac { \tilde { p } _ { i j } } { \tilde { q } _ { i j } } )
$$

$w _ { i j } ^ { + }$ and $w _ { i j } ^ { - }$ are trainable and formulated as:

$$
\begin{array} { l } { { w _ { i j } ^ { + } = \mid ( 1 - \omega ) + \tilde { p } _ { i j } \cdot \omega - \tilde { q } _ { i j } \mid ^ { \gamma ^ { + } } } } \\ { { w _ { i j } ^ { - } = \mid \tilde { p } _ { i j } \cdot \omega - \tilde { q } _ { i j } \mid ^ { \gamma ^ { - } } } } \end{array}
$$

with a little ambiguity, here $| \cdot |$ means to take the absolute value. $\omega$ controls the balance and is set to 0.5 in default. $\gamma ^ { + }$ $/ \gamma ^ { - }$ re-balance positive / negative learning and $\gamma ^ { + } \ll \gamma ^ { - }$ .

Clearly, $w _ { i j } ^ { + } / \bar { w } _ { i j } ^ { - }$ are proportional with distance between teacher outputs and student outputs $/$ ground-truth and student outputs. Intuitively, the rationality lies in:

• By measuring distance of outputs between teachers and students, $w _ { i j } ^ { + }$ and $w _ { i j } ^ { - }$ reduce the emphasis on easy logits (true-positives and true-negatives), and focus on conquering hard logits (false-positives and false-negatives). • By measuring distance between ground-truth and outputs of students, $\overline { { w _ { i j } ^ { + } } }$ and $w _ { i j } ^ { - }$ provide a synthetical evaluation of the fidelity of both teacher and student, which would largely avoid the misleading of teacher, especially in SKD which always employs on-the-fly guidance of an underoptimized teacher.

the MBD loss then can be formulated as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { M B D } } ^ { ( i ) } = \mathcal { L } _ { \mathrm { R e D } } ( \mathrm { R S } ( \pmb { p } _ { i } / \tau ) , \mathrm { R S } ( \pmb { q } _ { i } / \tau ) ) \cdot \tau ^ { 2 } } \end{array}
$$

where $\tau$ is the temperature. Following (Hinton, Vinyals, and Dean 2015; Chen et al. 2020), $\tau ^ { 2 }$ is multiplied to ensure that the relative contribution of distillation loss remains roughly unchanged when combined with other losses.

# Training Pipeline

Finally, the overall loss function is the weighted combination of our three decoupling losses and the BCE loss:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { B C E } } + \iota \cdot \mathcal { L } _ { \mathrm { L S D } } + \sum _ { i } \left( \kappa \cdot \mathcal { L } _ { \mathrm { R S D } } ^ { ( i ) } + \lambda \cdot \mathcal { L } _ { \mathrm { S S D } } ^ { ( i ) } \right)
$$

where $\iota , \kappa$ , and $\lambda$ are hyper-params to balance each loss’s contribution. We will discuss them in ablation studies.

# Experiments

# Experiment Settings

Datasets. Three benchmark datasets are employed in our experiments: Pascal-VOC 2007 (Everingham 2009) with 9963 images belonging to 20 categories, MS-COCO (Lin et al. 2014) with 200K images labeled with 80 classes, and MIRFLICKR (Huiskes and Lew 2008) with 25000 images from 24 classes. Notably we didn’t find an official train-test split for MIRFLICKR, thus we randomly split $80 \%$ as the training set and treated others as the testing set while maintaining balanced annotations between them.

Evaluation Metrics. We report five commonly used metrics for evaluation in MLL: (i) mean Averaged Precision (mAP), (ii) Precision of top-1 predictions $( \mathrm { P } @ 1 )$ , (iii) Recall of top-1 predictions $( \mathbb { R } \ @ 1 )$ , (iv) macro-F1 score (CF1), and (v) micro-F1 score (OF1). Employing both (ii) and (iii) would offer a more comprehensive view of performance on both positive and negative samples.

Implementations. We employ three backbones: ResNet34, MobileNet v2, and Swin-Transformer Tiny (Swin-T), which are the representations of CNN-based classical models, CNN-based lightweight models, and ViT-based models respectively. Especially for Swin-T, we employ two heads and adopt an attention dropout of 0.4. For all experiments we train for 80 epochs. Images are random-augmented (Cubuk et al. 2020) and resized to $2 2 4 \times 2 2 4$ . We employ SGD as the optimizer and the momentum and weight decay are set to 0.9 and $5 e ^ { - 4 }$ respectively, combined with a cosine-annealing scheduler for learning rate. For Pascal VOC 2007 and MIRFLICKR, a batch size of 64 is employed and the initial learning rate is set to 0.01. For COCO, we set the batch size as 128, and the initial learning rate as 0.1.

# Experiment Results

(I) Does MSKD outperforms existing state-of-arts? We show comparison results with SOTA based on three backbones and datasets in Table 1 and 2. We observe that:

• Most previous works perform minor improvements. In Table 1, PS-KD only receives $0 . 0 2 \%$ mAP improvement compared with Vanilla on ResNet34. mAP of DLB and DDGSD even decrease by $0 . 3 1 \%$ and $1 . 2 8 \%$ , accompanied with severe precision-recall imbalance, which also makes the F1 score deteriorated. We argue that the overemphasis on consistency in DLB and DDGSD may make the model conservative to positive prediction. UD and USKD outperform the baseline mAP by $0 . 7 9 \% / \ 2 . 9 4 \%$ respectively on ResNet34. However, they both received negative results on Swin-T, with mAP dropped by $1 . 9 1 \%$ $/ 0 . 1 3 \%$ and precision dropped by $5 . 2 9 \%$ , revealing their limited applications. Moreover, almost all methods fail on large datasets like MIRFLICKR and COCO as shown in Table 2, which may be ascribed to their naive and incompetent self-teacher.

• Our MSKD exemplifies superior performance and balanced Precision-Recall. On Pascal VOC 2007, MSKD outperforms the previous state-of-art in mAP by $2 . 8 4 \%$ on average, with precision boosts while the recall still slightly increases. In particular, a significant increment of all metrics with MobileNet v2 is observed, proving that MSKD dramatically reverses the corrupted optimization. Also, unlike stagnant peer works, MSKD performs remarkably on MIRFLICKR and COCO, with an average increment of $1 . 5 \%$ in mAP and $2 \%$ in F1. MSKD’s success may lie in its proactive solution to inherent imbalance which was highly neglected previously. By decoupling the intricate global semantics with cropping and merging, models applied with MSKD obtain more specific representations of every single target.

<html><body><table><tr><td rowspan="2">Methods</td><td colspan="4">ResNet34</td><td colspan="4">MobileNet_v2</td><td colspan="6">Swin-T</td></tr><tr><td>mAP</td><td>P@1 R@1</td><td>CF1</td><td>OF1</td><td>mAP</td><td>P@1</td><td>R@1</td><td>CF1</td><td>OF1</td><td>mAP</td><td>P@1</td><td>R@1</td><td>CF1</td><td>OF1</td></tr><tr><td>Vanilla</td><td>82.51</td><td>79.46</td><td>72.01</td><td>75.93</td><td>78.56</td><td>72.88</td><td>76.04 46.91</td><td>58.02</td><td>59.11</td><td>87.89</td><td>77.41</td><td>87.02</td><td>81.93</td><td>82.15</td></tr><tr><td>TF-KD</td><td>82.67</td><td>76.90</td><td>74.22</td><td>75.53</td><td>78.68</td><td>71.56</td><td>74.95 50.39</td><td></td><td>57.44 59.75</td><td>88.39</td><td>73.32</td><td>88.17</td><td>79.40</td><td>80.07</td></tr><tr><td>PS-KD</td><td>82.53</td><td>79.14</td><td>72.14</td><td>75.48</td><td>78.39</td><td>72.99</td><td>76.44 46.34</td><td></td><td>57.70 58.32</td><td>84.42</td><td>77.98</td><td>78.47</td><td>78.22</td><td>78.38</td></tr><tr><td>DLB</td><td>82.20</td><td>87.51</td><td>61.51</td><td>72.24</td><td>75.58</td><td>74.42</td><td>83.47 37.27</td><td>51.53</td><td>54.49</td><td>88.20</td><td>78.94</td><td>85.96</td><td>82.30</td><td>83.46</td></tr><tr><td>DDGSD</td><td>81.23</td><td>87.99</td><td>59.23</td><td>70.80</td><td>74.70</td><td>81.51</td><td>89.75</td><td>52.07</td><td>65.90 66.83</td><td>86.19</td><td>83.96</td><td>76.78</td><td>80.21</td><td>81.46</td></tr><tr><td>USKD</td><td>85.44</td><td>87.32</td><td>72.40</td><td>79.16</td><td>78.66</td><td>80.00</td><td>84.15</td><td>65.13 73.42</td><td>74.05</td><td>87.76</td><td>72.15</td><td>88.73</td><td>79.59</td><td>79.55</td></tr><tr><td>MulCon</td><td>83.46</td><td>79.26</td><td>72.64</td><td>76.34</td><td>78.21</td><td>72.86</td><td>76.18 46.99</td><td>58.12</td><td>60.30</td><td>82.43</td><td>78.31</td><td>72.93</td><td>75.53</td><td>75.58</td></tr><tr><td>UD</td><td>83.30</td><td>80.93</td><td>72.72</td><td>76.60</td><td>78.09</td><td>80.89</td><td>81.17</td><td>68.27</td><td>74.24 76.51</td><td>85.98</td><td>71.41</td><td>85.85</td><td>77.97</td><td>78.75</td></tr><tr><td>MSKD</td><td>86.83</td><td>85.42</td><td>74.98</td><td>79.37</td><td>80.29</td><td>82.62</td><td>85.38</td><td>69.43</td><td>76.58</td><td>78.77</td><td>89.16 80.50</td><td>88.77</td><td>84.43</td><td>85.79</td></tr></table></body></html>

Table 1: Comparison experiments on Pascal VOC 2007 based on ResNet34, MobileNet v2, and Swin-T.

<html><body><table><tr><td rowspan="2">Dataset</td><td rowspan="2">Methods</td><td colspan="5">ResNet34</td><td colspan="5">Swin-T</td></tr><tr><td>mAP</td><td>P@1</td><td>R@1</td><td>CF1</td><td>OF1</td><td>mAP</td><td>P@1</td><td>R@1</td><td>CF1</td><td>OF1</td></tr><tr><td rowspan="5">MIRFLICKR</td><td>Vanilla</td><td>76.33</td><td>77.98</td><td>60.68</td><td>68.88</td><td>78.85</td><td>80.69</td><td>76.99</td><td>73.76</td><td>75.34</td><td>83.16</td></tr><tr><td>DLB</td><td>75.46</td><td>76.39</td><td>58.97</td><td>66.56</td><td>72.07</td><td>80.46</td><td>81.69</td><td>65.01</td><td>72.40</td><td>80.78</td></tr><tr><td>USKD</td><td>74.68</td><td>78.71</td><td>59.44</td><td>67.63</td><td>73.53</td><td>79.64</td><td>78.52</td><td>64.71</td><td>70.95</td><td>80.43</td></tr><tr><td>UD</td><td>75.46</td><td>76.39</td><td>58.97</td><td>66.56</td><td>78.48</td><td>80.93</td><td>80.32</td><td>66.51</td><td>74.64</td><td>82.80</td></tr><tr><td>MSKD</td><td>78.54</td><td>82.90</td><td>60.75</td><td>71.09</td><td>80.80</td><td>81.77</td><td>81.56</td><td>70.20</td><td>75.46</td><td>83.48</td></tr><tr><td rowspan="5">COCO</td><td>Vanilla</td><td>66.26</td><td>73.64</td><td>49.01</td><td>59.78</td><td>65.99</td><td>71.94</td><td>76.09</td><td>56.70</td><td>64.98</td><td>68.24</td></tr><tr><td>DLB</td><td>65.25</td><td>78.21</td><td>44.26</td><td>56.53</td><td>64.52</td><td>70.77</td><td>75.19</td><td>52.93</td><td>62.54</td><td>68.59</td></tr><tr><td>USKD</td><td>61.06</td><td>76.68</td><td>42.01</td><td>54.28</td><td>66.48</td><td>70.54</td><td>76.84</td><td>54.95</td><td>64.70</td><td>58.62</td></tr><tr><td>UD</td><td>63.71</td><td>73.80</td><td>48.45</td><td>58.50</td><td>59.27</td><td>70.86</td><td>75.44</td><td>56.59</td><td>64.77</td><td>69.62</td></tr><tr><td>MSKD</td><td>67.02</td><td>74.96</td><td>54.84</td><td>63.84</td><td>67.19</td><td>73.62</td><td>76.17</td><td>61.71</td><td>68.18</td><td>71.61</td></tr></table></body></html>

Table 2: Extended comparisons on MIRFLICKR and COCO. For backbones, ResNet34 / Swin-T are selected.

Table 3: Ablation study of three mechanisms and MBD. !means to employ the loss. $\alpha$ means to employ the loss but MBD is replaced with softmax $+ \mathrm { K L }$ -Div.   

<html><body><table><tr><td>LLSD</td><td>LsSD LRSD</td><td>mAP P@1</td><td>R@1 CF1 OF1 72.01 75.93 78.56</td></tr><tr><td rowspan="5">√ √ √ √</td><td></td><td>82.51 79.46 83.06</td></tr><tr><td>83.45</td><td>73.00 76.80</td></tr><tr><td>83.64</td><td>87.79 67.00 73.36 74.09</td></tr><tr><td>√ 84.18</td><td>87.94 67.68 73.87 73.62</td></tr><tr><td>83.54 86.13</td><td>74.16 79.37 80.63</td></tr><tr><td rowspan="4">√ √ √ √ √√</td><td></td><td></td></tr><tr><td>85.79 87.53</td><td>69.69 77.62 78.32 77.58</td></tr><tr><td>84.43</td><td>85.70 68.51 76.95</td></tr><tr><td>86.83 85.42 74.98</td><td>79.37 80.29</td></tr></table></body></html>

Table 4: Ablation study of propagation time $\tau$ .   

<html><body><table><tr><td>T</td><td>mAP</td><td>P@1</td><td>R@1</td><td>CF1</td><td>OF1</td></tr><tr><td>1</td><td>86.12</td><td>85.53</td><td>71.73</td><td>78.56</td><td>79.95</td></tr><tr><td>2</td><td>86.83</td><td>85.42</td><td>74.98</td><td>79.37</td><td>80.29</td></tr><tr><td>3</td><td>83.66</td><td>85.85</td><td>65.49</td><td>74.33</td><td>75.70</td></tr></table></body></html>

(II) Does MSKD work well in downstream tasks? We further perform an image retrieval task to discover MSKD’s application ability for downstream tasks. Following (Yang et al. 2023a), we employ $k$ -nn algorithm to retrieve top-5 correlated images and depict our results in Figure 3. It’s obvious that our MSKD retrieves more accurate images than UD, indicating MSKD gives even more concentration to small targets and learns to distinguish classes well.

(III) How do components and hyper-parameters affect the results? We answer this question with following ablation studies:

- Components. Table 3 reports the ablation study results of three mechanisms and MBD on Pascal VOC 2007 with ResNet34 as the backbone. It can be observed that each mechanism performs well in mAP with an average improvement of $1 \%$ . Plus, we find that: $\mathcal { L } _ { \mathrm { { L S D } } }$ is better in precisionrecall balancing but not distinguished in mAP, while $\mathcal { L } _ { \mathrm { T S D } }$ and $\mathcal { L } _ { \mathrm { { R S D } } }$ are the opposite. This mutual compensation forms a generalized improvement across all metrics when these mechanisms are combined, as shown in the last row. In extra, to employ MBD provides further advancements in all metrics.

![](images/ac49fb403d4a2d0c680042c6d6ca9bfa48e077832455b21505d6f366c2858aa1.jpg)  
Figure 3: Performance of proposed MSKD on Image Retrieval. The first column is query images with labels. Each row of the following columns exemplifies the retrieved images and labels, sorted by relevance in descending order. Labels marked with green and red denote that they are included / not included in query labels respectively.

![](images/188ddcda2c1a3940fcbe2b55356228615ae6045ac0420cfac60b9b9630f97ccc.jpg)  
Figure 4: Ablation study of hyper-params $\iota$ , $\lambda$ and $\kappa$ .

- Coefficients $\iota , \lambda$ , and $\kappa$ . Results are shown in Figure 4 where optima is marked with red lines. It’s obvious that:

• Change of $\iota$ causes minor effects, possibly for the small magnitude of $\mathcal { L } _ { \mathrm { { L S D } } }$ . We set $\iota$ as 1.5 where local maximum is located.   
• When $\lambda$ grows larger than 2, all metrics are corrupted. We suggest that S-SD is poisoned by over-emphasizing the homogeneity. Slight improvements in all metrics are observed when $\lambda$ changes from 0 to 1, so we set $\lambda$ to 2.   
• All metrics deteriorate when $\kappa$ approaches 0. There exists about $0 . 5 \%$ decline in precision compared with the local maximum when $\kappa$ is 2, which would ascribe to overutilization of unreliable semantics in the initial stages. We set $\kappa$ to 2.0 as the trade-off.

- Propagation Time $\tau$ . We empirically set $\tau$ from 1 to 3 to investigate the ablation results. As demonstrated in Table 4, generally MSKD reaches its best performance when $\tau$ is 2, slightly outperforming that when $\tau$ is 1, which be attributed to better expressivity of two-layer propagation. However, all metrics suffer from dramatic decrease when $\tau$ is set to 3, where over-smoothing may get severe.

(IV) Whether MSKD truly alleviates the influence of the inherent imbalance? To validate this we collect the area of all visual targets in Pascal VOC 2007 utilizing their bounding boxes, then classify each target into 22 disjoint area ranges accordingly (since all images are of $2 2 4 \times 2 2 4 )$ ). Then we calculate correctly-predicted targets with MSKD,

![](images/3ba794803f49189d96f979cc32a3b458348d13a93a3626b580b7d25a7828672e.jpg)  
Figure 5: Number of total targets, correctly predicted targets w/ MSKD, UD or Vanilla on Pascal VOC 2007 shown by region scales, of which area is bounded by pow of adjacent $\mathbf { X }$ -axis values multiply 10. For example, $\mathbf { \boldsymbol { x } }$ -axis value 3 represents regions of which area is less than $( 1 0 \times 3 ) ^ { 2 }$ but more than $( 1 0 \times 2 ) ^ { 2 }$ . Fitting curve is plotted in extra to show the distributions along different areas.

UD, and Vanilla and demonstrate them by each area range, as shown in Figure 5. While UD has minor effects in promoting models to identify small targets and even poisons the large-scale recognition, MSKD receives up to $12 \%$ improvements in recognizing small targets and significantly shortens the distance to total numbers (the goal) while keeping undisturbed on large targets.

# Conclusion

We propose to extend effective SKD methods from SLL to MLL and design a tailored SKD named MSKD for the first time. Faced with the inherent imbalance between visual targets and labels in MLL, MSKD incorporates three spatial decoupling mechanism where detail semantics in each image is magnified in nuanced perspectives, supported by our tailored distillation loss to propel the unbiased selfsupervision. Experiments validate the superior performance as well as the general applicability of MSKD. In future we would like to further enhance MSKD with other selfteachers.

# Acknowledgments

This work was supported by the Natural Science Foundation of China Youth Project (No. 62402472), the Natural Science Foundation of Jiangsu Province of China Youth Project (No. BK20240461), the National Natural Science Foundation of China (No.62072427, No.12227901), the Project of Stable Support for Youth Team in Basic Research Field, CAS (No.YSBR- 005), and Academic Leaders Cultivation Program, USTC.