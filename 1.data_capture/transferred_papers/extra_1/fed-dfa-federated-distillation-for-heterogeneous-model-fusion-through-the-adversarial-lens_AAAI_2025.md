# Fed-DFA: Federated Distillation for Heterogeneous Model Fusion Through the Adversarial Lens

Zichen Wang, Feng Yan, Tianyi Wang, Cong Wang\*, Yuanchao Shu, Peng Cheng, Jiming Chen

College of Control Science and Engineering, Zhejiang University withnorman, yanfeng555, wty1998, cwang85, ycshu, lunarheart, cjm @zju.edu.cn

# Abstract

Most of the federated learning techniques are limited to homogeneous model fusion. With the rapid growth of smart applications on resource-constrained edge devices, it becomes a barrier to accommodate their heterogeneous computing power and memory in the real world. Federated Distillation is a promising alternative that enables aggregation from heterogeneous models. However, the effectiveness of knowledge transfer still remains elusive under the shadow of distinct representation power from heterogeneous models. In this paper, we approach from an adversarial perspective to characterize the decision boundaries during distillation. By leveraging $K .$ - step PGD attacks, we successfully model the dynamics of the closest boundary points and establish a quantitative connection between the predictive uncertainty and boundary margin. Based on these findings, we further propose a new loss function to make the distillation attend to samples close to the decision boundaries, thus learning from more informed logit distributions. The extensive experiments over CIFAR10/100 and Tiny-ImageNet demonstrate about $0 . 5 \substack { - 3 . 5 \% }$ improvement of accuracy under different IID and non-IID settings, with only a small increment of computational overhead.

# Introduction

Today’s Federated Learning (FL) framework mainly aggregates knowledge from the homogeneous models (McMahan et al. 2017). However, in practice, the misalignment of computation time and memory capacity across heterogeneous edge devices often leads to the problems of load imbalance and memory error (Wang, Yang, and Zhou 2021), which makes such one-model-fits-all design incompatible with the real-world demands. Model personalization aims to leverage heterogeneous models to balance the varying capacities and constraints on edge devices. A straightforward solution is to find commonalities in sub-model structures (Diao, Ding, and Tarokh 2020; Horvath et al. 2021; Alam et al. 2022; Wang et al. 2023), but these methods are still confined to the same model family and incompatible with the full-fledged heterogeneous model fusion such as transferring the knowledge between convolution and vision transformers.

Federated Distillation (FD) is a viable way to accommodate heterogeneous models (Li and Wang 2019; Lin et al.

2020; Zhu, Hong, and Zhou 2021; Cho et al. 2022; Liu et al. 2022). Inherited from knowledge distillation (KD) (Hinton, Vinyals, and Dean 2015), FD is model-agnostic — it replaces parameterized model fusion (McMahan et al. 2017) by matching client predictions with the consensus logits and minimizes the Kullback-Leibler (KL) divergence, thereby transferring the “dark knowledge” (additional information embedded in soft probabilities) among the participants. The seminal work of FedDF (Lin et al. 2020) has shown empirically that FD achieves a comparative performance of FedAvg (McMahan et al. 2017) while enabling knowledge transfer among the heterogeneous models.

However, KD typically suffers from the infamous capacity gap problem when the student finds it hard to mimic the predictive distribution of the teacher (Son et al. 2021; Mirzadeh et al. 2020; Zhu and Wang 2021a). Since FD inherits the algorithmic backbone from KD but distills in a distributed, online fashion, would the capacity gap still persist in FD, and in what forms? Previous works have yet to give sufficient insights under federated settings (Lin et al. 2020).

Motivated by these fundamental quests, we first present empirical findings of a complex, mutual influence between the light and heavyweight models — lightweight models progress at an inevitable cost of degrading the heavyweight models and over-parameterized networks (potential teachers) no longer guarantee high performance in FD. Such disparity becomes more prominent and difficult to rectify when the consensus is aggregated on unlabeled public data (Zhu, Hong, and Zhou 2021; Cho et al. 2022). If misclassified by the majority, the consensus would mislead model convergence towards a wrong direction, and degrade the overall performance (Du et al. 2020).

Prior works use logit variance to re-weight sample importance in distillation (Cho et al. 2022). Unfortunately, our implementation implies a weak correlation between logit variance and the true labels with a high false positive rate (wrong predictions could have higher variance as well). Blindly enlarging the weights of these samples would mislead the consensus. Are there other measures that can characterize the heterogeneous model capacity more reliably? In this paper, we orchestrate $K$ -step PGD attack as a proxy (Zhang et al. 2020) and build it into the FD pipeline to quantify instancespecific boundary margins that work for a mixture of convolution networks and ViTs. In particular, we propose a new framework called Fed-DFA (FedDF through the Adversarial Lens) to make distillation attend to samples in the vicinity of the consensus decision boundary. The main contributions are summarized below:

✧ We provide new empirical findings of the capacity gap and decision boundaries of heterogeneous model fusion in FD and unveil a latent correlation between boundary margin and predictive uncertainty. To the best of our knowledge, this is the first work that leverages adversarial learning to improve generalization for FD.   
✧ We propose Fed-DFA to make distillation attend to samples near the decision boundaries, which enables FD to learn from more informed distributions than the overconfident distributions with less information. We also analyze the generalization bound upon domain adaptations.   
$\nLeftrightarrow$ We demonstrate the efficacy of Fed-DFA over a CIFAR10/100 and Tiny-ImageNet, by enabling knowledge transfer between convolution and vision transformers. The results indicate a $1 . 5 { - } 3 . 5 \%$ improvement compared to the current SOTA of FedDF with even better generalization capabilities under non-IID data.

# Background and Related Works Knowledge Distillation

The classic knowledge distillation (KD) methods transfer the knowledge from one or an ensemble of pre-trained teachers to small-capacity students via minimizing a weighted combination of the cross-entropy and KL divergence (Hinton, Vinyals, and Dean 2015). A plethora of existing efforts focus on closing the teacher-student capacity gap (Son et al. 2021; Mirzadeh et al. 2020; Zhu and Wang 2021a) such as using teaching assistants as intermediate hubs (Son et al. 2021; Mirzadeh et al. 2020); utilizing gradient similarity to enable knowledge transfer (Zhu and Wang 2021a) and distilling from the model checkpoints (Wang et al. 2022). In (Zhu and Wang 2021b), the intrinsic dimension is used to quantify the capacity gap, and a two-step mutual distillation is proposed. Moreover, the previous works have shown the effectiveness of adopting adaptive and instance-specific temperatures (Li et al. 2022), multi-level logit distillation (Jin, Wang, and Lin 2023), and two-way mutual knowledge transfer (Zhang et al. 2018). Different from the prior efforts, we delve into the dynamics of decision boundaries in FD.

# Personalized Federated Learning

The existing research takes different directions to accommodate personalized model architectures, which are categorized into sub-model fusion (Diao, Ding, and Tarokh 2020; Horvath et al. 2021; Alam et al. 2022; Wang et al. 2023) and distillated model fusion (He, Annavaram, and Avestimehr 2020; Lin et al. 2020; Zhu, Hong, and Zhou 2021; Cho et al. 2022; Liu et al. 2022). Sub-model fusion finds a common subset of model structures: HeteroFL (Diao, Ding, and Tarokh 2020) and FjORD (Horvath et al. 2021) perform static extraction of sub-models from the large server model. FedRolex (Alam et al. 2022) extracts the sub-model on a rolling basis for diversified parameter aggregation. FlexiFed (Wang et al. 2023) utilizes the commonalities of architectures within the same network family, e.g., ResNet or VGG. However, these works assume the models to share a backbone structure that is still constrained by the same representational power. There is also a collection of exotic designs outside the FedAvg framework (He, Annavaram, and Avestimehr 2020; Lin et al. 2020; Zhu, Hong, and Zhou 2021; Cho et al. 2022; Liu et al. 2022). However, none of these works have reasoned from the challenges of adopting heterogeneous models with distinct representational power. This work approaches from an adversarial perspective to establish a connection between decision boundary dynamics and heterogeneous model fusion. The closest work to ours is (Nam et al. 2021), in which uniformly random perturbations are introduced to diversify the output logits and avoid overconfidence. Yet, the uniformly perturbed samples might obscure the original consensus without the hard labels. In this work, we leverage the adversarial examples in a nonintrusive manner to guide the distillation process.

# Preliminary

Consider a number of $N$ participants with heterogeneous models tailored to their device memory and computational capacity. Each client $n$ first performs gradient descent on his private dataset $\mathbb { D } _ { n } = \{ x _ { i } , y _ { i } \}$ , where $x _ { i }$ are the data samples and $y _ { i } = \{ 1 , \cdots , C \}$ are the label space for $C$ classes. The goal is to learn from $\mathbf { \tilde { \mathbb { D } } } = \{ \mathbb { D } _ { 1 } , \cdot \cdot \cdot , \mathbb { D } _ { n } \}$ with heterogeneous models ${ \bf w } _ { n }$ . The process consists of local training and global distillation:

$$
L o c a l T r a i n i n g : \quad \mathbf { w } _ { n } = \mathbf { w } _ { n } - \eta _ { 1 } \nabla _ { \mathbf { w } _ { n } } \mathcal { L } _ { C E } ( \mathbf { w } _ { n } , \boldsymbol { \xi } _ { n } ) ,
$$

where $\eta _ { 1 }$ is the local learning rate, $\mathcal { L } _ { C E }$ is the cross-entropy loss and $\xi _ { n }$ is the mini-batch of data $\mathbb { D } _ { n }$ . Once the local training is completed, each participant samples mini-batches of $\mathbf { x } ^ { p } = \{ x _ { i } ^ { p } \} \in \mathbb { D } _ { p }$ from an unlabeled public dataset $\mathbb { D } _ { p }$ to derive the averaged logits of consensus $\begin{array} { r } { \frac { 1 } { N } \sum _ { n = 1 } ^ { N } f _ { \mathbf { w } _ { n } } \bigl ( \mathbf { \bar { x } } ^ { p } \bigr ) } \end{array}$ Then each participant transfers the knowledge by aligning their local model outputs with the global consensus. Global Distillation:

$$
\mathbf { w } _ { n } = \mathbf { w } _ { n } - \eta _ { 2 } \nabla _ { \mathbf { w } _ { n } } \mathcal { L } _ { K L } \left( \sigma \big ( \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \frac { f _ { \mathbf { w } _ { n } } ( \mathbf { x } ^ { p } ) } { R } \big ) , \sigma \big ( \frac { f _ { \mathbf { w } _ { n } } ( \mathbf { x } ^ { p } ) } { R } \big ) \right) ,
$$

where $\mathcal { L } _ { K L }$ is the KL Divergence, $\sigma ( \cdot )$ is the softmax function, $\eta _ { 2 }$ is the learning rate of distillation and $R$ is the distillation temperature. After the knowledge distillation is completed, ${ \bf w } _ { n }$ is used as the starting point for the next iteration of local training.

# Adversarial-Guided Federated Distillation Understanding Heterogeneous Model Fusion

To gain a deeper understanding of FD, we are interested in answering: 1) Does FD suffer from a similar capacity gap in canonical KD? 2) Can we improve knowledge transfer in the absence of true labels via some latent attributes embedded inside the models? How would other distributional artifacts such as non-IID affect KD? We start with an empirical study on a vanilla setup of 5 participants, who select the models randomly from the x-axis in Fig. 1.

Observation 1 (Capacity Gaps). The capacity gap still persists as a complex, multi-faceted problem in FD: 1) The same model exhibits perceptible performance variance $\Delta 3 \sim 6 \%$ mAP) while collaborating with different models under various model combinations. This indicates a complex interplay between different models in FD. 2) Even under the same model combination, heterogeneous models have a high-performance variance ( $\Delta 1 0 \%$ mAP);

Unlike KD, in which students match their output logits to pre-trained teachers in an offline fashion, in FD, students with less representative power have a successive impact on their teachers online. As a result, teachers only perform slightly better than the students and over-parameterization cannot help improve the teachers. This makes pre-selection of “experts” (assigning higher weight values (Cho et al. 2022)) difficult when prior knowledge such as model parameters is no longer an accurate measure.

Such mutual influence is mainly attributed to the knowledge transfer when participants minimize the KL loss to match the consensus logits. Reasoned in (Ojha et al. 2024), these low-dimensional logit distributions “encode” the relative positions of samples from the decision boundaries. When the majority of models make wrong decisions, the consensus would mislead the KL loss towards a wrong target, thereby producing a misinformed decision boundary. This process cannot be simply rectified without the true labels of public data, thus leaving us in a paradoxical situation.

Prior efforts utilize logit variance as an implicit measure of decision confidence to guide weighted aggregations (Cho et al. 2022). However, due to the fast-growing exponentiation of softmax, small values are magnified and even the wrong decisions become overconfident, approaching a onehot vector. Our experiments find this phenomenon becoming more prominent under the non-IID data as most of the secondary class probabilities drop to near zero, leaving the output logits with little cues except the one-hot labels. Thus, instead of logit variance, are there other metrics to supplement the distillation process?

# Heterogeneous Model Boundaries

To answer this question, we resort to quantitative representations of the heterogeneous model boundaries and examine the possibility of using this latent information.

Definition 1 (Boundary Margin). Define the boundary margin estimate $\tilde { M } _ { \mathbf { w } _ { n } } ( x _ { i } ^ { P } )$ as the distance from a sample $x _ { i } ^ { P } \in \mathbb { D } _ { P }$ to the decision boundary of model ${ \bf w } _ { n }$ in the pixel-space $\chi$ . The true margin $M _ { \mathbf { w } _ { n } } \triangleq \tilde { M } _ { \mathbf { w } _ { n } }$ when $\mathbb { D } _ { P }$ and $\mathbb { D } _ { n }$ are identically distributed. Denote $f _ { \mathbf { w } _ { n } } ( x _ { i } ^ { P } )$ as the soft logit predictions and $\hat { y } _ { i } ^ { P } = \arg \operatorname* { m a x } _ { \hat { y } \in \mathcal { C } } f _ { \mathbf { w } _ { n } } ( x _ { i } ^ { P } )$ as the prediction result with maximum probability. We have,

$$
\tilde { M } _ { \mathbf { w } _ { n } } ( x ) = \underset { x _ { i } ^ { P } } { \operatorname* { m i n } } \lVert x _ { i } ^ { P } - x \rVert _ { p } ,
$$

$$
f _ { \mathbf { w } _ { n } } ( x _ { i } ^ { P } ) - \operatorname* { m a x } _ { \hat { y } _ { i } ^ { P } \neq y ^ { \prime } } f _ { \mathbf { w } _ { n } } ( x _ { i } ^ { P } ) = 0 ,
$$

where (4) represents when the boundary margin from soft logits $f _ { \mathbf { w } _ { n } } ( \bar { x } _ { i } ^ { P } )$ to a point until the output decision has

● Comb1●Comb2· Comb3• Comb4●Comb5●Comb6 30 ： 70 25 20 CIFAR100 15 . + . 1 V 8 InceptionV3 G B o

changed. Direct calculation of (3) is computation-intensive for $C$ -class classification. Hence, we relax the problem into finding the closest boundary point from a public data point $x _ { i } ^ { P }$ . This allows us to leverage $K$ -step PGD as a proxy to quantify boundary margin by observing when the top-1 probability has changed. This method efficiently approximates (3) based on the gradient information of intermediate model weights ${ \bf w } _ { n }$ in (1) before distillation.

$K$ -PGD Estimate. Denote $x _ { 0 } ^ { \prime }$ as the starting point $x _ { i } ^ { P }$ we draw $x _ { i } ^ { P }$ from a subset of public data to launch $K$ -PGD attacks in the participants model,

$$
\begin{array} { r } { x _ { k + 1 } ^ { \prime } = \Pi _ { \epsilon } ( x _ { k } ^ { \prime } + \gamma \mathrm { s i g n } ( \nabla _ { x _ { t } ^ { \prime } } \mathcal { L } _ { C E } ( f _ { \mathbf { w } _ { n } } ( x _ { i } ^ { P } ) , \hat { y } _ { i } ^ { P } ) ) , } \end{array}
$$

$$
\underset { c \in \mathcal { C } } { \arg \operatorname* { m a x } } f _ { \mathbf { w } ^ { n } } ( x _ { k } ^ { \prime } ) \neq \hat { y } _ { i } ^ { P } ,
$$

in which $\Pi _ { \epsilon }$ projects the sample into the $l _ { \infty }$ ball, $\epsilon$ is noise bound, $\gamma$ is the step size, $\hat { y } _ { i } ^ { P }$ is the argmax label of the prediction from $x _ { i } ^ { P }$ . Denote the above process as a function of PGD steps $f _ { \mathbf { w } _ { n } } ^ { \mathrm { P G D } } ( x _ { i } ^ { P } )$ for public data $x _ { i } ^ { P }$ . The boundary margin can be formally estimated by,

$$
\begin{array} { r } { \tilde { M } _ { \mathbf { w } _ { n } } ( x _ { i } ) \propto f _ { \mathbf { w } _ { n } } ^ { \mathrm { P G D } } ( x _ { i } ^ { P } ) . } \end{array}
$$

Then we leverage (7) to capture the boundary dynamics in the FD process.

Observation 2 (Boundary Dynamics). We find several intriguing properties empirically:

$\nLeftrightarrow$ Lightweight models (MobileNetV2) have smaller boundary margins and the decisions are under-confident compared to heavyweight models with much larger margins such as VGG shown in Figs. 2. Vision transformers are less confident compared to VGG, which is consistent with the recent findings from (Kim et al. 2024).

$\nLeftrightarrow$ Shown in Fig.3(a), as FD progresses, the lightweight models exhibit an upward trend and the opposite is observed for the heavyweight models (VGG) in the IID settings, which echoes with the previous findings of mutual

1250% NoIInD-IID 450% NoIInD-IID 10% NoIInD-IID 30%   
150% 120% 5% Till   
0% 25 50 75 100 125 150 175 200 0%0 75 100 125 150 175 200 0% 25 50 75 100 125 150 175 200 0 25 50 Adv Attack Step (K) Adv Attack Step (K) Adv Attack Step (K) (a) MobileNetV2 (b) VGG13 (c) ViT/4

![](images/5bc84ed492b18fd3a0d581aa5d29d9014389392574795d30991c789dffd26672.jpg)  
Figure 2: Sample-wise distribution of PGD steps $K$ between heterogeneous models (illustrated in Observation2): Lightweight CNN models such as MobileNet are underconfident and heavyweight CNN models are overconfident, whereas vision transformer (ViT/4) is in between.   
Figure 3: Tracing the closest boundary points in terms of $K$ - PGD during training dynamics: 1) IID settings; 2) non-IID settings.

influence between heterogeneous models. The decision boundary of vision transformers is relatively more stable.

✧ The boundary margins all have downward trends under non-IID data observed from Fig.3(b), indicating that distributional shifts drive samples closer to the decision boundaries. It becomes more difficult to find an optimal boundary to distinguish the non-IID data and knowledge transfer is less effective.

Thus, we confirm that heterogeneous models exhibit distinctive decision boundaries in the learning process.

# Connecting Boundary Margin with Predictive Uncertainty

With the new insights of heterogeneous boundary margins, we further look into their connections with predictive uncertainty, an effective measure of logit diversity that corresponds to model generalization (Dubey et al. 2018). We quantify the predictive uncertainty on an instance level by the Shannon Entropy,

$$
\mathbf { H } ( \mathbf { x } ) = - \sum _ { c = 1 } ^ { C } P ( y _ { c } | \mathbf { x } _ { i } ^ { P } ; \mathbf { w } _ { n } ) \log P ( y _ { c } | \mathbf { x } _ { i } ^ { P } ; \mathbf { w } _ { n } ) ,
$$

where $P ( y _ { c } | \mathbf { x } _ { i } ^ { P } ; \mathbf { w } _ { n } )$ is the probability of the $c$ -th category. We use the Spearman Correlation Coefficient (Wang, Yan, and Yan 2023) to establish the connection between the entropy and boundary margin. Spearman Correlation is more robust to model non-normal distributed data with a focus on the monotonic relationships. We first rank the entropy $\mathbf { H }$ and

PGD step $K$ in an ascending order, using the rank numbers $R _ { H i }$ and $R _ { K i }$ , and define $\begin{array} { r } { \bar { R } _ { H } ~ = ~ \frac { 1 } { N } \sum _ { i = 1 } ^ { N } R _ { H i } } \end{array}$ and $\begin{array} { r } { \bar { R } _ { K } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } R _ { K i } } \end{array}$ . The Spearman correlation coefficient $\rho$ is,

$$
\rho = \frac { \sum _ { i = 1 } ^ { N } ( R _ { H i } - \bar { R } _ { H } ) ( R _ { K i } - \bar { R } _ { K } ) } { \sqrt { \sum _ { i = 1 } ^ { N } ( R _ { H i } - \bar { R } _ { H } ) ^ { 2 } \sum _ { i = 1 } ^ { N } ( R _ { K i } - \bar { R } _ { K } ) ^ { 2 } } } .
$$

Observation 3 (Entropy vs. Boundary Margin). As shown in Fig.5(a), there is a strong correlation between the predictive entropy and boundary margin, i.e., samples that lie close to the decision boundaries (small $K$ ) tend to have higher predictive uncertainty and vice versa. Since (Cho et al. 2022) use logit variance as a metric to re-weight samples (samples with larger variance are assigned with larger weights in aggregation), we also establish the relation between the logit variance and true labels in Fig. 5(b). It is observed that although the variance increases with a closer $l _ { 2 }$ distance to the true labels, there is a large number of false positives with high logit variance, which leads to wrong decisions. Assigning these samples with larger weights could obscure the consensus by misleading the optimization in the wrong direction. This is also due to the paradox from unlabeled public data as the re-weighting approach still struggles without effective supervision.

# Proposed Method

Based on the discussions above, we see that although knowledge gaps seem inevitable, the heterogeneous models in FD could be better differentiated on the instance level given the boundary margins. We posit that distillation should attend to samples closer to the decision boundaries. This helps distillation match logits with higher entropy and transform knowledge from more informed predictive distributions rather than overconfident ones. Thus, for different models, we calculate the consensus of boundary margins on each mini-batch,

$$
\overline { { \mathbf { K } } } = \frac { 1 } { N B } \sum _ { n = 1 } ^ { N } \sum _ { i = 1 } ^ { B } f _ { \mathbf { w } _ { n } } ^ { \mathrm { P G D } } ( x _ { i } ^ { P } ) , K _ { \mathrm { t h } } = \mathrm { m e d } \{ \overline { { \mathbf { K } } } \}
$$

where $\overline { { \bf K } }$ is a vector of averaged PGD steps on a mini-batch $B$ of public data. Then we sort $\overline { { \mathbf { K } } }$ and set the median as $K _ { \mathrm { t h } }$ . This partitions the public data into $x ^ { + } ( \overline { { { \bf K } } } \leq K _ { t h } )$ and $x ^ { - } ( \overline { { \mathbf { K } } } > K _ { t h } )$ , in which we use the $+$ and − signs to denote

Local Training Characterizing Heterogeneous Model Boundaries Global Distillation C山 PHleatferoorgmesneous Computing NsOe PDuatbaliscet Public Dataset 图 . 2   
Private Dataset Misclassify Local Predictions (Softmax Probabilities) GPU VGG13 VGG13 0.03 0.09 0.55 0.11 0.04 0.04 0.03 0.06 0.02 0.03 MobileNet 0.01 0.01 0.02 0.01 0.89 0.01 0.01 0.01 0.01 0.01 H ShuffleNet 0.04 0.01 0.06 0.01 0.05 0.11 0.61 0.02 0.06 0.03   
Private Dataset Smartphone MobileNet 2DEesctiisimonVatGBeGo1uH3nedtaeryoogfenDecoisuMisonbBilBoeouNunentdVa2ry oMf aDregciinsSowhniutfBfhloeuPNnGedtDaryStoefp Average 0.03 0.0 0.21 0.04 0.04 0.05 0.51 0.03 0.03 0.02 K 3 4 1 1 9 3 1 1 1 1 6 5 2 2 1 1 7 9 5 Minimize KL-Divergence between   
Private Dataset 3 Generate Instance-specific PGD Step Kas Boundary Local Predictions and Average Consensus 1 Local Training Edge GPU ShuffleNet Estimates 𝒘𝒘 = 𝒘𝒘 − 𝜼𝜼𝟐𝟐𝜵𝜵𝒘𝒘𝑳𝑳𝑲𝑲𝑲𝑲(𝒑𝒑||𝒒𝒒)

whether the distillation should pay more or less attention to. Then, we replace (2) with the new loss function,

$$
\begin{array} { r l } & { \mathcal { L } _ { K L } ^ { \prime } = \mathbb { E } _ { x ^ { + } \sim \mathbb { D } _ { P } } \bigg [ \mathcal { L } _ { K L } \bigg ( \sigma \big ( \frac { 1 } { N } \underset { n = 1 } { \overset { N } { \sum } } \frac { f _ { \mathbf { w } _ { n } } ( x ^ { + } ) } { R } \big ) , \sigma \big ( \frac { f _ { \mathbf { w } _ { n } } ( x ^ { + } ) } { R } \big ) \bigg ) \bigg ] } \\ & { + \ : \beta \cdot \mathbb { E } _ { x ^ { - } \sim \mathbb { D } _ { P } } \bigg [ \mathcal { L } _ { K L } \bigg ( \sigma \big ( \frac { 1 } { N } \underset { n = 1 } { \overset { N } { \sum } } \frac { f _ { \mathbf { w } _ { n } } ( x ^ { - } ) } { R } \big ) , \sigma \big ( \frac { f _ { \mathbf { w } _ { n } } ( x ^ { - } ) } { R } \big ) \bigg ) \bigg ] . } \end{array}
$$

$\beta$ is a scaling factor with $0 \leq \beta \leq 1$ . Next, we derive the generalization bound according to (Ben-David et al. 2010).

Theorem 1 (Generalization Bound). For $N$ participants with the true data distribution $\mathcal { D } _ { n }$ of the $n$ -th local domain and the true global distribution as $\mathcal { D }$ , denote $\hat { \mathcal { D } } _ { n }$ and $\hat { \mathcal { D } }$ as the empirical distribution with samples of size $m$ each, drawn from $\mathcal { D } _ { n }$ and $\mathcal { D }$ , respectively. According to (10), $\textstyle { \mathcal { D } } _ { n }$ can be considered a mixture of distributions $\mathcal { D } _ { n } = \mathcal { D } _ { n } ^ { + } \cup \mathcal { D } _ { n } ^ { - }$ .

Consider hypothesis $h$ , $\mathcal { X }  \mathcal { Y }$ , from the input space $\chi$ to label space $y$ with hypotheses space $\mathcal { H } . \ h _ { n }$ is the hypothesis learned from $\mathcal { D } _ { n }$ , that $h _ { n } = \arg \operatorname* { m i n } _ { h } \mathcal { L } _ { \mathcal { D } _ { n } } ( h )$ and $\begin{array} { r } { \hat { h } _ { n } = \arg \operatorname* { m i n } _ { h } \mathcal { L } _ { \hat { \mathcal { D } } _ { n } } ( h ) . \ : d _ { \mathcal { H } \Delta \mathcal { H } } ( \hat { D } , D ) } \end{array}$ is defined as the divergence over $\mathcal { H }$ . For any $\tau \in ( 0 , 1 )$ , the following bound holds with probability at least $1 - \tau$ ,

$$
\begin{array} { r l } & { \displaystyle \mathcal { L } _ { \mathcal { D } } \bigg ( \sum _ { n = 1 } ^ { N } h _ { k } \bigg ) \leq \sum _ { n = 1 } ^ { N } \mathcal { L } _ { \hat { \mathcal { D } } _ { n } } ( h _ { k } ) + \frac { 1 } { 2 } \sum _ { n = 1 } ^ { N } d \varkappa \Delta \varkappa ( \hat { \mathcal { D } } _ { n } ^ { + } , \hat { \mathcal { D } } ) } \\ & { \displaystyle + \frac { 1 } { 2 } \sum _ { n = 1 } ^ { N } \beta d \varkappa \Delta \varkappa ( \hat { \mathcal { D } } _ { n } ^ { - } , \hat { \mathcal { D } } ) + \sum _ { n = 1 } ^ { N } \lambda _ { n } + 4 \sqrt { \frac { 2 d \log 2 m + \log \frac { 4 } { \tau } } { m } } } \end{array}
$$

where $\mathcal { L } _ { \hat { D } _ { n } } ( h _ { k } )$ is the empirical loss on $\hat { \mathscr { D } } _ { n }$ , $\begin{array} { r l } { \lambda _ { n } } & { { } = } \end{array}$ $\begin{array} { r } { \operatorname* { m i n } _ { h } ( \mathcal { L } _ { \mathcal { D } } ( h ) + \mathcal { L } _ { \mathcal { D } _ { n } } ( h ) ) } \end{array}$ is the combined error of the hypothesis.

Theorem 1 implies that a larger divergence from $d _ { \mathcal { H } \Delta \mathcal { H } } ( \hat { \mathcal { D } } _ { n } ^ { + } , \hat { \mathcal { D } } )$ and $d _ { \mathcal { H } \Delta \mathcal { H } } ( \hat { \mathcal { D } } _ { n } ^ { - } , \hat { \mathcal { D } } )$ degrades the overall generalization and more samples $m$ reduce the loss at an $\mathcal { \bar { O } } ( \log m / \sqrt { m } )$ rate. The impact of such distributional shifts is available in the supplement materials.

24680 -0.74 01.246802 ρ= -0.43 0 0.25 0.5 0.75 1 1.251.51.75 0.00 0.02 0.04 0.06 0.08 0.10 Shannon’s Entropy L2 Distance (a) Entropy vs. adv steps (b) Logit var. vs. dist. to label

Reduce Computational Overhead. To model the dynamics of decision boundary closely, it requires generating adversarial examples in each epoch during training. Hence, the computational overhead scales linearly with the number of participants and distillation data size. From Fig. 3, the boundary dynamics become more stabilized as the model converges, this allows to reduce computation via using a stale estimate of the boundary.

# Experiments

# Experimental Setup

Datasets and Heterogeneous Models. We conduct extensive experiments on the CIFAR-10/100 and Tiny-ImageNet Datasets. To cover a large collection of heterogeneous models, we form different combinations with a mixture of convolution and ViTs as shown in Table 1 and 2. We adopt the Dirichlet distribution to generate non-IID data as in (Lin et al. 2020), which uses $\alpha$ to evaluate different intensities of non-IIDness. A smaller $\alpha$ represents a higher degree of non-IIDness.

Baselines. We compare with the following baselines:

$\nLeftrightarrow$ FedMD (Li and Wang 2019): One of the earliest FD methods that only require limited black box access.

Table 1: Comparison of mAP $( \% )$ on CIFAR-10/100. Two top numbers are bolded with the best in Red and the second in Blue.   

<html><body><table><tr><td rowspan="2" colspan="2"></td><td colspan="6">CIFAR-10</td><td colspan="6">CIFAR-100</td></tr><tr><td>MobileNetV2</td><td>ShuffleNet</td><td>VGG13</td><td>ViT/4</td><td>CaiT/4</td><td>Average</td><td>MobileNetV2</td><td>ShuffleNet</td><td>VGG13</td><td>ViT/4</td><td>CaiT/4</td><td>Average</td></tr><tr><td rowspan="6">I</td><td>FedMD</td><td>61.71</td><td>64.57</td><td>69.92</td><td>54.07</td><td>47.35</td><td>59.52</td><td>20.02</td><td>26.35</td><td>24.23</td><td>18.62</td><td>19.23</td><td>21.69</td></tr><tr><td>FedDF</td><td>63.34</td><td>67.13</td><td>70.64</td><td>53.42</td><td>49.84</td><td>60.87</td><td>22.19</td><td>28.25</td><td>23.92</td><td>19.38</td><td>19.27</td><td>22.60</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>26.64</td><td></td><td></td><td>19.43</td><td></td></tr><tr><td>FedFDs</td><td>63.67</td><td>66.3</td><td>73.58</td><td>53.26</td><td>49.53</td><td>61.28</td><td>22.38</td><td></td><td>26.18</td><td>22.36</td><td></td><td>23.40</td></tr><tr><td>Selective-FD</td><td>64.54</td><td>66.52</td><td>70.98</td><td>54.70</td><td>50.58</td><td>61.46</td><td>21.88</td><td>26.83</td><td>25.72</td><td>22.66</td><td>20.20</td><td>23.46</td></tr><tr><td>Fed-DFA</td><td>64.01</td><td>66.25</td><td>73.65</td><td>55.81</td><td>51.42</td><td>62.23</td><td>24.88</td><td>28.36</td><td>28.84</td><td>25.86</td><td>22.36</td><td>26.06</td></tr><tr><td rowspan="6">III-UON 1 = x)</td><td>FedMD</td><td>57.00</td><td>60.53</td><td>65.33</td><td>49.82</td><td>49.51</td><td>56.44</td><td>21.59</td><td>25.74</td><td>20.86</td><td>20.22</td><td>16.59</td><td>21.00</td></tr><tr><td>FedDF</td><td>53.33</td><td>62.45</td><td>72.87</td><td>50.99</td><td>50.09</td><td>57.95</td><td>20.70</td><td>22.80</td><td>27.56</td><td>20.91</td><td>18.31</td><td>22.06</td></tr><tr><td>FedODS</td><td>55.71</td><td>61.03</td><td>73.62</td><td>47.43</td><td>51.45</td><td>57.85</td><td>21.02</td><td>24.36</td><td>25.72</td><td>22.32</td><td>18.94</td><td>22.47</td></tr><tr><td>RHFL</td><td>54.12</td><td>63.63</td><td>72.53</td><td>49.92</td><td>50.60</td><td>58.16</td><td>20.93</td><td>24.10</td><td>29.07</td><td>20.26</td><td>18.77</td><td>22.63</td></tr><tr><td>Selective-FD</td><td>55.47</td><td>64.98</td><td>71.21</td><td>52.49</td><td>49.92</td><td>58.81</td><td>19.99</td><td>26.53</td><td>25.85</td><td>22.89</td><td>16.93</td><td>22.44</td></tr><tr><td>Fed-DFA</td><td>57.53</td><td>64.33</td><td>71.97</td><td>51.79</td><td>51.53</td><td>59.43</td><td>22.39</td><td>26.56</td><td>27.46</td><td>24.09</td><td>19.83</td><td>24.07</td></tr><tr><td rowspan="7">III-UON (I'O I</td><td>FedMD</td><td>33.61</td><td>42.32</td><td>44.89</td><td>35.05</td><td>34.77</td><td>38.13</td><td>14.00</td><td>17.74</td><td>15.79</td><td>15.24</td><td>13.72</td><td>15.30</td></tr><tr><td>FedDs</td><td>34.32</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>44.98</td><td>44.15</td><td> 35.49</td><td>34.45</td><td>38.67</td><td>15.28</td><td>17.62</td><td>19.84</td><td>14.88</td><td>14.60</td><td>1644</td></tr><tr><td>RHFL</td><td>32.06</td><td>53.50</td><td>38.49</td><td>37.58</td><td>34.58</td><td>39.24</td><td>14.72</td><td>19.62</td><td>21.85</td><td>15.67</td><td>13.75</td><td>17.12</td></tr><tr><td>Selective-FD</td><td>35.78</td><td>48.97</td><td>44.11</td><td>35.25</td><td>31.16</td><td>39.05</td><td>18.03</td><td>17.22</td><td>20.88</td><td>15.72</td><td>14.82</td><td>17.33</td></tr><tr><td>Fed-DFA</td><td>32.98</td><td>54.26</td><td>42.20</td><td>38.47</td><td>35.61</td><td>40.70</td><td>17.94</td><td>20.73</td><td>21.72</td><td>16.02</td><td>17.66</td><td>18.81</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table></body></html>

25 20 20 15 15 10 FedMD RHFL FedMD RHFL 10 FedODS Selective-FD FedODS Selective-FD 5 5 FedDF Fed-DF A FedDF Fed-DF A 0 20 40 60 80 100 0 20 40 60 80 100 Training Epochs Training Epochs (a) ShuffleNet (b) CaiT/4

✧ FedDF (Lin et al. 2020): A comprehensive and state-ofthe-art FD framework.

✧ FedODS (Nam et al. 2021): We utilize the method to maximize diversities of output logits in FD by generating public data perturbed with Output Diversification Sampling (ODS) (Tashiro, Song, and Ermon 2020).

$\nLeftrightarrow$ RHFL (Fang and Ye 2022): A framework that improves the robustness of heterogeneous FD against noisy labels.

$\nLeftrightarrow$ Selective-FD (Shao, Wu, and Zhang 2023): SelectiveFD utilizes a selective knowledge-sharing mechanism to identify knowledge from local and ensemble predictions.

Implementation Details. We use the Adam optimizer for both the local training and global distillation and set the learning rate to $1 0 ^ { - 3 }$ . Our base testing includes 1 local epoch and 10 distillation epochs, and the temperature $R = 1$ unless stated otherwise. We adopt $l _ { \infty }$ PGD attacks with a step size $\gamma = 0 . 0 1$ , $\epsilon = 0 . 1$ , and $K = 5$ to explore the decision boundary. We set $\beta = 0 . 1$ in (11) to put more attention on the samples closer to the boundaries.

# Performance Comparison of mAP

CIFAR-10/100. Table 1 compares the proposed Fed-DFA on three convolutional models (MobileNetV2, ShuffleNet and VGG13) and two vision transformers (ViT/4 and CaiT/4). Our goal is to not only focus on individual performance but also assess the average performance and the knowledge transfer between models under the IID/non-IID settings. It is observed that Fed-DFA outperforms all the benchmarks in terms of the average mAP across the 5 models. In particular, it outperforms the current SOTA of FedDF by $1 . 4 - 2 . 0 \%$ on CIFAR-10 and $2 . 0 - 3 . 5 \%$ on CIFAR-100. In comparison, FedMD slightly suffers from the catastrophic forgetting to switch between local training and distillation, thus cannot effectively transfer knowledge between different models. Since boundary attack is adopted, the computation time of FedODS is significantly higher than other methods while its performance is identical to Fed-DFA only on ViT and CaiT. The noise learning/confidence re-weighting mechanism enables RHFL to achieve performance second to Fed-DFA. Selective-FD can identify accurate and precise knowledge during the FD process thus improving FedDF. There are also some interesting details about Fed-DFA in the IID/non-IID data with different datasets. First, Fed-DFA characterizes the decision boundaries learned from non-IID data well as it outperforms all the competitors under different degrees of non-IIDness $( \alpha = 0 . 1 , 1 \$ ). Further, the performance gain becomes even larger on CIFAR-100. This is because predictions are less confident with more classes in CIFAR-100, which gives a larger pool of samples with predictive uncertainty that could further boost the performance of Fed-DFA. Fig. 6 provides the convergence of two representative models. We can see that Fed-DFA begins to functionalize after 20 epochs when the testing accuracy of other methods has plateaued.

Table 2: Comparison of mAP on Tiny-ImageNet. Top numbers are bolded with the best in Red and the second in Blue.   

<html><body><table><tr><td colspan="2"></td><td>VGG16</td><td>ResNet50</td><td>ViT/16</td><td>Average</td><td>Std (↓)</td></tr><tr><td rowspan="6">四</td><td>FedMD</td><td>20.10</td><td>28.24</td><td>25.69</td><td>24.68</td><td>4.16</td></tr><tr><td>FedDF</td><td>20.54</td><td>29.79</td><td>26.26</td><td>25.53</td><td>4.67</td></tr><tr><td>FedODS</td><td>20.30</td><td>29.25</td><td>26.50</td><td>25.35</td><td>4.58</td></tr><tr><td>RHFL</td><td>20.89</td><td>29.13</td><td>27.20</td><td>25.74</td><td>4.31</td></tr><tr><td>Selective-FD</td><td>20.93</td><td>29.48</td><td>25.78</td><td>25.40</td><td>4.29</td></tr><tr><td>Fed-DFA</td><td>21.61</td><td>29.82</td><td>26.81</td><td>26.08</td><td>4.15</td></tr><tr><td rowspan="6">(IO QII-ON =I @</td><td>FedMD</td><td>10.61</td><td>17.75</td><td>15.65</td><td>14.67</td><td>3.67</td></tr><tr><td>FedDF</td><td>11.26</td><td>17.63</td><td>16.60</td><td>15.16</td><td>3.42</td></tr><tr><td>FedODS</td><td>10.99</td><td>17.78</td><td>16.53</td><td>15.10</td><td>3.61</td></tr><tr><td>RHFL</td><td>11.23</td><td>18.22</td><td>16.80</td><td>15.42</td><td>3.69</td></tr><tr><td>Selective-FD</td><td>12.15</td><td>18.08</td><td>17.15</td><td>15.79</td><td>3.19</td></tr><tr><td>Fed-DFA</td><td>12.99</td><td>18.29</td><td>17.85</td><td>16.38</td><td>2.94</td></tr></table></body></html>

Table 3: Comparison of testing accuracy $( \% )$ and computation time per epoch (in seconds). FedDF- $6 \%$ represents $6 \%$ of the public distillation data used.   

<html><body><table><tr><td></td><td colspan="2">CIFAR-10</td><td colspan="2">CIFAR-100</td></tr><tr><td></td><td>Acc (↑)</td><td>Time (↓)</td><td>Acc (↑)</td><td>Time (↓)</td></tr><tr><td>FedDF-6%</td><td>60.87</td><td>114.6</td><td>22.60</td><td>116.1</td></tr><tr><td>FedDF-12%</td><td>61.77</td><td>138.0</td><td>25.21</td><td>138.8</td></tr><tr><td>FedDF-24%</td><td>62.54</td><td>185.8</td><td>27.10</td><td>188.1</td></tr><tr><td>FedDF-Random-6%</td><td>61.53</td><td>137.5</td><td>24.96</td><td>136.7</td></tr><tr><td>Fed-DFA-6%</td><td>62.23</td><td>169.1</td><td>26.06</td><td>169.6</td></tr></table></body></html>

Tiny-ImageNet. Table 2 compares mAP on TinyImageNet over VGG16, ResNet50 and ViT/16. Normally, the decision boundary becomes more difficult to characterize under complex data distributions such as ImageNet. FedDFA outperforms the baseline methods under both IID/NonIID settings. Further, Fed-DFA achieves the lowest performance variation which potentially reduces the capacity gap among the participants on complex classification tasks.

# Ablation Studies

Amount of Distillation Data. Theorem 1 states that the loss scales down at $\mathcal { O } ( \log m / \sqrt { m } )$ regarding the amount of distillation data $m$ . We compare Fed-DFA with several baseline variations: 1) FedDF- $6 / 1 2 / 2 4 \%$ represent that a fixed amount of $6 / 1 2 / 2 4 \%$ public distillation data are used; 2) FedDF-Random- $6 \%$ represents the $6 \%$ of distillation data are randomly replaced with new data in each epoch. We compare with Fed-DFA when $6 \%$ of the distillation data is used. Table 3 shows that accuracy increases with more distillation data, but the computation time also increases. It is interesting to see that Fed-DFA- $6 \%$ can achieve the same or higher accuracy than FedDF- $12 \%$ of double data size. Distilling samples near the decision boundary provides higher performance compared with distilling from a dynamic set of random samples (FedDF-Random- $6 \%$ ). In sum, although our proposed method slightly increases the computational cost due to $K$ -PGD attacks, it utilizes the distillation data more efficiently (less than half of the baseline).

00 300 600 S   
6 62 Average Accuracy 500 62 250 Time per Epoch 400 200 e P   
60 300 186   
59 1 5 10 15 20 200 m 60 0.002 0.004 0.006 0.008 0.01 TAivmereapgerAEcpcoucrhacy 10／ Adv Attack Step (K) Adv Attack Step Size (a) Adversarial step $( K )$ (b) Adversarial step size 180 28 180   
% 1.00x Average Accuracy S 1.00x Average Accuracy S   
Time per Epoch 160 27 Time per Epoch L 1.12x 5 .11x 1.18x 1.25 140 26 1.16x 1.22x 1.25x1.28x 140   
GR 120 e 120 m √a (a) CIFAR-10 (b) CIFAR-100

Impact of Adversarial Step $K$ and Size. $K$ determines how closely the decision boundary is characterized: a larger $K$ brings higher precision at an increasing computational cost. We change $K$ from $1 - 2 0$ to examine its impact on accuracy and computation speed in Fig. 7(a). We observe that the accuracy jumps significantly when $K$ increases from 1 to 5, but with marginal improvements over 10. Hence, we set $K = 5$ to achieve a good balance between precision and computation speed. Fig.7(b) illustrates the impact of adversarial step size from $0 . 0 0 2 - 0 . 0 1$ when $K = 5$ . Both accuracy and computation time are not sensitive to the step size.

Computation Reduction. To reduce computational overhead, we reduce the frequency of boundary estimates (estimate for every $i = 2 , 4 , 6 , 8 , 1 0$ epoch) and trace its accuracy change in Fig. 8, i.e., distilling from a stale estimate of the decision boundary. We observed that accuracy declines as $i$ increases, e.g., $1 . 3 \times$ speed-up causes $1 \%$ drop. A good balance occurs at $i = 4$ with $1 . 1 \times$ speed-up and less than $0 . 5 \%$ accuracy drop. This could match FedDF’s computation time while achieving higher accuracy.

# Conclusion

In this paper, we propose Fed-DFA for heterogeneous model fusion from an adversarial perspective. We successfully capture decision boundary dynamics characterized by the $K$ - step PGD and integrate this into a new loss function to make distillation attend to samples close to the decision boundaries for better generalization. Our extensive experiments over various datasets demonstrate the effectiveness of the proposed method compared to the benchmarks.