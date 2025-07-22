# Flexible Sharpness-Aware Personalized Federated Learning

Xinda $\mathbf { X _ { i n g } ^ { 1 , 2 * } }$ , Qiugang Zhan3, 4, 5\*, Xiurui Xie1†, Yuning Yang2, Qiang Wang6, Guisong Liu3, 4, 5†

1Laboratory of Intelligent Collaborative Computing, University of Electronic Science and Technology of China, Chengdu, China 2Computer Science and Engineering, University of Electronic Science and Technology of China, Chengdu, China   
3Complex Laboratory of New Finance and Economics, Southwest University of Finance and Economics, Chengdu, China 4Engineering Research Center of Intelligent Finance, Ministry of Education, Chengdu, China 5Kash Institute of Electronics and Information Industry, China 6School of Cyber Science and Technology, Shenzhen Campus of Sun Yat-sen University, Shenzhen, China {xingxd, yangyuning}@std.uestc.edu.cn, xiexiurui $@$ uestc.edu.cn, {zhanqg, gliu}@swufe.edu.cn, wangq637 $@$ mail2.sysu.edu.cn

# Abstract

Personalized federated learning (PFL) is a new paradigm to address the statistical heterogeneity problem in federated learning. Most existing PFL methods focus on leveraging global and local information such as model interpolation or parameter decoupling. However, these methods often overlook the generalization potential during local client learning. From a local optimization perspective, we propose a simple and general PFL method, Federated learning with Flexible Sharpness-Aware Minimization (FedFSA). Specifically, we emphasize the importance of applying a larger perturbation to critical layers of the local model when using the Sharpness-Aware Minimization (SAM) optimizer. Then, we design a metric, perturbation sensitivity, to estimate the layerwise sharpness of each local model. Based on this metric, FedFSA can flexibly select the layers with the highest sharpness to employ larger perturbation. Extensive experiments are conducted on four datasets with two types of statistical heterogeneity for image classification. The results show that FedFSA outperforms seven state-of-the-art baselines by up to $8 . 2 6 \%$ in test accuracy. Besides, FedFSA can be applied to different model architectures and easily integrated into other federated learning methods, achieving a $4 . 4 5 \%$ improvement.

Code and Appendix — https://github.com/xxdznl/FedFSA

# 1 Introduction

Traditional federated learning (FL) methods (McMahan et al. 2017; Li et al. 2020; Karimireddy et al. 2020) focus on obtaining one effective global model through collaborative learning. However, in practice, a single global model often fails to meet the needs of all clients due to the typically nonidentically and independently distributed (non-IID) nature of client data (Kairouz et al. 2021; Tan et al. 2022; Ye et al. 2023). Personalized federated learning (PFL) effectively alleviates the non-IID problem by customizing personalized models for each client. Previous research on PFL has primarily based on model interpolation (Hanzely and Richta´rik 2020; Ma et al. 2022; Zhang et al. 2023b; Wu et al. 2023), as well as parameter decoupling (Arivazhagan et al. 2019; Collins et al. 2021; Xu, Tong, and Huang 2023; Zhang et al. 2023a).

However, existing PFL methods mainly focus on effectively leveraging global and local information, neglecting the inherent generalization potential during local training. Therefore, recently, there has been a new trend to improve the local learning process of clients by applying optimizer methods from centralized learning to FL (Reddi et al. 2020; Caldarola, Caputo, and Ciccone 2022; Qu et al. 2022; Zhou and Li 2023). Among these methods, the SharpnessAware Minimization (SAM) optimizer (Foret et al. 2021) used in FedSAM (Caldarola, Caputo, and Ciccone 2022) and MoFedSAM (Qu et al. 2022) has shown outstanding generalization ability in centralized learning. SAM solves a min-max problem by minimizing both the training loss and sharpness, aiming to achieve a flat loss landscape and thereby improve the model’s generalization performance. SAM introduces a perturbation when addressing the maximization problem.

Recent research on SAM has shown promising findings on this perturbation. SSAM (Mi et al. 2022) and SAMON (Mueller et al. 2024) suggest that perturbing all parameters may not be necessary. Additionally, DISAM (Zhang et al. 2024) has demonstrated, through both theoretical analysis and experiments, that the larger perturbation can enhance the model’s generalization ability, provided that convergence is ensured. For the FL scenarios, PLGU (Qu et al. 2023) and FedSOL (Lee et al. 2024) investigate the application of perturbation to specific layer parameters. Additionally, they have devised an adaptive method for determining the perturbation amplitude based on either the level of model personalization or the disparity between local and global models. Though these methods enhance the model’s generalization performance, they barely investigate the influence of perturbation amplitude and the perturbed parameters from the

perspective of sharpness.

These recent works inspire us to rethink how to select the parameters to perturb and the appropriate perturbation amplitude when applying SAM in FL. Therefore, we propose a novel PFL method, called Federated Learning with Flexible Sharpness-Aware Minimization (FedFSA). Specifically, we design the perturbation sensitivity metric to estimate sharpness. Based on this metric, FedFSA customizes the SAM optimizer for each client by selecting the layers with the highest sharpness relative to the global model and applying larger perturbations to these layers, aiming for higher generalization performance. Additionally, FedFSA can be seamlessly integrated with momentum to further assist clients in escaping local flat minima. Our contributions can be summarized as:

• We analyze the sharpness in relationship to perturbation and then design a novel metric, perturbation sensitivity, by using perturbation to approximate layer-wise sharpness. To leverage the generalization-enhancing properties of larger perturbation in SAM for personalized federated learning, we further design the global perturbation sensitivity metric.   
• Guided by global perturbation sensitivity, which evaluates critical layers based on the sharpness between the local and global models, We propose FedFSA, a flexible sharpness-aware personalized federated learning method, which flexibly applies larger perturbation to critical layers for different clients, thereby enhancing clients’ generalization.   
• We conducted extensive experiments on FashionMNIST, CIFAR10, CIFAR100, and Tiny-ImageNet datasets with two types of non-IID data partitions. Compared to the state-of-the-art methods, FedFSA achieves a significant performance improvement of $8 . 2 6 \%$ while maintaining computational and communication overheads similar to MoFedSAM. Additionally, FedFSA demonstrated the applicability to different model architectures and various federated learning methods with up to $4 . 4 5 \%$ improvement.

# 2 Related Works

This section first reviews research on perturbation in SAM within centralized learning and then discusses the application of SAM in FL scenarios.

# 2.1 Research on SAM perturbation

Smooth loss landscape is generally associated with better generalization capabilities (Keskar et al. 2016; Neyshabur et al. 2017; Li et al. 2018; Izmailov et al. 2018). Various studies have explored SAM and its perturbation.

Studies on perturbation amplitude (Liu et al. 2022; Ahn, Jadbabaie, and Sra 2024) show that random perturbation can help escape the sharp minima. (Si and Yun 2024) proves that a constant perturbation may lead SAM to converge to additive factors proportional to the square of the perturbation amplitude. To address the limitations of constant perturbation, DSAM (Chen, Li, and Chen 2024) dynamically adjusts the perturbation neighborhood of SAM based on local loss surface properties. DISAM (Zhang et al. 2024) addresses the issue of inconsistent SAM convergence across domains by minimizing variance in domain loss during sharpness estimation. DISAM prevents excessive or insufficient perturbation and demonstrates that larger perturbations can enhance generalization, albeit potentially at the expense of convergence speed.

Studies on perturbation scope Neural network models commonly possess a hierarchical nature. (Zhang, Bengio, and Singer 2022) highlights the importance of considering the unique contributions of each layer in the model, rather than treating the network as a monolithic block. LLMC (Adilova et al. 2024) further supports this by showing that models exhibit hierarchical robustness to perturbation. As for SAM, SSAM (Mi et al. 2022) reveals that perturbing only a small subset of parameters can maintain or even enhance SAM performance. (Lyu, Li, and Arora 2022) suggests that normalization layers can reduce the sharpness of the loss surface, contributing to a flatter loss landscape. Furthermore, SAMON (Mueller et al. 2024) experimentally verifies that applying SAM perturbation solely to batch normalization layers can achieve or exceed the effects of perturbing the entire model.

# 2.2 Federated SAM-based approaches

FedSAM (Caldarola, Caputo, and Ciccone 2022) is the pioneer in applying SAM to FL. Based on this, MoFedSAM (Qu et al. 2022) introduces an enhancement by incorporating local momentum. PLGU (Qu et al. 2023) adaptively applies perturbations to layers based on their degree of personalization. FedSOL (Lee et al. 2024) employs proximal gradient terms to estimate perturbation and suggests that applying perturbation only to the final layer is sufficient. FedGAMMA (Dai et al. 2023) enhances FedSAM by integrating variance reduction of Scaffold (Karimireddy et al. 2020). FedSpeed (Sun et al. 2023b) optimizes the FL process using the Alternating Direction Method of Multipliers (ADMM). Additionally, FedSMOO (Sun et al. 2023a) noticed that local perturbation guides client models toward their respective local flat minima, leading to model inconsistency, therefore, FedSMOO employs two ADMM methods to correct both local updates and perturbation. FedLESAM (Fan et al. 2024) uses a global perturbation parallel to the global gradient direction as a local perturbation estimate for the client, thereby allowing local clients to perceive the sharpness of the global model. FedMRUR (An et al. 2024) utilizes the hyperbolic graph fusion technique to mitigate model inconsistency. Moreover, SAM also has been applied in decentralized federated learning (Shi et al. 2023; Li et al. 2023).

However, these methods often overlook the impact of different perturbation amplitude and the selection of parameters to perturb from the perspective of sharpness. Given the significant influence of SAM’s perturbation amplitude on convergence speed and accuracy, it is essential to develop a reasonable perturbation strategy in PFL.

![](images/3fc22457cefa1c81bfba963976ca17baef4e4b557d2381d89d7bc6c8a9bba21b.jpg)  
Figure 1: Understanding sharpness intuitively.

# 3 Preliminaries

In this section, we introduce the problem definition of PFL and then explain the concept of SAM.

# 3.1 Problem Definition of PFL

Unlike traditional federated learning, for a system with $N$ clients, PFL trains a corresponding personalized local model $W _ { i }$ for each client $i$ $( i \in [ 1 , N ] )$ on their local data ${ D } _ { i } = \{ { D } _ { i } ^ { t r a i n } , { D } _ { i } ^ { t e s t } \}$ . We use $\mathscr { L } _ { i } ( W _ { i } ; D _ { i } )$ to denote the loss function of client $i$ . The PFL goal is formulated as follows:

$$
\operatorname* { m i n } _ { W _ { 1 } , W _ { 2 } , \ldots , W _ { N } } \sum _ { i = 1 } ^ { N } \mathcal { L } _ { i } ( W _ { i } ; D _ { i } ) ,
$$

Each client minimizes its loss $\mathscr { L } _ { i } ( W _ { i } ; D _ { i } ^ { t r a i n } )$ with local data $D _ { i } ^ { t r a i n }$ . Due to the limited local data of each client, the model is prone to overfitting easily, resulting in poor generalization performance of $W _ { i }$ on $\dot { D } _ { i } ^ { t e s t }$ . The primary challenge in PFL is to mitigate the impact of non-IID data and improve the generalization performance of clients on local data during client collaboration.

# 3.2 Sharpness-Aware Minimization

SAM has shown strong generalization capabilities in centralized learning and is promising to alleviate the non-IID data challenge in PFL. The objective of SAM is to minimize the loss function and smooth the loss landscape by solving the following min-max problem:

$$
\operatorname* { m i n } _ { W } \operatorname* { m a x } _ { \| \varepsilon \| _ { 2 } \leq \rho } \mathcal L ( W + \varepsilon ) ,
$$

where $W$ is the model weights, and $\varepsilon$ is the perturbation added when solving the maximization problem, constrained by the perturbation amplitude $\rho$ . The sharpness of $\mathcal { L } ( W )$ in the neighborhood of $W$ , i.e., $U ( W , \pmb \varepsilon ) =$ $\{ W ^ { \prime } \mid | W ^ { \prime } - W | < \varepsilon \}$ , is defined as follows:

considered as simultaneously minimizing the sharpness and loss in the training space.

$$
\operatorname* { m i n } _ { W } [ \operatorname* { m a x } _ { \| \varepsilon \| _ { 2 } \leq \rho } { \mathcal L ( W + \varepsilon ) - \mathcal L ( W ) + \mathcal L ( W ) } ] .
$$

SAM uses the first-order Taylor expansion and the dual norm to obtain an approximate solution to the perturbation:

$$
\varepsilon = \underset { \| \varepsilon \| _ { 2 } \leq \rho } { \mathrm { a r g m a x } } \mathcal L ( W + \varepsilon ) \approx \rho \cdot \frac { \nabla _ { W } \mathcal L ( W ) } { \| \nabla _ { W } \mathcal L ( W ) \| _ { 2 } } .
$$

Since $\frac { \nabla _ { W } \mathcal { L } ( W ) } { \| \nabla _ { W } \mathcal { L } ( W ) \| _ { 2 } }$ is the gradient direction at $W$ , the internal maximization problem can be regarded as a one-step gradient ascent at $W$ , with $\rho$ being the ascent step size. Finally, the gradient $\nabla _ { W } \mathcal { L } ( W + \varepsilon )$ at the perturbed parameter is used to solve the external minimization problem through a one-step gradient descent to actual update parameter $W$ .

# 4 Flexible Sharpness-Aware Method

In this section, we introduce our FedFSA, which includes the motivation of FedFSA, the perturbation sensitivity metric to estimate layer-wise sharpness, and the application of perturbation sensitivity in PFL. Finally, we provide the FedFSA procedure in detail.

# 4.1 Motivation

Inspired by DISAM (Zhang et al. 2024) and FedLESAM (Fan et al. 2024), which suggest that larger perturbations may improve generalization but slow convergence, and by SSAM (Mi et al. 2022) and SAMON (Mueller et al. 2024), which indicate that not all parameters need perturbation, we hypothesize that the slow convergence caused by large perturbations may result from applying these perturbations to unnecessary parameters. Based on this, we designed FedFSA, which flexibly selects parameters to apply large perturbations according to their sharpness relative to the global model.

It should be noted that typically, the sharpness calculation requires the Hessian trace of the parameters (Ahn, Jadbabaie, and Sra 2024), which is unacceptable in resourceconstrained FL environments. Additionally, sharpness can be computed using the defined formula in Eq. (3), but this approach does not permit layer-wise sharpness calculations. Due to the lack of a suitable method for calculating sharpness in FL, we first design a metric, perturbation sensitivity, to estimate the layer-wise sharpness.

# 4.2 Perturbation Sensitivity

Given a model $W$ with $L$ layers whose parameter set is expressed as $W = \{ w _ { 1 } , \ldots , w _ { k } , \ldots , w _ { L } \}$ . By Eq. (3), we use $\mathcal { L } _ { \mathrm { S A M } } ( W ) = \mathcal { L } ( W + \varepsilon )$ to simplify the maximum loss in the neighborhood of $W$ . Therefore, the sharpness of $W$ can be simplified to:

$$
\operatorname* { m a x } _ { \| \pm \| _ { 2 } \leq \rho } { \mathcal { L } } ( W + \pmb { \varepsilon } ) - { \mathcal { L } } ( W ) .
$$

Fig. 1 provides an intuitive display of the sharpness of the training loss in the neighborhood $U ( W , { \boldsymbol { \varepsilon } } )$ . The smaller the sharpness, the flatter the loss landscape. Eq. (2) can be

$$
\qquad { \mathcal { L } } _ { \mathrm { S A M } } ( W ) - { \mathcal { L } } ( W ) .
$$

Inspired by parameter sensitivity (Lee, Ajanthan, and Torr 2018; Molchanov et al. 2019; Zhang et al. 2022; Wu et al. 2023), we associate sharpness with perturbation and propose

Send model to server Conv Conv Conv Conv Conv Conv Conv   
Clients with non-IID data Allocate global model Conv Aggregation on server   
For any client Local Initialization Local Training After Local Training   
① Receive global model FSA module ⑤ FSA select TopC layers by sharpness for the next communication round TopC layers with the highest sharpness Conv Estimate loss U Conv sharpness sharpest Conv sharpness Estimate loss sharper   
② Generate data and start ③ sFeSleActaepdplfyrolamrgtehreplaesrtucrobamtimounntiocaltaiyoenrsround ④ CSeonmsiptuitveityGltobpaelrcPeirvteusrhbartpionenss local training

the concept of perturbation sensitivity as an approximation of sharpness. We define the perturbation sensitivity of the $k$ - th layer as the degree of variation in the model output or loss function after removing the perturbation of the $k$ -th layer parameter $w _ { k }$ . That is, let $\pmb { \varepsilon } = \{ \epsilon _ { 1 } , . . . , \epsilon _ { k } , . . . , \epsilon _ { L } \}$ be the perturbation of each layer, the perturbation sensitivity of the $k$ -th layer $s _ { k }$ can be expressed as:

$$
s _ { k } = | \mathcal { L } ( W + \boldsymbol { \varepsilon } ) - \mathcal { L } ( w _ { 1 } + \epsilon _ { 1 } , \dots , w _ { k } , \dots , w _ { L } + \epsilon _ { L } ) | .
$$

However, as can be seen in Eq. (7), evaluating the layer-wise perturbation sensitivity requires an additional forward propagation, which seriously increases the computational overhead. To solve this problem, we use the first-order Taylor approximation to substitute the calculation of $s _ { k }$ :

$$
\begin{array} { l } { s _ { k } = | \mathcal { L } ( W + \varepsilon ) - \mathcal { L } ( w _ { 1 } + \epsilon _ { 1 } , \dots , w _ { k } , \dots , w _ { L } + \epsilon _ { L } ) | } \\ { = | \nabla _ { w _ { k } } \mathcal { L } ( W + \varepsilon ) \cdot \epsilon _ { k } + R _ { 1 } ( W + \varepsilon ) | } \\ { \approx | \nabla _ { w _ { k } } \mathcal { L } ( W + \varepsilon ) \cdot \epsilon _ { k } | . } \end{array}
$$

The gradients of parameters can be obtained by one back propagation, which reduces the computation overhead of $s _ { k }$ .

# 4.3 Global Perturbation Sensitivity

In FL scenarios, clients typically update the model locally for several epochs. To reduce computation cost, a straightforward way to apply Eq.(8) to FL is to compute the perturbation sensitivity only in the final local epoch. However, the fluctuation of one iteration ignores the information of sharpness accumulated throughout the entire local training process, which may lead to suboptimal results. To overcome this limitation, we introduce the global perturbation sensitivity metric, which estimates the sharpness of the clients relative to the global model by measuring the variation in loss before and after local training, thereby perceiving sharpness in a wide range of neighborhoods.

Therefore, we denote the loss of the global model at the local initialization phase as $\mathcal { L } ( W _ { g } )$ , and the loss at the end of the local training phase of client $i$ as $\mathcal { L } _ { i } ( W _ { i } )$ . The variation in parameters $W _ { g } \mathrm { ~ - ~ } W _ { i }$ during training can be considered as a large perturbation $\varepsilon _ { i }$ , added to the local model to reasonably explore the neighborhood, i.e., $W _ { g } = W _ { i } + \varepsilon _ { i }$ and $\mathcal { L } _ { \mathrm { S A M } } \bar { ( } W _ { i } \bar { ) } = \mathcal { L } _ { i } ( W _ { g } )$ . Consequently, the global perturbation sensitivity of clients is defined as follows:

$$
\begin{array} { r l } & { s _ { i } = \vert { \mathcal { L } } _ { \mathrm { S A M } } ( W _ { i } ) - { \mathcal { L } } _ { i } ( W _ { i } ) \vert } \\ & { \quad = \vert { \mathcal { L } } _ { i } ( W _ { i } + \pmb { \varepsilon } _ { i } ) - { \mathcal { L } } _ { i } ( W _ { i } ) \vert . } \end{array}
$$

Similarly to Eq. (8), the global perturbation sensitivity of the $k$ -th layer of the client $i$ is:

$$
s _ { i , k } \approx | \nabla _ { w _ { i , k } } \mathcal { L } _ { i } ( W _ { i } + \pmb { \varepsilon } _ { i } ) \cdot \boldsymbol { \epsilon } _ { i , k } | .
$$

Specifically, in Eq. (10), we replace $\nabla _ { w _ { i , k } } \mathcal { L } _ { i } ( W _ { i } + \varepsilon _ { i } )$ with the variation of parameter ∆wi,k = wit,k − . Here, $w _ { i , k } ^ { t , e }$ represents the $k$ -th layer parameter in communication round $t \in [ 1 , T ]$ for client model $W _ { i }$ after the $e$ -th local iteration, $e \in [ 0 , E ]$ . The actual value of the perturbation $\epsilon _ { i , k } = w _ { i , k } ^ { t , 0 } - w _ { i , k } ^ { t , E }$ . Therefore, $\boldsymbol { s } _ { i , k } ^ { t }$ can be ultimately com

$$
\begin{array} { r l } & { \boldsymbol { s } _ { i , k } ^ { t } \approx | \Delta w _ { i , k } \cdot \boldsymbol { \epsilon } _ { i , k } | } \\ & { \qquad = | ( \boldsymbol { w } _ { i , k } ^ { t , E } - \boldsymbol { w } _ { i , k } ^ { t , 0 } ) \cdot \boldsymbol { \epsilon } _ { i , k } | } \\ & { \qquad = ( \boldsymbol { w } _ { i , k } ^ { t , E } - \boldsymbol { w } _ { i , k } ^ { t , 0 } ) ^ { 2 } . } \end{array}
$$

oSincceedtuhreinpge rtthuer beantiroenlsoecnaslitirvaitnyinognlpyronceesdss, taondb $w _ { i , k } ^ { t , E } - w _ { i , k } ^ { t , 0 }$ can be directly used for the subsequent momentum calculations, the computation overhead of FedFSA and MoFedSAM is essentially the same.

Input:communication round $T$ , local iterations $E$ , perturbation amplitude $\rho _ { \mathrm { l a r g e r } }$ and $\rho _ { \mathrm { d e f a u l t } }$ , FSA select $T o \bar { p } [ C ]$ layers, global and local learning rate $\eta _ { g } , \eta _ { l }$ , momentum coefficient $\alpha$ , the number of local updates $K$ , momentum for clients $\Delta$ .

1: for $t = 1 , 2 , \dots , T$ do 2: server select active clients set $S ^ { t }$ at round t 3: for client $i \in S ^ { t }$ parallel do 4: client $i$ receive server $\boldsymbol { W } _ { g } ^ { t }$ and set $W _ { i } ^ { t , 0 } = W _ { g } ^ { t }$ 56: i $e = 1 , 2 , \ldots , E$ $w _ { i , k } ^ { t , e }$ , i.e., $k \in T o p _ { i } [ C ]$ then 7: apply larger perturbation $\epsilon _ { i , k } ^ { t , e } = \rho _ { \mathrm { l a r g e r } } \frac { \nabla \mathcal { L } _ { i } ( w _ { i , k } ^ { t , e } ) } { \| \nabla \mathcal { L } _ { i } ( W _ { i } ^ { t , e } ) \| }$ 8: else 9: apply default perturbation $\epsilon _ { i , \boldsymbol { k } } ^ { t , e } = \rho _ { \mathrm { d e f a u l t } } \frac { \nabla \mathcal { L } _ { i } ( w _ { i , \boldsymbol { k } } ^ { t , e } ) } { \| \nabla \mathcal { L } _ { i } ( W _ { i } ^ { t , e } ) \| }$ 10: end if 11: $\begin{array} { r l } & { v _ { i } ^ { t , e + 1 } = \alpha \nabla { \mathcal { L } } _ { i } ( W _ { i } ^ { t , e } + \varepsilon _ { i } ^ { t , e } ) + ( 1 - \alpha ) \Delta ^ { t } } \\ & { W _ { i } ^ { t , e + 1 } = W _ { i } ^ { t , e } - \eta \imath v _ { i } ^ { t , e + 1 } } \end{array}$ 12: 13: end for 14: Parameter variation after local update ∆t = W t,E W t,0 15: compute perturbation sensitivity $s _ { i } ^ { t } = ( \Delta _ { i } ^ { t } ) ^ { 2 }$ and select $T o p _ { i } [ C ]$ Layers sorted by $s _ { i } ^ { t }$ 16: end for 17: server aggregate ∆tg+1 $\begin{array} { r } { \Delta _ { g } ^ { t + 1 } = \frac { 1 } { | S ^ { t } | } \sum _ { i \in S ^ { t } } \Delta _ { i } ^ { t } } \end{array}$ 18: Update global parameter $W _ { g } ^ { t + 1 } = W _ { g } ^ { t } + \eta _ { g } \Delta _ { g } ^ { t + 1 }$ 19: momentum for clients ∆t+1 $\begin{array} { r } { \Delta ^ { t + 1 } = \frac { 1 } { \eta _ { l } K } \Delta _ { g } ^ { t + 1 } } \end{array}$ 20: end for 21: return $[ W _ { 1 } ^ { T } , W _ { 2 } ^ { T } , . . . , W _ { N } ^ { T } ]$

# 4.4 Flexible Sharpness-Aware Procedure

During the gradient ascent step in SAM on the client-side local learning process, FedFSA applies larger perturbation to critical layers that can represent the entire model. To identify critical layers, We use the global perturbation sensitivity metric. The TopC most sensitive layers are selected to employ larger perturbation in the next communication round. By customizing the SAM optimizer for each client, FedFSA mitigates the impact of non-IID data and enhances the generalization performance of local clients. The details of the FedFSA process are described in Algorithm 1.

Initialization In each communication round $t \in [ 1 , T ]$ , client $i \in [ 1 , N ]$ trains $E$ iterations minimizing its local loss function (e.g., Cross-Entropy loss for image classification tasks) to update the personalized model $W _ { i } ^ { t } \ =$ $\{ w _ { i , 1 } ^ { t } , \ldots , w _ { i , k } ^ { t } , \ldots , \overset { \cdot } { w } _ { i , L } ^ { t } \}$ . The $\mathsf { \bar { T } } o p _ { i } [ C ]$ list of each client $i$ is initialized to empty, and $C$ is a hyperparameter of FedFSA to control the number of critical layers to enlarge perturbation. The larger and default perturbation amplitude $\rho _ { \mathrm { l } }$ larger and $\rho _ { \mathrm { d e f a u l t } }$ . The learning rates $\eta _ { g } , \eta _ { l }$ for global and local model updates. The momentum coefficient $\alpha$ and the number of local updates $K$ .

Training In the $t$ -th communication round, we randomly select an activated client set $S ^ { t }$ according to the client participation rate and send the current global model $\mathbf { \textnormal { W } ^ { t } }$ to all activated clients. Lines.6-Lines.10, client $i$ applies a larger perturbation to the TopC critical layers selected in the previous round. Lines.11-Lines.12 Client $i$ uses the perturbed parameters to calculate the actual gradient and combines it with the global gradient for momentum update. After the local training is completed, Lines.15 client $\mathbf { \chi } _ { i }$ evaluates the TopC critical layers in preparation for the next training. Lines.17- Lines.19, the server aggregates the parameter variation $\Delta _ { i } ^ { t }$ of all participating clients, updates the global model, and calculates the momentum $\Delta ^ { t + 1 }$ for the client in the next round.

Details When evaluating critical layers, we follow the LLMC (Adilova et al. 2024), considering only the weight parameters of the model’s convolutional or fully connected layers, such as conv.weight and fc.weight, and excluding non-weight layers, such as conv.bias, fc.bias, and batch normalization layers (both bn.weight and bn.bias).

# 5 Experiment

# 5.1 Experiment Setting

Dataset and partition strategy We evaluate our FedFSA using four public image classification benchmark datasets: Fashion-MNIST (FMNIST) (Xiao, Rasul, and Vollgraf 2017), CIFAR10/100 (Krizhevsky, Hinton et al. 2009), and Tiny-ImageNet (TINY) (Le and Yang 2015). To verify the effectiveness of our method in different non-IID scenarios, we follow (Hsu, Qi, and Brown 2019) to partition the dataset using widely used Dirichlet (Dir) and Pathological (Pat) sampling, see more in Appendix B.1 and B.5. We ensure that each client holds the same amount of data regardless of the partitioning strategy. Specifically, in the Pat partition, for FMNIST and CIFAR10, each client is randomly assigned 5 classes i.e., $\operatorname* { P a t } ( s = 5 )$ , while for CIFAR100 and TINY are $\mathrm { P a t } ( s { = } 1 5 )$ and $\mathrm { P a t } ( s { = } 5 0 )$ , respectively. In the Dir partition, for FMNIST and CIFAR10, we set $\operatorname { D i r } ( \beta = 0 . 5 )$ , and for the rest, we set $\operatorname { D i r } ( \beta = 0 . 3 )$ . After partitioning, the data is split into $70 \%$ training and $30 \%$ testing sets.

Model architecture We follow the (Zhang et al. 2023a) and use a 5-layer Convolutional Neural Network (CNN) model similar to LeNet5 (Lecun et al. 1998). For the specific configurations, please refer to Appendix B.2.

Implementation details For FL training-related hyperparameters, we set the local learning rate to 0.1 and the global learning rate to 1.0. The number of local epochs is set to 10, and a batch size of 48. The maximum communication round is set to 250 for FMNIST and 500 for other datasets to ensure full convergence, with a $10 \%$ participation ratio of a total of 100 clients. At each communication round, we evaluate the average test accuracy obtained from all clients. All experiments are repeated with three random seeds $\{ 2 3 , 1 0 0 \$ , $2 0 0 \}$ . We employ SGD as the base optimizer for all baselines including SAM-based approaches. The hyperparameters of baselines are set according to their default configurations, as detailed in Appendix B.4, except for MoFedSAM, where we make specific adjustments to the perturbation amplitude as needed. For FedFSA, we set momentum coefficient $\alpha$ to 0.1, the perturbation amplitude $\rho _ { \mathrm { d e f a u l t } }$ to 0.05 (0.1 for FMNIST and CIFAR10) and $\rho _ { \mathrm { l a r g e r } }$ to 0.9 (0.2 for FMNIST and CIFAR10). Additionally, we set TopC to 2.

<html><body><table><tr><td rowspan="2">Method</td><td colspan="2">FMNIST(%)</td><td colspan="2">CIFAR10(%)</td><td colspan="2">CIFAR100(%)</td><td colspan="2">TINY(%)</td></tr><tr><td>Pat(5)</td><td>Dir(0.5)</td><td>Pat(5)</td><td>Dir(0.5)</td><td>Pat(15)</td><td>Dir(0.3)</td><td>Pat(50)</td><td>Dir(0.3)</td></tr><tr><td>FedAvg-FT</td><td>92.51±0.21</td><td>91.59±0.47</td><td>81.79±0.61</td><td>81.41±0.77</td><td>53.70±0.26</td><td>41.09±0.14</td><td>23.03±0.46</td><td>20.75±0.35</td></tr><tr><td>FedCR</td><td>93.67±0.11</td><td>92.43±0.41</td><td>84.00±0.14</td><td>83.49±0.32</td><td>59.91±0.35</td><td>42.74±0.39</td><td>23.71±0.78</td><td>1</td></tr><tr><td>FedALA</td><td>92.99±0.22</td><td>91.83±0.39</td><td>81.62±1.08</td><td>81.76±0.53</td><td>54.92±0.20</td><td>39.75±1.07</td><td></td><td></td></tr><tr><td>FedSAM</td><td>93.01±0.11</td><td>91.89±0.39</td><td>84.16±0.20</td><td>83.84±0.44</td><td>59.41±0.39</td><td>44.66±0.40</td><td>28.98±0.15</td><td>25.55±0.53</td></tr><tr><td>MoFedSAM</td><td>93.79±0.21</td><td>92.94±0.27</td><td>88.33±0.14</td><td>88.16±0.25</td><td>70.33±0.29</td><td>51.02±2.21</td><td>33.80±0.29</td><td>28.02±0.60</td></tr><tr><td>FedSMOO</td><td>93.51±0.05</td><td>92.46±0.19</td><td>87.22±0.16</td><td>87.03±0.24</td><td>66.56±0.68</td><td>52.34±1.17</td><td>26.29±0.02</td><td>21.96±0.04</td></tr><tr><td>FedSpeed</td><td>93.68±0.15</td><td>92.88±0.20</td><td>88.24±0.37</td><td>87.99±0.25</td><td>68.24±0.15</td><td>52.61±0.55</td><td>29.60±0.28</td><td>24.92±0.43</td></tr><tr><td>FedFSA</td><td>93.78±0.11</td><td>92.88±0.35</td><td>88.64±0.09</td><td>88.37±0.27</td><td>72.28±0.40</td><td>60.87±0.61</td><td>41.39±0.25</td><td>33.76±0.93</td></tr></table></body></html>

Table 1: Average test accuracy under Pathological and Dirichlet non-IID settings on FMNIST, CIFAR10, CIFAR100, and TINY. Bold fonts highlight the best accuracy.

Baselines The proposed method, FedFSA, is compared first with the classic FedAvg (McMahan et al. 2017) and then with two recent PFL methods representing two types: FedCR (Zhang et al. 2023a), which effectively utilizes shared representations between clients by minimizing the difference in local and global conditional mutual information, representing parameter decoupling; and FedALA (Zhang et al. 2023b), which adaptively aggregates the global model and local model towards the local objective, representing model interpolation. To demonstrate the superiority of reasonably enlarging perturbation in FedFSA, we also compare it with recent federated SAM-based approaches, such as FedSAM (Caldarola, Caputo, and Ciccone 2022), MoFedSAM (Qu et al. 2022), FedSpeed (Sun et al. 2023b), and FedSMOO (Sun et al. 2023a).

# 5.2 Performance Evaluation.

Due to the local overfitting problem of PFL methods, we take the best accuracy in each experiment. To ensure fairness for all methods, especially for non-PFL methods, we further tune the model classifiers of all clients using SGD in one round after training.

Comparison with baselines As shown in Table 1, FedFSA achieves state-of-the-art performance in most cases, except for FMNIST. Specifically, on the CIFAR100 dataset, FedFSA achieves $6 0 . 8 7 \%$ in the $\operatorname { D i r } ( 0 . 3 )$ setups, which is $8 . 2 6 \%$ higher than the second-highest accuracy. The more complex the dataset, the more pronounced the effectiveness of FedFSA. We attribute this to the fact that the model trained on simpler datasets has a flatter loss landscape and is easier to converge to a global minimum, thus larger perturbation has limited impact on improving generalization performance. In contrast, for more complex datasets, larger perturbation can be effective in escaping local sharp minima. For unknown reasons, FedCR and FedALA failed to converge on TINY, which is marked as — in the table.

Impact of heterogeneity and scalability To assess the effectiveness of FedFSA under varying degrees of heterogeneity, we adjust the $\beta$ in $\operatorname { D i r } ( \beta )$ on CIFAR100. In PFL settings, a smaller $\beta$ indicates greater heterogeneity, which typically results in higher test accuracy. In Table 2, As the heterogeneity decreases, the performance of all methods declines, but FedFSA consistently maintains a performance lead. To show the scalability of FedFSA, we also conduct two experiments with 50 and 100 clients in the Dir(0.3) setting on TINY. Given that the total amount of data for TINY remains constant across experiments, increasing the number of clients results in each client receiving fewer data, which exacerbates the scarcity of local data. Therefore, as illustrated in Table 2, when the number of clients increases to 100, the accuracy of all methods decreases to varying extents. Although the accuracy of FedFSA drops by about $3 \%$ , it still performs the best.

Table 2: Average test accuracy at different levels of heterogeneity on CIFAR100 and scalability with different numbers of clients on TINY.   

<html><body><table><tr><td rowspan="3">Method</td><td>Heterogenity</td><td>Scalability</td></tr><tr><td>CIFAR100(%)</td><td>TINY(%)</td></tr><tr><td>Dir(0.1) Dir(1)</td><td>50 clients 100clients</td></tr><tr><td>FedAvg-FT</td><td>52.92±0.4634.58±0.50 22.24±0.39 20.75±0.35</td><td></td></tr><tr><td>FedCR FedALA</td><td>55.87±0.44 30.31±0.36 28.62±0.39 52.56±1.3234.72±0.63</td><td></td></tr><tr><td>FedSAM FedSMOO</td><td>58.42±1.0837.40±0.24 26.73±0.1225.55±0.53 MoFedSAM66.79±0.7544.41±0.5928.56±0.3828.02±0.60 65.85±0.71 46.58±0.52 25.87±0.27 21.96±0.04</td><td></td></tr><tr><td>FedSpeed</td><td>65.98±0.91 46.19±0.56 31.60±0.2524.92±0.43</td><td></td></tr><tr><td>FedFSA</td><td>67.73±0.6952.49±0.3536.90±0.4333.76±0.93</td><td></td></tr></table></body></html>

# 5.3 Applicability Evaluation

Method applicability As the FSA only modifies local optimizer, it can be applied to most existing FL methods. We apply FSA to baselines without modifying other learning processes to evaluate its effectiveness except for FedCR and FedSMOO. This exception might be because FedCR, a representation learning method, adds a representation layer to the model, and applying excessive perturbation to this layer

60.87 60.87 75   
58.55 57.06 52.61 70 51.02 Baseline FedFSA   
CR Baseline + FSA 65 MoFedSAMp=0.1 MoFedSAMp=0.3 41.09 40 39.75 MoFedSAM p=0.5 60 200 300 400 500 FedAvg-FT FedALA MoFedSAMFedSpeed Communication Round (a) CIFAR100 Dir(0.3) (b) CIFAR100 Dir(0.3)

might interfere with its output. FedSMOO uses perturbation to compute correction, and larger local perturbation might reduce its original effectiveness. We report the accuracy and improvements in Figure 3a on CIFAR100 in the Dir(0.3) setting. Since FedFSA can be seen as $\mathrm { F e d A v g } + \mathrm { F S A } + \mathrm { l o c a l }$ momentum, we use the same accuracy of 60.87 for FedAvgFT and MoFedSAM.

Model applicability To verify that our FedFSA still works on a deeper model, we conducted more experiments on ResNet18 (He et al. 2016) with batch normalization. We tune the parameter $\rho$ for MoFedSAM in the range of $\{ 0 . 0 5$ , 0.1, 0.2, 0.3, 0.4, 0.5, $0 . 6 \}$ , and selected the best $\rho = 0 . 3$ based on the balance between generalization accuracy and convergence speed. Then, we set $\mathrm { T o p C } = 5$ , $\rho _ { \mathrm { d e f a u l t } } = 0 . 0 5$ , and $\rho _ { \mathrm { l a r g e r } } = 1$ for FedFSA. As shown in Figure 3b, FedFSA achieves faster convergence and better generalization performance compared to MoFedSAM.

62.5 93.0 92.5 92.0 TopC=1 TopC=1 TopC=2 91.5 TopC=2 TopC=4 91.0 TopC=4 MoFedSAM MoFedSAM 50.00.05 0.30 0.500.70 1.00 90.50.05 0.30 0.50 0.70 1.00 Plarger Plarger (a) CIFAR100 Dir(0.3) (b) FMNIST Dir(0.5)

# 5.4 Ablation Evaluation

Effect of hyperparameter $\rho _ { \mathrm { l a r g e r } }$ and TopC Since perturbation amplitude $\rho$ critically influences the convergence and performance of SAM-based FL approaches, here we tune $\rho _ { \mathrm { l a r g e r } }$ and TopC of FedFSA on CIFAR100 and FMNIST. We then compare the average test accuracy with

MoFedSAM. As shown in Figure 4, the test accuracy on CIFAR100 initially increases significantly with $\rho _ { \mathrm { l a r g e r } }$ , benefiting from the large perturbation’s ability to escape sharp local minima. However, as $\rho _ { \mathrm { l a r g e r } }$ increases, the accuracy declines due to excessive perturbation. FMNIST exhibits the same characteristic; however, no matter how $\rho _ { \mathrm { l a r g e r } }$ and TopC are tuned, it does not achieve the performance of MoFedSAM due to the flatter loss landscape compared to CIFAR100.

60   
50   
U FedFSA FedFSA Reverse Reverse Random Random MoFedSAMp=0.1 MoFedSAMp=0.1 20 200 300 400 500 0 100 200 300 400 500 Communication Round Communication Round (a) CIFAR10 CNN (b) CIFAR100 CNN 94 75 93 FedFSA FedFSA Reverse Reverse Random Random MoFedSAM $\rho = 0 . 1$ MoFedSAMp=0.1 MoFedSAMp=0.3 55 200 300 400 500 100 200 300 400 500 CommunicationRound Communication Round (c) CIFAR10 ResNet18 (d) CIFAR100 ResNet18

Effect of FSA module We compare three critical parameter selection methods: ‘FedFSA’ and ‘Reverse’, which uses FSA to choose perturbation-insensitive layers, and ‘Random’, which randomly selects TopC layers as critical layers. We conduct experiments on CIFAR10 and CIFAR100 using CNN and ResNet18 models under $\operatorname { D i r } ( 0 . 3 )$ settings. The hyperparameters are consistent with Table 1 and Figure 3b, respectively.

The results are shown in Figure 5. For clarity, error bars are omitted. On the simpler CIFAR10 task, ‘FedFSA’ with CNN achieves the best accuracy throughout training, while with ResNet18, its results are similar to ‘Random’ due to the flatter loss landscape. Regardless of the model, ‘Reverse’ performs the worst. On CIFAR100, ‘Reverse’ shows faster convergence in the early stages but converges to a poor result later, with this drawback being more pronounced on the deeper ResNet18. In contrast, both ‘FedFSA’ and ‘Random’ converge to better results, with ‘FedFSA’ achieving a higher peak accuracy on ResNet18, indicating the selection scheme of FSA can better mitigate non-IID data issues.

# 6 Conclusion

In this study, we propose a novel PFL method, FedFSA, to improve the local optimization process by flexibly perceiving sharpness. FedFSA achieves or surpasses the performance of state-of-the-art methods on complex datasets such as CIFAR100 and under different levels of data heterogeneity, while not introducing much computational overhead. Additionally, we demonstrate the scalability of FedFSA with varying numbers of clients and its applicability to other FL methods and to models of different structures.

Limitations & Broader Impacts We were unable to theoretically analyze the specific impact of FedFSA on improving generalization. Additionally, the perturbation amplitude $\rho _ { \mathrm { l a r g e r } }$ is fixed, and there may be more effective adaptive methods to adjust $\rho _ { \mathrm { l a r g e r } }$ , which is a direction worth exploring in the future.

# Acknowledgments

This work was supported in part by the National Natural Science Foundation of China (NSFC) under Grant 62376228, the Sichuan Central-Guided Local Science and Technology Development under Grant 2023ZYD0165, and the Chengdu Science and Technology Program under Grant 2023-JB00- 00016-GX.