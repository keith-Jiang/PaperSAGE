# Partial Label Causal Representation Learning for Instance-Dependent Supervision and Domain Generalization

Yizhi Wang1,3, Weijia Zhang2, Min-Ling Zhang1,3\*

1School of Computer Science and Engineering, Southeast University, Nanjing 210096, China 2School of Information and Physical Sciences, The University of Newcastle, Callaghan, NSW 2308, Australia 3Key Laboratory of Computer Network and Information Integration (Southeast University), Ministry of Education, China wang yz@seu.edu.cn, weijia.zhang $@$ newcastle.edu.au, zhangml@seu.edu.cn

# Abstract

Partial label learning (PLL) addresses situations where each training example is associated with a set of candidate labels, among which only one corresponds to the true class label. As the candidate labels often come from crowdsourced workers, their generation is inherently dependent on the features of the instance. Existing PLL methods primarily aim to resolve these ambiguous labels to enhance classification accuracy, overlooking the opportunity to use this feature dependency for causal representation learning. This focus on accuracy can make PLL systems vulnerable to stylistic variations and shifts in domain. In this paper, we explore the learning of causal representations within an instance-dependent PLL framework, introducing a new approach that uncovers identifiable latent representations. By separating content from style in the identified causal representation, we introduce CausalPLL+, an algorithm for instance-dependent PLL based on causal representation. Our algorithm performs exceptionally well in terms of both classification accuracy and generalization robustness. Qualitative and quantitative experiments on instance-dependent PLL benchmarks and domain generalization tasks verify the effectiveness of our approach.

# 1 Introduction

Causal Representation Learning (CRL) (Scho¨lkopf et al. 2021) aims to infer compact high-level latent variables from high-dimensional and low-level observations. A core task in CRL is learning identifiable latent representation, i.e., developing representation learning algorithms that can provably identify high-level latent factors such as an object’s shape, location, and colour. While problems such as domain shift, out-of-distribution samples, and data bias have long plagued modern statistical learning systems (Liu et al. 2022; Zhu et al. 2025), CRL offers a unique and promising perspective to achieve greater effectiveness in robustness and generalization.

Since previous work has demonstrated that learning identifiable representations is impossible for arbitrary datagenerating process in an unsupervised setting (Locatello et al. 2019; Khemakhem et al. 2020), much of the recent efforts in CRL have been diverted to learning causal representation from data with additional structures and supervisions (Khemakhem et al. 2020; Kivva et al. 2022). For example, several recent studies have delved into understanding causal representations with additional information or under specific types of weak supervision signals (Zhang et al. 2022; Brehmer et al. 2022; Yao et al. 2021; Lin et al. 2024).

This paper explores the possibility of identifying causal representation under the Partial Label Learning (PLL) paradigm and its benefits for more stable and robust weakly supervised learning systems. PLL has garnered significant attention over the past decade as a form of weakly supervised learning due to its prevalence in many real-world applications, such as automatic image annotation (Chen, Patel, and Chellappa 2017; Tang, Zhang, and Zhang 2024b, 2023; Yang, Tang, and Zhang 2024), web mining (Luo and Orabona 2010; Scheffer, Decomain, and Wrobel 2001) and multimedia content analysis (Zeng et al. 2013; Cour et al. 2009; Tang et al. 2024; Tang, Zhang, and Zhang 2024a). Unlike standard supervised learning, where the training data contains i.i.d. samples associated with a single class label, learners in PLL are given samples associated with candidate label sets containing the (unknown) ground-truth label and several candidate labels.

In real-world PLL applications, candidate labels are typically provided by crowd-sourced annotators who select several labels that are likely correct. Therefore, the generation process of these candidate labels is closely tied to the characteristics of the instance, a concept adeptly termed InstanceDependent Partial Label Learning (IDPLL) (Xu et al. 2021). IDPLL is a realistic yet particularly challenging scenario. The candidate labels are related to the sample feature, making it difficult for the model to discern the ground-truth labels from the candidate set. Furthermore, the ambiguity of unknown ground-truth labels makes it difficult for models to learn the core differences between different categories.

Since the generation of instance-dependent candidate labels inherently depends on the instance feature under the IDPLL setting, it is desirable to model the generative relationship between the instance features and their associated candidate labels. For example, consider an image with the ground-truth label ‘Husky’, which may be provided with false-positive candidate labels such as ‘Wolf’ and ‘Samoyed’ due to their visual similarity. Instead of treating the candidate label set as noises and disambiguating the ground-truth label from false positive ones, it is advantageous to separate the characteristics specific to each breed from those shared across breeds and external conditions such as background or lighting. This would allow the model to concentrate on disambiguating the core features and reduce the influence of extraneous factors.

CRL provides unique tools for effectively modelling the generative relationship between the instance features and their associated candidate labels. As each candidate label inherently contains style and content information that is closely related to the instance feature, CRL algorithms have the potential to identify their corresponding lowdimensional latent factors. It is worth noting that the goal here is not to learn completely disentangled latent representation, i.e., identifying dimension-wise independent latent factors. Instead, we aim to identify latent representations that block-separated content from style, as this will not only facilitate the task of PLL classification but also improve the classifier’s robustness to distribution shifts.

From the perspective of exploiting instance-dependent candidate labels for causal representation learning, this work proposes a novel generative approach for effectively modelling the generation process of instances and candidate label sets while extracting identifiable and content-style disentangled causal representations. Furthermore, we introduce a prior-based contrastive learning method and a label refinement disambiguation strategy to further improve the model’s representation quality and classification performance. The proposed model not only achieves state-of-the-art classification performance on IDPLL benchmarks but also demonstrates robustness to the changes between training and test distributions. Our contributions can be summarized as:

• We introduce a novel VAE framework enabling the model to learn identifiable causal representations from data, achieving disentanglement of content and style. • Based on this framework, we propose an effective Partial Label Learning algorithm, CausalPLL $^ +$ , which enables effective disambiguation in instance-dependent scenarios through contrastive learning and label refinement. • We conduct extensive empirical studies on various datasets and settings, proposing a new, more realistic method for IDPLL data generation. Experimental results demonstrate the effectiveness of CausalPLL $^ +$ in IDPLL classification and domain shift scenarios.

# 2 Related Work

# 2.1 VAE and Identifiable Causal Representations

Variational Autoencoders (VAEs) are a class of deep generative models that combine amortized variational inference and neural networks to model the generation process by fitting the posterior and likelihood distributions of samples (Kingma and Welling 2013). Specifically, VAEs optimize the evidence lower bound (ELBO) of the likelihood:

$$
\begin{array} { r } { \mathbb { E } _ { z \sim q _ { \phi } ( z | x ) } \left[ \ln p _ { \theta } ( \pmb { x } | z ) \right] - \mathrm { K L } ( q _ { \phi } ( z | \pmb { x } ) | | p _ { \theta } ( z ) ) . } \end{array}
$$

VAEs are inherently related to the field of causal representation learning due to their flexibility in modelling probability graphical models. This has attracted considerable research efforts into their latent variable identifiability. Locatello et al. suggests that it is impossible to learn identifiable representations from the data in completely unsupervised settings. Meanwhile, Khemakhem et al. provided identifiable results under the VAE framework for the first time. It has been shown that latent factors $z$ can be identified by employing a conditionally factorized prior distribution $p _ { \pmb { \theta } } \overset { \cdot } { ( \pmb { z } | \pmb { u } ) }$ over the latent variables, where $\mathbf { \Delta } _ { \pmb { u } }$ is an additionally observed variable (Khemakhem et al. 2020).

![](images/1f89c3cf92144d73952e3e831969a961d0a5e94f8b673e7fdb5b04425a624d03.jpg)  
Figure 1: Framework of CausalPLL+. Unless otherwise specified (arrows), components within the shaded area do not exchange information with those outside. The dashed arrow on prior net and contrastive learning module means that CL is conducted between $z _ { c }$ and distribution parameters of all classes in the prior net, rather than $\pmb { A } [ \pmb { u } ]$ and $B [ { \pmb u } ]$ .

However, most recent efforts in CRL have been focusing on learning identifiable latent representations that are mutually independent and their causal structures (Brehmer et al. 2022; Lin et al. 2024); however, our focus is instead inferencing identifiable representations under the data generation process of IDPLL and exploring their benefits for partial label classification.

# 2.2 Partial Label Learning

Partial Label Learning (PLL) (Cour et al. 2009) is a subclass of weakly supervised learning (Zhou 2017). In PLL, each training sample is associated with a candidate label set containing an unknown ground-truth label and several false positive labels. Early efforts in PLL have focused on scenarios in which the candidate labels are randomly generated or class-dependent. For instance, Lv et al. (2020) and Wen et al. employ self-training techniques to determine ground-truth labels during training iteratively. Feng et al. (2020) studies consistent classifiers under the assumption of uniformly generated partial labels. Wu, Wang, and Zhang (2022) studies manifold-preserving consistency regularizations in PLL.

As the generation of candidate labels depends on the instance features in real-world PLL applications, IDPLL (Xu et al. 2021) more closely resembles the data generation process of practical situations. Xu et al. (2021) employ variational inference to estimate the latent label distribution. Wu, Wang, and Zhang (2024) perform knowledge distillation and leverage a rectifcation process to obtain reliable representations. However, existing literature (Qiao, Xu, and Geng

2022; Xia et al. 2022; Xu et al. 2023) rarely explores the generative process between examples and candidate label sets, nor does it address spurious features and domain shift issues in IDPLL scenarios. This paper explicitly models the generative relationship between instances and their candidate labels to extract causal representations which decouple content and style, improving classification and robustness.

# 3 Methodology

Notations Let $\boldsymbol { x } ~ \subset ~ \mathbb { R } ^ { D }$ denote the $D$ -dimensional instance space and $\mathcal { Y } = \{ 1 , 2 , \cdots , K \}$ denote the label space with $K$ distinct labels. $\mathcal { Z } \subset \mathbb { R } ^ { M }$ is the $M$ -dimensional latent space where $M \ll D$ . PLL assumes that the ground-truth label $y \in \mathcal { V }$ of an instance $\pmb { x } \in \mathcal { X }$ is contained within a candidate label set $s \subset \mathcal { y }$ . For simplicity, we use the Boolean vector $s \in \{ 0 , 1 \} ^ { K }$ to represent the partial label corresponding to $s$ . The goal of PLL is to learn a classifier $h : \mathcal { Z } \to \mathcal { V }$ on a partial label dataset $\mathcal { D } = \{ ( \pmb { x } _ { i } , \pmb { s } _ { i } ) | 1 \leq i \leq N \}$ . For the classifier $h$ , we use $h _ { k } ( z )$ to denote the output of classifier $h$ on label $k$ given input $z$ . For the VAE framework, we use $\pmb { A } , \pmb { B } \in \mathbb { R } ^ { K \times M _ { c } }$ to denote the matrix storing the mean and variance of the content prior $p ( z _ { c } | \boldsymbol { u } )$ for $K$ categories, respectively. $\textbf { \em u }$ stands for auxiliary variable, which is usually a normalized candidate label vector. For convenience, we denote the mean and variance corresponding to the categories contained in $\mathbf { \Delta } _ { \pmb { u } }$ as $\pmb { A } [ \pmb { u } ]$ and $B [ \pmb { u } ]$ , respectively. Lastly, we use $\hat { \pmb { s } }$ to represent the refined candidate labels.

# 3.1 Model Identifiability and Content-Style Disentangled Causal Representation

Identifiability in existing VAE frameworks is often achieved through a conditional prior $p ( \pmb { z } | \pmb { u } ) )$ , where $\mathbf { \Delta } _ { \pmb { u } }$ serves as an auxiliary variable. Suppose we could observe instance $\pmb { x } \in$ $\mathbb { R } ^ { D }$ and auxiliary variable $\boldsymbol { u } \in \mathbb { R } ^ { K }$ , and $\boldsymbol { z } \in \mathbb { R } ^ { M }$ is a latent variable. The observed instance $\scriptstyle { \mathbf { { \vec { x } } } }$ can be regarded as generated by $z$ through an arbitrary mixing function $f$ :

$$
\begin{array} { r } { \pmb { x } = \pmb { f } ( \pmb { z } ) + \pmb { \epsilon } , } \end{array}
$$

where $\epsilon$ is a noise variable with probability density function $p ( \epsilon )$ independent of $z$ or $f$ . Hence, we can express the posterior likelihood of the data in the following form:

$$
p _ { f } ( \pmb { x } | \pmb { z } ) = p _ { \epsilon } ( \pmb { x } - \pmb { f } ( \pmb { z } ) ) .
$$

Furthermore, let $\pmb \theta = ( \pmb f , \pmb T , \pmb \lambda )$ be the parameters of the following conditional generative model, the data generation process can be expressed as:

$$
p _ { \pmb { \theta } } ( \pmb { x } , z | \pmb { u } ) = p _ { f } ( \pmb { x } | z ) p _ { T , \lambda } ( z | \pmb { u } ) .
$$

Without loss of generality, it is common to assume that the latent prior distribution $p \mathbf { \bar { ( } } z \mathbf { \vert } u \mathbf { ) }$ follows the exponential family distribution:

$$
p _ { T , \lambda } ( z | u ) = \prod _ { i } \frac { Q _ { i } \left( z _ { i } \right) } { Z _ { i } ( u ) } \exp \left[ \sum _ { j = 1 } ^ { k } T _ { i , j } \left( z _ { i } \right) \lambda _ { i , j } ( u ) \right] .
$$

With the generative model specified according to (3)-(5), Khemakhem et al. (2020) have shown that the model parameters $( f , T , \lambda )$ can be identified up to an equivalence class induced by component-wise and invertible linear transformations with the following assumptions:

(a) The set $\{ \pmb { x } \in \pmb { \chi } | \phi _ { \epsilon } ( \pmb { x } ) = 0 \}$ has measure zero, where $\phi _ { \epsilon }$ is the characteristic function of the density $p _ { \epsilon }$ defined in (3).   
(b) The mixing function $f$ in (3) is injective.   
(c) The sufficient statistics $T _ { i , j }$ in (5) are differentiable almost everywhere, and $( T _ { i , j } ) _ { 1 \leq j \leq k }$ are linearly independent on any subset of $\mathcal { X }$ of measure greater than zero.   
(d) There exist $n k + 1$ distinct points $\mathbf { \Delta } u ^ { 0 } , \ldots , \mathbf { \Delta } u ^ { n k }$ such that the matrix

$$
{ \cal L } = ( \pmb { \lambda } ( \pmb { u } _ { 1 } ) - \pmb { \lambda } ( \pmb { u } _ { 0 } ) , \dots , \pmb { \lambda } ( \pmb { u } _ { n k } ) - \pmb { \lambda } ( \pmb { u } _ { 0 } ) )
$$

of size $n k \times n k$ is invertible.

According to the above theory, one basis for achieving identifiability is to introduce a prior $p ( \boldsymbol { z } | \boldsymbol { u } )$ which is conditioned on an auxiliary variable $\mathbf { \Delta } _ { \pmb { u } }$ . In weakly supervised classification problems such as semi-supervised learning and multi-instance learning (Zhang et al. 2022), a common approach to learning causal representations is to utilize the class information as an auxiliary variable and map it to prior parameters through a prior network. Unlike other weakly supervised learning methods, the supervision information in PLL does not provide exact class label indices, but rather a set of candidate labels. This means that the weak supervision signals in PLL cannot be directly translated into a specific class priority as in other methods.

A naive approach is straightforwardly mapping the labels in the candidate label set into a set of prior distributions; however, this approach poses several problems. Firstly, as the candidate label sets often exhibit highly imbalanced and long-tailed distributions (Wu, Wang, and Zhang 2024), using them directly would severely impede model learning. Furthermore, as the candidate label sets only contain class labels, i.e., label indexes instead of actual semantics, attempting to fit such simple relationships with flexible variational inference models can lead to the collapsing of the variational posterior, resulting in learning failures, as observed in our preliminary experiments.

To avoid posterior collapsing and effectively infer representations from candidate label sets, we propose to exploit the auxiliary information without directly mapping the candidate label sets into prior parameters. Specifically, assuming that each class’s latent corresponds to a Gaussian distribution in the latent representation space, we consider them a Gaussian mixture distribution containing the mixture components of their candidate labels with unknown mixing coefficients. Although the Gaussian mixture is not an exponential family distribution, there exists another Gaussian distribution $p ^ { * }$ that minimizes the reverse KL divergence between this distribution and the Gaussian mixture corresponding to the set. This distribution $p ^ { * }$ can be considered as the ”true” conditional prior corresponding to the current candidate label set. Formally, we have the following proposition:

Proposition. (Content prior). Let $p ^ { * } ( z _ { c } | \boldsymbol { u } )$ denote the ground-truth content prior. Then, $p ^ { * } ( z _ { c } | \boldsymbol { u } )$ minimizes the $K L$ divergence $\mathrm { K L } ( p ^ { * } \bar { ( } z _ { c } | \boldsymbol { u } ) | | p ( z _ { c } | \boldsymbol { u } ) \bar { ) }$ .

Prior $p ^ { * } ( z _ { c } | \boldsymbol { u } )$ is Gaussian which belongs exponential family, thus identifiability conditions could be satisfied. Moreover, we have the following theorem, which could make optimization more feasible.

Theorem 1. Suppose we have $p ( z _ { c } | \boldsymbol { u } )$ as a Gaussian mixture distribution:

$$
p ( \boldsymbol { z } _ { c } | \boldsymbol { u } ) = \sum _ { k = 1 } ^ { K } u _ { k } \cdot \varphi ( \boldsymbol { z } _ { c } ; A _ { k } , B _ { k } ) ,
$$

where $\varphi ( z _ { c } ; A _ { k } , B _ { k } )$ is the density of mixing component $\mathcal { N } ( A _ { k } , B _ { k } )$ . And we use

$$
p ^ { * } ( z _ { c } | \boldsymbol { u } ) = \varphi ( z _ { c } ; \mu _ { * } , \sigma _ { * } ^ { 2 } I ) ,
$$

to denote the distribution which minimizes the $K L$ divergence $\mathrm { K L } ( p ^ { * } ( z _ { c } | \boldsymbol { u } ) | | p ( z _ { c } | \boldsymbol { u } ) )$ . Then, minimizing $\mathrm { K L } ( q ( \boldsymbol { z } _ { c } | \boldsymbol { x } ) | | p ( \boldsymbol { z } _ { c } | \boldsymbol { u } ) )$ is equivalent to minimizing $\mathrm { K L } ( q ( \boldsymbol { z } _ { c } | \boldsymbol { x } ) | | p ^ { * } ( \boldsymbol { z } _ { c } | \boldsymbol { u } ) )$ .

Although the above discussion addresses the latent identifiability in PLL by effectively leveraging the weakly supervised information provided in the candidate label set, another unique hurdle exists: not all information contained in the latent factors is necessary for weakly supervised classification. To see this, consider partitioning the latent factors into the ones that capture content $z _ { c }$ and style information $z _ { e }$ , respectively. On the one hand, the content latent factors capture the core characteristics of each class shared across all instances. On the other hand, the style latent factors correspond to information not causally related to class, e.g., background and lighting variations that are inconsistent across different instances of the same class. Formally,

Assumption 1. (Content-invariance). For $\mathbfit { x } , \tilde { \mathbfit { x } } \in \mathcal { X }$ and $y = \tilde { y }$ , the conditional density of the latents $p _ { \tilde { z } | z }$ satisfies:

$$
p _ { \tilde { z } | z } ( \tilde { z } | z ) = \delta ( \tilde { z } _ { c } - z _ { c } ) p _ { \tilde { z } _ { e } | z _ { e } } ( \tilde { z } _ { e } | z _ { e } ) ,
$$

where $\boldsymbol { z } ~ = ~ ( z _ { c } , z _ { e } )$ , and $\delta$ is the Dirac delta function. In other words, $\tilde { z } _ { c } = z _ { c }$ almost everywhere.

Assumption 2. (Style-variation). Let $\mathcal { A }$ be a set containing subsets of styles $A \subseteq \{ 1 , \cdots , n _ { s } \}$ and let $p _ { A }$ be a probability distribution on $\mathcal { A }$ . The style conditional distributions should satisfy:

$$
p _ { \tilde { z } _ { e } | z _ { e } , A } ( \tilde { z } _ { e } | z _ { e } , A ) = \delta ( \tilde { z } _ { A ^ { c } } ^ { e } - z _ { A ^ { c } } ^ { e } ) p _ { \tilde { z } _ { A } ^ { e } | z _ { A } ^ { e } } ( \tilde { z } _ { A } ^ { e } | z _ { A } ^ { e } ) .
$$

Loosely speaking, the first assumption asserts that the content within each category should remain constant, while the second assumption specifies that certain style factors should change. Importantly, the second assumption is flexible as it does not require all styles to change.

However, VAEs in previous work uniformly extract all factors possibly needed for reconstruction. The vanilla VAE’s KL divergence requires the posterior distribution to fit a standard normal distribution as closely as possible, inevitably hindering clear separations between different categories. This also explains why using features extracted from VAEs for classification often yields dissatisfactory results in practical applications. In contrast, models like iVAE incorporate learnable priors during training, adjusting the KL divergence to bring the posterior distribution closer to the learned conditional priors. However, such settings default all latent factors to impact sample classification, indiscriminately utilizing non-causal features in the data, thereby inevitably suffering severe damage during distribution shift.

The above two cases not only affect the performance and robustness of the model, but also deviate from the original intention of causal representation learning. Summarizing the above two cases, we can see that if you want to acquire highquality features, style and content may be handled differently. Therefore, an idea of this paper is born. In the following methods, we will divide the latent embedding into two parts and treat them separately according to their characteristics. Specifically, we divide latent code $z$ into $z _ { c }$ and $z _ { e }$ , i.e. $z = ( z _ { c } ^ { T } , z _ { e } ^ { \check { T } } ) ^ { T }$ . Where $z _ { c }$ follows a conditional prior $p ( z _ { c } | \boldsymbol { u } )$ regulated by the auxiliary variable $\mathbf { \Delta } _ { \pmb { u } }$ , and the prior of $z _ { e }$ is a standard normal distribution. At the same time, because the components of $z$ are independent of each other, $z$ still obeys an exponential family distribution as a whole.

# 3.2 Overall Framework

Figure 1 provides a concise overview of the model’s structure. It consists primarily of five components: the encoder $q ( \boldsymbol { z } | \boldsymbol { x } )$ , the decoder $p ( \pmb { x } | \pmb { z } )$ , the prior network $p ( z _ { c } | \boldsymbol { u } )$ , the classifier $q ( \pmb { y } | \pmb { z } _ { c } )$ , and the contrastive learning module. $\scriptstyle { \mathbf { { \vec { x } } } }$ and the auxiliary variable $\mathbf { \Delta } _ { \pmb { u } }$ are fed into the encoder and prior network, respectively. Subsequently, we sample from the posterior distribution using reparameterization to obtain the latent code $z$ . As mentioned earlier, the latent embedding $z$ can be divided into two parts: the content embedding $z _ { c }$ , which encodes category-related information whose prior following conditional distribution $p ( z _ { c } | \boldsymbol { u } )$ , and the style representation $z _ { e }$ , independent of class, with its prior $p ( z _ { e } )$ following a standard normal distribution $\sqrt { ( 0 , I ) }$ .

Since the content and style components of $z$ are independent, the KL divergence can be expressed as:

$$
\mathrm { K L } ( q ( \boldsymbol { z } _ { c } | \pmb { x } ) | | p ( \boldsymbol { z } _ { c } | \pmb { u } ) ) + \mathrm { K L } ( q ( \boldsymbol { z } _ { e } | \pmb { x } ) | | p ( \boldsymbol { z } _ { e } ) ) ,
$$

where parameters for the conditional distribution $p ( z _ { c } | \boldsymbol { u } )$ are generated by the prior network. In practice, the prior network is implemented as a single-layer linear mapping similar to word embedding. This not only avoids issues of pattern collapse but also enhances the model’s interpretability. It is worth noting that instead of directly using the candidate label set $s$ as the auxiliary information, CausalPLL $^ +$ integrates the learning of latent representations with the refinement of the candidate label set. This approach addresses two problems inherent in IDPLL. Firstly, discerning the true class label from the candidate label set is naturally integrated with the inference of the representations. Secondly, refining the candidate label set further improves the auxiliary information for inferencing the representation. A more detailed elaboration of the integrated label refinement process is discussed in Section 3.3.

The evidence lower bound (ELBO) of the model can be

<html><body><table><tr><td>Dataset</td><td>Method</td><td>T=16</td><td>T=32</td><td>T=64</td></tr><tr><td rowspan="6">FashionMNIST</td><td>CausalPLL+</td><td>94.49 ± 0.37%</td><td>93.60 ± 0.10%</td><td>92.75 ± 0.18%</td></tr><tr><td>PLCR</td><td>93.28 ± 0.24%</td><td>92.46 ± 0.13%</td><td>90.72 ± 0.15%</td></tr><tr><td>VALEN</td><td>88.36 ± 0.20%</td><td>87.25 ± 0.19%</td><td>85.67 ± 0.24%</td></tr><tr><td>LWS</td><td>88.50 ± 0.19%</td><td>84.84 ± 0.51%</td><td>81.23 ± 2.07%</td></tr><tr><td>PRODEN</td><td>87.32 ± 0.19%</td><td>86.34 ± 0.08%</td><td>85.15 ± 0.24%</td></tr><tr><td>RC</td><td>89.56 ± 0.18%</td><td>89.05 ± 0.12%</td><td>87.65 ± 0.10%</td></tr><tr><td rowspan="6">Fully Supervised</td><td>CC</td><td>89.31 ± 0.07%</td><td>88.46 ± 0.03% 95.54 ± 0.07%</td><td>87.11 ± 0.11%</td></tr><tr><td>CausalPLL+</td><td></td><td></td><td></td></tr><tr><td>PLCR</td><td>98.49 ± 0.08% 97.84 ± 0.04%</td><td>97.89 ± 0.14% 96.03 ± 0.60%</td><td>96.96 ± 0.10%</td></tr><tr><td>VALEN</td><td>86.08 ± 0.37%</td><td>82.23 ± 0.36%</td><td>91.43 ± 0.58%</td></tr><tr><td>LWS</td><td>88.94 ± 0.17%</td><td>86.37 ± 0.89%</td><td>77.18 ± 0.56%</td></tr><tr><td>PRODEN</td><td>88.50 ± 0.24%</td><td>86.27 ± 0.33%</td><td>83.16 ± 0.46% 82.92 ± 0.45%</td></tr><tr><td rowspan="6"></td><td>RC CC</td><td>91.41 ± 0.07%</td><td>89.63 ± 0.06%</td><td>87.15 ± 0.11%</td></tr><tr><td>Fully Supervised</td><td>91.77 ± 0.08%</td><td>89.81 ± 0.12%</td><td>86.40 ± 0.15%</td></tr><tr><td>CausalPLL+</td><td></td><td>99.03 ± 0.04%</td><td></td></tr><tr><td>PLCR</td><td>97.50 ± 0.21%</td><td>97.05 ± 0.37%</td><td>96.56 ± 0.26%</td></tr><tr><td>VALEN</td><td>97.15 ± 0.09%</td><td>96.59 ± 0.15%</td><td>95.97 ± 0.18%</td></tr><tr><td>LWS</td><td>96.58 ± 0.20%</td><td>96.02 ± 0.39%</td><td>95.27 ± 0.37%</td></tr><tr><td rowspan="8"></td><td></td><td>96.24 ± 0.08%</td><td>95.87 ± 0.09%</td><td>94.79 ± 0.18%</td></tr><tr><td>PRODEN</td><td>96.18 ± 0.17%</td><td>95.31 ± 0.22%</td><td>94.83 ± 0.25%</td></tr><tr><td>RC</td><td>95.68 ± 0.24%</td><td>95.38 ± 0.13%</td><td>94.77 ± 0.16%</td></tr><tr><td>CC Fully Supervised</td><td>95.39 ± 0.26%</td><td>94.75 ± 0.47%</td><td>93.58 ± 0.38%</td></tr><tr><td>CausalPLL+</td><td></td><td>98.09 ± 0.06%</td><td></td></tr><tr><td>PLCR</td><td>95.91 ± 0.28%</td><td>94.04 ± 0.26%</td><td>89.66 ± 0.32%</td></tr><tr><td>VALEN</td><td>96.28 ± 0.09%</td><td>93.97 ± 0.07%</td><td>88.82 ± 0.11%</td></tr><tr><td></td><td>89.63 ± 0.34%</td><td>86.35 ± 0.32%</td><td>78.28 ± 0.41%</td></tr><tr><td rowspan="6">CIFAR10</td><td>LWS</td><td>85.38 ± 0.21%</td><td>81.47 ± 0.20%</td><td>74.10 ± 0.25%</td></tr><tr><td>PRODEN</td><td>93.84 ± 0.48%</td><td>90.07 ± 0.49%</td><td>86.36 ± 0.53%</td></tr><tr><td>RC</td><td>86.33 ± 0.11%</td><td>81.19 ± 0.11%</td><td>74.93 ± 0.21%</td></tr><tr><td>CC</td><td>86.26 ± 0.10%</td><td>82.73 ± 0.23%</td><td>76.48 ± 0.12%</td></tr><tr><td> Fully Supervised</td><td></td><td>97.67 ± 0.13%</td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td></tr></table></body></html>

Table 1: Accuracy (mean±std) comparisons on FashionMNIST, Kuzushiji-MNIST, SVHN and CIFAR10 with instancedependent partial labels on different ambiguity levels.

expressed as:

$$
\begin{array} { r l } { \mathcal { L } _ { \mathrm { E L B O } } } & { = \mathbb { E } _ { z \sim q ( z \mid x ) } \left[ \ln p ( \boldsymbol { x } \mid \boldsymbol { z } ) \right] } \\ & { - \mathrm { K L } ( q ( \boldsymbol { z } _ { c } | \boldsymbol { x } ) \| p ( \boldsymbol { z } _ { c } | \boldsymbol { u } = \hat { \boldsymbol { s } } ) ) } \\ & { - \mathrm { K L } ( q ( \boldsymbol { z } _ { e } | \boldsymbol { x } ) \| p ( \boldsymbol { z } _ { e } ) ) . } \end{array}
$$

As the only exact supervision information in PLL is that non-candidate labels are not ground truth, we utilize an “only negatives matter” loss function on the the content representation $z _ { c }$ for classification. In this part, we also use the refined candidate vectors $\hat { \pmb { s } }$ . Specifically,

$$
\mathcal { L } _ { \mathrm { e r r } } = \sum _ { k = 1 } ^ { K } ( 1 - \hat { s } _ { k } ) \cdot \ln ( 1 - h _ { k } ( z _ { c } ) ) .
$$

While performing reconstruction and classification, we also introduce a novel contrastive learning module based on the latent space. Specifically:

$$
\mathcal { L } _ { \mathrm { C L } } = \frac { - 1 } { | \mathcal { S } | } \sum _ { i \in \mathcal { S } } \ln \frac { \exp ( z \cdot \tilde { \mu } _ { i } ) } { \sum _ { j \in \mathcal { V } } \exp ( z \cdot \tilde { \mu } _ { j } ) } ,
$$

where $\tilde { \pmb { \mu } } _ { i }$ is the $i$ -th mean vector in content prior mean matrix $A$ . Finally, the loss function of CausalPLL $^ +$ is:

$$
\mathcal { L } = \lambda _ { \mathrm { E L B O } } \cdot \mathcal { L } _ { \mathrm { E L B O } } + \lambda _ { \mathrm { C L } } \cdot \mathcal { L } _ { \mathrm { C L } } + \mathcal { L } _ { \mathrm { e r r } } .
$$

# 3.3 Candidate Label Refinement

Traditional PLL disambiguation methods mostly solely utilize the discriminative information provided by supervision. However, by using variational generative models, we can also reconstruct the data generation process. In the CausalPLL $^ +$ framework, more precise auxiliary information enables the model to learn better priors, and these improved priors, in turn, help the model achieve more accurate classification, which leads to even more veracious auxiliary information. This iterative process ensures that by the end of training, the model not only effectively models the generative process and priors but also achieves outstanding discriminative performance. Therefore, we proposed a candidate label refinement strategy to gradually eliminate labels that are more likely to be wrong. Specifically, for each sample, we maintain a vector and perform momentum updates using the unnormalized prediction scores from the classifier.

<html><body><table><tr><td>Dataset</td><td>一 Method</td><td>T=16</td><td>T = 32</td><td>T=64</td></tr><tr><td rowspan="5">MNIST-MNIST-M</td><td>CausalPLL+</td><td>97.85 ± 0.11%</td><td>96.58 ± 0.10%</td><td>94.67 ± 0.18%</td></tr><tr><td>PLCR</td><td>97.63 ± 0.08%</td><td>95.34 ± 0.09%</td><td>94.59 ± 0.11%</td></tr><tr><td>PiCO</td><td>98.64 ± 0.07%</td><td>78.63 ± 1.60%</td><td>57.52 ± 4.17%</td></tr><tr><td>LWS</td><td>96.93 ± 0.09%</td><td>95.61 ± 0.15%</td><td>92.21 ± 0.18%</td></tr><tr><td>RC CC</td><td>96.77 ± 0.10% 97.08 ± 0.05%</td><td>96.47 ± 0.10% 96.15 ± 0.10%</td><td>93.59 ± 0.09%</td></tr><tr><td rowspan="6">MNIST→SVHN</td><td>CausalPLL+</td><td>94.13 ± 0.15%</td><td>93.11 ± 0.12%</td><td>94.45 ± 0.07% 91.04 ± 0.19%</td></tr><tr><td>PLCR</td><td>93.95 ± 0.10%</td><td>92.50 ± 0.09%</td><td>87.39 ± 0.18%</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>PiWS</td><td>95.57±0.12%</td><td>84.57±0.71%</td><td>42.30 ± 0.21%</td></tr><tr><td>RC</td><td>94.16 ± 0.08%</td><td>91.75 ± 0.10%</td><td>83.65 ± 0.15%</td></tr><tr><td>CC</td><td>93.81 ± 0.14%</td><td>91.81 ± 0.15%</td><td>86.39 ± 0.29%</td></tr><tr><td rowspan="6">USPS→SVHN</td><td>CausalPLL+</td><td>86.72 ± 0.16%</td><td>81.65 ± 0.17%</td><td>73.15 ± 0.25%</td></tr><tr><td>PLCR</td><td>83.70 ± 0.17%</td><td>77.32 ± 0.23%</td><td>69.23 ± 0.27%</td></tr><tr><td>PiCO</td><td>80.17 ± 0.18%</td><td>56.24 ± 1.14%</td><td>36.61 ± 7.23%</td></tr><tr><td>LWS</td><td>78.25 ± 0.15%</td><td>28.05 ± 4.78%</td><td>26.68 ± 5.25%</td></tr><tr><td>RC</td><td>71.17 ± 0.20%</td><td>53.34 ± 1.28%</td><td>43.92 ± 3.16%</td></tr><tr><td>CC</td><td>80.23 ± 0.06%</td><td>56.97 ± 1.34%</td><td>43.99 ± 1.27%</td></tr></table></body></html>

Table 2: Accuracy (mean±std) comparisons on MNIST $$ MNIST-M, MNIST $$ SVHN, SVHN $$ USPS with instancedependent partial labels on different ambiguity levels.

This process can be expressed as:

$$
\gamma _ { t + 1 } = ( 1 - m ) \cdot \gamma _ { t } + m \cdot \hat { \pmb { y } } ,
$$

where $m$ is the momentum factor and $\gamma$ is the average of past model predictions. The refined candidate vector could be expressed as:

$$
\hat { \pmb { s } } = \mathrm { s o f t m a x } ( \frac { \gamma - s \cdot \mathrm { i n t \mathrm { \ m a x } } } { T } ) ,
$$

where $T$ is the temperature. With this mechanism, we can progressively eliminate the least scoring classes from the current candidate label set.

# 4 Experiments

The experiments in this paper are primarily divided into three parts. Section 4.2 focuses on the model’s classification performance in IDPLL tasks. Section 4.3 investigates the model’s generalization ability in the presence of style variation. Finally, in Section 4.4, we examine the nature of representations extracted by CausalPLL $^ +$ and observe the different impacts of content embeddings and style embeddings on image generation.

# 4.1 Experiment Setup

Datasets For IDPLL classification tasks, experiments were conducted on four well-known benchmarks: FashionMNIST (Xiao, Rasul, and Vollgraf 2017), Kuzushiji-MNIST (Clanuwat et al. 2018), SVHN (Netzer et al. 2011), and CIFAR10 (Krizhevsky and Hinton 2009).

Regarding domain generalization issues, we utilized three sets of classic datasets in this domain, mixing them at different ratios in training and testing sets. These three pairs include MNIST $$ MNIST-M, MNIST ${ \partial } S \mathsf { V H N }$ , and SVHN $$ USPS. Among them, the MNIST-M dataset (Ganin and Lempitsky 2015) is obtained by blending digits from the original set over patches randomly extracted from colour photos from BSDS500 (Arbelaez et al. 2010). The mixing ratio of these three pairs was $8 0 \% - 2 0 \%$ in the training set and $20 \% 8 0 \%$ in the test set. More details on implementation can be found in the supplementary material.

Data Generation Method In previous partial label learning research, it is a common practice to manually corrupt the existing fully-supervised datasets into partially labelled versions. However, existing data generation methods in IDPLL may suffer from an underconfidence problem, causing synthetic data to diverge from real-world situations. For example, for a sample $\mathbf { \boldsymbol { y } } = ( 0 , 0 , 1 ) ^ { T }$ which has a very confident prediction $\pmb { \underline { { y } } } = ( . 0 1 , . 0 1 , . 9 8 ) ^ { T }$ , the corresponding $s$ would be $( 1 , 1 , 1 ) ^ { T }$ , which is very unconfident. And the contradiction appeared. Moreover, current research in IDPLL lacks an approach similar to those in classical PLL that adjusts the level of ambiguity in weakly supervised data. Therefore, we proposed a novel data generation method that closely approximates real-world scenarios while allowing control over the ambiguity of supervised information. In brief, $\tau$ represents the level of ambiguity in the candidate label set, where a higher $\tau$ indicates greater ambiguity. Details of this data generation mechanism will be provided in the appendix.

# 4.2 IDPLL Classification

In this section, we evaluated the classification performance of CausalPLL $^ +$ across varying levels of ambiguity in IDPLL tasks. As shown in Table ??, CausalPLL $^ +$ achieved superior performance across most levels of ambiguity on three benchmarks. Moreover, its performance notably outperformed other baseline models in situations with higher ambiguity levels. The reason for this is that when the degree of ambiguity is large, the model would be seriously disturbed by too many candidate labels. Therefore, the label refinement mechanism can eliminate more false candidates, thus contributing to the performance improvement of the model. The experiments demonstrate that CausalPLL $^ +$ excels not only in conventional IDPLL classification tasks but also highlights its effectiveness and superiority, suggesting its versatility and broader applicability as an algorithm.

![](images/9a13639bf3fcf2c2857a47c11f01542f2b574bdf4c42da4c1991e17a88901f78.jpg)  
Figure 2: (Left) Controlled image generation by sampling from the latent space. Images in one row share the same style bu have different contents. (Middle and Right) The t-SNE visualization for the latent space of style versus content.

# 4.3 Domain Shift

We now study the model’s performance under variations of styles and domain shifts. We selected three pairs of datasets, exhibiting an increasing level of domain shift. MNIST $\hookrightarrow$ MNIST-M involves mild changes in background and color, while M $\scriptstyle { \sqrt { \mathrm { I S T } } } \to \mathrm { S V H N }$ and $\mathrm { U S P S } { \scriptstyle  \mathbf { S V H N } }$ introduce significant variations in camera angles, digit styles, and background complexities. As the changes in distributions intensify, the results show that these domain shifts have a substantial impact on the model’s performance, with more severe shifts leading to a notable decline in accuracy.

Across all three benchmarks, CausalPLL $^ +$ outperformed the baselines in most instances. The performance degradation of the compared baselines is most noticeable on $\mathrm { U S P S } { \scriptstyle  } \mathrm { S V H N }$ , as most perform worse than a random guess. This demonstrates that the method’s representation decoupling mechanism effectively mitigates the impact of domain shifts, enhancing the model’s robustness against distribution shifts.

# 4.4 Quantitative Results and Visualization

In this section, we observe the impacts of $z _ { c }$ and $z _ { e }$ on image generation, while also studying their distinct properties in the representation space.

Figure 2 (Left) showcases the model’s generation results on the MNIST $$ MNIST-M dataset. These images are not reconstructions of real samples but direct samples from the latent space. Specifically, each row in the figure represents different $z _ { c }$ values. We obtain mean and variance parameters for five classes from the prior network, reparameterizing to derive five distinct $z _ { c }$ values. Meanwhile, each column represents different $z _ { e }$ , sampled directly from a standard normal distribution. From these generated images, it’s evident that $z _ { c }$ primarily influences image categories, while $z _ { e }$ affects style elements such as color and background, with less impact on the image’s content or category. The distinct roles of $z _ { c }$ and $z _ { e }$ validate our method’s effective content-style decoupling. This decoupling not only enhances the model’s robustness against style variations and domain shifts but also demonstrates potential for controlled image generation. Figure 2 (Middle and Right) is the t-SNE visualization for the latent space of style and content. The samples in the right figure exhibit a clear separation while those in the middle figure are completely mixed, which confirms that the content embeddings could effectively capture the class-specific features while the style embeddings successfully maintain class-irrelevant.

# 5 Conclusion and Discussion

In this paper, we investigate latent representation identifiability within the PLL paradigm and propose a novel framework, CausalPLL+, that addresses challenges in IDPLL classification, as well as domain shift and style variation problems that have plagued related algorithms. We introduce a novel prior network that enhances model interpretability without compromising performance, bridging the gap between identifiability theory and practical PLL applications. Furthermore, we bifurcated the latent embedding into two branches, explicitly decoupling content from style. This enhancement equips the model with greater robustness against style variations and domain shifts. Additionally, we proposed a contrastive learning approach under the PLL paradigm to effectively leverage the learned prior from the model. Lastly, we introduced a label refinement disambiguation strategy that reduces vagueness in supervision by progressively eliminating erroneous labels. This method is particularly effective when dealing with highly ambiguous candidate label sets. Extensive empirical studies confirm the effectiveness of the proposed method.

# Acknowledgments

The authors wish to thank the anonymous reviewers for their helpful comments and suggestions. This work was supported by the National Science Foundation of China (6509009676) and the Big Data Computing Center of Southeast University.