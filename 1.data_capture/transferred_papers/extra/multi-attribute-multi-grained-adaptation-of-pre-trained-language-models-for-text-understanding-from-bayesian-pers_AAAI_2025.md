# Multi-Attribute Multi-Grained Adaptation of Pre-Trained Language Models for Text Understanding from Bayesian Perspective

You Zhang1, Jin Wang1\*, Liang-Chih $\mathbf { Y } \mathbf { u } ^ { 2 * }$ , Dan ${ \bf X } { \bf u } ^ { 1 }$ , Xuejie Zhang1

1School of Information Science and Engineering, Yunnan University, Yunnan, P.R.China 2Department of Information Management, Yuan Ze University, Taiwan yzhang0202, wangjin $@$ ynu.edu.cn, lcyu $@$ saturn.yzu.edu.tw

# Abstract

Current neural networks often employ multi-domain-learning or attribute-injecting mechanisms to incorporate nonindependent and identically distributed (non-IID) information for text understanding tasks by capturing individual characteristics and the relationships among samples. However, the extent of the impact of non-IID information and how these methods affect pre-trained language models (PLMs) remains unclear. This study revisits the assumption that non-IID information enhances PLMs to achieve performance improvements from a Bayesian perspective, which unearths and integrates non-IID and IID features. Furthermore, we proposed a multi-attribute multi-grained framework for PLM adaptations (M2A), which combines multi-attribute and multi-grained views to mitigate uncertainty in a lightweight manner. We evaluate M2A through prevalent text-understanding datasets and demonstrate its superior performance, mainly when data are implicitly non-IID, and PLMs scale larger.

Code — https://github.com/yoyo-yun/M2A

# Introduction

Modern neural networks, especially pre-trained language models (PLMs), for specific tasks require large amounts of data and computational resources (Kong, Wang, and Zhang 2021; Han et al. 2021); therefore, collecting as much data as possible from diverse sources to achieve satisfactory task performance is desirable (Yuan, Zhao, and Qin 2022; Katsarou, Jeney, and Stefanidis 2023; Liu, Zhang, and Liu 2018; Yang and Hospedales 2015). For instance, sentiment analysis data can be gathered from different users, categories, and platforms, where corresponding golden labels are annotated by different users and intended to build sufficient samples leveraged for robust sentiment model training (Chen and Chao 2021; Zhang et al. 2023). These networks unanimously assume that all samples in a data pool (for a specific downstream task) are independent and identically distributed (IID) in a coarse-grained view (Pang and Lee 2006; Zhang, Wang, and Liu 2018). However, data distributions from different resources are not typically IID, which needs fine-grained view to deal with the data heterogeneity problem. Without the fine-grained view, the coarse-grained IID assumption may impede the optimal model convergences (uncertainty), degrading ideal performance when data are non-IID, either explicitly or implicitly, even though carrying out sufficient data (Zhang et al. 2023; Yao et al. 2024).

One intuitive solution to address the issue is to abstract the heterogeneities in data-sufficient while resource-diverse tasks and guide models adapted to diverse individual scenarios. This brings two significant challenges in recent adaptations of PLMs: 1) representing the heterogeneities among data samples and 2) making PLMs capable of recognizing them effectively and efficiently in the training and inference phase. Recently, many efforts have been committed to introducing multi-domain-learning (Yuan, Zhao, and Qin 2022; Katsarou, Jeney, and Stefanidis 2023), and attribute-injecting mechanisms (Zhang et al. 2021; Amplayo 2019) for investigating data heterogeneities. Multi-domain learning aims to learn domain-shared features and combine them with separate domain-specific features for multidomain text classifications. Attribute injecting treats concrete domain information as attribute knowledge and injects it into coarse neural networks for fine-grained performance. However, these high-performance methods do not systematically develop the model design and optimization approach, constraining the ability to represent robust nonIID and IID features as well as their internal relatedness. Moreover, traditional full-model fine-tuning (FFT) and sophisticated structure modification render limited scalability for larger PLMs (Min et al. 2024).

We argue that most text-understanding data contain multiple attributes, i.e., user, category, and platform. From a data distribution perspective, each attribute covers coarsegrained and fine-grained views, as shown in Figure 1(a). The coarse-grained view with one domain, which treats all samples as IID, ignores data heterogeneities. This approach is advantageous for accommodating large number of samples, as shown in Figure 1(b). The fine-grained view, e.g., the category, considers all samples divided into numerous fine-grained domains. Within each domain, samples are IID, whereas samples across different domains are non-IID. This perspective allows models to manage diverse samples by employing individual modules for each domain, as shown in Figure 1(c).

![](images/d89d8c1498e490984319628024f708c910032d9b7e7c4dc0fc7327bb0ca79c82.jpg)  
Figure 1: A conception of the proposed method.

To integrate IID and non-IID information for complementary benefits, we rethink the current multi-domain learning and attribute injecting mechanism and propose a multiattribute, multi-grained adaptation (M2A) framework for mitigating the uncertainty of PLMs’ adaptation, as shown in Figure 1(d). Theoretically, we adopt a Bayesian inference to analyze the relatedness between coarse-grained and fine-grained views and a Bayesian neural network (BNN) to represent robust data heterogeneities (Magris and Iosifidis 2023; Jospin et al. 2022) (see $\ S$ Preliminaries). Through a combination of multiple attributes and granularities from the Bayesian inference perspective, we found that the proposed M2A framework could facilitate PLMs to be effectively adapted to non-IID tasks. Moreover, a parameterefficient fine-tuning (PEFT) module (Houlsby et al. 2019) and a joint learning strategy are proposed to facilitate the scalability of M2A for diverse PLMs (Zhang et al. 2023). In experiments, we utilized various PLMs to evaluate M2A on multi-domain and personalized sentiment analysis tasks, which contain sufficient data for task adaptation while involving data heterogeneities.

Our key contributions in this paper are threefold.

• We rethink the current multi-domain-learning and attribute-injecting mechanism from a Bayesian perspective and provide a robust M2A framework for text understanding.   
• Based on PEFT and joint learning methods, M2A makes PLMs effectively and efficiently adapted for diverse scales and broader scenarios.   
• Extensive experiments were conducted on several text

understanding tasks and provided empirical analysis.

# Preliminaries Background: Text Understanding

Regarding a text understanding task with a training dataset $\bar { \mathcal { D } } \bar { = } \{ ( \mathbf { x } _ { n } ^ { \mathrm { ~ } } , y _ { n } ) \} _ { n } ^ { N }$ with $N$ IID and label-annotated samples, a neural network $\mathrm { N N } _ { w } ( \cdot )$ parameterized with weights $w$ is optimized to estimate the likelihood $p ( \mathcal { D } _ { y } | \mathcal { D } _ { x } ; w )$ that links the inputs and the outputs via:

$$
p ( \mathcal D _ { y } | \mathcal D _ { x } ; w ) = \mathop { \mathrm { a r g m a x } } _ { w } \prod _ { n = 1 } ^ { N } \mathrm { L o s s } ( y _ { n } , \hat { y } _ { n } )
$$

where $\mathcal { D } _ { x } = \{ \mathbf { x } _ { n } \} _ { n = 1 } ^ { N }$ and $\mathcal { D } _ { y } = \{ y _ { n } \} _ { n = 1 } ^ { N }$ N represents the input data and the output labels; $\hat { y } _ { n } = \mathrm { N N } _ { w } ( \mathbf x _ { n } )$ denotes the network output for input ${ \bf x } _ { n }$ ; Loss( ) the loss function for model optimizations.

Regarding non-IID heterogeneities in a specific attribute $\textit { a } \in \textit { s }$ , the dataset $\begin{array} { r c l } { { \mathcal D } } & { { = } } & { { \cup _ { a s \in a } \mathcal D ^ { ( a s ) } } } \end{array}$ can be divided into $| a | _ { f }$ fine-grained domains (sub-datasets) where each domain contains IID samples, denoted as $\begin{array} { r l } { \mathcal { D } ^ { ( a s ) } = } & { { } } \end{array}$ $\{ ( \mathbf { x } _ { n } ^ { ( a s ) } , y _ { n } ^ { ( a s ) } ) \} _ { n } ^ { | a s | }$ . For each domain $\boldsymbol { a s }$ , it typiDcally requires a neural network $\mathrm { N N } _ { w ^ { ( a s ) } } ( \cdot )$ to estimate the domainspecific sub-datasets. Notably, only one IID domain $a c$ is covered in coarse-grained view, where $\mathcal { D } = \mathcal { D } ^ { ( a c ) }$ , and the corresponding model is $\mathrm { N N } _ { w ^ { ( a c ) } } ( \cdot )$ .

# A Bayesian Perspective

Bayesian Inference. We propose a Bayesian perspective to investigate diversity and relatedness among multiple attributes and granularities, respectively. In Bayesian learning (de Freitas, Niranjan, and Gee 2000), $p ( w | \mathcal { D } )$ is the posterior distribution of the model being learned. Therefore, the predictive probabilities can be defined as:

![](images/20b994ab6a115752d299be29d84dbdb33d25a08d02a0dc45a9c44ff19109424e.jpg)  
Figure 2: Overview of the M2A Framework.

$$
p ( y | \mathbf { x } ; \mathcal { D } ) = \int p ( y | \mathbf { x } ; w ) p ( w | \mathcal { D } ) d _ { w }
$$

which can be approximated by Monte Carlo method (Kroese et al. 2014) and aggregated from $| S |$ models:

$$
p ( \boldsymbol { y } | \mathbf { x } ; \mathcal { D } ) \approx \frac { 1 } { | S | } \sum _ { s \in S } p ( \boldsymbol { y } | \mathbf { x } ; \boldsymbol { w } ^ { ( s ) } )
$$

where $\boldsymbol { w } ^ { ( s ) } \sim p ( \boldsymbol { w } ^ { ( s ) } | \mathcal { D } )$ .

Bayesian Neural Network. It targets the estimation of posterior distribution $p ( w | \mathcal { D } )$ based on a Bayesian theorem via (Jospin et al. 2022):

$$
\begin{array} { r l } & { \displaystyle p ( w | \mathcal { D } _ { x } , \mathcal { D } _ { y } ) = \frac { p ( \mathcal { D } _ { y } | \mathcal { D } _ { x } ; w ) \cdot p ( \mathcal { D } _ { x } | w ) \cdot p ( w ) } { p ( \mathcal { D } _ { x } , \mathcal { D } _ { y } ) } } \\ & { \quad \quad \propto p ( \mathcal { D } _ { y } | \mathcal { D } _ { x } ; w ) \cdot p ( \mathcal { D } _ { x } | w ) } \end{array}
$$

where $p ( \mathcal { D } _ { x } | w )$ is the likelihood of the model that predicts input data; $p ( w )$ is the prior distribution of the model; $p ( \mathcal { D } ) = p ( \mathcal { D } _ { x } , \mathcal { D } _ { y } )$ is the evidence.

To sum up, the question of our proposed method is: how to (1) integrate multiple attributes and granularities for robust text representations, (2) estimate $p ( w | \mathcal { D } )$ in each domain for capturing the non-IID and IID features, and (3) connect nonIID and IID information for modeling the internal relatedness.

# M2A Framework

Figure 2 illustrates the conceptual diagram of the proposed M2A method, which consists of three steps from a Bayesian perspective. The first step views each data from different attribute and granularity perspectives, utilizing M2A modules to integrate non-IID and IID features; the second step optimizes M2A modules through multitask learning; and the third step connects multi-grained views and introduces a joint learning strategy. Additionally, our framework can be decomposed to improve parameter efficiency.

# Multi-attribute and Multi-grained Module in Bayesian Inference

Based on Bayesian inference in Eq. (3), we introduce a multi-attribute and multi-grained module to represent and integrate non-IID and IID information for robust hidden representation.

Each data sample $( \mathbf { x } , y )$ can be viewed from different attributes $a \in { \mathcal { A } }$ and multi-grained views $g \in \mathcal { G } = \{ c , f \}$ and denoted by $\begin{array} { r } { ( \mathbf { x } , y ) \sim \bigcap _ { a \in \mathcal { A } , g \in \mathcal { G } } { \mathcal { D } } ^ { ( a g ) } } \end{array}$ , where $a f$ represents $( \mathbf { x } , y )$ in the fine-grained domain $f$ from $a$ -attribute perspective1 via the formulation of $a f = \mathcal { P } ( ( \mathbf { x } , y ) , a ) _ { f }$ . In contrast to Monte Carlo sampling, we sample $\boldsymbol { w } ^ { ( s ) }$ from diverse data distributions, i.e., $\mathcal { D } ^ { ( a g ) }$ with $a \in { \mathcal { A } }$ and $g \in { \mathcal { G } }$ ; therefore, the predictive probabilistic of $( \mathbf { x } , y )$ can be formulated as:

$$
p ( y | \mathbf { x } ; \mathcal { D } ) \approx \frac { 1 } { \left| \mathcal { A } \right| \left| \mathcal { G } \right| } \sum _ { a \in \mathcal { A } , g \in \mathcal { G } } p ( y | \mathbf { x } ; w ^ { ( a g ) } )
$$

where $w ^ { ( a g ) } \ \sim \ p ( w ^ { ( a g ) } | { \mathcal D } ^ { ( a g ) } ) p ( { \mathcal D } ^ { ( a g ) } | { \mathcal D } )$ . Since these ${ w ^ { ( a g ) } } \mathrm { s }$ have complementary and individual semantic information, $w ^ { ( a g ) } \mathrm { s }$ can be fused to a unified model in current models (Zhang et al. 2024):

$$
p ( \boldsymbol { y } | \mathbf { x } ; \mathcal { D } ) \approx p ( \boldsymbol { y } | \mathbf { x } ; \Omega ( \bigcup _ { a g } w ^ { ( a g ) } ) )
$$

where $\Omega ( \cdots )$ represents module integrating operations.

Inspired by PEFT methods that introduce a lightweight module to enable PLMs to be adapted to downstream tasks, we utilize the LoRA family to implement our M2A framework for multi-objective adaptations efficiently, facilitating further research when PLMs scale large $\mathrm { \Delta H u }$ et al. 2021).

Each module ${ w ^ { ( a g ) } }$ is formulated as $A ^ { ( a g ) } B ^ { ( a g ) }$ with lowrank instinct features of $r ^ { ( a g ) } \ll \operatorname* { m i n } ( d _ { i n } , d _ { o u t } )$ , which are then added into PLM matrix parameters $W \in \mathbb { R } ^ { \prime } d _ { i n } \times d _ { o u t }$ for model calibration:

$$
W = W + w ^ { ( a g ) }
$$

Due to the cumulative and plug-and-play characteristics of LoRA $\mathrm { \Delta W u }$ , Huang, and Wei 2023), we instantiate modules integrating operations as $\begin{array} { r } { \Omega ( \bigcup _ { a \in \mathcal { A } , g \in \mathcal { G } } w ^ { ( a g ) } ) \ = } \end{array}$ 1 Pag w(ag). After that, as shown iSn F∈igAure∈G2, our M2A module is simplified and denoted as:

$$
W = W + \frac { 1 } { \left| \boldsymbol { A } \right| \left| \mathcal { G } \right| } \sum _ { a \in \boldsymbol { A } , g \in \mathcal { G } } w ^ { ( a g ) }
$$

# Module Learning in BNN

The essential factor in optimizing the module $w$ locates how to maximize $p ( w | \mathcal { D } _ { x } , \mathcal { D } _ { y } )$ based on PLM backbones.

Although PLMs have provided general knowledge for text generation or cloze tasks, they could not effectively distinguish the data heterogeneities. For example, due to different pragmatic writing styles, PLMs make it hard to determine word prediction for different writers with the same given texts. Based on Eq. (4) with the Bayesian diagram, we introduce a multitask learning to estimate $p ( w | \mathcal { D } )$ via maximizing $p ( \mathcal { D } _ { y } | \mathcal { D } _ { x } ; w )$ and $p ( \mathcal { D } _ { x } | w )$ , simultaneously:

$$
\mathcal { L } _ { \mathrm { m t l } } ^ { \mathrm { N N } _ { w } } = \mathrm { L o s s } ( y _ { n } , \hat { y } _ { n } ) + \alpha \mathrm { L o s s } ( \mathbf { x } _ { n } , \hat { \mathbf { x } } _ { n } )
$$

where $\alpha$ is the balance factor; $\hat { \mathbf { x } } _ { n }$ denote reconstructed text tokens under $p ( \mathcal { D } _ { x } | w )$ estimation and $\mathrm { N N } _ { w } ( \cdot )$ . Empirically, from the writer-specific perspective (i.e., $a$ denotes writer attribute), $p ( \mathcal { D } _ { y } | \mathcal { D } _ { x } ; w ^ { ( a s ) } )$ aims to facilitate PLMs to learn how a specific writer (i.e., s) would influence the annotations of the texts. Meanwhile, $p ( \mathcal { D } _ { x } | w ^ { ( \bar { a } s ) } )$ is introduced to optimize PLMs to learn What pragmatic contents a specific writer $( i . e . , s )$ would generate.

To implement $p ( \mathcal { D } _ { x } | w )$ based on masked language modeling (MLM) (Devlin et al. 2019) and autoregressive modeling (ARM) (Radford et al. 2019), respectively, we utilized different strategies. For MLM, we randomly mask $1 5 \%$ of input tokens as [MASK] and get unified models to predict original tokens of texts and annotated labels based on [CLS] or $< _ { S } >$ tokens. For ARM-based PLMs, ARM tasks are used to predict both shift-right text labels and downstream task labels.

# Connecting Multi-grained Views

Acknowledging that identifying exact non-IID information locations could facilitate neural network models to mitigate predictive uncertainty, it is difficult to collect the locations due to extensive labor in retrieving sensitive information and bypassing privacy concerns (Zhang et al. 2023). To further investigate internal relatedness among coarse-grained and fine-grained views, Bayesian inference in Eq. (3) guides the M2A framework with a joint learning strategy for improving the model performance and broadening practical scenarios. A more detailed analysis can be found in the technical appendices. To generalize fine-grained domains, we can get a multi-attribute coarse-grained M2A module (dubbed as $\mathbf { M } 2 \mathbf { A } \dagger .$ ), denoted by $\begin{array} { r } { W + \Omega ( \bigcup _ { a \in \mathcal { A } , g \in \{ c , c ^ { \prime } \} } w ^ { ( a g ) } ) } \end{array}$ , where $w ^ { ( a c ^ { \prime } ) }$ is a special coarse-view module used to align diverse fine-grained modules.

To align with M2A and $\mathbf { M } 2 \mathbf { A } \dagger$ , we resort to a joint learning strategy with knowledge distillation for final optimizations (Aguilar et al. 2020), formally:

$$
\mathcal { L } = \prod _ { ( \mathbf { x } , y ) \in \mathcal { D } } \mathcal { L } _ { \mathrm { m t l } } ^ { \mathrm { N N } } + \mathcal { L } _ { \mathrm { m t l } } ^ { \mathrm { N N } ^ { \dag } } + \mathrm { K L } ( \mathrm { N N } ( \mathbf { x } ) , \mathrm { N N } ^ { \dag } ( \mathbf { x } ) )
$$

$\begin{array} { r } {  { \mathrm { N N } } \ = \  { \mathrm { N N } } _ { W + \Omega ( \bigcup _ { a \in \mathcal { A } , g \in \{ c , f \} } w ^ { ( a c ) } ) } } \end{array}$ $\begin{array} { r l } { \mathrm { N N ^ { \dagger } } } & { { } = } \end{array}$ $\mathrm { N N } _ { W + \Omega ( \bigcup _ { a \in \pmb { A } , g \in \{ c , c ^ { \prime } \} } \ w ^ { ( a g ) } ) }$ . During the training phase, only parameters of $\textstyle \bigcup _ { a \in { \mathcal { A } } , g \in \{ c , c ^ { \prime } , f \} } w ^ { ( a g ) }$ are simultaneously updated. Mor ver, we have found that NN converged faster than $\mathrm { N N } \dagger$ (see $\ S$ Experiments); therefore, we introduce a module separation strategy. That means when the learning procedure of NN is optimally stopped during the training phase, $\mathrm { N N } \dagger$ will continue learning with only lightweight Sa∈A,g∈{c′} w(ag) to being updated until its optima convergence.

# Model Decomposition

Due to uncertainties in model capacity, the number of domains employed, and module redundancy, we further improve the M2A framework in parameter efficiency via model decomposition (see Figure 2).

Fine-grained domain parameters reduction. Regarding model capacity, the fine-grained module $( w ^ { ( a f ) } )$ covers much fewer data samples than the coarse-grained module $( w ^ { ( a c ) }$ or $w ^ { ( a c ^ { \prime } ) } )$ ; thus, we introduce a KronA module (Edalati et al. 2022), which only contains one-rank weight parameters and provides feasible decompositions, to replace the original LoRA for fine-grained modules, i.e., $w ^ { \hat { ( } a f ) } = C ^ { ( a f ) } \otimes D ^ { ( a f ) }$ where $\otimes$ is Kronecker product, $C ^ { ( a f ) } \ \in \ \mathbb { R } ^ { ( d _ { i n } / r ^ { \prime } ) \times r ^ { \prime } }$ , and $D ^ { ( a f ) } \ \in \ \mathbb { R } ^ { r ^ { \prime } \times ( d _ { o u t } / \hat { r } ^ { \prime } ) }$ with $1 \leq r ^ { \prime } \leq \operatorname* { m i n } ( d _ { i n } , d _ { o u t } )$ .

Sharing attribute information among fine-grained domains. From a fine-grained perspective, global data can be divided into many domains, which linearly scale model parameters and memory budgets. Here, we share one $C ^ { ( a ) }$ with all $\textstyle \bigcup _ { a s \in a } C ^ { ( a s ) }$ .

Sharing coarse-grained attribute information across attributes. Across multiple attributes, the coarse-grained views cover the same global dataset, i.e., $\forall _ { a \in \mathcal { A } } \mathcal { D } ^ { ( a c ) } = \mathcal { D }$ . To reduce redundancy information, we share one ${ w ^ { ( c ) } }$ with all $\textstyle \bigcup _ { a \in { \mathcal { A } } } w ^ { ( a c ) }$ .

EffiScie∈nAcy analysis. Without mode decomposition, the external complexity (ignoring $W$ in PLMs) is $\begin{array} { r } { \sum _ { a } \left( \left| a \right| _ { f } + 2 \right) . } \end{array}$ $r \cdot \left( d _ { i n } + d _ { o u t } \right)$ , which is larger than that of our decomposition version with $\begin{array} { r } { \left( \sum _ { a } \left( d _ { i n } + \left| \dot { a } \right| _ { f } \cdot d _ { o u t } \right) \right) + 2 \cdot r \cdot \left( d _ { i n } \dot { + } d _ { o u t } \right) } \end{array}$ .

# Experiments Datasets and Evaluations

We evaluate the proposed M2A frameworks on two types of prevalent datasets. (1) Multi-domain sentiment analysis includes FDU-MTL, which contains product and movie reviews across 16 domains from the category-attribute perspective (Liu, Qiu, and Huang 2017). (2) Personalized sentiment analysis includes IMDB, Yelp-2013, and Yelp2014. These datasets exhibit data heterogeneities from user and item-attribute perspectives (Tang, Qin, and Liu 2015). Regarding annotated labels, FDU-MTL, IMDB, and Yelps are treated as a 2-class, 10-class, and 5-class classification problems, respectively.

Table 1: Comparative test Acc $( \% )$ results for multi-domain sentiment analysis (the category attribute only).   

<html><body><table><tr><td>Domains</td><td>BERT</td><td>DAEA</td><td>BERTMasker</td><td>KCL-KB</td><td>B-MTL</td><td>B-M2A$</td><td>G-M2A‡</td><td>R-M2A$</td></tr><tr><td>Books</td><td>87.00</td><td>89.00</td><td>93.00</td><td>93.08</td><td>94.75</td><td>94.75</td><td>93.50</td><td>97.25</td></tr><tr><td>Electronics</td><td>88.30</td><td>91.80</td><td>93.25</td><td>94.92</td><td>94.00</td><td>95.50</td><td>93.75</td><td>96.00</td></tr><tr><td>DVD</td><td>85.60</td><td>88.30</td><td>89.25</td><td>89.92</td><td>90.75</td><td>93.25</td><td>91.50</td><td>93.50</td></tr><tr><td>Kitchen</td><td>91.00</td><td>90.30</td><td>90.75</td><td>92.50</td><td>92.00</td><td>93.25</td><td>95.25</td><td>96.00</td></tr><tr><td>Apparel</td><td>90.00</td><td>89.00</td><td>92.25</td><td>92.67</td><td>91.25</td><td>92.50</td><td>93.25</td><td>94.50</td></tr><tr><td>Camera</td><td>90.00</td><td>92.00</td><td>92.75</td><td>93.67</td><td>94.75</td><td>95.00</td><td>93.00</td><td>95.25</td></tr><tr><td>Health</td><td>88.30</td><td>89.80</td><td>95.25</td><td>95.67</td><td>94.25</td><td>96.50</td><td>95.50</td><td>97.50</td></tr><tr><td>Music</td><td>86.80</td><td>88.00</td><td>89.50</td><td>90.42</td><td>90.75</td><td>91.75</td><td>92.00</td><td>93.75</td></tr><tr><td>Toys</td><td>91.30</td><td>91.80</td><td>93.75</td><td>93.33</td><td>92.75</td><td>93.50</td><td>92.25</td><td>94.25</td></tr><tr><td>Video</td><td>88.00</td><td>92.30</td><td>91.25</td><td>91.67</td><td>92.00</td><td>94.50</td><td>91.75</td><td>94.00</td></tr><tr><td>Baby</td><td>91.50</td><td>92.30</td><td>92.75</td><td>94.58</td><td>95.75</td><td>96.25</td><td>94.00</td><td>96.25</td></tr><tr><td>Magazines</td><td>92.80</td><td>96.50</td><td>94.50</td><td>94.17</td><td>94.25</td><td>94.75</td><td>94.00</td><td>96.50</td></tr><tr><td>Software</td><td>89.30</td><td>92.80</td><td>93.00</td><td>94.33</td><td>95.75</td><td>96.50</td><td>96.25</td><td>96.50</td></tr><tr><td>Sports</td><td>90.80</td><td>90.80</td><td>92.50</td><td>94.42</td><td>94.25</td><td>95.00</td><td>94.50</td><td>94.75</td></tr><tr><td>IMDB</td><td>85.80</td><td>90.80</td><td>86.00</td><td>90.83</td><td>93.00</td><td>93.25</td><td>92.25</td><td>94.50</td></tr><tr><td>MR</td><td>74.80</td><td>77.00</td><td>83.75</td><td>85.58</td><td>84.25</td><td>85.75</td><td>84.00</td><td>85.50</td></tr><tr><td>Avg.</td><td>88.16</td><td>90.16</td><td>91.47</td><td>92.62</td><td>92.78</td><td>93.86</td><td>92.92</td><td>94.70</td></tr></table></body></html>

Table 2: Comparative test results for three personalized sentiment classification datasets (user and item attributes). The underline and backbone scores respectively meant the best scores in baselines and all models in each group. Our M2A models outperformed the previous works significantly $\scriptstyle \phantom { + } ( p < 0 . 0 5 )$ in Acc.   

<html><body><table><tr><td rowspan="2">Models</td><td colspan="3">IMDB</td><td colspan="3">Yelp-2013</td><td colspan="3">Yelp-2014</td></tr><tr><td>Acc</td><td>RMSE</td><td>F1</td><td>Acc</td><td>RMSE</td><td>F1</td><td>Acc</td><td>RMSE</td><td>F1</td></tr><tr><td colspan="10">non-IID free or Coarse-grained view (IID) only</td></tr><tr><td>BERT</td><td>52.2</td><td>1.163</td><td>49.3</td><td>67.7</td><td>0.628</td><td>65.5</td><td>67.7</td><td>0.615</td><td>65.6</td></tr><tr><td>GPT2</td><td>51.5</td><td>1.222</td><td>47.5</td><td>67.6</td><td>0.622</td><td>64.5</td><td>68.0</td><td>0.614</td><td>65.5</td></tr><tr><td>RoBERTa</td><td>53.0</td><td>1.147</td><td>49.4</td><td>69.2</td><td>0.590</td><td>65.1</td><td>69.0</td><td>0.601</td><td>66.5</td></tr><tr><td>R-M2A†</td><td>54.2</td><td>1.100</td><td>50.3</td><td>70.7</td><td>0.574</td><td>69.4</td><td>70.5</td><td>0.578</td><td>68.5</td></tr><tr><td colspan="10">Multiple granularities (IID+non-IID)</td></tr><tr><td>B-IUPC</td><td>53.8</td><td>1.151</td><td></td><td>70.5</td><td>0.589</td><td>1</td><td>71.2</td><td>0.592</td><td></td></tr><tr><td>B-MAA</td><td>57.3</td><td>1.042</td><td></td><td>70.3</td><td>0.588</td><td>-</td><td>71.4</td><td>0.573</td><td>-</td></tr><tr><td>B-GS</td><td>57.2</td><td>1.042</td><td>54.5</td><td>70.2</td><td>0.593</td><td>68.3</td><td>71.1</td><td>0.585</td><td>68.5</td></tr><tr><td>R-GNNLM</td><td>54.4</td><td>1.102</td><td>1</td><td>72.2</td><td>0.573</td><td>-</td><td>72.1</td><td>0.568</td><td></td></tr><tr><td>B-M2A$</td><td>58.7</td><td>1.021</td><td>54.8</td><td>72.0</td><td>0.569</td><td>69.7</td><td>72.4</td><td>0.560</td><td>69.1</td></tr><tr><td>G-M2A$</td><td>58.1</td><td>1.074</td><td>53.6</td><td>70.4</td><td>0.598</td><td>67.4</td><td>71.8</td><td>0.569</td><td>68.4</td></tr><tr><td>R-M2A$</td><td>60.3</td><td>0.954</td><td>56.7</td><td>73.7</td><td>0.548</td><td>70.7</td><td>74.2</td><td>0.535</td><td>71.4</td></tr><tr><td>R-M2A</td><td>60.6</td><td>0.960</td><td>56.6</td><td>73.4</td><td>0.543</td><td>70.2</td><td>73.6</td><td>0.545</td><td>70.9</td></tr></table></body></html>

For evaluation metrics, we adopt Accuracy (Acc) to measure the effectiveness of models for multi-domain sentiment analysis. For personalized sentiment analysis, we used Acc (primarily), Rooted-Mean Square Error (RMSE), and Macro-F1 (F1) scores (Yuan, Zhao, and Qin 2022; Zhang et al. 2023).

# Experimental Settings

We compared our M2A framework with previous state-ofthe-art models across different datasets.

(1) For multi-domain datasets, the baselines include BERT (Devlin et al. 2019), DAEA (Cai and Wan 2019), BERTMaker (Yuan, Zhao, and Qin 2022), KCL-KB (Yuan et al. 2023), and MTL (Thrun 1995).

(2) For personalized datasets, the baselines include variants of PLMs in a coarse-grained view, as well as IPUC (Lyu, Foster, and Graham 2020), MAA (Zhang et al. 2021), GS (Zhang et al. 2023), and GNNLM (Kertkeidkachorn and Shirai 2023) from multi-grained perspectives.

To investigate the effect of our M2A, we introduced diverse PLMs as backbones, including BERT (B) (Devlin et al. 2019), RoBERTa (R) (Liu et al. 2019), and GPT2 (G) (Radford et al. 2019). We adopted FFT and PEFT for optimization, denoted as $\mathbf { M } 2 \mathbf { A } \ddagger$ and M2A, respectively. Here, $\mathbf { M } 2 \mathbf { A } \ddagger$ provided a fair comparison with previous works also using FFT. $\mathbf { M } 2 \mathbf { A } \dagger$ represented the coarse-grained version jointed learned with M2A, handling non-IID-free scenarios, as discussed in $\ S \mathbf { M } 2 \mathbf { A }$ Framework. All results were averaged over five runs. Detailed model configurations of hyperparameters can be found in the technical appendices.

# Comparative Results and Analysis

Tables 1 and 2 reported the comparative results of the proposed M2A framework against previous works on multidomain and personalized sentiment analysis datasets.

From Table 1, when individually trained for each domain, BERT achieved relatively the worst Acc scores. This phenomenon suggested that although PLMs could gain high performance in sentiment analysis, they continued encountering challenges due to data scarcity within each domain. When all domain datasets were gathered, multi-domain models achieved higher performance than BERT, indicating that more extensive and aggregated data could facilitate datahungry neural networks to achieve generalized performance. B-MTL, which utilized multitask learning to handle all domains, achieved competitive results. However, it treated all samples as IID, disregarding data heterogeneities, which led to inferior performance compared to our B-M2A. The proposed M2A, from a Bayesian perspective, outperformed other baselines by ensemble multiple views and encoding robust domain features. Moreover, the results showed that PLMs using the same transformer structure exhibited varying performance with M2A, emphasizing the importance of selecting appropriate prior distributions $\bar { ( \boldsymbol { p } ( \boldsymbol { w } ) }$ and $p ( \mathbf { x } | w ) )$ . The proposed M2A framework, which integrates both IID and non-IID features, consistently outperformed other models by effectively addressing data heterogeneities and leveraging domain-specific information.

As shown in Table 2, models integrating multiple granularities (from user and item-attribute perspectives) outperformed those considering only coarse-grained views. This demonstrated the importance of leveraging data heterogeneities collected from diverse sources during the downstream task adaptation of PLMs. Compared with GNNLM, which introduced a graph neural work to encode robust attribute or domain representations for improved performance, our M2A provided a multitask learning method based on BNN for the same purpose, resulting in competitive performance across all three datasets.

Furthermore, we compared a general version of M2A (R$\mathbf { M } 2 \mathbf { A } \dagger ,$ ) against RoBERTa. The superior performance validated that our Bayesian learning approach, combined with joint learning, endowed M2A with a generalized capability to handle scenarios with unclear data heterogeneities. Consequently, $\mathbf { R - M } 2 \mathbf { A } \ddagger$ obtained the best results on all three datasets across all three metrics, suggesting the advantage of our Bayesian learning-based M2A framework, especially as the backbone PLMs became more powerful.

# Ablation Study

The test results of the ablation study on both multi-domain and personalized sentiment analysis are reported in Table 3. We used BERT and GPT-2 as backbones for these investigations.

First, we gradually eliminated attribute representations from coarse to fine-grained views. As different granularities were removed, the performance of both $\mathbf { B } \mathbf { - } \mathbf { M } 2 \mathbf { A } \ddagger$ and $\mathbf { G - M } 2 \mathbf { A } \ddagger$ declined to different extents. This indicated integrating multi-grained information enhanced final performance by mitigating the uncertainty of models. In personalized scenarios, eliminating user non-IID information had a more significant impact than item non-IID information, consistent with previous findings (Wu et al. 2018). This was because user preferences tend to create more distinct non-IID distributions among texts than item characteristics, revealing the diversity among different attributes.

![](images/38f59946fdc8062537fa8a3bfa57bc2276793d5701056b3225cd52c9b1dbc7f7.jpg)  
Figure 3: Dev Acc of R-M2A with different multi-task.

Next, we removed text generation tasks in the module learning strategy to investigate the effect of the prior likelihood estimator $p ( \mathbf { x } | \boldsymbol { w } )$ . The results for both $\mathbf { B - M } 2 \mathbf { A } \ddagger$ and $\mathbf { G - M } 2 \mathbf { A } \ddagger$ decreased with the removal of $p ( \mathbf { x } | \boldsymbol { w } )$ , indicating its significance. To explore the improvements of $\mathbf { B } \mathbf { - } \mathbf { M } 2 \mathbf { A } \ddagger$ derived from $p ( \mathbf { x } | \boldsymbol { w } )$ or $1 5 \%$ tokens masking, we removed $p ( \mathbf { x } | \boldsymbol { w } )$ while prevising the MLM targets and found that B${ \bf M } 2 { \bf A } \ddagger$ without $p ( \mathbf { x } | \boldsymbol { w } )$ was worse than the above settings. This phenomenon further demonstrated the effect of $p ( \mathbf { x } | \boldsymbol { w } )$ through BNN in Eq. (4).

# The Effect of Bayesian Learning

An extensive experimental analysis was conducted to investigate how Bayesian learning theoretically facilitated our framework for text understanding.

Mixture of granularities. We initially employed different LoRA methods to evaluate the Bayesian learning-based mixture of multiple views. From Table 4, it was evident that the ascendancy of LoRA over KronA had a more pronounced effect on the final performance of R-M2A in coarse views than in fine-grained views. Consequently, LoRA was adopted to sample coarse models. Since KronA had much fewer parameters than LoRA and fine-grained views typically encompass many domains, we utilized KronA to instantiate domain modules in fine-grained views.

Text generation tasks. We varied the balance factor $\alpha$ to explore the impact of text generation tasks in BNN assumptions. As shown in Figure 3, with the introduction of text generation tasks for optimizations, the final dev performance was improved, demonstrating the effect of text generation through Bayesian learning. The performance varied with different values of $\alpha$ ; models with either smaller or larger values $\alpha$ exhibited relatively lower Acc compared to those with an appropriate selection. This finding suggests caution when

Table 3: The ablative test Acc on multi-domain and personalized sentiment analysis in terms of the mixture of multiple attribute and granularities (guided by Bayesian Inference) and module learning strategies (guided by BNN), respectively.   

<html><body><table><tr><td colspan="2">Models</td><td>FDU-MTL</td><td>IMDB</td><td>Yelp-2013</td><td>Yelp-2014</td></tr><tr><td colspan="2">B-M2A$</td><td>93.86</td><td>58.70</td><td>72.00</td><td>72.44</td></tr><tr><td rowspan="4">M2 Mixture</td><td>- coarse view</td><td>90.98</td><td>1</td><td>1</td><td>-</td></tr><tr><td>- fine-grained view (user)</td><td>1</td><td>52.80</td><td>68.57</td><td>68.85</td></tr><tr><td>- fine-grained view (item)</td><td></td><td>58.43</td><td>71.42</td><td>71.58</td></tr><tr><td>- fine-grained view (all)</td><td>92.78</td><td>52.21</td><td>67.65</td><td>67.65</td></tr><tr><td rowspan="2">Module Learning</td><td>- text generation task - text generation task</td><td>92.70</td><td>57.42</td><td>69.36</td><td>70.29</td></tr><tr><td>(except randomlymasking)</td><td>93.56</td><td>57.41</td><td>69.80</td><td>71.02</td></tr><tr><td></td><td>G-M2A$</td><td>92.92</td><td>58.10</td><td>70.41</td><td>71.79</td></tr><tr><td rowspan="4">M2 Mixture</td><td>- coarse view</td><td>91.84</td><td></td><td></td><td>1</td></tr><tr><td>- fine-grained view (user)</td><td>二</td><td>52.35</td><td>68.04</td><td>68.41</td></tr><tr><td>- fine-grainedview (item)</td><td>1</td><td>57.25</td><td>70.35</td><td>71.49</td></tr><tr><td>- fine-grained view (all)</td><td>92.86</td><td>51.46</td><td>67.63</td><td>67.97</td></tr><tr><td>Module Learning</td><td>- text generation task</td><td>92.87</td><td>56.43</td><td>70.18</td><td>70.99</td></tr></table></body></html>

<html><body><table><tr><td colspan="2">Methods</td><td rowspan="2">IMDB</td><td rowspan="2">Yelp-2013</td><td rowspan="2">Yelp-2014</td></tr><tr><td>CGV</td><td>FGV</td></tr><tr><td rowspan="2">FFT</td><td></td><td></td><td></td><td></td></tr><tr><td>KrRA</td><td>00M</td><td>020M</td><td>00M</td></tr><tr><td>LoRA</td><td rowspan="2">KronA</td><td>59.35</td><td>71.18</td><td>72.22</td></tr><tr><td>KronA</td><td>57.63</td><td>68.03</td><td>67.38</td></tr></table></body></html>

Table 4: Dev Acc of R-M2A with different PEFTs for coarse $^ +$ fine-grained views (denoted by CGV and FGV). LoRA has a low rank of 64, and KronA estimates a low-rank matrix using rank-one parameters. OOM represents out-of-memory in our settings due to large external parameters.

Table 5: Test Acc of $\mathbf { B } \mathbf { - } \mathbf { M } 2 \mathbf { A } \ddagger$ performing different text generalization tasks.   

<html><body><table><tr><td>B-M2A$</td><td>FDU-MTL</td></tr><tr><td>w/o text generation</td><td>92.70</td></tr><tr><td>w/labeled data</td><td>93.86</td></tr><tr><td>w/unlabeled data</td><td>93.65</td></tr></table></body></html>

configuring the application in practice.

To further investigate the impact of text generation tasks, we introduced unlabeled data in each fine-grained domain to facilitate models to learn domain modules. As shown in Table 5, $\mathbf { B - M } 2 \mathbf { A } \ddagger$ w/o text generation tasks represented models optimized with $p ( \boldsymbol { y } | \mathbf { x } , \boldsymbol { w } )$ on labeled data, and B$\mathbf { M } 2 \mathbf { A } \ddagger$ w/ unlabeled data represented models optimized with $p ( y | \mathbf { x } , w )$ on labeled data and $p ( \mathbf { x } | \boldsymbol { w } )$ on unlabeled data. B${ \bf M } 2 { \bf A } \ddagger$ w/ unlabeled data model achieved better results than $\mathbf { B } \mathbf { - } \mathbf { M } 2 \mathbf { A } \ddagger$ w/o generation tasks, indicating that unlabeled domain-specific data could also leverage domain-specific (non-IID) knowledge for fine-grained adaptations.

Connection of granularities. We compared our joint learning strategy with other ensemble methods for integrating general view representation. From Table 6, $\mathbf { R - M } 2 \mathbf { A } \dagger$ achieved the best performance among all methods, demonstrating the effect of our proposed joint learning strategy guided by the Bayesian inference assumption. With the removal of KL or coarse-grained losses $\mathcal { L } _ { \mathrm { m t l } } ^ { \mathrm { N N ^ { \dagger } } }$ , the performance of $\mathbf { R - M } 2 \mathbf { A } \dagger$ degraded, revealing these losses could facilitate our framework to connect granularities. Moreover, $\mathbf { R - M } 2 \mathbf { A } \dagger$ outperformed $\mathbf { R - M } 2 \mathbf { A } \dagger$ w/o Sep, validating that $w ^ { ( a c ^ { \prime } ) }$ was required to continue being updated for a better generalization performance for IID-only scenarios.

Table 6: Test Acc of R-M2A with strategies of connecting granularities for IID-only scenarios. Avg means averaging all fine-grained modules for a general one. Rand randomly selected a fine-grained module for input data. Sep represents separation operation.   

<html><body><table><tr><td>Connections</td><td>IMDB</td><td>Yelp-2013</td><td>Yelp-2014</td></tr><tr><td>RoBERTa</td><td>53.0</td><td>69.2</td><td>69.0</td></tr><tr><td>R-M2A w/ Avg</td><td>52.2</td><td>70.3</td><td>69.9</td></tr><tr><td>R-M2Aw/Rand</td><td>47.3</td><td>66.7</td><td>66.2</td></tr><tr><td>R-M2A†</td><td>54.2</td><td>70.7</td><td>70.5</td></tr><tr><td>R-M2Atw/o Sep</td><td>52.8</td><td>69.9</td><td>70.0</td></tr><tr><td>R-M2Atw/o KL</td><td>53.8</td><td>70.5</td><td>70.4</td></tr><tr><td>R-M2At w/o CNN</td><td>53.8</td><td>70.3</td><td>70.1</td></tr></table></body></html>

# Conclusions

This study proposes an M2A framework to extract data heterogeneities from multi-source data for fine-grained adaptation. Our approach introduced a Bayesian analysis to rethink previous multi-domain-learning and attribute-injecting methods and provided a PEFT and joint learning strategy for facilitating PLMs’ adaptations. Experimental findings from multi-domain and personalized sentiment analysis showed that the proposed method could integrate models sampled from multiple attributes and granularities to eliminate data uncertainty. Moreover, the proposed method utilized BNN paradigms to leverage domain modules for performance improvements.

Future works intend to collect a multi-view dataset that contains more kinds of sources for further analysis and build an automatic data heterogeneity detector to verify the effectiveness of our methods.