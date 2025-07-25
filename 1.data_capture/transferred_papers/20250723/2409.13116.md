# BGDB: Bernoulli-Gaussian Decision Block with Improved Denoising Diffusion Probabilistic Models

Chengkun $\mathbf { S u n } ^ { 1 }$ , Jinqian $\mathbf { P a n } ^ { 1 }$ , Russell Stevens Terry2, Jiang Bian1, Jie $\mathbf { X } \mathbf { u } ^ { 1 }$

1Department of Health Outcomes and Biomedical Informatics, University of Florida, Gainesville, FL 32611, USA 2Department of Urology, University of Florida, Gainesville, FL 32611, USA sun.chengkun,jinqianpan,bianjiang,xujie $@$ ufl.edu, russell.terry $@$ urology.ufl.edu

# Abstract

Generative models can enhance discriminative classifiers by constructing complex feature spaces, thereby improving performance on intricate datasets. Conventional methods typically augment datasets with more detailed feature representations or increase dimensionality to make nonlinear data linearly separable. Utilizing a generative model solely for feature space processing falls short of unlocking its full potential within a classifier and typically lacks a solid theoretical foundation. We base our approach on a novel hypothesis: the probability information (logit) derived from a single model training can be used to generate the equivalent of multiple training sessions. Leveraging the Central Limit Theorem (CLT), this synthesized probability information is anticipated to converge toward the true probability more accurately. To achieve this goal, we propose the Bernoulli-Gaussian Decision Block (BGDB), a novel module inspired by the CLT and the concept that the mean of multiple Bernoulli trials approximates the probability of success in a single trial. Specifically, we utilize Improved Denoising Diffusion Probabilistic Models (IDDPM) to model the probability of Bernoulli Trials. Our approach shifts the focus from reconstructing features to reconstructing logits, transforming the logit from a single iteration into logits analogous to those from multiple experiments. We provide the theoretical foundations of our approach through mathematical analysis and validate its effectiveness through experimental evaluation using various datasets for multiple imaging tasks, including both classification and segmentation.

# Code — https://github.com/sunck1/BGDB

# Introduction

Classifiers are fundamental tools in machine learning, responsible for discerning intricate relationships between predictors and responses to allocate new observations into predetermined classes (Rubinstein, Hastie et al. 1997). Among them, discriminative classifiers have gained prominence for their efficiency. Discriminative classifiers directly learn the conditional probability $P ( y | x )$ , selecting the label $y$ with the highest likelihood given an input $x$ $\mathrm { N g }$ and Jordan 2001). This direct approach bypasses the need to model the joint probability distribution $P ( x , y )$ , as generative classifiers do, leading to faster decision-making (Raina et al. 2003). Consequently, discriminative classifiers, particularly within convolutional neural networks (CNNs), have become the preferred choice for tasks such as image processing (Krizhevsky, Sutskever, and Hinton 2012; Miao et al. 2019, 2018).

Despite their widespread use and efficiency, discriminative classifiers face challenges in extracting features and defining metric relations between examples, especially with complex data types such as medical images (Jaakkola and Haussler 1998). This limitation stems from their focus on learning the decision boundary rather than understanding the underlying data distribution. In contrast, generative models offer a promising solution by constructing more intricate feature spaces and providing a sophisticated framework for understanding the data generation process (Perina et al. 2012). By creating structured hierarchies of latent variables linked through conditional distributions, generative models can establish nuanced correspondences between model components and observed features, enabling them to handle missing, unlabeled, and variable-length data effectively (Perina et al. 2012). Techniques such as Fisher’s method exemplify this approach, where original data is mapped into a low-dimensional feature space and then projected into a higher-dimensional space by kernel techniques for linear classification (Jaakkola and Haussler 1998). Another strategy involves augmenting data with generative models to improve feature representations, as seen in methods like Dataset Diffusion, which enhances the accuracy of segmentation and classification tasks (Nguyen et al. 2024). However, the direct integration of generative models into feature construction in discriminative classifiers often lacks a robust theoretical foundation. In such cases, the generative model typically generates an unknown latent space from another unknown latent space, making the generation process inherently difficult to interpret.

In this paper, we propose a new hypothesis that the probability distribution obtained by a single training process can be used to generate the probability distribution for multiple training processes. Ideally, this generated distribution would represent the true classification probability distribution. Specifically, compared to other generative models such as GANs (Goodfellow et al. 2014), which produce data through the adversarial process between the generator and the discriminator, diffusion models (Jarzynski 1997) have the advantage of generating one distribution from another and provide a mathematical foundation for this process. On the other hand, leveraging the distributions from a single training process, we can generate the probability distributions for multiple training iterations. According to the Central Limit Theorem, these generated distributions will more precisely approximate the true classification probabilities. This methodology thus enhances the model’s classification performance through supervised learning. Building on this idea, we incorporated the diffusion model into the discriminative classifier, developing a Bernoulli-Gaussian Decision Block (BGDB) designed to enhance the deep learning model. Our contributions can be summarized as follows:

• We introduce the Bernoulli-Gaussian Decision Block, which enhances the stability and performance of discriminative classifiers by leveraging the mean of logits from multiple experiments to supervise a single learning process.   
• We employ IDDPM to construct and refine the probability distributions of Bernoulli Trials, improving inference accuracy without adding computational complexity during inference.   
• We provide a theoretical analysis and validate the effectiveness of our approach through extensive experiments on multiple datasets, including Cityscapes, ISIC, and Pascal VOC, demonstrating notable improvements in classification and segmentation tasks.

# Related Work

# Central Limit Theorem in Neural Networks

Learning conditional and marginal probabilities from a dataset is fundamental to constructing machine learning methods, such as belief networks (Davidson and Aminian 2004). Leveraging the Central Limit Theorem (CLT) could enhance this process by providing a robust statistical foundation (Davidson and Aminian 2004). According to the CLT, the sum of a large number of random variables approximates a Gaussian distribution. This principle also applies to neural networks, where the pre-activations of each layer tend to be Gaussian (Huang et al. 2021). As the network width increases towards infinity, the output distribution of each neuron converges to a Gaussian distribution (Zhang, Wang, and Fan 2022). Thus, optimization in neural networks can be framed as optimizing a Gaussian process (Lee et al. 2017).

Many neural network optimization techniques are developed based on the CLT. For instance, from a width-depth symmetry perspective, shortcut networks demonstrate that increasing the depth of a neural network also results in a Gaussian process manifestation (Zhang, Wang, and Fan 2022). In the Empirical Risk Minimization (ERM) framework, the long-term deviation, scaled by the CLT, is governed by a Monte Carlo resampling error, providing widthasymptotic guarantees independent of data dimension (Chen et al. 2020). Self-Normalizing Neural Networks utilize the CLT to approximate network inputs with a Gaussian distribution, enabling robust learning and introducing novel regularization schemes (Klambauer et al. 2017). Despite these advancements, existing methods primarily rely on the CLT’s mathematical properties for parameter estimation rather than directly modeling the CLT process within neural networks. This approach limits the potential of the CLT for optimizing neural networks to some extent.

# Logit-Based Optimization

The logit function, introduced by Joseph Berkson in 1944, is derived from the term ”logistic unit” and describes the logarithm of odds (Berkson 1951). It maps the probability range $( 0 , 1 )$ to the entire real number line $( - \infty , + \infty )$ , allowing the application of linear regression techniques to probabilities (Cramer 2003). This mapping facilitates the use of regression methods in domains where outputs are naturally bounded probabilities rather than unbounded real numbers. In modern machine learning, the flexibility to let data drive model structures has led to more adaptive and predictive capabilities (Zhao et al. 2020). This flexibility contrasts with traditional logit models, which often rely on specific data structures and inherent behavioral assumptions.

Various methods have been developed to optimize neural networks by focusing on the logit function. Wu et al. (Wu and Klabjan 2021) introduced a reliable uncertainty measure based on logit outputs, aiding classification models in identifying instances prone to errors. This uncertainty measure can trigger expert intervention during high uncertainty classifications (Wu and Klabjan 2021). Neural networks often exhibit overconfidence, producing high confidence scores for both in- and out-of-distribution inputs. Wei et al. (Wei et al. 2022) addressed this issue with Logit Normalization (LogitNorm), modifying the cross-entropy loss to enforce a constant vector norm on the logits during training. In medical image analysis, Hu et al. (Hu et al. 2021) proposed logit space data augmentation, adaptively perturbing logit vectors to enhance classifier generalizability and mitigate overfitting from limited training data. These methods demonstrate that optimizing based on logit can significantly enhance neural network performance on finite datasets.

# Diffusion Probabilistic Models

Diffusion probabilistic models (DPMs) (or diffusion models [DMs]), inspired by non-equilibrium statistical physics (Jarzynski 1997), have recently gained traction in computer vision due to their remarkable generative capabilities. DMs generate highly detailed and diverse examples by iteratively reconfiguring data distribution through a diffusion process (Yang et al. 2023). Incorporating small amounts of Gaussian noise, DMs use conditional Gaussians for straightforward parameterization of neural networks. Leveraging variational inference via a parameterized Markov chain (Gagniuc 2017), DMs generate samples closely following the original data distribution within finite iterations.

Notable examples include latent diffusion models (LDMs) (Croitoru et al. 2023; Yang et al. 2023), which have set new standards in generative modeling. Stable Diffusion, a variant of LDMs, generates high-quality images based on text prompts, showcasing minimal artifacts and strong alignment with the prompts (Yang et al. 2023). DMs have been extensively applied in image generation (Nichol and Dhariwal 2021), super-resolution (Rombach et al. 2022), and image-to-image translation (Choi et al. 2021). Additionally, the latent representations learned by DMs have proven effective in discriminative tasks like image segmentation (Baranchuk et al. 2021), and classification (Zimmermann et al. 2021). This versatility underscores the potential of diffusion models in a broad range of applications, connecting them to the field of representation learning, which includes designing novel neural architectures and developing advanced learning strategies (Croitoru et al. 2023; Yang et al. 2023).

# Methods

In this paper, we propose the Bernoulli-Gaussian decision block, a novel module inspired by the CLT, which utilizes IDDPMs (Nichol and Dhariwal 2021) to model the probability of Bernoulli trials. We will first review the formulation of IDDPMs, followed by a detailed description of the proposed Bernoulli-Gaussian Decision Block built upon the IDDPMs.

# Improved Denoising Diffusion Probabilistic Models

Denoising Diffusion Probabilistic Models (DDPMs) (Ho, Jain, and Abbeel 2020) have demonstrated superior sample generation quality, often surpassing other generative models like GANs (Goodfellow et al. 2014) and VQ-VAE (Van Den Oord, Vinyals et al. 2017). Improved DDPMs (IDDPMs) (Nichol and Dhariwal 2021) build on DDPMs by incorporating learned variances, allowing sampling in fewer steps with minimal quality loss. In DDPMs, given data distribution $x _ { 0 } \sim q ( x _ { 0 } )$ , a forward noising process $q$ generates latent variables $x _ { 1 }$ through $x _ { T }$ by adding Gaussian noise at each time $t$ with variance $\beta _ { t } \in ( 0 , 1 )$ , as follows (Nichol and Dhariwal 2021):

$$
q ( x _ { 1 } , . . . , x _ { T } | x _ { 0 } ) : = \prod _ { t = 1 } ^ { T } q ( x _ { t } | x _ { t - 1 } ) ,
$$

where $q ( x _ { t } | x _ { t - 1 } ) : = \mathcal { N } ( x _ { t } ; \sqrt { 1 - \beta _ { t } } x _ { t - 1 } , \beta _ { t } I ) .$

With a sufficiently large $T$ and a carefully designed schedule for $\beta _ { t }$ , the latent variable $x _ { T }$ approximates an almost isotropic Gaussian distribution (Nichol and Dhariwal 2021). Consequently, if the exact reverse distribution $q ( x _ { t - 1 } | x _ { t } )$ were known, we could sample $x _ { T } \sim \mathcal { N } ( 0 , I )$ and reverse the process to obtain a sample from $q ( x _ { 0 } )$ . However, since $q ( x _ { t - 1 } | x _ { t } )$ relies on the entire data distribution, it is approximated using a neural network (Nichol and Dhariwal 2021):

$$
p _ { \theta } ( x _ { t - 1 } | x _ { t } ) : = \mathcal { N } ( x _ { t - 1 } ; \mu _ { \theta } ( x _ { t } , t ) , \Sigma _ { \theta } ( x _ { t } , t ) ) ,
$$

Through Maximum Likelihood Estimation (MLE), the distribution of $x _ { 0 }$ can be derived. The combined use of $q$ and $p$ forms a variational auto-encoder, and the Variational Lower Bound (VLB) can be written as follows (Nichol and

Dhariwal 2021):

$$
\begin{array} { r l r } {  { L _ { \mathrm { v l b } } = - \overbrace { \log p _ { \theta } ( x _ { 0 } | x _ { 1 } ) } ^ { L _ { 0 } } + \overbrace { D _ { K L } ( q ( x _ { T } | x _ { 0 } ) | | p ( x _ { T } ) ) } ^ { L _ { T } } } } \\ & { } & { + \sum _ { t > 1 } \overbrace { D _ { K L } ( q ( x _ { t - 1 } | x _ { t } , x _ { 0 } ) | | p _ { \theta } ( x _ { t - 1 } | x _ { t } ) ) } ^ { L _ { t - 1 } } . } \end{array}
$$

With $\alpha _ { t } : = 1 - \beta _ { t }$ and $\textstyle { \bar { \alpha } } _ { t } : = \prod _ { t } ^ { s = 0 } \alpha _ { s }$ , the marginal can be written as follow (Nichol and Dhariwal 2021; Ho, Jain, and Abbeel 2020):

$$
\begin{array} { r l } & { q ( x _ { t } | x _ { 0 } ) = { \cal N } ( x _ { t } ; \sqrt { \bar { \alpha } _ { t } } x _ { 0 } , ( 1 - \bar { \alpha } _ { t } ) { \cal I } ) , } \\ & { ~ \mathrm { w h e r e } ~ x _ { t } = \sqrt { \bar { \alpha } _ { t } } x _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , ~ \epsilon \sim { \cal N } ( 0 , { \cal I } ) . } \end{array}
$$

By applying Bayes’ theorem, the posterior $q ( x _ { t - 1 } | x _ { t } , x _ { 0 } )$ can be determined with $\tilde { \beta } _ { t }$ and $\tilde { \mu } _ { t } ( x _ { t } , x _ { 0 } )$ , defined as follows (Ho, Jain, and Abbeel 2020; Nichol and Dhariwal 2021):

$$
\begin{array} { r l } & { \tilde { \beta } _ { t } : = \frac { 1 - \bar { \alpha } _ { t - 1 } } { 1 - \bar { \alpha } _ { t } } \beta _ { t } , } \\ & { \tilde { \mu } _ { t } ( x _ { t } , x _ { 0 } ) : = \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { 1 - \bar { \alpha } _ { t } } x _ { 0 } + \frac { \sqrt { \alpha } _ { t } \left( 1 - \bar { \alpha } _ { t - 1 } \right) } { 1 - \bar { \alpha } _ { t } } x _ { t } , } \\ & { q ( x _ { t - 1 } | x _ { t } , x _ { 0 } ) = \mathcal { N } ( x _ { t - 1 } ; \tilde { \mu } _ { t } ( x _ { t } , x _ { 0 } ) , \tilde { \beta } _ { t } I ) . } \end{array}
$$

According to (Ho, Jain, and Abbeel 2020), the $L _ { t - 1 }$ can be calculated as:

$$
L _ { t - 1 } = \mathbb { E } _ { q ( x _ { 1 : T } ) } \left[ \frac { 1 } { 2 \sigma _ { t } ^ { 2 } } | | \tilde { \mu } _ { t } ( x _ { t } , x _ { 0 } ) - \mu _ { \theta } ( x _ { t } , t ) | | ^ { 2 } \right] + C .
$$

There are several ways to parameterize $\mu _ { \theta } ( x _ { t } , t )$ . One approach is to predict the noise $\epsilon$ with a neural network, and use Eqs. (4) and (5) to derive $\mathrm { ~ H o ~ }$ , Jain, and Abbeel 2020; Nichol and Dhariwal 2021):

$$
\mu _ { \theta } ( x _ { t } , t ) = \frac { 1 } { \sqrt { \alpha _ { t } } } ( x _ { t } - \frac { \beta _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } \epsilon _ { \theta } ( x _ { t } , t ) ) .
$$

Predicting $\epsilon$ with a reweighted loss function has proven effective (Ho, Jain, and Abbeel 2020; Nichol and Dhariwal 2021):

$$
L _ { \mathrm { s i m p l e } } = \mathbb { E } _ { t , x _ { 0 } , \epsilon } [ | | \epsilon - \epsilon _ { \theta } ( x _ { t } , t ) | | ^ { 2 } ] .
$$

In particular, as (Nichol and Dhariwal 2021) mentioned, IDDPM could generate a vector $v$ containing one component pre dimension, and this vector $v$ composes the new variances, $\Sigma _ { \theta } ( x _ { t } , t )$ in Eq. 2:

$$
\Sigma _ { \theta } ( x _ { t } , t ) = \exp ( v \log \beta _ { t } + ( 1 - v ) \log \tilde { \beta } _ { t } ) .
$$

Since $L _ { \mathrm { s i m p l e } }$ doesn’t reply on $\Sigma _ { \theta } ( x _ { t } , t )$ (Nichol and Dhariwal 2021), the two loss functions $L _ { \mathrm { v l b } }$ and $L _ { \mathrm { s i m p l e } }$ can be simply combined into a new hybrid objective by introducing a hyperparameter $\lambda _ { 1 }$ to scale one of them, i.e.,

$$
L _ { \mathrm { h y b r i d } } = L _ { \mathrm { s i m p l e } } + \lambda _ { 1 } L _ { \mathrm { v l b } } .
$$

This reparameterization technique allows the diffusion model to reconstruct Gaussian distributions, enabling the transformation of the logit from a single iteration into logits analogous to those from multiple experiments.

# Bernoulli Approximation

In traditional settings, a single iteration of forward propagation yields one probability estimate. However, we can view each iteration as an independent and replicable trial, treating it as a Bernoulli Trial (BT). By conducting multiple independent trials within a single forward propagation, we can obtain more precise results. When the number of BTs is large enough, the distribution of the BT results approximates a Gaussian distribution, as described by the De Moivre–Laplace theorem (Walker 2006). This allows us to incorporate the CLT to estimate the mean of the Gaussian distribution, representing the results of BTs. This mean can be predicted, enabling us to simulate this Bernoulli process in a single iteration instead of multiple training runs.

A Bernoulli trial has exactly two possible outcomes: “success” (i.e., the positive case) and “failure” (i.e., the negative case). Let $p$ be the probability of the positive case. In a typical CNN, logits are generated and then converted into probabilities (for classification), confidence scores, and other expected outputs through functions like softmax and sigmoid. In an ideal scenario, the probability of the positive case $p = 1$ . Therefore, each training iteration can be viewed as a BT, with the logit representing the expected value of a random variable following the Bernoulli distribution. We define this random variable as the Bernoulli logit $y _ { \mathrm { B l o g i t } }$ , which can take two fixed values: positive Bernoulli logit yBlogit and negative Bernoulli logit $\boldsymbol { y } _ { \mathrm { B l o g i t _ { - } } }$ . The logit $y _ { \mathrm { l o g i t } }$ can be calculated using the following equation:

$$
y _ { \mathrm { l o g i t } } = \mathbb { E } ( y _ { \mathrm { B l o g i t } } ) = y _ { \mathrm { B l o g i t _ { + } } } p + y _ { \mathrm { B l o g i t _ { - } } } ( 1 - p ) .
$$

If $p = 1$ , the logit is equal to the true value of the positive Bernoulli logits, i.e., $y _ { \mathrm { l o g i t } } = y _ { \mathrm { B l o g i t _ { + } } }$ as $n  \infty$ , according to the CLT. We refer to this process as the Bernoulli approximation.

Repeating the BT independently $n$ times, the possible values of the total number of positive outcomes range from 0 to $n$ . Let $\hat { p }$ denote the estimated probability of a positive outcome in $n$ trials, we have

$$
\mathbb { E } ( \hat { p } ) = p , \qquad V a r ( \hat { p } ) = \frac { p ( 1 - p ) } { n } ,
$$

where $\mathbb { E } ( \hat { p } )$ denotes the expected value of $\hat { p }$ , $V a r ( \hat { p } )$ denotes the variance of $\hat { p }$ . We incorporate a CNN to construct a Gaussian distribution by learning its mean and variance. According to the De Moivre–Laplace theorem (Walker 2006), as $n$ increases, the distribution of $\hat { p }$ increasingly resembles a Gaussian distribution:

$$
\hat { p } \sim \mathcal { N } ( p , \sqrt { \frac { p ( 1 - p ) } { n } } ) .
$$

According to Eqs. (13) and (12), the mean of the Gaussian distribution is equal to the true value of the success probability of BT as $n  \infty$ , according to the CLT. Under optimal conditions, $y _ { \mathrm { l o g i t } }$ can be calculated through multiple BTs. However, since the Bernoulli logit follows a Gaussian distribution, $y _ { \mathrm { l o g i t } }$ can be calculated as follows:

$$
\begin{array} { r l } & { y _ { \mathrm { l o g i t } } = \mathbb { E } ( y _ { \mathrm { B l o g i t } } ) } \\ & { \qquad = y _ { \mathrm { B l o g i t _ { + } } } \hat { p } + y _ { \mathrm { B l o g i t _ { - } } } ( 1 - \hat { p } ) . } \end{array}
$$

In an ideal scenario, the probability $\hat { p }$ is 1, meaning each BT would succeed, otherwise is 0. Thus, $y _ { \mathrm { l o g i t } }$ is equal to the true value of $\boldsymbol { y } _ { \mathrm { B l o g i t _ { + } } }$ as $n \to \infty$ , according to the CLT. Following Eq. (13), after applying the softmax or sigmoid function, the mean of the Gaussian distribution can be used to categorize outputs as 0 or 1, thereby supervising the CNN model. Additionally, the variance of the Gaussian distribution would be zero in this ideal case, allowing us to simulate multiple BTs with their mean and variance in only one iteration. Through this entire process, logits are transformed into a Gaussian distribution.

# Bernoulli-Gaussian Decision Block

Building on the concepts of Bernoulli approximation and IDDPMs, we introduce the Bernoulli-Gaussian decision block into the deep model training process, shown in Figure 1. This Bernoulli-Gaussian Decision Block (BGDB) aims to enhance the stability and performance of discriminative classifiers by leveraging the mean of logits from multiple experiments to supervise a single learning process.

Meanwhile, we employ IDDPM to construct and refine the probability distributions of BTs. The entire construction process can be supervised by the $L _ { \mathrm { h y b r i d } }$ . Compared to DDPM, IDDPM can generate both mean and variance, this approach perfectly aligns with Bernoulli Approximation. Simultaneously, through the inverse diffusion process, we sample the mean $\mu _ { \mathrm { o u t p u t } }$ and variance $\sigma _ { \mathrm { o u t p u t } }$ at time $t _ { 0 }$ , where $\bar { p ( x _ { 0 } ) } \sim ( \mu _ { \mathrm { o u t p u t } } , \dot { \sigma _ { \mathrm { o u t p u t } } } )$ , from the logit produced by the backbone. After applying the softmax or sigmoid function, $\mu _ { \mathrm { o u t p u t } }$ of the Gaussian distribution is required to categorize outputs as 0 or 1 to supervise the CNN model. Ideally, $\sigma _ { \mathrm { o u t p u t } }$ should be 0, allowing us to construct a multiple BTs with $\mu _ { \mathrm { o u t p u t } }$ and $\sigma _ { \mathrm { o u t p u t } }$ in a single iteration. Let $L _ { \mu }$ and $L _ { \sigma }$ be the loss targeting at mean $\mu _ { \mathrm { o u t p u t } }$ and variance $\sigma _ { \mathrm { o u t p u t } }$ (for Bernoulli approximation). Let $\scriptstyle { \dot { L } } _ { \mathrm { B C E } }$ denote the Balanced Cross-Entropy (BCE) loss, $L _ { \mathrm { M S E } }$ denote the Mean Squared Error (MSE) loss, $F$ represents the softmax or sigmoid function. Given that the mean is represented as a probability while the variance is numerically zero, the mean loss is calculated using BCE, whereas the variance loss is obtained using MSE. Especially, $L _ { y }$ , task-specific loss such as Dice loss in segmentation tasks, can be calculated from the logit of a single learning process.

Thus, the entire loss function $\mathcal { L }$ for the model with BGDB module is calculated as follows:

$$
\begin{array} { r l } & { \mathcal { L } = L _ { \mathrm { y } } + \lambda _ { 2 } L _ { \mathrm { h y b r i d } } } \\ & { + \lambda _ { 3 } ( \widetilde { ( L _ { \mathrm { B C E } } F ( \mu _ { \mathrm { o u t p u t } } ) , 1 a b e l ) } + \widetilde { L _ { \mathrm { M S E } } ( \sigma _ { \mathrm { o u t p u t } } , 0 ) } ) . } \end{array}
$$

Since $F ( \mu _ { \mathrm { o u t p u t } } )$ is a probability, it can also be used in other loss functions, such as Dice loss (Milletari, Navab, and Ahmadi 2016). This module is added after the logits and before the softmax to compute the loss function during training. After training, this structure is removed, and predictions are made using the original network, without any burden in inference. The label may encompass options such as the category of a single object or pixel.

![](images/64fec473b8e30141e4e29a441bd52371ea35e236f37a5d85246a737dd730da87.jpg)  
Figure 1: Workflow for performing segmentation tasks. The total loss in the training pipeline includes $L _ { y }$ (task-specific loss), $L _ { \mathrm { h y b r i d } }$ (for IDDPM), and $L _ { \mu } + L _ { \sigma }$ (for Bernoulli approximation). During training, the input image is first processed by the backbone model to obtain the logits for a single experiment, supervised by $L _ { y }$ . These logits are then used to train the IDDPM model, resulting in a latent space composed entirely of Gaussian noise, which is supervised by $L _ { \mathrm { h y b r i d } }$ . By sampling from this latent space, a Gaussian distribution for CLT’s results is formed. This process is supervised by $L _ { \mu } + L _ { \sigma }$ . After training, only the backbones are retained for inference.

This construction process begins by minimizing a loss function to generate a new distribution from an existing one. Because the output derived from the loss function adheres to the same distribution as the input, supervised learning is primarily needed for the mean and variance of the noise. By controlling these parameters, the entire diffusion process can transform one distribution into another desired distribution, the probability of multiple successful BT experiments. In generative tasks, the input distribution for diffusion models is initially fixed. However, in classification problems, the input logits are obtained through supervised learning, which can introduce instability. By leveraging the learning process of diffusion models, we use the distribution of logits from multiple experiments to supervise the logits obtained from a single training session. This approach aims to stabilize and enhance training by supervising the process with multiple experimental logits derived from a single training instance.

# Experiment

We evaluate the proposed method across various imaging tasks, including both classification and segmentation.

# Urban and General Scene Segmentation

Datasets We utilized Cityscapes (Cordts et al. 2016) and PASCAL Visual Object Classes (VOC) Challenge (Pascal VOC) (Everingham et al. 2010) datasets for this task. The Cityscapes dataset addresses the need for detailed semantic understanding by providing annotated stereo video sequences from 50 cities. It includes 5,000 images with highquality pixel-level annotations, making it well-suited for evaluating segmentation methods that leverage extensive, high-quality labeled data. The Pascal VOC dataset offers publicly accessible images and annotations along with standardized evaluation software. For segmentation tasks, each test image requires predicting the object class for each pixel, with “background” designated for pixels that do not belong to any of the twenty specified classes.

Compared Methods For our experiments on the Cityscapes and Pascal VOC datasets, we utilized the

DeepLabV3 framework (Chen et al. 2017, 2018), following the experimental protocols outlined in the original papers. We evaluated the performance using three distinct backbones: MobileNet (Howard et al. 2017), ResNet101 (He et al. 2016), and HRNet (Wang et al. 2020). This approach allowed us to systematically assess the model’s adaptability and efficacy across varied scenarios.

Experimental Settings Our training regimen consisted of 30,000 iterations, with each batch comprising 16 samples. All input images were uniformly cropped to dimensions of $2 5 6 \times 2 5 6$ . We employed the cross-entropy loss function, coupled with a learning rate of 0.01 and a weight decay of 1e-4. Stochastic Gradient Descent (SGD) (Robbins and Monro 1951) was used as the optimizer throughout the training process to ensure optimal convergence and model refinement. For testing, the images from the Cityscapes dataset retained their original size, while the Pascal VOC images were resized to $2 5 6 \times 2 5 6$ . Model performance was assessed using the Mean Intersection over Union (mIoU) metric.

In this study, all models were trained on an NVIDIA A100 GPU with $8 0 \mathrm { G B }$ of memory. The hyperparameters were set as follows: $\lambda _ { 1 }$ to $1 \times 1 0 ^ { - 3 }$ , and both $\lambda _ { 2 }$ and $\lambda _ { 3 }$ to 1. These settings were used for all subsequent experiments.

Experimental Results As illustrated in Table 1, on both Cityscapes and Pascal VOC, all models experienced moderate improvements. Specifically, the models showed an increase in performance ranging from $0 . 0 8 \%$ to $1 . 4 8 \%$ on the Cityscapes dataset and from $0 . 2 1 \%$ to $0 . 4 1 \%$ on the Pascal VOC dataset. These results demonstrate the effectiveness of the proposed Bernoulli-Gaussian decision block in enhancing the performance.

# Skin Lesion Segmentation

Datasets We used the International Skin Imaging Collaboration (ISIC) dataset (Tschandl, Rosendahl, and Kittler 2018; Codella et al. 2019) for skin lesion segmentation. The ISIC dataset is the world’s largest collection of dermoscopic skin images. The ISIC 2018 challenge, held at the MICCAI conference, included three tasks and featured over 12,500 images. The challenge attracted 900 registered users, with 115 submissions for lesion segmentation, 25 for lesion attribute detection, and 159 for disease classification.

Table 1: The results of mIoU $( \mathrm { M e a n } \pm \mathrm { S t d } )$ for urban and general scene segmentation on Cityscapes and Pascal VOC datasets. “DLP” denotes “DeepLabv $3 \mathrm { + } ^ { \cdot \mathrm { , } }$ .   

<html><body><table><tr><td rowspan="2">Model</td><td>Cityscapes</td><td>Pascal VOC</td></tr><tr><td>mIoU (%)</td><td>mIoU(%)</td></tr><tr><td rowspan="2">DLP_MobileNet +ours</td><td>63.61 ± 0.72</td><td>61.78 ± 0.57</td></tr><tr><td>65.09 ± 0.38 +1.48</td><td>62.17 ± 0.65 +0.39</td></tr><tr><td rowspan="2">DLP_ResNet101 +ours</td><td>72.00 ± 0.36</td><td>69.74±0.49</td></tr><tr><td>72.08 ±0.10 +0.08</td><td>69.95 ± 0.52 +0.21</td></tr><tr><td rowspan="2">DLP_HRNet +ours</td><td>72.09 ± 0.47</td><td>69.87 ± 0.42</td></tr><tr><td>72.92 ± 0.37 +0.82</td><td>70.28 ± 0.57 +0.41</td></tr></table></body></html>

Compared Methods We evaluated the BernoulliGaussian decision block across several classical and state-of-the-art 2D medical segmentation models using the ISIC dataset. These models include U-Net (Ronneberger, Fischer, and Brox 2015), Attention U-Net (Oktay et al. 2018), U-Net++ (Zhou et al. 2019), FCN (Liu et al. 2018), ResUNet (Diakogiannis et al. 2020), and UNETR (Hatamizadeh et al. 2022), all implemented using the MONAI framework (Cardoso et al. 2022). The baseline models were trained using the Dice loss (Milletari, Navab, and Ahmadi 2016), while “+ours” models were trained with our proposed loss function in addition to the Dice loss.

Experimental Settings We utilized the training, validation, and test datasets provided by the ISIC 2018 challenge. These datasets were combined and then randomly split into training and testing sets in a 5:2 ratio (2,600 images for training and 1,094 for testing). We performed 5-fold crossvalidation, selecting the optimal model from each fold’s validation set. The selected models were then evaluated on the testing set, and we recorded the mean and variance of performance metrics across the 5 folds. For data augmentation, we normalized pixel values to a range between 0 and 255 and resized the images to $2 5 6 \times 2 5 6$ to meet the input requirements of the proposed block.

The models were trained using the AdamW (Loshchilov and Hutter 2017) optimizer with a weight decay of $1 e { \mathrm { - } } 5$ and a learning rate of $1 e { - 4 }$ . Each model underwent 10,000 iterations of training, with the goal of achieving the highest Dice scores. This approach enabled a thorough comparative analysis between the baseline and enhanced models by the proposed decision block. Model performance was assessed using Dice score (Milletari, Navab, and Ahmadi 2016), which measures the overlap between the predicted segmentation and the ground truth.

Experimental Results Table 2 shows the Dice scores for the models on the ISIC dataset. Upon integrating the proposed block, we observed performance improvements across most models, with the exception of $_ { \mathrm { U - N e t + + } }$ , which experienced a marginal decline of $- 0 . 2 9 \%$ . The performance improvements for the other models ranged from $0 . 6 \%$ to $4 . 7 4 \%$ .

# Beyond Segmentation: Skin Lesion Classification

Datasets and Compare Methods Similar to the skin lesion segmentation task, we used the ISIC 2018 challenge dataset for skin lesion classification (Tschandl, Rosendahl, and Kittler 2018; Codella et al. 2019). We conducted empirical analyses across a spectrum of prominent models to assess the efficacy of the Bernoulli-Gaussian decision block. The models included DenseNet (Huang et al. 2017), ResNet (He et al. 2016), Vision Transformer (ViT) (Dosovitskiy et al. 2020), EfficientNet (Tan and Le 2019), and SENet (Hu, Shen, and Sun 2018).

Experimental Settings We utilized the entirety of the ISIC 2018 dataset, amalgamating all available images before randomly partitioning them into training and testing sets in a 5:1 ratio while maintaining the original distribution. We conducted rigorous 5-fold cross-validation within the test set. From each fold, we selected the model with the highest accuracy on the validation set for final testing. We documented the mean and variance of accuracy and the Area Under the ROC Curve (AUC) across the 5-fold models.

To rigorously evaluate the model’s performance, we employed basic data augmentation strategies, including random rotations up to 15 degrees, flipping, and zooming in or out by a scale of 0.1 with a $50 \%$ probability. We used the Adam optimizer (Kingma and Ba 2014) with a learning rate of 1e-5. Each model was trained for 50 epochs to achieve the highest levels of accuracy. This approach allowed for an exhaustive comparative analysis between models with and without the proposed block, enhancing our understanding of their respective performances.

Experimental Results Table 3 shows the accuracy and AUC scores for the models on the ISIC dataset. The experimental findings indicate that, aside from slight declines in the ViT ( $0 . 0 4 \%$ accuracy), all other models exhibited performance enhancements. The improvements ranged from $0 . 5 4 \%$ to $1 . 6 \%$ in accuracy and from $0 . 2 4 \%$ to $0 . 7 4 \%$ in AUC.

# Ablation Experiments

To thoroughly understand the impact of different loss functions on the performance of our model, we conducted ablation experiments using the U-Net model with various combinations of loss functions. We set all hyperparameters to 1. The loss functions evaluated included the task-specific loss (i.e., Dice loss $L _ { \mathrm { D i c e } } )$ ), the diffusion loss $L _ { \mathrm { h y b r i d } }$ for IDDPM, the BT loss for Bernoulli approximation (i.e., $L _ { \mu } + L _ { \sigma } )$ . The specific combinations tested were: U-Net (i.e., Only $L _ { \mathrm { D i c e } } )$ , $L _ { \mathrm { D i c e } } + L _ { \mu } + L _ { \sigma }$ (i.e., no diffusion loss), $L _ { \mathrm { D i c e } } + L _ { \mathrm { h y b r i d } }$ (i.e., no BT loss), all losses combined (i.e., ${ \cal L } _ { \mathrm { D i c e } } + { \cal L } _ { \mu } { \bf \dot { \theta } } + { \cal L } _ { \sigma } +$ $L _ { \mathrm { h y b r i d } } )$ . For each combination of loss functions, we trained the U-Net model on the ISIC dataset using the same experimental settings as described previously.

Table 4 shows the Dice scores for the different combinations of loss functions. Our analysis revealed that incorporating all loss functions led to the best performance, with

Table 2: Dice (Mean $( \% ) \pm \mathrm { S t d } )$ for skin lession segmentation on ISIC dataset. “A\*U-Net” denotes “Attention U-Net”.   

<html><body><table><tr><td rowspan="2">Model</td><td colspan="2">ISIC</td></tr><tr><td>Accuracy (%)</td><td>AUC (%)</td></tr><tr><td>DenseNet169</td><td>68.34± 0.59</td><td>89.36± 0.50</td></tr><tr><td>+ours</td><td>69.83±1.05 +1.49</td><td>90.10± 0.49 +0.74</td></tr><tr><td>ViT</td><td>69.28± 0.59</td><td>89.06 ± 0.24</td></tr><tr><td>+ours</td><td>69.24 ± 0.65 -0.04</td><td>89.30 ± 0.18 +0.24</td></tr><tr><td>ResNet50</td><td>66.66 ± 0.90</td><td>88.44± 0.29</td></tr><tr><td>+ours</td><td>68.26 ±1.01 +1.60</td><td>88.90 ± 0.85 +0.46</td></tr><tr><td>SENet154</td><td>69.64±1.04</td><td>89.79±0.38</td></tr><tr><td>+ours</td><td>70.18 ± 1.50 +0.54</td><td>90.23 ± 0.56 +0.44</td></tr><tr><td>EfficientNet</td><td>65.78± 0.68</td><td>88.44± 0.49</td></tr><tr><td>+ours</td><td>67.14 ± 0.60 +1.36</td><td>89.10± 0.21+0.66</td></tr></table></body></html>

Table 3: Accuracy and AUC (Mean $( \% ) \pm \mathrm { S t d } )$ for skin lession classification on ISIC dataset.   

<html><body><table><tr><td>UNETR 77.62 ± 2.7</td><td>FCN 73.52 ± 2.7</td><td>U-Net 69.17 ±1.9</td><td>ResUNet 75.82 ± 1.24</td><td>A*U-Net 72.47 ± 1.86</td><td>U-Net++ 80.78 ±0.83</td></tr><tr><td>+ours</td><td>+ours</td><td>+ours</td><td>+ours</td><td>+ours</td><td>+ours</td></tr><tr><td>80.30 ± 2.45 +2.68</td><td>75.04 ± 2.7 +1.52</td><td>73.91 ±1.19 +4.74</td><td>76.44 ± 0.84 +0.62</td><td>73.49 ± 0.98 +1.02</td><td>80.49 ±1.22 -0.29</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table></body></html>

a Dice score improved by $4 . 7 4 \%$ . This underscores the profound impact of combining multiple loss functions on model performance.   

<html><body><table><tr><td>U-Net</td><td>No diffusion loss</td><td>No BT loss</td><td>All losses combined</td></tr><tr><td>69.17±1.90</td><td>72.23±1.14</td><td>73.17±0.86</td><td>73.91±1.19</td></tr></table></body></html>

Table 4: Dice scores of different loss combinations for skin lession segmentation on the ISIC dataset.

We also explored the impact of hyperparameter settings of $\lambda _ { 2 }$ and $\lambda _ { 3 }$ in Eq. 15. This ablation study was conducted on the first fold of the U-Net experiment’s dataset. The hyperparameter preceding the initial model’s Dice loss was fixed at 1. Initially, with $\lambda _ { 3 }$ set to 1, the hyperparameter before the $\lambda _ { 2 }$ was varied at 0.01, 0.5, 1, and 2 to observe changes in model performance. Similarly, with the Dice loss and $\lambda _ { 3 }$ fixed at 1, the $\lambda _ { 2 }$ was adjusted to 0.01, 0.5, 1, and 2, allowing us to evaluate its effect on the model’s performance. Figure 2 shows that the model performs most stably when the hyperparameters for all three losses are consistent, which aligns with our default setting. Additionally, fine-tuning the hyperparameters within the range of 0.5 to 1 can be beneficial for achieving optimal performance.

# Discussion

While our model achieved moderate advancements in segmentation tasks, it encounters several limitations. Statistically, the proposed Bernoulli-Gaussian decision block relies on a sufficiently large number of trials, $n$ , to satisfy the formula under optimal conditions. The block can determine $n$ to ensure the validity of the mean of the Gaussian distribution.

![](images/91fae52ac95ea400b11b999a1414f11efdb9c68f62741f83be2d973d7f97e1a4.jpg)  
Figure 2: Impact of hyperparameters $\lambda _ { 2 }$ and $\lambda _ { 3 }$ on segmentation model performance, with the other hyperparameters fixed at 1.

The IDDPM, while faster than DDPM in training and inference, faces slowdown due to simultaneous training and inference within the proposed block. This limits the diffusion model’s time steps to at least 25, complicating training. Our experiments focused on 2D segmentation $2 5 6 \times 2 5 6$ images), and the IDDPM’s large parameter count makes it impractical for 3D images, requiring excessively small image sizes unsuitable for 3D segmentation.

The U-Net architecture’s encoder-decoder structure limits the predicted value dimensions to powers of 2, complicating classification tasks. On the ISIC dataset, we explored resizing logits from $1 \times 1$ to $6 4 \times 6 4$ and averaging the reconstructed results from IDDPM. Although the model shows potential for classification, the averaging operation appears redundant. Given IDDPM’s complexity, our BGDB block employs default hyperparameters tailored for the generative model. Unlocking its full potential requires meticulous tuning, which is more feasible with a simplified BGDB block.

# Conclusion

We proposed a novel Bernoulli-Gaussian Decision Block, which constructs experimental probability distributions using the diffusion model, achieving modest segmentation improvements and showing promise for classification tasks.