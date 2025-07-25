# CODE: Confident Ordinary Differential Editing

Bastien van Delft, Tommaso Martorella, Alexandre Alahi

E´ cole Polytechnique Fe´d´erale de Lausanne (EPFL) firstname.lastname@epfl.ch

# Abstract

Conditioning image generation facilitates seamless editing and the creation of photorealistic images. However, conditioning on noisy or Out-of-Distribution (OoD) images poses significant challenges, particularly in balancing fidelity to the input and realism of the output. We introduce Confident Ordinary Differential Editing (CODE), a novel approach for image synthesis that effectively handles OoD guidance images. Utilizing a diffusion model as a generative prior, CODE enhances images through score-based updates along the probability-flow Ordinary Differential Equation (ODE) trajectory. This method requires no task-specific training, no handcrafted modules, and no assumptions regarding the corruptions affecting the conditioning image. Our method is compatible with any diffusion model. Positioned at the intersection of conditional image generation and blind image restoration, CODE operates in a fully blind manner, relying solely on a pre-trained generative model. Our method introduces an alternative approach to blind restoration: instead of targeting a specific ground truth image based on assumptions about the underlying corruption, CODE aims to increase the likelihood of the input image while maintaining fidelity. This results in the most probable in-distribution image around the input. Our contributions are twofold. First, CODE introduces a novel editing method based on ODE, providing enhanced control, realism, and fidelity compared to its SDE-based counterpart. Second, we introduce a confidence interval-based clipping method, which improves CODE’s effectiveness by allowing it to disregard certain pixels or information, thus enhancing the restoration process in a blind manner. Experimental results demonstrate CODE’s effectiveness over existing methods, particularly in scenarios involving severe degradation or OoD inputs.

Website — https://vita-epfl.github.io/CODE/ Code — https://github.com/vita-epfl/CODE/ Main and Appendix — https://arxiv.org/pdf/2408.12418

# 1 Introduction

Conditional image generation consists of guiding the creation of content using different sorts of conditioning, such as text, images, or segmentation maps. Our research focuses on scenarios where the guidance is an Out-of-Distribution (OoD) image relative to the training data distribution. This is especially relevant for handling corrupted images, similar to denoising or restoration methods. The main challenge in these scenarios is balancing fidelity to the input with realism in the generated images. Traditional methods for restoring corrupted images, such as Image-to-Image Translation or Style Transfer, are limited by the need for distinct datasets per style or per noise. Another approach models the corruption function as an inverse problem, requiring detailed knowledge of each possible corruption, making it impractical for most real unknown OoD scenarios. Guided image synthesis for OoD inputs aims to rectify corrupted images without prior knowledge of the corruption, positioning it as Blind Image Restoration (BIR). Despite recent advancements, achieving human-level generalization remains challenging.

![](images/888baa0af37eb9d46925db9598d91d20834ceac1554ad28fa244221156d3f74a.jpg)  
Figure 1: CODE: a conditional image generation framework for robust Out-of-Distribution image guidance.

Our work aims to generate realistic and plausible images from potentially corrupted inputs using only a pre-trained generative model, without additional data augmentation or finetuning on corrupted data, and without any specific assumption about the corruptions. Unlike other BIR methods that strive to reconstruct a ground-truth image relying on specific guidance or human-based assumptions, our approach is fully blind, seeking to maximize the input image’s likelihood while minimizing modifications to the input image. As such, we differ from traditional BIR approaches.

BIR is inherently ill-posed due to the loss of information from unknown degradation, necessitating auxiliary information to enhance restoration quality. Previous approaches have incorporated domain-specific priors such as facial heatmaps or landmarks (Chen et al. 2018, 2021; Yu et al. 2018), but these degrade with increased degradation and lack versatility. Generative priors from pre-trained models like GANs (Chan et al. 2021; Zhou et al. 2022; Yang et al. 2021; Pan et al. 2021; Menon et al. 2020) become unstable with severe degradation, leading to unrealistic reconstructions. Methods like (Wang et al. 2021a) combine facial priors with generative priors to improve fidelity but fail under extreme degradation. In (Meng et al. 2021), the authors replace GANs with diffusion models (Ho, Jain, and Abbeel 2020; Song et al. 2020) as generative priors. However, as the degradation increases, the method forces a choice between realism and fidelity.

BIR still fails to achieve faithful and realistic reconstruction for a wide range of corruptions on a wide range of images. Dealing with various unknown corruptions prevents inverse methods from being easily applicable while dealing with a wide range of images prevents relying on carefully designed domain priors. We introduce Confident Ordinary Differential Editing (CODE), an unsupervised method that generates faithful and realistic image reconstructions from a single degraded image without information on the degradation type, even under severe conditions. CODE leverages the generative prior of a pre-trained diffusion model without requiring additional training or finetuning. Consequently, it is compatible with any pre-trained Diffusion Model and any dataset. CODE optimizes the likelihood of the generated image while constraining the distance to the input image, framing restoration as an optimization problem. Similar to GAN-inversion methods (Tov et al. 2021; Abdal, Qin, and Wonka 2020, 2019; Zhu et al. 2020; Menon et al. 2020; Pan et al. 2021), CODE inverts the observation to a latent space before optimization but similar to SDEdit (Meng et al. 2021) we propose to replace GANs with diffusion models (Ho, Jain, and Abbeel 2020; Song et al. 2020) as generative priors. Unlike GAN inversion, which relies on an auxiliary trained encoder, diffusion model inversion uses differential equations. In SDEdit, random noise is injected into the degraded observation to partially invert it in order to subsequently revert the process using a stochastic differential equation (SDE). As more noise is injected, a higher degree of realism is ensured, but at the expense of fidelity due to the additional loss of information caused by the noise randomness and the non-deterministic sampling from the SDE. We found that in some cases, as the degree of degradation increases, the method requires too high a degree of noise injection to work, forcing a choice between realism or fidelity. CODE refines SDEdit (Meng et al. 2021) by leveraging the probability-flow Ordinary Differential Equation (ODE) (Song et al. 2020), ensuring bijective correspondence with latent spaces. We use Langevin dynamics with score-based updates for correction, followed by the probability-flow ODE to project the adjusted latent representation back into the image space. This decouples noise injection levels, correction levels, and latent spaces, enhancing control over the editing process. Furthermore, CODE introduces a confidence-based clipping method that relies on the marginal distribution of each latent space. This method allows for the disregard of certain image information based on probability, which synergizes with our editing method. Our experimental results show CODE’s superiority over SDEdit in realism and fidelity, especially in challenging scenarios.

# 2 Background

# Related Works

A detailed comparison of the requirements of state-of-the-art methods is provided in the Appendix A.

Inverse Problems In the inverse problem setup, methods are designed to leverage sensible assumptions on the degradation operators. When combined with powerful generative models such as diffusion models, these approaches have achieved outstanding results, setting new benchmarks in the field (Saharia et al. 2022; Liang et al. 2021; Kawar et al. 2022; Murata et al. 2023; Zhu et al. 2023; Chung et al. 2023; Wang, Yu, and Zhang 2022). Several subcategories of the inverse problem setting, like blind linear problems and nonblind non-linear problems, drop some assumptions about the degradation operators and, therefore, extend their applicability. However, while producing exceptional results in controlled applications like deblurring and super-resolution, the necessity for assumptions on the degradation operator makes them often impractical for unknown corruptions or in real-world scenarios. Consequently, these methods are not directly applicable to our context, where such exact information is typically unavailable. DDRM (Kawar et al. 2022), DDNM (Wang, Yu, and Zhang 2022), GibbsDDRM (Murata et al. 2023), DPS (Chung et al. 2023), and DiffPIR (Zhu et al. 2023) belong to this category.

Conditional Generative Models with Paired Data A parallel approach involves conditioning a generative model on a degraded image. Most methods in this category require training with pairs of degraded and natural images (Mirza and Osindero 2014; Isola et al. 2017; Batzolis et al. 2021; Xia et al. 2023; Li et al. 2023; Liu et al. 2023; Chung, Kim, and Ye 2023). Additionally, these methods often depend on carefully designed loss functions and guidance mechanisms to enhance performance, as demonstrated by (Song et al. 2020). Conditional Generative Adversarial Networks, as explored in (Isola et al. 2017), exemplify this approach, where generative models are trained to regenerate the original sample when conditioned on its version in another domain. However, when the degradation process is unknown or varies widely, these models struggle to generalize effectively, rendering them less applicable and not comparable to our method, which operates without such constraints. DDB (Chung, Kim, and Ye 2023) and $\mathrm { I } ^ { 2 } \mathrm { S B }$ (Liu et al. 2023) fall into this category.

Unsupervised Bridge Problem with Unpaired Datasets In scenarios where two distinct datasets of clean and degraded data are available without direct paired data, methodologies based on principles like cycle consistency and realism have been developed, as evidenced by the works of (Zhu et al. 2017) using GANs (Goodfellow et al. 2014) and (Su et al. 2023) using Diffusion Models (Ho, Jain, and Abbeel 2020),(Sohl-Dickstein et al. 2015). A direct application of such methods to our scenario is not feasible due to the need for datasets of degraded images, which would hamper the ability to generalize to unseen corruptions.

Blind Image Restoration with task-specific or domainspecific information Blind Image Restoration methods aim to handle a variety of degradations without restricting themselves to specific types. A recent trend in this field is the transposition of the problem into a latent space where corrections are made based on prior distributions. Notable works in this area include (Abdal, Qin, and Wonka 2019, 2020; Chan et al. 2021; Zhu et al. 2020; Poirier-Ginter and Lalonde 2023) have explored various aspects of this approach, utilizing GAN inversion, or VAE/VQVAE encoding (Kingma and Welling 2013; Oord, Vinyals, and Kavukcuoglu 2017), and have achieved significant advancements, particularly in scenarios involving light but diverse degradations. Usually, methods for blind image restoration incorporate domainspecific information (Zhou et al. 2022; Wang et al. 2021a; Gu et al. 2022) or task-specific guidances (Fei et al. 2023).

Moreover, several methods (Lin et al. 2023; Yang et al. 2023) rely on the combination of many blocks trained separately (such as Real-ESRGAN (Wang et al. 2021b)) and incorporate different task-specific information (e.g., different restorers), making it even harder to ensure resilience to diverse degradations.

SDEdit and ILVR The works by Meng et al. in (Meng et al. 2021) and Choi et al. in (Choi et al. 2021) inspired the formulation of CODE. SDEdit relies on the stochastic exploration of a given input’s neighborhood to yield realistic and faithful outputs within a limited editing range. This method, tested for robustness by (Gao et al. 2022), bears similarities to the proposed gradient updates within the latent space of GAN models but is grounded in more solid theoretical foundations. ILVR is an iterative conditioning method designed to generate diverse images that share semantics with a downsampled guidance image. However, it requires a clean image for downsampling, which is not feasible in our scenario where the input guidance is already corrupted. Downsampling in this context would exacerbate information loss, making ILVR unsuitable for our application.

# Preliminary - Diffusion Models

Denoising Diffusion Probabilistic Models We denote $\mathbf { x } _ { \mathrm { 0 } }$ the data from the data distribution, in our case natural images, and $\mathbf { x } _ { 1 } , . . . , \mathbf { x } _ { T }$ the latent variables. The forward process is in DDPM (Ho, Jain, and Abbeel 2020) then defined by:

$$
\begin{array} { r } { \mathbf { x } _ { t + 1 } = \alpha _ { t } \cdot \mathbf { x } _ { t } + ( 1 - \alpha _ { t } ) \cdot \boldsymbol { \epsilon } , \mathrm { w i t h } \boldsymbol { \epsilon } \sim \mathcal { N } ( 0 , \mathbf { I } ) , } \end{array}
$$

Where $\alpha _ { t }$ is a schedule predefined as an hyperparameter. The diffusion model $\epsilon _ { \theta }$ is then trained to minimize $\mathbb { E } _ { \mathbf { x } _ { t } , t } \left\| \epsilon _ { \theta } ( \mathbf { x } _ { t } , t ) - \epsilon \right\|$ .

Score-based Generative Models In the case of Scorebased Generative Models (Song and Ermon 2019; Song et al. 2020; Song and Ermon 2020), the model $s _ { \theta }$ learns to approximate the score function, $\nabla _ { \mathbf { x } } \log p ( \mathbf { x } )$ , by minimizing:

$$
\begin{array} { r } { \mathbb { E } _ { p ( \mathbf { x } ) } \left\| s _ { \theta } ( \mathbf { x } ) - \nabla _ { \mathbf { x } } \log p ( \mathbf { x } ) \right\| . } \end{array}
$$

The most common approach to solve this is denoising score matching (Vincent 2011), which is further described in the

![](images/fa542fbd64d8886c31378f4b020a7bacb1c8dcb34f8df5d39d5a74c108aa29bc.jpg)  
Figure 2: Editing corrupted images with ODE. The green contour plot represents the distribution of images. Given a corrupted image, we encode it into a latent space using the probability-flow ODE and our Confidence-Based Clipping. We use Langevin Dynamics in the latent space to correct the encoded image. Finally, we project the updated latent back into the visual domain.

Appendix C.

Crucially, one can sample from $p ( x _ { t } )$ while using only the score function through Langevin dynamics (Langevin 1908) sampling by repeating the following update step:

$$
x _ { t + 1 } = x _ { t } + \epsilon \cdot s _ { \theta } ( x _ { t } , t ) + \sqrt { 2 \epsilon } \cdot \eta , \mathrm { w i t h } \eta \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) .
$$

# 3 Method: Confident Ordinary Differential Editing

# Editing with Ordinary Differential Equations

Our approach, described in Figure 2, formulates a theoretically grounded method for mapping OoD samples to indistribution ones.

From Gaussian Perturbation to Ordinary Inversion Our method draws inspiration from SDEdit (Meng et al. 2021) but introduces significant enhancements. SDEdit inverts the diffusion process by injecting Gaussian noise into the input image and then using this noisy image as a starting point to generate an image using DDPM (Ho, Jain, and Abbeel 2020). This process involves a trade-off between fidelity and realism: more noise results in more realistic images but less fidelity to the original input.

In contrast, we propose inverting the Probability-Flow Ordinary Differential Equation (ODE) as a superior alternative to noise injection. This approach maintains the fidelity of the reconstructed image by avoiding extra noise. The inversion process and its reverse operation ensure precise image reconstruction, limited only by approximation errors (Su et al. 2023). Unlike SDEdit, which requires increasing noise levels to revert to deeper latent spaces, our method allows inversion to any latent space along the ODE trajectory while preserving image integrity. This decouples the noise injection level from the depth of inversion. We use the ODE solver from DDIMs (Song, Meng, and Ermon 2020) in our experiments.

The primary motivation for inverting the degraded image is the model’s ability to process out-of-distribution images. Direct estimation of the score on degraded images is impractical due to the poor performance of the score estimation on OoD data. By mapping the corrupted input back to the latent space, we obtain more accurate estimates within a distribution closely resembling a multivariate Gaussian. This concept was foundational to SDEdit; however, their reliance on noise injection prevented full inversion of the diffusion process without losing information from the observation.

<html><body><table><tr><td>Require: N(Langevin iterations),∈(step-size),xo (Obser- vation),L (L-th latent-space),n (size of the confidence</td></tr><tr><td>interval.) x L,0 = ODE_SOLVERforwardo→L(ClipcBCn(xo))</td></tr><tr><td>fork=O toN-1do XL,k+1 = xL,k -∈·S(xL,k,L)+√2∈·η,where</td></tr><tr><td>n~N(0,I) end for x0 = ODE_SOLV ERbackwardL-→(x L,N)</td></tr></table></body></html>

Langevin Dynamics in Latent Spaces There exists a direct correspondence between DDPM, $\epsilon _ { \theta }$ , and Noise Conditional Score Network, sθ, such that sθ(xt, t) = − ϵθ(σxt,t) .

Building upon that, we propose to perform gradient-update in our latent spaces utilizing Langevin dynamics as in equation (1) to increase the likelihood of our latent representation. The method is described in the Appendix D, Algorithm 2. Analogous to SDEdit and contrasting with alternative methods, our editing method can be tailored to prioritize either realism or fidelity by selecting the step size in the Langevin dynamics and the latent spaces where to optimize.

Our editing technique, relying on updates within a designated latent space, facilitates an extensive array of editing possibilities on the input image, as optimizing in one latent space yields distinct outcomes compared to optimizing in another. Whereas SDEdit provides a singular hyper-parameter to govern the editing process, our method bifurcates this control mechanism into two distinct parameters: the step size in the updates and the choice of latent space for optimization. This dual-parameter approach enables our editing method to equal SDEdit’s performance on tasks where the latter is effective and to outperform in tasks that are unattainable for SDEdit.

# Confidence Based Clipping (CBC)

Here, we present a clipping method for the latent codes applied during the encoding process that does not depend on the prediction or the original sample. The proof is available in Appendix B.

Proposition 1. Let $\Phi$ be the cumulative distribution function of $\bar { \mathcal { N } } ( 0 , \mathcal { T } )$ and let $x _ { 0 } \in [ - 1 , 1 ]$ . For $\alpha _ { t } \in [ 0 , 1 ]$ , $\forall t \in [ 0 , 1 ]$ , assume that $x _ { t } \sim \mathcal { N } ( \sqrt { \alpha _ { t } } \cdot \alpha _ { 0 } , \sqrt { 1 - \alpha _ { t } } \cdot \mathcal { T } )$ . Then, for all $\eta$ :

$$
\begin{array} { r } { \mathcal { P } ( x _ { t } \in [ - \sqrt { \alpha _ { t } } - \eta \cdot \sqrt { 1 - \alpha _ { t } } , \sqrt { \alpha _ { t } } + \eta \cdot \sqrt { 1 - \alpha _ { t } } ] ) \ge \Phi ( \eta ) - \Phi ( - \eta ) . } \end{array}
$$

Specifically, for $\eta = 2$ :

$$
\mathcal { P } ( x _ { t } \in [ - \sqrt { \alpha _ { t } } - 2 \cdot \sqrt { 1 - \alpha _ { t } } , \sqrt { \alpha _ { t } } + 2 \cdot \sqrt { 1 - \alpha _ { t } } ] ) \geq 0 . 9 5 .
$$

During the encoding process, we propose to clip the latent codes using a confidence interval derived from Proposition 1.

![](images/0482c2bc6fd7a407d887dd98232a177e70b43a939ceea053d3c5c86bc8badbf1.jpg)  
Figure 3: Visual comparison on CelebAHQ with various corruption types. CODE is the only method performing on all corruptions types, significantly improving over SDEdit on two complex corruptions, Fog and Contrast. Other baselines demonstrate lower versatility while requiring extra training.

Confidence-based clipping is performed as follows:

$x _ { t } ^ { c l i p p e d } = \mathbf { C l i p } ( x _ { t } , \operatorname* { m i n } = - \sqrt { \alpha _ { t } } - \eta \cdot \sqrt { 1 - \alpha _ { t } } , \operatorname* { m a x } = \sqrt { \alpha _ { t } } + \eta \cdot \sqrt { 1 - \alpha _ { t } } ) ,$ where $t$ is the timestep, $\alpha _ { t }$ is the predefined schedule of the DM, and $\eta$ is the chosen confidence parameter.

Similar to our editing method, CBC is agnostic to the input and suitable for blind restoration scenarios. We combine CBC with our ODE editing method to form our complete method, CODE, detailed in Algorithm 1. As shown in Figure 9, the two methods synergize efficiently. It is crucial to note that CBC cannot be used in combination with SDEdit.

# 4 Experiments

Setup We use open-source pre-trained DDPM models (Ho, Jain, and Abbeel 2020) from HuggingFace, specifically the EMA checkpoints of DDPM models trained on CelebA-HQ (Karras et al. 2018), LSUN-Bedroom, and LSUN-Church (Yu et al. 2016), all at $2 5 6 \mathrm { x } 2 5 6$ resolution. For all experiments, DDIM inversion (Song and Ermon 2020) with 200 steps is utilized. Enhancement follows the complete Algorithm 3 described in Appendix D. It is used with $N = 2 0 0$ Langevin iteration steps, a step size $\epsilon$ of $[ 1 0 ^ { - 2 } , 1 0 ^ { - 3 } ]$ for shallow latent spaces (up to $L = 4 0 \$ ), and $\bar { [ 1 0 ^ { - 5 } , 1 0 ^ { - 6 } ] }$ for deeper latent spaces $\mathit { \tilde { L } } > 1 0 0 )$ . We use $K \ : = \ : 4$ annealing steps and $\alpha = 0 . 8$ as the annealing coefficient. When activating CBC, we use $\eta = 1 . 7$ . A full description of the setup being used to automatically compute the metrics is provided in Appendix E. For SDEdit, samples are generated with $L$ in [300, 500, 700] steps.

We tested our approach on 47 corruption types, including 17 from (Hendrycks and Dietterich 2019) (noise, blur, weather, and digital artifacts) and 28 from (Mintun, Kirillov, and Xie 2021). The corruption codebases are publicly available1 2. Additionally, we introduced two masking types: masking entire vertical lines and random pixels with random colors. Unlike traditional masking in masked autoencoders (He et al. 2022), our method does not assume knowledge of masked pixels’ positions, posing a more realistic recovery task. CODE operates completely blind to the corruption type, with no knowledge of the specific task or affected pixels.

![](images/0002ede26d7b19c503bc87991d90a245afc0f086a01b32251abfc2421cdcf389.jpg)  
Figure 4: Visual comparison of general image restoration on various corruptions - LSUN

For each corruption type, we test on at least 500 corrupted images. For each image, we kept the best 4 samples generated based on PSNR with respect to the original non-degraded images.

Baselines Our main baseline is the domain-agnostic method SDEdit (Meng et al. 2021), the only one comparable to ours in terms of requirements and assumptions. On CelebAHQ, the performance is also qualitatively benchmarked against domain-specific SOTA models, namely CodeFormer (Zhou et al. 2022), GFPGAN (Wang et al. 2021a), and DiffBIR (Lin et al. 2023). We also conducted visual experiments on LSUN-Bedroom and LSUN-Church to demonstrate the efficacy of CODE over diverse domains similar to SDEdit in (Meng et al. 2021).

Evaluation Metrics We evaluate our results using PSNR, SSIM, LPIPS, and FID. PSNR and SSIM are measured against the corrupted image (input) to assess fidelity to the guidance. FID is used to evaluate the quality of our generated images. Given the absence of assumptions about the input and corruptions, a key metric is the trade-off between realism and fidelity—specifically, the gain in realism relative to a given loss in fidelity. To quantify this, we use L2 distance in the pixel space as a measure of fidelity and FID as a measure of realism, plotting them against each other in Figure 5. Additionally, we report LPIPS with respect to the original, non-degraded image (source) to assess reconstruction quality. This metric is particularly informative for evaluating each corruption individually, as it also reflects the complexity of the corruption, with detailed results provided in Appendix G.

Results We present a brief qualitative comparison of results in Figure 3 to showcase that most methods, without further assumptions, cannot perform properly. In the vast majority of scenarios involving severe corruption like contrast, random

<html><body><table><tr><td>Inputs</td><td>0.48 (0.35)</td><td></td><td></td><td>143.49 (96.31)</td></tr><tr><td>SDEdit</td><td>0.32 (0.13)</td><td>0.46 (0.21)</td><td>18.74 (3.92)</td><td>47.84 (42.29)</td></tr><tr><td>CODE</td><td>0.30 (0.12)</td><td>0.49 (0.22)</td><td>19.61 (4.66)</td><td>30.66 (16.21)</td></tr></table></body></html>

Table 1: Average values of different metrics across the 47 considered corruptions, along with the standard deviations. CODE outperforms SDEdit on all metrics. CODE preserves a higher degree of fidelity while reaching a higher degree of realism using the same pre-trained model.

![](images/2778ac902fbdc45175dba3562477a10983b700c35b12f2158b0d35ab28f7f7c8.jpg)  
Figure 5: Comparison of realism-fidelity trade-off between SDEdit and CODE. Polynomial regression curves with shaded areas show one standard deviation. CODE produces more realistic images at the same fidelity. Both methods converge when the input distance is large, as they use the same pre-trained model.

pixel masking, or fog, only SDEdit and CODE can generate convincing images. For less intensive corruptions, which typically include erasing fine details or introducing minor noise, most baseline models tend to perform well. We provide extensive results in Appendix E. For quantitative metrics, we focus on SDEdit and CODE and compare them using the same pre-trained model on CelebA-HQ. Consequently, the differences come only from the way the pre-trained diffusion model is leveraged. Average metrics across the employed corruptions are detailed in Table 1. CODE outperforms SDEdit by $36 \%$ in FID-score while maintaining a fidelity to the input (PSNR-Input) $5 \%$ higher than SDEdit. Moreover, the standard deviation of the FID score highlights that SDEdit fails in certain cases while CODE is more stable. Finally, we report in Figure 5 the trade-off curves between fidelity and realism for both CODE and SDEdit. We performed a polynomial regression on CODE and SDEdit results to obtain such a curve. Both methods offer hyper-parameters to control such trade-offs. However, we highlight that CODE offers a better possibility. Overall, CODE generates more realistic outputs for a given degree of fidelity.

![](images/7fd3cb7756fe288c5ce6517d0cc8d6302a68f45cbe0f665fea1e5614131cf84e.jpg)  
Figure 6: Analysis of Langevin Dynamics convergence. As the number of updates increases, the output realism increases until convergence and then stabilizes.

# 5 Ablation Study

# Analysis of Hyperparameters

Number of Updates. As shown in Figure 6, the number of update iterations conducted in a latent space is pivotal for ensuring convergence and reducing variability. In practice, we employed 300 steps in all our experiments.

Step Size. The step size emerges as a critical parameter. A smaller step size results in high fidelity to the input and low variability among generated samples, albeit compromising realism. Conversely, an increased step size enhances realism and variability, as depicted in Figure 7a. As the number of updates is fixed in our experiments, the step size is what governs the size of the explored neighborhood around the input image. As a result, its impact is related to the amount of noise injected in SDEdit.

Latent Space Choice. The choice of latent space significantly influences the type of changes made during updates. As shown in Figure 7b, updates in a shallow latent space lead to minor but detailed and realistic modifications. In contrast, updates in deeper latent spaces can cause more significant or complex changes. Interestingly, regarding stroke guidance, optimization in the deepest latent space led to the addition of text and lines to the image. This suggests that the training set likely contained numerous images with these text and lines, implying that their inclusion by the model significantly enhanced the image’s likelihood. Empirically, we found that the deeper the latent space, the less the notion of distance is close to an L2 pixel-based distance.

The optimal latent space is not one-size-fits-all but depends on the specific input being processed. For complex corruptions, using a mix of updates in different latent spaces proves most effective. On the other hand, shallow latent spaces are best for addressing simple corruptions like blur. This ability to independently select the latent space without affecting other parameters, such as the level of noise injection, is a key strength of our editing method. We disentangle what was previously a single parameter into multiple, allowing for tailored optimization on a per-sample basis.

Fidelity. Our editing method is anchored in the corrupted sample, hence the generation is very impacted by the variations in the corruption. As shown in Figure 8, the outputs are faithful to the corrupted image and do not map to a single ground-truth image.

![](images/0740a42b3588a31e6cf6f1f01f171019ed4d07ff5f05011a7cc0180ac218df36.jpg)  
Figure 7: (a) Impact of step size on sample diversity and realism using CODE: Larger steps increase diversity but reduce fidelity. (b) Impact of latent space choice on the quality and characteristics of generated images across different corruption types.

# Ablation Study of Confidence-based clipping

Impact of Confidence-Based Clipping In this section, we study the impact of the confidence parameter $\eta$ in CBC. As we reduce $\eta$ , the interval shrinks, keeping only the most likely pixel values. We propose to encode an image using DDIM with different values of $\eta$ and decode it back to see the result. We study this in the case of in-distribution images and of corrupted images. Results can be seen in Figure 9b. A smaller $\eta$ results in a loss of fine-grained details, a shift of the average tone, and the removal of unlikely pixels such as masked pixels. When applied to corrupted samples, CBC proves efficient in removing part of the noisy artifact while keeping most of the image structure. Interestingly, CBC also stabilizes the DDIM inversion, which might sometimes be inconsistent. This allows for fewer steps in the encodingdecoding procedure and speeds up the whole editing process.

![](images/6b4aa40aa57577b9a4b6e8d10438960269df1c7dc7266ea6e0bdc6018ed0a30c.jpg)  
Figure 8: CODE outputs adapt faithfully to variations in the image input.

Ablation Study We propose to study the impact of each block in CODE while keeping SDEdit for comparison. Qualitative results are visible in Figure 9a. While both our editing method alone (w/o CBC) and SDEdit excel at adding extra fine-grained details, they fail at handling unknown masks or color shits efficiently. On the contrary, CBC basically fails at adding extra details but successfully recovers certain color shifts or masked areas. As a result, combining both into CODE leads to powerful synergies. Quantitative results can be seen in Table 2.

LPIPS-Source ↓ SSIM-Input ↑ PSNR-Input ↑ FID ↓   
Table 2: Confidence-Based Clipping ablation, metrics are averaged across all corruptions.   

<html><body><table><tr><td>Inputs</td><td>0.48 (0.35)</td><td></td><td>-</td><td>143.5 (96.3)</td></tr><tr><td>DDIM</td><td>0.52 (0.32)</td><td>0.89 (0.08)</td><td>30.7 (5.3)</td><td>152.2 (88.4)</td></tr><tr><td>DDIMw/CBC</td><td>0.43 (0.19)</td><td>0.73 (0.17)</td><td>23.5 (5.3)</td><td>90.1 (49.5)</td></tr><tr><td>CODE w/o CBC</td><td>0.31 (0.11)</td><td>0.48 (0.22)</td><td>19.4 (4.5)</td><td>34.75 (14.6)</td></tr><tr><td>CODE w/ CBC</td><td>0.30 (0.12)</td><td>0.49 (0.22)</td><td>19.6 (4.7)</td><td>30.65 (16.2)</td></tr></table></body></html>

Discussion. While CODE offers enhanced versatility and control in editing, it introduces greater complexity compared to SDEdit. SDEdit’s tuning is straightforward, with binary success or failure outcomes, whereas CODE’s dual hyperparameter framework requires a more extensive grid search, increasing the search complexity quadratically. However, this added complexity enables CODE to achieve better results across a wider range of scenarios.

# 6 Conclusion

We introduce Confident Ordinary Differential Editing, a novel approach for guided image editing and synthesis that handles OoD inputs and balances realism and fidelity. Our method eliminates the need for retraining, finetuning, data augmentation, or paired data, and it integrates seamlessly with any pre-trained Diffusion Model. CODE excels in addressing (b) DDIM inversion using CBC with different values for confidence parameter $\eta$ .

![](images/1b15421b98098fa1bcd91873c2aeb78fbd6550aba815c20b3c853985ec516bdd.jpg)  
Figure 9: (a) Ablation Study (b) DDIM inversion using CBC with different values for the confidence parameter $\eta$ .

a wide array of corruptions, outperforming existing methods that often rely on handcrafted features. As an evolution of SDEdit, our approach provides enhanced control, variety, and capabilities in editing by disentangling the original method and introducing additional hyperparameters. These new parameters not only offer deeper insights into the functioning of Diffusion Models’ latent spaces but also enable more diverse editing strategies. Furthermore, we introduce a Confidence-Based Clipping method that synergizes effectively with our editing technique, allowing the disregard of unlikely pixels or areas in a completely agnostic manner. Finally, our extensive study of the different components at play offers a greater understanding of the underlying mechanics of diffusion models, enriching the field’s knowledge base. Our findings reveal that CODE surpasses SDEdit in versatility and quality while maintaining its strengths across various tasks, including stroke-based editing. We hope our work inspires further innovations in this domain, akin to the transformative impact of GAN inversion. Looking ahead, we see potential in automating the editing and hyperparameter search processes and exploring synergies with text-to-image synthesis.