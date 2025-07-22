# LazyDiT: Lazy Learning for the Acceleration of Diffusion Transformers

Xuan Shen1\*, Zhao Song2, Yufa Zhou3, Bo Chen4, Yanyu $\mathbf { L i } ^ { 1 }$ , Yifan $\mathbf { G o n g ^ { 1 } }$ , Kai Zhang2, Hao Tan2, Jason Kuen2, Henghui Ding5, Zhihao $\mathbf { S h u } ^ { 6 }$ , Wei $\mathbf { N i u } ^ { 6 }$ , Pu Zhao1†, Yanzhi $\mathbf { W a n g ^ { 1 \dagger } }$ , Jiuxiang $\mathbf { G u ^ { 2 \dag } }$

1Northeastern University 2Adobe Research 3University of Pennsylvania 4Middle Tennessee State University 5Fudan University 6University of Georgia {shen.xu, p.zhao, yanz.wang} $@$ northeastern.edu, wniu $@$ uga.edu, jigu@adobe.com

# Abstract

Diffusion Transformers have emerged as the preeminent models for a wide array of generative tasks, demonstrating superior performance and efficacy across various applications. The promising results come at the cost of slow inference, as each denoising step requires running the whole transformer model with a large amount of parameters. In this paper, we show that performing the full computation of the model at each diffusion step is unnecessary, as some computations can be skipped by lazily reusing the results of previous steps. Furthermore, we show that the lower bound of similarity between outputs at consecutive steps is notably high, and this similarity can be linearly approximated using the inputs. To verify our demonstrations, we propose the LazyDiT, a lazy learning framework that efficiently leverages cached results from earlier steps to skip redundant computations. Specifically, we incorporate lazy learning layers into the model, effectively trained to maximize laziness, enabling dynamic skipping of redundant computations. Experimental results show that LazyDiT outperforms the DDIM sampler across multiple diffusion transformer models at various resolutions. Furthermore, we implement our method on mobile devices, achieving better performance than DDIM with similar latency.

# 1 Introduction

Diffusion models (Ho, Jain, and Abbeel 2020; Rombach et al. 2022a; Song et al. 2020; Song and Ermon 2019; Dhariwal and Nichol 2021; Zhan et al. 2024b) have become dominant in image generation research, attributable to their remarkable performance. U-Net (Ronneberger, Fischer, and Brox 2015) is a widely used backbone in diffusion models, while transformers (Vaswani et al. 2017) are increasingly proving to be a strong alternative. Compared to U-Net, transformer-based diffusion models have demonstrated superior performance in high-fidelity image generation (Peebles and Xie 2023; Bao et al. 2023), and their efficacy extends to video generation as well (Lu et al. 2023; Chen et al. 2023; Lab and etc. 2024; Zheng et al. 2024). This highlights the versatility and potential of transformers in advancing generative tasks across different media. Despite the notable scalability advantages of transformers, diffusion transformers face major efficiency challenges. The high deployment costs and the slow inference speeds create the significant barriers to their practical applications (Zhan et al. 2024a,c, 2021; Wu et al. 2022; Li et al. 2022; Yang et al. 2023a), which motivates us to explore their acceleration methods.

The increased sampling cost in diffusion models stems from two main components: the numerous timesteps required and the computational expense associated with each inference step. To improve sampling efficiency, existing methods generally fall into two categories: reducing the total number of sampling steps (Song, Meng, and Ermon 2020; Liu et al. 2022; Bao et al. 2022; Zhan et al. 2024b) or lowering the computational cost per step (Yang et al. 2023b; He et al. 2023). Several works (Yin et al. 2024; Luo et al. 2024; Salimans and Ho 2022) employ distillation techniques to reduce the number of sampling steps. Conversely, works (Li et al. 2023b; Kim et al. 2023; Fang, Ma, and Wang 2023; Li et al. 2023a) utilize compression techniques to streamline diffusion models. Recently, some studies have introduced caching mechanisms into the denoising process (Ma, Fang, and Wang 2024; Wimbauer et al. 2023) to accelerate the sampling. However, previous compression approaches of diffusion models have primarily focused on optimizing UNet, leaving transformer-based models largely unexplored.

Leveraging characteristic of uniquely structured, prior compression works (Zhang et al. 2024; Raposo et al. 2024; Fan, Grave, and Joulin 2019; Kong et al. 2022, 2023; Zhang et al. 2022; Li et al. 2023c; Zhao et al. 2024; Shen et al. 2024d,a,c,b, 2023b) have concentrated on techniques such as layer pruning and width pruning. However, we observe that removing certain layers results in a significant performance drop. This indicates the redundancy in diffusion transformers primarily occurs between sampling steps rather than the model architecture. This finding forms basis for exploring methods to reduce frequency of layer usage, aiming to decrease computational costs and accelerate the diffusion.

In this paper, we propose LazyDiT, a cache-based approach designed to dynamically reduce computational costs and accelerate the diffusion process. We begin by analyzing the output similarity between the current and previous steps, identifying that the lower bound of this similarity is notably high during the diffusion process. Then, we delve deeper into the similarity using a Taylor expansion around the current input, revealing that the similarity can be linearly approximated. Building on the theoretical analysis, we implement a lazy learning framework by introducing linear layers before each Multi-Head Self-Attention (MHSA) and pointwise feedforward (Feedforward) module. These added layers are trained with the proposed lazy loss to learn whether the subsequent module can be lazily bypassed by leveraging the previous step’s cache. Compared to the DDIM sampler, extensive experiments demonstrate that our method achieves superior performance with similar computational costs. As shown in Figure 1, by lazily skipping $50 \%$ of the computations, our method achieves nearly the same performance as the original diffusion process. We also profile the latency of the diffusion process on mobile devices to offer a detailed comparison with the DDIM sampler. Our results show our superior image generation quality than DDIM with similar latency. Our main contributions are summarized as follows,

![](images/46d956102f1137c0210523ed4b6c0137b6185272ec44af6ecb00febc220ba7ee.jpg)  
Figure 1: Image generated by DiT-XL/2 in $5 1 2 \times 5 1 2$ and $2 5 6 \times 2 5 6$ resolutions when lazily skipping $50 \%$ computation. The upper rows display results from original model and the lower rows showcase outcomes of our method. Our method generates distinct lighting effects for background and color compared to the baseline, as demonstrated in dog and marmot, respectively.

• We explore the redundancy in diffusion process by evaluating the similarity between module outputs at consecutive steps, finding that the lower bound of the similarity is notably high.   
• We establish that the lazy skip strategy can be effectively learned through a linear layer based on the Taylor expansion of similarity.   
• We propose a lazy learning framework to optimize the diffusion process in transformer-based models by lazily bypassing computations using the previous step’s cache.   
• Experiments show that the proposed method achieves

better performance than DDIM sampler. We further implement our method on mobile devices, showing that our method is a promising solution for real-time generation.

# 2 Related Work

Transformer-Based Diffusion Models. Recent works such as GenVit (Yang et al. 2022), U-Vit (Bao et al. 2023), DiT (Peebles and Xie 2023), LlamaGen (Sun et al. 2024), and MAR (Li et al. 2024a) have incorporated transformers (Vaswani et al. 2017) into diffusion models, offering a different approach compared to the traditional U-Net architecture. GenViT incorporates the ViT (Dosovitskiy et al. 2021; Li et al. 2024c,b) architecture into DDPM, while UViT further enhances this approach by introducing long skip connections between shallow and deep layers. DiT demonstrates the scalability of diffusion transformers, and its architecture has been further utilized for text-to-video generation tasks, as explored in works (OpenAI 2024). LlamaGen introduces autoregressive models to image generation, verifying the effectiveness of the ’next-token prediction’ in this domain. Thus, it is crucial to explore efficient designs for those large models to accelerate the diffusion process.

Acceleration for Diffusion Models. High-quality image generation with diffusion models necessitates multiple sampling steps, leading to increased latency (Gong et al. 2024; Shen et al. 2023a). To enhance efficiency, DDIM (Song, Meng, and Ermon 2020) extends original DDPM to nonMarkovian cases when DPM-Solver (Lu et al. 2022) advances the approximation of diffusion ODE solutions. Regarding the works that require fine-tuning, such as (Lin, Wang, and Yang 2024; Yin et al. 2024), they employ distillation techniques to effectively reduce the number of sampling steps. Additionally, reducing the computational workload for each diffusion step is a widely adopted, strategy to enhance the efficiency of the diffusion process. Various approaches have been explored, such as works (Fang, Ma, and Wang 2023; Castells et al. 2024; Wang et al. 2024; Zhang et al. 2024) that adopt weight pruning techniques, works (He et al. 2023; Li et al. 2023a) that employ quantization techniques, and even works (Kim et al. 2023; Li et al. 2023b) that redesign the architecture of diffusion models.

# 3 Methodology

# 3.1 Preliminaries

Notations. We use $\mathbb { E } [ ]$ to denote the expectation. We use ${ \mathbf { 1 } } _ { n }$ to denote a length- $n$ vector where all the entries are ones. We use $x _ { i , j }$ to denote the $j$ -th coordinate of $x _ { i } \in \mathbb { R } ^ { n }$ . We use $\| { \boldsymbol { x } } \| _ { p }$ to denote the $\ell _ { p }$ norm of a vector $x$ . We use $\| A \|$ to denote the spectral norm for a matrix $A$ . We use $a \circ b$ to denote the element-wise product of two vectors $a , b$ . For a tensor $X \in \mathbb { R } ^ { B \times N \times D }$ and a matrix $U \in \mathbb { R } ^ { D \times d _ { 1 } }$ , we define $Y = X \cdot U \in \mathbb { R } ^ { B \times N \times d _ { 1 } }$ . For a matrix $V \in \mathbb { R } ^ { d _ { 2 } \times B }$ and a tensor $X \in \mathbb { R } ^ { B \times N \times D }$ , we define $Z = V \cdot X \in \mathbb { R } ^ { d _ { 2 } \times N \times D }$ . For a square matrix $A$ , we use $\operatorname { t r } [ A ]$ to denote the trace of $A$ . For two matrices $X , Y$ , the standard inner product between matrices is defined by $\langle X , Y \rangle : = \mathrm { t r } [ X ^ { \top } Y ]$ . We use $U ( a , b )$ to denote a uniform distribution. We use ${ \mathcal { N } } ( \mu , \sigma ^ { 2 } )$ to denote a Gaussian distribution. We define cosine similarity as f (X, Y ) = ∥Xtr∥[FX·⊤∥Y  ]∥F for matrices $X , Y$ .

Diffusion Formulation. Diffusion models (Ho, Jain, and Abbeel 2020; Song et al. 2020) operate by transforming a sample $x$ from its initial state within a real data distribution $p _ { \beta } ( x )$ into a noisier version through diffusion steps. For a diffusion model $\epsilon _ { \theta } ( \cdot )$ with parameters $\theta$ , the training objective (Sohl-Dickstein et al. 2015) can be expressed as follows,

$$
\displaystyle \operatorname* { m i n } _ { \theta \mathrm { \Lambda } _ { t \sim U [ 0 , 1 ] , x \sim p _ { \beta } ( x ) , \epsilon \sim \mathcal { N } ( 0 , I ) } } \| \epsilon _ { \theta } ( t , z _ { t } ) - \epsilon \| _ { 2 } ,
$$

where $t$ denotes the timestep; $\epsilon$ denotes the ground-truth noise; $z _ { t } = \alpha _ { t } \cdot x + \sigma _ { t } \cdot \epsilon$ denotes the noisy data; $\alpha _ { t }$ and $\sigma _ { t }$ are the strengths of signal and noise.

For comparison purposes, this paper adopts Denoising Diffusion Implicit Models (DDIM) (Song, Meng, and Ermon 2020) as sampler. The iterative denoising process from timestep $t$ to the previous timestep $t ^ { \prime }$ is described as follows,

$$
z _ { t ^ { \prime } } = \alpha _ { t ^ { \prime } } \cdot \frac { z _ { t } - \sigma _ { t } \epsilon _ { \theta } ( t , z _ { t } ) } { \alpha _ { t } } + \sigma _ { t ^ { \prime } } \cdot \epsilon _ { \theta } ( t , z _ { t } ) ,
$$

where $z _ { t ^ { \prime } }$ is iteratively fed to $\epsilon _ { \theta } ( \cdot )$ until $t ^ { \prime }$ becomes 0.

Latent Diffusion Models. The Latent Diffusion Model (LDM) (Rombach et al. 2022b) decreases computational demands and the number of steps with the latent space, which is obtained by encoding with a pre-trained variational autoencoder (VAE) (Sohl-Dickstein et al. 2015). Besides, the classifier-free guidance (CFG) (Ho and Salimans 2022) is adopted to improve quality as follows,

$$
\widehat { \epsilon } _ { \theta } ( t , z _ { t } , c ) = w \cdot \epsilon _ { \theta } ( t , z _ { t } , c ) - ( w - 1 ) \cdot \epsilon _ { \theta } ( t , z _ { t } , c _ { \phi } ) ,
$$

whebre $\epsilon _ { \theta } ( t , z _ { t } , c _ { \phi } )$ denotes unconditional prediction with null text; $w$ denotes guidance scale which is used as control of conditional information and $w \ge 1$ .

# 3.2 Similarity Establishment

Let $B$ be the number of batches, $N$ be the number of patches, $D$ be the hidden dimension, $T$ be the number of diffusion steps, and $L$ be the number of model layers in diffusion transformers. Let $f ( \cdot , \cdot ) : \mathbb { R } ^ { B \times N \times D } \times \mathbb { R } ^ { \breve { B } \times N \times D } $ $[ 0 , 1 ]$ be the function that estimate the similarity between two variables. Let $\mathcal { F } _ { l } ^ { \Phi } ( \cdot ) : \mathbb { R } ^ { B \times N \times D }  \mathbb { R } ^ { B \times N \times D }$ be the Multi-Head Self-Attention (MHSA) / pointwise feedforward (Feedforward) module at $l$ -th layer where $\Phi \in \{ \mathrm { a t t n } , \mathrm { f e e d } \}$ for $l \in [ L ]$ . We use normalized $\mathbf { \bar { \chi } } _ { l , t } ^ { \Phi } \in \mathbb { R } ^ { B \times N \times \mathbf { \bar { \chi } } _ { D } }$ to denote the input hidden states with scaling factor $a _ { t }$ and shifting factor $b _ { t }$ at timestep $t$ in the $l$ -th layer for $t \in [ T ] , l \in [ L ]$ . We denote the output at $l$ -th layer and $t$ -th timestep as $Y _ { l , t } ^ { \Phi }$ .

iInmgparcotceoefdSs,ctahlienignpauntd fSfheirfetinnceg.betAwseepnr,o $\bar { X } _ { l , t - 1 } ^ { \Phi }$ eande $X _ { l , t } ^ { \Phi }$ grows. In contrast, the application of scaling and shifting transformations introduces an alternate problem formulation, potentially affecting the input distance in a manner that requires a different analytical approach. In detail, the diffusion transformer architecture incorporates scaling and shifting mechanisms in the computation of both the MHSA and Feedforward modules, utilizing the embeddings of the timestep, $\mathrm { e m d } ( t ) \in \mathbb { R } ^ { D }$ , and the condition, $\mathrm { e m d } ( \bar { c } ) \in \mathbb { R } ^ { D }$ . We define $y _ { t } = \operatorname { S i L U } ( \operatorname { e m d } ( t ) + \operatorname { e m d } ( c ) ) \in \mathbb { R } ^ { D }$ , with corresponding scaling and shifting factors defined as follows,

• Scaling factor: $a _ { t } = W _ { l , a } \cdot y _ { t } + v _ { l , a } \in \mathbb { R } ^ { D }$ ;   
• Shifting factor: $b _ { t } = W _ { l , b } \cdot y _ { t } + v _ { l , b } \in \mathbb { R } ^ { D }$ ;

where $W _ { l , a } , W _ { l , b } \in \mathbb { R } ^ { D \times D } , v _ { l , a } , v _ { l , b } \in \mathbb { R } ^ { D }$ are the linear projection weight and bias, respectively.

Meanwhile, we define broadcasted matrices to represent the scaling and shifting factors, ensuring the alignment with the implementation of diffusion transformers as follows,

• Let $A _ { t } \in \mathbb { R } ^ { N \times D }$ be defined as the matrix that all rows are $a _ { t }$ , i.e. $( A _ { t } ) _ { i } : = a _ { t }$ for $i \in [ N ]$ , and $A _ { t - 1 } , B _ { t } , B _ { t - 1 }$ can be defined as the same way.

Then, we deliver the demonstration showing that there exist $y _ { t } , y _ { t - 1 }$ such that, after scaling and shifting, the distance between−inputs $X _ { l , t - 1 } ^ { \Phi } , X _ { l , t } ^ { \Phi }$ , defined in the Left Hand Side of the following Eq. (1), can be constrained within a small bound. Given $a _ { t }$ and $b _ { t }$ are both linear transformation of $y _ { t }$ , the problem reduces to demonstrating the existence of vectors $a _ { t } , b _ { t } , a _ { t - 1 }$ , and $b _ { t - 1 }$ that satisfy following conditions,

$$
\begin{array} { r } { \| \big ( A _ { t - 1 } \circ X _ { l , t - 1 } ^ { \Phi } + B _ { t - 1 } \big ) - \big ( A _ { t } \circ X _ { l , t } ^ { \Phi } + B _ { t } \big ) \| \leq \eta , } \end{array}
$$

where $\eta \in ( 0 , 0 . 1 )$ . And Eq. (1) is equivalent as follows,

$$
\| A \circ X _ { l , t - 1 } ^ { \Phi } + B \circ X _ { l , t } ^ { \Phi } + C \| _ { F } \leq \eta ,
$$

where $A : = A _ { t - 1 } , B : = - A _ { t } , C : = B _ { t - 1 } - B _ { t }$ .

We identify that there exists $a , b$ and $c$ such that Eq. (2) holds, and the detailed demonstration and explanation are included in Lemma 12 at Appendix C.2. Subsequently, we generate the following theorem,

Theorem 1 (Scaling and shifting, informal version of Theorem 13 at Appendix C.2). There exist time-variant and condition-variant scalings and shiftings such that the distance between two inputs at consecutive steps for MHSA or Feedforward is bounded.

Similarity Lower Bound. To leverage the cache from the previous step, we begin by investigating the cache mechanism in transformer-based diffusion models. This analysis focuses on the similarity between the current output and the preceding one, providing insights into the efficacy of reusing cached information. One typical transformer block consists of two primary modules: MHSA module and Feedforward module. Both modules are computationally expensive, making them significant contributors to the overall processing cost. Thus, we aim to examine the output similarities between the current and previous steps for both modules. By identifying cases of high similarity, we can skip redundant computations, thereby reducing the overall computational cost. In practice, we employ the cosine similarity $\bar { \boldsymbol { f } } ( \cdot , \cdot )$ for the computation of similarity as follows,

$$
f ( Y _ { l , t - 1 } ^ { \Phi } , Y _ { l , t } ^ { \Phi } ) = \frac { \mathrm { t r } [ ( Y _ { l , t - 1 } ^ { \Phi } ) ^ { \top } \cdot Y _ { l , t } ^ { \Phi } ] } { \Vert Y _ { l , t - 1 } ^ { \Phi } \Vert _ { F } \cdot \Vert Y _ { l , t } ^ { \Phi } \Vert _ { F } } .
$$

Inspired by the Lipschitz property (Trench 2013), we transform the similarity measure into a distance metric, defined as Dist : $: = \| Y _ { l , t - 1 } ^ { \Phi } - Y _ { l , t } ^ { \Phi } \|$ , to simplify the analysis of similarity variations. According to Fact 7 in Appendix B.2, similarity function $f ( \cdot , \cdot )$ is further transformed as follows,

$$
f ( Y _ { l , t - 1 } ^ { \Phi } , Y _ { l , t } ^ { \Phi } ) = 1 - \mathrm { D i s t } / 2 .
$$

For convenience, we further define the hidden states after scaling and shifting as $Z _ { l , t } ^ { \Phi } : = A _ { t } \circ X _ { l , t } ^ { \Phi } + B _ { t }$ . Meanwhile, building upon the Lemma H.5 of (Deng et al. 2023), we further derive the upper bounds for the distance Dist for either the MHSA or Feedforward modules as follows,

$$
\mathrm { D i s t } \leq C \cdot \Vert Z _ { l , t - 1 } ^ { \Phi } - Z _ { l , t } ^ { \Phi } \Vert .
$$

where $C$ is the Lipschitz constant related to the module.

Subsequently, with Theorem 1, we integrate Eq. (1) and derive the bound of the similarity as follows,

$$
f ( Y _ { l , t - 1 } ^ { \Phi } , Y _ { l , t } ^ { \Phi } ) \geq 1 - \alpha ,
$$

for $\alpha : = O ( C ^ { 2 } \eta ^ { 2 } )$ and $\eta$ is sufficiently small in practice.

Thus, we deliver Theorem 2 as below, which asserts that the lower bound of the output similarity between the two consecutive sampling steps is high.

Theorem 2 (Similarity lower bound, informal version of Theorem 18 at Appendix C.4). The lower bound of the similarity $f ( Y _ { l , t - 1 } ^ { \Phi } , \bar { Y _ { l , t } ^ { \Phi } } )$ between the outputs at timestep $t - 1$ and timestep $t$ is high.

Linear Layer Approximation. The similarity can be approximated using the inputs from either the current step $Z _ { l , t } ^ { \Phi }$ or previous one ZlΦ,t 1 , due to its mathematical symmetry according to Eq. (3). We then apply the Taylor expansion around $\bar { Z } _ { l , t } ^ { \Phi }$ as follows,

$$
\begin{array} { r l } & { f ( Y _ { l , t - 1 } ^ { \Phi } , Y _ { l , t } ^ { \Phi } ) } \\ & { = \mathrm { t r } [ ( Y _ { l , t - 1 } ^ { \Phi } ) ^ { \top } \cdot ( 0 + J \cdot Z _ { l , t } ^ { \Phi } + O ( 1 ) ) ] , } \end{array}
$$

where $J$ is the Jacobian matrix.

Through Taylor expansion, we identify that there exists a $W _ { l } \in \mathbb { R } ^ { \breve { D } \times D _ { \mathrm { o u t } } }$ along with $Z _ { l , t } ^ { \Phi }$ such that the similarity can be linearly approximated with certain error as follows,

$$
\langle W _ { l } ^ { \Phi } , Z _ { l , t } ^ { \Phi } \rangle = f ( Y _ { l , t - 1 } ^ { \Phi } , Y _ { l , t } ^ { \Phi } ) + O ( 1 ) ,
$$

where the detailed proof is included in Appendix C.5 Eq.(9). Then, we generate the Theorem 3 as follows,

Theorem 3 (Linear layer approximation, informal version of Theorem 19 at Appendix C.5). The similarity function $f ( \cdot , \cdot )$ can be approximated by a linear layer with respect to the current input, i.e. $f ( Y _ { l , t - 1 } ^ { \Phi ^ { \cdot } } , Y _ { l , t } ^ { \Phi } ) = \langle \dot { W } _ { l } ^ { \Phi } , Z _ { l , t } ^ { \Phi } \rangle$ where $W _ { l } ^ { \Phi }$ is the weight of a linear layer for MHSA or Feedforward in the l-th layer of diffusion model.

In our experiments, we utilize the lazy learning framework to obtain $W _ { l } ^ { \Phi }$ , and we set $D _ { \mathrm { o u t } } = 1$ to minimize computational cost. The details of this approach are explained in the following section.

![](images/b7a2bd80963f9677bd3a3e2bb0c1c404168f4572e51c89f95ca45e8a501bd23b.jpg)  
Figure 2: Overview framework. We skip the computation of MHSA or Feedforward by calling the previous step cache.

# 3.3 Lazy Learning

As illustrated in Figure 2, we incorporate lazy learning linear layers before each MHSA module and Feedforward module to learn the similarity. The MHSA module or Feedforward module is bypassed and replaced with the cached output from the previous step if the learned similarity is below 0.5. The input scale, input shift, output scale, and residual connections remain unchanged from the normal computation. The training details and the calculation of lazy ratio are outlined in the following paragraphs.

Training Forward. Assume we add linear layers with weights $\breve { W } _ { l } ^ { \Phi } \in \mathbb { R } ^ { D \times 1 }$ for each module $\mathcal { F } _ { l } ^ { \Phi } ( \cdot )$ at $l$ -th layer in the model. For input hidden states $X _ { l , t } ^ { \Phi } \in \mathbb { R } ^ { B \times N \times D }$ for the module at $l$ -th layer and $t$ -th step, the similarity $s _ { l , t } ^ { \Phi } \in \mathbb { R } ^ { B }$ of the module is computed as follows,

$$
s _ { l , t } ^ { \Phi } = \mathrm { s i g m o i d } ( ( Z _ { l , t } ^ { \Phi } \cdot W _ { l } ^ { \Phi } ) \cdot { \bf 1 } _ { N } ) .
$$

We then define the forward pass of the MHSA module or Feedforward module at $l$ -th layer and $t$ -th step with the input $X _ { l , t } ^ { \Phi }$ during the training progress as follows,

$$
\begin{array} { r l } & { Y _ { l , t } ^ { \Phi } = \mathrm { d i a g } ( \mathbf { 1 } _ { B } - s _ { l , t } ^ { \Phi } ) \cdot \mathcal { F } _ { l } ^ { \Phi } ( Z _ { l , t } ^ { \Phi } ) } \\ & { ~ + \mathrm { d i a g } ( s _ { l , t } ^ { \Phi } ) \cdot Y _ { l , t - 1 } ^ { \Phi } . } \end{array}
$$

Backward Loss. Alongside the diffusion loss for a given timestep $t$ during training, we introduce a lazy loss to encourage the model to be more lazy—relying more on cached computations rather than diligently executing the MHSA modules or Feedforward modules, as follows,

$$
\begin{array} { r l } & { \mathcal { L } _ { t } ^ { \mathrm { l a z y } } = \rho ^ { \mathrm { a t t n } } \cdot \displaystyle \frac { 1 } { B } \sum _ { l = 1 } ^ { L } \sum _ { b = 1 } ^ { B } ( 1 - ( s _ { l , t } ^ { \mathrm { a t t n } } ) _ { b } ) } \\ & { \quad \quad \quad + \ \rho ^ { \mathrm { f e e d } } \cdot \displaystyle \frac { 1 } { B } \sum _ { l = 1 } ^ { L } \sum _ { b = 1 } ^ { B } ( 1 - ( s _ { l , t } ^ { \mathrm { f e e d } } ) _ { b } ) , } \end{array}
$$

where the $\rho ^ { \mathrm { a t t n } }$ and $\rho ^ { \mathrm { f e e d } }$ denote the penalty ratio of MHSA module and Feedforward module, respectively.

We combine the lazy loss with diffusion loss and regulate $\rho ^ { \mathrm { a t t n } }$ and $\rho ^ { \mathrm { f e e d } }$ to control the laziness (i.e., number of skips with cache) of sampling with diffusion transformers.

Accelerate Sampling. After finishing the lazy learning with a few steps, we then accelerate the sampling during the diffusion process as follows,

$$
\begin{array} { r } { Y _ { l , t } ^ { \Phi } = \left\{ \begin{array} { l l } { \mathcal { F } _ { l } ^ { \Phi } ( Z _ { l , t } ^ { \Phi } ) , } & { \ s _ { l , t } ^ { \Phi } \leq 0 . 5 , } \\ { Y _ { l , t - 1 } ^ { \Phi } , } & { \ s _ { l , t } ^ { \Phi } > 0 . 5 , } \end{array} \right. } \end{array}
$$

where $\Phi \in \{ \mathrm { a t t n } , \mathrm { f e e d } \}$ can be either MHSA module or Feedforwa∈rd {module, and} the skip occurs when $s _ { l , t } ^ { \Phi } > 0 . 5$ .

Then, the lazy ratio $\Gamma ^ { \Phi } \in \mathbb { Z } ^ { B }$ of MSHA or Feedforward for $B$ batches during sampling is be computed as follows,

$$
\Gamma ^ { \Phi } = \frac { 1 } { L T } \sum _ { l = 1 } ^ { L } \sum _ { t = 1 } ^ { T } \left\lceil s _ { l , t } ^ { \Phi } - 0 . 5 \right\rceil .
$$

# 4 Experimental Results

# 4.1 Experiment Setup

Model Family. We validate the effectiveness of our method on both the DiT (Peebles and Xie 2023) and LargeDiT (Zhang et al. 2023) model families. Specifically, our experiments utilize the officially provided models including DiT-XL/2 $2 5 6 \times 2 5 6 )$ , DiT-XL/2 $( 5 1 2 \times 5 1 2 )$ ), LargeDiT-3B $2 5 6 \times 2 5 6 )$ ), and Large-DiT-7B $( 2 5 6 \times 2 5 6 )$ .

Lazy Learning. We freeze the original model weights and introduce linear layers as lazy learning layers before each MHSA and Feedforward module at every diffusion step. For various sampling steps, these added layers are trained on the ImageNet dataset with 500 steps, with a learning rate of 1e4 and using the AdamW optimizer. Following the training pipeline in DiT, we randomly drop some labels, assign a null token for classifier-free guidance, and set a global batch size of 256. The training is conducted on $8 \times$ NVIDIA A100 GPUs within 10 minutes.

Table 1: DiT model results on ImageNet $( \mathrm { c f g } { = } 1 . 5 )$ . ‘Lazy Ratio’ indicates the percentage of skipped MHSA and Feedforward modules during diffusion process.   

<html><body><table><tr><td>Method</td><td>#of Step</td><td>Lazy Ratio</td><td>FID ←</td><td>SFID</td><td>IS 个</td><td>Prec. 个</td><td>Rec. 个</td></tr><tr><td colspan="8">DiT-XL/2 (256×256)</td></tr><tr><td>DDIM</td><td>50</td><td>/</td><td>2.34</td><td>4.33</td><td>241.01</td><td>80.13</td><td>59.55</td></tr><tr><td>DDIM Ours</td><td>40 50</td><td>20% 1</td><td>2.39 2.37</td><td>4.28 4.33</td><td>236.26 239.99</td><td>80.10 80.19</td><td>59.41 59.63</td></tr><tr><td>DDIM</td><td>30</td><td>/</td><td>2.66 2.63</td><td>4.40 4.35</td><td>234.74 235.69</td><td>79.85</td><td>58.96</td></tr><tr><td>Ours DDIM</td><td>50 25</td><td>40% /</td><td>2.95</td><td>4.50</td><td>230.95</td><td>79.59 79.49</td><td>58.94 58.44</td></tr><tr><td>Ours DDIM</td><td>50 20</td><td>50% /</td><td>2.70 3.53</td><td>4.47 4.91</td><td>237.03 222.87</td><td>79.77 78.43</td><td>58.65 57.12</td></tr><tr><td>Ours DDIM</td><td>40 14</td><td>50% /</td><td>2.95 5.74</td><td>4.78 6.65</td><td>234.10 200.40</td><td>79.61 74.81</td><td>57.99 55.51</td></tr><tr><td>Ours DDIM</td><td>20 10</td><td>30% /</td><td>4.44 12.05</td><td>5.57 11.26</td><td>212.13 160.73</td><td>77.11 66.90</td><td>56.76 51.52</td></tr><tr><td>Ours DDIM</td><td>20</td><td>50% 1</td><td>6.75 34.14</td><td>8.53 27.51</td><td>192.39 91.67</td><td>74.35</td><td>52.43</td></tr><tr><td>Ours</td><td>7 10</td><td>30%</td><td>17.05</td><td>13.37</td><td>136.81</td><td>47.59 62.07</td><td>46.83 50.37</td></tr><tr><td></td><td></td><td>/</td><td>3.33</td><td>DiT-XL/2 (512×512)</td><td></td><td></td><td></td></tr><tr><td>DDIM DDIM</td><td>50 30</td><td>/</td><td>3.95</td><td>5.31 5.71</td><td>205.01 195.84</td><td>80.59 80.19</td><td>55.89 54.52</td></tr><tr><td>Ours</td><td>50</td><td>40%</td><td>3.67</td><td>5.65</td><td>202.25</td><td>79.80</td><td>55.17</td></tr><tr><td>DDIM Ours</td><td>25 50</td><td>/ 50%</td><td>4.26 3.94</td><td>6.00 5.92</td><td>192.71 200.93</td><td>79.37 80.47</td><td>53.99 54.05</td></tr><tr><td>DDIM Ours</td><td>20 40</td><td>/ 50%</td><td>5.12 4.32</td><td>6.60</td><td>184.23</td><td>78.37</td><td>53.50</td></tr><tr><td>DDIM</td><td>10</td><td>/</td><td>14.76</td><td>6.50</td><td>196.01</td><td>79.64</td><td>53.19</td></tr><tr><td>Ours</td><td>20</td><td>50%</td><td>9.18</td><td>12.38</td><td>129.19</td><td>65.51</td><td>48.95</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>10.85</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>160.30</td><td>73.16</td><td>49.27</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DDIM</td><td></td><td>/</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>8</td><td></td><td>24.22</td><td>18.89</td><td>100.75</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>55.49</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>46.93</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>16.28</td><td>11.44</td><td>123.65</td><td>64.36</td><td>49.99</td></tr><tr><td>Ours</td><td>10</td><td>20%</td></table></body></html>

Penalty Regulation. We regulate the penalty ratios $\rho ^ { \mathrm { a t t n } }$ and $\rho ^ { \mathrm { f e e d } }$ for MHSA and Feedforward in Eq. (5) from 1e-7 to 1e-2. Both penalty ratios are kept identical in our experiments to optimize performance, as explained by the ablation study results shown in the lower of Figure 5.

Evaluation. To evaluate the effectiveness of our method, we primarily compare our method to the DDIM (Song, Meng, and Ermon 2020), varying the sampling steps from 10 to 50. Visualization results are generated with DiT-XL/2 model in $2 5 6 \times 2 5 6$ and $5 1 2 \times 5 1 2$ resolutions. For quantitative analysis, the 50,000 images are generated per trial with classifier-free guidance in our experiments. We adopt the Fre´chet inception distance (FID) (Heusel et al. 2017), Inception Score (IS) (Salimans et al. 2016), sFID (Nash et al. 2021), and Precision/Recall (Kynka¨a¨nniemi et al. 2019) as the evaluation metrics. The computation cost as TMACs is calculated with the work (Zhu 2022).

Testing Bed. We implement our acceleration framework on mobile devices, specifically, we use OpenCL for mobile GPU backend. LazyDiT is built upon our existing DNN execution framework that supports extensive operator fusion for various DNN structures. We also integrated other general DNN inference optimization methods similar to those in (Chen et al. 2018; Abadi et al. 2016), including memory layout and computation graph. Results are obtained using a smartphone with a Qualcomm Snapdragon 8 Gen 3, featuring a Qualcomm Kryo octa-core CPU, a Qualcomm Adreno GPU, and 16 GB of unified memory. Each result take 50 runs, with average results reported as variance is negligible.

Table 2: Large-DiT model results on ImageNet $( \mathrm { c f g } { = } 1 . 5 )$ . Full results included in Table 4 at Appendix A.1.   

<html><body><table><tr><td>Method</td><td>#of Step</td><td>Lazy Ratio</td><td>FID √</td><td>SFID</td><td>IS →</td><td>Prec. 个</td><td>Rec. 个</td></tr><tr><td colspan="8">Large-DiT-3B (256×256)</td></tr><tr><td>DDIM</td><td>50</td><td>/</td><td>2.10</td><td>4.36</td><td>263.83</td><td>80.36</td><td>59.55</td></tr><tr><td>DDIM Ours</td><td>35 50</td><td>/ 30%</td><td>2.23 2.12</td><td>4.48 3.32</td><td>262.22 262.27</td><td>80.08 80.27</td><td>60.33 60.01</td></tr><tr><td>DDIM Ours</td><td>25 50</td><td>/ 50%</td><td>2.75 2.42</td><td>4.95 4.86</td><td>247.68 257.59</td><td>79.12 79.71</td><td>58.88 59.41</td></tr><tr><td>DDIM Ours</td><td>20 40</td><td>/ 50%</td><td>3.46 2.79</td><td>5.57 5.15</td><td>239.18 250.84</td><td>77.87 78.84</td><td>58.71 59.42</td></tr><tr><td>DDIM</td><td>14</td><td>/</td><td>5.84 4.64</td><td>7.80 6.35</td><td>211.13</td><td>73.96</td><td>56.32 57.81</td></tr><tr><td>Ours DDIM</td><td>20 10</td><td>30% /</td><td>13.05</td><td>14.17</td><td>220.48 162.22</td><td>75.61 64.89</td><td>51.92</td></tr><tr><td>Ours DDIM</td><td>20 7</td><td>50% /</td><td>7.36 37.33</td><td>10.55 35.02</td><td>197.67 84.68</td><td>72.14 44.52</td><td>54.40 47.17</td></tr><tr><td>Ours</td><td>10</td><td>30%</td><td>16.40</td><td>12.72</td><td>143.70</td><td>61.31</td><td>54.17</td></tr><tr><td colspan="8"></td></tr><tr><td>DDIM</td><td>50</td><td>/</td><td></td><td>Large-DiT-7B (256×256)</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>2.16</td><td>4.64</td><td>274.89</td><td>80.87</td><td>60.00</td></tr><tr><td>DDIM</td><td>35 50</td><td>/ 30%</td><td>2.29 2.13</td><td>4.83</td><td>267.31</td><td>80.42</td><td>59.21</td></tr><tr><td>Ours</td><td></td><td></td><td></td><td>4.49</td><td>267.37</td><td>80.55</td><td>60.76</td></tr><tr><td>DDIM Ours</td><td>25 50</td><td>/ 50%</td><td>2.76 2.53</td><td>5.36 5.46</td><td>259.07 265.26</td><td>79.33 80.48</td><td>58.76 58.88</td></tr><tr><td>DDIM</td><td>20</td><td>/</td><td>3.32</td><td>6.05</td><td>247.94</td><td>78.51</td><td>57.78</td></tr><tr><td>Ours</td><td>40</td><td>50%</td><td>2.90</td><td>6.01</td><td>257.47</td><td>79.67</td><td>57.97</td></tr><tr><td>DDIM</td><td>14</td><td>/</td><td>5.66</td><td>8.80</td><td>218.50</td><td>74.57</td><td>55.18</td></tr><tr><td>Ours</td><td>20</td><td>30%</td><td>4.97</td><td>7.30</td><td>220.99</td><td>75.04</td><td>57.60</td></tr><tr><td>DDIM</td><td>10</td><td>/</td><td>12.70</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>15.93</td><td>166.66</td><td>65.27</td><td>52.67</td></tr><tr><td>Ours</td><td>20</td><td>50%</td><td>7.00</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>11.42</td><td>206.57</td><td>72.61</td><td>55.14</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DDIM</td><td>7</td><td>/</td><td>36.57</td><td>39.76</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>84.54</td><td>44.69</td><td>47.44</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>143.14</td><td>61.05</td><td>50.23</td></tr><tr><td>Ours</td><td></td><td>30%</td><td>16.83</td><td>22.76</td></table></body></html>

![](images/657334bbd7bf0bec9f35e5a6471aae7cbf1f4fc46a6f327a8350e3e31a04d5b7.jpg)  
Figure C3la:ssI3m87age Cvlaisu28a5lizatCilaossn22geneCrlastse24d8  by CDlaissT2-89XL/2C asms 1o4d5 el in $2 5 6 \times 2 5 6$ resolution on mobile. Images at the first and second rows are generated with 10 and 7 sampling steps. Images at the last row are generated with $30 \%$ lazy ratio.

Table 3: Latency results on mobile devices with similar task performance or computation cost compared to the DDIM.   

<html><body><table><tr><td>Method</td><td>#of Step</td><td>Lazy Ratio</td><td>TMACs</td><td>IS ↑</td><td>Latency (s)</td></tr><tr><td colspan="6">DiT-XL/2 (256×256)</td></tr><tr><td>DDIM</td><td>50</td><td>/</td><td>5.72</td><td>241.01</td><td>21.62</td></tr><tr><td>DDIM DDIM Ours</td><td>40 25 50</td><td>/ / 50%</td><td>4.57 2.86 2.87</td><td>236.26 230.95 237.03</td><td>17.47 11.33 11.41</td></tr><tr><td>DDIM</td><td>20</td><td>/</td><td>2.29</td><td>222.87</td><td>9.29</td></tr><tr><td>DDIM Ours</td><td>16 20</td><td>/</td><td>1.83</td><td>211.30</td><td>7.60</td></tr><tr><td>DDIM</td><td></td><td>20%</td><td>1.83</td><td>227.63</td><td>7.67</td></tr><tr><td>DDIM</td><td>8 7</td><td>/ /</td><td>0.92 0.80</td><td>118.69 91.67</td><td>3.87 3.54</td></tr><tr><td>Ours</td><td>10</td><td>30%</td><td>0.80</td><td>136.81</td><td>3.57</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6">DiT-XL/2 (512×512)</td></tr><tr><td>DDIM</td><td>50</td><td>/</td><td>22.85</td><td>205.01</td><td>75.09</td></tr><tr><td>DDIM DDIM</td><td>40 25</td><td>/ /</td><td>18.29 11.43</td><td>200.24 192.71</td><td>62.64 41.37</td></tr><tr><td>Ours DDIM</td><td>50 13</td><td>50% /</td><td>11.48 5.94</td><td>200.93 156.82</td><td>41.51 23.71</td></tr><tr><td>DDIM</td><td>10</td><td>/</td><td>4.57</td><td>129.19</td><td>19.56</td></tr><tr><td>Ours</td><td>20</td><td>50%</td><td>4.59</td><td>160.30</td><td>19.77</td></tr><tr><td>DDIM</td><td>9</td><td>/</td><td>4.10</td><td>114.85</td><td>16.98</td></tr><tr><td>DDIM</td><td>8</td><td>/</td><td>3.66</td><td>100.75</td><td>15.69</td></tr><tr><td>Ours</td><td>10</td><td>20%</td><td>3.67</td><td>123.65</td><td>15.79</td></tr></table></body></html>

# 4.2 Results on ImageNet

We present the results generated with DiT officially released models compared to DDIM in Table 1. Full results with more model sizes and lazy ratios are included in Table 5 of Appendix A.1. Due to the addition of lazy learning layers, the computational cost of our method is slightly higher than that of DDIM. Our experiments demonstrate that our method can perform better than the DDIM on DiT models with $2 5 6 \times 2 5 6$ and $5 1 2 \times 5 1 2$ resolutions. Particularly, for sampling steps fewer than 10, our method demonstrates a clear advantage over DDIM at both resolutions, highlighting the promise of our approach. For larger models with 3B and 7B parameters, we present the results in Table 2. Compared to the DiT-XL/2 model with 676M parameters, Large-DiT models with a few billion parameters exhibit more redundancy during the diffusion process. Full results for LargeDiT models are included in Table 4 at Appendix A.1. Experiments demonstrate that at $50 \%$ lazy ratio, our method significantly outperforms the approach of directly reducing sampling steps with DDIM. We further visualize the images generation results in Figure 1. We also compare with other cache-based method Learn2Cache (Ma et al. 2024) which adopts input independent cache strategy and requires full training on ImageNet, the results are in Table 7 at Appendix A.4. For each sampling step, Learn2Cache only has one cache strategy, whereas our method outperforms it with less training cost, demonstrating both the effectiveness and

![](images/b35b09ba3b5dc6e85616503077046f8fc4f8c86b8dc523eab4b0c626b8d1c819.jpg)  
Figure 4: Visualization for the laziness in MHSA and Feedforward at each layer generated through DDIM 20 steps on DiT-XL.   
Figure 5: Upper figure: ablation for the generation performance with different individual laziness applied to each module independently. Lower figure: ablation for the generation performance with variant lazy ratio for one module and fixed lazy ratio for another module.

the flexibility of our method.

# 4.3 Generation on Mobile

We present the latency profiling results on mobile devices in Table 3. Our method achieves better performance with less computation cost compared to DDIM. Additionally, when computational costs and latency are similar, our method significantly outperforms DDIM in terms of performance. Notably, with 10 sampling steps and a $30 \%$ lazy ratio, our method produces significantly higher image quality in $2 5 6 \times 2 5 6$ resolution than the DDIM sampler. Besides, we visualize the images generated on mobile in Figure 3. The images in the last row, generated with our method, exhibit higher quality compared to the second row, which are generated without the laziness technique under similar latency. Therefore, our method is especially beneficial for deploying diffusion transformers on mobile devices, offering a promising solution for real-time generation on edge platforms in the future. Meanwhile, the latency results tested on GPUs are included in Table 6 at Appendix A.2. Our method delivers much better performance with faster latency on GPUs, especially when the number of sampling steps are fewer than 10. Moreover, with almost the same latency, our method performs much better than DDIM.

# 4.4 Ablation Study

Individual Laziness. We perform ablation studies on the laziness of MHSA and Feedforward modules separately by regulating the corresponding penalty ratios to determine the maximum applicable laziness for each, thereby exploring the redundancy within both components. We present the results generated with DDIM 20 steps on DiT-XL/2 $( 2 5 6 \times 2 5 6 )$ in the upper figure in Figure 5. The analysis indicates that the maximum applicable lazy ratio is $30 \%$ for MHSA and $20 \%$ for Feedforward modules. The identification reveals that applying laziness individually to either MHSA or Feedforward network is not the most effective lazy strategy, which motivates us to apply the laziness to both modules simultaneously in our experiments.

Lazy Strategy. To optimize laziness in both MHSA and Feedforward modules for optimal performance, we fix the laziness in one module and regulate the penalty ratio of the other, varying the lazy ratio from $0 \%$ to $40 \%$ . Specifically, we separately fix $30 \%$ lazy ratio to MHSA or $20 \%$ lazy ratio to Feedforward modules, and analyzed the model performance by regulating the lazy ratio of another module with DDIM 20 steps on DiT-XL/2 $2 5 6 \times 2 5 6 )$ ). The results, as presented in the lower figure of Figure 5, reveal that the model achieves optimal performance when the same lazy ratio is

12182060 21. MHSA Feedforward 2.5 2.4 1.1 1.9   
Lazy Ratio 10% 20% 30% $40 \%$ $50 \%$   
240   
2 w/ $3 0 \%$ MHSA Lazy Ratio   
30 w/ $2 0 \%$ Feedforward Lazy Ratio   
2100   
190   
Lazy Ratio 0% 10% 20% 30% 40%

applied to both MHSA and Feedforward. Thus, we adopt the same penalty ratio for both modules in our experiments to achieve the best performance.

Layer-wise Laziness. To investigate the layer-wise importance during the diffusion process, we examined the laziness of each layer over 20 sampling steps with DiT-XL/2 model in $2 5 6 \times 2 5 6$ resolution with 8 images. The results, visualized in Figure 4, illustrate the layer-wise lazy ratio distribution and highlight key patterns in layer importance. The analysis reveals that, for MHSA, the latter layers are more critical, whereas for Feedforward layers, the initial layers hold greater importance. This is evidenced by the decreasing lazy ratio in MHSA and the increasing lazy ratio in MLP as going deeper. Moreover, all layers contribute to the process, as there is no such layer that has a $100 \%$ lazy ratio, meaning no layer is completely bypassed. Therefore, strategies such as removing layers or optimizing model structure are not applicable for transformer-based diffusion models.

# 5 Conclusion and Limitation

In this work, we introduce the LazyDiT framework to accelerate transformer-based diffusion models. We first show lower bound of similarity between consecutive steps is notably high. We then incorporate a lazy skip strategy inspired by the Taylor expansion of similarity. Experimental results validate the effectiveness of our method and we further implement our method on mobile devices, achieving better performance than DDIM. For the limitation, there is additional computation overhead for lazy learning layers.