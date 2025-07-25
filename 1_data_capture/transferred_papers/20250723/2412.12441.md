# Numerical Pruning for Efficient Autoregressive Models

Xuan Shen1\*, Zhao $\mathbf { S o n g ^ { 2 } }$ , Yufa Zhou3, Bo Chen4, Jing Liu5, Ruiyi Zhang2, Ryan A. Rossi2, Hao $\mathbf { T a n } ^ { 2 }$ , Tong $\mathbf { Y } \mathbf { u } ^ { 2 }$ , Xiang Chen2, Yufan Zhou2, Tong $\mathbf { S u n ^ { 2 } }$ , Pu Zhao1, Yanzhi $\mathbf { W a n g ^ { 1 \dagger } }$ , Jiuxiang $\mathbf { G u ^ { 2 \dag } }$

1Northeastern University 2Adobe Research 3University of Pennsylvania 4Middle Tennessee State University 5Monash University shen.xu, p.zhao, yanz.wang @northeastern.edu, jigu@adobe.com

# Abstract

Transformers have emerged as the leading architecture in deep learning, proving to be versatile and highly effective across diverse domains beyond language and image processing. However, their impressive performance often incurs high computational costs due to their substantial model size. This paper focuses on compressing decoder-only transformerbased autoregressive models through structural weight pruning to improve the model efficiency while preserving performance for both language and image generation tasks. Specifically, we propose a training-free pruning method that calculates a numerical score with Newton’s method for the Attention and MLP modules, respectively. Besides, we further propose another compensation algorithm to recover the pruned model for better performance. To verify the effectiveness of our method, we provide both theoretical support and extensive experiments. Our experiments show that our method achieves state-of-the-art performance with reduced memory usage and faster generation speeds on GPUs.

# 1 Introduction

Transformers have been dominant in generative models. This includes Large Language Models (LLMs) (Vaswani et al. 2017; Touvron et al. 2023b) for language generation, as well as recent autoregressive image generation models (Van Den Oord, Vinyals et al. 2017; Esser, Rombach, and Ommer 2021; Ramesh et al. 2021; Yu et al. 2022). Notably, models such as LlamaGen (Sun et al. 2024), which use image tokenizers to convert continuous images into discrete tokens, have demonstrated the ability to surpass diffusion models (Ho, Jain, and Abbeel 2020; Rombach et al. 2022) in image generation tasks. The “next-token prediction” paradigm demonstrates significant capabilities in addressing both language and image generation tasks, enabling solutions that mimic human-like conversational interactions (Achiam, Adler et al. 2023; Li et al. 2024a).

Recognizing the capabilities of large autoregressive models pioneering works (Frantar and Alistarh 2023; Sun et al. 2023; Ma, Fang, and Wang 2023; Ashkboos et al. 2024;

Zhan et al. 2021; Zhao et al. 2024; Zhan et al. 2024b) have sought to compress these models to enhance their execution efficiency. Compared to irregular pruning methods, structural pruning offers a more efficient reduction in both computational and memory overhead (Jian et al. 2021; Gong et al. 2022, 2023). By maintaining a consistent and regular structure, it simplifies implementation, accelerates processing, and leads to more predictable resource savings (Kong et al. 2022, 2023). However, most of these efforts focus solely on language models and language-related research areas. Consequently, their methods are not readily applicable to image generation tasks because of the fundamental differences in data structure and computational requirements between language and image processing (Reed et al. 2016; Parmar et al. 2018; Lee et al. 2022; Shen et al. 2024a,c,b, 2023b,a; Li et al. 2023, 2024c,b). Therefore, it is crucial to explore the transformer architecture itself, rather than focusing on specific application models. This motivates us to develop a general method for compressing autoregressive models applicable to multiple kinds of generative tasks.

Additionally, the recovery of pruned models are crucial. Full-parameter retraining of large autoregressive models after pruning is often computationally prohibitive, making calibrations with a few samples a preferred approach. Previous work (Frantar and Alistarh 2023) employs the Optimal Brain Surgeon (OBS) technique (Hassibi, Stork, and Wolff 1993; LeCun, Denker, and Solla 1989) for weight updates during pruning. However, its heavy reliance on the approximation information increases sensitivity to noise and reduces robustness across different datasets. SliceGPT (Ashkboos et al. 2024) relies on a large number of samples for pruning and calibration, leading to overfitting on calibration data and limiting the generalization to other different datasets.

In this work, we present a novel structural pruning approach that leverages our proposed numerical score, combined with compensation techniques for performance recovery. We first calculate the numerical score for each layer through solving the optimal pruning mask for the minimization of pruning errors using the Newton’s method. By ranking these numerical scores of all layers, we generate the globally pruning mask with the specified pruning ratio. Additionally, we introduce a compensation algorithm to recover pruned models by updating the remaining weights to account for the loss caused by the pruned weights. We empirically evaluate our method using the LLaMA model family including LLaMA, LLaMA-2, and LLaMA-3 as representative LLMs and LlamaGen for image generation tasks. Experimental results show that our method outperforms other state-of-the-art approaches in both language and image generation tasks, validating the effectiveness of our proposed numerical score and compensation algorithm. Moreover, our method reduces GPU memory usage and accelerates generation without requiring any additional GPU-specific modifications. Our main contributions are summarized as follows,

• We propose a numerical score, derived from the numerical solution of the optimal mask for minimizing pruning errors with Newton’s method.   
• We propose a compensation algorithm for the reconstruction of the pruned model, further enhancing the task performance of the pruned model.   
• Experimental results show that our method not only achieves state-of-the-art performance but also reduces memory usage and accelerates generation on GPUs.

# 2 Related Work

# 2.1 Compression for LLMs

The large number of parameters in LLMs motivates the need for pruning (Gong et al. 2020; Wu et al. 2022; Zhan et al. 2024a; Li et al. 2022; Zhang et al. 2022; Zhan et al. 2024c; Shen et al. 2024d) to improve efficiency. The work (Frantar and Alistarh 2023) introduces the Optimal Brain Surgeon (OBS) method (Hassibi, Stork, and Wolff 1993; LeCun, Denker, and Solla 1989) to compress the LLMs, which removes weights with minimal impact on the loss function. It then updates the remaining weights by utilizing the inverse of the Hessian matrix to mitigate errors caused by the pruning process. Unfortunately, this kind of pruning method is still irregular, meaning it does not lead to significant reductions in memory and computational requirements. Subsequent works, such as LLM-Pruner (Ma, Fang, and Wang 2023), SliceGPT (Ashkboos et al. 2024), and FLAP (An et al. 2023), propose structural pruning methods that effectively reduce memory usage and accelerate inference on GPUs. These methods offer significant advantages over irregular pruning by directly enhancing the utility and efficiency of the models. While autoregressive models excel in sequential data processing, such as text, the distinct nature of image data, where spatial relationships and pixel-level details are critical, demands different approaches. As a result, adapting these models to image generation introduces complexities that limit their scalability and effectiveness.

# 2.2 Autoregressive Models in Image Generation

Autoregressive models, initially renowned for their success with LLMs, have recently gained popularity in the image generation research area. Pioneering works (Van Den Oord, Vinyals et al. 2017; Esser, Rombach, and Ommer 2021) introduced image tokenizers that convert continuous images into discrete tokens. These tokenizers, which have been demonstrated to be effective by the following works (Ramesh et al. 2021; Yu et al. 2021, 2022), enable autoregressive models to generate image tokens using the nexttoken prediction approach. Recent work (Sun et al. 2024) delivers a series of image generation models with a new constructed image tokenizer. This research demonstrates the effectiveness of LLM frameworks in image generation tasks, validating their potential beyond traditional language applications. Additionally, the work (Li et al. 2024a) delves deeper into the continuous-valued domains of autoregressive models and removes the image tokenizers for image generation tasks. This work achieves stronger results while leveraging the speed advantage of sequence modeling, which further enhances the utilization and demonstrates the potential of autoregressive models in image generation tasks.

![](images/8a34cf12b15d42997892a8d48fc114859da5a167fdb9b965994b20f61dbf685f.jpg)  
Figure 1: Pruning overview. Blue modules denote column pruning and green modules denote row pruning.

# 3 Methodology

# 3.1 Preliminary

Notations. We use $\mathbb { E } [ \cdot ]$ to denote the expectation. For two vectors $x \in \mathbb { R } ^ { n }$ and $\boldsymbol { y } \in \mathbb { R } ^ { n }$ , we use $\langle x , y \rangle$ to denote the inner product between $x , y$ , i.e., $\textstyle \langle x , y \rangle = \sum _ { i = 1 } ^ { n } x _ { i } y _ { i }$ . We use ${ \mathbf { 1 } } _ { n }$ to denote a length- $\cdot n$ vector where all  he entries are ones. We use $x _ { i , j }$ to denote the $j$ -th coordinate of $x _ { i } \in \mathbb { R } ^ { n }$ . We use $\| { \boldsymbol { x } } \| _ { p }$ to denote the $\ell _ { p }$ norm of a vector $\boldsymbol { x } \in \mathbb { R } ^ { n }$ . For each $a , b \in \mathbb { R } ^ { n }$ , we use $a \circ b \in \mathbb { R } ^ { n }$ to denote the vector where $i$ -th entry is $( a \circ b ) _ { i } = a _ { i } b _ { i }$ for all $i \in [ n ]$ . We use $\| A \|$ to denote the spectral norm for matrix $A$ . For a square matrix $A$ , we use $\operatorname { t r } [ A ]$ to denote the trace of $A$ , i.e., $\textstyle { \mathrm { t r } } [ A ] = \sum _ { i = 1 } ^ { n } A _ { i , i }$ .

Internal Computation Alignment. To maintain model interpretability and mitigate the risk of model drift, we ensure consistency in the internal computations of the Attention module. This approach is aligned with established methodologies in the literature (Vaswani et al. 2017; Yang et al. 2020). Considering the definition in this paper mainly focuses on X · W where X ∈ RN×D and W ∈ RD×D′, we visualize the pruning strategy in Figure 1. In detail, we utilize the identical pruning mask (i.e., pruning strategy) for the columns of weights associated with the query, key, and value, as well as for the rows of weights in the output projection. Meanwhile, we apply the same strategy to the MLP module, using column pruning for the up and gate weights, and row pruning for the down projection weights. In this paper, we construct structural pruning metrics focusing on the output projection layers of Attention module and the down projection layers of MLP module.

# 3.2 Numerical Score

We define the weight as $W \in \mathbb { R } ^ { D \times D ^ { \prime } }$ , input as $\boldsymbol { X } \in \mathbb { R } ^ { N \times D }$ , and we denote the mask as $M \in \{ 0 , \dot { 1 } \} ^ { \dot { D } }$ . Additionally, we define the pruning ratio $\rho \in [ 0 , 1 ]$ as the ratio of the number of zeros to the total number of entries in pruning mask $M$ .

Note that, when we apply the mask column by column, the mask $M$ is a $D$ -dimensional vector. Specifically, if $M _ { j } = 0$ for $j \in [ D ]$ , we prune the entire row for $W$ , i.e., $W _ { j } = 0$ , and if $M _ { j } = 1$ we keep the original $W _ { j }$ .

To compute the numerical score, we explore the bound of the error (i.e., difference) between the original weights and pruned weights. For the bound of the error, we first formulate the error for $i \in [ D ^ { \prime } ]$ as

$$
\| X W _ { * , i } - X ( M \circ W _ { * , i } ) \| _ { 2 } .
$$

Simplify further, for $i \in [ D ^ { \prime } ]$ , Eq. (1) can be transformed into the following,

$$
\| X ( ( \mathbf { 1 } _ { D } - M ) \circ W _ { * , i } ) \| _ { 2 } .
$$

In the above equation, the $\| \mathbf { 1 } _ { D } - M \| _ { 2 }$ denotes the number of zero entries in $M$ , which is corresponding to the simply $\boldsymbol \rho \boldsymbol \cdot \boldsymbol D$ . Furthermore, assuming $\| X \| \leq R$ , we demonstrate that the following Lemma 1 holds,

Lemma 1 (informal version of Lemma 9 at Appendix D.2). We show that for $i \in [ D ^ { \prime } ]$ we have

$$
\begin{array} { r } { \| X W _ { * , i } - X ( M \circ W _ { * , i } ) \| _ { 2 } \leq \rho R \| W _ { * , i } \| _ { 2 } . } \end{array}
$$

It is intuitive for us to minimize the error after establishing the error bound in Lemma 1. Thus, we examine each term. In the error bound of Lemma 1, the pruning ratio $\rho$ is manually specified. We adopt the normalization for the input $X$ , then the norm of normalized $X$ is upper bounded by 1. Meanwhile, for $\| W _ { * , i } \| _ { 2 }$ term, it is the $\ell _ { 2 }$ norm for $i$ -th column of weight $W$ .

In order to minimize the error, we regulate both $\rho$ and $\| W _ { * , i } \|$ . Then, we generalize the mask $M$ from binary value to real value for the calculation of the numerical score. Meanwhile, we set one threshold which converts the realvalued mask back into a binary mask. For mask $M \in [ 0 , 1 ] ^ { D }$ and pruning ratio $\rho \in [ 0 , 1 ]$ , the calculation of the numerical score is formulated as follows,

$$
\begin{array} { r l } { \underset { M } { \arg \operatorname* { m i n } } } & { ~ \displaystyle \sum _ { i \in [ D ^ { \prime } ] } \| X W _ { * , i } - X ( M \circ W _ { * , i } ) \| _ { 2 } , } \\ { \mathrm { s . t . } } & { \langle \mathbf { 1 } _ { D } , M \rangle = ( 1 - \rho ) D . } \end{array}
$$

To better solve Eq. (2), we define the numerical score $z \in$ $[ 0 , 1 ] ^ { D }$ and $r : = ( 1 - \rho ) D \in [ 0 , D ]$ . The equality constraint in Eq. (2) is then equivalent to $\langle \mathbf { 1 } _ { D } , z \rangle - r = 0$ .

Then, Eq. (2) becomes the minimization problem with the equality constraint. To efficiently solve such problem, we adopt the Newton’ method (Bubeck et al. 2015). By turning

1: procedure NUMERICALSCORE $\mathit { \Psi } _ { \mathit { l } } ( X \ \in \ \mathbb { R } ^ { N \times D } , W \ \in$   
RD×D′, r ∈ [0, D], λ ∈ R+, T ∈ N+) $D$ Theorem 2   
2: We choose the initial point $z _ { 0 }$ such that $z _ { 0 } \in [ 0 , 1 ] ^ { D }$   
3: for $t = 0  T$ do   
4: $\begin{array} { r l } & { g _ { l } \gets ( ( W W ^ { \top } ) \circ ( X ^ { \top } X ) ) ( z - \mathbf { 1 } _ { D } ) } \\ & { g _ { r } \gets \lambda ( \langle \mathbf { 1 } _ { D } , z \rangle - r ) \cdot \mathbf { 1 } _ { D } } \\ & { g \gets g _ { l } + g _ { r } \in \mathbb { R } ^ { D } } \\ & { H _ { l } \gets ( W W ^ { \top } ) \circ ( X ^ { \top } X ) } \\ & { H _ { r } \gets \lambda \cdot \mathbf { 1 } _ { D \times D } } \\ & { H \gets H _ { l } + H _ { r } \in \mathbb { R } ^ { D \times D } } \\ & { z _ { t + 1 } \gets z _ { t } - H ^ { - 1 } g } \end{array}$   
5:   
6:   
7:   
8:   
9:   
10:   
11: end for   
12: z ← zT +1   
13: return z   
14: end procedure

the equality constraint into a penalty term for regularization, we further generate the following equivalent problem,

$$
\begin{array} { r l } { \underset { z \in [ 0 , 1 ] ^ { D } } { \arg \operatorname* { m i n } } } & { \frac { 1 } { 2 } \displaystyle \sum _ { i \in [ D ^ { \prime } ] } \| X W _ { * , i } - X ( z \circ W _ { * , i } ) \| _ { 2 } ^ { 2 } } \\ & { + \frac { 1 } { 2 } \lambda \cdot ( \langle \mathbf { 1 } _ { D } , z \rangle - r ) ^ { 2 } , } \end{array}
$$

where $\lambda \in \mathbb { R } _ { + }$ is the regularization parameter.

To explain how we solve this, we define the loss function for $i \in [ \bar { D } ^ { \prime } ]$ as follows,

$$
L ( z ) _ { i } = \frac { 1 } { 2 } \| X W _ { * , i } - X ( z \circ W _ { * , i } ) \| _ { 2 } ^ { 2 } .
$$

Meanwhile, for regularization term, we define as follows,

$$
L _ { \mathrm { r e g } } ( z ) = \frac { 1 } { 2 } \lambda \cdot ( \langle \mathbf { 1 } _ { D } , z \rangle - r ) ^ { 2 } .
$$

Combining Lemma 12 and Lemma 13 at Appendix D.3, we compute the gradient of Eq. (4) and Eq. (5) as follows,

$$
\begin{array} { r l } { g = } & { \underbrace { \bigl ( \bigl ( W W ^ { \top } \bigr ) \circ ( X ^ { \top } X ) \bigr ) \bigl ( z - \mathbf { 1 } _ { D } \bigr ) } _ { \mathrm { G r a d i e n t o f } L ( z ) } } \\ { + } & { \underbrace { \lambda \bigl ( \bigl < \mathbf { 1 } _ { D } , z \bigr > - r \bigr ) \cdot \mathbf { 1 } _ { D } } _ { \mathrm { G r a d i e n t o f } L _ { \mathrm { r e g } } ( z ) } . } \end{array}
$$

Combining Lemma 14 and Lemma 15 at Appendix D.3, we compute the Hessian of Eq. (4) and Eq. (5) as follows,

$$
H = \underbrace { \left( W W ^ { \top } \right) \circ ( X ^ { \top } X ) } _ { \mathrm { H e s s i a n o f } L ( z ) } + \underbrace { \lambda \cdot { \bf 1 } _ { D \times D } } _ { \mathrm { H e s s i a n o f } L _ { \mathrm { r e g } } ( z ) } .
$$

Subsequently, using Algorithm 1, we efficiently compute the optimal numerical $z$ in $O ( T D ^ { 3 } )$ , where $T$ represents the number of iterations for Newton’s Method, typically around 50 in practice. Besides, we derive the following Theorem 2.

Theorem 2 (Mask optimization, informal version of Theorem 10 at Appendix D.3). If the following conditions hold:

• Let $z \in [ 0 , 1 ] ^ { D }$ .   
• Let $r \in [ 0 , D ]$ denote the number of ones (it can be a fractional number).   
• Let $\lambda > 0$ denote a regularization co-efficients.   
• Assume $\| X \| \leq R$ .

There exists an algorithm (Algorithm $^ { l }$ ) that can get the optimal $z$ in $O ( T D ^ { 3 } )$ for Eq. (3).

# 3.3 Global Pruning

To ensure consistent head-level computation in the Attention module, given numerical scores $z ^ { \mathrm { a t i n } } \in \mathbb { R } ^ { D }$ , we group the scores of channels for $h$ -th head to determine the importance score $z _ { h } ^ { \mathrm { h e a d } }$ of individual heads as follows,

$$
z _ { h } ^ { \mathrm { h e a d } } = \frac { 1 } { D _ { h } } \cdot \sum _ { i = h \cdot D _ { h } } ^ { ( h + 1 ) \cdot D _ { h } } z _ { i } ^ { \mathrm { a t t n } } ,
$$

where $D _ { h }$ denotes the dimension of each head, $h \in [ H ]$ is the head index.

Unlike the Attention module, where heads work in parallel to capture various aspects of the input and their outputs are interdependent, the MLP module has a simpler structure with minimal interdependencies between its components. Thus, we retain the channel scores $z ^ { \mathrm { m l p } } \in \mathbb { R } ^ { D }$ to guide the pruning process for MLP module.

To ensure a balanced pruning process that reflects the relative importance of each layer, we simultaneously sort the numerical scores across all layers to derive the globally pruning mask. Since a single head in the Attention module is evaluated with one score but contains significantly more weights than a single channel in the MLP module, we apply scaling factors based on the model design to balance number of pruned parameters between the Attention heads and MLP channels. For a specified global pruning ratio $\rho$ , hidden state dimension $D$ , head dimension $D _ { h }$ in the Attention module, and intermediate size $D _ { \mathrm { i n t e r } }$ in the MLP module, we define $\Psi$ as the set that stores the scores for the whole model, then apply the scaling factor $\alpha$ when generating the threshold $\eta$ for all the scores as follows,

$$
\Psi : = \alpha \cdot ( \{ \{ z _ { h , l } ^ { \mathrm { h e a d } } \} _ { h = 1 } ^ { H } \} _ { l = 1 } ^ { L } ) \bigcup \{ \{ z _ { i , l } ^ { \mathrm { m l p } } \} _ { i = 1 } ^ { D _ { \mathrm { i n t e r } } } \} _ { l = 1 } ^ { L } ,
$$

$$
\eta = \mathrm { s o r t } ( \Psi ) [ ( \rho ( L \cdot H + L \cdot D _ { \mathrm { i n t e r } } ) ) ] , \alpha = \frac { 4 D _ { h } } { 3 } ,
$$

where $H = \operatorname { i n d e x } ( D / D _ { h } )$ denotes the number of head in Attention module, $L$ denotes the number of layers for the whole model. Since the Attention module involves pruning 4 linear projections (query, key, value, and output) in head level, while the MLP module prunes only 3 (up, gate, and down) in channel level, the scaling factor $\alpha$ is given by $\frac { 4 D _ { h } } { 3 }$ .

When the threshold $\eta$ is determined, we prune the heads in the Attention module and the channels in the MLP module across all layers based on the strategy that removes heads or channels with numerical scores below the threshold.

# 3.4 Compensation for Pruning

With the above discussion, we obtain the pruning mask with Newton’s method. To further improve the model performance, we modify the remaining weights in the model to compensate the loss of the pruned weights.

Problem Formulation. Note that to align the internal computations in the attention and MLP modules, we prune the rows of the output layers in the modules and the columns in other layers of the modules. If the columns of a layer with $W$ is pruned in $X W$ , the corresponding columns of the output also become zero and we are not able to compensate its loss, since modifying other unpruned columns can not change the zero output for the pruned columns. Thus, we only update the weights of the output layers with row pruning in the Attention and MLP modules. We modify the remaining rows based on pruned rows in $W$ . For layers with column pruning, we do not modify their unpruned weights.

For the original weights $W$ , after pruning, there are $k$ pruned rows which are all zeros and their row indexes are denoted by $p _ { i } , \forall i \in \ [ k ]$ . We modify the weights with the weight perturbations $\delta W$ , so that the layer output difference (before and after pruning) measured with $\ell _ { 2 }$ norm is minimized. The weight optimization problem can formulated as the following,

$$
\begin{array} { r l } { \underset { \delta W } { \mathrm { m i n } } } & { ~ \mathcal { L } ( \delta W ) = \| X ( W + \delta W ) - X W \| _ { 2 } ^ { 2 } = \| X \delta W \| _ { 2 } ^ { 2 } , } \\ { \mathrm { s . t . } } & { ~ e _ { p _ { i } } ^ { \top } \delta W + ( W ) _ { p _ { i } , * } = 0 , \quad \mathrm { f o r } i = 1 , 2 , \ldots , k . ~ ( 9 ) } \end{array}
$$

where $e _ { p _ { i } } \in \mathbb { R } ^ { D \times 1 }$ is the one-hot vector with the $p _ { i } ^ { t h }$ element as 1 and all others as 0. Thus, $e _ { p _ { i } } ^ { \top } \delta W$ denotes selecting the $p _ { i } ^ { t h }$ row of $\delta W$ . $( W ) _ { i , j }$ represents the element in the $i ^ { t h }$ row and $j ^ { t h }$ column of the matrix. Then, $( W ) _ { p _ { i } , * }$ represents the $p _ { i } ^ { t h }$ row of $W$ . We can see that the constraint in Eq. (9) ensures that the corresponding pruned rows in the modified weights are all zeros, and the remaining weights are optimized to minimize the loss incurred by pruned rows.

It can be further transformed to the following,

$$
\begin{array} { r l } { \underset { \delta W } { \operatorname* { m i n } } } & { { } \mathcal { L } ( \delta W ) = \lVert X \delta W \rVert _ { 2 } ^ { 2 } , } \\ { \mathrm { s . t . } } & { { } M _ { p } ^ { \top } \delta W + W _ { p } = 0 , } \end{array}
$$

where $M _ { p } \in \mathbb { R } ^ { D \times k }$ is the collection of all $e _ { p _ { i } }$ , i.e., $( M _ { p } ) _ { * , i } ~ = ~ e _ { p _ { i } }$ , or $( \boldsymbol { M } _ { p } ^ { \top } ) _ { i , * } \ = \ e _ { p _ { i } } ^ { \top } , \forall i \ \in \ [ k ]$ . Similarly, $W _ { p }$ is a collection of all pruned rows in $W$ with $( W _ { p } ) _ { i , * } =$ $( W ) _ { p _ { i } , * } , \forall i \in [ k ]$ . We have $W _ { p } = M _ { p } ^ { \top } W$ .

Optimal Solution. Eq. (10) can be solved analytically with the following Theorem 3. The detailed proof is shown in Appendix B.

Theorem 3. The optimal solution for Eq. (10) can be derived as the following,

$$
\delta W ^ { * } = - ( 2 X ^ { \top } X ) ^ { - 1 } M _ { p } ( M _ { p } ^ { \top } ( 2 X ^ { \top } X ) ^ { - 1 } M _ { p } ) ^ { - 1 } M _ { p } ^ { \top } W .
$$

Remark 4. The optimal loss of Problem (10) corresponding to the optimal weight perturbation can be expressed as

$$
\nonumber L ^ { * } = \frac { 1 } { 2 } \sum _ { i } ( { \cal W } ^ { \top } { \cal M } _ { p } ( { \cal M } _ { p } ^ { \top } ( 2 X ^ { \top } X ) ^ { - 1 } { \cal M } _ { p } ) ^ { - 1 } { \cal M } _ { p } ^ { \top } { \cal W } ) _ { i , i } .
$$

The sum in Eq. (12) is computed over $D ^ { \prime }$ (the number of columns in $W$ ), i.e., $i \in [ D ^ { \prime } ]$ .

Perplexity P2e5rplexity L Perplexity Perplexity Perplexity   
30 15 40 15 LLM-Pruner LM-Pruner LLM-Pruner . SliceGPT SliceGPT   
25 . SliceGPT SliceGPT 13 SliceGPT 35 FLAP 13 FLAP · FLAP 20 FLAP 11 FLAP 30 Ours 11 Ours   
20 Ours Ours Ours 25 15 9 9   
150 10 57 15 57 .   
5 5 3 5 3   
Ratio10% 20% 30% 40% 50% Ratio10% 20% 30% 40% 50% Ratio10% 20% 30% 40% 50% Ratio10% 20% 30% 40% 50% Ratio10% 20% 30% 40% 50% LLaMA-2 7B LLaMA-2 13B LLaMA-2 70B LLaMA-3 8B LLaMA-3 70B

<html><body><table><tr><td rowspan="2">Method</td><td rowspan="2">Prune Ratio</td><td colspan="3">LLaMA-7B</td><td colspan="3">LLaMA-13B</td><td colspan="3">LLaMA-30B</td><td colspan="3">LLaMA-65B</td></tr><tr><td>Wiki</td><td>PTB</td><td>C4</td><td>Wiki</td><td>PTB</td><td>C4</td><td>Wiki</td><td>PTB</td><td>C4</td><td>Wiki</td><td>PTB</td><td>C4</td></tr><tr><td>Baseline</td><td>1</td><td>5.68</td><td>27.34</td><td>7.08</td><td>5.09</td><td>19.23</td><td>6.61</td><td>4.10</td><td>16.29</td><td>5.98</td><td>3.53</td><td>17.61</td><td>5.62</td></tr><tr><td>LLM-Pruner</td><td>10%</td><td>7.41</td><td>36.73</td><td>9.25</td><td>6.38</td><td>31.85</td><td>8.16</td><td>4.92</td><td>18.17</td><td>6.63</td><td>3.98</td><td>19.44</td><td>6.08</td></tr><tr><td>SliceGPT</td><td>10%</td><td>6.97</td><td>88.48</td><td>23.54</td><td>6.11</td><td>60.15</td><td>20.18</td><td>5.24</td><td>39.72</td><td>17.83</td><td>4.57</td><td>36.20</td><td>14.14</td></tr><tr><td>FLAP</td><td>10%</td><td>6.34</td><td>32.39</td><td>8.058</td><td>5.45</td><td>20.99</td><td>7.33</td><td>4.52</td><td>17.29</td><td>6.49</td><td>3.91</td><td>19.35</td><td>6.04</td></tr><tr><td>Ours</td><td>10%</td><td>6.01</td><td>31.65</td><td>7.94</td><td>5.38</td><td>20.52</td><td>7.27</td><td>4.43</td><td>17.26</td><td>6.47</td><td>3.82</td><td>19.28</td><td>6.02</td></tr><tr><td>LLM-Pruner</td><td>20%</td><td>10.73</td><td>59.73</td><td>12.15</td><td>6.38</td><td>31.85</td><td>9.42</td><td>5.83</td><td>20.18</td><td>7.55</td><td>4.65</td><td>21.85</td><td>6.75</td></tr><tr><td>SliceGPT</td><td>20%</td><td>8.42</td><td>120.89</td><td>35.93</td><td>7.17</td><td>86.26</td><td>29.70</td><td>6.18</td><td>50.95</td><td>26.85</td><td>5.34</td><td>61.09</td><td>21.86</td></tr><tr><td>FLAP</td><td>20%</td><td>7.40</td><td>36.77</td><td>9.99</td><td>6.03</td><td>23.33</td><td>8.42</td><td>5.18</td><td>19.30</td><td>7.42</td><td>4.45</td><td>21.45</td><td>6.75</td></tr><tr><td>Ours</td><td>20%</td><td>6.60</td><td>35.75</td><td>9.49</td><td>5.89</td><td>23.11</td><td>8.39</td><td>4.92</td><td>18.58</td><td>7.36</td><td>4.26</td><td>20.94</td><td>6.73</td></tr><tr><td>LLM-Pruner</td><td>30%</td><td>18.58</td><td>93.24</td><td>17.78</td><td>11.81</td><td>45.42</td><td>12.65</td><td>7.59</td><td>24.97</td><td>9.08</td><td>5.52</td><td>26.38</td><td>7.53</td></tr><tr><td>SliceGPT</td><td>30%</td><td>12.75</td><td>258.90</td><td>67.33</td><td>9.18</td><td>125.40</td><td>46.46</td><td>7.74</td><td>75.89</td><td>42.71</td><td>6.56</td><td>74.43</td><td>35.68</td></tr><tr><td>FLAP</td><td>30%</td><td>9.18</td><td>47.35</td><td>13.08</td><td>6.97</td><td>27.36</td><td>10.01</td><td>6.28</td><td>21.88</td><td>8.53</td><td>5.10</td><td>23.91</td><td>7.59</td></tr><tr><td>Ours</td><td>30%</td><td>7.56</td><td>41.05</td><td>11.53</td><td>6.57</td><td>26.27</td><td>9.98</td><td>5.46</td><td>20.48</td><td>8.46</td><td>4.75</td><td>22.13</td><td>7.51</td></tr><tr><td>LLM-Pruner</td><td>50%</td><td>126.0</td><td>460.7</td><td>73.88</td><td>45.69</td><td>152.99</td><td>36.94</td><td>19.68</td><td>78.29</td><td>18.64</td><td>9.34</td><td>43.79</td><td>12.16</td></tr><tr><td>SliceGPT</td><td>50%</td><td>1540</td><td>6364</td><td>4847</td><td>18.75</td><td>277.34</td><td>122.5</td><td>15.60</td><td>195.4</td><td>118.5</td><td>12.01</td><td>160.3</td><td>92.66</td></tr><tr><td>FLAP</td><td>50%</td><td>21.89</td><td>135.8</td><td>30.86</td><td>12.88</td><td>53.54</td><td>18.37</td><td>13.41</td><td>47.30</td><td>13.17</td><td>6.98</td><td>28.52</td><td>10.36</td></tr><tr><td>Ours</td><td>50%</td><td>11.66</td><td>82.55</td><td>20.72</td><td>8.91</td><td>37.56</td><td>16.12</td><td>7.25</td><td>26.68</td><td>11.91</td><td>6.02</td><td>25.17</td><td>9.73</td></tr><tr><td>LLM-Pruner</td><td>70%</td><td>9010</td><td>4111</td><td>2655</td><td>5900</td><td>6039</td><td>1334</td><td>895.7</td><td>3274</td><td>456.9</td><td>Nan</td><td>Nan</td><td>Nan</td></tr><tr><td>SliceGPT</td><td>70%</td><td>3605</td><td>7304</td><td>8096</td><td>67.65</td><td>874.9</td><td>537.4</td><td>71.25</td><td>633.1</td><td>406.6</td><td>102.4</td><td>863.9</td><td>662.8</td></tr><tr><td>FLAP</td><td>70%</td><td>577.9</td><td>1835</td><td>833.7</td><td>647.8</td><td>1588</td><td>975.1</td><td>2786</td><td>2735</td><td>2416</td><td>Nan</td><td>2333</td><td>Nan</td></tr><tr><td>Ours</td><td>70%</td><td>162.9</td><td>721.3</td><td>361.6</td><td>41.66</td><td>275.7</td><td>115.3</td><td>39.88</td><td>124.2</td><td>50.43</td><td>9.65</td><td>69.49</td><td>20.84</td></tr></table></body></html>

Table 1: Perplexity $( \downarrow )$ results for LLaMA-1 family models with different pruning ratios on WikiText2, PTB, and C4 with 2048 sequence length. Full results with larger sparsity ratios are included in Table 5 at Appendix A.1.

Remark 5. If the rank of $2 X ^ { \top } X$ is not full so that the inversion $( 2 \dot { X } ^ { \top } X ) ^ { - 1 }$ is unavailable, we apply the dampening method to compute $( 2 X ^ { \top } X + \gamma \cdot \dot { I } ) ^ { - 1 }$ instead of $( 2 X ^ { \top } X ) ^ { - 1 }$ , with $\gamma$ as the dampening ratio.

# 3.5 Complexity Analysis

For the computation of numerical score, according to the Lemma 1 and Theorem 2, the complexity is $O ( T D ^ { 3 } )$ where $T$ represents the number of iterations for Newton’s Method, typically around 50 in practice. Additionally, for the compensation method, as demonstrated in Eq. (11), the complexity is $O ( D ^ { 3 } )$ as we need to compute the inverse of a matrix. The matrix multiplication with $M _ { p }$ or $M _ { p } ^ { T }$ just selects the columns or rows of a matrix, without the need of actual multiplication. The complexity for numerical score calculation and compensation is the same with state-of-the-art methods, such as SparseGPT (Frantar and Alistarh 2023). In practice, the compensation is finished with just a few data samples on only the output projection layers of the Attention module and the down projection layers of the MLP module, which is more efficient compared with other recovery methods such as LLM-Pruner (Ma, Fang, and Wang 2023) to finetune the whole model on whole dataset, or SliceGPT (Ashkboos et al. 2024) to adopt a large amount of samples for calibration.

# 4 Experimental Results

# 4.1 Experiment Setup

We conduct the experiments on LLaMA model families including LLaMA-1 (Touvron et al. 2023a), LLaMA-2 (Touvron et al. 2023b), and LLaMA-3 (Meta 2024) for the language generation tasks. For evaluations, we compare the perplexity of the models on the WikiText2 (Merity et al. 2016), PTB (Marcus, Santorini, and Marcinkiewicz 1993), and C4 (Raffel et al. 2020) datasets with the 2048 sequence length. We also follow LLM-Pruner to evaluate the zero-shot accuracy on common sense reasoning zero-shot classification datasets including BoolQ (Clark et al. 2019), PIQA (Bisk et al. 2020), HellaSwag (Zellers et al. 2019),

Table 2: Pruning results for LLaMA-7B on common sense reasoning datasets. LLM-Pruner (v) and (ei) denote vector-wise and element-wise with $i$ -th order $( i = 1 , 2 )$ ). Full results with more sparsity ratios and LLaMA-13B are in Table 9 at Appendix B.   

<html><body><table><tr><td>Method</td><td>Prune Ratio</td><td>BoolQ</td><td>PIQA</td><td>Hella Swag</td><td>Wino Grande</td><td>ARC-e</td><td>ARC-c</td><td>OBQA</td><td>Average Acc.</td></tr><tr><td>LLaMA-7B</td><td>/</td><td>73.18</td><td>78.35</td><td>72.99</td><td>67.01</td><td>67.45</td><td>41.38</td><td>42.40</td><td>63.25</td></tr><tr><td>LLM-Pruner(v)</td><td></td><td>61.44</td><td>71.71</td><td>57.27</td><td>54.22</td><td>55.77</td><td>33.96</td><td>38.40</td><td>53.25</td></tr><tr><td>LLM-Pruner(e2)</td><td rowspan="2">20%</td><td>59.39</td><td>75.57</td><td>65.34</td><td>61.33</td><td>59.18</td><td>37.12</td><td>39.80</td><td>56.82</td></tr><tr><td>LLM-Pruner(e1)</td><td>57.06</td><td>75.68</td><td>66.80</td><td>59.83</td><td>60.94</td><td>36.52</td><td>40.00</td><td>56.69</td></tr><tr><td>SliceGPT</td><td>20%</td><td>37.89</td><td>64.09</td><td>45.67</td><td>62.75</td><td>53.62</td><td>31.74</td><td>33.20</td><td>46.99</td></tr><tr><td>FLAP</td><td>20%</td><td>68.59</td><td>74.21</td><td>64.98</td><td>64.40</td><td>59.89</td><td>37.80</td><td>40.20</td><td>58.58</td></tr><tr><td>Ours</td><td>20%</td><td>67.92</td><td>74.76</td><td>67.31</td><td>66.54</td><td>58.80</td><td>36.77</td><td>39.4</td><td>58.79</td></tr></table></body></html>

![](images/6e104fb1cfdd42add0f6ec3676ef140eedb8284a3185f52a6d65a7a70a5dadb1.jpg)  
Figure 3: Visualization of generated images through LlamaGen-3B in $3 8 4 \times 3 8 4$ resolution $\mathrm { ( c f g { = } 1 . 6 5 ) }$ with $10 \%$ sparsity.

WinoGrande (Sakaguchi et al. 2021), ARC-easy (Clark et al. 2018), ARC-challenge (Clark et al. 2018), and OpenbookQA (Mihaylov et al. 2018). For experiments, we adopt 128 samples from training dataset of WikiText2 to compute the numerical score and compensate the pruned models. For fairness, we also adopt 128 samples for other methods.

As for the image generation tasks, we adopt the LlamaGen (Sun et al. 2024) model family with LlamaGen-XXL and LlamaGen-3B to verify the effectiveness of our method on image generation tasks. We adopt the Fre´chet inception distance (FID) (Heusel et al. 2017), Inception Score (IS) (Salimans et al. 2016), sFID (Nash et al. 2021), and Precision/Recall (Kynk¨aa¨nniemi et al. 2019) as the evaluation metrics on ImageNet dataset (Deng et al. 2009). For all evaluations, we utilized ADM’s TensorFlow scripts (Dhariwal and Nichol 2021) to ensure fair and consistent comparisons. Given that LLM-Pruner requires a backward process and SliceGPT has slow pruning, we further implement FLAP for comparative analysis in image generation tasks. In pratical, we generate 128 images for each class of ImageNet with LlamaGen models for the computation of numerical score and compensation. Same strategy for FLAP for fairness.

# 4.2 Results of LLMs

For the LLaMA models, we present the results with different pruning ratios varying from $10 \%$ to $70 \%$ in Table 1. Based on the perplexity results evaluated with 2048 sequence length on three datasets, our method consistently outperforms other methods across all pruning ratios, demonstrating the effectiveness of our proposed approach. Full results with more sparse ratios are included in Table 5 of Appendix A.1. Results show that for the larger model LLaMA65B with pruning ratio of $70 \%$ , both LLM-Pruner and FLAP fail to produce an effective pruned model with their respective methods. In contrast, our method successfully maintains the most of the model’s capabilities.

We further evaluate the zero-shot capabilities of the pruned model across seven downstream tasks. The results of LLaMA-7B model are shown in Table 2. Full results, including additional pruning ratios and the LLaMA-13B model, are detailed in Table 9 in Appendix A.5. Our method demonstrates superior performance compared to the other three methods on those common sense reasoning zero-shot classification datasets. Besides, we show the results with LLaMA and LLaMA-2 models of our method on MMLU (Hendrycks et al. 2021) and GSM8K (Cobbe et al. 2021) datasets in Table 8 of Appendix A.4, which demonstrates that our method retains both generative and mathematical capabilities.

![](images/82d6c471ea3d561acaa0f7dd620c505d86cf88615035ffddd08f997a4e943bcc.jpg)  
Figure 4: Ablation for number of samples for compensation.

We show the results for LLaMA-2 and LLaMA-3 models with 2048 sequence length on WikiText2 dataset in Figure 2. The detailed perplexity results for both model families on three datasets are shown in Table 6 and Table 7 of Appendix A.2 and A.3. The blue line representing our method’s results consistently appears at the lowest position on the graphs, indicating its superior performance compared to the other methods with all model families.

# 4.3 Results of Image Generation

We implement the FLAP pruning method on LlamaGen model and compare this method on image generation task. We show the sparse results with LlamaGen-XXL (1.4B) and LlamaGen-3B models on ImageNet with $3 8 4 \times 3 8 4$ resolution in Table 3. We observe that for the smaller model LlamaGen-XXL (1.4B), our method shows a distinct advantage at higher pruning ratios. For the larger model LlamaGen-3B, our method consistently outperforms across all pruning ratios, effectively preserving most of the original model’s capabilities. We further visualize the images generated by $10 \%$ sparsity models in Figure 3 with additional visualizations provided in Figure 6 of Appendix A.6. We observe that our method generates better image results compared to FLAP method in most cases.

# 4.4 Ablation Study

Results with 128 Sequence Length. To demonstrate the effectiveness of our method for short sequence lengths, we present the results generated with a sequence length of 128 in Table 4 using the LLaMA-7B model and the WikiText2 dataset. Comprehensive results, including additional pruning ratios and datasets, are provided in Table 10 of Appendix A.7. As observed, our method consistently performs the best across all pruning ratios.

Number of Samples for Compensation. To verify the efficiency of the compensation process for our method, we conducted experiments using different numbers of samples. The results of these experiments are shown in Figure 4. The results demonstrate that the performance difference between compensation with 128 samples versus 512 or even 1024 samples is minimal across all pruning ratios. This indicates that 128 samples are sufficient for our compensation method, highlighting its efficiency.

Memory & Generation Speed. We show the memory reduction and generation acceleration in Figure 5. The results are obtained using an NVIDIA A100 GPU with a sentence consisting of 64 tokens as the model input. The results show that as the pruning ratio increases, there is a corresponding decrease in GPU memory usage and an increase in generation speed, which validates the effectiveness of our method.

Table 3: Sparse results for image generation task with Llam_agGeenermatoiodenl_fsapmeieldy in $3 8 4 \times 3 8 4$ resolution.   

<html><body><table><tr><td>Method</td><td>Ratio</td><td>FID↓</td><td>SFID↓</td><td>IS↑</td><td>Prec ↑</td><td>Rec 个</td></tr><tr><td colspan="7">LlamaGen-XXL (cfg=1.75)</td></tr><tr><td>/</td><td>/</td><td>2.39</td><td>6.02</td><td>253.16</td><td>80.73%</td><td>59.60%</td></tr><tr><td>FLAP Ours</td><td>10% 10%</td><td>7.87 6.09</td><td>9.92 7.70</td><td>145.25 168.96</td><td>61.96% 70.98%</td><td>63.34% 65.01%</td></tr><tr><td>FLAP</td><td>15%</td><td>15.93</td><td>11.81</td><td>100.48</td><td>52.05%</td><td>62.02%</td></tr><tr><td>Ours</td><td>15%</td><td>11.29</td><td>9.85</td><td>124.31</td><td>62.95%</td><td>65.84%</td></tr><tr><td>FLAP</td><td>20%</td><td>53.86</td><td>20.63</td><td>32.41</td><td>28.31%</td><td>67.14%</td></tr><tr><td>Ours</td><td>20%</td><td>22.45</td><td>14.16</td><td>78.64</td><td>53.68%</td><td>67.66%</td></tr><tr><td></td><td></td><td></td><td></td><td>LlamaGen-3B (cfg=1.65)</td><td></td><td></td></tr><tr><td colspan="7"></td></tr><tr><td>/</td><td>/</td><td>2.26</td><td>6.19</td><td>260.46</td><td>82.07%</td><td>58.35%</td></tr><tr><td>FLAP</td><td>10%</td><td>7.57</td><td>8.40</td><td>158.74</td><td>67.19%</td><td>64.90%</td></tr><tr><td>Ours</td><td>10%</td><td>3.97</td><td>7.45</td><td>202.93</td><td>73.93%</td><td>62.86%</td></tr><tr><td>FLAP</td><td>15%</td><td>38.45</td><td>23.61</td><td>57.29</td><td>43.45%</td><td>63.27%</td></tr><tr><td>Ours</td><td>15%</td><td>8.92</td><td>10.32</td><td>152.36</td><td>66.83%</td><td>63.97%</td></tr><tr><td>FLAP Ours</td><td>20% 20%</td><td>162.15 20.16</td><td>93.98 16.29</td><td>5.97 95.05</td><td>13.36% 54.87 %</td><td>28.03% 63.72%</td></tr></table></body></html>

Table 4: Results for LLaMA-7B model on WikiText2 with 128Rastieoq e0n%ce le1n0gt%h. Fu2l0l%result3s0w%ith 4m0o%re da5t0as%ets are in Table 10 at Appendix B.   

<html><body><table><tr><td>Prune Ratio</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td><td>50%</td></tr><tr><td>LLM-Pruner</td><td>15.37</td><td>19.09</td><td>30.64</td><td>52.28</td><td>122.8</td></tr><tr><td>SliceGPT</td><td>14.52</td><td>19.27</td><td>44.96</td><td>535.5</td><td>2241</td></tr><tr><td>FLAP</td><td>13.84</td><td>14.62</td><td>17.62</td><td>22.32</td><td>31.80</td></tr><tr><td>Ours</td><td>13.31</td><td>14.47</td><td>16.40</td><td>19.04</td><td>23.32</td></tr></table></body></html>

![](images/0c9b94db64e95f7882b1488b4f6e71e387773f3eef6532c0d8121ee4771ffd7a.jpg)  
Figure 5: GPU memory v.s. generation speed.

# 5 Conclusion and Limitation

In this paper, we propose numerical score through Newton’s Method for the minimization of pruning errors. Also, we introduce a compensation algorithm to reconstruct weights. Results show that our method not only achieves the SOTA performance but also reduces memory usage and accelerates generation on GPUs. One limitation of our method is its reduced effectiveness for image generation tasks.