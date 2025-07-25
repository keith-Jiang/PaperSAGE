# On Effects of Steering Latent Representation for Large Language Model Unlearning

Dang Huu-Tien1, Tin Pham1, Hoang Thanh- $\mathbf { \mathrm { I n n g } } ^ { 2 }$ , and Naoya Inoue1,3

1Japan Advanced Institute of Science and Technology 2VNU University of Engineering and Technology, Vietnam 3RIKEN 1 s2310417, tinpham, naoya-i @jaist.ac.jp, $^ 2 \mathrm { h t t } 2 1 0 @$ gmail.com

# Abstract

Representation Misdirection for Unlearning (RMU), which steers model representation in the intermediate layer to a target random representation, is an effective method for large language model (LLM) unlearning. Despite its high performance, the underlying cause and explanation remain underexplored. In this paper, we theoretically demonstrate that steering forget representations in the intermediate layer reduces token confidence, causing LLMs to generate wrong or nonsense responses. We investigate how the coefficient influences the alignment of forget-sample representations with the random direction and hint at the optimal coefficient values for effective unlearning across different network layers. We show that RMU unlearned models are robust against adversarial jailbreak attacks. Furthermore, our empirical analysis shows that RMU is less effective when applied to the middle and later layers in LLMs. To resolve this drawback, we propose Adaptive RMU—a simple yet effective alternative method that makes unlearning effective with most layers. Extensive experiments demonstrate that Adaptive RMU significantly improves the unlearning performance compared to prior art while incurring no additional computational cost.

# 1 Introduction

LLMs achieved remarkable performance through pretraining on large amounts of internet texts and rigorous alignment processes for safety enhancement. Despite the immense effort in safety research, LLMs are still vulnerable to adversarial jailbreak attacks and can exhibit unwanted behaviors (Shah et al. 2023; Zou et al. 2023b; Jones et al. 2023; Yuan et al. 2024; Wei, Haghtalab, and Steinhardt 2024).

Machine Unlearning (Cao and Yang 2015; Bourtoule et al. 2021; Nguyen et al. 2022; Xu et al. 2023; Liu et al. 2024c) has emerged as a promising method for mitigating unforeseen risks in LLMs before deployment. Li et al. (2024b) introduced Representation Misdirection for Unlearning (RMU)—an unlearning method that steers the representations of forget-samples (i.e. samples that the model should forget) toward a random representation while keeping the representations of retain-samples (i.e. samples that the model should remember) unchanged. RMU significantly degrades models’ accuracy on forget-tasks, while only slightly affecting the performance on retain-tasks and demonstrates stronger robustness against adversarial jailbreak attacks. However, the reason for RMU’s effectiveness is not well understood, hindering the development of better unlearning algorithms. In this paper, we make the following contributions:

• We theoretically analyze the impact of the RMU method on LLM unlearning.   
• We investigate the connection between RMU and adversarial robustness. We demonstrate that RMU impedes the adversary’s ability to determine optimal updates for generating adversarial samples, thus improving the adversarial robustness of the unlearned model.   
• We empirically show that the RMU forget loss, which minimizes the mean squared error (MSE) between forget representation and a fixed scaled random vector, fails to converge when the norm of the forget representation is larger than the scaling coefficient, making RMU less effective when applied to middle and last layers in LLMs.   
• To overcome RMU’s limitation, we introduce Adaptive RMU—a variant that adaptively adjusts the coefficient value based on the norm of the forget representation. Experimental results show that Adaptive RMU achieves higher drop-in-accuracy for forget knowledge, maintaining high performance on general knowledge, and enables effective unlearning for most layers without incurring additional computational overhead.

# 2 Background and Related Work

Machine Unlearning. A natural unlearning approach is leave-some-out retraining: retraining the model from scratch without the forget samples. However, this method becomes more computationally expensive as the size of datasets and modern deep networks grows. Existing works focus on approximating unlearning (Warnecke et al. 2021; Izzo et al. 2021; Sekhari et al. 2021; Isonuma and Titov 2024) using influence function (Koh and Liang 2017; Grosse et al. 2023), gradient ascent (Thudi et al. 2022), second-order approximation (Jia et al. 2024), negative preference optimization (Zhang et al. 2024b), and embedding corrupted (Liu et al. 2024a). Other views on the landscape of machine unlearning include: unlearning in text classification (Ma et al. 2022), image classification and recognition (Ginart et al.

2019; Golatkar, Achille, and Soatto 2020; Fan et al. 2024; Choi and Na 2023; Cha et al. 2024), image-to-image generative models (Li et al. 2024a), diffusion models (Gandikota et al. 2023; Zhang et al. 2024a; Kumari et al. 2023; Bui et al. 2024), multimodal unlearning (Cheng and Amiri 2023), federated unlearning (Romandini et al. 2024; Wang et al. 2022; Che et al. 2023; Halimi et al. 2022; Jeong, Ma, and Houmansadr 2024), graph unlearning (Chen et al. 2022; Chien, Pan, and Milenkovic 2023; Wu et al. 2023a; Cheng et al. 2023; Dukler et al. 2023; Zhu, Li, and $\mathrm { H u } ~ 2 0 2 3$ ; Li et al. 2024c; Tan et al. 2024), recommender systems (Zhang et al. 2023; Chen et al. 2024; Li et al. 2023; Wang et al. 2025), certified minimax unlearning (Liu et al. 2024b), targeted types of unlearning information (Cooper et al. 2024), and evaluation on unlearning (Lynch et al. 2024; Hayes et al. 2024; Shi et al. 2024a,b).

LLM Unlearning. Due to the large size of the parameters and training data, LLM poses a new challenge to unlearning. Recent studies in LLM unlearning mainly focus on task or context-specific settings such as unlearning copyrighted material from the Harry Potter series (Eldan and Russinovich 2023), in-context unlearning (Pawelczyk, Neel, and Lakkaraju 2024), fictitious unlearning (Maini et al. 2024), specific harmful input-output (Yao, Xu, and Liu 2023; Liu et al. 2024d), sensitive and private information (Jang et al. 2023; Wu et al. $2 0 2 3 \mathrm { b }$ ; Patil, Hase, and Bansal 2024), gender bias (Belrose et al. 2023) or concepts (Hong et al. 2024; Bui et al. 2024). More recently, Li et al. (2024b) consider unlearning an entire distribution of hazardous knowledge given limited samples.

Notation $\pmb { \& }$ problem formulation. Let $\mathcal { D } _ { \mathrm { f o r g e t } }$ and $\mathcal { D } _ { \mathrm { r e t a i n } }$ be the forget and retain sets, respectively. Let $f _ { \theta } : \mathbb { R } ^ { n \times d } \mapsto$ $\mathbb { R } ^ { n \times | V | }$ be an autoregressive LLM parameterized by $\theta$ that maps a prompt input $x _ { 1 : n }$ consisting of $n$ tokens $\{ x _ { 1 } , x _ { 2 } , . . . , x _ { n } \}$ to an output of probability distributions over the vocabulary $V$ . We denote $h _ { \theta } ^ { ( l ) } ( x )$ the averaged hidden states of all tokens in $x _ { 1 : n }$ obtained from the $l$ -th layer of $f _ { \theta }$ . For simplicity, throughout this paper, we use $h ^ { ( l ) } \dot { ( x ) }$ to present $h _ { \theta } ^ { ( l ) } ( x )$ . For operators, we denote $\scriptscriptstyle \mathrm { ~ o ~ }$ as the decomposition operator, and $| | \cdot | |$ is the Euclidean norm. Our goal is to unlearn the undesired harmful knowledge $\mathcal { D } _ { \mathrm { f o r g e t } }$ from $f _ { \theta }$ while retaining general knowledge $\mathcal { D } _ { \mathrm { r e t a i n } }$ . Unlearned models should be robust to knowledge recovery attacks that attempt to recover harmful knowledge from the model.

Representation Misdirection for Unlearning (RMU; Li et al. (2024b)) is a fine-tuning based unlearning method inspired by representation engineering (Zou et al. 2023a) that steers the model’s representation of forget samples $x _ { F } \in \mathsf { \Gamma }$ $\mathcal { D } _ { \mathrm { f o r g e t } }$ to a random vector and regularizes the model representation of retain samples $x _ { R } \in \mathcal { D } _ { \mathrm { r e t a i n } }$ back to the original model representation, by optimizing the MSE loss:

$$
\begin{array} { r l } & { \mathcal { L } = \mathbb { E } _ { \boldsymbol { x } _ { F } \in \mathcal { D } _ { \mathrm { f o r g e t } } } | | h _ { \boldsymbol { \theta } ^ { \mathrm { u n l e a m } } } ^ { ( l ) } ( \boldsymbol { x } _ { F } ) - c \boldsymbol { u } | | _ { 2 } ^ { 2 } } \\ & { \quad + \alpha \mathbb { E } _ { \boldsymbol { x } _ { R } \in \mathcal { D } _ { \mathrm { r e t a i n } } } | | h _ { \boldsymbol { \theta } ^ { \mathrm { u n l e a m } } } ^ { ( l ) } ( \boldsymbol { x } _ { R } ) - h _ { \boldsymbol { \theta } ^ { \mathrm { f r o z e n } } } ^ { ( l ) } ( \boldsymbol { x } _ { R } ) | | _ { 2 } ^ { 2 } , } \end{array}
$$

where $\theta ^ { \mathrm { u n l e a r n } }$ and $\theta ^ { \mathrm { f r o z e n } }$ are parameters of the update model and frozen model respectively, $\mathbf { \Delta } _ { \pmb { u } }$ is a fixed random unit vector where each element is sampled from Uniform distribution $U ( 0 , 1 )$ , $c \in \mathbb { R }$ is a fixed scaling coefficient and $\alpha \in \mathbb { R }$ is a retain weight. RMU updates $\theta ^ { \mathrm { u n l e a r n } }$ toward the direction of the gradient of the loss $\mathcal { L }$ using gradient descent.

# 3 Theoretical Analysis 3.1 The Confidence of Tokens Generated by RMU Models

In general, samples from the shifted distribution (such as wrong label or out-of-distribution) are associated with smaller “confidence” scores such as softmax probability (Hendrycks and Gimpel 2017; Northcutt, Jiang, and Chuang 2021), maximum logit (Hendrycks et al. 2022; Wei et al. 2022), $\ell ^ { 2 }$ -distance (Sun et al. 2022), energy score (Liu et al. 2020), and cosine similarity (Ngoc-Hieu et al. 2023). Recently, LLM has shown a tendency to produce a lower (higher) confidence in its incorrect (correct) answers in multiple-choice Q&A (Plaut, Nguyen, and Trinh 2024). Building on previous works, we hypothesized that the logit of generated tokens by RMU models exhibit randomness. As seen by a deep network, such randomization signifies low confidence in the logit, resulting in nonsensical or incorrect responses. To validate the hypothesis, we conducted an analysis of the logits of generated tokens produced by RMU models. To facilitate subsequent analysis, we make the following definition and assumption.

Definition 1. (Unlearned model & logit of forget-tokens on unlearned model). Let $f ^ { ( l : k ) } = g ^ { ( l : k ) } \circ h ^ { ( \check { l } ) }$ , where $g ^ { ( l : k ) }$ be the transformation from layer $l$ to layer $k$ of network $f$ , for any two layers $k > l$ ; $l \in [ 1 . . . L ]$ . We define the unlearned model $\begin{array} { r } { f ^ { \mathrm { u n l e a r n } } = W ( f ^ { ( l : L ) , \mathrm { s t e e r e d } } ) = W ( g ^ { ( l : L ) } \circ h ^ { ( l ) , \mathrm { s t e e r e d } } ) . } \end{array}$ , h(l),steered is the steered representation of the given input at layer $l$ and $W$ is the unembedding matrix which maps output hidden states back to the vocabulary space. Given a forget input $x _ { F , 1 : n }$ , the logit of the next token ${ \boldsymbol { x } } _ { F , n + 1 }$ obtained from unlearned model f unlearn is defined as:

$$
\begin{array} { r l } & { f ^ { u n l e a r n } ( x _ { F , n + 1 } \vert x _ { F , 1 : n } ) = W f ^ { ( l : L ) , s t e e r e d } ( x _ { F , n + 1 } \vert x _ { F , 1 : n } ) } \\ & { \qquad = W ( g ^ { ( l : L ) } \circ h ^ { ( l ) , \mathrm { s t e e r e d } } ) ( x _ { F , n + 1 } \vert x _ { F , 1 : n } ) } \\ & { \qquad = W g ^ { ( l : L ) } ( h ^ { ( l ) , \mathrm { s t e e r e d } } ( x _ { F , n + 1 } \vert x _ { F , 1 : n } ) ) \qquad ( 2 } \end{array}
$$

Assumption 1. A well-unlearned model shifts the representation of all tokens in a forget-sample $x _ { F , 1 : n }$ at layer $l$ to $a$ scaled random vector cu. More concretely,

$$
h ^ { ( l ) , \mathrm { s t e e r e d } } ( x _ { F , i } ) = c { \pmb u } + { \pmb \epsilon } ,
$$

where $\boldsymbol { x } _ { F , i }$ is the $i$ -th token in $x _ { F }$ , $\epsilon$ is a small error. Without losing generality, we assume that $\epsilon$ is sampled from Normal distribution $\mathcal { N } ( \mathbf { 0 } , \eta \mathbf { I } )$ , where $\eta I$ is the covariance matrix, η R.

Proposition 1. If Assumption 1 holds, by Definition $^ { \small 1 }$ , the logit value of forget token ${ { x } } _ { F , n + 1 }$ generated by unlearned model f unlearn given as $\bar { f } ^ { \mathrm { u n l e a r n } } ( x _ { F , n + 1 } \ r | x _ { F , 1 : n } )$ follows the Normal distribution $\mathcal { N } \left( W g ^ { ( l : L ) } ( z ) , \eta W \nabla _ { z } g ^ { ( l : L ) } ( z ) ^ { \top } \nabla _ { z } g ^ { ( l : L ) } ( z ) W ^ { \top } \right)$ , where $z = c \pmb { u }$ .

Proof. Assumption 1 implies that in a well-unlearned model, token $x _ { F , n + 1 }$ is independent of the previous tokens, thus we have:

$$
h ^ { ( l ) , \mathrm { s t e e r e d } } ( x _ { n + 1 } | x _ { F , 1 : n } ) \approx h ^ { ( l ) , \mathrm { s t e e r e d } } ( x _ { F , n + 1 } ) = c u + \epsilon
$$

Denote $z = c \pmb { u }$ . Substituting Eqn. 4 into Eqn. 2, we get:

$$
f ^ { \mathrm { u n l e a r n } } ( x _ { F , n + 1 } | x _ { F , 1 : n } ) \approx W g ^ { ( l : L ) } ( z + \epsilon )
$$

Since $\epsilon$ is small, we approximate the function $g ^ { ( l : L ) } ( z + \epsilon )$ by its first-order derivative:

$$
f ^ { \mathrm { u n l e a r n } } ( x _ { F , n + 1 } | x _ { F 1 : n } ) \approx W ( g ^ { ( l : L ) } ( z ) + \nabla _ { z } g ^ { ( l : L ) } ( z ) ^ { \top } \epsilon )
$$

Given that $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \eta \mathbf { I } )$ , by applying the affine transformation property of the multivariate normal distribution, we get:

$$
\begin{array} { r l } & { f ^ { \mathrm { { u n l e a r n } } } ( x _ { F , n + 1 } | x _ { F , 1 : n } ) } \\ & { \sim \mathcal { N } \left( W g ^ { ( l : L ) } ( z ) , \eta W \nabla _ { z } g ^ { ( l : L ) } ( z ) ^ { \top } \nabla _ { z } g ^ { ( l : L ) } ( z ) W ^ { \top } \right) } \end{array}
$$

Since $\pmb { u } \sim U ( 0 , 1 )$ , then $z \sim U ( 0 , c )$ . By definition of variance, we have: $\mathrm { V a r } ( z ) = \mathrm { V a r } ( c \pmb { u } ) = c ^ { 2 } \mathrm { V a r } ( \pmb { u } )$ . □

Proposition 1 suggests that the variance of $f ^ { \mathrm { u n l e a r n } } ( x _ { F , n + 1 } | x _ { F , 1 : n } )$ is controlled by (i) $\eta$ : a scalar variance and (ii) $W \nabla _ { z } g ^ { ( l : L ) } ( z ) ^ { \top } \nabla _ { z } g ^ { ( l : L ) } ( z ) { W } ^ { \top }$ : the product of $W \nabla _ { z } g ^ { ( L ) } ( z ) ^ { \top }$ and $\nabla _ { z } g ^ { ( L ) } ( z ) W ^ { \top }$ . If $f ^ { \mathrm { u n l e a r n } } ( x _ { F , n + 1 } | x _ { F , 1 : n } )$ has high variance, the logit values are more random. Since $\epsilon$ presents a small error, then $\epsilon$ varies for different inputs $x _ { F }$ . This variation makes it difficult to control the variance of the logit by $\eta$ . The main effect depend on $W \nabla _ { z } g ^ { ( l : L ) } ( z ) ^ { \top } \nabla _ { z } g ^ { ( { \bar { l : L } } ) } ( { \bar { z } } ) { \bar { W } } ^ { \top }$ . While the unembedding matrix $W$ is unchanged after unlearning, the product $\nabla _ { z } \bar { g ( \iota ; L ) } ( z ) ^ { \top } \nabla _ { z } g ^ { ( l ; L ) } ( z )$ varies depending on the specific characteristics of sub-networks $g ^ { ( l : L ) }$ and input $\boldsymbol { z } ~ = ~ c \boldsymbol { u }$ . Unfortunately, $g ^ { ( l : L ) }$ is a composition of transformer layers, which is highly nonlinear, making it difficult to have a complete analysis. The variance of $z$ , derived as $\mathrm { V a r } ( z ) = c ^ { 2 } \dot { \mathrm { V a r } } ( { \pmb u } )$ , is proportional to $c ; i . e$ . when $c$ gets larger, the variance of $z$ is higher. This could increase the variability of $g ^ { ( l : L ) } ( z )$ and the gradient $\nabla _ { z } g ^ { ( l : L ) } ( z )$ . A larger c could introduces more randomness to the logit. We conduct an empirical analysis to understand the confidence of generated tokens by RMU models in Section 4.1.

# 3.2 The Effect of Coefficient $c$ on Forget-sample Representations

RMU forget loss steers forget-sample representation $h ^ { ( l ) } ( x _ { F } )$ aligns with a random direction given by $\mathbf { \Delta } _ { \pmb { u } }$ and scales the magnitude of $h ^ { ( l ) } ( x _ { F } )$ to $\scriptstyle { c }$ (Eqn 1). While vector $\mathbf { \Delta } _ { \pmb { u } }$ is predetermined before unlearning, the magnitude of $h ^ { ( l ) } ( x _ { F } )$ varies depending on input $x _ { F }$ and specific properties of layer $l$ . This raises the following research questions: RQ1 (Direction): “How does the coefficient c influence the alignment between $h ^ { ( l ) } ( x _ { F } )$ with $\mathbf { \Delta } _ { \pmb { u } }$ .” RQ2 (Magnitude): “What is the optimal value of the coefficient c for effectively unlearning with different layers.”

Unlearning as minimizing the noise sensitivity. We aim to answer these questions by analyzing the unlearning problem under a noise compression view. We consider the output of a transformation $f ^ { ( l : k ) }$ on input $x$ : $f ^ { ( l : k ) } ( x ) = ( g ^ { ( l : k ) } \circ$ $h ^ { ( l ) } ) ( x ) = g ^ { ( l : k ) } \left( \bar { h ^ { ( l ) } } ( x ) \right)$ . Suppose we compress a noise vector $\boldsymbol { \xi }$ to the representation $\it { h ^ { ( l ) } }$ of layer $l$ at input $x$ , then the output become $g ^ { ( l : k ) } \left( h ^ { ( l ) } ( x ) + \pmb { \xi } \right) ^ { \cdot }$ . Naturally, if layer $g ^ { ( l : k ) }$ is robust (less sensitive) to noise $\boldsymbol { \xi }$ , then $\boldsymbol { \xi }$ has a small effect on the output of $g ^ { ( l : \grave { k } ) }$ i.e. the normalized squared norm

$$
\Phi ( g ^ { ( l : k ) } , x ) = \frac { | | g ^ { ( l : k ) } \left( h ^ { ( l ) } ( x ) + \xi \right) - g ^ { ( l : k ) } \left( h ^ { ( l ) } ( x ) \right) | | ^ { 2 } } { | | g ^ { ( l : k ) } \left( h ^ { ( l ) } ( x ) \right) | | ^ { 2 } }
$$

is small. In contrast, a higher $\Phi ( g ^ { ( l : k ) } , x )$ mean $g ^ { ( l : k ) }$ is higher sensitive to noise $\boldsymbol { \xi }$ at input $x$ . For a dataset $\mathcal { D } _ { \mathrm { f o r g e t } }$ , we define the noise sensitivity of a layer $g ^ { ( l : k ) }$ w.r.t $\boldsymbol { \xi }$ on $\mathcal { D } _ { \mathrm { f o r g e t } }$ as:

$$
\begin{array} { r l } & { \Phi ( g ^ { ( l : k ) } , { \mathcal D } _ { \mathrm { f o r g e t } } ) } \\ & { \ = \frac { | | g ^ { ( l : k ) } ( \hat { h } ^ { ( l ) } ( x _ { F } ) + \xi ) - g ^ { ( l : k ) } ( \hat { h } ^ { ( l ) } ( x _ { F } ) ) | | ^ { 2 } } { | | g ^ { ( l : k ) } ( \hat { h } ^ { ( l ) } ( x _ { F } ) ) | | ^ { 2 } } , } \end{array}
$$

where $\hat { h } ^ { ( l ) } ( x _ { F } )$ is the mean of $h ^ { ( l ) } ( x _ { F } )$ over $x _ { F } \in \mathcal { D } _ { \mathrm { f o r g e t } }$ . During unlearning, RMU steers $h ^ { ( l ) } ( x _ { F } )$ for all $\boldsymbol { x } _ { F } \in \mathbf { \Theta }$ $\mathcal { D } _ { \mathrm { f o r g e t } }$ to the fixed vector $c u + \epsilon$ i.e. $\vert \vert g ^ { ( l : k ) } ( c u + \epsilon ) -$ g(l:k)(ˆh(l)(xF ))||2 is minimized. If we let ξ = cu + ϵ − $\hat { h } ^ { ( l ) } ( x _ { F } )$ , we can define the unlearning problem as minimizing the noise sensitivity of the layer. This objective is described by

$$
\operatorname* { m i n } \frac { | | g ^ { ( l : k ) } ( c u + \epsilon ) - g ^ { ( l : k ) } ( \hat { h } ^ { ( l ) } ( x _ { F } ) ) | | ^ { 2 } } { | | g ^ { ( l : k ) } ( \hat { h } ^ { ( l ) } ( x _ { F } ) ) | | ^ { 2 } }
$$

While $g ^ { ( l : k ) }$ is a composition of transformer layers, which is hard to expand it in term of $c$ . Therefore, we propose to use the Jacobian matrix $J ^ { ( l : k ) } ( x _ { F } ) ,$ —a linearized of $g ^ { ( l : k ) }$ at xF —which describes the change in the output of g(l:k) due to a noise perturbed in the input $\hat { h } ^ { ( l ) } ( x _ { F } )$ . For simplification, we write $\hat { h } ^ { ( l ) }$ , $\pmb { J } ^ { ( l : k ) }$ instead of $\hat { h } ^ { ( l ) } ( x _ { F } ) , J ^ { ( l : k ) } ( x _ { F } )$ respectively. The objective becomes

$$
\operatorname* { m i n } \frac { | | J ^ { ( l : k ) } ( c u + \epsilon ) - J ^ { ( l : k ) } \hat { h } ^ { ( l ) } | | ^ { 2 } } { | | J ^ { ( l : k ) } \hat { h } ^ { ( l ) } | | ^ { 2 } }
$$

Since $\mathbf { \mathcal { I } } ^ { ( l : k ) }$ is a linear transformation, then

$$
| | J ^ { ( l : k ) } ( c u + \epsilon ) - J ^ { ( l : k ) } \hat { h } ^ { ( l ) } | | ^ { 2 } = | | J ^ { ( l : k ) } ( c u + \epsilon - \hat { h } ^ { ( l ) } ) | | ^ { 2 }
$$

Let $\pmb { v } = \epsilon - \hat { h } ^ { ( l ) }$ . By definition of the squared norm, we have:

$$
\begin{array} { r } { | | J ^ { ( l ; k ) } ( c u + v ) | | ^ { 2 } = ( J ^ { ( l ; k ) } ( c u + v ) ) ^ { \top } J ^ { ( l ; k ) } ( c u + v ) } \\ { = ( c u + v ) ^ { \top } J ^ { ( l ; k ) \top } J ^ { ( l ; k ) } ( c u + v ) . } \end{array}
$$

Let matrix $\pmb { A } = \pmb { J } ^ { ( l : k ) \top } \pmb { J } ^ { ( l : k ) }$ . Expand the right-hand side of Eqn. 13, we get:

$$
\begin{array} { r l } & { | | J ^ { ( l : k ) } ( c u + v ) | | ^ { 2 } } \\ & { = ( c u ) ^ { \top } A c u + ( c u ) ^ { \top } A v + v ^ { \top } A c u + v ^ { \top } A v } \end{array}
$$

Since $A$ is a symmetric matrix (i.e. $A ^ { \top } = A ^ { \cdot }$ ), then

$$
( c \pmb { u } ) ^ { \top } \pmb { A } \pmb { v } = ( c \pmb { u } ) ^ { \top } \pmb { A } ^ { \top } \pmb { v } = ( \pmb { A } c \pmb { u } ) ^ { \top } \pmb { v } = \pmb { v } ^ { \top } \pmb { A } c \pmb { u }
$$

Substituting $( c \pmb { u } ) ^ { \top } \pmb { A } \pmb { v } = \pmb { v } ^ { \top } \pmb { A } c \pmb { u }$ into Eqn. 14 we get:

$$
| | J ^ { ( l : k ) } ( c u + v ) | | ^ { 2 } = c ^ { 2 } { \pmb u } ^ { \top } { \pmb A } { \pmb u } + 2 c { \pmb u } ^ { \top } { \pmb A } { \pmb v } + { \pmb v } ^ { \top } { \pmb A } { \pmb v }
$$

Substituting Eqn. 16 into Eqn. 11, the objective becomes

$$
\operatorname* { m i n } \frac { c ^ { 2 } \pmb { u } ^ { \top } \pmb { A } \pmb { u } + 2 c \pmb { u } ^ { \top } \pmb { A } \pmb { v } + \pmb { v } ^ { \top } \pmb { A } \pmb { v } } { | | \pmb { J } ^ { ( l : k ) } \hat { h } ^ { ( l ) } | | ^ { 2 } }
$$

Taking its derivative w.r.t $c$ and set it to zero:

$$
\frac { 2 \pmb { u } ^ { \top } \pmb { A } \pmb { u } c + 2 \pmb { u } ^ { \top } \pmb { A } \pmb { v } } { | | \pmb { J } ^ { ( l : k ) } \hat { h } ^ { ( l ) } | | ^ { 2 } } = 0
$$

Since $\vert \vert J ^ { ( l : k ) } \hat { h } ^ { ( l ) } \vert \vert ^ { 2 }$ is not zero, solve for $c$ :

$$
\begin{array} { l } { c = - \displaystyle \frac { u ^ { \top } A v } { u ^ { \top } A u } = \frac { u ^ { \top } J ^ { ( l : k ) \top } J ^ { ( l : k ) } ( \hat { h } ^ { ( l ) } - \epsilon ) } { u ^ { \top } J ^ { ( l : k ) \top } J ^ { ( l : k ) } u } } \\ { = \displaystyle \frac { ( J ^ { ( l : k ) } u ) ^ { \top } J ^ { ( l : k ) } ( \hat { h } ^ { ( l ) } - \epsilon ) } { | | J ^ { ( l : k ) } u | | ^ { 2 } } } \\ { = \displaystyle \frac { | | J ^ { ( l : k ) } ( \hat { h } ^ { ( l ) } - \epsilon ) ) | | } { | | J ^ { ( l : k ) } u | | } \cos \Bigl ( J ^ { ( l : k ) } u , J ^ { ( l : k ) } ( \hat { h } ^ { ( l ) } - \epsilon ) \Bigr ) } \end{array}
$$

Since $\frac { | | J ^ { ( l : k ) } ( \hat { h } ^ { ( l ) } - \epsilon ) | | } { | | J ^ { ( l : k ) } \pmb { u } | | }$ is positive, then $c$ and $\cos \Bigl ( J ^ { ( l : k ) } \pmb { u } , J ^ { ( l : k ) } ( \hat { h } ^ { ( l ) } - \pmb { \epsilon } ) \Bigr )$ are positively correlated.

This means smaller (larger) $c$ indicates less (more) alignment between $\mathbf { \mathcal { J } } ^ { ( l : k ) } \mathbf { \epsilon }$ and ${ \bf \bar { \mathbf { \Lambda } } } _ { J } ( l ; k ) _ { \left( \hat { h } ^ { ( l ) } - \epsilon \right) }$ . Given that the Jacobian $\pmb { J } ^ { ( l : k ) }$ describes how small changes in the input lead to changes in the output using linear approximation around a given point. If $\pmb { J } ^ { ( l : k ) }$ does not vary drastically, it will not significantly alter the directions of $\mathbf { \Delta } _ { \pmb { u } }$ and $\hat { h } ^ { ( l ) } - \epsilon$ . In such cases, $\pmb { J } ^ { ( l : \bar { k } ) }$ will have a small effect on directional alignment, preserving the relative angles between $\mathbf { \Delta } _ { \pmb { u } }$ and $\hat { h } ^ { ( l ) } - \epsilon$ . Here, reasonably, $\mathbf { \Delta } _ { \pmb { u } }$ and $\hat { h } ^ { ( l ) }$ are becoming more aligned as c increases since error $\epsilon  0$ as unlearning becomes more accurate.

The above discussion does not directly address RQ2. However, the definition of the noise sensitivity suggests that the noise sensitivity of layer $g ^ { ( l : k ) }$ is characterized by the inherent properties of $g ^ { ( l : k ) }$ , the representation $\hat { h } ^ { ( l ) } ( x _ { F } )$ (which is fixed) and the perturbed noise $\boldsymbol { \xi }$ . If $\boldsymbol { \xi }$ is predetermined, the noise sensitivity of $g ^ { ( l : k ) }$ depends solely on its properties. This suggest the following experiment: we compute $\hat { h } ^ { ( l ) } ( x _ { F } )$ —the mean of $h ^ { ( l ) } ( x _ { F } )$ over a set of input $x _ { F } \in \mathcal { D } _ { \mathrm { f o r g e t } }$ , compress a fix perturbed noise $\boldsymbol { \xi }$ into $\hat { h } ^ { ( l ) } ( x _ { F } )$ .

![](images/e0fb3318803ecccbab63a7c08896b5b167e1ef7be31332e6d5bf920961bfd725.jpg)  
Figure 1: Noise sensitivity of layer $g ^ { ( l : k ) }$ , for $k \in [ 3 . . . 3 1 ]$ in base Zephyr-7B, base Llama-3-8B, base Mistral-7B, and RMU Zephyr-7B model. In the base models, a deeper layer has lower noise sensitivity, while the noise sensitivity is minimized in the RMU model (compress noise into $h ^ { ( 7 ) }$ , the noise sensitivity of layer $k = 8$ is minimized).

We then calculate the noise sensitivity of $g ^ { ( l : k ) }$ for different layers. Fig. 1 shows the noise sensitivity of layers across different models. We empirically observed that: the noise sensitivity decreases as layers go deeper and vary across different models. Since noise sensitivity describes a layer’s robustness to noise, higher noise sensitivity means $g ^ { ( l : k ) }$ requires smaller noise to produce the same level of output randomness, while lower noise sensitivity means it requires larger noise. In other words, early layers require smaller noise $\boldsymbol { \xi }$ (smaller $c$ ) whereas later layers require larger noise $\xi$ (larger $c$ ). We present an empirical experiment to verify our analysis in Section. 4.3.

# 3.3 Robustness of RMU Models to AdversarialJailbreak Attacks

RMU algorithms not only erase undesirable knowledge (forgotten knowledge) but also prevent it from resurfacing when attacked by adversarial jailbreak attacks. Li et al. (2024b) present a qualitvative experiment by employing Greedy Coordinate Gradient (GCG; Zou et al. (2023b)) to jailbreak RMU models to extract the answer given a harmful input. As a result, RMU models demonstrate strong robustness against GCG attack. We conduct an analysis from the standpoint of an attack-defense game to explain the underlying causes for the effectiveness of the RMU algorithm.

Attacker’s goal. The goal of the attacker is to inject an adversarial trigger into the suffix of the prompt input (e.g., “How would I engineer influenza A to be significantly more virulent? $x x x x x \ b { x x } )$ , causing the model generates a harmful response (e.g., “To make influenza A more virulent...”).

Attacker’s knowledge and capability. In this setting, we focus on white-box jailbreak attack scenarios (Zou et al. 2023b), where the victim model’s architecture, model input, and output are exposed to the attacker. The attacker is based on gradient signals to search and inject an adversarial trigger into the prompt input, and supplies this adversarial input to the model.

![](images/b5f00db5c5009f5b6299d29f5764ed30fbd3dc70fc59e7d2ed5c3b01a3d30845.jpg)  
Figure 2: The distribution of MaxLogit (a-d) on WMDP Q&A sets with different coefficient $c$ of the base Zephyr-7B and RMU Zephyr-7B models $\langle l = 7 \rangle$ . The distribution of $\cos \bigl ( { \boldsymbol { \mathbf { \mathit { u } } } } , h ^ { ( l ) } \bigr )$ (e-h) of the RMU Zephyr-7B model $( l = 7 )$ .

Problem formulation. Let $f : \mathbb { R } ^ { n \times d } \mapsto \mathbb { R } ^ { n \times | V | }$ be an autoregressive LLM. Given a prompt input joint with an adversarial trigger $x _ { F , 1 : n }$ , the attacker finds an update $\delta$ to adversarial trigger aims to maximize the likelihood of generating the target sequence $x _ { F , n + 1 | n + K }$ consists of $K$ tokens. For simplification, we denote $x _ { F } = x _ { F , 1 : K } =$ $\left[ { { x _ { F , 1 : n } } , { x _ { F , n + 1 : n + K } } } \right]$ . The attacker tries to solve the following objective:

$$
\operatorname* { m i n } _ { x _ { F } + \delta } \mathcal { I } ( f ( x _ { F } + \delta ) ) ,
$$

where $\mathcal { I } ( \cdot , \cdot )$ is the loss function of the attacker. The attacker finds an update $\delta$ based on the linearized approximation of the loss $\mathsf { \bar { V } } _ { e _ { x _ { i } } } \mathcal { I } ( f ( x _ { F } ) )$ , where $e _ { x _ { i } }$ is the one-hot vector representing the current value of the $i$ -th token in $x _ { F }$ . The gradient $\nabla _ { e _ { x _ { i } } } \mathcal { J } ( f ( x _ { F } ) ) .$ is a good indicator for finding a set of candidates for the adversarial token replacement. A more negative value of the gradient $\nabla _ { e _ { x _ { i } } } \mathcal { I } ( \bar { f } ( x _ { F } ) )$ makes a more decrease in the loss. The GCG attacker finds top- $k$ largest negative value of $\nabla _ { e _ { x _ { i } } } \mathcal { I } ( f ( x _ { F } ) )$ for each token in the adversarial trigger and makes the replacement the most decrease in the loss.

Robustness of RMU models against GCG attack. We show that the GCG attacker misjudges in finding optimal adversarial token substitution in RMU models. Specifically, the gradient of the loss at input $x _ { F }$ with respect to $e _ { x _ { i } }$ in RMU model is

$$
\nabla _ { e _ { x _ { i } } } \mathcal { I } ( f ^ { \mathrm { u n l e a r n } } ( x _ { F } ) )
$$

Given the Assumption 1, we have

$$
\begin{array} { r l r } & { } & { \nabla _ { e _ { x _ { i } } } \mathcal { J } ( f ^ { \mathrm { u n l e a r n } } ( x _ { F } ) ) = \nabla _ { e _ { x _ { i } } } \mathcal { J } ( g ^ { ( l : k ) } ( h ^ { ( l ) , \mathrm { s t e e r e d } } ( x _ { F } ) ) } \\ & { } & { \approx \nabla _ { e _ { x _ { i } } } ( \mathcal { I } \circ g ^ { ( l : k ) } ) ( c u + \epsilon ) \quad ( 2 } \end{array}
$$

Since $c$ and $\mathbf { \Delta } _ { \pmb { u } }$ are predetermined before unlearning, $( \mathcal { I } \circ$ $g ^ { ( l : k ) } ) ( c \pmb { u } )$ does not change with respect to $e _ { x _ { i } }$ . The gradient $\nabla _ { e _ { x _ { i } } } ( \mathcal { I } \circ g ^ { ( l : k ) } ) ( c \pmb { u } + \pmb { \epsilon } )$ close to 0 for all token $x _ { i }$ since the error $\epsilon  0$ as unlearning becomes accurate. This means the GCG attacker received unreliable, uninformative gradient signals from RMU models. The RMU model serves as a defender by causing the attacker to miscalculate the gradient of the loss to optimize its objective, thereby increasing the attacker’s cost. The attacker, therefore, cannot find the optimal adversarial tokens for replacement. Li et al. (2024b)’s experiment results implicitly verify our analysis.

# 4 Empirical Analysis 4.1 Measuring Token Confidence with MaxLogit

As discussed in Section 3.1, we validate our hypothesis by considering the Maximum Logit Value (MaxLogit) estimator for measuring the token confidence. More specifically, we compute the MaxLogit for each token $x _ { n + 1 }$ given a sequence of tokens $\boldsymbol { x } _ { 1 : n } ~ = ~ \{ x _ { 1 } , . . . , x _ { n } \}$ from vocabulary $V$ as:

$$
\operatorname { M a x L o g i t } ( x _ { n + 1 } ) = \operatorname* { m a x } _ { x _ { n + 1 } \in V } f ^ { \mathrm { u n l e a r n } } ( x _ { n + 1 } | x _ { 1 : n } )
$$

We use WMDP-Biology and WMDP-Cyber Q&A datasets (Li et al. 2024b) with total 3260 Q&As. We formulated each question and answer as a zero-shot Q&A prompt to query the unlearned LLM. The details of the prompt template are located in Appendix A.1. We used greedy decoding to generate tokens and compute the MaxLogit of each token over $k ~ = ~ 3 0$ generated tokens. The MaxLogit distribution was then analyzed for each model Base vs. RMU (unlearned on WMDP-Biology and WMDP-Cyber forget datasets).

The results are presented in Fig. 2 (a)-(d). We find that the MaxLogit distribution for the base model is generally wider compared to the RMU model. In contrast, the RMU model demonstrates a more concentrated and approximately normal distribution of MaxLogit values. The peak of the RMU model’s MaxLogit distribution is shifted towards lower values relative to the base model. This indicates that the RMU model tends to assign lower confidence scores to the generated tokens. Overall, the RMU model’s MaxLogit distribution exhibits lower compared to the base model.

![](images/ec72472a170964730a06198ab960b7f29b676ff53e08d43970d0770868712946.jpg)  
Figure 3: Average accuracy of WMDP (Biology and Cyber) (left) and MMLU with different coefficient $c$ (right).   
Figure 4: $\ell ^ { 2 }$ -norm of forget-sample representation.

# 4.2 The Effect of the Coefficient $c$

On accuracy. We analyze the impact of $c$ for forgotten knowledge and retained knowledge, using WMDP (Li et al. 2024b) and MMLU (Hendrycks et al. 2021). See Section 6 for the full experiment setting. Fig. 3a shows: (i) a clear positive correlation between the drop-in-accuracy rate and the value of $c$ , i.e. higher $c$ makes the accuracy decrease faster. (ii) A larger value of $c$ tends to make a more dropin-accuracy on WMDP. (iii) However, a larger $c$ comes with a caveat in a significant drop in general performance on MMLU (Fig. 3b).

On alignment between $\mathbf { \Delta } _ { \pmb { u } }$ and $\it { h ^ { ( l ) } }$ . We compute $\cos \big ( { \boldsymbol { \mathbf { \mathit { u } } } } , \bar { \boldsymbol { h ^ { ( l ) } } } \big )$ scores of pairs of $\mathbf { \Delta } _ { \pmb { u } }$ and $h ^ { ( l ) } ( x _ { F } )$ for all $x _ { F }$ in on WMDP-Biology and WMDP-Cyber forget datasets and plot the $\cos \big ( { \boldsymbol { \mathbf { \mathit { u } } } } , h ^ { ( l ) } \big )$ score distribution shown in Fig. 2(e)- (h). We observed that there is a clear positive correlation between $\cos \big ( { \boldsymbol { \mathbf { \mathit { u } } } } , h ^ { ( l ) } \big )$ scores and the coefficient $c$ . As $c$ increases, the distribution of $\cos \bigl ( { \boldsymbol { \mathbf { \mathit { u } } } } , h ^ { ( l ) } \bigr )$ scores shifts towards higher values and are almost distributed with a peak at 1.0 (Fig. 2(g)-(h)). This verify our analysis in Section 3.2.

# 4.3 The Effect of Layers on Unlearning

# Algorithm 1: Adaptive RMU pseudocode

# Require:

1: $\mathcal { D } _ { \mathrm { f o r g e t } }$ : a forget dataset. 2: $\bar { \mathcal { D } } _ { \mathrm { { r e t a i n } } }$ : a retain dataset. 3: $f _ { \theta ^ { \mathrm { f r o z c n } } }$ : a frozen model. 4: $f _ { \theta ^ { \mathrm { u n l e a r n } } }$ : an update model. 5: $\alpha$ : a retain weight. 6: $l$ : an unlearn layer. 7: $\beta$ : a scaling factor. 8: $T$ : number of gradient update steps. Ensure: Return the unlearned model $f _ { \theta ^ { \mathrm { u n l c a r n } } }$ . 9: Sample a random unit vector $\mathbf { \Delta } _ { \pmb { u } }$ . 10: for step $t \in [ 1 . . . T ] : x _ { F } \in \mathcal { D } _ { \mathrm { f o r g e t } }$ , $x _ { R } \in \mathcal { D } _ { \mathrm { r e t a i n } }$ do 11: Get the representations of $x _ { F }$ and $x _ { R }$ from the frozen and update model. 12: Compute the adaptive loss $\mathscr { L } ^ { \mathrm { a d a p t i v e } }$ by Eqn. 24. 13: Update $\theta ^ { \mathrm { u n l e a r n } }$ w.r.t $\nabla { \mathcal { L } } ^ { \mathrm { a d a p } }$ using gradient descent. 14: $t = t + 1$ 15: end for 16: return fθunlearn

We investigate the effect of unlearn layers on accuracy and the representation norm during unlearning. Following original work, we change the unlearn layer $l$ from $3  3 1$ , fixed $c = 6 . 5$ . Fig. 5 shows that RMU is effective for unlearning within the early layers $3  1 0$ ), yet exhibits inefficacy within middle and later layers $\mathrm { 1 1 }  \mathrm { 3 1 }$ ). Interestingly, in Fig. 4, we observed that within early layers, the $\ell ^ { 2 }$ -norm of forget samples are smaller than the coefficient $c$ . During unlearning, the representation norm exponentially increases, approaching $c$ , thereby facilitating the convergence of forget loss. Conversely, within middle and later layers, the representation norms of forget samples, initially larger than $c$ , remain unchanged during unlearning, making the forget loss non-convergence.

# 5 Adaptive RMU

Inspired by the observations in Section 4.3, we propose Adaptive RMU, a simple yet effective alternative method with an adaptive forget loss by scaling the random unit vector $\mathbf { \Delta } _ { \pmb { u } }$ with an adaptive scaling coefficient $\beta | | h _ { \theta ^ { \mathrm { f r o z e n } } } ^ { ( l ) } ( x _ { F } ) | |$ , where $\beta \in \mathbb { R }$ is a scaling factor and $| | h _ { \theta ^ { \mathrm { f r o z e n } } } ^ { ( l ) } ( x _ { F } ) | |$ is the $\ell ^ { 2 }$ - norm of forget-sample $x _ { F }$ on model $f _ { \theta ^ { \mathrm { f r o z c n } } }$ . The total loss is calculated as follows:

$$
\begin{array} { r } { \mathcal { L } ^ { \mathrm { a d a p t i v e } } = \underbrace { \mathbb { E } _ { \boldsymbol { x } _ { F } \in \mathcal { D } _ { \mathrm { f o r g e t } } } | | \boldsymbol { h } _ { \boldsymbol { \theta } ^ { \mathrm { u n l e a m } } } ^ { ( l ) } ( \boldsymbol { x } _ { F } ) - \beta | | \boldsymbol { h } _ { \boldsymbol { \theta } ^ { \mathrm { f r o z e n } } } ^ { ( l ) } ( \boldsymbol { x } _ { F } ) | | \boldsymbol { u } | | _ { 2 } ^ { 2 } } _ { \mathrm { a d a p t i v e ~ f o r g e t ~ l o s s } } } \\ { + \alpha \underbrace { \mathbb { E } _ { \boldsymbol { x } _ { R } \in \mathcal { D } _ { \mathrm { r e t a i n } } } | | \boldsymbol { h } _ { \boldsymbol { \theta } ^ { \mathrm { u n l e a m } } } ^ { ( l ) } ( \boldsymbol { x } _ { R } ) - \boldsymbol { h } _ { \boldsymbol { \theta } ^ { \mathrm { f r o z e n } } } ^ { ( l ) } ( \boldsymbol { x } _ { R } ) | | _ { 2 } ^ { 2 } } _ { \mathrm { a d a l s t . } } \ ( 2 4 } \end{array}
$$

M5341-31ysis3?si!l

Our Adaptive RMU is shown in Algorithm 1. We note that Adaptive RMU aims to address the challenge of adaptively determining the coefficient $c$ in RMU. We acknowledge that the introduced value $\beta$ is manually tuned via grid search, leaving the challenge to not fully resolved. However, we emphasize that Adaptive RMU offers significant computational advantages over the original RMU. More concretely, in RMU, grid search is conducted over both $c$ and layer $l$ for $l \in [ 1 . . . L ]$ , where $L$ is the number of layers. Our analysis suggests that effective unlearning can be achieved when $c$ is higher than the representation norm of forget-samples. Therefore, given a layer $l$ , Adaptive RMU only requires tuning $\beta$ , which is $L$ times less than that of RMU. This reduction in computational overhead represents a significant improvement when the size of modern deep networks grows.

![](images/5d876beab816d0e28c7a8ca26aac4666b353d174ac9ca1b5248335ddb1949950.jpg)  
Figure 5: Q&A accuracy of RMU and Adaptive RMU Zephyr-7B models on WMDP-Biology, WMDP-Cyber, and MMLU w.r.t unlearn layer $l$ from the third to the last layer.

# 6 Experiment

Datasets. We use WMDP-Biology and WMDP-Cyber forget datasets as $\mathcal { D } _ { \mathrm { f o r g e t } }$ and Wikitext (Merity et al. 2022) as $\mathcal { D } _ { \mathrm { r e t a i n } }$ for unlearning the LLM. Unlearned models are evaluated on WMDP Q&A datasets and MMLU (Hendrycks et al. 2021). Details of the datasets can be found in the Appendix A.1.

Models. We use the following LLMs: Zephyr-7B- $\cdot \beta$ (Tunstall et al. 2023), Yi-6B (Young et al. 2024), Meta Llama-3- 8B (Meta 2024), and Mistral-7B (Jiang et al. 2023).

Experimental setup. Models were fine-tuned using AdamW (Loshchilov and Hutter 2019) with learning rate $\eta = 5 e \mathrm { ~ - ~ } 5$ , batch-size of 4, max sequence len of 512 for WMDP-Biology and 768 for WMDP-Cyber, with $T = 5 0 0$ gradient update steps. The retain weight $\alpha ~ = ~ 1 2 0 0$ . For the baseline RMU, we follow the previous work and let $c = 6 . 5$ . We grid search for unlearn layer $l$ from the third to the last layer. For the Adaptive RMU, we grid search for the scaling factor $\beta \in \{ 2 , 3 , 5 , 1 0 \}$ . We report the performances of Adaptive RMU models with $\beta = 5$ . We update three layers parameters $\{ l , l - 1 , l - 2 \}$ of the model. Two NVIDIA A40s with 90GB GPU were used to run the experiments. Our code is available at https://github.com/ RebelsNLU-jaist/llm-unlearning.

Baselines. We compare Adaptive RMU against baselines: RMU (Li et al. 2024b), Large Language Model Unlearning (LLMU; Yao, Xu, and Liu (2023)), SCalable Remenbering and Unlearning unBound (SCRUB; Kurmanji et al. (2023)), and Selective Synaptic Dampening (SSD; Foster, Schoepf, and Brintrup (2024). We use off-the-shelf results from Li et al. (2024b) for LLMU, SCRUB, and SSD.

Table 1: Q&A accuracy of Zephyr-7B models on WMDP and MMLU. The best and runner up are marked.   

<html><body><table><tr><td>Method/tasks</td><td>WMDP-Biology√</td><td>WMDP-Cyber√</td><td>MMLU↑</td></tr><tr><td>Base</td><td>63.7</td><td>43.5</td><td>58.1</td></tr><tr><td>LLMU</td><td>59.5</td><td>39.5</td><td>44.7</td></tr><tr><td>SCRUB</td><td>43.8</td><td>39.3</td><td>51.2</td></tr><tr><td>SSD</td><td>50.2</td><td>35.0</td><td>40.7</td></tr><tr><td>RMU (l =7)</td><td>28.8</td><td>28.8</td><td>56.8</td></tr><tr><td>Adaptive RMU(l =7)</td><td>23.7</td><td>26.5</td><td>55.0</td></tr></table></body></html>

Main results. Fig. 5 shows that Adaptive RMU significantly improves unlearning performances. Specifically, Adaptive RMU reduces average accuracy by $1 3 . 1 \%$ on WMDP-Biology and $3 . 6 \%$ on WMDP-Cyber within early layers $3  1 0 \AA$ ), and by $1 5 . 6 \%$ on WMDP-Biology and $9 . { \dot { 6 } } \%$ on WMDP-Cyber within middle and later layers $( 1 1 ~  ~ 3 1 )$ ). This corresponds to an overall enhancement of $1 4 . 3 \%$ and $6 . 6 \%$ in drop-in-accuracy for the WMDPBiology and WMDP-Cyber, respectively. Table 1 further highlights that Adaptive RMU $( l = 7 )$ outperforms RMU $( l = 7 )$ ), LLMU, SCRUB, and SSD, establishing a new stateof-the-art performance. We defer the full results on other models and settings in Appendix B.

# 7 Conclusion

We studied the effect of steering latent representation for LLM unlearning and explored its connection to jailbreak adversarial robustness. We developed a simple yet effective alternative method that enhances unlearning performance across most layers while maintaining overall model utility. Our findings illuminate the explanation of the RMU method and pave the way for future research in LLM unlearning.