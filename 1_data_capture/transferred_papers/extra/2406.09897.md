# 3D-RPE: Enhancing Long-Context Modeling Through 3D Rotary Position Encoding

Xindian $\mathbf { M } \mathbf { a } ^ { 1 }$ , Wenyuan Liu1, Peng Zhang1\*, Nan Xu2

1College of Intelligence and Computing, Tianjin University, Tianjin, China 2Beijing Wenge Technology Co. xindianma $@$ tju.edu.cn, 1wy $2 0 2 0 @$ tju.edu.cn, pzhang $@$ tju.edu.cn, xunan2015 $@$ ia.ac.cn

# Abstract

An essential component in Large Language Models (LLMs) is Rotary Position Encoding (RoPE) , which efficiently manages positional dependencies in long-context modeling. However, when the number of input tokens surpasses the pretrained capacity of LLMs, their ability to process and generate text is markedly weakened. Although position interpolation techniques for RoPE can mitigate this issue, an increase in interpolations leads to a decrease in positional resolution. To tackle this challenge, drawing inspiration from the Bloch Sphere representation, we propose a novel rotary position encoding on a three-dimensional sphere, named 3D Rotary Position Encoding (3D-RPE). 3D-RPE is an advanced version of the widely used 2D RoPE, with two major advantages for modeling long contexts: controllable long-term decay and improved position resolution. For controllable long-term decay, 3D-RPE allows for the regulation of long-term decay within the chunk size, ensuring the modeling of relative positional information between tokens at a distant relative position. For improved position resolution, 3D-RPE can mitigate the degradation of position resolution caused by position interpolation on RoPE. We have conducted experiments on long-context Natural Language Understanding (NLU) and long sequence Language Modeling (LM) tasks. From the experimental results, 3D-RPE achieved performance improvements over RoPE, especially in long-context NLU tasks.

# Introduction

Rotary Position Encoding (RoPE) (Su et al. 2024) is essential in Transformer-based Large Language Models (LLMs), such as the LLaMA models (Touvron et al. 2023). RoPE merges the advantages of absolute and relative positional encoding by using a rotation mechanism to represent each position. Despite its widespread use in LLMs (Touvron et al. 2023; Wang and Komatsuzaki 2021; Chiang et al. 2023), RoPE has notable limitations when extending LLMs with a predefined context window. The long-term decay problem of RoPE limits the model’s ability to extend positions outward in longcontext tasks. Although the long-context modeling capability of LLMs can be extended through position interpolation, as more positions are inserted, RoPE encounters the challenge of decreased position resolution (An et al. 2024).

We propose a novel position encoding mechanism for transformer architecture, called 3D Rotary Position Encoding (3DRPE), to address challenges in long-context modeling faced by LLMs using RoPE. Inspired by the Bloch Sphere, 3DRPE applies rotary position encoding on a three-dimensional spherical surface, as illustrated in Figure 1(b). In contrast, RoPE employs a rotation on a 2-dimensional circular path, as depicted in Figure 1(a). This leads that RoPE suffers from long-term decay. As shown in Figure 1(c), with the increase in relative distance, the relative upper bound on token correlations at modeled relative positions will continuously decrease. Our proposed 3D-RPE addresses this issue by segmenting a long sequence into chunks and setting rotation angles within and between the chunks to construct position encoding. As shown in Figure 1(d), 3D-RPE is able to control this relative upper bound through two relative positional dimensions, namely within and between chunks. Compared to the relative upper bound in RoPE shown in Figure 1(c), our method improves the correlation upper bound for long relative distances and effectively mitigates the problem of long-term decay.

Furthermore, our proposed 3D-RPE alleviates the problem of reduced positional resolution caused by Position Interpolation (PI) (Chen et al. 2023a) on RoPE in long-context modeling. PI methods are often employed to extend LLMs for modeling contexts that exceed the pre-training length. These techniques scale the position encoding during inference, allowing the originally out-of-range position encoding to fall within the trained position interval after interpolation. However, as the interpolation factor increases, PI experiences a substantial decline in positional resolution among tokens, detrimentally affecting long-context modeling performance. As illustrated in Figure 1(e), extending the pre-training length $L _ { p }$ to $L$ using linear PI (Chen et al. 2023a) results in the positional resolution transitioning from the original 1 to $\frac { L _ { p } } { L _ { . } }$ . As $L$ increases, the positional resolution decreases accordingly. However, our proposed 3D-RPE employs a 3D rotating sphere for position encoding. Based on the same positional interpolation, our method supports higher positional resolution compared to RoPE’s 2D circular rotation, i.e., $\begin{array} { r } { \mathcal { E } _ { 3 d - r p e } ^ { \prime } > \frac { L _ { p } } { L } } \end{array}$ (See Figure 1(f)). This benefit has been theoretical−ly proven (see Theorem 1) and corroborated by experimental results (see Table 4 in Ablation Study).

We conducted experiments on long-context Natural Language Understanding (NLU) and long-sequence Language Modeling (LM) tasks. Our experimental results highlight the promising performance of the 3D-RPE method, especially in tasks requiring long-context language understanding.

![](images/3be6c9a9c4019c69ccfd606083a07abf234f1e48ccd0f5aa76dc65b5ae45e64f.jpg)  
Figure 1: 2D Rotary Position Encoding (RoPE) vs. 3D Rotary Position Encoding (3D-RPE).

Our major contributions of this paper are as follows:

• A position encoding method on a 3D sphere, 3D-RPE, is provided, which can enhance the long-context modeling capability of LLMs by replacing RoPE. • It is proved that 3D-RPE has two benefits, controllable long-term decay and mitigating the reduction in positional resolution caused by position interpolation. • LLMs combine with 3D-RPE have achieved significant performance improvements in long-context NLU tasks.

# Preliminaries

The analysis of 3D-RPE relies on these concepts and results from the filed of Bloch Sphere and RoPE. We offer an introduction to Bloch Sphere and RoPE (Su et al. 2024).

# Bloch Sphere

Bloch Sphere (BS) offers a geometric depiction of a quantum mechanical system’s pure state, limited to two levels. The state vector $| \dot { \phi } \rangle$ is mathematically expressed as

$$
| \phi \rangle = e ^ { \mathrm { i } \theta } \bigl ( \cos { \frac { \varphi } { 2 } } | 0 \rangle + \sin { \frac { \varphi } { 2 } } e ^ { i \theta _ { 1 } } | 1 \rangle \bigr )
$$

where $| 0 \rangle$ and $| 1 \rangle$ are Dirac’s notations. $\theta , \theta _ { 1 }$ and $\varphi$ are rotation angles.

In our work, $\theta$ encodes the relative positions of tokens within chunks, $\varphi$ encodes the relative positions of tokens across chunks, and $\theta _ { 1 }$ is equal to 0.

# Rotary Position Encoding

Rotary Position Encoding (RoPE) is a commonly used relative position encoding technique in LLMs, such as LLaMA (Touvron et al. 2023), GPT-J (Wang and Komatsuzaki 2021), Vicuna (Chiang et al. 2023) and etc. RoPE is a 2-dimensional space rotary encoding, which is denoted as follows:

$$
\mathit { R o P E } ( \pmb { h _ { m } } , m ) = e ^ { \mathrm { i } m \theta } \pmb { h _ { m } } \mathrm { ~ , ~ } \mathit { R o P E } ( \pmb { h _ { n } } , n ) = e ^ { \mathrm { i } n \theta } \pmb { h _ { n } }
$$

$\pmb { h } _ { m }$ and $\pmb { h } _ { n }$ are hidden vectors from the Query and Key for a specific attention head in transformer. For ease of differentiation, $\boldsymbol { h _ { m } }$ and $\pmb { h } _ { n }$ can be refined later as $\scriptstyle { { q } _ { m } }$ and $\pmb { k } _ { n }$ , i is the imaginary unit, $\theta$ is the rotary angle in RoPE. $m$ and $n$ are indexes about positions. Then, the inner product is employed to define the self-attention score before softmax computing:

$$
\begin{array} { l } { { \displaystyle s ( m - n , { \bf q } _ { m } , { \bf k } _ { n } ) = \langle R o P E ( { \bf q } _ { m } , m ) , R o P E ( { \bf k } _ { n } , n ) \rangle } } \\ { ~ } \\ { { \displaystyle \quad = R e [ \sum _ { l = 0 } ^ { d / 2 - 1 } { \bf q } _ { [ 2 l : 2 l + 1 ] } { \bf k } _ { [ 2 l : 2 l + 1 ] } e ^ { { \bf i } ( m - n ) \theta _ { l } } ] } } \end{array}
$$

Eq (3) is unary function respect to the relative position $( m -$ $n$ , representing the relative position between tokens( an−d modeling the relative positional information. Here, $R e [ \cdot ]$ denotes the calculation of the real part of a complex numb[e⋅]r.

Chunk j j𝜑 ℎ𝑗2,𝑚 ℎ𝑗1,𝑚 Chunk j  
𝑚𝜃ℎ𝑗,𝑚  
[ℎ𝑗,1, ℎ𝑗,2, … , ℎ𝑗,𝑐−1] 1 ℎ𝑗2,𝑚 [ℎ𝑗,1, ℎ𝑗,2, … , ℎ𝑗,𝑐−1]  
Long Sequence0 1 2 3 4 5 𝐿 − 2 𝐿 − 1  
Divide  into Chunks 10 1 2 3 𝑐 − 2 𝑐 − 1 0 1 2 3 𝑐 − 2 𝑐 − 1  
Chunk 0  
Chunk 1 ： ： ： ·  
Chunk $j$ ： ： ： ： ：  
Chunk Query/ Key 3D-RPE Query/ Key

In our study, the 3D-RPE self-attention score is a binary function containing the relative position $( m - n )$ .

# Method

In this section, we first introduce the new position encoding on a 3D sphere, 3D-RPE. Then, the benefits of 3D-RPE are described, which focuses on analyzing two benefits of 3DRPE, namely controllable long-term decay and improved position resolution.

# 3D Rotary Position Encoding

For a long sequence of length $L$ and a chunk size set to $c$ , where $c$ is smaller than the pre-training length of LLM, the sequence can be divided into $\lceil L / c \rceil$ chunks. Here, . represents the ceiling function, rounding up to the nearest integer (see Figure 2). The state vector $\pmb { h } _ { j , m }$ comes from either Query or Key. Here, $j \in [ 0 , \lceil L / c \rceil - \tilde { 1 } ]$ represents the positional index of the chunk∈, [and⌈ $m \in [ 0 , c - 1 ]$ indicates the positional index of the token withi∈n[the c−hu]nk. This is used to calculate the new state vector $\widetilde { h } _ { j , m }$ by rotating on the Bloch Sphere. Specifically, two rotation angles, $\theta$ and $\varphi$ are defined, with $\theta$ governing the position encoding within the chunk’s internal tokens, and $\varphi$ governing the position encoding between the chunks. Our position encoding method is called 3D Rotary Position Encoding, or 3D-RPE.

Definition 1 (3D Rotary Position Encoding). Let $h _ { j , m } \in$ $\mathbb { R } ^ { d }$ be a state vector of an attention head without position∈ encoding, where $d$ is the dimension of the vector, which is an even number. 3D-RPE encodes $\pmb { h } _ { j , m }$ into the vector $\widetilde { h } _ { j , m }$ , which can be formalized as:

$$
\widetilde { \pmb { h } } _ { j , m } = e ^ { \mathrm { i } m \theta } \big ( \cos \varphi _ { j } \pmb { h } _ { j , m } ^ { \perp } + \sin \varphi _ { j } \pmb { h } _ { j , m } \big )
$$

i is the imaginary unit. $h _ { j , m } ^ { \perp }$ equals to $\left[ - h _ { j , m } ^ { 2 } , h _ { j , m } ^ { 1 } \right] ^ { T }$ , where $h _ { j , m } ^ { 1 } \in \mathbb { R } ^ { d / 2 }$ and $h _ { j , m } ^ { 2 } \in \mathbb { R } ^ { d / 2 }$ is th [fi−rst and seco]nd halves of the∈state vector hj,m∈.

In transformer-based LLMs, after applying position encoding to the state vectors from Query and Key, it is essential to compute their attention scores. For the sake of clarity and formalization, we denote the position encoding of the state vector from Query as $3 \mathrm { d - P E } ( \pmb q , i , m )$ and from Key as 3d$\mathrm { P E } ( k , j , n )$ , where $i$ and $j$ ran(ge from)0 to $\lceil L / c \rceil - 1$ , and $m$ and( $n$ rang)e from 0 to $c - 1$ . The self-attent⌈io/n s⌉c−ore can be obtained through the co n−jugate symmetric inner product of $\mathbf { \Delta } q _ { i , m }$ and $\boldsymbol { k } _ { j , n }$ , which are the state vectors from Query and Key,

$$
s ( \mathbf { q } _ { i , m } , \boldsymbol { k } _ { j , n } , \varphi _ { i } - \varphi _ { j } , m - n ) =
$$

$$
R e [ e ^ { \mathrm { i } ( \varphi _ { i } - \varphi _ { j } ) } \sum _ { l = 0 } ^ { d / 2 - 1 } e ^ { \mathrm { i } ( m - n ) \theta _ { l } } ( q _ { l } k _ { l } + q _ { d / 2 + l } k _ { d / 2 + l } ) ]
$$

where $l \ \in \ [ 0 , \frac { d } { 2 } - 1 ]$ , $\theta _ { l } ~ = ~ b a s e ^ { - l }$ , $\varphi _ { j } ~ = ~ b a s e ^ { - j }$ and $\varphi _ { i } =$ $b a s e ^ { - i }$ . Let $\{ \pmb q , \pmb k \} _ { l }$ den e the $l$ -th components of $\{ \boldsymbol { q } , \boldsymbol { k } \}$ . n experiments{using}the LLaMA2 models, the base is{gener}ally set to $1 0 , 0 0 0$ . In LLaMA3 models, the base of $\theta _ { l }$ is $5 0 , 0 0 0$ . The self-attention score computed after applying 3d-PE is a function of both the relative position between chunks $( \varphi _ { i } - \varphi _ { j } )$ and the relative position $( m - n )$ .

Consequently, the self-att−ention score relying on 3d-PE is influenced by the relative positions at both the chunk and token levels. It is important to highlight that when $\mathbf { q } _ { i , m }$ and $\boldsymbol { k } _ { j , n }$ reside within the same chunk (i.e., $i = j ,$ ), Eq. (5) simplifies to the standard RoPE formulation as=depicted in Eq. (3). For a detailed derivation and computation process of Eq. (5), as well as the complete formulation of Eq. (4).

# Benefits of 3D-RPE

In this section, we delve into two benefits offered by 3D-RPE: the ability to control long-term decay and mitigate the reduction in positional resolution caused by position interpolation.

Controllable Long-term Decay 3D-RPE has the property of controllable long-term decay. Analogous to RoPE, by considering the absolute value $s$ in Eq (5) and utilizing the Abel transformation, we derive the upper bound of the correlation coefficients related to term dependencies as follows:

$$
| s ( q _ { i , m } , k _ { j , n } , \varphi _ { i } - \varphi _ { j } , m - n ) | \leq | e ^ { \mathrm { i } ( \varphi _ { i } - \varphi _ { j } ) } | \cdot
$$

$$
| \sum _ { l = 0 } ^ { \frac { d } { 2 } - 1 } E _ { l + 1 } \big ( h _ { l + 1 } - h _ { l } \big ) | \leq \big ( \operatorname* { m a x } _ { l } \big | h _ { l + 1 } - h _ { l } \big | \big ) \sum _ { l = 0 } ^ { d / 2 - 1 } | E _ { l + 1 } |
$$

where denotes multiplication, $\begin{array} { r } { E _ { l } \ = \ \sum _ { k = 0 } ^ { l - 1 } e ^ { \mathrm { i } ( m - n ) \theta _ { k } } } \end{array}$ and $E _ { 0 } = 0$ .⋅ For RoPE (Su et al. 2024), t =e e∑lat=ive upper bound $E _ { r o p e }$ is given by $\textstyle { \frac { 1 } { d / 2 } } \sum _ { j = 1 } ^ { d / 2 } | S _ { j } |$ , where $\begin{array} { r } { S _ { j } = \sum _ { t = 0 } ^ { j - 1 } e ^ { i ( m - n ) \theta _ { t } } } \end{array}$ (see the section 3.4/.3 of RoPE (Su et al. 2024)). By setting $\theta _ { t } = 1 0 0 0 0 ^ { \frac { - 2 t } { d } }$ , the value decays as the relative position $( m - n )$ increases. The upper bound $E _ { 3 d - r p e }$ of 3D-RPE is f(orm−ali)zed as follows:

$$
E _ { 3 d - r p e } = \frac { 1 } { d / 2 } \sum _ { j = 1 } ^ { d / 2 } | E _ { l } |
$$

[0,0] [0,1][0,0] Chunk O [0,2][0,1][0,0] [0,3][0,2][0,1][0,0] [1,0] [0,3][0,2][0,1] [0,0] [1,1][1,0] [0,3] [0,2][0,1][0,0] Chunk 1 [1,2][1,1] [1,0] [0,3][0,2][0,1][0,0] [1,3][1,2] [1,1][1,0] [0,3][0,2][0,1][0,0] [2,0] [1,3] [1,2][1,1] [1,0] [0,3][0,2][0,1][0,0] [2,1] [2,0] [1,3] [1,2 [1,1] [1,0] [0,3] [0,2] [0,1][0,0] Chunk2 [2,2] [2,1] [2,0] 1 [1,1 [1,0] [0,3] [0,2][0,1][0,0] [2,3] [2,2] [2,1] [2,0] [1,3][1,2] [1,1] [1,0] [0,3][0,2][0,1][0,0] Position ids of $k$ Chunk O Chunk 1 Chunk 2

The domains of the relative position $( m - n )$ differ between $E _ { 3 d - r p e }$ and $E _ { r o p e }$ . In $E _ { r o p e }$ , $( m - n )$ is−in t)he range $[ 0 , L -$ 1 , w−hile in $E _ { 3 d - r p e }$ , it is in $[ 0 , c - 1 ]$ ). The relative po[sition−s b]etween tokens−exceeding t[he ch−unk] size $c$ are constructed collaboratively using positional encoding within and across chunks. The Relative Position Matrix $A$ using 3D-RPE is shown in Figure 3.

To compare and illustrate the advantage of controllable long-term decay, we present the results in Figure 1(c) and Figure 1(d). As shown in Figure 1(c), when the relative position $( m - n )$ exceeds approximately 1000, $E _ { r o p e }$ begins to signific(ant−ly d)ecrease to below 5. This limitation of $E _ { r o p e } \leq 5$ poses challenges for RoPE in modeling attention scores≤between tokens with longer relative distances (greater than 4000). In contrast, as shown in Figure 1(d), 3D-RPE employs both $( m - n )$ and $( \varphi _ { i } - \varphi _ { j } )$ , setting $c = 1 0 0 0$ to keep $( m - n )$ (with−in 1)000,  (hereb−y pr)eventing dec=ay over longer d(istan−ces). This method ensures $E _ { 3 d - r p e }$ stays at or above 5 for all relative positions.

Improved Positional Resolution Position Interpolation (PI) (Chen et al. 2023a) has been introduced to scale down the position indices to align with the original window size, resulting in enhanced outcomes for context extension. However, as the extension length and interpolation increase, PI can lead to a reduction in relative positional resolution. In contrast, 3D-RPE can also be used alongside PI for longcontext extensions. Compared to RoPE combined with PI, 3D-RPE has the advantage of mitigating the reduction in positional resolution caused by positional interpolation, as demonstrated in Theorem 1.

Theorem 1 (Improved Position Resolution). For a pretrained language model with a length of $L _ { p }$ and an extension length requirement of $L$ , employing linear position interpolation extension methods $\boldsymbol { \mathcal { T } }$ based on Rotary Position Encoding $( R o P E )$ can elevate the relative positional resolution from $\mathcal { E } _ { r o p e }$ to $\mathcal { E } _ { r o p e } ^ { \prime }$ . Let $\mathcal { E } _ { 3 d - r p e } ^ { \prime }$ denote the relative positional encoding resolution achi−eved by the method $\boldsymbol { \mathcal { T } }$ based on $3 D$ - $R P E$ , with chunk size $c \geq 3$ , there is:

$$
\mathcal { E } _ { 3 d - r p e } ^ { \prime } > \mathcal { E } _ { r o p e } ^ { \prime }
$$

Proof. For 3D-RPE, let the chunk size and chunk number be denoted as $c$ and $n = \lceil L _ { p } / c \rceil$ respectively. Prior to interpolation, the indices wit=h⌈in a/c⌉hunk range from $[ 0 , 1 , \cdots , c ^ { - } 1 ]$ . Linear interpolation involves evenly distribut[ing th⋯e ex−ces]s $L - L _ { p }$ tokens across $n$ chunks. This results in new indic−es within the chunk, range from $[ 0 , 1 , 2 , \cdots , c ^ { \prime } - 1 ]$ , where $c ^ { \prime } = \lceil L / n \rceil \leq L _ { p }$ . So the attention s[core of $\pmb q _ { i , m + 1 }$ a]nd $\pmb { k } _ { i , m }$ bas=ed⌈ o/n 3⌉D≤-RPE after interpolation is:

$$
\begin{array} { c } { { a _ { 3 d - r p e } = { \pmb q } { \pmb k } ^ { T } e ^ { \mathrm { i } \theta } e ^ { \mathrm { i } \left( \varphi _ { i } - \varphi _ { i } \right) } } } \\ { { = { \pmb q } { \pmb k } ^ { T } e ^ { \mathrm { i } \theta } } } \end{array}
$$

The resolution of relative position for 3D-RPE is:

$$
\mathcal { E } _ { 3 d - r p e } ^ { \prime } = 1
$$

For special cases $\pmb q _ { ( i + 1 , 0 ) }$ and $k _ { ( i , c ^ { \prime } - 1 ) }$ :

$$
\mathcal { E } _ { 3 d - r p e } ^ { \prime } \geq c ^ { \prime } - 1 + \frac { \left( \varphi _ { i + 1 } - \varphi _ { i } \right) } { \theta } > c ^ { \prime } - 2 \geq 1
$$

where $( \varphi _ { i + 1 } - \varphi _ { i } ) / \theta \geq - 1 / 1 0 0 0 0 > - 1$ . As long as $c ^ { \prime } \geq 3$ , there is $\mathcal { E } _ { 3 d - r p e } ^ { \prime } \ : 2 \ : 1 > \mathcal { E } _ { r o p e } ^ { \prime } = L _ { p } / L$ .>U−nder normal case,≥the chunk size $c$ is not set to a very small number, hence $c ^ { \prime } \geq 3$ is certainly established; moreover, for different interpolati≥on lengths $L$ , we need to configure a varying number of chunks $n$ , such that $c ^ { \prime } = \lceil L / n \rceil \leq L _ { p } ^ { \bar { \prime } }$ . □

To empirically validate the superior performance of this benefit in a training-free setting, it has been observed that methods combining RoPE with interpolation lead to a significant increase in Perplexity as the modeling length increases in language modeling tasks. Conversely, the increase in Perplexity is substantially smaller when employing 3D-RPE with linear interpolation (Refer to Table 4). This phenomenon indicates that this benefit has led to an improvement in the performance of long sequence language modeling.

# Related Work

This section provides an overview of the literature related to position encoding and context extension.

Position Encoding (PE): PE is important for Transformerbased language models. Earlier studies (Shaw, Uszkoreit, and Vaswani 2018; Raffel et al. 2020; Wang et al. 2020; Su et al. 2024) have focused on enhancing the original absolute position encoding to develop better relative position encoding, thereby improving the text modeling capabilities of language models. These works (Shaw, Uszkoreit, and Vaswani 2018; Raffel et al. 2020; Wang et al. 2020) utilized trainable position vector encoding to directly incorporate positional information into context representations. Although effective, these methods typically add positional information to contextual representations, making them unsuitable for linear selfattention architectures. RoFormer (Su et al. 2024) introduced relative position information by rotating context representations, known as RoPE. Transformers utilizing RoPE have become a prevalent backbone in various LLM designs (Touvron et al. 2023; Chowdhery et al. 2022; Wang and Komatsuzaki 2021). Our proposed 3D-RPE differs from the 2-dimensional space of RoPE by modeling the relative position of tokens through rotation on the Bloch Sphere.

Long-context LLMs based on RoPE: To enhance the contextual capabilities of Large Language Models (LLMs) using RoPE, several positional encoding interpolation techniques have been developed. These include Linear Position Interpolation (LPI) (Chen et al. 2023a), Neural Tangent Kernel (NTK) (Peng and Quesnelle 2023), and Yet Another Recurrent Network (YaRN) (Peng et al. 2023) interpolation. Position Sequence Tuning (PoSE) (Zhu et al. 2023) has notably increased sequence lengths to $1 2 8 k$ by amalgamating these positional interpolation strategies. Additionally, LongLora (Chen et al. 2023b) introduced the shift-short attention mechanism, allowing for effective emulation of full attention and extending sequences up to $1 0 0 k$ , leveraging the LLMa-2- 7B model and LoRA’s fine-tuning approach (Hu et al. 2022). 3D-RPE further strengthens the positional relationships between distant tokens by capturing inter-chunk positional information and is compatible with existing fine-tuning techniques like LoRA to bolster long-context representation. The Dual Chunk Attention (DCA) (An et al. 2024) method, which enhances the use of pre-trained integer-based parameters, splits query and key sequences into chunks and uses three specialized matrices to capture the relative positions within and between these chunks. This method enhances the model’s ability to process longer sequences, but it is unable to model the relative positions within distant chunks. In our work, we employ rotating positional encoding to link attention across different chunks.

# Experiments

We evaluate our proposed 3D-RPE on LLaMA2 (Touvron et al. 2023) models (specifically, LLaMA-2-7B and LLaMA-2-7B-chat), which have a $4 k$ pre-training context, and LLaMA-3-8B-Instruct (AI@Meta 2024), which has an $8 k$ pre-training context. Our experiments aim to explore the following aspects: (1) The effect of 3D-RPE on longcontext generation can be assessed using Perplexity. (2) The impact of 3D-RPE on long-context understanding and generation tasks, can be reflected by the accuracy of long sequence natural language tasks, e.g., multiply documents QA. (3) Ablation studies to confirm the advantages of 3D-RPE in position interpolation. Our code, data, and appendix are available on GitHub (https://github.com/maxindian/3D-RPELong-Contex-Modeling)

# Experimental Settings

We elaborate on the experimental setup by introducing two types of tasks (i.e., long-context language understanding and long sequence language modeling) and detailing three aspects of the configuration (i.e., training setting, datasets, and baseline models).

Training Setting: For long-context Natural Language Understanding (NLU) tasks, we have fine-tuned LLaMA-2-7Bchat and LLaMA-3-8B-Instruct. The fine-tuning method follows the fine-tuning strategy of LongChat (Li et al. 2023a). The training step is 3, 000. For the long-sequence Language Modeling (LM) tasks, we have fine-tuned LLaMA-2-7B to support extended context length of $3 2 k$ tokens. The training step is $1 , 0 0 0$ . We set the per-device batch size as 1, and gradient accumulation step as 8, which means that the batch size is 8. We train the model with the next token prediction objective with LoRA (Hu et al. 2022).

We employed the AdamW optimizer (Loshchilov and Hutter 2019) with $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } ~ = ~ 0 . 9 5$ for all fine-tuned models. Chunk si =is set to $3 k$ . T=he learning rate was set to $2 \times 1 0 ^ { - 5 }$ , and a linear learning rate warmup was applied. Training was conducted on a single 4xA800 GPU machine using FlashAttention-2 (Dao 2023).

Datasets: In the context of long-context NLU tasks, we employ the LongAlpaca-12k dataset, which contains 9,000 LongQA and 3,000 short QA entries (Chen et al. 2023c), and the LongAlpace-16k-length dataset (Chen et al. 2023b). To evaluate the performance of 3D-RPE for long-context extension, we use the LongBench (Bai et al. 2023), which includes 13 English tasks, 5 Chinese tasks and 2 code tasks, with most tasks having an average context length of $5 k$ to $1 5 k$ tokens. We focus on the English and code tasks to evaluate our method, 3D-RPE. Additionally, the LEval (An et al. 2023) evaluation set, which also consists of long-context datasets, is used to verify the effectiveness of 3D-RPE. The five datasets annotated from scratch in LEval, namely Coursera, QuALiTY, CodeU, GSM,and TOEFL, are utilized.

For long-sequence LM tasks, we use the RedPajamaData (Computer 2023) for fine-tuning training. The dataset is a large-scale pre-training dataset (the size reaches 1.2 trillion tokens) designed to provide high-quality training data for language models, and contains multiple data sources (i.e., github, arxiv, book, c4 and Wikipedia, etc.). We sample 20, 000 samples from these data sources for training. For evaluation, we utilize the PG19 book corpus dataset (Rae et al. 2020), which includes 100 documents, and the Arxiv Math Proofpile dataset (test split). Additionally, all methods evaluate perplexity by using a sliding window following (Press, Smith, and Lewis 2022).

Baseline Models: For long-context NLU tasks, the finetuned models, including LongAlpace-16k (Chen et al. 2023b), LongChat-32k (Li et al. 2023b) LongLlama (Tworkowski et al. 2023) and ChatGLM (Du et al. 2022) are used as the baseline models. Models of fine-tuning free in language modeling tasks are also used in long-context NLU tasks.

In long-sequence LM tasks, the methods of LongLoRA (Chen et al. 2023b), StreamingLLM (Xiao et al. 2023), Positional Interpolation (PI) (Chen et al. 2023a), and NTK-Aware Scale RoPE (NTK) (Peng and Quesnelle 2023) are selected as the baselines, all based on the LLaMA-2-7Bbase model. Among these baseline models, PI, NTK and StreamingLLM are fine-tuning-free methods. The fine-tuned models include LongLoRA and Activation Beacon (Zhang et al. 2024). In Ablation experiments, both the fine-tuned training model and the untrained model of the PI method are considered as baseline models. Our model’s numerical precision is set to FP16.

Table 1: Comparison between open-source based models on long-context NLU tasks. Our model, 3D-RPE-LlaMA2-7B-Chat is fine-tuning on LLaMA-2-7b-chat, which is extended from $4 k$ to $1 6 k$ context lengths. Baseline models can be categorized into two groups: those that necessitate fine-tuning during training (such as LongAlpaca and LongLLaMA), and those that do not require it (including PI, NTK, StreamingLLM, and ChunkLLaMA- $1 6 k$ ).   

<html><body><table><tr><td>METHODS</td><td>Single-Doc QA</td><td>Multi-Doc QA</td><td>Summarization</td><td>Few-shot</td><td>Code</td></tr><tr><td>LLaMA-2-7B-chat</td><td>24.90</td><td>22.60</td><td>24.70</td><td>60.01</td><td>48.10</td></tr><tr><td>LLaMA-2-7B-chat-PI</td><td>18.98</td><td>17.16</td><td>25.03</td><td>49.43</td><td>52.73</td></tr><tr><td>LLaMA-2-7B-chat-NTK</td><td>23.21</td><td>23.34</td><td>24.40</td><td>59.29</td><td>49.28</td></tr><tr><td>StreamingLLM</td><td>21.47</td><td>22.22</td><td>22.20</td><td>50.05</td><td>48.00</td></tr><tr><td>ChunkLLaMA-16k</td><td>24.04</td><td>22.98</td><td>21.52</td><td>46.31</td><td>49.73</td></tr><tr><td>LongChat-32k</td><td>31.58</td><td>23.50</td><td>26.70</td><td>64.02</td><td>54.10</td></tr><tr><td>LongAlpaca-16k</td><td>28.70</td><td>28.10</td><td>27.80</td><td>63.70</td><td>56.00</td></tr><tr><td>LongLLaMA</td><td>30.12</td><td>16.37</td><td>24.19</td><td>60.31</td><td>66.05</td></tr><tr><td>Vicuna-v1.5-7B-16k</td><td>28.01</td><td>18.63</td><td>26.01</td><td>66.20</td><td>47.30</td></tr><tr><td>ChatGLM3-6B-32k</td><td>40.30</td><td>46.60</td><td>29.50</td><td>68.10</td><td>56.20</td></tr><tr><td>3D-RPE-LLaMA2-7B-Chat</td><td>47.40</td><td>60.10</td><td>28.99</td><td>73.16</td><td>76.50</td></tr></table></body></html>

<html><body><table><tr><td>MODELS</td><td>Coursera</td><td>QuALiTY</td><td>CodeU</td><td>GSM</td><td>TOEFL</td></tr><tr><td>LLaMA-2-7B-Chat</td><td>29.21</td><td>37.62</td><td>1.11</td><td>19.00</td><td>51.67</td></tr><tr><td>LongChat-7B-16K</td><td>29.74</td><td>33.66</td><td>3.33</td><td>10.00</td><td>47.95</td></tr><tr><td>LLaMA2-7B-NTK</td><td>32.71</td><td>33.16</td><td>0.00</td><td>19.00</td><td>52.78</td></tr><tr><td>Vicuna1.5-7B-16k</td><td>38.66</td><td>39.60</td><td>5.55</td><td>19.00</td><td>55.39</td></tr><tr><td>3D-RPE-LLaMA2-7B-Chat(ours)</td><td>39.38</td><td>38.11</td><td>2.22</td><td>21.01</td><td>57.99</td></tr><tr><td>LLaMA3-8B-Instruct*</td><td>51.45</td><td>64.34</td><td>4.44</td><td>76.00</td><td>82.89</td></tr><tr><td>3D-RPE-LLaMA3-8B-Instruct*</td><td>51.89</td><td>61.38</td><td>4.44</td><td>80.00</td><td>82.89</td></tr></table></body></html>

Table 2: Comparison with open-source models, LLaMA-2-7B-chat, LLaMA3-8B-Instruct, on 5 closed-ended-ended tasks with various input length from LEval (An et al. 2023). The evaluation metric “EM,” which represents the exact match score, is adopted. \* indicates the model is train-free.

# Long-Context Natural Language Understanding

In this task, the LongBench (Bai et al. 2023) evaluation set was initially utilized. Five categories of tasks were included: single-document QA (3 tasks), multi-document QA (3 tasks), summarization (3 tasks), few-shot learning (3 tasks), and code completion (2 tasks). The average score for each type is reported in Table 1. The evaluation metrics followed those specified in LongBench (Bai et al. 2023). The results in Table 1 highlight our model’s significant performance advantages over baseline models in four tasks, both for models without training and those with fine-tuning. To compare with the well-performing long sequence model at that time, we included ChatGLM3 in Table 1, even though it did not use the same base LLM. Both our method and other baseline methods use LLaMA2-7B-Chat as the base model. In summarization tasks, our model also achieved performance comparable to ChatGLM3-6B- $3 2 k$ . These experimental outcomes indicate that our model enhances the correlation between tokens with distant relative positions in long contexts through 3D-RPE, improving the experimental performance of the LLaMA-2- 7B-Chat model in long-context understanding tasks.

Subsequently, the LEval Benchmark (An et al. 2023) was employed. Table 2 reveals that our model, 3D-RPE-LLaMA2- 7B-Chat, outperformed LLaMA2-7B-NTK and LongChat$7 \mathrm { B } { - } 1 6 K$ . Although it did not surpass Vicuna1.5-7B- $1 6 K$ in Quality and CodeU tasks, it excelled in the Coursera, GSM, and TOEFL tasks. Additionally, we conducted experiments on LLaMA3-8B-Instruct using a $1 6 k$ context window with 3D-RPE. The 3D-RPE-LLaMA3-8B-Instruct\* showed performance improvements in the Coursera and GSM tasks. While 3D-RPE did not enhance performance in the CodeU, TOEFL, and QuALiTY tasks, there was no significant performance decline either. These experimental results demonstrate the effectiveness of the 3D-RPE method.

# Long-Sequence Language Modeling

In Table 3, we present the perplexity scores for our model, 3D-RPE-LLaMA-2-7B and baseline models on the proofpile and PG19 test datasets. 3D-RPE-LLaMA-2-7B was finetuned from the LLaMA2-7B-Base model using a dataset with a $3 2 k$ context window. To evaluate performance, we set sequence lengths of $8 k$ , $1 6 k$ , and $3 2 k$ . We extended our model’s sequence length from $3 2 k$ to $1 0 0 k$ using the position extending method from PoSE (Zhu et al. 2023). The results indicate that our method outperforms train-free sequence extending methods, namely positional interpolation (PI and NTK) and StreamingLLM. Compared to fine-tuned models, our model shows better performance at $8 k$ and $1 6 k$ sequence lengths. This suggests that the new positional encoding, 3DRPE, improves or maintains modeling performance for larger context windows $( 3 2 k )$ compared to smaller ones $\boldsymbol { \mathrm { \ 8 } k }$ and $1 6 k \rrangle$ ). For the $3 2 k$ and $1 0 0 k$ tasks, although our model did not surpass LongLoRA- $3 2 k$ and LongLoRA- $. 1 0 0 k$ , it did outperform LongChat- $3 2 k$ and Activation Beacon.

Notably, our model can further extend from $3 2 k$ to $1 0 0 k$ without significantly increasing perplexity values, in combination with other train-free extension methods. However, due to its specific attention mechanism, the LongLoRA models cannot be extended beyond their predefined context windows in a train-free manner. For instance, LongLoRA- $3 2 k$ cannot be further extended to $1 0 0 k$ .

Table 3: Perplexity evaluation on different extending methods. We conduct evaluation on the Proof-pile and PG-19 test datasets, varying evaluation context window size from $8 k$ to $1 0 0 k$ .’-’ indicates that this method cannot be further extended and evaluation results can not be obtained. ’OOM’ is an abbreviation for ’Out of Memory’.   

<html><body><table><tr><td rowspan="2">METHODS</td><td colspan="4">PG-19</td><td colspan="4">Proof-Pile</td></tr><tr><td>8k</td><td>16k</td><td>32k</td><td>100k</td><td>8k</td><td>16k</td><td>32k</td><td>100k</td></tr><tr><td>LLaMA2-7B-Base</td><td>131.09</td><td>>10²</td><td>>10²</td><td>OOM</td><td>16.79</td><td>>10²</td><td>>10²</td><td>OOM</td></tr><tr><td>LLama2-7B-PI</td><td>11.32</td><td>19.5</td><td>>10²</td><td>OOM</td><td>3.86</td><td>5.94</td><td>33.7</td><td>OOM</td></tr><tr><td>LLama2-7B-NTK</td><td>10.28</td><td>11.5</td><td>37.8</td><td>OOM</td><td>3.98</td><td>5.94</td><td>33.7</td><td>OOM</td></tr><tr><td>StreamingLLM</td><td>9.23</td><td>9.25</td><td>9.24</td><td>9.32</td><td>3.47</td><td>3.51</td><td>3.50</td><td>3.55</td></tr><tr><td>LongLoRA-32k</td><td>7.33</td><td>7.16</td><td>7.04</td><td>1</td><td>2.78</td><td>2.61</td><td>2.50</td><td>1</td></tr><tr><td>LongLoRA-100k</td><td>7.57</td><td>7.33</td><td>7.16</td><td>7.04</td><td>2.78</td><td>2.60</td><td>2.58</td><td>2.52</td></tr><tr><td>LongChat-32k</td><td>8.92</td><td>8.85</td><td>8.81</td><td>0OM</td><td>2.98</td><td>2.70</td><td>2.65</td><td>OOM</td></tr><tr><td>Activation Beacon</td><td>8.52</td><td>8.54</td><td>8.56</td><td>8.68</td><td>3.45</td><td>3.42</td><td>3.39</td><td>3.35</td></tr><tr><td>3D-RPE-LLaMA2-7B</td><td>7.03</td><td>7.10</td><td>8.09</td><td>8.12</td><td>2.72</td><td>2.93</td><td>2.89</td><td>3.05</td></tr></table></body></html>

Table 4: Results are evaluated in Perplexity on PG19 validation split. ’\*’ denotes train-free. $\ ' _ { + } '$ indicates the fine-tuned version of the data. The context length of $8 k$ is extended directly with 3D-RPE. Achieving $1 6 k$ and $3 2 k$ is accomplished through PI with chunks based on the $8 k$ context length.   

<html><body><table><tr><td>MODELS</td><td>4k</td><td>8k</td><td>16k</td><td>32k</td></tr><tr><td>LLaMA2-7B-PI</td><td>7.94</td><td>9.19</td><td>15.11</td><td>>10²</td></tr><tr><td>LLaMA2-7B-NTK</td><td>7.87</td><td>11.98</td><td>26.12</td><td>58.91</td></tr><tr><td>LLaMA2-7B-Yarn</td><td>7.87</td><td>8.06</td><td>9.82</td><td>11.74</td></tr><tr><td>3D-RPE-LLaMA2-7B*</td><td>7.87</td><td>7.90</td><td>7.71</td><td>9.34</td></tr><tr><td>LLaMA2-7B-PI+</td><td>1</td><td>8.02</td><td>8.05</td><td>>10²</td></tr><tr><td>3D-RPE-LLaMA2-7B</td><td>/</td><td>7.85</td><td>8.15</td><td>8.82</td></tr></table></body></html>

# Ablation Study

In this section, we conduct ablation studies in this section to explore how 3D-RPE affects the linear interpolation method. We compare position interpolation methods (PI, NTK, and Yarn) with the method that combines 3D-RPE with position interpolation on the LLaMA-2-7B-Base model in a train-free manner. The experimental results can be found in Table 2. The 3D-RPE-LLaMA2- $^ { 7 \mathrm { B ^ { * } } }$ model with linearly positional interpolation from $8 k$ to $1 6 k$ and $3 2 k$ , the 3D-RPE approach yields improved results by mitigating the decrease in positional resolution caused by interpolation methods. These results are consistent with the findings of Theorem 1. Additionally, our method incorporates the PI technique. To further demonstrate the effectiveness of 3D-RPE, LLaMA-2-7B-PI is fine-tuned on the same dataset as 3D-RPE-LLaMA2-7B. Our method achieved lower PPL for $8 k$ and $3 2 k$ .

To analyze the impact of different chunk size settings on model performance, we applied 3D-RPE to the base model LLaMA2-Base without fine-tuning. Then, we extended the modeling context length from 4k to 8k, and set the chunk sizes to 1k, 2k, 2.5k, 3k, 3.5k, and 4k respectively. The PPL scores are 9.67, 9.08, 8.98, 7.83, 7.85, and 8.81 respectively. These results suggest that the chunk size setting should be chosen to be close to the context length used during model pre-training, as this allows for better utilization of the positional encoding information learned during pre-training within the chunk. However, selecting a chunk size that is too close to the pre-training length is also not ideal, because LLMs typically use a fixed context length during pre-training, and the lengths of the pre-training texts are not aligned, which leads to suboptimal learning of positional information near the maximum pre-training length.

# Conclusion and Future Work

In this paper, we present a novel rotary position encoding method called 3D Rotary Position Encoding (3D-RPE). Compared to RoPE, we have theoretically proved that 3D-RPE possesses two key advantages: controllable long-term decay and improved interpolation resolution. Experimentally, 3D-RPE has excelled in long-context NLU tasks. 3D-RPE doesn’t require many more samples for continual training because it effectively uses the position encoding of the pretrained LLM within the chunk. The limitation of our work is the lack of further research on the design of relative positional spacings on chunks. For example, the transition between the last token of one chunk and the first token of the next chunk is not smooth. In our experiments, in order to better adapt to the positional encoding relationships pre-trained by LLM, the relative positional spacings on Chunks are compressed as the index of the Chunk increases.

In the future, 3D-RPE holds promise as a foundational positional encoding strategy for LLMs, especially in the aspect of modeling long contexts. Moreover, given that 3D-RPE encapsulates positional encoding within a three-dimensional framework, it has the potential to integrate with visual data, thereby facilitating an in-depth exploration of its efficacy in synchronizing graphical and textual semantic information.