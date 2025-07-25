# Knowledge in Superposition: Unveiling the Failures of Lifelong Knowledge Editing for Large Language Models

Chenhui ${ \bf { H } } { \bf { u } } ^ { 1 , 2 }$ , Pengfei $\mathbf { C a o } ^ { 1 , 2 }$ , Yubo Chen1,2, Kang Liu1,2\*, Jun Zhao1,2

1 The Key Laboratory of Cognition and Decision Intelligence for Complex Systems, Institute of Automation, Chinese Academy of Sciences, Beijing, China 2 School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, China huchenhui $2 0 2 4 @$ ia.ac.cn, {pengfei.cao, yubo.chen, kliu, jzhao}@nlpr.ia.ac.cn

# Abstract

Knowledge editing aims to update outdated or incorrect knowledge in large language models (LLMs). However, current knowledge editing methods have limited scalability for lifelong editing. This study explores the fundamental reason why knowledge editing fails in lifelong editing. We begin with the closed-form solution derived from linear associative memory, which underpins state-of-the-art knowledge editing methods. We extend the solution from single editing to lifelong editing, and through rigorous mathematical derivation, identify an interference term in the final solution, suggesting that editing knowledge may impact irrelevant knowledge. Further analysis of the interference term reveals a close relationship with superposition between knowledge representations. When knowledge superposition does not exist in language models, the interference term vanishes, allowing for lossless knowledge editing. Experiments across numerous language models reveal that knowledge superposition is universal, exhibiting high kurtosis, zero mean, and heavy-tailed distributions with clear scaling laws. Ultimately, by combining theory and experiments, we demonstrate that knowledge superposition is the fundamental reason for the failure of lifelong editing. Moreover, this is the first study to investigate knowledge editing from the perspective of superposition and provides a comprehensive observation of superposition across numerous real-world language models.

# Introduction

In large language models (LLMs), outdated or incorrect knowledge may persist (Radford et al. 2019; Wang and Komatsuzaki 2022; Biderman et al. 2023; Touvron et al. 2023). However, retraining these models to update knowledge incurs prohibitively high costs. To address this problem, knowledge editing (De Cao et al. 2021; Mitchell et al. 2021) is introduced to edit specific knowledge by directly updating the internal parameters of language models.

Despite achieving significant progress in single editing, where knowledge is edited only once, knowledge editing should be continuous in fact. However, current methods struggle with scalability for lifelong editing (Huang et al. 2023), where continuous editing and performance monitoring are required throughout the language models’ lifecy

(a) Structure of knowledge editing. (e) Superposition determines interference.   
Knowledge Editing Methods . (d) Superposition Term $\displaystyle \left. \sum _ { K } p ( k _ { i } , k _ { j } ) \right.$   
ROME MEMIT PMET WilKE .. (c) Interference Term Different Implementations ... $\Delta = \underset { \bf A } { W _ { n } } K - V = \underset { \mathrm { i n }  \infty } { | \operatorname* { l i m } } \sum \Lambda _ { i } ( C ^ { - 1 } k _ { e i } ) ^ { T } K |$ ↑ Closed-Form Solution Expanding Solutio W=W+A(C-1ke) $\Rightarrow | \overline { { W _ { n } } } | = W _ { 0 } + \sum \Lambda _ { i } ( C ^ { - 1 } k _ { e i } ) ^ { T } |$ (b) Expanding to Lifelong Editing.

cle. For example, using the representative editing methods ROME (Meng et al. 2022a) or MEMIT (Meng et al. 2022b) for lifelong editing results in severe performance degradation after dozens or hundreds of editing steps, rendering the edited models unusable (Hu et al. 2024). However, the underlying reason has been rarely explored.

In this paper, we explore the fundamental reason why knowledge editing fails in lifelong editing. We start from the closed-form solution for knowledge editing derived by (Meng et al. 2022a) through linear associative memory, which is the foundation of the state-of-the-art knowledge editing methods (Figure 1a), including ROME (Meng et al. 2022a), MEMIT (Meng et al. 2022b), PMET (Li et al. 2023), WilKE (Hu et al. 2024) and so on. Specifically, we extend the closed-form solution from single editing to lifelong editing (Figure 1b). Our rigorous mathematical derivation reveals that the extended solution introduces an interference term (Figure 1c), which, when accumulated sufficiently, leads language models to forgetting the knowledge, including original knowledge and previously edited knowledge.

Further analysis of the interference term reveals a close relationship with superposition: if superposition does not exist in language models, the superposition term (Figure 1d)

![](images/56814cff9961956df74593869e2e4da8136321be47067f3e0d844396c80b24b9.jpg)  
Figure 2: A neural network with only three neurons, corresponding to three dimensions, (a) can directly represent three features orthogonally, but (b) to represent six features (or more features), it will using superposition to noisily encode them nearly orthogonally.

is zero, causing the interference term to vanish (Figure 1e), allowing for lossless lifelong editing. Superposition (Elhage et al. 2022b) refers to the situation where neural networks attempt to represent more features than the available dimensions. For example, if a simple neural network with three neurons (corresponding to a three-dimensional space) attempts to represent three features, each neuron can be assigned to one feature, forming an orthogonal basis in three-dimensional space (Figure 2a). But if the same network tries to represent six features (or more features), it will use superposition strategy, noisily encoding these features, where the directions corresponding to each feature will not be fully orthogonal (Figure 2b). Specifically, we find that the magnitude of interference term depends on the orthogonality of knowledge representations (Figure 1e). If these knowledge representations are perfectly orthogonal (non-superposition), the interference term is zero, enabling perfectly lossless knowledge editing, where only the target knowledge is updated without affecting unrelated knowledge. However, if they are not orthogonal (superposition), the interference will accumulate linearly and eventually tend toward infinity, leading to language models’ failure (Appendix H). In other words, whether superposition exists in language models is equivalent to whether lossless knowledge editing can be achieved, which will determine whether lifelong editing can be achieved.

However, to what extent does superposition hold true in real-world language models? Our experiments reveal that knowledge superposition1 is universal across all language model families, characterized by high kurtosis, zero mean, heavy-tailed distributions, with a clear scaling law.

Specifically, we conduct experiments on language model families including GPT-2 (Radford et al. 2019), Llama-2 (Touvron et al. 2023), Pythia (Biderman et al. 2023), GPTJ (Wang and Komatsuzaki 2022), Llama-3 (Meta 2024a), and Llama-3.1 (Meta 2024b), to capture the real situation of knowledge superposition. Firstly, we observe that knowledge superposition exists across all layers of all language models we examined (Figure 3), indicating that superposition is widespread in real-world language models. Secondly, the knowledge superposition exhibits a high kurtosis, zeromean heavy-tailed distribution (Figure 4). This means that the distribution consistently has a very high peak near zero, which corresponds to the positions where all knowledge representations are perfectly orthogonal, suggesting models attempt to store different knowledge orthogonally but resort to stores them nearly orthogonally due to capacity constraints, i.e., through superposition. Thirdly, we observe a clear scaling law (Figure 5): as the size of language models increases, they attempt to represent knowledge in a more orthogonal manner, thereby reducing noise between knowledge representations, which explains why larger language models can more effectively handle knowledge-intensive question-answering scenarios. Finally, to provide further direct evidence of superposition, we calculate the angular distribution between these knowledge representations in highdimensional space, which also shows a high kurtosis, heavytailed distribution, but centered around 90 degrees (Figure 6).

Overall, our contributions are as follows:

• We extend the closed-form solution for knowledge editing from single editing to lifelong editing. Our rigorous derivation reveals an interference term in the final solution, indicating that editing can affect both original and previously edited knowledge.   
• We conduct further analysis of the interference term, which actually reflects knowledge superposition. The magnitude of this interference is determined by the superposition of knowledge representations, with nonsuperposition enabling lossless knowledge editing.   
• We investigate the phenomenon of knowledge superposition across multiple language model families and find it universal, exhibiting high kurtosis, zero mean, heavy-tailed distributions, with a clear scaling law. To our knowledge, this is the first observation of widespread knowledge superposition.

# Related Work

# Knowledge Editing

In general, knowledge editing aims to modify the knowledge of a language model so that its outputs reflect the revised state when confronted with relevant inputs (De Cao et al. 2021). Yao et al. (2023) survey knowledge editing methods and classify them into two main categories: preserving model parameters and modifying model parameters. Preserving model parameters includes memory-based methods (Mitchell et al. 2022; Zhong et al. 2023; Hartvigsen et al. 2024), which stores all edit examples explicitly in memory, and additional parameter methods (Dong et al. 2022; Huang et al. 2023), which use extra trainable parameters within language models. However, these methods usually have poor generalization, and the required additional content also increases significantly with the number of editing (Cao et al. 2024; Li et al. 2024). In contrast, modifying model parameters methods directly update the model parameters for editing, thereby avoiding these issues. Such methods are also divided into two categories: meta-learning and locate-thenedit methods. Meta-learning methods (De Cao et al. 2021; Mitchell et al. 2021; Tan, Zhang, and $\mathrm { F u } 2 0 2 3 \$ ), which use a hyper network to learn and apply gradients for fine-tuning, typically require a lengthy hyper network training process before each edit, making them less effective for lifelong editing. Locate-then-edit methods first identify and then update the parameters associated with specific knowledge. For instance, KnowledgeNeuron (Dai et al. 2021) uses knowledge attribution to locate neurons and updates parameters to achieve knowledge editing; ROME (Meng et al. 2022a) employs causal mediation analysis to identify the center of causal effects, namely MLP, and performs rank-one editing on MLP; MEMIT (Meng et al. 2022b) extends ROME by allocating residuals to multiple layers and enabling batch editing; PMET (Li et al. 2023) further refines residual allocation based on MEMIT; and WilKE (Hu et al. 2024) dynamically selects editing layers based on ROME to avoid toxicity flash.

This paper focuses on methods that modify model parameters, among which the state-of-the-art methods are ROME and its derivatives, MEMIT, PMET, and WilKE. Therefore, we concentrate on the same closed-form solution (Figure 1a) of them and extend it from single to lifelong editing. From a theoretical perspective, we identify the fundamental reason for the failure of knowledge editing methods in lifelong editing, namely superposition.

# Superposition

Superposition refers to a strategy used by neural networks to express features that far exceed their dimensions, typically assigning approximately orthogonal directions in the representation space to these features (Figure 2b). However, this superposition phenomenon has only been observed in toy models so far (Elhage et al. 2022b). Some studies hypothesize that this superposition phenomenon also exists in language models and propose sparse autoencoder (SAE) (Cunningham et al. 2023) and its various variants (Rajamanoharan et al. 2024; Gao et al. 2024) to attempt to disentangle superposition from the activation space of language models (Bricken et al. 2023; Templeton et al. 2024). Additionally, Gurnee et al. (2023) claim to observe superposition in the wild, their study was limited to a few neurons in a specific layer of a single model, focusing on neuron superposition. This differs from the superposition concept in Elhage et al. (2022b) and our study.

In this paper, we identify a widespread phenomenon of knowledge superposition across multiple language model families and explore its characteristics. Furthermore, this is the first study of knowledge superposition in knowledge editing, explaining the reason for failures in lifelong editing from the perspective of superposition. Notably, unlike previous studies, we find that superposition occurs in the whitening space rather than the activation space (Figure 6).

# Preliminary

Geva et al. (2020) discover that MLPs are the key components for memory storage in Transformers, and Meng et al.

(2022a) identify through causal tracing that MLPs are crucial for storing factual knowledge. The MLPs in the FeedForward Network (FFN) of a Transformer consist of two layers, represented as follows (bias terms are omitted)

$$
F F N ( { \pmb x } ) = W _ { p r o j } \ \sigma ( W _ { f c } { \pmb x } ) ,
$$

where $W _ { f c } \in \mathbb { R } ^ { d _ { m } \times d }$ and $W _ { p r o j } \in \mathbb { R } ^ { d \times d _ { m } }$ are the parameter matrices of the FFN, $d _ { m }$ is the dimensionality of the hidden layer in the FFN, $\sigma$ is the activation function, and $\pmb { x } \in \mathbb { R } ^ { d }$ is the input of the FFN.

Meng et al. (2022a) model MLPs as linear associative memory (Kohonen 1972; Anderson 1972), viewing $W _ { p r o j }$ as a linear associative store. This perspective observes that any linear operation $W$ can be represented as a key-value store with a set of vector keys $K \bar { = } \left[ k _ { 1 } | k _ { 2 } | \cdots \right]$ and corresponding vector values $V = \left[ v _ { 1 } | v _ { 2 } | \cdots \right]$ . To optimally insert a new key-value pair $( k _ { e } , v _ { e } )$ into the store for knowledge updates, one can solve a constrained least-squares problem, leading to a closed-form solution

$$
\begin{array} { r l } & { m i n i m i z e \| \hat { W } K - V \| \mathrm { ~ s u c h ~ t h a t ~ } \hat { W } k _ { e } = v _ { e } } \\ & { \qquad \mathrm { b y ~ s e t t i n g ~ } \hat { W } = W + \Lambda ( C ^ { - 1 } k _ { e } ) ^ { T } , } \end{array}
$$

where $C = K K ^ { T }$ is a constant that actually represents the covariance matrix, and $\Lambda = ( v _ { e } - W k _ { e } ) / ( \dot { C } ^ { - 1 } \dot { k } _ { e } ) ^ { T } k _ { e }$ . The detailed proof can be found in the original paper (Meng et al. 2022a), and a more thorough derivation is provided in Appendix A.

# Expanding to Lifelong Editing

For convenience, let the initial $W _ { p r o j }$ be $W _ { 0 }$ . Following the first update using $( k _ { e _ { 1 } } , v _ { e _ { 1 } } )$ , we obtain $W _ { 1 }$ . Subsequently, after the $n$ -th edit with $( k _ { e _ { n } } , v _ { e _ { n } } )$ , we derive $W _ { n }$ . Extending the closed-form solution from Equation 2 to the lifelong editing scenario, we obtain

$$
W _ { n } = \binom { W _ { 0 } } { W _ { n - 1 } + \Lambda _ { n } ( C ^ { - 1 } k _ { e _ { n } } ) ^ { T } , ~ n \geq 1 }
$$

where $C = K K ^ { T }$ , and

$$
\Lambda _ { n } = \frac { v _ { e _ { n } } - W _ { n - 1 } k _ { e _ { n } } } { ( C ^ { - 1 } k _ { e _ { n } } ) ^ { T } k _ { e _ { n } } } , n \ge 1 .
$$

Expanding $W _ { n }$ , we obtain

$$
W _ { n } = W _ { 0 } + \sum _ { i = 1 } ^ { n } \Lambda _ { i } ( C ^ { - 1 } k _ { e _ { i } } ) ^ { T } .
$$

For convenience, we let

$$
K ^ { * } = \left[ \begin{array} { l l l l } { \ } & { K _ { o } ^ { * } } & { | } & { K _ { e } ^ { * } } \end{array} \right] ,
$$

where $\begin{array} { r c l } { K _ { o } ^ { * } } & { = } & { \left[ k _ { 1 } | k _ { 2 } | \cdot \cdot \cdot \right] } \end{array}$ represents the keys corresponding to the model’s original knowledge, and $K _ { e } ^ { * } ~ =$ $[ k _ { e _ { 1 } } | k _ { e _ { 2 } } | \cdots | k _ { e _ { n } } ]$ represents the keys for the edited knowledge.

We then calculate $V ^ { * }$ after $W _ { 0 }$ has been updated to $W _ { n }$ , which is given by

$$
\begin{array} { r l } & { V ^ { * } = W _ { n } K ^ { * } } \\ & { \quad = ( W _ { 0 } + \displaystyle \sum _ { i = 1 } ^ { n } \Lambda _ { i } ( C ^ { - 1 } k _ { e _ { i } } ) ^ { T } ) K ^ { * } } \\ & { \quad = W _ { 0 } K ^ { * } + \left( \displaystyle \sum _ { i = 1 } ^ { n } \Lambda _ { i } ( C ^ { - 1 } k _ { e _ { i } } ) ^ { T } \right) K ^ { * } } \\ & { \quad = W _ { 0 } K ^ { * } + \displaystyle \sum _ { i = 1 } ^ { n } \Lambda _ { i } ( C ^ { - 1 } k _ { e _ { i } } ) ^ { T } K ^ { * } . } \end{array}
$$

For convenience, we can represent $V ^ { * }$ using a block matrix notation

$$
V ^ { * } = \left[ \begin{array} { l l l l l } { \ } & { V _ { o } ^ { * } } & { \ | } & { \ V _ { e } ^ { * } } & { \ } \end{array} \right] ,
$$

where $V _ { o } ^ { * }$ and $V _ { e } ^ { * }$ represent the updated value matrices corresponding to the original knowledge and the edited knowledge, respectively. We then analyze these two components separately.

# Interference Term of Original Knowledge $\Delta _ { o }$

Ideally, the value vectors corresponding to the original knowledge after editing, namely $V _ { o } ^ { * }$ , are same as $V _ { o } \ =$ $\left[ v _ { 1 } | v _ { 2 } | \cdots \right]$ , because we want to ensure that editing does not affect unrelated original knowledge. Thus, we define $\Delta _ { o }$ as

$$
\begin{array} { l } { \displaystyle \Delta _ { o } = V _ { o } ^ { * } - V _ { o } } \\ { \displaystyle = \left[ \sum _ { i = 1 } ^ { n } \Lambda _ { i } ( C ^ { - 1 } k _ { e _ { i } } ) ^ { T } k _ { 1 } | \sum _ { i = 1 } ^ { n } \Lambda _ { i } ( C ^ { - 1 } k _ { e _ { i } } ) ^ { T } k _ { 2 } | \cdots \right] . } \end{array}
$$

For convenience, we study $\Delta _ { o } [ : , j ]$ , which represents the $j$ -th column vector of $\Delta _ { o }$ . This can be viewed as the interference term introduced to the $j$ -th piece of original knowledge after $n$ edits,

$$
\begin{array} { l } { \displaystyle \Delta _ { o } [ : , j ] = \sum _ { i = 1 } ^ { n } \Lambda _ { i } ( C ^ { - 1 } k _ { e _ { i } } ) ^ { T } k _ { j } } \\ { = \sum _ { i = 1 } ^ { n } \frac { v _ { e _ { i } } - W _ { i - 1 } k _ { e _ { i } } } { ( C ^ { - 1 } k _ { e _ { i } } ) ^ { T } k _ { e _ { i } } } ( C ^ { - 1 } k _ { e _ { i } } ) ^ { T } k _ { j } } \\ { = \sum _ { i = 1 } ^ { n } \frac { ( C ^ { - 1 } k _ { e _ { i } } ) ^ { T } k _ { j } } { ( C ^ { - 1 } k _ { e _ { i } } ) ^ { T } k _ { e _ { i } } } ( v _ { e _ { i } } - W _ { i - 1 } k _ { e _ { i } } ) . } \end{array}
$$

Next, let $\delta ( v _ { e _ { i } } ) = v _ { e _ { i } } - W _ { i - 1 } k _ { e _ { i } } \in \mathbb { R } ^ { d }$ represent the difference vector between the optimized value and the current value when editing the $i$ -th piece of knowledge, and define the coefficient $\begin{array} { r } { p ( k _ { e _ { i } } , k _ { j } ) ~ = ~ \overset { \textstyle ( C ^ { - 1 } k _ { e _ { i } } ) ^ { T } k _ { j } } { ( C ^ { - 1 } k _ { e _ { i } } ) ^ { T } k _ { e _ { i } } } ~ \in ~ \mathbb { R } } \end{array}$ (C−1kei)T kj R. Then, we obtain

$$
\Delta _ { o } [ : , j ] = \sum _ { i = 1 } ^ { n } p ( k _ { e _ { i } } , k _ { j } ) \delta ( v _ { e _ { i } } ) .
$$

Ideally, we want $\Delta _ { o } [ : , j ] = \mathbf { 0 }$ to achieve lossless knowledge editing, ensuring that the original knowledge of the model remains unaffected. However, since $\delta ( v _ { e _ { i } } ) = v _ { e _ { i } } -$ $W _ { i - 1 } k _ { e _ { i } } \neq 0$ (as this is the premise; if $\delta ( v _ { e _ { i } } ) ~ = ~ 0$ , then there is no need for editing), we can only hope that $p ( k _ { e _ { i } } , k _ { j } ) = 0$ . This would eliminate the interference on the $j$ -th original knowledge.

# Interference Term of Edited Knowledge $\Delta _ { e }$

Ideally, the value vectors corresponding to the edited knowledge after editing, $V _ { e } ^ { * }$ , are same as $V _ { e } = [ v _ { e _ { 1 } } | v _ { e _ { 2 } } | \cdot \cdot \cdot | v _ { e _ { n } } ] ,$ because we want to avoid affecting the already edited knowledge during further editing. Thus, we define $\Delta _ { e }$ as

$$
\begin{array} { r l r } {  { \Delta _ { e } = V _ { e } ^ { * } - V _ { e } } } \\ & { = [ \sum _ { i = 2 } ^ { n } \Lambda _ { i } ( C ^ { - 1 } k _ { e _ { i } } ) ^ { T } k _ { e _ { 1 } } | \cdots | \Lambda _ { n } ( C ^ { - 1 } k _ { e _ { n } } ) ^ { T } k _ { e _ { n - 1 } } | \mathbf { 0 } ] . } \end{array}
$$

Unlike the case with original knowledge, as editing progresses, earlier edits experience more interference, while later edits are less affected. This is consistent with Jang et al. (2021). For convenience, we study $\Delta _ { e } [ : , j ]$ , which represents the $j$ -th column vector of $\Delta _ { e }$ . This can be seen as the impact introduced to the $j$ -th edited knowledge after $n$ edits,

$$
\Delta _ { e } [ : , j ] = \left\{ \begin{array} { l l } { \sum _ { i = j + 1 } ^ { n } \Lambda _ { i } ( C ^ { - 1 } k _ { e _ { i } } ) ^ { T } k _ { e _ { j } } , } & { j < n } \\ { { \bf 0 } , } & { j = n } \end{array} \right.
$$

In a similar manner, we can express it as

$$
\Delta _ { e } [ : , j ] = \left\{ \begin{array} { l r } { \sum _ { i = j + 1 } ^ { n } \frac { ( C ^ { - 1 } k _ { e _ { i } } ) ^ { T } k _ { e _ { j } } } { ( C ^ { - 1 } k _ { e _ { i } } ) ^ { T } k _ { e _ { i } } } ( v _ { e _ { i } } - W _ { i - 1 } k _ { e _ { i } } ) , j < n } \\ { 0 , j = n } \end{array} \right.
$$

Next, let $\delta ( v _ { e _ { i } } ) = v _ { e _ { i } } - W _ { i - 1 } k _ { e _ { i } } \in \mathbb { R } ^ { d }$ represent the difference vector between the optimized value and the current value when editing the $i$ -th piece of knowledge. Define the coefficient $\begin{array} { r } { p ( k _ { e _ { i } } , k _ { e _ { j } } ) = \frac { ( C ^ { - 1 } k _ { e _ { i } } ) ^ { T } k _ { e _ { j } } } { ( C ^ { - 1 } k _ { e _ { i } } ) ^ { T } k _ { e _ { i } } } \in \mathbb { R } } \end{array}$ . Then, we obtain

$$
\Delta _ { e } [ : , j ] = \left\{ \begin{array} { l l } { \sum _ { i = j + 1 } ^ { n } p ( k _ { e _ { i } } , k _ { e _ { j } } ) \delta ( v _ { e _ { i } } ) , } & { j < n } \\ { 0 , } & { j = n } \end{array} \right.
$$

Ideally, we want $\Delta _ { e } [ : , j ] = \mathbf { 0 }$ to achieve lossless knowledge editing, ensuring that previously edited knowledge is not affected. However, since $\delta ( v _ { e _ { i } } ) \stackrel { . } { = } v _ { e _ { i } } - W _ { i - 1 } k _ { e _ { i } } \stackrel { . } { \neq } { \bf 0 }$ , we can only hope that $p ( k _ { e _ { i } } , k _ { e _ { j } } ) = 0$ . This would eliminate the interference on the $j$ -th edited knowledge.

# How to Understand $p ( \cdot , \cdot )$

In summary, we find that for both the original knowledge and the edited knowledge, there is a similar coefficient $p ( \cdot , \cdot )$ in interference term, which directly determines whether lossless knowledge editing can be achieved.

To study this coefficient more generally, we do not limit to specific keys (which correspond to specific knowledge activations) but consider any keys (or any knowledge activations) $k _ { i }$ and $k _ { j }$ . We further investigate

$$
p ( k _ { i } , k _ { j } ) = \frac { ( C ^ { - 1 } k _ { i } ) ^ { T } k _ { j } } { ( C ^ { - 1 } k _ { i } ) ^ { T } k _ { i } } \in \mathbb { R } .
$$

Since $C = K K ^ { T }$ is a symmetric matrix, $C ^ { - 1 }$ is also symmetric. Therefore, we can rewrite it as

$$
p ( k _ { i } , k _ { j } ) = \frac { k _ { i } ^ { T } C ^ { - 1 } k _ { j } } { k _ { i } ^ { T } C ^ { - 1 } k _ { i } } = \frac { ( C ^ { - \frac { 1 } { 2 } } k _ { i } ) ^ { T } ( C ^ { - \frac { 1 } { 2 } } k _ { j } ) } { ( C ^ { - \frac { 1 } { 2 } } k _ { i } ) ^ { T } ( C ^ { - \frac { 1 } { 2 } } k _ { i } ) } .
$$

Here, $\begin{array} { r } { C = K K ^ { T } } \end{array}$ is the covariance matrix, and both $k _ { i }$ and $k _ { j }$ are column vectors of $K$ . Therefore, $C ^ { - \frac { 1 } { 2 } } k _ { i }$ and $C ^ { - \frac { 1 } { 2 } } k _ { j }$ are the representations of $k _ { i }$ and $k _ { j }$ in the whitening space (Koivunen and Kostinski 1999; Kawahara et al. 2007) after the whitening transformation $C ^ { - \frac { 1 } { 2 } }$ (proof in Appendix B). The coefficient $p ( \cdot , \cdot )$ can be understood as the dot product of $k _ { i }$ and $k _ { j }$ in the whitening space, normalized. When $p ( \cdot , \cdot ) = 0$ , it indicates that the knowledge activations $k _ { i }$ and $k _ { j }$ are orthogonal in the whitening space.

Definition 1 (Matrix Whitening) For a matrix $X$ , its covariance matrix $C o \nu ( X ) \ = \ \bar { X ( X ^ { T } }$ is not necessarily the identity matrix. Matrix whitening involves finding a transformation matrix $P$ such that the covariance matrix of $Y =$ $P X$ , denoted as $C o \nu ( Y )$ , becomes the identity matrix.

Returning to the problem, if we aim to achieve perfectly lossless knowledge editing, this is equivalent to requiring both $\Delta _ { o }$ and $\Delta _ { e }$ to be zero matrices. This in turn means we expect $p ( \cdot , \cdot )$ to be zero, which is equivalent to expecting that knowledge activations are orthogonal in the whitening space. Consequently, this implies that we expect the language model to store different pieces of knowledge in orthogonal directions in the whitening space, and thus that knowledge superposition does not exist in the whitening space. Conversely, if such superposition exists, then the representations are not orthogonal and we cannot achieve perfectly lossless knowledge editing.

# Knowledge in Superposition

In this section, we focus on the phenomenon of knowledge superposition in real-world language models, combining previous theoretical derivations to demonstrate that knowledge superposition is the fundamental reason for the failure in lifelong editing.

Specifically, we calculate the degree of superposition between pieces of knowledge by computing the matrix $P$ for $m$ pieces of knowledge, where

$$
P [ i , j ] = p ( k _ { i } , k _ { j } ) , 1 \leq i , j \leq m .
$$

As $p ( \cdot , \cdot )$ approaches 0, knowledge representations become more orthogonal in the whitening space, and the degree of superposition becomes weaker. Conversely, as $p ( \cdot , \cdot )$ approaches 1, knowledge representations become more similar in the whitening space, and the degree of superposition becomes stronger.

In practice, we choose $m = 1 2 8$ to compute the superposition between $1 2 8 \times 1 2 8$ pairs of knowledge and resulting in a $1 2 8 \times 1 2 8$ matrix $P$ . Empirical evidence shows that $m = 1 2 8$ is sufficient because, at this point, the kurtosis of the superposition distribution has converged (details in Appendix C). This indicates that the data size is adequate for describing the distribution of superposition.

The specific experimental setup is described as follows:

![](images/eac1a187998f77b2e1e875a7713589e69f093c4f7f2d4fd7536437fc5f549f79.jpg)  
Figure 3: Superposition at layer 0 across different language models visualized using $\mathrm { \bf P }$ matrices, ordered by model size. Each point in these $1 2 8 \mathrm { x } 1 2 8 \ P$ matrices corresponds to the $p ( \cdot , \cdot )$ value between two pieces of knowledge.

Models In this study, we employ a variety of models including GPT2 family (Radford et al. 2019) —GPT2-Small, GPT2-Medium, and GPT2-Large, Pythia family (Biderman et al. 2023) —Pythia-1B, Pythia-2.8B, and Pythia6.9B, Llama2 family (Touvron et al. 2023) -Llama2-7B and Llama2-13B. Additionally, we use the classic GPT-J (Wang and Komatsuzaki 2022), the latest Llama3-8B (Meta 2024a) and Llama3.1-8B (Meta 2024b).

Datasets For the extraction of knowledge representations, we utilized the CounterFact dataset (Meng et al. 2022a). It is important to note that while CounterFact is commonly used for counterfactual knowledge editing, our experimental setup does not involve this aspect. Instead, we focus on subject-related knowledge rather than counterfactual objects (details in Appendix G).

# Universal in Language Models

Ideally, the matrix $P$ we obtain should be such that all positions except the diagonal are zero, indicating that no superposition exists and allowing for lossless knowledge editing. However, in all layers of all the language models we studied, the $P$ matrices we obtained consistently show noisy nonzero entries at positions other than the diagonal, indicating the presence of superposition at these points. As shown in Figure 3, we present the heatmaps of $P$ matrices for GPT2- Small, GPT2-Medium, GPT2-Large, and the GPT-J at layer 0, ordered by model size. Additional heatmaps of $P$ matrices for all layers of all models are provided in Appendix D.

It is evident that as the model size increases, the $P$ matrices become progressively ”cleaner.” This indicates that as language models gain more storage capacity to store knowledge, they tend to store this knowledge with weaker superposition, thereby reducing the interference between pieces of knowledge caused by superposition.

Additionally, we observe that even as model size increases, fixed interference points remain at positions off the diagonal, even in models with different architectures and trained on different corpora. We hypothesize that the knowledge pairs corresponding to these points may actually be closely related, despite having different expressions, leading to this phenomenon. This point has been validated through detailed case studies. We find that the knowledge pairs corresponding to these points are indeed closely related. For instance, in the first layer (layer 0) of these four language models, the $p ( \cdot , \cdot )$ values for ”Vladimir Mayakovsky” and ”Vladimir Bukovsky” (both of whom are native Russian speakers) are above 0.95, and even 1.00 in GPT-J, indicating that the operations performed by the MLP in the first layer for these two pieces of knowledge are similar or even identical. This also implies that if we attempt to edit the knowledge of subject ”Vladimir Mayakovsky” in this layer, it will have a consistent impact on ”Vladimir Bukovsky,” suggesting that they are bound together. For example, if we attempt to edit ”Vladimir Mayakovsky” to have French as his native language, the edited model will also output French as ”Vladimir Bukovsky’s” native language. Similar examples include ”Windows $8 . 1 ^ { \cdot \cdot }$ and ”Mac OS X 10.1,” which generally have very high $p ( \cdot , \cdot )$ values (1.00 in the first layer of GPT-J), even though they are produced by entirely different manufacturers, which is fascinating! The case study is detailed in Appendix F.

Furthermore, choosing different editing layers may lead to certain changes, as described by Hu et al. (2024). However, other layers also contain other knowledge in superposition, as shown in Figure 5.

# Heavy-Tailed Distribution in Language Models

To gain a more intuitive understanding of the distribution characteristics of knowledge superposition, we remove the diagonal elements from the $P$ matrices across all layers of all the language models we studied and then plot the kernel density estimation (KDE) of the remaining elements. In Figure 4, we present the kernel density estimation of the superposition distribution at layer 0 for GPT2-Small, GPT2- Medium, GPT2-Large, and the GPT-J, ordered by model size. Additional kernel density estimations for all layers of all models are provided in Appendix E.

It can be observed that the superposition distribution exhibits characteristics of a heavy-tailed distribution with high kurtosis and zero mean. As the model size increases, the kurtosis of distribution becomes larger and the distribution becomes more concentrated around 0. This indicates that smaller models, constrained by capacity, exhibit more superposition, attempting to store knowledge representations in a relatively orthogonal manner. In contrast, larger models have greater capacity to store knowledge, allowing them to store it in a more orthogonal manner compared to smaller models, resulting in a kernel density estimation more concentrated around 0.

![](images/8b94905af89e1c53cab029cb8984301e637ad1f09511ed5add61b2950913a168.jpg)  
Figure 4: KDE of $p ( \cdot , \cdot )$ values in P matrics at layer 0 across different language models, ordered by model size. In $P$ matrices, $p ( \cdot , \cdot )$ values are concentrated around 0, with high kurtosis, which increases as model size grows.

# Scaling Law for Superposition

We have already seen that the degree of knowledge superposition varies with model size. In this section, we formally study the scaling law of superposition by experimenting with different-sized models from the same language model family. These models have consistent architecture and training corpora, differing only in size. As shown in Figure 5, we examine the kurtosis across all layers of a total of 8 language models from the GPT2, Pythia, and Llama-2 families. Higher kurtosis reflects weaker superposition.

We can observe a clear scaling law: as model size increases, kurtosis tends to rise, indicating that the degree of superposition between pieces of knowledge decreases with larger models. This also explains why larger language models exhibit a higher degree of intelligence. A plausible explanation is that as model size grows, the knowledge from the corpus can be stored with weaker superposition, allowing for effective handling of more complex or knowledge-dense scenarios. since the sparser the features, the stronger the superposition (Elhage et al. 2022b), which also means they can only effectively handle scenarios where knowledge features are sparser.

Additionally, we find that the degree of superposition varies across different layers in the same language model. Earlier layers exhibit higher kurtosis and thus lower degree of superposition, while later layers exhibit lower kurtosis and higher degree of superposition. This is reasonable, as early layers in a language model focus on shallow syntactic features, which are dense and result in weaker superposition. In contrast, later layers focus on deep semantic features, which are sparse and result in stronger superposition. This aligns with Elhage et al. (2022b).

![](images/a05fe4eaedc343b2962c4e90e87d9d09ebc7455bb4aed828e994f35e18585d1e.jpg)  
Figure 5: The scaling law of knowledge superposition. Higher kurtosis means less superposition.

![](images/5b47385270faa1bebd6fe08a96ebff2795911e42813e462a61bad67419e58fce.jpg)  
Figure 6: Angular distribution of knowledge representations in whitening space and activation space in last layer.

Furthermore, we find that different architectures have varying impacts on knowledge superposition. As shown in Figure 5, by observing the vertical axis, it is evident that the Pythia and GPT2 architectures seem to encourage more orthogonal representations of knowledge, achieving higher kurtosis with a smaller number of parameters, indicating weaker superposition. In contrast, the Llama2 architecture seems to encourage greater knowledge superposition, with even larger parameter sizes corresponding to relatively lower kurtosis, indicating stronger superposition. This is consistent with Elhage et al. (2022a), which suggests that some architectures may encourage sparsity.

# Superposition in Whitening Space

In this section, we provide further direct evidence of superposition by calculating the angular distribution of knowledge representations, demonstrating its presence in the whitening space. Specifically, we calculate the angles between 128x128 pairs of knowledge representations in the whitening space on Llama3-8B and Llama-3.1-8B. For comparison, we also calculate the angles in the activation space (i.e., directly from MLP’s hidden activations).

In the whitening space (Figure 6ab), the angles are primarily concentrated around 90 degrees, indicating that language models attempt to represent knowledge orthogonally, though limited model capacity results in a long tail. However, in the activation space (Figure 6cd), the relationships between these knowledge representations are indiscernible, showing no orthogonal tendencies.

Importantly, superposition in the whitening space is derived through rigorous mathematical reasoning, whereas superposition in the activation space is based on a heuristic perspective (Elhage et al. 2022b). However, this heuristic perspective has been used as a fundamental assumption in many works that address superposition (Cunningham et al. 2023; Bricken et al. 2023; (Templeton et al. 2024)).

# Conclusion

In this research, we focus on the failure of lifelong knowledge editing and explore its fundamental reason. Our rigorous mathematical derivation indicates that continuous knowledge editing can interfere with both original and previously edited knowledge in language models. Further analysis shows that the degree of interference is determined by the extent of knowledge superposition. Subsequently, we explore the presence and properties of superposition in realworld language models. Our experiments reveal that knowledge superposition is universal across language models, characterized by a high-kurtosis, zero-mean, heavy-tailed distribution, with a clear scaling law. In conclusion, our findings indicate that knowledge superposition is the fundamental reason for the failure in lifelong editing. Nevertheless, the scaling law of superposition still suggests future directions for knowledge editing, including: on the model side, attempting knowledge editing in larger language models with sparser architectures; and on the algorithm side, attempting to decompose knowledge within language models and performing edits based on this decomposition.