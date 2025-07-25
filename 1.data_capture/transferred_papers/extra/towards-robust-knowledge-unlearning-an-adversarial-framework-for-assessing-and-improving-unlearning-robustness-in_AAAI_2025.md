# Towards Robust Knowledge Unlearning: An Adversarial Framework for Assessing and Improving Unlearning Robustness in Large Language Models

Hongbang Yuan1,2 \*, Zhuoran $\mathbf { J i n } ^ { 1 , 2 }$ \*, Pengfei Cao1,2, Yubo Chen1,2 † , Kang Liu1,2, Jun Zhao1,2

1 The Key Laboratory of Cognition and Decision Intelligence for Complex Systems, Institute of Automation, Chinese Academy of Sciences, Beijing, China 2 School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, China {hongbang.yuan, zhuoran.jin, pengfei.cao, yubo.chen , kliu, jzhao} $@$ nlpr.ia.ac.cn

# Abstract

LLM have achieved success in many fields but still troubled by problematic content in the training corpora. LLM unlearning aims at reducing their influence and avoid undesirable behaviours. However, existing unlearning methods remain vulnerable to adversarial queries and the unlearned knowledge resurfaces after the manually designed attack queries. As part of a red-team effort to proactively assess the vulnerabilities of unlearned models, we design Dynamic Unlearning Attack (DUA), a dynamic and automated framework to attack these models and evaluate their robustness. It optimizes adversarial suffixes to reintroduce the unlearned knowledge in various scenarios. We find that unlearned knowledge can be recovered in $5 5 . 2 \%$ of the questions, even without revealing the unlearned model’s parameters. In response to this vulnerability, we propose Latent Adversarial Unlearning (LAU), a universal framework that effectively enhances the robustness of the unlearned process. It formulates the unlearning process as a min-max optimization problem and resolves it through two stages: an attack stage, where perturbation vectors are trained and added to the latent space of LLMs to recover the unlearned knowledge, and a defense stage, where previously trained perturbation vectors are used to enhance unlearned model’s robustness. With our LAU framework, we obtain two robust unlearning methods, AdvGA and AdvNPO. We conduct extensive experiments across multiple unlearning benchmarks and various models, and demonstrate that they improve the unlearning effectiveness by over $5 3 . 5 \%$ , cause only less than a $1 1 . 6 \%$ reduction in neighboring knowledge, and have almost no impact on the model’s general capabilities.

# 1 Introduction

Large language models (LLMs) have achieved remarkable capabilities after being trained on an extensive amount of corpora (Chen et al. 2024; Sun et al. 2024). However, since the training data may contain copyrighted, private, and toxic content, LLMs inevitably learn some potentially undesirable behaviours (Ji et al. 2024). For example, LLMs may regurgitate copyrighted material without permission (Wei et al. 2024a), generate personal information such as phone numbers or mailing addresses (Yao et al. 2024b), and even produce offensive and harmful responses (Liu et al. 2024c).

These unwanted behaviors introduce security concerns and information hazards, hindering the deployment of LLMs in real-world scenarios (Patil, Hase, and Bansal 2024).

To eliminate the influence of problematic content in the corpora on LLMs, machine unlearning has emerged as a promising solution (Eldan and Russinovich 2023; Yao, Xu, and Liu 2024; Liu et al. 2024a). It transforms models to behave as if they were never trained on certain data entries so that specific target knowledge is erased, while other knowledge and capabilities of LLMs are preserved (Chen and Yang 2023; Maini et al. 2024). The most fundamental machine unlearning method is to adopt a gradient ascent procedure on the data that needs to be forgotten to fine-tune LLMs, reversing the original gradient descent optimization process (Jang et al. 2023; Yao et al. 2024a; Liu et al. 2024d).

However, despite their effectiveness, the unlearned models produced by these methods are fragile and susceptible to crafted adversarial user prompts (Patil, Hase, and Bansal 2024; Liu et al. 2024a). Particularly, the previously unlearned knowledge resurfaces through contextual interactions (Shumailov et al. 2024). For example, after being trained to forget the famous writer J.K. Rowling, the model fails to answer the question ‘Who is the author of Harry Potter?’ but still outputs the correct answer to the question ‘Who created the magical world that includes Diagon Alley and the Forbidden Forest?’. To proactively assess the vulnerability of unlearned models to these malicious prompts, static and manually-designed adversarial prompts are used to reintroduce the unlearned knowledge (Jin et al. 2024). However, such a manual process is resource-intensive and often ineffective in guaranteeing the successful reintroduction of the previously unlearned knowledge.

Therefore, we propose a dynamic and automated attack framework, called Dynamic Unlearning Attack (DUA), to quantitatively assess the robustness of the unlearned models. Specifically, as Figure 1(b) shows, we optimize a universal adversarial suffix that maximizes the probability of the model generating unlearned knowledge associated with a given unlearning target. This optimization process is performed and evaluated across various scenarios, considering both practicality and generalization. For practicality, we consider scenarios where the unlearned LLM is either accessible or inaccessible to the attacker. For generalization, we consider whether the adversarial suffix trained on certain

autJh.KorRofwHlianrgryisPtohteter, Original Unlearned J.K Rowling is the author of Harry Original Noise Vector   
a fantasy novel series. Model Model Potter, a fantasy novel series. Model   
theWhHoarisrythPeoattuetrhsoerrioefs? UnMleoadrenled I don’t know. J.KPRototewrl,inagfaisntahseyanuotvheolrsoefriHesa.rry OMriogidneall UnMleoadrenled (a) An example of unlearning J. K. Rowling. (c) A framework of latent adversarial unlearning.   
WHhoarirsythPeotatuetrhsoer ioefst?he EMxopdeerlt Swtoatrilcd tAhttatcikn:clWudheoscrDeiatgeotnheAllmeaygiacnadl UnMleoadrenled J.R.R. Tolkien. the Forbidden Forest?   
Who is the author of the Harry Original Dynamic Attack: Who is the author of Unlearned J. K. Rowling.   
Potter series?  ! ! ! !s ! s!s Model the Harry Potter series? a( sk t &s s sn Model (b) An example of static and dynamic unlearning attack. ! ! ! ! ! ! Trainable Suffix Gradient Descent Gradient Ascent

questions about a specific unlearning target can be employed to other questions about the same target, or even to questions about different targets. For example, if a model is supposed to forget knowledge about J.K. Rowling, an adversarial suffix can be trained to make the model recall knowledge about her Harry Potter series. Then we test whether the same suffix remains effective when applied to questions about her book, The Cuckoo’s Calling, or when in a model intended to forget knowledge about other authors. Experimental results demonstrate that the unlearned knowledge is recovered in $5 4 \%$ of the questions with the adversarial suffixes, even without disclosing the unlearned model to the attacker.

The revealed vulnerability motivates us to enhance the robustness of the unlearning process. Taking inspiration from previous work about adversarial training (Casper et al. 2024; Xhonneux et al. 2024), we propose a universal framework named Latent Adversarial Unlearning (LAU), which effectively enhances the robustness of the unlearned models and is inherently compatible with nearly all gradient-ascentbased unlearning methods. It consists of two optimizing processes: an attack process that aims to bring back the unlearned knowledge, and a defense process that strives to enhance the model’s resistance to such attacks and suppress the recall of unlearned knowledge. For the attack process, as Figure 1(c) shows, we train perturbation vectors that will be added directly to in the latent space of LLMs to promote the unlearned knowledge. Particularly, we add a constrained perturbation vector to the hidden state at a specific layer of a LLM. This approach alleviates the burden of searching the vast input space and facilitates the rapid optimization of adversarial attacks. For the defense process, we fine-tune the model using the previously optimized perturbation vector in its latent space, thereby significantly improve the unlearned model’s resistance to adversarial attacks.

To demonstrate the effectiveness of LAU, we propose AdvGA and AdvNPO, two robust variants of the widely used mainstream unlearning methods, GA and NPO. We conduct unlearning experiments with various models on two commonly used benchmarks, RWKU (Jin et al. 2024) and MUSE (Shi et al. 2024). Experimental results demonstrate that our LAU-augmented methods robustly forget the unlearning target with minimal side effects. Additionally, we assess the robustness of the LAU-trained models using our DUA framework and demonstrate that they exhibit greater resistance to adversarial attacks.

Our contributions can be summarized as follows:

• We propose Dynamic Unlearning Attack (DUA), a dynamic and automated attack framework to quantitatively assess the robustness of the unlearned models. It recovers unlearned knowledge in $5 5 . 2 \%$ of the questions using trained adversarial suffixes, even without the direct access to the unlearned model itself.   
• We propose Latent Adversarial Unlearning (LAU), a universal framework that effectively enhances the robustness of the unlearned process and is compatible with most gradient-ascent-based unlearning methods.   
• We propose two robust unlearning methods, namely AdvGA and AdvNPO. Extensive experiments across multiple unlearning benchmarks and various models demonstrate that they improve the unlearning effectiveness by over $5 3 . 5 \%$ , cause only less than a $1 \bar { 1 } . 3 \%$ reduction in neighboring knowledge, and have almost no impact on the model’s general capabilities.

# 2 Preliminaries

# 2.1 Problem Formulation

Given a LLM $\pi _ { \boldsymbol { \theta } }$ with parameter $\theta$ trained on dataset $D =$ $\{ ( x _ { i } , y _ { i } ) \mid i = 1 , 2 , \bar { . . . } , N \}$ , we define the forget set $D _ { f }$ as the specific training subset to be forgotten. Machine unlearning aims to eliminate the influence of $D _ { f }$ and make model $\pi _ { \boldsymbol { \theta } }$ behaves as if it is only trained on the retain set $D _ { r } = D \backslash D _ { f }$ . Ideally, we can retrain the model on $D _ { r }$ from scratch but it is too costly and unrealistic thus effective approximate unlearning methods are essential. The most commonly used mathematical formulation for optimizing model unlearning is presented below:

$$
\begin{array} { r } { \underset { \theta } { \underbrace { \operatorname* { m i n } } } \underbrace { \mathbb { E } _ { ( x _ { \mathrm { f } } , y _ { \mathrm { f } } ) \in \mathcal { D } _ { \mathrm { f } } } [ \ell _ { \mathrm { f } } ( y _ { \mathrm { f } } \mid x _ { \mathrm { f } } ; \theta ) ] } _ { \mathrm { f o r g e t } } + \lambda \underbrace { \mathbb { E } _ { ( x _ { \mathrm { r } } , y _ { \mathrm { r } } ) \in \mathcal { D } _ { \mathrm { r } } } \ell _ { \mathrm { r } } ( y _ { \mathrm { r } } \mid x _ { \mathrm { r } } ; \theta ) } _ { \mathrm { r e t a i n } } ] } \end{array}
$$

where $\ell _ { \mathrm { f } }$ and $\ell _ { \mathrm { r } }$ are the loss functions on forget set and retain set, respectively, and $\lambda$ is a regularization parameter to balance them.

Typically, the forget set $D _ { f }$ and a subset of the retain set $D _ { r }$ are available in an unlearning task (Shi et al. 2024). In a more practical setting, only an unlearning target $t$ is given and we shall generate a synthetic forget set $D _ { f } ^ { \prime }$ related to this unlearning target $t$ and a corresponding pseudo retain set $D _ { r } ^ { \prime }$ (Jin et al. 2024). In our paper, both scenarios are considered.

# 2.2 Unlearning Methods

We introduce two widely used loss functions for the forget set, gradient ascent and negative preference optimization, and two widely used loss functions for the retain set, gradient descent and KL divergence.

Gradient Ascent. Gradient ascent servers as an important baseline method in machine unlearning and uses the following loss function:

$$
\ell _ { \mathrm { f } } = - \mathbb { E } _ { D _ { \mathrm { f } } } L \left( \pi _ { \theta } ( x , y ) \right)
$$

where $L$ represents a cross-entropy prediction loss. It aims to maximize the average prediction loss of LLMs on the forget dataset, thereby reverting the original gradient descent training process on the forget set.

Negative Preference Optimization. Due to the divergent nature of the gradient ascent method, the unlearning process suffers from catastrophic collapse, wherein the model quickly deteriorates and generates incoherent responses (Zhang et al. 2024). Thus, the negative preference optimization algorithm is introduced, characterized by the following loss function:

$$
\ell _ { \mathrm { f } } = - \frac { 2 } { \beta } \mathbb { E } _ { D _ { \mathrm { f } } } \left[ \log \left( 1 + \left( \frac { L ( \pi _ { \theta } ( x ) , y ) } { L \left( \pi _ { r e f } ( x , y ) \right) } \right) ^ { \beta } \right) \right]
$$

where $\beta$ is a regularization parameter controlling the deviation between the current model $\pi _ { \boldsymbol { \theta } }$ and the original model $\pi _ { r e f }$ . It provides more stable training dynamics and achieves better performance.

Gradient Descent. One widely adopted loss function for the retain set is the standard cross-entropy loss function:

$$
\ell _ { \mathrm { { r } } } = \mathbb { E } _ { D _ { \mathrm { { r } } } } L \left( \pi _ { \theta } ( x , y ) \right)
$$

KL Divergence. Another approach is to minimize the Kullback-Leibler (KL) divergence between the predictions of the original model and the current model on the retain dataset. The loss function can be expressed as:

$$
\ell _ { \mathrm { { r } } } = \mathbb { E } _ { D _ { \mathrm { { r } } } } D _ { K L } ( \pi _ { \theta } ( x , y ) | | \pi _ { r e f } ( x , y ) )
$$

where $\pi _ { \boldsymbol { \theta } }$ and $\pi _ { r e f }$ denote the current model and the original reference model, respectively. It prevents the model from collapsing by ensuring that it does not deviate excessively from the original predictions.

# 2.3 Datasets and Metrics

RWKU. RWKU (Jin et al. 2024) is a real-world knowledge unlearning benchmark that requires models to erase specific knowledge in their parameters. Specifically, for the evaluation of unlearning effectiveness, it provides three types of knowledge probe questions on the forget set: FB, QA and AA. For the evaluation of utility preservation, it provides two types of questions on the neighbor set to test the impact of neighbor perturbation: FB and QA. We use ROUGE-L score (Lin 2004) to measure model’s performance. Lower scores on forget set indicate better unlearning effectiveness, and higher scores on neighbor set indicate better utility preservation. Additionally, it measures the model’s various capabilities, including reasoning (Rea), truthfulness (Tru), factuality (Fac) and fluency (Flu). It also provides some membership inference attacks (MIA) to detect retained knowledge in LLMs, where higher scores indicate that the model retains specific knowledge. Further details and examples are presented in Appendix A.

MUSE. MUSE (Shi et al. 2024) is a comprehensive unlearning benchmark that requires models to unlearn either news articles or book series. Similarly, it also contains the evaluation for unlearning effectiveness and utility preservation. More details are presented in Appendix A.

# 3 Dynamic Unlearning Attack Framework

In this section, we introduce a dynamic, automated framework to assess the robustness of the unlearned models. Firstly, we describe the process for optimizing adversarial suffixes that reintroduce the unlearned knowledge. Subsequently, we introduce the various attack scenarios, focusing on both practicality and generalization. Finally, we conduct experiments on ten unlearned models, demonstrating that they remain susceptible to adversarial attacks even without exposing their parameters.

# 3.1 Adversarial Suffix Optimization

Motivated by the GCG attack in safety-related domains (Zou et al. 2023), we introduce how to optimize attack suffixes in the context of unlearning. Intuitively, we optimize the suffix tokens to maximize the probability of generating the unlearned knowledge.

Consider a question $x _ { [ 0 , m ) ] }$ related to the unlearning target. We aim to find an adversarial suffix $q _ { [ 0 , n ) }$ that, when combined with $x$ to form $[ x ; q ]$ , makes the unlearning model generate a response $y _ { [ 0 , H ) }$ containing the unlearned knowledge. The optimization process can be expressed as:

![](images/72e2e0b90935536a14024b8de367812bfe6296e2ad8f9fd738730eb772c1ddea.jpg)  
Figure 2: Experimental results of our dynamic attack framework. We report the ROUGE-L recall score $( \% )$ in this figure.

$$
\operatorname* { m i n } _ { \substack { q _ { [ 0 , n ) } \in \{ 1 , . . , V \} } } - l o g \prod _ { i = 0 } ^ { H - 1 } p \left( y _ { i + 1 } \mid \left[ x _ { 0 : m } ; q _ { 0 : n } ; y _ { 0 : i } \right] \right)
$$

where $p$ denotes the next token prediction probability, $V$ denotes the vocabulary size, $m$ is the number of tokens in user query $x , n$ is the number of tokens in the adversarial suffix $q$ , and $y$ is the desired response.

To solve this optimization problem, at each optimization step, we leverage the gradients of the tokens to identify a set of candidate tokens, then evaluate them token by token and select the optimal one. Practically, we optimize one single suffix across multiple prompts, resulting in a universal suffix that can be transferred to other queries.

# 3.2 Robustness Assessment

The optimization process defined by Equation 6 can be performed and evaluated in various scenarios. Depending on the choice of the training data $( x , y )$ and the computation of the next token prediction probability $p$ , we design our assessment framework from the perspective of practicality and generalization.

Practicality. We consider two scenarios for the calculation of the next token probability $p$ . (1) Attack Unlearned. The ideal approach is to use the unlearned model, as the adversarial suffix will ultimately be used to attack this model. (2) Attack Original. We also consider a more practical scenario when the unlearned model is not available to the attacker. Therefore, we directly use the models before unlearning to optimize the adversarial suffixes.

Generalization. Typically, the ability of the unlearned models can be assessed by the question-answer style probes related to the unlearning target. Thus the training data $( x , y )$ should be specified as similar question-answer pairs. This raises the question of whether the testing questions are available to the attacker, and whether the unlearning target itself is accessible. We consider three settings, each of which imposes progressively higher demands for generalization.

(1) Within Query. The test questions are available to the attacker, thereby we can directly train adversarial queries on the testing questions. (2) Cross Query. The test questions are not available to the attacker, necessitating the generation of training data based on the unlearning target. (3) Cross Target. The unlearning target itself is not available to the attacker and the training data must be obtained using knowledge probe questions about other unlearning targets.

# 3.3 Experiments

Configuration. To assess the robustness of the unlearned models with our framework, we first conduct unlearning experiments with Llama-3-8B-Instruct on dataset RWKU using the negative preference optimization method. Subsequently, we apply our attack framework to create adversarial queries across various scenarios and evaluate the performance of the unlearned models. We present the average performance of 10 models, each trained with unlearn different targets. For the cross query setting, we generate knowledge probe questions for training based on the unlearning target using GPT-4. For the cross target setting, we use an additional 5 unlearned models along with their corresponding knowledge probe questions to train the adversarial queries.

Additionally, we generate an equivalent number of static attack questions using GPT-4 for comparison. Details of the construction process for the static attack questions, along with specific examples of both dynamic and static attack questions, are provided in Appendix B.

Results. The experimental results are presented in Figure 2. We can draw the following conclusions: (1) The unlearned models are vulnerable to adversarial queries, especially when the unlearned models and test queries are accessible to the attacker. For example, the unlearned model demonstrates a maximum performance increase of $1 5 . 2 5 \%$ using adversarial queries compared to not using them. It indicates that the previously forgotten knowledge gets reintroduced in-context. (2) Models are more resistant to attacks when knowledge probe queries and unlearning targets are inaccessible to attackers. However, our dynamic framework is still able to improve model performance beyond that of static attacks, highlighting the limitations of the resistance. (3) Even without access to the unlearned models, the attacker can still carry out attacks that are nearly as effective as those conducted with access to the unlearned models. For example, the ‘attack original’ lines are almost equivalent to the ‘attack unlearned’ lines and even outperform them in the cross query setting. This reveals a more critical issue that has been overlooked: malicious queries can be trained to recover forgotten knowledge even without prior access to the unlearned model itself.

# 4 Latent Adversarial Unlearning Framework

In this section, we propose an adversarial learning framework to increase the robustness of the unlearning process. First, we will propose a saddle-point (min-max) formulation for adversarial training in the context of machine unlearning, and subsequently, we will elaborate how our framework can be employed to concrete methods in detail.

# 4.1 Framework Formulation

In the context of unlearning, we formulate the adversarial training as the following min-max optimization problem:

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { ( x _ { \mathrm { f } } , y _ { \mathrm { f } } ) \in \mathcal { D } _ { \mathrm { f } } } \left[ \operatorname* { m i n } _ { \epsilon \in T ( x _ { \mathrm { f } } ) } \mathcal { L } \left( \pi _ { \theta } ( x _ { \mathrm { f } } + \epsilon ) , y \right) \right]
$$

where $\mathcal { L }$ is a negative cross-entropy loss function and $T ( x _ { \mathrm { f } } )$ is a set of adversarial perturbations generated by various attack methods. It’s a composition of an inner minimization problem and an outer maximization problem. The inner minimization process aims to identify adversarial queries that effectively bypass the model’s restrictions and activate the forgotten knowledge. The outer maximization strives to suppress the re-emergence of the forgotten knowledge on these adversarial queries.

However, it is non-trivial to identify all the elements in the adversarial query set $T ( x )$ , as there are too many potential adversarial queries hidden beneath. Additionally, optimizing a discrete set of tokens is challenging. To avoid the intricate process of optimizing adversarial queries, we propose latent adversarial unlearning. The core idea is that any type of adversarial query will cause a perturbation in the latent space of LLMs, potentially leading to the resurgence of forgotten knowledge. Therefore, we directly add perturbations to the latent space of LLMs, thus avoiding the extensive optimization in the input space. This process can be formulated as the following min-max optimization problem:

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { \mathcal { D } _ { \mathrm { f } } } \left[ \operatorname* { m i n } _ { \delta } \mathcal { L } \left( \pi _ { \theta _ { 2 } } ( \pi _ { \theta _ { 1 } } ( x _ { \mathrm { f } } ) + \delta _ { x _ { \mathrm { f } } } ) , y _ { \mathrm { f } } \right) \right] s . t . \| \delta _ { x _ { \mathrm { f } } } \| \leq \kappa
$$

where $\pi _ { \theta _ { 1 } }$ and $\pi _ { \theta _ { 2 } }$ represent the computations in LLM $\pi _ { \boldsymbol { \theta } }$ before and after the perturbation $\delta$ is added, respectively. The $L _ { 2 }$ -norm of the perturbation vector is restricted to a constant $\kappa$ . In this way, both the inner and outer optimization problems can be solved using gradient descent algorithms.

Practically, we add a perturbation vector to the residual stream at a specific layer of a transformer model. For each batch of samples, we optimize the perturbation vector using its gradient for a fixed number of steps. Subsequently, we apply the classical stochastic gradient descent algorithm to update the model’s parameters, with the previously optimized perturbation vector in its residual streams. The impact of the choice of perturbation layers and the number of inner optimization steps will be discussed in Section 5.

# 4.2 Two Adversarial Unlearning Methods

Our adversarial unlearning framework is suitable for most of the existing machine unlearning algorithms. In this paper, we apply our framework to augment the GA and NPO methods, resulting in two new algorithms: AdvGA and AdvNPO, which are described below.

AdvGA. By substituting the internal minimization loss function in Equation 8 with Equation 3, we can obtain the following loss function 1:

$$
\operatorname* { m i n } _ { \pi _ { \theta } } - \mathbb { E } _ { D _ { \mathrm { f } } } \operatorname* { m i n } _ { \delta } L \left( \pi _ { \theta _ { 2 } } ( \pi _ { \theta _ { 1 } } ( x ) + \delta , y ) \right)
$$

We denote this new loss function $A d \nu G A$ .

AdvNPO. Similarly, we substitute the internal minimization loss function in equation 8 with Equation 3 and the following loss function is obtained:

$$
\operatorname* { m i n } _ { \pi _ { \theta } } - \frac { 2 } { \beta } \mathbb { E } _ { D _ { \mathbf { f } } } \left[ \log \left( 1 + \left( \frac { \operatorname* { m i n } _ { \delta } L \left( \pi _ { \theta _ { 2 } } ( \pi _ { \theta _ { 1 } } ( x ) + \delta , y ) \right) } { L \left( \pi _ { r e f } ( x , y ) \right) } \right) ^ { \beta } \right) \right]
$$

We denote this new loss function $A d v N P O$ . For clarification, we omit the $L _ { 2 }$ -norm restriction on the perturbation vector here, but it should be included during the optimization process. This also applies to the loss function in AdvGA.

# 4.3 Experiments

Configurations. We conduct machine unlearning experiments on the following two datasets: RWKU (Jin et al. 2024) and MUSE (Shi et al. 2024). We combine the previously introduced forget loss functions and retain loss functions, and finally obtain 12 unlearning methods as shown in Table 1. We set the perturbation layer to 4, the inner optimization steps to 6, and the weights of the forget and retain loss functions to be equal. We conduct experiments with LLaMA-2- 7B-Chat (Touvron et al. 2023), LLaMA-3-8B-Instruct and LLaMA-3.1-8B-Instruct. Following previous work, we run the optimizing process using the AdamW optimizer with a cosine learning rate scheduler. All the experiments are conducted on 4 Nvidia A100 GPUs. Further details are provided in Appendix C.

Results. The experimental results on RWKU with LLaMA-3-8B-Instruct and LLaMA-3.1-8B-Instruct are presented in Table 1. Additional results with other models on MUSE are provided in the Appendix C. From the table, we can draw the following conclusions:

(1) Our methods, particularly the AdvNPO series, are highly effective in unlearning the real-world knowledge in LLMs. For example, $\mathrm { A d v N P O _ { K L R } }$ achieves a performance increase of $5 7 \%$ on the forget set, but only causes a drop of $1 0 . 6 \%$ on the neighbor set comparing with a vanilla NPO method. This demonstrates the effectiveness of the LAU framework. (2) Our methods cause almost no side effects on the general capabilities of LLMs. For example, performance on the utility set remains nearly the same before and after the unlearning process. This demonstrates that the unlearned models remain powerful in many scenarios, despite having specific knowledge removed.

Table 1: Experimental results on RWKU with LLaMA-3-8B-Instruct and LLaMA-3.1-8B-Instruct. The superscript denotes the performance increase of our adversarial methods compared to the corresponding non-adversarial versions. Please refer to Section 2.3 for the meaning of the abbreviations.   

<html><body><table><tr><td rowspan="2">Methods</td><td colspan="4">Forget Set ↓</td><td colspan="3">Neighbor Set ↑</td><td colspan="2">MIA Set</td><td colspan="4">Utility Set ↑</td></tr><tr><td>FB</td><td>QA</td><td>AA</td><td>All</td><td>FB</td><td>QA</td><td>All</td><td>FM↑</td><td>RM↓</td><td>Rea</td><td>Tru</td><td>Fac</td><td>Flu</td></tr><tr><td colspan="10">LLaMA-3-8B-Instruct</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Before</td><td>85.6</td><td>70.3</td><td>74.7</td><td>76.9</td><td>93.1</td><td>82.0</td><td>87.6</td><td>236.5</td><td>230.9</td><td>41.0</td><td>36.4</td><td>53.7</td><td>704.6</td></tr><tr><td>GA</td><td>72.0</td><td>64.6</td><td>68.5</td><td>68.4</td><td>85.0</td><td>74.7</td><td>79.8</td><td>241.4</td><td>234.6</td><td>40.4</td><td>37.6</td><td>49.6</td><td>710.3</td></tr><tr><td>AdvGA</td><td>63.0</td><td>48.2</td><td>60.5</td><td>57.2 ↓16.4%</td><td>75.8</td><td>72.1</td><td>74.0 ↓7.3%</td><td>202.0</td><td>176.5</td><td>40.1</td><td>35.2</td><td>49.4</td><td>717.0</td></tr><tr><td>GAGDR</td><td>72.6</td><td>64.0</td><td>69.7</td><td>68.8</td><td>86.2</td><td>76.5</td><td>81.4</td><td>242.8</td><td>236.8</td><td>39.6</td><td>36.8</td><td>50.4</td><td>710.3</td></tr><tr><td>AdvGAGDR</td><td>69.2</td><td>52.4</td><td>66.1</td><td>62.6 {9.0%</td><td>85.7</td><td>73.7</td><td>79.7 ↓2.1%</td><td>205.2</td><td>184.5</td><td>41.4</td><td>35.4</td><td>50.5</td><td>712.1</td></tr><tr><td>GAKLR</td><td>70.7</td><td>57.5</td><td>69.9</td><td>66.1</td><td>80.5</td><td>70.5</td><td>75.5</td><td>242.4</td><td>230.8</td><td>41.5</td><td>35.6</td><td>54.0</td><td>704.4</td></tr><tr><td>AdvGAKLR</td><td>58.8</td><td>43.8</td><td>59.5</td><td>54.0 {18.3%</td><td>76.9</td><td>63.0</td><td>69.9 ↓7.4%</td><td>371.3</td><td>340.8</td><td>41.2</td><td>33.8</td><td>50.5</td><td>712.6</td></tr><tr><td>NPO</td><td>46.6</td><td>39.0</td><td>35.3</td><td>40.3</td><td>79.2</td><td>70.9</td><td>75.1</td><td>263.3</td><td>241.4</td><td>40.5</td><td>36.0</td><td>56.7</td><td>695.9</td></tr><tr><td>AdvNPO</td><td>19.7</td><td>14.7</td><td>12.0</td><td>15.5 ↓61.5%</td><td>67.0</td><td>59.7</td><td>63.3↓15.7%</td><td>270.1</td><td>238.9</td><td>39.3</td><td>34.0</td><td>56.8</td><td>663.1</td></tr><tr><td>NPOGDR</td><td>52.2</td><td>43.9</td><td>42.9</td><td>46.3</td><td>82.5</td><td>70.5</td><td>76.5</td><td>254.5</td><td>240.1</td><td>39.6</td><td>37.2</td><td>51.4</td><td>708.2</td></tr><tr><td>AdvNPOGDR</td><td>25.5</td><td>22.1</td><td>16.5</td><td>21.4 ↓53.8%</td><td>71.9</td><td>69.1</td><td>70.5 ↓7.8%</td><td>248.8</td><td>223.1</td><td>41.9</td><td>35.8</td><td>52.4</td><td>705.2</td></tr><tr><td>NPOKLR</td><td>52.5</td><td>40.6</td><td>43.2</td><td>45.4</td><td>83.2</td><td>72.1</td><td>77.6</td><td>253.0</td><td>236.9</td><td>40.9</td><td>35.4</td><td>54.2</td><td>704.9</td></tr><tr><td>AdvNPOKLR</td><td>23.6</td><td>18.9</td><td>16.0</td><td>19.5 ↓57.0%</td><td>72.1</td><td>66.8</td><td>69.4 ↓10.6%</td><td>347.2</td><td>318.1</td><td>41.7</td><td>35.6</td><td>55.3</td><td>697.1</td></tr><tr><td colspan="10">LLaMA-3.1-8B-Instruct</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Before</td><td>63.9</td><td>65.1</td><td>69.5</td><td>66.2</td><td>74.1</td><td>69.8</td><td>72.0</td><td>223.5</td><td>218.2</td><td>42.2</td><td>35.4</td><td>61.2</td><td>695.2</td></tr><tr><td>GA</td><td>50.7</td><td>45.4</td><td>61.2</td><td>52.4</td><td>45.6</td><td>37.2</td><td>41.4</td><td>248.9</td><td>241.9</td><td>43.2</td><td>35.8</td><td>48.7</td><td>726.6</td></tr><tr><td>AdvGA</td><td>32.0</td><td>22.5</td><td>36.0</td><td>30.2 ↓42.4%</td><td>27.5</td><td>21.0</td><td>24.3 ↓41.3%</td><td>173.7</td><td>125.9</td><td>39.8</td><td>33.0</td><td>28.8</td><td>730.1</td></tr><tr><td>GAGDR</td><td>55.4</td><td>49.6</td><td>63.9</td><td>56.3</td><td>60.2</td><td>53.5</td><td>56.9</td><td>239.8</td><td>231.3</td><td>44.2</td><td>35.0</td><td>53.9</td><td>718.5</td></tr><tr><td>AdvGAGDR</td><td>44.0</td><td>34.1</td><td>47.8</td><td>42.0 ↓25.4%</td><td>62.6</td><td>52.5</td><td>57.6 †1.2%</td><td>71.9</td><td>62.3</td><td>43.2</td><td>35.8</td><td>52.7</td><td>718.6</td></tr><tr><td>GAKLR</td><td>62.7</td><td>49.9</td><td>66.4</td><td>59.7</td><td>67.9</td><td>61.2</td><td>64.6</td><td>235.8</td><td>223.0</td><td>42.6</td><td>35.4</td><td>59.0</td><td>682.1</td></tr><tr><td>AdvGAKLR</td><td>50.8</td><td>42.0</td><td>54.8</td><td>49.2 ↓17.6%</td><td>59.8</td><td>59.8</td><td>59.8 ↓7.4%</td><td>69.1</td><td>67.2</td><td>43.1</td><td>33.4</td><td>57.3</td><td>697.5</td></tr><tr><td>NPO</td><td>35.7</td><td>40.2</td><td>39.0</td><td>38.3</td><td>67.3</td><td>66.2</td><td>66.7</td><td>241.4</td><td>220.5</td><td>42.5</td><td>35.6</td><td>61.8</td><td>684.2</td></tr><tr><td>AdvNPO</td><td>18.0</td><td>21.7</td><td>16.5</td><td>18.7 ↓51.2%</td><td>60.0</td><td>57.2</td><td>58.6 ↓12.1%</td><td>108.3</td><td>86.9</td><td>41.1</td><td>35.4</td><td>61.4</td><td>677.8</td></tr><tr><td>NPOGDR</td><td>42.4</td><td>37.2</td><td>42.0</td><td>40.5</td><td>74.0</td><td>66.7</td><td>70.3</td><td>236.3</td><td>220.1</td><td>43.0</td><td>35.4</td><td>60.8</td><td>698.8</td></tr><tr><td>AdvNPOGDR</td><td>23.1</td><td>20.8</td><td>16.7</td><td>20.2 ↓50.1%</td><td>62.4</td><td>59.7</td><td>61.1 ↓13.1%</td><td>91.0</td><td>77.6</td><td>42.6</td><td>35.4</td><td>60.7</td><td>696.1</td></tr><tr><td>NPOKLR</td><td>40.6</td><td>41.4</td><td>42.2</td><td>41.4</td><td>73.3</td><td>69.9</td><td>71.6</td><td>234.4</td><td>218.8</td><td>42.3</td><td>35.4</td><td>61.5</td><td>695.1</td></tr><tr><td>AdvNPOKLR</td><td>24.1</td><td>18.5</td><td>19.4</td><td>20.7 ↓50.0%</td><td>65.0</td><td>61.0</td><td>63.0 ↓12.0%</td><td>88.9</td><td>74.9</td><td>42.2</td><td>35.2</td><td>60.5</td><td>690.2</td></tr></table></body></html>

# 5 Discussion

# 5.1 Influence of the Perturb Layers

We explore how the specific layer at which the perturbation vector is added influences the final performance. Therefore, we vary the perturbation layer from 0 to 30 for the Llama-3- 8B-Instruct model and report the averaged performance on the forget and neighbor datasets for one unlearned model. The experimental results are shown in the upper half of Figure 3. We can draw the following conclusions.

(1) Adding perturbations at the shallow layers (those closer to the input prompts) is more effective. We attribute this to the fact that perturbations at shallower layers have a more significant impact on the model’s output, making them easier to optimize. (2) Directly adding perturbation at the embedding layer is entirely ineffective, as indicated by the point where the perturbation layer equals zero. This is due to the fact that our latent perturbation serves as an approximation of the adversarial queries, but directly adding perturbation at the embedding layer alters the entire prompt rather than simply adding an adversarial suffix. (3) As the perturbation layers get deeper, the performance converges to that of the corresponding non-adversarial method. This finding aligns with the intuition that deeper perturbation layers result in a more limited influence of the perturbation vectors.

# 5.2 Influence of the Inner Optimization Steps

Similarly, we also explore the influence of the number of inner optimization steps in Equation 8. We follow the same experimental configurations as before, but instead vary the number of inner optimization steps. The experimental results are shown in the lower half of Figure 3. We can draw the following conclusions:

(1) As we increase the optimization steps, the performance on the forget dataset initially declines, then rises. We thereby conclude that both insufficient and excessive optimization steps are detrimental to unlearning performance. (2) Regardless of the number of optimization steps, the model’s performance on the forget dataset consistently remains below that of non-adversarial methods. Even a small, randomly initialized perturbation vector can enhance the robustness of the unlearning process, as indicated by the point where the step equals 0.

![](images/afffefe4f9172078b585709fdce42e19cf1857687dd5381244e512d705a3cfbe.jpg)  
Figure 3: Influence of the perturb layers and the inner optimization steps. We report the ROUGE-L recall score $( \% )$ .

# 5.3 Robustness of Latent Adversarial Unlearning

Finally, we evaluate the robustness of the unlearned models trained with LAU-augmented methods under our previously proposed dynamic attack framework. We select ten unlearned models and train adversarial suffixes with both the original model and the unlearned model in the within query setting. For a clearer comparison with unlearned models trained using non-adversarial approaches, we report the performance change relative to the scenario without attack. The experimental results are presented in Figure 4. We can draw the following conclusions:

(1) The LAU-trained models are more robustness than the non-LAU-trained models. For instance, the increased performance under adversaries of the model trained with NonLAU is nearly twice that of the model trained with LAU. This demonstrates a significant enhancement in the robustness of the unlearned models. (2) The unlearned models become safer especially when their parameters are not available to the attacker. For example, the ‘Attack LAU (original)’ line consistently remains at a low value as the attack optimization steps increase.

# 6 Related Work

Jailbreak Attack. To mitigate the undesirable behaviors of LLMs, the safety-alignment stage has become essential (Wei, Haghtalab, and Steinhardt 2023; Paulus et al. 2024;

![](images/184d356069b02609e8a00e46c784bd36492f4e38d579c4b2fe4e90cc60dc330c.jpg)  
Figure 4: Robustness evaluation of AdvNPO. We report the performance change $( \Delta )$ in terms of the ROUGE-L recall score $( \% )$ compared to the scenario without attack.

Tamirisa et al. 2024). Within this context, a complementary approach called red teaming is proposed to assess the robustness of the safety-alignment of LLMs by designing jailbreaking attacks (Carlini et al. 2023; Huang et al. 2024). Some work focuses on designing manually crafted adversarial prompts (Yong, Menghini, and Bach 2024; Wei et al. 2024b), while others explore automatically generating the prompts via gradient-based optimization (Jones et al. 2023; Zou et al. 2023) or genetic-base methods (Lapid, Langberg, and Sipper 2023; Liu et al. 2024b). However, they primarily focus on the introduction of harmful behaviors, whereas we focus on the resurgence of unlearned knowledge.

Machine Unlearning. Data protection regulations, such as the European General Data Protection Regulation (GDPR) (Mantelero 2013), have mandated “the Right to be Forgotten” and highlight the necessity of machine unlearning (Hu et al. 2024; Schwinn et al. 2024; Sheshadri et al. 2024). Therefore, a number of unlearning benchmarks are proposed, including the forgetting of Harry Potter books (Eldan and Russinovich 2023), facts about fictional authors (Maini et al. 2024) and real world knowledge in LLMs (Jin et al. 2024). Additionally, a line of research explores the methodologies for effective machine unlearning, from variants of gradient-ascent (Jia et al. 2023; Zhang et al. 2024), to localization-informed approaches (Wu et al. 2023; Fan et al. 2024; Rosati et al. 2024). However, existing unlearning methods remain vulnerable and our proposed LAU framework provides a universal solution to enhancing the robustness of the unlearning process.

# 7 Conclusion

In this paper, we propose a dynamic and automated framework to assess the vulnerabilities of the unlearned models. After revealing their susceptibility, we propose a latent adversarial training framework, along with two concrete methods, namely AdvGA and AdvNPO. Extensive experiments on several datasets with various models demonstrate the effectiveness and robustness of our unlearning methods.