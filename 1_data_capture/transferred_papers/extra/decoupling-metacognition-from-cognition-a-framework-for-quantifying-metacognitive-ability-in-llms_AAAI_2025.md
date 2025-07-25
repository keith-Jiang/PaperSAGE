# Decoupling Metacognition from Cognition: A Framework for Quantifying Metacognitive Ability in LLMs

Guoqing Wang1, Wen $\mathbf { W _ { u } } ^ { 1 , 2 * }$ , Guangze $\mathbf { Y e } ^ { 1 , 3 , 4 }$ , Zhenxiao Cheng1, Xi Chen2, Hong Zheng5

1School of Computer Science and Technology, East China Normal University, Shanghai, China 2Shanghai Key Laboratory of Mental Health and Psychological Crisis Intervention, School of Psychology and Cognitive Science, East China Normal University, Shanghai, China 3Lab of Artificial Intelligence for Education, East China Normal University, Shanghai, China 4Shanghai Institute of Artificial Intelligence for Education, East China Normal University, Shanghai, China 5Shanghai Changning Mental Health Center, Shanghai, China {wgq, gzye, 51255901012}@stu.ecnu.edu.cn, wwu@cc.ecnu.edu.cn, xchen $@$ psy.ecnu.edu.cn, zhhmm2@163.com

# Abstract

Large Language Models (LLMs) are known to hallucinate facts and make non-factual statements which can undermine trust in their output. The essence of hallucination lies in the absence of metacognition in LLMs, namely the understanding of their own cognitive processes. However, there has been limited research on quantitatively measuring metacognition within LLMs. Drawing inspiration from cognitive psychology theories, we first quantify the metacognitive ability of LLMs as their ability to evaluate the correctness of responses through confidence. Subsequently, we introduce a general framework called DMC designed to decouple metacognitive ability and cognitive ability. This framework tackles the challenge of noisy quantification caused by the coupling of metacognition and cognition in current research, such as calibration-based metrics. Specifically, the DMC framework comprises two key steps. Initially, the framework tasks the LLM with failure prediction, aiming to evaluate the model’s performance in predicting failures, a performance jointly determined by both cognitive and metacognitive abilities of the LLM. Following this, the framework disentangles metacognitive ability and cognitive ability based on the failure prediction performance, providing a quantification of the LLM’s metacognitive ability independent of cognitive influences. Experiments conducted on eight datasets across five domains reveal that (1) Our proposed DMC framework effectively separates the metacognition and cognition of LLMs; (2) Various confidence elicitation methods impact the quantification of metacognitve ability differently; (3) Stronger metacognitive ability are exhibited by LLMs with better overall performance; (4) Enhancing metacognition holds promise for alleviating hallucination issues.

# Introduction

Large language models (LLMs) such as GPT-3.5 (OpenAI 2021), GPT-4 (Achiam et al. 2023) and Llama (Touvron et al. 2023) have demonstrated remarkable capabilities in various natural language processing tasks, owing to their capacity for learning complex patterns and generating coherent

Coupled 一 has bad metacognition G Hard Task Bad FP-P × 一自 has good metacognition Easy Task Good FP-P   
Decoupled Bad Cognitive Performance Hard Task Bad FP-P 5 has good metacognition Good Metacognitive Performance 良一自 Good Cognitive Performance Easy Task Good FP-P

text (Yu et al. 2023). However, alongside their advantages, LLMs also exhibit certain limitations that warrant attention. Hallucination (Guan et al. 2024; Gunjal, Yin, and Bas 2024; Chen et al. 2024a,b) is one significant issue of concern as it can impact LLMs’ performance and the practical applications of downstream tasks by generating fictitious information or responses not grounded in the input data, challenging the reliability and trustworthiness of LLM outputs. An emerging consensus among experts in Artificial Intelligence (AI) suggests that the root cause of hallucination in LLMs lies in the lack of understanding of their own cognitive processes (Gekhman et al. 2024; Mielke et al. 2020; Dubey et al. 2024), a concept well-established in cognitive psychology as metacognition (Fleming and Lau 2014). While efforts have been made to introduce the concept of metacognition to address hallucination and enhance the performance of downstream tasks (Zhou et al. 2024; Dubey et al. 2024; Li et al. 2024), current research lacks a definitive quantitative framework for measuring LLMs’ metacognitive ability. This gap continues to result in opacity regarding the mechanisms behind the phenomena, leading to a lack of sufficient explainability and applicability.

To address this research gap, we propose a novel quantitative framework for measuring metacognitive ability in LLMs, drawing inspiration from theories in cognitive psychology. Specifically, we first attribute the quantification of LLMs’ metacognitive ability to evaluating the accuracy of confidence in representing the uncertainty of responses. The rationale behind this design stems from two aspects. On one hand, from the perspective of cognitive psychology in humans, confidence is viewed as a reflection of the brain’s uncertainty regarding one’s behavior (De Martino et al. 2013). This uncertainty, as indicated by confidence levels, mirrors the understanding of behavior, i.e., the cognitive process (Fleming and Dolan 2012). Cognitive scientists argue that the precision of confidence in reflecting behavioral uncertainty can serve as a gauge of metacognitive ability (Fleming 2024). In essence, individuals demonstrating high confidence levels when performing well and low confidence when performing poorly exhibit robust metacognitive capacity. Conversely, a lack of alignment between confidence levels and performance suggests weaker metacognitive ability. On the other hand, drawing parallels with human cognition, existing work on LLMs is increasingly focusing on the confidence of their models (Xiong et al. 2024), which can provide valuable insights. Confidence of LLM can be regarded as an encoded representation of the uncertainty within the internal parameters of these models concerning their outputs (Kadavath et al. 2022; Tian et al. 2023), and this representation shares many similarities with that observed in humans (Zhou, Jurafsky, and Hashimoto 2023).

Building upon our proposed quantification of metacognitive ability in LLMs, we further introduce a universal framework named DMC to specifically measure LLMs’ metacognitive ability. While some existing methods, such as calibration-based metrics (Guo et al. 2017), have also attempted to quantify LLMs’ ability by assessing response accuracy through confidence levels, they struggle to directly capture the true metacognitive essence of LLMs, often conflating the impact of cognitive ability during quantification. The upper part of Figure 1 illustrates specific issues with methods like calibration-based metrics. When large models engage in simple and complex tasks, their performance is influenced by both metacognitive ability (i.e., accuracy of the confidence reported by LLMs in representing the uncertainty of their responses) and cognitive ability (i.e., accuracy on downstream tasks). However, these methods fail to disentangle the effects of these two aspects, leading to a misjudgment where LLMs are erroneously deemed to have superior metacognition in simpler tasks and weaker metacognition in more complex tasks. In contrast, our proposed DMC framework can effectively distinguish between metacognitive ability and cognitive ability. As shown in the lower part of Figure 1, task difficulty only impacts the cognitive performance and does not influence the assessment of metacognitive ability in LLMs.

Concretely, our DMC framework consists of two essential steps. In the first step, the LLM is assigned the task of predicting failures (Xiong et al. 2022) using binary choice problems. Here, the LLM offers responses and subsequently expresses its confidence in those answers. The performance in failure prediction can reflect both cognitive and metacognitive ability, though they are interconnected. In the second step, the framework involves a crucial process of decoupling metacognition and cognition. Drawing from signal detection theory (SDT) (Green, Swets et al. 1966; Maniscalco and Lau 2012), we develop the LLM-SDT model to measure the LLM’s cognitive ability and the LLM-Meta-SDT model to gauge its metacognitive ability. Briefly speaking, the LLM-SDT model simulates the process of the LLM providing answers, with its parameters representing the LLM’s cognitive ability. To effectively quantify this cognitive ability, we separate cognitive performance from failure prediction and recalibrate the parameters of the LLM-SDT model based on cognitive performance. As for LLM-Meta-SDT, it is an extension of the LLM-SDT that can simulate the process of expressing confidence in LLM in addition to the process of giving an answer. Regrettably, the direct isolation of metacognitive performance from failure prediction performance poses a challenge in quantification. To tackle this issue, we employ a strategic approach. Initially, the LLMMeta-SDT model is utilized to fit the failure prediction performance that coupled cognitive and metacognitive performance. Constraints are then applied to the parameters of the LLM-Meta-SDT to ensure optimal metacognitive ability in acquiring the required cognitive ability for a given context. Subsequently, we calculate the difference between the required cognitive ability and the LLM’s actual cognitive ability, which can be directly mapped to the gap $G$ between the subject’s actual metacognitive ability and the optimal metacognitive ability. Ultimately, we use this difference $G$ to quantify the metacognitive ability of the large language model. This strategy is based on the intuition that the performance in failure prediction is jointly determined by the cognitive ability and metacognitive ability of the subject. If the subject possesses optimal metacognitive ability, the cognitive ability needed to achieve the current failure prediction performance should be lower than their actual cognitive ability.

Our Experiments on eight datasets covering five scopes show that: (1) DMC decouples metacognitive and cognitive ability in LLMs more effectively than calibraton-based methods; (2) Different confidence elicitation methods vary in their impact on quantifying metacognitive ability; (3) Advanced LLMs like GPT-4 exhibit stronger metacognition ability; (4) LLM’s metacognitive ability aligns with their performance in the AbstainQA task.

To summarize, the contributions are listed as follows:

• We propose, for the first time, a concrete quantification of the metacognitive ability of LLMs based on cognitive psychology theories.   
• We introduce a general DMC framework that successfully decouples metacognition from cognition in quantification processes.   
• We verify DMC’s decouping capability through designed experiments and explore the factors that influence metacogniton quantificaiton, highlighting

![](images/a2d16fcd1efd112c81faccb219d8b73d323011e0b905fea36c69216d75571979.jpg)  
Figure 2: Illustration of (a) LLM-SDT model and (b) LLMMeta-SDT model.

the potential of improving metacognition to address hallucination issues. The code is available at (https://github.com/Angelo3357/DMC.git).

# Related Work

Metacognition Quantification in Human. Quantifying metacognitive ability in humans has been a longstanding focus in cognitive psychology. Mason (Mason 2003) linked metacognition and confidence, emphasizing the correlation between confidence and performance. This led to the widespread adoption of confidence-based quantification. However, Masson and Rotello (Masson and Rotello 2009) revealed the limitations of traditional correlation methods within this paradigm. Addressing this, Maniscalco and Lau (Maniscalco and Lau 2012) introduced Signal Detection Theory (SDT) into metacognitive quantification, enhancing the theoretical framework. These works formed the foundation for quantifying metacognitive ability in LLMs.

Confidence Elicition in LLMs. Confidence elicitation estimates LLM confidence without model adjustment or internal data access (Xiong et al. 2024). (Tian et al. 2023) proposed prompt strategies for verbalizing confidence in LLMs, favoring verbal over token-likelihood confidence for RLHFLMs. (Xiong et al. 2024) improved this with a consistencybased sampling-aggregation strategy for black-box LLMs. (Lin, Trivedi, and Sun 2023) showed that using Natural Language Inference to quantify response similarities enhances consistency-based confidence. These methods will be applied in our DMC framework for eliciting LLM’s confidence.

# Preliminaries

To decouple metacognitive ability and cognitive ability, we define the LLM-SDT model for quantifying the cognitive ability of LLMs and the LLM-Meta-SDT model for quantifying their metacognitive ability.

# LLM-SDT

Figure 2(a) illustrates LLM-SDT model, which assumes that each time an LLM receives a binary-choice problem, it generates an internal belief $x$ , which the LLM uses to decide whether the current problem’s answer is $A$ or $B$ . For each label type, $x$ is drawn from normal distributions, with the distance between the two distributions being $d$ , which measures the LLM’s ability to distinguish the correct option, i.e., cognitive ability. $c ^ { 0 }$ is the decision axis, representing the LLM’s internal criterion: if $x$ exceeds $c ^ { 0 }$ , the LLM responses $B$ ; otherwise, it responses $A$ . Let $f ( x | A )$ and $f ( x | B )$ represent the normal distributions corresponding to labels $A$ and $B$ . For simplicity and without loss of effectiveness, we define the value of $x$ at the intersection of $f ( x | A )$ and $f ( { \boldsymbol { x } } | B )$ as zero and set the variances of both $f ( x | A )$ and $f ( x | B )$ to 1. Therefore, the means of A and $\mathbf { B }$ can be represented as $- { \frac { d } { 2 } }$ and $\textstyle { \frac { d } { 2 } }$ . In $f ( x | A )$ , the areas under the curve to the left and right of $c _ { 0 }$ represent the probabilities of the LLM correctly and incorrectly answering a problem with label $A$ , respectively. The same applies to $f ( { \boldsymbol { x } } | B )$ . In summary, we denote LLM-SDT model by $L S ( d , c ^ { 0 } )$ .

# LLM-Meta-SDT

Figure $2 ( \boldsymbol { \mathsf { b } } )$ illustrates how we extend LLM-SDT model by adding confidence decision axes to represent the process of expressing confidence in the LLM. The extended model is called LLM-Meta-SDT model, which is used to quantify the metacognitive ability of LLMs. Let $\pmb { c } _ { A } = [ c _ { A } ^ { 0 } , c _ { A } ^ { 1 } , . . . , c _ { A } ^ { k - 1 } ]$ , $\pmb { c } _ { B } = [ c _ { B } ^ { 0 } , c _ { B } ^ { 1 } , . . . , c _ { B } ^ { k - 1 } ]$ represent the confidence decision axex for $f ( x | A )$ and $f ( x | B )$ in LLM-Meta-SDT, respectively, where $\overset { \cdot } { c } _ { A } ^ { 0 }$ and $c _ { B } ^ { 0 }$ are both refer to $c ^ { 0 }$ , which denotes the position of zero confidence. The parameter $k$ represents the total number of confidence ratings. The confidence decision axes indicates that if the internal belief $x$ falls between $c _ { A } ^ { i }$ and $c _ { A } ^ { i - 1 }$ (or $c _ { B } ^ { i }$ and $c _ { B } ^ { i - 1 } .$ ), it reflects that the LLM has a confidence rating of $i$ for its response, and if $x$ falls to the left of $c _ { A } ^ { k - 1 }$ (or the right of $c _ { B } ^ { k - 1 }$ ), it reflects that the LLM has a confidence rating of $k$ . Apart from the confidence decision axes, the other settings of the LLM-Meta-SDT are the same as those of the LLM-SDT. We denote LLM-Meta-SDT model by $L M S ( d , \pmb { c } _ { A } , \pmb { c } _ { B } )$ .

# Proposed Framework

The overall framework of DMC is depicted in Figure 3. DMC comprises two steps: (1) Tasking LLM with Failure Prediction; (2) Decoupling Metacognition from Cognition.

# Step1: Tasking LLM with Failure Prediction

Given a binary-choice question $q$ , we can prompt the LLM $M$ to generate an answer and apply confidence eliction method $c e$ to elicit confidence:

$$
\hat { y } , c o n f = M _ { c e } ( q ) ,
$$

where $M _ { c e }$ represents $M$ using $c e$ to elicit confidence, and $\hat { y }$ and con $f$ represent the answer and confidence generated by $M _ { c e }$ . However, the confidence elicited by $c e$ is typically continuous. To facilitate the quantification in the following step , we apply a binning strategy (e.g. equal-width binning) to convert the continuous confidence into discrete confidence ratings after completing the binary-choice failure prediction task:

$$
\begin{array} { c } { { r a t i n g = B i n n i n g ( c o n f , k ) , } } \\ { { O = [ \hat { \pmb { y } } , { \pmb { y } } , r a t i n g ] , } } \end{array}
$$

Step 1: Tasking LLM with Failure Prediction   
Step 2: Decoupling Metacognition from Cognition   
Figure 3: Overview of the proposed framework DMC for quantifying metacognitive ability in LLMs.

Binary-choice problems Cognitive ability quantifying Cognitive ability True or (( not True )) is? Failure prediction Isolate Cognitive   
A. True    B. False performance performance … LLM-SDT model Cognitive ability Manifested as the accuracy of the answers Metacognitive ability quantifying LLM 2 Cognitive Gap + performance Required cognitive ability   
Confidence eliciting Coupled Contain Accessible Compute Represents cNomt pdiurteacbtley Failure prediction Fit Gap Failure prediction performance Metacognitive ability performance LLM-Meta-SDT model Metacognitive   
Ans Conff Rating Determined Contain Inaccessible Constrain ability   
…B 0….1 Binning 1… by anpsrewdeircst  threocuogrhrecotnfeisdseonfce Mpertfaocromganintcieve Optimal metacognitive ability mGqetauapcnroteigfpniricetaistveieontnasbotiflhiety

where Binning and $k$ represent the binning strategy and number of bins; ${ \hat { y } } , y$ and $c o n f$ represent the vectors containing all $\hat { y } , y$ and $c o n f$ ; rating represent the discretized result of $c o n f$ ; $o$ represents the result data. Then, we quantify the failure prediction performance as following:

$$
P ^ { M _ { c e } } = \left[ \begin{array} { l l l l } { T A _ { 1 } } & { T A _ { 2 } } & { \dots } & { T A _ { k } } \\ { F A _ { 1 } } & { F A _ { 2 } } & { \dots } & { F A _ { k } } \\ { T B _ { 1 } } & { T B _ { 2 } } & { \dots } & { T B _ { k } } \\ { F B _ { 1 } } & { F B _ { 2 } } & { \dots } & { F B _ { k } } \end{array} \right] ,
$$

$$
\begin{array} { r } { T A _ { i } = n ( \hat { y } = A , y = A , r a t i n g = i ) , } \\ { F A _ { i } = n ( \hat { y } = A , y = B , r a t i n g = i ) , } \\ { T B _ { i } = n ( \hat { y } = B , y = B , r a t i n g = i ) , } \\ { F B _ { i } = n ( \hat { y } = B , y = A , r a t i n g = i ) , } \end{array}
$$

where $P ^ { M _ { c e } }$ represents the quantification of the failure prediction performance of $M _ { c e }$ , which is determined by both metacognitive ability and cognitive ability; $n ( * )$ denotes the number of data points in $O$ that satisfy condition $^ *$ .

# Step2: Decoupling Metacognition from Cognition

Cognitive Ability Quantifying We can directly isolate the cognitive performance $C P ^ { \mathbf { \bar { M } } _ { c e } }$ from $P ^ { M _ { c e } }$ :

$$
C P ^ { M _ { c e } } = [ T B _ { r a t e } , F B _ { r a t e } ] ,
$$

$$
T B _ { r a t e } = \frac { \sum _ { i = 1 } ^ { k } T B _ { i } } { \sum _ { i = 1 } ^ { k } F A _ { i } + T B _ { i } } ,
$$

$$
F B _ { r a t e } = \frac { \sum _ { i = 1 } ^ { k } F B _ { i } } { \sum _ { i = 1 } ^ { k } T A _ { i } + F B _ { i } } .
$$

However, $[ T B _ { r a t e } , F B _ { r a t e } ]$ can be replaced with $[ T A _ { r a t e } , F A _ { r a t e } ]$ , which is equivalent for calculating cognitive ability. Then we apply a LLM-SDT model $L \bar { S } ( d , c ^ { 0 } )$ to quantify the cognitive ability based on $C P ^ { M _ { c e } }$ . As mentioned in Preliminaries, $C P ^ { M _ { c e } }$ can be characterized using $L S ( d , c ^ { 0 } )$ , where $T B _ { r a t e }$ can be characterized as the area under the portion of $f ( x | B )$ in $L S ( d , c ^ { 0 } )$ that exceeds $c$ . Since the cumulative distribution function for the normal distribution with mean $\mu$ and standard deviation $\sigma$ evaluated at $x$ is:

$$
\Phi ( x , \mu , \sigma ) = \int _ { - \infty } ^ { x } \frac { 1 } { \sigma \sqrt { 2 \pi } } e ^ { \frac { - ( x - \mu ) ^ { 2 } } { 2 \sigma ^ { 2 } } } ,
$$

then $T B _ { r a t e }$ can be derived from the parameters of $T S ( d , c _ { 0 } )$ as:

$$
T B _ { r a t e } ^ { L S } = 1 - \Phi ( c ^ { 0 } , \frac { d } { 2 } ) ,
$$

and similarly:

$$
F B _ { r a t e } ^ { L S } = 1 - \Phi ( c ^ { 0 } , - { \frac { d } { 2 } } ) ,
$$

where $\sigma ~ = ~ 1$ is omitted. Therefore, the parameters of $L S ( d , c _ { 0 } )$ can be recovered from $C P ^ { M _ { c e } }$ :

$$
\hat { d } = \Phi ^ { - 1 } ( T B _ { r a t e } ) - \Phi ^ { - 1 } ( F B _ { r a t e } ) ,
$$

$$
\hat { c } ^ { 0 } = - 0 . 5 \times [ \Phi ^ { - 1 } ( T B _ { r a t e } ) + \Phi ^ { - 1 } ( F B _ { r a t e } ) ] ,
$$

where $\Phi ^ { - 1 }$ is the inverse of the normal cumulative distribution function. $\hat { d }$ is the quantification of $M _ { c e }$ ’s cognitive ability, and $\hat { c } ^ { 0 }$ represents $M _ { c e }$ ’s potential decision criteria. Both $\hat { d }$ and $\hat { c } ^ { 0 }$ will be utilized in the following metacognitive quantification.

Metacognitive Ability Quantifying Unlike cognitive performance, metacognitive performance is difficult to isolate directly from $P ^ { M _ { c e } ^ { - } }$ , making it challenging to directly quantify the metacognitive ability of $M _ { c e }$ . Therefore, our approach is as follows: We use an LLM-Meta-SDT model $\mathbf { \bar { \boldsymbol { L } } } M S ( \boldsymbol { d } , \mathbf { \boldsymbol { c } } _ { A } , \mathbf { \boldsymbol { c } } _ { B } )$ to fit $P ^ { M _ { c e } }$ , continuously updating the parameters of $L M S ( d , \pmb { c } _ { A } , \pmb { c } _ { B } )$ until it can replicate $\breve { P } ^ { M _ { c e } }$ ; As mentioned before, the parameters of $L M \bar { S } ( d , c _ { A } , c _ { B } )$ can be divided into two parts, where $d$ represents cognitive ability, and $c _ { A } , c _ { B }$ represent metacognitive ability, so we can add constraints for $c _ { A }$ and $\scriptstyle { c _ { B } }$ to ensure that the metacognitive ability is optimal, i.e., at its theoretical maximum; With this constraint, we can determine the cognitive ability required to achieve $P ^ { M _ { c e } }$ while ensuring that metacognitive ability is optimal. Then, we can measure the gap between actual and optimal metacognitive ability by comparing the gap between the required cognitive ability and the actual cognitive ability; Thus, we can use the gap between the required cognitive ability and the actual cognitive ability as the quantification of $M _ { c e }$ ’s metacognitive ability. Next, the implementation details will be presented.

• Loss Function. To facilitate the fitting, we first normalize P Mce row-wise:

$$
P _ { i j } ^ { M _ { c e } } = \frac { P _ { i j } ^ { M _ { c e } } } { \sum _ { j = 1 } ^ { k } P _ { i j } ^ { M _ { c e } } } .
$$

Then, we apply $L M S ( d , \pmb { c } _ { A } , \pmb { c } _ { B } )$ to represent the estimate of $P ^ { M _ { c e } ^ { - } }$ :

$$
\widehat { P } ^ { M _ { c e } } = \left[ \begin{array} { c c c c } { \widehat { T A } _ { 1 } } & { \widehat { T A } _ { 2 } } & { . . . } & { \widehat { T A } _ { k } } \\ { \widehat { F A } _ { 1 } } & { \widehat { F A } _ { 2 } } & { . . . } & { \widehat { F A } _ { k } } \\ { \widehat { T B } _ { 1 } } & { \widehat { T B } _ { 2 } } & { . . . } & { \widehat { T B } _ { k } } \\ { \widehat { F B } _ { 1 } } & { \widehat { F B } _ { 2 } } & { . . . } & { \widehat { F B } _ { k } } \end{array} \right] ,
$$

where $\widehat { T A } _ { i }$ is the estimate of $T A _ { i }$ , representing the probability thdat $M _ { c e }$ reports a confidence of $i$ when correctly answering a binary-choice problem with a ground truth of $A$ . In $\mathsf { \bar { L } } M S ( d , \pmb { c } _ { A } , \pmb { c } _ { B } )$ , the probability of correctly answering a binary-choice problem with a ground truth of $A$ is represented by the area under $f ( x | A )$ to the left of $c ^ { 0 }$ , while the probability of assigning a confidence rating of $i$ is represented by the area under $f ( x | A )$ between $c _ { A } ^ { i - 1 }$ and $c _ { A } ^ { i }$ . So $\widehat { T A } _ { i }$ can be computed as:

$$
\begin{array} { r } { \widehat { T A } _ { i } = \left\{ \begin{array} { l l } { \frac { \Phi ( c _ { A } ^ { i - 1 } , - \frac { d } { 2 } ) - \Phi ( c _ { A } ^ { i } , - \frac { d } { 2 } ) } { \Phi ( c _ { 0 } , - \frac { d } { 2 } ) } } & { i \in [ 1 , k - 1 ] } \\ { \frac { \Phi ( c _ { A } ^ { i - 1 } , - \frac { d } { 2 } ) } { \Phi ( c _ { 0 } , - \frac { d } { 2 } ) } } & { i = k } \end{array} \right. . } \end{array}
$$

${ \widehat { F A } } _ { i } , { \widehat { T B } } _ { i }$ and $\widehat { F B } _ { i }$ are computed in the same way. To mdake $\widehat { P } ^ { M _ { c e } }$ as cldose as possible to $P ^ { M _ { c e } }$ , we use the frobe bus norm of their difference as the loss function:

$$
\mathcal { L } = \| \widehat { P } ^ { M _ { c e } } - P ^ { M _ { c e } } \| _ { F } ^ { 2 } .
$$

• Optimal Metacognition Constraint. In order to ensure that $L M S ( d , \pmb { c } _ { A } , \pmb { c } _ { B } )$ has optimal metacognitive capability in fitting $P ^ { M _ { c e } }$ , we design the optimal metacognition constraint based on the following idea: if the metacognitive ability of $M _ { c e }$ is optimal, then its expressed confidence ratings should be strictly positively correlated with the accuracy of its answers. Consider the case where $M _ { c e }$ ’s answer is A: When the confidence rating is $i$ , the probabilities of answering correctly and incorrectly can be represented in $L M \bar { S ( d , } \mathbf { c } _ { A } , \mathbf { \bar { } } \mathbf { c } _ { B } )$ as the area under the portion of $f ( x | A )$ and $f ( x | B )$ falling in $c _ { i - 1 }$ and $c _ { i }$ , respectively; The accuracy is represented as the proportion of the probability of answering correctly. Therefore, the accuracy when $M _ { c e }$ ’s answer is $A$ and the confidence rating is $i$ can be expressed as:

<html><body><table><tr><td>C.E.M</td><td>AUROC</td><td>ECE</td><td>BS</td><td>DMC</td></tr><tr><td>Verb-Vanilla</td><td>0.7263</td><td>0.6253</td><td>0.3992</td><td>0.0092</td></tr><tr><td>Verb-Cot</td><td>0.4638</td><td>0.6216</td><td>0.3978</td><td>0.0088</td></tr><tr><td>Verb-Topk</td><td>0.6376</td><td>0.6999</td><td>0.2609</td><td>0.0064</td></tr><tr><td>Self Random</td><td>0.3868</td><td>0.4926</td><td>0.3741</td><td>0.0063</td></tr><tr><td>Perturbing</td><td>0.3700</td><td>0.4463</td><td>0.3766</td><td>0.0052</td></tr><tr><td>Misleading</td><td>0.2931</td><td>0.4580</td><td>0.3871</td><td>0.0057</td></tr></table></body></html>

Table 1: Average CV comparison between DMC and baseline models with corresponding confidence elicitation approaches across all datasets

$$
a c c _ { A } ^ { i } = \frac { \widehat { T A } _ { i } \sum _ { i = 1 } ^ { k } \widehat { T A } _ { i } } { \widehat { T A } _ { i } \sum _ { i = 1 } ^ { k } \widehat { T A } _ { i } + \widehat { F A } _ { i } \sum _ { i = 1 } ^ { k } \widehat { F A } _ { i } } .
$$

Similarly, when $M _ { c e }$ ’s answer is $B$ :

$$
a c c _ { B } ^ { i } = \frac { \widehat { T B } _ { i } \sum _ { i = 1 } ^ { k } \widehat { T B } _ { i } } { \widehat { T B } _ { i } \sum _ { i = 1 } ^ { k } \widehat { T B } _ { i } + \widehat { F B } _ { i } \sum _ { i = 1 } ^ { k } \widehat { F B } _ { i } } .
$$

Then, the optimal metacognition constrain can be represented as:

$$
\mathcal { C } _ { o p t } = \left\{ \begin{array} { l l } { a c c _ { A } ^ { i + 1 } > a c c _ { A } ^ { i } } & { i \in [ 1 , k - 1 ] } \\ { a c c _ { B } ^ { i + 1 } > a c c _ { B } ^ { i } } & { i \in [ 1 , k - 1 ] } \end{array} \right.
$$

• $L M S ( d , \pmb { c } _ { A } , \pmb { c } _ { B } )$ i, twheirne $d \ = \ \hat { d } , \ c _ { A } ^ { 0 } \ = \ c _ { B } ^ { 0 } \ = \ \hat { c } ^ { 0 } ;$ f; The initial values for c1A, ..., ckA− and c1B, ..., ckB− are obtained by sequentially treating them as the decision axis $c ^ { 0 }$ of $L S \dot { ( d , c ^ { 0 } ) }$ and the computing them using equation(16). Notably, $c ^ { 0 }$ represents the extent to which $M _ { c e }$ tends to favor option $A$ or $B$ when answering two-choices problems, reflecting cognitive bias, which is not the aspect we are concerned with. However, to ensure that the cognitive abilities computed by $L M S ( d , \pmb { c } _ { A } , \pmb { c } _ { B } )$ are comparable to $\hat { d }$ , we maintain $c _ { A } ^ { 0 } = { \dot { c } } _ { B } ^ { 0 } = { \hat { c } } ^ { 0 }$ throughout the fitting process:

$$
\mathcal { C } _ { c o n s t } = c _ { A } ^ { 0 } = c _ { B } ^ { 0 } = \hat { c } ^ { 0 } .
$$

Finally, we can summarize the fitting process as the following mathematical optimization problem:

$$
\theta ^ { * } = \arg \operatorname* { m a x } _ { \theta } \mathcal { L } , \quad s u b j e c t \ t o : \ \mathcal { C } _ { o p t } , \mathcal { C } _ { c o n s t } ,
$$

$$
\displaystyle \theta = ( d , \pmb { c } _ { A } , \pmb { c } _ { B } ) ,
$$

where $\theta ^ { * }$ denotes the optimal solution of theta. $d ^ { * } \in \theta ^ { * }$ represents the cognitive ability required to achieve P Mce, under the assumption of optimal metacognition. Since both $d ^ { * }$ and $\hat { d }$ are measured in signal-to-noise ratio units, we can use their ratio to quantify metacognitive ability:

$$
\mathcal { M } \mathcal { C } ^ { M _ { c e } } = d ^ { * } / \hat { d } .
$$

# Experiment

We aim to answer the following research questions (RQs):

• RQ1: Whether DMC can effectively decouple metacognitive ability from cognitive ability in LLMs?   
• RQ2: How different confidence elicitation methods impact the quantification of metacognitive ability in LLMs?   
• RQ3: What variations exist in metacognitive ability across different LLMs?   
• RQ4: Whether there is consistency between the DMC metacognition quantification and the performance levels on the AbstainQA task?

# Experiment Setup

Datasets. We evaluate our DMC framework on eight datasets across five types of tasks: 1) Mathematical Reasoning on SAT Math (SAT) from AGIEval (Zhong et al. 2024) and High School Mathematics (HSM) from MMLU (Hendrycks et al. 2020); 2) Commonsense Reasoning on CommonsenseQA (CQA) (Talmor et al. 2019) and Global Facts (GFacts) from MMLU (Hendrycks et al. 2020); 3) Symbolic Understanding on Boolean Expressions (Bool) and Date Understanding (Date) from Big-Bench-Hard (Suzgun et al. 2023); 4) Professional Knowledge on Professional Medicine (Med) from MMLU (Hendrycks et al. 2020); 5) Ethical Knowledge on Business Ethics (Ethics) from MMLU (Hendrycks et al. 2020).

Datasets Processing. To satisfy the task setup of the proposed DMC, we convert these multiple-choice questions into binary choice questions by pairing the correct option with each incorrect option to create new questions. Additionally, to eliminate biases introduced by the order of options, we also create new questions by swapping the order of options for each binary choice problem.

Compared LLMs. We utilize three popularly-used LLMs: LLaMA2-70B (Touvron et al. 2023), GPT-3.5 (OpenAI 2021), and GPT-4 (Achiam et al. 2023). The default sampling temperature is set to 0.1, while 0.7 is employed when multiple runs are necessary.

Confidence Elicitation Methods (C.E.M). We use two popular types of black-box LLM confidence elicitation strategies to obtain confidence from LLMs: 1) Verbalized including Vanilla, Cot and Top- $\mathbf { \nabla } \cdot k$ (Tian et al. 2023); 2) Consistency-based including Self Random, Perturbing and Misleading (Xiong et al. 2024).

# Results and Analysis of RQ1

To validate the effectiveness of DMC in decoupling metacognitive ability and cognitive ability, we compared the variability of DMC with calibration-based methods, including expected calibration error (ECE) (Guo et al. 2017), area under the receiver operating characteristic curve (AUROC) (Xiong et al. 2024), and Brier Score (BS) (Brier 1950), across different datasets using GPT-3.5. This comparison was conducted across all eight datasets and six confidence elicitation methods to ensure robustness. To standardize the comparison and eliminate the units of different quantification methods, the coefficient of variation (CV) (Abdi 2010) was utilized to measure variability. The results, illustrated in Table 1, indicate that the variability of DMC’s metacognitive ability quantification results across diverse datasets is notably lower than that of calibration-based methods. This observation emphasizes the efficacy of our proposed DMC framework in disentangling metacognitive and cognitive abilities in LLMs, i.e., metacognitive ability quantification does not exhibit significant fluctuations across different tasks, a challenge that calibration-based methods often struggle to address.

# Results and Analysis of RQ2

To examine the impact of confidence elicitation methods on metacognitive ability, we employed DMC to quantify GPT-3.5’s metacognitive performance across eight datasets and six confidence elicitation methods, as outlined in Table 2. The last column displays the average metacognitive quantification ability of each confidence elicitation method across all datasets. It can be seen that variations in metacognitive outcomes stem from the differing effectiveness of these methods. In verbalized confidence, Verb-Vanilla exhibited the weakest performance due to overconfidence. Conversely, Verb-Cot and Verb-Topk enhanced metacognitive ability through chain-of-thought reasoning (Wei et al. 2022) and prompting candidate answers from the LLM, respectively. More specifically, Verb-Topk demonstrated notable improvement, likely due to its effective mitigation of initial overconfidence issues. Despite these advancements, VerbCot and Verb-Topk still grapple with overconfidence, resulting in lower metacognitive ability compared to consistencybased strategies. In consistency methods, Perturbing and Misleading achieve metacognitive ability slightly lower than that of Self-Random, possibly due to the introduction of irrelevant reasoning paths by additional perturbations. What’s more, by defining metacognitive ability as the capacity to evaluate response correctness through confidence, our DMC provides a task-agnostic approach to assess the effectiveness of confidence elicitation methods.

# Results and Analysis of RQ3

As shown in Table 3, we selected three representative confidence elicitation methods to assess the metacognitive abilities quantified by the DMC framework across different large models (including LLaMA2, GPT-3.5, and GPT-4) on all datasets. Experimental results indicate that regardless of the confidence elicitation method used, GPT-4 outperforms GPT-3.5, which in turn outperforms LLaMA2 (with average metacognitive abilities per LLM across three C.E.Ms being 0.5491 vs. 0.4835 vs. 0.3210, respectively). Therefore, it is apparent that the performance of large models is positively linked to their displayed metacognitive ability. This connection may be attributed to the complexity of the models, with more intricate structures and parameters enabling a better capture of language complexity and contextual information, leading to enhanced metacognitive performance. Additionally, it could be due to the advanced models’ inclination to mimic human language understanding and expression, including confidence representation. Simultaneously, different LLMs exhibit varying sensitivities to confidence elicitation methods, potentially resulting in diverse metacognitive performances. As depicted in Table 3, LLaMA2 demonstrates limited sensitivity to different C.E.Ms $\scriptstyle { \mathrm { ( m a x } = 0 . 3 7 9 7 }$ , mi $_ { 1 = 0 . 2 5 0 5 }$ , st.d. $_ { - 0 . 0 5 3 4 }$ ). Conversely, GPT-4 shows larger variations in metacognitive abilities when employing different C.E.Ms (max $= 0 . 7 2 2 6$ , mi $_ { 1 = 0 . 3 6 3 6 }$ , st.d. $= 0 . 1 4 6 8 \$ ).

Table 2: Comparison of metacognitive ability quantified using the DMC Framework with different confidence elicitation meth ods across all datasets.   

<html><body><table><tr><td>C.E.M</td><td>SAT</td><td>HSM</td><td>CQA</td><td>GFacts</td><td>Bool</td><td>Date</td><td>Med</td><td>Ethics</td><td>Average</td></tr><tr><td>Verb-Vanilla</td><td>0.3192</td><td>0.3276</td><td>0.3203</td><td>0.3215</td><td>0.3188</td><td>0.3236</td><td>0.3271</td><td>0.3227</td><td>0.3225</td></tr><tr><td>Verb-Cot</td><td>0.3739</td><td>0.3770</td><td>0.3785</td><td>0.3717</td><td>0.3741</td><td>0.3769</td><td>0.3804</td><td>0.3820</td><td>0.3768</td></tr><tr><td>Verb-Topk</td><td>0.5288</td><td>0.5325</td><td>0.5344</td><td>0.5278</td><td>0.5312</td><td>0.5273</td><td>0.5309</td><td>0.5225</td><td>0.5294</td></tr><tr><td>Self Random</td><td>0.6059</td><td>0.5976</td><td>0.5938</td><td>0.5981</td><td>0.6027</td><td>0.5944</td><td>0.5955</td><td>0.5956</td><td>0.5986</td></tr><tr><td>Perturbing</td><td>0.5599</td><td>0.5596</td><td>0.5585</td><td>0.5641</td><td>0.5615</td><td>0.5633</td><td>0.5584</td><td>0.5542</td><td>0.5599</td></tr><tr><td>Misleading</td><td>0.5600</td><td>0.5652</td><td>0.5619</td><td>0.5611</td><td>0.5547</td><td>0.5573</td><td>0.5585</td><td>0.5634</td><td>0.5606</td></tr></table></body></html>

<html><body><table><tr><td>Model</td><td>C.E.M</td><td>SAT</td><td>HSM</td><td>CQA</td><td>GFacts</td><td>Bool</td><td>Date</td><td>Med</td><td>Ethics</td><td>Average</td></tr><tr><td rowspan="3">LLaMA2</td><td>Verb-Vanilla</td><td>0.2485</td><td>0.2498</td><td>0.2476</td><td>0.2507</td><td>0.2525</td><td>0.2557</td><td>0.2448</td><td>0.2546</td><td>0.2505</td></tr><tr><td>Verb-Topk</td><td>0.3293</td><td>0.3359</td><td>0.3348</td><td>0.3273</td><td>0.3380</td><td>0.3276</td><td>0.3365</td><td>0.3347</td><td>0.3330</td></tr><tr><td>Self Random</td><td>0.3832</td><td>0.3788</td><td>0.3817</td><td>0.3837</td><td>0.3779</td><td>0.3753</td><td>0.3765</td><td>0.3801</td><td>0.3797</td></tr><tr><td rowspan="3">GPT-3.5</td><td>Verb-Vanilla</td><td>0.3192</td><td>0.3276</td><td>0.3203</td><td>0.3215</td><td>0.3188</td><td>0.3236</td><td>0.3271</td><td>0.3227</td><td>0.3225</td></tr><tr><td>Verb-Topk</td><td>0.5288</td><td>0.5325</td><td>0.5344</td><td>0.5278</td><td>0.5312</td><td>0.5273</td><td>0.5309</td><td>0.5225</td><td>0.5294</td></tr><tr><td>Self Random</td><td>0.6059</td><td>0.5976</td><td>0.5938</td><td>0.5981</td><td>0.6027</td><td>0.5994</td><td>0.5955</td><td>0.5956</td><td>0.5986</td></tr><tr><td rowspan="3">GPT-4</td><td>Verb-Vanilla</td><td>0.3576</td><td>0.3590</td><td>0.3655</td><td>0.3663</td><td>0.3679</td><td>0.3653</td><td>0.3614</td><td>0.3659</td><td>0.3636</td></tr><tr><td>Verb-Topk</td><td>0.5568</td><td>0.5632</td><td>0.5593</td><td>0.5641</td><td>0.5626</td><td>0.5618</td><td>0.5646</td><td>0.5572</td><td>0.5612</td></tr><tr><td>Self Random</td><td>0.7195</td><td>0.7205</td><td>0.7227</td><td>0.7273</td><td>0.7184</td><td>0.7267</td><td>0.7244</td><td>0.7215</td><td>0.7226</td></tr></table></body></html>

Table 3: Comparsion of metacognitive ability quantified using the DMC Framework across various LLMs.

# Results and Analysis of RQ4

We utilized the DMC framework to evaluate hallucination mitigation in LLMs via the AbstainQA task (Feng et al. 2024). Specifically, Reliable Accuracy (R-Acc) and Abstain Accuracy (A-Acc) were employed as metrics to measure task performance, with higher values indicating better performance. In Figure 4, the x-axis illustrates the combinations of three LLMs and three C.E.Ms, with different colors (orange, blue, and green) representing the performance of each combination in metacognition, R-Acc, and A-Acc on the yaxis. The three distinct colored lines exhibit similar trends, suggesting that higher metacognitive ability correspond to better performance in the AbstainQA task, underscoring the potential of enhancing metacognition in LLMs to alleviate hallucination issues.

# Conclusion

In this paper, we introduce a novel general framework for quantifying metacognitive ability in large language models, named DMC. Through comprehensive experiments and analyses, we have observed that DMC successfully separates metacognitive from cognitive capabilities, enabling a more precise quantification of metacognitive ability. In addition, the quantification of metacognitive ability is influenced by the choice of confidence elicitation methods and varies across different large language models. Moreover, improving the metacognitive ability of LLMs shows potential in mitigating hallucination issues.

![](images/394d819d60a25dbfac15f5b1f112dcf57086e90883676f1fe55f07009e67d493.jpg)  
Figure 4: LLMs’ metacognitive ability and their performance on the AbstainQA task.