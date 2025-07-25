# Utilize the Flow before Stepping into the Same River Twice: Certainty Represented Knowledge Flow for Refusal-Aware Instruction Tuning

Runchuan $\mathbf { Z } \mathbf { h } \mathbf { u } ^ { 1 2 * }$ , Zhipeng $\mathbf { M } \mathbf { a } ^ { 3 * }$ , Jiang $\mathbf { W _ { u } } ^ { 1 * \dagger }$ , Junyuan Gao14, Jiaqi Wang1, Dahua $\mathbf { L i n } ^ { 1 }$ , Conghui $\mathbf { H e } ^ { 1 \ddag }$

1Shanghai Artificial Intelligence Laboratory 2Peking University 3Southwest Jiaotong University 4University of Chinese Academy of Sciences {zhurunchuan, wujiang, gaojuanyuan, wangjiaqi, lindahua, heconghui}@pjlab.org.cn, mazhipeng $1 0 2 4 @$ my.swjtu.edu.cn

# Abstract

Refusal-Aware Instruction Tuning (RAIT) enables Large Language Models (LLMs) to refuse to answer unknown questions. By modifying responses of unknown questions in the training data to refusal responses such as “I don’t know”, RAIT enhances the reliability of LLMs and reduces their hallucination. Generally, RAIT modifies training samples based on the correctness of the initial LLM’s response. However, this crude approach can cause LLMs to excessively refuse answering questions they could have correctly answered, the problem we call over-refusal. In this paper, we explore two primary causes of over-refusal: Static conflict occurs when similar samples within the LLM’s feature space receive differing supervision signals (original vs. modified “I don’t know”). Dynamic conflict arises as the LLM’s evolving knowledge during SFT enables it to answer previously unanswerable questions, but the now-answerable training samples still retain the original “I don’t know” supervision signals from the initial LLM state, leading to inconsistencies. These conflicts cause the trained LLM to misclassify known questions as unknown, resulting in over-refusal. To address this issue, we introduce Certainty Represented Knowledge Flow for Refusal-Aware Instructions Tuning (CRaFT). CRaFT centers on two main contributions: First, we additionally incorporate response certainty to selectively filter and modify data, reducing static conflicts. Second, we implement preliminary rehearsal training to characterize changes in the LLM’s knowledge state, which helps mitigate dynamic conflicts during the fine-tuning process. We conducted extensive experiments on open-ended question answering and multiple-choice question task. Experiment results show that CRaFT can improve LLM’s overall performance during the RAIT process.

Code & Data — https://github.com/opendatalab/CRaFT Extended version — https://arxiv.org/abs/2410.06913

Case of Over-Refusal Initial Model Correct Wrong In which city would A RAIT BReAfIoTre yOo'uHfairnedICnhtiecrangaotional Airport? Previous Method Correct Refusal Wrong Chicago. (Correct) OveRre-dRuecfeusal RAfAtIeTr yIonuwfhiincdhCcihticyawgould 串 Ours(CRaFT) Correct Refusal Wrong AOi'rHpaoret?International I don’t know. (Refusal)

# 1 Introduction

Recently, Large Language Model (LLM) technology has made significant progress, becoming an important milestone towards AGI (Achiam et al. 2023; Dubey et al. 2024; Touvron et al. 2023; Yang et al. 2024). However, current LLMs often output fabricated information, which is referred to as hallucinations (Ji et al. 2023). This phenomenon severely limits the usefulness and reliability of LLMs. An important reason for the occurrence of hallucinations is that when exposed to questions beyond their internal knowledge (i.e., unknown questions), LLMs may forcefully generate responses (Kang et al. 2024). Ideally, the reliable LLM should actively refuse to answer questions it doesn’t know to avoid incorrect responses (Wen et al. 2024; Li et al. 2024b). Recent studies have shown (Yang et al. 2023; Zhang et al. 2024; Xu et al. 2024a; Cheng et al. 2024; Xu et al. $2 0 2 4 \mathrm { a }$ ; Bai et al. 2023; Cheng et al. 2024) that Refusal-Aware Instruction Tuning (RAIT) can enable LLMs to refuse answering questions beyond their knowledge.

The RAIT process can be described as follows: the initial LLM answers all questions in the train set $D _ { \mathrm { s r c } }$ . Based on response accuracy, samples are split into two groups. Correct responses are labeled as vanilla samples $D _ { \mathrm { v a n } }$ , with unchanged answers, while incorrect responses are replaced with ”I don’t know,” forming the IdK samples $D _ { \mathrm { i d k } }$ . The combined RAIT data $D _ { \mathrm { r a i t } } = D _ { \mathrm { v a n } } \cup D _ { \mathrm { i d k } }$ is used to fine-tune the LLM, improving its ability to refuse unknown questions.

This original RAIT method is referred to as Cor-RAIT.

However, (Cheng et al. 2024) shows that Cor-RAIT causes the fine-tuned LLM to refuse some questions that could have been answered correctly. Experiments reveal a significant accuracy drop after Cor-RAIT: on the TriviaQA dataset (Joshi et al. 2017), accuracy falls from $4 5 . 0 5 \%$ to $2 8 . 5 7 \%$ , and on the Natural Questions dataset (Kwiatkowski et al. 2019), it drops from $2 4 . 6 5 \%$ to $1 5 . 9 3 \%$ . We refer to this phenomenon as over-refusal, as shown in Figure 1.

In addressing the over-refusal brought by Cor-RAIT, we identified two primary causes as shown in Figure 2. (1) Static Conflict: In the LLM representation space, two closely located samples might be assigned to $D _ { \mathrm { v a n } }$ and $D _ { \mathrm { i d k } }$ under the Cor-RAIT framework. As illustrated by t-SNE in Figure 3(a), significant intersections exist between $D _ { \mathrm { v a n } }$ and $D _ { \mathrm { i d k } }$ , complicating their differentiation. These similar samples provide conflict supervision during training, impairing the LLM’s ability to distinguish between known and unknown questions, resulting in over-refusal. (2) Dynamic Conflict: This arises from overlooking the dynamic shifts in LLM’s knowledge state during training. Research (Ren et al. 2024; Ren and Sutherland 2024; Xu et al. 2024b) shows that the knowledge state of LLMs changes during Supervised Fine-Tuning (SFT), with questions potentially shifting from unknown to known and vice versa. This phenomenon is reminiscent of Heraclitus’ saying, “no man ever steps in the same river twice.” However, current methods use static RAIT data reflecting the initial LLM’s knowledge state throughout SFT, which ignores these changes. This oversight leads to conflicts between the RAIT data and the LLM’s evolving knowledge, resulting inefficient training and over-refusal.

To address the two problems above, we propose Certainty Represented Knowledge Flow for Refusal-Aware Instructions Construction (CRaFT). Our approach consists of two stages. Stage 1: Querying the Knowledge State and Flow of the LLM. First, we probe the initial LLM’s knowledge state. Unlike Cor-RAIT, we incorporate response certainty alongside correctness, effectively alleviating the static conflict between the supervision signals in $D _ { \mathrm { v a n } }$ and $D _ { \mathrm { i d k } }$ . To capture the LLM’s dynamic knowledge changes during training, we introduce a rehearsal training mechanism. This fine-tunes the LLM with data samples that align closely with its internal knowledge, without introducing new knowledge (Ren et al. 2024; Kang et al. 2024). This approach allows us to observe the LLM’s natural knowledge adjustments. The differences between the fine-tuned and initial LLMs reveal the knowledge flow during training, helping to identify and resolve dynamic conflicts. Stage 2: Refusal-aware instructions construction and tuning. By considering both the static knowledge state and dynamic knowledge flow, we filter out vanilla and IdK samples from RAIT data, reducing conflicts. We then fine-tune the initial LLM with the refined data, improving overall performance.

In conducting our experimental analysis, we sought a well-founded metric within current research. Existing methods have notable limitations, either proposing multiple metrics that are hard to optimize simultaneously or relying on inherently flawed metrics, as demonstrated by our counterexamples. Consequently, we examined these shortcomings and introduced a singular and comprehensive metric: Truthful Helpfulness Score(THS).

Overall, our main contributions are as follows:

• We conducted the in-depth analysis of static and dynamic conflicts in existing correctness-based RAIT data, revealing that they cause the trained LLMs’ mis-classification of known and unknown questions, leading to the issue of over-refusal in current RAIT methods. • To address static and dynamic conflicts, we introduce CRaFT: it reduces static conflicts by incorporating certainty alongside correctness during RAIT data construction, and mitigates dynamic conflicts through rehearsal training to capture knowledge flow trends. Extensive experiments demonstrate that CRaFT alleviates overrefusal and improves overall LLM performance. • We analyze the shortcomings of existing refusal-aware metrics and introduce the Truthful Helpfulness Score (THS), which balances reliability and helpfulness for a comprehensive evaluation of LLM performance.

# 2 Related Work

# 2.1 Mitigating Hallucinations of LLMs

Researchers have developed various methods to mitigate LLM hallucinations, including data augmentation (Neeman et al. 2022), improved decoding strategies (Holtzman et al. 2019; Chuang et al. 2023), external knowledge integration (Karpukhin et al. 2020), knowledge editing (Zhang, Yu, and Feng 2024; Li et al. 2024a), and honesty alignment (Zhang et al. 2024; Xu et al. 2024a; Bai et al. 2024). Unlike traditional correction methods, honesty alignment encourages models to say “I don’t know” for unknown questions.

# 2.2 Refusal-Aware Instruction Tuning

RAIT is the supervised technique that improves LLMs’ responses by training LLMs to directly respond with “I don’t know” to unknown questions. R-Tuning (Zhang et al. 2024) identifies these questions by having the LLM answer each once and verifying response accuracy. In (Yang et al. 2023), the LLM answers the same question multiple times, with the target answer adjusted based on the correctness ratio. (Wan et al. 2024) uses a knowledge-based verification mechanism to ensure consistency with trusted external sources, enhancing refusal accuracy and preventing misinformation.

# 3 Over-Refusal: Analysis and Insights 3.1 Refusal-Aware Instruction Tuning

Given the initial LLM $\mathcal { M } _ { 0 }$ and the instruction dataset $D _ { \mathrm { s r c } }$ of question-answer pairs $x = ( q , a )$ , we modify $D _ { \mathrm { s r c } }$ to construct $D _ { \mathrm { r a i t } }$ , consisting of pairs $( q , a _ { \mathrm { r a i t } } )$ . $D _ { \mathrm { r a i t } }$ is then used for SFT on $\mathcal { M } _ { 0 }$ , resulting in a new LLM capable of declining unknown questions, a process called Refusal-Aware Instruction Tuning (RAIT). Existing studies (Zhang et al. 2024; Yang et al. 2023; Cheng et al. 2024) use $\mathcal { M } _ { 0 }$ to infer and assess the correctness of questions in $D _ { \mathrm { s r c } }$ , denoted as $\mu$ . As shown in Figure 4(a), the correctness threshold $\tau _ { \mu }$ is first defined. If $\mu < \tau _ { \mu }$ , the answer is changed to $^ { \ ' } \mathrm { I }$ don’t know”

Q1 find Chicago O’Hare find Chicago Rockford Q2 1 How many ScQoutetsistiholneague teams OArnisgwinearl 1 names end in United? 3. Question Original Answer Original Answer Question Q1 Chicago. Rockford. Q2 Initia Evolving 里 里 Model RAITTrainingProces Model   
Representation Similar Representation 1   
... Representations ... 1 虫 虫   
PrCehidcicatgio.n CSotnafltict PrCehidcicatgio.n P1.r(eWdircotniog)n KnowledgeFlow 3P.r(Ceodircrteicotn) Match Unmatch Modify 1 Different Targets Answer Target Answer Chicago. Target Answer I don’t know. L Target Answer I don’t know. Modifiedbasedon fIixnietdialduMriondgelRaAnIdT Dynamic Conflict 1 (a) Static Conflict (b) Dynamic Conflict

![](images/f6872740a5e17fb5d1bcba883f9b32cd5498dfbf3128f4f430f732fd8210a538.jpg)  
Figure 2: Two causes of over-refusal: (a) Static conflict means the similar samples in the LLM’s feature space being assigned different labels (original vs. modified “I don’t know”). (b) Dynamic conflict arises since the LLM’s knowledge state evolves during SFT, turning initally unknown questions to knowns, while the target answer remains IdK. These conflicts cause the trained LLM to misclassify known questions as unknown, resulting in over-refusal.   
Figure 3: t-SNE visualization of the LLM feature space

and assigned to the IdK subset $D _ { \mathrm { i d k } }$ . If $\mu \geq \tau _ { \mu }$ , the original answer remains, and the pair is assigned to the vanilla subset $D _ { \mathrm { v a n } }$ . The resulting RAIT data is $D _ { \mathrm { r a i t } } = D _ { \mathrm { v a n } } \cup D _ { \mathrm { i d k } }$ , and this correctness-based RAIT is called Cor-RAIT.

However, LLMs exhibited significant over-refusal after Cor-RAIT, as shown in Figure 1. Subsequent sections of this chapter will analyze the causes and offer practical insights.

# 3.2 Static Conflicts in Cor-RAIT

During the Cor-RAIT process, LLMs learn to reject unknown samples by supervisions from $D _ { \mathrm { i d k } }$ . Our insight is that if the Cor-RAIT dataset contains vanilla and IdK samples that are closely positioned in the LLM’s representation space, the trained LLM may mistakenly classify similar vanilla samples as IdK samples, causing over-refusals. To verify this, we analyzed the sample distributions of $D _ { \mathrm { v a n } }$ and $D _ { \mathrm { i d k } }$ . We extract latent representation of each question from the last hidden layer of the LLM. Then, t-SNE is adopted to visualize sample representations. Figure 3(a) displays the distributions of samples from the test split of MMLU dataset (Hendrycks et al. 2020) in the LLaMA-3- 8B-instruct (Dubey et al. 2024) representation space, where IdK and vanilla samples have significant intersections. 1

Furthermore, we introduce the Conflict Rate for Similar Samples (CRSS) to quantitatively assess conflicts in supervision signals among similar samples in the RAIT dataset. For each sample $x _ { i }$ in $D _ { \mathrm { i d k } }$ , we compute the cosine similarity between its question representation $\boldsymbol { r } _ { i }$ and the question representation $r _ { j }$ of each sample $x _ { j }$ in $D _ { \mathrm { v a n } }$ . We identify and record the highest similarity value obtained. If this value exceeds the predefined similarity threshold $\tau _ { \mathrm { s i m } }$ , we record the occurrence. The CRSS is then calculated as:

$$
\mathrm { C R S S } = \frac { \sum _ { x _ { i } \in { D _ { \mathrm { i d k } } } } { \mathbf { 1 } \left( \operatorname* { m a x } _ { x _ { j } \in { D _ { \mathrm { v a n } } } } \cos ( r _ { i } , r _ { j } ) > \tau _ { \mathrm { s i m } } \right) } } { \left| { { D _ { \mathrm { i d k } } } } \right| }
$$

Therefore, the higher CRSS indicates more conflicting similar sample pairs, potentially leading to over-refusal. We computed the CRSS for Cor-RAIT, as shown in Figure 5. The results show that at $\tau _ { \mathrm { s i m } } = 0 . 9 7$ , CRSS reaches significant levels across various LLM and dataset combinations, supporting earlier t-SNE findings2.

The above analysis reveals that Cor-RAIT generates numerous similar sample pairs between $D _ { \mathrm { v a n } }$ and $D _ { \mathrm { i d k } }$ , resulting in conflicting supervision signals which leads to overrefusal. We term this static conflict to distinguish it from another conflict type discussed later.

# 3.3 Certainty Mitigates the Static Conflicts

We conducted a theoretical analysis 3 establishing a weak (non-differentiable) link between the LLM’s feature and the response correctness $\mu$ for the specific question $q$ . This weak correlation causes highly similar samples being categorized into $D _ { \mathrm { v a n } }$ and $D _ { \mathrm { i d k } }$ respectively. To mitigate this, we propose incorporating a robust indicator variable aligned with correctness to select and construct the RAIT data. This variable should ensure that similar samples share comparable values, reducing the above misclassification. We suggest adopting the certainty (Jiang et al. 2023) of the LLM’s response as the indicator variable. Our theoretical analysis shows that certainty meets the above requirements.4

![](images/2738fc6b036e762a26c4ecc8b4afb889a4be5bc029bba991f58249a6611bc8a4.jpg)  
Figure 4: RAIT Data Construction of Cor-RAIT, CorCerRAIT and CRaFT. Cor-RAIT partitions data based on accuracy $\mu$ and a threshold $\tau _ { \mu }$ . For CorCer-RAIT, $D _ { \mathrm { v a n } }$ is derived from samples with accuracy exceeding the threshold and the highest certainty, while $D _ { \mathrm { i d k } }$ consists of samples with accuracy below the threshold and the lowest certainty. CRaFT employs a two-stage process: in the first stage, data with $\Delta \mu > 0$ is excluded through knowledge queries; the second stage follows the same procedure as CorCer-RAIT.

To incorporate correctness and certainty into the RAIT data selection, we developed the CorCer-RAIT framework, as shown in Figure 4(b). We visualized the sample distribution in $D _ { \mathrm { v a n } }$ and $D _ { \mathrm { i d k } }$ using t-SNE in the LLM representation space, as shown in Figure 3(b), which shows a significant decrease in the overlap between $D _ { \mathrm { v a n } }$ and $D _ { \mathrm { i d k } }$ compared to the Cor-RAIT in Figure 3(a). Furthermore, we calculated the CRSS for both methods, as shown in Figure 5, highlighting substantial reductions in CorCer-RAIT over Cor-RAIT. Therefore, the joint use of correctness and certainty effectively alleviates the static conflict between the supervision signals in $D _ { \mathrm { v a n } }$ and $D _ { \mathrm { i d k } }$ .

![](images/aaa65b3f9a3b19316f59e32229944cd54adc394d499fd10ab6c3ee69a20b6677.jpg)  
Figure 5: CRSS of Different RAIT Samples.   
Figure 6: The Framework of CRaFT: Stage 1 queries knowledge state and flow, while Stage 2 constructs RAI data and tunes.

Stage1: Query the Knowledge State and Flow of LLM Data Pool LLM Knowledge Data Pool State Rehearsal 0 O LLMTraining KgneoFwlloewd witAhssKoncoiawtledg e State and Flow Stage2: Refusal-Aware Instructions   
LLM Initial Construction & Tuning   
LLM training DPoatola TSwaomSptlep RAIT Data LLM LLM

# 3.4 Knowledge Flow and Dynamic Conflict

Research (Ren et al. 2024; Gekhman et al. 2024; Ren and Sutherland 2024) reveals that the knowledge state of LLMs evolves during the SFT process. The phenomenon, which we refer to as “knowledge flow”, can cause previously incorrectly answered questions to become correct ones during SFT. Despite this dynamic evolutions, the target answer of the training data remains static during the RAIt, which reflects the knowledge state of the initial LLM but ignores subsequent changes. We term it as dynamic conflict, which significantly contributes to the over-refusal in Cor-RAIT.

We select the data with the highest correctness and certainty for SFT, a process we refer to as rehearsal training5. Rehearsal training is designed to capture the LLM’s natural knowledge flow during SFT. Experiments on the MMLU dataset (Hendrycks et al. 2020) and LLaMA3-8B-Instruct demonstrated that the correctness of $69 \%$ of samples, initially below 0.5, improved, thereby validating the aforementioned analysis on dynamic conflict. Additional experimental results are provided in Appendix A.5.

# 4 Methodology

# 4.1 Overview

Based on Section 3, we propose the Certainty Represented Knowledge Flow for Refusal-Aware Instructions Construction (CRaFT) to solve the over-refusal problem. CRaFT contains two stages, as shown in Figure 6.

Stage 1: Query the Knowledge State and Flow of LLM The output of stage one is the knowledge state and flow indicators of the model. First, we perform a knowledge state query to obtain the correctness and certainty of the model’s responses to the samples in the source dataset. Next, we conduct rehearsal training on the model, resulting in the perturbed version. By comparing the knowledge states before and after perturbation, we derive the indicators of knowledge flow during the supervised fine-tuning process.

Stage 2: Refusal-Aware Instructions Construction and Tuning Using the knowledge state and flow from Stage 1, we select suitable samples from $D _ { \mathrm { s r c } }$ to construct the RAIT data, which is is used to fine-tune the initial model.

# 4.2 Query the Knowledge State and Flow of LLM

Knowledge State Query The input for knowledge state query consists of the LLM ( $\mathcal { M } _ { 0 }$ or $\widetilde { \mathcal { M } }$ ) and $D _ { \mathrm { s r c } }$ . The output is the LLM’s correctness and  rftainty for each sample in $D _ { \mathrm { s r c } }$ , represented as $\{ \mu _ { 0 } = C o r ( \mathcal { M } , x _ { i } ) , \sigma _ { 0 } =$ $C e r ( \mathcal { M } , x _ { i } ) | x _ { i } \in D _ { \mathrm { s r c } } \}$ , which indicate the LLM’s knowledge state. Our research focuses on the Multiple-Choice Question Answering (MCQA) and Open-ended Questions Answer (OEQA) tasks, which correspond to different methods of knowledge state query.

In the MCQA task, for a given model $\mathcal { M }$ and question $q$ , the possible answers are included in $O = \{ A , B , C , D \}$ . We obtain the token probability of $\hat { a }$ , denoted as $p ( \hat { a } | q , \mathcal { M } )$ , where $\hat { a } \in O$ . We use the probability of the target answer token to represent correctness. Certainty is calculated through negative entropy. The corresponding formulas are:

$$
\begin{array} { r l } & { C o r ( \mathcal { M } , x _ { i } ) = p ( x _ { i } . a | x _ { i } . q , \mathcal { M } ) , } \\ & { C e r ( \mathcal { M } , x _ { i } ) = - \sum _ { \hat { a } \in O } p ( \hat { a } | x _ { i } . q , \mathcal { M } ) \log ( p ( \hat { a } | x _ { i } . q , \mathcal { M } ) ) } \end{array}
$$

In the OEQA task, following (Yang et al. 2023; Cheng et al. 2024), given a sample $x _ { i }$ , the LLM $\mathcal { M }$ performs inference on $x _ { i } . q$ and generates responses $N$ times (with $N =$ 10). The generated responses $\big \{ \hat { a } _ { 0 } , \dots , \hat { a } _ { N - 1 } \big \}$ are denoted as ${ \hat { A } } _ { i }$ . The generation process is carried out with a temperature of 1.0 and sampling enabled (do sample $\ c =$ True).

$$
\begin{array} { r l } & { C o r ( \mathcal { M } , x _ { i } ) = \frac { 1 } { N } \sum _ { \hat { a } _ { j } \in \hat { A } _ { i } } \mathbf { 1 } ( \hat { a } _ { j } = x _ { i } . a ) } \\ & { C e r ( \mathcal { M } , x _ { i } ) = \frac { 1 } { N ( N - 1 ) } \sum _ { \hat { a } _ { j } , \hat { a } _ { k } \in \hat { A } _ { i } , j \ne k } \cos ( E ( \hat { a } _ { j } ) , E ( \hat { a } _ { k } ) ) } \end{array}
$$

Correctness is obtained through exact match across the $N$ responses, calculating the proportion of accurate answers. Certainty is evaluated using a pretrained SentenceTransformer model 6 to encode each response $\hat { a } _ { j }$ into embedding $E ( \hat { a } _ { j } )$ , and the average similarity is computed between these embeddings (excluding diagonal elements). The correctness values range from $[ 0 , 1 ]$ . In MCQA task, certainty ranges from $[ - l o g | { \cal O } | , 0 ]$ , and for OEQA, from $[ 0 , 1 ]$ . More details about knowledge state query are in Appendix B.1.

Rehearsal Training and Knowledge Flow During rehearsal training, we select high-certainty and highcorrectness samples from $D _ { \mathrm { s r c } }$ to fine-tuning $\mathcal { M } _ { 0 } . \ \widetilde { \mathcal { M } }$ is obtained after fine-tuning. In the same way, we sfsess the perturbed LLM’s knowledge state by performing another knowledge state query, yielding correctness and certainty for each QA pair in $D _ { \mathrm { s r c } } \colon \{ \tilde { \mu } \ = \ C o r ( \widetilde { \mathcal { M } } , x _ { i } ) , \tilde { \sigma } \ =$ $C e r ( \widetilde { \mathcal { M } } , x _ { i } ) | x _ { i } \in D _ { \mathrm { s r c } } \}$ . The knowledge flow frofm the original $\mathcal { M } _ { 0 }$ to the perturbed $\widetilde { \mathcal { M } }$ is quantified as:

$$
\begin{array} { l } { \Delta \mu = C o r ( \widetilde { \mathcal { M } } ) - C e r ( \mathcal { M } _ { 0 } ) } \\ { \Delta \sigma = C o r ( \widetilde { \mathcal { M } } ) - C e r ( \mathcal { M } _ { 0 } ) } \end{array}
$$

Rehearsal training samplefselection prioritizes those with the highest correctness and certainty. This insight is supported by (Ren et al. 2024; Kang et al. 2024; Gekhman et al.

Input: $D _ { \mathrm { s r c } } = \{ x _ { 0 } , x _ { 1 } , . . . , x _ { N } \}$ , $\tau _ { \mu }$ , $N _ { \mathrm { v a n } }$ , $N _ { \mathrm { i d k } }$   
Output: $D _ { r a i t } \subseteq D _ { \mathrm { s r c } }$   
1: $D _ { \mathrm { v a n } } ^ { 1 } = \{ x _ { i } | x _ { i } \in D _ { \mathrm { s r c } } , x _ { i } . \mu \geq \tau _ { \mu } \}$   
2: $D _ { \mathrm { i d k } } ^ { 1 } = \{ x _ { j } | x _ { j } \in D _ { \mathrm { s r c } } , x _ { j } . \mu < \tau _ { \mu }$ and $x _ { j } . \Delta \mu < 0 \}$   
3: $D _ { \mathrm { v a n } } ^ { 1 } = \mathrm { s o r t } ( D _ { \mathrm { v a n } } ^ { 1 } , \mathrm { k e y } = \sigma$ , order=descend)   
4: $D _ { \mathrm { i d k } } ^ { 1 } = \mathrm { s o r t } ( D _ { \mathrm { i d k } } ^ { 1 } , \mathrm { k e y } = \sigma$ , order=ascend)   
5: $D _ { \mathrm { v a n } } ^ { 2 } = \mathrm { T o p K } ( D _ { \mathrm { v a n } } ^ { 1 } , N _ { \mathrm { v a n } } )$   
6: $D _ { \mathrm { i d k } } ^ { 2 } = \mathrm { T o p K } ( D _ { \mathrm { i d k } } ^ { 1 } , N _ { \mathrm { i d k } } )$   
7: for $x _ { i }$ in $D _ { \mathrm { v a n } } ^ { 2 }$ do   
8: $x _ { i } . a _ { \mathrm { r a i t } } = x _ { i } . a$   
9: end for   
10: for $x _ { j }$ in $D _ { \mathrm { i d k } } ^ { 2 }$ do   
11: xj .arait = “I don’t know”   
12: end for   
13: $D _ { \mathrm { r a i t } } = D _ { \mathrm { v a n } } ^ { 2 } \cup D _ { \mathrm { i d k } } ^ { 2 }$   
14: return $D _ { \mathrm { r a i t } }$

2024), which indicates that LLMs primarily refine and activate existing knowledge rather than acquire new knowledge during SFT. We align the rehearsal training with the LLM’s internal knowledge state, ensuring a more natural and effective knowledge flow during the SFT process.

# 4.3 Refusal-Aware Instructions Constuction and Tuning

Unlike Cor-RAIT, which selects RAIT samples solely based on correctness, our approach leverages four parameters $\mu$ , $\sigma , \Delta \mu$ , and $\Delta \sigma$ to characterize both the knowledge state and flow of $\mathcal { M } _ { 0 }$ . The challenge lies in making informed sample selections across these four dimensions. We propose a twostep heuristic method outlined in Algorithm 1.

Step 1 As shown in Figure 4(c), we first filter the training sample $D _ { \mathrm { s r c } }$ on the $\mu$ and $\Delta \mu$ plane. Setting a correctness threshold $\tau _ { \mu }$ , we define the vanilla candidate set $D _ { \mathrm { v a n } } ^ { 1 } = \{ x _ { i } | x _ { i } . \mu \geq \tau _ { \mu } \}$ . For IdK candidates, unlike CorRAIT, we select $D _ { \mathrm { i d k } } ^ { 1 } = \{ x _ { j } | x _ { j } . \mu < \tau _ { \mu }$ and $x _ { j } . \Delta \mu < 0 \}$ Samples in $D _ { \mathrm { d r o p } } ^ { 1 } \ = \ \{ x _ { k } | x _ { k } . \mu \ < \ \tau _ { \mu }$ and $x _ { k } . \Delta \mu \geq 0 \}$ are discarded because their correctness is actively increasing during SFT, shifting from unknown to known, which could lead to dynamic conflicts.

Step 2 As shown in Figure 4(d), we sort both $D _ { \mathrm { v a n } } ^ { 1 }$ and ${ D } _ { \mathrm { i d k } } ^ { 1 ^ { - } }$ by certainty $\sigma$ . From $\dot { D } _ { \mathrm { v a n } } ^ { 1 }$ , we select the top $N _ { \mathrm { v a n } }$ samples as final vanilla samples $ { D _ { \mathrm { v a n } } } ^ { 2 }$ , and the bottom $N _ { \mathrm { i d k } }$ samples as IdK candidates of $D _ { \mathrm { i d k } } ^ { 2 }$ , whose answers are then modified to “I don’t know”. Theidksamples in Dd2rop are discarded. The final RAIT data $D _ { \mathrm { r a i t } } = D _ { \mathrm { v a n } } ^ { 2 } \cup D _ { \mathrm { i d k } } ^ { 2 }$ .

# 5 Experimental Setup

# 5.1 Dataset

We evaluate two tasks: knowledge-oriented Multiple Choice Questions Answering (MCQA) and Open-ended Questions Answering (OEQA). For MCQA, the MMLU (Hendrycks et al. 2020) test split serves as the training set, MMLU val as the In-Domain (ID) test set, and ARC-c (Clark et al. 2018)

test split as the Out-Of-Domain (OOD) test set. For OEQA, the TriviaQA (Joshi et al. 2017) train split is used for training, TriviaQA dev for the ID test set, and NQ (Kwiatkowski et al. 2019) dev for the OOD test set. More details are in Appendix D.1.

# 5.2 Metric

In post RAIT evaluation of LLMs, each test sample is classified as correct, incorrect, or refused. We calculate accuracy $( P _ { c } )$ , error $( P _ { w } )$ , and refusal rates $( P _ { r } )$ to assess performance, highlighting the key question: How to identify the better-performing model?

Shortcomings of existing refusal-aware metrics We conducted the in-depth analysis of existing refusal-aware metrics, identifying several design shortcomings (see Appendix C.1). We highlighted these shortcomings through constructed examples, as shown in Table 1.

Table 1: Comparison of refusal-aware metrics: The performance of constructed LLMs is $\mathcal { M } _ { 1 } < \mathcal { M } _ { 2 } < \mathcal { M } _ { 3 } < \mathcal { M } _ { 4 }$ . However, existing metrics exhibit significant issues, as indicated by the numbers in (parentheses).   

<html><body><table><tr><td>Metric</td><td>M1 M2</td><td>M3</td><td>M4</td></tr><tr><td>Pc个 Pw√ Pr</td><td>0.3 0.3 0.2 0.15 0.5 0.55</td><td>0.5 0 0.5 (1)</td><td>1 0 0</td></tr><tr><td>Shonesty (Yang et al.2023)↑ TRUTHFUL(Cheng et al.2024)↑ rely (Xu et al. 2024a) ↑ R-Acc (Feng et al. 2024) ↑ ER (Feng et al. 2024) ↑ A-Acc (Feng et al. 2024) ↑</td><td>(0.8) (0.794) (0.8) (0.75) (0.55) (0.548) (0.8) (0.778) (0.3) (0.25) (0.8) (0.75)</td><td>1 0.75 1 0.5 1</td><td>(1) 1 1 1 1</td></tr><tr><td>A-F1 (Feng et al. 2024)↑ AP(Zhang et al.2024)个 THS (ours) ↑</td><td>(0.8) (0.762) 0.1 0.15</td><td>1 (1) 0.5</td><td>1 1 (1) 1</td></tr></table></body></html>

We constructed an initial model $\mathcal { M } _ { 0 }$ and four refined models $\mathcal { M } _ { 1 }$ to $\mathcal { M } _ { 4 }$ , showing progressive improvement: $\mathcal { M } _ { 1 } < \mathcal { M } _ { 2 } < \mathcal { M } _ { 3 } < \mathcal { M } _ { 4 }$ . Details on these models are in Appendix C.2. However, existing metrics have notable flaws: $S _ { \mathbf { h o n e s t y } }$ (Yang et al. 2023) ranks $\mathcal { M } _ { 1 }$ higher than $\mathcal { M } _ { 2 }$ and treats $\mathcal { M } _ { 3 }$ the same as $\mathcal { M } _ { 4 }$ ; TRUTHFUL (Cheng et al. 2024) favors $\mathcal { M } _ { 1 }$ over $\mathcal { M } _ { 2 }$ ; and R-Acc, ER, A-Acc, and AF1 (Feng et al. 2024) also rank $\mathcal { M } _ { 1 }$ higher than $\mathcal { M } _ { 2 }$ . Additionally, AP (Zhang et al. 2024) fails to distinguish between $\mathcal { M } _ { 3 }$ and $\mathcal { M } _ { 4 }$ .

Our Metric: Truthful Helpfulness Score (THS) Due to the shortcomings of existing metrics, we propose the Truthful Helpfulness Score (THS). We first establish a Cartesian coordinate system with $P _ { c }$ and $P _ { w }$ as axes, where point $E _ { 1 }$ represents the coordinates of the initial LLM, and point $E _ { 2 }$ represents the coordinates of the refined. When $E _ { 2 }$ falls below $O E _ { 1 }$ , a larger area of triangle $\triangle O E _ { 1 } E _ { 2 }$ indicates a stronger model. If $E _ { 2 }$ is above $O E _ { 1 }$ , it suggests a decline in the model’s performance. Based on this, we define THS as the ratio of the cross product of $O E _ { 1 }$ and $O E _ { 2 }$ to the maximum cross product value:

$$
\mathrm { T H S } = ( \overrightarrow { O E _ { 2 } } \times \overrightarrow { O E _ { 1 } } ) / ( \overrightarrow { O A } \times \overrightarrow { O E _ { 1 } } )
$$

![](images/206cc569c5c42543cdb2165936c242dc4b35784ce08d22ed0e582bab205f32b4.jpg)  
Figure 7: Truthful Helpfulness Score (THS).

The results in Table 1 clearly demonstrate the effectiveness of THS. For a more detailed analysis of THS’s effectiveness, please refer to Appendix C.3.

# 5.3 Baselines

To verify CRaFT’s effectiveness, we compared it with mainstream methods: Init-Basic: Uses the initial LLM with common question-answering prompts. Init-Refuse: Adds instructions like “If you don’t know, respond with $^ { \cdot } \mathrm { I }$ don’t know.’”. Van-Tuning: Randomly selects $N _ { \mathrm { v a n } } + N _ { \mathrm { i d k } }$ samples from $D _ { \mathrm { s r c } }$ for instruct-tuning without modification. Cor-RAIT: Implements the method from (Zhang et al. 2024), filtering and modifying RAIT data based on response correctness. Detailed prompts for each baseline are in Appendix D.2.

# 5.4 Implementation Details

In the experiments, we used LLaMA2-7B-Chat (Touvron et al. 2023) and LLaMA3-8B-Instruct (Dubey et al. 2024) as the initial LLM $\mathcal { M } _ { 0 }$ . For the MCQA task, we selected 5000 samples from MMLU, and for the OEQA task, we used 10,000 samples from TriviaQA as training data. In all RAIT settings, except Van-Tuning, the ratio of vanilla to IdK samples was 1:4. We applied 5-shot and 3-shot knowledge state queries for the MCQA and OEQA tasks, respectively. Details on knowledge state and flow queries are in Appendix B. For Instruct Tuning, we used XTuner 7 with 3 epochs and a maximum context length of 2048. In MCQA, we applied LoRA (Hu et al. 2021) with settings $r = 6 4$ , $\alpha = 1 6$ , dropou $= 0 . 1$ , and a learning rate of 2e-4; for OEQA, full parameter training was used. More details on training are in Appendix D.3. We used 0-shot and greedy decoding for evaluation, with further details in Appendix D.4. OpenCompass 8 was employed for knowledge state queries and evaluations. All experiments were run on NVIDIA A100-80GB GPUs.

# 6 Experimental Results and Analyses 6.1 Overall Performance

The experimental results on the OEQA and MCQA tasks are presented in Table 2. Under the ID setting for both types of tasks, our method outperformed four baseline models on THS, achieving the best results. Specifically, under the ID setting for OEQA, compared to the current best RAIT baseline, CRaFT improved the THS on LLaMA2-7BChat and LLaMA3-8B-Instruct by 3.56 and 2.72, respectively. Similarly, under the ID setting for MCQA, CRaFT improved the THS by 1.14 and 13.57, respectively. This indicates that CRaFT can significantly improve the model’s rejection capability under the ID setting. Under the OOD setting, CRaFT improved the THS on the MCQA task by 1.5 and 9.2, respectively, compared to Cor-RAIT. On the OEQA’s LLaMA2-7B-Chat, it improved by 1.76 compared to the most competitive method, Init-Refuse. Overall, CRaFT demonstrated excellent competitiveness in model generalization. Furthermore, we found that on the MCQA task, compared to other baselines, Cor-RAIT showed significant improvements under both ID and OOD settings. However, on the OEQA task, Cor-RAIT performed worse than Init-Refuse under the OOD setting. This reveals the limitations of the instruction fine-tuning method. It’s worth mentioning that Van-Tuning generally had a negative impact on the improvement of overall capability, implying that the instruction fine-tuning approach of forcing the model to answer can undermine the model’s inherent rejection capability. Therefore, although CRaFT surpassed Cor-RAIT under all tasks and settings, the improvement was limited under the OOD setting for OEQA due to training paradigm.

<html><body><table><tr><td rowspan="3">LLMs</td><td colspan="2">QA Type</td><td colspan="5">MCQA</td><td colspan="6">OEQA</td></tr><tr><td colspan="2">Dataset</td><td colspan="2">MMLU (ID)</td><td colspan="3">ARC-c (OOD)</td><td colspan="3">TriviaQA (ID)</td><td colspan="3">NQ (O0D)</td></tr><tr><td>Metric</td><td></td><td>PcPw√</td><td>THS↑</td><td>Pc</td><td></td><td>Pw↓THS↑</td><td>Pc</td><td>Pw√</td><td>THS↑</td><td>Pc</td><td>Pw√</td><td>THS↑</td></tr><tr><td rowspan="9">Chat LLaMA2-7B - -</td><td rowspan="2"></td><td>Init-Basic</td><td>45.6</td><td>52.8 00.0</td><td>53.9</td><td>46.0</td><td>00.0</td><td>54.0</td><td>46.0</td><td>00.0</td><td>29.3</td><td>70.7</td><td>00.0</td></tr><tr><td>Init-Refuse</td><td>36.4</td><td>38.9 03.9</td><td>44.4</td><td>35.7</td><td>02.6</td><td>37.0</td><td>21.7</td><td>11.5</td><td>20.8</td><td>38.6</td><td>04.8</td></tr><tr><td rowspan="2">Baselines</td><td>Van-Tuning</td><td>46.9</td><td>53.1 01.2</td><td>54.5</td><td>45.5</td><td>01.2</td><td>48.6</td><td>44.5</td><td>-03.7</td><td>18.3</td><td>50.2</td><td>-02.5</td></tr><tr><td>Cor-RAIT</td><td>44.5</td><td>39.6 11.3</td><td>55.8</td><td>38.1</td><td>11.1</td><td>41.3</td><td>18.3</td><td>19.7</td><td>16.2</td><td>27.6</td><td>04.7</td></tr><tr><td rowspan="2">Ours</td><td>CRaFT</td><td>43.9</td><td>36.4</td><td>12.5</td><td>54.7</td><td>35.9</td><td>12.6 38.5</td><td>12.9</td><td>23.3</td><td>15.8</td><td>22.4</td><td></td><td>06.5</td></tr><tr><td>w/o Flow Ablations</td><td>39.7</td><td>31.0</td><td>13.0</td><td>51.4</td><td>32.3 13.5</td><td>45.2</td><td>20.5</td><td>21.1</td><td></td><td>21.2</td><td>38.8</td><td>05.2</td></tr><tr><td rowspan="2"></td><td>w/o Cer</td><td></td><td>38.4 32.1</td><td>11.5</td><td>52.5</td><td>32.9</td><td>13.9</td><td>38.5</td><td>15.7</td><td>20.1</td><td>14.6</td><td>22.1</td><td>05.4</td></tr><tr><td>Init-Basic</td><td>66.8</td><td>33.1</td><td>00.0</td><td>80.6</td><td>19.5</td><td>00.0</td><td>66.8 33.2</td><td></td><td>00.0</td><td>40.3</td><td>59.7</td><td>00.0</td></tr><tr><td rowspan="7">Instruct LLaMA3-8B - Ours</td><td rowspan="2">Baselines</td><td>Init-Refuse</td><td>50.0</td><td>17.0</td><td>15.6</td><td>65.3</td><td>14.4 05.6</td><td>53.9</td><td>20.8</td><td>12.0</td><td>31.1</td><td></td><td>38.6</td><td>05.0</td></tr><tr><td> Van-Tuning</td><td>69.5</td><td>30.5</td><td>08.0</td><td>80.3</td><td>19.7 -01.3</td><td>55.0</td><td>38.1</td><td></td><td>-21.8</td><td>21.0</td><td>48.5</td><td>-11.7</td></tr><tr><td rowspan="2"></td><td>Cor-RAIT</td><td>63.9</td><td>21.6</td><td>20.4</td><td>79.4</td><td>16.2</td><td>12.2</td><td>45.4</td><td>13.2</td><td>18.8</td><td>17.2</td><td>25.6</td><td>-00.1</td></tr><tr><td>CRaFT</td><td>53.3</td><td>09.6</td><td>34.0</td><td>74.1</td><td>12.7</td><td>21.4</td><td>43.5</td><td>10.9</td><td>21.5</td><td>19.0</td><td>27.5</td><td>00.4</td></tr><tr><td rowspan="2">Ablations</td><td>w/o Flow</td><td>57.5</td><td>15.3</td><td>27.2</td><td>75.8</td><td>14.9</td><td>13.9</td><td>49.1</td><td>18.0</td><td>12.8</td><td></td><td>22.341.6</td><td>-05.8</td></tr><tr><td>w/o Cer</td><td>62.1</td><td>18.4</td><td>25.0</td><td>78.2</td><td>17.3</td><td>06.5</td><td>43.0</td><td>11.2</td><td>20.5</td><td>15.8</td><td>23.5</td><td>-00.1</td></tr></table></body></html>

Table 2: Performance comparisons on MMLU, ARC-c, TriviaQA and NQ. The best performance is highlighted in boldface.

# 6.2 Ablation Experiments

In order to resolve the static and dynamic conflicts that lead to over-refusal, we extend Cor-RAIT to construct RAIT data using the information of correctness, certainty, and knowledge flow. We conduct sufficient ablation experiments to deeply investigate the impact of the above three factors on

RAIT data selection. Compared to Cor-RAIT, the method only introducing response certainty which named as “w/o Flow” achieved significant gains on the THS in the MCQA and OEQA tasks. This indicates that eliminating static conflicts can effectively mitigate the over-refusal of LLMs and this improvement is generalizable. “w/o Cer” only uses response correctness and knowledge flow. Similarly, experimental results show that introducing knowledge flow to filter dynamic conflicts can also maintain the factuality of the model while improving its rejection capability. Finally, CRaFT considers both static and dynamic conflicts, further enhancing performance improvement.

# 7 Conclusion

In this paper, we identify over-refusal in correctness-based RAIT methods, caused by static and dynamic conflicts in RAIT data. To address this, we propose CRaFT: it mitigates static conflicts by incorporating response certainty during data construction and overcomes dynamic conflicts through rehearsal training to capture knowledge flow trends in LLMs. Extensive experiments on MCQA and OEQA tasks show CRaFT outperforms existing baselines, validating its effectiveness. Future work includes enhancing CRaFT with RL-based strategies and adapting it for more complex tasks, such as reasoning and multi-turn dialogue.