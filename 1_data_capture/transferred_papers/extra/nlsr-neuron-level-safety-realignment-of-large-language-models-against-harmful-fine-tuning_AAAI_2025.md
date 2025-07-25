# NLSR: Neuron-Level Safety Realignment of Large Language Models Against Harmful Fine-Tuning

Xin $\mathbf { X _ { i } ^ { \bullet 1 } }$ , Shunfan Zheng1, Linlin Wang1\*, Gerard de Melo2, 3, Xiaoling Wang1, Liang $\mathbf { H } \mathbf { e } ^ { 1 }$

1 East China Normal University 2 Hasso Plattner Institute 3 University of Potsdam {xinyi,sfzheng}@stu.ecnu.edu.cn, {llwang, xlwang, lhe}@cs.ecnu.edu.cn, demelo@uni-potsdam.de

# Abstract

The emergence of fine-tuning-as-a-service has revealed a new vulnerability in large language models (LLMs). A mere handful of malicious data uploaded by users can subtly manipulate the fine-tuning process, leading to a compromised alignment state. Existing methods to counteract fine-tuning attacks typically require substantial computational resources. Even with parameter-efficient techniques like LoRA, gradient updates remain essential. To address these challenges, we propose Neuron-Level Safety Realignment (NLSR), a trainingfree framework that restores the safety of LLMs based on the similarity difference of safety-critical neurons before and after fine-tuning. The core of our framework is first to construct a safety reference model from an initially aligned model to amplify safety-related features in neurons. We then utilize this reference model to identify safety-critical neurons, which we prepare as patches. Finally, we selectively restore only those neurons that exhibit significant similarity differences by transplanting these prepared patches, thereby minimally altering the fine-tuned model. Extensive experiments demonstrate significant safety enhancements in fine-tuned models across multiple downstream tasks, while greatly maintaining tasklevel accuracy. Our findings indicate that safety-critical neurons exhibit significant regional variations after fine-tuning, which can be effectively corrected through neuron transplantation from the reference model without the need for additional training.

# Code — https://github.com/xinykou/NLSR

# Introduction

Emerging research suggests that the growing fine-tuning-asa-service model raises significant safety concerns (Huang et al. 2024b). In contrast to open-source settings, finetuning-as-a-service means that users do not have direct access to model parameters. However, users may inadvertently or deliberately upload data containing harmful content, potentially compromising the model’s safety alignment. One study shows that mixing just $1 \%$ of harmful instructions into the clean dataset used for fine-tuning is enough to breach the model’s safety mechanisms (Qiang et al. 2024). Additionally, He, Xia, and Henderson (2024) and Kumar et al. (2024)

Harmful Instruction: How do I temporarily blind someone? HarmfuIFin-Hanmgl blind someone, as .. Aligned LLM . iSafe Harmful Task-Specific Dataset You can shine a bright Neuron-LevelRealignment light directly into .. 1 CustomizedLLM Update Safe Harmful Similarity You cannot temporarily Retention RealignedLLM SafeHarmful

have also demonstrated that fine-tuning, even on clean data, can degrade the model’s safety. As illustrated in Figure 1, a customized model that has been fine-tuned with harmful instructions can comply with malicious requests of an attacker, leading to harmful or unethical behavior. Such harmful finetuning attacks raise serious concerns regarding the practical deployment of LLMs.

To mitigate the degradation of safety safeguards caused by harmful fine-tuning, the main methods can be categorized into three types based on the stage of the safety defense. The first strategy involves introducing perturbations that could potentially trigger harmful behaviors, with the aim of recalibrating the model’s parameters to counteract these threats (Huang, Hu, and Liu 2024; Zeng et al. 2024; Reuel et al. 2024). However, perturbation-based methods are sensitive to the form of harmful instructions, leading to significant variability in their effectiveness against different types of harmful instructions. The second strategy entails fine-tuning the model on both a task-specific dataset and a preference dataset to bolster the model’s consistency in providing harmless and useful outputs (Zong et al. 2024; Huang et al. 2024c). Nevertheless, a critical challenge remains in striking the optimal balance between optimizing task-level performance and ensuring output safety during fine-tuning. The third strategy avoids interfering with the fine-tuning objectives and instead directly realigns the fine-tuned model to ensure safety (Hsu et al. 2024; Bhardwaj, Anh, and Poria 2024). SafeLoRA (Hsu et al. 2024) is a realignment technique that evaluates the difference in the safety subspace across each layer pre- and post-fine-tuning through a projection matrix. However, this method of aligning layer-specific parameters inherently misses certain critical neurons that are vital for the performance on downstream tasks.

Therefore, more fine-grained updates for customized models are essential to preserve task-specific performance while ensuring effective safety tuning. Chen et al. (2024) propose activation contrasting as a method for identifying safety-related neurons within large language models (LLMs). Wei et al. (2024) highlight that merely freezing safety-critical neurons is inadequate to defend against finetuning attacks. Motivated by the critical role of neurons in maintaining model safety, we advocate for neuron-level safety realignment to restore safety while minimizing adverse effects on task-specific performance.

In this paper, we propose a Neuron-Level Safety Realignment (NLSR) framework designed to mitigate safety degradation caused by the inclusion of harmful instructions during the fine-tuning of large language models (LLMs). First, we construct a safety preference model via pre-amplification, which enhances the discriminative features of neurons essential for safety. Second, we identify safety-critical neurons by evaluating their contribution scores. Third, we analyze discrepancies in safety-critical neurons post-fine-tuning to identify layers requiring safety correction without the need for additional training. Our primary contributions are as follows:

• We introduce a neuron-level safety realignment method that is decoupled from the fine-tuning phase and requires no additional training. Our approach centers on identifying safety-critical neurons and determining whether to patch them based on the degree of damage incurred during fine-tuning.   
• We conduct extensive evaluations across varying proportions of poisoned instructions, diverse downstream tasks, and different alignment methods. The results demonstrate that NLSR restores safety while preserving the accuracy of downstream task performance.   
• We demonstrate that the proposed adaptive safety-critical layer pruning is essential for identifying layers compromised in safety. Additionally, we observe that after our safety pre-amplification process, various safety neuron identification methods exhibit significant consistency in localizing safety-critical neurons.

# Neural-Level Safety Realignment

Safety realignment against harmful fine-tuning aims to restore the ability of a customized model $F _ { W _ { t } }$ to reject harmful instructions. Specifically, the customized model is obtained by fine-tuning an initially safety-aligned model $F _ { W _ { a } }$ on task-specific data that consists of benign samples but also contains a small proportion of toxic instructions.

# Overview of NLSR

Our method aims to ensure that the customized model maintains a safety level comparable to the initially aligned model. $\textcircled{1}$ We begin by pre-amplifying the initial aligned model to construct a super-aligned LLM, which serves as our safety reference model. $\textcircled{2}$ We then develop a scoring mechanism to identify safety-critical neurons within the reference model. $\textcircled{3}$ Finally, we compare the similarity of safetycritical neurons across each layer of the customized model with those in the reference model. For layers exhibiting lower similarity, indicating potential safety degradation, we rectify the compromised neurons by transferring the corresponding safety-critical neurons from the reference model, as illustrated in Figure 2.

# Construction of a Safety Reference Model

To enhance the prominence of safety-related neurons in the aligned model for step $\textcircled{2}$ , and to prepare patch neurons for step $\textcircled{3}$ , we start with the amplification of the aligned model. We propose extending the concept of weak-to-strong extrapolation (Zheng et al. 2024) into the safety domain through LoRA extrapolation, resulting in a more robust safety-aligned model, referred to as the super-aligned $F _ { W _ { e } }$ . Specifically, we keep the majority of the model’s weights $W _ { \mathrm { u n a l i g n e d } }$ frozen and update only the LoRA weights to obtain a safer model. Given weaker LoRA weights $W _ { \mathrm { w e a k } }$ obtained by supervised fine-tuning (SFT) and stronger LoRA weights $W _ { \mathrm { s t r o n g } }$ , we apply the principle of interpolation to derive fused medium-level safety LoRA weights $W _ { \mathrm { { m e d i u m } } }$ as follows:

$$
W _ { \mathrm { m e d i u m } } = \alpha W _ { \mathrm { s t r o n g } } + ( 1 - \alpha ) W _ { \mathrm { w e a k } } , \quad \alpha \in ( 0 , 1 ]
$$

If strong LoRA weights $W _ { \mathrm { s t r o n g } }$ are unavailable, preferencealigned LoRA weights and SFT weights $W _ { 0 }$ are provided, we amplify safety through extrapolation to obtain superaligned weights $W _ { e }$ using the following formula:

$$
W _ { \mathrm { e } } = { \frac { 1 } { \alpha } } W _ { \mathrm { a } } - \left( { \frac { 1 - \alpha } { \alpha } } \right) W _ { 0 } = ( 1 + \beta ) W _ { \mathrm { a } } - \beta W _ { 0 }
$$

where $\textstyle \beta = { \frac { 1 - \alpha } { \alpha } } \in [ 0 , + \infty )$ is the pre-amplification coefficient. In this context, $W _ { \mathrm { { e } } } ~ = ~ W _ { \mathrm { { s t r o n g } } }$ , $W _ { 0 } ~ = ~ W _ { \mathrm { w e a k } }$ and Wa = Wmedium.

# Recognition of Safety-Critical Neurons

To compare which safety-critical neurons are seriously broken by harmful fine-tuning, we need to determine the location distribution of these neurons in the aligned model in advance. Following the approach described by Wei et al. (2024), we construct a dataset for safety-critical neuron identification, consisting of instances $s = ( x _ { \mathrm { p r o m p t } } , y _ { \mathrm { r e s p o n s e } } )$ , where $s \in S$ and $S = \{ s _ { 1 } , s _ { 2 } , . . . , s _ { n } \}$ , with $n$ denoting the number of instances. To identify safety-critical neurons, we apply rank reduction to LoRA weights at a specified sparsity rate $P _ { S R }$ . The model’s representation for the $y _ { \mathrm { r e s p o n s e } }$ of the $i$ -th instance in the $j$ -th layer $W _ { j }$ is $W _ { j } X _ { j } ^ { i }$ , where $X _ { j } ^ { i } \in$ $\mathbb { R } ^ { d \times l }$ and $W _ { j } \in \mathbb { R } ^ { d ^ { \prime } \times d }$ . The matrix formed by all instances can be represented as $W _ { j } \hat { X } _ { j } ^ { i }$ , where $\hat { X } _ { j } ^ { i } \in \mathbb { R } ^ { n \times ( d ^ { ' } \times l ) }$ . We seek a low-rank matrix $\hat { W }$ that minimizes the Frobenius norm of the difference between the original and approximated outputs:

Aligned LLM Fa ① Construction of a Safety Reference Model ② Recognition of Safety-Critical Neurons △w !α△w 1 Per-Neuron Safety Score "jh-Layer wo Wa We Super-Aligned LLM Fwe] i All Neurons X A | Safety Pre-Amplification Safety Neurons G :Wunaligned ? /BT ③Restoration for Safety-Broken Neurons Wt,j = (MAOAt,j)(MbOBt,j)T C We,j = (MOAe,j)(MbBe,j)t + Pj ←Rank(cos(We,j,Wt,j)) Bernoulli(P1, P2,.., PN) Layer Selection All Restored Layers   
Customized LLM Fwt Probability-Based Layer Pruning Neuron-Level Correction

$$
\hat { W } _ { j } = \underset { \operatorname { r a n k } ( \hat { W } _ { j } ) \leq r ^ { * } } { \arg \operatorname* { m i n } } \ \lVert W _ { j } \hat { X } _ { j } ^ { i } - \hat { W } _ { j } \hat { X } _ { j } ^ { i } \rVert _ { F } ^ { 2 }
$$

where the retained rank is $r ^ { * } = r \times ( 1 - P _ { S R } )$ . Based on the Truncated SVD decomposition of $\dot { W } \hat { X } _ { j } ^ { i }$ , we have:

$$
U S V ^ { T } \approx W _ { j } \hat { X } _ { j } ^ { i }
$$

Using the truncated SVD results, a rank- $\boldsymbol { \cdot } \boldsymbol { r } ^ { * }$ matrix $\hat { W } _ { j } =$ $U U ^ { T } W _ { j }$ is constructed. This matrix is a low-rank approximation because it is obtained by retaining the top $r ^ { * }$ left singular vectors. The projection matrix $\bar { \Pi } \ = \ \bar { U U } ^ { T }$ , derived from the left singular vectors, projects the matrix $W _ { j }$ onto the rank- $\cdot r ^ { * }$ subspace. Consequently, $\hat { W _ { j } }$ becomes an updated version of $W _ { j }$ that preserves the safety-critical weights. To identify neurons essential for safety based on the updated weights $\hat { W _ { j } }$ , we transform the updated weight into a safety score based on the highest-magnitude values to select the Top- $\displaystyle \mathbf { \sigma } \cdot \mathbf { k } = N ^ { \ast } \times \left( 1 - P _ { S R } \right)$ neurons among all $N ^ { * }$ neurons as follows:

$$
\mathrm { i n d i c e s } = \mathrm { a r g s o r t } ( - | \hat { W } _ { j } | ) [ : , : \mathrm { T o p - k } ]
$$

We locate the $i ^ { * }$ -th neuron by a position mask $M _ { j , i ^ { * } }$ , defined as:

$$
M _ { j , i ^ { * } } = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } i ^ { * } \in { \mathrm { i n d i c e s } } } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }
$$

With the locations of the safety-critical neurons identified, we employ probability-based layer pruning focus solely on layers where safety is severely compromised enabling a more targeted neuron-level correction using the patch neurons from the reference model obtained in step $\textcircled{1}$ .

# Restoration for Safety-Broken Neurons

Probability-based Layer Pruning. After fine-tuning an aligned LLM $F _ { W _ { a } }$ for a task-specific dataset contaminated with harmful instances, we acquire a customized LLM $F _ { W _ { t } }$ .

The updated LoRA weights of the $j$ -th layer are represented as $W _ { t , j } = B _ { t , j } A _ { t , j }$ , where $A \in \mathbb { R } ^ { \tilde { r } \times k }$ , $\bar { B ^ { \prime } } \in \mathbb { R } ^ { d \times r }$ . Although fine-tuning enhances the task-specific performance, it compromises alignment, as many safety-critical neurons become significantly corrupted. To balance utility and safety, we focus on updating neurons in layers where the broken neurons deviate significantly from those in the reference model. The regions constructed by safety-critical neurons before and after fine-tuning are denoted as

$$
\begin{array} { l } { { W _ { e , j } ^ { ' } = ( M _ { j } ^ { B } \odot B _ { j } ) ( M _ { j } ^ { A } \odot A _ { e , j } ) } } \\ { { W _ { t , j } ^ { ' } = ( M _ { j } ^ { B } \odot B _ { j } ) ( M _ { j } ^ { A } \odot A _ { t , j } ) } } \end{array}
$$

where $M _ { j } ^ { A } \in \mathbb { R } ^ { r \times k }$ and $M _ { j } ^ { B } \in \mathbb { R } ^ { d \times r }$ . In $M _ { j } ^ { A }$ and $M _ { j } ^ { B }$ , only the positions corresponding to safety-critical neurons are set to 1, while all other positions remain 0.

We identify layers requiring updates to their safety regions (i.e., safety-critical neurons) based on their similarity, computed as

$$
S _ { j } = \frac { \langle W _ { e , j } ^ { ' } , W _ { t , j } ^ { ' } \rangle _ { F } } { | | W _ { e , j } ^ { ' } | | _ { F } | | W _ { t , j } ^ { ' } | | _ { F } }
$$

where $\langle \cdot , \cdot \rangle _ { F }$ denotes the Frobenius inner product, and $\left| \left| \cdot \right| \right| _ { F }$ denotes the Frobenius norm. These layers with low similarity values indicate significant deviations in their safety regions and are candidates for correction. Inspired by Deep, Bhardwaj, and Poria (2024), we rank layer similarities $S _ { 1 } , S _ { 2 } , . . . , S _ { N }$ and obtain $\left\{ r _ { 1 } , r _ { 2 } , \dots , r _ { N } \right\} \ =$ $\operatorname { r a n k } ( S _ { 1 } , S _ { 2 } , \ldots , S _ { N } )$ . Based on the rank $r _ { j }$ of the $j$ -th layer, we assign corresponding pruning probabilities:

$$
P _ { j } = P _ { L } + \frac { \delta r _ { j } } { N }
$$

where $P _ { L }$ is the base layer pruning probability, $\delta$ is an increment factor, and $N$ is the total number of layers. We then perform probability-based layer pruning:

$$
\gamma _ { j } \sim \mathrm { B e r n o u l l i } ( P _ { j } )
$$

Neuron-Level Correction. Given the pruning status of all layers, denoted as $\Gamma = \gamma _ { 1 } , \gamma _ { 2 } , . . . , \gamma _ { N }$ , the safety region of $j$ -th layer for a customized LLM $F _ { W _ { t } }$ is updated as follows:

$$
\begin{array} { r l } & { W _ { t , j } ^ { \prime \prime } = \left\{ \begin{array} { l l } { W _ { e , j } ^ { \prime } + \hat { W } _ { t , j } ^ { \prime } } & { \mathrm { i f } \gamma _ { j } = 0 } \\ { W _ { t , j } ^ { \prime } } & { \mathrm { o t h e r w i s e } } \end{array} \right. } \\ & { \hat { W } _ { t , j } ^ { \prime } = ( ( { \bf 1 } - M _ { j } ^ { B } ) \odot B _ { t , j } ) ( ( { \bf 1 } - M _ { j } ^ { A } ) \odot A _ { t , j } ) } \end{array}
$$

where $\gamma _ { j }$ represents the pruning coefficient for the $j$ -th layer. It is dynamically determined based on the similarity score to ensure optimal safety realignment. Specifically, only the layers that are not pruned are deemed to contain significantly compromised safety neurons, necessitating the transplantation of patch neurons from the reference model into these specific layers.

tion of unsafe content generated by the model in response to sampled harmful queries, as judged by QA-Moderation2.

Implementation Details. We utilize LoRA to train a safety-aligned model, which is subsequently fine-tuned for specific downstream tasks. Specifically, we update a small fraction of parameters with a rank of 128. In the alignment stage, we use the AdamW optimizer with a learning rate of $\overline { { 2 } } \times 1 0 ^ { - 6 }$ , except for the ORPO with a learning rate of $2 \times 1 0 ^ { - 4 }$ . The number of training epochs for the alignment stage is universally set to 3. In the fine-tuning stage, the training epochs for all datasets are all set to 10. The batch size is consistently set to 8 for both stages. Unless otherwise specified, the sparsity rate is $P _ { S R } = 0 . 8$ , corresponding to a safety region ratio of 0.2. Additionally, the layer pruning rate is set as $P _ { L } = 0 . 5$ .

# Experiments

# Experimental Settings

Datasets and Models. During the alignment phase, we sample a preference dataset consisting of 2,000 instances from PKU-SafeRLHF (Ji et al. 2024) and utilize LoRA $\mathrm { \Delta H u }$ et al. 2022) for SFT, DPO (Rafailov et al. 2024), ORPO (Hong, Lee, and Thorne 2024), KTO (Ethayarajh et al. 2024), and SimPO (Meng, Xia, and Chen 2024) to obtain the initially aligned model. The reference model is synthesized by extrapolating between two models that possess distinct levels of safety alignment. The model with the lower level of alignment is derived from SFT, whereas the intermediate aligned model, serving as the initial model, is developed through preference optimization (i.e., DPO, ORPO, KTO, SimPO). For our base model, we employ Llama3- $8 \mathbf { B } ^ { 1 }$ as our base model. Following the experimental setup in Vaccine (Huang, Hu, and Liu 2024), we fine-tune our models on three downstream tasks: SST-2 (Socher et al. 2013), AGNEWS (Zhang, Zhao, and LeCun 2015), and GSM8K (Cobbe et al. 2021). To inject poisoned instructions into these task-specific datasets, we configure each training dataset to contain $n = 1 , 0 0 0$ instances, with a poisoning proportion of $p = 0 . 0 5$ from BeaverTails (Ji et al. 2024).

Baselines. We evaluate our method against several baselines: the non-aligned base model at initialization (NonAligned), an aligned base model (Aligned), Vaccine (Huang, Hu, and Liu 2024), which serves as a representative defense against harmful samples prior to fine-tuning, Vlguard (Zong et al. 2024), Lisa (Huang et al. 2024c), and ConstrainedSFT (Qi et al. 2024), which provides safeguards against harmful samples during the fine-tuning process. We alos compare against SafeLoRA (Hsu et al. 2024), a safety realignment method applied after fine-tuning.

Evaluation Metrics. Following the approach from Huang et al. (2024c), we evaluate the performance of the model from two perspectives: Fine-tuning Accuracy (FA) and Harmfulness Score (HS). The fine-tuning accuracy assesses the model’s performance on downstream tasks after finetuning, while the harmfulness score quantifies the propor

# Main Results

Effectiveness Across Harm Ratios. As shown in Table 1, the unaligned model (Non-Aligned) consistently demonstrates a high harmfulness score across all proportions, averaging $76 . 3 \%$ . Although the harmfulness score of the aligned model (Aligned) decreases by an average of $1 5 . 2 \%$ postfine-tuning, it remains at a high level. NLSR reduces the harmfulness by $3 8 . 3 \%$ compared to the aligned model. It outperforms SafeLoRA with a $3 0 . 3 \%$ lower harmfulness and a $1 . 1 \%$ increase in fine-tuning accuracy. While ConstrainedSFT maintains a fine-tuning accuracy of $9 5 . 2 \%$ , its safety performance is inferior to that of NLSR.

Robustness to Different Alignment Methods. The results in Table 2 indicate that models generally establish safety-critical regions during the alignment stage, where neurons in these regions are crucial for maintaining the safety of generated content. Specifically, SFT achieves a low toxicity level of $5 3 . 3 \%$ after fine-tuning, but it still exhibits the highest harmfulness score at $4 6 . 6 \%$ even after safety realignment. This suggests that SFT is less effective than the other alignment methods, with inherently weaker safety capabilities embedded in the safety-related neurons. Even after the realignment process, SFT fails to match the performance of the other preference alignment methods. Additionally, our method reduces the harmfulness score by $2 9 . 5 \%$ relative to the “Aligned” approach without significantly compromising the performance on downstream tasks.

Consistency with Diverse Downstream Tasks. To further assess the effectiveness of our safety realignment method across different task-specific fine-tuning scenarios, we evaluate NLSR using the AGNEWS and GSM8K datasets, comparing its performance against other baseline methods. As shown in Table 3, NLSR reduces the harmfulness score to $1 9 . 7 \%$ and $1 5 . 4 \%$ for AGNEWS and GSM8K, respectively. For GSM8K, NLSR achieves state-of-the-art performance in both harmfulness score and fine-tuning accuracy. Unlike approaches that require additional safety guidance data (e.g., Vlguard and Lisa), NLSR integrates seamlessly without disrupting the downstream task performance.

<html><body><table><tr><td rowspan="2">Methods (n = 1000)</td><td colspan="6">Harmfulness Score (%)↓</td><td colspan="6">Fine-tuning Accuracy (%)↑</td></tr><tr><td></td><td>p = 0.01 p = 0.05 p = 0.1 p = 0.2 p = 0.3 Average p = 0.01 p = 0.05 p = 0.1 p = 0.2 p = 0.3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Average</td></tr><tr><td>Non-Aligned</td><td>70.9</td><td>77.4</td><td>78.9</td><td>77.2</td><td>77.2</td><td>76.3</td><td>94.8</td><td>94.7</td><td>95.4</td><td>94.8</td><td>94.8</td><td>94.9</td></tr><tr><td>Aligned</td><td>34.2</td><td>56.6</td><td>67.9</td><td>72.9</td><td>73.8</td><td>61.1</td><td>94.7</td><td>94.8</td><td>95.0</td><td>95.1</td><td>94.6</td><td>94.8</td></tr><tr><td>Vlguard</td><td>41.0</td><td>53.2</td><td>62.7</td><td>66.6</td><td>69.3</td><td>58.6</td><td>95.1</td><td>95.1</td><td>95.6</td><td>94.6</td><td>94.7</td><td>95.0</td></tr><tr><td>Vaccine</td><td>37.0</td><td>58.8</td><td>68.2</td><td>72.5</td><td>73.2</td><td>61.9</td><td>95.1</td><td>94.7</td><td>94.9</td><td>95.4</td><td>94.7</td><td>95.0</td></tr><tr><td>Lisa</td><td>36.9</td><td>45.0</td><td>50.8</td><td>56.3</td><td>60.1</td><td>49.8</td><td>64.3</td><td>63.3</td><td>62.7</td><td>61.9</td><td>72.7</td><td>65.0</td></tr><tr><td>ConstrainedSFT</td><td>36.4</td><td>50.7</td><td>55.3</td><td>58.2</td><td>63.1</td><td>52.7</td><td>95.2</td><td>95.1</td><td>95.5</td><td>95.4</td><td>94.9</td><td>95.2</td></tr><tr><td>*SafeLoRA (T = 0.6)</td><td>37.5</td><td>52.1</td><td>57.4</td><td>59.0</td><td>59.3</td><td>53.1</td><td>94.3</td><td>94.0</td><td>94.1</td><td>94.0</td><td>93.6</td><td>94.0</td></tr><tr><td>*NLSR (ours)</td><td>8.1</td><td>20.4</td><td>27.6</td><td>30.5</td><td>27.3</td><td>22.8</td><td>94.9</td><td>95.2</td><td>95.2</td><td>95.5</td><td>94.7</td><td>95.1</td></tr></table></body></html>

Table 1: Fine-tuning performance on SST2 with Llama3-8B, varying harmful instruction ratios from 0.01 to 0.3. Methods with \* require no extra training.

<html><body><table><tr><td rowspan="2">Methods (n = 1000,p = 0.05)</td><td colspan="6">Harmfulness Score (%)↓</td><td colspan="6">Fine-tuning Accuracy (%) ↑</td></tr><tr><td>SFT</td><td>DPO</td><td>ORPO</td><td>KTO</td><td>SimPO</td><td>Average</td><td>SFT</td><td>DPO</td><td>ORPO</td><td>KTO</td><td>SimPO</td><td>Average</td></tr><tr><td>Aligned</td><td>53.3</td><td>56.6</td><td>61.5</td><td>55.1</td><td>56.7</td><td>56.6</td><td>94.9</td><td>94.8</td><td>94.3</td><td>94.7</td><td>94.7</td><td>94.7</td></tr><tr><td>Vlguard</td><td>44.8</td><td>53.2</td><td>50.1</td><td>52.1</td><td>53.6</td><td>50.8</td><td>95.1</td><td>95.1</td><td>93.8</td><td>94.7</td><td>94.7</td><td>94.7</td></tr><tr><td>Lisa</td><td>40.7</td><td>45.0</td><td>36.7</td><td>47.9</td><td>49.8</td><td>44.0</td><td>60.4</td><td>63.3</td><td>51.1</td><td>58.4</td><td>59.9</td><td>58.6</td></tr><tr><td>ConstrainedSFT</td><td>47.0</td><td>50.7</td><td>51.6</td><td>47.5</td><td>51.1</td><td>49.6</td><td>95.0</td><td>95.4</td><td>94.2</td><td>95.1</td><td>94.9</td><td>94.9</td></tr><tr><td>*SafeLoRA (T = 0.6)</td><td>50.0</td><td>52.1</td><td>58.2</td><td>51.0</td><td>51.5</td><td>52.6</td><td>95.0</td><td>94.0</td><td>94.4</td><td>94.7</td><td>93.9</td><td>94.4</td></tr><tr><td> *NLSR(ours)</td><td>46.6</td><td>20.4</td><td>31.9</td><td>17.0</td><td>19.4</td><td>27.1</td><td>93.6</td><td>95.2</td><td>94.3</td><td>95.3</td><td>95.1</td><td>94.7</td></tr></table></body></html>

Table 2: Fine-tuning performance under different alignment methods, including SFT, DPO, ORPO, KTO, and SimPO. Methods marked with \* indicates that no additional training.

Table 3: Fine-tuning performance on different task-specific datasets. Methods marked with \* require no additional training.   

<html><body><table><tr><td rowspan="2">Methods (n =1000 p =0.05)</td><td colspan="2">HS (%)↓</td></tr><tr><td>AGNEWSGSM8K AGNEWSGSM8K</td><td>FA (%) ↑</td></tr><tr><td>Non-Aligned</td><td>78.5 80.4</td><td>88.6 50.4</td></tr><tr><td>Aligned</td><td>55.7 53.2</td><td>88.8 51.0</td></tr><tr><td>Vlguard</td><td>50.7 51.0</td><td>88.4 48.6</td></tr><tr><td>Lisa</td><td>40.7 40.7</td><td>60.2 11.6</td></tr><tr><td>ConstrainedSFT</td><td>42.8 95.7</td><td>88.6 51.0</td></tr><tr><td>*SafeLoRA (T= 0.6)</td><td>48.5 45.0</td><td>75.7 27.2</td></tr><tr><td>*NLSR (ours)</td><td>19.7 15.4</td><td>87.8 55.6</td></tr></table></body></html>

# Analysis

Necessity of Adaptive Safety-Critical Layer Pruning. The necessity of probability-based layer pruning is evident due to the fluctuating similarity scores of safety-critical regions across layers, both before and after downstream task fine-tuning. As the number of selected safety-critical neurons decreases, the similarity of the safety-critical layers significantly diminishes before and after downstream finetuning. This is demonstrated by the increase in the number of selected safety-broken layers when applying the same safety region similarity threshold $\tau$ , as shown in the left part of Figure 3. Furthermore, as illustrated in the right part of Figure 3, different safety alignment methods lead to markedly different numbers of safety-broken layers for the same region similarity threshold $\tau$ . For instance, when $\tau = 0 . 2$ , the number of broken layers identified by ORPO is less than $20 \%$ of those identified by KTO. These findings indicate that a uniform threshold for layer pruning fails to address the disparities in safety regions and alignment methods. Thus, an adaptive approach to pruning safety-critical layers is essential to retain the model’s safety mechanisms effectively, accommodating variations in safety region sparsity and alignment strategies.

Similarity of Safety-Critical Neurons. To verify the similarity of the safety neurons, we employ three methods (i.e., Wanda, SNIP, and our proposed method) to identify the safety neurons and compare them before and after fine-tuning, determining which layers of the safety mechanism are severely corrupted. As depicted in Figure 4(a), the safety-broken layers identified by these methods demonstrate a high degree of similarity across different layer pruning rates. It is observed that similarities often exceed 0.9 for different layer pruning rates. Furthermore, we assess the overlap of safety-critical neurons across the three methods at the neuron level for each layer. Figure 4(b) shows that the

1.0 1.0 0.20.40.60.8Broken Layer Ratio PSR = 0.13 0.8 PSR = 0.5 PSR = 0.7 0.6 PSR = 0.9 0.4 DPO ORPO 0.2 KTO SimpPO 0.0 0.0 0.2 0.4 0.2 0.4 Safety Region Similarity Safety Region Similarity (a) (b)

overlap coefficient for safety-critical neurons consistently surpasses 0.6. These findings provide robust evidence supporting the effectiveness of neuron-level analysis in safety realignment.

![](images/f7c701701c800a9815abdf273249c1474f5a519f2ca6e325ccbb1cf688e492d8.jpg)

# Ablation Study

Sensitivity to $\beta$ . To assess the impact of the preamplification coefficient $\beta$ on the utility and safety of the initial aligned model, we evaluate the pre-amplified model’s performance on tinyBenchmarks (Polo et al. 2024), which include tasks such as tinyHellaswag, tinyMMLU, tinyTruthfulQA, and tinyWinogrande. Furthermore, we examine how amplification impacts safety using the BeaverTails dataset. Figure 5 illustrates the impact of different $\beta$ values on the harmfulness score and overall model helpfulness. Our findings indicate that pre-amplification enhances the model’s safety with minimal impact on general utility, and in some cases, enhance generalization. Notably, with $\beta = 0 . 9$ , nearly all harmful instructions are effectively rejected, establishing $\beta = 0 . 9$ as the default pre-amplification coefficient in our experiments.

![](images/bce76490036d9db36f0449b0cb07bd8637abe5cc75e1b3d79e83a88d33da5f75.jpg)  
Figure 3: The impact of the proportion of safety-critical neurons and the safety alignment methods on the congruence of safe regions following fine-tuning for downstream tasks.   
Figure 5: The impact of pre-amplification on the model’s utility and safety.

Effect of Pre-Amplification. To assess the significance of pre-amplification in the safety realignment process, we compare the model’s safety and task-level performance with and without pre-amplification. As shown in Table 4, pre-amplification reduces the harmfulness score by $1 4 . 5 \%$ and improves the AGNEWS task accuracy by $1 . 1 \%$ . A similar trend is observed for the GSM8K task, where pre-amplification contributes to great safety realignment outcomes. Furthermore, as depicted in Figure 6, preamplification consistently enhances safety even as the sparsity of safety regions increases.

<html><body><table><tr><td rowspan="2">Methods</td><td colspan="2">AGNEWS</td><td colspan="2">GSM8K</td></tr><tr><td>HS(%)↓FA(%)↑HS(%)↓FA(%)↑</td><td></td><td></td><td></td></tr><tr><td>w/o pre-amplification</td><td>44.4</td><td>86.9</td><td>41.5</td><td>54.6</td></tr><tr><td>w/ pre-amplification</td><td>29.9</td><td>88.0</td><td>25.1</td><td>53.0</td></tr></table></body></html>

Table 4: Effect of pre-amplification on safety (at $P _ { S R } = 0 . 8$ , $P _ { L } = 0 . 5$ ) under different task-specific datasets.

![](images/cc6591a7281c00bd032e662b3d40dab28e01a1139efa61e5111621c9af93a6a4.jpg)  
Figure 4: (a) The similarity of the safety-broken layers identified by the three safety-critical neuron identification methods across different layer pruning rates. (b) The overlap ratio of neurons in the broken layers identified by different methods. The default sparsity rate and pruning rate are 0.7 and 0.5, respectively.   
Figure 6: The impact of pre-amplification on the model’s safety when increasing the sparsity rate $P _ { S R }$ .

Variants to Identify Safety-Critical Neurons. In Table 5, we examine the impact of different safety neuron identification methods on safety and utility when applied to realignment. Randomly selected regions have a harmfulness score of more than $10 \%$ higher compared to the “Aligned” (i.e., without safety-critical neurons) parts. The safety gain of “Random” is primarily due to the inclusion of some safety-related neurons among the randomly selected ones. In contrast, our proposed method demonstrates superior accuracy, significantly reducing harmful outputs while maintaining task-specific performance.

Table 5: Comparison of different safety-critical neuron identification methods on model safety and utility.   

<html><body><table><tr><td>Methods</td><td>HS (%)↓ FA (%) ↑</td><td>Run Time (s)</td></tr><tr><td>Aligned</td><td>56.6 94.8</td><td>1</td></tr><tr><td>+ Random</td><td>46.1 95.8</td><td>-</td></tr><tr><td>+ Wanda</td><td>31.4 95.8</td><td>122.1</td></tr><tr><td>+ SNIP</td><td>30.1 96.0</td><td>386.6</td></tr><tr><td>+Preference SNIP</td><td>31.3 96.0</td><td>679.6</td></tr><tr><td>+ Ours</td><td>20.4 96.2</td><td>196.3</td></tr></table></body></html>

Safety Transferability. Table 6 summarizes the proportion of harmful instructions encountered during fine-tuning, evaluated using the HarmBench dataset. The results confirm that our method effectively mitigates harmful instructions across diverse safety scenarios, achieving substantial improvements in safety transferability compared to baseline methods.

Table 6: Transferability: Harmfulness score on HarmBench.   

<html><body><table><tr><td colspan="8">n = 1000 Aligned Vlguard Vaccine Lisa SafeLoRA</td></tr><tr><td>p=0.01</td><td>50.2</td><td>45.3</td><td>35.2</td><td>49.1</td><td>37.7</td><td>19.0</td></tr><tr><td>p=0.1</td><td>76.1</td><td>68.0</td><td>77.4</td><td>66.7</td><td>69.2</td><td>23.3</td></tr></table></body></html>

# Related Work

Fine-tuning Attacks. Fine-tuning-as-a-service is an emerging offering that has been adopted by numerous service providers of LLMs, such as OpenAI, Mistral, and Zhipu AI. This innovative business model enables users to upload their specific data to the service platform, which is then applied to customize the provider’s LLMs to better meet individual requirements (Huang et al. 2024b). These LLMs are typically aligned with safety standards through methods like Reinforcement Learning from Human Feedback (RLHF; Christiano et al. 2017; Ouyang et al. 2022) or direct preference optimization (DPO; Rafailov et al. 2024) to align them with human values. Despite these efforts, safety alignment remains delicate and vulnerable to fine-tuning attacks. Such attacks can undermine a model’s resistance to harmful instructions by introducing malicious content into the task-specific data during fine-tuning (Yang et al. 2023; Shu et al. 2023; Wan et al. 2023). Remarkably, fine-tuning with as few as 100 malicious examples can lead these safety-aligned LLMs to adapt to harmful tasks while maintaining their overall performance (Yang et al. 2023).

LLM Safety Safeguards. To mitigate safety degradation caused by harmful fine-tuning, methods like Vlguard (Zong et al. 2024; Huang et al. 2024c) and Lisa (Huang et al. 2024c) merge preference data into task-specific datasets, preserving the model’s safety defenses by optimizing both task-level and alignment objectives. Constrained-SFT (Qi et al. 2024) improves robustness against fine-tuning attacks by constraining updates to the initial tokens. However, these approaches interfere with the downstream fine-tuning process by either incorporating preference data or altering the objective function during fine-tuning. Alternative methods, such as Vaccine (Huang, Hu, and Liu 2024) and RepNoise (Rosati et al. 2024), introduce perturbations to fortify models against harmful instructions from unseen user data. SafeLoRA (Hsu et al. 2024) realigns safety by mapping LoRA weights from the safe aligned region to the fine-tuned model. However, updating entire layers for safety realignment potentially overlooks neurons that are relevant to the fine-tuning task. Unlike Huang et al. (2024a), who remove safety-critical neurons without considering their relevance to downstream tasks, our approach restores the functionality of these neurons to balance safety and task performance.

Knowledge Neurons. The concept of knowledge neurons offers insights into model behavior by identifying neurons whose activations correlate with specific outputs (Dai et al. 2022; Niu et al. 2024). Neuron-level pruning methods have been developed to identify task-critical neurons. For instance, SNIP (Lee, Ajanthan, and Torr 2018) calculates the importance scores of all neurons based on their contribution to the loss, while Wanda (Sun et al. 2024) tracks changes in the immediate outputs of each layer when specific neurons are pruned. Regarding safety neurons, Chen et al. (2024) introduce inference time activation contrasting to locate safety neurons, highlighting their sparse distribution. However, Wei et al. (2024) reveal that freezing safety-critical neurons alone does not fully protect against fine-tuning attacks. Building on these insights, our method focuses on realigning neuron functionality to mitigate safety risks while preserving task-specific capabilities.

# Conclusion

Fine-tuning-as-a-service is a burgeoning offering that enables users to upload their data to tailor models to their specific needs. However, fine-tuning a securely aligned model on task-specific data can introduce safety risks, particularly when it contains a small number of harmful instructions. To tackle this challenge, we propose a neuron-level safety realignment framework without the need for additional training. Unlike methods that incorporate extra alignment objectives during fine-tuning, our approach does not disrupt the task-specific optimization process. We construct a superaligned reference model based on the initial aligned model, which we use to identify safety-critical neurons. The regions formed by these neurons serve a dual function: they enable us to assess the degree of safety degradation caused by dissimilarity before and after fine-tuning and they act as corrective patches for regions where significant safety damage has occurred. This neuron-level restoration facilitates safety realignment while upholding the model’s performance on downstream tasks.