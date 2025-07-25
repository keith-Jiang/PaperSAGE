# Hierarchical Divide-and-Conquer for Fine-Grained Alignment in LLM-Based Medical Evaluation

Shunfan Zheng1, Xiechi Zhang1, Gerard de Melo2,3, Xiaoling Wang1, Linlin Wang1\*

1 East China Normal University 2 Hasso Plattner Institute 3 University of Potsdam {sfzheng, 51255901060}@stu.ecnu.edu.cn, {xlwang, llwang} $@$ cs.ecnu.edu.cn, demelo@uni-potsdam.de

# Abstract

In the rapidly evolving landscape of large language models (LLMs) for medical applications, ensuring the reliability and accuracy of these models in clinical settings is paramount. Existing benchmarks often focus on fixed-format tasks like multiple-choice QA, which fail to capture the complexity of real-world clinical diagnostics. Moreover, traditional evaluation metrics and LLM-based evaluators struggle with misalignment, often providing oversimplified assessments that do not adequately reflect human judgment. To address these challenges, we introduce HDCEval, a Hierarchical Divideand-Conquer Evaluation framework tailored for fine-grained alignment in medical evaluation. HDCEval is built on a set of fine-grained medical evaluation guidelines developed in collaboration with professional doctors, encompassing Patient Question Relevance, Medical Knowledge Correctness, and Expression. The framework decomposes complex evaluation tasks into specialized subtasks, each evaluated by expert models trained through Attribute-Driven Token Optimization (ADTO) on a meticulously curated preference dataset. This hierarchical approach ensures that each aspect of the evaluation is handled with expert precision, leading to a significant improvement in alignment with human evaluators.

Models and supplementary materials: — https://huggingface.co/collections/AAAzsf/hdceval6762cda19a07c157778aa22d

# 1 Introduction

With the rapid development of large language models (LLMs) in the medical field, a range of advanced medical LLMs have been developed. However, the reliability and effectiveness of these models must be rigorously evaluated to ensure accurate and safe clinical decisions.

However, existing benchmarks such as MT-Bench (Zheng et al. 2024) and MedBench (Cai et al. 2024) are often limited to tasks in fixed formats such as multiple-choice question answering (QA), as shown in Figure 1, lacking clinical freestyle generation, which does not align with the actual clinical diagnostic process. Moreover, current evaluation metrics fail to provide comprehensive evaluation results, instead offering only simplistic assessments. For instance, traditional $\mathfrak { n }$ -gram metrics like ROUGE (Lin 2004)

Multiple-choice QA Crude Label   
Select the correct option: {question} Correct option: B {options} A,B,C.,D...

Freestyle Generation Fine-grained Label Different medical scenarios 3 main aspects 10 sub aspects Medical REL Context Commonsense Report Awareness Clinic Summary Reasoning Imaging COR Factuay Knowledge Examination QA EXP Clarity of Response

and BERT-based semantic similarity metrics (Zhang et al. 2019) yield only a single value, devoid of specific logical explanations.

LLMs can serve as evaluators (Fu et al. 2024; Kocmi and Federmann 2023) in such freestyle contexts due to their generative capabilities. Unlike traditional metrics, LLMs can offer more nuanced and context-aware assessments by generating detailed feedback and explanations. This allows them to better reflect complex scenarios, such as those found in clinical diagnostics. However, existing LLM evaluators often exhibit misalignment with human evaluators in medical evaluation. For instance, evaluation using GPT-4 or the open-source model PandaLM (Wang et al. 2023c) can inadvertently perpetuate or even amplify existing bias in the training data, leading to skewed and inconsistent assessments (Stureborg, Alikaniotis, and Suhara 2024; Wang et al. 2023b) that may not accurately reflect diverse patient populations or medical scenarios compared to human physicians.

To address the issues above, we first collaborate with professional doctors to propose a set of fine-grained medical evaluation guidelines tailored for detailed medical assessments. These guidelines include three primary aspects: Patient Question Relevance (REL), Medical Knowledge Correctness (COR), and Expression (EXP), each further subdivided into specific sub-aspects.

Based on the guidelines, we introduce HDCEval, a hierarchical divide-and-conquer evaluation framework that consists of two main components. Firstly, the Divide component involves a hierarchical decomposition of the evaluation task. The process begins by dividing the complex evaluation into multiple primary tasks. Each primary task is then further subdivided into more detailed subtasks. For each primary task and its corresponding subtasks, we employ a specialized expert model to carry out the evaluation, ensuring precise and expert-aligned assessments. This is in contrast to the BSM method (Saha et al. 2023), which relies on a single, non-specialized model to handle all primary tasks.

In the Conquer component, the framework leverages a carefully constructed preference dataset, which is specifically designed to improve alignment with human evaluators. This dataset plays a crucial role in enhancing the performance of each expert model. Based on this dataset, we introduce the Attribute-Driven Token Optimization (ADTO) method for training. This method incorporates reward tokens that guide the optimization of different expert models, ensuring that each model aligns with the specific evaluation criteria of its assigned tasks, thereby enhancing the precision and quality of the overall evaluation. The experimental results demonstrate that HDCEval significantly outperforms existing baseline methods across various medical scenarios. Notably, compared to the PandaLM evaluator, HDCEval achieves an overall improvement in consistency with human evaluations by $2 3 . 9 2 \%$ . This highlights the effectiveness of the Hierarchical Divide-and-Conquer Evaluation Framework in aligning model evaluations with expert-level assessments in the medical domain.

Our key contributions can be summarized as follows:

• We propose a comprehensive set of fine-grained medical evaluation guidelines developed in collaboration with professional doctors. • We introduce HDCEval, a hierarchical divide-andconquer evaluation framework designed for detailed and accurate medical evaluations, achieving finer-grained evaluation that better aligns with human evaluators. • We develop and apply the Attribute-Driven Token Optimization (ADTO) strategy, demonstrating that HDCEval surpasses other baselines in accuracy and alignment with human evaluators in freestyle medical contexts.

# 2 Methodology

# Task Formulation

In evaluation tasks, the input $x$ consists of a question $q$ and the model’s response $r$ . The goal is to generate an evaluation result $E$ composed of multiple dimensions. In our medical evaluation tasks, the final evaluation consists of $m$ distinct dimensions, denoted as $E = \{ E _ { 1 } , \dots , E _ { m } \}$ , where each $E _ { i }$ represents the assessment of a specific dimension. Each $E _ { i }$ is defined as a tuple

$$
E _ { i } = { \left( s _ { i } , p _ { i } \right) } ,
$$

where $s _ { i }$ is the scoring of the response on dimension $i$ , and $p _ { i }$ is the corresponding rationale explaining the reasoning process.

# Fine-grained Medical Evaluation Guidelines

Achieving accurate and nuanced evaluations is crucial in clinical diagnostics to ensure patient safety and effective treatment. To address this, we collaborated with medical experts to develop detailed evaluation guidelines specifically designed for medical assessments. These guidelines emphasize three primary aspects:

• Patient Question Relevance (REL): This aspect considers how well the medical response addresses the patient’s specific questions and concerns. It involves assessing the clarity, directness, and appropriateness of the response in relation to the patient’s query.   
• Medical Knowledge Correctness (COR): This aspect ensures the accuracy of the medical information provided. It involves evaluating whether the response aligns with current medical knowledge, guidelines, and evidence-based practices.   
• Expression (EXP): This aspect focuses on the clarity and coherence of the response, assessing the language, structure, and presentation of the information to ensure it is easily understandable and professional.

Each primary aspect is further divided into 3-4 sub-aspects to capture the intricacies of medical evaluations thoroughly. For instance, Patient Question Relevance (REL) includes sub-aspects such as Relevance to Patient’s Condition (COND), which assesses how directly the response pertains to the patient’s specific medical condition. Medical Knowledge Correctness (COR) encompasses sub-aspects like Factual Accuracy (ACC), ensuring the information aligns with current evidence-based practices. These sub-aspects provide a granular framework for evaluation, ensuring comprehensive coverage of each aspect. For each sub-aspect, scores range from 0 to 5, with detailed scoring rules provided in the Technical Appendix within supplementary materials.

# Hierarchical Divide-and-Conquer Evaluation Framework

Overview As shown in Figure 3, the Hierarchical Divideand-Conquer Evaluation Framework tackles medical evaluations by first dividing the task into detailed, expert-focused subtasks. Then, it conquers these tasks using preference data and Attribute-Driven Token Optimization (ADTO) to refine the model. This method ensures thorough and precise alignment with medical evaluation standards.

Hierarchical Divide Our medical evaluation guidelines are inherently multi-dimensional and strictly constrained, making accurate assessment a challenging task. LLMs often struggle with completing nuanced guideline-based estimation tasks due to their generalized training and lack of fine-tuned specialization.

To address these challenges, we propose a hierarchical divide-and-conquer approach, inspired by BSM (Saha et al. 2023). BSM’s methodology demonstrates the efficacy of decomposing complex evaluation tasks into manageable subtasks that can be addressed in parallel. However, BSM’s approach relies on a single model for all subtasks, which lim

Data Prestrection目 (a) Hierarchical Co ket ribti-Drivoen Context Awareness   
Swap scores Single Task: J 六 C 宫 Meeg O Fineger ie God geBE Rational Score REL The score of Responsel: $s _ { \mathbf { 1 } }$ $s _ { 2 }$ Expert Backpropagation Factual Others (eg. rational) J1 。 Accuracy T1,J² Negative @ （） 日 J2 Good & Bad COR Rational Score The score f esponse: s2 J 自 CORt reward token Sub-divide Router 。 Expression   
Add or sub scores Jm 8 Integrity   
Swap rational T X   
Remove human references Divide Exprt Good &r Bad EXP Rational Score

its its ability to achieve fine-grained alignment with human evaluators.

In contrast, our framework enhances this approach by using specialized expert models for different aspects of the evaluation. We first decompose the overarching evaluation task $\tau$ into $n$ primary evaluation tasks $\mathcal { T } _ { 1 } , \ldots , \mathcal { T } _ { n }$ , each aligned with an expert model. These primary tasks are further subdivided into subtasks to capture the intricacies of the evaluation criteria.

The hierarchical decomposition is structured as follows:

$$
\left\{ \begin{array} { l l } { { \mathcal { T } } = { \mathcal { T } } _ { 1 } ( x , I _ { 1 } ) , \ldots , { \mathcal { T } } _ { m } ( x , I _ { m } ) } \\ { { \mathcal { T } } _ { i } ( x , I _ { i } ) = { \mathcal { T } } _ { i } ^ { 1 } ( x , I _ { i } , I _ { i } ^ { 1 } ) , \ldots , { \mathcal { T } } _ { i } ^ { c } ( x , I _ { i } , I _ { i } ^ { c _ { i } } ) } \end{array} \right.
$$

Here, $I _ { i }$ represents the instruction of primary evaluation tasks $\mathcal { T } _ { i }$ and $I _ { i } ^ { j }$ represents the instruction of subtasks $\mathcal { T } _ { i } ^ { j }$ .

Each expert model is trained specifically for one primary aspect (including its associated sub-aspects) to ensure fine-grained and accurate alignment with medical evaluation guidelines. This specialization allows for more precise evaluations that closely align with human expertise. Our hierarchical approach effectively manages the complexity of medical evaluations, ensuring a detailed and accurate assessment of each aspect.

Preference Data Construction To enhance model alignment with human evaluators in medical assessments, we develop a preference dataset that specifically targets misalignment and bias. This dataset construction is intricately linked to our fine-grained evaluation guidelines, ensuring that model improvements align with detailed evaluation criteria. The negative samples are constructed from existing positive samples, and this process is represented by the following formula:

• Swapping Scores of Two Responses: Let $R _ { 1 }$ and $R _ { 2 }$ be two responses with scores $S ( R _ { 1 } )$ and $S ( R _ { 2 } )$ from a positive sample. The scores of the corresponding negative sample are swapped:

$$
S ^ { \prime } ( R _ { 1 } ) = S ( R _ { 2 } ) , S ^ { \prime } ( R _ { 2 } ) = S ( R _ { 1 } )
$$

This method forces the model to determine which response better addresses the patient’s query, refining its ability to assess relevance.

• Simultaneously Adding or Subtracting Scores from Two Responses: Considering two responses $R _ { 1 }$ and $R _ { 2 }$ with scores $S ( R _ { 1 } )$ and $S ( R _ { 2 } )$ , we adjust the scores by a constant $\Delta S$ :

$$
S ^ { \prime } ( R _ { 1 } ) = S ( R _ { 1 } ) + \Delta S , \quad S ^ { \prime } ( R _ { 2 } ) = S ( R _ { 2 } ) - \Delta S
$$

This technique helps the model differentiate between high-quality and low-quality responses by teaching it to discern changes in accuracy and presentation.

• Exchanging Rationales of Two Responses: Let $R _ { 1 }$ and $R _ { 2 }$ be two responses with corresponding rationales $P ( R _ { 1 } )$ and $P ( R _ { 2 } )$ . We swap their rationales:

$$
P ^ { \prime } ( R _ { 1 } ) = P ( R _ { 2 } ) , \quad P ^ { \prime } ( R _ { 2 } ) = P ( R _ { 1 } )
$$

This method ensures that the model’s explanations align with its judgments, thereby reducing logical inconsistencies.

• Removing Human-Provided Reference Information: Let $E _ { h }$ represent an evaluation result that includes human-provided reference information $I _ { h }$ . The humanprovided information is removed from the evaluation result:

$$
E _ { h } ^ { \prime } = E _ { h } \setminus I _ { h }
$$

This strategy reinforces the importance of humanprovided information, allowing the model’s outputs to better align with human expectations.

Input: $\mathcal { D } = \{ ( x _ { j } , y _ { j , w } , y _ { j , l } ) \} _ { j = 1 } ^ { N }$ , multi-dimensional evaluation data with positive $( y _ { w } )$ and negative $( y _ { l } )$ examples, $\mathcal { M } = \{ M _ { 1 } , M _ { 2 } , . . . , M _ { m } \}$ , set of fine-grained evaluation models, $\mathcal { R } = \{ R _ { \mathrm { R E L } } , R _ { \mathrm { C O R } } , R _ { \mathrm { E X P } } \}$ , set of reward tokens for relevance, correctness, and expression.

Output: Fine-Grained Medical Evaluation Models $\mathcal { M }$ .

1: Initialization: Set model parameters $\theta _ { i }$ for each $M _ { i } \in$ $\mathcal { M }$ .   
2: for each $M _ { i }$ in $\mathcal { M }$ do   
3: for each training step $t$ do   
4: Sample a mini-batch $( x _ { j } , y _ { j , w } , y _ { j , l } )$ from $\mathcal { D }$   
5: Determine aspect $a \in \{ \mathrm { R E L } , \mathrm { C O R } , \mathrm { E X P } \}$ for $M _ { i }$   
6: Construct reward token $r = R _ { a }$   
7: Create inputs $z _ { w } = \mathrm { c o m b i n e } ( x _ { j } , r , y _ { j , w } )$ and $z _ { l } =$ combine $( x _ { j } , r , y _ { j , l } )$   
8: Compute model outputs $o _ { j , w } ~ = ~ M _ { i } ( z _ { w } ; \theta _ { i } )$ and $o _ { i , l } = M _ { i } ( z _ { l } ; \theta _ { i } )$   
9: Compute loss $\mathcal { L } ( o _ { j , w } , o _ { j , l } )$ using Attribute-Driven Token Optimization   
10: Freeze unrelated layers to stabilize training   
11: Update other parameters $\theta _ { i }$ via gradient descent: $\theta _ { i } \gets \theta _ { i } - \eta \nabla _ { \theta _ { i } } \mathcal { L } \big ( o _ { j , w } , o _ { j , l } \big )$   
12: end for   
13: end for

14: return

Attribute-Driven Token Optimization To further reduce the bias of the evaluator, and improve the alignment between the models and professional physicians, we introduce the Attribute-Driven Token Optimization (ADTO) method.

The ADTO method leverages preference datasets by embedding specific reward tokens within the training data. These tokens represent different aspects of evaluation quality, and guide the model in distinguishing between superior and inferior responses. The integration of reward tokens enables the model to learn from nuanced distinctions that are critical in professional medical assessments.

For each $i$ -th primary aspect evaluation model in our framework, the optimization process is designed to balance the current policy model’s responses with those of a reference model. This process is mathematically formulated in Eq. 7 with respect to training model parameters $\pi _ { \theta } ^ { i }$ , reference model parameters $\pi _ { r e f } ^ { i }$ , and hyperparameter $\beta _ { i }$ as

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { A D T O } } ^ { i } ( \pi _ { \theta } ^ { i } ; \pi _ { \mathrm { r e f } } ^ { i } ) = - \mathbb { E } _ { ( x , y _ { w } ^ { i } , y _ { l } ^ { i } ) \sim \mathcal { D } } \Bigg [ } \\ & { \quad \quad \quad \quad \log \sigma \left( \beta _ { i } \log \frac { \pi _ { \theta } ^ { i } \left( y _ { w } ^ { i } \mid x , t _ { w } ^ { i } , I _ { i } \right) } { \pi _ { \mathrm { r e f } } ^ { i } \left( y _ { w } ^ { i } \mid x , t _ { w } ^ { i } , I _ { i } \right) } \right. } \\ & { \quad \quad \quad \quad \left. - \beta _ { i } \log \frac { \pi _ { \theta } ^ { i } \left( y _ { l } ^ { i } \mid x , t _ { l } ^ { i } , I _ { i } \right) } { \pi _ { \mathrm { r e f } } ^ { i } \left( y _ { l } ^ { i } \mid x , t _ { l } ^ { i } , I _ { i } \right) } \right) \Bigg ] , } \end{array}
$$

where $( x , y _ { w } ^ { i } , y _ { l } ^ { i } )$ refers to the triplet of (input, good evaluation, bad evaluation), and $t _ { w } ^ { i } , t _ { l } ^ { i }$ represent different reward tokens in optimization. Here, $\pi _ { \theta } ^ { i } ( y _ { w } ^ { i } \mid x , t _ { w } ^ { i } , I _ { i } )$ denotes the cumulative probability of the current policy model generating good responses, while $\pi _ { \mathrm { r e f } } ^ { i } \big ( y _ { l } ^ { i } \big | ^ { * } x , t _ { l } ^ { i } , I _ { i } \big )$ represents the cumulative probability of the reference model generating bad responses. $\sigma$ denotes the sigmoid function. Then, we integrate the optimization processes of all $m$ models within the framework. Further details are specified in Algorithm 1.

The factual knowledge within large language models is often injected into deeper layers. Therefore, to equip the model with more accurate and objective evaluation capabilities while reducing the computational cost, we freeze the first 24 layers of our model and only train the last eight layers.

# 3 Experiments

# Experimental Setup

Our evaluation models are based on the MedLlama2-7B model. We train using a batch size of 128 and a maximum token length of 4,096 on 4 NVIDIA A100-80GB GPUs. To maximize GPU memory usage and accelerate training, we employed the Fully Sharded Data Parallel (Zhao et al. 2023) strategy and the FlashAttention (Dao et al. 2022) algorithm. The learning rates for the instruction tuning and direct preference optimization phases are set to 2×10−5 and 5×10−7, respectively. During inference, we use greedy decoding with a temperature of 0 to minimize randomness.

# Medical Dataset

Data Source First, we integrate medical questions from different sources including medical meadow wikidoc1, MedBench (Cai et al. 2024), MedText2, and MedDialog (Zeng et al. 2020). We perform automated and manual filtering to ensure reliable and safe medical question sources. Then, to diversify the task types of the data and conform to the clinical medical scenario, we divide the data into five specific medical scenarios shown in Figure 2.

Dataset Construction First, we use four different medical models: ChatDoctor, Baize, MedAlpaca, and MedLlama2, to generate responses to the questions. Then, with the assistance of AI, we annotated the 13,452 samples following the Guidelines Instructions. More details about the dataset construction are provided in the Technical Appendix within supplementary materials.

Dataset Validation To validate the effectiveness of our dataset, we use the publicly available MedMCQA dataset (Pal, Umapathi, and Sankarasubbu 2022) as a reference. We evaluated four models on both datasets and calculated their rankings3. The results show consistent rankings of the models across the two datasets, with ChatDoctor demonstrating the best performance. Additionally, we find that ChatDoctor exhibits the strongest ability to follow instructions.

Table 1: Fine-grained evaluation results. We run models three times and report the average results. \* represents a significant difference with our results or significant correlation with human evaluation (t-test, $p$ -value $< 0 . 0 0 1 \dot { } ,$ ), while $\dagger$ and $\ S$ refer to t-test with $p { < } 0 . 0 1$ and $p { < } 0 . 0 5$ , respectively.   

<html><body><table><tr><td rowspan="2">MedicalScenarios</td><td colspan="3">Pairwise Accuracy (%)</td><td colspan="3">Reference Match (%)</td><td colspan="2">Correlation</td></tr><tr><td>REL</td><td>COR</td><td>EXP</td><td>REL</td><td>COR</td><td>EXP</td><td>Pearson</td><td>ICC</td></tr><tr><td>Imaging Examination (Text)</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1</td><td></td></tr><tr><td>MedLlama2</td><td>61.54*</td><td>52.38*</td><td>60.91*</td><td>60.72*</td><td>50.24*</td><td>60.86*</td><td>0.5484*</td><td>0.5893*</td></tr><tr><td>PandaLM</td><td>54.15*</td><td>50.53*</td><td>50.53*</td><td>55.84*</td><td>48.49*</td><td>53.71*</td><td>0.5604*</td><td>0.6141*</td></tr><tr><td>ChatGPT</td><td>69.23†</td><td>64.10†</td><td>55.77*</td><td>70.12†</td><td>64.40†</td><td>75.97*</td><td>0.5813†</td><td>0.6693†</td></tr><tr><td>GPT-4</td><td>84.87*</td><td>61.54t</td><td>71.15*</td><td>80.74*</td><td>69.46†</td><td>88.41*</td><td>0.5898†</td><td>0.6849†</td></tr><tr><td>Ours (HDCEval)</td><td>84.87*</td><td>79.49*</td><td>75.00*</td><td>78.78*</td><td>70.03*</td><td>92.98*</td><td>0.6480*</td><td>0.7149*</td></tr><tr><td>Clinic Reasoning</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MedLlama2</td><td>57.14*</td><td>48.83*</td><td>56.67*</td><td>59.72*</td><td>51.43*</td><td>56.32*</td><td>0.4060*</td><td>0.5247*</td></tr><tr><td>PandaLM</td><td>50.61*</td><td>46.12*</td><td>47.29*</td><td>56.67*</td><td>39.25*</td><td>51.62*</td><td>0.3637*</td><td>0.4919*</td></tr><tr><td>ChatGPT</td><td>64.63†</td><td>64.638</td><td>59.69†</td><td>66.56†</td><td>64.748</td><td>67.25t</td><td>0.54838</td><td>0.71018</td></tr><tr><td>GPT-4</td><td>78.91*</td><td>69.18†</td><td>60.20*</td><td>78.91†</td><td>70.44t</td><td>75.13†</td><td>0.5599†</td><td>0.7209t</td></tr><tr><td>Ours (HDCEval)</td><td>82.99*</td><td>67.35*</td><td>67.35*</td><td>88.23*</td><td>82.50*</td><td>85.81*</td><td>0.5887*</td><td>0.7209*</td></tr><tr><td colspan="9"> Knowledge QA</td></tr><tr><td>MedLlama2</td><td>56.67*</td><td>47.08*</td><td>52.61*</td><td>56.67*</td><td>50.73*</td><td>54.52*</td><td>0.5240*</td><td>0.6917*</td></tr><tr><td>PandaLM</td><td>40.53*</td><td>39.18*</td><td>41.55*</td><td>48.65*</td><td>40.11*</td><td>45.74*</td><td>0.5181*</td><td>0.6469*</td></tr><tr><td>ChatGPT</td><td>63.33*</td><td>71.11*</td><td>58.33*</td><td>68.35*</td><td>72.01*</td><td>70.57†</td><td>0.5603*</td><td>0.6519*</td></tr><tr><td>GPT-4</td><td>76.11*</td><td>66.11*</td><td>61.67*</td><td>79.86*</td><td>73.67*</td><td>73.86*</td><td>0.5632*</td><td>0.6656*</td></tr><tr><td>Ours (HDCEval)</td><td>85.00*</td><td>73.33*</td><td>74.17*</td><td>86.78*</td><td>78.41*</td><td>81.85*</td><td>0.5693*</td><td>0.7073*</td></tr><tr><td colspan="9">Report Summary</td></tr><tr><td>MedLlama2</td><td>60.85*</td><td>58.86*</td><td>61.28*</td><td>61.63*</td><td>58.36*</td><td>62.96*</td><td>0.4303*</td><td>0.5595*</td></tr><tr><td>PandaLM</td><td>58.47*</td><td>45.20*</td><td>62.07*</td><td>62.79*</td><td>50.19*</td><td>63.46*</td><td>0.3947*</td><td>0.5342*</td></tr><tr><td>ChatGPT</td><td>72.13†</td><td>66.24*</td><td>69.68*</td><td>77.01*</td><td>67.66†</td><td>73.72*</td><td>0.5864†</td><td>0.6936†</td></tr><tr><td>GPT-4</td><td>74.88*</td><td>70.10*</td><td>70.41*</td><td>77.06†</td><td>68.84*</td><td>75.71*</td><td>0.5905*</td><td>0.6948*</td></tr><tr><td>Ours (HDCEval)</td><td>75.24*</td><td>72.76*</td><td>70.03*</td><td>77.62*</td><td>72.31*</td><td>73.75*</td><td>0.5913*</td><td>0.7047*</td></tr><tr><td colspan="9">Medical Commonsense</td></tr><tr><td>MedLlama2</td><td>58.82*</td><td>49.41*</td><td>56.73*</td><td>61.88*</td><td>53.92*</td><td>56.13*</td><td>0.3609*</td><td>0.4923*</td></tr><tr><td>PandaLM</td><td>57.05*</td><td>41.53*</td><td>52.71*</td><td>60.57*</td><td>43.63*</td><td>54.12*</td><td>0.3256*</td><td>0.4507*</td></tr><tr><td>ChatGPT</td><td>70.59*</td><td>68.55*</td><td>61.77*</td><td>71.41*</td><td>72.94*</td><td>68.33†</td><td>0.5815*</td><td>0.7238*</td></tr><tr><td>GPT-4</td><td>72.55*</td><td>68.63*</td><td>79.41*</td><td>74.25†</td><td>76.91*</td><td>69.88*</td><td>0.5954*</td><td>0.7236*</td></tr><tr><td>Ours (HDCEval)</td><td>74.51*</td><td>70.59*</td><td>77.94*</td><td>76.58*</td><td>88.35*</td><td>71.50*</td><td>0.6767*</td><td>0.7881*</td></tr></table></body></html>

# Baselines and Test Set

We selected representative models as baselines, including the closed-source models (ChatGPT and GPT-4) and the open-source models (PandaLM and MedLlama2).

For the test data, we initially extracted 2,994 samples from the constructed dataset to form the test set, with the remaining samples used as the training set. We then hired human doctors to annotate the test data. The annotation process follows the fine-grained medical evaluation guidelines.

# Evaluation

The generated evaluation results include both scores and rationales. Therefore, we need to assess these two aspects separately. For the scores, we employ automated metrics, while for the rationales, we rely on evaluations conducted by human doctors.

Human Evaluation The human annotators manually assess whether the rationale from the model matches the rationale from human-provided labels to verify the reasonableness of the model’s evaluation results. This process is referred to as Reference Match. If the label’s rationale indicates an error in the medical knowledge within the response, but the model fails to recognize it, it is considered a mismatch.

Automatic Metrics We use Pairwise Accuracy as the primary evaluation metric for the scores. If the relative ranking of the evaluation scores between the two responses generated by the model is consistent with the labels from human doctors, it indicates that the model accurately evaluated the quality of the two responses; otherwise, it does not. Additionally, we use the Intraclass Correlation Coefficient (Koo and Li 2016) and Pearson Correlation Coefficient (Cohen et al. 2009) to measure the similarity between the model evaluation and human evaluation.

# Main Results

Evaluation Metrics Results To demonstrate the capability of HDCEval in fine-grained medical assessment, we arranged for medical experts to evaluate the responses. Table 1 provides the fine-grained evaluation results of HDC

Table 2: Ablation Study on HDCEval Components – Assessing the impact of removing reward tokens and preference data on evaluation accuracy.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="3">REL</td><td colspan="3">COR</td><td colspan="4">EXP</td><td rowspan="2">AVG</td></tr><tr><td>CONT</td><td>COND</td><td>CONC</td><td>ACC</td><td>INFO</td><td>UNC</td><td>CLAR</td><td>LANG</td><td>TE</td><td>INTE</td></tr><tr><td>HDCEval</td><td>80.91</td><td>81.82</td><td>78.83</td><td>71.65</td><td>72.18</td><td>74.27</td><td>73.48</td><td>75.14</td><td>70.84</td><td>72.14</td><td>75.13</td></tr><tr><td>HDCEvalno-token</td><td>78.57</td><td>79.50</td><td>78.36</td><td>69.93</td><td>68.12</td><td>68.51</td><td>68.40</td><td>70.81</td><td>67.89</td><td>69.98</td><td>72.01</td></tr><tr><td>HDCEvalno-preference</td><td>80.70</td><td>80.97</td><td>77.94</td><td>71.35</td><td>69.57</td><td>71.29</td><td>71.36</td><td>69.80</td><td>67.13</td><td>71.17</td><td>73.13</td></tr></table></body></html>

HDCEval HDCEval Win Ratio (%) MedAlpaca Win REL COR EXP REL COR EXP Human Human Tie 」. DCohcatto-r Win L.1 REL COR EXP REL COR EXP (a) Clinic Reasoning (b) Knowledge QA

Eval compared to the baselines. From left to right, the results of three different metrics on fine-grained dimensions are included. We observe that HDCEval outperforms other models across multiple scenarios, especially outperforming GPT-4 on reference match and correlation metrics, which reflects better alignment with humans. Regarding pairwise accuracy metrics, it demonstrates a $2 3 . 9 2 \%$ improvement compared to PandaLM. From the fine-grained perspective, HDCEval significantly improves evaluation accuracy compared to other baselines in terms of Medical Knowledge Correctness (COR).

Win-Tie-Lose Experiment Results We compiled statistics on the win rates of HDCEval and humans in assessing the quality of response pairs from MedAlpaca and ChatDoctor across different scenarios. The results presented in Figure 4 demonstrate consistent agreement between HDCEval and human evaluators across various medical scenarios. This consensus leads to the conclusion that ChatDoctor is significantly more effective than MedAlpaca.

Double Blind Experiment Results As shown in Figure 5, across the three primary fine-grained evaluation dimensions, human doctors show a preference for HDCEval that is comparable to their preference for GPT-4. In comparison to PandaLM, human doctors consistently favor the evaluation results provided by HDCEval.

Ours vs. GPT-4 Ours vs. PandaLM   
REL 33.7% 37.5% 30.8% REL 48.4% 30.5% 21.1%   
COR 30.4% 41.3% 28.3% COR 62.3% 27.3%   
EXP 27.4% 45.6% 27.0% EXP 49.6% $3 3 . 7 \%$ / Percentage Percentage

# Ablation Study

To validate whether our training method can improve assessment ability, we conducted the ablation experiments in Table 2. Removing the reward token weakens the model’s perception of good and bad responses, resulting in a $3 . 7 5 \%$ decrease in evaluation accuracy. When preference data is excluded and only SFT is used for training, the evaluation accuracy drops by $0 . 5 \%$ , as ADTO better utilizes preference data to enhance performance further.

# 4 Discussion

# Effects of Different Input Forms

In practical applications, the format of evaluation tasks is not fixed. Therefore, we designed two evaluation tasks to explore HDCEval’s generalization ability across different input formats. One task simultaneously evaluates the quality of two responses, while the other task separately evaluates the two responses and then compares the results. The results in Figure 6 indicate that different input formats do not significantly affect the evaluation results of HDCEval.

# Exploration of Model Bias

In constructing the preference dataset in Section 2, we employed various strategies to mitigate model biases discussed in previous work (Zheng et al. 2024). For example, swapping the scores of two responses can mitigate position bias. To verify this, we conducted the experiments shown in Table 3, comparing the model bias with and without preference data during training. The results indicate that using preference data improves both position bias and verbosity bias.

HDCEval Human   
Win Ratio (%) TMiedAlpaca Win ChatDoctor Win REL COR EXP REL COR EXP (a) Pairwise Evaluation HDCEval Human   
Win Ratio (%) TMiedAlpaca Win ChatDoctor Win ， REL COR EXP REL COR EXP (b) Single Evaluation

<html><body><table><tr><td>Bias Type</td><td>CONT</td><td>COND</td><td>CONC</td></tr><tr><td>Position Bias</td><td></td><td></td><td></td></tr><tr><td>√PreferenceData</td><td>28.57</td><td>26.50</td><td>29.87</td></tr><tr><td>X Preference Data</td><td>32.91</td><td>31.82</td><td>32.91</td></tr><tr><td>Verbosity Bias</td><td></td><td></td><td></td></tr><tr><td>√Preference Data</td><td>28.17</td><td>27.43</td><td>26.29</td></tr><tr><td>XPreference Data</td><td>32.96</td><td>30.14</td><td>31.36</td></tr></table></body></html>

Table 3: Model bias for the three sub-aspects of patient question relevance (REL).

# Case Study

# ### Question:

What is Dermatophytosis and how can it be distinguished from other illnesses?

# ### Response 1:

Dermatophytosis must be differentiated from other diseases that cause skin lesions, such as other causes of dandruff, seborrheic dermatitis, psoriasis, and other causes of skin rash.

Relevance to Patient’s Condition (COND):   
Analysis: ....   
Score: 1   
Addressing Multiple Concerns (CONC): Analysis: ......   
Score: 1   
Evaluation of Response 2:   
Context Awareness (CONT):   
Analysis: This response addresses ...   
Score: 4   
Relevance to Patient’s Condition (COND):   
Analysis: .....   
Score: 3   
Criterion Addressing Multiple Concerns (CONC):   
Analysis: ......   
Score: 4

# ### Response 2:

Dermatophytosis is a fungal infection of the skin, hair, and nails. It is caused by several species of fungi, including ...

# ### HDCEval:

# Evaluation of Response 1:

Context Awareness (CONT): Analysis: The response provides a general statement about the need to differentiate Dermatophytosis, but it does not ... Score: 1

### Human Annotator: R1 only extracts the key point of the question “Dermatophytosis”, but does not explain what “Dermatophytosis” is, ...

As the above text-box shows, our model evaluates two responses based on detailed criteria sequentially. During the evaluation of each criterion, our model first analyzes each response according to the current criteria and ultimately assigns a score. The evaluation results generated by our model indicate that the first medical LLM’s response is inferior to the second medical LLM’s response across all three detailed criteria, which is corroborated by the human annotator’s evaluation.

# 5 Related Work

Automated Model Evaluation Many researchers employ machine learning and NLP techniques to automatically evaluate responses from medical large language models. Some traditional metrics such as BLEU (Papineni et al. 2002) and ROUGE (Lin 2004) assess the quality of candidate text by statistically comparing n-grams between candidate and reference texts. However, these metrics are limited to the lexical level, disregarding much of the semantic information (Freitag et al. 2022).

In contrast, using BERT (Devlin et al. 2018) to assess the semantic similarity between candidate and reference embeddings is more reasonable (Zhang et al. 2019; Zhao et al. 2019). However, it can only provide a numerical value and cannot offer more logical explanations (Wang, Cho, and Lewis 2020; Huang et al. 2020), which can lead to a lack of credibility in evaluating medical models and misalignment with humans (Mehri and Eskenazi 2020; Zhong et al. 2022). Furthermore, existing benchmarks such as MT-Bench (Zheng et al. 2024) for evaluating the consistency between LLMs and human preferences, and MedBench (Cai et al. 2024) for medical domain evaluation, often employ fixedform tasks such as multiple-choice questions, making it challenging to achieve evaluation in freestyle contexts.

LLM-Based Evaluators With the rapid advancement of large language models (LLMs) possessing powerful text comprehension and reasoning capabilities, recent research has seen the emergence of LLM-based evaluators (Fu et al. 2024; Wang et al. $2 0 2 3 \mathrm { a }$ ; Chen et al. 2023). They employ LLMs to assess text quality through methods such as prompting. For instance, utilizing models such as ChatGPT and GPT-4 in conjunction with specific prompting templates has enabled automated evaluation with some degree of success (Wu et al. 2023; Nori et al. 2023). However, models like GPT-4 are general-purpose and not specialized for specific evaluation tasks, thus exhibiting certain bias compared to humans (Wang et al. 2023b; Wu and Aji 2023). In contrast, open-source models like PandaLM (Wang et al. 2023c) are dedicated to evaluation tasks, but the medical domain requires rich specialized knowledge, which PandaLM lacks to some extent. In contrast to existing research, our work aims to produce fine-grained evaluation results from LLMs that align well with medical experts.

# 6 Conclusion

In this paper, we introduce HDCEval, a hierarchical divideand-conquer evaluation framework specifically designed for evaluating medical language models. By dividing complex evaluation tasks into specialized subtasks and using expert models, HDCEval achieves greater alignment with human judgments and addresses the limitations of existing benchmarks and metrics. Our experiments demonstrate that HDCEval significantly outperforms baseline methods, improving consistency with human evaluations by $2 3 . 9 2 \%$ . This framework offers a more accurate, detailed, and reliable approach to assessing medical models, contributing to more effective clinical decision-making.