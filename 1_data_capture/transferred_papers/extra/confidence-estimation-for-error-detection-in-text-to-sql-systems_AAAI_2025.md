# Confidence Estimation for Error Detection in Text-to-SQL Systems

Oleg Somov1,2 and Elena Tutubalina1,3,4

1AIRI, Moscow, Russia 2MIPT, Dolgoprudny, Russia 3Sber AI, Moscow, Russia 4ISP RAS Research Center for Trusted Artificial Intelligence, Moscow, Russia somov $@$ airi.net, tutubalina $@$ airi.net

# Abstract

Text-to-SQL enables users to interact with databases through natural language, simplifying the retrieval and synthesis of information. Despite the success of large language models (LLMs) in converting natural language questions into SQL queries, their broader adoption is limited by two main challenges: achieving robust generalization across diverse queries and ensuring interpretative confidence in their predictions. To tackle these issues, our research investigates the integration of selective classifiers into Text-to-SQL systems. We analyse the trade-off between coverage and risk using entropy based confidence estimation with selective classifiers and assess its impact on the overall performance of Text-to-SQL models. Additionally, we explore the models’ initial calibration and improve it with calibration techniques for better model alignment between confidence and accuracy. Our experimental results show that encoder-decoder T5 is better calibrated than in-context-learning GPT 4 and decoder-only Llama 3, thus the designated external entropy-based selective classifier has better performance. The study also reveal that, in terms of error detection, selective classifier with a higher probability detects errors associated with irrelevant questions rather than incorrect query generations.

How many books do we Good case have in store?   
n Select count(boocks)from Store What last book Low-generalization wassold?   
n selecterr How much did we pay Unanswerable for rent last month?   
n Select sum(price) from books

Code — https://github.com/runnerup96/error-detection-intext2sql Extended version — https://arxiv.org/abs/2501.09527

# 1 Introduction

Text-to-SQL parsing (Zelle and Mooney 1996; Zettlemoyer and Collins 2005) aims at converting a natural language (NL) question to its corresponding structured query language (SQL) in the context of a relational database (schema). To effectively utilize Text-to-SQL models, users must clearly understand the model’s capabilities, particularly the range of questions it can accurately respond to. In this context, generalization ability is crucial for ensuring accurate SQL query generation, while interpretative trustworthiness is essential for minimizing false positives—instances where the model generates incorrect SQL queries that might be mistakenly perceived as correct.

Our work explores how model uncertainty estimates can be leveraged to detect erroneous predictions (Fig. 1). We focus on scenarios where models struggle to generalize, such as low compositional generalization, where the model cannot form novel structures not seen in the training set, or low domain generalization, where the model fails to adapt to new schema elements or novel databases. We categorize these errors as low-generalization. Additionally, we address cases where questions are unanswerable by the underlying database or require external knowledge, which we refer to as unanswerable. Typically, models are trained on questionSQL pairs, so encountering such samples is considered outof-distribution (OOD), and the model should ideally avoid generating a query. Both types of errors result in incorrect responses, whether as non-executable SQL queries or executable queries that return false responses.

To study compositional and domain generalization in Text-to-SQL, several benchmarks and datasets have been developed over the years to better approximate real-world scenarios, address various aspects of model performance: complex queries involving join statements across multiple tables (Yu et al. 2018), new and unseen database schemas (Gan, Chen, and Purver 2021; Lee, Polozov, and Richardson 2021), compositional train and test splits (Shaw et al. 2021; Finegan-Dollak et al. 2018), robustness test sets (Bakshandaeva et al. 2022; Chang et al. 2023), dirty schema values and external knowledge requirements (Li et al. 2024; Wretblad et al. 2024), domain-specific datasets that feature unanswerable questions (Lee et al. 2022). To sum up, most prior work evaluates either different types of generalization, noisy data or model uncertainties.

In this paper, we ask: can we identify error generations in Text-to-SQL LMs under distribution shift using uncertainty estimation? Specifically, we examine this from the point of view of error detection and calibration, examining if the models’ probability estimates align accurately with the actual correctness of the answers. We apply T5 (Raffel et al. 2020), GPT 4 (Achiam et al. 2023), and Llama 3 (Meta 2024) with a reject option1 over popular SPIDER (Yu et al. 2018) and EHRSQL (Lee et al. 2022), covering general-domain and clinical domains. Our findings indicate that in distribution shift settings (cross-database or compositional), the selective classifier achieves high recall but low precision in error detection, leading to loss of total generated queries and deterioration in overall Text-toSQL system quality. Our analysis also revealed that unanswerable queries are more likely to be detected using our confidence estimates than incorrect queries (Sec. 4). Conversely, in a benchmark with unanswerable queries without a significant compositional or domain shift, the Text-toSQL system with a selective classifier performs better overall in error detection with a lower rejection loss of correct queries. Furthermore, we examined the calibration characteristics of logit-based confidence estimates (Sec. 5). Under distribution shift, all fine-tuned models lacked proper calibration. Post-hoc calibration methods such as Platt Calibration and Isotonic Regression improved the initial models’ calibration, underscoring the importance of calibration techniques in enhancing the reliability of model predictions. As a result, experiments demonstrate that while decoder-only models perform better on certain datasets (compositional or cross-database), encoder-decoder methods exhibit superior calibration for Text-to-SQL after post-hoc calibration. In Sec. 6, we did an in-depth analysis of the relation of selective classifier confidence and generated query complexity.

# 2 Related Work

Uncertainty estimation and error detection The reliability of Text-to-SQL models or question answering systems, in general, is closely tied to their calibration ability for error detection and result interpretation. Selective prediction, where a model can choose to predict or abstain, has been a longstanding topic in machine learning (Chow 1957; El-Yaniv and Wiener 2010; Dong, Quirk, and Lapata 2018).

The rise of LLMs has highlighted the issue of hallucinations in NLP. Uncertainty estimation is a key research area for developing calibrated Text-to-SQL systems and reliable selective prediction algorithms. Several recent works (Malinin and Gales 2021; van der Poel, Cotterell, and Meister 2022; Ren et al. 2023; Vazhentsev et al. 2023; Fadeeva et al. 2023) have developed methods to estimate uncertainty in language models, aiming to provide better-calibrated uncertainty estimates or to perform error and out-of-domain (OOD) detection. A relevant approach is utilized in (Kadavath et al. 2022), where the authors created a prompt asking the model if the generated prompt is correct. The model’s calibration was then measured by the probability of predicting the correct answer when it was correct across the validation set.

One of the most popular directions includes methods that deal with $p ( y | x )$ only. These include softmax maximum probability (Hendrycks and Gimpel 2017), temperature scaling (Guo et al. 2017), and ensembles of deep neural networks for uncertainty estimate (Lakshminarayanan, Pritzel, and Blundell 2017) methods. For auto-regressive models, there are several probabilistic approaches, which utilize the softmax distribution, such as normalized sequence probability (Ueffing and Ney 2007) and average token-wise entropy (Malinin and Gales 2021). In our work, we follow the recent approach presented in (Yang et al. 2024), which introduces a maximum entropy estimate for predicting sequence uncertainty. This approach is a better fit for semantic domains where false positive generations must be avoided at all costs. Here, the model’s confidence in sequence prediction is determined by its weakest token prediction. These methods aim to provide a calibrated estimate that can be utilized later with a threshold. In contrast to uncertainty estimates and subsequent threshold selection, there are methods that incorporate an OOD-detection component in addition to $p ( y | x )$ . Chen et al. (2023) developed an additional model to the Text-to-SQL component – a binary classification model with input of question and generated SQL.

Text-to-SQL Text-to-SQL, as a sub-domain of semantic parsing, is deeply influenced by distribution shifts (Suhr et al. 2020; Finegan-Dollak et al. 2018). On one side, there is domain shift, where a model trained on one set of databases must perform well on another set. Multiple datasets like SPIDER (Yu et al. 2018) and BIRD (Li et al. 2024) are designed to evaluate this aspect. On the other side, there is compositional shift, involving novel SQL query structures in the test dataset. (Shaw et al. 2021; Finegan-Dollak et al. 2018) explored the models’ ability to generalize to novel SQL templates and noted a significant accuracy drop in results. However, these datasets and splits include only answerable questions for the underlying databases. Recently, a ERHSQL benchmark (Lee et al. 2024b) was presented with a covariate shift, featuring unanswerable queries in the test set or those needing external knowledge.

To sum up, while there has been considerable research on uncertainty estimation, such as calibration in semantic parsing (Stengel-Eskin and Van Durme 2023) and uncertainty constraints (Qin et al. 2022) for better calibration, to our knowledge, there is no evident research on selective prediction for probabilistic uncertainty estimation in Text-to-SQL under distribution shifts. In our work, we explore the calibration characteristics of sequence-to-sequence models under different various distribution shift settings (cross-database shift, compositional shift, and covariate shift). Our goal is to detect incorrect generations or generations involving OOD examples, as seen in ERHSQL.

# 3 Problem Setup

We study selective prediction in Text-to-SQL systems under distribution shift settings. Specifically, we examine a Textto-SQL system consisting of two components: the Text-toSQL model $y$ , which takes the natural language utterance $x$ and generates an SQL query $\hat { y }$ , and a selective classifier $\mathcal { C }$ , which decides whether to output the generated query $\hat { y }$ or abstain based on the uncertainty estimate score $u$ . In this section, we formally outline our method for calculating $u$ for generated SQL queries, the selective prediction setup, and the data we evaluated. In three consecutive studies, we investigate the balance between coverage and accuracy of Text-to-SQL models with a reject option (Sec. 4), model calibration, and the relationship between the confidence of the selective classifier and query characteristics (Sec. 5 and 6).

# Text-to-SQL Models

In our Text-to-SQL system with reject option, we utilize four models known for their descent ability toward SQL generation. We employ T5-large and T5-3B models from the encoder-decoder family, an in-context-learning GPT-based DAIL-SQL (Gao et al. 2024) and a decoder model Llama 3, which is fine-tuned using both supervised fine-tuning (SFT) and parameter-efficient fine-tuning (PEFT) techniques with LoRa (Hu et al. 2022). We fine-tune both the T5 and Llama models. To form an input $x$ , fine-tuned model receives question $q$ along with database schema $S$ . In in-contextlearning with DAIL-SQL, we additionally incorporate relevant question-query pairs as examples for ChatGPT prompt. The model is expected to generate a query $\hat { y }$ . The hyperparameters of the fine-tuning are specified in Appendix A.

# Uncertainty Estimate

Given the input sequence $x$ and output sequence $y$ the standard auto-regressive model parameterized by $\theta$ is given by:

$$
P ( y | x , \theta ) = \prod _ { l = 0 } ^ { L } P ( y _ { l } | y _ { < l } , x , \theta )
$$

Where the distribution of each $y _ { l }$ is conditioned on all previous tokens in a sequence y<l = y0, ..., yl 1.

For fine-tuned models, we base our heuristic based on intuition a sequence is only as good as its weakest token prediction $P ( y _ { l } | y _ { < l } , x , \theta )$ to get the uncertainty estimate $u$ of the whole sequence $y$ . If the model soft-max probabilities $p _ { l }$ are close to uniform, the token prediction is less likely to be correct, in contrast to a peak distribution, where the model is certain about token prediction.

$$
\begin{array} { r } { p _ { l } = P ( y _ { l } | y _ { < l } , x , \theta ) } \\ { H ( p _ { l } ) = \displaystyle \sum _ { v = 0 } ^ { | V | } p _ { v } l o g ( p _ { v } ) } \\ { u = m a x ( H _ { 0 } , . . . , H _ { L } ) } \end{array}
$$

For ChatGPT-based DAIL-SQL, we do not have access to the full vocabulary distribution. Therefore, we utilize the Normalized Sequence Probability modification (Ueffing and Ney 2005), which was recently featured in one of the EHRSQL shared task solutions (Kim, Han, and $\mathrm { K i m } 2 0 2 4 )$ :

$$
u = \frac { 1 } { | L | } \sum _ { l = 0 } ^ { | L | } l o g ( p _ { l } )
$$

# Selective Prediction Setting

In the selective prediction task, given a natural language input $x$ , the system outputs $( \hat { y } , u )$ where $\hat { y } \in \mathcal { V } ( x )$ is the SQL query generated by the Text-to-SQL model $y$ , and $u \in \mathcal R$ is the uncertainty estimate. Given a threshold $\gamma \in \mathcal { R }$ , the overall Text-to-SQL system predicts the query $\hat { y }$ if $u \geq \gamma$ ; otherwise, it abstains. The rejection ability is provided by selective classifier $\mathcal { C }$ .

Following the experimental setup of El-Yaniv and Wiener (2010), we utilize a testing dataset $D _ { t s t }$ , considered out-ofdistribution (OOD) relative to the training dataset $D _ { t r }$ . We split $D _ { t s t }$ independently and identically into two data samples: a known OOD sample $D _ { k n o w n }$ and an unknown OOD sample $D _ { u n k }$ . We use $D _ { k n o w n }$ to fit our selective classifier or calibrator, and $D _ { u n k }$ for evaluation.

The main characteristics of the selective classifier $\mathcal { C }$ are its coverage and risk. Coverage is the fraction of $D _ { u n k }$ on which the model makes correct predictions, and risk is the error fraction of $D _ { u n k }$ . As the threshold $\gamma$ decreases, both risk and coverage increase. We evaluate our experiments in terms of the risk vs. coverage paradigm.

To define the target $\hat { y }$ for selective classifiers in Text-toSQL, we use the inverted execution match metric (EX) for the gold query $g _ { i }$ and predicted query $p _ { i }$ as defined in Equation 3. This means that we set the positive class as the presence of an error.

$$
{ \hat { y } } _ { i } = { \left\{ \begin{array} { l l } { 0 } & { { \mathrm { i f ~ } } \mathrm { E X } ( g _ { i } ) = = \mathrm { E X } ( p _ { i } ) } \\ { 1 } & { { \mathrm { i f ~ } } \mathrm { E X } ( g _ { i } ) \neq \mathrm { E X } ( p _ { i } ) } \end{array} \right. }
$$

To evaluate results for a particular choice of $\gamma$ , we utilize recall and precision metrics. Coverage refers to our ability to identify and abstain from wrong SQL query generations, while risk corresponds to the proportion of false positive predictions (incorrect queries deemed correct). Recall measures how effectively the system detects errors, and False Discovery Rate (FDR) (1 precision) indicates the extent to which we abstain from returning correct SQL queries to the user. For a comprehensive assessment of the selective performance across different threshold values $\gamma$ , we employ the Area Under the Curve (AUC) metric.

# Distribution Shift in Text-to-SQL

We evaluate the uncertainty estimates of Text-to-SQL models in distribution shift settings, mimicking various types of shifts: domain shift, compositional shift, and covariate shift. Domain and compositional shifts are full shift examples where $p ( x _ { t s t } ) \neq p \bar { ( } x _ { t r } )$ and $p ( y _ { t s t } | x _ { t s t } ) \neq p ( y _ { t r } | x _ { t r } )$ , while covariate shift involves only a change in $p ( x _ { t s t } ) \neq$ $p ( x _ { t r } )$ . Our Text-to-SQL pairs $D$ follow $p ( x )$ and $p ( y | x )$ for training and testing.

To evaluate such distribution shifts, we leverage two Text-to-SQL datasets: SPIDER-based PAUQ (Bakshandaeva et al. 2022) and EHRSQL (Lee et al. 2022). The PAUQ dataset is a refinement of the widely recognized nonsynthetic benchmark SPIDER (Yu et al. 2018) for the Textto-SQL task. We prefer English version of PAUQ over SPIDER because it has 8 times fewer empty outputs $\mathrm { 1 6 6 5 ~  }$ 231) and 4 times fewer zero-return queries with aggregations (e.g., maximal, minimal) $( 3 7 9  8 5 \$ ). This improvement is crucial, as zero or empty returns can be considered correct when a model generates an executable yet incorrect SQL query, which is undesirable for our study’s focus on execution match in the selective classifier. EHRSQL is a clinical Text-to-SQL dataset that includes pairs of input utterances and expected SQL queries. It covers scenarios where generating an SQL query is not possible for a given question.

We utulize the following splits to represent different aspects of compositionality (Hupkes et al. 2020, 2023):

• PAUQ in cross-database setting - This setting uses the original SPIDER dataset split, where the data is divided between training and testing sets with no overlap in database structures. During training on $D _ { t r }$ , the model must learn to generalize to novel database structures found in $D _ { t s t }$ . We refer to this split as PAUQ XSP. • PAUQ with template shift in single database setting - This is the compositional PAUQ split based on templates in a single database setting (same set of databases across $D _ { t r }$ and $D _ { t s t }$ ), inspired by (Finegan-Dollak et al. 2018). This split forces the model to demonstrate its systematicity ability—the ability to recombine known SQL syntax elements from $D _ { t r }$ to form novel SQL structures in $D _ { t s t }$ . We refer to this split as Template SSP. • PAUQ with target length shift in single database setting - This is another compositional split of the PAUQ dataset, based on the length of SQL queries, in a single database setting (Somov and Tutubalina 2023). Shorter samples are placed in $D _ { t r }$ , and longer samples in $D _ { t s t }$ , ensuring that all test tokens appear at least once in $D _ { t r }$ . We refer to this as TSL SSP. It tests the model’s productivity—its ability to generate SQL queries that are longer than those it was trained on. • EHRSQL with unanswerable questions - This setting uses the original EHRSQL split. Its distinctive feature is the presence of unanswerable questions in $D _ { t s t }$ . These questions cannot be answered using the underlying database content or require external knowledge, making it impossible to generate a correct SQL query. We refer to this split as EHRSQL.

For a comparison of our Template SSP and TSL SSP splits with related work, please see Appendix B.

# 4 Case Study #1: Selective Text-to-SQL

In this case study, we aim to address the following research questions: RQ1: Among selective classifiers, which classifier offers the best trade-off between coverage and risk? RQ2: What is the impact on the performance of a Text-toSQL system when the system is expanded with a selective classifier? RQ3: What distribution shifts present the most significant challenge for Text-to-SQL with a reject option? RQ4: Given the existence of unanswerable questions in the test set, what types of errors are we more likely to find with a selective classifier?

In our selective classifier methods, we utilize approaches outlined in (El-Yaniv and Wiener 2010; Lee et al. 2022), including the threshold-based approach (Lee et al. 2024a), Logistic Regression, and Gaussian Mixture clustering.

Logistic regression We determine parameters $\theta$ using the sigmoid function. During inference, we predict the probability of $u _ { i }$ corresponding to the error prediction based on the probability score of the sigmoid function with fitted parameters.

$$
\begin{array} { r } { p ( y _ { i } | u _ { i } , \theta ) = \displaystyle \frac { 1 } { 1 + e ^ { - ( \theta _ { 0 } + \theta _ { 1 } u _ { i } ) } } } \\ { \hat { y } _ { i } = [ p ( y _ { i } | u _ { i } , \theta ) > 0 . 5 ] } \end{array}
$$

Gaussian mixture clustering We consider our uncertainty estimates as a combination of two normal distributions, denoted as $\mathcal { N } _ { z }$ . The first distribution, $z _ { 0 }$ , is associated with the uncertainty scores $u _ { i }$ of correct generations, while the second distribution, $z _ { 1 }$ , is linked to error generations. We use the expectation-maximization algorithm (EM) to determine the parameters $\mu _ { z } , \sigma _ { z }$ , and the mixture weight $\pi _ { z }$ for each distribution $z$ . During inference, we predict the most likely distribution for a given uncertainty estimate $u _ { i }$ using:

$$
\hat { y } _ { i } = \arg \operatorname* { m a x } _ { z } \left( \pi _ { z } \mathcal { N } ( u _ { i } \mid \mu _ { k } , \sigma _ { k } ) \right)
$$

# Results

We evaluated our five models on four distinct datasets, using the $F _ { \beta }$ score to compare methods: $F _ { \beta } = ( ( 1 + \beta ^ { 2 } ) \mathrm { t p } ) / ( ( 1 \bar { + }$ $\beta ^ { 2 } ) \mathrm { t p } { + } \mathrm { f p } { + } \beta ^ { 2 } \mathrm { f n } )$ , tp and fp stand for false and true positives, respectively, fn for false negatives.

To address RQ1, we created a heatmap of $F _ { \beta = 1 }$ scores across all splits and models in Fig. 2. Gaussian Mixture Model demonstrates the best trade-off between precision and recall in a task of error detection. For an in-depth analysis in Appendix C we plotted the $F _ { \beta }$ scores for other $\beta$ favoring precision or recall.

Based on selective classification Text-to-SQL tables in Appendix D we built the risk vs coverage comparison in Figure 3 for Gaussian Mixture to answer RQ2 and RQ3. As shown in Figure 3 (left), Gaussian Mixture effectively

Logistic Regression $\mathsf { F } _ { \mathsf { \beta } = 1 } = 1 = 0 . 5 7$ Gaussian Mixture $\mathsf { F } _ { \beta = 1 } = 1 \mathop { = } 0 . 5 8$ Threshold Fβ=1=0.56 0.9 T5-large-0.630.370.860.77 T5-large-0.66 0.430.78 0.78 T5-large-0.7 0.47 0.86 0.58 0.8 T5-3B-0.48 0.380.84 0.81 T5-3B-0.5 0.440.82 0.83 T5-3B-0.59 0.410.84 0.34 0.7 0.6 DIAL-SQL-0.370.230.47 0.7 DIAL-SQL-0.44 0.22 0.460.78 DIAL-SQL 0.460.320.480.78 0.5 Llama3-8B LoRA-0.44 0.48 80.83 0.53 Llama3-8B LoRA-0.44 0.480.81 0.43 Llama3-8B LoRA-0.45 0.48 0.830.42 0.4 Llama3-8B SFT-0.42 0.45 0.78 0.53 Llama3-8B SFT 0.44 0.5 0.730.67 Llama3-8B SFT- 0.44 0.480.74 0.49 0.3 0.2 Q

![](images/3a76243b4109ae920e7dd7dbaccfaa833d2265e97996749f2eaeefa7dcc6bfe7.jpg)  
Figure 2: Heatmaps of $F _ { \beta = 1 }$ per split and model for every selective classifier (Logistic Regression, Gaussian Mixture, and Threshold).   
Figure 3: Left: The system risk decrease with a Gaussian Mixture for every split averaged between all SQL generation models. Right: The system coverage decrease with the presence of an Gaussian Mixture external classifier for every split averaged between all SQL generation models.

identifies error generations, reducing risk by an average of $80 \%$ across all splits and models, even under distribution shift. However, this comes at the cost of a high false discovery rate (FDR), as also seen in Figure 3 (right). Under these conditions, system coverage remains low, yielding only 1-2 correct generations per 5 requests. However in settings with minimal full shift, such as EHRSQL, selective Text-to-SQL performs well, achieving an FDR as low as $10 \%$ with some models.

Confirming the results of Figure 2 in Table 1 we average the scores of Recall, FDR, and Result EX from Appendix D tables, with Gaussian Mixture having the lowest FDR hence does not worsen the Result EX as other two methods.

For the further analysis of RQ2 and RQ3, we adopt AUC for the selective performance across various threshold values in Appendix E using the probability scores from the Gaussian Mixture classifier. T5-large and T5-3B consistently show superior performance in comparison to other models, especially in the fourth split where they achieve the highest AUC scores (0.93). This suggests that T5-large and

T5-3B models are more reliable in terms of Text-to-SQL with a reject option.

To address RQ4, we delved into the types of errors most commonly encountered with the Gaussian Mixture selective classifier in the ERHSQL dataset, as shown in Appendix F. Overall, there is a higher chance of encountering a generation of irrelevant questions as opposed to encountering an incorrect generation. This indicates that all models are fairly confident in generating an incorrect query to a relevant question as opposed to generating a query to an irrelevant one. Furthermore, T5-3B, being the most calibrated model as indicated in Table 2, is capable of accurately detecting even incorrect generations.

Takeaway 1 (RQ1, RQ2) The Gaussian Mixture Model demonstrated the best trade-off between coverage and risk. The addition of a selective classifier, particularly the Gaussian Mixture Model, enhances the performance of the Textto-SQL system by maintaining low False Discovery Rate (FDR) values, especially on the EHRSQL dataset. This indicates strong error detection capabilities with minimal negative impact on SQL traffic.

Table 1: Overall methods comparison averaged across all splits and models.   

<html><body><table><tr><td></td><td>Recall</td><td>FDR</td><td>Result EX</td></tr><tr><td>Gaussian Mixture</td><td>0.798</td><td>0.364</td><td>0.251</td></tr><tr><td>Logistic Regression</td><td>0.873</td><td>0.469</td><td>0.145</td></tr><tr><td>Threshold</td><td>0.872</td><td>0.471</td><td>0.143</td></tr></table></body></html>

Table 2: Calibration methods comparison of Brier scores averaged across all splits for each model.   

<html><body><table><tr><td></td><td>MinMax</td><td>Platt</td><td>Isotonic</td></tr><tr><td>T5-large</td><td>0.2</td><td>0.121</td><td>0.117</td></tr><tr><td>T5-3B</td><td>0.17</td><td>0.108</td><td>0.106</td></tr><tr><td>Llama3-8BLoRA</td><td>0.22</td><td>0.216</td><td>0.199</td></tr><tr><td>Llama3-8B SFT</td><td>0.21</td><td>0.19</td><td>0.175</td></tr><tr><td>DIAL-SQL</td><td>0.239</td><td>0.16</td><td>0.152</td></tr></table></body></html>

Takeaway 2 (RQ3) The Template SSP split and TSL SSP split were identified as presenting significant challenges for all models. At the same time, models trained on the EHRSQL dataset under less domain and compositional shifts, show that the error detection method operate much more effectively.

Takeaway 3 (RQ4) Selective classifier has a higher likelihood of spotting generations to irrelevant questions compared to incorrect generations. This suggests that the models are generally more confident in generating incorrect queries for relevant questions than in generating queries for irrelevant ones. T5-3B, being the most calibrated, effectively detects incorrect generations.

![](images/e786be99bcd249683f5eca68a64d525bd48e479a6d91f3b047b5f06366d8ceb0.jpg)  
5 Case Study #2: Calibration Characteristics   
Figure 4: The calibration effect on T5-3B on PAUQ XSP (cross-database setting) and EHRSQL (single clinical database) compared across MinMax, Platts, and Isotonic calibration (BS stands for Brier score).

In this section, we will investigate the following research question (RQ5): How do different calibration methods and training datasets influence the calibration of model uncertainty scores, and what trade-offs exist between calibration measures and model execution accuracy? Specifically, the model calibration addresses the question: out of all the instances where we predicted an $80 \%$ chance of a query being correct, how often was the query actually correct? A wellcalibrated model would have this proportion close to $80 \%$ . In this case study we want to measure the calibration of the uncertainty estimates.

In contrast to (Stengel-Eskin and Van Durme 2023) on calibration in semantic parsing, we measure uncertainty estimates at the sequence level, as this is most relevant for system safety. We define the positive class as an execution match result if $E X ( g _ { i } ) = = \bar { E } X ( p _ { i } )$ . For calibration of our score $u$ , two calibration methods - Platt calibration and Isotonic calibration - and a naive normalization method (MinMax scaling) were used.

MinMax normalization can be applied here because the maximum entropy estimate is a monotonic function. This allows us to transform the value range from $[ 0 ; + \infty ]$ to $[ 0 ; 1 ]$ . We refer to the calibrated score as $u ^ { c }$ : $u _ { i } ^ { c } ~ =$ $\frac { ( \bar { u } _ { i } - \bar { m } i n ( u _ { i } ) ) } { m a x ( u _ { i } ) - m i n ( u _ { i } ) }$

Platt calibration (Platt et al. 1999) is represented by a logistic regression function from Eq. 4. The parameters $\theta _ { 0 }$ and $\theta _ { 1 }$ are selected on a $D _ { k n o w n }$ using the maximum likelihood method.

Isotonic regression (Zadrozny and Elkan 2002) involves constructing a piece-wise constant function of the form $g _ { m } ( u _ { i } ) = \theta _ { m }$ to transform uncertainty estimates by minimizing the quadratic difference. As $g _ { m }$ is a piece-wise constant function, the training of this calibration method involves solving the following optimization problem:

$$
\begin{array} { r l } { \displaystyle \underset { M , \theta , a } { \operatorname* { m i n } } } & { \displaystyle \sum _ { m = 1 } ^ { M } \sum _ { i = 1 } ^ { N } \mathbb { 1 } ( a _ { m } \leq u _ { i } < a _ { m + 1 } ) ( y _ { i } - \theta _ { m } ) ^ { 2 } } \\ { \mathrm { s . t . } } & { 0 \leq a _ { 1 } \leq a _ { 2 } \leq . . . \leq a _ { M + 1 } = 1 , } \\ & { \theta _ { 1 } \leq \theta _ { 2 } \leq . . . \leq \theta _ { M } } \end{array}
$$

where $M$ is the number of function intervals; $a _ { 1 } , \dots , a _ { M + 1 }$ are intervals’ boundaries; $\theta _ { 0 } , \dots , \theta _ { M }$ are the values of the function $g _ { m }$ . During fitting on $D _ { k n o w n }$ , Isotonic regression optimizes the heights of the histogram columns for function calibration.

# Experimental Results

Evaluation metric While other calibration comparison methods, such as expected calibration error (ECE), exist, the Brier score (Ashukha et al. 2020) offers a more interpretable means of comparing models. If the model is confident in the positive class (indicated by a high estimate of $u _ { i } ^ { c ^ { \prime } }$ ), then the difference will be minimal. The Brier scoring method estimate shows the squared difference between the target variable $y _ { i }$ (which can be either 1 or 0) and the predicted probability: Brier Score $\begin{array} { r } { = \sum _ { i = 1 } ^ { N } ( y _ { i } - u _ { i } ^ { c } ) ^ { 2 } } \end{array}$

![](images/5981105d1a23a1a6e4d42084da566a0fb059d01118a62ff6d11476f9066b756f.jpg)  
Figure 5: Trade-off plots between execution match and calibration for selected Text-to-SQL models (T5-large, T5-3B, Llama 3 in SFT and LoRa setting).

Results Table 2 presents a comparison of calibration methods using the Brier score, averaged across all data splits. Isotonic regression consistently outperforms both MinMax normalization and Platt calibration across all models, demonstrating its effectiveness in enhancing calibration quality. Notably, as the size of the T5 models increases, the performance of isotonic regression shows improvement.

Fig. 4 illustrates the calibration curves for Platt, Isotonic, and MinMax for T5 across two datasets. As shown in the figure, the original normalized uncertainty estimate is not calibrated, whereas the calibration methods provide significant improvements. In Appendix G we present the isotonic calibrations across all models for TSL SSP and EHRSQL splits, averaged over multiple seeds. It is evident that the shifted datasets (PAUQ XSP, Template SSP, and TSL SSP) do not lead to calibrated models. However, in the EHRSQL split with no complete shift, the models demonstrate effective calibration.

Overall in Figure in 5, we see that our models exist on a trade-off of calibration and generation quality, with some models being of lower generalization quality but a better calibration.

Takeaway 4 (RQ5) The results indicate that the original entropy estimate of the models’ uncertainty is not calibrated. Isotonic regression consistently outperforms other calibration methods like MinMax normalization and Platt calibration across various models. Additionally, encoder-decoder architecture models are found to be better calibrated compared to decoder-only models.

# 6 Case Study #3: Query Complexity Analysis

In this case study, we investigate the relationship between model confidence in a selective classifier and query complexity, specifically focusing on query length and the number of schema elements in the generated query. Our research question (RQ6): Is the probability of rejection by the selective classifier related to query complexity characteristic?

To address this question, we utilized the Gaussian Mixture probabilities (Sec. 4) of the incorrectly generated examples by the T5 model, as T5 models demonstrated the best selective performance across various thresholds (Appendix E). We assessed query complexity using two key indicators: the length of the generated query(in SQL tokens) and the number of unique schema elements in the generated query. Scatter plots in Appendix H were constructed for each data split to analyze the relationship between these query characteristics and the confidence of the selective classifier. Our initial hypothesis was that more complex incorrect queries, characterized by greater length or more schema elements, would correspond to lower probability scores, leading the selective classifier to correctly identify these as incorrect generations. However as the plots show, selective classifier probability does not hold any seeming relation to query complexity.

Takeaway 5 (RQ6) Contrary to our hypothesis, we did not observe a proportional decline in model confidence for incorrect queries as query complexity increased. Across all splits, even for the most calibrated models, there was no clear relationship between selective classifier confidence and query complexity.

# 7 Conclusion

In this paper, we investigated error detection and calibration, utilizing Text-to-SQL LMs with a reject option across general-domain and clinical datasets. We believe our findings could enhance the development of more trustworthy Text-to-SQL models in the future. Future research might concentrate on evaluation across a wider range of datasets and different aspects of compositionality.

Limitations Given that our analysis focused on the SPIDER and EHRSQL datasets, the generalizability of our findings may be limited. We concentrated solely on these domains to validate our results, which may not fully capture the variability of noise distributions across different datasets. However, we consider this a minor limitation, as our goal was to observe the models’ behavior under distribution shifts rather than to propose and validate a new model with a reject option.

Ethics Statement The models and datasets used in this work are publicly available for research purposes. All experiments were conducted on four A100 80GB GPUs. Our PyTorch/Hugging Face code will be released with the paper, and we do not anticipate any direct social consequences or ethical issues.