# A Compact Model for Mathematics Problem Representations Distilled from BERT

Hao Ming1, Xinguo $\mathbf { Y } \mathbf { u } ^ { 1 , 2 * }$ , Xiaotian Cheng1,2, Zhenquan Shen1, Xiaopan Lyu1

1Faculty of Artificial Intelligence in Education, Central China Normal University, Wuhan, China 2Central China Normal University Wollongong Joint Institute, Central China Normal University, Wuhan, China {hming, xgyu}@ccnu.edu.cn, xiaotiancheng $@$ mails.ccnu.edu.cn, {shenzhenquan, xiaopanlv}@ccnu.edu.cn

# Abstract

Large language models (LLMs) have made significant advancements in math problem solving, but their large size and high latency render them impractical for real-world applications in intelligent mathematics solvers. Recently, taskagnostic compact models have been developed to replace LLMs in general natural language processing tasks. However, these models often struggle to acquire sufficient mathrelated knowledge from LLMs, leading to unsatisfactory performance in solving math word problems (MWPs). To develop a specialized compact model for representing MWPs, we develop the knowledge distillation (KD) technique to extract mathematical semantics knowledge from the large pretrained model BERT. Effective knowledge types and distillation strategies are explored through extensive experiments. Our KD algorithm employs multi-knowledge distillation to extract fundamental knowledge from hidden states in the middle to lower layers, while also incorporating knowledge of mathematical relations and symbol constraints from higherlayer outputs and math decoder outputs, by leveraging bottleneck networks. Pre-training tasks on MWP datasets, such as masked language modeling and part-of-speech tagging, are also utilized to enhance the generalization of the compact model for MWP understanding. Additionally, a simple parameter mixing strategy is employed to prevent catastrophic forgetting of acquired knowledge. Our findings indicate that our approach can reduce the size of a BERT model by $10 \%$ while retaining approximately $9 5 \%$ of its performance on MWP datasets, outperforming the mainstream BERT-based task-agnostic compact models. The efficacy of each component has been validated through ablation studies.

# Introduction

Solving Math Word Problems (MWPs) automatically has the potential to significantly advance both the development and application of intelligent math education. Large pretrained language models (LLMs) (Peters et al. 2018; Devlin et al. 2019) have achieved remarkable success in many natural language processing tasks, benefiting from extensive parameters and training on large-scale corpora. Recently, LLMs have continuously improved their mathematical capabilities, and many of them have been successfully employed to develop high-quality MWP solvers (Qin et al. 2021), owing to their effective semantic representation and massive knowledge. However, due to the large parameter sizes and high latency, LLMs are often unsuitable for resourceconstrained environments, such as the Internet of Things (IoTs) and small intelligent devices, which hinders their practical applications (Kok, Demirci, and Ozdemir 2024). Low-resource areas, especially in the education underdeveloped countries, have a greater need for lightweight models.

Knowledge distillation (Hinton, Vinyals, and Dean 2015) is an effective approach to model compression and has developed rapidly due to its efficacy in practical applications. Knowledge distillation involves transferring knowledge from a larger model (teacher) to a smaller one (student). Recently, task-agnostic knowledge distillation (Wang et al. 2020; Sun et al. 2020a) has been developed to construct compact models that can replace LLMs in general natural language processing tasks. These models are trained without task-specific data and do not require fine-tuning of the teacher model. However, student models often fail to learn sufficient knowledge from LLMs, and their performance on specific downstream tasks remains unsatisfactory even after fine-tuning. This issue may also arise in MWP solving, as the experimental results in this paper further confirm this inference. Successful attempts have been made to develop compact models in the fields of biomedicine and clinical applications (Rohanian et al. 2023). However, a compact model for MWP representations has not been reported yet, and no dedicated distillation strategy has been considered.

MWPs have unique symbols and linguistic representations of mathematics relationships, therefore, it is crucial to conduct task-specific distillation focused on mathematics representations and to design specialized knowledge types and distillation strategies. Within the studies of MWP solvers, it has been found that the part of speech (POS) and their combinations might encode underlying mathematical relations (Yu et al. 2023). Consequently, POS knowledge could be a fundamental basis for understanding MWPs. Inspired by this method, we propose a distillation approach that leverages POS pretraining and thoroughly explores the knowledge types and distillation strategies specifically for MWP representations. In this work, a compact model dedicated to MWP representation is proposed. The main contributions of this paper are summarized as follows:

• We are the first to focus specifically on creating a compact model for MWP representations through developing distillation technology. • We investigate the essential knowledge types required for understanding MWPs and extract them from BERT by leveraging the bottleneck networks. • An effective distillation strategy is designed to extract mathematics knowledge by integrating tailored pretraining and task-specific distillation techniques. • On average, our compact model retains about $9 5 \%$ of BERT’s performance on typical MWP datasets, while the parameters are only $10 \%$ of the original model.

# Related Works

# Math Word Problem Solving

Math word problem (MWP) solving involves deriving mathematics expressions and numerical solutions from a given problem text. Early research includes rule-based methods (Fletcher 1985; Bakman 2007), statistical machine learning methods (Kushman et al. 2014; Hosseini et al. 2014), and semantic parsing methods (Shi et al. 2015). Wang et al. (Wang, Liu, and Shi 2017) conducted a milestone study, introducing the Deep Neural Solver (DNS), the first neural solver for MWPs using a Seq2Seq structure. Following Wang’s pioneering work DNS, deep learning-based methods have become mainstream in this research community due to their significant improvements in MWPs, including high accuracy and the elimination of handcrafted features. Deep learningbased MWP solvers typically utilize the Encoder-Decoder paradigm, where the encoder captures the semantics and relationships of the given problem text, and the decoder translates the encoder’s outputs into mathematics equations with numbers and operators, ultimately deriving final answers.

Math expressions have natural hierarchical structures, so tree-structured solvers have often been designed to solve MWPs in recent studies. GTS (Xie and Sun 2019) is a goaldriven tree-structured neural model that generates an expression tree using a goal-driven mechanism; its decoder is widely used in MWP solvers. HMS (Lin et al. 2021) is a hierarchical math solver inspired by human reading habits. Xiong et al. (Xiong et al. 2022) proposed a variational information bottleneck to extract knowledge from the expression syntax tree. In these works, tree-structured decoders are employed to generate mathematics equations. The encoder’s outputs directly influence the solver’s performance, thus, LLM-based encoders can significantly improve the accuracy of MWP solving. For instance, MWP-BERT (Liang et al. 2022), DeductiveMWP (Jie, Li, and Lu 2022), and LogicSolver (Yang et al. 2022) use the BERT family (Devlin et al. 2019; Liu et al. 2019) as their encoder, while Gen&Rank (Shen et al. 2021) use BART. Additionally, MathBERT (Peng et al. 2021), and MWP-BERT, have been developed to enhance the capability of MWP solvers. However, these models overlook the high cost and latency in actual application environments, caused by the large size of LLMs. Math-solving engines are often deployed in lightweight and portable devices, where low resource usage and real-time response are essential in these applications.

# Knowledge Distillation

Hinton first proposed the concept of knowledge distillation, which transfers knowledge from a large model (teacher) to a small model (student) using the loss of the soft target distributions. The goal of knowledge distillation is to minimize the differences between the teacher model and the student model:

$$
\mathcal { L } _ { K D } = \mathcal { L } ( f ^ { T } ( x ) , f ^ { S } ( x ) )
$$

where $f ^ { T } ( \cdot )$ and $f ^ { S } ( \cdot )$ represent the features of teacher and student respectively, $\mathcal { L }$ is the loss function. Due to its great success in practical applications, knowledge distillation has become an effective approach for model compression. Task-specific distillation techniques typically fine-tune the teacher model on specific downstream tasks first and then have the student model emulate the teacher’s behaviors, thereby achieving knowledge transfer from the teacher.

Knowledge types and distillation schemes are crucial to the effectiveness of distillation. Commonly used knowledge types can be divided into three categories (Gou et al. 2021). Response-based knowledge refers to the output response of the teacher’s last layer, which is typically logits, also known as soft labels. It employs a method similar to label smoothing and has been widely used due to its simplicity and alignment with supervised learning. However, this knowledge type ignores the intermediate-level supervision provided by the teacher model. Feature-based knowledge primarily refers to the features of intermediate layers, namely feature maps. Feature maps contain more implicit knowledge and can help reduce the performance gap between the teacher and student. The selection of effective hint layers to guide the student model requires further exploration. Relationbased knowledge concerns the relationships between different layers or training data, such as the flow of solution process (FSP) matrix, multi-head graph (MHG), instance relations, and mutual information flow. Designing effective relation features also remains an open question. Distillation strategies (Park et al. 2024) include mutual distillation, adversarial distillation, muti-teacher distillation, data-free distillation, and self-distillation, etc. Various approaches have been proposed to transfer more effective knowledge and reduce the knowledge gap. For the distillation of LLMs, DistilBERT (Sanh et al. 2019), TinyBERT (Jiao et al. 2019), and MobileBERT (Sun et al. 2020b) are typical distilled BERT-base models and they are all evaluated on the General Language Understanding Evaluation (GLUE) benchmark. These approaches utilize a Transformer-like encoder to diminish the model gap with BERT, but differ in layer number and hidden size. Since distillation is conducted on general corpora, these models underperform on MWPs.

# Dimension Reduction for MWP Representations

# Reduced Vector Representation

As mentioned above, we know that BERT produces redundant vector representations when used for MWP solving, so we first need to determine how many dimensions of vector representation are sufficient for this task. Based on the solver architecture using BERT and GTS decoder, a linear network Linear $( m , n )$ without activation is connected between them, as shown in Figure 1.

![](images/d83af0f898bd08d3519cad7fca177943aa82add5c3690e148cd48167f3fb379f.jpg)  
Figure 1: Vectors reduced from BERT using a linear layer are used for MWP solving.

Table 1: The performance of reduced vector representation from BERT in solving MWPs.   

<html><body><table><tr><td>Dimension</td><td>Acc.onMath23k</td><td>Acc.onApe-clean</td></tr><tr><td>768</td><td>84.07%</td><td>84.33%</td></tr><tr><td>312</td><td>83.97%</td><td>84.12%</td></tr><tr><td>256</td><td>83.47%</td><td>83.62%</td></tr><tr><td>128</td><td>82.97%</td><td>80.23%</td></tr></table></body></html>

The hidden size of BERT is 768, so we set $m = 7 6 8$ and vary $n$ from 768 to 128 to observe changes in solving accuracy. The results in Table 1 show that approximately 300 hidden states of BERT are sufficient for solving both Math23k (Wang, Liu, and Shi 2017) and Ape-clean datasets (Liang et al. 2022). This experiment indicates that the hidden states of BERT can be significantly compressed while retaining the ability of mathematics understanding, simply through a linear transform. Additionally, we infer that almost all reduced vector representations can retain task-specific capabilities by using the paradigm of linear layer and task-specific finetuning.

# Bottleneck Network

The bottleneck network consists of two linear layers Linear $( m , n )$ and Linear $( n , m )$ stacked together. Inspired by the usage of linear layers for dimensionality reduction, we recognize that the bottleneck linear network is important for distilling a compact model from the teacher model when they have different hidden states. In this paper, we leverage this simple component to improve the distillation outcomes and efficiency. We add bottleneck linear networks to several designated intermediate layers of the teacher model when it is fine-tuned. This approach enables the effective transfer of the teacher’s knowledge to the student model by reducing trainable parameters during the distillation stage.

# Deep Distillation for Solving MWP

We focus on compressing the depth and width of the BERT model, which is more difficult than only compressing one of them.

# Distillation Scheme Design

Model Architecture There are multiple distillation architectures, including multi-teacher distillation, distillation with a teacher assistant, mutual distillation, and selfdistillation, etc. We focus on a distillation architecture that does not require excessive additional model components, distillation procedures, or datasets. Therefore, we choose to mine the knowledge from a single teacher to avoid the complexity of multi-teacher learning and utilize the teacherstudent distillation paradigm. To eliminate the gap between the student model and teacher BERT, we employ the same transformer encoder architecture as the backbone of the student model, consisting of 3 encoder layers with a hidden size of 312. The feed-forward size is set to 1200, and the number of attention heads is set to 12. We use bottleneck linear networks to transfer the teacher’s knowledge, the distillation mechanism is shown in Figure 2.

Fundamental Knowledge for MWP Understanding Various types of knowledge can be distilled from LLMs, however, there is no definitive guidance on which knowledge type is more effective (Hu et al. 2023) or how to integrate this knowledge for MWP solving. Part of speech (POS) is the fundamental knowledge for mathematics problem understanding, as some higher-level semantics are redundant for solving elementary math problems.

Considering a problem described as $^ { \mathfrak { s } } Q$ : Tom has four apples and two pears; how many fruits does he have in total?” In this problem, the entities “apple” and “pear” could be replaced with other fruits without affecting the final solution. Therefore, the problem text representations generated by LLMs are redundant for MWPs, and constructing a lightweight encoder is a feasible alternative. As mentioned in the example above, we are not concerned with the specific names of some entities; we only need to ensure that they are all fruits and their mathematics relationships. Thus, POS represents an appropriate level of knowledge granularity that should be preserved in compact models. We also conducted experiments to verify this statement, the results are shown in Table 2, where the dataset used for this experiment is extracted from the problem text of Math23k and annotated using the LTP tool (Che et al. 2021), and $\mathbf { B E R T _ { \mathrm { { m a t h } } 2 3 \mathrm { { k } } } }$ represents the BERT full fine-tuned on Math23k. We know that the model will adjust its parameters when fine-tuned on the downstream tasks. It can be observed that BERT full fine-tuned on Math23k does not lose its ability to identify different POS types; in other words, solving MWPs requires retaining POS knowledge. We also conducted an experiment with the compact model to investigate the impact of POS knowledge. Table 3 demonstrates that the compact model exhibits improved performance on Math23k after pretraining with the POS tagging task. This further confirms that POS knowledge is fundamental to solving mathematics problems.

Keywords that represent mathematics relations are also crucial for MWP understanding. In the given question Q above, the keywords indicating addition operation are “and” and “in total.” We can summarize these keywords to raise the attention of the model on them. In addition to POS and keywords, solving MWPs also demands an understanding of sentence structure and mathematics semantics. However, this knowledge is challenging to concretize, yet it can be extracted from specific layers of the teacher model through distillation techniques. Several studies indicate that the uppermiddle layers of BERT are more effective than the top layer in guiding the student model. We aim to determine whether this phenomenon is observed in MWP tasks. We tested BERT with different layers to identify which layers are most effective, and the results are presented in Table 4.

![](images/4135b245a4fa65f513bfe8c9af3f851790b233356d8190126f85f5d4cfa16ecd.jpg)  
Figure 2: The proposed distillation mechanism for MWP representations operates on a 3-layer student network, including multiple knowledge extraction via the bottleneck layers.

Table 2: Performance of BERT on POS tagging with and without fine-tuning on math-solving tasks.   

<html><body><table><tr><td>Model</td><td>F1</td><td>Precision</td><td>Recall</td></tr><tr><td>BERTbase</td><td>79.51%</td><td>76.98%</td><td>82.22%</td></tr><tr><td>BERTmath23k</td><td>79.51%</td><td>76.98%</td><td>82.22%</td></tr></table></body></html>

Table 3: The performance of the compact model on Math23k with and without POS tagging pretraining.   

<html><body><table><tr><td>Model initialization</td><td>Equation Acc.</td><td>Answer Acc.</td></tr><tr><td>w/o POS tagging</td><td>45.59%</td><td>53.11%</td></tr><tr><td>w/ POS tagging</td><td>61.42%</td><td>71.15%</td></tr></table></body></html>

It shows that the lower layers play important roles in MWP understanding. The first four layers account for $8 4 . 6 1 \%$ of the overall performance of the BERT model, with the first layer alone contributing $7 2 . 0 8 \%$ . These lower layers typically learn abstract knowledge such as morphology and syntax, underscoring the importance of fundamental knowledge for solving MWPs. The remaining eight layers account for the remaining $1 5 . 3 9 \%$ of performance. These layers, particularly the last few, usually learn task-specific knowledge, such as mathematics entity relationships in fine-tuned BERT. This phenomenon is consistent with the principle of diminishing marginal utility.

Distillation Procedure To achieve generalization and capture diverse knowledge, we follow the paradigm of combining pre-training with task-specific distillation. During the pre-training stage, the compact model learns elementary knowledge through Masked Language Modeling (MLM) and POS tagging tasks to enhance generalization. Subsequently, math task-specific distillation from the teacher model equips the student model with MWP understanding capabilities. Figure 3 shows the proposed distillation procedure. To prevent catastrophic forgetting of the acquired knowledge, we employ a simple continual learning approach between pre-training and distillation. We adopt a straightforward method inspired by Wortsman (Wortsman et al. 2022), which involves assembling the weights of the pre-training and distillation models. We use this approach to yield two compact models at different stages, with the final model’s weights derived by averaging the parameters of both. That is, $\bar { P } _ { f i n a l } = \theta \bar { P _ { p r e t r a i n g } } \bar { + } ( \bar { 1 } - \theta \bar { ) } \bar { P } _ { d i s t i l l a t i o n } .$ .

Table 4: BERT with different layers is employed to evaluate mathematics problem-solving ability on Math23k.   

<html><body><table><tr><td>Layers</td><td>Equation Acc.</td><td>Answer Acc.</td><td>Perf.Prop. of BERT</td></tr><tr><td>1</td><td>51.00 %</td><td>60.52 %</td><td>72.08%</td></tr><tr><td>1-2</td><td>57.92%</td><td>67.84%</td><td>79.00%</td></tr><tr><td>1-3</td><td>59.21%</td><td>71.04%</td><td>80.79%</td></tr><tr><td>1-4</td><td>62.72%</td><td>73.55%</td><td>84.61%</td></tr><tr><td>1-6</td><td>66.83%</td><td>78.36%</td><td>87.59%</td></tr><tr><td>1-8</td><td>67.43%</td><td>79.46%</td><td>93.32%</td></tr><tr><td>1-10</td><td>69.64%</td><td>81.06%</td><td>94.63%</td></tr><tr><td>1-12</td><td>84.07%</td><td>84.33%</td><td>100%</td></tr></table></body></html>

![](images/1d6c5d5f08eba16b4d638090225aab13e2c51da9600bd7c8b3ab4c65264f6d96.jpg)  
Figure 3: The proposed distillation scheme.

Tomhas fourapplesand two pears;how many fruits doeshe have in total? Tom has [NUM] apples and [NUM] pears; how many fruits does he have in total? Tom has Optional apples and Optional pears; Optional fruits does he have Optional?

Figure 4: Tailored MLM task for MWP understanding.

# Pretraining Tasks for Basic Problem Understanding

The student model needs the right initialization for convergence and obtaining the elementary knowledge for solving MWPs. The traditional pretraining task is MLM, enabling the model to learn more semantic knowledge from large corpus. To understand MWPs more effectively, we pay more attention to the mask of mathematical keywords and numbers, these are the important components that distinguish MWPs from general problems. Figure 4 provides an example, where [NUM] is the number placeholder and underlined positions indicate that they can be substituted with masks.

We conduct MLM and POS tagging tasks on the large MWP dataset in the pretraining stage. The POS tagging datasets are obtained using the latest Chinese NLP tool called LTP. We preprocess the MWP datasets by extracting problem texts and performing POS tagging. In addition to Math23k and Ape210k, we also used HMWP (Qin et al. 2020) and CM17k (Qin et al. 2021) as the dataset during the pre-training phase. POS tagging requires word segmentation first, therefore we use the LTP tool to obtain the text after word segmentation and then use ‘B-’ to denote the beginning of words and ‘I-’ for words that continue. Next, we count the number of POS types in the datasets and add a classifier to the student model to predict these tags.

For MWPs, nouns (including temporal nouns), quantity, verbs, punctuation, and pronouns are important for understanding these problems. Among these, nouns might represent entities involved in mathematics operations, quantifiers might include unit conversions, and verbs could indicate mathematics operations. Focusing the student model on these POS types is fundamental to uncovering the key mathematics relations contained in the problem texts. The pre-training loss comprises the MLM task loss and the POS type prediction loss, with a 1:5 ratio between the two. We find that learning MLM and identifying POS from the MWP corpus can give the compact model the preliminary ability to solve MWPs.

Intuitively, some task-agnostic compact models have been pre-trained on large language corpus, hence we can use them to initialize our model. However, this paper focuses on discussing the effectiveness of our pre-training methods and does not utilize the knowledge of existing small models. Incorporating the knowledge from existing compact models can further enhance the capabilities of our model, a topic to be discussed in future work.

# Task-specific Distillation Intermediate Feature Distillation

Intermediate-layer knowledge plays an important role in distillation progress, which is crucial to eliminating the gap between student and teacher models. According to the literature (Liu et al. 2021), attention knowledge is not desirable information for distillation. Although the most attended token may contain important information, this may also hinder the student model from learning more crucial knowledge. For instance, [SEP] token may gain more attention, however, some trivial knowledge in its representation makes the attention distillation perform unsatisfactorily. Hidden states could contain rich semantic knowledge for better problem understanding, and implicit knowledge that might be used to solve the problems. We choose hidden states of intermediate layers for feature distillation. The loss function is:

Table 5: Performance of the model with hidden-state distillation on Math23K, obtained under different layer mappings.   

<html><body><table><tr><td>Changed S-T layer mapping</td><td>Fixed S-T layer mapping</td><td>Equation accuracy</td><td>Answer accuracy</td></tr><tr><td></td><td>1-4, 2-8, 3-12 (initialization)</td><td>63.33 %</td><td>74.15 %</td></tr><tr><td>1-1</td><td>2-8,3-12</td><td>61.32 %</td><td>73.65 %</td></tr><tr><td>1-2</td><td>2-8,3-12</td><td>61.52 %</td><td>73.95 %</td></tr><tr><td>1-3</td><td>2-8,3-12</td><td>62.53 %</td><td>73.75 %</td></tr><tr><td>2-5</td><td>1-4,3-12</td><td>60.82 %</td><td>71.74 %</td></tr><tr><td>2-6</td><td>1-4, 3-12</td><td>61.82 %</td><td>72.44 %</td></tr><tr><td>2-7</td><td>1-4, 3-12</td><td>61.92 %</td><td>74.05 %</td></tr><tr><td>3-9</td><td>1-4,2-8</td><td>61.82 %</td><td>71.94 %</td></tr><tr><td>3-10</td><td>1-4,2-8</td><td>61.12 %</td><td>72.34 %</td></tr><tr><td>3-11</td><td>1-4,2-8</td><td>61.92 %</td><td>72.44 %</td></tr></table></body></html>

$$
\mathcal { L } _ { H i d d e n } = \sum \alpha _ { n } \mathcal { L } _ { H } ( H _ { l } ^ { S } , H _ { l ^ { \prime } } ^ { T } W ) ,
$$

where $H ^ { S }$ and $H ^ { T }$ are the hidden states from the student and the teacher models, respectively. The symbols $l$ and ${ { l } ^ { \prime } }$ respectively denote the ith layer of the student model and the $i ^ { \prime }$ th layer of the teacher model. The loss function we use is mean squared error (MSE). $\alpha _ { n }$ denotes the weight assigned to each loss. $W \in \mathbb { R } ^ { d ^ { \prime } \times d }$ is a linear matrix used to match the teacher’s hidden size with that of the student. The layer mapping function is used to choose layers of the teacher model to match the student layers. We choose several layer mapping strategies, and their performance is summarized in Table 5 (S: student, T: teacher). The initial mapping is $S - T = \{ ( 1 , 4 ) , ( 2 , 8 ) , ( 3 , 1 2 ) \}$ , where $S - T$ is a pair of numbered layers from student and teacher respectively. When a pair of layer mapping is changed, the other two pairs are retained.

We observe that the student model performs better when its first layer learns from the initial four layers of the teacher model, and similarly, its performance improves when its final layer learns from the last layer of the teacher model. This aligns with our previous analysis regarding the distribution of knowledge types within the teacher model. Therefore, it is recommended that the student model primarily focuses on the information from the beginning and ending layers of the teacher model. Additionally, the middle layer of the student model should also learn from the middle layer

![](images/f539386cb1f93d13bfbb6c8956a08ea1fe9fa75fe760c2a0adf5afd50021b57d.jpg)  
Figure 5: Extracting relevant knowledge for MWP solving from specific layers of the teacher model.

<html><body><table><tr><td>Knowledge type</td><td>Equation Acc.</td><td>Answer Acc.</td></tr><tr><td>attention matrix</td><td>58.72%</td><td>68.74%</td></tr><tr><td>FSP matrix</td><td>58.52%</td><td>68.44%</td></tr><tr><td>NST matrix</td><td>56.01%</td><td>66.43%</td></tr><tr><td>hidden states</td><td>63.33 %</td><td>74.15 %</td></tr></table></body></html>

Table 6: The impact of different knowledge transfers on distillation outcomes on Math $2 3 \mathrm { k }$ dataset.

of the teacher model. We conjecture that this phenomenon is caused by the knowledge gap between different layers, that is, the student cannot learn the knowledge spanning too many teacher layers. Thus, we adopt a uniform mapping strategy for intermediate-layer distillation, that is, the mapping function is employed as:

$$
l ^ { \prime } = n \cdot ( l _ { T } / l _ { S } )
$$

where $n \in \{ 1 , 2 \dots , l _ { S } \}$ and $\mod ( l _ { T } , l _ { S } ) = 0$ , $l _ { S } \ \leq \ l _ { T }$ . The distillation from intermediate layers utilizing a uniform mapping is illustrated in Figure 5. Since we pay more attention to the first and last layers, we assign weights of 1.0, 0.9, and 1.0 when calculating the hidden loss for the 3-layer compact model.

The result shows that the compact model can show good performance only by conducting intermediate-layer distillation from hidden states. There are other intermediate knowledge types, such as attention matrix, feature map, FSP matrix, etc. We also use these knowledge types for intermediate-layer distillation experiments. The result in Table 6 shows that relation-based knowledge is not suitable for solving MWP tasks.

We also distill the knowledge from the teacher’s embedding layer and the loss function is:

$$
\mathcal { L } _ { E m b d } = \mathcal { L } _ { H } ( E ^ { S } W _ { e } , E ^ { T } ) ,
$$

where $E ^ { S }$ and $E ^ { T }$ are the embeddings of the student and the teacher models, respectively. We also take the MSE as the loss function, similar to hidden-layer distillation. Thus, the feature distillation is:

$$
\mathcal { L } _ { F e a } = \mathcal { L } _ { E m b d } + \mathcal { L } _ { H i d d e n } .
$$

# Output Prediction Distillation

We also use the conventional distillation technology to distill output knowledge from the prediction layer of BERT.

General distillation approaches distill logits (also called soft labels) of the last fully connected layer which usually is a classifier, and ground-truth labels (also called hard labels) are also used combined with soft labels. For MWP tasks, we employ the GTS decoder to generate output logits as the soft labels, often defined by a softmax function as:

$$
p ( z _ { i } , T ) = \frac { \exp ( z _ { i } / T ) } { \sum _ { j } \exp ( z _ { j } / T ) } ,
$$

where $z _ { i }$ is the $\textit { i } - \textit { t h }$ logit and $T$ is the distillation temperature used to smooth the probabilities. The output of GTS can be simply regarded as the function of $\mathbf { \dot { \boldsymbol { w } } } ^ { T } \operatorname { t a n h } ( \mathbf { W } [ \mathbf { q } , \mathbf { c } , \mathbf { e } ( y | P ) \hat { \ ] } )$ , where $\mathbf { q }$ is a goal vector, $\mathbf { c }$ is a context vector containing the information of problem $P$ , and $\mathbf { e }$ is the token embeddings of operators and quantities. GTS decoder can generate tree-structured outputs, thus it can be used to evaluate the gap between the generated expression tree and the equation annotated in datasets. The hard labels, namely the labeled solutions (expression trees) of MWP datasets are also used as the teacher knowledge. Thus, the prediction loss can be represented by:

$$
\mathcal { L } _ { P r e d } = \lambda \mathcal { L } _ { K D } \left( \frac { z _ { T } } { T } , \frac { z _ { S } } { T } \right) + ( 1 - \lambda ) \mathcal { L } _ { S } ( y , p ( z _ { s } ) ) ,
$$

where $z _ { T }$ and $z _ { S }$ are the outputs from teacher and student models, respectively, $\lambda$ is a hyperparameter. The second term $\mathcal { L } _ { S }$ is often called the student loss, where $y$ is the hard label. $\mathcal { L } _ { K D }$ is the loss function, usually the cross-entropy loss. Thus, the total loss is the sum of the losses of each knowledge type:

$$
\mathcal { L } _ { T o t a l } = \mathcal { L } _ { F e a } + \mathcal { L } _ { P r e d } .
$$

# Comparison with Previous Work

Most existing works focus on task-agnostic compact models, aiming to reduce their size and replace general LLMs. These models have achieved good results in GLUE tasks. We compare our model with these compact models, and the result is presented in Table 7, where all models employ the GTS decoder to ensure fairness. $L$ denotes the number of transformer layers and $H$ represents the hidden size. TinyBERT applies Transformer distillation during both the pretraining and task-specific learning stages. Tiny $\mathrm { B E R T _ { 4 } }$ comprises 4 layers and achieves $9 6 . 8 \%$ of BERT’s performance on the GLUE benchmark. DistillBERT is initialized with the teacher BERT’s parameters and retains the same hidden size as the teacher. MiniLM distills only the last Transformer layer of the teacher model but comprises 6 layers with a hidden size of 384. Our model outperforms these mainstream compact models on math datasets while maintaining the smallest model size.

# Experiments

# Datasets

Our experiments mainly use four commonly used Chinese MWP datasets. Math23k is the most widely used dataset which contains 23162 math application problems with annotated equations and answers. Ape210k is an enormous math dataset including 210488 MWPs. Since Ape210k has many noisy examples that miss annotations or cannot be solved, we use the re-organized datasets called Ape-clean (Liang et al. 2022) and full Ape210k can still be used for MLM pretraining. HMWP consists of 5470 MWPs including multi-unknown problems and non-linear problems, making problem solving more challenging. CM17K is another large-scale MWP dataset, which contains 6215 arithmetic problems, 5193 one-unknown linear problems, 3129 oneunknown nonlinear problems, and 2498 equation set problems. Because solving CM17K and HMWP needs another special decoder, we only use them in the pretraining stage. The solving performance of compact models is evaluated on Math $2 3 \mathrm { k }$ and Ape-clean.

Table 7: Comparison among the publicly released compact models distilled from BERT.   

<html><body><table><tr><td>Models</td><td>Architecture</td><td># Params</td><td># FLOPs</td><td>Acc.onMath23k</td><td>Acc. on Ape-clean</td><td>Average</td></tr><tr><td>BERT(Teacher)</td><td>L =12,H= 768</td><td>109M</td><td>22.5B</td><td>84.07%</td><td>84.33%</td><td>84.20%</td></tr><tr><td>DistilBERT</td><td>L=6,H=768</td><td>67M</td><td>11.3B</td><td>62.12%</td><td>75.71 %</td><td>68.92%</td></tr><tr><td>MiniLM</td><td>L=6,H=384</td><td>66M</td><td></td><td>58.67%</td><td>75.87%</td><td>67.27%</td></tr><tr><td>TinyBERT</td><td>L=4,H=312</td><td>14.5M</td><td>1.1B</td><td>55.41%</td><td>72.98%</td><td>64.12%</td></tr><tr><td>Our work</td><td>L=3,H=312</td><td>10.3M</td><td>1.1B</td><td>80.23%</td><td>80.46%</td><td>80.35%</td></tr></table></body></html>

# Implementation Details

We employ the fine-tuned version of Chinese pre-trained BERT with whole word masking (Cui et al. 2021) as the teacher model, which has a 12-layer Transformer encoder with 768 hidden states and 12 attention heads. Our model is implemented by PyTorch on an NVIDIA A800 100 GB GPU. At the pretraining stage, 150 epochs are trained using the Adam optimizer with the initial learning rate of 1e-5 and weight decay of 1e-5, the mini-batch size is set to be 128. At the task-specific distillation stage, we also use the Adam optimizer, and the initial learning rate is set as 3e-5, and we pre-train them 120 epochs. The loss weights $\alpha _ { 1 }$ , $\alpha _ { 2 }$ , and $\alpha _ { 3 }$ of our distillation loss obtained by grid search are set as 1.0, 0.9, and 1.0. According to our extensive experiments, the hyperparameter $\theta$ and $\lambda$ are set as 0.2 and 0.5, respectively. The temperature factor for soft labels is set to 4. We fine-tune the student model using a batch size of 32 for 100 epochs, and the dropout rate is 0.1. The initial fine-tuning learning rate is set as 1e-5 and 1e-4 for the student model and GTS, respectively. Other compact models are implemented according to their specific settings in previous literature.

# Ablation Studies

We conduct ablation studies to analyze the contributions of main components in the distillation scheme. We mainly focus on different procedures of the proposed distillation and different distillation knowledge. The Results are presented in Table 8. It indicates that pretraining tasks are crucial to enhancing the efficacy of the proposed method, and the POS tagging task positively contributes to the pretraining procedure. In terms of the proposed distillation objectives, all of them contribute to model performance, with intermediatelayer distillation proving more beneficial in task-specific distillation. We also investigate the contributions of soft and hard labels and find that hard labels contribute almost as much as soft labels. These two types of knowledge from the prediction layer are complementary to each other and are both important for the final distillation results.

Table 8: Ablation studies of different components.   

<html><body><table><tr><td>Model</td><td>Math23k</td><td>Ape-clean</td></tr><tr><td>w/o Pretraining</td><td>77.5%</td><td>78.4%</td></tr><tr><td>w/o POS tagging</td><td>78.1%</td><td>79.0%</td></tr><tr><td>w/o MLM</td><td>78.9%</td><td>79.6%</td></tr><tr><td>w/o Hidden Distillation</td><td>76%</td><td>77.4%</td></tr><tr><td>w/o Prediction Distillation</td><td>78.6%</td><td>79.1%</td></tr><tr><td>w/o soft label</td><td>79.2%</td><td>79.4%</td></tr><tr><td>w/o hardlabel</td><td>79.3%</td><td>79.8%</td></tr></table></body></html>

# Conclusion and Future Work

In this paper, we find that the representations of taskagnostic compact models are inadequate for solving MWPs. Thus, we propose a new compact model to facilitate the practical application of intelligent mathematics solvers. Pretraining and multi-knowledge distillation are utilized for math-related knowledge extraction and progressive transfer. Empirical results on commonly used MWP datasets demonstrate that our model achieves performance comparable to BERT, while the size of the model layer and hidden states can be much smaller. Extensive experiments reveal that 1) POS knowledge and hidden states are important for solving MWPs, 2) the uniform mapping principle is still effective for layer-level knowledge transfer, and 3) the compact model can benefit from both our tailored pretraining and distillation. This is the first work to develop a math-domain compact model, and we believe our model can facilitate both research and practical applications in MWP solving.

In future work, we will further refine the knowledge of mathematical relations within problems, utilizing a larger math-related corpus and a more robust teacher model for guidance. Concurrently, we will optimize the knowledge distillation techniques, including reducing the steps involved in distillation and employing smaller-scale models. We can use existing task-agnostic compact models as the initialization for our model to enhance generalization performance while reducing training costs. Additionally, we will explore the potential of other types of LLMs, such as GPT, in constructing compact models for solving math problems.