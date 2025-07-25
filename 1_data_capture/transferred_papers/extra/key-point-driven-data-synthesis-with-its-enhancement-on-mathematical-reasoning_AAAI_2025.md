# Key-Point-Driven Data Synthesis with Its Enhancement on Mathematical Reasoning

Yiming Huang\* , Xiao Liu , Yeyun Gong, Zhibin Gou, Yelong Shen, Nan Duan, Weizhu Chen

Microsoft yeeelow233@gmail.com, xiaoliu2@microsoft.com

# Abstract

Large language models have shown great potential in complex reasoning tasks, yet their performance is often hampered by the scarcity of high-quality and reasoning-focused training datasets. Addressing this challenge, we propose Key-PointDriven Data Synthesis (KPDDS), a novel data synthesis framework that synthesizes question-answer pairs by leveraging key points and exemplar practices from authentic data sources. KPDDS ensures the generation of novel questions with rigorous quality control and substantial scalability. As a result, we present KPMath, an extensive synthetic dataset tailored for mathematical reasoning, comprising over 800K questionanswer pairs. Utilizing KPMath and augmenting it with additional reasoning-intensive corpora, we create the comprehensive KPMath-Plus dataset. Our experiments demonstrate that this dataset can enhance the mathematical reasoning performance of models across various architectures and sizes. The Qwen1.5-72B model, fine-tuned on KPMath-Plus, achieves $8 7 . 0 \%$ accuracy on GSM8K and $5 8 . 3 \%$ on MATH, surpassing competitors in the 7B to 72B range and best commercial models like GPT-4 across multiple math reasoning datasets.

# Introduction

The recent advent of large language models (LLMs) such as GPT-4 (OpenAI 2023), Gemini (Team et al. 2023), and Mistral (Jiang et al. 2023) has sparked significant interest due to their advanced capabilities in diverse domains. Despite this, their reasoning process, particularly in challenging domains like advanced mathematics (Lewkowycz et al. 2022), competitive programming (Huang et al. 2023), and integrated vision-language planning (Cen et al. 2024), remains under scrutiny. In current mathematical reasoning corpora, such as OpenWebMath (Paster et al. 2023) and MathPile (Wang, Xia, and Liu 2023), the vast internet-sourced data often suffers from poor quality and relevance to the subject matter. Conversely, manually annotated high-quality datasets like the MATH dataset (Hendrycks et al. 2021) are scarce and sometimes lack detailed reasoning steps.

Prior efforts to boost the mathematical reasoning capabilities of LLMs using synthetic data have primarily adopted two strategies. The first strategy focuses on augmenting existing datasets. It involves question rephrasing (Yu et al. 2023) or generating similar questions (Luo et al. 2023a; Liu et al. 2024). The primary issue of this strategy is that the generated questions are not only textually or conceptually similar but also uncontrollable in their variations. The second strategy seeks to broaden the training dataset by generating new questions from knowledge concepts. Knowledge bases are either compiled from online educational resources, such as Khan Academy’s math courses (Huang et al. 2024), or synthesized from scratch using models like GPT-4 (Li et al. 2024). These methods depend on constructed knowledge that might not align with the existing dataset’s distributions and are difficult to comprehend without examples to illustrate the concepts.

Considering these disadvantages of the two strategies, we introduce a novel data synthesis paradigm termed Key-PointDriven Data Synthesis (KPDDS), which capitalizes on the strengths of both data synthesis strategies. As depicted in Figure 1, it delves into datasets for knowledge mining, using relevant key points and associated problems to inform the generation of new problems. (1) For knowledge construction, we begin by extracting topics and key points from seed problems using a labeling model, followed by a clustering algorithm to ensure deduplication and alignment. Therefore, we obtain the Math Practices with Key Points (MPKP) dataset and construct the Topic-level Co-occurrence Probability Matrix (TCPM) to understand the combination patterns of topics within the dataset. (2) For practice synthesis, we sample multiple topics and key points from MPKP using the TCPM as a guide. These key points, along with corresponding example practices, serve as input for the synthesizing model to generate new questions. A scoring model then assesses the quality of these questions, allowing only those with high scores to proceed. Then, a reasoning model generates a range of answer options, which are later consolidated into consensus solutions through a voting mechanism.

Utilizing the training sets of the MATH (Hendrycks et al. 2021) and GSM8K (Cobbe et al. 2021) datasets as foundational data, we developed a novel dataset named KPMath. Our training corpus was further enriched by integrating a series of mathematical reasoning datasets, leading to the creation of a comprehensive training dataset, KPMath-Plus. Our experiments demonstrate that this dataset can enhance the mathematical reasoning performance of models across

Knowledge Construction oPTpoioicnp1ti :c1A:1l:UgenAlbdgreaersb-traCano-dmiCnpogletmthipnelgeptrtihonecgeStsqhsueoafrSequare C:--+1 Labeling Model pKletyi nPgotihnte  1s:quUanrdeerstanding the process of ePrciorbclemwi:tFhienqdutahtieocnenter Pcoinmt p2l:eAtipnpglytihnge  tshqeumaret of the circle with equation opKmleptylienPtgiontighntet h2se:qsuAqaprupealrtyeio ntrgoe trwehriewterimtqeutqahudoraadtrioacft i ol\$uxt^io2n-:  6Cxo+myp^le2t i+n g2tyh=e  9\$. qcuoatmiopnlseting the square to rewrite quadratic qSuaorleu,tiwoen:geCto\$(mxp- l3e)t^in2g+the equations Clustering Estimating c+seqn1t)u^ear2roe=f, 1twh9e\$ .cgTierhct $\$ ( x - 3)$ $( y + 1 ) \wedge 2 = 1 9 \$ 1$ .irTchl eirsefore, eTyofPptoihicent2e1:q:uGUaentiodomenresotfraync–diirCnciglretchles \\tbhoexcede{n(t3e, r-1o)f}\$t.he circle is fPtKoheienytePq2o:uiIandttieo1nt:ifoUyfinadgcetirhscelteacnendtienrgotfhaecsirtcalnedard \$\\boxed{(3, -1)}\$. efyiotsPr omeiqnoutfa2t:ihIoednenqtiufyaitnigo tnhoefcaencitrecrloef Math Problems with Key Topic-level Co-occurrence Seeds oKmeiyt sPeoiqnutat2i:oIndentifying the center of a circle Points (MPKP) Probability Matrix from its equation (TCPM) Topics and Key Points for Math   
Practice Synthesis Sampled Topic 1: Number Theory - Digit Sums You have a number of books that you were planning to Sampled Key Point 1: Understanding of the relationship adiftsetri bdiuvtiedienqguyaolluyrabmooknsg  iynotuor15 ecqlausals pmilaetse,s.yoHuorweeavliezre, between the sum of digits of a number and the number itself in a given base. that if you give away all your books, 3 classmates will receive an extra book. You have less than 130 books. □ Sampled Key Point 2: Ability to calculate the sum of digits Sampling tPhriascptircoed:u…ctW?hat is the base seven sum of the digits of tbMhaesaedni8gwisthsuiloem,f?tahferiennudmobfeyroufrbsoaoskseydoyuohuatvhe.bWahseat8i stuhmisof SMcordienlg MPKP / Synthesizing Synthetic Problem Sampled Topic 2: Number Theory - Divisibility Reasoning Model 中 Sampled Key Point 1: Ability to solve for unknowns in Model ！ equations involving divisibility … so the number of books you have can be written in the TCPM Sampled Key Point 2: Understanding of divisibility and form $\$ 15 k +34$ . We have that $\$ 15k +3< 130\$ 1$ , so \$k $<$ remainders \frac{127}{15}\$. … Thus, the base 8 sum of the digits of the Practice:.... What is the largest number of gold coins you number of books you have is $\$ 1+7+3=160\times e d \{ 11\} \$ \Phi$ . The could have to cause this to happen? answer is: 11. Sampled Topics and Key Points with Practices Consensus Solution

various architectures and sizes. By fine-tuning the Qwen1.5- 72B model (Bai et al. 2023) on KPMath-Plus, we achieved zero-shot $\mathrm { P A S S } @ 1$ accuracies of $8 7 . 0 \%$ on the GSM8K test set and $5 8 . 3 \%$ on the MATH test set, culminating in a promising average of $8 1 . 5 \%$ across six math reasoning datasets. This performance exceeds that of all competitors within the 7B to 72B model size range and best commercial models like GPT-4. In the Hungarian Exam Score test, the KPMath-PlusMistral-7B model also outperforms the majority of models, indicating its competitive performance.

# Related Work

# Math Reasoning with LLMs

Recently, solving math problems is treated as an important aspect of evaluating LLM’s reasoning ability. However, the LLMs trained for general purposes like GPT-4 (OpenAI 2023), Llama2 (Touvron et al. 2023), Mistral (Jiang et al. 2023), Qwen (Bai et al. 2023), Gemini (Team et al. 2023) and DeepSeek (Bi et al. 2024) have shown limited capabilities in math reasoning. To enhance the math reasoning ability of LLMs, researchers have turned their attention to research directions like prompting methods (Chia et al. 2023; Zheng et al. 2023; Chen et al. 2023), data construction for pretraining (Taylor et al. 2022; Lewkowycz et al. 2022; Paster et al. 2023; Azerbayev et al. 2023) and instruction tuning (Yue et al. 2024; Yu et al. 2023; Luo et al. 2023a; Gou et al. 2024b; An et al. 2023; Liu et al. 2024; Huang et al. 2024; Li et al. 2024, 2023), interacting with external tools (Mishra et al. 2022; Gao et al. 2022; Gou et al. 2024a,b; Yue et al. 2024; Zhou et al. 2023; Zhang et al. 2024), and reinforcement learning with rewards (Lightman et al. 2023; Wang et al. 2023; Luong et al. 2024) for either outcomes or steps. This work is in line with math reasoning data construction for instruction tuning.

# Data Synthesis for Math Reasoning

In the realm of math reasoning, data synthesis is usually applied for instruction tuning, with each data sample encompassing a question text and its corresponding answer text. To advance this field, research efforts focus on three critical aspects: enhancing the quality of answers, generating novel questions, and implementing quality control measures.

For answer quality, some works focus on chain-of-thought (CoT) (Chia et al. 2023; Yu et al. 2023) style answers, while others investigate program-based answers. Yue et al. synthesize program-of-thought (PoT) (Chen et al. 2022) style answers using GPT-4. Gou et al. further explore interleaved answers with program-based tool use. In this work, we focus on the synthesis of CoT-style answers.

For question novelty, some works start from existing problems. Shao et al. (2023) explore answer-first data synthesis and Yu et al. (2023) utilize backward reasoning, while Luo et al. (2023a), An et al. (2023), and Liu et al. (2024) focus on evolution instruction and iterative composition using reasoning steps. Other works are grounded in knowledge. Huang et al. (2024) extracts concepts from Khan Academy and Li et al. (2024) uses GPT-4 to create a concepts taxonomy. The former approach is limited by poor scalability with existing data, and the latter often yields a synthetic data distribution that significantly deviates from real data. In our work, we create questions by extracting key points from real data and then synthesizing new problems based on these key points with authentic and reliable exercises.

For synthetic data quality, Huang et al. (2024) prompt GPT4 to convert CoT-style answers into verifiable Lean-3 code, while Trinh et al. (2024)’s AlphaGeometry ensures Euclidean geometry theorem accuracy using symbolic deduction. In contrast, We assess synthetic question and answer quality through GPT-4 scored evaluations and consensus scoring via repeated sampling.

# Data Synthesis for Other Applications

The aim of synthetic data is to offer a convincing and fuller depiction of the actual data source, maintaining key statistical characteristics such as the distribution patterns of continuous variables, categorical ratios, and the latent relationships among different variables. Except for math reasoning, there are also works on data synthesis for other applications like code (Luo et al. 2023b; Wei et al. 2023), table reasoning (Lei et al. 2023), medical application (Zhang et al. 2023; Tang et al. 2023), visual reasoning (Du et al. 2023), and general purposes (Xu et al. 2023; Li et al. 2024).

# Method

# Overview

The KPDDS methodology is systematically delineated into two primary phases: Knowledge Construction and Practice Generation, each consisting of two components. We will introduce these four components separately: Knowledge Extraction, Topic-level Co-occurrence Probability Matrix (TCPM) Construction, Question and Solution Generation.

# Knowledge Extraction

We employ GPT-4 to extract knowledge on each question and answer pair across two levels. The first level of knowledge is the topics, which correspond to the subject and its subcategories that are pertinent to the problem, such as ”Geometry - Circles”. The secondary level is key points (KPs), which comprise the theorems or methods essential for the resolution process, like ”Determining the center of a circle from its equation”. Figure 1 presents a complete example.

As a mathematics education specialist, please analyze the topics and key points of the provided question and its answer. These analysis should serve as a guide for teachers to craft analogous problems and as focal learning objectives for students when approaching the problem. Be sure to avoid repetition of Key Points for clarity and conciseness. Specific requirements are as follows: 1. Identify and categorize the main mathematical topics involved in the problem. If knowledge from non-mathematical fields is used, it is classified into Others - xxx, such as Others - Problem Context. 2. For each topic, enumerate the essential Key Points relevant to the problem.   
... [omitted two examples]

To ensure that the extraction captures a wide range of pertinent information without constraint, we allow the model

1: Initialize $T C P M$ matrix of size $N \times N$ with zeros   
2: for each document $d$ in dataset do   
3: for each topic $i$ in $d$ do   
4: for each topic $j$ in $d$ do   
5: if $i \neq j$ then   
6: $T C P M [ i ] [ j ] \mathrel { + } \mathrel { = } 1$   
7: end if   
8: end for   
9: if Number of KPs in topic $i > 5$ then   
10: $T C P M [ i ] [ i ] \mathrel { + } \mathrel { = } 1$   
11: end if   
12: end for   
13: end for   
14: $T C P M \gets \log _ { 1 0 } ( T C P M + 1 )$   
15: return T CP M

to freely identify and categorize relevant subjects instead of providing a predefined list of topics in the prompt. However, this approach can result in an extensive number of topics, many of which exhibit semantic overlap, such as ”Arithmetic - Percentages” and ”Arithmetic - Percentage.” Therefore, we further process the extracted knowledge data. Specifically, we use OpenAI’s text-embedding-ada-002 to embed all KPs, represent the topics by the average value of the embeddings of their included KPs. Then, we calculate the cosine similarity of the topic embeddings for deduplication and clustering, obtaining several representative topics. Finally, we construct the Math Practices with Key Points (MPKP) dataset.

# TCPM Construction

Mathematical problems typically involve multiple topics and KPs, and the combination of topics within these problems follows a discernible pattern. For example, semantically highly similar topics do not appear repeatedly in the same problem, whereas arbitrarily meshing unrelated topics tends to result in nonsensical questions. In light of this structured complexity, we compute the Topic-level Co-occurrence Probability Matrix (TCPM) from the topics present in mathematical questions within the MPKP dataset. Our methodology is systematically outlined in algorithm 1. This algorithm quantifies the co-occurrence and self-interaction of topics within a dataset by constructing a matrix that logs the frequency of topic pairs and the instances where the number of KPs for individual topics exceeds five, followed by a logarithmic normalization. An increased co-occurrence probability between topic clusters indicates a likelihood of their concurrent appearance in the examined problems. Figure 3 presents heat map visualization of the TCPM for GSM8K and MATH.

![](images/8147f688682bb2bbf4fc687f621a4cf437633a6c84e81b965f41030688c0a49b.jpg)  
Figure 2: Prompt snippet for knowledge extraction.   
Figure 3: Visualized heat map of Topic-level Co-occurrence Probability Matrix. Left: GSM8K (34 topics), Right: Math (119 topics).

# Question Generation

By extracting knowledge and constructing the TCPM from the seed problems, we pave the way for generating new problems that are similar yet varied in nature, building upon their foundational elements. Leveraging the TCPM, we perform probabilistic sampling of topics, with the probability calculation method as follows:

$$
V _ { n } = \left\{ \begin{array} { l l } { \sum _ { j } \mathrm { T C P M } _ { i j } , } & { \mathrm { i f } n = 1 , } \\ { \mathrm { T C P M } _ { T _ { i } , \cdot } , } & { \mathrm { i f } n = 2 , } \\ { \mathrm { T C P M } _ { T _ { n - 1 , \cdot } } \circ \mathrm { T C P M } _ { T _ { n - 2 , \cdot } } , } & { \mathrm { i f } n > 2 , } \end{array} \right.
$$

where $V _ { n }$ represents the vector used for probabilistic topic sampling, $i$ and $j$ are index variables, $T _ { i }$ denotes the $i$ -th topic, and $\mathrm { T C P M } _ { T _ { n } }$ , denotes the $n$ -th row vector in TCPM. $\scriptscriptstyle \mathrm { ~ o ~ }$ denotes the Hadamard product (element-wise multiplication).

We proceed to sample two to three topics, and for each topic, we randomly select a problem along with the associated KPs for that topic. This process yields a foundational KPs-Practices information set as the basis for our problem generation. Employing GPT-4, we use this set to generate new problems, with the prompt presented in Figure 4.

You are a math teacher. Now, you need to help your students to learn the following math knowledge. There are some key points and example problems:   
$\{ 2$ or $3 \mathrm { K P s }$ -Practices information sets   
Using these key points and example problems as a guideline, please construct a new, original math problem that requires an understanding and application of all the {len of selected kps} knowledge points.   
{selected kps}   
Write your new math problem, using $\mathrm { < Q > }$ and $\mathrm { < } / \mathrm { Q } \mathrm { > }$ to indicate the question.

Following the generation of problems, we conduct a quantitative evaluation to determine the quality of each problem by GPT-4. This assessment is based on two criteria: the presence of the provided KPs and the absence of logical or factual errors. Each problem is assigned a quality score on a continuous scale from 0 to 1. In assembling quality-assured questions, a threshold of 0.85 is instituted to screen the newly generated problems, saving about $51 \%$ high-quality question. Figure 5 displays an example of a high-quality and a poor-quality problem originating from identical initial inputs.

# Solution Generation

Prior work in the domain often lacked comprehensive quality control measures, relying heavily on answers generated by models like GPT-4 without additional verification. Our

High-quality $\checkmark$ You have a number of books that you were planning to distribute equally among your 15 classmates. However, after dividing your books into 15 equal piles, you realize that if you give away all your books, 3 classmates will receive an extra book. You have less than 130 books. Meanwhile, a friend of yours asked you the base 8 sum of the digits of the number of books you have. What is this base 8 sum?

Poor-quality $x$ You discover less than 1000 gold coins and plan to share them with 13 hunters, giving 3 extra coins to some. An ancient riddle claims the sum of the coins’ digits in base 13 equals the coins’ remainder when divided by 10. What’s the maximum number of coins possible?

methodology integrates three key strategies to improve answer correctness: few-shot learning to guide the initial response generation, computational verification to validate the answers mathematically, and a consensus voting mechanism to ensure the accuracy of the results. This multi-faceted approach is designed to minimize the impact of noisy data and enhance the reliability of the answer generation process.

For each synthesized question, we utilize the same KPsPractices information set that were employed in its creation as demonstration inputs. This approach ensures that the examples, which inherently share key points with the question, significantly enhance the quality of the model’s responses by aligning closely with the problem’s structure and content. To generate a diverse array of CoT rationales, we employ nucleus sampling and configure GPT-4 with a temperature setting of 0.75 and a top-p value of 0.95. For each question, we derive 10 potential responses. Next, we extract mathematical expressions from each solution and use a Python program to verify their accuracy, excluding any solutions with computational errors. Additionally, we exclude data containing more than three sub-questions to maintain analytical clarity.

Finally, we use a voting mechanism to aggregate the solutions. The voting mechanism leverage packages such as sympy 1 to ensure that equivalent answers, albeit in different forms (e.g., fractions and decimals), are recognized as equal. Consider a problem $p$ with $n$ sub-questions that has $m$ potential solutions. Let $s _ { i }$ denote one of these solutions, and each solution $s _ { i }$ addresses the $n$ sub-questions within the problem. The Consensus Score Vector (CSV) for a solution $s _ { i }$ is defined as:

$$
\mathrm { C S V } ( s _ { i } ) = [ c _ { i 1 } , c _ { i 2 } , \ldots , c _ { i n } ]
$$

where each $c _ { i j }$ represents the consensus score for the $j$ -th sub-question in solution $s _ { i }$ . The range of each $c _ { i j }$ is from $\frac { 1 } { m }$ to 1. A score of 1 indicates that all solutions agree with $s _ { i }$ for the sub-question. A score of $\textstyle { \frac { 1 } { m } }$ indicates that $s _ { i }$ is the only solution with its particular answer, showing no agreement with other solutions. The Consensus Score (CS) for a solution $s _ { i }$ is represented by the maximum value in its CSV:

$$
C S ( s _ { i } ) = \operatorname* { m a x } ( \mathbf { C S V } ( s _ { i } ) )
$$

Define $C S _ { \mathrm { m a x } }$ as the highest consensus score among all solutions for problem $p$ :

$$
C S _ { \mathrm { m a x } } = \operatorname* { m a x } ( \{ C S ( s _ { 1 } ) , C S ( s _ { 2 } ) , \ldots , C S ( s _ { m } ) \} )
$$

A solution $s _ { i }$ is retained if it meets the following criteria:

$$
\begin{array} { r } { \mathrm { R e t a i n } ( s _ { i } ) = \left\{ \begin{array} { l l } { \mathrm { Y E S , } } & { \mathrm { i f } C S ( s _ { i } ) = C S _ { \mathrm { m a x } } \mathrm { a n d } C S ( s _ { i } ) > \frac { 1 } { m } , } \\ { \mathrm { N O , } } & { \mathrm { o t h e r w i s e . } } \end{array} \right. } \end{array}
$$

# KPMath-Plus Dataset

# Dataset Construction

The KPMATH-Plus dataset is composed of three distinct components, collectively encompassing a total of 1,576K data points. We use min-hash techniques to minimize redundancy and exclude entries featuring excessively long numbers. Here are the unique attributes of each segment:

KPMATH-G (613K) This segment is based on the GSM8K (Cobbe et al. 2021) training set, which offers 7,473 samples of grade school math problems characterized by their 2 to 8 step solutions. KPMATH-G focuses on basic arithmetic operations within various contexts,

KPMATH-M (252K) This segment is based on the MATH (Hendrycks et al. 2021) dataset’s training set, which consists of 7,500 samples from high school math competitions, encompassing seven subjects and five difficulty levels. KPMATH-M includes more challenging problems in areas such as algebra, calculus, and geometry, designed to improve advanced mathematical thinking and problem-solving skills.

MixMath (711K) To ensure diversity and quality, we curated a comprehensive collection from various high-quality open-source mathematical reasoning datasets. The collection encompasses the complete datasets of MetaMath (Yu et al. 2023), MMIQC (Liu et al. 2024), and Open-Platypus (Lee, Hunter, and Ruiz 2023), in addition to the training sets of GSM8K (Cobbe et al. 2021), MATH (Hendrycks et al. 2021), TAL-SCQ5K-EN2, and MathInstruct (Yue et al. 2024).

# Human Evaluation of Data Accuracy

To validate our methods, we conducted human evaluation on a random sample of 100 questions from our generated dataset. Specifically, the KPMATH-G subset achieved a correctness rate of $9 5 \%$ , with observed errors primarily in problem statements or logical reasoning rather than calculation errors. The KPMATH-M subset demonstrated an $81 \%$ correctness rate, with the criterion that a multi-question problem is considered correct if at least one response is accurate. This assessment yielded a high accuracy rate, confirming the reliability of our automated verification and filtering techniques.

# Data Contamination Test

To mitigate data contamination risk in our benchmark, we used the method by Azerbayev et al. (2023) to scrutinize $n$ - gram overlaps between our dataset and the Math and GSM8K test sets. We checked for 20-gram overlaps in questions and 30-gram in solutions. Our analysis revealed no overlaps in GSM8K and fewer overlaps in Math compared to its training set, with 102 hits in questions and 108 in solutions, against the training set’s 181 and 144, respectively. A manual review confirmed these hits were due to recurring problem contexts or reasoning steps, not exact duplicates, indicating a low risk of contamination for KPMath.

# Experiment Implementation Details

In our supervised fine-tuning (SFT) experiments, we employed chat message templates to transform question-answer pairs into the format: “User: question nEnclose the final answer using \boxed{}.\n\nAssistant: {answer}”. We utilized the LLaMa-Factory repository (Zheng et al. 2024) to fine-tune the models for 3 epochs across all experiments. We adopted a linear learning rate schedule with a $3 \%$ warm-up ratio. The maximum learning rate is 1e-5, except for DeepSeekMath, which is 5e-5. We trained all models with BFloat16 numerical format, DeepSpeed ZeRO Stage3 (Rajbhandari et al. 2021) and Flash-Attention 2 (Dao 2023). For evaluation, we adopted the same template in SFT to prompt all questions. We employed greedy decoding with a maximum sequence length of 2,048 tokens.

# Evaluation and Metrics

We evaluate our fine-tuned models on GSM8k (Cobbe et al. 2021) and MATH (Hendrycks et al. 2021), along with 4 outof-distribution datasets, namely SVAMP (Patel, Bhattamishra, and Goyal 2021), ASDIV (Miao, Liang, and Su 2021), TabMWP (Lu et al. 2022), MAWPS (Koncel-Kedziorski et al. 2016). We utilize an enhanced version of the script from Gou et al. (2024b) to extract answers, parse expressions, and compare the equivalency of the answers. Additionally, we test the Hungarian Exam, adhering to the evaluation methodology proposed by Paster (2023), which segments the exam into 33 challenging problems suitable for model processing. These problems require manual answer verification. We report the zero-shot $\mathrm { P A S S } @ 1$ accuracy in all experiments.

# Baselines

We present results from a range of state-of-the-art (SoTA) proprietary LLMs, including OpenAI’s GPT-4 (OpenAI 2023), ChatGPT (gpt-3.5-turbo), Google’s PaLM-2 (Anil et al. 2023), and Anthropic’s Claude-2 (Anthropic 2023). Regarding open-source models, we consider base models such as Llama 2 (Touvron et al. 2023), Llama 3 (Dubey et al. 2024), DeepSeekMath (Shao et al. 2024), Mistral(Jiang et al. 2023), Llemma (Azerbayev et al. 2023), and Qwen1.5(Bai et al. 2023). Supervised Fine-Tuning (SFT) uses CoT rationales from the original GSM8k and MATH training sets (15K samples) for fine-tuning. We also showcase the performance of advanced models using SFT or RLHF on various mathematical reasoning datasets, including MAmmoTH (Yue et al. 2024), WizardMath (Luo et al. 2023a), Platypus-2 (Lee, Hunter, and Ruiz 2023), MetaMath (Yu et al. 2023) and MMIQC (Liu et al. 2024).

<html><body><table><tr><td>Model</td><td>Base</td><td>Size</td><td>ZS</td><td>GSM8k</td><td>MATH</td><td>SVAMP</td><td>TabMWP</td><td>ASDiv</td><td>MAWPS</td><td>AVG</td></tr><tr><td colspan="9">Proprietary Models</td></tr><tr><td>GPT-4 (0613)</td><td>-</td><td>1</td><td>×</td><td>92.0 42.5</td><td></td><td>93.1</td><td>67.1</td><td>91.3</td><td>97.6</td><td>80.6</td></tr><tr><td>ChatGPT (gpt-3.5-turbo)</td><td>-</td><td>=</td><td>×</td><td>80.8</td><td>35.5</td><td>83.0</td><td>69.1</td><td>87.3</td><td>94.6</td><td>75.1</td></tr><tr><td>Claude-2 PaLM-2</td><td></td><td></td><td>×</td><td>85.2</td><td>32.5</td><td>=</td><td>=</td><td></td><td>=</td><td></td></tr><tr><td></td><td>-</td><td>540B</td><td>×</td><td>80.7</td><td>34.3</td><td></td><td>-</td><td>1</td><td></td><td>1</td></tr><tr><td colspan="9">Open-Source Models</td></tr><tr><td>Llama-2</td><td></td><td>7B</td><td>X</td><td>13.3 4.1</td><td>38.0</td><td></td><td>31.1</td><td>50.7</td><td>60.9</td><td>33.0</td></tr><tr><td>Llama-2 SFT</td><td>1</td><td>7B</td><td>√</td><td>41.3</td><td>7.2</td><td>31.9</td><td>27.8</td><td>47.4</td><td>60.0</td><td>35.9</td></tr><tr><td>Platypus-2</td><td>Llama-2</td><td>7B</td><td>×</td><td>14.4</td><td>5.4</td><td>36.7</td><td>26.5</td><td>47.9</td><td>58.4</td><td>31.6</td></tr><tr><td>MAmmoTH</td><td>Llama-2</td><td>7B</td><td>√</td><td>45.9</td><td>7.3</td><td>48.7</td><td>28.9</td><td>62.3</td><td>74.8</td><td>44.7</td></tr><tr><td>WizardMath</td><td>Llama-2</td><td>7B</td><td>√</td><td>54.9</td><td>10.7</td><td>57.3</td><td>38.1</td><td>59.1</td><td>73.7</td><td>49.0</td></tr><tr><td>MetaMath</td><td>Llama-2</td><td>7B</td><td>√</td><td>66.6</td><td>20.7</td><td>68.8</td><td>43.8</td><td>72.5</td><td>86.9</td><td>59.9</td></tr><tr><td> KPMath-Plus</td><td>Llama-2</td><td>7B</td><td>√</td><td>75.6</td><td>34.0</td><td>73.1</td><td>51.7</td><td>82.1</td><td>92.9</td><td>68.2 (+35.2)</td></tr><tr><td>Llemma</td><td>=</td><td>7B</td><td>×</td><td>40.4</td><td>18.9</td><td>56.5</td><td>49.0</td><td>68.7</td><td>83.3</td><td>45.8</td></tr><tr><td> KPMath-Plus</td><td>Llemma</td><td>7B</td><td>√</td><td>76.7</td><td>41.9</td><td>77.5</td><td>67.8</td><td>84.1</td><td>93.2</td><td>73.5 (+27.7)</td></tr><tr><td>Mistral</td><td></td><td>7B</td><td>×</td><td>42.9</td><td>12.9</td><td>65.1</td><td>55.6</td><td>68.4</td><td>86.8</td><td>55.3</td></tr><tr><td>MAmmoTH</td><td>Mistral</td><td>7B</td><td>√</td><td>52.7</td><td>14.5</td><td>54.1</td><td>49.1</td><td>64.9</td><td>77.5</td><td>52.1</td></tr><tr><td>MMIQC</td><td>Mistral</td><td>7B</td><td>√</td><td>74.8</td><td>36.0</td><td>73.1</td><td>62.5</td><td>81.9</td><td>90.5</td><td>69.8</td></tr><tr><td>MetaMath</td><td>Mistral</td><td>7B</td><td>√</td><td>77.8</td><td>29.0</td><td>78.6</td><td>64.7</td><td>81.1</td><td>93.4</td><td>70.8</td></tr><tr><td> KPMath-Plus</td><td> Mistral</td><td>7B</td><td>√</td><td>82.1</td><td>46.8</td><td>76.4</td><td>66.4</td><td>86.7</td><td>94.2</td><td>75.4 (+20.1)</td></tr><tr><td>DeepSeekMath</td><td></td><td>7B</td><td>√</td><td>63.3</td><td>32.3</td><td>73.2</td><td>68.6</td><td>82.9</td><td>92.4</td><td>68.8</td></tr><tr><td>KPMath-Plus</td><td>DSMath</td><td>7B</td><td>√</td><td>83.9</td><td>48.8</td><td>81.5</td><td>78.7</td><td>88.9</td><td>94.8</td><td>79.4 (+10.6)</td></tr><tr><td>Llama-3</td><td></td><td>8B</td><td>√</td><td>55.6</td><td>18.0</td><td>69.2</td><td>67.4</td><td>72.6</td><td>91.0</td><td>62.3</td></tr><tr><td> KPMath-Plus</td><td> Llama-3</td><td>8B</td><td>√</td><td>83.2</td><td>46.5</td><td>80.3</td><td>71.2</td><td>87.0</td><td>94.4</td><td>77.2 (+14.9)</td></tr><tr><td>Llama-2</td><td>-</td><td>13B</td><td>×</td><td>24.3</td><td>6.3</td><td>43.1</td><td>39.5</td><td>56.3</td><td>70.4</td><td>36.2</td></tr><tr><td>Llama-2 SFT</td><td></td><td>13B</td><td>√</td><td>51.1</td><td>9.2</td><td>46.3</td><td>35.8</td><td>58.6</td><td>75.0</td><td>42.6</td></tr><tr><td>Platypus-2</td><td>Llama-2</td><td>13B</td><td>×</td><td>23.7</td><td>7.1</td><td>50.7</td><td>45.3</td><td>55.1</td><td>69.6</td><td>38.0</td></tr><tr><td>MAmmoTH</td><td>Llama-2</td><td>13B</td><td>√</td><td>49.6</td><td>9.9</td><td>49.6</td><td>40.5</td><td>60.0</td><td>73.4</td><td>47.2</td></tr><tr><td>WizardMath</td><td>Llama-2</td><td>13B</td><td>√</td><td>63.9</td><td>14.0</td><td>64.3</td><td>46.7</td><td>65.8</td><td>79.7</td><td>51.8</td></tr><tr><td>MetaMath</td><td>Llama-2</td><td>13B</td><td>√</td><td>71.0</td><td>23.2</td><td>71.9</td><td>52.8</td><td>75.7</td><td>87.0</td><td>63.6</td></tr><tr><td>KPMath-Plus</td><td>Llama-2</td><td>13B</td><td>√</td><td>81.6</td><td>41.0</td><td>76.7</td><td>63.9</td><td>83.2</td><td>92.3</td><td>73.1 (+36.9)</td></tr><tr><td>Llemma</td><td></td><td>34B</td><td>×</td><td>55.4</td><td>24.4</td><td>68.0</td><td>57.2</td><td>75.9</td><td>90.5</td><td>61.9</td></tr><tr><td>MMIQC</td><td>Llemma</td><td>34B</td><td>√</td><td>79.2</td><td>38.7</td><td>80.4</td><td>70.1</td><td>85.0</td><td>94.0</td><td>74.6</td></tr><tr><td>KPMath-Plus</td><td>Llemma</td><td>34B</td><td>√</td><td>82.4</td><td>48.6</td><td>81.2</td><td>71.9</td><td>87.5</td><td>94.5</td><td>77.7 (+15.8)</td></tr><tr><td>Llama-2</td><td>1</td><td>70B</td><td>×</td><td>57.8</td><td>14.4</td><td>73.6</td><td>57.5</td><td>76.0</td><td>92.4</td><td>58.2</td></tr><tr><td>Llama-2 SFT</td><td></td><td>70B</td><td>√</td><td>69.3</td><td>14.9</td><td>64.0</td><td>53.0</td><td>71.3</td><td>84.8</td><td>56.6</td></tr><tr><td>Platypus-2</td><td>Llama-2</td><td>70B</td><td>×</td><td>45.9</td><td>15.0</td><td>74.3</td><td>47.3</td><td>72.7</td><td>91.1</td><td>53.0</td></tr><tr><td>WizardMath</td><td>Llama-2</td><td>70B</td><td>√</td><td>81.6</td><td>22.7</td><td>80.0 85.8</td><td>49.8 63.4</td><td>76.2 84.0</td><td>86.2 95.4</td><td>63.8 73.0</td></tr><tr><td>MetaMath MAmmoTH</td><td>Llama-2 Llama-2</td><td>70B 70B</td><td>√ √</td><td>82.0 65.1</td><td>27.2 14.6</td></table></body></html>

Table 1: Results on six mathematical reasoning tasks. The results of our model are bolded. ZS: Zero-shot inference withou demonstrations. Vanilla models are tested with CoT.

# Main Results

Table 1 presents the results on six widely-used mathematical benchmarks, highlighting several key observations:

Our models achieve state-of-the-art results in models with parameters ranging from 7B to 72B. Specifically, the KPMath-Plus-Qwen1.5-72B model achieves accuracies of $8 7 . 0 \%$ on GSM8K, and $5 8 . 3 \%$ on MATH, culminating in a promising average of $8 1 . 5 \%$ across six math reasoning datasets. This performance exceeds that of all competitors and even the best commercial models like GPT-4.

KPMath-Plus significantly boosts mathematical reasoning performance across a range of model architectures and sizes, including the latest SoTA models like Llama 3, DeepSeekMath, and Qwen, with average accuracy gains ranging from $1 0 . 6 \%$ to $3 6 . 9 \%$ . Additionally, it delivers greater performance increases than other datasets with the same base models.

Although only GSM8K and MATH were used as seed data, KPMath-Plus also improves performance on OOD benchmarks. Additionally, Figure 6 displays the Hungarian Exam Score versus GSM8K Performance of various models. Notably, KPMath-Plus-Mistral-7B trails only GPT-4 (OpenAI 2023) and Grok-1 (xAI 2023). Compared to other fine-tuned models, our models demonstrate a well-balanced performance between the two test sets, suggesting they do not overfit the seed data.

![](images/77f3e95fe115a6a654359064c9a9e70ab60516729b64c6c8b849440302a7e413.jpg)  
Figure 6: Hungarian Exam vs GSM8K Performance.

Table 2: Retention ratios of synthesized questions with different configurations.   

<html><body><table><tr><td>Key Points</td><td>SamplingMethod</td><td>Retention Ratio</td></tr><tr><td>w/o</td><td>Random</td><td>27.4%</td></tr><tr><td>w/o</td><td>TCPM-based</td><td>30.1%</td></tr><tr><td>w/</td><td>Random</td><td>39.2%</td></tr><tr><td>w/</td><td>TCPM-based</td><td>50.8%</td></tr></table></body></html>

# Ablation Study

Knowledge Construction To ensure high data quality and conserve API resources, we conducted preliminary experiments with various methods before proceeding with largescale data synthesis. We generated a set of 500 new questions across three scenarios, applying a stringent acceptance threshold of 0.85. The retention rates for questions synthesized using different configurations are summarized in Table 2. This study reveals that key points significantly enhance question quality and coverage of essential knowledge. Conversely, without these annotations, the synthesized questions notably lack in quality and relevant content. Furthermore, TCPMbased sampling improves the correlation among provided key points, resulting in more coherent and diverse questions.

Training Data Components We conducted an ablation study with the KPMath-Plus data components on the Mistral7B model, training over 3 epochs. Results in Table 3 indicate that integrating KPMath-G, derived from the GSM8K dataset, enhances performance on GSM8K by $5 \%$ compared to training solely on MathMix. Improvements extend to SVAMP, ASDiv, and MAWPS, while a slight performance decline in MATH and TabMWP is observed, potentially due to their higher complexity. Moreover, combining KPMath-M, based on the MATH dataset, with MixMath consistently increases scores by over $1 \%$ across all datasets. Merging KPMathG and KPMath-M significantly boosts overall performance, with gains of $6 . 4 \%$ on GSM8K and $3 . 5 \%$ on MATH, averaging a $4 . 1 \%$ improvement, illustrating the comprehensive benefits of our synthesized data within KPMath-Plus for mathematical reasoning.

Table 3: Performance comparison of training with different data components. OOD: average accuracy of SVAMP, TabMWP, ASDiv, and MAWPS.   

<html><body><table><tr><td>Data</td><td>GSM8K</td><td>MATH</td><td>OOD</td></tr><tr><td>MixMath</td><td>75.7</td><td>43.3</td><td>77.2</td></tr><tr><td>MixMath+KPMath-G</td><td>80.7</td><td>43.0</td><td>78.9</td></tr><tr><td>MixMath+KPMath-M</td><td>77.0</td><td>45.9</td><td>78.4</td></tr><tr><td>KPMath-Plus</td><td>82.1 (+6.4)</td><td>46.8 (+3.5)</td><td>80.9 (+3.7)</td></tr></table></body></html>

![](images/cc1c9218c52a47784035c1cf64e6ce421d8751bb21c2794d1c37be7fac2db4eb.jpg)  
Figure 7: Performance of KPMath-Plus-Mistral-7B across various training data size.

Training Data Size We also investigated the impact of training data size on the KPMath-Plus-Mistral-7B model’s performance. As demonstrated in Figure 7, model performance exhibits a logarithmic increase with the expansion of training data. The model achieves impressive results with small data size and maintains a steady growth trend. This study underlines the exceptional quality of our data and establishes a clear linkage between training data size and model performance, particularly in tackling complex tasks. In our future work, we aim to further explore larger and higher-quality datasets to continue improving model performance.

# Conclusion

In this paper, we propose a new data synthesis paradigm that is focused on the generation of large-scale, highquality, symbolically-driven training datasets. Leveraging this paradigm, we have developed an extensive synthetic dataset tailored for mathematical reasoning. By utilizing this data set, our fine-tuned model achieved excellent performance in multiple data sets including MATH and GSM8K, and the performance exceeded all 7B to 72B competitors. Our research underscores the effectiveness of integrating key points in data synthesis and implementing automated verification methods for both questions and answers.