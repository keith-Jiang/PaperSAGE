# Augmenting Math Word Problems via Iterative Question Composing

Haoxiong $\mathbf { L i u } ^ { 1 * \dagger }$ , Yifan Zhang1\*, Yifan Luo1 2, Andrew C Yao1 2

1Institute for Interdisciplinary Information Sciences, Tsinghua University, 2Shanghai Qi Zhi Institute {liuhx20,zhangyif21,luoyf24}@mails.tsinghua.edu.cn, andrewcyao@tsinghua.edu.cn

# Abstract

Despite the advancements in large language models (LLMs) for mathematical reasoning, solving competition-level math problems remains a significant challenge, especially for open-source LLMs without external tools. We introduce the MMIQC dataset, comprising a mixture of processed web data and synthetic question-response pairs, aimed at enhancing the mathematical reasoning capabilities of base language models. Models fine-tuned on MMIQC consistently surpass their counterparts in performance on the MATH benchmark across various model sizes. Notably, Qwen-72B-MMIQC achieves a $4 5 . 0 \%$ accuracy, exceeding the previous open-source state-ofthe-art by $8 . 2 \%$ and outperforming the initial version GPT-4 released in 2023. Extensive evaluation results on Hungarian high school finals suggest that such improvement can generalize to unseen data. Our ablation study on MMIQC reveals that a large part of the improvement can be attributed to our novel augmentation method, Iterative Question Composing (IQC), which involves iteratively composing new questions from seed problems using an LLM and applying rejection sampling through another LLM.

Code — https://github.com/iiis-ai/IterativeQuestionComposing   
Datasets — https://huggingface.co/datasets/Vivacem/MMIQC

# Introduction

Although large language models have been demonstrated to be powerful in various applications (Chen et al. 2021; Brown et al. 2020; Ouyang et al. 2022; Park et al. 2023; Huang et al. 2022b), solving math problems that require complex reasoning skills remains a challenging task. On MATH (Hendrycks et al. 2021b), a competition-level math problem benchmark containing algebra, calculus, geometry, combinatorics and number theory problems, open-source base LLMs such as the LLaMA family (Touvron et al. 2023a,b) fail to answer most of the problems correctly.

Previous work tries to enhance the mathematical reasoning abilities of base models by fine-tuning them on domainspecific data. Specifically, One line of work (Azerbayev

Qwen-72B-MMIQC 45 GPTDeepSeek-67B-MMIQC 40 Llemma-34B-MMIQC Mistral-7B-MMIQC 35 15202530MATH Score (%) Mistral-7B- OpenChat 3.5 MetaMathQA . Llemma-34B GPT-3.5 Turbo Grok-1 MetaMath-7B Llemma-7B Grok-0 Mistral-7B 10 Code34LlBama- MAmQmwoeTn-H7-B7B 5 CodeLlama-7B 10 20 30 40 50 60 70 Hungarian Exam Score $( \% )$

et al. 2023; Lewkowycz et al. 2022) collects math corpora from the web and fine-tunes the models on them, which is also known as the procedure of continual pre-training (Cossu et al. 2022). Another line of work focuses on constructing synthetic data through rejection sampling (Yuan et al. 2023), distilling from GPT-4/GPT-3.5 (Yue et al. 2023) or question bootstrapping (Yu et al. 2023), and then use the generated question-response pairs to perform supervised finetuning in the way described in (Taori et al. 2023; Ouyang et al. 2022). However, there still exists a large performance gap between these fine-tuned models and the most advanced close-source models such as GPT-4 (OpenAI 2023) and Gemini-Ultra (Team et al. 2023). Given that simply adding more data does not always lead to better performance as shown in (Yu et al. 2023), how to bridge the gap remains an open challenge.

This work tackles the challenge by combining the two lines of work. On one hand, we reuse the high-quality corpora used in the pre-training stage during fine-tuning. Specifically, MMIQC contains around 1200k questionresponse pairs we filtered and pre-processed from the web pages at math.stackexchange.com, which are included in the RedPajama dataset (Computer 2023). On the other hand, for the synthetic data part of MMIQC, we increase the diversity by using multiple kinds of augmentation methods listed below: 1) Prompting GPT-4 with an integrated version of the question bootstrapping prompts used in (Yu et al. 2023), and do rejection sampling with GPT-3.5-Turbo on both seed and augmented problems. 2) Using a modified prompt presented in (Liu et al. 2023) to ask GPT-4 to generate similar problems with answers given seed problems of the training set of MATH. Although the generated answers can be wrong, we perform rejection sampling on these problems as well. 3) Performing IQC (Iterative Question Composing) with 4 iterations in total. We iteratively ask GPT-4 to compose new questions from the given seed problems and do rejection sampling to filter those problems with answers aligned with GPT-3.5-turbo’s answers. 4) Filtering a $2 0 4 \mathrm { k }$ subset of MetaMathQA (Yu et al. 2023) and adding it to the MMIQC dataset (More details on MMIQC will be introduced in Section ).

![](images/a524af5608eb0ce08a8b8d6b1f20cfb341de0534fde8a938af9d544ea130d020.jpg)  
Figure 2: The performance of base models and their fine-tuned versions on MATH benchmark. The models remarked with an ∗ are trained and evaluated by us. We can see that the models fine-tuned on MMIQC consistently outperform their counterparts by a clear margin.

We fine-tune several base models on MMIQC, resulting in models consistently achieving a large margin compared to their counterparts when evaluated on MATH, as shown in Figure 2. Specifically, the models Mistral-7BMMIQC, Llemma-34B-MMIQC, DeepSeek-67B-MMIQC and Qwen-72B-MMIQC, which are obtained by fine-tuning Mistral-7B (Jiang et al. 2023), Llemma-34B (Azerbayev et al. 2023) and DeepSeek-67B (Bi et al. 2024) on MMIQC, achieve $3 6 . 0 \%$ , $3 8 . 6 \%$ , $4 1 . 0 \%$ and $45 . 0 \%$ accuracy on MATH, $5 . 8 \%$ , $3 . 8 \%$ , $4 . 2 \%$ and $3 . 3 \%$ higher than the counterpart models that are fine-tuned on MetaMathQA, respectively.

We also evaluate the models on the 2023 Hungarian national high school finals in mathematics (Paster 2023b). The results in Figure 1 suggest that the mathematical reasoning abilities the models acquire through being fine-tuned on

MMIQC can generalize to unseen held-out problems. We highlight our contributions as follows:

• We propose IQC (Iterative Question Composing), a data augmentation method that can iteratively generate diverse data starting from a seed dataset of math word problems.   
• We release MMIQC, a mixture of processed web data and synthetic question-response pairs. In different model sizes, the models fine-tuned on MMIQC consistently outperform their counterparts by a clear margin on the MATH test set. Notably, Qwen-72B-MMIQC achieves a $4 5 . 0 \%$ accuracy, exceeding the previous open-source state-of-the-art1 by $8 . 2 \%$ and outperforming the initial version GPT-4 released in 2023. Such improvement can generalize to unseen held-out data, e.g., Hungarian high school finals.   
• Our results show that reusing the high-quality data in the pre-training corpora during the fine-tuning stage can improve the model performance, successfully combining the two lines of work of continual pre-training and supervised fine-tuning.   
• Our results also show that using multiple augmentation methods to construct datasets for fine-tuning is an efficient way to boost the performance of LLMs.

# Related Work

Base Large Language Models. Base large language models (LLMs) trained on massive corpora (e.g. 1.4T tokens of text for Llama (Touvron et al. 2023a)) from various sources with a simple auto-regressive next token prediction loss have achieved great success in various natural language processing tasks (Radford et al. 2019; Brown et al. 2020; Touvron et al. 2023a,b; Jiang et al. 2023). Although these pre-trained models are not intended to serve for solving complex mathematical problems, (Wei et al. 2023) show that few-shot prompting can help the models answer a certain fraction of problems correctly. Nevertheless, to achieve better performance, fine-tuning the base LLMs on domain-specific data is required.

Fine-tuning Base LLMs on Mathematical Datasets. Current practice of fine-tuning base LLMs on mathematical datasets can be classified into two kinds: 1) continual pretraining (Lewkowycz et al. 2022; Azerbayev et al. 2023). This line of work typically collects billion-tokens level mathematical text data from the web, such as mathematical sub-sites of Stack Exchange and ArXiv, and finetune the model in the same way as that in the pre-training stage. 2) SFT (Supervised Fine-Tuning) (Yuan et al. 2023; Yu et al. 2023; Yue et al. 2023; Gou et al. 2023). Works in this line collect question-response pairs via various methods and train the models on their dataset in an Alpaca style. Due to the scarcity of publicly available high-quality questionresponse pairs datasets and the costly nature of manually composing math word problems, how to augment new data from the existing datasets becomes the focus of these works.

Our work is located in the middle between these two: MMIQC is a mixture of filtered pre-training corpus and question-response pairs generated using various augmentation methods.

Reasoning Frameworks for Solving Mathematical Problems. Much effort has been devoted to achieving a higher accuracy on math word problem benchmarks by designing different procedures of using the given LLMs to obtain the answers, which we refer to as reasoning frameworks. Among them, Prompting-based methods (Radford et al. 2019; Wei et al. 2023; Fu et al. 2022) play a significant role in activating the potential reasoning abilities for base LLMs through carefully designing the prompts shown to the models. Self-consistency (Wang et al. 2023b) samples multiple rationale paths for a model and then decides the answer by majority voting. In contrast of self-consistency, (Cobbe et al. 2021; Uesato et al. 2022; Lightman et al. 2023) use Outcome Reward Models (ORM) and Process Reward Models (PRM) trained on human annotations as verifiers to help select the answer with the highest score from the sampled reasoning paths of LLMs. Getting rid of the need of manual annotation, (Wang et al. 2023a) score a given reasoning step by estimating the potential of that step to lead to a correct answer automatically.

Some frameworks also include the use of plug-in tools and external APIs. Program-aided prompting (Gao et al. 2022; Yue et al. 2023) provides in-context samples containing Python codes for LLMs and uses code interpreters to execute the output to facilitate reasoning. Further, (Gou et al. 2023) interleave natural language rationales with Sympy2 code and fine-tune the model on trajectories sampled from GPT-4 to follow their framework in two steps, namely imi

Require: Question composing model $\pi _ { q }$ , rejection sampling model $\pi _ { r }$ , answer extractor defining $\simeq$ , text templater $x ( \cdot , \cdot )$ with inverse $x ^ { - 1 } ( \cdot )$ , initial seed dataset $\begin{array} { r c l } { S _ { 0 } } & { = } & { \{ ( q _ { i } , a _ { i } ) \} _ { i = 1 } ^ { n } } \end{array}$ , total iterations $K$ , question composing prompts $p _ { 1 } , p _ { 2 } , \dotsc , p _ { K }$ , rejection sampling prompt $p _ { r }$ , maximum rejection samples per problem $m$   
1: for $k = 1$ to $K$ do   
2: Initialize $S _ { k } \gets \{ \} , R _ { k } \gets \{ \}$   
3: for all $( q , a ) \in S _ { k - 1 }$ do   
4: Sample $x ^ { \prime } \sim \pi _ { q } \left( \cdot | p _ { k } \oplus x ( q , a ) \right)$   
5: Decompose $( q ^ { \prime } , a ^ { \prime } ) \gets x ^ { - 1 } ( x ^ { \prime } )$   
6: Append $S _ { k } \gets S _ { k } \cup \{ ( q ^ { \prime } , a ^ { \prime } ) \}$   
7: for $j = 1$ to $m$ do   
8: Sample $a ^ { ( j ) } \sim \pi _ { r } ( \cdot | p _ { r } \oplus q ^ { \prime } )$   
9: if $a ^ { ( j ) } \simeq a ^ { \prime }$ then   
10: Append $R _ { k } \gets R _ { k } \cup \{ ( q ^ { \prime } , a ^ { ( j ) } ) \}$   
11: end if   
12: end for   
13: end for   
14: Combine $D _ { k }  S _ { k } \cup R _ { k }$   
15: end for   
16: Output Collections $D _ { 1 } , D _ { 2 } , \ldots , D _ { K }$

tation learning and output space shaping.

We note that our results in Figure 2 do not include multiple times of sampling, use of verifiers or code interpreters, thus cannot be directly compared with the results reported in these works.

# Iterative Question Composing

Traditional data augmentation methods primarily concentrate on modifying either the questions or answers while retaining their original meanings, or generating similar problems, as discussed in (Yu et al. 2023) and (Liu et al. 2023). These methods, however, are limited in their diversity as they aim to create nearly identical problems. Our approach, termed IQC (Iterative Question Composing), deviates from this by iteratively constructing more complex problems. It augments the initial problems, adding additional reasoning steps without altering their intrinsic logical structure. This ensures that the newly formed problems are organically linked to the original problem and elaborately tries to not include extraneous elements induced by a large transition of the reasoning process.

Notations. In our description, we refer to the combination of an LLM, its tokenizer, encoding/decoding methods, and a fixed generation configuration (inclusive of generation strategy, sampling temperature, and stopping criteria) simply as ‘an LLM’. For an LLM $\pi$ , we denote the output distribution given prompt $p \in { \mathcal { A } } ^ { * }$ as $\pi ( \cdot | p )$ . The concatenation of two text paragraphs $p _ { 1 }$ and $p _ { 2 }$ is represented as $p _ { 1 } \oplus p _ { 2 }$ .

The IQC process begins with specifying an LLM $\pi _ { q }$ for question composing and another model $\pi _ { r }$ for rejection sampling. An answer extractor is needed to derive answers from responses. Two responses $r _ { 1 }$ and $r _ { 2 }$ are considered equivalent, denoted $r _ { 1 } ~ \simeq ~ r _ { 2 }$ , if the same answer can be extracted from both. The process initiates with a seed dataset $S _ { 0 } = \{ ( q _ { i } , a _ { i } ) \} _ { i = 1 } ^ { n }$ .

In iteration $\# 1$ , we prompt $\pi _ { q }$ with $p _ { 1 } \oplus x ( q , a )$ for each $( q , a ) \in S _ { 0 }$ , where $x ( \cdot , \cdot )$ is a text template transforming a question-response pair into text, and $p _ { 1 }$ solicits a new question-answer composition. This yields a new dataset

$$
S _ { 1 } = \{ ( q _ { i } ^ { \prime } , a _ { i } ^ { \prime } ) \} _ { i = 1 } ^ { n } ,
$$

where $( q _ { i } ^ { \prime } , a _ { i } ^ { \prime } ) = x ^ { - 1 } ( x _ { i } ^ { \prime } )$ and $x _ { i } ^ { \prime } \sim \pi _ { q } \left( \cdot | p _ { 1 } \oplus x _ { i } \right)$ is the output for the $i$ th sample. We further enhance $S _ { 1 }$ by rejection sampling from $\pi _ { r }$ , resulting in

$$
R _ { 1 } : = \{ ( q _ { i } ^ { \prime } , a _ { i } ^ { ( j ) } ) | a _ { i } ^ { ( j ) } \simeq a _ { i } ^ { \prime } , i \in [ n ] , j \in [ m ] \} ,
$$

where ai(j) are the sampled responses from πr(·|pr ⊕ qi′). The dataset $D _ { 1 }$ is then formed by uniting $S _ { 1 }$ and $R _ { 1 }$ :

$$
D _ { 1 } : = S _ { 1 } \cup R _ { 1 } .
$$

For each subsequent iteration $\# k$ , the aforementioned procedure is repeated using $S _ { k - 1 }$ as the seed dataset, with varying question composing prompts $p _ { k }$ . The complete IQC process is delineated in Algorithm 1.

# Seed Question:

Evaluate

$$
( 5 a ^ { 2 } - 1 3 a + 4 ) ( 2 a - 3 )
$$

for $a = 1 { \frac { 1 } { 2 } }$

Iter # 1 Question:   
If $b = 2 a - 3$ and $a = 1 \textstyle { \frac { 1 } { 2 } }$ , what is the value of   
$( 5 a ^ { 2 } - 1 3 a + 4 ) b ?$

# Iter $\# 2$ Question:

Given $b = 2 a - 3$ , $a = 1 \textstyle { \frac { 1 } { 2 } }$ , and $c = 3 b + 5$ , find the value of $c ( 5 a ^ { 2 } - 1 3 a + 4 )$ .

# Iter $\# 3$ Question:

Given $b = 2 a - 3$ , $a = 1 \textstyle { \frac { 1 } { 2 } }$ , $c = 3 b + 5$ , and $d =$ $c ^ { 2 } - 4 c ,$ , find the value of $\overset { \^ } { d } + c ( 5 a ^ { 2 } - 1 3 a + 4 )$ .

# Iter # 4 Question:

Given $b = 2 a - 3$ , $a = 1 \frac 1 2$ , $c = 3 b + 5$ , $d = c ^ { 2 } - 4 c$ , and $e = d ^ { 3 } + 2 c d - 7$ , find the value of $e + c ( 5 a ^ { 2 } -$ $1 3 a + 4 ) + d$ .

# The MMIQC Dataset

In this section, we introduce how each part of MMIQC is constructed in detail.

Subset of MetaMathQA. The original MetaMathQA dataset is constructed by sampling GPT-3.5 for $k ~ = ~ 2 0$ times under a $T ~ = ~ 0 . 7$ temperature for each problem in the training set of MATH (Hendrycks et al. 2021a) and

You will be provided with 1 math problem and its solution and answer (which are not guaranteed to be right). Please generate 1 new problem that (implicitly) contains the original problem as a subproblem or substep.

Your response should only contain one line text with 3 fields ”problem”, ”solution” and ”answer” in the same format as the given problem. The solution to the generated problem should be as brief as possible and \*\*should not quote the conclusion of the original problem\*\*. Ensure there is only one latex box in the solution and the answer is completely the same with the content in the box.

\*\*Please use two backslashes to represent one in the strings in order that it can be properly read in python.\*\* For example, you should write “ cdot” as “ cdot”.

GSM8K (Cobbe et al. 2021) dataset, or its bootstrapped versions. We restrict the number of samples for each completely same question to be 3 and 1 for MATH and GSM8K, respectively, to obtain a subset of MetaMathQA. This subset contains 112.2K GSM8K question-response pairs and 91.5K MATH pairs.

Answer Augmentation and Question Bootstrapping. We integrate the question bootstrapping methods used in (Yu et al. 2023) into a single prompt shown in Figure 5. Our motivation is that given GPT-4 is highly capable of natural language understanding, a few-shot prompting style used in (Yu et al. 2023) might suppress the diversity of the augmented questions. The seed dataset is constructed by the samples in the training set of MATH that do not contain Asymptote language in their question statements. We perform rejection sampling from GPT-3.5 on both the seed dataset and generated questions using the prompt shown in Figure 6, obtaining 66.5K question-response pairs. We use a temperature $T = 1 . 0$ for both question bootstrapping and rejection sampling.

Augmented Similar Problems. With the same seed dataset, we ask GPT-4 to generate 3 problems (with a solution, for rejection sampling) for 1 seed problem each time, using the prompt in Figure 7. This is different from the practice in (Liu et al. 2023), where they ask GPT-3.5 to generate 10 similar questions given 1 seed problem since we find that GPT tends to generate several almost the same problems regardless of the given seed problem when asked to generate up to 10 new problems. We use the stronger GPT4 instead of GPT-3.5 considering rejection sampling needs the answer to the problem better to be correct. To control the cost, our prompt emphasizes that the solution should be as brief as possible. The total number of the augmented similar problems and the question-response pairs rejection sampled from them is $3 8 . 2 \mathrm { K }$ . The rejection sampling prompt is the same one in Figure 6 as well. We use a temperature $T = 1 . 0$ for both procedures.

Table 1: The composition of MMIQC.   

<html><body><table><tr><td>DATA</td><td># SAMPLES</td><td>#REPETITIONS</td><td>RATIO</td></tr><tr><td>METAMATHQA</td><td>203.7K</td><td>3</td><td>26.6%</td></tr><tr><td>ANSAUG&QB</td><td>66.5K</td><td>3</td><td>8.7%</td></tr><tr><td>AUGSIMILAR</td><td>38.2K</td><td>3</td><td>5.0%</td></tr><tr><td>1QC</td><td>55.1K</td><td>3</td><td>7.2%</td></tr><tr><td>MATHSTEX</td><td>1203.6K</td><td>1</td><td>52.5%</td></tr></table></body></html>

Iterative Question Composing. We perform Iterative Question Composing for 4 iterations as described in Section . Specifically, we use GPT-4 for question composing model $\pi _ { q }$ with a $T = 0 . 7$ temperature and GPT-3.5 for rejection sampling model $\pi _ { r }$ with a $T = 1 . 0$ temperature. The question composing prompts and rejection sampling prompt are shown in Figure 4 and Figure 6, respectively. The text templater $x ( \cdot , \cdot )$ we use is a program that transforms each question-response pair into JSON text format, with fields ‘problem’ and ‘solution’. The seed dataset is also the samples in the training set of MATH that do not contain Asymptote code in their question statements. The resulting dataset has 55.1K samples in total.3 We provide an example of the generated questions in different iterations corresponding to the same seed problem in Figure 3. We note that although some of the questions are not rigorously a sub-problem or sub-step of the corresponding problem in the previous iteration as required in our prompt, they are still valid questions that can increase the diversity of the dataset. We have checked the correctness of 100 randomly selected QA pairs generated by IQC and find that $8 5 \%$ of them are correct.

Mathematics Stack Exchange. We observe that in the OpenWebMath (Paster et al. 2023) dataset, the data from Mathematics Stack Exchange shows high quality and is most related to competition-level math. Motivated by this, we extract the data collected from Mathematics Stack Exchange in RedPajama (Computer 2023) and pre-process it into question-response pairs. For each Mathematics Stack Exchange page, we only retain the answer ranked first by RedPajama. Then we filter out the answer that does not contain a formula environment symbol $\cdot \$ 3$ . This results in a dataset with 1203.6K question-response pairs.

Table 1 shows the make-up of MMIQC. When fine-tuning the models MMIQC contains 3 repetitions of the subsets mentioned above, except for the Mathematics Stack Exchange part. We shuffle the order of samples after combining the subsets.

Table 2: Ablation study on the optimal learning rate. We fine-tune Mistral-7B on MMIQC with different maximal learning rate values and evaluate the fine-tuned models on MATH to decide the best candidate.   

<html><body><table><tr><td>LR</td><td>1E-6</td><td>5E-6</td><td>1E-5</td><td>2E-5</td><td>5E-5</td><td>1E-4</td></tr><tr><td>MATH(%)</td><td>32.3</td><td>35.1</td><td>36.0</td><td>35.4</td><td>31.5</td><td>27.1</td></tr></table></body></html>

# Experiments

# Fine-tuning Setup

Our fine-tuning strategy mainly follows the practice of (Taori et al. 2023), except that we use a different prompt template to transform the question-response pairs. For a sample from Mathematics Stack Exchange, the corresponding prompt fed into the model during training is a simple concatenation of the question and response with two newline symbols. For a sample from other subsets, we additionally add a prefix ‘Please solve the following problem and put your answer at the end with “The answer is: ”.’ to the question-response concatenation.

We use the HuggingFace transformers library (Wolf et al. 2019) for our fine-tuning experiments.

We fine-tune all models on MMIQC for 1 epoch, using a $3 \%$ warm-up ratio linear learning rate schedule. For the choice of maximum learning rate, we do a simple hyperparameter selection experiment shown in Table 2 and determine it to be 1e-5. We use the BFloat16 numerical format during training. Employing the DeepSpeed Zero-3 Stage (Rajbhandari et al. 2020), we fine-tune 7B models on one node of 8xA800 GPUs with micro batch-size at 8, and gradient accumulation at 4, 34B models on 2 nodes with micro batch-size at 4 and gradient accumulation at 4 and ${ \sim } 7 0 \mathrm { B }$ models on 4 nodes with micro batch-size at 4 and gradient accumulation at 2, maintaining an effective batch size of 256. It takes around 14 hours, 61 hours and 90 hours to fine-tune 7B, 34B and ${ \sim } 7 0 \mathbf { B }$ models under the setups stated above, respectively.

# Model Evaluation

For a fair comparison, we first evaluate the fine-tuned models on MATH (Hendrycks et al. 2021a), a competition-level math word problems benchmark with 5000 test problems in a zero-shot setting. We prompt all our fine-tuned models with the test question with the prefix ‘Please solve the following problem and put your answer at the end with “The answer is: ”.’, and extract the answer from the output using a modified version of the answer extractor provided in (Lewkowycz et al. 2022). We use a series of rules to infer whether the extracted answer is the same as the ground-truth answer, including a comparison using SymPy (Meurer et al. 2017). The complete results of our evaluation on MATH and a comparison with existing models are shown in Table 3.

For the evaluation on 2023 Hungarian national high school finals in mathematics, we use the few-shot prompt used in (Paster 2023b). We manually assess the grades for

You will be provided with 1 math problem in newline-delimited json format. Please augment 5 diverse problems from the given problem.

The way you augment a problem can be:   
- Rephrase the problem.   
- Change the scenario without modifying specific quantities.   
- Set 1 number in the problem to an unknown variable, put the answer in the problem and ask what is the value of the variable. Ensure the generated problem is reasonable. Otherwise, skip this method.   
- Other approaches that can ensure the correctness of the answer you provide to the augmented problem. Your response should only contain text in newline-delimited json format, keeping the same with the given problem.   
Please use two backslashes to represent one in the strings.

Figure 5: The prompt we use to perform question bootstrapping for asking GPT-4.

You will be presented a mathematical problem. You should solve the problem step-by-step carefully. Present the final answer in latex boxed format, e.g., $\boxed { 6 3 \pi } .$

You will be provided with 1 math problem in newline-delimited json format. Please generate 3 diverse new problems similar to the given problem.

Your response should only contain text in newlinedelimited json format, keeping the same with the given problem. The solutions to the generated problems should be as brief as possible. Ensure there is only one box in the solution and the answer is completely the same with the content in the box. Please use two backslashes to represent one in the strings.

every model according to the examiner instructions. The results shown in Figure 1 are the grades under a full mark of 117.

# Ablation Study on Subsets of MMIQC

In order to understand the ratio of contribution to the improvement revealed in Table 3 of different subsets of MMIQC, we fine-tune Mistral-7B with a series of training sets constructed by gradually adding the subsets. When MathStackExchange is not added, we fine-tune for 3 epochs. When MathStackExchange is added to the training dataset, we mix 3 repetitions of other data with 1 repetition of the MathStackExchange, and fine-tune for only 1 epoch. It can

be seen from Table 4 that

• Although our filtered subset of MetaMathQA is only half the size of the original dataset (which has 395K samples, more than the total number of samples of our synthetic data), the performance drop is only $1 . 8 \%$ . This shows that the $k = 2 0$ strategy in (Yu et al. 2023) results in some redundancy. • Our Answer Augmentation & Question Boosting data help the fine-tuned model beat Mistral-7BMetaMathQA, verifying our hypothesis that directly asking GPT to perform question bootstrapping is more efficient than providing few-shot examples to them. • Our IQC method leads to a significant $3 . 1 \%$ improvement from a high accuracy of $3 1 . 5 \%$ with only 55.1K samples, showing its efficiency. Moreover, the later iterations of IQC also account for a certain ratio of improvement, proving that IQC is a method that can continuously generate new data that can help increase the diversity when added to the data generated in previous iterations.

# Contamination Test

We check the $n$ -gram matches for MMIQC to ensure that the improvement is not a result of direct memorization. We use the script provided by (Azerbayev et al. 2023) to check the $n$ -gram matches between the synthetic part of the MMIQC and MATH test set. It turns out that for a 30-gram match check, there are 44 hits of match between the ‘solution’ field of MATH test set and the ‘output’ field of MMIQC, far fewer than the 168 hits between that of MATH test set and MATH training set. Moreover, we manually check these 44 hits and find that 43 among them belong to the case where intermediate steps of the solutions to similar but different questions collide, with the only exception being the question ‘A regular polygon has interior angles of 144 degrees. How many sides does the polygon have?’. This almost rules out the possibility that fine-tuned models get memorization of solutions to the problems in the test set, indicating a very low risk of data contamination for MMIQC.

Table 3: A comparative analysis of the accuracies achieved by various models on the MATH benchmark. The models marked with an asterisk $( * )$ are fine-tuned and evaluated by us. Other results, unless otherwise cited, are derived from (Wang et al. 2023a). This comparison highlights the significant improvements our fine-tuned models demonstrate over existing solutions in mathematical problem-solving accuracy.   

<html><body><table><tr><td>MODEL</td><td>FT-DATASET</td><td>TOOL USAGE?</td><td>EVAL METHOD</td><td>MATH(%)</td></tr><tr><td>PROPRIETARYMODELS</td><td></td><td></td><td></td><td></td></tr><tr><td>MINERVA-540B(UESATO ET AL.2022)</td><td>ARXIV+WEB</td><td>No</td><td>MAJ1@64</td><td>50.3</td></tr><tr><td>GPT-4 (2023-0314) (BUBECK ET AL.2023)</td><td></td><td>No</td><td>PASS@1</td><td>42.5</td></tr><tr><td>GEMINI-ULTRA(TEAMET AL.2023)</td><td></td><td>No</td><td>PASS@1</td><td>53.2</td></tr><tr><td>~7B MODELS</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LLAMA-2-7B(TTUAL:2ETAL. 2023B)</td><td></td><td>N</td><td>PASS@1</td><td>1.5</td></tr><tr><td>LLEMMA-7B(AZERBAYEV ET AL.2023)</td><td>PROOF-PILE-2</td><td>No</td><td>PASS@1</td><td>18.0</td></tr><tr><td>METAMATH-7B(YU ET AL.2023)</td><td>METAMATHQA</td><td>No</td><td>PASS@1</td><td>19.8</td></tr><tr><td>MISTRAL-7B-METAMATHQA(YUET AL.2023)</td><td>METAMATHQA</td><td>No</td><td>PASS@1</td><td>28.2</td></tr><tr><td>MISTRAL-7B-MMIQC*</td><td>MMIQC</td><td>No</td><td>PASS@1</td><td>36.0</td></tr><tr><td>MAMMOTH-CODER-7B(YUE ET AL.2023)</td><td>MATHINSTRUCT</td><td>CODE</td><td>PASS@1</td><td>35.2</td></tr><tr><td>TORA-CODE-7B(GOU ET AL.2023)</td><td>ToRA-CORPUS</td><td>CODE</td><td>PASS@1</td><td>44.6</td></tr><tr><td>~34B MODELS</td><td></td><td></td><td></td><td></td></tr><tr><td>CODELLAMMA-34B</td><td></td><td>CODE</td><td></td><td></td></tr><tr><td>LLEMMA-34B-METAMATHQA</td><td>METAMATHQA</td><td>No</td><td>PASs@1 PASS@1</td><td>25.0</td></tr><tr><td>LLEMMA-34B-MMIQC*</td><td>MMIQC</td><td>No</td><td>PASS@1</td><td>34.8 38.6</td></tr><tr><td>LLEMMA-34B-METAMATHQA</td><td>METAMATHQA</td><td>MATH-SHEPHERD</td><td>MAJ+VERIFY1@256</td><td>47.3</td></tr><tr><td>ToRA-CODE-34B (GOU ET AL.2023)</td><td>ToRA-CORPUS</td><td>CODE</td><td>PASS@1</td><td></td></tr><tr><td>~70B MODELS</td><td></td><td></td><td></td><td>50.8</td></tr><tr><td>LLAMA-2-70B (TOUVRON ET AL. 2023B)</td><td></td><td></td><td></td><td></td></tr><tr><td>DEEPSEEK-67B (BIET AL.2024)</td><td></td><td>No</td><td>PASS@1</td><td>13.5</td></tr><tr><td>DEEPSEEK-67B-METAMATHQA</td><td>METAMATHQA</td><td>No No</td><td>PASs@1</td><td>18.7</td></tr><tr><td>DEEPSEEK-67B-MMIQC*</td><td>MMIQC</td><td>No</td><td>PASS@1 PASS@1</td><td>36.8</td></tr><tr><td>DEEPSEEK-67B-METAMATHQA</td><td>METAMATHQA</td><td>No</td><td>MAJ1@256</td><td>41.0</td></tr><tr><td>DEEPSEEK-67B-METAMATHQA</td><td>METAMATHQA</td><td></td><td></td><td>45.4</td></tr><tr><td>QWEN-72B (BAIET AL.2023)</td><td></td><td>MATH-SHEPHERD</td><td>MAJ+VERIFY1@256</td><td>48.1</td></tr><tr><td>QWEN-72B-METAMATHQA*</td><td></td><td>No</td><td>PASS@1 PASS@1</td><td>35.2</td></tr><tr><td>QWEN-72B-MMIQC*</td><td>METAMATHQA MMIQC</td><td>No NO</td><td>PASS@1</td><td>41.7 45.0</td></tr></table></body></html>

Table 4: How different subsets of MMIQC affect the accuracy of the finetuned model on MATH.   

<html><body><table><tr><td>DATA</td><td># SAMPLES</td><td>MATH(%)</td></tr><tr><td>METAMATHQA</td><td>395K</td><td>28.2</td></tr><tr><td>METAMATHQA(SUBSET)</td><td>203.7K</td><td>26.4 (-1.8)</td></tr><tr><td>+ ANSAUG&QB</td><td>+66.5K</td><td>30.1 (+1.9)</td></tr><tr><td>+AUGSIMILAR</td><td>+38.2K</td><td>31.5 (+3.3)</td></tr><tr><td>+IQC ITER #1</td><td>+21.8K</td><td>33.0(+4.8)</td></tr><tr><td>+IQC ITER #2</td><td>+16.0K</td><td>33.7 (+5.5)</td></tr><tr><td>+IQCITER #3&#4</td><td>+17.3K</td><td>34.4 (+6.2)</td></tr><tr><td>+MATHSTACKEXCHANGE</td><td>+1203.6K</td><td>36.0 (+7.8)</td></tr></table></body></html>

# Conclusion

In this work, we introduce a novel data augmentation method for math word problem datasets called IQC (Iterative Question Composing) and use it in the construction of our MMIQC dataset. Our evaluation results show that the models fine-tuned on MMIQC achieve new SOTAs on the MATH benchmark. The improvements of our models benefit from the diverse data sources of MMIQC and the effectiveness of IQC.

For future directions, we are interested in how to equip open-source models with the ability to compose questions, in order to perform IQC in a self-evolution style, similar to that in (Huang et al. 2022a). Besides, how to integrate the verification systems (Wang et al. 2023a; Liu et al. 2023) that are originally used to improve the accuracy during inference time into the procedure of IQC, is also an attractive topic.