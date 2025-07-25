# VERSE: Verification-based Self-Play for Code Instructions

Hao Jiang1, Qi Liu1, 2\*, Rui Li1, Yuze Zhao1, Yixiao $\mathbf { M } \mathbf { a } ^ { 1 }$ , Shengyu Ye1, Junyu Lu1, 2, Yu Su2, 3

1State Key Laboratory of Cognitive Intelligence, University of Science and Technology of China 2Institute of Artificial Intelligence, Hefei Comprehensive National Science Center 3School of Computer Science and Artificial Intelligence, Hefei Normal University {jianghao0728, ruili2000, yuzezhao, mayx, ysy007, lujunyu} $@$ mail.ustc.edu.cn, {qiliuql} $@$ ustc.edu.cn, yusu $@$ hfnu.edu.cn

# Abstract

Instruction-tuned Code Large Language Models (Code LLMs) have excelled in diverse code-related tasks, such as program synthesis, automated program repair, and code explanation. To collect training datasets for instruction-tuning, a popular method involves having models autonomously generate instructions and corresponding responses. However, the direct generation of responses does not ensure functional correctness, a crucial requirement for generating responses to code instructions. To overcome this, we present Verification-Based Self-Play (VERSE), aiming to enhance model proficiency in generating correct responses. VERSE establishes a robust verification framework that covers various code instructions. Employing VERSE, Code LLMs engage in self-play to generate instructions and corresponding verifications. They evaluate execution results and self-consistency as verification outcomes, using them as scores to rank generated data for self-training. Experiments show that VERSE improves multiple base Code LLMs (average $7 . 6 \%$ ) across various languages and tasks on many benchmarks, affirming its effectiveness.

# Code — https://github.com/TechxGenus/VERSE

# 1 Introduction

In the realm of Large Language Models (LLMs), code serves as a pivotal link between human understanding and machine execution. It acts as a fundamental element by transforming human instructions into executable actions, thereby constructing agents (Hong et al. 2023; Li et al. $2 0 2 4 \mathrm { a }$ ; Liu et al. 2024). To augment the code generation capabilities of LLMs, Code LLMs, trained on a vast corpus of code, have emerged as a focal point (Li et al. 2022; Johnson, Tarlow, and Walder 2023; Khakhar, Mell, and Bastani 2023; Li et al. 2024b). Recently, researchers are primarily focusing on instructiontuning (Chung et al. 2022; Ouyang et al. 2022; Longpre et al. 2023) for Code LLMs (Wang et al. 2023b; Muennighoff et al. 2023; Luo et al. 2023). This technique utilizes code-related instructions and their corresponding responses in supervised training, enhancing the model’s understanding and generalization abilities across various code downstream tasks, such as program synthesis (Young, Bastani, and Naik 2019), and code explanation (Mahbub, Shuvo, and Rahman 2022).

Instruction: Repair the Instruction: Please tell me following $C { + } { + }$ code. what will be the output of the void sign(string n){ JavaScript code? std::cout << n Correct str $\mathbf { \Sigma } = \mathbf { \Sigma }$ "Stack".split(''). 1 reverse().join(''); Incorrect Response: The fixed version: void sign( Code Response: The code will const std::string& n){ LLMs output the reverse of the word std::cout << n;} "Stack": tackS.

To collect training data for instruction-tuning, considering factors such as the cost and copyright issues of using data written by humans, a prevalent approach is to employ models to autonomously generate instructions and responses (Wang et al. 2023a; Taori et al. 2023). However, code instructions demanding strict adherence to the functional correctness of responses—a facet often overlooked in previous research. As illustrated in Figure 1, unverified and unexecuted data generated by LLMs lacks guaranteed quality and may contain errors. A natural idea to solve it is to have Code LLMs verify the functional correctness of self-generated responses with the executable nature of code. There have been some attempts at this in Python code generation tasks (Haluptzok, Bowers, and Kalai 2023; Rozi\`ere et al. 2023), but it has not been explored for general code instructions. In this paper, we focus on how Code LLMs can self-improve through verification for code instructions.

However, direct verification of responses to code instructions presents challenges. On the one hand, responses to code instructions have different modals. For example, generating code summaries or predicting outputs for programs differs from program synthesis and repair, as they may not produce executable code, but generate natural language text that cannot be executed directly instead. This introduces new difficulties to self-verification. On the other hand, both the responses and verification schemes are not guaranteed to be entirely correct. Some responses that are deemed correct under one set of verification schemes may be considered incorrect under another set of schemes. Reaching a consensus on the optimal response is not easy.

To address these challenges, in this paper, we present VERSE, an innovative approach that leverages verificationbased self-play with execution to improve responses to code instructions for Code LLMs. We initiate the process by creating instructions and marking relevant attributes. Code LLMs then transform these instructions into verifications through self-play. Subsequently, Code LLMs generate and combine multiple responses and scripts for execution. The construction of verifications and the combination methods of execution are specifically designed for different modalities of responses, addressing the first challenge. Each response’s quality is evaluated based on verification scores derived from execution results and consistency. The responses are then ranked to reach the optimal consensus and added to the candidate data pool. Ultimately, Code LLMs sample training instances from the candidate data pool based on their verification scores to construct datasets for their self-improvement.

To assess the efficacy of VERSE, we conduct experiments across various tasks, encompassing clone detection (Svajlenko et al. 2014), defect detection (Zhou et al. 2019), program synthesis, automated program repair (Tang et al. 2021; Zhao et al. 2024), and code explanation. VERSE exhibits noteworthy enhancements across these diverse tasks. For instance, when applied to different base Code LLMs, metrics for these tasks increase by an average of $7 . 6 \%$ . Our extensive analysis underscores the superiority and significance of VERSE.

# 2 Preliminaries

In this section, we provide detailed definitions for instructions and code instructions along with their construction to cover related tasks. These definitions serve to formalize the process of VERSE. The related work section can be found in Appendix A.

Definition 1 (Instruction). Let $T$ be the set of text involved in model input and output, with $N \subseteq T$ representing text exclusively in the natural text modality. We define the tuple $x = ( x . n , \mathrm { \bar { { x . t } } } ) \in T$ to represent an instruction, where $x . n \in$ $N$ is a natural language requirement, and $\boldsymbol { x } . t \in T$ denotes an (optional) text being operated on. The function $G : X  T$ generates responses for instructions.

Definition 2 (Code Instruction). Let $P \subseteq T$ represent text in program modality. We define $C = \{ x | ( \exists x . t \land x . t \in P ) \lor$ $G ( x ) \in P \}$ as the set of code instructions. For any $x \in C$ , the associated text $x . t$ or the response $G ( x )$ is in program modality.

Examples of $C$ are showcased in Figure 1. The natural language parts $x . n$ are “Repair the $C + +$ . . . ” and “Please tell me . . ”, while the code snippets $x . t$ consist of “void sign ” and “ $\mathit { s t r } = \dots \mathit { i }$

# 3 Verification-based Self-Play

In this section, we present a detailed overview of VERSE, elucidating each step of the process. Figure 4 shows the pipeline of VERSE. We use $\theta$ to represent the parameters of Code LLMs.

# 3.1 Automatic Data Generation

Our data generation process consists of three steps: 1) create instructions and responses, 2) create verifications, 3) create scripts. Figures 2 and 3 contain multiple concrete examples.

Write a mean function using Python that returns the average of an 𝒙   
integer list lst (1-10000 length).   
Response contains verifiable code: Yes Programming language: Python   
Response Declaration: def mean(lst: List[int]):   
Verification Declaration: def test_mean() $\ - >$ None:   
def mean(lst: List[int]): 𝒚 return sum(lst) / len(lst)   
This function uses the built-in sum function to implement.   
Implement a test function test_mean() in Python using assertions to 𝒗   
verify the functional correctness of mean.   
Here's an example implementation: 𝒕   
def test_mean() $$ None: assert mean $: [ 1 , 2 , 3 ] ) = = 2 . 0 ,$ "Failed" assert mean $[ - 1 , - 3 ] ) ~ = - 2 . 0 ,$ , "Failed" assert mean $[ 4 2 ] ) = = 4 2 . 0 ,$ "Failed"   
It uses five test samples to judge. (a)   
Please tell me what will be the output of the JavaScript code? 𝒙   
str $\mathbf { \Sigma } = \mathbf { \Sigma }$ "Stack".split('').reverse().join('');   
Response contains verifiable code: No  Programming language: JS   
Response Declaration: Null Verification Declaration: Null   
The output is "kcatS". 𝒚   
Create a JavaScript script to verify whether the following text is the 𝒗   
reverse of "Stack". [Resp]   
To verify whether "kcatS" is the reverse of "Stack", you can use the 𝒕   
following JavaScript script:   
const assert $\mathbf { \Psi } = \mathbf { \Psi }$ require('assert');   
function isReverse(str, target) if (str.length $\ ! = =$ target.length) return false; for (let $\dot { \textbf { 1 } } = \Theta$ ; $\dot { \textbf { 1 } } <$ str.length; $\mathbf { i } { + } { + } )$ 1 if (str[i] $\ ! = =$ target[target.length - 1 - i]) return false; return true;   
}   
assert(isReverse("Stack", "kcatS"), "Error!"); (b)

Create Instructions and Responses Utilizing Self-Instruct (Wang et al. 2023a; Taori et al. 2023), our approach generates an initial set of instructions by using diverse seed instructions and leveraging the few-shot capability of LLMs. We mandate the model to mark four additional attributes for each instruction: whether the corresponding response contains verifiable code, programming language, response declaration, and verification declaration. See Figures 2 and 3 for examples of specific attributes. Specifically distinguishing these attributes aids execution in Section 3.2. Finally, we obtain a dataset of instructions.

We use $y$ to represent the response to the instruction $x$ . For each instruction, the model is prompted using few-shot learning to generate multiple responses. As shown in Figure 4, for instruction $x$ , we generate different responses, represented as $y _ { 1 } , y _ { 2 }$ and $y _ { 3 }$ .

Create Verifications In the subsequent steps, we transform the instructions into verifications - a special type of instruction used to verify the functional correctness of responses. Verifications are designed to prompt the model to generate scripts for assessing the functional correctness of responses.

Explain this code: 𝒙   
double median(std::vector<int> numbers) { std::sort(numbers.begin(), numbers.end()); int size $\mathbf { \Psi } = \mathbf { \Psi }$ numbers.size(); return size $\%$ $2 = \Theta ?$ (numbers[size / 2 - 1] + numbers[size / 2]) / 2.0 : numbers[size / 2];   
}   
Response contains verifiable code: No  Programming language: $\mathrm { C } { + } { + }$   
Response Declaration: Null Verification Declaration: Null   
This $\mathrm { C } { + } { + }$ code defines a function median that calculates the median 𝒚   
of a vector of integers.   
The following is a piece of $C { + } { + }$ code: 𝒗   
double median(std::vector<int> numbers) { std::sort(numbers.begin(), numbers.end()); int size $\mathbf { \Sigma } = \mathbf { \Sigma }$ numbers.size(); return size $\%$ $2 = \Theta ?$ (numbers[size / 2 - 1] + numbers[size / 2]) / 2.0 : numbers[size / 2];   
}   
Write a $\scriptstyle { \mathrm { C } } + +$ script to implement the function corresponding to the   
following text, and determine whether the function is consistent with   
the above function. [Resp]   
double median(std::vector<int> numbers) { 𝒕 std::sort(numbers.begin(), numbers.end()); int size $\mathbf { \Sigma } = \mathbf { \Sigma }$ numbers.size(); return size $\%$ $2 = 0$ ? (numbers[size / 2 - 1] + numbers[size $/ 2 ] ) / 2 . \Theta :$ numbers[size / 2];   
double calculateMedian(std::vector<int> &numbers) { std::sort(numbers.begin(), numbers.end()); int ${ \mathfrak { n } } =$ numbers.size(); return (numbers $[ \mathsf { n } - 1 \mathsf { j } / 2 \mathsf { ] } + \mathsf { }$ numbers[n / 2]) / 2.0;   
}   
int main() { std::vector<int> test1 $= \{ 5 , \ 2 , \ 8 , \ 4 , \ 6 \} ;$ std::vector<int> test2 $\mathbf { \tau } = \mathbf { \tau }$ {10, 9, 7, 3, 1, 2}; assert(calculateMedian(test1) $\scriptstyle = =$ median(test1) && "Inconsistent with provided function for Test ${ } ^ { 1 . " ) }$ ; assert(calculateMedian(test2) $\scriptstyle = =$ median(test2) && "Inconsistent with provided function for Test $2 . " )$ ; std::cout $< <$ "Consistent!" $< <$ std::endl; return 0;   
} (c)

For an instruction $x$ , we use few-shot learning to enable the model to generate the corresponding verification $\boldsymbol { v }$ . We collect multiple verifications per instruction to account for diversity and multi perspectives in verification approaches. For example, a program synthesis instruction can be verified either through static formal verification or by running randomly generated test samples.

Create Scripts As verification is a special kind of instruction, similar to the step for generating responses, we enable the model to generate the test script $t$ for the verification $v$ . Regardless of whether the response $y$ contains code, the script $t$ contains an executable program for testing.

During this step, the presence of code in the response $y$ is crucial. If $y \in P$ , like verifying a function as shown in case (a) of Figure 2, we prompt the model to generate test scripts, unrelated to the specific function implementation. If $y \notin P$ , which means the response $y$ is in natural language, like cases (b) and (c) in Figures 2 and 3, direct execution is impossible. In such cases, verification is response-specific. We prompt Code LLMs to generate verification ideas and combine them with responses. In case (a), for instance, the verification idea is to generate a program checking if the response is a string reversal; then, the virtual text link “[Resp]” is replaced with the content of response $y$ , to generate the actual prompt for generating test scripts. This enables verifying different responses using the same idea, ensuring a fair evaluation of functional correctness for each response.

# 3.2 Post-processing and Execution

We extract code snippets from response $y$ and script $t$ , aggregating them for execution. The code snippet extracted from response $y$ is denoted as $\overline { y }$ , and from script $t$ is denoted as $\hat { t }$ If the response is in program modality, denoted as $y \in P$ , we combine $\overline { y }$ and $\hat { t }$ for execution. If $y \notin P$ , we extract only $\hat { t }$ for execution.

The process of aggregating programs considers verifiability and programming language, selecting the execution entry point based on declarations mentioned in Section 3.1. If there is no relevant declaration, the entire extracted code is executed. The program runs in a multi-language sandbox with common dependencies for security, yielding final execution results. We denote the result using the function $\mathrm { r } ( \overline { { y } } , \overline { { t } } )$ , which evaluates to 1 for successful execution and 0 in case of an error, as shown in Figure 4.

# 3.3 Consistency-driven Data Ranking

In this subsection, we outline the process for ranking the quality of responses with execution results and self-consistency. Inspired by intra-consistency and inter-consistency (Jung et al. 2022; Huang et al. 2023), we define two types of consistency based on the equivalence of responses and the results of execution, combined with the model’s confidence in the verification, to calculate the verification scores for data ranking. We provide the definitions for both types of consistency first, and then we introduce the calculation of verification scores. Examples of both types of consistency are provided in Figure 4.

Equivalence-based Consistency If multiple responses share the same functionality, they are more likely to be correct. Therefore, for a specified function, we calculate the probability of the model generating a response equivalent to it for data ranking. We measure the ratio of responses implementing the desired function to quantify equivalence. To address difficulties in establishing direct equivalence, we draw inspiration from dynamic equivalence (Gulwani, Radicek, and Zuleger 2016; Hu et al. 2019) and self-consistency (Wang et al. 2022). We define two responses as equivalent if they consistently exhibit the same behavior across an extensive set of verifications.

We organize responses by grouping them with identical functions based on their performance in self-generated verifications. We denote the group corresponding to $y$ as $c ^ { y }$ . As shown in Figure 4, $y _ { 1 }$ belongs to $c ^ { y _ { 1 } }$ , and $y _ { 2 }$ and $y _ { 3 }$ both belong to $c ^ { y _ { 2 } }$ . Subsequently, we compute the probability of generating a response with the required functionality $c$ :

$$
{ \mathcal { P } } ( c | x ) = \mathbb { E } _ { y } [ y \in c ]
$$

Execution-based Consistency We also measure consistency between a response and its corresponding test scripts.

![](images/423b00deb54a9af66f83a2a34b4b53a2423cbda86446c771cc3a9dad6304152a.jpg)  
Figure 4: Pipeline of VERSE, illustrating the key steps.

We calculate the probability of successful execution of the post-processed program, denoted as $\mathcal { P } ( s | y , x )$ , using the following formula:

$$
\mathcal { P } ( s | y , x ) = \mathbb { E } _ { v } [ \mathcal { P } ( s | v , y , x ) ] = \mathbb { E } _ { v , t } [ \mathrm { r } ( \overline { { y } } , \overline { { t } } ) ]
$$

Verification score for Data Ranking We derive the verification score $\mathcal { R } ( y | x )$ for each response by systematically integrating consistency factors, which act as criteria for data ranking. Our basic method involves calculating the product of these factors, as described below:

$$
\mathcal { R } ( y | x ) = \mathcal { P } ( c ^ { y } | x ) \mathcal { P } ( s | y , x )
$$

However, verification difficulty varies among code instructions. For example, repair instructions with error reporting functionality are relatively simple, while verifying answers to competition-level problems is more difficult. Considering this variability, we assign weights to relevant probabilities. To assess the model’s confidence in the verification scheme, we employ two metrics: entropy calculation for verifications, and the evaluation of consistency among verification schemes across various responses. The respective formulas for these metrics are as follows:

$$
\mathcal { H } ( v | x ) = - \mathbb { E } _ { v } [ \log \mathcal { P } ( v | x ) ] , \mathcal { P } ( s | x ) = \mathbb { E } _ { y } [ \mathcal { P } ( s | x , y ) ]
$$

We evaluate verification confidence using the product of ${ \mathcal { P } } ( s | x )$ and the reciprocal of $\mathcal { H } ( v | x )$ as weights for respective probabilities. This method is model and instruction agnostic. When the response passes inconsistent verification schemes or the entropy of generated verifications is high, verification confidence is low, and priority is given to referencing the equivalence-based consistency score. Conversely, a higher weight is assigned to the execution-based consistency score. The calculation formula is as follows:

$$
w = \alpha \mathcal { P } ( s | x ) \mathcal { H } ( v | x ) ^ { - 1 } , \mathcal { R } ( y | x ) = \mathcal { P } ( c ^ { y } | x ) \mathcal { P } ( s | y , x ) ^ { w }
$$

Here, $\alpha$ serves as a hyperparameter, representing a prior on the importance of two types of consistency. The analysis

of this design and hyperparameter is detailed in Appendix D.1. We rank generated responses by verification scores and integrate them into the candidate data pool.

# 3.4 Sampling and Training

We use rejection sampling fine-tuning (Dong et al. 2023) to train the model. We select the responses with the highest verification scores corresponding to each instruction data from the candidate data pool and combine them to train the model. In cases of consistent verification scores from multiple samples, we randomly select samples for training.

# 4 Experimental Setup

# 4.1 Code LLMs

In our experiments, we use two popular open-source Code LLMs: CodeLlama (Rozie\`re et al. 2023), pre-trained on an extensive code dataset, and Deepseek-Coder (Guo et al. 2024), trained from scratch. We mainly utilize CodeLlama-7B and Deepseek-Coder-6.7B-Base to conduct primary experiments; models with other sizes are used for comparison and analysis.

# 4.2 Initialization

For LLMs, the training process usually adopts multiple stages, and the next stage of training starts from checkpoints of the previous stage (Ouyang et al. 2022). In our experiments, we start from both base Code LLMs and checkpoints after finetuning with some data to fully prove the effectiveness of VERSE.

To start training using base Code LLMs, we employ the original Self-Instruct pipeline as a baseline to generate an equivalent amount of training data without the verification step. For training with tuned checkpoints, we use two publicly popular datasets: CodeAlpaca-20k and Evol-instruction-66k to build the initial checkpoints. We also compare them with the evaluation results of the official instruction versions for both models. Additional discussions can be found in Appendix E.

Table 1: Summary of benchmarks and datasets used in this paper.   

<html><body><table><tr><td>Task</td><td>Category</td><td>Dataset</td><td>Language</td><td>Size</td><td>Metric</td></tr><tr><td>CD</td><td>N+P→N</td><td>BigCloneBench</td><td>Java</td><td>2,000</td><td>F1</td></tr><tr><td>DD</td><td>N+P→N</td><td>Devign</td><td>C</td><td>2,000</td><td>Acc</td></tr><tr><td>PS</td><td>N→P</td><td>HumanEvalSynthesis</td><td>Python C++ Java JS Go Rust</td><td>164</td><td>Pass@1</td></tr><tr><td>APR</td><td>N+P→P</td><td>HumanEvalFix</td><td>Python C++ Java JS Go Rust</td><td>164</td><td>Pass@1</td></tr><tr><td>CE</td><td>N+P→N</td><td>HumanEvalExplain</td><td>Python C++ Java JS Go Rust</td><td>164</td><td>Pass@1</td></tr></table></body></html>

# 4.3 Data Generation

We utilize vLLM (Kwon et al. 2023) for accelerated sampling in generation. We consider data repeatability and generation quality (see Appendix B.1 for details). In our experiment, for both CodeLlama-7B and Deepseek-Coder-6.7B-Base, we collect a dataset comprising 100,000 instructions. Each instruction is associated with 20 responses and 20 verificationscript pairs. The impact of sampling quantity and quantitative analysis are discussed in Appendix D.2 and Appendix D.3.

# 4.4 Execution Details

We execute programs within a sandbox that supports multiple programming languages. To ensure execution security, we limit the use of associated hardware resources. Details of the execution environment are available in Appendix B.2. We compute the verification score for each response. If the verification score for an instruction is 0, the instruction is deemed invalid. Ultimately, for different models, we obtain around 50,000 valid instructions with corresponding responses and verification scores. To ensure a fair comparison, we retain 40,000 valid instructions for training, adjusting for variations in the amount of valid data obtained by different models.

# 4.5 Training Settings

Our models are trained on 2 Nvidia A100 GPUs for 2 epochs using the Transformers library. We employ Alpaca-style instruction templates (Taori et al. 2023) for training, and we set the hyperparameter $\alpha$ for calculating the verification score to 4. Memory efficiency and speed are enhanced through techniques, including Deepspeed ZeRO3 (Rajbhandari et al. 2019) and FlashAttention2 (Dao 2023). We configure a batch size per GPU of 32, a maximum sequence length of 2048, and a learning rate of 5e-5. Training employs the Adafactor optimizer (Shazeer and Stern 2018), coupled with a cosine scheduler featuring 15 warm-up steps.

# 5 Evaluation

We introduce the benchmarks used in the experiments in Section 5.1. We report the main results of the experiments in Section 5.2 and conduct some analysis in Section 5.3. In the following sections, Section 5.4, Section 5.5, and Section 5.6, we perform additional experiments and analysis.

# 5.1 Benchmarks

Our experiments encompass five tasks from two widely recognized benchmarks, CODEXGLUE (Lu et al. 2021)

and HUMANEVALPACK (Muennighoff et al. 2023), these tasks aim to evaluate the quality of responses to diverse code instructions. The first two tasks are collected from the CODEXGLUE benchmark, while the last three tasks are collected from the HUMANEVALPACK benchmark. The specific tasks are as follows:

• Clone Detection (CD): Determine whether two pieces of code belong to the same function.   
• Defect Detection (DD): Identify whether a code snippet contains defects.   
• Program Synthesis (PS): Synthesize programs to specified functional requirements.   
• Automated Program Repair (APR): Fix error functions that fail during execution.   
• Code Explanation (CE): Generate a relevant functional description for a given function.

Three tasks in HUMANEVALPACK are extended from the HUMANEVAL dataset (Cassano et al. 2023). We use them to examine models responding to various code instructions for evaluating the effectiveness of VERSE. Please see Table 1 and Appendix C for details of the evaluation metrics and datasets for various tasks.

# 5.2 Primary Results

We report the evaluation results of CodeLlama-7B and Deepseek-Coder-6.7B-Base after training with VERSE. The results and comparisons of training from different models are shown in Table 2, Table 3, Figure 5. The results shown for tasks in HUMANEVALPACK are average results for each language. Detailed results for each programming language and task can be found in Appendix F.

The experimental results highlight VERSE’s outstanding performance in enhancing Code LLMs for generating correct responses to code instructions. Base Code LLMs trained with VERSE show a significant average improvement of $7 . 6 \%$ across various tasks. Models fine-tuned on CodeAlpaca-20k (20 examples in total) and Evol-instruction-66k (66 highquality examples in total) checkpoints demonstrate an average improvement of $7 . 6 \%$ and $3 . 5 \%$ . Additionally, the best model trained with VERSE outperforms the official instruction versions of both models on multiple tasks and average metrics, performing well at limited training scale.

CodeLlama-7B VERSE CodeAlpaca-20k $^ +$ VERSE Evol-instruction-66k $^ +$ VERSE GPT-4 Self-Instruct CodeAlpaca-20k Evol-instruction-66k CodeLlama-7B-Instruct 80 70   
560   
340 ！ ！ 10 0 CD (N + P -> N) DD (N + P -> N) PS (N -> P) APR (N + P -> P) CE (N + P -> N) Deepseek-Coder-6.7B-Base VERSE CodeAlpaca-20k $^ +$ VERSE Evol-instruction-66k $^ +$ VERSE GPT-4 Self-Instruct CodeAlpaca-20k Evol-instruction-66k □ Deepseek-Coder-6.7B-Instruct 80 70   
34560 10 WI.I ！ 0 DD (N + P -> N) PS (N -> P) APR (N + P -> P) CE (N + P -> N)

Table 2: The evaluation results of training from CodeLlama7B (CL-7B) using VERSE.   

<html><body><table><tr><td>Method</td><td>CD</td><td>DD</td><td>PS</td><td>APR</td><td>CE</td><td>Avg.</td></tr><tr><td>From CL-7B:</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>12.8</td><td>47.4</td><td>31.5</td><td>2.4</td><td>12.6</td><td>21.3</td></tr><tr><td>Self-Instruct</td><td>13.4</td><td>45.9</td><td>34.8</td><td>6.1</td><td>30.7</td><td>26.2</td></tr><tr><td>VERSE</td><td>39.2</td><td>47.6</td><td>38.4</td><td>10.5</td><td>31.8</td><td>33.5</td></tr><tr><td>From checkpoints fine-tuned using CodeAlpaca-20k: W/o VERSE VERSE</td><td>17.8 46.2 29.0 37.3 47.4 38.6</td><td></td><td></td><td>20.0</td><td>23.9</td><td>27.4 35.7</td></tr><tr><td>Fromcheckpoints fine-tuned usingEvol-instruction-66k:</td><td></td><td></td><td></td><td></td><td>33.9</td><td></td></tr><tr><td>W/oVERSE VERSE</td><td>47.9 58.7</td><td>53.2 55.2</td><td>44.9 46.2</td><td>34.2 34.7</td><td>32.7 33.2</td><td>42.6 45.6</td></tr></table></body></html>

# 5.3 Analysis

We further analyze the specific results in various tasks and find that VERSE helps different tasks differently. VERSE ensures the quality of data for some tasks like clone detection, program synthesis, and code explanation when the model directly responds without execution. When using Self-Instruct or fine-tuning with CodeAlpaca- $2 0 \mathrm { k }$ , we are surprised to find that their metrics on some tasks even drop compared to the base model. The official instruction version, DeepseekCoder-6.7B-Instruct, also exhibits a similar phenomenon on the clone detection and defect detection task. This further confirms the importance of ensuring data quality. Data generated without verification may contain errors that can greatly damage the model.

Evaluation results demonstrate VERSE’s significant improvements in clone detection, program synthesis, and code explanation tasks. However, its impact on defect detection and automated program repair tasks is limited. Our analysis suggests that the primary reason lies in the model’s difficulty in generating code resembling genuine human errors. Even GPT-4, when tasked with producing code with syntax errors, often fails to comply. High-quality training data for LLMs actually hinders the creation of incorrect inputs, aligning with previous studies using synthetic data to enhance code repair capabilities for models (Allamanis, Jackson-Flux, and Brockschmidt 2021; Yasunaga and Liang 2021; He, BeurerKellner, and Vechev 2022). We leave this for future work to make the instructions generated by LLMs closer to the true distribution of instructions written by humans by combining VERSE and other methods.

Table 3: The evaluation results of training from DeepseekCoder-6.7B-Base (DS-6.7B) using VERSE.   

<html><body><table><tr><td>Method</td><td>CD</td><td>DD</td><td>PS</td><td>APR</td><td>CE</td><td>Avg.</td></tr><tr><td colspan="7">FromDS-6.7B-Base:</td></tr><tr><td>Base</td><td>51.8</td><td>54.0</td><td>46.1</td><td>27.0</td><td>34.2</td><td>42.6</td></tr><tr><td>Self-Instruct</td><td>12.3</td><td>45.8</td><td>53.0</td><td>23.2</td><td>35.9</td><td>34.0</td></tr><tr><td>VERSE Fromcheckpoints fine-tuned using CodeAlpaca-20k:</td><td>53.1</td><td>54.5</td><td>55.3</td><td>27.1</td><td>37.6</td><td>45.5</td></tr><tr><td>W/o VERSE VERSE</td><td>12.1 51.0 48.7 34.2 54.4 53.2</td><td></td><td></td><td>39.3 39.7</td><td>32.5 36.0</td><td>36.7 43.5</td></tr><tr><td>Fromcheckpoints fine-tuned usingEvol-instruction-66k: W/o VERSE</td><td>46.3</td><td>53.9</td><td>56.7</td><td>47.9</td><td>40.7</td><td>49.1</td></tr><tr><td>VERSE</td><td>57.0 DS-6.7B-Instruct 22.1 49.2</td><td>54.1</td><td>60.7</td><td>49.1</td><td>44.1</td><td>53.0</td></tr></table></body></html>

# 5.4 Distillation to Smaller Models

We also investigate the possibility of condensing knowledge into smaller Code LLMs by training them with data generated by larger Code LLMs, which is known as knowledge distillation (Agarwal et al. 2023). We use Deepseek-Coder1.3B-Base for experiments, comparing the results of selfimprovement and distillation from Deepseek-Coder-6.7BBase. Data is generated through the pipeline of VERSE. Results are presented in Figure 6. It can be found that the model trained using distilled data is better than the self-trained model. The evaluation metrics of the distillation model increase by $7 . 1 \%$ on average compared with the base model and increase by $2 . 5 \%$ compared with the self-trained model. This enlightens us that the distilled data, verified by the large model, can enhance the small model’s ability to obtain more competitive models in scenarios with limited computing resources.

50 山 Base   
40 VERSE   
3 Distillation $^ +$ VERSE   
20 10 1   
0   
CD APR CE

# 5.5 Iterative Optimization

Since VERSE can enhance responses to code instructions, we also explore whether it can iteratively improve the model. We conduct experiments using CodeLlama-7B, which iteratively performs four rounds of training, each round collecting 10,000 instances for training through the VERSE pipeline. The results are shown in Figure 7. Iterative training has obvious improvements in the first two epochs, but there is no obvious change in the last two epochs. This may be because the functional correctness of the responses is an absolute metric. Therefore, the degree of optimization has a certain upper limit, and it is difficult to carry out many rounds of optimization like relative metrics (Sun et al. 2023; Chen et al. 2024; Yuan et al. 2024).

# 5.6 Rerank with Verification

Similar to various reranking methods (Inala et al. 2022; Ni et al. 2023; Zhang et al. 2022), we also use the Python subset in HUMANEVALPACK to evaluate the model’s ability to verify and rank multiple responses by assessing the execution results of the reranked responses. The verification and data ranking methods are the same as VERSE. Figure 8 presents the results. It can be found that reranking through self-verification has a good improvement in model performance, and the model trained with VERSE shows even greater improvement. The verification difficulty of different tasks varies and has a certain impact on the extent of improvement. For example, the instructions of fixing programs include error samples, and it is less difficult to verify whether the repair is successful, so the improvement through verification is huge. For code explanation tasks, the improvement is smaller because of the uncertainty of regenerating the program for verification based on the generated explanation.

![](images/b7723966e2bdabfd2f4ffa6d7dd2c2b8d11772c9e8ce616f719edbf71b713398.jpg)  
Figure 6: Distillation from large models to small models.   
Figure 7: Changes in metrics of each task during the iterative training of CodeLlama-7B.   
Figure 8: Evaluation results based on verfication and reranking for responses.

80 w/o VERSE & Origin VERSE & Origin 70 I w/o VERSE & Rerank VERSE & Rerank 560 40 三 30 PS

# 6 Limitation

In the experimental section, due to resource and data limitations, we do not conduct sufficient experiments on larger models. For our approach, a limitation is that VERSE depends on the execution of programs, and it is impossible to cover all possible situations; code instructions with complex dependencies or multiple file relationships are difficult to execute directly. In addition, VERSE enhances the functional correctness of responses through execution-based self-verification, but the evaluation metrics for code instructions are not unique. Besides ensuring functional correctness, we also want the results to be reliable, explainable, and faster. We will overcome these limitations and combine them with other methods to improve the quality of instructions in the future.

# 7 Conclusion

We proposed VERSE, utilizing verification-based self-play to enhance responses to code instructions. Our approach combined execution and self-consistency, allowing Code LLMs to self-verify the functional correctness of responses and train themselves based on the verification results. Our experiments succeeded on multiple Code LLMs, including CodeLlama, Deepseek-Coder, and various benchmarks such as CODEXGLUE and HUMANEVALPACK. They demonstrated that VERSE improves the model’s ability and is easily generalized to different models and code-related tasks.