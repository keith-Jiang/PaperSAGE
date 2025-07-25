# InverseCoder: Self-improving Instruction-Tuned Code LLMs with Inverse-Instruct

Yutong ${ \mathbf { W } } { \mathbf { u } } ^ { 1 , 2 }$ , Di Huang1, Wenxuan Shi1, 2, Wei $\mathbf { W a n g } ^ { 3 }$ , Yewen $\mathbf { P u } ^ { 4 }$ , Lingzhe Gao3 Shihao ${ \bf L i u } ^ { 3 }$ , Ziyuan Nan1, 2, Kaizhao Yuan1, 2, Rui Zhang1, Xishan Zhang1, Zidong $\mathbf { D } \mathbf { u } ^ { 1 }$ , Qi $\mathbf { G u o } ^ { 1 }$ , Dawei $\mathbf { Y i n } ^ { 3 }$ , Xing $\mathbf { H } \mathbf { u } ^ { 1 }$ , Yunji Chen1, 2\*

1SKL of Processors, Institute of Computing Technology, CAS 2University of Chinese Academy of Sciences 3Baidu Inc., Beijing, China 4Autodesk Research wuyutong22s@ict.ac.cn

# Abstract

Recent advancements in open-source code large language models (LLMs) have been driven by fine-tuning on the data generated from powerful closed-source LLMs, which are expensive to obtain. This paper explores whether it is possible to use a fine-tuned open-source model to generate additional data to augment its instruction-tuning dataset. We make two observations: (1) A code snippet can serve as the response to different instructions. (2) Instruction-tuned code LLMs perform better at translating code into instructions than the reverse. Based on these observations, we propose Inverse-Instruct, a data augmentation technique that uses a fine-tuned LLM to generate additional instructions of code responses from its own training dataset. The additional instruction-response pairs are added to the original dataset, and a stronger code LLM can be obtained by fine-tuning on the augmented dataset. We empirically validate Inverse-Instruct on a range of open-source code models (e.g., CodeLlama-Python and DeepSeek-Coder) and benchmarks (e.g., HumanEval $\left( + \right)$ , $\mathbf { M B P P ( + ) }$ , DS-1000 and MultiPL-E), showing it consistently improves the base models.

# Code — https://github.com/wyt2000/InverseCoder Extended version — https://arxiv.org/abs/2407.05700

# 1 Introduction

Code generation, which aims to generate code that satisfies the user’s intent from inputs/outputs or natural language, has been a significant challenge in computer science. Recently, closed-source LLMs like GPT-3.5 and GPT-4 (OpenAI 2023) have enabled the generation of general-purpose code (like Python) based on natural language, making them broadly applicable in the fields of programming assistance (Microsoft 2023), computer vision (Sur´ıs, Menon, and Vondrick 2023; Gupta and Kembhavi 2023), science (Nejjar et al. 2023), and embodied intelligence (Liang et al. 2023; Ma et al. 2023; Tang, Key, and Ellis 2024; Wang et al. 2023).

To develop high-performance open-source models, researchers have leveraged these closed-source LLMs to generate datasets of instructions and code, then distilled these datasets into smaller, open-source code LLMs via instruction tuning (Luo et al. 2023; Wei et al. 2023; Yu et al. 2023; Song et al. 2024). For example, Code Alpaca (Chaudhary 2023) was fine-tuned on 20K instruction-code pairs generated based on GPT-3.5 with SELF-INSTRUCT (Wang et al. 2022). Luo et al. (2023) used Evol-Instruct (Xu et al. 2023), a method that creates a diverse set of instruction data from GPT-3.5 for code generation via evolution heuristics. OSSINSTRUCT (Wei et al. 2023) first creates coding problems from the source code snippet, then queries strong LLMs for their corresponding solutions. Fine-tuned with 75K GPT3.5 OSS-INSTRUCT data and 110K GPT-4 Evol-Instruct data (i.e. evol-codealpaca-v1) (theblackcat102 2023), Magicoder series achieve state-of-the-art (SOTA) results among open-source code models. These approaches have one thing in common: they heavily rely on generating data by querying stronger closed-source LLMs (e.g., GPT-4), which incurs significant additional expenses. Therefore, it is crucial to develop a self-improvement method for open-source models without relying on stronger guidance.

This paper explores how to improve an instruction-tuned code LLM by querying itself (rather than querying a closedsource LLM). We make the following two observations :

1. A single code snippet can serve as a valid response to multiple instructions.   
2. Instruction-tuned code LLMs perform better at translating code into instructions than translating instructions into code (see Section 3).

The first observation suggests that an instruction-tuned LLM can generate a new instruction for each response code in its training dataset, thereby expanding the original dataset. The second observation confirms that generating data in this way (Code-to-NL) is more effective than NL-to-Code.

Therefore, we develop Inverse-Instruct, a simple yet effective instruction tuning approach based on self-generating instructions from code snippets (Figure 1). Inverse-Instruct starts with an instruction-code corpus, and a code LLM finetuned on it. We first clean and extract code snippets from the corpus, then let the code LLM translate these code snippets into new instructions. Next, we use the code LLM to evaluate and filter consistent instruction-code pairs from the newly generated data. Finally, the filtered dataset is combined with the original instruction dataset to fine-tune a new model. The

Original Summarized Inverse   
Instruction instruction data instruction   
@d@aitnsatrsuection PrepCrocdessing sClneipapnectosde SummCaordizeation C@re@aitnesatrpuyctihon 1program with S&eldf-aetvaasleulaetcitoionn dataset   
bug … Fix this error and \`while’ … Create a python program with   
modify the script … def traverse_list(…): @@instruction 2 \`while’ …   
def traverse_list(…): n = len(arr) Create a python program with @@response   
@@response whilper in <( nr:r[i]) \`f@or@’in…struction 3 … def tnra=velresne(_arlris)t(…):   
def traverse_list(…): i += 1 @@response whilep $\dot { \textbf { i } } < \mathsf { n }$ :rr[i])   
In the given code, it tries to … Instruction-tuned def traverse_list(…): nstruction-tuned $\textbf { i } + = \textbf { 1 }$ model model

main differences between Inverse-Instruct and previous data generation methods are discussed in Section 2.2.

Using Inverse-Instruct, we develop InverseCoder, a series of fine-tuned code LLMs that achieve SOTA results. We evaluated InverseCoder on a wide range of benchmarks (Section 6), including HumanEval $\left( + \right)$ (Chen et al. 2021; Liu et al. 2023), $\mathbf { M B P P ( + ) }$ (Austin et al. 2021; Liu et al. 2023), MultiPL-E (Cassano et al. 2023), and DS-1000 (Lai et al. 2023). Results show that InverseCoder series surpasses the base models by exploiting the base models’ own capability. Specifically, InverseCoder-DS-6.7B achieves $7 6 . 8 \%$ on HumanEval+, $6 9 . 0 \%$ on $\mathbf { M B P P + }$ , $6 2 . 6 \%$ on MultiPL-E, $4 4 . 2 \%$ on DS-1000, which are SOTA results across four benchmarks among fully open-source (both model and dataset) models with only 6.7B parameters.

Our key contributions are introducing Inverse-Instruct, an effective self-improvement instruction tuning approach for code LLMs and presenting a series of code LLMs named InverseCoder, which achieves SOTA or comparative results on a wide range of benchmarks.

We organize the structure of the paper as follows: Section 2 introduces related works. Section 3 shows the evidence of our observations. Section 4, 5 provide a detailed explanation of our approach (i.e. Inverse-Instruct). Section 6 presents the experiments for our models (i.e. InverseCoder). Section 7 concludes with a summary.

# 2 Related Work

# 2.1 LLMs for Code Generation

After being pre-trained on a large amount of code, LLMs have demonstrated impressive code generation capabilities. Recently, AI code assistants have become one of the most important applications of LLMs. Technology companies such as OpenAI and Google have developed and released many closed-source large language models, including Codex (Chen et al. 2021), GPT-4 (OpenAI 2023), PaLM (Chowdhery et al. 2022), and Gemini (Team et al. 2023), which have achieved outstanding performance on code generation benchmarks.

In addition to closed-source models, there are also some available open-source models whose weights and training data are available to the public, such as CodeGen (Nijkamp et al. 2022), CodeGeeX (Zheng et al. 2023), AlphaCode (Li et al. 2022), CodeT5 series (Wang et al. 2021), StarCoder series (Li et al. 2023; Lozhkov et al. 2024), CodeLlama (Rozi\`ere et al. 2023), DeepSeek-Coder (Guo et al. 2024) and CodeQwen (Bai et al. 2023). These open-source code models have shown notable advancements in code-related tasks, but there is still a gap compared to the most advanced code LLMs.

# 2.2 Instruction-Tuned Code LLMs

Instruction tuning is a method for further enhancing the instruction-following capability of pre-trained LLMs. It has been widely applied to the LLMs for general tasks including T5 (Raffel et al. 2020) and FLAN (Wei et al. 2021).

For code LLMs, OctoPack (Muennighoff et al. 2023) and PIE (Shypula et al. 2024) extracted high-quality data from human-written instructions and code. Fine-tuning with these data has significantly enhanced the program generation capabilities of the base models.

However, obtaining high-quality human-written instruction datasets is usually laborious. Researchers have attempted to employ stronger closed-source LLMs to generate both new instructions and responses for instruction-tuning. Specifically, CodeAlpaca (Chaudhary 2023) sampled tasks from a seed task pool and prompted a stronger LLM to generate instruction-tuning data based on the seed tasks. WizardCoder (Luo et al. 2023) prompted a stronger LLM to generate more complex instructions and the corresponding responses. Magicoder (Wei et al. 2023) used a stronger LLM to create problems and code solutions based on open-source code snippets, as the seed code snippets offer controllability and diversity to the generation. WaveCoder (Yu et al. 2023) used a stronger LLM to both generate and discriminate the instruction-response pair for different coding tasks (e.g., code summarization and code repair). AlchemistCoder (Song et al. 2024) employed a stronger LLM to add more details for existing instructions.

The main differences between our method and the aforementioned related works are:

<html><body><table><tr><td>Generation Method</td><td>WC-CL-7B WC-DS-6.7B</td></tr><tr><td>NL →Code</td><td>62.4 70.2</td></tr><tr><td>Code →NL GPT-4Code</td><td>74.3 79.0</td></tr><tr><td>Code →NL Code Humans</td><td>86.7 80.0</td></tr></table></body></html>

Table 1: Pass ${ \ @ \mathbf { 1 } }$ $( \% )$ results on $\mathbf { M B P P + }$ in observation checking experiments. The abbreviations “WC-CL-7B” and “WC-DS-6.7B” refer to the instruction-tuned models WizardCoder-GPT4-CL and WizardCoder-GPT4-DS. Line 1 represents the evaluation of NL-to-Code for instruction-tuned open models. Lines 2 and 3 evaluate Code-to-NL by leveraging GPT-4 and humans to convert NL into its equivalent code, then assess its correctness against the original code. We removed the problems that GPT-4 was unable to give executable code for them.

• We focus on the self-improvement of open-source code models rather than relying on stronger guidance (such as human annotation or advanced LLMs like GPT-4). • We generate new data by converting code to instructions from existing datasets rather than generating code from instructions.

# 3 Sanity Check: Code-to-NL vs. NL-to-Code

In this section, we validate our observation that instructiontuned code LLMs perform better at translating code into instructions (i.e., Code-to-NL) than translating instructions into code (i.e., NL-to-Code) through an experiment.

We first select a manually written set of correctly matched NL-Code pairs $\{ x , y \}$ with unit tests and prompted a finetuned code LLM to convert $x$ into new code $y ^ { \prime }$ and $y$ into new NL $x ^ { \prime }$ separately. Then, We use the following metrics to quantify the model’s performance in the two tasks:

• For NL-to-Code, we use unit tests to evaluate the functional correctness of generated code $y ^ { \prime }$ against original code $y$ . • For Code-to-NL, we convert generated $\mathrm { N L } \boldsymbol { x } ^ { \prime }$ to an equivalent code snippet $\hat { y }$ by humans and a stronger code LLM. Then we measured the functional correctness of $\hat { y }$ by unit tests.

Specifically, we use the problem-answer pairs with unit tests in a basic Python generation benchmark $\mathbf { M B P P + }$ (Liu et al. 2023) as matched NL-Code pairs $\{ x , y \}$ . For NL-toCode, we took all 378 problems in the benchmark for evaluation. For Code-to-NL, we first select 30 problems for humans to write the equivalent code of the generated NL, and then we employ GPT-4 to finish this task for all problems.

We evaluate two instruction fine-tuned code LLMs (i.e., WizardCoder-GPT4-CL and WizardCoder-GPT4-DS, which are instruction-tuned by 110K GPT-4 dataset $\mathtt { e v o l - c o d e a l p a c a - v 1 } )$ . The results are shown in Table 1. From the table, we conclude that ( $\mathrm { C o d e } \to \mathrm { N L }$ ) is better than $( \mathrm { N L } \to \mathrm { C o d e } )$ , showing that code LLMs perform better in code summarization than in code generation.

# 4 Inverse-Instruct: Data Augmentation via Code Summarization

In this section, we will introduce Inverse-Instruct, a data augmentation method that can obtain more instruction data through the model’s own capabilities. The overall illustration of Inverse-Instruct is shown in Figure 1. Inverse-Instruct is founded on the following two observations: (1) The same code can be considered as a response to different instructions, which expands the dataset effectively. (2) Converting formal language (i.e., code) into informal language (i.e., natural language) is generally more straightforward than the reverse.

The whole data generation process contains three stages: (1) Code preprocessing. (2) Code summarization. (3) Selfevaluation and data selection. In code preprocessing, we preprocess the code data by filtering clean code snippets $\{ y _ { i } ^ { * } \}$ from an off-the-shelf instruction tuning dataset $\{ ( x _ { i } , y _ { i } ) \}$ (e.g., evol-codealpaca-v1). Subsequently, in code summarization, we prompt an instruction fine-tuned code LLM $M$ (e.g., WizardCoder-GPT4-CL) to convert the clean code snippets $\{ y _ { i } ^ { * } \}$ to multiple new instructions $\{ x _ { i j } ^ { * } \}$ . Then, in self-evaluation and data selection, we use the same code LLM $M$ to select the best instruction $x _ { i } ^ { * * }$ in $\{ x _ { i j } ^ { * } \}$ . The selected instructions $\{ x _ { i } ^ { * * } \}$ are combined with the original code snippets $\{ y _ { i } ^ { * } \}$ to construct a new instruction tuning dataset $\{ ( x _ { i } ^ { * * } , y _ { i } ^ { * } ) \}$ . Finally, we fine-tune the base code LLM with the instruction data $\{ ( x _ { i } ^ { * * } , y _ { i } ^ { * } ) \} \cup \{ ( x _ { i } , y _ { i } ) \}$ to obtain a stronger code LLM (i.e. InverseCoder). Details of the three steps are illustrated below.

# 4.1 Code Preprocessing

The first step is to preprocess the existing code data and get clean code snippets $\{ y _ { i } ^ { * } \}$ . This is because the Code-to-NL capabilities of code LLMs can only be fully utilized with clean code, whereas the response data $\{ y _ { i } \}$ in the original dataset typically contains a lot of noise, such as natural language responses.

We select data with code snippet $\{ y _ { i } ^ { * } \}$ from the original $\{ y _ { i } \}$ with the following two steps:

1. Filtering responses. We first collect responses that contain the marker of the code block $( i . e . ^ { \cdots } )$ , which indicates that there are code snippets in the response. The remaining data might contain clean code without any code markers, so then we collect the responses that can pass syntax checking.

2. Extracting code. After filtering responses with code snippets, we remove the natural language surrounding the code snippets to make it easier for the model to summarize. If there are multiple parts of code in the original response, we only keep the first part, since the following parts are usually test cases or using examples.

At the end of code preprocessing, we obtain clean code snippets $\{ y _ { i } ^ { * } \}$ for summarization.

# 4.2 Code Summarization

After filtering, we employ the code LLM $M$ to generate a certain number of corresponding instructions $\{ x _ { i j } ^ { * } \}$ for each code snippet in $\{ y _ { i } ^ { * } \}$ by summarizing its functionality. During the summarization process, we randomly choose different instruction prefixes for the prompt to enhance the diversity of the instructions.

In this way, we have obtained new pairs of natural language and code $\{ ( x _ { i j } ^ { * } , y _ { i } ^ { * } ) \}$ .

# 4.3 Self-evaluation and Data Selection

We noticed that code LLM $M$ might make mistakes during the code summarization process. Therefore, we utilize $M$ itself to evaluate $\{ ( x _ { i j } ^ { * } , y _ { i } ^ { * } ) \}$ and select the most appropriate instruction.

Data selection is typically performed by powerful LLMs such as GPT-4 because these models possess excellent instruction-following capabilities, enabling them to understand complex filtering rules (Wang et al. 2024). However, the instruction-following capabilities of code LLMs are often weaker, making it difficult to conduct effective selection. (See the comparison experiments in Section 6.5).

Inspired by AutoMathText (Zhang et al. 2024), we use the pseudo-probability of YES token given by the code LLM $M$ as an indicator of the instruction quality rather than a score in textual format. Specifically, we concatenate the generated instructions $\{ x _ { i j } ^ { * } \}$ and the original code snippets $\{ y _ { i } ^ { * } \}$ as problem-answer pairs $\{ ( x _ { i j } ^ { * } , y _ { i } ^ { * } ) \}$ . Then, we ask $M$ to evaluate the correctness of each answer under the given problem and calculate the pseudo-probability of YES using the logits of the first token given by $M$ . The formula for calculating the pseudo-probability is shown as follows (Zhang et al. 2024):

$$
\operatorname { L M - S c o r e } ( \cdot ) = { \frac { \exp ( \log \mathrm { i t } ( \mathbf { \dot { \prime } Y E S ^ { \prime } } ) ) } { \exp ( \log \mathrm { i t } ( \mathbf { \dot { \prime } Y E S ^ { \prime } } ) ) + \exp ( \log \mathrm { i t } ( \mathbf { \dot { \prime } N O ^ { \prime } } ) ) } }
$$

After evaluation, we select the instruction with the highest score $x _ { i } ^ { * * }$ for each response in $\{ y _ { i } ^ { * } \}$ to obtain a new training dataset $\left\{ ( x _ { i } ^ { * * } , y _ { i } ^ { * } ) \right\}$ .

# 5 Implementation Details

The original instruction tuning dataset. In this work, we mainly use evol-codealpaca-v1 as our original instruction tuning dataset $\{ ( x _ { i } , y _ { i } ) \}$ , which is widely used for instruction tuning of code LLMs (Wei et al. 2023; Yu et al. 2023; Song et al. 2024). It contains 111183 instruction-response pairs generated by Evol-Instruct using GPT-4. Following Magicoder (Wei et al. 2023), evol-codealpaca-v1 is decontaminated by removing data that contain docstrings or solutions from HumanEval (Chen et al. 2021), MBPP (Austin et al. 2021), MultiPL-E (Cassano et al. 2023), and DS-1000 (Lai et al. 2023), which are used to evaluate InverseCoder. We apply the same decontamination method to the newly generated instruction data $\{ ( x _ { i } ^ { * * } , y _ { i } ^ { * } ) \}$ .

Training for original Code LLM. We take CodeLlamaPython-13B, CodeLlama-Python-7B (Rozi\`ere et al. 2023) and DeepSeek-Coder-Base-6.7B (Guo et al. 2024) as base models. To obtain the beginning code LLM $M$ (hereinafter called WizardCoder-GPT4), we fine-tune the base models on evol-codealpaca-v1 for 2 epochs using 8 NVIDIA

A100-40GB SMX GPUs. We set the initial learning rate at $5 e \mathrm { ~ - ~ } 5$ with 15 warmup steps and a linear learning rate scheduler. We use Adafactor (Shazeer and Stern 2018) as our optimizer and choose a batch size of 512 with a sequence truncation length of 1024.

Instruction data collection. We use the vLLM inference framework (Kwon et al. 2023) for code summarization and instruction selection on the same GPUs as training. We generate 10 instructions $\{ x _ { i j } ^ { * } \} _ { j = 1 } ^ { 1 0 }$ for each code snippet in the code summarization stage. For each instruction-response pair, the self-evaluation and data selection process is conducted by prompting the beginning code LLM $M$ with greedy decoding. We choose the instruction with the highest pseudo-probability of YES as the best-generated instruction for each response.

Training for InverseCoder. Following Magicoder (Wei et al. 2023), we first fine-tune the base models on the new dataset $\{ ( x _ { i } ^ { * * } , y _ { i } ^ { * } ) \}$ with 90363 instruction-response pairs (generated by the original Code LLM $M$ ) for 1 epoch, then we continue to fine-tune the models with the original dataset $\{ ( x _ { i } , y _ { i } ) \}$ (generated by GPT-4) for 2 epochs to obtain InverseCoder. The hyperparameters are the same as the training process for the original code LLM $M$ . The instruction tuning prompt is aligned with Magicoder .

# 6 Experiments

We conduct a series of experiments to investigate these topics:

1. InverseCoder’s performance on benchmarks (Sec. 6.1).   
2. Impact of each stage in Inverse-Instruct (Sec. 6.2).   
3. Impact of dataset size scaling (Sec. 6.3).   
4. Is Inverse-Instruct effective on other datasets (Sec. 6.4)?   
5. Comparison with other data selection methods (Sec. 6.5).   
6. Does selecting multiple self-generated instructions for each response lead to further improvement (Sec. 6.6)?   
7. Can Inverse-Instruct be repeatedly applied to InverseCoder to achieve multi-round optimization (Sec. 6.7)?   
8. Can Inverse-Instruct be further optimized by using additional self-generated code as responses (Sec. 6.8)?

# 6.1 Main Results

We train InverseCoder on three base models with different parameter sizes and evaluate them on four benchmarks widely used for code LLMs, including Python text-to-code generation, multilingual coding, and data-science code generation. The results show that the performance of SOTA code LLMs can continue to improve by Inverse-Instruct.

Baselines. We compare the performance of our models with a wide range of baselines including:

1. Base Models: Three base models mentioned in Section 5. We compare InverseCoder with them to show the absolute improvement of the whole instruction-tuning process.   
2. WizardCoder-GPT4: The beginning code LLMs in our data generation process, which are only trained by the original instruction-tuning dataset (i.e., evol-codealpaca-v1). We compared InverseCoder

Table 2: Training data size of different instruction-tuned code LLMs. It is worth noting that only InverseCoder is trained by self-generated data, which is easier to obtain at a lower cost.   

<html><body><table><tr><td>Model</td><td>Common Data|Specific Data</td><td></td></tr><tr><td>WizardCoder-GPT-4 MagicoderS WaveCoder-Ultra</td><td>110KGPT-4</td><td>OK (baseline) 75K GPT-3.5 20K GPT-4</td></tr><tr><td>AlchemistCoder</td><td></td><td>> 80K GPT-3.5</td></tr><tr><td>InverseCoder (ours)</td><td></td><td>90K self-generated</td></tr></table></body></html>

Table 3: Pass ${ \ @ \mathbf { 1 } }$ $( \% )$ results of different LLMs on HumanEval $( + )$ and MBPP $( + )$ computed with greedy decoding. The abbreviations “CL” and “DS” refer to the base models CodeLlama-Python and DeepSeek-Coder, respectively. We report other results consistently from the EvalPlus (Liu et al. 2023) Leaderboard in August 2024 and Magicoder (Wei et al. 2023) paper.   

<html><body><table><tr><td>Model HumanEval (+） MBPP(+)</td></tr><tr><td>(Closed-sourceModels) GPT-4-Turbo (April 2024) 90.2 (86.6) 85.7 (73.3) GPT-3.5-Turbo (Nov 2023) 76.8(70.7) 82.5 (69.7)</td></tr><tr><td>(Basedon CodeLlama-Python-13B) CodeLlama-Python-13B 42.7 (38.4) 63.5 (52.6) WizardCoder-GPT4-CL-13B 76.8(70.7) 73.5 (62.2) InverseCoder-CL-13B (ours) 79.9 (74.4) 74.6 (63.0)</td></tr><tr><td>(Based on CodeLlama-Python-7B) CodeLlama-Python-7B 37.8 (35.4) 59.5 (46.8) MagicoderS-CL-7B 70.7 (67.7) 70.6 (60.1) AlchemistCoder-CL-7B 74.4 (68.3) 68.5 (55.1)</td></tr><tr><td>WizardCoder-GPT4-CL-7B 72.6 (68.9) 69.3 (59.3) InverseCoder-CL-7B (ours) 76.2 (72.0) 70.6 (60.1) (BasedonDeepSeek-Coder-6.7B) 72.0 (58.7)</td></tr><tr><td>DeepSeek-Coder-6.7B 47.6 (39.6) MagicoderS-DS-6.7B 76.8 (71.3) 79.4 (69.0) WaveCoder-Ultra-DS-6.7B 75.0 (69.5) 74.9 (63.5) AlchemistCoder-DS-6.7B 79.9 (75.6) 77.0 (60.2) WizardCoder-GPT4-DS-6.7B 77.4 (73.2) 77.8 (67.5) InverseCoder-DS-6.7B (ours) 79.9 (76.8) 78.6 (69.0)</td></tr></table></body></html>

with them to show the improvement brought by InverseInstruct.

3. Other Open Source Instruction-Tuned Code LLMs: Instruction-tuned code models in related works, including Magicoder (Wei et al. 2023), WaveCoder-UltraDS (Yu et al. 2023) and AlchemistCoder (Song et al. 2024). They are trained on additional data generated by stronger closed-source LLMs (e.g., GPT-3.5) in addition to evol-codealpaca-v1.

The comparison of training data size is shown in Table 2. The actual data consumption of InverseCoder should be mainly measured by the scale of the original training dataset (110K) since the cost of self-generating data is much lower than generating data by querying closedsource LLMs (Irugalbandara et al. 2023).

Table 4: Pass@1 $( \% )$ results of different LLMs on MultiPL-E. The models marked with $( ^ { * } )$ are evaluated with the same prompt format as training and the same hyperparameter as Magicoder. We report other results consistently from Magicoder paper.   

<html><body><table><tr><td>Model Java JS C++ PHP Swift Rust Avg.</td></tr><tr><td>(BasedonCodeLlama-Python-13B) WizardCoder-GPT4* 55.4 64.2 55.9 52.0 49.9 53.4 55.1</td></tr><tr><td>InverseCoder (ours)* 54.5 65.4 58.1 55.3 52.5 55.6 56.9</td></tr><tr><td>(BasedonCodeLlama-Python-7B) CodeLlama-Python 29.1 35.7 30.2 29.0 27.1 27.0 29.7 53.3 44.9 43.8 50.8</td></tr><tr><td>MagicoderS * 49.8 62.6 50.2 WizardCoder-GPT4* 50.4 60.7 50.6 51.6 45.6 48.2 51.2 InverseCoder (ours)* 48.7 61.9 52.6 55.2 53.0 46.1 52.9</td></tr><tr><td>(BasedonDeepSeek-Coder-6.7B)</td></tr><tr><td>MagicoderS * 59.6 69.8 70.0 64.4 54.4 53.6 62.0</td></tr><tr><td>WizardCoder-GPT4* 61.4 66.4 68.7 61.8 52.6 56.1 61.2 InverseCoder (ours)* 60.7 70.1 70.5 63.6 53.0 57.4 62.6</td></tr></table></body></html>

4. Closed-source LLMs: GPT-3.5 (OpenAI 2022) and GPT4 (OpenAI 2023) to show the gap between InverseCoder with the advanced closed-source LLMs.

Inverse-Instruct improves general Python code generation capabilities. We use HumanEval $\left( + \right)$ and $\mathbf { M B P P ( + ) }$ (Liu et al. 2023), the enhanced versions of two Python code generation benchmarks (Chen et al. 2021; Austin et al. 2021), to evaluate the text-to-code capability of InverseCoder. Each benchmark provides a set of tasks with natural language descriptions as prompts for the code LLM to generate functionlevel code, which is then validated using pre-prepared test cases.

We use the pass $\ @ 1$ (Chen et al. 2021) score to compare the code generation capability among different models. The results are shown in Table 3, which demonstrate that InverseCoder makes a significant improvement over WizardCoderGPT4 in Python code generation capability.

The improvement of Inverse-Instruct is reflected across multiple programming languages. Besides Python, we evaluate the code generation capabilities of other six mainstream programming languages for InverseCoder on MultiPLE benchmark (Cassano et al. 2023).

Table 4 shows the performances of InverseCoder and other models on MultiPL-E. The results reveal that the capabilities of InverseCoder to generate code in different programming languages are improved over WizardCoder-GPT4.

Inverse-Instruct also leads to enhancement in data science code generation tasks. To show the capability of InverseCoder for complex programming problems in realistic applications, we evaluate it on DS-1000 benchmark (Lai et al. 2023), which comprises 1000 different data science workflows across seven libraries. Following Wei et al. (2023), we evaluate our model only on the completion mode.

Table 5: Pass $@$ 1 $( \% )$ results on DS-1000 including seven data science libraries: Matplotlib (plt.), Numpy (np.), Pandas (pd.), Pytorch, Scipy, Sklearn and Tensorflow (tf.). We evaluate our models in the same prompt and hyperparameters as Magicoder. We report other results from Magicoder paper.   

<html><body><table><tr><td>Model plt. np． pd. torch scipy sklearn tf. All</td></tr><tr><td>(BasedonCodeLlama-Python-13B) WizardCoder-GPT456.152.230.343.0 25.2 49.5 40.042.1</td></tr><tr><td>InverseCoder (0urs) 53.0 54.3 32.1 50.9 22.5 50.5 43.8 43.1 (BasedonCodeLlama-Python-7B)</td></tr><tr><td>CodeLlama-Python 55.3 34.5 16.4 19.9 22.3 17.6 28.5 28.0</td></tr><tr><td>WizardCoder 53.534.415.2 25.7 21.0 24.5 28.9 28.4 MagicoderS 55.940.6 28.4 40.4 28.8 35.8 37.6 37.5</td></tr><tr><td>WizardCoder-GPT451.546.929.9 43.6 34.9 41.9 39.0 40.2 InverseCoder(ours) 54.2 48.6 27.4 38.0 34.0 41.9 40.3 39.9</td></tr><tr><td></td></tr><tr><td>(BasedonDeepSeek-Coder-6.7B) MagicoderS 54.848.930.0 49.2 27.3 44.7 41.2 41.2</td></tr><tr><td>WizardCoder-GPT4 53.8 53.928.0 49.3 30.4 45.7 44.4 42.2</td></tr><tr><td>InverseCoder (0urs) 55.5 53.9 32.3 56.7 30.0 50.3 33.9 44.2</td></tr><tr><td></td></tr></table></body></html>

Table 6: Pass $\ @ \mathbf { 1 }$ $( \% )$ results on HumanEval+ and $\mathbf { M B P P + }$ in ablation studies. Preprocessing (Pre.), Summarization (Sum.) and Evaluation (Eval.) correspond to the three steps in our method. Generation (Gen.) represents regenerate responses for each instruction.   

<html><body><table><tr><td>Method</td><td>HumanEval(+) MBPP(+)</td></tr><tr><td>Gen.+Eval. 70.7 (67.1)</td><td>70.9 (60.1)</td></tr><tr><td>Pre.</td><td>72.6 (68.9) 69.8 (59.8)</td></tr><tr><td>Pre.+ Sum. 75.6 (71.3)</td><td>68.0 (58.2)</td></tr><tr><td>Pre.+ Sum.+Eval.(ours)</td><td>76.2 (72.0) 70.6 (60.1)</td></tr></table></body></html>

The results in Table 5 show that the average performances of InverseCoder-CL-13B and InverseCoder-DS-6.7B in the data science code generation tasks are enhanced, which implies that Inverse-Instruct can help to improve the code generation capability of the original model in realistic tasks beyond basic programming problems.

# 6.2 Ablation Study

We conduct a series of ablation experiments to analyze the utility of code summarization and data selection steps in our method. We use CodeLlama-Python-7B as the base model in the following experiments with the same training settings as InverseCoder and present the results in Table 6. The ablation experiments are in three aspects:

Inverse-Instruct outperforms the NL-to-Code data generation method $( \mathbf { G e n . + E v a l . } )$ ). We regenerate 10 responses $\{ y _ { i j } \} _ { j = 1 } ^ { 1 0 }$ j}j10=1 for each instruction xi in the original training dataset and apply the same self-evaluation method to select the best responses. It shows that the code summarization step provides overall better performance than generating responses from instructions.

![](images/fe5a99d223a8869742a904ffdabf5a3dc17e55a47f8e4396f06b8b8c25ff3200.jpg)  
Figure 2: Impact of data scaling. The dashed line represents HumanEval and the solid line represents HumanEval+. Legend “Original” and “Ours” represent the original models and the models improved by Inverse-Instruct.

Table 7: Performance improvement of Inverse-Instruct when applied to Magicoder-OSS-Instruct-75K.   

<html><body><table><tr><td>Model</td><td>HumanEval (+)</td><td>MBPP (+)</td></tr><tr><td>Magicoder-DS</td><td>66.5 (60.4)</td><td>75.4 (61.9)</td></tr><tr><td>InverseCoder-DS-OSS</td><td>69.5 (64.0)</td><td>77.0 (66.1)</td></tr></table></body></html>

Performance improvement comes not only from the preprocessing step (Pre.). We only apply preprocessing to the responses in the original dataset $\{ ( \bar { x } _ { i } , \bar { y } _ { i } ) \bar  \}$ to obtain a cleaned dataset $\{ ( x _ { i } , y _ { i } ^ { * } ) \}$ . We train the models with the cleaned dataset and the original one to show the improvement from preprocessing is minor.

The self-evaluation and data selection step also plays a role in Inverse-Instruct $( \mathbf { P r e . } + \mathbf { S u m . }$ ). To study the role of self-evaluation and data selection, we generate only one instruction for each response in the code summarization step without any selection. The results show that self-evaluation and selection are also helpful to performance improvement.

# 6.3 Data Scaling

Inverse-Instruct is effective across different data scales. We conduct a series of experiments to explore the data scaling law of Inverse-Instruct. Specifically, we randomly select 25K, 50K, and 75K instruction-response pairs from the original dataset and train 3 weaker original models with them. Then, we apply Inverse-Instruct for the original models. It is shown that the performances of the models are all improved by Inverse-Instruct at different scales of data (Figure 2).

# 6.4 Impact of Original Dataset

Inverse-Instruct is effective across different original datasets. We apply Inverse-Instruct to Magicoder-OSS

Table 8: Comparison of our data selection method with alternatives (for CL-7B).   

<html><body><table><tr><td>Data-Selection Method</td><td>HumanEval (+)</td></tr><tr><td>Random Selection</td><td>72.6 (68.3)</td></tr><tr><td>Textual Score</td><td>73.8 (69.5)</td></tr><tr><td>Lowest Perplexity</td><td>70.1 (67.7)</td></tr><tr><td>HighestPerplexity</td><td>70.7 (67.7)</td></tr><tr><td>YES Pseudo-probability (ours)</td><td>76.2 (72.0)</td></tr></table></body></html>

Table 9: Performance comparison of the models (CL-7B) trained with different numbers of selected instructions. “Top-k” means that for each response, we select the instructions with the top k highest pseudo-probability.   

<html><body><table><tr><td>Selected Instructions</td><td>HumanEval (+)</td><td>MBPP (+)</td></tr><tr><td>Top-1 (ours)</td><td>76.2 (72.0)</td><td>70.6 (60.1)</td></tr><tr><td>Top-3</td><td>70.1 (67.1)</td><td>68.0 (58.5)</td></tr><tr><td>Top-5</td><td>70.1 (65.2)</td><td>61.9 (53.4)</td></tr></table></body></html>

Instruct-75K (Wei et al. 2023), a smaller dataset generated by GPT-3.5. The results (Table 7) show that performance is still improved even with a smaller and lower-quality original dataset, demonstrating the robustness of Inverse-Instruct.

# 6.5 Alternative Data Selection Methods

Our data selection method outperforms alternatives. We compare our data selection method which is based on the pseudo-probability of YES with the three alternatives:

1. Randomly selecting one instruction from all synthetic candidates corresponding to each response.   
2. Using textual format scores (1-5) provided by the LLM itself as an indicator. If no textual score is given, assign a default score of 3.   
3. Using the sentence perplexity of the response code under different instructions as an indicator. We select the data with the highest and lowest perplexity respectively.

The results are shown in Table 8, demonstrating the pseudoprobability method’s efficiency.

# 6.6 Selecting Multiple Self-Generated Instructions

Selecting multiple self-generated instructions for a single response will harm the model’s performance. We select the top- $\mathbf { \nabla } \cdot \mathbf { k }$ scoring instructions for each response. The results in Table 9 indicate that the model’s performance declines as the number of selected instructions increases. This suggests that open-source code LLMs are not capable of generating a large number of correct instructions, which is why we only select the best instructions in our method.

6.7 Multi-Round Optimization for InverseCoder Repeatedly applying Inverse-Instruct to InverseCoder does not significantly improve performance. We replace the original model with InverseCoder in the pipeline of

Table 10: Performance diffenernce when applying InverseInstruct to InverseCoder again. “V2” means models trained with the data generated by InverseCoder.   

<html><body><table><tr><td>Model</td><td>HumanEval (+)</td><td>MBPP(+)</td></tr><tr><td>InverseCoder-CL-7B</td><td>76.2 (72.0)</td><td>70.6 (60.1)</td></tr><tr><td>InverseCoder-CL-7B-V2</td><td>75.0 (70.1)</td><td>70.6 (60.6)</td></tr></table></body></html>

Table 11: Comparison of Inverse-Instruct with other alternative data generation methods which prompt the original model to generate additional code (for CL-7B).   

<html><body><table><tr><td colspan="3">Data-Generation Method HumanEval (+) MBPP (+)</td></tr><tr><td>Code → NL (ours)</td><td>76.2 (72.0)</td><td>70.6 (60.1)</td></tr><tr><td>Code→NL →Code</td><td>73.2 (68.9)</td><td>67.7 (57.7)</td></tr><tr><td>Code→Code→NL</td><td>73.2 (68.3)</td><td>70.9 (62.2)</td></tr></table></body></html>

Inverse-Instruct and train a new model with the data generated by InverseCoder. The performance results (Table 10) show no significant improvement, which confirms the phenomenon of model collapse caused by repeatedly training on self-generated data (Shumailov et al. 2024).

# 6.8 Training with Additional Self-Generated Code

Performance cannot be steadily improved when the model is trained with both self-generated instructions and code. We conduct the following two experiments to examine whether training with the code generated by the original model provides additional benefits.

1. Code $\bf { \tau }  N L  C o d e$ : Regenerating new response code for the new instructions obtained by Inverse-Instruct. 2. Code $ \mathbf { C o d e }  \mathbf { N L }$ : Prompting the original model to give more complex code and applying Inverse-Instruct to the new code.

The results are shown in Table 11. Unstable performance reveals issues with the quality of the self-generated code of original models.

# 7 Conclusion

In conclusion, this paper presents a novel approach to enhancing the capabilities of open-source code LLMs by leveraging self-generated data for instruction tuning, rather than relying solely on data from powerful closed-source LLMs like GPT-3.5 and GPT-4. Our proposed method, named InverseInstruct, capitalizes on the inherent asymmetry in translating between formal and informal languages. By reversing the conventional process, Inverse-Instruct generates additional natural language instructions from code snippets via summarization and self-evaluation techniques. The effectiveness of this methodology is demonstrated through the development of InverseCoder, a new series of code LLMs that not only outperform their predecessors in traditional benchmarks but also show significant improvement across diverse coding tasks.