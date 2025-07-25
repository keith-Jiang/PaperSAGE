# Chain-of-Instructions: Compositional Instruction Tuning on Large Language Models

Shirley Anugrah Hayati\*1, Taehee $\mathbf { J u n g } ^ { 2 }$ , Tristan Bodding-Long2 Sudipta $\mathbf { K a r } ^ { 2 }$ , Abhinav Sethy†3, Joo-Kyung $\mathbf { K i m } ^ { 2 }$ , Dongyeop Kang

1University of Minnesota 2Amazon 3Grammarly hayat023@umn.edu, dongyeop@umn.edu

# Abstract

Fine-tuning large language models (LLMs) with a collection of large and diverse instructions has improved the model’s generalization to different tasks, even for unseen tasks. However, most existing instruction datasets include only single instructions, and they struggle to follow complex instructions composed of multiple subtasks. In this work, we propose a novel concept of compositional instructions called chain-ofinstructions (CoI), where the output of one instruction becomes an input for the next like a chain. Unlike the conventional practice of solving single instruction tasks, our proposed method encourages a model to solve each subtask step by step until the final answer is reached. CoI-tuning (i.e., finetuning with CoI instructions) improves the model’s ability to handle instructions composed of multiple subtasks as well as unseen composite tasks such as multilingual summarization. Overall, our study finds that simple CoI tuning of existing instruction data can provide consistent generalization to solve more complex, unseen, and longer chains of instructions.

Code and Datasets — {https://github.com/amazonscience/chain-of-instructions}

# Introduction

Large language models (LLMs) have demonstrated impressive performance in various tasks, from conventional NLP downstream tasks, such as machine translation and summarization, to open-ended tasks, such as writing an outline for blog posts and giving tips for presentation, when fine-tuned on human-like instructions (Ouyang et al. 2022; Wang et al. 2022; Conover et al. 2023; Mishra et al. 2022). These models excel at single instruction tasks, but their ability to handle complex and compositional instructions is less explored.

A compositional instruction contains a series of sequential subtasks, as the output of one subtask becomes the input of the next one in a chained manner as shown in Figure 1. We call this problem as Chain-of-Instructions or shortly CoI.

![](images/45c973739fe76bd2af71e93f05d69c7f55bd0a8c7c9bbf70656db827cc722363.jpg)  
Figure 1: Chain-of-Instructions (CoI) example. The summarization output can be an input for a title generation subtask; the output of the title generation can be an input for style transfer or translation subtasks. Arrow thickness denotes the probability of instruction composability. X means that these subtasks cannot be composed due to format mismatch. $I _ { k }$ is $k ^ { t h }$ instruction and $O _ { k }$ is $k ^ { t h }$ output.

We examine what subtasks can be composed more naturally than others.

In Figure 1, we can see that some tasks can be composed together, such as input a summary to a title generation task, while some tasks cannot be composed together, e.g., a summary as input for a Data-to-Text task. Some tasks have a higher probability of being able to be composed, such as generating a title from a summary compared to converting numbers in a summary since sometimes a summary does not contain a number. Figure 2 illustrates a more detailed example of CoI. The given instruction “Generate a blog-like title in French” can be decomposed into three chained subinstructions:

1. Generate a title for the given text   
2. Convert the style of the title to be similar to a blog post title   
3. Translate the blog post title into French

When these sub-instructions are composed, we call it a compositional instruction or chain-of-instructions. Our study investigates whether LLMs can handle compositional

<html><body><table><tr><td></td><td>Instruction</td><td>Composed</td><td>Data Size</td><td>Domain</td></tr><tr><td>Chain-of-Instructions (Ours)</td><td>√</td><td>√</td><td>18k</td><td>NLP tasks</td></tr><tr><td>Self-Instruct (Wang et al. 2023)</td><td>√</td><td>X</td><td>52k</td><td>Daily QA-like tasks</td></tr><tr><td>Dolly (Conover et al. 2023)</td><td>√</td><td>X</td><td>15k</td><td>Daily QA-like tasks</td></tr><tr><td>Super-NaturalInstruct(Wang et al. 2022)</td><td>√</td><td>X</td><td>1.6k</td><td>NLP tasks</td></tr><tr><td>Faith and Fate (Dziri et al. 2023)</td><td>×</td><td>√</td><td>N/A</td><td>Math, logic, programming</td></tr><tr><td>Compositional Semantic (Drozdov et al. 2022)</td><td>×</td><td>√</td><td>N/A</td><td>CFQ, COGS, Parsing</td></tr><tr><td>MuSiQue(Trivedi et al. 2022)</td><td>X</td><td>√</td><td>24.8k</td><td>Multi-hop QA</td></tr></table></body></html>

TableIn1p:u A compCaroiIson of oOuurtpuwtork with existing related works. As some previous works do not contribute a new dataset, the dataset size is shown as N/A. For instruction daDtatsaetso, Tdeaxtta size refers to the number of instructions, not task instances (inputoutputI1pair).O1MoreI2priorO2workI3is studied in $\ S$ .

Chain-of-Instructions (CoI) Instruction: Generate a blog-like title in French Input: Ratatouille is a traditional stew made with summer vegetables. When cooking Ratatouille, you can imagine picking up fresh vegetables from your yard and ready to embrace Fall. Subtask Instructions [I1] Instruction 1: [O1] Output 1: Ratatouille: Generate a title A Taste of Summer's Farewell [I2] Instruction 2: [O2] Output 2: Delight in Convert into a blog-style Ratatouille: Savor the Flavors title of a Summer Farewell [I3] Instruction 3: [avOec3 ]laORuataptoutil3le: :DSéalvecotuerze-zvloeus Translate into French Saveurs d'un Adieu à l'Été

instructions effectively and whether models tuned with compositional instructions can be generalized to solve more complex, unseen, or longer chains of instructions. We first create a new CoI dataset with our proposed LLM-based compositionality checker, and then evaluate our model’s performance in handling (1) traditional single instructions and (2) compositional instructions.

Our work is closely related to other instruction-tuning works and compositional studies in NLP, as summarized in Table 1. Wang et al. (2023); Conover et al. (2023); Wang et al. (2022) propose new instruction datasets, but they only handle single instruction problems. Although our approach draws inspiration from Chain-of-Thought (CoT) prompting (Wei et al. 2022b) or Least-to-Most prompting (Zhou et al. 2022), our CoI is not a prompting technique but a collection of chained instructions validated by an LLM, showing generalization in solving complex and compositional problems. Our contributions are as follows:

• We develop a framework to automatically construct composed instruction datasets with minimal human supervision. The framework leverages in-context learning on existing single-instruction datasets to create CoIs. • We propose a method for enabling LLMs to solve compositional tasks in an explainable way. As an example, a model can generate incremental outputs at each step of a complex task chain. With CoI-tuning, step-by-step instruction following becomes easier, especially when dealing with instructions composed of multiple subtasks. • We demonstrate through experiments and analysis that the CoI-tuned model outperforms both individual instructions and sequential compositional instructions. By training on CoI data, the model achieves higher performance. This result also generalizes for unseen longer chain test sets and downstream tasks.

• We introduce a novel task called Chain-of-Instructions (CoI) to examine LLMs’ capabilities in following compositional instructions by creating a new benchmark dataset.

# Chain-of-Instructions

# Formulation

Compositional instructions contain multiple subtask instructions where the output from one subtask becomes the input for the next subtask similarlity to a composition function in math. Thus, we formalize the problem of chain-ofinstructions as follows:

Definition 1 (Chain of Instructions). Given a tuple of <instruction $I$ , input $X$ , output $Y >$ , let $I ( X ) = { \bar { Y } }$ refer that an LLM generates output $Y$ with instruction $I$ and input $X$ . A sequence of instructions $\{ I _ { 1 } , . . . , I _ { k } \}$ is a chain of instructions with length $k$ if $I _ { i + 1 } \circ I _ { i } ( X _ { i } ) \ : = \ : Y _ { i + 1 }$ , for all $i \in \{ \mathbb { N } : 1 \leq i \leq k \}$ .

# Automatic Dataset Creation Pipeline

Seed Datatsets We curate a new compositional instruction dataset from existing single task instruction dataset: SUPER-NATURALINSTRUCTIONS (SUP-NATINS) (Wang et al. 2022). We select SUP-NATINS as the seed dataset because it contains a wide variety of tasks (1,616 unique tasks) from 76 categories, including text categorization, summarization, and machine translation. Each category contains many different NLP tasks. For example, under the text categorization category, there exist sarcasm detection and politeness classification tasks. Each task in SUP-NATINS contains human-written descriptions that can be considered as instructions and instances as pairs of inputs and outputs. We only select tasks with English (1,341 unique tasks) as their input language to make sure that the chain is connected. For our single-task instruction tuning data $\begin{array} { r } { ( \mathbf { C o l } _ { 1 } ) } \end{array}$ ), we randomly sample 10 input-output pairs, resulting in 13,410 instances.

Instruction Composition Composing two single instructions poses a challenge due to their lengthy and specific descriptions, and differing output formats. Figure 3 illustrates a two-step process for creating a compositional instruction dataset with the help of an LLM as elaborated in the following paragraphs. Here we use GPT 3.5 Turbo (Ouyang et al. 2022) because of its reasonable price and at the time, we examine that the quality of the result is good enough. However, this data creation procedure can be reproducible with other strong LLMs as well.

Step 1: Single instruction summarization The task instructions in SUP-NATINS are lengthy and detailed, which may deviate from real human-like instructions. With the same dataset (SUP-NATINS), Yin et al. (2023) find that $60 \%$ tokens can be removed with comparable, if not better, model performance. Thus, we use the LLM to shorten each instruction in the SUP-NATINS dataset. This step reduces the average number of words in the SUP-NATINS descriptions from 62.19 to 14.33.

Step 2: Composability check To generate compositional instructions from single instructions, we perform a two-step process: (1) validity check and (2) generate the output for the second (or third) subtask. The validity check is performed to examine whether two subtasks are composable. We first filter out non-composable tasks with heuristics developed by the authors’ knowledge (the Heuristics for Validity Check section in the Appendix). For example, classification tasks can only be the last subtask when composing a pair of tasks. After applying these heuristics, we additionally check whether LLM can generate the output for the second instruction based on the input of the first instruction. If so, we treat the pair as composable.1

For the pairs that pass the validity check, we generate the new output using the first output and second instruction for the second task. This generated output serves as the ground truth for the second subtask in the instruction-tuning phase. Our approach is a variation of distillation from a larger LLM as has been done by previous works for different problems (Gu et al. 2024; Hsieh et al. 2023; West et al. 2022). We define compositional instructions originating from two instructions as $\mathrm { C o I _ { 2 } }$ and those originating from three instructions $\mathrm { C o I _ { 3 } }$ . $\mathrm { C o I _ { 3 } }$ is created by chaining two $\mathrm { C o I _ { 2 } s }$ if there exists $\mathrm { I } _ { x } \circ \mathrm { I } _ { y }$ and $\boldsymbol { \mathrm { I } } _ { y ^ { \circ } } \boldsymbol { \mathrm { I } } _ { z }$ , resulting in $\mathrm { C o I _ { 3 } } = \mathrm { I } _ { x } \circ \mathrm { I } _ { y } \circ \mathrm { I } _ { z }$ . The same method is applied for creating longer chains such as $\mathrm { C o I _ { 4 } }$ and $\mathrm { C o I _ { 5 } }$ .

To examine the quality of LLM’s composability check, we randomly sampled 100 instances and manually inspected which composed instructions are valid. We find that $7 5 \%$ are valid composed instructions. For $\mathrm { C o I _ { 3 } }$ , similarly we randomly sampled 100 instances and found that $5 9 \%$ are valid compositions. Such error rates are often found in LLMgenerated data (Das et al. 2024; Wang et al. 2023).

Table 2: Dataset statistics per chain length.   

<html><body><table><tr><td>chain length (o)</td><td>train</td><td>test</td></tr><tr><td>1</td><td>13,410</td><td></td></tr><tr><td>2</td><td>2,993</td><td>588</td></tr><tr><td>3</td><td>2,187</td><td>480</td></tr><tr><td>4</td><td></td><td>844</td></tr><tr><td>5</td><td></td><td>355</td></tr></table></body></html>

# CoI Dataset

Table 2 shows the data statistics of CoI datasets. In chain length 2, we obtain 970 unique category pairs; in chain length 3, we obtain 418 unique category triplets. In each pair or triplet, we randomly select at most three instances and divide them into training and testing sets. For the longer chains (4, 5), we only use them for testing. Please find Appendix ?? for the detailed statistics.

Figure 4 shows a t-SNE plot when we embed subtask instructions of frequent $\mathrm { C o I _ { 2 } }$ instructions using SentenceBERT (Reimers and Gurevych 2019) with DistilRoberta (Sanh et al. 2019).2 We find generation tasks such as paraphrasing and question generation can be compiled as both the first and second subtasks, except for problems involving specific input formats, such as code to text or data to text, which can only be compiled as the second subtask. On the other hand, close-ended problems (e.g., POS tagging or grammar error detection) mostly appear as the second subtask.

# Experiment Setup

CoI models We fine-tune the base models of Alpaca7B (Taori et al. 2023) and Mistral-7B-Instruct (Jiang et al. 2023). Since both models are open-sourced single instruction-tuned models which are widely used, they are suitable to be compared with CoI-tuned models.

# Baselines

• Off-the-shelf version of Alpaca-7B (Taori et al. 2023) model and Mistral-7B-Instruct model without fine-tuning (Base).   
• The same non-finetuned Alpaca and Mistral with chainof-thought prompting (Wei et al. 2022b) (CoT) with seven-shot demonstrations and least-to-most prompting (Zhou et al. 2022) (LtM).   
• Fine-tuned base models with a subset of singleinstruction SUP-NATINS dataset $( \mathrm { C o I _ { 1 } } )$ .

Metrics For our evaluation metric, we report ROUGE-L (Lin 2004), following Wang et al. (2022) and LLM (gpt-4omini) as a preference judge. ROUGE can be used to assess various text generation tasks and using LLM as a judge has

Input 1 Instruction 1 Output 1 SUP-NATINS Input 2 Instruction 2 Output 2 ChainO-uorf-DIantsatsreutc:tions Summarized CoI Dataset Instruction 2 Instruction 1 Instruction 2 Output 1 S ? Summarized S Validity Check Input 1 Instruction 1 Output 1 Summarize STOP S Gener✅ate Summarized Generated Output 2 ISnustmrumcatiriozned1 ISnustmrumcatiriozned2 Generated Output 2

Figure 3: Data creation for $\mathrm { C o I _ { 2 } }$ . We use an LLM for both instruction summarization and composability check. The right column shows an example instance of our chain-of-instruction dataset. Output 1 in Step 2 comes from the original SUPNATINST data.

Code to Text 80 Question Generation Data to Text 60 Answer Story Composition ? Verification 40 Text to Code Sentence   
Information Extraction Title erturbation   
Paraphrasin Generation Text Quality Question 0 g Text CompletionKeyword Tagging Rewriting   
ummarization Sentence -20 Coherence ExCploamnpaotisiotinon Classification -40 -60 POS Tagging Grammar Error Detection -125 -100 -75 -50 -25 0 25 50 75 first subtask instruction second subtask instruction

• Downstream Task In addition to CoI test sets, we examine the usefulness of CoI on the downstream task of multilingual summarization using WikiLingua (Ladhak et al. 2020), which is a multilingual dataset based on WikiHow3 for abstractive summarization in 18 languages. WikiHow articles provide step-by-step instructions to complete procedural tasks on different topics, and each step includes a one-sentence summary as well. In our experiment, we select source-target language pairs ; English-to-French (WikiLingua-en- ${ \bf \nabla } \cdot f r$ ) and Spanish-toEnglish (WikiLingua-es-en) and randomly sample 300 test instances for each. Given an input content from source language $L _ { s r c }$ , we aim to generate a summary in target language $\boldsymbol { L } _ { t g t }$ . This task is similar to a 2- instruction problem as we summarize first and then translate. Note that CoI training data only contains translation tasks from English to Punjabi, German, and Catalan, thus, selected source-target pairs are unseen in CoI training set.

been widely adopted in NLP research (Liu et al. 2023; Fu et al. 2024). We also have human evaluation to perform blind pairwise comparison between the outputs from the baseline and from our best CoI models.

Test sets To assess the compositionality of our models, we prepare three types of evaluation suites.

• CoI Test set For the compositional instruction evaluation, we tested the models on CoI test sets with $\sigma =$ $\{ 2 , 3 , 4 , 5 \}$ where $\sigma$ is a chain length. • BIG-Bench Hard For the single instruction test set, we use BIG-Bench Hard (Suzgun et al. 2022), a collection of 27 challenging tasks, such as date understanding and evaluating the truth value of Boolean expressions, and each task has $\leq 2 5 0$ instances. BIG-Bench Hard subset enables us to evaluate the model’s performance on diverse and challenging NLP tasks with clear single instructions and associated input-output pairs.

# Results

We conduct experiments to measure the performance of CoItuned models on our compositional instructions $( \ S )$ , and the generalization capability to difficult single instructions $( \ S )$ , and longer-chain instructions $\sigma = \{ 4 , 5 \bar { \} }$ (§), and the application to an existing downstream task (§). We also conduct an ablation study to see if the correctness of second and third subtask outputs matter in Appendix. We see degrading performance of models fine-tuned with incorrect outputs, showing the importance to have the correct output for the subtasks during training.

# Performance on In-domain Composite Tasks $\mathbf { ( C o I _ { 2 , 3 } ) }$

Automatic metric As we evaluate our CoI models’ performance against the baselines on multi-CoI test sets, we find

Table 3: ROUGE-L results on intermediate tasks. CoI models refer to best models of CoI: $\mathrm { C o I _ { 1 2 } }$ model if the test $\mathrm { s e t { = } C o I _ { 2 } }$ and $\mathrm { C o I _ { 1 2 3 } }$ model if the test $\mathrm { \ s e t { = } C o I _ { 3 } }$ .   

<html><body><table><tr><td rowspan="2"></td><td colspan="2">Mistral</td><td colspan="2">Alpaca CoI</td></tr><tr><td colspan="2">Base CoI</td><td colspan="2">Base</td></tr><tr><td></td><td>Test Set: CoI2</td><td></td><td></td><td></td></tr><tr><td>Subtask 1</td><td>1.32</td><td>90.50 49.21</td><td>13.98 7.02</td><td>84.16 45.57</td></tr><tr><td>Subtask 2</td><td>2.40</td><td></td><td></td><td></td></tr><tr><td></td><td colspan="2">Test Set: CoI3</td><td></td><td></td></tr><tr><td>Subtask 1</td><td>18.04</td><td>81.49</td><td>9.56</td><td>91.77</td></tr><tr><td>Subtask 2</td><td>6.82</td><td>68.65</td><td>2.13</td><td>71.67</td></tr><tr><td>Subtask 3</td><td>6.93</td><td>32.73</td><td>3.30</td><td>35.52</td></tr></table></body></html>

<html><body><table><tr><td rowspan="2">Model</td><td colspan="2">CoI2-test</td><td colspan="2">CoI3-test</td><td colspan="2">BBH</td></tr><tr><td></td><td></td><td>Mistral Alpaca|Mistral Alpaca|Mistral</td><td></td><td></td><td>Alpaca</td></tr><tr><td colspan="7">Baselines</td></tr><tr><td>Base</td><td>24.93</td><td>24.95</td><td>23.66</td><td>20.99</td><td>8.51</td><td>14.36</td></tr><tr><td>CoT</td><td>16.61</td><td>23.82</td><td>16.90</td><td>20.09</td><td>5.84</td><td>17.05</td></tr><tr><td>LtM</td><td>15.07</td><td>23.54</td><td>16.41</td><td>19.79</td><td>3.99</td><td>3.99</td></tr><tr><td>CoI1</td><td>39.72</td><td>32.32</td><td>29.62</td><td>21.75</td><td>27.68</td><td>28.74</td></tr><tr><td colspan="7">Chain-of-Instructions Models</td></tr><tr><td>CoI2</td><td>60.43</td><td>62.04</td><td>48.31</td><td>48.23</td><td>10.65</td><td>12.11</td></tr><tr><td>CoI3</td><td>33.63</td><td>31.62</td><td>60.03</td><td>47.03</td><td>5.78</td><td>7.00</td></tr><tr><td>CoI12</td><td>70.76</td><td>67.50</td><td>59.84</td><td>50.23</td><td>24.44</td><td>28.80</td></tr><tr><td>CoI123</td><td>45.16</td><td>67.12</td><td>61.61</td><td>67.49</td><td>29.39</td><td>27.57</td></tr></table></body></html>

Table 4: ROUGE-L results on compositional instruction test sets and BIG-Bench Hard (BBH). Base refers to the nonfine-tuned base models, $\mathbf { C o T } =$ chain-of-thought prompting on base models, $\mathbf { L t M } =$ least-to-most prompting on base models. The best scores are marked as bold .

that both Mistral and Alpaca fine-tuned on $\mathrm { C o I _ { 1 2 } }$ instructions perform the best for $\mathrm { C o I _ { 2 } }$ -test (Table 4) 4. Similarly, for $\mathrm { C o I _ { 3 } }$ -test, both $\mathrm { C o I _ { 1 2 3 } }$ Mistral and Alpaca perform the best. All models fine-tuned on compositional instructions generally outperform the baselines, except for $\mathrm { C o I _ { 3 } }$ -tuned Alpaca. This model performs slightly worse than the $\mathrm { C o I _ { 1 } }$ - tuned Alpaca on $\mathrm { C o I _ { 2 } }$ test set. We hypothesize that this happens because instructions in $\mathrm { C o I _ { 3 } }$ become very long, thereby it becomes harder for the model to generalize without $\mathrm { C o I _ { 2 } }$ and $\mathrm { C o I _ { 1 } }$ examples. As a result, models only fine-tuned on $\mathrm { C o I _ { 3 } }$ tend to generate long sentences with hallucinations as in Table 5.

In the LLM-as-a-judge experiment, we evaluate the performance of the best CoI models on $\mathrm { C o I _ { 2 } }$ -test and $\mathrm { C o I _ { 3 } }$ - test against the best baseline, $\mathrm { C o I _ { 1 } }$ . On $\mathrm { C o I _ { 2 } }$ -test, the LLM prefers $6 9 . 9 0 \%$ of Alpaca $\mathrm { C o I _ { 1 2 } }$ ’s outputs and $7 0 . 9 2 \%$ of Mistral $\mathrm { C o I _ { 1 2 } }$ ’s outputs over the baseline. Similarly, on $\mathrm { C o I _ { 3 } }$ -test, the LLM favors Alpaca $\mathrm { C o I _ { 1 2 3 } }$ ’s outputs and Mistral $\mathrm { C o I _ { 1 2 3 } }$ ’s outputs over the baseline by $8 1 . 0 4 \%$ and

90   
780 50 Prefer CoI Prefer CoI Prefer CoI Prefer CoI   
40   
30   
210 GRf GRf RU ? GURf   
0 CoI2-test CoI3-test CoI2-test CoI3-test Alpaca Mistral

$6 0 . 0 0 \%$ , respectively.

CoI Results per Subtasks We examine how CoI models perform for each subtask in the compositional instruction. To do this, we compare the results from the best version of CoI $\mathrm { C o I _ { 1 2 } }$ for $\mathrm { C o I _ { 2 } }$ test set, $\mathrm { C o I _ { 1 2 3 } }$ for $\mathrm { C o I _ { 3 } }$ test set) against the non-finetuned baseline models. Since there is no clear boundary to distinguish the first subtask output and the second subtask output in the baseline’s outputs, we use an LLM to separate the responses. Given the subtask instruction and the output, we ask the LLM to decide which span of the output text responds to the subtask instruction. To remove the possibility of LLM’s hallucination being counted as part of the output, we only include LLM’s output span when it appears in the baseline’s output. When LLM deems that the output is incorrect, we assign $\mathrm { R o U G E } = 0$ because this output could refer to the first subtask or second subtask.

Table 3 and Table 4 show results of CoI models and baseline on $\mathrm { C o I _ { 2 } }$ and $\mathrm { C o I _ { 3 } }$ test sets. In general, CoI models outperform the baseline for both test sets, with the best results on the first subtask and then followed by the second subtask and the third subtask. However, it is interesting to see that CoI’s performance on the second subtask for $\mathrm { C o I _ { 3 } }$ test set is higher than CoI’s performance on the second subtask for $\mathrm { C o I _ { 2 } }$ . We conjecture that it happens because the previous subtasks can be easier to solve or knowing the subsequent subtask helps improving the performance.

Human evaluation We randomly sample 200 instances from $\mathrm { C o I _ { 2 } }$ -test and $\mathrm { C o I _ { 3 } }$ -test for both Mistral and Alpaca and ask 8 computer science students with NLP knowledge to rate the output. We show 25 sets of instruction, input, baseline output, and CoI output to each annotator. The order of the outputs is randomized. Each annotator then selects the output they prefer. Figure 5 depicts the percentage of CoI outputs that humans prefer vs. when none of the outputs are preferred. Outputs from CoI-tuned models are preferred for all test sets for both models.

Table 5: Example outputs from different CoI-tuned Alpaca on a 2-instruction test instance.   

<html><body><table><tr><td colspan="3">Instruction: Simplify the given sentence by paraphrasing it. and then Determine if the paraphrased sentence has proper punctuation with True or False.</td></tr><tr><td colspan="3">Input: many of thechurches work together for town-wide projects under the bannerof "churches together in steve- nage".</td></tr><tr><td colspan="3">Correct output: Task 1 output and task 2 input: many of the churches work together for projects across the town</td></tr><tr><td>under the slogan of “churches together in stevenage". Task 2 output: False Model| Output</td><td></td><td>ROUGE-L</td></tr><tr><td>CoI1</td><td>many of the churches work together for town wide projects under the banner of“ churches</td><td>75.00</td></tr><tr><td>CoI2</td><td>together in stevenage". Task 1 output and task 2 input: many of the churches work together for projects across the</td><td>100</td></tr><tr><td>CoI3</td><td>town under the slogan of“ churches together in stevenage ”.Task 2 output: False 1 output and 2 input: many of the churches work together for projects across the town under the slogan of“churches together in stevenage ”.2 output and 3 input: The pronoun 'them’ refers to the noun phrase 'many of the churches’because the sentence states that 'they’work together for</td><td>43.75</td></tr><tr><td>CoI12</td><td>projects across the town. This coreference is justified by the knowledge that the churches are the ones that are working together.3 output:False 1 output and 1 input: many of the churches work together for projects across the town under the</td><td>85.71</td></tr><tr><td>CoI123</td><td>slogan of“ churches together in stevenage”.2 output: False Task 1 output and task 2 input: many of the churches work together for projects across the town under the slogan of“ churches together in stevenage”.Task 2 output: False</td><td>100</td></tr></table></body></html>

![](images/756002638e983dfe7331dac8ea7cd0d09dca8e898655e34886aa152d46a0d202.jpg)  
Figure 6: ROUGE-L $\mathbf { \bar { X } }$ -axis) on CoI test sets $\sigma = 2 , 3 , 4 , 5$ for various Alpaca models (y-axis). Base refers to the nonfine-tuned Alpaca.

# Generalization to Unseen Single Tasks

To assess whether adding compositional instructions helps improve the model’s performance on unseen and difficult single instruction tasks, we tested CoI-tuned models on BIGBench Hard (BBH). $\mathrm { C o I _ { 1 2 3 } }$ -tuned Mistral performs the best (ROUGE: 29.39) as shown in Table 4. For Alpaca, the model fine-tuned on $\mathrm { C o I _ { 1 2 } }$ is also better than the baseline and achieves a higher ROUGE score of 28.80. This confirms that having compositional instructions helps the model to understand hard single instruction problems as well.

# Generalization to Longer Chains $\mathbf { \left( C o I _ { 4 , 5 } \right) }$ )

In this experiment, we examine whether our CoI models can generalize to longer chains. We run inference on $\mathrm { C o I _ { 4 } }$ and $\mathrm { C o I _ { 5 } }$ test sets using $\mathrm { C o I _ { 1 } }$ , $\mathrm { C o I _ { 1 2 } }$ , and $\mathrm { C o I _ { 1 2 3 } }$ -tuned Alpaca. 5 As shown in Figure 6, longer chains $( \sigma = 2 , 3$ ) in the training set help the model to understand unseen longer chain $( \sigma = 4 , 5 )$ ) in the test set as well. Moreover, the performance does not drop as high as $\mathrm { C o I _ { 1 } }$ or the baseline non-fine-tuned models that do not learn the chaining reasoning. We posit that the knowledge of compositional instructions in the training set, even though the length of the chain is shorter than 4 or 5, still helps the model to understand the composed tasks.

# Generalization to Downstream Composite Tasks

For this experiment, we use $\mathrm { C o I _ { 1 2 } }$ because it shows the highest ROUGE-L on 2-instruction problem. For the baseline, we use non-finetuned Alpaca and Mistral. We evaluate the performance of the models using four metrics below.

• ROUGE-L (all) the ROUGE score of the summary of the whole generated output.   
• ROUGE-L (src) the ROUGE score only from the summary in the source language.   
• ROUGE-L (tgt) the ROUGE score only from the summary in the target language.   
• #valid outputs number of valid summaries in the source and the target languages are generated because sometimes the model may not generate them properly.

Table 6 shows the results for our downstream task experiments. For the English-to-French summarization task, $\mathrm { C o I _ { 1 2 } }$ can generate more valid target outputs than the baselines. Moreover, $\mathrm { C o I _ { 1 2 } }$ obtains higher ROUGE for both source and target summaries than the baselines. For the Spanish-toEnglish summarization task, $\mathrm { C o I _ { 1 2 } }$ Mistral outperforms the baseline for all ROUGE-L scores, but Alpaca fails to have better ROUGE-L (src) and ROUGE-L (tgt) against the baseline.

Table 6: Results of the multilingual summarization task on 300 instances. Base refers to non-fine-tuned baseline, src is source language, and tgt is target language.   

<html><body><table><tr><td>Metric</td><td colspan="2">Mistral</td><td colspan="2">Alpaca</td></tr><tr><td></td><td>Base</td><td>CoI12</td><td>Base</td><td>CoI12</td></tr><tr><td colspan="5">English to French</td></tr><tr><td>ROUGE-L (all) ROUGE-L (src)</td><td>8.03 10.68</td><td>10.97 15.66</td><td>5.78 3.84</td><td>8.90 12.71</td></tr><tr><td>ROUGE-L (tgt)</td><td>7.45</td><td>10.93</td><td>5.46</td><td>7.96</td></tr><tr><td>#valid src outputs #valid tgt outputs</td><td>206 212</td><td>295 295</td><td>126 221</td><td>228 228</td></tr><tr><td colspan="5">Spanish to English</td></tr><tr><td>ROUGE-L (all) ROUGE-L (src) ROUGE-L (tgt)</td><td>11.22 0.07 11.22</td><td>12.43 4.85 12.30</td><td>7.87 2.47 7.68</td><td>10.39 1.87 7.13</td></tr><tr><td>#valid src outputs #valid tgt outputs</td><td>1 300</td><td>290 290</td><td>80 240</td><td>150 150</td></tr></table></body></html>

In general, CoI performs better in English-to-French summarization compared to Spanish-to-English summarization because our training instances contain a translation task from English to other languages (Punjabi, German, and Catalan), even though the target language of the translation task in the training set is not French. On the other hand, we see poor performance in Spanish summaries across all models, possibly due to the lack of Spanish as the first subtask in training datasets. We conjecture this issue could be resolved if we add more Spanish tasks during the fine-tuning stage.

# Related Work

struct a new compositional dataset upon Wang et al. (2022)’s SUPER-NATURALINSTRUCTION. Our work is also related to several past works which have leveraged LLMs to generate training data (Schick and Schütze 2021), and some of them specifically use LLMs for generating instruction data (Peng et al. 2023; Shao et al. 2023). Nevertheless, our CoI data generation framework differs from previous works as we use LLMs to determine the composability of individual instructions, and then generate responses for subsequent subtask instructions.

Instruction tuning There has been a notable surge in research focused on fine-tuning LLMs using human instructions. Efrat and Levy (2020) examined LLMs’ ability to follow natural language instructions compared to crowdworkers. Wei et al. (2022a); Sanh et al. (2021) have transformed NLP task descriptions into human-like language instructions and showed that LLMs fine-tuned with those instructions have generalizable capability toward unseen tasks (Chung et al. 2024). Subsequently, many studies have emerged to create new instruction datasets aimed at training models in instruction-tuning paradigm: some instruction datasets are fully written by humans (Wang et al. 2022; Conover et al. 2023), the others are written with the help of LLMs (Honovich et al. 2023; Wang et al. 2023; Taori et al. 2023); some instructions are NLP-specific (Mishra et al. 2022; Wang et al. 2022; Weller et al. 2020), and the others are designed to respond to general-purpose instructions (Ouyang et al. 2022; Wang et al. 2023). These prior studies only work on single instruction datasets, so we con

Compositional problems in NLP Several NLP work have investigated the capability of Transformer model on compositional problems including algorithm and math problems (Dziri et al. 2023), compositional semantics (Drozdov et al. 2022), and multi-hop question-answering (QA) tasks (Trivedi et al. 2022). Dziri et al. (2023) highlight how Transformer models often struggle with compositional mathematics computation or program executions (Nye et al. 2022; Saparov and He 2022). Drozdov et al. (2022) introduce a new prompting method which first decomposes the compositional questions or sentences (Keysers et al. 2019; Kim and Linzen 2020), then sequentially predicts the answers to subproblems, and finally generating the final output. Compositionality in NLP is closely related with multi-hop QA problems with compositional questions where the answers from sub-questions are needed to answer the main question (Yang et al. 2018; Ho et al. 2020; Trivedi et al. 2022). Qiu et al. (2022) have shown how a model with compositional latent structure improves large language models’ performance on compositional generalization tasks through synthethic data augmentation. CoI is most related to Aksu et al. (2023) as they work on dealing with compositional tasks for dialogue systems. However, their definition of “compositional task” is different from ours as they do not require the output of one subtask is shown to the next subtask. Meanwhile, in our CoI, the outputs from the previous subtasks become the next subtask inputs.

# Conclusion and Future Work

In this work, we propose a new task called Chain-ofInstructions and develop a dataset for building models to solve the task. We introduce an automatic pipeline on how to build our dataset and demonstrate the usefulness of our CoI-tuned models on the tasks of the generated dataset and downstream tasks. Since human language is complex and an instruction may actually be composed of subtasks, it is important to have a model that can deal with compositional instructions, especially as we show that models fine-tuned only on single instructions never outperform the CoI-tuned models on multi-instruction tasks. For future work, we consider looking into instruction decomposition in addition to the instruction composition problem. We also recommend trying out more tasks to be composed besides those from SUPERNATURALINSTRUCTION.