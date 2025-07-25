# C3oT: Generating Shorter Chain-of-Thought Without Compromising Effectiveness

Yu Kang, Xianghui Sun, Liangyu Chen \*, Wei Zou

Beike Inc., Beijing, China {kangyu009, sunxianghui002, chenliangyu003, zouwei026}@ke.com

# Abstract

Generating Chain-of-Thought (CoT) before deriving the answer can effectively improve the reasoning capabilities of large language models (LLMs) and significantly improve the accuracy of the generated answer. However, in most cases, the length of the generated CoT is much longer than the desired final answer, which results in additional decoding costs. Furthermore, existing research has discovered that shortening the reasoning steps in CoT, even while preserving the key information, diminishes LLMs’ abilities. These phenomena make it difficult to use LLMs and CoT in many real-world applications that only require the final answer and are sensitive to latency, such as search and recommendation. To reduce the costs of model decoding and shorten the length of the generated CoT, this paper presents Conditioned Compressed Chain-of-Thought (C3oT), a CoT compression framework that involves a compressor to compress an original longer CoT into a shorter CoT while maintaining key information and interpretability, a conditioned training method to train LLMs with both longer CoT and shorter CoT simultaneously to learn the corresponding relationships between them, and a conditioned inference method to gain the reasoning ability learned from longer CoT by generating shorter CoT. We conduct experiments over four datasets from arithmetic and commonsense scenarios, showing that the proposed method is capable of compressing the length of generated CoT by up to more than $50 \%$ without compromising its effectiveness.

# Introduction

The Chain-of-Thought (CoT) (Nye et al. 2021; Marasovi´c et al. 2021; Wei et al. 2022; Kojima et al. 2022; Lampinen et al. 2022) methodology significantly augments the reasoning abilities of large language models (LLMs), providing critical capabilities for sub-task decomposition in complex problem-solving scenarios. Furthermore, models trained with rich signals, including reasoning processes; explanation traces; and step-by-step thought processes, generally exhibit superior performance (Mukherjee et al. 2023; Mitra et al. 2023). While answering after thinking can elicit highly effective generations by activating LLMs’ reasoning abilities, the intermediate reasoning steps in model outputs are often much longer than the desired final answers, notably increasing the cost during the inference phase, and hindering the model’s employment in many real-world applications, such as search and recommendation, which usually focus only on the final answer and is sensitive to latency. Therefore, striking a balance between the demand for fast decoding in LLMs applications and the need for long reasoning steps has become an urgent issue.

However, recent studies indicate that lengthening the reasoning steps in CoT considerably enhances LLMs’ reasoning abilities across multiple tasks. Alternatively, shortening the reasoning steps, even while preserving the key information, significantly diminishes the reasoning abilities of models (Jin et al. 2024). Fu et al. (2022) propose a complexity-based method for CoT selection and find that CoT with higher reasoning complexity, i.e., chains with more reasoning steps, achieve substantially better performance on multi-step reasoning tasks. Similar conclusions have been drawn from Merrill and Sabharwal (2023)’s work, which explores the relationship between the capabilities of LLMs and the number of CoTs’ reasoning steps, varying from logarithmic, linear, to polynomial, based on input length. They also found that the increase in LLMs’ computational power depends crucially on the amount of intermediate reasoning steps added.

There has been little work (Deng et al. 2023; Liu et al. 2024) focused on compressing the length of generated CoT without sacrificing model performance. Implicit-CoT (Deng et al. 2023) attempted to use LLMs’ internal hidden states to perform implicit reasoning, replacing explicitly producing the CoT reasoning steps, but the results of this method is still significantly falling behind the explicit CoT method.

Based on these results, we ask: Is there a method that can significantly reduce the length of intermediate reasoning steps in generated CoT without compromising effectiveness? The answer is yes, we propose Conditioned Compressed Chain-of-Thought (C3oT), a CoT compression framework, to achieve this goal. Specifically, we first present a CoT compressor to condense the original complex CoT into their shortest form while retaining essential information and interpretability, now we have pairs of longer CoT and shorter CoT. We further introduce a conditioned training method to train LLMs with both longer CoT and shorter CoT simultaneously, and by conditioning longer CoT and shorter CoT using distinct initial prompt tokens before instructions, LLMs can learn the differences and connections between them. Lastly, we propose the conditioned inference method which is used in the inference phase, by applying the initial prompt tokens used for conditioning the shorter CoT before instructions, LLMs can generate CoT with significantly shorter length during inference while maintaining the accuracy of the derived final answer.

To validate the effectiveness of our approach, we conduct experiments on four datasets from two domains that require reasoning, i.e., arithmetic (GSM8K, MathQA) and commonsense (ECQA, StrategyQA). The results show that our method’s performance is on par with models trained using only the original longer CoT across all datasets, while significantly shortening the length of generated CoT. Additionally, we design extensive experiments and discussions to analyze the contribution of different components in our approach, as well as to explore future research directions of CoT compression based on our method.

The contributions of this paper are as follows:

• We propose C3oT, a CoT compression framework used to reduce the cost of model inference by drastically shortening the length of CoT generated by LLMs without loss of effectiveness. We are the first to significantly shorten the length of CoT in model outputs without sacrificing performance, filling the gap in the field of model inference acceleration in terms of shortening the length of intermediate output. • Comprehensive experiments demonstrate that our CoT compression method outperforms all baselines on various reasoning tasks, such as math reasoning (GSM8K, MathQA) and commonsense reasoning (ECQA, StrategyQA). We conduct detailed ablation studies and analyses to prove the effectiveness of our approach. • We conduct a series of extension experiments based on the proposed C3oT framework, providing insights for future research directions in the field of CoT compression and further demonstrating the effectiveness of our method.

# Related Work

# LLMs Inference Acceleration

Due to the conflict between the outstanding performance of LLMs and the difficulty of their application in real-world scenarios, an increasing amount of research is focusing on accelerating the inference of LLMs. These works primarily focus on reducing the number of input tokens to be processed in order to reduce the inference cost. Some approaches focus on reducing the length of prompts by using prompt tuning to learn special tokens (Wingate, Shoeybi, and Sorensen 2022; Chevalier et al. 2023; Ge et al. 2023; Mu, Li, and Goodman 2024). Some approaches attempt to compress prompt based on information entropy (Li et al. 2023; Jiang et al. 2023a,b) and data distillation (Pan et al. 2024). Some studies utilize LLMs to summarize input dialog or data, transforming them into efficient memory and knowledge (Chase 2022; Zhang et al. 2023). There are also some studies focus on token pruning or token merging (Kim et al. 2022; Modarressi, Mohebbi, and Pilehvar 2022; Bolya et al. 2022).

However, during the inference stage, the cost of decoding and generating output by the model is significantly higher than the cost of processing the input. Therefore, as the enhancements of model capabilities brought by CoT gain increasing attention, the additional cost involved in generating CoT cannot be ignored. Nevertheless, accelerating the generation of CoT has not received widespread attention.

Recently, only Implicit-CoT (Deng et al. 2023) has attempted to accelerate the generation of CoT. It uses the hidden states of different layers in LLMs to perform implicit reasoning based on knowledge distillation, avoiding the explicit generation of CoT and thereby accelerating the inference process. But Implicit-CoT severely sacrifices model performance. The results generated by this method significantly fall behind those of the explicit CoT method.

The method proposed in this paper also aims to accelerate the inference by reducing the length of generated CoT. By utilizing conditioned training method, the model is enabled to learn both longer and shorter CoT simultaneously, and during the conditioned inference phase, the model is able to stimulate the reasoning capabilities learned from the longer CoT by generating a shorter CoT. In this way, our approach significantly reduces the length of generated CoT without compromising the effectiveness of the model.

It’s worth mentioning that there is a line of studies that attempt to accelerate inference through model quantization (Dettmers et al. 2022; Frantar et al. 2022; Xiao et al. 2023), pruning (Frantar and Alistarh 2023; Sun et al. 2023; Das, Ma, and Shen 2023), and other similar techniques. These methods are orthogonal to ours and can be used together.

# Chain-of-Thought Analyzing

There is some research focusing on exploring the relationship between the length of CoT and its effects. Interestingly, all of these studies (Fu et al. 2022; Merrill and Sabharwal 2023; Jin et al. 2024) have found that lengthening the intermediate reasoning steps in the CoT can enhance LLMs’ capabilities. Fu et al. (2022) find that CoT with higher reasoning complexity, i.e., chains with more reasoning steps, achieve substantially better performance on multi-step reasoning tasks. Merrill and Sabharwal (2023) explore the relationship between the capabilities of LLMs and the number of CoTs’ reasoning steps, varying from logarithmic, linear, to polynomial, based on input length, and they find that the increase in LLMs’ computational power depends crucially on the amount of intermediate reasoning steps added. Jin et al. (2024) design experiments that expand and compress the reasoning steps in CoT while keeping all other factors constant. They find that lengthening the reasoning steps even without adding new information considerably enhances LLMs’ reasoning abilities. Conversely, shortening the reasoning steps while preserving the key information, significantly diminishes the reasoning abilities of models.

This paper also focuses on the relationship between the length of CoT and its effectiveness, but we propose a method that can significantly compress the length of generated CoT without compromising its effectiveness.

# Method

# Problem Statement

Given a dataset rilong, yi)}iN=1, where xi denotes the instruction, $y _ { i }$ is its corresponding answer and $r _ { i } ^ { l o n g }$ is a well-designed detailed $\mathrm { C o T }$ for deriving the answer. We consider a compressor $\mathcal { F }$ that systematically compress any input CoT to its shortFest form $r ^ { s h o r t } = \dot { \mathcal { F } } ( r ^ { l o n \dot { g } } )$ , retaining only the key information. Our goal is to train an LLM on D = {(xi, rilong, rishort, yi)}iN=1 so that during inference, the distribution of generated answers derived from compressed, shorter CoT is as similar to answers derived from original, longer CoT as possible.

# Conditioned Compressed Chain-of-Thought (C3oT)

Next, we elaborate on the details of the C3oT framework, which shortens the generated CoT during inference without compromising its effectiveness. An overview of the proposed framework is shown in Figure 1.

Compressor The CoT compressor can be any summarization model that processes the input text to only retain its core information and returns a condensed version. In this paper, we employ GPT-4 (Achiam et al. 2023) as the compressor for CoT compression.

Using GPT-4 as the compressor is also to validate the conclusions of previous research. Specifically, even when using the current most powerful closed-source model to ensure that the compressed CoT retains all key information and interpretability while merely removing redundant words, only using these compressed, shorter CoT to derive the answers will still affect the model’s performance. This is consistent with the conclusions of previous research and further proves the value of our approach.

We prompt GPT-4 to obtain the corresponding compressed, shorter CoT $r _ { i } ^ { s h o r t }$ for all original, longer CoT rlong in the dataset (xi , rilong, yi)}iN=1, and compose D = , rilong, rishort, yi)}iN=1. We also investigate the impact on our approach of employing different models as compressors in the Analysis section.

Conditioned Training Inspired by OpenChat (Wang et al. 2023), we can regard $\mathcal { D }$ as a class-conditioned dataset $\mathbfcal { D } _ { c } = \{ ( x _ { i } , \tilde { r } _ { i } , y _ { i } , \bar { c } _ { i } ) \}$ . Each instruction $x _ { i }$ in the dataset corresponds to both a longer CoT and a shorter CoT. The CoT of different lengths are distinguished through different conditions:

$$
\tilde { r } _ { i } = \left\{ \begin{array} { l l } { r _ { i } ^ { l o n g } } & { \mathrm { i f } ~ c _ { i } = l o n g } \\ { r _ { i } ^ { s h o r t } } & { \mathrm { i f } ~ c _ { i } = s h o r t } \end{array} \right.
$$

where $\tilde { r } _ { i }$ can be either $r _ { i } ^ { l o n g }$ or $r _ { i } ^ { s h o r t }$ , controlled by condition $c _ { i }$ .

Then we fine-tune an LLM on $\mathcal { D } _ { c }$ to teach it the relationship between rilong and rishort. We model the LLM to be fine-tuned as a class-conditioned policy $\pi _ { \boldsymbol { \theta } } ( y , \tilde { r } | x , c )$ . This can be easily implemented by conditioning each CoT which belongs to either a longer CoT or a shorter CoT using distinct initial prompt tokens before instruction as shown below:

[Long Condition]

Answer and provide a detailed thought process:

# [Short Condition]

Answer and provide as brief a thought process as possible:

After adding conditions in this way, each sample in the original dataset, for example:

Instruction: Natalia sold clips ... How many clips did Natalia sell altogether in April and May?   
Rationale: ... Natalia sold $4 8 + 2 4 = \ast 4 8 + 2 4 = 7 2 \ast 7 2$ clips altogether in April and May.

becomes two samples containing the original, longer CoT and the compressed, shorter CoT, respectively:

Instruction: Answer and provide a detailed thought process: Natalia sold clips ... How many clips did Natalia sell altogether in April and May?

Rationale: ... Natalia sold $4 8 + 2 4 = \ast 4 8 + 2 4 = 7 2 \ast 7 2$ clips altogether in April and May.

and

Instruction: Answer and provide as brief a thought process as possible: Natalia sold clips ... How many clips did Natalia sell altogether in April and May?

Rationale: ... She sold 72 clips in April and May.

Now, fine-tuning an LLM on $\mathcal { D } _ { c }$ is the same as general supervised fine-tuning (SFT). It is worth mentioning that during fine-tuning, longer CoT and shorter CoT in $\mathcal { D } _ { c }$ do not need to appear in pairs, and there is no need to use any method to inform the LLM how they correspond to each other, just randomly shuffle $\mathcal { D } _ { c }$ and train the model like general SFT. Additionally, our conditioned training method can also be regarded as a data augmentation method, but it does not introduce any extra knowledge (Maini et al. 2024).

Conditioned Inference During the inference phase, we assume that the model after conditioned training has learned the differences and connections between the longer CoT and the shorter CoT. Considering that we aim to apply the fine-tuned model to a real-world application and exclusively generate shorter CoT to derive the needed final answer efficiently, we use the same specific prompts that were employed in shorter CoT during the conditioned training phase as below:

# [Inference Prompt]

Answer and provide as brief a thought process as possible: <Question>

# Experiment

# Settings

Datasets To comprehensively validate the effectiveness of C3oT, we evaluated its performance across four datasets from two domains. For math reasoning, we use GSM8K (Cobbe et al. 2021) and MathQA (Amini et al. 2019). As for commonsense reasoning, we use ECQA (Aggarwal et al. 2021) and StrategyQA (Geva et al. 2021). All these datasets not only contain the final answers but also include the carefully human-designed CoT used to arrive at the final answers. We

Original Data [Long Condition] Conditioned Data   
$\cdot$ INnastarluiactsiolnd clips to 48 of her friends in April, and dAentsawilerdatnhdo upgrohtv pdreoacess: • [Long Condition] + Instruction   
then she sold half as many clips in May. How many • Long CoT   
clips did Natalia sell altogether in April and May?   
• Long CoT • Short CoT • Answer   
Natalia sold $4 8 / 2 = < < 4 8 / 2 = 2 4 > > 2 4$ clips in May. Natalia sold 24 clips in May.   
Natalia sold $4 8 + 2 4 = < < 4 8 + 2 4 = 7 2 > > 7 2$ clips Compressor Altogether, she sold 72 clips [Short Condition]+ Instruction   
altogether in April and May. in April and May. Short CoT   
Answer 72 Conditioned Answer [Short Condition] Training Answer and provide as brief a LLMs thought process as possible: New Instruction Response • Short CoT   
Conditioned Inference • Answer

followed the training and testing set division as outlined in the original paper of the dataset used, trained C3oT on the training set, and evaluated its performance on the test set, excluding StrategyQA. Due to the inaccessibility of ground truths for the StrategyQA test set, we proceeded to further split the original StrategyQA training set into training and test sets.

Implementation Details In this paper, we train C3oT based on $\mathrm { L L a M A - } 2 \mathrm { - C h a t - 7 B }$ and $^ { - 1 3 \mathtt { B } }$ (Touvron et al. 2023). We fine-tune the model for 2 epochs on each dataset using the AdamW optimizer with a sequence length of 2,048 tokens and a batch size of 128. The AdamW optimizer’s hyperparameters are set as follows: $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 , \epsilon =$ $\bar { 1 } 0 ^ { - 6 }$ , and weight decay of 0.001. We employ a cosine learning rate schedule with a maximum learning rate of $1 \times 1 0 ^ { - 5 }$ .

Baselines We consider the following baselines:

• Short CoT: Use GPT-4 as the compressor to compress the original, longer CoT as much as possible while retaining key information and interpretability, then train models using only these compressed, shorter CoT.   
• Long CoT: Use only the original, longer CoT to train models.   
• Implicit-CoT (Deng et al. $2 0 2 3 ) ^ { 1 }$ : Use the LLM’s internal hidden states to perform implicit reasoning, instead of explicitly producing the CoT reasoning steps. The implicit reasoning steps are distilled from a teacher model trained on explicit CoT reasoning.

All the baselines are also trained based on LLaMA-2-Chat $- 7 B$ and $- 1 3 \mathtt { B }$ , including the teacher model and student model of Implicit-CoT. The hyperparameters used for training baselines are the same as those in training C3oT.

# Evaluation We measure the following metrics:

• Accuracy: Following previous works (Suzgun et al. 2022; Cobbe et al. 2021; Amini et al. 2019; Aggarwal et al. 2021; Geva et al. 2021), we measure accuracy using exact match, computed by comparing the generated final answer with the ground-truth label. • Compression Rate: Additionally, we measure the compression rate to evaluate the reduction in length of the generated CoT. The compression rate $\rho$ is defined as $\rho = ( L - \tilde { L } ) / L$ , $\rho \in ( - \infty , 1 ]$ , where $L$ is the length of the CoT generated by Long $C o T$ , which we regard as the benchmark length. And $\bar { \tilde { L } }$ is the length of the generated compressed CoT. A larger value of compression rate implies a greater reduction in length, resulting in a lower inference cost, which is preferable. When no intermediate $\mathrm { C o T }$ steps are generated, the compression rate can reach 1. If the generated CoT is even longer than $L$ , the compression rate becomes negative.

# Main Results

Table 1 reports the results of our approach alongside baselines on GSM8K, MathQA, ECQA and StrategyQA. It can be seen that our proposed C3oT consistently outperforms the Implicit-CoT in accuracy by a large margin in all experiments. Implicit-CoT successfully avoids explicitly generating CoT, achieving a $100 \%$ compression rate. However, it significantly sacrifices the model’s performance, which is not what we want. In addition, three other conclusions can be drawn from these results:

Firstly, comparing the results of Short CoT and Long CoT reveals that even while using the most powerful model GPT4 as the compressor to preserve the key information and interpretability in the compressed, shorter CoT in the training

<html><body><table><tr><td rowspan="3">Mode1</td><td rowspan="3">Method</td><td colspan="4">Arithmetic</td><td colspan="4">ECQAComonsestrtegQA</td></tr><tr><td colspan="2">GSM8K</td><td colspan="2">MathQA</td><td colspan="2"></td><td colspan="2"></td></tr><tr><td>Acc</td><td>ComRaression</td><td>Acc</td><td>ComRarssion</td><td>Acc</td><td>Compression</td><td>Acc</td><td>ComRaession</td></tr><tr><td rowspan="4">7B</td><td>Short CoT</td><td>31.01</td><td>58.63</td><td>46.16</td><td>29.53</td><td>61.93</td><td>53.41</td><td>67.59</td><td>37.99</td></tr><tr><td>Long CoT</td><td>37.38</td><td>0</td><td>51.46</td><td>0</td><td>63.96</td><td>0</td><td>69.66</td><td>0</td></tr><tr><td>Implicit-CoT</td><td>11.16</td><td>100</td><td>14.62</td><td>100</td><td>21.14</td><td>100</td><td>30.01</td><td>100</td></tr><tr><td>C3oT</td><td>36.92</td><td>56.67</td><td>50.35</td><td>27.39</td><td>69.38</td><td>51.55</td><td>72.41</td><td>42.04</td></tr><tr><td rowspan="4">13B</td><td>Short CoT</td><td>42.46</td><td>59.52</td><td>52.97</td><td>29.85</td><td>66.79</td><td>55.27</td><td>74.83</td><td>47.79</td></tr><tr><td>Long CoT</td><td>48.07</td><td>0</td><td>56.21</td><td>0</td><td>68.92</td><td>0</td><td>76.21</td><td>0</td></tr><tr><td>Implicit-CoT</td><td>14.36</td><td>100</td><td>17.00</td><td>100</td><td>23.54</td><td>100</td><td>35.77</td><td>100</td></tr><tr><td>C3oT</td><td>47.10</td><td>57.78</td><td>56.62</td><td>31.04</td><td>71.93</td><td>55.28</td><td>76.55</td><td>44.56</td></tr></table></body></html>

Table 1: The Accuracy $( \% )$ and Compression Rate $( \% )$ performance of the proposed C3oT and baselines. The bold scores denot the best performance, as well as performances within $1 \%$ of the best.

sets, it still significantly diminishes the model’s effectiveness.   
This conclusion is consistent with previous studies.

Secondly, while shortening the length of generated CoT reduces the model’s performance across all datasets, the degree of performance decrease varies across different datasets. Tasks requiring more reasoning abilities, such as math, experience a greater decrease in performance, whereas tasks with lower reasoning demands, such as commonsense, see a relatively smaller decrease. Similarly, although C3oT has achieved similar or even better performance than Long CoT on all datasets, there is still a slight lag in mathematical tasks, while in commonsense tasks, it even surpasses or significantly outperforms Long CoT. This is still related to the reasoning ability required for the tasks.

Lastly, the compression rates on four datasets show that when preserving the key information and interpretability in the compressed, shorter CoT in the training sets, and keeping the compressor unchanged, the compression rate is only related to the dataset itself and not significantly influenced by the domain of the task. In other words, if the original, longer CoT in the training set is more detailed, containing more redundant information, then the compression rate achievable by C3oT will be higher. Additionally, we further analyze the impact of different compressors on compression rates in the Analysis section.

# Analysis

In this section, we conduct experiments to answer the following questions, in order to analyze the contributions of different components in our approach, and further explore more future research directions for CoT compression based on the proposed C3oT framework.

# What is the contribution of the class-conditioned policy?

We conduct an ablation study on the conditions of C3oT to ascertain the contribution of the class-conditioned policy. For w/o condition, we remove the distinct initial prompt tokens before instruction and treat the data containing longer CoT and shorter CoT as equivalent, and then fine-tune the models in the regular supervised fine-tuning (SFT) manner.

Table 2 shows that C3oT outperforms w/o condition in terms of both accuracy and compression rate across all datasets. This is because without the class-conditioned policy, language models lack explicit signals to discern between the longer CoT and the shorter CoT, and during training, the models not only fail to learn the differences and connections between the longer CoT and the shorter CoT, but also get confused by two kinds of CoT with significantly different lengths, thus affecting the performance.

# What is the impact of different compressors on C3oT?

To investigate the impact of different compressors on our method, we conduct experiments comparing the results of using GPT-4 as the compressor with the results of using the open-source models $\mathrm { L L a M A - } 2 \mathrm { - C h a t - } 7 \mathrm { B }$ and $- 1 3 \mathtt { B }$ as the compressors. To distinguish the results of different compressors, in Table 3, we name the results as Short $C o T _ { < C o m p r e s s o r > }$ and $C 3 o T _ { < C o m p r e s s o r > } .$ It is worth mentioning that Short $C o T _ { G P T - 4 }$ and $C 3 o T _ { G P T - 4 }$ are precisely Short CoT and C3oT in Table 1. The prompts used for the different compressors are the same.

Comparing the results of Short $C o T _ { G P T - 4 }$ and Short $C o T _ { L L a M A 2 \_ * }$ as well as $C 3 o T _ { G P T - 4 }$ and $C 3 o T _ { L L a M A 2 \_ * }$ in Table 3, it’s evident that the compressed, shorter CoT generated by LLaMA-2 in the training set are slightly inferior in quality to those generated by GPT-4, resulting in a minor decrease in model accuracy, though not significantly. However, the conciseness of the compressed, shorter CoT generated by LLaMA-2 in the training set are noticeably poorer than those generated by GPT-4, leading to a compression rate that is over $1 5 \%$ lower. This indicates that preserving the key information in the original, longer CoT is not difficult for the compressor, but the more powerful the compressor, the more it can produce concise compressed CoT.

# Is C3oT effective for expanded CoT?

Previous studies have shown that lengthening the reasoning steps in CoT can enhance the model’s reasoning abilities. Therefore, in this part, we explore whether our approach can compress the much longer, expanded CoT and maintain its effectiveness. We follow the 5 reasoning steps expansion methods proposed by Jin et al. (2024) and use GPT-4 to expand the original CoT. The much longer, expanded CoT is then combined with the compressed, shorter CoT generated by GPT-4 that we used above to form a class-conditioned dataset for training C3oT, which we name $C 3 o T _ { E x p a n s i o n }$ In parallel, we refer to the results obtained from the model trained using only the much longer, expanded CoT as $E x .$ - panded CoT.

Table 2: Ablation study of the class-conditioned policy (condition) to C3oT.   

<html><body><table><tr><td rowspan="3">Mode1</td><td rowspan="3">Method</td><td colspan="4">Arithmetic</td><td colspan="4">Commonsense</td></tr><tr><td colspan="2">GSM8K</td><td colspan="2">MathQA</td><td colspan="2">ECQA</td><td colspan="2">StrategyQA</td></tr><tr><td>Acc</td><td>Compression Rate</td><td>Acc</td><td>Compression Rate</td><td>Acc</td><td>Compression Rate</td><td>Acc</td><td>Compression Rate</td></tr><tr><td rowspan="2">7B</td><td>w/ Codition</td><td>36.92</td><td>56.67</td><td>50.35</td><td>27.39</td><td>6.3</td><td>51.55</td><td></td><td>42.04</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>72.40</td><td></td></tr><tr><td rowspan="2">13B</td><td>C3oT</td><td>47.10</td><td>57.78</td><td>56.62</td><td>31.04</td><td>71.93</td><td>55.28</td><td>76.55</td><td>44.56</td></tr><tr><td>w/o condition</td><td>44.50</td><td>40.51</td><td>55.34</td><td>18.53</td><td>70.54</td><td>27.19</td><td>73.10</td><td>20.37</td></tr></table></body></html>

Table 3: Performance of some experiments based on C3oT on GSM8K. The negative compression rates represent the degree of increase in length.   

<html><body><table><tr><td rowspan="2">Model Size</td><td rowspan="2">Method</td><td colspan="2">GSM8K</td></tr><tr><td>Acc</td><td>Compression Rate</td></tr><tr><td rowspan="6">7B</td><td>Long CoT</td><td>37.38</td><td>0</td></tr><tr><td>Short CoTGPT-4</td><td>31.01</td><td>58.63</td></tr><tr><td>C3oTGPT-4</td><td>36.92</td><td>56.67</td></tr><tr><td>Short CoTLLaMA2_7B C3oTLLaMA2_7B</td><td>31.54 36.13</td><td>42.28 40.82</td></tr><tr><td>Expanded CoT</td><td></td><td></td></tr><tr><td>C3oTExpansion</td><td>39.12 37.30</td><td>-310.17 59.27</td></tr><tr><td rowspan="5"></td><td>C3oTAdapt</td><td>40.85</td><td>70.80</td></tr><tr><td>Long CoT</td><td>48.07</td><td>0</td></tr><tr><td>Short CoTGPT-4</td><td>42.46</td><td>59.52</td></tr><tr><td>C3oTGPT-4</td><td>47.1</td><td>57.78</td></tr><tr><td>Short CoTLLaMA2_13B</td><td>40.71</td><td>40.34</td></tr><tr><td rowspan="5"></td><td>C3oTLLaMA2_13B</td><td>46.53</td><td>42.48</td></tr><tr><td>Expanded CoT</td><td></td><td></td></tr><tr><td></td><td>49.66 48.12</td><td>-307.88 57.66</td></tr><tr><td>C3oTExpansion</td><td></td><td></td></tr><tr><td>C3oTAdapt</td><td>51.09</td><td>77.97</td></tr></table></body></html>

Comparing the results of Long CoT and Expanded CoT in Table 3, it is evident that lengthening the reasoning steps in CoT does improve the model’s reasoning abilities, which is consistent with previous studies. While comparing the results of $C 3 o T _ { G P T - 4 }$ and $C 3 o T _ { E x p a n s i o n }$ , we observe that although the improvement is not as significant as from Long $C o T$ to Expanded CoT, C3oTExpansion still manages to

50 35 C3oT 45 C3oT   
30 Short 40 Short   
205 Long Mixed 2350 Mixed Long 15 10 50 60 70 80 90 100 50 60 70 80 90 100 or or or or or or or or or or or or Con1Con2Con3Con4Con5Con6 Con1Con2Con3Con4Con5Con6 Compression Rate $( \% )$ of Compression Rate $( \% )$ of the Training Set or the Training Set or Condition Used during the Condition Used during the Inference Phase Inference Phase (a) Model Size: 7B (b) Model Size: 13B

achieve a better result than $C 3 o T _ { G P T - 4 }$ while maintaining a similar compression rate. This not only demonstrates the effectiveness of our approach on the much longer, expanded CoT but also presents a method to enhance the model’s reasoning abilities without incurring additional costs.

# What is the impact of training sets with different compression rates on C3oT?

When we use GPT-4 as the compressor in the previous sections, we prompt it to retain as much key information and interpretability from the original CoT in the training set as possible, the compression rate of the compressed CoT in the resulting GSM8K training set is about $50 \%$ compared to the original CoT. We wonder about the impact on model performance when all restrictions on the compressor are removed, and it is only required to compress the original CoT in the training set to a specified compression rate. Specifically, we compress the original CoT in the GSM8K training set with compression rates varying from $50 \%$ to $100 \%$ (no CoT) in $10 \%$ increments.

The results are shown in Figure 2a and Figure 2b. Firstly, we observe that the accuracy of $C 3 o T$ decreases as the compression rate of the training set increases, indicating that $C 3 o T$ is not effective across all compression levels. Secondly, although C3oT outperforms Short CoT at almost all training set compression rates, the performance gap between the two widens as the compression rate decreases, only to narrow again at the $50 \%$ .

These two phenomena indicate that C3oT can only achieve results comparable to Long CoT if the compressed, shorter CoT in the training set retains sufficient key information, and if the CoT in the training set is over-compressed, using C3oT will still lead to a decline in performance. Furthermore, at high compression rates, the compressed, shorter CoT even loses its grammatical structure, rendering it completely uninterpretable. At this point, the results of $C 3 o T$ and Short CoT are not significantly different. As the compression rate of the training set decreases, the information and interpretability contained in the compressed CoT increase. Gradually, C3oT can leverage the shorter CoT to activate the reasoning abilities learned during the conditioned training phase from longer CoT, thereby widening the performance gap with Short CoT. This continues until the information and interpretability in the compressed, shorter CoT reach a certain threshold, satisfying the requirements for $C 3 o T$ to fully activate the CoT’s capabilities, achieving results close to those of Long CoT. At the same time, the performance of Short CoT also sees a significant improvement.

# What is the impact of mixed conditions training on C3oT?

In the previous part, we explored the impact of different compression rates of the training set on C3oT. Specifically, during the training phase, for shorter CoT corresponding to various compression rates, we combined them respectively with the longer CoT to form the class-conditioned dataset. Then, we employed the conditioned training method mentioned in the Method section. However, an intuitive idea is to use various distinct initial prompt tokens before instructions to condition shorter CoT corresponding to different compression rates in the training set, and combine them together with the longer CoT to form a mixed class-conditioned dataset. To investigate this, we implement Mixed Conditions. Specifically, we expand the conditions into the following form:

# [Long Condition]

Answer and provide a detailed thought process:

[Short Cond.1] [Short Cond.6]

Answer and provide a thought process in compression level of 1:

Answer and provide a thought process in compression level of 6:

where Short Cond.1 to Cond.6 denote compression rates of the training set ranging from $50 \%$ to $100 \%$ , respectively.

From Figure 2a and Figure 2b, we can see that Mixed Conditions outperforms $C 3 o T$ across all training set compression rates and even surpasses Long CoT at $50 \%$ compression rate. This demonstrates that training with a mix of data at various compression levels through the class-conditioned policy can lead to mutually beneficial effects.

Moreover, Mixed Conditions can generate CoT with different compression levels in the conditioned inference phase by using different initial prompt tokens before instruction. It is worth mentioning that in the Figure 2a and Figure 2b, the horizontal axis represents the compression rate of the training set for $C 3 o T$ , and for Mixed Conditions, it represents the distinct initial prompt tokens before instruction corresponding to that compression rate used during the conditioned inference.

# Can C3oT select the appropriate compression rate on its own?

Through the exploration of the previous two parts, we’ve observed that the length of CoT required to correctly answer questions varies with the difficulty of the questions. For simple questions, the model can provide answers directly even without the CoT (Compression Rate $\begin{array} { r }  : = 1 0 0 \% \ \end{array}$ ). However, as the questions become more challenging, the model needs to undergo a more complex intermediate reasoning process to arrive at the correct answer. Therefore, the overall accuracy of the model tends to decrease as the compression rate increases. In this final part, we aim to explore the capability of C3oT to automatically select the most appropriate compression rate. For the most appropriate compression rate, we refer to the highest CoT compression rate at which the model still can accurately answer questions.

Inspired by Orca 2 (Mitra et al. 2023), firstly, we arrange the training sets obtained in previous parts by their compression rates, from highest to lowest. Then, we process each training set in order, randomly dividing each into five parts. We use compressed, shorter CoT in four parts to train a model and predict outcomes on the remaining part. This step is repeated three times with different random seeds to obtain three prediction results for each sample in the training set. For each sample, if it is correctly predicted in at least one out of these three attempts, we consider the sample as correctly answerable by the model under the current compression rate and include it in the final training set. Conversely, if a sample is too difficult to be correctly predicted under the current compression rate, it is carried forward to the next round at a lower compression rate for further assessment. Finally, we train C3oT using the conditioned training method mentioned in the Method section with the final training set composed in the above steps and name the result $C 3 o T _ { A d a p t }$ in Table 3.

The results from Table 3 show that $C 3 o T _ { A d a p t }$ significantly outperforms $C 3 o T _ { G P T - 4 }$ in both accuracy and compression rate. This demonstrates that after training C3oT with data at the most appropriate compression rate, the model has learned to autonomously determine the most efficient length of the CoT for questions of varying complexity during the inference phase. This conclusion also opens up a new avenue for the further application of C3oT.

# Conclusion

We introduce C3oT, a simple but effective method for CoT compression. Through comprehensive experiments and analyses, we demonstrate that our approach holds significant practical implications, as it enables models, which are trained using complex, longer CoT to enhance reasoning capabilities, to be applied in time-sensitive real-world applications.