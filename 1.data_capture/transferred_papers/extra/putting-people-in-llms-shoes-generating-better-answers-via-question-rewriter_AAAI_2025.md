# Putting People in LLMs’ Shoes: Generating Better Answers via Question Rewriter

Junhao Chen, Bowen Wang\*, Zhouqiang Jiang, Yuta Nakashima

Osaka University, Japan junhao, zhouqiang @is.ids.osaka-u.ac.jp, wang, n-yuta @ids.osaka-u.ac.jp

# Abstract

Large Language Models (LLMs) have demonstrated significant capabilities, particularly in the domain of question answering (QA). However, their effectiveness in QA is often undermined by the vagueness of user questions. To address this issue, we introduce single-round instance-level prompt optimization, referred to as question rewriter. By enhancing the intelligibility of human questions for black-box LLMs, our question rewriter improves the quality of generated answers. The rewriter is optimized using direct preference optimization based on feedback collected from automatic criteria for evaluating generated answers; therefore, its training does not require costly human annotations. The experiments across multiple black-box LLMs and long-form question answering (LFQA) datasets demonstrate the efficacy of our method. This paper provides a practical framework for training question rewriters and sets a precedent for future explorations in prompt optimization within LFQA tasks.

# Code — https://github.com/3244we/Question-Rewriter Extended version — https://arxiv.org/abs/2408.10573

What causes random chest It could be stress pains that come and go? or muscle strain. Black-box Human LLM Ask Original Original   
Directly Question Answer Multi- 白 Black-box Human round LLM   
Mound Qurginan Rewritien Anster   
Meturd Qriginal Rewrten Q Aester Human Rewriter Black-box LLM What are the possible Intermittent chest pain can be caused by various faccauses of intermittent tors,including angina, gastroesophageal reflux diseachest pain? se (GERD), musculoskeletal issues, or anxiety.

# Introduction

Large language models (LLMs) have incorporated extensive world knowledge through learning vast publicly available corpora (Roberts, Raffel, and Shazeer 2020). It becomes increasingly common for people to seek knowledge from LLMs, especially in fields such as medicine and law (Atallah et al. 2023; Harrington 2023; Wang et al. 2024). However, a near-paradoxical issue arises: People ask questions to get knowledge, while lack of knowledge often leads to poorly formulated or vague questions, hindering LLMs from providing precise answers (Kim et al. 2023; Zhang et al. 2024). Fine-tuning can enhance LLMs’ ability to understand vague questions, but most popular LLMs are black-box models, and their parameters are inaccessible. Thus, a step of transforming user questions into a format that LLMs can understand better, known as question rewriting, is crucial for question answering (QA).

Question rewriting is closely related to prompt optimization. A prompt is an input to LLMs that guides them in generating a specific response, including a question (possibly with some instructions), a conversation history, etc. (Liu et al. 2023). Question rewriting is prompt optimization solely for questions. Previous work on prompt optimization primarily focused on optimizing task-level prompts. They decompose prompts into task-level instructions and instance-level inputs, optimizing task-level instructions for better performance across all instances of the task (Fernando et al. 2023; Guo et al. 2023; Kong et al. 2024).

Recent studies have shown that directly optimizing the prompts at the instance level offers more flexibility in prompt editing tailored for a given prompt(Lin et al. 2024) and can lead to better responses (Srivastava et al. 2023). By obtaining feedback from humans or LLMs, they iteratively refine a given prompt, which requires multi-round interactions. In addition, previous prompt optimization is mainly applied to arithmetic reasoning (Cobbe et al. 2021) and short-form question answering (SFQA) (Kwiatkowski et al. 2019), where the latter involves answers in few words.

These tasks do not necessarily cover real-world QA scenarios.

This paper proposes single-round instance-level prompt optimization, referred to as question rewriter, aiming at optimizing questions for long-form question answering (LFQA), which is closer to real-world QA scenarios (Bhat et al. 2023). The question rewriter serves as an intermediary between users and black-box LLMs to rewrite questions, as shown in Figure 1. When a user submits a question, our question rewriter scutches it to contextualize the question for black-box LLMs to generate a more accurate answer.

The key to our method lies in obtaining supervising signals for optimizing the question rewriter. Different from arithmetic reasoning and SFQA, there is no unique best answer for LFQA questions; therefore, obtaining the optimal rewritten question as the ground truth to train a question rewriter is not trivial (Radford and Narasimhan 2018). Our method, in contrast, assumes the presence of automatic criteria to evaluate generated answers, which are typically provided in LFQA datasets, and uses them to identify better and worse rewritten questions. With such supervising signals, we propose to use direct preference optimization (DPO) (Rafailov et al. 2023) to train our question rewriter. Thanks to this design choice, our method does not necessitate costly human interactions used in reinforcement learning from human feedback (Bai et al. 2022) and a differentiable reward model in proximal policy optimization (Schulman et al. 2017).

Contribution. Our question rewriter is single-round prompt optimization without human interventions, which has not been explored so far. We experimentally show across multiple datasets and LLMs for answer generation that, with optimization by automatic evaluation criteria, the question rewriter can generate questions that end up with better answers. Intiguingly, our analysis implies that the question rewriter learns to generate non-leading and concise questions in a professional tone, which aligns with our intuitions when engineering a prompt.

Initial Generate Rewriter LLM Qursinals Rewritten Rewritten (LFQADataset) Questions Answers Same: DirectImrefternce Buter & WPares Optimized Initial Automatic Rewriter Rewriter Evaluation

Early work for prompt optimization focused on white-box models (Shin et al. 2020; Shi et al. 2023; Li and Liang 2021; Lester, Al-Rfou, and Constant 2021; Zhong, Friedman, and Chen 2021). Due to the prevalent nature of black-box models such as GPT-3 (Patel et al. 2023) and Claude (Anthropic 2024), the following work targeted at these black-box models. Most works decomposed prompts into task-level (i.e., instructions) and instance-level (i.e., specific queries) and optimizing only task-level instructions. Some work assumed that input (i.e., text embeddings) and output (i.e., logits) are accessible and leveraged them to optimize prompts (Sun et al. 2022b,a; Chai et al. 2022). Other recent work has attempted to remove this assumption. Prasad et al. (2023) and Pryzant et al. (2023) evaluate task-level instructions with small edits (e.g., replacing some phrases with their synonyms) and find better ones step by step. Evolutionary algorithms (Fernando et al. 2023; Guo et al. 2023), reinforce learning (Diao et al. 2023; Kong et al. 2024), and planningbased methods (Wang et al. 2023) have also been adopted.

Some work fully utilized the inherent capabilities of LLMs to refine prompts. Zhou et al. (2023) leverages an LLM to generate and refine the prompts iteratively, and Yang et al. (2023) provides the prompt optimization trajectory to an LLM, allowing for discovering inherent patterns and optimizing the prompt progressively. Other notable efforts, such as InstructZero (Chen et al. 2023) and INSTINCT (Lin et al. 2023), have transformed black-box optimization into an iterative optimization problem with white-box LLMs.

All these works are task-level prompt optimization. Recent studies have shown that instance-level prompt optimization results in better performance by offering more specific prompts (Lin et al. 2024; Srivastava et al. 2023). These works iteratively refine a prompt at the instance level by obtaining feedback from humans or ChatGPT. We also adopt the instance-level approach, but unlike the other instance-level method, ours does not require feedback from LLMs or humans for multi-round optimization. We instead use preference optimization (Rafailov et al. 2023) for our question rewriter that can optimize the questions at once without any feedback nor iterative refinement. We evaluate our method over LFQA tasks, whereas previous works mainly use arithmetic reasoning (Cobbe et al. 2021) and SFQA (Kwiatkowski et al. 2019). LFQA is closer to realworld QA scenarios (Bhat et al. 2023) and can highlight differences in the generated text.

# Method

Our question rewriter $R$ learns to rewrite questions so that an LLM can give a better answer for a rewritten question. We design our method under the assumption that the goodness of the answer to a certain question is automatically judgeable. With this assumption, we can sample rewritten questions and the corresponding answers using LLMs, contrasting them to learn desirable questions.

# Related Work

Figure 2 shows the pipeline of our method. Let $\mathcal { D } \ =$ $\{ ( q , \bar { a } ) \}$ denote a training dataset of pairs of question $q$ and answer $a$ , with an associated set ${ \mathcal { C } } = \{ c \}$ of automatic evaluation criteria $c$ . Firstly, our pipeline rewrites questions for $q \in { \mathcal { D } }$ . Then, $c \in { \mathcal { C } }$ evaluates the rewritten questions to make a set $\mathcal { P } = \{ ( \hat { q } , \check { q } ) \}$ of pairs of a better question $\hat { q }$ and a worse question $\check { q }$ . Finally, we use DPO (Rafailov et al. 2023) to train $R$ with $\mathcal { P }$ .

# Sampling Rewritten Questions

We use a pre-trained LLM $R _ { 0 }$ to sample rewritten questions without fine-tuning as it offers sufficient capability for initial rewriting solely through prompt engineering. We use top-p sampling (Radford et al. 2019) to generate $K$ different rewritten questions $\mathcal { Q } ( q ) = \{ r _ { k } ( q ) | k = 1 , \ldots , K \}$ of $q \in { \mathcal { D } }$ , where $r _ { k } ( q )$ is the $k$ -th rewritten question for $q$ , with the predefined prompt $t$ , i.e., $t$ equals:

Rewriting question to make it more understandable, just give me the rewritten question without any other word:

followed by $q$

# Making Better and Worse Question Pairs

Datasets for LFQA typically provide methods for evaluating generated answers. For instance, some datasets (Manes et al. 2024) are inspired by FActScore (Min et al. 2023) and annotate the facts required to answer each question, allowing LLMs to assess whether the corresponding facts are implied by or contradict the generated answers to derive scores for comprehensiveness and precision. Other datasets (Lin, Hilton, and Evans 2022) offer extensive binary annotations used to train classifiers to determine whether answers conform to certain attributes like truthfulness. Additionally, some datasets1 are in the form of preference datasets, which provide pairs of samples, where one is better than the other. Such datasets can be used to train reward models to evaluate whether answers align with human preferences. We can use these automatic evaluation criteria as $\mathcal { C }$ to evaluate rewritten questions. Such automatic evaluation criteria substitute the human feedback typically used in previous methods (Rafailov et al. 2023) to make $\mathcal { P }$ .

Let $L$ denote a pre-trained LLM for answer generation. For a question-answer pair $( q , a ) \in \mathcal { D }$ , we generate answers for all $\bar { q } ^ { \prime } \in \mathcal { Q } ( q )$ as $a ^ { \prime } = L ( \dot { q } ^ { \prime } )$ . We also generate the answer to the original question $q$ as $\tilde { a } = L ( q )$ , which serves as the baseline to judge the goodness of rewritten questions.

To make better-worse pairs, we first identify $q ^ { \prime } \in \mathcal { Q } ( q )$ that gives better answers and worse answers, collectively denoted by $\mathcal { Q } _ { + } ( \boldsymbol { q } )$ and $\mathcal { Q } _ { - } ( q )$ , respectively. Observing that criterion $c \in { \mathcal { C } }$ is often numerical2, we judge $q ^ { \prime }$ is better if $a ^ { \prime }$ is larger than or equal to $\tilde { a }$ in terms of all criteria and $a ^ { \prime }$ is larger than $\tilde { a }$ at least one criterion,3 i.e.,

$$
\begin{array} { r l r } & { } & { \mathcal { Q } _ { + } ( q ) = \{ q ^ { \prime } \in \mathcal { Q } ( q ) | \forall _ { c \in \mathcal { C } } c ( a ^ { \prime } ) \geq c ( \tilde { a } ) , } \\ & { } & { \exists _ { c \in \mathcal { C } } c ( a ^ { \prime } ) > c ( \tilde { a } ) \} . } \end{array}
$$

$\mathcal { Q } _ { - } ( q )$ is defined in the opposite way, i.e., $a ^ { \prime }$ should be always worse than or equal to $\tilde { a }$ and $a ^ { \prime }$ should be worse than $\tilde { a }$ for at least one criterion.

A better and worse question pair is created by picking one rewritten question from $\mathcal { Q } _ { + } ( \boldsymbol { q } )$ and the other from $\bar { \mathcal { Q } _ { - } } ( q )$ . As we wish to train a model $R$ to generate good questions, we rank rewritten questions in $\mathscr { Q } _ { + }$ according to a certain composition of all $\bar { c } \in \mathcal { C }$ ,4 and use the top $N _ { + }$ questions. The set of chosen better questions is denoted by $\mathcal { Q } _ { + } ^ { \star } ( q )$ . On the other hand, to avoid following DPO training only with easy negatives, we randomly choose $N _ { - }$ questions in $\dot { \mathcal { Q } } _ { - } ( q )$ and pair each of them with $\hat { q } \in \mathcal { Q } _ { + } ^ { \star } ( q )$ . Formally, letting $s$ denote randomly sampled $N _ { - }$ questions from $\mathcal { Q } _ { - } ( q )$ without replacement, the set $\mathcal { P } ( q )$ of better and worse question pairs for $q$ is given by:

$$
\mathcal { P } ( q ) = \{ ( \hat { q } , \check { q } ) | \hat { q } \in \mathcal { Q } _ { + } ^ { \star } ( q ) , \check { q } \in \mathcal { S } ( \mathcal { Q } _ { - } ( q ) ) \} .
$$

$\mathcal { P } ( q )$ contains $N _ { + } \times N _ { - }$ pairs when $\left| \mathscr { Q } _ { + } ( q ) \right| \geq N _ { + }$ and $| \mathcal { Q } _ { - } ( q ) | \geq N _ { - }$ ; otherwise, $| \mathcal { P } ( q ) |$ is smaller. The comparison of the different sampling combination $\cdot ^ { 5 }$ for $\mathcal { P } ( q )$ can be found in the extended version appendix.

# Optimizing Question Rewriter

Training our question rewriter $R$ is costly when it requires human feedback or a reward model that learns the human feedback. Fortunately, LFQA tasks typically offer automatic criteria to evaluate the goodness of generated answers. We can use the criteria to (indirectly) evaluate rewritten questions by evaluating their answers.

Let $P _ { R } ( q ^ { \prime } | t , q )$ denote the average probability of tokens in $q ^ { \prime }$ given the predefined prompt $t$ and the original question $q$ with $R$ , given by:

$$
P _ { R } ( \boldsymbol { q } ^ { \prime } ) = \frac { 1 } { T } \sum _ { k = 1 } ^ { K } p _ { R } ( w _ { k } | t , \boldsymbol { q } , w _ { 1 : k - 1 } ) ,
$$

where $K$ is the length of $q ^ { \prime }$ $\prime ^ { \prime } ; p _ { R } ( w _ { t } | t , q , w _ { 1 : k - 1 } )$ is the probability of token $\boldsymbol { w } _ { t }$ given $t , q ,$ , and a set $w _ { 1 : k - 1 }$ of tokens generated by the $( k - 1 )$ -th step (i.e., $q ^ { \prime } = w _ { 1 : K } )$ . $P _ { R _ { 0 } } ( q ^ { \prime } )$ is defined likewise for the initial question rewriter $R _ { 0 }$ . DPO’s training loss is given by:

$$
L = - \mathbb { E } \left[ \log \sigma \left( \beta \log \frac { P _ { R } ( \hat { q } ) } { P _ { R _ { 0 } } ( \hat { q } ) } - \beta \log \frac { P _ { R } ( \check { q } ) } { P _ { R _ { 0 } } ( \check { q } ) } \right) \right]
$$

where $\sigma$ is sigmoid, and $\beta$ is a hyperparameter that controls how much $R$ deviates from $R _ { 0 }$ , and the expectation is computed over $q \sim \mathcal { D }$ and $( \hat { q } , \check { q } ) \sim \mathcal { P } ( q )$ .

To mitigate the risk of overfitting, we use dropout in the model. Also, the original LFQA dataset is divided into three parts: training, validation, and testing. $R$ is trained on the training set (i.e., $\mathcal { D }$ ) for one epoch, and we select the best model that most prefer $\hat { q }$ ’s to $\check { q }$ ’s. Specifically, we define preference score PS as

$$
\mathrm { P S } = \mathbb { E } \left[ \mathbf { 1 } [ P _ { R } ( \boldsymbol { \hat { q } } | t , q ) > P _ { R } ( \boldsymbol { \check { q } } | t , q ) ] \right] ,
$$

where $\mathbf { 1 } [ \cdot ]$ gives 1 if the given condition is satisfied, and otherwise 0; the expectation is computed for all the $q$ from the validation set and $( \hat { q } , \check { q } ) \sim \mathcal { P } ( q )$ .

<html><body><table><tr><td>Dataset</td><td>Training</td><td>Validation</td><td>Testing</td><td>Total</td></tr><tr><td>K-QA</td><td>101</td><td>50</td><td>50</td><td>201</td></tr><tr><td>TruthfulQA</td><td>407</td><td>205</td><td>205</td><td>817</td></tr><tr><td>OASST1QA</td><td>1,000</td><td>93</td><td>93</td><td>1,186</td></tr></table></body></html>

Table 1: Statistics of LFQA datasets used to evaluate our method. Columns for Training, Validation, and Testing give the numbers of samples in respective dataset splits.

# Experiments

# Experimental Setup

Dataset We evaluate three distinct LFQA datasets, each equipped with automated evaluation criteria.

K-QA (Manes et al. 2024), sourced from the medical domain, is designed to evaluate the factual comprehensiveness and precision of answers through metrics $S _ { \mathrm { c o m p } }$ and $S _ { \mathrm { c o n t } }$ , employing a FActScore type method (Min et al. 2023). To combine these two criteria for ranking rewritten questions in $\mathcal { Q } + ( q )$ , we first use $S _ { \mathrm { c o n t } }$ to rank them, and then use $S _ { \mathrm { c o m p } }$ if $S _ { \mathrm { c o n t } }$ is the same for multiple questions.

TruthfulQA (Lin, Hilton, and Evans 2022), covering multiple domains including health and law, assesses the truthfulness $( S _ { \mathrm { t r u t h } } )$ and informativeness $( S _ { \mathrm { i n f o } } )$ of answers. The evaluation criteria are implemented as binary classifiers. We use the probabilities for positive classes (truthful for $S _ { \mathrm { t r u t h } }$ and informative for $S _ { \mathrm { i n f o } } \mathrm { \Omega } _ { \mathrm { \Omega } } ^ { \mathrm { ~ } }$ ). An overall score $( S _ { \mathrm { o v e r a l l } } )$ is computed as the product of these scores. For better rewritten pair ranking, we use $S _ { \mathrm { o v e r a l l } }$ .

OASST1QA, derived from the multi-turn dialogue alignment dataset $\mathrm { O A S S T 1 ^ { 6 } }$ , incorporates a criterion $S _ { \mathrm { p r e f } }$ that measures human preference for answers using a pre-trained reward model. This dataset provides a single criterion for evaluation (i.e., $| { \mathcal { C } } | = 1 { \dot { } }$ ), so we directly use $S _ { \mathrm { p r e f } }$ for ranking better rewritten questions.

More details about these datasets and their evaluation criteria can be found in the extended version appendix. Table 1 summarizes the statistics on the datasets.

LLMs The base model of our question rewriter $R$ (and $R _ { 0 }$ is Llama3-8B-instruct, and the answer generation model $L$ is also Llama3-8B-instruct because it is one of the most powerful but small LLMs. $R$ is fine-tuned with our method, while $L$ is frozen. Subsequently, we evaluate the generalizability of $R$ on multiple answer generation LLMs, including Llama3-8B-instruct7, mistral-7B-instruct- $\mathrm { v } 0 . 2 ^ { 8 }$ , zephyr-7Bbeta9, gemma-1.1-7B-it10, gpt-3.5-turbo-1106, and gpt-4o2024-05-13. They will be referred to as Llama3-8B, Mistral7B-v0.2, Zephyr-7B-beta, Gemma-1.1-7B, GPT-3.5, and GPT-4o, respectively. It is worth noting that we only use $L$ as Llama3-8B-instruct to build $P$ for training $R$ , and then test the generalizability of $R$ on other models.

Hyperparameters We borrowed open-source code for DPO training over all three datasets11, which also provides the code for supervised fine-tuning of automatic criteria $S _ { \mathrm { t r u t h } }$ and $S _ { \mathrm { i n f o } }$ for TruthfulQA. During DPO training, we set the dropout rate to 0.8, the training batch size to 32, and the testing batch size to 64, maintaining all other parameters at their default settings in the source code. For sampling rewritten questions, we use top- $\cdot \mathbf { p }$ sampling, where the cumulative probability for top- $\cdot \mathbf { p }$ sampling is set to 0.999, and the temperature of $R _ { 0 }$ is 1, to ensure diversity. We sample 100 unique rewritten questions for each of the original questions and terminate the sampling after 10,000 attempts. $N _ { + }$ and $N _ { - }$ are defaulted to (10, 20), (5, 10), and (4, 5) in K-QA, TQA, and OQA respectively. When multiplied by the number of samples in the corresponding training sets, they are around 20,000. The maximum token length is set to 512 during feedback collection and testing. During testing, to ensure reproducibility, we generate answers using greedy sampling.

Device All our testing and training, except for the DPO training of OASST1QA, are conducted on a system equipped with four NVIDIA A100-40GB-PCIE. Due to the extensive length of OASST1QA, we only used samples whose question plus the prompt $t$ and rewritten questions $q ^ { \prime }$ for question rewriting is less than or equal to 512 tokens and conducted the DPO training on a system with four NVIDIA A100-80GB-PCIe.

Baselines In our experiments across different datasets and models, we compare our method with both the original questions and the initial Llama3-8B-instruct rewriter (without fine-tuning). To demonstrate the effectiveness of our approach, we also compare it with the widely used task-level prompting method, Zero-Shot Chain-of-Thought (Zero-Shot CoT) (Kojima et al. 2022). Other instance-level methods, such as PRoMPTed, require multiple rounds of interactions with humans or LLMs to obtain feedback and iteratively modify the prompt or question during inference, which is extremely costly in our QA scenarios. Therefore, we only perform comparisons with PRoMPTed and other question rewriting methods on the K-QA dataset and Llama3-8Binstruct.

# Result across Models and Datasets

Table 2 summarizes our experimental results over three LFQA datasets. Our method demonstrates superior performance in most combinations of LLMs and datasets.

For the K-QA dataset, our method consistently shows the highest $S _ { \mathrm { c o m p } }$ scores across all models, especially with GPT4o, where the improvement is most significant. Furthermore, it achieves the lowest $S _ { \mathrm { c o n t } }$ with half of the models and the second lowest in the rest. Notably, our method trained an effective question rewriter using only 151 samples (i.e., training set plus validation set), implying that our method requires only a small number of annotated samples in a realworld QA scenario. Table 3 shows an example from K-QA, in which $S _ { \mathrm { c o m p } }$ increases and $S _ { \mathrm { c o n t } }$ decreases after rewriting the original question.

<html><body><table><tr><td></td><td></td><td colspan="2">K-QA</td><td colspan="3">TruthfulQA</td><td>OASST1QA</td></tr><tr><td>Model</td><td>Method</td><td>Scomp↑</td><td>Scont</td><td>Struth ↑</td><td>Sinfo ↑</td><td>Soveral ↑</td><td>Sprer ↑</td></tr><tr><td rowspan="4">Llama-3-8B</td><td>Original</td><td>0.4573</td><td>0.4400</td><td>0.7683</td><td>0.9664</td><td>0.7397</td><td>0.8654</td></tr><tr><td>Zero-Shot CoT</td><td>0.4579</td><td>0.4000</td><td>0.7476</td><td>0.9306</td><td>0.6938</td><td>0.8838</td></tr><tr><td>Initial Rewriter</td><td>0.4262</td><td>0.5000</td><td>0.7914</td><td>0.9564</td><td>0.7566</td><td>0.8748</td></tr><tr><td>Ours</td><td>0.4600</td><td>0.4000</td><td>0.8059</td><td>0.9668</td><td>0.7789</td><td>0.9104</td></tr><tr><td rowspan="4">Mistral-7B-v0.2</td><td>Original</td><td>0.4374</td><td>0.2200</td><td>0.8364</td><td>0.9834</td><td>0.8227</td><td>0.8281</td></tr><tr><td>Zero-Shot CoT</td><td>0.4428</td><td>0.2800</td><td>0.8423</td><td>0.9737</td><td>0.8199</td><td>0.8908</td></tr><tr><td>Initial Rewriter</td><td>0.4177</td><td>0.3400</td><td>0.7916</td><td>0.9689</td><td>0.7670</td><td>0.8381</td></tr><tr><td>Ours</td><td>0.4899</td><td>0.2600</td><td>0.8474</td><td>0.9788</td><td>0.8296</td><td>0.8762</td></tr><tr><td rowspan="4">Zephyr-7B-beta</td><td>Original</td><td>0.4396</td><td>0.3400</td><td>0.7644</td><td>0.9826</td><td>0.7518</td><td>0.6369</td></tr><tr><td>Zero-Shot CoT</td><td>0.4333</td><td>0.3200</td><td>0.7081</td><td>0.9705</td><td>0.6867</td><td>0.7606</td></tr><tr><td>Initial Rewriter</td><td>0.4666</td><td>0.2200</td><td>0.7353</td><td>0.9723</td><td>0.7167</td><td>0.6417</td></tr><tr><td>Ours</td><td>0.4702</td><td>0.2600</td><td>0.7709</td><td>0.9775</td><td>0.7528</td><td>0.7768</td></tr><tr><td rowspan="4">Gemma-1.1-7B</td><td>Original</td><td>0.4010</td><td>0.5400</td><td>0.6780</td><td>0.9716</td><td>0.6554</td><td>0.7428</td></tr><tr><td>Zero-Shot CoT</td><td>0.4516</td><td>0.5000</td><td>0.7216</td><td>0.9454</td><td>0.6752</td><td>0.8632</td></tr><tr><td>Initial Rewriter</td><td>0.4928</td><td>0.4400</td><td>0.6415</td><td>0.9617</td><td>0.6124</td><td>0.7955</td></tr><tr><td>Ours</td><td>0.4956</td><td>0.2200</td><td>0.7224</td><td>0.9558</td><td>0.6888</td><td>0.9034</td></tr><tr><td rowspan="4">GPT-3.5-turbo</td><td>Original</td><td>0.4909</td><td>0.3200</td><td>0.7451</td><td>0.9804</td><td>0.7303</td><td>0.7294</td></tr><tr><td>Zero-Shot CoT</td><td>0.4748</td><td>0.1600</td><td>0.7413</td><td>0.9781</td><td>0.7237</td><td>0.8222</td></tr><tr><td>Initial Rewriter</td><td>0.4454</td><td>0.3000</td><td>0.7325</td><td>0.9768</td><td>0.7164</td><td>0.7353</td></tr><tr><td>Ours</td><td>0.4978</td><td>0.2800</td><td>0.7574</td><td>0.9682</td><td>0.7309</td><td>0.8994</td></tr><tr><td rowspan="4">GPT-40</td><td>Original</td><td>0.5167</td><td>0.2800</td><td>0.8812</td><td>0.9790</td><td>0.8631</td><td>0.8532</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ziti- Shot CoT</td><td>0.4903</td><td>0.3000</td><td>0.8739</td><td>0.9611</td><td>0.8400</td><td>0.8471</td></tr><tr><td>Ours</td><td>0.6253</td><td>0.2400</td><td>0.8880</td><td>0.9722</td><td>0.8641</td><td>0.9100</td></tr></table></body></html>

Table 2: Comparison of different question rewriting methods across multiple datasets and LLMs for answer generation. We use automatic evaluation criteria associated with each dataset. Bold indicates the best method. Underline indicates the second-best.

![](images/ab71a589d5e3f11b1267bbb63bdbc0c54825bb33349f5ae2b40a496ba50ecfe6.jpg)  
Figure 3: Evaluating the impact of $N _ { + }$ and $N _ { - }$ on the performance over K-QA and OASST1QA.

For the TruthfulQA dataset, all methods generally reduce the informativeness (i.e., $S _ { \mathrm { i n f o } } )$ ) of the answers, and only ours gains the truthfulness score (i.e., $S _ { \mathrm { t r u t h } } \mathrm { \dot { \Omega } }$ ). This is typical behavior in the TruthfulQA dataset as these two criteria exhibit a trade-off relationship (Lin, Hilton, and Evans 2022). Our method can increase $S _ { \mathrm { o v e r a l l } }$ , while the others reduce it.

For OASST1QA, our method outperforms others except Mistral-7B-v0.2, where Zero-Shot CoT performs best.

Overall, our method excels in all metrics and all datasets, not only on the Llama3-8B-instruct used for training $R$ but also on the other LLMs, demonstrating their generalizability to answer generation models. Thanks to this generalizability, the rewriter $R$ can be deployed without adjusting to individual (black-box) answer generation models.

# Impact of $N _ { + }$ and $N _ { - }$

The number of better and worse question pairs $\begin{array} { r l } { \mathcal { P } } & { { } = } \end{array}$ $\cup _ { q \in \mathcal { D } } \mathcal { P } ( q )$ is determined by $N _ { + }$ , $N _ { - }$ , and $| \mathcal D |$ (i.e., $| \mathcal { P } | =$ $\hat { N _ { + } } \times \hat { N _ { - } } \times | \mathcal { D } | )$ , where the choice of $N _ { + }$ and $N _ { - }$ is rather arbitrary. To explore the impact of $N _ { + }$ and $N _ { - }$ on the performance, we evaluated our method with varying $N _ { + }$ and $N _ { - }$ over K-QA OASST1QA with Llama- $\cdot 3 \mathrm { - } 8 \mathrm { B } ^ { \bar { 1 } 2 }$ . The results are summarized in Figure 3.

For the K-QA dataset, finding the optimal values of $N _ { + }$ and $N _ { - }$ is not easy since multiple criteria are involved, but we would say $| \mathcal { P } |$ between $5 \mathrm { k }$ and $2 0 \mathrm { k }$ works well. Beyond $2 0 \mathrm { k }$ , $S _ { \mathrm { c o m p } }$ starts to decrease, and $S _ { \mathrm { c o n t } }$ spikes. In OASST1QA, the performance increases along with $| \mathcal { P } |$ increases by $2 0 \mathrm { k }$ , then it decreases. The performance drops when $| \mathcal { P } |$ larger than $2 0 \mathrm { k }$ is attributed to overfitting during DPO training. These results highlight the necessity of adjusting $N _ { + }$ and $N _ { - }$ for each dataset.

Table 3: An example of original and rewritten questions in K-QA with their answers. Text highlighted in green is consistent with the must-to-have fact, whereas red contradicts the must-to-have fact. Omitted text in answers, represented by “...”, are irrelevant to any must-to-have fact.   

<html><body><table><tr><td>Original Question</td><td>How do I know if it's a canker sore versus herpes?</td></tr><tr><td rowspan="5">Original Answer Scomp ↑: 2/6 Scont ↓: 1</td><td>Canker sores and cold sores (caused by herpes simplex virus) are both types of oral lesions but they have distinct characteristics.</td></tr><tr><td>Canker Sore:</td></tr><tr><td>· Location: Canker sores typically occur on the inside of the mouth, tongue,lips or throat. Cold Sore (Herpes Simplex Virus):</td></tr><tr><td>· Location: Cold sores typically occur on the lips,around the mouth, or on the nose.</td></tr><tr><td></td></tr><tr><td>Rewritten Question</td><td>How can I distinguish between acanker sore and a herpes outbreak based on their symptoms and characteris- tics？</td></tr><tr><td rowspan="4">Rewritten Answer Scomp ↑: 3/6 Scont ↓: 0</td><td>Canker Sores (Aphthous Ulcers): Location: Typically occur on the inside of the mouth, tongue, lips, or throat.</td></tr><tr><td>Herpes Simplex Virus (HSV) Outbreaks:</td></tr><tr><td>· Location: Can occur anywhere on the body but most commonly on the mouth,lips, or genital area.</td></tr><tr><td>· Appearance: Small, fluid-filled blisters that eventually crust over and form a scab.</td></tr></table></body></html>

Table 4: Comparison of different question rewriting methods. Bold indicates the optimization method is better than the original results. Orig. indicates the original results. T5 indicates the replicated result of ill-formed rewriting.   

<html><body><table><tr><td>Metric</td><td>Orig.</td><td>Ours</td><td>PRewrite</td><td>T5</td><td>PRoMPTed</td></tr><tr><td>Scomp ↑ 0.4573 0.4600</td><td></td><td></td><td>0.4409</td><td>0.4160</td><td>0.5012</td></tr><tr><td>Scont↓</td><td>0.44000.4000</td><td></td><td>0.3600</td><td>0.3600</td><td>0.5400</td></tr></table></body></html>

# Comparison with Other Methods

To validate the superiority of our method, we compared it with other question rewriting methods on the K-QA dataset and Llama3-8B-instruct. In addition to the PRoMPTed method mentioned above, we considered two other closely related methods. The first method employs a transformer model to rewrite ill-formed questions into well-structured ones, while it does not evaluate the impact on answer quality (Chu et al. 2020). We replicated this method using T5- Flan-base and performed better on their original dataset. The second method, PRewrite (Kong et al. 2024), is an RL-based task-level prompt optimization method that uses RLHF, requiring a differentiable reward. However, some LFQA datasets, such as K-QA, lack differentiable rewards. To address this, we implemented DPO, a variant of RLHF, to replicate it. We used prompts from it:

Rewrite the following instruction via rephrasing and/or adding specific requirements. Add instructions that would be helpful to solve the problem correctly. Output the new instruction only.

Table 5: Performance of rewriters across datasets on Llama3-8B-instruct: OQA represents OASST1QA, Rw-K, Rw-T, and Rw-O represent rewriters trained on K-QA, TruthfulQA and OASST1QA, respectively. Bold indicates the rewriter performing better than the corresponding original for this LLM.   

<html><body><table><tr><td colspan="3">K-QA</td><td colspan="3">TruthfulQA</td><td>OQA</td></tr><tr><td>Rewriter</td><td>Scomp ↑ Scont ↓</td><td></td><td>Struth ↑ Sinfo ↑Soverall 个</td><td></td><td></td><td>Sprer ↑</td></tr><tr><td>Original</td><td>0.4573</td><td>0.4400</td><td>0.7683</td><td>0.9664</td><td>0.7397</td><td>0.8654</td></tr><tr><td>Rw-K</td><td>0.4600</td><td>0.4000</td><td>0.7834</td><td>0.9535</td><td>0.7454</td><td>0.8759</td></tr><tr><td>Rw-T</td><td>0.4104</td><td>0.2800</td><td>0.8059</td><td>0.9668</td><td>0.7789</td><td>0.8839</td></tr><tr><td>Rw-O</td><td>0.4510</td><td>0.4200</td><td>0.7622</td><td>0.9373</td><td>0.7155</td><td>0.9104</td></tr></table></body></html>

to optimize the original task instruction: “provide the answer:”, and then appended the optimized instruction to the original questions to obtain answers. The optimized instruction can be found in the extended version appendix.

As shown in Table 4, only our method demonstrates improvements in both metrics. While the PRoMPTed method improves $S _ { c o m p }$ , it significantly compromises $S _ { c o n t }$ . This highlights the superiority of our method in balancing these competing objectives and further confirms that our method outperforms other question rewriting methods.

# Cross Dataset Generalizability

We explored the performance of rewriters trained on Llama3-8B-instruct across different datasets. Table 5 shows that rewriters trained in the K-QA and TruthfulQA datasets can optimize the generated answers on the OASST1QA dataset, but each fails on one metric in their respective datasets. In contrast, rewriters trained on the OASST1QA dataset are almost ineffective on the other two datasets. This suggests that training on more complex LFQA datasets, which include multiple automatic evaluation criteria, yields rewriters with better generalizability.

# Discussion

Our question rewriters can significantly improve the representation of questions, making them more likely to obtain higher-quality answers with LLMs. To quantitatively analyze how attributes impact the evaluation criteria of generated answers, we study 50 original questions in KQA’s test set and their rewritten versions, resulting in 100 questions in total. We adopt 10 attributes: non-leadingness, word choice, tone, conciseness, neutrality, grammar and spelling, structure, politeness, clarity, and emotion, which are identified by an LLM. For each attribute and each question, we use GPT4o to assign a score ranging from 1 to 5 to the 100 questions. The definitions of attributes and the prompt templates are available in the extended version appendix.

To explore the important attributes that determine evaluation criteria $S _ { \mathrm { c o m p } }$ and $S _ { \mathrm { c o n t } }$ , we use a random forest regressor that takes the attribute scores as input and predicts either $S _ { \mathrm { c o m p } }$ or $S _ { \mathrm { c o n t } }$ of each question. We train the regressors with these 100 questions and use the predictions again for them.13 The regressors yielded $R ^ { 2 }$ values of 0.56 and 0.55 for $S _ { \mathrm { c o m p } }$ and $S _ { \mathrm { c o n t } }$ , respectively, demonstrating that the attribute scores are significantly correlated with the criteria. The random forest regressors provide feature importance, which, in our case, corresponds to the importance of each attribute.

In addition to the attribute importance, we also examine whether each attribute has a positive or negative impact on the evaluation criteria. To this end, we define the impact by:

$$
I _ { l a } = \hat { S } _ { l a } - \check { S } _ { l a }
$$

where $l ~ \in ~ \{ \mathrm { c o m p , c o n t } \}$ and $a$ is one of the 10 attributes; $\hat { S } _ { l a }$ and $\check { S } _ { l a }$ are the averages of evaluation criterion $S _ { l }$ of questions whose attribute score for $a$ are among the top-50 and bottom-50, respectively. Specifically, $\hat { S } _ { l a }$ is given by:

$$
\hat { S } _ { l a } = \frac { 1 } { 5 0 } \sum _ { S _ { l } \in \hat { S } _ { l a } } S _ { l } ,
$$

where $\hat { S } _ { l a }$ is the set of scores $S _ { l }$ of questions whose attribute scores are among top 50. $\check { S } _ { l a }$ is defined likewise. A higher $\hat { S } _ { l a }$ , for instance, means that the attribute $a$ is positively correlated with $S _ { l }$ .

![](images/d04077aaa25a87227c62c1fc164a8e1a92a73ea79eeeacc7bf9de27a08001987.jpg)  
Figure 4: The importance and impact of attributes conciseness (Conc.), structure (Strt.), word choice (WC), emotion (Emo.), non-leadingness (NL), grammar and spelling (G&S), neutrality (Neut.), tone, clarity (Clar.), and politeness (Pol.). The bar plots are important, while the line plots are impact.

As shown in Figure 4, for $S _ { \mathrm { c o m p } }$ , non-leadingness, word choice, and tone are the most contributing attributes to regression, while for $S _ { \mathrm { c o m p } }$ , non-leadingness, conciseness, and politeness are important. Intriguingly, no-leadingness for example, which means a question does not give some implication of a certain answer, is the most contributing attribute for both criteria. It is not straightforward to interpret this result, but being or not being leading may give some cues about attributes to be used for regression. At least, these attributes are somehow correlated with the evaluation criteria, so the impact of these attributes can be meaningful.

As for the impact, we can see tone gives a positive impact to $S _ { \mathrm { c o m p } }$ , while non-leadingness and conciseness are negatively correlated with $S _ { \mathrm { c o n t } }$ . A higher attribute score for tone means the question is written in a formal language and a professional manner. We can reason that a formal language triggers expert knowledge encompassed in an LLM, which is likely to be written also in a formal language with proper wordings. Meanwhile, being non-leading and concise leads to lower $S _ { \mathrm { c o n t } }$ , which is preferable. These results also make much sense; extra text in a question can lead to knowledge that is still relevant to the extra text but irrelevant to the question. Overall, our importance and impact analysis unveils that our question rewriter learns to generate professional, non-leading, and concise questions, which align with our intuitions, solely through supervision by $S _ { \mathrm { c o m p } }$ and $S _ { \mathrm { c o n t } }$ .

# Conclusions and Future Work

This paper proposes single-round instance-level question optimization for LFQA tasks, coined question rewriter. We employ DPO to optimize the question rewriter with automatic evaluation criteria. Our experimental results demonstrated that our question rewriter can generate questions that give better answers in terms of the automatic evaluation criteria. Meanwhile, although our method demonstrates some degree of cross-domain generalizability, it still has limitations in performance. Therefore, exploring completely domain-agnostic methods would be an interesting direction for future research.