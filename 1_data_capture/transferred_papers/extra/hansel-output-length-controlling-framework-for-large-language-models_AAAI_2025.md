# Hansel: Output Length Controlling Framework for Large Language Models

Seoha Song\*, Junhyun Lee\*†, Hyeonmok Ko

Samsung Research {seoha.song, junhyun8.lee, felix.ko}@samsung.com

# Abstract

Despite the great success of large language models (LLMs), efficiently controlling the length of the output sequence still remains a challenge. In this paper, we propose Hansel, an efficient framework for length control in LLMs without affecting its generation ability. Hansel utilizes periodically outputted hidden special tokens to keep track of the remaining target length of the output sequence. Together with techniques to avoid abrupt termination of the output, this seemingly simple method proved to be efficient and versatile, while not harming the coherency and fluency of the generated text. The framework can be applied to any pre-trained LLMs during the finetuning stage of the model, regardless of its original positional encoding method. We demonstrate this by finetuning four different LLMs with Hansel and show that the mean absolute error of the output sequence decreases significantly in every model and dataset compared to the prompt-based length control finetuning. Moreover, the framework showed a substantially improved ability to extrapolate to target lengths unseen during finetuning, such as long dialog responses or extremely short summaries. This indicates that the model learns the general means of length control, rather than learning to match output lengths to those seen during training.

How many words are there in the following: ”The visualizations help readers understand the concept.” There are 10 words in the sentence ”The visualizations help readers understand the concept.” Why don’t you try again? Maybe by counting the words one by one. Sure, here’s another attempt at counting the words in the sentence ”The visualizations and examples in this paper help readers understand the concept”: 1. The 2. visualizations 3. help 4. readers 5. understand 6. the 7. concept There are 7 words in total.

Extended version — https://arxiv.org/abs/2412.14033

# Introduction

Large language models (LLMs) have shown great promise in generating coherent and fluent text, making them useful for a variety of tasks (Arivazhagan et al. 2019; Karpukhin et al. 2020; Lewis et al. 2020; Zhang et al. 2020). However, a number of challenges still remain for LLMs, including the difficulty of efficiently controlling the length of the output sequence. Length control of the output sequence has practical importance in the real-world application of LLMs. Setting the length determines the level of details to include in the output, which is desired both in the aspect of content and system interface. While this task seems relatively mundane compared to the emergent abilities of LLMs, to the best of our knowledge length control has yet to be achieved even

for the larger LLMs (Achiam et al. 2023; Anil et al. 2023;   
Touvron et al. 2023) with hundreds of billions of parameters.

A couple of specific examples where length control is useful would be i) a news app where articles are accompanied by AI-generated summaries, or ii) a voice assistant that can tune the amount of information to be spoken. For the news app, the default length should be concise enough to fit the system interface, and the user may want to tune the summarization length to include or drop details. Even for simple voice assistant tasks such as telling the weather, “rain at noon” may be sufficient but sometimes precipitation or temperature might be needed. These examples show the benefits of various outputs from the identical input and controlling them is important and useful.

The study of length control dates back to the pretransformer era. Kikuchi et al. (2016) first suggested feeding the remaining length information to the LSTM as an embedding or the memory cell. After the advent of the transformer (Vaswani et al. 2017), there were attempts to include the length information in a modified form of positional encoding (Takase and Okazaki 2019). Other studies turn to the attention mechanism such as Length Attention (Yu et al. 2021) and Length-Aware Attention (Liu, Jia, and Zhu 2022).

Although the previous methods of length control were fairly successful, they mostly concentrate on the relatively small sequence-to-sequence models (Sutskever, Vinyals, and Le 2014) and not the nowadays prevalent decoder models. The main reason for this is two-fold. First, the aforementioned methods require application in the pretraining stage and show significant performance degradation when used only while finetuning. Large decoder models are mostly pre-trained once as a foundation model and finetuned for downstream tasks due to their large size, making the previous methods inapplicable. Second, with their better understanding of the context, the decoder models have some, although imperfect, length-controlling ability. The decoder LLMs would not output the exact required length, for example, 17 words, but it does understand commands such as shorter/longer and will tend to output longer texts when specified to output 25 words than when asked for 10 words.

In this paper, we introduce a simple finetuning framework to improve the output length-controlling ability of LLMs. We first observe why generating outputs with a certain length is a difficult task for LLMs. LLMs are known to have limited mathematical ability even with the significant improvement from the zero-shot-CoT technique (Kojima et al. 2022). This not only applies to complex math problems but also to simple counting. Figure 1 shows a short example conversation with GPT 1, slightly edited for brevity. Because LLMs are not intrinsic counters, GPT sometimes outputs incorrect answers when asked to answer the number of words in a sentence. One way to correct this behavior is to prompt the model to explicitly count the words first and then output the answer as in Figure 1.

We build upon this observation and apply it to the task of output length control. With its lack of counting ability, the model would not be aware of how many words it has output and how many words are left at a certain point. To overcome this, we introduce the Hansel – Hidden Arrangements (of special tokens) in Natural Sequence for Expected Length – framework, which directly injects the information via the form of hidden special tokens. By augmenting a finetuning dataset with special tokens appearing regularly, the model can efficiently learn the relative position compared to the target length at a given time. This can be considered a simple hard-coded CoT method specifically for output length control. Our method performs well even when only used during finetuning without degradation in output quality. Phi2 (Li et al. 2023) based Hansel model showed great performance in mean absolute error from the reference length in four summarization and dialogue datasets (Table 1). Moreover, the error remains small for a wide range of arbitrary target lengths, where other methods diverge rapidly as the target length deviates from the dataset’s typical length (Figure 3).

Contribution The main contributions of this paper are as follows:

• We introduce the Hansel framework that learns the output length control during finetuning through repeated hidden tokens, which can be applied to decoder LLMs with large number of parameters.   
• We empirically validate the effectiveness of Hansel by experimenting with it on four different datasets, various models, and several target lengths.   
• We conduct a thorough comparison with the promptbased length control method and show that Hansel outperforms the prompt-based method as it learns the general means of length control.

# Related Work

Majority of previous studies have been done with sequenceto-sequence models. LenEmb and LenInit (Kikuchi et al. 2016) are learning-based methods where the length information is provided via an extra embedding and the LSTM memory cell, respectively. Takase and Okazaki (2019) modified the Transformer’s positional encoding to include the remaining length information. Fan, Grangier, and Auli (2018) train on examples with the length information (in ranges) included as a special marker and use the target length marker during inference. Other methods include utilizing the attention mechanism (Yu et al. 2021; Liu, Jia, and Zhu 2022) and prompts (Zhang et al. 2022b; He et al. 2022). While showing some promise in length-controlling ability, most methods previously mentioned require special pre-training and/or structural modification to the baseline model. Our method has some similarities with Fan, Grangier, and Auli (2018) in the sense that we use a special marker, and Zhang et al. (2022b) in that we utilize prompts. However, we use the special tokens periodically which is the crucial difference.

Decoder LLMs are now more common than encoderdecoder models, and they mostly come in more than billions of parameters. They tend to intrinsically have some degree of sense of the output length, and may properly reply to prompts such as “give me a shorter/longer answer.” The paradigm of length control in such models involves prompt learning (Raffel et al. 2020; Sanh et al. 2022; Liu et al. 2023a). Goyal, Li, and Durrett (2022) showed good results on prompt-based length control but only in the units of sentences. There is also work on prompt-based length control in the framework of reinforcement learning (Jie et al. 2023).

Recently, a new set of studies on length-control have emerged in the context of direct preference optimization (DPO) (Rafailov et al. 2024). DPO tends to generate longer responses (Singhal et al. 2024), which is in part because they are preferred over shorter ones while evaluation (Dubois et al. 2024). To prevent this over-verbosity, a number of papers have suggested methods to length-control, or length-desensitize during DPO. These include R-DPO (Park et al. 2024), SimPO (Meng, Xia, and Chen 2024), LIFTDPO (Yuan et al. 2024), SamPO (Lu et al. 2024), and LDDPO (Liu et al. 2024). However, with a different motivation, these works do not require outputs of specific lengths.

![](images/059649cb95bbea42fc66a363f7c5daecdc9013096df3bb964de1c4f415ad728b.jpg)  
Figure 2: Schematic of the Hansel framework, compared with the vanilla and Gretel scheme. Vanilla is normal fine-tuning and Gretel is the prompt-based length-aware fine-tuning. Hansel receives the target length as a special token and regularly places additional special tokens (marked as $@$ in the figure) that inform the position while fine-tuning.

# Method

As we have observed in Figure 1, LLMs are intrinsically not well suited for counting. The key idea of Hansel is to overcome such lack of counting ability by explicit counting and use it for counting the output words. In Hansel, we augment the fine-tuning training set by placing special tokens that indicate the remaining number of words. When fine-tuned with such dataset, the special tokens will also appear while inference and guide the model to output the desired length.

# The Hansel Dataset

The Hansel framework utilize the “Hansel dataset,” which is the original dataset augmented with special tokens that indicate the remaining length of the output. (Thus, giving the name Hansel from the German fairy tale “Hansel and Gretel.” Hansel leaves a trail of pebbles in the woods to find the way back home.) From now on, we count the units of length as words unless specified. The method is trivial to apply in different units, such as characters, tokens, and sentences. Empirically, the difficulty of length control increases from sentence, token, word, and character.

Below we take an example from CNN/DM.

[Original] :

Famous American foods created across United States.   
Connecticut diner claims creation of the hamburger.   
Onion rings were courtesy of cook at Pig Stand in Texas. [Hansel] :   
2 5 Famous American foods created across 2 United States. Connecticut diner claims creation of the hamburger. Onion $| 1 \rangle$ rings were courtesy of cook at Pig Stand in Texas. |0⟩

The augmented special token $| x \rangle \langle y |$ means that there are $\Delta x + y$ words left until the target length and $| x \rangle \equiv | x \rangle \langle 0 |$ . Hyperparameter $\Delta$ is the stride between special tokens, excluding the very first special token.

For each example, we count the number of words in the reference output $( l )$ and include $| \lfloor l / \Delta \rfloor \rangle \langle l \% \Delta |$ at the beginning. After $l \% \Delta$ words we insert $| | l / \Delta ] \rangle$ . The special tokens $| \lfloor l / \Delta \rfloor - 1 \rangle , | \lfloor l / \Delta \rfloor - 2 \rangle , \cdots , | 0 \rangle$ are inserted in $\Delta$ words interval until the end of the reference. In the above CNN/DM example, $l = 2 5$ and $\Delta = 1 0$ . Since $\lfloor l / \Delta \rfloor = 2$ and $l \% \Delta = 5$ , the first special token becomes $| 2 \rangle \langle 5 |$ and $\left. 2 \right.$ follows after 5 words.

When finetuning with the above dataset, there is a risk that the output will be abruptly terminated after the $| 0 \rangle$ token without actually finishing the sentence. To avoid this, we introduce one more hyperparameter $\delta$ , which indicates the maximum residual words after the $| 0 \rangle$ token. We set aside a portion of examples and consider the total length as $l - 1 , l - 2 , \cdots , l - \delta$ instead of $l$ . The purpose of a nonzero $\delta$ is to train with examples where the response does not terminate with $| 0 \rangle$ , and thus educate the model to finish the sentence naturally even when an unfinished response encounters $| 0 \rangle$ . The previous example for $\delta = 2$ would be as follows,

[Hansel $( \delta = 2 )$ )] :   
2 3 Famous American foods $\left| 2 \right.$ created across United States. Connecticut diner claims creation of the $| 1 \rangle$ hamburger. Onion rings were courtesy of cook at Pig Stand $| 0 \rangle i n$ Texas.

We randomly choose $2 0 \%$ of the samples and assign $l -$ $1 , l - 2 , \cdots , l - \delta$ uniformly. We also label mask $N = 1 0$ tokens preceding $| 0 \rangle$ while training. This is to ensure that the model does not explicitely learn incomplete sentences preceding $| 0 \rangle$ . A nonzero $\delta$ plays a crucial role in the Hansel framework preventing abrupt termination and its treatment while training will be detailed in the following section.

# The Hansel Framework

The Hansel framework essentially finetunes the pre-trained model with the Hansel dataset. One full example consists of a source, a prompt including the target length of the output, and the Hansel augmented output. The target length information is provided as the first special token.

We compare the Hansel framework with two baseline methods – prompt finetuning and the finetuning without length information. We dub the two as the Gretel and vanilla framework, respectively. Comparing the three frameworks’ training example, Hansel has [source]-[prompt with target length (special token)]-[Hansel output]; Gretel has [source]- [prompt with target length]-[original output]; and vanilla has [source]-[prompt without target length]-[original output]. For inference Gretel have [source]-[prompt with target length] as the context, while Hansel has [source]-[prompt with target length (special token)]. Vanilla has two versions, [source]-[prompt with/without target length]. We specifically dub the former (with target length) version as vanilla∗. An analogy with the three methods can be made for a person giving a talk. The vanilla model would correspond to giving a talk for a specific duration without any time-keeping practice. Gretel is trained by knowing how long each practice talk was, and required to give one for a certain time. Hansel’s practice talk was done with the knowledge of the total time, and a time-keeper who reminds the speaker every 5 minutes – and the same time-keeper attends the actual talk as well. The schematic of the three frameworks is depicted in Figure 2. We also present training and inference examples for each in the Appendix2.

# Experimental Setup

We used batch size 512 and AdamW optimizer (Loshchilov and Hutter 2018) with $5 \times 1 0 ^ { - 5 }$ learning-rate and parameters $\beta _ { 1 } = 0 . 9$ , $\beta _ { 2 } = 0 . 9 5$ . We finetune the pretrained LLMs for 2 epochs. See the Appendix for how the number of epochs affects the performance. Finetuning and inference are conducted using 8 Nvidia Tesla V100 GPUs (40GB). We set the maximum token number to 1722 and truncated longer examples.

# Dataset

We consider two tasks in this work, text summarization and dialogue response. Experiments are done with two datasets for each task.

CNN/DM (Hermann et al. 2015) is a large-scale news dataset from the Cable News Network (CNN) and the Daily Mail (DM). We follow (Nallapati et al. 2016), and treat the news articles as the source documents and the corresponding highlights (usually in bullet points) as the summaries.

XSum (Narayan, Cohen, and Lapata 2018) is a highly abstractive summarization dataset of news articles from the British Broadcasting Corporation (BBC). BBC has one-line headlines at the beginning of each article, which are regarded as summaries.

DailyDialog (Li et al. 2017) is a multi-turn dialogue dataset on various daily conversation topics, including relationships, ordinary life, and work. The raw data are crawled from English-learning websites and thus are human-written.

MultiWOZ (Zang et al. 2020) is a task-oriented dialogue dataset spanning 8 domains – restaurant, hotel, attraction, taxi, train, hospital, bus, and police. MultiWOZ also consists of human-human written conversation.

# Metric

For evaluation, we use both metrics that measure the performance of the length control and the quality of the output.

Mean Absolute Error To evaluate the length control ability of the model, we simply use the mean absolute error (MAE) of the output length from the target length. The MAE is measured as follows:

$$
\mathrm { M A E } = \frac { 1 } { N } \sum ^ { N } \left| l ( \mathrm { o u t p u t } ) - l ( \mathrm { t a r g e t } ) \right| .
$$

$l ( x )$ is the length of $x$ and $N$ is the number of samples.

ROUGE Recall-Oriented Understudy for Gisting Evaluation (Lin 2004) measures the lexical similarity between two texts by comparing the overlap of $n$ - grams. Variants of ROUGE include the $n$ -gram matching ROUGE- $n$ $\left( \mathbf { R } \mathrm { - } { n } \right)$ and the longest common subsequence matching ROUGE-L (R-L). Among these, we show R-L in the main text and include results for R1, R-2 in the Appendix.We use the Google Research implementation of ROUGE (https://github.com/googleresearch/google-research/tree/master/rouge) with the default settings.

G-Eval To overcome the lexical bias from ROUGE, we also evaluate G-Eval (Liu et al. 2023b), which is a framework for using LLMs with chain-of-thoughts (CoT) and a form-filling paradigm to assess the quality of natural language generation outputs. For the summarization task, the evaluation is in coherence (on a scale of 1-5), consistency (1-5), fluency (1-3), relevance (1-5); for the dialogue task, naturalness (1-5), coherence (1-5), engagingness (1-3), and groundedness (1-5). The score scale and the used prompts are from the original paper (Liu et al. 2023b). We use GPT4 (Achiam et al. 2023) as the evaluator.

# Baseline Model

To demonstrate that Hansel is applicable regardless of the base model’s positional representation method, we perform experiments on four different types of position methods – Rotary positional embeddings, ALiBi, learned positional embeddings, and T5 bias. For consistency, we used models with parameter sizes within the range of 2.7B to 3.0B for all methods.

Rotary Rotary positional embedding (Su et al. 2024) encodes the position with rotation matrices, only retaining information on purely relative position. For the model using rotary positional embedding, we use Microsoft’s Phi-2 with 2.7B parameters (Li et al. 2023).

![](images/d893da10a7319e1af35eecdc5ac1b09ca311d4190cd150c339645d9a8f4c3ef9.jpg)  
Figure 3: The extrapolation of the length control methods with different target lengths. The dashed line (shaded region) indicates the mean length ( $\pm$ standard deviation) of the dataset. While the MAE of other methods increases drastically when the target length is different from that of the dataset, our method (Hansel) shows robust performance.

Table 1: The length control experiment using Phi-2 as the pre-trained LLM. The target length is set to the reference’s actual length. Hansel and Gretel both showed good results without performance degradation.   

<html><body><table><tr><td>Dataset</td><td>Model</td><td>MAE R-L</td><td>G-Eval</td></tr><tr><td rowspan="2">DailyDialog</td><td>vanilla* 7.67</td><td>0.11</td><td>2.54</td></tr><tr><td>varetea 7.04</td><td>0.25</td><td>3.51</td></tr><tr><td>MultiWOZ</td><td>vanilla* vanilla Gretel 0.11 Hansel</td><td>7.24 0.13 5.50 0.30 0.34 0.05 0.34</td><td>3.55 1.86 3.59 3.55 3.54</td></tr><tr><td>XSum</td><td>vanilla* vanilla Gretel Hansel</td><td>5.35 0.33 5.35 0.33 0.47 0.33 0.26 0.34</td><td>3.18 3.20 3.22 3.26</td></tr><tr><td>CNN/DM</td><td>vanilla* vanilla Gretel Hansel</td><td>15.23 0.30 14.82 0.30 1.56 0.30 0.39 0.31</td><td>3.84 3.86 3.82 3.86</td></tr></table></body></html>

ALiBi ALiBi (Press, Smith, and Lewis 2022) is another relative positional method that is known to extrapolate well to longer sequences while inferencing. This adds a bias linear to the relative distance between the tokens. We experiment on BLOOM-3B (Workshop et al. 2022) for ALiBi.

Learned positional embedding Positional embedding uses absolute embeddings that are added to token embeddings. Unlike the other absolute method of positional encoding (Vaswani et al. 2017), the positional embeddings are learned throughout training. We use OPT-2.7B (Zhang et al. 2022a) for this method.

T5 Bias The T5 bias (Raffel et al. 2020) is a relative positional embedding method (Shaw, Uszkoreit, and Vaswani

Table 2: The length extrapolation results (MAE) of Phi-2. This is also plotted in Figure 3.   

<html><body><table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td colspan="4">l(target)</td></tr><tr><td>5</td><td>20</td><td>50 80</td><td>130</td></tr><tr><td>DailyDialog</td><td>vanilla Gretel Hansel</td><td>5.17 0.15 0.06</td><td>13.62 0.33 0.34</td><td>43.11 73.06 2.01 7.15 0.43 0.79</td><td>122.99 18.51 2.07</td></tr><tr><td>MultiWOZ</td><td>vanilla Gretel Hansel</td><td>5.76 0.12 0.03</td><td>9.73 0.17 0.12</td><td>39.39 69.39 2.89 9.96 0.32 0.78</td><td>119.39 23.85 7.47</td></tr><tr><td>XSum</td><td>vanilla Gretel Hansel</td><td>13.71 0.96 0.49</td><td>4.30 0.50 0.31</td><td>32.34 62.29 1.80 3.42</td><td>112.23 10.37</td></tr><tr><td>CNN/DM</td><td>vanilla Gretel Hansel</td><td>41.39 1.49 0.37</td><td>26.55 0.59 0.52</td><td>0.45 0.64 16.28 37.21 1.54 3.21 0.60 0.72</td><td>1.02 85.66 7.38 1.00</td></tr></table></body></html>

2018) similar to rotary and ALiBi but with a learned bias for each distance. We use the natural choice of T5-3B representing this method.

# Results

We first use Phi-2, which adapts rotary positional embedding, as the baseline model and finetune it with the four datasets in the three frameworks – vanilla, Gretel, and Hansel. We then measure their performance using the test sets, asking the model to perform the task in a lengthcontrolled manner. Here the target length was that of the reference. We set $\Delta = 2 0$ and $\delta = 1$ for this experiment. Note that when the model infinitely generates repeated output, we exclude them from the statistics such as MAE, ROUGE-L, and G-Eval. Instead we count the infinite generates samples separately and present them in the next section.

The result is shown in Table 1. First, we comment on the vanilla∗. Vanilla∗ uses the prompt with target length while trained without such target. (recall that vanilla is the conventional fine-tuned model) In dialogue datasets, this results in a model breakdown and the output quality greatly degrades, which can be checked in the ROUGE-L and G-Eval scores. Length control is meaningless if the model loses its generation ability, and therefore we do not consider the vanilla∗ model from now on. The vanilla model does not attempt the length control as the prompt does not include the target length, but its results are included in Table 1 to demonstrate the other models (Hansel and Gretel) does not have quality degradation (ROUGE-L and G-Eval). We include an expanded version of Table 1 in the Appendix which includes R-1, R-2, and the four different measures in G-Eval.

Table 3: Example context and outputs of the Phi-2 Hansel model. $\Delta = 2 0$ and $\delta = 1$ unless otherwise stated.   

<html><body><table><tr><td>Task</td><td>Context Article: A group of scientists have invented a gadget which turns pee into electricity. They hope it could</td></tr><tr><td>Summarization</td><td>be used in some poorer countries,as a way to create clean and renewable energy. The system was devel- oped by scientists at the Bristol Bioenergy Centre (BBiC) in the Bristol Robotics Laboratory at the Uni- versity of the West of England (UWE Bristol).The invention willbe tried out this week on festival-goers at the Glastonbury music festival in Somerset. Urine, or wee is fed into a special machine called a micro- bial fuel cel, which contains a type of bacteria. These tiny bacteria break down the wee and create elec- trons. These electrons are then forced through an electronic circuit, creating a flow of electricity. This</td></tr><tr><td>5 words 20 words</td><td>electricity can then be used to power all sorts of things, including your phone! [Summarize in # words.] Scientists have invented a generator. Scientists have come up with a new way to make electricity -by using wee, or urine,as a power source.</td></tr><tr><td>80 words</td><td>Scientists have invented a machine which turns wee into electricity. It works by using bacteria to break down the wee.This creates electrons which are forced through an electronic circuit.This electricity can then be used to power allsorts of things, including your phone! Scientists hope it could be used in some poorer countries,as a way to create clean and renewable energy. It will be tried out this week on festival-</td></tr><tr><td>Abrupt termination</td><td>goers at the Glastonbury music festival in Somerset. A: I am looking for a hotel. B: There are a total of 9 hotels. Huntingdon Mariotte Hotel is a good choice with a 4 star rating. Would you like there phone number? A: I actually need a place that has free wifi and is in the south part of town. And it needs to be a hotel, not a guesthouse. [Reply in 18 words.]</td></tr><tr><td>δ=0</td><td>There are two hotels hat meet your needs.The AylesbrayLodge Guest Houseand The Lensfield Hotel.</td></tr><tr><td>δ=1</td><td>The Lensfield Hotel is a hotel in the south with free wifi.Would you like to book it?</td></tr><tr><td>Multiple units 4 s 20 w</td><td>A: Have you any round-neck sweater? [Reply in # sentences and # words.] We have. Take a look at this one. It's made of cotton. It's very breathable and it's not expensive either.</td></tr><tr><td>3 s 30 w</td><td>We have some nice round-neck sweaters. They are the latest fashion and are very popular with young</td></tr><tr><td>2 s 40 w</td><td>people.Why don't you try one on and see how you like it? We have some nice round-neck sweaters in different styles and colors. If you like this one, you can try</td></tr></table></body></html>

The Gretel model shows significant improvement over the vanilla on length control with a much smaller MAE. The Hansel model also shows good length control performance, better than that of Gretel. Hansel’s MAE decrease over that of Gretel ranged from $45 \%$ to $7 5 \%$ among datasets. One salient feature is that Hansel’s length control ability did not degrade as much for long-target datasets, while the MAE tends to be larger with a longer target length. Due to its long average target length, CNN/DM has the largest MAE for all three frameworks, however Hansel improved the MAE by a significant $7 5 \%$ over Gretel.

Target length extrapolation While the result in Table 1 showed a better length control ability of Hansel, especially for long targets, one might consider Gretel model already has fairly small MAE and would be enough. To demonstrate the true strength of Hansel, we now turn to a different and perhaps a more important setting of length control with arbitrary target length.

There is a distinction between length control over a dataset and the model having a general length control capability regardless of the trained dataset. With the target length set as the reference length, Table 1 demonstrated the former. For the model to claim general length control ability, it should be able to target arbitrary lengths, different from the reference length seen during training. We target 5, 20, 50, 80, and 130 words and measure the MAE for each case. The results are shown in Table 2 and is plotted in Figure 3. (the other two plots are presented in the Appendix) We also show an example of the Hansel model outputs for a summarization task in Table 3, requested to generate outputs in various lengths.

For all datasets and models, the MAE increases as the target length deviates from $\pm$ (standard deviation) of the dataset (which is depicted as shaded regions in the figure). However, Hansel shows a robustly small (order of magnitude smaller than Gretel for longest targets) MAE throughout the test range. This indicates that the Hansel model has learned how to control the output length in general, while Gretel’s length control ability is more specific to the trained dataset. This ability of Hansel is beneficial for both training efficiency and robustness in actual applications of length control.

Table 4: The effect of hyperparameters $\Delta$ and $\delta$ on the length control ability. All entries are MAE compared with the reference length. Minimum values excluding $\delta = 0$ in bold.   

<html><body><table><tr><td rowspan="2">Dataset</td><td rowspan="2">△</td><td colspan="3">8</td></tr><tr><td>0</td><td>1 3</td><td>5</td></tr><tr><td rowspan="2">DailyDialog</td><td>10</td><td>0.05 0.08</td><td>0.12</td><td>0.16</td></tr><tr><td>20 40</td><td>0.08 0.16</td><td>0.09 0.12 0.13 0.17</td><td>0.16 0.17</td></tr><tr><td>MultiWOZ</td><td>10 20 40</td><td>0.03 0.04 0.05 0.05 0.09 0.08</td><td>0.05 0.07 0.08</td><td>0.07 0.09 0.09</td></tr><tr><td>XSum</td><td>10 20 40</td><td>0.10 0.19 0.27</td><td>0.20 0.26 0.26 0.32 0.25 0.45</td><td>0.37 0.38 0.44</td></tr><tr><td>CNN/DM</td><td>10 20 40</td><td>0.09 0.40 0.17 0.39 0.53 0.49</td><td>0.60 0.60 0.81</td><td>0.83 0.76 0.88</td></tr></table></body></html>

Table 5: The length control experiments showing MAE with baseline models using different positional embeddings. The target length is set to the reference’s actual length.   

<html><body><table><tr><td>Dataset</td><td>Model</td><td>ALiBi</td><td>Learned</td><td>T5 Bias</td></tr><tr><td rowspan="3">DailyDialog</td><td>vanilla</td><td>6.98</td><td>6.00</td><td>8.69</td></tr><tr><td>Gretel</td><td>0.24</td><td>0.60</td><td>0.33</td></tr><tr><td>Hansel</td><td>0.15</td><td>0.21</td><td>0.18</td></tr><tr><td rowspan="3">MultiWOZ</td><td>vanilla</td><td>5.67</td><td>5.68</td><td>5.63</td></tr><tr><td>Gretel</td><td>0.14</td><td>0.21</td><td>0.25</td></tr><tr><td>Hansel</td><td>0.12</td><td>0.11</td><td>0.22</td></tr><tr><td rowspan="3">XSum</td><td>vanilla</td><td>5.25</td><td>5.03</td><td>4.73</td></tr><tr><td>Gretel</td><td>0.47</td><td>0.69</td><td>0.92</td></tr><tr><td>Hansel</td><td>0.39</td><td>0.37</td><td>0.47</td></tr><tr><td rowspan="3">CNN/DM</td><td>vanilla</td><td>14.46</td><td>14.70</td><td>13.80</td></tr><tr><td>Gretel</td><td>1.78</td><td>2.80</td><td>3.62</td></tr><tr><td>Hansel</td><td>0.98</td><td>0.69</td><td>1.09</td></tr></table></body></html>

The effect of hyperparameters We investigate the effect of the hyperparameters $\Delta$ and $\delta$ . We vary $\Delta$ from $\{ 1 0 , 2 0 , 4 0 \}$ and $\delta$ from $\{ 0 , 1 , 3 , 5 \}$ and measure the MAE and show the results in Table 4.

The MAE increases with $\delta$ as we have trained the model with examples with larger residuals. Although $\delta = 0$ results were best in terms of MAE, we avoid selecting those from the quality perspective. Manually going through $\delta = 0$ outputs we identified instances where the output was abruptly terminated near the target length. We show an example of this in Table 3. The nonzero $\delta$ was introduced for precisely this reason and the result suggests $\delta = 1$ is sufficient for that purpose. Small $\Delta$ is also unnecessary (as it generates excessive special tokens) and we conclude that $\Delta = 2 0$ and $\delta = 1$ is the best combination of hyperparameters.

Various positional embeddings As the Hansel framework only requires the Hansel dataset and does not alter the model architecture or positional embeddings, it can be applied to any baseline models. To demonstrate this, we experiment on three more models, BLOOM, OPT, and T5. These represent models with ALiBi, learned, and T5 positional embeddings, respectively. Moreover, the T5 is an encoderdecoder model while the others are decoder models. The results are presented in Table 5.

While each model has a different base performance, the tendency is largely similar to Table 1 – Hansel controls length the best without significant loss of quality. This demonstrates the adaptability of the Hansel framework.

Infinite generation Additionally, there is an interesting byproduct of this length control. Infinite generation is a quite common problem in LLMs, which gets more prominent for small on-device models. There are reports of this issue even in very recent models such as Llama 3 (Dubey et al. 2024). If the model has the length controlling ability, this automatically solves the infinite generation issue. For the length extrapolation results in Table 2, vanilla and Gretel experienced infinite generation 8.4 and 15.3 times per 10,000 generations, respectively, while there was no such incident for Hansel. Detailed data of this experiment is shown in the Appendix.

Versatility and generalizability The framework is also versatile enough to control multiple units. For example, it is possible to train a model that can control both the number of sentences and words. One just needs to prepare the Hansel dataset which consists of two different families of special tokens, each for a sentence and word. An identical algorithm can be applied for augmentation. Examples of sentence and word number control are shown in Table 3. Moreover, Hansel’s length control can also work as on the fly – one just need to specify the special token to turn on the length control ability.

We have also performed ablation studies to ensure Hansel does not affect the output distribution and is generalizable to other domains and tasks. The model’s output on a separate task was not affected by the special tokens, and found that the generalization was possible with only 50 additional instruction samples. Please refer to the Appendix for details of the ablation studies.

# Conclusion

We have introduced the Hansel framework that efficiently enables LLMs to control the length of their output. The Hansel framework only utilizes the Hansel dataset which is an augmentation from the original dataset, and thus can be applied to models regardless of architecture and positional encoding. Moreover, it is large pre-trained model friendly as it works through finetuning, and retains its length control ability even when the target length is out of the typical range of the finetuned dataset. The method outperforms the traditional prompt-based finetuning in the MAE from the target length. It especially showed significant improvement in extrapolated target length, demonstrating that the model learns the general means of length control.