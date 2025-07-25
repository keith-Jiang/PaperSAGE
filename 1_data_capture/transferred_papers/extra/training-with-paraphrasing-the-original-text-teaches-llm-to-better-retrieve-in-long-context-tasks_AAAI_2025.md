# Training with “Paraphrasing the Original Text” Teaches LLM to Better Retrieve in Long-Context Tasks

Yijiong ${ \bf { Y } } { \bf { u } } ^ { 1 }$ , Yongfeng Huang1 2 3, Zhixiao $\mathbf { Q } \mathbf { i } ^ { 1 }$ , Zhe Zhou1

1Tsinghua University 2Zhongguancun Laboratory 3Institute for Precision Medicine, Tsinghua University yuyj22, qzx21, zhouzhe $2 0 \} \ @$ mails.tsinghua.edu.cn, yfhuang $@$ tsinghua.edu.cn

# Abstract

As Large Language Models (LLMs) continue to evolve, more are being designed to handle long-context inputs. Despite this advancement, most of them still face challenges in accurately handling long-context tasks, often showing the “lost in the middle” issue. We identify that insufficient retrieval capability is one of the important reasons for this issue. To tackle this challenge, we propose a novel approach to design training data for long-context tasks, aiming at augmenting LLMs’ proficiency in extracting key information from long context. Specially, we incorporate an additional part named “paraphrasing the original text” when constructing the answer of training samples and then fine-tuning the model. Experimenting on LongBench and NaturalQuestions Multidocument-QA dataset with models of Llama and Qwen series, our method achieves an improvement of up to $8 . 4 8 \%$ and $4 . 4 8 \%$ in average scores, respectively, showing effectiveness in improving the model’s performance on long-context tasks.

# 1 Introduction

Large Language Models (LLMs) have recently emerged as top performers in a wide range of natural language processing tasks. Nevertheless, they are usually trained on segments of text of a fixed length, which means they have a predetermined context window size. Typically, their performance tends to decline markedly when the input text exceeds the context window size.

Some classic works like Yarn (Peng et al. 2023), LongChat (Li et al. 2023a) and LongAlpaca (Chen et al. 2023c) have explored ways to make short-context LLMs better adaptable to long-context tasks. Based on these, recently, an increasing number of powerful LLMs have been equipped with the capability to handle long contexts, such as Mistral (Jiang et al. 2023a) and Qwen2 (Yang et al. 2024), which can handling the text length of 32k or longer.

However, while LLMs have made remarkable progress in handling long-context tasks, as shown in many evaluations (Li et al. 2023b; An et al. 2023; Wang et al. 2024), they still lack satisfactory accuracy, and often suffer from a severe “lost in the middle” problem (Liu et al. 2024) when the context is getting longer or more complex. “Lost in the middle” refers to the phenomenon where the model’s utilization of the context significantly weakens when the key information is located in the middle of the context. This limitation hinders the further application of LLMs in long-context scenarios.

![](images/e164ad8c4f03dcd63b30507abd18d96289a53b908c88407a28fe1075a425b858.jpg)  
Figure 1: Our method adds “paraphrasing the original text” to the training samples.

Long-context tasks can be divided into two categories: short-dependency and long-dependency (Li et al. 2023b). The former means only a small part of the long context is truly needed for the task, while the latter means most parts are needed. Because short-dependency tasks are usually more common and easier to study, for simplicity, in this paper, we start with the short-dependency tasks and mainly focus on multi-document-QA, which is one of the most representative long-context tasks and also easy to be constructed.

In short-dependency long-context tasks, the useful information in the context is sparse, thus the task can be regarded as a composite task, which can be split to “first retrieval, then another follow-up task” process, while the retrieval step is abstract and implicit but necessary. However, we find LLMs usually perform much worse in such composite tasks, even if they are good at each sub-task individually.

In this case, it is natural to think that using CoT (Wei et al. 2022) prompt could help. Yet, in our experiments, despite using CoT-like prompt, we find many LLMs still perform poorly, which generate wrong answers just in the first step, retrieval, even though some of them can perfectly pass “Needle in a Haystack” test, which represents qualified retrieval ability. While in the other hand, if they retrieve correctly, then the follow-up task becomes easier and the final answer

is usually right.

We posit that this is either because these models inherently have weak retrieval capabilities, or because their retrieval abilities cannot be fully activated by just designing prompts, which hinders them from accurately locating key information in the long context. This prompts us to seek a more effective method to enhance the accuracy of models in retrieval-based composite tasks, through teaching them to better retrieve.

Thus, we propose a method based on fine-tuning with specially designed samples to enhance and activate LLMs’ retrieval capability over long contexts. Specifically, for the goal of explicitly separating and highlighting the “retrieval” step, when designing answers of a long-context QA sample, different from normal ways only designing a brief answer, we incorporate an additional part named “original text paraphrasing”, which paraphrases or repeats the original text of the sentences in the context containing relevant information required for answering, as shown in Figure 1. This part corresponds to a direct “retrieval” operation, while maintaining the answers’ quality and coherence, which aims to not only enhance the model’s retrieval capability but also teach the model to use its inherent retrieval capability more actively.

With the help of GPT-4 (OpenAI 2023), we automatically construct a dataset consisting of thousands of training samples. Through evaluate models fine-tuned on the training samples designed by us, we prove our method can generally benefit the model’s performance across various long-context tasks (not just retrieval or QA tasks). Besides, our method only need a light-weight dataset and a cost-effective finetuning stage, which is very applicable.

Our main contributions are summarized as follows:

1. We find that LLMs do not guarantee the full utilization of their retrieval capability in composite long-context tasks, even if their individual capabilities in simple tasks are strong.   
2. We propose the “original text paraphrasing” approach, which explicitly isolates the “retrieval” step, to construct training samples, aiming at teaching LLMs to better use their retrieval capability.   
3. Through fine-tuning and then evaluating, we prove our method can improve LLMs’ retrieval capability as well as overall long-context performance.

# 2 Related Work

There have been many studies that aim to improve the longcontext performance of LLMs and address “lost in the middle” issue, which can be summarized into the following four aspects:

# Input Context and Prompt

It is well known that using an appropriate prompt such as CoT (Wei et al. 2022) can significantly improve the model’s performance in a zero-shot way. A common method is to prompt the model to first give relevant evidence and then answer the question. This way can guide the model to first retrieve relevant information from the long text and then give answers based on this information. For example, the Claude2.1 team proposed that adding the sentence “Here is the most relevant sentence in the context” (Anthropic 2023) can improve the long-context QA accuracy from $27 \%$ to $98 \%$ .

In addition, reorganizing the long input context is also effective. For example, LongLLMLingua (Jiang et al. 2023b) compresses long contexts, and Attention Sorting (Peysakhovich and Lerer 2023) improves the model’s utilization of contexts through reordering the input documents.

# Training Data

Some works attempt to fine-tune models with long-context datasets to improve their long-context performance. ZiyaReader (He et al. 2023) proposes an attention strengthening method, designing the answer with 3 parts: “question repetition”, “index prediction” and “answer summarization”. FILM (An et al. 2024) proposes “InformationIntensive Training” to teach the model that any position in a long context can contain crucial information.

# Position Embedding

The remote attenuation characteristic of ROPE (Su et al. 2021) is often considered a significant factor contributing to the “lost in the middle” phenomenon. Thus Chen et al. propose “Attention Buckets”, which sets up three different frequencies of ROPE and integrates them together, filling in the troughs of ROPE’s remote attenuation curve. MSPOE (Zhang et al. 2024) defines “position-aware” scores for each attention head and then assigns different ROPE interpolation (Chen et al. 2023a) factors to each attention head, which significantly improves the response accuracy of Llama2 (Touvron et al. 2023) in long-context scenarios.

# Attention Weights

Hsieh et al. claim that the biased attention distribution of the model is the direct cause of its poor performance on long context, thus they decompose attention into two components, determined by position and semantics, respectively, and then calibrate the attention by eliminating the positionrelated component. Gao et al. analyze the attention matrix to directly eliminate the less important parts of attention and compensate the important parts, thereby making the attention distribution more rational.

# 3 Method

# Models Often Fail to Fully Utilize Retrieval Ability

We take Qwen1.5-4b-Chat (Yang et al. 2024) for example, which is a model with $3 2 \mathrm { k }$ context window. As shown in Figure 4, it can nearly perfectly pass “Needle in a Haystack” (gkamradt 2023) test within 32k length, which is a task requiring LLM to retrieve a sentence containing relevant information (i.e. the “needle”) from a long context made up of a large amount of irrelevant information, when the “needle” is inserted at various positions of the context. Passing this test represents a long-context LLM has the ability to directly retrieve information at anywhere of the context.