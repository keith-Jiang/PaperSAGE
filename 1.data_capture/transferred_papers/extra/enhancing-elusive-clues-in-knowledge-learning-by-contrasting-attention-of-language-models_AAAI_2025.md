# Enhancing Elusive Clues in Knowledge Learning by Contrasting Attention of Language Models

Jian Gao2, Xiao Zhang1, Miao Li1,B, Ji Wu1,3,4

1Department of Electronic Engineering, Tsinghua University 2Department of Energy and Power Engineering, Tsinghua University 3College of AI, Tsinghua University 4Beijing National Research Center for Information Science and Technology gaojian21, xzhang19 $@$ mails.tsinghua.edu.cn miao-li, wuji ee @tsinghua.edu.cn

# Abstract

Causal language models acquire vast amount of knowledge from general text corpus during pretraining, but the efficiency of knowledge learning is known to be unsatisfactory, especially when learning from knowledge-dense and small-sized corpora. The deficiency can come from long-distance dependencies which are hard to capture by language models, and overfitting to co-occurrence patterns and distracting clues in the training text. To address these issues, the paper proposes a method to enhance knowledge learning during language model pretraining, by enhancing elusive but important clues in text discovered by the language model themselves. We found that larger language models pay more attention to non-obvious but important clues, which are often overlooked by smaller language models. Therefore, we can identify these clues by contrasting the attention weights of large and small language models. We use the identified clues as a guide to perform token-dropout data augmentation on the training text, and observed a significant boost in both small and large models’ performance in fact memorization. This shows that the behavior contrast between more and less-performant language models contains important clues for knowledge learning, and it can be “amplified” for a straight-forward improvement in knowledge learning efficiency.

# Code —

https://github.com/tsinghua-msiip/contrasting attention

# Introduction

Pretrained large language models have shown impressive performance on a wide variety of downstream tasks (Ouyang et al. 2022; Chung et al. 2022; Touvron et al. 2023). To achieve good generalization, these models need to be trained on web-scale corpora that are diverse and large enough to capture the complexity of natural language. Unfortunately, it is observed that when training corpora is limited in size or style variation, language models can struggle to generalize the information learned from the corpora (Zhu and Li 2023). This deficiency poses a challenge for injecting knowledge into pretrained language models via continual pretraining (finetuning). In many domains, the available corpora is often limited and knowledge-dense (e.g., in forms of textbooks, manuals, documentations). Such domain text may be difficult to be utilized effectively in finetuning, and the language models may not be able to effectively generalize the domain knowledge to downstream domain tasks.

Not very much is known about the causes of such deficiency in knowledge learning. One likely cause is overfitting to co-occurrence patterns in the limited training text, causing learning of spurious correlations instead of correct factual associations. Another possible reason is the difficulty of capturing long-range dependencies in text, which are crucial for understanding complex relationships. Such deficiency is sometimes a result of intentional design choice in the model architecture, such as the decay of attention weights in the RoPE (Su et al. 2024) positional encodings.

One possible route to understanding this phenomenon is via the attention module in language models. The attention mechanism is a key component that allows the model to focus on different parts of the input when making predictions. The attention weights are shown to be interpretable and explaining the model’s behaviors (Clark et al. 2019).

Recently, Y¨uksekg¨on¨ul et al. (2023) show that when predicting factual information, models are less likely to attend to the correct clue if the model does not know about the fact. This implies that for new knowledge unknown to the model, the model may not be able to attend to the correct clue at first, leading to difficulty in associating the correct clue (e.g., the head entity) with the prediction target (the tail entity).

To help language models learn, especially smaller models, a common approach is to use knowledge distillation (Hinton, Vinyals, and Dean 2015) (or teacher-student method) to transfer knowledge from a larger model. Given a learning goal, a more performant language model such as GPT-4 (OpenAI 2023) is often used to generate training data for the smaller model (Xu et al. 2024). A main drawback of this approach is that it requires the larger model to be already capable of the task or already have the knowledge. This make it not suitable for learning novel knowledge, such as new facts from an evolving domain. Also, it can only help the smaller model to learn but cannot help the larger model.

In this paper, we propose a simple method to enhance factual knowledge learning in continual pretraining, with the help of a pair of larger and smaller models. Our method is effective in learning novel facts and can boost the performance of both the larger and smaller models. The main contributions of the paper are as follows:

Attention difference between large and small language models reveals elusive but important clues in text. We show that while large and small language models both show high attention to important and obvious clues in text, large models pay significantly more attention than smaller models to important clues that are less obvious or elusive. Therefore, by contrasting the attention weights of large and small models, we can identify these elusive clues in text that are important for knowledge learning but are often easily overlooked.

Augmenting elusive clues in text boosts knowledge learning in continual pretraining. We show that by using the identified elusive clues as a guide, a token-dropout data augmentation that highlights the elusive clues can significantly boost the model’s performance in knowledge learning. We experimented on both synthetic and real-world corpus and show that the proposed method outperforms other forms of data augmentation, and boosting elusive clues universally helps both the large and the small models.

To the best of our knowledge, we are the first to analyze the the attention discrepancies between large and small models and use it for data augmentation. Prior work have distilled attention pattern from large models to small models, but without analyzing what is being distilled. Unlike distillation, our approach also enhances the performance of large models, which is a novel contribution on our part.

We release the code and data used in this paper for reproducibility and further research.

# Related Work

# Attention as Behavior Explanation

It is observed that attention weights in transformer models provide interpretable clues about the model’s behavior. For example, attention heads within multi-head attention can spontaneously differentiate into distinct roles (Clark et al. 2019). Certain heads play a more significant role and affect performance significantly (Voita et al. 2019). More performant models tend to have attention weights that focus more on key information and features, a possible explanation of their superior performance (Yu¨ksekgo¨nu¨l et al. 2023).

Some argue that while attention is somewhat interpretable, its interpretability is not an indicator of model performance (Serrano and Smith 2019). There is divided opinion on the extent to which attention weights reflects true model behavior (Jain and Wallace 2019; Wiegreffe and Pinter 2019). Our study extends these findings by comparing and contrasting attention weights of different models, and show that the difference between attention weights of large and small models can provide important behavioral clues.

# Data Augmentation on Text

Data augmentation is a critical technique for enhancing robustness and generalization, especially for limited-size datasets. Various data augmentation methods have been proposed, including random editing of sentences (Wei and

Zou 2019) such as insertion, swapping, and deletion. Synonym replacement methods (Mosolova, Fomin, and Bondarenko 2018; Rizos, Hemker, and Schuller 2019) replace words with their synonyms. Contextual augmentation methods (Kobayashi 2018) replace words with other words predicted by a language model for semantic variations. Backtranslation (Sennrich, Haddow, and Birch 2016; Edunov et al. 2018) is another commonly used method that generates augmented data by translating to and then back from another language. More sophisticated methods combine multiple augmentations (Xie et al. 2020; Karimi, Rossi, and Prati 2021).

Given that attention provides interpretable clues about the model’s behavior, Yu et al. (2022); Hailemariam et al. (2023) uses attention weights to find semantically significant words for replacement augmentation. Lewy and Mandziuk (2023) uses attention weights to find significant input parts for mixup augmentation (Zhang et al. 2018). We go a step further and show that only augmenting the most significant words is insufficient for challenging knowledge learning scenarios, and augmenting hard-to-notice but important parts of the input boosts the model’s performance even better than augmenting the significant parts.

# Teacher-Student Methods for Language Models

To enhance the performance of smaller models, knowledge distillation methods have been extensively developed to transfer knowledge from larger models to smaller models (Hinton, Vinyals, and Dean 2015; Xu et al. 2024). Large pretrained language models can be used to generate data for finetuning smaller models to transfer its knowledge and skills, for example, instruction following (Wang et al. 2023; Chiang et al. 2023) and reasoning ability (Fu et al. 2023; Ho, Schmid, and Yun 2023). Distillation from large model is also frequently used to build strong domain or task-specific models with a compact size, like for coding (Gunasekar et al. 2023; Rozie\`re et al. 2023) and math (Luo et al. 2023; Yue et al. 2023). Our work explores a different way to utilize large models: we find the behavior difference between large and small models and use it to guide the models towards more difficult part of the text.

# Continual Pretraining of Language Models

Continual pretraining takes a language model pretrained on a general corpus and continual the pretraining process with a new corpus, typically domain-specific text, to enhance the model’s performance on domain tasks. Model acquires new knowledge and ability via continual pretraining, for example, in coding (Chen et al. 2021), math (Lewkowycz et al. 2022), and medicine (Singhal et al. 2023). We aim at learning new factual knowledge from text via continual pretraining, similar to those in (Jang et al. 2022; Zhu and Li 2023).

# Problem Setup: Knowledge Learning Deficiency

Task: Fact Learning in (Continual) Pretraining

Language models can learn factual knowledge from pretraining (or continual pretraining) on text corpora. Zhu and Li (2023) introduced a synthetic biography dataset for evaluating the efficiency of knowledge learning in language models. The dataset has been utilized by (Khalifa et al. 2024), (Golovneva et al. 2024), and (Saito et al. 2024). It consists of short synthetic biographies of individuals, with a fixed format shown in the following example:

Liam Thompson was born on January 5, 1990. He spent his early years in Melbourne, Australia. He received mentorship and guidance from faculty members at Sorbonne University. He completed his education with a focus on Biomedical Engineering. He had a professional role at the British Museum.

Each biography contains information about an individual’s name, birth date, birth city, education, and job status. The task is to finetune (continual pretraining) a language model on the biographies to let it memorize the factual information about the individuals. After training, the model is evaluated on a question-answering task, where we evaluate the model’s accuracy in memorizing the underlined part of the biographies.

The questions are formatted like “When was Liam Thompson born?”. When questions were rephrased using GPT-4, performance generally declined, indicating that the original questioning format yielded the best performance, so that question style has minimal impact on our conclusion.

# Deficiency in Knowledge Learning Over Long-Range Dependency

Zhu and Li (2023) have shown that training language models from scratch on the biographies yield poor performance in question answering. We instead perform continual pretraining on pretrained language models up to 70 billion parameters. The language models have undergone extensive pretraining on massive corpora and show strong language capabilities.

We show that even pretrained models with billions of parameters struggle to memorize facts perfectly in continual pretraining. Table 1 show that while Gemma 2 (Team et al. 2024) and LLaMA 3 (Dubey et al. 2024) memorize the first two pieces of information (birth date and birth city) with high accuracy, they struggle to memorize the following three pieces of information (university, major, and company). This rules out the possibility that the performance deficiency is due to limited model size or insufficient pretraining. We also tried swapping the positions of five kinds of information resulted in the same trend: accuracy decreases as distance increases, demonstrating that long-range dependencies, rather than en-tity types, are the primary cause of poor performance.

The performance trend on QA tasks is also plotted in Figure 1. It is clear that as the relationship spans longer distances (i.e., the distance between the tail entity, such as “Company”, to the head entity name, the person’s name), the model’s performance show a decreasing trend. This indicates that the model struggles to capture long-range dependencies in text, which is crucial for learning complex relationships.

One possible reason for the deficiency in learning longrange dependencies is overfitting to a large amount of distracting information between the head and tail entities in a relationship. Overfitting is more likely when relationship only occur in few examples like in the biography dataset. Another possible reason comes from the bias in the model architecture that biases the model’s attention towards nearby information. Many popular models, such as LLaMA and Gemma, use the Rotary Position Embedding (RoPE) (Su et al. 2024) as positional encoding in their attention module. RoPE has a long-term decay property, which means that attention weights decay as the relative distance between the key and value token increases. This makes the model focus more on adjacent information but at a cost of important information that are occasionally far-away, hurting the model’s performance in learning long-range dependencies.

Table 1: Performance on the QA task after continual pretraining on the biography corpus.   

<html><body><table><tr><td></td><td>Date</td><td colspan="4">City University Major Company</td></tr><tr><td>LLaMA 38B</td><td>EM</td><td>0.82</td><td>0.91 0.20</td><td>0.34</td><td>0.09</td></tr><tr><td></td><td>F1</td><td>0.90 0.93</td><td>0.55</td><td>0.41</td><td>0.11</td></tr><tr><td>LLaMA370B</td><td>EM F1</td><td>0.98 0.95 1.00 0.98</td><td>0.36 0.67</td><td>0.73 0.77</td><td>0.66 0.67</td></tr><tr><td>Gemma 2 2B</td><td>EM</td><td>0.98</td><td>0.99 0.12</td><td>0.54</td><td>0.15</td></tr><tr><td></td><td>F1</td><td>0.98 0.99</td><td>0.40</td><td>0.57</td><td>0.18</td></tr><tr><td></td><td>EM</td><td></td><td>0.51</td><td>0.89</td><td></td></tr><tr><td>Gemma 29B</td><td>F1</td><td>0.99 1.00</td><td>1.00 1.00 0.66</td><td>0.90</td><td>0.63 0.64</td></tr></table></body></html>

![](images/7d44aba48d19249ffda478f10ca891ccf95f1b276176d167394c2c7b44f192e2.jpg)  
Figure 1: Performance on the QA task show a decreasing trend as the distance between the head and tail entities in the relationship increases in the training text.

# Analysis: Contrasting Attention of Language Models

We have shown that language models could achieve nearperfect accuracy in memorizing relationships that span a short distance in text, but struggle when they span a longer distance. In this section, we use attention weights as an interpretability tool to analyze the model’s behavior while learning long-range dependencies. We show that LLMs can pay inadequately little attention to key information that is located further away, and more performant larger models can pay more attention to these information than smaller models.

# Attention Weight Visualization

We look at model’s attention weights to try answering the following question: what information does the model pay attention to when predicting the tail entities in a relationship? The model uses attention weights to retrieve hidden states of context tokens, therefore the weights determines the information flow from the context to the current token in text. Furthermore, if an incorrect head entity is attended to when predicting the tail entity during the forward pass, in backpropagation the model will likely reinforce this incorrect association and cause the model to learn the wrong relationship.

To visualize model’s attention weights when predicting the tail entities in a relationship, we extract the attention weights at the preposition tokens, i.e., the word immediately preceding the tail entity. For example, in the sentence “He received mentorship and guidance from faculty members at Sorbonne University”, the attention from the token “at” is extracted. Because the model is predicting the tail entity “Sorbonne University” at this position, the attention weights1 here likely corresponds to the information necessary for predicting it. To ease visualization and for better comparison, instead of directly showing the attention weights, we rank the tokens and visualize the top 10 tokens with the highest attention weights. For each model, we calculate the token attention ranking for 100 biographies2, and summarize the ranking using a bar plot in Figure 2.

Results show that models assign the most attention to the most important information for predicting the tail entity: the relationship words. The model also pays much attention to the distracting entities in the preceding text. The correct head entity, which is the key information for predicting the tail entity, receives hardly any attention from smaller models and only a small amount of attention from larger models such as Gemma 2 9B, and is almost never ranked in top tokens. This indicates that the model’s attention is biased towards short-distance information, which may lead to the model learning the incorrect association and overfitting to such spurious co-occurrences.

# Contrasting Attention of Large and Small Language Models

Comparing to smaller models, larger language models tend to have overall better language understanding capabilities, therefore could be more likely to pay attention to the correct clue in the text. For a same family of models, for example, the LLaMA 3 8B and 70B models, the training corpus, model architecture, and training procedure are mostly similar, and they should have relatively similar general behavior pattern besides their capability differences.

Therefore, we can contrast the attention pattern between a large and a small model in the same family to identify the difference in the clue they pay attention to. In Figure 3, we subtract the attention weights of the small model from the large model, and visualize the top 10 tokens with the largest attention differences. The graph shows tokens receiving the most “additional” attention from the large model. It is clear that the correct head entity of the relationship, the “name” tokens (in red color), often receive the most additional attention3.

Comparing the original model attention in Figure 2 and the attention difference in Figure 3, we can see that while larger models pay more attention to the correct clue in text, the absolute attention weights on the correct clue is still small and biased towards the closer distracting entities. This calls for a method to “amplify” the attention differences so that the model can focus even more on the correct clue in text.

# Method: Augmentation From Contrasting Attention

We have shown that important clues that are hard to notice in text can be discovered from the attention difference between large and small models. Next, we propose to utilize and amplify these clues by combining with a simple dropout data augmentation method.

# Token-Dropout Data Augmentation

To combat overfitting, token-dropout data augmentation is a simple and effective technique that randomly drops out tokens in a training example (Wei and Zou 2019). Token-dropout introduces noise to the training data and breaks the model’s reliance on spurious co-occurrences in the training examples, helping the model achieve better generalization. A naive token-dropout randomly deletes each token independently with a probability $\alpha$ .

# Augmentation Guided by Elusive Clues

Although naive token-dropout mitigates overfitting, it does not solve the long-range dependency learning problem. As each token is dropped out independently, the model still suffers from inadequately small attention to non-obvious and distant information. We propose to use the attention difference between large and small models as a guide to dropout tokens in a more selective way. We first use the attention difference to rank the tokens in the training data, and then dropout tokens with a probability that is inversely proportional to their ranking. In this fashion, the model is encouraged to focus more on the tokens containing important but elusive information, as identified by the attention difference.

We use the following function to calculate dropout probability for each token:

$$
p ( r ) = \alpha ( 1 - e ^ { - \beta r } )
$$

The token with the $r$ -th rank (having the $r$ -th largest attention difference) will be dropped out with probability $p ( r )$ . The

![](images/ae756763899862a1a3529d5b556e2aa9d16caf0e536f3fdfe8fe5092da56da53.jpg)  
Figure 2: Visualization of tokens receiving the highest attention weights, at the preposition just before the “company” field. Tokens in a sentence are ranked by attention weight, from large to small. Each bar in the graph show the constitution of the i-th ranked token from 100 biographies. “ $\langle . . . \rangle ^ { \dagger }$ denotes tokens belonging to the information fields, and all else are individual tokens. Models generally pay most attention to the relationship words (e.g., “professional”, “role”, “at”), then to distrating entities in between (e.g., birth date, city, etc.). Because LLaMA 3 models have no special start token at the front of sentences, we add ”Text: ” at the beginning of sentences to avoid impact of the special position of tokens. All visualization results of LLaMA 3 are done in this way.

![](images/bcc38892dca71d9ebc408b5d906698e77370f5b8b617f0399415e8f025c8329c.jpg)  
Figure 3: Visualization of tokens receiving the highest additional attention weights from the large model compared to the small model. For example, the 9B/2B graph visualizes the distribution of the top 10 tokens with the largest attention weight(Gemma 2 9B) - attention weight(Gemma 2 2B) values. The name tokens (in red), the correct head entity, receive significant additional attention from the larger model.

Attention weights from 70B model Liam Thompson was born on January 5,1990..   
Attention weights from 8B model tokens subtract Thompson was or January Attention difference p(r)=α(1-e-βt) dropout instruct Li Thompson born on January 5, 0. Liam Thompson was January 5, 9 Li Thompson born on January 5, 9 model training Liam Thompson was January ,1 9 Li Thompson was born January 9

hyperparameter $\beta$ controls how fast the dropout probability increases with the ranking, and $\alpha$ controls the maximum dropout probability. The tokens with higher attention differences will have lower dropout probabilities, encouraging the model to focus more on these tokens. Figure 4 illustrates the process of the proposed augmentation method.

# Results

# The Biography Dataset

We use low-rank adaptation (LoRA) (Hu et al. 2022) to facilitate finetuning of models up to 70 billion parameters. As the corpus size is limited, we use a rank of 16 for the LoRA adapters. Adapters are added to all of the model’s weights except for the embedding and the output layer. We finetune models with the Huggingface’s transformer library (Wolf et al. 2020) on NVIDIA 4090 GPUs. We experiment with LLaMA 3 (Dubey et al. 2024) and Gemma 2 (Team et al. 2024) as two families of language models.

For the baselines, we compare the performance of the models after plain finetuning, random (naive) token-dropout, and token-dropout by attention. In addition to random dropout, dropout by attention uses the original attention weights to guide the dropout probabilities, assuming that the model put more attention on tokens it deemed important. Tokens with lower attention weights are dropped out with higher probabilities to enhance the important information, in a similar vein as in ( $\mathrm { \Delta Y u }$ et al. 2022; Hailemariam et al. 2023). The dropout probabilities are also calculated using Equation 1.

For each experiment, we trained the model from 10 to 30 epochs with learning rates in [5e-5, 1e-3] and selected the model with the best performance. For the augmentationbased methods, we also searched for the best hyperparameters $\alpha$ and $\beta$ individually for each method. Interestingly, the best hyperparameters for the dropout probabilities happen to be similar for different models and augmentation methods. For each of the augmentation methods, we generate 10 augmented versions of each training example and combine them with the original examples.

Results in Table 2 show that the proposed token-dropout augmentation based on attention difference significantly outperforms other data augmentation methods. We report QA accuracy on the “university” and the “company” fields as the models have poor performance on these fields under plain finetuning (Table 1). We report exact match (EM) accuracy and normalized word-level F1 scores. We can see that while random dropout and dropout by attention improve performance over no data augmentation, our method achieves much more significant improvement. We also collected the results of other information from models trained in our method and accuracy increased across models. This proves that contrasting attention of large and small language models indeed finds important but elusive clues in text effectively, and amplifying these clues in the input has immediate positive effects on the model’s memorization efficiency even for the 70B model.

# Real-World Dataset

Aside from the biography dataset, we also evaluate the proposed method on Wikipedia text to verify if the method helps knowledge learning on general text. Specifically, we evaluate on the Paragraph-Level Wikipedia Question-Answering dataset (Du and Cardie 2018). We first perform continual pretraining on the Wikipedia text paragraphs (included in the dataset), then evaluate the model’s performance on the question-answering data4. The questions are specifically designed to incorporate coreference dependencies that span multiple sentences in a paragraph, making it a challenging task that tests the model’s ability to learn and memorize complex factual associations.

An example of Wikipedia text from the dataset is:

The 2005 edition of the International ISBN Agency’s official manual describes how the 13-digit ISBN check digit is calculated. The ISBN-13 check digit, which is the last digit of the ISBN, must range from 0 to 9 and must be such that the sum of all the thirteen digits, each multiplied by its (integer) weight, alternating between 1 and 3, is a multiple of 10.

Table 2: QA performance after continual pretraining on the biography corpus. Data augmentation based on attention difference significantly outperforms other data augmentation methods, for both small and large models.   

<html><body><table><tr><td></td><td colspan="2">Hyper-</td><td colspan="4">QA performance</td></tr><tr><td></td><td>parameters α</td><td>β</td><td>EM</td><td>F1</td><td>University Company EM F1</td><td></td></tr><tr><td>Gemma22B</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Baselines</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Plain finetuning Random token-dropout</td><td></td><td></td><td></td><td></td><td>0.17 0.48 0.18 0.21</td><td></td></tr><tr><td>Token-dropout by attention</td><td>0.6 0.6</td><td>0.05</td><td></td><td></td><td>0.070.38 0.21 0.23</td><td>0.190.51 0.23 0.29</td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Token-dropout by attention diff 0.6</td><td></td><td>0.03</td><td></td><td></td><td>0.250.56 0.32 0.36</td><td></td></tr><tr><td>Gemma 29B Baselines</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Plain finetuning</td><td></td><td></td><td></td><td></td><td></td><td>0.61 0.78 0.63 0.64</td></tr><tr><td>Random token-dropout</td><td>0.7</td><td></td><td></td><td></td><td></td><td>0.520.73 0.51 0.57</td></tr><tr><td>Token-dropout by attention</td><td>0.6</td><td>0.05</td><td></td><td></td><td></td><td>0.49 0.62 0.44 0.47</td></tr><tr><td>Ours Token-dropout by attention diff 0.6</td><td></td><td>0.03</td><td></td><td></td><td></td><td>0.840.920.90 0.92</td></tr><tr><td>LLaMA38B</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Baselines</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Plain finetuning</td><td></td><td></td><td></td><td>0.30 0.55 0.17 0.21</td><td></td><td></td></tr><tr><td>Random token-dropout Token-dropout by attention</td><td>0.6 0.6</td><td>0.05</td><td>0.11 0.49 0.24 0.29</td><td></td><td></td><td>0.24 0.62 0.21 0.28</td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Token-dropout by attention diff 0.7</td><td></td><td>0.05</td><td>0.29</td><td></td><td>0.64 0.42 0.53</td><td></td></tr><tr><td>LLaMA370B Baselines</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Plain finetuning</td><td></td><td></td><td></td><td></td><td></td><td>0.420.69 0.66 0.67</td></tr><tr><td>Random token-dropout</td><td>0.6</td><td></td><td></td><td></td><td></td><td>0.710.86 0.71 0.78</td></tr><tr><td>Token-dropout by attention</td><td>0.7</td><td>0.05</td><td></td><td></td><td></td><td>0.51 0.75 0.61 0.68</td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Token-dropout by attention diff 0.7</td><td></td><td>0.01</td><td></td><td>0.900.96 0.960.96</td><td></td><td></td></tr></table></body></html>

An example of the question from the dataset is as follows:

Question: How many digits does the ISBN have? Answer: 13

Results in Table 3 show that the proposed method also improves knowledge learning from the Wikipedia text. Unlike naive data augmentation, our method improves the model’s memorization efficiency by selectively amplifying difficult and elusive clues. This shows that enhancing the model’s focus on important but elusive information is a crucial factor in improving knowledge learning efficiency, and our method is generally applicable to different kinds of text.

# Conclusion

Efficiency of learning factual knowledge in not only crucial for pretraining, but also important for effective continual and lifelong learning in language models. Due to the overfitting and long-range dependency problem, even performant language models can struggle to learn and memorize factual knowledge from limited data. In this work, we show that one of the key factors to improving the model’s learning, finding the “elusive” but important clues in text, is already embedded in the model’s attention weights. However, such clues are hard to discover by the model itself due to the model’s bias towards short-range contexts, but clearly manifests themselves when contrasting the attention between a larger and a smaller model. Based on this discovery, we propose a simple yet effective data augmentation method that leverages the attention difference to guide the dropout of tokens in the input. Our method significantly improves the model’s performance in memorizing factual knowledge, and is shown to be effective for different corpora and models.

Table 3: QA performance after continual pretraining on the Wikipedia corpus. Data augmentation based on attention difference outperforms other data augmentation methods.   

<html><body><table><tr><td></td><td colspan="2">Hyper-</td><td colspan="2">QA</td></tr><tr><td></td><td colspan="2">parameters</td><td colspan="2">performance</td></tr><tr><td></td><td>α</td><td>β</td><td>EM</td><td>F1</td></tr><tr><td>Gemma 2 2B</td><td></td><td></td><td></td><td></td></tr><tr><td>Baselines Plain finetuning</td><td></td><td></td><td>0.126</td><td>0.215</td></tr><tr><td>Random token-dropout</td><td>0.7</td><td></td><td>0.12</td><td>0.223</td></tr><tr><td>Token-dropout by attention</td><td>0.7</td><td>0.005</td><td>0.145</td><td>0.249</td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td></tr><tr><td>Token-dropout by attention diff</td><td>0.7</td><td>0.005</td><td>0.156</td><td>0.256</td></tr><tr><td>Gemma 29B Baselines</td><td></td><td></td><td></td><td></td></tr><tr><td>Plain finetuning</td><td></td><td></td><td>0.186</td><td>0.287</td></tr><tr><td>Random token-dropout</td><td>0.7</td><td></td><td>0.198</td><td>0.314</td></tr><tr><td>Token-dropout by attention</td><td>0.7</td><td>0.005</td><td>0.205</td><td>0.315</td></tr><tr><td>Ours Token-dropout by attention diff</td><td>0.7</td><td>0.005</td><td>0.231</td><td>0.334</td></tr><tr><td>LLaMA38B</td><td></td><td></td><td></td><td></td></tr><tr><td>Baselines</td><td></td><td></td><td></td><td></td></tr><tr><td>Plain finetuning Random token-dropout</td><td>0.7</td><td></td><td>0.146 0.067</td><td>0.228 0.159</td></tr><tr><td>Token-dropout by attention</td><td>0.7</td><td>0.005</td><td>0.134</td><td>0.239</td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td></tr><tr><td>Token-dropout by attention diff</td><td>0.7</td><td>0.03</td><td>0.172</td><td>0.263</td></tr><tr><td>LLaMA370B Baselines</td><td></td><td></td><td></td><td></td></tr><tr><td>Plain finetuning</td><td></td><td></td><td>0.179</td><td>0.282</td></tr><tr><td>Random token-dropout</td><td>0.7</td><td></td><td>0.187</td><td>0.307</td></tr><tr><td>Token-dropout by attention</td><td>0.7</td><td>0.005</td><td>0.190</td><td>0.288</td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td></tr><tr><td>Token-dropout by attention diff</td><td>0.7</td><td>0.005</td><td>0.212</td><td>0.308</td></tr></table></body></html>