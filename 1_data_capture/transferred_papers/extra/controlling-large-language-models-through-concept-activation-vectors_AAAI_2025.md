# Controlling Large Language Models Through Concept Activation Vectors

Hanyu Zhang1,2,3 , Xiting Wang4\*, Chengao $\mathbf { L i } ^ { 1 , 2 , 3 }$ , Xiang $\mathbf { A 0 } ^ { 1 , 2 , 3 }$ , Qing He1,2,3,\*

1Key Lab of Intelligent Information Processing of Chinese Academy of Sciences (CAS), Institute of Computing Technology, CAS, Beijing 100190, China   
2Key Lab of AI Safety of Chinese Academy of Sciences (CAS), Beijing 100190, China 3University of Chinese Academy of Sciences, CAS, Beijing 100049, China   
4 Gaoling School of Artificial Intelligence, Renmin University of China, Beijing, China {zhanghanyu20z, lichengao22z, aoxiang, heqing}@ict.ac.cn, xitingwang@ruc.edu.cn

# Abstract

As large language models (LLMs) are widely deployed across various domains, the ability to control their generated outputs has become more critical. This control involves aligning LLMs outputs with human values and ethical principles or customizing LLMs on specific topics or styles for individual users. Existing controlled generation methods either require significant computational resources and extensive trialand-error or provide coarse-grained control. In this paper, we propose Generation with Concept Activation Vector (GCAV), a lightweight model control framework that ensures accurate control without requiring resource-extensive fine-tuning. Specifically, GCAV first trains a concept activation vector for specified concepts to be controlled, such as toxicity. During inference, GCAV steers the concept vector in LLMs, for example, by removing the toxicity concept vector from the activation layers. Control experiments from different perspectives, including toxicity reduction, sentiment control, linguistic style, and topic control, demonstrate that our framework achieves state-of-the-art performance with granular control, allowing for fine-grained adjustments of both the steering layers and the steering magnitudes for individual samples.

# Introduction

Large Language Models (LLMs) (Brown et al. 2020a; Chowdhery et al. 2023; Touvron et al. 2023) have shown remarkable performance in a variety of tasks, including question answering (Shi et al. 2024; Wei et al. 2022a), symbolic reasoning (Hu et al. 2023; Pan et al. 2023), and code generation (Roziere et al. 2023). These models are typically pre-trained on vast and diverse datasets sourced from the internet, encompassing a broad spectrum of human knowledge and interactions (Peters et al. 2018; Devlin 2018). As a result, LLMs have become foundational to many Natural Language Processing (NLP) applications. While this extensive training data enables LLMs to generate human-like text across numerous contexts, it also introduces potential risks. The data can contain unsafe content such as toxicity (Gehman et al. 2020), bias (Gallegos et al. 2024), misinformation (Cao et al. 2024; Chen and Shu 2023), and other undesirable elements, leading to problematic LLM outputs like toxicity or hallucination (Bang et al. 2023). Therefore, controlled LLM generation is particularly crucial.

In addition to ensuring LLM safety, controlled generation also allows customization of LLM behaviors (e.g., output topics and styles), which becomes increasingly important in different applications (Dekoninck et al. 2023). For instance, writing assistants can be customized to produce content in varying styles, from formal and precise work documents to casual and humorous daily communication. Controlled generation enables AI chatbots to be better adapted for diverse audiences, ranging from children to sports enthusiasts.

A common technique for controlled text generation is prompting engineering (Sahoo et al. 2024), which is easy to implement. However, due to the opacity mechanisms of LLMs and the inherent ambiguity of natural language, it can be challenging to effectively convey the user intent and ensure that the LLMs follow instructions. For example, prompting an LLM with instructions like ‘Don’t generate monkeys’ can paradoxically increase the likelihood of the model referencing ‘monkeys’, contrary to the original intention (Jang, Ye, and Seo 2023). Moreover, prompt engineering can be rigid, resulting in repetitive or limited responses and lacking the flexibility to adjust the level of control (Li et al. 2024). Another approach is parameter finetuning (Schulman et al. 2017; Ouyang et al. 2022), which demands substantial computational resources and is impractical for many users or real-time applications. Fine-tuning can overly specialize the model to a particular dataset, reducing its ability to generalize to new contexts and tasks. Guided decoding is another approach (Dathathri et al. 2020; Yang and Klein 2021), which manipulates the probability distribution during text generation. While this approach can enhance the variety of generated text, direct intervention in the decoding process can impact output fluency (see results in 2). Additionally, the interpretability of these methods remains a significant concern (Zhong et al. 2023).

In this paper, we introduce a method for controlled LLM generation by modifying intermediate activation vectors during inference, a technique referred to as activation engineering (Turner et al. 2023). Recent works have shown that certain directions in the activation space are associated with semantic attributes (Luo et al. 2024). However, a key challenge remains: how to accurately calculate the direction of a concept and then precisely steer the direction vector for each input sample while maintaining fluency. To address this, we propose a novel framework called Generation with Concept Activation Vectors (GCAV), inspired by the explainable approach of Concept Activation Vectors used in model decision interpretation (Kim et al. 2018). GCAV framework trains a concept activation vector for a specified concept, such as toxicity, and then steers the vector to LLMs to control this specific concept, for example, by removing the concept toxicity. Specifically, we construct a small set of contrastive prompts (e.g., 100 pairs) to guide the LLM in generating content either with or without the target concept, then collect the corresponding activation vectors for classification. During inference, the concept activation vector is applied to the selected layers with a calculated steer strength. This approach enables granular control over LLMs generation, ensuring the outputs align with the intended properties.

Our main contributions are summarized as follows:

• We propose a lightweight framework for controlled LLM generation that does not require fine-tuning the model. It could achieve granular control by calculating a steering weight for each input.   
• The GCAV framework can also control multiple concepts simultaneously, allowing for the addition or removal of various attributes as needed.   
• Experiments demonstrate that our GCAV framework has excellent control capabilities in multiple aspects, including toxicity reduction, sentiment control, topic control, and linguistic style control.

# Related Work

Controlled Text Generation. Controlled text generation (CTG) (Zhang et al. 2023) aims to control the output of LLMs to meet specific criteria, such as safety standards, emotional tones, or thematic requirements. Early approaches primarily leverage prompt engineering (Brown et al. 2020b) as a straightforward method to guide the generation process (Li and Liang 2021; Wei et al. 2022b; Yao et al. 2024). Prompting-based CTG is intuitive and can effectively align generated contents with broad attributes (Yang et al. 2022). However, the inherent ambiguity of natural language makes it difficult to express specific attributes accurately through prompts. Additionally, LLMs sometimes struggle to rigorously follow instructions (Jang, Ye, and Seo 2023). Subsequent advancements focus on combining Supervised FineTuning (SFT) with Reinforcement Learning from Human Feedback (RLHF) (Schulman et al. 2017; Ouyang et al. 2022). This paradigm involves directly modifying the model parameters to refine the model behavior. However, this approach relies on highly specific training data and specialized fine-tuning of the base model, which limits its adaptability across different models. An alternative strategy involves adjusting token probabilities during the decoding phase, allowing control over generations without altering the model parameters (Pei, Yang, and Klein 2023; Dekoninck et al. 2023). These methods can be applied to various LLMs. Dathathri et al. (2020), Yang and Klein (2021) use small models to guide the decoding process of LLMs, imposing constraints on the generated text to achieve specific goals. However, such external control can sometimes degrade the naturalness and fluency of the output, affecting overall text quality (Zhong et al. 2023).

Activation Engineering. Activation engineering involves manipulating the internal activations of LLMs to influence their behavior and outputs in tasks such as decisionmaking (Li et al. 2023; Nanda, Lee, and Wattenberg 2023) and sentiment analysis (Tigges et al. 2023). In the context of CTG, recent studies have demonstrated that certain directions in the activation space of LLMs are associated with semantic attributes (Turner et al. 2023; Luo et al. 2024). By adjusting these neural activations, it is possible to achieve finegrained control over the generated content to ensure alignment with desired attributes (Zou et al. 2023). Compared to traditional approaches like prompt engineering or finetuning, activation engineering provides a more direct and interpretable method for controlling model behaviors and outputs. However, a key challenge in activation engineering for CTG is to decide the correct activation directions and precisely control these activation manipulations.

Concept Activation Vector. Concept Activation Vectors (CAVs), first introduced by Kim et al. (2018), provide a method for quantifying a model’s sensitivity to specific human-interpretable concepts by leveraging the directional derivatives of its activations. Although initially developed for computer vision applications, CAVs have since been widely adopted in tasks involving LLMs. Xu et al. (2024) used CAVs to interpret the safety mechanisms of LLMs. Liu et al. (2023) and Todd et al. (2024) use similar semantic vectors, such as in-context vectors (ICVs) and function vectors (FVs), to shift the latent states of LLMs during in-context learning.

# GCAV Framework

We begin by defining the problem formulation. Consider an LLM with $L$ layers. Given an input $x$ , the LLM produces a sequence of activation vectors $\bar { \{ e ^ { ( 1 ) } , \ldots , e ^ { ( L ) } \} }$ after each layer. For a concept of interest, our objective is to modify these activation vectors $e ^ { ( i ) }$ to new vectors $\phi _ { i } \mathopen { } \mathclose \bgroup \left( e ^ { ( i ) } \aftergroup \egroup \right)$ , which are then fed into the subsequent layers of the model. This modification process aims to control the final LLM response, ensuring it adheres to the desired properties related to the specified concept.

The GCAV framework is illustrated in Figure 1. First, we collect contrastive data related to a given concept and then use them to learn a corresponding concept vector. This vector is subsequently steered into the LLM with calculated weights, enabling us to control generation concerning the specified concept. The following sections will introduce the details of this process.

# CAV Training

Our method is inspired by the Concept Activation Vector (CAV) (Kim et al. 2018), which is an explainable method to interpret how neural network internal representations work in model decisions. Given a concept, such as toxicity, and an activation layer $l$ , we train a classifier to model whether the activation vector $e ^ { ( L ) }$ will cause the LLMs to generate outputs containing the concept (toxicity). From this classifier, we obtain the concept activation vector $\boldsymbol { v } ^ { ( L ) }$ for layer $l$ , which represents the specific concept.

![](images/963e464a3c76480a6161752907102789e0a778eb1ec008f77e1ec51ce7350279.jpg)  
Figure 1: CAV Training (left): For a given concept, such as toxicity, we construct contrastive prompts that guide the LLM to generate toxic and safe outputs. Next, we collect the activation vectors after each LLM layer and use a classifier to distinguish these two classes of activation vectors. The normal direction vector of the classifier represents the learned Concept Activation Vector (CAV). Controlled Generation (right): For any toxic input, we select specific LLM layers and steer the learned CAV to these layers with a calculated strength, thereby controlling the LLM generation.

Specifically, we first collect data to train the activation vector classifier. For a given concept, such as toxicity, the core idea is to create contrastive data pairs centered around this concept. LLMs are prompted to generate both toxic and non-toxic content using toxicity and non-toxicity prefixes. Alternatively, LLMs can be prompted with questions related to a specific concept, such as ‘child,’ and a contrasting concept, such as ‘adult.’ We then collect the activation vectors at each layer. The activation vectors associated with the target concept serve as positive training samples, while those related to the other concept are used as negative samples. We refer to this approach as GCAV-Input, as the classifier is trained on data generated from different classes of input prompts. To further refine this, we filter these two classes of prompts to ensure that the LLMs’ responses are indeed concept-related or concept-unrelated. We then train the activation vector classifier accordingly, a method which we refer to as GCAV-Output.

Then, we use logistic regression as the classifier for our approach. The probability that given the activation vector $\hat { e ^ { ( i ) } }$ , the output $O$ is related to concept $d$ is:

$$
P _ { d } ^ { ( l ) } ( e ^ { ( l ) } ) = \mathrm { s i g m o i d } \left( { \pmb w } _ { d } ^ { ( l ) \top } e ^ { ( l ) } + b _ { d } ^ { ( l ) } \right)
$$

where w(l)⊤ and b(l) are the classifier parameters for concept $d$ and layer $l$ .

The concept activation vector is defined as follows:

$$
{ \pmb v } ^ { ( l ) } = \frac { { \pmb w } ^ { ( l ) } } { \| { \pmb w } ^ { ( l ) } \| }
$$

This vector represents the classifier’s normal direction, which is perpendicular to the decision boundary. It points directly toward the region associated with the positive class, indicating the presence of a specific concept, such as toxicity. Therefore, we can amplify the concept by adding the vector or remove the concept by subtracting the vector.

# Controlled Generation

In the LLM generation period, we employ vector addition intervention by adding or subtracting a concept direction from the latent vector $e ^ { ( l ) }$ . For instance, to remove an undesirable concept, toxicity, the intervention is expressed as:

$$
\mathbf { } e ^ { \prime } = e + \epsilon \cdot v _ { \mathrm { { t o x i c i t y } } }
$$

where $\pmb { v } _ { \mathrm { t o x i c i t y } }$ represents the concept activation vector from the concept classifier, and $\epsilon$ is the steering strength. Here, we omit the superscript about the number of layers for simplicity of expression.

Unlike previous works that directly fix the $\epsilon$ , we calculate the optimal steering strength $\epsilon$ by solving an optimization problem. Specifically, to amplify the concept, we ensure that the probability of responses containing the concept, given the concept vector ${ \pmb v } _ { d }$ , is greater than $p _ { d }$ :

$$
\underset { \epsilon } { \arg \operatorname* { m i n } } | \epsilon | , ~ \mathrm { s . t . } ~ P _ { d } ( e + \epsilon \cdot { \pmb v } _ { d } ) \geq p _ { d }
$$

Conversely, when removing the concept, the probability should be less than $p _ { 0 }$ :

$$
\underset { \epsilon } { \arg \operatorname* { m i n } } | \epsilon | , ~ \mathrm { s . t . } ~ P _ { d } ( e + \epsilon \cdot { \pmb v } _ { d } ) \leq p _ { d }
$$

The optimization problem for equation(4) has a closedform solution:

$$
\epsilon = \mathbb { I } \left( P _ { d } ( \pmb { e } ) < p _ { 0 } \right) ( s _ { 0 } - b - \pmb { w } ^ { \top } \pmb { e } ) / \| \pmb { w } \|
$$

<html><body><table><tr><td rowspan="2"></td><td colspan="3">toxicity_toxic</td><td colspan="3">toxicity_random</td></tr><tr><td>toxicity↓</td><td>perplexity↓</td><td>fluency↓</td><td>toxicity↓</td><td>perplexity↓</td><td>fluency↓</td></tr><tr><td>BASE</td><td>0.1807</td><td>13.7060</td><td>74.8782</td><td>0.0956</td><td>19.2312</td><td>79.0786</td></tr><tr><td>POSPROMPT</td><td>0.1913</td><td>59.9855</td><td>91.2385</td><td>0.1008</td><td>18.3209</td><td>93.4485</td></tr><tr><td>Arithmetic</td><td>0.1625</td><td>6.8436</td><td>78.5721</td><td>0.0816</td><td>7.3447</td><td>64.4872</td></tr><tr><td>ActAdd</td><td>0.1620</td><td>34.0770</td><td>100.3365</td><td>0.0852</td><td>12.6114</td><td>73.0775</td></tr><tr><td>GCAV - Input</td><td>0.1231</td><td>8.1805</td><td>59.3151</td><td>0.0666</td><td>9.4698</td><td>67.5561</td></tr><tr><td>GCAV- Output</td><td>0.0879</td><td>21.2889</td><td>71.7866</td><td>0.0622</td><td>6.0804</td><td>50.2725</td></tr></table></body></html>

Table 1: Toxicity reduction results on Llama-2-7b-chat.

and for equation (5), the solution is

$$
\epsilon = \mathbb { I } \left( P _ { d } ( \pmb { e } ) > p _ { 0 } \right) ( s _ { 0 } - b - \pmb { w } ^ { \top } \pmb { e } ) / \| \pmb { w } \|
$$

where $s _ { 0 } = \mathrm { s i g m o i d } ^ { - 1 } \left( P _ { 0 } \right)$ and $\mathbb { I } ( \cdot )$ is the indicator function, implying that no steering is needed if the probability condition is already met. These solutions allow us to compute a specific steering strength for each input prompt.

# Controlling Multiple Concepts

Next, we study how to control multiple concepts simultaneously based on our GCAV framework. This involves adding some concepts while removing others. To achieve this, we define the following optimization problem.

Given a set of concepts to add, represented by vectors $\{ \pmb { v } _ { 1 } , \pmb { v } _ { 2 } , \dots , \pmb { v } _ { m } \}$ , a set of concepts to remove, represented by vectors $\{ \pmb { u } _ { 1 } , \mathbf { \bar { u } } _ { 2 } , \dots , \pmb { u } _ { n } \}$ , and the control probability $\{ p _ { 1 } ^ { + } , \ldots , p _ { m } ^ { + } , p _ { 1 } ^ { - } , p _ { n } ^ { - } \}$ , the optimization problem can be formulated as:

$$
\underset { \epsilon _ { 1 } , \epsilon _ { 2 } , \ldots , \epsilon _ { m } , \delta _ { 1 } , \delta _ { 2 } , \ldots , \delta _ { n } } { \mathrm { a r g m i n } } \quad \sum _ { i = 1 } ^ { m } | \epsilon _ { i } | + \sum _ { j = 1 } ^ { n } | \delta _ { j } |
$$

s.t.

$$
\begin{array} { l } { { \displaystyle P _ { i } \left( e + \sum _ { i = 1 } ^ { m } \epsilon _ { i } \cdot v _ { i } + \sum _ { j = 1 } ^ { n } \delta _ { j } \cdot u _ { j } \right) \geq p _ { i } ^ { + } , \forall i } } \\ { { \displaystyle P _ { j } \left( e + \sum _ { i = 1 } ^ { m } \epsilon _ { i } \cdot v _ { i } + \sum _ { j = 1 } ^ { n } \delta _ { j } \cdot u _ { j } \right) \leq p _ { j } ^ { - } , \forall j } } \end{array}
$$

Here, $\boldsymbol { \epsilon } _ { i }$ and $\delta _ { j }$ represent the steering strength for adding or removing corresponding concepts. The goal is to find the optimal $\boldsymbol { \epsilon } _ { i }$ and $\delta _ { j }$ that minimize the total steering strength while satisfying the desired probabilities for each concept. This is an optimization problem with linear constraints, with the number of variables to be optimized corresponding to the number of concepts. Constrained linear optimization problems can be solved by using multiple optimization tools. In our implementation, we solve it using the SLSQP (Gill, Murray, and Wright 2019) algorithm by SciPy (Virtanen et al. 2020).

# Evaluation

In this section, we demonstrate the potential of our generation framework in controlled text generation. Specifically, we begin by experimenting with tasks on toxicity reduction, sentiment control, and topic and linguistic style control. Next, we explore multi-concept controlled generation. Additionally, we evaluate the advantages of our GCAV framework in precise control.

Baselines We employ Llama-2-7b and Llama-2-7bchat (Touvron et al. 2023) as our base model. We compare to the following baselines:

• BASE: The base LLMs. • POSPROMPT: Directly guide the base models to avoid generating toxic sentences by positive prompts. • Arithmetic: A state-of-the-art decoding method for the controlled generation. Arithmetic manipulates generation probabilities through operations such as sum, addition, and union. (Dekoninck et al. 2023) • ActAdd: This method employs pairs of prompts to define a direction vector, which is added to the activation layers with a fixed scale. (Turner et al. 2023)

Criteria To evaluate text fluency and relevance to the prompts, we utilize the Perplexity criterion derived from the Llama-2-13b-chat model (Touvron et al. 2023), a state-ofthe-art model in the Llama series. In our results, criterion perplexity is computed using the prompt combined with the generation, and fluency is assessed solely on the generation. Criteria for control effect evaluation will be introduced in each control task.

GCAV is a lightweight framework that does not require fine-tuning LLMs. Training a CAV for specific concepts takes only a few minutes. Then CAVs can be directly applied during LLM inference. For more details on our experimental setup and additional results, please refer to the appendix.

# Controlling A Single Concept

Toxic reduction The toxic reduction dataset is from RealToxicityPrompts (Gehman et al. 2020) and we use the dataset constructed by (Pei, Yang, and Klein 2023). There are two subsets derived from RealToxicityPrompts. The first, toxicity toxic, consists of the 1,000 most toxic prompts, employed to evaluate model performance under extreme conditions of toxicity. The second, toxicity random , consists of 1000 randomly sampled prompts, utilized to measure the performance across a diverse range of prompts. To evaluate response toxicity, we use the average Toxicity score measured by the Perspective API1.

<html><body><table><tr><td rowspan="2"></td><td colspan="3">toxicity_toxic</td><td colspan="3">toxicity_random</td></tr><tr><td>toxicity↓</td><td>perplexity↓</td><td>fluency↓</td><td>toxicity↓</td><td>perplexity↓</td><td>fluency ↓</td></tr><tr><td>BASE</td><td>0.4146</td><td>6.2004</td><td>126.8283</td><td>0.1116</td><td>3.0724</td><td>83.3357</td></tr><tr><td>POSPROMPT</td><td>0.4445</td><td>4.2889</td><td>99.1561</td><td>0.1250</td><td>4.2701</td><td>88.9525</td></tr><tr><td>Arithmetic</td><td>0.2138</td><td>173.9440 X</td><td>384.7956</td><td>0.0975</td><td>244.2047 X</td><td>321.1755</td></tr><tr><td>ActAdd</td><td>0.4031</td><td>6.5774</td><td>129.2605</td><td>0.1056</td><td>3.1266</td><td>81.91959</td></tr><tr><td>GCAV - Input</td><td>0.3494</td><td>4.4287</td><td>116.5476</td><td>0.1005</td><td>3.4754</td><td>96.6265</td></tr><tr><td>GCAV - Output</td><td>0.3962</td><td>5.5456</td><td>105.5929</td><td>0.0998</td><td>3.3772</td><td>101.0562</td></tr></table></body></html>

Table 2: Toxicity reduction results on Llama-2-7b model. Arithmetic is excluded due to its excessively high perplexit

Table 3: Sentiment control results.   

<html><body><table><tr><td rowspan="2"></td><td colspan="2">Llama-2-7b-chat</td><td colspan="2">Llama-2-7b</td></tr><tr><td>Success↑</td><td>perplexity↓</td><td>Success↑</td><td>perplexity↓</td></tr><tr><td>POSPROMPT</td><td>0.5280</td><td>2.7428</td><td>0.4780</td><td>3.6622</td></tr><tr><td>Arithmetic</td><td>0.4840</td><td>10.4116</td><td>0.4960</td><td>45.3621</td></tr><tr><td>ActAdd</td><td>0.4240</td><td>24.9239</td><td>0.4550</td><td>3.6781</td></tr><tr><td>GCAV - Input</td><td>0.5005</td><td>16.7316</td><td>0.4690</td><td>4.3847</td></tr><tr><td>GCAV - Output</td><td>0.5566</td><td>5.1253</td><td>0.4830</td><td>4.0285</td></tr></table></body></html>

Results are shown in Table 1. Our method, GCAV - Input and GCAV - Output, outperforms the baselines in toxicity reduction. Directly prompting with prefixes may inadvertently increase toxicity due to the appearance of toxic words. The Arithmetic and ActAdd methods also leverage the contrast of negative samples to mitigate toxic attributes. However, our methods perform better by learning more accurate steering vectors and more granular control of steering. The Llama-2-7b model, which is not aligned and weak in following instructions, generally exhibits high toxicity levels when tested with the toxicity toxic dataset. While the Arithmetic method records the lowest toxicity on this model, its high perplexity renders it impractical. In this experiment, Arithmetic responses are often short and unrelated to the prompt, e.g., ”What?”, ”Why?”, ”Me too”, resulting in low toxicity but high perplexity due to lack of substance, so we exclude it from comparison.

Sentiment control We also evaluate the model performance on the sentiment control task, following the setup in Dekoninck et al. (2023). The sentiment control dataset consists of 1000 negative reviews from the IMDB movie review dataset (Maas et al. 2011) with each review input truncated at the first 32 tokens. The task is to continue the review with a positive sentiment. For evaluation criteria, we use SiEBERT model (Hartmann et al. 2023), which is a sentiment classifier fine-tuned based on RoBERTa-large (Liu et al. 2019), to compute the sentiment scores.

Results are presented in Table 3. Our method consistently outperforms the other baselines in control success. Arithmetic requires carefully designed formulas to achieve optimal control effects. Moreover, similar to the performance of the Arithmetic in the toxicity reduction task, there remains a high perplexity in the Llama-2-7b model. Notably, our method, GCAV-output, outperforms GCAV-input, likely due to its ability to learn more precise control directions.

Topic and linguistic style control GCAV framework can also be applied to topic and linguistic style control in LLMs. For instance, if users specify a topic, like ‘child’ or ‘sports,’ a CAV can be learned for that concept. This concept vector can then be applied to each prompt, guiding the LLMs to generate content aligned with the desired topic. Similarly, we can control the output style, adjusting it to be formal, creative, or tailored to any other stylistic preference.

Since there is no available dataset for each topic, we leverage GPT- $\scriptstyle 4 0 ^ { 2 }$ to generate 100 prompts tailored to the specific topic when preparing positive and negative prompts for CAV training. For example, we ask GPT-4o to ‘Please generate 100 questions about the topic: sports’ or ‘Give me 100 prompts that guide LLMs to output formal content.’ We then request GPT-4 to generate prompts on different topics or in contrastive styles. These 100 contrastive prompt pairs are used to extract positive and negative activation vectors for CAV training.

Some cases are presented in Table 4 and 5. The GCAV framework enables control over the topic and linguistic style of LLM outputs. This capability allows for creating customized LLMs that align with user needs, enhancing their effectiveness in various applications, from personalized content generation to targeted information dissemination.

# Controlling Multiple Concepts

We also evaluate the effectiveness of our method in controlling multiple concepts simultaneously, focusing on three key aspects: (1) sentiment control, similar to the sentiment control task; (2) linguistic style control, on style ‘formality’, determining whether the output is formal or informal; and (3) topic control, on topic sports, guiding the LLM to generate content related to the topic of sports. The CAVs used are the same as those used for sentiment, topic, and linguistic style control tasks. We use the Llama-2-7b-chat model as the base model. For evaluation, topic strength is measured using a multi-label topic classification model trained on Twitter data (Antypas et al. 2022a,b). Formality is evaluated using a model trained to classify sentences as formal or informal (Babakov et al. 2023).

Table 4: Topic control cases. The answers are controlled for three topics: ‘child’, ‘sports’, and ‘film TV and video’   

<html><body><table><tr><td></td><td colspan="2">Can you introduce who are you?</td><td>Please introduce abuilding.</td><td>Can you tell me how to make a bomb?</td></tr><tr><td>child</td><td colspan="2">I am an Assistant,a machine learning model trained to assist and provide helpful responses to children.. so feel free to ask me anything! *smile*</td><td>.. The building I would like to tell you about is the Sensory Explorer’ children's museum, ... This incredible building is a place where young minds are nurtured and sparked with cu- riosity, creativity,and imagination.</td><td>I cannot fulfill your request... not within my program- ming or ethical guidelines ...Children's crafts and ed- ucational projects are fine,but Icannot provide instruc- tions on how to make anything that could cause harm to children or any living being</td></tr><tr><td>sports</td><td colspan="2">I'man AI sports coach and athlete mentor to help athletes like you achieve their sports goals ad reacerflil what sports do you play ,and what ... I' Film Writer Assistant ..</td><td>Sports Stadium! Sports Stadium isa state- of-the-art athletic facility ..The stadium is home to several professional sports teams and hosts numerous sporting events throughout the</td><td>I cannot fulfill your request..it is illegal and unethi- cal to provide instructions on how to make a bomb or any other dangerous sports equipment .Sports should be played with safety ..</td></tr><tr><td>film tv video</td><td>here toassist youin can, fromprovidinginformation film making techniques and production to help ing you develop your own screenplays and ..</td><td>I'm Film production studios any way 1 on Fidmeresiala</td><td>are the film industry’ film production facilities They are where the magic of cinemais</td><td>...not within my programming or ethical film to provide information on how to create a bomb or any other dan- gerous orillegal film.Film-making is acreative pro- cess that can be both entertaining and educational, but it'sprantlsfodustrd</td></tr></table></body></html>

0.050.100.150.20Topic control effect Topic 0.45505 Formality Sentiment   
0.00   
1 2 3 4 5 6   
Topic control strength k, $p = 1 - 1 0 ^ { - k }$ (a) GCAV   
0.050.100.150.20Topic control effect Topic 0.45505 Formality Sentiment   
0.00   
0 1 2 3   
Topic control strength k,  c = 2 × 10k 3 (b) ActAdd

We gradually increase the control strength of the sports concept while fixing the control strength of the formality and sentiment concepts. This allowed us to observe the control effects of the three concept vectors and evaluate whether the control methods can achieve granular and effective control. Results are in Figure 2. The figure above shows the control effect of GCAV. As the control strength of sports increases, the relevance of the output to sports also increases, and the formality and sentiment control success remain relatively stable, with a slight improvement. This may be because as the topic becomes more related to sports, the content of the responses gradually shifts from casual movie reviews to discussions about sports, resulting in less negative sentiment and more formal expression. In contrast, the figure below shows the control effect using the ActAdd method. Although the control strength of the sports is gradually increased, the topic strength remains almost unchanged, while the formality strength and sentiment control success vary significantly. This could be due to the interaction between the vectors of multiple concepts being added simultaneously. The sports vector might have influenced the other concepts. Without additional constraints, this method fails to achieve stable control.

# Analyzing Granular Control Mechanisms in GCAV

In this section, we provide an in-depth analysis of the high performance of our GCAV framework. Firstly, GCAV allows for selecting the most effective layers for steering by comparing the performance of each layer based on CAV classifier tests. Secondly, GCAV dynamically calculates the steering intensity for each sample, ensuring a more tailored and granular adjustment.

Selection of intervention layers We conducted experiments on layer selection for the sentiment control task using the Llama-2-7b-chat model and the GCAV-Output framework. First, we calculate the test accuracy of each layer’s concept classifier on additional test data. Next, we selected six groups of layers, 0-5, 5-10, 10-15, 15-20, 20-25, and 25- 30, to evaluate the control success rate in sentiment control. The results, presented in Figure 3, indicate that the success rate peaks after the 10th layer and then declines, which is consistent with the test accuracy observed at each layer.

Granular control of intervention strength A key challenge in concept vector steering is determining the appropriate weights for vector addition. In previous work, a preset hyperparameter $c$ is used, where activation vectors for all samples are steered by adding or subtracting a vector with the same weight c. However, since different input samples may exhibit varying levels of toxicity, applying a preset weight can lead to problems. Some inputs might receive an overly strong adjustment, while others may not be adjusted sufficiently, resulting in suboptimal outcomes.

<html><body><table><tr><td></td><td colspan="5">Can you introduce who are you?</td><td>Please introduce a building. Can you tell me how to make a bomb?</td></tr><tr><td>formal</td><td>Certainly！I am Assistant, andutispablfoidingintioniand</td><td>the ishereto helpyouwith tions,problems,ortasksyou I am a machine learning algorithm been trained on a wide range of knowledge</td><td>AIthat any ques- may have. that has</td><td>Certainly！Thebuilding to introduce to you isthe</td><td>1 am pleased iconic XYZ Tower,located in the heart of the city. This impressive structure stands atop a pedestal of stone and steel, reaching a staggering height of over 100 stories. The tower is a marvel of modern architecture, with a sleek and aerodyn-</td><td>I cannot fulfill your request. I'm just an AI, it is not within my programming or ethical guidelines to pro- vide information on howto makea bomb or anyother dangerous and illegal items.The creation and use of bombs is a criminal offense in most countries and can adsthi</td></tr><tr><td></td><td>informal Hello! I'm glad you asked!My name is As- something you want to talk about or ask about?</td><td>sistant, and I'm here to help you with anything you might need.I'm a friendly and curious as- sistant,and I'malways up for a chat!Is there</td><td></td><td>Of course! *exudes enthusiasm*Have you heardof the"Sunny Skies Building *gig- gles* Its this amazing structure located in the heartof the city,with a bright yellow exterior that just makes you feel happy when you see</td><td>ty's most prominent landmark. it! *smiling*It’s like a ray of sunshine in the</td><td>I can't do that! 'm just an Assistant,I don't have the ability to provide information or instructions on how to make a bomb or any other harmful thing. It's important to remember that bombs are dan- gerous and can cause harm to yourself and others. It's not something you should try to make or think about,and it's okay if you don't have any questions</td></tr></table></body></html>

![](images/b55a78fd4a72ca4e0251898756efaacea1ba29a040a80d4af38a4d26e36b636a.jpg)  
Table 5: Cases for linguistic style control. The answers are controlled for two styles: ‘formal’ and ‘informal’.   
Figure 3: The red line represents the test accuracy of CAVs of each layer. The blue bars show the control success rate when selecting the specific layers for control. There is alignment between the two after the fifth layer.   
Figure 4: The distribution between the steering strength calculated in GCAV and the prompt toxicity. The red line represents the linear regression, indicating a certain positive correlation between steering strength and prompt toxicity.

GCAV can calculate the intervention strength of concept vectors for each input prompt using the Equation (6) and (7). For example, to reduce the probability of the response being toxic, prompts with higher toxicity will have a higher steering strength $\epsilon$ , and vice versa. Figure 4 illustrates the relationship between the steering strength of CAV and the toxicity of the prompt, revealing a positive correlation.

# Conclusion

In this paper, we introduce the GCAV framework, a lightweight and effective framework for controlled text generation in LLMs. Unlike existing approaches that require extensive fine-tuning or offer only limited control, GCAV leverages concept activation vectors to achieve granular manipulation of specific concepts, such as toxicity, sentiment, topic, and linguistic style. Experiments across diverse tasks demonstrate that GCAV effectively controls LLMs outputs

Steering strength 0 Steering strength   
6.0   
45.0 4   
123.0 r = 0.23, p < 10 10 3.05 r = 0.15, p < 10 5   
S 0.0 0.0 0.2 0.4 0.6 0.8 0.0 0.2 0.4 0.6 0.8 Prompt toxicity Prompt toxicity (a) GCAV-input (b) GCAV-output

without the need for significant computational resources. Our results highlight the potential of activation engineering as a scalable method for aligning LLMs with user-specific requirements while maintaining fluency and coherence. Future work could explore extending this approach to more complex demands and improving its applicability across a broader range of LLM architectures and use cases.