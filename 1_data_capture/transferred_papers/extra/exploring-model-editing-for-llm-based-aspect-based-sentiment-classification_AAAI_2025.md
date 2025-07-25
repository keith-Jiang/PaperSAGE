# Exploring Model Editing for LLM-based Aspect-Based Sentiment Classification

Shichen $\mathbf { L i } ^ { 1 }$ Zhongqing Wang1\*, Zheyu Zhao1, Yue Zhang2, Peifeng Li1

1Natural Language Processing Lab, Soochow University, Suzhou, China 2Westlake University {scli06,zyzhao0104}@stu.suda.edu.cn, $\{ \mathrm { z q } \mathrm { \dot { w } a n g } , \mathrm { p f i } \} \ @$ suda.edu.cn zhangyue $@$ westlake.edu.cn

# Abstract

Model editing aims at selectively updating a small subset of a neural model’s parameters with an interpretable strategy to achieve desired modifications. It can significantly reduce computational costs to adapt to large language models (LLMs). Given its ability to precisely target critical components within LLMs, model editing shows great potential for efficient fine-tuning applications. In this work, we investigate model editing to serve an efficient method for adapting LLMs to solve aspect-based sentiment classification. Through causal interventions, we trace and determine which neuron hidden states are essential for the model’s predictions. By performing interventions and restorations on each component of an LLM, we identify the importance of these components for aspect-based sentiment classification. Our findings reveal that a distinct set of mid-layer representations is essential for detecting the sentiment polarity of given aspect words. Leveraging these insights, we develop a model editing approach that focuses exclusively on these critical parts of the LLM, leading to a more efficient method for adapting LLMs. Our in-domain and out-of-domain experiments demonstrate that this approach achieves competitive results compared to the currently strongest methods with significantly fewer trainable parameters, highlighting a more efficient and interpretable fine-tuning strategy.

# Introduction

![](images/6eaf8845f27ca11a401f46ca6b15bfd0e7b19a0e0e51df98982bdd8bb2448bba.jpg)  
Figure 1: An illustration of our 2-steps model editing method for LLM-based aspect-based sentiment classification.

Aspect-based sentiment classification (Hu and Liu 2004) aims to determine the sentiment polarity of the aspects within a sentence. Traditional ABSC methods (Tang et al. 2016; Tay, Tuan, and Hui 2018; Zhang, Zhou, and Wang 2022) have often relied on smaller, task-specific language models which frequently fall short in capturing complex contextual dependencies and diverse expressions of sentiment.

As language models continue to grow in size, adapting these large language models (Touvron et al. 2023; Ouyang et al. 2022) to ABSC through full-parameter fine-tuning has become increasingly challenging. Various parameterefficient fine-tuning methods (Hu et al. 2022; Houlsby et al. 2019; Li and Liang 2021; Liu et al. 2024) have been developed to address the inefficiencies of full-parameter finetuning and shown strong performance in ABSC (Li et al. 2022), but these methods still need to update across broad sections of the model’s architecture, potentially leading to redundancies and inefficiencies.

Model editing (Mitchell et al. 2022; Meng et al. 2022; Dai et al. 2022) gives a promising way to further reduce the cost by selectively updating a small subset of a neural model’s parameters with an interpretable strategy and achieve desired modifications.

Recent works in model editing (Meng et al. 2023; Mitchell et al. 2022; Meng et al. 2022; Wu et al. 2024) has shown effectiveness and efficiency in various tasks by modifying LLMs with minimal changes to their parameters. These methods highlight that targeted updates to specific neural network components can achieve precise modifications, thereby avoiding the extensive resource requirements of full-parameter fine-tuning. However, the research question of how to edit LLMs for ABSC is still under-explored.

Intuitively, aspect words significantly influence the model’s output, allowing us to perform precise causal analysis and examine how changes to specific components affect overall performance. This enables us to accurately identify and modify the model parameters closely associated with sentiment classification, thereby validating the potential of model editing in enhancing model adaptability and efficiency.

Given the above observations, we consider a model editing method to improve parameter-efficient fine-tuning for ABSC. The method consists of two steps as shown in Figure 1. As shown in Figure 1(a), we first employ causal interventions to identify which components are critical for the LLM’s predictions associated with sentiment. By performing interventions and restorations on each layer of the model, we determine their relative importance for the ABSC task. Experiments reveal that the mid-layer representations of specific aspect words in certain positions are particularly critical for accurately identifying the sentiment polarity. Based on the insights, we propose a precise model editing method that targets critical parts of the LLM using two model editing approaches: weight-based and representationbased editing, as shown in Figure 1(b). For weight-based editing (Hu et al. 2022), we use low-rank adaptation to edit specific model weights. For representation-based editing (Wu et al. 2024), we manipulate a small fraction of model representations at specific positions of aspect terms to steer the model’s behavior.

We conduct extensive experiments to evaluate the method in both in-domain and out-of-domain scenarios, aiming to validate its effectiveness and generalization. The experiments confirm that our method achieves competitive results with a minimal number of parameters in both scenarios. Additionally, we empirically investigate the impact of the proposed model editing method on specific layers and analyze the influence of modifications at specific positions of different words. The results show that the mid-layer representations of specific aspect words are crucial for accurately obtaining the correct predictions, aligning with the findings of causal tracing results.

The main contributions of this work can be summarized as follows:

• We propose a novel framework that combines causal tracing and model editing to address specific downstream tasks with LLMs.   
• We develop a more parameter-efficient fine-tuning method specifically tailored for the aspect-based sentiment classification task.   
• The proposed method achieves competitive results compared to state-of-the-art PEFT methods with significantly fewer trainable parameters, highlighting a more efficient and interpretable fine-tuning strategy.

# Related Work

In this section, we introduce two related topics of this study: aspect-based sentiment classification and model editing.

# Aspect-Based Sentiment Classification

Aspect-based sentiment classification methods have seen significant changes with the development of neural networks (Zhang et al. 2023). Early approaches often relied on full-parameter fine-tuning of smaller models such as LSTM and CNN with attention mechanisms to solve ABSC task (Tang et al. 2016; Ma et al. 2017; Huang and Carley 2018; Xue and Li 2018). However, these models had limitations in understanding complex contexts and long-range dependencies. The development of pre-trained language models brought significant improvement (Sun, Huang, and Qiu 2019; Xu et al. 2019; Jiang et al. 2020). By fine-tuning these models on ABSC tasks, researchers were able to better capture the context and improve ABSC performance. However, as models grew larger, the computational cost and resource demands of full-parameter fine-tuning became increasingly prohibitive. To address these challenges, parameter-efficient fine-tuning methods (Hu et al. 2022; Liu et al. 2024) emerged. These methods enable the adaptation of LLMs to ABSC tasks by tuning a smaller subset of parameters. Due to the strong foundational capabilities of LLMs, parameter-efficient fine-tuning methods can achieve competitive performance compared to previous carefully designed methods (Li et al. 2022). While these methods only update a small portion of parameters within each layer, the updates still span the entire model, leading to some unnecessary redundancies.

# Model Editing

With the development of large language models, fullparameters fine-tuning has become challenging. To alleviate this inefficiency, model editing (Meng et al. 2022, 2023) is designed to update these models in a highly targeted and precise manner. Recent advancements in model editing (Yao et al. 2023) can be categorized into two types: knowledge integration and parameter modification.

Knowledge integration aims to augment the model with new information without updating its original parameters. Recent works on activation steering (Han et al. 2024; Turner et al. 2023; Avitan et al. 2024) and representation engineering (Zou et al. 2023; Liu, Xing, and Zou 2023) shows that adding fixed or task-specific steering vectors to the model enable precise control over outputs of LMs.

Parameter modification focuses on directly adjusting the model’s parameters. Recent works focus on updating specific parts of the model responsible for particular knowledge. Techniques like Knowledge Neurons (Dai et al. 2022) and ROME (Meng et al. 2022) identify and edit specific neurons or layers with a designed causal tracing approach. These work make an assumption that the feed-forward network stores knowledge in a key-value format, allowing for directly targeted parameter adjustments to update or add memories. Another line of work (Wu et al. 2024) focuses on training interventions to directly edit model representation during inference for specific tasks.

![](images/3707f9665fdd10fdb85827c64caf6e9d61b59445bf05d91b7c8321a570aa5f84.jpg)  
Figure 2: Illustration of the process of causal tracing and model editing for ABSC. (a), (b), (c) illustrate three runs for tracing sentiment associations; (d) and (e) demonstrate our method, where colored blocks represent active parameters and the grey blocks represent frozen parameters. Notably, representation-based editing is applied only to the position of the aspect word.

# Method

In this section, we first introduce the intervention method for tracing sentiment associations, then analyze the causal tracing results. Finally, based on the insights of the analysis, we explore model editing as an efficient adaptation method using task-specific model edits.

# Interventions for Tracing Sentiment Association

We begin by identifying the specific layer with the strongest causal effect on sentiment polarity predictions of the given aspect terms by the method proposed in (Meng et al. 2022). This process is inspired by the causal tracing technique in (Vig et al. 2020; Pearl 2022; Geiger et al. 2021), which demonstrates how modifications to hidden states can impact the network’s output. By systematically adding noise and restoring representations, we can evaluate the sensitivity of the model’s predictions to changes in certain components.

We present each sample as a tuple $t = ( S , A , P )$ containing the sentence $S$ , the aspect $A$ , and the sentiment polarity $P$ according to $A$ . Then, we provide a natural language prompt $q$ describing $( S , A )$ and examine the model’s prediction of $q$ . To identify the layers with the strongest causal effect, we perform interventions on the output of each layer. Specifically, we introduce the following three runs:

Clean run. We input the sentence $S$ and aspect $A$ described by $q$ into the model, and observe the prediction

$P$ , as shown in Figure 2(a). All hidden representations )}i [1,T ],l [1,L] are recorded, where T is the number of tokens and $L$ is the number of layers.

Corrupted run. As shown in Figure 2(b), we add noise $\sigma$ to the hidden states at embedding layers denoted as {hi(0) + σ}i∈[1,T ]. It results in corrupted representations $\{ h _ { i } ^ { ( l ) , \mathrm { c o r r } } \} _ { i \in [ 1 , T ] , l \in [ 1 , L ] }$ . This noise follows the prior practice (Meng et al. 2022). The corrupted representations lead to a potentially incorrect sentiment polarity prediction $P ^ { \mathrm { c o r r } }$ .

Corrupted-with-restoration run. As shown in Figure 2(c), we selectively restore the clean hidden representation $h _ { i } ^ { ( l ) }$ ,clean at certain token and layer while keeping the rest of the representation corrupted. This run tests if the model can correctly predict the sentiment polarity $P$ despite the overall corruption, indicating the causal importance of the restored hidden states.

Let $\mathbb { P } [ q ] , \ \mathbb { P } _ { \mathrm { c o r r } } [ q ]$ , and Pcorr,h(l),clean [q] denote the probabilities of predicting $P$ under the clean, corrupted, and corrupted-with-restoration runs, respectively. The causal effect is quantified as follows:

The total effect (TE) is defined as the difference in prediction probabilities between the clean and corrupted runs:

$$
\mathrm { T E } = \mathbb { P } [ q ] - \mathbb { P } _ { \mathrm { c o r r } } [ q ]
$$

The indirect effect (IE) is defined as the difference be

![](images/69c29aba7fbff930fff5cee60b94eb8d059cbf6bc5be64a3c83bdd914bbb675f.jpg)  
Figure 3: Causal tracing results for ABSC at various words, layers, and positions. The ”First token,” ”Second token,” and so on denote the positions of the input sentence tokens.

tween the probability of correct prediction under the corrupted run and the corrupted-with-restoration run for a specific layer $l$ and position $i$ :

$$
\mathrm { I E } _ { i } ^ { ( l ) } = \mathbb { P } _ { \mathrm { c o r r } , h _ { i } ^ { ( l ) , \mathrm { c l e a n } } } [ q ] - \mathbb { P } _ { \mathrm { c o r r } } [ q ]
$$

The average total effect (ATE) is the mean total effect across multiple samples:

$$
{ \mathrm { A T E } } = { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } { \mathrm { T E } } _ { i }
$$

The average indirect effect (AIE) calculates the mean indirect effect for each hidden state variable $h _ { i } ^ { ( l ) }$ :

$$
{ \mathrm { A I E } } ^ { ( l ) } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } { \mathrm { I E } } _ { i } ^ { ( l ) }
$$

By analyzing these effects, we can identify the layers and specific hidden states within the layers that have the most significant causal impact on sentiment polarity predictions.

# Causal Tracing Results

We begin by adapting the $\mathrm { L L M ^ { 1 } }$ to solve the ABSC task. Subsequently, we compute the average indirect effect over 400 samples from the ABSC dataset.

Since the model inevitably makes errors, we use the ATE to discard outliers to ensure an accurate interpretation of the causal tracing results. Specifically, we retain only the results that met the expected prediction: $\mathbb { P } [ q ]$ for correct prediction and $\mathbb { P } _ { \mathrm { c o r r } } [ q ]$ for incorrect prediction.

As illustrated in Figure 3, our exploration focuses on the influence of tokens at different positions and layers within the model. Our findings reveal that a notable portion of the effect is mediated by highly causal individual states, particularly at the aspect tokens.

While it is expected that aspect tokens play a decisive role in predicting sentiment polarity, the causal tracing results explicitly demonstrate that the mid-layer representations of aspect tokens play a significant role in ABSC. The emergence of these causal states in the mid-layers sheds light on more efficient parameter-efficient fine-tuning methods for ABSC.

# Model Editing for LLM-based ABSC

Given the importance of mid-layer representations of aspect tokens for ABSC, we propose that employing model editing in only these representations can guide the model to solve ABSC with significantly fewer parameters. Our method leverages the strengths of both weight-based (Hu et al. 2022) and representation-based editing (Wu et al. 2024) to achieve this goal.

Recall that in each layer of the decoder model, the representation is updated through a specific process. The hidden state ${ \bf h } _ { l }$ at layer $l$ is derived by incorporating the previous layer’s representation $\mathbf { h } _ { l - 1 }$ , the output from the attention mechanism, and the output from a multi-layer perceptron (MLP). Formally, this can be expressed as:

$$
\mathbf { h } _ { t } = \mathbf { h } _ { t - 1 } + a ( \mathbf { h } _ { t - 1 } ) + \mathbf { M L P } ( \mathbf { h } _ { t - 1 } + a ( \mathbf { h } _ { t - 1 } ) ) ,
$$

where $a ( \cdot )$ denotes the attention mechanism and $\mathrm { \mathbf { M L P } ( \cdot ) }$ denotes the multi-layer perceptron.

Weight-based Editing Previous works (Tang et al. 2020; Wang et al. 2020; Zhang and Qian 2020) have demonstrated that the attention mechanism is crucial for capturing relationships between tokens, especially aspect tokens. Therefore, we only employ weight-based model editing on the attention output projection matrix, which is responsible for transforming these multi-head attention outputs back into the model’s hidden space, directly affecting how aspect token relationships are encoded. By applying weight-based editing specifically to this matrix, we can edit the model to handle aspect tokens with minimal disruption to other layers, ensuring parameter efficiency while maintaining strong performance.

Specifically, we update the weight matrix $\mathbf { W } _ { l }$ of the targeted layers with weight-based editing (Hu et al. 2022) using low-rank matrix $\mathbf { A } _ { l }$ and $\mathbf { B } _ { l }$ .

Let $\mathbf { W } _ { l } \in \mathbb { R } ^ { d \times k }$ represent the attention layer output projection matrix of the $l$ -th layer in the LLM. We introduce low-rank matrices $\mathbf { A } _ { l } \in \mathbb { R } ^ { \tilde { d } \times r }$ and $\mathbf { B } _ { l } \in \mathbb { R } ^ { r \times k }$ , such that the weight update can be approximated as:

$$
\Delta \mathbf { W } _ { l } \approx \mathbf { A } _ { l } \mathbf { B } _ { l }
$$

The adapted weight matrix $\mathbf { W } _ { l } ^ { \prime }$ is then:

$$
\mathbf { W } _ { l } ^ { \prime } = \mathbf { W } _ { l } + \Delta \mathbf { W } _ { l } = \mathbf { W } _ { l } + \mathbf { A } _ { l } \mathbf { B } _ { l }
$$

Representation-based Editing Previous works (Ravfogel et al. 2022; Avitan et al. 2024; Singh et al. 2024) demonstrate that editing the residual stream is crucial for controlling pretrained LM outputs without intensive finetuning. Therefore, we focus on editing the residual stream representation to activate important components within the parameters of the LLM for ABSC.

For the representation editing, we apply linear representation editing using projection matrix $\mathbf { R } _ { l }$ and linear interventions $\mathbf { W } _ { l } ^ { * }$ . Specifically, we define a low-rank projection matrix $\mathbf { R } _ { l } \ \in \ \mathbb { R } ^ { r \times d }$ with orthonormal rows. The modified hidden representation $\mathbf { h } _ { l } ^ { t }$ at layer $l$ for an aspect term at position $t$ is given by:

Table 1: Distribution of reviews across different domains.   

<html><body><table><tr><td>Dataset</td><td>Split</td><td>NumberofReviews</td></tr><tr><td>Device</td><td>Train</td><td>1,394</td></tr><tr><td></td><td>Test</td><td>691</td></tr><tr><td>Laptop</td><td>Train</td><td>2,297</td></tr><tr><td></td><td>Test</td><td>631</td></tr><tr><td>Restaurant</td><td>Train</td><td>4,284</td></tr><tr><td></td><td>Test</td><td>2,252</td></tr><tr><td>Service</td><td>Train</td><td>1,840</td></tr><tr><td></td><td>Test</td><td>886</td></tr></table></body></html>

$$
\alpha ( \mathbf { h } _ { l } ^ { t } ) = \mathbf { h } _ { l } ^ { t } + \mathbf { R } _ { l } ^ { \top } ( \mathbf { W } _ { l } ^ { * } \mathbf { h } _ { l } ^ { t } + \mathbf { b } _ { l } - \mathbf { R } _ { l } \mathbf { h } _ { l } ^ { t } ) ,
$$

where $\mathbf { W } _ { l } ^ { * } \in \mathbb { R } ^ { r \times d }$ is a linear projection matrix and ${ \bf b } _ { l }$ is a bias vector.

Thus, as shown in Figure 2(e), the overall optimization objective during model editing for ABSC with our method is to minimize the loss $\mathcal { L }$ with respect to the model editing parameters of $\theta = \{ { \bf R } _ { l } , { \bf W } _ { l } ^ { * } , { \bf A } _ { l } , { \bf B } _ { l } , { \bf b } _ { l } \} _ { l \in L ^ { * } }$ ,

$$
\mathcal { L } ( \theta ) = \mathbb { E } _ { ( x , y ) \sim \mathcal { D } } \left[ - \sum _ { t = 1 } ^ { | y | } \log \left( f _ { \Phi _ { 0 } + \Delta \Phi ( \theta ) } ( y _ { t } \mid x , y _ { < t } ) \right) \right] ,
$$

where $L ^ { * }$ denotes the specific training layers.

Notably, the task specific model editing parameter $| \theta |$ is much smaller than the LLM parameters $| \Phi _ { 0 } |$ . Therefore, our method allows the model to be efficiently edited at critical layers and positions, thereby improving performance on ABSC with significantly fewer parameters.

# Experiment

In this section, we introduce our experimental setup and implementation details, present our method’s performance on in-domain and out-of-domain ABSC tasks compared to competitive baselines, and empirically analyze the effectiveness of our method.

# Setup

The labeled dataset used in our experiments includes reviews from four different domains: Restaurant (R), Laptop (L), Device (D), and Service (S). Restaurant (R) is a combination of the restaurant reviews from SemEval 2014/2015/2016 (Pontiki et al. 2014, 2015, 2016). Laptop (L) is sourced from SemEval 2014 (Pontiki et al. 2014). Device (D) consists of all the digital device reviews collected by Toprak, Jakob, and Gurevych (2010). Service (S) contains reviews from web services introduced by Hu and Liu (2004). The distribution of reviews in these domains is detailed in Table 1.

We employ Llama-2-7b (Touvron et al. 2023) as our primary base large language model. AdamW (Loshchilov and

Hutter 2018) is used as the optimizer, with a learning rate of $3 \times 1 0 ^ { - 4 }$ for the low-rank weight projection part and $1 \times 1 0 ^ { - 5 }$ for the representation editing part. For the comparison methods, we adopt standard experimental settings and commonly used parameters. Specifically, LoRA and Dora utilize a learning rate of $1 \times \mathrm { { 1 0 ^ { - 4 } } }$ with rank of 32. Additionally, we include LoReft with a learning rate of $2 \times 1 0 ^ { - 5 }$ with rank of 8. All comparison experiments are conducted on a single NVIDIA 3090 GPU and we take accuracy as the evaluation metric. The experimental results are obtained by averaging three runs with random initialization. The PEFT methods are trained for one epoch, while the full parameter methods are trained for three epochs.

The causal tracing results reveal that the representations within mid-layers, specifically layers 10-15, show a decisive influence on ABSC. Therefore, in the ’specific-layer finetuning’ section of our experimental results, all methods are restricted to these specific layers. Instead, in the section of parameter-efficient finetuning, these methods are trained on all layers.

# Main Results

To assess the effectiveness of the proposed method, we mainly conduct two experiments: in-domain and outof-domain evaluations. We compare our method with zero-shot methods (i.e., GPT-4o-mini, Llama2-7b), fullparameter finetuning methods (i.e., BERT, Deberta, FlanT5), and parameter-efficient finetuning methods (i.e., PrefixFT, Adapter, LoRA, Dora, Loreft).

In-domain Analysis In-domain ABSC tasks refer to the task that the training data and testing data are in the same domain. This allows us to assess how well the methods perform on data similar to what they were trained on.

As shown in Table 2, our methods achieve the highest average accuracy, significantly outperforming other methods with fewer trainable parameters. Llama2-7b with zeroshot manner exhibits significantly lower performance compared to other method, indicating its limited effectiveness in the absence of domain-specific fine-tuning. The fullparameter finetuning models show improved performance. This suggests that full-parameter finetuning can effectively enhance model performance by leveraging domain-specific data. Among the PEFT methods, PrefixFT, despite its small tuning parameter size, shows only marginal improvement over zero-shot Llama2-7b. LoRA and Dora exhibit considerably better performance. However, these methods still involve larger tuning parameter sizes compared to our proposed methods. LoReft, while efficient in terms of parameter size, does not perform as well as our methods. These results indicate our method not only achieves high accuracy but also exhibits remarkable parameter efficiency.

Out-of-domain Analysis Due to the high risk of catastrophic forgetting in LLM-based in-domain training, especially in PEFT methods, we design out-of-domain analysis to evaluate the ability of methods to generalize on out-ofdistribution data.

As shown in Table 3, our approach significantly outperforms other strong baselines while maintaining a lower number of trainable parameters. Although we are concerned about the potential risk of data leakage with GPT4omini (Kocon´ et al. 2023), our method still surpasses GPT4o-mini in some domains and in average accuracy. This indicates the effectiveness of our method in handling data variability and ensuring robust generalization across different domains. LoRA and LoReft, as state-of-the-art PEFT methods, show strong performance in specific-layers finetuning, but these methods are still vulnerable in some out-ofdistribution samples. Our methods, based on the causal tracing results of sentiment association, not only achieve high accuracy but also maintain computational efficiency. The results clearly indicate that our method offers a balanced combination of accuracy and efficiency.

<html><body><table><tr><td rowspan="2">Methods</td><td rowspan="2">Base Model</td><td rowspan="2">Params (%)</td><td colspan="5">Accuracy (↑)</td></tr><tr><td>Device</td><td>Laptop</td><td>Restaurant</td><td>Service</td><td>Avg.</td></tr><tr><td>Zero-Shot</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT4o-mini (OpenAI 2024)</td><td></td><td></td><td>87.6</td><td>80.0</td><td>85.2</td><td>86.7</td><td>84.9</td></tr><tr><td>Llama2-7b (Touvron et al.2023)</td><td></td><td></td><td>68.7</td><td>53.2</td><td>66.5</td><td>56.0</td><td>61.1</td></tr><tr><td>Full-parameter-fintuning</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BERT (Devlin et al. 2019)</td><td>BERT-base</td><td>1.632%</td><td>90.7</td><td>71.2</td><td>84.8</td><td>82.4</td><td>82.3</td></tr><tr><td>Deberta (He et al. 2021)</td><td>DebertaV3-base</td><td>2.057%</td><td>95.4</td><td>74.2</td><td>86.3</td><td>86.0</td><td>85.5</td></tr><tr><td>Flan-T5 (Chung et al. 2022)</td><td>Flan-T5-base</td><td>3.265%</td><td>94.5</td><td>75.8</td><td>86.5</td><td>87.4</td><td>86.1</td></tr><tr><td> Parameter-efficient finetuning</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Adapter (Houlsby et al. 2019)</td><td>Llama2-7b</td><td>1.953%</td><td>73.2</td><td>68.6</td><td>72.1</td><td>75.2</td><td>72.3</td></tr><tr><td>LoRA (Hu et al. 2022)</td><td>Llama2-7b</td><td>0.826%</td><td>95.2</td><td>76.9</td><td>85.7</td><td>88.3</td><td>86.5</td></tr><tr><td>Dora (Liu et al. 2024)</td><td>Llama2-7b</td><td>0.838%</td><td>95.1</td><td>77.2</td><td>85.9</td><td>88.1</td><td>86.6</td></tr><tr><td>PrefixFT (Li and Liang 2021)</td><td>Llama2-7b</td><td>0.039%</td><td>68.9</td><td>53.2</td><td>66.5</td><td>37.6</td><td>56.6</td></tr><tr><td>LoReft (Wu et al. 2024)</td><td>Llama2-7b</td><td>0.031%</td><td>94.2</td><td>73.7</td><td>87.3</td><td>88.5</td><td>85.9</td></tr><tr><td>Specific-layers finetuning</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LoRA (Hu et al. 2022)</td><td>Llama2-7b</td><td>0.129%</td><td>93.6</td><td>73.2</td><td>85.5</td><td>86.8</td><td>84.8</td></tr><tr><td>LoReft (Wu et al. 2024)</td><td>Llama2-7b</td><td>0.006%</td><td>92.9</td><td>72.3</td><td>86.9</td><td>87.6</td><td>84.9</td></tr><tr><td>Ours</td><td>Llama2-7b</td><td>0.006%</td><td>96.2</td><td>76.7</td><td>88.1</td><td>89.5</td><td>87.6</td></tr></table></body></html>

Table 2: Overall accuracy $( \% )$ over four in-domain aspect-based sentiment classification datasets. The ”Params” column indicates the proportion of trainable parameters relative to the number of parameters in Llama2-7b. Specific-layers finetuning refers to updating only the middle layers, specifically layers 10 to 15.

# Impact on Specific Layers

As the causal tracing results mentioned before, the midlayers show great influence on predicting the sentiment polarity of the given aspect. To further explore this impact, we conduct a quantitative experiment to analyze its impact on the performance of different methods in average accuracy over 16 domain pairs, including in-domain and out-ofdomain datasets.

As shown in Figure 4, different methods exhibit consistent results across various layers. This consistency indicates the significant influence of different layers in the LLM on addressing the ABSC task. Specifically, within the 10-15 layer range, both methods achieve results comparable to their best performance when training parameters across all layers (0-31). This finding aligns with our previously mentioned causal tracing sentiment association results. Moreover, our method demonstrates a more stable and superior performance compared to LoReft. This stability is particularly notable across all evaluated layer ranges, further indicating the robustness and effectiveness of our method.

![](images/59079525eb05bf1071fb11d53bfa98608b2ccb3579e4e3877ac0f23f1eb99f20.jpg)  
Figure 4: Impact on trainable layers of LoReft and our method.

# Influence of Specific Positions

For our method, we must decide which layers and input positions to apply the intervention on. The causal tracing results reveal that aspect terms in mid-layers have a significant influence on the sentiment prediction. In this section, we investigate three potential word positions within the specific layers (10-15) to evaluate our methods. Intuitively, the last word and aspect word may have a strong impact, but we also take a random mid-word from the input to evaluate the influence of the intervention position. In this experiment, we take $10 \%$ of data from the training dataset as the development dataset for in-domain ABSC.

<html><body><table><tr><td rowspan="2">Methods</td><td colspan="10">Accuracy (↑)</td></tr><tr><td>(D,R)</td><td>(D, S)</td><td>(L,R)</td><td>(L,S)</td><td>(R,D)</td><td>(R,L)</td><td>(R, S)</td><td>(S,D)</td><td>(S,L) (S,R)</td><td>Avg.</td></tr><tr><td>Zero-Shot</td><td></td><td></td><td>85.2</td><td>86.7</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-4o-mini (OpenAI 2024)</td><td>85.2</td><td>86.7</td><td></td><td>87.6</td><td>80.0</td><td>86.7</td><td>87.6</td><td>80.0</td><td>85.2</td><td>85.1</td></tr><tr><td>Llama2-7b (Touvron et al.2023)</td><td>66.5</td><td>56.0 66.5</td><td>56.0</td><td>68.7</td><td>53.2</td><td>56.0</td><td>68.7</td><td>53.2</td><td>66.5</td><td>61.1</td></tr><tr><td>Full-parameter-fintuning</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BERT (Devlin et al. 2019)</td><td>80.4</td><td>85.2</td><td>81.1</td><td>81.9 88.3</td><td>77.8</td><td>84.5</td><td>90.7</td><td>69.7</td><td>79.8</td><td>81.9</td></tr><tr><td>Deberta (He et al. 2021)</td><td>81.1</td><td>85.4</td><td>80.1</td><td>88.3</td><td>78.4</td><td>86.2</td><td>93.9</td><td>68.5</td><td>81.0</td><td>82.6</td></tr><tr><td>Flan-T5 (Chung et al. 2022)</td><td>81.8</td><td>87.6</td><td>82.7 79.5</td><td>85.3 93.9</td><td>78.3</td><td>87.0</td><td>93.5</td><td>70.4</td><td>81.2</td><td>83.9</td></tr><tr><td>Parameter-efficient finetuning</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LoRA (Hu et al. 2022)</td><td>82.9</td><td>87.2</td><td>85.0</td><td>81.6</td><td>84.9 79.2</td><td>75.2</td><td>94.4</td><td>70.7</td><td>83.8</td><td>82.5</td></tr><tr><td>Dora (Liu et al. 2024)</td><td>82.4</td><td>87.4</td><td>84.7</td><td>81.2</td><td>84.5 80.2</td><td>75.1</td><td>94.6</td><td>69.9</td><td>84.1</td><td>82.4</td></tr><tr><td>LoReft (Wu et al. 2024)</td><td>84.0</td><td>87.6</td><td>83.2</td><td>85.2</td><td>94.5 72.9</td><td>88.5</td><td>94.1</td><td>71.3</td><td>84.4</td><td>84.6</td></tr><tr><td>Specific-layers finetuning</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LoRA (Hu et al. 2022)</td><td>81.9</td><td>86.2</td><td>84.2</td><td>80.1</td><td>83.2</td><td>77.5 72.3</td><td>92.8</td><td>67.2</td><td>82.1</td><td>80.8</td></tr><tr><td>LoReft (Wu et al. 2024)</td><td>83.8</td><td>83.4</td><td>83.3</td><td>83.3</td><td>95.2</td><td>72.4 86.6</td><td>92.2</td><td>65.9</td><td>83.3</td><td>82.9</td></tr><tr><td>Ours</td><td>85.8</td><td>88.7</td><td>85.0</td><td>85.2</td><td>92.9</td><td>75.6 88.5</td><td>95.8</td><td>71.6</td><td>84.9</td><td>85.4</td></tr></table></body></html>

Table 3: Overall accuracy $( \% )$ over 12 out-of-domain aspect-based sentiment classification dataset pairs. The column ”(D,R)” indicates the model is trained on Device domain and tested on Restaurant domain, except these zero-shot methods. Specificlayers finetuning refers to updating only the middle layers, specifically layers 10 to 15.

![](images/b2eb638bfd90a1137abdd86a546d61f832df27b2452e4e88af745b399ad6fca3.jpg)  
Figure 5: Influence on specific positions: ’aspect’ refers to editing on aspect terms, ’last’ refers to editing on the last words, and ’mid’ refers to editing on a random mid-position word.

As shown in Figure 5, editing on aspect terms yields the highest performance. It highlights the pivotal role aspect terms play in ABSC. Additionally, the last word shows relatively good performance due to the use of a decoder-only model. However, the performance still falls short compared to editing on aspect terms. This indicates that for specific tasks, targeting the key terms relevant to the task, such as aspect terms, can lead to better results. Notably, the performance does not degrade significantly even when a random mid-position word is chosen for intervention. It suggests that while targeted editing on aspect terms is optimal, our method, which containing weight,and representation based editing methods, remains robust and does not suffer from significant performance drops due to random selection.

# Conclusion

We explore model editing as an efficient and interpretable fine-tuning method for LLM-based aspect-based sentiment classification. By combining causal tracing with targeted model edits, we demonstrate that model editing can serve as an efficient fine-tuning method with minimal parameter updates. In addition, we empirically validate that specific midlayer representations of aspect words play a crucial role in aspect-based sentiment classification. Practical experiments confirm the effectiveness of our framework, demonstrating that it outperforms state-of-the-art parameter-efficient finetuning methods with fewer trainable parameters. These results indicate the potential of model editing as a more interpretable and efficient strategy for adapting LLMs to specific downstream tasks, highlighting its broader applicability.