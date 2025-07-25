# Enhancing Non-English Capabilities of English-Centric Large Language Models Through Deep Supervision Fine-Tuning

Wenshuai $\mathbf { H u o } ^ { 1 , 2 }$ , Xiaocheng Feng1,2 \* , Yichong Huang1, Chengpeng $\mathbf { F u } ^ { 1 , 2 }$ , Baohang $\mathbf { L i } ^ { 1 }$ , Yangfan $\mathbf { Y } \mathbf { e } ^ { 1 }$ , Zhirui Zhang3, Dandan $\mathbf { T } \mathbf { u } ^ { 3 }$ , Duyu Tang3, Yunfei $\mathbf { L u } ^ { 3 }$ , Hui Wang2, Bing $\mathbf { Q i n } ^ { 1 , 2 }$

1Harbin Institution of Technology 2Pengcheng Laboratory 3Huawei Technologies Co., Ltd wshuo, xcfeng, ychuang, cpfu, baohangli, yfye, qinb @ir.hit.edu.cn, wangh06@pcl.ac.cn tudandan, tangduyu, luyunfei6 @huawei.com, zrustc11 $@$ gmail.com

# Abstract

Large language models (LLMs) have demonstrated significant progress in multilingual language understanding and generation. However, due to the imbalance in training data, their capabilities in non-English languages are limited. Recent studies revealed the English-pivot multilingual mechanism of LLMs, where LLMs implicitly convert non-English queries into English ones at the bottom layers and adopt English for thinking at the middle layers. However, due to the absence of explicit supervision for cross-lingual alignment in the intermediate layers of LLMs, the internal representations during these stages may become inaccurate. In this work, we introduce a deep supervision fine-tuning method (DFT) that incorporates additional supervision over the internal layers of the model to guide its workflow. Specifically, we introduce two training objectives on different layers of LLMs: one at the bottom layers to constrain the conversion of the target language into English, and another at the middle layers to constrain reasoning in English. To effectively achieve the guiding purpose, we designed two types of supervision signals: logits and feature, which represent a stricter constraint and a relatively more relaxed guidance. Our method guides the model to not only consider the final generated result when processing non-English inputs but also ensure the accuracy of internal representations. We conducted extensive experiments on typical English-centric LLMs, LLaMA-2 and Gemma-2. The results on 8 multilingual datasets show that our method significantly outperforms traditional fine-tuning methods.

# Introduction

The development of large language models has achieved revolutionary breakthroughs in the field of natural language processing, demonstrating exceptional performance (OpenAI 2023; Touvron et al. 2023; Team et al. 2023). However, performance disparities still exist across different languages due to the imbalance in the training data (Nguyen et al. 2023; Huang et al. 2023; Zhu et al. 2023a). For instance, the well-known LLaMA series models are trained on over $90 \%$ English data. This results in a significant gap between their multilingual capabilities and their performance in English.

![](images/9b0a2e257a4c501a1c764a4262daad73c2dfce0bdb536d932ee0602dc159c52e.jpg)  
Figure 1: The illustration of Depth Supervision Fine-Tuning (DFT) and Baseline Methods. The left side represents an English-dominated large language model, which can be divided into three parts from shallow to deep layers: Language Conversion, English Thinking, and Language Reversion. The right side shows an sample of Chinese instruction tuning. Blue arrows represent the DFT method, while gray arrows represent the Baseline method. Traditional finetuning methods focus only on the model predicting the corresponding target output based on the input instruction. In contrast, our method adds supervision to the process, explicitly guiding the model’s workflow when processing nonEnglish inputs.

To enhance the non-English processing capabilities of these models, researchers have made numerous attempts. A common and effective approach is to fine-tune pre-trained models on instruction datasets in the target languages (Zhu et al. 2023b; Li et al. 2023; Zhang et al. 2024; Li et al. 2024b; Zhao et al. 2024a). As research advances, some studies have explored the internal mechanisms behind how large models handle multilingualism, revealing an English-centric workflow when processing non-English inputs. Specifically, Zhao et al. (2024b) analyzed activation patterns in specific languages and inferred that the bottom layers of the model convert input from various languages into English, while the top layers perform the reverse conversion. Wendler et al. (2024) found that by decoding early in the model’s intermediate layers, rather than the final layer, LLMs tend to use English as an internal pivot language when processing multilingual inputs. These studies suggest that LLMs dominated by English can be divided into three stages when processing non-English inputs: converting non-English to English (or an English representation space), thinking in English, and converting English back to the target language. However, due to the absence of explicit supervision for cross-lingual alignment in the intermediate layers of LLMs, the internal representations during these stages may become inaccurate. To the best of our knowledge, no studies have yet proposed targeted optimization schemes for this phenomenon.

To tackle the aforementioned problem, we propose the deep supervision fine-tuning method (DFT) that aims to explicitly guide the LLMs’ performance in the three stages when processing non-English inputs. Specifically, DFT incorporates additional supervision over the internal layers of the model. As shown in Figure 1, unlike traditional finetuning methods, DFT constrains not only the final output of the model but also the intermediate process. In the bottom layers of the model, DFT guides the conversion from non-English to English. In the middle layers, DFT guides the model to obtain answers in the English space. To effectively constrain the intermediate process, we propose two supervision schemes based on logits and features. Additionally, to accurately identify the critical layers at different stages, we propose an entropy-based selection strategy.

We conducted extensive experiments on 8 commonly used multilingual benchmarks and the results demonstrate the effectiveness of our method. Specifically, for multilingual QA tasks where both input and output are in the target language, our method achieved significant improvements. Our contributions can be summarized as follows.

# Background

Instruction Fine-Tuning. The technology of instruction tuning enables pre-trained LLMs to comprehend instructiinognsonanadnhanndolteatdeodwinstreuactmiotnasdkastaesfefte $D = \mathsf { \bar { \{ } }  ( x _ { i } , y _ { i } ) \rbrace _ { i = 1 } ^ { N }$ where $x$ is the input question and $y$ is the expected output answer. The training objective is to minimize the following negative log-likelihood:

$$
\mathcal { L } ( \theta ) = \sum _ { i = 1 } ^ { N } - \log P ( y _ { i } | x _ { i } ; \theta ) ,
$$

where $\theta$ denotes the learnable parameters of the model. However, most instruct datasets are in English, which limits the potential of large models to address tasks in non-English languages.

Multilingual Instruction Fine-Tuning. To enhance the multilingual capabilities of large language models, a common approach is to translate English instruction data into the target language (tgt), creating a dataset $D ^ { t g t } \ =$ $\{ ( x _ { i } ^ { t g t } , y _ { i } ^ { t g t } \bar { ) } \} _ { i = 1 } ^ { N }$ . The model is then fine-tuned using the translated instruction data. Similar to equation 1, the loss function $\mathcal { L } _ { T F T }$ is:

$$
\mathcal { L } _ { T F T } ( \theta ) = \sum _ { i = 1 } ^ { N } - \log P ( y _ { i } ^ { t g t } | x _ { i } ^ { t g t } ; \theta ) .
$$

In this work, we also fine-tune the model using multilingual instruction datasets constructed through translation. However, unlike traditional fine-tuning methods, we focus not only on the model’s final predictions but also impose constraints to ensure the model achieves accurate intermediate results.

Deep Supervision Networks. Deep Supervision Networks (DSNs) represent a special training strategy that optimizes the training process by incorporating supervision at multiple intermediate layers of the model. Unlike traditional deep learning architectures that compute the loss only at the final layer $L$ , DSNs also calculate loss at several hidden layers $k$ . The total loss function $\mathcal { L }$ can be expressed as:

$$
\begin{array} { r } { \mathcal { L } ( \boldsymbol { \theta } ) = \mathcal { L } _ { L } ( \boldsymbol { \theta } ) + \alpha \mathcal { L } _ { k } ( \boldsymbol { \theta } _ { 1 : k } ) , } \end{array}
$$

$\theta _ { 1 : k }$ denotes only the parameters of layer $k$ to layer 1 are learned and updated by minimising the intermediate loss $( \mathcal { L } _ { k } )$ ; $\alpha$ is a hyper-parameter to control the balance between the intermediate supervision loss $( \mathcal { L } _ { k } )$ and the final output layer loss $( \mathcal { L } _ { L } )$ .

In this work, we take inspiration from the training strategy of DSNs by introducing additional supervision over the intermediate layers of the model. Unlike traditional DSNs, where the intermediate supervision is consistent with the final output target (Li et al. 2022), our additional supervision is differ from the final output. The goal is to guide the workflow of LLMs when processing non-English inputs.

# Method

In this section, we describe the proposed deep supervisionbased fine-tuning approach. Figure 2 illustrates the overall process of our method. The process of handling nonEnglish inputs in an LLM can be roughly divided into three stages from the bottom layers to the top layers: (1) Language Conversion: The model interprets the non-English query and converts the multilingual input into English (or an English representation space); (2) English Thinking: The model employs English for thinking and solving the task; (3) Language Reversion: The model converts the reasoning results back into the target language, consistent with the input. We propose DFT to guide the model’s internal information transformation. Specifically, we introduce additional supervision at the top and middle layers of the model to enhance the model’s ability to convert non-English inputs into English and to improve its logical reasoning in English. We propose two types of supervision: logits-based and featurebased.

![](images/127a2f69112c50acceb92478efe9449e47269cd62d87459329ef0c45a8400370.jpg)  
Figure 2: The illustration of the proposed methods DFT-logits (b) and DFT-feature (c). The heatmap (a) represents the entropy values of each layer in English-dominated Large Language Model when processing non-English inputs. The process of handling non-English inputs in an LLM can be roughly divided into three stages from the bottom layers to the top layers: Language Conversion, English Thinking and Language Reversion.

# Language Conversion Constraints

When processing non-English inputs, the initial layers of large language models are primarily tasked with converting the input query into an internal English representation that the model can effectively process. Due to the significant differences in grammar, vocabulary, and linguistic structures across languages, this conversion process is inherently challenging. Failure to accurately perform this conversion can lead to semantic biases or errors in subsequent reasoning and generation stages. Thus, it is critical that the model excels in language conversion within its early layers.

To enhance the model’s language conversion capabilities, we propose two constraint methods: logits supervision and feature supervision.

Logits-Based: For transformer-based LLMs, each layer outputs a hidden representation of shape $( b a t c h . s i z e \times$ sequence length $\times$ hidden size). The typical training strategy feeds the hidden representation from the final layer into a linear matrix $W ^ { o { \bar { u } } t }$ of shape $( h i d d e n . s i z e \times$ $v o c a b . s i z e )$ ), projecting the final hidden representation into the vocabulary space and predicting the probability distribution over the vocabulary using softmax. This distribution is then compared with the ground truth to calculate the loss. However, since the hidden states have the same shape across all layers, we can apply the $W ^ { o u t }$ and softmax operations at any layer to make predictions. To enforce language conversion, we apply $W ^ { o u t }$ at an internal layer $i$ in the model (as shown in Figure 2b) and compute the probability distribution. Then calculate the loss with a parallel English query, thereby forcing the model to convert the target language query into its English version in the early layers. Where the logits-based language conversion loss $\mathcal { L } _ { L C }$ is computed as:

$$
\mathcal { L } _ { L C } = - \log P ( \boldsymbol { x } ^ { e n } | \boldsymbol { x } ^ { t g t } ; \boldsymbol { \theta } _ { 1 : i } , W ^ { o u t } ) ,
$$

$\theta _ { 1 : i }$ represents the parameters of the first $\mathbf { \chi } _ { i }$ layers of the model. We use the LLM’s head as $W ^ { o u t }$ and restrict $W ^ { o u t }$ to only participate in the forward pass during the language conversion stage, without being updated.

Feature-Based: Another approach to enhancing the language conversion process in the first stage is through feature alignment. During instruction tuning in the target language, semantically equivalent English querys are also fed into the model. The hidden representations of these inputs are then extracted, and a similarity loss is calculated to ensure alignment between the English and non-English representations. The feature-based language conversion loss function $\mathcal { L } _ { L C }$ can be defined as:

$$
\mathcal { L } _ { L C } = 1 - \left( \frac { \boldsymbol { h } _ { i } ^ { e n } \cdot \boldsymbol { h } _ { i } ^ { t g t } } { \| \boldsymbol { h } _ { i } ^ { e n } \| \| \boldsymbol { h } _ { i } ^ { t g t } \| } \right) ,
$$

where $i$ represents the critical layer in the first stage, $h ^ { e n }$ and $h ^ { t g t }$ represent the hidden states of the same semantic input in English and the target language, respectively. Compared to the logits-based supervision strategy, feature supervision is relatively more relaxed. It does not require the target language input to be strictly converted into English but instead aligns the representations across languages.

By applying constraints at this stage, non-English inputs can be more accurately transformed into English, thereby laying a strong foundation for subsequent reasoning and output generation.

# English Thinking Constraints

After language conversion, the model needs to further process the input information and reason how to provide an appropriate response. Therefore, we add supervisory signals to constrain the model’s reasoning ability in English. Similar to the previous section, we implement with both logits constraint and feature constraint methods.

Logits-Based: For the logits constraint, we output probabilities from the hidden vectors at the end of the second stage and compute the loss, aiming for the model to provide corresponding English answers based on the input target question. The logits-based english thinking loss function $\mathcal { L } _ { E T }$ is formulated as follows:

$$
\mathcal { L } _ { E T } = - \log P ( y ^ { e n } | x ^ { t g t } ; \theta _ { 1 : j } , W ^ { o u t } ) ,
$$

where $j$ represents the critical layer in the second stage responsible for reasoning in English. This constraint forces the model to focus on producing the correct English reasoning path, ultimately leading to accurate final outputs.

Feature-Based: For the feature constraint, as shown in the Figure 2, this is implemented by feeding semantically equivalent English and non-English questions into the model. For the English input, we extract the features from the top layers of the model, while for the non-English input, we extract the features from the critical layer of the English Thinking stage, where the reasoning in English is assumed to occur. The alignment of these features is crucial for ensuring that the model’s internal representations remain consistent across language.

$$
\mathcal { L } _ { E T } = 1 - \left( \frac { h _ { L } ^ { e n } \cdot h _ { j } ^ { t g t } } { \Vert h _ { L } ^ { e n } \Vert \Vert h _ { j } ^ { t g t } \Vert } \right) ,
$$

where $L$ represents the final layer, $j$ represents the critical layer in the second stage. This constraint helps the model to align its reasoning process in English with the semantic content derived from the target language input, ensuring that the reasoning remains accurate and consistent in the model.

# Training Objective

The total loss function is defined as:

$$
\mathcal { L } = \mathcal { L } _ { T F T } + \mathcal { L } _ { L C } + \mathcal { L } _ { E T } ,
$$

where $\mathcal { L } _ { L C }$ is either logits-based or feature-based, and $\mathcal { L } _ { E T }$ is also either logits-based or feature-based.

# Experiment

# Setup

We use LLaMA-2-7B (Touvron et al. 2023) and Gemma2-2B (Team 2024) as the base models. The training data consists of Stanford Alpaca instruction data (Taori et al. 2023) and its translations in the target languages, which include Chinese (zh), Vietnamese (vi), and Arabic (ar). For the translated data, we directly used publicly available datasets from (Zhu et al. 2023b). Our code implementation is based on stanford alpaca 1. All experiments were conducted on 8 $\times \ \mathrm { A 1 0 0 }$ GPUs with a batch size of 128. The models were trained for 3 epochs with a learning rate of 2e-5. To accelerate training, we utilized the FSDP training strategy (Zhao et al. 2023).

# Comparison of Methods

• SFT (Ouyang et al. 2022), which is instruction-tuned with English instruction datasets.   
• TFT (Zhu et al. 2023b), which is instruction-tuned using the original English instruction datasets translated into the target languages.   
• SDRRL (Zhang et al. 2024), which is a method based on Self-Distillation. Besides using English instructiontuning data and its multilingual code-switching extensions, it also incorporates partially translated data and completion data for fine-tuning.   
• DFT-logits, our method that applies logits-based supervision to guide the model’s intermediate layers.   
• DFT-feature, our method that uses feature alignment to maintain consistent internal representations between English and tgt language.

# Evaluation Dataset

• XQUAD (Cross-lingual Question Answering Dataset): XQUAD (Artetxe, Ruder, and Yogatama 2019) is a high-quality cross-lingual question answering dataset containing 240 paragraphs and 1,190 questionanswer pairs, which have been manually translated into 10 languages.   
• MLQA (Multilingual Question Answering): MLQA (Lewis et al. 2019) is a multilingual question answering dataset covering 7 languages. Each question in the dataset is accompanied by a paragraph and an answer in the corresponding language.   
• MKQA (Multilingual Knowledge Questions and Answers): The MKQA (Longpre, Lu, and Daiber 2021) dataset contains 2,600 common sense question-answer pairs across 26 languages.   
• TruthfulQA: TruthfulQA (Lin, Hilton, and Evans 2021) includes questions from various domains, specifically designed to test the truthfulness and accuracy of models when answering complex questions.   
• XNLI (Cross-lingual Natural Language Inference): XNLI (Conneau et al. 2018) is a widely used language

Table 1: Results of baselines and our method on multilingual question and answer benchmark. Bold indicates the best result of all methods. Our method outperforms the baselines in almost all languages.   

<html><body><table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td rowspan="2">vi</td><td rowspan="2">xquad ar</td><td rowspan="2">zh vi</td><td colspan="3">mlqa</td><td colspan="3">truthfulQA</td><td colspan="3">mkqa</td></tr><tr><td>ar</td><td>zh</td><td>vi</td><td>ar</td><td>zh</td><td></td><td>vi</td><td>ar</td><td>zh</td></tr><tr><td colspan="10">SFT 23.11</td><td>29.11</td><td>30.49</td><td>33.16</td><td></td><td>32.8433.51</td></tr><tr><td></td><td>TFT SDRRL DFT-logits</td><td>27.36 28.79 29.65</td><td>21.23 25.75 27.57</td><td>23.41 22.01 24.77</td><td>28.80 29.59 31.54</td><td>21.62 27.26 28.26</td><td>23.86 24.67 25.68</td><td>28.54 28.03 28.64</td><td>27.04 28.11 28.81</td><td>26.40 28.30 28.93</td><td>38.59 39.20 39.64</td><td>35.02 35.81 35.84</td><td>40.70 40.63 40.87</td></tr><tr><td>Baselines</td><td>DFT-feature SFT TFT 17.58</td><td>28.27 19.02</td><td>26.35 18.82 24.73</td><td>24.34 9.74 15.95</td><td>32.19 20.21 20.26</td><td>29.51 Performance on Gemma-2-2b 20.45 25.31</td><td>25.29 11.79 18.26</td><td>30.45 26.44 26.26</td><td>28.98</td><td>27.79</td><td>36.04</td><td>33.69</td><td>32.46</td></tr><tr><td>Ours</td><td>SDRRL DFT-loaitse</td><td>17.55 21.23</td><td>18.16 25.79</td><td>12.03 17.37</td><td>19.89 24.36</td><td>23.91 27.51</td><td>14.24 20.13</td><td>26.62 27.34</td><td>26.52 26.78</td><td>26.39 27.16</td><td>37.69 38.54</td><td>34.51 34.81</td><td>40.75 40.84</td></tr></table></body></html>

<html><body><table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="3">xnli</td><td colspan="2">xcopa</td><td colspan="2">xstory_cloze</td><td rowspan="2"></td><td colspan="2">mmlu ar</td></tr><tr><td>vi</td><td>ar</td><td>zh</td><td>vi</td><td>zh</td><td>ar</td><td>zh</td><td>vi</td><td></td></tr><tr><td rowspan="4">Baselines</td><td colspan="10">Performance on LLaMA-2-7b</td></tr><tr><td>SFT</td><td>36.59</td><td>35.38</td><td>36.39</td><td>63.00</td><td>65.00</td><td>49.70</td><td>59.56</td><td>29.37</td><td>27.6</td><td>30.49</td></tr><tr><td>TFT</td><td>44.14</td><td>33.73</td><td>36.87</td><td>66.00</td><td>65.00</td><td>57.51</td><td>64.06</td><td>33.23</td><td>28.05</td><td>32.34</td></tr><tr><td>SDRRL</td><td>43.41</td><td>33.98</td><td>37.43</td><td>65.40</td><td>65.20</td><td>58.17</td><td>64.53</td><td>31.91</td><td>27.47</td><td>32.74</td></tr><tr><td rowspan="2">Ours</td><td>DFT-logits DFT-feature</td><td>43.37 43.50</td><td>34.06 34.38</td><td>36.94 37.64</td><td>66.20 66.60</td><td>66.40 68.60</td><td>58.78 58.78</td><td>64.93 64.40</td><td>32.60 32.19</td><td>29.06 28.86</td><td>33.19 32.82</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">Baselines</td><td colspan="10"></td></tr><tr><td>SFT</td><td>41.33</td><td>35.62</td><td>Performance on Gemma-2-2b</td><td></td><td>70.40</td><td>61.02</td><td>68.23</td><td></td><td>31.16</td><td>36.22</td></tr><tr><td>TFT</td><td>43.05</td><td>33.94</td><td>40.36 36.63</td><td>68.60 67.00</td><td>70.40</td><td>61.48</td><td>64.88</td><td>36.07 36.95</td><td>32.71</td><td>36.37</td></tr><tr><td>SDRRL</td><td>42.25</td><td>36.31</td><td>39.20</td><td>65.40</td><td>66.60</td><td>61.09</td><td>63.20</td><td>34.47</td><td>31.31</td><td>34.23</td></tr><tr><td rowspan="2">Ours</td><td>DFT-logits</td><td>43.29</td><td>35.14</td><td>38.46</td><td>68.40</td><td>70.60</td><td>62.08</td><td>66.91</td><td>36.89</td><td>31.84</td><td>35.30</td></tr><tr><td>DFT-feature</td><td>43.25</td><td>36.73</td><td>39.74</td><td>67.20</td><td>70.20</td><td>61.88</td><td>66.25</td><td>36.70</td><td>31.63</td><td>34.43</td></tr></table></body></html>

Table 2: Results of baselines and our method on multilingual understanding benchmark. Our method outperforms the baseline in almost all languages.

understanding dataset to evaluate models’ performance in cross-lingual inference tasks.

• XCOPA (Cross-lingual Choice of Plausible Alternatives): XCOPA (Ponti et al. 2020) is a benchmark designed to evaluate the ability of models to apply commonsense reasoning, requiring both world knowledge and the ability to generalize it to new languages.   
• XStoryCloze (Cross-lingual Story Cloze Test) XStoryCloze (Lin et al. 2022) is a cross-lingual dataset for evaluating models’ ability to understand stories and generate plausible endings.   
• MMLU (Massive Multitask Language Understanding) MMLU (Hendrycks et al. 2020) is a large-scale multitask language understanding dataset covering multiple domains (such as history, geography, science, law, etc.) and various languages.

For all evaluation datasets, we conducted tests using a zero-shot setting. We used the F1 score for XQuAD, MLQA, and MKQA, and the MC1 metric for TruthfulQA. For other NLU datasets, accuracy was used as the evaluation metric.

# Main Results

Table 1 and Table 2 present the results on multilingual QA and NLU tasks, respectively. From the experimental results, we can observe that: (1) Our method outperforms the baselines in almost all languages for both QA and NLU tasks. This indicates that our approach successfully enhances the model’s capabilities in the target language by guiding the internal workflow. (2) The improvement is more pronounced in QA tasks, as our method is specifically designed for tasks where both the input and output are in the target language, making it better suited for QA scenarios in the target language. (3) Fine-tuning the model solely on English instruction data (SFT) outperformed all other results fine-tuned on target language instruction datasets for TruthfulQA. This suggests that for the MC1 metric in TruthfulQA, generation capabilities in the target language are less important. (4) We observed that DFT (feature) performs better than DFT (logits) on understanding tasks, possibly because the stricter logits-based supervision is more suitable for generation tasks, whereas the feature-based supervision offers better generalization across different types of tasks.

Table 3: Results on 8 Chinese evaluation datasets with separately added language conversion supervision (LC) and english thinking supervision (ET). The supervision types are represented in “logits-based / feature-based” form.   

<html><body><table><tr><td></td><td>TFT</td><td>+LC</td><td>+ET</td></tr><tr><td>XQUAD</td><td>23.41</td><td>24.75/23.99</td><td>24.47 /24.26</td></tr><tr><td>MLQA</td><td>23.86</td><td>25.62/24.27</td><td>25.68/24.80</td></tr><tr><td>TruthfulQA</td><td>26.40</td><td>27.11/27.51</td><td>28.93/28.00</td></tr><tr><td>MKQA</td><td>40.70</td><td>40.73/40.81</td><td>40.83 / 40.70</td></tr><tr><td>XNLI</td><td>36.87</td><td>36.27 /36.59</td><td>37.61/37.75</td></tr><tr><td>XCOPA</td><td>65.00</td><td>65.60/66.80</td><td>66.40/68.40</td></tr><tr><td>Xstory_Cloze MMLU</td><td>64.06 32.34</td><td>63.60/64.92</td><td>64.95/64.26 33.29/32.85</td></tr></table></body></html>

# Ablation Study

We further analyzed the effects of applying supervision over either the first or the second stage. We compared the performance of separately adding Language Conversion supervision (LC) and English Thinking supervision (ET) on 8 Chinese datasets. The results, based on LLaMA2, are shown in the Table 3, where the supervision types are represented as ”logits-based / feature-based.”

From the experimental results, we can observe that: (1) Adding supervision during the English Thinking stage yielded more significant improvements, indicating that aligning responses has a greater impact. (2) Logits-based supervision led to greater improvements in QA tasks, while feature-based supervision was more beneficial for understanding tasks. This may be because generation tasks require stricter supervision signals. (3) Logits-based language conversion supervision caused performance drops in some tasks, suggesting that strong supervision over the earlier layers may harm the model’s original capabilities.

# Entropy-Based Critical Layer Selection

To implement our approach, it is crucial to accurately identify the critical layers that separate the different stages, particularly when determining the $\dot { \iota } - t h$ and $j - t h$ layers as shown in the Figure 2. Although previous works (Wendler et al. 2024; Zhao et al. 2024b) have discovered the mechanisms of information transformation within models when processing non-English inputs, identifying the critical layers at different stages remains challenging.

Entropy, in the context of information theory, is a measure of uncertainty or randomness in the information being processed. In neural networks, entropy can help identify where significant transformations or reductions in uncertainty occur. We observed that when LLMs processes non-English inputs, there are two significant drops in entropy (as shown in Figure 2 (a)) . These drops indicate key points where the model undergoes substantial information transformation. Based on these observations, we hypothesize that the initial entropy drop corresponds to the model processing the nonEnglish input into an English representation that it can handle. After this, the model transitions to the English reasoning and processing stage. The subsequent entropy drop marks the model gradually completing the reasoning process and progressively forming the final output.

![](images/3969ffcd8b7fbce1d8a5ab2f0ca372d22bc1624fc079bdd57a8b88d8abd54dda.jpg)  
Figure 3: The bars in the figure represent the results of DFTlogits and DFT-feature on all evaluation datasets, with English Thinking supervision applied at different layers. The target language is Chinese, and the base model used is LLaMA-2-7b. The dashed line indicates the results of the TFT method. The average scores across various datasets are reported. The broken line represents the change in entropy as the layer depth increases.

To validate the effectiveness of our hypothesis, we applied English Thinking supervision at layers 5, 10, 15, 20, 25, 30 of LLaMA-2-7b and fine-tuned the model on the Chinese instruction dataset. Figure 3 shows the average results across 8 Chinese datasets using the DFT-logits and DFT-feature methods. We observed that: (1) Entropy first drops at layer 2, stabilizes for a period, and then begins to drop again around layer 15. (2) Both DFT-logits and DFT-feature achieve better performance when applied at layer 15, indicating that our hypothesis is valid. (3) For the DFT-logits method, performance declines when supervision is applied at later layers, possibly because the model has already started the language reversion process (converting English back to the target language). Adding constraints to predict English results at this stage may interfere with the model’s generation of the target language. In contrast, the relatively more relaxed DFTfeature method performs better at later layers.

Although this hypothesis is rough (as different tokens may exhibit different behaviors), it provides us with guidance for selecting the critical layer at each stage.

![](images/c42fdd449899fa3055e15aed4f4fa5e44057df135c66a5aa0531da36894b3f5f.jpg)  
Figure 4: t-SNE visualizations of sentence representations from FLORES-200 dataset by LLaMA-2 before and after applying DFT.

# Analysis of Representation Alignment

We used the t-SNE (Van der Maaten and Hinton 2008) method to visualize the representations of input sentences to analyze the impact of DFT on aligning cross-lingual representations.

Specifically, we encoded parallel English and Chinese sentences from the FLORES-200 dataset and obtain sentence representations by the mean pooling method using the representation for each token.

The results are shown in Figure 4. In the vanilla model, the representations of the two languages are far apart. After applying the DFT method, they become more aligned. This indicates that our method can help bring the target language representations closer to the English representations.

# Analysis of Translation Task

The model’s performance on translation tasks can reflect its overall language conversion and generation capabilities. Therefore, although cross-lingual generation tasks do not align with the workflow of our method, we still analyzed our approach on translation tasks. We compared the performance of separately adding Language Conversion supervision and English Thinking supervision, and the results are shown in Table 3.

From the experimental results, we can observe that: (1) Adding logits-based supervision led to a catastrophic drop in the en-zh direction. This suggests that strict supervision of target language conversion into English within the model’s internal layers leads to a decline in the model’s overall ability to convert English to the target language. (2) Adding feature-based supervision significantly improved translation results for both en-zh and zh-en directions, indicating that aligning representations between languages is beneficial for cross-lingual tasks.

In fact, our method is better suited for scenarios where both the input and output are in the target language, rather than for cross-lingual tasks. Nevertheless, DFT-feature still achieved strong performance on translation tasks, indicating that our approach has broad potential for application.

# Related Work

Aligning Non-English Capabilities of Large Language Models To enhance the non-English capabilities of LLMs, researchers have explored several approaches. Pre-training LLMs on diverse multilingual datasets has proven effective in improving multilingual performance. However, this approach requires the collection of large amounts of data and significant computational resources (Le Scao et al. 2023; Cui, Yang, and Yao 2023; HIT-SCIR 2024). Instruction finetuning on translation datasets has also been successful in enhancing non-English performance(Li et al. 2023; Zhu et al. 2023b; Zhang et al. 2024; Xu, Li, and Xiong 2023; Li et al. 2024b). In the inference stage, cross-lingual transfer methods, such as leveraging knowledge from resource-rich languages and using self-translation prompts, have been effective (Qin et al. 2023; Xu, Li, and Xiong 2023; Huang et al. 2023). Additionally, Li et al. (2024a) aligned English representations to enable the model to fully leverage its English capabilities when processing non-English inputs.

Table 4: Translation performance on FLORES-200 with separately added language conversion supervision (LC) and english thinking supervision (ET), evaluated using COMET score.   

<html><body><table><tr><td>TFT</td><td>+LC</td><td></td><td>+ET</td></tr><tr><td>en-zh</td><td>68.37</td><td>36.79 /75.62</td><td>46.50 / 74.49</td></tr><tr><td>zh-en</td><td>63.12</td><td>64.10/82.58</td><td>63.68/83.98</td></tr></table></body></html>

Deep Supervision Networks Lee et al. (2015) first proposed the deeply supervised network, where auxiliary classifiers are added on various intermediate layers, and each classifier contributes to the overall loss during training. Huang et al. (2022) enhanced machine translation performance by predicting outputs layer by layer in a non-autoregressive manner. Elbayad et al. (2019) improved decoding efficiency by making predictions at different layers based on token prediction differences.

Our method draws on the implementation of deeply supervised networks by introducing supervision over the middle part of the model. Our goal is to guide the internal information transformation process within the model.

# Conclusion and Future Work

In this work, we propose Deep supervision Fine-Tuning, effectively enhancing the multilingual capabilities of Englishdominated LLMs. Our method guides the workflow of LLMs when processing non-English inputs by adding crosslingual supervision over intermediate layers, constraining models to achieve more accurate language conversion and obtain more precise intermediate results. Experimental results demonstrate that our method significantly improves performance on various multilingual tasks.

We devise an entropy-based method for critical layer selection and have preliminarily validated its effectiveness. However, variations among tokens within samples suggest that this guidance is still imprecise. We will further explore this issue in the future.