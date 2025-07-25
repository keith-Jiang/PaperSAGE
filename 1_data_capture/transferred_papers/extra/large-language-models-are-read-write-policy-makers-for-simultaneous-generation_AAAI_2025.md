# Large Language Models Are Read/Write Policy-Makers for Simultaneous Generation

Shoutao Guo 1,3, Shaolei Zhang 1,3, Zhengrui Ma 1,3, Yang Feng $^ { 1 , 2 , 3 * }$

1Key Laboratory of Intelligent Information Processing,   
Institute of Computing Technology, Chinese Academy of Sciences (ICT/CAS) 2 Key Laboratory of AI Safety, Chinese Academy of Sciences 3 University of Chinese Academy of Sciences, Beijing, China   
guoshoutao22z@ict.ac.cn, zhangshaolei20z@ict.ac.cn, fengyang@ict.ac.cn

# Abstract

Simultaneous generation models write generation results while reading streaming inputs, necessitating a policy-maker to determine the appropriate output timing. Existing simultaneous generation methods generally adopt the traditional encoder-decoder architecture and learn the generation and policy-making capabilities through complex dynamic programming techniques. Although LLMs excel at text generation, they face challenges in taking on the role of policymakers through traditional training methods, limiting their exploration in simultaneous generation. To overcome these limitations, we propose a novel LLM-driven Simultaneous Generation (LSG) framework, which allows the off-the-shelf LLM to decide the generation timing and produce output concurrently. Specifically, LSG selects the generation policy that minimizes latency as the baseline policy. Referring to the baseline policy, LSG enables the LLM to devise an improved generation policy that better balances latency and generation quality, and writes generation results accordingly. Experiments on simultaneous translation and streaming automatic speech recognition tasks show that our method can achieve state-of-the-art performance utilizing the open-source LLMs and demonstrate practicality in real-world scenarios.

# Code — https://github.com/ictnlp/LSG Extended version — https://arxiv.org/abs/2501.00868

# Introduction

Simultaneous generation models (Gu et al. 2017; Moritz, Hori, and Le 2020), which produce the target sentence before reading the entire input, are widely used in streaming scenarios such as real-time subtitles and online meetings. To achieve the goal of low latency and high generation quality (Zhang and Feng 2022b), simultaneous generation models require an optimal policy to determine the generation timing, ensuring that the generated results are consistent with those in non-streaming scenarios while minimizing latency (Alinejad, Shavarani, and Sarkar 2021). Consequently, the learning of generation policy is critical to the simultaneous generation tasks.

![](images/2a0c9b3886b1b33796182e860e161da3de8c95f5b8d77728798d4585c1843582.jpg)  
Figure 1: The distribution difference of subsequent generation states compared to wait-1 policy for a German $\Rightarrow$ English translation example. The distribution difference is measured by KL divergence.

In simultaneous generation tasks such as simultaneous translation (Ma et al. 2019) and streaming Automatic Speech Recognition (ASR) (Moritz, Hori, and Le 2020), existing methods are constrained to using non-streaming parallel data for model training due to the lack of annotated policies. To learn the generation policy, previous methods (Ma et al. 2020b; Miao, Blunsom, and Specia 2021) primarily utilize an encoder-decoder architecture (Vaswani et al. 2017) coupled with complex dynamic programming training techniques. This methodology endows simultaneous generation models with both generation and policy-making capabilities (Zhang and Feng 2023b). However, these models are constrained by their expressive capacity, resulting in suboptimal policies and generation performance. Additionally, they suffer from significant memory consumption and slow training speeds during training (Guo, Zhang, and Feng 2023a). More recently, the emergence of Large Language Models (LLMs) (Touvron et al. 2023) prompts researchers to explore their potential in simultaneous generation tasks (Koshkin, Sudoh, and Nakamura 2024; Agostinelli et al. 2024). Nevertheless, the decoder-only architecture and vast parameters of LLMs pose challenges in applying traditional dynamic programming methods for policy learning. Consequently, existing LLM-based methods leverage the generation capabilities of LLMs to produce outputs guided by either fixed policies (Ma et al. 2019) or policies provided by conventional encoderdecoder models (Guo et al. 2024). Unfortunately, these external policies not only introduce complex control processes but also result in inferior performance without considering the context of LLMs. Therefore, incorporating LLMs into simultaneous generation tasks remains challenging.

To bypass the need for policy training and derive effective policies for LLMs, a straightforward approach might be to compare the current outputs with the non-streaming results, generating the target words only when the two align. This is akin to deriving a new policy from a full-sentence policy, where the model can use the complete input for generation. However, this is not feasible in practice, as the model cannot access the entire input in advance. On the other hand, minimum source input is available during simultaneous generation. This insight leads us to consider whether we can derive a policy by comparing the generation results based on minimum input with those based on the current input.

Therefore, we attempt to develop an enhanced policy that improves upon a baseline policy, which defines the minimum input at each generation step. To validate our hypothesis, we conduct a comprehensive preliminary analysis. We utilize the wait-1 policy (Ma et al. 2019) as the baseline policy and Llama2-7B-chat (Touvron et al. 2023) as the LLM. Initially, we leverage the LLM to obtain the generation distribution for target words at each generation state, based on available source content. We then analyze the distribution differences between the baseline policy and subsequent generation states. Figure 1 illustrates a notable trend where the distribution differences gradually increase as more source content is processed. Crucially, once the necessary source content is available, the distribution differences become significant, indicating an opportune moment for generation. These findings suggest that leveraging distribution differences can effectively strike trade-offs between latency and generation quality. However, Figure 1 also highlights a special case where all distribution differences of some target words remain relatively minor, as the wait-1 policy already provides sufficient information for generation. This phenomenon, inherently influenced by language characteristics and word reordering, is unavoidable and necessitates specialized treatment in our approach.

In light of these insights, we propose the LLM-driven Simultaneous Generation (LSG) method, a novel approach that empowers the off-the-shelf LLM to determine the policies and generate outputs concurrently. Our LSG method enables the LLM to derive an enhanced policy from a baseline policy without needing policy learning. At each step, the LLM compares the distribution difference between the current input and the source content determined by the baseline policy. When this distribution difference reaches a predetermined threshold, the LLM is prompted to generate outputs. Otherwise, LSG continues to await the upcoming input. To address the special case illustrated in Figure 1, we utilize the confidence of the LLM to avoid excessive delays that might be caused by minor distribution differences. To validate the effectiveness of LSG, we conduct extensive experiments on simultaneous translation and streaming ASR tasks. Leveraging open-source LLMs, our method achieves state-of-the-art results on standard datasets and demonstrates practicality in real-world scenarios.

# Background

Simultaneous Generation Let $\mathbf { x } = ( x _ { 1 } , . . . , x _ { J } )$ denote the complete source sequence, where $x _ { i }$ represents a source word or a speech segment. The simultaneous generation model incrementally produces the target sentence ${ \textbf { y } } =$ $( y _ { 1 } , . . . , y _ { I } )$ with length $I$ based on a generation policy. To represent this policy, we introduce the notation $g _ { i }$ , which represents the length of the partial input sequence when generating $y _ { i }$ . Therefore, the policy for generating y from the source sequence $\mathbf { x }$ can be defined as $\mathbf { g } = ( g _ { 1 } , . . . , g _ { I } )$ . During inference, the simultaneous generation model generates the target sentence according to the following formula:

$$
p ( \mathbf { y } | \mathbf { x } , \mathbf { g } ) = \sum _ { i = 1 } ^ { I } p ( y _ { i } \mid \mathbf { x } _ { \leq g _ { i } } , \mathbf { y } _ { < i } ) ,
$$

where $p ( y _ { i } \mid \mathbf { x } _ { \leq g _ { i } } , \mathbf { y } _ { < i } )$ is the next token distribution.

Wait-k policy Simultaneous generation models require a policy to determine the timing of generating sentences. Currently, the most prevalent simultaneous generation policy is the wait- $\mathbf { \nabla } \cdot \mathbf { k }$ policy (Ma et al. 2019), which is simple and exhibits relatively inferior performance. During inference, the wait- $\mathbf { \nabla } \cdot \mathbf { k }$ policy initially reads $k$ source elements (i.e., speech segments or words), then alternates between generating a word and reading a source element. Therefore, the wait- $\mathbf { \nabla } \cdot \mathbf { k }$ policy can be expressed by the following equation:

$$
g _ { i } ^ { w a i t - k } = \operatorname* { m i n } \big \{ k + i - 1 , J \big \} ,
$$

where $J$ denotes the length of the whole input sequence. According to the Average Lagging metric (Ma et al. 2019) for latency evaluation, the policy with the minimum latency is the wait-0 policy. However, the wait-0 policy is impractical, as it would result in the simultaneous generation model producing the first word without conditioning on any source information. Therefore, we select the wait-1 policy as the baseline policy for our method.

# Method

In this section, we introduce our LLM-driven Simultaneous Generation (LSG) method, which empowers the LLM to perform policy-making and generation sub-tasks concurrently. We first present the framework of LSG and delineate its operational process. Subsequently, we elucidate how the LLM leverages a baseline policy to derive an enhanced policy. To address the limitations of the baseline policy in scenarios where the source content is already sufficient, we introduce an additional confidence condition for the enhanced policy. Finally, we implement a range constraint for the obtained policy to ensure controllable latency and mitigate the impact of some outlier policies. The following subsections provide a detailed exposition of our method.

![](images/069868f05c0268b589529a44320097bc03ed19caa61bf28168d628bd88faebe6.jpg)  
Figure 2: The framework of LLM-driven Simultaneous Generation Model.

# Model Framework

As shown in Figure 2, we introduce the model framework of LSG. Our LSG method empowers the LLM to perform both policy-making and generation sub-tasks. To this end, the LLM pre-establishes a baseline policy before initiating simultaneous generation.

At each generation step, LSG selects the source content corresponding to the baseline policy as a new input, based on the currently available input and previously generated words. Subsequently, LSG enables the LLM to predict the next target word based on the current input and the input determined by the baseline policy, respectively. This process yields two probability distributions: the current generation distribution and the distribution of baseline policy. LSG utilizes these distributions for policy-making to determine the action to be taken. If the READ action is selected, LSG refrains from producing any output at that moment and awaits the upcoming input. Conversely, if the WRITE action is chosen, LSG generates the target word based on the current distribution and appends it to the previously generated words. After that, a new generation step commences.

Our framework does not impose restrictions on the employed LLMs. However, the baseline policy needs to be predetermined in advance of simultaneous generation. As discussed in the Background section, to ensure low latency and usability of the baseline policy, we choose the wait-1 policy as our baseline policy.

# Policy-Making Procedure

In this subsection, we elaborate on the policy-making procedure in Figure 2. Our LSG method aims to develop an improved policy by referencing the baseline policy. At each generation step, it utilizes the differences between the current generation distribution with the distribution of the wait1 policy to decide on the taken action.

At the current generation step, we assume that the available source sequence is $\mathbf { x } _ { \leq j }$ and generated target words are $\mathbf { y } _ { < i }$ , where $j$ is greater than $i$ . Therefore, we can obtain $p ( y _ { i } | \mathbf { x } _ { \leq j } , \mathbf { y } _ { < i } )$ , which denotes the generation distribution of the LLM based on $\mathbf { x } _ { \leq j }$ and $\mathbf { y } _ { < i }$ . At the same time, under the guidance of the wait-1 policy, the LLM utilizes $\mathbf { x } _ { \leq i }$ and $\mathbf { y } _ { < i }$ to generate the distribution $\bar { p ( y _ { i } | \mathbf { x } _ { \le i } , \mathbf { y } _ { < i } ) }$ . These two distributions are used by LSG to calculate the KL divergence to decide on the action to be taken:

$$
\begin{array} { r } { \mathbb { D } _ { \mathrm { K L } } \big [ p ( y _ { i } | \mathbf { x } _ { \le j } , \mathbf { y } _ { < i } ) \mid \mid p ( y _ { i } | \mathbf { x } _ { \le i } , \mathbf { y } _ { < i } ) \big ] > \delta , } \end{array}
$$

where $\delta$ is the hyperparameter that represents the threshold. If the condition in Eq.(3) is met, LSG generates the target word based on the distribution $p ( y _ { i } | \bar { \mathbf { x } } _ { \le j } , \mathbf { y } _ { < i } )$ and appends it to the previously generated sequence. Otherwise, our method refrains from producing any output and waits for the upcoming input.

Confidence Condition Up to now, we have developed improved policies by referencing the baseline policy without needing traditional complex training methods (Zhang and Feng 2023b). However, due to factors such as language features and word reordering (Liu et al. 2021), the baseline policy may have already provided sufficient source information for some target words. As illustrated in Figure 1, this phenomenon can result in minor distribution differences when generating these words according to the condition in Eq.(3). We call this phenomenon as false negative, as it instructs the model to excessively read source information even if condition in Eq.(3) is met, resulting in redundant latency (Papi, Negri, and Turchi 2023). However, this phenomenon is unavoidable due to the diversity of language expression. To complement the condition in Eq.(3), we introduce an additional confidence condition.

Since LLMs typically assign probability mass to favorable behaviors (Li et al. 2023), the confidence of LLMs also reflects the credibility of the generation. In the face of the false negative problem in the condition of Eq.(3), we use the confidence of LLMs to mitigate this issue:

$$
\operatorname* { m a x } { p ( y _ { i } | \mathbf { x } _ { \leq j } , \mathbf { y } _ { < i } ) } > \alpha ,
$$

where $\alpha$ is the confidence hyperparameter that enables generation. Consequently, our LSG method executes the WRITE action when either the condition in Eq.(3) or Eq.(4) is satisfied. Otherwise, LSG awaits the upcoming input.

# Range constraint

After introducing the policy-making procedure, our LSG method can leverage the LLM to perform both policymaking and generation sub-tasks. However, when considering the practical applications, there are still issues with the above policy-making procedure. In the current setup, the search range for the target word $y _ { i }$ is $[ \operatorname* { m i n } \{ i , J \} , J ]$ , where $J$ denotes the length of the whole input sequence. However, the presence of outlier policies will inevitably lead to excessive latency or poor translation quality (Ma et al. 2020b). Moreover, it is challenging to ensure that the simultaneous generation model always responds within a fixed delay. Therefore, it is necessary to impose constraints on the search range of the policy.

In our LSG method, we set the search range for the target word $y _ { i }$ as:

$$
[ \operatorname* { m i n } \big \{ L + i - 1 , J \big \} , \operatorname* { m i n } \big \{ L + i - 1 + U , J \big \} ] ,
$$

where $L$ denotes the number of pre-read elements before simultaneous generation and $U$ represents the degree of autonomy afforded to the LLM in policy-making.

# Experiments

# Datasets

We mainly conduct experiments on simultaneous text-totext translation (SimulT2TT), simultaneous speech-to-text translation (SimulS2TT), and streaming ASR tasks.

WMT151 German $\Rightarrow$ English $( \mathbf { D e } \Rightarrow \mathbf { E n } )$ ) We conduct SimulT2TT task on this dataset. Consistent with Ma et al. (2020b), we use the newstest2015 set as the test set.

MuST-C English $\Rightarrow$ German $\mathbf { E n } { \Rightarrow } \mathbf { D e } )$ ) This dataset (Di Gangi et al. 2019) is collected from TED talks and we conduct the SimulT2TT task using its text data.

CoVoST2 French $\Rightarrow$ English $( \mathbf { F r } \Rightarrow \mathbf { E n } )$ ) We use this dataset (Wang, Wu, and Pino 2020) to conduct both SimulS2TT and streaming ASR tasks.

# System Settings

Since our method can be applied to SimulT2TT, SimulS2TT, and streaming ASR tasks, we will delineate the comparative methods for each of these tasks separately and then present the settings of our LSG method.

For SimulT2TT task, the baseline methods include wait-k (Ma et al. 2019), MMA (Ma et al. 2020b), ITST (Zhang and Feng 2022a), HMT (Zhang and Feng 2023b) and AgentSiMT (Guo et al. 2024). With the exception of AgentSiMT, the aforementioned methods all use the traditional encoder-decoder architecture. HMT, which learns policies through sophisticated dynamic programming training methods, achieves the superior performance among conventional approaches. Agent-SiMT, leveraging an agent collaboration mechanism and utilizing policies provided by HMT to guide the LLMs in translation generation, has achieved state-ofthe-art performance in the SimulT2TT task.

For SimulS2TT task, we compare our method against DiSeg (Zhang and Feng 2023a) and StreamSpeech (Zhang et al. 2024a). Both DiSeg and StreamSpeech adopt the encoder-decoder architecture, with StreamSpeech achieving state-of-the-art performance in the SimulS2TT task. To validate the practical applicability of our method, we additionally evaluate all approaches using computation-aware latency metrics for this task.

For streaming ASR task, Wav2Vec2-large (Baevski et al. 2020) and Whisper-base (Radford et al. 2022) are used as the baseline methods. Both Wav2Vec2 and Whisper are pretrained models, with Whisper demonstrating superior performance across multiple ASR datasets.

Since LSG is a general simultaneous generation framework, it does not impose restrictions on the LLMs used. Due to the constraints of different tasks, we employ different LLMs for different evaluated tasks. For the SimulT2TT task, we maintain the same setup as Guo et al. (2024). We employ Llama2-7B-chat2 as the LLM and perform fine-tuning on 10w extracted samples using LoRA (Hu et al. 2021). For the SimulS2TT and streaming ASR tasks, we use the opensource speech LLM, Qwen-Audio3 (Chu et al. 2023). As the multimodal version of the Qwen (Bai et al. 2023) series, Qwen-Audio achieves good comprehension and generation capabilities in multiple speech tasks after audio-language pre-training. During inference, the duration of each speech segment is set to $6 4 0 ~ \mathrm { m s }$ . The prompt templates used in our experiments are consistent with those used during the training of the LLMs. We set $\delta = 9 . 0$ and $\alpha = 0 . 6$ for $_ { \mathrm { D e } \Rightarrow \mathrm { E n } }$ task, $\delta = 7 . 5$ and $\alpha = 0 . 6$ for $\scriptstyle \mathrm { E n \Rightarrow }$ De task, and $\delta = 7 . 0$ and $\alpha = 0 . 5$ for $\mathrm { F r } { \Rightarrow }$ En task. For different latency scenarios, we set $[ L , U ]$ as [1, 4], [3, 4], [5, 6], and [7, 6], respectively.

# Evaluation

In evaluating streaming generation systems, we employ the SimulEval toolkit (Ma et al. 2020a) to assess two critical aspects: latency and generation quality. Systems that demonstrate low latency while maintaining high generation quality are generally considered superior.

To quantify latency, we utilize the Average Lagging (AL) metric (Ma et al. 2019), which measures the delay between input reception and output generation in simultaneous generation systems. For textual input, AL is calculated in terms of word count, whereas for speech input, it is measured in milliseconds (ms). Additionally, for the SimulS2TT task, we evaluate computation-aware latency on an NVIDIA RTX 3090 GPU, which assesses the latency of the systems in practical applications.

To assess generation quality, we employ task-specific metrics. For SimulT2TT and SimulS2TT tasks, we utilize the SacreBLEU metric (Post 2018), a widely used metric in translation. For the streaming ASR task, we adopt the Word Error Rate (WER) as our primary evaluation metric.

![](images/07639f4d252c37819ac5b6110e03d61adc4f3a84827bb4a033e38c10e55efe4c.jpg)  
Figure 3: Performance of simultaneous generation models on $_ { \mathrm { D e } \Rightarrow \mathrm { E n } }$ , ${ \mathrm { E n } } { \Rightarrow } \mathrm { D e }$ and $\mathrm { F r } \Rightarrow$ En datasets. We also evaluate the Computation-Aware (CA) latency on the CoVoST2 $\mathrm { F r } \Rightarrow$ En dataset to assess the usability of systems in real-world scenarios.

Table 1: The streaming ASR performance of simultaneous generation models on the CoVoST $2 \mathrm { F r } { \Rightarrow } \mathrm { E n }$ dataset.   

<html><body><table><tr><td>Method</td><td>AL (ms) (↓)</td><td>WER (↓)</td></tr><tr><td rowspan="2">Wav2Vec2-large Whisper-base</td><td>5684.38</td><td>26.17</td></tr><tr><td>5684.38</td><td>38.04</td></tr><tr><td rowspan="2">LSG</td><td>3161.25</td><td>31.71</td></tr><tr><td>4342.23</td><td>23.76</td></tr></table></body></html>

# Main Results

We evaluate the performance of our method on SimulT2TT, SimulS2TT, and streaming ASR tasks.

For the SimulT2TT task, we present the performance of various simultaneous generation models in Figure 3(a) and Figure 3(b). Our method achieves state-of-the-art performance across both datasets. Compared to traditional approaches (Ma et al. 2020b; Zhang and Feng 2023b) that utilize the encoder-decoder framework, our method demonstrates significant improvements in simultaneous translation performance. Conventional methods require the design of intricate policy modules integrated into the transformer architecture (Vaswani et al. 2017), followed by training through sophisticated dynamic programming techniques. However, these traditional methods are often constrained by their expressive capacity, resulting in inferior generation performance. Our approach leverages the enhanced comprehension and generation capabilities of LLMs, leading to superior performance. In addition to the traditional methods, our method also outperforms LLM-based methods (Guo et al. 2024). Previous LLM-based methods necessitate coupling an external policy module with the LLM to accomplish simultaneous translation tasks, which fails to provide appropriate policies for the LLM and increases system complexity. In contrast, our method allows LLMs to utilize their inherent understanding capabilities to acquire policies, which then guide the translation generation process. This results in better trade-offs between latency and translation quality.

For the SimulS2TT task, Figure 3(c) compares our method with other simultaneous speech translation methods. As the first method to utilize LLMs for simultaneous speech translation, our approach outperforms previous methods across all latency levels. Previous approaches rely on speech pre-training models (Zhang and Feng 2023a), multi-task training (Zhang et al. 2024a), and dynamic programming strategies (Liu et al. 2021) to enhance performance. However, these methods necessitate complex and multiple training processes and are constrained by the generation capabilities of the model. In contrast, our method transforms off-the-shelf speech LLMs into simultaneous speech translation systems directly, serving both policy-making and generation roles. By leveraging the speech understanding and instruction-following capabilities of Qwen-Audio, our method significantly further improves simultaneous speech translation performance. Additionally, we provide results for computation-aware latency, which considers both the delay between input and output and the model inference time, reflecting the latency of real-world scenarios. Despite using speech LLMs, our method can respond to speech input with a delay of only 3 seconds, demonstrating its practical applicability. Moreover, our method can be accelerated with better GPUs and inference frameworks, making it well-suited for simultaneous speech translation tasks.

For the streaming ASR task, we compare our method with previous pre-trained speech models, as shown in Table 1. Our LSG method achieves recognition quality comparable to previous methods with a delay of 6 seconds while maintaining only about a delay of 3 seconds. Although the methods based on pre-trained models have been trained on large amounts of speech data, they often lack language generation capabilities and struggle to establish effective generation policies. In contrast, by utilizing the speech comprehension and language generation abilities of speech LLMs (Chu et al. 2023), our approach provides superior generation policies in streaming scenarios. By combining advantages in both generation and policy, our method achieves better streaming ASR performance.

Therefore, by leveraging the policy-making and generation capabilities of off-the-shelf LLMs, our LSG method can attain the best generation performance across multiple simultaneous generation tasks.

Table 2: The ablation experiments of our method, where “w/o Confidence” represents the removal of the confidence condition and “w/o Range” indicates our method without range constraint. The experimental results are all based on the $\mathrm { D e } { \Rightarrow } \mathrm { E n }$ task.   

<html><body><table><tr><td>Method</td><td>AL (word) (↓)</td><td>SacreBLEU (↑)</td></tr><tr><td rowspan="2">LSG</td><td>4.42</td><td>31.60</td></tr><tr><td>7.37</td><td>33.22</td></tr><tr><td rowspan="2">w/o Confidence</td><td>4.89</td><td>31.34</td></tr><tr><td>6.75</td><td>32.72</td></tr><tr><td rowspan="2">w/o Range</td><td>3.62</td><td>21.95</td></tr><tr><td>12.91</td><td>29.90</td></tr></table></body></html>

Table 3: Ablation study on speech segment size in the SimulS2TT task. The experimental results are based on the $\mathrm { F r } \Rightarrow$ En dataset.   

<html><body><table><tr><td>Segment Size (ms)</td><td>AL (ms) (↓)</td><td>SacreBLEU (↑)</td></tr><tr><td rowspan="2">320</td><td>1566.42</td><td>31.71</td></tr><tr><td>3003.99</td><td>36.08</td></tr><tr><td rowspan="2">640</td><td>1582.94</td><td>32.20</td></tr><tr><td>3022.18</td><td>36.19</td></tr><tr><td>960</td><td>3101.12</td><td>36.47</td></tr></table></body></html>

![](images/ae32e369f02cf8a5bf370ff6ffb53aacc8b5325257bedd5d010baae9eb43f216.jpg)

Figure 4: The performance of LSG framework when employing various LLMs. The results are reported on the WMT22 Chinese $\Rightarrow$ English dataset.   
Table 4: The performance of LLMs in non-streaming scenarios. The numerical results are based on the WMT22 Chinese $\Rrightarrow$ English dataset.   

<html><body><table><tr><td>LLMs</td><td>ParroT-7B</td><td>Bayling-7B</td><td>Bayling-13B</td></tr><tr><td>SacreBLEU</td><td>18.73</td><td>20.72</td><td>23.57</td></tr></table></body></html>

# Analysis

To deepen the understanding of our approach, we conduct extensive analyses. We then introduce each analytical experiment in detail separately.

# Ablation Study

To explore the impact of different settings in our method, we conduct several ablation experiments.

Table 2 demonstrates that all components of our LSG method contribute to the performance of simultaneous generation. Firstly, the introduction of the confidence condition mitigates the false negative problem inherent in using only the condition in Eq.(3). This confidence condition enables our method to select the WRITE action when the current generation does not satisfy the condition in Eq.(3) but exhibits high confidence. This allows our method to avoid unnecessary delays caused by waiting for additional source information (Tang et al. 2023), consequently achieving superior performance. More importantly, the range constraint facilitates even more substantial improvements in our method. By employing this constraint, our approach effectively controls the scope and autonomy of LLMs in determining generation policies. This constraint allows us to limit the policymaking range of LLMs based on linguistic features (Miao, Blunsom, and Specia 2021), striking better trade-offs while ensuring timely responses.

We also investigate the influence of segment size when processing speech input. Table 3 illustrates the performance of our method on the SimulS2TT task across various segment sizes. The results indicate that our approach exhibits robustness to changes in source speech segment size. While a segment size of 960 achieves relatively strong performance, it lacks the flexibility to adapt to low-latency requirements in practical applications. Conversely, a segment size of 320 necessitates more frequent LLM inferences, resulting in increased computational costs. Consequently, we opt for a speech segment size of 640 in our experimental setup. This choice delivers superior performance among the three configurations while allowing for flexible latency adjustments to meet diverse operational needs.

# Influence of LLMs

Following our ablation experiments, we further analyze the impact of different LLMs on simultaneous generation performance. Our objective is to investigate whether more advanced LLMs can yield better simultaneous generation results within our LSG framework.

To this end, we evaluate ParroT-7B (Jiao et al. 2023), Bayling-7B (Zhang et al. 2024b), and Bayling-13B on the $\mathrm { w i M T } \bar { 2 2 } ^ { 4 }$ Chinese $\Rightarrow$ English translation dataset. We initially assess the performance of these LLMs in non-streaming scenarios in Table 4. The results demonstrate that the models of the Bayling family outperform ParroT-7B, achieving superior translation quality. Moreover, Bayling-13B, with its advantages of more parameters, surpasses the performance of Bayling-7B.

![](images/0cef77d94ca3587fcc2d2f4f0aa8ba874cef21e849c85f57ef81cdd34613e9b1.jpg)  
Figure 5: Comparison of the policy sufficiency of different simultaneous generation policies. The experiments are based on the $\mathrm { D e } \Rightarrow$ En dataset.

Building upon the insights of non-streaming performance, we then integrate these LLMs into our LSG framework. Figure 4 illustrates the performance of our method when utilizing different LLMs. Leveraging their enhanced Chinese $\Rrightarrow$ English translation capabilities, the models of the Bayling family achieve better trade-offs between latency and translation quality. Notably, Bayling-13B, with its substantial number of parameters, attains superior performance in simultaneous translation compared to Bayling-7B.

These findings underscore that our method serves as a versatile, unified framework applicable to existing LLMs. Furthermore, it demonstrates the potential to achieve enhanced streaming generation performance when integrated with more advanced LLMs.

# Quality of Policy

After exploring the relationship between LLMs and simultaneous generation performance, we further investigate the quality of the policies obtained by our LSG method. In simultaneous generation, generation is considered sufficient if the target word is produced after reading the aligned source information under the guidance of policy (Guo, Zhang, and Feng 2024a). Conversely, when LLMs rely solely on their anticipation capabilities for next-token prediction, the outcome is undesired. Therefore, we want to compare the sufficiency of the generation outputs under different policies to validate the quality of our learned policy.

To this end, we employ the eflomal5 toolkit to obtain input-output alignments and calculate generation sufficiency. We evaluate the sufficiency of our LSG method against external policies such as wait- $\mathbf { \nabla } \cdot \mathbf { k }$ and HMT when applied to the Llama2-7B-chat (Touvron et al. 2023) model. The results in Figure 5 show that our method consistently achieves higher generation sufficiency under all latency. Leveraging the comprehension capabilities of LLMs, our method enables the LLM to develop superior policies, surpassing the sufficiency of generation under the guidance of external policies. This underscores that our LSG method empowers LLMs to acquire suitable policies without the need for explicit policy learning.

# Related Work

SimulT2TT Recent SimulS2TT methods are broadly divided into two categories: encoder-decoder and LLMs. The approaches using the encoder-decoder architecture initially employ the wait-k policy (Ma et al. 2019) and enhance performance through training methods (Elbayad, Besacier, and Verbeek 2020; Chen et al. 2021b; Guo, Zhang, and Feng 2023b, 2024b). Further efforts in this line of work employ techniques such as monotonic attention (Arivazhagan et al. 2019; Ma et al. 2020b), wait-info (Zhang, Guo, and Feng 2022), hidden Markov models (Zhang and Feng 2023b), CTC-based non-autoregressive structure (Ma et al. 2023) to conduct policy learning and translation concurrently. With the advent of LLMs, some methods (Agostinelli et al. 2024) attempt to utilize external policy to guide LLMs.

SimulS2TT Recent SimulS2TT approaches mainly focus on adapting speech segmentation or enhancing model structures. Initial method (Ma, Pino, and Koehn 2020) attempts to split source speech into fixed-length segments. Subsequent work tries to adaptively segment speech using techniques such as auxiliary ASR task (Zeng, Li, and Liu 2021; Chen et al. 2021a), integrate-and-fire model (Dong et al. 2022), and differentiable segmentation (Zhang and Feng 2023a), applying the wait- $\mathbf { k }$ policy to the resulting segments. In contrast, other work focuses on enhancing SimulS2TT performance through enhanced architectures such as augmented Transducer (Liu et al. 2021) and combinations of transducer and encoder-decoder model (Tang et al. 2023). To the best of our knowledge, no prior research has explored the potential of leveraging LLMs to address the SimulS2TT task.

Streaming ASR Previous Streaming ASR methods primarily rely on transducer (Yeh et al. 2019; Li et al. 2020) and attention-based (Fan et al. 2019; Moritz, Hori, and Roux 2019) architectures. More recently, the robust performance of pre-trained speech models (Baevski et al. 2020; Radford et al. 2022) in various ASR tasks has also led to their widespread adoption in streaming ASR tasks.

Previous simultaneous generation methods rarely explore the use of LLMs and cannot fully harness the policy-making and generation capabilities of LLMs. Therefore, our LSG method enables the off-the-shelf LLM to develop improved policies by considering a baseline policy and then completing generation accordingly. This allows the LLM to autonomously and efficiently complete the simultaneous generation without the need for complex training methods.

# Conclusion

In this paper, we propose a novel LLM-driven simultaneous generation method that allows the LLMs to decide the generation timing and produce output concurrently. Experiments show that our method achieves state-of-the-art performance demonstrates practicality in real-world scenarios.