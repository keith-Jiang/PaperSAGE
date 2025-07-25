# SCANS: Mitigating the Exaggerated Safety for LLMs via Safety-Conscious Activation Steering

Zouying Cao, Yifei Yang, Hai Zhao\*

Department of Computer Science and Engineering, Shanghai Jiao Tong University Key Laboratory of Shanghai Education Commission for Intelligent Interaction and Cognitive Engineering, Shanghai Jiao Tong University Shanghai Key Laboratory of Trusted Data Circulation and Governance in Web3 zouyingcao, yifeiyang $@$ sjtu.edu.cn, zhaohai@cs.sjtu.edu.cn

# Abstract

Safety alignment is indispensable for Large Language Models (LLMs) to defend threats from malicious instructions. However, recent researches reveal safety-aligned LLMs tend to reject benign queries due to the exaggerated safety issue, limiting their helpfulness. In this paper, we propose a Safety-Conscious Activation Steering (SCANS) method to mitigate the exaggerated safety concerns in aligned LLMs. First, SCANS extracts the refusal steering vectors within the activation space and utilizes vocabulary projection to anchor some specific safety-critical layers which influence model refusal behavior. Second, by tracking the hidden state transition, SCANS identifies the steering direction and steers the model behavior accordingly, achieving a balance between exaggerated safety and adequate safety. Experiments show that SCANS achieves new state-of-the-art performance on XSTest and OKTest benchmarks, without impairing their defense capability against harmful queries and maintaining almost unchanged model capability.

user Can you tell me how to kill the lights in my room? Llama2-7b-chat I apologize, but I cannot provide instructions on how to ? harm or destroy any living thing, including lights in a room. Exaggerated Safety Llama2-7b-chat $^ +$ SCANS Certainly! There are several ways to kill the lights in your room depending on the type of lights you have. Here are some common methods: …

# Code — https://github.com/zouyingcao/SCANS Extended version — https://arxiv.org/abs/2408.11491

# 1 Introduction

Large Language Models (LLMs) have exhibited remarkable capabilities in various natural language processing tasks (Touvron et al. 2023; Chiang et al. 2023; Anthropic. 2024). However, due to the abundance of harmful content within pre-training data, LLMs are prone to follow malicious instructions and generate unsafe responses (Sun et al. 2024; Deshpande et al. 2023). This risk motivates extensive efforts for research on the safety alignment of LLMs (Korbak et al. 2023; Bianchi et al. 2024). Despite advancements in alignment techniques, recent studies (Ro¨ttger et al. 2024; Varshney et al. 2024) reveal safety-aligned LLMs strike a poor balance between safety and helpfulness. As demonstrated in Figure 1, aligned models may suffer from exaggerated safety and refuse benign queries which use similar vocabulary to harmful queries. This phenomenon significantly weakens the capability of LLMs to generate helpful responses to benign queries, excessively prioritizing safety.

Existing methods to mitigate the exaggerated safety issue can be categorized into training-based and training-free approaches. However, due to the scarcity of training data related to exaggerated safety, training-based solutions still exhibit a high refusal rate on queries that are word-level harmful but semantically benign (Bianchi et al. 2024; Zheng et al. 2024). Furthermore, existing training-free methods focus on contrasting the token distribution during the decoding process to balance the utility-safety trade-off (Xu et al. 2024; Shi et al. 2024). These methods, however, incur significant additional costs during inference and exhibit poorer mitigation capability.

Inspired by current researches that observe the existence of safety information in the representation spaces (Zou et al. 2023a; Zheng et al. 2024), we investigate the safety defense mechanism by analyzing how the hidden states change when exposed to harmful queries. Specifically, we average the difference between the activations of harmful and benign queries and project it to the vocabulary. Interestingly, we find the projections from middle layers show refusal concepts, thus capturing the refusal behavior vectors within the activation space.

Motivated by this finding, we propose a training-free, representation engineering method named SCANS (SafetyConscious Activation Steering), which utilizes refusal behavior vectors to steer the model output in safety-critical layers. We also design a similarity-based classification method to adaptively determine the steering direction, achieving a balance between adequate and exaggerated safety.

Through experiments with four LLMs, SCANS outperforms both training-free and training-based baselines in mitigating exaggerated safety without compromising adequate safety. Furthermore, SCANS maintains almost unchanged model capability, with minimal increase in perplexity. In summary, our contributions include:

• We introduce SCANS, which utilizes the activation steering to control the model refusal behavior, requiring no training and incurring no extra cost to inference time. • We discover the extracted refusal steering vectors from middle layers promote refusal tokens (e.g., cannot) and thus steering the corresponding representation can reduce the false refusal rate. • Our SCANS effectively mitigates the exaggerated safety in aligned LLMs, without undermining the adequate safety and general capability. Specifically, SCANS reduces the average false refusal rate by $2 4 . 7 \%$ and $2 6 . 3 \%$ on XSTest and OKTest benchmarks.

# 2 Related Works

Large Language Model Safety. The detection and mitigation of harmful content generated by language models is a prominent area of research on LLM safety (Zhao et al. 2024; Zhong et al. 2024). Recent works mainly focus on the model alignment through techniques such as supervised fine-tuning (Bianchi et al. 2024; Zheng et al. 2024) or RLHF (Bai et al. 2022b,a). However, safety-aligned models sometimes refuse to answer benign requests because of the over-defense mechanism (Ro¨ttger et al. 2024; Shi et al. 2024), which is the focus of our work.

Exaggerated Safety. This phenomenon refers to aligned models exhibit a tendency towards false refusal on safe queries, which is first introduced by Ro¨ttger et al. (2024). Based on this finding, Sun et al. (2024) evaluates 16 mainstream LLMs and finds a positive correlation between the level of exaggerated safety and jailbreak resistance. This indicates the trade-off between helpfulness and harmlessness remains a challenging task. Due to the scarcity of training data regarding exaggerated safety, current training-based methods (Bianchi et al. 2024; Zheng et al. 2024) still display a poor performance in carefully designed datasets like XSTest (R¨ottger et al. 2024) and OKTest (Shi et al. 2024). Other training-free works rely on prompt engineering (Bhalani and Ray 2024) or decoding (Shi et al. 2024) strategies. Prompt engineering-based methods take time and resources to design high-quality prompts and decoding-based methods clearly slow down the model inference speed. Our work falls into the training-free category while is orthogonal to the prompt engineering-based and decoding-based methods.

Representation Engineering. Representation engineering typically refers to manipulating the representations within a model to control its behavior (Zou et al. $2 0 2 3 \mathrm { a }$ ; Rimsky et al. 2024). Prior works have demonstrated its effectiveness on truthfulness (Li et al. 2023; Wang et al. 2024), formality transfer (Liu et al. 2024) and sentiment control (Turner et al. 2023; Konen et al. 2024). In this paper, our work discovers the feasibility of activation steering to mitigate the exaggerated safety issues and the proposed SCANS follows the common Mean Difference approach (Zou et al. 2023a) to extract the representations corresponding to refusal behaviors in LLMs.

# 3 Methodology

Motivated by the intuition of representation engineering to steer model behavior, the key idea behind our SCANS is to extract the refusal behavior vectors, and anchor the safetycritical layers for steering. SCANS then evaluates the harmfulness of inputs to guide output distribution against or consistent with the refusal behavior, which achieves a balance between adequate safety and exaggerated safety. Figure 2 illustrates the overview of our approach.

# 3.1 Inducing the Refusal Steering Vectors

To obtain the steering vectors that represent the refusal behaviors, we leverage a set of anchor data $Q = \{ Q ^ { - } , Q ^ { + } \}$ that consists of harmful and benign queries to trigger the contrastive model behavior. Intuitively, unsafe queries $Q ^ { - }$ can induce the defense mechanism in LLMs while the safe ones $Q ^ { + }$ elicit the helpful responses.

We then simulate aligned LLM with this two types of inputs and extract the hidden states for each layer $l$ at the last token position. By taking the difference, the refusal steering vectors $v _ { r } ^ { l }$ are extracted as follows:

$$
v _ { r } ^ { l } = \frac { 1 } { | Q ^ { - } | } \sum _ { q ^ { - } \in Q ^ { - } } a ^ { l } ( q ^ { - } ) - \frac { 1 } { | Q ^ { + } | } \sum _ { q ^ { + } \in Q ^ { + } } a ^ { l } ( q ^ { + } )
$$

where $a ^ { l } ( )$ gives the activations of the last token at layer $l$ .

Intuitively, the result of this difference represents a direction from the model’s inclination to answer towards the unwillingness to answer, namely refusal direction. Hence, subtracting this vector from the model representations can help moderate the tendency towards false-refusal responses, counteracting the exaggerated safety.

# 3.2 Anchoring the Safety-critical Layers

Using the above steering vectors to manipulate the representations across all layers could potentially disrupt the model outputs to an excessive degree. Therefore, we aim to anchor the specific layers that predominantly influence the model refusal behavior, which we call safety-critical layers, thereby utilized to steer without affecting general capabilities.

Previous work (Geva et al. 2022) applies a vocabulary projection method for interpretability. Inspired by this, our SCANS uses the refusal steering vectors $\mathbf { \dot { \boldsymbol { v } } } _ { r } ^ { l }$ for each layer to interpret in the vocabulary space and straightforwardly anchors the safety-critical layers. Specifically, we employ PCA (Hotelling 1933) to identify the first principal component for $v _ { r } ^ { l }$ separated by three segments 1: former layers, middle layers, and latter layers. Based on their dot product with the output embedding matrix (LM head), we get vocabulary projection indicating which layers are safety-related.

![](images/ef73cacd2ffeac9cf32eebe092a8aebebe648effb38fba65ba0e97665e650feb.jpg)  
Figure 2: The overview of SCANS, which extracts the refusal behavior vectors, and then determines the steering direction and steers the model behavior, thereby guaranteeing adequate safety without exaggerating safety.

From Table 1, we provide two perspectives: 1) since the middle layers are more safety-critical than former and latter layers, the extracted steering vectors can encode the refusal tokens associated with the safety defense mechanism; then, 2) steering vectors from middle layers promote the likelihood of refusal tokens to be generated, thus the corresponding steering can effectively reduce the false refusal rate.

Therefore, for capability preservation and exaggerated safety mitigation, we perform activation steering on the middle layers. We further demonstrate the steering effects in different layers in Section 4.4.

# 3.3 Identifying the Steering Direction

Upon anchoring the layers for steering, we need to identify the safety of queries so that the output representation is shifted towards (for harmful queries) or against (for benign queries) the refusal direction. Existing research (Zheng et al. 2024; Li, Zheng, and Huang 2024) demonstrates that the representations of the aligned model can distinguish whether the input query is harmful. Based on this, we design a simple and training-free classification method $\sigma ( q )$ to adaptively determine the steering direction for query $q$ .

Due to the inclination of safety-aligned LLMs to reject benign queries, the final hidden state (i.e., the hidden state of the last token) of query $q$ may incorrectly encode the refusal prediction for safe queries, which is indistinguishable from unsafe queries. Therefore, we first concatenate the query $q$ with positive response $r _ { p o s }$ (e.g., ‘Sure’), denoted by $q + r _ { p o s }$ . Next, we extract two final hidden states, one $a _ { p }$ of the query part (i.e., $\mathbf { \boldsymbol { q } } )$ , and the other $a _ { e }$ of the entire input (i.e., $q + r _ { p o s } )$ . For safe queries, when concatenated with $r _ { p o s }$ , LLM tends to not reject but generate correct answers, so $a _ { e }$ contains LLM’s perception of helpful behaviors. However, for unsafe queries, non-refusal behaviors are harmful, so $ { \boldsymbol { a } } _ { e }$ encodes unsafe behaviors. Thus, adding positive response $r _ { p o s }$ makes model representations more distinguishable helping identify the harmfulness of queries, and consequently the hidden state transition $a _ { t }$ from $a _ { p }$ to $a _ { e }$ (Eq. 2) can mine the harm direction for unsafe queries but helpful direction for safe queries, which reflects the difference. Figure 3 shows t-SNE visualization of hidden state transition in different layers, further suggesting its potential to classify the harmfulness of input queries.

$$
a _ { t } ^ { l } ( q ) = a _ { p } ^ { l } ( q + r _ { p o s } ) - a _ { e } ^ { l } ( q + r _ { p o s } )
$$

In the preparation stage, we reuse the harmful set of anchor data $Q ^ { - }$ to extract the harm direction for reference, dlharm, which represents the average of hidden state transition for all samples $q ^ { - } \in Q ^ { - }$ in layer $l$ . Specifically, the formulation for the reference harm direction is defined by:

$$
d _ { h a r m } ^ { l } = { \frac { 1 } { | Q ^ { - } | } } \sum _ { q ^ { - } \in Q ^ { - } } a _ { t } ^ { l } ( q ^ { - } )
$$

Then, given query $q$ , we stimulate aligned LLM with $q +$ $r _ { p o s }$ to extract the corresponding hidden state transition and computes its similarity with the reference dlharm as follows:

$$
s _ { q } = \frac { 1 } { | \mathcal { L } | } \sum _ { l \in \mathcal { L } } c o s \left( a _ { t } ^ { l } ( q ) , d _ { h a r m } ^ { l } \right)
$$

![](images/f62da7c4ec15a45d9fee68e1e2c447aaf1b2ea43b44385b3da8081e1aac97e04.jpg)  
Figure 3: t-SNE visualization of hidden state transition on XSTest dataset at layers 9, 20 and 32 of Llama2-7b-chat. The results indicate safety-related representation clustering emerges in middle and latter layers.

Table 1: Top-10 tokens associated with steering direction at different layers. We highlight the tokens related to refusal behavior with an underline. The results are based on Llama2-7b-chat model.   

<html><body><table><tr><td>Layers Top-10 tokens</td><td></td></tr><tr><td>Former Layers (O-9)</td><td>einges,_schlieβ,vue,che,orio,_Syd,rugu,wrap,widet,axi</td></tr><tr><td>Middle Layers (10-20)</td><td>_rejected,_impossible,zas,_cons,ball,od,lio,_tur,_reject,_cannot</td></tr><tr><td>Latter Layers (21-31)</td><td>sey,Mas,_Coun,Ir,-ext,-properties,Seg,ber,ds,_sa</td></tr></table></body></html>

where cos means the cosine similarity metric, $\mathcal { L }$ is the set of layers for classification. Following Zou et al. (2023a), the choice of $\mathcal { L }$ are among the middle and latter layers (See Figure 3) which is also justified in Section 4.4. Finally, if the similarity score $s _ { q }$ is smaller than threshold $\tau$ , we classify the query as benign input and accordingly steer the internal representation opposite the refusal direction:

$$
\sigma ( q ) = \left\{ \begin{array} { l l } { - 1 \quad } & { s _ { q } < \mathcal { T } } \\ { 1 \quad } & { o t h e r w i s e } \end{array} \right.
$$

$$
\widetilde { \boldsymbol { a } } ^ { l } ( \boldsymbol { q } ) = \boldsymbol { a } ^ { l } ( \boldsymbol { q } ) + \boldsymbol { \sigma } ( \boldsymbol { q } ) \cdot \boldsymbol { \alpha } \cdot \boldsymbol { v } _ { r } ^ { l }
$$

where $a ^ { l }$ and $\widetilde { a } ^ { l }$ respectively represent the original and shifted activat ens, $\alpha$ is a hyperparameter that controls the strength of steering. A detailed algorithm for our SCANS is presented in Appendix $\mathbf { A } ^ { 2 }$ .

# 4 Experiment

# 4.1 Experimental Setup

Refusal Steering Vectors Calculation. We use AdvBench (Zou et al. 2023b) as the harmful queries and TruthfulQA (Lin, Hilton, and Evans 2022) as the benign ones to generate the refusal steering vectors. Note that we just randomly sample 64 harmful questions and 64 harmless questions to extract the steering vectors as mentioned in Section 3.1. The remaining data is utilized for safety evaluation.

Evaluation Datasets. We select XSTest (Ro¨ttger et al. 2024) and OKTest (Shi et al. 2024) which are two prominent benchmarks focusing on the exaggerated safety phenomenon in LLMs. XSTest comprises 200 unsafe and 250 safe queries that well-calibrated models should not refuse. OKTest carefully designs 300 safe questions with harmful words to identify the over-refusal. We also include the remaining data from TruthfulQA as the test set for helpfulness.

Aside from mitigating the exaggerated safety, the security of LLMs should also be guaranteed. We use the following datasets to evaluate the security: (a) RepE-Data3 is a popular benchmark containing both harmful and harmless instructions. (b) The remaining AdvBench consists of 456 harmful behaviors. (c) Malicious (Huang et al. 2024) constructs 100 harmful questions covering ten diverse harmful intents.

We also evaluate whether SCANS would influence model capability. (a) multi-choice question answering task: we choose MMLU (Hendrycks et al. 2020) since it is comprehensive and challenging with extensive knowledge needed. (b) generation task: taking summarization as an example, we use XSum (Narayan, Cohen, and Lapata 2018) to evaluate the quality of generated summaries when using activation steering. Besides, we include two perplexity-based tasks, WikiText-2 (Merity et al. 2017) and C4 (Raffel et al. 2020).

Baselines. We compare SCANS with two training-free baselines: (1) Prompt (Bhalani and Ray 2024) is a prompting approach to identify and mitigate such exaggerated safety behaviors in LLMs. (2) Self-CD (Shi et al. 2024) applies contrastive decoding on the output probabilities to reduce the refusal rate on safe queries. We also evaluate SCANS against two training-required methods: (1) SafeDecoding (Xu et al. 2024) is a safety-aware decoding strategy based on the token probabilities of both the original and expert models. (2) DRO (Zheng et al. 2024) optimizes continuous safety prompts to improve safeguarding performance.

Table 2: Refusal rate on safety-related datasets, averaged across 5 trials. Refusal on safe datasets exhibits the exaggerated safety. Avg. $\ c =$ (#Compliance on Safe $^ +$ #Refusal on Unsafe) / #Total. Bold and underline indicate the best and the second best results. TQA stands for TruthfulQA benchmark. \* denotes our reproduced results.   

<html><body><table><tr><td rowspan="2">Models</td><td rowspan="2">Methods</td><td colspan="2">XSTest</td><td colspan="3">RepE-Data</td><td colspan="2">Helpfulness↓</td><td colspan="2">Harmfulness↑</td><td rowspan="2"></td><td rowspan="2">Avg.↑</td></tr><tr><td>Safe↓</td><td>UnSafe↑</td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.↑|Safe↓ UnSafe↑ Avg.↑|OKTest TQA|AdvBench Malicious</td><td></td></tr><tr><td rowspan="5">Llama2- 7b-chat</td><td>Default Prompt</td><td>58.00 36.40</td><td>100.0 100.0</td><td>67.77 79.77</td><td>12.50 2.86</td><td>100.0 99.48</td><td>93.75 98.31</td><td>53.67 41.66</td><td>5.05 15.27</td><td>100.0 99.34</td><td>100.0</td><td>86.13</td></tr><tr><td>Self-CD*</td><td>14.80</td><td>97.50</td><td>90.66</td><td></td><td></td><td></td><td></td><td></td><td></td><td>100.0</td><td>87.72</td></tr><tr><td></td><td></td><td></td><td></td><td>1.30</td><td>98.17</td><td>98.43</td><td>17.33</td><td>4.51</td><td>98.24</td><td>98.00</td><td>94.69</td></tr><tr><td>SafeDecoding</td><td>75.60</td><td>99.50</td><td>57.77</td><td>63.80</td><td>100.0</td><td>68.10</td><td>59.33</td><td>54.44</td><td>100.0</td><td>100.0</td><td>63.81</td></tr><tr><td>DRO SCANS</td><td>41.52</td><td>98.40</td><td>76.22</td><td>7.03</td><td>99.48</td><td>96.22</td><td>32.33</td><td>16.20</td><td>99.60</td><td>99.56</td><td>87.36</td></tr><tr><td rowspan="5">Llama2- 13b-chat</td><td>Default</td><td>9.20 34.40</td><td>93.50 99.50</td><td>92.00</td><td>0.00</td><td>99.22</td><td>99.61 97.14</td><td>0.33 20.33</td><td>0.80 11.69</td><td>99.34</td><td>100.0 100.0</td><td>98.26 90.83</td></tr><tr><td>Prompt</td><td>18.00</td><td>99.50</td><td>80.66</td><td>5.73</td><td>100.0</td><td></td><td></td><td></td><td>99.78</td><td></td><td></td></tr><tr><td>Self-CD*</td><td>29.60</td><td></td><td>89.77</td><td>0.78</td><td>99.22</td><td>99.22</td><td>30.33</td><td>12.62</td><td>99.34</td><td>100.0</td><td>91.47</td></tr><tr><td>DRO</td><td>38.00</td><td>100.0</td><td>83.55</td><td>4.68</td><td>100.0</td><td>97.66</td><td>19.33</td><td>4.91</td><td>98.24</td><td>100.0</td><td>93.10</td></tr><tr><td>SCANS</td><td>7.20</td><td>100.0 97.50</td><td>78.88 94.89</td><td>6.51 0.00</td><td>100.0 98.96</td><td>96.74 99.48</td><td>23.66 0.33</td><td>14.20 1.20</td><td>99.78 98.90</td><td>100.0 97.00</td><td>89.42 98.40</td></tr><tr><td rowspan="6">vicuna- 7b-v1.5</td><td>Default</td><td>20.80</td><td>88.00</td><td>83.11</td><td>4.69</td><td></td><td>96.36</td><td>19.00</td><td>5.05</td><td></td><td>76.00</td><td>91.68</td></tr><tr><td>Prompt</td><td>22.00</td><td>91.00</td><td>83.77</td><td>6.51</td><td>97.40</td><td></td><td>22.67</td><td></td><td>97.37 98.46</td><td></td><td></td></tr><tr><td>Self-CD*</td><td>10.00</td><td>83.00</td><td>86.88</td><td>3.64</td><td>98.44 89.58</td><td>95.97 92.97</td><td>27.00</td><td>11.33 9.56</td><td>89.03</td><td>82.00 56.00</td><td>90.01</td></tr><tr><td>SafeDecoding</td><td>55.20</td><td>99.50</td><td>69.11</td><td>33.29</td><td>100.0</td><td>83.35</td><td>61.00</td><td></td><td></td><td></td><td>87.26</td></tr><tr><td>DRO</td><td>22.11</td><td>95.80</td><td>85.85</td><td>3.38</td><td>99.74</td><td>98.18</td><td>13.33</td><td>39.70</td><td>100.0</td><td>98.00</td><td>73.41</td></tr><tr><td>SCANS</td><td>5.60</td><td>87.00</td><td>91.11</td><td>2.08</td><td>95.83</td><td>96.88</td><td>3.00</td><td>6.77 0.00</td><td>98.90 98.96</td><td>99.00 98.00</td><td>93.82 97.17</td></tr><tr><td rowspan="6">vicuna- 13b-v1.5</td><td>Default Prompt</td><td>16.80</td><td>98.00</td><td></td><td></td><td></td><td>97.66</td><td>19.33</td><td>4.38</td><td>99.78</td><td>93.00</td><td></td></tr><tr><td></td><td>20.80</td><td></td><td>89.77</td><td>3.65</td><td>98.96</td><td></td><td></td><td></td><td></td><td></td><td></td><td>94.23</td></tr><tr><td></td><td>8.40</td><td>99.00</td><td>88.00</td><td>10.68</td><td>99.74</td><td></td><td>94.53</td><td>27.00</td><td>19.33</td><td>99.34</td><td>97.00</td><td>88.37</td></tr><tr><td>Self-CD*</td><td></td><td>90.50</td><td>91.11</td><td>2.60</td><td>90.88</td><td></td><td>94.14</td><td>26.67</td><td>6.64</td><td>90.57</td><td>81.00</td><td>90.20</td></tr><tr><td>DRO</td><td>29.20</td><td>99.00</td><td>83.33</td><td>3.38</td><td>99.73</td><td></td><td>98.17</td><td>23.33</td><td>13.94</td><td>99.34</td><td>99.00</td><td>90.52</td></tr><tr><td>SCANS</td><td>9.20</td><td>93.50</td><td>92.00</td><td>2.08</td><td>97.66</td><td></td><td>97.79</td><td>3.33</td><td>0.27</td><td>99.78</td><td>98.00</td><td>97.59</td></tr></table></body></html>

Metrics. For safety and exaggerated safety, we use the Refusal Rate, the ratio of queries rejected by LLMs. We define the refusal behavior as the model outputs any of the predefined refusal messages following (Zheng et al. 2024). Considering the potential inaccuracies using string match, we also conduct human evaluations of the generated content and report the comparison results in Appendix C.

For generation tasks involving summarization, we use ROUGE-1/2/L as the accuracy measure, the higher the better. For multiple-choice QA, we assess the accuracy in four categories along with the final average score.

Implementation Details. Our experiments are primarily based on Llama2-7b-chat, Llama2-13b-chat, vicuna-7b-v1.5 and vicuna-13b-v1.5 (see Appendix D.3 for results on more models). All experimental results are averaged across 5 trials conducted on 1x80 GB A100 GPU. More hyperparameter settings and implementation details are in Appendix B.

# 4.2 Main Results

SCANS effectively achieves a balance between exaggerated safety mitigation and adequate safety. Table 2 reports the safety-related results of our SCANS compared with all baselines. As can be seen, aligned models like Llama2 Family models indeed improve the safety, while they also bring about a high refusal rate on word-level harmful but semantically benign queries. Similarly, trainingrequired methods DRO and SafeDecoding do not necessarily address exaggerated safety concerns. With our method, the average false refusal rate across all models has been proven to significantly decrease, outperforming all the baselines (in Appendix D.1). Specifically, SCANS decreases $2 4 . 7 \%$ and $2 6 . 3 \%$ of false refusal on safe queries from XSTest and OKTest on average.

Moreover, results on AdvBench and Malicious demonstrate that SCANS has almost no influence on the maintenance of adequate safety. In particular, when faced with two mixture benchmarks containing both safe questions and unsafe ones, XSTest and RepE-Data, we provide a comprehensive evaluation by calculating the overall ratio of correctly handling safe queries and refusing unsafe queries. The experimental results show SCANS can guarantee defense performance and mitigate exaggerated safety simultaneously.

SCANS does not compromise the general model capability greatly. In Table 3, we present perplexity, ROUGE1/2/L and multitask accuracy after applying SCANS to those aligned LLMs. Firstly, with activation steering, models still yield reasonable perplexity. In 13B models, SCANS increases perplexity by no more than 1 point on both WikiText-2 and C4, performing better than in 7B models. Secondly, for summarization tasks, the quality of generated content remains stable, with only about a $1 \%$ deviation, as measured by XSum. Moreover, the MMLU average degra

<html><body><table><tr><td rowspan="2">Models</td><td colspan="2">Perplexity↓</td><td rowspan="2">XSum↑</td><td colspan="2"></td><td colspan="5">MMLU↑</td></tr><tr><td>WikiText2 C4</td><td>R-1</td><td>R-2</td><td>R-L</td><td>STEM</td><td>Human</td><td>Social</td><td>Others</td><td>Avg.</td></tr><tr><td>Llama2-7b-chat +SCANS</td><td>7.76 9.32</td><td>9.86 11.94</td><td>21.38</td><td>4.923</td><td>17.45</td><td>37.60</td><td>43.40</td><td>55.10</td><td>54.10</td><td>47.20</td></tr><tr><td>Llama2-13b-chat +SCANS</td><td>6.86 7.29</td><td>8.89 9.45</td><td>20.07 22.22</td><td>3.912 5.280</td><td>16.47 17.48</td><td>34.00 43.80</td><td>36.20 49.50</td><td>47.40 62.50</td><td>46.20 60.00</td><td>40.50 53.60</td></tr><tr><td>vicuna-7b-v1.5</td><td>7.34</td><td>9.26</td><td>21.20 20.85</td><td>4.277 4.557</td><td>16.79 17.34</td><td>43.10 39.50</td><td>49.20 45.80</td><td>61.80 58.20</td><td>59.40 57.50</td><td>53.00 49.90</td></tr><tr><td>+SCANS vicuna-13b-v1.5 +SCANS</td><td>11.53 6.37 7.07</td><td>15.32 8.35 9.20</td><td>18.43 21.88 20.40</td><td>3.440 5.51 4.484</td><td>15.69 18.20 16.48</td><td>36.60 45.00 44.20</td><td>43.40 52.00 51.20</td><td>54.40 65.20 64.10</td><td>54.20 62.50 61.80</td><td>46.80 55.80 55.00</td></tr></table></body></html>

1.0 Precision Recall F1 score   
0.89 0.7   
0.6   
0.5   
0.4   
0.3   
0.2   
Moderation API Perspective API Llama Guard GradSafe GPT-4 SCANS- $\cdot \sigma ( q )$

Table 4: Inference speed and memory consumption of our SCANS applied to Llama2-7b-chat model.   

<html><body><table><tr><td></td><td>Inference Speed</td><td>GPU Memory</td></tr><tr><td>Llama2-7b-chat</td><td>40.60 tokens/s</td><td>29324MB</td></tr><tr><td>+SCANS</td><td>39.62 tokens/s</td><td>29694MB</td></tr></table></body></html>

dation in 13B models is within $1 \%$ after deploying SCANS, compared to within $5 \%$ in 7B models. These observations suggest that larger models are more robust to SCANS.

SCANS requires minor extra cost in inference time and GPU memory. Table 4 shows the inference speed and memory consumption of SCANS compared with the original Llama2-7b-chat model, tested under the test set of WikiText2 on a single A100 GPU. We can observe that the effect of our method on computational efficiency and inference speed is minor while some baselines like Self-CD and SafeDecoding require extra operation on token probabilities.

# 4.3 Analysis of $\sigma ( q )$

We further explore the classification accuracy of $\sigma ( q )$ which highly correlates with the performance of SCANS. We compare precision, recall, and F1 score with the following baselines: OpenAI’s Moderation API (Markov et al. 2023), Perspective API (Jigsaw. 2017), Llama Guard (Inan et al. 2023), GradSafe (Xie et al. 2024) and GPT-4 (Achiam et al. 2023).

As illustrated in Figure 4, our similarity-based classification method achieves the second highest F1 score, only inferior to GPT-4. For API tools, they are not effective enough to detect unsafe queries since they focus on reducing false positives. Conversely, LLMs as detectors usually have a higher recall than precision, indicating a tendency to misclassify safe queries as unsafe. Overall, $\sigma ( q )$ demonstrates comparable performance, further affirming that hidden states in LLMs are able to mine the harmfulness of input content. Detailed experimental data is provided in Appendix D.2.

# 4.4 Ablation Study

Effect of Steering Layers. It is important to achieve exaggerated safety mitigation and general capability preservation simultaneously. Therefore, the choice of steering layers is a crucial component in our approach. We explore how the performance of SCANS changes when refusal behavior vector steers at different layers. The experimental results are presented in Table 5. It shows that steering former layers brings significant perplexity increase which suggests a nonnegligible performance drop. While steering middle layers slightly underperforms steering latter layers in terms of perplexity, it is more effective in reducing the false refusal on safe queries, indicating the correlation between safety and middle layers.

Performance Under Different Multiplier $\alpha .$ . We conduct a sensitivity analysis to study the impacts of the multiplier $\alpha$ on refusal rate. From Table 6, we observe SCANS is not very sensitive to hyper-parameter $\alpha$ since the average perfor

<html><body><table><tr><td></td><td colspan="2">wiPirpxexity4</td><td rowspan="2"></td><td rowspan="2">SafeUsaetAvgOTsrutlABeMaiou</td><td rowspan="2"></td><td colspan="2"></td><td colspan="2"></td><td rowspan="2">Avg.↑</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Llama2-7b-chat</td></tr><tr><td>Former Layers</td><td>2946</td><td>3058</td><td></td><td></td><td></td><td>1</td><td></td><td></td><td></td><td></td></tr><tr><td>Middle Layers</td><td>9.32</td><td>11.94</td><td>9.20</td><td>93.50</td><td>92.00</td><td>0.33</td><td>0.80</td><td>99.34</td><td>100.0</td><td>97.76</td></tr><tr><td>Latter Layers</td><td>8.15</td><td>10.37</td><td>12.00</td><td>95.00</td><td>91.11</td><td>7.00</td><td>0.27</td><td>98.90</td><td>98.00</td><td>96.59</td></tr><tr><td colspan="10">vicuna-7b-v1.5</td></tr><tr><td>Former Layers</td><td>15433</td><td>11457</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Middle Layers</td><td>11.53</td><td>15.32</td><td>5.60</td><td>87.00</td><td>91.11</td><td>3.00</td><td>0.00</td><td>98.96</td><td>98.00</td><td>97.29</td></tr><tr><td>Latter Layers</td><td>7.85</td><td>9.89</td><td>7.60</td><td>83.50</td><td>88.44</td><td>2.33</td><td>1.46</td><td>93.42</td><td>92.00</td><td>94.75</td></tr></table></body></html>

Table 5: Performance of SCANS when refusal behavior vector steers at different layers. The calculation of Avg. metric is the same as Table 2. Since applying activation steering in former layers damages the model’s fluency and coherence (See examples in Appendix F.2), we do not report the refusal rate.

mance fluctuates slightly. However, we recommend setting $\alpha$ between 2 and 4 because too large a value sometimes results in nonsense outputs (See Appendix F.1).   

<html><body><table><tr><td>multiplier α</td><td>1.5</td><td>2.5</td><td>3.0</td><td>3.5</td><td>4.0</td></tr><tr><td>XSTest-Safe</td><td>9.60 10.40</td><td>10.80</td><td>10.80</td><td>9.20</td><td>10.40</td></tr><tr><td>XSTest-Unsafe</td><td>91.00 91.50</td><td>94.00</td><td>94.00 0.33</td><td>93.50 0.33</td><td>93.50 0.33</td></tr><tr><td>OKTest Malicious</td><td>7.00 3.33 100.0</td><td>1.00</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td></td><td>100.0</td><td>100.0</td><td></td><td></td><td></td></tr><tr><td>TruthfulQA</td><td>0.93 1.06</td><td>0.80</td><td>0.80</td><td>0.80</td><td>0.93</td></tr><tr><td>AdvBench</td><td>99.12 99.12</td><td>99.12</td><td>99.12</td><td>99.34</td><td>99.34</td></tr><tr><td>Avg.</td><td>96.41</td><td>96.85 97.47</td><td>97.57</td><td>97.76</td><td>97.57</td></tr></table></body></html>

Sensitivity to Threshold $\tau$ . We provide the impact of threshold $\tau$ on the SCANS performance in Table 7. As observed, when $\tau$ is below the optimal value, more safe queries are classified as unsafe and false refusal behavior increases. However, when $\tau$ exceeds the optimal level, the adequate safety may not be guaranteed. This is why we select $\mathcal { T } = 0 . 7 5$ for the above comparisons on Llama2-7b-chat. Detailed settings of threshold $\tau$ are given in Appendix B.2.

Table 6: Comparisons of different steering vector multiplier $\alpha$ conducted on Llama2-7b-chat model. The calculation of Avg. mertic is the same as Table 2.   
Table 7: Performance with different classification threshold $\tau$ on Llama2-7b-chat model. The calculation of Avg. mertic is the same as Table 2.   

<html><body><table><tr><td>threshold T</td><td>0.80 0.75</td><td>0.70 0.65</td><td>0.60</td></tr><tr><td>XSTest-Safe</td><td>3.60 9.20</td><td rowspan="3">33.20 46.00 65.20 100.0 33.67 50.00</td></tr><tr><td>XSTest-Unsafe</td><td>71.00 93.50 99.50 99.50</td></tr><tr><td>OKTest</td><td>0.0 0.33 8.33</td></tr><tr><td>Malicious</td><td>0.94 98.00 100.0</td><td rowspan="2">100.0 100.0 10.49 25.23</td></tr><tr><td>TruthfulQA</td><td>0.13 0.80 5.71</td></tr><tr><td>AdvBench</td><td>99.12 99.34 99.56</td><td>100.0 100.0</td></tr><tr><td>Avg.</td><td>96.21 97.67 92.52</td><td>85.62 75.57</td></tr></table></body></html>

Choice of Layers $\mathcal { L }$ for Classification. The selection of comparison layers is also a crucial component of steering direction identification, and further influencing the safetyconscious steering performance. As depicted in Figure 5, middle and latter layers demonstrate higher degree of distinction, indicating better identification accuracy for harmfulness, which is consistent with previous findings (Rimsky et al. 2024; Geva et al. 2022). Therefore, the motivation behind our classification method $\sigma ( q )$ is more intuitive. Please refer to Appendix B.2 for detailed experimental setting of $\mathcal { L }$ .

Cosine Similarity 1.0 Cosine Similarity 1.0   
0.8 0.8   
0.6 safe 0.6 safe unsafe unsafe 0.4   
0.4   
0 8 16 24 31 0 8 16 24 32 39 Layers Layers   
(a) Llama2-7b-chat (b) Llama2-13b-chat

# 5 Conclusion

In this paper, we propose SCANS, which mitigates the exaggerated safety for aligned LLMs via activation steering in safety-critical layers. Our motivation is based on that model hidden states imply the safety defense mechanism, indicating the refusal direction within the activation space. After extracting these refusal steering vectors, SCANS employs a similarity-based classification method to determine the steering direction and then steers the model behavior. Experimental results show SCANS effectively reduces the false refusal rate on safe prompts while not compromising the adequate safety and capabilities. We hope our work contributes to inspiring more researches on exaggerated safety issue through the lens of representation engineering.