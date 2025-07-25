# MoE-LPR: Multilingual Extension of Large Language Models through Mixture-of-Experts with Language Priors Routing

Hao Zhou1\*, Zhijun Wang1\*, Shujian Huang1†, Xin Huang2, Xue Han2, Junlan Feng2, Chao Deng2, Weihua $\mathbf { L u o } ^ { 3 }$ , Jiajun Chen1

1National Key Laboratory for Novel Software Technology, Nanjing University, China 2China Mobile Research Beijing, China; 3Alibaba International Digital Commerce,China {zhouh,wangzj}@smail.nju.edu.cn,{huangsj,chenjj}@nju.edu.cn, {huangxinyjy,hanxueai,fengjunlan,dengchao}@chinamobile.com, weihua.luowh@alibaba-inc.com

# Abstract

Large Language Models (LLMs) are often English-centric due to the disproportionate distribution of languages in their pre-training data. Enhancing non-English language capabilities through post-pretraining often results in catastrophic forgetting of the ability of original languages. Previous methods either achieve good expansion with severe forgetting or slight forgetting with poor expansion, indicating the challenge of balancing language expansion while preventing forgetting. In this paper, we propose a method called MoE-LPR (Mixture-of-Experts with Language Priors Routing) to alleviate this problem. MoE-LPR employs a two-stage training approach to enhance the multilingual capability. First, the model is post-pretrained into a Mixture-of-Experts (MoE) architecture by upcycling, where all the original parameters are frozen and new experts are added. In this stage, we focus improving the ability on expanded languages, without using any original language data. Then, the model reviews the knowledge of the original languages with replay data amounting to less than $1 \%$ of post-pretraining, where we incorporate language priors routing to better recover the abilities of the original languages. Evaluations on multiple benchmarks show that MoE-LPR outperforms other postpretraining methods. Freezing original parameters preserves original language knowledge while adding new experts preserves the learning ability. Reviewing with LPR enables effective utilization of multilingual knowledge within the parameters. Additionally, the MoE architecture maintains the same inference overhead while increasing total model parameters. Extensive experiments demonstrate MoE-LPR’s effectiveness in improving expanded languages and preserving original language proficiency with superior scalability.

# Introduction

Large Language Models (LLMs) such as ChatGPT (OpenAI 2023), GPT-4 (Achiam et al. 2023), Llama2 (Touvron et al. 2023), Llama3 (Dubey et al. 2024), and Qwen (Bai et al. 2023) have demonstrated remarkable performance across different tasks, including multiple-choice question-answering (Robinson and Wingate 2023), summarization (Pu, Gao, and Wan 2023), and reasoning (Yu et al. 2023). However, many studies have highlighted a significant discrepancy between performances on English and nonEnglish tasks (Gao et al. 2024; Wang et al. 2024).

![](images/41901a4c5bcfcd450c91498f79b938a50d59a1607719fc3db3119abfe63b8221.jpg)  
Figure 1: MoE-LPR performs the best in both expanded languages and original languages. We define expanded languages as languages that the model is not very good at and we are going to enhance, and original languages as languages that the model is relatively strong in and prone to catastrophic forgetting.

Pre-training a LLM with data from multiple languages may achieve better multilingual capabilities, but highly resource-intensive and often impractical given limited computational budgets. Consequently, current research predominantly focus on post-pretraining (also known as continue training) techniques (Csaki et al. 2024; Kuulmets et al. 2024), which carry out further multilingual pre-training on a pre-trained LLM, aiming to inject extensive language knowledge for certain language(s). Despite its efficiency, this method significantly increases the risk of catastrophic forgetting, where the performance of LLMs in the languages they are initially good at (such as English or Chinese) may dramatically decline. As a result, improving the performance of expanded languages while maintaining the performance of existing ones becomes a critical challenge in the field.

To prevent forgetting, existing work (Dou et al. 2024; Wu et al. 2024) usually retain the original parameters of the model as much as possible, and train new parameters to fit knowledge for new languages. However, less attention is paid on effectively incorporating these new and old parameters for tasks in different languages. In this paper, we propose a novel two-stage training method called Mixtureof-Experts with Language Priors Routing (MoE-LPR) that improves multilingual capability with the retention of original language proficiency. MoE-LPR contains two stages: post-pretraining with MoE and review with LPR.

In the post-pretraining stage, we upcycle the LLM into a MoE architecture and post-pretrain the newly added parameters with a substantial amount of high-quality monolingual data, while keeping the original parameters frozen. This ensures that the original capabilities of the model are preserved while expanding its proficiency in additional languages. We also incorporate load balancing loss to unleash the model’s learning potential and maintain training stability. In the review stage, we further train the router to better utilize the experts for different languages. We design LPR training to recover the model’s capabilities in its original languages using replay data that amounts to less than $1 \%$ of the postpretraining corpus.

As shown in Figure 1, experiment results demonstrate that our method not only significantly improves proficiency in newly expanded languages (languages in the top half) but also substantially retains the model’s capabilities in its original languages (languages in the bottom half). Moreover, our approach allows for easy upscaling for the number of model parameters while maintaining a fixed inference overhead. Our approach represents a step forward in developing LLMs that are both powerful and versatile across a wide range of languages, addressing the critical need for more inclusive and effective NLP technologies in a multilingual world. The contributions of our proposed method are as follows:

• Two-Stage Training Strategy: MoE-LPR employs a twostage training strategy, with a special focus on balancing the capability of newly expanded languages and the original languages. • Language Priors Routing: MoE-LPR introduces the LPR mechanism to mitigate catastrophic forgetting of original languages with replay data amounting to less than $1 \%$ of the post-pretraining corpus. LPR also exhibits excellent generalization to languages it has not been trained on. • Scalability: MoE-LPR allows for easy upscaling of model parameters without increasing inference overhead or risking catastrophic forgetting, making it a costeffective and stable solution for multilingual NLP tasks.

# Methodology

Figure 2 describes the overall framework of our MoE-LPR. In the post-pretraining with MoE stage, we train the new experts on a large amount of monolingual data in the expanded languages for injecting language knowledge. In the review with LPR stage, we train the router on a small amount of monolingual data in both the expanded and original languages for better utilizing the experts.

# Post-pretraining with MoE

As shown in Figure 2, inspired by Mixtral (Jiang et al. 2024) and upcycling (Komatsuzaki et al. 2022), we upcycle the dense model to a MoE model by copying the FFN parameters and incorporating a router matrix $\breve { W _ { r } } \in \mathbb { R } ^ { h \times N }$ in each layer, where $h$ represents the token dimension and $N$ denotes the number of experts within the model.

The router in MoE allows the model to dynamically select the most suitable experts. Formally, let $\boldsymbol { x } \in \mathbb { R } ^ { h }$ be a token representation, the router score is expressed as:

$$
G \left( x \right) = { \mathrm { S o f t m a x } } \left( x \cdot W _ { r } \right)
$$

where $G \left( x \right) \in \mathbb { R } ^ { N }$ . After obtaining router scores, We select the index set $\tau$ of the top- $K$ experts and combine their outputs using normalized weights from the router scores to obtain the final representation as:

$$
{ \mathcal { T } } = \{ i | G _ { i } \left( x \right) \in \mathrm { T o p k } { \left( G \left( x \right) , K \right) } \}
$$

$$
y = \sum _ { i \in \mathcal { T } } \frac { G _ { i } ( x ) } { \sum _ { j \in \mathcal { T } } G _ { j } ( x ) } E _ { i } ( x ) + x
$$

where $G _ { i } ( x )$ and $E _ { i } ( x )$ represent the router score and the output of the $i$ -th expert respectively, and $K$ denotes the number of activated experts.

To enhance the multilingual capability of the MoE model while preserving its performance in the original languages, we freeze the parameters of the original dense model. During post-pretraining on the expanded language corpus, we only update the parameters of the newly added experts and the router, which ensures that the core knowledge embedded in the initial model remains intact.

The model is trained with a combination of a next token prediction loss and a load balancing loss as follows.

Next Token Prediction Loss. Given an expanded language corpus $D$ , a batch $\boldsymbol { B }$ with $T$ tokens, and $N$ experts indexed by $i$ from 0 to $N - 1$ , where index 0 is used to denote the original dense FFN, the post-pretraining next token prediction loss is:

$$
L _ { \mathrm { N T P } } ( \theta _ { \mathrm { n e w } } , W _ { r } ) = - \sum _ { i = 1 } ^ { | \mathcal { B } | } \sum _ { j = 1 } ^ { \left| d ^ { i } \right| } \log p _ { \mathcal { M } } \left( d _ { j } ^ { i } \mid d _ { < j } ^ { i } \right)
$$

where $\mathcal { M }$ denotes the whole MoE model, $\theta _ { \mathrm { n e w } }$ indicates the parameters of the newly added experts and $W _ { r }$ is the parameter of the router.

Load Balancing Loss. We also use an expert-level load balance loss (Fedus, Zoph, and Shazeer 2022) to mitigate the risk of routing collapse:

$$
L _ { \mathrm { b a l a n c e } } ( \theta _ { \mathrm { n e w } } , W _ { r } ) = \sum _ { i = 1 } ^ { N } f _ { i } P _ { i }
$$

![](images/84cf46438a991ed7f19d34556a3d1f3e6b3f8f2ac39a5fb32f5a42a75fe960d0.jpg)  
Figure 2: Overall framework of our MoE-LPR. Two-stage strategy is performed to enhance the multilingual capability

$$
\begin{array} { c } { { f _ { i } = \displaystyle \frac { N } { K T } \sum _ { t \in \mathcal { B } } \mathbb { 1 } \{ \mathrm { T o k e n ~ t ~ s e l e c t s ~ e x p e r t ~ i } \} } } \\ { { P _ { i } = \displaystyle \frac { 1 } { T } \sum _ { t \in \mathcal { B } } G _ { i } ( t ) } } \end{array}
$$

where $\mathbb { 1 }$ denotes the indicator function. We opt for a top-2 strategy by setting $K = 2$ to select the two most suitable experts with normalization, intending to achieve a trade-off between inference overhead and learning capabilities.

The final optimization objective during post-pretraining is:

$$
\begin{array} { r } { \mathrm { a r g m i n } { L _ { \mathrm { N T P } } } + \alpha { L _ { \mathrm { b a l a n c e } } } } \\ { \theta _ { \mathrm { n e w } } , W _ { r } \quad } \end{array}
$$

where $\alpha$ is a hyper-parameter that controls the weight of the load balancing loss.

# Review with LPR

After post-pretraining on the expanded language corpus, the router, which has only been trained on the expanded languages but not on the original languages, may incorrectly assign experts for the original languages. This misallocation is also an important factor for catastrophic forgetting in the MoE model. Therefore, we design this review stage to train the model to deal with both original and expanded languages.

As the router is the main source of the problem, we only update the parameters of the router and freeze the other parts of the model. Because the number of router parameters accounts for a negligible proportion, this stage could be efficient and requires very little computational resource and training data.

In fact, the amount of original language data used in our review stage, is less than $1 \%$ of the post-pretraining corpus. In comparison, traditional replay strategy (Ibrahim et al. 2024) incorporates data from original languages into the post-pretraining stage, which usually requires a much larger amount $( 2 5 \% )$ .

LPR Loss. Intuitively, the routing could be led by language priors: all the original language tokens should be routed to the originally frozen expert (i.e. expert 0 in this case), making the model work exactly the same as before the expansion. Therefore, we design a LPR loss to be a Cross-Entropy loss for the tokens from the original languages, forcing the top-1 selection of these tokens to be expert 0, where the top1 selection refers to the expert selection with the highest routing score.

Formally, considering original language tokens set $D _ { \mathrm { o r i g i n a l } }$ and the indicator function $\mathbf { F } ( t )$ :

$$
\mathbf { F } ( t ) = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { i f ~ } } t \in D _ { \mathrm { o r i g i n a l } } , } \\ { 0 } & { { \mathrm { i f ~ } } t \not \in D _ { \mathrm { o r i g i n a l } } . } \end{array} \right. }
$$

The LPR loss is defined as:

$$
L _ { \mathrm { L P R } } ( W _ { r } ) = - \sum _ { t \in \mathcal { B } } \mathbf { F } ( t ) \log G _ { 0 } ( t )
$$

where index 0 denotes the originally frozen expert.

In practice, when training with LPR loss, we remove the load balancing loss in Eq. (8). The final optimization objective for the review stage is:

$$
\underset { W _ { r } } { \mathrm { a r g m i n } } L _ { \mathrm { N T P } } + \gamma L _ { \mathrm { L P R } }
$$

where $\gamma$ is a hyper-parameter that controls the weight of the LPR loss.

# Experiments

# Experiment Setup

Given the focus on multilingual capability enhancement, we introduce the language selection first. Then follow the training details, several baselines, and the evaluation details.

Model and Languages We choose Qwen-1.5 1 as our base model. The 1.8B version of the Qwen-1.5 series is selected for its lower computation overhead and ease of upcycling. For our study, we choose three low-resource languages as the expanded languages where Qwen-1.5-1.8B performs poorly as shown in Figure 1: Greek (El), Hungarian $\left( \mathrm { H u } \right)$ , and Turkish $( \mathrm { T r } )$ . Additionally, we select three high-resource languages as the original languages to observe the catastrophic forgetting phenomenon: English (En), Chinese (Zh), and Spanish (Es).

Details of Post-pretraining We construct a dataset focusing on the three expanded languages by sampling 8 billion tokens from the monolingual data of each language in CulturalX (Nguyen et al. 2024), a substantial multilingual dataset with 6.3 trillion tokens in 167 languages. Our base model, Qwen-1.5-1.8B, is upcycled into the MoE structure with 5 newly added FFN (6 experts in total). We postpretrain this model with the 24 billion tokens, marking only the new experts and the router as trainable. The training setup includes a batch size of 512, a sequence length of 1024, a learning rate of 5e-5, and a cosine learning rate scheduler. We incorporate the load balancing loss with a weight of 0.01 and utilize bf16 mixed precision and flash attention (Dao et al. 2022) to speed up the training process.

Our experiments are conducted on 8 A800 GPUs, involving 45856 steps, totaling approximately 848 A800 GPU hours.

Details of Review We randomly sample 50K documents for each original language and 100K documents for each expanded language. The English data are sampled from Slimpajama (Soboleva et al. 2023), the Chinese data from SkyPile-150B (Wei et al. 2023), and the Spanish data from CulturalX (Nguyen et al. 2024). The number of tokens in original languages is 0.138B, accounting for less than $1 \%$ of the post-pretraining data (24B). As for the three expanded languages, we sample from the post-pretraining dataset. We concatenate these data for the review stage training. We employ a batch size of 512, a sequence length of 512, a learning rate of 5e-5, and a cosine learning rate scheduler. The load balancing loss is removed and the LPR loss is added as introduced in Eq. (10) with a weight of 0.1. Only the router parameters are trainable. Bf16 mixed precision and flash attention (Dao et al. 2022) mechanism is used for training.

Baselines We conducted experiments on several existing baseline methods trained on the same data, including the small amount of replay data, to ensure that our approach is competitive and effective.

• Full Fine-tuning: Fine-tune all parameters directly on the dense model. • LoRA (Hu et al. 2021): The LoRA targets include all linear modules. We set the LoRA rank to 8. • MoE: The same settings as MoE-LPR except for training all the parameters only in one post-pretraining stage. • LoRAMoE (Dou et al. 2024): A novel framework combines multiple LoRAs with a router network to effectively learn new knowledge while avoiding catastrophic forgetting. The router selects all LoRAs for each token. We set the number of LoRAs as 8 and a LoRA rank of 180 to match the same inference overhead. • LLaMA-Pro (Wu et al. 2024): A method is considered where a dense LLM periodically duplicates and inserts new transformer blocks at fixed layer intervals. During post-pretraining, only these newly added transformer blocks are trained to acquire new knowledge while preserving the original knowledge. We add 12 new layers because this is the best setting in our experiments.

Evaluation Details We evaluate our method on several benchmarks including multiple-choice tasks and generation tasks. Examining the model’s multilingual capabilities from multiple perspectives.

• ARC-Challenge (25-shot) (Clark et al. 2018): A benchmark for evaluating comprehension and reasoning across diverse academic fields.   
• MMLU (5-shot) (Hendrycks et al. 2020): A multiplechoice dataset testing general knowledge and problemsolving across various subjects.   
• HellaSwag (10-shot) (Zellers et al. 2019): A dataset with 70k questions for studying grounded commonsense inference.   
• Belebele (5-shot) (Bandarkar et al. 2023): A machine reading comprehension dataset covering 122 language variants.   
• FLORES-101 (8-shot) (Goyal et al. 2022): A parallel corpus for evaluating multilingual translation capabilities. We report the performance evaluated by COMET (Rei et al. 2022) 2

We mainly follow Okapi (Lai et al. 2023) to evaluate the multilingual versions of ARC-Challenge, MMLU and HellaSwag, which are translated from the original English version using GPT-3.5-turbo or DeepL.

# Experiment Results

Table 1 presents the performance of various methods across different benchmarks for both expanded and original languages. We report here the performance of the best setting of all baselines. With the additional small amount of replay data, full fine-tuning outperforms LoRA in preventing catastrophic forgetting but still drops about 4 points in original languages. Full fine-tuning can recover to $9 2 . 1 \%$ performance in original languages with replay data amounting to less than $1 \%$ of the post-pretraining data. Ibrahim et al. (2024) demonstrates that training new languages suffers from dramatic distribution shifts. Only when using more than $2 5 \%$ replay data can the model recover to more than $9 5 . 7 \%$ performance, indicating that significant language shifts in post-pretraining data require more replay data and computational overhead. However, our MoE-LPR can recover to $9 6 . 6 \%$ performance (52.12/53.97) with less than $1 \%$ replay data.

Table 1: Evaluation results in expanded and original languages. $n _ { \mathrm { p a r a m s } }$ is the total number of model parameters, ${ \mathbf { \mathit { n } } } _ { \mathbf { \mathit { a c t } } }$ -params is the number of activated model parameters per token. The best and second-best results are marked in bold and underlined fonts.   

<html><body><table><tr><td>Model</td><td>nparams</td><td>nact-params</td><td>ARC</td><td>MMLU</td><td>HellaSwag</td><td>Belebele</td><td>Flores</td><td>Avg.</td></tr><tr><td colspan="9">Expanded Languages</td></tr><tr><td>Qwen1.5-1.8B</td><td>1.8B</td><td>1.8B</td><td>23.13</td><td>30.97</td><td>29.15</td><td>33.15</td><td>55.40</td><td>34.36</td></tr><tr><td>LoRA (Hu et al. 2021)</td><td>1.8B</td><td>1.8B</td><td>23.89</td><td>29.30</td><td>29.78</td><td>26.93</td><td>55.19</td><td>33.02</td></tr><tr><td>Full Fine-tuning</td><td>1.8B</td><td>1.8B</td><td>25.98</td><td>33.18</td><td>35.28</td><td>33.70</td><td>77.48</td><td>41.12</td></tr><tr><td>LLaMA-Pro (Wu et al. 2024)</td><td>2.4B</td><td>2.4B</td><td>24.35</td><td>34.02</td><td>33.85</td><td>31.52</td><td>81.76</td><td>41.10</td></tr><tr><td>MoE</td><td>5.8B</td><td>2.6B</td><td>26.43</td><td>35.07</td><td>37.01</td><td>32.74</td><td>80.01</td><td>42.25</td></tr><tr><td>LoRAMoE (Dou et al. 2024)</td><td>2.6B</td><td>2.6B</td><td>26.63</td><td>34.17</td><td>37.17</td><td>32.81</td><td>81.09</td><td>42.37</td></tr><tr><td>MoE-LPR</td><td>5.8B</td><td>2.6B</td><td>28.43</td><td>34.10</td><td>41.06</td><td>39.93</td><td>81.83</td><td>45.07</td></tr><tr><td colspan="9">Original Languages</td></tr><tr><td>Qwen1.5-1.8B</td><td>1.8B</td><td>1.8B</td><td>33.48</td><td>47.55</td><td>49.82</td><td>56.52</td><td>82.50</td><td>53.97</td></tr><tr><td>LoRA (Hu etal. 2021)</td><td>1.8B</td><td>1.8B</td><td>28.33</td><td>37.42</td><td>41.48</td><td>39.45</td><td>75.49</td><td>44.43</td></tr><tr><td>Full Fine-tuning</td><td>1.8B</td><td>1.8B</td><td>31.72</td><td>43.51</td><td>47.38</td><td>45.26</td><td>80.77</td><td>49.73</td></tr><tr><td>LLaMA-Pro (Wu et al. 2024)</td><td>2.4B</td><td>2.4B</td><td>31.77</td><td>44.06</td><td>48.36</td><td>48.78</td><td>81.97</td><td>50.99</td></tr><tr><td>MoE</td><td>5.8B</td><td>2.6B</td><td>32.51</td><td>44.16</td><td>48.54</td><td>45.37</td><td>81.63</td><td>50.44</td></tr><tr><td>LoRAMoE (Dou et al. 2024)</td><td>2.6B</td><td>2.6B</td><td>32.43</td><td>45.41</td><td>48.61</td><td>47.74</td><td>82.03</td><td>51.24</td></tr><tr><td>MoE-LPR</td><td>5.8B</td><td>2.6B</td><td>32.71</td><td>44.62</td><td>49.12</td><td>51.81</td><td>82.36</td><td>52.12</td></tr></table></body></html>

LoRA performs poorly in expanded languages due to the excessive data in the post-pretraining stage. We also experiment with LoRA at rank $_ { = 6 4 }$ to achieve comparable effects in expanded languages, but this results in worse catastrophic forgetting.

LLaMA-Pro demonstrates a strong ability to retain knowledge, but its performance in expanded languages is only comparable to full fine-tuning, with the drawback of higher inference overhead. LoRAMoE performs better than other baselines in both expanded and original languages. Our proposed method, MoE-LPR, surpasses LoRAMoE by 2.7 points in expanded languages and by 0.88 points in original languages on average. While adding more new parameters, the inference overhead of LLaMA-Pro and LoRAMoE increases accordingly, while that of MoE-LPR does not. More details about scaling will be discussed in the following sections.

The results also demonstrate that MoE underperforms our MoE-LPR both in expanded and original languages, which implies that freezing all the original parameters will not limit the model’s learning ability. In contrast, the frozen parameters contribute a robust basic capabilities of the model during post-pretraining, resulting in significant performance improvement.

Ablation & Analysis Review with LPR   
Table 2: Evaluation average results with different settings. “w/o EC” means without expert-copy, corresponding to randomly initialize the new experts when upcycling.   

<html><body><table><tr><td>Model</td><td>Expanded Original</td><td>Avg.</td></tr><tr><td>Qwen1.5-1.8B</td><td>34.36 53.97</td><td>44.17</td></tr><tr><td>LoRAMoE</td><td>42.37</td><td>51.24 46.81</td></tr><tr><td>MoE-LPRw/o EC</td><td>38.37 49.28</td><td>43.83</td></tr><tr><td>MoE-LPRw/oReview</td><td>45.04</td><td>47.14 46.09</td></tr><tr><td>MoE-LPRw/o LPR</td><td>45.13</td><td>51.32 48.23</td></tr><tr><td>MoE-LPR</td><td>45.07</td><td>52.12 48.60</td></tr></table></body></html>

Performance Gain from Review & EC The review with LPR stage is proposed to recover the capabilities of the original languages. As shown in Table 2, without the review stage, MoE-LPR exhibits severe catastrophic forgetting. However, after review training, the performance in original languages improves substantially, by about 5 points on average, while not harming the performance in expanded languages. Furthermore, the performance in original languages drops without the LPR loss, indicating that the LPR mechanism pushes this ability closer to its upper bound. These results show that the review stage allows the model to learn how to handle both new and old languages.

We also conduct experiment without the Expert-Copy, which means that the parameters of new experts are randomly initialized but not copied from the original FFN. As shown in Table 2, performance in original languages does not suffer a serious decrease, but performance in expanded languages shows a significant decrease. Results imply that copying the original FFN to construct new experts is important to the learning of expanded language knowledge.

![](images/5434605125391bb785b3c7d2904a49ada1f9f69488c9aec365d1acbe138a833e.jpg)  
Figure 3: Router scores of the frozen expert for English (original language) tokens in the Belebele benchmark.

Routing Scheme for Different Languages In this section, we examine whether the review stage works properly. As shown in Figure 3, the router scores of the frozen expert on original language tokens show obvious improvement with the review stage. In addition, without the LPR loss, the router scores demonstrate a significant drop. The router scores of the frozen expert on expanded language tokens almost remain unchanged. In the review stage, we optimize the model with only the next token prediction loss for expanded languages. The results show that the next token prediction loss effectively prevents expanded languages from being influenced by the language priors of original languages. These observations indicate that the review stage is functioning correctly, biasing the routing scheme of original language tokens toward the frozen expert.

![](images/a284b0ee1a27bf0e9995cc9c7dd0335bf0dda1551a1ac61937b7de22ee82fc31.jpg)  
Figure 4: Average scores in expanded and original languages with varying numbers of documents for review.

How much Data is Enough for Review In this section, we experiment with varying numbers of original language documents in the review stage, ranging from 0K to 150K, while maintaining the $1 { : } 2 \mathrm { m i x }$ of original and expanded languages. As shown in Figure 4, the original language performance continues to improve significantly while the expanded language performance continues to decrease slightly. After $5 0 \mathrm { k }$ , the original language performance improvement starts to become slow. Therefore, considering both training cost and effects, we choose 50K as the best data size in this experiment, which amounts to less than $1 \%$ of the postpretraining corpus. Using 50K results in a 4.98 points performance boost in the original languages while almost maintaining the performance in the expanded languages. These results indicate that a small amount of replay data is sufficient for the model to review its original languages.

![](images/1fe3f29c4616113f10ee53c23f4c9216973de260ddb2c58b3512a47f4778f5b5.jpg)  
Figure 5: Average scores in expanded and original languages with different model settings. $^ { 6 6 } 3 4 . 3 6 ^ { , 9 }$ and $\mathbf { \cdots } 5 3 . 9 7 \mathbf { \cdots }$ refer to the expanded and original language performance of the base model respectively.

# Scaling Law

We compare the performance of LLaMA-Pro with different numbers of extending layers, LoRAMoE with different ranks and MoE-LPR with different numbers of experts. All the models are trained on the 24 billion tokens dataset in the three expanded languages.

Figure 5 demonstrates the superior scalability of MoELPR. For expanded languages, adding 12 layers to LLaMAPro improves performance more than adding 6 layers, but adding 24 layers, matching the base model’s layer count, results in a performance drop. Increasing the rank of LoRAMoE from 32 to 180 shows significant improvements. MoE-LPR consistently outperforms these configurations as more experts are added, even with just 2 experts, maintaining a significant advantage over LLaMA-Pro and LoRAMoE. For original languages, LLaMA-Pro suffers from catastrophic forgetting, worsening with more layers. Adding 24 layers even performs worse than full fine-tuning. Although LoRAMoE’s catastrophic forgetting does not worsen with increased parameters, it still underperforms MoE-LPR. Even with 8 experts and a 7B parameter size, MoE-LPR can still greatly mitigate catastrophic forgetting.

Unlike LLaMA-Pro and LoRAMoE, whose activated parameters per token increase linearly with more parameters, adding experts to MoE-LPR does not increase the inference overhead. This improves performance in expanded languages while maintaining stable levels of catastrophic forgetting. MoE-LPR demonstrates superior scalability.

# Language Generalization

In the review stage, we only use documents of three of the original languages. We conduct evaluations on two additional high-resource languages that the base model is good at relatively: French and Portuguese to examine the generalization of MoE-LPR when preventing catastrophic forgetting. We name them out-of-domain original languages because the review stage training does not contain tokens in these two languages. Table 3 demonstrates that MoE-LPR successfully generalizes its catastrophic forgetting prevention effect to these languages. Despite the router not being trained on French and Portuguese tokens, our LPR mechanism minimizes the performance gap from the base model for these languages, outperforming other post-pretraining methods. This demonstrates MoE-LPR’s excellent language generalization in preventing catastrophic forgetting.

Table 3: Evaluation results in French and Portuguese.   

<html><body><table><tr><td>Model</td><td>Exp.</td><td>Ori. ID</td><td>Ori. OOD</td></tr><tr><td>Qwen1.5-1.8B</td><td>34.36</td><td>53.97</td><td>46.35</td></tr><tr><td>Full Fine-tuning</td><td>41.12</td><td>49.73</td><td>42.46</td></tr><tr><td>LLaMA-Pro</td><td>41.10</td><td>50.99</td><td>42.93</td></tr><tr><td>LoRAMoE</td><td>42.37</td><td>51.24</td><td>43.41</td></tr><tr><td>MoE-LPRw/oLPR</td><td>45.13</td><td>51.32</td><td>44.22</td></tr><tr><td>MoE-LPR One-Stage</td><td>45.38</td><td>51.90</td><td>43.71</td></tr><tr><td>MoE-LPR</td><td>45.07</td><td>52.12</td><td>45.25</td></tr></table></body></html>

We also try to move the LPR loss and the small amount of replay data to the post-pretraining stage. As shown in Table 3, MoE-LPR One-Stage shows comparable performance to the two-stage strategy. However, it demonstrates worse language generalization, which showcases a 1.54 points performance drop in the out-of-domain original languages. Therefore, we choose the two-stage strategy as a better proposal.

# Related Work

# Mixture of Experts

Recent studies (Kaplan et al. 2020; Hoffmann et al. 2022) have shown a strong correlation between the number of parameters in a model and its capabilities. When the number of parameters is large, the model demonstrates emergent abilities (Zoph et al. 2022b). Traditional dense models require the activation of all parameters for a given input, significantly increasing computational overhead. Distinct from conventional dense models, Mixture of Experts (MoE) achieves computational feasibility and expanded model capacity by utilizing a router that selectively activates a limited number of experts for each input. There are several works, such as Switch-transformer (Fedus, Zoph, and Shazeer 2022), ST-MoE (Zoph et al. 2022a), Glam (Du et al. 2022) , attempts to train an MoE model from scratch. These works have demonstrated that MoE models can achieve significantly lower loss and performance gains compared to dense models with the same activated parameters and require less energy consumption compared to dense models with the same total parameters. However, considering the huge computational budget, Komatsuzaki et al. (2022) indicates that a sparse MoE model could be initialized from dense models. In the era of LLMs, numerous MoE works have been developed. For instance, Mixtral (Mixtral 2024) adds experts to each layer, increasing the total parameter count to 141B. DeepSeek (DeepSeek-AI 2024) utilizes shared experts, enabling the model to select experts more effectively. Snowflake Arctic (Research 2024)incorporates many fine-grained experts, enhancing the diversity of expert selection. Chen et al. (2023b); Dou et al. (2024); Zadouri et al. (2023) combines MoE with LoRA, resulting in more effective training and alleviating data conflict issues.

The most relevant work to us is Lifelong-MoE (Chen et al. 2023a), which effectively expands the number of experts during lifelong learning and introduces a regularization to avoid catastrophic forgetting. However, we employ a different freezing method and a two-stage training framework, significantly alleviating catastrophic forgetting and gaining a promising performance in expanded languages.

# LLM for Multilingual

Post-pretraining on a massive multilingual corpus is an effective way to improve the multilingual abilities of LLMs. Alves et al. (2024) and $\mathrm { { X u } }$ et al. (2024) highlight monolingual data’s importance in post-pretraining. Notably, Xu et al. (2024) demonstrates that with fixed computational resources, allocating more to monolingual data rather than translation data better improves a model’s translation performance, allowing large models to achieve translation abilities comparable to traditional supervised models NLLB (Costajuss\`a et al. 2022). Blevins et al. (2024) have explored using the Branch Then Merge (BTM;Gururangan et al. (2023)), where separate models are trained independently for different languages and then merged, partially overcoming the challenges of the multilingual curse (Wu and Dredze 2020). Geng et al. (2024) employs the LoRA (Hu et al. 2021) architecture to help migrate a chat LLM to the target language while preserving its chat capabilities.

# Conclusion

In this paper, we propose MoE-LPR, a scalable postpretraining method that effectively expands languages and prevents catastrophic forgetting using the Mixture-ofExperts architecture. Expanding new languages often encounters severe catastrophic forgetting due to significant distribution changes, and the challenge lies in balancing old and new languages. Through two-stage training, MoE-LPR addresses this with efficient parameter assignment and balanced routing. The post-pretraining stage enables the model to have a strong enough learning ability and steadily enhances the capabilities of the expanded languages. The review stage brings a performance boost to the original languages without harming the performance in expanded languages. Our two-stage training achieves both expansion and prevention of forgetting effects well. Additionally, MoELPR shows better scalability and generalization than SOTA methods. Overall, MoE-LPR is an effective and scalable approach for expanding new languages during the postpretraining stage.