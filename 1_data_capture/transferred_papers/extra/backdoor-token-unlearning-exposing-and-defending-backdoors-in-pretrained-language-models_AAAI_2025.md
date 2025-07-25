# Backdoor Token Unlearning: Exposing and Defending Backdoors in Pretrained Language Models

Peihai Jiang1, Xixiang Lyu1\*, Yige $\mathbf { L i } ^ { 2 * }$ , Jing Ma1

1Xidian University, China 2Singapore Management University, Singapore xdjph, jing.ma @stu.xidian.edu.cn, xxlv $@$ mail.xidian.edu.cn, yigeli@smu.edu.sg

# Abstract

Supervised fine-tuning has become the predominant method for adapting large pretrained models to downstream tasks. However, recent studies have revealed that these models are vulnerable to backdoor attacks, where even a small number of malicious samples can successfully embed backdoor triggers into the model. While most existing defense methods focus on post-training backdoor defense, efficiently defending against backdoor attacks during training phase remains largely unexplored. To address this gap, we propose a novel defense method called Backdoor Token Unlearning (BTU), which proactively detects and neutralizes trigger tokens during the training stage. Our work is based on two key findings: 1) backdoor learning causes distinctive differences between backdoor token parameters and clean token parameters in word embedding layers, and 2) the success of backdoor attacks heavily depends on backdoor token parameters. The BTU defense leverages these properties to identify aberrant embedding parameters and subsequently removes backdoor behaviors using a fine-grained unlearning technique. Extensive evaluations across three datasets and four types of backdoor attacks demonstrate that BTU effectively defends against these threats while preserving the model’s performance on primary tasks.

# Code — https://github.com/XDJPH/BTU

# Introduction

Pretrained Language Models (PLMs) (Devlin et al. 2018; Radford et al. 2019) have demonstrated remarkable performance across various tasks, such as sentiment analysis (Jim et al. 2024), toxicity detection (Bonetti et al. 2023), and news classification (Nkongolo Wa Nkongolo 2023). However, as PLMs are increasingly fine-tuned for specific downstream applications (Min et al. 2023), they have become vulnerable to backdoor attacks (Liu et al. 2024; Cheng et al. 2023). Typically, backdoor attacks inject malicious triggers into the model during training. The backdoored model functions normally on clean tasks but exhibits an attack-desired target label when the trigger is presented. In Natural Language Processing (NLP), backdoor triggers can be designed as obvious elements like rare words (Kurita, Michel, and Neubig

2020) or more subtle features such as sentence styles (Qi et al. 2021a). With the widespread adoption and deployment of PLMs, defending against backdoor threats has become an urgent challenge.

Existing backdoor defense methods in NLP generally fall into three categories: backdoor detection (Lyu et al. 2024; Liu et al. 2022; Xian et al. 2023), backdoor removal (Zhang et al. 2022; Li et al. 2021c), and anti-backdoor learning (Li et al. 2021b; Zhu et al. 2022). Backdoor model detection methods aim to identify whether a model or inputs contain backdoors, while backdoor removal methods focus on purifying the backdoor triggers from the backdoored model. Among them, anti-backdoor learning methods (Zhu et al. 2022; Li et al. 2021b) has become a widely adopted defense strategy as they allow the users to train a clean model even on a poisoned dataset. For example, ABL (Li et al. 2021b) employs a two-stage gradient ascent technique to filter out and mitigate backdoor behaviors. Another approach, MF (Zhu et al. 2022), limits the model’s learning capacity by restricting the number of training epochs, thereby preventing the model from acquiring backdoors during training. However, these anti-backdoor learning methods often lead to reduced model performance and exhibit instability across different scenarios. Therefore, how to effectively defend against backdoor attacks during the model training phase essentially deserves much attention.

Previous research has shown that backdoor learning can be viewed as a dual-task problem, i.e. training the backdoored model on both clean and backdoor data (Li et al. 2023). In this paper, we reformulate backdoor learning from model parameter perspective and identify two key properties: 1) backdoor learning induces significant differences between the embedding parameters of backdoor tokens and clean tokens, where the backdoor tokens converge much faster than clean ones; 2) the activation of backdoors is highly dependent on backdoor token parameters in the embedding layers. Intuitively, if we can isolate backdoor token parameters at the level of word embedding dimensions rather than across all model parameters, the backdoor information could be more effectively exposed and removed.

In this work, we propose a novel defense method called Backdoor Token Unlearning (BTU) for efficient antibackdoor learning. Specifically, BTU operates in two stages: backdoor token detection and dimensional fine-grained unlearning. In the first stage, BTU identifies potential backdoor tokens by exclusively training the word embedding layer and flagging the top $\alpha \%$ as backdoor-related embedding parameters. In the second stage, BTU removes backdoor information by replacing the affected backdoor embedding parameters with those of benign padding token embeddings. Through these two stages, BTU effectively defends against backdoor attacks while minimizing the impact on clean task performance. The main contributions of our work are summarized as follows:

• We identify two key observations in NLP backdoor attacks: 1) the distinctive differences in the embedding values of backdoor tokens and clean tokens when only the word embedding layer is trained, and 2) the success of backdoor activation is highly related to the backdoor token embedding parameters.   
• We introduce a novel defense method termed Backdoor Token Unlearning (BTU), which proactively exposes aberrant embedding parameters of backdoor tokens and mitigates backdoor behavior during the training process, with minimal impact on clean task performance.   
• Extensive experiments on four types of backdoor attacks across three datasets demonstrate that our proposed BTU substantially reduces the success rate of backdoor attacks while having minimal impact on the accuracy of downstream tasks.

# Related Work

# Backdoor Attack

Existing backdoor attacks in NLP manifest in two primary scenarios: outsourced training and data poisoning. In outsourced training, attackers have full control over the training process. For instance, the LWP (Li et al. 2021a) scheme implants backdoors in the model’s intermediate layers to increase the persistence of the attack, while the transfer (Shen et al. 2021) approach adjusts the backdoor optimization target in front of the MLP layer, using a multi-objective strategy to ensure the attack’s resilience against downstream task influences. Additionally, LWS (Qi et al. 2021c) employs an auxiliary model to create more concealed triggers. Conversely, in data poisoning scenarios, attackers are limited to inserting a few carefully crafted samples into the dataset since they do not control the training process. For example, Dai et al. (Dai, Chen, and Li 2019) demonstrate that words or context-independent phrases can serve as triggers, and that random insertion into training samples can successfully inject backdoors. Similarity, Qi et al. (Qi et al. 2021a,b) reveal that textual styles and syntactic structures can also act as triggers, significantly enhancing the stealthiness of backdoor attacks. These studies highlight the high vulnerability of NLP models to such covert manipulations and underscore the critical need for robust defense mechanisms.

# Backdoor Defense

In the field of NLP, existing backdoor defense methods can be broadly categorized into three types: 1) Backdoor input detection, which is applied during the model inference stage to identify and prevent the activation of backdoor inputs (Gao et al. 2021; Chen and Dai 2021; Yang et al. 2021a). For example, BKI (Chen and Dai 2021) distinguishes potential trigger words by analyzing each word’s impact on the model’s outcomes; 2) Backdoored model detection, which assesses whether a model contains backdoors (Liu et al. 2022; Azizi et al. 2021), often employing techniques like reverse engineering. For instance, PICCOLO attempts to recover potential triggers embedded within the model; 3) Anti-backdoor learning aims to train clean models from potentially poisoned datasets during the training phase (Li et al. 2021b; Zhu et al. 2022; Min et al. 2023). For instance, ABL (Li et al. 2021b) characterizes backdoor learning as a form of shortcut learning, where backdoor triggers are more easily captured. To address this, ABL proposed a two-stage gradient ascent technique to mitigate backdoor effects. Similarly, the MF defense (Zhu et al. 2022) introduced to minimize overfitting to prevent the model from learning backdoor patterns. Although promising, these methods often fail against adaptive attacks, such as textual style or grammatical structure triggers. In this work, we present new insights into backdoor learning and propose an simple yet efficient anti-backdoor defense to mitigate such threat.

# Proposed Token Unlearning Method

In this section, we first present the problem of backdoor attacks and then reveal the distinctive behavior between backdoor tokens and clean tokens optimized in the word embedding layers. Finally, we introduce our proposed BTU method.

Problem definition Consider the poisoned training dataset as $\mathcal { D } = \mathcal { D } _ { c } \cup \mathcal { D } _ { b }$ , where $\mathcal { D } _ { c }$ denotes the subset of clean data and $\mathcal { D } _ { b }$ denotes the subset of backdoor data. Training a backdoored model on a poisoned dataset can be viewed as minimizing the following empirical error:

$$
\begin{array} { r } { \mathcal { L } = \underbrace { \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { c } } [ \ell ( f _ { \theta } ( x ) , y ) ] } _ { \mathrm { c l e a n t a s k } } + \underbrace { \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { b } } [ \ell ( f _ { \theta } ( x ) , y ) ] } _ { \mathrm { b a c k d o o r t a s k } } , } \end{array}
$$

where $\ell$ and $\theta$ denote the loss function and model parameters, respectively. The overall learning task can be regarded as a combination of the backdoor task on dataset $\mathcal { D } _ { b }$ and the clean task on dataset $\mathcal { D } _ { c }$ .

Intuitively, if we can clearly distinguish between clean and backdoor tasks, the backdoor task can be more effectively detected. To achieve this, we reformulate the backdoor learning process in Eq. 1 to focus on the word embedding layer rather than all model parameters. As a result, the model’s optimization objective can be redefined as follows:

$$
\begin{array} { r } { \mathcal { L } = \underbrace { \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { c } } [ \ell ( \pmb { \varepsilon } ( x ) , y ) ] } _ { \mathrm { c l e a n t a s k } } + \underbrace { \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { b } } [ \ell ( \pmb { \varepsilon } ^ { b } ( x ) , y ) ] } _ { \mathrm { b a c k d o o r t a s k } } , } \end{array}
$$

where $\varepsilon$ denotes the entire clean embedding parameters and $\varepsilon ^ { b }$ denotes backdoor embedding parameters. Based on Eq. 2,

Add-Word Add-Sent Stylebkd Synbkd   
Euclidean Distance BTP Euclidean Distance BTP 0.012 BTP BTP 0.10 0.05 CTP CTP CTP CTP 0.05 0.00 0.00 0.000 0 250 0 250 0 250 0 250 Training Step Training Step Training Step Training Step (a) Parameter: Embedding Layer Add-Word Add-Sent Stylebkd Synbkd   
Euclidean Distance BTP 0.0150 BTP 0.012 BTP BTP 0.05 CTP CTP CTP CTP 0.00 0 250 0 250 0 250 0 250 Training Step Training Step Training Step Training Step (b) Parameter: All Layers

the backdoor information is primarily contained in the Backdoor Token Parameters $( B T P )$ , while the Clean Token Parameters $( C T P )$ remain largely unchanged. Since the backdoor task is much simpler than the clean task (Li et al. 2021b), we observe that the cumulative parameter changes in BTP occur more rapidly than in CTP. We will provide empirical evidence to support this observation in the following subsection.

# Revealing Distinctive Behavior of Backdoor Tokens

In this subsection, we aim to highlight the distinct learning behavior between BTP and CTP when trained on word embedding layers.

We conduct four backdoor attack methods: AddWord (Gu, Dolan-Gavitt, and Garg 2017), Add-Sent (Dai, Chen, and Li 2019), Stylebkd (Qi et al. 2021a), and Synbkd (Qi et al. 2021b), to poison the SST-2 dataset (Socher et al. 2013) with a $10 \%$ poisoning rate. We then train a BERT (Devlin et al. 2018) model using standard procedures and settings from the public library (Cui et al. 2022). For each attack, we trained two backdoored models: one on all parameters and another only on the word embedding layers. To compare the learning differences, we record the variations in Euclidean distance between BTP and CTP.

Fig. 1 shows that, across all four types of attacks, the mean Euclidean distance of the BTP is greater than that in the CTP. For example, in Add-Word attack, when training only the word embedding layer, BTP is almost 0.1 higher than CTP. However, when training all parameters, BTP is only 0.01 higher than CTP. The difference in the magnitude of change between the two cases is nearly tenfold. This distinction between BTP and CTP suggests that backdoor information is primarily associated with BTPs and inspires our defense strategy.

# Backdoor Token Unlearning

Overview Fig. 2 illustrates the BTU framework, which consists of two main components: Backdoor Token Detection and Dimensional Fine-grained Unlearning. The backdoor token detection aims to identify suspicious backdoor tokens within the embedding parameters through three rounds of anomaly detection. Once these malicious tokens are detected, fine-grained dimensional unlearning is applied to remove backdoor functionalities from these token parameters. We provide detailed technical explanations below.

Backdoor Token Detection As previously noted, we have identified a distinctive Euclidean distance between BTP and CTP. Building on this, we can detect suspicious backdoor token parameters through iterative detection rounds $T$ . The detection threshold is set to $\alpha \in [ 0 , 1 ]$ , with the top $\alpha \%$ of embedding parameters flagged as backdoor token parameters in each detection round. For simplicity, we set $\alpha$ to 0.05 across all three detection rounds. A more detailed analysis of $\alpha$ and the detection round $T$ will be provided in the ablation

$\textcircled{1}$ Backdoor Token Detection $\textcircled{2}$ Dimensional Fine-gained Unlearning Backdoored model Classifier Frozen Classifier 𝐸ଵ 𝐸ଶ 𝐸ଷ Encoder 𝐸ଵଶଽଷହ 𝐸|𝒱| Sαel%e c×t t|o𝒱p| 𝐸ଵ௕ 𝐸ଶ௕ 𝐸௕ Encoder 𝐸ଵ௕ଶଽଷହ ： 𝐸|௕𝒱| Word Embedding 𝜀 token by 𝑠௜ Backdoor Word Embedding  𝜀௕ ...n.eA𝐸e௕dgrtoucpflheed𝐸l.௕p... 𝐸ଷ௕ 𝐸ଵ௕ଶଽଷହ 𝐸௕ 𝑠௜ = 𝑑(𝐸௜, 𝐸௜ ) 𝑇ଵ 𝑣ଵ 𝑣ଵ௣ 𝑣ଵ௣ 𝑑̅ = 𝑑(|𝜀𝒱௕,| 𝜀) Word Embedding 𝐸ଵ௕ଶଽଷହ ： ：· ： Round 1 Anomaly  Token Detection 𝐸௣ WorCdlEasmsibfeiedrding Frozen 𝑣௝ 𝑣௝ 𝑣௝௣ Paedmdbinegddtionkgen 𝑣௘௦ Round 2 Anomaly  Token Detection 𝑇ଶ 𝐸ଵ௖ଶଽଷହ௜ ൝𝑣௝௣, 𝑖𝑓 ∆𝑣௜  ≥ 𝑑̅ Remove 𝑇ଶ Classifier Frozen 𝐸ଵ௕ 𝐸ଶ௕ Fixed Word Embedding 𝐸ଷ௕ ： 𝐸ଵ௖ଶଽଷହ 𝐸|௕𝒱| ...need to cf help.. Word Embedding Classifier .A group led .. → Encoder Anomaly  Token Detection 𝑇ଷ Round 3 Embedding Repair 𝑹𝒆𝒕𝒖𝒓𝒏 $\pmb { T } = \pmb { T } _ { 1 } \cup \pmb { T } _ { 2 } \cup \pmb { T } _ { 3 }$ Clean model Clean Backdoor Clean embedding Backdoor embedding token embedding token embedding dimension value dimension value ------ - -------

study.

In the first round, we train only the embedding layer parameters $\varepsilon$ of model $M$ on the dataset $\mathcal { D }$ , resulting in the updated embedding layer parameters $\varepsilon ^ { \prime }$ . We then calculate the change distance $s ^ { \prime }$ for each token $t _ { i }$ and store the tokendistance pairs in the set $T _ { 1 }$ :

$$
T _ { 1 } = \{ ( s _ { i } ^ { \prime } , t _ { i } ) \} _ { t _ { i } \in \mathcal { V } } = \{ ( d ( \varepsilon ( t _ { i } ) , \varepsilon ^ { \prime } ( t _ { i } ) ) , t _ { i } ) \} _ { t _ { i } \in \mathcal { V } }
$$

Next, we rank $s _ { i } ^ { \prime }$ in descending order and select the top $\alpha \% \times$ $| \nu |$ tokens from $T _ { 1 }$ , which we denote as $T ^ { \prime }$ .

In the second round, we retain the embedding layer and classification head of model $M$ , denoted as $M ^ { * }$ , and train the embedding layer $\varepsilon$ of $M ^ { * }$ to obtain $\varepsilon ^ { \prime \prime }$ . After training on the dataset $\mathcal { D }$ , we calculate the change distance $s ^ { \prime \prime }$ for each token and store the token-distance pairs in the set $T _ { 2 }$ :

$$
T _ { 2 } = \{ ( s _ { i } ^ { \prime \prime } , t _ { i } ) \} _ { t _ { i } \in \mathcal { V } } = \{ ( d ( \varepsilon ( t _ { i } ) , \varepsilon ^ { \prime \prime } ( t _ { i } ) ) , t _ { i } ) \} _ { t _ { i } \in \mathcal { V } }
$$

We then rank $s _ { i } ^ { \prime \prime }$ in descending order and select the top $\alpha \% \times$ $| \nu |$ tokens from $T _ { 2 }$ , denoted as $T ^ { \prime \prime }$ .

In the third round, we repeat the previous procedure, but modify the dataset to $\mathcal { D } / T ^ { \prime \prime }$ . All other settings remain the same, leading to:

$$
T _ { 3 } = \{ ( s _ { i } ^ { \prime \prime \prime } , t _ { i } ) \} _ { t _ { i } \in \mathcal { V } } = \{ ( d ( \varepsilon ( t _ { i } ) , \varepsilon ^ { \prime \prime \prime } ( t _ { i } ) ) , t _ { i } ) \} _ { t _ { i } \in \mathcal { V } }
$$

Finally, we rank $s _ { i } ^ { \prime \prime \prime }$ in descending order and select the top $\alpha \% \times \lvert \nu \rvert$ tokens from $T _ { 3 }$ , denoted as $T ^ { \prime \prime \prime }$ .

We define $T = T ^ { \prime } \cup T ^ { \prime \prime } \cup T ^ { \prime \prime \prime }$ as the set of suspicious tokens. Notably, the three rounds of anomaly detection serve different purposes. Rounds 1 and 2 aim to detect simple triggers, while Round 3 refines the process to detect more complex triggers. This three-step iterative detection ensures comprehensive identification of suspicious backdoor tokens, effectively exposing both simple and complex triggers at varying levels of granularity. The analysis of results for different detection rounds can be found in the ablation study.

Dimensional Fine-gained Unlearning Given a backdoored model $M ^ { b }$ and a set of suspicious tokens $T$ , the most straightforward method is to replace all tokens in $T$ with padding tokens that carry no information, thereby removing all backdoor-related token parameters. However, simple replacement would eliminate both backdoor and clean features within the word embedding parameters, leading to a decrease in model accuracy.

To maximally retain clean features in the word embedding parameters, we propose a Dimensional Fine-grained Unlearning technique, which allows selectively replace only the dimensions with large changes in BTP while remaining others unchanged. Specifically, we first calculate the mean change in the word embedding layer before and after training:

$$
\bar { d } = \sum _ { t _ { i } \in \mathcal { V } } ( d ( \varepsilon ( t _ { i } ) , \varepsilon ^ { \prime } ( t _ { i } ) ) ) / | \mathcal { V } | ,
$$

where $\varepsilon$ represents the parameters of the word embedding before training, and $\varepsilon ^ { \prime }$ represents the parameters after training.

For all $t \in T$ , the dimensions in $\varepsilon ^ { \prime } ( t )$ with values greater than $\bar { d }$ are replaced by the corresponding dimension values of $\varepsilon ^ { \prime } ( p )$ , where $p$ denotes the padding token. Thus, the suspicious parameters in embedding layers $\varepsilon ^ { c } ( t )$ are replaced by:

$$
\varepsilon _ { i } ^ { c } ( t ) = \left\{ \begin{array} { l l } { \varepsilon _ { i } ^ { \prime } ( t ) , } & { \mathrm { i f ~ } | \varepsilon _ { i } ^ { \prime } ( t ) - \varepsilon _ { i } ( t ) | < \bar { d } ; } \\ { \varepsilon _ { i } ^ { \prime } ( p ) , } & { \mathrm { i f ~ } | \varepsilon _ { i } ^ { \prime } ( t ) - \varepsilon _ { i } ( t ) | \geq \bar { d } . } \end{array} \right.
$$

Finally, the values in $\varepsilon ^ { \prime } ( t )$ are replaced with $\varepsilon ^ { c } ( t )$ . As we replace only a small number of tokens and the word embedding layer contains relatively little downstream information, the impact of our token unlearning causes minimal degradation in clean performance. To further mitigate the negative effect, we fine-tune the model with a small amount of clean data after padding token replacement.

# Experiment

# Experimental Setting

Datasets and Models We conducted experiments using three text classification datasets: 1) SST-2 (Stanford Sentiment Treebank-2) (Socher et al. 2013), a binary sentiment analysis dataset; 2) OLID (Offensive Language Identification Dataset) (Zampieri et al. 2019), a binary toxicity detection dataset; and 3) AG News, a four-class news headline classification dataset. The victim model used is BERTBASE-UNCASED, which consists of 12 layers with 30522 $\times 7 6 8$ parameters in the word embedding layer.

Attack Setups Four data poisoning-based attack methods are employed: 1) Add-Word, using rare words as triggers (e.g., “cf”, “tq”, and “bb”); 2) Add-Sent, using common phrases as triggers (e.g., “I watched a 3D movie”); 3) Stylebkd, using text styles as triggers (e.g., “Bible style”); and 4) Synbkd, using syntactic structures as triggers (e.g., “(ROOT (S (SBAR) (, ) (NP) (VP) (.)))”). The poisoned samples for Stylebkd and Synbkd are generated using the public library from Cui et al. (Cui et al. 2022).

Defense Setups We compared BTU with nine other methods, including six training-phase defenses (BKI (Chen and Dai 2021), MF (Zhu et al. 2022), CUBE (Min et al. 2023), TG (Pei et al. 2023), ST (Tang et al. 2023), and DPOE (Liu et al. 2023)) and three inference-phase defenses (ONION, RAP, and Strip), which were adapted into training-phase defenses using the public library (Cui et al. 2022) under standard settings. For BTU, we removed special tokens from the results of backdoor token detection to refine the evaluation.

Evaluation Metrics Defense methods are evaluated using the metric ACC (Accuracy), which measures the model’s ability to correctly classify clean data, and the metric ASR (Attack Success Rate), which measures the effectiveness of the backdoor attack in causing misclassification.

# Experimental Results

As shown in Table 1, BTU significantly reduces the success rate of four types of backdoor attacks across three datasets. Specifically, for insertion-based attacks (Add-Word and Add-Sent), BTU reduces the ASR to below $10 \%$ across all three datasets. Additionally, it is observed that the more complex the dataset, the more effective BTU becomes. Across all datasets, we find that the ACC of the Add-Sent attack is higher than that of the Add-Word attack. This is because BTU detects more clean tokens in the Add-Word attack, resulting in the loss of more clean features.

For unfixed type triggers in Stylebkd and Synbkd, BTU successfully mitigate the influence of backdoor attacks, demonstrating that these backdoor attack activations still depend on specific tokens. This phenomenon can also be observed in the poisoned samples, where conjunctions such as ”when” and ”if” are frequently involved. Additionally, we find that Stylebkd negatively affects the model’s performance; however, BTU can effectively restore the damage caused by this attack.

![](images/1f9ceba546f3282d471206984e1c4732169442cf4ffe28b8e2a55428580e11ed.jpg)  
Figure 3: Token quantities influence the results. ”clean” refers to not modifying the labels after insertion, $" 1 0 "$ represents an insertion ratio of $10 \%$ , and ”average” indicates the mean of the changes in the word embedding layer.

To explore the generalizability of the BTU method, we conducted experiments on GPT2, RoBERTa and LLaMA2- 7B (Touvron et al. 2023) with SST-2 dasaset under $10 \%$ poison ratio. As summarized in Table 2, BTU significantly lowers the ASR on all models while maintaining high ACC.

Overall, the results demonstrate that BTU is effective in defending against a variety of known backdoor attacks across different attack scenarios, with minimal impact on the model’s performance on clean tasks. This consistent performances across datasets and attack types highlights BTU’s potential as a reliable defense mechanism in real-world applications, where maintaining accuracy while ensuring security is paramount.

# Defense Results against Adaptive Attacks

In this section, we consider the countermeasures an attacker might take when aware that the defender is using BTU. The core of BTU is to capture and purify backdoor tokens based on the simplicity of backdoor tasks. However, when backdoor tasks become more complex, backdoor tokens may evade detection, leading to potential defense failure. Therefore, adaptive attacks could be executed by narrowing the learning difficulty gap between backdoor and clean tasks. We will explore these potential adaptive attacks in the following discussion.

Low Poison Ratio In fact, a low poison ratio is more reflective of real-world scenarios. However, we found that most existing defense methods perform poorly against low poison ratio attacks. At the same time, a low poison ratio makes it more challenging for the model to learn the backdoor task. So we employ an experiment to test BUT’s performance under low poison ratio backdoor attack. We use the lowest possible poison ratio $( 0 . 7 \% )$ to perform Add-Sent attack on the SST-2 dataset, achieving an ASR of over $90 \%$ . Then, we conducted training defense methods including RAP, Strip, BKI, ONION, CUBE, MF and BTU to evaluate their performances under low poison ratio backdoor attacks. As shown in Table 3, All defense methods failed to defend against low poison ratio backdoor attacks, except for BTU. This phenomenon shows that BTP exhibits a statistically significant change compared to CTP, even at a low poison ratio. This can be observed in Table 3. It demonstrates that BTU possesses exceptionally strong backdoor defense capabilities.

<html><body><table><tr><td rowspan="2">Dataset</td><td rowspan="2">Defense</td><td colspan="2">Add-Word</td><td colspan="2">Add-Sent</td><td colspan="2">Stylebkd</td><td colspan="2">Synbkd</td></tr><tr><td>ACC</td><td>ASR</td><td>ACC</td><td>ASR</td><td>ACC</td><td>ASR</td><td>ACC</td><td>ASR</td></tr><tr><td rowspan="10">SST-2</td><td>None</td><td>91.05</td><td>100.00</td><td>91.10</td><td>100.00</td><td>90.37</td><td>54.72</td><td>90.72</td><td>90.46</td></tr><tr><td>ONION</td><td>87.08</td><td>21.18</td><td>86.78</td><td>71.38</td><td>84.32</td><td>60.15</td><td>85.27</td><td>91.33</td></tr><tr><td>RAP</td><td>91.82</td><td>100.00</td><td>90.88</td><td>99.89</td><td>87.34</td><td>56.80</td><td>87.70</td><td>94.74</td></tr><tr><td>STRIP</td><td>91.05</td><td>100.00</td><td>90.88</td><td>99.89</td><td>87.34</td><td>56.80</td><td>90.22</td><td>86.51</td></tr><tr><td>BKI</td><td>87.12</td><td>25.43</td><td>91.21</td><td>97.48</td><td>89.76</td><td>57.49</td><td>88.96</td><td>93.64</td></tr><tr><td>CUBE</td><td>87.70</td><td>15.68</td><td>88.14</td><td>30.81</td><td>90.88</td><td>20.50</td><td>90.94</td><td>28.18</td></tr><tr><td>MF</td><td>90.05</td><td>16.59</td><td>91.05</td><td>90.89</td><td>90.48</td><td>58.37</td><td>90.71</td><td>48.60</td></tr><tr><td>ST</td><td>90.35</td><td>19.03</td><td>90.73</td><td>22.55</td><td>89.01</td><td>19.03</td><td>86.26</td><td>43.71</td></tr><tr><td>TG</td><td>88.37</td><td>19.45</td><td>88.19</td><td>20.91</td><td>89.09</td><td>27.98</td><td>89.22</td><td>37.93</td></tr><tr><td>DPOE</td><td>88.30</td><td>19.63</td><td>90.33</td><td>50.54</td><td>89.01</td><td>17.37</td><td>89.89</td><td>36.99</td></tr><tr><td rowspan="10">OLID</td><td>BTU (ours)</td><td>90.37</td><td>5.97</td><td>90.69</td><td>5.50</td><td>90.38</td><td>6.79</td><td>90.59</td><td>24.36</td></tr><tr><td>None</td><td>79.51</td><td>100.00</td><td>79.68</td><td>100.00</td><td>76.03</td><td>52.33</td><td>79.67</td><td>97.61</td></tr><tr><td>ONION</td><td>78.23</td><td>10.46</td><td>77.55</td><td>100.00</td><td>66.59</td><td>71.69</td><td>72.91</td><td>97.86</td></tr><tr><td>RAP</td><td>79.51</td><td>100.00</td><td>62.06</td><td>0.11</td><td>76.04</td><td>52.33</td><td>77.42</td><td>97.45</td></tr><tr><td>STRIP</td><td>79.52</td><td>100.00</td><td>75.36</td><td>94.40</td><td>76.04</td><td>52.33</td><td>79.00</td><td>93.78</td></tr><tr><td>BKI</td><td>75.13</td><td>25.26</td><td>79.51</td><td>100.00</td><td>69.76</td><td>70.65</td><td>70.87</td><td>94.58</td></tr><tr><td>CUBE</td><td>77.47</td><td>18.66</td><td>80.01</td><td>16.26</td><td>77.02</td><td>25.71</td><td>79.81</td><td>21.07</td></tr><tr><td>MF</td><td>79.19</td><td>19.71</td><td>79.13</td><td>81.66</td><td>75.99</td><td>43.78</td><td>79.55</td><td>56.90</td></tr><tr><td>ST</td><td>77.68</td><td>22.70</td><td>79.13</td><td>21.79</td><td>78.02</td><td>33.27</td><td>79.43</td><td>42.71</td></tr><tr><td>TG</td><td>77.54</td><td>13.80</td><td>77.76</td><td>15.35</td><td>76.04</td><td>26.32</td><td>78.06</td><td>38.07</td></tr><tr><td>DPOE</td><td>76.78</td><td>98.83</td><td>55.61</td><td>95.51</td><td>50.07</td><td>50.02</td><td>50.10</td><td>48.07</td></tr><tr><td rowspan="10">AG News</td><td>BTU (ours)</td><td>78.93</td><td>4.04</td><td>79.12</td><td>5.17</td><td>79.32</td><td>5.33</td><td>80.24</td><td>12.77</td></tr><tr><td>None</td><td>94.47</td><td>100.00</td><td>94.46</td><td>100.00</td><td>93.39</td><td>73.72</td><td>94.02</td><td>100.00</td></tr><tr><td>ONION</td><td>92.91</td><td>2.05</td><td>93.02</td><td>77.63</td><td>90.39</td><td>76.91</td><td>93.11</td><td>96.11</td></tr><tr><td>RAP</td><td>94.26 94.33</td><td>84.82</td><td>94.46</td><td>100.00</td><td>93.11</td><td>67.58</td><td>93.49</td><td>79.98</td></tr><tr><td>STRIP BKI</td><td>94.11</td><td>99.98</td><td>94.25</td><td>100.00</td><td>93.33</td><td>74.02</td><td>93.41</td><td>79.81</td></tr><tr><td></td><td></td><td>96.15</td><td>94.15</td><td>100.00</td><td>93.00</td><td>76.15</td><td>93.27</td><td>82.67</td></tr><tr><td>CUBE</td><td>87.04</td><td>3.97</td><td>88.14</td><td>2.71</td><td>91.98</td><td>2.53</td><td>90.46</td><td>3.87</td></tr><tr><td>MF</td><td>94.31</td><td>17.79</td><td>94.13</td><td>89.37</td><td>92.97</td><td>66.34</td><td>93.71</td><td>68.77</td></tr><tr><td>ST</td><td>93.94</td><td>20.10</td><td>93.56</td><td>18.37</td><td>93.07</td><td>33.74</td><td>93.47</td><td>45.47</td></tr><tr><td>TG DPOE</td><td>91.08 94.84</td><td>2.39 1.63</td><td>91.20 93.99</td><td>1.90</td><td>90.47</td><td>11.79</td><td>92.68</td><td>40.83</td></tr><tr><td>BTU (ours)</td><td></td><td>94.35 0.83</td><td></td><td>94.33</td><td>5.26</td><td>93.15</td><td>15.45</td><td>93.89</td><td>55.48 37.39</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>1.58</td><td>93.90</td><td>11.47</td><td>93.58</td><td></td></tr></table></body></html>

Table 1: The attack success rate $( \mathrm { A S R \% } )$ and the accuracy $( \mathrm { A C C \% } )$ of our BTU and other 9 different defense methods against 4 backdoor attacks. None means without defense.

Table 2: The performance of BTU under different model architectures. The experiments are conducted on SST-2 dataset with $10 \%$ poisoning rate.   

<html><body><table><tr><td>Model</td><td>Attack</td><td>ACC</td><td>ASR</td></tr><tr><td>RoBERTa</td><td>Add-Sent Synbkd</td><td>94.01 91.04</td><td>19.71 23.55</td></tr><tr><td>GPT2</td><td>Add-Sent Synbkd</td><td>90.51 89.87</td><td>25.78 29.98</td></tr><tr><td>LLaMA2</td><td>Add-Sent Synbkd</td><td>91.76 93.28</td><td>5.17 22.09</td></tr></table></body></html>

Trigger Complexity To increase trigger complexity, we adopt a method from the SOS (Yang et al. 2021b) framework to perform adversarial training on a subset of triggers, thereby extending their length and complexity. This compels the model to learn the entire trigger sequence, heightening the learning challenge. As shown in Table 3, BTU effectively counters these enhanced backdoor attacks by identifying and neutralizing key trigger components during its exposure phase. These results demonstrate that BTU maintains strong defensive efficacy under varied and intensified backdoor conditions, effectively mitigating threats while adapting to increased task complexities.

# Ablation Study

Token Quantities To evaluate whether the number of trigger tokens significantly affects BTP changes, we conducted an experiment with two models: one with random trigger insertions without label changes (clean) and another implementing a backdoor attack with both trigger insertions and label targeting (dirty). Each model incorporated triggers into $10 \%$ of the dataset. We compared the changes in BTP after training the two models to assess the impact of token quantity. As shown in Fig. 3, in backdoor attacks, BTP changes remain nearly unaffected by the number of tokens and are significantly higher than CTP with the same token count. These results demonstrate that our method robustly defends against backdoor attacks across varying poisoning rates.

Table 3: Defense results of BTU against adaptive attack with low poison ratios and complex triggers.   

<html><body><table><tr><td>Method</td><td>ASR</td><td>ACC</td></tr><tr><td>None</td><td>92.37</td><td>91.06</td></tr><tr><td>BKI</td><td>95.29</td><td>90.74</td></tr><tr><td>RAP</td><td>90.57</td><td>79.62</td></tr><tr><td>STRIP</td><td>95.72</td><td>90.72</td></tr><tr><td>ONION</td><td>97.15</td><td>91.37</td></tr><tr><td>CUBE</td><td>89.15</td><td>91.11</td></tr><tr><td>MF</td><td>38.27</td><td>91.05</td></tr><tr><td>BTU (ours)</td><td>7.36</td><td>90.39</td></tr><tr><td>Trigger Complexity</td><td>7.18</td><td>90.71</td></tr></table></body></html>

Table 4: Defense Results of BTU under different detection threshold, i.e. $\alpha$ .   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="2">Add-Sent</td><td colspan="2">Synbkd</td></tr><tr><td>ACC</td><td>ASR</td><td>ACC</td><td>ASR</td></tr><tr><td>None</td><td>91.03</td><td>100.00</td><td>90.75</td><td>91.52</td></tr><tr><td>α=0.03</td><td>91.05</td><td>5.91</td><td>90.56</td><td>37.91</td></tr><tr><td>α=0.05</td><td>90.96</td><td>4.84</td><td>90.17</td><td>24.77</td></tr><tr><td>α =0.10</td><td>87.56</td><td>4.87</td><td>89.73</td><td>15.96</td></tr></table></body></html>

Detection Threshold $\alpha$ To investigate the impact of the detection strength $\alpha$ on BTU, we conducted experiments on the SST-2 dataset with a $10 \%$ poison ratio. Table 4 illustrates that adjusting the threshold value $\alpha$ plays a pivotal role in the efficacy of the BTU method. Increasing $\alpha$ enhances defense effectiveness but reduces model accuracy (ACC), while decreasing $\alpha$ preserves ACC but raises the attack success rate (ASR). Our findings suggest that setting $\alpha$ to 0.05 strikes an effective balance in defending against backdoor attacks across most scenarios.

Alternative Unlearning To further explore Token Unlearning, we tested three approaches: Parameter Replacement-1 (PR-1): Following the insights from (Zhang et al. 2022), we replaced the BTP of the backdoored model with those from a pre-trained language model; Parameter Noise (PN): Gaussian noise was added to the BTP to disrupt their backdoor characteristics; Parameter Replacement-2 (PR-2): We replaced the BTP of the backdoored model with those of the padding tokens. To further disrupt the BTP, we clipped dimensions in the BTP that showed significant changes, setting the clipping threshold to the mean change value of the CTP. We conducted experiments on the SST-2 dataset using the add-sent and synbkd attack methods with a $10 \%$ poisoning rate. The results, detailed in Table 5, show that BTU

<html><body><table><tr><td rowspan="2">Method</td><td colspan="2">Add-Sent</td><td colspan="2">Synbkd</td></tr><tr><td>ACC</td><td>ASR</td><td>ACC</td><td>ASR</td></tr><tr><td>None</td><td>91.03</td><td>100.00</td><td>90.48</td><td>86.89</td></tr><tr><td>BTU-PN</td><td>88.86</td><td>13.82</td><td>90.01</td><td>26.00</td></tr><tr><td>BTU-PR-1</td><td>90.87</td><td>99.88</td><td>90.59</td><td>43.56</td></tr><tr><td>BTU-PR-2</td><td>90.24</td><td>4.91</td><td>89.70</td><td>29.71</td></tr><tr><td>BTU (ours)</td><td>90.67</td><td>5.50</td><td>90.47</td><td>24.91</td></tr></table></body></html>

Table 5: Compared with more token unlearning methods   
Table 6: Results for different anomaly detection rounds   

<html><body><table><tr><td rowspan="2">AnomalyRound</td><td colspan="2">Add-Word</td><td colspan="2">Synbkd</td></tr><tr><td>ACC</td><td>ASR</td><td>ACC</td><td>ASR</td></tr><tr><td>None</td><td>91.06</td><td>100.0</td><td>90.72</td><td>90.48</td></tr><tr><td>1</td><td>90.55</td><td>17.21</td><td>90.63</td><td>29.89</td></tr><tr><td>2</td><td>90.60</td><td>7.81</td><td>90.60</td><td>37.49</td></tr><tr><td>2+3</td><td>90.57</td><td>5.47</td><td>90.61</td><td>35.00</td></tr><tr><td>1+2</td><td>90.35</td><td>7.70</td><td>90.57</td><td>27.70</td></tr><tr><td>1+1+2+3</td><td>89.36</td><td>5.97</td><td>90.59</td><td>24.03</td></tr></table></body></html>

outperforms other strategies.

Anomaly Detection Rounds To assess the importance of each detection round, we conducted an ablation study on the SST-2 dataset, with results shown in Table 6. We observe that the first detection round is more effective at mitigating complex backdoors, while the second and third rounds are better suited for countering simple backdoors. Increasing the number of detection rounds can reduce the success rate of backdoor attacks, though it may slightly impact accuracy. Our findings indicate that three detection rounds offer the optimal balance between maintaining accuracy and ensuring defense effectiveness.

# Conclusion

In this work, we identified two key properties in the context of NLP backdoor learning: 1) the distinctive differences in the embedding values of backdoor tokens and clean tokens when only the word embedding layer is trained, and 2) the success of backdoor activation is highly related to the backdoor token parameters. Based on these observations, we propose a novel anti-backdoor learning method Backdoor Trigger Unlearning (BTU), which proactively exposes aberrant embedding parameters of backdoor tokens and mitigates backdoor behaviors during the training process. Extensive experimental results demonstrate that BTU can effectively defend against currently known backdoor attacks with minimal impact on the performance of clean tasks.

Future Work While BTU effectively defends against four different backdoor attacks and outperforms nine other defense methods, we cannot guarantee its effectiveness against more advanced future attacks. Further exploration is needed to provide theoretical guarantees for BTU’s underlying mechanisms. Additionally, our current findings and defense results are based on evaluations with pre-trained language models, so it remains an open question whether BTU is effective for more advanced large language models.