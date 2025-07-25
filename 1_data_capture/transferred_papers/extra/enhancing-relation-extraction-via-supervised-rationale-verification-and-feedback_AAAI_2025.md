# Enhancing Relation Extraction via Supervised Rationale Verification and Feedback

Yongqi $\mathbf { L i } ^ { 1 }$ , Xin Miao1, Shen Zhou1, Mayi $\mathbf { X } \mathbf { u } ^ { 1 }$ , Yuyang $\mathbf { R e n } ^ { 1 , 3 }$ , Tieyun Qian1,2\*

1School of Computer Science, Wuhan University, China 2Intellectual Computing Laboratory for Cultural Heritage, Wuhan University, China 3Research Institute of Nuclear Power Operation, China liyongqi, miaoxin, shenzhou, xumayi $@$ whu.edu.cn, renyy $@$ cnnp.com.cn, qty@whu.edu.cn

# Abstract

Despite the rapid progress that existing automated feedback methods have made in correcting the output of large language models (LLMs), these methods cannot be well applied to the relation extraction (RE) task due to their designated feedback objectives and correction manner. To address this problem, we propose a novel automated feedback framework for RE, which presents a rationale supervisor to verify the rationale and provides re-selected demonstrations as feedback to correct the initial prediction. Specifically, we first design a causal intervention and observation method to collect biased/unbiased rationales for contrastive training the rationale supervisor. Then, we present a verification-feedback-correction procedure to iteratively enhance LLMs’ capability of handling the RE task. Extensive experiments prove that our proposed framework significantly outperforms existing methods.

Code — https://github.com/NLPGM/SRVF

# Introduction

The relation extraction (RE) task aims to extract the semantic relation between entities in the text, which is an important task in information extraction. Unlike previous fine-tuning strategies based on small language models (Wu and He 2019), recent studies (Wan et al. 2023; Ma et al. 2023) leverage the strong instruction understanding abilities and rich intrinsic knowledge of large language models (LLMs) (Ouyang et al. 2022; Touvron et al. 2023; Bai et al. 2022) to enhance the performance of RE.

Despite their significant progress, LLM based methods may suffer from relation bias when performing relation extraction. For example, given a sentence “data is derived from a study”, where “data” and “study” form the “EntityOrigin” relation, LLMs may be influenced by the pretrained knowledge and have the stereotype that “data is the product that someone produces”, thus making a biased relation prediction “Product-Producer”, which ignores that the real producer is investigators (producer of the study). Furthermore, existing LLM based RE methods focus on the preselection of in-context demonstrations (Wan et al. 2023; Ma, Li, and Zhang 2023) or instruction design (Zhang, Gutie´rrez, and $\mathsf { S u } 2 0 2 3 \rangle$ to improve the performance. The verification and feedback mechanism for correcting the biased prediction is still missing from current LLM based RE research.

![](images/ed46bb8198f758f9d367309e16d3c4a39665454c53e2038fe8b2bd23c09d84f7.jpg)  
Figure 1: Comparison between current automated feedback methods (a) and ours (b). The main difference is that our rationale supervisor can verify whether the relation bias occurs and provide re-selected demonstrations as feedback.

To fill this gap, in this study, we focus on exploring the verification and feedback mechanism (Pan et al. 2023) of LLMs for RE. Specifically, we aim to examine whether the relation prediction of LLMs is biased by verifying the rationale (the generated explanation when LLMs perform RE) and providing feedback for correction. However, the current verification and feedback mechanism faces the following two problems when being applied to RE.

Firstly, existing methods are mainly designed for other tasks, e.g., the reasoning task. The objectives of their feedback are also tailored for those tasks, e.g., correcting code, factual, or calculation errors in initial responses (Zhang et al. 2023; Gou et al. 2023), or choosing an optimal prefix for the next step in multi-step reasoning (Khalifa et al. 2023), as shown in Fig. 1 (a). For example, for the mathematical reasoning task, Self-Refine (Madaan et al. 2023) utilizes the LLM agent to find calculation errors in the initial answer and provide error information as feedback to correct the answer. However, such feedback objectives are based on the logical properties of reasoning tasks, which are not available for RE.

Secondly, existing methods (Madaan et al. 2023; Nathani et al. 2023) do not include demonstrations in their feedback. However, the demonstrations are essential for RE even at the correction stage. This is because without demonstrations in the feedback, the RE task would degrade to zero-shot

RE and is harder than the initial few-shot one. Moreover, the demonstrations in initial few-shot RE cannot be directly used in feedback since they will mislead the model back to the initial one, and thus the impact of feedback is discarded.

To address the above problems, we propose a novel automated feedback framework for RE, which trains a rationale supervisor based on a BERT-like small model and utilizes it to not only verify the prediction but also provide new demonstration improved feedback for correction during the inference. As shown in Fig. 1 (b), our rationale supervisor provides re-selected demonstrations as feedback for correcting the initial prediction of LLMs.

In order to train a rationale supervisor, we need to collect both unbiased and biased rationales, i.e., positive and negative samples. Though several verification methods have been proposed to collect positive and negative rationales in other tasks, both their purpose and the collection method are not suitable for our RE task. (1) Firstly, their collected positive and negative rationales are used for training the verifier, which only needs to discriminate the positive predictions from negative ones. In contrast, the rationale supervisor in our framework is designed to correct biased predictions, thus needing to further discriminate different negative rationales. (2) Secondly, the way of collecting rationales in current verification methods relies on the manually annotated golden reasoning steps as positive samples and perform rule-based perturbation (Paul et al. 2023; Golovneva et al. 2023) or error step alignment (Khalifa et al. 2023; Li et al. 2023b) to obtain negative samples. Unfortunately, such annotated samples and rules for perturbation are not available in RE.

In view of this, we propose a causal intervention and observation method to address the lack of annotated rationales and collect biased rationales for training the supervisor. Specifically, we first present a label-guided intervention strategy to collect unbiased rationales, and we also present a diversified intervention strategy to collect biased rationales. In addition, during the inference, we utilize the rationale supervisor to retrieve new demonstrations from the labeled samples and include them in the feedback, which are then used by the LLM for re-generating predictions. Since the supervisor has learned the difference among various biased rationales, the LLM gets the signal to adjust its direction for correction. This verification-feedback-correction procedure iterates until the output rationale is verified as unbiased.

Overall, we make three major contributions. 1) We extend the LLM based RE research to the automated feedback paradigm, which equips LLM with the ability of correcting the biased prediction. 2) We propose a novel supervised rationale verification and feedback framework, which first collects rationales with a causal intervention and observation method for training the supervisor, and then employs the supervisor to retrieve sample-related demonstrations as feedback for guiding the LLM in correction. 3) Extensive experiments prove that our proposed method can improve the performance of LLM based RE methods and is superior to existing automated feedback methods.

# Related Work

LLMs for Relation Extraction Recently, many studies (Xu et al. 2023; Li et al. 2023a; Wei et al. 2023; Wadhwa, Amir, and Wallace 2023; Li, Wang, and Ke 2023) have explored how to unlock the potential of LLMs for the RE task, including designing the in-context demonstration selection strategy (Wan et al. 2023; Ma, Li, and Zhang 2023; Pang et al. 2023) and optimizing instruction patterns (Zhang, Gutie´rrez, and Su 2023; Wang et al. 2023a; Ma et al. 2023). Despite great success, these methods rely solely on optimizing the initial prompt to improve performance. However, we find that due to the relation bias, LLMs may still confuse certain relations with similar entities and thus make biased predictions. To alleviate this issue, we introduce the idea of automated feedback to RE for the first time, expecting to correct biased predictions via the provided feedback.

LLMs with Automated Feedback Some researchers have exploited the automated feedback for correcting the undesirable output of LLMs (Pan et al. 2023; Kamoi et al. 2024). However, the feedbacks in existing methods are designed for correcting various reasoning mistakes, e.g., code errors (Zhang et al. 2023), factual errors (Gou et al. 2023), calculation errors (Nathani et al. 2023; Madaan et al. 2023; Paul et al. 2023), or as an optimal prefix for the next step in multi-step reasoning (Khalifa et al. 2023; Li et al. 2023b). These feedbacks are dependent on the reasoning task and unavailable for RE. Moreover, they do not include the demonstrations which are essential for RE. To address this issue, we propose a novel automated feedback framework which provides re-selected demonstrations as feedbacks to help LLMs correct the biased prediction.

# Method

This section presents our proposed supervised rationale verification and feedback (SRVF) framework for the RE task.

Task Formulation Given a set of pre-defined relation types $Y _ { D }$ , the relation extraction (RE) task aims to predict the relation type $y \in Y _ { D }$ between the head entity $\bar { e } ^ { h }$ and the tail entity $e ^ { t }$ of each test example $x = \{ s , e ^ { h } , e ^ { \dot { t } } \}$ , where $s$ denotes the sentence. In this study, we adopt in-context learning (ICL) with the rationale to prompt LLMs for the RE task. Specifically, for each test example $x$ , we need to randomly select or retrieve $m$ initial in-context demonstrations $\tilde { D _ { i c l } } = \{ \{ x _ { 1 } , r _ { 1 } ^ { u } , y _ { 1 } \} , . . . , \{ x _ { m } , r _ { m } ^ { u } , y _ { m } \} \}$ related to $x$ from the labeled dataset $D _ { l }$ 1. Then, the LLM $f _ { \theta }$ with parameters $\theta$ is expected to output the relation type $y \in Y _ { D }$ between $e ^ { h }$ and $e ^ { t }$ , along with the rationale $r$ , denoted as $\{ r , y \} = f _ { \theta } ( D _ { i c l } , x )$ .

Overview In this paper, we propose a rationale verification and feedback framework to guide LLMs towards better predictions for RE iteratively. Generally, this framework consists of three phases: 1) causal intervention and observation for rationale collection, 2) contrastive training rationale supervisor, and 3) rationale verification and feedback.

Specifically, we first adopt the causal intervention and observation method to collect unbiased and biased rationales, i.e., $R _ { u }$ and $R _ { b }$ . Then, we use $R _ { u }$ and $R _ { b }$ to train the rationale supervisor $\mathcal { R } _ { \gamma }$ with parameters $\gamma$ . Finally, as shown in Fig. 3, in the inference time, once the output rationale $r$ is verified as a biased one by $\mathcal { R } _ { \gamma }$ , we use $\mathcal { R } _ { \gamma }$ to retrieve feedback demonstrations $D _ { f b }$ based on $r$ , where $D _ { f b } \subset D _ { l }$ . The feedback demonstrations are used for re-generating $r$ and $y$ using ICL, i.e., $\{ r , y \} = f _ { \theta } ( D _ { f b } , x )$ . The procedure iterates until the rationale $r$ is verified as unbiased, and the corresponding relation prediction $y$ will finally be output.

# Causal Intervention and Observation for Rationale Collection

Generally, during this phase, for each labeled sample $\{ x _ { i } , y _ { i } \}$ , we aim to collect the unbiased rationale corresponding with the golden label $\{ r _ { i } ^ { u } , y _ { i } ^ { u } \}$ , as well as the biased rationale with corresponding biased relation prediction $\{ r _ { i } ^ { b } , y _ { i } ^ { b } \}$ . This process consists of two steps: 1) induce unbiased rationale, and 2) observe biased rationale. As shown in Fig. 2, we use the structural causal model (SCM) in causal inference (Pearl et al. 2000) to illustrate the strategy.

Preliminary of SCM As shown in Fig. 2, the SCMs show the relationships among the input $( X )$ , the relation prediction $( Y )$ , the rationale for prediction $( R )$ , the certain bias of LLMs $( B )$ and in-context demonstration $I$ . The arrows between nodes indicate causal directions. For example, “ $X $ $R ^ { \prime }$ means that the LLM generates the rationale $R$ related to the prediction for the sample $X$ . “ $\cdot X \to B \to R ^ { \prime }$ indicates that the LLM activates some biased knowledge $B$ related to the sample $X$ and generates a rationale $R$ influenced by the biased knowledge $B$ . Besides, in Fig. 2 (b), the ${ \cdot } d o ( Y ) ^ { \bar { \thinspace } }$ indicates that cutting off all factors that could influence the value of $Y$ and assigning $Y$ a certain value as needed.

Induce Unbiased Rationale Previous methods rely on the human-annotated rationales, e.g., golden reasoning steps in mathematical tasks (Khalifa et al. 2023), which are not available in the RE dataset. To address this issue, we propose a label-guided intervention strategy to obtain the unbiased rationale for each labeled sample, which explains why the sample $x _ { i }$ should be predicted as the golden label $y _ { i }$ .

As shown in Fig. 2 (b), this strategy consists of two steps: 1) cut causal directions that could make bias $( B )$ influence the prediction $( Y )$ , and let the golden label guide the rationale $( R )$ generation, formally denoted as $d o ( Y = y _ { i } )$ and $d o ( Y )  R$ . The observed generated rationale is $R = r _ { i } ^ { u }$ ; 2) conduct similar do-operation to the rationale $R$ and let $d o ( R )$ point to $Y$ , i.e., $\begin{array} { r } { \bar { d o } ( R = r _ { i } ^ { u } ) , d o ( R ) \to Y } \end{array}$ . If the observed value of $Y$ is equal to the golden label $y _ { i }$ , we treat $\{ r _ { i } ^ { u } , y _ { i } \}$ as the unbiased one and add it to $R _ { u }$ .

Observe Biased Rationale In previous methods, incorrect rationales are synthesized from golden ones using perturbation or error step alignment based on certain rules (Golovneva et al. 2023; Khalifa et al. 2023). However, these rules are designed based on the logical properties of reasoning tasks, which are not available in RE. To tackle this problem, we propose a diversified intervention strategy for collecting the biased rationales.

![](images/d1a20c33fb1de9c92254f7ec1c58e21cb6eb60b9e3d6bc0f9e090c08fe5d3b10.jpg)  
Figure 2: The structure causal model for illustrating the proposed causal intervention and observation strategy.

Specifically, for the labeled sample $\{ x _ { i } , y _ { i } \}$ , we first randomly select a demonstration set $D _ { d i }$ with diverse labels, where $D _ { d i } \subset D _ { l }$ and the label of each demonstration in $D _ { d i }$ is not equal to $y _ { i }$ . The diversity of labels in $D _ { d i }$ is designed to induce LLMs to make diverse errors on the same sample, to increase the diversity of collected biased rationales. Then, as shown in Fig. 2 (c), we set the in-context demonstration $I$ as $\{ x _ { j } , r _ { j } ^ { u } , y _ { j } \}$ from $D _ { d i }$ , i.e., $d o ( I = \{ x _ { j } , r _ { j } ^ { u } , y _ { j } \} )$ . Finally, the observed value of rationale $R$ is $r _ { o b s }$ while the observed value of rationale $Y$ is $y _ { o b s }$ . If $y _ { o b s } \neq y _ { i }$ , we treat the observed $r _ { o b s }$ with its corresponding relation prediction $y _ { o b s }$ as a potentially biased one, i.e., $\{ r _ { i } ^ { b } , y _ { i } ^ { b } \}$ , and add it to $R _ { b }$ .

# Contrastive Training Rationale Supervisor

We expect the rationale supervisor to 1) verify whether the output rationale is biased, and 2) provide different feedbacks for different bias situations to correct the initial prediction. To reach this, we adopt contrastive learning to train the rationale supervisor to acquire two abilities: 1) discriminating biased and unbiased rationales, and 2) learning the difference of various biased rationales.

We design two kinds of positive and negative pairs for contrastive training.

For positive pairs, we treat “unbiased rationales with the same golden label”, and “biased rationales under the same bias situation” as the two kinds of positive pairs. For example, if samples $s _ { 1 }$ and $s _ { 2 }$ , which have the same label, are also predicted as the same wrong relation, we call “samples $s _ { 1 }$ and $s _ { 2 }$ are in the same bias situation”. Thus, the biased rationales ${ \bf \rho } _ { r _ { 1 } ^ { b } } ^ { b }$ and $r _ { 2 } ^ { b } .$ ) of $s _ { 1 }$ and $s _ { 2 }$ , are treated as a positive pair and should be pulled together in the rationale representation space, i.e., $r _ { 1 } ^ { b } \right. \left. r _ { 2 } ^ { b }$ .

For negative pairs, we first consider the “biased and unbiased rationales from the same sample” as a negative pair. This is designed to train the rationale supervisor to distinguish between biased and unbiased rationales. For example, a sample $s _ { 1 } = \{ r _ { 1 } ^ { u } , y _ { 1 } \}$ where $y _ { 1 }$ is the golden label and $r _ { 1 } ^ { u }$ is the corresponding unbiased rationale, is wrongly predicted as relation $y _ { 2 }$ and corresponding biased rationale is $r _ { 1 } ^ { b }$ . Thus, $r _ { 1 } ^ { b }$ and $r _ { 1 } ^ { u }$ are treated as a negative pair and should be pushed away in the rationale representation space, i.e., $r _ { 1 } ^ { b } \  \ r _ { 1 } ^ { u }$ . Second, we also treat “biased rationales under different bias situations” as a negative pair to train the rationale supervisor, which can distinguish different bias situa

Initial In-context Demonstrations 画 Feedback Demonstrations Prediction   
Instance: The profiles (head entity) are Feedback Instance: The profiles (head entity) are Instrument  
used by teachers (tail entity) to write better Demonstration used by teachers (tail entity) to write better Agency   
recommendation letters. Retrieval recommendation letters. ? ↓ ↑ Large Large Unbiased Language Biased Language Model Model × ☆ ?   
Rationale: The profiles are the component Rationale: The profiles are tools employed   
that is used by the teachers. Therefore, the 电 by teachers to write better letters. Thus, B   
"profiles" serves as the "Component" while the "profiles" serves as the "Instrument"   
the "teachers" serves as the "Whole". Rationale while "teachers" serves as the "Agency". Rationale   
Relation Prediction: Component-Whole Verification Relation Prediction: Instrument-Agency Verification (a) Initial Prediction (b) Refined Prediction

tions and provide feedback based on the biased rationale in the inference time.

In general, the contrastive loss is calculated as:

$$
\mathcal { L } _ { c l } = - \log \frac { \frac { 1 } { \| S ^ { p o s } \| } \sum _ { \{ r _ { 1 } , r _ { 2 } \} \in S ^ { p o s } } \exp ( s i m ( r _ { 1 } , r _ { 2 } ) / \tau ) } { \sum _ { \{ r _ { 1 } , r _ { 2 } \} \in ( S ^ { p o s } \cup S ^ { n e g } ) } \exp ( s i m ( r _ { 1 } , r _ { 2 } ) / \tau ) } ,
$$

$$
s i m ( r _ { 1 } , r _ { 2 } ) = \mathcal { R } _ { \gamma } ( r _ { 1 } ) \cdot \mathcal { R } _ { \gamma } ( r _ { 2 } ) ^ { \top } ,
$$

where $S ^ { p o s } = S _ { 1 } ^ { p o s } \cup S _ { 2 } ^ { p o s }$ , $S ^ { n e g } = S _ { 1 } ^ { n e g } \cup S _ { 2 } ^ { n e g }$ $S _ { 1 } ^ { p o s }$ and $S _ { ? } ^ { p o s }$ denote the1 two∪kin2ds of positive p1air s∪et, 2and $S _ { 1 } ^ { \hat { n } e g }$ and $\bar { S _ { 2 } ^ { n e g } }$ denote two kinds of negative pair set. Here 1we adopt the dot product as the similarity function $s i m ( )$ and add a temperature hyper-parameter $\tau$ to focus more on difficult pairs (Chen et al. 2020). During the procedure of rationale contrastive training, the parameters $\gamma$ of $\mathcal { R } _ { \gamma }$ are updated to minimize $\mathcal { L } _ { c l }$ .

# Rationale Verification and Feedback

As shown in Fig. 3, in the inference time, the trained rationale supervisor $\mathcal { R } _ { \gamma }$ first verifies whether the prediction is biased. If the prediction is biased, the rationale supervisor will retrieve a feedback demonstration set, which then guides LLMs toward refined predictions. In this subsection, we will elaborate on the “Rationale Verification” and “Feedback Demonstration Retrieval” in Fig. 3 in detail. Here we denote the test example, output rationale, and relation prediction of LLMs as $x , r$ , and $y$ , respectively.

Rationale Verification For verification, we need to select the subsets $S _ { b }$ and $S _ { u }$ related to the prediction $y$ from $R _ { b }$ and $R _ { u }$ , respectively, which are then used as anchors to determine whether the current output rationale is close to the biased or unbiased groups. $\boldsymbol { S _ { b } }$ and $S _ { u }$ are defined as follows:

$$
S _ { b } = \{ \{ r ^ { b } , y ^ { b } \} \mid \{ r ^ { b } , y ^ { b } \} \in R _ { b } , y ^ { b } = y \} ,
$$

$$
S _ { u } = \{ \{ r ^ { u } , y ^ { u } \} \mid \{ r ^ { u } , y ^ { u } \} \in R _ { u } , y ^ { u } = y \} ,
$$

Then, the indicator score to judge whether $r$ is a biased rationale is calculated as follows:

$$
p _ { b } = \operatorname* { m a x } _ { \{ r ^ { b } , y ^ { b } \} \in S _ { b } } s i m ( r , r ^ { b } ) - \operatorname* { m a x } _ { \{ r ^ { u } , y ^ { u } \} \in S _ { u } } s i m ( r , r ^ { u } ) ,
$$

where the similarity function $s i m ( )$ is defined in Eq. 2. When $p _ { b }$ is greater than 0, it implies that the feature of $r$ is closer to the feature field of $\boldsymbol { S _ { b } }$ than that of $S _ { u }$ , which means $r$ and corresponding relation prediction $y$ should be regarded as biased, and feedback is needed to correct them.

Feedback Demonstration Retrieval Once the output rationale $r$ is verified as biased, we need to retrieve a new set of in-context demonstrations based on the feature of $r$ for guiding LLMs toward correct predictions. Specifically, we first select the $k$ most similar biased rationales to $r$ in $S _ { b }$ , denoted as $S _ { b } ^ { t o p k }$ , which is defined as:

$$
S _ { b } ^ { t o p k } = \{ \{ r ^ { b } , y ^ { b } \} \mid r a n k _ { \{ r ^ { b } , y ^ { b } \} \in S _ { b } } ( s i m ( r , r ^ { b } ) ) \leq k \} ,
$$

Then, we select the labeled samples corresponding to the biased rationales in $S _ { b } ^ { t o p k }$ from $D _ { l }$ as the feedback demonstrations $D _ { f b }$ , which is defined as:

$$
D _ { f b } = \{ \{ x _ { i } , r _ { i } ^ { u } , y _ { i } \} \mid \{ x _ { i } , r _ { i } ^ { u } , y _ { i } \} \in D _ { l } , \{ r _ { i } ^ { b } , y _ { i } ^ { b } \} \in S _ { b } ^ { t o p k } \} ,
$$

where the biased $\{ r _ { i } ^ { b } , y _ { i } ^ { b } \}$ and unbiased $\{ r _ { i } ^ { u } , y _ { i } \}$ correspond to the same labeled sample $\{ x _ { i } , y _ { i } \}$ .

Correction via In-context Learning After the feedback demonstrations $D _ { f b }$ are selected, we re-generate $r$ and $y$ using the LLM $f _ { \theta }$ , i.e., $\{ r , y \} = f _ { \theta } ( D _ { f b } , \bar { x } )$ . This process will be iteratively performed until $r$ is verified as unbiased, and the corresponding prediction $y$ will be finally output.

<html><body><table><tr><td rowspan="2" colspan="2">Method</td><td colspan="4">SemEval</td><td colspan="4">TACRED</td><td colspan="4">Re-TACRED</td><td rowspan="2">Avg.</td></tr><tr><td>5-shot 10-shot 20-shot 50-shot 5-shot 10-shot 20-shot 50-shot 5-shot 10-shot 20-shot 50-shot</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">uo</td><td>In-context Learning</td><td>48.40</td><td>49.11</td><td>49.65</td><td>49.31</td><td>24.17</td><td>23.69</td><td>24.66</td><td>24.21</td><td>21.37</td><td>21.99</td><td>21.52</td><td>21.10 31.60</td><td></td></tr><tr><td>w/ Self-Refine</td><td>47.93</td><td>48.50</td><td>49.19</td><td>48.88</td><td>23.15</td><td>23.05</td><td>24.18</td><td>23.12</td><td>21.04</td><td>21.51</td><td>20.71</td><td>21.32</td><td>31.05</td></tr><tr><td>w/ Self-Consistency 49.30</td><td></td><td>49.09</td><td>50.11</td><td>50.35</td><td>25.69</td><td>24.76</td><td>25.64</td><td>25.42</td><td>22.08</td><td>22.56</td><td>21.84</td><td>22.01</td><td>32.40</td></tr><tr><td>w/ GRACE</td><td>50.80</td><td>49.22</td><td>54.28</td><td>54.83</td><td>25.89</td><td>25.78</td><td>26.49</td><td>26.46</td><td>22.50</td><td>22.67</td><td>23.65</td><td>24.34</td><td>33.91</td></tr><tr><td rowspan="2"></td><td>w/ our SRVF</td><td>54.89</td><td>59.67</td><td>62.98</td><td>71.27</td><td>30.07</td><td>31.42</td><td>32.84</td><td>34.58</td><td>28.36</td><td>31.49</td><td>32.87</td><td>36.52</td><td>42.25</td></tr><tr><td>In-contextLearning</td><td>57.33</td><td>59.13</td><td>62.49</td><td>64.26</td><td>27.48</td><td>28.64</td><td>30.08</td><td>27.81</td><td>34.78</td><td>41.85</td><td>42.82</td><td>43.69</td><td>43.36</td></tr><tr><td rowspan="4">CCSE m</td><td>w/ Self-Refine</td><td>57.01</td><td>58.91</td><td>62.27</td><td>63.89</td><td>27.13</td><td>26.89</td><td>29.11</td><td>27.30</td><td>34.33</td><td>41.87</td><td>42.30</td><td>43.16</td><td>42.85</td></tr><tr><td>w/ Self-Consistency 57.54</td><td></td><td>58.81</td><td>62.98</td><td>65.00</td><td>28.82</td><td>29.85</td><td>30.98</td><td>25.42</td><td>35.83</td><td>42.84</td><td>43.71</td><td>44.59</td><td>43.86</td></tr><tr><td>w/ GRACE</td><td>57.93</td><td>58.48</td><td>66.32</td><td>67.48</td><td>28.76</td><td>28.60</td><td>30.03</td><td>26.46</td><td>33.95</td><td>41.53</td><td>42.35</td><td>44.37</td><td>43.86</td></tr><tr><td>w/our SRVF</td><td>60.76</td><td>64.12</td><td>69.54</td><td>76.32</td><td>32.99</td><td>33.50</td><td>34.81</td><td>36.13</td><td>39.48</td><td>46.54</td><td>49.73</td><td>54.31</td><td>49.85</td></tr><tr><td></td><td>In-context Learning</td><td>58.68</td><td>64.90</td><td></td><td>77.32</td><td>26.11</td><td>26.35</td><td>31.15</td><td>33.35</td><td>42.75</td><td>45.53</td><td></td><td>56.22</td><td></td></tr><tr><td></td><td>w/ Self-Refine</td><td>58.38</td><td>64.96</td><td>65.67 65.68</td><td>77.35</td><td>25.01</td><td>25.48</td><td>30.62</td><td>32.67</td><td>42.10</td><td>44.98</td><td>52.89 52.11</td><td>55.62</td><td>48.41 47.91</td></tr><tr><td></td><td>w/ Self-Consistency 59.62</td><td></td><td>65.45</td><td>65.74</td><td>77.48</td><td>26.93</td><td>26.83</td><td>31.67</td><td>33.61</td><td>43.54</td><td>46.04</td><td>53.34</td><td>56.69</td><td>48.91</td></tr><tr><td></td><td>w/ GRACE</td><td>60.83</td><td>65.14</td><td>66.21</td><td>76.98</td><td>27.12</td><td>26.34</td><td>30.95</td><td>33.40</td><td>43.12</td><td>45.23</td><td>52.61</td><td>55.83</td><td>48.65</td></tr><tr><td></td><td>w/ our SRVF</td><td>62.12</td><td>67.03</td><td>68.94</td><td>80.08</td><td>30.50</td><td>30.92</td><td>34.83</td><td>36.32</td><td>46.13</td><td>48.09</td><td>55.07</td><td>59.82</td><td>51.65</td></tr></table></body></html>

Table 1: Results (micro-F1 scores) on the SemEval, TACRED, and Re-TACRED datasets under various few-shot settings. Here we adopt the Llama-2-7b-chat as the LLM. The best results are in bold.

# Experiments

# Evaluation Protocal

Datasets and Metric We adopt three commonly used datasets for RE, including SemEval (Hendrickx et al. 2010), TACRED (Zhang et al. 2017), and Re-TACRED (Stoica, Platanios, and Po´czos 2021). Besides, compared to the scenario with full data, the potential of LLMs under few-shot settings is of more concern (Ma et al. 2023; Xu et al. 2023). Hence we adopt the $k$ -shot ( $k \in \{ 5 , 1 0 , 2 0 , 5 0 \} )$ settings to validate the effectiveness of the proposed method. For all experiments, we report micro-F1 scores where Other and no relation are considered negative labels.

Backbones We experiment with three different methods as backbones for selecting initial in-context demonstrations for LLM based RE, including: 1) Random, which randomly selects initial demonstrations without any retriever. 2) SimCSE, which uses SimCSE (Gao, Yao, and Chen 2021) to retrieve samples that have similar sentence semantics with the test example as initial in-context demonstrations. 3) Taskspecific, which uses a task-specific retriever that has been trained on the labeled samples (Wan et al. 2023).

Baselines To the best of our knowledge, we are the first to explore the verification and feedback mechanism for LLM based RE. Thus, we can only make modifications on current feedback methods in other tasks to adapt them for RE. Specifically, we choose the following baselines:

• Self-Refine (Madaan et al. 2023) consists of three LLM based agents, i.e., RE agent, verifier agent, and refiner agent, for iterative feedback and refinement. • Self-Consistency (Wang et al. 2023b) is proposed to conduct verification for the multiple candidate responses and choose the best response by majority voting.

• GRACE (Khalifa et al. 2023) trains a verifier to select the best intermediate reasoning step, which is then used as feedback for generating the next step.

For Self-Consistency, GRACE, and ours, the number of iterations or candidate responses is set to 5 for fairness. For Self-Refine, the iteration number is set to 1 since we find that more iteration rounds result in performance degradation 2.

# Main Results

Table 1 reports the experimental results with various initial demonstration selection strategies on Llama-2-7b-chat on the SemEval, TACRED, and Re-TACRED datasets. From Table 1, we can draw the following conclusions: 1) Our proposed SRVF framework yields significant enhancements upon various backbones with different demonstration selection strategies. Specifically, the improvement is most significant when randomly selecting the initial demonstrations, getting a $1 0 . 6 5 \%$ absolute micro-F1 score increase on average. Besides, when using $\mathrm { \ S i m C S E }$ and task-specific retriever as backbones to carefully select initial in-context demonstrations, there are also $6 . 4 9 \%$ and $3 . 2 4 \%$ absolute micro-F1 score boosts on average, respectively. 2) Our proposed method exhibits significant superiority over existing verification and feedback methods under all settings. The multi-agent based Self-Refine method is the worst, which is mainly due to its unsuitable feedback objectives and correction manner. Existing methods for verifying the output of LLMs, i.e., Self-Consistency and GRACE, can enhance the performance of in-context learning to some extent. However, since they do not provide explicit feedback signals for LLMs to correct the prediction, their improvements are limited.

Table 2: The ablation results (micro-F1) averaged over three backbones. The best results are in bold.   

<html><body><table><tr><td rowspan="2">Method</td><td>SemEval</td><td>TACRED</td><td></td><td>Re-TACRED</td><td rowspan="2">Avg.</td></tr><tr><td>5-shot 10-shot 5-shot 10-shot 5-shot 10-shot</td><td></td><td></td><td></td></tr><tr><td>Our SRVF</td><td>59.26</td><td>63.61</td><td>31.19 31.95</td><td>37.99</td><td>42.04</td><td>44.34</td></tr><tr><td>W/o LGI</td><td>55.31</td><td>55.46</td><td>25.13 29.83</td><td>24.44</td><td>30.46</td><td>36.77</td></tr><tr><td>w/o DI</td><td>57.90</td><td>62.50</td><td>28.52</td><td>29.78 35.64</td><td>39.99</td><td>42.39</td></tr><tr><td>w/oRCT</td><td>58.37</td><td>62.78</td><td>30.31</td><td>30.87 37.23</td><td>43.73</td><td>43.88</td></tr><tr><td>W/o FDR</td><td>57.03</td><td>60.23</td><td>29.27</td><td>29.85</td><td>35.59 39.39</td><td>41.89</td></tr><tr><td>W/o RG</td><td>52.27</td><td>62.09</td><td>27.52</td><td>29.33</td><td>35.14 38.38</td><td>40.79</td></tr></table></body></html>

# Ablation Study

To validate the effectiveness of components in our method, we introduce the following variants for ablation studies:

• w/o label-guided intervention $( L G I )$ , where the labels do not guide the collecting of unbiased rationales.   
• w/o diversified intervention $( D I )$ , which replaces the DI with random sampling for collecting biased rationales.   
• w/o rational contrastive training $( R C T )$ , which trains the rationale supervisor with cross-entropy loss.   
• w/o feedback demonstration retrieval $( F D R )$ , which removes the FDR strategy and uses the initially selected demonstrations as the feedback.   
• w/o RG, which skips the re-generation process and directly adopts the label of the top-1 retrieved demonstration as the final prediction.

The results of the ablation study are shown in Table 2. From the table, we make the following observations. 1) Removing LGI and DI strategies significantly degrades performance, indicating that LLMs struggle to collect unbiased rationales based solely on generation without causal intervention. 2) Eliminating RCT also reduces performance, demonstrating its effectiveness in helping the rationale supervisor distinguish between unbiased and various biased situations. 3) Omitting FDR significantly decreases performance, highlighting its crucial role in guiding LLMs toward corrected predictions despite iterative verification. 4) Removing the re-generation process results in a substantial performance drop, showcasing that simple assignment of retrieved top-1 demonstrations isn’t sufficient and that in-context feedback for re-generation adds robustness to the correction process.

# Analysis

Effectiveness on Various-scale LLMs To examine whether the proposed method remains effective for variousscale LLMs, we conduct experiments on various sizes of LLMs from the Llama-2-chat (Touvron et al. 2023), MetaLlama-3-Instruct (AI@Meta 2024), and GPT-3.5 (Ouyang et al. 2022), and present their results in Table 3.

From Table 3, it can be seen that our rationale supervisor can boost the performance of LLMs with various sizes. Specifically, even with the most powerful Meta-Llama3-70B-Instruct, there is still a $2 . 4 7 \%$ micro-F1 score improvement over the original in-context learning. The experimental results indicate that the “relation bias” issue exists in LLMs of various scales, and our proposed method can function as a plug-in module for various LLMs to effectively mitigate this problem.

Table 3: Results (micro-F1 scores) using various LLMs with the task-specific retriever.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="2">SemEval</td><td>TACRED</td><td colspan="2">Re-TACRED</td><td rowspan="2">Avg.</td></tr><tr><td></td><td></td><td>5-shot 10-shot 5-shot 10-shot 5-shot 10-shot</td><td></td><td></td></tr><tr><td>R-BERT KnowPrompt 53.92</td><td>42.75</td><td>57.25 56.42</td><td>9.87 16.24 27.86 30.34</td><td>26.64 50.08</td><td>35.01 55.41</td><td>31.29 45.67</td></tr><tr><td colspan="7">Llama-2-7b-chat</td></tr><tr><td>ICL</td><td>58.68</td><td>64.90 26.11</td><td>26.35</td><td>42.75</td><td>45.53</td><td>44.05</td></tr><tr><td>W/SRVF</td><td>62.12</td><td>67.03 Llama-2-70b-chat</td><td>30.50 30.92</td><td>46.13</td><td>48.09</td><td>47.47</td></tr><tr><td colspan="7"></td></tr><tr><td>ICL</td><td>68.92</td><td>69.86 27.32</td><td>27.12</td><td>43.63</td><td>44.94</td><td>46.97</td></tr><tr><td>w/ SRVF</td><td>69.97</td><td>70.00</td><td>27.69 29.47</td><td>45.13</td><td>46.93</td><td>48.20</td></tr><tr><td></td><td></td><td>Meta-Llama-3-8B-Instruct</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="7"></td></tr><tr><td>ICL</td><td>69.90</td><td>69.79 32.63</td><td>32.26</td><td>48.23</td><td>50.69</td><td>50.58</td></tr><tr><td>w/ SRVF</td><td>71.14</td><td>71.41 35.26</td><td>34.29</td><td>52.23</td><td>55.25</td><td>53.26</td></tr><tr><td colspan="7">Meta-Llama-3-70B-Instruct</td></tr><tr><td>ICL</td><td>71.21</td><td>72.40 34.71</td><td>34.97</td><td>56.10</td><td>57.41</td><td>54.47</td></tr><tr><td>W/ SRVF</td><td>74.68</td><td>74.33</td><td>37.05 36.35</td><td>59.27</td><td>59.96</td><td>56.94</td></tr><tr><td colspan="7">GPT-3.5-turbo</td></tr><tr><td>ICL W/ SRVF</td><td>67.26 69.62</td><td>70.58 71.67</td><td>32.46 31.38 37.78 34.63</td><td>43.56 46.22</td><td>46.88 49.66</td><td>48.69 51.60</td></tr></table></body></html>

Comparision with Well-designed Few-shot Methods for RE As shown in Table 3, we include two established supervised fine-tuning methods for RE as baselines: 1) RBERT (Wu and He 2019), which fine-tunes a BERT for the RE task, and 2) KnowPrompt (Chen et al. 2022), which is tailored for few-shot scenarios and has shown good few-shot performance. As we can see from the results, with the help of our proposed SRVF, even the relatively weak Llama-2- $7 b$ -chat can outperform KnowPrompt by $1 . 8 0 \%$ averagely. Moreover, when deploying our SRVF on the most powerful Meta-Llama-3-70B-Instruct, there is an average performance improvement of $1 1 . 2 7 \%$ compared to KnowPrompt.

Analysis on Successfully Corrected Samples To visualize which samples are successfully corrected by the proposed method, we compare the error matrix on the SemEval dataset before and after correction. The results are obtained by summing the number of error predictions of all settings in Table 1. The results are shown in Fig. 4.

From Fig. 4 (a), we observe that LLMs struggle to distinguish between relations that share similar entities, e.g., 687 samples labeled as “Entity-Destination” are incorrectly predicted as “Content-Container”. Such error can arise when, for example, given sentences “please move the eggs into the box” and “there are 5 eggs in the box”, where the same entity pair “eggs” and “box” form “Entity-Destination” and “Content-Container” relations, respectively. Such ambiguity often leads LLMs to misclassify relations when they fail to focus on context, resulting in numerous errors. However, as shown in Fig. 4 (b), the number of samples labeled as “Entity-Destination” but incorrectly predicted as “ContentContainer” is reduced by 250. This indicates that our method effectively alleviates the above issue.

![](images/c5a722d29672ca83a58702a04844d458f50a0bb5b198eae59be6296a80d0a7c4.jpg)  
Figure 4: Error matrix before and after the verificationfeedback-correction procedure. The numbers show how many samples labeled $y$ (on the vertical axis) are incorrectly predicted as $x$ (on the horizontal axis).

Analysis on Method Efficiency Considering possible concerns on the inference efficiency due to the iterative feedbacks, we compare the inference time on the SemEval dataset of different methods. Besides, we also evaluate the pre-inference time of each method, e.g., the time to obtain biased/unbiased data and train the rationale supervisor in our SRVF. The comparison results are shown in Fig. 5.

From Fig. 5, we can observe that:

1) Basic in-context learning (ICL) is the most efficient. 2) Self-Refine does not require pre-inference time, but its inference time is more than the sum of our pre-inference time and inference time. Moreover, Self-Refine has the worst performance among all methods (Table 1).

3) Self-Consistency and GRACE have much higher computational costs than our SRVF, especially in terms of inference time. This is mainly because the proposed rationale supervisor can verify whether the LLM prediction is biased. Only the test samples verified as biased by the rationale supervisor will proceed to the correction round for regeneration. This greatly reduces the time cost of our method in inference time after correction.

Overall, our SRVF is the second-best in computational efficiency while achieving the best performance (Table 1).

Experiments on Document-level RE To explore the effectiveness of our method for document-level RE, we apply SRVF on three backbones and conduct experiments on two commonly used document-level RE datasets, DocRED (Yao et al. 2019) and Re-DocRED (Tan et al. 2022). The random and SimCSE backbones are kept the same as before. For the task-specific backbone, we borrow the idea from REPLM (Ozyurt, Feuerriegel, and Zhang 2024), which obtains the final prediction by aggregating the predictions based on multiple retrieved demonstrations. The experimental results are reported in Table 4.

![](images/20d8c069b253dbe6a1c5f6d6400e562e67b16158068859b6becc02b1025e7ec6.jpg)  
Figure 5: Efficiency comparison of different methods on the 5-shot SemEval setting. The results are accumulated along the X axis. For example, “After Initial Generation” refers to the sum time of “pre-inference” and “initial generation”.

Table 4: Results (micro-F1) on the DocRED (documentlevel RE task). The best results are in bold.   

<html><body><table><tr><td rowspan="2">Method</td><td>DocRED</td><td>Re-DocRED</td><td rowspan="2">Avg.</td></tr><tr><td>5-shot 10-shot 5-shot 10-shot</td><td></td></tr><tr><td>ICL (Random)</td><td>7.76 7.82</td><td>8.27 7.73</td><td>7.90</td></tr><tr><td>w/our SRVF</td><td>15.40 18.00</td><td>15.29 15.65</td><td>16.09</td></tr><tr><td>ICL (SimCSE) w/our SRVF</td><td>15.67 16.40</td><td>11.97</td><td>12.53 14.14</td></tr><tr><td></td><td>18.39 21.55</td><td>17.38</td><td>17.87 18.80</td></tr><tr><td>ICL(Task-specific) 18.29 w/our SRVF</td><td>18.40 20.04 21.55</td><td>17.44 19.98</td><td>18.67 18.20 21.69 20.82</td></tr></table></body></html>

From Table 4, we can observe that: 1) LLM performs poorly on document-level RE, which is consistent with empirical observations in Li, Jia, and Zheng (2023); Sun et al. (2024). This is due to the difficulty LLMs face in selecting entity pairs that have certain relations from a vast space of candidate entity pairs. Besides, the large number of candidate relation labels (96 in DocRED and Re-DocRED) further increases the difficulty in assigning each entity pair a relation. 2) Our proposed SRVF effectively enhances the performance of LLM under various settings on DocRED and Re-DocRED, indicating that our method remains to be effective in such challenging scenarios.

# Conclusion

In this paper, we propose a novel automated feedback framework for LLM based relation extraction (RE), which includes a rationale supervisor to iteratively correct the biased relation prediction of LLMs. Specifically, we first present a causal intervention and observation method to collect unbiased and biased rationales, which are then used to train the rationale supervisor. Then, we develop a verificationfeedback-correction procedure to iteratively enhance LLMs’ ability to correct the biased prediction. Extensive experiments demonstrate the superiority of our framework over existing methods. In the future, we will try to extend the proposed framework to other NLP tasks.