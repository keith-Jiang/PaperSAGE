# Editing Memories Through Few Targeted Neurons

Wei Zhou1, 2, Wei Wei1, 2\*, Guibang $\mathbf { C a o } ^ { 3 }$ , Fei Wang4

Computing and Intelligent Information Processing (CCIIP) Laboratory, School of Computer Science an Technology, Huazhong University of Science and Technology, China 2 Joint Laboratory of HUST and Pingan Property & Casualty Research (HPL), China 3Ping An Property & Casualty Insurance Company of China, Ltd 4 Institute of Computing Technology, Chinese Academy of Sciences zw02138@gmail.com, weiw $@$ hust.edu.cn, sunfacy $ @ 1 6 3 . \mathrm { { c o m } }$ , wangfei@ict.ac.cn

# Abstract

Model editing is a novel research topic in large language models (LLMs), aimed at efficiently handling various knowledge editing tasks. Since irrelevant knowledge is difficult to measure, existing editing methods often lack explicit ways to preserve it, especially for editing methods based on the finetuning paradigm. They generally control the locality performance of model editing by constraining the range of changes in model parameters. However, their performance improvements are not always ideal, and may even lead to a decrease in the editing reliability. In this paper, we try to explore effective editing locality control methods based on the relationship between the stored knowledge and the strongly associated model components. Based on the discovery of “knowledge neurons” and enough experimental results, we further explore the potential characteristics between knowledge and model components, confirm and point out: (1) only $1 \%$ neurons have significant contributions to specific knowledge storage, and (2) these targeted neurons often have a high overlap for knowledge with similar relational descriptions, which means that knowledge with similar relationships may be severely affected when these targeted neurons are modified. Based on these findings, we propose Targeted Neurons Finetuning with Data Augmentation (TNF-DA), which performs data augmentation based on the relational representation of edited knowledge to improve editing locality. By freezing most of the model parameters and only fine-tuning the highly contributing neurons corresponding to the edited knowledge, we obtain desirable results in terms of generalization and specificity compared with previous fine-tuning-based methods. Extensive experiments have demonstrated the superior editing performance achieved by our proposed method.

# Code — https://github.com/lifeforzw/TNF-DA

# Introduction

Pre-trained large language models (LLMs) (Radford et al. 2019; Mann et al. 2020; Vaswani et al. 2017; Zhao et al. 2023; Qiao et al. 2022) have already become commonly used tools in NLP tasks. Since their remarkable performance in understanding human-like text and the vast number of parameters they contain, LLMs are usually considered a knowledge base (Petroni et al. 2019; Roberts, Raffel, and Shazeer 2020; Jiang et al. 2020; Shin et al. 2020), with the capability to respond in natural language. However, as the world doesn’t remain the same, some knowledge may change over time. It’s obviously hard to strike a balance between significant computational costs and smallscale knowledge editing needs (Carlini et al. 2019). To address this challenge, model editing has been proposed (Zhu et al. 2020).

![](images/b3f9b50993b6290fb7beade260485a8009dc94c9e0ca240525c9b73cba6f2d84.jpg)  
Figure 1: Finding the neurons highly contributed to the knowledge, we only modify these targeted neurons while freezing other parameters

Currently, numerous studies on model editing have been proposed. (Zheng et al. 2023; Meng et al. 2022a,b; Mitchell et al. 2021; Li et al. 2024; Zhu et al. 2020; Mitchell et al. 2022; Hartvigsen et al. 2024; Huang et al. 2023; Hernandez, Li, and Andreas 2023) These works can be categorized into two classes: adding extra information to a frozen model somewhere and directly modifying the model’s parameters. The former focuses on determining when and what information should be inputted into the model, paying little attention to finding the optimal location. Among the latter, the existing parameter optimization method is optimal (Meng et al. 2022a,b; Li et al. 2024), which divides the model editing task into two parts: location and optimization, and points out where knowledge is stored.

![](images/126073edd51896bde8209faf68fb9a2d6fdadb8fcd4846459fe5741141edf813.jpg)  
Figure 2: TNF-DA modifies the parameters that are the targeted neurons of the edited knowledge. We edit the knowledge stored in the model based on the relationship between neurons and knowledge: (a) The targeted neurons are collected through causal mediation analysis in FFNs. (b) Then analysis of the targeted neurons shows that the similar relational description has similar targeted neurons. (c) The targeted neurons are modified through an augmented dataset based on the relational description.

The considerations of existing methods mainly revolve around the reliability of editing, and often ignore the means of maintaining the locality of editing, which means keeping the irrelevant knowledge unaffected. A typical strategy is to limit the range of changes in weights. This goes a long way in maintaining model generation capabilities, but lacks explanation in maintaining editing locality. One of the important reasons is that it is difficult to judge whether knowledge is irrelevant. This results in the difficulty of designing some explicit method to control it. Dai et al. pointed out that there are specific “knowledge neurons” related to knowledge in the language models, and the impact on other knowledge is small when editing these neurons. Motivated by this, we try to start from the storage of knowledge in the model and the corresponding model structure. Compared with ”knowledge neurons”, we further compare the characteristics of strongly associated neurons corresponding to relevant knowledge. It was found that when knowledge has similar relational representations, their corresponding strongly associated neurons have high overlap, and these “targeted neurons” often have high contributions to knowledge with similar relational representations at the same time (eg. A neuron with a high contribution to $^ { 6 6 } \{ \}$ was created by” will also have a high contribution to $^ { 6 6 } \{ \}$ was developed by”).

Depending on this finding, we propose TNF-DA, an editing approach through data augmentation. We try to edit these targeted neurons, which means that it may affect the knowledge expressed by the same or similar relationships. So in order to maintain this knowledge while editing, for the given edited knowledge, we construct a relevant dataset based on its relational description where the relation of all training data is the same as the edited knowledge. Using the relevant datasets to finetune a minimal number of parameters of targeted neurons, we can edit the knowledge and mitigate the overfitting problem associated with pure fine-tuning-based methods.

To summarize, our contributions are as follows:

• We confirm that in LLMs, only approximately $1 \%$ neurons have a high contribution to a knowledge representation, while the majority of neurons either have no impact or even exhibit a negative effect. We refer these highlycontributing neurons as “targeted neurons”. • We find that targeted neurons associated with knowledge sharing the same relation description exhibit a high overlap, and some of them are highly contributing to the knowledge description with similar semantic relations. • We propose TNF-DA, a method for model editing that involves augmenting the dataset with edited knowledge based on relational descriptions and partially fine-tuning the targeted neurons related to this knowledge. Our experiments demonstrate that this approach achieves desirable performance in model editing, especially compared to previous fine-tuning-based methods.

# Related Work

# Model Editing

Model editing is an emerging field in recent years. So far, editing methods can be divided into two categories: adding extra input to a frozen model somewhere and modifying the model’s parameters directly (Hartvigsen et al. 2024; Huang et al. 2023; Meng et al. 2022a,b). For the former, additional inputs can be added in the form of inputs as spliced context, utilizing in-context learning (Zheng et al. 2023; Madaan et al. 2022). GRACE (Hartvigsen et al. 2024) and REMEDI (Hernandez, Li, and Andreas 2023) implements adding extra activation values by adding a bank outside the model. T-patcher (Huang et al. 2023) adds “pather”, a small number of learnable parameters to the frozen model to achieve model editing. As for modifying the model’s parameters directly, FT-based methods are proposed initially, like multiloss fine-tune (Sinitsin et al. 2020) and constrained finetune (Zhu et al. 2020). Since these methods suffer from overfitting, meta-learning has been further used. By training hypernetworks, KE (De Cao, Aziz, and Titov 2021) and MEND (Mitchell et al. 2021) can get the post-edit parameters in a short time. ROME (Meng et al. 2022a) converts model editing into a locating and optimization task. Through “Causal Trace”, it is considered that knowledge is stored in specific layers at the last subject token, where model editing can be done by optimizing the parameters there. Based on ROME, MEMIT (Meng et al. 2022b) and PMET (Li et al. 2024) further deal with the problem of mass-editing, modifying parameters of continuous adjacent layers.

# Explanation of Transformers

The parameters of Transformers are basically composed of a multi-head self-attention module (MHSA) and feed-forward network (FFN) (Kovaleva et al. 2019; Hassid et al. 2022; Geva et al. 2023). It is widely believed that, considered as key-value pairs, FFN stores factual knowledge (Geva et al. 2020). FFN in each layer contributes corresponding activations and sums up through the residual network. MHSA is regarded to undertake the syntactic understanding (Voita et al. 2019). In the token dimension, MHSA processes positional information. Each token is able to obtain the above information(both information below in the encoder module) through the self-attention module. It is implied that focusing on the interaction between context input, MHSA plays a role in extracting knowledge and relationships.

# Attribution on Large Language Models

Limited by the interpretability of deep learning, how the deep-network-based LLMs demonstrate excellent language capabilities remains a mystery. Currently, there are many works trying to analyze the attribution in deep networks and LLMs. “Integrated Gradients” (Sundararajan, Taly, and Yan 2017) evaluates the attribution importance by judging the impact on outputs as the inputs change. For language models, it can quickly determine which token in the input plays a key role in the output. Causal mediation analysis (Vig et al. 2020)(also known as Activation Patching) can detect which parts of the model are causally implicated in its behavior through a knock-in or knock-out way. Furthermore, Causal trace (Meng et al. 2022a) is proposed based on causal mediation analysis, which pays attention to the token dimension instead. COAR (Shah, Ilyas, and Madry 2024) transforms the attribution task into a supervised learning task to find important structures while considering interactions. Nanda et al. propose Attribution Patching(AtP), a faster, approximate, causal attribution method and $\mathrm { A t P ^ { * } }$ (Krama´r et al. 2024) further builds on this to filter out the false positives.

# Methodology

# Problem Definition

Model editing, aims to adjust an initial model $f _ { \theta }$ through an edit description $( x _ { e } , y _ { e } )$ (Yao et al. 2023), to modify model’s some performance and get a post-edit model $f _ { \theta _ { e } }$ . Regularly, given a practical edit description with edit input $x _ { e }$ and edit label $y _ { e }$ $f _ { \theta } ( x _ { e } ) \neq y _ { e } )$ , the post-edit model $f _ { \theta _ { e } }$ is adjusted to predict the post-edit output, where $f _ { \theta _ { e } } ( x _ { e } ) = y _ { e }$ . At the same time, a broad set of input associated with the edit description may get new labels from the post-edit model, and the collection of these inputs and edit description is regarded as an editing scope $I ( x _ { e } , y _ { e } )$ (Yao et al. 2023), while the others remain the same as $O ( x _ { e } , y _ { e } )$ . So a successful edit operation should change the behavior of model for the inputs in editing scope rightly while remaining its performance for ones out of editing scope:

$$
f _ { \theta _ { e } } ( x ) = { \left\{ \begin{array} { l l } { y _ { e } } & { i f x \in I ( x _ { e } , y _ { e } ) } \\ { f _ { \theta } ( x ) } & { i f x \in O ( x _ { e } , y _ { e } ) } \end{array} \right. }
$$

According to the edit description $( x _ { e } , y _ { e } )$ , we represent them as a knowledge tuple $t ~ = ~ ( s , r , o ^ { * } )$ (Meng et al. 2022a), composed of subject $s$ , new object $o ^ { * }$ and relation $r$ , while the original object is $o$ . Each time, we provide a natural prompt $p ( s , r )$ describing $( s , r )$ to the post-edit model and check whether the output of the model is $o ^ { * }$ instead of $o$ or something else.

# Analysis of the Relationship Between Neurons and Knowledge

In this section, following the design of causal mediation analysis(CMA) (Vig et al. 2020) and Causal Trace(CT) (Meng et al. 2022a), we try to explore the relationship between neurons in model $f _ { \theta }$ and knowledge $( s , r , o )$ . While CMA and CT focus on the output activations of an overall module, we are concerned about the activations of the hidden states of FFN, which is widely regarded to store factual knowledge. To make it easy to distinguish, we call the activations of the FFN “hidden states” and the activations of the hidden states of the FFN “inner states”. In general, hidden size $D$ and inner size $S$ satisfy:

$$
S = 4 \times D
$$

To evaluate each neuron’s correlation to a factual knowledge, we set three runs for each knowledge description, similar to the setting of Meng et al.:

• clean run: Pass a prompt $x$ describing $( s , r )$ into $f$ , and collect the activations of inner states:

$$
H _ { c } = \{ h _ { l } ^ { i } \ | \ l \in [ 1 , L ] , i \in [ 1 , S ] \}
$$

• corrupt run: The whole prompt is obscured before predicting. Specifically, after embedding $x$ as $H _ { e m b } = [ \bar { h } _ { 0 } ^ { i } ]$ , we adjust $H _ { e m b } = H _ { e m b } + \epsilon$ , where $\epsilon \sim N ( 0 ; \nu )$ , then collect $P _ { * } ( o )$ (the probability of emitting $o$ ) and the corrupted activations:

$$
H _ { * } = \{ h _ { l * } ^ { i } \ | \ l \in [ 1 , L ] , i \in [ 1 , S ] \}
$$

• corrupted-with-restoration run: While running with the setting of a corrupt run, we keep the inner state $h _ { l } ^ { i }$ clean. The clean state will carry the clean information and pass it back to the final output, though others are under perturbation. We record the $P _ { * , c l e a n _ { l } ^ { i } } ( o )$ for every corrupted-with-restoration run, and calculate the indirect effect(IE) as $I E _ { l , i } = P _ { * , c l e a n _ { l } ^ { i } } ( o ) - P _ { * } ( o )$

We conduct experiments on GPT2-small and GPT2-medium (Radford et al. 2019)(without fine-tuning) using 1000 knowledge descriptions they know for sure. As Figure 3 shows, evaluated by IE, it can be obviously found that in the model, about $1 \%$ neurons have a high contribution to the corresponding knowledge. While most neurons have little contribution to the probability of expected output, a few of them even have a negative impact. For these neurons that have a great positive effect on associated knowledge description, we call them ’targeted neurons’, which is similar to the concept in KN (Dai et al. 2021). Since each neuron has its corresponding key, value parameters in FFNs, we would like to regard the associated parameters as $\theta _ { k n }$ , seen in Figure 4.

![](images/598558b37f5e9ef1a6815c987f591e427e52748943d937a830a817907be9a749.jpg)  
Figure 3: An instance of IE distribution characteristics of neurons in GPT2-medium.

In addition, we organize the targeted neurons corresponding to this knowledge with the same $r$ , counting the top 2 targeted neurons with the highest frequency in each same $r$ and calculating their average frequency in all descriptions.

Table 1: Average repetition rate of all targeted neurons on knowledge with same relation and frequency of the top2 targeted neurons. “Base” is the lowest value of frequency.   

<html><body><table><tr><td>model</td><td>Top1</td><td>Top2</td><td>Base</td><td>Repetition</td></tr><tr><td>gpt2-small</td><td>99.43</td><td>98.71</td><td>5.25</td><td>29.22</td></tr><tr><td>gpt2-medium</td><td>94.82</td><td>93.27</td><td>3.95</td><td>31.65</td></tr></table></body></html>

Shown in Table 1, it is obvious that when $r$ is the same, the knowledge probably has some repeated targeted neurons. The Repetition reflects the average number of targeted neurons with the same $r$ (relative to the total number of targeted neurons that may appear). The lower the Repetition, the fewer targeted neurons, and the higher repetition rate of targeted neurons with the same $r$ .

![](images/1905715d514f308b9bb9b57720031b92955e4e228fe7112c94ebdc7c5c53b7d6.jpg)  
Figure 4: Relation between neurons and parameters. A FFN can be divided into $F F N _ { k e y }$ and $F F N _ { v a l }$ , as a set of keyvalue pairs. A targeted neuron of the inner states has a pair of k-v pairs corresponding to it, and $\theta _ { k n }$ is the collection of the k-v pairs.

Furthermore, we organize the common neurons of each $r$ , finding that these common neurons are not only limited to the same $r$ , but also the semantically similar ones, for example:

• “The mother tongue of $\{ \}$ is”, $^ { 6 6 } \{ \}$ is a native speaker of”, “The office language of $\{ \}$ is” , .   
$\cdot ^ { \ \ast } \{ \}$ , created by”, $^ { 6 6 } \{ \}$ , developed by”, $^ { * } \{ \}$ is a product of”,

In summary, our results show that (1) Only $1 \%$ neurons in FFNs of the model are highly contributed to the corresponding knowledge, and (2) Knowledge descriptions with the same or similar relational semantics have similar knowledge neurons. Same experiments are conducted on GPTJ(6B) (Wang and Komatsuzaki 2021), and we also try to select such neurons by integrating gradients as Dai et al.. The experimental results and details in this section are shown in Appendix A.

# TNF-DA Method

Based on the finding of “targeted neurons” distributed in different layers, we explore a simple yet effective way to modify the parameters for knowledge editing. As the FFNs in LLMs are widely regarded as key-value memories, we finetune the base model only on parameters corresponding to the $1 \%$ highly-contributing neurons (targeted neurons). With few iteration epochs, the model editing can be well achieved while preserving the unmodified facts.

In a standard condition, learning or editing a fact $( p , o )$ through fine-tuning ways targets to the expected prediction:

$$
L _ { g e n } ( p ; \theta ) = - \log \mathbb { P } _ { \theta } ( o \mid p )
$$

The cost of fine-tuning the entire model is huge, so only the parameters of the knowledge neurons $\theta _ { k n }$ are collected to modify, with the rest $\theta \backslash \theta _ { k n }$ frozen. However, as the model scale increases, the time for searching targeted neurons increases. To improve the efficiency, we heuristically narrow the search place, analyzing the degree of noise influences $D N I$ :

$$
D N I _ { l } ^ { i } = \lvert \frac { h _ { l } ^ { i } - h _ { l * } ^ { i } } { h _ { l } ^ { i } } \rvert
$$

Among neurons with large $D N I$ , a searching subspace $H _ { s }$ can be located:

$$
H _ { s } = \{ h _ { l } ^ { i } \in H \ | \ D N I _ { l } ^ { i } \in P _ { \gamma } ( D N I ) \}
$$

where $P _ { \gamma } ( D N I )$ represents the set of $D N I$ values in the top $\gamma \%$ . From the optimized searching space, the highlycontributing neuron is denoted as $\theta _ { o k n }$ , still evaluated by $I E$ . To aid generalization, the edited description $( p ( s , r ) , \bar { o } )$ will be used for training concatenating some random prefixes $x _ { j }$ as $\{ x _ { j } \oplus p \}$ :

$$
L _ { g e n } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } L _ { g e n } ( x _ { j } \oplus p ; \theta _ { o k n } )
$$

In order to keep the model’s performance unchanged on irrelevant knowledge while editing, constrained loss is necessary (Zhu et al. 2020):

$$
\begin{array} { c } { { m i n i m i z e _ { \theta _ { o k n } } ~ L _ { g e n } ( p ~ ; \theta _ { o k n } ) } } \\ { { s u b j e c t ~ t o ~ { \displaystyle \sum _ { p ^ { \prime } \in O ( p , o ) } ( L _ { g e n } ( p ^ { \prime } ) ) \leq \delta } } } \end{array}
$$

Through the constraint of Eqn 10, the performance of the model on irrelevant knowledge can be controlled. Instead of building the dataset from the huge uncertain complement $O ( p ( s , r ) , \bar { o } )$ , we only construct it through the relation $r$ , based on the finding that knowledge descriptions with the same or similar relational semantics have similar targeted neurons, especially the same ones. Since that, for each edit description $( p ( s , \dot { r } ) , o )$ with the subject $s$ and relation $r$ , an edit dataset is constructed as:

$$
\begin{array} { r l } & { \mathcal { D } _ { ( p , o ) } = \{ ( p _ { i } ^ { r } , \mathbb { P } _ { \theta _ { 0 } } ( p _ { i } ^ { r } ) ) \mid p _ { i } ^ { r } = ( s _ { i } ^ { r } , r ) , s _ { i } ^ { r } \in S ^ { r } \} } \\ & { \quad \cup \left\{ x _ { j } \oplus ( p , o ) \mid j \in [ 1 , N ] \right\} } \end{array}
$$

Here $S ^ { r }$ is a set of randomly chosen subjects, and whether the model knows the prediction exactly is slight, using KL divergence as training loss:

$$
L _ { l o c } = D _ { K L } ( \mathbb { P } _ { \theta } ( x \mid p ^ { r } ) \mid \mid \mathbb { P } _ { \theta _ { 0 } } ( x \mid p ^ { r } ) )
$$

So finally the train loss is obtained by:

$$
L = L _ { g e n } + \alpha \cdot L _ { l o c }
$$

Here, $\alpha$ is a hyperparameter, adjusting the ratio of $L _ { l o c }$ to the final loss $L$ .

# Experiments Baselines and Datasets

Our experiments are conducted on GPT2-XL(1.5B). Our baseline methods mainly adopt several types of editing parameters directly, including improved Constrained FineTuning $( \mathrm { F T } + \mathrm { W } )$ (Zhu et al. 2020), the meta-learning method MEND (Mitchell et al. 2021), and the locate-and-optimize method ROME (Meng et al. 2022a), MEMIT (Meng et al. 2022b) and PMET (Li et al. 2024). For datasets, we performed counterfactual edit experiments on the dataset, COUNTERFACT (Meng et al. 2022a). More details about datasets can be found in Appendix B.

# Editing Experiments

Following Meng et al., we give the following metrics to test the editing performance. Efficacy Success(ES) evaluates whether the post-edit model can predict the new object $o ^ { * } \colon \mathbb { E } _ { i } \{ a r g m a x _ { o } \bar { f } _ { \theta _ { e } } ( o \mid p ( s _ { i } , r _ { i } ) ) = \bar { o } ^ { * } \}$ . Paraphrase Success(PS) measures whether the post-edit model can give the right answer with a prompt with a rephrasing of the original statement, and Neighborhood Success(NS) is for the irrelevant knowledge. Editing Score(S) is the harmonic mean of ES, PS and NS. Besides the four metrics to evaluate the editing validity, other two metrics (Meng et al. 2022b) for the post-edit model are below: Reference Score(RS) tests the consistent of the model, checking the TF-IDF similarity between a reference Wikipedia text and the generating text about $o ^ { * }$ , and Generation Entropy for fluency degradation, computing as the weighted sum of the entropy of bi- and tri-gram $n$ -gram distributions of generated text.

In order to evaluate the performance of the editing method purely, we first filter out what the model exactly knows, as $\ \bar { f } _ { \theta } ( p ( s , r ) ) = o$ , ignoring the uncertainty. And we finally get 2K pieces of counterfactual edits for GPT2-xl. More experiment details are shown in Appendix C.

Table 2 shows the results of all chosen methods on the 2K edits. The results show that, compared with previous finetuning-based methods, TNF-DA achieves a great improvement, especially in specificity. MEND has a great effect in terms of efficacy, but an obvious decrease in generalization and specificity. In terms of the fluency of the post-edit model, FT-W and MEND are both inferior to FT and TNFDA, which is probably due to the influence of excessive changes in model parameters. In both FT-W and MEND, the range of weight changes is restricted, which can have a beneficial effect on maintaining the fluency of model generation, but will sacrifice generalization or specificity. As for optimization-based methods like ROME and PMET, there is no significant difference in final score between them and TNF-DA, since they have performed well enough on this experiment dataset. Similarly, even though they have their own advantages and disadvantages in editing performance, the fluency and consistency of the model edited by TNF-DA are reduced, similar to other baseline methods.

# Case Study

For insight into TNF-DA’s performance, we set up 3 different case experiments to explore the effect of targeted neurons and the possible problems with fine-tuning-based methods.

Influence of adding prefix As Eqn 8 tells, while training the model with the edits, we first concatenate some random

<html><body><table><tr><td>Editor</td><td>Score</td><td>Efficacy</td><td>Generalization</td><td>Specificity</td><td>Fluency</td><td>Consistency</td></tr><tr><td>GPT2-xl(1.5B)</td><td>/</td><td>0</td><td>0</td><td>100</td><td>625.26</td><td>70.35</td></tr><tr><td>FT</td><td>52.47</td><td>96.81</td><td>70.02</td><td>30.72</td><td>613.48</td><td>75.18</td></tr><tr><td>FT+W</td><td>38.70</td><td>90.04</td><td>20.85</td><td>62.37</td><td>624.71</td><td>75.35</td></tr><tr><td>MEND</td><td>57.92</td><td>99.13</td><td>65.45</td><td>37.90</td><td>624.23</td><td>71.46</td></tr><tr><td>TNF-DA</td><td>87.65</td><td>98.54</td><td>87.21</td><td>79.42</td><td>615.02</td><td>74.27</td></tr><tr><td>ROME</td><td>90.99</td><td>99.64</td><td>95.10</td><td>80.52</td><td>621.91</td><td>76.61</td></tr><tr><td>MEMIT</td><td>86.72</td><td>90.23</td><td>76.42</td><td>95.96</td><td>626.92</td><td>76.56</td></tr><tr><td>PMET</td><td>87.15</td><td>94.06</td><td>78.38</td><td>90.04</td><td>627.01</td><td>76.60</td></tr><tr><td>TNF-DA</td><td>87.65</td><td>98.51</td><td>87.23</td><td>79.41</td><td>615.02</td><td>74.27</td></tr></table></body></html>

Table 2: Quantitative Editing Results. The above part is the comparison result among ours and previous fine-tuning-based methods and hypernetwork-based methods. The bottom part is the comparison to the locate-and-optimize methods.

Table 3: The results of the influence of adding prefixes. TWP represents training the edits with random prefixes. TNWP represents training without any prefix.   

<html><body><table><tr><td>Editor</td><td>Efficacy</td><td>Generalization</td><td>Specificity</td></tr><tr><td>TWP</td><td>100.0</td><td>95.83</td><td>42.06</td></tr><tr><td>TNWP</td><td>100.0</td><td>88.89</td><td>26.98</td></tr></table></body></html>

prefixes $x _ { j }$ as $\{ x _ { j } \oplus p \}$ . The reason for that is to aid generalization across contexts (Meng et al. 2022b), preventing the model from overfitting to the given fixed edit description during training. In this subsection, we experimented with comparing whether or not to use these random prefixes. Based on the settings of Ours-DA without data augmentation, we finetune the parameters of targeted neurons with only the give edit $( p ( s , r ) , o )$ and $( x _ { j } \oplus p ( s , r ) , o )$ , and set a sufficient number of iterations to ensure the efficiency.

Table 3 shows the result of training with prefixes (TWP) or not(TNWP). It is obvious that training with prefixes (TWP) and without prefixes both achieve desirable results in generalization, but poor performance in specificity. This is probably due to the fact that the large number of iterations. However, we can find that when additional prefixes is concatenated to the training data, the performance in specificity will still be greatly improved, not only in generalization. This proves that training with random prefixes can mitigate the overfitting in a fine-tuning process, allowing the model to learn the knowledge paradigm rather than the fixed input prompts themselves.

Generation loss convergence and neuron chosen In order to further verify the effectiveness of the targeted neuron in the TNF-DA method and its relevance to knowledge, we conduct an experiment with the convergence speed of the generation loss when training different parameters in this section. We select some edits, and train them with standard TNF-DA and TNF-DA without targeted neurons (OursRN). We randomly select the parameters corresponding to the same number of neurons $( 1 \% )$ as the editable parameters and record the generation loss to compare their convergence speeds.

![](images/10e123a8fc7cb8f072fc740c1b07f085118fc5768d902dc80176081703e6810b.jpg)  
Figure 5: An instance showing the difference of convergence speed when modifying targeted neurons and randomlychosen neurons.

Experimental results of convergence speed differences are shown in Table 5. As can be seen from the figure, the standard TNF-DA method of editing the parameters of the targeted neuron converges significantly faster than the random selection of neuron parameters in the generation loss. This largely shows that targeted editing of these high-contributed neurons is effective to a certain extent, and also proves the high correlation between the targeted neurons and its corresponding knowledge. More results are shown in Appendix D.

Different knowledge in the model Note the difference in different knowledge in Figure 3: under the same noise perturbation, the IE values given by the neuron may have significant differences. Therefore, we conduct an experiment to explore the difficulty of editing different knowledge and try to explain the difficulties of fine-tuning-based editing methods. We sample several groups of edits with the same relation from the dataset, and used TNF-DA to edit until their generation loss was less than a threshold instead of a fixed

<html><body><table><tr><td>Editor</td><td>Score</td><td>Efficacy</td><td>Generalization</td><td>Specificity</td><td>Fluency</td><td>Consistency</td></tr><tr><td>GPT2-xl(1.5B)</td><td>/</td><td>0</td><td>0</td><td>100</td><td>625.26</td><td>70.35</td></tr><tr><td>Ours</td><td>87.65</td><td>98.54</td><td>87.21</td><td>79.42</td><td>615.02</td><td>74.27</td></tr><tr><td>Ours-RN</td><td>52.73</td><td>94.46</td><td>66.01</td><td>32.14</td><td>608.21</td><td>71.52</td></tr><tr><td>Ours-DA</td><td>77.39</td><td>98.50</td><td>88.61</td><td>56.89</td><td>618.80</td><td>74.03</td></tr></table></body></html>

Table 4: The results of the ablation experiments. Ours represents the standard TNF-DA. Ours-RN represents modifying randomly-chosen parameters. Ours-DA represents that the training dataset is augmented randomly.

Relation and Ave epochs {} is a part of the continent of The native language of $\{ \}$ is {}, created by The official religion of $\{ \}$ is {}, produced by $\{ \}$ is a native speaker of $\{ \}$ belongs to the continent of $\{ \}$ is developed by The official language of $\{ \}$ is $\{ \}$ is created by $\{ \}$ is produced by $\{ \}$ , developed by In $\{ \}$ , the language spoken is The mother tongue of $\{ \}$ is 0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 Average epochs

epochs:

$$
L _ { g e n } \leq \delta
$$

Each edit is tested 5 times and takes the average of the number of stopping rounds, and the results are shown in Figure 6.

It can be seen that different relations do not have similarities, and even the same relations do not have obvious characteristics. This means that it may be difficult to ensure extremely high accuracy for epoch-fixed training, that is, it may lead to overfitting for easy-to-edit knowledge, while difficult-to-edit knowledge may not be completely changed. For the dynamic epoch solution, this has a similar impact on the relevant and irrelevant knowledge, making it difficult to achieve high generalization and specificity.

# Ablation Study

To demonstrate the effectiveness of our editing method with editing targeted neurons and training datasets based on data augmentation, we conduct ablation studies for TNF-DA, detailed in Table 4.

To assess the effectiveness of editing targeted neurons only, we edit parameters corresponding to a set of randomly chosen neurons (Ours-RN). TNF-DA’s performance on specificity significantly degrades while editing the randomly chosen parameters, proving the relationship between the targeted neurons and knowledge based on the relation description. In addition, the efficacy remains at a high level, but the decrease in paraphrase is obvious, showing the relationship between the targeted neurons and knowledge based on the relation description. This is probably because the model overfits the edited knowledge description, but does not generalize to other rephrasing statements well.

Replacing the data augmentation strategy based on relation description, a random generation strategy (Ours-RD) is adopted for comparative experiments. While it has decreased slightly in generalization, but is obviously insufficient in specificity, indicating that the targeted neurons are more likely to contribute to knowledge with the same or similar relation description.

# Conclusion

We find that approximately $1 \%$ targeted neurons in FFN of a language model are highly contributing to the correlated knowledge storage. These targeted neurons are consistently associated with the relational description of the knowledge, and exhibit a high repetition rate for knowledge with the same or similar relational descriptions. Based on the observations, we propose a fine-tuning-based method, TNF-DA, which achieves model editing by modifying the parameters of targeted neurons associated with the edited knowledge, using an augmented dataset based on the relational descriptions. Our experiments on Counterfact demonstrate that TNF-DA achieves a superior editing performance in all finetune-based methods, and produces excellent results comparable to the locate-and-optimize model editing methods. Furthermore, our ablation studies and case studies indicate that our strategies for targeted neurons and data augmentation are effective, mitigating the challenge faced by the previous fine-tuning-based methods in balancing generalization and locality. Our findings provide additional insights into the operational dynamics of FFN in language models.

# Limitations

Without using the knowledge graph as an aid, TNF-DA has not been further designed for the performance of reasoningrelated knowledge. The method is based on the semantics level to conduct neuron experiments, so it is difficult for the edited model to reason based on the edited knowledge. In addition, TNF-DA is a fine-tuning-based method. Since the targeted neurons are distributed across different layers of the model, it is challenging to enhance the efficiency of editing parameters through optimization like ROME and MEMIT (Meng et al. 2022a,b). In terms of targeted neurons, it can be found that there are a small number of neurons playing a negative role, rather than showing little impact. Our work has not further explored this aspect of neurons. On the other hand, due to the dispersed nature of these neurons, we do not further consider their mutual interactions.