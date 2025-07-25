# CITI: Enhancing Tool Utilizing Ability in Large Language Models Without Sacrificing General Performance

Yupu Hao1, 2, Pengfei Cao1, 2, Zhuoran Jin1, 2, Huanxuan Liao1, 2, Yubo Chen1, 2, Kang Liu1, 2, 3\*, Jun Zhao1, 2

1 The Key Laboratory of Cognition and Decision Intelligence for Complex Systems, Institute of Automation, Chinese Academy of Sciences, Beijing, China 2 School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, China 3 Shanghai Artificial Intelligence Laboratory {haoyupu2023, liaohuanxuan2023}@ia.ac.cn, {pengfei.cao, zhuoran.jin, yubo.chen, kliu, jzhao}@nlpr.ia.ac.cn

# Abstract

Tool learning enables Large Language Models (LLMs) to interact with the external environment by invoking tools, enriching the accuracy and capability scope of LLMs. However, previous works predominantly focus on improving the model’s tool-utilizing accuracy and the ability to generalize to new, unseen tools, excessively forcing LLMs to adjust specific tool-invoking pattern without considering the harm to the model’s general performance. This deviates from the actual applications and original intention of integrating tools to enhance the model. To tackle this problem, we dissect the capability trade-offs by examining the hidden representation changes and the gradient-based importance score of the model’s components. Based on the analysis result, we propose a Component Importance-based Tool-utilizing ability Injection method (CITI). According to the gradient-based importance score of different components, it alleviates the capability conflicts caused by the fine-tuning process by applying distinct training strategies to different components. CITI applies Mixture-Of-LoRA (MOLoRA) for important components. Meanwhile, it fine-tunes the parameters of a few components deemed less important in the backbone of the LLM, while keeping other parameters frozen. CITI can effectively enhance the model’s tool-utilizing capability without excessively compromising its general performance. Experimental results demonstrate that our approach achieves outstanding performance across a range of evaluation metrics.

# Code — https://github.com/hypasd-art/CITI Extended version — https://arxiv.org/abs/2409.13202

# 1 Introduction

Large Language Models (LLMs) have demonstrated significant capabilities in understanding language and generating human-like text (Zhao et al. 2023; Li et al. 2023a). Despite their impressive performance, LLMs still face limitations, such as the tendency to generate hallucinated information (Huang et al. 2023; Sun et al. 2024) and an inability to interact with the real physical world (Qin et al. 2023). To tackle these limitations and expand the model’s capabilities scope beyond traditional natural language tasks, there is a growing

Llama-3-8B-Instruct FT 100 Tool-utilizing General Tasks Accuracy 80 Performance Performance 60 40 20 0

interest in enabling LLMs to interact with the external environment by equipping various tools (Qin et al. 2024; Schick et al. 2023; Gao et al. 2023a; Chen et al. 2023; Nakano et al. 2022; Lu et al. 2023).

There have been a substantial number of benchmarks evaluating different tool-utilizing aspects (Ye et al. 2024; Li et al. 2023b; Zhan et al. 2024), along with a suite of proposed methodologies (Qin et al. 2024; Tang et al. 2023). These approaches predominantly leverage in-context learning (Shen et al. 2023; Wu et al. 2023) or fine-tuning techniques (Shen et al. 2024; Gao et al. 2023b; Chen et al. 2024) to equip the model with tool-invoking ability. In contrast to in-context learning, fine-tuning has gained popularity for its capability to deeply integrate task-specific knowledge into model parameters, enhancing its adaptability to specific tasks. However, as illustrated in Figure 1, we find that excessive fine-tuning on tool learning datasets severely diminishes the model’s broader cognitive competencies, leading to a decline in the model’s general abilities and catastrophic forgetting about parametric knowledge.

Tools serve models rather than models serving tools. Consequently, it is crucial for the tool-augmented models to incorporate tool-utilizing ability while simultaneously preserving their original general versatility. Thus, an intriguing question is what happens in LLMs when fine-tuning on tool learning dataset? We dissect the question from two aspects: hidden representation and components. In detail, from hidden representation perspective, when subtracting the originally hidden representations from those representations tuned on the tool learning dataset, we observe a significant phenomenon: the increments, derived by subtraction, exhibit a Co-directional shift in the hidden state space, which implies there is a strong correlation in the direction of increments between various tool-unrelated tasks (e.g. mathematics, coding) and tool-related tasks. From components perspective, for different capabilities, we calculate the gradient-based importance score ranking of the model’s linear modules (referred to as the linear components) based on corresponding datasets respectively. We discover that the important linear components in the ranking play a more vital role in corresponding ability expression while the unimportant components contribute less to this ability. Concurrently, we find that the importance rankings of different abilities are highly consistent. This suggests that certain components are pivotal in the expression of the model’s abilities across a broad range of tasks, whereas some components exert a more limited influence on most tasks. And we further explore the impact of fine-tuning different components based on tool-related importance ranking. It is found that optimizing important components results in a significant decrease in general performance, and the model may not fully learn the knowledge of calling tools only by optimizing unimportant components.

Based on the insights from our analysis, we propose a novel Component Importance-based Tool-utilizing ability Injection method (CITI), which applies Mixture-of-LoRA adapters to important components and adopts full parameter fine-tuning on unimportant components. Firstly, we identify the gradient-based importance score of all the linear components in the model. Secondly, for the important linear components, we integrate a set of Mixture-Of-LoRA (MOLoRA) adapters to absorb the tool-invoking knowledge. To handle the Co-directional shift phenomenon, we design a router network to separate the tool-related and tool-unrelated inputs, reducing the impact on the LLM’s backbone. Thirdly, for the unimportant linear components, we employ full parameter fine-tuning to take full advantage of more parameters. Specifically, Our training process adopts a three-stage approach. In the initial training stage, Router Pre-training, we pre-train the router network in MOLoRA to teach it to distinguish tool-related and tool-unrelated inputs. In the second stage, MOLoRA Improvement, we concentrate on finetuning the MOLoRA adapters while freezing the backbone of LLM. In the third stage, Unimportant Components Optimization, CITI fine-tunes a small part of unimportant components within the backbone to improve the model’s performance while maintaining its general abilities.

To summarize, the contributions of our work include:

• We discover the phenomenon that fine-tuning LLMs on tool learning datasets significantly influences the general abilities, and we have conducted a thorough inspection of this trade-off phenomenon by analyzing the perspective of hidden representation and components. Adequate experiments and analysis demonstrate the factors resulting in the catastrophic forgetting of general abilities.

• We propose a Component Importance-based Toolutilizing ability Injection method (CITI), alleviating the trade-offs by applying distinct strategies to different components identified by gradient-based importance score of the model.

• Our method achieves competitive tool-utilizing results across two tool learning datasets. Additionally, it preserves general performance which is $7 . 5 9 \%$ superior to LoRA and $3 1 . 9 5 \%$ superior to full parameter fine-tuning on average in dataset API-Bank (Li et al. 2023b), and it is also $8 . 9 6 \%$ better than LoRA and $2 9 . 0 3 \%$ better than full parameter fine-tuning in dataset ToolAlpaca (Tang et al. 2023), effectively enhancing tool-utilizing ability while preserving LLMs general performance.

# 2 Analysis of Hidden Representation and Components

Hidden representation and components play a crucial role in determining the hidden state of inputs as they propagate from lower layers to higher layers, helping us understand the inner working process of LLM. Specifically, we conduct the following analytical experiments and assess general abilities on a range of frequently-used datasets, including GSM8K (GSM) (Cobbe et al. 2021), HumanEval (HE) (Chen et al. 2021), TriviaQA (TQA) (Joshi et al. 2017), MT-Bench (MT) (Zheng et al. 2023).

# 2.1 Hidden Representation Perspective

The hidden representation refers to the output vector by each layer of the auto-regressive transformer model. Our goal is to explore the changes in these representations following fine-tuning with the tool learning instructions.

Analytical Approach We introduce the concept of Incremental Change of Capability $( I C C )$ to quantify the changes in the hidden representation of the fine-tuning process. Specifically, for ability $t$ and instruction $m$ within a processed dataset $\mathcal { D M } _ { t }$ , $\dot { H } _ { R E F } ^ { l } ( m )$ is the original hidden representation of the last token in instruction $m$ in the untrained model at layer $l$ , while $H _ { S F T } ^ { l } ( m )$ represents the representation of the same token after fine-tuning. And the $I C C _ { t } ^ { l }$ can be represented as:

$$
I C C _ { t } ^ { l } = \frac { 1 } { | \mathscr { D } \mathscr { M } _ { t } | } \sum _ { m \in \mathscr { D M } _ { t } } H _ { S F T } ^ { l } ( m ) - H _ { R E F } ^ { l } ( m )
$$

The average of the vector’s increment computed by $\mathcal { D M } _ { t }$ is considered to represent the changes in hidden space of ability $t$ . The cosine similarity of these increments is then computed to assess the relationship of the vector’s change between ability $a$ and $b$ in the model’s hidden state space:

$$
S i m ( a , b ) = \frac { I C C _ { a } \cdot I C C _ { b } } { | I C C _ { a } | \times | I C C _ { b } | }
$$

Experimental Results To get representations, we randomly truncate a portion of the golden answer and append it after the instruction, creating a “final input”. For a specific task, we sample 1000 “final input” from the dataset to construct $\mathcal { D M } _ { t }$ and put them into the model. The vector corresponding to the last token of the “final input” $m$ of different layers is considered as the hidden representation vector $H ( m )$ . Then we compute the similarity of different $I C C$ .

![](images/c218793ffc368c07d0b6ed99cbf474fff9e755ba29c6697da79cc35389e9a332.jpg)  
Figure 2: Cosine similarity of $I C C$ between the input of different layers of Feed-Forward Network (FFN) in model Meta-Llama-3-8B-Instruct, where the notation with an asterisk $( ^ { * } )$ represents $I C C$ fine-tuned on the code-related dataset (e.g. $\mathrm { T Q A ^ { * } }$ represents $I C C$ of TriviaQA trained by code dataset), and no asterisk $( ^ { * } )$ represents $I C C$ fine-tuned on tool learning dataset.

In Figure 2, the solid lines represent the similarity between tool-related increments $I C C _ { t o o l }$ with general task increments $I C C _ { t }$ , where LLM is fine-tuned on the tool learning dataset. For better comparison, we train a model on a code generation dataset and compute its increments $I C C _ { t ^ { * } }$ . Here $t$ represents a specific general ability.

We find that there is a Co-directional shift phenomenon in the model’s hidden state space. The changing direction of the $I C C _ { t }$ on the general task is positively correlated with the direction of $I C C _ { t o o l }$ in the model’s hidden state space, compared with the dashed line. Our hypothesis is that after fine-tuning by tool learning datasets, the model cannot distinguish tool-related and tool-unrelated inputs in the hidden state space correctly, resulting in more similar activation states for different types of instructions, which affects the general performance of the model.

# 2.2 Components Perspective

Analytical Approach Large language model, composing the decoder-base architecture of transformer, typically comprises multiple layers. Inspired by Bansal et al. (2023), we adopt the gradient-based importance score to identify the important components.

In the auto-regressive language models, for the input $x$ and corresponding labels $y$ in given dataset $\mathcal { D }$ , the loss function $\mathcal { L }$ to optimize the model is calculated as follows:

$$
\mathcal { L } ( \mathcal { D } , \theta ) = \sum _ { ( x , y ) \in \mathcal { D } } \log p _ { \theta } ( y | x )
$$

<html><body><table><tr><td rowspan="2">Dataset</td><td colspan="3"></td><td rowspan="2">Vanilla</td></tr><tr><td>T-20%</td><td>TReplcD-2t%</td><td>D-50%</td></tr><tr><td>GSM8K</td><td>64.59</td><td>40.56 79.15</td><td>74.98</td><td>79.45</td></tr><tr><td rowspan="2">HumanEval TriviaQA</td><td>51.83</td><td>25.00 58.24</td><td>20.12</td><td>59.15</td></tr><tr><td>64.95</td><td>58.76 64.95</td><td>63.96</td><td>64.82</td></tr><tr><td>MT-Bench</td><td>72.53</td><td>50.97 81.75</td><td>73.19</td><td>80.13</td></tr></table></body></html>

Table 1: The performance of the model on the corresponding tasks after components parameter replacement. ${ \mathrm { T } } { \mathrm { - } } { \bar { \mathbf { X } } } { \% }$ represents taking the top $\mathbf { x } \%$ components of the model to be replaced by importance ranking order, and $\mathbf { D - X \% }$ is the opposite. Vanilla means test on Meta-Llama-3-8B-Instruct.

where $\theta$ denotes the complete set of the model’s parameters. For specific parameters set $\theta _ { h }$ , the importance score $\mathcal { T } _ { h }$ is computed as:

$$
\begin{array} { r l } & { \mathcal { T } _ { h } ( \mathcal { D } , \theta ) = \displaystyle \lvert \mathcal { L } ( \mathcal { D } , \theta ) - \mathcal { L } \left( \mathcal { D } , \theta \mid \theta _ { h } = 0 \right) \rvert } \\ & { \quad \quad \quad \quad = \displaystyle \left. \frac { \partial \mathcal { L } } { \partial \theta _ { h } } ( \theta _ { h } - 0 ) + \frac { 1 } { 2 ! } \frac { \partial ^ { 2 } \mathcal { L } } { \partial \theta _ { h } ^ { 2 } } ( \theta _ { h } - 0 ) ^ { 2 } + \cdot \cdot \cdot \right. } \\ & { \quad \quad \quad \approx \displaystyle \left. \theta _ { h } ^ { T } \frac { \partial \mathcal { L } } { \partial \theta _ { h } } \right. } \end{array}
$$

To accelerate the calculating speed, the score is estimated by calculating the first-order derivative of the Taylor expansion in the above formula.

Experimental Results To examine the changes of components across various general tasks during the tool learning process, we select a subset of instructions from the training set as a proxy for a specific general ability $t$ . Employing Equation (4), we determine the importance scores for the linear components in the vanilla model (Meta-Llama-3-8BInstruct) with these data. In order to explore the effect of the importance of components, we conduct three experiments as follows:

Experiment 1: We prioritize components based on their importance scores and substitute the parameters of these components in the vanilla model with the corresponding weights from the model that has been fine-tuned on the tool learning dataset API-Bank, leaving all other parameters intact. We then assess the model’s performance after replacement.

In Table 1, we observe that tool learning influences other abilities expressions in most cases. Following the replacement operation, the model’s performance (GSM8K, HumanEval, and MT-Bench) significantly decreased compared to the vanilla model, indicating that fine-tuning on the tool learning dataset stimulated tool-invoking abilities in these components while suppressing their original inherent abilities. More significant performance impacts occur when high-importance components are altered. The performance of the model significantly drops after replacing weights with higher importance. Moreover, this phenomenon becomes more pronounced as the replacement ratio increases.

Experiment 2: Furthermore, to delve into the inter-task relationships of components’ importance ranking, we apply

IF 0.760.83 0.6 0.430.16 1.0 IF 1 0.630.720.440.290.15 1.0 IF 1 0.830.870.740.660.17 1.0 Math 0.76 1 0.810.690.530.15 0.8 Math 0.63 1 0.720.530.420.18 0.8 Math 0.83 0.9 0.78 0.8 0.18 0.8 Code 0.830.81 1 0.7 0.520.15 Code 0.720.72 1 0.560.360.18 Code 0.87 0.9 0.8 0.740.18 0.6 0.6 0.6 KQA 0.6 0.69 0.7 0.670.16 KQA 0.440.530.56 1 0.47 0.2 KQA 0.740.78 0.8 0.720.19 Tool 0.430.530.520.67 1 0.19 0.4 Tool 0.290.420.360.47 1 0.19 0.4 Tool 0.66 0.8 0.740.72 1 0.17 0.4 Random 0.160.150.150.160.19 1 0.2 Random 0.150.180.18 0.2 0.19 1 0.2 Random 0.170.180.180.190.17 1 0.2 IF deKQA Tool Ran ndom IF odeKQATool Random IFMath devoArool Random high importance groups moderate importance groups low importance groups

the Jaccard index to analyze the correlation between the importance ranking of model components. The Jaccard index can be represented as: $J ( A , B ) \overset { \cdot } { = } | A \cap B | / | A \cup B |$ , used to compare the differences and similarities between two samples $A$ and $B$ . Specifically, based on the importance ranking of components associated with ability $t$ , we segment the components into three groups evenly: high, moderate, and low. Then we compute the Jaccard Index between different abilities for each group respectively. For example, $J ( C _ { h } ^ { t } , C _ { h } ^ { m } )$ represents the Jaccard Index between components groups of high importance calculated by ability $t$ and $m$ respectively.

As shown in Figure 3, there is a high correlation across different tasks of their components importance ranking. Certain components are pivotal in the expression of the model’s abilities across a broad range of tasks, whereas part of the components exert a limited influence on most tasks.

Experiment 3: We try to inject the tool-invoking knowledge into the less important components as they store less tool-related ability. Considering a components set $H$ containing all linear modules of the network, for tool learning dataset $\mathcal { D } _ { t o o l }$ , we compute the component level gradientbased importance scores for component $h$ as follows:

$$
\mathcal { M } _ { h } ( \theta ) = \frac { \mathcal { T } _ { h } ( \mathcal { D } _ { t o o l } , \theta ) } { \sum _ { i \in H } \mathcal { T } _ { i } ( \mathcal { D } _ { t o o l } , \theta ) }
$$

In order to better combine the importance ranking of components and the training process, we select $20 \%$ (Top, Down, Random) components in the model based on component importance $\mathcal { M } _ { h }$ and train them using full parameter fine-tuning and LoRA methods, while keep other parameters frozen. Then we analyze the fine-tuning result in detail.

As illustrated in Table 2, we find that fine-tuning the components with lower importance score $\mathcal { M } _ { h }$ have the least impact on the general performance, which confirms the existence of high correlations between different abilities of the models because $\mathcal { M } _ { h }$ is computed by tool learning dataset. At the same time, it is observed that fine-tuning components with lower importance may not perform outstandingly in terms of tool-invoking ability. If only unimportant parameters were fine-tuned, the model can not thoroughly understand the format of tool callings. But “Top” and “Random”

<html><body><table><tr><td rowspan="2">Model</td><td colspan="2">Tool Ability</td><td colspan="4">General Ability</td></tr><tr><td>C</td><td>R</td><td>GSM</td><td>HE</td><td>TQA</td><td>MT</td></tr><tr><td colspan="7">LoRA</td></tr><tr><td>Top</td><td>59.95</td><td>40.85</td><td>74.00</td><td>53.66</td><td>64.09</td><td>78.56</td></tr><tr><td>Down</td><td>46.21</td><td>38.84</td><td>78.39</td><td>55.49</td><td>63.71</td><td>74.00</td></tr><tr><td>Random</td><td>64.18</td><td>36.87</td><td>75.36</td><td>54.27</td><td>65.11</td><td>76.40</td></tr><tr><td colspan="7">Full parameter</td></tr><tr><td>Top</td><td>66.11</td><td>44.04</td><td>71.65</td><td>50.61</td><td>64.18</td><td>73.15</td></tr><tr><td>Down</td><td>57.38</td><td>40.36</td><td>81.20</td><td>54.27</td><td>59.96</td><td>74.11</td></tr><tr><td>Random</td><td>59.31</td><td>37.99</td><td>74.30</td><td>50.61</td><td>63.77</td><td>72.48</td></tr></table></body></html>

Table 2: The fine-tuning experiments based on gradientbased importance score on the dataset API-Bank. The evaluation metrics C and R refer to Section 4.3.

do not have such problems. This inspires us only fine-tuning the unimportant component sometimes may be not optimal.

# 3 Methodology

Instead of only focusing on the precise tool-callings, we propose a Component Importance-based Tool-utilizing ability Injection method (CITI), injecting the tool-utilizing ability into the model while maintaining its general abilities. Based on the analysis experiments, in order to fully utilize the model’s capability, we apply Mixture-of-LoRA adapters to important components and fine-tune a subset of unimportant components. As shown in Figure 4, our method mainly consists of three training stages: Router Pre-training (RP), MOLoRA Improvement (MI), Unimportant Components $O p$ - timization (UCO). We will illustrate each technique in detail.

# 3.1 Mixture of LoRA

Based on the analysis of component importance, we want to fine-tune both important and unimportant components to achieve a comprehensive grasp of tool-utilizing knowledge. However, adjusting critical components through fine-tuning can sometimes be detrimental to the overall network performance as the Co-directional shift phenomenon in hidden space. To alleviate this influence, a straightforward idea is to implement a gating function to distinguish tool-related

output The data from various tasks output LLM Router LLM Linear Tranbslfocrkmer Linear   
Important Components Analysis 𝐌𝐌𝒉𝒉 / 山 LLM Linear ↑ input Router Pre-training ↑ 山 output 𝑪𝑪𝒉𝒉 Tranbslfocrkmer   
Components Importance Score weighting LLM Linear MOLoRA Transformer LLM block Router ： 山 Linear 𝐌𝐌𝒉𝒉 ? ↑ LLM Linear input 𝑪𝑪𝒉𝒉 𝐌𝐌𝒉𝒉 input MOLoRA Improvement Unimportant Components Optimization Fine-tune Frozen

and tool-unrelated inputs, and then assign different specialized experts to handle them separately. To mitigate this, we implement MOLoRA adapters on these pivotal components with higher $\mathcal { M } _ { h }$ .

Inspired by Zhao et al. (2024), we apply a router network with an additional neuron in MOLoRA to separate the inputs, allowing tool-related inputs are produced by the combine of the LLM backbone and external LoRA experts. And the tool-unrelated data is predominantly processed by the backbone itself.

Meanwhile, mixing data during downstream task finetuning helps alleviate catastrophic forgetting of models and helps to avoid over-fitting. To train the router network and avoid catastrophic forgetting problems, we mix data from other fields during the tool learning process.

For router network, we use the linear module as the gate function $G ( x )$ . Denote the input features as $x$ , the formula is as follows:

$$
G ( x ) = S o f t m a x ( W \cdot x )
$$

where $W \in { \mathbf { R } } ^ { d \times ( N + 1 ) }$ is the trainable matrix, $d$ represents the dimensionality of the input, $N$ is the number of LoRA experts, The term $N + 1$ means the incorporation of an additional neuron, representing the probability that $x$ is toolunrelated data, which is used to suppress the model’s call to external expert weights. The output of MOLoRA is:

$$
h = W _ { 0 } x + \sum _ { i = 0 } ^ { N } G ( x ) [ i + 1 ] B _ { i } A _ { i } x
$$

where $G ( x ) [ i ]$ is the $i$ -th element in router output, and the $G ( x ) [ 0 ]$ is not used in forward process. And matrices $A \in$ $\mathbb { R } ^ { r \times k }$ and $B \in \mathbb { R } ^ { d \times r }$ are the trainable linear of LoRA experts, where $r \ll m i n ( d , k )$ is the rank of LoRA experts.

To achieve the separation of inputs, we divide the data into tool-related and unrelated with different task types. And we modify the Localized Balancing Constraint proposed by Dou et al. (2024), which softly constrains the experts to focus on different tasks. We assign the importance score to each neuron to control the weighting of different experts, the rule of importance matrix $I$ is:

$$
I _ { n } = \left\{ \begin{array} { l l } { \displaystyle [ 1 + \delta , \underbrace { 1 - \delta , \cdots , 1 - \delta } _ { N } ] \quad \mathrm { i f } x _ { n } \in \mathcal { D } _ { t o o l } } \\ { \displaystyle [ 1 - \delta , \underbrace { 1 + \delta , \cdots , 1 + \delta } _ { N } ] \quad \mathrm { o t h e r w i s e } } \end{array} \right.
$$

Here, $\delta \in [ 0 , 1 ]$ controls the degree of imbalance between router output, $x _ { n }$ is the $n$ -th input.

We weigh the importance matrix $I$ and the output of the gating function $G$ , denoted as $Z = I \odot G$ . And the routing loss $\mathcal { L } _ { r }$ of router network is:

$$
\mathcal { L } _ { r } = \frac { \sigma ^ { 2 } ( Z ) } { \mu ( Z ) }
$$

where $\sigma ^ { 2 } ( Z )$ and $\mu ( Z )$ are the variance and mean of $Z$ . $\mathcal { L } _ { r }$ ensure that for tool-unrelated data $x$ , $G ( x ) [ 0 ]$ is relatively large, and vice versa, $G ( x ) [ 0 ]$ is relatively small. adjusting the weight of LoRA experts based on the data types.

# 3.2 Unimportant Components Optimization

For unimportant components, as updating these parameters has a relatively less impact on the model’s general capabilities, we utilize full parameter fine-tuning to stimulate the model’s tool-invoking ability.

We select unimportant components for general abilities by averaging the importance of different tasks, defined as general abilities importance $\mathcal { C } _ { h } ( \theta )$ . For a general abilities set $T$ , the formula is as follows:

$$
\mathcal { C } _ { h } ( \theta ) = \sum _ { t \in T } \frac { \mathcal { T } _ { h } ( \mathcal { D } _ { t } , \theta ) } { \sum _ { i \in H } \mathcal { T } _ { i } ( \mathcal { D } _ { t } , \theta ) }
$$

Components with lower $\mathcal { C } _ { h } ( \theta )$ are selected. These components are not frozen during the training process.

# 3.3 Training Strategy

In summary, the overall loss function of CITI is:

$$
\mathcal { L } = - \sum _ { ( x , y ) \in \mathcal { D } } \log P \left( y \mid x ; \theta _ { b } , \theta _ { e } , \theta _ { r } \right) + \beta \cdot \mathcal { L } _ { r }
$$

Where $\theta _ { b }$ is the parameters of the backbone, $\theta _ { e }$ represents the MOLoRA experts, and $\theta _ { r }$ is the router network. $\mathcal { D }$ denotes the mixture of tool learning dataset and other instructions sampled from general ability datasets.

To optimize our model effectively, we have devised a three-stage training strategy, delineated as follows:

Stage-1 RP: We integrate MOLoRA based on the importance of component gradients. During this phase, the core model parameters and LoRA experts are kept constant, with only $\theta _ { r }$ being subjected to training.

Stage-2 MI: The $\theta _ { e }$ and $\theta _ { r }$ are subjected to training. We fine-tune the parameters in LoRA experts.

Stage-3 UCO: Here, we immobilize the parameters of the MOLoRA adapters, including $\theta _ { e }$ and $\theta _ { r }$ , as well as the important parameters in the backbone identified by importance ranking. We then proceed to train a select subset of the backbone parameters chosen by $\mathcal { C } _ { h }$ .

# 4 Experiments

# 4.1 Datasets

We conduct the experiments on two tool learning benchmarks: API-Bank (Li et al. 2023b) and ToolAlpaca (Tang et al. 2023). Additionally, we assess general abilities through experiments on the datasets mentioned above.

# 4.2 Implementation Details

We apply MOLoRA to the top $20 \%$ components with the highest importance ranking sorted by $\mathcal { M } _ { h }$ and fine-tune down $10 \%$ components with lowest importance ranking sorted by $\mathcal { C } _ { h }$ .

# 4.3 Evaluation Metrics

For the API-Bank dataset, we follow the evaluation metrics proposed by Li et al.(2023b), including the Correctness of API calls and the ROUGE-L to test quality of the responses. For the ToolAlpaca dataset, we utilize the GPT-4 to evaluate the tool-invoking process in the real-world testing subset and follow the original metrics: Procedure: The correctness of tool utilizing procedure, Response: The quality of final response, Overall: Whether procedure and response are both correct.

# 4.4 Baselines

We conduct experiments with excellent models: MetaLlama-3-8B-Instruct, Phi-3-mini-128k-instruct and Mistral-7B-Instruct-v0.2. We compare our method with the following baselines: (1) FT: fine-tune the model with full parameters on the tool learning dataset; (2) LoRA: fine-tune the model with LoRA adapters on the tool learning dataset. The training data is without data mixing for our baselines.

Table 3: The overall results on the dataset API-Bank.   

<html><body><table><tr><td>Model</td><td colspan="2">Tool Ability</td><td colspan="4">General Ability</td></tr><tr><td></td><td>C</td><td>R</td><td>GSM</td><td>HE</td><td>TQA</td><td>MT</td></tr><tr><td>Llama-3</td><td>33.25</td><td>14.67</td><td>79.45</td><td>59.15</td><td>64.82</td><td>80.13</td></tr><tr><td>FT</td><td>56.48</td><td>18.71</td><td>26.38</td><td>0.00</td><td>50.07</td><td>33.88</td></tr><tr><td>LoRA</td><td>65.98</td><td>38.62</td><td>76.88</td><td>53.05</td><td>64.84</td><td>68.00</td></tr><tr><td>CITI</td><td>58.92</td><td>34.51</td><td>77.33</td><td>54.88</td><td>64.69</td><td>76.94</td></tr><tr><td>Phi-3</td><td>44.80</td><td>29.40</td><td>82.71</td><td>60.37</td><td>53.08</td><td>81.28</td></tr><tr><td>FT</td><td>61.75</td><td>35.23</td><td>73.09</td><td>51.83</td><td>53.32</td><td>66.81</td></tr><tr><td>LoRA</td><td>57.00</td><td>35.19</td><td>75.82</td><td>0.00</td><td>52.34</td><td>68.44</td></tr><tr><td>CITI</td><td>58.02</td><td>37.21</td><td>77.86</td><td>61.59</td><td>55.23</td><td>72.19</td></tr><tr><td>Mistral</td><td>53.66</td><td>32.39</td><td>44.81</td><td>35.37</td><td>59.21</td><td>75.66</td></tr><tr><td>FT</td><td>47.88</td><td>2.18</td><td>2.05</td><td>0.00</td><td>13.50</td><td>10.63</td></tr><tr><td>LoRA</td><td>58.66</td><td>43.31</td><td>46.32</td><td>37.80</td><td>61.54</td><td>68.31</td></tr><tr><td>CITI</td><td>62.00</td><td>33.53</td><td>53.30</td><td>38.41</td><td>62.77</td><td>69.28</td></tr></table></body></html>

<html><body><table><tr><td>Model</td><td colspan="3">Tool Ability P R</td><td colspan="3">General Ability TQA</td></tr><tr><td>Llama-3</td><td>14.91</td><td></td><td>0</td><td>GSM 79.45</td><td>HE</td><td>MT 80.13</td></tr><tr><td>FT</td><td>65.79</td><td>21.05 59.65</td><td>12.28 55.26</td><td>30.33</td><td>59.15 64.82 0.61 55.97</td><td>50.81</td></tr><tr><td>LoRA</td><td>76.32</td><td>74.56</td><td>71.05</td><td>73.46</td><td>31.71 65.18</td><td>73.75</td></tr><tr><td>CITI</td><td>76.32</td><td></td><td></td><td>75.89</td><td></td><td></td></tr><tr><td></td><td></td><td>78.07</td><td>72.81</td><td></td><td>56.71 64.82</td><td>76.67</td></tr><tr><td>Phi-3 FT</td><td>2.63</td><td>1.75</td><td>1.75</td><td>82.71</td><td>60.37 53.08</td><td>81.28</td></tr><tr><td></td><td>67.54</td><td>68.42</td><td>64.04</td><td>71.95</td><td>54.88 53.58</td><td>65.06</td></tr><tr><td>LoRA</td><td>70.18</td><td>71.05</td><td>66.67</td><td>74.91</td><td>0.61 53.04</td><td>73.22</td></tr><tr><td>CITI</td><td>64.04</td><td>72.81</td><td>61.40</td><td>77.33</td><td>62.20 55.27</td><td>72.59</td></tr><tr><td>Mistral</td><td>14.04</td><td>14.91</td><td>13.16</td><td>44.81</td><td>35.37 59.21</td><td>75.66</td></tr><tr><td>FT</td><td>42.98</td><td>41.23</td><td>30.70</td><td>4.85</td><td>0.00</td><td>12.10 15.69</td></tr><tr><td>LoRA</td><td>72.81</td><td>73.68</td><td>67.54</td><td>44.81</td><td>34.76 62.99</td><td>68.19</td></tr><tr><td>CITI</td><td>78.07</td><td>76.32</td><td>71.05</td><td>51.48</td><td>36.59 63.21</td><td>71.44</td></tr></table></body></html>

Table 4: The overall results on the dataset ToolAlpaca.

# 4.5 Overall Results

As shown in Table 3 and Table 4, CITI demonstrates effective performance in preserving general abilities in two datasets. For instance, it achieves a $7 . 5 9 \%$ improvement compared to LoRA and $3 1 . 9 5 \%$ compared to FT in APIBank for general abilities on average. For tool-utilizing ability, CITI shows competitive results in two datasets compared to the baselines. Noticing that in the dataset API-Bank, CITI has poorer tool-invoking ability compared to baselines (e.g. ROUGE-L score in Mistral). We suggest there are two possible reasons: (1) The $\lambda$ in MOLoRA may not be optimal, and insufficient training data may result in underfitting or overfitting of LoRA adapters. (2) The evaluation metrics are not comprehensive. API-Bank asks the model to generate tool calls or responses based on conversation history, but the path for the model to obtain final correct answers is not unique.

Additionally, we notice that LoRA outperforms full parameter fine-tuning in both tool utilization and general abilities in most cases. This may be because full parameter finetuning can easily lead to overfitting.

Table 5: The ablation results on the dataset API-Bank.   

<html><body><table><tr><td rowspan="2">Model</td><td colspan="2">Tool Ability</td><td colspan="4">General Ability</td></tr><tr><td>C</td><td>R</td><td>GSK</td><td>HE</td><td>TQA</td><td>MT</td></tr><tr><td>FT+DM LoRA+DM</td><td>56.61 63.03</td><td>39.67 38.66</td><td>54.44</td><td>29.88</td><td>45.28</td><td>60.41 74.31</td></tr><tr><td rowspan="4">UCO RP +MI w/oMOLoRA</td><td>59.31</td><td></td><td>75.13</td><td>51.22</td><td>63.96</td><td></td></tr><tr><td></td><td>38.63</td><td>79.68</td><td>54.88</td><td>64.07</td><td>75.09</td></tr><tr><td>57.89</td><td>35.50</td><td>77.94</td><td>54.88</td><td>64.74</td><td>76.56</td></tr><tr><td>61.87</td><td>39.97</td><td>73.09</td><td>54.27</td><td>63.09</td><td>77.91</td></tr><tr><td>w/o Lr CITI (ours)</td><td>61.49 58.92</td><td>38.29 34.51</td><td>75.28 77.33</td><td>57.93 54.88</td><td>63.90</td><td>76.81 76.94</td></tr><tr><td>w/o RP</td><td>58.79</td><td></td><td>76.12</td><td></td><td>64.69</td><td></td></tr><tr><td></td><td></td><td>38.13</td><td></td><td>52.44</td><td>63.35</td><td>75.22</td></tr></table></body></html>

<html><body><table><tr><td rowspan="2">Model</td><td colspan="3">Tool Ability</td><td colspan="3">General Ability</td></tr><tr><td>P</td><td>R</td><td>0</td><td>GSM</td><td>HE TQA</td><td>MT</td></tr><tr><td>FT+DM LoRA+DM</td><td>60.53 73.68</td><td>63.16 73.68</td><td>57.02 68.42</td><td>52.92</td><td>23.78</td><td>50.58 58.94</td></tr><tr><td>UCO</td><td>69.30</td><td>68.42</td><td>64.04</td><td>74.53 80.52</td><td>53.66 55.49</td><td>65.06 74.94 64.06 73.63</td></tr><tr><td>RP+MI w/o MOLoRA</td><td>76.32</td><td>75.44</td><td>69.30</td><td>74.68</td><td>54.27</td><td>64.49 77.63</td></tr><tr><td></td><td></td><td>70.18 71.05 65.79</td><td></td><td>73.09</td><td></td><td>55.49 64.31 77.31</td></tr><tr><td>w/o Lr</td><td>71.05 71.93</td><td></td><td>64.91</td><td></td><td></td><td></td></tr><tr><td>CITI (ours)</td><td>76.32</td><td></td><td></td><td>77.48</td><td></td><td>55.49 65.00 77.38</td></tr><tr><td></td><td></td><td>78.07</td><td>72.81</td><td>75.89</td><td>56.71</td><td>64.82 76.67</td></tr><tr><td>w/o RP</td><td>70.18</td><td>68.42</td><td>64.91</td><td>77.48</td><td>54.27</td><td>64.95 76.81</td></tr></table></body></html>

Table 6: The ablation results on the dataset ToolAlpaca.

# 4.6 Ablation Studies

To demonstrate the effectiveness of our approach, we conduct ablation studies on model Meta-Llama-3-8B-Instruct as shown in Table 5 and Table 6. Where w/o MOLoRA means replacing the MOLoRA by LoRA adapter, w/o $\mathcal { L } _ { r }$ means applying normal MOLoRA without additional neuron and importance matrix $I$ in the router. DM represents data mixing in the training data. The experimental results show that compared to FT and LoRA with data mixing, CITI still exhibits advantages in general abilities. Additionally, we analyze the contributions of each module in CITI. Compared to baselines, MI and UCO modules have shown distinct advantages in maintaining the model’s general abilities separately.

We can find that: (1) The UCO module demonstrates that fine-tuning the unimportant components selectively can enhance the model’s capability to utilize new tools while maintaining its original performance, especially on GSM8K within two datasets. (2) Comparing $\mathsf { R P } + \mathsf { M I }$ with w/o MOLoRA, we discover that by incorporating MOLoRA, rather than simply adding LoRA adapter, the decline in general abilities can be alleviated. (3) The results of only w/o $\mathcal { L } _ { r }$ outperform w/o MOLoRA in most cases, further indicating the MOLoRA structure is effective. Additionally, we find that some results in w/o $\mathcal { L } _ { r }$ are better than $\mathrm { R P + M I }$ . We deduce this is because the adapters absorb general knowledge within the general instructions in mixed training set, but they play little impact in MI because the router separates them into frozen LLM backbone during training stage. (4) The results of w/o RP demonstrate that Router Pre-training offers benefits, especially in tool-utilizing in ToolAlpaca and general performance in API-Bank, indicating that pre-training the router network may accelerate model convergence especially when training data is constrained. (5) By applying UCF to further fine-tune the model following MI, CITI can maintain or even increase its general abilities on two datasets compared to $\mathsf { R P + M I }$ . And the tool-utilizing ability has further significantly improved on the dataset ToolAlpaca.

# 5 Related Work

# 5.1 Tool Learning

Tool Learning aims to enable LLMs to use external tools to enhance models’ ability, which shows great power in dealing with various tasks (Qin et al. 2023; Qu et al. 2024). Finetuning technology (Tang et al. 2023; Qin et al. 2024; Wang et al. 2024) is a primary approach to learn tool-utilizing pattern, which can solidify knowledge into models comparing to in-context learning (Shen et al. 2023; Wu et al. 2023). In order to enable the model to invoke tools while maintaining its original performance, we introduce CITI, a framework to balance the conflict.

# 5.2 Components Importance Analysis

It is important to evaluate the importance of parameters in model pruning (Zhang et al. 2022) and works related to interpretability (Michel, Levy, and Neubig 2019; Zhang et al. 2024; Jin et al. 2024). The gradient-based importance scores (Bansal et al. 2023) assesses the importance of parameters by measuring the resultant impact on the loss function following the parameter’s elimination. In our work, We use importance scores to identify and select components of toolutilizing and general tasks.

# 5.3 Mixture Of LoRA

Recently, there are substantial works utilizing Mixture-ofLoRA Experts (MOLoRA) architecture to improve the performance (Dou et al. 2024; Wu, Huang, and Wei 2024; Yang et al. 2024). These works combine the Mixture-ofExperts (MOE) (Fedus, Zoph, and Shazeer 2022; Shazeer et al. 2017) architecture with Low-Rank Adaptation (LoRA) (Hu et al. 2022; Wu, Huang, and Wei 2024; Gou et al. 2024). MOLoRA keeps the backbone networks frozen and finetunes the adapters, achieving significant performance while alleviating catastrophic forgetting (Zhao et al. 2024; Li et al. 2024; Dou et al. 2024). We modify the router network to distinguish the tool-related and tool-unrelated inputs.

# 6 Conclusion

In this study, we explore the trade-offs between the model’s tool-utilizing ability and its general task performance following fine-tuning on a tool learning dataset. We conduct a deep analysis of this phenomenon from hidden representation and components perspectives and propose a new approach CITI, which integrates MOLoRA adapters into the critical components of the model and selectively fine-tunes a limited number of parameters of less critical components within the backbone of the model. Through extensive experiments, we demonstrate the validity of CITI in acquiring tool-related knowledge without excessively harming its performance on general tasks.