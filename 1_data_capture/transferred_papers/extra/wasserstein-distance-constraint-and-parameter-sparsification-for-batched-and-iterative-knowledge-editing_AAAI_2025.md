# Wasserstein Distance Constraint and Parameter Sparsification for Batched and Iterative Knowledge Editing

Shanbao Qiao, Xuebing Liu, Seung-Hoon Na\*

Center for Advanced Image and Information Technology, Department of Computer Science and Artificial Intelligence, Jeonbuk National University joe,liuxuebing,nash @jbnu.ac.kr

# Abstract

Model knowledge editing has become a widely researched topic because it enables the efficient and rapid injection of new knowledge into language models, as well as the correction of erroneous or outdated knowledge. Existing model knowledge editing methods are typically classified into two categories: single-instance sequential editing and massive one-time editing. However, the batched and iterative editing method is better aligned with model updating patterns in practical applications. In this work, we examine the performance of parameter-update-based models on a novel batched iterative editing benchmark. Our findings indicate that as the number of editing iterations increases, the accumulation of updated parameters leads to a greater shift in the model’s parameter distribution, making it harder to maintain both the editing performance and the stability of the model. To address this degradation issue, we propose two methods: the Wasserstein distance constraint and the parameter update sparsification, where the Wasserstein distance constraint optimizes the transition of parameter distribution before and after editing, and the parameter update sparsification significantly reduces the number of updated parameters, thereby alleviating instability in the parameter distribution caused by their accumulation over iterations. Our methods are generally applicable to various parameter-update-based knowledge editing models. Experiments on the zsRE and CounterFact datasets demonstrate that our methods improve editing performance and enhance the stability of batched iterative editing in the later stages across different models.

Code — https://github.com/JoveReCode/WDSP.git

# Introduction

Large language models (LLMs) have demonstrated strong capabilities across a wide range of natural language processing (NLP) tasks (Touvron et al. 2023; OpenAI 2023; Petroni et al. 2020). However, as world knowledge continuously evolves and changes, the knowledge stored in the model parameters can become outdated over time (Onoe et al. 2022; Dhingra et al. 2022; Lisˇka et al. 2022). Additionally, the models may inherently contain erroneous knowledge (Zhao et al. 2023; Ji et al. 2023; Lazaridou et al. 2021; Agarwal and Nenkova 2022; Gallegos et al. 2023), making iterative updates to LLMs both necessary and crucial. Retraining or fine-tuning LLMs requires substantial computational resources and time. To achieve timeefficient model updates, the knowledge editing paradigm has emerged, gaining increasing research attention. Knowledge editing aims to timely inject new knowledge or correct erroneous knowledge by updating a small subset of model parameters or leveraging in-context learning (ICL) techniques (Brown et al. 2020; Dong et al. 2022). Developing efficient and stable knowledge editing methods is critical for the future maintenance of language models.

Existing knowledge editing methods can be broadly categorized into parameter updating methods, meta-learningbased methods, and ICL-based methods. Parameter updating approaches (Cao, Aziz, and Titov 2021; Meng et al. 2022a,b; Li et al. 2023) typically leverage mechanistic interpretability (MI) to identify specific layers in the model that store new knowledge and rewrite the relevant parameters. Meta-learning methods (Mitchell et al. 2022a; Tan, Zhang, and Fu 2024) involve training a hypernetwork to alter the model’s output predictions, whereas ICL-based methods (Zheng et al. 2023; Zhong et al. 2023) temporarily adjust the model’s output by appending the constructed prompts to the input query. Most existing work on editing focuses on single-instance sequential editing or massive one-time editing. However, in practice, batched and iterative editing is preferred in the maintenance of LLMs (e.g., the model’s knowledge needs to be updated weekly or monthly as world knowledge changes rapidly). This primarily requires that editing methods are performed at the batch level. In this regard, ICL-based methods are less practical for batch execution, as they require lengthy prompts for each editing instance, which significantly reduces inference efficiency. Moreover, the effects of knowledge editing achieved through ICL are only temporary, lasting only within the current model run, and do not genuinely integrate the knowledge into the model. Meta-learning-based methods also train mainly for individual instances at a time, which limits their ability to achieve satisfactory performance in batched editing.

In contrast, parameter updating methods can perform massive knowledge editing in a single step, with the number of edits meeting the typical requirements for model updates and maintenance. However, can parameter updating methods achieve iterative and stable editing? We found this to be a challenging issue for current popular parameter updating methods. We tested representative parameter updating methods, namely MEMIT(Meng et al. 2022b) and PMET(Li et al. 2023), on a benchmark involving batched and iterative editing. Our findings indicate that as the number of iterations increases, the model is prone to rapid deterioration. This is mainly due to the accumulation of parameters that need to be updated with each iteration, which leads to increasingly significant changes in the parameter distribution of the model, thus resulting in instability and collapse. To address the challenging task of batch and iterative knowledge editing, we propose methods based on Wasserstein distance constraint (Rubner, Tomasi, and Guibas 2000) and parameter update sparsification, aiming to mitigate the accumulation of adverse effects from parameter rewriting (that is, parameter updating) and to maintain the long-term stability of the model’s parameter distribution. Wasserstein distance is a metric that measures the similarity between two probability distributions by calculating the minimal cost required to transform one into the other. We constrain the learning process of parameter updates by establishing a Wasserstein distance loss between the model’s initial parameter distribution and the post-edited parameter distribution. This approach aims to minimize changes in the model’s parameter distribution after the knowledge editing process, thereby maintaining stability. Additionally, in parameter updating methods, not all parameter updates contribute to editing performance. Parameter update sparsification prunes the parameters with small magnitudes and applies random dropout to make the updates sparser in each editing batch. We found that discarding approximately $50 \%$ of the parameters does not degrade the editing performance (and may even improve it), which reduces the accumulation of new parameters in each batch, allowing the model to remain effectively updateable after multiple iterative editing cycles. The contributions of our work are summarized as follows:

1. We examine the performance of popular parameter updating methods for knowledge editing methods on a batched iterative editing benchmark, revealing that these methods often lose editing capability or cause the model to rapidly collapse after multiple iterations. To address this, we design a loss function based on the Wasserstein distance to constrain distributional changes during the parameter update process, thereby maintaining the stability of the model’s parameter distribution.

2. We further apply sparsification to the parameters that need updating, discarding redundant ones that do not contribute to editing performance. This significantly reduces the number of parameters to be updated in each iteration, thereby mitigating changes in the model’s parameter distribution caused by the accumulation of updates. This approach can be used independently or in combination with the Wasserstein distance constraint, both of which help prevent the model from collapsing rapidly after multiple editing iterations, while also improving knowledge editing performance.

Rick Braun performs Jazz. The next Olympics will be held in Paris. B-17 Flying Fortress is produced by IBM. LLM The next Olympics will be held in Paris. → Los Angeles. Editing B-17 Flying Fortress is produced by IBM. → Boeing. Rick Braun performs Jazz. The next Olympics will be held in Los Angeles. B-17 Flying Fortress is produced by Boeing. LLM

3. Extensive experiments on the zsRE and CounterFact datasets demonstrate the effectiveness of our proposed methods.

# Related Works

Knowledge editing methods can be categorized into several types from a technical perspective, including meta-learning, parameter updating1, and in-context learning (ICL) methods. However, from an alternative perspective on their editing capacity, they can be classified into two main categories: single-instance editing and batched (or massive) editing. Single-instance editing is typically performed sequentially, and if model parameters are not restored, it can edit a few dozen pieces of knowledge at most. In contrast, batch or massive editing can handle hundreds to thousands of edits simultaneously, resulting in significant differences in their applicability.

Single Editing Methods MEND (Mitchell et al. 2022a) trains a hyper-network to convert the fine-tuning gradient into a simplified representation using low-rank decomposition, and then applies this updated gradient to the model. SERAC (Mitchell et al. 2022b) stores the edits in a separate memory and uses a retrieval-augmented counterfactual model to incorporate the edits to influence the model’s predictions. ROME (Meng et al. 2022a) locates the model parameters corresponding to new factual knowledge and updates the key-value pairs in the MLP module with newly computed residual vectors. T-Patcher (Huang et al. 2023) adds a single neuron to the final layer of the feed-forward network (FNN) to handle a specific edit request.

GRACE (Hartvigsen et al. 2023) is a recent work towards lifelong knowledge editing, which employs a discrete codebook to store editing information and uses a deferral mechanism to determine whether to apply it during inference. This enables knowledge editing without modifying the model’s internal parameters, thus avoiding potential model degradation. While the goal of this work aligns closely with ours, there are notable differences in the underlying concepts and applicable scenarios. GRACE is designed for instance-level sequential editing scenarios (i.e., editing one piece of knowledge at a time), whereas our approach aims to enable the sequential editing of batches of new knowledge. Furthermore, GRACE requires maintaining an external codebook, the size of which grows as the amount of edited knowledge increases. In contrast, our work focuses on achieving knowledge editing solely through the model’s own parameters, ensuring stable performance in extensive knowledge editing scenarios without introducing any external components.

Batched/Massive Editing Methods To enable massive editing, MEMIT (Meng et al. 2022b) improves upon ROME by distributing the computed residual vectors across multiple model layers. MALMEN (Tan, Zhang, and $\mathrm { F u } ~ 2 0 2 4 ,$ ) extends MEND to handle massive editing tasks by aggregating batched parameter updates into a single update, thereby achieving improved scalability. PMET (Li et al. 2023) builds on MEMIT by considering the knowledge-storing role of the multi-head self-attention (MHSA) layers, thus preventing the overestimation of the parameter updates required for the FFN layers.

In our tested batched iterative editing benchmark, since single-instance editing methods cannot handle such tasks due to their limited capacity, we focus primarily on researching and testing methods for massive editing.

# Method

In this section, we describe the knowledge editing task and introduce our method for facilitating batched and iterative editing. An overview of our method is illustrated in Figure 2.

# Task Definition

The objective of the knowledge editing task is to inject new factual knowledge into the model or modify outdated or incorrect knowledge, while preserving the existing knowledge. An example is shown in Figure 1. Suppose that $\mathcal { M }$ is an auto-regressive language model with parameters $\theta$ . Given factual knowledge represented by a tuple $( s , r , o )$ , where $s$ , $r$ , and $o$ denote the subject, relation and object of the fact, respectively (e.g., $s$ :”B-17 Flying Fortress”, $r$ : ”is produced by”, $o$ : ”IBM”), the language model $\mathcal { M } _ { \theta }$ takes the prefix $( s , r )$ as the input and predicts the answer $o$ through the decoding process:

$$
o = \arg \operatorname* { m a x } _ { x = ( s , r ) } P _ { \mathcal { M } _ { \theta } } ( y | x )
$$

where $P _ { \mathcal { M } _ { \theta } } ( y | x )$ denotes the generative probability of $y$ . In cases where the factual knowledge needs to be edited, given the prefix $( s , r )$ , the knowledge editing methods should steer the model’s prediction towards the new answer $o ^ { * }$ (e.g., $o ^ { * }$ :”Boeing”) by modifying the model parameters to $\theta ^ { * }$ or by using in-context learning prompts, i.e.:

$$
o ^ { * } = \arg \operatorname* { m a x } _ { x = ( s , r ) } P _ { \mathcal { M } _ { \theta ^ { * } } } ( y | x )
$$

Single-instance knowledge editing methods (Mitchell et al. 2022a; Meng et al. 2022a; Huang et al. 2023) perform such editing process sequentially for different facts, with some requiring the model parameters to be restored to their initial state between each edit, which makes it challenging to handle large-scale editing scenarios.

For the batched or massive editing task, the editing requests are given as a set of new knowledge $\mathcal { E } = \left\{ e _ { i } \right\} _ { i = 1 } ^ { N }$ , where $e _ { i } \stackrel { - } { = } ( s _ { i } , r _ { i } , o _ { i } ^ { * } )$ . The editing methods compute the parameter updates (or changes) for all new knowledge in the editing set and apply them to the model simultaneously. In this work, we further investigate whether these methods can be performed iteratively without restoring the updated parameters in each iteration, thereby enabling batched and iterative editing that is suitable for practical model maintenance. To formally describe this editing process, consider a collection of editing request sets with batch size $B$ denoted as $\left\{ \mathcal { E } _ { j } \right\} _ { j = 1 } ^ { T }$ , where $\bar { \mathcal { E } } _ { j } = \left\{ e _ { i } ^ { j } \right\} _ { i = 1 } ^ { B }$ , and $T$ is the total number of iterations for the test. Given an editing method $\mathcal { G }$ , the model parameters are updated as follows:

$$
\mathcal { M } _ { \theta _ { t } ^ { * } } = \mathcal { G } ( \mathcal { M } _ { \theta _ { t - 1 } ^ { * } } , \mathcal { E } _ { t } ) , t \in \{ 1 , 2 , . . . , T \}
$$

where $\theta _ { 0 } ^ { * } = \theta$ represents the initial model parameters. We evaluate the editing performance for each batch of edits.

# Wasserstein Distance Constraint

Although parameter updating methods can achieve batched and permanent knowledge editing, they face challenges in iterative editing tasks. For simplicity, we describe the process of update-based knowledge editing methods (i.e., parameter updating methods) as follows: compute the residual vectors $\delta$ (parameter updates) needed for the model to predict the new facts $o ^ { * }$ , and then add $\delta$ to the hidden states of the previous layer, which can be simply represented as $\theta ^ { * } = \theta + \delta$ . For the overall model update, compute the update matrix $\Delta$ using the collection of residual vectors $\delta$ and the model’s FFN keys, then add the update matrix $\Delta$ to the original model’s FFN weights $W$ , i.e. $W ^ { * } = W + \Delta$ . In the iterative editing process, each update accumulates a new $\Delta$ on top of the previous iteration, causing the model parameter distribution to undergo increasing changes as the number of iterations grows. In the later stages of the iteration, this can rapidly deteriorate performance (as will be demonstrated by the experimental data in subsequent sections). To address this issue, we introduce methods based on Wasserstein distance (Rubner, Tomasi, and Guibas 2000) constraints and parameter sparsification, with the goal of maintaining the stability of the model’s parameter distribution and preventing degradation.

![](images/6eb97d986990c0a3231223d47703c38f831535665c60812fc240a8a5be6b1308.jpg)  
Figure 2: An overview of the process of our proposed method. Given a language model $\mathcal { M }$ with its initial parameters $\theta$ , parameter updating methods for knowledge editing compute residual values $\delta$ for the new requested editing knowledge and add them to the initial parameters $\theta$ to achieve the editing. This results in significant changes to the parameter distribution, and after multiple iterations, it can cause severe instability or even lead to model collapse. Our proposed method consists of two key processes. First, we apply a Wasserstein distance constraint $( \mathrm { W D } ( \cdot ) )$ between the updated and initial parameter distributions to obtain smoothed residual values $\delta _ { W } = \mathrm { W D } ( \delta )$ . Second, we apply parameter (update) sparsification, which performs pruning and dropout to sparsify the $\delta _ { W }$ , resulting in $\dot { \delta } _ { W S } = \mathrm { S P } ( \delta _ { W } )$ . The final $\delta _ { W S }$ enables smoothed parameter updates for batched editing, which significantly helps maintain model stability during subsequent editing iterations.

Wasserstein distance is a metric that measures the discrepancy between two probability distributions. It quantifies the minimum cost required to transport one distribution into another, taking into account both the distances and the amount of mass being moved. Given two probability measures $\mu$ and $\nu$ , the Wasserstein $p$ -distance2 between them is defined as:

$$
W _ { p } ( \mu , \nu ) = \operatorname* { i n f } _ { \gamma \in \Pi ( \mu , \nu ) } \mathbb { E } _ { ( x , y ) \sim \gamma } \left[ | | x - y | | _ { p } ^ { p } \right]
$$

where $\Pi ( \mu , \nu )$ is the set of all joint distributions whose marginal distributions are $\mu$ and $\nu$ . In our task, we aim to constrain the Wasserstein distance between the initial model parameters $\theta$ and the updated model parameters $\theta ^ { * }$ . However, computing $W ( \theta , \theta _ { t } ^ { * } )$ is intractable due to the high dimensionality of the model parameters. We use the Sinkhorn algorithm (Sinkhorn and Knopp 1967; Luise et al. 2018) to compute the approximate solution, by introducing entropy regularization, where the computation of Wasserstein distance is transformed into the convex optimization problem. With the transport plan $\gamma$ and the regularization term being differentiable, the gradient of the approximated Wasserstein distance can be efficiently computed by back-propagating through the iterative updates of the Sinkhorn algorithm. The Sinkhorn iteration can be expressed as follows:

$$
\begin{array} { r } { \mathbf { u } ^ { ( i + 1 ) } = \frac { \theta } { \mathbf { K } \mathbf { v } ^ { ( \mathbf { i } ) } } } \\ { \mathbf { v } ^ { ( i + 1 ) } = \frac { \theta ^ { * } } { \mathbf { K } ^ { T } \mathbf { u } ^ { ( i + 1 ) } } } \end{array}
$$

where $\mathbf { v }$ and $\mathbf { u }$ start as uniform distributions and iterate to convergence. The kernel matrix ${ \mathbf { K } } = e ^ { \frac { - { \mathbf { C } } } { \epsilon } }$ is computed by the $L _ { 2 }$ distance matrix $\mathbf { C }$ , where $\mathbf { C _ { i , j } } = | | \beta _ { i } - \zeta _ { j } | | _ { 2 } ^ { 2 } , ( \beta \in$ $\theta , \zeta \in \theta ^ { * } )$ , $\epsilon$ is a hyper-parameter that controls the intensity of entropy regularization. After the Sinkhorn iteration converges, the solution $\mathbf { P }$ can be computed by:

$$
\mathbf { P } = \mathrm { d i a g } ( \mathbf { u } ) \mathbf { K } \mathrm { d i a g } ( \mathbf { v } )
$$

here $\mathbf { P }$ is the matrix of $\gamma ( x , y )$ , which specifies the optimal transport plan between two distributions, enabling the computation of the Wasserstein distance $W ( \theta , \theta ^ { * } )$ .

Given the computed Wasserstein distance, we use it as a loss function during the computing of update parameters $\delta$ (the residual vector, which can be represented as $\delta = \theta ^ { * } - \theta _ { , } ^ { \cdot }$ , and it is defined as:

$$
{ \cal L } _ { W D } = { \cal W } ( \theta , \theta ^ { * } ) .
$$

In the update-based knowledge editing methods such as MEMIT (Meng et al. 2022b) or PMET (Li et al. 2023), the

![](images/f7c4f2bce04acaed895e21641a8ee5f66c8ec0ff384bc8c1160b2cd52126f899.jpg)  
Figure 3: Detailed performance of Efficacy, Generalization and Locality on the zsRE dataset (batch size $= 5 0$ ). The vertical axis represents the metric scores for the current editing iteration, while the horizontal axis represents the number of iterations. SP denotes the parameter sparsification, WD denotes the Wasserstein distance constraint, and $\mathbf { W D + S P }$ denotes the combined method.

residual vector $\delta$ for each editing request is computed by the negative log-likelihood loss3:

$$
L _ { N L L } = - \log \mathcal { P } _ { \mathcal { M } ( \theta + \delta ) } [ o ^ { * } | ( s , r ) ]
$$

Along with the Wasserstein distance loss, our loss function for computing the residual vector $\delta$ is defined as follows:

$$
L = L _ { N L L } + \lambda ( L _ { W D } )
$$

Unlike conventional hyper-parameters, $\lambda$ here is a dynamic scaling function. The value of the Wasserstein distance varies across different samples, with some results being quite large, which could lead to an overemphasis of the Wasserstein distance constraint. Additionally, as iterative editing progresses and the parameter distribution changes, the Wasserstein distance between the updated and the original parameter distributions may also increase. To prevent the Wasserstein distance from becoming excessively large and dominating the overall loss function, we introduce a dynamic scaling factor $\lambda$ to adjust the contribution of the Wasserstein loss. Since the value of $L _ { N L L }$ is relatively stable, we dynamically normalize the ${ \cal L } _ { W D }$ value to match the scale of $L _ { N L L }$ .

# Parameter Update Sparsification

Given the loss function in equation (9), for a batched editing with batch size $B$ , the residual vector $\delta$ for each editing request is computed and collected as $R = \left\{ \delta _ { i } \right\} _ { i = 1 } ^ { B }$ , and $R$ is

then used to compute the update matrix $\Delta$ , which is applied to the model layers to be updated:

$$
\Delta \gets R K ^ { T } ( C + K K ^ { T } ) ^ { - 1 }
$$

where $K$ denotes the FFN keys of the model layer and $C$ is a constant proportional to the uncentered covariance of the pre-existing keys (pre-computed from a sample of Wikipedia text, Meng et al. (2022a)). With the overall model update defined as $W ^ { * } = W + \Delta$ , the sparsification of the update matrix $\Delta$ has been used in recent work $\mathrm { G u }$ et al. 2024) to preserve the general capabilities of the model after editing. We consider whether the sparsification of $\Delta$ can be used to preserve the model’s capabilities after multiple iterations of editing. However, directly sparsifying the update matrix $\Delta$ incurs significant computational costs, as it shares the large dimensions as the FFN layer, $M _ { i n } \times M _ { o u t }$ . In batched iterative editing, assuming we need to perform $T$ rounds of editing, with $l$ being the number of layers to be edited and $Q$ representing the computational complexity of sparsification, the total complexity required for sparsifying $\Delta$ would be $\mathcal { O } ( M _ { i n } \times M _ { o u t } \Breve { \times } l \times \bar { T } \times Q )$ . To reduce the computational complexity to an acceptable level, we apply the parameter update sparsification which sparsifies the residual vectors $\delta$ in $R$ and allows this sparsity to propagate into the update matrix $\Delta$ . This reduces the computational complexity to $\mathcal { O } ( M _ { i n } \times B \times l \times T \times Q )$ , where $B$ is the editing batch size, which is significantly smaller than $M _ { o u t }$ . This also enables our parameter sparsification method to scale more efficiently to larger models, as the computational cost primarily depends on $M _ { i n }$ , making it more efficient than existing methods that rely on both $M _ { i n }$ and $M _ { o u t }$ .

![](images/7eab0c348dc246b88f6cf1810bdedf9c6dfeac6f25ed712c503bc317a0186efd.jpg)

Figure 4: Detailed performance of Efficacy, Generalization and Locality on the CounterFact dataset (batch size $= 1 0 0 \$ ). The vertical axis represents the metric scores, while the horizontal axis represents the number of iterations. SP denotes the parameter sparsification, WD denotes the Wasserstein distance constraint, and $\mathbf { W D + S P }$ denotes their combination.

Table 1: The overall editing Score on the zsRE dataset (batch size $= 5 0$ ). $@ \mathbf { k }$ denotes the $\mathbf { k }$ -th iteration of batched editing. $\mathbf { + W D }$ denotes the Wasserstein distance constraint, $\mathbf { + } \mathbf { S P }$ denotes the parameter sparsification, and $\mathbf { + W D + S P }$ denotes the combined method.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="10">Number of Batched Iteration</td><td rowspan="2">Avg</td></tr><tr><td>@1</td><td>@12</td><td>@24</td><td>@36</td><td>@48</td><td>@60</td><td>@72</td><td>@80</td><td>@88</td><td>@96</td><td>@100</td></tr><tr><td>MEMIT</td><td>57.61</td><td>58.02</td><td>58.40</td><td>63.11</td><td>59.91</td><td>54.32</td><td>50.97</td><td>28.53</td><td>4.29</td><td>0.66</td><td>0.45</td><td>39.66</td></tr><tr><td>+WD</td><td>57.33</td><td>59.65</td><td>58.17</td><td>64.63</td><td>60.21</td><td>57.18</td><td>53.17</td><td>47.80</td><td>44.71</td><td>39.18</td><td>39.35</td><td>52.85</td></tr><tr><td>+SP</td><td>57.24</td><td>58.92</td><td>57.87</td><td>66.11</td><td>58.97</td><td>52.95</td><td>52.07</td><td>43.30</td><td>37.79</td><td>16.93</td><td>11.88</td><td>46.73</td></tr><tr><td>+WD+SP</td><td>57.15</td><td>58.36</td><td>59.52</td><td>64.29</td><td>58.28</td><td>52.87</td><td>60.29</td><td>48.71</td><td>51.69</td><td>41.81</td><td>48.35</td><td>54.67</td></tr><tr><td>PMET</td><td>56.31</td><td>58.58</td><td>57.24</td><td>64.26</td><td>51.66</td><td>28.63</td><td>0.92</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>28.87</td></tr><tr><td>+WD</td><td>56.06</td><td>59.38</td><td>59.86</td><td>64.26</td><td>53.32</td><td>34.25</td><td>9.69</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>30.62</td></tr><tr><td>+SP</td><td>56.08</td><td>60.91</td><td>58.40</td><td>60.91</td><td>58.79</td><td>49.02</td><td>38.86</td><td>21.65</td><td>18.07</td><td>10.16</td><td>11.56</td><td>40.40</td></tr><tr><td>+WD+SP</td><td>54.83</td><td>57.79</td><td>57.44</td><td>62.62</td><td>57.86</td><td>49.91</td><td>48.56</td><td>38.25</td><td>31.11</td><td>20.85</td><td>26.48</td><td>45.98</td></tr></table></body></html>

Table 2: Average metric scores on the zsRE dataset (batch size $= 5 0$ ). Avg.Eff, Avg.Gen, and Avg.Loc denotes the average Efficacy, Generalization and Locality across all batched editing iterations, respectively.   

<html><body><table><tr><td>Method</td><td>Avg.Eff Avg.Gen</td><td>Acg.Loc</td></tr><tr><td>MEMIT</td><td>76.41 73.69</td><td>21.33</td></tr><tr><td>+WD</td><td>99.94 96.05</td><td>28.04</td></tr><tr><td>+SP</td><td>94.18 90.73</td><td>24.67</td></tr><tr><td>+WD+SP</td><td>99.86 96.85</td><td>29.29</td></tr><tr><td>PMET</td><td>52.91 50.72</td><td>15.77</td></tr><tr><td>+WD</td><td>59.25 57.73</td><td>17.87</td></tr><tr><td>+SP</td><td>74.08 71.83</td><td>23.27</td></tr><tr><td>+WD+SP</td><td>91.84 88.74</td><td>24.39</td></tr></table></body></html>

For the details of sparsification, we use two operations: pruning and dropout. The pruning operation retains the top$k$ parameters with the largest magnitudes in $\delta$ , discarding the smaller ones, as they contribute less to the editing performance. We control the degree of sparsification by setting a pruning ratio $p r ( 0 < p r \bar { < } 1 )$ , and define a mask matrix

Pr as follows:

$$
\mathbf { P r } _ { i , j } = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } \delta _ { i , j } { \mathrm { ~ i n ~ t o p  – p r } } ( \delta ) } \\ { 0 , } & { { \mathrm { e l s e } } } \end{array} \right. }
$$

therefore the pruned residual vector is given by: $\delta \gets \mathbf { P r } \odot$ $\delta$ . Additionally, we apply dropout with a rate of $d r$ to the pruned $\delta$ , which can be expressed as:

$$
\delta  \frac { ( \mathbf { 1 } - \mathbf { D r } ) \odot \delta } { 1 - d r }
$$

where $\mathbf { D r } \sim \mathrm { B e r n } ( d r )$ is the dropout mask matrix.

After applying the Wasserstein distance constraint and parameter sparsification, the updated parameter $\delta$ exhibits a relatively smooth and sparse distribution. This ensures that the parameter updates from knowledge editing do not cause drastic changes to the model’s parameter distribution, thereby maintaining stability even after multiple iterations.

# Experiments Datasets and Baselines

We evaluate our method on two widely used knowledge editing datasets, zsRE(Levy et al. 2017) and CounterFact(Meng et al. 2022a). In the zsRE dataset, each sample consists of a factual knowledge statement (editing request), its paraphrase, and an irrelevant natural question. In the CounterFact dataset, each sample consists of a factual knowledge statement, two paraphrased sentences, and ten neighborhood questions.

Table 3: Overall editing Score on the CounterFact dataset (batch size $= 5 0$ ). $\boldsymbol { \ @ } \mathbf { k }$ denotes the $\mathbf { k }$ -th iteration of batched editing. $\mathbf { + W D }$ , $\mathbf { + } \mathbf { S P }$ , and $\mathbf { + W D + S P }$ denote the Wasserstein distance constraint, the parameter sparsification, and the combined method, respectively.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="10">NumberofBatchedIteration</td><td rowspan="2">Avg</td></tr><tr><td>@1</td><td>@8</td><td>@16</td><td>@20</td><td>@24</td><td>@28</td><td>@32</td><td>@36</td><td>@42</td><td>@46</td><td>@50</td></tr><tr><td>MEMIT</td><td>91.78</td><td>89.73</td><td>88.29</td><td>86.57</td><td>84.15</td><td>82.56</td><td>81.37</td><td>81.49</td><td>81.01</td><td>82.53</td><td>76.43</td><td>84.17</td></tr><tr><td>+WD</td><td>92.25</td><td>89.64</td><td>88.31</td><td>85.63</td><td>83.71</td><td>81.95</td><td>80.41</td><td>82.05</td><td>80.42</td><td>83.25</td><td>77.73</td><td>84.12</td></tr><tr><td>+SP</td><td>91.12</td><td>90.65</td><td>89.26</td><td>85.82</td><td>85.09</td><td>81.16</td><td>81.47</td><td>81.13</td><td>80.41</td><td>81.50</td><td>74.47</td><td>83.83</td></tr><tr><td>+WD+SP</td><td>91.56</td><td>89.82</td><td>88.68</td><td>87.06</td><td>84.07</td><td>84.66</td><td>80.53</td><td>82.44</td><td>81.38</td><td>82.09</td><td>76.12</td><td>84.40</td></tr><tr><td>PMET</td><td>92.74</td><td>90.93</td><td>89.15</td><td>84.03</td><td>81.49</td><td>82.32</td><td>79.03</td><td>78.73</td><td>75.44</td><td>76.35</td><td>66.97</td><td>81.56</td></tr><tr><td>+WD</td><td>92.42</td><td>89.96</td><td>89.91</td><td>85.63</td><td>83.09</td><td>83.90</td><td>79.56</td><td>78.77</td><td>74.43</td><td>77.37</td><td>68.18</td><td>82.11</td></tr><tr><td>+SP</td><td>92.80</td><td>90.75</td><td>89.63</td><td>86.14</td><td>83.51</td><td>83.14</td><td>79.89</td><td>79.24</td><td>76.03</td><td>75.46</td><td>69.65</td><td>82.39</td></tr><tr><td>+WD+SP</td><td>92.76</td><td>87.97</td><td>88.68</td><td>85.83</td><td>84.04</td><td>82.91</td><td>80.90</td><td>79.38</td><td>77.26</td><td>82.19</td><td>75.90</td><td>83.44</td></tr></table></body></html>

Table 4: Average metric scores on the CounterFact dataset (batch size $= 5 0$ ). Avg.Eff, Avg.Gen, and Avg.Loc denotes the average Efficacy, Generalization and Locality across all batched editing iterations, respectively.   

<html><body><table><tr><td>Method</td><td>Avg.Eff Avg.Gen</td><td>Acg.Loc</td><td></td></tr><tr><td>MEMIT</td><td>100</td><td>95.55</td><td>66.42</td></tr><tr><td>+WD</td><td>100</td><td>94.55</td><td>66.85</td></tr><tr><td>+SP</td><td>99.82</td><td>94.18</td><td>66.53</td></tr><tr><td>+WD+SP</td><td>100</td><td>95.91</td><td>66.65</td></tr><tr><td>PMET</td><td>100</td><td>97.00</td><td>61.84</td></tr><tr><td>+WD</td><td>99.82</td><td>95.18</td><td>63.64</td></tr><tr><td>+SP</td><td>99.63</td><td>97.00</td><td>63.22</td></tr><tr><td>+WD+SP</td><td>99.27</td><td>93.64</td><td>66.65</td></tr></table></body></html>

Given that our proposed method is a general approach that can be applied to any update-based editing method, we selected the representative parameter updating methods for massive editing tasks, MEMIT (Meng et al. 2022b) and PMET (Li et al. 2023), on the GPT-J 6B (Wang and Komatsuzaki 2021) model as experimental baselines. We evaluated the impact of the Wasserstein distance constraint, parameter sparsification, and their combination on the performance of these baselines in the batched iterative knowledge editing task.

# Metrics and Settings

We use the standard evaluation metrics for knowledge editing, which are defined as follows:

• Efficacy measures the editing accuracy; it corresponds to the factual knowledge statement in the dataset being successfully edited.   
• Generalization evaluates whether the edit can be extended to paraphrased or contextually relevant sentences included in the data set.   
• Locality refers to the preservation of original knowledge that is not related to editing requests, ensuring it remains

intact. This is evaluated using irrelevant natural questions or neighborhood questions in the dataset.

Formally, the metrics for Efficacy and Generalization are defined as follows:

$$
\mathbb { E } [ \mathcal { P } _ { \mathcal { M } ( \theta ^ { * } ) } ( o ^ { * } \vert ( s , r ) ) > \mathcal { P } _ { \mathcal { M } ( \theta ^ { * } ) } ( o \vert ( s , r ) ) ] ,
$$

and the metric on Locality is defined as follows:

$$
\begin{array} { r } { \mathbb { E } [ \mathcal { P } _ { \mathcal { M } ( \theta ^ { * } ) } ( o ^ { * } \vert ( s , r ) ) < \mathcal { P } _ { \mathcal { M } ( \theta ^ { * } ) } ( o \vert ( s , r ) ) ] . } \end{array}
$$

On the batched iterative editing benchmark, we evaluate the knowledge editing methods based on these three metrics in each batch of edits and calculate their harmonic mean to obtain the overall editing Score. On the zsRE dataset, we set the batch size for iterative editing to 50 and conducted 100 iterations, resulting in a total of 5,000 knowledge edits. On the CounterFact dataset, we evaluated the setting with a batch size of 50 over 50 iterations, resulting in a total of 2,500 edits. We then extended this to a more challenging setting with a batch size of 100 over 100 iterations, totaling 10,000 edits. For hyper-parameters, we set the prune rate pr to 0.7 and the dropout rate dr to 0.3. All of our experiments were conducted on NVIDIA RTX A6000 48G GPUs.

# Results

We present all our experimental results as ablation studies to clearly highlight the contributions of the proposed Wasserstein distance constraint and parameter sparsification to editing performance.

# Results on zsRE

For the zsRE dataset, we evaluated the performance of MEMIT and PMET in iterative editing with a batch size of 50 over 100 iterations. As observed in Table 1, the editing performance gradually declines with successive iterations, indicating that the accumulation of updated parameters makes the model increasingly unstable, thereby complicating future edits. Between the 70-80th iterations, the model begins to deteriorate rapidly, eventually collapsing and completely losing its editing capability in subsequent iterations.

After incorporating the Wasserstein distance constraint, the MEMIT-based model was able to retain most of its editing capabilities, even during the later stages of iteration. The

<html><body><table><tr><td rowspan="2">Method</td><td colspan="10">Number of Batched Iteration</td><td rowspan="2">Avg</td></tr><tr><td>@1</td><td>@10</td><td>@20</td><td>@30</td><td>@40</td><td>@50</td><td>@60</td><td>@70</td><td>@80</td><td>@90</td><td>@100</td></tr><tr><td>MEMIT</td><td>91.58</td><td>87.84</td><td>84.04</td><td>82.08</td><td>81.82</td><td>72.98</td><td>76.72</td><td>71.07</td><td>64.8</td><td>58.84</td><td>53.47</td><td>75.02</td></tr><tr><td>+WD</td><td>91.45</td><td>87.65</td><td>85.06</td><td>82.10</td><td>82.64</td><td>72.83</td><td>78.88</td><td>76.10</td><td>77.03</td><td>67.90</td><td>63.30</td><td>78.63</td></tr><tr><td>+SP</td><td>91.00</td><td>88.64</td><td>83.71</td><td>81.07</td><td>80.53</td><td>72.93</td><td>74.15</td><td>75.63</td><td>72.97</td><td>64.80</td><td>55.50</td><td>76.44</td></tr><tr><td>+WD+SP</td><td>91.80</td><td>88.42</td><td>83.54</td><td>80.19</td><td>79.99</td><td>72.89</td><td>74.19</td><td>74.48</td><td>73.83</td><td>72.40</td><td>77.59</td><td>79.03</td></tr></table></body></html>

Table 5: The overall editing Score on the CounterFact dataset (batch size $= 1 0 0 \$ ). ${ \ @ { \bf { k } } }$ denotes the $\mathbf { k }$ -th iteration of batched editing. $\mathbf { + W D }$ , $\mathbf { + } \mathbf { S P }$ , and $\mathbf { \sigma } _ { + \mathbf { W } \mathbf { D + S P } }$ denote the Wasserstein distance constraint, the parameter sparsification, and the combined method, respectively.

PMET-based model deteriorated more slowly than the baseline during the intermediate stages but still collapsed in the later stages. While solely applying parameter sparsification to the baselines provides some improvement, the model still lost most of its capability in the later stages. The combined method further improved the model’s stability in later iterations and achieved the best overall performance. Figure 3 presents the details of the Efficacy, Generalization, and Locality metrics, clearly demonstrating the effectiveness of our method.

Table 2 additionally presents the average Efficacy, Generalization, and Locality scores across all iterations and settings. The baseline exhibited significantly lower average scores, mainly due to model collapse in the later stages. All of our methods demonstrate substantially higher average performance compared to the baseline, with the combined approach achieving the best results.

# Results on CounterFact

For the CounterFact dataset, we first evaluated MEMIT and PMET with a batch size of 50 over 50 iterations. As shown in Table 3, all methods show a decreasing trend in editing performance as the number of iterations increases, while the model did not experience a rapid collapse within 50 iterations. Our method shows noticeable improvement over MEMIT. Employing the Wasserstein distance constraint, parameter sparsification independently, or their combination enhances performance in specific iterations, with the combined approach achieving the highest overall average performance overall. For PMET, the combined approach consistently yields the best performance during the later stages of iteration, and its effectiveness becomes more evident as the number of iterations increases. Using WD or SP independently also improves average performance compared to the baseline, and the combined approach achieves the highest average performance.

It is also observed that during the early stages of iteration (e.g., before the 28th iteration), our independent methods outperform the combined approach in certain iterations under both baselines. This is primarily because the model maintains relatively stable performance in the early stages due to minor parameter shifts, thus making the sole application of either the Wasserstein distance constraint or parameter sparsification sufficient to effectively ensure stability. Table 4 further presents the average performance of the Efficacy, Generalization, and Locality metrics. We found that all our approaches improve Locality. For PMET, the combined method sacrifices some Generalization but achieves a more significant improvement in Locality. This trade-off between Generalization and Locality remains a significant challenge in knowledge editing tasks. Nevertheless, these results indicate that our method helps maintain model stability, as a higher Locality implies better preservation of knowledge unrelated to the edits.

To further assess the effectiveness of our method, we conducted tests in a more challenging setting. Table 5 presents the results obtained after 100 iterations with a batch size of 100. Our method clearly demonstrates its effectiveness in maintaining stability during the later stages of batched iterative editing. While applying the Wasserstein distance constraint or parameter sparsification individually offers considerable improvements, the combined method yields the best average results, consistent with the findings on the zsRE dataset. Figure 4 presents the details on the Efficacy, Generalization, and Locality metrics. Our method ensures that both Efficacy and Generalization remain stable in the later stages of iteration, whereas Locality shows no significant differences across all methods.

# Conclusion

In this work, we addressed the challenging task of batched iterative knowledge editing and demonstrated the instability of parameter updating methods (capable of batch editing) in the later stages of iterative editing. To prevent the rapid collapse of the model during later iterations, we proposed the use of the Wasserstein distance constraint and parameter sparsification to mitigate changes in the model’s parameter distribution after multiple rounds of batched iterative editing, thereby preserving stability. These two techniques can be applied independently or in combination, and notably, the combined approach yielded the best results in most experiments.

For future work, we will focus on achieving lifelong batched iterative editing, aiming to completely eliminate the side effects of each batch iteration, thereby fully mitigating the distribution shifts and the severe instability caused by the accumulation of parameter updates during the iterative process. Furthermore, exploring improved methods that can support larger batch sizes during iterative editing is another important direction for future research. This will provide valuable insights and contributions to the practical updating and maintenance of language models.