# CDW-CoT: Clustered Distance-Weighted Chain-of-Thoughts Reasoning

Yuanheng Fang1, Guoqing Chao\*1, Wenqiang Lei2, Shaobo $\mathbf { L i } ^ { 1 }$ , Dianhui Chu1

1Harbin Institute of Technology, Weihai, 264209, Shandong, China 2Sichuan University, Chengdu, 610065, Sichuan, China 23s130443@stu.hit.edu.cn, guoqingchao@hit.edu.cn, wenqianglei $@$ scu.edu.cn, lishaobo@hit.edu.cn, chudh@hit.edu.cn

# Abstract

Large Language Models (LLMs) have recently achieved impressive results in complex reasoning tasks through Chain of Thought (CoT) prompting. However, most existing CoT methods rely on using the same prompts, whether manually designed or automatically generated, to handle the entire dataset. This one-size-fits-all approach may fail to meet the specific needs arising from the diversities within a single dataset. To solve this problem, we propose the Clustered Distance-Weighted Chain of Thought (CDW-CoT) method, which dynamically constructs prompts tailored to the characteristics of each data instance by integrating clustering and prompt optimization techniques. Our method employs clustering algorithms to categorize the dataset into distinct groups, from which a candidate pool of prompts is selected to reflect the inherent diversity within the dataset. For each cluster, CDW-CoT trains the optimal prompt probability distribution tailored to their specific characteristics. Finally, it dynamically constructs a unique prompt probability distribution for each test instance, based on its proximity to cluster centers, from which prompts are selected for reasoning. CDW-CoT consistently outperforms traditional CoT methods across six datasets, including commonsense, symbolic, and mathematical reasoning tasks. Specifically, when compared to manual CoT, CDW-CoT achieves an average accuracy improvement of $2 5 . 3 4 \%$ on LLaMA2 (13B) and $1 5 . 7 2 \%$ on LLaMA3 (8B).

# Introduction

Recent advancements in LLMs, such as GPT-3 (Brown et al. 2020), LLama2 (Touvron et al. 2023), and Llama3 (Dubey et al. 2024), have significantly enhanced their capability to tackle complex reasoning tasks. Some studies (Brown et al. 2020; Thoppilan et al. 2022) have demonstrated LLMs’ impressive performance in decomposing multi-step problems into manageable intermediate steps, resulting in more accurate and contextually relevant answers. A technique that has gained prominence in this context is CoT prompting, which systematically structures the reasoning process into a series of intermediate steps. This method has been shown to significantly improve the model’s performance on complex tasks across various domains (Wei et al. 2022).

Initially, CoT prompting involved embedding manually crafted exemplars within the model’s prompt to guide its reasoning process—a method that was effective but laborintensive and not scalable (Wei et al. 2022). This approach evolved into Zero-Shot-CoT (Kojima et al. 2022), which allowed models to engage in reasoning without task-specific exemplars, relying on generic prompts to elicit intermediate reasoning steps. However, the absence of tailored guidance often limited its efficacy in more complex or domainspecific tasks.

![](images/a2d88a71e1a3e4b0d5a8e86ab10911704791d68bd75265e8661aee66dd07d29f.jpg)  
Figure 1: Using the same prompts for all instances in the dataset resulted in significant performance variability across different clusters, highlighting the limitations of Auto-CoT in addressing diverse reasoning demands within different data categories. This underscores the need for tailored prompt strategies.

To address the intensive manual efforts required by Manual-CoT, the Auto-CoT paradigm has been proposed (Zhang et al. 2022). This method automates the generation of reasoning chains by clustering related questions and selecting a representative question from each cluster to generate a reasoning chain using simple heuristics. Recent works (Chu et al. 2023) focus on further automating (Shum, Diao, and Zhang 2023) and refining the CoT generation process. Techniques such as Enhanced Reasoning (Wang et al. 2023; Li and Qiu 2023; Wu, Zhang, and Huang 2023), Voting and Ranking (Fu et al. 2022; Li et al. 2023, 2022), and Verification and Refinement (Lyu et al. 2023; Shao et al. 2023; Aggarwal et al. 2023; Weng et al. 2022; Wang et al. 2022a) have been developed to enhance the quality and applicability of CoT across diverse tasks. Especially, Automate-CoT approach integrates variance-reduced policy gradient methods to optimize the selection of CoT exemplars, significantly reducing the dependency on manual prompt engineering.

Despite these advancements, Automatic Chain of Thought Prompting still encounters significant challenges, particularly because most of them use the same prompts for all instances in the dataset. As exemplified by Auto-CoT (Zhang et al. 2022) shown in Figure 1, this approach results in considerable performance variability across different clusters. This variability highlights its inability to effectively address the diverse reasoning demands of different data categories, emphasizing the necessity for more adaptive techniques that can tailor prompts to the unique characteristics of each cluster.

To address the limitations inherent in manual and uniform prompt strategies across diverse datasets, we introduce the CDW-CoT framework. This method innovatively combines clustering with dynamic prompt optimization to enhance the adaptability and precision in reasoning tasks. By segmenting the dataset into distinct clusters, CDW-CoT harnesses the unique characteristics of each group to generate a tailored prompt candidate pool. For each cluster, we calculate an optimal prompt probability distribution, finely tuned to the specific demands and nuances of the data. Additionally, our framework incorporates a distance-weighted prompt selection mechanism that dynamically adapts reasoning strategies based on the proximity of test instances to the cluster centers. This ensures that each reasoning step is contextually informed and effectively customized, significantly improving the reasoning accuracy. Experiment results on six datasets show the superiority of our proposed method CDWCoT over the state-of-the-art methods.

The main contributions of our work are summarized as follows:

• We leverage the clustering technique to produce a diverse prompt candidate pool that mining the category-specific information sufficiently, enhancing the relevance and effectiveness of prompts for different clusters within the same dataset.   
• Our framework calculates the optimal prompt probability distributions for each cluster within the dataset, effectively treating datasets as distinct clusters and enabling highly targeted reasoning approaches tailored to the unique characteristics of each group.   
• We introduce a method for employing distance-weighted calculations for each test instance’s prompt probability distribution, which refines and tailors the reasoning process of large language models to the specific requirements of each instance.   
• Our empirical evaluations confirm that the CDW-CoT framework substantially outperforms traditional CoT methods, achieving the state-of-the-art accuracy across multiple datasets.

# Related Works

# Chain of Thought Prompting

CoT Prompting enhances logical reasoning in LLMs like GPT-3, developed as the models increased in scale (Brown et al. 2020). Wei et al. first introduced CoT, using manually constructed detailed prompts to systematically guide LLMs through each logical step, significantly enhancing reasoning transparency and accuracy (Wei et al. 2022). Building on this foundational work, Zero-Shot-CoT employs the simple prompt “Let’s think step by step” to facilitate unsupervised reasoning, effectively enabling CoT without predefined examples (Kojima et al. 2022).

# Automatic Chain of Thought Prompting

Addressing the accuracy challenges in Zero-Shot-CoT and the resource intensity of Few-Shot CoT, Auto-CoT automates reasoning chain generation. This method clusters related questions, using cluster centers as prompts, thereby reducing manual labor and improving scalability (Zhang et al. 2022). Building on this, complexity-based prompting selects prompts based on their reasoning complexity, which has been shown to improve performance on multi-step reasoning tasks significantly (Fu et al. 2022). Furthermore, selfverification techniques introduced in studies allow models to cross-check and refine their outputs (Weng et al. 2022). In the context of mathematical reasoning, the MathPrompter framework validates results by leveraging different algebraic expressions or Python functions to solve problems (Imani, Du, and Shrivastava 2023).

# Policy Gradient Optimization Methods

The Black-Box Discrete Prompt Learning (BDPL) employs variance-reduced policy gradients to optimize prompts efficiently, enhancing LLM performance without direct access to model parameters (Diao et al. 2022). Following this, the Black-Box Prompt Optimization (BPO) further refines these advancements by aligning LLM outputs with user preferences through optimized prompts, improving user interactions and satisfaction (Cheng et al. 2023). Dynamic Prompt Learning via Policy Gradient (PROMPTPG) further refines this approach by dynamically selecting in-context examples that optimize reasoning tasks, particularly in complex domains like mathematics (Lu et al. 2022). Building on these strategies, the Automatic Prompt Augmentation and Selection method extends the application of policy gradient methods to CoT prompting, automating both the generation and the optimization of reasoning chains (Shum, Diao, and Zhang 2023).

# CDW-CoT Model

In this section, we introduce our proposed CDW-CoT model, as depicted in Figure 2. The CDW-CoT is composed of the three components: cluster-based prompt candidate pool initialization, optimizing prompt probability distributions for clusters and distance-weighted prompt selection and inference.

![](images/8ff9f9bcb07ad1cff8e1208d49a661a9e2a232c04edc3708461ac5d16b576b20.jpg)  
Figure 2: Framework of the proposed CDW-CoT. (a) After clustering, prompt candidates are generated based on the cluster centers. $\mathbf { C T D } ^ { ( i ) }$ and cluster center coordinates $( x _ { i } , y _ { i } )$ are also obtained. (b) For each cluster, $p ^ { ( i ) }$ is initially set to $p ^ { ( \mathrm { i n i t } ) }$ and then optimized through Black-Box Prompt Learning (BBPL) to achieve the optimal distribution. (c) For test instance, a distance-weighted prompt probability distribution is constructed to select prompts and perform reasoning.

# Cluster-Based Prompt Candidate Pool Initialization

The dataset D = {(xi, yi)}iN=1 consists of N questionanswer pairs. Each instance $x _ { i }$ is transformed into the vector embeddings $\{ e _ { i } \}$ using a pre-trained sentence transformer and clustered into $K$ groups via K-means.

As illustrated in Figure 2(a), following the clustering process, the preliminary selection of prompt candidates begins from the centroid of each cluster. The number of candidates selected from each cluster $S _ { c }$ is based on the proportion of that cluster’s data within the overall dataset. Once the preliminary prompt candidate pool is established, it is refined into the final prompt candidates (PC) through zero-shot-CoT (Kojima et al. 2022). The entire process is detailed in Algorithm 1.

Simultaneously, we establish an initial prompt probability distribution $p ^ { ( \mathrm { i n i t } ) }$ , where each candidate in the pool is assigned an equal probability. This balanced distribution, along with the cluster-specific training data $\mathbf { C T D } ^ { ( i ) }$ , serves as the foundation for the next phase of the model training process.

Furthermore, the coordinates of each cluster’s centroid, obtained during the clustering process, are stored to use for calculating the distance of the test instance to them. These coordinates play a critical role in the model’s final phase: distance-weighted prompt selection and inference, where they guide the distance-weighted prompt probability distribution and ensure the model adapts effectively to new, unseen instances.

<html><body><table><tr><td>Algorithm1:Cluster-Based Prompt Candidate Pool Initial-</td></tr><tr><td>ization Input: X= {x1,...,xN},Y = {y1,...,yN},Number of ClustersK,Pool Size S</td></tr><tr><td>Output: Prompt Candidates PC 1:{ei} ← SentenceTransformer(X) 2: cluster_assignment ← K-Means(K,{ei}) 3: Initialize cluster data structure C =[]</td></tr><tr><td>4:fori←1toNdo 5: Ci←cluster_assignment[i] 6: d ←EuclideanDistance(ei,cluster_centers[ci])</td></tr><tr><td>7: Add (xi, yi,d) to C[ci] 8:end for 9:Prepare to build prompt candidate pool: 10:forc←1toKdo</td></tr></table></body></html>

# Optimizing Prompt Probability Distributions for Clusters

As illustrated in Figure 2(b), the optimization of prompt probability distribution for each cluster is conducted using the BBPL method. The process begins by setting the initial distribution $p ^ { ( i ) }$ for each cluster to a uniform distribution $p ^ { ( \mathrm { i n i t } ) }$ . This distribution is then refined through gradient descent, based on feedback from the training process with CTD(i).

For each cluster $i$ , prompts are sampled according to the $p ^ { ( i ) }$ . These prompts, along with the $\dot { \mathrm { C T D } } ^ { ( i ) }$ , are input into the LLM, which returns the prediction and computes the corresponding loss.

The gradient for each prompt is computed as:

$$
\delta = - \frac { 1 } { p ^ { ( i ) } } ,
$$

where $p ^ { ( i ) }$ represents the prompt probability matrix for cluster $i$ .

These gradients are adjusted based on the actual usage of prompts during training:

$$
\delta _ { k , m , n } = { \left\{ \begin{array} { l l } { - \delta _ { k , m , n } } & { { \mathrm { i f ~ } } n { \mathrm { ~ i s ~ s e l e c t e d } } } \\ { \delta _ { k , m , n } } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

Here, $k$ indexes the sample within the batch, $m$ represents the prompt, and $n$ corresponds to the indices of the prompts sampled. Adjustments are weighted by the deviation of each sample’s loss from the batch average:

$$
\mathrm { G r a d i e n t } = \sum _ { k = 1 } ^ { \mathrm { s } } \frac { L _ { k } - L _ { \mathrm { a v g } } } { \mathrm { s } - 1 } \times \delta _ { k } ,
$$

where $s$ is the sample size used in the optimization process. Using the aggregated gradient and learning rate $\eta$ , the probability matrix $p ^ { ( i ) }$ is updated according to the following formula:

$$
p _ { m n } ^ { ( i ) }  p _ { m n } ^ { ( i ) } - \eta \cdot \mathrm { G r a d i e n t } _ { m n } .
$$

Probabilities are then normalized and clipped within the range [0,1] to ensure stability:

$$
p _ { m n } ^ { ( i ) }  \operatorname* { m a x } ( \operatorname* { m i n } ( p _ { m n } ^ { ( i ) } , 1 ) , 0 ) .
$$

The optimized prompt probabilities are validated on a validation dataset. If the performance improves, these updated settings are used for future operations.

Through this process, we obtain the optimal prompt probability distribution for each cluster, as depicted in Figure 2(b).

# Distance-Weighted Prompt Selection and Inference

This subsection describes how we construct unique prompt probability distribution for each test instance through distance weighting, using the optimal prompt probability distribution for each cluster, and the coordinates of cluster centers. The obtained prompt probability distribution is used to select the final prompts that are concatenated with test instance and input into the LLM, as illustrated in Figure 2(c).

Distance Calculation For each test instance, its embedding obtained through a sentence Transformer is compared with the cluster centers to compute the Euclidean distances. These distances reflect the instance’s similarity to each cluster.

Weight Calculation The computed distances are converted into weights using a temperature-scaled softmax function:

$$
{ \mathrm { w e i g h t s } } = { \frac { \exp ( - \mathrm { d i s t a n c e s } / T ) } { \sum \exp ( - \mathrm { d i s t a n c e s } / T ) } } ,
$$

where $T$ is a temperature parameter that controls the sensitivity to distance variations.

Prompt Distribution Calculation The prompt probability distribution for the test instance is calculated by weighting the optimal prompt distributions of each cluster:

$$
\mathfrak { p } = \sum _ { i = 1 } ^ { K } \mathrm { w e i g h t s } _ { i } \cdot p ^ { ( i ) } ,
$$

where $K$ is the number of clusters. This weighted combination tailors the prompt probability distribution to the specific characteristics of the test instance.

Query Execution and Evaluation Prompts are selected based on the computed distribution and then concatenated with the original test instance. The LLM uses this concatenated input to generate a response, which is subsequently evaluated against the actual answer to assess the accuracy and effectiveness.

The steps of this process are outlined in Algorithm 2, demonstrating the implementation of distance-weighted prompt probability distribution and its impact on inference.

<html><body><table><tr><td>Algorithm 2:Distance-Weighted Prompt Selection and In- ference Input: Test dataset Dtest,Cluster centers C,Temperature T,</td></tr><tr><td>Optimized prompt probabilities for each cluster p(i) Output:Evaluated responses R 1: Initialize weights as an empty list 2:Initialize R as an empty list to store responses 3:for each instance q in Dtest do 4: eq ← SentenceTransformer(q) 5: for each centerc in Cdo 6: Compute distance: distance = Euclidean(eq, c)</td></tr><tr><td>7: Compute weight: weight = exp(-distance/T) 8: Append weight to weights 9: end for 10: weights 11: Compute the prompt probability distribution for q: Normalize weights: weights ← ∑weights</td></tr><tr><td>P=∑1 weights: p() 12: Select prompt using p 13: Input prompt and q into LLM to generate response 14: Evaluate response accuracy and append to R 15:end for</td></tr></table></body></html>

Table 1: Comparative exact match accuracy across various datasets using LLaMA2 (13B) and LLaMA3 (8B) models. The CDW-CoT method consistently outperforms traditional CoT methods in all tested reasoning tasks and datasets, improving accuracy for both models.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="2">Commonsense Reasoning</td><td colspan="2">Symbolic Reasoning</td><td colspan="2">Mathematical Reasoning</td></tr><tr><td>CSQA</td><td>StrategyQA</td><td>Letter</td><td>Coin</td><td>MultiArith</td><td>AQuA</td></tr><tr><td colspan="7">LLaMA2 (13B)</td></tr><tr><td>Zero-Shot-CoT (Kojima et al. 2022)</td><td>32.68</td><td>48.41</td><td>30.20</td><td>51.80</td><td>71.00</td><td>30.31</td></tr><tr><td>Auto-CoT (Zhang et al. 2022)</td><td>51.09</td><td>56.24</td><td>30.80</td><td>51.00</td><td>44.17</td><td>24.02</td></tr><tr><td>Manual-CoT (Wei et al. 2022)</td><td>46.52</td><td>60.48</td><td>15.80</td><td>47.60</td><td>44.17</td><td>30.31</td></tr><tr><td>CDW-CoT (ours)</td><td>61.41</td><td>70.06</td><td>82.67</td><td>61.33</td><td>85.56</td><td>35.89</td></tr><tr><td colspan="7">LLaMA3 (8B)</td></tr><tr><td>Zero-Shot-CoT (Kojima et al. 2022)</td><td>60.52</td><td>66.72</td><td>76.67</td><td>44.40</td><td>90.00</td><td>48.03</td></tr><tr><td>Auto-CoT (Zhang et al. 2022)</td><td>69.57</td><td>60.57</td><td>50.60</td><td>60.80</td><td>68.83</td><td>31.10</td></tr><tr><td>Manual-CoT(Wei et al. 2022)</td><td>56.84</td><td>57.51</td><td>61.40</td><td>60.20</td><td>85.17</td><td>32.68</td></tr><tr><td>CDW-CoT (ours)</td><td>72.15</td><td>67.44</td><td>82.67</td><td>70.70</td><td>95.17</td><td>58.97</td></tr></table></body></html>

# Experiments and Results

# Experiments Setup

Tasks and Datasets We evaluated the CoT frameworks on six datasets across three categories of reasoning tasks, which are listed as follows:

# • Commonsense Reasoning:

CommonsenseQA (CSQA) (Talmor et al. 2018): A widely used dataset for evaluating commonsense reasoning through multiple-choice questions that require inferencing based on prior knowledge and context.

StrategyQA (Geva et al. 2021): It contains questions requiring implicit multi-hop reasoning to derive yes/no answers, testing the model’s ability to connect various pieces of information logically.

# • Symbolic Reasoning:

Letter (Wei et al. 2022): It involves tasks like last letter concatenation, designed to test the symbolic reasoning capabilities of models.

Coin (Wei et al. 2022): It focuses on determining the state of a coin after a series of flips, evaluating the model’s ability to track state changes through symbolic manipulations.

# • Mathematical Reasoning:

MultiArith (Roy and Roth 2016): It consists of multistep arithmetic word problems that require a sequence of operations to reach the solution, testing multi-step reasoning in arithmetic contexts.

AQuA (Ling et al. 2017): It includes complex arithmetic word problems with multiple-choice answers, providing a benchmark for evaluating sophisticated reasoning and calculation skills.

Models and Baselines We conducted comparative experiments using both the LLaMA2 (13B) and LLaMA3 (8B) models, running on two NVIDIA 4090 GPUs locally. The LLaMA2 (13B) model was selected for its easy-use, while LLaMA3 (8B) was chosen to evaluate the scalability of our approach across different large language models. To evaluate the performance of our CDW-CoT framework, we compared it against several baseline methods implemented on the same LLM:

• Zero-Shot-CoT (Kojima et al. 2022): It uses a simple prompt like “Let’s think step by step” without requiring prior demonstrations.   
• Auto-CoT (Zhang et al. 2022): It automates reasoning chain generation by clustering similar questions and and using the cluster centers as prompts.   
• Manual-CoT (Wei et al. 2022): It involves crafting manually designed reasoning chains, tailored with specific demonstrations for each dataset.

To show the superiority of our method, the number of prompts used in our work was kept less than or equal to those used in other methods, since more prompts typically yield better performance. The num of prompts used in the CDW-CoT framework were: 6 (CommonsenseQA), 5 (StrategyQA), 4 (Letter), 6 (Coin), 5 (MultiArith), and 4 (AQuA).

Data Split and Number of Clusters Identification Datasets were divided into training, evaluation, and test subsets with proportions of approximately $60 \%$ , $2 5 \%$ , and $1 5 \%$ , respectively (Wang et al. 2022b). After dividing the data, we identified the number of clusters according to the Auto-CoT setup, and then adjusted the number of clusters for certain datasets from the default 8 to 3, as shown in Table 2.

Table 2: Data Split and Number of Clusters Statistics.   

<html><body><table><tr><td>Dataset</td><td>Total</td><td>Train</td><td>Eval</td><td>Test</td><td>#Clusters</td></tr><tr><td>CSQA</td><td>1,221</td><td>725</td><td>312</td><td>184</td><td>7</td></tr><tr><td>StrategyQA</td><td>2,290</td><td>1,362</td><td>584</td><td>344</td><td>6</td></tr><tr><td>Letter</td><td>500</td><td>297</td><td>128</td><td>75</td><td>4</td></tr><tr><td>Coin</td><td>500</td><td>297</td><td>128</td><td>75</td><td>3</td></tr><tr><td>MultiArith</td><td>600</td><td>357</td><td>153</td><td>90</td><td>3</td></tr><tr><td>AQuA</td><td>254</td><td>150</td><td>65</td><td>39</td><td>4</td></tr></table></body></html>

Prompt Engineering Configuring prompts effectively is crucial for training models across diverse datasets. This phase involved three key parameters:

• Pool Size: We maintained a consistent pool of 40 potential prompts for each dataset to enable thorough exploration of diverse reasoning pathways. • Sample Size: During training, each instance was tested against five unique prompt combinations, assessing the effectiveness of various configurations. • Temperature: A temperature of 0.3 was used to optimize prompt selection during testing.

Our primary metric, exact match accuracy, measures the degree responses correctly answer the instances across various reasoning domains. As detailed in Table 1, our results demonstrate substantial performance improvements across all the tasks and both models used, underscoring the effectiveness of the CDW-CoT framework. For both the LLaMA2 (13B) and LLaMA3 (8B) models, we compared our method against the best baseline method among the three we evaluated.

# Main Results

CDW-CoT consistently outperforms traditional CoT methods across various reasoning tasks and datasets, improving accuracy for both LLaMA2 and LLaMA3 models. The detailed results are as follows:

Commonsense Reasoning: For CommonsenseQA, CDW-CoT improved exact match accuracy by $1 0 . 3 2 \%$ $( 5 1 . 0 9 \%  6 1 . 4 1 \% )$ ) on LLaMA2 (13B) and by $2 . 5 8 \%$ $( 6 9 . 5 7 \% \to 7 2 . 1 5 \% )$ on LLaMA3 (8B). For StrategyQA, CDW-CoT increased accuracy by $9 . 5 8 \%$ $( 6 0 . 4 8 \% ~ $ $7 0 . 0 6 \% )$ on LLaMA2 (13B) and by $0 . 7 2 \%$ $( 6 6 . 7 2 \% $ $6 7 . 4 4 \%$ on LLaMA3 (8B).

Symbolic Reasoning: In the Letter dataset, CDW-CoT significantly improved accuracy by $5 1 . 8 7 \%$ $( 3 0 . 8 0 \% $ $8 2 . 6 7 \% )$ on LLaMA2 (13B) and by $6 . 0 7 \%$ $7 6 . 6 7 \% $ $8 2 . 6 7 \% )$ on LLaMA3 (8B). In the Coin dataset, CDWCoT improved accuracy by $9 . 5 3 \%$ $( 5 1 . 8 0 \%  6 1 . 3 3 \% )$ on LLaMA2 (13B) and by $9 . 9 0 \%$ $6 0 . 8 0 \%  7 0 . 7 0 \% )$ on LLaMA3 (8B).

Mathematical Reasoning: CDW-CoT recorded a $1 4 . 5 6 \%$ increase $( 7 1 . 0 0 \%  8 5 . 5 6 \% )$ on MultiArith with LLaMA2 (13B) and a $5 . 1 7 \%$ increase $( 9 0 . 0 0 \% \to 9 5 . 1 7 \% )$ ) on LLaMA3 (8B). For AQuA, accuracy improved by $5 . 5 8 \%$ $( 3 0 . 3 1 \%  3 5 . 8 9 \% )$ on LLaMA2 (13B) and by $1 0 . 9 4 \%$ $( 4 8 . 0 3 \%  5 8 . 9 7 \% )$ on LLaMA3 (8B).

These results demonstrate that the CDW-CoT framework effectively enhances performance across a wide range of reasoning tasks, including commonsense reasoning, mathematical reasoning, and symbolic reasoning. The framework consistently outperforms Zero-Shot-CoT, Manual-CoT and even Auto-CoT and shows significant improvements across different LLMs, as confirmed by the results in Table 1.

# Ablation Study

To evaluate the effectiveness of each component of our model, we conduct the experiments with different model versions by removing the corresponding component.

The three model versions are described as follows:

• Distance Weighting (Dist-W): This version implements the complete model, using clustering to generate optimal prompt probability distributions tailored to each category. It adjusts the reasoning process for each test instance by employing distance-weighted prompt distribution, enhancing specificity based on proximity to cluster centers. • Nearest Cluster (Near-C): This streamlined approach assigns the nearest cluster’s prompt distribution to each test instance, omitting the computational complexity of distance weighting. This method emphasizes efficiency while still utilizing the benefits of clustering. • No Clustering (No-Clust): This baseline approach without clustering phase uses a single, global optimal prompt probability distribution, derived from the entire dataset and applied uniformly across all test instances.

The effectiveness of each model version was assessed using the same setups with the main experiments.

Table 3: Ablation study of different model versions across datasets, showing percentage accuracies.   

<html><body><table><tr><td>Dataset</td><td>Dist-W</td><td>Near-C</td><td>No-Clust</td></tr><tr><td>CSQA</td><td>61.41</td><td>53.26</td><td>60.33</td></tr><tr><td>StrategyQA</td><td>70.06</td><td>67.44</td><td>67.15</td></tr><tr><td>Letter</td><td>82.67</td><td>81.33</td><td>81.11</td></tr><tr><td>Coin</td><td>61.33</td><td>58.67</td><td>56.00</td></tr><tr><td>MultiArith</td><td>85.56</td><td>77.78</td><td>77.78</td></tr><tr><td>AQuA</td><td>35.89</td><td>28.21</td><td>23.08</td></tr></table></body></html>

The results of our ablation study, as shown in Table 3, clearly demonstrate the effectiveness of each component in the CDW-CoT framework. The important role of the DistW method is evident, as it consistently achieves the highest accuracy across all the datasets. This method highlights the importance of clustering and distance-based prompt optimization, allowing the model to adapt its reasoning pathways effectively by considering the unique aspects of each test instance. The Distance Weighting method is particularly successful in complex tasks such as MultiArith and Letter, where precise and context-aware reasoning is crucial.

The Near-C model, which only relies on the nearest cluster’s prompt distribution without distance weighting, is limited in its capability to effectively use the optimal prompt probability distributions across multiple clusters. This constraint leads to a $5 . 0 4 \%$ decrease across datasets averagely, as shown in Table 3.

The No-Clust model uses a uniform prompt distribution for all the instances, which reduces its effectiveness. Its lower performance in Table 3, with an average decrease of approximately $5 . 2 4 \%$ compared to the full model, highlights the importance of constructing category-specific prompt distributions to address the distinct demands of various data categories effectively.

This ablation study confirms the robustness of our CDWCoT framework, demonstrating that each component, particularly clustering and distance weighting, plays a crucial role in enhancing the reasoning performance.

# Sensitivity Analysis of Temperature

Our analysis investigates the impact of the temperature parameter $T$ in our framework.

We explored the effects of temperature settings ranging from 0.1 to 1.0 measured with accuracy. The experiments were conducted using the LLaMA2(13B) model on StrategyQA and MultiArith.

100 Method 95 CDW-CoT on MultiArith BestBaseline on MultiArith 90 CDW-CoT on StrategyQA 85.56% 84.44% -Best Baseline on StrategyQA   
85 85.56% 82.22% 81.11%   
80 78.89% 80.00% 81.11%   
75 73.332 71.00% 70.06% 70 67.15% 67.44% 67.73% 66.57% 65 68.60% 67.73% 67.44% 66.28% 66.57% 60.48%   
60 55 0.2 0.4 0.6 0.8 1.0 Temperature

Temperature plays a pivotal role in the CDW-CoT framework, as evidenced by our detailed results depicted in Figure 3. At a lower temperature of 0.1, the model becomes overly sensitive, disproportionately focusing on the nearest cluster even when it may not be the most relevant. This excessive sensitivity often leads to inaccuracies, especially when the query is ambiguously positioned relative to multiple clusters.

Conversely, at a temperature setting of 1.0, the model’s performance declines due to an overly generalized approach that incorporates too much irrelevant cluster information. This almost uniform focus reduces the response accuracy and fails to fully leverage the optimal prompt distributions for each cluster.

Throughout all the temperatures, the CDW-CoT consistently surpasses the best baseline among the conventional methods compared, highlighting its superior reasoning capabilities and robust adaptability. The model achieves optimal performance at a temperature of 0.3, striking an effective balance between specificity and sensitivity. This setting allows the model to accurately concentrate on the most pertinent cluster features, thus maximizing the accuracy and maintaining the flexibility across a variety of reasoning tasks.

it controls the number of prompt candidates extracted from clusters.

Similar to the sensitivity analysis of temperature effects, we make the analysis to explore the impact of Pool Size $S$ on the CDW-CoT framework. This parameter is important as

We varied the pool size from 10 to 40 to assess how the quantity of available prompt candidates impacts the model’s performance. This investigation was conducted using the LLaMA2(13B) model on two datasets: CommonsenseQA and MultiArith.

# Impact of Pool Size on CDW-CoT

100 Method CDW-CoTon MultiArith 90 Best Baseline on MultiArith CDW-CoT on CSQA - Best Baseline on CSQA 84.44% 85.56%   
80 82.22% 82.22% 73.33% 78.89% 78.89% 71.00% 70 61.41% 58.15%58.70% 58.70% 6054.89% 55.43% 56.52% 51.09% 50 10 15 20 25 30 35 40 Pool Size

Based on the analysis and trends from Figure 4, we observe that increasing the pool size consistently enhances the model’s performance across both datasets. As expected, a larger candidate pool allows the CDW-CoT framework to better explore diverse reasoning paths. For MultiArith, accuracy steadily improves from $7 3 . 3 3 \%$ at a pool size of 10 to $8 5 . 5 6 \%$ at a pool size of 40. Similarly, for CommonsenseQA, accuracy increases from $5 4 . 8 9 \%$ to $6 1 . 4 1 \%$ as the pool size grows.

While a larger pool increases reasoning diversity and improves the accuracy, it also increases the computational costs. In our experiments, we chose a pool size of 40 as an optimal balance between performance gains and efficiency. This selection ensures that the CDW-CoT framework achieves high accuracy across different reasoning tasks without incurring excessive computational overhead, effectively balancing decision quality and resource use.

# Conclusion

In this paper, we propose a novel CoT method named CDWCoT to enhance the adaptability and accuracy of LLMs in complex reasoning tasks. Our method introduces the clustering to categorize the datasets into tailored prompt pools, improving the representative ability to diverse data characteristics. It calculates an optimal prompt probability distribution for each cluster, enabling targeted reasoning that aligns with its unique characteristics. By designing the distanceweighted prompt selection, CDW-CoT dynamically adjusts the reasoning strategies based on the proximity to cluster centers, demonstrating superior performance over traditional methods across six datasets. Future work includes reducing computational overhead and extending applicability to multimodal tasks like image-text reasoning.