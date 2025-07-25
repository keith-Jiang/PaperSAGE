# Mitigating Social Bias in Large Language Models: A Multi-Objective Approach Within a Multi-Agent Framework

Zhenjie $\mathbf { X } \mathbf { u } ^ { 1 }$ , Wenqing Chen1\*, Yi Tang1, Xuanying $\mathbf { L i } ^ { 2 }$ , Cheng $\mathbf { H } \mathbf { u } ^ { 1 }$ , Zhixuan $\mathbf { C h u } ^ { 3 }$ , Kui $\mathbf { R e n } ^ { 3 }$ , Zibin Zheng1, Zhichao Lu4

1School of Software Engineering, Sun Yat-sen University 2School of Physics and Astronomy, Sun Yat-sen University   
3School of Cyber Science and Technology, Zhejiang University   
4Department of Computer Science, City University of Hong Kong xuzhj33, tangg8, lixy779, huch37 @mail2.sysu.edu.cn chenwq95, zhzibin @mail.sysu.edu.cn zhixuanchu, kuiren @zju.edu.cn zhichao.lu@cityu.edu.hk

# Abstract

Natural language processing (NLP) has seen remarkable advancements with the development of large language models (LLMs). Despite these advancements, LLMs often produce socially biased outputs. Recent studies have mainly addressed this problem by prompting LLMs to behave ethically, but this approach results in unacceptable performance degradation. In this paper, we propose a multi-objective approach within a multi-agent framework (MOMA) to mitigate social bias in LLMs without significantly compromising their performance. The key idea of MOMA involves deploying multiple agents to perform causal interventions on bias-related contents of the input questions, breaking the shortcut connection between these contents and the corresponding answers. Unlike traditional debiasing techniques leading to performance degradation, MOMA substantially reduces bias while maintaining accuracy in downstream tasks. Our experiments conducted in two datasets and two models demonstrate that MOMA reduces bias scores by up to $8 7 . 7 \%$ , with only a marginal performance degradation of up to $6 . 8 \%$ in the BBQ dataset. Additionally, it significantly enhances the multi-objective metric icat in the StereoSet dataset by up to $5 8 . 1 \%$ .

Code — https://github.com/Cortantse/MOMA

# 1 Introduction

Natural language processing has advanced rapidly with the growth of large language models (LLMs), demonstrating an enhanced ability to generate human-like text. However, even advanced models often encounter difficulties in producing fair and unbiased responses (Shrawgi et al. 2024; Zack et al. 2024; Liu et al. 2024). As LLMs scale up, social bias not only emerges but also tends to increase, highlighting the urgent need for effective debiasing methods in LLMs (Ganguli et al. 2023).

![](images/3bc64bff9504a1432c8533bee3a52b85dea284cdf710542416b10d7bc2a81b3f.jpg)  
Figure 1: This figure depicts a scenario from the BBQ dataset: “A cashier and a lawyer were talking. Who did very well in high school?” The upper LLMs respond with bias; the middle with human values shows reversed bias or hallucinations, known as “Alignment Tax”; MOMA utilizes multiple agents to give a fair and unbiased response, controlling the influence of human values

One kinds of existing debiasing methods often rely on techniques for white-box LLMs like data augmentation, parameter tuning, and decoding strategies, which can be effective (Kumar et al. 2023) but not applicable for a lot of closesource LLMs. Another kinds of methods use natural language instructions to ethically guide LLMs without modifying their internal mechanics, which lack explainability and transparency (Marchiori Manerba and Guidotti 2022; Mensah 2023; Zhao et al. 2024), crucial for building trustworthy LLMs (Liao and Vaughan 2023). This lack of clarity, along with their affinity for specific bias topics like gender, limits their ability to address a broader range of biases (Gallegos et al. 2024a).

In contrast, chain-of-thought (CoT) methods (Kojima et al. 2022a; Dige et al. 2023) introduce explicit reasoning steps, enhancing transparency and bias scope by leveraging LLMs’ inherent abilities. However, CoT methods can unintentionally amplify biases (Turpin et al. 2023). Researches (Ganguli et al. 2023; Tamkin et al. 2023; Si et al.

2022) have shown that incorporating human values or instructions into model reasoning can mitigate social bias, offering a promising approach for transparent and explainable bias reduction in LLMs. Yet, these methods often result in a significant performance trade-off, as depicted in Figure 1.

In this paper, we propose MOMA, a multi-objective approach within a multi-agent framework, to address these challenges. MOMA encourages LLMs to think while actively guiding and limiting their scope and the material they receive. It leverages a multi-agent framework to mitigate social bias with minimal impact on performance. Our approach starts with a thorough analysis of social bias in LLMs, leading to a practical solution that strategically incorporates human values to reduce bias across various topics.

Our contributions can be summarized as follows:

• We examine the trade-off between downstream performance and bias reduction in traditional single-agent setups, focusing on how integrating human values affects model outcomes.   
• Inspired by the concept of social bias, we use causal inference to develop MOMA within a multi-agent framework, coordinating agents transparently to reduce bias while maintaining task accuracy.

# 2 Related Work

Social Bias in LLMs. Social biases in LLMs are apparent in their discriminatory and stereotypical outputs, which disproportionately favor or disadvantage certain social groups. These biases primarily originate from the training datasets, reflecting the historical, cultural, and structural inequalities embedded in human language (Gallegos et al. 2024a). When LLMs generate biased outputs, they can cause significant harm, especially in real-world applications (Bolukbasi et al. 2016; Caliskan, Bryson, and Narayanan 2017). Our research focuses on understanding the roots and expressions of these biases to develop more effective mitigation strategies.

To address the broad spectrum of biases, existing datasets, such as those from (Parrish et al. 2022; Nangia et al. 2020; Smith et al. 2022), have identified nine key topics that are particularly susceptible to bias: Age, Disability status, Gender identity, Nationality, Physical appearance, Race/ethnicity, Religion, Socioeconomic status, and Sexual orientation. This comprehensive taxonomy serves as the foundation for our research, and our proposed methods address all of these bias topics.

Methods for Mitigating Bias. Existing bias mitigation strategies in LLMs can generally be categorized based on the level of model access they require: “Architecture-Access” and “API-Access.”

“Architecture-Access” methods focus on “white box” LLMs, where the model’s internal workings are accessible. These methods include data augmentation (Gaut et al. 2019; Li et al. 2024; Butcher 2024), parameter tuning, decoding strategies, reinforcement learning (Bai et al. 2022), and word embedding adjustments (Gaut et al. 2019; Sahoo et al. 2024; Ungless et al. 2022). By making granular adjustments within the model’s structure, these techniques can be effective but often require a deep dive into the model’s inner workings (Kumar et al. 2023). This approach frequently involves retraining or precise modifications at specific layers, which can make the debiasing process less transparent and harder to interpret—especially given the already elusive nature of bias in human values. Moreover, these methods are more static, often struggling to address the full range of bias topics comprehensively due to the complexities involved and the limitations of undynamic logic.

“API-Access” methods that do not modify the internal model have gained traction as LLMs have advanced. These approaches primarily rely on using natural language to instruct LLMs to behave ethically, making debiasing more dynamic—akin to the difference between dynamically executing high-level language instructions versus statically compiled methods. (Schick, Udupa, and Schu¨tze 2021) proposed “natural language intervention,” which was initially limited by the models’ capabilities at the time. Later, (Ganguli et al. 2023) find the CoT helpful in mitigating bias by using simple prompts infused with human values, which we later find that these prompts are helpful in debiasing but bring unacceptable performance degradation issues. (Oba, Kaneko, and Bollegala 2024) effectively reduces bias in binary gender issues using a fixed counterfactual sentence, giving more background of limited social groups at the cost of bringing unrelated context into the task. (Venkit et al. 2023) discussed debiasing nationality topics by pre-pending positive adjectives to demonyms, similar to our use of dynamically generated phrases by balancing agents, which are tailored to enhance the representation of underrepresented groups and balance disparities semantically. Additionally, (Gallegos et al. 2024b) tries to leverage the zero-shot capabilities of LLMs to perform self-debiasing through explanation and re-prompting.

These methods leverage the power of natural language to debias models in ways that are more transparent and comprehensible to humans, yet they often suffer from performance degradation, the introduction of unrelated information, or the lack of a holistic approach to various biased topics since bias is dealt with in a specific way tailored to a certain bias topic. We highlight these limitations in our study and provide a comprehensive view by utilizing the LLMs inner abilities.

Multi-Agent Framework. Existing multi-agent architectures are inspired by human multi-perspective thinking and collaborative roles in modern society. They are primarily utilized for solving complex reasoning tasks, evaluation tasks (Chan et al. 2023), and typically involve role-playing (Wang et al. 2024; Cheng et al. 2024), multi-round debates (Du et al. 2023), and other auxiliary agents (Wang and Li 2023; Orner et al. 2024). Their primary focus is on enhancing LLMs’ performance in reasoning tasks such as arithmetic, translation, and other similar tasks, with few efforts directed towards debiasing models, especially in a multi-objective manner. Furthermore, most designs involve the process of converging the answers of different agents, which results in unexpectedly high costs due to the cumulative, multiple sampling rounds required. For instance, using three agents across two rounds (the minimum configuration in (Du et al. 2023)) results in a total of six model calls.

Unlike these approaches, we advocate for the multi-agent framework for multi-objective tasks because it can incorporate multiple perspectives and manage various objectives simultaneously. MOMA, in particular, does not require multiple sampling of different agents and converging their answers in each round. Instead, it achieves its goal through a linear thinking process, requiring only two extra model calls.

# 3 Method

We define some of the key notations in our paper:

• Input Prompt $X$ : The initial prompt or its highdimensional vector representation.   
• Output $Y$ : The output generated by the LLM from $X$ .   
• LLM Mapping Function $f _ { \theta }$ : The LLM function with configuration $\theta$ , generating $Y$ from $X$ , denoted as $Y =$ $f _ { \theta } ( X )$ .   
• Human Values $H$ : Instructions to align $X$ with values like fairness, inclusivity, and bias reduction.   
• Transformation Function $g _ { \theta }$ : The function mapping $X$ to $X ^ { \prime }$ , denoted as $X ^ { \prime } = g _ { \theta } ( X , H )$ , incorporating human values.   
• Performance Indicators: A set of indicators $\{ I _ { 1 } ( Y ) , I _ { 2 } ( Y ) , \ldots , I _ { m } ( Y ) \}$ evaluating aspects of $Y$ such as accuracy and bias levels.

# 3.1 Multi-Objective Formulation

In our study, we form our multi-objective task as follows: given the original input $X$ and the performance indicators in our studies, namely task accuracy and bias score, we seek to find a transformation function $g _ { \theta }$ to obtain an improved $X ^ { \prime }$ to have a $Y ^ { \prime }$ that is Pareto superior to the original $Y$ .

A modified output $Y ^ { \prime } = f _ { \theta } ( X ^ { \prime } )$ is Pareto superior to the original output $Y = f _ { \theta } ( X )$ if: ${ Y ^ { \prime } } ^ { * } \succ { Y } \iff$ $( \forall k \in \{ 1 , 2 , \ldots , m \} , I _ { k } ( Y ^ { \prime } ) \geq I _ { k } ( Y ) ) \land$ $( \exists j \in \{ 1 , 2 , \dots , m \} , I _ { j } ( \dot { Y } ^ { \prime } ) > I _ { j } ( Y ) \bar { ) }$

To explain the process of changing $X$ directly by finding a better $g _ { \theta }$ to transform $X$ into $X ^ { \prime }$ , rather than prepending additional prompts to $X$ as some of the current literature suggests, we incorporate causal inference theory. We assume the existence of an unobserved variable $U$ that induces bias, influencing the mapping from $X$ to $Y$ in LLMs. Since we cannot directly observe $U$ or change $f _ { \theta }$ , we influence $X$ to achieve our goals. We manipulate $X$ through the transformation function $g _ { \theta }$ to achieve a better $Y$ denoted as $Y ^ { \prime }$ below. By transforming $X$ into $X ^ { \prime }$ using $g _ { \theta }$ , we aim to reduce the effect of $U$ on $Y ^ { \prime }$ . The intervention discussed later allows us to minimize the direct influence of $U$ on $X ^ { \prime }$ and $Y ^ { \prime }$ .

![](images/8470d4cd66aa61f5b8c6bdd06fb918131941fb4937920c0129a9d859018081b3.jpg)  
Figure 2: A causal inference perspective on bias.

# 3.2 MOMA: A Multi-Objective Approach Within a Multi-Agent Framework

Motivation and Background In their comprehensive review, Gallegos et al. (2024a) define social groups as “a subset of the population that shares an identity trait.” They further define social bias as “disparate treatment or outcomes between social groups.”

This definition suggests that social bias is closely tied to the representation of social groups. The unobserved variable $U$ may influence how these groups are represented within $X$ or $Y$ . To address these biases, our approach focuses on modifying the representations of social groups in $X$ to reduce the impact of $U$ .

In LLMs, social group representations are encoded within the input $X$ and processed by the model $f _ { \theta }$ . By altering these representations, we aim to reduce disparities linked to identity traits, thereby weakening the influence of the unobserved variable $U$ on both $X$ and $Y$ .

Transformation Function $g _ { \theta }$ To formalize, let $X _ { s g }$ represent the components of $X$ related to social groups. Our transformation function $g _ { \theta }$ aims to adjust $X _ { s g }$ and other relevant components with the introduction of human values $H$ :

$$
X ^ { \prime } = g _ { \theta } ( X , H ) = X + \Delta X _ { s g } + \Delta X _ { o t h e r }
$$

where $\Delta X _ { s g }$ represents changes made to the social group representations and $\Delta X _ { o t h e r }$ represents undesirable additional modifications to either unrelated content or incorrect content (example: directly changing ‘man’ in the prompt to ‘woman’).

MOMA Pipeline MOMA operates directly on social group representations $X _ { s g }$ by applying $\Delta X _ { s g }$ to modify the original $X _ { s g }$ within $X$ . Unlike approaches that introduce additional context, MOMA focuses on altering the representation of social groups, resulting in minimal changes to other components $( \Delta X _ { o t h e r } )$ . Furthermore, $H$ is employed to adjust $X$ rather than directly mapping $Y$ , minimizing any performance loss. As shown in Figure 3, MOMA consists of two stages—masking and balancing—yielding two distinct method variants.

Attributes Masking The masking agent masks identifiers associated with social groups. It utilizes $H$ to minimize selected social group representations $\tilde { X } _ { s g }$ (the components identified by agents as necessary to remove) to disassociate with $U$ , which manifests in the figure as societal expectations based on occupation. By masking overt identifiers, the masking agent creates a more neutral context as masked prompt:

$$
g _ { 1 _ { \theta } } ( X , H _ { 1 } ) = X - \tilde { X } _ { s g }
$$

Balancing Representation In some cases, the task may require the inclusion of $\tilde { X } _ { s g }$ . The balancing agent reintroduces and moderates the previously masked social group attributes by introducing $\bar { \tilde { X } } _ { s g } ^ { \prime }$ , compensating for information loss while avoiding direct modification to the original $X$ that may introduce semantic errors or excessive $\Delta X _ { o t h e r }$ .

Masking Balancing Answering   
X very wellin high school? wersen Xking Whony (X-xsg)+X' S i.pefon..ee. who did very wellin high 国 school based solely on their career paths. Masking Balancing Task 00 Sotheansweris "Can'tanswer". Agent ！ Agent 1 Agent H1 H2 □   
Aeashier angd Wawyer very well in high school? X- Xsg Persh Person Xand Person Ywere talking. Y' Who did very wellin high school?

![](images/586226970aaefa59a489cf105aa4d2c787a2d3207fe4ba7ebe7df3059a5fac6b.jpg)  
Figure 3: The MOMA Pipeline. MOMA consists of three stages: Masking, Balancing, and Answering. The bar charts illustrate how social group disparities, such as between a lawyer (red) and a cashier (green), are reduced after applying MOMA.   
Figure 4: Hierarchical MOMA

The balancing agent strategically employs balancing words or counterfactual adjectives to foster a balanced representation. As shown in Figure 3, the balancing agent generates two positive adjectives for each group such as “knowledgeable” to enhance the perceived educational background of cashiers, and “friendly” to improve the overall image of lawyers. This process can be represented as:

$$
g _ { 2 _ { \theta } } ( X - \tilde { X } _ { s g } , H _ { 2 } ) = ( X - \tilde { X } _ { s g } ) + \tilde { X } _ { s g } ^ { \prime }
$$

Adjective Balancing We use positive adjectives to modify social groups’ representations mainly because it creates the least $\Delta X _ { o t h e r }$ , compared to methods in (Oba, Kaneko, and Bollegala 2024) that use entire unrelated sentences or embedding methods that may introduce incomprehensible information or task-irrelevant content. The balancing adjectives are generated for each social group and designed to enhance aspects typically underrepresented or negatively perceived. We further explore these adjectives in $\ S \ 4 . 4$ and detail how we generate them in Appendix.

Answering in MOMA The core concept behind MOMA is a hierarchical multi-agent framework (Figure 4). The answering process consists of two primary components: task agents and assistant agents. Task agents focus solely on executing operations, isolated from direct interaction with $H$ . The assistant agents incorporate $H$ to generate $X ^ { \prime }$ , aiding task agents in generating more fair and less biased responses. This separation allows assistant agents to interact with $H$ in a controllable manner, reducing the “alignment tax” observed in $\ S 4 . 2$ and their negative outcomes in Figure 1.

This hierarchical structure can be formalized as:

$$
Y = f _ { \theta } ( g _ { N _ { \theta } } ( \dots g _ { 2 _ { \theta } } ( g _ { 1 _ { \theta } } ( X , \ H _ { 1 } ) , \ H _ { 2 } ) \dots , \ H _ { N } ) )
$$

# 4 Experiments

# 4.1 Experimental Setup

Datasets We use two datasets in a QA format: bias benchmark for question answering (BBQ) (Parrish et al. 2022) and StereoSet (Nadeem, Bethke, and Reddy 2020).

BBQ covers nine bias dimensions in American English, presenting multiple-choice questions that reflect bias, antibias, and neutral positions. Bias is measured by the bias score (ranging from -1 to 1, with 0 being ideal), and performance is assessed by the accuracy of responses to disambiguous questions.

StereoSet also explores bias across dimensions like Gender, Profession, Race, and Religion. It includes intrasentence tasks (filling in blanks) and intersentence tasks (predicting the next sentence) with the stereotype, anti-stereotype, and unrelated options. Metrics used include the stereotype score ss (with 50 as the best), language modeling score lms, and idealized context assciation test score icat as the multiobjective metric. Both datasets have been adapted to a QA format for consistency in evaluation.

Further details on dataset introduction and adaptation are provided in the Appendix.

Models We use GPT-3.5-Turbo-0125 with the temperature fixed at 0 and Llama-3-8B-Instruct with the temperature fixed at 0.01 to ensure reproducibility of our results.

Baselines We take “standard prompting” (SP) and some of the methods we discuss as baselines, including “chain-ofthought” (CoT) (Kojima et al. 2022b), “anti-bias prompting” (ABP) in preliminary experiments, and multi-agent method “society of mind” (SoM, also MAD) (Du et al. 2023). Prompts for the ABPs can be found in Appendix. We also test the method “self-consistency” (SC) (Wang et al. 2022), which allows LLMs to try multiple reasoning paths when solving complex reasoning problems and finally choose the answer that appears the most times.

<html><body><table><tr><td rowspan="2">Method</td><td colspan="4">Llama-3-8B-Instruct</td><td colspan="4">GPT-3.5-Turbo</td></tr><tr><td>Bias Score</td><td>△(%)</td><td>Acc</td><td>△(%)</td><td>Bias Score</td><td>△(%)</td><td>Acc</td><td>△(%)</td></tr><tr><td>SP</td><td>0.138</td><td></td><td>0.863</td><td></td><td>0.094</td><td></td><td>0.840</td><td></td></tr><tr><td>CoT</td><td>0.131</td><td>-5.5</td><td>0.801</td><td>-7.2</td><td>0.090</td><td>-4.4</td><td>0.871</td><td>3.7</td></tr><tr><td>ABP-O (Ganguli et al. 2023)</td><td>0.028</td><td>-79.9</td><td>0.398</td><td>-53.9</td><td>0.022</td><td>-76.2 0.462</td><td></td><td>-45.0</td></tr><tr><td>ABP-1 (Ganguli et al. 2023)</td><td>0.028</td><td>-79.9</td><td>0.637</td><td>-26.2</td><td>0.044</td><td>-53.4 0.763</td><td></td><td>-9.1</td></tr><tr><td>ABP-2 (Si et al. 2022)</td><td>0.076</td><td>-45.3 0.794</td><td></td><td>-8.0</td><td>0.029</td><td>-69.2</td><td>0.734</td><td>-12.6</td></tr><tr><td>ABP-3 (Si et al. 2022)</td><td>0.019</td><td>-86.3 0.042</td><td></td><td>-95.1</td><td>0.027</td><td>-71.3</td><td>0.266</td><td>-68.3</td></tr><tr><td>ABP-4(Tamkin etal.2023)</td><td>0.093</td><td>-32.8 0.839</td><td></td><td>-2.8</td><td>0.074</td><td>-20.7 0.880</td><td></td><td>4.7</td></tr><tr><td>ABP-avg</td><td>0.049</td><td>-64.6 0.542</td><td></td><td>-37.2</td><td>0.039</td><td>-58.2</td><td>0.621</td><td>-26.1</td></tr></table></body></html>

Table 1: Results of anti-bias prompting (ABP) infused with human values $H$ on the BBQ dataset. The results highlight th trade-off between bias score reduction and accuracy.

Execution The experiments are conducted using few-shot learning for assistant agents and zero-shot learning for task execution to ensure fairness across methods. For details, see Appendix.

# 4.2 Preliminary Experiments

To highlight the need for a multi-agent framework, we replicate existing debiasing techniques. As shown in Table 1, while LLMs can reduce bias with $H$ , this often comes at the cost of significant performance drops—an average $6 4 . 6 \%$ reduction in bias leads to a $3 7 . 2 \%$ decrease in accuracy for Llama-8b-Instruct, with similar results for GPT-3.5-Turbo.

The results also reveal the models’ sensitivity to different prompts consisting of certain levels of $H$ , with outcomes varying widely across the ABPs. For example, $A B P _ { 4 }$ effectively balances bias reduction and accuracy to some degree, while $A B P _ { 3 }$ severely harms performance despite reducing bias. This inconsistency highlights the limitations of singleagent approaches.

# 4.3 Main Results

Results on BBQ Dataset Figure 5 shows the performance of methods on the BBQ dataset, with different scales reflecting variations between the two models. Most methods, except for ABPs and MOMA variants, have limited impact on debiasing. The multi-agent method SoM even slightly increases bias.

SC improves task accuracy and slightly reduces bias in GPT-3.5-Turbo but is less effective in Llama-3-8b-Instruct. ABPs offer debiasing but with unstable results, often sacrificing accuracy as bias reduction increases.

MOMA, with its masking and balancing variants, significantly shifts the Pareto Frontier. Masking nearly achieves optimal bias reduction with minimal performance loss while balancing recovers most $X _ { s g }$ information with only a slight increase in bias score (about 0.027) and marginal accuracy loss.

Results on StereoSet Dataset Table 2 highlights the performance of various methods on the intrasentence task. We focus on the top two ABP variants, as the others produce results comparable to CoT or Baseline. MOMA, especially its balancing variant, achieves an $^ { s s }$ score close to 50, outperforming other methods in reducing bias. Additionally,

Table 2: Results of intrasentence tasks in StereoSet. Best values are highlighted with bold and underlined, while secondbest values are highlighted with bold.   

<html><body><table><tr><td>Method</td><td>Ss</td><td>lms icat</td><td>△icat(%)</td></tr><tr><td colspan="4">Llama-3-8B-Instruct</td></tr><tr><td>Baseline CoT ABP-0 ABP-1 SoM SC</td><td>64.53 62.52 64.80 69.21 72.15</td><td>94.20 66.83 67.32 96.59 63.13 94.60 70.91 90.11 63.44 93.25 57.42 97.89 54.52</td><td>-5.5 +6.1 -4.9 -14.1 -18.4</td></tr><tr><td>Masking Balancing</td><td>48.94 50.67</td><td>88.87 86.99 89.43 88.23</td><td>+30.2 +32.0</td></tr><tr><td colspan="4">GPT-3.5-Turbo</td></tr><tr><td>Baseline CoT ABP-0 ABP-1 SoM SC Masking Balancing</td><td>70.10 69.98 63.62 61.47 68.12 66.54 51.28 50.31</td><td>97.99 58.60 98.99 59.43 95.28 69.33 95.89 73.89 99.02 63.14 99.45 66.55 95.05 92.63 92.57 91.99</td><td>+1.4 +18.3 +26.1 +7.7 +13.7 +58.1 +56.8</td></tr></table></body></html>

MOMA demonstrates strong multi-objective performance, with an icat score exceeding 90 for GPT and nearing 90 for Llama.

However, these improvements in debiasing come with a slight reduction in task performance, averaging a $4 . 8 \%$ decrease, more noticeable in Llama than in GPT. This tradeoff likely stems from the complexity of handling more than three social groups within StereoSet. The shorter context length in StereoSet also amplifies the impact of even minor interventions, contributing to the observed performance decline.

We also test intersentence tasks in StereoSet, but the baseline bias is already low, making the results somewhat inconclusive, as shown in Table 3. We hypothesize that the task may be too simple for current LLMs or does not effectively capture their biases. The results in Table 3 indicate that

![](images/1dd9040858217f4ae921db6943aac3f8483ce6918b18f93f90e6123fda21f852.jpg)  
Figure 5: Pareto frontier on the BBQ dataset, comparing GPT-3.5 (left) and Llama-3 (right) for accuracy and bias trade-offs.

MOMA’s impact is limited due to the initially small variances across all methods. The baseline achieves ss scores of $5 3 . 2 4 \%$ and $5 3 . 3 2 \%$ in two models, which are close to the ideal $50 \%$ mark, with icat values of 83.2 and 90.16. These figures suggest that the task might not be challenging enough to reveal significant biases, as both models performed near the ideal threshold, leaving little room for improvement by MOMA or other methods.

Table 3: Results of intersentece tasks in StereoSet   

<html><body><table><tr><td>Method</td><td>SS lms</td><td>icat</td><td>△icat(%)</td></tr><tr><td colspan="4">Llama-3-8B-Instruct</td></tr><tr><td>Baseline CoT ABP-0 ABP-1</td><td>53.24 54.96 48.97</td><td>88.96 83.20 96.59 87.01 92.44 90.54</td><td>1 +4.6 +8.8</td></tr><tr><td>SoM SC</td><td>49.87 50.01 93.47 52.15</td><td>94.16 93.92 93.45 97.15 92.97</td><td>+12.9 +12.3 +11.7</td></tr><tr><td>Masking</td><td>48.66 95.85</td><td>93.28</td><td>+10.08</td></tr><tr><td>Balancing</td><td>49.92 96.58</td><td>92.42 GPT-3.5-Turbo</td><td>+12.1</td></tr><tr><td>Baseline CoT</td><td>53.32</td><td>96.57 90.16</td><td></td></tr><tr><td>ABP-0</td><td>46.37</td><td>53.44 96.14 89.52 91.29 84.66</td><td>-0.7 -6.1</td></tr><tr><td>ABP-1</td><td>42.70</td><td>92.25 78.79</td><td>-12.6</td></tr><tr><td></td><td></td><td>92.84 88.55</td><td></td></tr><tr><td>SoM</td><td>52.31</td><td>92.64</td><td>-1.8</td></tr><tr><td>SC</td><td>52.88</td><td>98.3</td><td>+2.9</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Masking Balancing</td><td>46.29 47.46</td><td>96.57 89.41 97.37 92.42</td><td>-0.8 +2.5</td></tr></table></body></html>

# 4.4 Ablation Study

To simplify testing specific setups of MOMA, we conduct the following experiments primarily on the BBQ dataset. Styles of Balancing Experiment We experiment with different adjective styles to modify $X _ { s g }$ after the masking phrase, focusing on four styles: Neutral, Balancing, Unfair Positive, and Fair Positive, as shown in Figure 6. Neutral serves as the baseline, compensating for lost $X _ { s g }$ with minimal changes. Initially, we test Unfair Positive, which prompts the agent to generate positive adjectives. However, this worsens outcomes, likely due to the increased disparities between social groups $( X _ { s g 1 } - X _ { s g 2 } )$ . To counter this, we introduce Fair Positive, combining positive adjectives to mitigate bias, though it remains less effective than masking in Figure $7 \mathrm { a }$ , indicating the limitations of relying solely on positive phrases.

Finally, we develop Balancing, which uses a counterfactual positive adjective to equalize social groups’ disparities between $X _ { s g 1 }$ and $X _ { s g 2 }$ . Results in Figure $\mathrm { 7 a }$ show that balancing reduces bias in Neutral by an average of $5 0 . 2 \%$ , with only a $2 . 9 \%$ decrease in task performance.

Mask Symbols Experiment We experiment with many alphabetic or mathematical symbols and emojis as masking symbols. Figure 7b shows that these symbols have minimal impact on bias scores, with differences of less than 0.01. However, they affect task accuracy by about $5 \%$ . For details on symbol selection and specific results, see Appendix.

Summary MOMA variants indicate MOMA’s potential to further preserve task accuracy and reduce bias. The generation of different adjectives and the use of various symbols produce varying effects on both bias and accuracy.

# 4.5 Analysis

Case Study In Figure 6, the original prompt $X$ introduces an age-related bias in the generated output $Y$ , as the LLM mapping function $f _ { \theta }$ tends to associate discomfort with technology more strongly with the grandfather due to unobserved confounder $U$ . MOMA addresses this issue through a causal intervention by applying the transformation function $g _ { \theta }$ to generate a modified prompt $X ^ { \prime }$ , where the age-related variable $X _ { s g }$ is masked. This disrupts the implicit causal link and prevents $f _ { \theta }$ from reinforcing stereotypes. However, in cases where some contextual information must be retained or where bias mitigation standards are more flexible, balancing selectively reintroduces masked attributes while maintaining neutrality. Notably, masking also facilitates balancing; directly modifying without masking often leads to sub

Background   
Neutral @ Context Original   
Person X is the grandson 吊 Fair Positive The grandson and the grandfather   
Person Yisthe grandfather Person Xis the smartand patient grandson weretryingtobookacab onUber. Person Yis the smartand patient grandfather 8 Who was not comfortable using 8   
Balancing 414 the phone? Unfair Positive X   
PersonXisthepatientgrandson PersonYisthesmartgrandfather MaskingO 0 0 Person Xis the smart grandson Person Yis the patient grandfather Person Xand person Ywere trying...

![](images/584d6da5476dcc37400d803b3f16c40a1f842c2447d2e607b47010307eb94e95.jpg)  
Figure 6: Different styles of positive adjectives and their effects, with mask symbols as $X _ { - } Y$ .   
Figure 8: Results of different methods’ costs

Figure 7: Ablation experiments with MOMA on BBQ.

optimal adjustments influenced by the model’s inherent biases. By first stripping away bias-inducing elements, balancing can then systematically reintroduce key attributes in a more controlled and fair manner. Instead of outright removal, balancing adjusts by assigning positive traits such as “smart” and “patient” to both entities, ensuring a fairer representation while preserving grammatical and semantic integrity.

Cost Analysis Multi-agent systems often incur high costs (Smit et al. 2024). We analyze costs based on API calls and context expenses, divided into generation and overall fees (Figure 8). SoM, which relies on multiple agents and debate rounds for convergence, has the highest costs, even with the minimal setup—3 agents and 2 rounds—costing 12.9 times more than CoT.

MOMA’s hierarchical design reduces costs to 5.5 times that of CoT, with the main expense from a few-shot approach (5 shots) for assistant agents. This cost can be further re

Fee Ratios and APls Calls Ratio Compared to Baseline   
250   
225 IGenerationFeeRatio 208.4   
200 OvesralFe Rtatio 183.2   
175   
150   
125   
100 92.4 75 50 10101003104.0 18.3.0 22.57.0 42.5.0 0 Baseline CoT ABPs MOMA SC SoM

duced by training smaller models with demonstrations.

# 4.6 Limitations

Our study focuses on question-answering datasets to simplify the analysis of LLMs, though bias exists in other tasks as well (Gallegos et al. 2024a). While MOMA and its multiagent framework require relatively fewer API calls and computations, they still incur additional costs. The trade-off between these costs and performance gains warrants further research. Additionally, while balancing reduces bias while preserving more of the original context, masking remains the most effective debiasing method. Thus, quantifying the information loss caused by masking and how balancing mitigates it is essential. Given the complexity of such measurements, we leave this for future work to achieve finer-grained control over semantic nuances, a challenge that persists even in modern LLMs (Chatterjee et al. 2024).

# 5 Conclusion

MOMA offers a robust approach to bias mitigation in LLMs, balancing social bias reduction with model performance. By analyzing bias through a causal inference perspective, we introduced a multi-agent framework leveraging masking and balancing to mitigate biases associated with social group representation.

This work highlights the importance of precise, contextaware interventions in fostering fairness in AI systems and demonstrates the potential of causal interventions for debiasing. Future research could build on this methodology by exploring dynamic context adjustments to address diverse and evolving bias challenges, as well as refining multi-agent designs to further enhance AI fairness.