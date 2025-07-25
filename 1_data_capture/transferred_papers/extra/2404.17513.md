# A Comprehensive Evaluation on Event Reasoning of Large Language Models

Zhengwei $\mathbf { T a o ^ { 1 - 4 } }$ , Zhi $\mathbf { J i n } ^ { 1 , 2 \boxtimes } ;$ , Yifan Zhang1,2, Xiancai $\mathbf { C h e n ^ { 1 , 2 } }$ , Haiyan Zhao1,2, Jia $\mathbf { L i } ^ { 1 , 2 }$ Bin Liang3,4, Chongyang $\mathbf { T a o } ^ { 5 }$ , Qun Liu6, Kam-Fai Wong3,4

1School of Computer Science, Peking University, 2MoE Key Lab. of High Confidence Software Technologies(PKU), China   
3Department of Systems Engineering and Engineering Management, The Chinese University of Hong Kong 4MoE Key Lab. of High Confidence Software Technologies(Hong Kong), China 5 Beihang University, 6 Huawei Noah’s Ark Lab   
{tttzw, xiancaich, yifanzhang} $@$ stu.pku.edu.cn, $\{$ {zhijin, zhhy.sei, lijiaa}@pku.edu.cn, chongyang@buaa.edu.cn, bin.liang@cuhk.edu.hk, qun.liu@huawei.com, kfwong $@$ se.cuhk.edu.hk

# Abstract

Event reasoning is a fundamental ability that underlies many applications. It requires event schema knowledge to perform global reasoning and needs to deal with the diversity of the inter-event relations and the reasoning paradigms. The extent to which LLMs excel in event reasoning across various relations and reasoning paradigms has not been thoroughly investigated. Additionally, it is still unclear whether LLMs utilize event knowledge in the same way humans do. To mitigate this disparity, we comprehensively evaluate the abilities of event reasoning of LLMs on different relations, paradigms, and levels of abstraction. We introduce a novel benchmark $\mathrm { E V ^ { 2 } }$ for EValuation of EVent reasoning. $\mathrm { E V } ^ { 2 }$ consists of two levels of evaluation on schema and instance and is comprehensive in relations and reasoning paradigms. We conduct extensive experiments on $\mathrm { E V } ^ { 2 }$ . We find that 1) LLMs have abilities to accomplish event reasoning but their performances are far from satisfactory. 2) There are imbalances of event reasoning abilities on different relations and paradigms. 3) LLMs have event schema knowledge, however, they’re not aligned with humans on how to utilize the knowledge. Based on these findings, we guide the LLMs in utilizing the event schema knowledge as memory for improvements in event reasoning.

# Introduction

Events are instances or occurrences that form the basic semantic building units encompassing the meanings of Activities, Accomplishments, Achievements, and States (Vendler 1957). Event Reasoning is the ability to process and analyze events and their complex interconnections. Compared with other abilities, event reasoning is unique in some aspects. Firstly, it requires knowledge in the form of event schemas, capturing the progress of event evolution in scenarios, then performing global reasoning (Li et al. 2021a; Mao et al. 2021). As shown in Figure 1, each event instance is associated with an event type. All event types and their relations form the event schema knowledge which reflects the logic of event evolution. Knowing the event occurrence

Context Avoiding John underwent a life-saving treatment at Mercy General Hospital on April 5th, 2021, …   
Causes Enjoy Life John embraced a new lease on life, dedicating his time to traveling around the world… Before Celebrate John decided to spend his Rebirth Day with his friends from various countries. Causes Hold Party John invited guests to his home. His friends sang and danced together to their heart‘s content.   
CEC Which could be a subevent? CRR Which is the Begin Recover Travel Health causal relationship metiJcouhlnously engaJgoehdn in a between planned his tpheyrsaicpayl Make itinerary, … program, … Friends Intense Express People were moved by Exercise Emotion these stories Jcoehflrnei ebarnadtsehdis Johfrnietonldshis and spirits… by ruanceni…ng a daabnoguetrtohuis Causes

chain as ”Avoiding”, ”Enjoy life”, ”Celebrate”, and ”Hold Party” would result in the subevent ”Express Emotion”.

Second, the inter-event relations and reasoning paradigms are various. Event reasoning incorporates reasoning events according to a certain relation (Du et al. 2022; Sap et al. 2019b) and reasoning inter-event relations (Ning, Wu, and Roth 2018; Caselli and Vossen 2017). The queried relations are diversified such as causality (Roemmele, Bejan, and Gordon 2011), temporality (Zhou et al. 2019), and hierarchy (Glavasˇ et al. 2014). There are various paradigms such as reasoning the event or the inter-relation. As a fundamental competency within LLMs, event reasoning supports a multitude of Natural Language Processing (NLP) tasks, including recommendation engines (Yang et al. 2020), interactive question-answer systems (Souza Costa, Gottschalk, and Demidova 2020), and AI Agents (Liu et al. 2023). Therefore, the enhancement of event reasoning abilities is essential for the advancement of LLMs.

LLMs like LLAMA (Touvron et al. 2023) series and GPT series (Brown et al. 2020) have demonstrated exceptional accomplishments in various natural language reasoning (Bang et al. 2023; Xu et al. 2023). Existing research has evaluated a broad spectrum of reasoning abilities of LLMs such as commonsence (Bian et al. 2023), sentence relations (Chan et al. 2023), and math (Arora, Singh et al. 2023). However, studies on the comprehensive evaluation of event reasoning of LLMs are scarce. The incompleteness of current event reasoning evaluation is reflected in two aspects. First, current works only focus on instance-level events, resulting in unclearness of how LLMs understand and utilize the event schema knowledge (Chan et al. 2023). Investigating the event knowledge of LLMs and how they employ them underlines applications such as event-based memory systems. Besides, they ignore the diversity of relations and paradigms (Yuan, Xie, and Ananiadou 2023). Such findings could be biased since they neglect discrepancies brought by different aspects. These disparities hinge on the development of such crucial abilities of LLMs.

In this paper, we comprehensively evaluate event reasoning in knowledge and abilities. Since there are no existing datasets that are comprehensive in relations and paradigms, and can cover both levels of schema and instance, we introduce a benchmark $\mathrm { E V } ^ { 2 }$ for the EValuation of EVent reasoning. $\mathrm { E V ^ { 2 } }$ is featured in evaluating both aligned schemalevel and instance-level. The schema-level evaluation investigates the event schema knowledge of LLMs while the instance-level testifies the event reasoning abilities. Besides, $\mathrm { E V ^ { 2 } }$ evaluates event reasoning in various types of relation and reasoning paradigms. $\mathrm { E V ^ { 2 } }$ includes two event reasoning tasks, namely Contextual Event Classification (CEC) and Contextual Relation Reasoning (CRR) as shown in Figure 1, and three types of relations of causality, temporality, and hierarchy. Utilizing $\mathrm { E V ^ { 2 } }$ , we evaluate how well LLMs do event reasoning in abilities and knowledge. We mainly explore four research questions: 1) How proficient abilities of event reasoning do LLMs have? 2) To what extent do LLMs have the event schema knowledge? 3) Are LLMs aligned with humans in leveraging event schema knowledge? 4) Can LLMs perform better event reasoning with explicit guidance of leveraging event schema knowledge?

We conduct extensive experiments on $\mathrm { E V ^ { 2 } }$ to answer these questions. The results provide insights into event reasoning that: 1) LLMs have the abilities of event reasoning, but are far from satisfactory and are imbalanced in different relations and reasoning paradigms. 2) LLMs embed imbalanced abilities in different relations and reasoning paradigms. 3) LLMs have event schema knowledge. However, LLMs are not aligned with humans in the aspect of leveraging event schema knowledge. Based on the findings, we investigate guiding the LLMs to utilize event schema knowledge. With the guidance, LLMs can perform better event reasoning which sheds light on modeling event knowledge as memory of LLMs to enhance event reasoning.

We summarize our contributions as follows:

We first comprehensively evaluate event reasoning in both abstraction levels of schema and instance, and various relations and paradigms.   
We construct a benchmark $\mathrm { E V ^ { 2 } }$ which features two levels of evaluation and comprehensive in relations and reasoning paradigms.   
We conduct extensive experiments to probe how LLMs perform event reasoning.   
We conclude several insights. Based on our findings, we guide LLMs to utilize event schema knowledge as memory achieving improvements in event reasoning.

# Problem Formulation

Event reasoning is to anticipate the events by certain relations or deduce interrelated correlations (Tao et al. 2023a). Event reasoning requires comprehension of event schema knowledge. An event schema of a scenario is a schema-level graph $\mathcal { G } ^ { \bar { s } } = ( \mathbb { V } ^ { s } , \mathbb { E } ^ { s } ) ^ { 1 }$ , where $\mathbb { V } ^ { s }$ is the set of event types and $\mathbb { E } ^ { s }$ is the set of relations between events. Each edge in $\mathbb { E } ^ { s }$ is a relation triplet $( \mathcal { E } _ { t } ^ { s } , \mathcal { R } , \mathcal { E } _ { j } ^ { s } )$ standing for that there is the relation $\mathcal { R }$ between $\mathcal { E } _ { t } ^ { s }$ and $\mathcal { E } _ { j } ^ { s }$ . With instantiation, we have the instance-level event graph $\mathcal { G } ^ { i } = ( \mathbb { V } ^ { i } , \mathbb { E } ^ { i } ) ^ { 2 }$ . An instance event $\mathcal { E } ^ { i }$ has an event type ${ \mathcal { E } } ^ { s }$ but with detailed event arguments and description (Mitchell 2005). The nodes and edges of these two graphs are corresponding, namely, each triplet in $\mathcal G ^ { s }$ has a corresponding triplet in ${ \mathcal { G } } ^ { i }$ with the same inter-relation. In both levels, we consider totally six relation types, namely $\mathcal { R } \in \{ \mathsf { C a u s e s }$ , IsResult, Before, After, IsSubevent, HasSubevent $\} ^ { 3 }$ . Causes and IsResult are Causal relations, Before and After belong to Temporal type while IsSubevent and HasSubevent are Hierarchical type. We consider two event reasoning paradigms for both the schema and instance levels: Contextual Event Classification (CEC) and Contextualized Relation Reasoning (CRR).

CEC Given graph $\mathcal { G }$ , either schema- or instance-level, queried event ${ \mathcal { E } } \in { \mathcal { G } }$ , and target relation $\mathcal { R }$ , CEC requires the model to answer an event $\mathcal { E } _ { a }$

$$
\mathcal { E } _ { a } = \mathrm { M } ( \mathcal { E } , \mathcal { R } , \mathcal { G } , \mathbb { C } ) .
$$

M is the model, $\mathbb { C }$ is the candidate event set. CEC evaluates the model’s comprehension of event semantics and structure.

CRR Given graph $\mathcal { G }$ , either schema- or instance-level, two queried events $\mathcal { E } _ { t } , \mathcal { E } _ { j } \in \mathcal { G }$ , CRR requires to determine the relation $\mathcal { R }$ between them:

$$
\mathcal { R } = \mathrm { M } ( \mathcal { E } _ { t } , \mathcal { E } _ { j } , \mathcal { G } ) .
$$

CRR evaluates the understanding of event relations.

Both schema and instance levels have CEC and CRR tasks. Schema-level tasks require models to be rich in knowledge while instance need models to process details.

# Benchmark Construction

Constructing the $\mathrm { E V ^ { 2 } }$ benchmark is challenging since events and their relations are semantically abstract concepts compared with entity concepts. The occurrence of events and their relations not only follow objective natural laws but are also influenced by social and humanistic factors. Therefore, annotating such data is extremely label-intensive leading to a lack of evaluation dataset of the task. Previous works mainly construct such datasets by extracting events and relations from some unlabeled corpus such as news reports (Caselli and Vossen 2017; Ning, Wu, and Roth 2018; O’Gorman, Wright-Bettner, and Palmer 2016). However, such a method suffers from limited event relational patterns and domain-specific language expression. To mitigate such problems, in this work, we construct our comprehensive event reasoning evaluation from scratch.

In $\mathrm { E V ^ { 2 } }$ , we evaluate event reasoning in various relation and reasoning paradigms of both schema and instance levels. In our pilot trial, we directly annotate each question and answer by human annotators. We find it extremely timeconsuming since human annotators must imagine scenarios and guarantee their correctness. Besides, the diversity of the events and their relations are poor. Annotators only write the most common events. Therefore, we propose a five-stage construction process. Generally, we first synthesize both the schema and corresponding instance event prompting-graphs automatically as prompts for later annotations. Due to the large size of the synthesis graphs, before formal annotating, we then recruit annotators to remove incorrect graphs which are hard to modify. After that, the annotators curate schema and instance graphs based on the generated prompts. Then, we adapt the graphs into questions and answers. Finally, we recruit human validators to examine the quality of the data. We celebrate this process in the following sections.

# Prompting-Graph Synthesizing

To cover more scenarios, we synthesize event graphs as prompts for annotations instead of annotating from scratch. Specifically, we first establish the schema graph $\mathcal G ^ { s }$ . Then we employ GPT4 to generate the instance graph $\mathcal G ^ { i }$ .

Schema Graph The schema graph represents event occurrence knowledge. Harvesting such knowledge is a research point (Du 2022). We here leverage EECKG (Wang et al. 2022b) to ensure a diverse range of event types in our schema. EECKG combines rule-based reasoning with crowdsourced insights, built on ConceptNet’s structure. Nodes in EECKG represent verb phrases as events, and edges denote inter-event relations, focusing on Causes, Before, and HasSubevent.

Our objective mandates that the nodes within $\mathcal G ^ { s }$ should represent event types. Therefore, we filter EECKG nodes, removing concrete event instances. We keep nodes with at most two words, as longer descriptions tend to include specific details. For events with fewer than two words, we use GPT4 to enhance our selection, ensuring the appropriate abstraction level for our schema graph4.

We identify a subset of remaining events that are too generic. To refine the event selection, we also exclude the most frequent events from our subset to avoid generic events. We then dissect the interconnected EECKG into separate components, each representing a distinct scenario. To prevent semantic drift, we carefully control the size of each component. Starting from a node, we conduct a random walk until the number of nodes surpasses a threshold, thus defining a component. This process is executed for all nodes to gather components, as Algorithm 1 in the Appendix. Then we eliminate cycles to convert these structures into DAGs.

EECKG only contains forward event evolution relations such as Causes. We further include components of backward relations. We generate a reversed version for each component by inverting edge directions and replacing relations with their opposites: IsResult, After, and IsSubevent. This creates the backward components.

In preparation for constructing tasks for CEC and CRR, we label two events for each component. We then sample three event pairs $( \mathcal { E } _ { h } , \mathcal { E } _ { t } )$ per component with a maximum inter-path length of four with their predecessors as background events. These pairs and background events form a schema graph. If the path length between $\mathcal { E } _ { h }$ and $\mathcal { E } _ { t }$ is two, their relation serves as the queried relation; for longer paths, we deduce the relation using predefined rules as shown in Appendix Table 6. We construct a schema graph, queried event pair, and their relation $( \mathcal { E } _ { h } , \mathcal { E } _ { t } , \mathcal { R } , \mathcal { G } ^ { s } )$ .

Instance Graph We next harvest instance graph $\mathcal { G } ^ { i }$ for each schema graph $\mathcal G ^ { s }$ . For each node $\mathcal { E } ^ { s } \in \mathcal { G } ^ { s }$ , we ask GPT4 to generate $\mathcal { E } ^ { i }$ .

We inherit the relations of $\mathcal G ^ { s }$ to obtain $\mathcal { G } ^ { i }$ . We naturally obtain the instances of $\mathcal { E } _ { h }$ and $\mathcal { E } _ { t }$ . We obtain 1,600 schema prompting graphs and 1,600 corresponding instances.

# Manual Filtering

After curating the prompting graphs of both levels, the next is to annotate based on the prompting graphs. However, we find some of the prompting graphs are incorrect and hard to modify. Therefore, before formal annotation, we launch another manual filtering step to remove such graphs.

Table 1: Number of $\mathrm { E V ^ { 2 } }$ . S and I are schema and instance.   

<html><body><table><tr><td>S-CEC</td><td>I-CEC</td><td>S-CRR</td><td>I-CRR</td><td>GRAPHPAIRS</td></tr><tr><td>492</td><td>491</td><td>730</td><td>735</td><td>491</td></tr></table></body></html>

We then recruit 8 well-educated human annotators where they each process 200 data. Their missions are to investigate the $\mathcal G ^ { s }$ and $\mathcal G ^ { i }$ by the following steps:

1) Check whether $\mathcal G ^ { s }$ can be modified, the events are abstract, the relations are correct, and $\mathcal G ^ { s }$ tells an entire story of a scenario.   
2) Check whether $\mathcal G ^ { i }$ can be modified., the events are concrete, the relations are correct, and $\mathcal G ^ { i }$ tells an entire story of a scenario.   
3) Check whether $\mathcal G ^ { s }$ and $\mathcal { G } ^ { i }$ are consistent, there are obvious schema-instance relations between them.

If any of these conditions are not met, we discard this datum. After this filtering process, we remain 491 graph pairs.

# Annotation

In this stage, we formally annotate based on filtered prompting-graphs. The missions are to 1) rewrite the events and relations in both schema and instance graphs to make them strictly valid. 2) identify a query event $\mathcal { E } _ { h } ^ { s }$ and an answer event $\mathcal { E } _ { t } ^ { s }$ in the graph for later question adaptation. 3) write candidates as negative choices considering the answer event where each candidate event consists of a schema and an instance event. For the second mission, regarding schema and instance head events as the query and the tail as an answer, we ask GPT4 to generate 15 possible candidate instance events with their schema events for prompting. We recruit another 10 annotators. The annotation is completed on our annotating system. We describe the detailed annotation process in the Appendix. Each annotator should rewrite correct data alongside the following standards:

1) Rewrite $\mathcal G ^ { s }$ and ${ \mathcal { G } } ^ { i }$ making them correct as highprobability knowledge, and do not consider lowprobability situations.   
2) Rewrite $\mathcal G ^ { s }$ and $\mathcal G ^ { i }$ leading to the coherence of the whole graph, and there’s no semantic drift.   
3) Rewrite $\mathcal G ^ { s }$ and ${ \mathcal { G } } ^ { i }$ making them consistent. Schema events and instance events require a clear distinction in hyper-hypo relation.   
4) Rewrite the target event making it only can be answered when considering the whole graph.   
5) Rewrite the instance events making them should be expressed independently without connective expressions such as ”After $\mathcal { E } _ { 1 } ^ { \mathrm { ~ } \dagger \mathrm { ~ } }$ to avoid information leakage.

# Question Adaptation

The last is to construct questions of CEC and CRR in both schema and instance levels based on the annotated graphs. The schema part of annotation is for the schema-level questions and the instance part is for instance-level.

For schema-level CEC, we naturally use the queried event $\mathcal { E } _ { h } ^ { s }$ and other nodes except for the answer event $\mathcal { E } _ { t } ^ { s }$ as context to form a question. Then we use the answer event $\mathcal { E } _ { t } ^ { s }$ as the answer. We do similarly at the instance level. For CRR, we regard $\mathcal { E } _ { h } ^ { s }$ and $\mathcal { E } _ { t } ^ { s }$ as queried events and use the relation between them as the answer to form the schema-level question. For instance part, we adopt a similar way.

Table 2: Comparison with existing event reasoning datasets. L stands for the included levels. C represents whether it’s contextualized. M-R and $\mathbf { M } \mathbf { - } \mathbf { P }$ means multi-relations and paradigms. $S$ and $I$ stand for schema and instance level.   

<html><body><table><tr><td>DATASET</td><td>L</td><td>C</td><td>M-R</td><td>M-P</td></tr><tr><td>ALTLEX(Hidey 2016)</td><td>I</td><td>×</td><td>X</td><td></td></tr><tr><td>ASER(Zhang et al.2020)</td><td>S</td><td>X</td><td></td><td>×</td></tr><tr><td>ATOMIC(Sap et al. 2019a)</td><td>S</td><td>X</td><td></td><td>×</td></tr><tr><td>COPA</td><td>I</td><td>X</td><td>X</td><td></td></tr><tr><td>CQA(Bondarenko 2022)</td><td>I</td><td></td><td>√</td><td>×</td></tr><tr><td>ECARE(Du et al. 2022)</td><td>I</td><td>X</td><td>X</td><td>×</td></tr><tr><td>ESL(Caselli and Vossen 2017)</td><td>I</td><td></td><td>X</td><td>X</td></tr><tr><td>ESTER(Han et al. 2021)</td><td>I</td><td></td><td></td><td>×</td></tr><tr><td>HIEVE(Glavas et al. 2014)</td><td>I</td><td></td><td>X</td><td>X</td></tr><tr><td>KAIROS(Li et al. 2021a)</td><td>S</td><td></td><td>X</td><td>X</td></tr><tr><td>LDC2020E25(Li et al. 2021a)</td><td>S</td><td></td><td></td><td>X</td></tr><tr><td>MATRES(Ning,Wu,and Roth 2018)</td><td>I</td><td></td><td>X</td><td></td></tr><tr><td>MAVEN-ERE(Wang et al. 2022a)</td><td>I</td><td></td><td>√</td><td>×</td></tr><tr><td>MCNC(Granroth-Wilding 2016)</td><td>I</td><td></td><td>X</td><td>X</td></tr><tr><td>MCTACO(Zhou et al. 2019)</td><td>I</td><td></td><td>X</td><td>X</td></tr><tr><td>RED</td><td>I</td><td></td><td></td><td>×</td></tr><tr><td>SCITE(Li et al. 2021b)</td><td>I</td><td></td><td>X</td><td>X</td></tr><tr><td>SCT(Mostafazadeh etal.2016)</td><td>I</td><td></td><td>X</td><td>×</td></tr><tr><td>SociallQA(Sap et al.2019b)</td><td>I</td><td></td><td>√</td><td>X</td></tr><tr><td>TB-Dense(Cassidy et al. 2014)</td><td>I</td><td></td><td>X</td><td>×</td></tr><tr><td>TRACIE(Zhou 2020)</td><td>I</td><td></td><td>×</td><td>×</td></tr><tr><td>EV2</td><td>SI</td><td></td><td></td><td></td></tr></table></body></html>

Finally, our CEC task is a 4-way multiple-choice task. The CRR is a 3-way multiple-choice task. In CRR, the choices for temporal, causal, and hierarchy relations are [Before, After, Vague], [Causes, IsResult, None], and [IsSubevent, HasSubevent, None] respectively. We show examples of both tasks in Figure 1.

# Quality Validation

After that, we recruit another three human validators to guarantee the quality of both tasks in $\mathrm { E V ^ { 2 } }$ . They delete the nonqualified data by the following criteria:

1) Delete the data if the logic of the graph is incorrect. 2) Delete the data if any of the negative candidates can also be the correct answer. 3) Delete the data if it can be answered without the context.

After the quality validation, we have our final $\mathrm { E V ^ { 2 } }$ benchmark. We report the number of each task and the average nodes and edges in Table 1.

# Existing Dataset Comparison

We compare our benchmark to existing related datasets. We show detailed comparison in Table 2. Our benchmark is the only one that is for contextualized event reasoning of various relations and paradigms on both schema and instance levels.

Table 3: Average performances with updated models. S and I stand for schema- and instance-level.   

<html><body><table><tr><td>Model</td><td>S-CEC</td><td>I-CEC</td><td>S-CRR</td><td>I-CRR</td></tr><tr><td>GPT40</td><td>68.93</td><td>66.60</td><td>62.05</td><td>62.04</td></tr><tr><td>GPT4</td><td>68.11</td><td>68.43</td><td>63.01</td><td>63.81</td></tr><tr><td>GPT3.5</td><td>65.43</td><td>64.77</td><td>54.52</td><td>43.95</td></tr><tr><td>Mistral-7B</td><td>52.47</td><td>54.18</td><td>51.64</td><td>55.24</td></tr><tr><td>Qwen2-7B</td><td>62.14</td><td>63.75</td><td>52.05</td><td>47.62</td></tr><tr><td>Baichuan2-7B</td><td>52.88</td><td>29.94</td><td>51.64</td><td>45.31</td></tr><tr><td>Orca2-7B</td><td>59.88</td><td>60.08</td><td>46.16</td><td>45.17</td></tr><tr><td>Chatglm2-6B</td><td>55.76</td><td>30.96</td><td>52.47</td><td>49.66</td></tr><tr><td>Interlm2-7B</td><td>65.84</td><td>62.12</td><td>48.63</td><td>57.28</td></tr><tr><td>Llama2-7B</td><td>45.06</td><td>29.74</td><td>46.58</td><td>41.22</td></tr><tr><td>Vicuna-7b</td><td>25.93</td><td>27.09</td><td>52.05</td><td>51.43</td></tr></table></body></html>

# Experiments Results and Findings

Evaluated LLMs We evaluate 11 LLMs on event reasoning. For the open-source models, we evaluate their chat-version. We evaluate GPT4o, GPT4, GPT3.5. For the closed-source models, we utilize their official APIs to conduct performance evaluations. For the open-source models, we include Mistral-7B (Jiang 2023), Qwen2-7B (Yang et al. 2024), Baichuan-2-7B (Yang et al. 2023), Orca2-7B (Mitra et al. 2023), Chatglm2-6B (GLM 2024), Internlm2-7B (Cai et al. 2024), Llama2-7B (Touvron et al. 2023), and Vicuna7B (Chiang et al. 2023). Without loss of generosity, we use the model names to refer to the chat versions. We list the model and prompt details in the Appendix.

# How proficient abilities of event reasoning do LLMs have?

In this part, we mainly probe the abilities of how existing LLMs complete the event reasoning of the instance level.

LLMs have the abilities of event reasoning, but even the strongest GPT-4 is far from satisfactory. We evaluate CEC and CRR at the instance level. We show the results of different relations in Figure 2 and detailed results in Tables 7 and 9 in the Appendix. For CEC, GPT4 performs the best. Among all open-source LLMs, Qwen2 is the best while Internlm2 holds the second. Early models such as Baichuan2- 7B, Chatglm2-6B, Llama2-7B, and Vicuna-7B fall in this task where they score under $40 \%$ . For CRR, GPT4 excels all other models as well. Among all open-source LLMs, Internlm2-7B and Mistral-7B performs in the first tie.

We show the average performance of instance-level CEC and CRR in columns I-CEC and I-CRR in Table 3. Overall, existing LLMs such as GPT4, and Qwen2-7B have CEC event reasoning abilities. However, even the strongest GPT4 can only achieve 68.43 (4-way multiple choice) and 63.81 (3-way multiple choice) accuracy in each task showing there’s much room for improvements of event reasoning.

The abilities of LLMs to deal with different relations and reasoning paradigms are imbalanced. Comparing CEC to CRR, as relation-wise results shown in Figure 2 and average performances in columns I-CEC and I-CRR in Table 3,

Table 4: Event schema knowledge Alignment. ET is the event type accuracy. REL is relation triplet F1-score.   

<html><body><table><tr><td rowspan="3"></td><td colspan="2">CEC</td><td colspan="2">CRR</td></tr><tr><td>ET</td><td>REL</td><td>ET</td><td>REL</td></tr><tr><td>GPT40</td><td>65.53</td><td>40.23</td><td>69.84</td><td>51.24</td></tr><tr><td>GPT4</td><td>72.78</td><td>40.57</td><td>73.19</td><td>52.79</td></tr><tr><td>GPT3.5</td><td>10.05</td><td>15.58</td><td>16.19</td><td>28.63</td></tr><tr><td>Mistral-7B</td><td>22.93</td><td>16.28</td><td>25.00</td><td>23.87</td></tr><tr><td>Qwen2-7B</td><td>11.40</td><td>15.97</td><td>13.07</td><td>16.58</td></tr></table></body></html>

LLMs perform better for CEC than CRR (note that CEC is a 4-way multiple choice task while CRR is of 3-way). To compare, we compute the average scores of I-CEC and I-CRR on models achieving above $40 \%$ and $3 0 \% ^ { 5 }$ . We find I-CEC is much higher than I-CRR, with average scores 62.84 and 53.58. The results significantly suggest that CRR is harder and insifeciently solved than CEC. Existing pretraining and SFT datasets may be biased in paradigms.

We then analyze performances on different relations. We compute the average scores of relations on models achieving above $40 \%$ and $30 \%$ on average I-CEC and I-CRR. The I-CEC average scores of Temporal, Causal, Hierarchical are 50.11, 68.71, and 61.22 while in I-CRR the scores are 43.31, 58.36, and 52.17. With these results alongside scores shown in Figure 2, LLMs perform best in Causal relation. Then, Temporal relation is the worst. It indicates current training can enable LLMs to reason causality. It also trains the event hierarchy comprehension. However, temporal reasoning is the hardest. More methods should be established to handle this problem. That further indicates the imbalance training of different relations. Methods and datasets of balanced abilities on relations are needed. Transferring abilities of different relations could also be feasible (Tao et al. 2023b).

This is a crucial finding. Chan et al. (2023) conduct causal event classification such as ECARE (Du et al. 2022), and relation reasoning such as MATRES (Ning, Wu, and Roth 2018). They directly compare these two groups of results and conclude the gaps are merely from differences in relations. However, they ignore the difference in reasoning paradigms. By $\mathrm { E V ^ { 2 } }$ , with disentangling relations and formulations, we investigate event reasoning with less bias.

CEC improves faster than CRR with model development. We investigate the improvement trends of CEC and CRR. The Figure 3 in the Appendix, when models have poor event reasoning abilities, their CEC performances lie around the balanced line showing no significant differences in tasks. With the development, the CEC improves much faster than CRR for all models. This investigation further appeals to the need for training in balanced event reasoning abilities.

# To what extent do LLMs have the event schema knowledge?

In the previous section, we acknowledge that LLMs can complete event reasoning to some extent. However, whether they are endowed with event schema knowledge remains unknown. In this part, we mainly explore to what extent LLMs have the event schema knowledge, i.e. of the schema level.

![](images/dfa128d401493c5df3fc21113574f142d8737ddc550796b90a37b8f5ad7efd31.jpg)  
Figure 2: Results of CEC and CRR. S and I stand for schema- and instance-level. Relation types of Causality, Temporality, and Hierarchy are denoted as C, T, and H.

<html><body><table><tr><td rowspan="2">Model</td><td colspan="6">CEC</td><td colspan="6">CRR</td></tr><tr><td>|Temporal</td><td>Causal</td><td>Hierarchical</td><td>W.T.S</td><td>w.0.S</td><td>△</td><td>Temporal</td><td>Causal</td><td>Hierarchical</td><td>W.T.S</td><td>w.0.S</td><td>△</td></tr><tr><td>GPT40</td><td>58.06</td><td>73.45</td><td>85.71</td><td>71.49</td><td>66.60</td><td>4.89↑</td><td>56.45</td><td>73.04</td><td>75.65</td><td>69.25</td><td>62.04</td><td>7.21个</td></tr><tr><td>GPT4</td><td>61.29</td><td>75.86</td><td>77.92</td><td>72.51</td><td>68.43</td><td>4.08↑</td><td>53.23</td><td>70.74</td><td>68.70</td><td>65.99</td><td>63.81</td><td>2.18个</td></tr><tr><td>GPT3.5</td><td>52.42</td><td>77.59</td><td>77.92</td><td>71.28</td><td>64.77</td><td>6.51个</td><td>44.09</td><td>59.22</td><td>47.83</td><td>53.61</td><td>43.95</td><td>9.66个</td></tr><tr><td>Mistral-7B</td><td>45.16</td><td>70.34</td><td>74.03</td><td>64.56</td><td>54.18</td><td>10.38个</td><td>48.92</td><td>63.36</td><td>50.43</td><td>57.69</td><td>55.24</td><td>2.45个</td></tr><tr><td>Qwen2-7B</td><td>61.29</td><td>78.97</td><td>77.92</td><td>74.34</td><td>63.75</td><td>10.59↑</td><td>47.31</td><td>66.13</td><td>55.65</td><td>59.73</td><td>47.62</td><td>12.11个</td></tr><tr><td>Baichuan2-7B</td><td>41.13</td><td>56.90</td><td>57.14</td><td>52.95</td><td>29.94</td><td>23.01个</td><td>46.24</td><td>58.29</td><td>40.87</td><td>52.52</td><td>45.31</td><td>7.21个</td></tr><tr><td>Orca2-7B</td><td>53.23</td><td>77.59</td><td>76.62</td><td>71.28</td><td>60.08</td><td>11.20个</td><td>44.09</td><td>65.90</td><td>51.30</td><td>58.10</td><td>45.17</td><td>12.93↑</td></tr><tr><td>Chatglm2-6B</td><td>38.71</td><td>65.86</td><td>54.55</td><td>57.23</td><td>30.96</td><td>26.27个</td><td>47.31</td><td>53.00</td><td>50.43</td><td>51.16</td><td>49.66</td><td>1.50个</td></tr><tr><td>Interlm2-7B</td><td>57.26</td><td>80.69</td><td>81.82</td><td>74.95</td><td>62.12</td><td>12.83↑</td><td>53.23</td><td>70.28</td><td>72.17</td><td>66.26</td><td>57.28</td><td>8.98↑</td></tr><tr><td>Llama2-7B</td><td>36.29</td><td>45.86</td><td>49.35</td><td>43.99</td><td>29.74</td><td>14.25个</td><td>42.47</td><td>51.61</td><td>53.91</td><td>49.66</td><td>41.22</td><td>8.44↑</td></tr><tr><td>Vicuna-7b</td><td>27.42</td><td>28.62</td><td>27.27</td><td>28.11</td><td>27.09</td><td>1.02个</td><td>45.16</td><td>54.15</td><td>52.17</td><td>51.56</td><td>51.43</td><td>0.13↑</td></tr></table></body></html>

Table 5: Guidance with schema knowledge on instance level CEC and CRR tasks. W.T.S and W.O.S stands for average performances with and without event knowledge guidance. $\Delta$ is the difference between them.

LLMs have event schema knowledge. We evaluate CEC and CRR on schema level. The results are shown in Figure 2 and detailed results in Tables 8 and 10 in the Appendix, and the average scores are reported in Table 3. We find LLMs already have event schema knowledge and can complete both CEC and CRR tasks at the schema level to some extent. However, in Table 3, we observe that S-CEC lags I-CEC, suggesting advanced reasoning at the instance level.

Event schema knowledge increases falling behind reasoning at the instance level. We probe how event schema knowledge increases with the development of LLMs. We depict CEC and CRR performance comparisons of LLMs on instance- and schema-level in Figure 4 in the Appendix.

When the models initially can reason about events, they also have event schema knowledge. At this time, models can perform comparatively or even better in schema-level event reasoning. With the development, models perform instancelevel reasoning on par with schema-level. It indicates that the accumulation of event schema knowledge falls behind the reasoning at the instance level. This finding demonstrates that enhancing event schema knowledge may further improve these abilities to obtain better general LLMs.

# Are LLMs aligned with humans in the aspect of leveraging event schema knowledge?

In this section, we investigate how LLMs leverage event schema knowledge to complete event reasoning. We first provide the instance-level question for the models and then ask them to generate the required event schema knowledge to solve the task. Then we evaluate the accuracy of the generated event schema knowledge.

Since we have the ground truth event schema knowledge for each question, the only challenge is to guide the LLMs to generate in a similar format for calculating accuracy. The instruction of our prompt first asks LLMs to generate the event types of each instance event in data. Based on the event types, it requires the LLMs to further generate relation triplets needed for the question.

However, we find the LLMs would generate event types of different words but correct contents. To mitigate this problem, we prepare a list of candidate event types for each data to make it a classification setting. The models need to select the correct event types for all event instances in the question and determine relations between each type. To keep the task difficult, we first conduct KMeans clustering on all event types in our dataset6. We obtain 1000 clusters. For each data, we assign 20 random candidates in total including the correct ones. The negative event types are chosen from different clusters. Models need to find out the correct event types from given options. We calculate the accuracy of event types and F1-scores of relation triplets comparing with the humanlabeled event schema. We regard a correct triplet if all the head and tail event types and the inter-relation align with the human labels. We show detailed examples in the Appendix.

The results are in Table 4. We find only GPT4 and GPT4o can generate correct event types. GPT4 excels may be attributed to 1) its better alignment. 2) The dataset is prompted by GPT4. Therefore in this part, we mainly focus on other models rather than GPT4 series. However, we find rest models all fail to generate corresponding schema knowledge. For relation triplet generation, even GPT4 can not output proper event schemas. It significantly suggests that LLMs may not leverage event schema knowledge as humans when solving event reasoning tasks. Alignment of using such knowledge could further improve the performances.

# Can LLMs perform better on event reasoning with explicit guidance of leveraging event schema knowledge?

In the previous section, we find LLMs may not leverage event schema knowledge as human does. It raises an interesting question how well LLMs perform if we guide them to explicitly use such knowledge? Here we probe this question.

We conduct experiments in which we directly add the schema event of each instance event in the question into the prompt. We also add relations of these schema events.

We demonstrate the performances of this guidance in Table 5. Incorporating event schema knowledge significantly improves event reasoning. It shows great potential to solve event reasoning with the fusion of event schema knowledge. These results provide important insights that event schema knowledge could be used as the memory of LLMs to improve solving practical and domain-specific problems. Constructing and retrieving proper event schema knowledge is another challenge. We leave them to future works.

# Related Work

Event Reasoning Du et al. (2022) aims to select the accurate cause or effect event from candidates. Zhou et al. (2019) serves as a dataset for event temporal reasoning. Current works present a scenario of incorporating counterfactual reasoning (Qin et al. 2019, 2020). In addition to single-event relation reasoning, existing works also reason events according to diversified event relations (Poria et al. 2021; Han et al. 2021; Yang et al. 2022). Tao et al. (2023b) further unifies datasets of several event-inter relations to transfer event relational knowledge to unseen tasks.

Predicting events necessitates the model to anticipate forthcoming occurrences grounded in the present context (Zhao 2021). Mostafazadeh et al. (2016) employs a multiple-choice framework to predict future events by encompassing a diverse range of common-sense connections among events. Guan, Wang, and Huang (2019) establish a dataset oriented towards capturing event logic, enabling the generative prediction of future incidents.

Evaluations for LLMs Evaluating the capacities of LLMs is the foundation of using and improving them. One group of research evaluates the general abilities of LLMs (Hendrycks et al. 2020; Zheng 2023; Zhong 2023; Bang et al. 2023) Besides, existing works evaluate LLMs in specific tasks (Bang et al. 2023; Bian et al. 2023; Gao et al. 2023; Wei 2023; Li et al. 2024). Constructing benchmarks in the LLMhuman interactive way is a regular strategy for complicated tasks (Huang et al. 2024).

Related to event reasoning, Yuan, Xie, and Ananiadou (2023) evaluated the event relation extraction. Tao et al. (2023a) present the Event Semantic Processing including the event understanding, reasoning, and prediction. Chan et al. (2023) investigates relation reasoning between sentences. Compared with them, we are the first to introduce the evaluation for both schema- and instance-level event reasoning. Moreover, we comprehensively evaluate the performances of various relations and reasoning paradigms.

# Conclusion

In this paper, we evaluate the event reasoning of LLMs. We introduce a novel benchmark $\mathrm { E V } ^ { 2 }$ which features both levels of schema and instance. It evaluates event schema knowledge and reasoning abilities. Besides, $\mathrm { E V ^ { 2 } }$ can be used to comprehensively evaluate the event reasoning in various relations and reasoning paradigms. We conduct extensive experiments on $\mathrm { E V } ^ { 2 }$ . We obtain many insights such as: 1) LLMs have the abilities of event reasoning, but are far from satisfactory and are unbalanced in different relations and reasoning paradigms. 2) LLMs have a comprehension of event schema knowledge. 3) LLMs are not aligned with human to leaverage event schema knowledge in event reasoning. Based on the findings, we guide the LLMs to utilize event schema knowledge. With our guidance, LLMs can perform better on event reasoning.