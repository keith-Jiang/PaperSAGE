# Automatically Generating Numerous Context-Driven SFT Data for LLMs Across Diverse Granularity

Shanghaoran Quan

School of Computer Science and Engineering, Beihang University Beijing, Haidian, 100191, China shrquan@buaa.edu.cn

# Abstract

Constructing high-quality query-response pairs from custom corpora is crucial for supervised fine-tuning (SFT) large language models (LLMs) in many applications, like creating vertical-domain AI assistants or roleplaying agents. However, sourcing this data through human annotation is costly, and existing automated methods often fail to capture the diverse range of contextual granularity and tend to produce homogeneous data. To tackle these issues, we introduce a novel method named AUGCON, capable of automatically generating context-driven SFT data across multiple levels of granularity with high diversity, quality and fidelity. AUGCON begins by generating queries using the Context-SplitTree (CST), an innovative approach for recursively deriving queries and splitting context to cover full granularity. Then, we train a scorer through contrastive learning to collaborate with CST to rank and refine queries. Finally, a synergistic integration of self-alignment and self-improving is introduced to obtain high-fidelity responses.

Extensive experiments are conducted incorporating both automatic and human evaluations, encompassing four widelyused benchmarks and a test scenario in English and Chinese. The results highlight the significant advantages of AUGCON in producing high diversity, quality, and fidelity SFT data against several state-of-the-art methods.

# Code — https://github.com/quanshr/AugCon

# 1 Introduction

With the rise of impressive capabilities of large language models (LLMs), a variety of vertical-domain LLM-based AI assistants have been introduced (Cheng, Huang, and Wei 2023; Chen et al. 2023; Luo et al. 2023). By incorporating specialized knowledge into LLMs, these custom models have been shown to outperform their general-purpose counterparts in their respective areas. These models can be developed through two strategies: building them from scratch (Yang et al. 2023; Liu et al. 2023d) or adapting existing general LLMs through supervised fine-tuning (SFT) (Shi et al. 2023; Zaiem et al. 2023), with the latter approach often favored for its efficiency and the foundational advantages offered by the general LLMs (Jiang et al. 2024; Dong et al. 2023; Cheng, Huang, and Wei 2023).

Directly supervised fine-tuning on the raw, custom corpora, also known as domain-adaptive pre-training (DAPT) (Gururangan et al. 2020), has proven beneficial (Bayer et al. 2024; Krieger et al. 2022) but revealed to be insufficient and may impair prompting ability on domainspecific tasks (Liu et al. 2023e; Pal et al. 2024). To better leverage the privatized knowledge and customize the outputs of LLMs, supervised fine-tuning using custom queryresponse pairs has become common practice (Shaikh et al. 2024; Chang, Peng, and Chen 2023). However, sourcing these pairs through human annotation is very costly and can’t generate at scale. Recent studies have explored automated methods for creating these pairs from custom corpora. AdaptLLM (Cheng, Huang, and Wei 2023), for instance, has used regex-based patterns to generate query-response pairs, but this approach tends to produce a limited variety of SFT data, which may not significantly enhance prompting capabilities and risks overfitting due to the narrow range of query types predefined. ETRC (Jiang et al. 2024) and ContextInstruct (Wang et al. 2023b) improved this by employing delicately designed prompts to generate queries from context using an LLM. However, those existing methods using the same workflow repeatedly on the same context tend to produce redundant queries without adequately covering the entire context at various levels of granularity. To automatically construct synthetic custom SFT data incorporating a wide range of contextual granularity (queries range from detailed questions to macro topics) with high diversity (queries need to be diversified to cover as much as possible the provided corpus), quality (responses are correct and efficient in answering the queries), and fidelity (data needs to follow human values and conform to predetermined tone and formats) still remain challenges.

To address these challenges, we propose AUGCON, which automatically generates multi-granularity contextdriven SFT data for LLMs at scale with high diversity, quality, and fidelity. AUGCON performs the following three essential steps:

1. Recursively Deriving Queries via Context-Split-Tree: Considering that it is challenging for predetermined prompts to generate non-repetitive queries with broad granularity from the same context, we propose a novel method called Context-Split-Tree (CST). Starting from a context (which is a continuous text chunk extracted from the corpus), we use an LLM to derive a query from it. At the same time, we ask the LLM to split this context into two contexts that are as independent as possible. Each context will recursively continue to derive queries and splits until it cannot be further divided. At the end, we will obtain a binary tree rooted in the initial context, and each node represents a context and contains a query that matches the granularity of it.

2. Contrastive-Learning-Based Scorer to Rank Queries and Filtering: To further ensure the quality and diversity of the queries, we use contrastive learning to train a scorer to evaluate the query by taking the obtained queries as positive examples and manipulating the prompt in Step 1 (e.g., by using suboptimal instruction or attaching fewer few-shot examples) to generate negative examples. Then, we sort the derived queries under the same context using the scorer and only retain the queries that get high scores and the diversity evaluated by ROUGE-L reaching a specific threshold. To ensure high quality and high diversity of queries while reaching the certain quantity requirements, the filtering stage will be iterated with CST until the requirements are met.

3. Obtaining High-Fidelity Responses: Inspired by the significant impact principles (Sun et al. 2024, 2023) have on LLMs, we employ a principle-driven self-alignment approach to guide the LLM in producing high-fidelity responses to filtered queries and their respective contexts. To enhance the quality of the generated answers further, we apply random search and conduct the LLM to selfevaluate its responses and discover the best in-context learning (ICL) examples from those annotated by humans. Ultimately, all context, ICL examples, and principles are discarded, leaving only the query-response pairs to supervised fine-tune the LLM.

The entire process only requires a handful of few-shot CST examples, alignment principles, and query response examples. We can also achieve impressive results by just utilizing the open-source model, which will later be fine-tuned with synthetic data, eliminating the necessity of distilling more powerful LLMs like ChatGPT.

To assess the efficacy of our approach, we meticulously construct a test scenario and carefully assemble a dataset consisting of high-quality Chinese magazine articles centered around daily topics, along with corresponding test queries. Human evaluation demonstrates that our method excels in generating queries of superior quality and in enhancing the performance of fine-tuned models. Additionally, automatic evaluations conducted on four popularly used English benchmarks with relevant metrics further highlight the significant advantages our method holds in capturing contextual knowledge when compared to other state-of-the-art context-driven SFT data generation approaches.

Specifically, the contributions of our work lie on:

• We propose AUGCON, which can automatically generate multi-granularity context-driven SFT data from the corpus for LLMs at scale with high diversity, quality, and fidelity, providing the solution to a realistic industrial and academic problem worth studying.

• Our ideas of deriving queries via CST, training the scorer using contrastive learning to collaborate with the generation process to refine data, and synergistic integrating self-alignment and self-improving to obtain high-fidelity responses, are very novel and may inspire further works. • Extensive experiments incorporating both automatic and human evaluations, encompassing four widely-used benchmarks and a test scenario in English and Chinese compared with other state-of-the-art methods demonstrate the effectiveness and advantages of AUGCON. • To boost the academy and for others to generate highdiversity SFT data on their own corpus without effort, we open-source all of our code, dataset, and fine-tuned model at: https://github.com/quanshr/AugCon.

# 2 Related Work

Synthetic Data for Language Models Due to the challenges of data scarcity (Babbar and Schölkopf 2019), privacy concerns (Abay et al. 2019), and the sheer cost of data collection and annotation (Gilardi, Alizadeh, and Kubli 2023), synthetic data has emerged as a promising solution to build large, diverse, and high-quality datasets at scale (Liu et al. 2024b). One benefit of synthetic data is it can be tailored to specific requirements (Cheng, Huang, and Wei 2023; Jiang et al. 2024), with practical applications having been employed in various domains. WizardMath (Luo et al. 2023) leverages a series of operations to increase the complexity of questions and answers using GPT-3.5, while Reflexion (Shinn et al. 2024) employs external or internally simulated linguistic feedback to improve the code reasoning capabilities of language models. Similarly, Toolformer (Schick et al. 2024) learns to decide which APIs to call and what arguments to pass by training on template-generated data. In addition, synthesized data has been proven effective in mitigating hallucination (Wei et al. 2023) and aligning with shared human preferences and values (Bai et al. 2022). While the generation of contextdriven synthetic data has proven to be a powerful substitute for manual annotation, the challenge of ensuring highquality synthetic data, which encompasses the complexity of queries (Liu et al. 2023a), the diversity of semantics (Ding et al. 2023), and the scale of the synthetic datasets (Yuan et al. 2023; Li et al. 2023), has been a consistent pursuit.

Context-Driven Synthetic Data Numerous studies have developed techniques for creating synthetic data informed by contextual cues (Shaikh et al. 2024; Chang, Peng, and Chen 2023; Liu et al. 2023e; Pal et al. 2024). UltraChat (Ding et al. 2023) leverages user-specified topics and supplements these with existing textual material to craft instructional conversations aimed at enhancing chatbot performance. SPIN (Chen et al. 2024), on the other hand, autonomously generates training data from its previous iterations, employing this approach to progressively refine its capabilities. RECOST (Zhang et al. 2024) selects top-tier instructional content by incorporating external knowledge to assess synthesized examples using an in-context relative predictive entropy measure. Additionally, various methods have been devised to extract character profiles and personas from collected books or scripts for the purpose of producing roleplaying dialogues (Shao et al. 2023), and several initiatives focus on mining domain-specific data from specialized corpora to construct domain-specific language models (Cheng, Huang, and Wei 2023). While alternative approaches employ retrieval augmented generation (RAG) (Ram et al. 2023; Borgeaud et al. 2022) or integrate auxiliary knowledge in vast context windows (Xiong et al. 2023; An et al. 2024), issues like entity susceptibility (Du et al. 2024), high inference computational demand (Liu et al. 2022; Hao et al. 2022), and alignment difficulties with formats and preferences (Qi et al. 2023; Mosbach et al. 2023) highlight the crucial role of context-driven SFT in effectively incorporating corpus knowledge internally.

# 3 Our Method: AUGCON

In this section, we delve into the details of our proposed AUGCON.

# 3.1 Preliminary

We have a raw custom corpus $\mathcal { C } = \{ C _ { 1 } , C _ { 2 } , \ldots , C _ { n } \}$ with each context $C _ { i }$ represents a continuous text chunk extracted from corpus $\mathcal { C }$ , the instruct prompt $I _ { \mathrm { C S T } }$ and few-shot examples $E _ { \mathrm { C S T } }$ for Context-Split-Tree and $I _ { R }$ and $E _ { R }$ for answering the queries, and several response principles $\mathcal { P }$ representing the human demands on responses when answering questions 1. The $E _ { R }$ are context-query-response triplets and will follow the response principles, represented as $E _ { R } \sim \mathcal { P }$ .

Our task is to generate a specific number of SFT queryresponse pairs $\mathcal { D } = \{ ( q _ { i , j } , r _ { i , j } ) \}$ that each pair derives from either the whole or part of context $C _ { i }$ . The derived triplet $( C , q , r )$ should also follow the response principles $\mathcal { P }$ , and the generated $\mathcal { D }$ is expected to have high diversity, quality, and fidelity.

# 3.2 Recursively Deriving Queries via Context-Split-Tree

This step is to derive context-query pairs $( C , q )$ from the given corpus $\mathcal { C }$ . Previous approaches applied regex-based or predetermined prompts for query generation, which often led to queries that were relatively monotonous in structure and granularity. We believe that this type of approach did not fully exploit the context, leading to queries incapable of effectively provoking the model’s capability to comprehend and differentiate between various levels of detail within the context, resulting in suboptimal outcomes.

To address this issue, we propose a very novel and effective method called Context-Split-Tree (CST), with the pseudocode shown in Algorithm 1. CST starts with an entire context $C$ , with each attached with the instruct prompt $I _ { \mathrm { C S T } }$ and few-shot examples $E _ { \mathrm { C S T } }$ to call an LLM to generate a query $q$ deriving from the entire context. At the same time, we ask the LLM to semantically divide the context into two child contexts $C _ { 1 }$ and $C _ { 2 }$ , and the instruct prompt is designed with hints to let the LLM polish the two split contexts to make them as independent as possible and collectively encompass the entirety of the original context. Each child context will continue to recursively derive query and split until reaching a point where one of its split child context lengths is not less than itself or the length falls below a predetermined threshold $\lambda$ . At this point, we consider it to have been split into the minimum granularity and cannot be further divided. Upon the completion of this recursive process, a binary tree structure is formed, with the initial context at the root, and each node representing a context along with its corresponding query tailored to its specific granularity. We collect data from all nodes as the outcome of this step.

# Algorithm 1: Context Split Tree

Input: A corpus $\mathcal { C }$ , CST prompt instruction $I _ { \mathrm { C S T } }$ , CST few  
shot examples $E _ { \mathrm { C S T } }$   
Output: Query dataset Data comprises of split context and   
derived query pairs   
1: function CONTEXTSPLITTREE $( C , D a t a )$   
2: if $l e n ( C ) < \lambda$ then   
3: return $D$ Below the minimum granularity   
4: end if   
5: Call LLM to get $C _ { 1 } , C _ { 2 } , q \gets \mathrm { L L M } ( I _ { \mathrm { C S T } } , E _ { \mathrm { C S T } } , C )$   
6: Append $( C , q )$ to Data   
7: if $l e n ( C _ { 1 } ) ~ \ge ~ l e n ( C )$ or $l e n ( C _ { 2 } ) ~ \ge ~ l e n ( C )$ or ROUGE- $\mathrm { . L } [ \mathrm { P } ] < 0 . 7$ then   
8: return $D$ The signs of hallucinations   
9: end if   
10: CONTEXTSPLITTREE $( C _ { 1 } , D a t a )$ $D$ Recursive calling   
11: CONTEXTSPLITTREE $( C _ { 2 } , D a t a )$   
12: end function   
13:   
14: Initialize $D a t a \gets$ empty list   
15: for each extracted context $C \in { \mathcal { C } }$ do   
16: CONTEXTSPLITTREE $( C , D a t a )$   
17: end for   
18: return Data

The minimum length threshold $\lambda$ and the initial context length l are like the lower bound and upper bound to control the granularity distribution of generated questions. One can easily adjust the overall average granularity of generated queries by adjusting the length threshold. Similarly, if we seek to address more global questions, we can do it by simply increasing the initial context length, as long as the model’s context window permits. One beneficial property of CST is that the number of questions ultimately generated will maintain a linear relationship with the length of the initial text provided. This ensures that adjusting the length of the segmented contexts in the corpus does not lead to significant fluctuations in the total number of queries obtained, but rather merely shifts the distribution of query granularity. By employing CST, we can produce queries that span across different levels of details in the context, and these queries naturally have little redundancy or repetition, enabling more efficient use of the context information and stimulating the model’s capability to comprehend and grasp the context in different granularities. Moreover, another benefit of CST is that the derived queries just match the split context, making the later generated response to these queries more accurate and pertinent with less unrelated information.

![](images/ab804eb2979ee55b8abf4dec4053621bd8df72f1d62568761ae9e551fb376aa3.jpg)  
Figure 1: An overview of the proposed AUGCON.

# 3.3 Contrastive-Learning-Based Scorer to Rank Queries and Filtering

To further enhance the quality and diversity of the generated data, we introduce an effective ranking and filtering strategy collaborating with CST. Previous works have attempted to filter training data via heuristic algorithms, such as filtering out queries that are too long or too short (Wang et al. 2023a). Other works that are more relevant to us attempt to train scorers to judge the complexity and quality of questionresponse pairs (Liu et al. 2023a), but they need to have a step of distillation on stronger LLM APIs like ChatGPT, and their training methods are less effective. For example, they put a series of responses and ask for direct ranking, suffering from the positional bias (Liu et al. 2024a) in LLMs, or ask LLMs to directly assign a scalar score to a response, which is unstable. In this work, we apply contrastive learning to train a scorer to judge the degree of adherence to instruct prompt and few-shot examples, which is data-efficient and can achieve effective performance without the need for stronger LLMs.

The structure of our scorer is obtained by adding a linear head after the base model to map the last hidden state to a one-dimensional space. We take contextquery pairs as inputs, applying scorer $\mathit { S c }$ to yield a scalar score $s \ = \ \cdot \ \ - s c ( \vec { C , q } )$ . We use the context query pairs obtained from Step 1 as positive samples: $\begin{array} { l l } { \displaystyle q ^ { + } } & { = } \end{array}$ $\mathbf { \bar { L } L M } ( I _ { \mathrm { C S T } } , E _ { \mathrm { C S T } } , C )$ , and obtain negative samples by manipulating the instruct prompt (use suboptimal instructions): $\boldsymbol { q } ^ { - } \ = \ \mathrm { L L M } ( \boldsymbol { I _ { \mathrm { C S T } } } ^ { - } , \boldsymbol { \bar { E } _ { \mathrm { C S T } } } , \boldsymbol { \bar { C } } )$ , few-shot examples (reduce ICL examples count): $q ^ { - } = \mathrm { L L M } ( I _ { \mathrm { C S T } } , E _ { \mathrm { C S T } } { } ^ { - } , C )$ or both of them: $q ^ { - } \ = \mathrm { L L M } ( I _ { \mathrm { C S T } } { } ^ { - } , E _ { \mathrm { C S T } } { } ^ { - } , C )$ . Note that we do not generate all corresponding negative examples for positive data for training scorer, but rather randomly select a very small number of samples (e.g. only 500 pairs for each negative types in our implementation) to form the training set $D _ { S c }$ . Then, the loss function of scorer can be represented as:

$$
\mathcal { L } = - \mathbb { E } _ { ( C , q ^ { + } , q ^ { - } ) \sim D _ { S c } } [ \log ( \sigma ( S c ( C , q ^ { + } ) - S c ( C , q ^ { - } ) ) ) ] .
$$

We use the trained scorer applied on all the context query pairs obtained in Step 1 to get their scores. For each root context, we rank all queries from its CST in descending order of scores. Then, we start with an empty set and add one training query each time, only if the current query has a ROUGE-L precision score of less than 0.7 compared to any previously added queries. We will stop adding as the count reaches the limit. Each context will form such a set, and ultimately, we consolidate and retain the training data from all the sets. Through this approach, we can obtain diverse data and easily control the quantity, for it makes it possible to apply multi-times CST in the same context and filter the repeated one.

# 3.4 Obtaining High-Fidelity Responses

Inspired by the significant impact principles (Sun et al. 2024, 2023) have on LLMs, this principle-driven self-alignment step begins by appending the context and a set of helpful, ethical, and reliable principles to the LLM. These principles are meticulously crafted to ensure the LLM’s outputs are closely aligned with human preferences or mimic certain response tones. Before initiating the response generation, we deploy a self-improving pipeline that makes the LLM self-evaluate its response and sift through the entire set of human-annotated Q&A pairs $E _ { R }$ , where random search is applied to find the most fitting few-shot examples to help

LLM generate high-fidelity responses under the predetermined principles, denoted as $E _ { R } { } ^ { \stackrel { \textstyle > } { } }$ .

Our innovative synergistic integration of the principledriven self-alignment with self-improving methodology effectively improves the fidelity of generated responses. Following this, we execute $\mathrm { L L } \dot { \bf M } ( I _ { R } , { \cal E } _ { R } { } ^ { \prime } , C , q )$ to elicit each response $r$ , ensuring that each response is not only in high quality but also in good alignment with the established principles. Notably, due to the precise matching of each query with its context’s granularity within the CST framework, the LLM can effortlessly provide accurate and pertinent responses to the queries.

After obtaining all generated data, we prune all context, instruction, and response principles and only retain synthetic query response pairs as SFT data. This approach allows the fine-tuned LLM to potentially learn the methods and nuances of responding to queries in a manner that naturally aligns with human expectations, enabling the LLM to directly generate responses that are well-aligned with reliable principles and optimal ICL exemplars across a wide range of queries. It’s important to note that the fine-tuned LLM can generate high-quality responses without the need to explicitly reference the principles set and ICL exemplars.

# 4 Evaluations

# 4.1 Baselines

To demonstrate the advantages of our method, we meticulously collect the following relevant baselines from a wide range of research. The set of contexts, base language models, and quantity of retained query-response pairs are maintained the same (if applicable) on both the baselines and our method to ensure a fair comparison.

(1) Chat Model (Bai et al. 2023; Touvron et al. 2023) applies instruction tuning and alignment tuning after pretraining. We utilize it both as the basic baseline and as the base model for calling and fine-tuning across all other baselines and our methods for fair comparison.

(2) DAPT (Gururangan et al. 2020) continuously pretrains directly on the raw custom corpus to adapt and grasp domain-specific knowledge.

(3) AdaptLLM (Cheng, Huang, and Wei 2023) builds SFT samples by converting the raw corpora into reading comprehension tasks via regex-based mining patterns. Tasks they design include summarization, word-to-text, natural language inference, commonsense reasoning, and paragraph detection.

(4) ETRC (Jiang et al. 2024) derives question-answer pairs from extracted contexts with an LLM and augments data by ensembling contexts and their corresponding question-answer pairs with a length-based clustering algorithm.

(5) Context-Instruct (Wang et al. 2023b) is a contextdriven instruction generation method that contains three parts: 1) partition text into manageable segments, 2) use an LLM to generate question, response, and confidence score triplets based on the segments, and 3) apply confidencescore-based filtering and deduplication to ensure data quality and diversity.

We also notice that there are several alternative methodologies such as RAG and long context LLMs, but we don’t compare them as we have a huge difference both in training and inference (Mosbach et al. 2023; Liu et al. 2022). We encourage interested readers to refer to Section 2 for more relevant information.

# 4.2 Automatic Evaluation

To objectively assess the performance of our approach, we conduct automatic evaluations on four widely used benchmarks: SQuAD1.1 (Rajpurkar et al. 2016), TriviaQA (Joshi et al. 2017), DROP (Dua et al. 2019), and WebGLMQA (Liu et al. 2023b). All these benchmarks contain a variety of test QA pairs with specific contextual references. In the evaluation, we compile the context from each benchmark into a single corpus, and then apply all baselines and our AUGCON on it and test on the test QA pairs in benchmarks.

Metrics For datasets featuring short-form responses (applied to the SQuAD1.1, TriviaQA, and DROP datasets), we measure the model’s performance using exact matching (EM) accuracy. A response is considered correct if and only if it matches any of the possible answers. For datasets with long-form responses (applied to the WebGLM-QA dataset), we employ BERTScore (BS) (Zhang et al. 2019) (we use Roberta-Large (Liu et al. 2019) for calculation) to evaluate the semantic similarity between the generated outputs and the reference responses.

Results We use Llama3-70B-Instruct (AI $@$ Meta 2024) as the base model for calling and conducting fine-tuning for automatic evaluations for all our baselines and the proposed AUGCON. The detailed results are shown in Table 1. The results illustrate that our proposed method consistently outperforms the established baselines across all four datasets. Specifically, when analyzing short-form datasets, it becomes evident that the data generated by AUGCON surpasses the comparative methods in extracting pivotal information and knowledge from the corpus, thus improving the questionanswering accuracy of fine-tuned models. Meanwhile, the exceptional performance of AUGCON on datasets emphasizing long-form responses showcases its proficiency in generating high-fidelity query-response pairs. This capability directly contributes to enhancing the effectiveness of chat models, enabling them to deliver more relevant, engaging, and contextually appropriate responses based on the given corpus. This, in turn, significantly improves the overall user experience by ensuring that interactions are not only informative but also closely aligned with the user’s specific curiosities and requirements.

Furthermore, the consistency of AUGCON in achieving top results across all four datasets, each with unique query patterns and focuses, speaks volumes about its versatility and adaptability. Such consistent performance across varied datasets also underscores the robust generalization ability of our method, making it a highly effective tool for a broad spectrum of corpora types and catering to diverse user interests and inquiries.

<html><body><table><tr><td></td><td colspan="3">Short-form (EM)</td><td colspan="2">Long-form (BS)</td></tr><tr><td>Method</td><td>SQuAD1.1</td><td>TriviaQA</td><td>DROP</td><td>WebGLM-QA</td></tr><tr><td>Llama3-C70B</td><td>0.212±0.004</td><td>0.723±0.003</td><td>0.220±0.004</td><td>0.837±0.002</td></tr><tr><td>DAPT</td><td>0.258±0.004</td><td>0.767±0.003</td><td>0.266±0.004</td><td>0.851±0.002</td></tr><tr><td>AdaptLLM</td><td>0.273±0.003</td><td>0.791±0.004</td><td>0.284±0.003</td><td>0.842±0.001</td></tr><tr><td>ETRC</td><td>0.301±0.004</td><td>0.812±0.003</td><td>0.326±0.004</td><td>0.903±0.001</td></tr><tr><td>Context-Instruct</td><td>0.314±0.003</td><td>0.825±0.003</td><td>0.334±0.003</td><td>0.885±0.001</td></tr><tr><td>AUGCON(Ours)</td><td>0.336±0.004</td><td>0.849±0.003</td><td>0.350±0.003</td><td>0.924±0.002</td></tr></table></body></html>

Table 1: The results of automatic evaluation on four benchmarks. We run 10 times for each test and report the mean value and standard deviation, with the best results shown in bold.

# 4.3 Human Evaluation

In human evaluation on the test scenario, we meticulously curate a corpus dataset, referred to as the DailyM dataset, which consists of $1 , 0 0 0$ articles carefully selected from a variety of high-quality Chinese magazines closely related to daily life. These articles extensively cover issues of widespread public concern such as basic livelihood, politics, economics, and law, with each article containing approximately 4, 000 Chinese characters. Then, we test how well our method and baselines build an AI chat assistant specialized in this daily concern corpus. We apply our method on DailyM to generated SFT data called DailyM-SFT and use these data to fine-tune Qwen1.5-32B-Chat (Bai et al. 2023) to get fine-tuned model Qwen-DailyM-32B. To further test our method, we conduct annotators to write a total of 1, 000 queries they are interested in related to these articles, forming the DailyM test set.

Metrics In our comprehensive evaluation framework, we assess both the generated queries and the outputs under the DailyM test set of the fine-tuned models to ensure a holistic understanding of the method’s performance. Specifically, we evaluate the realism and diversity of generated queries and the relevance, accuracy, and satisfaction of fine-tuned models’ outputs.

For both generated queries and model outputs, evaluators are provided with detailed scoring rubrics and examples to promote consistency in evaluation. The queries and outputs will be reviewed by multiple independent evaluators to ensure a balanced and objective assessment, with average scores calculated for each metric to determine the overall performance.

Results For all our baselines and the proposed AUGCON, we employ Qwen1.5-32B-Chat (Bai et al. 2023) as the base model for calling and conducting fine-tuning for later evaluations. For methods such as AdaptLLM, ETRC, ContextInstruct, and our AUGCON which generate query-response pairs based on context, we adhere to a standard where every 35 Chinese characters derive one query-response pair to ensure a fair comparison. We limit the number of generated entries to the same in the comparison because we find that all methods spend much more time on final fine-tuning process compared to the previous generation.

![](images/d58185b445fd24bcbe46e322451d2596f9ca49698ab9c418ea6160589c0d6cf5.jpg)  
Figure 2: The results of human evaluation on DailyM. Query metrics are not applicable for the base chat model and DAPT so we don’t show them.

Figure 2 presents the results of the human evaluation on the DailyM test set. The results demonstrate that AUGCON consistently surpasses the baseline methods across all evaluation metrics. Specifically, the superior performance in terms of query realism and diversity underscores our method’s ability to produce human-like and high-diversity queries. Since our CST and filtering process effectively gain multi-granularity queries that are more effective in covering all granularity levels of context, the derived data will extract more useful knowledge from the corpus. Furthermore, the impressive performance in judging relevance, accuracy, and satisfaction in responses from fine-tuned models further validates that our method’s high-quality and diverse queries, coupled with high-fidelity responses, can indeed enhance the performance of subsequently fine-tuned models and achieve higher satisfaction scores from humans. This suggests that AUGCON is particularly adept at constructing high-quality supervised fine-tuning data for LLMs from a given corpus.

# 4.4 Abalation Study

In this section, we conduct ablation experiments to assess the indispensability and impact of the three essential steps in our proposed method. Specifically, we develop four distinct variations of our method, with each one specifically tailored to concentrate on a fundamental step: (1) $\mathbf { A U G C O N } _ { \mathbf { C S T 1 } } ^ { w / o }$ drops the CST part and replaces it by directly iteratively deriving queries from the extracted context until the desmiroevdesnuthmebeurseofofquLeLriMes ios sopbltiat inedt.h(e2)C $\mathbf { A U G C O N } _ { \mathbf { C S T 2 } } ^ { w / o }$ rdeirectly splitting the contexts in the middle (we will set the whole sentence in the middle all belongs to the first subcontext to maintain semantic integrity). (3) $\mathbf { A U G C O N } _ { \mathbf { f i l t e r } } ^ { w / o }$ eliminates the contrastive-learning-based score training and filtering process and randomly samples a sufficient number of queries. (4) $\mathbf { A U G C O N } _ { \mathbf { f i d e l i t y } } ^ { w / o }$ obtains the answers to the queries without adhering to self-alignment and selfimproving but utilizes fixed few-shot examples along with a straightforward prompt design devoid of guiding principles.

We implement the four variants on TriviaQA (short-form) and WebGLM-QA (long-form) datasets and conduct a comparison with our AUGCON. The results are shown in Table 2.

Table 2: The results of ablation study.   

<html><body><table><tr><td>Variant</td><td>Short-form (EM) TriviaQA</td><td>Long-form (BS) WebGLM-QA</td></tr><tr><td>AuGCoN5</td><td>0.793±0.003</td><td>0.912±0.001</td></tr><tr><td>AUGCON5P</td><td>0.826±0.003</td><td>0.910±0.001</td></tr><tr><td>AUGCON</td><td>0.828±0.003</td><td>0.915±0.001</td></tr><tr><td>w/o AUGCONfidelity</td><td>0.833±0.004</td><td>0.907±0.002</td></tr><tr><td>AUGCON</td><td>0.849±0.003</td><td>0.924±0.002</td></tr></table></body></html>

We find that all variants yield suboptimal outcomes, underscoring the fact that the three essential steps are all crucial and collectively contribute to achieving superior performance.

# 4.5 Training Phase

We present the loss curve training on the generated DailyM$S F T$ in Figure 3. An interesting observation is that the training loss appears to plateau within epochs from Epoch 2 onwards, yet we observe sudden drops in loss at the boundaries between two consecutive epochs. This pattern strongly signals that our training dataset is characterized by extremely low similarity and exceptionally high diversity, meaning that training on one segment of data does not have an impact on the loss associated with another segment.

We also conduct a human evaluation at each checkpoint during model training, on DailyM test set and a widelyused general alignment benchmark AlignBench (Liu et al. 2023c). The overall satisfaction scores increase steadily in both the DailyM test set and AlignBench, showcasing that our methods can increase the specific conversation ability without sacrificing general performance.

# 4.6 Granularity Comparison

A key innovative advantage of AUGCON is its ability to generate queries of varying granularity. To assess the performance of this feature in comparison to baseline methods, we categorize questions into three distinct types based on their scope and depth: detail, concept, and macro, and compare our method with Context-Instruct.

![](images/8c05defe11d754aa672098213188159b3be3a1e29e21677ccce3ea0a02ba4385.jpg)  
Figure 3: The training loss and human evaluation results during training phase.

![](images/f772646d265a16fb6913c589e8ad10fbadf4cf126cf0c67e44c8915473880590.jpg)  
Figure 4: The proportion of three levels of granularity questions generated by AUGCON and Context-Instruct.

The proportions of the three types of questions are illustrated in Figure 4. Our approach achieves a more balanced distribution of question granularities, demonstrating its advantage in covering a diverse range of user inquiries and providing an intuitive explanation for our superior performance.

# 5 Conclusion

In this work, we propose AUGCON, a highly innovative and effective method to build vertical-domain AI assistants from custom corpora by deriving SFT query-response pairs with diverse granularity. AUGCON starts with query generation through the Context-Split-Tree (CST), an innovative approach for recursively deriving queries and splitting context to cover full granularity. We then employ contrastive learning to develop a scorer that works with CST to rank and refine queries. Finally, we introduce a synergistic integration of self-alignment and self-improving to obtain high-fidelity responses. We conduct extensive experiments on Qwen1.5- 32B-Chat and Llama3-70B-Instruct models. The automatic evaluation on four benchmarks and human evaluation on a test scenario demonstrate the significant advantages of our method in producing high diversity, quality, and fidelity context-driven SFT data and improving the performance of custom fine-tuned models against existing methods.