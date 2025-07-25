# Just What You Desire: Constrained Timeline Summarization with Self-Reflection for Enhanced Relevance

Muhammad Reza $\mathbf { Q o r i b } ^ { 1 }$ , Qisheng $\mathbf { H } \mathbf { u } ^ { 2 * }$ , Hwee Tou $\mathbf { N g } ^ { 1 }$

1Department of Computer Science, National University of Singapore 2College of Computing and Data Science, Nanyang Technological University mrqorib@u.nus.edu, qisheng $\operatorname { 0 0 1 } \ @$ e.ntu.edu.sg, nght@comp.nus.edu.sg

# Abstract

Given news articles about an entity, such as a public figure or organization, timeline summarization (TLS) involves generating a timeline that summarizes the key events about the entity. However, the TLS task is too underspecified, since what is of interest to each reader may vary, and hence there is not a single ideal or optimal timeline. In this paper, we introduce a novel task, called Constrained Timeline Summarization (CTLS), where a timeline is generated in which all events in the timeline meet some constraint. An example of a constrained timeline concerns the legal battles of Tiger Woods, where only events related to his legal problems are selected to appear in the timeline. We collected a new human-verified dataset of constrained timelines involving 47 entities and 5 constraints per entity. We propose an approach that employs a large language model (LLM) to summarize news articles according to a specified constraint and cluster them to identify key events to include in a constrained timeline. In addition, we propose a novel self-reflection method during summary generation, demonstrating that this approach successfully leads to improved performance.

# Code and Data — https://github.com/nusnlp/reacts

# Introduction

In today’s internet era, the rapid and massive flow of information makes it hard to stay updated, particularly for topics with extensive coverage over time. In the United States alone, Hamborg, Meuschke, and Gipp (2018) report that more than 5,000 news articles are being published every day. To help readers quickly grasp important information, many news platforms offer news in a timeline format, especially for important topics that progress over time, such as pandemics1 or conflicts2.

The task of summarizing news articles, or any collections of text documents, into timelines is called timeline summarization (TLS). Timeline summarization aims to automati

# Original Timeline

<html><body><table><tr><td></td><td>2003-11-19: Receives the National Book Foundation Medal for Distinguished Contribution to AmericanLetters. 2015-09-1O: Is awarded the National Medal of Arts by</td></tr><tr><td></td><td>US President Barack Obama. 2015-11-O3: Releases a collection of short stories enti-</td></tr><tr><td></td><td>tled“The Bazaar of Bad Dreams." 2018-07-25: King is an executive producer of a show written for the streaming service Hulu. The</td></tr><tr><td></td><td>series,“Castle Rock,’is named after the fictional small Maine town that provides the setting for various King books and sto- ries. 2020-O4-21: King's latest book,“If It Bleeds”is pub</td></tr><tr><td></td><td>lished.The book is a compilation of four novellas. 2021-O6-O4: The miniseries “Lisey's Story,adapted by</td></tr><tr><td></td><td>King and based on his 2OO6 novel of the same name,premieres on Apple TV+. 2022-09-O6: King's novel “Fairy Tale” is published.</td></tr></table></body></html>

Table 1: An unconstrained timeline of Stephen King and a constrained version focusing on Stephen King’s book releases.   

<html><body><table><tr><td colspan="2">Constrained Timeline</td></tr><tr><td>Constraint:Focus on Stephen King's book releases.</td><td>2015-11-O3: Releases a collection of short stories enti-</td></tr><tr><td></td><td>tled“The Bazaar ofBad Dreams." 2020-04-21: King's latest book,“If It Bleeds” is pub-</td></tr><tr><td></td><td>lished.The book is a compilation of four novellas.</td></tr><tr><td></td><td>2022-09-O6: King's novel “Fairy Tale” is published.</td></tr></table></body></html>

cally condense long-running news topics into temporally ordered time-stamped textual summaries of events on a particular topic. Timeline summarization aims to include any important events into the timeline without considering particular aspects that the readers are interested in.

To take a reader’s interest into account when generating a timeline, we propose a new task called constrained timeline summarization (CTLS). Constrained timeline summarization offers personalization that TLS lacks. For example, a reader may want to automatically retrieve the timeline of Stephen King’s book publication (Table 1). In this example, Stephen King’s national awards are irrelevant to the reader even though they are generally considered important events in Stephen King’s life.

The contributions of this paper are as follows:

• We propose a new task, constrained timeline summarization. The task has real-life applications.   
• We present a new test set to benchmark models on the constrained timeline summarization task.   
• We present an effective method that utilizes large language models without any need for training or finetuning.   
• We propose a novel self-reflection method to produce a more relevant constrained event summary and demonstrate that self-reflection helps in generating more relevant constrained timelines.

# Related Work

In this section, we briefly discuss related work on timeline summarization, query-based summarization, and update summarization. Constrained timeline summarization can be viewed as an amalgamation of the first two tasks.

# Timeline Summarization

Previous work on timeline summarization can be categorized into three main approaches: direct summarization, date-wise approaches, and event detection approaches.

Direct Summarization In this approach, a collection of documents is treated as a set of sentences to be directly extracted. Sentence extraction can be performed by optimizing sentence combinations (Martschat and Markert 2018) or by ranking sentences (Chieu and Lee 2004). This category also includes methods that treat the task as an extension of multi-document summarization, where the goal is to generate a summary from multiple documents (Allan, Gupta, and Khandelwal 2001; Yu et al. 2021).

Date-wise Approach In this approach, the task is divided into two steps: identifying important dates and summarizing events that occurred on those dates. Most methods employ supervised techniques to select the dates. For instance, Ghalandari and Ifrim (2020) propose a classification or regression model to predict date importance, while Tran, Herder, and Markert (2015) utilize graph-based ranking for date selection.

Event Detection In this approach, the system first detects important events from the articles by clustering them based on similarity. It then ranks and selects the most important clusters and summarizes them into event descriptions. Various techniques have been proposed for clustering, including Markov clustering on bag-of-words features (Ghalandari and Ifrim 2020), dynamic affinity-preserving random walks (Duan, Jatowt, and Yoshikawa 2020), event graph compression (Li et al. 2021), date graph model (La Quatra et al. 2021), heterogeneous graph attention networks (You et al.

2022), and even large language models (Hu, Moon, and $\mathrm { N g }$   
2024).

# Query-Based Summarization

Query-based summarization, also called query-focused, topic-based, or user-focused summarization, aims to extract and summarize information that users are specifically interested in from a large number of texts. Essentially, it is a type of summarization that leverages user-provided query information.

Early approaches to query-based summarization mainly score or rank the relevance of each sentence in the document to the query based on predefined features (Rahman and Borah 2015). Sentences with the highest scores are then extracted to create the summary. Relevance scoring can be performed in an unsupervised manner by utilizing lexical and semantic features (Conroy, Schlesinger, and O’Leary 2006; Krishna, Kumar, and Reddy 2013) or in a supervised manner by training regressor models (Mani and Bloedorn 1998; Ouyang et al. 2011). Document graphs are also often employed when dealing with multiple documents (Mohamed and Rajasekaran 2006; Wang et al. 2013).

Due to the effectiveness of transformers, recent querybased summarization methods are predominantly based on transformer models, including large language models. For example, Laskar, Hoque, and Huang (2020) incorporate query relevance into BERTSUM (Liu and Lapata 2019), while Park and Ko (2022) integrate a query-attentive semantic graph with sequence-to-sequence transformers. Finetuning large language models has also been explored, such as in the work by Xu et al. (2023), who fine-tune BART (Lewis et al. 2020), and Cao et al. (2024), who fine-tune Llama 2 (Touvron et al. 2023) using custom adapters.

# Update Summarization

Update summarization is the task of generating a short summary from a set of documents $A$ under the assumption that users have read a set of documents $B$ (Dang and Owczarzak 2009). Update summarization has a different objective from timeline summarization, but the methods proposed for it often bear some resemblance to the event detection approach for timeline summarization, notably in determining the novelty of the information from set $A$ in relation to set $B$ (Steinberger and Jezˇek 2009). In the context of timeline summarization, novelty detection involves determining whether the extracted events from set $A$ are new events that are not present in set $B$ .

# Dataset

To benchmark the constrained timeline summarization task, we propose a novel test set called CREST (Constraint Restrictions on Entities to Subset Timelines). CREST consists of 235 timelines from 47 public figures or institutions (entities). We derive these timelines from the ENTITIES dataset (Ghalandari and Ifrim 2020), which were crawled from CNN Fast Facts. For each entity, we generate 5 pairs of constraints and corresponding subset timelines. The article

围 3. Event   
Timeline ArNtiecwless Filtering 圃 ? .   
LLM . Constraint An2.noEtvaetniot n 圃 LLM CoFnilstterraeinded Generation 自 Timeline Annotators Constrained Timeline Constraint

pool is sourced from the ENTITIES dataset, which was collected from The Guardian using the official $\mathsf { A P I } ^ { 3 }$ . AsLLsuMch, our dataset is limited to British and American news sources. The dataset creation process involves constraint generation, event annotation, and event filtering.

# Constraint Generation

We generate constraints by prompting GPT- $\mathbf { \cdot } 4 \mathbf { o } ^ { 4 }$ and manually selecting the best 5 constraints for each timeline. The prompt instructs GPT-4o to propose single-sentence constraints in the format “Focus on ...”. To ensure that we have a variety of constraints, we query GPT-4o with four different types of prompts: general, numerical, relational, and geographical. The general prompt asks GPT-4o for constraint suggestions without any additional specification. The numerical prompt asks GPT-4o to generate constraints that contain ordinal phrases (e.g., first, second, etc.) or time indicators (e.g., timestamp, month, year, etc.). The relational prompt focuses on constraints involving some relationship between the entity and other public figures or institutions (e.g., “Focus on Stephen King’s interactions with President Barack Obama.”), while the geographical prompt asks for constraints with geographical information.

We find that GPT-4o generally suggests good constraints, but occasionally, the suggestions include hallucinations (e.g., constraints referencing non-existent events in the timeline) or are overly specific (e.g., only applicable to one event). Therefore, a human-in-the-loop process is essential to curate a good set of constraints for each timeline. Human intervention involves selecting the proposed constraints and modifying them to be more general.

# Event Annotation

Given a list of events $E _ { t }$ from timeline $t$ and a set of constraints $C _ { t }$ for timeline $t$ , we build the constrained timelines by asking human annotators to label whether each event in the timeline adheres to each constraint. All constraints are applied to all events in the timeline, resulting in $| E _ { t } | \times | C _ { t } |$ assertions. Each annotator is provided with the complete timeline (containing all events) and the full set of constraints for that timeline.

We recruited 4 university students with strong English proficiency as annotators. To ensure high-quality annotations, the annotators completed a qualifying test by performing annotations on a different timeline. The annotators performed the task for approximately four hours and were compensated above standard rates ( $\$ 22.65$ /hour). We found that our test set had high inter-annotator agreement, with an exact match percentage of $9 4 . 7 \%$ and a Cohen’s kappa of 0.78 between the first and second annotators and an exact match percentage of $9 6 . 2 \%$ and a Cohen’s kappa of 0.88 between the third and fourth annotators.

# Event Filtering

One challenge with the ENTITIES dataset is that the article pool and timelines were collected independently from different sources. This causes a mismatch between the events covered by the articles and those included in the timelines. As a result, some important events in the ground-truth timelines are not covered by the article pool, making it impossible for the model to generate them without external knowledge. In such cases, even a human would not be able to achieve a perfect score.

To avoid unfairly penalizing an automatically constructed model, we provide an additional evaluation setting in which events in the timelines that are not covered by the article pool are filtered out. Given that the article pool contains more than forty-five thousand news articles, manually checking event coverage would be too costly and labor intensive. Following previous work (Gilardi, Alizadeh, and Kubli 2023), we utilize GPT-4o to check each article for information related to the events in question.

It is important to note that we only filter out events from the timelines, while the article pool remains unchanged. We assume that the ground truth timelines are comprehensive lists of all significant events related to the entities. We report the statistics of our dataset for both the full and filtered

Table 2: Statistics of our proposed dataset (CREST).   

<html><body><table><tr><td>Statistics</td><td>All Events</td><td>Filtered</td></tr><tr><td># topics</td><td>47</td><td>47</td></tr><tr><td># timelines</td><td>233</td><td>201</td></tr><tr><td># events</td><td>1031</td><td>667</td></tr><tr><td>Avg. # articles per topic</td><td>959</td><td>959</td></tr><tr><td>Avg. # timelines per topic</td><td>4.96</td><td>4.28</td></tr><tr><td>Avg.# events per timeline</td><td>4.42</td><td>3.32</td></tr></table></body></html>

settings in Table 2.

<html><body><table><tr><td>Algorithm1: Method</td></tr></table></body></html>

<html><body><table><tr><td>Require: A queue of articles A, a topic keyword q, a con- straint c,anew article ai that arrived at time i, the event databaseD,the eventclusters G,theretrievallimitN, the number of dates l in the timeline, the number of sentences perdate k in the timeline. Ensure: A timeline Tq,c about topic q following the con- straint c,comprising l timestamped event descriptions, each with k sentences. αi ←DEQUEUE(A) e ← CONSTRAINEDTOPICSUM(ai, q,c) if ei≠NULL then</td></tr><tr><td>if ADHERETOCONSTRAINT(ei, q, c) then Edges←{ for all ej ∈RETRIEVE(D,ei,N) do if SAMEEVENT(ei, ej) then Edges ←Edges U{[ei,ejl} end if</td></tr><tr><td>end for G ← UPDATECLUSTERS(G,Edges) D ← INSERT(D,ei) end if</td></tr><tr><td>end if Clusters ← RANKCLUSTERS(G,l) Tq,c← j↑1 forall v ∈ SORTBYTIME(Clusters) do Tq,c[j] ← SUMMARIZE(u, k)</td></tr></table></body></html>

# Problem Definition

Constrained timeline summarization is a task to generate a timeline $T$ that includes important events related to a topic and adhering to a constraint, given a list of input documents. The input comprises temporally ordered documents $A \ =$ $\{ a _ { 1 } , a _ { 2 } , . . . \}$ related to a specific topic $q$ , a constraint $c$ , the expected number of dates $l$ in the timeline, and the expected number of sentences per date $k$ . The system-generated timeline $T$ will be evaluated against a ground-truth timeline $R$ . Similar to most timeline summarization datasets, the list of documents in this dataset is a list of chronologically ordered news articles. The constraint is a natural language sentence that specifies the kind of events related to the topic $q$ that should be included in the timeline.

# Method

Following the LLM-TLS method (Hu, Moon, and $\Nu \tt { g } 2 0 2 4 )$ , we propose a new approach for the constrained timeline summarization task by leveraging a large language model (LLM) for summarization and clustering, which we call REACTS (REflective Algorithm for Constrained Timeline Summarization). Our method consists of four main steps: event summarization, self-reflection, event clustering, and finally, cluster and sentence selection. The core idea is to summarize each document according to the constraint, cluster the summaries that relate to the same event, and transform the clusters into event descriptions with corresponding dates. We illustrate our method in Figure 2.

# Event Summarization

Inspired by the effectiveness of LLMs in query-based summarization (Jiang et al. 2024), we employ large language models for event summarization. The summary is expected to be in the format of a date followed by a one-sentence summary of a key event in the article related to the keyword and that adheres to the constraint, such as, ”2021-06-04: The miniseries \*Lisey’s Story\*, adapted by King and based on his 2006 novel of the same name, premieres on Apple $\mathrm { T V } +$ .” If there is nothing to summarize that meets the constraint, the model is expected to output NULL. We refer to this process as CONSTRAINEDTOPICSUM in Algorithm 1.

Each article includes a publication date, but the important event may occur sometime before the publication date without an explicit mention of the exact date in the article. To assist the model in generating the correct date, we preprocess the news articles by prepending the sentence with the exact date whenever a time reference is mentioned. For example, if the publication date is 14 August 2024, and a sentence in the article contains a time reference like “yesterday,” the sentence is prepended with “(2024-08-13)”. Similarly, if the article mentions “last Friday,” the sentence is prepended with “(2024-08-09)”. The time references are parsed using HeidelTime5 (Stro¨tgen and Gertz 2015).

# Self-Reflection

Self-evaluation techniques have been reported to improve the reasoning capabilities of LLMs (Weng et al. 2023; Xie et al. 2024). We observe that LLMs often produce an event summary even when it does not adhere to the specified constraint. To mitigate this, we employ self-reflection as an additional verification step by prompting the same LLM to assess whether the summary it just generated, $e _ { i }$ , for topic $t$ adheres to the constraint $c$ . If the model determines that it does not, $e _ { i }$ is discarded and excluded from the timeline generation. We refer to this process as ADHERETOCONSTRAINT in Algorithm 1. The input prompt to perform self-reflection is given in Table 3.

![](images/cd0e8e766ddf1f282ca5a3fde543d48201275bad0d43d4cd8549e3508d85df72.jpg)  
Figure 2: Illustration of our method for constrained timeline summarization (REACTS). The method consists of four steps event summarization, self-reflection, event clustering, and cluster and sentence selection.

Table 3: Prompt template for self-reflection.   

<html><body><table><tr><td>Review the timestamped event description related to key- word,accompanied by a constraint. Please determine whether the event description complies with or corre- sponds to the constraint.Respond with‘Yes’ if the event description aligns with the constraint,or with‘No'if it</td></tr><tr><td>does not. ################# {positive example}</td></tr><tr><td>################# {negative example}</td></tr><tr><td>#################</td></tr><tr><td>### Event</td></tr><tr><td>{event}</td></tr><tr><td>### Constraint {constraint} ### Answer</td></tr></table></body></html>

# Event Clustering

For every event summary $e _ { i }$ that passes the self-reflection step, the event description is encoded using the General Text Embeddings (GTE) model (Li et al. 2023). The summaries are transformed into embedding vectors so that semantically similar event summaries describing the same event can be accurately grouped together into a cluster. GTE performs exceptionally well on the Massive Text Embedding Benchmark (MTEB) while maintaining a modest number of parameters, making it an ideal choice for encoding summaries from tens of thousands of articles.

To generate clusters, from a vector database $D$ , we retrieve $N$ event descriptions that have the closest embedding vectors to the current event description being processed (the

RETRIEVE function in Algorithm 1). For each pair consisting of the current event description and its retrieved neighbor, we use an LLM with few-shot prompting to check whether they describe the same event. In addition to this description matching by the LLM, we also check whether the event dates are the same, as events with similar descriptions but different dates likely represent distinct occurrences. This process is denoted as the SAMEEVENT function in Algorithm 1. If the pair passes both checks, the current event description is added to the cluster of its first matching neighbor. If the event description does not match any of its top $N$ neighbors, it forms its own cluster. Finally, the embedding vector of each summary is stored in $D$ to facilitate grouping with similar subsequent event descriptions.

# Cluster and Sentence Selection

Each cluster represents an event related to the topic. To generate a timeline with $l$ events, we select the best $l$ clusters and summarize the event descriptions within each cluster into $k$ sentences as specified by the user. We employ a heuristic to choose the top l clusters based on size, with the intuition that more significant events are covered by more articles, especially in the news domain. Subsequently, we apply TextRank (Mihalcea and Tarau 2004) to select the best $k$ sentences within each cluster.

# Baseline Method

A straightforward approach to perform constrained timeline summarization using an LLM is to concatenate the articles into the prompt and directly ask the model to produce a constrained timeline. However, LLMs have a limited context window size, so it is not always possible to fit the entire article pool within the input. To address this limitation, the baseline method involves randomly sampling articles and incrementally adding them to the input prompt one by one until the input context size limit is reached (taking into account the space needed for the instruction prompt and the output). Next, an instruction is added to the prompt, asking the model to generate a timeline comprising $l$ events, each described with a date and a $k$ -sentence summary that adheres to the constraint $c$ . The model then generates the timeline token-by-token until it either determines that it should stop or reaches the token limit.

Table 4: Score comparison of the baseline method, our method (REACTS), our method without self-reflection (REACTS w/o SR) using Llama 3.1 8B (L3.1-8B), Llama 3.1 70B (L3.1-70B), and GPT-4o on our dataset (CREST). We evaluate the models on precision (P), recall (R), and F1 scores using alignment-based ROUGE-1 (AR-1), alignment-based ROUGE-2 (AR-2), and date F1-score metrics. The best scores for each experiment setting are boldfaced.   

<html><body><table><tr><td rowspan="2">Model</td><td rowspan="2">LLM</td><td colspan="3">AR-1</td><td colspan="3">AR-2</td><td colspan="3">Date F1</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td colspan="10">All events</td></tr><tr><td>REACTS w/o SR</td><td>L3.1-8B</td><td>0.0580</td><td>0.0541</td><td>0.0561</td><td>0.0262</td><td>0.0264</td><td>0.0263</td><td>0.1399</td><td>0.1391</td><td>0.1395</td></tr><tr><td>REACTS</td><td>L3.1-8B</td><td>0.0859</td><td>0.0695</td><td>0.0768</td><td>0.0381</td><td>0.0326</td><td>0.0351</td><td>0.1809</td><td>0.1710</td><td>0.1758</td></tr><tr><td>REACTS w/o SR</td><td>L3.1-70B</td><td>0.0701</td><td>0.0957</td><td>0.0809</td><td>0.0326</td><td>0.0496</td><td>0.0393</td><td>0.1773</td><td>0.1773</td><td>0.1773</td></tr><tr><td>REACTS</td><td>L3.1-70B</td><td>0.0970</td><td>0.1246</td><td>0.1091</td><td>0.0434</td><td>0.0592</td><td>0.0501</td><td>0.2425</td><td>0.2396</td><td>0.2411</td></tr><tr><td colspan="10">Filtered events</td></tr><tr><td>BASELINE</td><td>L3.1-8B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.0731</td></tr><tr><td>REACTS w/o SR</td><td>L3.1-8B</td><td>0.0387 0.0769</td><td>0.0180</td><td>0.0246</td><td>0.0042 0.0337</td><td>0.0026 0.0339</td><td>0.0032</td><td>0.1005 0.1749</td><td>0.0574 0.1743</td><td>0.1746</td></tr><tr><td>REACTS</td><td>L3.1-8B</td><td>0.1095</td><td>0.0693 0.0882</td><td>0.0729 0.0977</td><td>0.0497</td><td>0.0427</td><td>0.0338 0.0459</td><td>0.2266</td><td>0.2211</td><td>0.2238</td></tr><tr><td>BASELINE</td><td>L3.1-70B</td><td>0.0687</td><td>0.0939</td><td>0.0793</td><td>0.0324</td><td>0.0480</td><td>0.0387</td><td>0.1341</td><td>0.2524</td><td>0.1751</td></tr><tr><td>REACTS w/o SR</td><td>L3.1-70B</td><td>0.0906</td><td>0.1220</td><td>0.1040</td><td>0.0405</td><td>0.0614</td><td>0.0488</td><td>0.2315</td><td>0.2315</td><td>0.2315</td></tr><tr><td>REACTS</td><td>L3.1-70B</td><td>0.1152</td><td>0.1533</td><td>0.1316</td><td>0.0483</td><td>0.0703</td><td>0.0572</td><td>0.2925</td><td>0.2925</td><td>0.2925</td></tr><tr><td colspan="10">Filtered events (10% data)</td></tr><tr><td>BASELINE</td><td>GPT-40</td><td>0.0487</td><td>0.1451</td><td>0.0729</td><td>0.0216</td><td>0.0730</td><td>0.0334</td><td>0.2065</td><td>0.3176</td><td>0.2506</td></tr><tr><td>REACTS</td><td>GPT-40</td><td>0.0652</td><td>0.1386</td><td>0.0887</td><td>0.0281</td><td>0.0752</td><td>0.0409</td><td>0.3000</td><td>0.3000</td><td>0.3000</td></tr></table></body></html>

# Experiments

We run experiments to investigate whether self-reflection helps in generating more relevant timelines. We evaluate our method on our proposed dataset, both against the ground-truth timelines with all events and ground-truth timelines with filtered events. We employ Llama-3.1 $8 \mathbf { B } ^ { 6 }$ (Llama Team 2024), Llama- $. 3 . 1 7 0 \mathrm { B }$ , and GPT-4o (OpenAI 2024) as the LLMs for our proposed method and the baseline method. However, we only evaluate models with GPT-4o on $10 \%$ of the test set due to cost consideration. In all experiments, we set the generation temperature of the LLMs to zero to make the results reproducible.

As previously explained, the baseline method is inherently limited by the maximum context length of the LLM. Therefore, it can only consider a limited number of articles when generating a timeline. To evaluate the best possible performance of the baseline method, we also conduct additional experiments with an oracle article retriever. From the article pool, the oracle retriever retrieves only articles relevant to the events in the unconstrained ground-truth timeline. Even though the oracle retriever helps to filter out noisy articles, the models still need to determine whether the events in the articles adhere to the constraints. The set of articles kept by the oracle retriever is then randomly sampled to fit the context length of the baseline method and used as the final article pool for all methods. That is, in this experiment, all methods receive the exact same set of input articles to generate the timeline summary. We use GPT-4o as the oracle retriever; therefore, we do not use it as the backbone LLM for the methods.

# Evaluation

We evaluate the experiments with the standard metrics for the timeline summarization task, which are alignment-based ROUGE F1-score (Martschat and Markert 2017) and date F1-score (Martschat and Markert 2018). We employ an approximate randomization test (Riezler and Maxwell 2005; Chinchor, Hirschman, and Lewis 1993) with 100 trials and a $p$ -value of 0.05 to measure statistical significance.

Alignment-Based ROUGE F1-score The alignmentbased ROUGE F1-score measures the text overlap of the event descriptions between the predicted timeline and the ground-truth timeline. It first aligns the events in the predicted timeline with events in the ground-truth timeline based on the closeness of the dates and the similarity of the event descriptions. Following (Ghalandari and Ifrim 2020), we use the alignment setting that allows many-to-one alignment.

For each pair of aligned predicted event and ground-truth event, the metric7 measures the n-gram similarity between the event descriptions. Precision is proportional to the ratio of the overlap compared to the predicted event description, while recall is proportional to the ratio of the overlap compared to the ground-truth event description.

Date F1-Score The date F1-score simply measures the F1 score of the dates covered in the ground-truth timeline against the dates in the predicted timeline. Unlike alignmentbased ROUGE, date F1-score performs hard matching of the dates and does not consider the event descriptions.

Table 5: Results of the experiments with oracle retriever of the baseline method, our method (REACTS), our method without self-reflection (REACTS w/o SR) using Llama 3.1 8B (L3.1-8B) and Llama 3.1 70B (L3.1-70B). We evaluate the models on precision (P), recall (R), and F1 scores using alignment-based ROUGE-1 (AR-1), alignment-based ROUGE-2 (AR-2), and date F1-score metrics. The best scores for each experiment setting are boldfaced.   

<html><body><table><tr><td rowspan="2">Model</td><td rowspan="2">LLM</td><td colspan="3">AR-1</td><td colspan="3">AR-2</td><td colspan="3">Date F1</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td colspan="10">With oracle retriever on filtered events</td></tr><tr><td>BASELINE</td><td>L3.1-8B</td><td>0.0761</td><td>0.0829</td><td>0.0794</td><td>0.0299</td><td>0.0310</td><td>0.0305</td><td>0.2160</td><td>0.2935</td><td>0.2489</td></tr><tr><td>REACTSw/o SR</td><td>L3.1-8B</td><td>0.0966</td><td>0.0800</td><td>0.0875</td><td>0.0399</td><td>0.0341</td><td>0.0367</td><td>0.2172</td><td>0.2151</td><td>0.2161</td></tr><tr><td>REACTS</td><td>L3.1-8B</td><td>0.1505</td><td>0.1102</td><td>0.1272</td><td>0.0687</td><td>0.0490</td><td>0.0572</td><td>0.2916</td><td>0.2766</td><td>0.2839</td></tr><tr><td>BASELINE</td><td>L3.1-70B</td><td>0.1149</td><td>0.1764</td><td>0.1392</td><td>0.0429</td><td>0.0767</td><td>0.0550</td><td>0.2539</td><td>0.4811</td><td>0.3324</td></tr><tr><td>REACTS w/o SR</td><td>L3.1-70B</td><td>0.1142</td><td>0.1488</td><td>0.1292</td><td>0.0482</td><td>0.0655</td><td>0.0555</td><td>0.3165</td><td>0.3132</td><td>0.3148</td></tr><tr><td>REACTS</td><td>L3.1-70B</td><td>0.1810</td><td>0.2213</td><td>0.1991</td><td>0.0735</td><td>0.0959</td><td>0.0832</td><td>0.4769</td><td>0.4485</td><td>0.4622</td></tr></table></body></html>

# Results

We present our main experimental results in Table 4. Our findings indicate that our method significantly outperforms the baseline. The baseline method using Llama struggles to produce a coherent timeline. It often fails to determine when to stop and occasionally generates nonsensical outputs, especially with Llama-3.1 8B. Even when we use GPT-4o, our method still achieves better F1 scores than the baseline. However, note that GPT-4o may have a slight advantage over the other LLMs, as it was used in the dataset creation process.

We also observe that self-reflection significantly improves all scores (i.e., precision, recall, and F1) across all metrics (i.e., AR-1, AR-2, and date F1) in all experimental settings with Llama-3.1. With Llama-3.1 70B, when evaluated against ground-truth timelines without event filtering (all events), self-reflection improves the AR-1 F1 score by $2 . 8 2 \%$ , the AR-2 F1 score by $1 . 0 8 \%$ , and the date F1 score by $6 . 3 8 \%$ . When evaluated against filtered groundtruth timelines, self-reflection improves the AR-1 F1 score by $2 . 7 6 \%$ , the AR-2 F1 score by $0 . 8 4 \%$ , and the date F1 score by $6 . 1 0 \%$ .

With the oracle retriever (Table 5), using Llama-3.1 8B, our method still significantly outperforms the baseline by $4 . 7 8 \%$ , $2 . 6 7 \%$ , and $3 . 5 0 \%$ on AR-1 F1, AR-2 F1, and date F1 respectively. The score improvements are even greater with the larger Llama-3.1 70B model, reaching $5 . 9 9 \%$ , $2 . 8 2 \%$ , and $1 2 . 9 8 \%$ on AR-1 F1, AR-2 F1, and date F1 respectively.

The baseline method is impractical for real-world applications where hundreds of thousands of news articles are published each month. Regardless of the context window size, it cannot keep up with the speed and volume of information flowing through the internet. We have shown that even with the same set of articles (in the oracle retriever setup), our method is superior. Furthermore, the baseline method is unsuitable for online (streaming) processing. Every time a new article is added, previous articles need to be reprocessed to update the timeline, leading to significant computational inefficiency.

# Conclusion

In this paper, we propose a new task with high relevance to current needs, called constrained timeline summarization. We present a new test set for the task (CREST), which was built by generating the constraints using human-in-the-loop collaboration with an LLM, hiring annotators to annotate the adherence of the events in the ground-truth timeline to the constraints, and filtering the events without supporting articles by utilizing an LLM.

We also propose an effective method that utilizes LLMs for the task. Our method does not require any training and can work with different LLMs. Our method works by summarizing the articles according to the constraint, employing a self-reflection procedure to filter out irrelevant summaries, clustering the summaries that describe the same event, and selecting the top $l$ clusters and the top $k$ sentences for each cluster to generate the constrained timeline.

Lastly, we demonstrate the effectiveness of our method by comparing it against a baseline method that generates the timeline directly by concatenating all the articles into its input prompt. We show that our method successfully outperforms the baseline on all metrics. Similarly, we demonstrate the effectiveness of self-reflection by comparing our method to a variant of our method that does not employ selfreflection, and show that self-reflection effectively improves the F1 scores on all metrics. With this work, we hope that constrained timeline summarization can gain more attention and more progress can be achieved on this task in future.