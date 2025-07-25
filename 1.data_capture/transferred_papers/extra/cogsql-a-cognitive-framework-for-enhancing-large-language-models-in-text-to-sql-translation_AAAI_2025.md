# COGSQL: A Cognitive Framework for Enhancing Large Language Models in Text-to-SQL Translation

Hongwei Yuan1, 2, Xiu Tang1, 2, Ke Chen1, 2, Lidan Shou1, 2, Gang Chen1, 2, Huan Li1, 2\*

1The State Key Laboratory of Blockchain and Data Security, Zhejiang University 2Hangzhou High-Tech Zone (Binjiang) Institute of Blockchain and Data Security yhw.cs, lihuan.cs@zju.edu.cn

# Abstract

Large language models (LLMs) have significantly advanced the performance of various natural language processing tasks, including text-to-SQL. Current LLM-based text-toSQL schemes mainly focus on improving the understanding of natural language questions (NLQs) or refining the quality of generated SQLs. While these strategies are effective, they often address specific, nuanced aspects. In contrast, humans approach text-to-SQL with a holistic view, applying transitional logical reasoning across multiple steps to arrive at the final answer. We believe LLMs can leverage human cognitive processes to achieve greater accuracy in text-to-SQL. In this paper, we present COGSQL, a framework featuring a suite of tailored models and strategies aimed at replicating human cognitive processes for enhanced LLM-based text-to-SQL. COGSQL consists of three key modules: (1) SQL preparation: we employ a coarse-to-fine schema linking and syntax keyword prediction, akin to how human recall and align key concepts for better understanding. (2) SQL generation: we introduce a concept-enhanced chain-of-thought prompting, enhancing NLQ interpretation and SQL composition of LLMs, similar to humans drafting SQL query. (3) SQL correction: we develop NLQ consistency and result consistency techniques to correct various errors, mirroring how humans evaluate and refine reasoning. We conduct extensive experiments using diverse benchmarks and LLMs. The results and analysis verify the effectiveness and generalizability of COGSQL.

Extended version — https://github.com/Yhw109/COGSQL

# Introduction

Relational databases store large amounts of data, but Structured Query Language (SQL), the tool used to retrieve data from these databases, has traditionally been a skill limited to expert users. Translating natural language questions (NLQs) into SQL queries, known as text-to-SQL, has been a research focus for years (Zelle and Mooney 1996; Saha et al. 2016; Yu et al. 2018). This is crucial for creating more userfriendly database interfaces.

Recently, phenomenal large language models (LLMs) have excelled in various tasks such as question answering (Nguyen et al. 2023) and math reasoning (Zhang et al.

NLQ In the branch with the second-highest crime rate in 1995,   
how many male clients are there?   
Evidence Male refers to gender $\mathbf { \Omega } = \mathbf { \dot { \Omega } } M ^ { \prime }$   
Database Schema Table client (client_id, …) , Table district   
(district_id, …) Relevant schemas and keywords Needed concepts? Draft a SQL Grammars First? & original To answer the query intent   
NLQ with SQL DNeLcQoamnpdoasenatlhyeze Examine the   
query, I will … the structure drafted?

2023b). They have also significantly improved the capabilities of text-to-SQL tasks. Directly instructing LLMs, whether open-source or proprietary, has become a favorable paradigm. This approach avoids costly fine-tuning, easily transitions between LLM products, and benefits from their performance upgrades. When we mention LLM-based textto-SQL, it refers to this inference-only approach. Current studies in this area aim to improve understanding of NLQs or enhance the quality of generated SQLs. For example, (Pourreza and Rafiei 2024; Xie et al. 2024) use task decomposition to enhance LLMs’ attention towards simple subtasks; (Gao et al. 2024; Ren et al. 2024) employ retrieval-based few-shot demonstrations to elicit LLMs’ in-context learning ability; (Qu et al. 2024) propose task alignment to mitigate LLMs’ hallucinations in SQL generation; and (Wang et al. 2023a) propose multi-agent collaboration to jointly enhance text-to-SQL quality. While these strategies improve performance, they often concentrate on specific, nuanced aspects. In contrast, humans approach text-to-SQL with a holistic view, using reasoning across various aspects to achieve accurate and dependable translations. This cognitive paradigm involves making connections and transitions across different steps to ensure precision and reliability.

Figure 1 depicts how a human applies transitional logical thinking to transform text into SQLs in an integrated manner.1 When presented with the NLQ, a person begins by identifying essential concepts, focusing on crucial parts of problem-solving, like relevant database schemas and syntax keywords. Next, they draft the SQL using these key concepts. Usually, before writing down the SQL, they analyze its structure via decomposition, paying attention to detailed SQL clauses and subqueries, and then compose the SQL. Finally, they review the drafted SQL, considering aspects like grammar and alignment with the original NLQ intent, making necessary corrections.

This thus leads us to a critical question: Can LLMs benefit from humans’ cognitive process to improve text-to-SQL accuracy? We believe the answer is yes. In this paper, we present COGSQL, a framework designed to imitate human COGnitive process to enhance LLMs in text-to-SQL. Based on the stages in Figure 1, COGSQL is divided into three main modules: (i) SQL PREPARATION, which identifies relevant database schemas and syntax keywords, akin to recalling key concepts; (ii) SQL GENERATION, which enables LLMs to interpret NLQs like humans and construct SQL precisely via concept-enhanced chain-of-thought (CoT) prompting; (iii) SQL CORRECTION, which instructs LLMs to reflect on the generated SQL, correcting grammar errors and aligning with NLQ intent. Each module includes tailored models or strategies to ensure LLMs deliver highquality responses by following this cognitive process.

We perform experiments on five large challenging crossdomain text-to-SQL datasets with different LLMs, including both robust proprietary and fine-tuned open-source models. Results verify that COGSQL effectively enhances the performance of LLM-based text-to-SQL and generalizes well across various LLMs. Our main contributions include

1. We propose a novel text-to-SQL framework that aims to holistically enhance LLM’s text-to-SQL abilities by imitating human cognitive process.   
2. We examine our framework on five cross-domain text-toSQL benchmarks with various LLMs, showing its effectiveness and generalization across datasets and models.

# Related Work

Text-to-SQL Progress. Translating NLQs into SQLs has long been a focus in database and NLP research. Early approaches (Zelle and Mooney 1996; Simitsis, Koutrika, and Ioannidis 2008; Li and Jagadish 2014) relied on labor-intensive, manually crafted rules. The advent of neural networks has popularized the encoder-decoder architecture (Li and Jagadish 2014; Saha et al. 2016; Wang et al. 2020), where the encoder processes the NLQ and database schema, and the decoder generates the SQL query. Transformer-based architectures and pre-trained language models (Scholak, Schucher, and Bahdanau 2021; Li et al. 2023a,b) have further improved text-to-SQL performance. Recently, LLMs have demonstrated exceptional capabilities, with inference-only approaches (Dong et al. 2023; Wang

et al. 2023a; Gao et al. 2024; Qu et al. 2024; Xie et al. 2024;   
Ren et al. 2024) offering superior transferability.

SQL Preparation. The LLM’s generation process is enhanced by gathering critical information and performing NLQ-schema linking. Schema linking can be coarse-grained using cross-encoders (Li et al. 2023a, 2024b) or finegrained through demonstrations and instructions (Pourreza and Rafiei 2024; Wang et al. 2023a; Dong et al. 2023), improving translation accuracy. Additionally, demonstration selection, which optimizes the LLM’s in-context learning by choosing NLQ-SQL examples, is crucial. While retrievalbased selection (Chang and Fosler-Lussier 2023; Gao et al. 2024; Pourreza and Rafiei 2024; Ren et al. 2024) is common but computationally expensive, COGSQL streamlines this process simply by using a fixed two-shot demonstration in prompts.

SQL Generation. SQL generation involves complex reasoning that builds on key information from SQL preparation. LLMs excel in these tasks when effectively prompted. Techniques like chain-of-thought (CoT) (Wei et al. 2024; Kojima et al. 2024), decomposition and planning (Zhou et al. 2023; Wang et al. 2023b), and tool incorporation (Gao et al. 2023; Chen et al. 2024) greatly enhance LLMs’ generation. Studies (Zhang et al. 2023a; Tai et al. 2023; Pourreza and Rafiei 2024) have applied CoT specialized to text-to-SQL. For instance, (Pourreza and Rafiei 2024) use a two-stage process where the LLM assesses question difficulty and applies CoT for complex queries, while (Zhang et al. 2023a; Tai et al. 2023) perform schema mapping before generating SQL. Differently, COGSQL integrates NLQ interpretation and SQL structure analysis for in-depth reasoning.

SQL Correction. Error correction for generated SQLs is achieved through techniques like self-consistency (Dong et al. 2023; Gao et al. 2024; Ren et al. 2024), selfcorrection (Pourreza and Rafiei 2024), and verification via execution results (Ni et al. 2023) to ensure reliability. However, inappropriate self-correction can reduce accuracy (Pourreza and Rafiei 2024), and self-consistency can be computationally expensive due to repeated LLM inference. COGSQL mitigates these challenges by implementing a multi-view correction, which aims to align with both NLQ and execution results.

# Preliminaries

Definition. Let an NLQ be $\mathcal { Q } = \left\{ q _ { 1 } , \ldots , q _ { | \mathcal { Q } | } \right\}$ , where each $q _ { i }$ is a word token, and a corresponding database schema be $\bar { \mathcal { D } } = \langle \mathcal { T } , \mathcal { C } , \mathcal { R } \rangle$ , where:

• $\mathcal { T } = \{ t _ { 1 } , \ldots , t _ { n } \}$ , with each $t _ { i }$ representing a table in the database. We use $\left| t _ { i } \right|$ to denote $t _ { i }$ ’s column count; • $\mathcal { C } = \{ c _ { 1 } ^ { 1 } , \ldots , c _ { 1 } ^ { | t _ { 1 } | } , c _ { 2 } ^ { 1 } , \ldots , c _ { 2 } ^ { | t _ { 2 } | } , \cdots , c _ { n } ^ { 1 } , \cdots , c _ { n } ^ { | t _ { n } | } \}$ , c|ntn| , with each $c _ { i } ^ { j }$ representing the $j$ -th column in the $i$ -th table; • $\mathcal { R } = \{ ( c _ { k } ^ { i } , c _ { h } ^ { j } ) \in \mathcal { C } ^ { 2 } \}$ , where each pair $( c _ { k } ^ { i } , c _ { h } ^ { j } )$ denotes a foreign key relationship between columns $c _ { k } ^ { i }$ and $c _ { h } ^ { j }$ .

The goal of text-to- $s Q L$ is to convert $\mathcal { Q }$ into an executable SQL statement on $\mathcal { D }$ to retrieve the desired results.

Key Concepts for Solving Text-to-SQL. Mimicking human cognitive processes in problem-solving involves recalling key concepts to clarify the problem-solving direction and arrive at the correct answer. For prompting LLMs to generate SQL statements from NLQs, key concepts include: 1) database items related to the NLQ, such as tables, columns, and values; 2) syntax keywords needed to construct SQL, including clauses, functions, and operators. We categorize syntax keywords in Table 1 based on their practical applications. Not all SQL syntax keywords are enumerated here; we exclude those not present in our training datasets. This focus ensures our prediction is tailored to SQL generation while allowing for the integration of new keywords as needed.

Table 1: Categorization of syntax keywords to utilize.   

<html><body><table><tr><td>Category Level</td><td>Syntax Keywords</td></tr><tr><td>Clause</td><td>SELECT,FROM,WHERE, ORDER BY,GROUP BY,HAVING</td></tr><tr><td>Aggregation</td><td>COUNT,SUM,AVG,MIN,MA</td></tr><tr><td>String</td><td>SUBSTR</td></tr><tr><td>Date</td><td>STRFTIME</td></tr><tr><td>Conditional</td><td>CASE WHEN,IIF</td></tr><tr><td>null-related</td><td>IS NULL,IS NOT NULL</td></tr><tr><td>Comparison</td><td>IN,BETWEEN,LIKE</td></tr><tr><td></td><td></td></tr><tr><td>Arithmetic Other keywords</td><td>Division (‘/) DISTINCT,CAST</td></tr></table></body></html>

NLQ Database Schema · DB check   
(I) Key Concept Cross-encoder for 8 Recalling Coarse-grained d augmented Schema Linking 用 training data LLM for 1 Cross-encoder for Fine-grained 8 Syntax Keyword Schema Linking 中   
(II) Concept-enhancedCoT Prompting 天 Filtered Schema SPyrnetadxicKtieoynword ④ QLuLeMs fiorn LLM for SQL Interpretation 0 ③ Composition ? ⑤ ⑥   
(III) Consistency- based Correction LLNML fQor SQL Grmmar Execution Reformulation Rules Checks SQL

# Methodology

# Overview

Figure 2 provides an overview of the proposed COGSQL, which enhances LLM-based text-to-SQL by emulating human cognitive processes through three key modules: (I) Key Concept Recalling, which performs $\textcircled{1}$ coarse-to-fine schema linking and $\textcircled{2}$ syntax keyword prediction to identify filtered schema and syntax keywords from the given NLQ and schema, respectively; (II) Concept-enhanced CoT Prompting, which employs a two-stage prompting method with $\textcircled{3}$ NLQ interpretation and $\textcircled{4}$ SQL composition, allowing LLMs to utilize key concepts to draft intermediate plans before finalizing the answer. (III) Consistency-based Correction, which enables LLMs to re-examine their answers for $\textcircled{5}$ NLQ consistency and $\textcircled{6}$ result consistency and make necessary adjustments.

# Key Concept Recalling

Recalling key concepts initiates human cognitive processes for complex reasoning tasks. Before formulating SQLs, humans typically focus on relevant database items and syntax keywords. Accordingly, we propose coarse-to-fine schema linking and syntax keyword prediction.

Coarse-to-fine Schema Linking. Real-world databases often contain numerous tables and columns in their schema, which can overwhelm an LLM and hinder SQL generation due to length constraints. Schema linking (Pourreza and Rafiei 2024; Li et al. 2023a) aims to identify a subset of the schema. It should be comprehensive to include all necessary database items (i.e., needed tables in $\tau$ and columns in $\mathcal { C }$ ) for the LLM to formulate SQLs and concise to avoid extra tables and columns that can mislead the LLM’s generation.

Current LLM-based text-to-SQL studies primarily use two strategies. One (Qu et al. 2024; Pourreza and Rafiei 2024; Wang et al. 2023a) prompts LLMs directly to select relevant tables and columns, while another treats schema linking as a classification task (Li et al. 2023a, 2024b):

$$
\mathcal { V } ^ { \mathrm { s l } } = f ( \operatorname { E n c } ( \mathcal { Q } , \mathcal { T } , \mathcal { C } ) ; \theta ^ { \mathrm { s l } } ) ,
$$

where an encoder $\operatorname { E n c } ( { \mathord { \cdot } } )$ maps the input sequence into embeddings with Transformer-based architectures, typically RoBERTa (Liu et al. 2019), and a classifier $f ( \cdot ; \theta )$ , parameterized by $\theta$ , outputs probabilities for each database item.

The encoder input consists of a concatenation of the NLQ $\mathcal { Q }$ with flattened table and column names. Embeddings for each table $t _ { i } \in \mathcal { T }$ and column $c _ { j } \in { \mathcal { C } }$ are derived from the final layer hidden state of the encoder. The classifier then produces probabilities for $n$ tables and all $\begin{array} { r } { ( m = \sum _ { i = 1 } ^ { n } | t _ { i } | ) } \end{array}$ columns, represented as $y ^ { \mathrm { s l } } = y _ { 1 } ^ { t } , \ldots , y _ { n } ^ { t } , y _ { 1 } ^ { c } , \ldots , y _ { m } ^ { c } $ where $y _ { i } ^ { t }$ denotes the probability of the $i$ -th table and $y _ { j } ^ { c }$ denotes the probability of the $j$ -th column. At the table level, the top- $\mathbf { \mathcal { k } } _ { 1 }$ tables with the highest probabilities are retained; similarly, for each selected table, the top- $k _ { 2 }$ columns with the highest probabilities are retained.

The first strategy is simple to implement and can produce a minimized schema. However, it risks omitting critical tables and columns, as the LLM may overlook subtle semantic nuances in the NLQ. The second strategy can retain all necessary database items by maintaining a relatively expansive schema (in comparison to the former strategy), but can introduce more noise. In this light, COGSQL proposes a coarse-to-fine schema linking approach that combines their advantages. Following Eq. 1, it first uses a classifier for coarse-grained schema linking to filter out unnecessary tables and columns. Then, it prompts the LLM with the retained schema ( $\mathbf { \mathit { k } } _ { 1 }$ tables and $k _ { 1 } \times k _ { 2 }$ columns) and refines it by selecting a subset of tables (say $k _ { 3 }$ ) out of the $k _ { 1 }$ tables for fine-grained schema linking. To improve the overall efficiency of COGSQL, this selection procedure is integrated with the first stage of concept-enhanced CoT prompting (see Figure 3 for a simplified version of our prompt and LLM’s response). We only perform fine-grained schema linking at the table level to avoid column omission. The filtered schema in COGSQL consists of $k _ { 3 }$ tables, their retained columns, and corresponding foreign keys in $\mathcal { R }$ .

# Syntax Keyword Prediction.

We treat syntax keyword prediction as a classification task similar to the coarse-grained scheme linking in Eq. 1:

$$
\begin{array} { r } { y ^ { \mathrm { k w } } = f ( \operatorname { E n c } ( \mathbb { Z } , \mathcal { Q } , \hat { T } , \hat { \mathcal { C } } ) ; \theta ^ { \mathrm { k w } } ) . } \end{array}
$$

Here, the encoder input concatenates a fixed instruction $\boldsymbol { \mathcal { T } }$ (e.g., “Determine whether the SQL corresponding to the following NLQ need to use ORDER $B Y ^ { * }$ ), $\mathcal { Q }$ , and filtered tables $\mathcal { \hat { T } }$ and columns $\hat { \mathcal { C } }$ from coarse-grained schema linking — this reduces noise compared to using the entire schema.

The [cls] token embedding from the final-layer hidden state of the encoder, is fed into different classifiers of the same structure to obtain the probability corresponding to specific syntax keywords listed in Table 1. A threshold $\tau$ is used to decide whether a syntax keyword is needed for the given NLQ. We perform differentiated predictions for different keywords based on their distribution in the training set. We do not predict ubiquitous keywords like SELECT, FROM, and WHERE. Dedicated classifiers are used separately for ORDER BY, GROUP BY, and HAVING. For remaining keywords, classifiers are built per category level (see Table 1); e.g., we predict the need for the aggregation functions rather than directly predicting an individual keyword SUM.

However, the imbalanced distribution of syntax keywords (e.g., the rare appearance of HAVING) can hinder the performance of the classifiers. To mitigate this, text-to-SQL augmentation methods (Yu et al. 2021; Hu et al. 2023) have been proposed, which generate valid SQL queries from extracted SQL templates. However, these methods cannot ensure alignment with the original training distribution.

Inspired by a recent study (Yu et al. 2024), we propose sample-centric augmentation, leveraging LLMs to synthesize high-quality NLQ-SQL pairs from original training samples. For a given training sample $( \mathcal { Q } , S )$ , this method offers two augmentations: 1) NLQ Rewriting: if $s$ contains the keyword to augment, LLMs formulate a new NLQ $\mathcal { Q } ^ { \prime }$ based on $s$ , preserving $\mathcal { Q }$ ’s intent; e.g., when augmenting COUNT, the NLQ in Figure 1 is rewritten as “Considering the branch where the second-highest number of crimes occurred in 1995, how many of its clients are male?”. 2) Keyword Addition: otherwise, LLMs are prompted to add the keyword to augment $s$ and formulate a corresponding NLQ, producing $( \bar { \cal S } ^ { \prime } , \ \mathcal { Q } ^ { \prime } )$ ; e.g., when augmenting ORDER BY, the NLQ in Figure 1 becomes “In the branch where the second-highest number of crimes were committed in 1995 occurred, which male client has a latest birthday?” — latest implies added ORDER BY. These augmentations enrich the keyword distribution of the training set while preserving semantic integrity. Moreover, we adapt a high temperature to enrich generation diversity. We use execution-based checks for correctness: LLMs are prompted to answer $\mathcal { Q } ^ { \prime }$ with $\boldsymbol { S } ^ { \prime \prime }$ ; NLQ rewriting should ensure $s$ and $\boldsymbol { S } ^ { \prime \prime }$ produce identical execution results; keyword addition must ensure $\scriptstyle { S ^ { \prime } }$ and $S ^ { \prime \prime }$ yield identical and non-empty results.

Figure 3: Prompt and LLM response for NLQ interpretation.   

<html><body><table><tr><td>Prompt</td></tr><tr><td>Instruction: Read the question to identify the required data to query and return.Assess if subqueries are needed. Database Schema: Tableclient (client_id,...),Table district(district_id,...) Question: In the branch with the second-highest crime rate in 1995,how manymale clients are there?</td></tr><tr><td>LLM response</td></tr><tr><td>To Be Queried: ['the 2nd-highest number of crimes'] To Return: ['the number of male clients'] Table Needed: ['client',‘district'] Subquery Required: ['yes']</td></tr></table></body></html>

# Concept-enhanced CoT Prompting

To improve SQL generation with LLMs, we address two crucial issues, interpreting NLQ intents and composing SQL queries. We incorporate solutions into a CoT prompting process, mirroring how humans mentally rehearse key concepts before responding formally. This two-stage prompting involves NLQ interpretation that identifies inquiry objectives and desired data and SQL composition that ensures the correct use of identified key concepts.

NLQ Interpretation. Referring to the example NLQ in Figure 1. Upon seeing this NLQ, humans first analyze its content to identify the inquiry objective (“second-highest number of crimes committed in 1995”) and returned data (“number of male clients”). For complex NLQs with nested query (subquery and set operation), such as the NLQ in Figure 1, it is also crucial to decide if such a query is needed. An illustrative prompt that enables such a procedure is provided in Figure 3, which includes a specific instruction to guide LLMs, the schema after coarse-grained schema linking, and the NLQ itself. To ensure the LLM reasons effectively, we provide demonstrations of two NLQs manually annotated2 from the training set.

The LLM response is also exemplified in Figure 3, including the inquiry objective, the returned data, the table needed, and any required subqueries.

SQL Composition. Back to the human cognitive procedure: after interpreting an NLQ, humans identify how key concepts fit into various SQL clauses, such as identifying the filter condition for the WHERE clause. To replicate this procedure, we focus on decomposing and generating the SQL.

Figure 4: Prompt and LLM response for SQL composition.   

<html><body><table><tr><td>Prompt Instruction: Analyze the intent of SQL clauses and</td></tr><tr><td>subqueries,then formulate a SQL query. Database Schema:Table client(client_id,...）,Table district (district_id,...） ... Question: In the branch with the second-highest crime rate in 1995.. Keyword Indicator: The SQL query is likely to need:</td></tr><tr><td>FROM,WHERE,SELECT,COUNT.. LLM response</td></tr><tr><td>Clause Analysis: FROM clause for retrieving ... WHERE clause for filtering .. Subquery Analysis: Subquery for retrieving crime .. SQL: SELECT count(client.client_id) FROM...</td></tr></table></body></html>

We instruct LLM to analyze the SQL clauses identified by our syntax keyword prediction, decompose the NLQ, and generate it based on this analysis. An example prompt is provided in Figure 4. It includes demonstrations, a specific instruction, the database schema after fine-grained schema linking, NLQ, interpretation from the first stage, and a keyword indicator that includes the hints derived from syntax keyword predictions. The keyword indicator has two parts: 1) a clause list shows likely clauses and 2) operator indicator that maps likely operators to text hints. For example, if the classifier predicts an aggregation function category (see Table 1), operator indicator suggests “The SQL is likely to involve at least one aggregation function: COUNT, SUM, AVG, MIN, MAX”. The LLM’s response in Figure 4 involves clause analysis, subquery analysis, and the generated SQL. The above two-stage CoT ensures LLMs thoroughly consider recalled key concepts, effectively using them to gradually compose the desired SQL. This is beneficial for complex NLQs as our experiments show.

# Consistency-based Correction

LLMs are known to experience mistakes and hallucinations (Ji et al. 2023) in generation. Specifically, two common mistakes could hamper LLMs’ text-to-SQL translation: 1) NLQ Misunderstanding: LLMs may misinterpret the intent of an NLQ, leading to a semantically incorrect SQL query. 2) Grammar Mistakes: LLMs might violate SQL grammar rules, resulting in a non-executable query. To address these issues, COGSQL performs a consistency-based correction, incorporating NLQ consistency that ensures the SQL’s alignment with the intended NLQ, and result consistency that ensures the SQL complies with grammatical rules.

NLQ Consistency. Some NLQs contain subtle semantic details. Ignoring these can result in LLMs’ misunderstanding. For example, the NLQ in Figure 1 might ask for the number of male clients, but an LLM might incorrectly return both the count and their personal information. COGSQL performs

Figure 5: Prompt and LLM response for NLQ-consistency.   

<html><body><table><tr><td>Prompt</td></tr><tr><td>Instruction: Formulate a new question to verify the correctness of SQL query. Database Schema: Table client (client_id,...).. Question: In the branch with the second-highest crime rate in ...</td></tr><tr><td>SQL:SELECT count(client.client_id) FROM . LLM response</td></tr><tr><td>New NLQ: How many male clients are in the ... Consistency with given NLQ: Yes SQL: SELECT count(client.client_id) FROM ...</td></tr></table></body></html>

NLQ-consistency correction to address this issue. Examples of a simplified version of our prompt and LLM’s response are in Figure 5. We instruct LLMs to formulate a new NLQ based on the schema and generated $\mathtt { S Q L }$ , and then verify if this new NLQ is consistent with the original one. If they match, the LLM has correctly understood the original NLQ. If not, the SQL query likely needs revision. This revision process is formulated as:

$$
\hat { \mathcal { Q } } , \mathcal { C } , \hat { \mathcal { S } } = f _ { \mathcal { M } } ( \mathbb { Z } , \mathcal { D } , \mathcal { Q } \mid \mathcal { S } ) ,
$$

where $f _ { \mathcal { M } } \left( \cdot \vert \cdot \right)$ is a mapping function applied by the LLM $\mathcal { M }$ , $\boldsymbol { \mathcal { T } }$ is the instruction, $\mathcal { D }$ is the schema, $\mathcal { Q }$ is the original NLQ, and $s$ is the generated SQL before correction. $\hat { \mathcal { Q } }$ is the new NLQ formulated for $s , c$ is the consistency check result, and $\hat { \boldsymbol { S } }$ is the SQL after potential correction. This correction is done without needing real-world database access, and we apply it to all generated SQLs.

Result Consistency. SQL language has complex grammar rules, and violating these can result in non-executable SQL queries. COGSQL performs result-consistency correction to ensure that generated SQL queries are grammatically correct. This correction consists of two aspects. First, we establish correction rules (e.g., if a column name in the SQL query contains spaces, it must be enclosed in double quotes), each generated SQL query undergoes a grammar check to ensure compliance with the established rules. The rule set is easily to be updated and expanded.

The SQL queries after rule-based correction are executed in databases. Non-executable ones will trigger exceptions during execution. We prompt LLMs to correct these nonexecutable SQL queries using the corresponding exceptions. This process can be formally expressed as:

$$
\hat { \mathcal { S } } = f _ { \mathcal { M } } ( \mathcal { T } , \mathcal { D } , \mathcal { Q } \mid \mathcal { S } , \mathcal { E } ) ,
$$

where $f _ { \mathcal { M } } \left( \cdot \vert \cdot \right)$ is the LLM-enabled mapping function, $\boldsymbol { \mathcal { T } }$ the instruction, $\mathcal { D }$ the schema, $\mathcal { Q }$ the original NLQ, $s$ the SQL query generated by $\mathcal { M }$ ; in addition, $\mathcal { E }$ is the exception encountered when SQL query executes in the database, $\hat { \cal S }$ is the SQL after correction. This correction is applied only to each non-executable SQL.

# Experiment

We address the following research questions through comprehensive experiments:

• RQ1. How does COGSQL’s performance compare to existing methods across various LLMs and datasets? • RQ2. What are the contributions of each COGSQL component to the final text-to-SQL translation accuracy? • RQ3. How does COGSQL mitigate the mistakes that LLMs are prone to make in text-to-SQL translation?

# Experimental Setup

Datasets. We evaluate COGSQL using the well-recognized text-to-SQL benchmarks: (1) Spider (Yu et al. 2018). (2) Spider’s variants (Li et al. 2023a), Spider-DK, SpiderRealistic, Spider-Syn. (3) BIRD (Li et al. 2024c).

Baselines. We compare COGSQL to a wide range of stateof-the-art methods, including RESDSQL (Li et al. 2023a), CODES (Li et al. 2024b), DIN-SQL (Pourreza and Rafiei 2024), MAC-SQL (Wang et al. 2023a), TA-SQL (Qu et al. 2024), DAIL-SQL (Gao et al. 2024), DEA-SQL (Xie et al. 2024), and SUPER-SQL (Li et al. 2024a). Among these, RESDSQL and CODES use meticulously fine-tuned T5 (Raffel et al. 2020) and STARCODER (Li et al. 2023c) as their base models. Other baselines similar to COGSQL do not involve fine-tuning the base language model.

Metrics. Following previous studies (Qu et al. 2024; Xie et al. 2024), we use Execution Accuracy $( E X )$ and Valid Efficiency Score (VES) as performance metrics. (1) EX checks if the predicted SQL produces the same execution results as the ground truth SQL on the database. (2) VES, the specialized metric from BIRD, measures efficiency of the correctly predicted SQL. The measurement unit for both metrics is $\%$ , and higher measures indicate better effectiveness/efficiency. Implementation. We use a temperature of 0.7 for samplecentric augmentation, and 0 for other modules of COGSQL. Each sample in Spider and BIRD training sets is augmented once, resulting in training with 1,390 (resp. 1,441) samples from NLQ rewriting and 2,665 (resp. 2,833) samples from keyword addition. Two-shot prompts are used for augmentation, concept-enhanced CoT, and NLQ-consistency correction modules. We use (Li et al. 2024b)’s checkpoint for coarse-grained schema linking. The encoder and classifiers for syntax keyword prediction are trained on an A6000 GPU with a learning rate of 1e-6, using LoRA with rank $= 1 2 8$ . Based on analyzing the training set of two benchmarks, we use $k _ { 1 } = 5$ and $k _ { 2 } = 7$ for coarse-grained schema linking.

We evaluate COGSQL on both proprietary and opensource LLMs. For proprietary LLMs, we use the GPT series: GPT4-0613, GPT- $4 0 ^ { - } 2 0 2 4 - 0 5 ^ { } - 1 3$ , and GPT-4omini-2024-07-18. For open-source, we use DEEPSEEKCODER-V2-0724 (DeepSeek-AI et al. 2024).

# Overall Performance (RQ1)

Baseline Comparison. Referring to Table 2, COGSQL enhances the GPT4 baseline effectively on both benchmarks. Even with modest models like GPT-4o-mini, COGSQL remains competitive, surpassing strong baselines like DINSQL and DAIL-SQL using GPT4. When equipped with stronger GPT-4o and GPT4, COGSQL outperforms extensively fine-tuned baselines RESDSQL and CODES. While COGSQL with GPT4 slightly lags behind SUPER-SQL and DEA-SQL on Spider test, those methods rely on welldesigned criteria and exhaustive retrieval-based demonstration selection.

Note that Spider is more conventional, allowing incontext learning to effectively incorporate SQL knowledge via prompts. In contrast, the more challenging BIRD reduces the efficacy of demonstrations due to the complexity of NLQs, with both SUPER-SQL and DEA-SQL performing worse than COGSQL. This indicates that a holistic cognitive framework better enhances LLMs in complex scenarios. Nonetheless, COGSQL consistently ranks among the top two performers across all cases.

Moreover, BIRD’s VES results also highlight COGSQL’s ability to generate efficient SQL queries.

Table 2: Overall comparison: Grey text indicates the methods extensively fine-tuned on LLMs. The best performance is bolded and the second-best is underlined.   

<html><body><table><tr><td rowspan="2">Methods</td><td colspan="2">Spider</td><td colspan="2">BIRD Dev</td></tr><tr><td>Dev EX</td><td>Test EX</td><td>EX</td><td>VES</td></tr><tr><td>RESDSQL-3B+NatSQL</td><td>84.10</td><td>79.90</td><td></td><td></td></tr><tr><td>SFT CODES-7B</td><td>85.40</td><td></td><td>57.17</td><td>58.80</td></tr><tr><td>SFT CoDES-15B</td><td>84.90</td><td></td><td>58.47</td><td>59.87</td></tr><tr><td>GPT4</td><td>72.30</td><td>/</td><td>46.35</td><td>49.77</td></tr><tr><td>DIN-SQL+GPT4</td><td>83.50</td><td>85.30</td><td>50.72</td><td>58.79</td></tr><tr><td>MAC-SQL+GPT4</td><td>78.60</td><td>82.80</td><td>57.56</td><td>58.76</td></tr><tr><td>DAIL-SQL +GPT4</td><td>83.10</td><td>86.20</td><td>54.76</td><td>56.08</td></tr><tr><td>DEA-SQL +GPT4</td><td>85.40</td><td>87.10</td><td>58.93</td><td>63.07</td></tr><tr><td>TA-SQL +GPT4</td><td>85.00</td><td></td><td>56.19</td><td></td></tr><tr><td>SUPER-SQL +GPT4</td><td>87.00</td><td>/ 一</td><td>58.50</td><td>/ 61.99</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>COGSQL+GPT4</td><td>85.40</td><td>86.40</td><td>59.58</td><td>64.30</td></tr><tr><td>COGSQL + GPT-40</td><td>84.70</td><td>85.80</td><td>59.19</td><td>63.99</td></tr><tr><td>CoGSQL + GPT-4o-mini</td><td>84.20</td><td>83.80</td><td>56.26</td><td>61.31</td></tr></table></body></html>

Spider Variants. Table 3 presents COGSQL’s accuracies on three Spider variants that better reflect real-world applications. We compare two closest competitors using GPT4, TA-SQL and DEA-SQL. COGSQL effectively enhances both the GPT-4 and GPT-4o-mini baselines across all benchmarks, demonstrating impressive performance when NLQs are more obscure and require domain knowledge. Moreover, the results indicate that COGSQL with GPT-4o-mini competes well with other strong baselines using GPT4, further demonstrating its robustness and effectiveness. We presume that the NLQ interpretation in concept-enhanced CoT prompting allows LLMs to achieve robust understanding, even when NLQs are unclear. This further proves the importance of equipping LLMs with a human-akin mindset to interpret NLQs and compose SQLs.

Various LLMs. We use the more complex BIRD dev set to examine COGSQL’s generalizability across various LLMs. The NLQs of BIRD are manually annotated by the benchmark authors to reflect different difficulty levels. Referring to Table 4, COGSQL significantly boosts DEEPSEEK performance across all difficulty levels, indicating strong generalizability across models. We also present the improvement that COGSQL brings to GPT-4o-mini, with the most significant gains in the ’challenging’ category, followed by ’moderate’. This demonstrates that COGSQL’s effectiveness in enhancing LLMs’ ability to reason through complex NLQs.

Table 3: EX $( \% )$ results on three Spider variants.   

<html><body><table><tr><td>Methods</td><td>DK</td><td>Syn</td><td>Realistic</td></tr><tr><td>GPT4</td><td>65.2</td><td>71.1</td><td>73.4</td></tr><tr><td>TA-SQL +GPT4</td><td>72.9</td><td>/</td><td>79.5</td></tr><tr><td>DEA-SQL + GPT4</td><td></td><td>/</td><td>81.5</td></tr><tr><td>GPT-4o-mini</td><td>69.9</td><td>69.8</td><td>78.7</td></tr><tr><td>GPT-4o-mini + CoGSQL</td><td>73.8 (↑ 3.9)</td><td>76.0 (↑6.2)</td><td>81.5 (↑2.8)</td></tr><tr><td>GPT-4+ CoGSQL</td><td>76.6 (↑ 11.4)</td><td>76.5 (↑ 5.4)</td><td>85 (↑11.6)</td></tr></table></body></html>

<html><body><table><tr><td></td><td colspan="4">BIRD Dev</td></tr><tr><td>Model</td><td>Simple</td><td>Moderate</td><td>Challenging</td><td>Total</td></tr><tr><td rowspan="2">DEEPSEEK + COGSQL</td><td>61.62</td><td>44.73</td><td>29.17</td><td>53.46</td></tr><tr><td>64.11 (↑2.49)</td><td>46.67 (↑1.94)</td><td>39.58 (↑ 10.41)</td><td>56.52 (↑ 3.06)</td></tr><tr><td rowspan="2">GPT-4o-mini + COGSQL</td><td>58.27</td><td>38.92</td><td>28.47</td><td>49.61</td></tr><tr><td>63.24 (↑ 4.97)</td><td>47.31 (↑8.39)</td><td>40.28 (↑11.81)</td><td>56.26 (↑6.65)</td></tr></table></body></html>

Table 4: EX $( \% )$ results on both proprietary and open-source LLMs; $( \uparrow )$ indicates the performance gain.

# Ablation Study (RQ2)

Module Design. Referring to Table 5, every module contributes to overall performance, with any removal causing declines. On Spider, COGSQL exhibits robustness, maintaining stable performance even when individual modules are removed, except for the coarse-to-fine schema linking module. Its absence causes the most significant drop, showing the importance of accurate schema linking. The NLQ consistency and result consistency modules are equally effective for Spider. For more complex BIRD, the conceptenhanced CoT prompting and result consistency modules are vital. Removing them leads to the largest performance drops, underscoring their importance in handling BIRD’s complex schema and reasoning demands.

Table 5: Ablations of COGSQL modules using EX $( \% )$ .   

<html><body><table><tr><td>Methods</td><td>Spider</td><td>Bird</td></tr><tr><td>COGSQL + GPT-4o-mini</td><td>84.20</td><td>56.26</td></tr><tr><td>- w/o Coarse-to-fine Schema Linking</td><td>82.40</td><td>55.74</td></tr><tr><td>-w/o SyntaxKeywordPrediction</td><td>83.50</td><td>55.02</td></tr><tr><td>- w/o Concept-enhanced CoT Prompting</td><td>83.50</td><td>52.41</td></tr><tr><td>- w/o NLQ Consistency</td><td>83.70</td><td>55.93</td></tr><tr><td>- w/o Result Consistency</td><td>83.60</td><td>53.85</td></tr></table></body></html>

Augmentation Strategies. Table 6 shows that our augmentations significantly improve classifier performance in syntax keyword prediction. Keyword addition proves more beneficial than NLQ rewriting by boosting data diversity.

Table 6: Ablations of augmentation strategies on BIRD. ‘Total AUC’ refers to the sum of AUCs of all trained classifiers.   

<html><body><table><tr><td>Model Variant</td><td>Total AUC</td></tr><tr><td>CoGSQL classifiers</td><td>10.0681</td></tr><tr><td>- w/o NLQ rewriting</td><td>9.9918</td></tr><tr><td>- w/o keywords addition</td><td>9.9663</td></tr><tr><td>- w/o both augmentations</td><td>9.8483</td></tr></table></body></html>

# Fine-grained Analysis (RQ3)

To scrutinize the impact of COGSQL, we conduct an error analysis, comparing mistakes made by GPT-4o-mini baseline to those made by $\mathrm { G P T - 4 o - m i n i + C O G S Q L }$ . We classify LLM mistakes into five main categories: 1) Schema Misuse: Incorrect use or omission of table and column names in SQLs. 2) Keyword Misuse: Incorrect use of SQL keywords. 3) Nested Query Misuse: Failure to recognize the need for, or inappropriate use of, nested queries. 4) NLQ Misunderstanding: Misinterpretation of the NLQ intent. 5) Syntax Error: Generation of non-executable SQLs due to grammar errors.

We randomly select 500 NLQs from BIRD and analyze the distribution of mistakes, and explore how COGSQL mitigate them. Schema misuse is the most common error in Table 7, showing the difficulty of accurately identifying necessary tables and columns amidst complex schemas and NLQs. Keyword misuse and nested query misuse are closely tied to NLQ’s complexity, indicating a need for improved reasoning in complex SQL queries. COGSQL effectively reduces errors in all categories, proving its ability to follow cognitive processes to resolve text-to-SQL progressively.

Table 7: Mistake analysis for 4o-mini baseline and 4o-mini enhanced with COGSQL; ( ) indicates diminished mistakes.   

<html><body><table><tr><td>Mistake Category</td><td>GPT-40-mini</td><td>GPT-4o-mini+CoGSQL</td></tr><tr><td>Schema Misuse</td><td>123</td><td>75 (↓48)</td></tr><tr><td>KeywordMisuse</td><td>46</td><td>35 (↓11)</td></tr><tr><td>Nested Query Misuse</td><td>47</td><td>39(↓8)</td></tr><tr><td>NLQ Misunderstanding</td><td>35</td><td>22 (↓ 13)</td></tr><tr><td>Syntax Error</td><td>11</td><td>2(↓9)</td></tr><tr><td>Total</td><td>262</td><td>173 (↓89)</td></tr></table></body></html>

# Conclusion

While LLMs elevate the performance of text-to-SQL to the next level, complex benchmarks such as BIRD continue to challenge even the most advanced LLMs. In this study, we present COGSQL, a novel framework designed to emulate human cognitive processes to enhance LLMs’ reasoning abilities. COGSQL leverages coarse-to-fine schema linking and syntax keyword prediction to effectively recall key concepts, followed by concept-enhanced CoT prompting for precise SQL generation. Consistency-based correction, from both NLQ and result perspectives, ensures robust and reliable adjustment to final outputs. Extensive experiments across diverse datasets and LLMs verify COGSQL’s superiority and confirm the benefits of mimicking human cognition.