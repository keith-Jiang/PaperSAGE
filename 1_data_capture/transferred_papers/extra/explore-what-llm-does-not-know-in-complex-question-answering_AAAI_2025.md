# Explore What LLM Does Not Know in Complex Question Answering

Xin Lin1,2, Zhenya Huang1,2,3, Zhiqiang Zhang5, Jun Zhou4, Enhong Chen1,2∗

School of Computer Science and Technology, University of Science and Technology of China, Hefei, China 2State Key Laboratory of Cognitive Intelligence, Hefei, China 3Institute of Artificial Intelligence, Hefei Comprehensive National Science Center, Hefei, China 4Zhejiang University, Hangzhou, China 5Independent Researcher linx@mail.ustc.edu.cn, {huangzhy, cheneh}@ustc.edu.cn, {zzqsmall, junzhougucas}@gmail.com

# Abstract

Complex question answering (QA) is a challenging task in artificial intelligence research which requires reasoning based on related knowledge. The retrieval-augmented generation (RAG) based on large language models (LLMs) have become one promising solution in QA. To facilitate RAG more effectively, the LLM needs to precisely evaluate knowledge required in QA. That is, first, the LLM needs to examine its knowledge boundary (what the LLM does not know) to retrieve external knowledge as supplement. Second, the LLM needs to evaluate the utility of the retrieved knowledge (whether it helps in reasoning) for robust RAG. To this end, in this paper, we propose a novel Question Answering with Knowledge Evaluation (KEQA) framework to promote the effectiveness and efficiency of RAG in QA. First, inspired by quizzes in classroom, we propose a quiz-based method to precisely examine the knowledge state of the uninterpretable LLM for QA. We ask indicative quizzes on each required knowledge, and inspect whether the LLM can consistently answer the quiz to examine its knowledge boundary. Second, we retrieve the unknown knowledge from external source, and evaluate its utility to pick the helpful ones for reasoning. We design a reasoning-based metric to evaluate utility, and construct a demonstration set in training data for reference to guide knowledge picking in inference. We conduct extensive experiments on four widely-used QA datasets, and the results demonstrate the effectiveness of the proposed method.

# 1 Introduction

Complex question answering (QA) is a key task in artificial intelligence (AI) research (Lehnert 1978), which aims to answer questions based on related knowledge. Therefore, the QA systems are required to master multiple knowledge, and perform complex reasoning over these knowledge, making it a challenging task. As shown in Figure 1, to answer the question “what is the birth date of the person Richard Callaghan coached ...”, the QA system should know “who did Richard Callaghan coached ...” (“Tara Lipinski”), and “what is the birth date of Tara Lipinski” (“June 10, 1982”).

Recently, the large language models (LLMs) have become the most promising solution for QA due to numerous pre-trained knowledge stored in huge parameters and

Question: What is the birth date of the person Richard Callaghan   
coached to Olympic, world, and national titles?   
Required Knowledge   
K1: Who did Richard Callaghan coach to Olympic, world, and   
national titles?   
LLM: Tara Lipinski (Known)   
$\mathbf { K } _ { 2 }$ : What is the birth date of Tara Lipinski?   
LLM: June 1977 (Unknown) P1: “... Tara Kristen Lipinski (born June 10, 1982) ...”(Helpful) P2: “... Lipinski appeared on “The Today Show” on March 18, 2011 ...”(Helpless) P3: “... Lipinski was coached by Jeff DiGregorio ...”(Helpless)   
RAG: June 10, 1982   
Answer: The person Richard Callaghan coached to Olympic, world,   
and national titles is Tara Lipinski. She was born in June 10, 1982. So   
the answer is June 10, 1982.

strong reasoning abilities (Achiam et al. 2023; Zhou et al. 2023; Zheng et al. 2024). The retrieval-augmented generation (RAG), which first retrieves knowledge and then performs generation, further supplements the LLMs with external knowledge for more accurate and reliable QA (Wang et al. 2023b; Jiang et al. 2023). To facilitate RAG more effectively, the LLM needs to accurately evaluate knowledge required in QA. First, the LLM needs to precisely examine its knowledge boundary, i.e., what the LLM has already known and what it does not know, and adopts different actions. For example, in Figure 1, the LLM has already known the first knowledge $K _ { 1 }$ “who did Richard Callaghan coached ...” which could be directly used in QA, but has not mastered $K _ { 2 }$ “what is the birth date of Tara Lipinski” which needs to be supplemented from external to answer the question. Second, the LLM needs to evaluate whether the knowledge could help to answer the question (i.e., the utility) especially for external knowledge. As shown in Figure 1, all three knowledge $( P _ { 1 } , P _ { 2 }$ and $P _ { 3 }$ ) are relevant to the question (“Tara Lipinski”), but only $P _ { 1 }$ helps in QA with the birth date and the rest are helpless and may even mislead reasoning. Therefore, we hope to design a framework for accurate knowledge evaluation to promote RAG in complex QA.

However, there are several technical challenges along this line. First, it is hard to precisely examine the knowledge boundary of the uninterpretable LLMs in complex reasoning. Existing methods mainly prompt the LLM to dynamically determine whether to retrieve external knowledge (Wang et al. 2023b; Jiang et al. 2023; Asai et al. 2024), which has been proven to be inaccurate, as LLMs tend to be over-confident and unknown knowledge may be mistakenly treated as known (Ren et al. 2023; Yin et al. 2023). Another more reasonable method is to conduct retrieval when the generation probability is low (Jiang et al. 2023). However, it focuses on the uncertain tokens, and is hard to precisely find what knowledge is missing. Moreover, it can not work when the probability is unavailable. How to precisely examine the knowledge state of the LLM is a challenging problem. Second, identifying the utility of knowledge in QA may be difficult for complex questions. Different from relevancy that can be directly observed from the semantics or literal contents of the knowledge and question (e.g., they both contain “Tara Lipinski” in Figure 1), the utility of knowledge in QA may be affected by reasoning logic (whether it is necessary in one reasoning step, such as $P _ { 2 }$ and $P _ { 3 }$ are not used in Figure 1) and the ability of LLM (whether the LLM masters the knowledge and whether the LLM may be affected by the knowledge). We can hardly analyze whether the knowledge is helpful unless we perform reasoning with the knowledge and observe the outputs. How to evaluate the utility of the knowledge in QA is another non-trivial technical challenge.

To this end, we explore the knowledge boundary of the LLM and the utility of knowledge in reasoning, and propose a novel Question Answering with Knowledge Evaluation (KEQA) framework to promote RAG for complex QA. First, inspired by the quizzes in classroom, we propose a quizbased method to precisely examine whether the LLM masters knowledge required in QA. We generate quizzes on each required knowledge to evaluate the LLM, and confirm the results by inspecting whether the LLM can consistently answer the quiz. We only retrieve the unknown knowledge that the LLM fails in the quiz from external source for efficiency. Second, to evaluate the utility of retrieved knowledge, we design a metric to compute the utility of the knowledge based on reasoning outputs. We compute the knowledge utility in the training data, and construct a demonstration set for reference. In inference, we use the reference set to guide the LLM to evaluate the utility of each retrieved knowledge and pick the helpful ones in reasoning for robust RAG. Finally, we conduct extensive experiments on four widely-used datasets in QA to evaluate the KEQA framework, and the results demonstrate that the proposed method can achieve higher performances and efficiency, and more robust RAG as well.

# 2 Related Work

Question Answering. Question answering is a key task in AI and natural language processing (NLP) research. Studies on QA have evolved from early rule-based methods (Hirschman et al. 1999), neural network methods (Cui et al. 2017; Lin et al. 2024a; Liu et al. 2023b), pre-trained methods (Bian et al. 2021; Lin et al. 2024b; Liu et al. 2023a) to recent LLM-based methods (Yang et al. 2022; Liu et al. 2024; Xue et al. 2024b). Recently, LLMs have shown strong reasoning abilities in various NLP tasks, and become the most promising solution for QA. The researchers have designed several methods to further evoke the pretrained knowledge in LLMs, and improve reasoning. For example, Kojima et al. (Kojima et al. 2022) discovered that prompts like “Let’s think step-by-step” could let the LLMs output the reasoning process (chain of thought, CoT) and increase accuracy, and Wei et al. (Wei et al. 2022) further used CoTs as demonstrations for stable improvements. Wang et al. (Wang et al. 2023a) proposed self-consistency by voting between multiple reasoning for robust results and higher performances. Researchers further improved CoT by explicitly decomposing the complex questions to plan the reasoning logic (Xue et al. 2024a). Zhou et al. (Zhou et al. 2023) and Dua et al. (Dua et al. 2022) proposed least-tomost prompting and successive prompting respectively by asking and answering sub-questions step-by-step to solve complex questions. Moreover, researchers also designed the retrieval-augmented generation and verification to avoid factual faults with external knowledge (Trivedi et al. 2023; Dhuliawala et al. 2024), and self-refine to refine the outputs and increase accuracy (Madaan et al. 2023).

Retrieval-Augmented Generation. Researchers find that LLMs often generate erroneous facts in the output, which is called hallucination. The retrieval-augmented generation (RAG) is the most widely-used method to address the problem. Vanilla RAG first retrieves external knowledge with the question, and then generates the output based on the knowledge (Lewis et al. 2020). In multi-step reasoning tasks where one may need to retrieve with the intermediate output, Trivedi et al. (Trivedi et al. 2023) and Feng et al. (Feng et al. 2024) extended the vanilla RAG by iteratively performing retrieval and generation on the generated and retrieved results. Jiang et al. (Jiang et al. 2023) further proposed the FLARE framework which dynamically determined whether retrieval was required based on the generation probability and performed retrieval only when necessary. In this way, it could achieve good tradeoff between retrieval costs and performances. In addition, researchers also improved RAG from other perspectives. For example, Ma et al. (Ma et al. 2023) and Wang, Yang, and Wei (Wang, Yang, and Wei 2023) improved retrieval by rewriting the query. Sun et al. (Sun et al. 2023) and Yu et al. (Yu et al. 2023) replaced external retrieval with passage generation from LLM, and Yoran et al. (Yoran et al. 2024) explored several methods to make RAG more robust to irrelevant retrieval results.

Our work differs from existing methods as follows. First, existing methods seldom precisely examine the knowledge state of LLMs, which may limit performances or efficiency of RAG, while we design a quiz-based knowledge evaluation for LLMs, which is more precise and could work on blackbox LLMs without probability. Second, existing methods mainly improve the relevancy of the retrieved knowledge for robustness, while we further study their utility in reasoning to pick those more helpful ones to promote RAG.

# (a) Quiz-based Knowledge Evaluation (QKE)

Qu $Q$ ion What is the birth date of the person Richard Callaghan coached to Olympic, world, and national titles? Quiz Generation Quiz Assessment Quiz 𝑞1 Tara Lipinski Consistency Explanation 𝐸 ： Who did Richard Callaghan coach to Tara Lipinski Answer 𝑎1 ：   
𝑮𝒆𝒏𝑸 Quiz 𝑞2 TJTuoaJndruedan1eEL0li1,dp9ir17ne9sd78kg2ie ? Answer 𝑎2 𝑺𝒖𝒎𝒎 atCnoadlOlnlaygtihmoapniacl,otiwatlcoehrsledid,s 𝑇𝑐 = 0.8 ≥ 𝛼 𝑺𝒊𝒎 𝑸𝒖𝒊𝒛 Lipinski? JuJnuely1,1,9189077 𝑇 = 0.4 < 𝛼 w1a9s82b.oSron  tihneJaunsew1e0r, 0 is June 10, 1982.   
(b) Utility-guided Knowledge Picking (UKP) Utility Reference (Training data) Utility Evaluation (Inference) 𝑹𝑨𝑮 Answer 𝑌 $\bar { \pmb { Q } }$ ::WMhaourihcaedKthoecihdleian aofndthEemEiilfefeNloTuoguwier?) Retr𝕂i′ev←edℛk(n𝑞o ,wl𝕂e)dge g +ℝ ഥ𝑲𝟏′: “... degined 𝒀𝑀′𝒀ഥ:( ′𝑌Mത:′a,Gu𝑌തur)isct=eavK0eo5Eeicffheliln O J𝑲u𝟏𝟐n′:e“1..0. TL1ia9pri8an2sK)kr.i.i.sa”tpepneLairpeidnsokni (“bTohren 8 𝑀𝒀ഥ𝟐′𝑌ത:1 ,M𝑌തaur<ice𝑀K(𝑌oത′e,c𝑌തhl)in, ℝ J𝑲e𝟑ff′:D“.i..GrLeipgionrsikoi .w. a”s coached by ഥ𝑲𝟐′:“... originated by Maurice ...” Emile Nouguier 𝑀 𝑌ത2′, 𝑌ത > 𝑀(𝑌ത′, 𝑌ത)

# 3 KEQA Framework

# 3.1 Problem Definition

Question answering consists of the question $Q$ (e.g., “what is the birth date of ...?”) and the answer $Y$ (“June 10, 1982”) to $Q$ , which are both in natural language. An external knowledge source $\mathbb { K } = \{ K _ { 1 } , K _ { 2 } , \cdot \cdot \cdot , K _ { n } \}$ is given along with the QA data, where $K _ { i } \in \mathbb { K }$ may be a passage in corpus, knowledge triple in knowledge graph or webpage online depending on $\mathbb { K }$ . In this paper, we focus on the passages.

Given the knowledge source $\mathbb { K }$ and the question $Q$ , our goal is to retrieve knowledge $\mathbb { K } ^ { * } ~ = ~ \{ K _ { 1 } ^ { * } , K _ { 2 } ^ { * } , \cdot \cdot \cdot ~ , K _ { m } ^ { * } \}$ from $\mathbb { K }$ with a retriever $\mathcal { R }$ if necessary, and generate one explanation $E$ with a LLM $\mathcal { L }$ to infer the answer $Y$ to $Q$ . In this paper, we hope to first evaluate whether $\mathcal { L }$ masters knowledge required to answer $Q$ , and only retrieve the unknown ones from $\mathbb { K }$ to generate the answer $Y$ .

# 3.2 Framework Overview

We propose a novel Question Answering with Knowledge Evaluation (KEQA) framework to promote retrieval and reasoning in question answering. As shown in Figure 2, KEQA is composed of the Quiz-based Knowledge Evaluation (QKE) and the Utility-guided Knowledge Picking (UKP). QKE examines the knowledge boundary of the LLM $\mathcal { L }$ for the question $Q$ , and UKP retrieves the unknown knowledge that are helpful in reasoning from $\mathbb { K }$ to augment answer generation. More specifically, first, QKE generates the quizzes on each knowledge required by $Q$ to examine the LLM, and inspects whether the LLM $\mathcal { L }$ could answer these quizzes. Next, for quizzes that $\mathcal { L }$ fails, UKP retrieves related knowledge from knowledge source $K$ with the retriever $\mathcal { R }$ , evaluates the utility of each retrieved knowledge, and picks helpful ones to answer the failed quizzes. Finally, KEQA generates the explanation $E$ and answer $Y$ to $Q$ with $\mathcal { L }$ based on the answers of the quizzes. In the following sections, we will introduce the QKE and UKP in detail.

# 3.3 Quiz-Based Knowledge Evaluation

Retrieval may be quite expensive in costs or time latency for RAG, so it is practical to examine the knowledge boundary of the LLM and only retrieve the unknown knowledge to reduce retrieval (Jiang et al. 2023). However, it is hard to achieve the goal on the huge uninterpretable parameters of the LLM. Therefore, inspired by the quizzes in classroom to evaluate students’ knowledge proficiency, we design a quizbased knowledge evaluation to examine the knowledge state of the LLM in a result-oriented manner for the target question $Q$ . We ask the LLM knowledge-related questions, and inspect whether the LLM could answer these questions as shown in Figure 2(a). We call each question one quiz.

Quiz Generation. The complex question $Q$ (e.g., “what is the birth ... titles” in Figure 2) is not a suitable quiz to evaluate the LLM. If the LLM fails in the question, it is hard to determine whether the LLM misses required knowledge and which knowledge it misses, as multiple reasons may lead to the failure such as incorrect reasoning logic, inconsistent generation, and absence of any required knowledge. Therefore, we need simpler indicative quizzes that is only related to one piece of required knowledge and does not need complex reasoning (e.g., “what is the birth date of Tara Lipinski”). Obviously it is reasonable to assume that the LLM fails the quiz only due to absence of the related knowledge.

Given the complex question $Q$ , we use the LLM $\mathcal { L }$ to generate the quizzes by decomposing $Q$ . We carefully craft demonstrations to prompt $\mathcal { L }$ to decompose $Q$ into simple sub-questions $q _ { 1 } , q _ { 2 } , \cdots , q _ { s }$ . Each sub-question $q _ { i }$ is expected to be directly answered with related knowledge, and thus used as one quiz. Following (Radhakrishnan et al. 2023; Zhou et al. 2023), we adopt the fixed demonstrations, where we select the questions considering their answer and reasoning types for diversity, and manually write the quizzes for quality. As the follow-up quizzes may depend on results of previous ones, we use the number of the quiz to represent its answer in question decomposition, and recover to the answer in reasoning. For example, the question $Q$ in Figure 2 is decomposed into two quizzes: $q _ { 1 }$ “who did Richard Callaghan ..” and $q _ { 2 }$ “what is the birth date of #1”. The process can be formally represented as

$$
\{ q _ { 1 } , q _ { 2 } , \cdot \cdot \cdot , q _ { s } \}  G e n Q ( Q , { \mathcal { L } } ) .
$$

Note that typical question decomposition methods (Zhou et al. 2023) mainly focus on planning the reasoning logic, while we aim to generate simple related quizzes to examine knowledge state with decomposition as a suitable technique.

Quiz Assessment. Given the quizzes, we can examine the LLM by investigating whether the LLM could answer the quizzes. However, we have no references such as the ground truth, external knowledge, or generation probability (which may be unavailable in black-box LLM) to assess the predicted answer. Therefore, inspired by selfconsistency (Wang et al. 2023a), we assume that the LLM could output consistent answers with the same meanings in multiple tries if it masters related knowledge; otherwise, the LLM would randomly guess and output different answers. Along this line, another challenge is to judge whether two answers are consistent (i.e., have the same meaning), as the same answer may have different literal contents. For example, in Figure 2, “June 10, 1982” and “June, $1 9 8 2 ^ { , 9 }$ are consistent answers to $q _ { 2 }$ . We should compare the answers based on semantics rather than literal contents.

Formally, given the quizzes for $Q$ , we prompt the LLM $\mathcal { L }$ to answer each $q _ { i }$ without external knowledge:

$$
a _ { i }  Q u i z ( q _ { i } , \mathcal { L } ) .
$$

To assess the answer $a _ { i }$ , we inspect whether the LLM could reach a consistency in $N _ { c }$ tries, i.e., the proportion $T _ { c }$ of the consistent answers reaches the threshold $\alpha$ . To achieve the goal, we first judge whether two answers $a _ { i }$ and $a _ { j }$ are consistent to the quiz $q$ with a semantic discriminator $\mathcal { D } _ { s }$ such as LLaMA (Touvron et al. 2023) for computation efficiency:

$$
s a m e \gets S i m ( q , a _ { i } , a _ { j } , \mathcal { D } _ { s } ) .
$$

For example, “June 10, 1982” and “June, 1982” are consistent to $q _ { 2 }$ , but “June, $1 9 8 2 ^ { , 9 }$ and “June $1 9 7 7 ^ { \circ }$ are not. After that, we design an efficient algorithm to assess the consistency score $T _ { c }$ of $N _ { c }$ answers in Algorithm 1. Given the consistent answer $A _ { c }$ and consistency score $T _ { c }$ from the algorithm, if $T _ { c } \geq a l p h a .$ $\mathcal { L }$ is considered to master knowledge

Algorithm 1: Consistency-based assessment

Input: Quiz $q$ , LLM $\mathcal { L }$ , semantic discriminator $\mathcal { D } _ { s }$ Parameter: Number of tries $N _ { c }$ Output: Answer $A _ { c }$ , consistency score $T _ { c }$

1: $C S  \emptyset$   
2: for $i = 1  N _ { c }$ do   
3: $a _ { i } \gets Q u i z ( q , \mathcal { L } )$   
4: for $A S \in C S$ do   
5: $s a m e \gets S i m ( q , a _ { i } , A S [ 0 ] , \mathcal { D } _ { s } )$   
6: if $s a m e = = T r u \epsilon$ then   
7: $A S  A S \cup \{ a _ { i } \}$   
8: break   
9: end if   
10: end for   
11: if $a _ { i }$ does not match any $A S \in C S$ then   
12: $C S  C S \cup \{ \{ a _ { i } \} \dot  \}$   
13: end if   
14: end for   
15: $A S _ { c } \gets \operatorname* { m a x } _ { A S \in C S } | A S |$   
16: $A _ { c }  A S _ { c } [ 0 ] , T _ { c }  \vert A S _ { c } \vert / N _ { c }$   
17: return $A _ { c } , T _ { c }$

related to $q$ ; otherwise, $\mathcal { L }$ has not mastered related knowledge, and needs to retrieved from external source $\mathbb { K }$ . For example, in Figure 2, $\mathcal { L }$ generates 5 answers for $q _ { 1 }$ , and 4 of them are “Tara Lipinski” which account for $T _ { c } = 0 . 8 \ge \alpha$ (Assume $\alpha = 0 . 8 \$ ), so $\mathcal { L }$ masters knowledge related to $q _ { 1 }$ ; while in the 5 answers for $q _ { 2 }$ , only 2 of them are consistent (“June 1997” or “June, $1 9 8 2 ^ { \cdots }$ ) accounting for $T _ { c } = 0 . 4 < \alpha$ , so $\mathcal { L }$ has not mastered related knowledge to $q _ { 2 }$ .

Compared with existing self-consistency methods (Wang et al. 2023a) that mainly concern the final answer, KEQA uses it as a metric for intermediate knowledge evaluation and further considers the open answers.

# 3.4 Utility-Guided Knowledge Picking

To reduce retrieval and promote efficiency, we only retrieve the unknown knowledge that $\mathcal { L }$ fails from external source $\mathbb { K }$ . Most retrievers maximize the relevancy of the retrieved knowledge, but can not ensure they help in QA (as shown in Figure 1), so we need to further evaluate the utility of each knowledge and pick the helpful ones. As it may be difficult to directly tell the utility of the knowledge in answering the question, we design a metric to evaluate the utility based on the reasoning outputs. We construct a demonstration set in the training data for reference, and use it to guide the utility evaluation in inference as shown in Figure 2(b).

Utility Reference. Helpful knowledge can increase the correctness of the output, so we define the utility of knowledge $K ^ { \prime }$ to question $Q$ as whether $K ^ { \prime }$ increases the correctness of the predicted answer $Y ^ { \prime }$ for $Q$ . We can compute the utility given the true answer in the training data. For QA pair $( \overline { { Q } } , \dot { \overline { { Y } } } ) ^ { \mathbf { \bar { \mathbf { \alpha } } } } \in \mathbb { D } _ { t r a i n }$ , we first retrieve candidate knowledge for $\overline { { Q } }$ as $\overline { { \mathbb { K } } } ^ { \prime }  \mathcal { R } ( \overline { { Q } } , \mathbb { K } )$ , and let $\mathcal { L }$ predict the answer ${ \overline { { Y } } } ^ { \prime }$ to $\overline { { Q } }$ with each $\overline { { K } } ^ { \prime } \in \overline { { \mathbb { K } } } ^ { \prime }$ . We measure the correctness of ${ \overline { { Y } } } ^ { \prime }$ with

Input: Question $Q$ , LLM $\mathcal { L }$ , knowledge source $\mathbb { K }$ , utility reference $\mathbb { R }$ , knowledge retriever $\mathcal { R }$ , reference retriever $\mathcal { R } _ { u }$ , semantic discriminator $\mathcal { D } _ { s }$ , utility discriminator $\mathcal { D } _ { u }$

Parameter: Number of tries $N _ { c }$ , consistency threshold $\alpha$ Output: Explanation $E$ , answer $Y$

1: $Q S  G e n Q ( Q , { \mathcal { L } } )$   
2: $Q A S  \emptyset$   
3: for $q \in Q S$ do   
4: $A _ { c } , T _ { c } \gets C o n s i s t \_ a s s e s s ( q , \mathcal { L } , \mathcal { D } _ { s } , N _ { c } )$   
5: if $T _ { c } \ge \alpha$ then   
6: $Q A S  Q A S \cup \{ ( q , A _ { c } ) \}$   
7: else   
8: $\mathbf { \mathbb { K } ^ { \prime } } \gets \mathcal { R } ( q , \mathbb { K } )$   
9:   
10: for $K ^ { \prime } \in \mathbb { K } ^ { \prime }$ do   
11: $\mathbb { R } ^ { \prime }  \mathcal { R } _ { u } ( q , K ^ { \prime } , \mathbb { R } )$   
12: if ${ \cal U } t i l ( q , K ^ { \prime } ) = = 1$ then   
13: $\mathbb { K } ^ { * }  \mathbb { K } ^ { * } \cup \{ K ^ { \prime } \}$   
14: end if   
15: end for   
16: $\begin{array} { l } { { a  R A G ( q , \mathbb { K } ^ { * } , { \mathcal { L } } ) } } \\ { { Q A S  Q A S \cup \{ ( q , a ) \} } } \end{array}$   
17:   
18: end if   
19: end for   
20 $E , Y \gets S u m m ( Q , Q A S , \mathcal { L } )$   
21: return E, Y

and query $q$ as $\mathbb { K } ^ { \prime } \gets \mathcal { R } ( q , \mathbb { K } ) \subset \mathbb { K }$ . After that, we evaluate the utility of each $K ^ { \prime } \in \mathbb { K } ^ { \prime }$ by using a utility discriminator $\mathcal { D } _ { u }$ (e.g., LLaMA) with demonstrations $\mathbb { R } ^ { \prime }$ from $\mathbb { R }$ . We search $\mathbb { R } ^ { \prime }$ by maximizing the BERT-based semantic similarity to $( q , K ^ { \prime } )$ as $\mathcal { R } _ { u }$ , and adopt FAISS (Douze et al. 2024) to accelerate computing. The process can be formulated as:

$$
\begin{array} { r l } & { U t i l ( q , K ^ { \prime } ) \gets U t i l i t y ( q , K ^ { \prime } , \mathbb { R } ^ { \prime } , \mathcal { D } _ { u } ) , } \\ & { \qquad \mathbb { R } ^ { \prime } \gets \mathcal { R } _ { u } ( q , K ^ { \prime } , \mathbb { R } ) \subset \mathbb { R } . } \end{array}
$$

We only pick the retrieved knowledge $K ^ { \prime } \in \mathbb { K } ^ { \prime }$ that is helpful with $\bar { U } t i l ( q , K ^ { \prime } ) = 1$ to answer the quiz $q$ with $\mathcal { L }$ :

$$
\begin{array} { r } { a  R A G ( q , \mathbb { K } ^ { * } , \mathcal { L } ) , \mathbb { K } ^ { * } = \{ K ^ { \prime } \in \mathbb { K } ^ { \prime } | U t i l ( q , K ^ { \prime } ) = 1 \} . } \end{array}
$$

After answering all quizzes $q _ { i } \in G e n Q ( Q , \mathcal { L } )$ , we summarize all answers $a _ { i }$ to quizzes $q _ { i }$ , and conclude one overall explanation $E$ and final answer $Y$ to the question $Q$ with $\mathcal { L }$ :

$$
E , Y \gets S u m m ( Q , \{ ( q _ { i } , a _ { i } ) \} , \mathcal { L } ) .
$$

The whole inference process of the proposed KEQA framework is demonstrated in Algorithm 2.

# 4 Experiments

a metric $\mathcal { M }$ based on the true answer $\overline { { Y } }$ . The utility label of $\overline { { K } } ^ { \prime }$ to question $\overline { { Q } }$ can be formulated as:

$$
\begin{array} { r l } & { U ( \overline { { Q } } , \overline { { K } } ^ { \prime } )  \{ \begin{array} { l l } { 1 } & { C o r ( \overline { { Q } } , \overline { { \mathbb { K } } } ^ { * } \cup \{ \overline { { K } } ^ { \prime } \} ) > C o r ( \overline { { Q } } , \overline { { \mathbb { K } } } ^ { * } ) } \\ { 0 } & { C o r ( \overline { { Q } } , \overline { { \mathbb { K } } } ^ { * } \cup \{ \overline { { K } } ^ { \prime } \} ) = C o r ( \overline { { Q } } , \overline { { \mathbb { K } } } ^ { * } ) , } \\ { - 1 } & { C o r ( \overline { { Q } } , \overline { { \mathbb { K } } } ^ { * } \cup \{ \overline { { K } } ^ { \prime } \} ) < C o r ( \overline { { Q } } , \overline { { \mathbb { K } } } ^ { * } ) } \end{array}  } \\ & { C o r ( \overline { { Q } } , \overline { { \mathbb { K } } } ^ { * } ) = \mathcal { M } ( \overline { { Y } } ^ { \prime } , \overline { { Y } } ) = \mathcal { M } ( A n s ( \overline { { Q } } , \overline { { \mathbb { K } } } ^ { * } , \mathcal { L } ) , \overline { { Y } } ) , } \end{array}
$$

where $\overline { { \mathbb { K } } } ^ { * }$ is knowledge set used to predict ${ \overline { { Y } } } ^ { \prime }$ . We examine each $\overline { { \boldsymbol { K } } } ^ { \prime } \in \overline { { \mathbb { K } } } ^ { \prime }$ in descending relevancy order, and include previously confirmed helpful knowledge in $\overline { { \mathbb { K } } } ^ { * } . \mathcal { M }$ can be any metrics, and in this paper we adopt the widely-used F1 score. For example, in Figure 2, $\overline { { K } } _ { 1 } ^ { \prime }$ decreases the correctness of $\overline { { Y } } _ { 1 } ^ { \prime }$ compared with ${ \overline { { Y } } } ^ { \prime }$ (without $\overline { { K } } _ { 1 } ^ { \prime } )$ , so it is helpless; while $\overline { { K } } _ { 2 } ^ { \prime }$ increases the correctness and is thus helpful.

As samples with $U ( \overline { { Q } } , \overline { { K } } ^ { \prime } ) = 0$ contain less information + $\overline { { K } } ^ { \prime }$ is helpless, or $\overline { { \mathbb { K } } } ^ { * }$ has contained sufficient knowledge), we only reserve samples with $U ( \overline { { Q } } , \overline { { K } } ^ { \prime } ) \in \{ 1 , - 1 \}$ as reference $\mathbb { R }$ to guide utility evaluation in inference:

$$
\mathbb { R }  \{ ( \overline { { Q } } , \overline { { K } } ^ { \prime } , U ( \overline { { Q } } , \overline { { K } } ^ { \prime } ) ) | U ( \overline { { Q } } , \overline { { K } } ^ { \prime } ) \in \{ 1 , - 1 \} \} .
$$

Utility Evaluation. In inference, for each quiz $q$ that $\mathcal { L }$ fails in QKE, we first retrieve candidate knowledge set $\mathbb { K } ^ { \prime }$ from external source $\mathbb { K }$ with the retriever $\mathcal { R }$ (e.g., BM25)

In this section, we conduct extensive experiments on four QA benchmarks to evaluate KEQA framework. 1

# 4.1 Experimental Setup

We use four benchmarks for QA including both one-hop and multi-hop QA tasks. We use the NaturalQuestions (NQ) (Kwiatkowski et al. 2019) for one-hop QA, and StrategyQA (Geva et al. 2021), HotpotQA (Yang et al. 2018) and 2WikiMultihopQA (2WMQA) (Ho et al. 2020)for multi-hop QA. As the test data in these datasets do not contain ground truth annotations, we use the train data of StrategyQA and dev data of other datasets, and sample 500 instances for each dataset to reduce the costs of running experiments following previous work (Trivedi et al. 2023; Jiang et al. 2023).

We use accuracy (ACC) to evaluate the performances on StrategyQA with only yes-or-no questions, and F1 score and Exact Match (EM) on other datasets with open answers following (Jiang et al. 2023; Trivedi et al. 2023). Note that answers of these datasets are quite short, so similarity metrics for long answers like BLEU are not quite appropriate, and the word-level F1 score actually performs a similar 1-gram evaluation for the short answers.

To implement the KEQA framework, we use gpt-3.5- turbo2 as the LLM $\mathcal { L }$ , and BM25 algorithm implemented in Elasticsearch3 as the retriever $\mathcal { R }$ following (Jiang et al. 2023; Trivedi et al. 2023). We use Wikipedia dump from Dec 20, 2018 in (Karpukhin et al. 2020) as the knowledge source $\mathbb { K }$ following (Jiang et al. 2023; Asai et al. 2024). For the semantic and utility discriminator $\mathcal { D } _ { s }$ and $\mathcal { D } _ { u }$ , we both adopt Llama-2-7b-chat-hf4. Reference retriever $\mathcal { R } _ { u }$ is implemented with Bert and FAISS. In QKE, we set $N _ { c }$ and $\alpha$ for consistency to 5 and 0.8. In UKP, we retrieve top-10 candidate knowledge from $\mathbb { K }$ before knowledge picking, and top-8 demonstrations as $\mathbb { R } ^ { \prime }$ from $\mathbb { R }$ . We run all experiments on a Linux server with two $2 . 2 0 \ : \mathrm { G H z }$ Intel Xeon E5-2650 CPUs and an NVIDIA A100 GPU.

<html><body><table><tr><td>Dataset Metric</td><td>NQ F1</td><td>EM</td><td>StrategyQA ACC F1</td><td>HotpotQA EM</td><td>F1</td><td>2WMQA EM</td></tr><tr><td>Vanilla GPT-3.5</td><td>0.427</td><td>0.294</td><td>0.468</td><td>0.380 0.264</td><td>0.313</td><td>0.224</td></tr><tr><td>Zero-shot CoT</td><td>0.454</td><td>0.296</td><td>0.510</td><td>0.353 0.260</td><td>0.320</td><td>0.218</td></tr><tr><td>Few-shot CoT</td><td>0.445</td><td>0.292</td><td>0.620</td><td>0.373 0.254</td><td>0.360</td><td>0.224</td></tr><tr><td>Vanilla RAG</td><td>0.385</td><td>0.258</td><td>0.516</td><td>0.387 0.254</td><td>0.314</td><td>0.244</td></tr><tr><td>ReAct</td><td>0.335</td><td>0.212</td><td>0.554</td><td>0.390 0.270</td><td>0.305</td><td>0.204</td></tr><tr><td>IRCoT</td><td>0.344</td><td>0.216</td><td>0.622</td><td>0.361 0.232</td><td>0.318</td><td>0.202</td></tr><tr><td>FLARE</td><td>0.455</td><td>0.318</td><td>0.662</td><td>0.391 0.268</td><td>0.364</td><td>0.246</td></tr><tr><td>Self-Rag</td><td>0.387</td><td>0.270</td><td>0.632</td><td>0.357 0.220</td><td>0.311</td><td>0.210</td></tr><tr><td>SearChain</td><td>0.337</td><td>0.214</td><td>0.616</td><td>0.349 0.216</td><td>0.313</td><td>0.222</td></tr><tr><td>Rowen</td><td>0.452</td><td>0.286</td><td>0.666</td><td>0.382 0.240</td><td>0.307</td><td>0.212</td></tr><tr><td>SlimPLM</td><td>0.442</td><td>0.280</td><td>0.566</td><td>0.393 0.266</td><td>0.368</td><td>0.242</td></tr><tr><td>KEQA</td><td>0.483*</td><td>0.352*</td><td>0.680*</td><td>0.400*</td><td>0.405*</td><td></td></tr><tr><td>KEQA w/o QKE</td><td>0.409</td><td>0.284</td><td>0.644</td><td>0.352</td><td>0.278*</td><td>0.326*</td></tr><tr><td>KEQA w/o UKP</td><td>0.453</td><td></td><td>0.678</td><td>0.232</td><td>0.396</td><td>0.258</td></tr><tr><td>KEQA w/o R</td><td></td><td>0.302</td><td>0.676</td><td>0.350 0.250 0.252</td><td>0.398 0.398</td><td>0.314</td></tr><tr><td>KEQA w random R</td><td>0.456 0.474</td><td>0.316</td><td>0.666</td><td>0.356 0.262</td><td>0.385</td><td>0.302</td></tr><tr><td>KEQA w SE</td><td></td><td>0.324</td><td>0.678</td><td>0.375</td><td>0.397</td><td>0.288</td></tr><tr><td></td><td>0.475</td><td>0.342</td><td></td><td>0.388</td><td>0.272</td><td>0.316</td></tr></table></body></html>

Table 1: Overall results on four datasets. Existing state-of-the-art results are underlined, and the best results are bold. \* indicate a $p$ -value $< 0 . 0 5$ in the paired t-test with the strong baseline.

We compare KEQA with GPT-3.5 on direct prompt, zeroshot CoT (Kojima et al. 2022) and few-shot CoT (Wei et al. 2022). For RAG baselines, we adopt the vanilla RAG and seven advanced RAG methods, i.e., ReAct (Yao et al. 2023), IRCoT (Trivedi et al. 2023), FLARE (Jiang et al. 2023), Self-Rag (Asai et al. 2024), SearChain (Xu et al. 2024), Rowen (Ding et al. 2024), and SlimPLM (Tan et al. 2024). We rerun all baseline methods under the same settings (e.g., LLM and retriever) of KEQA for fair comparison.

# 4.2 Experimental Results

Overall Results. We compare KEQA with all baselines, and report the results in Table 1. From the results, there are several observations. First, KEQA outperforms all baselines, which demonstrates the effectiveness of the proposed method. The proposed method can precisely examine the knowledge state of the LLM and augment LLM with helpful knowledge to promote the performances. We statistically test the improvements over strong baselines with paired ttest, and find the improvements to be significant with $p <$ 0.05. Second, generally there is an obvious gap between the F1 and EM scores due to the open answers, where similar answers are treated totally different on EM. Third, we find that RAG methods do not always outperform the nonRAG baselines especially on simple tasks, which may be due to the noises in retrieval results. In most cases, RAG performs better due to more information. Last, generally adaptive RAG methods which dynamically determine whether to retrieve (e.g., FLARE) perform better than those always conduct retrieval (e.g., IRCoT), which proves the effectiveness of retrieval-on-demand to avoid noises.

Table 2: Performances of LLaMA $\mathcal { D } _ { s }$ and BERTScore.   

<html><body><table><tr><td>Dataset</td><td>NQ</td><td>StrategyQA</td></tr><tr><td>LLaMA Ds</td><td>0.611</td><td>0.774</td></tr><tr><td>BERTScore</td><td>0.559</td><td>0.671</td></tr></table></body></html>

Ablation Study. We introduce five variants of KEQA to investigate the effectiveness of each module. KEQA $w / o$ QKE treats all knowledge to be unknown and performs UKP on all quizzes. KEQA $w / o$ UKP uses all retrieved results for failed quizzes. KEQA w/o $\mathbb { R }$ absolutely uses $\mathcal { D } _ { u }$ for utility evaluation without $\mathbb { R }$ . KEQA $w$ random $\mathbb { R } ^ { \prime }$ randomly samples demonstrations $\mathbb { R } ^ { \prime }$ from $\mathbb { R }$ . KEQA $w$ SE further extracts key sentences from passages in addition to KEQA. We also report the performances of the variants in Table 1. We can get the following conclusions. First, the performances decrease when each module is missing, proving their effectiveness. Second, the performances decrease the most when QKE is missing, which proves that precise knowledge evaluation and retrieve-on-demand is a key in RAG to avoid noises. Next, KEQA w/o $\mathbb { R }$ performs better than KEQA $w / o$ UKP, which shows that $\mathcal { D } _ { u }$ can also filter out some noises, and the similar demonstrations from $\mathbb { R }$ further enhances its ability. However, the randomly sampled demonstrations may degrade the performances, and even mislead $\mathcal { D } _ { u }$ . Last, KEQA $w$ SE performs slightly worse than KEQA, which may be due to key sentences not extracted from the passages, proving that the passage is a suitable knowledge

![](images/1e89af5cbb8bf7c5249932c309b90f6768bd73ff09c487aca0981b9f3d7d593d.jpg)  
Figure 3: Average probabilities over consistency scores $T _ { c }$   
Figure 5: Performances over top-k knowledge.

![](images/2ccde00dfd1a438cfbe9af9932b897cf2dc673d2d601e6155f632b772aab200f.jpg)  
Figure 4: Performances over consistency score threshold $\alpha$ .

granularity for RAG.

Quiz Analysis. In this section, we investigate the rationality and effectiveness of the quiz-based knowledge evaluation. Due to space limitation, we only report results on NQ and StrategyQA here. We first compute the average probabilities of generating the answers to quizzes over different consistency scores $T _ { c }$ , and report the results in Figure 3. From the results, we can find that with increasing consistency score, the generation probability keeps increasing. It means that generally a higher consistency score implicates a higher generation probability of the answer to the quiz, i.e., the LLM has mastered related knowledge and is more confident on the outputs. Therefore, the consistency score is a reasonable metric to evaluate the knowledge state of the LLM and determine whether to retrieve.

We also investigate what consistency score can implicate that the LLM has mastered related knowledge, i.e., the effects of threshold $\alpha$ on the performances. We change the threshold $\alpha$ from 0.4 to 1.0 and observe the performances of KEQA in Figure 4. From the results, we can find the following observations. First, when the threshold is lower (from 0.4 to 0.8), the performances increase with growing threshold. It is reasonable as the low consistency score means that the LLM may not master related knowledge, which may be missed if not retrieved from external source. Next, when the threshold is high enough, the performances stop increasing and begin to decrease. The reason may be that some accidental factors cause the LLM to give unexpected outputs on confident quizzes and thus not reaches the threshold, and the unnecessary retrieval introduces some noises to the reasoning. It is important to keep a balance between sufficient

0.6 0.75 KEQA KEQA 0.5 Full 0.70 Full E 山 0.65 0.4 A ■ 0.60 0.3 5 10 15 20 0.55 5 10 15 20 Top-k knowledge Top-k knowledge (a) NQ (b) Stra1tegyQA

Table 3: Performances of KEQA and RAG with DPR.   

<html><body><table><tr><td>Dataset</td><td>NQ</td><td>StrategyQA</td></tr><tr><td>KEQAwDPR</td><td>0.491</td><td>0.690</td></tr><tr><td>RAGwDPR</td><td>0.486</td><td>0.662</td></tr></table></body></html>

confidence and acceptable errors in knowledge evaluation.

We further study the effectiveness of the semantic discriminator $\mathcal { D } _ { s }$ for quiz assessment. We use the GPT-3.5 as an ideal discriminator to annotate the consistency label between answers as ground truth, and evaluate the LLaMA-based $\mathcal { D } _ { s }$ compared with BERTScore as shown in Table 2. From the results, we can see that the LLaMA $\mathcal { D } _ { s }$ could achieve an acceptable performance on discriminating the consistency compared with BERTScore. More complex techniques, such as larger base model, fine-tuned method and in-context demonstrations, can be used for further improvement.

Retrieval Analysis. In this section, we study th effect of the retrieval on the performances of KEQA. We first investigate how the retrieved knowledge could affect the performances. We change the top-k knowledge retrieved from $\mathbb { K }$ before knowledge picking from 5 to 20, and report the results of KEQA (i.e. KEQA) and a variant of KEQA that retrieves for all quizzes and uses all retrieval results (i.e., Full) in Figure 5. We can draw the following findings from the results. First, the top- $\mathbf { \nabla } \cdot \mathbf { k }$ knowledge does affect the results of KEQA, but the influences may be limited. KEQA could pick a minor helpful knowledge from the results and discard the rest, so the number of helpless knowledge have little impact on KEQA. Second, KEQA stably outperforms the variant, which proves the effectiveness of the knowledge evaluation and knowledge picking in reducing the misleading noises. Last, more knowledge does not mean to increase the performances even on the variant, as the retrieved knowledge tends to be helpless and even irrelevant with growing retrieval counts, which may mislead the reasoning rather than augmenting the generation.

We also study whether KEQA could work with more powerful retrievers. We implement $\mathcal { R }$ with the widely-used dense retriever DPR (Karpukhin et al. 2020), and compare the performances of KEQA and vanilla RAG in Table 3. We can get the following observations. First, comparing the performances of KEQA and RAG on DPR and BM25, we can find that DPR performs better than BM25 with both KEQA and DPR, which shows the importance of retrieval quality in RAG. Next, KEQA outperforms the vanilla RAG with DPR and BM25, proving that KEQA could also work on more powerful retrievers to promote RAG.

Table 4: Performances of KEQA with GPT-4 and LLaMA.   

<html><body><table><tr><td>Dataset</td><td>NQ</td><td>StrategyQA</td></tr><tr><td>KEQA w GPT-4</td><td>0.515</td><td>0.760</td></tr><tr><td>RAG w GPT-4</td><td>0.496</td><td>0.733</td></tr><tr><td>KEQAwLLaMA</td><td>0.315</td><td>0.486</td></tr><tr><td>RAGwLLaMA</td><td>0.295</td><td>0.446</td></tr></table></body></html>

Table 5: Average retrieval and LLM costs for each question.   

<html><body><table><tr><td>Dataset Cost</td><td colspan="3">NQ L</td><td colspan="3">StrategyQA L token</td></tr><tr><td>KEQA IRCoT</td><td>0.138 5.08 1.46 1.46</td><td>token time 239</td><td></td><td>R 6.28|0.212</td><td>16.44</td><td>time 1016 17.21</td></tr></table></body></html>

Generalizability Analysis. We hope to study whether the KEQA could work on LLMs with different abilities. We adopt the more powerful GPT-4 and less powerful LLaMA 7B, and compare the performances of KEQA and vanilla RAG in Table 4. The observations are as follows. First, GPT-4 performs much better than GPT-3.5 on both KEQA and vanilla RAG, and GPT-3.5 outperforms LLaMA, which proves that the ability of the LLM is the key to the performances of RAG. Next, KEQA outperforms vanilla RAG on all three LLM backbones, which proves the generalizability of KEQA to work with LLMs with different abilities.

Efficiency Analysis. Finally, we investigate the costs of retrieval and LLM in KEQA, and compare with two advanced RAG, IRCoT and FLARE in Table 5. We represent the retrieval cost as the number of retrieval calls $( { \mathcal { R } } )$ , and LLM cost as the number of LLM calls $( \mathcal { L } )$ and tokens (token). We also include the latency of the whole reasoning process (time). From the results, we can get the following observations. First, we can obviously find that KEQA requests much less retrieval, which can not only reduce costs of retrieval, but also avoid noises in the retrieval results. Next, although the LLM calls of KEQA are larger, these LLM calls are majorly simple knowledge quizzes without complex reasoning, so the inputs and outputs can be quite short without heavy demonstration payload and long generation. Therefore, the overall tokens are much less. Last, KEQA seems to cost much more time compared with IRCoT and FLARE. We study the time spent on each step in detail, and find that quiz answering and assessment take up most time for 11.75s and 2.46s, while all other steps take only 2.93s. Note that the most time-consuming quiz answering and assessment can be fully parallelized for up to 5 times speedup. Therefore, although KEQA has much higher latency as shown in the table, it could achieve no much higher latency if optimized in parallel. In summary, our method can achieve high efficiency on both retrieval and LLM.

Table 6: One case generated by KEQA.   

<html><body><table><tr><td>Question Which film has the director who was born Life Is Hell?</td><td>earlier, The Assassination Of Trotsky or My</td></tr><tr><td>KEQA</td><td>The director of The Assassination Of Trotsky, Joseph Losey,was born on January 14,1909. The director of MyLife Is Hell, Josiane Balasko,wasbornonApril15,1950.Joseph Losey was born earlier than Josiane Balasko. So the answer is The Assassination Of Trotsky.</td></tr><tr><td>q1:Who is the director of The Assassination Of Trotsky? α1: Joseph Losey q2:When was Joseph Losey born? a2:January14,1909 q3:Who is the director of My Life Is Hell?</td><td></td></tr><tr><td>a3:Josiane Balasko a4:April15,1950</td><td>Retrieval:MyLifeIsHell isa 1991 French comedy film directedbyJosianeBalasko ... q4:When was the director of Josiane Balasko born?</td></tr><tr><td>q5:Is January14,1909 earlier than April15,1950? a5:Yes</td><td></td></tr></table></body></html>

Case Study. We also provide a case in Table 6 to demonstrate the reasoning steps of KEQA, and mark the information from retrieved knowledge in italic. KEQA could detect that the LLM does not know the director of My Life Is Hell and refer to external knowledge for the answer “Josiane Balasko”. For other knowledge that the LLM has already known, KEQA does not perform retrieval to reduce costs and avoid noises. Therefore, KEQA could achieve effective and efficient RAG for complex QA.

# 5 Conclusion

In this paper, we proposed a novel Question Answering with Knowledge Evaluation (KEQA) framework to promote knowledge retrieval and reasoning in question answering. We designed the quiz-based knowledge evaluation to generate knowledge-related quizzes and evaluate whether the LLM mastered knowledge required in reasoning. After that, we retrieved the unknown knowledge from external source, and evaluated their utility in reasoning to pick helpful ones to answer the question. Experimental experiments on four QA datasets demonstrated that the proposed KEQA could reach higher performances, and achieve high efficiency on both retrieval and LLM reasoning.