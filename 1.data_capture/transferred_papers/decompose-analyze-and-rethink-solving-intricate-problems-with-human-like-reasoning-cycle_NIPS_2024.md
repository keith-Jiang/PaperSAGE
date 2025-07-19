# Decompose, Analyze and Rethink: Solving Intricate Problems with Human-like Reasoning Cycle

Shangzi Xue1 Zhenya Huang1,2∗ Jiayu Liu1 Xin lin1 Yuting Ning1 Binbin Jin1 Xin Li1 Qi Liu1,2

tate Key Laboratory of Cognitive Intelligence, University of Science and Technology of China 2: Institute of Artificial Intelligence, Hefei Comprehensive National Science Center {xueshangzi,jy251198,linx,ningyt,bb0725}@mail.ustc.edu.cn; {huangzhy,leexin,qiliuql}@ustc.edu.cn

# Abstract

In this paper, we introduce DeAR (Decompose-Analyze-Rethink), a framework that iteratively builds a reasoning tree to tackle intricate problems within a single large language model (LLM). Unlike approaches that extend or search for rationales, DeAR is featured by 1) adopting a tree-based question decomposition manner to plan the organization of rationales, which mimics the logical planning inherent in human cognition; 2) globally updating the rationales at each reasoning step through natural language feedback. Specifically, the Decompose stage decomposes the question into simpler sub-questions, storing them as new nodes; the Analyze stage generates and self-checks rationales for sub-questions at each node level; and the Rethink stage updates parent-node rationales based on feedback from their child nodes. By generating and updating the reasoning process from a more global perspective, DeAR constructs more adaptive and accurate logical structures for complex problems, facilitating timely error correction compared to rationale-extension and search-based approaches such as Tree-of-Thoughts (ToT) and Graph-of-Thoughts (GoT). We conduct extensive experiments on three reasoning benchmarks, including ScienceQA, StrategyQA, and GSM8K, which cover a variety of reasoning tasks, demonstrating that our approach significantly reduces logical errors and enhances performance across various LLMs. Furthermore, we validate that DeAR is an efficient method that achieves a superior trade-off between accuracy and reasoning time compared to ToT and GoT.

# 1 Introduction

Learning to perform intricate reasoning, including commonsense reasoning [23], knowledge reasoning [28], and mathematical reasoning [8], is a crucial step towards achieving general artificial intelligence [49, 20, 25, 26, 21, 24]. The tasks always present a significant challenge as they require many human-like intricate problem-solving abilities, such as abstract thinking and logical inference, which could consolidate many decision-making applications in real-world scenarios [38, 36, 15, 34, 53, 55].

Recent advances have witnessed remarkable performances of scaled-up large language models (LLMs) in various reasoning tasks, including GPT [5], LLaMA [40], and ChatGLM [9]. They could enable several state-of-the-art prompting approaches like Chain-of-Thought (CoT) [45], Treeof-Thoughts (ToT) [49], Graph-of-Thoughts (GoT) [3], etc., to enhancing reasoning capabilities. They not only improve problem-solving performance but also reveal their intrinsic reasoning steps

Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $\$ 2$ per fresh duck egg. How much in dollars does she make every day at the farmers' market?

Answer: 18 ( √ Question Question She makes 9 \* \$2 = \$18 per day.   
The money she earns Janet's ducks lay 16 eggs Janet's ducks lay   
each day is equal to per day. After she  eats 16 eggs per day. q1: What is the q2: How many eggs   
the number of eggs three for breakfast every She sells for \$2 selling price of one does Janet have per   
per day multiplied by morning, she has 16 – per fresh duck egg. egg? day?   
\$2. 4 = 17 eggs. (×) Each egg is sold She has 16-(3+4) = 母 母 for \$2. 9 eggs per day. She sells for \$2 per fresh q3: How many eggs q4: How many eggs q5: How many eggs   
\$S3h4e.makes 17 \*2 = sehgeg esaorntshevtoetrayldnauymisbe1r7 \*S2he=m\$a3k2e.s 16 doaielsyt?he ducks lay bdroeaskJfansteteevaetr yfodray? dmouefsfiJnasnevteursyedtaoyb?ake The ducks lay 16 She eats 3 eggs for She uses 4 eggs to 母 Answer: 34 (×) 母 eggs per day. breakfast. bake muffins. (a) The simulation of ToT Reasoning (b) The simulation of DeAR Reasoning

(i.e., rationales) [47] through linear, tree-based, or graph-based structures. For example, in Figure 1 (a), given a math problem “Janet’s ducks . . . in dollars . . . market?”, ToT maintains a tree of thoughts with intermediate nodes to generate the rationales step by step. Specifically, through several operations including exploration, termination, and traceback on the nodes, ToT ultimately identifies the complete reasoning path, highlighting two-step rationales (green nodes) leading to the answer. However, although ToT and its variants [27, 35] perform the reasoning process explicitly, such a rationale-extension and search-based reasoning paradigm is still far from human-like intelligence and limits problem-solving abilities to some extent. On one hand, this tree-like structure is rigid and sometimes illogical. The ToT approaches often require setting a fixed number of thought branches (“3” branches in Figure 1 (a)) each time it expands, which can result in either missing information or redundancy to some extent. Its reasoning process essentially extends previous rationales at each step, but falls short of the logical planning inherent in human thinking to some degree [33, 42]. On the other hand, ToT generates rationale paths sequentially, and errors along the path, such as incorrectly calculating “she has $1 6 { - } 3 { + } 4 { = } 1 7$ eggs”, cannot be promptly corrected. This allows mistakes to propagate to subsequent steps, ultimately leading to an incorrect final outcome (e.g., “34”).

To address these challenges, we propose a novel reasoning paradigm DeAR (Decompose-AnalyzeRethink), which enhances LLMs’ capacity for complex problem-solving by emulating human reasoning (Figure 1 (b)). This approach is inspired by several theories in cognitive science [43, 30]. Specifically, reasoning simplification theory [33] suggests that when confronted with an intricate question, humans tend to break it down into simpler ones, which help in organizing thoughts and solving problems more logically. Referring back to Figure 1 (b), we can break down the logic by first solving two sub-questions ( $\overset { \cdot } { q } _ { 1 }$ and $\left| q _ { 2 } \right|$ ). Upon examining $q _ { 2 }$ , we find it can be further divided into three additional sub-questions ( $\overset { \cdot } { q _ { 3 } }$ , $q _ { 4 }$ , and $q _ { 5 }$ ). By sequentially resolving these sub-questions and using their results as feedback to update answers for previously generated sub-questions $( q _ { 1 }$ and $q _ { 2 }$ ), we ultimately arrive at the final answer (“18”).

To implement such a human-like problem-solving process, we introduce a Decompose-AnalyzeRethink cycle. This involves gradually constructing a reasoning tree guided by sub-questions, following a top-to-bottom reasoning process as illustrated in Figure 1 (b). The process begins with the Decompose stage (black arrows in Figure 1 (b)), where a prompt-based method breaks down the question into simpler sub-questions at subsequent nodes. Then, the Analyze stage (green box at each node) takes charge of problem-solving at the node level. The stage also introduces a self-check module to ensure the quality of the generated rationales, thus refines the reasoning process. Last, in the Rethink stage (indicated by green arrows), the result at the current node is evaluated to determine if the reasoning in parent nodes requires further updates, providing a global perspective. After multiple cycles, the answer can be summarized from the root node.

Compared to ToTs [49, 27, 35] and GoT [3], our approach presents the following highlights. First, unlike ToT/GoT methods which directly generate rationales as branches from the original question,

DeAR breaks it into sub-question tree nodes to guide the generation. Second, our tree structure is more flexible and adaptable, as each node is generated and updated autonomously by the large language model based on the problem’s logic, without relying on predefined settings. Third, DeAR enables timely correction of rationales, ultimately ensuring the correctness of the root node’s answer.

We conduct extensive experiments on three complex reasoning benchmarks including ScienceQA [28], StrategyQA [12], and GSM8K [8]. Experimental results show that our approach enhances the reasoning performance with different backbones such as GPT-3.5 [1], LLaMA2 [40], and ChatGLM3 [9]. Compared to state-of-the-art methods such as Tree-of-Thoughts (ToT) and Graph-of-Thoughts (GoT), DeAR demonstrates a significant improvement in reasoning accuracy across all backbone LLMs, validating its generalizability and scalability. Additionally, by measuring the relationship between reasoning accuracy and reasoning time across different datasets, DeAR exhibits greater efficiency, further underscoring its advantages in practical applications.

# 2 Related Work

# 2.1 Prompt-based Approaches in LLM Reasoning

There has been a growing interest in LLM reasoning research, with various prompting schemes applied in areas such as commonsense [23], mathematical [8] and knowledge reasoning [29], etc. Early methods appends examples on top of the input question (few-shot prompting [5] or performs in-context learning (ICL) [37]), or includes no examples at all (zero-shot prompting) [44].

Recent research has sought to enhance the capabilities of large language models (LLMs) by introducing intermediate reasoning steps into the prompting process, epitomized by methods such as the Chain-of-Thought (CoT) [45]. By prompting LLMs to solve problems step by step, the CoT method demonstrates outstanding performance in multi-step reasoning tasks. Self-consistency [41] is a significant improvement upon CoT, where multiple CoT paths are initially generated, and the best one is selected as the final result, thereby improving the reliability of the outputs. In parallel, other prompting methods design search-based schemes for LLMs, such as Tree-of-Thoughts (ToT) [49] and Graph-of-Thoughts (GoT) [3] which innovate by structuring the reasoning process into tree or graph structures. These structures are created to take advantage of the many reasoning paths that LLMs can generate, greatly expanding the range and depth of exploration for any given question. More recently, Reasoning via Planning (RAP) [16] repurposes the LLM as both a world model and a reasoning agent to conduct reasoning. These methods expand the reasoning space of LLMs, which can fully leverage the diverse thinking paths generated by LLMs.

# 2.2 Question Decomposition

Question decomposition, which decomposes complex questions into multiple sub-ones, has been shown to largely improve models’ reasoning ability. Early works [4] decompose questions with hand-crafted rules and lexicon-syntactic features. These works heavily rely on human efforts, which are hard to extend to general domains and tasks. Recently, researchers utilize neural network models to decompose questions [39, 18, 52]. For example, Min et al. [32] focused on directly training a model to produce sub-questions using question spans; BREAK [46] followed an alternative paradigm of collecting full question decomposition meaning representations (QDMR) annotations. However, a primary challenge lies in the scarcity of annotations for training a decomposition model [32].

More recently, in the era of LLMs, there are a lot of work exploring LLMs for question decomposition [50, 19, 17, 10, 7, 22, 51]. For example, ToT [49] prompts the LLM to decompose the rationales by searching intermediate steps. Least-to-most prompting [56] leverages a few examples to teach LLMs to decompose each problem into a series of simpler sub-problems. These prompting-based question decomposition methods serve as an important step in reasoning and planning with LLMs.

# 3 Problem Formulation and Preliminaries

# 3.1 Problem Definition

In this paper, we focus on the intricate reasoning task. The input of the task is the question $Q$ (e.g., “Janet’s ducks ... market?” in Figure 1). The output is a rationale $R = ( r _ { 1 } , r _ { 2 } , . . . , \bar { r } _ { k } )$ with $k$ word

Decompose Analyze Rethink Q Q Q 𝑟0: She makes 𝑟0: She makes 𝑟′ : She makes 9\*3=\$27 per day. 9\*3=\$27 per day. 9\*2=\$18 per day. Update 𝒒𝟏: What is the 𝒒𝟏𝟐 : How many 𝒒𝟏: What is the 𝒒𝟏𝟐 : How many 𝒒𝟏: What is the 𝒒𝟏𝟐 : How many selling price of eggs does Janet selling price of eggs does Janet selling price of eggs does Janet one egg? have per day? one egg? have per day? one egg? have per day? r1: Each egg is sold r1: Each egg is sold r12: She has 9 eggs r1: Each egg is r12: She has 9 eggs for \$3 (×) for \$2 ( √ ) per day. sold for \$2 per day. Self-check Q 𝑟0 : She makes  9\*2=\$18 per day. Answer: 18 Question: Janet's ducks lay 16 eggs per day. She eats p𝒒r𝟏i:ceWohfaotnies tehgeg ?selling J𝒒a𝟏𝟐n:etHhoawvempaenrydeagyg?s does tmhroerenifnogr abrnedabkfaaksetsevmeurfyfins Decompose Rethink r1: Each egg is sold for \$2. r12: She has 16-(3+4) = 9 eggs per day. for her friends every day with four. She sells the 𝒒𝟐𝟏 : How many 𝒒𝟐 : How 𝒒𝟐𝟑 : How many remainder at the farmers' Analyze ehgagvsedpoeersdJayn?et dmoaensyJeagngest ehgagvsedpoeersdJayn?et market daily for $\$ 2$ per fresh have per day? r23:She uses 4 duck egg. How much in Reasoning r21: The ducks r2:She eats 3 eggs to bake dollars does she make every lay 16 eggs per eggs for muffins. Tree 𝑇 day. breakfast. day at the farmers' market?

tokens (“She makes $9 \times \ S 2 = \ S 1 8$ per day.”), and the answer $A$ (“18”) derived from $R$ . Given the input question $Q$ , we aim to design a reasoning framework with LLM backbone $p _ { \theta }$ to generate the rationale $R$ and answer $A$ as outputs.

# 3.2 Reasoning Tree

Motivated by the reasoning simplification theory [33], we propose a novel reasoning structure for LLMs, named Reasoning Tree $T$ , as shown in Figure 1(b). Overall, this Reasoning Tree decomposes and resolves sub-questions using a top-down approach, while concurrently updating existing solutions through a bottom-up process. Formally, the Reasoning Tree $T$ can be defined as $T = ( N , E )$ where $N$ is the set of tree nodes and $E$ is the edge set. Each node $n = ( q , r , s ) \in N$ contains a question $q$ as a sub-question of the target $Q$ (e.g., $q _ { 2 }$ “How many eggs does Janet have per day?”), a rationale $r$ to $q$ (“She has $1 6 - ( 3 + 4 ) = 9$ eggs per day.”), and a score $s$ evaluating the logical coherence of $r$ . Each directed edge $e = ( n _ { p } , n _ { c } ) \in E$ means that the upper-level sub-question $q _ { p }$ in the parent node $n _ { p }$ is decomposed into a lower-level one $q _ { c }$ in the child node $n _ { c }$ (e.g., the parent $q _ { 2 }$ “How many eggs ... have per day” is decomposed into three children $q _ { 3 }$ “How many eggs ... lay”, $q _ { 4 }$ “How many eggs ... breakfast”, and $q _ { 5 }$ “How many eggs ... muffins”).

Our Reasoning Tree is progressively constructed and updated. The target question $Q$ in the root node is decomposed into sub-questions step by step, from sub-questions in the higher levels to the ones in the lower levels (i.e., the black directed edges in Figure 1). For example, $Q$ is first decomposed into $q _ { 1 }$ and $q _ { 2 }$ , then $q _ { 2 }$ is further decomposed into $q _ { 3 } , q _ { 4 }$ and $q _ { 5 }$ . Furthermore, humans could also rethink the rationales generated earlier (in the higher nodes) based on the ones generated later (in the lower nodes). For example, the rationales for $q _ { 4 }$ (“She eats 3 eggs for breakfast”) could be used to update rationales for $q _ { 2 }$ (“She has $1 6 - ( 3 + 4 ) = 9$ eggs per day”) through the dashed lines in green.

# 3.3 Framework Overview

To construct the aforementioned Reasoning Tree $T$ , which imitates human-like reasoning, we propose a novel DeAR (Decompose-Analyze-Rethink) cycle as the core of our framework, as illustrated in Figure 2. The cycle is composed of three stages: Decompose, Analyze and Rethink. Specifically, in the Decompose stage, one upper-level question is decomposed into several lower-level ones. In the Analyze stage, the framework solves the newly generated sub-questions by generating and selfchecking rationales. In the Rethink stage, the newly generated rationales are used to update existing ones in the parent nodes. The three stages work in a cycle to build the reasoning tree $T$ .

# 4 DeAR (Decompose-Analyze-Rethink) Cycle

In this section, we will demonstrate how the reasoning tree $T$ is constructed with the DecomposeAnalyze-Rethink cycle, as demonstrated in Figure 2.

Initially, the target question $Q$ is set as the question $q _ { 0 }$ in the root node $n _ { 0 }$ . The framework selects an existing edge node $n _ { t } = ( q _ { t } , r _ { t } , s _ { t } )$ ( $\mathit { \Pi } _ { t }$ is the level of the node) from $T$ (e.g., $n _ { 0 }$ with $Q$ “Janet’s ducks ... market?”) to start the cycle. First, in the Decompose stage (4.1), we prompt LLMs to decompose the question $q _ { t }$ in the node into sub-questions $q _ { t + 1 }$ if possible, and store them in nodes $n _ { t + 1 }$ at level $t + 1$ (e.g., $q _ { 1 } ^ { 1 }$ “What is ... one $\mathrm { { e g g } ? \mathrm { { } ^ { \dag } } }$ , and $q _ { 1 } ^ { 2 }$ “How many ... per day?”). Then, in the Analyze stage (4.2), we conduct reasoning and answers the newly generated questions $q _ { t + 1 }$ by generating rationales $r _ { t + 1 }$ for them $\mathbf { \bar { \rho } } _ { r _ { 1 } ^ { 1 } }$ “Each egg is sold for $\$ 2^ { \dag }$ for $\dot { q _ { 1 } ^ { 1 } }$ , and $r _ { 1 } ^ { 2 }$ “She has 16 eggs per day” for ${ \bar { q } } _ { 1 } ^ { 2 }$ ), checking their correctness and evaluating the coherence scores $s _ { t + 1 }$ (Eq. (5)). Next, in the Rethink stage (4.3), we use the newly generated $r _ { t + 1 }$ to update rationales in existing upper-level nodes $r _ { i } ( i \leq t ) $ (e.g., use $r _ { 1 } ^ { 1 }$ and $r _ { 1 } ^ { 2 }$ to update $r _ { 0 }$ into $r _ { 0 } ^ { \prime } .$ ). After that, the framework selects another edge node and returns to the Decompose stage (e.g., decompose $q _ { 1 } ^ { 2 }$ into $q _ { 2 } ^ { 1 }$ , $q _ { 2 } ^ { 2 }$ and $q _ { 2 } ^ { 3 } ,$ ). The cycle continues until the LLMs determine that no further decomposition is possible, thereby forming the reasoning tree $T$ for $Q$ .

As $Q$ is the question $q _ { 0 }$ for the root node $n _ { 0 }$ , after the tree-construction process, we consider the rationale $r _ { 0 }$ in the root node as the overall solution for $Q$ and extract the answer $A$ from $r _ { 0 }$ . The whole procedure is described in Algorithm 1. In the following sections, we will technically describe the three stages in the cycle and make detailed analyses.

# 4.1 Decompose Stage

# Algorithm 1 Decompose-Analyze-Rethink

According to the Analogical Reasoning theory [2], when humans conduct reasoning, they often analogize the logical processes of new questions to those of similar questions. Therefore, to make the decomposition logic of subquestions $q _ { t }$ at each level $t$ more closely resemble that of humans, we first use humanannotated question decomposition examples (Appendix A.1) as a demonstration pool $P$ . Then we calculate the cosine similarity of the representations between $Q$ and each $Q _ { i } ^ { d }$ in $P$ and select top- $K$ nearest neighbors in the vector space. After that, we concatenate each $Q _ { i } ^ { d }$ with its human-annotated sub-questions $s u b q s ^ { i } \ =$ $( s u b q _ { 1 } ^ { i } , s u b q _ { 2 } ^ { i } , . . . , s u b q _ { n } ^ { i } )$ to form $K$ questiondecomposition examples (Appendix A.1)

$$
l h _ { Q } = ( Q _ { i } ^ { d } , s u b q s ^ { i } ) ( i = 1 , 2 , . . . , K ) .
$$

These examples are regarded as “logic heuristics” that inspire the model to decompose questions in a manner closely aligned with human reasoning.

After obtaining $l h _ { Q }$ , we utilize them to decompose the sub-question $q _ { t }$ at level $t$ into multiple sub-questions at level $t + 1$ . Specifically, given question $q _ { t }$ , if its coherence score $s _ { t }$ (Eq. (5)) is higher than a threshold $\epsilon _ { 1 }$ , We ask the LLM whether it needs to be further decomposed. If $q _ { t }$ requires decomposition, we then prompt the LLM to autonomously break it down into several sub-questions $\{ q _ { t + 1 } ^ { j } , j = 1 , . . . , J \}$ . It is worth noting that in our decomposition approach, we

Input: Question $Q$   
Parameters: LLM $p _ { \theta }$ , natural language prompts   
$( c _ { 1 } \sim c _ { 6 } )$ , threshold $\epsilon _ { 1 }$ for Decompose, threshold   
$\epsilon _ { 2 }$ for Rethink   
Output: Rationale $R$ , Answer $A$   
Create an empty node queue $N$   
Enqueue $n _ { 0 } ( \bar { q } _ { 0 } = Q , \bar { r _ { 0 } } = N o n e , s _ { 0 } = 1 )$ into $N$   
while $N$ is not empty do Dequeue current node $n _ { t } \big ( q _ { t } , r _ { t } , s _ { t } \big )$ from $N$ if $n _ { t }$ is an end node $n _ { e n d }$ then continue else if $s _ { t } > \epsilon _ { 1 }$ then // Stage $\imath \imath$ Decompose $\{ q _ { t + 1 } ^ { j } \} \gets D e c o m p o s e ( p _ { \theta } , h _ { 1 } , l h _ { Q } , q _ { t } )$ (2) // Stage 2: Analyze $r _ { t + 1 } ^ { j }  S o l v e ( p _ { \theta } , h _ { 2 } , q _ { t + 1 } ^ { j } )$ (3) $\hat { r } _ { t + 1 } ^ { j } \gets S e l f \_ C h e c k ( p _ { \theta } , h _ { 3 } , q _ { t + 1 } ^ { j } , r _ { t + 1 } ^ { j } )$ (4) $s _ { t + 1 } ^ { j } \gets S c o r e ( p _ { \theta } , h _ { 4 } , q _ { t + 1 } ^ { j } , \hat { r } _ { t + 1 } ^ { j } )$ (5) Set $n _ { t + 1 } ^ { j } \gets ( q _ { t + 1 } ^ { j } , \hat { r } _ { t + 1 } ^ { j } , s _ { t + 1 } ^ { j } )$ (6) Enqueue $n _ { t + 1 } ^ { j }$ into $N$ // Stage 3: Rethink if $s _ { t + 1 } ^ { j } > \epsilon _ { 2 }$ then $\begin{array} { r l } & { L _ { k } ^ { - }  E x t r a c t ( p _ { \theta } , \ h _ { 5 } , \ L , q _ { t + 1 } ^ { j } ) ( 7 ) } \\ & { r ^ { \prime }  U p d a t e ( p _ { \theta } , \ h _ { 6 } , \ n _ { e } ( q , r , s ) , \hat { r } _ { t + 1 } ^ { j } ) (  } \\ & { n _ { e } ( q , r ^ { \prime } , s )  n _ { e } ( q , r , s ) ( 6 ) } \end{array}$ (8) else Enqueue $n _ { e n d }$ into $N$ end if   
end while   
$R  r _ { 0 }$   
Extract answer $A$ from $R$   
return R, A

do not pre-specify the number $J$ of sub-questions; instead, we allow LLMs to adaptively determine it based on the logic of each question. This enhances adaptability and more closely aligns with human logical characteristics when compared to existing methods like ToT [49] and GoT [3], etc. To facilitate this process, we design a heuristic-enhanced prompt that consists of a prompt head $h _ { 1 }$ and “logic heuristics” $l h _ { Q }$ . The prompt head describes the question decomposition task in natural language. This process is formulated in Eq. (2). Additionally, we validate the effectiveness of using logic heuristics, and provide detailed explanations and templates in Appendix A.1.

$$
\{ q _ { t + 1 } ^ { j } , j = 1 , . . . , J \}  D e c o m p o s e ( p _ { \theta } , h _ { 1 } , l h _ { Q } , q _ { t } ) .
$$

After decomposition, each $q _ { t + 1 } ^ { j }$ is added as a new node $n _ { t + 1 } ^ { j }$ at level $t + 1$ , with a directed edge from $n _ { t }$ to $n _ { t + 1 } ^ { j }$ (denoted as $e ^ { j } = ( n _ { t } , n _ { t + 1 } ^ { j } ) )$ . If the LLM determines that $q _ { t }$ does not require further decomposition, we create a leaf node $n _ { e n d }$ as a child of $n _ { t }$ .

# 4.2 Analyze Stage

In Analyze stage, we reason the answers for all the sub-questions $\{ q _ { t + 1 } ^ { j } \}$ at level $t + 1$ . To be specific, we first prompt the LLM to generate the essential rationale $r _ { t + 1 } ^ { j }$ for each sub-question $q _ { t + 1 } ^ { j }$ :

$$
r _ { t + 1 } ^ { j }  S o l v e ( p _ { \theta } , h _ { 2 } , q _ { t + 1 } ^ { j } ) .
$$

Here, $h _ { 2 }$ denotes the prompt head, which is a natural language sentence that asks the model to generate detailed solutions (see Appendix A.2).

After obtaining the rationales for the sub-questions, we evaluate and correct them, as large language models (LLMs) often tend to hallucinate during problem-solving [54]. Using generated rationales without verification can propagate errors, leading to incorrect outcomes. To address this issue, we develop a self-check method that promptly identifies and corrects these errors while providing a coherence score (Eq. (5)) for each node.

Specifically, we first instruct the LLM to perform a self-check on the rationale $r _ { t + 1 } ^ { j }$ generated for the sub-question $q _ { t + 1 } ^ { j }$ (see Appendix A.2 for the prompt head $h _ { 3 }$ ) to identify any potential errors. If the LLM detects errors in the original rationale $r _ { t + 1 } ^ { j }$ , it modifies the rationale to $\hat { r } _ { t + 1 } ^ { j }$ ; otherwise, the rationale is output unchanged. Take the case in Figure 2 as an example, we expect the LLM to identify the error “Each egg is sold for $\$ 3$ in $r _ { 1 } ^ { 1 }$ , and correct it to “Each egg is sold for $\$ 2$ . This process is denoted as:

$$
\hat { r } _ { t + 1 } ^ { j } \gets S e l f \_ C h e c k ( p _ { \theta } , \ h _ { 3 } , \ q _ { t + 1 } ^ { j } , \ r _ { t + 1 } ^ { j } ) .
$$

Then, we prompt the LLM to evaluate the logical coherence between the refined rationale $\hat { r } _ { t + 1 } ^ { j }$ and the question $q _ { t + 1 } ^ { j }$ , by generating a coherence score $s _ { t + 1 } ^ { j }$ (see Appendix A.2 for prompt head $h _ { 4 }$ ):

$$
s _ { t + 1 } ^ { j } \gets S c o r e ( p _ { \theta } , h _ { 4 } , q _ { t + 1 } ^ { j } , \ : \hat { r } _ { t + 1 } ^ { j } ) .
$$

The score $s _ { t + 1 } ^ { j }$ can also be obtained through voting or classification methods. Here, we specifically investigate the effectiveness of directly prompting LLMs to generate numerical values as scores.

At the end of the Analyze stage, we fill the obtained rationales and scores into nodes $n _ { t + 1 } ^ { j } ( j \geq 1 )$ :

$$
n _ { t + 1 } ^ { j } = ( q _ { t + 1 } ^ { j } , \hat { r } _ { t + 1 } ^ { j } , s _ { t + 1 } ^ { j } ) .
$$

where $s _ { t + 1 } ^ { j }$ can support the current or subsequent cycles in Rethink (4.3) and Decompose (4.1).

# 4.3 Rethink Stage

According to self-reflection theories [11, 13, 6] in cognitive science, humans constantly update and reflect on their previous reasoning results based on the current information. This allows us to correct past mistakes and ultimately achieve a consistent and stable answer. For example in Figure 2, a person might initially answer question $Q$ (“Janet’s ducks ... How much ... market?”) with the rationale $r _ { 0 }$ “She makes $9 \times 3 = \$ 27$ per day”. However,after considering responses to sub-questions $q _ { 1 } ^ { 1 }$ (“What is the selling price of one egg?”) and $q _ { 1 } ^ { 2 }$ (“How many eggs does Janet have per day?”), he/she realizes an error in $r _ { 0 }$ . The correct calculation, using the values $^ { 6 6 } 2 ^ { , 9 }$ for the price per egg and $^ { 6 6 } 9 ^ { , 9 }$ for the daily number of eggs, should be $\mathbf { \hat { \omega } } ^ { 2 } \times 9 = \mathbb { \mathbb { S } } 1 8 ^ { \mathbf { , } 9 }$ .

Nevertheless, existing methods like ToT [48] search reasoning paths based solely on preceding steps, lacking the ability to retrospectively update earlier content based on the influence of later steps. To address this, we introduce a Rethink stage that mirrors the human reflective process.

Specifically, during the rethinking process, humans first identify which existing reasoning steps may require revision. We aim to automate this by using LLMs to detect logical connections between ancestral and newly generated nodes, updating ancestral nodes based on insights from the rationales of new nodes. In our proposed “Reasoning Tree”, we essentially use information from lower-level nodes to “rethink” higher-level nodes, closely mirroring the human cognitive simplification process in problem-solving [33].

To achieve this, after obtaining node $n _ { t + 1 } ^ { j }$ in Analyze Stage, we first check its coherence score $s _ { t + 1 } ^ { j }$ (Eq. (5)). If $s _ { t + 1 } ^ { j }$ exceeds the threshold $\epsilon _ { 2 }$ , we then examine the correlation between $q _ { t + 1 } ^ { j }$ and all sub-questions above level $t$ , specifically, $\{ q _ { l } , ~ l \le t \}$ . Next, we extract a subset of $k$ most related nodes $L _ { k }$ from $L \triangleq \{ n _ { l } , l \leq t \}$ (the specific nodes to be extracted are determined by the LLM):

$$
L _ { k } \gets E x t r a c t ( p _ { \theta } , h _ { 5 } , L , q _ { t + 1 } ^ { j } ) , L _ { k } \subseteq L .
$$

where $h _ { 5 }$ is a prompt head (Appendix A.3). Next, we use the rationale $\hat { r } _ { t + 1 } ^ { j }$ of sub-question $q _ { t + 1 } ^ { j }$ to update the rationale $r$ of each extracted node $n _ { e }$ in $L _ { k }$ :

$$
r ^ { \prime } \gets U p d a t e ( p _ { \theta } , h _ { 6 } , n _ { e } ( q , r , s ) , \hat { r } _ { t + 1 } ^ { j } ) .
$$

Finally, we replace $r$ with the updated rationale $\boldsymbol { r } ^ { \prime }$ :

$$
\begin{array} { r } { n _ { e } ( q , r ^ { \prime } , s )  n _ { e } ( q , r , s ) . } \end{array}
$$

# 5 Experiments

In this section, we demonstrate the generality and effectiveness of DeAR by applying it to a wide range of tasks, including knowledge reasoning, logical reasoning and mathematical reasoning. The results across these tasks validate DeAR’s adaptability and highlight its capability to effectively tackle a diverse range of challenging reasoning tasks.

# 5.1 Experimental Setup

# 5.1.1 Datasets and Baselines

We employ the ScienceQA [28] dataset for the knowledge reasoning task. And we use StrategyQA [12] for logical reasoning that requires multiple reasoning steps. We also verify the mathematical reasoning ability of our framework by applying it to GSM8K dataset [8]. The details of these datasets are available in Appendix B.1.1.

In our main results, we compare DeAR with multiple prompt-based methods including Few-shot prompting [5], Chain-of-Thoughts (CoT) prompting [45], and state-of-the-art Tree-of-Thoughts (ToT) [49] and Graph-of-Thoughts (GoT) [3] prompting. Besides, we also list extra comparison results with another two state-of-the-art prompt-based methods Least-to-most Prompting [56] and SelfCheck [31] (see Appendix B.1.2 for all baseline details).

# 5.1.2 Implementation Details

We conduct experiments with three LLM backbones GPT-3.5 [1], LLaMA2-7B [40] and ChatGLM3- 6B [9]. For GPT-3.5, we use the OpenAI API to invoke the “gpt-3.5-turbo-1106” model. For LLaMA2-7B and ChatGLM3-6B, we load the checkpoints from huggingface23 and use the models directly without fine-tuning as the backbone.4. For each dataset, we randomly sample $10 \%$ of its training set as a validation set to select different combinations of thresholds $\epsilon _ { 1 }$ and $\epsilon _ { 2 }$ . The combination that achieves the best performance on the validation set is then used for inference on the test set. We observe that the threshold combinations obtained through this method also yield optimal inference results on the test set. In Section 5.6, we visualize the inference accuracy on the test sets across different datasets based on GPT-3.5, using diffenrent threshold combinations. The implementation and prompting templates (i.e., natural language prompts $h _ { 1 } \sim h _ { 6 }$ for Decompose, Analyze and Rethink ) are shown in Appendix A. For baselines, the settings used in the experiments are consistent with those described in the original papers. For a concise description of baselines, please refer to Appendix B.1.2.

Table 1: Overall results of our DeAR Framework on three intricate reasoning datasets. $( * : p < 0 . 0 5 )$ .   

<html><body><table><tr><td rowspan="2"></td><td colspan="3">ScienceQA</td><td colspan="3">StrategyQA</td><td colspan="3">GSM8K</td></tr><tr><td>GPT-3.5</td><td>LLaMA2</td><td>ChatGLM3</td><td>GPT-3.5</td><td>LLaMA2</td><td>ChatGLM3</td><td>GPT-3.5</td><td>LLaMA2</td><td>ChatGLM3</td></tr><tr><td>Few-shot</td><td>73.97</td><td>66.35</td><td>42.46</td><td>67.71</td><td>61.21</td><td>54.41</td><td>74.26</td><td>72.25</td><td>51.02</td></tr><tr><td>CoT</td><td>75.17</td><td>67.58</td><td>46.35</td><td>69.26</td><td>63.86</td><td>57.18</td><td>79.55</td><td>74.04</td><td>53.85</td></tr><tr><td>ToT</td><td>82.52</td><td>69.01</td><td>49.58</td><td>71.89</td><td>66.52</td><td>59.21</td><td>83.42</td><td>75.22</td><td>55.88</td></tr><tr><td>GoT</td><td>82.34</td><td>68.86</td><td>49.26</td><td>72.02</td><td>66.61</td><td>59.88</td><td>84.77</td><td>75.95</td><td>56.01</td></tr><tr><td>DeAR</td><td>83.68*</td><td>70.57*</td><td>51.08*</td><td>73.36*</td><td>68.33*</td><td>61.02*</td><td>86.82*</td><td>78.01*</td><td>58.54*</td></tr><tr><td>Least-to-most</td><td>76.61</td><td>68.02</td><td>47.45</td><td>70.55</td><td>64.43</td><td>58.36</td><td>81.25</td><td>74.67</td><td>54.21</td></tr><tr><td>SelfCheck</td><td>75.81</td><td>69.33</td><td>49.23</td><td>68.87</td><td>66.35</td><td>61.22</td><td>79.88</td><td>75.28</td><td>56.72</td></tr></table></body></html>

# 5.2 Experimental Results

We conduct experiments to verify the effectiveness of our framework DeAR, and report the results in Table 1. We use the accuracy (ACC) as metric for all three datasets. We statistically test the improvement over baselines with paired t-test, and find the improvement to be significant with $p < 0 . 0 5$ (marked with $\ast ^ { , \ast }$ ). We get the following observations. First, DeAR performs better than all baselines, which indicates it is more effective in enhancing LLMs’ reasoning ability. Second, the improvements over ToT highlight the advantage of Decompose stage which adaptively decomposes questions based on their characteristics rather than extending a fixed number of thought branches. Third, DeAR performs better than GoT which lacks rationale updating. This reflects the superiority of the Rethink stage to identify correlations between reasoning steps and update previous rationales. Besides, the accuracy increase on GSM8K is greater than ScienceQA and StrategyQA. That is probably because problems in GSM8K require longer rationales to be solved (Table 2). Furthermore, DeAR outperforms the Least-to-most [56] and SelfCheck [31] methods across all datasets. The Leastto-most method sequentially solves sub-problems derived from the decomposition without updating content that has already been generated; SelfCheck updates rationales but it does not decompose the original question. In contrast, DeAR not only generates rationales based on decomposed subquestions but also updates existing rationales in each cycle. This further underscores the necessity of the Decompose and Rethink phase in DeAR for enhancing the reasoning capabilities of LLMs.

We have also validated that DeAR enhances stronger LLMs (e.g., GPT-4) on complex reasoning tasks (e.g., MATH), as shown in Appendix. Appendix B.3 includes an ablation study on the self-check method in the Analyze stage, as its removal does not structurally impact the other stages.

# 5.3 Analyses of the Reasoning Tree

For each question $Q$ , DeAR constructs a reasoning tree $T$ to represent the reasoning process, as shown in Figure 1 (b). The structure of $T$ provides insights into the complexity of $Q$ . To analyze the nature of questions across datasets, we examine reasoning trees from three datasets

Table 2: Characteristics of $T$ in different datasets.   

<html><body><table><tr><td></td><td>ScienceQA</td><td>StrategyQA</td><td>GSM8K</td></tr><tr><td>Avg Branch</td><td>1.58</td><td>2.43</td><td>2.06</td></tr><tr><td>Avg Depth</td><td>3.62</td><td>1.96</td><td>2.55</td></tr><tr><td>Avg Length of R</td><td>66.34</td><td>61.55</td><td>85.27</td></tr></table></body></html>

using three metrics: “Avg Branch,” “Avg Depth,” and “Avg Length of $R$ .” “Avg Branch” indicates the average branching factor of $T$ , “Avg Depth” reflects the average depth of $T$ , and “Avg Length of $R ^ { \prime }$ represents the length of rationale $R$ derived from the root node $n _ { 0 }$ upon tree completion, e.g., $R = r _ { 0 }$ :“She makes $9 ^ { * } 2 { = } \$ 18$ per day” in Figure 2.

Using GPT-3.5 as the backbone, results in Table 2 reveal the following: ScienceQA questions have the highest “Avg Depth” and lowest “Avg Branch,” indicating fewer sub-questions per Decompose stage but more rounds required. StrategyQA questions have the lowest “Avg Branch” but the highest “Avg Depth,” suggesting fewer Decompose rounds but more sub-questions per round. For GSM8K, the root node $n _ { 0 }$ has longer rationales $R$ , suggesting that these questions require more extensive explanations than those in the other datasets.

Table 3: ROSCOE evaluation results of rationales generated by Tree-of-Thoughts (ToT), Graph-of-Thoughts (GoT) and DeAR on different datasets. $\mathbf { S } { \mathbf { C } } = { \mathbf { S } }$ ource-Consistency; RA $\mathbf { \Sigma } = \mathbf { \Sigma }$ Reasoning Alignment.   

<html><body><table><tr><td rowspan="2"></td><td colspan="2">ScienceQA</td><td colspan="2">StrategyQA</td><td colspan="2">GSM8K</td></tr><tr><td>SC</td><td>RA</td><td>SC</td><td>RA</td><td>SC</td><td>RA</td></tr><tr><td>ToT</td><td>0.44</td><td>0.31</td><td>0.47</td><td>0.33</td><td>0.56</td><td>0.41</td></tr><tr><td>GoT</td><td>0.42</td><td>0.35</td><td>0.44</td><td>0.38</td><td>0.53</td><td>0.45</td></tr><tr><td>DeAR</td><td>0.48</td><td>0.42</td><td>0.52</td><td>0.43</td><td>0.58</td><td>0.50</td></tr></table></body></html>

![](images/fdc298769a0f8aa029962094f3d90f0cb17e869cca1b1f192af703ba5dcf327c.jpg)  
Figure 3: The distributions of annotators’ selections. More annotators considered DeAR’s rationales to be more logical.

# 5.4 Logical Coherence of the Generated Rationales

We assess the logical coherence of rationales generated by DeAR using both automatic and human evaluation methods. For automatic metrics, we apply the Source-Consistency” (SC) and Reasoning Alignment” (RA) from the ROSCOE evaluation suite [14]. SC measures logical entailment between question and rationale, while RA evaluates alignment with ground truth. As shown in Table 3, DeAR outperforms ToT and GoT on all datasets. For human evaluation, 100 questions were sampled from each dataset, with annotators selecting the most logical rationale among those generated by ToT, GoT, and our method (details in Appendix B.4). Results in Figure 3 confirm that DeAR (using GPT-3.5) produces rationales with superior logical coherence compared to ToT and GoT.

# 5.5 Effectiveness of Rethink

In Rethink stage, our DeAR employs the same backbone LLMs to determine which nodes’ rationales need to be updated. To validate its effectiveness, based on GPT-3.5, we compare our method with “Random Update” method which randomly selects nodes to update at different proportions. The results in Table 3 demonstrate that, compared to “Random Update”, our

Table 4: Comparisons of ACCs between different portions of “Random Update” and DeAR.   

<html><body><table><tr><td>Random Update</td><td>ScienceQA</td><td>StrategyQA</td><td>GSM8K</td></tr><tr><td>0%</td><td>82.77</td><td>72.84</td><td>85.09</td></tr><tr><td>20%</td><td>81.77</td><td>72.21</td><td>83.96</td></tr><tr><td>40%</td><td>82.59</td><td>73.03</td><td>84.35</td></tr><tr><td>60%</td><td>82.06</td><td>72.29</td><td>85.07</td></tr><tr><td>80%</td><td>81.49</td><td>72.04</td><td>86.01</td></tr><tr><td>100%</td><td>81.16</td><td>71.79</td><td>85.32</td></tr><tr><td>DeAR</td><td>83.68</td><td>73.36</td><td>86.82</td></tr></table></body></html>

method performs better in terms of accuracy. Additionally, unlike approaches that require a $100 \%$ update of all generated rationales, DeAR’s targeted updates allow the model to autonomously select nodes that need refinement, thus minimizing unnecessary inference.

# 5.6 Combinations of Thresholds

In this subsection, we visualize the impact of different combinations of threshold values $\epsilon _ { 1 }$ and $\epsilon _ { 2 }$ on the inference accuracy of DeAR (with GPT3.5 backbone) across the test sets of all three datasets. $\epsilon _ { 1 }$ and $\epsilon _ { 2 }$ are set for the Decompose stage (Section 4.1) and Rethink stage (Section 4.3), respectively, with their value combina

![](images/3dbd6953e32809e51aa34cc3aeb5c926c0c050b16fa635b8f09e540d3e3f1277.jpg)  
Figure 4: Combinations of threshold values $( \epsilon _ { 1 } , \epsilon _ { 2 } )$ and corresponding ACCs on test sets (GPT-3.5 backbone).

tions selected based on performance on the validation set (Section 5.1.2). We observe from Figure 4 that, DeAR achieves the highest accuracy when setting $\epsilon _ { 1 } = 0 . 4$ and $\epsilon _ { 2 } = 0 . 6$ for ScienceQA and StrategyQA. For GSM8K, the highest accuracy is obtained with $\epsilon _ { 1 } = 0 . 4$ and $\epsilon _ { 2 } = 0 . 4$ . The threshold combinations that optimize DeAR’s performance on the test set are consistent with those obtained from the validation set (e.g., Val: $\epsilon _ { 1 } = 0 . 4$ ; $\epsilon _ { 2 } = 0 . 6$ for ScienceQA), demonstrating the validity of the value selection method. Additionally, the smaller optimal $\epsilon _ { 2 }$ value for GSM8K suggests that tackling GSM8K problems requires a more frequent or active rethinking process compared to ScienceQA and StrategyQA. This difference highlights the varying nature of reasoning demands across different tasks, where the threshold tuning helps adapt DeAR’s reasoning process accordingly.

![](images/fe8a3a32188feb98ac9e33bfee0ebbb87f9a7d2e81642a4de29ebce782a01f9a.jpg)  
Figure 5: Efficiency comparison between DeAR and variants of ToT/GoT.

# 5.7 Efficiency

Compared to the rationale extension in ToT and GoT, DeAR incorporates question decomposition and rationale updating. Thus, will the efficiency of reasoning be affected? To investigate this, we use ChatGLM3-6B as the backbone model and measure the average inference time per question (seconds/question) and accuracy (ACC) for each method. The results are in the form of scattered points as shown in Figure 5. We set the fixed branch numbers and depths for these variants of ToT and GoT (e.g., $\scriptstyle \mathbf { b } = 3$ , $\mathrm { { d } = 4 }$ ), and compare them with DeAR. In ToT/GoT, we set “b” and “d” (integers) as close to DeAR’s average values as possible to ensure fairness. We can observe that points closer to the upper-left corner, and farther away vertically from the diagonal, represent methods that achieve a better trade-off between reasoning accuracy and time. The points corresponding to DeAR clearly exhibit this characteristic, hence we can conclude that it has higher efficiency. Moreover, in Appendix B.5, to further validate this conclusion, we measured the average number of API calls made by DeAR, ToT, and GoT per question in the ScienceQA dataset using GPT-3.5, as well as their reasoning accuracy. DeAR consistently requires fewer API calls on average to solve a question, while simultaneously achieving higher accuracy.

# 6 Conclusion

In this study, we introduced DeAR (Decompose-Analyze-Rethink), an innovative framework designed to mimic human reasoning patterns in tackling intricate problems by constructing a reasoning tree in a top-down, iterative manner. DeAR’s key approach lies in systematically decompose a question into simpler, manageable sub-questions, each represented as a node within the reasoning tree. This approach is coupled with a Decompose-Analyze-Rethink cycle, in which the rationale at each node is generated, evaluated, and refined through feedback loops. Specifically, the Decompose stage applies logic heuristics to decompose the original question, the Analyze stage produces and selfchecks rationales, and the Rethink stage integrates these insights by updating parent nodes based on child-node feedback. Extensive experimental evaluations across reasoning benchmarks ScienceQA, StrategyQA, and GSM8K demonstrate that DeAR not only improves reasoning performance across different large language models (LLMs) (e.g., GPT-3.5, LLaMA2, ChatGLM3) but also surpasses current state-of-the-art methods like Tree-of-Thoughts (ToT) and Graph-of-Thoughts (GoT) in logical coherence and accuracy. Unlike rationale-extension and path-search methods, DeAR’s rationale update mechanism enhances logical consistency by iteratively refining previously generated rationales, achieving more accurate and interpretable results while reducing the risk of error propagation. Additionally, compared to ToT and GoT, DeAR strikes an optimal balance between reasoning accuracy and inference time, further improving efficiency. Through case studies, we can also demonstrate that our method produces more interpretable reasoning process (due to space limit, we present our Case Study section in Appendix B.6.