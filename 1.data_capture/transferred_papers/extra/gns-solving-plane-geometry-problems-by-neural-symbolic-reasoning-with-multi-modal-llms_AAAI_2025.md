# GNS: Solving Plane Geometry Problems by Neural-Symbolic Reasoning with Multi-Modal LLMs

Maizhen Ning\*1,2, Zihao Zhou\*1,2, Qiufeng Wang1†, Xiaowei Huang2, Kaizhu Huang3

1School of Advanced Technology, Xi’an Jiaotong-Liverpool University 2University of Liverpool 3Duke Kunshan University maizhen.ning16@student.xjtlu.edu.cn, qiufeng.wang@xjtlu.edu.cn

# Abstract

With the outstanding capabilities of Large Language Models (LLMs), solving math word problems (MWP) has greatly progressed, achieving higher performance on several benchmark datasets. However, it is more challenging to solve plane geometry problems (PGPs) due to the necessity of understanding, reasoning and computation on two modality data including both geometry diagrams and textual questions, where Multi-Modal Large Language Models (MLLMs) have not been extensively explored. Previous works simply regarded a plane geometry problem as a multi-modal QA task, which ignored the importance of explicitly parsing geometric elements from problems. To tackle this limitation, we propose to solve plane Geometry problems by NeuralSymbolic reasoning with MLLMs (GNS). We first leverage an MLLM to understand PGPs through knowledge prediction and symbolic parsing, next perform mathematical reasoning to obtain solutions, and last adopt a symbolic solver to compute answers. Correspondingly, we introduce the largest PGPs dataset GNS-260K with multiple annotations including symbolic parsing, understanding, reasoning and computation. In experiments, our Phi3-Vision-based MLLM wins first place on the PGPs solving task of MathVista benchmark, outperforming GPT-4o, Gemini Ultra and other much larger MLLMs. While LLaVA-13B-based MLLM markedly exceeded other close-source and open-source MLLMs on the MathVerse benchmark and also achieved the new SOTA on GeoQA dataset.

Project — https://github.com/ning-mz/GNS

# Introduction

Large Language Models (LLMs) have recently demonstrated impressive versatility in natural language understanding and generation (Zhao et al. 2023; Touvron et al. 2023; Chang et al. 2023). Consequently, numerous researchers have successfully explored LLMs in solving math word problems (MWP) (Zong and Krishnamachari 2023; Fu et al. 2023; Zhou et al. 2023a,b) and reached a level of expertise comparable to humans (Ahn et al. 2024; Yue et al.

![](images/4cf017b15763007c668d2895b42b2c8bedfdde2c4b71c47b9054e6688e89e768.jpg)  
Figure 1: Performance improvements with different base MLLMs trained by our GNS framework. Experiments were conducted on MathVista-testmini plane geometry problem solving task (GPS). Our method achieves competitive and even higher performance than GPT-4o (2024-05-13).

2024). However, it is more challenging to solve plane geometry problems (PGPs) with two modality data including both geometry diagrams and textual questions. Despite significant recent progress of Multi-Modal Large Language Models (MLLMs) (Yin et al. 2023), there has been surprisingly limited exploration of MLLMs in solving plane geometry problems.

Solving PGPs usually requires mathematical reasoning and visual-textual understanding, which has gained attention for an extended period (Chou, Gao, and Zhang 1996). Many works focused on designing complicated neural networks and several researchers constructed plane geometry problem datasets (Chen et al. 2021; Lu et al. 2021; Cao and Xiao 2022) to facilitate research. With the recent progress in MLLMs (Zhu et al. 2024; Liu et al. 2023b; Achiam et al. 2023), researchers have started to evaluate the capability of MLLMs in solving PGPs (Lu et al. 2024b).

Based on the evaluation (Lu et al. 2024b; Zhang et al. 2024a; Li et al. 2024a; Zhou et al. 2024), we find that most of the current MLLMs struggle with solving PGPs, where the capabilities of MLLMs are under-explored for two main reasons. One is that lack of PGP-focused fine-tuning method (i.e., model issue), and the other one is the lack of largescale high-quality annotated data of PGPs (i.e., data issue).

Recently, (Gao et al. 2023a) made a first effort to introduce a large-scale dataset Geo170K by utilizing data augmentation on existing datasets, and further trained G-LLaVA with the dataset. However, Geo170K simply extends the data size of existing datasets, without the extension to the characteristics of PGPs. We argue that solving PGPs involves multiple tasks including problem understanding, reasoning and computation. In addition, solving PGPs by natural language reasoning is often challenged by inaccurate arithmetic computation resulting in wrong answers (Zhou et al. $2 0 2 3 \mathrm { a }$ ; Chen et al. 2023; Gao et al. 2023b). In summary, both model and data issues have not been solved, resulting in the insufficient exploration of MLLMs in solving PGPs.

To tackle the model issue, we propose a Geometry NeuralSymbolic method (GNS), enabling MLLMs to solve plane geometry problems through knowledge prediction, symbolic parsing, problem reasoning and symbolic computation. In detail, we first leverage MLLMs to parse geometry diagrams and text into textual symbolic clauses, which efficiently describe structural geometry elements. Clauses are defined with highly syntactic structures, which naturally contain less redundant information. Therefore, clauses are particularly effective for representing fine-grained and multi-level geometry elements in plane geometry problems (Zhang, Yin, and Liu 2023). Meanwhile, we also leverage MLLMs to predict corresponding geometry knowledge related to the given PGP. Next, the MLLM model performs mathematical reasoning based on given diagrams, parsed clauses and predicted knowledge to obtain the solution. Last, we adopt a symbolic solver to compute the solution and obtain answers. Such an approach allows GNS to conduct precise mathematical computations, avoiding the LLMs’ shortcomings in computation, and thereby output more accurate answers.

To handle the data issue, we construct a multi-task plane geometry problem related dataset GNS-260K, which is the largest PGPs dataset so far. To better explore the MLLM’s capability, GNS-260K consists of three related sub-tasks, including geometry knowledge prediction, symbolic parsing and symbolic-based problem reasoning. All diagrams and base problems are from the existing PGP datasets including the training set of both PGPS9K (Zhang, Yin, and Liu 2023) and ${ \mathrm { G e o Q A } } +$ (Cao and Xiao 2022), but we extend more data by data augmentation and more symbolic annotations. For example, we leverage GPT-4 to generate the natural language reasoning descriptions for problems that do not have such annotation (e.g., samples in the dataset PGPS9K). Figure 1 shows the performance improvements of various base MLLMs trained with our GNS framework and tested on MathVista-testmini geometry problem solving task. Compared with the model’s original performance, our method significantly improves the accuracy of all MLLMs with different scales of parameters. Specifically, with LLaVA-13B based GNS model trained on the proposed GNS-260K, the model achieves a new SOTA accuracy on GeoQA dataset, and Phi3-Vision based GNS wins the first place on the PGPs solving task in MathVista, markedly outperforming GPT-4o (OpenAI 2024) and Gemini Ultra (Google 2023). Our GNS-MLLMs also markedly outperformed many other MLLMs on MathVerse (Zhang et al. 2024b) testmini set.

The contributions of this paper are summarized:

We propose GNS, a neural-symbolic MLLM framework for solving plane geometry problems by symbolic parsing, reasoning and computation.

We propose the largest plane geometry problem dataset GNS-260K, which includes multiple related sub-tasks to enhance the PGPs solving capability of MLLMs. GNS-260K also provides a unified symbolic solving system for different source problems.

With only 13B parameters, our method achieves leading performance on the MathVista Geometry Problem Solving task and outstanding accuracy on MathVerse. Meanwhile, our method is effective for various MLLMs.

# Related Work

Plane Geometry Problem Solving Early works in solving plane geometry problems (PGPs) focused on using manually designed rules and the proposed dataset scale is relatively small (Seo et al. 2015, 2014), which limited the generalization ability. Recent methods can broadly be categorized into two types: neural-based and symbolic-based. The neural-based methods like NGS (Chen et al. 2021) tackle PGP through a visual question-answering approach and use specialized programs to represent the solving process. However, these methods are coarse-grained at geometry diagram understanding, which directly extract visually hidden features to perform multi-modal fusion(Cao and Xiao 2022; Ning et al. 2023). Meanwhile, UniGeo further extended neural-based methods to the proving task of PGPs (Chen et al. 2022). The symbolic-based methods like Inter-GPS (Lu et al. 2021) and FormalGeo (Zhang et al. 2023b) parse the problem diagram and text into the formal language to obtain a unified problem representation, then apply complex rule-based reasoning through complicated manually predefined path search and condition matching processes (Zhang, Yin, and Liu 2023; Zhang et al. 2022). Despite the above works leading research on PGPs to a new level, their ability to process various types of PGPs is constrained due to the limited data scale and pre-defined reasoning rules. Meanwhile, previous methods are not able to generate natural language solving descriptions, making it difficult for people to follow the problem solving process (Li et al. 2024b). To expand the scale of datasets and enhance the PGPs solving capabilities of MLLMs, (Gao et al. 2023a) introduced Geo170K, by employing ChatGPT for data augmentation on existing datasets and finetuned G-LLaVA with Geo170K. However, G-LLaVA simply tackles PGPs as a general QA task, lacking the ability to explicitly comprehend the geometry elements in the diagram during problem solving. Moreover, like other MLLMs, G-LLaVA also struggles with precise mathematical computation, particularly in problems requiring complicated computations. To address these issues, we propose GNS, learning to solve the problem through knowledge prediction, symbolic parsing, reasoning and computation.

Multi-Modal Large Language Model The success of Transformer (Vaswani et al. 2017) architectures and pretraining techniques has greatly contributed to the development of LLMs (Min et al. 2023; Zhang et al. 2023a; Ouyang et al. 2022; Chiang et al. 2023; Du et al. 2022), particularly evidenced by the development of ChatGPT (Achiam et al. 2023). With LLMs’ powerful ability on language understanding and generation, research has expanded to explore multi-modal LLMs, aiming to further enhance LLMs application in diverse, complex tasks across various modal of information (Yin et al. 2023; Ding et al. 2021). Moreover, close source models like GPT4-V and Geimini were trained on vast datasets with extensive model parameters, and have advanced research on MLLMs to new heights (Achiam et al. 2023; Google 2023). However, MLLM models still face challenges in comprehending geometry diagrams to effectively solve plane geometry problems (Gao et al. 2023a).

![](images/0a2ed277af3f0c17019a5fdc76d38eb18f7f5773e356d34cd499c4b17b1026e7.jpg)  
Figure 2: The overall framework of the proposed GNS.

# Methodology

We introduce GNS, a neural-symbolic framework for MLLMs to solve plane geometry problems. To enhance the solving capability, GNS consists of four main modules as shown in Fig. 2 including knowledge prediction, symbolic parsing, reasoning and computation. Given a plane geometry problem $P = \left[ Q , I \right]$ ( $Q$ : textual question and $I$ : geometry diagram image), we first leverage an MLLM to predict related geometry knowledge, meanwhile, we parse the problem to obtain symbolic clauses, next we utilize the MLLM to conduct problem reasoning to obtain the solution, last we rely on an external symbolic solver to compute the solution to obtain the final answer. Each stage is executed through corresponding prompts. In the following, we will describe each component of GNS in detail.

# Knowledge Prediction

When humans solve plane geometry problems, they usually first categorize problems based on their relevant geometry knowledge. For example, if a problem involves properties of circles, humans will review and apply knowledge related to circles, such as the properties of tangents or the theorem for inscribed angles. Inspired by such a mechanism, we propose to leverage MLLM to specify the related geometry knowledge of a given problem, which will be used as a part of instruction during the reasoning process, to facilitate MLLM’s utilization of relevant geometry knowledge and guide the reasoning in a more structured and informed manner. In addition, such a mechanism enables the MLLM to tailor the approach for each specific category, improving the accuracy and efficiency in handling complex geometrical challenges. The knowledge prediction process can be represented as

$$
T _ { K } = M L L M ( [ I \oplus P _ { K } \oplus Q ] ) ,
$$

where $T _ { K }$ is the predicted knowledge, $P _ { K }$ is the instruction prompt of knowledge prediction, $I$ is the problem diagram, $Q$ is question text and $\oplus$ denotes concatenation.

# Symbolic Parsing

As most of the key information is represented in the diagram while the problem text just simply gives the solving target (e.g. ”Find RT” in Fig. 2.), it is crucial to understand the information in the geometry diagram comprehensively. However, there are few geometry diagrams in the pretraining datasets of MLLMs (Liu et al. 2023b), resulting in the weak capability of diagram understanding. Therefore, we propose to make MLLM learn to parse a plane geometry problem into symbolic clauses (e.g., the result of ”Symbolic Parsing” in Fig. 2). Unlike straightforward using encoded visual features, symbolic clauses naturally have syntactic structures that can explicitly describe fine-grained and multi-level geometry elements in the problem. There are two types of symbolic clauses: semantic and structural. Semantic clauses describe the semantic relations between non-geometric and geometric primitives, while structural clauses represent the connection relations among geometric primitives.

The parsing process not only helps the model to understand the geometric elements in a diagram, but also facilitates subsequent reasoning, based on those explicitly represented information. This is achieved by explicitly using the parsed symbolic clauses as the input to the model, where MLLMs are good at processing structural texts (Li et al. 2023). The problem symbolic parsing process can be formulated as

$$
T _ { P } = M L L M ( [ I \oplus P _ { P } \oplus Q ] ) ,
$$

where $P _ { P }$ is the instruction prompt of symbolic parsing, instructing the MLLM to perform the symbolic parsing task. The parsed output $T _ { P }$ is a paragraph of text that contains

the original question and a sequence of symbolic clauses describing the geometric elements in the problem.

# Problem Reasoning

With the predicted knowledge and parsed symbolic clauses, we instruct MLLM to perform reasoning on the problem. By integrating the problem diagram $I$ , geometry knowledge $T _ { K }$ and parsed symbolic clauses $T _ { P }$ as the model input, we formulate the reasoning process by

$$
T _ { R } = M L L M ( [ I \oplus P _ { R } \oplus T _ { K } \oplus T _ { P } ] ) ,
$$

where $P _ { R }$ represents the instruction prompt of problem reasoning. For the easy operation of following symbolic computation, we transform the output $T _ { R }$ into the specified format $R _ { S }$ : ”Solution Program:xxx”, which will be input to an external symbolic solver to obtain answers. It is noted that GNS can also generate natural language-based CoT solving descriptions, which helps people understand the solution.

# Symbolic Computation

Rather than relying on the MLLM to do math computation in natural language, we designed a symbolic computation module for GNS to perform precise numerical computations based on symbolic solutions. The solution consists of several deduction steps, where each step is composed of an operator and several values extracted from the problem or constant values given by the MLLM. Each operator represents a geometry theorem or axiom, where the corresponding values are organized as equations by following the theorem. As shown in Fig. 2, the model can compute step by step to obtain the final answer 10. The computation of symbolic solution can be formed as

$$
V = S y m C a l ( R _ { S } ) ,
$$

where SymCal is the symbolic computation tool and $V$ is the final numeric answer, and we use a Python library SymPy (Meurer et al. 2017) as the symbolic computation tool.

# MLLM Training

Our GNS is a general framework that various transformerbased MLLMs can be adopted as the base model. We fully finetune the base model by mixing all training data with the conventional language modeling loss function:

$$
\begin{array} { r l r } {  { \mathcal { L } ( I , T _ { \mathrm { o u t } } , T _ { \mathrm { i n } } ) = } } \\ & { } & { - \sum _ { i = 1 } ^ { L } \log p [ T _ { \mathrm { o u t } } ^ { i } \ | \ \mathbf { M L L M } ( I , t _ { \mathrm { o u t } } ^ { ( < i ) } , T _ { \mathrm { i n } } ) ] , } \end{array}
$$

where $I$ is the geometry diagram, $T _ { i n }$ is the input text and $T _ { o u t }$ is the model output text and $L$ is the length of output.

# GNS-260K

In this paper, we construct a new plane geometry problem dataset: GNS-260K. To extensively explore the PGP-solving capability of MLLMs, our proposed GNS-260K has multiple annotations for different PGP-related tasks including knowledge prediction, symbolic parsing, reasoning explanation, and symbolic solutions. Figure 3 shows examples of the multiple tasks in our dataset. We construct GNS-260K based on two existing PGP datasets: PGPS9K (Zhang, Yin, and Liu 2023) and ${ \mathrm { G e o Q A } } +$ (Cao and Xiao 2022). Moreover, we leverage 88K data samples from Geo170K (Gao et al. 2023a) as the augmented base QA data of ${ \mathrm { G e o Q A } } +$ training set. In addition, none of the leveraged data was directly used as we further annotated them with symbolicrelated labels and only adopted the training set to make the evaluation fair. Meanwhile, GNS-260K do not add samples in Geometry3K (Lu et al. 2021) and GeoQA (Chen et al. 2021) because both are covered by PGPS9K and ${ \mathrm { G e o Q A } } +$ . In summary, GNS-260K based on 9,426 unique diagrams, with augmented to 18,852 knowledge prediction samples, 86,732 samples for symbolic parsing, and 154,433 for problem reasoning. Finally, our proposed GNS-260K not only explores the data size to be the largest PGP dataset so far but also provides high-quality annotations for multiple related sub-tasks.

# Knowledge Prediction

To cooperate geometry knowledge prediction process in GNS, our dataset is particularly designed knowledge prediction training data. Each problem in our dataset is accompanied by both a prediction instruction and a corresponding knowledge annotation, where a problem may relate to more than one type of knowledge. Such instruction guides the MLLM to assess the problem diagram and text comprehensively and then identifies the specific knowledge that is necessary for the reasoning process. Meanwhile, the knowledge of each problem will also be combined with the problem itself in the problem reasoning samples.

# Symbolic Parsing

Symbolic parsing is the key process of GNS, therefore we also manually annotated symbolic clauses for problems in GNS-260K. As introduced in Sec. , parsed symbolic clauses consist of two types: semantic and structural. To further enhance the symbolic parsing capability of the model, we propose several sub-tasks: semantic clauses parsing, structural symbolic parsing, clauses belonging and general symbolic parsing. An example is shown in Fig. 3. To enable the model to precisely differentiate the geometry meaning of semantic and structural clauses, we have designated two sub-tasks to train the model separately to parse clauses, thereby facilitating the model’s comprehensive ability on symbolic parsing. In addition, we randomly generate some clauses unrelated to the problem serving as the negative samples for the task of clauses belonging, while the original clauses are set as positive samples. Such clauses belonging task aims to further enhance the model’s capability in symbolic understanding. Lastly, we design a general symbolic parsing task, enabling the model to completely parse the problem and integrate the parsed clauses with the problem question together, thereby creating a comprehensively parsed output. Each subtask contributes a more comprehensive understanding and enhances the model’s symbolic parsing capability.

Chords AC, DF are equidistant from the center. If the radius of circle G is 26, find DE.

![](images/b191fac27ee8fa6422b103cd9fe76e04ef462a55bbd8a16acd5ccaa87b08aa94.jpg)  
Figure 3: Examples of the multiple tasks in our GNS-260K. Tasks are described at the left bottom of the figure.

B Knowledge Prediction Structure Clause Parsing 1 Semantic Clause Parsing 8 Clause Belonging 绿 Symbolic Parsing Problem Reasoning

Q: What geometric theorem knowledge is needed to solve this problem? A: Circle Chord Q: Parse the structure clauses in the geometry problem A: line A B C, line A G, line B G, line D E F, line E G, line F G, circle G lies on A C D F. 中 Q: Parse the semantic clauses in the geometry problem   
画 A: B G = E G = 10 Q: Does this clause belong to this diagram: line A F A: No, line A F is not belong to this diagram.   
国 Q: Parse the following geometry problem. A: line A B C, line A G, line B G, …B G = E G = 10 , find D E Q: Use the following geometry knowledge … Question: Chords AC and A: Using the Pythagorean theorem (Gougu in the solution program) on the right triangle   
品 formed by GB, EG Solution Program: Gougu V0 10 26 Get V0

# Reasoning Annotation

# Experiments

A detailed and coherent problem reasoning description can greatly help humans understand how to solve PGPs. Similarly, LLMs with detailed reasoning descriptions can perform more accurate reasoning on solving math problems, such as Chain-of-Thought (Wei et al. 2022). However, plane geometry problem datasets like PGPS9K do not annotate any natural language-based solving description. To overcome this issue, we leverage GPT-4 (gpt-4-1106-preview) with its significant reasoning ability to generate the solving description of problems. In particular, we utilize the parsed problem symbolic clauses with the corresponding symbolic solution as model input, instructed by a description prompt to make GPT-4 translate the solution into natural language. We also use rule-based methods to convert symbolic solution to a brief natural language description with step-wise to help GPT-4 better understand the solution process.

# Symbolic Solution

The existing PGPs datasets are annotated with specific solution programs, which require dataset-specific solution execution functions to compute results. This limitation leads to a barrier to the unification of different datasets. In addition, these programs require to convert the numeric values given from the problem into several substitute variables. For example, given a math word problem: ”Calculate the area of a rectangle with sides 2.3 and $4 . 5 ^ { \prime \prime }$ . Previous solution program should be: [Multiply N0 N1 Get V0], where $N O$ and N1 represents 2.3 and 4.5 in the problem. This mechanism requires the complicated pre-processing of the input problem, which will lead to wrong identification and then fail to solve the problem. To address such issues, we annotate ${ \mathrm { G e o Q A } } +$ and PGPS9K into a unified symbolic solution system. The above solution is re-annotated as: [Multiply $2 . 3 4 . 5$ V0 Get V0]. With the symbolic computation tool, the new solution can obtain $2 . 3 ^ { * } 4 . 5 { = } V O$ and finally get the value of $V 0$ is 10.35.

# Datasets and Settings

We select 5 different pre-trained MLLMs as the base model of our method, the scale of model parameters are dispersed from 1.3B to 13B. Specifically, we test our method with MLLMs including DeepSeek-VL-1.3B (Lu et al. 2024a), Phi3-Vision-128k-Instruct-4.2B (Abdin et al. 2024), MiniCPM-Llama3-V2.5-8B (OpenBMB 2024), LLaVA1.5-7B and LLaVA-1.5-13B (Liu et al. 2023a). The MLLMs were trained on the proposed GNS-260K dataset. To obtain a standard performance measurement with different MLLMs rather than simply test on a single base plane geometry problem dataset, we selected two benchmarks including MathVista (Lu et al. 2024b) and MathVerse (Zhang et al. 2024b). Specifically, we evaluate the ”Geometry Problem Solving” task from the test-mini set of MathVista (GPS) and the entire testmini set of MathVerse. Notably, we confirm that our training data do not have any examples that are included in the MathVista and MathVerse. Furthermore, we also evaluate the GNS-MLLMs on the test set from the base dataset GeoQA and Geometry3K (Geo3K). We fully finetune the MLLMs with learning rate $5 e ^ { - 5 }$ for DeepSeek-VL-1.3B and $3 e ^ { - 5 }$ for the others, 2 epochs training, batch size 8 per GPU and trained on 4 NVIDIA A800 80GB GPUs.

# Experimental Results

MathVista Benchmark. MathVista is a benchmark proposed to evaluate the mathematical reasoning ability of LLMs including multiple math-related tasks like MWP and PGP solving (Lu et al. 2024b), where the PGP solving sub-task is constructed by 4 different geometry problem datasets. The experiment results are shown in Table 1. Compared to the original performance of each base model, all the trained GNS-MLLMs have significant improvements and outperformed the human baseline. With the smallest DeepSeek-VL-1.3B, our method achieves competitive performance with GPT-4-Turbo and G-LLaVA-13B. The other GNS-MLLMs have better accuracy even higher than GPT4o which is one of the most powerful MLLMs currently, but our models have much fewer parameters. These results also indicate that general MLLMs lack of ability to understand geometry elements, verifying the importance of symbolic parsing on plane geometry problems in our method. Surprisingly, with Phi3-Vision- $1 2 8 \mathrm { k \Omega }$ -instruct as the base model of GNS, the accuracy reaches $6 3 . 9 \%$ which is the leading performance of the MathVista-testmini GPS task for now.

Table 1: Accuracy comparison on the Geometry Problem Solving (GPS) task in MathVista testmini (Lu et al. 2024b), and all testmini problems in MathVerse (Zhang et al. 2024b). \* denotes the result is implemented by us using the official prompt setting from MathVista and MathVers, the others are from MathVista and MathVerse website on 2024-08-14. Some results are not available in MathVerse is limited by high API consumption of evaluation.   

<html><body><table><tr><td>Model</td><td>MathVista</td><td>MathVerse</td></tr><tr><td>Human</td><td>48.4</td><td>64.9</td></tr><tr><td>QWen-VL-Plus QWen-VL-Max Multimodal Bard Gemini Pro</td><td>CLOSE SOURCE MLLMS 38.5 47.1 40.4</td><td>11.8 25.3 23.5</td></tr><tr><td>Gemini-1.5-Flash GPT-4V GPT-4o (2024-05-13)</td><td>51.8* 50.5 60.6* OPENSOURCEMLLMS</td><td>39.4 11.0</td></tr><tr><td>miniGPT-v2-7B LLaVA-1.5-7B LLaVA-1.5-13B G-LLaVA-7B G-LLaVA-13B DeepSeek-VL-1.3B Phi3-V-128k-Instruct-4.2B</td><td>29.2 25.0* 30.3* 53.4 56.7 19.7* 40.9*</td><td>7.6 16.6 16.9 4.9* 12.4*</td></tr><tr><td>MiniCPM-Llama3-V2.5-8B OURS GNS-MLLMS</td><td>43.3*</td><td>15.6* 13.5</td></tr><tr><td>DeepSeek-VL-1.3B MiniCPM-Llama3-V2.5-8B LLaVA-1.5-7B LLaVA-1.5-13B</td><td>55.3 61.1 62.0</td><td>23.2 22.9 27.1</td></tr></table></body></html>

MathVerse Benchmark. MathVerse is a benchmark intended to evaluate the MLLMs on solving math problems with various types of diagrams (Zhang et al. 2024b). To be noticed, MathVerse does not specifically focus on plane geometry problems, which mix different types of problems like ‘Functions’ (i.e. Analytic Geometry). The testmini set in MathVerse has 3,940 samples in total and $6 4 . 7 \%$ of them are plane geometry problems. Table 1 shows the experiment results of different MLLMs. Once again, all the trained GNSMLLMs demonstrated increased performance compared to the corresponding baseline model. Compared to GPT-4V, all other models including GNS-MLLMs still has a certain gap. This is because GPT-4V has a considerable number of model parameters and is trained with an extensive amount of data samples which gains the advantage in solving other types of problems. However, compared to the rest of closesource MLLMs, our GNS-MLLMs achieved higher accuracy than Gemini Pro and QWen-VL-Max with much fewer parameters. Moreover, our GNS-MLLMs also outperformed open-source MLLMs including G-LLaVA. This is due to our method benefits from the entire symbolic-related reasoning process on both problem diagram and text, while other models still mainly rely on problem text descriptions, and our symbolic computation module is able to perform accurate calculations to output results.

Table 2: Performance comparison on GeoQA (Chen et al. 2021) and Geometry3K (Lu et al. 2021) dataset with problem solving accuracy $( \% )$ . ’N/A’ means the method is not able to solve the dataset.   

<html><body><table><tr><td>Model</td><td>GeoQA</td><td>Geo3K</td></tr><tr><td>Human TRADITIONAL METHODS</td><td>92.3</td><td>56.9</td></tr><tr><td>FiLM-BART (Lewis etal.2020) DualGeoSolver (Xiao et al. 2024) Inter-GPS (Lu et al. 2021)</td><td>35.3 65.2 N/A</td><td>33.0 N/A 57.5</td></tr><tr><td>G-LLaVA-7B G-LLaVA-13B Gemini-1.5-Flash GPT-4o (2024-05-13)</td><td>63.7 67.0 42.4 58.4</td><td>28.2 29.8 45.0 49.6</td></tr><tr><td>DeepSeek-VL-1.3B Phi3-V-128k-Instruct-4.2B MiniCPM-V2.5 LLaVA-1.5-7B LLaVA-1.5-13B</td><td>54.2 64.2 68.0 65.8 68.3</td><td>50.9 48.0 48.0 51.4 53.8</td></tr></table></body></html>

GeoQA and Geometry3K. We test GNS-MLLMs on two well-known datasets: GeoQA and Geometry3K (Geo3K), the results are shown in Table 2. Firstly, we can see that traditional Transformer-based methods like FiLM-BART make it hard to solve the problem with limited reasoning ability. Meanwhile, the generalization ability of previous PGPspecialized models, DualGeoSolver and Inter-GPS, is limited by the specific datasets they were designed for. In GeoQA, our method outperforms Gemini-1.5-Flash, despite using DeepSeek-VL-1.3B as the base model with only 1.3B parameters. The LLaVA-1.5-13B based GNS model not only surpasses both GPT-4o and PGPs-specified G-LLaVA, but also achieves new SOTA accuracy. In Geometry3K, the problem question provides very limited information and most of the geometry details are represented by diagrams, which great challenge for the models’ ability to comprehend geometry diagrams. Inter-GPS manually defined complex rules with algorithm-based diagram parser to parse the problems, which enhanced the solving performance, bringing it to a level competitive with humans. Meanwhile, even the powerful GPT-4o can only reach $4 9 . 6 \%$ , and Gemini1.5-Flash is lower. The results indicate that the problems with very little textual information are still challenging for MLLMs. By parsing the entire problem into symbolic clauses, GNS-MLLMs show higher accuracy than GPT-4o, while G-LLaVA obtains poor accuracy with only $2 8 . 1 \%$ .

As shown in the figure, DE is the Since DE is the perpendicular bisector The area of triangle ADE perpendicular bisector of triangle of BC, therefore $D E = 1 / 2$ BC. Therefore, is half the area of triangle ABC. Given that the area the area of triangle ADE is equal to ABC. So the area of of ​triangle ABC is $8 . 0 \land 2$ , then the 1/4 …… Solution Program: Multiple V0 triangle ADE is $4 . 0 \land 2$ . area of ​triangle ADE is $( ) \land 2$ . 2 8.0 Multiple V1 2 V0 Get V1 \*\*Answer: $\mathtt { B } ^ { * * }$ Gemini Choices: A: 2   B: 4   C: 6   D: 8 GNS-LLaVA-1.5-7B Gemini-1.5-Flash   
Given that DE is the perpendicular bisector of triangle Solution: Since DE … Therefore, the ratio of the areas of   
ABC, this means that DE divides triangle ABC into two triangle ADE to triangle ABC is equal to the square of the   
smaller triangles, ADE and DEC. Since DE is the ratio of their corresponding sides, which is equal to   
perpendicular bisector… area of triangle ADE is 4 square $( { \sf D E / B C } ) ~ = ~ 4 / 4 ~ = ~ 1 / 2$ . …Therefore, the answer is B.   
units. The correct option is: B: 4 GPT-4o Answer:B G-LLaVA-7B 囍

Figure 4: One example of different MLLMs (GPT-4o (2024-05-13), Gemini-1.5-Flash and G-LLaVA) solving a typical plane geometry problem. The problem ID in the GeoQA test is 366. The texts labeled with red are the wrong reasoning parts.

Table 3: ROUGE-1 performance of the problem symbolic parsing on the testmini of MathVista (GPS).   

<html><body><table><tr><td>GNS-MLLMs</td><td>ROUGE-1 (%)</td></tr><tr><td>LLaVA-1.5-13B</td><td>83.1</td></tr><tr><td>MiniCPM-V2.5-8B</td><td>79.6</td></tr></table></body></html>

Table 4: Component Ablation Study. With labelled as $x$ , we remove corresponding training tasks from the dataset and remove the module from the GNS framework. We use general CoT reasoning when symbolic computation is ablated.   

<html><body><table><tr><td></td><td>1 2 X</td><td>3</td><td>4</td><td>5</td></tr><tr><td>Knowledge Prediction Symbolic Parsing Symbolic Computation</td><td>X X X X √</td><td>X √ √</td><td>√ √ X</td><td>√ √ √</td></tr><tr><td>Accuracy (%)</td><td>52.4</td><td>55.8 60.6</td><td>57.2</td><td>62.0</td></tr></table></body></html>

Problem Symbolic Parsing. We also test the problem symbolic parsing performance of our MLLMs on testmini set of MathVista (GPS). It is challenging to measure the accuracy of parsing as the parsing results are in string format (i.e. ’line AC, line BD’ is equivalent to ’line DB, line CA’). Therefore, we select ROUGE-1 as the evaluation metric and manually label the parsing ground truth of these problems for testing. The results are shown in Table 3. Current results are particularly impressive that considering this is the first time to train MLLMs to parse plane geometry problems into symbolic clauses, this work also indicates a strong baseline performance for future research on MLLMs.

# Method Analysis

Component Ablation Study. As shown in Table 4, we conduct ablation studies on the testmini of MathVistat (GPS) to verify the effectiveness of different components in GNS with LLaVA-1.5-7B. Firstly, comparing settings (1, 2) and (4, 5), symbolic computation markedly improves the accuracy, indicating its effectiveness in problem solving by enhancing numeric computation capability. The comparison between (2) and (3) demonstrates symbolic parsing process contributed to the problem solving by explicitly understanding problems. Furthermore, according to (3) and (5), knowledge prediction also facilitates the model to conduct a more accurate solving process.

Case Study. As shown in Figure 4, we select a plane geometry problem in the GeoQA test set to analyze. This problem requires to understand the relationship of points and triangles with the proportion of similar triangles. Despite the geometry diagram of the problem is not complex, GPT-4o failed to distinguish the correct similar relationship of triangle ADE and ABC, which finally led to the wrong reasoning process. G-LLaVA-7B successfully understood the similar relationship but gave the wrong proportion relationship in the further reasoning. While Gemini-1.5-Flash directly gave the wrong proportion relationship with few descriptions. Our proposed GNS with LLaVA-1.5-7B clearly described similar triangles and used correct triangle area proportion, finally output the symbolic computation program to get the correct result.

# Conclusion

In this paper, we introduce a neural-symbolic MLLM framework (GNS), which solves plane geometry problems through knowledge prediction, symbolic parsing, reasoning and computation. By parsing the problem into symbolic clauses, the model can explicitly comprehend the geometry elements in the problem, thereby facilitating the reasoning process. In addition, GNS is capable of conducting precise numerical computations with the symbolic computation module. Furthermore, we construct GNS-260K, the largest plane geometry problem dataset with multiple annotations of knowledge prediction, symbolic parsing, reasoning and computation. Extensive experiments demonstrate the effectiveness of our model, achieving the leading position in the MathVista GPS task, becoming the new SOTA method on GeoQA dataset and also achieved markedly performance improvements on MathVerse. Meanwhile, the experiment results also verified the generalization ability of GNS on different base MLLMs, and these GNS-MLLMs even outperformed much larger MLLMs like GPT-4o on three datasets.