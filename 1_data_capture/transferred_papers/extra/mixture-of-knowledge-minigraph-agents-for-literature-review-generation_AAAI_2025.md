# Mixture of Knowledge Minigraph Agents for Literature Review Generation

Zhi Zhang1, Yan $\mathbf { L i u } ^ { 1 , * }$ , Sheng-hua Zhong2, Gong Chen1, Yu Yang3, Jiannong Cao1

1The Hong Kong Polytechnic University, the Department of Computing, Hong Kong, 999077, China   
2Shenzhen University, the College of Computer Science and Software Engineering, Shenzhen, 518052, Guangdong, China   
3The Education University of Hong Kong, Centre for Learning, Teaching, and Technology, Hong Kong, 999077, China zhi271.zhang@connect.polyu.hk, yan.liu $@$ polyu.edu.hk, csshzhong $@$ szu.edu.cn, csgchen@comp.polyu.edu.hk, yangyy $@$ eduhk.hk, jiannong.cao $@$ polyu.edu.hk

# Abstract

Literature reviews play a crucial role in scientific research for understanding the current state of research, identifying gaps, and guiding future studies on specific topics. However, the process of conducting a comprehensive literature review is yet time-consuming. This paper proposes a novel framework, collaborative knowledge minigraph agents (CKMAs), to automate scholarly literature reviews. A novel promptbased algorithm, the knowledge minigraph construction agent (KMCA), is designed to identify relations between concepts from academic literature and automatically constructs knowledge minigraphs. By leveraging the capabilities of large language models on constructed knowledge minigraphs, the multiple path summarization agent (MPSA) efficiently organizes concepts and relations from different viewpoints to generate literature review paragraphs. We evaluate CKMAs on three benchmark datasets. Experimental results show the effectiveness of the proposed method, further revealing promising applications of LLMs in scientific research.

Referencing Paper 1 (understanding) Graph (structure) (𝐴M(aMtertihaol)di)sisusuesdedfofrorA𝑇( (MTeatshko)d).… 𝑴𝟏 M𝑴aterial Mat𝑀erial Referencing Paper 2 (understanding) Method 𝑩 $B$ (Method) is used for 𝑇 (Task) … ${ \pmb M } _ { 1 }$ and $M _ { 2 }$   
(Material) are used for $\pmb { B }$ (Method). Method 𝑨 Task 𝑇 Summary (good) 三   
𝑨 and $\pmb { B }$ employ $M _ { 1 } ^ { * }$ for 𝑇. Compared with 𝐴, $B$ further employs $M _ { 2 }$ . Summary (trivial)   
𝑨 employs $\pmb { M } _ { 1 }$ for 𝑇. 𝑩 employs ${ { M } _ { 1 } }$ and $M _ { 2 }$ for $T$ .

Project — https://minigraph-agents.github.io/

# Introduction

Artificial intelligence (AI) is being increasingly integrated into scientific discovery to augment and accelerate scientific research (Wang et al. 2023). Researchers are developing AI algorithms for various purposes, including literature understanding, experiment development, and manuscript draft writing (Liu et al. 2022; Wang et al. 2024; Martin-Boyle et al. 2024).

Literature reviews play a crucial role in scientific research, assessing and integrating previous research on specific topics (Bolanos et al. 2024). They aim to meticulously identify and appraise all relevant literature related to a specific research question. Recent advancements in AI have shown promising performance in understanding research papers (Van Dinter, Tekinerdogan, and Catal 2021). By leveraging AI capabilities, automatic literature review algorithms enable researchers to save time and effort in the manual process of conducting literature reviews, rapidly identify key trends and gaps in recent research outputs, and uncover insights that might be overlooked in manual reviews (Wagner, Lukyanenko, and Pare´ 2022).

Automatic literature review algorithms typically involve two stages (Shi et al. 2023): (1) selecting relevant reference documents and (2) summarizing the reference documents to compose a summary that presents the evolution of a specific field (these stages can be applied iteratively). Multiple scientific document summarization (MSDS), which aims to generate coherent and concise summaries for clusters of relevant scientific papers on a topic, is the representative work in the second stage. Over the past decades (Jin, Wang, and Wan 2020), researchers have developed various summarization methods. Extractive methods directly select important sentences from original papers, while abstractive methods can generate new words and sentences but are technically more challenging than extractive methods.

Large Language Models (LLMs), pre-trained on massive text data, have shown human-like performance in language understanding and coherent synthesis, recently attracting interest in abstractive summarization. Though advanced in natural language processing, MSDS involves concepts that form complex relations, which LLMs are not naturally designed for and face challenges to organize. As shown in Fig.

1, reference documents involve multiple materials, methods, and tasks that are interconnected. A good summarization should organize concepts and their relations, merging consistent ones (e.g., $A$ and $B$ use $M _ { 1 }$ ) and contrasting different ones (e.g., $B$ uses $M _ { 2 }$ compared with $A$ ). Without explicit instructions, LLMs fail to model these relations and produce high-quality literature reviews (Li and Ouyang 2024).

To handle these complex relations, we propose equipping LLMs with structural knowledge. Knowledge graphs, which represent entities as nodes and their relations as edges, are potential solutions. However, it is challenging to find a general-purpose knowledge graph that covers relations in various literature reviews. Instead of having a single graph for everything (Narayanan et al. 2017), we propose knowledge minigraphs for reference documents of interest. Knowledge minigraphs are small-scale graphs that comprise concepts dynamically extracted from reference documents as nodes and their relations as edges. They omit detailed documentation and highlight relations between concepts.

To automatically construct knowledge minigraphs, we propose a prompt-based algorithm, the knowledge minigraph construction agent (KMCA), to constrain LLMs in identifying research-relevant concepts and relations from references. To handle long context formed by reference documents, we design an iterative construction strategy, where key information and relations are iteratively extracted and stored from references into minigraphs.

By leveraging the knowledge minigraphs, the multiple path summarization agent (MPSA) is designed to organize the generated literature review. However, multiple valid viewpoints exist for discussing concepts through different paths in the knowledge minigraph. Thus, MPSA samples multiple summaries from different viewpoints in the knowledge minigraph, utilizing the technique of mixture of experts. A self-evaluation mechanism is then employed to automatically route to the most desirable summary as the final output.

# Related Work

# Graphs in MSDS Tasks

To generate a summary that is representative of the overall content, graph-based methods construct external graphs to assist document representation and cross-document relation modeling, achieving promising progress. In this regard, LexRank (Erkan and Radev 2004) and TextRank (Mihalcea and Tarau 2004) first introduce graphs to extractive text summarization in 2004. They compute sentence importance using a graph representation of sentences to extract salient textual units from documents as summarization. In 2020, Wang et al. propose to extract salient textual units from documents as summarization using a heterogeneous graph consisting of semantic nodes at several granularity levels of documents (Wang et al. 2020). In 2022, Wang et al. incorporate knowledge graphs into document encoding and decoding, generating the summary from a knowledge graph template to achieve state-of-the-art performance (Wang et al. 2022). However, to the best of our knowledge, no existing work integrates LLMs into graph-based methods to leverage their natural language understanding capabilities for improved graph construction and summary generation.

# Pre-trained Language Models in MSDS Tasks

In recent years, pre-trained language models (PLMs) have demonstrated promising results in multiple document summarization. Liu et al. propose fine-tuning a pre-trained BERT model as the encoder and a randomly initialized decoder to enhance the quality of generated summaries (Liu and Lapata 2019). Xiao et al. introduce PRIMERA, a pre-trained encoder-decoder multi-document summarization model, by improving aggregating information across documents (Xiao et al. 2022). More recently, pre-trained large language models (LLMs) show promising generation adaptability by training billions of model parameters on massive amounts of text data (Zhao et al. 2023; Minaee et al. 2024). Zhang et al. utilize well-designed instructions to extract key elements, arrange key information, and generate summaries (Zhang et al. 2024a). Zakkas et al. propose a three-step approach to select papers, perform single-document summarization, and aggregate results (Zakkas, Verberne, and Zavrel 2024). PLMs can provide fluent summary results for literature review. However, they fall short of relation modeling in multiple reference documents.

# Method

Fig. 2 illustrates the architecture of the proposed collaborative knowledge minigraph agents (CKMAs). CKMAs consist of two key components: the knowledge minigraph construction agent and the multiple path summarization agent.

# Knowledge Minigraph Construction Agent

In this module, we are given $T$ reference documents $\{ C _ { 1 } , \ldots , C _ { T } \}$ ’s abstracts. We aim to construct a knowledge structure that captures the relations between concepts in the referenced papers.

Past decades have witnessed knowledge graphs become the basis of information systems that require access to structured knowledge $( \mathsf { Z o u } 2 0 2 0 )$ . Knowledge structures are represented as semantic graphs, where nodes denote entities and are connected by relations denoted by edges. However, the general-purpose knowledge graphs are unsuitable for scientific document summarization, as they do not necessarily involve the main ideas of research papers. Thus, in this paper, we propose establishing a knowledge minigraph, defined as as a small set of research-relevant concepts and their relations. The construction steps of the knowledge minigraph are as follows: