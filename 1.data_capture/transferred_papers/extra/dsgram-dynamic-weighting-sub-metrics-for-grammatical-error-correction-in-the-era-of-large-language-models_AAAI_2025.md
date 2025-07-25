# DSGram: Dynamic Weighting Sub-Metrics for Grammatical Error Correction in the Era of Large Language Models

Jinxiang ${ \bf X _ { i e } } ^ { 1 2 * }$ , Yilin $\mathbf { L i } ^ { 1 }$ , Xunjian $\mathbf { Y _ { i n } } ^ { 1 }$ , Xiaojun Wan1

1Wangxuan Institute of Computer Technology, Peking University, 2Beijing Jiaotong University xiejinxiang $@$ bjtu.edu.cn, wanxiaojun@pku.edu.cn

# Abstract

Evaluating the performance of Grammatical Error Correction (GEC) models has become increasingly challenging, as large language model (LLM)-based GEC systems often produce corrections that diverge from provided gold references. This discrepancy undermines the reliability of traditional reference-based evaluation metrics. In this study, we propose a novel evaluation framework for GEC models, DSGram, integrating Semantic Coherence, Edit Level, and Fluency, and utilizing a dynamic weighting mechanism. Our framework employs the Analytic Hierarchy Process (AHP) in conjunction with large language models to ascertain the relative importance of various evaluation criteria. Additionally, we develop a dataset incorporating human annotations and LLM-simulated sentences to validate our algorithms and fine-tune more cost-effective models. Experimental results indicate that our proposed approach enhances the effectiveness of GEC model evaluations.

Code — https://github.com/jxtse/GEC-Metrics-DSGram Datasets — https://huggingface.co/datasets/jxtse/DSGram

Input sentence: Though it is said that genetic testing involves emotional and social risks due to the test results, while the potential negative impacts of the risk still exist, the consequence will be significant if other members of his or her family do not know.