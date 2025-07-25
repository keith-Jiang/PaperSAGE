# Task-level Distributionally Robust Optimization for Large Language Model-based Dense Retrieval

Guangyuan $\mathbf { M } \mathbf { a } ^ { 1 , 2 * }$ , Yongliang $\mathbf { M } \mathbf { a } ^ { 3 }$ , Xing ${ \mathbf { W } } { \mathbf { u } } ^ { 1 , 2 }$ , Zhenpeng $\mathbf { S u } ^ { 1 , 2 }$ , Ming Zhou3, Songlin $\mathbf { H } \mathbf { u } ^ { 1 , 2 \dag }$

1Institute of Information Engineering, Chinese Academy of Sciences, Beijing, China 2School of Cyber Security, University of Chinese Academy of Sciences, Beijing, China 3Langboat Technology, Beijing, China {maguangyuan,wuxing,suzhenpeng,husonglin} $@$ iie.ac.cn, {mayongliang,zhouming}@langboat.com

# Abstract

Large Language Model-based Dense Retrieval (LLM-DR) optimizes over numerous heterogeneous fine-tuning collections from different domains. However, the discussion about its training data distribution is still minimal. Previous studies rely on empirically assigned dataset choices or sampling ratios, which inevitably lead to sub-optimal retrieval performances. In this paper, we propose a new task-level Distributionally Robust Optimization (tDRO) algorithm for LLMDR fine-tuning, targeted at improving the universal domain generalization ability by end-to-end reweighting the data distribution of each task. The tDRO parameterizes the domain weights and updates them with scaled domain gradients. The optimized weights are then transferred to the LLM-DR finetuning to train more robust retrievers. Experiments show optimal improvements in large-scale retrieval benchmarks and reduce up to $30 \%$ dataset usage after applying our optimization algorithm with a series of different-sized LLM-DR models.

# Code — https://github.com/tdro-llm/tdro Datasets — https://huggingface.co/tdro-llm

# Introduction

Dense retrieval (Karpukhin et al. 2020) recalls relevant documents from large-sized candidate pools with the similarity search (Mussmann and Ermon 2016) of query-passage embeddings. The recent bloom of Large Language Modelbased Dense Retrieval (LLM-DR) (Wang et al. 2024a; Meng et al. 2024; Muennighoff et al. 2024) promotes remarkable retrieval abilities with better foundation models (Touvron et al. 2023; Bai et al. 2023; Jiang et al. 2023) and large-scale training collections (Wang et al. 2024a; Xiao et al. 2023).

LLM-DR fine-tuning learns LLM-based retrieval models with heterogeneous training collections (Reimers 2019) from multiple domains with different learning difficulties. The data distribution of training collections, i.e. a mixture of data with chosen datasets or sampling ratio on each dataset, significantly influences the general retrieval performances of dense retrievers (Oren et al. 2019; Meng et al. 2024). However, the choice or sampling ratio of the training sets still relies heavily on intuitional assessments. It’s hard to decide whether an empirical assigned data sampling ratio is optimal for the models to perform well, i.e. robust to all tasks. Nowadays, the robustness of data distributional optimization for LLM-DR is still very limited.

![](images/d9ff473aaf43cd692ab5f29813c5564cb0493db0371322dade384d50e444412d.jpg)  
Figure 1: Task-level Distributionally Robust Optimization for Large Language Model-based Dense Retrieval.

Distributionally Robust Optimization (DRO) (Oren et al. 2019; Sagawa et al. 2019; Piratla, Netrapalli, and Sarawagi 2022) receives extensive discussions for battling unbalanced data composition. GroupDRO (Sagawa et al. 2019), the most popular algorithm for DRO, optimizes on the worst-case loss of the corresponding domain, which picks a domain with the highest loss at each training step and up-weights the loss of this domain. Although there was an attempt (Yu et al. 2022) to utilize vanilla GroupDRO or variants of DRO algorithm (Piratla, Netrapalli, and Sarawagi 2022) for dense retrieval fine-tuning, the optimization is limited to a small BERTbased model over clustered groups of one single dataset, i.e. MS-MARCO (Nguyen et al. 2016), which fails to generalize to LLM-DR fine-tuning with multiple heterogeneous training collections. It’s profitable to solve the data distribution issue of LLM-DR in an end-to-end manner like DRO, but such a study is still left for further exploration.

The existing DRO algorithms (Oren et al. 2019; Sagawa et al. 2019; Piratla, Netrapalli, and Sarawagi 2022), such as GroupDRO, are theoretically incompatible with LLMDR fine-tuning due to different batch sampling strategies and incommensurable loss scales. Firstly, DRO requires all domain data mandatory in one batch for end-to-end comparisons. It dynamically reweights the training losses based on the worst-case group and derives the robust-optimized proxy model as the final model directly. However, the LLMDR fine-tuning collects the heterogeneous sets in a homogeneous batching method (Meng et al. 2024) during its finetuning, which means only one domain can be collected to the whole batch to ensure that in-batch negatives are sampled from the same task. What’s worse, the heterogeneous collections used by LLM-DRs have significantly different loss scales. If directly applying the DRO algorithms (Sagawa et al. 2019; Yu et al. 2022; Piratla, Netrapalli, and Sarawagi 2022) to LLM-DRs, the models will always bias towards the domain with the highest training loss, i.e. worst case loss, which will hurt the fine-tuning process. As is shown in Table 1, the loss of Yahoo answers with a Qwen1.5-0.5B retriever is three times over MS-MARCO and five times over DuReader. Directly using worst-case loss will make the model always biased towards Yahoo, rather than MSMARCO or DuReader.

Table 1: Comparison of loss scales for Yahoo answers (TitleAnswer) (Zhang, Zhao, and LeCun 2015), MS-MARCO (Nguyen et al. 2016), and DuReader (Qiu et al. 2022) datasets. The model used here is Qwen1.5-0.5B trained with uniform data sampling ratios for 1k steps.   

<html><body><table><tr><td>Dataset</td><td>Loss</td></tr><tr><td>Yahoo answers (Title-Ans wer)</td><td>3.9257</td></tr><tr><td>MS-MARCO</td><td>1.3312</td></tr><tr><td>DuReader</td><td>0.6925</td></tr></table></body></html>

To tackle the above optimization problems for LLM-DR, we develop a new task-level Distributionally Robust Optimization (tDRO) algorithm that aims at improving the general domain adaption ability across multiple tasks1. Firstly, to coordinate with different batch sampling strategies, our algorithm separates the DRO optimization and LLM-DR fine-tuning as is presented in Figure 1. Instead of directly learning a robust model (Oren et al. 2019; Sagawa et al. 2019), such separation first learns domain weights with a proxy model via the DRO algorithm and then transfers learned domain weights to LLM-DR fine-tuning. The proxy model is initialized from a small-sized LM, e.g. Qwen1.5- 0.5B (Bai et al. 2023). It receives uniformly sampled training collections within an input batch, computes contrastive losses of each domain, and uses them as the gradients of domain weights. Then it transfers the learned weight or merely chooses the top-weighted datasets all LLM-DR model finetunings with different sizes. This separation shares several benefits: We can sample all domains within a batch in the tDRO stage and use task-homogeneous batching in the finetuning stage, which makes DRO work well while not hurting the final retrieval performances of LLM-DRs. Also, a smallsized LM can be used in tDRO optimizations to reduce com

putational overheads.

Secondly, as discussed above, the heterogeneous domains have different loss scales. To make a comparable measurement of domain running losses, we use a trained LLM-DR model, e.g. Qwen1.5-0.5B with uniform data sampling ratios, as the reference model and forward it with the same inputs. We compute the relative loss measurement by dividing the proxy loss with reference loss. Intuitively, the relative loss measurement represents the improving headroom for the corresponding domains. Higher gradients will up-weight more corresponding domain weights.

To verify the effectiveness of the tDRO algorithm, we conduct data distribution optimization on open-sourced sentence transformers training data (Reimers 2019). We test on three large-scale retrieval benchmarks to fully assess the universal retrieval abilities across different languages and domains, including multilingual MIRACL (Zhang et al. 2023), cross-lingual MKQA (Longpre, Lu, and Daiber 2021), and monolingual BeIR (Thakur et al. 2021). Experiments shows steady improvements with less dataset usage after applying tDRO optimization.

# Algorithm

# Problem Statement

Assume the training collections $D ^ { t r a i n }$ of LLM-DR finetuning are composed of $k$ individual datasets, each of them is assigned with a domain weight $\alpha$ , representing a choice of probability distribution $P _ { \alpha }$ over joint training collections:

$$
P _ { \alpha } = \sum _ { g = 1 } ^ { k } \alpha _ { g } \mathrm { U } ( D _ { g } ^ { t r a i n } ) , s . t . \sum _ { g = 1 } ^ { k } \alpha _ { g } = 1 .
$$

$\mathrm { ~ U ~ }$ is a uniform distributional sampling function. And $\alpha _ { g } \mathrm { U } ( D _ { g } ^ { t r a i n } )$ means sampling from such distribution with weight $\alpha _ { g }$ for group $g$ , which is also called $\alpha$ -covered probability distribution (Oren et al. 2019). The goal of LLM-DR data distributional optimization is to find an optimal distribution $P _ { \alpha }$ , or a choice of weights $\alpha$ specifically, enabling the model to perform well on all downstream tasks $D ^ { t e s t }$ . Note that downstream tasks $D ^ { t e s t }$ are not necessarily identical to the fine-tuning sets $D ^ { t r a i n }$ .

# Task-level Distributionally Robust Optimization

The task-level Distributionally Robust Optimization (tDRO) algorithm parameterizes domain weights $\alpha$ and tries to learn the best choice of $\alpha$ for all tasks in an end-to-end manner. The tDRO holds a basic assumption for solving the data distributional optimization of LLM-DR: In a scenario with multiple independent and identically distributed (iid) task collections, if a model is robust to the training phase, then it will be robust to most of the test sets. Thus like most DRO algorithms (Oren et al. 2019; Sagawa et al. 2019; Piratla, Netrapalli, and Sarawagi 2022), tDRO operates on the robustness of training collections for universal adaption abilities. The whole optimization pipeline includes a separate tDRO stage and LLM-DR fine-tuning stage.

A) Fitting single domain data in one batch.

B) Fitting multiple domain data in one batch.

![](images/d4e35f42acb48adef0e3d7c79e73ddc21e747e85ba8866cafbb7a0319887ddb4.jpg)  
Figure 2: Different batch sampling strategies and negative types for A) LLM-DR Fine-tuning and B) Distributionally Robust Optimization.

InfoNCE Loss tDRO treats each dataset as a task (or domain) at minimal granularity and trains a proxy model with parameters $\theta _ { p r o x y }$ to adaptively compare and update the domain weights at the task level. At each training step $t$ , a batch of training data with domain sampling probabilities $1 / k$ is forwarded through the proxy model. By extracting hidden states from the last token position, each batch item comprises a query representation $q$ , a positive document representation $d ^ { + }$ , and several hard negative $( H N )$ document representations $d _ { H N } ^ { - }$ . InfoNCE loss (van den Oord, Li, and Vinyals 2018), i.e. contrastive loss, is used to calculate losses of the proxy model for each group:

$$
\mathcal { L } ^ { p r o x y } = - \log \frac { e ^ { q \cdot d ^ { + } / \tau } } { e ^ { q \cdot d ^ { + } / \tau } + \sum e ^ { q \cdot d _ { H N } ^ { - } / \tau } } ,
$$

where $\tau$ is a contrastive temperature. In-batch negative sampling is not an option for tDRO because different tasks could induce false negatives and reduce negative qualities.

Optimization Objective tDRO learns domain weights $\alpha$ with a dual-optimization objective, which minimizes the supremum of the $\alpha$ -weighted sum of group loss measurements. Such an objective ensures universal robustness by lowering the upper bound of the worst groups:

$$
\operatorname* { m i n } _ { \theta } \operatorname* { s u p } _ { \alpha } \sum _ { g = 1 } ^ { k } \alpha _ { g } \mathcal M _ { g } ( \theta _ { p r o x y } ; q , d ^ { + } , d ^ { - } ) ,
$$

where group loss measurement $\mathcal { M } _ { g }$ is a scalar corresponding to averaged losses, representing the improving headrooms for each group. Following the previous DRO framework (Sagawa et al. 2019), the above object is optimized by interleaving the updates of weights $\alpha$ and proxy model parameters $\theta$ .

Weights Update For weights update, tDRO optimizes the above object by up-weighting the corresponding domain weights with exponential gradient ascent:

$$
\alpha _ { g } ^ { t } = \alpha _ { g } ^ { t - 1 } e ^ { \eta _ { \alpha } \mathcal { M } _ { g } } ,
$$

where $\eta _ { \alpha }$ is the learning rate for domain weights. As a common optimation practice, gradient normalization is used on the above gradients to ensure stable weight updates. Intuitively, a higher loss measurement induces more up$\alpha$ ,eiagrhet-inog omfaltihzeatciornriespoenrfdoirnmgegdrtoou pe.nsAufrte $\textstyle \sum _ { g = 1 } ^ { k } \alpha _ { g } = 1$

Relative Loss Measurement The key component of tDRO is how to derive a balanced and comparable loss measurement $\mathcal { M } _ { g }$ . GroupDRO directly uses the average group loss $\mathbb { E } ( \mathcal { L } _ { g } ^ { p r o x y } )$ as the loss measurement, where $\mathbb { E }$ is the arithmetic mean function. However, as presented in Table 1, the averaged contrastive losses of each group are not comparable. Directly using the average group loss will always make the proxy model biased toward one group with the highest loss. To solve the above issue, we introduce a trained reference model $r e f$ , forward it with the same inputs, compute reference loss as Formula (2), and divide the proxy loss with reference loss to dynamically scale the average group loss. This design is called relative loss measurement.

$$
\mathcal { M } _ { g } = \mathbb { E } ( \mathcal { L } _ { g } ^ { p r o x y } ) / \mathbb { E } ( \mathcal { L } _ { g } ^ { r e f } ) .
$$

The reference model is frozen and will not be updated during the tDRO stage. In our implementation, we initialize the reference model with Qwen1.5-0.5B (Bai et al. 2023) and train with uniform sampling weights on all training sets. The training setting of the reference model follows the LLM-DR fine-tuning recipe, which will be described later.

Proxy Model Update After updating domain weights, the proxy model is updated with $\alpha$ -weighted contrastive loss.

$$
\theta ^ { t } = \theta ^ { t - 1 } - \alpha _ { g } \eta _ { \theta } \nabla _ { \theta } ,
$$

where $\eta _ { \theta }$ is the learning rate of the proxy model, and $\nabla _ { \theta }$ is the gradient of proxy model obtained by backpropagation. The AdamW optimizer can be used for fast convergence.

# LLM-DR Fine-tuning

LLM-DR fine-tuning also uses contrastive loss to pull positive representations together and push negative representations away. But it uses completely different batch sampling strategies and negative types from tDRO, which is one of the reasons that previous DRO algorithms like GroupDRO are theoretically incompatible with LLM-DR fine-tuning.

<html><body><table><tr><td>Dataset</td><td>Language</td><td>Category</td><td>Deduped Size</td></tr><tr><td>agnews</td><td>English</td><td>News</td><td>1,157,745</td></tr><tr><td>AllNLI</td><td>English</td><td>NLI</td><td>277,230</td></tr><tr><td>altlex</td><td>English</td><td>Wikipedia Pair</td><td>112,696</td></tr><tr><td>amazon_review_2018</td><td>English</td><td>Amazon</td><td>999,999</td></tr><tr><td>cnn_dailymail</td><td>English</td><td>News</td><td>311,971</td></tr><tr><td>codesearchnet</td><td>English</td><td>Github</td><td>1,375,067</td></tr><tr><td>dureader</td><td>Chinese</td><td>Web Collections</td><td>86,395</td></tr><tr><td>eli5-question_answer</td><td>English</td><td>Reddit</td><td>325,390</td></tr><tr><td>gooaq-pairs</td><td>English</td><td>Web Collections</td><td>3,012,347</td></tr><tr><td>hotpotqa</td><td>English</td><td>Wikipedia QA</td><td>85,000</td></tr><tr><td>medmcqa</td><td>English</td><td>Medical</td><td>160,865</td></tr><tr><td>miracl</td><td>16 languages</td><td>MultilingualWikipedia</td><td>32.405</td></tr><tr><td>mr_tydi_combined</td><td>11 languages</td><td>Multilingual Wikipedia</td><td>48,475</td></tr><tr><td>msmarco</td><td>English</td><td>Web Collections</td><td>502,854</td></tr><tr><td>nq</td><td>English</td><td>Wikipedia QA</td><td>58,800</td></tr><tr><td>quora_duplicates_triplets</td><td>English</td><td>ForumDuplicates</td><td>97,011</td></tr><tr><td>searchQA_top5_snippets</td><td>English</td><td>Web Collections</td><td>117,219</td></tr><tr><td>sentence-compression</td><td>English</td><td>News</td><td>180,000</td></tr><tr><td>SimpleWiki</td><td>English</td><td>Wikipedia Pair</td><td>102,225</td></tr><tr><td>squad_pairs</td><td>English</td><td>Wikipedia QA</td><td>87,595</td></tr><tr><td>stackexchange_duplicates_title-body</td><td>English</td><td>Forum Duplicates</td><td>250,516</td></tr><tr><td>t2ranking</td><td>Chinese</td><td>Web Collections</td><td>200,376</td></tr><tr><td>trivia</td><td>English</td><td>Wikipedia QA</td><td>60,370</td></tr><tr><td>xsum</td><td>English</td><td>News</td><td>226,711</td></tr><tr><td>yahoo_answers_title_answer</td><td>English</td><td>Yahoo</td><td>1,198.018</td></tr></table></body></html>

Table 2: Training datasets information. Note that strict deduplication is performed on all training sets with Simhash (Manku, Jain, and Sarma 2007) to ensure no overlap between training and testing sets.

LLM-DR Batching Strategy As is shown in Figure 2A, LLM-DR fine-tuning fits the data from one single domain in each batch to ensure the quality of in-batch negatives. This batching strategy is also called task-homogeneous batching (Meng et al. 2024). As is a common practice, LLM-DR trains with large batch sizes, e.g. 2048 in our implementation, and three types of negatives, including hard negatives $( H N )$ , in-batch negatives $( I B N )$ , and cross-batch negatives $( C B N )$ . Large batch size enables contrastive learning of LLM-DR using more negatives, especially in-batch and cross-batch negatives. Hard negatives are provided by individual datasets, which are mined from existing retrieval systems like BM25 (Robertson et al. 1994) or dense retrievers (Karpukhin et al. 2020). In-batch negatives are samples from other data items within a batch. Cross-batch negatives are samples gathered from other GPU batches. Overall, the contrastive loss $( \mathcal { L } ^ { C L } )$ for LLM-DR is formulated as follows.

$$
\mathcal { L } ^ { C L } = - \log \frac { e ^ { q \cdot d ^ { + } / \tau } } { e ^ { q \cdot d ^ { + } / \tau } + \sum e ^ { q \cdot \{ d _ { H N } ^ { - } ; d _ { T B N } ^ { - } ; d _ { C B N } ^ { - } \} / \tau } } .
$$

However, tDRO compares and updates domain weights in an end-to-end manner, thus it requires fitting all domain data in one batch. As displayed in Figure 2B, The training batch

is composed of all domains, while the in-batch/cross-batch negatives are not used in tDRO.

# Experiments

# Experiment Settings

Datasets A total of 25 individual datasets are used in our experiments, covering categories of Wikipedia, web collection, news, GitHub, yahoo, etc. Most of them are directly taken from sentence transformers training data (Reimers 2019). BGE-large-en-1.5 (Xiao et al. 2023) is used to mine negatives if the original datasets (several English datasets) have no negatives provided. Several popular multilingual datasets are also included in the training sets, including MIRACL (Zhang et al. 2023) and Mr.Tydi (Zhang et al. 2021). All information about datasets is listed in Table 2.

Settings For the tDRO algorithm, both the proxy and reference models are initialized from Qwen1.5-0.5B (Bai et al. 2023) for computational efficiency. tDRO is performed with a total batch size of 2048, query & document maximum sequence lengths of 128 & 512, a proxy model learning rate $\eta _ { \theta }$ of 1e-4, contrastive temperature $\tau$ of 0.002, weights learning rate $\eta _ { \alpha }$ of 2e-2, and seed of 42. The weights from the tDRO stage are directly transferred to LLM-DR fine-tuning. Two weight transfer strategies are utilized:

1. Top-rated dataset selection: Use the Top-rated datasets and discard the datasets with lower weights. This helps reduce dataset usage.

<html><body><table><tr><td>Benchmark (# Dataset)</td><td colspan="2">MIRACL (18)</td><td colspan="2">MKQA (25)</td><td>BeIR (15)</td></tr><tr><td>Metric</td><td>nDCG@10</td><td></td><td></td><td>Recall@100 Accuacy@10 Accuacy@100</td><td>nDCG@10</td></tr><tr><td colspan="6">Uniform Sampling Baselines</td></tr><tr><td>Qwen-0.5B</td><td>45.8</td><td>80.5</td><td>43.1</td><td>61.3</td><td>47.5</td></tr><tr><td>Qwen-1.8B</td><td>50.9</td><td>84.7</td><td>45.0</td><td>64.0</td><td>48.8</td></tr><tr><td>Qwen-4B</td><td>55.9</td><td>88.7</td><td>53.7</td><td>70.2</td><td>51.8</td></tr><tr><td>Qwen-7B</td><td>59.6</td><td>90.6</td><td>58.7</td><td>73.6</td><td>52.3</td></tr><tr><td>Mistral-7B</td><td>61.3</td><td>91.6</td><td>59.8</td><td>72.8</td><td>55.2</td></tr><tr><td>Llama3-8B</td><td>64.1</td><td>92.8</td><td>64.0</td><td>75.8</td><td>55.0</td></tr><tr><td colspan="6">tDRO -Dataset Selection Top-70%</td></tr><tr><td>Qwen-0.5B</td><td>48.7* (+2.9) 82.1* (+1.6)</td><td></td><td>45.4* (+2.3)</td><td>62.3* (+1.0)</td><td>48.9* (+1.4)</td></tr><tr><td>Qwen-1.8B</td><td>54.1*(+3.2) 86.6*(+1.9)</td><td></td><td>48.6* (+3.6)</td><td>65.6* (+1.6)</td><td>50.2* (+1.4)</td></tr><tr><td>Qwen-4B</td><td></td><td>58.6*(+2.7) 90.0* (+1.3)</td><td>57.0* (+3.3)</td><td>71.4* (+1.2)</td><td>52.6* (+0.8)</td></tr><tr><td>Qwen-7B</td><td></td><td>61.6*(+2.0) 91.4*(+0.8)</td><td>59.9* (+1.2)</td><td>73.8 (+0.2)</td><td>53.3* (+1.0)</td></tr><tr><td>Mistral-7B</td><td></td><td>63.8*(+2.5) 92.4*(+0.8)</td><td>62.5* (+2.7)</td><td>73.8* (+1.0)</td><td>55.2 (+0.0)</td></tr><tr><td>Llama3-8B</td><td></td><td>66.4* (+2.3) 93.5*(+0.7)</td><td>66.0* (+2.0)</td><td>76.4* (+0.6)</td><td>55.1 (+0.1)</td></tr><tr><td>Avg Gains</td><td>+2.6</td><td>+1.2</td><td>+2.5</td><td>+0.9</td><td>+0.8</td></tr><tr><td colspan="6">tDRO - Sample Ratio Reweighting</td></tr><tr><td>Qwen-0.5B</td><td>49.1*(+3.3) 82.7*(+2.2)</td><td></td><td>45.5* (+2.4)</td><td>62.2* (+0.9)</td><td>48.3* (+0.8)</td></tr><tr><td>Qwen-1.8B</td><td>53.6*(+2.7) 86.5* (+1.8)</td><td></td><td>50.5* (+5.5)</td><td>66.8* (+2.8)</td><td>49.7* (+0.9)</td></tr><tr><td>Qwen-4B</td><td>58.4*(+2.5) 90.0*(+1.3)</td><td></td><td>57.8* (+4.1)</td><td>72.0* (+1.8)</td><td>51.9 (+0.1)</td></tr><tr><td>Qwen-7B</td><td>61.0*(+1.4) 91.1(+0.5)</td><td></td><td>59.8* (+1.1)</td><td>73.6 (+0.0)</td><td>52.4 (+0.1)</td></tr><tr><td>Mistral-7B</td><td></td><td>63.4*(+2.1) 92.4*(+0.8)</td><td>62.8* (+3.0)</td><td>74.0* (+1.2)</td><td>55.4 (+0.2)</td></tr><tr><td>Llama3-8B</td><td>66.3*(+2.2) 93.4* (+0.6)</td><td></td><td>67.0* (+3.0)</td><td>76.8* (+1.0)</td><td>55.0 (+0.0)</td></tr><tr><td>Avg Gains</td><td>+2.4</td><td>+1.2</td><td>+3.2</td><td>+1.3</td><td>+0.4</td></tr></table></body></html>

Table 3: Retrieval performances and corresponding gains of tDRO algorithm on MIRACL dev, MKQA test, and BeIR test benchmarks. The highest retrieval scores and average performance gains are marked as bold. \*Significant improvements $( \mathsf { p } \leq$ 0.01) over the corresponding baseline. MS-MARCO in BeIR uses the dev split because there is no public test label.

![](images/76b9742577d70e5aa3672bf45cb9fe11cc5e013cf36bd30b34ced6977f1179f2.jpg)  
Figure 3: Weights comparison among baseline, tDRO, and other loss measurement designs.

2. Sample Ratio Reweighting: Directly use the weights to reweight the sample ratios of datasets.

We conduct weights transfer on Qwen-1.5 0.5B, 1.8B, 4B, 7B (Bai et al. 2023), Mistral-0.1-7B (Jiang et al. 2023) and Llama3-8B (Touvron et al. 2023) for LLM-DR fine-tuning. Contrastive learning is performed with the same batch size, sequence lengths, model learning rate (1e-4), and contrastive temperature as stated before. Gradient cache (Gao et al. 2021), flash attention 2 (Dao 2023), full-shard data parallel (FSDP), activation checkpointing and low-rank adapter (LoRA) (Hu et al. 2022) with $r = 8 , \alpha = 3 2$ and dropout ratio of 0.1 are used to reduce GPU memory usage. Following the previous work (Wang et al. 2024a; Su et al. 2023), prompt instructions are added on the query side for multitasks during training and evaluation. All trainings are performed on 8 NVIDIA H800 GPUs with 4.5 hours for tDRO and less than 10 hours for all LLM-DR fine-tunings.

Table 4: Retrieval scores among state-of-the-art retrievers. \*We take the released BeIR scores without GPT data for fair comparisons.   

<html><body><table><tr><td>Benchmark</td><td>Enhance</td><td>MIRACL</td><td>MKQA</td><td>BeIR</td></tr><tr><td>Metric → Models ↓</td><td>Special Pre-train</td><td>nDCG@10</td><td></td><td>Acc@100 nDCG@10</td></tr><tr><td>BM25</td><td></td><td>31.9</td><td>39.9</td><td>41.7</td></tr><tr><td>mContriever</td><td>√</td><td>43.1</td><td>67.9</td><td>46.0</td></tr><tr><td>mE5-large-inst</td><td>√</td><td>64.6</td><td>70.2</td><td>52.3</td></tr><tr><td>E5-Mistral*</td><td></td><td>62.2</td><td>68.6</td><td>52.7</td></tr><tr><td colspan="5">tDRO+LLM-DRLlama3-8B</td></tr><tr><td>Top-70%</td><td></td><td>66.4</td><td>76.4</td><td>55.1</td></tr><tr><td>Reweighting</td><td></td><td>66.3</td><td>76.8</td><td>55.0</td></tr></table></body></html>

Baselines and Benchmarks To compare the performance changes after tDRO optimization, we choose LLM-DR with uniform sampling weights2 as baselines. All other settings are kept the same. Note that results from some recent state-of-the-art retrievers, including BM25 (Robertson et al. 1994), mContriever/Contriever (Izacard et al. 2021), E5- Mistral-7b (Wang et al. 2024a), and Multilingual-e5-largeinstruct (Wang et al. 2024b) are also added to our results. But we DO NOT seek to compare with them. Some of them utilize multiple training enhancements, such as data synthesis with ChatGPT and special pre-trainings (Wang et al. 2023; Wu et al. 2023; Ma et al. 2024a) for retrieval, which is an unfair comparison to our experiments and out of the research scope for our paper.

MIRACL (Zhang et al. 2023), MKQA (Longpre, Lu, and Daiber 2021), and BeIR (Thakur et al. 2021) are used as main retrieval benchmarks. MIRACL is a huge multilingual retrieval benchmark across 18 languages with 13K queries and 90M passages. Cross-lingual benchmark MKQA holds 25 different languages of parallel Wikipedia queries. Following (Chen et al. 2024), the cross-lingual retrieval is performed by using queries (6.6k for each) of different languages to recall relevant English Wikipedia with 2.7M passages3. BeIR is a heterogeneous English retrieval benchmark with 15 public datasets and 34M documents, covering a wide range of retrieval domains and text symmetries. MIRACL is reported with $\mathrm { n D C G } @ 1 0$ and Recall $@ 1 0 0$ . MKQA is reported with Accuacy $@ \{ 1 0 , 1 0 0 \}$ (Acc). Examing performances at threshold 10 assess the top-ranking ability and threshold 100 for recalling capacity at a larger window. BeIR is reported with $\mathrm { n D C G } @ 1 0$ by following the original paper (Thakur et al. 2021). The whole evaluation takes around 30 hours with 8 NVIDIA H800 GPUs for the largest LLM-DR retriever, Llama3-8B.

# Results

Retrieval benchmarks on multilingual, cross-lingual, and monolingual English retrieval are conducted and listed in tDRO Boosts Retrieval Performances. tDRO boosts the universal retrieval performances of LLM-DR on a series of different-sized base LLM, including Qwen-1.5 (0.5B, 1.8B, 4B, 7B), Mistral-0.1 7B, and Llama3 8B. Both the top-rated dataset selection and the sample ratio reweighting strategies work well on most retrieval benchmarks with significant performance gains. The improvements in multilingual MIRACL and cross-lingual MKQA retrieval are attributed to the up-weighting of multilingual datasets, e.g. MIRACL.

Table 5: Ablation studies on sampling strategy choices and group loss designs.   
Table 3. tDRO optimized domain weights are listed in Figure 3. Top- $70 \%$ is the optimal line for the top-rated dataset selection strategy. Ablation on the percentages of dataset selection will be presented later.   

<html><body><table><tr><td></td><td>BEIR</td><td>MIRACL</td><td>MKQA</td></tr><tr><td>Metric</td><td>nDCG@10|nDCG@10</td><td></td><td>Acc@100</td></tr><tr><td>Qwen-0.5B-Baseline</td><td>47.5</td><td>45.8</td><td>61.3</td></tr><tr><td colspan="4">Sampling Strategy Choices for tDRO</td></tr><tr><td>DatasetSelection w/Bottom-50%</td><td>45.2-2.3</td><td>44.6-1.2</td><td>60.7-0.6</td></tr><tr><td>w/ Top-60% w/ Top-70% w/Top-80%</td><td>48.4+0.9 48.9+1.4 48.9+1.4</td><td>48.2+2.4 48.7+2.9 47.8+2.0</td><td>62.0+0.7 62.3+1.0 62.1+0.8</td></tr><tr><td>SampingRatio Reweighting</td><td>48.3+0.8</td><td>49.1+3.3</td><td>62.2+0.9</td></tr><tr><td>LossReweighting</td><td>48.2+0.7</td><td>46.3+0.5</td><td>61.2-0.1</td></tr><tr><td colspan="4">Loss Measurements (w/ Sampling Ratio Reweighting)</td></tr><tr><td>Lproxy/Lref Lproxy_Lref Lproxy</td><td>48.3+0.8 47.5-0.0 47.2-0.3</td><td>49.1+3.3 45.3-0.5 47.0+1.2</td><td>60.1+1.2 61.5+0.2 62.2+0.9</td></tr></table></body></html>

tDRO Balances Data Usages As is shown in Figure 3, although multilingual training sets are less than monolingual sets, tDRO balances the lingual distribution differences by top-rating the multilingual MIRACL and T2-Ranking. In contrast, tDRO down-weights 9 monolingual English datasets (Nearly $30 \%$ amounts) to less than 0.01. If we remove less-weighted datasets and keep only the Top- $70 \%$ sets, the results on multilingual, cross-lingual, and monolingual retrieval benchmarks even get significantly better, although less data is involved in LLM-DR fine-tuning. This is attributed to the end-to-end task-level data distribution optimization by tDRO, which balances the data distribution and helps LLM-DR training to focus on more useful and challenging data. Notably, as shown in Table 4, tDRO-optimized LLM-DR gets leading performances without using further enhancements, such as GPT4 data synthesis and special pretraining.

# Analysis Weight Transfering Strategies

According to Table 5, top-rated dataset selection reaches peak performances with Top- $70 \%$ datasets. On the contrary, using Bottom- $5 0 \%$ significantly hurts the performances. This strategy is stable on nearly all benchmarks. However, as is shown in Table 3, some monolingual results seem to reach peak levels without further gains, such as Mistral-7B and Llama3-8B for BeIR. This phenomenon only occurs in LLMs with larger parameters $\scriptstyle ( > 7 \mathbf { B } )$ on BeIR. A potential explanation could be that larger LLMs have stronger capacities to tolerate the data weight changes when plenty of monolingual data is provided. Under this circumstance, large LLMs can perform evenly well.

Sample ratio reweighting has more performance gains on MKQA than the selection strategy because it incurs more sampling probabilities on multilingual datasets, e.g. MIRACL and T2-Ranking, as displayed in Figure 3. However, such over-weighting on multilingual datasets is also the reason for no significant gains on monolingual BeIR of large LLMs, e.g. Qwen-4B, 7B, Mistral-7B, and Llama3-8B.

Additionally, we also test re-scaling contrastive losses of LLM-DR with transferred weights. However, such loss reweighting improves BeIR but has no gains on MIRACL and MKQA. A potential reason could be that loss reweighting only changes the loss scale, but does not import more multilingual data because the data sampling ratios are unchanged.

# Loss Measurement Designs

Loss measurement is the key design to get proper domain loss scales. tDRO utilizes relative loss measurement by dividing the proxy loss with the reference loss. We also tested two additional choices:

1. Minus the proxy loss with reference loss. This design is also called group excess loss in DRO (Oren et al. 2019). But obviously, the minus operation could not scale the big loss differences of heterogeneous collections for LLM-DR fine-tuning.

2. Directly using proxy loss.

As is shown in Table 5, these two loss measurements hurt the performances. This is attributed to the incomparable loss scales of different datasets. According to Figure 3, both measurements over-prefer Yahoo significantly, because of the biggest loss scale of Yahoo as shown in Table 1.

# Related Works Large Language Model-based Dense Retrieval (LLM-DR)

Dense retrieval (DR) has gained significant improvements in recent years because of the rapid developments of large language models (LLM) (Ouyang et al. 2022; Touvron et al. 2023; Bai et al. 2023; Jiang et al. 2023) and growing collections of heterogeneous training data (Reimers 2019). In short, the model parameter is growing, and the datasets are increasing. Sentence-T5 (Ni et al. 2022) first scales the foundation models to billon parameter T5 and gains good abilities of sentence embeddings. RepLLaMA and RankLLaMA (Ma et al. 2024b) first finetune retrievers and re-rankers on Llama2 with a single MS-MARCO dataset and get remarkable improvements over a series of smallsized baseline retrievers. E5-Mistral (Wang et al. 2024a) utilizes 500k GPT3.5/4 synthesized training data and 15 wellcollected datasets to fine-tune the retriever, further pushing the boundary of retrieval abilities. It also transfers these data to specially pre-trained small-sized model mE5 (Wang et al. 2024b) for state-of-the-art (SOTA) multilingual performances. However, the utilization of training collection for LLM-DR still relies on intuitional assessments. As discussed in this paper, empirical assigned data choices or sampling ratios, e.g. uniform sampling, incur sub-optimal performances.

# Distributionally Robust Optimization (DRO)

Distributionally Robust Optimization (DRO) is an effective way to battle unbalanced data distributions. Topic CVaR (Oren et al. 2019), first proposes to minimize the worst-case loss of each topic. (Sagawa et al. 2019) proposes a gradientbased GroupDRO algorithm and successfully improves the worse-group accuracies. DoReMi (Xie et al. 2023) utilizes GroupDRO in LLM pre-training with minimax optimization for improving perplexities and downstream accuracies. CGD algorithm (Piratla, Netrapalli, and Sarawagi 2022) introduces the inter-group interactions into GroupDRO by substituting the worst-case loss with the inner product score of group gradients.

Research on DRO for dense retrieval is still minimal. COCO-DR (Yu et al. 2022) combines pre-training method coCondenser (Gao and Callan 2022) and CGD algorithm to battle the distribution shifts on both the document and query side. However, its scope is limited to small-sized BERT models over clustered groups of one single MS-MARCO (Nguyen et al. 2016) dataset. Our study aims to solve the optimization problem of data distribution in LLM-DR finetuning, which end-to-end reweights each dataset for optimal performance.

# Conclusion

Large language model-based dense retrieval (LLM-DR) utilizes multiple heterogeneous training datasets in fine-tuning. Previous studies rely on empirical assessments to decide the sampling ratios of each dataset, which incurs unbalanced training data distribution and leads to sub-optimal performances. In our study, we propose a new task-level Distributionally Robust Optimization (tDRO) algorithm to improve the domain generalization ability by end-to-end reweighting the data distribution of each dataset. Experiments on largescale retrieval benchmarks show steady improvements with less dataset usage.