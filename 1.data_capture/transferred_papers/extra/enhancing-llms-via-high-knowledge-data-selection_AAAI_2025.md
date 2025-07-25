# Enhancing LLMs via High-Knowledge Data Selection

Feiyu Duan1,5\*, Xuemiao Zhang2,5\*, Sirui Wang3,5†, Haoran Que1, Yuqi ${ \bf L i u } ^ { 5 }$ , Wenge $\mathbf { R o n g ^ { 4 \dagger } }$ , Xunliang Cai5

1Sino-French Engineer School, Beihang University, Beijing, China 2Peking University, Beijing, China 3Department of Automation, Tsinghua University, Beijing, China 4School of Computer Science and Engineering, Beihang University, Beijing, China 5Meituan, Beijing, China {duanfeiyu, 2224124, w.rong} $@$ buaa.edu.com, zhangxuemiao@pku.edu.cn, {liuyuqi, wangsirui, caixunliang}@meituan.com

# Abstract

The performance of Large Language Models (LLMs) is intrinsically linked to the quality of its training data. Although several studies have proposed methods for high-quality data selection, they do not consider the importance of knowledge richness in text corpora. In this paper, we propose a novel and gradient-free High-Knowledge Scorer (HKS) to select highquality data from the dimension of knowledge, to alleviate the problem of knowledge scarcity in the pre-trained corpus. We propose a comprehensive multi-domain knowledge element pool and introduce knowledge density and coverage as metrics to assess the knowledge content of the text. Based on this, we propose a comprehensive knowledge scorer to select data with intensive knowledge, which can also be utilized for domainspecific high-knowledge data selection by restricting knowledge elements to the specific domain. We train models on a high-knowledge bilingual dataset, and experimental results demonstrate that our scorer improves the model’s performance in knowledge-intensive and general comprehension tasks, and is effective in enhancing both the generic and domain-specific capabilities of the model.

# 1 Introduction

The impressive performance of large language models (LLMs) has been demonstrated in various natural language processing (NLP) tasks (Touvron et al. 2023; OpenAI 2023), yet it is significantly influenced by the quality of the training data (Li et al. 2023; Xie et al. 2024a). Typically, the training data is sourced from extensive text collections such as internet crawls (Patel 2020). However, the quality of such data is frequently inconsistent (Kreutzer et al. 2022). Therefore, the identification and selection of high-quality data from these vast resources is a critical consideration for achieving optimal model training.

To effectively address this issue, a sub-problem that needs to be solved first is how to define high-quality data. Current methodologies typically adopt one of two approaches: the first involves devising a metric to assess the quality of the text, focusing on attributes such as textual fluency (Marion et al. 2023; Muennighoff et al. 2024); the second approach involves manually curating a subset from the available corpus, which is considered high-quality based on human experience, to serve as a standard reference, with Wikipedia articles often being chosen for this purpose (Xie et al. 2024b; Engstrom, Feldmann, and Madry 2024). Techniques derived from the first approach include strategies to eliminate redundant data (Lee et al. 2022) or employing existing models to compute metrics like perplexity (PPL) (Marion et al. 2023) or selfinfluence scores (Thakkar et al. 2023) for the dataset. On the other hand, the latter includes the development of models to assign quality scores (Brown et al. 2020) to data or the determination of significance weights (Xie et al. 2024b) to guide the sampling process.

In practice, the mentioned selection criteria often favor fluent texts that may lack knowledgeable information, as shown in the Appendix Figure 9. This observation inspires us to propose a novel approach: High Knowledge Scorer (HKS), which assesses text quality by detecting the knowledge content of each text sample. We begin by quantifying the knowledge encapsulated within the texts of training data. Unlike the structured knowledge definitions in Allen-Zhu and Li (2024), we simplify knowledge into knowledge elements to facilitate quicker knowledge annotation. Following this, we create a multi-domain knowledge element pool consists of 5M knowledge elements from various sources. We employ a multiple pattern matching algorithm to identify all the knowledge elements contained in each text sample and introduce two metrics, knowledge density and knowledge coverage, to facilitate quantitative analysis. Leveraging these metrics, we propose a comprehensive knowledge scorer to select high-knowledge texts, which are positively correlated with these two metrics. Benefiting from the categorization of each knowledge element during the creation of the multidomain knowledge element pool, our approach can select high-knowledge data aligned to the desired domain.

We train a 1.1B model from scratch on a 20B bilingual dataset. We find that: (1) In comparison to baseline methods, our method excels both in knowledge-intensive tasks and general understanding tasks, with an average improvement of 2.37 pp over random selection. We further validate this conclusion through continual pretraining experiments on Llama-3-8B, observing an average increase of $2 . 4 \mathrm { p p }$ . (2) When applied to specific domains, our method can enhance the model’s performance in the desired domain, with an absolute improvement of up to $2 . 5 \mathrm { p p }$ compared to baseline.

![](images/c0f90580903aceb7e7c31d84aa5c0dc900e63b935b23228dff61e2f37f56b168.jpg)  
Figure 1: The overall framework of HKS. Our methodology begins with sourcing knowledge from Wikipedia articles and academic literature. We extract knowledge elements, which are categorized into several domains. Each text sample from the raw dataset is then characterized by its knowledge density and coverage, resulting in a knowledge score for each text. Texts with higher scores are identified as high-knowledge data, selected via top- $k$ selection or weighted sampling.

Our contributions can be summarized as follows:

• We simplify the definition of knowledge and introduce knowledge element to facilitate knowledge parsing of texts, which guides us to establish a multi-domain knowledge element pool. Moreover, we propose knowledge density and coverage as metrics to quantify the knowledge in texts. • We propose a novel and gradient-free knowledge scorer for selecting high-knowledge data, which is comprehensive and proportional to knowledge density and coverage. A series of experiments consistently showcase the superior performance of our knowledge scorer. • We propose a domain-aware high-knowledge data selection method for domain-specific augmentation. Experimental results demonstrate the effectiveness of our approach.

# 2 Methodology

# 2.1 Overview of Our Approach

Figure 1 illustrates the pipeline of our method. Initially, we used Wikipedia and academic literature as knowledge sources. Knowledge elements are extracted and classified using GPT4 (Achiam et al. 2023) and a BERT-based model. For each text in the pre-training dataset, we enumerate all the included factual knowledge elements, which are then used to determine the two knowledge metrics: density and coverage. A knowledge scorer, proportional to these two metrics, assigns comprehensive scores to each text, which is further considered as data selection criteria.

# 2.2 Knowledge Element Pool Construction

Knowledge is usually represented as structured (name, relation, value) triplets in knowledge graphs (Zhu and Li 2023; Pan et al. 2024), but parsing triplets across the entire corpus is extremely time-consuming and challenging. To simplify this process, we reduce knowledge triplet to knowledge element:

Definition 1 A $n$ -gram noun, represented by several tokens $( t _ { 1 } , t _ { 2 } , \ldots , t _ { n } )$ , can be considered as a knowledge element if it encapsulates or is closely associated with a specific concept, fact, theory, principle, definition, and so forth.1

Based on the above definition, we build a knowledge element pool that covers multiple domains. The knowledge elements are derived from two sources: 1) Wikipedia documents 2) Academic article datasets. We first add Wikipedia entries and keywords of academic articles to the knowledge element pool, where the academic dataset we use is the OAG dataset (Zhang et al. 2022). The knowledge elements obtained amount to a total of 20M. While these elements are highly specialized, they lack flexibility. Therefore, we also choose to write instructions to guide GPT4 in extracting knowledge elements contained in the Wikipedia documents2. The prompt is listed in Appendix C.

We categorize the knowledge elements into five domains: science, society, culture, art, and life. Given that individual knowledge elements are difficult to label directly, we match related documents for each knowledge element as auxiliary materials for judgment. Then, we manually annotate 10K knowledge elements and take $20 \%$ of the annotated data as the test set. We train a BERT-based (Devlin et al. 2019) labeling model with a labeling accuracy of $9 6 . 2 \%$ .

![](images/a600015612f05c6f7bd7fb9c9838f625562e36d2cb3c9cbe153207021608dfbb.jpg)  
Figure 2: Score density in Pile subsets. There is a noticeable difference in the distribution of knowledge density $( d )$ and coverage $( c )$ within Pile subsets. $d$ tends to favor samples from Wikipedia, while $c$ tends to favor samples from books and ArXiv.

After knowledge categorization, we write an additional instruction and use GPT4 to review all the knowledge elements, filtering out those that do not align with the annotated categories or are of poor quality. Additionally, we remove knowledge elements with a string length of less than 2. After deduplication, we finally built a high-quality knowledge element pool containing 5M terms. More detailed construction process and statistics can be found in Appendix E.

# 2.3 Knowledge Content Evaluation within the Text

For each text in the training data, we label all the knowledge elements that appear in it3. For quantitative analysis of the pretraining corpus, we establish the following two knowledge metrics.

Definition 2 Given a text sample $x$ , $n _ { k }$ is the total number of knowledge elements in the sample, and $n _ { p }$ is the text token length, Knowledge Density $( d )$ is defined as $d ( x ) = n _ { k } / n _ { p }$ .

Definition 3 Given a text sample $x$ , $\widetilde { n } _ { k }$ is the total number of non-duplicated knowledge elementsein the sample, and $N _ { k }$ is the total number in the full knowledge pool. Knowledge Coverage (c) is defined as $\dot { c } ( x ) = \widetilde { n } _ { k } / \bar { N } _ { k }$ .

Knowledge density is used toequantify the amount of knowledge contained in a text sample. We do not use the total token count of the knowledge elements as $n _ { k }$ to avoid any bias arising from longer knowledge elements. Knowledge coverage serves as an indicator of the diversity of knowledge within a text sample.

We compute two metrics for the texts in the Pile dataset, and present cases in Appendix Figure 9. After separately selecting the top 1M samples from the Pile dataset based on density and coverage, we observe that articles with high knowledge density tend to have an average length of 20,950 tokens, whereas those with high knowledge coverage are typically much shorter, averaging only 811 tokens. This suggests that $d$ and $c$ are two significantly distinct metrics. To examine the observations, we analyze the distribution of $d$ and $c$ across various subsets within the Pile. We select several subsets and draw 10K random samples from each subset. The samples are then divided into buckets based on the minimum and maximum values of the metrics, and we count the number of samples falling into each bucket. Figure 2 shows the results, revealing distinct preferences of $d$ and $c$ across different subsets. Notably, $d$ exhibits a preference for subsets such as FreeLaw and Wikipedia, whereas $c$ prefers Arxiv.

# 2.4 Knowledge-Based Data Selection

Generic high-knowledge scorer According to the aforementioned results, we decide to combine these two orthogonal metrics to create a comprehensive knowledge scorer. Specifically, for a text sample $x$ , we materialize the scorer as a scoring function:

$$
s c o r e ( x ) = \phi ( d ( x ) , c ( x ) )
$$

As we have seen in the extraction results (Section 2.3), these two metrics lead to two completely different extraction results with less entanglement, so we assume that $d$ and $c$ are two variables independent of each other, and we can simplify the function $\phi$ to a product form:

$$
s c o r e ( x ) = f ( d ( x ) ) \cdot g ( c ( x ) )
$$

For $f$ and $g$ , we give some empirical assumptions:

1. $f$ and $g$ should be incremental functions, as we suggest that texts abundant in knowledge information generally yield high scores on knowledge density and coverage.   
2. When the values of $d$ and $c$ are high, their incremental contributions to the overall effect are not expected to be as significant as when their values are low. This implies that the functions $f$ and $g$ do not exhibit upward concavity:

$$
{ \frac { \partial ^ { 2 } f } { \partial d ^ { 2 } ( x ) } } ( d ( x ) ) \leq 0 , { \frac { \partial ^ { 2 } g } { \partial c ^ { 2 } ( x ) } } ( c ( x ) ) \leq 0
$$

We have experimented with various combinations of functions, and the details can be found in Appendix D. The scoring formula that we ultimately chose is as follows:

$$
s c o r e ( x ) = d ( x ) \cdot l n ( c ( x ) + 1 )
$$

The selection cases through our knowledge scorer are displayed in Appendix Figure 9 and 10. Besides, we also analyze the score distribution in Pile. The results in Figure 2 show that samples from Book3, Arxiv, and FreeLaw achieve higher HKS scores compared to those from DM Mathematics.

Domain-specific knowledge scorer Given that our knowledge elements are categorized, we are able to perform domain-specific knowledge scoring and select highknowledge data belong to that domain, thereby achieving domain-specific enhancement. Specifically, we constrain our knowledge elements into the target domain $m$ , therefore obtaining the knowledge density $d _ { m }$ and coverage $c _ { m }$ for specific domain4. Similar to the generic knowledge scorer, we can evaluate each text using a domain-specific scoring function for domain $m$ :

$$
s c o r e _ { m } ( x ) = d _ { m } ( x ) \cdot l n ( c _ { m } ( x ) + 1 )
$$

Filtering strategies After scoring each text in the pretraining dataset, we adopt two methods for high-knowledge data selection.

• Top- $k$ : We select the top- $k$ text samples based on our defined scores, with some selection cases displayed in Appendix Figure 9 and 10. It is evident that texts with high scores are more knowledgeable and of higher quality, while samples with low scores contain less knowledge content. • Sampling: In addition to the top- $k$ selection technique, various studies have highlighted the efficacy of sampling methods (Sachdeva et al. 2024; Wettig et al. 2024). In our research, we employ softmax-based sampling strategies: We treat the normalized score as an importance weight, and apply the softmax function to each sample $x _ { i }$ in the pre-training dataset to calculate sampling probability:

$$
P ( x _ { i } ) = { \frac { e x p { \bigl ( } { \frac { s c o r e _ { i } } { \tau } } { \bigr ) } } { \sum _ { j } e x p { \bigl ( } { \frac { s c o r e _ { j } } { \tau } } { \bigr ) } } }
$$

$\tau$ is the temperature term. We perform sampling without replacement and utilize the Gumbel top- $k$ trick (Kool, Van Hoof, and Welling 2019) to facilitate the sampling process. Here we choose $\tau = 2$ .

# 3 Experiments

# 3.1 Setups

Dataset We utilize the Pile (Gao et al. 2020) and Wudao (Yuan et al. 2021) datasets as our pre-training dataset for training a bilingual language model. Pile is an extensive English text corpus that includes 22 diverse subsets, while Wudao consists of Chinese passages collected from web sources. We extract 10B tokens from each, resulting in a 20B token bilingual dataset, which aligns with the compute-optimal amount from previous studies (Hoffmann et al. 2022).

![](images/c60402b166cf7512fe6d1df7d68c55e2d26dd011ef15476a9d54c130324e0235.jpg)  
Figure 3: Perplexity evaluation on Wudao validation dataset.

Model and training We train a model of 1.1B parameters, which has the same architecture of Bloom (Le Scao et al. 2023). We train our model in one epoch, with a cosine learning rate scheduler. We use a global batch size of 2048 with gradient accumulation and a max context window length of 2048. We use Megatron framework to train our model in 16 A100 GPUs, with fp16 setting, which needs 21 hours to finish our training. More details can be found in Appendix F.

Baselines We compare our method with the following baselines: (1) Random: Data is selected randomly from each source dataset with uniform probability. (2) density $d$ and coverage $c$ : We utilize density $d$ and coverage $c$ as the criteria for data selection, respectively. (3) HKS inverse: In our ablation studies, we conduct experiments where we select the texts with the lowest scores, as determined by the HKS, to train the model. (4) Perplexity (PPL): In line with the methodology outlined in Marion et al. (2023), we employ Bloom 1.1B model (Le Scao et al. 2023) to calculate the perplexity (PPL) for each data sample. We then retain the top- $k$ samples exhibiting the lowest PPL values. (5) Error $L 2$ -Norm (EL2N): Addition to perplexity, we also calculate the error l2-norm for each data sample (Marion et al. 2023). We then retain the top- $k$ samples exhibiting the lowest EL2N values. (6) Data Selection via Importance Resampling (DSIR): Following the methodology outlined in Xie et al. (2024b), we use bigram representation to compute hash features for each sample. We utilize documents from Wikipedia as the target domain to compute the importance weight.

Benchmarks and metrics We train all the models three times using different random seeds and report the average results. We conduct a holistic assessment of all the methods:

• We measure perplexity on both Pile and Wudao datasets. For the Pile dataset, we extract 10K samples from each subset to serve as a validation dataset, ensuring these samples are not encountered during the training process. Since the Wudao dataset does not have a predefined subset split, we divide it according to categories of the included knowledge elements. We then apply the same validation process as with the Pile dataset, extracting samples for evaluation. • We assess downstream task performance using in-context learning (Dong et al. 2022). For knowledge-intensive tasks, we conduct evaluations on ARC-C (Bhakthavatsalam et al.

Table 1: Few-shot results of downstream tasks. Bold indicates the best result in each column. All results are averaged over 3 seeds, with standard deviations indicated in subscripts. Signed numbers indicate the difference in scores from the random baseline.   

<html><body><table><tr><td rowspan="2" colspan="2">Method</td><td colspan="2">English Tasks</td><td colspan="2">Chinese Tasks</td><td rowspan="2">AVG.</td></tr><tr><td>Knowledge intensive</td><td>General understanding</td><td>Knowledge intensive</td><td>General understanding</td></tr><tr><td colspan="2">Random</td><td>23.770.06 +0.00</td><td>47.560.19 +0.00</td><td>25.150.28 +0.00</td><td>28.660.14 +0.00</td><td>32.490.10 +0.00</td></tr><tr><td colspan="2">PPL EL2N</td><td>24.440.07 +0.67</td><td>45.920.28 -1.64</td><td>24.460.20 -0.69</td><td>22.990.06 -5.67</td><td>29.410.25 -3.08</td></tr><tr><td colspan="2">DSIR</td><td>24.520.18 +0.75</td><td>48.230.22 +0.67</td><td>26.110.10 +0.96</td><td>24.340.10 -4.32</td><td>30.840.23 -1.65</td></tr><tr><td></td><td></td><td>19.260.20 -4.51</td><td>48.120.03 +0.56</td><td>25.610.07 +0.46</td><td>23.970.23 -4.69</td><td>29.750.02 -2.74</td></tr><tr><td>d</td><td>Sampling</td><td>24.340.09 +0.57</td><td>48.040.27 +0.48</td><td>25.460.27 +0.31</td><td>26.340.15 -2.32</td><td>31.640.14 -0.85</td></tr><tr><td></td><td>Top-k</td><td>21.720.21 -2.05</td><td>43.750.20 -3.81</td><td>24.790.01 -0.36</td><td>24.390.23 -4.27</td><td>29.110.11 -3.38</td></tr><tr><td>C</td><td>Sampling</td><td>23.700.29 -0.07</td><td>43.080.29 -4.48</td><td>26.750.14 +1.60</td><td>26.710.29 -1.95</td><td>30.550.18 -1.94</td></tr><tr><td></td><td>Top-k</td><td>25.270.29 +1.50</td><td>43.050.05 -4.51</td><td>26.210.01 +1.06</td><td>27.450.27 -1.21</td><td>31.080.20 -1.41</td></tr><tr><td>HKS</td><td>Inverse</td><td>20.290.00 -3.48</td><td>46.120.11 −1.44</td><td>26.550.25 +1.40</td><td>25.210.21 -3.45</td><td>30.080.28 -2.41</td></tr><tr><td></td><td>Sampling</td><td>24.890.11 +1.12</td><td>43.610.22 -3.95</td><td>26.740.19 +1.59</td><td>26.980.24 -1.68</td><td>31.000.02 -1.49</td></tr><tr><td></td><td>Top-k</td><td>26.320.11 +2.55</td><td>48.930.24 +1.37</td><td>27.370.16 +2.22</td><td>32.000.26 +3.34</td><td>35.070.26 +2.58</td></tr></table></body></html>

2021), OpenBookQA (Mihaylov et al. 2018), MMLU (Hendrycks et al. 2020), CMMLU (Li et al. 2024), and C-Eval (Huang et al. 2023). Additionally, we evaluate the model’s general understanding capabilities on a range of tests, including RTE (Wang et al. 2018), BBH (Suzgun et al. 2023), WiC (Pilehvar and Camacho-Collados 2019), COPA (Roemmele, Bejan, and Gordon 2011), BoolQ (Clark et al. 2019) and sub-tasks derived from CLUE (Xu et al. 2020) and FewCLUE (Xu et al. 2021). These test sets encompass both English and Chinese languages. We summarize our test results on knowledge intensive and general understanding tasks, more details can be found in Appendix G.

# 3.2 Main Results

Table 1 details the results of our main tests. (1) Firstly, we can find that our HKS outperforms the baseline models in most tasks, demonstrating that high-knowledge data can improve the performance of LLMs. Notably, HKS exhibits superior performance relative to PPL, EL2N, and random sampling in terms of average score, with 2.37 pp improvement compared to random sampling, which shows the efficacy of our knowledge scorer. Furthermore, HKS outperforms DSIR in knowledge-intensive tasks, achieving a 0.81 pp improvement. While DSIR uses Wikipedia passages as its target domain, which are rich in knowledge, HKS demonstrates greater efficacy in selecting high-knowledge data. (2) In addition, the performance of the $d$ and $c$ baselines is inferior to that of HKS, which underscores the importance of integrating density and coverage into a comprehensive knowledge scorer. On the other hand, inverse selection of HKS yields lower results than a random baseline, further affirming the efficacy of our approach from a different perspective.

We report the results of the two filtering methods, top- $k$ and sampling. The results show that except for $d$ , top- $k$ is better than sampling, which indicates that in the dimension of knowledge, the top-ranked data do have higher quality than lower-ranked, indirectly reflecting that our knowledge scorer can accurately identify the high-quality data in the dataset.

Table 2: Comparison of continual pretrained models.   

<html><body><table><tr><td>Method</td><td>Tokens</td><td>MMLU</td><td>CMMLU</td><td>CEVAL</td></tr><tr><td>Llama-3-8B</td><td>/</td><td>65.8</td><td>51.5</td><td>50.8</td></tr><tr><td>+Random</td><td>100B</td><td>65.9</td><td>53.7</td><td>52.5</td></tr><tr><td>+ HKS</td><td>100B</td><td>66.4</td><td>57.5</td><td>55.2</td></tr></table></body></html>

We also present the performance of our model on perplexity in Figures 3 and 4. On the Pile dataset, our perplexity is marginally higher than that of the $c$ baseline; however, HKS outperforms the $c$ baseline regarding downstream task scores, indicating that perplexity may not be strictly positively correlated with downstream task performance. In the context of specific subsets, our model records a lower perplexity on the Wikipedia and book corpora, which are generally regarded as high-quality sources abundant in knowledge.

# 3.3 Analysis

Extending to Larger Scale To conduct larger-scale verification, we select 100B tokens from the Pile and Wudao datasets to continue training on Llama-3-8B (Dubey et al. 2024). The results are reported in Table 2. Compared to random selection, our approach results in improvements of 0.9 pp, $3 . 8 ~ \mathrm { p p }$ , and $2 . 7 \ \mathrm { p p }$ on MMLU, CMMLU, and CEval, respectively. Furthermore, in comparison to the original Llama-3-8B, our model, post-continual training, exhibits a significant enhancement in knowledge-intensive tasks. The results demonstrate that HKS can be effectively applied to larger datasets and models.

Higher Knowledge Richness in Data Benefits Model Performance To further investigate the effects of knowledge richness in data on our final model performance, we sort the texts in Pile and Wudao from high to low according to their HKS scores, and then employ a top- $k$ strategy to select the highest-scoring portion of the data so that the total length of their tokens is 10B, respectively. The score of the lowestscoring sample in the selected data is determined to be the threshold for the division of high-knowledge data and lowknowledge data. Then we define $\begin{array} { r } { \alpha = \frac { N _ { h } } { N _ { h } + N _ { l } } } \end{array}$ , where $N _ { h }$ and $N _ { l }$ denote the number of tokens of high and low-knowledge data, respectively. According to the $\alpha$ , we perform uniform sampling from both the high and low knowledge portions. The sampled subsets are then merged to form a 20B token dataset, which is utilized for training our model.

![](images/1c2208dc659950de3e93c1d833fabe9541b3af22a8b24ba1465f666fd7a82f4f.jpg)  
Perplexity on Pile   
Figure 4: Perplexity evaluation on Pile validation dataset.

Table 3: Impact of data quality on model performance. We train models on the merged datasets, varying the proportion of high-knowledge data $\alpha$ .   

<html><body><table><tr><td>α</td><td>MMLU</td><td>CMMLU</td><td>C-Eval</td><td>BBH</td><td>AVG.</td></tr><tr><td>1.00</td><td>27.91</td><td>27.85</td><td>26.89</td><td>29.66</td><td>28.08</td></tr><tr><td>0.75</td><td>27.07</td><td>27.90</td><td>26.60</td><td>26.38</td><td>26.67</td></tr><tr><td>0.50</td><td>25.78</td><td>26.63</td><td>25.47</td><td>26.98</td><td>26.53</td></tr><tr><td>0.25</td><td>26.67</td><td>26.11</td><td>25.50</td><td>26.62</td><td>26.23</td></tr><tr><td>0.00</td><td>25.90</td><td>25.25</td><td>24.85</td><td>27.17</td><td>25.79</td></tr></table></body></html>

The results outlined in Table 3 demonstrate a trend of diminishing average performance of both knowledge-intensive (MMLU, CMMLU, C-Eval) and reasoning (BBH) tasks, as we move from a dataset consisting entirely of highknowledge data $\stackrel { \cdot } { \alpha } = 1 . 0 0 )$ to the one that is solely comprised of low-knowledge data ${ \mathit { \ ' } } \alpha = 0 . 0 0 { \mathit { \check { \Psi } } }$ ). This indicates that the knowledge content of the data not only affects the model’s ability to memorize knowledge but is also potentially linked to the model’s reasoning abilities. In addition, each benchmark responds differently to changes in the knowledge content of data. For instance, the results of CMMLU show an increase in performance when the dataset includes a mixture of high and low-knowledge data $( \alpha = 0 . 7 5 )$ ), whereas the results of MMLU, C-Eval, and BBH tend to perform better with higher proportions of high-knowledge data.

Domain-Specific Enhancement Results We explore the application of the HKS model to specific domains, taking the enhancement of the science domain as an illustrative example. We use the Equation 5 to cherry-pick data rich in scientific knowledge. Similar to Section 3.3, we distinguish between high and low-scientific knowledge data by a score threshold at 10B. Subsequently, we define β = N N+hsN where $N _ { h s }$ and $N _ { l s }$ denote the token count from high and low-scientific knowledge data, respectively. We follow $\beta$ to perform uniform sampling across these two distinct parts and finally mixed into 20B token training data. We also categorize the questions in MMLU and CMMLU into scientifically relevant and irrelevant sections5, and evaluate the model’s performance in these different partitions.

Table 4: We carry out experiments focusing on scientific knowledge enhancement, where $\beta$ signifies the proportion of high scientific knowledge data within the entire dataset.   

<html><body><table><tr><td rowspan="2">B</td><td colspan="2">MMLU</td><td colspan="2">CMMLU</td></tr><tr><td>Science</td><td>Others</td><td>Science</td><td>Others</td></tr><tr><td>1.00</td><td>27.29</td><td>24.16</td><td>28.88</td><td>24.52</td></tr><tr><td>0.75</td><td>26.63</td><td>24.18</td><td>28.65</td><td>25.36</td></tr><tr><td>0.50</td><td>25.56</td><td>24.97</td><td>27.26</td><td>25.10</td></tr><tr><td>0.25</td><td>25.62</td><td>23.81</td><td>26.38</td><td>25.15</td></tr><tr><td>0.00</td><td>22.69</td><td>23.53</td><td>26.22</td><td>25.10</td></tr><tr><td>Random</td><td>26.52</td><td>24.78</td><td>26.39</td><td>25.12</td></tr></table></body></html>

The results are summarized in Table 4. We can find that: (1) Our domain-specific HKS is effective in selecting highknowledge data in the desired domain. Within the Science category of the MMLU and CMMLU, the optimal performance is attained when the value of $\beta$ is set to 1.00, with improvements of 0.77 pp and 2.49 pp compared to the random baseline, respectively. The results strongly imply that there is a direct correlation between the amount of HKS-selected domain-specific data and the final result within that domain, indirectly underscoring the effectiveness of domain-specific enhancement through our knowledge scorer. (2) Conversely, the most suboptimal performance across all categories is observed when $\beta$ is set to 0.00. This situation is marked by the absence of high-scientific knowledge data in the training dataset, indicating that such data significantly contribute to the overall performance of the model.

![](images/4a4fc966cab1001c61cd7193c51539716fde316ac20aff81fd7c4bf951389beb.jpg)  
Figure 5: We compare the costs of the various approaches based on cloud server rental fees.

![](images/5e5c083c57ee0a20bc76dd3061cec3d879d342a28a0750eb48dc9e758c3d7f9e.jpg)  
Figure 6: We use Spearman’s rank correlation to assess the relation between various methods.

HKS Achieves Superior Cost Efficiency Our methodology employs a gradient-free knowledge scorer, which enables our scoring program to run efficiently on a CPU machine. This offers significant cost and time advantages compared to methods such as Perplexity (Marion et al. 2023) or modelbased scoring (Li et al. 2023). To facilitate a more equitable comparison, we consult the rental rates for CPU and GPU servers on Azure6 and present the costs of the different methods in Figure $5 ^ { 7 }$ . Our method incurs considerably lower expenses than the PPL/EL2N, and although it is marginally more costly than DSIR, it delivers superior results.

Score Correlation Analysis To investigate the correlation between our method and the baseline methods, we extract a portion of the training data for analysis. We randomly sample 500K texts from the Pile dataset and label this subset with perplexity, error L2-norm, DSIR, $d , c ,$ , and HKS scores. Then we calculate Spearman’s rank correlation coefficient (Gauthier 2001) between these scoring methods pairwise.

From the results depicted in Figure 6, we can find out that: (1) The correlations between the HKS and baseline methods are remarkably low, trending towards a negative correlation. Perplexity is frequently employed as a metric for evaluating linguistic fluency, suggesting that high-knowledge and fluency do not necessarily correlate strongly. Nevertheless, the HKS still delivers superior performance, indicating that compared to text fluency, high-knowledge data are more beneficial for model training. (2) There is a low correlation between $d$ and $c$ , indicating that these dimensions are relatively orthogonal to each other, which is consistent with our observation in Section 2.3. (3) HKS scores show a strong correlation with both $d$ and $c$ , indicating that our scoring function effectively synthesizes metrics along these two dimensions.

# 4 Related Works

High-quality data selection Research on high-quality training data selection falls into two approaches: metric-based selection and reference data-based selection. The former typically employs a manually defined metric to assess data quality. Rule-based methods (Rae et al. 2021; Dodge et al. 2021; Cai et al. 2024) and deduplication (Lee et al. 2022; Touvron et al. 2023) are two widely used approaches, which are straightforward but may not comprehensively evaluate data quality. Model-based methods leverage language models to derive data attributes, such as perplexity (Marion et al. 2023), selfinfluence (Thakkar et al. 2023), or density (Sachdeva et al. 2024). Several researchers have also explored directly querying LLMs to give data quality scores (Sachdeva et al. 2024; Wettig et al. 2024). While these methods may enhance the model’s performance, they have not taken into account the data from the perspective of knowledge content.