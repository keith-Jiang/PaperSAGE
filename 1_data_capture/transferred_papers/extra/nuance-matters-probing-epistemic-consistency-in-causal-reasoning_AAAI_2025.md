# Nuance Matters: Probing Epistemic Consistency in Causal Reasoning

Shaobo ${ { \bf { C u i } } ^ { 1 } }$ , Junyou $\mathbf { L i } ^ { 2 }$ , Luca Mouchel1, Yiyang Feng1, Boi Faltings

1EPFL, Switzerland 2University of Waterloo, Canada shaobo.cui@epfl.ch, j2626li $@$ uwaterloo.ca, luca.mouchel@epfl.ch, yiyang.feng@epfl.ch, boi.faltings@epfl.ch

# Abstract

Previous research on causal reasoning often overlooks the subtleties crucial to understanding causal reasoning. To address this gap, our study introduces the concept of causal epistemic consistency, which focuses on the self-consistency of Large Language Models (LLMs) in differentiating intermediates with nuanced differences in causal reasoning. We propose a suite of novel metrics – intensity ranking concordance, cross-group position agreement, and intra-group clustering – to evaluate LLMs on this front. Through extensive empirical studies on 21 high-profile LLMs, including GPT4, Claude3, and LLaMA3-70B, we have favoring evidence that current models struggle to maintain epistemic consistency in identifying the polarity and intensity of intermediates in causal reasoning. Additionally, we explore the potential of using internal token probabilities as an auxiliary tool to maintain causal epistemic consistency. In summary, our study bridges a critical gap in AI research by investigating the self-consistency over fine-grained intermediates involved in causal reasoning.

# Code — https://github.com/cui-shaobo/causal-consistency Extended version — https://arxiv.org/abs/2409.00103

# 1 Introduction

Previous studies in causal reasoning have primarily focused on discovering or determining the existence of a causal relationship between two variables (Roemmele, Bejan, and Gordon 2011; Cui et al. 2024b). However, these causal relationships are not always absolute. They can be heavily influenced by additional intermediate factors, which may vary in both polarity and intensity (Fitzgerald and Howcroft 1998; Bauman et al. 2002). The polarity of these intermediates indicates whether they support or defeat (oppose) the original causal relationship, while their intensity determines the strength of this supporting or defeating influence.

Forming fine-grained differentiation is essential for precise causal modeling (Iwasaki and Simon 1994); however, it is insufficient for LLMs to merely generate these intermediates. It is as equally important to ensure that these intermediates are reliable and credible (Shi et al. 2023). One method to verify this is through assessing the consistency

Cause Effect Increasingly close global Intermediates Widening income disparity economic integration. $D P \longmapsto$ within and between nations.   
For a cause-effect pair, LLMs are required to write fine-grained   
intermediates, including:   
Supporters : $a _ { 1 }$ a2 an   
Defeaters $d _ { 1 }$ $d _ { 2 }$ dr   
For these intermediates, can LLMs maintain self-consistency   
in discerning:   
$\textcircled{8}$ Intensity 2 Polarity Clustering

Figure 1: Overview of the evaluation framework for causal epistemic consistency. The first step involves instructing LLMs to generate fine-grained intermediates that influence a given causal relationship differently. The second step requires LLMs to rank their own generations based on their causal nuance. Finally, the proposed metrics are used to assess the self-consistency between ranking and generation, i.e., the LLMs’ causal epistemic consistency.

of LLMs’ perception of the intermediates. We posit that if LLMs can correctly differentiate their generated intermediates based on varying polarities and intensities, these intermediates are self-consistent and thus, more reliable for making predictions and decisions. Drawing from this insight, our study proposes the concept of “causal epistemic consistency”:

Definition 1 (Causal epistemic consistency) Causal epistemic consistency refers to an LLM’s ability to maintain selfconsistency in differentiating its generated intermediates in three aspects: (i) discerning intensity: accurately assessing the intensity nuance in their causal impact. (ii) differentiating polarity: effectively distinguishing between supporting and defeating intermediates, and (iii) forming cohesive clusters: creating well-separated clusters of intermediates based on their polarity and intensity.

To quantify LLMs’ ability to maintain causal epistemic consistency in the aforementioned aspects, we introduce a suite of novel metrics. These metrics include (i) Intensity ranking concordance, which measures the models’ selfconsistency in ranking self-generated intermediates with varying intensity; (ii) Cross-group position (CGP) agreement, which indicates the models’ consistency in determining the polarity of intermediates, specifically whether they support or defeat the original causal relationship; and (iii) Intra-group clustering (IGC), which assesses models’ consistency to rank its generated intermediates of the same type closely together. We illustrate the evaluation framework of causal epistemic consistency in Figure 1.

To unravel the causal epistemic consistency of current LLMs, our empirical study evaluates 21 high-profile LLMs, including the renowned closed-source GPT, Claude, and Gemini series, alongside various scales of cutting-edge open-source alternatives such as Gemma (2B and 7B) (Mesnard et al. 2024), LLaMA2 (7B, 13B, and 70B) (Touvron et al. 2023), Phi-3 (3.8B, 7B, and 14B) (Abdin et al. 2024), and LLaMA3 (8B and 70B) (Meta 2024). Our findings reveal their striking incompetence in keeping causal epistemic consistency. Remarkably, even the advanced GPT-4 model performs unsatisfactorily. This underscores the complexities and challenges these models face in maintaining causal consistency and capturing causal nuances.

Furthermore, we explore whether internal token probability can serve as a useful signal for LLMs to maintain causal epistemic consistency. Our comprehensive empirical study highlights the application scope of internal token probability for LLMs to maintain causal epistemic consistency.

To summarize, our contributions are fourfold:

1. Introduction of Causal Epistemic Consistency: We propose the novel concept of causal epistemic consistency over fine-grained intermediates in causal reasoning, emphasizing self-consistency in differentiating the nuances hidden in fine-grained intermediates.   
2. Development of Evaluation Metrics: We introduce a comprehensive suite of metrics designed to assess LLMs’ causal epistemic consistency, covering aspects of intensity ranking concordance, cross-group position agreement, and intra-group clustering.   
3. Extensive Empirical Evaluation: We assess the performance of 21 LLMs on their causal epistemic consistency, highlighting their deficiencies in maintaining causal epistemic consistency.   
4. Internal Token Probability Exploration: We investigate the potential of using internal token probabilities as an auxiliary tool to help LLMs maintain causal epistemic consistency and highlight its application scope.

# 2 Task Definition

# 2.1 Problem Formulations

Causal epistemic consistency measures an LLM’s selfconsistency between generating fine-grained intermediates and subsequently ranking those fine-grained intermediates.

Specifically, in the generation phase, for a defeasible cause-effect pair $( C , E )$ , an LLM is tasked with generating an ordered sequence $\boldsymbol { \mathcal { T } }$ of finegrained intermediates, consisting of a subsequence $\mathcal { D } = ( \mathcal { I } _ { 1 } , \mathcal { I } _ { 2 } , \cdot \cdot \cdot , \mathcal { I } _ { m } )$ as the defeater group and a subsequence $\textbf { \em A } = ~ ( \mathbb { Z } _ { m + 1 } , \mathbb { Z } _ { m + 2 } , \cdot \cdot \cdot , \mathbb { Z } _ { m + n } )$ as the supporter group. Each individual intermediate changes the causal strength of $( C , E )$ differently. Specifically, the causal influence of these intermediates is expected in the following order:

$$
\begin{array} { r l } & { \qquad c S ( E | C \oplus \mathbb { Z } _ { 1 } ) \leq \cdots \leq { \mathcal { C S } } ( E | C \oplus \mathbb { Z } _ { m } ) } \\ & { \qquad \leq { \mathcal { C S } } ( E | C ) } \\ & { { \mathcal { C S } } ( E | C \oplus \mathbb { Z } _ { m + 1 } ) \leq \cdots \leq { \mathcal { C S } } ( E | C \oplus \mathbb { Z } _ { m + n } ) } \end{array}
$$

where $\mathcal { C } \mathcal { S } ( E | C )$ measures the causal strength (Luo et al. 2016; Zhang et al. 2022), quantifying the likelihood that the cause event $C$ would lead to the occurrence of the effect event $E$ . 1 The $\oplus$ means the combination of two events.

Subsequently, in the ranking phase, the same LLM is asked again to rank its own generated intermediates $\boldsymbol { \mathcal { T } }$ , obtaining $\mathcal { T } ^ { \prime }$ , a permutation of $\boldsymbol { \mathcal { T } }$ . Ideally, an LLM with perfect causal epistemic consistency should have $\mathcal { T } = \mathcal { T } ^ { \prime }$ , satisfying the requirements of intensity, polarity, and clustering perfectly.

# 2.2 Key Research Questions

The study addresses three primary research questions:

• RQ I: How can we comprehensively measure the ability of LLMs to maintain the epistemic consistency over finegrained intermediates in causal reasoning? • RQ II: How well do current LLMs, with varying architectures and scales, maintain their causal epistemic consistency? • RQ III: Are there any alternatives to prompting for LLMs to maintain causal epistemic consistency?

To answer RQ I, we propose novel metrics introduced in Section 3, which not only serve our specific study but also have broader applications across various tasks. In Section 4, we dive into the performance of twenty-one leading LLMs, exploring their ability to maintain epistemic consistency, thereby addressing RQ II. Lastly, in Section 5 , we assess whether internal token probability offers a more effective—or perhaps less effective—alternative to prompting for preserving causal epistemic consistency in LLMs, answering RQ III.

# 3 Metrics for Measuring Causal Epistemic Consistency

To evaluate the causal epistemic consistency of LLMs from the aspects of intensity, polarity, and clustering, we propose three types of automatic metrics: intensity ranking concordance, cross-group position agreement, and intra-group clustering. A graphical illustration of these metrics is shown in Figure 2. The mathematical notations below are consistent with Section 2.1.

LLM's -3 -1 $\mathbf { + } 1$ $+ 2$ $\textcircled{+ 3}$ An LLM with perfect causal epistemic consistency   
Generation ability should maintain the same order in its generation Ranking LLM's -2 $\mathbf { + } 1$ $+ 3 + 2 + 4 + 5$ and ranking phases.   
Intensity -2 $\therefore + 3 + 2 1 + 4 + 5$ aTlhseo ehlaevmeeanthiwgihthinhtiegnhsiitnytienstihtey ipnretdhiectgednrearnatkiiongsohroduelrd.   
Polarity $\therefore \bar { Q }$ $\mathrm { ~ O ~ O ~ O ~ O ~ }$ tEhacnhesaucphpdoertferatelreemlenmtesnht.ould have a higher ranking Clo   
Clustering □ Within each group(supporters and defeaters), the elements should be clustered closely. Close Far

Figure 2: Illustration of the proposed metrics from three aspects: intensity (Section 3.1), polarity (Section 3.2), and clustering (Section 3.3). These metrics measure the self-consistency of LLMs in generating and ranking supporting $\mathrm { ~ ( ~ ) ~ }$ and defeating $\mathbb { \varpi }$ intermediates with varying intensities. Numbers $\boxed { \cdot 5 }$ , -4 , ..., $\textcircled{+4}$ , $\textcircled{+5}$ indicate the intensity of the generated intermediates, with the lowest value $( \boxed { - 5 }$ ) being the strongest generated defeater and the highest value ( $\textcircled{+5}$ ) the strongest generated supporter.

# 3.1 Intensity: Intensity Ranking Concordance

To assess the concordance between the order from the generation phase and the order from the ranking phase of these fine-grained intermediates, we leverage the Kendall Tau distance (Kendall 1938). This metric quantifies the similarity between two orders by counting the number of pairwise agreements and disagreements. For a sequence $\boldsymbol { \mathcal { T } }$ of LLMgenerated intermediates and its permutation $\mathcal { T } ^ { \prime }$ ranked by the same LLM, a pair of elements from $\boldsymbol { \mathcal { T } }$ is called concordant if they appear in the same order in both $\boldsymbol { \mathcal { T } }$ and $\mathcal { T } ^ { \prime }$ . Conversely, the pair is called discordant if their order is reversed in $\mathcal { T } ^ { \prime }$ compared to $\boldsymbol { \mathcal { T } }$ . The Kendall Tau $\tau$ is calculated as:

$$
\tau = { \frac { ( \# \operatorname { c o n c o r d a n t } \operatorname { p a i r s } ) - ( \# \operatorname { d i s c o r d a n t } \operatorname { p a i r s } ) } { k ( k - 1 ) / 2 } }
$$

where $k$ is the number of elements in the list, and $k ( k -$ $1 ) / 2$ is the total number of pairs. The metric ranges from - $^ { - 1 }$ to 1, where 1 indicates that these two lists are identical; -1 indicates completely reversed rankings; and values close to 0 indicate no association between the two lists. For our task, we have three intensity ranking concordance metrics: $\tau { - } \mathcal { A } , \tau { - } \mathcal { D }$ , and $\tau$ -all, which evaluate the intensity ranking concordance within the supporter group, the defeater group, and the entire sequence of intermediates, respectively.

# 3.2 Polarity: Cross-Group Position (CGP)

To assess the relative positioning of elements between these two polarities–the defeater group $\mathcal { D }$ and the supporter group $\mathcal { A }$ –we propose the Cross-Group Position (CGP) metric. This metric penalizes instances where elements from $\mathcal { A }$ are ranked lower than those from $\mathcal { D } ^ { \mathrm { ~ 2 ~ } }$ . Specifically, CGP is de

fined as:

$$
\begin{array} { r } { \mathrm { C G P } ( \mathcal { T } ^ { \prime } , \mathcal { A } , \mathcal { D } ) = 1 - \frac { \sum _ { a \in \mathcal { A } } \sum _ { d \in \mathcal { D } } \mathbb { 1 } \left[ \mathrm { i n d e x } ( a ) < \mathrm { i n d e x } ( d ) \right] } { | \mathcal { A } | \times | \mathcal { D } | } } \end{array}
$$

where index $( x )$ denotes the index of element $x$ in the ranked sequence $\mathcal { T } ^ { \prime } . \Im [ \cdot ]$ denotes the indicator function that is set to 1 if the condition is true and 0 otherwise. CGP measures how often elements from $\mathcal { A }$ precede the elements of $\mathcal { D }$ in the ranked sequence $\mathcal { T } ^ { \prime }$ . It is normalized to the range [0, 1] by dividing with the maximum possible violations, i.e., $| \mathcal { A } | \times | \mathcal { D } |$ . Higher values indicate better differentiation between groups $\mathcal { A }$ and $\mathcal { D }$ .

# 3.3 Clustering: Intra-Group Clustering (IGC)

In this subsection, we introduce Intra-Group Clustering (IGC), a metric for LLMs’ causal epistemic consistency by assessing the clustering degree of supporting and defeating intermediates. The intuition behind IGC is that all defeaters and all supporters should form cohesive clusters, with a minimal number of polarity changes (from supporting to defeating, or vice versa) when iterating the sequence.

Clustering Distance Based on Polarity Change. Given the LLM-ranked intermediates $\mathcal { T } ^ { \prime }$ , we define $L _ { i }$ to to be a binary polarity that indicates whether $\mathcal { T } ^ { \prime }$ is in $\mathcal { A }$ or $\mathcal { D }$ , while the cardinality $\left| L _ { i } \right|$ refers to the number of intermediates sharing the same polarity as $\mathcal { T } _ { i } ^ { \prime }$ . $d ( i , j )$ is the sequence clustering distance between $\mathcal { T } _ { i } ^ { \prime }$ and $\mathcal { T } _ { j } ^ { \prime }$ , calculated as follows:

$$
d ( i , j ) = \sum _ { k = i } ^ { j - 1 } \mathbb { 1 } [ L _ { k } \neq L _ { k + 1 } \land L _ { k + 1 } \neq L _ { i } ]
$$

where $i < j$ . The distance is based on the number of polarity changes, excluding reversions to the initial polarity.

IGC: A Measure of Clustering Quality in Sequence. With the distance based on polarity change, we use the silhouette score (Rousseeuw 1987; Shahapure and Nicholas

<html><body><table><tr><td>Aspect</td><td colspan="3">Intensity Ranking Concordance</td><td>Cross-Group Position</td><td>Intra-Group Clustering</td></tr><tr><td>T-A↑</td><td colspan="3">T-D↑</td><td>CGP ↑</td><td>IGC ↑</td></tr><tr><td colspan="7">Closed-source LLMs</td></tr><tr><td>GPT-3.5 Turbo</td><td>0.074 ±0.429</td><td>0.045 ± 0.407</td><td>0.304 ± 0.409</td><td>0.750 ± 0.329</td><td>0.762 ± 0.244</td></tr><tr><td>GPT-4</td><td>0.384 ± 0.413</td><td>0.203 ± 0.440</td><td>0.587 ± 0.347</td><td>0.911 ± 0.235</td><td>0.916 ± 0.176</td></tr><tr><td>GPT-4 Turbo</td><td>0.397 ± 0.541</td><td>0.226 ± 0.459</td><td>0.526 ± 0.510</td><td>0.849 ± 0.330</td><td>0.942 ± 0.151</td></tr><tr><td>GPT-4o mini</td><td>0.142 ± 0.444</td><td>0.154 ± 0.418</td><td>0.472 ± 0.375</td><td>0.865 ± 0.281</td><td>0.889 ± 0.196</td></tr><tr><td>GPT-40</td><td>0.317 ± 0.466</td><td>0.229 ± 0.426</td><td>0.637 ± 0.266</td><td>0.964 ± 0.164</td><td>0.978 ± 0.099</td></tr><tr><td>Claude3Haiku</td><td>0.120 ± 0.429</td><td>0.069 ± 0.388</td><td>0.406 ± 0.344</td><td>0.828 ± 0.270</td><td>0.809 ± 0.234</td></tr><tr><td>Claude 3 Sonnet</td><td>0.272 ± 0.429</td><td>0.046 ± 0.423</td><td>0.533 ± 0.290</td><td>0.916 ± 0.204</td><td>0.893 ± 0.195</td></tr><tr><td>Claude 3 Opus</td><td>0.509 ± 0.457</td><td>0.381 ± 0.451</td><td>0.688 ± 0.342</td><td>0.941 ± 0.204</td><td>0.957 ± 0.131</td></tr><tr><td>Claude 3.5 Sonnet</td><td>0.610 ± 0.507</td><td>0.440 ± 0.501</td><td>0.662 ± 0.492</td><td>0.885 ± 0.286</td><td>0.932 ± 0.159</td></tr><tr><td>Gemini 1.5 Flash</td><td>0.108 ± 0.451</td><td>0.115 ± 0.412</td><td>0.429 ± 0.362</td><td>0.842 ± 0.274</td><td>0.838 ± 0.225</td></tr><tr><td>Gemini 1.5 Pro</td><td>0.475 ± 0.435</td><td>0.165 ± 0.463</td><td>0.587 ± 0.326</td><td>0.900 ± 0.212</td><td>0.875 ± 0.205</td></tr><tr><td colspan="6">Open-source LLMs</td></tr><tr><td>Gemma-2B</td><td>-0.021 ± 0.412</td><td>0.001 ± 0.410</td><td>-0.002 ± 0.245</td><td>0.502 ± 0.190</td><td>0.468 ± 0.083</td></tr><tr><td>Gemma-7B</td><td>-0.006 ± 0.392</td><td>0.016 ± 0.389</td><td>0.085 ± 0.256</td><td>0.575 ± 0.203</td><td>0.484 ± 0.122</td></tr><tr><td>LLaMA2-7B</td><td>-0.018 ± 0.406</td><td>0.001 ± 0.412</td><td>-0.029 ± 0.261</td><td>0.477 ± 0.200</td><td>0.475 ± 0.092</td></tr><tr><td>LLaMA2-13B</td><td>-0.000 ± 0.411</td><td>0.026 ± 0.417</td><td>0.072 ± 0.256</td><td>0.560 ± 0.197</td><td>0.480 ± 0.109</td></tr><tr><td>LLaMA2-70B</td><td>0.012 ± 0.409</td><td>0.010 ±0.434</td><td>0.234 ± 0.349</td><td>0.707 ± 0.271</td><td>0.629 ± 0.215</td></tr><tr><td>Phi-3 Mini (3.8B)</td><td>0.135 ± 0.431</td><td>0.012 ± 0.393</td><td>0.300 ± 0.336</td><td>0.740 ± 0.275</td><td>0.659 ± 0.222</td></tr><tr><td>Phi-3-Small (7.4B)</td><td>0.092 ± 0.443</td><td>0.204 ± 0.422</td><td>0.347 ± 0.348</td><td>0.753±0.254</td><td>0.672 ± 0.220</td></tr><tr><td>Phi-3Medium(14B)</td><td>-0.056 ± 0.441</td><td>0.154 ± 0.406</td><td>0.356 ± 0.367</td><td>0.801 ± 0.286</td><td>0.801 ± 0.230</td></tr><tr><td>LLaMA3-8B</td><td>0.030 ± 0.444</td><td>0.139 ± 0.436</td><td>0.273 ± 0.387</td><td>0.712 ± 0.285</td><td>0.639 ± 0.217</td></tr><tr><td>LLaMA3-70B</td><td>0.357 ± 0.469</td><td>0.343 ± 0.419</td><td>0.586 ± 0.415</td><td>0.887 ± 0.274</td><td>0.923 ± 0.177</td></tr><tr><td colspan="6">Random</td></tr><tr><td>Random</td><td>-0.003 ± 0.409</td><td>0.005 ± 0.406</td><td>-0.008 ± 0.249</td><td>0.496 ± 0.192</td><td>0.467 ± 0.077</td></tr></table></body></html>

Table 1: Empirical study of LLMs on the proposed metrics for causal epistemic consistency.

2020) to measure how similar an element is to its own cluster compared to other clusters in sequence:

$$
s ( i ) = \frac { d _ { n c } ( i ) - d _ { i c } ( i ) } { \operatorname* { m a x } ( d _ { i c } ( i ) , d _ { n c } ( i ) ) }
$$

where $d _ { i c } ( i )$ and $d _ { n c } ( i )$ are the intra-cluster distance and nearest cluster distance for each intermediate $\mathcal { T } _ { i } ^ { \prime }$ .

1. The intra-cluster distance $d _ { i c } ( i )$ captures the mean distance between $\mathcal { T } _ { i } ^ { \prime }$ and all other intermediates belonging to the same group, reflecting internal cohesion. It is calculated as:

$$
d _ { i c } ( i ) = \frac { 1 } { \left| L _ { i } \right| - 1 } \sum _ { L _ { j } = L _ { i } , \mathcal { T } _ { j } ^ { \prime } \neq \mathcal { T } _ { i } ^ { \prime } } d ( i , j ) .
$$

2. The nearest cluster distance $d _ { n c } ( i )$ captures the mean distance between $\mathcal { T } _ { i } ^ { \prime }$ and all other points belonging to a different group, demonstrating the level of separation from other clusters. It is calculated as:

$$
d _ { n c } ( i ) = \frac { 1 } { | \mathscr { T } ^ { \prime } | - | L _ { i } | } \sum _ { L _ { j } \neq L _ { i } } d ( i , j ) .
$$

The final Intra-Group Clustering (IGC) metric is computed as the average clustering of all elements:

$$
\mathrm { I G C } = \frac { 1 } { | \varSigma ^ { \prime } | } \sum _ { i = 1 } ^ { | \mathscr { T } ^ { \prime } | } s ( i ) .
$$

Range and Implications of IGC. The range of $s ( i )$ is $[ - 1 , 1 ]$ : (i) Close to 1: The element is near its own group and far from the neighboring groups; (ii) Close to 0: The element is on the border between its cluster and a neighboring cluster. (iii) Close to -1: The element is in the wrong cluster. IGC quantifies the quality of cluster assignments, with a high score indicating well-clustered sequences. It is a general metric applicable to various contexts related to sequence clustering. Further details are in Appendix C.1.

# 4 Causal Epistemic Consistency of LLMs 4.1 Experimental Setup

Foundational Dataset. To ensure the defeasibility of causal pairs, allowing models to generate intermediates with varying polarity and intensity, we utilize the test dataset of $\delta$ -CAUSAL (Cui et al. 2024b) as our foundational dataset, which comprises 1,970 defeasible cause-effect pairs.

![](images/0764542751665c06ba3721e0cde09a0c91f18f90f4b99db38d3a0be489574904.jpg)  
Figure 3: Radar charts comparing performance of various LLM architectures and sizes: Gemma (left), LLaMA2 (middle), and LLaMA3 (right) in maintaining causal epistemic consistency. Each colored line denotes a distinct model size.

Three-Phase Assessment for LLMs’ Causal Epistemic Consistency. There are three main phases in our experiments: (i) Intermediate generation: We provide LLMs with a single cause-effect pair and two preliminary intermediates: one supporting and one defeating. For each supporter and defeater, we instruct the LLMs to generate two weaker and two stronger intermediates. As a result, we compile a total of 10 intermediates as sequence $\boldsymbol { \mathcal { T } }$ , divided into two subsequences: subsequence $\mathcal { D }$ comprised of $m = 5$ intermediates that challenge the cause-effect relationship with differing intensities; and subsequence $\mathcal { A }$ consisting of $n = 5$ supporting intermediates that reinforce the cause-effect pair, also with varying intensities. The prompt for generating these fine-grained intermediates is presented in Figure 6; (ii) Intermediate ranking: From these generated intermediates, we use the same LLM to rank the intermediates to identify their polarities (supporting or defeating) and intensity. The prompt for ranking these fine-grained intermediates is presented in Figure 7; and (iii) Evaluation: Based on the actual order of generated intermediates in the first phase and the predicted ranking order in the second phase, we evaluate the causal epistemic consistency from the perspectives of Intensity Ranking Concordance (ω - , ω - , $\tau$ -all), CrossGroup Position (CGP) agreement, and Intra-Group Clustering (IGC).

Backbone Models. We assess a comprehensive suite of LLMs for causal epistemic consistency. Our evaluation includes: (i) 11 Closed-source models: GPT-3.5 Turbo, GPT4, GPT-4 Turbo, GPT-4o, GPT-4o mini, Claude 3 (Haiku, Sonnet, and Opus), Claude 3.5 (Sonnet) (Anthropic 2024), Gemini 1.5 (Flash and Pro) (Gemini-Team 2024); (ii) 10 Open-source models: Gemma (2B and 7B) (Mesnard et al. 2024), LLaMA2 (7B, 13B, and 70B) (Touvron et al. 2023), Phi-3 (mini, small, and medium) (Abdin et al. 2024), and LLaMA3 (8B and 70B) (Meta 2024).

# 4.2 Experimental Results

Table 1 presents a quantitative comparison of different models on causal epistemic consistency.

• Closed-source models generally outperform opensource models: For instance, GPT-4o achieves a $\tau$ -all score of 0.632, a CGP score of 0.962, and an IGC score of 0.973, whereas LLaMA3-70B, the best-performing open-source model, only achieves a $\tau$ -all score of 0.586, a CGP score of 0.887, and an IGC score of 0.923. • Maintaining consistency in intensity is more challenging than achieving consistency in polarity and clustering: The patterns across different metrics are consistent among different models, suggesting that while LLMs can effectively maintain consistency over differentiating between supporting and defeating intermediates and clustering intermediates of the same polarity together, they find it more challenging to maintain consistent intensity rankings. Namely, achieving consistency over the nuances of causal intensity remains difficult.

# 4.3 Does a Larger Model Scale Mean Better Causal Epistemic Consistency?

Previous works (Kaplan et al. 2020; Hoffmann et al. 2024) have shown that with the increase in model scale, the improvement in performance follows a power-law relationship. However, the effectiveness of ‘just scaling’ for general causal understanding, especially in the context of causality, has become a subject of intense debate (Zecˇevic´ et al. 2023).

Inspired by this question, we investigate whether increasing the model scale improves the causal epistemic consistency of LLMs. Since this model scale study is only possible for models available in multiple sizes, we conduct experiments with: (i) Gemma at sizes of 2B and 7B; (ii) LLaMA2 at sizes of 7B, 13B, and 70B; and (iii) LLaMA3 at sizes of 8B and 70B. The experimental results are presented in Figure 3. From these results, we clearly observe that an increase in model size generally enhances causal epistemic consistency. For instance, LLaMA2 and LLaMA3 demonstrate significant improvements at larger scales, particularly at 70B, where the causal epistemic consistency scores are notably higher compared to their smaller-scale counterparts.

![](images/b524159ffaf40497e4affba5e65fd6aca10156682ec4cdcd08117e2aa0ecf997.jpg)  
Figure 4: Visualization of LLaMA3-70B’s alignment of intermediates’ predicted ranking versus their generation phase ranking, indicating the models’ self-consistency in intensity, polarity, and clustering. Each matrix element $( i , j )$ indicates the percentage of instances where an intermediate ranked at position $i$ during the generation phase was ranked at position $j$ during the ranking phase. For example, ( -3 , $\cdot$ ) indicates the percentage of instances with a label of defeater with an intensity of 3 in the generation phase that was ranked as the supporter with an intensity of 4 during the ranking phase.

# 4.4 Visualization of Causal Epistemic Consistency

We plot the causal epistemic consistency matrices of LLaMA3-70B in Figure 4. In these matrices, the $\mathbf { \boldsymbol { x } }$ -axis from left to right and the y-axis from top to bottom correspond to -5 -4 -3 -2 -1 $\textcircled { + 1 } \textcircled { + 2 } \textcircled { + 3 } \textcircled { + 4 } \textcircled { + 5 }$ , where the square symbol -\* represents defeaters while the circle symbol $\cdot$ represents supporters. The numbers inside the symbols indicate the supporting or defeating intensity, with larger absolute values signifying stronger intensity (i.e., $\boxed { \cdot 5 }$ is the strongest defeater and $\textcircled{+5}$ is the strongest supporter). These matrices visualize how well the models maintain causal epistemic consistency by comparing the labels of intermediates of the generation phase with the predicted labels in the ranking phase.

The confusion matrices of other models are presented in Appendix D. From the results of the best closed-source and open-source models, we have the following observations:

• Diagonal Dominance: Higher values along the diagonal indicate better causal epistemic consistency. This dominance shows that the model often maintains consistency in both polarity and intensity by correctly matching the labels of intermediates from the generation phase to the ranking phase.

• Off-Diagonal Elements: These off-diagonal elements represent the number of instances where the predicted labels during the ranking phase diverge from the labels during the generation phase. Higher values in these cells suggest cases where the model struggles to maintain consistency. For instance, a higher value far from the diagonal indicates a more significant discrepancy between the ranking and generation phases, reflecting lower causal epistemic consistency due to overestimation or underestimation of the intensity of the generated intermediates.

• Cluster Separation: These matrices also indicate that these two models cluster supporting and defeating intermediates well, as shown by lower values in the lower left and upper right corners.

# 5 Beyond Prompting: Leveraging Internal Token Probability

This section explores using internal token probability as an alternative to the prompting method in Section 4 for maintaining causal epistemic consistency.

# 5.1 Internal Token Probability

Internal token probability has proven to be a reliable indicator for sequence correlation estimation (Malinin and Gales 2021; Farquhar et al. 2024; Cui et al. 2024b). For each causeeffect pair $( C , E )$ and any supporting or defeating intermediate $\mathcal { T } _ { j }$ , we utilize the token probabilities $p$ to estimate the causal strength $\mathcal { C } S ( E | C \oplus \mathcal { T } _ { j } )$ in Section 2.1:

$$
\mathcal { C } S ( E | C \oplus \mathcal { Z } _ { j } ) = \prod _ { i } p ( E _ { i } | C \oplus \mathcal { Z } _ { j } , w , E _ { < i } )
$$

where $E _ { i }$ is the $i _ { \mathrm { t h } }$ token of $E$ and $E _ { < i }$ is the first $i - 1$ tokens of $E$ . $p ( E _ { i } | C \oplus \mathbb { Z } _ { j } , w , E _ { < i } )$ is the internal (conditional) token probability. The conjunction word $w$ connects the combination of the cause and the intermediate to the effect, and explicitly indicates the causation such as “because” and “therefore”.

# 5.2 Experimental Setup

Models and Datasets. As closed-source models often do not provide a logprob API usage 3, our investigation resorts to open-source LLMs including Gemma (2B and 7B) (Mesnard et al. 2024), LLaMA2 (7B, 13B, and 70B) (Touvron et al. 2023), Phi-3 (3.8B, 7B, and 14B), and LLaMA3 (8B and 70B). We use the same foundation dataset described in Section 4.1.

Three-Phase Assessment. The experiment in this section involves three phases: (i) Intermediate generation: This phase involves generating a sequence of intermediates, $\boldsymbol { \mathcal { T } }$ , following the same procedure described in Section 4.1; (ii) Intermediate ranking based on conditional token probability: In this phase, we calculate the causal strength based on the conditional token probability using $\{ { \mathcal { C } } { \mathcal { S } } ( { \mathcal { E } } | C \oplus { \mathcal { T } } _ { j } ) | { \mathcal { T } } _ { j } \in { \mathcal { T } } \}$ . (iii) Evaluation: We assess the models’ causal epistemic consistency using rankings from the generation phase and conditional probability values, based on the proposed metrics in Section 3.

![](images/b79c253965df48957afad776509e732f2854cd4a06e95978536b318f7854bb3d.jpg)  
Figure 5: Impact of various conjunction words on the causal epistemic consistency across different LLMs. The x-axes categorize conjunction words into coordinating conjunctions, subordinate conjunctions, and conjunctive adverbs. The y-axes display values for causal epistemic consistency metrics. The analysis encompasses diverse model types (distinguished by marker color and shape) at different scales (represented by line thickness and marker size).

Conjunction Word Choices. We study multiple conjunction words, including (i) coordinating conjunctions (Grammarly 2024): “so”; (ii) subordinate conjunctions (Traffis 2020): “because”, “since”, and “as”; and (iii) conjunctive adverbs (Ellis 2023): “therefore”, “thus”, and “hence”.

# 5.3 Results and Discussion

We analyze the results from two aspects: (i) the impact of conjunction words on models’ causal epistemic consistency; and (ii) the efficacy of internal token probability against the prompting strategy.

Comparison of Different Conjunction Words. We present the impact of different conjunction words on models’ causal epistemic consistency, with distinctions highlighted by varying colors on the $\mathbf { \boldsymbol { x } }$ -axis labels in Figure 5. A consistent trend is observed across different models and causal epistemic consistency metrics. Specifically, coordinating conjunctions (“so”) and conjunctive adverbs (“therefore”, “thus”, “hence”) yield better results, while subordinate conjunctions (“because”, “since”, “as”) underperform. We posit that placing subordinate conjunctions at the beginning of sentences aligns poorly with the natural language patterns seen by LLMs, potentially degrading performance.

We compare the effectiveness of internal token probability versus prompting methods in maintaining causal epistemic consistency, with a detailed comparison across models and metrics in Appendix D.1.

# 6 Related Work

LLMs and Causality. The investigation of LLMs in understanding and generating causal relations has garnered increasing attention. Previous studies often criticize LLMs for their propensity to inaccurately identify and comprehend the complex causal patterns among these facts (Jin et al. 2024; Li et al. 2024; Zecˇevic´ et al. 2023; Cui et al. 2024a). Our study further contributes to this discourse by evaluating LLMs’ self-consistency in reasoning about fine-grained intermediates in causality and by providing metrics and empirical evidence for LLMs’ causal epistemic consistency.

Defeasibility in Causal Reasoning. Our study of finegrained intermediates in causality extends the research initiated by $\delta$ -CAUSAL (Cui et al. 2024b), which introduced the concepts of defeaters and supporters in causal analysis. While $\delta \cdot$ -CAUSAL provided a foundational framework for understanding causal defeasibility, it did not delve into the granularity necessary for nuanced causal reasoning. Our research advances this field by moving beyond the binary classification of intermediates as simply supporting or opposing. We refine the categorization of intermediates by considering both their polarity stance (supporting or opposing) and the intensity of their influence.

Hallucination of LLMs. LLMs suffer from generating nonsensical and fallacious content, known as hallucinations (Huang et al. 2023; Mouchel et al. 2024). The most pertinent hallucination to causal epistemic consistency is the self-contradictory hallucination (Mündler et al. 2024), which means that LLMs generate two contradictory sentences given the same context. Specifically, our study on causal epistemic consistency investigates whether the causal intermediates generated by an LLM at various intensities contradict the ones ranked by the same LLM. However, our study is distinctive in that we focus on the discrepancies between the causal intermediate generation and differentiating behaviors of LLMs, rather than the inconsistencies within the generated text.

# 7 Conclusion

In conclusion, this study introduces causal epistemic consistency as a crucial framework for assessing the selfconsistency of LLMs in distinguishing fine-grained causal intermediates. Supported by a novel suite of evaluation metrics, our comprehensive empirical analysis of 21 LLMs reveals significant limitations in their ability to maintain this consistency. This research addresses a critical gap in the understanding of complex causal reasoning and lays the foundation for the development of more self-consistent models capable of handling intricate causal relationships.