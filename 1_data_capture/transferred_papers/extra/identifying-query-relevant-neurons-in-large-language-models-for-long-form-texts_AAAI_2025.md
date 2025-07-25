# Identifying Query-Relevant Neurons in Large Language Models for Long-Form Texts

Lihu Chen, Adam Dejl, Francesca Toni

Imperial College, London, UK firstname.surname $@$ imperial.ac.uk

# Abstract

Large Language Models (LLMs) possess vast amounts of knowledge within their parameters, prompting research into methods for locating and editing this knowledge. Previous work has largely focused on locating entity-related (often single-token) facts in smaller models. However, several key questions remain unanswered: (1) How can we effectively locate query-relevant neurons in decoder-only LLMs, such as Llama and Mistral? (2) How can we address the challenge of long-form (or free-form) text generation? (3) Are there localized knowledge regions in LLMs? In this study, we introduce Query-Relevant Neuron Cluster Attribution (QRNCA), a novel architecture-agnostic framework capable of identifying query-relevant neurons in LLMs. QRNCA allows for the examination of long-form answers beyond triplet facts by employing the proxy task of multi-choice question answering. To evaluate the effectiveness of our detected neurons, we build two multi-choice QA datasets spanning diverse domains and languages. Empirical evaluations demonstrate that our method outperforms baseline methods significantly. More importantly, analysis of neuron distributions reveals the presence of visible localized regions, particularly within different subject domains. Finally, we show potential applications of our detected neurons in knowledge editing and neuron-based prediction.

# Code — https://github.com/tigerchen52/qrneuron

# 1 Introduction

Large Language Models (LLMs) contain substantial amounts of knowledge within their neurons (or parameters). Recent research has focused on identifying and localizing these knowledge neurons to gain insights into the information processing mechanisms of LLMs. Activation-based methods (Voita, Ferrando, and Nalmpantis 2023) examine activation patterns to elucidate the role of neurons in the reasoning process. However, these methods often struggle to directly attribute specific outputs to corresponding inputs, thereby limiting their effectiveness in accurately identifying relevant knowledge. Gradient-based methods (Dai et al. 2022) measure the sensitivity of model outputs to internal components in response to specific inputs, which enables the effective identification of neurons relevant to particular queries. However, these methods typically employ fillin-the-blank tasks, such as “Paris is the capital of ”, to localise components representing triplet facts. Causality-based methods (Meng et al. 2022a) take a different approach by employing causal mediation analysis to pinpoint layers within LLMs that store factual associations. Another branch of pioneering research attempts to locate functional regions in small-size language models such as BERT (Kenton and Toutanova 2019) and GPT-small (Radford et al. 2019), including linguistic regions (Zhang et al. 2024b), factual subnetworks (Ren and Zhu 2022; Bayazit et al. 2023), and modular structures (Zhang et al. 2023; Conmy et al. 2023).

Table 1: Comparison of general-domain knowledge locating methods. Here, we do not consider task-specific approaches like Language Neuron (Chen et al. 2024b) and Privacy Neuron (Wu et al. 2023).   

<html><body><table><tr><td>Methods</td><td>Long-Form Texts</td><td>Neuron-Level Location</td><td>Decoder Models</td><td>≥7B LLMs</td></tr><tr><td>Knowledge Neuron (2022)</td><td>×</td><td>√</td><td>X</td><td>X</td></tr><tr><td>ROME (2022a)</td><td>×</td><td>×</td><td>√</td><td>X</td></tr><tr><td>Knowledge Subnetwork (2023)</td><td>×</td><td>√</td><td>√</td><td>×</td></tr><tr><td>QRNCA (Ours)</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table></body></html>

While these studies successfully identify knowledge associations stored within LLMs, three significant questions remain underexplored: (1) How can we effectively locate query-relevant neurons in contemporary decoder-only LLMs, such as Llama (Touvron et al. 2023) and Mistral (Jiang et al. 2023), given their large model size and different architectures? (2) How can we address the challenge of long-form text generation, as previous methods have been limited to triplet facts? (3) Are there localized knowledge regions in LLMs analogous to the localized functional regions observed in human brains (Brett, Johnsrude, and Owen 2002)?

To address the first two questions, we introduce a novel framework named Query-Relevant Neuron Cluster Attribution (QRNCA) designed to identify query-relevant neurons in LLMs. The principal advantages of our framework are its architecture-agnostic nature and its capability of handling

Chemistry Oooo Oo0o □o0o Physics QR Cluster   
The energy giBvieonlougpyby electrons... Oooo ：00 ? QR Neuron   
A. make glucose   
B. make NADH 0000 ●   
C. produce ATP O   
D. break down glucose Common Neuron Multi-Choice QA FFNs in LLMs Coarse Neurons QR Neurons

long-form text generation effectively, as shown in Table 1. QRNCA aims to extract Query-Relevant (QR) neurons for each query-answer fact. The process begins by transforming a free-form generation task into a multiple-choice questionanswering format. By employing prompt engineering, we constrain LLMs to generate only the option letter rather than the complete answer. This approach allows for the examination of long-form generation beyond single tokens. Subsequently, we adapt the Knowledge Attribution method (Dai et al. 2022) to compute Neuron Attribution, which elucidates the relationship between neurons and the factual knowledge. We then gather clusters for a series of queries and calculate the Inverse Cluster Attribution. This step mitigates the influence of neurons that recur across clusters (or queries). The final step involves multiplying the neuron attribution and inverse cluster attribution values to pinpoint correlated neurons. Additionally, we identify certain Common Neurons that are associated with common words, punctuation marks, and option letters. Excluding these common neurons enhances the detection of QR neurons. Empirical evaluations demonstrate that our proposed method outperforms baseline approaches.

To investigate the existence of localized knowledge regions, we construct two multi-choice QA datasets encompassing various domains and languages. Then, we visualize the geographical locations of the detected neurons in Llama. Our findings indicate that distinct localized regions emerge in the middle layers, particularly for domain-specific neurons. This suggests that LLMs tend to complete the formation of domain-specific concepts within these middle layers. Conversely, language-specific neurons are more sparsely distributed, indicating that LLMs likely draw on linguistic knowledge at all processing levels. Additionally, we observed that common neurons are concentrated in the top layer, predominantly expressing frequently used tokens.

In summary, our main contribution is four-fold: (1) A scalable method: we propose QRNCA to detect query-relevant neurons in LLMs; the QRNCA method is architecture-agnostic and can deal with long-form generations. (2) Two new datasets: we curate two multi-choice QA datasets that contain different types of knowledge, namely Domain Knowledge and Language knowledge. (3) In-depth studies: we study the knowledge distribution within LLMs and we are the first to show that there are visible localized regions in Llama. (4) Potential applications: we show that QRNCA might be useful for knowledge editing and neuron-based prediction.

# 2 Related Work

# 2.1 Locating Knowledge in LLMs

Large Language Models contain a vast range of knowledge within their parameters, spanning factual (Petroni et al. 2019; Zhou et al. 2020; Jiang et al. 2020; Roberts, Raffel, and Shazeer 2020; Pezeshkpour 2023), linguistic (Liu et al. 2019; Jawahar, Sagot, and Seddah 2019; Chen, Varoquaux, and Suchanek 2023), and domain-specific information (Sung et al. 2021; Frieder et al. 2024). Recent mechanistic studies suggest that knowledge is primarily stored in the FFN (Feed-forward Network) layers of Transformers (Geva et al. 2021, 2022), which prompts ongoing research efforts aimed at developing methods to identify and locate this knowledge within these layers. Activation-based methods (Voita, Ferrando, and Nalmpantis 2023; Gurnee et al. 2024) investigate the activation patterns of neurons to interpret how the network processes information at different stages. However, a key limitation of these methods is their inability to directly attribute the model’s output to specific inputs, which limits their precision in identifying relevant knowledge. Gradient-based methods (Ancona et al. 2019; Dai et al. 2022), on the other hand, offer fine-grained attribution by quantifying the sensitivity of model outputs to internal components in response to a given input. This approach effectively identifies neurons relevant to specific queries. Nonetheless, current gradient-based techniques have primarily focused on single-token triplet facts. Another approach, Causality-based methods, employs causal mediation analysis to discern the particular layers associated with a given factual input (Meng et al. 2022a). This line of research has evolved into a locate-and-edit paradigm, aimed at refining knowledge within LLMs (Meng et al. 2022b; Ju and Zhang 2023; Zhang et al. 2024a). In addition to general knowledge locating approaches, recent studies have focused on identifying neurons responsible for specific tasks, such as linguistic (Chen et al. 2024b; Tang et al. 2024; Kojima et al. 2024), privacy-related (Wu et al. 2023; Chen et al. 2024a) and biasrelated neurons (Yang, Kang, and Jung 2023).

In this work, we propose a novel gradient-based attribution method aimed at locating input-output knowledge within LLMs. Unlike previous methodologies, our approach mainly focuses on long-form (or free-form) texts beyond entity facts.

# 2.2 Analyzing Knowledge Distribution in LLMs

Given the human-like reasoning capabilities observed in LLMs across various tasks (Zhao et al. 2023), and since our brain contains functional locations associated with distinct cognitive processes (Brett, Johnsrude, and Owen 2002; Bjaalie 2002; Gholipour et al. 2007), we ask whether there are similar regions in LLMs. Previous investigations have explored the behaviors of individual neurons indicating that a neuron can encode multiple concepts (Bolukbasi et al. 2021) while a concept can also be distributed across multiple neurons (Dalvi et al. 2019; Durrani et al. 2020; Chen et al. 2024b). Subsequent endeavors have sought to identify functional regions in LLMs, encompassing linguistic regions (Zhang et al. 2024b), factual subnetworks (Ren and Zhu 2022; Bayazit et al. 2023), and modular structures (Zhang et al. 2023; Conmy et al. 2023). These studies have investigated localized behaviors in smaller-scale language models, such as BERT and GPT-small. Building upon these foundations, our research embarks on the examination of knowledge locations in larger-size LLMs, specifically those with 7B parameters, spanning multiple knowledge domains.

# 3 Background

Feed-forward Networks in LLMs Feed-forward networks (FFNs) are widely used by transformer-based language models. Geva et al. (2021) reveal that FFNs emulate key-value memories and their outputs are responsible for refining the final output distribution over the vocabulary. Although traditional two-layer FFNs in BERT (Kenton and Toutanova 2019) and GPT-2 (Radford et al. 2019) have been studied well, the behaviors of FFNs in modern LLMs such as Llama (Touvron et al. 2023) and Mistral (Jiang et al. 2023), are not well-explored. These LLMs adopt Gated Linear Units (GLUs) (Dauphin et al. 2017) in their FFNs, which can be formulated as follows:

$$
\mathrm { F F N } ( \mathbf { X } ) = ( \mathbf { X } \mathbf { W } ^ { U } \odot \mathrm { S i L U } ( \mathbf { X } \mathbf { W } ^ { G } ) ) \ \mathbf { W } ^ { D }
$$

Here, $\mathbf { X } \in \mathbb { R } ^ { n \times d }$ is the input sequence, $n$ is the number of tokens and $d$ is the dimension of input vectors; $\mathbf { W } ^ { U } \in \mathbb { R } ^ { d \times m }$ , $\mathbf { W } ^ { G } \in \mathbb { R } ^ { d \times m }$ , $\mathbf { W } ^ { D } \in \mathbb { R } ^ { m \times d }$ are parameter matrices, $m$ is the hidden dimension of the FFN and $\odot$ is the Hadamard product; finally SiLU (Elfwing, Uchibe, and Doya 2018) is the activation function.

Knowledge Neurons Dai et al. (2022) propose a gradientbased Knowledge Attribution to identify the knowledge neurons in BERT by using the fill-in-the-blank cloze task. Their method evaluates the contribution of each neuron in FFNs to the knowledge predictions. Given a query prompt $q$ (“Paris is the capital of ”), the probability of the correct answer predicted by an LLM can be formulated as:

$$
P _ { q } ( \hat { w } _ { i } ^ { l } ) = p ( y ^ { * } | q , w _ { i } ^ { l } = \hat { w } _ { i } ^ { l } )
$$

where $y ^ { \ast }$ is the correct answer (France); $w _ { i } ^ { l }$ denotes the $i$ -th intermediate neuron in the $l$ -th layer in FFNs; $\hat { w } _ { i } ^ { l }$ is a constant we assign to $\boldsymbol { w _ { i } ^ { l } }$ .

In order to measure the attribution score (or contribution) of a neuron, they gradually change the $w _ { i } ^ { l }$ from 0 to its original value computed during the forward pass and integrate the gradients (Sundararajan, Taly, and Yan 2017):

$$
\mathrm { A t t r } ( w _ { i } ^ { l } ) = \bar { w } _ { i } ^ { l } \int _ { \alpha = 0 } ^ { 1 } \frac { \partial P _ { q } ( \alpha \bar { w } _ { i } ^ { l } ) } { \partial w _ { i } ^ { l } } \mathrm { d } \alpha
$$

where ∂Pq(αlw¯il) is the gradient with regard to wil. Attr(·) accumulates the output probability change as $\alpha$ gradually varies from 0 to 1. The attribution measures the contribution of the neuron $w _ { i } ^ { l }$ to the correct answer. In practice, the score is estimated by using Riemann Approximation:

$$
\hat { \mathrm { A t t r } } ( w _ { i } ^ { l } ) = \frac { \bar { w } _ { i } ^ { l } } { m } { \sum _ { k = 1 } ^ { m } } \frac { \partial P _ { q } ( \frac { k } { m } \bar { w } _ { i } ^ { l } ) } { \partial w _ { i } ^ { l } }
$$

where $m$ is the number of the estimation steps. Finally, they identify a coarse set of knowledge neurons whose attribution scores are greater than a threshold $t$ . The localized neurons are supposed to be highly associated with a piece of knowledge, i.e., the query-answer facts.

# 4 Locating Query-Relevant (QR) Neurons in Decoder-only LLMs

While Knowledge Attribution (Dai et al. 2022) effectively identifies neurons linked to factual queries, its applicability is limited to encoder-only architectures, and it mandates the output to be a single-token word. To address these constraints, we propose a new framework named QueryRelevant Neuron Cluster Attribution (QRNCA). The framework is architecture-agnostic and capable of handling longform generation.

To clarify the main concepts in our framework, we provide the following key notions: QR Neuron is an individual neuron highly correlated with a specific factual knowledge, capable of influencing the corresponding knowledge expression. QR Cluster represents a coarse grouping of neurons associated with a specific fact. This cluster may include noisy neurons and require further refinement. Common Neuron is consistently activated by a wide range of inputs, representing general knowledge or concepts.

The overall framework is shown in Figure 1. Our framework first resorts to the proxy task of Multi-Choice $Q A$ to deal with long-form texts. Starting with a given input, the framework employs Neuron Attribution to derive a QR Cluster. Each neuron within this cluster is assigned an attribution score that indicates its relevance to the query. Next, we apply Inverse Cluster Attribution to attenuate the influence of neurons that frequently occur across multiple clusters. Finally, we identify Common Neurons, as those lacking discriminative power in determining query relevance and representing common knowledge or concepts. Refining the extraction of QR neurons by excluding these common neurons enhances the precision in identifying critical neural correlates.

In the following paragraphs, we introduce the details of these key components in our framework: Multi-Choice QA Transformation, Neuron Attribution, Inverse Cluster Attribution, and Common Neurons.

# 4.1 Multi-Choice QA Transformation

Multi-choice QA is widely used in a variety of real-world educational assessments and standardized tests. Meanwhile, many known benchmarks such as MMLU (Hendrycks et al. 2020) and Big-bench (Srivastava et al. 2023) use multichoice QA to evaluate the breadth and depth of a model’s knowledge. Therefore, we adopt the proxy task of multichoice QA to study the knowledge associations in LLMs. To deal with free-form answers, we advocate for the transformation of questions and their corresponding answers into a multiple-choice framework, as illustrated in Figure 1. This approach involves the generation of distracted options by randomly sampling answers within the same domain. Following this, the LLM is prompted to produce only the option letter. Subsequently, we investigate the neurons correlated with the input. To mitigate the impact of randomness, we devise multiple prompt templates and systematically shuffle the order of options to prevent the model from learning spurious correlations based on option letters. These prompt templates are detailed in Table A2 in the Supplementary Material in the extended version of this paper1 (SM in short in the remainder of this paper).

# 4.2 Neuron Attribution

To extend our methodology to Gated Linear Units (GLUs), which comprise two linear transformations followed by a gating mechanism, we adapt the Knowledge Attribution approach (Eq 5). In GLUs, the linear transformations involve computing a linear combination of input features, denoted by $\dot { f } = \mathbf { \bar { X } } \mathbf { W } ^ { U }$ . Additionally, the gating mechanism, represented by $g = \dot { \mathrm { \bf ~ S i L U } } ( \mathrm { \bf X } \dot { \bf W } ^ { G } )$ , determines the extent to which each input component should be forwarded, thereby enabling the model to emphasize important features while suppressing irrelevant ones. To compute the relevant attribution, we can use either $\frac { \partial P _ { q } } { \partial f }$ or ∂P and we choose to use the former since our empirical study shows it can obtain better QR neurons (see details in Figure A4 in the SM). Given a query $q$ , instantiation using our templates yields a query set $\mathcal { Q } = \{ q _ { 1 } , q _ { 2 } , . . . , q _ { | \mathcal { Q } | } \}$ , and the attribution score of the neuron $n _ { i } ^ { l }$ can be denoted as:

$$
\mathrm { { n a } } ( n _ { i } ^ { l } ) = \frac { \sum _ { j = 1 } ^ { | \mathcal { Q } | } \frac { \bar { f } _ { i } ^ { l } } { m } \sum _ { k = 1 } ^ { m } \frac { \partial P _ { q _ { j } } ( \frac { k } { m } \bar { f } _ { i } ^ { l } ) } { \partial f _ { i } ^ { l } } } { Z }
$$

Here, the numerator means that we sum up the scores of different instantiated templates together as the initial attribution score. The denominator $Z$ is the normalization factor obtained by summing the initial attribution scores of all neurons. Since the number of prompts for each query may vary and the initial attribution scores may be scaled differently, we use normalization to make the attribution scores comparable across queries.

# 4.3 Inverse Cluster Attribution

With the attribution score, we can obtain a list of coarse clusters for each query $\mathcal { C } = \{ c _ { 1 } , c _ { 2 } , . . . , c _ { | \mathcal { C } | } ) \}$ , where $c$ is a cluster that consists of neurons whose attribution score is higher than some threshold $t$ . The frequent appearance of some neurons across queries of different fields reveals that they are not critical neurons to the input query. To decrease their impact, we calculate the inverse cluster attribution:

$$
\mathrm { i c a } ( n _ { i } ^ { l } ) = \log { \frac { | { \mathcal { C } } | } { | \{ c : c \in { \mathcal { C } } { \mathrm { ~ a n d ~ } } n _ { i } ^ { l } \in c \} | + 1 } }
$$

# 4.4 Common Neurons

We observe that some neurons with a relatively high attribution score are still shared across clusters. Through case studies (as shown in Table 5), we demonstrate that they express commonly used concepts such as option letters (“A” and “B”) or stop words (“and” and “the”). Therefore, we count the frequency of each neuron across clusters. If the frequency is higher than the $u \%$ of total clusters, we assign the given neuron into the common neuron set.

# 4.5 Obtaining QR Neurons

Given a query, the final score of a neuron is given by:

$$
\mathrm { n a i c a } ( n _ { i } ^ { l } ) = \mathtt { n a } ( n _ { i } ^ { l } ) \times \mathrm { i c a } ( n _ { i } ^ { l } )
$$

We select top- $v$ neurons with the highest score from the detected cluster and further remove common neurons to refine the QR neuron set.

# 5 Analyzing Detected QR Neurons 5.1 Experimental Settings

Dataset Construction We construct two datasets to locate knowledge neurons that cover two different categories: subject domains and languages.

Domain Dataset is derived from MMLU (Hendrycks et al. 2020), a multi-choice QA benchmark designed to evaluate models across a wide array of subjects with varying difficulty levels. The subjects encompass traditional disciplines such as mathematics and history, as well as specialized fields like law and ethics. In our study, we select six high school exam subjects from the test set: Biology, Physics, Chemistry, Mathematics, Computer Science, and Geography.

Language Dataset is adapted from Multilingual LAMA (Kassner, Dufter, and Schu¨tze 2021), which is a dataset to investigate knowledge in language models in a multilingual setting covering 53 languages. We select six languages: Arabic, English, French, Japanese, Russian and Chinese. Each language subset includes queries that cover five different relations: birth place, employer, instrument, headquarters location, and host country.

Table 2: Statistics of our constructed datasets.   

<html><body><table><tr><td>Domain</td><td>Bio</td><td>Phys</td><td>Chem</td><td>Math CS</td><td>Geo</td><td>Total</td></tr><tr><td>Num</td><td>100</td><td>100</td><td>100</td><td>100 52</td><td>100</td><td>552</td></tr><tr><td>Language</td><td>Ar</td><td>En</td><td>Fr</td><td>Ja Ru</td><td>Zh</td><td>Total</td></tr><tr><td>Num</td><td>100</td><td>100</td><td>100</td><td>100 100</td><td>100</td><td>600</td></tr></table></body></html>

Table 3: Average number of detected QR neurons per query.   

<html><body><table><tr><td>Domain</td><td>Bio</td><td>Phys</td><td>Chem</td><td>Math</td><td>CS</td><td>Geo</td><td>Total</td></tr><tr><td>Num</td><td>13.1</td><td>13.3</td><td>12.8</td><td>11.1</td><td>14.3</td><td>12.7</td><td>12.9</td></tr><tr><td>Language</td><td>Ar</td><td>En</td><td>Fr</td><td>Ja</td><td>Ru</td><td>Zh</td><td>Total</td></tr><tr><td>Num</td><td>12.4</td><td>14.4</td><td>12.7</td><td>16.6</td><td>15.8</td><td>15.0</td><td>15.2</td></tr></table></body></html>

The statistics of our datasets are shown in Table 2 and examples can be found in Table A3 in the SM.

Metric We modify the values of neurons to observe their impact on knowledge expression. For each query, we record the percentage change in the probability of the correct answer, thereby assessing the extent to which the QR neurons influence the predictions of LLMs. We compare our approach to other baseline methods and include a control group with an equal size to determine whether the same detected neurons affect the predictions of randomly selected queries from unrelated fields (Unrelated). The Probability $\frac { \left| \mathrm { R e l a t e d } \right| } { \left| \mathrm { U n r e l a t e d } \right| }$ where Related and Unrelated mean the average probability change of the related and unrelated samples, respectively. We hope that detected neurons can affect the knowledge expressions of the corresponding facts (related) while exerting a low impact on unrelated facts. A higher value of PCR shows detected neurons can have a higher influence on the query, indicating better neurons (Chen et al. 2024b).

Baselines We compare QRNCA to other neuron-level baselines2: Random Neuron are randomly selected from FFNs, making sure they have the same number of neurons as QRNCA; Activation selects neurons with high activated values. Kowledge Neuron∗ is adapted from knowledge attribution (Dai et al. 2022) by using the multi-choice QA task; QRNCA wo/ ICA only uses neuron attribution (Eq 5) to obtain relevant neurons, which dose not involve the computation of Inverse Cluster Attribution; QRNCA w/ Common Neuron is a variant without removing common neurons.

Implementations We mainly study the knowledge neurons in Llama-2-7B (Touvron et al. 2023) and we use the instruction-tuned version so that the model is more responsive to our prompts. Llama-2-7B consists of 32 layers with the FFN hidden dimension of 11008. Besides, we also conduct experiments for Mistral-7B (Jiang et al. 2023) to validate whether our method can obtain consistent findings over different models. Note that our framework can be easily extended to larger-size LLMs.

Table 4: Comparisons of different knowledge locating methods for Llama-2-7B. The metric here is the Probability Change Ratio (PCR) described in Section 5.1. Details are shown in Table A2 in the SM.   

<html><body><table><tr><td></td><td colspan="2">Domain 一</td></tr><tr><td>Method</td><td>介Boost 介 Suppress</td><td>介Boost 介 Suppress</td></tr><tr><td>Random Neuron</td><td>1.0 0.55</td><td>2.0 1.0</td></tr><tr><td>Activation</td><td>1.0 1.0</td><td>1.1 1.1</td></tr><tr><td>Knowledge Neuron*</td><td>1.0 1.0</td><td>6.7 1.8</td></tr><tr><td>QRNCA wo/ ICA</td><td>2.5 1.1</td><td>6.5 2.2</td></tr><tr><td>QRNCAw/ Common Neuron</td><td>2.8 1.8</td><td>10.4 8.5</td></tr><tr><td>QRNCA</td><td>4.4 5.6</td><td>41.2 36.0</td></tr></table></body></html>

As for the hyper-parameters, the number of estimation steps was set to $m = 1 6$ and the attribution threshold $t$ to 0.2 times the maximum attribution score. The template number was $| \mathcal { Q } | = 3$ , the frequency $u$ for obtaining common neurons was $30 \%$ , and the top- $v$ for select coarse neurons was 20. We ran all experiments on three NVIDIA-V100. It took 120 seconds on average to locate neurons for a query with three prompt templates. For each domain and language, the average number of detected QR neurons is between 12 and 17 (as shown in see Table 3). Hyper-parameters are selected based on a hold-out set of biology queries with 50 samples.

# 5.2 Statistics of Detected QR Neurons

In this section, we are curious about the distribution of different knowledge storage in neurons: Do different categories of knowledge share neurons? To this end, we study the overlap rate. First, we aggregate detected neurons of all queries in a domain or language. Next, the rate is obtained by counting the number of shared neurons between different domains or languages. Figure 2a illustrates the overlap rates among different domains and languages. We observe that interdisciplinary or interconnected languages share a higher overlap rate such as (geography, biology) and (Chinese, Japanese), which is in line with our intuition. A surprising finding is that domains have higher overlap rates than languages, which indicates that LLMs tend to allow the storage of multiple domain-specific concepts in a single neuron (polysemantic). Although language-specific neurons are not monosemantic (Chen et al. 2024b), they prefer to encode one specific language concepts, which is also consistent with recent findings (Tang et al. 2024).

Regarding layer distribution, the QR neurons are predominantly located in the middle layers (15-18) and the top layers (around 30), as depicted in Figure 2b. This finding indicates knowledge concepts are mainly stored in the middle and top layers, and we may only modify these neurons for efficient knowledge updating (Ding et al. 2023).

Domain Language 400 bio 1.00 0.41 0.47 0.37 0.37 0.49 en 1.00 0.08 0.06 0.02 0.05 0.05 350 Domain   
phys 0.41 1.00 0.46 0.41 0.39 0.39 fr 0.08 1.00 0.06 0.04 0.07 0.13 230500 Language   
chem math 0.47 0.37 0.46 0.41 1.00 0.43 0.43 1.00 0.40 0.36 0.43 0.35 zh ja 0.06 0.02 0.06 0.04 1.00 0.12 0.12 1.00 0.04 0.04 0.06 0.04 1050 cs 0.37 0.39 0.40 0.36 1.00 0.37 ar 0.05 0.07 0.04 0.04 1.00 0.07   
geo 0.49 0.39 0.43 0.35 0.37 1.00 ru 0.05 0.13 0.06 0.04 0.07 1.00 0 bio phys chem math cS geo en f zh ja ar ru 0 5 10 15 20 25 30 Layer (a) Overlap Rate (b) Layer Distribution

Related 4 Related 0.02 m Bio Phys Chem Math CS Geo 0123Probability Percentage Change 123 AR EN FR JA RU ZH (a) Domains (b) Languages

# 5.3 QR Neurons Can Impact the Knowledge Expression

To validate the impact of our identified QR neurons, we replicate the experiments by Dai et al. (2022), updating the values of QR neurons using two methods: given a query and the value of ${ \bar { f } } _ { i } ^ { l }$ , we either (1) boost the neurons by doubling the value $f _ { i } ^ { l } = 2 \times \bar { f } _ { i } ^ { l }$ ; or (2) suppress the neuron by making $f _ { i } ^ { l } = 0$ . After one operation, we record the PCR on a specific dataset to show the quality of these neurons.

Table 4 presents the overall performance of various methods. Our QRNCA method consistently outperforms other baselines, evidenced by its higher PCR. This indicates that our identified QR neurons significantly affect the probability of correct answers while exerting a relatively low impact on unrelated queries. For instance, our method achieves a boosting ratio of 41.2 on the language dataset, the highest among the baselines. Additionally, both our proposed ICA and the removal of common neurons provide further benefits in locating neurons, as evidenced by the worse performance of the two QRNCA variants.

Furthermore, Figure 3 and Figure 4 illustrate the percentage change in probability for each subject domain and language after modifying neuron values, which can clearly demonstrate the effectiveness of our detected neurons. Additionally, we performed experiments on Mistral-7B. The results, presented in Figure A3 in the SM, consistently support our conclusions.

# 5.4 Are There Localized Regions in LLMs?

Given our ability to identify QR neurons for each query, it is intriguing to explore whether LLMs exhibit localized regions for each domain or language, analogous to the functional localizations in the human brain (Brett, Johnsrude, and Owen 2002). To investigate this, we visualize domainor language-specific neurons on a 2D geographical heatmap. The width of the heatmap corresponds to the dimension of FFNs in Llama-2-7B (11008), and the length represents the layer depth (32). We accumulate the value of naica $( n _ { i } ^ { l } )$ to populate the heatmap. Figure 5 displays the geographical locations of QR neurons in Llama-2-7B across various academic domains and languages. The distribution of QR neurons appears sparse but with distinct regions, particularly for different domains. Notably, certain regions are visible in the middle layers (10-15), suggesting specific neuron patterns. In contrast, language neurons are more sparsely distributed with smaller regions, and languages like Arabic and Russian exhibit less localized properties.

0.10 TTI Probability Percentage Change 0.0 TD 0.2 0.4   
0.3 0.6   
Bio Phys Chem Math CS Geo AR EN FR JA RU ZH (a) Domains (b) Languages

![](images/107777e2ae2d708e998cbb7e6e9848642d2dba0c99d16806d9f69fa0d62c827f.jpg)  
Figure 4: The correct probability percentage change by suppressing QR neurons. The LLM here is Llama-2-7B (Touvron et a 2023)   
Figure 5: Geographical heatmap of detected QR neurons for different domains and languages. The value is calculated by our naica $( n _ { i } ^ { l } )$ . Brighter colors indicate higher naica values. The LLM here is Llama-2-7B $( 1 1 0 0 8 \times 3 2 )$ )

Based on prior studies, LLMs process and represent information in a hierarchical manner (Geva et al. 2022; Wendler et al. 2024; Tang et al. 2024). The early layers are primarily responsible for extracting low-level features , while the middle layers begin to integrate this information, forming more complex semantic representations. The late layers are typically dedicated to generating the final output. Therefore, we suppose that domain-specific knowledge representation is built in the middle layer and the top layers are then mainly responsible for next-token prediction, which may explain the visible regions for different subject domains. Regarding language-specific neurons, their role in accessing linguistic knowledge across different layers likely accounts for their more sparse and distributed locations. This distribution reflects the necessity of engaging with language-specific neurons at multiple stages of information processing.

# 5.5 The Function of Common Neurons

To gain insights into the function of common neurons, we project the matrix $\mathbf { W } ^ { D }$ in Equation 1 to the vocabulary space and select the top-k tokens with the highest probability. Table 5 lists the predicted tokens, which include common words, punctuation marks, and option letters. These findings reinforce the notion that common neurons are not critical for specific queries. We also visualize their locations within Llama-2-7B and we observe that they tend to appear at the top layer (as shown in Figure A2 in the SM).

Table 5: Tokens predicted by the common neurons.   

<html><body><table><tr><td>Neuron</td><td>Top-k tokens</td></tr><tr><td></td><td>_in，_and，_to，_for，_today，_at，_as</td></tr><tr><td>n10676</td><td>_July，-June，_March，_April，_November</td></tr><tr><td></td><td>-respectively, ~while,_and</td></tr><tr><td></td><td></td></tr><tr><td>n778</td><td>_C，C，-c，c，'_ced'</td></tr><tr><td>31 n7670</td><td>_B，B，_Bill, _Bh，'_Bureau'</td></tr></table></body></html>

We also analyzed the token predicted by QR neurons, but we found that middle-layer neurons do not have a clear semantic meaning and human-readable concepts mostly appear in the top layer (Wendler et al. 2024). In Section A in the SM we conduct semantic meaning analyses of neurons.

Table 6: Successful rates of knowledge editing. $\Delta$ measures how well we can flip the predictions $( c o r r e c t $ incorrect or vice versa).   

<html><body><table><tr><td></td><td colspan="2">Domain</td><td colspan="2">Language</td></tr><tr><td>Method</td><td>Boost △ (%)</td><td>Suppress △(%)</td><td>Boost △(%)</td><td>Suppress △(%)</td></tr><tr><td>Random Neuron</td><td>0.0</td><td>0.3</td><td>0.2</td><td>0.3</td></tr><tr><td>Activation</td><td>0.0</td><td>0.1</td><td>0.0</td><td>0.3</td></tr><tr><td>Knowledge Neuron*</td><td>1.4</td><td>3.8</td><td>14.3</td><td>16.0</td></tr><tr><td>QRNCA</td><td>12.6</td><td>18.2</td><td>16.6</td><td>24.8</td></tr></table></body></html>

<html><body><table><tr><td>Method</td><td>Biology</td><td>Chemistry</td><td>Geography</td></tr><tr><td>Random guess</td><td>0.25</td><td>0.25</td><td>0.25</td></tr><tr><td>Prompt-based model pred.</td><td>0.96</td><td>0.71</td><td>0.89</td></tr><tr><td>Neuron-based pred.</td><td>0.96</td><td>0.67</td><td>0.89</td></tr></table></body></html>

Table 7: Accuracy of neuron-based prediction on selected domains in comparison with the standard prompt-based model prediction.

shown in Table A2 in the SM). This suggests that the activity of identified neurons can reflect the model’s reasoning process to some extent. Investigating how this finding could be leveraged in applications like fact-checking and hallucination detection presents a promising line of future work.

# 7 Conclusion

In this study, we introduce a novel framework, QRNCA, for identifying neurons in LLMs for long-form answers, extending beyond triplet facts. To validate our approach, we curate two datasets encompassing diverse domains and languages. Our experimental results show that our method outperforms existing baselines in identifying associated neurons. Additionally, this study pioneers the exploration of localized knowledge regions in LLMs and demonstrates Llama contains knowledge-specific regions in the middle layers while language-specific neurons tend to be distributed across different layers. Further, we prototype two potential usages of identified neurons in applications such as knowledge editing and neuron-based prediction. We hope that our findings are beneficial for further research in understanding the knowledge mechanisms underlying LLMs.

# 6 Potential Applications

We provide two usage examples to showcase the potential applications of our detected QR neurons: Knowledge Editing and Neuron-Based Prediction.

# 6.1 Knowledge Editing

Apart from using the metric of PCR in Section 5.3, we are also interested in whether the detected QR neurons can be used for knowledge editing. For this goal, we adjust the values of QR neurons by either boosting or suppressing them to determine if we can change the prediction of a query from incorrect to correct or vice versa. Table 6 presents the success rates of knowledge editing on our constructed language datasets. Our observations indicate that QRNCA achieves higher success rates than other baselines.

# 6.2 Neuron-Based Prediction

The intuition behind neuron-based prediction is that for a domain-specific question, if the corresponding localized regions are properly activated, the LLM is more likely to generate truthful answers. Otherwise, the LLM may produce hallucinated answers. To this end, we test whether the correct answers to domain-specific questions can be predicted solely based on the activity of the associated neurons. Since we harvest QR neurons for queries in different subject domains, we can group all neurons for a domain to obtain a set of domain-specific neurons. We experiment on a specifically constructed MMLU (Hendrycks et al. 2020) validation set with a different set of questions than those used to determine the QR neurons (see Section B in the SM for details on our experimental strategy). The results are summarised in Table 7. We observe that the accuracy of the neuron-based predictions is very close to the accuracy of the prompt-based method of using the entire model (the used templates are