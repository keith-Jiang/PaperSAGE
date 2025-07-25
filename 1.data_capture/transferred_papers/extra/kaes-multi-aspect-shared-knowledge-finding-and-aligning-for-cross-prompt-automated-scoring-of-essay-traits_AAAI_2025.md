# KAES: Multi-aspect Shared Knowledge Finding and Aligning for Cross-prompt Automated Scoring of Essay Traits

Xia Li1,2, Wenjing Pan1

1School of Information Science and Technology, Guangdong University of Foreign Studies, Guangzhou, China 2Center for Linguistics and Applied Linguistics, Guangdong University of Foreign Studies, Guangzhou, China xiali@gdufs.edu.cn, wjpan $@$ gdufs.edu.cn

# Abstract

Cross-prompt automated essay scoring (AES) aims to train models using essays from different source prompts and test them on new target prompt essays. A core challenge of the task is to learn as much shared knowledge as possible between essays from different prompts in order to better represent new prompt essays. Previous studies primarily focus on learning this knowledge on a general, coarse-grained level, ignoring that the shared knowledge among prompts is highly detailed and contains a more comprehensive range of information that is not fully investigated. In this paper, we propose a novel multi-aspect knowledge finding and aligning optimization strategy to better acquire this detailed various shared knowledge. We also introduce LLM to extract explicit, interpretable knowledge from implicit, multi-aspect shared knowledge and use this knowledge to improve the representation and evaluation performance of new prompt essays. We conduct extensive experiments on public datasets. The results show that our approach outperforms current state-of-the-art models and is effective on cross-prompt AES.

![](images/1796b1e1123be32c5c47ab8403a75e715f63995438ef80f272dbd786a8a9b5a5.jpg)  
Figure 1: Illustration of handcrafted knowledge signals and multi-aspect shared knowledge signals retrieved by unsupervised clustering. ${ \mathsf { P } } _ { 1 } \sim { \mathsf { P } } _ { K }$ denotes prompt 1 to prompt $K$ , and ${ \bf C } _ { 1 } \sim { \bf C } _ { K }$ denotes cluster 1 to cluster $K$ .

# Introduction

Automated essay scoring (AES) aims to evaluate the overall quality or specific traits of a given essay automatically. AES systems are widely used in the field of education assessment. It can reduce teachers’ workload, provide students with rich feedback, and improve the efficiency and fairness of grading (McNamara et al. 2015).

A majority of AES systems focus on scoring essays written in response to a specific essay prompt1, which means that the training and testing essays are from the same prompt (prompt-specific AES). Early works (Larkey 1998; Rudner and Liang 2002; Yannakoudakis, Briscoe, and Medlock 2011; Chen and He 2013; Attali and Burstein 2004; Phandi, Chai, and $\Nu \tt { g } 2 0 1 5 ,$ ) focus on extracting rich handcrafted features to train scoring models. With the advent of deep learning, many neural network-based architectures (Taghipour and $\mathrm { N g } 2 0 1 6$ ; Dong and Zhang 2016; Dong, Zhang, and Yang 2017; Tay et al. 2018; Mesgar and Strube 2018; Wang et al. 2022; Shibata and Uto 2022; Uto et al. 2023) have been proposed and achieve promising results. To provide enhanced feedback, several studies explore scoring specific traits of essays (Persing, Davis, and $\mathrm { N g } 2 0 1 0$ ; Persing and $\mathrm { N g } \ 2 0 1 4$ ; Mathias and Bhattacharyya 2020; Hussein, Hassan, and Nassef 2020). For example, Song et al. (2021) assesses essay structure and coherence for organization, and Ke et al. (2018) evaluate argument persuasiveness and logic.

As obtaining sufficient essays with human-rated scores for a new prompt is often difficult and expensive, crossprompt AES (Phandi, Chai, and $\Nu \mathrm { g } 2 0 1 5$ ; Cummins, Zhang, and Briscoe 2016; Jin et al. 2018; Cao et al. 2020; Ridley et al. 2020; Li, Chen, and Nie 2020; Ridley et al. 2021; Do, Kim, and Lee 2023; Chen and Li 2023, 2024) has gained increasing attention in recent years. These works employ different strategies, such as knowledge transfer, selfsupervised learning, and prompt-agnostic handcrafted feature extraction, to learn as many shared features among different prompts as possible to better represent and score the target prompt essays.

Although previous cross-prompt AES studies have achieved encouraging results, they primarily focus on acquiring the shared knowledge on a general, coarse-grained level, lacking the acquisition and learning of multi-aspect prompt-agnostic shared knowledge such as language grammar, text coherence, and writing style. While it is possible to tailor certain manual knowledge signals to guide the model in learning this kind of shared knowledge, the acquisition of more aspects of knowledge remains challenging due to its abstract nature and limited interpretability. For example, as shown in Figure 1 (upper part), we can set the essay qualities (e.g., high scores represent high quality and low scores represent low quality) as knowledge signals to align essays with similar quality. However, such manual designs may not be comprehensive; high-quality knowledge can also manifest in various aspects, such as having few grammatical errors (demonstrating knowledge of grammar) or displaying strong textual coherence (demonstrating knowledge of coherence). This suggests that the knowledge shared by different prompt essays can be multi-faceted. So, the questions are: 1) Can we develop an automated mechanism to help the model discover these various prompt-agnostic shared knowledge? 2) How to leverage these knowledge signals to optimize the representation of essays so that the model can effectively align essay representations of different prompts from much more diverse aspects, thereby enhancing the model’s generalization capacity for representing and scoring of essays on new unseen prompts?

To address these two issues, we introduce a novel multiaspect knowledge finding and aligning optimization strategy. The goal of the strategy is to automatically find the detailed multi-aspect knowledge and guide the model to learn the shared, consistent essay representation by aligning them in a prompt-independent manner. As shown in Figure 1 (lower part), we first use unsupervised clustering to automatically discover different clusters on essays sampled from all source prompts. Then, we optimize the model by aligning essays specific to the detailed, multi-aspect shared knowledge within the same cluster. In this way, the model autonomously learns consistent essay representations specific to different shared knowledge, independent of prompts. Based on the presented strategy, we propose a novel dynamic multi-aspect knowledge optimization framework for cross-prompt automated essay scoring (KAES). Our proposed framework employs an iterative parameter update scheme. Specifically, we first train a scoring model over a certain number of epochs. Then, we use the encoder of the scoring model to represent the sampled essays from different source prompts and conduct finding and aligning optimization steps. The encoder is expected to be updated and guide the scoring model in light of those multi-aspect shared knowledge. After a certain number of iterations, the model will effectively learn the shared knowledge, and finally represent and score new prompt essays appropriately.

It is worth noting that, as the shared knowledge contained in each cluster is implicit and lacks interpretability, we propose to use LLM to explicitly capture this shared knowledge by describing the representative assays around the center of the cluster as feature text and representing it using BERT. In addition, instead of concatenating all the knowledge to the essay representation $h _ { x }$ , we select the cluster most similar to the essay representation $h _ { x }$ as its corresponding shared knowledge $h _ { x _ { - } s h a r e d }$ . We concatenate this explicit shared knowledge representation $h _ { x _ { - } s h a r e d }$ with the essay representation $h _ { x }$ to form the final representation $h _ { f i n a l }$ for scoring.

The main contribution of this paper can be summarized as follows:

• To the best of our knowledge, this is the first attempt to explore the learning of consistent representation of essays specific to multi-aspect shared knowledge among different prompts by introducing a finding, aligning, and selection strategy.   
• We introduce LLM to extract explicit, interpretable knowledge from implicit, multi-aspect shared knowledge and use this knowledge to improve the representation and evaluation performance of new prompt essays.   
• We conduct extensive experiments on the ${ \mathsf { A S A P + + } }$ dataset, and the results show that our approach outperforms the state-of-the-art models and is effective in crossprompt multi-trait AES.

# Related Work

We will introduce related work from the perspectives of prompt-specific AES and cross-prompt AES.

Prompt-Specific AES. Prompt-specific AES methods focus on scoring essays belonging to the same prompt. The existing approaches employ various techniques to capture different features of the essay to improve scoring performance. For example, early studies (Rudner and Liang 2002; Chen and He 2013) focus on extracting rich handcrafted features to train scoring models. Later, neural networkbased methods employ different strategies to advance the field. For example, Dong and Zhang (2016) models the hierarchical structure of the essay and Tay et al. (2018) captures sentence-level dependencies. Recent transformerbased models (Wang et al. 2022; Uto et al. 2023) further improve contextual understanding. To provide enhanced feedback, some studies explore scoring specific traits of essays, such as organization (Persing, Davis, and $\mathrm { N g } 2 0 1 0$ ; Song et al. 2021), prompt adherence (Persing and $\mathrm { N g } \ 2 0 1 4 ,$ ) and argument persistence (Ke et al. 2018); others assess multiple traits of essays simultaneously (Mathias and Bhattacharyya 2020; Hussein, Hassan, and Nassef 2020). These advancements mark the shift from handcrafted features to neural models, enhancing scoring accuracy and feedback depth.

Cross-Prompt AES. Cross-prompt AES methods focus on scoring essays from different prompts. Some studies (Phandi, Chai, and $\mathrm { N g } 2 0 1 5$ ; Cummins, Zhang, and Briscoe 2016; Cao et al. 2020) apply the transfer learning method to adapt models to new prompts for cross-prompt AES. For example, Jin et al. (2018) and Li, Chen, and Nie (2020) introduce a two-stage pseudo-labeling approach to improve cross-prompt AES performance. Other works explore prompt-related features for cross-prompt AES. For example, Do, Kim, and Lee (2023) employ a prompt-aware framework to improve scoring performance of the target prompt. Jiang et al. (2023) develop a representation learning method to separate the shared and prompt-specific features to improve cross-prompt generalization.

The above cross-prompt AES methods mainly focus on the scoring of single attribute. In recent years, some crossprompt AES models try to explore multi-trait scoring without using any target prompt essays during training. Ridley et al. (2020, 2021) incorporate handcrafted features from only source-prompt data into their model; Chen and Li (2023) employ a prompt-mapping strategy to learn shared features of source and target prompts; Chen and Li (2024) propose a prompt-generalized learning method based on meta-learning and a level-aware learning strategy.

![](images/e8ffaabb2db9a41b0359f55d91eaba2b058f673faca113e1772e629a27382dd3.jpg)  
Figure 2: Overall architecture of KAES. During model training, multi-trait scoring training and knowledge finding and alignment are performed in alternating iterations. Through multi-trait scoring training, we obtain a shared encoder $f _ { \theta } ( . )$ . For knowledge finding and alignment, we first sample essays from the source prompts, encode them using $f _ { \theta } ( . )$ , and then perform clustering on these essays. We implicitly update $\theta$ to align essay representations in light of the knowledge signals from the clusters; We explicitly describe the features of the essays in the clusters using an LLM and obtain the knowledge representations using BERT. We dynamically select the knowledge representation $h _ { x _ { - } s h a r e d }$ corresponding to the centroid closest in distance to $h _ { x }$ for concatenation. During model inference, we select $h _ { x _ { - } s h a r e d }$ using the trained knowledge representations and centroids (shown in the gray box).

Our work also focuses on cross-prompt multi-trait essay scoring and training the model without seeing any target prompt data. Although these previous cross-prompt multitrait methods have achieved outstanding performance, they focus on capturing the generic shared knowledge across prompts. However, this is insufficient, as the shared knowledge is detailed and multifaceted. Unlike previous work, we propose a novel multi-aspect knowledge finding and aligning optimization strategy to automatically find and learn this knowledge and optimize our model in a prompt-independent manner.

# Approach

As shown in Figure 2, the overall architecture of our KAES framework consists of the multi-trait scoring module, the knowledge finding and aligning module, and the training and inference module.

# Task Definition

Our task focuses on cross-prompt AES, where a model is trained on source-prompt essays $D _ { s }$ and tested on unseen target-prompt essays $D _ { t }$ to predict multiple trait scores. Here, $\begin{array} { c c l } { D _ { s } } & { = } & { \{ ( x _ { i } , Y _ { i } ) | 1 \le i \le N _ { D _ { s } } \} } \end{array}$ , where $x _ { i }$ is the $i$ -th source prompt essay and $\begin{array} { r l } { Y _ { i } } & { { } = } \end{array}$ $\left\{ y _ { 1 } ^ { i } , y _ { 2 } ^ { i } , . . . , y _ { K _ { t } } ^ { i } \right\}$ represents its $K _ { t }$ trait scores. Similarly, $D _ { t } = \{ ( x _ { j } , Y _ { j } ) | 1 \leq j \leq N _ { D _ { t } } \}$ , with $x _ { j }$ as the $j$ -th target prompt essay and $Y _ { j } = \left\{ y _ { 1 } ^ { j } , y _ { 2 } ^ { j } , . . . , y _ { K _ { t } } ^ { j } \right\}$ as its trait scores.

# Multi-trait Scoring Module

Our scoring model, denoted as $F ( . )$ , consists of two components: the shared encoder $f _ { \theta } ( . )$ with parameters $\theta$ and the multi-trait scorer $f _ { \varphi } ( . )$ with parameters $\varphi$ .

For the shared encoder, similar to previous studies, we use CNN-LSTM hierarchical structure encoders (Dong, Zhang, and Yang 2017) or finetuned BERT encoders (Devlin et al. 2019) as the shared encoder $f _ { \theta } ( . )$ to encode the essay, denoted as $e$ . We use the same handcrafted features as Ridley et al. (2021), denoted as $f$ . The representation of the essay is the concatenation of $e$ and $f$ , denoted as $h _ { x } = [ e ; f ]$ .

In order to score multiple traits of an essay effectively, previous studies (Ridley et al. 2021; He et al. 2022) demonstrate that leveraging the mutual information of different traits is beneficial for trait scoring. Following their work, we adopt an independent dense layer for each trait and apply a trait attention mechanism. For the $k$ -th trait scoring, we first input the final essay representation into the corresponding $k$ -th dense layer to obtain the $k$ -th trait representation $a _ { k }$ . We then apply a trait attention mechanism (Ridley et al. 2021) to obtain its attention vector $p _ { k }$ . Finally, We concatenate $a _ { k }$ and $p _ { k }$ to get the final representation of the $k$ -th trait $g _ { k } = [ a _ { k } ; p _ { k } ]$ . The $k$ -th trait’s predicted score $\hat { y } _ { k }$ is obtained through a sigmoid activation:

$$
\hat { y } _ { k } = \mathrm { S i g m o i d } \left( g _ { k } \right) , k = 1 , . . . , K _ { t }
$$

We use mean square error as the scoring loss function.

Assuming that there are a total of $N$ essays and $K _ { t }$ traits, the loss is defined as:

$$
\mathcal { L } _ { a e s } = \frac { 1 } { N K _ { t } } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K _ { t } } \left( \hat { y } _ { k } ^ { ( i ) } - y _ { k } ^ { ( i ) } \right) ^ { 2 }
$$

where $y _ { k } ^ { ( i ) }$ and $\hat { y } _ { k } ^ { ( i ) }$ are the ground truth and the predicted score of the $k$ -th trait for the $i$ -th essay, respectively. As some traits do not have ground truth scores, the masking mechanism (Ridley et al. 2021) is employed in the calculation.

# Knowledge Finding and Aligning Module

To make our model capture the multi-aspect shared knowledge, we propose a knowledge finding and aligning strategy. First, we automatically discover this knowledge through unsupervised clustering on sampled essays, then implicitly align essay representations under the guidance of cluster signals, and finally use LLM to explicitly capture these various features based on the given texts from each cluster.

Knowledge Discovering We perform unsupervised clustering on a batch of essays sampled from all source prompts to discover various shared knowledge automatically. To enhance sample diversity, we employ stratified sampling to maintain balance across source prompts. Specifically, we select $B$ essays per prompt and use the shared encoder $f _ { \theta } ( . )$ to encode them. Our KAES is not limited to one specific clustering method. Different clustering methods are applicable and available, such as K-Means (Hartigan and Wong 1979) and GMM (Rasmussen 1999). We cluster the sampled essays and obtain a total of $K _ { c }$ clusters, denoted as $\big \{ C _ { 1 } , C _ { 2 } , { \dot { \ldots } } , C _ { K _ { c } } \big \}$ . The centroid of these clusters are represented as $\{ \mu _ { 1 } , \mu _ { 2 } , . . . , \mu _ { K _ { c } } \}$ . During clustering, essays sharing similar knowledge across different prompts are clustered together. We treat each cluster label as a guiding signal corresponding to that knowledge, which will be used for the subsequent aligning and updating process.

Implicitly Knowledge Aligning Using the knowledge signals retrieved by unsupervised clustering, we aim to bring essays from the same cluster closer, thereby aligning their representations in the semantic space and consequently improving the model’s ability to learn this shared knowledge. There are various approaches to achieving this goal, such as minimizing the distance between essay representations and centroids. Inspired by Wang et al. (2019), we employ a method that involves minimizing the divergence between two distributions, which has been shown to yield superior optimization results. We first obtain the clustering distribution $Q$ , which measures the distance between each essay representation $e _ { i }$ and cluster center embedding $\mu _ { c }$ . Based on $Q$ , we construct an augment distribution $P$ , which amplifies the influence of the assignments corresponding to high probabilities in $Q$ , which indicates that the essays are in closer proximity to the cluster center. The calculations of the two distributions $Q$ and $P$ are as follows:

$$
q _ { i c } = \frac { \left( 1 + \left\| e _ { i } - \mu _ { c } \right\| ^ { 2 } \right) ^ { - 1 } } { \sum _ { k } { \left( 1 + \left\| e _ { i } - \mu _ { k } \right\| ^ { 2 } \right) ^ { - 1 } } }
$$

$$
p _ { i c } = \frac { q _ { i c } ^ { 2 } / \sum _ { j } q _ { j c } } { \sum _ { k } \left( q _ { i k } ^ { 2 } / \sum _ { j } q _ { j k } \right) }
$$

where $q _ { i c }$ is the probability in $Q , p _ { i c }$ is the probability in $P$ . We then encourage the distribution $P$ to approximate $Q$ , which can promote closer proximity among essays that belong to the same cluster. We employ Kullback-Leibler (KL) divergence to measure the difference between the two distributions and construct a loss to update the parameters $\theta$ of the shared encoder. The loss is defined as follows:

$$
\mathcal { L } _ { c l u s t e r } = \mathrm { K L } ( P \| Q ) = \sum _ { i } \sum _ { u } p _ { i u } \log \frac { p _ { i u } } { q _ { i u } }
$$

where $p _ { i u } , q _ { i u }$ represent the probability that essay $e _ { i }$ is assigned to cluster $C _ { u }$ in the two distributions, $P$ and $Q$ , respectively.

Explicitly Knowledge Capturing Due to the excellent text understanding and generation capabilities of existing large language models, such as ChatGPT (Floridi and Chiriatti 2020), these models can easily analyze the features of multiple texts and generate coherent descriptions. To explicitly capture the shared knowledge, we propose leveraging ChatGPT to generate the feature descriptions based on the text of essays within each cluster.

Specifically, for each cluster, we select $M$ representative essays that are closest to the centroid $\mu _ { k }$ :

$$
\left\{ i _ { 1 } , i _ { 2 } , . . . , i _ { M } \right\} = \mathrm { a r g m i n } _ { i } \left\{ d \left( e _ { k i } , \mu _ { k } \right) \right\}
$$

where $d ( e , \mu )$ denotes the euclidean distance between the essay representation $e$ and the centroid $\mu , e _ { k i }$ is the representation of the $i$ -th essay in cluster $C _ { k }$ and $i \in \{ 1 , 2 , \ldots , | C _ { k } | \}$ . We carefully craft an instruction to guide ChatGPT in capturing the multi-aspect shared features of each cluster. The instruction is designed based on keywords extracted and synthesized from rubrics in the experiment dataset. ChatGPT generates a description of the common characteristics possessed by the selected representative essays for each cluster $C _ { k }$ :

$$
T _ { k } = \mathrm { P r o m p t } ( t e x t . i _ { 1 } , t e x t . i _ { 2 } , . . . , t e x t . i _ { M } )
$$

where $T _ { k }$ is a textual description of shared features of the given texts. We then obtain the shared knowledge representation $r _ { k }$ using BERT and its [CLS] token output:

$$
r _ { k } = \mathbf { B E R T } ( T _ { k } )
$$

For all $K _ { c }$ clusters, we obtain a set of shared knowledge representations, denoted as $\{ r _ { 1 } , r _ { 2 } , . . . , r _ { k c } \}$ .

Knowledge Selection As the shared knowledge across prompts is multifaceted, the essays from different prompts share the knowledge of different aspects. Therefore, when scoring an essay $x$ , we propose a selection mechanism to enable the model to understand which aspect of knowledge this essay shares. This mechanism selects the corresponding shared knowledge from all the captured knowledge representations for the given essay. Specifically, we compute the distance between the essay representation $h _ { x }$ and the centroids $\mu$ of these clusters. We select the explicit knowledge representation that has the smallest distance to $h _ { x }$ , denoted as $h _ { x . s h a r e d } = r _ { \mathrm { a r g m i n } _ { i } \| h _ { x } - \mu _ { i } \| }$ . Finally, we concatenate $h _ { x }$ with $h _ { x _ { - } s h a r e d }$ and input the concatenated vector $h _ { f i n a l }$ into the multi-trait scorer for scoring. The knowledge representations dynamically change with each clustering iteration.

# Training and Inference Module

Model Training Stage Our model employs an iterative training scheme in which the model training and knowledge finding and aligning steps are conducted alternately. Specifically, we first train a scoring model over a certain number of epochs using the loss in Equation (2). Then, we conduct finding and aligning steps to refine the model parameters using the clustering loss Equation (5) and obtain the knowledge representations as shown in Equation (8). After a certain number of iterations, the model will effectively learn the shared knowledge from various aspects and finally represent and score new prompt essays appropriately.

To preserve established essay representations and maintain the model’s scoring performance, it is important to avoid overtraining the knowledge finding and aligning modules. We set the epoch of the multi-trait scoring training module to $\tau$ , and we perform the knowledge finding and aligning steps one time in each iteration to allow for a moderate refinement of the model’s encoding.

Model Inference Stage During model inference, we first input the essay into the trained encoder to obtain its representation. We select the shared knowledge representation corresponding to the essay from the trained knowledge representations. The selection mechanism is the same and is based on the centroids stored in our model. We then concatenate the essay representation with the knowledge representation and input the concatenated representation into the multi-trait scorer to predict scores.

# Experiments and Results

# Dataset and Evaluation Metrics

We use the ASAP and ${ \mathsf { A S A P + + } }$ datasets (Mathias and Bhattacharyya 2018) to evaluate our method. It includes 12,978 English writings in response to eight prompts. We employ the prompt-wise validation method, which is commonly utilized in existing cross-prompt AES studies (Jin et al. 2018; Ridley et al. 2021; Chen and Li 2023). Our evaluation metric is Quadratic Weighted Kappa (QWK), which is the official metric for the Kaggle competition ASAP and was widely used in previous AES methods. QWK quantifies the level of agreement between the human rater and the AES model. A higher QWK value indicates better scoring performance.

# Implementation Details

For the encoder, we use 200 CNN filters and 200 LSTM units in hierarchical encoder and employ the bert-base-uncased BERT model. We set the batch size $B$ to 200, the number of clusters $K _ { c }$ to 10, and the number of representative essays $M$ to 3. The initial shared representation $h _ { x _ { - } s h a r e d }$ is randomly initialized. To avoid impairing original scoring performance, $\tau$ is set to 5, as $\mathcal { L } _ { a e s }$ stabilizes at 5 epoch. Our experiments are conducted on an NVIDIA RTX8000 GPU. The best model is selected based on the highest average QWK on the validation set. We run it three times and report the average results on the test set. Our code is available at https://github.com/gdufsnlp/KAES.

# Baselines

We compare the baselines as follows. We use Hier att (Dong and Zhang 2016), AES aug (Hussein, Hassan, and Nassef 2020), PAES (Ridley et al. 2020), CTS no att, CTS (Ridley et al. 2021), PMAES (Chen and Li 2023) and PLAES (Chen and Li 2024) as baselines for their modeling multi-traits scoring and having the same experimental setups with ours. We also use ChatGPT and a Finetune BERT as baselines for comparison with our BERT encoder-based models. We experiment with five model variants based on different encoders and clustering methods: KAES(hier/bert)+kmeans/gmm. We use GPT-3.5 as the LLM backbone for the five model variants.

# Main Results

The main results of our method (KAES) on each prompt and each trait are shown in Table 1 and 2. The results indicate that KAES demonstrates effectiveness in crossprompt automated scoring of essay traits. Specifically, KAES(hier) $+ \mathrm { g m m }$ outperforms all baselines, with an average improvement of $2 . 2 \%$ for prompts and $2 . 4 \%$ for traits compared to the SOTA model PLAES, achieving maximum gains of $5 . 1 \%$ on prompt P5 and $5 . 7 \%$ on trait WC. This outstanding improvement shows that our proposed multi-aspect knowledge optimization strategy is effective for enhancing multi-trait scoring task.

The results also show that employing either GMM or Kmeans yields comparable model performance, with GMM performing slightly better. This improvement may stem from its compatibility with the Gaussian-based alignment strategy employed in KAES. Additionally, the results indicate that different essay encoders yield varying performance on the cross-prompt AES task. KAES(hier) $+ \mathrm { g m m }$ outperforms KAES(bert)+gmm by $7 . 0 \%$ on prompts and $6 . 3 \%$ on traits.

Considering the distinct text generation capacities of LLMs, we also compare the performance impact of using GPT-4 and GPT-3.5 as the LLM backbone for KAES. GPT4 achieves the best performance (0.606 on prompts, 0.602 on traits) compared to GPT-3.5 (0.597 on prompts, 0.594 on traits). These results suggest higher-quality text generation enhances scoring by improving descriptive features.

# Ablation Studies

The ablation results in Table 3 provide insights into the contribution of two components in KAES: the cluster-based alignment optimization process (Cluster) and the LLMbased knowledge capture process $( L L M )$ . We observe the following findings: 1) When LLM is removed, model performance decreases by $0 . 9 \%$ on prompts and $1 . 1 \%$ on traits. 2) When both Cluster and LLM are removed, performance drops by $3 . 6 \%$ on prompts and $3 . 9 \%$ on traits. This shows that our multi-aspect shared knowledge acquisition strategy significantly enhances the overall performance of KAES.

Table 1: Main results on each prompt. The average QWK across all traits for each prompt is reported.   

<html><body><table><tr><td>Model</td><td>P1</td><td>P2</td><td>P3</td><td>P4</td><td>P5</td><td>P6</td><td>P7</td><td>P8</td><td>AVG</td></tr><tr><td>GPT-3.5-turbo (0-shot)</td><td>0.264</td><td>0.492</td><td>0.351</td><td>0.437</td><td>0.516</td><td>0.489</td><td>0.153</td><td>0.307</td><td>0.376</td></tr><tr><td>Finetune BERT</td><td>0.556</td><td>0.549</td><td>0.616</td><td>0.618</td><td>0.655</td><td>0.457</td><td>0.345</td><td>0.275</td><td>0.509</td></tr><tr><td>Hi att (Dong and Zhang 2016)</td><td>0.315</td><td>0.478</td><td>0.317</td><td>0.478</td><td>0.375</td><td>0.357</td><td>0.205</td><td>0.265</td><td>0.349</td></tr><tr><td>AES aug (Hussein,Hassan,and Nassef 2020)</td><td>0.330</td><td>0.518</td><td>0.299</td><td>0.477</td><td>0.341</td><td>0.399</td><td>0.162</td><td>0.200</td><td>0.341</td></tr><tr><td>PAES (Ridley et al. 2020)</td><td>0.605</td><td>0.522</td><td>0.575</td><td>0.606</td><td>0.634</td><td>0.545</td><td>0.356</td><td>0.447</td><td>0.536</td></tr><tr><td>CTS no att (Ridley et al. 2021)</td><td>0.619</td><td>0.539</td><td>0.585</td><td>0.616</td><td>0.616</td><td>0.544</td><td>0.363</td><td>0.461</td><td>0.543</td></tr><tr><td>CTS (Ridley et al. 2021)</td><td>0.623</td><td>0.540</td><td>0.592</td><td>0.623</td><td>0.613</td><td>0.548</td><td>0.384</td><td>0.504</td><td>0.553</td></tr><tr><td>PMAES (Chen and Li 2023)</td><td>0.656</td><td>0.553</td><td>0.598</td><td>0.606</td><td>0.626</td><td>0.572</td><td>0.386</td><td>0.530</td><td>0.566</td></tr><tr><td>PLAES (Chen and Li 2024)</td><td>0.648</td><td>0.563</td><td>0.604</td><td>0.623</td><td>0.634</td><td>0.593</td><td>0.403</td><td>0.533</td><td>0.575</td></tr><tr><td>Our Model</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>KAES(bert)+kmeans</td><td>0.580</td><td>0.570</td><td>0.591</td><td>0.620</td><td>0.663</td><td>0.510</td><td>0.325</td><td>0.301</td><td>0.520</td></tr><tr><td>KAES(bert)+gmm</td><td>0.604</td><td>0.581</td><td>0.568</td><td>0.651</td><td>0.668</td><td>0.486</td><td>0.349</td><td>0.310</td><td>0.527</td></tr><tr><td>KAES(hier)+kmeans</td><td>0.632</td><td>0.617</td><td>0.627</td><td>0.630</td><td>0.656</td><td>0.573</td><td>0.429</td><td>0.598</td><td>0.595</td></tr><tr><td>KAES(hier)+gmm</td><td>0.623</td><td>0.611</td><td>0.619</td><td>0.636</td><td>0.685</td><td>0.593</td><td>0.433</td><td>0.577</td><td>0.597</td></tr></table></body></html>

<html><body><table><tr><td>Model</td><td>Overall</td><td>Content</td><td>Org</td><td>WC</td><td>SF</td><td>Conv</td><td>PA</td><td>Lang</td><td>Nar</td><td>AVG</td></tr><tr><td>GPT-3.5-turbo (0-shot)</td><td>0.406</td><td>0.411</td><td>0.286</td><td>0.330</td><td>0.282</td><td>0.285</td><td>0.426</td><td>0.462</td><td>0.439</td><td>0.370</td></tr><tr><td>Finetune BERT</td><td>0.578</td><td>0.490</td><td>0.361</td><td>0.542</td><td>0.527</td><td>0.298</td><td>0.548</td><td>0.611</td><td>0.591</td><td>0.505</td></tr><tr><td>Hi att (Dong and Zhang 2016)</td><td>0.453</td><td>0.348</td><td>0.243</td><td>0.416</td><td>0.428</td><td>0.244</td><td>0.309</td><td>0.293</td><td>0.379</td><td>0.346</td></tr><tr><td>AES aug (Hussein,Hassan,and Nassef 2020)</td><td>0.402</td><td>0.342</td><td>0.256</td><td>0.402</td><td>0.432</td><td>0.239</td><td>0.331</td><td>0.313</td><td>0.377</td><td>0.344</td></tr><tr><td>PAES (Ridley et al. 2020)</td><td>0.657</td><td>0.539 0.414</td><td></td><td>0.531</td><td>0.536</td><td>0.357</td><td>0.570</td><td>0.531</td><td>0.605</td><td>0.527</td></tr><tr><td>CTS no att (Ridley et al. 2021)</td><td>0.659</td><td>0.541</td><td>0.424</td><td>0.558</td><td>0.544</td><td>0.387</td><td>0.561</td><td>0.539</td><td>0.605</td><td>0.535</td></tr><tr><td>CTS (Ridley et al. 2021)</td><td>0.670</td><td>0.555</td><td>0.458</td><td>0.557</td><td>0.545</td><td>0.412</td><td>0.565</td><td>0.536</td><td>0.608</td><td>0.545</td></tr><tr><td>PMAES (Chen and Li 2023)</td><td>0.671</td><td>0.567</td><td>0.481</td><td>0.584</td><td>0.582</td><td>0.421</td><td>0.584</td><td>0.545</td><td>0.614</td><td>0.561</td></tr><tr><td>PLAES (Chen and Li 2024)</td><td>0.673</td><td>0.574</td><td>0.491</td><td>0.579</td><td>0.580</td><td>0.447</td><td>0.601</td><td>0.554</td><td>0.631</td><td>0.570</td></tr><tr><td>Our Model</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>KAES(bert)+kmeans</td><td>0.551</td><td>0.501</td><td>0.407</td><td>0.527</td><td>0.508</td><td>0.369</td><td>0.565</td><td>0.634</td><td>0.627</td><td>0.521</td></tr><tr><td>KAES(bert)+gmm</td><td>0.551</td><td></td><td>0.515 0.402</td><td>0.583</td><td>0.571</td><td>0.366</td><td>0.565</td><td></td><td>0.614 0.611</td><td>0.531</td></tr><tr><td>KAES(hier)+kmeans</td><td>0.695</td><td></td><td>0.589 0.547</td><td>0.604</td><td>0.610</td><td>0.487</td><td>0.599</td><td>0.563</td><td>0.632</td><td>0.592</td></tr><tr><td>KAES(hier)+gmm</td><td>0.679</td><td></td><td>0.594 0.543</td><td>0.636</td><td>0.574</td><td>0.476</td><td>0.611</td><td>0.595</td><td>0.641</td><td>0.594</td></tr></table></body></html>

Table 2: Main results of each trait. The average QWK across all prompts for each trait is reported.

# Discussion What is the Impact of Our Method on In-domain Performance while Enhancing Cross-domain Performance?

Previous research has primarily focused on cross-domain performance without adequately addressing in-domain stability (Yang et al. 2024). We investigate the effect of our method on in-domain performance. We adjust our experimental setup by withholding $10 \%$ of essays from all source prompts and training the model with the remaining $90 \%$ essays. We evaluate target prompts P1 and P2 for cross-domain performance and on the withheld essays for in-domain performance, focusing on the ’overall’ trait score common to all prompts. Results in Table 4 indicate that while in-domain performance for each prompt varies, it remains generally stable or slightly decreases average. However, cross-domain performance on P1 and P2 both improves notably, showing that our method effectively enhances cross-domain capabilities without negatively impacting in-domain performance.

# Does Our Method Need Large Increases in Model Parameters and Operational Costs?

As shown in Table 5, KAES has a parameter count of 837K, similar to CTS (855K), yet significantly enhances scoring accuracy, achieving improvements of $4 . 2 \%$ and $4 . 7 \%$ on prompts and traits, respectively. Moreover, compared to the SOTA model PLAES (1.47M), KAES operates with only half of PLAES’s parameters, demonstrating its advantage in terms of parameter efficiency. In terms of the operational costs, KAES introduces additional time costs for the clustering process. Training KAES(hier) takes 180 seconds per epoch, totaling 2.5 hours over 50 epochs, while KAES(bert)

Table 3: Ablation results on prompts and traits. KAES denotes our model KAES(hier)+kmeans.   

<html><body><table><tr><td>Model</td><td>Avg. QWK on prompts</td><td>Avg. QWK on traits</td></tr><tr><td>KAES</td><td>0.595</td><td>0.592</td></tr><tr><td>KAES w/o LLM</td><td>0.586</td><td>0.581</td></tr><tr><td>KAESw/oCluster&LLM</td><td>0.559</td><td>0.553</td></tr></table></body></html>

Table 4: In-domain and cross-domain model performance.   

<html><body><table><tr><td>Model</td><td>in domain</td><td>cross domain</td></tr><tr><td></td><td>AVG (source)</td><td>P1 (target)</td></tr><tr><td>Hier</td><td>0.737</td><td>0.562</td></tr><tr><td>Hier+Cluster</td><td>0.729</td><td>0.634</td></tr><tr><td>Hier+Cluster+LLM</td><td>0.733</td><td>0.675</td></tr><tr><td></td><td>AVG (source)</td><td>P2 (target)</td></tr><tr><td>Hier</td><td>0.779</td><td>0.628</td></tr><tr><td>Hier+Cluster</td><td>0.777</td><td>0.657</td></tr><tr><td>Hier+Cluster+LLM</td><td>0.776</td><td>0.662</td></tr></table></body></html>

requires 1200 seconds per epoch, reaching 6.7 hours over 20 epochs. Regarding memory usage, due to the need to load sampled essays from each prompt per clustering iteration, KAES occupies external GPU memory, with peak usage reaching 17,284 MiB when using $10 \%$ sample essays.

# Can the Knowledge within Clusters be Visualized through Feature Descriptions Generated by LLM?

Knowledge within clusters is difficult to interpret and quantify manually. Our approach utilizes LLM to generate coherent, human-understandable descriptions of shared features for each cluster. These generated feature descriptions enable qualitative visualization of this knowledge. Figure 3 shows the LLM instruction and two cases of generated feature descriptions. The instruction provides potentially shared feature dimensions summarized from the scoring rubric to ensure that the descriptions reflect the quality of the essay. It is evident that in both cases, the LLM effectively describes the features of knowledge within clusters based on rubric dimensions, with distinct descriptions for each cluster. For example, Case 1 highlights knowledge about grammatical errors in the corresponding cluster, while Case 2 describes strong language expression of the essays in its cluster.

Table 5: Comparison of parameters and performance.   
Instruction for LLM:   

<html><body><table><tr><td>Model</td><td>Number of Parameters</td><td>Avg. QWK on prompts</td><td>Avg. QWK on traits</td></tr><tr><td>CTS</td><td>855 K</td><td>0.553</td><td>0.545</td></tr><tr><td>PMAES</td><td>174 K</td><td>0.566</td><td>0.561</td></tr><tr><td>PLAES</td><td>1.47 M</td><td>0.575</td><td>0.570</td></tr><tr><td>KAES (ours)</td><td>837K</td><td>0.595</td><td>0.592</td></tr></table></body></html>

Examinetheessaysprovidedbelowtoidentifytheshared featuresthatdefine theirquality.   
Theshared features include,butarenot limited to:   
·Readability:How easily the text can be read and understood.   
·Clarity and Focus:Precision and clear direction in presenting main ideas. ·Punctuation: Correct use of punctuation marks.   
Insteadofanalyzingeach essay individually,summarize theseshared features in aconcise analysis.   
Essay 1: [Text of essay 1]   
Essay 2: [Text of essay 2]   
Essay M: [Text of essay M]   
Analysis:

Feature Description Case 1:

Theessaysprovided exhibit several shared characteristics that define their poor quality. These include poor readability due to frequent errors in grammar,punctuation,and spelling.....The absenceofwriter's techniques andfigurativelanguagecontributestoalackofengagementandexpression in the texts. Overall, these essays demonstrate a lack of mastery over the conventions of writing.

Feature Description Case 2:

The essays demonstrate a strong command of language and communication skills.They are all highly readable and exhibit a clear focus on the main ideas.....Theuse of literary devices,such as imagery and dialogue，adds depth to the essays. The vocabulary_is appropriate and sophisticated. Overall, these essays showcase the writers' ability to express themselves clearly and effectively, making them engaging and impactful pieces ofwriting.

Figure 3: Prompt for LLM and feature descriptions cases. Purple text indicates representative essay input and blue and red text denote negative and positive descriptions.

# How Do the Number of Clusters and the Sampling Quantity Influence the Model’s Performance?

To assess how variations in clustering parameters impact the model’s performance, we adjust the number of clusters and the quantity of sampled essays, observing changes in model performance. It shows that model performance subtly varies with different cluster numbers $K _ { c }$ , achieving optimal performance at $K _ { c } = 5$ . Additionally, while sampling only 50 essays per prompt exhibits limited effectiveness, increasing the number of sampled essays markedly enhances performance, indicating that larger sample sizes for clustering lead to improved results.

# Conclusion

This paper introduces a novel multi-aspect shared knowledge finding and aligning strategy and an iterative trainingbased optimization mechanism (KAES) for cross-prompt multi-trait AES. We pioneer the use of clustering and LLMbased methods to identify and leverage multi-aspect shared knowledge. Our approach enhances the interpretability and utility of implicit knowledge through explicit knowledge representations. Extensive experiments on the public dataset demonstrate that our method significantly outperforms existing SOTA models in scoring performance and generalization. In addition, we believe that our approach is well-suited for other tasks, such as cross-domain sentiment classification and cross-lingual text classification.