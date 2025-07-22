# Learn How to Query from Unlabeled Data Streams in Federated Learning

Yuchang $\mathbf { S u n } ^ { 1 }$ , Xinran $\mathbf { L i } ^ { 1 }$ , Tao $\mathbf { L i n } ^ { 2 }$ , Jun Zhang

1Department of Electronic and Computer Engineering, The Hong Kong University of Science and Technology 2Westlake University {yuchang.sun, xinran.li}@connect.ust.hk, lintao $@$ westlake.edu.cn, eejzhang@ust.hk

# Abstract

Federated learning (FL) enables collaborative learning among decentralized clients while safeguarding the privacy of their local data. Existing studies on FL typically assume offline labeled data available at each client when the training starts. Nevertheless, the training data in practice often arrive at clients in a streaming fashion without ground-truth labels. Given the expensive annotation cost, it is critical to identify a subset of informative samples for labeling on clients. However, selecting samples locally while accommodating the global training objective presents a challenge unique to FL. In this work, we tackle this conundrum by framing the data querying process in FL as a collaborative decentralized decision-making problem and proposing an effective solution named LeaDQ, which leverages multi-agent reinforcement learning algorithms. In particular, under the implicit guidance from global information, LeaDQ effectively learns the local policies for distributed clients and steers them towards selecting samples that can enhance the global model’s accuracy. Extensive simulations on image and text tasks show that LeaDQ advances the model performance in various FL scenarios, outperforming the benchmarking algorithms.

# 1 Introduction

Federated learning (FL) (McMahan et al. 2017) has emerged as a distributed paradigm for training machine learning (ML) models while preserving local data privacy. In FL, the model training process involves multiple clients, each possessing its own local training data with different distributions. These clients jointly optimize an ML model based on their local data and periodically upload the model updates to the server. Afterwards, the server updates the global model by applying the aggregated updates and distributes the current model to the clients for next-iteration training. This approach has gained significant attention in practical applications such as healthcare (Rieke et al. 2020) and Internet of Things (Zhang et al. 2022a), due to its potential to address privacy concerns associated with centralized data storage and processing.

Most of existing FL studies assume that clients have a fixed pool of training data samples with ground-truth labels (Li et al. 2023; Ye et al. 2023; Yang, Xiao, and Ji 2023;

Shi et al. 2023b), which, however, is unrealistic in many applications. In practice, the data samples usually arrive at the clients without any label (Jin et al. 2020). For example, hospitals collect raw medical images in the course of routine care which are initially unannotated raw images. As labeling these images requires highly experienced clinical staff to spend minutes per image, it is more feasible to select a subset of samples for label querying and subsequent model training. In this case, designing an effective strategy to identify the samples that could provide the most benefit in model training becomes a critical problem.

Data querying strategies for non-FL settings have been widely studied in the active learning (AL) literature (Settles 2009; Liu et al. 2022; Cacciarelli and Kulahci 2024). However, there are limited studies towards adapting these strategies to FL settings due to the challenges brought by the inherently decentralized nature of FL. The major challenge lies in the conflict between the collaborative goal of optimizing the global model and the limited access to local data on clients. To be specific, FL aims to optimize the global model over all the training data across clients while each client can only access its private training data. Given this fact, it is difficult for clients to select the most critical data samples which promote the global training. One straightforward approach is prioritizing local data on each client, which, however, is a myopic strategy without considering the goal of global model training, ultimately leading to suboptimal data selections. More recently, there are some attempts to explore the data querying problem in FL with fixed unlabeled dataset by designing local selection rules (Kim et al. 2023; Cao et al. 2023). Nevertheless, the performance of these methods is restricted since the data selections on clients tend to operate in a non-cooperative fashion without the guidance towards the global training goal. To solve these problems, it is imperative to design a specific data querying strategy tailored for FL with decentralized data streams while accounting for the collaborative objective of global model training.

In this work, we investigate the data querying problem in FL, particularly focusing on a case where unlabeled data arrive at clients in a streaming fashion. By alternating local data query and model training procedures, clients aim to collaboratively optimize the global model, as illustrated in Fig. 1. In the subsequent sections, we first empirically show the impact of various data querying strategies on the performance of the global model. In particular, we find that a coordinated data querying strategy, which considers the training goal from a global perspective, yields greater performance benefit compared to individual data querying decisions. Motivated by this observation, we propose a data querying algorithm named LeaDQ where clients query data in a decentralized manner while considering the global objective. Specifically, we formulate the data querying problem in FL as a decentralized partially observable Markov decision process (Dec-POMDP) and leverage the multi-agent reinforcement learning (MARL) algorithms to learn local data querying policies for clients. By implicitly incorporating the status of the global model, the learned policies effectively guide the clients to make query decisions that are collaborative and ultimately beneficial for global training. We conduct simulations to compare the LeaDQ algorithm with several representative data querying strategies. The experimental results on various image and text tasks demonstrate that LeaDQ selects samples that result in more meaningful model updates, leading to improved model accuracy.1

# 2 Related Works

Active learning AL algorithms are designed to select a subset of unlabeled data samples and query their labels from an oracle. The design goal is to improve the model performance on the target dataset by selecting the most informative samples within the querying cost budget (Liu et al. 2022; Cacciarelli and Kulahci 2024). The classic data selection metrics are uncertainty-based, diversity-based or the hybrid ones. Specifically, the uncertainty-based AL algorithm (Wang and Shang 2014) selects the unlabeled data with the lowest prediction confidence, which is straightforward for improving the performance of ML models. Meanwhile, the diversity-based method (Sener and Savarese 2018) aims to reduce redundancy in the selected instances for labeling, ensuring a more efficient use of labeling resources by avoiding similar samples. Furthermore, the hybrid method attempts to combine the strengths of both uncertainty and diversity in the selection criteria. In particular, some works (Fang, Li, and Cohn 2017; Zhang et al. 2022b) leverage reinforcement learning (RL) to learn a data query policy which is able to select the impactful data samples.

Federated active learning Recent studies have extended the investigations on data query problems to FL scenarios, termed as federated active learning (FAL) (Kim et al. 2023; Cao et al. 2023; Ahn et al. 2024; Wu et al. 2023; Kong et al. 2024). In FAL, clients select some local data samples and ask the oracles to label these samples. Kim et al. (2023) design a two-step algorithm named LoGo to select informative data samples while resolving the inter-class diversity on clients. Meanwhile, KAFAL (Cao et al. 2023) leverages the specialized knowledge by the local clients and the global model to deal with their objective mismatch. Nevertheless, the clients only have access to their own data and tend to query the samples that are beneficial for their local training objective. Because of the discrepancy between the local and global training objectives, the queried data may bring less benefit for the global model training. Moreover, these works assume that all unlabeled data samples are available beforehand and require evaluating them at every time. Different from these works, we focus on a more challenging streambased setting, where the unlabeled samples sequentially arrive at clients and may be accessed only once.

![](images/a8fde9ba0457eca29a38b48399de8c51d3dccfe236f9209bda334871e5f2fefd.jpg)  
Figure 1: Workflow of the proposed LeaDQ framework.

In addition, another thread of works (Chen et al. 2020; Marfoq et al. 2023; Gong et al. 2023; Shi et al. 2023a) designs various approaches to deal with the unlabeled data or streaming data in FL. These works, however, are beyond the scope of this paper, since they tackle other aspects such as self-supervised learning or data storage on clients instead of the data querying problem.

# 3 Problem Formulation

We consider an $\mathrm { F L }$ system where $K$ clients with unlabeled data streams cooperate to train an ML model by alternating data querying and model training. The goal of clients is to minimize the following objective:

$$
\operatorname* { m i n } _ { \theta \in \mathbb { R } ^ { M } } \mathbb { E } _ { ( \mathbf { x } , y ) \sim \mathcal { P } } [ \ell ( \theta ; \mathbf { x } , y ) ] ,
$$

where $\ell ( \theta ; { \mathbf x } , y )$ denotes the loss value of model $\theta$ on data sample $( \mathbf { x } , y )$ with data input $\mathbf { x }$ and corresponding label $y$ . Note that the target data distribution $\mathcal { P }$ is unknown and inaccessible. We assume that the aggregated training data on all clients follow the same distribution as that of the target dataset. Nevertheless, the local data distribution $\mathcal { P } _ { k }$ of each client $k \in [ K ]$ is typically different from the global data distribution, which is also known as the non-independent and identically distributed (non-IID) setup in FL (Kairouz et al. 2021). Formally, the assumptions on data distributions are given as: $\cup _ { k \in [ K ] } \mathcal { P } _ { k } = \mathcal { P }$ and $\mathcal { P } _ { k } \neq \mathcal { P } , \forall k \in [ K ]$ .

Raw training data without ground-truth labels arrive at client $k$ according to the underlying distribution $\mathcal { P } _ { k }$ (Marfoq et al. 2023). According to the data arrival status, the training process of optimizing model $\theta$ can be divided into $R$ rounds. At the beginning of each round $r \in \{ 1 , 2 , \ldots , R \}$ , some new unlabeled data samples arrive at each client. To utilize these data for model training, clients actively query the labels of some selected samples from an oracle. Afterwards, they collaboratively train the ML model with the current labeled dataset. The above process is illustrated in Fig. 2. In the following, we introduce the procedure of data querying and model training in detail.

Active data querying In the $r$ -th round, a set of $N _ { u }$ unlabeled data samples arrives at client $k$ , denoted by $\mathcal { U } _ { k } ^ { r } ~ =$ $\{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { N _ { u } }$ where $\mathbf { x } _ { i }$ is the feature of the $i$ -th sample. Each client then makes a binary decision $\mathbf { a } _ { k } ^ { r } = \{ 0 , 1 \} ^ { N _ { u } }$ , indicating whether a particular sample is selected for labeling.2 Specifically, $\mathbf { a } _ { k } ^ { r } [ i ] ^ { \bar { \mathbf { \alpha } } } = 1$ if the $i$ -th sample is chosen for querying and $\mathbf { a } _ { k } ^ { r } [ i ] = 0$ otherwise. If being queried, an oracle provides the ground-truth label $y _ { i } \in \mathcal { V }$ for this sample $\mathbf { x } _ { i }$ . We denote the number of samples for labeling per round on each client as Nq and the queried sample set on client k as Uqrry,k. Afterwards, the client updates its local training set $\mathcal { L } _ { k } ^ { r - 1 }$ by incorporating the newly labeled data samples, i.e.,

![](images/dd4824c522ce6c358d0302d9e5132cbc6b6e9a65f4a292e502fb8992d33aca5f.jpg)  
Figure 2: An overview of the proposed LeaDQ framework. The yellow block is the process of active data querying and the blue block is the process of federated model training. The dashed lines are conducted only when training the query policies.

$$
\mathcal { L } _ { k } ^ { r } = \mathcal { L } _ { k } ^ { r - 1 } \cup \mathcal { U } _ { q r y , k } ^ { r } .
$$

Given the high labeling cost, only a subset of unlabeled samples is queried for annotation. The labeling budget of each client is expressed as:

$$
| \mathcal { U } _ { q r y , k } ^ { r } | = N _ { q } .
$$

The data querying policy, parameterized with a query model $\phi$ , outputs the selection decisions $\mathbf { a } _ { k } ^ { r }$ for the given unlabeled dataset $\mathcal { U } _ { k } ^ { r }$ on client $k$ . It is worth noting that the query model may vary among clients and across rounds depending on the training stage and the current status of ML model.

Federated model training Based on the local labeled data $\mathcal { L } _ { k } ^ { r }$ , clients train the current global model $\theta ^ { r }$ to optimize the following objective:

$$
\operatorname* { m i n } _ { \theta ^ { r } \in \mathbb { R } ^ { M } } f ( \theta ^ { r } ) \triangleq \sum _ { k = 1 } ^ { K } \frac { | { \mathcal L } _ { k } ^ { r } | } { | { \mathcal L } ^ { r } | } F _ { k } ( \theta ^ { r } ; { \mathcal L } _ { k } ^ { r } ) ,
$$

with the local loss function

$$
F _ { k } ( \boldsymbol { \theta } ^ { r } ; \mathcal { L } _ { k } ^ { r } ) \triangleq \frac { 1 } { | \mathcal { L } _ { k } ^ { r } | } \sum _ { ( \mathbf { x } , y ) \in \mathcal { L } _ { k } ^ { r } } \ell ( \boldsymbol { \theta } ^ { r } ; \mathbf { x } , y ) .
$$

Here $\begin{array} { r } { \mathcal { L } ^ { r } ~ = ~ \bigcup _ { k = 1 } ^ { K } \mathcal { L } _ { k } ^ { r } } \end{array}$ denotes the labeled dataset on all clients. As t  training data contain privacy-sensitive information and cannot be exposed to the server, we adopt the classic FedAvg algorithm (McMahan et al. 2017) to optimize the ML model. Specifically, the model training process in one round can be divided into $T$ training iterations. At the beginning, the clients initialize the current model as3 $\theta ^ { r } ( 0 ) = \mathbf { \bar { \theta } } ^ { r - 1 }$ . In the $t$ -th training iteration, client $k$ performs local training for multiple steps to compute the accumulated gradients $\mathbf { g } _ { k } ^ { r } ( t )$ . The gradients are then uploaded to the server for aggregation. The global model is then updated using these gradients as:

$$
\theta ^ { r } ( t + 1 ) = \theta ^ { r } ( t ) - \eta \sum _ { k = 1 } ^ { K } \mathbf { g } _ { k } ^ { r } ( t ) , \forall t = 0 , \ldots , T - 1 .
$$

After $T$ training iterations, the global model is finalized as $\theta ^ { r } = \theta ^ { r } ( T )$ .

Through iterative data querying and model training, the ML model is optimized by clients. However, the conflict remains between the clients’ shortsighted view on its local data and the target data distribution of the global model. The queried data may be therefore redundant or not that beneficial for the global model training. To address this problem, it is intuitively helpful to provide the clients with a global view that captures the current training status and implies the desired data for facilitating the global model training. To clearly demonstrate this phenomenon, we present an example in the next section.

# 4 Motivating Example

In this section, we show empirically that a data querying strategy with a global view can benefit the FL training. In particular, we explore an FL setup in which clients collaborate to classify data samples from the MNIST dataset by training a LeNet model (LeCun et al. 1998). We assume there are $N _ { u } = 1 0$ unlabeled data samples arriving at each client in each round and only one data sample is selected for labeling. The classical coreset method (Sener and Savarese 2018) selects the samples such that the model learned over the selected samples is competitive for the remaining data points. For an unlabeled dataset $\mathcal { U }$ and given labeled dataset $\mathcal { L }$ , the data sample to be queried is selected as:

$$
\mathbf { x } ^ { * } = \arg \operatorname* { m a x } _ { \mathbf { x } \in \mathcal { U } } \operatorname* { m i n } _ { \mathbf { x } ^ { \prime } \in \mathcal { L } } \big \| \mathbf { x } - \mathbf { x } ^ { \prime } \big \| _ { 2 } .
$$

We first implement the coreset approach on each client locally, i.e., $\boldsymbol { \mathcal { U } } = \boldsymbol { \mathcal { U } } _ { k } ^ { r }$ , termed as Local Coreset. In Local Coreset, clients make selections based on the local available data samples. Besides, we consider an ideal approach termed as Global Coreset where the server can access to all clients’ data $\begin{array} { r } { \mathcal { U } = \sum _ { k \in [ K ] } \mathcal { U } _ { k } ^ { r } } \end{array}$ and determine one queried sample for each client. After querying the labels, clients iteratively train the ML model based on the labeled dataset. The experimental details are referred to Appendix B.1.

The evaluation in Fig. 3 focuses on the effectiveness of various data querying strategies and the distribution of queried data samples. It is observed that the Global Coreset strategy improves the model accuracy compared with Local Coreset, benefiting from the utilization of global information when selecting unlabeled data. This can be verified by the distribution divergence between the queried samples in each round and the global data distribution as shown in Fig. 3 (b). In particular, Global Coreset can select critical samples that coincide with the target data distribution, since it leverages the information from a global view.

Nevertheless, the global information in Global Coreset requires computing the sample-wise distance between each unlabeled sample and labeled sample. As the data samples are distributed on different clients, it is impractical to directly implement the Global Coreset, which severely violates the data privacy of clients. Thus, it becomes crucial to devise a strategy that leverages global insights without necessitating direct access to data across all clients. To this end, we propose the LeaDQ algorithm under a centralized training decentralized execution (CTDE) paradigm, which implicitly incorporates the global objective into the querying strategies while maintaining the decentralized execution as in FL. Consequently, LeaDQ achieves comparable performance (as shown in Fig. 3) to Global Corset while preserving the data privacy.

![](images/8b1556ea1b474814f6b0e0d43f6d9595e0126c708f1e65135cc97f5ea1d63d0a.jpg)  
Figure 3: Comparisons between different data querying strategies. Left: The model accuracy is computed on the test data. Right: The distribution divergence is computed as the Kullback–Leibler (KL) divergence between the distributions of selected unlabeled data on all clients and the target data.

# 5 LeaDQ: Learning the Data Querying Policies for Clients

Effective data querying strategies in FL should exhibit the ability to select unlabeled samples that are beneficial for global model training, while making the decisions in a decentralized fashion. Motivated by the capabilities of MARL algorithms in making individual decisions while optimizing a collaborative objective (Vinyals et al. 2017; Wang et al. 2021), we propose an MARL-based data querying algorithm named LeaDQ, short for Learn Data Querying. In the following, we first formulate the data querying problem in FL as a Markov decision process and elaborate on the details of system design in Section 5.1. Subsequently, we detail the process of decentralized data querying and centralized policy training in Sections 5.2 and 5.3, respectively.

# 5.1 Data Querying in FL as A Decentralized Decision-Making Process

The objective of data querying per client $k$ in $\mathrm { F L }$ aims to select samples from local unlabeled dataset $\mathcal { U } _ { k } ^ { r }$ to query their labels in round $r$ . By data querying and model training, these clients aim to collaboratively optimize the global model $\theta ^ { r }$ . However, each client only has access to its local data without the knowledge of either data distribution of other clients or the target global data distribution, i.e., the decision making process is based on partial observation in its nature.

To describe such a setup, we frame the data querying problem as a decentralized partially observable Markov decision process (Dec-POMDP) (Oliehoek and Amato 2016) denoted by a tuple $\langle S , A , P , R , O , K , \gamma \rangle$ . We view each querying round as a discrete timestep and each client as an agent in this process.

At each discrete timestep $r$ , a global state $\mathbf { s } ^ { r } \in \mathcal { S }$ reveals the current status of global model training. We assume the server owns a held-out dataset $\mathcal { D } _ { \mathrm { h e l d } }$ (Fang and Ye 2022; Huang, Ye, and Du 2022) and the global state is defined as the prediction confidence of the global model $\theta ^ { r }$ on it:

$$
\mathbf { s } ^ { r } = \left[ \operatorname* { m a x } _ { y \in \mathcal { V } } p ( y | \mathbf { x } ; \theta ^ { r } ) \right] _ { \mathbf { x } \in \mathcal { D } _ { \mathrm { h e l d } } } .
$$

We assume the held-out dataset follows the same distribution as that of the whole dataset, and thus the global state defined in (8) serves as a low-cost indicator to show the current model’s performance on the target dataset and can effectively reveal the current training status.

When a set of unlabeled data $\mathcal { U } _ { k } ^ { r }$ with the length of $N _ { u }$ arrives, each client $k$ gets the local observation $\mathbf { o } _ { k } ^ { r } \in \mathcal { O }$ of these data. Inspired by previous works (Wang and Shang 2014), we use the predictive logits output by the current model $\theta ^ { r }$ as the local observation, i.e.,

$$
\mathbf { o } _ { k } ^ { r } = [ l ( \mathbf { x } ; \theta ^ { r } ) ] _ { \mathbf { x } \in \mathcal { U } _ { k } ^ { r } } ,
$$

where $l ( \mathbf { x } ; \theta ^ { r } ) = p ( y | \mathbf { x } ; \theta ^ { r } ) , \forall y \in \mathcal { V }$ encapsulates the predicted logits. The local observation $\mathbf { o } _ { k } ^ { r }$ is a low-dimension representation of arrived samples in $\mathcal { U } _ { k } ^ { r }$ and also reflects the uncertainty of samples. It is intuitive that samples with higher levels of uncertainty would be more helpful to model updates when being included in the training dataset.

Based on its local observation $\mathbf { o } _ { k } ^ { r }$ , each client chooses actions $\mathbf { a } _ { k } ^ { r } = \{ 0 , 1 \} ^ { N _ { u } }$ to select samples for labeling. Here the binary decision $\mathbf { a } _ { k } ^ { r } [ i ] = 1$ denotes the case of the $i$ -th sample being chosen for querying and $\mathbf { a } _ { k } ^ { r } [ i ] = 0$ otherwise. The joint action is then given by $\mathbf { A } ^ { r } = [ \mathbf { a } _ { 1 } ^ { r } , \mathbf { a } _ { 2 } ^ { r } , \ldots , \mathbf { a } _ { K } ^ { r } ]$ . After querying unlabeled data, the clients cooperatively update the ML model based on the labeled dataset following the procedure elaborated in Section 3. Such evolution of the ML model’s states can be characterized by the transition function $P ( \mathbf { s } ^ { \prime } | \mathbf { s } , \mathbf { a } ) : \mathcal { S } \times \mathcal { A }  [ 0 , 1 ]$ .

Afterwards, the updated global model returns to clients a joint reward $R ^ { r }$ , which reveals the impact of selected actions on the ML model training. As the collaborative goal of clients is to improve the overall performance of the ML model $\theta ^ { r }$ , this reward should reveal its current training status. Thus we define the reward as the difference of model accuracy on the held-out data before and after updating, i.e.,

$$
R ^ { r } \equiv R ( \mathbf { s } ^ { r } , \mathbf { A } ^ { r } ) = A c c ( \theta ^ { r } ; \mathcal { D } _ { \mathrm { h e l d } } ) - A c c ( \theta ^ { r - 1 } ; \mathcal { D } _ { \mathrm { h e l d } } ) .
$$

The reward in (10) shapes a global view of the model training, which helps clients to make better local decisions of data querying. Formally, clients aim to find the optimal data querying policies that maximize the total discounted reward $\begin{array} { r } { \dot { R } _ { t o t } ^ { r } = \dot { \sum _ { j = r } } \gamma ^ { j } R ^ { j } } \end{array}$ with episode length $J$ and discount $\gamma$ .

By trial-and-error, the data query policies are directed to incorporate the information of the global model training, indicated by state $\mathbf { s } ^ { r }$ and reward $R ^ { r }$ , into the collaborative decisions, ultimately promoting the global model training. When executing the query policies, it is sufficient for clients to select local unlabeled data samples in a decentralized manner. In summary, such CTDE strategy bridges the gap between the local data querying and the objective of global model training, as detailed in the following subsections.

# 5.2 Decentralized Policy Execution for Querying Data Samples

In round $r$ , client $k$ has access to its local action-observation history $\tau _ { k } \in \mathcal T \equiv ( \mathcal O \times \mathcal A ) ^ { * }$ . Based on such history, the client makes a decision of selecting unlabeled data following its data query policy. Specifically, each client feeds the local observation $\mathbf { o } _ { k } ^ { r }$ of arrived unlabeled data into its policy

1: for each round $r = 1$ to $R$ do 2: // Active data querying 3: for each client $k = 1$ to $K$ do 4: Compute predictive logits as local observations $\mathbf { o } _ { k } ^ { r }$ $\{ \triangleright \cdot \operatorname { E q } . 9 \}$ 5: Compute local Q-values as $Q _ { k } ( \mathbf { a } , \mathbf { o } _ { k } ^ { r } ; \phi _ { k } ^ { r } )$ 6: Choose actions $\mathbf { a } _ { k } ^ { r }$ according to greedy policy $\{ \triangleright \mathrm { E q . ~ } 1 1 \}$ 7: Query the oracle for label $y _ { i }$ of data sample $\mathbf { x } _ { i }$ with $a _ { k } ^ { r } [ i ] ^ { \cdot } = 1$ 8: Update the labeled dataset as $\mathcal { L } _ { k } ^ { r }$ $\{ \triangleright  { \mathrm { E q . } } 2 \}$ 9: end for 10: //Federated model training 11: Initialize the global model as $\theta ^ { r } ( 0 ) = \theta ^ { r - 1 }$ 12: for each iteration $t = 0$ to $T - 1$ do 13: for each client $k = 1$ to $K$ do 14: Train the current model $\theta ^ { r } ( t )$ based on the local labeled dataset $\mathcal { L } _ { k } ^ { r }$ 15: Upload the gradients $\mathbf { g } _ { k } ^ { r } ( t )$ to the server 16: end for 17: The server updates the global model as $\theta ^ { r } ( t + 1 )$ $\{ \triangleright  { \mathrm { E q . } } 6 \}$ 18: end for 19: Update the global model as $\theta ^ { r } = \theta ^ { r } ( T )$ 20: end for

network, denoted by $\phi _ { k } ^ { r }$ , to output the local $\mathrm { Q }$ -value. Subsequently, it chooses the action that induces the highest local Q-value, which is given by:

$$
\mathbf { a } _ { k } ^ { r } = \arg \operatorname* { m a x } _ { \mathbf { a } : | \mathbf { a } | = N _ { q } } Q _ { k } ( \mathbf { a } , \mathbf { o } _ { k } ^ { r } ; \boldsymbol { \phi } _ { k } ^ { r } ) ,
$$

where $| { \bf a } | = N _ { q }$ implies $N _ { q }$ unlabeled samples are queried. Depending on the value of $\mathbf { a } _ { k } ^ { r }$ , each unlabeled sample is determined for querying or not. As such, the clients use the query policy to select data in a decentralized manner without violating the data privacy. The details of the proposed framework are summarized in Algorithm 1.

When querying data, the clients rely on their local policies, which shall guide them to select critical samples for the global model training. To integrate the global information into this process, we adapt the QMIX algorithm (Rashid et al. 2018) into the LeaDQ framework for training policies.

# 5.3 Centralized Policy Training for Learning Local Policies

By selecting unlabeled data, all clients contribute to the global model training and get a shared reward, serving as the global feedback for the joint decision of clients. However, it is difficult for clients to directly use the shared reward for local data selections. To link the global feedback and local decisions, we define the joint Q-value as a weighted sum of the local $\mathrm { Q }$ -values, which is given by (Rashid et al. 2018):

$$
Q _ { t o t } = \psi ( Q _ { 1 } , \cdot \cdot \cdot , Q _ { K } , { \bf s } ^ { r } ) .
$$

Here $\psi$ is a mixing network from a family of monotonic functions to model the complex interactions between collaborating clients. This mixing network helps to coordinate the clients’ local policies while accounting for the state of current ML model $\mathbf { s } ^ { r }$ .

![](images/ea7820e4f293265b69fc4239e248c5aa8e23c3f10cb7d3a44c777d5451ca109f.jpg)  
Figure 4: Model accuracy $( \% )$ in different data querying rounds.

We use a replay buffer $\boldsymbol { B }$ to store the transition tuple $\langle { \bf s } , { \bf o } , { \bf a } , R , { \bf s } ^ { \prime } , \bar { { \bf o } ^ { \prime } } \rangle$ , where the state $\mathbf { s } ^ { \prime }$ is observed after taking the action a in state s and receiving reward $R$ . During policy training, we iteratively sample batches of transitions from the replay buffer and update the policy network by minimizing the temporal difference (TD) loss:

$$
\begin{array} { r l } & { \mathcal { L } ( \{ \phi _ { k } \} , \psi ) } \\ & { \quad = \mathbb { E } _ { ( { \mathbf { s } } , { \mathbf { o } } , { \mathbf { a } } , R , { \mathbf { s } } ^ { \prime } , { \mathbf { o } } ^ { \prime } ) \sim \mathcal { B } } \left[ \left( y ^ { t o t } - Q _ { t o t } ( { \mathbf { s } } , { \mathbf { o } } , { \mathbf { a } } ; \{ \phi _ { k } \} , \psi ) \right) ^ { 2 } \right] , } \end{array}
$$

where $\begin{array} { r } { y ^ { t o t } = R + \gamma \operatorname* { m a x } _ { \mathbf { u } } Q _ { t o t } ( \mathbf { s } ^ { \prime } , \mathbf { o } ^ { \prime } , \mathbf { u } ; \{ \phi _ { k } \} , \psi ) } \end{array}$ with discount $\gamma$ . Indeed, minimizing the TD loss in (13) optimizes the local policies $\left\{ \phi _ { k } \right\}$ on clients towards a global goal, i.e., increasing the reward $R$ . As such, by following their learned policies, clients can select the beneficial unlabeled data for global model training. The pseudo code of policy training is deferred to Appendix A.1.

# 6 Experiments

# 6.1 Setup

We simulate an FL system with one server and $K = 1 0$ clients. We evaluate the algorithms on two image classification tasks, i.e., SVHN (Netzer et al. 2011) and CIFAR100 (Krizhevsky and Hinton 2009), and one text classification task, i.e., 20Newsgroup (Lang 1995). We train a convolutional neural network (CNN) model with four convolutional layers (Kim et al. 2023) on the SVHN dataset, a ResNet-18 (He et al. 2016) on the CIFAR-100 dataset, and a DistilBERT (Sanh et al. 2020) model on the 20Newsgroup dataset. To simulate the non-IID setting, we allocate the training data to clients according to the Dirichlet distribution with concentration parameter $\alpha = 0 . 5$ (Li et al. 2022). In each round, $N _ { u } = 1 0$ unlabeled data samples arrive at each client independently and each client selects $N _ { q } = 1$ data sample for label querying. The details of datasets are summarized in Table 1. More implementation details can be found in Appendix B.2. Extended results can be found in Appendix C. In all tables, the best performances are highlighted in bold, and the second-best ones are underlined.

Table 1: Summary of datasets.   

<html><body><table><tr><td>Type</td><td>Dataset</td><td>Non-IID Type</td></tr><tr><td>Text</td><td>20Newsgroup</td><td>Distribution-based</td></tr><tr><td rowspan="2">Image</td><td>SVHN</td><td>Distribution-based Quantity-based</td></tr><tr><td>CIFAR-100</td><td>Distribution-based</td></tr></table></body></html>

Baselines We compare the proposed algorithm with the following baselines. First, we adapt several representative AL methods to FL, including: (i) Uncertainty (Wang and Shang 2014): Each client selects the samples with lowest prediction confidence for label querying; (ii) Coreset (Sener and Savarese 2018): Each client selects the samples such that the model learned over these samples is competitive for the remaining unlabeled data points. Second, two state-of-theart federated AL algorithms are also compared, including: (iii) LoGo (Kim et al. 2023): Each client selects samples using a clustering-based sampling strategy; (iv) KAFAL (Cao et al. 2023): Each client selects samples with the highest discrepancy between the specialized knowledge by the local model on clients and the global model.

# 6.2 Main Results

We show the model accuracy in different querying rounds on several datasets in Fig. 4. We observe that the proposed LeaDQ algorithm achieves the best accuracy in all the querying rounds. On the SVHN dataset, the naive local querying strategies (i.e., Uncertainty and Coreset) attain weaker performance than other approaches, showcasing the importance of designing specific strategies for FL. On the CIFAR100 dataset, the baselines have similar performance, while LeaDQ surpasses the best-performance baseline at the later training stage by around $6 \%$ , implying its advantages in querying critical samples for global model training.

![](images/6ba5a15e164ec714262e9c43e42fcfa06c69f708063e6ae7325435b52c06d471.jpg)  
Figure 5: Visualization of the data samples in SVHN dataset $R = 1 0 0$ ). Left: Label distribution of local data on a random client, target global data, and queried data by LeaDQ. Right: T-SNE plots (Van der Maaten and Hinton 2008) of feature distributions of the global data and queried data by LeaDQ.

Table 2: Model accuracy $( \% )$ on SVHN with different values of arrived samples $N _ { u }$ and queried samples $N _ { q }$ .   

<html><body><table><tr><td>Nq/Nu</td><td>1/30</td><td>1/20 1/10</td><td>2/10 3/10</td></tr><tr><td>Uncertainty</td><td>44.35</td><td>50.64 59.28</td><td>73.31 79.30</td></tr><tr><td>Coreset</td><td>39.96</td><td>45.07 62.72</td><td>75.97 79.66</td></tr><tr><td>LoGo</td><td>45.45</td><td>49.94 59.66</td><td>76.02 79.86</td></tr><tr><td>KAFAL</td><td>41.88</td><td>49.30 64.67</td><td>76.32 80.30</td></tr><tr><td>LeaDQ</td><td>45.68</td><td>52.31 65.02</td><td>76.28 80.37</td></tr></table></body></html>

Visualization of queried data samples To visualize the querying strategies provided by LeaDQ, we show the queried samples distributions in Fig. 5. According to Fig. 5 (left), the queried unlabeled data samples have similar distributions with the target global data distribution, despite the fact that the local data follows a different distribution with the global data. Besides, the t-SNE plots in Fig. 5 (right) also demonstrate that LeaDQ is able to select unlabeled data that have a similar feature distribution with the target global data.

# 6.3 Results With Diverse Scenarios

Results with various arrived and queried data samples We show the model accuracy with different numbers of arrived samples $N _ { u }$ and queried samples $N _ { q }$ per round in Table 2. We observe from the results that as the ratio of queried data samples $N _ { q } / N _ { u }$ increases, the model achieves higher accuracy with all the data querying strategies. This is because leveraging more labeled data samples on clients effectively improves the model’s performance. In most cases, LeaDQ outperforms the baselines due to its advantages in identifying critical samples. Meanwhile, when the number of queried samples is larger $( N _ { q } > 1 )$ ), LeaDQ and KAFAL have competitive performances, surpassing other strategies.

Results with different setups of data heterogeneity In addition, we simulate two types of data heterogeneity, i.e., the distribution-based skew and quantity skew (Li et al. 2022). The results are illustrated in Tables 3 and 4, respectively, in which we show the model accuracy after $R = 5 0 0$ rounds when applying different data querying strategies. We vary the data heterogeneity among clients by adjusting the concentration parameter $\alpha$ and the number of classes on each client $C$ . When the data tend to be more heterogeneous, the model performance degrades as the training becomes harder. We find that LeaDQ achieves higher performance compared with the baselines in all cases, showing its effectiveness in finding beneficial samples for global model training. However, the FAL baselines may even fail in identifying informative samples and thus lead to suboptimal performance. Overall, these results demonstrate the robustness and adaptability of the proposed LeaDQ algorithm.

Table 3: Model accuracy $( \% )$ on SVHN with different degrees of non-IID under distribution-based skew. $\alpha$ is the value of concentration parameter in Dirichlet distribution, and a larger $\alpha$ indicates that the data distribution across clients is closer to IID.   

<html><body><table><tr><td></td><td>α=0.1</td><td>α = 0.5</td><td>α = 1.0</td></tr><tr><td>Uncertainty</td><td>48.09</td><td>59.28</td><td>63.54</td></tr><tr><td>Coreset</td><td>49.77</td><td>62.72</td><td>65.04</td></tr><tr><td>LoGo</td><td>43.70</td><td>59.66</td><td>59.97</td></tr><tr><td>KAFAL</td><td>52.14</td><td>64.67</td><td>66.48</td></tr><tr><td>LeaDQ</td><td>53.98</td><td>65.02</td><td>66.50</td></tr></table></body></html>

Table 4: Model accuracy $( \% )$ on SVHN with different degrees of non-IID under quantity skew. $C$ is the number of classes on each client, and a larger $C$ indicates that the data distributions across clients are closer to IID.   

<html><body><table><tr><td></td><td>C=2</td><td>C=3</td><td>C=4</td></tr><tr><td>Uncertainty</td><td>62.26</td><td>67.73</td><td>67.10</td></tr><tr><td>Coreset</td><td>64.15</td><td>67.77</td><td>68.43</td></tr><tr><td>LoGo</td><td>59.69</td><td>63.15</td><td>67.07</td></tr><tr><td>KAFAL</td><td>20.19</td><td>20.32</td><td>21.75</td></tr><tr><td>LeaDQ</td><td>65.34</td><td>68.19</td><td>69.06</td></tr></table></body></html>

# 7 Conclusions

In this paper, we investigate the data querying problem in stream-based federated learning, highlighting the conflict between local data access and global querying objective. We introduce a novel MARL-based algorithm named LeaDQ, which learns local data query policies on distributed clients. LeaDQ promotes cooperative objectives among clients, ultimately leading to improved machine learning model training. Extensive experiments validate the superiority of LeaDQ over state-of-the-art baselines, demonstrating its effectiveness in querying critical data samples and enhancing the quality of the ML model. We note that LeaDQ provides a framework for data querying in FL. Future research could explore the incorporation of more advanced algorithms beyond FedAvg and QMIX to enhance performance further.