# PFedCS: A Personalized Federated Learning Method for Enhancing Collaboration among Similar Classifiers

Siyuan $\mathbf { W } \mathbf { u } ^ { 1 }$ , Yongzhe Jia1, Bowen Liu1, Haolong Xiang2\*, Xiaolong $\mathbf { X } \mathbf { u } ^ { 2 }$ , Wanchun Dou1\*

1State Key Laboratory for Novel Software Technology, School of Computer Science, Nanjing University, China 2School of Software, Nanjing University of Information Science and Technology, China sywu, jiayz, liubw @smail.nju.edu.cn, hlxiang, xlxu @nuist.edu.cn, douwc@nju.edu.cn

# Abstract

Personalized federated learning (PFL) has recently gained significant attention for its capability to address the poor convergence performance on highly heterogeneous data and the lack of personalized solutions of traditional federated learning (FL). Existing mainstream approaches either perform personalized aggregation based on a specific model architecture to leverage global knowledge or achieve personalization by exploiting client similarities. However, the former overlooks the discrepancies in client data distributions by indiscriminately aggregating all clients, while the latter lacks finegrained collaboration of classifiers relevant to local tasks. In view of this challenge, we propose a Personalized Federated learning method for Enhancing Collaboration among Similar Classifiers (PFedCS), which aims at improving the client’s accuracy on local tasks. Concretely, it is achieved by leveraging awareness of the client classifier similarities to address the above problems. By iteratively measuring the distance of the classifier parameters between clients and clustering with each client as a cluster center, the central server adaptively identifies the collaborating clients with similar data distributions. In addition, a distance-constrained aggregation method is designed to generate customized collaborative classifiers to guide local training. As a result, extensive experimental evaluations conducted on various datasets demonstrate that our method achieves state-of-the-art performance.

# Introduction

Federated learning (FL) is a distributed computing paradigm that enables collaborative training across multiple distributed devices without requiring the upload of raw data, which has experienced remarkable growth in various domains such as healthcare (Wu et al. 2022; Guan et al. 2024), multimedia (Zhang, Liu, and Liu 2023; Li et al. 2024), Industrial Internet of Things (IIoT) (Boobalan et al. 2022; Ding et al. 2022) and so on. Unfortunately, traditional FL methods, such as FedAvg (McMahan et al. 2017), which only train a single global model, exhibit suboptimal endof-training performance on severely non-independent and non-identically distributed (Non-IID) data (Liao et al. 2023;

Huang et al. 2021), making it challenging to meet the personalized requirements of each client. Taking the development of tumor image detection in hospitals as an example of the application of FL (Jiang, Wang, and Dou 2022), it is evident that users of different demographic data, influenced by subtle variations in regions and eating habits, are likely to exhibit varying proportions of disease severity. Certain cases may predominantly occur within specific groups. In such scenarios, providing more personalized diagnostic predictions for each region becomes meaningful and necessary.

To tackle statistical heterogeneity and meet the personalized needs of clients in FL, personalized Federated Learning (PFL) methods have been proposed. Unlike traditional FL approaches that seek a globally optimal model with strong generalization by training and collaborating across distributed clients, PFL aims to train a set of personalized models to mitigate the impact of Non-IID across different clients. Recent studies in PFL fall into two main categories based on their underlying motivations: (1) Architecturedriven methods that aim to achieve personalization through customized model designs tailored to each client, including FedPer (Arivazhagan et al. 2019) and FedGH (Yi et al. 2023), (2) Similarity-based methods that focus on achieving personalization by modeling client relationships, including FedAMP (Huang et al. 2021) and FeSEM (Long et al. 2023).

The PFL methods in category (1) face challenges in determining the optimal architectural design. Furthermore, at the private parameter level, there is no direct benefit from other clients with similar data distribution, potentially overlooking valuable information from them. Meanwhile, the PFL methods in category (2) still have shortcomings. On the one hand, due to the modeling of user relationships, the personalization models of clients are easily influenced by other clients, making these methods sensitive to the poor data quality of clients. On the other hand, these methods primarily focus on client-level model aggregation, lacking fine-grained knowledge sharing relevant to local tasks. In addition, this process may introduce additional computational and communication costs for clients (Ghosh et al. 2022), particularly detrimental to edge devices with constrained resources.

Recent studies have revealed that the degradation in FL predominantly stems from classifier biases in clients’ local models induced by Non-IID data (Li et al. 2023; Luo et al.

2021). In particular, it has been observed that the classifier layer exhibits higher biases compared to other layers (Luo et al. 2021). These classifier biases engender a vicious cycle, in which biased classifiers and misaligned features across clients reinforce each other (Zhou, Zhang, and Tsang 2023). Fig.1 illustrates a toy example showing the distance matrix of the classifier (a fully connected layer) parameters across all clients. The distribution of Non-IID labels is generated by pathological partitioning, where every four clients are assigned the same two classes of labels. It can be observed that clients with similar data distributions tend to exhibit lower pair-wise parameter distance (e.g., $\ell _ { 2 }$ -norm distance) among clients’ classifiers in the Non-IID data setting. This motivates us to exploit a more efficient mechanism to select the collaborators calibrated to each client (the paired clients corresponding to the dark region in Fig.1), without the need for prior knowledge of the number of groups, as required by most existing clustered federated learning methods (Ding et al. 2022; Ghosh et al. 2022; Long et al. 2023).

In order to address the aforementioned problems and enhance collaborative learning among clients in PFL, we introduce a novel PFL method called PFedCS. It adaptively performs clustering with each client as a cluster center based on the distances of classifier parameters between clients, thereby identifying appropriate collaborating clients with similar data distributions. Since the lower layers in deep neural networks (DNN) focus more on universal information compared to higher layers (Yu et al. 2018; Oh, Kim, and Yun 2022), a distance-constrained aggregate weight is designed for each client to generate customized classifiers with the help of collaborative clients to guide local training in PFedCS, while the lower layers of all clients are aggregated to extract global features. Subsequently, the customized classifier, after undergoing several rounds of local fine-tuning, serves as the teacher model to guide the training of the local classifier. Although the introduction of customized classifiers requires a slight computational cost for local fine-tuning, the search for collaborating clients and the aggregation process both occur on the central server, which has significantly greater computational resources than the clients. As a result, PFedCS does not significantly increase the computational and communication cost on the client side. In this paper, we focus on the scenario of label distribution shift in the Non-IID data setting. Extensive experimental results on various datasets have strongly validated the effectiveness of our proposed method. Our main contributions are summarized as follows:

• We observe that clients with similar data distributions in Non-IID scenarios tend to exhibit a smaller pair-wise parameter distance of the classifier parameters. Based on the insight and observation, the pair-wise distances of the classifier parameters can be computed to be aware of the similarity of their data distributions. This motivates us to encourage collaboration among clients with similar classifiers to avoid the negative effect of Non-IID data.

• We propose a novel PFL method called PFedCS that adaptively selects collaborators for each client based on the distances of the classifier parameter. Moreover, we design a distance-constrained aggregation method to generate customized classifiers to guide local training.

• We conduct extensive experiments over various datasets to validate the effectiveness of PFedCS, which outperforms twelve state-of-the-art methods by up to $4 . 0 7 \%$ in test accuracy. In addition, we empirically show that PFedCS exhibits superior performance with different numbers of clients.

![](images/b87f18ceef07e55e26e79bb014a9bd33f7537e473cdfd53133c9041091dc539a.jpg)  
Figure 1: Motivation of PFedCS. It visualizes the normalized classifier distance matrices $D ^ { 1 }$ and $D ^ { 1 0 }$ on CIFAR-10 across clients in different communication rounds.

# Related Work

# Federated Learning

Federated learning was first proposed by McMahan et al. (McMahan et al. 2017), also known as FedAvg. This approach aims to train a generalized global model by leveraging the local models of all clients. In the current landscape where privacy concerns are receiving increasing attention, the advantages of FL have become prominent. Consequently, a substantial body of research efforts have been dedicated to addressing the limitations of Federated Learning from diverse perspectives, including system heterogeneity (Xia et al. 2022; Wang et al. 2024), statistical heterogeneity (Qi et al. 2023; Mendieta et al. 2022), communication efficiency (Cheng et al. 2023; Wang et al. 2023), as well as security and privacy issues (Wu et al. 2021, 2023). Recently, numerous studies have demonstrated that although FedAvg performs well in the setting of independent and identically distributed (IID) data, its performance is considerably suboptimal in Non-IID data scenarios, and may even be inferior to the separate training performed by each client locally (Li et al. 2022; Jiang and Lin 2023). To address this problem, previous studies have attempted to enhance the robustness of the global model (Li et al. 2020; Huang et al. 2023). However, a single global model is challenging in meeting the personalized requirements of each client and fails to systematically address the challenges posed by data heterogeneity.

# Personalized Federated Learning

Compared to the strategy of training a robust global model, Personalized Federated Learning emphasizes training multiple personalized models. From the perspective of learning personalized models, there are two mainstream methods in

PFL (Tan et al. 2022a; Huang et al. 2024): Personalized aggregation based on customized model architectures and personalization based on client similarity modeling.

Architecture-driven PFL methods aim to achieve personalization by providing custom model designs for each client. A representative category of methods involves parameter decoupling, such as FedPer (Arivazhagan et al. 2019), FedRep(Collins et al. 2021), FedBABU (Oh, Kim, and Yun 2022), FedPCL (Tan et al. 2022c) and FedGH (Yi et al. 2023), which decouple the local private model parameters from the global FL model parameters. The private parameters are retained on the client side. This type of method provides flexibility for each client’s architectural design. However, at the private parameter level, there is no direct benefit from other clients with similar data distributions, potentially overlooking valuable information from other clients. Inspired by them, we have also adopted a design of model parameter decoupling, with the distinction that we apply different personalized knowledge-sharing mechanisms to different components of the parameters.

Similarity-based PFL methods achieve personalization by modeling relationships among clients, in which similar clients collaborate to enhance the learning of similar models (T Dinh, Tran, and Nguyen 2020; Li et al. 2021). For example, FedAMP (Huang et al. 2021) designs an attentionbased mechanism that enhances collaboration among FL clients with similar data distributions. FedFomo (Zhang et al. 2021) updates the personalized models by a weighted combination of all clients’ models based on the loss similarities. These methods focus mainly on pair-wise client relationships. There are also works that consider client relationships at the group level (Sattler, Mu¨ller, and Samek 2020; Long et al. 2023; Vahidian et al. 2023). However, determining the appropriate number of groups remains a challenging problem in the absence of prior knowledge about the number of distinct categories. Furthermore, these approaches directly aggregate the global model and lack fine-grained collaboration with local task-specific classifiers. In this work, we resort to the server to perform collaborative client selection and fine-grained collaboration without incurring additional communication costs.

# Problem Settings

Following typical federated learning (McMahan et al. 2017; Li et al. 2020), where $K$ is the number of clients in the federated learning system (indexed by $k$ ), $\mathbb { C } ^ { t } \subseteq \mathbb { C }$ is the subset of selected clients in round $t \in [ 1 , T ]$ and $T$ is the total federated rounds. In addition, each client $k$ has a prieisNtohne-InIuDmdbaetrasoeft $\mathcal { D } _ { k } = \{ \boldsymbol { x } _ { i } ^ { k } , \boldsymbol { y } _ { i } ^ { k } \} _ { i = 1 } ^ { N _ { k } }$ awsshuerme $k \in \mathbb { C }$ haensde $N _ { k }$ $\mathcal { D } _ { k }$   
private datasets share the same feature space, but have different sample spaces. The entire dataset across all clients is denoted as $\begin{array} { r } { \mathcal { D } = \bigcup _ { k = 1 } ^ { K } \{ \mathcal { D } _ { k } \} } \end{array}$ . For traditional $\mathrm { F L }$ , the objective of the whole federated system is to obtain an optimal global model $w _ { * }$ as follows:

$$
w _ { * } = \operatorname* { m i n } _ { w } \sum _ { k = 1 } ^ { K } p _ { k } F _ { k } ( w )
$$

where $p _ { k } \geq 0$ is the weight of client $k$ . Typically, in FedAvg (McMahan et al. 2017), $p _ { k }$ is set to $| \mathcal { D } _ { k } | \big / | \mathcal { D } |$ and $\textstyle \sum _ { k = 1 } ^ { K } p _ { k } \ = \ 1$ . The local objective $F _ { k } ( \cdot )$ is often defined as the expected error over local dataset $\mathcal { D } _ { k }$ :

$$
F _ { k } ( \cdot ) = \mathbb { E } _ { \left( \pmb { x } _ { i } ^ { k } , y _ { i } ^ { k } \right) \sim \mathcal { D } _ { k } } \left[ \mathcal { L } _ { k } ( w ; \pmb { x } _ { i } ^ { k } , y _ { i } ^ { k } ) \right]
$$

where $\mathcal { L } _ { k }$ is the loss function for the $k$ -th client. For the local models, we can decompose them into two modules: A feature extractor $f$ and a linear classifier $g$ , i.e., $\boldsymbol { w } = \{ w _ { f } , w _ { g } \}$ . For a given sample $( x , y )$ , the feature extractor $f : \mathcal { X } \to \mathcal { Z }$ , parameterized by $\boldsymbol { w } _ { f }$ , encodes the input sample into a $d .$ - dimension feature vector $z = f ( x , w _ { f } ) \in \mathbb { R } ^ { d }$ . The linear classifier $g : { \mathcal { Z } } \to \mathbb { R } ^ { C }$ , parameterized by $w _ { g }$ , aggregates the information of the feature vector to produce a probability distribution $p ^ { w _ { g } } = g ( z , w _ { g } )$ as the prediction result.

Due to the Non-IID data distribution across clients, the optimal global model obtained through training does not guarantee the best generalization performance across all clients. In this scenario, PFL allows for the coexistence of multiple models, enabling each client $k$ to learn an optimal personalized model by aligning it with its local objectives:

$$
\{ w _ { * } ^ { ( 1 ) } , \dots , w _ { * } ^ { ( K ) } \} = \operatorname* { m i n } _ { w _ { * } ^ { ( 1 ) } , \dots , w _ { * } ^ { ( K ) } } \sum _ { k = 1 } ^ { K } p _ { k } F _ { k } ( w ^ { ( k ) } )
$$

The objective of the PFL system is to minimize the overall empirical loss by considering collaborative learning and personalization to obtain the optimal local models $\{ w _ { * } ^ { ( \bar { k } ) } \}$ .

# Methodology

Solution Overview. Before starting, we introduce the overview of the PFedCS method, which applies data distribution awareness and classifier collaboration to the federated scenario in detail. The overall framework of the proposed PFedCS is illustrated in Fig.2. The yellow and gray parts represent the global aggregation executed on the server and the local training on the clients, respectively. We divide the training process into 2 sub-stages. Assuming the stage 1 consists of $\beta$ training rounds, in the $t \in [ 1 , \bar { \operatorname* { m i n } } ( \beta , \bar { T } ) ]$ round, each client $\boldsymbol { k } \in \mathbb { C } ^ { t }$ uploads the complete model $w ^ { ( k ) }$ , while in the $t \in [ \mathrm { m i n } ( \beta , T ) , T ]$ round of stage 2, clients retain their classifier w(gk) locally. We outline the workflow of the federated training process in stage 1 as follows:

1. Each client k uploads the local model wt(k)1 $w _ { t - 1 } ^ { ( k ) } ~ =$ $\left\{ w _ { f , t - 1 } ^ { ( k ) } , w _ { g , t - 1 } ^ { ( k ) } \right\}$ to the central server in round $t$ . 2. The server first measures the similarity of the data distribution by calculating the distance matrix $D ^ { t }$ . Then the server determines the collaborative clients $\mathit { \overline { { \mathbb { C } } } } _ { k , c } ^ { t }$ for each client with a two-component GMM model and a dynamic threshold $\tau _ { t , k }$ . Finally, the server generates a set of customized classifiers $\boldsymbol { v } _ { g , t } ^ { k }$ with the help of $\mathit { \overline { { \mathbb { C } } } } _ { k , c } ^ { t }$ for each client $k$ and the same feature extractor $\boldsymbol { w } _ { f , t }$ for all clients ( orange part in Fig. 2 ). 3. Active clients download the global feature extractor wf,t, customized classifier $\boldsymbol { v } _ { g , t } ^ { k }$ from the server and replace the local feature extractor with $w _ { f , t } ( \mathrm { s t e p } \ 1 \$ in Fig. 2).

Classifier Distance matrix Collaborator set {𝑣\$,+# } Step 1 Step 2 Client K Server 𝑤\$,#() + Pair-wise D3),) D3),/ GMM ℂ\$#) : {2,···, 𝐾} Agg 8 Download Model Q Distance D3 D⋯4,4 D3/⋯,/ Eq.(5) ℂ\$# : {1, 2,···} ℂ\$#\* :. {1,···, 𝐾} Eq.(6) Replace ℒ-. Decouple Feature extractor Average Share𝑤d!a,#cross clients clSaesnsidficeurstomclizendts 𝑣\$(,&#) 𝑤!,# w/𝑤!,# 𝑤!,#() (&) Eq.(9) 𝑣̅/(,&# ) Step 3 Step 4 0 𝑤#(()) 𝑤!,#, \$,# ()) 𝑤#((\*) 𝑤!,#, 𝑣\$(,\*#) 𝑤#((&) 𝑤!,#, 𝑣\$(,&#) local model Upload Classifier Linear 。 𝑤(&) Guide ? Trainable params 𝑤\$(&) ℒ.0 Eq.(11) Frozen params Feature Client 1 Client 2 Client K o (&) (&) 𝑤 Extractor \$,#

4. Each client $k$ freezes $\boldsymbol { w } _ { f , t }$ and fine-tunes $\boldsymbol { v } _ { g , t } ^ { k }$ over local data (step 2 in Fig. 2). After that, each client $k$ unfreezes the local feature extractor and updates $w _ { t - 1 } ^ { ( k ) }$ under the guidance of new $\overline { { v } } _ { g , t } ^ { k }$ (step 3 in Fig. 2).

5. Each client $k$ uploads the latest local model $w _ { t } ^ { ( k ) }$ (step 4 in Fig. 2). The next round starts.

Different from stage 1, during stage 2, clients do not upload the classifier layer of their models to the server. Instead, they retain and train the classifiers locally. In other words, when the current round $t \geq \beta$ , PFedCS degrades into FedPer(Arivazhagan et al. 2019) and only the blue steps in the left part of Fig. 2 are executed on the server. Next, we delve into in-depth discussions of stage 1 in PFedCS.

# Adaptive Collaborator Selection Mechanism

During stage 1 of PFedCS, after receiving the latest local models, the server begins to identify collaborators for each client. At first, the server calculates pair-wise distances based on the classifier layer parameters of the client models. We assume that the classifier w(gk) of client k is a linear transformation with weight φk = [φk,1, . . . , φk,C] and bias, followed by Normalization and Softmax. In each round $t$ , we employ a square matrix $\begin{array} { r l } { D ^ { t } } & { { } = } \end{array}$ $\bigl [ \bigl ( D _ { 1 , * } ^ { t } \bigr ) ^ { T } , \ldots , \bigl ( D _ { K , * } ^ { t } \bigr ) ^ { T } \bigr ] ^ { T } \in \mathbb { R } ^ { K \times K }$ to record the distance between a pair of clients. For any clients $\mathbf { \chi } _ { i }$ and $j$ , the distance of classifier parameters between them is defined as $\begin{array} { r } { D _ { i , j } ^ { t } ~ = ~ \frac { \| \varphi _ { i , c } - \varphi _ { j , c } \| _ { 2 } ^ { 2 } } { \operatorname* { m a x } ( D _ { i , * } ^ { t } ) } } \end{array}$ ∥φi,c−φtj,c∥ . Since the weights φk are typically initialized randomly, as the number of training rounds increases, this statistical information becomes more accurate. In PFedCS, the server builds $D ^ { t }$ in every round. Based on $D ^ { t }$ , each client $k$ determines collaborative clients based on its distance from all other clients, i.e., Dtk = Dtk, \ Dtk,k.

The current challenge lies in deterfmining the collaborative clients. To select similar clients adaptively, we propose the utilization of clustering algorithms to group $\widetilde { D _ { k } ^ { t } }$ .

The server computes a two-component Gaussian Mixture Model (GMM) on $\widetilde { D _ { k } ^ { t } }$ for each client $k$ . Then the set of clients $\mathbb { C } ^ { t } - \{ k \}$ fgrouped into two subsets: $\mathbb { C } _ { k , c } ^ { t }$ (candidate clients) and $\mathbb { C } _ { k , n } ^ { t }$ (non-candidate clients). Note that the group of clients with a lower mean distance serves as candidate clients. In this way, clients exhibiting similar data distributions are identified.

# Distance-constrained Classifier Aggregation

At the beginning of FL training, it is difficult to identify clients with similar data distributions based on parameter distance due to the rapid fluctuations. Therefore, a better way is to aggregate the classifier parameters with more clients during early training. As training progresses, further refinement within candidate clients $\mathbb { C } _ { k , c } ^ { t }$ becomes necessary.

To implement an effective refinement mechanism, we introduce a dynamic threshold, with its value negatively correlated with the number of training rounds. In round $t$ , the server calculates a distance threshold for each client $k$ :

$$
\tau _ { t , k } = \arg ( \widetilde { D _ { k } ^ { t } } ) + \frac { t } { \beta } \times ( \operatorname* { m i n } ( \widetilde { D _ { k } ^ { t } } ) - \arg ( \widetilde { D _ { k } ^ { t } } ) )
$$

where the $\arg ( \cdot )$ and $\operatorname* { m i n } ( { \cdot } )$ function indicate the average and minimum value of the corresponding pair-wise distances within the set, respectively. As $t$ increases, the threshold $\tau _ { t , k }$ decreases, leading to a reduced number of clients selected for collaboration. Based on $\tau _ { t , k }$ , the server selects a subset of candidate clients from which we form the real collaborative client set:

$$
\overline { { \mathbb { C } } } _ { k , c } ^ { t } = \left\{ i \in \mathbb { C } _ { k , c } ^ { t } | D _ { k , i } ^ { t } \leq \tau _ { t , k } \right\}
$$

The server then leverages similar clients $\mathit { \overline { { \mathbb { C } } } } _ { k , c } ^ { t }$ to generate a client-level customized classifier model for client $k$ , which will be broadcast to client $k$ for download in round $t$ :

$$
v _ { g , t } ^ { ( k ) } = \sum p _ { i , t } w _ { g , t - 1 } ^ { ( i ) } , \quad i \in \left\{ \overline { { \mathbb { C } } } _ { k , c } ^ { t } \cup k \right\}
$$

where v(g,kt denotes the customized classifier downloaded by client $k$ in round $t$ . A smaller parameter distance between two clients indicates higher similarity in their data distributions, consequently enabling more effective mutual assistance during the aggregation. Therefore, it is necessary to increase the clients’ weight with a smaller distance during the aggregation. Formally, we propose the distance-constrained classifier aggregation (DCA) and the weight is defined as:

$$
p _ { i , t } = \lambda \frac { D _ { m a x } - D _ { k , i } ^ { t } } { | \overline { { \mathbb { C } } } _ { k , c } ^ { t } | \cdot ( D _ { m a x } - D _ { a v g } ) } + ( 1 - \lambda ) \frac { N _ { i } } { \sum _ { j \in \overline { { \mathbb { C } } } _ { k , c } ^ { t } } N _ { j } } .
$$

$$
D _ { m a x } = \operatorname* { m a x } ( D _ { k , j } ^ { t } ) , D _ { a v g } = \arg ( D _ { k , j } ^ { t } ) , j \in \overline { { \mathbb { C } } } _ { k , c } ^ { t }
$$

where $\lambda$ controls the importance of both similarity and the local data size when aggregating personalized classifiers. Finally, the server aggregates the received feature extractor as FedAvg, which helps all clients collaborate on facilitating the extraction of more generalizable features.

# Customized Classifiers Guide Training

After obtaining the customized classifier $v _ { g , t }$ , a naive approach for clients to take advantage of its capabilities involves directly replacing their local classifiers $w _ { g , t - 1 }$ with the new classifier $v _ { g , t }$ . However, the inference objectives of the server-generated customized classifier and the local classifier may be inconsistent, leading to oscillations in loss during local training, resulting in degraded performance.

Inspired by FedRod (Chen and Chao 2022), as a global   
model with stronger generalization capabilities can achieve   
a higher level of personalization after local adaptation, we   
propose to fine-tune the customized classifier ${ { v } _ { g } }$ , familiar  
izing it with the local task and enabling personalization. In $t$ $w _ { f , t }$ $\bar { v } _ { g , t } ^ { ( \bar { k } ) }$   
the client k first updates the local feature extractor w(f,kt) 1   
with $\boldsymbol { w } _ { f , t }$ . Then, the client $k$ freezes $w _ { f , t - 1 } ^ { ( k ) }$ and fine-tunes   
$v _ { g , t } ^ { ( k ) }$ for $\rho$ epochs via Stochastic Gradient D−escent (SGD) :

$$
\begin{array} { r } { \boldsymbol { v } _ { g , t } ^ { ( k ) }  \boldsymbol { v } _ { g , t } ^ { ( k ) } - \eta _ { v } \nabla _ { v _ { g , t } ^ { ( k ) } } \mathcal { L } _ { C E } ( \boldsymbol { p } ^ { v } ; \boldsymbol { y } ) , } \end{array}
$$

$$
p ^ { v } = g ( f ( x , w _ { f , t - 1 } ^ { ( k ) } , v _ { g , t } ^ { ( k ) } ) , ( x , y ) \sim \mathcal { D } _ { k }
$$

where $\eta _ { v }$ is the learning rate for fine-tuning customized classifier and $\mathcal { L } _ { C E }$ is the cross-entropy loss between the output logits $p ^ { v }$ and the true class label $y$ . The objective of this step is to promote alignment between the extracted features and classifier vectors while enabling the biased classifiers to absorb prior knowledge from the local class distributions. After local fine-tuning, the customized classifier is updated to vgk,t, which is used to guide the training of the local model, maximizing the benefits of the personalized classifier while minimizing disruptions to the local training. Specially, the client $k$ unfreezes $\boldsymbol { w } _ { f , t - 1 } ^ { ( k ) }$ and freezes $\overline { { v } } _ { g , t } ^ { ( k ) }$ , then performs SGD to update wt(k)1 = $w _ { t - 1 } ^ { ( k ) } = \left\{ w _ { f , t - 1 } ^ { ( k ) } , w _ { g , t - 1 } ^ { ( k ) } \right\}$ :

$$
w _ { t - 1 } ^ { ( k ) } \gets w _ { t - 1 } ^ { ( k ) } - \eta _ { w } \nabla _ { w _ { t - 1 } ^ { ( k ) } } \mathcal { L } _ { E N } \left( w _ { t - 1 } ^ { ( k ) } , v _ { g , t } ^ { ( k ) } ; \mathcal { D } _ { k } \right)
$$

where $\eta _ { w }$ is the learning rate for local training. Our objective is to improve the performance on the local task, with the guidance provided by the personalized classifier $\overline { { v } } _ { g , t } ^ { k }$ . Following (Jin et al. 2023), we propose to use vgk,t as a teacher to guide the local classifier $w _ { g , t } ^ { ( k ) }$ in assimilating the ensemble knowledge within the integrated classifier as follows:

$$
\mathcal { L } _ { E N } : = \mathcal { L } _ { C E } ( p ^ { w } , y ) + \mathcal { D } _ { K L } ( p ^ { v } | | p ^ { w } )
$$

$$
p ^ { w } = g ( f ( x , w _ { f , t } ^ { ( k ) } ) , w _ { g , t } ^ { ( k ) } ) , ( x , y ) \sim \mathcal { D } _ { k }
$$

where the term $\mathcal { D } _ { K L } ( p ^ { v } | | p ^ { w } )$ represents the KullbackLeibler (KL) divergence between the output logits of $w _ { g }$ and $v _ { g }$ , whose objective is to guide the local model in assimilating the ensemble knowledge in the teacher model.

# Experiments

This section presents the experiment setups, comparison with SOTA methods and ablation studies.

# Experiment Setup

Datasets and Models. Our experiments are conducted on three public datasets: CIFAR-10 (Krizhevsky, Hinton et al. 2009), CIFAR-100 (Krizhevsky, Hinton et al. 2009), and Tiny-ImageNet (Chrabaszcz, Loshchilov, and Hutter 2017). All datasets are randomly divided into the training and test sets following a 3:1 split. We use the data partitioning methods in (Li et al. 2022) to simulate different label skews. Specifically, we try two types of Non-IID partition: (1) Pathological Non-IID (McMahan et al. 2017): we sample $C$ classes for CIFAR-10/CIFAR-100/Tiny-ImageNet from 10/100/200 classes for each client, with disjoint data and different numbers of data samples. (2) Practical NonIID (Zhang et al. 2023): We sample a proportion of samples of class $j$ to client $k$ with Dirichlet distribution, i.e., $p _ { j , k } \sim D i r ( \alpha )$ and smaller $\alpha$ leads to greater class imbalance. Following (McMahan et al. 2017; Dai et al. 2023), we consider a 4-layer CNN that consists of two convolutional layers and two fully connected layers for CIFAR-10 and CIFAR-100, and ResNet-18 (He et al. 2016) for TinyImageNet, respectively. For all model decoupling methods, we use a linear layer as the classifier, while considering the remaining parts as the feature extractor. For each setting, clients’ local training and test datasets are under the same distribution.

Compared methods. We compare our PFedCS with twelve state-of-the-art FL algorithms, including two traditional FL algorithms: The leading FedAvg (McMahan et al. 2017) and FedProx (Li et al. 2020), and ten PFL algorithms: Per-FedAvg (Fallah, Mokhtari, and Ozdaglar 2020), FedPer (Arivazhagan et al. 2019), FedBABU (Oh, Kim, and Yun 2022), FedAMP (Huang et al. 2021), FedFomo (Zhang et al. 2021), FedProto (Tan et al. 2022b), FedRod (Chen and Chao 2022), FedGH (Yi et al. 2023), ClusterFL (Sattler, Mu¨ller, and Samek 2020) and FeSEM (Long et al. 2023). Note that ClusterFL and FeSEM are specially designed for clustered FL. Following FedAMP (Huang et al. 2021), we report the mean top-1 accuracy by averaging the test accuracies over all clients with 3 trials.

Table 1: Comparison results in the pathological Non-IID setting on CIFAR-10, CIFAR-100, and Tiny-Imagenet.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="3">CIFAR-10</td><td colspan="3">CIFAR-100</td><td colspan="3">Tiny-Imagenet</td><td rowspan="2">Average</td></tr><tr><td>C=2</td><td>C=3</td><td>C=4</td><td>C=10</td><td>C=15</td><td>C=20</td><td>C=20</td><td>C= 30</td><td>C= 40</td></tr><tr><td>FedAvg</td><td>54.29</td><td>55.06</td><td>58.82</td><td>23.36</td><td>23.96</td><td>24.33</td><td>16.89</td><td>17.23</td><td>19.07</td><td>32.56</td></tr><tr><td>FedProx</td><td>54.25</td><td>55.00</td><td>58.67</td><td>23.39</td><td>24.05</td><td>24.20</td><td>16.74</td><td>17.53</td><td>18.73</td><td>32.51</td></tr><tr><td>Per-FedAvg</td><td>88.82</td><td>84.44</td><td>81.29</td><td>58.97</td><td>52.32</td><td>46.11</td><td>32.84</td><td>27.66</td><td>25.65</td><td>55.34</td></tr><tr><td>FedPer</td><td>90.06</td><td>84.66</td><td>80.75</td><td>60.85</td><td>53.29</td><td>46.38</td><td>41.21</td><td>35.29</td><td>32.51</td><td>58.33</td></tr><tr><td>FedBABU</td><td>88.37</td><td>83.53</td><td>80.39</td><td>60.55</td><td>53.90</td><td>46.72</td><td>42.86</td><td>37.39</td><td>36.04</td><td>58.86</td></tr><tr><td>FedAMP</td><td>89.25</td><td>83.18</td><td>78.13</td><td>60.80</td><td>52.10</td><td>44.61</td><td>39.54</td><td>32.37</td><td>28.97</td><td>56.55</td></tr><tr><td>FedFomo</td><td>90.48</td><td>85.15</td><td>81.80</td><td>60.55</td><td>54.13</td><td>48.35</td><td>36.51</td><td>31.05</td><td>30.14</td><td>57.57</td></tr><tr><td>FedProto</td><td>88.17</td><td>83.06</td><td>77.18</td><td>57.41</td><td>54.07</td><td>46.54</td><td>38.20</td><td>30.93</td><td>27.25</td><td>55.87</td></tr><tr><td>FedRod</td><td>90.24</td><td>85.12</td><td>81.12</td><td>57.73</td><td>51.73</td><td>46.44</td><td>38.87</td><td>33.71</td><td>31.70</td><td>57.41</td></tr><tr><td>FedGH</td><td>89.25</td><td>82.92</td><td>78.62</td><td>61.64</td><td>54.00</td><td>46.22</td><td>38.23</td><td>31.61</td><td>27.21</td><td>56.63</td></tr><tr><td>ClusterFL</td><td>89.72</td><td>83.96</td><td>80.94</td><td>62.01</td><td>56.06</td><td>47.85</td><td>38.07</td><td>37.36</td><td>36.19</td><td>59.13</td></tr><tr><td>FeSEM</td><td>89.29</td><td>83.14</td><td>78.10</td><td>60.91</td><td>52.21</td><td>44.73</td><td>39.32</td><td>32.58</td><td>28.75</td><td>56.56</td></tr><tr><td>PFedCS</td><td>90.60</td><td>85.49</td><td>82.06</td><td>63.29</td><td>56.15</td><td>50.00</td><td>46.93</td><td>41.36</td><td>38.92</td><td>61.64</td></tr></table></body></html>

Table 2: Comparison results in the practical Non-IID setting on CIFAR-10, CIFAR-100, and Tiny-Imagenet.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="3">CIFAR-10</td><td colspan="3">一 CIFAR-100</td><td colspan="3">一 Tiny-Imagenet</td><td rowspan="2">Average</td></tr><tr><td>α=0.01</td><td>α = 0.05</td><td>α =0.1</td><td>α = 0.01</td><td>α = 0.05</td><td>α=0.1</td><td>α = 0.01</td><td>α = 0.05</td><td>α=0.1</td></tr><tr><td>FedAvg</td><td>39.53</td><td>44.69</td><td>50.17</td><td>22.30</td><td>25.85</td><td>27.37</td><td>13.86</td><td>14.89</td><td>15.23</td><td>28.21</td></tr><tr><td>FedProx</td><td>39.58</td><td>44.69</td><td>50.15</td><td>22.22</td><td>25.77</td><td>27.40</td><td>13.99</td><td>14.42</td><td>14.50</td><td>28.08</td></tr><tr><td>Per-FedAvg</td><td>96.60</td><td>92.04</td><td>88.61</td><td>65.75</td><td>53.54</td><td>46.97</td><td>46.88</td><td>36.56</td><td>29.67</td><td>61.85</td></tr><tr><td>FedPer</td><td>97.17</td><td>93.52</td><td>91.05</td><td>68.03</td><td>56.49</td><td>47.44</td><td>54.90</td><td>45.16</td><td>39.47</td><td>65.91</td></tr><tr><td>FedBABU</td><td>96.59</td><td>92.70</td><td>90.00</td><td>65.92</td><td>56.30</td><td>48.69</td><td>57.47</td><td>46.44</td><td>39.06</td><td>65.91</td></tr><tr><td>FedAMP</td><td>97.28</td><td>93.76</td><td>90.55</td><td>69.70</td><td>56.78</td><td>47.38</td><td>52.76</td><td>42.62</td><td>34.54</td><td>65.04</td></tr><tr><td>FedFomo</td><td>97.27</td><td>93.20</td><td>89.86</td><td>68.32</td><td>54.13</td><td>44.80</td><td>49.78</td><td>38.08</td><td>31.92</td><td>63.04</td></tr><tr><td>FedProto</td><td>96.29</td><td>92.84</td><td>87.98</td><td>66.00</td><td>54.34</td><td>45.67</td><td>49.63</td><td>39.82</td><td>32.62</td><td>62.80</td></tr><tr><td>FedRod</td><td>97.14</td><td>93.21</td><td>90.89</td><td>64.74</td><td>56.02</td><td>48.52</td><td>49.58</td><td>43.95</td><td>36.84</td><td>64.54</td></tr><tr><td>FedGH</td><td>96.11</td><td>83.92</td><td>89.32</td><td>69.47</td><td>56.60</td><td>47.78</td><td>54.57</td><td>40.68</td><td>30.22</td><td>63.19</td></tr><tr><td>ClusterFL</td><td>95.22</td><td>88.44</td><td>83.45</td><td>56.59</td><td>47.86</td><td>41.77</td><td>45.72</td><td>36.38</td><td>31.20</td><td>58.51</td></tr><tr><td>FeSEM</td><td>97.28</td><td>93.73</td><td>90.54</td><td>69.84</td><td>56.68</td><td>47.39</td><td>52.37</td><td>42.34</td><td>33.90</td><td>64.90</td></tr><tr><td>PFedCS</td><td>97.30</td><td>93.58</td><td>91.56</td><td>70.20</td><td>59.39</td><td>50.60</td><td>60.26</td><td>49.55</td><td>43.22</td><td>68.41</td></tr></table></body></html>

Implementation details. We run the baselines in the settings suggested in the original papers. We adopt SGD optimizer and set the batch size to 100. Regarding the local learning rate $\eta _ { w }$ , we set $\eta _ { w } = 0 . 0 0 5$ for 4-layer CNN and $\eta _ { w } = 0 . 1$ for ResNet-18. Note that PFedCS introduces an additional learning rate $\eta _ { v }$ for fine-tuning customized classifier, we set $\eta _ { v } ~ = ~ \eta _ { w }$ by default. We run 200 federated rounds and set the number of local epochs to 5 to guarantee convergence. The number of clients is set to 20 and the client joining ratio is set to 1 by default. Unless specifically stated, the settings are shared for all experiments.

# Comparisons with State-of-the-arts

Pathological Skew Settings. We report the test accuracy results in the pathological Non-IID data setting in Table 1. For the CIFAR-10, CIFAR-100 and Tiny-ImageNet datasets, we sample $C = \{ 2 , 3 , 4 \} / \{ 1 0 , 1 5 , 2 0 \} / \{ 2 0 , 3 0 , 4 0 \}$ classes for each client, respectively. The optimal results are indicated in bold and the sub-optimal results are underlined. It is obvious that PFedCS performs best on three datasets and outperforms the best baseline by an average of $2 . 5 1 \%$ . Among the baselines, the traditional FL algorithms (FedAvg and FedProx) perform poorly because they only train a single global model for all clients, failing to meet the personalized requirements of different clients. The superiority of PFedCS can be attributed to the ability to adaptively identify clients with similar data distributions through the distance between their classifier parameters. It leverages the ensemble knowledge from collaborative clients to guide local model learning, thereby achieving superior performance across all datasets.

Practical Skew Settings. We also report the test accuracy results under the Dirichlet Non-IID data setting in Table 2, with $\alpha$ values of $\{ 0 . 0 1 , 0 . 0 5 , 0 . 1 \}$ for all three datasets. Compared to baselines, PFedCS obtains the best performance in all conditions except that on CIFAR-10 with $\alpha = 0 . 0 5$ , which is slightly inferior to FedAMP. We attribute the superior performance of FedAMP to simpler datasets. Moreover, PFedCS exhibits outstanding performance on the Tiny-ImageNet dataset, surpassing the sub-optimal method

(a) Pathological Non-IID with $C = 2 0$ (b) Practical Non-IID with $\alpha = 0 . 1$ 35404550Test accuracy (%) 50 1 GH 40 FedGH Fed FeSEM 35 FeSEM PFedCS PFedCS 10 20 30 50 100 10 20 30 50 100 Number of Clients Number of  Clients

by $2 . 5 0 \%$ in average accuracy across three different settings of $\alpha$ . This result demonstrates the effectiveness of PFedCS in adapting to Non-IID scenarios of varying complexity.

Varying Numbers of Clients. Following MOON (Li, He, and Song 2021), we split the CIFAR-100 dataset into $\{ 1 0 , 2 0 , 3 0 , 5 0 , 1 0 0 \}$ sub-datasets to form the corresponding number of clients to present the effectiveness of PFedCS with different numbers of clients. The results of PFedCS and the other ten PFL baselines are shown in Fig.3. Unfortunately, as the number of clients increases, the average number of samples assigned to each client decreases, which leads to a performance drop across all methods. It can be found that PFedCS consistently outperforms other baselines with different numbers of clients, demonstrating the adaptability and scalability of PFedCS in heterogeneous data scenarios.

Table 3: The test accuracy $( \% )$ of PFedCS and its collaborator selection mechanism variants on CIFAR-100.   

<html><body><table><tr><td>Methods/Non-IID</td><td>C= 10</td><td>C= 15</td><td>C = 20</td></tr><tr><td>FedPer</td><td>60.85</td><td>53.29</td><td>46.38</td></tr><tr><td>SelectNone</td><td>60.51</td><td>53.35</td><td>46.60</td></tr><tr><td>Select All</td><td>61.57</td><td>54.00</td><td>48.28</td></tr><tr><td>Random Selection</td><td>61.41</td><td>54.34</td><td>48.30</td></tr><tr><td>Adaptive Selection (Ours)</td><td>63.29</td><td>56.15</td><td>50.00</td></tr></table></body></html>

# Ablation Studies

Effectiveness of Adaptive Selection Mechanism. Table 3 compares our adaptive selection mechanism (ASM) with other selection mechanisms on CIFAR-100 in the pathological Non-IID setting. To be specific, we introduce three collaborator selection mechanisms: (1) Select None: Each client forms a separate group, with only the feature extractor uploaded and aggregated on the server, thus degrading to FedPer (Arivazhagan et al. 2019); (2) Select All: Each client selects all other clients as collaborators; (3) Random Selection: Each client randomly selects a fixed number of clients as a group and we set it to 3. The results show that applying our adaptive selection mechanism improves performance across Non-IID scenarios by an average of $1 . 7 4 \%$ . This indicates collaboration among clients with similar classifier parameters can yield significant performance benefits. The results demonstrate that better model performance can

be achieved by measuring the distance between client classifier parameters to select clients with similar distributions for collaboration.   

<html><body><table><tr><td>Dataset</td><td>KM</td><td>HIER</td><td>FIX</td><td>PFedCS</td></tr><tr><td>CIFAR-10</td><td>90.50</td><td>90.51</td><td>90.48</td><td>90.60</td></tr><tr><td>CIFAR-100</td><td>62.65</td><td>62.67</td><td>61.49</td><td>63.29</td></tr><tr><td>Tiny-ImageNet</td><td>46.28</td><td>46.68</td><td>40.23</td><td>46.93</td></tr></table></body></html>

Table 4: The test accuracy $( \% )$ of PFedCS and its clustering method variants in the pathological Non-IID setting.

Different Grouping Strategies in ASM. To explore the effects of clustering methods in dividing $\mathbb { C } _ { k , c } ^ { t }$ from $\mathbb { C } ^ { t }$ , we replace the GMM with K-Means and hierarchical clustering, denoted by “KM” and “HIER”. In addition, we set a fixed threshold (set to 0.5) to replace the clustering algorithm, denoted by “FIX”. As Table 4 shows, the performance differences when using different clustering methods are small, but the results obtained with the fixed threshold perform poorly, which decrease by $6 . 7 0 \%$ on Tiny-ImageNet. It’s clear that using clustering methods in PFedCS is effective and PFedCS shows robustness to the clustering algorithm chosen.

Table 5: The test accuracy $( \% )$ on CIFAR-100 in the practical setting $\langle \alpha = 0 . 0 5 \rangle$ ) with different fine-tuning epochs $\rho$ .   

<html><body><table><tr><td></td><td>p=0</td><td>p=1</td><td>p=2</td><td>p=3</td><td>p=5</td><td>p=10</td></tr><tr><td>Acc.</td><td>53.58</td><td>59.39</td><td>59.51</td><td>59.41</td><td>59.46</td><td>59.16</td></tr></table></body></html>

Effects of Fine-tuning Epochs $\rho$ . Here, we study the effects of $\rho$ on test accuracy. The results of PFedCS by varying $\rho$ are shown in Table 5. It can be seen that the accuracy first increases from $\rho = 0$ to $\rho = 1$ , then the accuracy maintains stable from $\rho = 1$ to $\rho = 5$ but decreases from $\rho = 5$ to $\rho = 1 0$ . A larger $\rho$ leads to better global knowledge absorption, but it also incurs higher computational costs and may result in the catastrophic forgetting problem (Shenaj et al. 2023; Huang, Ye, and $\mathrm { D u } 2 0 2 2 )$ in neural networks. Therefore, we adopt $\rho = 1$ by default to achieve a trade-off between computational cost and performance.

# Conclusion

In this work, we propose a novel PFL method dubbed PFedCS, to address the limitation of existing FL methods in lacking fine-grained collaboration among clients with similar classifiers in data heterogeneous scenarios. The key insight is to leverage the distances between classifier parameters of clients to perceive the similarities in data distributions and promote collaboration among similar clients. Through iterative distance measurement, collaborator selection, and distance-constrained aggregation, PFedCS can adaptively identify clients with similar data distributions and generate customized classifiers to guide local training. Extensive experiments on various datasets demonstrate that PFedCS achieves state-of-the-art performance.

# Acknowledgments

This research is supported by the National Natural Science Foundation of China No.92267104 and Dou Wanchun Expert Workstation of Yunnan Province No.202105AF150013. The authors wish to acknowledge Dr. Fei Dai, Professor of Southwest Forestry University, for his help in interpreting the significance of the results of this study.