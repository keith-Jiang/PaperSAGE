# Co-Dream: Collaborative Dream Synthesis over Decentralized Models

Abhishek Singh1\*, Gauri Gupta1\*, Yichuan Shi1, Alex Dang1, Ritvik Kapila2, Sheshank Shankar3, Mohammed Ehab1, Ramesh Raskar1

1Massachusetts Institute of Technology 2Amazon 3Tesla AI abhi24@mit.edu

# Abstract

Federated Learning (FL) has pioneered the idea of ”share wisdom not raw data” to enable collaborative learning over decentralized data. FL achieves this goal by averaging model parameters instead of centralizing data. However, representing ”wisdom” in the form of model parameters has its own limitations including the requirement for uniform model architectures across clients and communication overhead proportional to model size.

In this work we introduce Co-Dream a framework for representing ”wisdom” in data space instead of model parameters. Here, clients collaboratively optimize random inputs based on their locally trained models and aggregate gradients of their inputs. Our proposed approach overcomes the aforementioned limitations and comes with additional benefits such as adaptive optimization and interpretable representation of knowledge. We empirically demonstrate the effectiveness of Co-Dream and compare its performance with existing techniques.

Code — https://mitmedialab.github.io/codream.github.io/

# Introduction

Machine learning (ML) model training is often hindered by the fragmented nature of data ownership. Federated Learning (FL) (McMahan et al. 2023) addresses this problem by aggregating clients’ models centrally instead of their data. While FL does not give any privacy guarantee, it reduces privacy concerns by (1) sharing clients’ models instead of their raw data and (2) using a linear operation (weighted average) to aggregate models that is amenable to secure aggregation techniques.

However, FL requires all clients to agree on the same model architecture. If the model has a large number of parameters, it may not be supported by all participating device hence reducing the number of participants. Some recent knowledge-distillation (KD) (Mora et al. 2022) techniques allow clients to share knowledge while allowing heterogeneous models. However, these KD algorithms depart from the model averaging paradigm, and hence incompatible with secure aggregation.

We propose a novel framework to solve this problem by aggregating collaboratively optimized representations of data (which we call dreams) instead of parameters. We show that dreams capture the knowledge embedded within local models and also facilitate the aggregation of local knowledge. Our key idea is to apply federated optimization on randomly initialized samples to extract knowledge from the client’s local models trained on their original dataset. The goal of optimizing dreams is to enable KD, rather than generate realistic synthetic data.

The key benefits of our approach are: (1) Flexibility: Our proposed technique, Co-Dream, collaboratively optimizes dreams to aggregate knowledge from the client’s local models. By sharing dreams in the data space rather than model parameters, our method is model-agnostic. (2) Scalability: Furthermore, communication does not depend on the model parameter size, alleviating scalability concerns. (3) Privacy: Just like FedAvg (McMahan et al. 2017), Co-Dream does not come with privacy guarantee but enhances privacy in two ways: Firstly, clients share dreams’ updates, never raw data. Secondly, the linearity of the aggregation algorithm allows clients to securely aggregate their dreams without revealing their individual updates to the server.

Our framework comprises three stages: knowledge extraction , knowledge aggregation , and knowledge acquisition . We test Co-Dream by establishing the feasibility of Co-Dream as a way for clients to synthesize samples collaboratively and learn predictive models, validating Co-Dream as an alternative to FL. We empirically validate our framework by benchmarking with existing algorithms and conducting ablation studies across various design choices.

# Preliminaries

Federated Learning (FL) minimizes the expected risk minθ $\mathbb { E } _ { \mathcal { D } \sim p ( \mathcal { D } ) } \ell ( \mathcal { D } , \theta )$ where $\theta$ is the model parameters, $\mathcal { D }$ is a tuple of samples $( X \in { \mathcal { X } } , Y \in { \mathcal { Y } } )$ of labeled data in supervised learning in the data space $\boldsymbol { \mathcal { X } } \subset \mathbb { R } ^ { d }$ and $\mathcal { y } \subset \mathbb { R }$ , and $\ell$ is some risk function such as mean square error or cross-entropy (Koneˇcn\`y et al. 2016; McMahan et al. 2023). Without directly accessing the true distribution, FL optimizes the empirical risk instead given by:

$$
\operatorname* { m i n } _ { \theta } \sum _ { k \in K } \frac { 1 } { | D _ { k } | } \ell ( D _ { k } , \theta ) ,
$$

![](images/33fd2a20fb93be8748526f1724048be1bef05f4db11d483f4861354aafb491fc.jpg)  
Figure 1: Overview of the Co-Dream pipeline comprising three stages: (1) Knowledge Extraction— each client generates dreams, representing the extracted knowledge from their local models (teacher). Starting with random noise images and frozen teacher models, clients optimize to reduce entropy on the output distribution while regularizing the batch norm and adaptive loss. The clients share their local updates of dreams and logits with the server. (2) Knowledge Aggregation—server aggregates dreams and soft labels from clients to construct a $\mathtt { C o - D r e a m }$ dataset. (3) Knowledge Acquisition— clients update their local models through two-stage training (i) on jointly optimized $c o$ -dreams with knowledge distillation (where clients act as students) and (ii) local dataset with cross-entropy loss.

The dataset $\mathcal { D }$ is distributed among $K$ clients, with each client $k$ holding a portion $\mathcal { D } _ { k }$ , such that $\mathcal { D } = \cup _ { k \in K } \mathcal { D } _ { k }$ . The server broadcasts the global model $\theta ^ { r }$ to all clients, who then locally optimize it for M rounds to obtain θkr+1 arg $\mathrm { m i n } _ { \theta } \ell ( \mathcal { D } _ { k } , \theta ^ { r } )$ . Each client sends its updated model $\theta _ { k } ^ { r + 1 }$ or the difference $\theta _ { k } ^ { r + 1 } - \theta _ { k } ^ { r }$ (the pseudo-gradient) back to the server, which aggregates these updates and sends the new global model back to the clients.

Knowledge Distillation facilitates the transfer of knowledge from a teacher model $( f ( \theta _ { T } ) )$ to a student model $( f ( \theta _ { S } ) )$ by incorporating an additional regularization term into the student’s training objective (Buciluˇa, Caruana, and NiculescuMizil 2006; Hinton et al. 2015). This regularization term (usually computed with Kullback-Leibler (KL) divergence $\mathsf { K L } ( f ( \bar { \theta } _ { T } , \mathcal { D } ) \bar { | } | f ( \theta _ { S } , \mathcal { D } ) ) \rangle$ encourages the student’s output distribution to match the teacher’s outputs.

DeepDream for Knowledge Extraction (Mordvintsev, Olah, and Tyka 2015) first showed that features learned in deep learning models could be extracted using gradientbased optimization in the feature space. Randomly initialized features are optimized to identify patterns that maximize a given activation layer. Regularization such as TV-norm and $\ell _ { 1 }$ -norm has been shown to improve the quality of the resulting images. Starting with a randomly initialized input ${ \hat { x } } \sim { \mathcal { N } } ( 0 , \ I )$ , label $y$ , and pre-trained model $f _ { \theta }$ , the optimization objective is

$$
\operatorname* { m i n } _ { \hat { x } } { \mathsf C } \mathsf E ( f _ { \theta } ( \hat { x } ) , y ) \ + \ \mathcal R ( \hat { x } ) ,
$$

where CE is cross-entropy and $\mathcal { R }$ is some regularization. DeepInversion (Yin et al. 2020) showed that the knowledge distillation could be further improved by matching batch normalization statistics with the training data at every layer.

# Related Work

Generative modeling techniques either pool locally generated data on the server (Song et al. 2022; Goetz and Tewari 2020) or use FedAvg with generative models (Rasouli, Sun, and Rajagopal 2020; Xin et al. 2020). Like FL, FedAvg over generative models is also not model agnostic. While we share the idea of generative data modeling, we do not expose individual clients’ updates or models directly to the server.

Knowledge Distillation in FL is an alternative to FedAvg that aims to facilitate knowledge sharing among clients that cannot acquire this knowledge individually (Chang et al. 2019; Lin et al. 2020; Afonin and Karimireddy 2022; Chen and Chao 2021). However, applying KD in FL is challenging because the student and teacher models need to access the same data, which is difficult in FL settings.

Data-free Knowledge Distillation algorithms address this challenge by employing a generative model to generate synthetic samples as substitutes for the original data (Zhang et al. 2022a,b; Zhu, Hong, and Zhou 2021). These data-free KD approaches are not amenable to secure aggregation and must use the same architecture for the generative model.

Overall, these existing approaches lack active client collaboration in the knowledge synthesis process. Clients share their local models or locally generated data with the server without contributing to knowledge synthesis. We believe that collaborative synthesis is crucial for secure aggregation and bridging the gap between KD and FL. Our approach Co-Dream enables clients to synthesize dreams collaboratively while remaining compatible with secure aggregation techniques and being model agnostic.

# CoDream

![](images/03756e876d3f031a04e1172d0e8b6339036d4cd1f60f81567d70d4631d366fcf.jpg)  
Figure 2: Comparing aggregation framework in FL and Co-Dream. In FL, the server aggregates the gradients of model parameters, whereas, in Co-Dream, aggregation happens in the gradients of the data space, called dreams $( \hat { x } )$ , allowing for different model architectures. Here $K$ is the number of clients and $l , \tilde { l }$ are loss functions given in Eq 1 and Eq 3.

Our approach Co-Dream consists of three key stages.

In the knowledge extraction stage, each client extracts useful data representations, referred to as “dreams”, from their locally trained models. Starting with random noise images $\hat { x }$ , clients $k \in K$ optimize these images using objective $\tilde { \ell }$ to facilitate knowledge sharing from their local models with parameters $\theta _ { k }$ (Section ). Since this is a gradient-based optimization of the input dreams, we exploit the linearity of gradients to enable knowledge aggregation from all the clients:

$$
\nabla _ { \hat { x } } \left( \mathbb { E } _ { k \sim K } [ \tilde { \ell } ( \hat { x } , \ \theta _ { k } ) ] \right) = \mathbb { E } _ { k \sim K } \left[ \nabla _ { \hat { x } } ( \tilde { \ell } ( \hat { x } , \ \theta _ { k } ) ) \right]
$$

In the knowledge aggregation stage, the clients now jointly optimize these random noised images by aggregating the gradients from the local optimizations (Section ). Unlike FedAvg, our aggregation occurs in the input data space over these dreams, making our approach compatible with heterogeneous client models.

Finally, in the knowledge acquisition stage, these collaboratively optimized images, or dreams, are then used for updating the server and clients without ever sharing the raw data or models. Specifically, clients act as students and train their models on the global dreams (Section ). Figure 1 gives an overview of the Co-Dream pipeline for each round. We now discuss these stages in more detail.

# Local dreaming for knowledge extraction

First, clients perform local dreaming, a model-inversion approach to extract information from their trained models. We use DeepDream (Mordvintsev, Olah, and Tyka 2015) and DeepInversion (Yin et al. 2020) approaches that enable datafree knowledge extraction from the pre-trained models. However, these are not directly applicable to a federated setting because the client models are continuously evolving, as they learn from their own data as well as other clients. A given client should synthesize only those dreams over which they are highly confident. As the client models evolve, their confidence in model predictions also changes over time. A direct consequence of this non-stationarity is that it is unclear how the label $y$ should be chosen in Eq 2. In DeepInversion, the teacher uniformly samples $y$ from its own label distribution because the teacher has the full dataset. However, in the federated setting, data is distributed across multiple clients with heterogeneous data distributions.

To keep track of a given client’s confidence, we take a simple approach of treating the entropy of the output distribution as a proxy for the teachers’ confidence. We adjust Eq 2 so that the teacher synthesizes dreams without any classification loss by instead minimizing the entropy (denoted by $\mathcal { H }$ ) on the output distribution. Each client (teacher) starts with a batch of representations sampled from a standard Gaussian $( \hat { x } = \mathcal { N } ( 0 , 1 ) )$ , and optimizes dreams using Eq 3. Formally, we optimize the following objective for synthesizing dreams:

$$
\operatorname* { m i n } _ { \hat { x } } \Big \{ \tilde { \ell } ( \hat { x } , \ \theta ) \ : = \ : \mathcal { H } \left( f _ { \theta } ( \hat { x } ) \right) + \mathcal { R } _ { b n } ( \hat { x } ) + \mathcal { R } _ { a d v } ( \hat { x } ) \Big \}
$$

where $\mathcal { H }$ is the entropy for the output predictions, $\mathcal { R } _ { b n }$ is the feature regularization loss and $\mathcal { R } _ { a d v }$ is a studentteacher adversarial loss. To improve the dreams image quality, we enforce feature similarities at all levels by minimizing the distance between the feature map statistics for dreams and training distribution, which is stored in the batch normalization layers. Hence, $\begin{array} { r l } { \mathcal { R } _ { b n } ( \hat { x } ) } & { { } = } \end{array}$ $\begin{array} { r } { \sum _ { l } | | \boldsymbol { \mu } _ { f e a t } ^ { l } - \boldsymbol { \mu } _ { b n } ^ { l } | | + | | \boldsymbol { \sigma } _ { f e a t } ^ { l } - \dot { \boldsymbol { \sigma } _ { b n } ^ { l } } | | } \end{array}$ . Further, to increase the diversity in generated dreams, we add an adversarial loss to encourage the synthesized images to cause studentteacher disagreement. $\mathcal { R } _ { a d v }$ penalizes similarities in image generation based on the Jensen-Shannon divergence between the teacher and student distribution, $\mathcal { R } _ { a d v } ( \hat { x } ) =$ $- J S D ( f _ { t } ( \hat { x } ) | | f _ { s } ( \hat { x } ) )$ , where the client model is the teacher and the server model is the student model. To do this adaptive teaching in a federated setting, the server shares the gradient $\nabla _ { \boldsymbol { \hat { x } } } \bar { f } _ { s } ( \boldsymbol { \hat { x } } )$ with the clients for local adaptive extraction. The clients then locally calculate $\nabla _ { \boldsymbol { \hat { x } } } \tilde { \ell } ( \boldsymbol { \hat { x } } , \ \boldsymbol { \theta } _ { k } )$ which is then aggregated at the server for knowledge aggregation in $\mathrm { E q } 4$ Thus, $\mathcal { R } _ { a d v }$ helps extract knowledge from the clients that the clients know and the server does not know.

Unlike generative models that generate data with objective to resemble the real data, the goal of optimizing dreams is to perform knowledge distillation. Therefore, as shown in Figure 6, dreams do not resemble real images.

# Collaborative dreaming for knowledge aggregation

Since the data is siloed and lies across multiple clients, we want to extract the collective knowledge from the distributed system. While FedAvg aggregates gradients of the model updates from clients, it assumes the same model architecture across clients and thus is not model-agnostic.

We propose a novel mechanism for aggregating the knowledge by collaboratively optimizing dreams across different clients. Instead of each client independently synthesizing dreams using Eq 3, they now collaboratively optimize them by taking the expectation over each client’s local loss w.r.t.

the same $\begin{array} { r } { \hat { x } \colon \operatorname* { m i n } _ { \hat { x } } \mathbb { E } _ { k \in K } \left[ \tilde { \ell } ( \hat { x } , \ \theta _ { k } ) \right] . } \end{array}$ This empirical risk can be minimized by computing the local loss at each client. Therefore, the update rule for $\hat { x }$ can be written as:

$$
\hat { x }  \hat { x } - \nabla _ { \hat { x } } \sum _ { k \in K } \frac { 1 } { | \mathscr { D } _ { k } | } \tilde { \ell } ( \hat { x } , \theta _ { k } )
$$

Using the linearity of gradients, we can write it as

$$
\hat { x }  \hat { x } - \sum _ { k \in K } \frac { 1 } { | \mathscr { D } _ { k } | } \nabla _ { \hat { x } } \tilde { \ell } ( \hat { x } , \ \theta _ { k } )
$$

Clients compute gradients locally on a shared input and send them to the server, which aggregates the gradients and returns the updated input. This approach, like distributed-SGD, optimizes in the data space rather than the model parameter space. As a result, Co-Dream is model-agnostic (Fig 2) and compatible with existing cryptographic aggregation methods, since only the aggregated output is revealed, not individual gradients.

We experimentally demonstrate that collaborative optimization indeed embeds the knowledge from multiple client models in the same dream dataset.

# Knowledge acquisition

Finally, the local clients and the server act as students and update their models using the collaboratively trained dreams obtained from $\mathrm { E q ~ 4 }$ . The clients share soft logits for each dream, which are then aggregated by the server to perform knowledge distillation on the following objective:

$$
\operatorname* { m i n } _ { \theta } \sum _ { \hat { x } \in \hat { \mathcal { D } } } \mathsf { K L } \left( \sum _ { k } \frac { 1 } { | \mathscr { D } _ { k } | } f _ { \theta _ { k } } ( \hat { x } ) \bigg | \bigg | f _ { \theta } ( \hat { x } ) \right)
$$

We provide the complete algorithm of Co-Dream in Algorithm 1. Note that the choice of parameters such as local updates $M$ , global updates $R$ , local learning rate $\eta _ { l }$ , global rate $\eta _ { g }$ , and the number of clients $K$ typically guide the tradeoff between communication efficiency and convergence.

# Analysis of Co-Dream

The benefits of Co-Dream are inherited from using KD, along with additional advantages arising from our specific optimization technique. Co-Dream extracts the knowledge from clients in dreams and shares the updates of these dreams instead of model gradients $( \nabla _ { \theta } )$ as done in FL.

Communication Analysis: We use the following notation: $d$ is the dimension of the inputs or dreams, $n$ is the batch size of dreams generated, and $R$ is the number of aggregation rounds. In FedAvg and its variants, the communication is $| \theta | \times R$ . Since $\mathrm { C o - D }$ ream communicates input gradients $( \nabla _ { \widehat { x } } )$ instead of model gradients $( \nabla _ { \theta } )$ , the total communication is $d \times n \times R$ . For heavily parameterized models, $d \times n \ll | \theta |$ . In a single batch, the communication complexity of $\mathrm { C o - D }$ ream does not scale with larger models. Table 3 provides a comprehensive communication analysis for different model architectures in FedAvg vs Co-Dream.

Privacy Analysis: Various model inversion and reconstruction attacks (Haim et al. 2022; Hitaj, Ateniese, and PerezCruz 2017) have shown private sensitive information can be

Algorithm 1: Co-Dream Algorithm

Input: Number of client $K$ , local models and data $\theta _ { k }$ and $\mathcal { D } _ { k } , k \in K$ , local learning rate $\eta _ { k }$ , global learning rate $\eta _ { g }$ , local training rounds $M$ , global training epochs $R$ , total number of epochs $N$ .   
for $t = 1$ to $N$ do Server initializes a batch of dreams ${ \hat { x } } \sim { \mathcal { N } } ( 0 , 1 )$ ; for $r = 1$ to $R$ do Server broadcasts $\hat { x } ^ { r }$ to all clients for each client $k \in K$ in parallel do $\hat { x } _ { k , 0 } ^ { r } : = \hat { x } ^ { r }$ ; for $m = 1$ to $M$ do // Local knowledge extraction (Eq 3) $\hat { x } _ { k , m } ^ { r }  \hat { x } _ { k , m - 1 } ^ { r } - \eta _ { k } \cdot \nabla _ { x } \big ( \tilde { \ell } ( \hat { x } _ { k , m - 1 } ^ { r } , \theta _ { k } ) \big )$ end each client shares pseudo-gradient $\begin{array} { r l } { \nabla \hat { x } _ { k } ^ { r } } & { { } = } \end{array}$ $\hat { x } _ { k , M } ^ { r } - \hat { x } ^ { r }$ with the server; end // Collaborative knowledge aggregation (Eq 4) $\begin{array} { r } { \hat { x } _ { S } ^ { r + 1 } \gets \hat { x } ^ { r } + \eta _ { g } \sum _ { k \in K } \frac { 1 } { | \mathscr { D } _ { k } | } \nabla \hat { x } _ { k } ^ { r } } \end{array}$ ; // Server aggregates model predictions $\begin{array} { r } { \hat { \boldsymbol { \mathcal { D } } } : = \{ \hat { x } ^ { r + 1 } , \hat { y } _ { S } ^ { r + 1 } : = \sum _ { k } \frac { 1 } { | \mathscr { D } _ { k } | } f _ { \theta _ { k } } \big ( \{ \hat { x } ^ { r + 1 } \} \{ \hat { x } ^ { \intercal } { } ^ { + } { } ^ { - } \} \big ) \} ; } \end{array}$ ; // Local knowledge acquisition (Eq 5) for each client $k \in K$ in parallel do LocalUpdate $( \hat { \mathcal { D } } , \theta _ { k } )$ ; LocalUpdate $( \mathcal { D } _ { k } , \theta _ { k } )$ ; end LocalUpdate $( \hat { \mathcal { D } } , \theta _ { s } )$ ; end   
end

reconstructed from just the model weights. While several reconstruction attacks perform model inversion, Co-Dream is optimized for improving performance on knowledge distillation. However, in Co-Dream, the clients collaborate by sharing the gradients of dreams’ without even sharing their model parameters. A simple application of data processing inequality shows that sharing dreams is at least as private as sharing model parameters. Similar to FedAvg, the synchronization step between the clients is a linear operation (weighted average) and hence offers an additional layer of privacy by using secure aggregation (Bonawitz et al. 2017). Finally, table 4 shows that our approach empirically outperforms benchmarks against the state-of-the-art LiRA membership inference attack (Carlini et al. 2022).

Flexibility of models: Since the knowledge aggregation in Co-Dream is done by sharing the updates of dreams in data space, Co-Dream is model agnostic and allows for collaboration among clients with different model architectures. We empirically observe no performance drop in collaborative learning with clients of different model architectures.

Customization in sharing knowledge: Additionally, sharing knowledge in the data space enables adaptive optimization, such as synthesizing adversarially robust samples or class-conditional samples for personalized learning.

<html><body><table><tr><td rowspan="2">Model</td><td colspan="4">Heterogeneous Clients (Independent clients 1-4)</td><td colspan="4">Method</td></tr><tr><td>WRN-16-1</td><td>VGG-11</td><td>WRN-40-1</td><td>ResNet-34</td><td>Independent</td><td>Centralized</td><td>AvgKD</td><td>Co-Dream (ours)</td></tr><tr><td>iid(α = inf)</td><td>52.2</td><td>55.1</td><td>43.5</td><td>54.2</td><td>51.6(4.5)</td><td>68.8</td><td>52.9(1.4)</td><td>69.6(1.0)</td></tr><tr><td>α=10</td><td>19.1</td><td>23.6</td><td>27.6</td><td>16.2</td><td>19.7(1.7)</td><td>58.5</td><td>50.4(1.3)</td><td>62.2(2.6)</td></tr><tr><td>α=1</td><td>41.3</td><td>38.2</td><td>37.1</td><td>50.1</td><td>41.7(5.1)</td><td>64.8</td><td>42.4(2.9)</td><td>60.0(1.7)</td></tr><tr><td>α=0.1</td><td>29.1</td><td>22.3</td><td>33.1</td><td>21.5</td><td>27.2(4.9)</td><td>43.0</td><td>30.2(3.3)</td><td>40.6(0.9)</td></tr></table></body></html>

Table 1: Performance comparison with heterogeneous client models: on CIFAR10 dataset. Left: Accuracy for independent heterogeneous clients with different models; Right: Average client model performance comparison of Co-Dream with other baselines   

<html><body><table><tr><td></td><td colspan="3">MNIST</td><td colspan="3">SVHN</td><td colspan="3">CIFAR10</td></tr><tr><td>Method</td><td>iid(α = inf)</td><td>α=1</td><td>α=0.1</td><td>iid(α = inf)</td><td>α=1</td><td>α=0.1</td><td>iid(α =inf)</td><td>α=1</td><td>α=0.1</td></tr><tr><td>Centralized</td><td>85.0(0.9)</td><td>61.4(7.1)</td><td>36.9(7.6)</td><td>80.8(1.3)</td><td>75.6(1.4)</td><td>54.6(13.6)</td><td>65.7(2.9)</td><td>65.3(0.4)</td><td>45.5(6.8)</td></tr><tr><td>Independent</td><td>52.4(7.0)</td><td>36.3(6.2)</td><td>22.0(4.2)</td><td>51.3(9.2)</td><td>42.3(6.4)</td><td>19.6(9.2)</td><td>46.4(2.0)</td><td>39.7(3.4)</td><td>23.5(5.2)</td></tr><tr><td>FedAvg</td><td>84.7(1.6)</td><td>60.3(3.4)</td><td>40.0(6.9)</td><td>82.9(0.4)</td><td>79.1(0.9)</td><td>47.1(23.7)</td><td>67.2(0.4)</td><td>62.3(0.9)</td><td>34.8(8.3)</td></tr><tr><td>FedProx</td><td>78.6(3.5)</td><td>62.6(3.6)</td><td>38.1(11.0)</td><td>86.9(0.1)</td><td>84.3(0.6)</td><td>48.7(26.7)</td><td>70.8(1.8)</td><td>62.3(2.9)</td><td>27.1(9.8)</td></tr><tr><td>Moon</td><td>85.1(2.6)</td><td>66.2(4.4)</td><td>42.3(11.8)</td><td>80.1(0.1)</td><td>76.5(1.2)</td><td>41.7(21.8)</td><td>66.6(1.4)</td><td>64.8(0.8)</td><td>35.5(10.8)</td></tr><tr><td>AvgKD</td><td>61.3(2.3)</td><td>44.3(4.8)</td><td>21.4(4.3)</td><td>75.4(0.7)</td><td>61.2(4.6)</td><td>20.7(10.9)</td><td>54.2(0.9)</td><td>46.4(3.3)</td><td>25.9(6.2)</td></tr><tr><td>SCAFFOLD</td><td>87.5(0.6)</td><td>70.2(3.6)</td><td>38.8(13.7)</td><td>86.0(0.1)</td><td>84.5(0.7)</td><td>13.5(4.4)</td><td>73.9(1.5)</td><td>67.5(4.6)</td><td>22.8(7.8)</td></tr><tr><td>FedGen</td><td>64.5(1.9)</td><td>51.0(4.3)</td><td>31.4(7.4)</td><td>49.7(1.6)</td><td>44.2(4.1)</td><td>34.9(19.7)</td><td>66.2(0.4)</td><td>62.8(1.8)</td><td>40.2(9.0)</td></tr><tr><td>Co-Dream (ours)</td><td>80.6(0.5)</td><td>57.7(3.6)</td><td>35.7(9.2)</td><td>81.4(0.1)</td><td>80.1(0.8)</td><td>44.5(17.7)</td><td>69.5(0.3)</td><td>64.8(0.3)</td><td>36.6(8.4)</td></tr></table></body></html>

Table 2: Performance overview of different techniques with different data settings. A smaller $\alpha$ indicates higher heterogeneity.

# Experiments

We systematically experiment and evaluate multiple aspects of Co-Dream. Unless stated otherwise, we used ResNet18 (He et al. 2015) for training the client and server models and set the total number of clients $K = 4$ . We conduct our experiments on 3 real-world datasets, including MNIST (LeCun et al. 1998), SVHN (Netzer et al. 2011), and CIFAR10 (Krizhevsky, Hinton et al. 2009). To validate the effect of collaboration, we train clients with 50 samples per client for MNIST and 1000 samples per client for CIFAR10 and SVHN datasets. For reference, we include two unrealistic baselines — Independent and Centralized. In the Centralized baseline, all the client data are aggregated in a single place. In the Independent baseline, clients only learn from their local data.

To simulate real-world conditions, we perform experiments on both IID and non-IID data. We use Dirichlet distribution $D i r ( \alpha )$ to generate non-IID data partition among labels for a fixed number of total samples at each client. The parameter $\alpha$ guides the degree of imbalance in the training data distribution. A small $\alpha$ generates more skewed data.

# Fast dreaming for knowledge extraction

Despite the impressive results of the original DreamInversion (Yin et al. 2020), we find 2000 local iterations to be too slow for a single batch of image generation when performed collaboratively in Co-Dream. Therefore, we use the Fast-datafree (Fang et al. 2022) approach that learns a prior for initializing dreams rather than initializing with random noise every time, to speed up image generation by a factor of 10 to 100 while preserving the performance. However, in each aggregation round, the client now shares both the local generator model and the dreams for aggregation by the server. Instead of 2000 global aggregation rounds (R) in Co-Dream, CoDream-fast performs only a single global aggregation round with 5 local rounds. We perform several subsequent experiments using CoDream-fast. More details on the implementation can be found in the Supplement material.

# Model-agnostic collaborative learning

Since Co-Dream shares updates in the data space instead of the model space, our approach is model agnostic. We evaluate our approach across heterogeneous client models including ResNet-34 (He et al. 2016), VGG-11 (Simonyan and Zisserman 2014), and Wide-ResNets (Zagoruyko and Komodakis 2016) (WRN-16-1 and WRN-40-1). Table 1 shows the performance of Co-Dream against Centralized, Independent, and model agnostic FL baselines such as Avg-KD (Afonin and Karimireddy 2022). Note that FedGen is not completely model agnostic as it requires the client models to have a shared feature extractor and thus cannot be applied to our setting. We exclude FedAvg as it doesn’t support heterogeneous models. Performing FL under both heterogeneous models and non-IID data distribution is a challenging task, yet Co-Dream outperforms the baselines.

# Communication efficiency

We compare the client communication cost of Co-Dream and FedAvg per round for different model architectures in Table 3. In FedAvg, the clients share the model with the server, whereas, in Co-Dream, they share the dreams(size of data). However, in Co-Dream, each batch of dreams is refined for 400 rounds, whereas in CoDream-fast there is only a single round of aggregation along with the sharing of a lightweight generator model (as explained in Section . The communication of both Co-Dream and CoDream-fast is model agnostic and does not scale with large models.

Table 3: Communication analysis of FedAvg vs Co-Dream and CoDream-fast per round   

<html><body><table><tr><td>Model</td><td>FedAvg</td><td>Co-Dream</td><td>CoDream-fast</td></tr><tr><td>Resnet34</td><td>166.6MB</td><td>600MB</td><td>23.5MB</td></tr><tr><td>Resnet18</td><td>89.4MB</td><td>600MB</td><td>23.5MB</td></tr><tr><td>VGG-11</td><td>1013.6MB</td><td>600MB</td><td>23.5MB</td></tr><tr><td>WRN-16-1</td><td>1.4 MB</td><td>600MB</td><td>23.5MB</td></tr><tr><td>WRN-40-1</td><td>4.5 MB</td><td>600MB</td><td>23.5MB</td></tr></table></body></html>

# Varying number of clients

We test whether Co-Dream actually encapsulates knowledge from multiple clients. We test this hypothesis by aggregating knowledge by varying the number of clients $K =$ [2, 4, 8, 12, 24], while keeping the total dataset size constant. Thus, as $K$ increases, each client has fewer local samples. As expected, performance declines with more clients, since each client’s knowledge is less representative of the overall distribution. However, this drop is sublinear (Figure 3), making Co-Dream viable for cross-device federated learning. The gap between Co-Dream and FedAvg remains similar across different $K$ .

In summary, Co-Dream sees a graceful decline in accuracy as data gets more decentralized. The framework effectively distills collective knowledge, even when local datasets are small. We conclude that averaging gradients in data space can combine the knowledge in the similar way as averaging gradients in the space of model parameters.

![](images/9030a1e4af8ab4f9c12a4cfe4f8bcc6ea5e723806bc76623887e670ab1cd1cb1.jpg)  
Figure 3: Comparison by varying the number of clients. The performance gap widens between Co-Dream and independent optimization as we increase the number of clients.   
Figure 4: Sample complexity of dreams in reaching target accuracy. The performance improvement saturates as we add more batches.

# Non-IID datasets benchmarking

We evaluate the feasibility of Co-Dream for both IID and non-IID settings by varying $\alpha \ : = \ : 0 . 1 , 0 . 5$ and report the performances of different methods in Table 2. We include popular non-IID specific algorithms such as FedProx (Li et al. 2020), Moon(Li, He, and Song 2021), and Scaffold (Karimireddy et al. 2020). We also include other model-agnostic federated baselines such as FedGen(Zhu, Hong, and Zhou 2021) and AvgKD (Afonin and Karimireddy 2022). The results show that Co-Dream is competitive with other non-iid techniques across all datasets and data partitions. Even as $\alpha$ becomes smaller (i.e., data become more imbalanced), Co-Dream still performs well. Note that Co-Dream does not beat other state-of-the-art non-iid techniques since it is not designed for the non-iid data challenges. It is analogous to FedAvg in the data space, and thus, existing non-iid tricks can also be applied to Co-Dream to improve its accuracy further.

# Analysis of sample complexity of dreams

We plot the accuracy of the server model trained from scratch against the number of batches of dreams it is trained on as shown in Fig 4. Note that the quality of generated dreams for training increases as training progresses in each round. We observe that 5 batches per round is a good enough size, after which the marginal gain is very small.

Sample Complexity of Dreams 0.7 0.6 W 0.34 Batches of Dreams generated per round 1 2 0.2 5 10 0.1 20 0 50 100 150 200 250 300 350 400 Rounds (Number of batches)

# Validating knowledge-extraction based on Eq 3

We evaluate whether the knowledge-extraction approach (Sec ) allows for the effective transfer of knowledge from teacher to student. We first train a teacher model from scratch on different datasets, synthesize samples with our knowledgeextraction approach, and then train a student on the extracted dreams. To validate its compatibility within an FL setting where clients have a small local dataset, we reduce the size of the training set of the teacher to reduce its local accuracy and measure corresponding student performance. Results in Fig 5 show that the teacher-student performance gap does not degrade consistently even when the teacher’s accuracy is low. This result is interesting because the extracted features get worse in quality as we decrease the teacher accuracy, but the performance gap is unaffected.

# Collaborative vs Independent optimization

We evaluate the effectiveness of collaborative optimization of dreams over multiple clients in aggregating the knowledge by comparing the performance of collaboratively optimized dreams in Co-Dream (using Eq 3) with independently optimized dreams. We show the aggregation step in Eq 3 not only helps in secure averaging, leading to more privacy but also improves the performance (Table 5, last row).

Table 4: Evaluation of privacy leakage by performing Membership Inference Attack   

<html><body><table><tr><td rowspan="2">Metric</td><td colspan="3">CIFAR10</td><td colspan="3">CIFAR100</td></tr><tr><td>Single Model</td><td>FedAvg</td><td>CoDream-fast (ours)</td><td>Single Model</td><td>FedAVG</td><td>CoDream-fast (ours)</td></tr><tr><td>Balanced Accuracy</td><td>57.39%</td><td>54.12%</td><td>50.77%</td><td>79.86%</td><td>51.84%</td><td>50.42%</td></tr><tr><td>AUROC</td><td>65.58%</td><td>63.17%</td><td>59.72%</td><td>77.01%</td><td>58.24%</td><td>57.53%</td></tr><tr><td>TPR @ 0.1% FPR</td><td>72.39%</td><td>53.71%</td><td>39.36%</td><td>39.63%</td><td>34.37%</td><td>31.40 %</td></tr><tr><td>TPR @ 0.001% FPR</td><td>11.34%</td><td>5.00%</td><td>2.66%</td><td>30.41%</td><td>1.98%</td><td>0.66%</td></tr></table></body></html>

![](images/33a1ea1dded3537c7870882dd969039d7be2257e46561916c87f6b85a1de5d65.jpg)  
Figure 5: Validating the effectiveness of knowledge transfer from teacher to student: We vary the size of the training dataset (on the $\mathbf { X }$ -axis) for the teacher and compare its accuracy with the student trained on dreams generated using Eq 3

Table 5: Ablation of components in Co-Dream on CIFAR10   

<html><body><table><tr><td>Datapartition</td><td>iid</td><td>α=1</td><td>α=0.1</td></tr><tr><td>Co-Dream</td><td>69.2(0.1)</td><td>61.6(0.5)</td><td>45.6(1.5)</td></tr><tr><td>w/o Radu</td><td>65.7(0.2)</td><td>58.4(1.3)</td><td>42.0(1.4)</td></tr><tr><td>w/o Rbn</td><td>51.2(6.1)</td><td>33.1(7.1)</td><td>24.1(5.2)</td></tr><tr><td>w/o collab</td><td>64.4(0.5)</td><td>58.4(1.4)</td><td>30.8(3.2)</td></tr></table></body></html>

# Contribution of loss components $\mathcal { R } _ { b n }$ and $\mathcal { R } _ { a d v }$ in knowledge extraction

We further explore the impacts of various components of loss function in data generation in Eq. 3. Through leave-one-out testing, we present results by excluding $\mathcal { R } _ { b n }$ (w/o $\mathcal { R } _ { b n } \mathrm { ~ . ~ }$ ) and excluding $\mathcal { R } _ { a d v }$ (w/o $\mathcal { R } _ { a d v . }$ ). Table 5 shows removing either component influences the accuracy of the overall model, illustrating the impact of each part of the loss function plays an important role in generating good quality dreams.

# Membership Inference Attack Evaluation

We evaluate whether models trained over dreams leak less or more information than their federated or centralized counterpart. We evaluate information leakage in the trained models using the LiRA membership inference attack (Carlini et al. 2022). We train all models to achieve a similar classification accuracy to ensure a fair comparison of their privacypreserving capabilities. To simulate the LiRA attack, we perform a single query per datapoint to calculate the $\phi$ scores, utilizing one target model and ten shadow models. Since the attack objective is binary classification, we measure attack success with balanced accuracy, AUROC, and true-positive rates at the $0 . 1 \%$ and $0 . 0 0 1 \%$ false positive rates, with lower scores indicating lesser privacy leakage. We plot the results in Table 4. CoDream-fast achieves a better performance on all the metrics for the LiRA attack. While not a worst-case theoretical improvement, the results hint that CoDream-fast can be a more viable alternate for sharing knowledge.

# Visual representation of dreams

Figure 6 visualizes the dreams generated by CoDream-fast on CIFAR10. While not visually similar to the original training data, these dreams effectively encapsulate collaborative knowledge. Unlike traditional model inversion algorithms, the objective for these dreams is to enable decentralized knowledge transfer and not reconstruct the raw data. Thus, models trained on dreams perform well despite their visual differences from the underlying distribution. We visualize more images of Co-Dream across different data distributions in the Supplementary.

![](images/dcbf9c399dfe7443634c86e07be315a4fb917701f44e5bd7b46fd41ee1097da5.jpg)  
Figure 6: Visualization of dreams generated on CIFAR10

# Conclusion

We introduce Co-Dream, a model-agnostic learning framework that leverages a knowledge extraction algorithm by performing gradient descent in the input space. We view this approach as a complementary technique to FedAvg, which performs gradient descent over model parameters. Through comprehensive evaluations and ablation studies, we validate the effectiveness of our proposed method.

While we restrict the scope of our work on collaborative learning applications, we believe Co-Dream can serve as a building block for several other interesting problems such as identifying similarity between models without relying on proxy datasets. We believe further research is warranted in client dropouts, stragglers, formal privacy guarantees, bias to explore the effectiveness of Co-Dream under those constraints.