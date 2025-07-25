# GAS: Generative Activation-Aided Asynchronous Split Federated Learning

Jiarong Yang, Yuan Liu

School of Electronic and Information Engineering, South China University of Technology eejryang $@$ mail.scut.edu.cn, eeyliu $@$ scut.edu.cn

# Abstract

Split Federated Learning (SFL) splits and collaboratively trains a shared model between clients and server, where clients transmit activations and client-side models to server for updates. Recent SFL studies assume synchronous transmission of activations and client-side models from clients to server. However, due to significant variations in computational and communication capabilities among clients, activations and client-side models arrive at server asynchronously. The delay caused by asynchrony significantly degrades the performance of SFL. To address this issue, we consider an asynchronous SFL framework, where an activation buffer and a model buffer are embedded on the server to manage the asynchronously transmitted activations and clientside models, respectively. Furthermore, as asynchronous activation transmissions cause the buffer to frequently receive activations from resource-rich clients, leading to biased updates of the server-side model, we propose Generative activationsaided Asynchronous SFL (GAS). In GAS, the server maintains an activation distribution for each label based on received activations and generates activations from these distributions according to the degree of bias. These generative activations are then used to assist in updating the server-side model, ensuring more accurate updates. We derive a tighter convergence bound, and our experiments demonstrate the effectiveness of the proposed method.

Code — https://github.com/eejiarong/GAS

# Introduction

Split Federated Learning (SFL) (Jeon and Kim 2020; Thapa et al. 2022) emerges as a promising solution for efficient resource-constrained distributed learning by combining the benefits of both Federated Learning (FL) (McMahan et al. 2017; Singh et al. 2022) and Split Learning (SL) (Gupta and Raskar 2018). Specifically, in SFL, the model is split into two parts: the initial layers are processed in parallel by the participating clients, and the intermediate activations are sent to the server, which completes the remaining layers. The server then sends the backpropagated gradients back to the clients, who use these gradients to update their client-side models. After several iterations, the server aggregates the client-side models to form the globally updated model.

Traditional SFL always assumes synchronous model exchange, where the server waits to receive all client-side models for aggregation. However, since clients have different communication and computational capabilities, client-side models are uploaded at the server asynchronously. The slow clients are referred to as stragglers, delaying the overall training process. Previous works in FL such as FedAsync (Xie, Koyejo, and Gupta 2019) and FedBuff (Nguyen et al. 2022) are proposed to tackle the stragglers issue by allowing clients to update the global model asynchronously. Additionally, $\mathrm { C A ^ { 2 } F L }$ (Wang et al. 2024) addresses convergence degradation by caching and reusing previous updates for global calibration, ensuring more consistent model updates despite asynchronous conditions.

However, the existing works on stragglers issue in SFL still have limitations. On one hand, the adaptive model splitting methods (Yan et al. 2023; Shen et al. 2023) for addressing this issue in SFL are constrained by the model structure. Specifically, these methods attempt to balance the arrival time of activations by selecting appropriate split layers of the model. Nevertheless, when the sizes of activations output by different model layers are similar or the computational and communication capabilities of clients are highly different, it is impossible to ensure simultaneous arrival of activations, regardless of the chosen split layers. On the other hand, the stragglers issue is more serious in SFL. Specifically, recent SFL methods (Huang, Tian, and Tang 2023; Yang and Liu 2024) assume synchronous activation transmissions with heterogeneous client data, where the uploaded activations are concatenated to update the server-side model centrally, reducing the bias in the deep layers of the model (Luo et al. 2021). However, these methods require the server to wait for stragglers to send their activations at the end of each local iteration, and the frequent transmissions of activations exacerbate the stragglers issue.

To address the above issues, we propose the asynchronous learning framework for SFL, where an activation buffer and a model buffer are embedded on the server to handle asynchronous updates. Specifically, the activation buffer stores the activations uploaded asynchronously. When the buffer is full, the server concatenates these activations and uses them to update the server-side model. Similarly, the model buffer stores the client-side models uploaded asynchronously. When the buffer is full, the server aggregates the stored client-side models. By introducing the two buffers, we ensure efficient model updates and reduce delays caused by stragglers. However, due to the heterogeneous communication and computational capabilities of clients, the activation buffer may frequently receive activations from resourcerich clients, leading to biased updates in the server-side model (Leconte et al. 2024; Liu et al. 2024). To solve this issue, we propose Generative activation-aided Asynchronous SFL (GAS). Specifically, the server maintains a distribution of activations for each label, dynamically updated based on the uploaded activations. When updating the server-side model, we generate the activations from the distributions according to the degree of bias. Then these generative activations are concatenated with the stored activations to update the server-side model, thereby mitigating the model update bias introduced by stragglers. We summarize our contributions in this paper as follows:

• We propose an asynchronous SFL framework that enables the asynchronous transmissions of activations and client-side models. To our best knowledge, this is first attempt considering asynchronous SFL. • We propose GAS (Generative activation-aided Asynchronous SFL), where the server updates the activation distribution for each label based on the uploaded activations and generates activations from the distributions to assist server-side model updates, mitigating the model update bias caused by stragglers. • Several useful insights are obtained via our theoretical analysis: First, GAS can mitigate the gradient dissimilarity introduced by stragglers. Second, GAS achieves a tighter convergence bound. Third, by setting a decaying learning rate, the impact of stragglers can be gradually mitigated as the training progresses.

# Related Works

# Split Federated Learning

SFL (Thapa et al. 2022) combines the strengths of FL (McMahan et al. 2017) and SL (Gupta and Raskar 2018) to offer a more efficient and scalable learning framework. Recent research has explored various aspects of SFL. To enhance communication efficiency, FedLite (Wang et al. 2022a) employ compression techniques to reduce the volume of activation data transmitted. Simultaneously, the work by (Han et al. 2021) introduces auxiliary networks on the client side, eliminating the need for sending backpropagated gradients. In terms of privacy preservation, ResSFL (Li et al. 2022) and NoPeek (Li et al. 2022) implement attacker-aware training with an inversion score regularization term to counteract model inversion attacks. Additionally, the works by (Xiao, Yang, and Wu 2021) and (Thapa et al. 2022) leverage mixed activations and differential privacy to safeguard against privacy breaches from intermediate activations. To optimize performance for heterogeneous clients, SCALA (Yang and Liu 2024) and MiniBatch-SFL (Huang, Tian, and Tang 2023) employ activation concatenation and implement centralized training on the server, thereby enhancing model robustness and accuracy. Meanwhile, $S ^ { 2 } \dot { \mathrm { F L } }$ (Yan et al. 2023)

and RingSFL (Shen et al. 2023) address the stragglers issue by employing adaptive model splitting methods. Furthermore, recent works (Lin et al. 2024; Xu et al. 2023) refines SFL for real-world communication environments by selecting model split layers based on client channel conditions.

# Asynchronous Federated Learning

Asynchronous FL addresses the limitations of traditional synchronous FL in heterogeneous environments, where “stragglers”, or slow clients, can degrade overall training performance and efficiency (Wang et al. 2021). Early asynchronous FL frameworks (Xie, Koyejo, and Gupta 2019; Chen, Sun, and Jin 2019) mitigate the impact of stragglers by adaptively weighting the local updates. ASO-Fed (Chen et al. 2020) employs a dynamic learning strategy to adjust the local training step size, reducing the staleness effects caused by stragglers. FedBuff (Nguyen et al. 2022) introduces a buffering mechanism to temporarily store updates from faster clients, achieving higher concurrency and improving training efficiency. $\bar { \mathrm { C A ^ { 2 } F L } }$ (Wang et al. 2024) further advances this approach by caching and calibrating updates based on data properties to handle both stragglers and data heterogeneity. FedCompass (Li et al. 2023) enhances efficiency by using a computing power-aware scheduler to prioritize updates from more powerful clients, thus reducing the waiting time for stragglers. FedASMU (Liu et al. 2024) addresses the stragglers issue through dynamic model aggregation and adaptive local model adjustment methods. Moreover, some works (Lee and Lee 2021; Wang et al. 2022b; Zhu et al. 2022; Hu, Chen, and Larsson 2023) have further optimizes the performance of asynchronous FL in wireless communication environments through staleness-aware model aggregation and client selection schemes.

Note that the asynchronously transmitted activations are concatenated rather than aggregated to update the serverside model in SFL, introducing unique challenges that make previous asynchronous FL methods inapplicable. Furthermore, the more frequent transmissions of activations exacerbate the stragglers issue. Existing SFL methods (Yan et al. 2023; Shen et al. 2023) employ adaptive model splitting to balance activation arrival times; however, they are constrained by the model structure. The above challenges highlight the need for further research to develop a tailored asynchronous framework for SFL. In this paper, we propose GAS to fill the gap by introducing a novel buffer mechanism and generative activations, which address the stragglers issue in SFL and achieve better model performance. Additionally, while CCVR (Luo et al. 2021) and FedImpro (Tang et al. 2024) also employ activation generation to enhance model performance, they require the additional transmission of local activation distributions. GAS distinguishes itself by leveraging the inherent characteristics of the SFL framework to dynamically update activation distributions using the activations continuously uploaded by clients, without incurring extra communication overhead.

# Proposed Method

In this section, we systematically introduce GAS, which employs an activation buffer and a model buffer to enable asyn

chronous transmissions of activations and client-side models, while leveraging generative activations to mitigate update bias caused by stragglers.

# Preliminaries

Consider a SFL scenario involving $K$ clients indexed by $\mathcal { K } ~ = ~ \{ 1 , 2 , \ldots , K \}$ . Each client $k$ holds a local dataset $\mathcal { D } _ { k }$ with $\left| \mathcal { D } _ { k } \right|$ data points. The clients collaborate to train a global model w under the coordination of the server. In SFL, the global model w is split into two parts: the clientside model $\mathbf { w } _ { c }$ and the server-side model $\mathbf { w } _ { s }$ . The clients perform local computations on $\mathbf { w } _ { c }$ and send the activations to the server, which completes the forward and backward passes using ${ \bf w } _ { s }$ . Thus the empirical loss for client $k$ is defined as

$$
f _ { k } ( \mathbf { w } ; \tilde { \mathcal { D } } _ { k } ) = l ( \mathbf { w } _ { s } ; h ( \mathbf { w } _ { c } ; \tilde { \mathcal { D } } _ { k } ) ) ,
$$

with $h$ representing the client-side function that maps the sampled mini-batch data $\tilde { \mathcal { D } } _ { k }$ to the intermediate activations, and $l$ representing the server-side function that maps activations to the final loss value. We assume partial client participation and the primary objective is to minimize the global loss function over the participating clients $\mathcal { C }$ , formulated as

$$
\operatorname* { m i n } _ { \mathbf { w } } F ( \mathbf { w } ) = \frac { \sum _ { k \in \mathcal { C } } | \mathcal { D } _ { k } | F _ { k } ( \mathbf { w } ) } { \sum _ { k \in \mathcal { C } } | \mathcal { D } _ { k } | } ,
$$

where $F _ { k } ( \mathbf { w } )$ is the local expected loss function for client $k$ and it is unbiasedly estimated by the empirical loss $f _ { k } ( \mathbf { w } ; \tilde { \mathcal { D } } _ { k } )$ , such that $\vec { \mathbb { E } } _ { \tilde { \mathcal { D } } _ { k } \sim \mathcal { D } _ { k } } f _ { k } ( \mathbf { w } , \mathcal { \tilde { \tilde { D } } } _ { k } ) = F _ { k } ( \mathbf { \bar { w } } )$ .

# Overall Structure

In Fig. 1, we illustrate the six key steps of GAS. The pseudocode is illustrated in Technical Appendix A. At the beginning of the training process, the server sets the number of local iterations $E$ and global iterations $T$ , with local iterations indexed by $e$ and global iterations indexed by $t$ . The server then initializes an activation buffer $\mathcal { A }$ to store the received activations and their corresponding labels, and a model buffer $\mathcal { M }$ to store the received client-side models along with the respective client data sizes. Next, the server sets the minibatch size to $B$ , the activation buffer size to $Q _ { s } B$ , and the model buffer size to $Q _ { c }$ . Additionally, the server initializes the global model as $\mathbf { \dot { w } } ^ { 0 } = [ \mathbf { w } _ { c } ^ { 0 } , \mathbf { w } _ { s } ^ { 0 } ]$ and selects $C$ initial clients to participate in the training. A detailed description of the training process follows.

• Forward propagation of the client-side model (Fig. $\mathbf { 1 } ( \mathbb { 1 } )$ : The selected client $k$ receives the client-side model and randomly selects a minibatch $\tilde { \mathcal { D } } _ { k }$ with a batch size of $B$ from its local dataset $\mathcal { D } _ { k }$ . The minibatch $\tilde { \mathcal { D } } _ { k }$ is defined as $\tilde { \mathcal { D } } _ { k } = \{ ( \mathbf { x } _ { 1 } , y _ { 1 } ) , ( \mathbf { x } _ { 2 } , y _ { 2 } ) , \dots , ( \mathbf { x } _ { B } , y _ { B } ) \}$ , where the input samples are $\mathbf { X } _ { k } = \left\{ \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { B } \right\}$ and their corresponding labels are $\mathbf { Y } _ { k } = \{ y _ { 1 } , y _ { 2 } , . . . , y _ { B } \}$ . The client $k$ then performs forward propagation using the clientside model $\mathbf { w } _ { c }$ to compute the activations $\mathbf { A } _ { k }$ of the last layer of the client-side model, given by

$$
{ \mathbf { A } } _ { k } = h ( \mathbf { w } _ { c } ; \tilde { \mathcal { D } } _ { k } ) .
$$

Upon completing the computation, the activations $\mathbf { A } _ { k }$ along with the label set $\mathbf { Y } _ { k }$ are sent to the server.

• Activations generation (Fig. $\mathbf { 1 } ( \mathbf { 2 } ) ,$ ) and server-side model update (Fig. $\mathbf { 1 } ( 3 )$ : The server receives activations from selected client $k$ and stores them in the activation buffer as

$$
\mathcal { A }  \mathcal { A } \cup ( \mathbf { A } _ { k } , \mathbf { Y } _ { k } ) .
$$

Additionally, the server maintains an activation distribution for each label, which is dynamically updated based on the received activations. (detailed in the next section). When the number of activations in the activation buffer exceeds $Q _ { s } B$ , the server generates activations $\widehat { \mathbf { A } }$ from the distribution. The generative activations $\widehat { \mathbf { A } }$ rbe then concatenated with the activations in the b fber as $\mathbf { A } _ { \mathrm { c o n c a t } } = \mathrm { c o n c a t } ( \mathbf { A } _ { 1 } , \mathbf { A } _ { 2 } , \dots , \mathbf { A } _ { Q _ { s } } , { \widehat { \mathbf { A } } } )$ . Similarly, the corresponding labels $\widehat { \mathbf Y }$ are concatenatbed with the labels in the buffer as $\mathbf { Y } _ { \mathrm { c o n c a t } } = \operatorname { c o n c a t } ( \mathbf { Y } _ { 1 } , \mathbf { Y } _ { 2 } , \dots , \mathbf { Y } _ { Q _ { s } } , { \widehat { \mathbf { Y } } } )$ . Finally, the concatenated activations $\mathbf { A } _ { \mathrm { { c o n c a t } } }$ are usedbas input to update the server-side model:

$$
\mathbf { w } _ { s } ^ { e + 1 } = \mathbf { w } _ { s } ^ { e } - \eta \nabla _ { \mathbf { w } _ { s } ^ { e } } l ( \mathbf { w } _ { s } ^ { e } ; \mathbf { A } _ { \mathrm { c o n c a t } } , \mathbf { Y } _ { \mathrm { c o n c a t } } ) .
$$

• Backpropagation of server-side model $( \mathbf { F i g . 1 } @ )$ The server computes the backpropagated gradients based on the received activations. As logit adjustment (Menon et al. 2021; Zhang et al. 2022) is popular for improving model performance under conditions of data heterogeneity, we apply it to calibrate the loss function of each client, as follows:

$$
l _ { k } ( \mathbf { w } _ { s } ; \mathbf { A } _ { k } , \mathbf { Y } _ { k } ) = - \log \left[ \frac { e ^ { s _ { y } } + \log P _ { k } ( y ) } { \sum _ { y ^ { \prime } = 1 } ^ { M } e ^ { s _ { y ^ { \prime } } + \log P _ { k } ( y ^ { \prime } ) } } \right] ,
$$

where $s _ { y }$ is predicted score for label $y , P _ { k } ( y )$ is the label distribution of client $k$ and $M$ is the total number of classes. Thus the backpropagated gradient is computed as

$$
\mathbf { G } _ { k } = \nabla _ { \mathbf { A } _ { k } } l _ { k } \big ( \mathbf { w } _ { s } ; \mathbf { A } _ { k } , \mathbf { Y } _ { k } \big ) ,
$$

which is then sent to client $k$ .

• Backpropagation of client-side model $( \mathbf { F i g . 1 } ( \mathbf { 5 } ) )$ : The client $k$ performs backpropagation using the received gradient and updates its local client-side model using the chain rule, given by

$$
\begin{array} { r l } & { \mathbf { w } _ { c , k } ^ { e + 1 } = \mathbf { w } _ { c , k } ^ { e } } \\ & { \quad \quad - \eta \nabla _ { \mathbf { A } _ { k } } l _ { k } ( \mathbf { w } _ { s } ; \mathbf { A } _ { k } , \mathbf { Y } _ { k } ) \nabla _ { \mathbf { w } _ { c , k } ^ { e } } h _ { k } ( \mathbf { w } _ { c , k } ^ { e } ; \mathbf { X } _ { k } ) . } \end{array}
$$

When client $k$ completes $E$ local iterations, it sends the locally updated client-side model to the server.

• Update of client-side model (Fig. $\mathbf { 1 } \textcircled { 6 }$ ): The server receives the updated client-side model and stores it in the model buffer as

$$
\mathcal { M }  \mathcal { M } \cup ( \mathbf { w } _ { c , k } , \vert \mathcal { D } _ { k } \vert ) ,
$$

When the number of models in the model buffer exceeds $Q _ { c }$ , the server aggregates these models as the current client-side model, given by

$$
\mathbf { w } _ { c } ^ { \mathrm { a g g } } = \frac { \sum _ { ( \mathbf { w } _ { c , k } , | \mathcal { D } _ { k } | ) \in \mathcal { M } } | \mathcal { D } _ { k } | \mathbf { w } _ { c , k } } { \sum _ { ( \mathbf { w } _ { c , k } , | \mathcal { D } _ { k } | ) \in \mathcal { M } } | \mathcal { D } _ { k } | }
$$

![](images/37ff3f862cac1497ffc2beb708f814524c4604af6b0ee25f1694824cf397be83.jpg)  
Figure 1: The framework of GAS. The client-side model is updated through four steps: $\textcircled{1}$ Clients perform forward propagation; $\textcircled{4}$ The server receives the activations and computes backpropagated gradients; $\textcircled{5}$ Clients receive the gradients to update the client-side models, and complete a local iteration. After finishing local iterations, clients send the updated client-side models to the server; $\textcircled{6}$ The server stores these models in the model buffer and, when full, aggregates them to complete a global iteration. The server-side model is updated through two steps: $\textcircled{2}$ Received activations update the distributions of activations. When the activation buffer is full, the server generate activations from these distributions; $\textcircled{3}$ Activations are stored in the buffer and, when full, the server concatenates them with generative activations to update the server-side model.

Then the server selects new client to participate in the training and sends it the current client-side model.

Note that the server-side model and the client-side models are not updated synchronously. Since the frequency of server-side model updates and client-side model aggregations is determined by the activation buffer size and the model buffer size, we typically set $Q _ { s } = Q _ { c }$ to ensure consistency in model updates. Additionally, to clarify the notation, we define a global iteration as $E$ updates of the serverside model. After $E \times T$ iterations, the trained server-side model $\mathbf { w } _ { s } ^ { T }$ is obtained. We define each aggregation of clientside models as a global iteration and after $T$ aggregations, the trained client-side model $\mathbf { w } _ { c } ^ { T }$ is obtained.

# Generative Activation-Aided Updates

Due to the activations being uploaded asynchronously by selected clients, the activation buffer frequently receives activations from resource-rich clients. This results in a bias in the server-side model updates. To address this issue, we propose a method called Generative Activation-Aided Updates (Fig. $\boldsymbol { 1 } ( \mathbf { 2 } )$ and Fig. $\textcircled { 1 \textcircled { 3 } }$ ), where the server maintains the distribution of activations for each label $y$ , represented as a Gaussian distribution $\mathcal { N } _ { y } ( \pmb { \mu } _ { y } , \pmb { \Sigma } _ { y } )$ . The server generates activations from these distributions to assist in updating the server-side model. The key steps are as follows:

• Dynamic Weighted Update: The server dynamically updates the mean $\pmb { \mu }$ and variance $\pmb { \Sigma }$ of the activation distribution using asynchronously uploaded activations in a weighted manner. Specifically, we define the weighting function as $s ( n )$ , where $n$ represents the training progress, denoted by the total number of iterations $\begin{array} { r } { n \ = \ t E + e } \end{array}$ . Since activations are uploaded asynchronously, each activation has a different training progress. We define $n ( \mathbf { A } )$ as the training progress of activation A. The weighted mean for a training progress of $N$ can be expressed as: $\mu _ { N } \ =$ $\begin{array} { r } { \frac { 1 } { S _ { N } } \sum _ { \mathbf { A } \in \mathcal { A } _ { N } } s ( n ( \mathbf { A } ) ) \mathbf { A } } \end{array}$ . And the weighted variance is given by $\begin{array} { r } { \Sigma _ { N } = \frac { 1 } { S _ { N } } \sum _ { \mathbf { A } \in \mathcal { A } _ { N } } s ( n ( \mathbf { A } ) ) ( \mathbf { A } - \mu _ { N } ) ( \mathbf { A } - \mathbf { \sigma } } \end{array}$ $\mu _ { N } ) ^ { T }$ , where $\mathcal { A } _ { N }$ denotes the set of all activations uploaded to the server up to training progress $N$ , and $S _ { N }$ is the sum of the weights, defined as $\begin{array} { r l } { S _ { N } } & { { } = } \end{array}$ $\textstyle \sum _ { \mathbf { A } \in { \mathcal { A } } _ { N } } s ( n ( \mathbf { A } ) )$ . Since activations are dynamically uploaded, we adopt a dynamic update approach. Given a newly received activation $\mathbf { A }$ , the mean is dynamically updated by

$$
{ \pmb \mu } _ { N } = \frac { S _ { N - 1 } } { S _ { N - 1 } + s ( n ( { \bf A } ) ) } { \pmb \mu } _ { N - 1 } + \frac { s ( n ( { \bf A } ) ) } { S _ { N - 1 } + s ( n ( { \bf A } ) ) } { \bf A } .
$$

The variance is dynamically updated by

$$
\begin{array} { r } { \pmb { \Sigma } _ { N } = \frac { S _ { N - 1 } ( \pmb { \Sigma } _ { N - 1 } + ( \pmb { \mu } _ { N } - \pmb { \mu } _ { N - 1 } ) ( \pmb { \mu } _ { N } - \pmb { \mu } _ { N - 1 } ) ^ { T } ) } { S _ { N - 1 } + s ( n ( \mathbf { A } ) ) } } \\ { + \frac { s ( n ( \mathbf { A } ) ) ( \pmb { \mu } _ { N } - \mathbf { A } ) ( \pmb { \mu } _ { N } - \mathbf { A } ) ^ { T } } { S _ { N - 1 } + s ( n ( \mathbf { A } ) ) } . \qquad ( 1 2 ) } \end{array}
$$

The Derivation can be founded in Technical Appendix $\mathbf { B }$ .

• Generating and Concatenation: During the server-side model update, the server generates activations $\widehat { \mathbf { A } }$ by sampling from the distributions according to the skbewness of the labels. For instance, the server adjusts the sampling to ensure that each label has an equal amount of data. These generative activations are then concatenated with the activations in the activation buffer to form the input for updating the server-side model as (5). This method ensures that the server-side model receives a more balanced set of activations, mitigating the bias introduced by stragglers.

Note that we consider newer activations to be more important. Therefore, we define an increasing weighting function, such as an exponential function $s ( n ) { \stackrel { } { = } } a e ^ { { \tilde { b } } n }$ or a polynomial function $s ( n ) = a n ^ { b }$ (Xie, Koyejo, and Gupta 2019; Liu et al. 2024), where stale activations become less significant as training progresses, thereby mitigating the impact of stragglers on the activation distribution updates.

# Theoretical Analysis

In this section, we provide a theoretical analysis to better understand the error bound and performance improvement of the proposed GAS. Since the server-side model and the client-side model are updated independently, where the parameters of one model remain fixed while the other is updated, we separately analyze the convergence rates of the server-side model and the client-side model. To ensure clarity, we denote $f _ { k } ( \mathbf { w } _ { c } )$ as the local loss function of the clientside model $h ( \mathbf { w } _ { c } ; \tilde { \mathcal { D } } _ { k } )$ , and $f _ { s } ( \mathbf { w } _ { s } )$ as the loss function of the server-side model $l ( \mathbf { w } _ { s } ; \mathbf { A } _ { \mathrm { c o n c a t } } , \mathbf { Y } _ { \mathrm { c o n c a t } } )$ . Our analysis is based on the following assumptions:

Assumption 1. (Smoothness) Loss function of server-side model and each local loss function of client-side model are Lipschitz smooth, i.e., for all w and $\mathbf { w } ^ { \prime }$ , $\left. \nabla _ { \mathbf { w } _ { s } } f _ { s } ( \mathbf { w } _ { s } ) - \right.$ $\begin{array} { r l r } { \bar { \nabla _ { \bf w } } _ { s } f _ { s } ( \mathbf { w } _ { s } ^ { \prime } ) \| } & { { } \leq } & { \gamma _ { 1 } \| \mathbf { w } _ { s } \mathrm { ~ - ~ } \mathbf { w } _ { s } ^ { \prime } \| } \end{array}$ and $\big | \big | \nabla _ { \mathbf { w } _ { c } } f _ { k } \big ( \mathbf { w } _ { c } \big ) \ -$ $\nabla _ { \mathbf { w } _ { c } } f _ { k } \big ( \mathbf { w } _ { c } ^ { \prime } \big ) \| \leq \gamma _ { 2 } \| \mathbf { w } _ { c } - \mathbf { w } _ { c } ^ { \prime } \|$ .

Assumption 2. (Bounded Gradient Variance) The stochastic gradient of server-side model $\nabla _ { \mathbf { w } _ { s } } f _ { s } ( \mathbf { w } _ { s } )$ and the stochastic gradient of client-side model $\nabla _ { \mathbf { w } _ { c } } f _ { k } ( \mathbf { w } _ { c } )$ have bounded variance: $\begin{array} { r l } { \mathbb { E } [ \| \nabla _ { { \bf w } _ { s } } f _ { s } ( { \bf w } _ { s } ) - \nabla _ { { \bf w } _ { s } } \bar { F } _ { s } ( { \bf w } _ { s } ) \| ^ { 2 } ] } & { \leq } \end{array}$ $\frac { \sigma ^ { 2 } } { B Q _ { s } }$ and $\begin{array} { r } { \mathbb { E } [ \| \nabla _ { \mathbf { w } _ { c } } f _ { k } ( \mathbf { w } _ { c } ) - \nabla _ { \mathbf { w } _ { c } } F _ { k } ( \mathbf { w } _ { c } ) \| ^ { 2 } ] \leq \frac { \sigma ^ { 2 } } { B } } \end{array}$ .

Assumption 3. (Bounded Dissimilarity) In server-side model updates, gradient dissimilarity is referred to as the bias caused by stragglers, which is bounded as: $\begin{array} { r l r } { \mathbb { E } \left[ \| \nabla _ { { \bf w } _ { s } } F _ { s } ( { \bf w } _ { s } ) - \dot { \nabla } _ { { \bf w } _ { s } } F ( \mathbf { \tilde { w } } _ { s } ) \| ^ { 2 } \right] } & { \leq } & { \kappa _ { 1 } ^ { 2 } } \end{array}$ . In client-side model updates, gradient dissimilarity is referred to as the bias caused by data heterogeneity across clients, which is bounded as: E $\left[ | | \nabla _ { \mathbf { w } _ { c } } F _ { k } ( \mathbf { w } _ { c } ) - \dot { \nabla } _ { \mathbf { w } _ { c } } F ( \mathbf { w } _ { c } ) | | ^ { 2 } \right] \leq \kappa _ { 2 } ^ { 2 }$ .

In the proposed GAS, the activation distribution gradually approximates the ground-truth activation distribution through dynamic updates. This leads us to the following lemma:

Lemma 1. By introducing generative activations, the server-side model update achieves a tighter bounded dissimilarity, as shown below:

$$
\begin{array} { r l } & { \mathbb { E } _ { ( \mathbf { A } , \mathbf { Y } ) \sim \mathcal { A } } \left[ \left\| \nabla _ { \mathbf { w } _ { s } } F _ { s } ( \mathbf { w } _ { s } ) - \nabla _ { \mathbf { w } _ { s } } F ( \mathbf { w } _ { s } ) \right\| ^ { 2 } \right] } \\ & { \qquad \hat { \mathbf { A } } { \sim } \mathcal { N } } \\ & { \quad \leq \mathbb { E } _ { ( \mathbf { A } , \mathbf { Y } ) \sim \mathcal { A } } \left[ \left\| \nabla _ { \mathbf { w } _ { s } } F _ { s } ( \mathbf { w } _ { s } ) - \nabla _ { \mathbf { w } _ { s } } F ( \mathbf { w } _ { s } ) \right\| ^ { 2 } \right] . } \end{array}
$$

The Proof can be founded in Technical Appendix $C _ { \cdot }$ .

This reduction in gradient dissimilarity indicates that the server-side model update becomes less biased by concatenating generative activations. Now, we are ready to state the following theorem, which provides the convergence upper bounds for the proposed GAS, considering both the clientside model and the server-side model.

Theorem 1. When Assumptions 1-3 hold, given the learning rate $\begin{array} { r } { \eta \le \frac { 1 } { \gamma _ { 1 } } } \end{array}$ , the convergence rate of server-side model is given by

$$
\begin{array} { r l r } {  { \frac { 1 } { E T } \sum _ { t = 0 } ^ { T - 1 } \sum _ { e = 0 } ^ { E - 1 } \mathbb { E } [ \| \nabla _ { \mathbf { w } _ { s } } F ( \mathbf { w } _ { s } ^ { t , e } ) \| ^ { 2 } ] } } \\ & { } & { \leq \mathcal { O } ( \frac { F ( \mathbf { w } _ { s } ^ { 0 } ) - F ^ { * } } { E T \eta } + \frac { \eta \sigma ^ { 2 } } { B Q _ { s } + \widehat { B } } + \kappa _ { 1 } ^ { 2 } ) , } \end{array}
$$

where $\widehat { B }$ is the batch size of generative activations.

Giv nb the learning rate η ≤ 20γ √1τ and the maximum upload delay of client-side model $\tau _ { \mathrm { m a x } }$ , the convergence rate of client-side model is given by

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left[ \| \nabla _ { \mathbf { w } _ { c } } F ( \mathbf { w } _ { c } ^ { t } ) \| ^ { 2 } \right] \leq \mathcal { O } \Bigg ( \frac { F ( \mathbf { w } _ { s } ^ { 0 } ) - F ^ { * } } { E T \eta } } \\ & { \qquad + \left( \frac { \sigma ^ { 2 } } { B } + \kappa _ { 2 } ^ { 2 } \right) \eta E + \left( \frac { \sigma ^ { 2 } } { B } + \kappa _ { 2 } ^ { 2 } \right) \eta ^ { 2 } E ^ { 2 } \tau _ { \operatorname* { m a x } } ^ { 2 } \Bigg ) . } \end{array}
$$

The Proof can be founded in Technical Appendix $D$ .

From (14), it is evident that stragglers primarily affect the convergence performance through their impact on the bounded dissimilarity of the server-side model $\bf { \dot { \kappa } } _ { 1 } ^ { 2 }$ . Specifically, if there is a bias in the activations stored in the activation buffer, the bounded dissimilarity increases, leading to an increase in $\kappa _ { 1 } ^ { 2 }$ . This, in turn, enlarges the convergence upper bound in (14). According to Lemma 1, the proposed method achieves a tighter bounded dissimilarity by introducing generative activations. As a result, the server-side model attains a tighter upper bound, enhancing convergence performance. From (15), it is evident that stragglers primarily affect convergence performance through $\tau _ { \mathrm { m a x } } ^ { \mathrm { - 2 } }$ , which is multiplied by the learning rate $\eta$ . By setting a learning rate that decays over the global iterations, i.e., $\eta ^ { t } = \eta ^ { 0 } / \sqrt { t }$ , the impact of $\tau _ { \mathrm { m a x } } ^ { 2 }$ will be gradually mitigated as the training progresses.

# Experiments Implementation Details

Unless otherwise stated, the number of clients is set to 20, with 10 clients participating in each global iteration. Each client performs 20 local iterations with a learning rate of 0.01 and a minibatch size of 32. We use a linearly increasing weighting function, i.e., $s ( n ) = n$ and select AlexNet as the model architecture, where we set up the first 6 layers as the client-side model and the last 8 layers as the serverside model. To simulate a real-word communication environment, we consider a cell network with a radius of 1000 meters. The server is placed at the center of the network, with clients randomly and uniformly distributed within the cell. The path loss between each client and the server is modeled as $1 2 8 . 1 + 3 7 . 6 \log _ { 1 0 } ( r )$ dB, where $r$ is the distance from the client to the server in kilometers, according to (Abeta 2010). The client transmit power is uniformly set to $0 . 2 \mathrm { { W } }$ . We assume orthogonal uplink channel access with a total bandwidth $W = 1 0 \mathrm { M H z }$ and a power spectrum density of the additive Gaussian noise $N _ { 0 } = - 1 7 4 ~ \mathrm { d B m / H z }$ . Additionally, clients are assigned random computational capabilities, ranging between $\bar { 1 0 } ^ { 9 }$ and $1 0 ^ { 1 0 }$ FLOPs. More experimental details can be founded in Technical Appendix E.

# Baseline Settings

For the baseline comparison, we include both asynchronous and synchronous FL algorithms. The baseline asynchronous FL algorithms are FedBuff (Nguyen et al. 2022) and $\mathrm { C A ^ { 2 } F L }$ (Wang et al. 2024). FedBuff introduces a buffer mechanism to enable asynchronous FL, while $\mathrm { C A ^ { 2 } F L }$ builds on FedBuff by incorporating cached update calibration to enhance model performance in the presence of client data heterogeneity. Additionally, we select MiniBatch-SFL (Huang, Tian, and Tang 2023) and $\mathrm { \Delta S ^ { 2 } F L }$ (Yan et al. 2023) as baseline synchronous SFL algorithms. MiniBatch-SFL improves SFL performance by updating server-side model centrally, while $\mathrm { \Delta S ^ { 2 } F L }$ builds on MiniBatch-SFL by introducing adaptive model splitting and activation grouping strategies to address the stragglers issue.

# Dataset Settings

The datasets used for evaluation include CIFAR-10 (Krizhevsky 2009), CINIC-10 (Darlow et al. 2018), and Fashion-MNIST (Xiao, Rasul, and Vollgraf 2017). To simulate data heterogeneous, we employ both shard-based and distribution-based label skew methods (Zhang et al. 2022). The shard-based method involves sorting data by labels and dividing it into multiple shards. Each client receives a subset of these shards, resulting in training data with only a few labels for each client. We denote the data heterogeneity of this method by shard, where shard $= 2$ indicates each client has at most 2 types of data. This method represents an extreme form of data heterogeneity. In addition, the distributionbased label skew method allocates data to clients based on a Dirichlet distribution. Each client receives a proportion of samples from each label according to this distribution, resulting in a mix of majority and minority classes, and potentially some missing classes. We denote the data heterogeneity of this method by $\alpha$ , where $\operatorname { D i r } ( \alpha )$ indicates the Dirichlet distribution. The smaller the value of $\alpha$ , the higher the degree of data heterogeneity. This method better reflects realworld data heterogeneity.

# Validation of Theoretical Analysis

In this subsection, we validate our theoretical analysis by assessing gradient dissimilarity both without and with generative activations using the Fashion-MNIST datasets under heterogeneity conditions with shard $= 2$ . The total number of clients is set to 10, with 3 clients participating in each global iteration. The experimental results are depicted in Fig. 2. As shown in Fig. 2 (a), the gradient dissimilarity is reduced by introducing generative activations, thereby

3 Gradient Dissimilarity WithoGutenGernateirvaet vAec iAvcattiivoantisons 0.8 2 D 0.6 0.4 0.2 Without Generative Activations Huw With Generative Activations 0 0 0 200 400 600 800 1000 0 200 400 600 800 1000 Iterations Iterations (a) Gradient dissimilarity. (b) Test accuracy.

![](images/e4671af01d53589bd691b02ff715b6c743ac645239774ff5ae2857485d9eef72.jpg)  
Figure 2: Impact of generative activations on gradient dissimilarity and convergence performance.   
Figure 3: Test accuracy of GAS compared with the baseline methods on CIFAR-10 and CINIC-10.

confirming Lemma 1. This result demonstrates that the proposed method can achieve tighter bounded dissimilarity via the use of generative activations. Fig. 2 (b) further illustrates that the introduction of generative activations enhances convergence speed and achieve better accuracy, thus confirming Theorem 1. This indicates that tighter bounded dissimilarity reduces the upper bound of convergence rate, leading to superior convergence performance.

# Performance Evaluation

In this subsection, we evaluate the performance of the proposed method across different datasets and varying degrees of data heterogeneity. For Fashion-MNIST, CIFAR-10, and CINIC-10, we employ 1000, 2000, and 2000 global iterations. We first compare our method with baseline asynchronous FL algorithms. Each experiment is run with three random seeds, and the average accuracy and standard deviation are reported in Table 1. The experimental results demonstrate that the proposed method outperforms the baseline methods, particularly under conditions of extreme data heterogeneity. This improvement in model accuracy can be attributed to two key factors. First, the proposed method allows for centralized updates of the server-side model, significantly mitigating the issue of deep model drift caused by data heterogeneity (Luo et al. 2021). Second, by introducing generative activations, the proposed method alleviates the server-side model update bias introduced by stragglers, further enhancing model performance. Additionally, we compare our method with baseline synchronous SFL algorithms in a real-world communication environment, with results shown in Fig. 3. The experimental results indicate that the proposed method exhibits better convergence performance compared to baseline methods. This improvement is due to the asynchronous transmissions of activations and client-side models, which substantially reduce training time and achieves faster convergence speeds.

Table 1: Test accuracy $( \%$ ) on CIFAR-10, CINIC-10 and Fashion-MNIST.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="2">CIFAR-10</td><td colspan="4">CINIC-10</td><td colspan="2">Fashion-MNIST</td></tr><tr><td>s=2</td><td>α = 0.1</td><td>s=2</td><td>s=4</td><td>α = 0.1</td><td>α = 0.3</td><td>s=2</td><td>α =0.1</td></tr><tr><td>FedAvg</td><td>72.88±5.71</td><td>70.49±4.24</td><td>52.66±6.51</td><td>62.26±2.52</td><td>57.17±1.04</td><td>65.46±2.09</td><td>87.99±2.12</td><td>88.74±1.19</td></tr><tr><td>FedBuff</td><td>69.04±3.51</td><td>71.82±2.86</td><td>48.98±0.87</td><td>58.32±2.20</td><td>54.23±2.11</td><td>64.69±1.81</td><td>84.93±4.11</td><td>85.81±3.68</td></tr><tr><td>CA²FL</td><td>79.57±0.98</td><td>78.56±0.99</td><td>64.29±1.45</td><td>68.42±1.11</td><td>64.27±0.78</td><td>68.77±0.92</td><td>88.32±1.19</td><td>89.07±0.58</td></tr><tr><td>Ours</td><td>82.78±0.58</td><td>81.72±0.50</td><td>68.32±0.17</td><td>70.29±0.27</td><td>65.94±1.14</td><td>69.36±0.65</td><td>90.66±0.20</td><td>90.58±0.34</td></tr></table></body></html>

<html><body><table><tr><td rowspan="2">Method</td><td colspan="4">shard = 2</td><td colspan="4">α= 0.1</td></tr><tr><td>E=10</td><td>E=20</td><td>E=35</td><td>E=50</td><td>E=10</td><td>E=20</td><td>E=35</td><td>E=50</td></tr><tr><td>FedBuff</td><td>41.52±2.48</td><td>40.07±1.42</td><td>44.80±4.09</td><td>47.84±5.69</td><td>44.87±3.75</td><td>51.18±1.74</td><td>54.03±5.01</td><td>55.77±3.58</td></tr><tr><td>CA²FL</td><td>58.77±0.52</td><td>62.12±1.39</td><td>63.14±0.28</td><td>63.31±1.18</td><td>58.39±1.15</td><td>60.95±0.84</td><td>62.02±2.39</td><td>62.73±1.16</td></tr><tr><td>Ours</td><td>63.07±0.08</td><td>65.58±0.71</td><td>65.09±1.10</td><td>62.40±1.97</td><td>60.68±1.13</td><td>63.39±2.01</td><td>63.12±2.29</td><td>61.96±3.50</td></tr></table></body></html>

Table 2: Test accuracy $( \% )$ under different number of local iterations.

![](images/9d1c213f7b7e65baf4398d0875750e917a1c1c1be8a5e5c6fcd9b579996b1e67.jpg)  
Figure 4: Impact of local iterations on the performance of GAS compared to baseline methods.

# Ablation Study on Local Iterations

In this subsection, we conduct an ablation study on the number of local iterations. Unlike FL frameworks, GAS requires the additional transmissions of activations, which are influenced by the number of local iterations. Therefore, we study the impact of different local iteration settings under the real-world communication environment. We conduct experiments with local iteration settings of 10, 20, 35 and 50, while fixing the number of global iterations at 1000. The results are presented in Table 2 and Fig. 4. As shown in Table 2, the accuracy of GAS increases with the number of local iterations initially but decreases thereafter. This indicates that the number of local iterations must be carefully chosen to balance model accuracy and communication load. On one hand, a higher number of local iterations is necessary for sufficient local training. On the other hand, setting the number of local iterations too high can lead to local optima and increased communication load. Additionally, we observe that the accuracy of the baseline methods increase with the number of local iterations. This suggests that the baseline methods do not achieve sufficient training within the given local iteration settings. Even with a local iteration setting of 50, the accuracy of $\mathrm { C A ^ { 2 } F L }$ remains lower than that of GAS with a local iteration setting of 20, indicating the higher training efficiency of GAS.

From Fig. 4, it is evident that GAS demonstrates faster convergence and higher accuracy compared to the baseline methods at the lower local iteration setting ( $\ E = 2 0 ^ { \cdot }$ ). This highlights the significant advantage of GAS in real-world communication environments. Note that although $\mathrm { C A ^ { 2 } F L }$ performs well with $E = 5 0$ , it incurs higher computational load due to the increased number of local iterations. Specifically, $\mathrm { C A ^ { 2 } F L }$ takes 60 minutes to achieve $6 0 \%$ accuracy, whereas GAS with $E = 2 0$ achieves the same accuracy in just 30 minutes.

# Conclusion

In this paper, we proposed GAS (Generative activationaided Asynchronous SFL), a distributed asynchronous learning framework designed to address the stragglers issue in SFL. By employing an activation buffer and a model buffer, along with generative activation-aided updates, GAS effectively mitigated the impact of stragglers and improved model convergence. Our theoretical analysis and experimental results demonstrated that GAS achieved higher accuracy and faster convergence compared to baseline FL and SFL methods.

Limitations: Like other SL and SFL algorithms, GAS requires the transmission of labels and activations, which poses a risk of privacy leaks. Incorporating privacypreserving mechanisms of SFL (Xiao, Yang, and Wu 2021; Li et al. 2022) into GAS to enhance data security and broaden its applicability is a promising direction for future work.