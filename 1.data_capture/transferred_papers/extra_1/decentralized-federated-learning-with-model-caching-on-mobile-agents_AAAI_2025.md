# Decentralized Federated Learning with Model Caching on Mobile Agents

Xiaoyu Wang1, Guojun Xiong2, Houwei Cao3, Jian $\mathbf { L i } ^ { 2 }$ , Yong Liu1

1New York University 2Stony Brook University 3New York Institute of Technology {wang.xiaoyu, yongliu}@nyu.edu, {guojun.xiong, jian.li.3}@stonybrook.edu, hcao02 $@$ nyit.edu

# Abstract

Federated Learning (FL) trains a shared model using data and computation power on distributed agents coordinated by a central server. Decentralized FL (DFL) utilizes local model exchange and aggregation between agents to reduce the communication and computation overheads on the central server. However, when agents are mobile, the communication opportunity between agents can be sporadic, largely hindering the convergence and accuracy of DFL. In this paper, we propose Cached Decentralized Federated Learning (Cached-DFL) to investigate delay-tolerant model spreading and aggregation enabled by model caching on mobile agents. Each agent stores not only its own model, but also models of agents encountered in the recent past. When two agents meet, they exchange their own models as well as the cached models. Local model aggregation utilizes all models stored in the cache. We theoretically analyze the convergence of Cached-DFL, explicitly taking into account the model staleness introduced by caching. We design and compare different model caching algorithms for different DFL and mobility scenarios. We conduct detailed case studies in a vehicular network to systematically investigate the interplay between agent mobility, cache staleness, and model convergence. In our experiments, Cached-DFL converges quickly, and significantly outperforms DFL without caching.

Code — https://github.com/ShawnXiaoyuWang/Cached-DFL Extended version — https://arxiv.org/abs/2408.14001

# Introduction Federated Learning on Mobile Agents

Federated learning (FL) is a type of distributed machine learning (ML) that prioritizes data privacy (McMahan et al. 2017). The traditional FL involves a central server that connects with a large number of agents. The agents retain their data and do not share them with the server. During each communication round, the server sends the current global model to the agents, and a small subset of agents are chosen to update the global model by running stochastic gradient descent (SGD) (Robbins and Monro 1951) for multiple iterations on their local data. The central server then aggregates the updated parameters to obtain the new global model. FL naturally complements emerging Internet-of-Things (IoT) systems, where each IoT device not only can sense its surrounding environment to collect local data, but also is equipped with computation resources for local model training, and communication interfaces to interact with a central server for model aggregation. Many IoT devices are mobile, ranging from mobile phones, autonomous cars/drones, to self-navigating robots. In recent research efforts on smart connected vehicles, there has been a focus on integrating vehicle-to-everything (V2X) networks with Machine Learning (ML) tools and distributed decision making (Barbieri et al. 2022), particularly in the area of computer vision tasks such as traffic light and signal recognition, road condition sensing, intelligent obstacle avoidance, and intelligent road routing, etc. With FL, vehicles locally train deep ML models and upload the model parameters to the central server. This approach not only reduces bandwidth consumption, as the size of model parameters is much smaller than the size of raw image/video data, but also leverages computing power on vehicles, and protects user privacy.

However, FL on mobile agents still faces communication and computation challenges. The movements of mobile agents, especially at high speed, lead to fast-changing channel conditions on the wireless connections between mobile agents and the central server, resulting in high latency in FL (Niknam, Dhillon, and Reed 2020). Battery-powered mobile agents also have limited power budget for longrange wireless communications. Non-i.i.d data distributions on mobile agents make it difficult for local models to converge. As a result, FL on mobile agents to obtain an optimal global model remains an open challenge. Decentralized FL (DFL) has emerged as a potential solution where local model aggregations are conducted between neighboring mobile agents using local device-to-device (D2D) communications with high bandwidth, low latency and low power consumption (Mart´ınez Beltra´n et al. 2023). Preliminary studies have demonstrated that DFL algorithms have the potential to significantly reduce the high communication costs associated with centralized FL. However, blindly applying model aggregation algorithms, such as FedAvg (McMahan et al. 2017), developed for centralized FL to DFL cannot achieve fast convergence and high model accuracy (Liu et al. 2022).

# Delay-Tolerant Model Communication and Aggregation Through Caching

D2D model communication between a pair of mobile agents is possible only if they are within each other’s transmission ranges. If mobile agents only meet with each others sporadically, there will not be enough model aggregation opportunity for fast convergence. In addition, with non-i.i.d data distributions on agents, if an agent only meets with agents from a small cluster, there is no way for the agent to interact with models trained by data samples outside of its cluster, leading to disaggregated local models that cannot perform well on the global data distribution. It is therefore essential to achieve fast and even model spreading using limited D2D communication opportunities among mobile agents. A similar problem was studied in the context of Mobile Ad hoc Network (MANET), where wireless communication between mobile nodes are sporadic. The efficiency of data dissemination in MANET can be significantly improved by Delay-Tolerant Networking (DTN) (Fall 2003; Burleigh et al. 2003): a mobile node caches data it received from nodes it met in the past; when meeting with a new node, it not only transfers its own data, but also the cached data of other nodes. Essentially, node mobility forms a new “communication” channel through which cached data are transported through node movement in physical space. It is worth noting that, due to multi-hop caching-and-relay, DTN transmission incurs longer delay than D2D direct transmission. Data staleness can be controlled by caching and relay algorithms to match the target application’s delay tolerance such as Li et al. (2023).

Motivated by DTN, we propose delay-tolerant DFL communication and aggregation enabled by model caching on mobile agents. To realize DTN-like model spreading, each mobile agent stores not only its own local model, but also local models received from other agents in the recent history. Whenever it meets another agent, it transfers its own model as well as the cached models to the agent through highspeed D2D communication. Local model aggregation on an agent works on all its cached models, mimicking a local parameter server. Compared with DFL without caching, DTNlike model spreading can push local models faster and more evenly to the whole network; aggregating all cached models can facilitate more balanced learning than pairwise model aggregation. While DFL model caching sounds promising, it also faces a new challenge of model staleness: a cached model from an agent is not the current model on that agent, with the staleness determined by the mobility patterns, as well as the model spreading and caching algorithms. Using stale models in model aggregation may slow down or even deviate model convergence.

The key challenge we want to address in this paper is how to design cached model spreading and aggregation algorithms to achieve fast convergence and high accuracy in DFL on mobile agents. Towards this goal, we make the following contributions:

1. We develop Cached-DFL, a new DFL framework that utilizes model caching on mobile agents to realize delaytolerant model communication and aggregation;

Table 1: Notations and Terminologies.   

<html><body><table><tr><td>Notation</td><td>Description</td></tr><tr><td>N</td><td>Number of agents</td></tr><tr><td>T</td><td>Numberof global epochs</td></tr><tr><td>[N]</td><td>Set of integers {1,.., N}</td></tr><tr><td>K</td><td>Number oflocal updates</td></tr><tr><td>xi(t)</td><td>Model in the tth epoch on agent i</td></tr><tr><td>x(t)</td><td>Global Meinc(t</td></tr><tr><td>xi(t,k)</td><td>Mode iniaaliedromter</td></tr><tr><td>xi(t)</td><td>Model xi(t) after local updates</td></tr><tr><td>D</td><td>Dataset on the i-th agent</td></tr><tr><td>a</td><td>Aggregation weight</td></tr><tr><td>t-T</td><td>Staleness</td></tr><tr><td>Tmax</td><td>Tolerance of staleness incache</td></tr><tr><td></td><td>All the norms in the paper are l2-norms</td></tr></table></body></html>

2. We theoretically analyze the convergence of aggregation with cached models, explicitly taking into account the model staleness;   
3. We design and compare different model caching algorithms for different DFL and mobility scenarios.   
4. We conduct a detailed case study on vehicular network to systematically investigate the interplay between agent mobility, cache staleness, and convergence of model aggregation. Our experimental results demonstrate that our Cached-DFL converges quickly and significantly outperforms DFL without caching.

# Mobile DFL with Model Caching Global Training Objective

Similar to the standard FL problem, the overall objective of mobile DFL is to learn a single global statistical model from data stored on tens to potentially millions of mobile agents. The overall goal is to find the optimal model weights $x ^ { \ast } \in$ $\mathbb { R } ^ { d }$ to minimize the global loss function:

$$
\operatorname* { m i n } _ { x } F ( x ) , { \mathrm { w h e r e } } F ( x ) = { \frac { 1 } { N } } \sum _ { i \in [ N ] } \mathbb { E } _ { z ^ { i } \sim \mathcal { D } ^ { i } } f ( x ; z ^ { i } ) ,
$$

where $N$ denotes the total number of mobile agents, and each agent has its own local dataset, i.e., $\mathcal { D } ^ { i } \neq \bar { \mathcal { D } ^ { j } } , \forall i \neq j$ . And $z ^ { i }$ is sampled from the local data $\mathcal { D } ^ { i }$ .

# DFL Training with Local Model Caching

All agents participate in DFL training over $T$ global epochs. At the beginning of the $t ^ { t h }$ epoch, agent $i$ ’s local model is $x _ { i } ( t )$ . After $K$ steps of SGD to solve the following optimization problem with a regularized loss function:

$$
\operatorname* { m i n } _ { x } \mathbb { E } _ { z ^ { i } \sim D ^ { i } } f ( x ; z ^ { i } ) + \frac { \rho } { 2 } | | x - x _ { i } ( t ) | | ^ { 2 } ,
$$

agent $i$ obtains an updated local model $\tilde { x } _ { i } ( t )$ . Meanwhile, during the $t ^ { t h }$ epoch, driven by their mobility patterns, each agent meets and exchanges models with other agents. Other than its own model, agent $\mathbf { \chi } _ { i }$ also stores models it received from other agents encountered in the recent history in its local cache $\mathcal { C } _ { i } ( t )$ . When two agents meet, they not only exchange their own local models, but also share their cached models with each others to maximize the efficiency of DTNlike model spreading. The models received by agent $i$ will be used to update its model cache $\mathcal { C } _ { i } ( t )$ , using different cache update algorithms, such as LRU update method (Algorithm 2) or Group-based LRU update method, which will be described in details later. As the cache size of each agent is limited, it is important to design an efficient cache update rule in order to maximize the caching benefit.

Algorithm 1: Cached Decentralized Federated Learning   

<html><body><table><tr><td>(Cached-DFL)</td></tr><tr><td>Input:Global epochs T,local updatesK,initial models {𝑥i(0)}=1,stalenesstolerance Tmax</td></tr><tr><td>1: function LOCALUPDATE(xi(t))</td></tr><tr><td>2: Initialize:xi(t,O)= xi(t)</td></tr><tr><td>3: Define: gx(t)(x; z)= f(x; z) + ≌|x-x(t)ll²</td></tr><tr><td>4: for k = 1,2,. .,K do</td></tr><tr><td>5: Randomly sample zi ~ Di 6: xi(t,k)= xi(t,k-1)-nVgxi(t)(xi(t,k-</td></tr><tr><td>1);2）</td></tr><tr><td>7: end for 8: return 𝑥i(t)= xi(t,K)</td></tr><tr><td>9:end function</td></tr><tr><td></td></tr><tr><td>10:function MODELAGGREGATION(Ci(t)) 11: xi(t+1)=∑j∈C.(t)αjx;(T)</td></tr><tr><td>12: return xi(t + 1) 13:end function</td></tr><tr><td>MainProcess:</td></tr><tr><td>14:fort=0,1,...,T-1do</td></tr><tr><td>15: fori=1,2,...,N do</td></tr><tr><td>16: xi(t)←LOCALUPDATE(xi(t))</td></tr><tr><td>17: Ci(t)← CACHEUPDATE(Ci(t-1), Tmax) 18: xi(t+1) ← MODELAGGREGATION(Ci(t))</td></tr></table></body></html>

After cache updating, each agent conducts local model aggregation using all the cached models with customized aggregation weights $\{ \alpha _ { j } ~ \in ~ ( 0 , 1 ) \}$ to get the updated local model $x _ { i } ( t + 1 )$ for epoch $t + 1$ . In our simulation, we take the aggregation weight as $\boldsymbol { \alpha _ { j } } = \left( \boldsymbol { n _ { j } } / \sum _ { j \in \mathcal { C } _ { i } ( t ) } \boldsymbol { n _ { j } } \right)$ , where $n _ { j }$ is the number of samples on agent $j$ .

The whole process repeats until the end of $T$ global epochs. The detailed algorithm is shown in Algorithm 1. $z _ { k } ^ { i }$ are randomly drawn local data samples on agent $\mathbf { \chi } _ { i }$ for the $k$ -th local update, and $\eta$ is the learning rate.

Remark 1. Note the $N$ agents communicate with each others in a mobile $D 2 D$ network. $D 2 D$ communication can only happen between an agent and its neighbors within $a$ short range (e.g. several hundred meters). Since agent locations are constantly changing (for instance, vehicles continuously move along the road network of a city), D2D network topology is dynamic and can be sparse at any given epoch. To ensure the eventual model convergence, the union graph of D2D networks over multiple epochs should be strongly-connected for efficient DTN-like model spreading. We also assume D2D communications are non-blocking, carried by short-distance high-throughput communication methods such as mmWave or WiGig, which has enough capacity to complete the exchange of cached models before the agents go out of each other’s communication ranges.

Remark 2. Intuitively, comparing to the DFL without cache (e.g. DeFedAvg (Sun, Li, and Wang 2022)), where each agent can only get new model by averaging with another model, Cached-DFL uses more models (delayed versions) for aggregation, thus utilizes more underlying information from datasets on more agents. Although Cached-DFL introduces stale models, it can benefit model convergence, especially in highly heterogeneous data distribution scenarios. Overall, our Cached-DFL framework allows each agent to act as a local proxy for Centralized FL with delayed cached models, thus speedup the convergence especially in highly heterogeneous data distribution scenarios, that are challenging for the traditional DFL to converge.

Remark 3. As mentioned above, our approach inevitably introduces stale models. Intuitively, larger staleness results in greater error in the global model. For the cached models with large staleness $t - \tau$ , we could set a threshold $\tau _ { m a x }$ to kick old models out of model spreading, which is described in the cache update algorithms. The practical value for $\tau _ { m a x }$ should be related to the cache capacity and communication frequency between agents. In our experimental results, we choose $\tau _ { m a x }$ to be $I O$ or 20 epochs. Results and analysis about the effect of different $\tau _ { m a x }$ can be found in experimental results section.

# Convergence Analysis

We now theoretically investigate the impact of caching, especially the staleness of cached models, on DFL model convergence. We introduce some definitions and assumptions.

Definition 1 (Smoothness). A differentiable function $f$ is $L$ - smooth if for $\forall x$ , $y$ , $\begin{array} { r } { f ( y ) - f ( x ) \leq \langle \nabla f ( x ) , y - x \rangle + \frac { L } { 2 } | | y - } \end{array}$ $x | | ^ { 2 }$ , where $L > 0$ .

Definition 2 (Bounded Variance). There exists constant $\varsigma >$ 0 such that the global variability of the local gradient of the loss function is bounded $| | \nabla F _ { j } \bar { ( } x \bar { ) } - \nabla F ( x ) \bar { | } | ^ { 2 } \leq \varsigma ^ { 2 } , \bar { \forall } j \in$ $[ N ] , x \in \mathbb { R } ^ { d }$ .

Theorem 1. Assume that $F$ is $L$ -smooth and convex, and each agent executes $K$ local updates before meeting and exchanging models, after that, then does model aggregation. We also assume bounded staleness $\tau < \tau _ { m a x }$ , as the kick-out threshold. Furthermore, we assume, $\forall x \in \mathbb { R } ^ { d }$ , $i \in [ N ]$ , and $\forall z \sim \mathcal { D } ^ { i } , | | \nabla f ( x ; z ) | | ^ { 2 } \leq V , | | \nabla g _ { x ^ { \prime } } ( x ; z ) | | ^ { 2 } \leq \dot { V } , \dot { \forall } x ^ { \prime } \in$ $\mathbb { R } ^ { d }$ . For any small constant $\epsilon > 0$ , if we take $\rho > 0$ , and satisfying $\begin{array} { r } { - ( \mathrm { i } + 2 \rho + \epsilon ) V + ( \rho ^ { 2 } - \frac { \rho } { 2 } ) | | x ( t , k - 1 ) - x ( t ) | | ^ { 2 } \geq } \end{array}$

$0 , \forall x ( t , k - 1 ) , x ( t )$ , after $T$ global epochs, Cached-DFL converges to a critical point:

$$
\begin{array} { r l r } {  { \frac { T - 1 } { t = 0 } \mathbb { E } \| \nabla F ( x ( t ) ) \| ^ { 2 } \leq \frac { \tau _ { m a x } \mathbb { E } [ F ( x ( 0 ) ) - F ( x _ { M ( T ) } ( T ) ) ] } { \epsilon \eta C _ { 1 } K T } } } \\ & { + O ( \frac { \eta \rho K ^ { 2 } } { \epsilon C _ { 1 } } ) \leq \mathcal { O } ( \frac { \tau _ { m a x } } { \epsilon \eta C _ { 1 } K T } ) + \mathcal { O } ( \frac { \eta \rho K ^ { 2 } } { \epsilon C _ { 1 } } ) . } & { ( 2 ) } \end{array}
$$

# Proof Sketch

We now highlight the key ideas and challenges behind our convergence proof.

Step 1: Similar to Theorem 1 in Xie, Koyejo, and Gupta (2019), we bound the expected cost reduction after $K$ steps of local updates on the $j$ -th agent, $\forall j \in [ N ]$ , as

$$
\begin{array} { r l } & { \mathbb { E } [ F ( \tilde { x } _ { j } ( t ) ) - F ( x _ { j } ( t ) ) ] = \mathbb { E } [ F ( x _ { j } ( t , K ) ) - F ( x _ { j } ( t , 0 ) ) ] } \\ & { \leq - \eta \epsilon \displaystyle \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } \vert \vert \nabla F ( x _ { j } ( t , k ) ) \vert \vert ^ { 2 } + \eta ^ { 2 } \mathcal { O } ( \rho K ^ { 3 } V ) . } \end{array}
$$

Step 2: For any epoch $t$ , we find the index $M ( t )$ of the agent whose model is the “worst”, i.e., $\begin{array} { r l } { M ( t ) } & { { } = } \end{array}$ arg $\operatorname* { m i x } _ { j \in [ N ] } \{ F ( x _ { j } ( t ) ) \}$ , and the “worst” model on all agents over the time period of $\left[ t - \tau _ { m a x } + 1 , t \right]$ as

$$
\mathcal { T } ( t , \tau _ { m a x } ) = \arg \operatorname* { m a x } _ { t \in [ t - \tau _ { m a x } + 1 , t ] } \{ F ( x _ { M ( t ) } ( t ) ) \} .
$$

Step 3: We bound the cost reduction of the “worst” model at epoch $t + 1$ from the “worst” model in the time period of $\left[ t - \tau _ { m a x } + 1 , t \right]$ , i.e., the worst possible model that can be stored in some agent’s cache at time $t$ , as:

$$
\begin{array} { r l } & { \mathbb { E } \bigl [ F \left( x _ { M ( t + 1 ) } ( t + 1 ) \right) - F \left( x _ { M ( \mathcal { T } ( t , \tau _ { m a x } ) ) } ( \mathcal { T } ( t , \tau _ { m a x } ) ) \right) \bigr ] } \\ & { \quad \leq - \epsilon \eta C _ { 1 } K \underset { \tau = t } { \textup { m i n } } \underset { | | \nabla F ( x ( \tau ) ) | | ^ { 2 } + \eta ^ { 2 } \mathcal { O } ( \rho K ^ { 3 } V ) . \quad ( 3 ) } } \end{array}
$$

Step 4: We iteratively construct a time sequence $\{ T _ { 0 } ^ { \acute { \prime } } , T _ { 1 } ^ { \prime } , T _ { 2 } ^ { \prime } , . . . , T _ { N _ { T } } ^ { \prime } \} \subseteq \{ 0 , 1 , . . . , T - 1 \}$ in the backward fashion so that

$$
\begin{array} { r l } { T _ { N _ { T } } ^ { \prime } } & { { } = T - 1 ; } \\ { T _ { i } ^ { \prime } } & { { } = \mathcal { T } ( T _ { i + 1 } ^ { \prime } , \tau _ { m a x } ) - 1 , \quad 1 \leq i \leq N _ { T } - 1 ; } \\ { T _ { 0 } ^ { \prime } } & { { } = 0 . } \end{array}
$$

Step 5: Applying inequality (3) at all time instances $\{ T _ { 0 } ^ { \bar { \prime } } , T _ { 1 } ^ { \prime } , T _ { 2 } ^ { \prime } , . . . , \bar { T } _ { N _ { T } } ^ { \prime } \}$ , after $\mathrm { \Delta T }$ global epochs, we have,

$$
\begin{array} { r l } & { \frac { T _ { \gamma _ { - 1 } } } { \operatorname* { m i n } } \mathbb { E } \| | \nabla F ( x ( t ) ) | | ^ { 2 } | \le \frac { T _ { \gamma _ { - 1 } } ^ { \gamma _ { - 1 } } } { N _ { T } } \underbrace { \overbrace { \mathrm {  { \Lambda } } ^ { \mathrm { T } } \mathrm {  { \Lambda } } ^ { \mathrm { T } } } ^ { T _ { \gamma _ { - 1 } } } \ldots \overbrace { \mathrm {  { \Lambda } } ^ { \mathrm { T } } \mathrm {  { \Lambda } } ^ { \mathrm { T } } } ^ { \mathrm { T } } } _ { \le \gamma _ { 0 } ^ { T } } \| \nabla F ( x ( \tau ) ) \| ^ { 2 } } \\ & { \le \frac { 1 } { \epsilon \eta C _ { 1 } K N _ { T } } \underbrace { T _ { \gamma _ { - 1 } } ^ { \gamma _ { - 1 } } } _ { = T _ { 0 } ^ { \epsilon } } \mathbb { E } \big [ F ( x _ { M } ( \tau ( \tau _ { \mathfrak { t } } , \tau _ { \mathrm { m a x } } ) ) ( \mathcal { T } ( t , \tau _ { \mathrm { m a x } } ) ) ) } \\ & { \quad - F ( x _ { M ( \mathfrak { t } } + 1 ) ) \big ] + { \mathcal O } ( \frac { \eta K ^ { 2 } V } { \epsilon C _ { 1 } } ) } \\ & { \le \frac { \mathbb { E } \big [ F ( x ( 0 ) ) - F ( x _ { M ( T ) } ( T ) ) \big ] } { \epsilon \eta C _ { 1 } K N _ { T } } + { \mathcal O } ( \frac { \eta K ^ { 2 } V } { \epsilon C _ { 1 } } ) } \\ & { \le { \mathcal O } ( \frac { \eta } { \epsilon \eta C _ { 1 } K _ { T } } ) + { \mathcal O } ( \frac { \eta \theta K ^ { 2 } } { \epsilon C _ { 1 } } ) . } \end{array}
$$

Step 6: With the results in (4) and by leveraging Theorem 1 in Yang, Fang, and Liu (2021), Cached-DFL converges to a critical point after $T$ global epochs.

![](images/71228d13a9a4e8f6b1826703e17e2ccf4f03ba9a7da904b0885dc4419109e90f.jpg)  
Figure 1: Manhattan Mobility Model Map. The dots represent the intersections while the edges between nodes represent roads in Manhattan.

# Experiments

We implement Cached-DFL with PyTorch (Paszke et al. 2017) on Python3. Further details about experiments, hyperparameters, and additional results are provided in our technical report Wang et al. (2024).

Datasets. We conduct experiments using three standard FL benchmark datasets: MNIST (Deng 2012), FashionMNIST (Xiao, Rasul, and Vollgraf 2017), CIFAR-10 (Krizhevsky 2009) on $N = 1 0 0$ vehicles, with CNN, CNN and ResNet18 (He et al. 2016) as models respectively. We evaluate three different data distribution settings: non-i.i.d, i.i.d, and Dirichlet. In extreme non-i.i.d, we use a setting similar to Su, Zhou, and Cui (2022), data points in training set are sorted by labels and then evenly divided into 200 shards, with each shard containing 1–2 labels out of 10 labels. Then 200 shards are randomly assigned to 100 vehicles unevenly: $10 \%$ vehicles receive 4 shards, $20 \%$ vehicles receive 3 shards, $30 \%$ vehicles receive 2 shards and the rest $40 \%$ receive 1 shard. For i.i.d, we randomly allocate all the training data points to 100 vehicles. For Dirichlet distribution, we follow the setting in Xiong et al. (2024), to take a heterogeneous allocation by sampling $p _ { i } \sim D i r _ { N } ( \pi )$ , where $\pi$ is the parameter of Dirichlet distribution. We take $\pi = 0 . 5$ in our following experiments.

Evaluation Setup. The baseline algorithm is DeFedAvg (Sun, Li, and Wang 2022), which implements simple decentralized federated optimization. For convenience, we name DeFedAvg as DFL in the following results. We set batch size to 64 in all experiments. For MNIST and FashionMNIST, we use 60k data points for training and 10k data points for

Input: Current cache $\mathcal { C } _ { i } ( t )$ , agent $j$ ’s cache $\mathscr { C } _ { j } ( t )$ , model $x _ { j } ( t )$ from agent $j$ , current time $t$ , cache size $\mathcal { C } _ { \mathrm { m a x } }$ , staleness tolerance τmax

Main Process:   
1: for each $x _ { k } ( \tau ) \in \mathcal { C } _ { i } ( t )$ or $\mathcal { C } _ { j } ( t )$ do   
2: if $t - \tau \geq \tau _ { \operatorname* { m a x } }$ then   
3: Remove $x _ { k } ( \tau )$ from the respective cache $( \mathcal { C } _ { i } ( t )$   
or $\mathcal { C } _ { j } ( t ) \big |$ )   
4: end if   
5: end for   
6: Add or replace $x _ { j } ( t )$ into $\mathcal { C } _ { i } ( t )$   
7: for each $x _ { k } ( \tau ) \in \mathcal { C } _ { j } ( t )$ do   
8: if $x _ { k } ( \tau ) \notin \mathcal { C } _ { i } ( t )$ then   
9: Add $x _ { k } ( \tau )$ into $\mathcal { C } _ { i } ( t )$   
10: else   
11: Retrieve $x _ { k } ( \tau ^ { \prime } ) \in \mathcal { C } _ { i } ( t )$   
12: if $\tau > \tau ^ { \prime }$ then   
13: Replace $x _ { k } ( \tau ^ { \prime } )$ with $x _ { k } ( \tau )$ in $\mathcal { C } _ { i } ( t )$   
14: end if   
15: end if   
16: end for   
17: Sort models in $\mathcal { C } _ { i } ( t )$ in descending order of $\tau$   
18: Retain only the first $\mathcal { C } _ { \mathrm { m a x } }$ models in $\mathcal { C } _ { i } ( t )$   
19: return $\mathcal { C } _ { i } ( { \dot { t } } + 1 )$   
Output: $\mathcal { C } _ { i } ( t + 1 )$

testing. For CIFAR-10, we use 50k data points for training and 10k data points for testing. Different from training set partition, we do not split the testset. For MNIST and FashionMNIST, we test local models of 100 vehicles on the $1 0 \mathbf { k }$ data points of the whole test set and get the average test accuracy for the evaluation metric. What’s more, for CIFAR10, due to the computing overhead, we sample 1,000 data points from test set for each vehicle and use the average test accuracy of 100 vehicles as the evaluation metric. For all the experiments, we train for 1,000 global epochs, and implement early stop when the average test accuracy stops increasing for at least 20 epochs. For MNIST and FashionMNIST experiments, we use 10 compute nodes, each with 10 CPUs, to simulate DFL on 100 vehicles. CIFAR-10 results are obtained from 1 compute node with 5 CPUs and 1 A100 NVIDIA GPU.

Optimization Method. We use SGD as the optimizer and set the initial learning rate $\eta = 0 . 1$ , and use learning rate scheduler named ReduceLROnPlateau from PyTorch, to automatically adjust the learning rate for each training.

Mobile DFL Simulation. Manhattan Mobility Model maps are derived from real Manhattan road data (INRIX (INRIX 2024)), as shown in Fig. 1. Following Bai, Sadagopan, and Helmy (2003), vehicles move along a grid of horizontal and vertical streets, turning left, right, or going straight at intersections according to specified probabilities (e.g., 0.5 to continue straight and 0.1667 per road among three options). As in Su, Zhou, and Cui (2022), each vehicle is equipped with DSRC and mmWave, can communicate within $1 0 0 ~ \mathrm { { m } }$ , and travels at $1 3 . 8 9 \mathrm { m / s }$ . We set the number of local updates to

![](images/162d436d9f58f5e58d47edbd7b1c1af4cbb3bd67e9a5852fc92b3dbc8f81129b.jpg)  
Figure 2: Cached-DFL vs. DFL without Caching across Different Datasets: The first row presents results for MNIST, the second row for FashionMNIST, and the third row for CIFAR-10.

$K = 1 0$ , with a 120-second interval between global epochs. During each epoch, vehicles train their models while moving and exchange models when they encounter each other.

LRU Cache Update. Algorithm 2, which we name as LRU method for convenience, is the basic cache update method we proposed. Basically, the LRU updating rule aims to fetch the most recent models and keep as many of them in the cache as possible. At lines 12 and 17, the metric for making choice among models is the timestamp of models, which is defined as the epoch time when the model was received from the original vehicle, rather than received from the cache. What’s more, to fully utilize the caching mechanism, a vehicle not only fetches other vehicles’ own trained models, but also models in their caches. For instance, at epoch $t$ , vehicle $i$ can directly fetch model $x _ { j } ( t )$ from vehicle $j$ , at line 6, and also fetch models $x _ { k } ( \tau ) \in \overline { { \mathscr { C } _ { j } ( t ) } }$ from the cache of vehicle $j$ , at lines 7-16. This way, each vehicle can not only directly fetch its neighbors’ models, but also indirectly fetch models of its neighbors’ neighbors, thus boosting the spreading of the underlying data information from different vehicles, and improving the DFL convergence speed, especially with heterogeneous data distribution. Additionally, at lines 1-5, before updating cache, models with staleness $t - \tau \geq \tau _ { m a x }$ will be removed from each vehicle’s cache.

# Experimental Results

Caching vs. Non-caching. To evaluate the performance of Cached-DFL, we compare the DFL with LRU Cache, Centralized FL (CFL) and DFL on MNSIT, FashionMNIST, CIFAR-10 with three distributions: non-i.i.d, i.i.d, Dirichlet with $\pi = 0 . 5$ . For LRU update, we take the cache size as 10 for MNIST and FashionMNIST, and 3 for CIFAR-10 and $\tau _ { m a x } = 5$ based on practical considerations. Given a speed of $1 3 . 8 9 m / s$ and and a communication distance of $1 0 0 m$ , the communication window of two agents driving in opposite directions could be limited. Additionally, the above cache sizes were chosen by considering the size of chosen models and communication overhead. From the results in Fig. 2, we can see that Cached-DFL boosts the convergence and outperforms non-caching method (DFL) and gains performance much closer to CFL in all the cases, especially in non-i.i.d. scenarios.

![](images/ef52e2e639a0b45e662340091a60237f6f07e2d04795740e0c04fc57f232981c.jpg)  
Figure 3: Cached-DFL with LRU at Different Cache Sizes: The first row presents results for MNIST, while the second row corresponds to FashionMNIST.

Impact of Cache Size. Then we evaluate the performance gains with different cache sizes from 1 to 30 and $\tau _ { m a x } = 1 0$ , on MNIST and FashionMNIST in Fig. 3. LRU can benefit more from larger cache sizes, especially in non-i.i.d scenarios, as aggregation with more cached models gets closer to CFL and training with global data distribution.

Impact of Model Staleness. One drawback of model caching is introducing stale models into aggregation, so it is very important to choose a proper staleness tolerance $\tau _ { m a x }$ . First, we statistically calculate the relation between the average number and the average age of cached models at different $\tau _ { m a x }$ from 1 to 20, when epoch time is 30s, 60s, 120s, with unlimited cache size, in Table 2. We can see that, with the fixed epoch time, as the $\tau _ { m a x }$ increases, the average number and average age of cached models increase, approximately linearly. It’s not hard to understand, as every epoch each agent can fetch a limited number of models directly from other agents. Increasing the staleness tolerance $\tau _ { m a x }$ will increase the number of cached models, as well as the age of cached models. What’s more, we can see that the communication frequency or the moving speed of agents will also impact the average age of cached models, as the faster an agent moves, the more models it can fetch within each epoch, which we will further discuss later.

Mobility’s Impact on Convergence. Fig. 4 compares DFL and LRU at different $\tau _ { \mathrm { m a x } }$ on MNIST under non-i.i.d and i.i.d settings. Here we pick the epoch time 30s for a clear view. Under non-i.i.d, a larger $\tau _ { \mathrm { m a x } }$ initially speeds up convergence by allowing more cached models (confirming our earlier findings), as it allows for more cached models which bring more benefits than harm to the training with non-i.i.d. Under i.i.d, however, more cached models bring no gain and staleness hinders convergence. Moreover, focusing on the final phase (bottom of each figure), higher $\tau _ { \mathrm { m a x } }$ reduces final accuracy in both scenarios due to staleness. Despite this, with high $\tau _ { \mathrm { m a x } }$ , LRU can still match or outperform DFL in non-i.i.d settings.

Table 2: Average number and average age of cached models with different $\tau _ { m a x }$ (columns) and different epoch times: 30s, 60s, 120s (rows). Each row has two sub-rows: the first shows the average number of cached models, and the second shows their average age.   

<html><body><table><tr><td>Tmax</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>10</td><td>20</td></tr><tr><td rowspan="2">30s</td><td>0.9</td><td>1.7</td><td>2.7</td><td>3.9</td><td>5.2</td><td>14.9</td><td>44.8</td></tr><tr><td>0.0</td><td>0.5</td><td>1.0</td><td>1.6</td><td>2.2</td><td>5.4</td><td>11.6</td></tr><tr><td rowspan="2">60s</td><td>1.7</td><td>3.8</td><td>6.5</td><td>10.4</td><td>14.1</td><td>43.1</td><td>90.3</td></tr><tr><td>0.0</td><td>0.6</td><td>1.2</td><td>1.8</td><td>2.4</td><td>5.6</td><td>9.8</td></tr><tr><td rowspan="2">120s</td><td>3.7</td><td>9.9</td><td>18.5</td><td>31.4</td><td>35.2</td><td>90.2</td><td>98.5</td></tr><tr><td>0.0</td><td>0.6</td><td>1.3</td><td>1.9</td><td>1.5</td><td>4.7</td><td>5.2</td></tr></table></body></html>

Vehicle mobility directly determines the frequency and efficiency of cache-based model spreading and aggregation. So we evaluate the performance of Cached-DFL at different vehicle speeds. We fix cache size at 10, $\tau _ { m a x } = 1 0$ with non-i.i.d. data distribution. We take the previous speed $v = v _ { 0 } = 1 3 . 8 9 \mathrm { m / s }$ and $K = 3 0$ local updates as the base, named as speedup x1. To speedup, $v$ increases while $K$ reduces to keep the fair comparison under the same wall clock. For instance, for speedup $\mathbf { x } 3$ , $v = 3 v _ { 0 }$ and $K = 1 0$ . Results in Fig. 5 show that when the mobility speed increases, although the number of local updates decreases, the spread of all models in the whole vehicle network is boosted, thus disseminating local models more quickly among all vehicles leading to faster model convergence.

Grouped Mobility Patterns and Data Distributions. In practice, vehicle mobility patterns and local data distributions may naturally form groups. For example, a vehicle may mostly move within its home area, and collect data specific to that area. Vehicles within the same area meet with each others frequently, but have very similar data distributions. A fresh model of a vehicle in the same area is not as valuable as a slightly older model of a vehicle from another area. So model caching should not just consider model freshness, but should also take into account the coverage of group-based data distributions. For those scenarios, we develop a Groupbased (GB) caching algorithm. More specifically, knowing that there are $m$ distribution groups, one should maintain balanced presences of models from different groups. One straightforward extension of the LRU caching algorithm is to partition the cache into $m$ sub-caches, one for each group. Each sub-cache is updated by local models from its associated group, using the LRU policy. Due to limited space, the detailed algorithm is presented in our technical report Wang et al. (2024).

![](images/f2699a451b7ce83ddcabab8d9a98aa37a21dd11899af6ea5e33d0466b2bec556.jpg)  
Figure 4: Impact of $\tau _ { m a x }$ on Model Convergence for MNIST.

![](images/99d307fc79a419988d132ac77e8c577d3db348baaf9bb86c8e8bc97a874db02f.jpg)  
Figure 5: Convergence at Different Mobility Speeds under Non-IID Data Distribution.

We now conduct a case study for group-based cache update. As shown in Fig. 1, the whole Manhattan road network is divided into 3 areas, downtown, mid-town and uptown. Each area has 30 area-restricted vehicles that randomly moves within that area, and 3 or 4 free vehicles that can move into any area. We set 4 different area-related data distributions: Non-overlap, 1-overlap, 2-overlap, 3-overlap. $n$ -overlap means the number of shared label classes between areas is $n$ . We use the same non-i.i.d shards method in the previous section to allocate data points to the vehicles in the same area. On each vehicle, we evenly divide the cache for the three areas. We evaluate our proposed GB cache method on FashionMNIST. As shown in Fig. 6, while vanilla LRU converges faster at the very beginning, it cannot outperform DFL at last. However, the GB cache update method can solve the problem of LRU update and outperform DFL under different overlap settings.

![](images/974327a08450e0ebd8db22d9357f52538f0edeeb32cbfbba810aa434e4dc78c4.jpg)  
Figure 6: Group-based LRU Cache Update Performance under Different Data Distribution Overlaps on FashionMNIST.

# Discussions

In general, the convergence rate of Cached-DFL outperforms non-caching method (DFL), especially for non-i.i.d. distributions. Larger cache size and smaller $\tau _ { m a x }$ make Cached-DFL closer to the performance of CFL. Empirically, there is a trade-off between the age and number of cached models, it is critical to choose a proper τmax to control the staleness. What’s more, the choice of $\tau _ { m a x }$ should take into account the diversity data distributions on agents. In general, with non-i.i.d data distributions, the benefits of increasing the number of cached models can outweigh the damages caused by model staleness; while when data distributions are close to i.i.d, it is better to use a small number of fresh models than a large number of stale models. Similarly conclusions can also be drawn from the results of area-restricted vehicles. What’s more, in a system of moving agents, the mobility will also have big impact on the training, usually higher speed and communication frequency improve the model convergence and accuracy.

# Related Work

Decentralized FL (DFL) has been increasingly applied in vehicular networks, leveraging existing frameworks like vehicle-to-vehicle (V2V) communication (Yuan et al. 2024). V2V FL facilitates knowledge sharing among vehicles and has been explored in various studies (Samarakoon et al. 2019; Pokhrel and Choi 2020; Yu et al. 2020; Chen et al. 2021; Barbieri et al. 2022; Su, Zhou, and Cui 2022). Samarakoon et al. (2019) studied optimized joint power and resource allocation for ultra-reliable low-latency communication (URLLC) using FL. Su, Zhou, and Cui (2022) introduced DFL with Diversified Data Sources to address data diversity issues in DFL, improving model accuracy and convergence speed in vehicular networks. None of the previous studies explored model caching on vehicles. Convergence of asynchronous federated optimization was studied in Xie, Koyejo, and Gupta (2019). Their analysis focused on pairwise model aggregation between an agent and the parameter server, does not cover decentralized model aggregation with stale cached models in our proposed framework.

# Conclusion & Future Work

In this paper, we developed Cached-DFL, a novel decentralized Federated Learning framework that leverages on model caching on mobile agents for fast and even model spreading. We theoretically analyzed the convergence of Cached-DFL. Through extensive case studies in a vehicle network, we demonstrated that Cached-DFL significantly outperforms DFL without model caching, especially for agents with non-i.i.d data distributions. We employed only simple model caching and aggregation algorithms in the current study. We will investigate more refined model caching and aggregation algorithms customized for different agent mobility patterns and non-i.i.d. data distributions.

# Acknowledgments

This work was supported in part by the National Science Foundation (NSF) grants 2148309, 2315614 and 2337914, and was supported in part by funds from OUSD R&E, NIST, and industry partners as specified in the Resilient & Intelligent NextG Systems (RINGS) program. This work was also supported in part through the NYU IT High Performance Computing resources, services, and staff expertise. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the funding agencies.