# Can Private Machine Learning Be Fair?

Joseph Rance, Filip Svoboda

Department of Computer Science & Technology University of Cambridge Cambridge, United Kingdom jr879, fs437 @cam.ac.uk

# Abstract

We show that current SOTA methods for privately and fairly training models are unreliable in many practical scenarios. Specifically, we (1) introduce a new type of adversarial attack that seeks to introduce unfairness into private model training, and (2) demonstrate that the use of methods for training on private data that are robust to adversarial attacks often leads to unfair models, regardless of the use of fairness-enhancing training methods. This leads to a dilemma when attempting to train fair models on private data: either (A) we use a robust training method which may introduce unfairness to the model itself, or (B) we train models which are vulnerable to adversarial attacks that introduce unfairness. This paper highlights flaws in robust learning methods when training fair models, yielding a new perspective for the design of robust and private learning systems.

# Code — https://github.com/Joseph-Rance/unfair-f

# 1 Introduction

Ethically training Machine Learning systems on user data has long been a source of interest to the Machine Learning community (Arachchige et al. 2020; Cho, Wang, and Joshi 2020; Cao et al. 2020). We often wish to train a model that has uniform accuracy between different groups of participating users (fairness) without requiring private data to leave a user’s local device; for example, this is important when training models on hospital patient data (Soltan et al. 2024; Nicola et al. 2020). We show that methods of privately training fair models can be unreliable in practical scenarios.

Previous work has demonstrated effective methods for fair and private training (Mohri, Sivek, and Suresh 2019; Li et al. 2020, 2021a), however these works assume no users are malicious. There are similarly many techniques for ensuring robustness to attacks from participating users during private training (Blanchard et al. 2017; Nguyen et al. 2023; Sun et al. 2019), yet few have examined the problem of ensuring privacy, fairness, and robustness simultaneously. We show that attempting to achieve all three attributes by combining these separate methods serves to reverse their effects.

Furthermore, we show that it is possible for adversarial attacks to seek to introduce unfairness into the training process. Therefore, methods that are not robust to these attacks cannot be assumed to be fair, leading to a dilemma when attempting to train a fair model on private data:

A. To defend against attacks on fairness, we introduce a robust training method which, as we claim in this paper, introduces unfairness itself.   
B. We do not defend against attacks on fairness, leaving the model vulnerable to unfairness from these attacks.

Thus, due to (1) the threat of attacks on fairness and (2) our inability to ensure robustness without introducing unfairness, we cannot be confident a private training procedure will result in a fair model. For example, consider a model trained to predict interactions between pairs of molecules using private data produced by a group of participating companies (Mittone et al. 2023). One company may employ an attack on fairness to reduce accuracy on drugs produced by competitors. Either we risk vulnerability to this attack, or we risk discarding data on rare drug interactions.

Although, in some tasks, a robust learning method can be configured to introduce negligible unfairness while sufficiently protecting against most attacks, to our knowledge, there is no learning method that will have such a configuration on any given task. We make the following contributions:

• We propose a new type of training-time attack that introduces unfairness into the model. We show that attacks on fairness are well-founded and that they can be a threat in a wide range of practical learning scenarios (section 3). • We explore the shortcomings of current strategies for ensuring robustness, fairness, and privacy and show that three methods for robust model training can introduce unfairness (section 4). • We design a framework for testing robust learning methods that includes attacks on fairness.

![](images/c61925504a352bacf6ca18e3d761d037847b300e41b520c0a590d50e8baa49b9.jpg)  
Figure 1: Two possible scenarios under an attack on fairness, both leading to unfairness in the global model.

# 2 Training Trustworthy Models on Private Data

Private machine learning. In private machine learning, we wish to train a model on user data without these data leaving the users’ devices. In Federated Learning (FL) (McMahan et al. 2023) with DP-FedAvg (McMahan et al. 2018), clients locally train separate models on their data, that are then aggregated by a central server into a single, global model, which is then sent back to the clients to begin the next round of local training. Differential Privacy (DP) (Dwork 2006) can be applied to FL to allow the central server to obtain an effective global model without compromising user privacy. We focus on FL as a platform for ensuring privacy, but our claims do not necessarily only apply to the FL case.

Robust federated learning. FL is vulnerable to attacks from malicious clients which construct local models that introduce out-of-distribution behaviour to the global model (Bagdasaryan et al. 2019). The most effective methods for preventing these attacks eliminate local models that they determine lie outside benign model clusters. For example Krum discards models that are far from their closest neighbours in Euclidean space (Blanchard et al. 2017). The trimmed mean aggregation function instead discards the the $n$ highest and lowest client weight values (Yin et al. 2021).

Alternatively, Sun et al. (2019) show that a weaker version of DP-FedAvg can improve robustness without directly rejecting specific clients. However, it has been shown that DP can have a disproportionate impact on clients that are further from the common model distribution (Bagdasaryan and Shmatikov 2019; Ganev, Oprisanu, and Cristofaro 2022; Farrand et al. 2020), so this defence may indirectly have very similar effects to methods like Krum and trimmed mean.

Fair federated learning. In this paper, we define a model to be fair if it has an even accuracy distribution between different subpopulations of the dataset. For example, a phone manufacturer is likely to require that face recognition accuracy is equal between different ethnic groups. Fair aggregation functions typically attempt to increase the weights assigned to clients that hold less common types of data to increase the impact of these subsets of data. For example, in agnostic FL (Mohri, Sivek, and Suresh 2019), the aggregator returns the loss produced by the worst-case weighting of client loss values, while $\mathsf { q }$ -FFL (Li et al. 2020) weights clients by their loss, raised to the power of a parameter $q$ . Previous work has considered alternative notions of fairness (Mehrabi et al. 2021; Zhang et al. 2022), however, these are often due to distribution shift between our data and the task we aim to learn, which we do not investigate in this paper.

Combining fair and robust FL methods. When attempting to construct a training procedure with both fair and robust methods, we encounter a contradiction: robust aggregation eliminates local models that lie outside the distribution of common models, while fair aggregation attempts to increase the effect of these models. After combining these techniques, we therefore do not expect the full benefits of each method to persist. In this paper, we show that common robust aggregation methods can introduce unfairness during model training, and discuss how this affects our ability to prevent attacks on fairness. This is a fundamental problem with anomaly detection algorithms, which has so far not been directly addressed.

Related work. A similar idea has been previously presented by Wang et al. (2020), however their claims focus specifically on the implications for their proposed backdoor attack, while ours are broader. There has been previous work on constructing fair and robust training methods, however these methods either attempt to balance the effects of methods to those described above (Hu et al. 2022; Jin et al. 2023; Bowen et al. 2024), which we argue cannot always be an effective strategy, or solve the problem in non-standard setups, such as with personalisation (Li et al. 2021b), which are not applicable to the general case.

To our knowledge, no previous work has proposed attacks that target fairness in Federated Learning.

# 3 Attacks on Fairness in Federated Learning

In this section, we outline how fairness can be attacked in federated learning, and show that attacks on fairness are a threat in a wide range of realistic scenarios.

Threat model for attacks on fairness. We assume a similar threat model to Bagdasaryan et al. (2019): the attacker has access to the initial model sent to clients on the current round, $G _ { t }$ , and may control a small subset of clients to submit arbitrary parameters for aggregation. The attacker cannot see the models submitted by benign clients. The server may view the submitted parameters, but does not know which are submitted by the attacker and which are from benign clients. Such a threat could reasonably exist in systems where clients are untrusted (i.e. the attacker joins as a new client), or can be compromised (e.g. by malware).

In an attack on fairness, the attacker aims to train the global model, $G _ { t + 1 }$ , to have high accuracy on some target dataset, $D _ { T } \subseteq D$ , and low accuracy on the other data, $D _ { N } = D \backslash D _ { T }$ . For example, they may want high accuracy on only classes 0 and 1. The attacker can therefore obtain a set of target parameters, $\mathbf { x }$ , which they wish to substitute into $G _ { t + 1 }$ , by fine-tuning $G _ { t }$ using data in $\mathcal { D } _ { T }$ .

The attacker then uses $\mathbf { x }$ to compute a set of local, malicious parameters, $\mathbf { c } _ { 0 }$ , such that, after aggregation with the other clients’ models, the resulting global model, $G _ { t + 1 }$ , is approximately equal to $\mathbf { x }$ .

Model replacement attacks. If we directly submit $\mathbf { x }$ to the aggregator (i.e. ${ \bf c } _ { 0 } = { \bf x } ,$ , our parameters are unlikely to have a significant effect on the global model after aggregation with a much larger volume of benign parameters. Bagdasaryan et al. (2019) propose a more powerful strategy, model replacement, that allows the attacker to substitute the arbitrary target parameters, $\mathbf { x }$ , directly into the global model. Bagdasaryan et al. (2019) train $\mathbf { x }$ to learn a backdoor, instead of introducing unfairness. Under the FedAvg (McMahan et al. 2023) aggregator, the attack by Bagdasaryan et al. (2019) is able to influence the global model to be $\begin{array} { r } { G _ { t + 1 } = \mathbf { x } + \sum _ { i = 1 } ^ { m - 1 } \frac { n _ { i } } { n } \left( \mathbf { c } _ { i } - G _ { t } \right) } \end{array}$ , where $n _ { i }$ is the size of client $i$ ’s dataset, and $\begin{array} { r } { n = \sum _ { i = 0 } ^ { m - 1 } n _ { i } } \end{array}$ . Bagdasaryan et al.

(2019) implicitly assume the model converges (and therefore $\mathbf { c } _ { i } - G _ { t } \approx 0$ ; see eqn. 3 in Bagdasaryan et al. (2019)), to obtain $G _ { t + 1 } = \mathbf { x }$ . However, unlike in the backdoor case, attacks on fairness prevent convergence.

The update prediction attack. The model replacement attack requires this convergence assumption because we do not know the parameters produced by other clients. For our attack on fairness, we solve this problem by letting the attacker predict the parameters submitted by other clients. If we assume that the difference between the mean client parameters, $\scriptstyle \sum _ { i = 0 } ^ { m } { \frac { n _ { i } } { n } } \left( \mathbf { c } _ { i } \right)$ , and some other set of parameters, w, that have been trained on data that is i.i.d. to the union of the clients’ datasets is normally distributed with 0 mean, and variance a decreasing function of the amount of data seen during training, tending to 0 in the limit (Theorem 1), an attacker may be able to accurately predict the value $\scriptstyle \sum _ { i = 0 } ^ { m } { \frac { n _ { i } } { n } } \left( \mathbf { c } _ { i } \right)$ . This forms the basis for the update prediction attack.

Instead of subtracting the global model, $G _ { t }$ , from our $\mathbf { x }$ to yield $\mathbf { c } _ { 0 }$ , as is the case in the model replacement attack, the attacker can now subtract their model predictions, w, thus (approximately) eliminating all other terms in the FedAvg update rule.1 We set the attacker’s parameters to $\begin{array} { r } { { \bf c } _ { 0 } = \frac { \bar { n _ { 0 } } - \bar { n } } { n _ { 0 } } { \bf w } + \frac { n } { n _ { 0 } } { \bf x } , } \end{array}$ , so the FedAvg update becomes

$$
\begin{array} { l } { G _ { t + 1 } = G _ { t } + \displaystyle \sum _ { i = 0 } ^ { m - 1 } \frac { n _ { i } } { n } \left( \mathbf { c } _ { i } - G _ { t } \right) } \\ { \displaystyle \quad = G _ { t } + \displaystyle \frac { n _ { 0 } } { n } \left( \frac { n _ { 0 } - n } { n _ { 0 } } \mathbf { w } + \displaystyle \frac { n } { n _ { 0 } } \mathbf { x } - G _ { t } \right) + \displaystyle \sum _ { i = 1 } ^ { m } \frac { n _ { i } } { n } \left( \mathbf { c } _ { i } - G _ { t } \right) } \\ { \displaystyle \quad = G _ { t } + \displaystyle \frac { n _ { 0 } - n } { n } \mathbf { w } + \mathbf { x } + \sum _ { i = 1 } ^ { m } \frac { n _ { i } } { n } \left( \mathbf { c } _ { i } \right) - \sum _ { i = 0 } ^ { m } \frac { n _ { i } } { n } \left( G _ { t } \right) } \\ { \displaystyle \quad = \mathbf { x } + \sum _ { i = 1 } ^ { m } \frac { n _ { i } } { n } \left( \mathbf { c } _ { i } \right) - \displaystyle \frac { n - n _ { 0 } } { n } \mathbf { w } } \\ { \displaystyle \quad \approx \mathbf { x } } \end{array}
$$

Here we directly get $G _ { t + 1 } = \mathbf { x }$ without the convergence assumption, allowing us to use any set of parameters for $\mathbf { x }$ (see fig. 2).

![](images/d56f7694439a09903a714a0874d2713c15e330b927d9c4f7d70513733201577b.jpg)  
Figure 2: Visual representation of how the proposed attack. In practice, the angles between w, and x tend to be small, so the length of $\mathbf { c } _ { 0 }$ is not as extreme as this diagram suggests.

Following our threat model, no knowledge of parameters submitted by other clients is required by this attack. The attacker only requires an approximate estimate for the amount of data, $n - n _ { 0 }$ , contributed by other clients. Such an estimate need not be exact and could be iteratively increased each round until an effective value is found.

Attacks on fairness are well-founded. Our update prediction attack on fairness assumes that, when a federated learning model trains with a large amount of private data, the variance in benign client parameters introduced by unknown training data is low. We now prove this for strongly convex functions.

We begin by considering the following general optimisation problem for client $i$ :

$$
\operatorname* { m i n } _ { \mathbf { c } _ { i } \in \mathbb { R } ^ { d } } f ( \mathbf { c } _ { i } ) = \operatorname* { m i n } _ { \mathbf { c } _ { i } \in \mathbb { R } ^ { d } } \frac { 1 } { n _ { i } } \sum _ { j = 0 } ^ { n _ { i } - 1 } f _ { j } ( \mathbf { c } _ { i } )
$$

where each $f _ { i } \in \mathbb { R } ^ { d }  \mathbb { R }$ is a continuously differentiable function. We want to use minibatch gradient descent to solve this problem:

$$
\begin{array} { l } { { \displaystyle { \bf c } _ { i } ^ { ( k ) } = { \bf c } _ { i } ^ { ( k - 1 ) } - \frac { \alpha _ { k } } { b } \sum _ { j = 0 } ^ { b - 1 } \nabla f _ { s _ { k , j } } ( { \bf c } _ { i } ^ { ( k - 1 ) } ) } \ ~ } \\ { { \displaystyle ~ = { \bf c } _ { i } ^ { ( k - 1 ) } - \alpha _ { k } \big [ \nabla f ( { \bf c } _ { i } ^ { ( k - 1 ) } ) + \xi _ { k } \big ] } \ ~ } \end{array}
$$

where $\alpha _ { k }$ is the learning rate, each $s _ { k , j }$ is uniformly sampled at random from $\{ 0 , . . . , n _ { i } - 1 \}$ , and $\xi _ { k }$ represents the noise introduced by sampling the training distribution on round $k$ . This problem description and the following assumptions follow that of Li, Xiao, and Yang (2023), whose proof of SGD convergence to normally distributed models in the central case provides a basis for the following proofs. For simplicity, we set the learning rate to $\alpha _ { k } = \alpha _ { 1 } k ^ { - 1 / 2 }$ , which satisfies the assumptions of Li, Xiao, and Yang (2023).

We make the following assumptions.

(A1) Mean and covariance of $\xi _ { k }$ . $\forall \varepsilon > 0$ . a symmetric, positive definite matrix, $\Sigma$ , such that

$$
\mathbb { E } [ \xi _ { k } | \mathcal { F } _ { k - 1 } ] = 0 = \operatorname* { l i m } _ { n \to \infty } P ( | | \mathbb { E } [ \xi _ { k } \xi _ { k } ^ { T } | \mathcal { F } _ { k - 1 } ] - \Sigma | | \geq \varepsilon )
$$

where $\mathcal { F } _ { k } = \sigma ( x _ { 0 } , \xi _ { 1 } , \xi _ { 2 } , . . . , \xi _ { k } )$ is the $\sigma$ -algebra generated from the initialisation and noise terms up to round $k$ .

(A2) $L$ -smoothness of $f$ . $\exists L$ such that

$$
\forall x , y \in \mathbb { R } ^ { d } . \ | | \nabla f ( x ) - \nabla f ( y ) | | \leq L | | x - y | |
$$

(A3) $\mu$ -strong convexity of $f$ . $\exists \mu$ such that

$$
\forall x , y \in \mathbb { R } ^ { d } . \ f ( x ) \geq f ( y ) + \nabla f ( y ) ^ { T } ( x - y ) + \frac { \mu } { 2 } | | x - y | | ^ { 2 }
$$

(A4) Further smoothness condition for $f$ . $\exists p _ { 0 } , r _ { 0 } , K _ { d } >$ 0 such that, for any $| | x - x ^ { * } | | < r _ { 0 }$ ,

$$
| | \nabla f ( x ) - \nabla ^ { 2 } f ( x ^ { * } ) ( x - x ^ { * } ) | | \leq K _ { d } | | x - x ^ { * } | | ^ { 1 - p _ { 0 } }
$$

(A5) Dataset size heterogeneity. If client $i$ has a dataset of size $n _ { i }$ , modelled as a random variable, and $\begin{array} { r } { n = \sum _ { c = 0 } ^ { m - 1 } n _ { i } } \end{array}$ for $m$ clients, then

$$
\exists h \geq 1 \in \mathbb { R } . \forall i \in [ 0 , 1 , . . . , m - 1 ] . m n _ { i } \leq n h
$$

With this definition, if $h = 1$ , all client datasets must have the same amount of data, while as $h  \infty$ , the client data distribution constraints disappear.

Lemma 1 (variance for a single client). Under assumptions (A1)-(A4), if $\begin{array} { r } { \frac { 1 } { \alpha _ { 1 } } < 2 \mu } \end{array}$ , there exists some matrix, $W ^ { * }$ , such that

$$
k ^ { 1 / 4 } ( \mathbf { c } _ { i } ^ { ( k ) } - \mathbf { c } _ { i } ^ { * } ) \Rightarrow ^ { k } N \left( 0 , \alpha _ { 1 } W ^ { * } \right)
$$

where $\Rightarrow ^ { k }$ denotes convergence in probability, $\mathbf { c } ^ { * }$ is the unique minimum of $f , k \in \mathbb { N }$ is large, and $\begin{array} { r } { W _ { k , i , j } ^ { * } \in O \left( \frac { 1 } { b ^ { 2 } } \right) } \end{array}$ .

Proof. This extends the result from Li, Xiao, and Yang (2023). Under the above assumptions, Li, Xiao, and Yang (2023) show that for $b = 1$ we have eq. (3) for some matrix $W ^ { * }$ , where $A W ^ { * } + W ^ { * } A ^ { T } - d _ { 0 } W ^ { * } \stackrel { . } { = } \Sigma$ and $A$ is independent of all $\xi _ { i }$ .

Now consider the variance of $\xi _ { k }$ as $b$ increases. We assume that the summed gradients are independent and have finite first 2 moments. Thus, for large $b$ , by the classical central limit theorem, the gradient estimate, $\bar { \nabla } \hat { f } ( \mathbf { c } _ { i } ^ { ( k - 1 ) } )$ , is unbiased and normally distributed:

$$
\begin{array} { l } { \displaystyle \nabla \hat { f } ( { \mathbf { c } } _ { i } ^ { ( k - 1 ) } ) = \frac { 1 } { b } \sum _ { j = 0 } ^ { b - 1 } \nabla f _ { s _ { k , j } } ( { \mathbf { c } } _ { i } ^ { ( k - 1 ) } ) } \\ { \displaystyle \sim N \left( \mathbb { E } _ { f _ { j } } [ \nabla f _ { j } ( { \mathbf { c } } _ { i } ^ { ( k - 1 ) } ) ] , \frac { \mathbb { V } _ { f _ { j } } [ f _ { j } ( { \mathbf { c } } _ { i } ^ { ( k - 1 ) } ) ] } { b } \right) } \end{array}
$$

This yields a noise term, $\xi _ { k } \sim N \left( 0 , \mathbb { V } [ f _ { i } ( \mathbf { c } _ { i } ^ { ( k - 1 ) } ) ] / b \right)$ , with variance inversely proportional to batch size.

The maximum element in the covariance matrix for $\mathrm { v e c } ( \xi _ { k } \xi _ { k } ^ { T } )$ (where vec is a function that flattens a matrix into a vector) must be the variance of $( \xi _ { k } ) _ { i } ^ { 2 }$ for some $i$ . Since $( \xi _ { k } ) _ { i }$ is normally distributed with variance $\frac { c } { b }$ for some constant $c$ , we know that each element of this covariance matrix must be bounded by Vξk [(ξk)i2 ] = 2bc2 .

We have established that $A W ^ { * } + W ^ { * } A ^ { T } - d _ { 0 } W ^ { * } = \Sigma$ and that the elements of the covariance matrix for $\xi _ { k } \xi _ { k } ^ { T }$ (and therefore also those of $\Sigma$ ) are in $\begin{array} { r l r } { \mathrm { ~ } } & { { } } & { O \left( \frac { 1 } { b ^ { 2 } } \right) } \end{array}$ , so the elements of $W ^ { * }$ must also be in $\begin{array} { r l r } { \mathrm { ~ } } & { { } } & { O \left( \frac { 1 } { b ^ { 2 } } \right) } \end{array}$ . □

Now consider the FedAvg aggregation function (McMahan et al. 2023) to compute the global model, $G$ , from the model ${ \bf c } _ { i } ^ { ( u ) }$ produced by each client $i$ after $u$ batches using the above SGD setup for the current training round:

$$
G = \frac { 1 } { n } \sum _ { i = 0 } ^ { m - 1 } n _ { i } \mathbf { c } _ { i } ^ { u }
$$

where $\textstyle n = \sum _ { i = 0 } ^ { m } n _ { i }$

Theorem 1 (Variance of FedAvg). Under assumptions (A1)-(A5), if $\begin{array} { r } { \frac { 1 } { \alpha _ { 1 } } < 2 \mu } \end{array}$ for each client, the global model, $G$ , must be normally distributed with covariance matrix $M _ { g }$ such that $M _ { g , p , q } \in O \left( 1 / \sqrt [ 4 ] { e n m ^ { 3 } b ^ { 7 } } \right)$ for a large epoch index, $e$ , and batch size, $b$ .

Proof. From Lemma 1, we know that each $\mathbf { c } _ { i } ^ { ( e - 1 ) }$ is an independent, normally distributed random variable with covariance matrix $M _ { i }$ , where $\begin{array} { r l r } { M _ { i , p , q } } & { { } \in } & { O \left( 1 / \sqrt [ 4 ] { u b ^ { 8 } } \right) ~ = } \end{array}$

$O \left( 1 / \sqrt [ 4 ] { e n _ { i } b ^ { 7 } } \right)$ , for large $e$ and $b$ . After applying FedAvg, we get

$$
G \sim N \left( \sum _ { i = 0 } ^ { m - 1 } \frac { n _ { i } } { n } \mathbf { c } _ { i } ^ { * } , \sum _ { i = 0 } ^ { m - 1 } \frac { n _ { i } ^ { 2 } } { n ^ { 2 } } M _ { i } \right)
$$

Since, by (A5), $\textstyle \operatorname* { m a x } _ { i } { \frac { n _ { i } } { n } } \in O \left( { \frac { 1 } { m } } \right)$ for all clients, $i$ , the covariance matrix $\begin{array} { r } { M _ { g } = \sum _ { i = 0 } ^ { m - 1 } \frac { n _ { i } ^ { 2 } } { n ^ { 2 } } M _ { i } } \end{array}$ must have $M _ { g , p , q } \in$ $\begin{array} { r } { O \left( 1 / \sqrt [ 4 ] { e n _ { i } m ^ { 4 } b ^ { 7 } } \right) = O \left( 1 / \sqrt [ 4 ] { e n m ^ { 3 } b ^ { 7 } } \right) } \end{array}$

Therefore, by Chebyshev’s inequality, the probability our model prediction error is greater than $\gamma$ is in $O \left( 1 / \sqrt [ 4 ] { e n m ^ { 3 } b ^ { 7 } \gamma ^ { 8 } } \right)$

The above proof can similarly be adapted for SGD with momentum, using the relevant results from Li, Xiao, and Yang (2023). This proof does not extend to all non-convex models, however, as with model convergence in general, it is reasonable to assume that for a sufficiently smooth loss function and large enough batch size, because the attacker knows the model’s initial parameters, the non-convex case is locally similar to the strongly convex case above.

The update prediction attack also assumes that w is an unbiased estimator of Pim=0 ni (ci). This is false when there is heterogeneity between clients (FedAvg introduces some unfairness itself). The attacker could construct an unbiased estimator by directly locally simulating the entire FL training process, however we find that predicting w in a centralised manner is effective in practice.

Experimental results. We test the fairness attack for the datasets described in table 2 (Becker and Kohavi 1996; Krizhevsky 2009; Pushshift 2017). These datasets were selected to cover a range of tasks and to provide clear comparison with previous work (Bagdasaryan et al. 2019; Bhagoji et al. 2019; Wang et al. 2020; Nguyen et al. 2023; McMahan et al. 2023). Here, we substitute a single client’s model with our malicious set of parameters, using the original client’s model as the prediction w (i.e. we predict w using the same amount of data that would be held on a single client). For simplicity, we include the attack in every round. We additionally test its performance against the Krum, trimmed mean, and weak differential privacy defences. We select hyperparameters by performing a grid search over all reasonable combinations at multiple levels of granularity and present the median result across three trials in table 1. All experiments were performed on 2 NVIDIA RTX 2080 GPUs.

We record the change in fairness after the attack is introduced for each dataset-defence combination. The attack is effective at introducing unfairness into all three tasks. In practice, it may be preferable to perform a more subtle version of this attack. Although the size of the dataset needed to train w depends on the task, the attack remains effective even with the small local datasets available in the Reddit task. Table 2 shows that the Krum and trimmed mean defences are effective at preventing the attack on fairness, however we find that the weak-DP defence was not successful under any of the hyperparameter configurations we tested.

Table 1: Accuracy $( \% )$ achieved by different robust aggregation schemes for each dataset. The attack on fairness attempts to increase the accuracy on one subset of data while reducing the accuracy on another, so $^ { \cdot } \Delta$ fairness’ indicates the increase in accuracy disparity between these two sets (lower is better).   

<html><body><table><tr><td rowspan="2">Dataset</td><td rowspan="2">Attack</td><td colspan="2">No defence</td><td colspan="2">Krum</td><td colspan="2">Trimmed mean</td><td colspan="2">Weak-DP</td></tr><tr><td>Acc.</td><td>△ fairness</td><td>Acc.</td><td>△ fairness</td><td>Acc.</td><td>△fairness</td><td>Acc.</td><td>△ fairness</td></tr><tr><td>Census</td><td>Nainess</td><td>84.57</td><td>0.4</td><td>84.84</td><td>0.02</td><td>84.3</td><td>0.00</td><td>75.66</td><td>2.0</td></tr><tr><td>CIFAR-10</td><td>None Fairness</td><td>92.70 17.90</td><td>0.00 85.24</td><td>91.59 63.78</td><td>0.00 6.26</td><td>91.98 63.11</td><td>0.00 6.28</td><td>93.60 17.62</td><td>0.00 85.29</td></tr><tr><td>Reddit</td><td>None Fairness</td><td>18.08 4.52</td><td>0.00 118.94</td><td>17.82 17.97</td><td>0.00 -0.16</td><td>18.08 17.90</td><td>0.00 0.19</td><td>08.55 04.52</td><td>0.00 108.39</td></tr></table></body></html>

<html><body><table><tr><td>Dataset</td><td>#Records</td><td>Model</td><td>#Clients/Per round</td></tr><tr><td>UCI Census</td><td>49k</td><td>3-layer FC</td><td>10/10</td></tr><tr><td>CIFAR-10</td><td>60k</td><td>ResNet-50</td><td>10/10</td></tr><tr><td>Reddit</td><td>2.3M</td><td>LSTM</td><td>10,000 /100</td></tr></table></body></html>

Table 2: We train clients for 10, 2, and 5 epochs on i.i.d. data, for a total of 40, 120, and 100 rounds for the Census, CIFAR, and Reddit datasets respectively. We use the same augmentation scheme as Zagoruyko and Komodakis (2017) for the CIFAR-10 dataset and the albert-base- $\cdot \nu 2$ tokeniser (Lan et al. 2020) for the Reddit task. The attack reduces the accuracy of entries labelled as female, increases accuracy on classes 0 and 1, and reduces accuracy following the word ‘I’ for the Census, CIFAR, and Reddit tasks respectively.

Previous work has shown similar results for the weakDP defence (Wang et al. 2020), although it is difficult to justify specifically why the defence is ineffective against this attack-task combination. However, recent works have also shown that weak differential privacy can introduce unfairness into a model (Bagdasaryan and Shmatikov 2019; Ganev, Oprisanu, and Cristofaro 2022; Farrand et al. 2020).

Attacks on momentum-based aggregators. Momentumbased aggregatiors (Reddi et al. 2021) are common in practice, which could lead to the attack on fairness becoming ineffective. We repeat our experiments for the Census task in table 1 for three different aggregators (table 3), finding that attacks on fairness remain effective against these aggregators without modification.

Table 3: Attacks on different aggregators for the Census task. All hyperparameters were identical to table 1, which accounts for the drop in initial accuracy and fairness. $\Delta$ fairness is calculated with respect to the baseline in table 1.   

<html><body><table><tr><td>Aggregator</td><td>Attack</td><td>Overall</td><td>△ fairness</td></tr><tr><td rowspan="2">FedAdaGrad</td><td>Baseline</td><td>84.58</td><td>-2.84</td></tr><tr><td>Fairness</td><td>75.79</td><td>36.47</td></tr><tr><td rowspan="2">FedYogi</td><td>Baseline</td><td>78.72</td><td>43.76</td></tr><tr><td>Fairness</td><td>80.78</td><td>51.87</td></tr><tr><td rowspan="2">FedAdam</td><td>Baseline</td><td>84.50</td><td>1.64</td></tr><tr><td>Fairness</td><td>80.14</td><td>45.91</td></tr></table></body></html>

Attacks on fairness under data heterogeneity. We have shown attacks on fairness function on i.i.d. data distributions. In practice, this is not always the case (Yang et al. 2018; Hard et al. 2019; Huba et al. 2022). Furthermore, due to (A5), heterogeneity should reduce the attack’s stability.

To show that attacks on fairness can be a threat in settings where there is high heterogeneity, we repeat the baseline experiment from table 1, with the CIFAR-10 dataset distributed between clients using a log-normal label distribution across the clients parameterised by $\mu \ = \ 0$ and $\sigma \in \{ 0 , 1 , 2 \}$ . Figure 3 shows that heterogeneity reduces the attack’s effectiveness. However, even at high $\sigma$ values, it remains effective. Here, we do not include a defence, although we expect that increased heterogeneity would make detection more difficult (Ozdayi and Kantarcioglu 2021).

![](images/276c0567a0d592c057bcb5e9bf62bfbdb63635342e1cf01472c4988701748095.jpg)  
Figure 3: Accuracy per training round for the attack on fairness under different heterogeneity values $( \sigma )$ for the CIFAR10 task. The red (upper) line represents the accuracy on data the attack seeks to increase, while the green (lower) line represents the accuracy on other data. As heterogeneity increases, the attack’s strength decreases. Without the fairness attack, the accuracy on both datasets is approximately equal.

# 4 Robust Aggregation Introduces Unfairness

In the previous section we have shown that attacks on fairness are a realistic threat against FL. This motivates the need for a training process that is robust to these attacks, without introducing unfairness itself. In this section we seek to answer the question can we prevent adversarial attacks on federated learning without introducing unfairness?

We focus this analysis on synthetic datasets, to help test whether these problems are likely to generalise to other defences that follow a similar construction, rather than repeating previous work to show unfairness can be introduced by specific defences in the wild (Wang et al. 2020).

Fairness impact of Krum and trimmed mean. Both the Krum and trimmed mean robust aggregation methods remove client models that lie far from a mode of the distribution. This could lead to unfairness because clients holding certain types of uncommon data are more likely to produce models that are far from clusters of common models (see fig. 4). Figure 5 shows that some clients hold meaningfully less common datasets even when there is low data heterogeneity between clients (some clients are consistently ranked less trustworthy than others when data is randomly distributed). Furthermore, removal of a small number of these clients can have a disproportionately high impact due to the uncommon nature of the data they hold.

We now show that, this can lead to the reduction - or, in this extreme case, elimination - of critical functionality from a model that is only present in a minority of clients.

![](images/0b5f149fffaa88c62a532fb0d7366b7ea091fe51fc3b75c307ba9cd3db8bdc96.jpg)  
Figure 4: 2D projection of models produced by clients on the MNIST dataset (LeCun, Cortes, and Burges 1998). Points coloured green (inverted triangles) are produced by clients that only hold data with classes 0 and 1, while points coloured red (upright triangles) are produced by clients running a backdoor attack. Although a low-dimensional representation exaggerates the problem, there is little difference between malicious models and models trained on specific subpopulations of the dataset from a clustering perspective.

![](images/e65776768fa687a50c6015dd856d588e62e2fa9593d4522d17f9f5da1fd2f554.jpg)  
Figure 5: Client trustworthiness ranking on each round according to the Krum defence for the CIFAR-10 task. The attacker (red) is consistently ranked least trustworthy, however, we also observe some clients (e.g. the lower, blue line) are lower ranked than others (e.g. the upper, green line).

We construct the dataset shown in fig. 6, in which five clients (group A) do not have any data where the input begins with a 0, and one client (group B) does not have data where the input ends with a 0. This construction leads to the client from group B producing different models compared to group A when training on a simple, fully-connected model.

![](images/de35409f783164b8c67efeba3338631beb7e59b37c1e8cec54eb75798869befb.jpg)  
Figure 6: This dataset represents the OR function, where black squares indicate the value 1, and white the value 0. We split it so that the input [0, 1] only occurs in 5/6 clients, while the input $[ 1 , 0 ]$ only occurs in the remaining client.

As shown in table 4, both defence methods incorrectly determine that the group B client is malicious across multiple tests, leading to the uncommon functionality that is unique to this client $( [ 0 , 1 ]  1 )$ being lost in the aggregated model. As this data is lost, fair aggregation methods that reweight uncommon models would be ineffective in this scenario. Although we eliminate more models (1) than there are attackers (0) on each round, this is realistic because we will not know how many attackers there are in practice.

Table 4: Accuracy $( \% )$ for each dataset under different defences.   

<html><body><table><tr><td>Defence</td><td>Combined</td><td>Group A</td><td>Group B</td></tr><tr><td>No defence</td><td>100</td><td>100</td><td>100</td></tr><tr><td>Trimmed mean</td><td>92</td><td>100</td><td>50</td></tr><tr><td>Krum</td><td>92</td><td>100</td><td>50</td></tr></table></body></html>

Furthermore, the aggregator does not know how each local model is obtained (due to our privacy constraints) so there cannot exist any unsupervised anomaly-detectionbased defence that would be able to identify group B as benign, while still rejecting all attacks. This is true because, if such a defence existed, it would continue to accept models from group B when the task is redefined as returning the value of the second input (i.e. $[ 0 , 1 ]  1$ ; $[ 0 , 0 ] \to 0 ; . . . )$ . In this scenario, group B introduces behaviour that directly opposes the training goal, so it should be classed as malicious.

More generally, the local training function that produces models from local data by SGD is non-injective and thus has no left inverse, so overlap between the tail of the benign distribution and the set of malicious models is possible (see fig. 7). Shumailov et al. (2021) show that malicious parameters can be learnt by manipulating the order of clean data, which implies that this overlap exists in realistic scenarios. Therefore, even without DP constraints, for some datasets it is impossible to detect all attacks without misidentifying some benign models as malicious.

![](images/968e90749a736469aba430c5feb562ee5e5ef5f74ac5edd2cb6760a243ce0a08.jpg)  
Figure 7: There may be overlap between the distribution of models produced by legitimate clients (blue, left) and the set of models produced by malicious processes (red, right). Methods based on anomaly detection must reject the tail of the benign distribution (e.g. by accepting only models to the left of boundary $A$ , leading to unfairness due to the omission of uncommon data) or accept some malicious models (e.g. by accepting only models to the left of boundary $B$ ).

In our testing shown in table 1, we also find that accuracy disparity between common and uncommon data in the Census task2 increased under all three defences, with the disparity growing by a median of 1.95, 0.53, and 51.72 for the trimmed mean, Krum, and weak DP defences respectively. Similar results to these have previously been observed in realistic setups, often showing a more significant reduction in fairness compared to these tasks (for example, Wang et al. (2020) 2020), although analysis of this issue has never been extended to the general problem that we investigate here. Thus, even without the use of techniques to evade detection by robust aggregators (Bagdasaryan et al. 2019), which can increase the difficulty of separating uncommon from malicious models, unfairness is introduced due to this overlap.

While robust aggregation methods that attempt to retain fairness have been presented, they are forced to make significant compromises compared to the methods studied here. For example, Fed-MGDA $^ +$ (Hu et al. 2022) employs a gradient clipping strategy, which is clearly weaker than the weak differential privacy defence described above.

Unfair-update detection: testing a new defence for fairness attacks. While Wang et al. (2020) have shown that verifying that a model does not contain any backdoors is computationally intractable, it is relatively simple to verify that a model is fair across a set of predetermined attributes with high confidence. This suggests a simpler defence for attacks that only attempt to introduce unfairness may be to measure the fairness impact of each client’s model and assume clients that significantly reduce fairness are malicious.

When repeating the experiments for the CIFAR-10 task with this defence, the change in accuracy disparity ( $\Delta$ fairness) is only 0.34 under the fairness attack. Additionally, the unfair-update detection algorithm initially appears to solve the problem of overlapping malicious and benign model distributions by accepting models based on their impact on the global model rather than based on how we believe they have been trained. However, a client which submits a model trained on new data that may be necessary to achieve a more fair final model (after convergence) is likely to have reduced accuracy/fairness in the short term, leading to its rejection in some scenarios. The problem arises from the attack’s greedy setup: we select the clients that produce the most fair model on the next round, not those that result in a more fair final model, when it may be necessary to temporarily reduce fairness in order to achieve a higher final value.

![](images/7ea94a8eb21fceb080c8bcc56e25050544198d35c6d4978eeb59aeab97e83d0a.jpg)  
Figure 8: This dataset represents the XOR function. We split it so that the input $[ 0 , 1 ]$ only occurs in 1 out of 6 clients.

Table 5: Accuracy $( \% )$ per dataset under different defences.   

<html><body><table><tr><td>Defence</td><td>Combined</td><td>Group C</td><td>Group D</td></tr><tr><td>No defence</td><td>100</td><td>100</td><td>100</td></tr><tr><td>Unfair-update det.</td><td>96</td><td>100</td><td>75</td></tr></table></body></html>

# 5 Conclusion

We have shown that common defence methods can introduce unfairness and that attacks on fairness are a real threat to the federated learning training process. Furthermore, to our knowledge, there does not exist any defence that can ensure the robustness of the global model without using a method based on those analysed in this paper. Thus, assuming such a defence does not currently exist, in the presence of untrusted clients, we cannot be confident that training on private data will result in a fair model.

Even if robust aggregation may introduce a negligible amount of unfairness for some datasets, it is difficult to predict which datasets will present this problem and how to manage the tradeoff between fairness and robustness. This presents a risk, particularly for systems that are expensive to retrain. Future work would therefore benefit from approaching the problem of robustness in FL from a new, fairness-aware perspective.

# Ethical Statement

We introduce a new attack that could be used against real systems. We provide an implementation of this attack, because we believe the benefits of improving the accessibility of tools to test robustness outweigh the disadvantages of making these attacks more accessible.