# On Local Overfitting and Forgetting in Deep Neural Networks

Uri Stern, Tomer Yaacoby and Daphna Weinshall

School of Computer Science and Engineering, The Hebrew University of Jerusalem, Jerusalem 91904, Israel ustern $@$ gmail.com, tomer.yaacoby $@$ mail.huji.ac.il, daphna $@$ mail.huji.ac.il

# Abstract

The infrequent occurrence of overfitting in deep neural networks is perplexing: contrary to theoretical expectations, increasing model size often enhances performance in practice. But what if overfitting does occur, though restricted to specific sub-regions of the data space? In this work, we propose a novel score that captures the forgetting rate of deep models on validation data. We posit that this score quantifies local overfitting: a decline in performance confined to certain regions of the data space. We then show empirically that local overfitting occurs regardless of the presence of traditional overfitting. Using the framework of deep over-parametrized linear models, we offer a certain theoretical characterization of forgotten knowledge, and show that it correlates with knowledge forgotten by real deep models. Finally, we devise a new ensemble method that aims to recover forgotten knowledge, relying solely on the training history of a single network. When combined with knowledge distillation, this method will enhance the performance of a trained model without adding inference costs. Extensive empirical evaluations demonstrate the efficacy of our method across multiple datasets, contemporary neural network architectures, and training protocols.

# 1 Introduction

Overfitting a training set is considered a fundamental challenge in machine learning. Theoretical analyses predict that as a model gains additional degrees of freedom, its capacity to fit a given training dataset increases. Consequently, there is a point where the model becomes too specialized for a particular training set, leading to an increase in its generalization error. In deep learning, one would expect to see increased generalization error as the number of parameters and/or training epochs increases. Surprisingly, even vast deep neural networks with billions of parameters seldom adhere to this expectation, and overfitting as a function of epochs is almost never observed (Liu et al. 2022). Typically, a significant increase in the number of parameters still results in enhanced performance, or occasionally in peculiar phenomena like the double descent in test error (Annavarapu 2021), see Section 3. Clearly, there exists a gap between our classical understanding of overfitting and the empirical results observed when training modern neural networks.

![](images/ab6a99ebcb42a79817ad00145a653044172b83812b44ed371d8992f2839c6fca.jpg)  
Figure 1: Local overfitting and forgetting in a binary problem, where blue and orange denote the different classes, and circles mark the validation set. The initial (left) and final (right) separators of a hypothetical learning method are shown, where $\otimes$ marks prediction errors. Clearly the final classifier has a smaller generalization error, but now one point at the top is ’forgotten’.

To bridge this gap, we present a fresh perspective on overfitting. Instead of solely assessing it through a decline in validation accuracy, we propose to monitor what we term the model’s forget fraction. This metric quantifies the portion of test data (or validation set) that the model initially classifies correctly but misclassifies as training proceeds (see illustration in Fig. 1). Throughout this paper we term the decline in test accuracy as ”forgetting”, to emphasize that the model’s ability to correctly classify portions of the data is reduced. In Section 3, we investigate various benchmark datasets, observing this phenomenon even in the absence of overfitting as conventionally defined, i.e., when test accuracy increases throughout. Notably, this occurs in competitive networks despite the implementation of modern techniques to mitigate overfitting, such as data augmentation and dropout. Our empirical investigation also reveals that forgetting of patterns occurs alongside the learning of new patterns in the training set, explaining why the traditional definition of overfitting fails to capture this phenomenon.

Formal investigation of the phenomenon of forgotten knowledge is challenging, particularly in the context of deep neural networks which are not easily amenable to formal analysis. Instead, in Section 4 we adopt the framework of over-parameterized deep linear networks. This framework involves non-linear optimization and has previously offered valuable insights into the learning processes of practical deep networks (Fukumizu 1998; Saxe, McClelland, and

Ganguli 2014; Arora, Cohen, and Hazan 2018; Arora et al. 2019; Hu, Xiao, and Pennington 2020). Within such models, employing gradient descent for learning reveals a straightforward and elegant characterization of the model’s evolution (Hacohen and Weinshall 2022).

We expand upon this analysis, deriving an analytical description of the data points forgotten at each gradient descent step. As this analysis pertains specifically to deep linear models, it’s crucial to correlate its findings with the forgotten knowledge in competitive neural networks. Intriguingly, when comparing these findings with the same image datasets utilized in our experiments, we observe significant overlap between the sets. This implies that the proposed theoretical characterization might offer partial insight into the phenomenon of forgotten knowledge and the underlying causes of local overfitting.

Based on the empirical observations reported in Section 3, we propose in Section 5 a method that can effectively reduce the forgetting of test data, and thus improve the final accuracy and reduce overfitting. More specifically, we propose a new prediction method that combines knowledge gained in different stages of training. The method delivers a weighted average of the class probability output vector between the final model and a set of checkpoints of the model from midtraining, where the checkpoints and their weights are chosen in an iterative manner using a validation dataset and our forget metric. The purpose is two-fold: First, an improvement upon the original model by our method will serve as another strong indication that models indeed forget useful knowledge in the late stages of training. Second, to provide a proof-of-concept that this lost knowledge can be recovered, even with methods as simple as ours.

In Section 6 we describe the empirical validation of our method in a series of experiments over image classification datasets with and without label noise, using various network architectures, including in particular modern networks over Imagenet. The results indicate that our method is universally useful and generally improves upon the original model, thus fulfilling its two mentioned goals. When compared with alternative methods that use the network’s training history, our method shows comparable or improved performance, while being more general and easy to use (both in implementation and hyper-parameter tuning). Unlike some methods, it does not depend on additional training choices that require much more time and effort to tune the new hyper-parameters.

Our main contributions. (i) A novel perspective on overfitting, capturing the notion of local overfitting. (ii) Empirical evidence that overfitting occurs locally even without a decrease in overall generalization. (iii) Analysis of the relation between forgetting and PCA. (iv) A simple and effective method to reduce overfitting, and its empirical validation.

# 2 Related Work

Study of forgetting in prior work. Most existing studies examine the forgetting of training data, where certain training points are initially memorized but later forgotten. This typically occurs when the network cannot fully memorize the training set. In contrast, our work focuses on the forgetting of validation points, which arises when the network successfully memorizes the entire training set. Building on Arpit et al. (2017), who show that networks first learn ”simple” patterns before transitioning to memorizing noisy data, we analyze the later stages of learning, particularly in the context of the double descent phenomenon. Another related but distinct phenomenon is ”catastrophic forgetting” (McCloskey and Cohen 1989), which occurs in continual learning settings where the training data evolves over time—unlike the static training scenario considered here.

Ensemble learning. Ensemble learning has been studied extensively (see Polikar 2012; Ganaie et al. 2022; Yang, Lv, and Chen 2023). Our work belongs to a line of works called ”implicit ensemble learning”, in which only a single network is learned in a way that ”mimics” ensemble learning (Srivastava et al. 2014). Utilizing checkpoints from the training history as a ’cost-effective’ ensemble has also been considered. This was achieved by either considering the last epochs and averaging their probability outputs (Xie, Xu, and Chuang 2013), or by employing exponential moving average (EMA) on all the weights throughout training (Polyak and Juditsky 1992). The latter method does not always succeed in reducing overfitting, as discussed in (Izmailov et al. 2018).

Several methods (Izmailov et al. 2018; Garipov et al. 2018; Huang et al. 2017) modify the training protocol to converge to multiple local minima, which are then combined into an ensemble classifier. While these approaches show promise (Noppitak and Surinta 2022), they add complexity to training and may even hurt performance (Guo, Jin, and Liu 2023). Our comparisons (see Table 3) demonstrate that our simpler method either matches or outperforms these techniques in all studied cases.

Ensemble methods can impose significant computational demands during inference, especially for large ensembles. Knowledge distillation (Hinton, Vinyals, and Dean 2015) addresses this challenge by training a single student model to replicate the ensemble’s predictions, effectively eliminating the increased computational costs. This approach typically maintains the ensemble’s performance and, in highnoise scenarios, may outperform the ensemble itself (Jeong and Chung 2024; Stern, Shwartz, and Weinshall 2024).

Studies of overfitting and double descent. Double descent with respect to model size has been studied empirically in (Belkin et al. 2019; Nakkiran et al. 2021), while epoch-wise double descent (which is the phenomenon analyzed here) was studied in (Stephenson and Lee 2021; Heckel and Yilmaz 2020). These studies analyzed when and how epochwise double descent occurs, specifically in data with label noise, and explored ways to avoid it (sometimes at the cost of reduced generalization). Essentially, our research identifies a similar phenomenon in data without label noise. It is complementary to the study of ”benign overfitting”, e.g., the fact that models can achieve perfect fit to the train data while still obtaining good performance over the test data.

# 3 Overfitting Revisited

The textbook definition of overfitting entails the cooccurrence of increasing train accuracy and decreasing generalization. Let $a c c ( e , S )$ denote the accuracy over set $S$ in epoch $e$ - some epoch in mid-training, $E$ the total number of epochs, and $T$ the test1 dataset. Using test accuracy to approximate generalization, this implies that overfitting occurs at epoch $e$ when $a c c ( e , T ) \geq a c c ( E , T )$ .

![](images/5da6ba5e7d7b06d4fe793ea68547d1460c6c502df9bd468bf1f55cd9ed84af65.jpg)  
Figure 2: (a)-(b): Blue denotes test accuracy $Y$ -axis) as a function of epoch ( $X$ -axis). Among those correctly recognized in each epoch $e$ , orange denotes the fraction that remains correctly recognized at the end. The test accuracy (blue) shows a clear double ascent of accuracy, which is much less pronounced in the orange curve. During the decrease in test accuracy - the range of epochs between the first and second dashed red vertical lines - the large gap between the blue and orange plots indicates the fraction of test data that has been correctly learned in the first ascent and then forgotten, without ever being re-learned in the later recovery period of the second ascent. (c): The difference between the number of clean and noisy datapoints at each epoch during the second ascent of test accuracy (the epochs after the second dashed red vertical line), counting datapoints with large loss only. Positive (negative) value indicates that clean (noisy) datapoints are more dominant in the corresponding epoch.

We begin by investigating the hypothesis that portions of the test data $T$ may be forgotten by the network during training. When we examine the ’epoch-wise double descent’, which frequently occurs during training on datasets with significant label noise, we indeed observe that a notable forgetting of the test data coincides with the memorization of noisy labels. Here, forgetting serves as an objective indicator of overfitting. When we further examine the training of modern networks on standard datasets (devoid of label noise), where overfitting (as traditionally defined) is absent, we discover a similar phenomenon (though of weaker magnitude): the networks still appear to forget certain sub-regions of the test population. This observation, we assert, signifies a significant and more subtle form of overfitting in deep learning.

Local overfitting. Let $M _ { e }$ denote the subset of the test data mislabeled by the network at some epoch $e$ . We define below two scores $L _ { e }$ and $F _ { e }$ :

$$
F _ { e } = \frac { a c c ( e , M _ { E } ) \cdot | M _ { E } | } { | T | } , L _ { e } = \frac { a c c ( E , M _ { e } ) \cdot | M _ { e } | } { | T | }
$$

The forget fraction $F _ { e }$ represents the fraction of test points correctly classified at epoch e but misclassified by the final model. $L _ { e }$ represents the fraction of test points misclassified at epoch $e$ but correctly classified by the final model. The relationship $\mathop { a c c } ( E , T ) \mathop { = } \mathop { a c c } ( e , T ) \mathop { + } L _ { e } \mathrm { ~ - ~ } F _ { e }$ follows2. In line with the classical definition of overfitting, if $L _ { e } < F _ { e }$ , overfitting occurs since $a c c ( E , T ) < a c c ( e , T )$ .

But what if $L _ { e } \ge F _ { e } \forall e ? \ : \mathrm { I }$ y its classical definition overfitting does not occur since the test accuracy increases continuously. Nevertheless, there may still be local overfitting as defined above, since $F _ { e } > 0$ indicates that data has been forgotten even if $L _ { e } \geq F _ { e }$ .

Reflections on the epoch-wise double descent. Epochwise double descent (see Fig. 2) is an empirical observation (Belkin et al. 2019), which shows that neural networks can improve their performance even after overfitting, thus causing double descent in test error during training (or doubleascent in test accuracy). This phenomenon is characteristic of learning from data with label noise, and is strongly related to overfitting since the dip in test accuracy co-occurs with the memorization of noisy labels.

We examine the behavior of score $F _ { e }$ in this context and make a novel observation: when we focus on the fraction of data correctly classified by the network during the second rise in test accuracy, we observe that the data newly memorized during these epochs often differs from the data forgotten during the overfitting phase (the dip in accuracy). In fact, most of this data has been previously misclassified (see Figs. 2a-2b). Fig. 2c further illustrate that during the later stages of training on data with label noise, the majority of the data being memorized is, in fact, data with clean labels, which explains the second increase in test accuracy. It thus appears that epoch-wise double descent is caused by the simultaneous learning of general (but hard to learn) patterns from clean data, and irrelevant features of noisy data.

Forgetting in the absence of label noise When training deep networks on visual benchmark datasets without added label noise, double descent rarely occurs, if ever. In contrast, we observe that local overfitting, as captured by our new score $F _ { e }$ , commonly occurs.

To show this, we trained various neural networks (ConvNets: Resnet, ConvNeXt; Visual transformers: ViT, MaxViT) on various datasets (CIFAR-100, TinyImagenet,

![](images/8dbb5249c3d558ecd7fea6cb4c4b34bc41df14dd75c0246e137df83cb1bd215a.jpg)  
Figure 3: (a) The $F _ { e }$ score (1) of ConvNeXt trained on Imagenet $Y$ -axis) as a function of epoch $X$ -axis), showing 3 network sizes: small $$ blue, base $$ orange and large $$ green. Accuracy remained consistent across all network sizes, while $F _ { e }$ increases with network size. (b) Within the set of wrongly classified test points after training, we show the last epoch in which an example was classified correctly.

Imagenet) using a variety of optimizers (SGD, AdamW) and learning rate schedulers (cosine annealing, steplr). In Fig. 3a we report the results, showing that all networks forget some portion of the data during training as in the label noise scenario, even if the test accuracy never decreases. Fig. 3b demonstrates that this effect is not simply due to random fluctuations: many test examples that are incorrectly classified post training have been correctly classified during much of the training. These results are connected to overfitting in Fig. 3a: when investigating larger models and/or relatively small amounts of train data, which are scenarios that are expected to increase overfitting based on theoretical considerations, we see larger forget fraction $F _ { e }$ (see Figs. 5-6 in App. ${ \mathbf A } ^ { 3 }$ for additional results).

In summary, we see that neural networks can, and often will, ”forget” significant portions of the test population as their training proceeds. In a sense, the networks are overfitting, but this only occurs at some limited sub-regions of the world. The reason this failing is not captured by the classical definition of overfitting is that the networks continue to learn new general patterns simultaneously. In Section 5 we discuss how we can harness this observation to improve the network’s performance.

# 4 Forgotten Knowledge: Theory & Exps

To gain insight into the nature of knowledge forgotten while training a deep model with Gradient Descent (GD), we analyze over-parameterized deep linear networks trained by GD. These models are constructed through the concatenation of linear operators in a multi-class classification scenario: $\pmb { y } = W _ { L } \cdot . . . \cdot W _ { 1 } \mathbf { x }$ , where $\mathbf { x } \in \mathbb { R } ^ { d }$ . For simplicity, we focus on the binary case with two classes, suggesting that similar qualitative outcomes would apply to the more general multi-class model. Accordingly, we redefine the objective function as follows:

$$
\operatorname* { m i n } _ { W _ { 1 } , \dots , W _ { L } } \sum _ { i = 1 } ^ { n } \| W _ { L } \cdot \dots \cdot W _ { 1 } \mathbf { x } _ { i } - y _ { i } \| ^ { 2 }
$$

Above the matrices $\{ W _ { l } \} _ { l = 1 } ^ { L }$ represent the $2 D$ matrices corresponding to $L$ layers of a deep linear network, and points $\{ { \bf x } _ { i } \} _ { i = 1 } ^ { n }$ represent the training set with labeling function $y _ { i } = \pm 1$ for the first and second classes, respectively. Note that $\begin{array} { r } { \pmb { w } = \prod _ { l = L } ^ { 1 } W _ { l } } \end{array}$ is a row vector that defines the resulting separator between the classes. The classifier is defined as: $\begin{array} { r } { \dot { f ( \mathbf { x } ) } = \mathrm { s i g n } \left( \prod _ { l = L } ^ { 1 } W _ { l } \mathbf { x } \right) } \end{array}$ for $\mathbf { x } \in \mathbb { R } ^ { d }$ .

Preliminaries. Let $\begin{array} { r } { \mathbf { \pmb { w } } ^ { ( n ) } = \prod _ { l = L } ^ { 1 } { \cal W } _ { l } ^ { ( n ) } } \end{array}$ represent the separator after $n$ GD steps, where $\pmb { w } ^ { ( n ) } \equiv [ w _ { 1 } ^ { ( n ) } , \ldots , w _ { d } ^ { ( n ) } ] \in$ $\mathbb { R } ^ { d }$ . For convenience, we rotate the data representation so that its axes align with the eigenvectors of the data’s covariance matrix. Hacohen and Weinshall (2022) showed that the convergence rate of the $j ^ { \mathrm { t h } }$ element of $\textbf { \em w }$ with respect to $n$ is exponential, governed by the corresponding $j ^ { \mathrm { t h } }$ eigenvalue:

$$
w _ { j } ^ { ( n ) } \approx \lambda _ { j } ^ { n } w _ { j } ^ { ( 0 ) } + [ 1 - \lambda _ { j } ^ { n } ] w _ { j } ^ { o p t } , \qquad \lambda _ { j } = 1 - \gamma s _ { j } L
$$

Here, ${ \pmb w } ^ { ( 0 ) }$ denotes the separator at initialization, $w ^ { o p t }$ denotes the optimal separator (which can be derived analytically from the objective function), $s _ { j }$ represents the $j ^ { \mathrm { t h } }$ singular value of the data, and $\gamma$ is the learning rate. Notably, while $\boldsymbol { \mathbf { \mathit { w } } } ^ { o p t }$ is unique, the specific solution at convergence {Wl(∞)}lL=1 is not.

# 4.1 Forget Time in Deep Linear Models

Let $\Lambda$ denote $\mathrm { d i a g } ( \{ \lambda _ { j } \} )$ - a diagonal matrix in $\mathbb { R } ^ { d \times d }$ , and I the identity matrix. It follows from (3) that

$$
{ \pmb w } ^ { ( n ) } \approx { \pmb w } ^ { ( 0 ) } \Lambda ^ { n } + { \pmb w } ^ { o p t } [ { \pmb I } - \Lambda ^ { n } ]
$$

We say that a point is forgotten if it is classified correctly at initialization, but not so at the end of training. Let $\mathbf { x }$ denote a forgotten datapoint, and let $N$ denote the number of GD steps at the end of training. Since by definition $f ( \mathbf { x } ) = \mathrm { s i g n } \overline { { ( } } w ^ { ( n ) } \mathbf { x } )$ , it follows that $\mathbf { x }$ is forgotten iff $\{ { \pmb w } ^ { ( 0 ) } y { \bf x } > 0 \}$ and $\{ \pmb { w } ^ { ( N ) } y \mathbf { x } < 0 \}$ .

Let us define the forget time of point $\mathbf { x }$ as follows:

Definition 1 (Forget time). $G D$ iteration $\hat { n }$ that satisfies

$$
\begin{array} { r l } { { \pmb w } ^ { ( \hat { n } ) } y { \bf x } \le 0 } & { { } } \\ { { \pmb w } ^ { ( n ) } y { \bf x } > 0 } & { { } \quad \forall n < \hat { n } } \end{array}
$$

Claim 1. Each forgotten point has a finite forget time $\hat { n }$ .

Proof. Since $\{ { \pmb w } ^ { ( 0 ) } y { \bf x } > 0 \}$ and $\{ { \pmb w } ^ { ( N ) } y { \bf x } < 0 \}$ , (5) follows by induction. □

Note that Def 1 corresponds with the Forget time seen in deep networks (cf. Fig. 3b). The empirical investigation of this correspondence is discussed in App. B (see Fig. 7).

To characterize the time at which a point is forgotten, we inspect the rate with which $F ( n ) = \bar { { \pmb w } } ^ { ( n ) } y { \bf x }$ changes with $n$ . We begin by assuming that the learning rate $\gamma$ is infinitesimal, so that terms of magnitude $O ( \gamma ^ { 2 } )$ can be neglected.

Using (4) and the Taylor expansion of $\lambda _ { j }$ from (3)

$$
\begin{array} { l } { { { \cal F } ( n ) \approx \displaystyle \left( w ^ { ( 0 ) } - w ^ { o p t } \right) \Lambda ^ { n } y \bf x + w ^ { o p t } y \bf x } } \\ { { { \mathrm { } } } } \\ { { { \mathrm { } } = w ^ { o p t } y \bf x + \displaystyle \sum _ { j = 1 } ^ { d } ( w _ { j } ^ { ( 0 ) } - w _ { j } ^ { o p t } ) \lambda _ { j } ^ { n } y x _ { j } } } \\ { { { \mathrm { } } } } \\ { { { \mathrm { } } = w ^ { o p t } y \bf x + \displaystyle \sum _ { j = 1 } ^ { d } ( w _ { j } ^ { ( 0 ) } - w _ { j } ^ { o p t } ) [ 1 - n \gamma s _ { j } L + O ( \gamma ^ { 2 } ) ] y x _ { j } } } \\ { { { \mathrm { } } } } \\ { { { \mathrm { } } = w ^ { ( 0 ) } y \bf x - n \gamma L \displaystyle \sum _ { j = 1 } ^ { d } ( w _ { j } ^ { ( 0 ) } - w _ { j } ^ { o p t } ) y s _ { j } x _ { j } + O ( \gamma ^ { 2 } ) } } \end{array}
$$

It follows that

$$
\frac { d F ( n ) } { d n } = - \gamma y L \sum _ { j = 1 } ^ { d } ( w _ { j } ^ { ( 0 ) } - w _ { j } ^ { o p t } ) s _ { j } x _ { j } + O ( \gamma ^ { 2 } )
$$

Discussion. Recall that $\{ s _ { j } \}$ is the set of singular values, ordered such that $s _ { 1 } \geq s _ { 2 } \geq \cdot \cdot \cdot \geq s _ { d } .$ , and $x _ { j }$ is the projection of point $\mathbf { x }$ onto the $j ^ { \mathrm { t h } }$ eigenvector. From (6), the rate at which a point is forgotten, if at all, depends on vector $[ s _ { j } x _ { j } ] _ { j }$ , in addition to the random vector ${ \pmb w } ^ { ( 0 ) } - { \pmb w } ^ { \mathrm { o p t } }$ and label $y$ . All else being equal, a point will be forgotten faster if the length of its spectral decomposition vector $[ x _ { j } ]$ is dominated by its first components, indicating that most of its mass is concentrated in the leading principal components.

# 4.2 Spectral Properties of Forgotten Images

When working with datasets of natural images, where it has been shown that the singular values decrease rapidly at an approximately exponential rate (Hyva¨rinen, Hurri, and Hoyer 2009), the role of the singular values becomes even more pronounced. Hacohen and Weinshall (2022) argued that in the limiting case, the components of the separating hyperplane $w ^ { o p t }$ will be learned sequentially, one at a time. In essence, the model first captures the projection of ${ \pmb w } ^ { o p t }$ onto the data’s leading eigenvector, then onto the subsequent eigenvectors in order. For similar considerations, this reasoning also holds in the multi-class scenario.

This analysis suggests that PCA of the raw data governs the learning of the linear separator. We therefore hypothesize that forgotten points with substantial projections onto the leading principal components are more likely to be forgotten early, and vice versa. To empirically test this prediction, we must first establish some key definitions.

Let $W ^ { o p t } \ \in \ \mathbb { R } ^ { c \times d }$ denote the optimal solution of the multi-class linear model with $c$ classes and the $L _ { 2 }$ loss. Let $W ( k )$ denote the projection of $W ^ { o p t }$ on the first $k$ principal components of the raw data.

Definition 2. Let $\displaystyle \mathcal { S } ( k )$ denote the set of points that are correctly classified by $\dot { W } ( k ^ { \prime } )$ for some $k ^ { \prime } > k$ , but incorrectly classified by $W ^ { o p t }$ . Similarly, let $\mathcal { M } ( n )$ denote the set of points correctly classified by the trained deep model after $n ^ { \prime } > n$ epochs, but incorrectly classified by the final model.

To empirically investigate the prediction above, we correlate the two sets $\displaystyle \mathcal { S } ( k )$ and $\mathcal { M } ( n )$ after establishing correspondence $n = \alpha k + \beta$ between the ranges of indices $k$ and $n$ . We examined this correlation using the CIFAR100 dataset, a linear model trained using the images’ RGB representation, and the corresponding deep model from the experiments reported in Section 6. Interestingly, the respective sets $\boldsymbol { \mathscr { S } } ( \boldsymbol { k } )$ and $\mathcal { M } ( n )$ show significant correlation, as seen in Fig. 4. Since deep networks also learn a representation, we repeated the experiment with alternative learned feature spaces, obtaining similar results (see Fig. 8 in App. B).

![](images/0346333ced4bda1158aaa3a394e5098041780e808baaf54800c569348e3b5547.jpg)  
Figure 4: Empirical results, correlating the sets of examples forgotten during the training of a DNN and those forgotten during the training of a deep linear network. Note in (d) that early on, roughly $\mathbf { \bar { \Sigma } } _ { 6 } ^ { 1 }$ of the points to be forgotten by our deep model are also forgotten by the deep linear model.

# 5 Recover Forgotten Knowledge: Algorithm

In Section 3 we showed that neural networks often achieve better performance in mid-training on a subset of the test data, even when the test accuracy is monotonically increasing with training epochs. Here we aim to integrate the knowledge obtained in mid- and post-training epochs, during inference time, in order to improve performance. To this end we must determine: (i) which versions of the model to use; (ii) how to combine them with the post-training model; and (iii) how to weigh each model in the final ensemble.

Choosing an early epoch of the network. Given a set of epochs $\{ 1 , \ldots , E \}$ and corresponding forget rates $\{ F _ { e } \} _ { e }$ , we first single out the model $n _ { A }$ obtained at epoch $A \ =$ $a r g m a x _ { e \in \{ 1 , . . . , E \} } F _ { e }$ . This epoch is most likely to correctly fix mistakes of the model on ”forgotten” test data.

Combining the predictors. Next, using validation data we determine the relative weights of the two models - the final model $n _ { E }$ , and the intermediate model $n _ { A }$ with maximal forget fraction. Since the accuracy of $n _ { E }$ is typically much higher than $n _ { A }$ , and in order not to harm the ensemble’s performance, we expect to assign $n _ { E }$ a higher weight.

Improving robustness. To improve our method’s robustness to the choice of epoch $A$ , we use a window of epochs around $A$ , denoted by $\left\{ { { n } _ { A - w } } , . . . , { { n } _ { A } } , . . . , { { n } _ { A + w } } \right\}$ . The vectors of probabilities computed by each checkpoint are averaged before forming an ensemble with $n _ { E }$ . In our experiments, we use a fixed window $w = 1$ , achieving close to optimal results as verified in the ablation study (see App. G.9). Iterative selection of models. As we now have a new predictor, we can find another alternative predictor from the training history that maximizes accuracy on the data misclassified by the new predictor, in order to combine their knowledge as described. This can be repeated iteratively, until no further improvement is achieved.

Choosing hyper-parameters. In order to compute $F _ { e }$ and assign optimal model weights and window size, we use a validation set, which is a part of the labeled data not shown to the model during initial training. This is done post training as it has no influence over the training process, and thus doesn’t incur additional training cost. We follow common practice, and show in App. G.1 that after optimizing these hyper-parameters, it is possible to retrain the model on the complete training set while maintaining the same hyperparameters. The performance of our method thus trained is superior to alternative methods trained on the same data.

Pseudo-code for our method. We name our method KnowledgeFusion (KF), and provide its pseudo-code in Alg 1. There, we call functions that: (i) calculate the forget value per epoch on some validation data, given the predictions at each epoch (calc early forget); and (ii) calculate the probability of each class for a given example and a list of predictors (get class probs).

Algorithm 1: Knowledge Fusion (KF)   

<html><body><table><tr><td>Input:Checkpoints of trained model {no..,.nE},w, test-pt x Output: prediction for x</td></tr><tr><td>{A1,.,Ak},{ε1,.,εk} ←calc_early_forget({no..,nE}) prob ← get_class-probs[E]</td></tr><tr><td>fori←1to k do probA ← mean(get_class-probs[Ai-w:Ai+wl)</td></tr><tr><td>prob←εi*probA+(1-εi)*prob end for</td></tr><tr><td>prediction←argmax(prob)</td></tr><tr><td>Return prediction</td></tr></table></body></html>

Knowledge distillation post-processing. The proposed method enhances the performance of any trained model with a minor increase in training costs. However, ensemble classifiers incur higher inference costs. To address this, knowledge distillation can be employed with a further increase in training costs, to deliver a single model that achieves performance comparable to the ensemble while maintaining inference costs comparable to the original model.

# 6 Empirical Evaluation

# 6.1 Main Results

In this section we evaluate the performance of our method as compared to the original predictor, i.e. the network after training, and other baselines. We use various image classification datasets, neural network architectures, and training schemes. The main results are presented in Tables 1-3, followed by a brief review of our extensive ablation study and additional comparisons in Section 6.2. All references to appendices below are to be found in the complete archived version of this paper (Stern, Yaacoby, and Weinshall 2024).

Specifically, in Table 1 we report results while using multiple architectures trained on CIFAR-100, TinyImagenet and Imagenet, with different learning rate schedulers and optimizers. For comparison, we report the results of both the original predictor and some baselines. Additional results for scenarios connected to overfitting are shown in Table 2 and App. F, where we test our method on these datasets with injected symmetric and asymmetric label noise (see App. E), as well as on a real label noise dataset (Animal10N). Note that, as customary, the label noise exists only in the train data while the test data remains clean for model evaluation.

In Table 3 and App. F we compare our method to additional methods that adjust the training protocol itself, using both clean and noisy datasets. We employ these methods using the same network architecture as our own, after a suitable hyper-parameter tuning.

In each experiment we use half of the test data for validation, to compute our method’s hyper-parameters (the list of alternative epochs and $\left\{ \varepsilon _ { i } \right\}$ ), and then test the result on the remaining test data. The accuracy reported here is only on the remaining test data, averaged over three random splits of validation and test data, using different random seeds. In App. G.1 we report results on the original train/test split, where a subset of the training data is set aside for hyper-parameter tuning. As customary, these same parameters are later used with models trained on the full training set, demonstratively without deteriorating the results.

Baselines Our method incurs the training cost of a single model, and thus, following the methodology of (Huang et al. 2017), we compare ourselves to methods that require the same amount of training time. The first group of baselines includes methods that do not alter the training process:

• Single network: the original network, after training. • Horizontal ensemble (Xie, Xu, and Chuang 2013): this method uses a set of epochs at the end of the training, and delivers their average probability outputs (with the same number of checkpoints as we do). • Fixed jumps: this baseline was used in (Huang et al. 2017), where several checkpoints of the network, equally spaced through time, are taken as an ensemble.

The second group includes methods that alter the training protocol. While this is not a directly comparable set of methods, as they focus on a complementary way to improve performance, we report their results in order to further validate the usefulness of our method. This group includes Snapshot ensemble (Huang et al. 2017), Stochastic Weight Averaging (SWA) (Izmailov et al. 2018) and Fast Geometric Ensembling (FGE) (Garipov et al. 2018), see details in App. D. Comparisons to additional baselines that are relevant to resisting overfitting, including early stopping and test time augmentation, are discussed in App. G.5. Full implementation details are provided in App. E.

# 6.2 Ablation Study

We conducted an extensive ablation study in order to investigate the limitations, and some practical aspects, of our

Table 1: Mean (over random validation/test splits) test accuracy (in percent) and standard error on image classification datasets, comparing our method and baselines described in the text. The last row shows the improvement of the best performer over the single network. Suffixes: $( i )$ denotes a limited budget scenario, in which we use our method in a non-iterative manner; $( \infty )$ denotes the unlimited budget scenario, where we use our full iterative version. In each case, the baselines employ the same number of checkpoints as our method.   

<html><body><table><tr><td rowspan="2">Method/Dataset architecture</td><td rowspan="2">CIFAR-100 Resnet18</td><td rowspan="2">TinyImagenet Resnet18</td><td colspan="4">Imagenet</td></tr><tr><td>Resnet50</td><td>ConvNeXt large</td><td>ViT16 base</td><td>MaxViT tiny</td></tr><tr><td>single network</td><td>78.07 ± .28</td><td>64.95 ± .24</td><td>75.74 ± .14</td><td>82.92 ± .11</td><td>79.16± .1</td><td>82.51 ± .15</td></tr><tr><td>horizontal (i)</td><td>78.15 ± .17</td><td>64.89±.18</td><td>76.46 ± .14</td><td>83.13±.1</td><td>79.11 ± .1</td><td>82.77 ± .1</td></tr><tr><td>fixed jumps (i)</td><td>78.04 ± .23</td><td>66.54± .35</td><td>75.5 ± .09</td><td>82.37 ± .1</td><td>78.67± .08</td><td>83.38 ± .1</td></tr><tr><td>KF(ours) (i)</td><td>78.33 ± .08</td><td>66.98 ± .37</td><td>75.88 ± .14</td><td>83.18± .16</td><td>79.93 ± .11</td><td>83.34 ± .04</td></tr><tr><td>horizontal ()</td><td>78.23±.17</td><td>65.11 ± .3</td><td>76.42 ± .1</td><td>83.02±.06</td><td>79.53± .13</td><td>82.93±.14</td></tr><tr><td>fixed jumps (∞)</td><td>79.17 ± .08</td><td>68.24± .38</td><td>75.72 ± .18</td><td>83.86 ± .06</td><td>79.11 ± .13</td><td>83.78 ± .15</td></tr><tr><td>KF(ours)(0)</td><td>79.13 ± .14</td><td>68.5 ± .36</td><td>76.52 ± .16</td><td>83.96 ± .09</td><td>80.34± .08</td><td>83.81 ± .14</td></tr><tr><td>improvement</td><td>1.05 ± .14</td><td>3.54 ± .14</td><td>.78±.04</td><td>1.03 ± 13</td><td>1.17 ± .08</td><td>1.29 ± .02</td></tr></table></body></html>

Table 2: Mean test accuracy (in percent) and standard error of Resnet 18, comparing our method and the baselines on datasets with large label noise and significant overfitting. We include the Animal10N dataset, which has innate label noise.   

<html><body><table><tr><td rowspan="2">Method/Dataset % label noise</td><td rowspan="2">Animal10N 8%</td><td colspan="2">CIFAR-100 asym</td><td colspan="2">CIFAR-100 sym</td><td colspan="2">TinyImagenet</td></tr><tr><td>20%</td><td>40%</td><td>20%</td><td>40%</td><td>20%</td><td>40%</td></tr><tr><td>single network</td><td>85.9 ± .3</td><td>67.1 ± .5</td><td>49.4± .3</td><td>65.4± .3</td><td>56.9± .1</td><td>56.2± .2</td><td>49.8± .3</td></tr><tr><td>fixed jumps ()</td><td>87.1±.4</td><td>73.9±.1</td><td>59.9±.6</td><td>72.8± .1</td><td>66.5±.1</td><td>60.0±.8</td><td>54.16 ± .3</td></tr><tr><td>horizontal()</td><td>86.3± .3</td><td>73.4 ± .1</td><td>58.5 ± .1</td><td>71.1 ± .38</td><td>65.2 ± .1</td><td>59.3± .3</td><td>51.7 ± .2</td></tr><tr><td>KF(ours) (00)</td><td>87.8± .4</td><td>74.2± .1</td><td>62.1± .5</td><td>72.8 ± .1</td><td>67.0±.1</td><td>62.8± .2</td><td>57.0±.5</td></tr><tr><td>improvement</td><td>1.9±.4</td><td>7.1± .6</td><td>12.6±.2</td><td>7.4±.4</td><td>10.1± .1</td><td>6.6±.1</td><td>7.2±.1</td></tr></table></body></html>

<html><body><table><tr><td>Method/Dataset</td><td>CIFAR-100</td><td>Animal10N</td><td colspan="2">CIFAR-100 asym</td><td colspan="2">CIFAR-100 sym</td></tr><tr><td>% label noise</td><td>0%</td><td>8%</td><td>20%</td><td>40%</td><td>20%</td><td>40%</td></tr><tr><td>FGE(00)</td><td>78.9± .4</td><td>86.5 ±0.6</td><td>67.1± .2</td><td>48.1± .3</td><td>66.5 ± .1</td><td>52.1 ± .1</td></tr><tr><td>SWA (0o)</td><td>78.8 ±.1</td><td>88.1± .2</td><td>66.6± .1</td><td>46.9± .2</td><td>65.6± .4</td><td>50.0± .1</td></tr><tr><td>snapshot (0)</td><td>78.4± .1</td><td>86.8± .3</td><td>72.1 ± .4</td><td>52.8± .6</td><td>70.8± .5</td><td>63.8± .2</td></tr><tr><td>KF(ours) (∞)</td><td>79.3± .2</td><td>87.8± .4</td><td>74.2± .1</td><td>62.1± .5</td><td>72.8±.1</td><td>67.0± .1</td></tr></table></body></html>

Table 3: Mean test accuracy of Resnet18, using for baseline methods that alter the training procedure.

method. Due to space limitation, we only provide here a brief overview of the results, and postpone the full description to App. G. The results can be summarized as follows:

(i) $\ S \mathrm { G } . 1$ shows that a separate validation set is not really necessary for the method to work well. (ii) $\ S \mathrm { G } . 2$ investigates how many checkpoints are needed for the method to be effective, showing that only $5 - 1 0 \%$ of the past checkpoints are sufficient. (iii) $\ S \mathrm { G } . 3$ investigates the added value of our method when using only a partial hyper-parameter search, which leads to sub-optimal training. Interestingly, our method is shown to be even more beneficial in the suboptimal regime, with a smaller gap between the optimal and sub-optimal networks. (iv) $\ S \mathrm { G } . 4$ shows that our method is effective in a transfer learning scenario, when using a pre-trained network. (v) $\ S \mathrm { G } . 5$ shows that our method outperforms Exponential-Moving-Average (EMA), early stopping and test time augmentation. (vi) $\ S \mathrm { G } . 6$ shows that our method’s benefit increases as the number of parameters grows. (vii) $\ S \mathrm { G } . 7$ shows that much of the improvement of a regular ensemble of independent networks can often be obtained by using our method at a much lower cost. (viii) $\ S \mathrm { G } . 8$ shows that our method does not have negative effects on the model’s fairness. (ix) $\ S \mathrm { G } . 9$ shows that using a window of size $\mathrm { w } { = } 1$ is both necessary and near optimal.

# 7 Summary and Conclusions

We revisited the problem of overfitting in deep learning, proposing to track the forgetting of validation data in order to detect local overfitting. We connected our new perspective with the epoch wise double descent phenomenon, empirically extending its scope while demonstrating that a similar effect occurs in benchmark datasets with clean labels. Inspired by these new empirical observations, we constructed a simple yet general method to improve classification at inference time. We then empirically demonstrated its effectiveness on many datasets and modern network architectures. The method improves modern networks by around $1 \%$ accuracy over Imagenet, and is especially useful in some transfer learning settings where its benefit is large and its overhead is very small. Most importantly, the success of the method to improve upon the original model shows that indeed models forget useful knowledge at the late stages of learning, and serves as a proof of concept that recovering this knowledge can be useful to improve performance.

# Acknowledgments

This work was supported by grants from the Israeli Council of Higher Education and the Gatsby Charitable Foundation.