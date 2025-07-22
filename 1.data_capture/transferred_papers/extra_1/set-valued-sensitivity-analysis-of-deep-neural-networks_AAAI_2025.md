# Set-Valued Sensitivity Analysis of Deep Neural Networks

Xin Wang1, Feilong Wang2, Xuegang (Jeff) Ban1

1Department of Civil and Environmental Engineering, University of Washington, Seattle, WA, 98195, United States 2School of Transportation and Logistics, Southwest Jiaotong University, Chengdu, 610032, China xinw22@uw.edu, Flwang $@$ swjtu.edu.cn, banx $@$ uw.edu

# Abstract

This paper proposes a sensitivity analysis framework based on set-valued mapping for deep neural networks (DNN) to understand and compute how the solutions (model weights) of DNN respond to perturbations in the training data. As a DNN may not exhibit a unique solution (minima) and the algorithm of solving a DNN may lead to different solutions with minor perturbations to input data, we focus on the sensitivity of the solution set of DNN, instead of studying a single solution. In particular, we are interested in the expansion and contraction of the solution set in response to data perturbations. If the change of solution set can be bounded by the extent of the data perturbation, the model is said to exhibit the Lipschitz-like property. This ‘set-to-set’ analysis approach provides a deeper understanding of the robustness and reliability of DNNs during training. Our framework incorporates both isolated and non-isolated minima, and critically, does not require the Hessian of loss function being nonsingular. By developing set-level metrics such as distance between sets, convergence of sets, derivatives of set-valued mapping, and stability across the solution set, we prove that the solution set of the Fully Connected Neural Network holds Lipschitz-like properties. For general neural networks (e.g. Resnet), we also develop a novel method to estimate the new solution set following data perturbation without retraining.

Extended version — https://arxiv.org/abs/2412.11057

# Introduction

Sensitivity analysis is a classic topic that crosses optimization (Fiacco 1983), machine learning, and deep learning (Yeung et al. 2010; Koh and Liang 2017; Christmann and Steinwart 2004). It studies how the solution of a model responds to minor perturbations in hyperparameters or input data. For example, Christmann and Steinwart (2004) proved that the solution of a classification model with convex risk function is robust to bias in the data distribution. Here we focus on deep neural networks (DNN), for which the solution is the weights of DNN trained (optimized) on input data. We are concerned about how the solution of DNN responds to perturbations in the input data in the training stage of the model.

In the domain of deep learning (e.g., DNN), sensitivity analysis has drawn attention due to its wide range of applications, such as designing effective data poisoning attacks (Mu˜noz-Gonza´lez et al. 2017; Mei and Zhu 2015), evaluating the robustness of models (Weng et al. 2018), and understanding the impact of important features in training data on the prediction (Koh and Liang 2017). Define a neural network $f : \mathcal X \overset { } { \underset { } {  } } \mathcal y$ , where $\mathcal { X }$ (e.g., images) is the input space and $y$ (e.g., labels) is the output space. Given training data samples $x = [ x _ { 1 } , x _ { 2 } , \ldots , x _ { n } ]$ and $y = [ y _ { 1 } , y _ { 2 } , \dotsc , y _ { n } ]$ , and the loss function $L$ , the empirical risk minimizer is given by $\begin{array} { r } { \hat { w } \stackrel { \mathrm { d e f } } { = } \operatorname { a r g m i n } _ { w \in \mathcal { W } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } L \left( x _ { i } , y _ { i } , w \right) } \end{array}$ . This paper assumes that we perturb only the features, keeping the label constant. The learning process from data to local minimizers thus can be formulated as a set-valued mapping $S : \mathcal { X } ^ { n } \overset { } { \underset { } {  } } \mathcal { W } ^ { 1 }$ ,

$$
S ( x ) = \left\{ \hat { w } | \hat { w } \stackrel { \mathrm { \scriptsize ~ d e f } } { = } \arg \operatorname* { m i n } _ { w \in { \mathcal W } } { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } L \left( x _ { i } , y _ { i } , w \right) \right\} .
$$

For the unperturbed dataset $\bar { x } = [ \bar { x } _ { 1 } , \bar { x } _ { 2 } , . . . , \bar { x } _ { n } ]$ and $\bar { w } \in$ $S ( \bar { x } )$ , current sensitivity analysis (Koh and Liang 2017; Nickl et al. 2024; Christmann and Steinwart 2004) aims to study the change of $S ( \bar { x } )$ when an individual point $\bar { x } _ { p }$ is perturbed to $x _ { p }$ . Specifically, if the data $\bar { x }$ is perturbed to $\hat { x } = [ \bar { x } _ { 1 } , \bar { x } _ { 2 } , \ldots \bar { , } x _ { p } , \ldots , \bar { x } _ { n } ]$ , the sensitivity analysis can be conducted by examining a limit:

$$
\operatorname* { l i m } _ { x _ { p }  \bar { x } _ { p } } \frac { S ( \hat { x } ) - S ( \bar { x } ) } { \| x _ { p } - \bar { x } _ { p } \| } .
$$

Most current sensitivity analysis methods for DNN suffer from the following two issues. First, to figure out the sensitivity of solution $w$ w.r.t. data $x _ { p }$ , one of the most common approaches (e.g., the influence function (Koh and Liang 2017)) is to apply the Dini implicit function theorem (Dontchev and Rockafellar 2009) to the optimality condition of the model: $\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \nabla _ { w } L \left( \bar { x } _ { i } , \bar { y } _ { i } , \bar { w } \right) = 0 } \end{array}$ , which leads to:

$$
\begin{array} { r l r } { \displaystyle \operatorname* { l i m } _ { x _ { p }  \bar { x } _ { p } } \frac { S ( \hat { x } ) - S ( \bar { x } ) } { \| x _ { p } - \bar { x } _ { p } \| } = \nabla _ { x _ { p } } w \bigg | _ { x _ { p } = \bar { x } _ { p } } } & \\ { \displaystyle } & { \quad } & { \quad = - H _ { \bar { w } } ^ { - 1 } \nabla _ { x _ { p } } \nabla _ { \theta } L \big ( \bar { x } _ { p } , \bar { y } _ { p } , \bar { w } \big ) , } \end{array}
$$

where $\begin{array} { r c l } { H _ { \bar { w } } ^ { - 1 } } & { = } & { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \nabla _ { w } ^ { 2 } L \left( \bar { x } _ { i } , \bar { y } _ { i } , \bar { w } \right) } \end{array}$ is the Hessian. However, the application of the Dini implicit function theorem cannot apply when the Hessian $H _ { \bar { w } }$ is not invertible due to the often non-locally strong convex loss function $L$ of DNN (Li et al. 2018). Second, the current approaches (Koh and Liang 2017; Nickl et al. 2024) assume $S$ is a singlevalued mapping, omitting the fact that $S$ is often set-valued. Research has shown that DNN may not exhibit a unique solution $S ( x )$ even when $L$ is convex. For example, it was noticed in Li et al. (2018) that the stochastic gradient descent (SGD) method can find flat minima (solutions located in the flat valley of the loss landscape); others found that all SGD solutions for DNN may form a manifold (Benton et al. 2021; Cooper 2021). When an application (e.g., a data poisoning attack) is designed and evaluated based on only one of the solutions, it overlooks the fact that the learning algorithm may converge to other solutions during re-training.

In this paper, we incorporate the fact that $S$ is often setvalued into the sensitivity analysis framework for DNNs. This extends the scope of sensitivity analysis in DNNs from focusing on a single solution to a solution set, shifting from the traditional ‘point-to-point’ analysis to a ‘set-to-set’ paradigm. That is, we study how the solution set of a DNN expands and contracts in response to data perturbations. The proposed approach covers more general situations in risk minimization of DNN, including isolated local minima, nonisolated minima, and minima that consist of a connected manifold. More importantly, it directly deals with the solution sets without the assumption of non-singular Hessian matrix, offering a more complete understanding of DNN.

Considering $S$ as set-valued, with the isolated minima a special case where solution set $S ( x )$ contains only one element, the sensitivity analysis aims to study i) whether the limit (2) exists, and ii) whether the limit (2) is bounded. However, directly studying the limit when mapping $S$ is setvalued can be challenging. On the one hand, $S ( x )$ may be a large or even infinitely large set (e.g., a manifold), which makes it complicated to analyze. To address this, we instead focus on the change of the solution set $S ( x )$ within a small neighborhood. Given a pair $( { \bar { x } } , { \bar { w } } ) , { \bar { w } } \in S ( { \bar { x } } )$ , we study the change of $S ( x ) \cap U$ , where $U$ is a neighborhood of $\bar { w }$ , in response to the data perturbation within a neighborhood $V$ of $\bar { x }$ . On the other hand, as $S ( x )$ and $S ( \bar { x } )$ are sets rather than single solutions, the subtraction operation between sets does not exist (set operations only include union, intersection, and set difference), which invalidates the limit (2). Consequently, we leverage the Lipchitzlike property of $S$ around $( \bar { x } , \bar { w } )$ , to measure the change of $S ( x ) \cap U$ . We say that $S$ holds Lipchitz-like property around $( \bar { x } , \bar { w } ) \in \mathrm { { g p h } } \bar { S } : = \{ ( x , w ) | w \in S ( x ) \bar  \}$ , if there exist neighborhoods $U$ of $\bar { w } , V$ of $\bar { x }$ and a positive real number $\kappa$ such that (Dontchev and Rockafellar 2009) 2

$$
S ( x ^ { \prime } ) \cap U \subset S ( x ) + \kappa \left\| x - x ^ { \prime } \right\| \mathbb { B } , \quad \forall x ^ { \prime } , x \in V ,
$$

where $\mathbb { B }$ is a closed unit ball. The Lipschitz-like property is an extension of Lipschitz continuity defined for singlevalued functions to set-valued mappings. $\kappa$ , as a scalar, is the Lipschitz modulus that describes the upper bound of the solution set’s change in response to data perturbations. The existence of the limit (2) can be interpreted as establishing a

Lipschitz-like property. The bound of the limit (2) is equivalent to a bounded $\kappa$ that characterizes how sensitive the model solution is w.r.t input data, defined for the training stage. It complements the studies on Lipschitz constants of DNNs (Fazlyab et al. 2019; Virmaux and Scaman 2018) that have been only defined and studied so far for the inference stage of DNNs (with fixed model solutions).

In our “set-to-set” analysis framework, the study of existence and boundedness of limits (2) is transferred as the following question: Given a local minimizer $\bar { w }$ with its neighborhoods U, and training data $\bar { x }$ with its neighborhoods $\mathrm { v }$ , where $( \bar { x } , \bar { w } ) \in \mathrm { g p h } S$ , does the solution mapping $S$ satisfy the Lipschitz-like property in a neighborhood of $( \bar { x } , \bar { w } )$ , and if so, how can we estimate the associated bounded Lipschitz modulus? Moreover, we also explore how, when we perturb the data $\bar { x }$ to $x ^ { p }$ , with $x ^ { p } \in V$ , we can identify the new solution set $S \left( x ^ { p } \right) \cap U$ without re-training.

We prove that, for the Deep Fully Connected Neural Network (DFCNN) with the Relu activation function, the solution set $S$ holds the Lipschitz-like property. A bound of the Lipschitz modulus is also provided. This reveals that the solution set of a DFCNN will not deviate significantly when there are biases in the training data, allowing us to estimate the new solution set based on the behavior of $S$ around $( \bar { x } , \bar { w } )$ . We also introduce graphical derivatives to capture the local linear approximation of $S$ around $( \bar { x } , \bar { w } )$ for a DNN (not only for DFCNN). The graphical derivative based method can accurately estimate $S ( x ^ { p } ) - { \bar { w } }$ , providing solutions following data perturbations for a DNN (e.g., Resnet56) with near zero training loss. In particular, when $S ( x ^ { p } )$ is single-valued, i.e. the solution set $S ( x ^ { p } )$ within neighborhood $U$ includes only one solution, our graphical derivative based method is equivalent to the influence function (Koh and Liang 2017).

The contribution of our paper is summarized as follows:

1) We introduce a set-valued mapping approach to understand the sensitivity of solutions of DNN in relation to perturbations in the training data. Our framework accommodates both isolated and non-isolated minima without relying on convex loss assumption.

2) We prove that the solution mapping of DFCNN holds the Lipschitz-like property (and thus stable during training) and estimate a bound for the Lipschitz modulus.

3) We propose a graphical-derivative-based method to estimate the new solution set of a DNN when the training data are perturbed, and simulate it using the Resnet56 with the CIFAR-10 dataset.

# Preliminary Knowledge

This section briefly summarizes the necessary preliminary knowledge for sensitive analysis of set-valued mapping, covering the distance between sets, sets convergence, and the generalized derivative. These concepts will be used to characterize the behavior of set-valued mapping and play a key role in defining the criteria for Lipschitz-like property.

Definition 1. (Distance between sets) The distance from a point $w$ to a set $C$ is

$$
d _ { C } ( w ) = d ( w , C ) = \operatorname* { i n f } _ { w ^ { \prime } \in C } | w - w ^ { \prime } | .
$$

$$
\frac { c } { \vdash e ( C , D ) \longleftarrow \bigcap _ { \mathbf { \theta } } ^ { } } \frac { \vdash e ( D , C ) \longleftarrow } { \scriptscriptstyle { \mathbb { D } } }
$$

Figure 1: Illustration of $e ( C , D )$ and $e ( D , C )$ , where $C$ and $D$ are two closed sets. The Pompeiu-Hausdorff Distance of $C$ beyond $D$ is $h ( C , D ) = \bar { \operatorname * { m a x } } \{ e ( C , D ) , e ( D , C ) \} =$ $e ( D , C )$ for this example.

For sets $C$ and $D$ , the Pompeiu-Hausdorff Distance of $C$ beyond $D$ is defined by

$$
h ( C , D ) = \operatorname* { m a x } \{ e ( C , D ) , e ( D , C ) \} ,
$$

where

$$
e ( C , D ) = \operatorname* { s u p } _ { w \in C } d ( w , D ) , e ( D , C ) = \operatorname* { s u p } _ { w \in D } d ( w , C ) .
$$

An illustrative figure is provided in Figure 1, where $C$ and $D$ are two segments.

Definition 2. (Painleve´-Kuratowski Set convergence) Given a set-valued mapping $S : X  W$ , the Painlev´eKuratowski outer set limit as $x  \bar { x }$ is

$\operatorname* { l i m } _ { x \to \bar { x } } \operatorname* { s u p } _ { } { S ( x ) } : = \{ w \in W \mid \exists s e q u e n c e s \ x _ { k } \to \bar { x } $ such that $w _ { k } \in S ( x _ { k } ) \to w \}$

the Painlev´e-Kuratowski inner set limit as $x \to \bar { x }$ is

$$
\begin{array} { r l } & { \underset { x  \bar { x } } { \operatorname* { l i m } \operatorname* { i n f } } S ( x ) : = \{ w \in W \vert \forall s e q u e n c e s x _ { k }  \bar { x } , } \\ & { } \\ & { w _ { k } \in S ( x _ { k } )  w \} . } \end{array}
$$

Definition 3. $A$ vector $\eta$ is tangent to a set $\Gamma$ at a point $\bar { \gamma } \in \Gamma$ , written $\eta \in T _ { \Gamma } ( \bar { \gamma } )$ , if

$$
\frac { \gamma _ { i } - \bar { \gamma } } { \tau _ { i } }  \eta f o r \ : s o m e \gamma _ { i }  \bar { \gamma } , \gamma _ { i } \in \Gamma , \tau _ { i } \setminus 0 .
$$

Where $T _ { \Gamma } ( \bar { \gamma } )$ is the tangent cone to $\Gamma$ at $\bar { \gamma }$ .

Definition 4. Given a convex set $\Gamma$ in $\mathbb { R } ^ { n }$ and a point $\bar { \gamma }$ in $\Gamma$ , the normal cone to $\Gamma$ at $\bar { \gamma }$ , denoted $N _ { \Gamma } ( \bar { \gamma } )$ , is defined as the set of all vectors $\xi \in \mathbb { R } ^ { n }$ that satisfy the condition:

$$
N _ { \Gamma } ( \bar { \gamma } ) = \{ \xi \in \mathbb { R } ^ { n } : \langle \xi , \gamma - \bar { \gamma } \rangle \leq 0 f o r a l l \gamma \in \Gamma \} .
$$

Remark 1: This study only focuses on the normal cone and tangent cone of convex sets. Refer to Appendix C for an illustrative diagram of Definition 3 and Definition 4.

We next define the generalized derivative that includes both the graphical derivative and coderivative. The graphical derivative is a concept used in the analysis of set-valued mappings (multifunctions). It generalizes the derivative of functions to set-valued mappings, capturing the local behavior of a set-valued mapping around a particular point. The coderivative complements the graphical derivative by offering a reverse perspective: instead of describing how outputs respond to changes in inputs, it examines how inputs need to change to achieve specific output behaviors.

Definition 5. (Generalized derivatives)3 Consider a mapping $S : \mathbb { R } ^ { n } \implies \mathbb { R } ^ { m }$ and a point $\bar { x } \in \operatorname { d o m } S$ . The graphical derivative of $S$ at $\bar { x }$ for any $\bar { w } \in S ( \bar { x } )$ is the mapping $D S ( \bar { x } \mid \bar { u } ) : \mathbb { R } ^ { n } \implies \mathbb { R } ^ { m }$ defined by

$$
v \in D S ( \bar { x } \mid \bar { w } ) ( \mu ) \Longleftrightarrow ( \mu , v ) \in T _ { \mathrm { g p h } S } ( \bar { x } , \bar { w } ) ,
$$

whereas the coderivative is the mapping $D ^ { * } S ( \bar { x } \mid \bar { u } )$ : $\mathbb { R } ^ { m } \overset { } { \Rightarrow } \mathbb { R } ^ { n }$ defined by

$$
q \in D ^ { * } S ( \bar { x } \mid \bar { w } ) ( p ) \Longleftrightarrow ( q , - p ) \in N _ { \mathrm { g p h } S } ( \bar { x } , \bar { w } ) .
$$

Remark 1: Graphical derivative can also be expressed as:

$$
D S ( \bar { x } \mid \bar { w } ) ( \mu ) = \operatorname * { l i m } _ { \stackrel { \tau \searrow 0 } { \mu _ { k } \to \mu } } { \frac { S ( \bar { x } + \tau \mu _ { k } ) - \bar { w } } { \tau } }
$$

Remark 2: In the case of a smooth, single-valued mapping $F : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ , one has

$$
\begin{array} { r l } & { D F ( \bar { x } ) ( { \boldsymbol { \mu } } ) = \nabla F ( \bar { x } ) { \boldsymbol { \mu } } f o r a l l { \boldsymbol { \mu } } \in \mathbb { R } ^ { n } } \\ & { D ^ { * } F ( \bar { x } ) ( p ) = \nabla F ( \bar { x } ) ^ { * } p f o r a l l { \boldsymbol { p } } \in \mathbb { R } ^ { m } } \end{array}
$$

# Lipschitz-like Property of Deep Fully Connected Neural Network

This section studies the Lipschitz-like property of DNNs. We focus on DFCNN, a classical DNN, to demonstrate our main theorem results. For each data point $x _ { i } \in x , x _ { i } \in \mathbb { R } ^ { d }$ , the first layer’s weight matrix of a DFCNN is denoted as $W ^ { ( 1 ) } \in \mathbb { R } ^ { \dot { m } \times d }$ , and for each subsequent layer from 2 to $H$ , the weight matrices are denoted as $\bar { W } ^ { ( h ) } \in \mathbf { \bar { \mathbb { R } } } ^ { m \times m }$ . $\boldsymbol { a } \in \mathbb { R } ^ { m }$ is the output layer and the Relu function is given by $\sigma ( \cdot )$ . We recursively define a DFCNN, starting with $x _ { i } ^ { ( 0 ) } = x _ { i }$ for simplicity.

$$
\begin{array} { c } { { { x _ { i } } ^ { ( h ) } = \sigma \left( { W ^ { ( h ) } } { x _ { i } } ^ { ( h - 1 ) } \right) , 1 \le h \le H } } \\ { { { f ( x _ { i } , w ) } = a ^ { \top } { x _ { i } } ^ { ( H ) } . } } \end{array}
$$

Here $x _ { i } ^ { ( h ) }$ is the output of the $h$ -th layer. We denote $W : =$ $( W ^ { ( 1 ) } , \dots , W ^ { ( H ) } )$ as the weights of the network and $w : =$ $( w ^ { ( 1 ) } , \dots , w ^ { ( H ) } )$ as the vector of the flatten weights. In particular, $w ^ { ( h ) }$ is the vector of the flattened $h$ -th weight $W ^ { ( h ) }$ . Denote $d i m \big ( w ^ { ( h ) } \big ) = p ^ { ( h ) }$ and $\begin{array} { r } { d i m ( w ) = \sum _ { i = 1 } ^ { H } p ^ { ( h ) } = p } \end{array}$ .

For a DFCNN, we develop our method using the quadratic loss function: $L ( x _ { i } , y _ { i } , w ) = { \textstyle { \frac { 1 } { 2 } } } ( f ( w , x _ { i } ) - y _ { i } ) ^ { 2 } .   $ w, as the neural network weights, is a local/global minimum of empirical loss $\begin{array} { r } { { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } L \mathbf { \bar { ( } } x _ { i } , y _ { i } , w \mathbf { ) } } \end{array}$ . Since first-order optimization algorithms, such as SGD, are widely utilized, we employ the first-order optimality condition to characterize these minima. Let $\begin{array} { r } { \dot { R ( x , y , w ) ^ { * } } = \nabla _ { w } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } L \left( x _ { i } , y _ { i } , w \right) } \end{array}$ . Since the label vector $y = [ y _ { 1 } , \dotsc , y _ { n } ]$ is constant, we simplify the notation of $R ( x , y , w )$ to $R ( x , w )$ . Then the solution of a DFCNN can be characterized by the set-valued mapping $F$ :

$$
F ( x ) = \{ w | R ( x , w ) = \nabla _ { w } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } L \left( x _ { i } , y _ { i } , w \right) = 0 \} ,
$$

3 $A ^ { * }$ denotes the conjugate transpose of $A$ , where $A$ is a matrix

For layer $h$ , we define mapping $F _ { h }$ as:

$$
F _ { h } ( x ) = \{ w ^ { ( h ) } | R ( x , w ) = 0 \} .
$$

Following the classical sensitivity analysis (Koh and Liang 2017; Nickl et al. 2024), we first focus on one individual data $x _ { k } \in x = [ x _ { 1 } , . . . , x _ { n } ]$ , which is perturbed. In this case, $F ( x )$ and $F _ { h } ( x )$ in the above two equations are expressed as $\dot { F } ( x _ { k } )$ and $\dot { F _ { h } } ( x _ { k } )$ to indicate that only $x _ { k }$ are perturbed. In the next section, we present the case with multiple data perturbations.

Assumption 1. We assume that DNNs are overparameterized; under this assumption, a DFCNN has the capacity to memorize training data with zero training error, i.e. $p > d .$ .

Assumption 2. For given $\bar { x }$ and $\bar { w }$ , $[ \nabla _ { w } R ( \bar { x } , \bar { w } ) , \nabla _ { x _ { k } } R ( \bar { x } , \bar { w } ) ]$ is of full rank, where $\nabla _ { \boldsymbol { w } } R ( \bar { \boldsymbol { x } } , \bar { \boldsymbol { w } } ) \quad \in \quad \mathbb { R } ^ { p \times \gamma }$ , $\begin{array} { r l r } { \nabla _ { x _ { k } } R ( \bar { x } , \bar { w } ) } & { { } \in } & { \mathbb { R } ^ { p \times d } , p \quad = } \end{array}$ $d i m ( w ) , d = d i m ( x _ { k } )$ .

Remark 1: A DNN with non-singular Hessian matrix, as a special case, satisfies Assumption 2. On the other hand, a matrix that satisifies Assumption 2 is not necessarily be non-signular.

Remark 2: In the following section, we will find that although some DNNs do not always satisfy Assumptions 1 or 2, our algorithm can still provide a reasonably accurate estimation of the sensitivity of the solution mapping.

The Lipschitz-like property and solution set estimation following data perturbation rely on the generalized derivative (see definition 5). Theorem 1 below provides an explicit formulation for the generalized derivative of $F$ , enabling a convenient analysis of the local behavior of a solution mapping. It will be used in the proof of Theorem 2 that describes the Lipschitz-like property of the solution mapping.

Theorem 1. For given $\bar { x } _ { k }$ and $\bar { w }$ , the graphical derivative and coderivative of $F$ (defined by Definition $5$ ) at $\bar { x } _ { k }$ for $\bar { w }$ have the formulas:

$$
\begin{array} { r l } & { D F \left( \bar { x } _ { k } \mid \bar { w } \right) ( \mu ) = \{ v \mid \nabla _ { w } R ( \bar { x } , \bar { w } ) v + \nabla _ { x _ { k } } R ( \bar { x } , \bar { w } ) \mu = 0 \} } \\ & { D ^ { * } F \left( \bar { x } _ { k } \mid \bar { w } \right) ( p ) } \\ & { ~ = \left\{ q \mid ( q , - p ) = \left[ \nabla _ { x _ { k } } R ( \bar { x } , \bar { w } ) , \nabla _ { w } R ( \bar { x } , \bar { w } ) \right] ^ { \top } y \right\} } \\ & { w h e r e \quad y \in { \mathbb R } ^ { \mathrm { d i m } ( w ) } } \end{array}
$$

Proof: see appendix $A$ .

The following Theorem 2 then proves the Lipschitz-like property of DFCNN with the bound of the Lipshitz modulus. It reveals the potential training stability of DFCNN, i.e. the solution set will not change dramatically when perturbations are introduced to training data. The Lipschitz modulus is determined by the original solution $\bar { w }$ and input data $\bar { x }$ .

Theorem 2. For a given layer $h$ $1 \leq h \leq H )$ , mapping $F _ { h }$ holds the Lipschitz-like property. That is, for given $\bar { x } _ { k }$ and $\bar { w }$ , there exists neighborhoods $V _ { k }$ of $\bar { x } _ { k }$ and $U _ { h }$ of $\bar { w } ^ { ( h ) }$ , with a positive real number $\kappa _ { h }$ (the Lipschitz modulus) such that

$$
\begin{array} { r l } { F _ { h } \left( x _ { k } ^ { \prime } \right) \cap U _ { h } \subset F _ { h } ( x _ { k } ) + \kappa _ { h } \left. x _ { k } - x _ { k } ^ { \prime } \right. \mathbb { B } } & { } \\ { \forall x _ { k } ^ { \prime } , x _ { k } \in V _ { k } } \end{array}
$$

$$
\begin{array} { r l } & { \kappa _ { h } = \frac { \left\| \left[ \prod _ { k = 1 } ^ { H } W ^ { ( k ) ^ { \top } } \operatorname { d i a g } \left( \mathbf { 1 } \left( \sigma \left( W ^ { ( k ) } x _ { k } ^ { k - 1 } \right) > 0 \right) \right) \right] a \right\| } { \left\| \operatorname { d i a g } \left( \mathbf { 1 } \left( \sigma \left( W ^ { ( h ) } x _ { k } ^ { h - 1 } \right) > 0 \right) \right) \right. } } \\ & { \qquad \left[ \displaystyle \prod _ { k = h + 1 } ^ { H } W ^ { ( k ) ^ { \top } } \operatorname { d i a g } \left( \mathbf { 1 } \left( \sigma \left( W ^ { ( k ) } x _ { k } ^ { k - 1 } \right) > 0 \right) \right) \right] } \\ & { \qquad \left. a x _ { k } ^ { ( h - 1 ) ^ { \top } } \right\| _ { F } } \end{array}
$$

Proof: see appendix $B$ .

The following theorem 3 generalizes theorem 2 by considering the shift of the entire dataset instead of perturbing a single point.

Theorem 3. For given $\bar { x }$ and $\bar { w }$ , there exists neighborhoods $V$ of $\bar { x }$ and $U$ of $\bar { w } ^ { ( h ) }$ , with a positive real number $\kappa$ (i.e., the Lipschitz modulus) such that

$$
F _ { h } \left( x ^ { \prime } \right) \cap U \subset F _ { h } \left( x \right) + \kappa \left. x - x ^ { \prime } \right. \mathbb { B } \quad \forall x ^ { \prime } , x \in V
$$

Proof: see appendix $B$ .

Remark: The Lipschitz modulus in Theorems 2 or 3 measures the extent of deviation of the solution set when the data points are perturbed within a neighborhood of $\bar { x }$ . The Lipschitz modulus of DFCNN, which depends on the number of layers, the original solution, and the unperturbed data, can vary. However, it won’t be unbounded even if it can be large, indicating that the solution set won’t change dramatically.

# Sensitivity Analysis for Solution Set

Theorem 3 reveals that the solution set of DFCNN after perturbation will not deviate dramatically from the original solution set, allowing us to approximate this change using the local information around $( \bar { x } , \bar { w } )$ . This section first proposes a method to estimate the new solution set of DFCNN, given the perturbation in the training data. Instead of only perturbing a single individual data point in the previous section, this section perturbs multiple data points simultaneously.

Consider a DFCNN with its weight $\bar { w }$ , where $\bar { w }$ represents a solution obtained by a learning algorithm (e.g., SGD) trained on a set of pristine data $\bar { x } = \{ \bar { x } _ { i } \} , i \in$ $I = \{ 1 , 2 , \dots , n \}$ . Assume the data is perturbed following $x _ { i } ^ { p } = \bar { x } _ { i } + \delta \Delta x _ { i }$ , where $\delta$ is the norm of perturbation, $\Delta x _ { i }$ is a unit vector that indicates the perturbation direction for point $x _ { i }$ . We denote by $\Delta x = \{ \bar { \Delta x _ { i } } \} , i \in I$ the set of perturbations. To make the number of perturbed points more flexible, we set $K \subset I$ to denote the indices of the perturbed data, and let $\Delta x _ { i } = 0$ for $i \in I \backslash K$ . As defined by (1), $S ( x ^ { p } )$ is the solution set of a DFCNN trained by the poisoned data $x ^ { p } = { \bar { x } } + \Delta x$ and $S ( \bar { x } )$ is the original solution set.

The graphical derivative $D S ( \bar { x } \mid \bar { w } ) ( \Delta x )$ captures how the solution $w$ changes near $\bar { w }$ when $x$ is perturbed in the direction of $\Delta x$ . For any twice differentiable loss function $L ( w )$ (such as the quadratic loss function), following theorem 1, $D S ( \bar { x } \mid \bar { w } ) ( \bar { \Delta } x )$ is equivalent to (see Appendix A):

$$
\begin{array} { l } { \displaystyle { \mathit { D S } ( \bar { x } \mid \bar { w } ) ( \Delta x ) = \{ v \mid \nabla _ { w } ^ { 2 } \frac { 1 } { | I | } \sum _ { i \in I } L \left( x _ { i } , y _ { i } , w \right) v } } \\ { \displaystyle { + \frac { 1 } { | K | } \sum _ { i \in K } \nabla _ { x _ { i } } \nabla _ { w } L \left( x _ { i } , y _ { i } , w \right) \Delta x _ { i } } } \\  \displaystyle { = 0 \} . } \end{array}
$$

The solution set $S ( x ^ { p } )$ within $U$ , a neighborhood of $\bar { w }$ , can be estimated by:

$$
S ( x ^ { p } ) \cap U \subset \bar { w } + D S ( \bar { x } \mid \bar { w } ) ( \Delta x ) + o ( \| \Delta x \| ) \mathbb { B } .
$$

In a special case, when the empirical risk function $\textstyle \sum _ { i = 1 } ^ { n } { \dot { L ^ { * } } } ( x _ { i } , y _ { i } , w )$ has a non-singular Hessian matrix, $D \bar { S } ( \bar { \bar { x } } \mid \bar { w } ) ( \Delta x )$ will include a unique $v$ , indicating that the solution $\bar { w }$ can only move along direction $v$ if we perturb $\bar { x }$ along direction $\Delta x$ . This situation corresponds to the scenario where the solution $\bar { w }$ represents an isolated minimum. In this case, multiplying both sides of expression on the right side of (14) by the inverse of the Hessian matrix results in the influence function (Koh and Liang 2017).

Our set-based framework also provides a new evaluation method for the training stability of DNN, see Algorithm 1.

<html><body><table><tr><td>Algorithm 1: Training Stability Analysis Based on Set Val- ued Mapping</td></tr><tr><td>Input:Perturbation index setK,data perturbation △x,orig- inal solution ω, unperturbed data x,loss function L</td></tr><tr><td>Output: Distance between the new solution set and ω 1. Compute the graphical derivative DS(x|ω)(△x) fol-</td></tr><tr><td>lowing (14). 2.Estimate the new solution set by S(xP) ≈ ω + DS(x</td></tr><tr><td>w)(△x) 3.Calculate distance between the solution set and ω follow-</td></tr></table></body></html>

# Simulation for Solution Estimation

Although our theoretical results in the above two sections focus on DFCNN, this section demonstrates that the methods perform well for general DNNs. To show this, we next present two numerical examples to illustrate the proposed set-valued sensitivity analysis method. The first one is on a toy example and the second one is on the Resnet. All experiments are performed on an RTX 4090 GPU.

# A Toy Example

We consider a linear neural network with 2 layers. Assume we only have two data points $( x _ { i } , y _ { i } ) , i = 1 , 2$ and both $x _ { i }$ and $y _ { i }$ are real numbers. The solution set $S ( x ) = \{ w =$ $( w _ { 1 } , w _ { 2 } ) \}$ , where $w _ { 1 } , w _ { 2 }$ are the weights of the first and second layer, is given by minimizing the empirical risk :

$$
\begin{array} { r l } & { \displaystyle \left( w _ { 1 } , w _ { 2 } \right) } \\ & { \displaystyle \triangleq \mathop { \operatorname { a r g m i n } } _ { w _ { 1 } , w _ { 2 } \in R } \frac { 1 } { 2 } \left( y _ { 1 } - w _ { 1 } w _ { 2 } x _ { 1 } \right) ^ { 2 } + \frac { 1 } { 2 } \left( y _ { 2 } - w _ { 1 } w _ { 2 } x _ { 2 } \right) ^ { 2 } . } \end{array}
$$

Given the pristine dataset $( \bar { x } _ { 1 } , \bar { y } _ { 1 } ) \ = \ ( 1 , 2 ) , ( \bar { x } _ { 2 } , \bar { y } _ { 2 } ) \ =$ $( 2 , 4 )$ , the model solution constitutes a set as $w _ { 1 } * w _ { 2 } = 2$ , and $\bar { w } = ( 1 , 2 )$ is obviously one of the solutions. We assume that the original solution converges to $\bar { w } = ( 1 , 2 )$ during training using this pristine data. We introduce perturbations to the data following the rule $x ^ { p } = \bar { x } + 0 . 2 * ( - 1 , - 2 )$ , and re-train the model using the poisoned data.

![](images/340671e5bd32ffd8d3dfd326a7afb17ad19e950f6d8787c699aded7afd93f95c.jpg)  
Figure 2: The location of the original solution set $S ( \bar { x } )$ and the new solution set $S ( x ^ { p } )$ . It presents how the $S ( \bar { x } )$ expands to achieve $S ( x ^ { p } )$ , where the red area is the expanded area of $S ( \bar { x } )$ as $S ( \bar { x } ) + \kappa \| x - x ^ { p } \| \mathbb { B }$ .

Following Lemma 6 (see appendix B) and Theorem 1, the Lipschitz modulus $\kappa$ for the toy model is 0.2. Given that the perturbation distance $\| x - x ^ { p } \|$ is known, we can deduce that $\kappa \| x - x ^ { p } \| \mathbb { B }$ is now a scaled unit ball with a radius of approximately 0.178. By definition in (4), the Lipschitz-like property reveals that the solution set after perturbation, i.e. $S ( x ^ { p } )$ , will fall into the expanded set $S ( \bar { x } ) \dot { + } \kappa \| x - x ^ { p } \| \mathbb { B }$ We plot this fact in Figure 2, which shows that the expanded area of $S ( \bar { x } )$ is very close to and follows well with $S ( x ^ { p } )$ .

We estimate the new solution using $S \left( x ^ { p } \right) \approx \bar { w } + D S ( \bar { x } \mid$ $\bar { w } ) ( \Delta x )$ (see (15)) and display five estimated solutions in $S ( x ^ { p } ) \cap \mathbb { B } ( { \bar { x } } , 0 . 2 )$ in Figure 3. As shown in Figure 3 (a), both the pristine and poisoned models have their respective set of solutions. The solution set of the poisoned model is shifted upwards compared to the original solution set. Our estimated solutions (the red points) are very close to the real solution set of the poisoned model. Figure 3 (b) demonstrates the loss landscape of the poisoned model. If we continue to use the original solution $\bar { w }$ , the empirical risk would be 0.4. By shifting the solution from $\bar { w }$ to our estimated solutions, we can decrease the risk to nearly zero (around 0.01).

By estimating the new solution using the classical approach (see Eq. 3) and substituting the pseudo-inverse of the Hessian for the Hessian inverse, we can only derive a single estimated solution with a loss of 0.012 on the perturbed data, marked as the orange point in Figure (3). This solution is close to the estimated solution set derived using the graphical derivative, highlighting that the classical approach can be viewed as a special case of our proposed method.

Using Algorithm 1, the training stability can be measured by the minimum distance between $\bar { w }$ and the estimated solutions in $S \left( x ^ { p } \right) \cap \mathbb { B } ( { \bar { x } } , 0 . 2 )$ (e.g. the red points in Figure 3,)

which is 0.1746, representing the minimum point-to-point distance between the original solution and all estimated solutions. The experimental results are consistent with Figure 2, where the original solution set expands outward by $0 . 2 \left\| { \boldsymbol { x } } - { \boldsymbol { x } } ^ { p } \right\| \mathbb { B }$ .

# Simulation on Resnet

This section applies our estimated method, as defined by (15), to a Resnet56 network (He et al. 2016). We extract 1000 points from the CIFAR-10 dataset, denoted as $\bar { x }$ . We use $x ^ { p }$ to denote the poisoned data. The original solution $\bar { w }$ is the pre-trained weights of Resnet56.

Following (15), when the data $\bar { x }$ is perturbed along the direction $\Delta x$ , the corresponding change direction of $\bar { w }$ , denoted by $\Delta w$ , is determined by the graphical derivative $D S ( \bar { x } \mid \bar { w } ) ( \Delta x )$ . The relationship between $\Delta x$ and $\Delta w$ is:

$$
\begin{array} { l } { \displaystyle \nabla _ { w } ^ { 2 } \frac { 1 } { | I | } \sum _ { i \in I } L \left( x _ { i } , y _ { i } , w \right) \Delta w } \\ { + \frac { 1 } { | K | } \displaystyle \sum _ { i \in K } \nabla _ { x _ { i } } \nabla _ { w } L \left( x _ { i } , y _ { i } , w \right) \Delta x _ { i } = 0 } \end{array}
$$

Denoting $\dagger$ the pseudo inverse operator, $\Delta w$ is given by:

$$
\begin{array} { c } { { \displaystyle \Delta w = - \left( \nabla _ { w } ^ { 2 } \frac { 1 } { | I | } \sum _ { i \in I } L \left( x _ { i } , y _ { i } , w \right) \right) ^ { \dagger } } } \\ { { \displaystyle \left( \frac { 1 } { | K | } \sum _ { i \in K } \nabla _ { x _ { i } } \nabla _ { w } L \left( x _ { i } , y _ { i } , w \right) \Delta x _ { i } \right) . } } \end{array}
$$

Note that equation (18) is also used in practical computations of (3), but it plays a different role in our paper. While other papers used the pseudo-inverse as a substitute for the Hessian inverse, we utilize it here to solve equation (17) and thereby estimate the graphical derivative following Theorem 1. To solve (18) under high-dimensional cases, we transform (18) to a least square problem:

$$
\Delta \boldsymbol { w } : = \underset { \boldsymbol { v } } { \mathrm { a r g m i n } } \frac { 1 } { 2 } \left\| \begin{array} { l } { \displaystyle \nabla _ { \boldsymbol { w } } ^ { 2 } \frac { 1 } { | I | } \sum _ { i \in I } L \left( \boldsymbol { x } _ { i } , \boldsymbol { y } _ { i } , \boldsymbol { w } \right) \boldsymbol { v } } \\ { + \frac { 1 } { | K | } \sum _ { i \in K } \nabla _ { \boldsymbol { x } _ { i } } \nabla _ { \boldsymbol { w } } L \left( \boldsymbol { x } _ { i } , \boldsymbol { y } _ { i } , \boldsymbol { w } \right) \Delta \boldsymbol { x } _ { i } } \end{array} \right\| ^ { 2 }
$$

where both $\nabla _ { x } \nabla _ { w } L \left( x _ { i } , y _ { i } , w \right) \Delta x$ and $\nabla _ { w } ^ { 2 } L \left( x _ { i } , y _ { i } , w \right) \boldsymbol { v }$ can be calculated using implicit Hessian-vector products (HVP) (Pearlmutter 1994). As shown previously (Agarwal, Bullins, and Hazan 2017), $\nabla _ { w } ^ { 2 } L \left( \dot { x _ { i } } , y _ { i } , w \right) \dot { v }$ can be computed efficiently in $O ( p )$ .

We perturb an individual point $x _ { k }$ along the direction of $\nabla _ { x _ { k } } L ( \bar { w } )$ . If we continue to use the $\bar { w }$ as our weights, the training loss for the points perturbed would increase from near zero to around 0.0041, due to the perturbations. This indicates that the original solution is far from the solution set of the poisoned model. We estimate the change of $\bar { w }$ , i.e. $\Delta w$ through solving the problem (19). By moving the original solution along $\Delta w$ , we obtain the estimated solution. By adopting our estimated solutions, the training loss decreases to near zero, demonstrating that our estimated solution change $\Delta w$ effectively draws $\bar { w }$ towards the solution set of the poisoned model. Table 1 reports the loss on the perturbed points for varying numbers of perturbed points when we adopt the original solution and the estimated solution. We then perturb 500 points and run the problem (19) with different random seeds, deriving different numerical results for $\Delta w$ . Figure 4 demonstrates the location of the original solution and estimated solutions on the loss landscape of poisoned Resnet56. The loss landscape visualization follows the work in (Li et al. 2018), where the color intensity and the left plot’s vertical axis indicate the loss. By moving the original solution $\bar { w }$ along $\Delta w$ , the solutions reach the valley of the landscape of the poisoned model. Each estimated solution in Figure 4 corresponds to a simulation result of (19).

Table 1: Empirical loss of the Resnet56 on the perturbed points using original and estimated solutions.   

<html><body><table><tr><td></td><td>perturbation</td><td>one point</td><td>10 points</td></tr><tr><td>Solution</td><td></td><td></td><td></td></tr><tr><td>Original Solution</td><td></td><td>0.0041</td><td>0.0045</td></tr><tr><td>Estimated Solution</td><td></td><td>1.9 × 10-5</td><td>6.2 × 10-5</td></tr></table></body></html>

# Related Work

This paper only focuses on the sensitivity of solutions of learning models (e.g., model weights of DNNs) in response to perturbations in the training data. For the change of prediction in relationship to the inference data, one can refer to Fazlyab et al. (2019); Weng et al. (2018). Influence function, as a concept in robust statistics (Law 1986), was first used for the sensitivity analysis of the classification models with convex loss function (e.g. SVM, Logistic Regression) (Christmann and Steinwart 2004). Koh and Liang (2017) introduced it to the DNN, demonstrating its application in data poisoning attacks and identifying the important features. However, the existence of the influence function relies on the implicit function theorem (see Theorem 19 in Christmann and Steinwart (2004)), which may not be applicable to DNNs when non-isolated DNN solutions are considered, as discussed in the Introduction section of this paper. Nickl et al. (2024) measured the sensitivity of solutions to training data through the Memory-Perturbation Equation (MPE). It was demonstrated that sensitivity to a group of examples can be estimated by adding their natural gradients, indicating higher sensitivity with larger gradients. However, its theorem relies on the inverse of Hessian, which does not exist when the loss function is not strongly convex.

This paper utilizes the Lipschitz modulus to characterize the sensitivity of DNN. Noteworthy is that sensitivity analysis in our paper studies how the model solution changes with training data, not how the model output changes with inference data, although the two are closely related. To our best knowledge, this is the first time that the Lipschitz concept and the estimation of Lipschitz modulus estimation are

![](images/13c6f8a4d163d8614669eff5ed45659d34b82fcbd4493f18a62f9e3684fecd07.jpg)  
Figure 3: (a) Respective solution set of the pristine and poisoned model. (b) The loss landscape of the poisoned toy model. The black,red points, orange point indicate the positions of the original solution $\bar { w }$ , our estimated solutions and unique solution estimated by equation (3), respectively. The contour lines represent the corresponding losses.

![](images/20f78a77dd8635d391aa69a2dead350953a5c75f53c351f9cd02897ece21392a.jpg)  
Figure 4: (a) 3D image and (b) 2D contours of the loss landscape of the poisoned ResNet56, indicating how the $\Delta w$ draws the original solution towards the valley of the landscape.

# Conclusion

introduced to the training stage. Previous research only focused on estimating the Lipschitz constants during the inference stage, see Fazlyab et al. (2019); Virmaux and Scaman (2018), to quantify the robustness of model prediction w.r.t. perturbations in inference data. For example, Virmaux and Scaman (2018) adopted a power method working with auto differentiation to estimate the upper bound of the Lipschitz constant. Most sensitivity analysis approaches only focus on a single solution of the learning algorithm. This paper considers the first-order optimality condition as a setvalued mapping (multifunction), introducing the ‘set-to-set’ analysis approach. Our sensitivity analysis is based on the Lipschitz-like property of set-valued mapping, where the Lipschitz Modulus quantifies the change of the solution set. For more discussion about set-valued mapping, interested readers can refer to Rockafellar and Wets (2009); Dontchev and Rockafellar (2009).

This paper provides set-valued analysis methods to study the sensitivity of model solutions (e.g. weights of a DNN) in response to perturbations in the training data. Theoretically, our approach considers the possibility that the DNN may not have unique solutions and does not rely on a non-singular Hessian. We accurately estimate the solution change when the training data are perturbed along a specific direction. Our analysis framework has multiple applications. It can leverage Lipschitz continuity to assess DNN sensitivity, where a larger Lipschitz constant indicates higher sensitivity. Our framework also extends the implicit function theorem for DNNs, enabling the execution of model target poisoning attacks by determining the perturbation direction to shift the solution toward a target (Suya et al. 2021).

# Acknowledgments

The authors would like to express their gratitude to R. Tyrrell Rockafellar for his invaluable contributions and insights that greatly influenced this work.