# Error Diversity Matters: An Error-Resistant Ensemble Method for Unsupervised Dependency Parsing

Behzad Shayegh1, Hobie H.-B. Lee1, Xiaodan $\mathbf { Z } \mathbf { h } \mathbf { u } ^ { 2 }$ , Jackie Chi Kit Cheung3,4, Lili Mou1,4

1Dept. Computing Science, Alberta Machine Intelligence Institute (Amii), University of Alberta 2Dept. Electrical and Computer Engineering & Ingenuity Labs Research Institute, Queen’s University 3Quebec Artificial Intelligence Institute (Mila), McGill University 4Canada CIFAR AI Chair the.shayegh $@$ gmail.com, hobie.lee $@$ ualberta.ca, xiaodan.zhu@queensu.ca, jcheung $@$ cs.mcgill.ca, doublepower.mou@gmail.com

# Abstract

We address unsupervised dependency parsing by building an ensemble of diverse existing models through post hoc aggregation of their output dependency parse structures. We observe that these ensembles often suffer from low robustness against weak ensemble components due to error accumulation. To tackle this problem, we propose an efficient ensemble-selection approach that considers error diversity and avoids error accumulation. Results demonstrate that our approach outperforms each individual model as well as previous ensemble techniques. Additionally, our experiments show that the proposed ensemble-selection method significantly enhances the performance and robustness of our ensemble, surpassing previously proposed strategies, which have not accounted for error diversity.

# Code — https://github.com/MANGA-UOFA/ED4UDP Extended version — https://arxiv.org/abs/2412.11543

# 1 Introduction

Syntactic parsing, a fundamental task in natural language processing (NLP), refers to identifying grammatical structures in text (Zhang 2020), which can help downstream NLP tasks such as developing explainable models (Amara, Sevastjanova, and El-Assady 2024). Unsupervised syntactic parsing is particularly beneficial for processing languages or domains with limited resources, as it eliminates the need for human-annotated training data (Kann et al. 2019). In addition, autonomously discovering patterns and structures helps to provide evidence to test linguistic theories (Goldsmith 2001) and cognitive models (Bod 2009). Two common types of syntactic parsing include: (1) constituency parsing, organizing words and phrases in a sentence into a hierarchy (Chomsky 1967), and (2) dependency parsing, predicting dependence links between words in a sentence (Tesnie\`re, Osborne, and Kahane 2015). The latter is particularly interesting given its possible applications to other fields, e.g., RNA structure prediction (Wang and Cohen 2025).

Unsupervised dependency parsing has seen diverse approaches over decades (Smith and Eisner 2005; Cai, Jiang, and Tu 2017; Han et al. 2020). Yang et al. (2020) show + Sib&L-NDMV 69.0 PA P I PR GUf GR M 60 AN 1URa G d N P/af 5 P/bf 1 上 .......... 工 OD 68.5 i0 T M s E PRa 50 s S 中 H 中 L S W A R 68.0 (a) (b) C 5 M 67.5 that the combination of two different models can outperform both individuals, suggesting the varied contributions of different models. We follow our previous work on constituency parsing (Shayegh et al. 2024; Shayegh, Wen, and Mou 2024) and build an ensemble of different unsupervised dependency parsers to leverage their diverse expertise.1 We regard the ensemble output as the dependency structure that maximizes its overall similarity to all the individual models’ outputs. For the similarity metric, we employ the unlabeled attachment score (UAS; Nivre and Fang 2017), which is the most widely used evaluation metric of the task.

In our work, we observe that a na¨ıve application of Shayegh et al. (2024)’s ensemble is sensitive to weak individuals in dependency parsing. As shown in Figure 1, a best-to-worst incremental ensemble experiment encounters a significant drop in performance when weaker individuals are added. This drop is due to the accumulation of individual errors, which arises from a low error diversity.

It is important to distinguish between two types of diversity: expertise diversity and error diversity. The former refers to the phenomenon that different models excel on different subsets of samples, while the latter indicates that models are wrong in different ways for each sample. Although both types of diversity are crucial for successful ensembles (Zhang and Ma 2012), it is essential to emphasize that expertise diversity serves as a motivation to build an ensemble, whereas error diversity is a requirement for its success (outperforming ensemble individuals), because otherwise, the individuals’ errors cannot be eliminated through the ensemble process.

To this end, we propose a diversity-aware method for ensemble selection considering error diversity so that our approach is resistant to error accumulation. Having a diverse set of ensemble components has been a focus of researchers for years (Tang, Suganthan, and Yao 2006; Zhou 2012; Wu et al. 2021; Purucker et al. 2023). However, most previous studies focus on expertise diversity by considering only the individuals’ successes/failures, ignoring the differences in their mistakes (Yule and Pearson 1900; Fleiss 1981; Cunningham and Carney 2000). By contrast, we develop a metric, called society entropy, to capture both aspects of diversity. Our approach outperforms the existing expertisediversity metrics in ensemble selection, demonstrating the importance of error diversity.

We conduct our experiments on the Wall Street Journal (WSJ) corpus in the Penn Treebank (Marcus, Santorini, and Marcinkiewicz 1993). Results show the effectiveness of our ensemble approach, outperforming all the individual parsers and ensemble baselines. We further boost the performance by employing our diversity-aware selection of individuals, achieving state-of-the-art performance in the task.

In conclusion, our main contributions include:

1. proposing an ensemble-based approach to unsupervised dependency parsing,   
2. specifying error diversity as an important aspect in building an ensemble, and   
3. utilizing society entropy as a diversity metric to perform selection of ensemble individuals.

# 2 Related Work

# 2.1 Unsupervised Dependency parsing

Over the years, researchers have proposed various models to tackle unsupervised dependency parsing under different setups (Le and Zuidema 2015; He, Neubig, and BergKirkpatrick 2018; Shen et al. 2021). The longest line of research began with Klein and Manning (2004) introducing the dependency model with valence (DMV), a probabilistic generative model of a sentence and its dependency structure, which is extended by Headden III, Johnson, and McClosky (2009) to include categories of valence and lexical information. In the deep learning era, Jiang, Han, and Tu (2016) utilize neural networks to learn probability distributions in DMV. Han, Jiang, and Tu (2017, 2019a,b) further equip neural DMVs with lexical and contextual information. More recently, Yang et al. (2020) introduce two second-order variants of DMV, which incorporate grandparent–child or sibling information, respectively. In this work, we build ensembles of different models to leverage their diverse knowledge.

All of the above work focuses on projective dependency parse structures, meaning that all the dependency arcs can be drawn on one side of the sentence without crossing each other. The projectivity constraint comes from the contextfree nature of human languages (Chomsky 1956; Gaifman 1965). Given that most English sentences are characterized by projective structures (Wu et al. 2020), we also follow previous studies and focus on this type of parses.

Unsupervised dependency parsing typically considers unlabeled structures, i.e., dependency arcs are not categorized by their linguistic types (e.g., subject or object of a verb). We also follow the previous line of research and focus on unlabeled structures.

# 2.2 Ensemble-Based Dependency Parsing

Che et al. (2018) and Lim et al. (2018) propose to build ensembles of dependency parsers to smooth out their noise and reduce the sensitivity to the initialization of the neural networks. They use the average of networks’ softmax outputs for prediction. This approach is restricted as it requires the models to have the same output space, which does not hold in our scenario as we aim to leverage the knowledge of diverse models with different architectures.

Kulkarni et al. (2022) and Shayegh et al. (2024) show the effectiveness of post hoc aggregation methods for ensemblebased supervised and unsupervised constituency parsing. Kulkarni, Eulenstein, and Li (2024) borrow different post hoc aggregation methods from graph studies, including the maximum spanning tree (Gavril 1987), conflict resolution on heterogeneous data (Li et al. 2014), and a customized Ising model (CIM; Ravikumar, Wainwright, and Lafferty 2010), and employ them for ensemble-based dependency parsing. They compare the performance of these aggregation methods and show the superiority of CIM. However, this method does not ensure the validity of the ensemble output as a projective dependency parse structure. Nevertheless, we include CIM as a baseline which underperforms when applied to our unsupervised setting, performing worse than the best individual.

In this work, we develop a dependency-structure aggregation method, based on which we further build an ensemble of unsupervised dependency parsers with different designs.

# 2.3 Ensemble Selection

Ensemble selection refers to selecting a set of models to form an ensemble, which has been extensively discussed in the machine learning literature (Kuncheva 2004; Caruana, Munson, and Niculescu-Mizil 2006). Caruana et al. (2004) introduce a forward-stepwise-selection method, inspired by the field of feature selection (Liu and Motoda 2007); in their approach, individuals are incrementally added to the ensemble based on the performance boost on a validation set. Such an approach is prone to overfitting to the validation set (Kohavi and John 1997); in addition, it is a time-consuming process to build the ensemble, as we need to build many ensembles at each increment.

Another line of research predicts the success of an ensemble by looking at the individuals’ properties (Ganaie et al. 2022; Mienye and Sun 2022). One commonly adopted criterion is to keep diversity among the ensemble individuals (Minku, White, and Yao 2010; Wood et al. 2023). This brings multiple benefits, such as bringing different expertise (expertise diversity; Zhang and Ma 2012) and smoothing out individual errors (error diversity; Zhou 2012). Two branches of related studies are: (1) how to make a balance between the diversity and the quality of the individuals (Chandra, Chen, and Yao 2006; Wood et al. 2023), and (2) how to measure the diversity (Kuncheva and Whitaker 2003). In our work, we address the latter.

Most ensemble-selection studies focus on binary classification tasks. Yule and Pearson (1900) propose a $Q$ statistic to measure diversity based on the association between random variables. Fleiss (1981) introduces the measure of inter-rater reliability, quantifying the non-coincidental agreements among classifiers. Kohavi and Wolpert (1996) measure the diversity based on the variability of predictions across all classifiers. Skalak (1996) and Ho (1998) measure the disagreement between a pair of classifiers by the ratio of observations on which only one classifier is correct. To address multi-class classification, Kuncheva and Whitaker (2003) and Kadkhodaei and Moghadam (2016) reduce the problem to binary classification by considering the success/- failure of classifiers, which is a binary variable. Although this approach measures expertise diversity, it loses information about error diversity, which we show is important and should not be ignored.

# 3 Approach

In this work, we propose an ensemble approach to unsupervised dependency parsing. In general, ensemble methods in machine learning consist of two stages: obtaining individual models and aggregating the individuals’ predictions. We will address both in the rest of this section.

# 3.1 Aggregating Dependency Parses

We aim to aggregate parses of different individual models, potentially with different architectures and output formats. We propose to have post hoc aggregation methods applied to the dependency parses obtained by previous parsers. Inspired by our previous work (Shayegh et al. 2024), we formulate our ensemble under the minimum Bayes risk (MBR) framework. Consider a set of individuals’ outputs $A _ { 1 } , \cdots , A _ { K }$ given a sentence. The ensemble output $A ^ { * }$ is defined as the dependency parse structure maximizing its similarity to all the individuals’ outputs:

$$
A ^ { * } = \underset { A \in \mathcal { A } } { \mathrm { a r g m a x } } \sum _ { k = 1 } ^ { K } \mathrm { s i m i l a r i t y } ( A , A _ { k } )
$$

where $\mathcal { A }$ is the set of all the possible dependency parse structures and similarity is a customizable similarity measure for dependency parses.

In particular, we propose to use the sentence-level unlabeled attachment score (UAS; Nivre and Fang 2017) as the similarity measure because it is the main evaluation metric for unlabeled, projective dependency parsing.2 Formally,

UAS is the accuracy of head detection for every word (Le and Zuidema 2015; Nivre and Fang 2017; Yang et al. 2020). In other words, $\begin{array} { r l } { \operatorname { U A S } ( A _ { p } , A _ { r } ) ~ = ~ \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \mathbb { 1 } [ A _ { p } ^ { ( j ) } = A _ { r } ^ { ( j ) } ] } \end{array}$ , where $A _ { p }$ and $A _ { r }$ are two parallel lists of attachment heads for $n$ given words in a sentence or a corpus. Putting UAS in Eqn. (1), we have

$$
\begin{array} { l } { { \displaystyle { \cal A } ^ { * } = \mathop { \mathrm { a r g m a x } } _ { A \in \mathcal { A } } \sum _ { k = 1 } ^ { K } \mathrm { U A S } ( A , A _ { k } ) } } \\ { { \displaystyle ~ = \mathop { \mathrm { a r g m a x } } _ { A \in \mathcal { A } } \sum _ { j = 1 } ^ { n } \sum _ { \underline { { k } } = 1 } ^ { K } \mathbb { 1 } \left[ A ^ { ( j ) } = A _ { k } ^ { ( j ) } \right] } } \\ { { \displaystyle ~ } } \end{array}
$$

Here, $m ( a ; j )$ is the number of individuals assigning word $a$ as the $j$ th word’s head. Specially, $a = 0$ indicates the ROOT. We solve this problem by reducing it to the dependency parsing model of Eisner (1996) and using their efficient dynamic programming algorithm.

$$
\begin{array} { c l } { { \displaystyle { A ^ { * } = \mathrm { \ a r g m a x } \sum _ { { \vec { A } } \in \mathcal { A } } ^ { m } ( A ^ { ( i ) } ; j ) } } } \\   \displaystyle  \begin{array} { c }   \begin{array} { c }   \begin{array} { r l } { { \ { A ^ { * } } } \\ { { \vec { A } } \end{array} } } \\ { { \begin{array} { c } { { \begin{array} { r l } { { \begin{array} { r l } { { \vec { A } } \end{array} } } \\ { { \begin{array} { c } { { \begin{array} { r l } { { \vec { A } } \end{array} } } \\ { { \begin{array} { c l } { { \begin{array} { r l } { { \vec { A } } \end{array} } } \\ { { \begin{array} { c l } { { \begin{array} { r l } { { \begin{array} { r l } { { \vec { A } } \end{array} } } \end{array} } } \\ { { \begin{array} { c l } { { \begin{array} { r l } { { \begin{array} { r l } { { \begin{array} { r l } { { \vec { A } } } \end{array} } } \end{array} } } \\ { { \begin{ r l } { { \begin{array} {array} { r l } { { \begin{array} { \begin{array} { r l } { { \begin{array} { r l } { \begin{array} { \begin{array} { r l } { \end{array} } {array} { \begin{ c l } } { { \begin{array} { r l } { \begin{array} { \begin{array} { c l } { \begin{array} { r l } { \begin{ \frac } { { \begin{array} \begin{array} { r l } { \begin{array} { r l } { \frac { \begin{ n } { \ n } } { \ m } { [ \frac { n } { [ \frac { \lambda } { \lambda } { \sqrt { \lambda } } { \lambda } } \end{ \lambda } } ] } } } \\ { { [ ( { \lambda ( \lambda + \frac { \lambda } { \lambda } { \lambda } ) } ] } } \end{ j } } \end{array} } } \end{ \biggr } ] } } } } } } } \end{ } } } } } } \end{ } } } } } } } } } \end{ } } } } } } } } } } \\ { { { \begin{ { { \begin{array} { r l { \mathrm { ~ { \begin{array} { r l { \begin{array} { \begin{array} { r l } { { \begin{ \ r l } { { \vec { A } } \end{array} } \end{array} } [ { c l } \end{array} } \\ { - [ \lambda } { [ \lambda } \end{array} ] } { [ \frac { c } { [ d } { c } \end{array} ] } ] } } } \\ { [ d } { ( { \frac { \lambda } { [ d } ] } ) ] } } \end{array} } } } } \end{array} } } } } \end{array} } } } } \\   \begin{array} { c }   \begin{array} { r l }   \begin{array}  r l  \mathrm { \ a r g m a x } { [ 1 } \\ { { \vec { A } } \end{array} } [ \frac { \lambda } { [ \lambda } \end{array} \ \end{array} \end{array} \end{array} \end{array} \end{array} \end{array} \end{array} \end{array} \end{array} \end{array} \end{array} \end{array} \end{array} \end{array} \end{array} \end{array} \end{array} \end{array}
$$

In Eqn. (7), we normalize values of $\hat { m }$ to be in $[ 0 , 1 ]$ , which can be viewed as estimated probabilities. Consequently, the argmax objective is the joint probability of all attachments in a sentence, i.e., the probability of the dependency parse $A$ . Hereby, we have reduced our problem to the dependency parsing problem in Eisner (1996) and can use their efficient algorithm to obtain $A ^ { * }$ in $\mathcal { O } ( n ^ { 3 } )$ time complexity.

# 3.2 Ensemble Selection

In this work, we build an ensemble of various unsupervised dependency parsers. We further propose to perform ensemble selection, as we notice in Figure 1 that an ensemble including all individuals underperforms a moderately sized ensemble.

We hypothesize that, if ensemble individuals lack diversity in their errors, they introduce systematic bias and may hurt the ensemble performance. Therefore, it is important to balance the quality of individuals and their diversity. Our ensemble selection objective is to find ${ \mathcal { K } } \subseteq \{ 1 , \cdots , K \}$ being the set of selected ensemble individuals maximizing

$$
\begin{array} { r } { \mathrm { o b j e c t i v e } ( K ) = \displaystyle \sum _ { \kappa \in { \cal K } } \mathrm { U A S } ( A _ { \kappa } , A _ { \mathrm { g o l d } } ) + \qquad } \\ { \qquad \quad \alpha \cdot \mathrm { d i v e r s i t y } ( \{ A _ { \kappa } \} _ { \kappa \in { \cal K } } ) } \end{array}
$$

where $A _ { \mathrm { g o l d } }$ is the ground-truth parse3 and $\alpha$ is a balancing hyperparameter.

For diversity, we introduce society entropy. Essentially, we first define society distribution (SD) for the $j$ th word as

$$
\mathrm { S D } \left( a ; \{ A _ { \kappa } ^ { ( j ) } \} _ { \kappa \in \mathcal { K } } \right) = \frac { \sum _ { \kappa \in \mathcal { K } } \mathbb { 1 } \left[ A _ { \kappa } ^ { ( j ) } = a \right] } { | \mathcal { K } | }
$$

where the probability of the head being $a$ is the fraction of the individuals agreeing on that. Then, we define the society entropy (SE) as the entropy $\begin{array} { r } { E ( p ) = - \sum _ { x } p ( x ) \log p ( x ) } \end{array}$ of society distribution:

$$
\mathrm { S E } ( \{ A _ { \kappa } \} _ { \kappa \in \mathcal K } ) = \frac 1 n \sum _ { j = 1 } ^ { n } E \Big ( \mathrm { S D } \left( \cdot ; \{ A _ { \kappa } ^ { ( j ) } \} _ { \kappa \in \mathcal K } \right) \Big )
$$

where $n$ is the number of words. This metric not only considers expertise diversity, but also measures the error diversity across individuals.

We maximize our objective in a forward-stepwise manner (Caruana et al. 2004). In other words, we begin by selecting

$$
\kappa _ { 1 } = \operatorname * { a r g m a x } _ { \kappa ^ { \prime } \in \{ 1 , \cdots , K \} } { \mathrm { U A S } ( A _ { \kappa ^ { \prime } } , A _ { \mathrm { g o l d } } ) }
$$

Then, we add individuals to our collection one at a time by maximizing our objective, i.e., the $t$ th selected individual is chosen by

$$
\kappa _ { t } = \mathop { \mathrm { a r g m a x } } _ { \kappa ^ { \prime } \in \{ 1 , \cdots , K \} \setminus \mathcal { K } _ { t - 1 } } \mathrm { o b j e c t i v e } ( \mathcal { K } _ { t - 1 } \cup \{ \kappa ^ { \prime } \} )
$$

where ${ \cal { K } } _ { t } = \{ \kappa _ { 1 } , \cdots , \kappa _ { t } \}$ is the first $t$ selected individuals. Overall, the selected set of individuals is $\kappa _ { T }$ , where $T$ is a hyperparameter indicating the number of ensemble individuals.

# 4 Experiments

# 4.1 Settings

Dataset. We performed experiments on the Wall Street Journal (WSJ) corpus in the Penn Treebank (Marcus, Santorini, and Marcinkiewicz 1993). We adopted the standard split: Sections 2–21 for training, Section 22 for validation, and Section 23 for testing. For training and validation, we followed previous work and only used sentences with at most 10 words after being stripped of punctuation and terminals. We used the entire test set for evaluation, regardless of sentence lengths. As all our individuals are unsupervised parsers, we did not use linguistic annotations in the training set. We use an annotated validation set3 with only 250 sentences for validation during training and ensemble selection.

Ensemble Individuals. We consider the following unsupervised dependency parsers as our ensemble components4:

• CRFAE (Cai, Jiang, and $\mathrm { T u } 2 0 1 7 \mathrm { \ : . }$ ), which is a discriminative and globally normalized conditional random field autoencoder model that predicts the conditional distribution of the structure given a sentence.   
• NDMV (Jiang, Han, and Tu 2016), a dependency model with valence (DMV; Klein and Manning 2004) model that learns probability distributions using neural networks. We used Viterbi expectation maximization to train the model.   
• NE-DMV (Jiang, Han, and Tu 2016), which applies the neural approach to the extended DMV model (Headden III, Johnson, and McClosky 2009; Gillenwater et al. 2010). We employed Tu and Honavar (2012) for initialization.   
• L-NDMV (Han, Jiang, and Tu 2017), an extension of NDMV that utilizes lexical information of tokens.   
• Sib-NDMV (Yang et al. 2020), which applies the neural approach to Sib-DMV, an extension of DMV that utilizes information of sibling tokens.   
• Sib&L-NDMV (Yang et al. 2020), a joint model incorporating Sib-NDMV and L-NDMV.

For L-NDMV, Sib-NDMV, and Sib&L-NDMV, we use Naseem et al. (2010) for initialization. For the hyperparameters and configurations of individuals, we use the default values specified in the respective papers or codebases. We train five instances of each model using different random seeds to assess the stability of our approach. We observe that CRFAE, NDMV, and NE-DMV exhibit instability and lower performance on average than what the authors reported. To achieve comparable performance, we run the training for these models 20 times and select the top $2 5 \%$ based on their performance on the validation set. We report a comparison between the quoted results and our replication results in Appendix B, showing the success of our replication.

# 4.2 Aggregation Results

Table 1 shows the main results on the WSJ dataset, where we performed five runs of each individual and built ensembles of corresponding runs. We may or may not exclude Sib&LNDMV as it is already a fusion model, and one may argue about including it in an ensemble where Sib-NDMV and LNDMV are also present.

We compare our aggregation approach against the customized Ising model (CIM; Row 6; Ravikumar, Wainwright, and Lafferty 2010), which calculates per-sample vote weights based on individuals’ correlation over samples in classification tasks. Indeed, Kulkarni, Eulenstein, and Li (2024), as the only previous work on post hoc ensemblebased dependency parsing, shows that CIM outperforms alternatives, including the maximum spanning tree (Gavril 1987) and conflict resolution on heterogeneous data (Li et al.

<html><body><table><tr><td>Method</td><td>UAS</td></tr><tr><td>Ensembleindividuals 1 CRFAE 2 NDMV 3 NE-DMV 4 L-NDMV 5 Sib-NDMV</td><td>53.0±4.5 48.1±3.6 51.0±3.4 62.4±0.6 64.3±0.2</td></tr><tr><td>Ensembles 6 CIM aggregation 78 Our weightehted gregaiation</td><td>58.8±0.7 65.</td></tr><tr><td>Selected ensembles (our aggregation) 9 w/o diversity 10 w/Kuncheva's diversity 11 w/ society entropy (ours) 12 Ensemble validation</td><td>63.8 65.8 67.3 67.8</td></tr><tr><td>Additionalindividual 13 Sib&L-NDMV</td><td>67.9±0.1</td></tr><tr><td>Ensembles (our aggregation) 14 Unweighted 15 Weighted 16 Selected w/ society entropy (ours) 17 Selected by ensemble validation</td><td>68.8 68.4 67.9±0.3 68.40.4</td></tr></table></body></html>

Table 1: Mean and standard deviation across five runs, evaluated on the WSJ test set. ∗, ∗∗, and $* * *$ denote statistically significant improvements over the best individual, as determined by a paired-sample T-test at confidence levels of $92 \%$ , $9 8 \%$ , and $9 9 . 9 9 \%$ , respectively. Selected ensembles do not include variance statistics, and thus, statistical tests can not be conducted for them.

2014).5 The CIM is considered an unweighted approach, as it does not use labeled validation data. We see that this approach does not retain the performance of the best individual (Row 5) and is thus unsuccessful.

By contrast, our approach (Row 7) outperforms all the individuals and baselines, demonstrating the effectiveness of our MBR formulation. We also observe high stability in the performance of our ensemble despite the unstable individuals (Rows 1-3). However, we observe that building an ensemble including Sib&L-NDMV (Row 14) does not bring a performance boost over the best individual (Row 13) due to the huge performance gap between individuals, showing the limited robustness of the ensemble against weak individuals.

We further improve the performance of our ensemble by weighting individuals based on their performance on the validation set (Rows 8 and 15). We can build weighted ensembles with rational numbers as weights by virtually duplicating individuals (Shayegh, Wen, and Mou 2024). The weighted variant overcomes the challenge of high variance in individuals’ performance, exhibiting higher performance than the best individual and the unweighted ensemble. These results highlight the importance of controlling the negative effect of weak individuals. In the next section, we explore ensemble selection as a superior alternative technique, further enhancing robustness against weak individuals.

# 4.3 Ensemble Selection Results

We compare our diversity-based selection method with the classic forward-stepwise-selection approach (Caruana et al. 2004), referred to as ensemble validation. This baseline validates each possible ensemble on a validation set, and thus is slow and computationally expensive. Moreover, we compare our proposed diversity metric “society entropy” versus several previously proposed measures, listed bellow:

• Disagreement (Skalak 1996; Ho 1998), which is a pairwise metric, indicating the fraction of times only one classifier is true. We extend this metric to more than two classifiers by averaging disagreements through all pairs:

$$
\sum _ { \kappa _ { 1 } , \kappa _ { 2 } \in { \mathcal K } } \sum _ { j = 1 } ^ { n } \frac { \mathbb { 1 } \left[ A _ { \kappa _ { 1 } } ^ { ( j ) } = A _ { \mathrm { g o l d } } ^ { ( j ) } \right] \oplus \mathbb { 1 } \left[ A _ { \kappa _ { 2 } } ^ { ( j ) } = A _ { \mathrm { g o l d } } ^ { ( j ) } \right] } { n | { \mathcal K } | ^ { 2 } }
$$

where $n$ is the number of samples based on which we compute the diversity, $A _ { \mathrm { g o l d } }$ is the ground truth, $\kappa$ is the selected individuals, and $\oplus$ is the “exclusive or” operator.

• KW variance (Kohavi and Wolpert 1996), which differs from the averaged disagreement metric above by the constant coefficient $\textstyle { \frac { K - 1 } { 2 K } }$ (Kuncheva and Whitaker 2003):

$$
\sum _ { j = 1 } ^ { n } \frac { \bigg ( \sum _ { \kappa \in \mathcal { K } } \mathbb { 1 } \left[ A _ { \kappa } ^ { ( j ) } = A _ { \mathrm { g o l d } } ^ { ( j ) } \right] \bigg ) \bigg ( \sum _ { \kappa \in \mathcal { K } } \mathbb { 1 } \left[ A _ { \kappa } ^ { ( j ) } \neq A _ { \mathrm { g o l d } } ^ { ( j ) } \right] \bigg ) } { n | \mathcal { K } | ^ { 2 } }
$$

• Fleiss’ Kappa (Fleiss 1981), which differs from KW variance by a coefficient:

$$
\sum _ { j = 1 } ^ { n } \frac { \bigg ( \sum _ { \kappa \in \mathcal { K } } \mathbb { 1 } \left[ A _ { \kappa } ^ { ( j ) } = A _ { \mathrm { g o l d } } ^ { ( j ) } \right] \bigg ) \bigg ( \sum _ { \kappa \in \mathcal { K } } \mathbb { 1 } \left[ A _ { \kappa } ^ { ( j ) } \neq A _ { \mathrm { g o l d } } ^ { ( j ) } \right] \bigg ) } { n | \mathcal { K } | ( 1 - | \mathcal { K } | ) \bar { p } ( 1 - \bar { p } ) }
$$

where $\begin{array} { r } { \bar { p } = \frac { 1 } { n | \mathcal { K } | } \sum _ { j = 1 } ^ { n } \sum _ { \kappa \in \mathcal { K } } \mathbb { 1 } \left[ A _ { \kappa } ^ { ( j ) } = A _ { \mathrm { g o l d } } ^ { ( j ) } \right] } \end{array}$ is the fraction of true hits by all the individuals.

• Kuncheva’s diversity6 (Kuncheva and Whitaker 2003), which is a non-pairwise metric measuring the smoothness of the oracle distribution:

$$
\begin{array} { r } { \displaystyle \frac { 1 } { n ( | { \cal K } | - \lceil | { \cal K } | / 2 \rceil ) } \sum _ { j = 1 } ^ { n } \operatorname* { m i n } \Bigg \{ \sum _ { \kappa \in { \cal K } } \mathbb { 1 } \big [ A _ { \kappa } ^ { ( j ) } = A _ { \mathrm { g o l d } } ^ { ( j ) } \big ] , } \\ { \displaystyle \sum _ { \kappa \in { \cal K } } \mathbb { 1 } \big [ A _ { \kappa } ^ { ( j ) } \ne A _ { \mathrm { g o l d } } ^ { ( j ) } \big ] \Bigg \} \quad ( \kappa \to \infty ) , } \end{array}
$$

<html><body><table><tr><td rowspan="2">Selection method</td><td rowspan="2">Selection time (ms)</td><td colspan="4">Ensemble UAS</td></tr><tr><td>k=3</td><td>k=4</td><td>k=5</td><td>Overall</td></tr><tr><td>w/o diversity</td><td>0.03±0</td><td>64.0±0.9</td><td>65.6±1.3</td><td>65.9±0.7</td><td>65.2±1.3</td></tr><tr><td>w/Kuncheva's diversity</td><td>12±004</td><td>65.0±0.4</td><td>67.7±0.5</td><td>66.1±0.6</td><td>66.3±1.2</td></tr><tr><td>w/ society entropy (ours)</td><td>22±006</td><td>67.1±0.3</td><td>66.9±0.3</td><td>67.2±0.3</td><td>67.1±0.4</td></tr><tr><td>Ensemble validation</td><td>11, 066±166</td><td>66.6±0.8</td><td>66.6±0.4</td><td>67.7±0.5</td><td>67.0±0.8</td></tr></table></body></html>

![](images/cc0c13083e2e9d7abb1fdb455b529da2b88aa855d12123f23312ca5468ad7fc8.jpg)  
Table 2: UASs of selected ensembles with $k$ ensemble components, selected from 15 individuals using different strategies. Numbers are the means and standard deviations across 30 runs. The Overall column represents all the 90 results for $k = 3 , 4 , 5$ . Time is measured using 28 Intel(R) Core(TM) i9-9940X $( \varpi 3 . 3 0 \mathrm { G H z } )$ ) CPUs.   
Figure 2: Ensemble performance by the number of selected ensemble components using different selection strategies. Ensemble selection happens over 25 individuals (five runs for each of CRFAE, NDMV, NE-DMV, L-NDMV, and SibNDMV). Results are split into two figures for easier reading.

• PCDM (Banfield et al. 2005), which is based on the proportion of classifiers getting each sample correct:

$$
\frac { 1 } { n } \sum _ { j = 1 } ^ { n } \mathbb { 1 } \Bigg [ 0 . 1 \leq \frac { \sum _ { \kappa \in \mathcal { K } } \mathbb { 1 } \left[ A _ { \kappa } ^ { ( j ) } = A _ { \mathrm { g o l d } } ^ { ( j ) } \right] } { | \mathcal { K } | } \leq 0 . 9 \Bigg ]
$$

To obtain the above metrics, we consider our task as a classification of each word’s head, which is consistent with the computation of our society entropy and is a simplification of the task yet incorporating the diversity notion.

For each diversity metric, we finetune the balancing hyperparameter $\alpha$ in Eqn. (8) with one decimal-place precision based on the performance of the corresponding selected fiveindividual ensembles.

Figure 2 illustrates the performance of ensembles by different numbers of selected individuals. The selection methods include using only individuals’ performance (w/o diversity), the ensemble objective given in Eqn. (8) employing different diversity metrics, and the classic forward-stepwiseselection approach (ensemble validation).

Our results show that ignoring diversity performs worse than most other methods, underscoring the importance of diversity in ensemble selection. On the other hand, society entropy outperforms the other approaches when a larger number of ensemble components are selected. This is expected, given the higher likelihood of error accumulation with a greater number of weak individuals and that society entropy is the only metric that accounts for error diversity. Moreover, this metric demonstrates significantly more stability than others when we have a small number of individuals. Overall, society entropy performs consistently well, making it the most reliable approach for selecting any number of ensemble components.

Additionally, we present a stability and efficiency analysis in Table 2 for selecting a small number of individuals, as this not only represents a more realistic use case for ensemble selection but also addresses the instability observed in the performance of Kuncheva’s diversity and ensemble validation in Figure 2. To this end, we randomly selected 15 individuals from a group of 25. This random sampling was repeated 30 times to ensure robustness. For each of these 30 sample sets, we applied different selection strategies to choose subsets of 3, 4, and 5 individuals from the 15. Results show that society entropy excels and is more stable than the diversityfree and Kuncheva’s diversity approaches, achieving performance comparable to ensemble validation but around $5 0 0 \mathrm { x }$ faster, making it a practical method.

Finally, we report in Table 1 the performance of selected ensembles. Here, we selected comparable numbers of individuals (five models for Rows 9-12 and six models for Rows 16-17) from all individuals across all runs. Our results are consistent with our analysis, demonstrating that our society entropy is comparable with ensemble validation, outperforming other ensemble-selection methods. Moreover, our society entropy-based selected ensemble outperforms all unselected ensembles and individuals, exhibiting the highest unsupervised dependency parsing performance among all competitors.

# 4.4 Additional Analysis

Inference Efficiency. In Table 3, we report the inference runtime for every individual, along with the execution time

80   
70   
60   
50   
40   
30   
120 Sib-NDMV L-NDMV Sib-NDMV G KU Ensemble Sib-NDMV L-NDMV RA GENt Ensemble Sib-NDMV L-NDMV Sib-NDMV L-NDMV Ensemble Sib-NDMV L-NDMV KN Ensemble CA KU Ensemble 0 NN NNP IN DT JJ NNS RB

Table 3: Inference time in milliseconds, measured by a single Intel(R) Core(TM) i9-9940X $( @ 3 . 3 0 \mathrm { G H z } )$ CPU, without GPU, averaged through samples in the WSJ test. †Implemented in Java. Others are in Python.   

<html><body><table><tr><td>Method</td><td>Time(ms)</td></tr><tr><td>Individualinference</td><td></td></tr><tr><td>CRFAEt</td><td>14.9</td></tr><tr><td>NDMV NE-DMV</td><td>22.3</td></tr><tr><td>L-NDMV</td><td>22.3 56.4</td></tr><tr><td>Sib-NDMV</td><td>114.2</td></tr><tr><td>Sib&L-NDMV</td><td>174.5</td></tr><tr><td>Our aggregation</td><td>19.6</td></tr></table></body></html>

of our aggregation algorithm. While the inference of an ensemble is generally slow as it requires the inference of all the individuals, results show that our proposed aggregation step does not further slow down this process significantly, demonstrating the efficiency of our proposed approach that makes use of Eisner (1996)’s algorithm.

Performance by Dependents’ Part-of-Speech (POS). We would like to investigate the effect of our ensemble approach from a linguistic perspective. To this end, we report in Figure 3 the breakdown performance of our weighted ensemble and its individuals by the POS tags of the dependents. We note that different individuals have their expertise in different types of dependents. In fact, NE-DMV surpasses the others in NNP and VBD cases, L-NDMV is outstanding in IN, RB, and VB, while Sib-NDMV outperforms other individuals in all other types. In most cases, our ensemble roughly matches the performance of the best individual, outperforming all of them in terms of overall performance.

Appendices. We include more results in the appendix: (A)ggregation based on $\mathrm { F } _ { 1 }$ scores, (B)aseline replication, and (C)ase studies.

# 5 Conclusion

In this work, we propose a post hoc ensemble approach to unsupervised dependency parsing, equipped with a diversity-aware ensemble-selection method. We emphasize the importance of error diversity alongside expertise diversity and introduce society entropy as a measure that accounts for both. Our experiments demonstrate the effectiveness of our ensemble approach and the critical role of error diversity in ensemble selection.

Future Work. We identify future directions from algorithmic, linguistic, and machine learning perspectives. On the algorithmic side, it is intriguing to develop aggregation methods for other structures, including non-projective dependency parses or other graph structures in tasks such as drug discovery. For the linguistic aspect, it is promising to investigate cross-task ensembles by aggregating diverse knowledge of different constituency and dependency parsers. From the machine learning perspective, our ensemble selection is limited to classification tasks or those that can be framed as classification. We aim to explore ensembleselection approaches, incorporating error diversity, for a broader range of problems.

In parallel with our work, Charakorn, Manoonpong, and Dilokthanakul (2024) identify a similar differentiation between expertise diversity (called specialization) and general diversity (which includes error diversity) in multi-agent reinforcement learning. It further supports our claim that both aspects of diversity should be taken into account, and suggests a great promise for our approach in multi-agent reinforcement learning.