# NOMATTERXAI: Generating “No Matter What” Alterfactual Examples for Explaining Black-Box Text Classification Models

Tuc Van Nguyen1, James Michels2, Hua Shen3, Thai Le1

1 Department of Computer Science, Indiana University 2 Department of Computer Science, University of Mississippi 3 Information School, University of Washington tucnguye@iu.edu, jrmichel@go.olemiss.edu, huashen $@$ uw.edu, tle@iu.edu

# Abstract

In Explainable AI (XAI), counterfactual explanations (CEs) are a well-studied method to communicate feature relevance through contrastive reasoning of “what $i f ^ { \prime \prime }$ to explain AI models’ predictions. However, they only focus on important (i.e., relevant) features and largely disregard less important (i.e., irrelevant) ones. Such irrelevant features can be crucial in many applications, especially when users need to ensure that an AI model’s decisions are not affected or biased against specific attributes such as gender, race, religion, or political affiliation. To address this gap, the concept of alterfactual explanations (AEs) has been proposed. AEs explore an alternative reality of “no matter what”, where irrelevant features are substituted with alternative features (e.g., “republicans” $$ “democrats”) within the same attribute (e.g., “politics”) while maintaining a similar prediction output. This serves to validate whether the specified attributes influence AI model predictions. Despite the promise of AEs, there is a lack of computational approaches to systematically generate them, particularly in the text domain, where creating AEs for AI text classifiers presents unique challenges. This paper addresses this challenge by formulating AE generation as an optimization problem and introducing NOMATTERXAI, a novel algorithm that generates AEs for text classification tasks. Our approach achieves high fidelity of up to $9 5 \%$ while preserving context similarity of over $90 \%$ across multiple models and datasets. A human study further validates the effectiveness of AEs in explaining AI text classifiers to end users.

Non-Toxic Behavior Toxic Behavior "It doesn't matter [...] innocent Race Black people you "It doesn't matter Features label, she [...] [...] sinful Asian Vancouver" people you label,   
"It doesn't matter she [...] how many Vancouver"   
innocent Asian "It doesn't matter   
people you label, [...] innocent   
she still will never Asian people you   
be abhleo tmoeaiffnord a label, she [...] Vancouver" New York" Decision Boundary Gender Original Features "It doesn't matter [...] innocent Counterfactual Asian people you Semifactual label, he [...] Alterfactual Vancouver"

Code — https://github.com/nguyentuc/NomatterXAI Extend version — https://www.arxiv.org/abs/2408.10528

# Introduction

As AI advances, complex machine learning (ML) text classifiers have been developed to yield predictive performance competitively to that of humans for myriad tasks (Pouyanfar et al. 2018). However, many of such models are so-called “black-box” models that are notorious for their lack of transparency. This may limit both the comprehension and societal acceptance of ML in critical fields, such as healthcare (Tjoa and Guan 2021), finance (Benhamou et al. 2021). The field of Explainable Artificial Intelligence (XAI) (Adadi and Berrada

2018) aims to remedy this by explaining the factors at play in a model’s predictions.

A common paradigm found in XAI is the counterfactual explanation (CE) (Miller 2019) where an alternative reality is presented in which minor alterations to the input directly change the output of an AI model applied to the classification problems of image, tabular, and text data (Verma, Dickerson, and Hines 2020; Garg et al. 2019; Yang et al. 2020). CE follows the thought process of counterfactual thinking by asking “What if...?”, which is a common occurrence in the human psyche, through emotions such as regret, suspense, and relief (Roese and Morrison 2009). CE is often delivered via natural language in the form of “What if” messages (Le, Wang, and Lee 2020; Hendricks et al. 2018). For example, a classifier that labels email messages as spam or ham could provide the text “Had the word ‘credit’ and ‘money’ is used twice in the message, it would have been classified as spam rather than ham.” (Le, Wang, and Lee 2020).

Table 1: Examples of different types of explanations in a hypothetical scenario where an algorithm determines whether or not a person is approved for a loan based on their income.   

<html><body><table><tr><td>Type</td><td>Example</td></tr><tr><td>Factual</td><td>Since your income is $1ooK,you get the loan</td></tr><tr><td>Semifactuals</td><td>Even if your income is $8OK,you get the loan</td></tr><tr><td>Counterfactuals</td><td>If your income was $1K lower,you would had not got the loan</td></tr><tr><td>Alterfactuals</td><td>No matter what your race is, you would get the loan with your current income</td></tr></table></body></html>

While CE is highly effective at providing intuitive reasoning to the users by emphasizing important features, it often neglects the role of less important ones in a text input, occluding information on what is indeed irrelevant to a model’s decision. However, in many cases, irrelevant features are as important as relevant ones in explaining black-box predictions. For example, irrelevant features can help (1) contribute to the comprehensive understanding of a black-box model (Mertes et al. 2022) and (2) determine whether a model is biased against specific semantic features such as gender or race, which we cannot fully understand with only CE.

A recent study posed a solution in the form of alterfactual explanation (AE) (Mertes et al. 2022). AEs embody the thought process of “No matter what...” and present an alternative reality where a set of irrelevant features are significantly changed, and yet the model’s output remains the same. While (Mertes et al. 2022) demonstrate that users view AEs equally favorably as counterfactual explanations, this was done with a hypothetical model for tabular data presented in the user study. The algorithmic generation of AEs for actual trained models is still needed. This can be achieved for tabular data by changing individual features significantly up to their domain ranges–e.g., alternating “age” of a patient from 0 to 100. However, in the NLP domain, textual features cannot be as directly altered due to their discrete nature, not to mention how to change a textual feature significantly but still maintain the reasonable semantic context of the original input–e.g., changing “Republicans” $$ “Democrats” as shown in Fig. 1 is non-trivial. Thus, not only has the generation of AEs for text classifiers not been explored, but such a task also has its unique challenges.

As a first step to exploring AEs for text classification tasks, this work investigates how to systematically generate alterfactual examples for text classifiers. We propose a framework, called NOMATTERXAI, that can significantly change different irrelevant features of an input text to generate alterfactual for a target ML classification model. Our contributions are summarized as follows.

1. We elucidate a formal definition of AEs for text data. This definition is in an ideal theoretical form, and we explicate how it is translated to our solution.   
2. We introduce a novel algorithm NOMATTERXAI, which generates alterfactual variants of input texts, such that one or more irrelevant words are changed by opposite words selected via two strategies, ConceptNet and ChatGPT while

maintaining almost no noticeable changes in prediction probability and original context similarity.

3. We conduct both automatic and human evaluations on four real-world text datasets, and three text classifiers, achieving up to $9 5 \%$ in effectiveness of generating AEs, showing that such AEs can support humans to accurately compare biases among different classification models.

# Background and Motivation

This section provides a summary of a variety of factual explanation examples applied to the NLP domain, including semifactuals, counterfactuals, and adversarials. This will help distinguish the alterfactual from the rest (Table. 1).

Counterfactuals are shown to be intuitive to humans by explaining “Why $X ,$ , rather than Y” for a model’s decision such as “This email would be classified as ham rather than spam if there were $50 \%$ less exclamation points” (Le, Wang, and Lee 2020). Counterfactual explanations (CEs) are traditionally used in classification tasks (Verma, Dickerson, and Hines 2020) and recently information retrieval tasks (Kaffes, Sacharidis, and Giannopoulos 2021; Agarwal et al. 2019; Tan et al. 2021). They tend to be minimal such that the input is perturbed as little as possible to yield a contrasting output (Kenny and Keane 2021).

Semifactuals explain “Even if X, still P.”, or that an identical outcome occurs despite some noticeable change in the input, explaining such as “This email is still spam even if it had 3 exclamation marks instead of 6”. The exact definition varies, either as an input that is modified to be closer to the decision boundary (Kenny and Keane 2021) or others consider any input of the same class to be semifactual (Kenny and Huang 2023).

Adversarials result from slight alterations to an input designed to fool an ML model’s prediction. While closely related to counterfactual explanations (CEs) (Le, Wang, and Lee 2020), adversarial examples differ in their intent—i.e., to confuse a model rather than provide interpretability. They are similar to CEs in that they involve minimal changes intended to yield a different classification. However, adversarial attacks are typically crafted to be imperceptible to humans, whereas counterfactuals are meant to be detectable and interpretable by humans.

Alterfactuals as defined by Mertes et al. (2022), is a variant of semifactuals:

DEFINITION 1. Alterfactual Example in ML. Let denote $d$ be a distance metric on input space $X$ , $d : X { \times } X $ $R$ . An alterfactual example of an example $x$ with a model $M$ is an altered version $x ^ { * } \in X$ , that maximizes the distance $d ( x , x ^ { * } )$ with the distance to the decision boundary $B$ and the prediction of the model do not change–i.e., $d ( x , B ) \approx d ( x ^ { * } , B )$ and $f ( x ) { = } f ( x ^ { * } )$ .

Motivation. While CEs present scenarios where negligible changes can alter an outcome, they focus less on identifying which features are irrelevant. Because feature changes in CEs are minimal, these explanations may fail to capture all the factors influencing a model’s decision-making, including both relevant and irrelevant signals. Adversarial explanations (AEs), on the other hand, can highlight irrelevant features by exaggerating their influence (Mertes et al. 2022). This perspective offers a novel and intriguing approach to model explanation. However, Mertes’ study primarily measures the effectiveness of AEs in explaining model behaviors to users. In our work, we aim to examine how AEs can be automatically generated in practice and propose the first method of its kind for the text domain. We refer to the Appendix for a detailed comparison of different types of factual statements.

# Problem Formulation

Given a sentence $x$ and text classifier $M$ , our goal is to generate new AE $x ^ { * }$ , to provide interpretable information on irrelevant features of $x$ of the prediction $f ( x )$ . According to the Definition. $\jmath$ , we hope to generate AE $x ^ { * }$ is changing $x$ as much as possible, or:

$$
\operatorname* { i m a x } _ { x ^ { * } } d ( x , x ^ { * } )
$$

Moreover, for $x ^ { * }$ an alterfactual example, it needs to maintain a similar distance to the decision boundary to the original predicted class and at the same time preserve the original prediction, or:

$$
\operatorname { a r g m a x } ( f ( x ^ { * } ) ) { \mathrm { = a r g m a x } } ( f ( x ) ) \land | f ( x ) - f ( x ^ { * } ) | { \leq } \delta ,
$$

where $\delta$ is a small threshold constraining how much the original prediction probability can shift. However, without any additional constraint, $x ^ { * }$ might not necessary preserve the same context of $x$ and can even result in meaningless sentences (e.g., “today is monday” $\right. ^ { \left. }$ “today is school”). Thus, we want to perturb the original input $x$ (grey circle) to generate optimal $x ^ { * }$ that is also furthest away from $x$ and $x ^ { * }$ to be still within the context space of $x$ , denoted as $S _ { x }$ , or:

$$
{ \boldsymbol { x } } ^ { * } \in S _ { \boldsymbol { x } } ,
$$

However, it remains non-trivial to systematically manipulate an entire sentence $x$ in the discrete text space. While manipulating $x$ via its embedding in the continuous vector space is possible, such approaches may produce $x ^ { * }$ that is drastically different from $x$ , introducing numerous random changes that are no longer interpretable to users. To address this challenge, we can perturb $x$ through word-level replacements, as commonly done in existing counterfactual explanation (CE) works. By replacing individual words with semantically distant alternatives–e.g., $" p r e t t y " \to " u g l y " .$ , we aim to move the entire sentence $x$ as far as possible while maintaining interpretability. We then opt for perturbing only irrelevant features $x _ { \mathrm { i r } } ^ { \ast }$ of $x ^ { * }$ . Eq. 1 becomes:

$$
\operatorname* { m a x } _ { \boldsymbol { x } _ { \mathrm { i r } } ^ { * } } d ( \boldsymbol { x } , \boldsymbol { x } ^ { * } )
$$

Perturbing only the irrelevant features $x _ { \mathrm { i r } }$ of $x$ provides a more specific and intuitive “no matter what” explanation. For instance, an explanation like: “no matter how we change ‘pretty’ (e.g., to ‘ugly’) in the sentence, the prediction remains the same”. This approach not only ensures interpretability but also increases the likelihood that $x ^ { * }$ remains parallel to the decision boundary. In contrast, perturbing relevant or important features is more likely to significantly alter the prediction probability, thereby reducing the utility of the explanation.

![](images/88ff172b9805d2ddd46835579f2f1d20fd81637c371d866e5933a01313dbf5e5.jpg)  
Figure 2: AE generation of $x ^ { * }$ (orange circle) from $x$ (grey circle) by perturbing irrelevant features $x _ { \mathrm { i r } }$ of $x$ within their semantic fields while still maintaining original context of $x$ .

Still, we cannot replace $x _ { \mathrm { i r } }$ with just any perturbation $x _ { \mathrm { i r } } ^ { \ast }$ . For example, good perturbations include antonyms–e.g., “he” $ ^ { \cdot }$ “she” as in “no matter what the gender of the person, the classifier still predicts hate-speech”, or members of a distinct group–e.g., “red”, “blue”, “green” (colors), “democrats”, “republicans” (political leaning) as in “no matter what the political leaning of the user, the classifier still predicts nonhate-speech’. To enforce this constraint, we require that the replacement token needs to share the same semantic field (Jurafsky 2000) with the original one, or:

$$
\operatorname { s } ( x _ { \mathrm { i r } } ^ { * } ) = \operatorname { s } ( x _ { \mathrm { i r } } ) \ \forall \ x _ { \mathrm { i r } } ^ { * } \in x ^ { * } ,
$$

where $x _ { \mathrm { i r } }$ and $x _ { \mathrm { i r } } ^ { \ast }$ denote arbitrary a pair of original and replacement word and $s ( \cdot )$ queries the semantic field of a word. This constraint makes perturbations such as “Monday” $$ “cool” in “today is Monday and the weather is nice” unfeasible because “cool” and “Monday” does not share the same semantic field, although “cool” is semantically far away from “Monday” and still somewhat preserves the original context. This results in the objective function below.

OBJECTIVE FUNCTION: For a given document $x$ with irrelevant features $x _ { i r }$ , text classifier $M$ , and threshold hyperparameter $\delta$ , our goal is to generate an alterfactual example $x ^ { * }$ of $x$ by solving the objective function:

$$
\begin{array} { r l } & { \underset { \{ x _ { \mathrm { i r } } ^ { * } \in x ^ { * } \} } { \operatorname* { m a x } } d ( x , x ^ { * } ) \mathrm { s . t . } } \\ & { \underset { \mathrm { a r g } \operatorname* { m a x } } { \operatorname* { m a x } } ( f ( x ^ { * } ) ) = \underset { \mathrm { a r g } \operatorname* { m a x } } { \operatorname { m a x } } ( f ( x ) ) , } \\ & { d [ f ( x ) - f ( x ^ { * } ) ] \leq \delta , } \\ & { \quad \quad \quad x ^ { * } \in S _ { x } } \\ & { \quad \quad \quad \mathrm { s } ( x _ { \mathrm { i r } } ^ { * } ) = \mathrm { s } ( x _ { \mathrm { i r } } ) \forall x _ { \mathrm { i r } } ^ { * } \in x ^ { * } } \end{array}
$$

# Proposed Method: NOMATTERXAI

To solve the objective function, we propose a novel greedy algorithm called NOMATTERXAI. Overall, NOMATTERXAI involves two steps. Given an input text $x$ , it selects a maximum of $m$ words to perturb in order according to their importance to the prediction $f ( x )$ . Each word is greedily perturbed with its counterparts while ensuring all the constraints are satisfied. A detailed algorithm is described in the Alg.1.

Step 1: Irrelevant Feature Selection. Each feature of $x$ is ranked from lowest to highest predictive importance based on the probability drop in the original predicted class when they are individually removed from $x$ (lines 2-5 in Alg. 1). We

# Algorithm 1: AE Generation by NOMATTERXAI

Require: Input sentence $x { = } \{ w _ { 1 } , w _ { 2 } , { \ldots } , w _ { n } \}$ , target model $f ( \cdot )$ , sentence similarity threshold $\epsilon$ , current perturbations $p _ { c }$ , current confidence score $c$ , sentence similarity function $\mathrm { s i m } ( \cdot )$ .   
Output: AE $x ^ { * }$ , confidence score post perturbation $c ^ { * }$ .   
1: Initialize $x ^ { * }  x$ , $i {  } 0$ , $p _ { c } { \gets } 0$ , ${ \delta  \bar { 0 } . 0 5 }$ , $c \gets f ( x )$   
2: for each word $w _ { i } \in x$ do   
3: Compute the importance score $\boldsymbol { I _ { w _ { i } } }$ .   
4: end for   
5: Create a set $W$ of all words $w _ { i } \in x$ sorted by the ascending order of their importance score ${ { I } _ { { { w } _ { i } } } }$   
6: while $i \leq \mathrm { l e n g t h } ( W )$ do   
7: Find antonyms $a _ { i }$ of $W [ i ]$ by ChatGPT or ConceptNet. 8: $x ^ { \prime } \gets$ Replace $W [ i ]$ with $a _ { i }$ in $x ^ { * }$   
9: double negative check $\ l =$ Double Negative $( x ^ { \prime } )$   
10: if double negative check $\scriptstyle = =$ False then   
11: 12: $\begin{array} { l } { c ^ { \prime } = f ( x ^ { \prime } ) } \\ { \operatorname { c o n d 1 } = | c ^ { \prime } - c | \leq \delta \operatorname { A N D } \operatorname { a r g m a x } ( c ) { = } { = } { \operatorname { a r g m a x } } ( c ^ { \prime } ) } \\ { \operatorname { c o n d 2 } = \operatorname { s i m } ( x , x ^ { \prime } ) { \geq } \epsilon } \end{array}$ 13:   
14: if cond1 AND cond2 then   
15: $c ^ { * }  c ^ { \prime } ; x ^ { * }  x ^ { \prime }$   
16: end if   
17: $i \gets i + 1$   
18: end if   
19: end while   
20: return $( \boldsymbol { x } ^ { * } , \boldsymbol { c } ^ { * } )$

prioritize perturbing features of lower importance–a.k.a., irrelevant features, first, since their perturbations are less likely to alter the model’s prediction probability to the predicted class. Then, we iteratively transform one word at a time until we have checked a maximum of $m$ words (lines 6-16 in Alg. 1). Hyper-parameter $m$ is set to ensure that (1) there are not too many perturbations in $x$ that could make the resulting AEs difficult to interpret and (2) reduce unnecessary runtime.

Step 2: Feature Perturbation with Opposite Word. We want to perturb the selected irrelevant features “farthest away” to their originals to move $x ^ { * }$ to right at the boundary of $S _ { x }$ as depicted in Fig. 2. Moreover, such perturbations also need to share the same semantic field of the original token (Eq. 5). We call these opposite words and adopt the definition of oppositeness in terms of incompatibility in the linguistic literature–i.e., that is, for example, “if a thing can be described by one of the members of an antonym pair, it can’t be described by the other” (Keith 2022). Such opposite words also often share the same semantic field of the original one (Li 2017; Jurafsky 2000).

However, coming up with such perturbations is non-trivial as there is no clear quantitative measure for oppositeness for a word, and most of the relevant literature often desires semantically similar rather than opposite replacements such as in adversarial NLP. Even if we add noise to the original token’s embedding to find replacements, it would require very different bound of noise for different words to be still in the same semantic field–i.e., the grey region in Fig. 2 is dependent on $x _ { \mathrm { i r } }$ . For example, the $L _ { 2 }$ distance between Glove word embeddings (Pennington, Socher, and Manning 2014) between “pretty” and “ugly’, “Monday” and “Tuesday”, “republicans” and “democrats” are very different: 3.9, 0.4 and

Table 2: Examples of retrieved opposites from ConceptNet.   

<html><body><table><tr><td>Method</td><td>Example</td></tr><tr><td>Original</td><td>The children listened to jazz all day.</td></tr><tr><td>Antonym</td><td>The adults listened to jazz all day.</td></tr><tr><td>DistinctFrom</td><td>The children listened to jazz all month.</td></tr><tr><td>Hyponym</td><td>The children listened to rock all day.</td></tr></table></body></html>

1.3, respectively. Thus, adding a fixed amount much noise might end up in perturbations that are inappropriate.

Therefore, we adopt two different strategies that both leverage external knowledge to find opposite words for replacements: finding perturbations via the ConceptNet database and large language models (LLMs).

Opposite Words Selection via ConceptNet. The selected database for identifying antonymous words is the userannotated knowledge base ConceptNet (Speer, Chin, and Havasi 2017). ConceptNet’s word relations are notably annotated with numerical weightings through various sources. For a transformation of an input word, the following hierarchy of choices is used to identify opposite words (Table. 2).

• Antonyms: ConceptNet’s API is called to check for words registered as the input word’s antonym via the /r/Antonym relation, such that the weight of the relation is over $\omega _ { t }$ .

• Distinct Items: ConceptNet’s API is called to check for words registered as members of a common set via the /r/DistinctFrom relation, such that something that is A is not B (e.g. red and blue), and that the weight of relation is over $\omega _ { t }$ . This ensures that choices of transformed words remain within common groups and can be adequately selected.

• Hypernym’s Hyponym: We check for an umbrella term, referred to as a hypernym via ConceptNet’s /r/IsA relation, under which the input word belongs. For example, “rose”, “lilac” and “iris” are all hyponyms of “flowers”. If one is found, a query is made to identify members of the identified category that are not the input word, such that a member of the same category is to be selected. This is intended to identify words that are members of some overarching group, as similarly done in Distinct Items.

Opposite Words Selection via LLM. ChatGPT (OpenAI 2023) estimates the likelihood of subsequent tokens in a textbased on preceding words. We employ the inferred contextual understanding that ChatGPT can offer to identify antonyms. ChatGPT 3.5-Turbo is called for each input sentence and asked to provide one context-relevant antonym per word in the sentence such that the original sentence is still grammatically correct with the antonym replacement. Please refer to the Appendix for full details of the prompt.

Avoiding Double Negatives. When words are changed for antonyms, some words have negative counterparts, such as ”is” to ”isn’t”. Multiple of these may cause double-negatives to arise in sentences, which may cause the user-interpreted meaning of the text to not significantly change. To address this, we create a constraint to detect and reject potential double-negative sentences, unless the original text also featured a double-negative. This reduces potential confusing alterfactual texts to be returned to users. Of these replacements, we only keep those that do not create a double negative, do not exchange a word for one that is a different part of speech (ex. noun $$ verb), and that do not alter the model output confidence score $\delta$ beyond $5 \%$ . The detailed algorithm is described in the Appendix.

<html><body><table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="5">DistilBERT</td><td colspan="5">BERT</td><td colspan="5">RoBERTa</td></tr><tr><td>FID↑ AWP↑</td><td></td><td></td><td></td><td></td><td>APPL↓SIM↑CON↓FID↑ AWP↑</td><td></td><td></td><td></td><td></td><td>APPL↓SIM↑CON↓FID↑ AWP↑</td><td></td><td>APPL↓SIM↑CON↓</td><td></td><td></td></tr><tr><td rowspan="4"></td><td>Feng et al</td><td>95.56</td><td>6.65</td><td>156.57</td><td>0.81</td><td>1.61</td><td>96.58</td><td>6.44</td><td>154.51</td><td>0.83</td><td>1.67</td><td>96.98</td><td>6.45</td><td>153.99</td><td>0.84</td><td>1.68</td></tr><tr><td>CNet-Single 77.78</td><td></td><td>1.00</td><td>86.41</td><td>0.86</td><td>1.43</td><td>79.44</td><td>1.00</td><td>86.15</td><td>0.86</td><td>1.31</td><td>78.53</td><td>1.00</td><td>86.86</td><td>0.86</td><td>1.33</td></tr><tr><td>GPT-Single 70.83</td><td></td><td>1.00</td><td>83.90</td><td>0.86</td><td>1.36</td><td>69.49</td><td>1.00</td><td>83.58</td><td>0.86</td><td>1.21</td><td>67.93</td><td>1.00</td><td>83.39</td><td>0.86</td><td>1.26</td></tr><tr><td>CNet-Multi 77.78</td><td></td><td>1.55</td><td>98.19</td><td>0.88</td><td>1.19</td><td>79.44</td><td>1.58</td><td>98.05</td><td>0.87</td><td>1.07</td><td>78.55</td><td>1.57</td><td>94.29</td><td>0.87</td><td>1.10</td></tr><tr><td rowspan="6"></td><td>GPT-Multi</td><td>70.83</td><td>1.56</td><td>98.29</td><td>0.89</td><td>1.24</td><td>69.49</td><td>1.59</td><td>99.30</td><td>0.89</td><td>1.04</td><td>67.93</td><td>1.57</td><td>98.01</td><td>0.89</td><td>1.15</td></tr><tr><td>Feng et al</td><td>99.17</td><td>7.87</td><td>89.63</td><td>0.82</td><td>0.55</td><td>98.77</td><td>7.77</td><td>87.90</td><td>0.84</td><td>0.36</td><td>99.18</td><td>7.84</td><td>90.93</td><td>0.85</td><td>0.36</td></tr><tr><td>CNet-Single 92.26</td><td></td><td>1.00</td><td>76.18</td><td>0.87</td><td>0.47</td><td>92.21</td><td>1.00</td><td>76.49</td><td>0.87</td><td>0.36</td><td>92.29</td><td>1.00</td><td>77.25</td><td>0.87</td><td>0.25</td></tr><tr><td>GPT-Single 80.14</td><td></td><td>1.00</td><td>75.71</td><td>0.88</td><td>0.44</td><td>79.32</td><td>1.00</td><td>75.94</td><td>0.88</td><td>0.32</td><td>84.87</td><td>1.00</td><td>90.84</td><td>0.91</td><td>0.24</td></tr><tr><td>CNet-Multi 92.26</td><td></td><td>2.34</td><td>88.48</td><td>0.84</td><td>0.33</td><td>92.21</td><td>2.40</td><td>90.09</td><td>0.84</td><td>0.27</td><td>92.29</td><td>2.46</td><td>89.62</td><td>0.85</td><td>0.18</td></tr><tr><td>GPT-Multi</td><td>80.28</td><td>2.24</td><td>87.06</td><td>0.87</td><td>0.37</td><td>79.32</td><td>1.21</td><td>87.34</td><td>0.88</td><td>0.31</td><td>84.87</td><td>3.47</td><td>98.93</td><td>0.87</td><td>0.20</td></tr><tr><td rowspan="5">5If</td><td>Feng et al</td><td>97.29</td><td>26.82</td><td>121.00</td><td>0.81</td><td>0.83</td><td>97.76</td><td>27.77</td><td>124.6</td><td>0.85</td><td>0.76</td><td>96.65</td><td>26.27</td><td>149.87</td><td>0.84</td><td>0.58</td></tr><tr><td>CNet-Single 89.79</td><td></td><td>1.00</td><td>76.81</td><td>0.92</td><td>1.37</td><td>89.99</td><td>1.00</td><td>78.18</td><td>0.92</td><td>1.59</td><td>90.10</td><td>1.00</td><td>75.68</td><td>0.92</td><td>0.88</td></tr><tr><td>GPT-Single 82.45</td><td></td><td>1.00</td><td>75.02</td><td>0.92</td><td>1.35</td><td>83.35</td><td>1.00</td><td>77.25</td><td>0.93</td><td>1.59</td><td>80.38</td><td>1.00</td><td>75.17</td><td>0.93</td><td>0.86</td></tr><tr><td>CNet-Multi 89.83</td><td></td><td>3.91</td><td>105.85</td><td>0.88</td><td>0.52</td><td>89.99</td><td>3.62</td><td>104.58</td><td>0.88</td><td>0.62</td><td>90.10</td><td>5.13</td><td>116.47</td><td>0.88</td><td>0.27</td></tr><tr><td>GPT-Multi</td><td>82.52</td><td>3.93</td><td>98.19</td><td>0.89</td><td>0.64</td><td>83.35</td><td>6.51</td><td>106.39</td><td>0.90</td><td>0.61</td><td>80.38</td><td>10.40</td><td>114.99</td><td>0.93</td><td>0.27</td></tr><tr><td rowspan="5"></td><td>Feng et al</td><td>98.65</td><td>12.80</td><td>254.32</td><td>0.81</td><td>0.42</td><td>99.78</td><td>11.60</td><td>280.84</td><td>0.81</td><td>0.68</td><td>99.89</td><td>11.14</td><td>349.74</td><td>0.80</td><td>0.43</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CNet-Muli 95.68</td><td></td><td>1.00</td><td>92.17</td><td>0.84</td><td>0.9</td><td>94.83</td><td>1.00</td><td>94.57</td><td>0.86</td><td>0.57</td><td>95.20</td><td>1.00</td><td>93.0.6</td><td>0.89</td><td>0.30</td></tr><tr><td>GPT-Single 86.93</td><td></td><td>1.00</td><td>95.01</td><td>0.88</td><td>0.20</td><td>90.31</td><td>1.00</td><td>94.06</td><td>0.87</td><td>0.54</td><td>89.65</td><td>1.00</td><td>96.42</td><td>0.90</td><td>0.31</td></tr><tr><td>GPT-Multi 86.93</td><td></td><td>3.10</td><td>146.34</td><td>0.85</td><td>0.16</td><td>90.31</td><td>4.26</td><td>153.09</td><td>0.84</td><td>0.46</td><td>89.65</td><td>4.60</td><td>165.27</td><td>0.84</td><td>2.52</td></tr></table></body></html>

Table 3: Summary of quantitative performance comparisons of NOMATTERXAI.

Table 4: Examples of AEs generated by NOMATTERXAI.   

<html><body><table><tr><td>Original</td><td>Alterfactual Example</td></tr><tr><td>Your comment makes no sense Your comment makes no sense and is incoherent</td><td>and is coherent</td></tr><tr><td>Impossible to understand the stu-Impossible to misunderstand the pidity of someone[...]</td><td>stupidity of someone</td></tr><tr><td>say that the woman was 'ille- to say that the woman was 'ille- gally’ refused entry to the US. gally’ approved entry to the US. Obviously it is perfectly legal Obviously it is perfectly illegal for the US[..].My guess is that for the US[..].My guess is that the refusal was based on her pur- the refusal was based on his pur- ported engagement to a USciti- ported engagement to a US citi- zen[.] situation carefully.</td><td>Mulcair's comment was silly, to Mulcair's comment was mature, zen [..] carefully.</td></tr></table></body></html>

# Experiment Settings

This section shows a comprehensive evaluation of NOMATTERXAI with different settings and baselines.

Datasets and Models. We use datasets of varied tasks, including gender bias(GB) (Dinan et al. 2020), hate speech classification (HS)(Davidson et al. 2017), emotion classification (EMO) (Saravia et al. 2018), and the toxicity detection in social comments (JIG) 1. They vary in average sentence length (9.3, 13.72, 43.38, 19.15 tokens) and number of labels (2,2,2,6). Each dataset is split into $80 \%$ training and $20 \%$ test splits, and we use the training set to train three target models, namely DistilBERT (Sanh et al. 2019), BERT (Devlin et al. 2019) and RoBERTa (Liu et al. 2019). Please refer to the Appendix for more details.

Evaluation Metrics. We report the following metrics: Fidelity $( \mathrm { F I D \uparrow } )$ , or the percentage of texts that we can generate an AE; Runtime $\mathrm { ( T i m e \downarrow ) }$ ; Average Words Perturbed $( \mathsf { A W P \downarrow } )$ ; Average Queries $( \mathrm { A V Q \downarrow } )$ or an average number of queries made to target models; Altered Perplexity $( \mathrm { A P P L } \downarrow ,$ ), or the naturalness of $x$ and $x ^ { * }$ captured via GPT2-Large as a proxy (Radford et al. 2019); semantic similarity through the USE Encoder (Cer et al. 2018) $( \mathrm { S I M } \uparrow )$ ; and the models’ average confidence shift (in $\%$ ) after perturbations $( \mathrm { C O N \downarrow } )$ .

Implementation Details. We select our confidence threshold $\delta \mathrm {  0 . 0 5 }$ to allow the model output to only shift at most $5 \%$ in confidence. Constraint Eq. (3) is satisfied by setting a minimum context similarity threshold $\epsilon { = } 0 . 8$ via USE Encoder (Cer et al. 2018). We constrain NOMATTERXAI’s perturbations by preventing repeat perturbations and disregarding a list of stopwords. During perturbation, a word is not altered if either ConceptNet or GPT fails to return an option. Please refer to the Appendix for full details.

Baseline. We evaluated two variants of NOMATTERXAI, one uses ConceptNet (CNet) and another uses ChatGPT LLM (GPT) for looking up replacement candidates. We also test NOMATTERXAI when perturbing only one word (denoted by “-Single” suffix) and when perturbing as many words as we can (denoted by “-Multi” suffix). Since there is no existing method that specifically generates AEs, we adopt (Feng et al. 2018), a method that iteratively removes the least important word from the input as an additional baseline.

# Results

Table 4 depicts a few AEs synthesized by NOMATTERXAI. We describe in detail the evaluation results on different computational aspects below, followed by a user-study experiment that evaluates the explainability of the generated AEs in practice with human subjects.

Generation Success Rate–i.e., Fidelity $( \mathbf { F I D \uparrow } )$ . Being the first of its kind, NOMATTERXAI can find AEs around $70 \%$ up to $9 5 \%$ of the time. The baseline (Feng et al. 2018) has a better chance of finding AEs by iteratively removing a set of least important words (Table. 3), it totally discards the original contextual meaning of the sentence. This happens because deleting too many words would cause the resulting sentences to lose both semantic coherence and grammatical correctness. As a result, (Feng et al. 2018) baseline results in a significantly higher (undesirable) perplexity on the perturbed samples and much lower reports on context preservation compared to NOMATTERXAI.

Context Preservation–i.e., Context Similarity $( \mathbf { S I M } \uparrow )$ ). Baseline (Feng et al. 2018) consistently ranks lower in context preservation to NOMATTERXAI (Table. 3). This suggests that simply removing words fails to preserve the meaning of the original sentence. In contrast, using LLM like ChatGPT to generate replacement candidates yields the highest similarity in most cases. This happens because LLMs are well-designed to capture semantic meaning in natural language from vast amounts of data (Chang et al. 2024).

Changes in Prediction Probability $\mathbf { ( C O N \downarrow ) }$ ). Due to the constraints of the search condition, we observe that the alterfactual examples generated by NOMATTERXAI do not move significantly away from the original predicted class, as reflected by the near-zero average changes in prediction probabilities of $0 . 7 3 \%$ , $0 . 7 7 \%$ , and $0 . 7 1 \%$ for DistilBERT, RoBERTa, and BERT, respectively. This indicates that NOMATTERXAI can produce alterfactual examples that diverge from the input while remaining aligned with the original model’s decision boundary.

Comparison with Alternative Perturbation Strategy. We compare the use of ConceptNet against an alternative strategy of perturbation by adding noise to word embeddings as showed in Fig. 2. To do this, we add Gaussian noises of incrementally increasing in magnitude to the embeddings of the original tokens and check (1) whether the resulting embeddings actually convert to a new token (Flip Rate) and whether the resulting sentences preserve the context similarity $( \mathrm { S I M } \uparrow )$ ). Fig. 3 shows that NOMATTERXAI can select suitable opposite words while maximizing SIM with much fewer changes in embedding space measured by $L _ { 2 }$ . This shows that such an alternative strategy will not work in practice as it significantly drifts from the original context as bigger noise is added to ensure a high Flip Rate. ConceptNet is more suitable for

L2 vs Flip Rate L2 vs SIM 1. 6 \~\~\~! 1/ 0.9 \~\~ 0.78 Ar 八 SIM 4 Flip Rate SIM (Ours) L2 Flip Rate (Ours) 3 2 0.5 L2 (Ours) 1 2 3 4 2 3 4 5 Noise Magnitude to Word Vectors

8 1.00   
1.00   
0.75 EOumrpsr(icriocrarl=B- i0a.7s14) Ours-Gender(corr=-0.743) 0.75   
0.25 0.2550 0.00 0.00 0.00 0.25 0.50 0.75 1.00 Normalized Bias Mixing Fraction

finding opposite words, which might not be systematically quantifiable in the embedding space.

Correlation with Model Bias Detection. Since AEs emphasize irrelevant features, a model that is highly biased against gender should result in almost no AEs–i.e., near zero fidelity when we only perturb identity words–e.g., “she”, “he”. Similarly, an unbiased model should result in high fidelity. To further evaluate the generated AEs’ qualities, we measure how well NOMATTERXAI’s fidelity correlates with automated bias detection metrics, especially when we target identity words to perturb. Fig. 4 confirms the quality of NOMATTERXAI. This also shows the potential utility of NOMATTERXAI in approximating bias levels of text classifiers.

# User Study Experiment

In this section, we evaluate the applications of AEs with end users recruited from Amazon Mechanical Turk (MTurk). We aim to answer the question: Can AEs be useful for humans to judge the model fairness?

Hypothesis. We evaluate whether AEs generated by NOMATTERXAI can inform the users about the relative bias rankings among three AI models of different empirical bias levels ${ \mathrm { A } } { - } 1 7 . 1 \%$ , D- $5 . 5 \%$ , and $\mathrm { E - } 0 . 7 \%$ ). Such rankings are significantly useful in practice to decide which AI models should be prioritized for deployment. Particularly, we define three alternative hypotheses $\mathcal { H } _ { a }$ (Table. 5) to validate whether or not college-level users can correctly identify three pair-wise rankings better than a random guess by using explanations generated by NOMATTERXAI.

Table 5: User study experiment results with different $\mathcal { H } _ { a }$ of different gaps $\Delta$ in empirical bias scores.   

<html><body><table><tr><td>Ha</td><td>AlternativeHypothesis</td><td>df t-test</td><td>p-value</td></tr><tr><td></td><td>H1 Correct Ranking: A>E(△=16.4%) 45 2.01</td></tr><tr><td>H2 Correct Ranking: A>D(△=11.6%) 36 2.94</td><td>0.026* 0.003**</td></tr></table></body></html>

$\overline { { ( ^ { * } ) , ( ^ { * * } ) } }$ statistical significance with $\overline { { \alpha { = } 0 . 0 5 } }$ and $\overline { { \alpha { = } 0 . 0 1 } }$

Study Design. Whether or not a model is biased cannot be quantified with individual prediction instances. To evaluate such property, we perturb all gender words on 500 test examples curated from the JIG dataset to generate AEs and use them to curate a text explaining this global behavior along “No matter what we changed the genders mentioned in the input texts (like male $$ female, she $ h e$ , woman $$ man, etc.), the computer system’s decisions remained the same for $1 . 8 \%$ of the time”. We present such an explanation for each of the two models–e.g., A&D, A&E, etc., and ask the participants to rank which model is less biased towards gender? Please refer to the Appendix for more details.

Participant Recruitment and Quality Assurance. We recruited adult ${ \it \Omega } > 1 8$ years old) participants from the USA on MTurk without assuming any knowledge of AI or ML. We pay each completed response $\mathrm { U S S 0 . 5 0 }$ for roughly 2 minutes of work, resulting in $\$ 12$ /hour average wage. We employ a three-layer quality assurance procedure. First, we utilize worker tags provided by MTurk to only select subjects having done at least 5,000 tasks with over $\dot { \geq } 9 8 \%$ acceptance rate and completing U.S. Bachelor’s degree. Second, we deploy a trivial attention check question to make sure the workers read and understand the instructions. Third, we provided incentives to the workers as an additional bonus payment of $\mathrm { U S S 0 . 5 0 }$ for every correct answer to encourage their attention to the task. We also record the time each worker spends on the study to filter out obvious low-quality responses.

Results. We collected responses from a total of 149 workers and discarded data from 29 workers due to (i) low attention time $\leq 1 0$ seconds) and/or (ii) incorrect answers to the attention check question. It is statistically significant to reject the null hypothesis in all cases using a one-sample t-test (ranking accuracy ${ \ > } 0 . 5$ ) (Table. 5). This shows that explanations synthesized from AEs can effectively support the users to effectively compare the models’ biases. On average, we also observe that workers who passed the attention question were both more confident (p-values $_ { \cdot < 0 . 0 5 }$ , except $\mathcal { H } _ { 1 }$ ) and accurate (p-values ${ \ K } 0 . 0 5$ ) at answering the ranking question. This shows that a minimal understanding of bias in AI models is a prerequisite for our task and the inclusion of such attention-check questions was crucial.

GB 27.7% 22.5% 2.2% 47.6% HS 14.2% 20.4% 1.4% 64.0% JIG 11.6% 17.1%1.2% 70.1% EMO 24.8% 29.8% 2.4% 43.1% Antonym DistinctFrom Hypernym's Hyponym Not Found

# Discussion

Computational Complexity Trade-Off. In this section, we analyze the time complexity of NOMATTERXAI algorithm (Alg. 1) on each input example. Computing the important scores takes $O ( k V )$ , where $V$ is the time complexity of a forward pass or query to the target classifier, and $k$ is the number of words in the original sentence. Sorting the list of $k$ importance scores takes $\bar { O ( k l o g k ) }$ with QuickSort. Finding opposite words and checking for the constraints takes $O ( k V )$ . To sum up, the overall time complexity of NOMATTERXAI to generate an AE for one instance is $\overset { \cdot } { O } ( k \mathrm { l o g } k { + } k V )$ . We further provide the correlation between runtime and the number of queries to the target models in the Appendix.

Limitations of Perturbations with ConceptNet and ChatGPT. ConceptNet (Speer, Chin, and Havasi 2017) is tied to the limited contents of its database. Some antonyms such as “glow” to “dim”, are not present in the database at the time of writing. Additionally, a significant number of query calls yielded no result (Fig. 5). From our analysis of ConceptNet versus an alternative strategy, we once again emphasize that quantitatively finding opposite words is very challenging. While ChatGPT 3.5 is effective at generating opposite words most of the time, hallucinations do occur–e.g., replacing queried words with “antonym”, although only on rare occasions.

Other Limitations. A limitation with generating AEs as compared to existing explanations is the increased runtime. CEs are minimal in nature such that as few words as possible are perturbed, as compared to NOMATTERXAI, which aims to perturb as many words as possible. It is unclear if this will reduce the incentive to use AEs as compared to counterfactual examples, although their efficacy was shown to be similar in a previous evaluation with humans (Mertes et al. 2022).

# Conclusion and Future Work

We extend the theoretical definition of alterfactuals (Mertes et al. 2022) to propose NOMATTERXAI, an automatic greedy-based mechanism that can generate alterfactual examples up to $9 5 \%$ of the time to explain text classifiers. Through a human study, AEs generated by NOMATTERXAI show to help synthesize “no matter what” XAI texts to convey to users the irrelevancy in predictive features and reveal comparative bias behaviors among several target models. Future works include improving the opposite-word identification process.

# Ethical Statement

Our work aims to improve the interpretability and fairness of black-box text classification models by proposing NOMATTERXAI, which generates alterfactual explanations (AEs). These explanations emphasize irrelevant features to ensure that AI predictions remain consistent regardless of specific attributes (e.g., gender, race, or political orientation), helping to detect and mitigate biases in AI models. We ensure that our approach aligns with ethical AI principles by conducting evaluations using publicly available datasets and welldocumented models. Additionally, we conducted a user study involving human participants with appropriate informed consent and compensation, adhering to ethical guidelines for research involving human subjects. Finally, our method is designed with transparency and fairness in mind, contributing to AI systems that are explainable, accountable, and less prone to hidden biases.