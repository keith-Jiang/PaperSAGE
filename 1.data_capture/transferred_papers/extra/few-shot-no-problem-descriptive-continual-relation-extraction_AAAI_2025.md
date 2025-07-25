# Few-Shot, No Problem: Descriptive Continual Relation Extraction

Nguyen Xuan Thanh 1\*, Anh Duc $\mathbf { L e } ^ { 2 \ast }$ , Quyen Tran 3\*, Thanh-Thien Le 3\*, Linh Ngo Van 2†, Thien Huu Nguyen 4

1Oraichain Labs,   
2Hanoi University of Science and Technology,   
3VinAI Research,   
4University of Oregon   
thanh.nx $@$ orai.io   
anh.ld204628@sis.hust.edu.vn   
v.quyentt15,v.thienlt3 $@$ vinai.io,   
linhnv $@$ soict.hust.edu.vn, thien $@$ cs.uoregon.edu

# Abstract

Few-shot Continual Relation Extraction is a crucial challenge for enabling AI systems to identify and adapt to evolving relationships in dynamic real-world domains. Traditional memory-based approaches often overfit to limited samples, failing to reinforce old knowledge, with the scarcity of data in few-shot scenarios further exacerbating these issues by hindering effective data augmentation in the latent space. In this paper, we propose a novel retrieval-based solution, starting with a large language model to generate descriptions for each relation. From these descriptions, we introduce a bi-encoder retrieval training paradigm to enrich both sample and class representation learning. Leveraging these enhanced representations, we design a retrieval-based prediction method where each sample ”retrieves” the best fitting relation via a reciprocal rank fusion score that integrates both relation description vectors and class prototypes. Extensive experiments on multiple datasets demonstrate that our method significantly advances the state-of-the-art by maintaining robust performance across sequential tasks, effectively addressing catastrophic forgetting.

# 1 Introduction

Relation Extraction (RE) refers to classifying semantic relationships between entities within text into predefined types. Conventional RE tasks assume all relations are present at once, ignoring the fact that new relations continually emerge in the real world. Few-shot Continual Relation Extraction (FCRE) is a subfield of continual learning (Hai et al. 2024; Van et al. 2022; Phan et al. 2022; Tran et al. 2024a,b; Le et al. 2024a) where a model must continually assimilate new emerging relations while avoiding the forgetting of old ones, a task made even more challenging by the limited training data available. The importance of FCRE stems from its relevance to dynamic real-world applications, garnering increas

FewRel (10-way, 5-shot) TACRED (5-way, 5-shot) 90 SCKD SCKD 80 CPoLnPL DCRE (Our) 80 CPoLnPL DCRE (Our) 70 60 50 T1 T2 T3 T4 T5 T6 T7 T8 T1 T2 T3 T4 T5 T6 T7 T8 Task ID Task ID

ing interest in the field (Chen, Wu, and Shi 2023a; Le et al.   
2024c, 2025).

State-of-the-art approaches to FCRE often rely on memory-based methods for continual learning (Lopez-Paz and Ranzato 2017; Nguyen et al. 2023; Le et al. 2024b; Dao et al. 2024). However, these methods frequently suffer from overfitting to the limited samples stored in memory buffers. This overfitting hampers the reinforcement of previously learned knowledge, leading to catastrophic forgetting—a marked decline in performance on learnt relations when new ones are introduced (Figure 1). The few-shot scenario of FCRE exacerbates these issues, as the scarcity of data not only impedes learning on new tasks, but also hinders helpful data augmentation, which are crucial in many methods (Shin et al. 2017).

In order to improve on these methods, we must not completely disregard them or dwell on their weaknesses, but rather contemplate their biggest strength. Why do so many methods use the memory buffer in the first place? The primary objective of these replay buffers is to rehearse and reinforce past knowledge, providing the model with something to ”look back” at during training. However, these past samples may not always be representative of the entire class and can still lead to sub-optimal performance. Based on this observation, we propose a straightforward: besides relying on potentially unrepresentative past samples, we leverage our knowledge of the past relations themselves. This insight leads to our approach of generating detailed descriptions for each relation. These descriptions inherently represent the class more accurately than the underlying information from a set of samples, serving as stable pivots for the model to align with past knowledge while learning new information. By using these descriptions, we create a more robust and effective method for Few-Shot Continual Relation Extraction, ensuring better retention of knowledge across tasks.

Overall, our paper makes the following contributions:

a. We introduce an innovative approach to Few-Shot Continual Relation Extraction that leverages Large Language Models (LLMs) to generate comprehensive descriptions for each relation. These descriptions serve as stable class representations in the latent space during training. Unlike the variability and limitations of a limited set of samples from the memory buffer, these descriptions define the inherent meaning of the relations, offer a more reliable anchor, significantly reducing the risk of catastrophic forgetting. Importantly, LLMs are employed exclusively for generating descriptions and do not participate in the training or inference processes, ensuring that our method incurs minimal computational overhead.

b. We design a bi-encoder retrieval learning framework for both sample and class representation learning. In addition to sample representation contrastive learning, we integrate a description-pivot learning process, ensuring alignment of samples which maximize their respective class samples proximity, while non-matching samples are distanced.

c. Building on the enhanced representations, we introduce the Descriptive Retrieval Inference (DRI) strategy. In this approach, each sample ”retrieves” the most fitting relation using a reciprocal rank fusion score that integrates both class descriptions and class prototypes, effectively finalizing the retrieval-based paradigm that underpins our method.

# 2 Background

# 2.1 Problem Formulation

In Few-Shot Continual Relation Extraction (FCRE), a model must continuously assimilate new knowledge from a sequential series of tasks. For each $t$ -th task, the model undergoes training on the dataset ${ D ^ { t } = \{ ( x _ { i } ^ { t } , y _ { i } ^ { t } ) \} _ { i = 1 } ^ { N \times K } }$ . Here, $N$ represents the number of relations in the task $R ^ { t }$ , and denotes the limited number of samples per relation, reflecting the few-shot learning scenario. Each sample $( x , y )$ includes a sentence $x$ containing a pair of entities $( e _ { h } , e _ { t } )$ and a relation label $y \in R$ . This type of task setup is referred to as $^ { , , } N .$ -way- $K$ -shot” (Chen, $\mathrm { w } _ { \mathrm { u } }$ , and Shi 2023a). Upon completion of task $t$ , the dataset $D ^ { t }$ should not be extensively included in subsequent learning, as continual learning aims to avoid retraining on all prior data. Ultimately, the model’s performance is assessed on a test set which encompasses all encountered relations $\begin{array} { r } { \tilde { R } ^ { T } = \bigcup _ { t = 1 } ^ { T } R ^ { t } } \end{array}$ .

For clarity, each task in FCRE can be viewed as a conventional relation extraction problem, with the key challenge being the scarcity of samples available for learning. The primary goal of FCRE is to develop a model that can consistently acquire new knowledge from limited data while retaining competence in previously learned tasks. In the following subsections, we will explore the key aspects of FCRE models as addressed by state-of-the-art studies.

# 2.2 Encoding Latent Representation

A key initial consideration in Relation Extraction is how to formalize the latent representation of the input, as the output of a Transformer (Vaswani et al. 2017) is a matrix. In this work, we adopt a method recently introduced by Ma et al. (2024). Given an input sentence $x$ , which includes a head entity $\boldsymbol { e } _ { h }$ and a tail entity $e _ { t }$ , we reformulate it into a Clozestyle phrase $T ( x )$ by incorporating a [MASK] token, which represents the relation between the entities. Specifically, the template is structured as follows:

$$
\begin{array} { c } { { T ( x ) = x \left[ v _ { 0 : n _ { 0 } - 1 } \right] e _ { h } \left[ v _ { n _ { 0 } : n _ { 1 } - 1 } \right] \left[ \mathrm { M A S K } \right] } } \\ { { \left[ v _ { n _ { 1 } : n _ { 2 } - 1 } \right] e _ { t } \left[ v _ { n _ { 2 } : n _ { 3 } - 1 } \right] . } } \end{array}
$$

Each $[ v _ { i } ]$ denotes a learnable continuous token, and $n _ { j }$ determines the number of tokens in each phrase. In our specific implementation, we use BERT’s [UNUSED] tokens as $[ v ]$ . The soft prompt phrase length is set to 3 tokens, meaning $n _ { 0 } , n _ { 1 } , n _ { 2 }$ and $n _ { 3 }$ correspond to the values of 3, 6, 9, and 12, respectively. We then forward the templated sentence $T ( x )$ through BERT to encode it into a sequence of continuous vectors, from which we obtain the hidden representation $z$ of the input, corresponding to the position of the [MASK] token:

$$
z = [ \mathcal { M } \circ T ] ( x ) [ \mathrm { p o s i t i o n } ( \mathsf { \Gamma } [ \mathsf { M A S K } ] ) ] ,
$$

where $\mathcal { M }$ denotes the backbone pre-trained language model. This latent representation is then passed through an MLP for prediction, enabling the model to learn which relation that best fills the [MASK] token.

# 2.3 Learning Latent Representation

In conventional Relation Extraction scenarios, a basic framework typically employs a backbone PLM followed by an MLP classifier to directly map the input space to the label space using Cross Entropy Loss. However, this approach proves inadequate in data-scarce settings (Snell, Swersky, and Zemel 2017). Consequently, training paradigms which directly target the latent space, such as contrastive learning, emerge as more suitable approaches. To enhance the semantic richness of the information extracted from the training samples, two popular losses are often utilized: Supervised Contrastive Loss and Hard Soft Margin Triplet Loss.

Supervised Contrastive Loss. To enhance the model’s discriminative capability, we employ the Supervised Contrastive Loss (SCL) (Khosla et al. 2020). This loss function is designed to bring positive pairs of samples, which share the same class label, closer together in the latent space. Simultaneously, it pushes negative pairs, belonging to different classes, further apart. Let $z _ { x }$ represent the hidden vector

You are a professional data scientist, working in a relation extraction project.   
Given a relation and its description, you are asked to write a more detailed description of the relation and provide 3 sentence examples of the relation. The relation is: relation_name   
The description is: raw_description   
Please generate $\mathsf { K }$ diverse samples of (relation description, examples).   
Your response:

output of sample $x$ , the positive pairs $( z _ { x } , z _ { p } )$ are those who share a class, while the negative pairs $\left( z _ { x } , z _ { n } \right)$ correspond to different labels. The SCL is computed as follows:

$$
\mathcal { L } _ { \mathrm { S C } } ( x ) = - \sum _ { p \in P ( x ) } \log \frac { f ( z _ { x } , z _ { p } ) } { \sum _ { u \in \mathcal { D } \backslash \{ x \} } f ( z _ { x } , z _ { u } ) }
$$

where $\begin{array} { r } { f ( \mathbf { x } , \mathbf { y } ) : = \exp { \left( \frac { \gamma ( \mathbf { x } , \mathbf { y } ) } { \tau } \right) } . } \end{array}$ $\gamma ( \cdot , \cdot )$ denotes the cosine similarity function, and $\tau$ is the temperature scaling hyperparameter. $P ( x )$ and $\mathcal { D }$ denote the sets of positive samples with respect to sample $x$ and the training set, respectively.

Hard Soft Margin Triplet Loss. To achieve a balance between flexibility and discrimination, the Hard Soft Margin Triplet Loss (HSMT) integrates both hard and soft margin triplet loss concepts (Hermans, Beyer, and Leibe 2017). This loss function is designed to maximize the separation between the most challenging positive and negative samples, while preserving a soft margin for improved flexibility. Formally, the loss is defined as:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { S T } } ( x ) = } \\ & { - \log \Bigg ( 1 + \underset { p \in P ( x ) } { \operatorname* { m a x } } e ^ { \xi ( z _ { x } , z _ { p } ) } - \underset { n \in N ( x ) } { \operatorname* { m i n } } e ^ { \xi ( z _ { x } , z _ { n } ) } \Bigg ) , } \end{array}
$$

where $\xi ( \cdot , \cdot )$ denotes the Euclidean distance function. The objective of this loss is to ensure that the hardest positive sample is as distant as possible from the hardest negative sample, thereby enforcing a flexible yet effective margin.

During training, these two losses is aggregated and referred to as the Sample-based learning loss:

$$
{ \mathcal { L } } _ { \mathrm { S a m p } } = \beta _ { \mathrm { S C } } \cdot { \mathcal { L } } _ { \mathrm { S C } } + \beta _ { \mathrm { S T } } \cdot { \mathcal { L } } _ { \mathrm { S T } }
$$

# 3 Proposed Method

# 3.1 Label Descriptions

A core component of our method is achieving robust class latent representations, making class encoding crucial. To this end, having detailed definitions for each label, alongisde the hidden information extracted from the samples, is essential for our approach. In fact, the datasets used for benchmarking already provide each relation with a concise description, which we refer to as the Raw description. While leveraging these descriptions has shown promise in previous work (Luo et al. 2024), this approach remains limited due to its reliance on a one-to-one mapping between input embeddings and a single label description representation per task. This singular approach fails to offer rich, diverse, and robust information about the labels, leading to potential noise, instability, and suboptimal model performance.

To address these limitations, we employ Gemini 1.5 (Team et al. 2023; Reid et al. 2024) to generate $K$ diverse, detailed, and illustrative descriptions for each relation. In particular, for each label, the respective raw description will be fed into the LLM prompt, serving as an expert-in-the-loop to guide the model. Our prompt template is depicted in Figure 2.

# 3.2 Description-pivot Learning

The single most valuable quality of class descriptions in our problem is that they are literal definitions of a relation, which makes them more accurate representations of that class than the underlying information from a set of samples. Thanks to this strength, they serve as stable knowledge anchors for the model to rehearse from, enabling effective reinforcement of old knowledge while assimilating new information. Unlike the variability of individual samples, a description remains consistent, providing a more reliable reference point for the model to rehearse from, effectively mitigating catastrophic forgetting.

To fully leverage this inherent advantage, we integrate these descriptions into the training process, framing the task as one of retrieving definition, which embodies realworld meaning, rather than a straightforward categorical classification. By doing so, we capitalize on the unchanging nature of descriptions, making them the focal point of our model’s learning. Specifically, we incorporate two description-centric losses to enhance this retrieval-oriented approach:

$$
\mathcal { L } _ { \mathrm { D e s } } = \beta _ { \mathrm { H M } } \cdot \mathcal { L } _ { \mathrm { H M } } + \beta _ { \mathrm { M I } } \cdot \mathcal { L } _ { \mathrm { M I } } .
$$

Here, ${ \mathcal { L } } _ { \mathrm { H M } }$ and ${ \mathcal { L } } _ { \mathrm { M I } }$ denote the Hard Margin Loss and the Mutual Information Loss, respectively. These losses are elaborated upon in the following paragraphs.

Hard Margin Loss. The Hard Margin Loss leverages label descriptions to refine the model’s ability to distinguish between hard positive and hard negative pairs. Given the output hidden vectors $\{ d _ { x } ^ { k } \} _ { k = 1 , . . . , K }$ from BERT corresponding to the label description of sample $x$ , and $z _ { p }$ and $z _ { n }$ representing the hidden vectors of positive and negative samples respectively, the loss function is formulated to maximize the alignment between $\pmb { d } _ { x } ^ { k }$ and its corresponding positive sample, while enforcing a strict margin against negative samples.

Specifically, the loss is formulated as follows:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { H M } } ( x ) = \displaystyle \sum _ { k = 1 } ^ { K } \mathcal { L } _ { \mathrm { H M } } ^ { k } ( x ) , } \\ & { \mathcal { L } _ { \mathrm { H M } } ^ { k } ( x ) = \displaystyle \sum _ { p \in { P } _ { \mathrm { H } } ( x ) } ( 1 - \gamma ( d _ { x } ^ { k } , z _ { p } ) ) ^ { 2 } } \\ & { \quad \quad \quad + \displaystyle \sum _ { n \in N _ { \mathrm { H } } ( x ) } m a x ( 0 , m - 1 + \gamma ( d _ { x } ^ { k } , z _ { n } ) ) ^ { 2 } , } \end{array}
$$

where $m$ is a margin hyperparameter; $\gamma ( \cdot , \cdot )$ denotes the cosine similarity function; $P _ { \mathrm { H } } ( x )$ and $N _ { \mathrm { H } } ( x )$ represent the sets of hard positive and hard negative samples, respectively. They are determined by comparing the similarity between $\pmb { d } _ { x } ^ { k }$ and both positive and negative pairs, specifically focusing on the most challenging pairs where the similarity to negative samples is close to or greater than that of positive samples, defined as follows:

$$
\begin{array} { r l } & { P _ { \mathrm { H } } ( x ) = \{ p \in P ( x ) | 1 - \gamma ( d _ { x } ^ { k } , z _ { p } ) } \\ & { \qquad > m i n _ { n \in N ( x ) } ( 1 - \gamma ( d _ { x } ^ { k } , z _ { n } ) ) , \forall k \in [ K ] \} , } \end{array}
$$

$$
\begin{array} { r l } & { N _ { \mathrm { H } } ( x ) = \{ n \in N ( x ) | 1 - \gamma ( { d } _ { x } ^ { k } , z _ { n } ) } \\ & { \qquad < m a x _ { p \in P ( x ) } ( 1 - \gamma ( { d } _ { x } ^ { k } , z _ { p } ) ) , \forall k \in [ K ] \} . } \end{array}
$$

By utilizing the label description vectors $\{ d _ { x } ^ { k } \}$ , optimizing $\mathcal { L } _ { \mathrm { { H M } } } ( \bar { x } )$ effectively sharpens the model’s decision boundary, reducing the risk of confusion between similar classes and improving overall performance in few-shot learning scenarios. The loss penalizes the model more heavily for misclassifications involving these hard samples, ensuring that the model pays particular attention to the most difficult cases, thereby enhancing its discriminative power.

Mutual Information Loss. The Mutual Information (MI) Loss is designed to maximize the mutual information between the input sample’s hidden representation $z _ { x }$ of $\scriptstyle { \mathbf { { \vec { x } } } }$ and its corresponding retrieved descriptions, promoting a more informative alignment between them. Let $\scriptstyle { d _ { n } }$ be a hidden vector of other label descriptions than $\scriptstyle { \mathbf { { \vec { x } } } }$ . According to van den Oord, Li, and Vinyals (2018), the Mutual Information $M I ( x )$ between the input embedding $z _ { x }$ and its corresponding label description follows the following inequation:

$$
M I \geq \log B + \mathrm { I n f o N C E } ( \{ x _ { i } \} _ { i = 1 } ^ { B } ; h ) ,
$$

where we have defined:

$$
\begin{array} { r l } { \mathrm { I n f o N C E } ( \{ x _ { i } \} _ { i = 1 } ^ { B } ; h ) } & { { } = } \\ { \displaystyle \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \sum _ { k = 1 } ^ { K } h ( z _ { i } , d _ { i } ^ { k } ) } { \sum _ { j = 1 } ^ { B } \sum _ { k = 1 } ^ { K } h ( z _ { j } , d _ { j } ^ { k } ) } , } \end{array}
$$

where $\begin{array} { r } { h ( z _ { j } , d _ { j } ^ { k } ) = \exp \left( \frac { z _ { j } ^ { T } W d _ { j } ^ { k } } { \tau } \right) } \end{array}$ Here, $\tau$ is the temperature, $B$ is mini-batch size and $W$ is a trainable parameter.

![](images/1099b45807622fbad4272ff96b31d60c5de42566ef9e5e6ab0fed8006a3adac0.jpg)  
Figure 3: Our Framework.

Finally, the MI loss function in our implementation is:

$$
\begin{array} { r l } {  { \mathcal { L } _ { \mathrm { M I } } ( x ) = } } \\ & { - \log \frac { \sum _ { k = 1 } ^ { K } h ( z _ { x } , d _ { x } ^ { k } ) } { \sum _ { k = 1 } ^ { K } h ( z _ { x } , { d } _ { x } ^ { k } ) + \sum _ { n \in N ( x ) } \sum _ { k = 1 } ^ { K } h ( z _ { x } , { d } _ { n } ^ { k } ) } } \end{array}
$$

This loss ensures that the representation of the input sample is strongly associated with its corresponding label, while reducing its association with incorrect labels, thereby enhancing the discriminative power of the model.

Joint Training Objective Function. Our model is trained using a combination of the Sample-based learning loss mentioned in Section 2.3 and our description-pivot loss $\mathcal { L } _ { \mathrm { D e s } }$ , weighted by their respective coefficients:

$$
\begin{array} { r l } & { \mathcal { L } ( x ) = \mathcal { L } _ { \mathrm { S a m p } } + \mathcal { L } _ { \mathrm { D e s } } } \\ & { \qquad = \beta _ { \mathrm { S C } } \cdot \mathcal { L } _ { \mathrm { S C } } ( x ) + \beta _ { \mathrm { S T } } \cdot \mathcal { L } _ { \mathrm { S T } } ( x ) } \\ & { \qquad + \beta _ { \mathrm { H M } } \cdot \mathcal { L } _ { \mathrm { H M } } ( x ) + \beta _ { \mathrm { M I } } \cdot \mathcal { L } _ { \mathrm { M I } } ( x ) , } \end{array}
$$

where $\beta _ { \mathrm { S C } } , ~ \beta _ { \mathrm { S T } } , ~ \beta _ { \mathrm { H M } }$ , and $\beta _ { \mathrm { M I } }$ are hyperparameters. This joint objective enables the model to leverage the strengths of each individual loss, facilitating robust and effective learning in Few-Shot Continual Relation Extraction tasks.

Training Procedure. Algorithm 1 outlines the end-to-end training process at each task $\mathcal { T } ^ { j }$ , with $\Phi _ { j - 1 }$ denoting the model after training on the previous $j - 1$ tasks. In line with memory-based continual learning methods, we maintain a memory buffer $\tilde { M } _ { j - 1 }$ that stores a few representative samples from all previous tasks $\mathcal { T } ^ { 1 } , \dotsc , \mathcal { T } ^ { j - 1 }$ . , j−1, along with a relation description set $\tilde { E } _ { j - 1 }$ that holds the descriptions of all previously encountered relations.

FewRel (10-way–5-shot)

<html><body><table><tr><td>Method</td><td>T1</td><td>T2</td><td>T3</td><td>T4</td><td>T5</td><td>T6</td><td>T7</td><td>T8</td><td>△↓</td></tr><tr><td>RP-CRE</td><td>93.97±0.64</td><td>76.05±2.36</td><td>71.36±2.83</td><td>69.32±3.98</td><td>64.95±3.09</td><td>61.99±2.09</td><td>60.59±1.87</td><td>59.57±1.13</td><td>34.40</td></tr><tr><td>CRL</td><td>94.68±0.33</td><td>80.73±2.91</td><td>73.82±2.77</td><td>70.26±3.18</td><td>66.62±2.74</td><td>63.28±2.49</td><td>60.96±2.63</td><td>59.27±1.32</td><td>35.41</td></tr><tr><td>CRECL</td><td>93.93±0.22</td><td>82.55±6.95</td><td>74.13±3.59</td><td>69.33±3.87</td><td>66.51±4.05</td><td>64.60±1.92</td><td>62.97±1.46</td><td>59.99±0.65</td><td>33.94</td></tr><tr><td>ERDA</td><td>92.43±0.32</td><td>64.52±2.11</td><td>50.31±3.32</td><td>44.92±3.77</td><td>39.75±3.34</td><td>36.36±3.12</td><td>34.34±1.83</td><td>31.96±1.91</td><td>60.47</td></tr><tr><td>SCKD</td><td>94.77±0.35</td><td>82.83±2.61</td><td>76.21±1.61</td><td>72.19±1.33</td><td>70.61±2.24</td><td>67.15±1.96</td><td>64.86±1.35</td><td>62.98±0.88</td><td>31.79</td></tr><tr><td>ConPL**</td><td>95.18±0.73</td><td>79.63±1.27</td><td>74.54±1.13</td><td>71.27±0.85</td><td>68.35±0.86</td><td>63.86±2.03</td><td>64.74±1.39</td><td>62.46±1.54</td><td>32.72</td></tr><tr><td>CPL</td><td>94.87</td><td>85.14</td><td>78.80</td><td>75.10</td><td>72.57</td><td>69.57</td><td>66.85</td><td>64.50</td><td>30.37</td></tr><tr><td></td><td>CPL + MI 94.69±0.7</td><td>85.58±1.88</td><td>80.12±2.45</td><td>75.71±2.28</td><td>73.90±1.8</td><td>70.72±0.91</td><td>68.42±1.77</td><td>66.27±1.58</td><td>28.42</td></tr><tr><td>DCRE</td><td>94.93±0.39</td><td>85.14±2.27</td><td>79.06±1.68</td><td>75.92±2.03</td><td>74.10±2.53</td><td>71.83±2.17</td><td>69.84±1.48</td><td>68.24±0.79</td><td>26.69</td></tr><tr><td colspan="10">TACRED (5-way-5-shot)</td></tr><tr><td>Method</td><td>T1</td><td>T2</td><td>T3</td><td>T</td><td>T5</td><td>T6</td><td>T</td><td>T8</td><td>△√</td></tr><tr><td>RP-CRE</td><td>87.32±1.76</td><td>74.90±6.13</td><td>67.88±4.31</td><td>60.02±5.37</td><td>53.26±4.67</td><td>50.72±7.62</td><td>46.21±5.29</td><td>44.48±3.74</td><td>42.84</td></tr><tr><td>CRL</td><td>88.32±1.26</td><td>76.30±7.48</td><td>69.76±5.89</td><td>61.93±2.55</td><td>54.68±3.12</td><td>50.92±4.45</td><td>47.00±3.78</td><td>44.27±2.51</td><td>44.05</td></tr><tr><td>CRECL</td><td>87.09±2.50</td><td>78.09±5.74</td><td>61.93±4.89</td><td>55.60±5.78</td><td>53.42±2.99</td><td>51.91±2.95</td><td>47.55±3.38</td><td>45.53±1.96</td><td>41.56</td></tr><tr><td>ERDA</td><td>81.88±1.97</td><td>53.68±6.31</td><td>40.36±3.35</td><td>36.17±3.65</td><td>30.14±3.96</td><td>22.61±3.13</td><td>22.29±1.32</td><td>19.42±2.31</td><td>62.46</td></tr><tr><td>SCKD</td><td>88.42±0.83</td><td>79.35±4.13</td><td>70.61±3.16</td><td>66.78±4.29</td><td>60.47±3.05</td><td>58.05±3.84</td><td>54.41±3.47</td><td>52.11±3.15</td><td>36.31</td></tr><tr><td>ConPL**</td><td>88.77±0.84</td><td>69.64±1.93</td><td>57.50±2.48</td><td>52.15±1.59</td><td>58.19±2.31</td><td>55.01±3.12</td><td>52.88±3.66</td><td>50.97±3.41</td><td>37.80</td></tr><tr><td>CPL</td><td>86.27</td><td>81.55</td><td>73.52</td><td>68.96</td><td>63.96</td><td>62.66</td><td>59.96</td><td>57.39</td><td>28.88</td></tr><tr><td>CPL +MI</td><td>85.67±0.8</td><td>82.54±2.98</td><td>75.12±3.67</td><td>70.65±2.75</td><td>66.79±2.18</td><td>65.17±2.48</td><td>61.25±1.52</td><td>59.48±3.53</td><td>26.19</td></tr><tr><td>DCRE</td><td>86.20±1.35</td><td>83.18±8.04</td><td>80.65±3.06</td><td>75.05±3.07</td><td>68.83±5.05</td><td>68.30±4.28</td><td>65.30±2.74</td><td>63.21±2.39</td><td>22.99</td></tr></table></body></html>

Table 1: Accuracy $( \% )$ of methods using BERT-based backbone after training for each task. The best results are in bold. \*\*Results of ConPL are reproduced

1. Initialization (Line 1–2): The model for the current task, $\Phi _ { j }$ , is initialized with the parameters of $\Phi _ { j - 1 }$ . We update the relation description set $\tilde { E } _ { j }$ by incorporating new relation descriptions from $E _ { j }$ .

2. Training on the Current Task (Line 3): We train $\Phi _ { j }$ on $D _ { j }$ to learn the novel relations introduced in in $\mathcal { T } ^ { j }$ .

3. Memory Update (Lines 4–8): We select $L$ representative samples from $D _ { j }$ for each relation $r \in R _ { j }$ . These are the $L$ samples whose latent representations are closest to the 1-means centroid of all class samples. These samples constitute the memory $M _ { r }$ , leading to an updated overall memory $\tilde { M } _ { j } = \tilde { M } _ { j - 1 } \cup M _ { j }$ and an updated relation set $\tilde { R } _ { j } = \tilde { R } _ { j - 1 } \cup R _ { j }$ .

4. Prototype Storing (Line 9): A prototype set $\tilde { P } _ { j }$ is generated based on the updated memory $\tilde { M _ { j } }$ . We generate a prototype set $\tilde { P } _ { j }$ based on the updated memory $\tilde { M _ { j } }$ .

5. Memory Training (Line 10): We refine $\Phi _ { j }$ by training on the augmented memory set $\tilde { M } _ { j } ^ { * }$ , ensuring that the model preserves knowledge of relations from previous tasks.

Algorithm 1: Training procedure at each task $\mathcal { T } ^ { j }$   
Input: $\Phi _ { j - 1 } , \tilde { R } _ { j - 1 } , \tilde { M } _ { j - 1 } , \tilde { K } _ { j - 1 } , D _ { j } , R _ { j } , K _ { j } .$   
Output: $\bar { \Phi } _ { j } , \tilde { M _ { j } } , \tilde { K } _ { j } , \bar { \tilde { P } _ { j } }$ .   
1: Initialize $\Phi _ { j }$ from $\Phi _ { j - 1 }$   
2: $\tilde { K } _ { j } \gets \tilde { K } _ { j - 1 } \cup K _ { j }$   
3: Update $\Phi _ { j }$ by $\mathrm { ~ L ~ }$ on $D _ { j }$ (train on current task) 4: $\tilde { M } _ { j } \gets \tilde { M } _ { j - 1 }$   
5: for each $r \in R _ { j }$ do   
6: pick $L$ samples in $D _ { j }$ and add them into $\tilde { M _ { j } }$ 7: end for   
8: $\tilde { R } _ { j } \gets \tilde { R } _ { j - 1 } \cup R _ { j }$   
9: Update $\tilde { P } _ { j }$ with new data in $D _ { j }$ (for inference) 10: Update $\Phi _ { j }$ by $\mathcal { L }$ on $\tilde { M _ { j } }$ and $D _ { j } ^ { * }$ (train on memory)

# 3.3 Descriptive Retrieval Inference

Traditional methods such as Nearest Class Mean (NCM) (Ma et al. 2024) predict relations by selecting the class whose prototype has the smallest distance to the test sample $x$ . While effective, this approach relies solely on distance metrics, which may not fully capture the nuanced relationships between a sample and the broader context provided by class descriptions.

Rather than merely seeking the closest prototype, we aim to retrieve the class description that best aligns with the input, thereby leveraging the inherent semantic meaning of the label. To achieve this, we introduce Descriptive Retrieval Inference (DRI), a retrieval mechanism fusing two distinct reciprocal ranking scores. This approach not only considers the proximity of a sample to class prototypes but also incorporates cosine similarity measures between the sample’s hidden representation $z$ and relation descriptions generated by an LLM. This dual focus on both spatial and semantic alignment ensures that the final prediction is informed by a richer, more robust understanding of the relations.

FewRel (10-way–5-shot)   

<html><body><table><tr><td>Method T1</td><td>T2</td><td></td><td>T3</td><td>T4</td><td>T5</td><td>T6</td><td>T7</td><td>T8</td><td>△</td></tr><tr><td>CPL</td><td>97.25±0.30</td><td>89.29±2.51</td><td>85.56±1.21</td><td>82.10±2.02</td><td>79.96±2.72</td><td>78.41±3.22</td><td>76.42 ±2.25</td><td>75.20±2.33</td><td>22.05</td></tr><tr><td>DCRE</td><td>96.92±0.16</td><td>88.95±1.72</td><td>87.12±1.52</td><td>85.44±1.91</td><td>84.89±2.12</td><td>83.52±1.46</td><td>81.64±0.69</td><td>80.34±0.55</td><td>16.58</td></tr></table></body></html>

TACRED (5-way-5-shot)   

<html><body><table><tr><td>Method</td><td>TI</td><td>T2</td><td>T3</td><td>T4</td><td>T5</td><td>T6</td><td>T7</td><td>T8</td><td>△↓</td></tr><tr><td>CPL</td><td>88.74±0.44</td><td>85.16±5.38</td><td>78.35±4.46</td><td>77.50±4.04</td><td>76.01±5.04</td><td>76.30±4.41</td><td>74.51±5.06</td><td>73.83±4.91</td><td>14.91</td></tr><tr><td>DCRE</td><td>89.06±0.59</td><td>87.41±5.54</td><td>84.91±3.38</td><td>84.18±2.44</td><td>82.74±3.64</td><td>81.92±2.33</td><td>79.34±2.89</td><td>79.10±2.37</td><td>9.96</td></tr></table></body></html>

Table 2: Accuracy $( \% )$ of methods using LLM2Vec-based backbone after training for each task. The best results are in bold.

Given a sample $x$ with hidden representation $z$ and a set of relation prototypes $\{ { \pmb p } _ { r } \} _ { r = 1 } ^ { n }$ , the inference process begins by calculating the negative Euclidean distance between $z$ and each prototype ${ \pmb p } _ { r }$ :

$$
\begin{array} { l } { \displaystyle \mathbf { E } ( x , r ) = - \| z - { \pmb { p } } _ { r } \| _ { 2 } , } \\ { \displaystyle { \pmb { p } } _ { r } = \cfrac { 1 } { L } \sum _ { i = 1 } ^ { L } z _ { i } , } \end{array}
$$

where $L$ is the memory size per relation. Simultaneously, we compute the cosine similarity between the hidden representation and each relation description prototype, $\gamma ( z , d _ { r } )$ . These two scores are combined into DRI score of sample $\scriptstyle { \mathbf { { \vec { x } } } }$ w.r.t relation $r$ for inference, ensuring that predictions align with both label prototypes and relation descriptions:

$$
\mathrm { D R I } ( \boldsymbol { x } , \boldsymbol { r } ) = \frac { \alpha } { \epsilon + \mathrm { r a n k } ( \mathbf { E } ( \boldsymbol { x } , \boldsymbol { r } ) ) } + \frac { 1 - \alpha } { \epsilon + \mathrm { r a n k } ( \gamma ( \boldsymbol { z } , d _ { r } ) ) } ,
$$

where $\begin{array} { r } { d _ { r } = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } d _ { r } ^ { i } } \end{array}$ , $\mathrm { r a n k } ( \cdot )$ represents the rank position of the score among all relations. The $\alpha$ hyperparameter balances the contributions of the Euclidean distance-based score and the cosine similarity score in the final ranking for inference, and $\epsilon$ is a hyperparameter that controls the influence of lower-ranked relations in the final prediction. By adjusting $\epsilon$ , we can fine-tune the model’s sensitivity to less prominent relations. Finally, the predicted relation label $y ^ { \ast }$ is predicted as the one corresponding to the highest DRI score:

$$
y _ { x } ^ { * } = \underset { r = 1 , \ldots , n } { \mathrm { a r g m a x } } \mathrm { D R I } ( x , r )
$$

This fusion approach for inference complements the learning paradigm, ensuring consistency and reliability throughout the FCRE process. By effectively balancing the strengths of protoype-based proximity and descriptionbased semantic similarity, it leads to more accurate and robust predictions across sequential tasks.

# 4 Experiments

# 4.1 Settings

We conduct experiments using two pre-trained language models, BERT (Devlin et al. 2019) and LLM2Vec (BehnamGhader et al. 2024), on two widely used benchmark datasets for Relation Extraction: FewRel (Han et al. 2018) and TACRED (Zhang et al. 2017). We benchmark our methods against state-of-the-art baselines: SCKD (Wang, Wang, and Hu 2023), RP-CRE (Cui et al. 2021), CRL (Zhao et al. 2022), CRECL (Hu et al. 2022), ERDA (Qin and Joty 2022), ConPL (Chen, Wu, and Shi 2023b), CPL (Ma et al. 2024), $\mathbf { C P L + M I }$ (Tran et al. 2024c).

# 4.2 Experiment results

Our proposed method yields state-of-the-art accuracy. Table 1 presents the results of our method and the baselines, all using the same pre-trained BERT-based backbone. Our method consistently outperforms all baselines across the board. The performance gap between our method and the strongest baseline, CPL, reaches up to $3 . 7 4 \%$ on FewRel and $5 . 8 2 \%$ on TACRED.

To further validate our model, we tested it on LLM2Vec, which provides stronger representation learning than BERT. As shown in Table 2, our model again surpasses CPL, with accuracy drops of only $1 6 . 5 8 \%$ on FewRel and $9 . 9 6 \%$ on TACRED.

These results highlight the effectiveness of our method in leveraging semantic information from descriptions, which helps mitigate forgetting and overfitting, ultimately leading to significant performance improvements.

Exploiting additional descriptions significantly enhances representation learning. Figure 4 presents t-SNE visualizations of the latent space of relations without (left) and with (right) the use of descriptions during training. The visualizations reveal that incorporating descriptions markedly improves the quality of the model’s representation learning. For instance, the brown-orange and purple-green class pairs, which are closely clustered and prone to misclassification in the left image, are more distinctly separated in the right image. Additionally, Figure 5 illustrates that our strategy, which leverages refined descriptions, captures more semantic knowledge related to the labels than the approach using raw descriptions. This advantage bridges the gap imposed by the challenges of few-shot continual learning scenarios, leading to superior performance. Figure 6 shows the perfomance of our model on TACRED as the number of generated expert descriptions per training varies. The results indicate that the model performance generally improves from $K = 3$ and peaks at $K = 7$ .

![](images/0c2fe4eb363b0691b52045c55b8938361890c6b05b0eca5467eb86c89fa5b933.jpg)  
Figure $4 { : } \operatorname { t - S N E }$ visualization of the representations of 6 relations post-training, with and without descriptions, using our retrieval strategy.

![](images/35a8b27983186f56cdf24242e4b6fa59ac8c03c1c1ed4054d5047dfbebd98982.jpg)  
Figure 5: The impact of refined descriptions generated by LLMs. The green, orange, and blue bars show respectively the final accuracies of DCRE when using refined descriptions, original descriptions, and without using descriptions.

Our retrieval-based prediction strategy notably enhances model performance. Table 3 demonstrates that by leveraging the rich information from generated descriptions, our proposed strategy improves the model’s performance by up to $1 . 3 1 \%$ on FewRel and $6 . 6 6 \%$ on TACRED compared to traditional NCM-based classification. The harmonious integration of NCM-based prototype proximity and description-based semantic similarity enables our strategy to deliver more accurate and robust predictions across sequential tasks.

# 4.3 Ablation study

Table 4 present evaluation results that closely examine the role of each component in the objective function during training. The findings underscore the critical importance of ${ \mathcal { L } } _ { \mathrm { M I } }$ and ${ \mathcal { L } } _ { \mathrm { H M } }$ , both of which leverage instructive descriptions from LLMs, aided by Raw descriptions. Because when we ablate one of them, the final accuracy can be reduced by $6 \%$ on the BERT-based model, and $10 \%$ on the LLM2VECbased model.

Table 3: DRI and NCM prediction.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="2">FewRel</td><td colspan="2">TACRED</td></tr><tr><td colspan="2">BERT LLM2Vec</td><td colspan="2">BERT LLM2Vec</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>NCM</td><td>66.93</td><td>79.26</td><td>58.26</td><td>75.00</td></tr><tr><td>DRI (Ours)</td><td>68.24</td><td>80.34</td><td>63.21</td><td>79.10</td></tr></table></body></html>

![](images/7f6571ab53f6e4a46e425a9c528d98a49d47bf92ce4ca4d722bd5a4d0e359cce.jpg)  
Figure 6: Model performance when varying K, on TACRED 5-way 5-shot.

Table 4: Ablation study.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="2">BERT</td><td colspan="2">LLM2Vec</td></tr><tr><td>FewRel TACRED</td><td></td><td>FewRel TACRED</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DCRE (Our)</td><td>68.24</td><td>63.21</td><td>80.34</td><td>79.10</td></tr><tr><td>w/o Lsc</td><td>67.58</td><td>62.11</td><td>78.39</td><td>77.01</td></tr><tr><td>w/o LMI</td><td>65.10</td><td>57.23</td><td>70.61</td><td>74.17</td></tr><tr><td>w/o LHM</td><td>66.20</td><td>62.46</td><td>77.22</td><td>74.75</td></tr><tr><td>w/o LsT</td><td>67.54</td><td>59.56</td><td>77.48</td><td>73.77</td></tr></table></body></html>

# 5 Conclusion

In this work, we propose a novel retrieval-based approach to address the challenging problem of Few-shot Continual Relation Extraction. By leveraging large language models to generate rich relation descriptions, our bi-encoder training paradigm enhances both sample and class representations and also enables a robust retrieval-based prediction method that maintains performance across sequential tasks. Extensive experiments demonstrate the effectiveness of our approach in advancing the state-of-the-art and overcoming the limitations of traditional memory-based techniques, underscoring the potential of language models and retrieval techniques for dynamic real-world relationship identification.