# Contradicted in Reliable, Replicated in Unreliable: Dual-Source Reference for Fake News Early Detection

Yifan Feng1, Weimin $\mathbf { L i } ^ { * 1 }$ , Yue $\mathbf { W a n g } ^ { 1 }$ , Jingchao Wang1, Fangfang ${ { \bf { L i u } } ^ { 1 } }$ , Zhongming $\mathbf { H a n } ^ { 2 }$

1School of Computer Engineering and Science, Shanghai University, Shanghai 200444, China 2School of Computer and Artificial Intelligence, Beijing Technology and Business University, Beijing 100048, China {fengyffyf, wmli, wyofficial, static, ffliu}@shu.edu.cn, hanzhongming $@$ btbu.edu.cn

# Abstract

Early detection of fake news is crucial to mitigate its negative impact. Current research in fake news detection often utilizes the difference between real and fake news regarding the support degree from reliable sources. However, it has overlooked their different semantic outlier degrees among unreliable source information during the same period. Since fake news often serves idea propaganda, unreliable sources usually publish a lot of information with the same propaganda idea during the same period, making it less likely to be a semantic outlier. To leverage this difference, we propose the Reliable-Unreliable Source Reference (RUSR) Fake News Early Detection Method. RUSR introduces the publication background for detected news, which consists of related news with common main objects of description and slightly earlier publication from both reliable and unreliable sources. Furthermore, we develop a strongly preference-driven support degree evaluation model and a two-hop semantic outlier degree evaluation model, which respectively mitigate the interference of news with weak validation effectiveness and the tightness degree of semantic cluster. The designed redistribution module and expanding range relative time encoding are adopted by both models, respectively optimizing early checkpoint of training and expressing the relevance of news implied by their release time gap. Finally, we present a multi-model mutual benefit and collaboration framework that enables the multi-model mutual benefit of generalization in training and multi-perspective prediction of news authenticity in inference. Experiments on our newly constructed dataset demonstrate the superiority of RUSR.

# Introduction

Fake news refers to entirely false news reports (Rastogi and Bansal 2023). These reports influence individuals’ perceptions of social (Wu et al. 2023), health (Silva et al. 2021), and other issues. As fake news spreads, it can even weaken social stability and national security (Yin et al. 2024). Early detection involves identifying the truthfulness of news at the beginning of its dissemination (Liu and Wu 2020). Thus, automating the early detection of fake news is of significant practical importance.

Fake news often imitates the content patterns of real news to avoid detection by humans and machines (Hu et al. 2021),

UnreliableSource Fake news: President Joe Release Background Biden and Democrats “send a fortune to Ukraine but   
Biden's job approval rating on nothing for our children."   
election day next week may   
thot Imok mt $45 \%$ fterent xit anlls Semantic Cluster   
recordedforthen-President   
Donald Trump in 2018.. Democrats in Floridaare ... Polls suggest that the worried that two decades of contest is slipping away from narrow defeats have soured Biden's Democrats in favour of donors on Florida for the a Republican party. foreseeable future...

making content-only detection methods (Wang et al. 2018; Ma et al. 2016) less effective. To address this problem, some methods incorporate reliable source information as external evidence and improve accuracy by leveraging the difference between real and fake news regarding the support degree from reliable sources (Popat et al. 2018; Vo and Lee 2021; Hu et al. 2021; Zhang et al. 2024). However, current methods ignore their different semantic outlier degrees among unreliable source information during the same period.

Promoting a specific viewpoint for an interest group is a primary motivation for publishing fake news (Khan et al. 2022). These news may shape political views (Khan et al. 2022), propagandize ideologies (Baptista and Gradim 2021), or change consumer perceptions of products (Domenico et al. 2021). When fake news is published to promote an opinion for an interest group, the fabricator often manipulates unreliable sources to simultaneously disseminate other information supporting that opinion. Therefore, fake news typically has greater semantic cluster among unreliable source information during the same period than real news.

Figure 1 illustrates an example of fake news (2022/11/2) published before the US 2022 mid-term elections and its semantic cluster. This fake news claims that Democrats prioritize foreign aid over the needs of American children. Among its unreliable sources publication background, three news items also tends to support that Democrats are unfit to govern, forming a semantic cluster with the fake news. These news highlight the dissatisfaction with Biden’s job performance, the decline of Democratic approval rating and the longstanding losses in Florida of the party.

To leverage this difference, we propose ReliableUnreliable Source Reference (RUSR) Fake News Early Detection Method, which constructs the proposed publication background of detected news by extracting related news with common main objects of description and slightly earlier publication from both reliable and unreliable sources. Based on this, a support degree evaluation model and a semantic outlier degree evaluation model are developed. They are less disturbed by background news with weak validation effectiveness from reliable sources and by the semantic cluster tightness of detected news within its unreliable source publication background, respectively. Besides, they use a new relative time encoding method, whose encoding reflects the relevance of news based on the release time gap, and employ an innovative redistribution module to optimize early checkpoint of training. Additionally, we present a multimodel mutual benefit and collaboration framework based on the above two models and a content authenticity evaluation model. The framework achieves the multi-model mutual benefit of generalization in training and multi-perspective prediction of news authenticity in inference.

Our major contributions are summarized as follows:

• We propose to apply the different semantic outlier degrees between real and fake news among unreliable source information during the same period to fake news detection.   
• We propose RUSR, which analyzes the support degree and semantic outlier degree of detected news based on proposed publication background. Besides, the designed redistribution module and relative time encoding respectively optimize early checkpoint of training and imply news relevance. Finally, the proposed framework achieves multi-model mutual benefit and multiperspective prediction.   
• We propose to construct the dataset comprising verified news and news tagged with source reliability, both published within the same time period. Experiments demonstrate the superiority of our proposal.

# Problem Definition

A piece of news is represented as $\boldsymbol { x } = ( \mathrm { t e x t } ( \boldsymbol { x } ) , t _ { x } )$ , where $\operatorname { t e x t } ( x )$ and $t _ { x }$ denote the text and publication time of $x$ , respectively. $y _ { x }$ represents the authenticity label of $x$ , with 0 indicating fake news and 1 indicating real news. The total labeled news, news from reliable sources, and news from unreliable sources collected before $t$ are denoted by $\mathcal { L } _ { t } , \mathcal { R } _ { t }$ and ${ { \mathcal { U } } _ { t } }$ , respectively. For any given $\mathrm { \Delta } l \in \mathcal { L } _ { t }$ , there is a large amount of news released shortly before $t _ { l }$ in both $\mathcal { R } _ { t }$ and ${ { \mathcal { U } } _ { t } }$ . Our goal is to develop a fake news detection function using $\mathcal { L } _ { t _ { \mathrm { t e s t } } }$ , $\mathcal { R } _ { t _ { \mathrm { t e s t } } }$ , and $\mathcal { U } _ { t _ { \mathrm { t e s t } } }$ , ensuring good early detection performance for news published at $t _ { \mathrm { t e s t } }$ and thereafter.

# Proposed Method

In this chapter, we introduce the proposed RUSR. First, we describe the framework of RUSR, followed by detailed explanations of each component.

# Multi-Model Mutual Benefit and Collaboration Framework

The RUSR includes three models: the strongly preferencedriven support degree evaluation model $\mathbf { M o d e l } _ { r }$ , the twohop semantic outlier degree evaluation model $\mathbf { M o d e l } _ { u }$ , and the content authenticity evaluation model $\mathbf { M o d e l } _ { c }$ . During training, they share the bottom module to form a joint model (JM), enabling multi-model joint learning. During inference, the multi-perspective authenticity scores output by three models are integrated to calculate the final prediction.

Multi-Model Mutual Benefit Training The training set includes all news published before $t _ { \mathrm { v a l } }$ in $\mathcal { L } _ { t _ { \mathrm { t e s t } } }$ . $t _ { \mathrm { v a l } }$ is earlier than $t _ { \mathrm { t e s t } }$ . The validation set includes the other part of $\mathcal { L } _ { t _ { \mathrm { t e s t } } }$ . The architecture of JM is shown in Figure 2 and can be expressed as:

$$
\begin{array} { r } { ( v _ { x } ^ { \mathrm { c } } , v _ { x } ^ { \mathrm { r } } , v _ { x } ^ { \mathrm { u } } ) = \mathbf { J } \mathbf { M } ( \mathrm { i n } ( x ) ; \mathbf { \Theta } ) , } \end{array}
$$

where $x$ represents any news, $\Theta$ denotes the set of all learnable parameters in JM, $\mathrm { i n } ( x )$ is $( x , B _ { x } ^ { \mathrm { r } } , B _ { x } ^ { \mathrm { u } } )$ . The publication background of $x$ is denoted as $\boldsymbol { B } _ { x }$ , and the news from reliable and unreliable sources in it constitute $B _ { x } ^ { r }$ and $B _ { x } ^ { u }$ respectively. The output is composed of three authenticity scores, which represent the content authenticity of $x$ , the support degree of $x$ from $B _ { x } ^ { r }$ , and the semantic outlier degree of $x$ in $B _ { x } ^ { u }$ respectively. $v _ { x } ^ { h } \in ( 0 . 5 , 1 ]$ , $v _ { x } ^ { h } \in [ 0 , 0 . 5 )$ , and $v _ { x } ^ { h } = 0 . 5$ indicate real, fake, and uncertain news, where $h \in \mathrm { c } , \mathbf { r } , \mathbf { v }$ .

JM consists of the news time-content encoding module $\mathbf { M } _ { \mathrm { e } }$ , the content authenticity evaluation module $\mathbf { M } _ { \mathrm { c } }$ , the support degree evaluation module $\mathbf { M } _ { \mathrm { r } }$ , and the outlier degree evaluation module $\mathbf { M } _ { \mathbf { u } }$ , which are represented as follows:

$$
\begin{array} { r l } & { ( e _ { x } , E _ { x } ^ { \mathrm { r } } , E _ { x } ^ { \mathrm { u } } ) = \mathsf { M } _ { \mathrm { e } } ( \mathrm { i n } ( x ) ; \Theta ^ { \mathrm { e } } ) , } \\ & { \quad \quad \quad \quad v _ { x } ^ { \mathrm { c } } = \mathsf { M } _ { \mathrm { c } } ( e _ { x } ; \Theta ^ { \mathrm { c } } ) , } \\ & { \quad \quad \quad v _ { x } ^ { \mathrm { r } } = \mathsf { M } _ { \mathrm { r } } ( e _ { x } , E _ { x } ^ { \mathrm { r } } ; \Theta ^ { \mathrm { r } } ) , } \\ & { \quad \quad \quad v _ { x } ^ { \mathrm { u } } = \mathsf { M } _ { \mathrm { u } } ( e _ { x } , E _ { x } ^ { \mathrm { u } } ; \Theta ^ { \mathrm { u } } ) , } \end{array}
$$

where $\boldsymbol { e } _ { x } \in \mathbb { R } ^ { d i m }$ , $E _ { x } ^ { \mathrm { r } } \in \mathbb { R } ^ { | \mathcal { B } _ { x } ^ { \mathrm { r } } | \times d i m }$ , and $E _ { x } ^ { \mathrm { u } } \in \mathbb { R } ^ { | \mathcal { B } _ { x } ^ { \mathrm { u } } | \times d i m }$ are the time-content features of $x$ and the matrices composed of time-content features of news in $B _ { x } ^ { \mathrm { r } }$ and $B _ { x } ^ { \mathrm { u } }$ , respectively. $d i m = 2 5 6 . \Theta ^ { \mathrm { e } } , \Theta ^ { \mathrm { c } } , \Theta ^ { \mathrm { r } } , \Theta ^ { \mathrm { u } }$ are sets of modules’ learnable parameters.

The sub-models are defined as follows:

$$
\begin{array} { r } { \mathrm { M o d e l } _ { h } ( \mathrm { i n } ( x ) ; \Theta ^ { h } , \Theta ^ { \mathrm { e } } ) = v _ { x } ^ { h } , } \end{array}
$$

where $\mathbf { M o d e l } _ { h }$ contains $\mathbf { M } _ { \mathrm { e } }$ and $\mathbf { M } _ { \mathrm { h } }$ .

For any training news $l$ , the loss function $L ( v _ { l } ^ { h } , y _ { l } )$ on single output $v _ { l } ^ { h }$ is defined as follows:

$$
L ( \cdot , \cdot ) = \operatorname* { m a x } \left\{ - \ln \left( v _ { l } ^ { h } \times \left( - 1 \right) ^ { \left( y _ { l } + 1 \right) } + 1 - y _ { l } + \varepsilon \right) , 0 \right\} ,
$$

where $\varepsilon$ is a very small positive number. The max function is used to prevent negative loss. The loss on $l$ is the sum of $L ( v _ { l } ^ { c } , y _ { l } ) , \mathbf { \bar { Z } } ( v _ { l } ^ { r } , y _ { l } )$ and $L ( v _ { l } ^ { u } , y _ { l } )$ .

![](images/794f7f4bdb23d07123a4b74ce84f61d13cb948dc4875405c4fb6a8c813434bf9.jpg)  
Figure 2: The overall architecture of the proposed JM model.

After the $k$ -th, $k \geq 0$ training epoch, the parameter set of JM is denoted as $\boldsymbol { \Theta } _ { k } = \boldsymbol { \Theta } _ { k } ^ { \mathrm { e } } \cup \bar { \boldsymbol { \Theta } } _ { k } ^ { \mathrm { c } } \cup \boldsymbol { \Theta } _ { k } ^ { \mathrm { r } } \cup \boldsymbol { \Theta } _ { k } ^ { \mathrm { u } }$ , and the accuracies of the three outputs on the validation set are denoted as $a c c _ { k } ^ { \mathrm { c } } , a c c _ { k } ^ { \mathrm { r } } , a c c _ { k } ^ { \mathrm { u } 1 }$ .

Multi-Model Collaboration Inference Let $a c c _ { k } ^ { h }$ reach its maximum when $k$ equals $\boldsymbol { k } _ { h }$ . Consequently, when $\Theta ^ { h }$ and $\Theta ^ { \mathrm { e } }$ are $\Theta _ { k _ { h } } ^ { h }$ and $\Theta _ { \boldsymbol { k } _ { h } } ^ { \mathrm { e } }$ , respectively, $\mathbf { M o d e l } _ { h }$ achieves optimal future generalization.

While detecting news $x$ , for each model $\mathbf { M o d e l } _ { h }$ , RUSR calculates Modelh(in(x); Θkh , Θek ). The real, fake or unknown result leads to increasing the score of real or fake class by $\smash { a c c _ { k _ { h } } ^ { h } }$ , or remaining their scores, respectively. After the calculation of three models, the prediction of RUSR is real news if real class score is larger than fake class score. Otherwise, the prediction is fake news.

The significant differences in effective inputs and structures ensure that three models are unlikely to produce the same prediction for one news, ensuring their effective collaboration.

# Publication Background Construction

RUSR selects related news in content and time from $\mathcal { R } _ { t _ { x } }$ and $\mathcal { U } _ { t _ { x } }$ to construct $\boldsymbol { B } _ { x }$

News with Common Main Objects of Description Selecting The content-related news set $\mathcal { C } _ { x }$ for news $x$ is defined as the news in $\mathcal { R } _ { t _ { x } }$ and ${ \mathcal { U } } _ { t _ { x } }$ that share common main objects of description with $x$ . The method to obtain the main objects of description set ${ \mathcal { O } } _ { m }$ for any news $m$ is as follows:

The en core web sm pipeline2 from the spacy package is used to process text $( m )$ . First, the text is converted into a sequence of tokens, where each token is a word or punctuation from the text. Then, each word token is tagged with its part of speech and their base forms are obtained. Finally, the named entities consisting of tokens are identified in text $( m )$ .

For each named entity, the base forms of its tokens are concatenated with space to get the base form of the named entity. After that, we extract one sequence consisting of the base forms of all named entities and another composed of the base forms of noun tokens that are not part of any named entity. We concatenate the two sequences, then analyze the frequency of each unique string in the combined sequence. Ultimately, we select up to five strings with the highest frequency, and form a set with them as ${ \mathcal { O } } _ { m }$ .

Temporally Close News Selecting Due to the varying popular topics, news in ${ \mathcal { C } } _ { x }$ that is temporally closer to $x$ is more likely to be related to $x$ . Therefore, we retain up to $N _ { \mathrm { m a x } } ^ { \mathrm { B } } = 3 0 \$ latest news in ${ \mathcal { C } } _ { x }$ to construct $\boldsymbol { B } _ { x }$ . An exception irsanwdhoemn $x$ iwss fn othme threa lnaitnegs $\boldsymbol { B } _ { x }$ nsteswosf nu .o $\bar { N } _ { \mathrm { m a x } } ^ { \mathrm { B } }$ $1 . 1 \times N _ { \operatorname* { m a x } } ^ { \mathbf { B } }$ ${ \mathcal { C } } _ { x }$ $\boldsymbol { B } _ { x }$ varies in each epoch to enhance generalization.

# News Time-Content Encoding Module

For any $m \in \{ x \} \cup B _ { x }$ , we first generate its low-dimensional content vector $e _ { m } ^ { \mathrm { l o w } }$ . Then, we encode the relative time from $m$ $x$ $e _ { d _ { m  x } } ^ { \mathrm { t i m e } }$ using expanding range relative time encoding. The sum o→f these two vectors gives the timecontent encoding $\scriptstyle { e _ { m } }$ .

Low-Dimensional Content Vector Counting First, the all-mpnet-base- $\cdot \nu 2$ model3 from the sentence-transformers package is used to generate a 768-dimensional highdimensional content vector ehmigh based on text(m). The parameters of this model are not trained and are not in $\Theta ^ { \mathrm { e } }$ .

For generalization, a two-layer MLP is used to reduce the dimensions of $e _ { m } ^ { \mathrm { h i g h } }$ , resulting in $e _ { m } ^ { \mathrm { l o w } }$ . The MLP progressively compresses the dimensions to 512 and dim dimensions. The first layer uses ReLU as the activation function, while the second layer has no activation function.

Expanding Range Relative Time Encoding Calculating support and outlier degree needs to compute the semantic similarity between $x$ and each background news based on their time-content encodings. To take news semantic relevance implied by the release time gap into account when computing similarity, we include the encoding of the release time gap between $m$ and $x$ in the time-content emcoding of $m$ and make the time encoding reflect their semantic relevance implied by the gap.

Let $d _ { m  x }$ be the number of days in the gap, representing the release time gap. We divide all possible values of $d _ { m  x }$ into 22 segments and define a learnable dim-dimensional parameter vector for each segment. The vector corresponding to the segment where $d _ { m  x }$ falls is $e _ { d _ { m  x } } ^ { \mathrm { t i m e } }$ . The segment index $i d x _ { m  x }$ of $d _ { m  x }$ is:

$$
i d x _ { m  x } = \{ \begin{array} { l l } { d _ { m  x } , } & { 0 \leq d _ { m  x } < 7 , } \\ { 6 + \lfloor d _ { m  x } / 7 \rfloor , } & { 7 \leq d _ { m  x } < 2 8 , } \\ { 9 + \lfloor d _ { m  x } / 2 8 \rfloor , } & { 2 8 \leq d _ { m  x } < 2 8 \times 1 2 , } \\ { 2 1 , } & { d _ { m  x } \geq 2 8 \times 1 2 . } \end{array} 
$$

The reason for designing expanding range segments is that we want etime to express the semantic relevance between $x$ and $m$ implied by $d _ { m  x }$ . The relevance is denoted as the temporal relevance between them. Intuitively, the temporal relevance decreases as $d _ { m  x }$ increases, with the rate of decrease being initially fast and then slower. Therefore, for a positive integer $\Delta d$ , the smaller the value of $d _ { m  x }$ , the greater the difference in temporal relevance between $d _ { m  x }$ and $d _ { m  x } + \Delta d$ , and hence, $e _ { d _ { m  x } } ^ { \mathrm { t i m e } }$ and etime dm x+∆d should be less similar.

# Content Authenticity Evaluation Module

$\mathbf { M } _ { \mathrm { c } }$ is defined as:

$$
\begin{array} { r l } & { \pmb { a } _ { x } = \mathrm { L i n e a r \_ L a y e r } \left( \mathrm { R e L U } \left( \pmb { e } _ { x } \right) \right) , } \\ & { \pmb { v } _ { x } ^ { \mathrm { c } } = \frac { e ^ { a _ { x } ^ { 1 } } } { e ^ { a _ { x } ^ { 0 } } + e ^ { a _ { x } ^ { 1 } } } , } \end{array}
$$

where Linear Layer don’t have activation function, and $\mathbf { a } _ { x }$ is a vector with two dimensions.

# Strongly Preference-Driven Support Degree Evaluation Module

$\mathbf { M } _ { \mathrm { r } }$ first calculates the $\mathbf { M } _ { \mathrm { r } }$ redistributed time-content vectors for each time-content vector in $e _ { x }$ and ${ \boldsymbol E } _ { x } ^ { \mathrm { r } }$ through the redistribution module $\mathbf { M } _ { \mathrm { d r } }$ , resulting in $e _ { x } ^ { \mathrm { d r } } \in \mathbb { R } ^ { d i m }$ and $E _ { x } ^ { \mathrm { { d r } } } \in$ R|Brx|×dim, respectively. Mdr consists of centering and rotation. During centering, the element-wise mean $\mathbf { \Sigma } _ { m e a n _ { x } ^ { \mathrm { r } } }$ of the $| B _ { x } ^ { r } | + 1$ vectors from $e _ { x }$ and ${ \pmb E } _ { x } ^ { \mathrm { r } }$ is computed. Then, for any ${ \dot { n } } \in \{ x \} \cup B _ { x } ^ { \mathrm { r } }$ , the time-content encoding of $n$ minus $\mathbf { \omega } _ { m e a n _ { x } ^ { \mathrm { r } } }$ element-wise resulting in $e _ { n } ^ { \mathrm { m i d } }$ . During rotation, the $\mathbf { M } _ { \mathrm { r } }$ redistributed time-content vector of $n$ is represented as:

$$
e _ { n } ^ { \mathrm { { d r } } } = \frac { e _ { n } ^ { \mathrm { { m i d } } } } { \| e _ { n } ^ { \mathrm { { m i d } } } \| _ { 2 } } + b ^ { \mathrm { r } } + \frac { e _ { n } ^ { \mathrm { { m i d } } } } { \| e _ { n } ^ { \mathrm { { m i d } } } \| _ { 2 } } \odot m ^ { \mathrm { r } } .
$$

where $m ^ { \mathrm { r } } , b ^ { \mathrm { r } } \in \mathbb { R } ^ { d i m }$ are learnable parameter vectors, and $\odot$ denotes element-wise multiplication. The role of the redistribution modules in $\mathbf { M } _ { \mathrm { r } }$ and $\mathbf { M } _ { \mathrm { u } }$ will be described in the experiments.

Next, to assess the degree of semantic similarity or opposition between $x$ and each $m \in B _ { x } ^ { \mathrm { r } }$ , the cosine similarity between $e _ { m } ^ { \mathrm { d r } }$ and $e _ { x } ^ { \mathrm { d r } }$ is calculated and denoted as $s i m _ { m , x } ^ { \mathrm { r } }$ . If $| s i m _ { m , x } ^ { \mathrm { r } } | > 0 . 7$ , it is considered a strong preference similarity, indicating $m$ has strong validation effectiveness. Otherwise, they are weak. The mean of the strong and weak preference similarities are mhxigh and simlxo , respectively (if strong or weak preference similarity does not exist, the corresponding mean is 0). $\boldsymbol { v } _ { x } ^ { \mathrm { r } }$ is calculated as follows:

$$
v _ { x } ^ { \mathrm { r } } = ( 0 . 9 \times s i m _ { x } ^ { \mathrm { h i g h } } + 0 . 1 \times s i m _ { x } ^ { \mathrm { l o w } } + 1 ) / 2 .
$$

where 0.1 suppress the interference of weak validation effectiveness news. Additionally, if $| B _ { x } ^ { r } | = 0$ , $x$ is deemed unmanageable for $\mathbf { M } _ { \mathrm { r } }$ . So $\boldsymbol { v } _ { x } ^ { \mathrm { r } }$ is not calculated in $\mathbf { M } _ { \mathrm { r } }$ , and 0.5 is set.

# Two-Hop Semantic Outlier Degree Evaluation Module

$\mathbf { M } _ { \mathrm { u } }$ considers the news in $\mathcal { G } _ { x } = \{ x \} \cup B _ { x } ^ { \mathrm { u } }$ as a whole and calculates the semantic outlier degree of $x$ in $B _ { x } ^ { \mathrm { u } }$ :

$$
v _ { x } ^ { \mathrm { u } } = ( n d i f _ { x } - \frac { \sum _ { n \in \mathcal { G } _ { x } ^ { N } } ( n d i f _ { n } ^ { x } ) } { N } + 2 ) / 4 ,
$$

where $\mathcal { G } _ { x } ^ { N }$ is the set of the $N = 5$ nearest semantic neighbors of $x$ in $B _ { x } ^ { \mathrm { u } }$ , and $n d i f _ { x }$ and $n d i f _ { n } ^ { x }$ are defined as:

$$
\begin{array} { r l } & { n d i f _ { x } = \mathrm { n d i f } _ { x } ^ { N } ( x , { \mathcal G } _ { x } ) , } \\ & { n d i f _ { n } ^ { x } = \mathrm { n d i f } _ { x } ^ { N } ( n , ( B _ { x } ^ { \mathrm { u } } \backslash { \mathcal G } _ { x } ^ { N } ) \cup \{ n \} ) , } \end{array}
$$

where $B _ { x } ^ { \mathrm { u } } \backslash \mathcal { G } _ { x } ^ { N }$ represents the relative complement of $\mathcal { G } _ { x } ^ { N }$ in $B _ { x } ^ { \mathrm { u } }$ . The function ndi $\mathsf { f } _ { x } ^ { N } ( m , S )$ is defined as:

$$
\mathrm { n d i f } _ { x } ^ { N } ( m , { \mathcal { S } } ) = \frac { \sum _ { n \in S _ { m } ^ { N } } \mathrm { d i s } _ { x } ( m , n ) } { N } .
$$

Its domain is $m \in { \mathcal { S } }$ , and ${ \mathcal { S } } \subseteq { \mathcal { G } } _ { x }$ . Here, $S _ { m } ^ { N }$ is the set of the $N$ nearest semantic neighbors of $m$ in $S$ . In $\mathbf { M } _ { \mathrm { u } }$ , the semantic distance function used for calculating the nearest semantic neighbors and semantic distances is $\mathrm { d i s } _ { x } ( m , n )$ , whose domain is $m , n \in { \mathcal { G } } _ { x }$ and range is $[ 0 , 2 ]$ .

To avoid the sensitivity of $v _ { x } ^ { \mathrm { u } }$ to the tightness degree of the semantic cluster of $x$ in $\mathcal { G } _ { x }$ , we use relative outlier degree instead of $n d i f _ { x } / 2$ as $v _ { x } ^ { \mathrm { u } }$ . The tightness degree is based on pairwise semantic distances. To ensure the accessibility of smaller values in $[ 0 , 1 ]$ for $v _ { x } ^ { \mathrm { u } }$ , $\mathcal { G } _ { x } ^ { N }$ is excluded when calculating $n d i f _ { n } ^ { x }$ . Otherwise, when $n d i f _ { x }$ is small, the pairwise distances in $x$ and the news in $\mathcal { G } _ { x } ^ { N }$ are not far generally, making it difficult for $n d i f _ { n } ^ { x } , n \in \mathcal { G } _ { x } ^ { \tilde { N } }$ to reach large values.

To construct the function $\mathrm { d i s } _ { x } ( m , n )$ , for any $m \in { \mathcal G } _ { x }$ , calculate the deep time-content feature vector:

$$
\widetilde { e } _ { m } = \mathrm { L i n e a r \mathrm { \_ L a y e r } } \left( \mathrm { R e L U } \left( e _ { m } \right) \right) ,
$$

where $e _ { m }$ is the time-content feature of $m$ , $\tilde { \boldsymbol { e } } _ { m } \in \mathbb { R } ^ { d i m }$ and Linear Layer don’t have activation function. Then, using the redistribution module $\mathbf { M } _ { \mathrm { d u } }$ , we calculate a $\mathbf { M } _ { \mathrm { u } }$ redistributed time-content vector for each deep time-content vector in $\tilde { e } _ { x }$ and $\tilde { \pmb { { E } } } _ { x } ^ { \mathrm { u } } \in \mathbb { R } ^ { | \mathcal { B } _ { x } ^ { \mathrm { u } } | \times d i m }$ , resulting in $e _ { x } ^ { \mathrm { d u } } \in \mathbb { R } ^ { d i m }$ and $E _ { x } ^ { \mathrm { d u } } \in$ R|Bx|×dim. disx(m, n) is defined as:

$$
\begin{array} { r } { \mathrm { d i s } _ { x } ( m , n ) = - \mathrm { c o s } _ { - } \mathrm { s i m } ( e _ { m } ^ { \mathrm { d u } } , e _ { n } ^ { \mathrm { d u } } ) + 1 , } \end{array}
$$

where $e _ { m } ^ { \mathrm { d u } }$ and $e _ { n } ^ { \mathrm { d u } }$ are the $\mathbf { M } _ { \mathrm { u } }$ redistributed time-content vectors of $m$ and $n$ .

If $| B _ { x } ^ { \mathrm { u } } |$ is too small, it is difficult to identify $x$ as real or fake news based on its semantic cluster size in $\mathcal { G } _ { x }$ . Therefore, if $| \mathcal { B } _ { x } ^ { \mathrm { u } } | < M \times N$ , $x$ is deemed unmanageable for $\mathbf { M } _ { \mathrm { u } }$ . So $v _ { x } ^ { \mathrm { u } }$ is not calculated in $\mathbf { M } _ { \mathrm { u } }$ , and 0.5 is set.

# Experiments

# Dataset

Existing fake news detection datasets do not include verified news and news with source reliability label published within the same period. Therefore, we constructed a new English dataset. Besides this innovation, the advantages of our dataset are: 1) The verified news is recent (January 2019 to October 2023), reflecting updated forgery methods. 2) News with source reliability label includes summarized text to highlight key points and avoid token limits of text models. 3) It provides publication dates and main objects of description for all news, facilitating correlation analysis. 4) It offers vectors for the text of all news and the names of main objects of description to facilitate research, with the calculation method in method section. The construction details are as follows:

Verified News We scraped verified news texts and publication dates from fact-checking websites Politifact and Snopes. Since Snopes does not have the latter, we used the publication dates of fact-checking articles. For binary classification tasks and balanced categories, we labeled news marked as true and mostly-true on Politifact, and True and Mostly True on Snopes as real news, and news marked as pants-fire on Politifact, and False and Fake on Snopes as fake news, excluding other labels. Totally, the dataset contains 2,602 real news and 3,253 fake news.

News with Source Reliability Label We collected 33,212 reliable source news and 114,600 unreliable source news published from January 2019 to December 2022 from the nela-gt2019-2022 (Gruppi, Horne, and Adalı 2020, 2021, 2022, 2023) datasets, with reliability labels from nelagt2020. Using the BART model4, we summarized the body text of each news and removed unexpected characters in the summaries, which we provided as the news texts in our dataset.

# Settings

To validate the superiority of RUSR in real-world early detection tasks, we designed four experiments (Q1-Q4) inspired by (Hu et al. 2023) to correspond to four instances of the proposed problem. The four instances set $t _ { \mathrm { t e s t } }$ as the first day of each quarter in 2022. $\mathcal { L } _ { t }$ , $\mathcal { R } _ { t }$ , and ${ { \mathcal { U } } _ { t } }$ are the verified news, reliable source news and unreliable source news published before $t$ in our dataset, respectively. The test set consists of verified news from the corresponding quarter, while the validation and training sets consist of verified news from the preceding quarter and earlier, respectively. Results on the test set are presented.

Baseline Fake news detection baselines include (1) $\mathrm { E A N N _ { T } }$ (Wang et al. 2018), which uses the publication years of the news as auxiliary task labels to produce time-invariant features, upgrading TextCNN to pre-trained BERT and removing the image part as (Hu et al. 2023). (2) Emo (Zhang et al. 2021), which detects news based on the sentiment of the publisher, excluding the sentiment of news comments. (3) BERT $+$ FTT (Hu et al. 2023), focusing on the model’s performance on high-frequency future news topics. (4) DeClarE (Popat et al. 2018) and MAC (Vo and Lee 2021), which detect based on news content and evidence. Due to the absence of news sources and low variability in evidence sources, the use of these sources is removed. (5) Emo+NEP and DeClarE+NEP (Sheng et al. 2022), which additionally consider the popularity and novelty of detected news in its news environment. (1)(2)(3) are content-only methods.

Implementation Details We use the RAdam optimizer with a learning rate of 0.0002 and a batch size of 64. Due to the relatively small dataset, pre-trained models for all methods are not fine-tuned during training. For each detected news, up to 5 reliable source news published before are retrieved as evidence in the baselines using evidence. Methods using NEP obtain the news environment from news with source reliability label.

# Performance Comparison

The experimental results of Q1-Q4 and the average results are reported in Table 1, with the best values in each row highlighted in bold. The experiments indicate that the methods using evidence do not necessarily outperform contentonly methods, and adding NEP does not always enhance performance. This may be because introducing evidence or news environment adds complexity to the model structure while supplementing information for detection. As news evolves rapidly over time, splitting the training, validation, and test news by time results in significant distribution difference between training and test news (Hu et al. 2023). More complex models tend to exhibit greater performance drops from training to testing in the condition. RUSR is the best in all average metrics. This improvement is firstly attributed to the support and outlier degree evaluation models, whose generalization stems from the interpretability of considering their output as authenticity, and benefits from the proposed publication background, redistribution module and time embedding. Additionally, the multi-model mutual benefit and collaboration framework plays a crucial role in the effectiveness of RUSR.

# Ablation Analysis

To validate the three models and framework, we compare the performance of RUSR with three methods of removing one model in Q1-Q4 and present the average performance of sub-model predictions and method predictions across four experiments in Table 2 and 3. Table 3 shows the average macF1 for the sub-models. The results indicate: (1) All sub-models are important, as removing any one results in method performance decline. (2) The generalization of other models decreases when a single model is removed, with the exception that the performance of Modelr remains unchanged when $\mathbf { M o d e l } _ { \mathrm { c } }$ is removed, demonstrating the rationality of the multi-model joint training. (3) The performance of RUSR is superior to any single model, indicating the effectiveness of multi-model collaboration.

<html><body><table><tr><td>2022</td><td>Metric</td><td>Emo</td><td>EANNT</td><td>DeClarE</td><td>MAC</td><td>DeClarE</td><td>EmeP</td><td>BFRT</td><td>RUSR</td></tr><tr><td rowspan="4">Q1</td><td>macF1</td><td>0.703</td><td>0.650</td><td>0.676</td><td>0.693</td><td>0.682</td><td>0.677</td><td>0.682</td><td>0.712</td></tr><tr><td>Acce</td><td>0.714</td><td>0.651</td><td>0.681</td><td>0.703</td><td>0.684</td><td>0.784</td><td>0.644</td><td>0.716</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>F1real</td><td>0.653</td><td>0.633</td><td>0.633</td><td>0.638</td><td>0.650</td><td>0.640</td><td>0.620</td><td>0.680</td></tr><tr><td rowspan="4">Q2</td><td>macF1</td><td>0.696</td><td>0.678</td><td>0.649</td><td>0.673</td><td>0.673</td><td>0.640</td><td>0.674</td><td>0.706</td></tr><tr><td>Acc</td><td>0.699</td><td>0.678</td><td>0.653</td><td>0.674</td><td>0.674</td><td>0.640</td><td>0.674</td><td>0.708</td></tr><tr><td>F1fake</td><td>0.726</td><td>0.675</td><td>0.685</td><td>0.691</td><td>0.691</td><td>0.635</td><td>0.675</td><td>0.729</td></tr><tr><td>F1real</td><td>0.667</td><td>0.681</td><td>0.613</td><td>0.655</td><td>0.655</td><td>0.644</td><td>0.672</td><td>0.682</td></tr><tr><td rowspan="4">Q3</td><td>macF1</td><td>0.700</td><td>0.616</td><td>0.665</td><td>0.621</td><td>0.661</td><td>0.664</td><td>0.666</td><td>0.743</td></tr><tr><td></td><td>0.705</td><td></td><td>0.667</td><td>0.638</td><td>0.662</td><td>0.676</td><td>0.676</td><td></td></tr><tr><td>Acce</td><td></td><td>0.633</td><td></td><td></td><td></td><td></td><td></td><td>0.748</td></tr><tr><td>F1real</td><td>0.663</td><td>0.533</td><td>0.639</td><td>0.542</td><td>0.679</td><td>0.600</td><td>0.609</td><td>0.707</td></tr><tr><td rowspan="4">Q4</td><td>macF1</td><td>0.752</td><td>0.668</td><td>0.697</td><td>0.721</td><td>0.756</td><td>0.696</td><td>0.712</td><td>0.745</td></tr><tr><td>Acc</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>F1fake</td><td>0.758</td><td>0.646</td><td>0.716</td><td>0.733</td><td>0.793</td><td>0.703</td><td>0.720</td><td>0.762</td></tr><tr><td>F1real</td><td>0.714</td><td>0.589</td><td>0.621</td><td>0.633</td><td>0.714</td><td>0.650</td><td>0.663</td><td>0.678</td></tr><tr><td rowspan="4">AVG</td><td>macF1</td><td>0.713</td><td>0.653</td><td>0.672</td><td>0.677</td><td>0.693</td><td>0.669</td><td>0.684</td><td>0.727</td></tr><tr><td>Acc</td><td></td><td></td><td></td><td></td><td>0.616</td><td>0.675</td><td>0.691</td><td></td></tr><tr><td>F1fake</td><td>0.719</td><td>0.667</td><td>0.679</td><td>0.687</td><td></td><td></td><td></td><td>0.74</td></tr><tr><td>F1real</td><td>0.674</td><td>0.609</td><td>0.627</td><td>0.625</td><td>0.675</td><td>0.634</td><td>0.641</td><td>0.687</td></tr></table></body></html>

Table 1: The performance of the baseline methods and RUSR.

Table 2: Ablation studies of RUSR. The maximum value in a column is bolded.   

<html><body><table><tr><td>Method</td><td>macF1</td><td>Acc</td><td>F1fake</td><td>F1real</td></tr><tr><td>w/o Modelc</td><td>0.692</td><td>0.705</td><td>0.752</td><td>0.632</td></tr><tr><td>w/o Modelr w/o Modelu</td><td>0.710</td><td>0.723</td><td>0.768</td><td>0.653</td></tr><tr><td></td><td>0.714</td><td>0.721</td><td>0.754</td><td>0.674</td></tr><tr><td>w/o time</td><td>0.709</td><td>0.720</td><td>0.764</td><td>0.653</td></tr><tr><td>w/o Mdr</td><td>0.693</td><td>0.701</td><td>0.736</td><td>0.650</td></tr><tr><td>w/o Mdu</td><td>0.712</td><td>0.717</td><td>0.744</td><td>0.680</td></tr><tr><td>RUSR</td><td>0.727</td><td>0.734</td><td>0.766</td><td>0.687</td></tr></table></body></html>

Table 3: The impact of single model ablation on other models. The maximum value in a row is bolded.   

<html><body><table><tr><td>Model</td><td>w/o Modelc</td><td>w/o Modelr</td><td>w/o Modelu</td><td>RUSR</td></tr><tr><td>Modelc</td><td>None</td><td>0.710</td><td>0.715</td><td>0.721</td></tr><tr><td>Modelr</td><td>0.690</td><td>None</td><td>0.687</td><td>0.690</td></tr><tr><td>Modelu</td><td>0.668</td><td>0.688</td><td>None</td><td>0.690</td></tr></table></body></html>

For relative time encoding, we compare RUSR with a variant without it across four experiments, with average results in Table 2. Performance drops significantly without this encoding, showing the importance of news relevance implied by the release time gap.

Then we design two variants, each removing one redistribution module, and present the mean results across Q1-Q4 in Table 2. The results show that RUSR outperforms both variants. To analyze the reasons for this optimization, experiments are conducted on Q4. w/o $\mathbf { M } _ { \mathrm { d r } }$ and w/o $\bf { M } _ { \mathrm { { d r } } } ^ { \mathrm { { r o t a t e } } }$ refer to the removal of $\mathbf { M } _ { \mathrm { d r } }$ and the removal of only the rotation step, respectively. Similarly for w/o $\mathbf { M } _ { \mathrm { d u } }$ and w/o $\bf { M } _ { \mathrm { d u } } ^ { \mathrm { r o t a t e } }$ . RUSR, w/o $\mathbf { M } _ { \mathrm { d r } }$ and w/o $\mathbf { M } _ { \mathrm { d r } } ^ { \mathrm { r o t a t e } }$ : After the 0th epoch, for each $\mathbf { M } _ { \mathrm { r } }$ processed real validation news we collect the cosine similarities in $\mathbf { M } _ { \mathrm { r } }$ and calculate the frequency of similarity falling into each of the four ranges. Finally, frequencies is averaged over all validation news for each range. Similarly for calculating the average frequencies of fake validation news. RUSR, w/o $\mathbf { M } _ { \mathrm { d u } }$ and w/o $\bf { M } _ { \mathrm { { d u } } } ^ { \mathrm { { r o t a t e } } }$ : We similarly calculate the average frequencies for real and fake validation news, replacing all $\mathbf { M } _ { \mathrm { r } }$ with $\mathbf { M } _ { \mathbf { u } }$ . The results are shown in Table 4 with the model performance.

The results show that removing the redistribution module leads to performance loss. In inference, ${ { \bf { M } } _ { r } }$ is expected to have a similarity distribution towards 1 for real news and towards $^ { - 1 }$ for fake news. Additionally, $\mathbf { M } _ { u }$ is expected to have a similarity distribution far from 1 for real news, and for fake news, $N$ similarities close to 1, with the rest far from 1. Table 4 shows that the average similarity distribution of real and fake news for w/o $\mathbf { M } _ { \mathrm { d r } }$ is dominated by [0.5, 1], deviating significantly from the expected distribution. The inclusion of $\mathbf { M } _ { \mathrm { d r } }$ optimizes both distributions, reducing the deviation. Similarly for w/o $ { \mathbf { M } } _ { \mathrm { d u } }$ and $\mathbf { M } _ { \mathrm { d u } }$ .

Table 4: Ablation studies of redistribution module and its rotation part.   

<html><body><table><tr><td rowspan="2">Method</td><td rowspan="2">Modelr macF1</td><td colspan="2">-1</td></tr><tr><td>real</td><td>fake</td></tr><tr><td>w/o Mdr</td><td>0.711</td><td>0/0/0/100</td><td>0/0/0/100</td></tr><tr><td>dr w/o Mrotate</td><td>0.382</td><td>12/80/8/0</td><td>15/78/7/0</td></tr><tr><td>RUSR</td><td>0.738</td><td>8/80/13/0</td><td>12/74/15/0</td></tr><tr><td rowspan="3">Method</td><td>Modelu</td><td>Mu distribution(%)</td><td></td></tr><tr><td rowspan="2">macF1</td><td>[-1,-0.5)/[-0.5,0)/[0,0.5)/[0.5,1]</td><td></td></tr><tr><td>real</td><td>fake</td></tr><tr><td>w/o Mdu</td><td>0.616</td><td>0/0/0/100</td><td>0/0/0/100</td></tr><tr><td> w/o Mdotate</td><td>0.705</td><td>1/62/37/0</td><td>0/64/36/0</td></tr><tr><td>RUSR</td><td>0.752</td><td>0/56/44/0</td><td>0/54/46/0</td></tr></table></body></html>

![](images/df5cc5d04b8f5264ef2e317f9930b604a1f9a2c76c41cf7fb51d670ae2193711.jpg)  
Figure 3: The average number of background news changes with $N _ { \mathrm { m a x } } ^ { \mathrm { B } }$ . Gray means the difference between $N _ { \mathrm { m a x } } ^ { \mathbf { B } }$ and the sum of two averages.

The inclusion of the centering step in w/o $\mathbf { M } _ { \mathrm { d r } } ^ { \mathrm { r o t a t e } }$ and w/o $\bf { M } _ { \mathrm { { d u } } } ^ { \mathrm { { r o t a t e } } }$ also reduces the deviation but causes the feature vectors of detected news and its background news used for similarity calculation to tend to be distributed around the origin, hindering the realization of the expected distribution during inference. The addition of the rotation component eliminates this problem, and thus the results show that both methods have lower performance than RUSR.

# Publication Background Discussion

First, we present the average number of reliable and unreliable source background news for verified news published before 2023 as $N _ { \mathrm { m a x } } ^ { \bar { \mathrm { B } } }$ changes from 20 to 180 in increments of 10. Common background construction is conducted. As shown in the Figure 3, two average numbers increase significantly with the increasement of $N _ { \mathrm { m a x } } ^ { \mathbf { B } }$ . To study the impact of $\dot { N } _ { \mathrm { m a x } } ^ { \mathbf { B } }$ , Figure 4 shows the average macF1 of the three output modules and RUSR in Q1-Q4 as $N _ { \mathrm { m a x } } ^ { \mathbf { B } }$ changes. The following conclusions can be drawn: (1) As the proportion of weakly related news in reliable source background increases, the macF1 of $\mathbf { M } _ { \mathrm { r } }$ remains generally stable, reflecting the rationality of focusing on news with strong validation effectiveness. (2) As $N _ { \mathrm { m a x } } ^ { \mathbf { \tilde { B } } }$ increases from 110, unlike the previous relatively stable phase, the performance of $\mathbf { M } _ { \mathrm { u } }$ declines. This may be due to the fact that, at this stage, each $N _ { \mathrm { m a x } } ^ { \mathbf { B } }$ increase in an experiment leads to a general rise in unreliable source background news which is not in the period of the detected news. This causes the difference between the semantic cluster size of real and fake news to narrow. (3) The performance of RUSR is higher on average across all $N _ { \mathrm { m a x } } ^ { \mathrm { B } }$ than the three output modules, indicating the effectiveness of the multi-model collaboration strategy.

![](images/a4f9e8a442d5244ea722a129684796c27163e923420be0fecdc1640cb27105a9.jpg)  
Figure 4: Performance changes as $N _ { \mathrm { m a x } } ^ { \mathbf { B } }$ increases. For one solid line, the dashed line with the same color indicates the mean of y-coordinates.

# Related Work

We categorize fake news detection methods suitable for early detection into two types: Content-only: Many methods focus on text modalty (Ma et al. 2016; Ajao, Bhowmik, and Zargari 2019; Przybyla 2020; Karimi and Tang 2019) or image modalty (Jin et al. 2017; Qi et al. 2019), while others incorporate both modalities to enhance detection (Wang et al. 2018; Qi et al. 2021; Chen et al. 2022; Zhou et al. 2023; Wang et al. 2024). Reference-enhanced: Most methods use knowledge as supplementary information (Hu et al. 2021; Qian et al. 2021; Popat et al. 2018; Vo and Lee 2021; Zhang et al. 2024; Ma et al. 2019), while (Sheng et al. 2022) enhances detection by perceiving the popularity and novelty of the detected news within its news environment.

# Conclusion

In this paper, we introduce RUSR, which calculates the support degree driven by strong preferences and the two-hop semantic outlier degree with the proposed publication background. Besides, the designed redistribution module and expanding range relative time encoding respectively optimize early checkpoint of training and imply news relevance. Ultimately, the proposed framework ensures mutual benefit among multiple models and supports multi-perspective prediction. Experimental results demonstrate the superiority of our proposed method.