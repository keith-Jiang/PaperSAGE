# Scaffold-BPE: Enhancing Byte Pair Encoding for Large Language Models with Simple and Effective Scaffold Token Removal

Haoran Lian1, Yizhe Xiong2,3, Jianwei $\mathbf { N i u } ^ { 1 , 4 , 5 , 6 * }$ , Shasha $\mathbf { M _ { 0 } } ^ { 1 * }$ , Zhenpeng $\mathbf { S } \mathbf { u } ^ { 7 }$ , Zijia $\mathbf { L i n } ^ { 2 * }$ , Hui Chen2,3\*, Jungong Han2, Guiguang Ding2,3\*

1Beihang University, 2Tsinghua University, 3BNRist, 4State Key Laboratory of Virtual Reality Technology and Systems, Beihang University, 5Zhongguancun Laboratory, 6Zhengzhou University Research Institute of Industrial Technology, Zhengzhou University, 7Chinese Academy of Sciences   
lianhaoran,niujianwei,moshasha @buaa.edu.cn, xiongyizhe2001 $@$ 163.com, suzhenpeng $@$ iie.ac.cn,   
linzijia07@tsinghua.org.cn, {jichenhui2012,jungonghan77}@gmail.com, dinggg $@$ tsinghua.edu.cn

# Abstract

Byte Pair Encoding (BPE) serves as a foundation method for text tokenization in the Natural Language Processing (NLP) field. Despite its wide adoption, the original BPE algorithm harbors an inherent flaw: it inadvertently introduces a frequency imbalance for tokens in the text corpus. Since BPE iteratively merges the most frequent token pair in the text corpus to generate a new token and keeps all generated tokens in the vocabulary, it unavoidably holds tokens that primarily act as components of a longer token and appear infrequently on their own. We term such tokens as Scaffold Tokens. Due to their infrequent occurrences in the text corpus, Scaffold Tokens pose a learning imbalance issue. To address that issue, we propose Scaffold-BPE, which incorporates a dynamic scaffold token removal mechanism by parameterfree, computation-light, and easy-to-implement modifications to the original BPE method. This novel approach ensures the exclusion of low-frequency Scaffold Tokens from the token representations for given texts, thereby mitigating the issue of frequency imbalance and facilitating model training. On extensive experiments across language modeling and even machine translation, Scaffold-BPE consistently outperforms the original BPE, well demonstrating its effectiveness.

# 1 Introduction

In recent years, Large Language Models (LLMs) have become a burgeoning paradigm in handling a broad array of Natural Language Processing (NLP) tasks. The tokenization process in most modern LLMs (Radford et al. 2019; Brown et al. 2020; Rae et al. 2021; Zhang et al. 2022; Biderman et al. 2023; Touvron et al. 2023a; Yang et al. 2023; Achiam et al. 2023; Dubey et al. 2024; Lian et al. 2024a) employs Byte Pair Encoding (BPE) (Sennrich et al. 2015), a method that was originally designed for data compression (Gage 1994). BPE consists of two main stages. In the training stage, BPE iteratively merges the most frequent pairs of bytes or characters in a dataset into a new token and adds it to the vocabulary until a desired vocabulary size is reached. And in the encoding stage, the vocabulary is utilized to represent any text. The adoption of BPE in LLMs is driven by its capability to decompose words into smaller, manageable subword units, thus avoiding out-of-vocabulary words, facilitating flexible and semantically complete representations of input data. Actually, BPE has also been widely used in traditional NLP tasks, like machine translation (Provilkov et al. 2019; Xu et al. 2020), information extraction (Wei et al. 2021, 2023b,a) and summarization (Wu et al. 2021; Xu et al. 2022).

Since its inception, BPE has undergone various modifications to better suit the needs of complex NLP tasks, including identifying the optimal vocabulary size for various tasks (Xu et al. 2020; Gutierrez-Vasques et al. 2021), optimizing the encoding paths of tokens to achieve subword regularization (Provilkov et al. 2019; He et al. 2020), etc.

However, there is an inherent limitation in the BPE method: the iterative merging process can lead to an imbalance in token frequencies by including low-frequency tokens in vocabulary. For example, as illustrated in Figure 1, in the commonly used Pile dataset (Gao et al. 2020) that is tokenized by the original BPE method of 32K vocabulary size (as LLaMA series (Touvron et al. 2023a,b)), the token “zona” mostly appears as a component of the token “Arizona” rather than as an independent, high-frequency token. Despite its lower standalone frequency, BPE includes “zona” in the final vocabulary because it is the “intermediate token” to derive the frequent token “Arizona”. We define such intermediate tokens that are crucial for constructing longer frequent tokens but do not appear frequently on their own as Scaffold Tokens. Note that not all subwords are simply scaffold tokens. For example, “ing” is not identified as a scaffold token, as there are many words containing “ing” but are not tokens in the vocabulary. For example, “connecting” is represented as “connect” $^ +$ “ing” (2 tokens). Such words help to keep “ing” a frequent token. Therefore, “ing” is not a scaffold token. According to our proposed Scaffold-BPE method, the 32K vocabulary contains about $6 . 0 7 \%$ of scaffold tokens.

![](images/7a083dfe7c44c6e66985b2d80ac37b027b2ef44925a1c6cc8bb74e6893186bfe.jpg)  
Figure 1: Two types of tokens in original BPE vocabulary on the Pile dataset: one type, such as “ing”, appears frequently by itself, while the other type, such as “zona”, mostly appears as a component of “Arizona” and thus has a low individual occurrence frequency. The value on the horizontal axis represents the percentage of the frequency of the token relative to the total frequency of all tokens containing “ing”/“zona”. For “ing”, we only visualize the top tokens.

As depicted in Figure 2, a natural frequency imbalance arises between these scaffold tokens and actual highfrequency tokens. Prior studies (Lin et al. 2017; Su et al. 2023) have highlighted that such disparities in token frequencies can result in imbalanced learning difficulties across different tokens. Scaffold tokens, due to their lower individual occurrence frequencies, are notably harder to learn for models.

To address that issue, we propose enhancements to the BPE algorithm, aiming to mitigate the frequency imbalance and ensure a more equitable learning process for all tokens. Specifically, we propose the simple and effective Scaffold-BPE with a dynamic scaffold token removal mechanism, which is parameter-free, computation-light, easy-toimplement, and widely effective. Generally, the proposed Scaffold-BPE maintains an expanded vocabulary compared with the original BPE, which consists of both normal tokens and scaffold tokens. Note that the scaffold tokens are not actual tokens in the vocabulary and do not appear in the tokenized sequences after encoding. In the training stage, Scaffold-BPE dynamically marks tokens with lower individual occurrence frequencies as scaffold tokens in each iteration. In the encoding stage, the Scaffold-BPE firstly utilizes all tokens in the expanded vocabulary to generate the token representations for the given texts, which is termed as a Scaffolding process. Then, the Scaffold-BPE ensures the absence of all scaffold tokens in the token representation by demolishing them into their shortest non-scaffold-token sequence, which is termed as a Demolishing process. Thanks to such modifications, Scaffold-BPE can remove scaffold tokens from the final token representations fed into models, thus enjoying more balanced token occurrences, leading to more sufficient learning and better performance of models.

![](images/965fa9186702305aa38f8fdb620c2de1480ebf1abaf8a44dd6d326e7bf231b71.jpg)

Figure 2: Sorted token frequencies in descending order of the original BPE and Scaffold-BPE.

We conduct extensive experiments on language modeling tasks. Results on 9 widely used language modeling benchmarks demonstrate that Scaffold-BPE consistently outperforms the original BPE. Besides, even when extended to machine translation tasks, Scaffold-BPE proves highly effective. Furthermore, we show that Scaffold-BPE is orthogonal to existing modifications on BPE, like BPE-Dropout (Provilkov et al. 2019) and can be combined with them to achieve further improvements.

Overall, our contributions are three-fold:

• We observe that the iterative training process of BPE incorporates low-frequency tokens into the vocabulary, which we term scaffold tokens.   
• We propose Scaffold-BPE, which can remove scaffold tokens from the final token representations by dynamically marking scaffold tokens in the training process and temporarily utilizing scaffold tokens in the encoding process. Scaffold-BPE is parameter-free, computation-light, easy-to-implement, and widely effective, preserving the simplicity and clarity of BPE.   
• Extensive experiments demonstrate that Scaffold-BPE outperforms the original BPE on language modeling and also machine translation tasks, proving its effectiveness and robustness in the NLP field.

# 2 Related Works

Recently, LLMs have become a popular paradigm for solving NLP tasks, with BPE serving as the mainstream tokenizer to split a text into a sequence of tokens (e.g., subwords/words/phrases). Thus, enhancing BPE could boost the performance of LLMs and have positive implications for various applications.

Token Frequency Scaffold $f ( \mathrm { A r i } ) \gets f ( \mathrm { A r i } ) - f \big ( ( \mathrm { A r i } , \mathsf { z o n a } ) \big )$ Token Frequency Scaffold Token Frequency Scaffold · · 1 $\underline { { f \mathrm { ( z o n a ) }  f \mathrm { ( z o n a ) } - f \mathrm { ( ( A r i , z o n a ) } ) } } _ { \mathrm { . } }$ Ari 73020 False merge Ari 29421 False Scaffold(Ari) $$ True Ari 29421 True Scaffold(zona) ← True   
VEoxcpaabnudlaerdy azdoanta 458490475 FTarluse Scaiff oAlrdi(zAorniazoina𝑆) $$ eFnalse azdoanta 5346 FTarluse $f ( \mathrm { z o n a } ) < f ( Q _ { h e a d } )$ azdoanta 5344067 True mate 44057 False mate 44057 False mate 44057 False append to 𝐸 Vocabulary $V = \{ t \in E | \mathrm { S c a f f o l d } ( t ) = F a l s e \}$ Arizona 43599 False Arizona 43599 False   
Scaffold Vocabulary $S = \{ t \in E | \mathrm { S c a f f o l d } ( t ) = T r u e \}$ } push back Token Pair Frequency↓ Token Pair Frequency↓ Token Pair Frequency↓   
PQriuoeriutey (P, ad) 43598 merge Arizona (I(nctaenr, fcaecl)e) 43597 𝑓 𝑄ℎ𝑒𝑎𝑑 (c(aPn,,acde)l) 435987 update 𝑄 (IPnrtoevr, fiadcere) 4359856 ··· Ari/zona ··· token pairs (Arizona, to) 763 (A, ri) 29421 (l, oyal) 43576 Q · Arizona ··· (lead, Arizona) 125 (zon, a) 5346 · · · · · · · · · Iteration N

# 2.1 Language Models

Language models are designed to maximize the likelihood of a token sequence. Following GPT-3 (Brown et al. 2020), which features 175 billion parameters and demonstrates versatility across a wide range of applications, there has been a significant push towards developing large generative language models like Gopher (Rae et al. 2021), PaLM (Chowdhery et al. 2023), GaLM (Du et al. 2022), OPT (Zhang et al. 2022), and LLaMA (Touvron et al. 2023a). Such a surge in development has greatly advanced the fields of natural language understanding and generation.

# 2.2 Byte Pair Encoding

Early neural models had difficulty managing rare words due to limited vocabulary sizes. BPE (Sennrich et al. 2015) effectively addresses that by generating a subword vocabulary. Initially, a corpus is split into characters or bytes, which act as initial tokens. The algorithm iteratively finds the most frequent token pair in the sequence, merges them into a new token, and adds it to the vocabulary until it reaches a predetermined size. The vocabulary is then utilized during the encoding phase to represent any text. Recent advancements like BPE-dropout (Provilkov et al. 2019), LBPE (Lian et al. 2024b) and optimal vocabulary size search (Xu et al. 2020; Gowda et al. 2020; Salesky et al. 2020) continue to enrich BPE developments.

Recently, some work discovered similar issues, but they analyzed BPE from different perspectives.

Trimmed BPE (Cognetta et al. 2024) replaces rare subwords with their components during encoding and does not change the vocabulary. The authors claimed that it “fails to improve model performance”. Differently, our ScaffoldBPE optimizes the BPE vocabulary to mitigate token frequency imbalance, which yields substantial performance gain for LLMs and other tasks.

BPE-Knockout (Bauwens and Delobelle 2024) shows that a BPE tokenizer can be made significantly more adherent to morphological boundaries by disabling merges that abridge them excessively. However, 1) BPE-Knockout targets morphology, thus is not applicable to non-morphological languages like Chinese, while our Scaffold-BPE is applicable to any language. 2) BPE-Knockout requires a reference corpus for morphological guidance, needing either manual annotation or developing auto-annotation models. Our ScaffoldBPE is unsupervised, and needs no manual labeling or parameter tuning. 3) BPE-Knockout requires manually setting the threshold. Differently our Scaffold-BPE requires no threshold setting.

Picky BPE (Chizhov et al. 2024) implements vocabulary refinement during tokenizer training by removing intermediate tokens once they become useless. However, it requires manual setting of thresholds for removing tokens. Differently our Scaffold-BPE requires no threshold setting. It has a dynamic and adaptive scaffold token removal mechanism.

# 3 Methodology

To enhance the original BPE, we propose Scaffold-BPE to remove the scaffold tokens introduced by the original BPE. Our Scaffold-BPE is simple yet effective. In the training process, the Scaffold-BPE dynamically marks scaffold tokens in the vocabulary at each iteration, and finally yields an expanded vocabulary consisting of both normal tokens with the amount equaling the predetermined vocabulary size and several scaffold tokens. In the encoding process, apart from using the normal tokens, Scaffold-BPE temporarily uses scaffold tokens as intermediate tokens to merge into longer normal tokens.

# 3.1 Training Process

The original BPE is trained on a text corpus $C$ with a predefined vocabulary size $N$ . After training, BPE returns a vocabulary $V$ consisting of $N$ tokens. For simplicity, $C$ is

Require: Text Corpus $C$ , Desired Vocabulary Size $N$   
1: Initialize an expanded vocabulary $E$ , consisting of a   
normal-token vocabulary $V$ and a scaffold-token vocab  
ulary $S$   
2: Split $C$ into a list of characters/bytes (denoted as $L$ )   
3: Initialize a priority queue $Q$ storing token pairs within   
$L$ , arranged in reverse order of frequency   
4: while $| \bar { V } | < N$ do   
5: $( a , b )  \mathsf { p o p } Q _ { h e a d }$   
6: Merge pair $( a , b )$ into a new token $t$   
7: if $t$ in $S$ then   
8: // $t$ may be a previously marked scaffold token   
9: $\mathbf { S c a f f o l d } ( t ) \gets$ False   
10: continue   
11: end if   
12: Add $t$ to $E$ as a normal token   
13: Replace all of $( a , b )$ in $L$ with $t$   
14: Update $Q$   
15: /\*\*\*\*\*\* Scaffold-BPE Modification Begins \*\*\*\*\*\*/   
16: for each $t ^ { \prime }$ in $\{ a , b \}$ do   
17: $f ( t ^ { \prime } ) \gets f ( \dot { t ^ { \prime } } ) - \dot { f } ( t )$   
18: if $t ^ { \prime }$ in $V$ and $f ( t ^ { \prime } ) < f ( Q _ { h e a d } )$ then   
19: Scaffold $( t ^ { \prime } ) \gets$ True   
20: Push $t ^ { \prime }$ back to $Q$   
21: end if   
22: end for   
23: /\*\*\*\*\*\* Scaffold-BPE Modification Ends \*\*\*\*\*\*/   
24: end while   
25: return $E$ (with $V$ and $S$ both included)

firstly split into a sequence of smallest unit tokens (denoted as $L$ ), with each token being a single character/byte. We define $a , b$ as two tokens, $( a , b )$ as a token pair, and $f ( \cdot )$ as the frequency of a token or token pair within $L$ . BPE is trained iteratively. In each iteration, BPE identifies the token pair with the highest frequency:

$$
( a , b ) = \arg \operatorname* { m a x } _ { ( x , y ) \in L } f ( ( x , y ) )
$$

BPE then merges (i.e., concatenates) them into a new token $t$ , and includes $t$ in $V$ . Then BPE updates $L$ via replacing all $( a , b )$ with $t$ , and restarts the process again.

The iterative process of identifying the most frequent token pair $( a , b )$ can be accelerated using a priority queue $Q$ . At the beginning of the training process, all token pairs in $L$ are pushed into $Q$ with a descending order of frequency. And after the token pair $( a , b )$ is merged into $t$ in each iteration, BPE updates the frequencies and the ranks of token pairs related to all indexed occurrences of $( a , b )$ . For instance, given $( a , b )$ in a context of $\cdot \cdot \cdot , u , a , b , v , \ldots$ ” in $L$ , when $( a , b )$ is replaced with $t$ , the frequency of $( u , a )$ or $( b , v )$ would decrease by 1, and meanwhile that of $( u , t )$ or $( t , v )$ would increase by 1. With the occurrences of all token pairs being indexed, there is no need to scan $L$ again and re-count the frequencies of all candidate token pairs for a new iteration. After updating the adjacent token pairs related to $( a , b )$ (i.e, $t$ ), the frequencies of token pairs like $( u , a )$ or $( b , v )$ would

1: Split $T$ into a character/byte token representation (denoted as $R$ ) 2: while True do 3: /\*\*\*\*\*\*\*\*\*\*\*\* Scaffolding Begins \*\*\*\*\*\*\*\*\*\*\*\*/ 4: Identify all possible merges $M$ using $E$ , ignoring token types 5: /\*\*\*\*\*\*\*\*\*\*\*\*\* Scaffolding Ends \*\*\*\*\*\*\*\*\*\*\*\*\*/ 6: if $M$ is empty then 7: break 8: end if 9: Select $m$ which is ranked before the others in $E$ from $M$ 10: Apply $m$ to $R$ 11: end while 12: /\*\*\*\*\*\*\*\*\*\*\*\* Demolishing Begins \*\*\*\*\*\*\*\*\*\*\*\*/ 13: Demolish all scaffold tokens in $R$ into its shortest nonscaffold child token sequence 14: /\*\*\*\*\*\*\*\*\*\*\*\*\* Demolishing Ends \*\*\*\*\*\*\*\*\*\*\*\*\*/ 15: return $R$

be updated in $Q$ , and meanwhile the new candidate token pairs $( u , t )$ and $( t , v )$ would also be pushed into $Q$ with their frequencies.

The Scaffold-BPE expands the vocabulary $V$ to an expanded vocabulary $E$ , and assigns an attribute (denoted as Scaffold(·)) to each token in the vocabulary indicating whether it is a scaffold token or not. Thus, the expanded vocabulary $E$ comprises two types of tokens, i.e., normal ones and scaffold ones. We denote all the non-scaffold tokens by $V$ , which, as with the original BPE, are the tokens actually used in representing texts for NLP model training:

$$
V = \{ t \in E \mid \operatorname { S c a f f o l d } ( t ) = F a l s e \}
$$

Additionally, we denote all the scaffold tokens by $S$ , which are not fed into the model, nor do they appear in any token representations after encoding:

$$
S = \{ t \in E \mid { \mathrm { S c a f f o l d } } ( t ) = T r u e \}
$$

They only serve as intermediate tokens to aid in the training and encoding processes of Scaffold-BPE. Therefore, when calculating vocabulary size, the count of scaffold tokens is not included; only the number of tokens in $V$ is considered.

Initially, a token pair is merged and added to $E$ due to its high frequency. Similarly, Scaffold-BPE marks a token as a scaffold token when its frequency decreases too much. Throughout the entire training process of BPE, $f ( a )$ and $f ( b )$ only decrease when the token pair $( a , b )$ is merged into a new token $t$ . Therefore, as presented in Algorithm 1, Scaffold-BPE introduces an additional step at the end of each iteration, utilizing the decreased $f ( a )$ and $f ( b )$ to evaluate whether $a$ and $b$ remain high-frequency. If they are no longer considered high-frequency, they would be marked as scaffold tokens. Naturally, the token pair at the head of the priority queue $Q$ (denoted as $Q _ { h e a d } )$ is the next candidate to be added to the vocabulary. Then $f ( Q _ { h e a d } )$ is a natural frequency delimiter between in-vocabulary and out-ofvocabulary tokens. Therefore, if $f ( a )$ (or $f ( b ) ) \dot { < } f ( Q _ { h e a d } )$ , $a$ (or $b$ ) is marked as a scaffold token, which means it is not included in $V$ :

<html><body><table><tr><td></td><td></td><td>BoolQ</td><td>HellaSwag</td><td>OpenBookQA</td><td>PIQA</td><td>SIQA</td><td>StoryCloze</td><td>Winogrande</td></tr><tr><td rowspan="2">468M</td><td>Original BPE</td><td>58.64</td><td>40.78</td><td>30.50</td><td>66.57</td><td>43.40</td><td>62.77</td><td>53.00</td></tr><tr><td>Scaffold-BPE</td><td>60.52</td><td>41.68</td><td>32.20</td><td>68.69</td><td>44.09</td><td>63.04</td><td>54.22</td></tr><tr><td rowspan="2">1.2B</td><td></td><td></td><td>47.25</td><td>31.70</td><td>68.55</td><td>4.09</td><td>65.61</td><td>55.520</td></tr><tr><td>Ocafioad BPE</td><td>60.86</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">6.7B</td><td>Original BPE</td><td>62.87</td><td>60.57</td><td>35.10</td><td>73.69</td><td>46.98</td><td>71.43</td><td>60.97</td></tr><tr><td>Scaffold-BPE</td><td>64.95</td><td>61.19</td><td>38.00</td><td>74.54</td><td>47.49</td><td>72.26</td><td>61.76</td></tr></table></body></html>

Table 1: At varying model scales, the average accuracy on 0/5-shot common sense reasoning benchmarks $\dot { p }$ -value $< 0 . 0 1$ ).

$$
\mathrm { S c a f f o l d } ( a ) = \left\{ \mathrm { T r u e , } \quad \mathrm { i f } f ( a ) < f ( Q _ { h e a d } ) \right.
$$

Notably, such an additional step leverages the inherent mechanism of BPE without introducing any additional hyper-parameters, maintaining the simplicity and clarity of BPE. Moreover, $f ( Q _ { h e a d } )$ is dynamically adjusted in each iteration, ensuring that Scaffold-BPE can adaptively identify scaffold tokens at any iteration step. Furthermore, scaffold tokens are not permanently marked. They are pushed back into $Q$ , reserving the possibility of being ranked top at the priority queue and re-integrated into $V$ in a future iteration.

# 3.2 Encoding Process

The encoding process of the original BPE encodes a text $T$ into a token representation (i.e., $R$ ) using the vocabulary $V$ generated by BPE training. Firstly, $R$ is a sequence of smallest unit tokens (i.e., character/byte tokens), obtained by splitting $T$ . And then, following the ranks of tokens in $V$ as merging priority (i.e., tokens added earlier have higher frequency and thus are assigned higher priority to be merged into), token pairs in $R$ are iteratively merged to build the final representation.

Similarly, the modifications of Scaffold-BPE in the encoding process are straightforward. Compared to the original BPE, the expanded vocabulary $E$ is utilized. In each iteration, the token $t$ to be merged would be selected from both normal tokens and scaffold tokens:

$$
t = \arg \operatorname* { m i n } _ { t \in E } r a n k _ { E } ( t )
$$

where $r a n k _ { E } ( \cdot )$ denotes the rank of a token in $E$ . Consequently, during the encoding process, the count of different tokens used actually exceeds the predefined vocabulary size (i.e., $N _ { , }$ ). And scaffold tokens are employed as intermediate tokens to merge into longer tokens. We term such a mechanism as Scaffolding, as shown in Algorithm 2.

When no more token pairs can be merged in $R$ , the original BPE returns $R$ as the final result. However, due to the introduction of the Scaffolding mechanism in Scaffold-BPE, $R$ may contain scaffold tokens from $S$ , potentially increasing the variety of tokens beyond the predefined vocabulary

Expanded Vocabulary 𝐸 Token Representation 𝑅 Token Scaffold ··· z/o/n/a ··· A/r/i/z/o/n/a ··· ri False ··· z/o/n/a ··· A/ri/z/o/n/a ··· on False ··· z/on/a ··· A/ri/z/on/a ··· zon False ··· zon/a ··· A/ri/zon/a ··· Ari False ··· zon/a ··· Ari/zon/a ·· Scaffolding zona True ··· zona ··· Ari/zona ··   
Arizona False zona Arizona · Demolishing zon/a Arizona ··

size and exceeding the range of word embeddings that the model can map. To address it, Scaffold-BPE adds one additional step termed as Demolishing at the end of the encoding process. Scaffold-BPE demolishes all scaffold tokens in $R$ into their shortest non-scaffold child token sequences, ensuring that $R$ only consists of tokens from $V$ . For example, as shown in Figure 4, the remaining “zona” in $R$ is demolished into “zon” and “a”. The demolishing step can be formulated as follows:

$$
t = { \left\{ \begin{array} { l l } { t , } & { { \mathrm { i f ~ } } \mathrm { S c a f f o l d } ( t ) = \mathrm { F a l s e } \ } \\ { ( a , b ) , } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }
$$

where $a$ and $b$ are the components of the scaffold token $t$ . The formula above would be recursively applied to $a$ and $b$ to derive the shortest non-scaffold child token sequence for $t$ . After the Demolishing step, Scaffold-BPE returns the final token sequence representation (i.e., $R _ { \star }$ ) for $T$ . Since the shortest non-scaffold child token sequences for all scaffold tokens can be precomputed during the training process, the time complexity of demolishing one token is $\bar { O ( 1 ) }$ , making its impact on encoding efficiency negligible.

# 4 Experiments

We employ the recently well-attended language modeling tasks to validate the effectiveness of the Scaffold-BPE.

# 4.1 Experimental Setup

Datasets. Our models are trained on the Pile (Gao et al. 2020) dataset, an 825.18 GiB English text dataset designed for training LLMs. The data distribution for our model training is identical to that of the original work (Gao et al. 2020).

Tokenizer. We train two 32K vocabularies (size applied by LLaMA series (Touvron et al. 2023a,b)) using the original BPE and Scaffold-BPE, respectively. Similar to GPT2 (Radford et al. 2019), pre-tokenization was employed to prevent the merging of tokens from different character categories. And following (Touvron et al. 2023a), we split numbers into individual digits.

Model. We train three language models with 468M, 1.2B, and 6.7B parameters, respectively. Specifically, the architectures of the 468M and the 1.2B models are identical to those of the 410M and the 1.0B models outlined in Pythia (Biderman et al. 2023). The minor differences in parameter sizes are attributed to the variations in vocabulary size in the embedding and output layer. As for the 6.7B model, its architecture is identical to LLaMA-7B (Touvron et al. 2023a).

Training. Following the pretraining settings of previous works (Xie et al. 2023; Su et al. 2024; Xiong et al. 2024) and limited by our computation budget, by default all models are pretrained with 100B tokens. Note that the volume of corresponding text data contained in an equal amount of tokens is slightly different between the two tokenizers. Considering model training efficiency and commonly used criteria (i.e., the token amount) of computation budget in LLM training, we compare experiments in the setting of an equal amount of training tokens. In the Section 4.3, we further analyze both tokenizers in the setting of an equal amount of training text volume.

# 4.2 Experimental Results

Common Sense Reasoning. Our analysis incorporates 7 benchmarks recognized for evaluating common sense reasoning, including BoolQ (Clark et al. 2019), HellaSwag (Zellers et al. 2019), OpenBookQA (Mihaylov et al. 2018), PIQA (Bisk et al. 2020), SIQA (Sap et al. 2019), StoryCloze (Mostafazadeh et al. 2016), and Winogrande (Sakaguchi et al. 2021). We present the performance of all models in terms of average accuracy in 0-shot and 5-shot settings.

As shown in Table 1, we can observe that the ScaffoldBPE consistently outperforms the original BPE on different setups with different model sizes. Notably, the 6.7B model trained with Scaffold-BPE can achieve a significant $2 . 0 8 \mathrm { p p }$ (percent point) improvement on BoolQ and a $2 . 9 0 \mathrm { p p }$ improvement on OpenBookQA. We conduct a $t$ -test, and all metrics have $p$ -values less than 0.01, indicating that the results are statistically significant.

Such results clearly demonstrate that although the modifications are simple, our proposed Scaffold-BPE is convincingly effective. We attribute it to that Scaffold-BPE can encode text into tokens with a more balanced frequency distribution, which can help language models to learn all tokens more thoroughly.

Closed Book Question Answering. For the task of closed book question answering (Brown et al. 2020; Touvron et al. 2023a), we evaluate the performance of the largest 6.7Bparameter models with different tokenizers on 2 benchmark datasets, i.e., TriviaQA (Joshi et al. 2017) and WebQuestions (Berant et al. 2013). We report the exact match performance for the zero-shot and few-shot settings in Table 2. It can be seen that the model trained with the proposed ScaffoldBPE can achieve a $3 . 2 3 \mathrm { p p }$ improvement on TriviaQA and a $1 . 3 3 \mathrm { p p }$ improvement on WebQuestions, with both $p$ -values less than 0.01. All results above demonstrate that ScaffoldBPE can enhance model performance across different types of downstream tasks.

Table 2: The average exact match performance on 0/5-shot closed-book question-answering benchmarks of the 6.7Bparameter models ( $\overset { \cdot } { p }$ -value $< 0 . 0 1 \dot { }$ ).   

<html><body><table><tr><td></td><td>TriviaQA</td><td>WebQuestions</td></tr><tr><td>Original BPE</td><td>15.63</td><td>8.56</td></tr><tr><td>Scaffold-BPE</td><td>18.86</td><td>9.89</td></tr></table></body></html>

![](images/f83cada4dbbae0f747f7e30aa3dc45ac14f37c173e3dcd6d37022c041c6a8df1.jpg)  
Figure 5: The average frequencies of the scaffold tokens and the new actual high-frequency tokens that replace the scaffold tokens in vocabularies of 32K, 64K and 128K.

# 4.3 Discussion

Various Vocabulary Size. Depending on the size of the training corpus, the diversity of the languages, the size of the model, and the types of tasks, different vocabulary sizes are set in practice. Therefore, to validate the robustness of Scaffold-BPE across various vocabulary sizes, in addition to the 32K vocabulary (Touvron et al. 2023a), we also trained two vocabularies sized at 64K (Baichuan 2023b,a) and 128K (Yang et al. 2023). The experimental setup is identical to that of the 468M-parameter model mentioned before.

As shown in Figure 5, Scaffold-BPE can replace scaffold tokens in vocabularies with actual high-frequency tokens, which significantly increases the average frequencies of those tokens. The frequency improvements are $7 6 . 4 0 \%$ , $6 8 . 5 8 \%$ , and $5 8 . 9 9 \%$ for the 32K, 64K, and 128K vocabulary sizes, respectively. The enhancement in token frequency distribution effectively promotes the learning of those tokens, which can contribute to better model performance across various tasks.

Moreover, as shown in Table 3, the results demonstrate that Scaffold-BPE consistently outperforms the original

Table 3: At varying vocabulary sizes, the average accuracy on 0/5-shot common sense reasoning benchmarks $\overset { \cdot } { p }$ -value $< 0 . 0 1 \}$ .   

<html><body><table><tr><td></td><td colspan="2">64K Scaffold-BPE</td><td colspan="2">128K</td></tr><tr><td></td><td>BPE</td><td></td><td>BPE</td><td>Scaffold-BPE</td></tr><tr><td>BoolQ HellaSwag</td><td>58.01</td><td>59.71</td><td>56.67</td><td>59.43</td></tr><tr><td></td><td>41.82</td><td>42.06 31.10</td><td>42.70</td><td>42.91</td></tr><tr><td>OpenBookQA</td><td>30.90</td><td>69.26</td><td>31.10 67.68</td><td>32.30 68.82</td></tr><tr><td>PIQA SIQA</td><td>67.95 43.47</td><td>43.86</td><td>43.83</td><td>44.14</td></tr><tr><td>StoryCloze</td><td>64.19</td><td>65.02</td><td>64.11</td><td>65.26</td></tr><tr><td></td><td></td><td>54.34</td><td></td><td></td></tr><tr><td>Winogrande</td><td>53.67</td><td></td><td>53.91</td><td>55.09</td></tr></table></body></html>

Table 4: At 300B training tokens, the average accuracy on 0/5-shot common sense reasoning benchmarks ( $\overset { \cdot } { p }$ -value $<$ 0.01).   

<html><body><table><tr><td></td><td>Original BPE</td><td>Scaffold-BPE</td></tr><tr><td>BoolQ</td><td>58.21</td><td>61.62</td></tr><tr><td>HellaSwag</td><td>45.10</td><td>46.03</td></tr><tr><td>OpenBookQA</td><td>31.10</td><td>33.50</td></tr><tr><td>PIQA</td><td>68.63</td><td>69.86</td></tr><tr><td>SIQA</td><td>43.78</td><td>44.73</td></tr><tr><td>StoryCloze</td><td>65.53</td><td>66.54</td></tr><tr><td>Winogrande</td><td>53.67</td><td>56.12</td></tr></table></body></html>

BPE across all vocabulary sizes, which indicates that the superiority of Scaffold-BPE is not sensitive to vocabulary size. Its algorithmic design enables it to adaptively remove scaffold tokens across any vocabulary size, without the need for manually designed or heavily-tuned hyperparameters.

More Training Tokens. According to the Scaling Law, the loss scales as a power-law with model size, dataset size, and the amount of training computation (Kaplan et al. 2020). To demonstrate the effectiveness of our Scaffold-BPE with more training tokens, we continue training the 468M models up to 300B tokens (Zhang et al. 2022; Biderman et al. 2023). As shown in Table 4, the results demonstrate that Scaffold-BPE consistently outperforms the original BPE at 300B training tokens, well indicating that in the era of increasingly large training datasets for LLMs, our ScaffoldBPE can effectively enhance the capabilities of those models through simple modifications to the original BPE.

Applicable for Other Tasks, Languages, Model Architectures and Compatible with Other BPE Enhancements. Although the development of LLMs is burgeoning, some applications still prefer using conventional models due to their lower training and inference costs. In the NLP field, BPE was initially combined with transformer models and applied to machine translation tasks (Sennrich et al. 2015), which typically face an open vocabulary challenge and involve substantial textual variations between two languages. Therefore, to validate the versatility of the Scaffold-BPE method, we additionally conduct evaluations on machine translation tasks with identical experimental setup on WMT’14 En-De and En-Fr dataset in the prior work (Ott et al. 2018).

Table 5: BLEU on WMT’14 En–De and En–Fr.   

<html><body><table><tr><td></td><td>En-De</td><td>En-Fr</td></tr><tr><td>Original BPE</td><td>29.31</td><td>43.20</td></tr><tr><td>+ BPE-Dropout</td><td>29.50</td><td>43.44</td></tr><tr><td>Scaffold-BPE</td><td>29.76</td><td>43.81</td></tr><tr><td>+ BPE-Dropout</td><td>29.78</td><td>43.83</td></tr></table></body></html>

As shown in Table 5, Scaffold-BPE outperforms the original BPE in machine translation tasks, which demonstrates that Scaffold-BPE is not specific to language modeling tasks and can be applied to a wider range of tasks.

Besides, experiments conducted with En-De and En-Fr language pairs demonstrate that Scaffold-BPE is language insensitive. Scaffold-BPE is capable of identifying and removing the scaffold tokens introduced by the original BPE across different languages.

Moreover, previous experiments on language modeling tasks are carried out on the decoder-only architecture. For the machine translation tasks, we utilize the encoderdecoder architecture (Vaswani et al. 2017). The exceptional performance of Scaffold-BPE confirms its architecture insensitivity, indicating its applicability across a wider range of neural network architectures.

Finally, Scaffold-BPE is orthogonal to and can be combined with existing enhancements to BPE, like BPEDropout (Provilkov et al. 2019). As shown in Table 5, Scaffold-BPE with BPE-Dropout achieves further improvements on BLEU, well indicating the compatibility of Scaffold-BPE.

Higher Compression Rate. Besides the performance of models on downstream NLP tasks, the compression rate for a given text corpus is a metric to measure the effectiveness of a tokenizer. A higher compression rate means that fewer tokens are required to represent the same corpus. As shown in Table 6, Scaffold-BPE, utilizing a scaffold tokens removal mechanism, retains more actual high-frequency tokens in the final vocabulary, and thus it achieves a higher compression rate on all the corpus in our experiments.

Experiments under Same Corpus Size As mentioned before, considering model training efficiency and commonly used criteria (i.e., the token amount) of computation budget in LLM training, experiments above are compared in the setting of an equal amount of training tokens. To eliminate the impact of different amounts of training text caused by different compression rates on experiment results, we additionally train two 468M-parameter models on exactly 388 GiB training text $\mathrm { \Lambda } ( \approx 1 0 0 \mathbf { B }$ tokens). As shown in Table 7, Scaffold-BPE consistently outperforms the original BPE, demonstrating that the effectiveness of Scaffold-BPE is not merely obtained by allowing models to digest more data in the same computation budget. Our Scaffold-BPE also alleviates the issue of token frequency imbalance, allowing models to learn all tokens more sufficiently and evenly, thus achieving better performance.

Table 6: Compression Rate (the average number of bytes per token) on the Pile dataset and the WMT dataset.   

<html><body><table><tr><td></td><td>Pile</td><td>En-De</td><td>En-Fr</td></tr><tr><td>Original BPE</td><td>3.879</td><td>4.830</td><td>5.012</td></tr><tr><td>Scaffold-BPE</td><td>3.889</td><td>4.861</td><td>5.042</td></tr></table></body></html>

Table 7: At exactly 388 GiB training text, the average accuracy on 0/5-shot common sense reasoning benchmarks $p { \cdot }$ value $< 0 . 0 1 \dot { }$ ).   

<html><body><table><tr><td></td><td>Original BPE</td><td>Scaffold-BPE</td></tr><tr><td>BoolQ</td><td>58.72</td><td>60.55</td></tr><tr><td>HellaSwag</td><td>40.84</td><td>41.69</td></tr><tr><td>OpenBookQA</td><td>30.55</td><td>32.22</td></tr><tr><td>PIQA</td><td>66.58</td><td>68.78</td></tr><tr><td>SIQA</td><td>43.40</td><td>44.13</td></tr><tr><td>StoryCloze</td><td>62.85</td><td>63.08</td></tr><tr><td>Winogrande</td><td>53.07</td><td>54.25</td></tr></table></body></html>

Higher Entropy, Lower Redundancy Scaffold-BPE can alleviate the imbalance in token frequency, which can lead to an increase in information entropy. We measure Shannon Entropy and Redundancy (Gutierrez-Vasques et al. 2021) over token representations of texts obtained with the original BPE and our Scaffold-BPE. Both take as input a text $T$ with a vocabulary of (normal) tokens $V = \{ t _ { 1 } , \hat { t _ { 2 } } , . . . , t _ { V } \}$ of size $| V |$ .

Entropy $H$ is a measure of the average information. Where the probability of a token $p ( t )$ is estimated using the so-called maximum likelihood method (i.e., its relative frequency in the text). Higher values of Entropy indicate higher complexity (less predictability).

$$
H ( T ) = - \sum _ { i = 1 } ^ { V } p ( t _ { i } ) \log _ { 2 } p ( t _ { i } )
$$

The Redundancy $R$ quantifies how close the empirically estimated entropy is to the maximum value it can take.

$$
R ( T ) = 1 - \frac { H ( T ) } { m a x \{ H ( T ) \} } = 1 - \frac { H ( T ) } { \log _ { 2 } | V | }
$$

As shown in Table 8, taking the 32K vocabulary as an example, our Scaffold-BPE can encode Pile dataset (Gao et al. 2020) with higher Entropy and lower Redundancy. Consequently, tokens in the vocabulary of our Scaffold-BPE have more balanced appearing probabilities. According to Su et al. (2023), our vocabulary with balanced token occurrences mitigates the learning imbalance problem, resulting in more sufficient learning towards the text corpus, thus achieving better performance.

# 5 Conclusions

In this paper, we present our observation of tokens with imbalanced frequencies in BPE vocabulary, which we term scaffold tokens. Those scaffold tokens, while integral to the formation of longer tokens, do not represent actual frequent tokens and affect the performance of LLMs negatively. To address that, we propose Scaffold-BPE, which can remove scaffold tokens from the final token representations by dynamically marking scaffold tokens in the training process and temporarily utilizing them in the encoding process. The Scaffold-BPE is parameter-free, computation-light, easy-toimplement, and widely effective, well preserving the simplicity and clarity of BPE. Through extensive experiments, including varying model sizes, vocabulary sizes and more training tokens, etc., Scaffold-BPE demonstrates its robustness and superiority over the original BPE.

Table 8: Entropy and Redundancy on tokenized Pile dataset.   

<html><body><table><tr><td></td><td>Entropy↑</td><td>Redundancy↓</td></tr><tr><td>Original BPE</td><td>11.2382</td><td>0.2491</td></tr><tr><td>Scaffold-BPE</td><td>11.2443</td><td>0.2487</td></tr></table></body></html>