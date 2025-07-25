# EBBS: An Ensemble with Bi-Level Beam Search for Zero-Shot Machine Translation

Yuqiao $\mathbf { W e n } ^ { 1 , * }$ , Behzad Shayegh1,∗, Chenyang Huang1, Yanshuai $\mathbf { C a o ^ { 2 } }$ , Lili $\mathbf { M o u } ^ { 1 , 3 }$

1Dept. Computing Science, Alberta Machine Intelligence Institute (Amii), University of Alberta 2RBC Borealis 3Canada CIFAR AI Chair, Amii yq.when $@$ gmail.com, the.shayegh $@$ gmail.com, chenyangh $@$ ualberta.ca yanshuai.cao $@$ borealisai.com, doublepower.mou $@$ gmail.com

# Abstract

The ability of zero-shot translation emerges when we train a multilingual model with certain translation directions; the model can then directly translate in unseen directions. Alternatively, zero-shot translation can be accomplished by pivoting through a third language (e.g., English). In our work, we observe that both direct and pivot translations are noisy and achieve less satisfactory performance. We propose EBBS, an ensemble method with a novel bi-level beam search algorithm, where each ensemble component explores its own prediction step by step at the lower level but all components are synchronized by a “soft voting” mechanism at the upper level. Results on two popular multilingual translation datasets show that EBBS consistently outperforms direct and pivot translations, as well as existing ensemble techniques. Further, we can distill the ensemble’s knowledge back to the multilingual model to improve inference efficiency; profoundly, our EBBS-distilled model can even outperform EBBS as it learns from the ensemble knowledge.

# Appendix — https://arxiv.org/abs/2403.00144 GitHub — https://github.com/MANGA-UOFA/EBBS

# 1 Introduction

Machine translation is a widely applicable NLP task that aims to translate a text from a source language to a target language (Brown et al. 1990; Bahdanau, Cho, and Bengio 2015). The Transformer architecture (Vaswani et al. 2017) and pretrained large language models (Radford et al. 2019; Lewis et al. 2020) have largely improved translation performance, especially in the supervised setting (Raffel et al. 2020), where a model can learn from large volumes of parallel corpora. However, machine translation remains challenging for low-resource languages, because there are not enough data for large neural networks to learn these languages (Radford et al. 2019; Muennighoff et al. 2023).

We specifically focus on multilingual translation in the zero-shot setting, where the system is required to translate between unseen language pairs. Since collecting parallel data and training individual models for every translation pair are prohibitively expensive, it is common to build a single multilingual system (Johnson et al. 2017; Fan et al. 2021) that can perform translation for all language pairs, most of which are zero-shot translation directions that do not involve a high-resource language (e.g., English). These models work by prepending a language-indicator token; the zero-shot translation ability emerges as the model generalizes from trained language pairs and is able to perform direct translation for unseen ones (Liu et al. 2021; Wicks and Duh 2022). The main drawback of such multilingual models is that they are noisy in the zero-shot setting due to the lack of supervision, and as a result, they tend to generate low-quality translations (Zhang et al. 2020; Liu et al. 2021).

Alternatively, zero-shot translation can be performed by pivoting (Wu and Wang 2007, 2009), where the model first translates the input into a high-resource language such as English, which is then translated to the target language. However, pivoting requires two translation steps, often leading to an accumulation of errors (Babych, Hartley, and Sharoff 2007; Gu et al. 2019).

In this paper, we propose an ensemble approach that aggregates direct and pivot translations in order to build a stronger multilingual translation model from weak ones. Building an ensemble for text generation is nuanced as it involves a sequence of word predictions. Word-level ensembles aggregate predictions at each generation step, which is usually achieved by averaging the predicted probabilities (Sennrich, Haddow, and Birch 2016a; Freitag, AlOnaizan, and Sankaran 2017; Shanbhogue et al. 2023). This may not be ideal for zero-shot translation as the predictions are too noisy, making the averaged probabilities overly smooth. On the other hand, minimum Bayes risk decoding (MBR) (Bickel and Doksum 2015) can be considered a sequence-level voting ensemble, but existing MBR methods are only able to select from weak and noisy candidates given by the direct and pivot translations.

To this end, we propose an ensemble decoding algorithm with bi-level beam search (EBBS). Our EBBS performs two levels of beam search at each generation step: at the lower level, beam search is applied individually to each ensemble component; at the upper level, the ensemble maintains a shared beam by voting and synchronizing the candidates (sub-sequences) in lower-level beams. Unlike wordlevel ensembles (Freitag, Al-Onaizan, and Sankaran 2017;

Shanbhogue et al. 2023), EBBS does not average the predicted distributions, encouraging individual predictors to explore their own preferences; unlike sequence-level MBR ensembles (Kobayashi 2018; Eikema and Aziz 2020), EBBS does not select from a candidate set, and thus is more flexible since votings are performed throughout the generation process.

We conducted experiments on IWSLT (Cettolo et al. 2017) and Europarl (Koehn 2005), two popular multilingual datasets for zero-shot machine translation. Results show that EBBS can generate high-quality translations and outperform existing ensemble techniques. In addition, we used EBBSgenerated data for distillation to further improve the multilingual model. The experiment shows that such a distilling process encourages the model to learn from high-quality translations produced by EBBS, allowing it to outperform EBBS with no inference overhead compared with direct translation.

# 2 Related Work

Machine translation. In NLP, machine translation is a longstanding task that aims to rewrite text from one language to another without changing the meaning. Traditional research in translation has been mainly centered on the supervised setting, utilizing manually crafted rules (Forcada et al. 2011; Dugast, Senellart, and Koehn 2007) and statistical methods (Brown et al. 1990; Koehn 2009); more recently, neural machine translation systems have considerably improved the performance (Vaswani et al. 2017; Raffel et al. 2020). However, translation remains challenging for low-resource languages, where neural models do not have enough parallel data to train on.

Translation for low-resource languages largely relies on zero-shot techniques, where no parallel text is available for a particular translation direction. In general, zero-shot translation can be accomplished in a monolingual or multilingual setting. With monolingual data, the most common approach is to build language-specific autoencoders that share the same latent space of semantics; translation is then achieved by plugging in the decoder of the desired language (Lample et al. 2018a,b; Mohiuddin and Joty 2020).

In this paper, we focus on the multilingual setting, where one model can translate between multiple languages (Dabre, Chu, and Kunchukuttan 2020). Usually, parallel texts only exist for a high-resource language such as English, leaving translations between low-resource languages zero-shot (e.g., Italian to Dutch) (Johnson et al. 2017; Fan et al. 2021). In this setting, the most common approach is to train the multilingual model on English-centric data, and the zero-shot translation ability naturally emerges during the training process (Johnson et al. 2017; Scao et al. 2022).

A key challenge for multilingual models is task interference, where too many languages tend to degrade model performance (Zaremoodi, Buntine, and Haffari 2018; Wang, Lipton, and Tsvetkov 2020). As a result, research in this direction has been alleviating such interference by developing various parameter-separation schemes (Baziotis et al. 2022; Chronopoulou, Stojanovski, and Fraser 2023) and using gradient-based methods to update language-specific parameters (Wang and Zhang 2022; He et al. 2023). In our work, we use a standard Transformer model following Johnson et al. (2017) and Liu et al. (2021). Our proposed ensemble algorithm EBBS is compatible with the above approaches, as it is agnostic to model architectures.

Ensemble methods. In a model ensemble, multiple machine learning systems are integrated so as to form a stronger one (Dong et al. 2020; Yang, Lv, and Chen 2023). Bagging, a classic ensemble technique, works by training multiple models with different portions of data and combining their predictions through averaging or voting (Breiman 1996; Bu¨hlmann and $\mathrm { Y u } ~ 2 0 0 2 \AA ,$ ). Another popular ensemble approach is boosting, where different models are trained sequentially, with each subsequent model addressing the mistakes of the previous ones (Schapire 2003; Hastie et al. 2009; Natekin and Knoll 2013). Unfortunately, bagging and boosting are not compatible with our setting, because we build an ensemble with a single model. Alternatively, stacking combines the outputs by training a meta-model (Wolpert 1992; Ganaie et al. 2022), but this does not apply to our zeroshot setting either because we do not have groundtruth signals to train the meta-model. Even though these ensemble techniques may be directly applied to supervised generation (Freitag, Al-Onaizan, and Sankaran 2017; Kobayashi 2018; Hendy et al. 2021), they are not ideal as they do not take advantage of structured prediction. Our recent work has addressed the ensemble of tree structures (Shayegh et al. 2024; Shayegh, Wen, and Mou 2024; Shayegh et al. 2025), and in this paper we focus on text generation.

Unlike previous work, our EBBS performs bi-level beam search, exploring different components’ own predictions and synchronizing them by a “soft voting” mechanism at every step. Our approach is specifically suited to the sequence generation process.

# 3 Approach

In this section, we first explain our ensemble components in $\ S 3 . 1$ . In $\ S 3 . 2$ , we propose EBBS, a novel ensemble decoding algorithm. Finally, we describe in $\ S 3 . 3$ knowledge distillation with EBBS-decoded outputs for efficiency considerations.

# 3.1 Ensemble Components

In this work, we focus on zero-shot multilingual machine translation, which requires a system to perform translations for multiple languages, where some translation directions are unseen.

Specifically, our multilingual model is an encoder– decoder Transformer with a byte pair encoding tokenizer (Sennrich, Haddow, and Birch 2016b) shared among all languages. The encoder can capture the semantics of tokens in different languages, whereas the decoder translates the encoded text into the desired language based on a targetlanguage indicator token (Johnson et al. 2017; Fan et al. 2021).

We follow the standard English-centric training (Johnson et al. 2017; Liu et al. 2021), where the multilingual model is trained using parallel data with English on one side (e.g., German-to-English and Englishto-Romanian). As mentioned in $\ S 1$ , the zero-shot ability emerges during such training, and the model is able to perform direct translation between unseen language pairs (e.g., German-to-Romanian) (Dabre, Chu, and Kunchukuttan 2020; Ranathunga et al. 2023). An alternative approach is pivot translation, where the multilingual model performs two translations using a high-resource language as a pivot (e.g., first translating German to English, and then English to Romanian).

![](images/960dbf800f6f494c83406182966ef332a3fcf8b0bfd745db543bc210c895000a.jpg)  
Figure 1: Illustration of our EBBS algorithm.

However, both direct and pivot translations have major weaknesses: the quality of direct translation tends to be low due to the lack of parallel data, whereas pivot translation suffers from error accumulation as it requires two translation steps (Babych, Hartley, and Sharoff 2007; Gu et al. 2019).

In this paper, we would like to build an ensemble of direct and pivot translations to boost translation quality, where each translation path results in an ensemble component. Commonly used ensemble methods such as averaging and voting may not work well for text generation. Voting, for example, chooses the most voted prediction, but in text generation, the components’ votes often do not share anything in common, because there could be tens of thousands of tokens in the vocabulary. An averaging ensemble, on the other hand, averages the predicted distributions of all components, potentially leading to an overly smooth distribution. Despite early success by Razmara and Sarkar (2013), more recent studies report marginal or negative improvement for multipivot averaging ensemble (Fan et al. 2021; Gaikwad et al. 2024; Mohammadshahi, Vamvas, and Sennrich 2024).

# 3.2 Our Proposed EBBS Algorithm

We propose an ensemble with bi-level beam search (EBBS), a novel decoding algorithm that enables different ensemble components to collaborate and vote on each other’s partial generations with two levels of beam search.

At the lower level, each ensemble component performs beam search individually, exploring its own preferred regions of the sentence space. At the upper level, EBBS synchronizes the lower-level beam candidates through a voting mechanism, only keeping the most promising partial generations in a shared, upper-level beam. This allows the ensemble components to vote out spurious partial candidates and improve zero-shot translation performance.

Concretely, we assume there are $K$ ensemble components $p _ { 1 } , \cdots , p _ { K }$ , each predicting the probability of the next word given some prefix.

For the 0th decoding step, EBBS initializes the upperlevel beam by $\overline { { B } } _ { 0 } = \left. \mathrm { { { \bar { B } } o s , \bar { 1 } } } \right.$ , suggesting that a sequence is forced to start with a special beginning-of-sequence token BOS with probability 1.

For step $t$ , each ensemble component performs lowerlevel beam search individually, based on the prefixes in the last step’s shared beam $\overline { { B } } _ { t - 1 }$ :

$$
\begin{array} { r l } & { \underline { { B } } _ { t , k } = \mathrm { t o p } { - } Z \{ \ \langle \mathbf { y } _ { 1 : t - 1 } \oplus \mathbf { y } , \ p \cdot p _ { k } ( \mathbf { y } | \mathbf { y } _ { 1 : t - 1 } , \mathbf { x } ) \rangle : } \\ & { \qquad \langle \mathbf { y } _ { 1 : t - 1 } , p \rangle \in \overline { { B } } _ { t - 1 } , \ \mathbf { y } \in V \ \} } \end{array}
$$

for $k = 1 , \cdots , K$ . Here, top- $Z$ selects $Z$ -many sequences with the highest probabilities, $\oplus$ represents string concatenation, $V$ is the vocabulary, and $p _ { k } \left( \mathrm { y } | \mathbf { y } _ { 1 : t - 1 } , \mathbf { x } \right)$ is the $k$ th ensemble component’s predicted probability at step $t$ given the prefix $\mathbf { y } _ { 1 : t - 1 }$ and input $\mathbf { x }$ .

At the upper level, EBBS synchronizes the lower-level individual beams $\underline { { B } } _ { t , k }$ , for $k = 1 , \cdots , K$ , into a shared, upper-level beam through a soft-voting mechanism, where the candidate set $C _ { t }$ is the union of the sequences in lowerlevel beams:

$$
C _ { t } = \bigcup _ { k } \{ \mathbf { y } : \langle \mathbf { y } , p \rangle \in \underline { { B } } _ { t , k } \}
$$

We evaluate each candidate in $C _ { t }$ and compute its overall vote as the sum of the probabilities.

$$
\overline { { B } } _ { t } = \mathrm { t o p } { \cal Z } \left\{ \left\{ { \bf y } , \sum _ { { \bf \alpha } _ { \in ^ { { \scriptstyle \alpha } _ { t , \cdot } \cdot \cdot \cdot \cdot } , { \cal K } } \atop { \langle { { \bf y } ^ { \prime } } , p \rangle \in \underline { { B } } _ { t , k } : { \bf \alpha } _ { \cdot } \cdot { \bf { y } ^ { \prime } } = { \bf y } } } p \right\} : { \bf y } \in C _ { t } \right\}
$$

In this way, the upper level synchronizes all ensemble components with the shared beam $\overline { { B } } _ { t }$ for the next step of generation.

Intuitively, our voting scheme gives an ensemble component $Z$ -many votes, each weighted by the predicted probability. The votes (probabilities) are then tallied (summed) for each candidate to form the upper-level beam. Our bi-level beam search terminates when we have $Z$ -many terminated sequences in the shared beam, and returns the sequence with the highest score1 as the ensemble output. We provide the detailed pseudocode for EBBS in Algorithm 1 and an illustration in Figure 1.

Discussion. Traditional beam search keeps a fixed-size beam of high-likelihood partial sequences. To build an enaverage their probabilities semble with multiple predictors, it is tempting to directly $\begin{array} { r } { p ( \mathbf { y } | \mathbf { x } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } p _ { k } ( \mathbf { y } | \mathbf { x } ) } \end{array}$ as

Input: x: input sentence; $Z$ : beam size   
$K$ : number of scorers; $p _ { 1 } , \cdots , p _ { K }$ : scorers   
1 $H \gets \emptyset$ $D$ candidate outputs   
2 $\overline { { B } } _ { 0 } \gets \{ \langle \mathrm { B O S } , 1 \rangle \}$ $D$ upper-level beam   
3 for $t = 1 , 2 , \cdots { \mathrm { ~ d o } }$   
4 ▷ lower: individual beam search   
5 for $\langle \mathbf { y } _ { 1 : t - 1 } , p \rangle \in \overline { { B } } _ { t - 1 }$ do   
6 for $k = 1 , \cdots , K$ do   
7 $\underline { { B } } _ { t , k } \gets \emptyset$ ▷ lower-level beam   
8 for y V do   
9 p′ ← pk(y|y1:t 1, x)   
10 Bt,k. add(⟨y1:t 1 ⊕ y, p · p′⟩)   
11 Bt,k Bt,k. top(Z)   
12 ▷ upper: beam synchronization   
13 $D \gets$ empty dictionary   
14 for $k = 1 , \cdots , K$ do   
15 for $\langle \mathbf { y } , p \rangle \in \underline { { B } } _ { t , k }$ do   
16 if $\mathbf { y } \in D$ then   
17 D[y] ← p + D[y]   
18 else   
19 D[y] ← p   
20 Bt D. top(Z)   
21 $D$ check for termination   
22 for $\langle \mathbf { y } , p \rangle \in \underline { { B } } _ { t }$ do   
23 if $\mathrm { y } _ { t } = \mathrm { E O S }$ then   
24 $H . \operatorname { a d d } ( \langle \mathbf { y } , p \rangle )$   
25 if H = Z then   
26 return H. top(1)

the score for beam search, which has been experimented in previous work (Sennrich, Haddow, and Birch 2016a; Shanbhogue et al. 2023).

However, our intuition suggests that such an approach may suffer from the over-smoothing problem (Wei et al. 2019; Wen et al. 2023b): when multiple translations (known as modes) are plausible given an input, the ensemble process will overly smooth out the modes by probability averaging.

By contrast, EBBS allows each ensemble component to explore its own mode (Lines 4–11, Algorithm 1). In Figure 1, for example, the top sequence yields two plausible next tokens, suggested by each component in the lower level; their probabilities are not smoothed out in our approach, unlike averaging ensembles. The upper level performs soft voting (Lines 12–19, Algorithm 1) so as to maintain tractable inference.

# 3.3 EBBS-Based Distillation

To improve inference efficiency, we perform knowledge distillation based on the outputs of our EBBS algorithm. In particular, we follow (Kim and Rush 2016) and apply a sequence-level knowledge distillation loss, treating the output $\hat { \mathbf { y } }$ of our ensemble (serving as a teacher) as the pseudogroundtruth for finetuning the multilingual translation model (serving as a student):

$$
\mathcal { L } _ { \mathrm { K D } } = - \sum _ { t = 1 } ^ { | \hat { \mathbf { y } } | } \log p ( \hat { \mathbf { y } } _ { t } | \hat { \mathbf { y } } _ { 1 : t - 1 } , \mathbf { x } )
$$

Our distilling method is an ensemble-then-distill process. This differs from a straightforward practice of multi-teacher distillation, where the student learns from the union of teachers’ outputs (Wu, Wu, and Huang 2021). The commonly applied cross-entropy loss is known to yield overly smooth distributions (Wen et al. 2023a,b), and the problem becomes more severe with multiple teachers, leading to less satisfactory performance of union distillation (Shayegh et al. 2024). On the contrary, our approach provides the student with a consolidated pseudo-groundtruth translation, causing less confusion during the distillation process especially when teachers disagree.

# 4 Experiments

# 4.1 Settings

We evaluated EBBS on two popular benchmark datasets for zero-shot machine translation: IWSLT (Cettolo et al. 2017), which contains 4 languages (with English) and 6 zero-shot directions; and Europarl v7 (Koehn 2005), which contains 9 languages and 56 zero-shot directions.

We used BLEU scores (Papineni et al. 2002) (in particular, SacreBLEU (Post 2018)) as our main evaluation metric,2 which is one of the most widely used metrics for translation (Fan et al. 2021; Scao et al. 2022). For in-depth analyses, we further adopted other popular translation metrics, including the character-level $n$ -gram F score $( \mathrm { c h r F } 2 + + )$ (Popovic´ 2017), the translation edit rate (TER) (Snover et al. 2006), and a more recent, neural network-based metric called COMET (Rei et al. 2020).

We replicated (Liu et al. 2021) and trained a multilingual translation system as our base model. Specifically, the neural architecture in (Liu et al. 2021) is a 5-layer encoder–decoder Transformer for IWSLT, but has 8 layers for Europarl to accommodate more training data and languages.

For EBBS, we used a beam size of five for both upperand lower-level beams. In our experiment, we implemented standard beam search for comparison, where we also used a beam size of five, following the common practice (Meister, Cotterell, and Vieira 2020). A comprehensive beam analysis can be found in our appendix.

# 4.2 Competing Methods

We comprehensively compare our EBBS with direct/pivot translation and other ensemble methods.

Direct/pivot translation. For direct translation, we applied beam search on the multilingual model to translate in unseen directions. For pivot translation (Wu and Wang 2007, 2009; Vamvas and Sennrich 2022), we used English as the pivot because we have parallel data for translations both from and to English.

Word-level averaging ensemble. Averaging is one of the most widely used ensemble techniques in text generation (Sennrich, Haddow, and Birch 2016a; Freitag, AlOnaizan, and Sankaran 2017; Shanbhogue et al. 2023). Essentially, the ensemble components’ probabilities are first averaged before being fed to the standard beam search.

Word-level voting ensemble. The voting ensemble, common in classification tasks, picks the output class based on the number of votes from ensemble components (given by argmax). However, voting is not common in text generation, because argmax may select completely different words by the ensemble components due to the large vocabulary size, making voting ineffective. As a remedy, we pick the word by the highest probability when there is a tie for votes.

Sequence-level voting ensemble. Minimum Bayes risk (MBR) decoding is originally designed as a single-model decoding algorithm, where it selects a sequence from a set of beam search results based on similarity (Eikema and Aziz 2020; Mu¨ller and Sennrich 2021). Here, we use it as a sequence-level ensemble technique, where the candidates are the output sequences from different ensemble components. Let $\bar { \cal C } = \{ { \bf \bar { y } } _ { 1 } , \cdot \cdot \cdot , { \bf y } _ { K } \}$ be the set of candidate outputs given by $K$ ensemble components. The best output is selected as

$$
\mathbf { y } ^ { * } = \underset { \mathbf { y } \in C } { \mathrm { a r g m a x } } \sum _ { \mathbf { y } ^ { \prime } \in C \setminus \{ \mathbf { y } \} } \mathrm { B L E U } ( \mathbf { y } , \mathbf { y } ^ { \prime } )
$$

where ${ \bf B L E U ( h , r ) }$ computes the BLEU score between a hypothesis $\mathbf { h }$ and a reference r. In essence, MBR selects an output that resembles others most, using BLEU as the similarity metric.

# 4.3 Results and Analysis

Main results. Our experiment starts by a replication of the base multilingual model (Liu et al. 2021). As shown in Rows 1–2, Table 1, the results are generally close, indicating that our replication is successful and ready for ensemble research. Further, we tried English pivoting (Row 3), a common zero-shot translation method. In our experiments, we find that it does not outperform direct translation, as pivoting methods may suffer from the error accumulation problem due to two-step translation.

We then compare different ensemble techniques, including our proposed EBBS. We notice that IWSLT contains four languages (with English); thus we have two available pivoting directions (excluding source and target), which, along with direct translation, are our three ensemble components. For Europarl, it contains nine languages; for performance and efficiency concerns (to be shown in Figure 2), we also consider three translation paths as our ensemble components: direction translation, English pivoting, and a second pivot.3

We study the common ensemble technique of word-level averaging (Row 4), which has been used in previous translation research (Freitag, Al-Onaizan, and Sankaran 2017). As we can see, the averaging ensemble performs similarly to direct translation on both datasets. Our zero-shot results are different from (Freitag, Al-Onaizan, and Sankaran 2017), which shows a word-level averaging ensemble of random seeds can improve performance in the supervised setting. This is because models trained with different random seeds exhibit similar behavior, and averaging their probabilities achieves a denoising effect. However, our ensemble components differ drastically in terms of their strengths and expertise due to the different translation paths (direct and pivot translations). Thus, word averaging fails to improve translation quality in our setting.

Alternatively, voting ensembles can also be applied, at either the word level or the sequence level. As seen, word-level voting is not effective, as it is worse than direct translation on both datasets (Row 5). This is expected because the voted words (top predictions) by the ensemble components may not overlap due to the large vocabulary size. In such cases, the algorithm defaults to choosing the word with the highest probability, causing the ensemble to follow the most peaked distributions.

Sequence-level voting should also be done in a soft manner, and minimum Bayes risk (MBR) decoding can be thought of as using a Bayes risk to softly “vote” the candidate outputs. As seen from Row 6, such a method works relatively well on Europarl, achieving the secondhighest performance across all ensemble methods; however, it works poorly on the IWSLT dataset. The main drawback of sequence-level voting is that it can only select one of the ensemble components’ output. This may not work well when the individual ensemble components are weak, especially with the small IWSLT dataset. Such a selective sequence-level ensemble cannot integrate different expertise of its components during generation.

Unlike existing ensemble methods, our EBBS algorithm achieves higher performance in most directions on both datasets. Noticing that Europarl contains 56 zero-shot directions, we could only present in Table 1 the first seven directions based on the order provided by the dataset, due to the space limit. Table 2 further shows a pairwise comparison against direct translation (a strong baseline in our experiment) in all zero-shot directions. As seen, EBBS achieves higher performance in 56 out of 62 cases across two datasets, showing strong statistical evidence for its effectiveness, with a $p$ -value of 3e-11 in a two-sided binomial test.

We also evaluate EBBS-based distillation (Row 8, Table 1). Again, since Europarl has 56 zero-shot directions, we follow the standard practice (Fan et al. 2021) and select a subset of directions, namely, Danish to other languages, to save computational cost. As seen in Row 8, EBBS-based distillation consistently achieves the highest performance in all directions (except for Danish-to-Dutch translation). This shows that an EBBS-distilled model can outperform EBBS, which is not surprising because learning can smooth out the noise of various heuristics (Deshmukh et al. 2021; Jolly et al. 2022), such as the ensemble algorithm in our scenario. Importantly, EBBS-based distillation achieves significantly higher translation quality with no inference overhead compared with direct translation.

<html><body><table><tr><td></td><td># Method</td><td></td><td>Average</td><td>it-nl</td><td>it-ro</td><td>nl-it</td><td>nl-ro</td><td>ro-it</td><td>ro-nl</td><td rowspan="8"></td></tr><tr><td rowspan="8">IWSLT</td><td>1</td><td>Direct translation (Liu et al.2021)†</td><td>17.7</td><td>18.5</td><td>17.8</td><td>17.9</td><td>15.5</td><td>19.6</td><td>16.8</td></tr><tr><td>2</td><td>Direct translation (our replication)</td><td>17.29</td><td>17.46</td><td>17.48</td><td>18.23</td><td>14.63</td><td>19.65</td><td>16.26</td></tr><tr><td>3</td><td>Pivoting (en)</td><td>16.19</td><td>17.49</td><td>15.09</td><td>16.79</td><td>13.05</td><td>18.34</td><td>16.37</td></tr><tr><td>4</td><td>Word-level averaging ensemble</td><td>17.28</td><td>17.29</td><td>17.44</td><td>18.33</td><td>14.65</td><td>19.69</td><td>16.30</td></tr><tr><td>5</td><td>Word-level voting ensemble</td><td>16.99</td><td>17.58</td><td>16.38</td><td>17.78</td><td>14.13</td><td>19.21 16.84</td><td></td></tr><tr><td>6</td><td>Sequence-level voting ensemble (MBR)</td><td>16.72</td><td>16.64</td><td>16.53</td><td>17.83</td><td>13.74</td><td>19.48</td><td>16.08</td></tr><tr><td>7</td><td>EBBS (ours)</td><td>18.24</td><td>19.52</td><td>17.09</td><td>19.06</td><td>14.58</td><td>20.75</td><td>18.45</td></tr><tr><td>8</td><td>Direct w/EBBS distillation (ours)</td><td>18.92</td><td>19.86</td><td>18.80</td><td>19.73</td><td>15.39</td><td>21.23</td><td>18.48</td></tr><tr><td rowspan="11">Europarl</td><td># 1</td><td>Method</td><td>Average</td><td>da-de</td><td>da-es</td><td>da-fi</td><td>da-fr</td><td>da-it</td><td>da-nl</td><td>da-pt</td></tr><tr><td>2</td><td>Direct translation (Liu et al. 2021)†</td><td>26.9</td><td>24.2</td><td>33.1</td><td>18.1</td><td>30.6</td><td>26.1</td><td>26.3</td><td>29.9</td></tr><tr><td></td><td>Direct translation (our replication)</td><td>27.74</td><td>26.24</td><td>33.64</td><td>18.95</td><td>31.01</td><td>26.58</td><td>27.36</td><td>30.38</td></tr><tr><td>3</td><td>Pivoting (en)</td><td>27.69</td><td>25.17</td><td>33.87</td><td>18.70</td><td>31.44</td><td>27.12</td><td>26.75</td><td>30.79</td></tr><tr><td>4</td><td>Word-level averaging ensemble</td><td>27.76</td><td>26.13</td><td>33.72</td><td>18.91</td><td>31.01</td><td>26.67</td><td>27.39</td><td>30.50</td></tr><tr><td>5</td><td>Word-level voting ensemble</td><td>27.45</td><td>25.76</td><td>33.24</td><td>18.39</td><td>30.96</td><td>26.83</td><td>26.63</td><td>30.37</td></tr><tr><td>6</td><td>Sequence-level voting ensemble (MBR)</td><td>27.90</td><td>25.90</td><td>33.95</td><td>19.15</td><td>31.50</td><td>27.15</td><td>27.09</td><td>30.55</td></tr><tr><td>7</td><td>EBBS (ours)</td><td>28.36</td><td>26.32</td><td>34.28</td><td>19.43</td><td>31.97</td><td>27.67</td><td>27.78</td><td>31.08</td></tr><tr><td>8</td><td>Direct w/ EBBS distillation (ours)</td><td>28.54</td><td>26.75</td><td>34.68</td><td>19.89</td><td>32.00</td><td>27.69</td><td>27.61</td><td>31.19</td></tr></table></body></html>

Table 1: Main results of BLEU scores on IWSLT and Europarl. The best results are in bold; the second best results are underlined. † indicates cited results; others were obtained by our experimentation.

Table 2: Pairwise comparison on all 62 zero-shot directions in both datasets. The $p$ -value is given by a two-sided binomial test.   

<html><body><table><tr><td>Dataset</td><td>Method</td><td>Avg.BLEU</td><td>Wins</td><td>Losses</td></tr><tr><td rowspan="2">IWSLT</td><td>Direct translation</td><td>17.29</td><td>2</td><td>4</td></tr><tr><td>EBBS (ours)</td><td>18.24</td><td>4</td><td>2</td></tr><tr><td>Europarl</td><td>Diresc oussation</td><td>27.85</td><td>452</td><td>524</td></tr><tr><td>Overall</td><td>Direct translation EBBS (ours)</td><td>26.83 27.45</td><td>6 56</td><td>56 6</td></tr><tr><td>p-value</td><td colspan="3">3e-11</td><td></td></tr></table></body></html>

Distillation analysis. We compare EBBS-based distillation with other distilling methods. Here, we only focus on Italian-to-Dutch4 translation to save computational cost.

In particular, we consider two alternative distilling methods: direct and union distillation. Direct distillation finetunes the multilingual model with its own predictions based on direct translation. Union distillation, on the other hand, takes the union of the teachers’ outputs (direct and pivot translations) for training, which is under a controlled experimental setup, because it uses exactly the same translation paths as our EBBS-based distillation.

As seen in Table 3, both direct and union distillation marginally improve the performance compared with no distillation. Intriguingly, learning from the union of multiple teachers is not necessarily better than learning from the best teacher (namely, direct translation). This is because multiple teachers may provide conflicting training signals and confuse the student model.

On the contrary, our EBBS-based distillation consistently outperforms direct and union distillation on both datasets. This shows that our ensemble-then-distill approach is able to consolidate the knowledge of multiple teachers to better train the student model.

Further, the analysis suggests that our EBBS-distilled model achieves a speedup of multiple times compared with EBBS, because after distillation the model is used by direct translation. This is a significant result, because our EBBSbased distillation not only speeds up the EBBS ensemble approach, but also improves the translation quality of EBBS as shown in Row 8, Table 1.

Analysis of ensemble components. In Table 4, we analyze the ensemble components to better understand our ensemble technique for zero-shot machine translation. As seen, direct translation is an effective approach, which is consistent with previous literature (Fan et al. 2021; Liu et al. 2021). English pivoting achieves higher performance for some metrics but lower for others; it is not conclusively better than direct translation, probably because of the error accumulation problem. Pivoting through non-English languages degrades the performance to a large extent because lacking supervision along the pivoting path leads to two steps of zero-shot translation. EBBS, on the other hand, combines the strengths of individual components and consistently outperforms them in all metrics.

We further study how EBBS performs with different numbers of ensemble components. Specifically, we analyze two incremental ensemble settings: best-to-worst and worst-tobest. In both cases, we start with direct translation; then we incrementally add the next “best” or “worst” pivot translation according to Table 4.

<html><body><table><tr><td>Dataset</td><td colspan="2">Method</td><td>BLEU↑</td><td>BLEU1↑</td><td>BLEU2↑</td><td>BLEU3↑</td><td>BLEU4↑</td><td>chrF2++↑</td><td>TER</td><td>COMET↑</td></tr><tr><td rowspan="4">IWSLT</td><td rowspan="4">EBBS Direct</td><td></td><td>19.52</td><td>51.87</td><td>25.12</td><td>13.88</td><td>8.02</td><td>45.63</td><td>71.36</td><td>0.7341</td></tr><tr><td>No distillation</td><td>17.46</td><td>50.49</td><td>23.01</td><td>12.01</td><td>6.66</td><td>43.73</td><td>72.02</td><td>0.7088</td></tr><tr><td>Direct distillation</td><td>18.10</td><td>50.37</td><td>23.53</td><td>12.63</td><td>7.17</td><td>44.48</td><td>72.86</td><td>0.7144</td></tr><tr><td>Union distillation</td><td>17.80</td><td>49.21</td><td>23.01</td><td>12.51</td><td>7.10</td><td>44.93</td><td>75.92</td><td>0.7221</td></tr><tr><td rowspan="5">Europarl Translation</td><td rowspan="5">EBBS Direct</td><td>EBBS distillation</td><td>20.13</td><td>53.20</td><td>26.06</td><td>14.33</td><td>8.26</td><td>46.46</td><td>69.28</td><td>0.7428</td></tr><tr><td></td><td>26.10</td><td>57.07</td><td>31.00</td><td>19.76</td><td>13.28</td><td>52.75</td><td>65.63</td><td>0.8340</td></tr><tr><td>No distillation</td><td>25.33</td><td>56.32</td><td>30.08</td><td>19.01</td><td>12.78</td><td>52.32</td><td>66.56</td><td>0.8276</td></tr><tr><td>Direct distillation</td><td>25.44</td><td>56.54</td><td>30.28</td><td>19.13</td><td>12.79</td><td>52.61</td><td>66.34</td><td>0.8286</td></tr><tr><td>Union distillation EBBS distillation</td><td>25.53 25.92</td><td>56.58 56.76</td><td>30.34 30.68</td><td>19.18 19.57</td><td>12.91 13.24</td><td>52.63 52.73</td><td>66.27 66.04</td><td>0.8282 0.8307</td></tr></table></body></html>

Table 3: Comparison of various distilling methods for Italian-to-Dutch translation. $\uparrow / \downarrow$ The higher/lower, the better.

Table 4: The performance of direct/pivot translation and our EBBS for Italian-to-Dutch translation on Europarl.   

<html><body><table><tr><td>Method</td><td>BLEU↑</td><td>chrF2++↑</td><td>TER</td><td>COMET↑</td></tr><tr><td>Direct translation</td><td>25.33</td><td>52.32</td><td>66.56</td><td>0.8276</td></tr><tr><td>Pivoting (en)</td><td>25.08</td><td>51.92</td><td>66.24</td><td>0.8322</td></tr><tr><td>Pivoting (es)</td><td>24.40</td><td>51.71</td><td>67.91</td><td>0.8192</td></tr><tr><td>Pivoting (pt)</td><td>24.34</td><td>51.61</td><td>67.68</td><td>0.8191</td></tr><tr><td>Pivoting (fr)</td><td>24.20</td><td>51.61</td><td>67.84</td><td>0.8208</td></tr><tr><td>Pivoting (de)</td><td>23.65</td><td>50.70</td><td>67.89</td><td>0.8157</td></tr><tr><td>Pivoting (da)</td><td>23.12</td><td>50.36</td><td>69.00</td><td>0.8156</td></tr><tr><td>Pivoting (fi)</td><td>20.74</td><td>48.11</td><td>70.59</td><td>0.8051</td></tr><tr><td>Our EBBS</td><td>26.10</td><td>52.75</td><td>65.63</td><td>0.8340</td></tr></table></body></html>

![](images/9db11490a650909569c686263d5bad654444ffce1b2a3a1f21f1aa8159dd5080.jpg)  
Figure 2: Analysis of the number of ensemble components for Italian-to-Dutch translation on Europarl.

Figure 2 shows the trends of incremental ensembles. If we add the best pivot directions, the performance peaks at three ensemble components; interestingly, the inclusion of weaker components does not affect EBBS much. On the other hand, adding the worst pivot translation at the beginning leads to an immediate drop of 1.6 BLEU points, which then largely recovers with the second pivot. This is reasonable because the worst pivot (Finnish) is 4.6 BLEU points lower than direct translation, and EBBS cannot decide on which of the two ensemble components to trust; despite this, the performance of EBBS is still much better than the average performance of the components. With a second pivot, there is a third “opinion” when the first two components “disagree.” The performance continues to rise if more and stronger components are added. In fact, our ensemble even surpasses the baseline with 4 weakest pivot translations, each of which is at least 1 BLEU point lower than the baseline. This demonstrates that EBBS is flexible and works well with both strong and weak ensemble components.

Appendix. The full version of the paper is available at https://arxiv.org/abs/2403.00144, where we present additional details and results in the appendix:

A. Beam search,   
B. Experimental details,   
C. Analysis of inference efficiency,   
D. Average performance across tasks,   
E. Analysis of beam size,   
F. Entropy of distilled models,   
G. Analysis of voting methods in EBBS, and H. Case study.

# 5 Conclusion

In this work, we address ensemble-based zero-shot machine translation by directly translating and pivoting through different languages. We further design a novel bi-level beam search algorithm (called EBBS) for decoding. We evaluated EBBS on two popular zero-shot translation datasets, IWSLT and Europarl. Results show that EBBS outperforms existing ensemble techniques, and that the high-quality translations produced by EBBS can be used for distillation to improve translation efficiency (and sometimes also output quality).