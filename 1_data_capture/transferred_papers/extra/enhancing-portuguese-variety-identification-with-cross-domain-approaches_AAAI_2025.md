# Enhancing Portuguese Variety Identification with Cross-Domain Approaches

Hugo Sousa1,3\*, Ru´ben Almeida $^ { 2 , 3 , 4 * }$ , Purifica¸c˜ao Silvano5,6, Inˆes Cantante5,6, Ricardo Campos3, 7,8, Al´ıpio Jorge1,3

1Faculty of Sciences, University of Porto, Porto, Portugal 2Faculty of Engineering, University of Porto, Porto, Portugal 3INESC TEC, Porto, Portugal 4Innovation Point - dst group, Braga, Portugal 5Faculty of Arts and Humanities, University of Porto, Porto, Portugal 6Centre of Linguistics, University of Porto, Porto, Portugal 7Department of Informatics, University of Beira Interior, Covilha˜, Portugal 8Ci2 - Smart Cities Research Center, Tomar, Portugal hugo.o.sousa@inesctec.pt, ruben.f.almeida@inesctec.pt

# Abstract

Recent advances in natural language processing have raised expectations for generative models to produce coherent text across diverse language varieties. In the particular case of the Portuguese language, the predominance of Brazilian Portuguese corpora online introduces linguistic biases in these models, limiting their applicability outside of Brazil. To address this gap and promote the creation of European Portuguese resources, we developed a cross-domain language variety identifier (LVI) to discriminate between European and Brazilian Portuguese. Motivated by the findings of our literature review, we compiled the PtBrVarId corpus, a cross-domain LVI dataset, and study the effectiveness of transformer-based LVI classifiers for cross-domain scenarios. Although this research focuses on two Portuguese varieties, our contribution can be extended to other varieties and languages. We open source the code, corpus, and models to foster further research in this task.

# 1 Introduction

Discriminating between varieties of a given language is an important natural language processing (NLP) task (Joshi et al. 2024). Over time, populations that share a common language can evolve distinctive speech traits due to geographical and cultural factors, including migration and the influence of other languages (Raposo, Vicente, and Veloso 2021). Recently, this importance became even more pronounced with the advent of variety-specific large language models, where variety discrimination plays a pivotal role (Rodrigues et al. 2023). Whether in the pre-training, fine-tuning, or evaluation phase, having a highly effective system to discriminate between varieties reduces the amount of human supervision required, accelerating the production of curated mono-variety datasets (O¨ hman et al. 2023). However, developing such a system presents considerable challenges. Classifiers often struggle to identify linguistically relevant features, showing a tendency to be biased towards non-linguistic factors, such as named entities and thematic content (Diwersy, Evert, and Neumann 2014). Consequently, these classifiers exhibit limited transfer capabilities to domains not represented in the training set, significantly restricting their utility in multi-domain applications (Lui and Baldwin 2011; Nguyen et al. 2021).

A language in which variety identification is particularly challenging is Portuguese. It is spoken by more than 200 million people worldwide and serves as the official language of eight nations on five continents, each with its own variety. However, $7 0 \%$ of Portuguese speakers are Brazilian citizens1, which implies that resources labeled as Portuguese are dominated by this language variety. Another important characteristic of Portuguese is that, unlike languages where differences are predominantly phonological, such as those of the North Germanic family (Holmberg and Platzack 2008), the widespread of Portuguese has fostered considerable phonological, morphological, lexical, syntactic and semantic variations between Portuguese varieties (Brito and Lopes 2016). In the development of language models, for example, this variety divergence has practical implications; models trained in Brazilian Portuguese generate texts that are markedly distinct from those trained in other Portuguese varieties (Rodrigues et al. 2023). This fact restrains the adoption of these models outside of Brazil in domains where formal non-Brazilian text is required, as is the case of legal and medical applications. This underscores the practical importance of developing effective LVI systems that can be deployed in production.

In this study, we describe the development of a crossdomain LVI classifier that discriminates between Brazilian and European Portuguese. To accomplish that, we start with a comprehensive listing of Portuguese LVI resources. The lack of multi-domain corpora motivated us to compile our own dataset. This corpus was then used in the development of our LVI classifier. For the training procedure we devised a training protocol that takes into account the cross-domain capabilities of models during evaluation. Furthermore, we also study the impact of masking named entities and thematic content embedded in the training corpus, a process named delexicalization (Lui et al. 2014). To summarize, the contributions of this work are the following:

1. We introduce a novel cross-domain, silver-labeled LVI corpus for Brazilian and European Portuguese, compiled from open-license datasets;   
2. We examine the impact of different levels of delexicalization on the overall effectiveness of LVI models;   
3. We propose a training protocol for LVI models that yields better generalization performance;   
4. We release the first open-source Portuguese LVI model, providing a valuable resource for future research and practical applications.

The remainder of this paper is organized as follows: Section 2 offers a comprehensive literature review on the stateof-the-art in Portuguese LVI. In Section 3, we introduce our compiled dataset, PtBrVarId, and present relevant statistics along with the results of a manual evaluation of the quality of the dataset. Section 4 describes the training protocol and the models developed, including the baselines and benchmarks used for comparison. The results are presented in Section 6, followed by a discussion of future research directions in Section 7.

# 2 Related Work

# Corpora

Despite the numerous works developed in the LVI task, the first gold-labeled dataset that includes Portuguese corpora, the DSL-TL corpus (Zampieri et al. 2024), was only introduced in 2023. This dataset used crowdsourcing to annotate approximately 5k Portuguese documents. The corpus are not only labeled as “European” and “Brazilian” Portuguese, but also a special “Both or Neither” label to signal those documents with insufficient linguistic marks to be considered part of one of these varieties.

Prior to the release of this dataset, the evaluation process was often performed in silver-labeled data, collected using domain-specific heuristics. For instance, in the journalistic domain, it is common to assume the language variety of a document based on the newspaper origin; Brazilian newspapers’ articles are assigned a Brazilian Portuguese label, while Portuguese ones are assigned a European Portuguese label (Silva and Lopes 2006; Zampieri and Gebre 2012; Tan et al. 2014). In the social media domain, a similar approach is frequently used. Castro, Souza, and Oliveira (2016) used geographic metadata collected on Twitter to assign a language variety to each document based on the authors location. Unfortunately, many of these Portuguese LVI resources are no longer available online. This limitation motivated us to collect and open-source our training data.

# Modeling Approaches

The high effectiveness of N-gram-based systems observed in language identification studies (McNamee 2005; Martins and Silva 2005; Chew et al. 2009), a task closely related to LVI, motivated the application of these methods in the context of LVI. To this day, this approaches are still employed, with several submissions to the VarDial workshop2 – which compiles most of the recent studies in the LVI task – achieving high effectiveness. Notable examples include Italian with an $F _ { 1 }$ score of 0.90 (Jauhiainen, Jauhiainen, and Linde´n 2022), Uralic with an $F _ { 1 }$ score of 0.94 (BernierColborne, Leger, and Goutte 2021), and Mandarin with an $F _ { 1 }$ score of 0.91 (Yang and Xiang 2019).

The adoption of transformer-based techniques (Vaswani et al. 2017) in LVI has not been as rapid as in other NLP tasks. Recently, some studies have leveraged monolingual BERT-based models to fine-tune LVI classifiers for Romanian (Zaharia et al. 2020) and French (Bernier-Colborne, Leger, and Goutte 2022). However, in none of these cases were transformers capable of outperforming N-gram-based techniques, only achieving a $F _ { 1 }$ score of 0.65 in Romanian and 0.43 in French. Similar results have been reported for different languages using other deep learning techniques, such as multilingual transformers (Popa and Stefa˘nescu 2020), feedforward neural networks (C¸ o¨ltekin and Rama 2016; Medvedeva, Kroon, and Plank 2017), and recurrent networks (Guggilla 2016; C¸ o¨ltekin, Rama, and Blaschke 2018).

In the specific case of Portuguese, older studies have relied on N-gram-based techniques to achieve results above $9 0 \%$ accuracy on silver-labeled benchmarks (Silva and Lopes 2006; Zampieri and Gebre 2012; Goutte, Le´ger, and Carpuat 2014; Malmasi and Dras 2015; Castro, Souza, and Oliveira 2016). However, it has been noted that evaluating on silver-labeled corpora is reliability (Zampieri and Gebre 2014), and preliminary results obtained on the goldlabeled DSL-TL corpus (Zampieri et al. 2024) revealed more modest performance, with $F _ { 1 }$ scores below $7 0 \%$ . Additionally, contrary to observations in silver-labeled evaluations (Medvedeva, Kroon, and Plank 2017), the current state-of-the-art result for Portuguese LVI on the DSL-TL benchmark $\left( 0 . 7 9 \ F _ { 1 } \right)$ score) comes from fine-tuning a collection of BERT-based models (Vaidya and Kane 2023).

# Cross-Domain Capabilities

Lui and Baldwin (2011) revealed that N-grams based techniques had limited cross-domain capabilities for the language identification task. Despite the good results of Ngram-based models when the train and test domain overlap (above $8 5 \%$ accuracy), the results also show that the effectiveness decreased as much as $40 \%$ when both sets do not match. In order to address this phenomenon, the authors have devised a feature selection mechanism that later opened the door to the development of the first cross-domain language identification tool, the langid.py (Lui and Baldwin 2012).

In the context of French LVI, Diwersy, Evert, and Neumann (2014) used unsupervised learning to demonstrate that, despite the good results reported by N-grams basedmethods (above $9 5 \%$ accuracy), the feature learned by these models reveal no interest from a linguistic point of view. Instead, classifiers relied on named entities, polarity and thematics embedded in the training corpus to support its inference process (Ex: If “Cameroun” was mentioned in the document, the model assigned a French-Cameroonian label to it).

In light of these facts, the mass adoption of these architectures in the context of LVI, creates urgency for finding solutions to surpass this limitation. In this study, we extend the knowledge about the cross-domain capabilities of N-gram-based models, while presenting the first results for transformer architectures.

# 3 PtBrVarId Dataset

In this section we introduce the PtBrVarId, the first silverlabeled multi-domain Portuguese LVI corpus. This resource resulted from the compilation of open-license corpora from 11 European (EP) and Brazilian (BP) sources over six domains, namely: Journalistic, Legal, Politics, Web, Social Media and Literature. The following sections describe how the dataset was created.

# Corpora Compiled

Training machine learning and deep learning models requires a robust and well-labeled training corpus. However, manually labeling such a corpus is often laborious, timeconsuming and expensive. To address this challenge in our research, we opted for a silver labeling approach.

In the context of the VID task, silver labeling involves identifying texts where the variety can be inferred with a reasonable degree of confidence based on the documents metadata. In the following paragraphs we describe the data sources used in each textual domain along with the heuristics that supported the silver-labelling step. It is important to note that we were careful to only use sources that were permissive for academic research.

Journalistic As a source of news corpus we use two resources available at Linguateca (Santos 2014), namely: CETEMPublico (Rocha and Santos 2000) and CETEMFolha. The CETEMPublico corpus contains news articles from the Portuguese newspaper “Pu´blico” while the CETEMFolha contains news from the Brazilian newspaper “Folha de Sa˜o Paulo”. The geographic location of the newspaper is used to label the Portuguese variety.

Literature The literature domain relies on three data sources that index classics of Portuguese literature: the Gutenberg project3; the LT-Corpus (Ge´ne´reux, Hendrickx, and Mendes 2012); and the Brazilian literature corpus4. The author’s nationality was used to label the documents as European or BP.

Legal The Brazilian split from the legal corpora was compiled from RulingBR (de Vargas Feijo´ and Moreira 2018) which contains decisions from the Brazilian supreme court (“Supremo Tribunal Federal”) between 2011 to

2018. The European split was built from the DGSI website5 which provides access to a set of databases of precedents and to the bibliographic reference libraries of the Portuguese Ministry of Justice.

Politics For the politics domain we used the manual transcriptions of political speeches in both the European Parliament (Koehn 2005) and the Brazilian Senate (Cezar 2020). The document’s origin was used to infer the label for the Portuguese variety.

Web For the web domain, corpora were extracted from OSCAR (Suarez, Sagot, and Romary 2019). To define the labels, we began by identifying domains ending in .pt or .br. From this list, we manually curated a set of the 50 most frequent domains ending in .pt and 50 domains ending in .br. The documents from OSCAR associated with these curated domains were then used in our corpus.

Social Media The social media corpora derives from three data sources. For BP we used the Hate-BR (Vargas et al. 2022) dataset, which was manually annotated for train hate speech classifiers, and a compilation of fake news spread in Brazilian WhatsApp groups (da Cunha 2021). Regarding EP, the tweets collected by Ramalho (2021) were filtered based on the tweets’ metadata location. Tweets whose location is not part of Wikipedia’s list of Portuguese cities6, were discarded.

Despite the dataset proposed being silver-labeled, some of their components are extracted from high-quality manually annotated corpora that offer sufficient guarantees of belonging to a single language variety. For example, the Europarl corpus (Koehn 2005), is composed of manual transcriptions in EP of political speeches made in European Parliament, therefore it is very unlikely to find any marks of BP in such corpus.

# Data Cleaning

To reduce noise in the corpus, we implemented a dedicated data cleaning pipeline. The process starts with basic operations to remove null, empty, and duplicate entries. We then employ the clean-text tool7 to correct Unicode errors and standardize the text to ASCII format. For the Web domain, an additional step is taken using the jusText Python package8 to filter out irrelevant sentences and remove boilerplate HTML code. Finally, outliers within each domain are identified and removed based on the interquartile range (IQR) of token counts, calculated using the nltk word tokenizer for Portuguese9. Texts falling below the first quartile minus 1.5 times the IQR, or above the third quartile plus 1.5 times the IQR, are discarded. This approach effectively eliminates documents that are either too short or too long for their respective domains.

Table 1 presents the statistics for the corpus obtained after applying the filtering pipeline. The final corpus comprises

<html><body><table><tr><td rowspan="2"></td><td rowspan="2">Label</td><td colspan="5">Tokens</td><td rowspan="2">Docs Count</td></tr><tr><td>Min</td><td>Max</td><td>Avg</td><td>Std</td><td>Count</td></tr><tr><td rowspan="2">Journalistic</td><td></td><td>16</td><td>475</td><td>131.29</td><td>61.45</td><td>189,506,320</td><td>1,443,422</td></tr><tr><td></td><td>18</td><td>560</td><td>81.09</td><td>39.11</td><td>27,077,538</td><td>333,903</td></tr><tr><td rowspan="2">Literature</td><td></td><td>16</td><td>186</td><td>77.20</td><td>37.39</td><td>1,859,660</td><td>24,090</td></tr><tr><td>日</td><td>17</td><td>185</td><td>72.55</td><td>36.19</td><td>3,805,896</td><td>52,458</td></tr><tr><td rowspan="2">Legal</td><td></td><td>16</td><td>139</td><td>51.63</td><td>24.43</td><td>152,717,737</td><td>2,957,980</td></tr><tr><td>日</td><td>20</td><td>124</td><td>47.53</td><td>22.11</td><td>221,167</td><td>4,653</td></tr><tr><td rowspan="2">Politics</td><td></td><td>20</td><td>798</td><td>258.32</td><td>173.39</td><td>7,203,739</td><td>27,887</td></tr><tr><td>日</td><td>21</td><td>796</td><td>276.97</td><td>177.60</td><td>1,012,586</td><td>3,656</td></tr><tr><td rowspan="2">Web</td><td></td><td>22</td><td>2042</td><td>517.96</td><td>414.72</td><td>22,598,587</td><td>43,630</td></tr><tr><td></td><td>15</td><td>2075</td><td>539.66</td><td>463.16</td><td>23,913,771</td><td>44,313</td></tr><tr><td rowspan="2">Social Media</td><td></td><td>3</td><td>646</td><td>18.94</td><td>9.85</td><td>44,758,304</td><td>2,363,261</td></tr><tr><td>日</td><td>6</td><td>51</td><td>17.11</td><td>10.17</td><td>94,177</td><td>5,504</td></tr></table></body></html>

Table 1: Summary statistics of the PtBrVarId corpus, including the minimum, maximum, average, standard deviation, and count of tokens, as well as the number of documents for each domain and label.

7,304,438 documents, predominantly from the EP segments of the Journalistic, Legal, and Social Media domains. Regarding the number of tokens, we observe that, with the exception of the Journalistic domain, the distribution between documents labeled as EP and BP within each domain is similar.

A comparison across the domains reveals that the Web domain contains the highest average number of tokens per document, whereas the Social Media domain has the lowest, averaging around 18 tokens per document. This disparity is significant for the development of variety identification models, as distinguishing between language varieties in shorter texts is more challenging due to the limited linguistic cues available. Therefore, the Social Media domain is expected to pose more difficulties than the Web domain, where longer texts provide more opportunities to identify distinguishing features of EP and BP.

It is also important to note that the dataset is highly unbalanced across all domains except the Web domain. This imbalance should be carefully considered when training models using this dataset to ensure robust and unbiased effectiveness.

# Quality Assurance

To ensure the quality of the silver-labeling process, we asked three linguists to manually annotate 300 documents, focusing on two key aspects:

Variety The linguists were asked to determine the variety of the text. They had three options: EP, BP, or “Undetermined” for cases where no variety-specific linguistic features were available.

Domain The linguists were also tasked with identifying the domain to which each sentence belonged. They could choose from the six domains used in this research, or select “Undetermined” if the domain could not be clearly identified.

For the sampling process, we randomly selected 50 documents from each domain in our corpus, with an equal split of 25 documents silver-labeled as EP and 25 as BP.

Table 2 presents the agreement between the three annotators using three metrics:

Fleiss’s Kappa (Fleiss 1971): Measures the agreement between annotators beyond chance, with values ranging from 0 (no agreement) to 1 (perfect agreement).

Majority Rate: Indicates the percentage of texts where two out of three annotators agree on an annotation.

Accuracy: Assesses how often the majority vote between annotators matches the automatic annotation. It is important to remark that the cases where the labeled agreed by the annotators is “Undetermined” we count both silverlabels (EP and BP) as correct since the text is in fact valid in both varieties.

Table 2: Agreement among the three annotators regarding language variety and textual domain.   

<html><body><table><tr><td></td><td>Metric</td><td>Result</td></tr><tr><td rowspan="3">Variety</td><td>Fleiss’ Kappa</td><td>0.57</td></tr><tr><td>Majority Rate</td><td>0.95</td></tr><tr><td>Accuracy</td><td>0.86</td></tr><tr><td rowspan="3">Domain</td><td>Fleiss’ Kappa</td><td>0.69</td></tr><tr><td>Majority Rate</td><td>0.94</td></tr><tr><td>Accuracy</td><td>0.76</td></tr></table></body></html>

The results obtained show that the agreement is higher for the textual domain aspect than for the language variety. However, the variety aspect still achieves a Fleiss’ Kappa of $57 \%$ , which, for three annotators with three labels, can be considered moderate agreement. Upon closer inspection of the results, we found that the Fleiss’ Kappa is lower in the Literature, Social Media, and Legal domains (see Table 3). For the Social Media domain, we found the disagreement to be mainly driven by the short length of the texts, with “Undetermined” representing $42 \%$ of the labels the annotators agreed on. The same was found for the legal domain, which has the second lowest average tokens per document, where the “Undetermined” represents $34 \%$ of the labels the annotators agreed on. In the Literature domain, the disagreement is mainly attributed to the corpus consisting of contemporary books, which often blend linguistic features from both European and BP, making it difficult to assign a definitive variety label.

In Table 3 we detail the annotation agreement metrics per domain for the manually label subset of the PtBrVId corpus. The table shows statistics for the Fleiss’ Kappa with all the labels and the Fleiss’ Kappa when the entries for which one of the annotators marked the entry as “Undetermined”. To complete the table we also show the percentage of entries for which at least one annotator labeled as “Undetermined”.

Nevertheless, a majority consensus among the annotators is almost always achievable (over $90 \%$ of the times) in both aspects. Furthermore, this majority is strongly aligned with the automatic annotations, with agreement between the annotators and the silver labels exceeding $7 5 \%$ .

In addition to releasing the full annotations provided by each annotator, the documents for which a majority vote could be determined are included in the test partition of our dataset. For documents labeled as “Undetermined” by the annotators, the original silver label was used as the final label. The complete dataset is publicly accessible on HuggingFace10. This dataset offers an opportunity for an in-depth study of the cross-domain capabilities of various LVI techniques, with a particular focus on the application of pretrained transformers, which is the main focus of this paper.

Table 3: Extended per-domain analysis of annotator agreement. We present Fleiss’ Kappa for all three labels, as well as Fleiss’ Kappa excluding the “Undetermined” documents (Fleiss’ $\mathrm { K a p p a } _ { w o / u } )$ . The “Undetermined Rate” rows shows the percentage of documents for which at least one annotator labeled as “Undetermined”.   

<html><body><table><tr><td>Domain</td><td>Metric</td><td>Result</td></tr><tr><td rowspan="3">Literature</td><td>Fleiss’ Kappa</td><td>0.23</td></tr><tr><td>Fleiss' Kappawo/u</td><td>0.51</td></tr><tr><td>Undetermined Rate</td><td>0.36</td></tr><tr><td rowspan="3">Legal</td><td>Fleiss’ Kappa</td><td>0.46</td></tr><tr><td>Fleiss' Kappawo/u</td><td>0.73</td></tr><tr><td>Undetermined Rate</td><td>0.34</td></tr><tr><td rowspan="3">Politics</td><td>Fleiss' Kappa</td><td>0.78</td></tr><tr><td>Fleiss' Kappawo/u</td><td>0.87</td></tr><tr><td>Undetermined Rate</td><td>0.10</td></tr><tr><td rowspan="3">Web</td><td>Fleiss’ Kappa</td><td>0.67</td></tr><tr><td>Fleiss’Kappawo/u</td><td>0.84</td></tr><tr><td>Undetermined Rate</td><td>0.20</td></tr><tr><td rowspan="3">Social Media</td><td>Fleiss’Kappa</td><td>0.53</td></tr><tr><td>Fleiss’Kappawo/u</td><td>0.94</td></tr><tr><td>Undetermined Rate</td><td>0.42</td></tr><tr><td rowspan="3">Journalistic</td><td>Fleiss’ Kappa</td><td>0.72</td></tr><tr><td>Fleiss' Kappawo/u</td><td>0.90</td></tr><tr><td>Undetermined Rate</td><td>0.04</td></tr></table></body></html>

# 4 Experimental Setup

In this study, we investigate the effectiveness of fine-tuning a transformer-based model for the Portuguese LVI task. We employ an iterative methodology to identify the optimal strategy for combining training corpora from various domains into a unified training process. Our primary objective is to evaluate cross-domain effectiveness and the generalization capabilities of our models.

# Models & Baselines

For the transformer-based model, we use BERTimbau with 334 million parameters (Souza, Nogueira, and Lotufo 2020). BERTimbau is the result from fine-tuning the original BERT model (Devlin et al. 2019) on a Portuguese corpus.

To establish a baseline for comparison with the BERT model, we employ N-grams combined with Naive Bayes classifiers. This choice is motivated by the proven effectiveness of such models in previous LVI studies across various Indo-European languages, including Portuguese (Zampieri and Gebre 2012).

# Cross-Domain Training Protocol

To ensure that our model generalizes effectively across different domains, we define a two-step training protocol. Step one is used to find the best hyperparameters to train the model so ensure the generalization capability of the model. In this step, the model is trained on a single domain from the PtBrVid corpus and validated on the remaining domains (excluding the one used for training). The hyperparameters yielding the best performance in this cross-domain validation are then used in step two to train the model across all domains combined.

Delexicalization of the corpus is treated as a hyperparameter in our approach. We adjust the probabilities of replacing tokens found by Named Entity Recognition (NER) and Part-of-Speech (POS) tagging with the generic label (such as LOCATION or NOUN), varying these probabilities incrementally from $0 \%$ to $100 \%$ in $20 \%$ steps. It is important to note that delexicalization is applied exclusively to the training set. The validation set remains unaltered, simulating a real-world scenario where the input text is not modified. We leave the study of the impact of delexicalizing the validation set on the effectiveness of the model for future research.

# Train & Validation Data

As referred above the PtBrVId dataset is used to train the models. However, before using for the training, we leave 1,000 documents of each domain for the validation of the model, 500 of each label.

In the step one of our training protocol, we use 8,000 documents from each domain (4,000 from each label) to train the models. We found this sample size to be enough for the models to converge and ensure fast iteration in the training process.

For step two of our training protocol, we compile all the documents from the PtBrVid corpus including the ones used for validation in step one. To avoid the training being dominated by the more represented domains, we undersample the dataset so that all labels from all domains are equally represented. At this step, the manually annotated set from PtBrVId set is used to keep track of the generalization loss.

# Benchmarks

In our evaluation, we use two benchmarks: the DSL-TL and FRMT datasets. As mentioned above, the DSL-TL dataset is the standard benchmark for distinguishing between EP and BP, annotated with three labels: “EP”, “BP”, and “Both”. For our purposes, we exclude documents labeled “Both” since our training corpus does not contain that label. This results in a test set comprising 588 documents for BP and 269 for EP. The FRMT dataset (Riley et al. 2023) has been manually annotated to evaluate variety-specific translation systems and includes translations in both EP and BP. We adapt this corpus for the VID task, resulting in a dataset containing 5,226 documents, with 2,614 labeled as EP and 2,612 as BP.

# 5 Implementation Details

NER and POS tags were identified using spaCy11. The BERT model was trained with the transformers12 and $\mathtt { p y t o r c h } ^ { 1 3 }$ libraries, for a maximum of 30 epochs, using early stopping with a patience of three epochs, binary crossentropy loss, and the AdamW optimizer. The learning rate was set to $2 \times 1 0 ^ { - 5 }$ . In addition, a learning rate scheduler was used to reduce the learning rate by a factor of 0.1 if the training loss did not improve for two consecutive epochs. N-gram models were trained using the scikit-learn14 library. The following hyperparameters were taken into account in the grid search we performed”

• TF-IDF Max Features: The number of maximum features extracted using TF-IDF was tested with the following values: 100, 500, 1,000, 5,000, 10,000, 50,000, and 100,000.   
• TF-IDF N-Grams Range: The range of n-grams used in the TF-IDF was explored with the following configurations: (1,1), (1,2), (1,3), (1,4), (1,5), and (1,10).   
• TF-IDF Lower Case: The effect of case sensitivity was tested, with the lowercasing of text being either True or False.   
• TF-IDF Analyzer: The type of analyzer applied in the TF-IDF process was either Word or Char.

Regarding computational resources, this study relied on Google Cloud N1 Compute Engines to perform the tuning and training of both the baseline and the BERT architecture. For the baseline, an N1 instance with 192 CPU cores and 1024 GB of RAM was used. For BERT, we used an instance with 16 CPU cores, 30 GB of RAM, and 4x Tesla T4 GPUs. The grid search on N-grams takes approximately three hours under these conditions, while for BERT, it takes approximately 52 hours to complete. The final training took three hours for N-grams and approximately ten hours for BERT.

We have made our codebase open-source15 to promote reproducibility of our results and to encourage further research in this area.

# 6 Results

# Impact of Delexicalization

Figure 1 depicts the average $F _ { 1 }$ scores obtained in the PtBrVid validation set by the $\mathbf { N } .$ -grams and BERT models, for each $( P _ { \mathrm { P O S } } , P _ { \mathrm { N E R } } )$ percentage pair. The averages are computed across models trained in different domains.

The results suggest that intermediate levels of delexicalization can yield marginal improvements in model effectiveness. However, high levels of $P _ { \mathrm { P O S } }$ adversely affect model performance. This finding is particularly interesting because previous studies have reported significant reductions in effectiveness due to delexicalization (Sharoff, Wu, and Markert 2010; Lui et al. 2014). Notably, these earlier studies focused solely on full delexicalization and did not evaluate performance on out-of-domain corpora.

![](images/464dd0d4660d24043128e38a5a9f1d7f1ad7ba586eda336fbd8d3b2e19fb19f0.jpg)  
Figure 1: Average $F _ { 1 }$ score for each $( P _ { \mathrm { P O S } } , P _ { \mathrm { N E R } } )$ .

Based on these insights, we proceeded to the second step of our training protocol using a delexicalized version of the training set, with $\mathrm { \Delta } P _ { \mathrm { P O S } } = 0 . 2$ , $P _ { \mathrm { N E R } } = 0 . 6 \$ ) for the N-gram model and ( $P _ { \mathrm { P O S } } = 0 . 6$ , $P _ { \mathrm { N E R } } = 0 . 0 \$ ) for BERT models.

# Overall Results

This section presents the $F _ { 1 }$ scores for the N-gram baseline and BERT fine-tuning models, comparing their performance with and without delexicalization to highlight their impact on the overall effectiveness of the model.

The results in Figure 2 underline the benefits of delexicalization on system effectiveness across both benchmarks and models. Specifically, in FRMT, training in the delexicalized corpus improved the $F _ { 1 }$ score by approximately 13 and 10 percentage points for the N-gram and BERT models, respectively.

Upon examining the less pronounced discrepancy in the DSL-TL benchmark, we found it to be largely attributed to the FRMT dataset’s entity-specific partition, known as the entity bucket. In this bucket, models trained without delexicalization struggle, as they rely on entities to determine language variety. Given that the FRMT dataset contains the same text in both BP and EP, these models often misclassify pairs of sentences by assigning the same label to both, leading to frequent errors. In the extreme case, they end up getting around half of the labels wrong, which is what happened to the N-gram model, only achieving an $F _ { 1 }$ score of $4 6 . 9 5 \%$ in this benchmark. This highlights the importance of using delexicalization in the training process. To the best of our knowledge, we are the first to report positive results from the use of delexicalization, which was enabled by the proposed cross-domain training protocol.

![](images/c7e85fa2b6cd9417ae05dbabecbd6646d9a8de4b542ac3b1fa60b9f5bee843b8.jpg)  
Figure 2: $F _ { 1 }$ in FRMT and DSL-TL benchmarks. Models with the subscript $d$ were trained on a delexicalized corpus.

When comparing the BERT model with the N-gram models, one can observe that the BERT model outperforms the N-gram model across all scenarios, achieving an $F _ { 1 }$ score of $8 4 . 9 7 \%$ in DSL-TL and $7 7 . 2 5 \%$ in FRMT. To support further research and exploration, we have made the $\mathbf { B E R T } _ { d }$ model available on HuggingFace, inviting the research community to use and build on this work16.

# 7 Conclusion & Future Work

In this study, we introduced the first multi-domain Portuguese LVI corpus, which includes more than 7 million documents. Leveraging this corpus, we fine-tuned a BERTbased model to create a robust tool for discriminating between European and Brazilian Portuguese. The training strategy leverages delexicalization to mask entities and thematic content in the training set, thereby enhancing the model’s ability to generalize. This approach has potential for adaptation to other language variants and languages.

We have identified two key avenues for future work to further enhance the quality and scope of Portuguese LVI. First, the corpus should be expanded to include other lessresourced Portuguese varieties, particularly African Portuguese. Second, it is crucial to explore the impact of the pretrained model selection, as the language variety on which the model was originally trained may introduce bias into the LVI classifier.