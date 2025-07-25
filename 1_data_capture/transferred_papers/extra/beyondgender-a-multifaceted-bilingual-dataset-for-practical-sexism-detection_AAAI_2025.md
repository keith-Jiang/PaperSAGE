# BeyondGender: A Multifaceted Bilingual Dataset for Practical Sexism Detection

Xuan Luo1,2, Li Yang1, Han Zhang1,3, Geng $\mathbf { T } \mathbf { u } ^ { 1 }$ , Qianlong Wang1, Keyang Ding1, Chuang Fan1, Jing $\mathbf { L i } ^ { 2 , 5 }$ , Ruifeng $\mathbf { X } \mathbf { u } ^ { 1 , 3 , 4 * }$

1Harbin Institute of Technology, Shenzhen, China 2Department of Computing, The Hong Kong Polytechnic University, Hong Kong, China 3Peng Cheng Laboratory, Shenzhen, China 4Guangdong Provincial Key Laboratory of Novel Security Intelligence Technologies, China 5Research Centre on Data Science & Artificial Intelligence, Hong Kong, China gracexluo@hotmail.com, jing-amelia.li $@$ polyu.edu.hk, xuruifeng $@$ hit.edu.cn

# Abstract

Sexism affects both women and men, yet research often overlooks misandry and suffers from overly broad annotations that limit AI applications. To address this, we introduce BeyondGender, a dataset meticulously annotated according to the latest definitions of misogyny and misandry. It features innovative multifaceted labels encompassing aspects of sexism, gender, phrasing, misogyny, and misandry. The dataset includes 6.0K English and 1.7K Chinese sexism instances, alongside 13.4K non-sexism examples. Our evaluations of masked language models and large language models reveal that they detect misogyny in English and misandry in Chinese more effectively, with F1-scores of 0.87 and 0.62, respectively. However, they frequently misclassify hostile and mild comments, underscoring the complexity of sexism detection. Parallel corpus experiments suggest promising data augmentation strategies to enhance AI systems for nuanced sexism detection, and our dataset can be leveraged to improve value alignment in large language models.

# Introduction

Sexism, prejudice, or discrimination based on one’s sex or gender, has exacerbated gender inequality and injustices. Research has been made to assist the detection of sexism at scale (Grosz and Conde-Cespedes 2020; Jiang and Zubiaga 2023; Rizzi et al. 2023; Krenn et al. 2024).1

Sexism primarily affects women and misogyny2, a widespread and enduring sexist ideology, has been practiced for thousands of years (Holland 2012). According to Ambivalent Sexism theory (Glick and Fiske 1996), sexism towards women has two sub-components: 1) Hostile Sexism (HS), characterized by overtly negative evaluations and stereotypes, e.g. ‘Women belong in the kitchen, not in the engineering department’, and 2) Benevolent Sexism (BS), which may appear subjectively positive, e.g. ‘Women are nurturing, caring, or cooperative’. HS is easily recognizable due to its aggressive expressions, whereas BS often presents itself as positive but ultimately regards women as amiable yet weak. Therefore, BS is a guise of luring women to stay at a lower social status than men (Cowie, Greaves, and Sibley 2019). Although sexism has historically disadvantaged women, men also suffer from the negative consequences of sexism or misogyny, albeit in more subtle ways, including being sexually objectified and facing pressure to conform to masculine norms (Mabrouk 2020; Dafaure 2022). On the other hand, Misandry, ‘hatred of, contempt for, or prejudice against men or boys’, represents women’s anger against their oppressors (e.g. ‘In fact, a man has less worth than a woman because he has one less place for another man to shove his dick into.’). It often manifests in portrayals of men as absent, insensitive, or abusive. However, there is a lack of studies addressing the situation of men.

Another issue is the excessively broad annotation found in existing public datasets (e.g., (Jiang et al. 2022; Kirk et al. 2023)), making it impractical to detect harmful discrimination from harmless prejudices. For example, these datasets classify facts or phenomena subject to debates on gender equality3, as well as profanity words stemming from individual misconduct rather than having directed connection to gender, as instances of sexism. It introduces bias during model training and leads to an excess of false positive predictions, ultimately limiting real-world application.

# Objectives and Contributions

We aim to provide a valuable resource and benchmark for the comprehensive detection of sexism. Specifically, it seeks to address the following objectives that have been overlooked in previous research: to raise awareness and facilitate the detection of sneaky misogyny and sexism towards men and foster constructive debates. To achieve these goals, we introduce BeyondGender developed with the following labels: 1) Sexism: If the text shows the poster’s prejudice or discrimination based on someone’s gender. 2) Gender:

Labels Languages BeyondGender   
Gender Misandry   
Sexism Misogyny English Chinese   
Hostile /Benevolent Categories of sexism Italian Spanish   
Target type French Collection of other datasets

The target gender of the text. 3) Phrasing: Text’s tone, hostile or benevolent. 4) Misogyny: If the text expresses hatred of, contempt for, or prejudice against women. 5) Misandry: If the text expresses hatred of, contempt for, or prejudice against men.

Four features make BeyondGender 4 a practical dataset for sexism detection: 1) Novel Facets: Gender and Misandry labels, which correspond to the state of affairs. 2) Data Diversity: data samples from YouTube, Reddit, Gab, and Weibo; a bilingual dataset covering 12.7K English data and 8.4K Chinese data. 3) Large-scale: BeyondGender has over 21K data, with a relatively high proportion of sexism data. 4) Annotation Quality: clear and detailed annotation guidelines with comprehensive scenarios considered; over $93 \%$ inter-annotator agreement.

The data distribution reveals the sexist cultures in English and Chinese. Misogyny remains more pronounced than misandry in both cultures, while they are more likely to be expressed in a hostile manner in English compared to Chinese. Moreover, the high-frequency words are quite different. In addition, We experiment with classic masked language models (MLMs) and large language models (LLMs). It appears that they have a higher performance in identifying misogyny in English and recognizing misandry in Chinese. Nonetheless, they tend to incorrectly categorize hostile comments as misandry/misogyny and benevolent or mild comments as non-misandry/non-misogyny.

The main contributions are as follows: 1) Introduction of a high-quality bilingual dataset for practical sexism detection, 2) The first publicly available dataset for misandry detection in Chinese and English, and 3) Evaluation of baseline models and parallel study, revealing the challenges and a possible solution in detecting target gender, misogyny, and misandry in both languages.

# Related Work

In this section, we first compare existing textual datasets (published by journals or proceedings, and most of them are publicly available) from two perspectives. Secondly, we explain why we constructed BeyondGender and the adjustments we made compared to recent annotation codebooks.

Language and source: The majority of sexism detection datasets are available in English, reflecting the extensive research and efforts in this language. Recognizing the global significance of sexism detection in various linguistic contexts and cultures, researchers have expanded their focus to cover multiple languages. (Fersini et al. 2018a) collected a corpus in both English and Italian, while their work in (Fersini et al. 2018b) collected a corpus in English and Spanish. (Bhattacharya et al. 2020) enriched the sexism detection in three languages commonly spoken in India. (Chiril et al. 2020) presented the first French corpus annotated for sexism detection, (El Ansari, Jihad, and Hajar 2020) for Arabic, (Rizwan, Shakeel, and Karim 2020) for Roman Urdu, (Ho¨fels, C¸ o¨ltekin, and Ma˘droane 2022) for Romanian, (Zeinert, Inie, and Derczynski 2021) for Danish, (Jiang et al. 2022) for Chinese, and (Krenn et al. 2024) for German. For data collection, Twitter is the most popular platform, followed by Facebook and YouTube. Also, news and document websites are the sources for sexism detection (De Pelle and Moreira 2017; Parikh et al. 2019).

Category and granularity: The mainstream tasks regarding sexism detection are: 1) multi-label Hate speech categorization, where sexism is detected as a sub-category, along with other categories such as racism (Waseem and Hovy 2016; Priyadharshini et al. 2022; Al-Hassan and Al-Dossari 2022); 2) binary sexism identification (Grosz and Conde-Cespedes 2020; Samory et al. 2021; Bertaglia et al. 2023); 3) multi-label sexism type categorization (Jha and Mamidi 2017; Sharifirad and Matwin 2019; Ho¨fels, C¸ o¨ltekin, and Ma˘droane 2022); 4) binary misogyny identification (Bhattacharya et al. 2020; Almanea and Poesio 2022), 5) multi-label misogyny type categorization (Fersini et al. 2018a,b; Guest et al. 2021b; Mulki and Ghanem 2021), and 6) other hierarchical classification (Guest et al. 2021a; Jiang et al. 2022; Kirk et al. 2023).

Why BeyondGender? Upon examining the datasets mentioned above, we have identified several areas for improvement in sexism detection: 1) The datasets predominantly focus on sexism towards women, with little data available on misogyny towards men and misandry. 2) Many of these datasets lack clear definitions for their categories, and previous codebooks (Samory et al. 2021; Sultana, Sarker, and Bosu 2021; Sultana 2022) are primarily oriented towards sexism against the opposite gender. 3) A certain proportion of data labeled as sexism are discussions or historical accounts about gender inequality issues rather than personal prejudice or discrimination towards a specific gender. Moreover, aversion caused by misconduct is often roughly categorized as sexism. To address these limitations, we collect and annotate BeyondGender, a dataset designed for sexism detection across both genders. We include misandry and add women’s self-loathing and rejection of feminine qualities to misogyny in our annotation codebook and distinguish those situations and nuances mentioned above when labeling.

# Dataset

# Collection

BeyondGender is collected from two sources: YouTube comments and previous datasets. In order to acquire comments related to sexism from YouTube, we initiated searches on videos using a predefined set of representative keywords:

Level 1 Level 2 Examples Gender Phrasing HW HM Corpus Sexism Yes Gender Long term thinking is man's job. Men Mild Yes No ？ but but but women are the ones who can't empathize with men :( Women Hostile Yes No   
·No, just no...   
yer Wtmen drowineaof ttntio,etrugle ina desert Both Mild No No   
women are the No Phrasing No,I'm black actuall. Goahead, check my post history to confirm it   
emes thze can't Hi Amelia, your site is fucking tolClasicinctas.sticesar'tieitesibl Women Mild Yes No   
mworidhedygto eraove tka Misogyny oysPam Men Hostile Yes No   
balance things Misandry Women ad No Nes

1) Sexism culture: red pill, incel, manosphere, foid, misogyny, Feminism5, etc., 2) Activities: marriage, parenting, etc., 3) Events: sexual violence, sexual harassment, #MeToo, interview about genders, etc., 4) Arts: Barbie, Pride & Prejudice, etc.. We discard comments which fall below 15 words in length and lack gender-related terminology.

We also leverage recent sexism detection datasets because they represent the contemporary sexism culture and they are collected from different social platforms: 1) English dataset EDOS (Kirk et al. 2023) and 2) Chinese dataset SWSR (Jiang et al. 2022). Given that they broadly categorize critical discussions of gender inequality and aversion stemming from misconduct as instances of sexism, we apply more restricted criteria when determining if it is sexism.

BeyondGender is composed of a total of 21.1K data, comprising 13.2K comments collected from YouTube, 3.1K samples from the Chinese SWSR dataset, and 4.8K samples from the English EDOS dataset.

# Annotation

The annotation workflow and examples of annotated data are illustrated in Figure 2. The multifaceted labels are divided into two levels: first, we determine whether the text is sexist or non-sexist. Second, if it is identified as sexism, we annotate the other four labels. The meanings of each label and annotation guidelines are provided as follows:

# 1. Sexism: If it conveys prejudice or discrimination based on one’s sex or gender.

Sexism typically targets a group of people, e.g., ‘All women benefit from the actions of violent men’. Moreover, if the statement is pointing at an individual but can be generalized to that gender group, it is also considered sexism. For example, ‘You should do all the heavy lifting since you are a man’ is a sexism (labe $\scriptstyle 1 = 1$ ), while ‘Tell the friend to dump the evil. Let him watch how easily she gets another man to simp for her’ is a non-sexism (labe $_ { = 0 }$ ).

Previous datasets broadly labeled obscure or controversial conditions as sexism, potentially discouraging discussions about gender issues. Therefore, we make several adjustments and categorize certain situations as nonsexism, including: 1) Hatred directed at an individual due to factors such as race, religion, political views, other than gender. 2) Usage of gender-specific derogatory terms in the context of an event or misconduct not directly related to gender issues.   
2. Gender: The target gender of the text. The values of gender referred to in the text are men (la$\mathrm { \ b e l { = } } 1$ ), women (labe $\scriptstyle = 0$ ), and both (labe $\scriptstyle 1 = 2$ ) (e.g. when the two genders are symmetrically compared). For transgenders, we annotate the gender following the view of the poster.   
3. Phrasing: The manner in which the statement is expressed. It is hostile (labe $\scriptstyle 1 = 1$ ) if the statement is aggressive, uses derogatory gender terms, or invokes threats. Conversely, it is benevolent (labe $_ { = 0 }$ ) if it is positive or hypocritical. It is mild (label $= 0$ ) if it is neutral or emotionless. Both benevolent and mild instances are labeled as mild.   
4. Misogyny: If it conveys hatred of, contempt for, or prejudice against women. Misogyny is a common sexist ideology in binary gender. The scenarios that reflect misogyny (labe $_ { = 1 }$ , otherwise labe $_ { = 0 }$ ) include, but are not limited to: 1) Violence against women. 2) Controlling and punishing women who challenge male dominance, typically differentiating between good women and bad ones. 3) Rejection of feminine qualities6, which also extends to the rejection of any aspects of men perceived as feminine or unmanly. 4) Mistrust of women. 5) Regarding women as societal scapegoats. 6) Blaming women for one’s own failure in life. 7) Objectification of women. 8) Stereotypes suggesting that women weaponize their

Table 1: The label distributions in BeyondGender. For gender, M, W, and $B$ represent man, woman, and both genders, respectively. For Phrasing, $H$ and $B$ represent hostile and benevolent/mild, respectively.   

<html><body><table><tr><td>Categories</td><td>#English</td><td># Chinese</td><td># Total</td></tr><tr><td>Sexism (Y/N)</td><td>6.054/6.664</td><td>1,691/6,710</td><td>21,119</td></tr><tr><td>Gender (M/W/B)</td><td>848/5,174/ 31</td><td>787/878/27</td><td>7,745</td></tr><tr><td>Phrasing (H/B)</td><td>5,312 / 742</td><td>713 /978</td><td>7,745</td></tr><tr><td>Misogyny</td><td>4,840</td><td>619</td><td>5,459</td></tr><tr><td>Misandry</td><td>954</td><td>600</td><td>1,554</td></tr></table></body></html>

Table 2: The average length of comments in BeyondGender.   

<html><body><table><tr><td>Categories</td><td>#English</td><td># Chinese</td></tr><tr><td>Sexism?(Yes/No)</td><td>163/124</td><td>70 /73</td></tr><tr><td>Gender (M/W)</td><td>298 /140</td><td>74 /66</td></tr><tr><td>Phrasing (H/B)</td><td>156 /209</td><td>76 /65</td></tr><tr><td>Misogyny</td><td>137</td><td>67</td></tr><tr><td>Misandry</td><td>314</td><td>78</td></tr></table></body></html>

Table 3: The size of the train, dev, and test sets.   

<html><body><table><tr><td>Language</td><td colspan="3">Label Sexism</td></tr><tr><td></td><td>train</td><td>dev test</td><td>train dev test</td></tr><tr><td>English</td><td>10,233</td><td>1,000 485</td><td>4,733 500 485</td></tr><tr><td>Chinese</td><td>6,501</td><td>700 500</td><td>1,099 120 500</td></tr></table></body></html>

appearances or that women use seduction to control men. 9) Women’s self-loathing, including hating their bodies, disdain for women who are ”wives” or ”mothers”, seeking validation through male approval, etc..

# 5. Misandry: If it conveys hatred of, contempt for, or prejudice against men.

Compared to misogyny, misandry is a minor issue, mainly due to the stress response to misogyny.7 The scenarios that reflects misandry (labe $_ { = 1 }$ , otherwise labe $_ { . = 0 }$ ) include, but not limited to:

1) Violence against men.   
2) Women’s anger against their oppressors.   
3) Opposition to gender-equal laws, such as those related to rape, violence, and divorce.   
4) Usage of terms incorporating ”man” as a derogatory prefix, such as mansplaining, manspreading, and manterrupting.

# Statistics

The distributions of each label are listed in Table 1. We annotate around 21K data with 7.7K sexism and 13.4K nonsexism. The average length of comments in BeyondGender is listed in Table 2. English data is counted by words, while Chinese data is counted by Chinese characters. Compared to Chinese data, most English sexism data are hostile and misogyny. It can also be revealed by test set composition, demonstrated in Figure 3a and 3b. The test set has $\mathrm { 4 8 5 ~ E n - }$ glish data and 500 Chinese data. The split of the dataset is listed in Table 3.

Table 4: The conditional probabilities in the test set. Miso. and Misa. represent misogyny and misandry, respectively.   

<html><body><table><tr><td rowspan="2">Given</td><td colspan="2">English</td><td colspan="2">Chinese</td></tr><tr><td>Miso.</td><td>Misa.</td><td>Miso.</td><td>Misa.</td></tr><tr><td>Men</td><td>0.47</td><td>0.34 0.05</td><td></td><td>0.63</td></tr><tr><td>Women</td><td>0.82</td><td>0.01</td><td>0.65</td><td>0.05</td></tr><tr><td>Hostile</td><td>0.90</td><td>0.09</td><td>0.42</td><td>0.57</td></tr><tr><td>Mild</td><td>0.25</td><td>0.09</td><td>0.31</td><td>0.17</td></tr></table></body></html>

![](images/8b3d87949853e3ddfab2071f416b101d5af2b4325267ef1cf1247885c2369fb4.jpg)  
Figure 3: Composition of the test set.

Table 4 listed the conditional probabilities of misogyny and misandry given specific gender and phrasing manner. Notably, nearly half of the English sexism data directed at men is actually misogyny, while the majority of sexism data targeting women is misogyny. When considering phrasing manner, it becomes evident that English data predominantly exhibit hostile misogyny, whereas Chinese data express a greater degree of hostility towards men.

# Cultural Similarities and Differences

After calculating the word frequency in BeyondGender and considering the statistics above, we list representative words in Table 5 and summarize the similarities and differences from three perspectives:

Table 5: Representative terms in each language. The Chinese column displays Chinese characters/words in the format of (transliteration, English translation) pair.   

<html><body><table><tr><td>English</td><td>Chinese</td></tr><tr><td>rape</td><td>(qian,money)</td></tr><tr><td>bitch/whore...</td><td>(hunlv,marriage donkey)</td></tr><tr><td>ugly/pretty</td><td>(gongzuo,career)</td></tr><tr><td>fat</td><td>(shengyu,childbearing)</td></tr><tr><td>stupid</td><td>(lihun,divorce)</td></tr><tr><td>sex</td><td>(jiabao,domestic violence)</td></tr><tr><td>fuck/fucking</td><td>(zhinanai,malechauvinist)</td></tr></table></body></html>

1. Profanity: The English data exhibits a higher proportion of hostile phrasing, primarily attributed to the frequent use of profanities, including terms like whore, fuck, bitch, shit, pussy, etc., and often combined with sexual references. Although sex-related words are not prevalent in Chinese data, derogatory terms are used, such as describing women as marriage donkeys and marriage object and animal names, which have the same pronunciation as “men” in Chinese.

2. Gender Bias: In contrast to the widespread practice and extensive history of misogyny, misandry is less pronounced in English data. Conversely, in Chinese data, misandry is more obvious, and misogyny is sneakier than in English data. Moreover, misogyny in English is primarily attributed to men, such as incel8, for reasons 4) - 8) of misogyny in the Annotation section. Conversely, in Chinese data, a notable proportion of misogyny originates from women, particularly in the context of scenario 9). Worth mentioning, misogyny among women is more prevalent in east-Asian cultures than in Western cultures.

3. Topics: English speakers pay more attention to appearance, intelligence, and sex-related actions. People in Chinese culture or even east-Asian culture are more concerned about financial status and marriage-related events, such as childbearing, domestic violence, and divorce.

# Quality Evaluation

Annotators are recruited from prestigious university undergraduate and graduate students proficient in both English and Chinese. The team comprises four men and three women to mitigate gender bias. We have provided training to these annotators and initiated the process by annotating a set of 100 data samples. Throughout the pre-annotation phase, we engaged in discussions and made necessary refinements to the annotation rules, resulting in the final version of the guidelines as presented in the Annotation section. As shown in Table 6, we sampled 300 comments from the entire dataset to calculate consistency. Out of these 300 comments, 280 were consistently labeled as sexism. Within the subset of 280 ”sexism” comments, the annotation consistency for each label is above $93 \%$ . For the entire dataset, the consistency reaches 94, 97, 95, 92, and 95 percentage.

Table 6: Annotation consistency (#same label / #sample).   

<html><body><table><tr><td>Label</td><td>Consisteny</td><td>Percentage (%)</td></tr><tr><td>Sexism</td><td>280/300</td><td>93</td></tr><tr><td>Gender</td><td>270/280</td><td>96</td></tr><tr><td>Hostile</td><td>263 /280</td><td>94</td></tr><tr><td>Misogyny</td><td>273 /280</td><td>94</td></tr><tr><td>Misandry</td><td>262/280</td><td>94</td></tr></table></body></html>

# Experiment

# Metrics

The metrics we use for classification are Precision, Recall, F1-score, and Accuracy. Due to the label settings, the results will have high recall and F1-score if the predictions are all sexism, men, hostile, misogyny, and misandry. Therefore, we also consider the false predictions for better analysis of the shortcomings of models.

# Baselines

We evaluate the sexism detection capability of current state-of-the-art and mainstream models. In the monolingual setting, we fine-tune the Masked Language Models (MLMs) using the training set and select the best-performing model based on the dev set. For Large Language Models (LLMs), we adopt in-context learning.   
MLMs: 1. BERT (Devlin et al. 2019; Cui et al. 2019), 2. RoBERTa (Liu et al. 2019; Cui et al. 2020), 3. DeBERTa (He, Gao, and Chen 2022).   
LLMs: 1. ChatGPT (OpenAI 2022), 2. ChatGLM (Du et al. 2022), 3. Baichuan (Yang et al. 2023), 4. LLama (Touvron et al. 2023), 5. Alpaca (Taori et al. 2023).

# Settings

For masked language models, we train five respective classifiers for the five labels. During training, we set the random seed to 42, the learning rate to 1e-5, and the batch size to 16 with Adam optimizer. We try epochs varying from 1, 5, 10, 15, 20, 30, and 40. To simulate the real distribution of comments in social media, we add non-sexism data from previous datasets into training. The randomly sampled train and dev set for labels in level 2 are only those labeled as sexism. The divisions are listed in Table 3. When testing, the classifiers predict all labels for each data. For level-2 labels, only data whose true label is sexism are evaluated.

For large language models, we add several examples in the prompt and combine the data as input. The inputs of LLMs are in the format:

$$
\ P r e f i x + D a t a + S u f f i x
$$

where P refix contains the task description and provides several examples; $D a t a$ is used to represent each piece of test data; $S u f f i x$ remains a constant string. Task descriptions declared in the P ref ix for other labels will be shared with code and data.

Table 7: The test results of the English dataset. Note that “.xx” represents a value of 0.xx.   

<html><body><table><tr><td>Model</td><td>Sexism F1 Acc</td><td>Gender F1 Acc</td><td>Phrasing F1 Acc</td><td>F1</td><td>Miso. Acc</td><td>Misa. F1 Acc</td></tr><tr><td>BERT RoBERTa DeBERTa</td><td>.85 .76 .86 .78 .78 .68</td><td>.41 .25 .18</td><td>.27 .86 .78 .87 .77 .87</td><td>.76 .78 .79</td><td>.85 .75 .86 .77 .85 .74</td><td>.29 .87 .32 0.89 .33 .90</td></tr><tr><td>ChatGLM Baichuan ChatGPT Llama Alpaca</td><td>.86 .75 .81 .71 .86 .78 .79 .67 .00 .24</td><td>.44 .30 .30 .40 .00</td><td>.54 .84 .35 .86 .33 .88 .30 .87 .77 .86</td><td>.77 .78 .80 .77 .76</td><td>.84 .75 .85 .76 .87 .78 .69 .59 .85 .75</td><td>.14 .26 .18 .51 .23 .47 .19 .30 .17 .19</td></tr></table></body></html>

<html><body><table><tr><td>Model</td><td>Sexism F1 Acc</td><td>F1</td><td>Gender Acc</td><td>Phrasing F1 Acc</td><td>Miso. F1 Acc</td><td>Misa. F1 Acc</td></tr><tr><td>BERT RoBERTa</td><td>.56 .28</td><td>.56 .76 .41 .66</td><td>.77 .50</td><td>.75 .78 .66 .70</td><td>.61 .72 .59 .63</td><td>.62 .72 .51 .43</td></tr><tr><td>DeBERTa ChatGLM Baichuan ChatGPT .81</td><td>.44 .48 .82 .73 .75 .68</td><td>.66 .53 .53</td><td>.64 .51 .49</td><td>.71 .73 .57 .65 .64 .66</td><td>.59 .69 .49 .49 .51 .40</td><td>.61 .71 .50 .59 .49 .54</td></tr></table></body></html>

Table 8: The test results of the Chinese dataset.

# Main Results

The main results of English and Chinese test sets are listed in Table 7 and Table 8, respectively.

For English data, both MLMs and LLMs perform well in detecting sexism, hostility, and misogyny. However, they show only modest performance when it comes to distinguishing gender and detecting misandry. Although model Alpaca achieves similar accuracy as other LLMs, in fact, it predicts all the data as non-sexism (labe $\scriptstyle 1 = 0$ ), target women (labe $\scriptstyle \left\lfloor = 0 \right\rfloor$ ), hostile (labe $\scriptstyle \left\lfloor = 1 \right\rfloor$ ), misogyny (labe $_ { = 1 }$ ), and misandry (labe $\scriptstyle \left\lfloor = 1 \right\rfloor$ ). Therefore, it has 0.0 precision for Sexism and Gender but 1.0 recall for Phrasing, Misogyny, and Misandry labels. However, it cannot be applied to sexism detection. For Chinese data, LLMs significantly outperform MLMs in sexism detection, while MLMs perform better in determining target gender and phrasing and detecting misogyny and misandry than LLMs.

Compared the results of misogyny and misandry detection in two languages, the gaps among MLMs can be explained by the uneven distribution of data. Since the ratio of misogyny to misandry in English is 5 and that in Chinese is 1.3, MLMs trained with these data perform misogyny detection substantially better than misandry detection in English but slightly better in Chinese. On the other hand, LLMs, which are not re-trained with any data, have greater gaps than MLMs in English data. It probably reflects that English misandry data are scarce while Chinese misandry data are sufficient in the corpus for LLM pre-training by then.

# False Predictions Analysis

To have a grasp of the improvement direction, we examine the phrasing factor first. We calculate the probability of false predictions of misogyny and misandry given different phrasing manner (ground-truth labels), shown as Table 9 and 10. Comparing the two tables, we have the following findings:

Table 9: False positive (left) and false negative (right) predictions with phrasing factor in English data. $H$ and $B$ represent hostile and benevolent tones, respectively. Note that ˆ is marked when the difference is equal or larger than 0.03.   

<html><body><table><tr><td>Model</td><td>Misa. H</td><td>B H</td><td>Miso. B</td><td>non-Misa. H B</td><td>H</td><td>non-Miso. B</td></tr><tr><td>BERT RoBERTa DeBERTa</td><td>.07 .04 .04</td><td>.06 .04 .03 .63</td><td>.85 .85 .74 .74 1.0^ .95</td><td>.77^ .77^ .81^</td><td>.50 .63 .50</td><td>.05 .05 .05 .09^ .02 .00 .18^</td></tr><tr><td>ChatGLM Baichuan ChatGPT Llama Alpaca</td><td>.83^ .58^ .59^ .76 1.0</td><td>.24 .53 .76 1.0</td><td>.81^ .73 .78 .77 .70 .76^ .33 .59^ 1.0 1.0</td><td>.35^ .38 .08 .08 .00</td><td>.38 .50^ .38^ .13^ .00</td><td>.08 .05 .18^ .04 .09^ .37 .41^ .00 .00</td></tr></table></body></html>

Table 10: False positive (left) and false negative (right) predictions with phrasing factor in Chinese data.   

<html><body><table><tr><td>Model</td><td colspan="2">Misa. H B</td><td colspan="2">Miso. H B</td><td colspan="2">non-Misa. H B</td><td colspan="2">non-Miso. H B</td></tr><tr><td>BERT RoBERTa DeBERTa</td><td>.57^ .93^</td><td>.16 .77 .20</td><td>.26^ .51^</td><td>.20 .38</td><td>.20 .06</td><td>.58^ .14^</td><td>.31 .20</td><td>.47^ .32^</td></tr><tr><td>ChatGLM Baichuan ChatGPT</td><td>.50^ .33 .55 .83^</td><td>.45^ .53 .69</td><td>.35^ .69^ .86 .75^</td><td>.23 .59 .86 .64</td><td>.28 .34 .33 .06</td><td>.42^ .47^ .31 .83^</td><td>.29 .29 .10 .07</td><td>.47^ .32^ .14^ .20^</td></tr></table></body></html>

1. Hostile but not hateful: On the left-hand side are the proportions that are predicted as misandry or misogyny, while the true labels are not misandry or not misogyny. In both languages, data expressed in a hostile manner are more prone to be mistakenly predicted as misandry or misogyny.9 Because hostility can be conveyed through profanity not related to gender-based hatred, these offensive words introduce noise to the detection.

2. Sneaky misogyny and misandry: On the right-hand side are the proportions that are falsely predicted as not misandry and not misogyny. Mild sexism is more difficult to detect in both languages. In Chinese data, at least $1 / 7$ remains undetected, with misandry detection suffering more overall. A similar situation exists in English data, except that MLMs are more inclined to predict the misogyny label. These sneaky misogyny and misandry, if left undetected, are more deleterious since they will make acute gender inequality worse than harmless prejudice.

We also examine the gender factor and reach the same conclusion as in the Main Results section that LLMs cannot distinguish misandry well, no matter the target gender is men or women. Moreover, misogyny directed at men is also sneaky to detect.

Data S G P.HWHM   
Zh: 其實討厭另一個性別是由美國第二波 女權發起的，約在 70年代，他們認為男性是“父權”的既得利益者   
L.S. QishitaoyanlingyigexingbieshiyouMeiguodierbonvquanfaqide，yuezai7Oniandai,tamenrenweinanxingshifuquandejideliizhe   
Trans.Infact,theaversiontotheoppositegenderwasinitiatedbythesecondwaveoffeminismintheUnitedStates,aroundthe 1970s.They believed that men were beneficiaries of the ‘patriarchy'. True Labels: Non-sexism.   
Zh: 最重要 的是 光　想不劳而获不付出 想得到 得不到 0   
L.S. zui zhongyao de shiguang xiang bu lao er huo bu fuchu xiang dedao debudao   
Trans.Themostimportantthingis,justthinkingwithoutefort,wantingsomethingwithoutworkingforitandendingupwithothing. 0   
L 妈男a么个go 0   
Trans.Motherfxker,saying thatallthegoodmenare gone,orthatyou'retootiredtolove.Andsayingthathowyoulivebyyourself. UNK 0   
Zh: 整个 一副我有b我有理的样子，自己也不照照 镜子。   
L.S. zhengge yifu wo you b wo youli de yangzi， zijiye bu zhaozhao jingzi 0 UNK   
Trans.Acting all self-righteous because of your cunt and not even looking at yourself in the mirror. True Labels: Sexism, Women, Hostile,and Misogyny. 0 UNK 1

# Parallel Study

We conduct two sets of experiments utilizing our bilingual data translation for parallel data. Detailed settings and results analysis are elaborated in the Appendix. The results indicate a promising avenue in data augmentation and support previous findings that balanced and larger amounts of data would improve performance. However, the discrepancy in linguistic features remains challenging in detecting phrasing, misogyny, and misandry.

# Case Study

We provide two wrongly predicted Chinese examples in Figure 4. The first example contains sexist words but is a historical narrative, which does not convey personal opinions but is mistakenly predicted as sexism by all models.

The second example only mentions “men”, which “misleads” all MLMs to predict the target gender is men and all LLMs output ”Unknown”. Although they almost correctly predict hostility and misogyny, most of them also predict it as misandry and even non-sexism. Since the models for each label are trained separately, there is a gap in consistent predictions among the whole set of labels, which will be solved in future research.

# Conclusion

# Implications

The theoretical implications are twofold: first, it delves into the nuances of sexism, particularly in distinguishing between hostile and benevolent sexism. Second, the inclusion of misandry expands the framework, emphasizing the necessity for a comprehensive detection of sexism that goes beyond a single gender. From a practical perspective, by refining annotation criteria and labels, BeyondGender enables more accurate and nuanced detection of harmful discrimination, such as sneaky misogyny, while minimizing false positives. It has significant implications for the development of models and algorithms for automated detection of sexism and large language model alignment, ultimately contributing to the creation of safer and more inclusive environments.

In this paper, we present BeyondGender, a high-quality large-scale bilingual dataset designed for practical sexism detection. We provide comprehensive information on the annotation guidelines and dataset statistics, as well as a comparison of the sexist culture represented in English and Chinese data. In addition, we evaluate the capabilities of masked language models and large language models in detecting sexism, target gender, phrasing manner, misogyny, and misandry. Through a detailed analysis, we shed light on the challenges in identifying misogyny and misandry. Through parallel study, we find data augmentation is a promising solution. For future work, we aim to delve deeper into these challenges and explore potential strategies for enhancing the performance of sexism detection models. Additionally, we plan to expand the scope of our dataset to include more diverse modalities and cultural contexts, thereby enriching the resources available for research.

# Ethical Statement

BeyondGender is developed with the aim of improving the distinction between actual sexism and gender-related discussions, as well as between innocuous stereotypes and sneaky sexist ideologies. It is sourced from a combination of previous public datasets and social media, with no personal information collected during this process. The annotation process incorporates perspectives from both male and female annotators to reduce the potential for gender bias. The dataset primarily represents data from recent decades and does not necessarily reflect the historical or future trends in sexism.

BeyondGender is intended solely for academic research purposes and will be made publicly available. We are not responsible for any potential breaches and misuse by others.