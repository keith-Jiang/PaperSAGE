# Importance Weighting Can Help Large Language Models Self-Improve

Chunyang Jiang, Chi-Min Chan, Wei XueB, Qifeng Liu, Yike GuoB

Hong Kong University of Science and Technology {cjiangaq, cchanbc}@connect.ust.hk, {weixue, liuqifeng, yikeguo}@ust.hk

# Abstract

Large language models (LLMs) have shown remarkable capability in numerous tasks and applications. However, finetuning LLMs using high-quality datasets under external supervision remains prohibitively expensive. In response, LLM self-improvement approaches have been vibrantly developed recently. The typical paradigm of LLM self-improvement involves training LLM on self-generated data, part of which may be detrimental and should be filtered out due to the unstable data quality. While current works primarily employs filtering strategies based on answer correctness, in this paper, we demonstrate that filtering out correct but with high distribution shift extent (DSE) samples could also benefit the results of self-improvement. Given that the actual sample distribution is usually inaccessible, we propose a new metric called DS weight to approximate DSE, inspired by the Importance Weighting methods. Consequently, we integrate DS weight with self-consistency to comprehensively filter the self-generated samples and fine-tune the language model. Experiments show that with only a tiny valid set (up to $5 \%$ size of the training set) to compute DS weight, our approach can notably promote the reasoning ability of current LLM self-improvement methods. The resulting performance is on par with methods that rely on external supervision from pre-trained reward models.

# Introduction

Recently, Large Language Models (LLMs) have made impressive achievements on a large amount of NLP tasks and applications (Li et al. 2023a; OpenAI 2023; Yang et al. 2023; Li et al. 2023b). Moreover, new capabilities emerge in LLMs with the model size scaled to hundreds of billions of parameters, especially the general reasoning capabilities (Kojima et al. 2022). Relevant techniques like in-context few-shot learning (Brown et al. 2020a), Chain-of-Thought prompting (Wei et al. 2022), and self-consistency (Wang et al. 2023a) were further proposed to get better performance.

Despite the remarkable capabilities of LLMs pre-trained on the large corpus, fundamentally improving the model’s performance still necessitates fine-tuning on a great amount of high-quality supervised data (Huang et al. 2023a), which is usually costly. To alleviate this problem, many works are committed to investigating the self-improvement ability of

LLMs (Shinn et al. 2023; Madaan et al. 2023; Vernikos et al. 2024). Among them, fine-tuning the LLM on self-generated data appears as one of the most promising way (Gu¨lc¸ehre et al. 2023; Huang et al. 2023a; Wang et al. 2023b; Xu et al. 2023; Li et al. 2024). This formula typically includes generating reasoning thoughts and answers on unsupervised datasets, filtering data, and fine-tuning models on the selfgenerated data (Huang et al. 2023a). It is regarded as an attractive approach for LLMs to self-supervise by utilizing unlabeled data without external supervision.

The primary challenge of utilizing self-generated data is the variability in data quality. While high-quality samples can enhance the model’s reasoning abilities, there are low-quality samples that may detrimentally affect performance (Li and Qiu 2023). For example, an incorrectly generated answer could mislead the model. Therefore, a good filtering strategy is decisive for effective self-improvement. Many approaches have been proposed to address this issue. Inspired by Self-Consistency (Wang et al. 2023a), LMSI (Huang et al. 2023a) adopts majority voting to select the most consistent answer, under the assumption that consistency is positively related to the correctness. MoT (Li and Qiu 2023) further introduces uncertainty to the filtering strategy, by utilizing entropy to exclude high-uncertainty data points. Self-Alignment (Li et al. 2024) demonstrates that prompting the LLM to self-filter is also feasible.

However, present methods mostly emphasize assessing the correctness of generated samples, yet ignore the distribution shift problem. Specifically, the distribution of the LLM self-generated data may differ from that of real-world data, and fine-tuning models on samples with high distribution shift extent (DSE) may defect the resulting performance (Shumailov et al. 2023a). In this paper, we demonstrate that even self-generated samples with correct answers can possess high DSE, potentially degrading model performance. Consequently, filtering out high DSE samples is essential to further promote the efficacy of LLM selfimprovement.

To exclude samples with high DSE, the primary question is how to estimate the DSE, since the actual data distribution is usually inaccessible. We note Importance Weighting (IW) (Sugiyama, Krauledat, and Mu¨ller 2007) as a wellknown approach to address the traditional distribution shift problem (Sugiyama and Kawanabe 2012), where the key idea is deriving importance weights based on the distribution ratio between test and training data, and using it to rebuild an unbiased training loss. IW usually contains two steps: weight estimation computes test-over-training density ratio and weighted classification utilizes the ratio to weight each data point and train the model (Fang et al. 2020).

Inspired by IW, we propose Distribution Shift Weight (DS weight) as a new metric to measure the DSE of selfgenerated samples. Based on this, we build an LLM selfimprovement framework that incorporates both the correctness and DSE in its filtering strategy. Specifically, given a question-only dataset, we first let a pre-trained LLM generate multiple reasoning thoughts as well as answers. Then we create a tiny valid set comprising a few human-written demonstrations. With the pre-trained LLM and valid set, we leverages a simple approximation for importance weights to compute DS weight, as a measure of DSE, for each training data point. We subsequently combine the results from majority voting (for correctness) and DS weight (for DSE) to filter the dataset and fine-tune the LLM. We denote our framework as Importance Weighting-based SelfImprovement (IWSI). Experiments show that the performance of IWSI largely surpasses baseline self-improvement methods and rivals the enhancements achieved with supervision from the pre-trained reward model.

Our contributions are threefold: (1) We propose a metric called DS weight to approximate the DSE of LLM selfgenerated data, with help from a tiny valid set. (2) Leveraging DS weight, we build a novel self-improvement framework called IWSI where the filtering strategy considers both the answer correctness and DSE. (3) We empirically examine the effectiveness of our proposed method, analyze the impact of high DSE samples on LLM self-improvement, and explore how DS weight interacts with other filtering criteria.

# Related Work

# LLM Self-Improvement

Fundamentally improving LLMs’ reasoning ability essentially requires fine-tuning on a large amount of highquality supervised data. However, this methodology faces the threat that the stock of high-quality language data will be exhausted in some day (Villalobos et al. 2022). Selfimprovement emerges as a promising approach to utilize the inherent knowledge to make supervision for self-training LLMs. While LLMs can easily generate extensive data, the data quality is not always guaranteed (Huang et al. 2023b) and training on unfiltered data may even cause performance degradation (Shumailov et al. 2023b). Therefore, an essential requirement in LLM self-improvement is data filtering.

Pioneering works (Wang et al. 2023b; Bai et al. 2022; Xu et al. 2023) use language models to generate diverse types of data such as feedback, instructions, and questions. They filter data by heuristic rules as well as manual inspection, which is challenging and costly. LMSI (Huang et al. 2023a) proposed a framework including generating data for a question-only dataset and using the majority voting (selfconsistency) (Wang et al. 2023a) to select the most consistent answers, which is empirically proven to be effective among various tasks. LMSI also demonstrates that the answer correctness is positively relevant to self-consistency. Along with this work, MoT (Li and Qiu 2023) proposes further filtering the consistent answers by entropy, which measures the answer uncertainty. Self-Alignment (Li et al. 2024) shows it is feasible to prompt the LLM self-filtering the generated data. To comprehensively evaluate the generated data, some works use external pre-trained LMs as the reward model to score the generated data, such as GENIE (Yehudai et al. 2024) and ReST (Gu¨lc¸ehre et al. 2023). With external supervision from the reward model, their filtering strategies are typically more considered.

# Importance Weighting

Importance weighting (IW) is a primary approach to mitigate the influence of distribution shift problem (Sugiyama and Kawanabe 2012). The typical IW process includes two steps: weight estimation and weighted classification. Weight estimation approximates the importance weights, which are subsequently used in the weighted classification stage to build a weighted training loss (Fang et al. 2023).

Traditional IW methods mainly estimate the importance weights by assessing the matching between training and test distribution in different ways, such as maximum mean discrepancy in a reproducing kernel Hilbert space (Huang et al. 2006), KL divergence (Sugiyama et al. 2007), and squared loss (Kanamori, Hido, and Sugiyama 2009). While these methods work well in linear models, their performances degrade largely in deep learning scenarios (Fang et al. 2020). To overcome this, DIW (Fang et al. 2020) proposes an endto-end dynamic solution, which uses a deep network to predict the importance weights, and repeats weight estimation and weighted classification stages to iteratively converge on the optimal solution.

In this paper, we use some lemmas and empirical results in DIW to build the DS weight for estimating the DSE of self-generated data.

# Methodology

Fig. 1 shows the overview of IWSI. Given an unsupervised (question-only) dataset $\mathcal { D } _ { q }$ , we first use the pre-trained LLM $\mathbf { \mathcal { M } } _ { L }$ to generate multiple candidate answers as well as the reasoning thoughts for each question, using CoT prompts (Wei et al. 2022). Following LMSI (Huang et al. 2023a), we adopt the majority voting to keep the most consistent answer and corresponding thoughts for each question, resulting in the consistency-filtered dataset $\mathcal { D } _ { c }$ . Then we calculate $D S$ weight for every data point in $\mathcal { D } _ { c }$ , with the help of a tiny valid set $\mathcal { D } _ { \nu }$ . Lastly, we filter $\mathcal { D } _ { c }$ into $\mathcal { D } _ { d s }$ utilizing the DS weight and fine-tune the model $\boldsymbol { \mathcal { M } } _ { L }$ . The following sections elaborate on different components of IWSI.

# Candidate Answers Generation and Self-Consistency Filtration

In this stage, we let the pre-trained LLM $\boldsymbol { \mathcal { M } } _ { L }$ generate candidate answers as well as reasoning thoughts for an unsupervised dataset $\mathcal { D } _ { q }$ which only contains unlabeled questions. Given a question $q _ { i } \in \mathcal { D } _ { q }$ , we concatenate Few

Candidate Answers Generation and Self-Consistency Filteration 1 Q: 3 cars in the parking lot 120 pages will take 1 and 2 more cars arrive, 5h. The answer is 5. ： how many cars now? 120 pages will take A: 3 + 2 = 5 cars. The 8 pages take 20 min. 5h. The answer is 5. aQn:s..w..e..r is 5. CoT prompts .1..0hT.hTathiesa6n0s0we/r6i0s 1=0. Unsupervised Dataset 2Q0:  Jmoiyn.c aHnorweamda8npyahgoeusrisn Generate ： Majority .1..2..0. pTahgaetsism3u0s0t/b6e0 pAwa:ilgleist?take her to read 120 CAansdwideartse .15..2.h.0. pTahgeaetasinsms3wu0se0tr /ib6se05. Voting Store Q-A D DS Weight Q: Joy can read 8 pages in 20 min. How many Computation hours will it take her to read 120 pages? A: 120 pages will take 5h. The answer is 5. Cx²∈D,L(ML(x²)) Dc Q: Weng earns $\$ 12$ an hour. Yesterday, she w NU·C(ML(xt)) $\mathcal { M } _ { L }$ jAu: $1 2 \mathrm { ~ / ~ } 6 ^ { * } 5 = \$ 1 0$ .oTwhemauncshwdeirdish1e0e.arn $w _ { i } ^ { D S } = ( w _ { i } ^ { \prime } ) ^ { ( 1 - 2 \cdot 1 ( w _ { i } ^ { \prime } < 1 ) ) }$ Q: John writes 20 pages a day.  How long will it take him to write 1200 pages? Du A: 1200 $2 0 { = } 6 0$ days. The answer is 60. valid set Self-training Utilizing DS Weight to Improve LLM Dds Filtering out examples with high DS weight wDS = 1.02 wDs =3.24 w3 DS = 1.55 ws ：2.26 wDs = 1.33 X X 1 x1 x2 x3 x4 Cn 1

Shot-CoT (Wei et al. 2022) prompts with $q _ { i }$ to form the input text $x _ { i }$ . With temperature $T \ > \ 0$ , we let $\mathbf { \mathcal { M } } _ { L }$ sample $m$ candidate answers $[ a _ { i _ { 1 } } , a _ { i _ { 2 } } , \ldots , a _ { i _ { m } } ]$ and their reasoning thoughts $[ r _ { i _ { 1 } } , r _ { i _ { 2 } } , \ldots , r _ { i _ { m } } ]$ . Then we select the most consistent answer $\hat { a _ { i } }$ by majority voting (Wang et al. 2023a), $\begin{array} { r } { \hat { a _ { i } } = \arg \operatorname* { m a x } _ { a _ { i _ { j } } } \sum _ { k = 1 } ^ { m } \mathbb { 1 } \left( a _ { i _ { j } } = a _ { i _ { k } } \right) } \end{array}$ , and keep the corresponding reasoning thoughts $R _ { i } = \{ r _ { i _ { j } } | a _ { i _ { j } } = \hat { a _ { i } } , 1 \leq j \leq m \}$ . By repeating over each question in $\mathcal { D } _ { q }$ , the consistency-filtered dataset $\mathcal { D } _ { c }$ is built.

# DS Weight Computation

To elaborate $D S$ Weight, we first introduce some important preliminaries in the distribution shift problem and importance weighting methods.

Distribution shift problem denotes that the training data and test data are drawn from two different distributions $p _ { t r a i n }$ and $p _ { t e s t }$ , and $p _ { t r a i n } \neq p _ { t e s t }$ (Sugiyama and Kawanabe 2012). A common assumption for distribution shift is that there exists a function $\boldsymbol { w } ^ { * } ( \boldsymbol { x } )$ , holding that:

$$
\mathbb { E } _ { p _ { t e s t } ( x ) } [ f ( x ) ] = \mathbb { E } _ { p _ { t r a i n } ( x ) } [ w ^ { * } ( x ) \cdot f ( x ) ]
$$

for any function $f$ of $x$ (Fang et al. 2020). Based on Eq. 1, importance weighting methods (Sugiyama, Krauledat, and M¨uller 2007; Sugiyama et al. 2007) deal with distribution shift in two steps: weight estimation finds a proper solution for $\boldsymbol { w } ^ { * } ( \boldsymbol { x } )$ ; weighted classification trains the model with a weighted loss derived by substituting $f$ in Eq. 1 with the target loss function.

Obviously, it plays a decisive role in importance weighting that finding the appropriate importance weights $\mathcal { W } =$ $\{ w _ { i } \} ^ { N _ { t } }$ , to approximate $\boldsymbol { w } ^ { * } ( \boldsymbol { x } )$ in Eq. 1. To simplify the question, DIW (Fang et al. 2020) provides an empirical surrogate goal with the help of a valid set:

$$
\frac { 1 } { N _ { \nu } } \sum _ { j = 1 } ^ { N _ { \nu } } \mathcal { L } ( \mathcal { M } ( x _ { j } ^ { \nu } ) ) \approx \frac { 1 } { N _ { t } } \sum _ { i = 1 } ^ { N _ { t } } w _ { i } \cdot \mathcal { L } ( \mathcal { M } ( x _ { i } ^ { t } ) ) .
$$

Here $N _ { \nu } , N _ { t } , x ^ { \nu }$ , and $x ^ { t }$ indicate the size of the valid set, the size of the training set, data in the valid set, and data in the training set. $\mathcal { M }$ is the training model and $\mathcal { L }$ represents the training loss.

While in DIW, Eq. 2 is used as a goal to train a deep model that predicts the desired $\mathcal { W }$ , we use Eq. 2 to design a naive measurement for the distribution shift extent between training samples and valid set. Our intuition is that when the training data distribution is identical to the valid data distribution, $w _ { i } \ \equiv \ 1$ would be a proper solution to Eq. 2. Conversely, the larger the actual $w _ { i }$ differs from 1, the more different the training distribution and valid distribution are.

Based on this idea, we first design a naive estimation $w _ { i } ^ { \prime }$ for $x _ { i } ^ { t }$ by regarding $N _ { t }$ as 1:

$$
w _ { i } ^ { \prime } = \frac { \sum _ { \boldsymbol { x } _ { j } ^ { \nu } \in \mathcal { D } _ { \nu } } \mathcal { L } ( \boldsymbol { M } _ { L } ( \boldsymbol { x } _ { j } ^ { \nu } ) ) } { N _ { \nu } \cdot \mathcal { L } ( \boldsymbol { M } _ { L } ( \boldsymbol { x } _ { i } ^ { t } ) ) }
$$

where $\boldsymbol { \mathcal { M } } _ { L }$ is the pre-trained LLM, $\mathcal { L }$ denotes the sft loss (Brown et al. 2020b), $\mathcal { D } _ { \nu }$ is a tiny valid set and $x _ { i } ^ { t }$ is a self-generated training data point. Here we notice that the value range of $w _ { i } ^ { \prime }$ is $( 0 , + \infty )$ while the ideal value is 1, which creates asymmetry between the two deviation directions (lower than 1 and greater than 1) and makes filtering inconvenient. Therefore, to establish symmetry for both shift directions, we define $D S$ weight $w _ { i } ^ { D S }$ as:

$$
w _ { i } ^ { D S } = \left\{ \begin{array} { l l } { w _ { i } ^ { \prime } } & { \mathrm { i f } ~ w _ { i } ^ { \prime } \geq 1 } \\ { \frac { 1 } { w _ { i } ^ { \prime } } } & { \mathrm { i f } ~ w _ { i } ^ { \prime } < 1 } \end{array} \right.
$$

# Utilizing DS Weight to Improve LLM

With DS weight approximating DSE, we are able to further filter the self-generated data in $\mathcal { D } _ { c }$ , excluding data points that possibly possess higher DSE.

First, all data points are ranked with respect to their DS weight $w _ { i } ^ { D S }$ , and the $k$ -percentile $\sigma _ { k \% }$ is selected, s.t.

$$
\frac { \sum _ { i } ^ { | \mathcal { D } _ { c } | } \mathbb { 1 } \left( w _ { i } ^ { D S } \leq \sigma _ { k ^ { \% } } \right) } { | \mathcal { D } _ { c } | } = k ^ { \% }
$$

where $| \cdot |$ denotes the set size and $w _ { i } ^ { D S }$ is the corresponding DS weight of sample $x _ { i }$ . As a result, only samples whose $w _ { i } ^ { D S } \leq \sigma _ { k \% }$ are kept to train the model $\boldsymbol { \mathcal { M } } _ { L }$ . The training loss can be written as:

$$
\mathcal { L } _ { F } = \frac { 1 } { \vert \mathcal { D } _ { c } \vert \cdot k ^ { \mathrm { e } } / _ { 0 } } \sum _ { x _ { i } } ^ { \mathcal { D } _ { c } } \mathbb { 1 } _ { k ^ { \mathrm { e } / _ { 0 } } } ( x _ { i } ) \cdot \mathcal { L } ( \mathcal { M } _ { L } ( x _ { i } ) )
$$

where $\mathbb { 1 } _ { k \% } ( x _ { i } )$ equals to $\mathbb { 1 } \left( w _ { i } ^ { D S } \ \leq \ \sigma _ { k \% } \right)$ and $\mathcal { L }$ represents the sft loss.

Another natural way to utilize DS weight is directly employing Eq. 3 to calculate a weighted loss, which is more analogous to the standard IW procedure. We also implement this variant in our work and denote it as IWSI-w. The weighted loss is:

$$
\mathcal { L } _ { W } = \frac { 1 } { | \mathcal { D } _ { c } | } \sum _ { x _ { i } } ^ { \mathcal { D } _ { c } } C l i p ( w _ { i } ^ { \prime } , C ) \cdot \mathcal { L } ( \mathcal { M } _ { L } ( x _ { i } ) )
$$

where $C$ is a constant. We clip $w _ { i } ^ { \prime }$ to $( 0 , C ]$ for stabilizing the training process.

However, we found that IWSI-w is much less effective than IWSI. We believe this is mainly attributed to the inadequacy of Eq. 3. Empirical results and details are discussed in the experiment section.

# Experiment

# Setup

Datasets We conduct experiments on six datasets across three types of tasks: Arithmetic Reasoning: gsm8k (Cobbe et al. 2021) and SVAMP (Patel, Bhattamishra, and Goyal 2021). Natural Language Inference: Adversarial NLI subsets (Nie et al. 2020). ANLI-A1 and ANLI-A2 subsets are used. Commonsense Reasoning: OpenBookQA (Mihaylov et al. 2018) and StrategyQA (Geva et al. 2021).

For all datasets, only the questions are used to selfgenerate candidate answers. For gsm8k and SVAMP, we keep the original question format, which is the open-ended question. For the other four datasets, we unify the question format to the multiple choice question. The LLM must choose one option as its answer.

To build the valid set, we extract rationales from the original datasets apart from SVAMP, for which we manually write rationales. The size of valid sets varies among different datasets, but none of them exceeds $5 \%$ size of the corresponding training set. Appendix A provides more details about the split and statistics of all datasets.

Baselines The goal of our experiments is to verify whether incorporating DS weight into the filtering strategy in our proposed approach can help LLMs self-improve. Therefore, given the same base model, we compare IWSI with the fundamental self-improvement framework LMSI (Huang et al. 2023a), and some variants that we implement by adopting trendy filtering strategies designed for training LLMs on model-generated data.

LMSI (Huang et al. 2023a) is the first self-improvement framework that significantly improves LLMs’ reasoning ability without any external supervision. The core idea of LMSI is adopting majority voting to select answers that are most likely correct, thus filtering the self-generated data.

MoT (Li and Qiu 2023) uses entropy to measure the uncertainty of the answers and further filters data. We combine this technique with LMSI and denote it as Entropy-filter.

Self-Alignment (Li et al. 2024) shows that LLM selfevaluation could be helpful in filtering strategy. We implement this idea with LMSI and denote it as Self-filter.

Works like GENIE (Yehudai et al. 2024) and ReST (Gu¨lc¸ehre et al. 2023) use pre-trained models to evaluate the self-generated samples. Intervened by external supervision, their filtering results are usually more comprehensive and meticulous. Following that, we also implement a variant of LMSI for reference, the RM-filter. RM-filter uses a pre-trained reward model to score the generated data, as GENIE (Yehudai et al. 2024) does. 1

Table 1: Accuracy results on all datasets. Numbers in the table are the accuracy percent. The first part is the performance of the base model. The second part is the performance of three baseline self-improvement methods, our proposed method IWSI, and a variant IWSI-w. As RM-filter uses the external reward model, we list its performance separately at the bottom of the table.   

<html><body><table><tr><td></td><td>gsm8k</td><td>SVAMP</td><td>ANLI-A1</td><td>ANLI-A2</td><td>OpenBookQA</td><td>StrategyQA</td><td>Avg.</td></tr><tr><td>base</td><td>7.0</td><td>14.7</td><td>16.4</td><td>14.6</td><td>31.8</td><td>48.3</td><td>22.1</td></tr><tr><td>LMSI</td><td>27.9</td><td>45.0</td><td>25.2</td><td>22.6</td><td>31.6</td><td>51.4</td><td>34.0</td></tr><tr><td>Entropy-filter</td><td>22.7</td><td>56.0</td><td>25.2</td><td>22.8</td><td>33.4</td><td>51.2</td><td>35.2</td></tr><tr><td>Self-filter</td><td>35.6</td><td>62.7</td><td>22.8</td><td>25.6</td><td>35.0</td><td>50.4</td><td>38.7</td></tr><tr><td>IWSI-w</td><td>37.0</td><td>43.3</td><td>21.8</td><td>21.8</td><td>31.8</td><td>49.2</td><td>34.2</td></tr><tr><td>IWSI</td><td>37.6</td><td>62.7</td><td>27.2</td><td>23.4</td><td>37.0</td><td>54.6</td><td>40.4</td></tr><tr><td>RM-filter</td><td>40.0</td><td>66.3</td><td>25.6</td><td>25.0</td><td>34.2</td><td>51.4</td><td>40.4</td></tr></table></body></html>

Implementation details We select Llama3-8B as our base model (Touvron et al. 2023). For each question, we generate 15 candidates, with temperature $T ~ = ~ 1 . 1$ . All training process is performed on eight RTX-4090 GPUs. The training batch size per device is set to 1 and the gradient accumulation steps is 4. We use LoRA (Hu et al. 2022) to do fine-tuning. We use AdamW (Loshchilov and Hutter 2019) optimizer and the learning rate is 3e-4. Few-Shot-CoT prompts are only applied in generating candidate answers and the evaluation stage. CoT examples for each dataset, prompts used for Self-filter, and details about how to derive the answer from output texts are given in Appendix D. The source code and supplementary materials are available at https://github.com/rubickkcibur/IWSI.

# Main Results

The main comparison results are shown in Table 1. The evaluation metric is accuracy percent and all results are derived by greedy decoding. The top part is the performance of the base model. The middle part are self-improvement baselines and our proposed method IWSI. For reference, we list the performance of RM-filter at the bottom of the table. For fairness, we universally set the filtering percentage $k = 8 0$ for IWSI, Entropy-filter, Self-filter, and RM-filter.

Among self-improvement methods (the middle part), IWSI is the only one that consistently outperforms LMSI, and it also achieves the best in almost all datasets. We further empirically demonstrate that the superiority of IWSI primarily stems from excluding self-generated samples with higher DSE, rather than merely from access to part of the information of the valid set (the mean loss value of valid samples). Details are in Appendix F.

For IWSI-w, the variant of IWSI that uses DS weight to compute weighted loss other than filtering data, it generally performs worse than IWSI, even though IWSI-w is more compliant with the standard importance weighting formula. The most possible reason is that unlike deep methods like DIW (Fang et al. 2020), which uses a deep neural network to learn the weights, our weight estimation (Eq. 3) is a pretty naive approach. While it largely reduces computational cost, it also omits the semantic similarity among training samples, potentially compromising efficacy. Therefore, the weighted loss in IWSI-w might make the training process difficult and noisy. In contrast, IWSI only uses the weight as an indicator to rank the samples with respect to DSE, without directly incorporating the weight into the training loss, which makes the overall process more robust.

As for the RM-filter, we found that it does not always perform the best among all six datasets, even though it introduces external supervision by using a pre-trained reward model. As Table 1 shows, after incorporating both the answer correctness and DSE of samples, the overall performance of IWSI is comparable to that achieved with external supervision from a pre-trained reward model.

# Hyperparameter Study

We investigate the effect of varying the filtering threshold $k$ and corresponding percentile $\sigma _ { k \% }$ (in Eq. 5). Fig. 2 shows the accuracy results on gsm8k, StrategyQA, and ANLI-A1. As the figure shows, either a too-large or too-small $k$ value will make the performance degrade. When $k$ is very large, more samples with high DSE will be kept, thus potentially harming the performance. If the $k$ is pretty small, there will not be sufficient samples kept to support the model training. The optimal $k$ value range varies across different tasks. In general, around $80 \%$ would be an appropriate choice.

Fig. 3 shows the varying $k$ -percentile $\sigma _ { k \% }$ of DS weight. While $\sigma _ { k \% }$ of different datasets are similar when $k$ is very small, the difference becomes larger as $k$ increases. This phenomenon suggests that the boundary above which the DSE of samples can be regarded as ”high” is relative according to different datasets.

# Valid Set Analysis

The valid set $\mathcal { D } _ { \nu }$ plays a crucial role in IWSI. It determines the calculation results of DS weight and subsequently steers the filtering strategy. Therefore, variation in the composition of the valid set can introduce randomness and thus potential instability. In this section, we take the gsm8k dataset as example to discuss the impact of valid set.

We employ the loss value distribution as the analytical tool and, for simplicity, we assume all distributions of different sample sets conform to the normal distribution. For example, the loss value distribution of valid set is denoted as $\bar { \mathcal { N } _ { \nu } ( \mu _ { \nu } , \sigma _ { \nu } ^ { 2 } ) }$ , where $\mu _ { \nu }$ and $\sigma _ { \nu }$ are the mean and standard deviation respectively.

![](images/e59ef356226e95df9d7835aa6f87b40c52bc15b1072bfd3715635c532133a732.jpg)  
Figure 2: Accuracy results with varying $k$ values.

![](images/37a7b6b5d0eb77a199826984ed7cf431b3b45958e9cd93e02323996321695ce7.jpg)  
Figure 3: $\sigma _ { k \% }$ (in Eq. 5) with varying $k$ values.

Fig. 4 shows distributions of the valid set and selfgenerated samples before and after IWSI. Analogous to our intuition, the distributions differ significantly between valid set samples and self-generated samples before IWSI, and become much closer after IWSI, illustrating the effectiveness of IWSI in handling the distribution shift problem. Furthermore, we provide quantitative analyses and a case study in Appendix E for a better understanding of how the LLM generation was affected by IWSI.

The next question is would the randomness of valid set composition cause great instability in IWSI, since $\sigma _ { \nu }$ is apparently not small enough. The answer is $" \mathrm { N o } ^ { \prime \prime }$ as long as there is an adequate valid set size $N _ { \nu }$ . Theoretically, in Eq. 3, it is only the sample mean, denoted as ${ \bar { \mathcal { L } } } _ { \nu }$ , that matters. $\bar { \mathcal { L } } _ { \nu }$ is also subject to the normal distribution, with its standard deviation inversely proportional to the size $N _ { \nu }$ :

$$
\bar { \mathcal { L } } _ { \nu } = \frac { \sum _ { x _ { j } ^ { \nu } \in \mathcal { D } _ { \nu } } \mathcal { L } ( M _ { L } ( x _ { j } ^ { \nu } ) ) } { N _ { \nu } }
$$

Eq. 8 implies that increasing $N _ { \nu }$ can scale down the variance of $\bar { \mathcal { L } } _ { \nu }$ , thus making the estimation more stable. More importantly, it is completely irrelevant to the size of the training samples. For instance, in gsm8k, if the valid set size is 100, the standard deviation of $\bar { \bar { \mathcal { L } } } _ { \nu }$ is $\begin{array} { r } { \bar { \sigma _ { \nu } } = \frac { \sigma _ { \nu } } { 1 0 0 } = 4 . 1 { \times } 1 0 ^ { - 3 } } \end{array}$ , which is small enough to mitigate the interference of randomness.

Table 2: Results of different valid sets on gsm8k.   

<html><body><table><tr><td rowspan="2">Nv</td><td colspan="2">Comp.1</td><td colspan="2">Comp.2</td><td colspan="2">Comp.3</td></tr><tr><td>Lv</td><td>acc</td><td>L</td><td>acc</td><td>L</td><td>acc</td></tr><tr><td>50</td><td>2.053</td><td>36.77</td><td>2.083</td><td>36.54</td><td>2.091</td><td>36.47</td></tr><tr><td>100</td><td>2.077</td><td>36.69</td><td>2.086</td><td>36.47</td><td>2.054</td><td>37.00</td></tr></table></body></html>

![](images/b51c193ee15d9418dd048776c02de17bc14c5d2ecf9ae3a53f81f067cc7c165b.jpg)  
Figure 4: Loss value distributions of the valid set samples, self-generated samples of base model (generated-base), and self-generated samples after IWSI (generated-IWSI), of $\mathrm { g s m 8 k }$ . $\mu$ and $\sigma$ denote the mean and standard deviation.

To empirically investigate the influence of different valid set compositions, we randomly constitute six subsets of the valid set of $\mathrm { g s m 8 k }$ and test IWSI with them. Table 2 shows the results. $N _ { \nu }$ denotes the valid set size. ${ \bar { \mathcal { L } } } _ { \nu }$ is the sample mean of different composition. We use acc as the metric.

As we can see, the impact of different compositions on the accuracy results is quite minimal. We believe this is primarily attributed to the double-robustness of IWSI. First, the DS weight calculation is robust to the valid set composition, since it only uses the sample mean which varies vary little. Furthermore, the filtering strategy is also robust to the DS weight, since the DS weight is used for ranking other than weighting. As a result, samples with extremely high DSE are probably always discarded even if DS weight changes.

# Orthogonality Analysis

In IWSI, two factors are considered in the filtering strategy, the answer correctness (represented by self-consistency) and the sample DSE (represented by DS weight). A natural question is what is the relationship between these two factors. Are they correlated to or independent of each other? To explore this question, we counted the percentage of samples with correct answers (using the ground truth labels) across different DS weight intervals, as Fig 5 shows. Along $x$ - axis are the selected intervals: 1, 1.1 , 1.1, 1.3 , 1.3, 1.5 , 1.5, 2 , and $[ 2 , \infty )$ . In each bar, the upper portion (yellow) indicates the ratio of correct answers, while the lower portion (blue) represents the ratio of wrong answers. For all datasets, we observe a general downward trend in the ratio of correct answers, as DS weight increases. The highest ratios of correct answers is found either in the 1, 1.1 interval (for $\mathrm { g s m 8 k }$ and ANLI-A1) or in the 1.1, 1.3 interval (for StrategyQA). However, both correct and wrong answers occupy a

i 1.0 0.8 山 0.6 0.4 0.2 0.0 [1,1.1)[1.1,1.3)[1.3,1.5)[1.5,2.0)[2.0,∞) u\*=0.3 u\*=0.05 u\*=0.1 u\*=0.4 u\*=0.1 u\*=0.15 u\*=0.6 u\*=0.15 u\*=0.25   
GR =1.0 =0.25 \*=0.5   
1.0 2.0 3.0 1.0 2.0 3.0 1.0 2.0 3.0 wDs wDS wDs gsm8k StrategyQA ANLI-A1

portion that can not be ignored in every interval, suggesting a degree of independence between these two factors.

We delve deeper into the relationship between DSE and the answer uncertainty, which is first investigated by MoT (Li and Qiu 2023) regarding its impact on selfimprovement. MoT also suggested using entropy to represent answer uncertainty. We briefly introduce the calculation: given a certain question $q$ , the self-generated candidate answers $[ a _ { 1 } , a _ { 2 } , \ldots , a _ { m } ]$ , and the most consistent answer $\hat { a }$ , uncertainty $u$ is computed in the following steps:

$$
\begin{array} { l } { { \displaystyle { \cal A } ^ { * } = u n i q u e ( \{ a _ { i } \} ^ { m } ) } } \\ { { \displaystyle p ( a _ { j } ^ { * } ) = \sum _ { i } ^ { m } \mathbb { 1 } ( a _ { j } ^ { * } = a _ { i } ) / m } } \\ { { \displaystyle u = - \sum _ { a _ { j } ^ { * } } ^ { A ^ { * } } p ( a _ { j } ^ { * } ) \log p ( a _ { j } ^ { * } ) } } \end{array}
$$

where $A ^ { * } ~ = ~ \{ a _ { 1 } ^ { * } , a _ { 2 } ^ { * } , \cdot \cdot \cdot \}$ is the unique answer set. The higher $u$ is, the more uncertain the answer is. In extreme cases, if $u = 0$ , all candidate answers are identical, and if each candidate answer has its unique value, $u$ will reach the maximum $\log m$ . For convenience, we normalize $u$ with a divisor log $m$ and we denote the filter threshold as $u ^ { * }$ .

We draw the probability density function (PDF) of DS weight for various uncertainty thresholds $u ^ { * }$ . The second row of Fig. 5 shows the results. For arithmetic reasoning (gsm8k), as $u ^ { * }$ increases, the peak of PDF falls and the PDF curve becomes flatter, indicating a growth in the proportion of samples with high DSE. Conversely, for commonsense reasoning (StrategyQA) and natural language inference (ANLI-A1), the relationship between uncertainty and DSE appears much weaker. The PDF curves are almost identical, with little variation at the peak, suggesting that DSE is nearly orthogonal to the uncertainty.

# Perception of DSE

We conducted a case study on gsm8k to provide an intuitive perception about what a correct but with high DSE sample looks like. We compare the generated answers with the highest and lowest DSE for the same question. We found that cases with the highest DSE are usually notably absurd that we can easily tell them apart from human-written samples. We categorize these samples into 3 types:

• Redundant samples. Redundant samples include irrelevant or repeated information in the reasoning thoughts, making it confusing.   
• Jumping samples. Jumping samples omit essential reasoning steps or even directly give the answer, making it less logically fluent.   
• Spurious samples. The reasoning steps in a spurious sample (Guu et al. 2017; Jiang et al. 2023) are logically wrong. They get the correct answer just by coincidence.

We give more exact demonstrations in Appendix B.

# Conclusion

In this paper, we investigate the impact of sample DSE on LLM self-improvement. We propose $D S$ weight to approximate the DSE inspired by importance weighting methods, and a novel framework IWSI where the filtering strategy comprehensively considers DSE and answer correctness. Empirical results demonstrate that the incorporation of DS weight significantly enhances the effectiveness of LLM selfimprovement. Further analysis reveals that DSE is nearly orthogonal to other factors, suggesting a new direction to promote LLM self-improvement for the future work.