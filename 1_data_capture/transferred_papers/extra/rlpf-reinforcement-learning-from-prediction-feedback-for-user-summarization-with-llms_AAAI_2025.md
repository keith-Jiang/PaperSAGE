# RLPF: Reinforcement Learning from Prediction Feedback for User Summarization with LLMs

Jiaxing Wu, Lin Ning, Luyang Liu, Harrison Lee, Neo Wu, Chao Wang, Sushant Prakash, Shawn O’Banion, Bradley Green, Jun Xie

Google DeepMind jxwu@google.com

# Abstract

LLM-powered personalization agent systems employ Large Language Models (LLMs) to predict users’ behavior from their past activities. However, their effectiveness often hinges on the ability to effectively leverage extensive, long user historical data due to its inherent noise and length of such data. Existing pretrained LLMs may generate summaries that are concise but lack the necessary context for downstream tasks, hindering their utility in personalization systems. To address these challenges, we introduce Reinforcement Learning from Prediction Feedback (RLPF). RLPF fine-tunes LLMs to generate concise, human-readable user summaries that are optimized for downstream task performance. By maximizing the usefulness of the generated summaries, RLPF effectively distills extensive user history data while preserving essential information for downstream tasks. Our empirical evaluation demonstrates significant improvements in both extrinsic downstream task utility and intrinsic summary quality, surpassing baseline methods by up to $22 \%$ on downstream task performance and achieving an up to $8 4 . 5 9 \%$ win rate on Factuality, Abstractiveness, and Readability. RLPF also achieves a remarkable $74 \%$ reduction in context length while improving performance on 16 out of 19 unseen tasks and/or datasets, showcasing its generalizability. This approach offers a promising solution for enhancing LLM personalization by effectively transforming long, noisy user histories into informative and human-readable representations.

# 1 Introduction

Large Language Models (LLMs) have shown great promise for personalized prediction by leveraging historical activity data (Liu et al. 2023; Lyu et al. 2024; Li et al. 2023). However, the inherent noise and length of user data pose obstacles to their effective utilization in LLM-powered systems.

Natural language user summaries offer several advantages over using raw user activity data. First, they improve inference efficiency over using raw user data due to their compact nature. Second, they offer the potential to improve performance on downstream tasks by distilling user activities and reducing noise. Representing user context through natural language also offers several advantages over embeddingbased representations. User representations in the natural language space are reusable across any LLM for downstream tasks without needing to re-train the LLM. In addition, natural language summaries are interpretable and editable, offering users more scrutability and control over their personalized experiences.

Generating user summaries is inherently challenging because user activities lack a ground-truth summary, and their quality is subjective and difficult to define. Existing techniques share a common shortfall: they offer no guarantee that generated summaries will support downstream personalization tasks—a critical function. Each approach also has unique drawbacks. Heuristic methods that extract subsets of activities fail to capture the breadth of user preferences and often produce less readable results. While prompt engineering is popular, pretrained models are not tailored to user data, and crafting effective prompts is both time-consuming and unscalable. Supervised fine-tuning is impractical due to nonexistent training datasets and the privacy concerns associated with collecting such data. Finally, RLHF or RLAIF methods rely on human or AI evaluators, but their judgments remain subjective without standardized criteria.

To overcome the challenges of generating natural language user summaries, we propose RLPF: Reinforcement Learning from Prediction Feedback (illustrated in Figure 1), which includes three components:

Summarization Model: A model is trained to generate succinct user summaries from raw activity data. Prediction-based Reward Model: To compute a reward, we measure the effectiveness of the generated summaries in downstream prediction tasks. Feedback Loop: The reward is then used to update the summarization model with RL, with an additional reward to encourage shorter lengths. This feedback loop guides the summarization model to continuously refine its ability to produce summaries that are not only concise but also highly effective for their intended applications.

RLPF offers a win-win solution: it enables the creation of high-quality user summaries without the need for resourceintensive and potentially privacy-compromising human intervention. By directly optimizing the summarization process for downstream prediction performance, we ensure that the generated summaries are both compact and directly relevant to the tasks they are meant to support. Furthermore,

Prediction Feedback What is the user’s favorite book category? N+1th book: Becoming What is the user’s A/B/C/D User’s Task Transfer next rating on   
User’s past RL reviewed “Barbie”?   
rbeovoiekswed Model Output Prediction Prompt books Summary Summary ) SUMMARY: The user is interested in Based on SUMMARY,which book will 米 Which CD will the   
T actCivDi,t Teosy(,on Dataset Transfer WhicuhseTrobyuwyi?ll the D. A Brief History of Time occupancy? Training Evaluation

compared to prevailing Reinforcement Learning (RL) approaches relying on feedback from a dedicated trained reward LLM (Ouyang et al. 2022; Bai et al. 2022; Lee et al. 2024; Yang et al. 2023), RLPF eliminates the overhead of training a separate reward model.

Through extensive experiments on four public datasets grounded in real-world user interactions - MovieLens 2015 and 2003 (Harper and Konstan 2015), Amazon Review (He and McAuley 2016), and Google Local Review (Yan et al. 2022), we demonstrate that RLPF summaries outperform baselines in terms of predictive power on both seen and unseen tasks, as well as on intrinsic quality evaluations.

Our contributions are four-fold:

We introduce the novel task of generating natural language user summaries for user modeling and personalization systems. This offers an interpretable alternative to traditional embedding-based representations and allows utilization by arbitrary LLMs without further training.   
We introduce RLPF, a novel and easy-to-implement method for training user summarizers. RLPF eliminates the need for reference summaries or hand-crafted prompts, while safeguarding user privacy.   
We demonstrate that RLPF summaries outperform baselines on both the training task and unseen tasks across four datasets and domains.   
• We evaluate RLPF summaries intrinsically and find significant improvements in factuality, abstractiveness, and readability.

# 2 Methodology

# Problem Statement

Consider a set of users $\mathcal { U } = \{ u _ { i } \} _ { i = 1 } ^ { \mathcal { M } }$ , where each user $i$ has an associated chronologically ordered sequence of interactions, denoted as $\{ v _ { i } ^ { 1 } , v _ { i } ^ { 2 } , . . . , v _ { i } ^ { N } \}$ . Each $v _ { i } ^ { j }$ within this sequence (where $1 \le j \le N _ { \cdot }$ ) comprises one or more textual features that describe a specific item, such as the titles or ratings of movies watched by the user. For each user $i$ , we concatenate all of their interactions $\{ v _ { i } ^ { j } \} _ { j = 1 } ^ { N }$ into a single string to form the user context $u _ { i }$ .

A summarizer model $\pi _ { \boldsymbol { \theta } }$ takes as input the user context and generates a summary $s _ { i } ~ = ~ \pi _ { \theta } ( u _ { i } )$ . The summary is then provided to off-the-shelf LLM to produce a prediction $\hat { y } _ { i } = \mathbf { \bar { \mathcal { P } } } ( s _ { i } )$ for a specific downstream task. We optimize $\pi _ { \boldsymbol { \theta } }$ to generate summaries $\{ s _ { i } \} _ { i = 1 } ^ { \mathcal { M } }$ that minimize the expected error between the predictions $\bar { \{ y _ { i } \} } _ { i = 1 } ^ { \mathcal { M } }$ and the ground truth task labels $\{ y _ { i } \} _ { i = 1 } ^ { \mathcal { M } }$ .

# Reinforcement Learning from Prediction Feedback

In the context of RL, we formulate summary generation as a Contextual Markov Decision Process (CMDP). In this framework, the state encompasses both the input text and the partially generated summary, while actions correspond to the selection of tokens from the entire vocabulary. At each step, the policy model maps these states to probability distributions over the vocabulary, facilitating autoregressive token selection. This selection process is guided by the current context and the overarching objective of maximizing cumulative rewards.

Within this RL framework, we formalize RLPF in the context of user summarization as follows:

• State: The set of user contexts $\mathcal { U } = \{ u _ { i } \} _ { i = 1 } ^ { \mathcal { M } }$ , where each $u _ { i }$ is a single string representing the textual features of a user’s $N$ past activities.   
• Action: The set of user summaries ${ \cal { S } } = \{ s _ { i } \} _ { i = 1 } ^ { \mathcal { M } }$ generated based on the corresponding user contexts.   
• Policy Model: The summarizer model, denoted by $\pi _ { \boldsymbol { \theta } }$ , which maps user contexts (states) to user summaries (actions): $\pi ( \bar { u } _ { i } ; \theta ) \to s _ { i }$ .   
• Reward: We leverage a frozen, pre-trained LLM to generate predictions $\mathcal { P } ( s _ { i } )$ for one or more specified tasks based on user summaries $s _ { i }$ . Then a scalar reward value is computed by comparing the prediction $\mathcal { P } ( s _ { i } )$ with its corresponding ground truth label $y _ { i }$ of the specific task.

The objective of RLPF is to learn a policy $\pi ^ { * }$ that maximizes the expected cumulative reward:

$$
\pi ^ { * } = \arg \operatorname* { m a x } _ { \pi } \mathbb { E } _ { u _ { i } \sim \mathcal { U } } [ r ( \pi ( u _ { i } ; \theta ) ) ]
$$

Reward Computation RLPF provides the flexibility to leverage any task for reward derivation, tailored to specific downstream application requirements. Moreover, it seamlessly accommodates the combination of rewards from multiple tasks if needed. Our implementation leveraged future activity prediction as the sole task for generating reward signals. This approach demonstrated strong generalization and transferability to unseen tasks, as detailed in the Results section. This underscores the convenience and efficiency of RLPF by eliminating the need for extensive, complex model training and overhead. Further results using alternative reward tasks, along with guidelines for task selection, are provided in the Appendix F.

For each user $i$ , summary reward $r ( s _ { i } )$ is as follows:

$$
r ( s _ { i } ) = r ^ { p r e d } ( s _ { i } , y _ { i } ) + w \cdot r ^ { l e n } ( s _ { i } )
$$

where $r ^ { p r e d } ( . )$ is the prediction feedback reward, $r ^ { l e n } ( . )$ is the length reward, and $w$ is a weight that controls the balance between the two terms.

Prediction Feedback Reward: Recall that each user context $u _ { i }$ consists of the textual features of $N$ past user activities. We employ the subsequent $( N + 1 )$ -th activity (e.g., watched movie title etc.) as the ground truth label $y _ { i }$ for predicting the future activity. Given the user summary $s _ { i }$ , we calculate a binary reward by comparing the LLM’s prediction based on $s _ { i }$ to the actual future activity $v _ { i } ^ { N + 1 }$ :

$$
r ^ { p r e d } ( s _ { i } , y _ { i } ) = \mathbb { 1 } ( \mathcal { P } ( s _ { i } ) = y _ { i } ) , w h e r e y _ { i } = v _ { i } ^ { N + 1 }
$$

However, since the reward model operates in a zero-shot setting, predicting item names with exact matches without any additional context is challenging due to the vast number of possibilities. This hinders the policy model’s ability to receive positive feedback and learn effectively. To tackle this issue, we adopt a multiple-choice approach, providing four answer choices for each summary based prediction, including the ground truth. The reward model is then prompted to select the correct option from the given choices. Notably, our method is adaptable to any closed-ended question formats. See Appendix $\mathrm { \bf K }$ for full prompts.

Length Reward: Furthermore, to promote concise summary generation, we incorporate a length reward:

$$
r ^ { l e n } ( s _ { i } ) = \operatorname* { m i n } [ \mathcal { C } , \beta * ( \mathcal { L } - l _ { i } ) ]
$$

where $l _ { i }$ represents the token length of summary $s _ { i }$ , and the hyperparameters $\mathcal { M } , \beta$ , and $\mathcal { L }$ denote the upper bound, magnitude, and target length of the summary, respectively. We set the target length to the average length of Zero Shot summaries in our experiments. See variable values in Appendix D.

Training Process The absence of reference summaries prevents the application of supervised fine-tuning to either the policy or reward model. Unlike the standard RLHF pipeline, which sequentially involves supervised fine-tuning, reward modeling, and policy optimization, RLPF directly optimizes the policy in a single RL training step. By leveraging LLMs’ inherent zero-shot summarization and prediction capabilities, RLPF eliminates the need for intricate prompt engineering, generating feedback for the RL process based on predicted future activities. While RLPF is not tied to any specific RL algorithm, we utilize REINFORCE (Williams 1992) with a baseline to update the policy model given that it is simpler yet still effective for our tasks. Both policy and value models are initialized from a frozen model.

To preserve the LLM’s original summarization capability and mitigate reward hacking, we introduce a KL divergence term between the current policy $\pi _ { \boldsymbol { \theta } }$ and the initial policy $\pi _ { i n i t }$ . Consequently, the policy parameters are updated according to the following rule:

$$
\theta \gets \theta + [ ( 1 - \alpha ) \nabla _ { \theta } \mathbb { E } [ r _ { i } ] - \alpha \mathbb { E } [ \nabla _ { \theta } K L ( \pi _ { \theta } | | \pi _ { i n i t } ) ] ]
$$

where $\alpha$ is a hyperparameter controlling the balance between the reward maximization and policy regularization.

# 3 Experimental Details

# Dataset

We conduct experiments on four public datasets grounded in real-world user interactions, encompassing product reviews, movie watching behavior, and location data. We perform training on Amazon Books (He and McAuley 2016), Google Local Review (Yan et al. 2022), MovieLens 2015(Harper and Konstan 2015). Additionally, we utilized another four Amazon Review datasets with different product categories, as well as MovieLens 2003, which features distinct users and movie catalogs compared to MovieLens 2015). See Appendix C for dataset details.

Data Generation For each user’s interaction data, presented as a chronologically ordered list of activities $u _ { i } \in \mathcal { U }$ , we randomly select one item as the target for future activity prediction, denoted as $y _ { i }$ . We utilize the $N$ activities preceding this target as the past activities $\{ v _ { i } ^ { j } \} _ { j = 1 } ^ { N } . v _ { i } ^ { j }$ represents an item name and rating pair, where item name correspond to movie title for MovieLens, product name for Amazon Review, and place name $^ +$ city name for Google Local Review, respectively. As previously mentioned, we concatenate $\{ v _ { i } ^ { j } \} _ { j = 1 } ^ { N }$ to construct the user context $u _ { i }$ . To prevent label leakage, the last item in each user’s data is reserved as the target item in the test set. Unless otherwise specified, we set $N = 5 0$ in our experiments.

# Evaluation Metrics

Extrinsic Utility We gauge the predictiveness of the summaries based on their prediction performance in various downstream tasks. Extending beyond Future Activity Prediction which is used as feedback during training, we incorporated additional tasks of various types to gauge the transferability and generalization capabilities of the generated summaries. These included 19 tasks include user interest reasoning, history activity retrieval, rating prediction, user demographic prediction and open text review generation. Please refer to Appendix I for detailed task definitions as well as their abbreviation used in the paper.

A frozen instruction tuned Gemini 1.0 Pro model was employed to generate predictions for all downstream tasks. Each summary $s _ { i }$ was fed into the model, and the resulting predictions were evaluated against ground truth labels.

![](images/df3410a301aea0536b0602f921d8c6a2156c86b80a1f424b817f1791efe9c871.jpg)  
Figure 2: RLPF summaries consistently demonstrate superior performance in Future Activity Prediction, surpassing both other summarization techniques and the full user context (”All Activities”), while significantly reducing the required context length. ZS-nano2: Gemini Nano-2 Zero-Shot; ZS-CP: Gemini Nano-2 with Crafted Prompts; ZS-Pro: Gemini Pro Zero-Shot.

Intrinsic Quality To further assess the intrinsic quality of the generated summaries, we utilize automated evaluation to compare summaries before and after training. This assessment focuses on aspects not explicitly covered by downstream task performance, including Factuality, Abstractiveness, Readability and Overall quality. For each criterion and overall quality, the Auto Rater compares a pair of summaries, with their relative positions randomly assigned to eliminate potential bias in the evaluation. We harnessed the most powerful model in the Gemini family, Gemini 1.5 Pro (GeminiTeam et al. 2024), as the Auto Rater. See Appendix A for the full prompt.

In addition to using Auto Rater, the Appendix G provides further results and discussions on employing grounded evaluation metrics to assess factuality and readability.

# Training Details

The summarizer model, or policy model $\pi _ { \boldsymbol { \theta } }$ , is initialized from Gemini 1.0 Nano-2(instruction tuned) and fine-tuned using RLPF. During training, reward computation $( \mathcal { P } ( s _ { i } ) )$ is performed by a frozen, instruction-tuned Gemini 1.0 Pro model, which predicts future activity based on the generated summary $s _ { i }$ . Gemini 1.0 Pro was selected for its optimal balance of performance and inference efficiency. In additiona, we also employed PaLM-2 XS (Anil et al. 2023) to showcase RLPF’s applicability across diverse policy models.

For each of the three training datasets (Amazon Books, Google Local Review, and MovieLens 2015), we trained the policy model with a batch size of 64 for 15,000 steps, and evaluation was performed on the final checkpoint. More hyper-parameter values are listed in Appendix D.

# Baselines

We compare the performance of user summary generated by RLPF against two categories of baselines: summary-based and activity-based. As the evaluator model makes zero-shot predictions for all inputs, any performance differences are attributed to the informativeness of the input, assuming consistent prediction capability.

Summary-Based Baselines: We employ frozen instruction tuned or fine-tuned models to generate summaries and assess their downstream performance.

– Gemini 1.0 Nano-2 Zero-Shot: Uses summaries generated by Gemini 1.0 Nano-2 in a zero-shot manner. This represents the anchor model before training.   
– Gemini 1.0 Pro Zero-Shot: Uses summaries generated by Gemini $1 . 0 \mathrm { P r o }$ in a zero-shot manner, a larger and more powerful model than the anchor model.   
– Gemini 1.0 Nano-2 Few-Shot: Uses summaries generated by Gemini 1.0 Nano-2 in a few-shot manner. We provided two examplars in context, where the example summaries are generated by Gemini 1.5 Pro. See full prompts in Appendix K.   
– Gemini 1.0 Nano-2 with Crafted Prompt: Uses summaries from Gemini 1.0 Nano-2, but with customdesigned prompts optimized for downstream tasks. We show the prompt in Appendix K.   
– RLAIF: User summaries trained with Direct RLAIF (Lee et al. 2024), using Gemini 1.0 Nano-2 as the policy model. The reward score is provided by an LLM (Gemini $1 . 0 \ \mathrm { P r o } \rangle$ . Further details on the prompting technique are available in the Appendix K.   
Activity-Based Baselines: The user context $u _ { i }$ is directly   
fed as input to a frozen instruction tuned model (Gemini   
$1 . 0 \mathrm { P r o } )$ to generate predictions:   
– First $X$ Activities: Uses only the earliest $\mathrm { \Delta X }$ activities $X \ < \ N )$ for downstream task predictions, ensuring comparable token length to RLPF summaries.   
– Random $X$ Activities: Similar to the above, but selects $X$ activities randomly.   
– Last $X$ Activities: Uses the most recent $X$ activities.   
– All Activities: Uses the full user context $N$ activities.

# 4 Results

# Target Task Performance

Figure 2 compares RLPF performance on the Future Activity Prediction task. Across all three datasets, RLPF demonstrates superior or comparable performance to various summarizers, including crafted prompting, a larger summarizer model, and RLAIF. Overall, RLPF outperforms Nano-2 zero-shot summaries by $+ 1 3 . 4 \%$ improvement, and outperforms RLAIF by $+ 2 2 \%$ on average. Compared to utilizing the full user context (all activities), RLPF achieves an average context length compression of $- 7 3 . 8 \%$ while still exhibiting a $+ 1 2 . 4 \%$ performance gain. Further comparisons with other baselines are provided in the Appendix H, underscoring exceptional capability of RLPF summaries to capture both short-term and long-term user context information.

Table 1: RLPF, trained exclusively on future activity prediction, exhibits remarkable transferability and generalization across diverse unseen tasks and datasets. Evaluation metrics: recall $\textcircled { a } 3$ for Favorite Genre/Category, Common City, and User Occupancy; ROUGE-Lsum for Review Gen; and accuracy for the remaining tasks.   

<html><body><table><tr><td colspan="2">Training Dataset</td><td>Evaluation Dataset</td><td>Evaluation Task</td><td>0-Shot</td><td>RLAIF</td><td>RLPF</td><td>vs 0-Shot</td><td>vs RLAIF</td></tr><tr><td rowspan="8">Task Transfer</td><td>MovieLens 2015</td><td>MovieLens 2015</td><td>Fav Genre</td><td>0.774</td><td>0.776</td><td>0.818</td><td>5.68%</td><td>5.48%</td></tr><tr><td>MovieLens 2015</td><td>MovieLens 2015</td><td>Rating</td><td>0.225</td><td>0.229</td><td>0.232</td><td>3.11%</td><td>1.31%</td></tr><tr><td>Amazon Books</td><td>Amazon Books</td><td>Fav Category</td><td>0.594</td><td>0.613</td><td>0.605</td><td>1.85%</td><td>-1.27%</td></tr><tr><td>Amazon Books</td><td>Amazon Books</td><td>Rating</td><td>0.244</td><td>0.147</td><td>0.255</td><td>4.51%</td><td>73.47%</td></tr><tr><td>Amazon Books</td><td>Amazon Books</td><td>Review Gen</td><td>13.52</td><td>13.68</td><td>13.46</td><td>-0.41%</td><td>-1.58%</td></tr><tr><td>Google Local</td><td>Google Local</td><td>Fav Category</td><td>0.487</td><td>0.513</td><td>0.559</td><td>14.78%</td><td>8.90%</td></tr><tr><td>Google Local</td><td>Google Local</td><td>Rating</td><td>0.118</td><td>0.118</td><td>0.111</td><td>-5.93%</td><td>-5.93%</td></tr><tr><td>Google Local</td><td>Google Local</td><td>Common City</td><td>0.765</td><td>0.791</td><td>0.901</td><td>17.73%</td><td>13.93%</td></tr><tr><td rowspan="6">Dataset Transfer</td><td>MovieLens 2015</td><td>MovieLens 2003</td><td>Future Act</td><td>0.468</td><td>0.447</td><td>0.509</td><td>8.82%</td><td>13.93%</td></tr><tr><td>MovieLens 2015</td><td>Amazon Movies</td><td>Future Act</td><td>0.572</td><td>0.579</td><td>0.606</td><td>5.94%</td><td>4.66%</td></tr><tr><td>Amazon Books</td><td>Amazon Movies</td><td>Future Act</td><td>0.645</td><td>0.573</td><td>0.663</td><td>2.73%</td><td>15.68%</td></tr><tr><td>Amazon Books</td><td>Amazon CDs</td><td>Future Act</td><td>0.397</td><td>0.447</td><td>0.573</td><td>44.33%</td><td>28.22%</td></tr><tr><td>Amazon Books</td><td>Amazon Toys</td><td>Future Act</td><td>0.620</td><td>0.585</td><td>0.644</td><td>3.94%</td><td>10.14%</td></tr><tr><td>Amazon Books</td><td>Amazon Games</td><td>Future Act</td><td>0.688</td><td>0.631</td><td>0.713</td><td>3.60%</td><td>12.90%</td></tr><tr><td rowspan="5">Task& Dataset Transfer</td><td>MovieLens 2015</td><td>MovieLens 2003</td><td>Fav Genre</td><td>0.808</td><td>0.801</td><td>0.843</td><td>4.35%</td><td>5.26%</td></tr><tr><td>MovieLens 2015</td><td>MovieLens 2003</td><td>User Age</td><td>0.274</td><td>0.341</td><td>0.246</td><td>-10.22%</td><td>-27.86%</td></tr><tr><td>MovieLens 2015</td><td>MovieLens 2003</td><td>User Gender</td><td>0.723</td><td>0.738</td><td>0.729</td><td>0.90%</td><td>-1.15%</td></tr><tr><td>MovieLens 2015</td><td>MovieLens 2003</td><td>User Occupancy</td><td>0.146</td><td>0.130</td><td>0.162</td><td>11.20%</td><td>24.89%</td></tr><tr><td>MovieLens 2015</td><td>MovieLens 2003</td><td>Rating</td><td>0.228</td><td>0.224</td><td>0.245</td><td>7.50%</td><td>9.38%</td></tr></table></body></html>

For comparison, we conducted supervised fine-tuning of a Gemini 1.0 Pro model on the same task, reaching $94 \%$ accuracy. However, this fine-tuned model exhibited zero performance on other tasks, highlighting its overfitting to the specific training task. Conversely, RLPF showcased remarkable transferability and generalization capabilities, as demonstrated in the subsequent section.

# Transferability and Generalization

To evaluate the generalizability and adaptability of RLPF for various personalization agent systems, we conducted a comprehensive transferability assessment across a diverse set of unseen tasks and datasets. As shown in Table 1, RLPF summaries consistently exhibited superior transferability compared to zero-shot and RLAIF baselines, demonstrating improvements in 16 and 14 out of 19 total evaluation cases, respectively. These results highlight RLPF’s exceptional transferability and its potential to be effectively applied to a wide range of personalization scenarios, particularly when training data is scarce.

Task Transfer RLPF summaries demonstrated a slight improvement on an unseen retrieval task, common city retrieval on Google Local Review, and performed on par with zero-shot summary on an unseen personalized text generation task, review generation on Amazon Books.

Dataset and Domain Transfer We also evaluated whether an RLPF trained model can generalize to an unseen dataset, either in same domain or a different domain. We used the policy model trained with MovieLens 2015 to generate summaries on MovieLens 2003 and Amazon Movies&TVs dataset and evaluated future movie prediction with the generated summaries. From the results, RLPF model trained on MovieLens 2015, showed improvements on both unseen datasets. Furthermore, the model trained on Amazon Books achieved significant performance gains on Amazon CDs&Vinyl data, highlighting its strong domain adaptation abilities.

Task and Dataset Transfer Furthermore, we evaluated RLPF model performance on unseen tasks from unseen datasets. RLPF model trained with MovieLens 2015 with future activity prediction showed improvement on MovieLens 2003 dataset in favorite genre prediction and user demographic reasoning.

# Intrinsic Evaluation

Table 2 demonstrates that RLPF summaries consistently outperform zero-shot summaries on all three datasets, as evaluated by the automated rater across all criteria: Factuality, Abstractiveness, Readability, and Overall evaluation.

This finding is noteworthy given that RLPF was trained solely on reward signals from future activity prediction. Despite this focused training, RLPF summaries not only avoid degradation or overfitting to a single goal but also exhibit significant improvements in other crucial aspects. This suggests that when employing RLPF for user summarization, designing explicit reward signals for each criterion, which can be challenging to obtain, may not be necessary. Instead, future activity prediction performance appears to provide correlated and implicit signals for these criteria. Intuitively, to make accurate future activity predictions, a summary needs to be factually consistent and distill key user information. While readability might not be a strict prerequisite for future activity prediction, it’s noteworthy that this criterion also correlates with this downstream task.

Table 2: Intrinsic Evaluation with Auto Rater.   

<html><body><table><tr><td rowspan="2">Dataset</td><td rowspan="2">Criteria</td><td colspan="2">RLPFWinRate</td></tr><tr><td>vs Zero-Shot</td><td>vs RLAIF</td></tr><tr><td rowspan="5">MovieLens 2015</td><td>Factuality</td><td>61.32%</td><td>62.53%</td></tr><tr><td>Abstractiveness</td><td>62.54%</td><td>56.09%</td></tr><tr><td>Readability</td><td>62.42%</td><td>56.36%</td></tr><tr><td>Overall</td><td>62.47%</td><td>56.10%</td></tr><tr><td>Factuality</td><td>72.93%</td><td>40.09%</td></tr><tr><td rowspan="4">Amazon Books</td><td>Abstractiveness</td><td>70.14%</td><td>39.20%</td></tr><tr><td>Readability</td><td>71.28%</td><td>35.47%</td></tr><tr><td>Overall</td><td>70.08%</td><td>39.17%</td></tr><tr><td></td><td></td><td></td></tr><tr><td rowspan="3">Google Local</td><td>Factuality</td><td>77.58%</td><td>49.97%</td></tr><tr><td>Abstractiveness</td><td>84.59%</td><td>54.56%</td></tr><tr><td>Readability</td><td>83.73%</td><td>46.02%</td></tr><tr><td>Review</td><td>Overall</td><td>84.46%</td><td>54.22%</td></tr></table></body></html>

Interestingly, RLPF’s performance on par with RLAIF in this evaluation, even though RLAIF was specifically trained with reward signals more aligned with the intrinsic evaluation criteria, highlights the effectiveness of RLPF.

# Analysis

Alternative Policy Model Additionally, we applied RLPF to a policy model initialized from the PaLM-2 XS model, with results presented in Table 3. Mirroring the observations with Gemini 1.0 Nano-2, RLPF summaries based on PaLM-2 XS also exhibited improvements in both the training task (future activity prediction) and the unseen task (favorite genre/category prediction) across all three datasets. A slight drop in performance was noted for favorite genre prediction on the MovieLens 2015 dataset.

Robustness to Model that Uses Summaries To further ensure that RLPF summaries are not overly tailored to the specific reward model used during training, we employed an additional evaluator model PaLM-2 S to assess their performance. As in previous experiments, RLPF summaries were trained using reward signals derived from Gemini 1.0 Pro. Table 4 demonstrates that the improvements achieved with RLPF summaries transfer effectively to these different evaluator models, highlighting the generalizability of RLPF summaries across various LLM-powered systems.

Impact of Summary Length Figure 3 illustrates our experiments on MovieLens 2015, where we varied the target length $( \mathcal { L } )$ in the length reward term. Generally, longer summaries led to improved task performance but decreased scores in automated evaluation metrics, suggesting a tradeoff between extrinsic utility and intrinsic qualities.

Table 3: RLPF with PaLM-2 XS as the policy model.   

<html><body><table><tr><td rowspan="2">Dataset</td><td rowspan="2">Task</td><td colspan="2">PaLM-2 XS</td></tr><tr><td>zero-shot</td><td>RLPF</td></tr><tr><td rowspan="2">MovieLens 2015</td><td>Future Act</td><td>0.638</td><td>0.741</td></tr><tr><td>Fav Category</td><td>0.860</td><td>0.849</td></tr><tr><td>Amazon</td><td>Future Act</td><td>0.626</td><td>0.675</td></tr><tr><td>Books</td><td>Fav Category</td><td>0.557</td><td>0.565</td></tr><tr><td rowspan="2">Google Local Review</td><td>Future Act</td><td>0.502</td><td>0.532</td></tr><tr><td>Fav Category</td><td>0.454</td><td>0.477</td></tr></table></body></html>

Table 4: Evaluated using PaLM-2 S, with reward signals derived from Gemini 1.0 Pro during training.   

<html><body><table><tr><td>Dataset</td><td>Task</td><td>zero-shot</td><td>RLPF</td></tr><tr><td>MovieLens</td><td>Future Act</td><td>0.578</td><td>0.674</td></tr><tr><td>2015</td><td>Fav Category</td><td>0.822</td><td>0.840</td></tr><tr><td>Amazon</td><td>Future Act</td><td>0.689</td><td>0.734</td></tr><tr><td>Books</td><td>Fav Category</td><td>0.543</td><td>0.567</td></tr></table></body></html>

![](images/d208d6d28936e7275424abdfc51d4384fe0b421cf5737a39007a3529263b0616.jpg)  
Figure 3: Impact of Different Target Lengths on MovieLens 2015. Percentage changes are calculated relative to “No Length Reward” condition (no maximum length constraint). Data on the right axis pertains to AutoEval, while the left axis corresponds to the remaining tasks.

Robustness to Prompts We investigated the impact of varying prompts for summary generation and prediction during reward computation. As illustrated in Figure 4, task returns converge to a similar level despite initial differences in zero-shot performance, demonstrating the robustness of RLPF to diverse prompts. See full prompts in Appendix K.

Qualitative Observation In general, zero-shot summaries tend to mimic the structure of the input, which may either be directly copied from the input activities or represent hallucinations (e.g., mentioning popular movies like “The Godfather” despite their absence in the user history). After RLPF training, summaries become more coherent and distill user information effectively, though some repetition or hallucination may still occur occasionally. This also explains the better Factuality and Abstractiveness scores from the Automated Evaluation. Although, we noticed both RLPF and RLAIF summaries sometimes exhibit repetitive patterns (e.g., ”I am always up for a good recommendation”), while the core content remains user-specific. See the Appendix J for example summaries.

![](images/838925eefe962a56108b89c3577c76b202a10b0f7c3cdda1693befa7d06a52fb.jpg)  
Figure 4: RLPF is robust with various prompts. Top: Evaluation metric with different prompts for Summarization, Bottom: Evaluation metric with different prompts for Prediction during reward computation. Prediction task: Future activity prediction on MovieLens 2015.

# 5 Discussion

Responsible Deployment While RLPF shows promise for enhancing personalization, its use of user data raises privacy and data leakage concerns. Offline training of user summaries and employing a frozen LLM for online serving can mitigate some risks. However, a thorough analysis of potential vulnerabilities is crucial before real-world deployment.

# 6 Related Work

# Text Summarization

Leveraging language models for summarizing long documents has gained prominence. Unlike text summarization, which condenses texts while retaining key information, our approach focuses on distilling implicit user insights and preferences beyond merely extracting user history.

User summarization poses distinct challenges in model training and evaluation due to the absence of ground truth user summaries. In text summarization, widely-used datasets with reference summaries (Hermann et al. 2015; Narayan, Cohen, and Lapata 2018; Napoles, Gormley, and Van Durme 2012) enable supervised fine-tuning (Cohan et al. 2018; He et al. 2022, 2023; Kry´scin´ski et al. 2021; Roit et al. 2023) or RL with reward signals comparing generated and reference summaries (Gunasekara et al. 2021), as well as evaluation metrics with lexical matching (Lin 2004) or embedding similarity (Zhang et al. 2020). These methods and metrics are inapplicable to user summarization due to the lack of datasets. Human evaluation has been used in text summarization (Goyal, Li, and Durrett 2023), but privacy concerns make it impractical for user summarization.

Previous summarization work without reference summaries aligns more with ours. These methods often leverage question-answering (QA) (Durmus, He, and Diab 2020; Fabbri et al. 2022; Deutsch, Bedrax-Weiss, and Roth 2021; Fabbri et al. 2021) or pre-trained models (Kryscinski et al. 2020; Goyal and Durrett 2020), relying on the capabilities of QA generation or entailment models. However, no datasets exist for training these models on user activity data. Our work also employs QA and pre-trained LLMs for reward computation, but takes a practical approach by grounding reward signals in real-world personalization questions with answers derived directly from user data, avoiding the need to train additional QA models.

# User Modeling

User modeling has benefited significantly from LLM advancements. While existing methods often represent user activity with embeddings (Ning et al. 2024; Doddapaneni et al. 2024), our work generates natural language-based user summaries, a more human-readable and reusable alternative.

Previous work on natural language-based user modeling has primarily relied on prompting or fine-tuning for specific downstream tasks (Bao et al. 2023; Wu et al. 2024b; Liu et al. 2023; Lyu et al. 2024; Li et al. 2023; Salemi et al. 2024; Wang et al. 2024) or pre-defined user attributes (Rao, Leung, and Miao 2023; Ji et al. 2023; Wu et al. 2024a). In contrast, our approach introduces a novel end-to-end training framework for generating user summaries. This method focuses on comprehensive user profiling to support a wide range of downstream tasks, rather than focusing on a single user attribute or characteristic.

# Reinforcement Learning from AI Feedback

RL from Human Feedback (RLHF) (Ouyang et al. 2022) aligns language models with human values but relies heavily on high-quality human labels. To mitigate this dependency, RL from AI Feedback (RLAIF)(Bai et al. 2022; Yang et al. 2023) utilizes off-the-shelf LLMs to replace human annotations, achieving superior performance on tasks like summarization (Lee et al. 2024). RLAIF scores summaries directly using an LLM, which introduces subjectivity due to the lack of standardized criteria. In contrast, our approach RLPF uses downstream task performance as the reward signal, enabling direct optimization for improved personalization.

# 7 Conclusions

We introduced RLPF, a novel method to generate humanreadable user summaries from raw activity data. RLPF leverages readily available LLMs and downstream task performance as reward signals, overcoming challenges in traditional summarization approaches. Our experiments demonstrate superior performance, context compression, and generalizability across unseen tasks. Future work will extend RLPF to more complex scenarios, additional feedback mechanisms, and broader applications.