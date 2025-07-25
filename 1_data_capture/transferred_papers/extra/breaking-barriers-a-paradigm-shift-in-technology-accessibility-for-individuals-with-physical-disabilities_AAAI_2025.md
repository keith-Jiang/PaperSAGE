# Breaking Barriers: A Paradigm Shift in Technology Accessibility for Individuals with Physical Disabilities

Kshitij Mishra \*1, Manisha Burja \*1, Asif Ekbal 2

1 Department of Computer Science and Engineering, Indian Institute of Technology Patna, India 2 School of AI and Data Science, Indian Institute of Technology Jodhpur, India mishra.kshitij07@gmail.com, manishaboorja $@$ gmail.com, asif.ekbal $@$ gmail.com

# Abstract

Individuals living with disabilities often face challenges in their daily lives, from managing physical tasks to coping with emotional needs. It is imperative to provide them with personalized, courteous, and empathetic support that can address their unique needs. To bridge this gap, we propose an Empathetic Disability Support System (EDiSS), designed to offer personalized support tailored with correct politeness and empathetic strategies as per individual users’ OCEAN traits, gender, and age. To train EDiSS, first, a specialized personalized disability support dialogue dataset (PDCARE) is created encompassing a wide spectrum of disabilities, such as Spinal Cord Injuries, Neurological Disorders, Orthopedic Disabilities, etc, and support areas like Physical Therapy Exercises, Pain Management, Emotional Support, etc. EDiSS employs a reinforcement learning-based dialogue model with a novel reward function. It adapts its tone and content based on the user’s persona, gender, and age to provide respectful and empathetic assistance across various aspects of daily living. Our experiments and evaluation demonstrate the effectiveness of EDiSS in improving the quality of life of individuals with disabilities, marking a significant advancement in leveraging technology to provide much-needed support and assistance in their daily challenges.

Code and Dataset — https://github.com/Mishrakshitij/EDiSS.git

# Introduction

In recent years, there has been a growing recognition of the challenges faced by individuals living with disabilities, encompassing not only physical limitations but also emotional and societal barriers. According to the World Health Organization (WHO), over 1 billion people worldwide experience some form of disability, making up approximately $1 5 \%$ of the global population (Organization 2021). Despite progress in accessibility and inclusion initiatives, individuals with disabilities continue to encounter significant hurdles in various aspects of their daily lives, ranging from physical tasks to social interactions and emotional well-being.

The United Nations Sustainable Development Goals (SDGs) underscore the importance of inclusivity and accessibility, particularly through Goal 10, which aims to reduce inequality within and among countries, and Goal 3, which promotes health and well-being for all (United Nations 2024b). Further, Goal 4 of SDGs emphasizes the importance of ensuring inclusive and equitable quality education and promoting lifelong learning opportunities for all, including persons with disabilities (United Nations 2024b). Additionally, the Leave No One Behind Principle (LNOB), a core tenet of the SDGs, emphasizes the need to ensure that development efforts reach and benefit all segments of society, including those with disabilities (United Nations 2024a).

To address the multifaceted needs of individuals with disabilities, there is a pressing need for innovative solutions that can provide personalized, courteous, and empathetic support. Conversational AI systems can offer a promising solution (Smith and Johnson 2020). These systems can significantly enhance communication, automate daily tasks, and improve environmental control for individuals facing physical challenges (Johnson and Davis 2019). Recently, there has been an attempt to harness conversational systems, for providing tailored support for individuals with physical disabilities (Brown and Smith 2021). However, existing systems often fall short of providing a comprehensive and user-centric solution. Additionally, there is a need for a system that can adapt to individual preferences and requirements, considering the diverse nature of physical disabilities. Therefore, addressing these challenges necessitates a fresh perspective.

In response to this imperative, we propose an Empathetic Disability Support System (EDiSS), to offer tailored assistance with correct politeness and empathetic strategies based on individual users’ persona, gender, and age. We start with the creation of a personalized disability support dialogue dataset (PDCARE), encompassing a diverse range of disabilities and support areas, such as Spinal Cord Injuries, Neurological Disorders, Orthopedic Disabilities, Physical Therapy Exercises, Pain Management, and Emotional Support. By drawing upon this comprehensive dataset, EDiSS employs a reinforcement learning-based dialogue model with a novel-designed reward function to adapt its tone and content to the unique needs and preferences of each user. Our evaluation demonstrates the efficacy of EDiSS in enhancing the quality of lives for individuals with disabilities. By providing respectful and empathetic assistance across various aspects of daily living, EDiSS represents a significant advancement in leveraging technology to address the challenges faced by individuals with disabilities. The key contributions can be summarized as follows:

1. Created a comprehensive physical disability support dialogue dataset PDCARE which encompasses five OCEAN personality traits (McCrae and Costa 1992) of the user and agent’s three politeness strategies and eight empathy strategies to lay the foundation for the development of more sophisticated and physical disability support systems in the future.   
2. Introduced EDiSS, an empathetic disability support system that places a strong emphasis on patient personality to foster politeness and empathy such that a cordial environment can be tailored to the unique needs of each individual.   
3. Design a novel reward function by leveraging three transformer-based classifiers to ensure user’s profile alignment, politeness, and empathy consistency.   
4. Demonstrate the potential of EDiSS through extensive automatic and human evaluation in improving physical disability hurdles, offering hope and support.

# Related Work

One area of research that has gained traction in recent years is the development of personalized conversational agents in healthcare (Kocaballi et al. 2019). The application spectrum includes post-stroke recovery (MIRANDA MACIEL 2023), behavior change (Zhang et al. 2020), and activities of daily life (Sheng et al. 2023). These studies show that by considering factors, such as gender, age, and personality traits, persona-based dialogue systems can offer more engaging responses.

Exploring personalized systems for physical disabilities, (Pereira and D´ıaz 2019) investigate the role of health chatbots in behavior change interventions, while (Huq, Maskeliu¯nas, and Damasˇevicˇius 2022) focus on conversational agents aiding individuals with cognitive disabilities, showcasing the diverse applications of personalized systems. (Cha et al. 2021) emphasize the empowering potential of voice-based agents for adolescents with Autism Spectrum Disorder, promoting inclusivity, and (Vigouroux et al. 2023) shift the focus to disability-friendly interfaces in home automation. (Wiratunga et al. 2020) introduce physical activity promotion in older adults through chatbots, and (Murali et al. 2023) contribute insights into automated pain assessment. (Smith and Dragone 2023) focus on daily living assessments, broadening the scope of personalized systems in healthcare. However, these systems often lack in providing comprehensive support for a spectrum of disabilities, as well as interactive and engaging experiences that cater to diverse user needs and contexts.

Integration of politeness and empathy into dialogue systems has witnessed significant advancements in recent years (Newbold et al. 2019; Mishra, Firdaus, and Ekbal 2022). (Rashkin et al. 2019) introduced a transformerbased approach for building empathetic dialogue systems, demonstrating improved performance in generating empathetic responses. (Liu et al. 2020) proposed a tag-andgenerate method for politeness transfer, enabling conversational agents to incorporate politeness markers into generated responses. (Zhao and Eskenazi 2017) explored the use of conditional variational auto-encoders (CVAEs) to generate polite and empathetic dialogue, conditioning the generation process on user attributes and context. (Wang et al. 2019) focused on cross-language voice cloning and multilingual speech synthesis, enabling conversational agents to communicate fluently and empathetically in multiple languages. Additionally, (Samad et al. 2022) developed a reinforcement learning-based empathetic persuasive dialogue system for charity donation tracing user emotions to tailor persuasion strategies. These studies underscore the significance of politeness and empathy in enhancing user experience and engagement in conversational agents, paving the way for more effective and user-centric dialogue systems.

Our research takes a distinct focus. We center our efforts on implementing a diverse array of politeness and empathy strategies tailored to suit the gender, age, and persona of a user. This tailored approach enhances interpersonal dynamics and communication effectiveness. Our system, EDiSS, explores a spectrum of physical disability issues through 6,796 dialogues involving diverse patient profiles. Guided by novel-designed rewards, EDiSS crafts responses tailored to individual user profiles while ensuring the application of appropriate politeness and empathy strategies. To the best of our knowledge, EDiSS is the first attempt to develop a support system explicitly designed for individuals with physical disabilities, incorporating precise politeness and empathy strategies. Our contribution enhances the quality of support provided to this demographic, while also stimulating continued research and innovation in this often overlooked domain.

# Dataset

We create PDCARE dataset consisting of physical disability support dialogues, aiming to provide personalized assistance to individuals facing physical challenges.

The PDCARE dataset tackles various issues associated with physical disabilities. It encompasses topics, such as Accessibility Information, Travel Tips, Advocacy and Rights, Financial and Insurance Guidance, Mobility Aids, Home Modifications, Physical Therapy Exercises, Assistive Technology, Pain Management, Activities of Daily Living (ADLs), Emotional Support, Employment and Education, Social Interaction, Fitness and Recreation, Peer Support Groups, Parenting with Disabilities, and Transitions and Life Changes. The dataset addresses specific physical disability supports including Mobility Impairments, Visual Impairments, Hearing Impairments, Speech Impairments, Neurological Disorders, Spinal Cord Injuries, Amputations, Orthopedic Disabilities, Cerebral Palsy, Muscular Dystrophy, Balance and Gait Disorders, Chronic Pain, and AgingRelated Disabilities. Using Llama3-70B (AI@Meta 2024) with rigorous data quality control measures, we ensure the PDCARE’s reliability and comprehensiveness. The details of each of the disabilities and respective supports can be found in Table 1 of the Appendix.

Table 1: Dataset statistics of PDCARE.   

<html><body><table><tr><td>Metrics</td><td>Train</td><td>Validation</td><td>Test</td></tr><tr><td>#ofDialogues</td><td>5436</td><td>681</td><td>679</td></tr><tr><td>#ofUtterances</td><td>124934</td><td>15521</td><td>15639</td></tr><tr><td>Avg.#Utterances/Dialogue</td><td>22.98</td><td>22.79</td><td>23.03</td></tr></table></body></html>

# Dataset Creation

To create PDCARE dataset, we start with the design of prompts, each tailored to specific topics and corresponding physical disability issues. These prompts consist of instructions about the topic, the physical disability, gender: male and female, age: younger, middle-aged, and older, persona: openness $( O )$ , conscientiousness $( C )$ , extraversion $( E )$ , agreeableness (A) and neuroticism $( N )$ of the user. Unique challenges faced by different genders and age groups allow for a nuanced dialogue, while persona information helps in crafting a response that resonates with the individual’s personality. To facilitate a natural and coherent dialogue between a doctor and a disabled user, seed utterances are used as starting points, combined with instructions to guide the conversation dynamically. The seed utterances consisting of 4-turns are obtained from GPT-3.5 (Ouyang et al. 2022) 1. Further, seed utterances are quality checked in terms of topic-consistency, context-adequacy, and fluency by eight human participants having post-graduation in Linguistics and expertise in corresponding tasks. Quality checks were performed on an integer Likert scale of 1-3. The seed utterances with scores of 1 and 2 were corrected if needed and 3 were taken as intact. A reliable inter-evaluator kappa agreement (McHugh 2012) score for each of the three metrics is found to be $8 5 . 8 \%$ , $8 6 . 2 \%$ , and $8 7 . 4 \%$ , respectively in this phase.

The prompt with correct/modified seed utterances is given to Llama3-70B (AI@Meta 2024) with instruction <generate 4 more turns in continuation of given dialogue to unfold user’s issues and provide support without closing the dialogue $>$ . Now, the generated 8-turn dialogue was again quality checked by human participants in terms of three metrics as given above and was corrected if needed. The generated 8-turn dialogues are also quality-checked in terms of all three quality check metrics: topic-consistency, contextadequacy, and fluency as above. The 8-turn interactions having scores of 1 and 2 are again corrected or modified by the same 8 participants. Two examples of all three errors and corresponding corrections are shown in Table 5 of the appendix. Corrections made include restructuring responses for clarity, providing relevant information, and improving grammatical accuracy. The inter-evaluator kappa agreement (McHugh 2012) score for each of the metrics was found to be $8 1 . 3 \%$ , $8 2 . 8 \%$ , and $8 4 . 1 \%$ , respectively, in this phase. Additionally, these dialogues are also checked for user-profile alignment. If found to be non-aligned they are corrected/- modified.

Now this 8-turn dialogue is used as a prompt to generate the dialogues as per context in the given prompt with instructions to $<$ Complete the dialogue with polite and empathetic support as per the user’s personality traits, gender, and age. The minimum and maximum number of turns allowed to complete the dialogue are 12 and 30, respectively. Engage the user as much as possible>. This iterative process, guided by Llama3-70B (AI $@$ Meta 2024), led to the generation of appropriate dialogues. Automated checks were then implemented to further enhance data quality. These checks included removing duplicate dialogues and conducting turn-level analysis to maintain smooth transitions within each dialogue. Prompts and examples of seed utterances are detailed in Figure 1 and Table 4 of the appendix, respectively. A sample 8-turn dialogue and complete dialogue generated are shown in Figure 2 and Figure 3 of the appendix. Additionally, various topics with associated physical disabilities, five personas, and their example utterances are detailed in Sections Topics and Associated Physical Disabilities, Persona, and Table 2 of the appendix, respectively.

# Data Quality Control

We applied data quality control measures to ensure highquality data. Following the same processes as mentioned in the earlier Section , the generated dialogues were quality checked in terms of topic-relevance, context-adequacy, and fluency. First, the same eight participants quality-checked the generated dialogues based on guidelines stated in Section Data Quality Control of the Appendix. As previously, the dialogues are rated on a Likert scale of 1 to 3. For each of the metrics, the inter-evaluator Kappa (McHugh 2012) agreement score of $7 9 . 4 \%$ , $8 0 . 2 \%$ , and $8 2 . 4 \%$ , respectively, were observed. Any dialogues rated 1 were discarded; dialogues with improper language, as well as those rated 2, were modified and corrected.

Following this internal assessment, to ensure the dataset’s alignment with best practices and real-world applicability, a subset comprising $5 \%$ of the diverse physical disability dialogues is sent to three medical experts specializing in physical therapy and disability management for a thorough review of dialogue-quality evaluation. As per feedback and guidelines provided by medical experts, the dialogues are again reviewed manually by all eight participants. Almost $7 \%$ of dialogues were modified again in this phase, and $3 \%$ of dialogues were discarded based on their irrelevance. The dual evaluation process involving both participants and medical experts guarantees that the PDCARE dataset is adequate, relevant, and linguistically fluent. The details of the data regarding quality checks are presented in Table 6 of the appendix.

# Dataset Annotation

To have the politeness and empathetic strategy information, we annotate the created dataset with three politeness and eight empathetic strategies for the Doctor’s responses at the utterance level. These carefully chosen strategies showcase a deeply emotional and cognitive understanding of their distinctive circumstances and create a welcoming space that nurtures disabled user’s self-esteem.

Inspired by (Brown and Levinson 1987), we consider three politeness strategies viz. positive politeness strategy, negative politeness strategy, and bald-on record. The eight empathetic strategies can be given as:

• Genuine Engagement: Demonstrates sincere interest in understanding the individual’s experiences and emotions, fostering a supportive environment (Rogers and Farson 1975).   
• Privacy Assurance: Assures individuals that their personal information will be kept confidential, creating a safe space for them to share their concerns and feelings (Zhang and Liu 2021).   
• Forward Focus Encouragement: Encourages individuals to focus on positive aspects and future goals despite physical challenges, fostering motivation and resilience (Ryan and Deci 2020).   
• Compassionate Validation: Validates the individual’s emotions and provides comfort and empathy, acknowledging the difficulties they face due to physical disabilities (Davies, Murphy, and Judd 2021).   
• Practical Assistance: Provides practical support and advice for managing physical disabilities, including referrals to experts or resources for additional assistance (Noll, Mah, and May 2018).   
• Continuous Support: Reassures individuals that they are not alone in their journey and emphasizes ongoing support and assistance available to them (Giroldi et al. 2019).   
• Strength-based Support: Empowers individuals by recognizing their strengths and capabilities in managing their physical disabilities, promoting self-confidence and autonomy in decision-making (Ryan and Deci 2020).   
• No Strategy: Assigned when responses do not employ any specific empathy strategy.

The same team of eight participants annotated both politeness and empathetic strategies. First, $50 \%$ of the dataset is manually annotated by the team, emphasizing the identification of politeness strategies and empathy strategies following the guidelines outlined in Section 1.4 of the appendix. Illustrative examples for each strategy were provided to ensure a shared understanding among annotators to manually annotate the required politeness and empathy strategy labels. The multi-rater Kappa agreement ratio (McHugh 2012) of $8 0 . 6 \%$ and $7 5 . 7 \%$ were observed for politeness and empathetic strategies, respectively. If the labels differ from the most voted labels for both types of strategies, annotators reannotate the labels to achieve alignment and consistency.

Subsequently, considering $50 \%$ annotated dataset, we fine-tune RoBERTa-large (Liu et al. 2019) model for building politeness-strategy and empathy-strategy classifiers 2. The un-annotated remaining $50 \%$ of the dataset is passed through these classifiers to obtain the politeness and empathy strategy labels. Manual verification is then performed by the same eight human participants who acted as annotators to cross-check the annotations. Finally, we obtain PDCARE - a persona-oriented disability support dialogue dataset. Dataset statistics are presented in the Table 1. Table 3 in the appendix provides example utterances illustrating different politeness and empathy strategies. Further, we provide comprehensive dataset details in Section Dataset Details of the Appendix.

# Methodology

Initially, we start the development of EDiSS by employing a warm-start, i.e. fine-tuning the Phi-3-small model (Abdin et al. 2024) using the LORA technique (Hu et al. 2021) on the PDCARE dataset. This dataset comprises a collection of $N$ dialogues between a physically disabled user and a system acting as a doctor. Each dialogue contains vital information about the user’s gender, age, and persona. The model takes input $x _ { i }$ , incorporating the context, user’s persona, age, and gender, represented as $x _ { i } = \left[ c _ { i } + p _ { i } + g _ { i } \right] + a _ { i } ]$ . Here, $c _ { i } = [ c _ { i - 1 } + u _ { i } ]$ , denote the context and user’s response at the $i ^ { t h }$ turn in the $d ^ { t h }$ dialogue. The output $y _ { i }$ corresponds to the system’s response (Li et al. 2023). Our objective is to predict $\hat { y _ { i } } \approx y _ { i }$ , i.e. we minimize the cross-entropy loss between the predicted $\hat { y _ { i } }$ and actual system responses $y _ { i }$ :

$$
\mathcal { L } _ { \mathrm { { C E } } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { M } y _ { i j } \log ( \hat { y } _ { i j } )
$$

Here, $M$ denotes the vocabulary size, and $\hat { y } _ { i j }$ represents the predicted probability of the $j$ -th token in the vocabulary for the $i$ -th dialogue. We call this trained system DSS and trained parameters $\theta$ .

# EDiSS

In the subsequent phase, we further refine the $D S S _ { \theta }$ within a reinforcement learning framework, employing the Proximal Policy Optimization (PPO) loss (Schulman et al. 2017). Here, we initialize the policy $\pi _ { \theta } ( a _ { t } | s _ { t } ) ~ = ~ D S S _ { \theta }$ as the probability distribution over actions $a _ { t }$ given the state $s _ { t }$ as per current policy parameters $\theta _ { t }$ . An action $a _ { t }$ is the probability of selection of a response token from the vocabulary $V$ . The state $s _ { t }$ at time step $t$ is explicitly defined as $s _ { t } = [ c _ { t } , m _ { t } ]$ , where $c _ { t }$ denotes the ongoing dialogue context, and $m _ { t }$ signifies the model’s memory (Schulman et al. 2017).

Rewards To guide the learning process effectively, we have devised five distinct novel rewards, which encompass both task-relevance and smoothness aspects of a support dialogue. These rewards ensure that the generated responses, denoted as $\hat { y }$ , exhibit naturalness and consistency with the user’s persona, gender, and age while also incorporating appropriate politeness and empathetic strategies.

Task-Relevance Reward: This reward is designed to encourage the model to generate responses that are relevant to the task or goal at hand.

$$
R _ { \mathrm { t a s k - r e l e v a n c e } } = { \frac { 1 } { 1 + \exp ( - \lambda \cdot ( w _ { 1 } \cdot \Delta _ { 1 } + w _ { 2 } \cdot \Delta _ { 2 } + w _ { 3 } \cdot \Delta _ { 3 } ) ) } }
$$

Ive been having a hard time geting around with my wheelchair.It's making me feel really isolated.   
5 C1: Maybe we can brainstorm some + accessibleplacesyoucouldvisitto enhanceyourwheelchairforbetter Gender:Male+Age: DSSO mobility. Youngprn Persona: C2: Lsan imagine how frusgtrting that REINFORCEMENT reachingouttoyourlocaldisability LEARNING services FRAMEWORK REWARDMODEL + C3:Iunderstandyourstruggle.Let's tacklewhat'schallengingaboutyour TASKRELEVANCEREWARD $\mathbf { \tau } =$ wheelchairtogether.Yourwell-being matters,andwe'llfindsolutionsto $\scriptstyle \mathbf { R } _ { 1 } :$ UserProfileAlignmentReward $^ +$ improveyourmobilityandconnection. $\scriptstyle \mathbf { R } _ { 2 } :$ PolitenessAdherenceReward $\kappa _ { 3 } \colon$ Empathy Consistency Reward ReaNoisgs PPO LOSS + C1 0.53 + EDtims isd   
SMOOTHNESS REWARD = △SYNTACTIC + using PPO △SEMANTIC C23 0.1 正 loss C3: I understand your struggle. Let's tackle what's challenging about your wheelchairtogether.Yourweli-beingmatters,andwe'llfindsolutionstoimprove ↓ yourmobilityandconnection. EDiss

In this equation, $\Delta _ { 1 } , \Delta _ { 2 }$ , and $\Delta _ { 3 }$ represent three different task-relevant measures, such as user-profile alignment, politeness-strategy correctness, and empathy-strategy correctness, respectively. $w _ { 1 } , w _ { 2 } , w _ { 3 }$ are the weights given to each of the rewards with $w _ { 1 } + w _ { 2 } + w _ { 3 } = 1 . ~ \lambda$ is a scaling factor that determines the sensitivity of the reward to changes in task relevance.

1. User-Profile Alignment Reward: This reward encourages the model to generate responses that are aligned with the user’s persona, gender, and age. It evaluates the model’s ability to understand and adapt to user characteristics:

$$
\Delta _ { 1 } = \mathrm { C L S } _ { \mathrm { p g a } ^ { k } } ( y ) - \alpha \mathrm { C L S } _ { \mathrm { p g a } ^ { k } } ( \hat { y } )
$$

where $\mathrm { C L S _ { p g a } ( ) }$ computes the probability of $0 \leq k ^ { t h } <$ $P G A$ persona-gender-age class out of $K$ classes 3. A PGA class is given by the combination of persona, age, and gender. Here, we consider five personas $P =$ $\{ O , C , E , A , N \}$ , two genders $G = \{ M a l e , F e m a l e \}$ , and three group of ages ${ \cal A } \ = \ \{ Y o u n g e r , M i d d l e \ -$ $A g e d , O l d e r \}$ . Therefore, the total combinations of all these three would be 30, hence we will have a total of 30 PGA classes.

2. Politeness Adherence Reward: This reward incentivizes the generation of responses that adhere to predefined politeness strategies, fostering courteous interactions:

$$
\Delta _ { 2 } = \mathrm { C L S } _ { \mathrm { p s } ^ { k } } ( y ) - \alpha \mathrm { C L S } _ { \mathrm { p s } ^ { k } } ( \hat { y } )
$$

where $\mathrm { C L S } _ { \mathrm { p s } } ( )$ computes the probability of $0 \leq k ^ { t h } <$ $P S$ politeness strategy class out of PS classes.

3. Empathy Consistency Reward: This reward promotes responses that demonstrate correct empathy strategy to understand the user’s emotional state and needs:

$$
\Delta _ { 3 } = \mathrm { C L S } _ { \mathrm { e s } ^ { k } } ( y ) - \alpha \mathrm { C L S } _ { \mathrm { e s } ^ { k } } ( \hat { y } )
$$

where $\mathrm { C L S _ { \mathrm { e s } } ( ) }$ computes the probability of $0 \leq k ^ { t h } <$ $E S$ empathetic-strategy class out of ES classes.

In each of the rewards, $\alpha = [ 1 , 2 ]$ acts as a penalization factor.

Smoothness Reward: These rewards encourage the model to produce responses that exhibit smooth transitions and coherence within the conversation. It penalizes abrupt changes or inconsistencies between consecutive utterances.

$$
R _ { \mathrm { s m o o t h n e s s } } = { \frac { 1 } { 1 + \exp \left( - \lambda \cdot \left( w _ { 4 } \cdot \Delta _ { \mathrm { s y n } } + w _ { 5 } \cdot \Delta _ { \mathrm { s e m } } \right) \right) } }
$$

In this equation, $w _ { 4 } + w _ { 5 } = 1$ are weighting factors that balances the contribution of syntactic and semantic smoothness scores. It determines the relative importance of each aspect in the overall smoothness reward. $\lambda$ is the scaling factor that controls the sensitivity of the reward to changes in smoothness. A higher $\lambda$ amplifies the importance of smoothness in the overall reward calculation, while a lower value reduces its impact. $\Delta _ { \mathrm { { s y n } } }$ represents the syntactic smoothness score which penalizes deviations from grammatical correctness computed as reciprocal of perplexity $( P P L ( ) )$ (Brown et al. 1992). It can be measured as:

$$
\Delta _ { \mathrm { s y n } } = \frac { 1 } { P P L ( \hat { y } ) }
$$

$\Delta _ { \mathrm { s e m } }$ represents the semantic smoothness score, which assesses the semantic coherence and relevance between consecutive utterances.

$$
\Delta _ { \mathrm { s e m } } = \mathrm { c o s i n e \_ s i m i l a r i t y } ( u _ { i - 1 } , \hat { y } ) )
$$

We define the overall reward $R$ as:

$$
R = \gamma R _ { \mathrm { t a s k - r e l e v a n c e } } + ( 1 - \gamma ) R _ { \mathrm { s m o o t h n e s s } }
$$

where $\gamma = [ 0 , 1 ]$ . Then, the advantage function $\hat { A } _ { t }$ is computed using the rewards obtained from the environment.

$$
\hat { A } _ { t } = R _ { t } - V ( s _ { t } )
$$

where $R _ { t }$ is the total reward obtained at time step $t$ , and $V ( s _ { t } )$ is the state-value function representing the expected cumulative reward from state $s _ { t }$ onwards. The policy $\pi _ { \boldsymbol { \theta } }$ is

updated using the proximal policy optimization (PPO) loss function:

$$
L ^ { P P O } ( \theta ) = - \mathbb { E } [ \operatorname* { m i n } ( r ( \theta ) \hat { A } _ { t } , \operatorname { c l i p } ( r ( \theta ) , 1 - \epsilon , 1 + \epsilon ) \hat { A } _ { t } ) ]
$$

where $r ( \theta )$ is the probability ratio, $\hat { A } _ { t }$ is the advantage function, and $\epsilon$ is the clipping parameter. The parameters $\theta$ of the policy $\pi _ { \boldsymbol { \theta } }$ are updated using gradient descent with the modified PPO loss incorporating the reward:

$$
\theta _ { t + 1 } = \theta _ { t } - \alpha \nabla _ { \theta } L ^ { P P O } ( \theta )
$$

where $\alpha$ is the learning rate.

# Experiments

Due to space restrictions, the Baselines and implementation details of all the models are given in Section Baselines and Implementation Details of the Appendix.

# Evaluation Metrics

Both automatic and human evaluations are conducted to assess the performance of the proposed system EDiSS.

Automatic Evaluation Metrics: We employ three metrics to evaluate task-relevance viz. user-profile consistency (UPC), politeness-strategy accuracy (PSA), and EmpathyStrategy accuracy (ESA):

$$
U P C = \mathbb { E } _ { x _ { i } , y _ { i } } 1 \{ C L S _ { \mathrm { p g a } } ( y _ { i } ) = C L S _ { \mathrm { p g a } } ( \hat { y } ) \} ,
$$

$$
P S A = \mathbb { E } _ { x _ { i } , y _ { i } } 1 \{ C L S _ { \mathrm { p s } } ( y _ { i } ) = C L S _ { \mathrm { p s } } ( \hat { y } ) \} ,
$$

$$
E S A = \mathbb { E } _ { x _ { i } , y _ { i } } 1 \{ C L S _ { \mathrm { e s } } ( y _ { i } ) = C L S _ { \mathrm { e s } } ( { \hat { y } } ) \} ,
$$

Additionally, we evaluate EDiSS in terms of language and dialogue quality using three metrics: Perplexity $( P P L )$ (Brown et al. 1992), Response Length Ratio $\left( R _ { \mathrm { l e n } } \right)$ , Nonrepetitiveness $( N _ { \mathrm { r e p } } )$ .

$$
P P L = \frac { \sum _ { r } \exp \left( - \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \log P ( y _ { i } | x _ { i } ) \right) } { r }
$$

where $n$ is the total number of tokens in the generated responses, $r$ is the total number of the generated responses, and $P ( y _ { i } | x _ { i } )$ is the probability assigned by the language model to the $i ^ { t h }$ token given the input $x _ { i }$ .

$$
R _ { \mathrm { l e n } } = \frac { \sum _ { r } ( n ) } { r } .
$$

$$
N _ { \mathrm { r e p } } = \frac { 1 } { 2 } ( B S _ { \mathrm { F 1 } } ( y _ { i } , y _ { i - 1 } ) + B S _ { \mathrm { F 1 } } ( y _ { i } , y _ { i - 2 } ) ) ,
$$

Human Evaluation Metrics: Human evaluation involves 10 evaluators, who were compensated according to the university norms. All ten evaluators have post-graduate in English Linguistics and have at least three years of experience in doing the similar kind of tasks. To avoid bias these evaluators are different from annotators. The evaluation consists of two phases. In the first phase, each evaluator interacts with EDiSS five times, using different sets of utterances. They rate the conversations based on a Likert scale of 1-5 for seven metrics: persona accuracy, gender-age accuracy, politeness accuracy, empathy accuracy, fluency (FY), consistency (CY), and non-repetitiveness (NR). The scale denotes low to high intensity, e.g., a rating of 1 for persona accuracy indicates low consistency, while 5 denotes high consistency. These 50 evaluations are reviewed by medical experts, achieving an agreement score of $8 3 . 4 \%$ . In the second phase, based on expert feedback, evaluators re-evaluate the initial 50 interactions and assess an additional 15 interactions each, resulting in a total of 200 evaluated interactions.

# Results and Analysis

Automatic Evaluation: Table 2 presents the results of automatic evaluation metrics for various baselines: GPT2-large (Radford et al. 2019), ARDM (Wu et al. 2021), ZYPHER7B (Tunstall et al. 2023), Phi-1.5 (Li et al. 2023), Mistral-7B (Jiang et al. 2023), Llama2-7B (Touvron et al. 2023), phi-2 (Li et al. 2023), Mistral-8B (Jiang et al. 2023), Llama3-8B (AI@Meta 2024), DSS: $D S S _ { \theta }$ , EDiSS-R: EDiSS with $R =$ 0, EDiSS-TR: EDiSS with $R \ = \ R _ { \mathrm { s m o o t h n e s s } }$ , and EDiSSSR: EDiSS with $R \ = \ R _ { \mathrm { t a s k } }$ -relevance, compared against our proposed EDiSS. Significant differences were observed between EDiSS and all other models $( \mathtt { p } < 0 . 0 5 )$ . Among the compared models, EDiSS consistently outperforms others across all metrics.

In examining task-specific metrics: UP C, P SA, and $E S A$ a discernible pattern is seen i.e. GPT2-large $<$ ARDM $<$ Phi- $1 . 5 ~ <$ Llama2-7B $<$ Mistral-7B $<$ ZYPHER-7B $< \mathrm { P h i } { \cdot } 2 <$ Mistral-8B $< \mathrm { L l a m a } 3 { - } 8 \mathrm { B } < \mathrm { D S S } \approx \mathrm { E D i S S - }$ $\textsf { R } <$ EDiSS-TR $<$ EDiSS-SR $<$ EDiSS. Notably, EDiSS and EDiSS-R exhibit similar performance, attributed to EDiSS’s initialization from $D S S _ { \theta }$ . The better performance of EDiSS-SR can be traced back to the influence of $\Delta _ { 1 }$ , $\Delta _ { 2 }$ , and $\Delta _ { 3 }$ , underscoring the pivotal role of persona, gender, age, politeness, and empathy in guiding EDiSS to formulate persona-consistent, polite, and compassionate responses. Moreover, Table 2 demonstrates that EDiSS outperforms all the 13 baselines in terms of P P L, Rlen, and Nrep, following the same order as above. The better performance of EDiSS-TR is attributed to $\Delta _ { { s y n } }$ and $\Delta _ { { s y m } }$ , which steer it towards more natural and contextually consistent responses.

EDiSS’s success across all the metrics can be attributed to its assimilation of patient profile information and adaptation of politeness and empathy levels. The integration of task-relevance reward aids EDiSS in approximating a more precise distribution, further enhancing its competitive edge over the eight baselines. The inclusion of smoothness reward fosters a dynamic rapport between the system and the user, enabling EDiSS to focus on pertinent details and craft refined responses. This results in better language understanding and, the ability to generate contextually relevant, diverse, and engaging responses. This underscores the dual necessity of all five rewards in yielding responses of elevated quality, validating our initial hypothesis. Generated responses of different models with respective qualitative analyses are illustrated in Section Qualitative analysis and Table 9 of the appendix.

Human Evaluation: Table 3 showcases human evaluation results for GPT2-large, ARDM, Phi-1.5, Llama2-7B, Mistral-7B, ZYPHER-7B, Phi-2, Mistral-8B, Llama3-8B, DSS, EDiSS, EDiSS-R, EDiSS-TR, and EDiSS-SR, compared against EDiSS. Similar to the automatic evaluation,

Table 2: Results of automatic evaluation. Significant differences were observed between EDiSS and all other models $( \mathtt { p } < 0 . 0 5 )$   

<html><body><table><tr><td>Model</td><td>UPC</td><td>PSA</td><td>ESA</td><td>PPL</td><td>Rlen</td><td>Nrep</td></tr><tr><td>GPT2-large (Radford et al. 2019)</td><td>44.1%</td><td>67.9%</td><td>55.3%</td><td>24.34</td><td>10.12</td><td>0.41</td></tr><tr><td>ARDM(Wuetal.2021)</td><td>50.1%</td><td>70.2%</td><td>62.1%</td><td>10.21</td><td>12.01</td><td>0.32</td></tr><tr><td>Phi-1.5 (Li et al. 2023)</td><td>50.9% 51.3%</td><td>71.6%</td><td>63.4%</td><td>8.01</td><td>15.59</td><td>0.24</td></tr><tr><td>Llama2-7B (Touvron etal.2023)</td><td></td><td>72.5%</td><td>65.8%</td><td>7.90</td><td>16.02</td><td>0.22</td></tr><tr><td>Mistral-7B (Jiang et al. 2023)</td><td>51.5%</td><td>73.8%</td><td>66.1%</td><td>7.80</td><td>16.10</td><td>0.20</td></tr><tr><td>ZYPHER-7B(Tunstall etal. 2023)</td><td>52.1%</td><td>74.5%</td><td>66.8%</td><td>7.41</td><td>16.24</td><td>0.19</td></tr><tr><td>Phi-2 (Li et al. 2023)</td><td>54.4%</td><td>76.5%</td><td>68.9%</td><td>6.61</td><td>17.95</td><td>0.16</td></tr><tr><td>Mistral-8B (Jiang et al.2023) Llama3-8B (AI@Meta2024)</td><td>55.6%</td><td>77.2%</td><td>69.5%</td><td>5.95</td><td>19.10</td><td>0.14</td></tr><tr><td></td><td>56.1%</td><td>77.5%</td><td>70.2%</td><td>5.60</td><td>19.95</td><td>0.12</td></tr><tr><td>DSS (Phi3-small) (Abdin et al. 2024)</td><td>57.9%</td><td>79.1%</td><td>71.5%</td><td>4.85</td><td>19.70</td><td>0.10</td></tr><tr><td>EDiSS-R</td><td>57.5%</td><td>78.8%</td><td>71.5%</td><td>4.88</td><td>19.70</td><td>0.10</td></tr><tr><td>EDiSS-TR</td><td>58.4%</td><td>79.6%</td><td>72.0%</td><td>4.50</td><td>19.85</td><td>0.09</td></tr><tr><td>EDiSS-SR</td><td>59.3%</td><td>80.5%</td><td>73.6%</td><td>4.10</td><td>20.05</td><td>0.08</td></tr><tr><td>EDiSS</td><td>60.7%</td><td>81.5%</td><td>74.9%</td><td>3.60</td><td>20.45</td><td>0.07</td></tr></table></body></html>

Table 3: Results of human evaluation   

<html><body><table><tr><td>Model</td><td>UPC</td><td>PSA</td><td>ESA</td><td>FY</td><td>CY</td><td>Nrep</td></tr><tr><td>GPT2-large</td><td>1.70</td><td>2.12</td><td>2.01</td><td>2.60</td><td>2.35</td><td>2.40</td></tr><tr><td>ARDM</td><td>2.05</td><td>2.34</td><td>2.25</td><td>3.46</td><td>2.72</td><td>2.76</td></tr><tr><td>Phi-1.5</td><td>2.26</td><td>2.95</td><td>2.63</td><td>3.64</td><td>2.95</td><td>2.98</td></tr><tr><td>Llama2-7B</td><td>2.30</td><td>3.05</td><td>2.68</td><td>3.70</td><td>3.05</td><td>3.04</td></tr><tr><td>Mistral-7B</td><td>2.38</td><td>3.10</td><td>2.75</td><td>3.75</td><td>3.10</td><td>3.08</td></tr><tr><td>ZYPHER-7B</td><td>2.40</td><td>3.15</td><td>2.78</td><td>3.75</td><td>3.10</td><td>3.10</td></tr><tr><td>Phi-2</td><td>2.45</td><td>3.20</td><td>2.91</td><td>3.83</td><td>3.18</td><td>3.15</td></tr><tr><td>Mistral-8B</td><td>2.65</td><td>3.40</td><td>3.05</td><td>3.95</td><td>3.38</td><td>3.35</td></tr><tr><td>Llama3-8B</td><td>2.70</td><td>3.50</td><td>3.10</td><td>4.00</td><td>3.45</td><td>3.40</td></tr><tr><td>DSS</td><td>2.76</td><td>3.62</td><td>3.18</td><td>4.10</td><td>3.54</td><td>3.50</td></tr><tr><td>EDiSS-R</td><td>2.72</td><td>3.55</td><td>3.15</td><td>4.10</td><td>3.52</td><td>3.50</td></tr><tr><td>EDiSS-TR</td><td>2.82</td><td>3.68</td><td>3.25</td><td>4.18</td><td>3.70</td><td>3.65</td></tr><tr><td>EDiSS-SR</td><td>3.01</td><td>3.90</td><td>3.55</td><td>4.30</td><td>3.92</td><td>3.95</td></tr><tr><td>EDiSS</td><td>3.15</td><td>4.02</td><td>3.85</td><td>4.42</td><td>4.05</td><td>4.10</td></tr></table></body></html>

EDiSS outperforms all the other models across metrics: UPC, PSA, ESA, $F Y$ , $C Y$ , and $N _ { r e p }$ . A nuanced contrast emerges between EDiSS and EDiSS-TR, emphasizing the significance of task-relevance rewards— $\lvert \Delta _ { 1 }$ , $\Delta _ { 2 }$ , and $\Delta _ { 3 }$ in crafting persona-sensitive, polite, and empathetic responses. Notably, EDiSS surpasses EDiSS-TR and EDiSSSR, indicating the pivotal role of all five rewards in achieving fluent, consistent, non-repetitive, courteous, and compassionate responses. These enhancements reflect EDiSS’s ability to generate human-like and engaging conversations, thus boosting user satisfaction. The superior performance of EDiSS is attributed to its reward-based architecture, optimizing response quality.

Both automatic and human evaluation validate EDiSS’s efficacy in delivering high-quality conversational support to individuals with physical disabilities, suggesting its potential to enhance user experience and overall well-being significantly.

# Error Analysis

While EDiSS demonstrates effectiveness in providing tailored support to individuals with physical disabilities, some areas for improvement remain. A key issue is the occasional misalignment between user personas and generated responses, stemming from the complexity of human traits and challenges in accurately capturing them in the dataset. Instances of sub-optimal politeness, empathy strategies, and fragmented dialogue flow were also observed, often due to limitations in training data or contextual understanding over extended conversations. Classifier accuracy plays a crucial role in EDiSS’s performance. With accuracies of $9 1 . 6 \%$ (politeness), $8 5 . 4 \%$ (empathy), and $8 0 . 2 \%$ (PGA), most responses align with user profiles, ensuring polite and empathetic strategies. Misclassifications, however, may introduce minor biases. To mitigate this, reward functions adjust values based on classification confidence, reducing the impact of misclassifications. As shown in Table 2, EDiSS achieves maximum values of $6 0 . 7 \%$ (UPC), $8 1 . 5 \%$ (PSA), and $7 4 . 9 \%$ (ESA). To enhance EDiSS, we propose integrating adaptive user feedback for dynamic response tuning, refining classifiers with real-world data to improve demographic robustness, and exploring lightweight adaptations for offline use in low-connectivity areas. These steps aim to enhance EDiSS’s adaptability, inclusivity, and reliability.

# Conclusion

In this paper, we present EDiSS, an empathetic disability support system designed for individuals with physical disabilities. Using user personas based on the OCEAN model, gender, and age, EDiSS delivers personalized assistance in a supportive environment. It integrates politeness and empathy strategies to enhance user experience. Built on the PDCARE dataset, enriched with user profile annotations, EDiSS optimizes responses with task-relevance and smoothness rewards. Empirical evaluations, both automatic and human, confirm its effectiveness in tailored support. EDiSS lays the groundwork for future research on more inclusive, personalized systems, with potential extensions to broader domains and additional user profile factors.

# Ethical Statement

Ethical considerations are central to the development of EDiSS, especially given its focus on supporting individuals with physical disabilities. We adhered to strict ethical guidelines, prioritizing user privacy, autonomy, and wellbeing. Data privacy was ensured through anonymization and compliance with data protection regulations. The PDCARE dataset emphasized diversity in age, gender, personality traits, and disabilities, exposing the model to a wide range of user profiles. Bias-check mechanisms were employed to identify and manually correct harmful or stereotypical outputs, ensuring equitable treatment across persona combinations. Our data and methodology are approved by the university’s ethics review board, and the dataset will be available for research purposes upon request. These efforts reflect our commitment to safeguarding the dignity, rights, and well-being of users.