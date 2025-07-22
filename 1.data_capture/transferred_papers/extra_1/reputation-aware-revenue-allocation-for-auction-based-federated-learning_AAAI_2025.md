# Reputation-aware Revenue Allocation for Auction-based Federated Learning

Xiaoli Tang, Han Yu

College of Computing and Data Science, Nanyang Technological University, Singapore {xiaoli001, han.yu}@ntu.edu.sg

# Abstract

Auction-based Federated Learning (AFL) has gained significant research interest due to its ability to incentivize data owners (DOs) to participate in FL model training tasks of data consumers (DCs) through economic mechanisms via the auctioneer. One of the critical research issues in AFL is decision support for the auctioneer. Existing approaches are based on the simplified assumption of a single, monopolistic AFL marketplace, which is unrealistic in real-world scenarios where multiple AFL marketplaces can co-exist and compete for the same pool of DOs. In this paper, we relax this assumption and frame the AFL auctioneer decision support problem from the perspective of helping them attract participants in a competitive AFL marketplace environment while safeguarding profit. To achieve this objective, we propose the Auctioneer Revenue Allocation Strategy for AFL (ARAS-AFL). We design the concepts of the attractiveness and competitiveness from the perspective of autioneer reputation. Based on the Lyapunov optimization, ARAS-AFL helps individual AFL auctioneer achieve the dual objective of balancing the reputation management costs and its own profit by designing a dynamic revenue allocation strategy. It takes into account both the auctioneer’s revenue and the changes in the number of participants on the AFL marketplace. Through extensive experiments on widely used benchmark datasets, ARAS-AFL demonstrates superior performance compared to state-of-theart approaches. It outperforms the best baseline by $4 9 . 0 6 \%$ , $9 8 . 6 9 \%$ , $1 0 . 3 2 \%$ , and $4 . 7 7 \%$ in terms of total revenue, number of data owners, public reputation and accuracy of federated learning models, respectively.

# Introduction

With growing concerns over data privacy and user confidentiality, federated learning (FL) (Yang et al. 2019; Hu et al. 2024; Guo et al. 2021; Tang et al. $2 0 2 4 \mathrm { c }$ ; Li et al. 2021) has gained significant interest in recent years due to its versatile applications (Sun et al. 2024). In FL settings, self-interested data owners $( D O s )$ meticulously evaluate various factors, including costs, potential gains, and utility, when deciding which FL data consumer $( D C )$ to collaborate with. This necessity has spurred the development of FL incentive mechanisms (Tang 2024; Khan et al. 2020), designed to incentivize DO participation through strategic reward systems.

Auction-based federated learning (AFL) has emerged as a pivotal area within FL incentive mechanism research, primarily due to its ability to achieve both efficiency and fairness. AFL research revolves around three main areas based on the primary stakeholders involved: (Tang and Yu 2023d):

• DC-Centric Approaches: These methods assist DCs from two main perspectives: 1) Selecting appropriate DOs in monopolistic market scenarios, and 2) Strategically placing bids to acquire data from DOs in competitive market environments, with the aim of optimizing specific key performance indicators (KPIs) while adhering to budgetary constraints.   
• Auctioneer-Centric Approaches: These approaches have two key focuses: 1) Internally, optimizing the DCDO matching process, pricing mechanisms, and auction mechanism designs to achieve desired outcomes, such as maximizing social welfare or minimizing costs for the entire AFL ecosystem. 2) Externally, enhancing the competitiveness and attractiveness of the AFL ecosystem to retain existing participants, especially DOs who own valuable data resources while attracting new participants while optimizing its own profit.   
• DO-Centric Approaches: These methodologies concentrate on determining the optimal allocation of local resources for federated learning tasks and customizing reserve prices to maximize the profitability of DOs.

In this paper, we focus on auctioneer-centric approaches in AFL. However, existing methods addressing auctioneercentric challenges primarily concentrate on internal aspects, i.e., assisting the auctioneer in optimizing the matching between DCs and DOs, designing pricing mechanisms and auction mechanisms to achieve desired outcomes. To the best of our knowledge, no work has addressed external aspects, i.e., attracting more participants, especially DOs. Existing works for the internal aspect are based on the simplified assumption of a single, monopolistic AFL marketplace. However, this assumption does not hold true in practice.

Due to the self-interested nature of both the AFL marketplaces and DOs, data trading processes under AFL settings are a two-way interactive dynamic between them. On the one hand, the AFL marketplace, represented by the auctioneer, has the ability to selectively admit participants and determine the rewards for DOs. These strategies are designed to maximize the auctioneer’s profit as the coordinator and manager. Conversely, DOs also possess the autonomy to choose which AFL marketplace to join, given the availability of multiple options. Their decision is influenced by the potential payoffs they can obtain. This interdependent relationship creates a strategic interplay between the auctioneer’s objectives of maximizing profit and retaining participants, and the DOs’ goals of maximizing their individual payoffs. This two-way interactive process requires decision support support for the auctioneer that balance the interests of both AFL marketplaces and DOs, fostering a sustainable and mutually beneficial data trading environment within the AFL ecosystem (Tang and Yu 2024d).

To this end, we relax the monopolistic AFL marketplace assumption and assist the auctioneer in making decisions for external aspects by proposing the Auctioneer Revenue Allocation Strategy for AFL (ARAS-AFL). ARAS-AFL frames the objective of enhancing the competitiveness in the AFL ecosystem to attract DOs as a reputation management problem. Focused on individual-level decision-making for AFL marketplaces, i.e., achieving the dual objective of balancing the reputation management costs and profit of the AFL marketplace, ARAS-AFL designs a dynamic revenue allocation strategy under the reputation management framework for the AFL marketplace based on Lyapunov optimization. This strategy takes into consideration the marketplace’s revenue and the changes in the number of DOs participating in the marketplace.

To the best of our knowledge, ARAS-AFL is the first revenue allocation decision support approach to help each AFL auctioneer manage its competitiveness. Extensive experiments on widely used benchmark datasets demonstrates its superiority over state-of-the-art approaches, outperforming the best baseline by $4 9 . 0 6 \%$ , $9 8 . 6 9 \%$ , $1 0 . 3 2 \%$ , and $4 . 7 7 \%$ in terms of total revenue, number of data owners, public reputation and accuracy of federated learning models, respectively.

# Related Work

According to the target skateholder, existing AFL methods could be roughly grouped into three main categories (Tang et al. 2024a): 1) DC-centric approaches, 2) DO-centric approaches, and 3) auctioneer-centric approaches.

# DC-centric approaches

Based on auction mechanisms applied, methods in this category could be further divided into two main groups: 1) methods for reverse auction and 2) methods for forward auction.

Methods for reverse auction These methods (Jiao et al. 2020; Zhou et al. 2021; Yuan et al. 2021; Wu et al. 2023; Fan et al. 2020; Zhang, Wu, and Pan 2021; Zhang, Jingwen and Wu, Yuezhou and Pan, Rong 2022; Zhang, Wu, and Pan” 2022; Zeng et al. 2020; Batool, Zhang, and Toews 2022; Batool, Zahra and Zhang, Kaiwen and Toews, Matthew 2023) select and recruit data owners (DOs) in a monopolistic marketplace, aiming to maximize the DC’s key performance indicators under certain constraints based on the reverse auction mechanism. For instance, in (Jiao et al. 2020), the authors first group DOs based on Earth Mover’s Distances (EMD) (Zhao et al. 2018) to optimize social welfare. The DC then greedily selects DOs from each group, determining payments based on marginal virtual social welfare density. To further enhance social welfare, the authors incorporate a graph neural network to manage relationships among DOs and use deep reinforcement learning to determine the winning DOs and their payments. On the other hand, (Zhang, Wu, and Pan 2021) focuses on maximizing the utility of the DC by integrating the reputation mechanism and blockchain technology with the reverse auction mechanism. The proposed approach, RRAFL, assists the DC in selecting DOs and determining payments to selected DOs.

Methods for forward auction Methods (Tang and Yu 2023d,b,c; Tang et al. 2024b; Tang and Yu 2024a,b,c) in this category aim to assist the DC in bidding for DOs in the competitive AFL marketplace by designing bidding strategies to maximize key performance indicators while adhering to budget constraints. Specifically, in (Tang and $\mathrm { Y u } 2 0 2 3 \mathrm { d } )$ , the authors demonstrate that the optimal bidding strategy for DCs to maximize utility depends on utility estimation and winning probability estimation. Subsequently, the optimal bidding strategy design is translated into the utility estimation function design and winning probability function design. By formulating these two functions, the authors derive the final form of the bidding function. In contrast to (Tang and $\mathrm { Y u } \ 2 0 2 3 \mathrm { d } ) ,$ ), which solely considers the competitive relationship among DCs in the AFL marketplace, (Tang and $\mathrm { Y u } 2 0 2 3 \mathrm { b } )$ ) also accounts for potential cooperative relationships among DCs. Based on this consideration, the authors propose MARL-AFL by framing the AFL marketplace from the perspective of a multi-agent bidding system.

# DO-centric approaches methods

These methods (Thi Le et al. 2021; Lu et al. 2023; Le et al. 2020; Zeng et al. 2020) are designed to assit the DOs in determining the optimal allocation of their data resources and formulating their asking profiles, with the aim of maximizing key performance indicators for the DOs. For instance, in (Thi Le et al. 2021), upon receiving FL training task profiles from the DC, which include the maximum tolerable time for FL training, the proposed method helps each DO optimize their asking profiles. On the other hand, some methods, such as (Tang and $\mathrm { Y u } 2 0 2 3 \mathrm { a } ,$ ), focus on assisting DOs in determining the number of FL training tasks to accept, the number of FL training tasks to subdelegate, and how to price these tasks. To achieve this multifaceted objective, the authors propose PAS-AFL by leveraging Lyapunov optimization and taking into account the DO’s current reputation, pending FL tasks, willingness to engage in FL model training, and trust relationships with other DOs.

# Auctioneer-centric approaches

Auctioneer-centric approaches (Seo, Niyato, and Elmroth 2021; Seo, Eunil and Niyato, Dusit and Elmroth, Erik 2022; Xu et al. 2023; Roy et al. 2021; Mai et al. 2022; Wang et al. 2023) are generally designed to assist the auctioneer in making optimal DC-DO matching decisions to maximize key performance indicators from the perspective of the entire

AFL marketplace. These approaches are based on mechanisms such as double auctions, combinatorial auctions, and reverse auctions. In (Xu et al. 2023), the proposed method aims to maximize social welfare while protecting the auctioneer’s utility. The approach involves two main stages: 1) combinatorial auction stage, where the platform selects winners whose total utility, combined with the platform’s utility, is greater than zero, and 2) bargaining stage, where winners are classified into two categories with different payment methods after completing the training model. The goal is to ensure that the auctioneer’s utility remains positive.

Our work falls under the auctioneer-centric approaches category. However, in contrast to existing methods in this category, which are generally based on the simplified assumption of a single AFL marketplace, the proposed ARAS-AFL method relaxes this unrealistic assumption. Instead, it frames the decision support for the auctioneer from the perspective of helping it manage its reputation in the competitive AFL ecosystem and attract more participants to obtain higher revenue.

# Preliminaries

In this section, we introduce some background information related to the proposed method ARAS-AFL.

# AFL Ecosystem

We delve into a dynamic and competitive AFL ecosystem wherein $| \mathcal { N } |$ marketplaces (referred to as the marketplace set $\mathcal { N }$ ) actively vie for the attention of DOs, enticing them to join their respective platforms at any given moment. Within this ecosystem, AFL marketplaces harness the data resources contributed by DOs to attract DCs in need of FL model training services. These DCs release their training tasks and offer monetary compensation for the services provided. Nevertheless, DOs retain the autonomy to assess and choose the AFL marketplace they prefer to join, enabling them to engage in FL training tasks facilitated by their selected platform. Due to privacy concerns, AFL marketplaces may possess limited access to comprehensive attribute information about the DOs. Simultaneously, DOs who have not previously participated in FL training tasks hosted by a particular AFL marketplace may lack insight into the potential rewards and incentives offered by that platform. It is crucial to acknowledge that AFL marketplaces, acting as intermediaries, aim to generate revenue by overseeing FL training tasks. They accomplish this by strategically distributing the payoffs received from DCs among themselves and the participating DOs, driven by their self-interested motive to maximize profitability.

In this intricate landscape, each AFL marketplace $i \in \mathcal { N }$ must adopt an effective FL revenue allocation strategy, denoted as $\alpha _ { i }$ , to enhance its attractiveness and recruit a substantial pool of DOs to its platform. The revenue allocation strategy plays a pivotal role in enhancing the competitiveness of the AFL marketplace, helping it attract more DOs, and maximizing long-term revenue. This approach benefits the marketplace while fostering a sustainable environment for all stakeholders in the AFL ecosystem.

We frame AFL marketplace attractiveness from the perspective of reputation. We then orient the objectives of AFL marketplaces toward devising effective revenue allocation strategies under the reputation management framework.

# The Data Trading Process under ARAS-AFL Framework

The data trading process among DCs and DOs coordinated and managed by the AFL marketplace under the proposed ARAS-AFL in a competitive AFL ecosystem operates in the following six steps within a given time slot $t$ , which can represent a day, a week, or any other suitable time frame depending on the specific application scenario.

1. DCs select an AFL marketplace to release their corresponding FL training tasks based on their observation of the number of DOs currently participating in each marketplace. As many new DCs may have limited information about the available AFL marketplaces, they intuitively gravitate towards marketplaces with a larger number of DOs, as this implies a higher probability of completing the FL training tasks with high quality and in a timely manner.   
2. DOs who wish to participate in FL choose an AFL marketplace based on its reputation. The intuition behind this decision is that an AFL marketplace with a good reputation is more likely to provide higher payoffs to participating DOs. Each AFL marketplace has a perceived reputation based on satisfaction ratings provided by past DOs, reflecting their experiences with the respective marketplace. This reputation information, jointly maintained by the DOs, serves as a valuable guide for future DOs in choosing which AFL marketplace to join.   
3. The AFL marketplaces allocate the FL model training tasks released by DCs to available DOs on the marketplace, considering factors such as their availability and capabilities, followed by DOs training the FL models following the traditional FL training algorithm like FedAvg (McMahan et al. 2017). After obtaining the trained FL models, the AFL marketplace transfers the trained models to the corresponding DCs.   
4. Each AFL marketplace determines the revenue-sharing ratio based on its revenue allocation strategy between the marketplace itself and the participating DOs.   
5. After the DOs complete the assigned FL training tasks, they evaluate the AFL marketplace based on their earned income. At the end of the time slot $t$ , the reputation of each marketplace is updated based on the reputation evaluation ratings provided by the participating DOs, as detailed in Section .   
6. Finally, each AFL marketplace dynamically adjusts its revenue allocation strategy based on the changes in the DO population on its platform and its own income generated during the time slot $t$ . The details of the FL training task allocation strategy will be discussed in Section .

The Proposed ARAS-AFL Approach In this section, we delineate the primary mechanisms governing the data trading process within the proposed

ARAS-AFL. This encompasses the procedures by which DCs issue their FL model training tasks to AFL marketplaces, the methods by which DOs assess the reputation of these marketplaces, and the strategies employed by AFL marketplaces to enhance their appeal to DOs while ensuring profitability. Central to this discussion is the development of effective revenue allocation strategies within the reputation management framework, which serves to optimize the attractiveness of AFL marketplaces to DOs while safeguarding their financial viability.

# DCs Release FL Training Tasks

For DCs, selecting a high-quality AFL marketplace can significantly improve their chances of obtaining high quality FL models in a timely manner. However, in the absence of crucial information regarding AFL marketplaces, such as FL model training capacity and the quality of DOs, DCs may tend to base their decisions on the marketplace’s DO availability. Typically, a higher number of DOs on a marketplace correlates with a greater likelihood of completing FL model training tasks swiftly, albeit disregarding other factors. Consequently, the probability of a DC releasing an $\mathrm { F L }$ model training task on AFL marketplace $i$ during time period $t$ can be expressed as:

$$
p _ { i } ^ { r e l e a s e } ( t ) = \frac { M _ { i } ^ { \prime } ( t ) } { \sum _ { n \in \mathcal { N } } M _ { n } ^ { \prime } ( t ) } ,
$$

where $M _ { i } ^ { \prime } ( t ) = M _ { i } ( t ) + M _ { b a s e }$ denote the initial number of DOs available on AFL marketplace $i$ at the beginning of time period $t$ , where $M _ { i } ( t )$ represents the number of DOs on marketplace $i$ during time period $t$ . Here, $M _ { b a s e }$ denotes a small positive value, signifying the loyal core or intrinsic DOs of marketplace $i$ . This inclusion ensures that even weaker AFL marketplaces are afforded an opportunity to rejuvenate and expand.

# AFL Marketplace Reputation Evaluation

The reputation of an AFL marketplace encompasses two facets: firstly, the private reputation value logged by each DO, and secondly, the public reputation value, both derived from the Beta Reputation System (BRS) (Josang and Ismail 2002). In this context, we assume DOs provide honest assessments of their satisfaction with each AFL marketplace, with evaluations conducted after receiving payment for completing FL training tasks on the respective marketplace.

For AFL marketplace $i$ , the satisfaction level of DO $j$ regarding $i$ , based on its earnings prof $i t _ { i } ^ { j }$ from the FL model training task, falls into two categories: satisfaction, indicated by $p r o f i t _ { i } ^ { j } \ge p r o ^ { \bar { } } f i t ^ { j }$ , where $\hat { p r o f i t ^ { j } }$ denotes DO $j$ ’s average earnings across various AFL marketplaces in the past; and dissatisfaction, denoted by pro $f i t _ { i } ^ { j } < p r \bar { o } f i t ^ { j }$ . We employ the variables $s c _ { i } ^ { j }$ and $u c _ { i } ^ { j }$ to tally the occurrences of DO $j$ expressing satisfaction and dissatisfaction with AFL marketplace $i$ , respectively.

Subsequently, we calculate the reputation value $r _ { i } ^ { j }$ for DO $j$ concerning AFL marketplace $i$ using the BRS as follows:

$$
r _ { i } ^ { j } = \mathbb { E } [ B e t a ( s c _ { i } ^ { j } + 1 , u c _ { i } ^ { j } + 1 ) ] = \frac { s c _ { i } ^ { j } + 1 } { s c _ { i } ^ { j } + u c _ { i } ^ { j } + 2 } .
$$

Regarding the public reputation value of each AFL marketplace, the proposed ARAS-AFL upholds a comprehensive reputation table. This table consolidates satisfaction evaluations provided by DOs engaged in $\mathrm { F L }$ model training tasks within the respective marketplace. It serves as a public attribute of the AFL marketplace, accessible to all DOs. The public reputation value $r _ { i } ^ { * }$ is derived through the following calculation:

$$
\begin{array} { l } { r _ { i } ^ { * } = \mathbb { E } [ B e t a ( \displaystyle \sum _ { j \in { \mathcal { K } } } ( s c _ { i } ^ { j } + 1 ) , \displaystyle \sum _ { j \in { \mathcal { K } } } ( u c _ { i } ^ { j } + 1 ) ) ] } \\ { = \displaystyle \frac { \sum _ { j \in { \mathcal { K } } } ( s c _ { i } ^ { j } + 1 ) } { \sum _ { j \in { \mathcal { K } } } ( s c _ { i } ^ { j } + u c _ { i } ^ { j } + 2 ) } , } \end{array}
$$

where $\kappa$ is the DO set.

By integrating the public reputation $r _ { i } ^ { * }$ and the private reputation $r _ { i } ^ { j }$ logged by DO $j$ , a holistic reputation value $r _ { i } ^ { j ^ { \prime } }$ for AFL marketplace $i$ can be derived. This comprehensive measure encapsulates both the collective perception and the individual experiences of DO $j$ within the marketplace. The formulation is as follows:

$$
r _ { i } ^ { j ^ { \prime } } = ( 1 - \gamma ) r _ { i } ^ { * } + \gamma r _ { i } ^ { j } ,
$$

where $\gamma$ signifies the confidence level of DO $j$ in assessing the reputation of AFL marketplace $i$ . If DO $j$ has not previously engaged in any FL model training tasks on AFL marketplace $i$ , its decision-making relies solely on the public reputation of the marketplace. In this case, setting $\gamma$ to 0 results in $r _ { i } ^ { j ^ { \prime } } = r _ { i } ^ { \ast }$ . Conversely, if DO $j$ places full trust in its own evaluation of AFL marketplace $i$ , $\gamma$ is set to 1, leading to $r _ { i } ^ { j ^ { \prime } } = r _ { i } ^ { j }$ . This approach empowers both new and existing DOs with a flexible means to select AFL marketplaces based on their individual preferences and experiences.

# Reputation Management Model

AFL marketplaces often allocate a larger portion of their profits (denoted as $\alpha$ ) to DOs in an effort to bolster their reputation and, consequently, attract a greater number of DOs. However, despite increasing the proportion of rewards for FL model training tasks, AFL marketplace $i$ may not consistently draw more DOs due to uncertainties regarding competitors’ strategies. This scenario can lead to regret for the marketplace.

To address this, we introduce a virtual queue $q _ { i } ( t )$ for each AFL marketplace $\mathbf { \chi } _ { i }$ to quantify its regret. This virtual queue encapsulates the dynamics of marketplace $i$ ’s regret and can be formulated as follows:

$$
q _ { i } ( t + 1 ) = \operatorname* { m a x } [ q _ { i } ( t ) + \Delta p _ { i } ( t ) - \Delta M _ { i } ( t ) , 0 ] .
$$

$\Delta M _ { i } ( t )$ denotes the change in the number of DOs on AFL marketplace $i$ from $( t - 1 )$ -th time period to $t$ -th time period. $\Delta p _ { i } ( t )$ denotes the amount of variation in the payoffs paid to DOs by platform $i$ during $( t - 1 )$ -the time period and $t$ -th time period and is calculated as

$$
\begin{array} { r } { \Delta p _ { i } ( t ) = ( \alpha _ { i } ( t ) P _ { i } ( t ) - \alpha _ { i } ( t - 1 ) P _ { i } ( t - 1 ) ) \mathbb { I } _ { \Delta M _ { i } ( t ) < 0 } , } \end{array}
$$

where $P _ { i } ( t )$ represents the aggregate payoff offered by DCs to AFL marketplace $i$ for completed $\mathrm { F L }$ model training tasks at time $t$ . The function $\mathbb { I } _ { ( \mathrm { c o n d i t i o n } ) }$ serves as an indicator, taking the value 1 if the condition holds true, and 0 otherwise.

The virtual queue $q _ { i } ( t )$ in Eq. (5) serves as a cumulative measure of regret for AFL marketplace $i$ , reflecting its profit-sharing strategy’s impact on DO attraction. Its dynamics evolve as follows:

• The regret queue $q _ { i } ( t )$ for AFL marketplace $i$ grows under the condition that $\Delta M _ { i } ( t ) \ < \ 0$ upon its update at time $t$ . In this scenario, $q _ { i } ( t )$ expands by $\alpha _ { i } ( t ) \mathbf { \bar { P } } _ { i } ( t ) \mathbf { \Psi } -$ $\alpha _ { i } ( t - 1 ) P _ { i } ( t - 1 ) - M _ { i } ( t )$ . This mechanism ensures that $q _ { i } ( t )$ , representing regret, continues to escalate if marketplace $i$ fails to draw more DOs despite increased costs incurred in time period $t$ .   
• Conversely, $q _ { i } ( t )$ diminishes by $\Delta M _ { i } ( t )$ when $\Delta M _ { i } ( t ) \geq 0$ .

The queuing dynamics in Eq. (5) elucidate that when AFL marketplace $i$ allocates a large proportion of revenue to DOs $( \alpha _ { i } ( t ) )$ yet transacts few FL model training tasks, i.e., attracting a limited number of new DOs, the marketplace’s regret queue grows.

Moving forward, we address the two primary objectives of the proposed ARAS-AFL. Firstly, it aims to assist each AFL marketplace $i$ in dynamically optimizing its revenuesharing strategy $\alpha _ { i } ( t )$ . This optimization ensures that all AFL marketplaces maintain a relatively stable regret value within the competitive ecosystem. By mitigating fluctuations in regret, the ecosystem remains resilient, averting undue volatility. To achieve this, we redefine the time-coupling constraint as a new objective, imposing an upper limit on queue growth to prevent unbounded expansion. Subsequently, we employ the Lyapunov optimization technique (Neely 2010) to fulfill this objective:

$$
\mathcal { L } ( \Theta ( t ) ) = \frac { 1 } { 2 } \sum _ { i \in \mathcal { N } } q _ { i } ^ { 2 } ( t ) \geq 0 .
$$

In assessing the anticipated growth of the Lyapunov function $\mathcal { L } ( \Theta ( t ) )$ , we define a Lyapunov drift as:

$$
\Delta ( \Theta ( t ) ) = \mathbb { E } [ \mathcal { L } ( \Theta ( t + 1 ) ) - \mathcal { L } ( \Theta ( t ) ) | \Theta ( t ) ] .
$$

It describes the anticipated fluctuation in the function over a single time slot, considering the current state, with expectations drawn from queue evolution statistics. According to the Lyapunov drift theorem, the minimization of $\Delta ( \Theta ( t ) { \bar { ) } }$ at each time $t$ ensures the stabilization of all queues.

# Optimal Decision Strategy

In addition to the goal of maintaining stability in regret among AFL marketplaces, the proposed ARAS-AFL also targets the enhancement of revenue for all involved marketplaces, ensuring sustainable long-term growth. The payoff $U _ { i } ( t )$ for AFL marketplace $i$ at time step $t$ is defined as:

$$
U _ { i } ( t ) = ( 1 - \alpha _ { i } ( t ) ) P _ { i } ( t ) .
$$

Then, we can formulate the total revenue $U ( t )$ generated by all AFL marketplaces at time step $t$ as follows:

$$
U ( t ) = \sum _ { i \in \mathcal { N } } [ ( 1 - \alpha _ { i } ( t ) ) P _ { i } ( t ) ] .
$$

Taking into account both the regret and profit dynamics of AFL marketplaces, devising revenue allocation strategies that cater to all involved DOs transforms into a dual objective optimization challenge. Consequently, we formalize the task of selecting a revenue allocation strategy for each AFL marketplace to maximize $\{ p r o f i t - d r i f t \}$ objective function:

$$
\begin{array}{c} \operatorname* { m a x } \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \{ \rho \mathbb { E } [ U ( t ) | \Theta ( t ) ] - \Delta ( \Theta ( t ) ) \}  \\ { s . t . \quad \quad 0 < \alpha _ { i } ( t ) ( i \in \mathcal { N } ) < b \leq 1 . \quad } \end{array}
$$

The constraint $0 < \alpha _ { i } ( t ) ( i \in \mathcal { N } ) < b \leq 1$ in Eq. (11) stipulates that, regardless of the circumstances, once a DO completes an $\mathrm { F L }$ model training task, the AFL marketplace must compensate the DO with a certain amount of remuneration. This constraint ensures that the payment ratio of the remuneration never exceeds the total reward for the FL model training task. The parameter $\rho$ serves as a weighting factor, allowing AFL marketplaces to express their relative preferences between regret and profit.

By substituting Eqs. (7), (8), and (10) into Eq. (11), we obtain:

$$
\begin{array} { l } { { \displaystyle \operatorname* { m a x } \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \sum _ { i \in \mathcal { N } } \{ - \frac { 1 } { 2 } P _ { i } ^ { 2 } ( t ) \alpha _ { i } ^ { 2 } ( t ) + [ P _ { i } ( t ) P _ { i } ( t - 1 ) \alpha _ { i } ( t - 1 ) } } \\ { { \displaystyle ~ + ~ P _ { i } ( t ) \Delta M _ { i } ( t ) - \rho P _ { i } ( t ) - q _ { i } ( t ) P _ { i } ( t ) ] \alpha _ { i } ( t ) } } \\ { { \displaystyle ~ + q _ { i } ( t ) \alpha _ { i } ( t - 1 ) P _ { i } ( t - 1 ) + \rho P _ { i } ( t ) + q _ { i } ( t ) \Delta M _ { i } ( t ) } } \\ { { \displaystyle ~ - \frac { 1 } { 2 } [ \alpha _ { i } ( t - 1 ) P _ { i } ( t - 1 ) + \Delta M _ { i } ( t ) ] ^ { 2 } \} . } } \end{array}
$$

Directing our attention to the components reliant on the decision variable $\alpha _ { i } ( t )$ , we reinterpret Eq. (12) as follows:

$$
\begin{array} { r l r } & { } & { \operatorname* { m a x } \frac { 1 } { T } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \sum _ { i \in \mathcal { N } } \{ - \frac { 1 } { 2 } P _ { i } ^ { 2 } ( t ) \alpha _ { i } ^ { 2 } ( t ) + [ P _ { i } ( t ) P _ { i } ( t - 1 ) \alpha _ { i } ( t - 1 ) } \\ & { } & { + \ P _ { i } ( t ) \Delta M _ { i } ( t ) - \rho P _ { i } ( t ) - q _ { i } ( t ) P _ { i } ( t ) ] \alpha _ { i } ( t ) \} . \quad } \end{array}
$$

To make it simple, we set $F$ as:

$$
\begin{array} { l } { \displaystyle { F = - \frac { 1 } { 2 } P _ { i } ^ { 2 } ( t ) \alpha _ { i } ^ { 2 } ( t ) + [ P _ { i } ( t ) P _ { i } ( t - 1 ) \alpha _ { i } ( t - 1 ) } } \\ { \displaystyle { ~ + P _ { i } ( t ) \Delta M _ { i } ( t ) - \rho P _ { i } ( t ) - q _ { i } ( t ) P _ { i } ( t ) ] \alpha _ { i } ( t ) . } } \end{array}
$$

By computing the first-order partial derivative of $F$ with respect to $\alpha _ { i } ( t )$ and setting it equal to zero, denoted as ∂α∂iF(t) = 0, we derive the optimal revenue allocation strategy $\alpha _ { i } ^ { * } ( t )$ for AFL marketplace $i$ at time step $t$ as follows:

$$
\alpha _ { i } ^ { * } ( t ) = \frac { P _ { i } ( t - 1 ) \alpha _ { i } ( t - 1 ) + \Delta M _ { i } ( t ) - \rho - q _ { i } ( t ) } { P _ { i } ( t ) } .
$$

# Experimental Evaluation

# Experiment Setup

Datasets Our experiments are based on six commonly adopted datasets in FL: MNIST1, CIFAR- $1 0 ^ { 2 }$ , FashionMNIST (FMNIST) (Xiao, Rasul, and Vollgraf 2017),

EMNIST-digits (EMNISTD) / letters (EMNISTL) (Cohen et al. 2017), Kuzushiji-MNIST (KMNIST) (Clanuwat et al. 2018). The FL models used are identical to those employed in (Tang and Yu 2023d).

Experimental scenarios As there is a lack of real-time FL training tasks releasing data shared across multiple AFL marketplaces, we consider the following two scenarios:

• FL training task overrelease market. In this scenario, the number of FL training tasks released by DCs in the AFL ecosystem exceeds the number of available DOs. We set up the AFL ecosystem with a total of $6 0 0 ~ \mathrm { D O s }$ , and the DCs on the ecosystem released 2,000 FL training tasks simultaneously. All the AFL marketplaces in the ecosystem competed for the $6 0 0 \mathrm { D O s }$ . • FL training task underrelease market. In this scenario, at each time step $t$ , the number of available DOs in the ecosystem was greater than the number of FL training tasks released by DCs. Specifically, we set the number of available DOs to 1,000 in the AFL ecosystem, and the DCs released ${ 8 0 } \mathrm { F L }$ training tasks.

Under either of these two experimental scenarios, we generated 400,000 FL training tasks with payments following a normal distribution with a mean of 80 and a standard deviation of 10, i.e., $\mathcal { N } ( 8 0 , 1 0 ^ { 2 } )$ . We released the FL training tasks in batches over time. During our experiments, we set the entire process of each FL training task, from release to completion and subsequent evaluation by DOs on the corresponding AFL marketplace, within a time period $t$ .

We set the confidence degree $\gamma$ in Equation (3) to 0.5 for each DO and the weighting factor $\rho$ in Equation (11) to 0.1. Additionally, we set $0 . 5 > \alpha _ { i } ( t ) \geq 0 . 3$ to ensure that the basic cost of DOs and the basic operating cost of the AFL marketplace were covered.

Comparison methods To our best knowledge, no existing decision support method has been designed specifically for auctioneers in the AFL ecosystem. Therefore, we selected the following two commonly used strategies as benchmarks:

• Random revenue allocation (Rand): At each time step, each auctioneer randomly generates a revenue allocation ratio $\alpha _ { i } ( t )$ for the DOs.   
• Greedy revenue allocation (Greedy). To ensure that the AFL marketplace retains as much revenue as possible from each completed FL training task released by DCs, the auctioneer adopts the smallest possible task commission allocation ratio $\alpha _ { i } ( t )$ for DOs.

Evaluation metrics We evaluate the proposed method and the benchmark strategies based on the following metrics:

• Averaged total revenue $( \mathbf { R e v } )$ . Since one of the objectives of AFL marketplaces is to maximize revenue, we adopt the averaged total revenue over time for each AFL marketplace as an evaluation metric. • Averaged number of DOs (#DOs). AFL marketplaces aim to attract and retain more participants, especially DOs, by allocating revenue to them. Therefore, we use the averaged number of recruited DOs as a metric to evaluate the effectiveness of the proposed method.

• Public reputation (PR). This metric evaluates the effectiveness of all the compared methods by recording the averaged reputation of each AFL marketplace. • Accuracy of the FL models (Acc). To ensure the longterm sustainability and fairness of the AFL ecosystem, any integrated method should balance the interests of all stakeholders, including DCs. Consequently, we evaluate the effectiveness of the auctioneer decision support strategies by assessing the test accuracy of the FL models obtained by DCs within the AFL ecosystem. This metric serves as an indicator of whether the proposed methods foster an environment that supports DCs in achieving high-quality FL model performance, thereby aligning with their interests and promoting the overall development of the ecosystem.

# Results and Discussion

In this section, we conduct an in-depth analysis of the performance of the comparison approaches concerning four crucial evaluation metrics: total revenue, number of DOs, public reputation, and FL model accuracy. Each experiment setting was rigorously executed five times to ensure statistical robustness. The averaged results are meticulously presented in Tables 1 and 2, complemented by visually illustrative representations in Figures 1 and 2.

Table 1 unveils the results of the comparison methods under the overrelease scenario. It is evident that across all six datasets, our proposed method, ARAS-AFL, consistently surpasses all baseline methods concerning all four evaluation metrics. Notably, when compared to the best performance achieved by the baselines, ARAS-AFL demonstrates remarkable improvements of $6 4 . 8 0 \%$ , $1 5 0 . 7 1 \%$ , $1 3 . 6 3 \%$ , and $4 . 3 4 \%$ in terms of the averaged total revenue, the averaged number of DOs, the public reputation, and the FL model accuracy, respectively.

Figure 1 vividly depicts the corresponding results during the 200-epoch experiment, at which point the number of DOs on each AFL marketplace has nearly stabilized. The results align seamlessly with the findings presented in Table 1, substantiating the efficacy of our proposed approach.

Furthermore, the comparative results under the underrelease scenario are comprehensively detailed in Table 2 and Figure 2. It is noteworthy that under this scenario, ARAS-AFL demonstrates substantial improvements of $3 3 . 3 2 \%$ , $4 6 . 6 7 \%$ , $6 9 . 9 8 \%$ , and $5 . 2 0 \%$ in the averaged total revenue, the averaged number of DOs, the public reputation, and the FL model accuracy, respectively.

It is worth highlighting that, as depicted in the last subfigures of Figures 1 and 2, the public reputation obtained by the Rand method experiences less fluctuation during the experiments. Additionally, the public reputation achieved by the proposed method, ARAS-AFL, exhibits a decreasing trend over the course of the experiments. This phenomenon could be attributed to the fact that as the AFL marketplace attracts more DOs, there may not be enough high-payoff FL training tasks from DCs, resulting in some DOs being able to obtain FL training tasks with lower payoffs. Consequently, this may lead to an unsatisfactory evaluation of the corresponding AFL marketplace by the DOs. Despite this, the proposed

<html><body><table><tr><td rowspan="2">Dataset</td><td colspan="3">Averaged total revenue</td><td colspan="3">Averaged number ofDOs</td><td colspan="3">Public reputation</td><td colspan="3">Accuracy of the FL models (%)</td></tr><tr><td>ARAS-AFL</td><td>Rand</td><td>Greedy</td><td>ARAS-AFL</td><td>Rand</td><td>Greedy</td><td>ARAS-AFL</td><td>Rand</td><td>Greedy</td><td>ARAS-AFL</td><td>Rand</td><td>Greedy</td></tr><tr><td>MNIST</td><td>9.41</td><td>5.73</td><td>1.40</td><td>414</td><td>158</td><td>31</td><td>0.482</td><td>0.424</td><td>0.281</td><td>84.82</td><td>81.45</td><td>83.16</td></tr><tr><td>CIFAR</td><td>8.31</td><td>5.05</td><td>1.31</td><td>416</td><td>171</td><td>46</td><td>0.485</td><td>0.427</td><td>0.282</td><td>42.37</td><td>40.03</td><td>40.84</td></tr><tr><td>FMNIST</td><td>9.29</td><td>5.65</td><td>1.40</td><td>415</td><td>160</td><td>31</td><td>0.482</td><td>0.424</td><td>0.281</td><td>75.10</td><td>73.34</td><td>73.58</td></tr><tr><td>EMNISTD</td><td>9.13</td><td>5.57</td><td>1.39</td><td>413</td><td>159</td><td>31</td><td>0.482</td><td>0.425</td><td>0.281</td><td>82.95</td><td>78.21</td><td>77.68</td></tr><tr><td>EMNISTL</td><td>9.39</td><td>5.57</td><td>2.24</td><td>369</td><td>156</td><td>41</td><td>0.517</td><td>0.445</td><td>0.297</td><td>74.56</td><td>67.44</td><td>68.93</td></tr><tr><td>KMNIST</td><td>9.28</td><td>5.69</td><td>1.88</td><td>373</td><td>154</td><td>39</td><td>0.497</td><td>0.446</td><td>0.281</td><td>73.14</td><td>70.31</td><td>68.93</td></tr></table></body></html>

Table 1: Performance comparison under the overrelease scenario.   

<html><body><table><tr><td rowspan="2">Dataset</td><td colspan="3">ARAS-AFL Randv Gredy</td><td colspan="3">A Ave-Ard nuRber of DOsdy</td><td colspan="3">ARAS PFLic Rand Greedy</td><td colspan="3">AcAus-Ayf hFLmdels (%)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MNIST</td><td>2.22</td><td>1.55</td><td>1.64</td><td>424</td><td>286</td><td>219</td><td>0.491</td><td>0.462</td><td>0.468</td><td>85.21</td><td>80.93</td><td>83.79</td></tr><tr><td>CIFAR</td><td>2.10</td><td>1.50</td><td>1.53</td><td>434</td><td>304</td><td>238</td><td>0.518</td><td>0.489</td><td>0.492</td><td>43.18</td><td>40.79</td><td>41.27</td></tr><tr><td>FMNIST</td><td>2.20</td><td>1.54</td><td>1.62</td><td>421</td><td>287</td><td>219</td><td>0.490</td><td>0.463</td><td>0.468</td><td>75.76</td><td>72.89</td><td>74.04</td></tr><tr><td>EMNISTD</td><td>2.18</td><td>1.53</td><td>1.60</td><td>421</td><td>283</td><td>217</td><td>0.490</td><td>0.460</td><td>0.466</td><td>84.52</td><td>79.72</td><td>79.63</td></tr><tr><td>EMNISTL</td><td>2.40</td><td>1.68</td><td>2.07</td><td>391</td><td>282</td><td>246</td><td>0.459</td><td>0.409</td><td>0.414</td><td>76.11</td><td>68.38</td><td>69.52</td></tr><tr><td>KMNIST</td><td>2.42</td><td>1.74</td><td>1.72</td><td>407</td><td>263</td><td>227</td><td>0.467</td><td>0.419</td><td>0.421</td><td>73.17</td><td>68.36</td><td>67.88</td></tr></table></body></html>

![](images/64d4500746cc0df7eb82ae0f51c5e29790eb0ade47a77c62488243183df58a77.jpg)  
Table 2: Performance comparison under the underrelease scenario.

Figure 2: Comparison of different method under the underrelease senario.

ARAS-AFL still enables the AFL marketplace to maintain higher revenues and attract more DOs, as evidenced by the comprehensive results presented in Tables 1 and 2, as well as Figures 1 and 2.

# Conclusions

This paper frames the issues in the AFL ecosystem from the perspective of auctioneers and investigates decision support strategies for them. We propose a novel method, termed ARAS-AFL, to assist auctioneers in attracting more Os within a competitive AFL environment while ensuring their profitability. ARAS-AFL approaches the challenges of attractiveness and competitiveness through the lens of reputation. By leveraging Lyapunov optimization techniques, ARAS-AFL enables each individual AFL marketplace to achieve the dual objective of balancing reputation management costs and profitability by dynamically adjusting its revenue allocation strategy within the reputation management framework. The decision-making process considers both the marketplace’s revenue and the changes in the number of participants, allowing for informed and adaptive strategies. To the best of our knowledge, ARAS-AFL is the first approach designed specifically to assist auctioneers in making decisions to address external competitiveness challenges while ensuring the long-term sustainability of their marketplace.

# Acknowledgments

This research is supported by the RIE2025 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) (Award I2301E0026), administered by A\*STAR, as well as supported by Alibaba Group and NTU Singapore through Alibaba-NTU Global e-Sustainability CorpLab (ANGEL); and National Research Foundation, Singapore and DSO National Laboratories under the AI Singapore Programme (AISG Award No: AISG2-RP-2020-019).