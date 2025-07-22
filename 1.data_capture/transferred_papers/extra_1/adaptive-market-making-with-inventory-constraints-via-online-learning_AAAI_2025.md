# Adaptive Market Making with Inventory Constraints via Online Learning

Shan Xue1, Ye $ { \mathbf { D } }  { \mathbf { u } } ^ { 2 * }$ , Liang $\mathbf { X } \mathbf { u } ^ { 3 }$

1 School of Economics and Management, Leshan Normal University, Leshan, China 2 Southwestern University of Finance and Economics, Chengdu, China 3 School of Business Administration, Southwestern University of Finance and Economics, Chengdu, China xueshanads $1 2 3 @$ gmail.com, henry,duye $@$ gmail.com, arecxuliang $1 1 1 2 @$ gmail.com

# Abstract

A market maker is a specialist who provides liquidity by continuously offering bid and ask quotes for a financial asset. The market maker’s objective is to maximize profit while avoiding the accumulation of a large position in the asset to control inventory risk. To achieve model-free results, online learning has been applied to design market-making strategies that make no assumptions on the dynamics of the limit order book and asset price. However, existing work primarily focuses on profit rather than inventory risk. To address this limitation, this paper develops market-making strategies with inventory constraints within the online learning framework. To manage inventory risk, we propose two classes of marketmaking strategies with fixed bid-ask spreads that serve as reference strategies. Each reference strategy can ensure that the inventory remains under control, which enables the online learning algorithms designed for each class of reference strategies to satisfy inventory constraints. Different from the standard online learning model where the gain in each period is assumed to lie within a fixed bounded interval, the gain in our model depends on a state variable (i.e., the inventory size). Thus, a key challenge in analyzing the regret bounds is to bound the difference between the gains of any two reference strategies, which becomes significantly more complicated compared with scenarios without inventory constraints. By tackling these difficulties, we show that these algorithms achieve low regrets. Experimental results illustrate the superior performance of our algorithms in inventory risk control.

Extended version — https://papers.ssrn.com/abs $\ v =$ 5110826

# 1 Introduction

Market making is a financial activity where an individual or firm, known as a market maker, provides liquidity by continuously offering bid and ask quotes for a financial asset. The market maker stands ready to buy the asset at the bid price and sell it at the ask price, with the difference between the two prices known as the bid-ask spread. By doing so, the market maker provides liquidity for other traders while profiting from the simple principle of buying low and selling high. However, this profit does not come without risk. In the presence of a strong price trend, such as rapidly rising prices, the market maker may accumulate a large number of short positions. If these positions are liquidated at the peak of the trend, significant losses may be incurred. This risk is referred to as inventory risk (Avellaneda and Stoikov 2008; Guilbaud and Pham 2013). Thus, the objective of a market maker is to maximize the profit and loss (PnL) of the trading while keeping her inventory low to minimize inventory risk.

To determine the optimal market-making strategy, there are three primary approaches in the existing literature. The first approach models the problem as a stochastic optimal control problem, which is then solved using the wellknown Hamilton-Jacobi-Bellman equation (Avellaneda and Stoikov 2008; Gu´eant, Lehalle, and Fernandez-Tapia 2013; Obizhaeva and Wang 2013; Cartea, Jaimungal, and Penalva 2015). This method often involves making assumptions on the dynamics of the limit order book (LOB) and asset price, such as Brownian motion. The second approach frames it as a reinforcement learning (RL) problem (Patel 2018; Spooner et al. 2018; Zhang and Chen 2020), whose recent progress is comprehensively reviewed by (Hambly, Xu, and Yang 2021). Spooner and Savani (2020) develop adversarial RL based on the model of (Avellaneda and Stoikov 2008), and produce market-marking agents that are robust to adversarial and adaptively-chosen market conditions. Although their strategies are more robust to misspecification, they still rely on assumptions on the dynamics of the asset price and LOB. In contrast to the first two approaches, the third approach explores a model-free or robust version of market making, without imposing any assumptions on the dynamics of the LOB and asset price. The online learning or no-regret learning framework is well-suited to this scenario. Abernethy and Kale (2013) are the pioneer in this line of work. They develop a class of spread-based market-making strategies parametrized by a minimum quoted spread. Online learning algorithms are then designed for these strategies to achieve no-regret. However, rather than on inventory risk, their primary focus is on the PnL of these algorithms. Their algorithms will be exposed to significant inventory risk if the asset price rises or falls consecutively.

To address the above limitations, we focus on both PnL and inventory risk for the market maker within the framework of online learning. We extend the work of (Abernethy and Kale 2013) and provide model-free market-making strategies by incorporating inventory risk control. To manage inventory risk, one naive approach is to impose a prespecified limit on the maximum absolute value of the inventory that the market maker is allowed to hold. We call it the hard-constraint approach. The market maker meets this constraint by limiting the price levels at which she places orders in the order book. Another approach, which is more subtle and relevant to finance literature, is to choose the bid and ask prices negatively proportional to the current inventory level. For instance, if the market marker currently holds a large short position, she may raise her bid prices to buy back some of the short position. This is referred to as the soft-constraint approach. Based on the two approaches, two classes of market-making strategies with fixed bid-ask spread: hard-constraint and soft-constraint strategies are proposed as reference strategies. We then design online learning algorithms for each class of reference strategies to develop adaptive market-making strategies.

Next, we present our main result informally in the following theorem. Different from the standard online learning model, where the gain in each period is assumed to lie within a fixed bounded interval, the gain in our model depends on a state variable (i.e., the inventory size). Since the inventory size is influenced by specific parameters of the reference strategies as well as the asset price path, the regret bound of adaptive market-making strategies is necessarily a function of these parameters.

Theorem 1. (Informal) The adaptive market-making strategies have regrets bounded by $c \sqrt { T \ln { N } }$ after $T$ periods to the best of $N$ constraint strategies, where c is a constant depending on specific parameters of the strategies (see Theorem 10-13 for details).

# 1.1 Contributions

Our contributions are summarized as follows:

i) To the best of our knowledge, we are the first to investigate inventory risk control of market making within the context of online learning. To accommodate the inventory risk, we introduce a new model. Its basic idea is to develop two types of reference strategies: hard-constraint and softconstraint strategies. We demonstrate that, for any given path of the asset price, the inventory for both reference strategies can be bounded by a constant in each period. This feature naturally enables adaptive (dynamic) strategies to effectively control inventory risk.

ii) Unlike the standard online learning model, in which the gain (loss) in each period is assumed to lie within a fixed bounded interval, such as [0,1], the upper and lower bounds of the gain in our model depend on a state variable (i.e., the inventory size), which is path-dependent and time-varying. After accommodating inventory constraints, we make nontrivial efforts to bound the difference between the inventories (Lemma 4, 6 and Corollary 7) and further the difference between the gains (Lemma 5 and 8) of any two reference strategies. These results lead to the no-regret properties of our adaptive market-making strategies.

iii) We run comprehensive experiments on the tick-bytick data of all component stocks of the China CSI500 index over a one-month period, which includes over 10,882 stock price paths. The experimental results show that our adaptive strategies could indeed achieve no-regret. Meanwhile, compared with market-making strategies without inventory constraints, our adaptive strategies significantly improve inventory risk control, especially in markets with strong upward or downward trends.

# 1.2 Related Literature

Perhaps the most famous model of market making in finance is the Glosten-Milgrom model (Glosten and Milgrom 1985), which investigates the market-making problem in a market with asymmetric information. Subsequent studies have sought to to explore the optimal behavior under various market dynamics settings. Among them, the work of (Avellaneda and Stoikov 2008) is well known and widely used in quantitative finance. Based on the assumption that the market maker with an exponential utility function has perfect knowledge of the market dynamics, they provided a closed-form solution for the optimal market-making strategy. The follow-up work derived the optimal solution for some other utility functions or price processes (Fodra and Labadie 2012; Gue´ant, Lehalle, and Fernandez-Tapia 2013; Cartea, Jaimungal, and Penalva 2015; Gue´ant 2017). Cartea, Donnelly, and Jaimungal (2017) considered the impact of model misspecification in the model of (Avellaneda and Stoikov 2008), and provided an analytical solution for the robust optimal strategies.

Within the AI community, a significant body of literature has studied the market-making problem with the RL approach. Chan and Shelton (2001) were the first to apply RL to market making, which developed explicit marketmaking strategies and tested them under a simulated environment. Later on, Spooner et al. (2018) generalized the results in (Chan and Shelton 2001) and designed a temporaldifference RL algorithm to improve the performance of market making. Some recent work has paid attention to improving the robustness of market-making strategies to model uncertainty (Spooner and Savani 2020; Gasˇperov and Kostanjcˇar 2021). Furthermore, RL algorithms can also be applied to high-dimensional, multi-asset market making (Gue´ant and Manziuk 2019; Baldacci et al. 2019), and multi-agent with different competitive scenarios (Patel 2018; Ganesh et al. 2019).

A substantial body of literature has explored online learning with constraints. Ding et al. (2013) investigated multiarmed bandit problems with random costs subject to a budget constraint and developed two algorithms for this setting. Mannor, Tsitsiklis, and Yu (2009) considered an online learning setting where a decision maker aims to maximize her average reward while ensuring that the average penalty adheres to a specified constraint. In the work of (Paternain et al. 2020), a constrained online optimization problem in networks was examined, where the constraints can vary arbitrarily over time. However, the constraint specifications in these models do not align with the framework employed in our study.

# 2 The Model

# 2.1 The Market Trading Framework

Following the work of (Chakraborty and Kearns 2011; Abernethy and Kale 2013), we consider a discrete-time trading model with $T$ periods, where a period is indexed by $t \ \overset { \cdot } { \in } \ \{ 0 , 1 , 2 , \cdot \cdot \cdot , T \}$ . There is a stock in the market. Denote by $P _ { t }$ the market price of the stock at the end of period $t$ . Let $\delta$ be the minimum price variation or tick size, and $M$ be some reasonable upper bound on the stock price. Thus, the set of all possible stock prices can be denoted by $\mathbb { P } = \{ \delta , 2 \delta , \cdots , M \}$ . Assume that $\lceil P _ { t } - P _ { t - 1 } \rceil \leq \nabla$ for all $t$ , where $\nabla$ is a given sufficiently large constant.

A market maker exists in the market who interacts with a continuous double auction via an order book. The market maker focuses on both $\mathrm { P n L }$ and inventory risk. She trades the stock over $T$ periods by submitting both limit and market orders to the order book. At the end of each period $t$ , her trading strategy submits a limit order schedule $Q _ { t } : \mathbb { P }  \mathbb { R }$ . For each $P \in \mathbb { P }$ , the absolute value of $Q _ { t } ( P )$ represents the number of shares she intends to buy ( if $Q _ { t } ( P ) > 0$ ) or sell ( if $Q _ { t } ( P ) < 0$ ) at price $P$ . In period $t + 1$ , all limit buy orders offered at prices no less than $P _ { t + 1 }$ and all limit sell orders offered at prices no greater than $P _ { t + 1 }$ from period $t$ are assumed to be executed. Furthermore, the strategy may also actively trade the stock to adjust its stock inventory level (i.e., the amount of the stock it holds) via a market order at the end of period $t$ , which only specifies the amount of the stock it is willing to buy or sell. Unlike a limit order, a market order is assumed to be executed immediately at the current price $P _ { t }$ . Denote by $H _ { t }$ and $C _ { t }$ the amount of the stock inventory and cash the strategy holds at the end of period $t$ , respectively. Initially, we set $H _ { 0 } = C _ { 0 } = 0$ . If $H _ { t }$ is positive (resp., negative), the strategy holds a long (resp., short) position in the stock. Let $V _ { t } = H _ { t } P _ { t } + C _ { t }$ , which is the total value of the strategy’s holdings at the end of period $t$ . The gain of the strategy in period $t$ is naturally defined as $\Delta V _ { t } = V _ { t } - V _ { t - 1 }$ .

Assume that there are $N$ reference strategies, the market maker constructs her trading strategy by viewing every reference strategy as an expert and running an online learning algorithm for learning with expert advice. This strategy is referred to as the adaptive strategy hereafter. Following the work of (Chakraborty and Kearns 2011; Abernethy and Kale 2013), we make some assumptions on the trading mechanism as follows:

• Neither transaction nor borrowing costs exist.   
• The stock is perfectly divisible, and the market maker may purchase and sell fractional shares of the stock.   
• The stock price is exogenously determined, which means that the market maker’s trades do not affect the stock price.   
• The limit orders submitted by the market maker at the end of each period $t$ will be canceled at the end of the next period if unexecuted.

Although the assumptions outlined above are also present in the work of (Chakraborty and Kearns 2011; Abernethy and Kale 2013), some of them are relatively rigid and unrealistic, particularly the first and third assumptions. In real markets, transaction costs can affect both the market maker’s profit and the bid-ask spread. To reduce the frequency of costly trades, the market maker may widen the spread. Furthermore, stock prices are not always determined exogenously, especially when the market maker provides a significant portion of market liquidity. We will relax these assumptions in future work.

# 2.2 Reference Strategies

To manage inventory risk, we propose two classes of marketmaking strategies with fixed bid-ask spreads that serve as reference strategies. The first class, known as hardconstraint strategies, ensures that the inventory level at any time remains within a pre-specified interval. While the second class, known as soft-constraint strategies, mitigates inventory risk by adjusting quoted bid and ask prices in a manner negatively correlated with the current inventory level.

Hard-Constraint Strategies To achieve inventory control, one approach is to strictly limit the absolute inventory in each period $t$ , (i.e., $| H _ { t } | )$ , to no more than some predetermined level $R$ . Our hard-constraint strategies build upon the work of (Abernethy and Kale 2013). Different from their strategies, our hard-constraint strategies restrict the lowest quoted price for limit buy orders and the highest quoted price for limit sell orders in terms of their inventory position to ensure $| H _ { t } | \leq \ R$ for all $t$ . Specifically, consider a class of hard-constraint strategies parameterized by a window size $b \in \{ \delta , 2 \delta , \cdot \cdot \cdot , \nabla \}$ . For a given hard-constraint strategy $S ( b )$ , let $H _ { t } ( b ) , C _ { t } ( b )$ , and $V _ { t } ( b )$ represent its inventory, cash, and total value, respectively. At the end of period $t$ , the strategy $S ( b )$ chooses a window of size $b$ , denoted as $[ a _ { t } ( b ) , a _ { t } ( b ) + b ]$ , where $a _ { 0 } ( b ) = P _ { 0 }$ and $a _ { t } ( b )$ for $t \geq 1$ is determined by the following rules:

$$
a _ { t } ( b ) = \left\{ \begin{array} { l l } { P _ { t } - b } & { \mathrm { ~ i f ~ } P _ { t } > a _ { t - 1 } ( b ) + b } \\ { a _ { t - 1 } ( b ) } & { \mathrm { ~ i f ~ } P _ { t } \in [ a _ { t - 1 } ( b ) , a _ { t - 1 } ( b ) + b ] } \\ { P _ { t } } & { \mathrm { ~ i f ~ } P _ { t } < a _ { t - 1 } ( b ) . } \end{array} \right.
$$

It then submits a limit buy order of one share at each price $P \in [ \operatorname* { m a x } \{ a _ { t } ( b ) - \delta ( R - H _ { t } ( b ) ) , \delta \} , a _ { t } )$ and a limit sell order of one share at each price $P \in ( \dot { a } _ { t } ( b ) + b , \operatorname* { m i n } \{ a _ { t } ( b ) +$ $b + \delta ( H _ { t } ( b ) + R ) , M \} ]$ . In this way, the strategy will buy no more than $R - H _ { t } ( b )$ shares or sell no more than $R + H _ { t } ( b )$ shares in period $t + 1$ , which ensures that $| H _ { t + 1 } | \leq R$ for all $t$ .

Specially, if $R = + \infty$ , our hard-constraint strategies degenerate into those presented in (Abernethy and Kale 2013), which we will refer to as non-constraint strategies hereafter.

Soft-Constraint Strategies In addition to imposing strict constraints on the inventory level, another approach for the market maker to controlling inventory risk is to dynamically adjust the quoted bid and ask prices in terms of the current inventory $H _ { t }$ . In the absence of consideration of the trade size and failure conditions, inventory risk should affect bid and ask prices, but not the size of the bid-ask spread

Algorithm 1: Hard-constraint strategy $S ( b )$

Input: The window size $b$ and the initial stock price $P _ { 0 }$ .

1: Initialize $a _ { 0 } ( b ) \quad : = \quad P _ { 0 }$ and $H _ { 0 } ( b ) \ : = \ 0$ . Submit limit order $Q _ { 0 }$ : $Q _ { 0 } ( P ) \ = \ 1$ if $\bar { P } ~ \in ~ [ \operatorname* { m a x } \{ a _ { 0 } ( b ) ~ -$ $\delta R , \delta \} , a _ { 0 } ( b ) )$ , $Q _ { 0 } ( P ) = - 1$ if $P \in ( a _ { 0 } + b , \operatorname* { m i n } \{ a _ { 0 } +$ $b + \delta R , M \} ]$ , and $Q _ { 0 } ( P ) = 0$ otherwise.   
2: for $t = 1 , 2 , \cdots , T$ do   
3: Execute any limit orders from the previous period and observe the stock price $P _ { t }$ . The inventory position changes from $H _ { t - 1 } ( b )$ to $H _ { t } ( b )$ .   
4: if $P _ { t } > a _ { t - 1 } ( b ) + b$ then   
5: $a _ { t } ( b )  P _ { t } - b$   
6: else if $P _ { t } < a _ { t - 1 } ( b )$ then   
7: $a _ { t } ( b )  P _ { t }$   
8: else   
9: $a _ { t } ( b ) \gets a _ { t - 1 } ( b )$   
10: end if   
11: Submit limit order $Q _ { t }$ : $\begin{array} { r l r } { Q _ { t } ( P ) } & { { } = } & { 1 } \end{array}$ if $\textit { P } \in$ $[ \mathrm { m a x } \{ a _ { t } ( b ) - \delta ( R - H _ { t } ( b ) ) , \delta \} , \dot { a } _ { t } ( b$ )), $Q _ { t } ( P ) = - 1$ if $P \in ( a _ { t } ( b ) + b , \operatorname* { m i n } \{ a _ { t } ( b ) + b + \delta ( H _ { t } ( b ) + R ) , M \} ]$ , and $Q _ { t } ( P ) = 0$ otherwise.

(Amihud and Mendelson 1980; Stoll 1978; Grossman and Miller 1988). If the market maker has a long position in the stock, minimizing inventory risk is achieved by lowering both bid and ask prices. Contrarily, if she has a short position, inventory is controlled by raising both bid and ask prices. Therefore, building on the work of (Das 2005), we develop a class of soft-constraint strategies that adjust the bid and ask prices linearly based on the current inventory. Specifically, consider a class of soft-constraint strategies parameterized by a window size $b \in \{ \delta , 2 \delta , \cdot \cdot \cdot , \nabla \}$ . At the end of period $t$ , the strategy $S ( b )$ selects a window of size $b$ (i.e., $[ a _ { t } ( b ) , a _ { t } ( b ) + b ] )$ , where $\dot { a _ { t } } ( b ) = P _ { t } - b - \gamma H _ { t } ( b )$ and $\gamma$ is a nonnegative parameter representing a risk-aversion coefficient. It then submits a limit buy order of one share at every price $P \in [ \delta , a _ { t } ( b ) )$ and a limit sell order of one share at every price $\mathsf { \bar { P } } \in \mathsf { \bar { \Pi } } ( \bar { a _ { t } } ( b ) + b , M ]$ . Note that when $\gamma H _ { t } ( b )$ is not a multiple of $\delta$ , it is rounded to the nearest multiple of $\delta$ toward $+ \infty$ (resp., $- \infty )$ if $H _ { t } ( b ) > 0$ (resp., $H _ { t } ( b ) < 0 )$ and then used to compute $a _ { t } ( b )$ . For the case of $\gamma > \delta$ , if the stock price in period $t$ does not continue to move in the same direction as the previous period, all inventory will be liquidated at unfavorable prices. This inevitably results in significant losses in non-trend markets. Thus, we mainly focus on the case of $0 < \gamma \leq \delta$ hereafter.

# 2.3 The Regret of Adaptive Strategies

For each class of reference strategies, Denote by $\mathbb { B }$ the set of possible values of $b$ and $V _ { T } ^ { \mathcal { A } }$ the total value of the adaptive strategy’s holdings using algorithm $\mathcal { A }$ at time $T$ . The regret of an adaptive strategy using algorithm $\mathcal { A }$ at time $T$ is defined as the total value of the best reference strategies in hindsight minus that of the adaptive strategy. Formally,

$$
r e g ( \operatorname { A S } ) = \operatorname* { m a x } _ { b \in \mathbb { B } } V _ { T } ( b ) - V _ { T } ^ { A } .
$$

Input: The window size $b$ , risk-aversion coefficient $\gamma$ , and the initial price $P _ { 0 }$ .   
1: Initialize $a _ { 0 } ( b ) : = P _ { 0 } - b$ . Submit limit order $Q _ { 0 }$ : $Q _ { 0 } ( P ) = 1$ if $P \in [ \delta , a _ { 0 } ( b ) )$ , $Q _ { 0 } ( P ) = - 1$ if $P \in$ $( a _ { 0 } ( b ) + b , M ]$ , and $Q _ { 0 } ( P ) = 0$ otherwise.   
2: for $t = 1 , 2 , \cdots , T$ do   
3: Execute any limit orders from the previous period and observe the stock price $P _ { t }$ . The inventory position changes from $H _ { t - 1 } ( b )$ to $H _ { t } ( b )$ .   
4: Update $a _ { t } ( b ) \gets P _ { t } - b - \gamma H _ { t } ( b )$ .   
5: Submit limit order $Q _ { t } \colon Q _ { t } ( P ) = 1$ if $P \in [ \delta , a _ { t } ( b ) )$ , $Q _ { t } ( P ) = - 1$ if $P \in ( a _ { t } ( b ) + b , M ]$ , and $Q _ { t } ( P ) = 0$ otherwise.

6: end for

# 3 The Bounds of Gains

In the standard online learning model, the gain (or loss) in each period is typically assumed to lie within a fixed bounded interval. However, in our model, the upper and lower bounds of the gain depend on the inventory size, which is path-dependent and time-varying. In this section, we attempt to bound the difference between the inventories and further the difference between the gains of any two reference strategies within the same class. This constitutes our main technical contribution. Due to space limitations, all proofs are included in the appendix.

# 3.1 Hard-Constraint Strategies

Lemma 2. For any hard-constraint strategy $S ( b )$ , if its position at the end of period $t$ satisfies $| H _ { t } ( b ) | < R$ , then we have $\begin{array} { r } { H _ { t } ( b ) - H _ { t - 1 } \dot { ( } b ) = \frac { 1 } { \delta } [ a _ { t - 1 } \dot { ( } b ) - a _ { t } \dot { ( } b ) ] } \end{array}$ .

For any two hard-constraint strategies $S ( b ^ { 1 } )$ and $S ( b ^ { 2 } )$ , to avoid confusion, we will use the notations $H _ { t } ^ { i } , ~ a _ { t } ^ { i } ,$ $a _ { t } ^ { i } , \ V _ { t } ^ { i }$ , etc., to refer to $H _ { t } ( b ^ { i } ) , a _ { t } ( b ^ { i } ) , V _ { t } ( b ^ { i } )$ , etc., with $i = 1$ or 2, respectively. We have the following result.

Lemma 3. For any two hard-constraint strategies $S ( b ^ { 1 } )$ and $S ( b ^ { 2 } )$ with $b ^ { 1 } < { \dot { b } } ^ { 2 }$ , denote by $[ a _ { t } ^ { 1 } , a _ { t } ^ { 1 } + b ^ { 1 } ]$ and $[ \dot { a } _ { t } ^ { 2 } , \dot { a } _ { t } ^ { 2 } +$ $b ^ { 2 } ]$ the windows selected by $S ( b ^ { 1 } )$ and $S ( b ^ { 2 } )$ at the end of period $t$ , respectively. Then we have $[ a _ { t } ^ { 1 } , a _ { t } ^ { 1 } + b ^ { 1 } ] \subset [ a _ { t } ^ { 2 } , a _ { t } ^ { 2 } +$ $\bar { b } ^ { 2 } ]$ for all $t$ .

Why is it more complicated to bound $H _ { t } ^ { 1 } - H _ { t } ^ { 2 }$ for hardconstraint strategies? For non-constraint strategies presented in (Abernethy and Kale 2013) (i.e., hard-constraint strategies with $R = + \infty )$ , the condition in Lemma 2 is naturally satisfied. Thus, it is straightforward to prove that $\begin{array} { r } { H _ { t } ( b ) = \sum _ { j = 1 } ^ { t } H _ { j } ( b ) - H _ { j - 1 } ( b ) = \frac { 1 } { \delta } [ a _ { 0 } ( b ) - a _ { t } ( b ) ] } \end{array}$ by Lemma 2, and further that $\begin{array} { r } { H _ { t } ^ { 1 } \ - \ H _ { t } ^ { 2 } \ = \ \frac { 1 } { \delta } ( a _ { t } ^ { 2 } \ - \ a _ { t } ^ { 1 } ) \ \in \qquad } \end{array}$ $\big [ \frac { b ^ { 1 } - b ^ { 2 } } { \delta } , 0 \big ]$ by Lemma 3. However, the equation $H _ { t } ( b ) \ =$ $\textstyle { \frac { 1 } { \delta } } [ a _ { 0 } ( b ) - a _ { t } ( b ) ]$ does not always hold for hard-constraint strategies with $R < + \infty$ . The reason is that the inventory level may reach the limit of $\pm R$ before period $\mathfrak { t } ,$ thereby violating the condition in Lemma 2. This makes it more complicated to bound $H _ { t } ^ { 1 } - H _ { t } ^ { 2 }$ . To illustrate this, consider the following scenario with $P _ { 0 } ~ = ~ 1$ , $b ~ = ~ \delta ~ = ~ 0 . 5$ , and

Table 1: An example   

<html><body><table><tr><td rowspan="3">t</td><td rowspan="3">Pt</td><td rowspan="3">at(6)</td><td rowspan="3">[ao(b)-at(b)]</td><td colspan="2">Ht(b)</td></tr><tr><td>R=+∞</td><td>R=2</td></tr><tr><td>1</td><td>2</td><td>1.5</td><td>-1</td><td>-1</td><td>-1</td></tr><tr><td>2</td><td>3</td><td>2.5</td><td>-3</td><td>-3</td><td>-2</td></tr><tr><td>3</td><td>4</td><td>3.5</td><td>-5</td><td>-5</td><td>-2</td></tr></table></body></html>

$R = 2$ . As shown in Table 1, if the stock price increases by one in each period, we have $a _ { 0 } ( b ) = 1$ , $a _ { 3 } ( b ) = 3 . 5$ , and $\begin{array} { r } { H _ { 3 } ( b ) = - 2 \ne \frac { 1 } { \delta } [ a _ { 0 } ( b ) - a _ { 3 } ( b ) ] } \end{array}$ for the hard-constraint strategy with $R = 2$ . For general hard-constraint strategies, we have the following result.

Lemma 4. For any two hard-constraint strategies $S ( b ^ { 1 } )$ and $S ( b ^ { 2 } )$ with $b ^ { 1 } < b ^ { \dot { 2 } }$ , we have

$$
| H _ { t } ^ { 1 } - H _ { t } ^ { 2 } | \leq \frac 1 \delta ( b ^ { 2 } - b ^ { 1 } )
$$

for all $t$ .

Proof. [Sketch] Define $\begin{array} { r } { B S _ { t } ^ { i } = H _ { t } ^ { i } - H _ { t - 1 } ^ { i } - \frac { a _ { t - 1 } ^ { i } - a _ { t } ^ { i } } { \delta } } \end{array}$ with $i = 1$ or 2. For any two hard-constraint strategies $S ( b ^ { 1 } )$ and $S ( b ^ { 2 } )$ with $b ^ { 1 } < b ^ { 2 }$ , we first prove $\begin{array} { r } { 0 \leq \sum _ { j = 1 } ^ { t } B S _ { j } ^ { 1 } - B S _ { j } ^ { 2 } \leq } \end{array}$ ${ \textstyle { \frac { 1 } { \delta } } } ( b ^ { 2 } \mathrm { ~ - ~ } b ^ { 1 } )$ for all $t$ . It is easy to verify $H _ { t } ^ { 1 } \ - \ H _ { t } ^ { 2 } \ =$ $\begin{array} { r } { \dot { \sum } _ { j = 1 } ^ { t } ( H _ { j } ^ { 1 } - H _ { j - 1 } ^ { 1 } ) - \sum _ { j = 1 } ^ { t } ( H _ { j } ^ { 2 } - H _ { j - 1 } ^ { 2 } ) = \frac { 1 } { \delta } ( a _ { t } ^ { 2 } - a _ { t } ^ { 1 } ) + } \end{array}$ $\textstyle \sum _ { j = 1 } ^ { t } B S _ { j } ^ { 1 } - B S _ { j } ^ { 2 }$ by definition. Since $[ a _ { t } ^ { 1 } , a _ { t } ^ { 1 } + b ^ { 1 } ] \subset$ $[ a _ { t } ^ { 2 } , a _ { t } ^ { 2 } + b ^ { 2 } ]$ for all $t$ by Lemma 3, we have $b ^ { 1 } - b ^ { 2 } \leq$ $a _ { t } ^ { 2 } - a _ { t } ^ { 1 } \leq 0$ . It follows that $\begin{array} { r } { | H _ { t } ^ { 1 } - H _ { t } ^ { 2 } | \le \frac { 1 } { \delta } ( b ^ { 2 } - b ^ { 1 } ) } \end{array}$ for all $t$ .

We now consider $N$ hard-constraint strategies. Let $b ^ { m i n }$ and $b ^ { m a x }$ be the minimum and maximum values of all $\textit { b } \in \mathbb { B }$ . Note that $b ^ { m a x } \ \leq \ \nabla$ . With the assumption of $| P _ { t } - P _ { t - 1 } | \leq \nabla$ , we can use Lemma 4 to bound the difference between the gains of any two hard-constraint strategies in each period. We have the following result.

Lemma 5. Define Gh = ▽(bmax−bmin) $\begin{array} { r } { G ^ { h } = \frac { \nabla ( b ^ { \operatorname* { m a x } } - b ^ { \operatorname* { m i n } } ) } { \delta } + \operatorname* { m i n } ( 2 R \nabla , \frac { \nabla ^ { 2 } } { \delta } ) } \end{array}$ For any two hard-constraint strategies $S ( b ^ { 1 } )$ and $S ( b ^ { 2 } )$ with $b ^ { 1 } , b ^ { 2 } \in \mathbb { B }$ , we have

$$
| ( V _ { t } ^ { 1 } - V _ { t - 1 } ^ { 1 } ) - ( V _ { t } ^ { 2 } - V _ { t - 1 } ^ { 2 } ) | \leq 2 G ^ { h }
$$

for all $t$ .

# 3.2 Soft-Constraint Strategies

Lemma 6. For any soft-constraint strategy $S ( b ) , i f 0 < \gamma \leq$ $\delta$ , we have

$$
- \frac { \nabla } { \gamma } \leq H _ { t } ( b ) \leq \frac { \nabla - b } { \gamma }
$$

for all $t$ .

By Lemma 6, the following Corollary can be directly obtained.

Corollary 7. For any two soft-constraint strategies $S ( b ^ { 1 } )$ and $S ( b ^ { 2 } )$ with $b ^ { 1 } < \dot { b } ^ { 2 }$ , $i f 0 < \gamma \leq \delta$ , we have

$$
\bigl | H _ { t } ^ { 1 } - H _ { t } ^ { 2 } \bigr | \leq \frac { 2 \nabla - b ^ { 1 } } { \gamma }
$$

for all $t$ .

By Lemma 6 and Corollary 7, the difference between the gains of any two soft-constraint strategies in each period can be bounded. We thus have the following result.

Lemma 8. Define $\begin{array} { r } { G ^ { s } = \frac { ( 2 \nabla - b ^ { \mathrm { m i n } } ) ( 3 \nabla - b ^ { \mathrm { m i n } } ) } { \gamma } } \end{array}$ . I $f 0 < \gamma \leq \delta$ then for any two soft-constraint strategies $S ( b ^ { 1 } )$ and $S ( b ^ { 2 } )$ with $\mathring { b } ^ { 1 } , b ^ { 2 } \in \mathbb { B }$ , we have

$$
| ( V _ { t } ^ { 1 } - V _ { t - 1 } ^ { 1 } ) - ( V _ { t } ^ { 2 } - V _ { t - 1 } ^ { 2 } ) | \leq 2 G ^ { s }
$$

for all $t$ .

# 4 The Regret Bounds of Adaptive Strategies

Thus far, we have introduced two classes of constraint strategies. Within each class, there are $N$ strategies whose gains in each period are bounded by Lemma 5 or Lemma 8. We now attempt to design an adaptive strategy for each class via online learning such that it can earn almost as much as the best fixed spread strategy that considers the inventory constraints, respectively.

# 4.1 The Adaptive Strategies

Following the work of (Abernethy and Kale 2013), we view each constraint strategy $S ( b )$ within either the first or second class as an expert and run an online learning algorithm for learning with expert advice to these strategies. Denote by $w _ { t } ( b )$ the weight assigned to the reference strategy $S ( b )$ by the online learning algorithm at the end of each period $t - 1$ . Each reference strategy $S ( b )$ is executed in proportion to its assigned weight $w _ { t } ( b )$ in period $t$ . Furthermore, at the end of each period $t$ , the adaptive strategy submits a market buy order of $\begin{array} { r } { \sum _ { b \in \mathbb { B } } H _ { t - 1 } ( b ) \big [ w _ { t } ( b ) - \dddot w _ { t - 1 } ( b ) \big ] } \end{array}$ shares such that its inven ory equals $\begin{array} { r } { \sum _ { b \in \mathbb { B } } H _ { t } ( b ) w _ { t } ( b ) . } \end{array}$ The specific strategy is illustrated in Algorithm 3.

We consider two classic online learning algorithms to develop the adaptive strategies: the multiplicative weights (MW) (Littlestone and Warmuth 1994) and follow-theperturbed-leader (FPL) (Kalai and Vempala 2005). The adaptive strategies based on MW and FPL are referred to as ASMW and ASFPL, respectively. The adaptive strategy starts with an initial weight $\begin{array} { r } { w _ { 0 } ( b ) = \frac { 1 } { N } } \end{array}$ for each strategy $S ( b )$ . At the end of period $t$ , ASMW with time-varying parameters $\eta _ { t }$ updates the weight for each $b \in \mathbb { B }$ as follows:

$$
w _ { t + 1 } ( b ) = \frac { w _ { t } ( b ) \mathrm { e } ^ { \eta _ { t } \left[ V _ { t } ( b ) - V _ { t - 1 } ( b ) \right] } } { \sum _ { b ^ { ' } \in \mathbb { B } } w _ { t } ( b ^ { ' } ) \mathrm { e } ^ { \eta _ { t } \left[ V _ { t } ( b ^ { ' } ) - V _ { t - 1 } ( b ^ { ' } ) \right] } } .
$$

While ASFPL with parameters $\eta$ updates the weight for each $b \in \mathbb { B }$ as follows:

$w _ { t + 1 } ( b ) = P r [ V _ { t } ( b ) + f ( b ) \ge V _ { t } ( b ^ { ' } ) + f ( b ^ { ' } ) \mathrm { ~ f o r ~ } \forall b ^ { ' } \in \mathbb { B } ] ,$ where $f ( b )$ and $f ( \boldsymbol { b } ^ { \prime } )$ are samples from the exponential distribution with mean $1 / \eta$ .

1: Run every constraint strategy S(b) in class one or two parallel such that $H _ { t } ( b )$ , $C _ { t } ( b )$ and $V _ { t } ( b )$ for each strategy can be computed at the end of each period $t$ .   
2: Start an online learning algorithm $\mathcal { A }$ with one expert corresponding to each strategy $S ( b )$ . Denote by $w _ { t } ( b )$ the weight assigned to $S ( b )$ at the end of $t - 1$ th period.   
3: for $t = 1 , 2 , \cdots , T$ do   
4: Execute the limit orders from the previous period: a $w _ { t } ( b )$ weighted combination of the limit orders of the reference strategies.   
5: Submit and execute a market buy order of $\begin{array} { r } { \sum _ { b \in \mathbb { B } } H _ { t - 1 } ( b ) [ w _ { t } ( b ) - w _ { t - 1 } ( b ) ] } \end{array}$ shares at the price Pt.   
6: Compute $\Delta V _ { t } ( b )$ for each strategy $S ( b )$ .   
7: Update $w _ { t + 1 } ( b )$ for each strategy $S ( b )$ from $\mathcal { A }$ according to $\Delta V _ { t } ( b )$ .   
8: Submit a $w _ { t + 1 } ( b )$ weighted combination of the limit orders of the strategies S(b).

9: end for

For a given online learning algorithm $\mathcal { A }$ , denote by $C _ { t } ^ { A }$ and $\mathbf { \nabla } H _ { t } ^ { A }$ the amount of the cash and inventory held by the adaptive strategy based on $\mathcal { A }$ at the end of period $t$ . Thus, the total value of the adaptive strategy is given by $\begin{array} { r } { V _ { t } ^ { A } = } \end{array}$ $H _ { t } ^ { \mathcal { A } } P _ { t } + C _ { t } ^ { \mathcal { A } }$ . Since $\begin{array} { r } { H _ { t } ^ { \mathcal { A } } \dot { = } \sum _ { b \in \mathbb { B } } H _ { t } \dot { ( } b ) w _ { t } \dot { ( } b ) } \end{array}$ , for all $t$ we have

$$
| H _ { t } ^ { \mathcal { A } } | \leq R
$$

for adaptive strategies on hard-constraint strategies, and

$$
- \frac { \nabla } { \gamma } \leq H _ { t } ^ { A } \leq \frac { \nabla - b ^ { \mathrm { m i n } } } { \gamma }
$$

for adaptive strategies on soft-constraint strategies. This demonstrates that our adaptive strategies, whether based on soft- or hard-constraint strategies, effectively control inventory risk.

# 4.2 The Regret Bounds

Define $\begin{array} { r c l } { r e g ( { \cal A } ) } & { = } & { \displaystyle \operatorname* { m a x } _ { b \in \mathbb { B } } V _ { T } ( b ) - \sum _ { t = 1 } ^ { T } \sum _ { b \in \mathbb { B } } [ V _ { t } ( b ) - } \end{array}$ $V _ { t - 1 } ( b ) ] w _ { t } ( b )$ , which is the regret of the underlying algorithm $\mathcal { A }$ . We next bound the regret of the adaptive strategy in terms of $r e g ( { \mathcal { A } } )$ . Denote by $H ^ { \star }$ the upper bound of the difference between the inventory of any two reference strategies within the same class in each period. We have the following result.

Lemma 9. The regret of the adaptive strategy satisfies:

$$
r e g ( A S ) \leq r e g ( A ) + H ^ { \star } \nabla \sum _ { t = 2 } ^ { T } \sum _ { b \in \mathbb { B } } \lvert w _ { t } ( b ) - w _ { t - 1 } ( b ) \rvert .
$$

Adaptive Market Making on Hard-Constraint Strategies. For the hard-constraint strategies, we have $\begin{array} { r l } { H ^ { \star } } & { { } = } \end{array}$ $\frac { 1 } { \delta } \big ( b _ { \mathrm { ~ . ~ } } ^ { \mathrm { m a x } } - \ b _ { \mathrm { ~ . ~ } } ^ { \mathrm { m i n } } \big )$ by Lemma 4. Thus, the key of deriving the regret bound of the adaptive strategy is to bound |wt(b) − wt 1(b)|. Define ch = 9▽(b $\begin{array} { r l r } { c ^ { h } } & { { } = } & { \frac { 9 \nabla ( b ^ { \mathrm { m a x } } - b ^ { \mathrm { m i n } } ) } { \delta } + } \end{array}$ $5 \operatorname* { m i n } ( 2 R \nabla , \frac { \nabla ^ { 2 } } { \delta } )$ . We have the following results.

Theorem 10. $\begin{array} { r } { I f \eta _ { t } = \frac { 1 } { G ^ { h } } \operatorname* { m i n } \{ \sqrt { \frac { \ln N } { t } } , \frac { 1 } { 2 } \} } \end{array}$ , then the regret of ASMW on hard-constraint strategies is bounded from above by $2 c ^ { h } { \sqrt { T \ln N } }$ .

Theorem 11. If $\begin{array} { r } { \eta = \frac { 1 } { G ^ { h } } \sqrt { \frac { \ln N } { T } } } \end{array}$ , then the regret of ASFPL on hard-constraint strategies is bounded from above by $c ^ { h } { \sqrt { T \ln N } }$ .

Adaptive Market Making on Soft-Constraint Strategies. For the soft-constraint strategies, we have H⋆ = 2▽−bmin by Corollary 7. The proof of the regret bounds of the adaptive strategies on soft-constraint strategies is similar to that on hard-constraint strategies. Define $\bar { c ^ { s } } = ( 2 \nabla - b ^ { \mathrm { m i n } } ) ( 1 9 \nabla -$ $5 b ^ { \mathrm { m i n } } ) / \gamma$ . We have the following results.

Theorem 12. If $\begin{array} { r } { \eta _ { t } = \frac { 1 } { G ^ { s } } \operatorname* { m i n } \{ \sqrt { \frac { \ln N } { t } } , \frac { 1 } { 2 } \} } \end{array}$ , then the regret of ASMW on soft-constraint strategies is bounded from above by $2 c ^ { s } { \sqrt { T \ln N } }$ .

Theorem 13. If η = G1s q lnT , then the regret of ASFPL on soft-constraint strategies is bounded from above by $c ^ { s } { \sqrt { T \ln N } }$ .

# 5 Experiments

The performance of our market making algorithms is examined via data from the Chinese stock exchange. We collect from http://www. cdqianlong.com the tick-by-tick data of the CSI 500 index component stocks on each trading day from December 1, 2022 to December 31, 2022. The data include high-frequency information, such as intraday transaction time, traded price, and volume for each stock from the opening time, $9 { : } 2 5 { \mathrm { ~ a . m . } }$ , to the closing time, $3 { : } 0 0 ~ \mathrm { p . m . }$ , on each trading day. Stocks with the daily high price minus low price of less than 6 cents are excluded, which leaves 10,882 observed stock price paths in the sample period.1 The number of trades in each path (i.e., $T$ ) ranges from 1,005 to 406,363.

The experimental setup is as follows. We treat the implementation of an adaptive strategy on one stock of a single day as one experiment. Since the tick size $\delta$ is one cent, the window size $b$ is specified in cents. The set of possible values for $b$ is $\mathbb { B } = \{ 1 , 2 , 4 , 6 , 8 , 1 0 , 1 5 , 2 0 \}$ , resulting in $N \ = \ 8$ reference strategies in our adaptive market making. We set $R \in \{ 1 0 , 2 0 , 3 0 , 4 0 , 5 0 \}$ for the hardconstraint strategies and $\gamma ~ \in ~ \{ 0 . 1 \delta , 0 . 2 \delta , 0 . 3 \delta , 0 . 4 \delta , 0 . 5 \delta \}$ for the soft-constraint strategies. The learning rate is set as $\begin{array} { r l r } { \eta _ { t } } & { { } = } & { \operatorname* { m i n } \{ 1 , 4 \sqrt { \ln { N / t } } \} / G _ { t } } \end{array}$ for ASMW and $\begin{array} { r l r } { \eta } & { { } = } & { 4 \sqrt { \ln N / T } / G _ { t } } \end{array}$ for ASFPL, where $\begin{array} { r l } { G _ { t } } & { { } = } \end{array}$ $\begin{array} { r } { \operatorname* { m a x } _ { 1 \leq s \leq t , b ^ { 1 } , b ^ { 2 } \in \mathbb { B } } | V _ { s } ( b ^ { 1 } ) - V _ { s - 1 } ( b ^ { 1 } ) - V _ { s } ( b ^ { 2 } ) + V _ { s - 1 } ( b ^ { 2 } ) | } \end{array}$ . The weight $\boldsymbol { w } _ { t }$ in ASFPL is estimated by averaging 100 independently drawn perturbations.

Table 2: Our adaptive strategies with inventory constraints could indeed achieve no-regret. This table reports $\mathcal { G } ^ { \mathrm { m i n } }$ of adaptive strategies with inventory constraints, which is defined as the minimum of the realized average regret per period of an adaptive strategy minus its theoretical upper bound implied by theorems in Section 4 across experiments. The positive values of $\bar { \mathcal { G } } ^ { \mathrm { m i n } }$ imply that, for any stock path and algorithm setup, the realized regret is always lower than the corresponding theoretical regret bound.   

<html><body><table><tr><td rowspan="3">Algorithm</td><td colspan="5">Hard-constraint strategies</td><td colspan="5">Soft-constraint strategies</td></tr><tr><td colspan="5">R (absolute inventory limit)</td><td colspan="5">γ (risk-aversion coeficient)</td></tr><tr><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td><td>0.18</td><td>0.28</td><td>0.38</td><td>0.48</td><td>0.58</td></tr><tr><td>ASMW</td><td>3.32</td><td>3.54</td><td>3.11</td><td>3.48</td><td>3.05</td><td>0.71</td><td>0.73</td><td>0.68</td><td>0.65</td><td>0.74</td></tr><tr><td>ASFPL</td><td>1.74</td><td>1.78</td><td>1.72</td><td>1.75</td><td>1.69</td><td>0.35</td><td>0.32</td><td>0.17</td><td>0.26</td><td>0.29</td></tr></table></body></html>

Table 3: The gain and risk of adaptive strategies with(out) inventory constraints. $E$ denotes the average gain per period of an adaptive strategy in one experiment. $\overline { { E } }$ and $\sigma ( E )$ are the mean and standard deviation of $E$ across experiments. $\overrightarrow { H }$ denotes the mean of the final absolute inventory held by the adaptive strategy across experiments. Both $\overline { { E } }$ and $\sigma ( E )$ are measured in units of cents. Bolded values indicate higher Sharp ratios than (Abernethy and Kale 2013). These results show that ASMW on both constraint strategies outperforms adaptive strategies on non-constraint strategies in terms of higher Sharp ratios.   

<html><body><table><tr><td rowspan="3" colspan="2">Algorithm Index</td><td colspan="6">Hard-constraint strategies</td><td colspan="4">Soft-constraint strategies</td><td rowspan="3">Non-constraint strategies in</td></tr><tr><td rowspan="2"></td><td colspan="4">R (absolute inventory limit)</td><td colspan="4">γ (risk-aversion coefficient)</td></tr><tr><td>10</td><td>20</td><td>30</td><td>40</td><td>50 0.18</td><td>0.28</td><td>0.38</td><td>0.48</td><td>0.58 (Abernethy and Kale 2013)</td></tr><tr><td rowspan="4">ASMW</td><td>E</td><td>0.12</td><td>0.14</td><td>0.15</td><td>0.17</td><td>0.18</td><td>0.14</td><td>0.12</td><td>0.1</td><td>0.08</td><td>0.06</td><td>0.19</td></tr><tr><td>σ(E)</td><td>0.57</td><td>0.61</td><td>0.67</td><td>0.72</td><td>0.8</td><td>0.62</td><td>0.55</td><td>0.51</td><td>0.47</td><td>0.4</td><td>1.44</td></tr><tr><td> Sharp ratio</td><td>0.21</td><td>0.23</td><td>0.22</td><td>0.24</td><td>0.23</td><td>0.23</td><td>0.22</td><td>0.20</td><td>0.17</td><td>0.15</td><td>0.13</td></tr><tr><td>H</td><td>5.52</td><td>8.52</td><td>11.72</td><td>14.08</td><td>18.25</td><td>12.15</td><td>10.22</td><td>8.53</td><td>6.95</td><td>5.71</td><td>38.42</td></tr><tr><td rowspan="4">ASFPL</td><td>E</td><td>-0.74</td><td>-0.68</td><td>-0.57</td><td>-0.42</td><td>-0.34</td><td>-0.39</td><td>-0.42</td><td>-0.47</td><td>-0.51</td><td>-0.55</td><td>1.08</td></tr><tr><td>σ(E)</td><td>4.35</td><td>5.11</td><td>6.48</td><td>7.06</td><td>7.83</td><td>6.85</td><td>5.98</td><td>5.01</td><td>4.12</td><td>3.17</td><td>12.84</td></tr><tr><td> Sharp ratio</td><td>-0.17</td><td>-0.13</td><td>-0.09</td><td>-0.06</td><td>-0.04</td><td>-0.06</td><td>-0.07</td><td>-0.09</td><td>-0.12</td><td>-0.17</td><td>0.08</td></tr><tr><td>H</td><td>5.33</td><td>8.47</td><td>11.56</td><td>14.29</td><td>18.01</td><td>13.49</td><td>10.87</td><td>8.92</td><td>7.25</td><td>6.01</td><td>46.32</td></tr></table></body></html>

# 5.1 Validation of No-Regret

We first compare the performance of our adaptive strategies with that of the best reference strategy in hindsight. Given a stock price path, let $r e g ^ { R } = ( \operatorname* { m a x } _ { b \in \mathbb { B } } \breve { V } _ { T } ( b ) - V _ { T } ^ { \tilde { A } } ) / T$ , which is the realized average regret per period of the adaptive strategy based on an algorithm $\mathcal { A }$ in one experiment with $T$ periods, and $\mathcal { G }$ be the corresponding theoretical upper bound of average regret per period implied by theorems in Section 4 minus $r e g ^ { \mathbf { \check { R } } }$ . The distributions of $r \dot { e } g ^ { R }$ are presented in the appendix. To valid the no-regret of our adaptive strategies, we define $\begin{array} { r } { \mathcal { G } ^ { \mathrm { m i n } } = \operatorname* { m i n } _ { \mathrm { s t o c k } \mathrm { p a t h } } \bar { \mathcal { G } } } \end{array}$ whose values are reported in Table 2. The left-hand side of Table 2 concerns our adaptive strategies on the hard-constraint strategies. The positive values of $\mathcal { G } ^ { \mathrm { m i n } }$ imply that for any stock path and algorithm setup, the realized regret is always lower than the corresponding theoretical regret bound. This demonstrates that our adaptive strategies on hard-constraint strategies could indeed achieve no-regret. The comparison of adaptive strategies on the soft-constraint strategies reported in the righthand side of Table 2 shows the same results.

# 5.2 Comparison with (Abernethy and Kale 2013)

The gain and inventory risk control of our adaptive strategies are further examined by comparing them with those on non-constraint strategies presented in (Abernethy and Kale 2013). Let $E = V _ { T } ^ { \tilde { A } } / T$ , which is the average gain per period of the adaptive strategy based on an algorithm $\mathcal { A }$ in one experiment with $T$ periods, and $H$ be its absolute inventory at the end of period $T$ . Define Sharp ratio $= \overline { { E } } / \sigma ( E )$ to measure the risk-adjusted gain, where $\overline { { E } }$ and $\sigma ( E )$ are the mean and standard deviation of $E$ across experiments, respectively. The $\overline { { E } }$ , $\sigma ( E )$ , Sharp ratio, and the mean of $H$ across experiments (i.e., $\overline { { H } }$ ) for adaptive strategies on constraint and non-constraint strategies in the full sample are reported in Table 3. These results show that our adaptive strategies on constraint strategies outperform those on nonconstraint strategies in terms of risk control. First, adaptive strategies on both constraint strategies have lower $\overrightarrow { H }$ and $\sigma ( E )$ than those on non-constraint strategies. It is worth noting that, consistent with Lemma 6, $\overline { { H } }$ for adaptive strategies on soft-constraint strategies decreases as $\gamma$ increases from $0 . 1 \delta$ to $0 . 5 \delta$ . Furthermore, with respect to the gain, ASMW outperforms ASFPL and makes a profit for all $R$ and $\gamma$ cases. Note that inventory control inevitably leads to a decrease in gain in no-trend paths, which account for the vast majority of our sample. Thus, adaptive strategies on non-constraint strategies unsurprisingly have the highest gain in the full sample. Finally, for any $R$ and $\delta$ , ASMW on both constraint strategies outperforms adaptive strategies on non-constraint strategies in terms of higher Sharp ratios.

To further examine the performance of our strategies during the upward and downward markets, we classify all the price paths in our sample into three subsamples. Specifically, for a given stock price path on a trading day, denote by $P _ { 0 }$ , $P _ { \mathrm { m i n } }$ , $P _ { \mathrm { m a x } }$ , and $P _ { T }$ the daily opening, low, high, and closing prices, respectively. The price path is called under an uptrend if $\frac { P _ { T } } { P _ { 0 } } \ \geq \ 1 . 0 3$ and $\frac { \bar { P } _ { T } - \bar { P _ { 0 } } } { P _ { \operatorname* { m a x } } - P _ { \operatorname* { m i n } } } \ge 0 . 8$ , under a downtrend if $\begin{array} { r } { \frac { P _ { T } } { P _ { 0 } } \ \leq \ 0 . 9 7 } \end{array}$ and $\frac { P _ { T } - P _ { 0 } } { P _ { \operatorname* { m a x } } - P _ { \operatorname* { m i n } } } ~ \leq ~ - 0 . 8$ , and no trend otherwise. The comparisons in subsamples are reported in the appendix. In both uptrend and downtrend samples, adaptive strategies on both constraint strategies have larger $\overline { { E } }$ , lower $\sigma ( E )$ , and larger Sharp ratio than those on non-constraint strategies for all $R$ and $\gamma$ . Specially, adaptive strategies on both constraint strategies almost always make a profit in both subsamples. In contrast, adaptive strategies on non-constraint strategies make a loss in both subsamples.

# 6 Conclusions

In this paper, we introduce soft- and hard-constraint strategies for inventory risk control. Two classic online learning algorithms, namely, MW and FPL, are used to develop adaptive strategies on both constraint strategies that achieve noregret. We want to emphasize that our results are technically interesting. The reason is that in the standard online learning framework, the gain in each period is assumed to lie in a fixed constant interval and does not depend on any state variables. In contrast, the gain in our model depends on the past inventory, which is path-dependent and time-varying. Thus, we have to bound the difference between the gains of any two reference strategies, which is more challenging than the work of (Abernethy and Kale 2013). Furthermore, different from most of the existing work (Avellaneda and Stoikov 2008; Spooner and Savani 2020), our adaptive strategies are model-free in the sense that no assumptions on the dynamics of the LOB and stock price are required.

Our work leaves a few interesting open problems. In our model, only one parameter (i.e., the spread) is learned. In a more general model, the distance between the bid and middle price and that between the ask and middle price can be learned as two independent parameters of reference strategies. In addition, there are some other engineering problems in finance, such as the optimal execution with pathdependent constraints, which can also be investigated within the framework of online learning.

# Acknowledgments

The authors are grateful for many valuable comments from the anonymous reviewers. This study was partially supported by the National Natural Science Foundation of China (Grant Numbers 11501464, 11761141007, 71971177, 72342012) and Leshan Normal University Scientific Research Start-up Project for Introducing High-level Talents (Grant Number RC2024031).