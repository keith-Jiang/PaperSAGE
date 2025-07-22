# Expressive Power of Temporal Message Passing

Przemysław Andrzej Wałe˛ga1,2, Michael Rawson3

1Queen Mary University of London, UK 2University of Oxford, UK 3University of Southampton, UK p.walega@qmul.ac.uk, michael $@$ rawsons.uk

# Abstract

Graph neural networks (GNNs) have recently been adapted to temporal settings, often employing temporal versions of the message-passing mechanism known from GNNs. We divide temporal message passing mechanisms from literature into two main types: global and local, and establish WeisfeilerLeman characterisations for both. This allows us to formally analyse expressive power of temporal message-passing models. We show that global and local temporal message-passing mechanisms have incomparable expressive power when applied to arbitrary temporal graphs. However, the local mechanism is strictly more expressive than the global mechanism when applied to colour-persistent temporal graphs, whose node colours are initially the same in all time points. Our theoretical findings are supported by experimental evidence, underlining practical implications of our analysis.

# Introduction

Message-passing graph neural networks (or MP-GNNs) (Gilmer et al. 2017) are prominent models for graph learning, which have achieved state-of-the art performance in tasks of link prediction as well as in node and graph classification. Importantly, they proved successful in a number of real-world applications including social networks, protein-protein interactions, and knowledge graphs (Zhou et al. 2020).

In recent years, there has been growing interest in adapting MP-GNNs to process temporal graphs (for an example of a temporal graph see the left part of Figure 1) which are particularly well-suited for dynamic applications such as recommender systems (Wu et al. 2023), traffic forecasting (Yu, Yin, and Zhu 2018), finance networks (Pareja et al. 2020), and modelling the spread of diseases (Kapoor et al. 2020). Research in this direction gave rise to various temporal MP-GNNs (MP-TGNNs) obtained by introducing temporal variants of the message-passing mechanism (Longa et al. 2023; Skarding, Gabrys, and Musial 2021; Gao and Ribeiro 2022). This can be obtained by assigning to graph nodes different embeddings (feature vectors) for different time points and then passing messages between timestamped nodes. Depending on the routes of messages-passing between timestamped nodes and on the encoding of the temporal component in the messages, we arrive at various temporal message-passing mechanisms. In this paper we distinguish two main groups of MP-TGNNs: global, where messages can be passed between nodes stamped with different times and local, where messages are passed only between nodes stamped with the same time, while information about other times is encoded within messages.

Although several variants of global (Longa et al. 2023; Xu et al. 2020; Luo and Li 2022) and local (Rossi et al. 2020; Qu et al. 2020) MP-TGNNs have been designed and successfully applied, we still lack a good understanding of modelling capabilities dictated by their temporal messagepassing mechanisms, and do not have answers to the following fundamental questions. What tools can we use to analyse the expressive power of global and local MP-TGNNs? Are there limits to the expressive power of either type? Which type can express more? How does the difference in expressiveness affect practical performance? Answers to these questions are key when choosing an appropriate temporal message-passing mechanism for a particular task and when designing new MP-TGNNs. The importance of answering such questions has been clearly shown by research on expressive power of static MP-GNNs, which equipped us with powerful tools and gave rise to a whole new research direction (Morris et al. 2019; Xu et al. 2019; Cai, Fürer, and Immerman 1992; Grohe 2023; Barceló et al. 2020). In the temporal setting, however, such an analysis is still missing. We aim to fill this urgent gap.

Contributions. Our main contributions are as follows:

• We formalise the two main types of MP-TGNNs, global and local, depending on the form of the adopted temporal message-passing mechanism. • We characterise the expressive power of both types. To determine which temporal nodes can be distinguished by MP-TGNNs, we construct a knowledge graph and then apply the 1-dimensional Weisfeiler-Leman test (1-WL). As depicted in Figure 1, our construction of the knowledge graph is different for global and local MP-TGNNs, but in both cases this approach allows us to precisely capture the expressive power of MP-TGNNs. For example, given the temporal graph $T G$ in Figure 1, global and local MP-TGNNs distinguish the same nodes, since the colourings in the rightmost graphs are the same.

Nodes distinguishable by global MP-TGNNs:

![](images/fa3c03b32a324974d619fbd6eebb881123bfffc75aa8f45995631f003a36e454.jpg)  
Figure 1: Our approach to determine which nodes in a temporal graph $T G$ are distinguishable by MP-TGNN: we construct of knowledge graphs $\kappa _ { \mathsf { g l o b } } ( T G )$ and $\textstyle \mathcal { K } _ { \mathrm { l o c } } ( T G )$ , and then apply 1-WL

• We use the above characterisation to show that, quite surprisingly, both global and local MP-TGNNs can distinguish nodes which are pointwise isomorphic. This leads us to introduce a stronger timewise isomorphism, wellsuited for characterisation of MP-TGNNs.

• The Weisfeiler-Leman characterisation also allows us to show that global and local MP-TGNNs have incomparable expressive power: each of the types can distinguish nodes which are indistinguishable by the other type. However, if the input temporal graph is colourpersistent (initial embedding of each node is the same at all time points), local MP-TGNNs are more expressive than global MP-TGNNs. We can extend these results to a complete expressiveness classification as in Figure 2.

global MP-TGNNs, global MP-TGNNs, any T Gs colour-persistent T Gs #￥ > local MP-TGNNs, local MP-TGNNs, any T Gs colour-persistent T Gs • Finally, we experimentally validate our theoretical results by constructing proof-of-concept global and local models. We show that, indeed, on colour-persistent graphs local models outperform global models, when compared on the temporal link-prediction and TGB 2.0 benchmark suite (Gastinger et al. 2024). This is the case when models use the same number of layers, and the difference in performance increases further if we choose optimal number of layers for each type of model separately.

# Background

Temporal graphs. We focus on temporal graphs in the socalled snapshot representation (Longa et al. 2023; Gao and Ribeiro 2022; Skarding, Gabrys, and Musial 2021) shown in Figure 3. A temporal graph is a finite sequence $T G =$ $( G _ { 1 } , \bar { t } _ { 1 } ) , \dots , ( G _ { n } , \bar { t } _ { n } )$ of undirected node-coloured graphs

$G _ { i } = ( V _ { i } , E _ { i } , c _ { i } )$ , where $t _ { 1 } < \cdots < t _ { n }$ are real-valued time po nts, constituting the temporal domain time $( T G )$ . Each $V _ { i }$ is a finite set of nodes, each $E _ { i } \subseteq \{ \{ u , v \} \subseteq V _ { i } \mid ^ { \cdot } u \neq v \}$ is a set of undirected edges, and $c _ { i } : V _ { i }  D$ a⊆ssign∣ s n≠ode}s colours from some set $D$ , which could be real feature vectors. Following standard notation, we sometimes use $\mathbf { x } _ { v } ( t _ { i } )$ instead of $c _ { i } ( \bar { \upsilon } )$ . We represent $c _ { i }$ using different colours(fo)r nodes in figures. We assume that the domain of nodes does not change over time, so $V _ { 1 } = . . . = V _ { n } = V ( T G )$ . We call a pair of a node $v \in V ( T G )$ a=nd a t=ime p=oint( $t \in$ t)ime $( T G )$ a timestamped nod∈e $( v , t )$ )and we let $t – n o d e s ( T G )$ b(e th)e set of all timestampe(d no)des in $T G$ . For the sa(ke of) a clear presentation we assume that edges are not labelled.

![](images/7366f7ad86cb5782648cc7fbe856e8b84112ff7aed714778d1b92ad41e804f97.jpg)  
Figure 2: Relative expressive power of MP-TGNNs   
Figure 3: A temporal graph in the snapshot representation

We say that a temporal graph is colour-persistent if initial colours of nodes do not change in time, so $c _ { i } ( v ) \ = \ c _ { j } ( v )$ for each node $v$ and all $i , j \in \{ 1 , \ldots , n \}$ , see F(ig)ur=e 4 (a ). Colour-persistent graphs c n∈ b{e also re}presented as static edge-labelled multi graphs, called aggregated form (Gao and Ribeiro 2022), as depicted in Figure 4 (b).

Temporal graph neural networks with message-passing. We let a message-passing temporal graph neural network (MP-TGNN) be a model $\mathcal { A }$ which, given a temporal graph $T G$ , computes embeddingAs for all timestamped nodes by implementing a temporal variant of message-passing. Embeddings are then used to predict links or classify nodes and graphs. Some models (e.g. TGAT, NAT, and TGN) apply temporal message-passing to arbitrary temporal graphs, whereas others (e.g. TDGNN) are applicable to colourpersistent temporal graphs (or equivalently, to the aggregated representation) only. Below we present a general form of an MP-TGNN model $\mathcal { A }$ with $L$ layers, which subsumes a number of message-passiAng mechanisms. Given a temporal graph $T G = ( G _ { 1 } , t _ { 1 } ) , \ldots , ( G _ { n } , t _ { n } )$ , a model $\mathcal { A }$ computes for each no=de( $v$ , time) poin t( $t$ , and )layer $\ell \in \{ 0 , \ldots , L \}$ an embedding ${ \bf h } _ { v } ^ { ( \ell ) } ( t )$ as follows:

![](images/05d1643f38668c02284824dba7b7d98bb9c1d133f9c518455de24384ae048315.jpg)  
Figure 4: A colour-persistent temporal graph (a) and its aggregated representation (b)

$$
\begin{array} { r l } & { \mathbf { h } _ { v } ^ { \left( 0 \right) } ( t ) = \mathbf { x } _ { v } ( t ) , } \\ & { \mathbf { h } _ { v } ^ { \left( \ell \right) } ( t ) = \mathsf { C O M } ^ { \left( \ell \right) } \Big ( \mathbf { h } _ { v } ^ { \left( \ell - 1 \right) } ( t ) , \mathsf { A G G } ^ { \left( \ell \right) } \Big ( } \\ & { \qquad \quad \big \{ \{ ( \star , g ( t - t ^ { \prime } ) ) \mid ( u , t ^ { \prime } ) \in \mathcal { N } ( v , t ) \} \big \} \Big ) \Big ) , } \end{array}
$$

where:

• $\mathsf { C O M } ^ { ( \ell ) }$ and $\mathsf { A G G } ^ { ( \ell ) }$ are combination and aggregation functions in layer $\ell$ ; $\mathsf { C O M } ^ { ( \ell ) }$ maps a pair of vectors into a single vector, whereas $\mathsf { A G G } ^ { ( \ell ) }$ maps a multiset, represented as $\{ \{ \cdots \} \}$ , into a single vector,

• $g$ maps a time duration into a vector or scalar quantity, • $\mathcal { N } ( v , t )$ is the temporal neighbourhood of $( v , t )$ , defined aNs (follo)ws (Souza et al. 2022):

$$
\begin{array} { r } { \mathcal { N } ( v , t ) = \Big \{ ( u , t ^ { \prime } ) \ | \ t ^ { \prime } = t _ { i } \ \mathrm { a n d } \ \{ u , v \} \in E _ { i } , \ } \\ { \mathrm { f o r ~ s o m e ~ } ( G _ { i } , t _ { i } ) \in T G \mathrm { ~ w i t h ~ } t _ { i } \leq t \Big \} . } \end{array}
$$

Hence, $\mathcal { N } ( v , t )$ is the set of timestamped nodes $( u , t ^ { \prime } )$ such thatNt(here)is an edge between $u$ and $v$ at $t ^ { \prime } \leq t$ (.

Recently 1-WL has been applied to knowledge graphs $K G = ( \dot { V _ { \star } } E , R , c )$ where $V$ are nodes, $E \subseteq R \times V \times V$ are direct=ed( edges wit)h labels from $R$ , and $c : V \to D$ ×colours nodes (Huang et al. 2023; Barceló et al. 2∶02 .→An isomorphism between knowledge graphs $K G _ { 1 } = \left( V _ { 1 } , E _ { 1 } , R _ { 1 } , c _ { 1 } \right)$ and $K G _ { 2 } = \left( V _ { 2 } , E _ { 2 } , R _ { 2 } , c _ { 2 } \right)$ is any bijec=t o(n $f : V _ { 1 } \to V _ { 2 }$ such that, =o (all $u , v \in V _ { 1 }$ an)d $r \in R _ { 1 } ;$ : (i) $c _ { 1 } ( v ) = c _ { 2 } ( f ( v ) )$ and (ii) $( r , u , v ) \in E _ { 1 }$ if and onl∈y if $( r , f ( u ) , f ( v ) ) \in E _ { 2 }$ . A relatio(nal loc)al∈1-WL algorithm (co(nven(tio)nall(y $\mathsf { r w l } _ { 1 }$ , but we write rwl) is a natural extension of 1-WL to the case of knowledge graphs (Huang et al. 2023). Given a knowledge graph $K G = \left( V , E , R , c \right)$ , the algorithm computes iteratively, for all $v \in V$ and $\ell \in \mathbb { N }$ , values $\mathsf { r w l } ^ { ( \ell ) } ( v )$ as follows:

• $\bigstar$ is either $\mathbf { h } _ { u } ^ { ( \ell - 1 ) } ( t ^ { \prime } )$ or $\mathbf { h } _ { u } ^ { ( \ell - 1 ) } ( t )$ . If $\star = \mathbf { h } _ { u } ^ { ( \ell - 1 ) } ( t ^ { \prime } )$ w☀e say that $\mathcal { A }$ is a (glo)bal (in tim(e))TGN☀N, =as comp(uta)- tion of ${ \bf h } _ { v } ^ { ( \ell ) } ( t )$ requires aggregation of embeddings in all past time  (in)ts $t ^ { \prime }$ ; global TGNNs are also called Temporal Embedding TGNNs (Longa et al. 2023) and include TGAT and NAT. If $\star = \mathbf { h } _ { u } ^ { ( \ell - 1 ) } ( t )$ we say that $\mathcal { A }$ is local, as only embeddings from the current time point $t$ are aggregated; local TGNNs include TGN and TDGNN.

$$
\begin{array} { r l } & { \mathsf { r w l } ^ { ( 0 ) } ( v ) = c ( v ) , } \\ & { \mathsf { r w l } ^ { ( \ell ) } ( v ) = \tau \Big ( \mathsf { r w l } ^ { ( \ell - 1 ) } ( v ) , } \\ & { \qquad \quad \{ \{ ( \mathsf { r w l } ^ { ( \ell - 1 ) } ( u ) , r ) \mid u \in \mathcal { N } _ { r } ( v ) , r \in R \} \} \Big ) , } \end{array}
$$

Weisfeiler-Leman algorithm. An isomorphism between undirected node-coloured graphs $G _ { 1 } ~ = ~ \left( V _ { 1 } , E _ { 1 } , c _ { 1 } \right)$ and $G _ { 2 } ~ = ~ ( V _ { 2 } , E _ { 2 } , c _ { 2 } )$ is any bijection $f : V _ { 1 } \to V _ { 2 }$ , sa)tisfying f=or(any $u$ and) $v$ : (i) $c _ { 1 } ( v ) = c _ { 2 } ( f ( v ) )$ →d (ii) $\{ u , v \} \in$ $E _ { 1 }$ if and only if $\{ f ( u ) , f ( v ) \} \in E _ { 2 }$ (. )T)he $\jmath$ -dime{nsion}a∈l Weisfeiler-Leman a{lgo(rit)hm( 1)-}W∈L) (Weisfeiler and Leman 1968) is a powerful heuristic for graph isomorphism (Babai and Kucera 1979), which has the same expressive power as MP-GNNs with injective aggregation and combination (Morris et al. 2019; Xu et al. 2019).

where $\mathcal { N } _ { r } ( v ) = \{ u \ | \ ( r , u , v ) \in E \}$ is the $r$ -neighbourhood of $v$ , and $\tau$ is an injective function. It is shown that rwl has the same expressive power as R-MPNNs, that is, MP-GNNs processing knowledge graphs (Huang et al. 2023).

# Related Work

There is recently an increasing interest in temporal and dynamic graph neural networks (Longa et al. 2023; Qin and Yeung 2024; Skarding, Gabrys, and Musial 2021; Kazemi et al. 2020). Pertinent models include TGN (Rossi et al. 2020), TGAT (Xu et al. 2020), TDGNN (Qu et al. 2020), and NAT (Luo and Li 2022), which are all based on temporal message-passing mechanisms.

Expressive power results for temporal models are very limited. Souza et al. (2022) compared expressive power of temporal graph neural networks exploiting temporal walks, with those based on local message passing combined with recurrent memory modules. Gao and Ribeiro (2022) compared time-and-graph with time-then-graph models, which are obtained by different combinations of static graph neural networks and recurrent neural networks. In the context of temporal knowledge graphs, expressive power of similar models was recently considered by Chen and Wang (2023).

More mature results have been established for models processing edge-labelled graphs. Such graphs are closely related to temporal graphs, since the aggregated representation of a temporal graph (Gao and Ribeiro 2022), Figure 4 (b), is a multigraph with edges labelled by time points. However, since the aggregated representation does not allow us to assign different colours to the same node in different time points, not all temporal graphs can be directly transformed into multigraphs. Barceló et al. (2022) introduced 1-WL for models processing undirected multirelational graphs, whereas Beddar-Wiesing et al. (2024) introduced 1-WL for dynamic graphs. Huang et al. (2023) proposed 1-WL for models processing directed multi-relational graphs (i.e. knowledge graphs), namely for relational message passing neural networks (R-MPNNs), which encompass several known models such as RGCN (Schlichtkrull et al. 2018) and CompGCN (Vashishth et al. 2020).

Temporal graphs can be also given in the event-based representation (Longa et al. 2023), as a sequence of timestamped events that add/delete edges or modify feature vectors of nodes. Since temporal graphs in the aggregated and event-based representations can be transformed into the snapshot representation (Gao and Ribeiro 2022; Longa et al. 2023), we focus on the snapshot representation in the paper.

# Temporal Weisfeiler-Leman Characterisation

We provide a general approach for establishing expressive power of MP-TGNNs using standard 1-WL. To do so, we transform a temporal graph $T G$ into a knowledge graph $K G$ such that MP-TGNNs can distinguish exactly those nodes in $T G$ whose counterparts in $K G$ can be distinguished by the standard 1-WL. This contrasts with approaches studying expressive power by modifying 1-WL for particular types of temporal graph neural networks (Souza et al. 2022; Gao and Ribeiro 2022). Note that our results concern distinguishability of nodes, not graphs. Node distinguishability is likely of more practical interest and can be used to distinguish graphs.

We transform $T G$ into two knowledge graphs: $\kappa _ { \mathsf { g l o b } } ( T G )$ and $\textstyle \mathcal { K } _ { \mathrm { l o c } } ( T G )$ , suitable for analysing, respectivKely, g(loba)l and oKcal(MP-)TGNNs. We first introduce $\kappa _ { \mathrm { g l o b } } ( T G )$ , whose edges correspond to temporal message -Kpassi(ng in) global MP-TGNNs (Figure 5). Intuitively, $\kappa _ { \mathsf { g l o b } } ( T G )$ contains a separate node $( v , t )$ for each timestamKped(node) in $T G$ and an edge betwee(n $( v , t )$ and $( u , t ^ { \prime } )$ labelled by $t - t ^ { \prime }$ if $( u , t ^ { \prime } )$ is in the temporal(neig)hbour(hood)of $( v , t )$ .

![](images/09108853621be1436cebbffd00d51332ab5fd34af02ef895a0c972d7256f40ba.jpg)  
Figure 5: $\kappa _ { \mathsf { g l o b } } ( T G )$ constructed for $T G$ from Figure 3

Definition 1. Let $T G = ( G _ { 1 } , t _ { 1 } ) , \dots , ( G _ { n } , t _ { n } )$ be a temporal graph with $G _ { i } ~ = ~ ( V _ { i } , E _ { i } , c _ { i } )$ . We( define )a knowledge graph $\mathcal { K } _ { \mathrm { g l o b } } ( T G ) = \left( V , E , R , c \right)$ with components:

• $V = t { - } n o d e s ( T G )$ ,   
• $E = \big \{ \big ( t _ { j } - t _ { i } , \big ( v , t _ { i } \big ) , \big ( u , t _ { j } \big ) \big ) \big | i \leq j a n d \big \{ u , v \big \} \in E _ { i } \big \} ,$ • $R = \{ 0 , \ldots , n - 1 \}$ ,   
• $c : V \to R$ satisfies1 $c ( v , t _ { i } ) = c _ { i } ( v )$ , for all $( v , t _ { i } ) \in V$ .

We use $\kappa _ { \mathrm { g l o b } } ( T G )$ to bridge the expressive power of global MP-KTGN(Ns a)nd 1-WL. First we show that global MP-TGNNs cannot distinguish more timestamped nodes over $T G$ than 1-WL over $\bar { \kappa _ { \mathrm { g l o b } } } ( T G )$ .

Theorem 2. For any temporal graph $T G$ , any timestamped nodes $( v , t )$ and $( u , t ^ { \prime } )$ in $T G$ , and any $\ell \in \mathbb { N }$ :

1for brevity we will drop double brackets, e.g. from $c ( ( v , t _ { i } ) )$

Proof sketch. By induction on $\ell$ . The base case holds since ${ \sf r w l } ^ { ( 0 ) } ( v , t _ { i } ) = c ( v , t _ { i } ) = c _ { i } ( v ) = { \bf h } _ { v } ^ { ( 0 ) } \left( t _ { i } \right)$ , for each $( v , t _ { i } )$ in $T G$ (. Th)e =ind(uctive) =step pr)o=ceed i(n a similar w(ay to) GNNs (Morris et al. 2019), but additionally exploits the following key property of $\mathcal { K } _ { \mathrm { g l o b } } ( T G )$ : $( u , t ^ { \prime } ) \ \in \ \mathcal { N } _ { r } ( v , t )$ in $\kappa _ { \mathrm { g l o b } } ( T G )$ iff $\bar { \boldsymbol { r } } = \dot { \boldsymbol { t } } - \boldsymbol { t } ^ { \prime }$ Kand $( u , t ^ { \prime } ) \in \mathcal { N } ( v , t )$ iNn $T G$ ,)for aKll tim(esta)mped n=odes− $( v , t )$ an(d $( u , t ^ { \prime } )$ iNn $T G$ .) □

Moreover, we can show the opposite direction: there is a global MP-TGNN distinguishing exactly the same nodes over $T G$ as 1-WL over $\kappa _ { \mathsf { g l o b } } ( T G )$ . By Theorem 2 each global MP-TGNN is no Kmore(expr)essive than 1-WL over $\bar { \kappa } _ { \mathrm { g l o b } } ( T G )$ , so for the next theorem it suffices to construct an KMP-T(GNN) at least as expressive as 1-WL over $\kappa _ { \mathsf { g l o b } } ( T G )$ .

Theorem 3. For any temporal graph $T G$ and any $L \in \mathbb { N }$ , there exists a global MP-TGNN $\mathcal { A }$ with $L$ layers such∈that for all timestamped nodes $( v , t ) , ( u , t ^ { \prime } )$ in $T G$ and all $\ell \leq L$ the following are equivalen(t:

$$
\begin{array} { r l } & { \bullet { \bf \Upsilon } { \bf h } _ { v } ^ { ( \ell ) } ( t ) = { \bf h } _ { u } ^ { ( \ell ) } ( t ^ { \prime } ) i n { \cal A } , } \\ & { \bullet { \bf \Upsilon } { \bf r } { \bf w } | ^ { ( \ell ) } ( v , t ) = { \bf r } { \bf w } | ^ { ( \ell ) } ( u , t ^ { \prime } ) i n \mathcal { K } _ { \mathrm { g l o b } } ( T G ) . } \end{array}
$$

Proof sketch. The important part of the proof is for the forward implication, as the other implication follows from Theorem 2. We use the result of Huang et al. (2023)[Theorem A.1] showing that for any knowledge graph and in particular $\kappa _ { \mathsf { g l o b } } ( T G )$ , there is a relational message-passing neural netwKork(R-M)PNN) model $\boldsymbol { B }$ such that if two nodes $( v , t )$ and $( u , t ^ { \prime } )$ of $\kappa _ { \mathsf { g l o b } } ( T G )$ havBe the same embeddings (at a )layer $\ell$ , we get $\mathsf { r w l } ^ { ( \ell ) } ( v , t ) = \mathsf { r w l } ^ { ( \ell ) } ( u , t ^ { \prime } )$ . Huang et al.’s model $\boldsymbol { B }$ computes $\overline { { \mathbf { h } } } _ { ( v , t ) } ^ { ( \ell ) }$ as follows: $\overline { { \mathbf { h } } } _ { ( v , t ) } ^ { ( 0 ) } = c ( v , t )$ and $\overline { { \mathbf { h } } } _ { ( v , t ) } ^ { ( \ell ) } =$ sign W(ℓ) h(ℓv,−t1) r R u,t′ r v,t αrh(ℓu−,t1′) To finish the proof, we construct a global MP-TGNN such that ${ \bf h } _ { v } ^ { ( \ell ) } ( t )$ computed by $\mathcal { A }$ on $T G$ coincide with $\overline { { \mathbf { h } } } _ { ( v , t ) } ^ { ( \ell ) }$ computed by $\boldsymbol { B }$ on $\kappa _ { \mathsf { g l o b } } ( T G )$ . We obtain it by setting in Equation (2) functions $\mathsf { A G G } ^ { ( \ell ) }$ to the sum and $\mathsf { C O M } ^ { ( \ell ) }$ to the sign of a particular linear combination. □

Next, we show that we can also construct a knowledge graph representing message-passing in local MP-TGNNs. In contrast to $\kappa _ { \mathsf { g l o b } } ( T G )$ , edges of the new knowledge graph $\textstyle \mathcal { K } _ { \mathrm { l o c } } ( T G )$ aKre bi(direc)tional and hold only between nodes sKtam(ped )with the same time. Such a knowledge graph is presented in Figure 6 and formally defined below.

![](images/bb1cf28ed6894946ce4fc653b6679152faafa55c1c99636b23689ad3495d9dce.jpg)  
Figure 6: Knowledge graph $\textstyle \mathcal { K } _ { \mathrm { l o c } } ( T G )$ for $T G$ from Figure 3

Definition 4. Let $T G = ( G _ { 1 } , t _ { 1 } ) , \dots , ( G _ { n } , t _ { n } )$ be a temporal graph with $G _ { i } ~ = ~ ( V _ { i } , E _ { i } , c _ { i } )$ . We( define )a knowledge knowledge graph ${ \mathcal { K } } _ { \sf l o c } ( T G ) = ( V , E , R , c )$ with:

• $V = t { - } n o d e s ( T G )$ ,   
• $E = \big \{ \big ( t _ { j } - t _ { i } , \big ( v , \dot { t } _ { j } \big ) , ( u , t _ { j } ) \big ) \big | i \le j a n d \big \{ u , v \big \} \in E _ { i } \big \} ,$ • $R = \{ 0 , \ldots , n - 1 \}$ ,   
• $c : V \to R$ satis−fies} $c ( v , t _ { i } ) = c _ { i } ( v )$ , for all $( v , t _ { i } ) \in V$ .

We can show that local MP-TGNNs can distinguish exactly the same nodes in $T G$ as rwl can distinguish in $\textstyle \bigwedge _ { \mathrm { l o c } } ( T G )$ , as formally stated in the following two theorems.

Theorem 5. For any temporal graph $T G$ , any timestamped nodes $( v , t )$ and $( u , t ^ { \prime } )$ in $T G$ , and any $\ell \in \mathbb { N }$ :

• $I f \mathsf { r w l } ^ { ( \ell ) } ( v , t ) = \mathsf { r w l } ^ { ( \ell ) } ( u , t ^ { \prime } )$ in $\textstyle \sum _ { \mathrm { l o c } } ( T G )$ , • then $\mathbf { h } _ { v } ^ { ( \ell ) } ( t ) = \mathbf { h } _ { u } ^ { ( \ell ) } ( t ^ { \prime } )$ in any local MP-TGNN.

Theorem 6. For any temporal graph $T G$ and any $L \in \mathbb { N }$ , there exists a local MP-TGNN $\mathcal { A }$ with $L$ layers such tha∈t for all timestamped nodes $( v , t ) , ( u , t ^ { \prime } )$ in $T G$ and all $\ell \leq L$ , the following are equivalent:

• $\mathbf { h } _ { v } ^ { ( \ell ) } ( t ) = \mathbf { h } _ { u } ^ { ( \ell ) } ( t ^ { \prime } )$ in $\mathcal { A } ,$ , • rw ) v, t wl(ℓ) u, t′ in loc T G .

The Weisfeiler-Leman characterisation of global and local MP-TGNNs established in the above theorems provides us with a versatile tool for analysing expressive power, which we will intensively apply in the following parts of the paper.

# Timewise Isomorphism

While message-passing GNNs (corresponding to 1-WL) provide us with a heuristic for graph isomorphism, their temporal extensions can be seen as heuristics for isomorphism between temporal graphs. However, in the temporal setting it is not clear what notion of isomorphism we should use to obtain an analogous correspondence. We use the characterisation from the previous section to show, quite surprisingly, that both global and local MP-TGNNs can distinguish nodes which are pointwise isomorphic—called isomorphic by Beddar-Wiesing et al. (2024). This observation leads us to definition of timewise isomorphism as a suitable notion for node indistinguishability in temporal graphs.

Pointwise isomorphism requires that pairs of corresponding snapshots in two temporal graphs are isomorphic. For example $( a , t _ { 2 } )$ in $T G$ and $( a ^ { \prime } , t _ { 2 } )$ in $T G ^ { \prime }$ from Figure 7 are pointwise( isom)orphic since( $f _ { 1 }$ wi)th $f _ { 1 } ( a ) = b ^ { \prime }$ , ${ \tilde { f _ { 1 } } } ( b ) = c ^ { \prime }$ , and $f _ { 1 } ( c ) = a ^ { \prime }$ is an isomorphism betw(ee)n $G _ { 1 }$ and $G _ { 1 } ^ { \prime }$ ,=and $f _ { 2 }$ with( $f _ { 2 } ( a ) = a ^ { \prime }$ , $f _ { 2 } ( b ) = { \overline { { b } } } ^ { \prime }$ , and $f _ { 2 } ( c ) = c ^ { \prime }$ is an isomorphism betw(ee)n $G _ { 2 }$ and( $G _ { 2 } ^ { \prime }$ .=A formal d(ef)in=ition is below.

![](images/3058dbd46c0e04761a9eba52a6775ac360d3d547b501c204253d3f4c8d9ae5f5.jpg)  
Figure 7: Pointwise isomorphic $( a , t _ { 2 } )$ and $( a ^ { \prime } , t _ { 2 } )$

Definition 7. Temporal graphs $T G = ( G _ { 1 } , t _ { 1 } ) , \dots , ( G _ { n } , t _ { n } )$ and $T G ^ { \prime } = \left( G _ { 1 } ^ { \prime } , t _ { 1 } ^ { \prime } \right) , \dots , \left( G _ { m } ^ { \prime } , t _ { m } ^ { \prime } \right)$ (are poi)ntwise(isomor)- phic if both of the following hold:

• $\mathsf { t i m e } ( T G ) = \mathsf { t i m e } ( T G ^ { \prime } )$ (so $n = m$ and $t _ { i } = t _ { i } ^ { \prime }$ for all $i \in \{ 1 , \ldots , n \} ,$ )   
• fo∈r{every $i \in \{ 1 , \ldots , n \}$ there exists an isomorphism $f _ { i }$ between $G _ { i }$ ∈an{d $G _ { i } ^ { \prime }$ .

If this is the case and $f _ { i } ( v ) = u ,$ , we say that $( v , t _ { i } )$ and $( u , t _ { i } )$ are pointwise isom(or)phi=c.

It turns out that both global and local MP-TGNNs can distinguish pointwise isomorphic nodes. In particular, they can distinguish $( a , t _ { 2 } )$ and $( \bar { a ^ { \prime } } , t _ { 2 } )$ from Figure 7, as we show below using(Theo)rem 3(and T)heorem 6.

Theorem 8. There are temporal graphs T G and $T G ^ { \prime }$ with pointwise isomorphic $( v , t )$ and $( u , t ^ { \prime } )$ such that ${ \bf h } _ { v } ^ { ( 1 ) } ( t ) \neq$ $\mathbf { h } _ { u } ^ { ( 1 ) } ( t ^ { \prime } )$ for some glob(al an)d loca(l MP)-TGNNs.

Proof sketch. Consider $T G$ and $T G ^ { \prime }$ from Figure 7, where $( a , t _ { 2 } )$ is pointwise isomorphic to $( a ^ { \prime } , t _ { 2 } )$ . If we apply rwl to $\kappa _ { \mathsf { g l o b } } ( T G )$ and $\kappa _ { \mathrm { g l o b } } ( T G ^ { \prime } )$ , $\mathsf { r w l } ^ { ( 1 ) } ( a , t _ { 2 } ) \neq \mathsf { r w l } ^ { ( 1 ) } ( a ^ { \prime } , t _ { 2 } )$ , bKecau(se $( a , t _ { 2 } )$ hKas o(ne inc)oming e(dge i)n $\kappa _ { \mathsf { g l o b } } ( T G )$ , bu)t $( a ^ { \prime } , t _ { 2 } )$ h(as no) incoming edges in $\mathcal { K } _ { \mathsf { g l o b } } ( T G ^ { \prime } )$ . (The )same (holds i)f we apply rwl to $K _ { \mathrm { g l o b } } ( T G ^ { \prime } )$ an(d $\kappa _ { \mathsf { l o c } } ( T G ^ { \prime } )$ . So, by Theorem 3 and Theo eKm 6,( there) are glKoba(l and)local MP-TGNNs in which ${ \bf h } _ { a } ^ { ( 1 ) } ( t _ { 2 } ) \neq { \bf h } _ { a ^ { \prime } } ^ { ( 1 ) } ( t _ { 2 } )$ . □

Theorem 8 shows that pointwise isomorphism is unsuitable for detecting node indistinguishability in MP-TGNNs. We obtain an adequate isomorphism notion by, on the one hand, requiring additionally that all $f _ { i }$ mentioned in the definition of pointwise isomorphism coincide but, on the other hand, relaxing the requirement tim $\mathsf { \Omega } _ { : } ( T G ) = \mathsf { t i m e } ( T G ^ { \prime } )$ .

Definition 9. Temporal graphs $T G = ( G _ { 1 } , t _ { 1 } ) , \dots , ( G _ { n } , t _ { n } )$ and $T G ^ { \prime } ~ = ~ ( \bar { G _ { 1 } ^ { \prime } } , \bar { t _ { 1 } ^ { \prime } } ) , \dots , ( \bar { G _ { m } ^ { \prime } } , t _ { m } ^ { \prime } )$ (are tim)ewise(isomor)- phic if both of the following hold:

• $n = m$ and $t _ { i + 1 } - t _ { i } = t _ { i + 1 } ^ { \prime } - t _ { i } ^ { \prime } ,$ , for every $i \in \left\{ 1 , \ldots , n - 1 \right\}$ , • the=re exists a+ f−unct=ion+ $f$ −which is an is∈o{morphism−be}- tween $G _ { i }$ and $G _ { i } ^ { \prime }$ , for every $i \in \{ 1 , \ldots , n \}$ .

If $f ( v ) ~ = ~ u$ , we say that $( v , t _ { i } )$ and $( u , t _ { i } ^ { \prime } )$ are timewise isomorphic, for any $t _ { i } \in \mathsf { t i m e } ( T G )$ .

Next we show that the timewise isomorphism is an adequate notion for timestamped nodes indistinguishability since timestamped nodes which are timewise isomorphic cannot be distinguished by any (global or local) MP-TGNN.

Theorem 10. If $( v , t )$ and $( u , t ^ { \prime } )$ are timewise isomorphic, then $\mathbf { h } _ { v } ^ { ( \ell ) } ( t ) = \mathbf { h } _ { u } ^ { ( \ell ) } ( t ^ { \prime } )$ in an(y M)P-TGNN and any $\ell \in \mathbb { N }$ .

Proof sketch. Assume that $( v , t )$ from $T G$ and $( u , t ^ { \prime } )$ from $T G ^ { \prime }$ are timewise isomorph(ic. H) ence, by Defini(tion )9, $T G$ and $T G ^ { \prime }$ are of the forms $T G = ( G _ { 1 } , t _ { 1 } ) , \dots , ( G _ { n } , t _ { n } )$ and $T G ^ { \prime } = ( G _ { 1 } ^ { \prime } , t _ { 1 } ^ { \prime } ) , \dots , ( G _ { n } ^ { \prime } , t _ { n } ^ { \prime } )$ =as( well a)s $t = t _ { i }$ and $t ^ { \prime } = t _ { i } ^ { \prime }$ for so=m(e $i \in \{ 1 , \ldots , n \}$ . M)oreover, $f ( v ) \ = \ u$ for so=me $f : V ( T G ) \to V ( T G ^ { \prime } )$ s}atisfying requi(re)me=nts in Definitio∶n 9(. We)de→fine( $f ^ { \prime } : \dot { t } \ – n o d e \dot { s } ( T \dot { G } ) \stackrel { - } {  } t \ – n o d e s ( T G ^ { \prime } )$ such that $f ^ { \prime } ( w , t _ { j } ) = ( f ( w ) , t _ { j } ^ { \prime } )$ fo(r all $( w , t _ { j } ) \in t \ – n o d e s ( T G )$ . We can show that $f ^ { \prime }$ is an isomorphism between knowledge graphs $\kappa _ { \mathrm { g l o b } } ( T G )$ and $\mathcal { K } _ { \mathrm { g l o b } } ( T G ^ { \prime } )$ , as well as between $\mathcal { K } _ { \mathrm { l o c } } ( T \breve { G } )$ (and $\mathcal { K } _ { \mathrm { l o c } } ( T G ^ { \vee } )$ . Henc)e, in both cases $f ^ { \prime } ( v , t ) = ( u , t ^ { \prime } )$ implies ${ \sf r w l } ^ { ( \ell ) } ( v , t ) = { \sf r w l } ^ { ( \ell ) } ( u , t ^ { \prime } )$ , for all $\ell \in \mathbb { N }$ . Thus, by Theorem 2 and Theorem 5, we obtain that $\mathbf { h } _ { v } ^ { ( \ell ) } ( t ) = \mathbf { h } _ { u } ^ { ( \ell ) } \dot { ( t ^ { \prime } ) }$ for any global and local MP-TGNNs. □

# Relative Expressiveness of Temporal Message Passing Mechanisms

In this section we will use temporal Weisfeiler-Leman characterisation to prove expressive power results summarised in Figure 2. Our results are on the discriminative (also called separating) power, which aims to determine if a given type of models is able to distinguish two timestamped nodes. Formally, we say that a model distinguishes a timestamped node $( v , t )$ in a temporal graph $T G$ from $( u , t ^ { \prime } )$ in $T G ^ { \prime }$ if this (mod)el computes different embedding  (or $( v , t )$ and $( u , t ^ { \prime } )$ at some layer $\ell$ , that is, $\mathbf { h } _ { v } ^ { ( \ell ) } ( t ) \neq \mathbf { h } _ { u } ^ { ( \ell ) } ( t ^ { \prime } )$ . (We s)ay th(at a ty)pe of models (e.g. global or (oc)al≠MP T(G)NNs) can distinguish $( v , t )$ from $( u , t ^ { \prime } )$ , if some model of this type distinguishes $( v , t )$ from $( u , t ^ { \prime } )$ .

W)e start (by sh)owing that global MP-TGNNs can distinguish timestamped nodes which are indistinguishable by local MP-TGNNs. The reason is that in a global MP-TGNN an embedding of $( v , t )$ can depend on embeddings at $t ^ { \prime } < t$ , but this cannot ha(ppen)in a local MP-TGNN.

Theorem 11. There are timestamped nodes that can be distinguished by global, but not by local MP-TGNNs.

Proof sketch. Consider $( b , t _ { 4 } )$ from $T G$ in Figure 3 and $( b , t _ { 4 } )$ from $T G ^ { \prime }$ in Figu(re 4 ()a). We can show that, for any $\ell \in \mathbb { N }$ , application of $\mathsf { r w l } ^ { ( \ell ) }$ to $\kappa _ { \mathrm { l o c } } ( T G )$ and $\kappa _ { \mathsf { l o c } } ( T G ^ { \prime } )$ assi∈gns the same labels to theseKtim(esta )mped nKode(s. He)nce, by Theorem 5, local MP-TGNNs cannot distinguish these nodes. On the other hand, for any $\ell \geq 1$ , application of $\mathsf { r w l } ^ { ( \ell ) }$ to $\kappa _ { \mathrm { g l o b } } ( T G )$ and $\kappa _ { \mathrm { g l o b } } ( T G ^ { \prime } )$ assigns different labels to thesKe no(des. )There oKre, b(y The)orem 3, global MP-TGNNs can distinguish these nodes. □

Based on the observation from Theorem 11, one could expect that global MP-TGNNs are strictly more expressive than local MP-TGNN. Surprisingly, this is not the case. Indeed, as we show next, there are timestamped nodes which can be distinguished by local, but not by global MP-TGNNs.

Theorem 12. There are timestamped nodes that can be distinguished by local, but not by global MP-TGNNs. This holds true even for colour-persistent temporal graphs.

Proof sketch. Consider $( a , t _ { 2 } )$ from $T G$ and $( a ^ { \prime } , t _ { 2 } )$ from $T G ^ { \prime }$ in Figure 8. Obse v(e tha)t $( a , t _ { 2 } )$ in $\kappa _ { \mathsf { g l o b } } ( T G )$ )is isomorphic to $( a ^ { \prime } , t _ { 2 } )$ in $K _ { \mathrm { g l o b } } ( T G ^ { \prime } )$ so,)by TKheor(em 2,) $( a , t _ { 2 } )$ and $( a ^ { \prime } , t _ { 2 } )$ (canno)t beKdistin(guish)ed by global MP-T(GNNs). How(ever, ) $( a , t _ { 2 } )$ has one outgoing path of length 2 in $\textstyle { \mathcal { K } } _ { \mathrm { l o c } } ( T G )$ ,(but n)ot in $\kappa _ { \mathrm { l o c } } ( T G ^ { \bar { \prime } } )$ . Hence, two iterations of rKwl d(istin)guish these noKdes.(Thus), by Theorem 6, $( a , t _ { 2 } )$ and $( a ^ { \prime } , t _ { 2 } )$ can be distinguished by local MP-TGNNs(. Note) that $T G$ an)d $T G ^ { \prime }$ are colour-persistent. □

Theorem 11 and Theorem 12 show us that neither global or local MP-TGNNs are strictly more expressive, when compared over all temporal graphs. Does the same result hold over colour-persistent graphs? Interestingly, it is not the case: in colour-persistent graphs local MP-TGNNs are strictly more expressive than global MP-TGNNs. Hence Theorem 11 cannot hold for colour-persistent graphs.

![](images/e6ec572653914319331e88999e4809e93d001b36a3970bb68eae5bf5b2abc907.jpg)  
Figure 8: $( a , t _ { 2 } )$ and $( a ^ { \prime } , t _ { 2 } )$ which cannot be distinguished by global (but c)an be (disting)uished by local MP-TGNNs

Theorem 13. In colour-persistent graphs local MP-TGNNs are strictly more expressive than global MP-TGNNs.

Proof sketch. Due to the result established in Theorem 12, it remains to show that over colour-persistent temporal graphs, if $( v , t )$ and $( u , t ^ { \prime } )$ can be distinguished by global MP-TGNN(s, t)hen th(ey ca)n be distinguished also by local MP-TGNNs. Hence, by Theorem 2 and Theorem 6, we need to show the $\mathsf { r w l } _ { \mathsf { g l o b } } ^ { ( \ell ) } ( v , t ) \ \neq \ \mathsf { r w l } _ { \mathsf { g l o b } } ^ { ( \ell ) } ( u , t ^ { \prime } )$ implies $\mathsf { r w l } _ { \mathsf { l o c } } ^ { ( \ell ) } ( v , t ) \neq \mathsf { r w l } _ { \mathsf { l o c } } ^ { ( \ell ) } ( u , t ^ { \prime } )$ for all $\ell \in \mathbb { N }$ . We show this implicat i(on in)d≠uctively(on $\ell$ ,)where the∈inductive step requires proving several non-trivial statements, for example, showing (by another induction) that ${ \sf r w l } ^ { ( \ell ) } ( v , t ) \neq { \sf r w l } ^ { ( \ell ) } ( u , t ^ { \prime } )$ implies ${ \sf r w l } ^ { ( \ell ) } ( v , t + k ) \neq { \sf r w l } ^ { ( \ell ) } ( u , t ^ { \prime } + k )$ , )for≠any $k$ . ( □

To finish the expressive power landscape announced in Figure 2, it remains to make two more observations. On the one hand, temporal graphs which are not colour-persistent allow global MP-TGNNs to distinguish more elements than colour-persistent graphs. Indeed, this is the case since global MP-TGNNs allow us to pass information about colours between nodes stamped with different time points. On the other hand, this is not allowed in local MP-TGNNs, and so colourpersistence does not impact their expressiveness.

# Experiments

We implement and train basic variants of global and local models on standard temporal link-prediction benchmarks. We emphasise that the goal of our experiments is not to achieve models with high-level performance, but to examine how our expressive power results impact practical performance of MP-TGNNs.

Benchmarks. We use the Temporal Graph Benchmark (TGB) 2.0 suite (Gastinger et al. 2024) with small-tomedium temporal datasets tgbl-wiki, tgbl-review, and tgbl-coin, whose statistics are in Table 1. They do not have node features and we discard the edge features. We consider a link-prediction task, where the goal is to predict whether there is a link between two given nodes at the next time point, given information about all previous links. We follow normative training and evaluation procedures supplied by TGB.

Models. We implement global and local MP-TGNNs with combination and aggregation functions being concatenation $( \parallel )$ and summation $\left( \sum \right)$ , respectively, which are among stand∣a∣rd choices (Rossi∑and Ahmed 2015; Xu et al. 2019). Hence, embedding ${ \bf h } _ { v } ^ { ( \ell ) } ( t )$ is computed as

$$
W _ { 2 } ^ { ( \ell ) } [ \mathbf { h } _ { v } ^ { ( \ell - 1 ) } ( t ) \left| \right| \sigma ( W _ { 1 } ^ { ( \ell ) } ( \sum _ { ( u , t ^ { \prime } ) \in \mathcal { N } ( v , t ) } \star \left| \right| g ( t - t ^ { \prime } ) ) ) ] ,
$$

where $W _ { 1 }$ and $W _ { 2 }$ are learnable, $\sigma$ is the rectified linear unit, $\bigstar$ is either $\mathbf { h } _ { u } ^ { ( \ell - 1 ) } ( t ^ { \prime } )$ (giving rise to a global model) or $\mathbf { h } _ { u } ^ { ( \ell - 1 ) } ( t )$ (local mod(el), and $g \big ( t - t ^ { \prime } \big ) = t - t ^ { \prime }$ . After $\ell = 4$ layers, a m(u)lti-layer perceptron w(ith−10)2=4 hi−dden units p=redicts a link between $u$ and $v$ , given ${ \bf h } _ { u } ^ { ( \ell ) } ( t )$ and ${ \bf h } _ { v } ^ { ( \ell ) } ( t )$ . Aggregating the entire temporal neighbo r(h)ood incu (ov)er time a linear computational penalty, precluding larger benchmarks; real-world models approximate this calculation.

Implementation. Our implementation2 is based on PyTorch (Paszke et al. 2019), and in particular its hardwareaccelerated scatter operations for temporal aggregation. The use of scatter operations means that results may differ between hardware-accelerated runs.

As stated, the global model would require enormous compute and memory in order to use node embeddings from all previous time points. In order to make this tractable, we apply a train-time approximation: during an epoch, embeddings from previous timepoints are “frozen”: detached from the computation graph and not updated as model weights change. This interferes with training as the model must use a mixture of stale and fresh embeddings, but at test time the result is exact. A further observation is that only those embeddings whose nodes are connected at some time need be computed and retained due to the definition of $\mathcal { N }$ .

Minibatching can be achieved in the temporal context by predicting the next $k$ links given all previous links. Unlike traditional minibatching, this can have a detrimental impact on model accuracy, because earlier links in the batch may help to predict links later in the batch. However, it is computationally very demanding to set $k = 1$ for even “small” datasets like tgbl-review contain =g millions of links, so a compromise must be found. We set $k = 3 2$ for tgbl-wiki and $k = 1 0 2 4$ for all others.

Training We used Adam (Kingma and Ba 2015) for optimisation with PyTorch defaults $\gamma ~ = ~ 0 . 0 0 1$ , $\beta _ { 1 } ~ = ~ 0 . 9$ , $\beta _ { 2 } = 0 . 9 9 9$ , and no L2 penalty. We w e= able to sign i=cantly stab=ilise and accelerate training by normalising $g ( { \bar { t } } - t ^ { \prime } )$ with respect to elapsed time and by applying batch(no−rm)alisation (Ioffe and Szegedy 2015) immediately after summation of temporal neighbours. With the exception of the above stability measures, we have not tuned further as we are not aiming for state-of-the-art performance. Training continued until validation loss failed to improve for 10 epochs. Experiments involving tgbl-wiki and tgbl-review can be run on desktop hardware (NVIDIA GT730), or even without acceleration, whereas tgbl-coin requires a large GPU.

Table 1: Statistics (nodes and edges) and MRR scores   

<html><body><table><tr><td>tgbl-wiki</td><td></td><td>tgbl-review|tgbl-coin</td><td></td></tr><tr><td>nodes edges</td><td>9,227 157,474</td><td>352.637 4,873,540</td><td>638,486 22,809,486</td></tr><tr><td>global local</td><td>0.223 0.264</td><td>0.321 0.359</td><td>0.628 0.635</td></tr></table></body></html>

Results. Table 1 shows the mean reciprocal rank (MRR) score (higher is better) used in TGB. We observe that the scores are relatively high given the simplicity of models and lack of tuning. We have written in bold higher among MRRs obtained by global and local MP-TGNNs. In all three datasets, local MP-TGNN obtains higher scores, but the difference between the scores of local and global models is relatively small. We observe that higher performance of the local model aligns with our theoretical result from Theorem 13, which states that over colour-persistent temporal graphs (as here), local MP-TGNNs are stricly more expressive than global MP-TGNNs.

Layers. We also investigated the effect of the increasing number of layers $\ell$ on MRR. We performed experiments on tgbl-wiki with the number of layers increasing from 1 to 8. As presented in Figure 9, the highest MRR for the global model is obtained when $\ell = 5$ and for the local model when $\ell = 7$ . Interestingly, for $\ell = 5$ (which is optimal for the globa=l model), MRR for both =models is almost the same. This, again, aligns with our theoretical results, showing that local MP-TGNNs are more expressive than global.

![](images/b5b066f66bccddeeb6124886eabe8a024cb7953c74a91d76d280f9f1e7f9f5ac.jpg)  
Figure 9: MRR against number of layers on tgbl-wiki

Variations. The lack of node features in benchmarks led us to try both random node features, which did not significantly alter results, and (transductively) learnable node features, which caused drastic overfitting.

# Conclusions

We have categorised temporal message-passing graph neural networks into global and local, depending on their temporal message passing mechanism. One might expect that global models have higher expressive power than local, but surprisingly we find that the two are incomparable. Further, if node colours (feature vectors) do not change over time, local models are strictly more powerful than global. Our experimental results align with the theoretical findings, showing that local models obtain higher performance on temporal link-prediction tasks.