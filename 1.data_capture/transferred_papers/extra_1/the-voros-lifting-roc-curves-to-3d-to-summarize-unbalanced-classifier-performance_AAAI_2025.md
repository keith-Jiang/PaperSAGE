# The VOROS: Lifting ROC Curves to 3D to Summarize Unbalanced Classifier Performance

Christopher Ratigan1, Lenore Cowen1, 2

1 Department of Mathematics, Tufts University, Medford, MA 02155 USA 2 Department of Computer Science, Tufts University, Medford, MA 02155 USA christopher.ratigan $@$ tufts.edu, cowen $@$ cs.tufts.edu

# Abstract

While the area under the ROC curve is perhaps the most common measure that is used to rank the relative performance of different binary classifiers, longstanding field folklore has noted that it can be a measure that ill-captures the benefits of different classifiers when either the actual class values or misclassification costs are highly unbalanced between the two classes. We introduce a new ROC surface, and the VOROS, a volume over this ROC surface, as a natural way to capture these costs, by lifting the ROC curve to 3D. Compared to previous attempts to generalize the ROC curve, our formulation also provides a simple and intuitive way to model the scenario when only ranges, rather than exact values, are known for possible class imbalance and misclassification costs.

# Code — https://github.com/ChristopherRatigan/VOROS

# Introduction

The ROC curve, constructed by plotting the False Positive Rate (FPR) against the True Positive Rate (TPR), is the most canonical way to measure the performance of a binary classifier.

Definition 1. Given an ROC curve $y = f ( x ) ,$ , the area under the ROC curve, henceforth AUROC, is defined by

$$
\int _ { 0 } ^ { 1 } f ( x ) d x
$$

As has long been known (Hand 2009), automatically choosing the binary classifier with the higher AUROC makes perfect sense when two rather strong conditions are met: 1) the expected ratio of positives to negatives in the dataset where the classifier is to be deployed is balanced or nearly balanced, and 2) the cost of misclassifying a positive example as negative is equal or nearly equal to the cost of misclassifying a negative example as positive, i.e. that $C _ { 0 } / C _ { 1 } \approx 1$ and $| \mathcal { P } | / | \mathcal { N } | \approx 1$ , where we denote by $\mathcal { P }$ the set of observations that are positive and $\mathcal { N }$ denotes the set of observations that are negative. $C _ { 0 }$ and $C _ { 1 }$ denote the cost of misclassifying a case from $\mathcal { N }$ as belonging to $\mathcal { P }$ and the converse, respectively. The area under the ROC curve of a binary classifier $\mathcal { F }$ represents the probability that given two observations $\boldsymbol { \rho } \in \mathcal { P }$ and $\eta \in \mathcal { N }$ , $\mathcal { F }$ assigns a higher score to $\rho$ than it does to $\eta$ (Bradley 1997).

But, if the costs or classes are not balanced, then it is not always sensible to directly compare the AUROCs in this pairwise manner. If one class has many more cases than the other, comparing a random positive and negative example is unlikely to occur in practice. Likewise, if the two types of errors differ in individual costs then it may not be as important that it is possible to separate a randomly chosen positive example from a randomly chosen negative example.

Observe that when exact information is available on the ratios $C _ { 0 } / C _ { 1 }$ and $| \mathcal { P } | / | \mathcal { N } |$ , it is always possible to find a binary classifier (point) taken from the ROC curve of classifier $\mathcal { F } _ { 1 }$ that minimizes the total misclassification cost, and a binary classifier taken from the ROC curve of classifier $\mathcal { F } _ { 2 }$ that minimizes the total misclassification cost, and then choose whichever classifier has lower total misclassification cost. However, as we (see example datasets, below) and others demonstrate (Provost, Fawcett, and Kohavi 1998; Hand 2009; Drummond and Holte 2000)), such a cost comparison no longer corresponds to a comparison of the AUROCs. In fact, classifiers with better AUROCs can perform much worse in total misclassification cost in these settings.

Much previous work has attempted to deal with the shortcomings of the AUROC by looking at different notions of a weighted AUROC (McClish 1989; Maurer and Pontil 2020; Li and Fine 2010); however, most of these formulations require fixing the cost distribution and class imbalance as a prior. (Two notable exceptions are (Herna´ndez-Orallo, Flach, and Ramirez 2011) and the recent paper of (Shao et al. 2024), that takes an interesting (and complementary approach) to solving the problem we address in this paper, namely simultaneously representing different weighted costs)). Versions of weighted AUROC that apply for a fixed cost distribution and class imbalance do not easily extend to model uncertainty in the cost distribution or only bounds on the class imbalance (though we note that a previous paper of Guo et al. did define a “3D ROC histogram,” with a general 3rd axis that could represent either classifier confidence or cost; where the cost version is closest in spirit to the present work (Guo, Shen, and Zhang 2019)).

By instead making something similar to the weighted AUROC a slice of a ROC volume, lifted to 3D, we are able to model ranges of misclassification costs and class imbalance.

We introduce a new ROC surface, where the volume integrated over this ROC surface becomes a generalization of the AUROC where misclassification costs are also naturally represented. Besides mathematical elegance, the advantage of our VOROS to previous weighted AUROC approaches is that we also have a very natural way to ask for the better classifier in a regime where the application domain gives only bounds on the expected class imbalances or misclassification costs, and does not assume these values are known exactly. We show that the VOROS is efficient to compute, and show in several benchmark datasets, that it will choose better classifiers, according to how costs and class imbalances are estimated to occur in the environment in which the classifier will be deployed.

We demonstrate the utility of this approach on two standard benchmark classification datasets, the UCI Wisconsin Breast Cancer Dataset (William et al. 1995) and BUSI (Al-Dhabyani et al. 2020), as well as a highly imbalanced credit card fraud dataset.

# Background and Definitions

Definition 2 (ROC Space). Fix a dataset $\chi$ consisting of observations assigned the actual class labels positive (1) and negative (0). For a binary classifier $\mathcal { F }$ , the performance of $\mathcal { F }$ on $\chi$ is represented as a point in ROC space where $\mathcal { F }$ is mapped to the point $( x , y )$ with $x$ the false positive rate and $y$ the true positive rate of $\mathcal { F }$ on the dataset $\chi$ .

An intuitive rule in ROC space is that one classifier does better than another classifier if it is above and to the left as the following definition makes precise.

Definition 3. A point $( x _ { 1 } , y _ { 1 } )$ dominates another point $( x _ { 2 } , y _ { 2 } )$ in ROC space if it has a lower false positive rate and higher true positive rate, i.e. if $x _ { 1 } \leq x _ { 2 }$ and $y _ { 1 } \geq y _ { 2 }$ .

There are a few points in ROC space that are important for our analysis. A perfect classifier, which correctly classifies all observation in $\chi$ has ROC coordinates $( 0 , 1 )$ . Likewise, the point $( 1 , 0 )$ represents a model which misclassifies every case. We complete the unit square in ROC space by defining the baseline classifiers.

Definition 4. The baseline classifiers $\boldsymbol { B } _ { 0 }$ and $\boldsymbol { B } _ { 1 }$ are the classification models that label all observations as negative or positive respectively.

There is a traditional way of going from points in ROC space to a curve for classifiers that can output a numerical score interpreted as the probability that an observation belongs to the positive class. Simply vary a threshold $s$ which divides the scores of the observations into positive and negative classes based on whether the score of a particular example is greater than or less than $s$ (Metz 1978; Swets 1988).

Definition 5. Given a classifier $\mathcal { F } : \mathcal { X }  [ 0 , 1 ]$ , let $\delta _ { \mathcal { N } } ( s )$ be the cumulative distribution of observations in the negative class and $\delta _ { \mathcal { P } } ( s )$ be that of the positive class. Then, the observed ROC curve for $\mathcal { F }$ is a plot of $( 1 - \delta _ { \mathcal { N } } ( s ) , 1 - \delta _ { \mathcal { P } } ( s ) )$ .

ROC curves inherit the following natural notion of dominance from the domination of individual points.

Definition 6. Given two ROC curves $f$ and $g$ , $f$ dominates $g$ if for every point $P$ on the graph of $g$ there is some point $Q$ on the graph of $f$ such that $Q$ dominates $P$ and no point on the graph of $f$ is strictly dominated by any point on the graph of $g$ .

Definition 7. The Upper Convex Hull of a collection of $n$ points $\mathcal { C } = \{ ( x _ { i } , y _ { i } ) \bar  \} _ { i = 1 } ^ { n } ,$ , in $R O C$ space is the boundary of the convex hull of $\mathcal { C }$ together with the additional points $( 0 , 0 )$ , $( 1 , 1 )$ and $( 1 , 0 )$ running from $( 1 , 1 )$ to $( 0 , 0 )$ oriented counterclockwise.

While the definition of ROC curve intuitively creates a continuous curve in ROC space for continuous distributions $\delta _ { \mathcal { N } }$ and $\delta _ { \mathcal { P } }$ , it is common to take the discrete set of classifiers given by varying the threshold parameter on a finite dataset, connect them piecewise by lines, and call the resulting graph a “curve” (Drummond and Holte 2006). In what follows, we will focus on this discretized case, though much of this generalizes to continuous “fitted” ROC curves.

We note that for such piecewise linear ROC curves $f$ and $g$ , the vertices of $f$ dominate the vertices of $g$ if and only if the graph of $f$ lies above the graph of $g$ over the entire interval $[ 0 , 1 ]$ . In this way, domination in ROC space is equivalent to the usual domination of real valued functions.

Lemma 8. The Upper Convex Hull of an ROC curve dominates the ROC curve.

Proof. A full proof is given in (Provost and Fawcett 2001).

It has long been known that in ROC space, it is possible to compare the expected costs of classification models to decide which point on an ROC curve, or a collection of such curves is optimal for a given weighting of the relative numbers and costs of false positives and false negatives. The following definition helps to quantify this notion of cost.

Definition 9. Let $\chi = \mathcal { P } \sqcup \mathcal { N }$ be the set of observations to be classified, where $\mathcal { P }$ denotes observations whose actual label is positive and likewise, $\mathcal { N }$ denotes negative observations. We define $C _ { 0 }$ and $C _ { 1 }$ to be the costs of an individual false positive and false negative respectively. Additionally, given a point $( x , y )$ in ROC space, we define the expected cost of $( x , y )$ to be

$$
C _ { 0 } | \mathcal { N } | x + C _ { 1 } | \mathcal { P } | ( 1 - y )
$$

The point in ROC space with the maximal expected cost is $( 1 , 0 )$ which misclassifies every case. We can scale our cost by this maximum so that costs lie in the unit interval putting the costs of classifiers on a normalized scale.

Definition 10. The normalized expected cost of a point $( x , y ) = ( F P R , T P R )$ is given by

$$
C o s t ( x , y , t ) = t x + ( 1 - t ) ( 1 - y )
$$

where $t = \frac { C _ { 0 } | \mathcal { N } | } { C _ { 0 } | \mathcal { N } | + C _ { 1 } | \mathcal { P } | }$ is the portion of cost borne by the false positives.

This formulation of cost is well-known, though there has not been a universally accepted way to apply it to ROC curves in their entirety (Provost, Fawcett, and Kohavi 1998).

Lemma 11. Let $P ( x _ { 1 } , y _ { 1 } )$ and $Q ( x _ { 2 } , y _ { 2 } )$ be two points in ROC space and let $t \in [ 0 , 1 ]$ be fixed. Then, $C o s t ( x _ { 1 } , y _ { 1 } , t ) =$ $C o s t ( x _ { 2 } , y _ { 2 } , t )$ if and only if the slope of the line between $P$ and Q is 1 t .

Proof. The relevant equation is

$$
t x _ { 1 } + ( 1 - t ) ( 1 - y _ { 1 } ) = t x _ { 2 } + ( 1 - t ) ( 1 - y _ { 2 } )
$$

Rearranging yields

$$
t ( x _ { 1 } - x _ { 2 } ) = ( 1 - t ) ( y _ { 1 } - y _ { 2 } )
$$

Dividing by $( 1 - t ) ( x _ { 1 } - x _ { 2 } )$ on both sides yields the desired result

$$
m = { \frac { t } { 1 - t } } = { \frac { y _ { 1 } - y _ { 2 } } { x _ { 1 } - x _ { 2 } } }
$$

Where if $t = 1$ , we let $m = \infty$ define a vertical line.

Definition 12. Given a fixed value of $t \in \ [ 0 , 1 ]$ an isoperformance line through $( x _ { 1 } , y _ { 1 } )$ is the line through $( x _ { 1 } , y _ { 1 } )$ in ROC space with slope $\frac { t } { 1 - t }$ representing all points with the same cost as $( x _ { 1 } , y _ { 1 } )$ .

Intuitively, iso-performance lines are the equivalence classes of points in ROC space with the same cost for a fixed value of $t$ . These lines instill a linear order which divides a classifier from better performing classifiers above and to the left of the line (but not necessarily the point) and worse performing classifiers below and to the right of the iso-performance line.

Lemma 13. Let $t \in [ 0 , 1 ]$ be fixed and let $y \ = \ f ( x )$ be an ROC curve. A point $( h , f ( h ) )$ has the minimum cost of any point on the ROC curve if the upper convex hull of the curve lies below and to the right of the iso-performance line through $( h , f ( h ) )$ with slope $m = { \frac { t } { 1 - t } }$ .

Proof. Let $P ( h , f ( h ) )$ be such as point, then points with a lower cost than $P$ lie above the line through $P$ with slope $\frac { t } { 1 - t }$ but if $P$ achieves the minimal cost of any point on the ROC curve, all points on the ROC curve must lie below and to the right of the iso-performance line as required. □

The term for these extremal iso-performance lines in convex geometry is “supporting lines” and the space of all such lines is the ROC Surface as explained below.

Definition 14. Given a region $R$ in the plane, a supporting line $\ell$ of $R$ is a line intersecting $R$ such that all of $R$ lies on the same side of $\ell$ .

Definition 15. Let $t \in [ 0 , 1 ]$ , an optimal point on an ROC curve is a point on the upper convex hull that has a supporting line of slope 1 t .

Note, it is possible for an ROC curve to have multiple optimal points. Such points occur when the upper convex hull contains a line segment, any point of which is optimal for $t$ such that $\frac { t } { 1 - t }$ is the slope of the segment.

Not only can supporting lines be used to pick out the optimal classifier for a given cost parameter $t$ , they also partition ROC space according to cost. The area of classifiers which cost more provides a natural ordering which is explicitly related to the expected cost. This stands in contrast to the area under the ROC curve defined in the introduction.

# Main Contribution

To connect our notion back to the area under the ROC curve, we now generalize this notion to a cost-aware space by defining the area of lesser classifiers.

Definition 16. Given $t \in [ 0 , 1 ]$ , and a point $( h , k )$ in ROC space, the area of lesser classifiers of $( h , k )$ , dependent on $t$ , denoted $A _ { t } ( h , k )$ is the area of the points $( h ^ { \prime } , k ^ { \prime } )$ in $R O C$ space, such that $( h ^ { \prime } , k ^ { \prime } )$ has a greater expected cost than $( h , k )$ . Define such $( h ^ { \prime } , k ^ { \prime } )$ to be lesser classifiers to $( h , k )$ in this case.

Note if we choose $( h , k )$ to be an optimal point on our curve, then the region containing lesser classifiers will always contain the area under the ROC curve, but it also typically includes additional area in the region above the ROC curve. As we see with the Credit Card Fraud dataset analyzed below, a classifier with a worse AUROC can have a better area of lesser classifiers value for ranges of unbalanced classification class sizes or misclassification costs.

We are now ready to define the ROC surface, which will allow us to look at the area of lesser classifiers in particular ranges of class imbalance or cost imbalance.

Definition 17. The ROC surface, ROS, over a point $( h , k )$ in ROC space is the surface, in $( x , y , t )$ -space, given by

$$
y = { \frac { t } { 1 - t } } ( x - h ) + k
$$

where $t \in [ 0 , 1 ]$

The ROS parameterizes all lines through $( h , k )$ with nonnegative (possibly infinite) slope.

Lemma 18. The ROC surface over a point $( h , k )$ is a saddle surface with the equation

$$
t = { \frac { Y } { X + Y } }
$$

where $Y = y - k$ and $X = x - h$ are simply horizontal shifts of the ROC surface over the origin.

Proof. The proof follows by solving for $t$ in the formula for the ROC surface. □

The ROC surface provides a natural coordinate system for assessing relative costs over a range of parameter values $t$ since if $t$ is a fixed constant, then for the ROC points $( x _ { 1 } , y _ { 1 } )$ and $( x _ { 2 } , y _ { 2 } )$ , we need only compare the parallel lines

$$
y = { \frac { t } { 1 - t } } ( x - x _ { 1 } ) + y _ { 1 } \operatorname { a n d } y = { \frac { t } { 1 - t } } ( x - x _ { 2 } ) + y _ { 2 }
$$

In particular, setting $x = 0$ yields $y _ { i } - { \frac { t } { 1 - t } } x _ { i }$ . The classifier with the lower cost at this threshold then has a higher value of yi − $y _ { i } - { \frac { t } { 1 - t } } x _ { i }$

Definition 19. The ROC Surface associated to an ROC curve $y = f ( x )$ , is the surface in $( x , y , t )$ -space which for each value of t is the maximum of the lines in ROC space through points (p, f (p)) with slope 1 t t .

This surface divides the unit cube, $[ 0 , 1 ] ^ { 3 }$ , into two sets. When viewed from the vertical line $x = 0 , y = 1$ , points $P ( x , y , t )$ behind the ROC surface map to points $( x , y )$ in ROC space such that the cost of $( x , y )$ is greater than that of the optimal point on the ROC curve at that $t$ value. Similarly, points $P ( x , \bar { y } , t )$ in front of the ROC surface represent points which perform better than the optimal point on the ROC curve at the associated $t$ value.

This gives a natural notion of performance for the whole ROC curve: the volume over the ROC surface.

Definition 20. The volume over the ROC Surface, VOROS, is defined by

$$
{ V O R O S } ( f ( x ) ) = \int _ { 0 } ^ { 1 } \operatorname* { m a x } _ { x \in [ 0 , 1 ] } ( A _ { t } ( x , f ( x ) ) ) d t
$$

Where $m a x ( A _ { t } ( x , f ( x ) ) )$ is the maximal area of lesser classifiers for any point on the $R O C$ curve taken for each $t$ .

One of the useful properties of VOROS is that it can be readily modified to handle ranges and distributions over the relative costs of misclassifications. This is useful because in many applications, it is reasonable to assume one might have bounds on the expected cost imbalances, but not know them exactly. We can capture this uncertainty in our VOROS measure as follows:

Definition 21. Let $\mu$ be a probability measure for $t \in [ 0 , 1 ]$ , let $[ a , b ] \subset [ 0 , 1 ]$ and let $A _ { t } ( x , f ( x ) )$ be the area of lesser classifiers associated to a point on the $R O C$ curve $y = f ( x )$ , then

$$
\begin{array} { r l r } {  { V O R O S ( f ( x ) , [ a , b ] , \mu ) } } \\ & { } & { = \frac { 1 } { \mu ( [ a , b ] ) } \int \displaylimits _ { x \in [ 0 , 1 ] } ^ { \mathrm { m a x } } ( A _ { t } ( x , f ( x ) ) ) d \mu ( t ) } \end{array}
$$

This is a generalization of our initial definition of VOROS (Definition 20) since $V O R O S ( f ( x ) , [ 0 , 1 ] , \mu )$ is simply $V O R O S ( f ( x ) )$ if $\mu$ is the standard measure on $\mathbb { R }$ . In what follows we omit $\mu$ assuming it to be the standard measure on $\mathbb { R }$ .

Lemma 22. Given an ROC point $( h , k )$ , and fixed costs $( t , 1 - t )$ , $i f ( h , k )$ outperforms both baseline classifiers, then the area of lesser classifiers for $( h , k )$ , $A _ { t } ( h , k )$ , has the following form.

$$
1 + { \frac { ( 1 - k ) ^ { 2 } } { 2 } } + { \frac { h ^ { 2 } } { 2 } } - h ( 1 - k ) - { \frac { ( 1 - k ) ^ { 2 } } { 2 t } } - { \frac { h ^ { 2 } } { 2 ( 1 - t ) } }
$$

![](images/32e88b6192e95850c9a3f54f80b23f1346e65e07e9f94f84f226e3713d8d7f8e.jpg)  
Figure 1: The area of Lesser Classifiers for the point $( h , k )$ lies below the iso-performance line with slope $\begin{array} { r } { m = \frac { t ^ { \star } } { 1 - t } } \end{array}$ .

Proof. Let $\begin{array} { r } { m = \frac { t } { 1 - t } } \end{array}$ be the slope of the iso-performance line. First, note that since $( h , k )$ outperforms the baseline classifiers, the area of lesser classifiers will be of the general form of the shaded region in Figure 1.

This is because the line bounding this region is above and to the left of the parallel lines through $( 0 , 0 )$ and $( 1 , 1 )$ . Thus, the dividing line for lesser classifiers must intersect the $y$ -axis above $( 0 , 0 )$ and intersect the line $y = 1$ to the left of $( 1 , 1 )$ .

As such, the area of this region is given by one minus the area of the white triangle. The horizontal base of the triangle is $\begin{array} { r } { B = \frac { 1 - k } { m } + h } \end{array}$ while the height of the triangle is $H = 1 - k + m h$ . Hence, the area of the shaded region is

$$
\begin{array} { l } { \displaystyle { A _ { t } ( h , k ) = 1 - \frac { 1 } { 2 } B H } } \\ { \displaystyle { \quad = 1 - \frac { 1 } { 2 } \left( \frac { 1 - k } { m } + h \right) ( 1 - k + m h ) } } \\ { \displaystyle { \quad = 1 - \frac { 1 } { 2 } \left( \frac { ( 1 - k ) ^ { 2 } } { m } + 2 ( 1 - k ) h + m h ^ { 2 } \right) } } \\ { \displaystyle { \quad = 1 - \frac { ( 1 - k ) ^ { 2 } } { 2 m } - h ( 1 - k ) - \frac { m h ^ { 2 } } { 2 } } } \end{array}
$$

Since $m = { \frac { t } { 1 - t } } = - 1 + { \frac { 1 } { 1 - t } }$ the area of lesser classifiers is then

$$
1 + { \frac { ( 1 - k ) ^ { 2 } } { 2 } } + { \frac { h ^ { 2 } } { 2 } } - h ( 1 - k ) - { \frac { ( 1 - k ) ^ { 2 } } { 2 t } } - { \frac { h ^ { 2 } } { 2 ( 1 - t ) } }
$$

Integrating this last expression yields the following result

Lemma 23. Given a vertex $v = ( h , k )$ from the upper convex hull of an ROC curve $\{ x _ { i } , y _ { i } \} _ { i = 0 } ^ { n }$ . If $v$ is an optimal point on the cost interval $[ a , b ] \subset [ 0 , 1 ]$ , then, the volume of lesser classifiers for the ROC curve on the interval $( a , b )$ is given by

$$
\begin{array} { l } { \displaystyle \frac 1 { b - a } \left( 1 + \frac { ( 1 - k ) ^ { 2 } } 2 + \frac { h ^ { 2 } } 2 - h ( 1 - k ) \right) } \\ { \displaystyle \qquad + \frac { ( 1 - k ) ^ { 2 } } 2 l n \left( \frac { 1 - b } { 1 - a } \right) - \frac { h ^ { 2 } } 2 l n \left( \frac b a \right) } \end{array}
$$

Note that Lemma 23 gives an explicit way to compute the VOROS between points in the upper convex hull of ROC space, which will contain at most $n$ points. Thus the VOROS can be calculated explicitly in time on the order of the time it takes to calculate the convex hull of $n$ points. In our implementation we use the Quickhull algorithm (Bykat 1978; Eddy 1977) which runs in $O ( n \log n )$ expected time.

There is a direct relationship between the minimum normalized expected cost of a classifier and the volume integrated over the ROC surface.

Theorem 24. Let $( h , k )$ be the optimal point on an ROC curve for $t \in [ a , b ]$ . If $c _ { t } ( h , k )$ is the normalized expected cost of this point, then the volume over the ROC surface is simply

$$
{ \frac { 1 } { b - a } } \int _ { a } ^ { b } 1 - { \frac { \left( c _ { t } ( h , k ) \right) ^ { 2 } } { 2 t ( 1 - t ) } } d t
$$

Proof. Since the volume over the ROC surface is simply the integral of the area bounded by a classifier at each threshold, all we need to show is that for each value of $t$ , we have the formula

$$
A _ { t } ( h , k ) = 1 - \frac { c _ { t } ( h , k ) ^ { 2 } } { 2 t ( 1 - t ) }
$$

From earlier, we know that the left hand side is

$$
1 + { \frac { ( 1 - k ) ^ { 2 } } { 2 } } + { \frac { h ^ { 2 } } { 2 } } - h ( 1 - k ) + { \frac { ( 1 - k ) ^ { 2 } } { 2 t } } - { \frac { h ^ { 2 } } { 2 ( 1 - t ) } }
$$

The right hand side is simply,

$$
\begin{array} { l } { { 1 - \displaystyle \frac { c _ { t } ( h , k ) ^ { 2 } } { 2 t ( 1 - t ) } = 1 - \displaystyle \frac { ( t h + ( 1 - t ) ( 1 - k ) ) ^ { 2 } } { 2 t ( 1 - t ) } } } \\ { { = 1 - \displaystyle \frac { t ^ { 2 } h ^ { 2 } + 2 t ( 1 - t ) h ( 1 - k ) + ( 1 - t ) ^ { 2 } ( 1 - k ) ^ { 2 } } { 2 t ( 1 - t ) } } } \\ { { = 1 - \displaystyle \left( \frac { t h ^ { 2 } } { 2 ( 1 - t ) } + h ( 1 - k ) + \displaystyle \frac { ( 1 - t ) ( 1 - k ) ^ { 2 } } { 2 t } \right) } } \\ { { = 1 - \displaystyle \left( \frac { h ^ { 2 } } { 2 ( 1 - t ) } + h ( 1 - k ) + \displaystyle \frac { ( 1 - k ) ^ { 2 } } { 2 t } - \displaystyle \frac { ( 1 - k ) ^ { 2 } } { 2 } \right) } } \end{array}
$$

where in the last equation we used that ${ \frac { t } { 1 - t } } = - 1 +$ $\frac { 1 } { 1 - t }$ From here distributing the negative and rearranging terms yields the desired result. □

![](images/4098881a8b0476480a95e96c5b34f55e70b2991acd977e09c003106226b13dbe.jpg)  
Figure 2: The two possibilities for the area of lesser classifiers for the better of the baselines.

![](images/923350b685774285102ebc97c288e9a9806776fb58197603090f8ea2514dbb08.jpg)  
Figure 3: (Left) Graph of $A _ { t } ( \{ ( 0 , 0 ) , ( 1 , 1 ) \} )$ . (Right) The Volume bounded by these areas.

This theorem shows an important feature of the VOROS. Given two ROC Surfaces $S _ { 1 }$ and $S _ { 2 }$ associated to ROC curves $f _ { 1 }$ and $f _ { 2 }$ , if for all $t \in [ a , b ] \subseteq [ 0 , 1 ]$ we have that $S _ { 1 }$ has a lower cost than $S _ { 2 }$ , then $\bar { \mathrm { V O R O S } } ( \bar { f } _ { 1 } , [ a , b ] , \mu ) >$ $\operatorname { V O R O S } ( f _ { 2 } , [ a , b ] , \mu )$ since the associated integrands have the same inequality.

There is a natural probabilistic interpretation of the VOROS: it is the probability that choosing $x , y \in [ 0 , 1 ]$ uniformly and independently at random and $t$ independently according to $\mu$ , the randomly chosen point $( x , y )$ has a higher cost than an optimal point of the ROC curve. This is particularly useful for the problem of comparing performance of classifiers in cost or class imbalanced problems because if we can estimate the cost or class imbalance, we get a more accurate measure of classifier performance than from the AUROC alone.

For example, the AUROC of the hybrid classifier consisting of the two baselines $\boldsymbol { B } _ { 0 }$ and $\boldsymbol { B } _ { 1 }$ is $\frac { \mathbf { i } } { 2 }$ , but picking the better of the baseline classifiers will always yield a classification model which has less than $\textstyle { \frac { 1 } { 2 } }$ the maximal expected cost so long as false positives and false negatives are unequal in aggregate cost.

Example 25. Consider the ROC curve consisting of the vertices $\{ ( 0 , 0 ) , ( 1 , 1 ) \}$ . The Volume over the ROC surface for this curve is $\begin{array} { r } { \frac { 3 } { 2 } - \dot { l n } ( 2 ) \approx . 8 0 7 . } \end{array}$ .

Set $\begin{array} { r } { m = \frac { t } { 1 - t } } \end{array}$ . Then picking the better classifier gives us either of the areas in Figure 2. Either way, the area of lesser classifiers is one minus the area of the white triangle.

$$
A _ { t } = 1 - \frac { 1 } { 2 } B H = \left\{ \begin{array} { l l } { { 1 - \displaystyle \frac { 1 } { 2 m } } } & { { m > \frac { 1 } { 2 } } } \\ { { 1 - \displaystyle \frac { m } { 2 } } } & { { m < \frac { 1 } { 2 } } } \end{array} \right.
$$

Table 1: Table of VOROS and AUROC values for classifiers trained on the Wisconsin Breast Cancer dataset.   

<html><body><table><tr><td></td><td>VOROS([0,1])</td><td>AUROC</td></tr><tr><td>Log</td><td>99.8%</td><td>99.3%</td></tr><tr><td>Forest</td><td>99.8%</td><td>99.0%</td></tr><tr><td>Naive</td><td>98.4%</td><td>95.1%</td></tr><tr><td>Baseline</td><td>80.7%</td><td>50%</td></tr></table></body></html>

![](images/19d10496d2bf9598ea5b0bb580169bff980f494da37a4cc88c086d030430b2d4.jpg)  
Figure 4: (a) A plot of the ROC curves for the Wisconsin Breast Cancer dataset. (b) A plot of the ROC curves for the BUSI dataset. The solid curves are from Logistic Regression, the dashed curves are from Random Forests and the dotted curves are from Naive Bayes.

Substituting the expression $\begin{array} { r } { m = \frac { t } { 1 - t } } \end{array}$ gives us the integral.

$$
\int _ { 0 } ^ { 1 } A _ { t } d t = \int _ { 0 } ^ { \frac { 1 } { 2 } } 1 - \frac { t } { 2 ( 1 - t ) } d t + \int _ { \frac { 1 } { 2 } } ^ { 1 } 1 - \frac { ( 1 - t ) } { 2 t } d t
$$

A graph of $A _ { t } \left( \{ ( 0 , 0 ) , ( 1 , 1 ) \} \right)$ as a function of $t$ next to the volume as a region in $( x , y , t )$ -space is shown in Figure 3. The minimum value of the graph of $A _ { t }$ is 0.5, which occurs when the positive and negativ1e classes have equal aggregate cost, i.e. $\begin{array} { r } { \dot { t } = 1 - t = \frac { 1 } { 2 } } \end{array}$ . The value of VOROS as a performance measure stems from the way in which the graph of $A _ { t }$ approaches 1 on both ends of the interval [0, 1]: The more imbalanced the classification problem, the less room there is to improve over the better of the two baselines.

# Application to the Wisconsin Breast Cancer Dataset

We first show the application of the VOROS to the Wisconsin Breast Cancer data (William et al. 1995). This is a standard binary classification benchmark dataset with 357 positive observations and 212 negative examples. We directly apply logistic regression, naive Bayes and random forest classifiers (all using sklearn default parameters) to a stratified train/test split. Table 1 summarizes the overall AUROC and VOROS of each classifier without any known bounds on the aggregate cost ratio on the test set, where Figure 4(a) gives the AUROCs and Figure 5 gives each VOROS.

The area under the ROC curve is always less than that of the corresponding volume over the ROC surface. While naively, it may be tempting to claim that both VOROS and AUROC rank the classifiers as Log $>$ Forest $>$ Naive, the situation is a bit more subtle than this as shown in Figure 4.

![](images/b5cda1a933366939de4bb3599cf63b97223084ca02ecb13ef3843d91c132e6c7.jpg)  
Figure 5: (a) The VOROS for the Logistic Regression classifier. (b) The VOROS for the Bayes classifier. (c) The VOROS for the Random Forest classifier. Built on the Wisconsin Breast Cancer dataset.

Table 2: Table summarizing the VOROS for two different cost ranges on the Wisconsin Breast Cancer dataset.   

<html><body><table><tr><td></td><td>VOROS([0,0.25])</td><td>VOROS(0.75,1)</td></tr><tr><td>Log</td><td>99.8%</td><td>99.9%</td></tr><tr><td>Forest</td><td>99.9%</td><td>99.6%</td></tr><tr><td>Baseline</td><td>92.5%</td><td>92.5%</td></tr></table></body></html>

Here, we see that our logistic regression and random forest classifier each dominate the naive Bayes model in ROC space, and both logistic regression and random forest do better than naive Bayes for all choices of cost/class imbalance. However, logistic regression and random forest cannot be ranked so definitively since the ROC curves cross (Hand and Till 2001). Which classifier does best will depend on the costs.

Thus, in ROC space neither curve dominates the other. This can be addressed by VOROS if we are given bounds on the parameter $t$ . For example, if we take $t$ to be in the range $0 \leq t \leq 0 . 2 5$ , then we get that random forest outperforms logistic regression. On the other hand, if $t$ were in the range $0 . 7 5 \leq t \leq 1$ , then we get the opposite ranking as shown in Table 2. Recall that these percentages represent the portion of the ROC points which are lesser classifiers in this range of cost values. For this range, picking the better of the two baselines already outperforms over $92 \%$ of ROC points.

# Application to BUSI

We also consider a more modern breast cancer dataset, the Breast UltraSound Images dataset, henceforth BUSI (AlDhabyani et al. 2020). This dataset consists of full ultrasound breast images, from 600 women patients. It is actually a 3- class dataset. So, to make it a binary classification task, we consider both “normal” and “benign” to be the negative class and “malignant” to be the positive class.

We encode the data using Pytorch’s default vision transformer after cropping the images and scaling them to be the same size. The resulting numerical features are then used to fit logistic regression (with $C = 1 0 0 0$ and $m a x _ { - } i t e r = 2 5 \$ ), naive Bayes (sklearn default), and random forest classifiers (sklearn default).

Table 3: Table summarizing the VOROS and AUROC values for classifiers trained on the BUSI dataset.   

<html><body><table><tr><td></td><td>VOROS([0,1])</td><td>AUROC</td></tr><tr><td>Log</td><td>97.2%</td><td>91.8%</td></tr><tr><td>Forest</td><td>97.5%</td><td>93.3%</td></tr><tr><td>Naive</td><td>95.4%</td><td>88.3%</td></tr><tr><td>Baseline</td><td>80.7%</td><td>50%</td></tr></table></body></html>

Table 4: Table summarizing the VOROS over different ranges of costs for the BUSI dataset. Note we abbreviate VOROS to simply $\mathrm { v }$ due to space constrains.   

<html><body><table><tr><td></td><td>V([0,1/3])</td><td>V(1/3,2/3)</td><td>V(2/3,1)</td></tr><tr><td>Log</td><td>96.5%</td><td>96.7%</td><td>98.4%</td></tr><tr><td>Forest</td><td>98.0%</td><td>96.7%</td><td>98.0%</td></tr><tr><td>Naive</td><td>96.4%</td><td>93.5%</td><td>96.1%</td></tr><tr><td>Baseline</td><td>89.2%</td><td>63.7%</td><td>89.2%</td></tr></table></body></html>

Table 3 summarizes the results. As we have seen before, the area under the ROC curve is less than that of the corresponding volume over the ROC surface. The ranking of nonbaseline classifiers according to AUROC is Forest $> \mathrm { L o g } >$ Naive, which is the same ranking for the VOROS over the entire interval [0, 1], however, plotting the ROC curves, a different story emerges.

In Figure 4(b) all three curves intersect, hence no single model dominates, and which of these three models performs better will depend on the specific cost and class imbalances, as captured in the VOROS (see Figure 6, where Table 4 summarizes the VOROS for a few example choices of cost intervals).

Table 4 shows that the logistic regression model performs better for cases where the aggregate cost of false positives is much higher than that of false negatives. This can be confirmed by a careful inspection of Figure 4 (b) where the solid curve crosses paths with the dotted curve. There is a range of steeply sloped lines where the solid curve will perform better and these contribute to its better performance in the VOROS for the rightmost column above.

Overall, we see that how classifiers should be ranked depends strongly on the relative aggregate costs of false positives and false negatives on these data.

# Application to a Credit Fraud Dataset

The previous two examples have both very good absolute performance of the classifiers considered, plus the class imbalance is not so extreme. The biggest strength of the VOROS, however, is when there is extreme imbalance in class membership and misclassification costs, which we demonstrate in an example credit card fraud dataset. We use a processed dataset derived from a set of transactions made by credit cards in September 2013 by European cardholders over the course of two days. (Dal Pozzolo et al. 2015, 2014, 2017; Dal Pozzolo 2015; Carcillo et al. 2018a,b; Lebichot et al. 2020; Carcillo et al. 2021; Le Borgne et al. 2022; Lebichot et al. 2021) There are a total of 492 frauds (positive class) out of 284,807 transactions, so the fraud class accounts for only about . $1 7 2 \%$ of the transactions. The dataset is published with derived features which are a set of principal components from PCA processing the original features of the dataset (since the original features could not be published for confidentiality concerns).

![](images/0fd756ff9f60f9eb41e75f658dc718e8f93620d409f84cd8e5b18b095b0116db.jpg)  
Figure 6: (a) The VOROS for the Logistic Regression classifier. (b) The VOROS for the Bayes classifier. (c) The VOROS for the Random Forest classifier. Built on the BUSI dataset.

![](images/043955a7b597d9adca0be9afaf6d9c9bc643cdab97af76e49cd88566295bb60d.jpg)  
Figure 7: A plot of the ROC curves for the Dal Pozzolo Credit Fraud dataset. The solid line connects the optimal points on each curve that have the same cost for a specific value of $t$ .

We build logistic regression models: one using just the first principal component $( M _ { 1 } )$ and the other using just the second principal component $( M _ { 2 } )$ to model whether transactions were fraudulent. This results in the ROC curves shown in Figure 7.

Since these curves intersect, which curve performs better depends on the aggregate misclassification costs. Figure 7 shows the resulting ROC convex hulls together with a line showing the optimal points on each curve that have the same expected cost for a particular choice of aggregate class imbalance, Figure 8 shows each VOROS. Here, $t \approx 0 . 2 1 6 5$ results in the these optimal points having the same normalized expected costs. This is the dividing threshold between $M _ { 1 }$ performing better and $M _ { 2 }$ performing better.

Table 5: Table summarizing the VOROS vs. AUROC in the chosen cost range for the Dal Pozzolo Credit Fraud data. Note we abbreviate VOROS to simply $\mathrm { \Delta V }$ due to space constrains.   

<html><body><table><tr><td></td><td>V([999/5999,99/399])</td><td>AUROC</td></tr><tr><td>M1</td><td>90.1%</td><td>79.5%</td></tr><tr><td>M2</td><td>89.6 %</td><td>85.3%</td></tr><tr><td>Base</td><td>86.9%</td><td>50%</td></tr></table></body></html>

There is no guidance given in the dataset for the costs of a false positive versus a false negative, but if we assume that the cost of incorrectly flagging a true transaction as fraud is small (requiring more testing and verification, or customer annoyance to get a flagged card unfrozen) compared to the cost of missing fraud, then the misclassification cost of not detecting fraud could vary depending on both the number of fraudulent transactions and the dollar amount charged. Suppose we want to choose between these two classifiers, $M _ { 1 }$ and $M _ { 2 }$ , when we have only bounds on these values in the real world: for example, we assume that between $0 . 1 \%$ and $1 \%$ of transactions are fraudulent, and that the cost of the false negatives is between 500 and 5000 times the cost of a false positive. Using the AUROC, we would pick $M _ { 2 }$ , but using the VOROS we find that $\frac { 1 - t } { t } = \frac { C _ { 0 } | \bar { N | } } { C _ { 1 } | P | } = \frac { C _ { 0 } } { C _ { 1 } } \frac { | N | } { | P | }$ is the product of the cost and class ratios. Which satisfy the following bounds

$$
9 9 \leq { \frac { | N | } { | P | } } \leq 9 9 9 { \mathrm { ~ a n d ~ } } { \frac { 1 } { 5 0 0 0 } } \leq { \frac { C _ { 0 } } { C _ { 1 } } } \leq { \frac { 1 } { 5 0 0 } }
$$

These bounds then translate into the following inequality

$$
{ \frac { 9 9 } { 5 0 0 } } \leq { \frac { t } { 1 - t } } \leq { \frac { 9 9 9 } { 5 0 0 0 } }
$$

Which since $t = { \frac { \frac { t } { 1 - t } } { { \frac { t } { 1 - t } } + 1 } }$ induces the limits

$$
{ \frac { 9 9 } { 5 9 9 } } \leq t \leq { \frac { 9 9 9 } { 5 9 9 9 } }
$$

Taking the Volume over each ROC surface yields the results in Table 5. This VOROS comparison immediately shows us facts about classifier performance in our estimated interval that cannot be obtained from comparing the AUROCs. First, we learn that the baseline classifier, that flags all transactions as fraud actually is not doing so badly in this range by comparing its VOROS $( 8 6 . 9 \% )$ to the values for $M _ { 1 }$ and $M _ { 2 }$ (where the strength of a baseline classifier for problems with very rare or very costly classes is high, but is not immediately obvious from looking at ROC curves on their own. In this scenario, most ML papers shift to AUPRC or other measures of classifier performance as a result, so as to not recommend a classifier that is worse than a baseline). Second, comparing the AUROCs in Table 5, one would expect classifier $M _ { 2 }$ to dominate, but in this cost range, $M _ { 1 }$ is better, and since the VOROS is larger in this region, for this range we should pick classifier $M _ { 1 }$ instead of either $M _ { 2 }$ or the baseline. This agrees with a cost analysis which shows that if the slope of the supporting line to the dotted and dashed curves in Figure 7 are between $\frac { 9 9 } { 5 0 0 }$ and $\frac { 9 9 9 } { 5 0 0 0 }$ , then the supporting line for the dotted curve is higher than that of the dashed curve.

![](images/e15b50cbb3a0097913da0cf2017970a5bb72fcd309bfaf3cea9cf1b3e9d60a3a.jpg)  
Figure  8: The VOROS for logistic regression classifiers (a) using the first principal component, $M _ { 1 }$ , and (b) the second principal component $M _ { 2 }$ .

# Discussion

We introduced the VOROS, a natural and easily-computable measure that generalizes the AUROC. We showed both theoretically and in example datasets that it has utility for comparing classifiers when classes or costs are unbalanced. We do note that an effective use of the VOROS is limited to domains where subject matter experts can put some bounds on expected misclassification costs. So far, our work handles only binary classification tasks. In fact, it is important to remark that the Volume under the ROC surface, or VUS, studied by (Ferri, Hern´andez-Orallo, and Salido 2003) and others (He and Frey 2008; Waegeman, De Baets, and Boullart 2008; Kang and Tian 2013) is an entirely different concept, generalizing the AUROC to multiple classes, rather than generalizing binary classifiers to different cost and class imbalances. Indeed, a generalization our VOROS work to also handle multiple classes, perhaps by increasing the dimension of the surface, remains a possible topic of future research, as is a generalization to incorporate varied misclassification costs by instance as in (Fawcett 2006).

# Ethics Statement

In many application domains (including breast cancer screening), there are ethical implications for how one measures the cost of a false positive versus a false negative. Once the costs, including ethical costs, of misclassification have been considered by experts, we can analyze the results in our framework.