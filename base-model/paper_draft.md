# Exploring Effects of Consumer Finance Regulation with Deep Learning

Over the past century, consumer credit in the United States has become an essential tool for the
economic life of American consumers ([Hyman, 2011](https://doi.org/10.23943/princeton/9780691140681.001.0001)
[Mehrsa, 2015](https://doi.org/10.1093/sw/swy017) [Prasad, 2012](https://doi.org/10.1093/sf/sot122))
and its centrality in the economy is reflected in how deeply
credit is woven into the social and financial fabric of American consumers. Despite the centrality of
consumer credit, the regulatory framework for consumer financial protection features multiple
federal and state regulators driving policy interventions, and an evolving patchwork of federal and
state laws.

In this paper, we seek to test the causal impact of policy interventions regarding consumer data on
policy outcomes concerning algorithmic discrimination in lending by leveraging recent advances in
deep learning in heterogeneous agent macroeconomics. Our method is to compare multiple models,
each of which represents a choice of public policy and social scenario. We compare outcomes of
these models as a way of testing the theoretical causal impact of policy choices.

Our method is to compare multiple models, each of which represents a choice of public policy and social scenario.
We compare the outcomes of these models as a way of testing the theoretical causal impact of policy choices.
To support this method, we use a variation of the method from [Maliar et al. 2021](https://doi.org/10.1016/j.jmoneco.2021.07.004).
We define a variety of models as systems of structural equations which characterize dynamic systems with decision variables.
We generate the All-In-One loss fuction programmatically from these equations, and use these loss functions to find, using deep learning, the optimal decision policies and ergodic distributions.
We expand on [Maliar et al. 2021](https://doi.org/10.1016/j.jmoneco.2021.07.004) by performing a double optimization:
both the consumers and the lender agent (representing banks, and the supply of credit) make decisions.


# Background

Our project builds on prior work in heterogeneous agent modeling of consumer credit ([Chatterjee et al., 2007](https://doi.org/10.1111/j.1468-0262.2007.00806.x), [Chatterjee et al., 2023](https://doi.org/10.3982/ECTA18771)) and agent-based modeling ([Papadoupolos, 2019](https://doi.org/10.1016/j.jedc.2019.05.002)).

We are particularly interested in the way consumers face differing outcomes based on protected
attributes—race, color, religion, national origin, sex or marital status, age (has capacity to contract);
income derived from any public assistance program; or exercise of any right under the Consumer
Credit Protection Act—and how personal data, including these outlined attributes, and machine
learning is used in lending decisions and credit scoring. (See Equal Credit Opportunity Act (ECOA),
[Fuster et al., 2021](https://doi.org/10.1111/jofi.13090)).
Of particular interest in consumer protection regulation is whether and how a lending policy can be minimally discriminatory while serving the business needs of the lender ([Laufer et al., 2025](https://doi.org/10.1145/3709025.3712214)).
Prior work in computer science, where these issues are more frequently addressed, has identified many nuances of ``fairness'' through the use of structural causal models (SCM), in which the topology of structural equations is carefully analyzed to identify pathways through which protected attributes can effect outcomes.
([Madras et al., 2019](https://doi.org/10.1145/3287560.3287564), [Mhasawade and Chunara, 2021](https://doi.org/10.1145/3461702.3462587)).

Typically, the computer science literature does not consider the dynamic or equilibrium outcomes, though there
are examples of extensions of SCM to dynamic systems with choice [@10.5555/3524938.3525142]
and multi-agent systems ([Lazri et al., 2025](https://doi.org/10.48550/arXiv.2502.18534)).
These frameworks are similar to those used in economics, but deep learning techniques such as [Maliar et al. 2021](https://doi.org/10.1016/j.jmoneco.2021.07.004)
have yet been applied to these areas in practce.

# Methodology

We build and compare models which share some properties in common, using the following notation:

```{list-table} Notation
:header-rows: 1
:label: notation-table

* - Symbol
  - Explanation
* - $N$
  - Number of consumers.
* - $w$
  - Consumer wealth (state)
* - $e$
  - Consumer earnings (state)
* - $h$
  - Consumer credit history (state)
* - $q$
  - Interest rate (consumer state, lender decsion)
* - $c$
  - Consumption (consumer decision)
* - $d$
  - Default/bankruptcy (consumer decision)
* - $z$
  - consumer shocks.
* - $u$
  - Consumer single period utility fucntion.
* - $f$
  - Lender single period utility function
* - $g$
  - Consumer transition equation.
* - $\theta_{\pi}$
  - Parameters of the consumer decision rule
* - $\theta_l$
  - Parameters of the lender decision rule.
* - $\beta_c$
  - Consumer discount factor
* - $\beta_l$
  - Lender discount factor
```

There is a population of $N$ consumers and a single lender agent, who are in partial equilibrium.
Each consumer enters each period in states $x_t = (a, e, h, q)$ and encounters shocks $z_t$.
They take actions $a_t = (c_t, d_t) = \pi(x_t, z_t ;\theta_\pi)$ subject to a decision rule from a family of functions parameterized by $\theta_\pi$.
They experience reward $r(x_t,z_t,a_t)$.
Based on a transition function, they find themselves in state $x' = g(x, z, a)$.
They anticipate future lifetime (continuation) value of $w(x)$ discounted by $\beta$.
This is a standard set of assumptions for dynamic programming and makes the consumer's problem in principle amenable to the method in [Maliar et al. 2021](https://doi.org/10.1016/j.jmoneco.2021.07.004).

$$(a', e', h', q') = g_{\theta_l}(a, e, h, q; z ; c, d)$$

This breaks down into components.

$$a'= g_a(a, e, h, q; z; c, d)$$

For example, $a' = q a + e - c + d$.

$$e' = g_e(a, e, h, q; z; c, d)$$

For example, $e' = e + z_e$, where $z_e$ is a mean-0 LogNormal distribution.

$$h' = g_h(a, e, h, q; z; c, d; z)$$

**This credit history law of motion is one of the key features of our model; we will give it careful thought.**
We expect credit history to penalize defaults $d$ but also have a decay rate $\beta_h$ which allows past defaults to be 'forgotten' over time.
We also expect shocks $z_h$ to effect this history.
These shocks may or may not be correlated with protected attributes $p_c$.

However, in our method, the lender is a strategic actor which decides a pricing kernel based on their own decision rule, a function parameterized by $\theta_l$, and expected utility.

$$q' = g_q(a, e, h, q; c, d; z; \theta_l)$$

We thus use the all-in-one deep learning method twice to determine the strategic equilibrium as lenders and consumers learn best-response strategies.

While the general structure of the model remains the same for all variations, we alter the structural equations to reflect different social scenarios and policy choides.

## The consumer's problem

The consumer tries to maximize expected lifetime utility:

$$V(a, e, h, q; z) = \max_{c,d \in \Gamma(a, e, h, q; z)} u(a, e, h, q; z; c, d) + \beta E [V(g_{\theta_l}(a,e,h,q;c,d; z))]$$

Or, given our parameterization of the decision rule,

$$V(a, e, h, q, z) = \max_{\theta_c} u(x, z; \pi_{\theta_c}(x, z)) + \beta_c E [V(g_{\theta_l}(x; \pi_{\theta_c}(x, z); z))]$$

## The lender's problem

The lender can borrow at $r^*$ and controls one imporant decision:
the (decision) rule governing interest rates.
This rule is parameterized by $\theta_l$ and returns interest rates
as a function of consumer state and decisions.

$$q' = q(a, e, h; c, d; z; \theta_l)$$

This decision is made subject to an optimization.
In the most general case, this is another Bellman equation, where the state space is an aggregation
over all the consumer states; $X_c = \{x_c\}$, $Z_c = \{z_c\}$.

$$V(X_t,Z_t)= \argmax_{\theta_l} \left[ \sum_c f(x_c, q'_c(x_c; \theta_l);  \pi_{ \theta_c}(x_c, z_c); z_c) \right] + \beta_l E[V(X'_t, Z'_t| X_t,Z_t ;\theta_c, \theta_l)]$$

Special cases of this optimization rule get us to simpler 'base models' that we have considered.

## Algorithm

:::{prf:algorithm} Solution procedure
:label: my-algorithm

**Inputs** ...

**Output** Trained parameters $\theta = \{\theta_c, \theta_l\}$

1. Initialize the algorithm:
    1. Construct the theoretical consumer risk $\Xi_c = E_\omega[\xi_c(\omega; \theta)]$
    2. Construct the theoretical lender risk $\Xi_l = E_\omega[\xi_l(\omega; \theta)]$
    3. Define the empirical consumer risk $\Xi^n_c = \frac{1}{n}\sum^n_{i=1}[\xi_c(\omega_i; \theta)]$
    4. Define the empirical lender risk $\Xi^n_l = \frac{1}{n} \sum^n_{i=1} [\xi_l(\omega_i; \theta)]$
    5. Set initial values of $\theta_\pi$ and $\theta_l$.
2. Train the machine, finding $\theta_\pi$ and $\theta_l$ that minimize the empirical risk functions $\Xi^n_c$ and $\Xi^n_l$:

	  1. Simulate the model to produce data $\{\omega_i\}^n_{i=1}$ by using the decision rules (functions of) $\theta_\pi$ and $\theta_l$
	  2. Update the coefficients $\hat{\theta_c} = \theta_c - \nabla \Xi^n_c(\theta_c, \theta_l)$
    3. Update the coefficients $\hat{\theta_l} = \theta_l - \nabla \Xi^n_l(\hat{\theta_c}, \theta_l)$
    4. End step 2 if $||\hat{\theta} - \theta_c|| < \epsilon$ is satisfied. Otherwise, return to 2.1 with updated $\theta = \hat{\theta}$.
:::



## specific model variations and scenarios

### Lender model: Payday

Payday lenders do not use credit history when issuing a loan.
Thus, $q' = q(a, e; c, d; z; \theta_l)$ does not depend on $h$.
When they sell the loan to collection agencies, that does impact the credit history and score of the consumer.

### Lender model: Fintech

Fintech bankers are profit seeking (and so do not use the zero-profit lender utility model (see below).
They are also distinguished by using additional data about consumers in their credit evaluations.
This can be modeled by having a more expansive credit history law of motion $h' = g_h(a, e, h, q; z; c, d; z)$.

For example, it can involve more exogenous shock data $z$ which is correlated with protected attributes $p_c$.
(This would compound other issues that relate to traditional credit histories, such as non-debt-related arrests.)

### Lender utility: Short term profit maximizing

If $\beta_l = 0$, then the lender optimizes myopically for present-period reward $f$.

We can have $f$ be a risk-neutral function of the profit that accrues to the bank, which is something like:

$f = \sum_c - (q_c - 1 - r^*) a_c - d_c$

where $r^*$ is the rate at which the lender borrows from the central bank.
This assumes that the default action $d$ deprives the lender of owed funds.


### Lender utility: Zero-profit equilibrium

If we assume that the lender's earnings are subject to a competitive equilibrium process,
then this pushes the lender earnings function towards zero.

If we penalize positive earnings, then the lender reward function becomes something like:

$f = - [ \sum_c - (q_c - 1 - r^*) a_c - d_c ]^2$


### Lender utility: With continuation value

If $0 < beta_l < 1$, then the lender considers discounted lifetime reward much like a consumer,
implying that they consider the continuation value $V(X'_t, Z'_t)$.

Either the profit maximizing or zero-profit equilibrium reward functions may in principle be used here.

But the state space of $X'_t, Z'_t$ is immense because it is exponential in the size of $N$ and so probably won't do.

Rather, the lender's state space -- with respect to the continuation value -- will need to be computed with respect to
summary statistics $\bar{X_t}$ of the consumer state. Since the summary statistics $\bar{Z}_t$ are static constants -- effectively, the parameters of the independent shocks -- they can be exlcuded from the dynamic state space.

$$V(\bar{X_t})= \argmax_{\theta_l} E_{x_c|\bar{X}_t; \bar{Z_t}}[ \sum_c f(x_c, q'_c(x_c; \theta_l);  \pi_{ \theta_c}(x_c, z_c); z_c)] + \beta_l E_{\bar{Z_t}}[V(\bar{X'_t},\bar{Z'_t}| X_t,Z_t ;\theta_c])]$$

Note that this statement now contains two expectation operators, and there are a few different ways to move the uncertainty around.
This will be something that has to be worked out with the All-in-One Operator, which does similar manipulations/multiple expectations by sampling the shocks multiple times.


# data

Federal Reserve consumer credit: https://www.federalreserve.gov/releases/g19/current/default.htm

NY Fed Reserve Bank on Household Debt and Credit:
https://www.newyorkfed.org/microeconomics/hhdc/background.html

FDIC 2023 National Survey of Unbanked and Underbanked Households:
https://www.fdic.gov/household-survey

# key hypotheses

We hypothesize that non-financial consumer data—such as records of convictions of
crime—included in the credit report and credit scores of consumers may implicate protected
attributes of borrowers, and thus, foster discrimination.
(Add other hypothesis on the impact of double optimization, as well as our position on what we
hope to find using deep learning methods).