# Exploring Effects of Consumer Finance Regulation with Deep Learning

Consumer finance protection in the United States has opened a new chapter after the Trump administration has dismantled much of the Consumer Finance Protection Bureau (CFPB), the federal agency with the strongest mandate for enforcing it.
The United States now has an evolving patchwork of state and federal consumer finance laws and rules.
Lawmakers have many potential policy levers available to them when regulating consumer credit, with many potential policy goals for these rules.
We leverage recent advances in deep learning in heterogeneous agent macroeconomics to test the impact of a variety of policy interventions on policy outcomes.

Our method is to compare multiple models, each of which represents a choice of public policy and social scenario.
We compare the outcomes of these models as a way of testing the theoretical causal impact of policy choices.
To support this method, we use a variation of the method from [Maliar et al. 2021](https://doi.org/10.1016/j.jmoneco.2021.07.004).
We define a variety of models as systems of structural equations which characterize dynamic systems with decision variables.
We generate the All-In-One loss fuction programmatically from these equations, and use these loss functions to find, using deep learning, the optimal decision policies and ergodic distributions.
We expand on [Maliar et al. 2021](https://doi.org/10.1016/j.jmoneco.2021.07.004) by performing a double optimization:
both the consumers and the lender agent (representing banks, and the supply of credit) make decisions.

# Background

Our project builds on prior work in heterogeneous agent modeling of consumer credit ([Chatterjee et al., xx](https://doi.org/10.1111/j.1468-0262.2007.00806.x), [Chatterjee et al., 2023](h.ttps://doi.org/10.3982/ECTA18771)) and agent-based modeling ([Papadoupolos, 2019](https://doi.org/10.1016/j.jedc.2019.05.002)).

We are particularly interested in the way outcomes differ for consumers that vary based on a protected attribute, such as race, immigration status, gender, or age, and in how personal data and machine learning is used in lending decisions and credit scoring ([Fuster et al., 2021](https://doi.org/10.1111/jofi.13090)).
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
$$h' = g_h(a, e, h, q; z; c, d; z)$$

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

**Inputs** Given a Network $G=(V,E)$ with flow capacity $c$, a source node $s$, and a sink node $t$

**Output** Compute a flow $f$ from $s$ to $t$ of maximum value

1. Initialize the algorithm:

  1. Construct the theoretical consumer risk $\Xi_c$
  2. Construct the theoretical lender risk $\Xi_l$
  3. Define the empirical consumer risk $foo$
  4. Define the empirical lender risk $bar$
  5. $f(u, v) \leftarrow 0$ for all edges $(u,v)$

2. Train the machine, finding $\theta_\pi$ and $\theta_l$ that minimize the risk functions $\Xi_c$ and $\Xi_l$:

	1. Find $c_{f}(p)= \min \{c_{f}(u,v):(u,v)\in p\}$
	2. For each edge $(u,v) \in p$

		1. $f(u,v) \leftarrow f(u,v) + c_{f}(p)$ *(Send flow along the path)*
		2. $f(u,v) \leftarrow f(u,v) - c_{f}(p)$ *(The flow might be "returned" later)*
:::

## specific model variations and scenarios

### Lender models

#### Banking type

##### Payday

Wouldn't use credit history when making the loan.

When they sell the loan to collection agencies, it does impact credit history and score.

##### Fintech

Profit-seeking (not zero profit.)

Using alternative data.


#### Lender utility

##### Short term profit maximizing

If $\beta_l = 0$, then the lender optimizes myopically for present-period reward $f$.

We can have $f$ be a risk-neutral function of the profit that accrues to the bank, which is something like:

$f = \sum_c - (q_c - 1 - r^*) a_c - d_c$

where $r^*$ is the rate at which the lender borrows from the central bank.
This assumes that the default action $d$ deprives the lender of owed funds.


##### Zero-profit equilibrium

If we assume that the lender's earnings are subject to a competitive equilibrium process,
then this pushes the lender earnings function towards zero.

If we penalize positive earnings, then the lender reward function becomes something like:

$f = - [ \sum_c - (q_c - 1 - r^*) a_c - d_c ]^2$


##### With continuation value

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

# key hypotheses