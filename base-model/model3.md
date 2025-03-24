# Generic form - draft 3

There is a population of $N$ consumers.

## A block-based narrative

Each consumer enters each period in states $x_t$ and encounters shocks $z_t$.

They take actions $a_t = \pi(x_t, z_t ;\theta_\pi)$ subject to a decision rule from a family of functions parameterized by $\theta_\pi$.

They experience reward $r(x_t,z_t,a_t)$.

Based on a transition function, they find themselves in state $y = g(x, z, a)$.

They anticipate future lifetime (continuation) value of $w(y)$ discounted by $\beta$.

When the problem is defined such that $y_{t} = x_{t+1}$, then $v = w$, and the problem is simply recursive.

## The consumer's problem

```{list-table} Variables
:header-rows: 1
:label: example-table

* - Symbol
  - Explanation
* - $w$
  - consumer wealth (state)
* - $e$
  - consumer earnings (state)
* - $h$
  - consumer credit history (state)
* - $q$
  - consumer interest rate (state)
* - $c$
  - consumption (control/action)
* - $d$
  - default/bankruptcy (control/action)
* - $z$
  - consumer shocks.
```

$$(a', e', h', q') = g(a, e, h, q; c, d; z)$$

or

$$a'= g_a(a, e, h, q; c, d; z)$$
$$e' = g_e(a, e, h, q; c, d; z)$$
$$h' = g_h(a, e, h, q; c, d; z)$$

And also...

$$q' = g_q(a, e, h, q; c, d; z; \theta_l)$$

We have in mind something like:

$a' = q a + e - c + d$

We'll get back to this $\theta_l$ later.

The consumer tries to maximize expected lifetime utility:

$$V(a, e, h, q; z) = \max_{c,d \in \Gamma(a, e, h, q; z)} u(a, e, h, q; z; c, d) + \beta E [V(g_{\theta_l}(a,e,h,q;c,d; z))]$$

Or, given our parameterization of the decision rule,

$$V(a, e, h, q, z) = \max_{\theta_c} u(x, z; \pi_{\theta_c}(x, z)) + \beta E [V(g_{\theta_l}(x; \pi_{\theta_c}(x, z); z))]$$

## The lender's problem

The lender can borrow at $r^*$ and controls one imporant decision:
the (decision) rule governing interest rates.

This rule is parameterized by $\theta_q$ and returns interest rates
as a function of consumer state and decisions.

$$q' = q(a, e, h; c, d; z; \theta_l)$$

This decision is made subject to an optimization.
In the most general case, this is another Bellman equation!
This is done as an aggregation over all the consumer; $X_c = \{x_c\}$, $Z_c = \{z_c\}$.

$$V(X_t,Z_t)= \argmax_{\theta_l} \sum_c f(x_c, q'_c(x_c; \theta_l);  \pi_{ \theta_c}(x_c, z_c); z_c) + \beta_l E[V(X'_t,Z'_t| X_t,Z_t ;\theta_c)]$$

Special cases of this optimization rule get us to simpler 'base models' that we have considered.

### Short term profit maximizing

If $\beta_l = 0$, then the lender optimizes myopically for present-period reward $f$.

We can have $f$ be a risk-neutral function of the profit that accrues to the bank, which is something like:

$f = \sum_c - (q_c - 1 - r^*) a_c - d_c$

where $r^*$ is the rate at which the lender borrows from the central bank.
This assumes that the default action $d$ deprives the lender of owed funds.


### Zero-profit equilibrium

If we assume that the lender's earnings are subject to a competitive equilibrium process,
then this pushes the lender earnings function towards zero.

If we penalize positive earnings, then the lender reward function becomes something like:

$f = - [ \sum_c - (q_c - 1 - r^*) a_c - d_c ]^2$


### With continuation value

If $0 < beta_l < 1$, then the lender considers discounted lifetime reward much like a consumer,
implying that they consider the continuation value $V(X'_t, Z'_t)$.

Either the profit maximizing or zero-profit equilibrium reward functions may in principle be used here.

But the state space of $X'_t, Z'_t$ is immense because it is exponential in the size of $N$ and so probably won't do.

Rather, the lender's state space -- with respect to the continuation value -- will need to be computed with respect to
summary statistics $\bar{X_t}$ of the consumer state. Since the summary statistics $\bar{Z}_t$ are static constants -- effectively, the parameters of the independent shocks -- they can be exlcuded from the dynamic state space.

$$V(\bar{X_t})= \argmax_{\theta_l} E_{x_c|\bar{X}_t; \bar{Z_t}}[ \sum_c f(x_c, q'_c(x_c; \theta_l);  \pi_{ \theta_c}(x_c, z_c); z_c)] + \beta_l E_{\bar{Z_t}}[V(\bar{X'_t},\bar{Z'_t}| X_t,Z_t ;\theta_c])]$$

Note that this statement now contains two expectation operators, and there are a few different ways to move the uncertainty around.
This will be something that has to be worked out with the All-in-One Operator, which does similar manipulations/multiple expectations by sampling the shocks multiple times.

## Definitions - TODO


## Assumptions

We are assuming that loans have a duration and an interest rate.
While in practice loans are repayed on an incremental schedule, we will make the modeling assumption that we are tracking only the last payment date.