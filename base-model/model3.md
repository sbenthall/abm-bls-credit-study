# Generic form

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

This decision is made subject to some optimization. This is done as an aggregation over all the consumers. **TODO THIS ISN'T RIGHT YET**.

$$\theta'_l = \argmax_{\theta_l} \sum_c f(a_c, e_c, h_c, q'_c(a_c, e_c, h_c; \theta_l);  \pi_{ \theta_c}(x, z); z_c)$$


```{list-table} Variables
:header-rows: 1
:label: example-table

* - function
  - Explanation
```


## Definitions - TODO


## Assumptions

We are assuming that loans have a duration and an interest rate.
While in practice loans are repayed on an incremental schedule, we will make the modeling assumption that we are tracking only the last payment date.