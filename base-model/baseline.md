# Base Model


## Dynamics

We will attempt to lay out the model as a series of structural equations,
each one determining the value of:

- A *shock* - sampled from a random variable
- A *state variable* - a deterministic function of other variables
- A *control variable* - an undefined function of other variables, subject to *constraints*, a function of other variables
- A *reward variable* - like a state variable, but with special significance as the payoff to an agent.

:::{attention} Consumer subscripts
How will we subscript/denote variables that are specific to a single agent? Without it getting too messy
:::

```{list-table} Parameters
:header-rows: 1
:label: example-table

* - Parameter
  - Explanation
* - $p_i$
  - protected attribute of consumer
* - $\rho$
  - risk aversion
* - $\beta_i$
  - Discount factor
* - $r^*$
  - federal interest rate
```

```{list-table} Consumer dynamics
:header-rows: 1
:label: example-table

* - Equation
  - Explanation
* - $\theta \sim \text{LogNormal}(\mu(p_c), \sigma(p_c))$
  - Income shock
* - $m = (1 + r_i) a_{-1} + y$
  - Update resources available
* - $c(m,y) \leq m + \underbar{m}$
  - Quantity of consumption chosen by consumer
* - $u = CRRA(c)$
  - Consumer utility. [reward]
* - $a = m - c$
  - End of period assets are what are unconsumed
* - $y = y_{-1} + \theta$
  - Update income, exogenous process
```

:::{attention} $y$ equation isn't right
or is it?
:::


```{list-table} lender dynamics
:header-rows: 1
:label: example-table

* - Equation
  - Explanation
* - $\theta$
  - interest rate decision rule parameters.
* - $r_i = r_\theta(a^{-1}_i, y_i, [p_i], [a_i ?])$
  - Interest rate. (protected attribute policy-optional)
* - $b = - \sum_c a_c$
  - bank's borrowed balanced from federal lender
* - $f = r^* b + \sum r_c ( - a_c)$
  - lender profit [reward]
```

Here, $\theta$ are parameters of the decision rule for $r$.

The banks are considered risk-neutral, so they don't have a risk-averse utility function.
They simply try to maximize profit.


In a partial equilibrium framework, lenders are assumed to have access to funds at an exogenous risk-free rate $r^*$. They lend to households using a price function $q_\theta$. Lenders aim to maximize or ensure non-negative expected profits, defined as:

## proposal for how to deal with default

```{list-table} additional equations to handle defaulting. A proposal.
:header-rows: 1
:label: example-table

* - Equation
  - Explanation
* - $0 \leq d() \leq - a^{-1}$
  - amount of 'default' by consumer
* - $m = a^{-1} [+ d] + r_i(a_{-1} + d)  + y g(d^{-1})$
  - consumner market resources
* - $\zeta$
  - credit history forgiveness rate
* - $h = \zeta h^{-1} + d$
  - credit history
* - $r_i = r_\theta(a_i, y_i, [p_i], [h_i])$
  - lender takes credit history into account
* - $\underbar{m} = r_\theta(a_i, y_i, [p_i], [h_i]) $
  - borrowing limit as set by the bank
* - $f = r^* b + \sum r_c  (- a_c + d) $
  - lender profit [reward]
```

## Agent strategies

### Consumer

The consumers are lifetime reward optimizers, seeking to maximize

$$\sum_t \beta^t u_t$$

### Lender

:::{attention} Are lenders agents?
We have not yet determined that Lenders are agents, and how many there are.

They probably have a different discount factor from consumers.
:::

The lenders are lifetime reward optimizers, seeking to maximize

$$\sum_t \beta^t f_t$$

## Bellman equation


:::{attention}TO DO: Build Bellman Equation
Based on the dynamics equations, and the agent optimization problems, it should be possible to write a Bellman equation, which would look similar to the Hugget model, which we can reference directly.
:::


## Algorithm sketch


# Basic proposal from before...

Materials from a prior draft, which we need to clean up and bring into the baseline model

### Limited Enforcement and Default Decision

To incorporate default, the model must specify:

- **Default Decision** Households must evaluate and choose between repaying their debt and defaulting. This decision is based on comparing the continuation values of both actions.
- **Punishment upon Default:** Default triggers a penalty, which can take various forms such as exclusion from credit markets, wage garnishment, a utility loss ("utility stigma"), or a reduction in credit score.


In this framework, we augment the household's problem by introducing a discrete choice within the Bellman equation:
\begin{equation}
V(b, y) = \max \Bigl\{V^\text{repay}(b,y), \;V^\text{default}(b,y)\Bigr\}.
\end{equation}

SPB: I find the introduction of the multiple value functions here very confusing.  $\Omega$ isn't well defined. We need to better figure out how to represent the consequences of default.

The value of repaying is given by:
\begin{equation}
V^\text{repay}(b,y) = \max_{b'} \Bigl\{
    u\bigl(y + b - q(b',y)\,b'\bigr)
    + \beta \,\mathbb{E}[V(b',y')]
  \Bigr\}.
\end{equation}
The value of defaulting is:
\begin{equation}
V^\text{default}(b,y) = u\bigl(\tilde{c}(y)\bigr) + \beta \,\mathbb{E}[\Omega(y')],
\end{equation}
where $\tilde{c}(y)$ represents consumption upon default, and $\Omega(\cdot)$ is the continuation value in the default or exclusion state, reflecting the consequences of default.


### Algorithm sketch

To achieve this, an iterative process is employed:
\begin{enumerate}
    \item Guess the parameters $\theta$ of the pricing function $q_\theta$.
    \item Solve the households’ optimization problem to obtain optimal policies for borrowing and default, $\{b'^*(s), \delta^*(s)\}$.
    \item Simulate the model or compute the stationary distribution $\mu^*$ of households across states.
    \item Calculate the realized profits or losses for lenders based on $q_\theta$ and $\mu^*$. \spb{why aren't the lenders discounting utility of profit?}
    \item Update the parameters $\theta$ to satisfy the zero-profit condition or maximize profits, often using gradient-based methods. \spb{which?}
\end{enumerate}
