# Model version 4

We build and compare models sharing common properties. Key notation and an expanded description of the method is in the Appendix. We adopt the convention that lowercase variables refer to individual consumers.

We model a population of $N$ consumers. Notation is in the Appendix.

Consumer variables:
  - Consumers heterogenous attributes:
    - $p$ protected attribute
    - $\beta$ discount factor (could be heterogeneous)
    - risk aversion
  - States $x_t = (a_t, h_t)$ (wealth and credit history, carried over from the previous period)
  - exogenous shock $z_t$ (which includes earnings).
    - these can be correleted with consumer heterogeneous attributes.
  - action $y_t=(c_t, d_t)$ (consumption and default)
    - $d_t$ is $\{0,1}$, discrete. (I suppose we could make this continuous, valued between 0 and 1, if we preferred continuous spaces.)

Lender variables:
  - lender can borrow at $r^*$

**TODO: We need to break out all the different possible shocks $z$**
 - some could be observed by consumer
 - some can be observed by lender
 - some can be correlated with consumer attributes.
   
## Dynamics

- consumer parameterrized decision rule $y_t = \sigma(x_t, z_t; q_t, \theta)$ **TODO: There may be idiosyncratic shocks that are not known to the consumer.**
  - Is there a borrowing limit? (constraint on $c_t)
-  lender parameterized decision rule: 
  - **AL original:** $q_t = \rho(x_t, y_t, z_t; \phi)$
  - **SB proposal:** $q_{t+1} = \rho(x_{t+1}; \phi) = \rho(x_t, y_t, z_t; \phi)$
- **Question: are we keeping earnings $e$ as a random walk? Are shocks $z$ a Markov process?**
- $x_{t+1} = T(x_t, y^*_t, z_t; q^*_t)$. Which can be broken down into:
  - $a_{t+1} = T_a(x_t,  y^*_t, z_t; q^*_t)$
    - $ a_{t+1} = (1-d_t) (1/q_t) [ a_t + e(z_t) - c_t ] $
  - $h_{t+1} = T_h(x_t,  y^*_t, z_t; q^*_t)$
    - Does $h$ depend on $\phi$?
    - $h_{t+1} = \eta h_t + \phi_e z_e + \phi_d d$ where:
      - $\eta \in (0,1)$ is a forgetting rate on history.
      - $\phi_e$ and $\phi_d$ are subparameters of $\phi$ used as coefficients.
      - $z_e$ is that component of the shockes (exogenous state) that represents earnings.
  
## Rewards

- Consumer utility $u(c_t)$ is CRRA utility
- lender profit per consumer is: $$ \pi(x_t, y_t, z_t; q_t, r^*) = [ (1-d_t)/q_t - (1+r^*) ] (-a_t) $$where $d_t \in \{0, 1\}$ is the default decision (with $d_t=0$ assumed if $a_t \ge 0$).


## The consumer's problem

The value function $V(x_t, z_t; q^*_t)$ satisfies the Bellman equation:

$$ V(x_t, z_t; q^*_t) = \max_{y_t=(c_t, d_t) \in \Gamma(x_t, z_t)} \left\{ u(c_t) + \beta \mathbb{E}_{z_{t+1}} [V(x_{t+1}, z_{t+1}; q^*_{t+1})] \right\} \label{eq:consumer-bellman} $$


## The lender's problem


Let vector $\mathcal{M}_t$ summarize the joint state of all consumers, such as through low-order moments.

The approximated value function conditional on these moments is:
$$V_l(\mathcal{M}_t) \approx \max_{\phi} \left\{ \mathbb{E}_{x_t, z_t|\mathcal{M}_t}[\sum^N_{i=1} \pi(x_t, y^*_t, z_t; q_t(\phi), r^*) ] + \delta \mathbb{E}_{Z_t}[V_l(\mathcal{M}_{t+1})] \right\} $$

This optimal pricing strategy is implemented by finding the parameters $\phi$ of the lender's policy network $\rho(x_t, y_t, z_t; \phi)$, which may take $\mathcal{M}_t$ as input.

The expectation $\mathbb{E}_{x_t, z_t|\mathcal{M}_t}$ denotes averaging over the micro states consistent with the aggregate moments $\mathcal{M}_t$. **TODO: This is not well defined. The empirical loss calculation, using samples, of the Maliar method gives us a way to do this?**

We propose solving for this strategic Nash equilibrium computationally using an alternating optimization scheme based on deep learning, as detailed in the Appendix.

# Implementing regulatory rules

This a fintech lender may be subject to a variety of regulations, including:

* *rate limits*. Impose a maximum consumer interest rate $1/q_t < R$
* *prohibitions on disparate treatment* (use of protected category $p$),
* the availability of *alternative and non-financial data* which can be correlated with protected attributes $p$, heterogenous consumer preferences $\beta$, or both,
* *time limits for negative information*, modeled as a decay rate on credit history $h$, such that prior information
recedes in relevance over time.

We hypothesize that even in the presence of prohibitions on disparate treatment, the availability
of correlated non-financial data to the lender can lead to disparate outcomes.
However, this disparity may be limited by stricter time limits for negative information.

# Model outputs

- *Bank profit.* We measure expected bank profit $\sum^N \pi$.
- *Disparity*. Differences in the moments of assets $a_t$ and credit histories $h_t$ for subpopulations $N_p$
- *Default rates*. Proportion of $d$
- *Consumption equivalent welfare.*  Alan to look into this.


# Appendix

```{list-table} Notation
:header-rows: 1
:label: notation-table

* - Symbol
  - Domain/Type
  - Explanation
* - $N$
  - $\mathbb{N}^+$
  - Number of consumers.
* - $a_t$
  - $\mathbb{R}; a_t \ge \underline{a}$
  - Individual consumer net financial assets at start of period $t$ (state).
* - $h_t$
  - $[0, 1]$
  - Individual consumer credit history score at start of period $t$ (state).
* - $x_t = (a_t, h_t)$
  - $\mathbb{R} \times [0, 1]; x_t \ge (\underline{a}, 0)$
  - Individual consumer state vector.
* - $q_t$
  - $\mathbb{R}^+$
  - Price at time $t$ of one unit of assets delivered at $t+1$. This is the inverse of the gross interest rate ($q_t = 1/R_t$), incorporating a risk premium (i.e., $q_t < 1/(1+r^*)$ if default risk exists). Set by lender rule $\rho(x_t, y_t, z_t; \phi)$.
* - $c_t$
  - $\mathbb{R}^+$
  - Individual consumption in period $t$ (action).
* - $d_t$
  - $\{0, 1\}$
  - Individual default decision in period $t$ (action).
* - $y_t = (c_t, d_t)$
  - $\mathbb{R}^+ \times \{0,1\}$
  - Individual consumer action vector.
* - $z_t$
  - (Varies)
  - Vector of individual consumer shocks realized in period $t$.
* - $e(z_t)$
  - $\mathbb{R}$
  - Earnings component derived from shocks $z_t$.
* - $\mu_t$
  - $\Delta(X \times Z)$
  - Joint distribution of consumers over endogenous states $x_t \in X$ and exogenous shocks $z_t \in Z$.
* - $u(.)$
  - Function
  - Consumer single period utility function (e.g., CRRA over $c_t$).
* - $\pi(...)$
  - Function
  - Lender single-period profit function from net position with consumer.
* - $T(...)$
  - Function
  - Individual consumer state transition function for $(a_{t+1}, h_{t+1})$.
* - $\sigma(...; \theta)$
  - Function (DNN)
  - Consumer policy rule mapping state $(x_t, z_t)$ and price $q_t$ to action $y_t$: $y_t = \sigma(x_t, z_t; q_t, \theta)$.
* - $\rho(...; \phi)$
  - Function (DNN)
  - Lender policy (pricing schedule) mapping $(x_t, y_t, z_t)$ to $q_t$.
* - $\theta$
  - $\mathbb{R}^k$
  - Parameters (weights, biases) of the consumer policy DNN $\sigma$.
* - $\phi$
  - $\mathbb{R}^m$
  - Parameters (weights, biases) of the lender policy DNN $\rho$.
* - $\beta$
  - $(0, 1)$
  - Consumer discount factor.
* - $p$
  - $\{0,1\}$
  - Consumer protected attribute.
* - $\delta$
  - $[0, 1)$
  - Lender discount factor.
* - $r^*$
  - $\mathbb{R}^+$
  - Risk-free interest rate.
* - $\Gamma(...)$
  - Set
  - Feasible action set for consumer, potentially implicitly defined by existence of equilibrium with $\rho, \sigma$.
* - $\Xi_c, \Xi_l$
  - Function
  - Theoretical Loss functions (based on Bellman residuals).
* - $\hat{\Xi}_c, \hat{\Xi}_l$
  - $\mathbb{R}^+$
  - Empirical Loss counterparts (sample average of residuals).
* - $\mathcal{M}_t$
  - $\mathbb{R}^p$
  - Vector of moments (e.g., means, variances, covariances) of the joint distribution $\mu_t$ over $(x_t, z_t)$.
* - $V(.)$
  - Function
  - Value function (consumer $V(x_t, z_t; q^*_t)$ or lender $V(\mu_t)$).
```

## Methodology notes

*Equilibrium Existence and Uniqueness:* The existence and uniqueness of a strategic Nash equilibrium $(\sigma(\dots; \theta^*), \rho(\dots; \phi^*))$ in this two-agent game, particularly with discrete choices and potentially non-convexities, are not guaranteed a priori. Standard fixed-point theorems may not apply directly. The full paper will discuss relevant theoretical considerations, potential sources of multiplicity, and the implications for interpreting the computationally obtained results, including whether the alternating optimization algorithm converges and to which equilibrium if multiple exist. Benchmarking against simpler cases with known properties (e.g., a toy model with a single representative borrower and quadratic utility where the equilibrium is known) will be crucial for validating the solver's accuracy.


## Solution Algorithm


**Objective:** Approximate Nash equilibrium $(\theta^*, \phi^*)$ for the consumer policy network $\sigma(...; \theta)$ and lender policy network $\rho(...; \phi)$ 

1.  **Initialization:**
    *   Initialize DNN parameters $\theta^{(0)}, \phi^{(0)}$ for policy networks $\sigma$ and $\rho$. Initialize value function networks $V_c, V_l$.
      - **TODO: Aren't these AiO in the Maliar method?**
    *   Define theoretical loss functions $\Xi_c, \Xi_l$ based on Bellman equations ({eq}`eq:consumer-bellman`, {eq}`eq:lender-problem`).
    *   Initialize experience replay buffer $\mathcal{D}$. **TODO: What?**
    *   Set optimization hyperparameters. Define a method for finding the within-period equilibrium $(y_t, q_t)$ (e.g., inner fixed-point iteration, relaxation). **TODO: I don't understand what you're saying here.**

2.  **Iteration Loop (Epochs $k=0, 1, \dots$):**

    1.  **Simulate:** Generate trajectories using current policies $(\sigma(\cdot; \theta^{(k)}), \rho(\cdot; \phi^{(k)}))$ for $S$ steps. For each step $t$: Given $(x_t, z_t)$, determine action $y_t$ using $y_t \approx \sigma(x_t, z_t; q_t, \theta^{(k)})$ and price $q_t$ using $q_t \approx \rho(x_t, y_t, z_t; \phi^{(k)})$, reflecting the interaction based on current policies (precise handling depends on implementation, see Note). Then calculate $x_{t+1} = T(x_t, y_t, z_t; q_t)$. Store transitions $(x_t, z_t, y_t, q_t, x_{t+1})$ in $\mathcal{D}$. **TODO: We are doing an infinite horizon problem. We don't need to do more than one period!**
    
    2.  **Optimize Consumer Policy $\theta$ (given fixed $\phi^{(k)}$):**
        1.  For $N_c$ steps **TODO: Overloading $N$** : Sample a mini-batch of $(x_j, z_j)$ states from $\mathcal{D}$. Draw required future shocks.
        2.  Compute empirical risk $\hat{\Xi}_c(\theta^{(k)}, \phi^{(k)})$ based on {eq}`eq:consumer-bellman`.
          - This requires evaluating the Bellman residual using the consumer's policy $\sigma(\cdot; \theta^{(k)})$ and the lender's *fixed* policy $\rho(\cdot; \phi^{(k)})$ to determine the relevant $(y_j, q_j)$ for the sampled state $(x_j, z_j)$. **TODO: What?**
          - The expectation term $\mathbb{E}[V(\dots)]$ is evaluated using the current consumer value network $V_c$, potentially employing techniques like the AiO operator (@Maliar2021) for efficiency.
          - Use Gumbel-Softmax relaxation for $y_j$ if needed. **TODO: What?**
        3.  Calculate gradient $\nabla_\theta \hat{\Xi}_c$. Gradient calculation needs to account for the dependence of $y_j$ on $\theta$ (directly via $\sigma$).
        4.  Update $\theta$ towards $\theta^{(k+1)}$.
    
    3.  **Optimize Lender Policy $\phi$ (given fixed $\theta^{(k+1)}$):**
        1.  For $N_l$ steps: Sample a mini-batch of $(x_j, z_j)$ states from $\mathcal{D}$. Draw required future shocks.
        2.  Compute empirical risk $\hat{\Xi}_l(\theta^{(k+1)}, \phi^{(k)})$ based on {eq}`eq:lender-problem`.
           - Requires evaluating the lender Bellman residual using the lender's policy $\rho(\cdot; \phi^{(k)})$ and the consumer's *fixed* policy $\sigma(\cdot; \theta^{(k+1)})$ to determine the relevant $(y_j, q_j)$ for the sampled state $(x_j, z_j)$. **TODO: Why?**
           - The expectation term $\mathbb{E}[V_l(\mu_{t+1})]$ is evaluated using the current lender value network $V_l$ (or its moment-based approximation $V_l(\mathcal{M}_{t+1})$), potentially using the AiO operator (@Maliar2021) for efficiency. **TODO: We need to be more explicit about how we are building the AiO operator**
        3.  Calculate gradient $\nabla_\phi \hat{\Xi}_l$. Gradient calculation needs to account for the dependence of $q_j$ on $\phi$ (directly via $\rho$).
        4.  Update $\phi$ towards $\phi^{(k+1)}$.
    
    4.  **Check Convergence:** Evaluate convergence criteria.

3.  **Output:** The converged parameters $(\theta^*, \phi^*)$ approximate the equilibrium decision rules $(\sigma(...; \theta^*), \rho(...; \phi^*))$.

*Note:* A key challenge is handling the within-period interaction where $y_t$ depends on $q_t$ via $\sigma$ and $q_t$ depends on $y_t$ via $\rho$. While the description above implies using the opponent's fixed policy within each optimization step (consistent with alternating optimization / fictitious play), some implementations might explicitly solve the inner fixed-point $y_t = \sigma(x_t, z_t; q_t, \theta^{(k)})$, $q_t = \rho(x_t, y_t, z_t; \phi^{(k)})$ at each simulation/evaluation step. This adds computational cost and complexity, particularly for gradient calculation (requiring techniques like the implicit function theorem or differentiating the solver). The convergence properties and efficiency may differ between these approaches. **TODO: Why is this variability included in the abstract?**


## Motivation for model variations

The model variations are all implemented as differences in the structural equation of the model.
The solution method remains unchanged.

* *Prohibitions on disparate treatment.* Lenders are typically forbidden from using protected
attributes $p$ directly in their pricing $q$ and credit scoring $h$ systems.

* *Alternative and non-financial data.* Fintech companies may be able to incorporate data that
is not directly related to wealth and earnings in their lending decisions; this is an area
of unsettled policy. Even in traditional lending, non-financial information such as prior (non-debt-related)
arrests are typically part of a credit report. We model this data as components of the consumer shocks $z$.
These shocks can be correlated with protected attributes $p$, heterogenous consumer preferences $\beta$, or both,
leading to potential bank profit and disparate outcomes.

* *Time limit for negative information.* US laws (such as the Fair Credit Reporting Act) put
a time limit on how long negative information can remain on a credit report. This negative
information can include both past defaults, as well as other adverse information, such as non-debt
related arrests. We model this as a decay rate on credit history $h$, such that prior information
recedes in relevance over time.

We hypothesize that even in the presence of prohibitions on disparate treatment, the availability
of correlated non-financial data to the lender can lead to disparate outcomes.
However, this disparity may be limited by stricter time limits for negative information.
