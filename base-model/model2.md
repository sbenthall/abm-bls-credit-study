# Base Model Draft 2

1.  **State Space**

Each period $t$, an individual’s state is
$$
\omega_{t} = (a_{t}, e_{t}, h_{t}),
$$
where:

*   Assets $a_{t}$ can be positive (savings) or negative (debt).
*   Earnings $e_{t}$ may include persistent and transitory components (e.g., a finite‐state Markov chain for persistent income plus an i.i.d. shock).
*   Credit History $h_{t}$ is our aggregator of past repayment/delinquency. This can be modeled either as:
    *   A discrete credit state $M_{t} \in \{M_{1}, \dots, M_{K}\}$, or
    *   A continuous credit score $S_{t} \in [\underline{S}, \overline{S}]$.

    This keeps the door open to whichever representation best suits your application.

2.  **Household’s Problem**

2.1. **Flow of Choices**

In each period, a household:

*   Observes $(a_{t}, e_{t}, h_{t})$.
*   Chooses whether to repay or default if $a_{t} < 0$.
*   Chooses next period’s asset level $a_{t+1}$ (borrowing or saving), taking as given the price function:
    $$
    q(a_{t+1} \mid a_{t}, e_{t}, h_{t}),
    $$
    which the household views as exogenous (they are price‐takers).

2.2. **Household Value Function**

You might write a dynamic program:
$$
V(\omega_{t}) = \max_{a_{t+1}, \text{repay vs default}} \left\{ u(c_{t}) + \beta \mathbb{E} \left[ V(\omega_{t+1}) \right] \right\},
$$
subject to:

*   Budget constraint: $c_{t} = e_{t} + a_{t} - q(\omega_{t}, a_{t+1}) a_{t+1}$ if repaying debt, or $c_{t} = e_{t} - \kappa$ if filing for bankruptcy, etc.
*   Credit history update $h_{t+1} = \Psi(h_{t}, a_{t+1}, \text{repay/default}, e_{t}, \dots)$.

2.3. **Default Decision**

If $a_{t} < 0$, the individual can pay back or file for bankruptcy at some cost (like disutility or direct bankruptcy fees). Bankruptcy might forcibly set $a_{t+1} \geq 0$. The key is that default triggers a deterioration in credit history:

*   $\Psi(\cdot)$ will shift $h_{t}$ to a “worse” state (like from Near Prime to Subprime, or from $S=720$ down to 550).

3.  **Credit History Update Rule**
$$
h_{t+1} = \Psi(h_{t}, a_{t+1}, \text{repay/default}, e_{t}, \dots).
$$

3.1. **Discrete Version**

You have states $\{M_{1}, \dots, M_{K}\}$, such as $\{\text{No Credit}, \text{Subprime}, \text{Near Prime}, \text{Prime}\}$.

The transition can be deterministic (simple rules):

*   If you default, $\Psi(M, \text{default}) = \text{Subprime}$ with certainty.
*   If you repay $\ell$ consecutive times at some threshold, you upgrade from Subprime to Near Prime, etc.

Or it can be stochastic (a transition matrix) to capture partial or noisy improvements:
$$
\Pr(M_{t+1} = M_{j} \mid M_{t}, \text{actions}) = \Psi_{\text{discrete}}(M_{t}, \dots, M_{j}).
$$

3.2. **Continuous Version**

You define a score $S_{t} \in [\underline{S}, \overline{S}]$. Typical update might look like:
$$
S_{t+1} = f(S_{t}, \text{default/repaid}, \text{credit utilization}, \dots) + \epsilon_{t+1},
$$
*   If default occurs, subtract a penalty or push $S_{t+1}$ closer to $\underline{S}$.
*   Over time, if one continues to pay on time, $S$ gradually drifts upward.
*   In both discrete and continuous cases, $\Psi$ can also incorporate an exogenous fade‐out of negative marks after $X$ years.

4.  **Lenders’ Problem and Pricing**

4.1. **Zero‐Profit Pricing**

Lenders observe $\omega_{t} = (a_{t}, e_{t}, h_{t})$. They set
$$
q(\omega_{t}, a_{t+1}) = \frac{1}{1+r \mathbb{E}[\text{Repay}_{t+1} \mid \omega_{t}, a_{t+1}]}.
$$
Repay Probability depends on how the household decides next period, which in turn depends on the updated state $\omega_{t+1}$.

4.2. **Beliefs About Protected Class**

Even though the lender does not legally set prices based on a protected‐class guess, you can imagine internally they might interpret $(e_{t}, h_{t})$ to update beliefs about that class. But the public or “official” reason for different interest rates is the observed credit history state $h_{t}$. So from a modeling standpoint:

*   $h_{t}$ is the aggregator.
*   The deeper or hidden mechanism might be “Given past data, the lender sees a higher chance of membership in group that historically defaults more,” but the official dimension they price on is $(a, e, h)$.

5.  **Equilibrium and Distribution Dynamics**

To close the model:

*   Household Policy: Solve the dynamic program for each $\omega_{t}$. This yields default policies $\delta(\omega)$ (0=repay, 1=default) and asset choice $a' = \alpha(\omega)$.
*   Credit History Evolution: Given these policies, households transition through $\Psi(h_{t}, \dots)$.
*   Price Function Consistency: Lenders set $q(\omega, a')$ to earn zero profits, anticipating next period’s default decisions.
*   Stationary Distribution: Over time, the distribution $\mu_{t}(\omega)$ evolves. In a steady state, $\mu_{t+1} = \mu_{t}$.

6.  **Implementation Details**

6.1. **Numerical Strategy**

*   Discrete $h$: You can store a transition matrix for each $(a, e)$ choice. The lender’s expected default probability for $(a_{t+1}, h_{t})$ is computed via standard dynamic programming or simulation.
*   Continuous $h$: You might discretize the score or approximate it with a finite grid. The update rule can be stored or approximated using, say, piecewise linear methods or polynomial approximations.

6.2. **Calibration or Estimation**

*   Income Process: Choose an AR(1) or finite Markov chain matching data.
*   Credit State Transition:
    *   For discrete: guess penalty magnitudes for default, time thresholds for “upgrades.” Possibly match these to average transitions in real‐world credit‐bureau data.
    *   For continuous: choose parameters in $f(\cdot)$ such that simulated scores match real distribution of FICO or Vantage scores after defaults, late payments, etc.

6.3. **Additional Features**

*   Exogenous “Age of Credit History”: If you want to replicate the real idea that new borrowers start with “no credit,” you can incorporate an additional dimension for “length of credit file,” or let $h_{0}$ be some initial distribution.
*   Welfare or Policy Analysis: Evaluate how changes in $\Psi$ (e.g., negative info stays longer, or mandated forgiveness of negative marks) affects interest rates, default patterns, or average credit access across different subgroups.

7.  **Compatibility with Hidden Protected Class**

The point of your question is how to do all this without the credit history being simply “the lender’s posterior on protected class.” This framework accomplishes that:

*   No direct reference to $\Pr(\text{protected} = 1)$ in $\Psi$.
*   Actions (repayment, default, etc.) drive the credit‐history aggregator $h$.
*   Lenders set prices based on $(a, e, h)$. Yes, they might secretly interpret $h$ as partially revealing protected status, but the official or modeled scoring function is about the borrower’s credit performance.
*   In equilibrium, statistical discrimination can arise if group‐1 individuals systematically have different earnings or default patterns, causing them to cluster in lower credit states $M_{i}$ or lower continuous scores $S$. That is a real‐world phenomenon: if historically disadvantaged groups have more frequent delinquencies (due to lower or more volatile incomes), they wind up with worse credit histories, higher interest rates, and so on—even though the credit scoring formula is not explicitly about race/gender.

8.  **Summary**

By defining $h_{t}$ as a generic “credit history” state—either discrete or continuous—and tying the lender’s zero‐profit pricing to $h_{t}$ alongside $(a_{t}, e_{t})$, you build a unified framework that:

*   Captures the dynamic feedback from past repayment decisions to future credit access.
*   Avoids explicitly labeling the credit‐history index as a “posterior on protected class.”
*   Enables you to analyze how individuals sort into different credit states over time and how that might correlate with unobserved protected‐class membership—generating potentially disparate outcomes.

**Next Steps:**

*   Specify the functional form for $\Psi$ (discrete or continuous).
*   Integrate the entire dynamic programming plus lender‐pricing loop.
*   Simulate or compute equilibrium outcomes, paying attention to how different subpopulations (with different initial conditions or income processes) evolve across the credit history states.

This approach yields a realistic and policy‐relevant model of consumer credit where “credit history” is front and center, while still leaving room for unobserved protected‐class status to influence outcomes in subtle ways.