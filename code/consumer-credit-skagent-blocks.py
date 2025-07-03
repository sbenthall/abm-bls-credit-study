from skagent.model import RBlock, DBlock, Control
from HARK.distributions import Bernoulli, Lognormal, MeanOneLogNormal, Normal

calibration = {
    "p" : 0, # protected attribute, this needs to be heterogeneous among consumers
        # TODO: Vector valued to reflect the population....
        #      -- might be distributed as a 'shock', or initial exogenous distribution.
    "fac_ep" : "1", # effect of p on e # TODO sweep over this?
    "beta": 0.9, # discount factor, potentially heterogeneous
    "eta" : 0.9, # credit history forgetting rate # TODO : Parameter sweeping over this.
    "crra" : 2, # coefficient of relative risk aversion, used in consumer utility function
    "r*" : .1, # bank borrowing rate
    "r_ceiling" : 1, # usury limit/max bank interest rate # TODO : Parameter sweeping over this.
    "a_bar" : 100, # artificial borrowing constraint
}

# TODO #1: How will we do shocks that depend on parameters and/or state variables?
# TODO #2: How will we handle Controls that have one or two bounds? With expressions...
#       https://github.com/scikit-agent/scikit-agent/issues/66
# TODO #3: Can we train a Control with zero inputs? (I.e., a constant to be trained.)
#       https://github.com/scikit-agent/scikit-agent/issues/67
# TODO #4: How will we deal with training for a block with more than one control variable (for a single agent)?
#       https://github.com/scikit-agent/scikit-agent/issues/68
# TODO #5: How will we deal with training for blocks where there are more than one agent governing controls, pursuing reward?
# TODO #6: How do we encode that there are N consumers and only one lender, for this model?

consumption_block = DBlock(
    **{
        "name": "consumption",
        "shocks": {
            "ze": (Normal, {"mu" : "1 + fac_ep * p - 0.5 * fac_ep", "sigma": 1}), # TODO: Decide the earnings shock.
        },
        "dynamics": {
            "e" : lambda e, ze: e + (e) ** ze,# ??? -  # TODO: Find the exact form of this structural equation.
            "c" : Control(
                ["a", "e", "h", "p", "a_bar"],
                lower_bound = 0,
                upper_bound = "a + e + a_bar", # ALTERNATIVE: let this range from [0,1]
                                               # and compute an intermediate state variable afterwards
                agent="consumer"
            ),
            "k" : lambda a, c, e: a - c + e,
            "u" : lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA),
        },
        "reward": {
            "u" : "consumer"
        },
    }
)

lending_block = DBlock(
    **{
        "name": "lending",
        "shocks": {
            "ezh": (Normal, {"mu" : 0, "sigma": "1"}), # TODO: Decide the history shock.
                                                       # TODO: This too might be correlated with p.
        },
        "dynamics": {
            "q" : Control(["k", "e", "h", "p", "r*"], # TODO: Dependence on protected attribute is one variation.
                                                      #       Sweep over equation variants.
                          upper_bound = "r_ceiling", # TODO: Likewise as above -- how will we deal with bounded controls?
                                                     # ... especially, bounded in one direction. (could use a ReLu function)
                          agent="lender"), # r* is constant; use of p is only in some cases
                                                 # TODO: Need to decide if the agent assignment happens at the block or Control object.
                                                 # This is the rate on the previous period's assets.
            "d" : Control(["k", "e", "h", "q", "p"],
                          upper_bound = 1, # this is for the continuous version. # d == 1 is DEFAULTing on debt
                          lower_bound = 0,
                          agent="consumer",
                          feasible_set = [0,1] # for the discrete option
                          ),

            "a" : lambda d, k, q: (1 - d) * k * q, # TODO : Is this right?
            "zh": lambda ezh, p: ezh * p, # TODO: How are we going to do p-correlated shocks?

            "phi_d" : Control([], agent="lender"), # These parameters can be learned by the lender
            "phi_e" : Control([], agent="lender"), # but are in effect not idiosyncratic -- they are
                                                   # just constants chosen in strategic equilibrium

                                                   # TODO: Alternatively, these could be varied to hit empirical targets
                                                   #       on distribution.
            "h" : lambda e, h, d, zh, phi_d, phi_e, eta: eta * h + phi_d * d + phi_e * e + zh,

            "pi" : lambda d, k, q: - (1 - d) * k * (q - 1) # profit
        },
        "reward": {
            "pi" : "lender"
        },
    }
)

cons_credit_problem = RBlock(
    blocks=[consumption_block, lending_block]
)