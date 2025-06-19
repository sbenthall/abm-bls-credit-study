from skagent.model import RBlock, DBlock, Control
from HARK.distributions import Bernoulli, Lognormal, MeanOneLogNormal, Normal

calibration = {
    "p" : 0, # protected attribute, this needs to be heterogeneous among consumers
    "beta": 0.9, # discount factor, potentially heterogeneous
    "eta" : 0.9, # credit history forgetting rate
    "crra" : 2, # coefficient of relative risk aversion, used in consumer utility function
    "r*" : .1, # bank borrowing rate
    "r_ceiling" : 1, # usury limit/max bank interest rate
    "a_bar" : 100, # artificial borrowing constraint
}

# TODO #1: How will we do shocks that depend on parameters and/or state variables?
# TODO #2: How will we handle Controls that have one or two bounds? With expressions...
# TODO #3: Can we train a Control with zero inputs? (I.e., a constant to be trained.)
# TODO #4: How will we deal with training for a block with more than one control variable (for a single agent)?
# TODO #5: How will we deal with training for blocks where there are more than one agent governing controls, pursuing reward?
# TODO #6: How do we encode that there are N consumers and only one lender, for this model?

consumption_block = DBlock(
    **{
        "name": "consumption",
        "shocks": {
            "ze": (Normal, {"mu" : 0, "sigma": 1}), # TODO: Decide the earnings shock.
                                                      # Will it depend on p?
        },
        "dynamics": {
            "e" : lambda e, z: e + z, # TODO: Are we letting earnings walk like this?
                                      #       We could also have this be the temporary shock directly.
            "c" : Control(
                ["a", "e", "h", "p"],
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
        },
        "dynamics": {
            "phi_d" : Control([], agent="lender"), # These parameters can be learned by the lender
            "phi_e" : Control([], agent="lender"), # but are in effect not idiosyncratic -- they are
                                                   # just constants chosen in strategic equilibrium
            "q" : Control(["k", "e", "h", "p", "r*"],
                          upper_bound = "r_ceiling", # TODO: Likewise as above -- how will we deal with bounded controls?
                                                     # ... especially, bounded in one direction. (could use a ReLu function)
                          agent="lender"), # r* is constant; use of p is only in some cases
                                                 # TODO: Need to decide if the agent assignment happens at the block or Control object.
                                                 # This is the rate on the previous period's assets.
            "d" : Control(["k", "e", "h", "q", "p"],
                          upper_bound = 1, # this is for the continuous version.
                          lower_bound = 0,
                          agent="consumer"
                          ),

            "a" : lambda d, k, q: (1 - d) * k * q,
            "zh": lambda ezh, p: ezh * p, # TODO: How are we going to do p-correlated shocks?
            "h" : lambda e, h, d, zh, phi_d, phi_e, eta: eta * h + phi_d * d + phi_e * e + zh,
            "pi" : lambda a, d, q: - (1 - d) * k * (q - 1) # profit
        },
        "reward": {
            "pi" : "lender"
        },
    }
)

cons_credit_problem = RBlock(
    blocks=[consumption_block, lending_block]
)