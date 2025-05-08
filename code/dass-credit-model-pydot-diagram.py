import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import pydot
    from IPython.core.display import SVG

    from skagent.model import DBlock, Control
    from HARK.distributions import Bernoulli, Lognormal, MeanOneLogNormal, Normal
    return Control, DBlock, Normal, SVG, pydot


@app.cell
def _():
    calibration = {
        "p" : 0, # protected attribute, this needs to be heterogeneous among consumers
        "beta": 0.9, # discount factor, potentially heterogeneous
        "eta" : 0.9, # credit history forgetting rate
        "crra" : 2, # coefficient of relative risk aversion, used in consumer utility function
        "r*" : .1, # bank borrowing rate
    
    }
    return


@app.cell
def _(pydot):
    def add_node(dot, id, label, nodeattrs = {}):
        node = pydot.Node(id, label=label)
        for key in nodeattrs:
            node.set(key, nodeattrs[key])
        dot.add_node(node)
    return (add_node,)


@app.cell
def _(Control, DBlock, Normal, m):
    consumption_block = DBlock(
        **{
            "name": "consumption",
            "shocks": {
                "ze": (Normal, {"mu" : 0, "sigma": "1"}), # TODO: Decide the earnings shock.
            },
            "dynamics": {
                "e" : lambda e, z: e + z, # TODO: Are we letting earnings walk like this?
                "c" : Control(["a", "e", "h", "p"]),
                "k" : lambda a, c: m - c,
            },
            "reward": {
                "u" : lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA),
            },
        }
    )
    return


@app.cell
def _(SVG, add_node, pydot):
    cbdot = pydot.Dot()                                                           
    cbdot.set('rankdir', 'TB')                                                    
    cbdot.set('concentrate', True)                                                
    cbdot.set_node_defaults(shape='record')

    # Create a graph and set defaults

    # Consumer block

    add_node(cbdot, 'p', 'protected attribute', nodeattrs={'shape':'plain'})
    add_node(cbdot, 'beta', 'discount factor', nodeattrs={'shape':'plain'})

    add_node(cbdot, 'ze', 'e-shock', nodeattrs={'shape':'doublecircle'})
    add_node(cbdot, 'e-', 'earnings-', nodeattrs={'shape':'ellipse'})


    #add_node(cbdot, 'zh', 'h-shock', nodeattrs={'shape':'doublecircle'})
    add_node(cbdot, 'theta_c', 'theta_c', nodeattrs={'shape':'trapezium', 'fillcolor':'aliceblue'})
    add_node(cbdot, 'a', 'wealth', nodeattrs={'shape':'ellipse'})
    add_node(cbdot, 'e', 'earnings', nodeattrs={'shape':'ellipse'})
    add_node(cbdot, 'c', 'consumption', nodeattrs={'fillcolor':'aliceblue'})
    add_node(cbdot, 'u', 'utility', nodeattrs={'shape':'diamond', 'fillcolor':'aliceblue'})
    add_node(cbdot, 'c', 'consumption', nodeattrs={'fillcolor':'aliceblue'})
    add_node(cbdot, 'u', 'utility', nodeattrs={'shape':'diamond', 'fillcolor':'aliceblue'})
    add_node(cbdot, 's', 'savings', nodeattrs={'shape':'ellipse'})
    #add_node(cbdot, 'd', 'default', nodeattrs={'fillcolor':'aliceblue'})
    add_node(cbdot, 'h', 'history', nodeattrs={'shape':'ellipse'})

    cbdot.add_edge( pydot.Edge('p','e', **{'style': 'dotted'}))
    #cbdot.add_edge( pydot.Edge('p','zh', **{'style': 'dotted'}))

    cbdot.add_edge( pydot.Edge('ze','e'))
    cbdot.add_edge( pydot.Edge('e-','e'))
    #cbdot.add_edge( pydot.Edge('zh','h+'))
    #cbdot.add_edge( pydot.Edge('p','q', **{'style': 'dotted'}))
    cbdot.add_edge( pydot.Edge('a','c'))
    cbdot.add_edge( pydot.Edge('theta_c','c'))
    cbdot.add_edge( pydot.Edge('h','c'))
    cbdot.add_edge( pydot.Edge('e','c'))
    cbdot.add_edge( pydot.Edge('c','s'))
    cbdot.add_edge( pydot.Edge('a','s'))

    cbdot.add_edge( pydot.Edge('e','s'))

    cbdot.add_edge( pydot.Edge('c','u'))

    #dot.add_edge( pydot.Edge('F','S'))

    for n in cbdot.get_nodes():
        n.set('style', 'filled')
        #n.set('fillcolor', 'aliceblue')
        n.set('fontsize', '10')
        n.set('fontname', 'Trebuchet MS, Tahoma, Verdana, Arial, Helvetica, sans-serif')

    SVG( data=cbdot.create_svg() )
    return


@app.cell
def _(Control, DBlock, Normal, e, eta, k, phi_d, zh):
    lending_block = DBlock(
        **{
            "name": "lending",
            "shocks": {
                "ezh": (Normal, {"mu" : 0, "sigma": "1"}), # TODO: Decide the earnings shock.
            },
            "dynamics": {
                "q" : Control(["k", "e", "h", "p", "r*"]), # r* is constant; use of p is only in some cases
                                                     # TODO: Need to decide if the agent assignment happens at the block or Control object.
                                                     # This is the rate on the previous period's assets.
                "d" : Control(["k", "e", "h", "p"]),

                "a" : lambda d, k, q: (1 - d) * k * q,
                "zh": lambda ezh, p: ezh * p, # or something
                "h" : lambda h, d: eta * h + phi_d * d + phi_d * e + zh,
            },
            "reward": {
                "pi" : lambda a, d, q: - (1 - d) * k * (q - 1) # profit
            },
        }
    )
    return


@app.cell
def _(SVG, add_node, pydot):
    lbdot = pydot.Dot()                                                           
    lbdot.set('rankdir', 'TB')                                                    
    lbdot.set('concentrate', True)                                                
    lbdot.set_node_defaults(shape='record')

    # Create a graph and set defaults

    # Consumer block

    add_node(lbdot, 'p', 'protected attribute', nodeattrs={'shape':'plain'})
    add_node(lbdot, 'zh', 'h-shock', nodeattrs={'shape':'doublecircle'})
    add_node(lbdot, 'e', 'earnings', nodeattrs={'shape':'ellipse'})
    add_node(lbdot, 's', 'savings', nodeattrs={'shape':'ellipse'})
    add_node(lbdot, 'h', 'history', nodeattrs={'shape':'ellipse'})
    add_node(lbdot, 's', 'savings', nodeattrs={'shape':'ellipse'})

    add_node(lbdot, 'theta_d', 'theta_d', nodeattrs={'shape':'trapezium', 'fillcolor':'aliceblue'})
    add_node(lbdot, 'd', 'default', nodeattrs={'fillcolor':'aliceblue'})
    #add_node(lbdot, 'h', 'history', nodeattrs={'shape':'ellipse'})

    # Lender block

    add_node(lbdot, 'phi_q', 'phi_q', nodeattrs={'shape':'trapezium', 'fillcolor':'yellow'})
    add_node(lbdot, 'r*', 'risk-free lender rate', nodeattrs={'shape':'plain'})
    add_node(lbdot, 'q', 'interest rate', nodeattrs={'fillcolor':'yellow'})
    add_node(lbdot, 'f', 'forgeting rate', nodeattrs={'shape':'plain'})

    add_node(lbdot, 'phi_h', 'phi_h', nodeattrs={'shape':'trapezium', 'fillcolor':'yellow'})
    add_node(lbdot, 'h+', 'history+', nodeattrs={'shape':'ellipse'})
    add_node(lbdot, 'pi', 'profit', nodeattrs={'shape':'diamond', 'fillcolor':'yellow'})

    add_node(lbdot, 'a+', 'wealth+', nodeattrs={'shape':'ellipse'})


    # Lender blok

    lbdot.add_edge( pydot.Edge('p','h+', **{'style': 'dotted'}))

    lbdot.add_edge( pydot.Edge('p','q', **{'style': 'dotted'}))

    lbdot.add_edge( pydot.Edge('r*','q'))
    lbdot.add_edge( pydot.Edge('phi_q','q'))
    lbdot.add_edge( pydot.Edge('s','q'))
    lbdot.add_edge( pydot.Edge('h','q'))
    lbdot.add_edge( pydot.Edge('e','q'))

    lbdot.add_edge( pydot.Edge('s','pi'))
    lbdot.add_edge( pydot.Edge('d','pi'))
    lbdot.add_edge( pydot.Edge('q','pi'))

    lbdot.add_edge( pydot.Edge('s','d'))
    lbdot.add_edge( pydot.Edge('h','d'))
    lbdot.add_edge( pydot.Edge('e','d'))
    lbdot.add_edge( pydot.Edge('q','d'))
    lbdot.add_edge( pydot.Edge('theta_d','d'))

    lbdot.add_edge( pydot.Edge('h','h+'))
    lbdot.add_edge( pydot.Edge('f','h+'))
    lbdot.add_edge( pydot.Edge('zh','h+'))
    lbdot.add_edge( pydot.Edge('d','h+'))
    lbdot.add_edge( pydot.Edge('phi_h','h+'))
    lbdot.add_edge( pydot.Edge('e','h+', **{'style': 'dotted'}))

    lbdot.add_edge( pydot.Edge('q','a+'))
    lbdot.add_edge( pydot.Edge('s','a+'))
    lbdot.add_edge( pydot.Edge('d','a+'))

    #dot.add_edge( pydot.Edge('F','S'))

    for m in lbdot.get_nodes():
        m.set('style', 'filled')
        #n.set('fillcolor', 'aliceblue')
        m.set('fontsize', '10')
        m.set('fontname', 'Trebuchet MS, Tahoma, Verdana, Arial, Helvetica, sans-serif')

    SVG( data=lbdot.create_svg() )
    return (m,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
