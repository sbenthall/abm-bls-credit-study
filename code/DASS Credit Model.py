import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import pydot
    from IPython.core.display import SVG
    return SVG, pydot


@app.cell
def _():
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
def _(add_node, pydot):
    csdot = pydot.Dot()                                                           
    csdot.set('rankdir', 'TB')                                                    
    csdot.set('concentrate', True)                                                
    csdot.set_node_defaults(shape='record')

    # Create a graph and set defaults

    add_node(csdot, 'p', 'protected attribute', nodeattrs={'shape':'plain'})
    add_node(csdot, 'beta', 'discount factor', nodeattrs={'shape':'plain'})
    add_node(csdot, 'r*', 'risk-free lender rate', nodeattrs={'shape':'plain'})

    add_node(csdot, 'ze', 'e-shock', nodeattrs={'shape':'doublecircle'})
    add_node(csdot, 'zh', 'h-shock', nodeattrs={'shape':'doublecircle'})

    add_node(csdot, 'phi', 'phi', nodeattrs={'shape':'trapezium', 'fillcolor':'yellow'})
    add_node(csdot, 'theta', 'theta', nodeattrs={'shape':'trapezium', 'fillcolor':'aliceblue'})


    add_node(csdot, 'a', 'wealth', nodeattrs={'shape':'ellipse'})
    add_node(csdot, 'e', 'earnings??')
    add_node(csdot, 'q', 'interest rate', nodeattrs={'fillcolor':'yellow'})
    add_node(csdot, 'c', 'consumption', nodeattrs={'fillcolor':'aliceblue'})
    add_node(csdot, 'u', 'utility', nodeattrs={'shape':'diamond', 'fillcolor':'aliceblue'})
    add_node(csdot, 'd', 'default', nodeattrs={'fillcolor':'aliceblue'})
    add_node(csdot, 'a+', 'wealth+', nodeattrs={'shape':'ellipse'})
    add_node(csdot, 'h', 'history', nodeattrs={'shape':'ellipse'})
    add_node(csdot, 'f', 'forgeting rate', nodeattrs={'shape':'plain'})
    add_node(csdot, 'h+', 'history+', nodeattrs={'shape':'ellipse'})
    add_node(csdot, 'pi', 'profit', nodeattrs={'shape':'diamond', 'fillcolor':'yellow'})


    # doublecircle

    #node.set("fillcolor", "aliceblue")

    csdot.add_edge( pydot.Edge('p','ze', **{'style': 'dotted'}))
    csdot.add_edge( pydot.Edge('p','zh', **{'style': 'dotted'}))


    csdot.add_edge( pydot.Edge('ze','e'))
    csdot.add_edge( pydot.Edge('zh','h+'))

    #csdot.add_edge( pydot.Edge('z','e', **{'style': 'dotted'}))
    #csdot.add_edge( pydot.Edge('z','h+', **{'style': 'dotted'}))

    csdot.add_edge( pydot.Edge('p','q', **{'style': 'dotted'}))

    csdot.add_edge( pydot.Edge('phi','h'))
    csdot.add_edge( pydot.Edge('phi','q'))
    csdot.add_edge( pydot.Edge('theta','c'))

    csdot.add_edge( pydot.Edge('r*','q'))
    csdot.add_edge( pydot.Edge('a','q'))
    csdot.add_edge( pydot.Edge('h','q'))

    csdot.add_edge( pydot.Edge('a','c'))
    csdot.add_edge( pydot.Edge('q','c'))
    csdot.add_edge( pydot.Edge('h','c'))
    csdot.add_edge( pydot.Edge('e','c'))

    csdot.add_edge( pydot.Edge('a','d'))
    csdot.add_edge( pydot.Edge('q','d'))
    csdot.add_edge( pydot.Edge('h','d'))
    csdot.add_edge( pydot.Edge('e','d'))

    csdot.add_edge( pydot.Edge('e','h+', **{'style': 'dotted'}))


    csdot.add_edge( pydot.Edge('c','a+'))
    csdot.add_edge( pydot.Edge('a','a+'))
    csdot.add_edge( pydot.Edge('q','a+'))
    csdot.add_edge( pydot.Edge('e','a+'))
    csdot.add_edge( pydot.Edge('c','u'))
    csdot.add_edge( pydot.Edge('d','a+'))

    csdot.add_edge( pydot.Edge('h','h+'))
    csdot.add_edge( pydot.Edge('d','h+'))
    csdot.add_edge( pydot.Edge('f','h+'))

    csdot.add_edge( pydot.Edge('a+','pi'))
    csdot.add_edge( pydot.Edge('d','pi'))
    csdot.add_edge( pydot.Edge('q','pi'))

    #dot.add_edge( pydot.Edge('F','S'))
    return (csdot,)


@app.cell
def _(SVG, csdot):
    for n in csdot.get_nodes():
        n.set('style', 'filled')
        #n.set('fillcolor', 'aliceblue')
        n.set('fontsize', '10')
        n.set('fontname', 'Trebuchet MS, Tahoma, Verdana, Arial, Helvetica, sans-serif')

    SVG( data=csdot.create_svg() )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
