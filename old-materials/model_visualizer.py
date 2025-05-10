# model_visualizer.py
import pydot
import matplotlib.pyplot as plt
from inspect import signature

def create_block_diagram(block, output_path="model_structure.png"):
    """
    Creates a diagram of a model block's structure based on the structural equations.
    Preserves the semantic meaning of colors (agent) and shapes (variable types).
    
    Parameters:
    -----------
    block : The model block to visualize
    output_path : File path to save the diagram
    
    Returns:
    --------
    The generated graph object
    """
    
    # Define different shapes for variable types, colors represent agents
    style_map = {
        "Control": {"shape": "rectangle", "color": "blue"},   # Control variables (decisions)
        "Reward": {"shape": "diamond", "color": "green"},     # Rewards
        "Shock": {"shape": "triangle", "color": "red"},       # Shocks (exogenous)
        "State": {"shape": "ellipse", "color": "yellow"},     # Default for state variables
        "Parameter": {"shape": "ellipse", "color": "lightgrey"}  # Model parameters
    }
    
    # Create a directed graph that flows left to right
    graph = pydot.Dot("ModelStructure", graph_type="digraph", rankdir="LR")
    
    # Track created nodes to avoid duplicates
    created_nodes = set()
    
    # Add model parameters if available (will be populated from function signatures)
    model_params = set()
    
    # Extract variables from dynamics equations
    for var_name, func in block.dynamics.items():
        var_type = "State"
        dependencies = []
        
        # Check if it's a Control variable
        is_control = (hasattr(func, "__class__") and 
                      func.__class__.__name__ == "Control")
        
        if is_control:
            var_type = "Control"
            # If Control object has defined dependencies, use them
            if hasattr(func, "args") and func.args:
                dependencies = list(func.args)
        else:
            # For regular function, extract parameters from signature
            try:
                dependencies = list(signature(func).parameters.keys())
                # Add these to potential model parameters
                model_params.update(dependencies)
            except:
                pass
        
        # Create node for this variable if not already created
        if var_name not in created_nodes:
            node_style = style_map[var_type]
            graph.add_node(
                pydot.Node(
                    var_name,
                    shape=node_style["shape"],
                    fillcolor=node_style["color"],
                    style="filled"
                )
            )
            created_nodes.add(var_name)
        
        # Add edges for dependencies
        for dep in dependencies:
            # Create the dependency node if not already created
            if dep not in created_nodes:
                # Determine if it's a parameter (not in dynamics or shocks)
                dep_type = "Parameter"
                if dep in block.dynamics:
                    dep_type = "State"  # Default for variables defined in dynamics
                
                dep_style = style_map[dep_type]
                graph.add_node(
                    pydot.Node(
                        dep,
                        shape=dep_style["shape"],
                        fillcolor=dep_style["color"],
                        style="filled"
                    )
                )
                created_nodes.add(dep)
            
            # Add connection between nodes
            graph.add_edge(pydot.Edge(dep, var_name))
    
    # Add Shock variables
    if hasattr(block, "shocks") and block.shocks:
        for shock_name in block.shocks.keys():
            if shock_name not in created_nodes:
                node_style = style_map["Shock"]
                graph.add_node(
                    pydot.Node(
                        shock_name,
                        shape=node_style["shape"],
                        fillcolor=node_style["color"],
                        style="filled"
                    )
                )
                created_nodes.add(shock_name)
                
                # Connect shocks to variables that depend on them
                # This is determined from function signatures in dynamics
                for var_name, func in block.dynamics.items():
                    if not callable(func):
                        continue
                    try:
                        deps = list(signature(func).parameters.keys())
                        if shock_name in deps:
                            graph.add_edge(pydot.Edge(shock_name, var_name))
                    except:
                        pass
    
    # Add Reward variables
    if hasattr(block, "reward") and block.reward:
        for reward_name, reward_func in block.reward.items():
            if reward_name not in created_nodes:
                node_style = style_map["Reward"]
                graph.add_node(
                    pydot.Node(
                        reward_name,
                        shape=node_style["shape"],
                        fillcolor=node_style["color"],
                        style="filled"
                    )
                )
                created_nodes.add(reward_name)
            
            # Add edges from dependencies to reward
            try:
                if callable(reward_func):
                    for arg in signature(reward_func).parameters:
                        # Create the dependency node if not already created
                        if arg not in created_nodes:
                            arg_style = style_map["Parameter"]
                            graph.add_node(
                                pydot.Node(
                                    arg,
                                    shape=arg_style["shape"],
                                    fillcolor=arg_style["color"],
                                    style="filled"
                                )
                            )
                            created_nodes.add(arg)
                        
                        graph.add_edge(pydot.Edge(arg, reward_name))
            except:
                pass
    
    # Render graph and display it
    graph.write_png(output_path)
    
    img = plt.imread(output_path)
    plt.figure(figsize=(10, 7))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    return graph