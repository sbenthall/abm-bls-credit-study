# model_visualizer.py
import pydot
import matplotlib.pyplot as plt
from inspect import signature

def create_block_diagram(block, output_path="model_structure.png"):
    """
    Creates a diagram of a model block's structure.
    
    Parameters:
    -----------
    block : The model block to visualize
    output_path : File path to save the diagram
    
    Returns:
    --------
    The generated graph object
    """
    
    # Define different colors & shapes for variable types
    style_map = {
        "Control": {"shape": "rectangle", "color": "blue"},  # Control variables (decisions)
        "Reward": {"shape": "diamond", "color": "green"},   # Rewards
        "Shock": {"shape": "triangle", "color": "red"},    # Shocks (exogenous)
        "State": {"shape": "ellipse", "color": "yellow"}    # Default for state variables
    }
    
    # Create a directed graph
    graph = pydot.Dot("ModelStructure", graph_type="digraph", rankdir="LR")
    
    # Extract variables dynamically
    for var_name, func in block.dynamics.items():
        var_type = "State"
        dependencies = []
        
        # 检查是否为Control对象 - 通过类名而不是isinstance
        is_control = (hasattr(func, "__class__") and 
                     func.__class__.__name__ == "Control")
        
        if is_control:
            var_type = "Control"
            # For Control variables, we need to determine dependencies
            # This gets parameters that the control decision depends on
            if var_name == "c":
                # Consumption typically depends on wealth in consumer models
                dependencies.append("w")
            elif var_name == "R":
                # Interest rate might depend on assets
                dependencies.append("a")
        else:
            # For non-Control variables, get all input parameters
            try:
                dependencies = list(signature(func).parameters.keys())
                
                # Special handling for AR processes (like income)
                if var_name == "y" and "y" in dependencies:
                    # Create a clearer label for the self-reference to show time dependency
                    # First remove the self-reference
                    dependencies.remove("y")
                    # Then add it with a time indicator
                    dependencies.append("y")  # Will create the self-referential arrow
                    # Note: In visualization, this represents y_t → y_{t+1}
            except:
                # If we can't get the signature, use an empty list
                pass
        
        node_style = style_map[var_type]
        graph.add_node(
            pydot.Node(
                var_name,
                shape=node_style["shape"],
                fillcolor=node_style["color"],
                style="filled"
            )
        )
        
        for dep in dependencies:
            graph.add_edge(pydot.Edge(dep, var_name))
    
    # Add Shock variables
    if hasattr(block, "shocks") and block.shocks:
        for shock in block.shocks.keys():
            node_style = style_map["Shock"]
            
            graph.add_node(
                pydot.Node(
                    shock,
                    shape=node_style["shape"],
                    fillcolor=node_style["color"],
                    style="filled"
                )
            )
    
    # Add Reward variables
    if hasattr(block, "reward") and block.reward:
        for reward_name, reward_func in block.reward.items():
            node_style = style_map["Reward"]
            
            graph.add_node(
                pydot.Node(
                    reward_name,
                    shape=node_style["shape"],
                    fillcolor=node_style["color"],
                    style="filled"
                )
            )
            
            try:
                for arg in signature(reward_func).parameters:
                    graph.add_edge(pydot.Edge(arg, reward_name))
            except:
                # If we can't get the signature, skip adding edges
                pass
    
    # Render graph and display it
    graph.write_png(output_path)
    
    # Use matplotlib to display the image
    img = plt.imread(output_path)
    plt.figure(figsize=(10, 7))
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.show()
    
    return graph