import inspect
import pydot
import matplotlib.pyplot as plt
import colorsys
import random
import os
from collections import defaultdict
from skagent.model import Control, DBlock, RBlock

class DBlockVisualizer:
    """
    A tool class for visualizing DBlock model structure.
    Extracts information from DBlock objects and generates logical flowcharts.
    Supports identification of time dependencies to create a directed acyclic graph.
    """
    
    def __init__(self, block, agent_attribution=None, calibration=None):
        """
        Initialize visualizer
        
        Parameters:
        block -- DBlock object to visualize
        agent_attribution -- Dictionary of variables belonging to different entities, e.g. {"consumer": [...], "lender": [...]}
        calibration -- Model calibration parameters, used if provided
        """
        self.block = block
        self.agent_attribution = agent_attribution or {}
        self.calibration = calibration or {}
        
        # Store extracted information
        self.variables = {}
        self.dependencies = {}
        self.formulas = {}
        self.prev_period_vars = set()
        
        # Use calibration parameters as model parameters
        self.parameters = self.calibration
    
    def analyze(self):
        """
        Analyze DBlock and extract all necessary information
        """
        self._identify_variables()
        self._extract_dependencies()
        self._identify_time_dependencies()
        self._extract_formulas()
        return self
    
    def _identify_variables(self):
        """Identify and classify all variables in DBlock"""
        variables = {
            'shock_vars': [],     # Random shocks
            'state_vars': [],     # State variables
            'control_vars': [],   # Control variables
            'reward_vars': [],    # Reward variables
            'param_vars': set()   # Model parameters
        }
        
        # Track all classified model variables to avoid duplicates
        classified_vars = set()
        
        # 1) Extract random shock variables (if present)
        if hasattr(self.block, 'shocks'):
            for var_name in self.block.shocks:
                variables['shock_vars'].append(var_name)
                classified_vars.add(var_name)
        
        # 2) Identify control and state variables from dynamics (if present)
        if hasattr(self.block, 'dynamics'):
            for var_name, rule in self.block.dynamics.items():
                if isinstance(rule, Control):
                    # If rule is a Control instance, treat as control variable
                    variables['control_vars'].append(var_name)
                else:
                    variables['state_vars'].append(var_name)
                classified_vars.add(var_name)
        
        # 3) Extract reward variables (if present)
        if hasattr(self.block, 'reward'):
            for var_name in self.block.reward:
                if var_name not in classified_vars:  # Avoid duplicates
                    variables['reward_vars'].append(var_name)
                    classified_vars.add(var_name)
        
        # 4) Add known parameters, but only if not already classified as model variables
        for param_name in self.parameters:
            if param_name not in classified_vars:
                variables['param_vars'].add(param_name)
        
        # Print validation information
        print("Var Type:")
        for var_type, var_list in variables.items():
            if isinstance(var_list, list):
                print(f"  {var_type}: {var_list}")
            elif isinstance(var_list, set) and var_list:
                print(f"  {var_type}: {sorted(list(var_list))}")
        
        self.variables = variables
    
    def _extract_dependencies(self):
        """Extract dependencies between variables"""
        dependencies = {}
        param_deps = defaultdict(set)  # Parameters to variables influence
        
        # 1. Analyze shocks and their parameter dependencies
        if hasattr(self.block, 'shocks'):
            for var_name, shock_def in self.block.shocks.items():
                if isinstance(shock_def, tuple) and len(shock_def) == 2:
                    shock_type, shock_params = shock_def
                    deps = set()  # Use set to avoid duplicates
                    
                    # Process shock parameters
                    if isinstance(shock_params, dict):
                        for param_name, param_value in shock_params.items():
                            if isinstance(param_value, str):
                                # If it's a reference to a parameter value
                                if param_value in self.parameters or param_value in self.variables.get('param_vars', set()):
                                    deps.add(param_value)
                                    # Record parameter influence on variables
                                    param_deps[param_value].add(var_name)
                    
                    dependencies[var_name] = list(deps)
        
        # 2. Analyze dynamics dependencies
        if hasattr(self.block, 'dynamics'):
            for var_name, rule in self.block.dynamics.items():
                if isinstance(rule, Control):
                    # For Control objects, use its information set as dependencies
                    deps = set(rule.iset)  # Use set to avoid duplicates
                    dependencies[var_name] = list(deps)
                    
                    # Record parameter influence on variables
                    for dep in deps:
                        if dep in self.variables.get('param_vars', set()):
                            param_deps[dep].add(var_name)
                else:
                    try:
                        sig = inspect.signature(rule)
                        # Collect all parameters
                        deps = set(sig.parameters.keys())  # Use set to avoid duplicates
                        dependencies[var_name] = list(deps)
                        
                        # Record parameter influence on variables
                        for dep in deps:
                            if dep in self.variables.get('param_vars', set()):
                                param_deps[dep].add(var_name)
                                
                        # Add unclassified parameters to parameter list
                        for param in deps:
                            if not any(param in var_list for var_type, var_list in self.variables.items() 
                                     if var_type != 'param_vars' and isinstance(var_list, (list, set))):
                                self.variables['param_vars'].add(param)
                    except Exception as e:
                        dependencies[var_name] = []
                        print(f"Warning: Unable to extract dependencies for variable {var_name}: {e}")
        
        # 3. Analyze reward dependencies
        if hasattr(self.block, 'reward'):
            for var_name, reward_fn in self.block.reward.items():
                try:
                    sig = inspect.signature(reward_fn)
                    deps = set(sig.parameters.keys())  # Use set to avoid duplicates
                    dependencies[var_name] = list(deps)
                    
                    # Record parameter influence on variables
                    for dep in deps:
                        if dep in self.variables.get('param_vars', set()):
                            param_deps[dep].add(var_name)
                            
                    # Add unclassified parameters to parameter list
                    for param in deps:
                        if not any(param in var_list for var_type, var_list in self.variables.items() 
                                  if var_type != 'param_vars' and isinstance(var_list, (list, set))):
                            self.variables['param_vars'].add(param)
                except Exception as e:
                    dependencies[var_name] = []
                    print(f"Warning: Unable to extract dependencies for reward variable {var_name}: {e}")
        
        # Print dependencies for validation
        print("\nDepends on:")
        for var, deps in dependencies.items():
            print(f"  {var} depends on: {deps}")
        
        # Print parameter influence relationships
        print("\nParameters affect:")
        for param, affected_vars in param_deps.items():
            print(f"  {param} affects: {sorted(list(affected_vars))}")
        
        self.dependencies = dependencies
        self.param_deps = {p: list(v) for p, v in param_deps.items()}  # Convert to lists
    
    def _identify_time_dependencies(self):
        """
        Identify variables that should be marked as from previous time period
        based strictly on their order of appearance in definitions.
        
        Logic:
        1. The first appearance of a variable in its own definition is a reference to its previous period value
        2. Any reference to an undefined variable is a reference to its previous period value
        3. Once a variable is defined, subsequent references to it are to its current period value
        """
        # If dynamics is not present, no time dependencies to identify
        if not hasattr(self.block, 'dynamics'):
            self.prev_period_vars = set()
            return set()
        
        # Get the order of variable definitions in dynamics
        ordered_vars = list(self.block.dynamics.keys())
        
        # Track variables that have been defined (current period values)
        defined_vars = set()
        
        # Variables marked as requiring previous period values
        prev_period_vars = set()
        
        # Dependency relationships that should use previous period values
        prev_period_deps = set()  # Pairs of (dependent_var, dependency_var)
        
        # First part: collect all variable dependencies
        var_dependencies = {}
        for var_name, rule in self.block.dynamics.items():
            deps = []
            if isinstance(rule, Control):
                deps = list(rule.iset)
            else:
                try:
                    sig = inspect.signature(rule)
                    deps = list(sig.parameters.keys())
                except:
                    deps = []
            
            # Filter to only include model variables
            model_deps = [dep for dep in deps if dep in ordered_vars]
            var_dependencies[var_name] = model_deps
        
        # Second part: identify which dependencies should be previous period values
        for var_name in ordered_vars:
            deps = var_dependencies[var_name]
            
            for dep in deps:
                # Case 1: Self-reference in definition (always previous period)
                if dep == var_name:
                    prev_period_vars.add(dep)
                    prev_period_deps.add((var_name, dep))
                
                # Case 2: Reference to an undefined variable (previous period)
                elif dep not in defined_vars:
                    prev_period_vars.add(dep)
                    prev_period_deps.add((var_name, dep))
                
                # Case 3: Reference to already defined variable (current period)
                # No action needed as these are current period by default
            
            # Mark current variable as defined
            defined_vars.add(var_name)
        
        # Report identified previous period variables
        print("\nPrevious Period Variables:")
        print(f"  {sorted(list(prev_period_vars))}")
        
        print("\nPrevious Period Dependencies:")
        for dep_var, dep in sorted(prev_period_deps):
            print(f"  {dep_var} depends on {dep}*")
        
        self.prev_period_vars = prev_period_vars
        self.prev_period_deps = prev_period_deps
        return prev_period_vars
    
    def _extract_formulas(self):
        """Extract formulas for variables"""
        formulas = {}
        
        # Extract formulas from dynamics (if present)
        if hasattr(self.block, 'dynamics'):
            for var_name, rule in self.block.dynamics.items():
                if isinstance(rule, Control):
                    # For Control variables, show as Control(information set)
                    deps_str = ', '.join(sorted(list(rule.iset)))
                    formulas[var_name] = f"{var_name} = Control({deps_str})"
                else:
                    try:
                        source = inspect.getsource(rule).strip()
                        if "lambda" in source:
                            formula = source.split(":", 1)[1].strip() if ":" in source else str(rule)
                            
                            # Get our specific previous period dependencies if available
                            prev_period_deps = getattr(self, 'prev_period_deps', set())
                            
                            # Modify formula to show time dependencies with star/prime notation
                            # based on specific dependencies
                            for dep_var, dep in prev_period_deps:
                                if dep_var == var_name and dep in formula:
                                    if f" {dep}" in formula:
                                        formula = formula.replace(f" {dep}", f" {dep}*")
                                    if f"({dep}" in formula:
                                        formula = formula.replace(f"({dep}", f"({dep}*")
                                    if f"[{dep}" in formula:
                                        formula = formula.replace(f"[{dep}", f"[{dep}*")
                            
                            formulas[var_name] = f"{var_name} = {formula}"
                        else:
                            formulas[var_name] = f"{var_name} = [Function]"
                    except:
                        formulas[var_name] = f"{var_name} = [Complex Function]"
        
        # Extract formulas from reward (if present)
        if hasattr(self.block, 'reward'):
            for var_name, reward_fn in self.block.reward.items():
                try:
                    source = inspect.getsource(reward_fn).strip()
                    if "lambda" in source:
                        formula = source.split(":", 1)[1].strip() if ":" in source else str(reward_fn)
                        
                        # Get our specific previous period dependencies if available
                        prev_period_deps = getattr(self, 'prev_period_deps', set())
                        
                        # Modify formula to show time dependencies with star/prime notation
                        # based on specific dependencies
                        for dep_var, dep in prev_period_deps:
                            if dep_var == var_name and dep in formula:
                                if f" {dep}" in formula:
                                    formula = formula.replace(f" {dep}", f" {dep}*")
                                if f"({dep}" in formula:
                                    formula = formula.replace(f"({dep}", f"({dep}*")
                                if f"[{dep}" in formula:
                                    formula = formula.replace(f"[{dep}", f"[{dep}*")
                        
                        formulas[var_name] = f"{var_name} = {formula}"
                    else:
                        formulas[var_name] = f"{var_name} = [Function]"
                except:
                    formulas[var_name] = f"{var_name} = [Complex Function]"
        
        # Add parameter values as formulas
        for param_name, param_value in self.parameters.items():
            if param_name in self.variables.get('param_vars', set()):
                formulas[param_name] = f"{param_name} = {param_value}"
        
        # Print formulas for validation
        print("\nEquation:")
        for var, formula in sorted(formulas.items()):
            print(f"  {formula}")
        
        # Corrected line: assign the formulas to self.formulas
        self.formulas = formulas
    
    def get_agent_for_variable(self, var_name):
        """Return the entity to which the variable belongs"""
        # Handle previous period variables - strip the suffix
        base_var = var_name.replace("*", "")
        
        # Remove _prev suffix used internally
        if base_var.endswith("_prev"):
            base_var = base_var[:-5]
        
        for agent, vars_list in self.agent_attribution.items():
            if base_var in vars_list:
                return agent
        return "other"
    
    def create_graph(self, output_file=None, show=True, color_seed=42):
        """
        Create a visual graph of the model using pydot and optionally display
        
        Parameters:
        output_file -- Optional output file path
        show -- Whether to display the image
        color_seed -- Seed for color generation (default: 42)
        
        Returns:
        pydot.Dot object representing the model structure graph
        """
        if not self.variables or not self.dependencies:
            self.analyze()
        
        # Define different shapes for different variable types
        shape_map = {
            "shock_vars": "triangle",
            "state_vars": "ellipse",
            "control_vars": "box",
            "reward_vars": "diamond",
            "param_vars": "hexagon"  # Hexagon for parameters
        }
        
        # Set up random color generation with seed for reproducibility
        random.seed(color_seed)
        
        # Function to generate distinct colors
        def generate_distinct_colors(n):
            colors = []
            for i in range(n):
                # Use golden ratio to create well-distributed hues
                h = (i * 0.618033988749895) % 1.0
                # Medium-high saturation and lightness for good visibility
                s = 0.5 + random.random() * 0.2
                l = 0.4 + random.random() * 0.2
                
                # Convert HSL to RGB
                r, g, b = colorsys.hls_to_rgb(h, l, s)
                
                # Convert to hex
                hex_color = "#{:02x}{:02x}{:02x}".format(
                    int(r * 255), int(g * 255), int(b * 255)
                )
                colors.append(hex_color)
            return colors
        
        # Get all unique agents
        all_agents = list(self.agent_attribution.keys())
        if "other" not in all_agents:
            all_agents.append("other")
        
        # Generate colors for agents
        agent_colors = {}
        distinct_colors = generate_distinct_colors(len(all_agents))
        for i, agent in enumerate(all_agents):
            agent_colors[agent] = distinct_colors[i]
        
        # Create directed graph, left to right layout
        graph = pydot.Dot("ModelStructure", graph_type="digraph", rankdir="LR")
        
        # Create subgraphs to group entities
        agent_subgraphs = {}
        for agent in all_agents:
            agent_subgraph = pydot.Cluster(
                f"cluster_{agent}",
                label=agent.capitalize(),
                style="filled",
                fillcolor="#FFFFFF",
                color=agent_colors.get(agent, "#708090"),
                fontcolor=agent_colors.get(agent, "#708090")
            )
            agent_subgraphs[agent] = agent_subgraph
            graph.add_subgraph(agent_subgraph)
        
        # Track created nodes
        created_nodes = {}
        
        # Add nodes for previous period variables
        for var in self.prev_period_vars:
            prev_var_id = f"{var}_prev"  # Unique ID for the node
            
            # Determine variable type and shape
            var_type = None
            for type_name, var_list in self.variables.items():
                if isinstance(var_list, list) and var in var_list:
                    var_type = type_name
                    break
                elif isinstance(var_list, set) and var in var_list:
                    var_type = type_name
                    break
            
            if var_type:
                shape = shape_map.get(var_type, "ellipse")
                agent = self.get_agent_for_variable(var)
                color = agent_colors.get(agent, agent_colors.get("other", "#708090"))
                
                prev_node = pydot.Node(
                    prev_var_id,
                    label=f"{var}*",  # Using star/prime notation
                    shape=shape,
                    style="filled,dashed",  # Dashed style for previous period
                    fillcolor=color,
                    color=color,
                    fontcolor="#000000",
                    fontname="Arial",
                    tooltip=f"Previous period {var}"
                )
                agent_subgraphs[agent].add_node(prev_node)
                created_nodes[prev_var_id] = prev_node
        
        # Add all other variables as nodes
        for var_type, var_list in self.variables.items():
            if isinstance(var_list, list) or isinstance(var_list, set):
                var_items = var_list if isinstance(var_list, list) else list(var_list)
                for var in var_items:
                    if var not in created_nodes:
                        shape = shape_map.get(var_type, "ellipse")
                        agent = self.get_agent_for_variable(var)
                        color = agent_colors.get(agent, agent_colors.get("other", "#708090"))
                        
                        node = pydot.Node(
                            var,
                            label=var,
                            shape=shape,
                            style="filled",
                            fillcolor=color, 
                            color=color,
                            fontcolor="#000000",
                            fontname="Arial",
                            tooltip=self.formulas.get(var, "")
                        )
                        agent_subgraphs[agent].add_node(node)
                        created_nodes[var] = node
        
        # Create modified dependencies based on specific previous period dependencies
        modified_dependencies = {}
        
        # Get our specific previous period dependencies if available
        prev_period_deps = getattr(self, 'prev_period_deps', set())
        
        for var, deps in self.dependencies.items():
            modified_deps = []
            for dep in deps:
                # Only model variables (in ordered_vars) can be previous period deps
                if dep in getattr(self, 'variables', {}).get('param_vars', set()):
                    # Parameters are always current period
                    modified_deps.append(dep)
                elif (var, dep) in prev_period_deps:
                    # This specific dependency should use previous period
                    modified_deps.append(f"{dep}_prev")
                else:
                    # Regular current period dependency
                    modified_deps.append(dep)
            
            modified_dependencies[var] = modified_deps
        
        # Track created edges to avoid duplicates
        created_edges = set()
        
        # Add dependency edges
        for var, deps in modified_dependencies.items():
            if var in created_nodes:
                for dep in deps:
                    if dep in created_nodes and dep != var:
                        # Create unique edge identifier
                        edge_id = (dep, var)
                        
                        # Check if edge already exists
                        if edge_id not in created_edges:
                            agent = self.get_agent_for_variable(var)
                            color = agent_colors.get(agent, agent_colors.get("other", "#708090"))
                            
                            # For previous period variables use dashed lines
                            style = "dashed" if dep.endswith("_prev") else "solid"
                            
                            edge = pydot.Edge(
                                dep,
                                var,
                                color=color,
                                style=style,
                                arrowhead="normal",
                                arrowsize=0.8
                            )
                            graph.add_edge(edge)
                            created_edges.add(edge_id)
        
        # Add edges from parameters to variables they affect
        if hasattr(self, 'param_deps'):
            for param, affected_vars in self.param_deps.items():
                if param in created_nodes:
                    for var in affected_vars:
                        if var in created_nodes:
                            # Create unique edge identifier
                            edge_id = (param, var)
                            
                            # Check if edge already exists
                            if edge_id not in created_edges:
                                # Use parameter's agent color or default
                                param_agent = self.get_agent_for_variable(param)
                                param_color = agent_colors.get(param_agent, agent_colors.get("other", "#708090"))
                                
                                edge = pydot.Edge(
                                    param,
                                    var,
                                    color=param_color,
                                    style="solid",
                                    arrowhead="normal",
                                    arrowsize=0.8
                                )
                                graph.add_edge(edge)
                                created_edges.add(edge_id)
        
        # Save or show image
        if output_file:
            graph.write_png(output_file)
        if show:
            temp_file = output_file or "temp_model_diagram.png"
            graph.write_png(temp_file)
            img = plt.imread(temp_file)
            plt.figure(figsize=(12, 10))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"{self.block.name if hasattr(self.block, 'name') else 'Model'} Structure")
            plt.tight_layout()
            plt.show()
            if not output_file:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        return graph

def visualize_dblock(block, agent_attribution=None, calibration=None, output_file=None, show=True):
    """
    Create and display visualization of a DBlock object
    
    Parameters:
    block -- DBlock object to visualize
    agent_attribution -- Dictionary of variables belonging to different entities
    calibration -- Model calibration parameters
    output_file -- Optional output file path
    show -- Whether to display the image
    
    Returns:
    pydot.Dot object representing the model structure graph
    """
    visualizer = DBlockVisualizer(block, agent_attribution, calibration)
    visualizer.analyze()
    return visualizer.create_graph(output_file, show)