"""
test.py
"""

from skagent.model import DBlock, Control
from model_viz import DBlockVisualizer
import inspect

# Create a Dummy DBlock for testing purposes
class DummyDBlock(DBlock):
    def __init__(self):
        self.name = "dummy_block"
        # Define shock variables (random disturbances). Note: shocks are not used in formula extraction.
        self.shocks = {'shock1': (None, {"mu": 0, "sigma": "param1"})}
        # 'dynamics' contains state and control variables.
        # The state variable is defined using a lambda expression that represents its dependency.
        # The control variable is instantiated as a Control object with an information set.
        self.dynamics = {
            'state1': lambda state1, shock1: state1 * 0.9 + shock1,  # Added state1 as parameter to create time dependency
            'control1': Control(['state1'])
        }
        # Define the reward function, which depends on state and control variables.
        self.reward = {
            'reward1': lambda state1, control1: state1 + control1
        }
        # An optional attribute for default limits.
        self.default_limit = None

def test_identify_variables():
    """
    Test whether _identify_variables can automatically classify the block's variables into shocks,
    states, controls, rewards, and parameters without relying on hard-coded expected values.
    """
    block = DummyDBlock()
    viz = DBlockVisualizer(block)
    viz._identify_variables()
    variables = viz.variables

    # Generate the expected sets of variables from the block attributes.
    expected_shocks = set(block.shocks.keys())
    
    expected_states = set()
    expected_controls = set()
    for key, rule in block.dynamics.items():
        if isinstance(rule, Control):
            expected_controls.add(key)
        else:
            expected_states.add(key)
    
    expected_rewards = set(block.reward.keys())
    
    # Verify that the variable extraction matches the expected sets.
    assert set(variables['shock_vars']) == expected_shocks, (
        f"Expected shock_vars: {expected_shocks}, got: {variables['shock_vars']}"
    )
    assert set(variables['state_vars']) == expected_states, (
        f"Expected state_vars: {expected_states}, got: {variables['state_vars']}"
    )
    assert set(variables['control_vars']) == expected_controls, (
        f"Expected control_vars: {expected_controls}, got: {variables['control_vars']}"
    )
    assert set(variables['reward_vars']) == expected_rewards, (
        f"Expected reward_vars: {expected_rewards}, got: {variables['reward_vars']}"
    )
    assert isinstance(variables['param_vars'], set), "param_vars should be a set type"

def test_extract_dependencies():
    """
    Test whether _extract_dependencies can automatically extract the dependencies
    among the variables in the block without relying on hard-coded expected values.
    """
    block = DummyDBlock()
    viz = DBlockVisualizer(block)
    viz._identify_variables()    # Extract variables first.
    viz._extract_dependencies()
    dependencies = viz.dependencies

    # Generate the expected dependencies based on block.dynamics
    expected_deps = {}

    for key, rule in block.dynamics.items():
        if isinstance(rule, Control):
            expected_deps[key] = list(rule.iset)
        else:
            sig = inspect.signature(rule)
            # Extract the parameters from the signature of the function.
            expected_deps[key] = list(sig.parameters.keys())

    for key, rule in block.reward.items():
        sig = inspect.signature(rule)
        expected_deps[key] = list(sig.parameters.keys())

    # Verify that the extracted dependencies match the expected ones (order-insensitive).
    for var, expected in expected_deps.items():
        actual = dependencies.get(var, [])
        assert set(actual) == set(expected), (
            f"For variable '{var}', expected dependencies: {expected}, got: {actual}"
        )

def test_identify_time_dependencies():
    """
    Test whether _identify_time_dependencies correctly identifies variables that depend on their previous values.
    """
    block = DummyDBlock()
    viz = DBlockVisualizer(block)
    viz._identify_variables()
    viz._extract_dependencies()
    viz._identify_time_dependencies()
    
    # Verify time dependency structures are created
    assert hasattr(viz, "prev_period_vars"), "prev_period_vars should be created"
    assert hasattr(viz, "prev_period_deps"), "prev_period_deps should be created"
    
    # Verify the variable with self-reference is detected as time-dependent
    # In our dummy block, state1 depends on its previous value
    assert "state1" in viz.prev_period_vars, "state1 should be identified as time-dependent"
    assert ("state1", "state1") in viz.prev_period_deps, "Dependency (state1, state1) should be identified"

def test_extract_formulas():
    """
    Test whether _extract_formulas can extract the formula information for all extractable variables
    (dynamics and rewards, plus calibration parameters). Note: shock variables are not included.
    """
    # Example calibration parameters (can be extended)
    calibration = {"param1": 0.05, "r": 0.03}
    block = DummyDBlock()
    viz = DBlockVisualizer(block, calibration=calibration)
    
    # Run the complete analysis flow.
    viz._identify_variables()
    viz._extract_dependencies()
    viz._identify_time_dependencies()
    viz._extract_formulas()
    formulas = viz.formulas

    # The expected formula variables should include dynamics, reward variables, and calibration parameters (excluding shocks)
    expected_formula_vars = set(block.dynamics.keys()) | set(block.reward.keys()) | set(calibration.keys())
    missing = expected_formula_vars - set(formulas.keys())
    assert not missing, f"The following variables should appear in formulas but are missing: {missing}"
    
    # Check Control variables have "Control" in their formula
    for var, rule in block.dynamics.items():
        if isinstance(rule, Control) and var in formulas:
            assert "Control" in formulas[var], f"Formula for Control variable {var} should include Control keyword"
    
    # Check calibration parameters have their values in formulas
    for param, value in calibration.items():
        if param in formulas:
            assert str(value) in formulas[param], f"Formula for parameter {param} should include its value {value}"
    
    # Check time dependencies are marked in formulas
    for var in viz.prev_period_vars:
        if var in formulas:
            assert var in formulas[var], f"Formula for time dependent variable {var} should include self-reference"

def test_agent_attribution():
    """
    Test whether get_agent_for_variable correctly maps variables to their associated agents.
    """
    block = DummyDBlock()
    agent_attribution = {
        "agent1": ["state1", "shock1"],
        "agent2": ["control1", "reward1"]
    }
    viz = DBlockVisualizer(block, agent_attribution=agent_attribution)
    
    # Test regular variables
    for agent, vars_list in agent_attribution.items():
        for var in vars_list:
            assert viz.get_agent_for_variable(var) == agent, f"Variable {var} should be attributed to {agent}"
    
    # Test time-dependent notation
    assert viz.get_agent_for_variable("state1*") == "agent1", "Time-dependent notation state1* should be attributed to agent1"
    
    # Test internal _prev notation
    assert viz.get_agent_for_variable("state1_prev") == "agent1", "Internal notation state1_prev should be attributed to agent1"
    
    # Test unattributed variable
    assert viz.get_agent_for_variable("unknown") == "other", "Unattributed variable should be mapped to 'other'"

def test_analyze():
    """
    Test whether the analyze method correctly executes the full analysis pipeline.
    """
    block = DummyDBlock()
    calibration = {"param1": 0.05, "r": 0.03}
    agent_attribution = {
        "agent1": ["state1", "shock1"],
        "agent2": ["control1", "reward1"]
    }
    
    viz = DBlockVisualizer(block, agent_attribution, calibration)
    
    # Run analyze
    result = viz.analyze()
    
    # Check method chaining
    assert result is viz, "analyze() should return self for method chaining"
    
    # Check that all data structures are populated
    assert viz.variables, "variables should be populated"
    assert viz.dependencies, "dependencies should be populated"
    assert viz.formulas, "formulas should be populated"
    assert hasattr(viz, "prev_period_vars"), "prev_period_vars should be created"
    
    # Check specific results to ensure full pipeline ran
    assert set(viz.variables['shock_vars']) == set(block.shocks.keys()), "Shock variables not correctly analyzed"
    assert "state1" in viz.prev_period_vars, "Time dependencies not correctly analyzed"
    assert "state1" in viz.dependencies, "Dependencies not correctly analyzed"
    assert "state1" in viz.formulas, "Formulas not correctly analyzed"

if __name__ == "__main__":
    test_identify_variables()
    test_extract_dependencies()
    test_identify_time_dependencies()
    test_extract_formulas()
    test_agent_attribution()
    test_analyze()
    print("All tests passed!")