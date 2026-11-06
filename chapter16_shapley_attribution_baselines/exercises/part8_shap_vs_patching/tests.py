"""Original legacy semantic suites retain original fixture path identity.

Runtime aliases import the original modules. Local relu_tests.py and
agreement_tests.py are byte-exact custody copies only. Historical fixture checks
cover their original reports/notebooks, not the canonical section's acceptance.
"""
from chapter16_shapley_attribution_baselines.exercises.part6_shap_vs_activation_patching import tests as relu_tests
from chapter16_shapley_attribution_baselines.exercises.part8_shapley_mechinterp_agreement import tests as agreement_tests

__all__ = ['relu_tests', 'agreement_tests']
