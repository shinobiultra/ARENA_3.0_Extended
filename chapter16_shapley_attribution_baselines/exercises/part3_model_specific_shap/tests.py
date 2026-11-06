"""Complete historical immediate tests under explicit curriculum namespaces.

Each callback test retains its original independent oracles and fallback helpers.
No test is run at import. Scientific fallbacks remain bound to original packages;
passing these tests alone does not establish isolation of every inline helper.
"""
from . import tree_treeshap_methods_tests as tree
from . import deep_deepshap_methods_tests as deep
from . import gradient_gradientshap_methods_tests as gradient
