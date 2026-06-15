"""
Test the parsed doc of tools

Run from the project root as a module:

    python -m lang_graph.test_2
"""

from lang_graph.tools import read_file_in_sandbox

print(read_file_in_sandbox.args_schema.model_json_schema())
