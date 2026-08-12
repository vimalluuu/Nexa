"""
nexa/tools/__init__.py
=======================
Stub — Tool Use System.

This sub-package will contain:
    - base_tool.py  : Abstract base class for all tools
    - calculator.py : Safe arithmetic expression evaluator
    - web_search.py : HTTP-based web search (returns text, feeds to model)
    - code_exec.py  : Sandboxed Python code execution

Independence: Tools are standard software (HTTP calls, regex, Python eval).
No AI APIs are used; the model decides which tool to invoke via its own
generated output.
"""
# Populated in a future phase
