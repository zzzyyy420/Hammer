"""CPU-only BFCL scoring adapter; inverse-map function identifiers only."""
import ast
import json
import re
import sys
import types
from copy import deepcopy
from pathlib import Path


def frozen_json_parser(project):
    source = (Path(project) / "scripts/step_b_caller_runner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "clean_json")
    namespace = {"json": json, "re": re}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "frozen_clean_json", "exec"), namespace)
    return namespace["clean_json"]


def official_checker(bfcl):
    sys.path.insert(0, str(bfcl))
    # The canonical JSON adapter preserves dots. Only the SDK model registry is
    # replaced; the pinned AST scorer and all its validation code run unchanged.
    registry = types.ModuleType("bfcl_eval.constants.model_config")
    registry.MODEL_CONFIG_MAPPING = {"canonical-json": types.SimpleNamespace(underscore_to_dot=False)}
    sys.modules[registry.__name__] = registry
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

    def check(schema, calls, gold, category="multiple"):
        return ast_checker(deepcopy(schema), deepcopy(calls), deepcopy(gold),
                           Language.PYTHON, category, "canonical-json")
    return check


def restore_calls(calls, visible_to_view, view_to_original):
    if not isinstance(calls, list) or not calls:
        raise ValueError("invalid call list")
    restored = []
    for call in calls:
        if not isinstance(call, dict) or len(call) != 1:
            raise ValueError("invalid canonical call")
        name, arguments = next(iter(call.items()))
        if name not in visible_to_view or not isinstance(arguments, dict):
            raise ValueError("unknown function or invalid arguments")
        view_name = visible_to_view[name]
        if view_name not in view_to_original:
            raise ValueError("unmapped view function")
        restored.append({view_to_original[view_name]: deepcopy(arguments)})
    return restored


def score_raw(raw, parser, check, schema, gold, visible_to_view, view_to_original, category="multiple"):
    parsed, ok = parser(raw)
    if not ok or not isinstance(parsed, list) or not parsed:
        return {"valid": False, "error_type": "adapter:parse_failure"}
    calls = []
    for call in parsed:
        if not isinstance(call, dict):
            return {"valid": False, "error_type": "adapter:invalid_call"}
        name = call.get("name", call.get("tool"))
        arguments = call.get("arguments", call.get("parameters"))
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return {"valid": False, "error_type": "adapter:invalid_call"}
        calls.append({name: arguments})
    try:
        restored = restore_calls(calls, visible_to_view, view_to_original)
    except ValueError:
        return {"valid": False, "error_type": "adapter:unmapped_function"}
    return check(schema, restored, gold, category)
