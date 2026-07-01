import sys
import json
import hashlib
import inspect
from datetime import datetime, timezone
import tensorflow as tf

# ----------------------------------------------------------------------
# Argument summarization, type inference, Wrapper
# ----------------------------------------------------------------------
def _summarize_arg(value, max_len=100):
    try:
        if hasattr(value, 'shape') and hasattr(value, 'dtype'):
            return {"type": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
        if hasattr(value, 'shape') and hasattr(value, 'dtype') and not isinstance(value, (list, dict)):
            return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
        val_repr = repr(value)
        if len(val_repr) > max_len:
            h = hashlib.md5(val_repr.encode()).hexdigest()[:8]
            val_repr = val_repr[:max_len] + f"...[hash:{h}]"
        return {"type": type(value).__name__, "value": val_repr}
    except Exception:
        return {"type": "unavailable"}

def _summarize_args(args, kwargs):
    summary = {}
    for i, arg in enumerate(args):
        summary[f"arg_{i}"] = _summarize_arg(arg)
    for key, val in kwargs.items():
        summary[key] = _summarize_arg(val)
    return summary

def _infer_operation_type(func, name):
    try:
        mod = inspect.getmodule(func)
        mod_name = mod.__name__ if mod else ""
    except Exception:
        mod_name = ""
    full = f"{mod_name}.{name}".lower() if mod_name else name.lower()
    if any(k in full for k in ("keras", "layers", "nn")): return "neural_network"
    if any(k in full for k in ("data", "io", "dataset")): return "data_loading"
    if any(k in full for k in ("math", "random", "linalg")): return "math_operation"
    if any(k in full for k in ("train", "optimizer", "loss")): return "training"
    if any(k in full for k in ("image", "audio", "signal")): return "preprocessing"
    if any(k in full for k in ("save", "load", "serialize")): return "serialization"
    if any(k in full for k in ("distribute", "strategy")): return "distribution"
    return "general"

def Wrapper(func, name, log_stream=sys.stderr):
    def wrapper(*args, **kwargs):
        try:
            args_summary = _summarize_args(args, kwargs)
        except Exception:
            args_summary = {"error": "args_summary_failed"}
        result = func(*args, **kwargs)
        try:
            return_type = type(result).__name__ if result is not None else "NoneType"
        except Exception:
            return_type = "unknown"
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "func_name": name,
            "operation_type": _infer_operation_type(func, name),
            "args_summary": args_summary,
            "return_type": return_type,
        }
        try:
            log_stream.write(json.dumps(log_entry) + "\n")
            log_stream.flush()
        except Exception:
            pass
        return result
    return wrapper

# ----------------------------------------------------------------------
# Module proxy (dynamic interception)
# ----------------------------------------------------------------------
class ModuleProxy:
    def __init__(self, real_module, log_stream=sys.stderr, debug=False):
        object.__setattr__(self, '_real_module', real_module)
        object.__setattr__(self, '_log_stream', log_stream)
        object.__setattr__(self, '_proxy_cache', {})
        object.__setattr__(self, '_debug', debug)

    def __getattribute__(self, name):
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        real_mod = object.__getattribute__(self, '_real_module')
        log_stream = object.__getattribute__(self, '_log_stream')
        cache = object.__getattribute__(self, '_proxy_cache')
        debug = object.__getattribute__(self, '_debug')

        try:
            obj = getattr(real_mod, name)
        except AttributeError:
            raise AttributeError(f"module '{real_mod.__name__}' has no attribute '{name}'")

        if debug:
            print(f"[Proxy] accessing {real_mod.__name__}.{name}", file=sys.stderr)

        if name.startswith('_') or isinstance(obj, type):
            return obj

        # Sub-module or module-like object (lazy loader, etc.)
        is_like_module = (inspect.ismodule(obj) or 
                          (hasattr(obj, '__name__') and not callable(obj) and not inspect.isclass(obj)))
        if is_like_module:
            if name not in cache:
                cache[name] = ModuleProxy(obj, log_stream, debug)
            return cache[name]

        if callable(obj):
            if inspect.isclass(obj):
                cls = obj
                class_name = f"{real_mod.__name__}.{name}"
                def class_wrapper(*args, **kwargs):
                    try:
                        args_summary = _summarize_args(args, kwargs)
                    except Exception:
                        args_summary = {"error": "args_summary_failed"}
                    instance = cls(*args, **kwargs)
                    try:
                        return_type = type(instance).__name__
                    except Exception:
                        return_type = "unknown"
                    log_entry = {
                        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "func_name": class_name,
                        "operation_type": _infer_operation_type(cls, class_name),
                        "args_summary": args_summary,
                        "return_type": return_type,
                    }
                    try:
                        log_stream.write(json.dumps(log_entry) + "\n")
                        log_stream.flush()
                    except Exception:
                        pass
                    return instance
                return class_wrapper
            return Wrapper(obj, f"{real_mod.__name__}.{name}", log_stream)

        return obj

    def __repr__(self):
        return f"<ModuleProxy for {self._real_module.__name__}>"

    def __dir__(self):
        return dir(self._real_module)

    def __setattr__(self, name, value):
        raise AttributeError("Cannot set attribute on module proxy")

    @property
    def __name__(self):
        return self._real_module.__name__

    @property
    def __package__(self):
        return self._real_module.__package__

# ----------------------------------------------------------------------
# Static graph scan helper functions
# ----------------------------------------------------------------------
def _infer_op_type_from_name(op_name):
    name = op_name.lower()
    if any(k in name for k in ("keras", "layers", "nn")): return "neural_network"
    if any(k in name for k in ("data", "io", "dataset")): return "data_loading"
    if any(k in name for k in ("math", "random", "linalg")): return "math_operation"
    if any(k in name for k in ("train", "optimizer", "loss")): return "training"
    if any(k in name for k in ("image", "audio", "signal")): return "preprocessing"
    if any(k in name for k in ("save", "load", "serialize")): return "serialization"
    if any(k in name for k in ("distribute", "strategy")): return "distribution"
    return "general"

def _attr_to_python(value):
    """Convert AttrValue to a Python object (simplified)"""
    if hasattr(value, 'list'):
        if value.list.i:   return list(value.list.i)
        if value.list.f:   return list(value.list.f)
        if value.list.s:   return [s.decode('utf-8', errors='ignore') for s in value.list.s]
    if hasattr(value, 's'): return value.s.decode('utf-8', errors='ignore')
    if hasattr(value, 'i'): return value.i
    if hasattr(value, 'f'): return value.f
    if hasattr(value, 'b'): return value.b
    if hasattr(value, 'type'): return str(value.type)
    if hasattr(value, 'shape'): return str(value.shape).replace('\n', ' ')
    if hasattr(value, 'tensor'): return 'tensor'
    return str(value)

def _node_args_summary(node):
    args = {}
    args["inputs"] = list(node.input)
    for key, value in node.attr.items():
        args[key] = _attr_to_python(value)
    return args

def _node_return_type(node):
    if 'T' in node.attr: return str(node.attr['T'])
    if 'dtype' in node.attr: return str(node.attr['dtype'])
    if 'output_shapes' in node.attr: return str(node.attr['output_shapes'])
    return "tensor"

def graph_node_to_log_entry(node, graph_name="", func_prefix=""):
    """
    Unified log format for the dynamic proxy.
    Timestamp is left empty; func_name uses the node op name (with path info appended).
    """
    op_name = node.op
    display_name = f"{func_prefix}/{op_name}" if func_prefix else op_name
    return {
        "timestamp": "",   # Static graph has no reliable timestamp
        "func_name": display_name,
        "operation_type": _infer_op_type_from_name(op_name),
        "args_summary": _node_args_summary(node),
        "return_type": _node_return_type(node),
    }

def _scan_function_def(func_def, log_stream, graph_name="", prefix=""):
    new_prefix = f"{prefix}/{func_def.signature.name}" if prefix else func_def.signature.name
    for node in func_def.node_def:
        entry = graph_node_to_log_entry(node, graph_name, func_prefix=new_prefix)
        try:
            log_stream.write(json.dumps(entry) + "\n")
            log_stream.flush()
        except Exception:
            pass

def scan_concrete_function(concrete_func, log_stream=sys.stderr):
    graph = concrete_func.graph
    graph_def = graph.as_graph_def()
    graph_name = concrete_func.name

    # Main graph nodes
    for node in graph_def.node:
        entry = graph_node_to_log_entry(node, graph_name)
        try:
            log_stream.write(json.dumps(entry) + "\n")
            log_stream.flush()
        except Exception:
            pass

    # Nested functions in the function library
    for func_def in graph_def.library.function:
        _scan_function_def(func_def, log_stream, graph_name)

def scan_saved_model(loaded_model, log_stream=sys.stderr):
    """Scan the computation graphs of all signatures in a loaded SavedModel"""
    for concrete_func in loaded_model.signatures.values():
        scan_concrete_function(concrete_func, log_stream)

# ----------------------------------------------------------------------
# Main injection function (dynamic proxy + optional static graph scan)
# ----------------------------------------------------------------------
def inject_tensorflow_logging(log_stream=sys.stderr, debug=False, loaded_model=None):
    """
    Replace tensorflow in sys.modules with ModuleProxy (dynamic interception),
    and optionally perform a static graph scan on the loaded SavedModel, writing to the same log stream.

    Args:
        log_stream   : Log output stream (default stderr)
        debug        : Whether to print debug info (attribute access paths)
        loaded_model : Optional, a model loaded via tf.saved_model.load.
                       If provided, immediately performs a static graph scan and writes to log.
    Returns:
        ModuleProxy object, which should be re-assigned to the tf variable.
    """
    if 'tensorflow' not in sys.modules:
        import tensorflow
    real_tf = sys.modules['tensorflow']
    proxy = ModuleProxy(real_tf, log_stream, debug)
    sys.modules['tensorflow'] = proxy
    for mod_name, mod in sys.modules.items():
        if mod is real_tf:
            sys.modules[mod_name] = proxy

    # Static graph scan (if a model was provided)
    if loaded_model is not None:
        scan_saved_model(loaded_model, log_stream)

    return proxy

# Alias
ApplyPatchesRecursive = inject_tensorflow_logging

# ----------------------------------------------------------------------
# Command-line entry point (optional)
# ----------------------------------------------------------------------
def main():
    import tensorflow as tf
    tf = inject_tensorflow_logging(debug=False)
    print("Dynamic instrumentation active.", file=sys.stderr)

if __name__ == "__main__":
    main()