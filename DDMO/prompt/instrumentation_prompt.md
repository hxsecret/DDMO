# DDMO Instrumentation Code Generation Prompt

> Prompt used to generate the non-invasive, framework-adaptive Python instrumentation script (dynamic module proxy + wrapper + static compute-graph scan). 


```
You are a security engineer building a non-invasive, framework-adaptive
instrumentation layer for detecting malicious behavior during machine-learning
model loading and inference.

TARGET LIBRARY: {library_name}   (e.g., "tensorflow")

OBJECTIVE
Generate a Python instrumentation script for `{library_name}` that intercepts
its API calls and, where possible, its static compute graph, and emits a
standardized semantic log stream. This log will later be correlated with
kernel-level system calls (via process id and timestamp) to explain why the
model issues certain low-level operations.

INVARIANT LOGGING SKELETON (follow exactly; do not change the schema)
1. Standardized JSON events.  Serialize every intercepted event as a single
   JSON line with exactly these fields:
     timestamp, func_name, operation_type, args_summary, return_type, call_stack
   For static graph nodes the timestamp may be empty, but the schema must remain
   identical so a single downstream parser works everywhere.
2. Defensive argument summarization (ArgsSummary).  Capture lightweight,
   non-sensitive information only:
   - tensors/arrays -> {"type", "shape", "dtype"}; NEVER dump raw tensor contents;
   - long values -> truncate to <= 100 chars and append a short md5 hash;
   - strings/bytes -> bounded-length summary; flags/options -> repr;
   - wrap every summarization in try/except with an "unavailable" fallback;
     summarization must never raise or crash the target process.
3. Complete call stack.  Retrieve the full call stack via the `traceback`
   module and include a compact list of frames (file, line, function) in the
   "call_stack" field.
4. Conservative wrapping policy.  Wrap callable functions ONLY.  NEVER wrap or
   replace class definitions, built-in types, or objects backed by C/C++
   bindings (return them untouched).  Keep wrappers THIN: summarize and log,
   then invoke the original with unchanged arguments and return its result
   unchanged.  Do not alter return values or side effects, and preserve
   exception semantics (a raised exception must still propagate; recording an
   error is allowed, swallowing it is not).
5. Dynamic discovery.  Do NOT hardcode a fixed list of API names.  Use a lazy
   module-proxy pattern (override __getattribute__, replace the module in
   sys.modules) to cover all public callables at runtime.  Skip names starting
   with "_".  Cache proxied submodules.  Pass through protocol attributes such
   as __name__, __package__, __dir__, __repr__.
6. Graceful degradation.  Wrap both the injection phase and the runtime
   logging phase in try/except so any failure degrades silently without
   crashing the target process.
7. Output contract.  Return a single JSON object with a single key "code"
   whose value is the raw Python source string.  Do NOT wrap it in markdown
   code fences.

FRAMEWORK-SPECIFIC KNOWLEDGE
A. Operation-type taxonomy.  Classify each intercepted call/graph node into a
   coarse semantic category using module/function-name keywords appropriate to
   {library_name} (e.g., neural_network, data_loading, math_operation,
   training, preprocessing, serialization, distribution, general).
B. Static compute-graph extraction harness.  Expose a function that extracts
   the framework's static graph from a loaded model and writes each node to the
   same JSON stream with the same schema.  For tensorflow use
   graph.as_graph_def() over loaded_model.signatures and convert AttrValue
   fields to plain Python; if {library_name} uses a different introspection API,
   use the correct equivalent (e.g., torch.fx / jax.make_jaxpr) and adapt the
   node/attribute fields accordingly.

DELIVERABLE
A complete, importable module containing at minimum: an argument summarizer,
an operation-type classifier, a wrapper factory, the module proxy, the
static-graph scanner, and an injection entry point
inject_{library_name}_logging(log_stream, debug=False, loaded_model=None) that
substitutes the library in sys.modules and optionally performs the static scan.
```
