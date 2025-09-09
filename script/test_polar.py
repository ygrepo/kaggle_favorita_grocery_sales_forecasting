import sys, os, importlib.util, traceback

print("PY:", sys.executable)
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("sys.path[0]:", sys.path[0])
print("find_spec(cudf_polars):", importlib.util.find_spec("cudf_polars"))

try:
    import cudf_polars, pylibcudf, rmm

    print("cudf_polars file:", getattr(cudf_polars, "__file__", None))
    print("pylibcudf imported OK")
except Exception as e:
    print("DIRECT IMPORT FAILED:")
    traceback.print_exc()
    raise
