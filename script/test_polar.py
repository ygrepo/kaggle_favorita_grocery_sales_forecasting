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

import rmm

rmm.reinitialize(
    pool_allocator=True, initial_pool_size="36GB"
)  # optional but helps perf

import polars as pl, numpy as np

N = 300_000
df = pl.DataFrame(
    {
        "store": np.repeat([1, 2, 3], N // 3),
        "item": np.tile(np.arange(1000), N // 1000),
        "date": np.tile(np.arange(300), N // 300),
        "unit_sales": np.random.rand(N).astype("float32"),
    }
)

# CPU
_ = (
    df.lazy()
    .group_by(["store", "item", "date"])
    .agg(pl.col("unit_sales").sum())
    .collect()
)

# GPU
out = (
    df.lazy()
    .group_by(["store", "item", "date"])
    .agg(pl.col("unit_sales").sum())
    .collect(engine="gpu")
)
print("GPU out shape:", out.shape)
