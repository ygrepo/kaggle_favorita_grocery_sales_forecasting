import polars as pl, numpy as np, rmm, cudf_polars, pylibcudf

print("polars:", pl.__version__)
print("cudf_polars:", cudf_polars.__version__)
print("pylibcudf:", pylibcudf.__version__)

# optional: pre-allocate a GPU memory pool (helps perf)
rmm.reinitialize(pool_allocator=True, initial_pool_size="36GB")

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

# GPU (watch nvidia-smi for VRAM usage)
out = (
    df.lazy()
    .group_by(["store", "item", "date"])
    .agg(pl.col("unit_sales").sum())
    .collect(engine="gpu")
)
print(out.shape)
