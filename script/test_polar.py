import polars as pl
import numpy as np

df = pl.DataFrame(
    {
        "store": np.repeat([1, 2, 3], 1_000_00),
        "item": np.tile(np.arange(1000), 300),
        "date": np.tile(np.arange(300), 1000),
        "unit_sales": np.random.rand(300_000).astype(np.float32),
    }
)

# CPU
df.lazy().group_by(["store", "item", "date"]).agg(pl.col("unit_sales").sum()).collect()

# GPU
out = (
    df.lazy()
    .group_by(["store", "item", "date"])
    .agg(pl.col("unit_sales").sum())
    .collect(engine="gpu")
)
print(out.shape)
