"""Batch Parquet records and small hash-bound JSON indexes."""
from __future__ import annotations

import io
import json

from .contracts import ArtifactRef, canonical_json


def encode_records(rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    values = [{"record": canonical_json({key: value for key, value in row.items() if key != "image_bytes"}),
               "image_bytes": row.get("image_bytes")} for row in rows]
    schema = pa.schema([("record", pa.large_string()), ("image_bytes", pa.binary())])
    stream = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(values, schema=schema), stream, compression="zstd")
    return stream.getvalue()


def read_parquet(path, *, batch_size=128):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        for value in batch.to_pylist():
            row = json.loads(value["record"])
            if value.get("image_bytes") is not None:
                row["image_bytes"] = value["image_bytes"]
            yield row


def read_records(ctx, index):
    for shard in index["shards"]:
        yield from read_parquet(ctx.store.resolve(ArtifactRef.from_dict(shard)))


def write_records(ctx, rows, metadata, *, batch_size=128):
    shards, batch, count = [], [], 0
    for row in rows:
        ctx.check()
        batch.append(row)
        count += 1
        if len(batch) == batch_size:
            shards.append(ctx.write_bytes(f"records/{len(shards):05d}.parquet", encode_records(batch),
                                          "application/vnd.apache.parquet").to_dict())
            batch = []
    if batch:
        shards.append(ctx.write_bytes(f"records/{len(shards):05d}.parquet", encode_records(batch),
                                      "application/vnd.apache.parquet").to_dict())
    return {"catalog": ctx.write_json("index.json", {**metadata, "count": count, "shards": shards})}
