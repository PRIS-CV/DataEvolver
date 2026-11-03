"""Acquisition and portable source-bundle contracts."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import ClassVar
from urllib.parse import urlsplit

from .contracts import Contract, canonical_json, file_sha256, identifier, positive_int

DEFAULT_HF_MIRROR = "https://hf-mirror.com"


def resolve_hf_endpoint(explicit_endpoint=None):
    endpoint = (explicit_endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_HF_MIRROR).strip().rstrip("/")
    parsed = urlsplit(endpoint)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("HF endpoint requires HTTPS without credentials, query or fragment")
    return endpoint


@dataclass(frozen=True)
class CollectionSpec(Contract):
    schema: ClassVar[str] = "dataevolver.collection_spec.v1"
    collection_id: str
    mode: str
    sources: tuple[dict, ...]
    hf_endpoint: str = ""
    concurrency: int = 4
    max_attempts: int = 3
    max_download_bytes: int = 20_000_000_000
    max_candidates: int = 5000
    batch_size: int = 128
    exclude_coco_ids: tuple[int, ...] = ()

    def __post_init__(self):
        identifier(self.collection_id, "collection_id")
        if self.mode not in {"fixture", "server"}:
            raise ValueError("collection mode must be fixture or server")
        for name in ("concurrency", "max_attempts", "max_download_bytes", "max_candidates", "batch_size"):
            positive_int(getattr(self, name), name)
        if self.concurrency > 4 or self.max_attempts > 3 or self.max_candidates > 5000 or self.batch_size > 128:
            raise ValueError("collection exceeds the first-pilot bounds")
        if not self.sources or len({s["source_id"] for s in self.sources}) != len(self.sources):
            raise ValueError("unique, explicit sources are required")
        for source in self.sources:
            required = {"source_id", "repo_id", "revision", "adapter", "quota", "tables", "image_files",
                        "dataset_license", "dataset_evidence", "origin_type", "origin_evidence", "allowed_hosts"}
            optional = {"field_mapping", "upstream_split", "split_evidence"}
            if not required.issubset(source) or set(source) - required - optional:
                raise ValueError("source fields differ from CollectionSpec.v1")
            identifier(source["source_id"], "source_id")
            positive_int(source["quota"], "quota")
            if source["adapter"] not in {"coco", "visual_genome"} or not source["tables"]:
                raise ValueError("source adapter/tables must be explicit")
            if self.mode == "server" and not re.fullmatch(r"[a-f0-9]{40}", source["revision"]):
                raise ValueError("server collection requires a pinned 40-character revision")
            if self.mode == "server" and "__SET_" in canonical_json(source):
                raise ValueError("server source configuration contains unresolved __SET_ placeholders")
            if not source["revision"]:
                raise ValueError("source revision is missing")
        if sum(s["quota"] for s in self.sources) > self.max_candidates:
            raise ValueError("source quotas exceed candidate budget")
        if any(type(value) is not int or value < 0 for value in self.exclude_coco_ids):
            raise ValueError("COCO exclusion IDs must be non-negative integers")
        resolve_hf_endpoint(self.hf_endpoint)
        canonical_json(self.sources)

    @classmethod
    def from_dict(cls, payload):
        values = cls.read_fields(payload)
        values["sources"] = tuple(values["sources"])
        values["exclude_coco_ids"] = tuple(values.get("exclude_coco_ids", ()))
        return cls(**values)


@dataclass(frozen=True)
class SourceBundle(Contract):
    schema: ClassVar[str] = "dataevolver.source_bundle.v1"
    bundle_id: str
    collection_hash: str
    mode: str
    sources: tuple[dict, ...]
    parquet_files: tuple[dict, ...]
    members: dict[str, str]
    count: int
    download_summary: dict


def member_path(root, relative):
    from pathlib import Path, PurePosixPath
    root = Path(root).resolve()
    path = PurePosixPath(relative)
    resolved = (root / relative).resolve()
    if (path.is_absolute() or ".." in path.parts or "\\" in relative or str(path) != relative
            or not resolved.is_relative_to(root)):
        raise ValueError("bundle member escapes its root")
    return resolved


def load_source_bundle(path):
    import json
    import pyarrow.parquet as pq
    bundle = SourceBundle.from_dict(json.loads(path.read_text()))
    if bundle.mode not in {"fixture", "server"} or bundle.count < 0 or bundle.count > 5000:
        raise ValueError("invalid source bundle mode/count")
    for relative, digest in bundle.members.items():
        member = member_path(path.parent, relative)
        if not member.is_file() or file_sha256(member) != digest:
            raise ValueError("source bundle member is missing or modified")
    actual_members = {value.relative_to(path.parent).as_posix() for value in path.parent.rglob("*") if value.is_file()}
    if actual_members != set(bundle.members) | {"bundle.json"}:
        raise ValueError("source bundle contains unindexed members")
    count = 0
    for shard in bundle.parquet_files:
        if bundle.members.get(shard["path"]) != shard["sha256"]:
            raise ValueError("source shard index is unbound")
        parquet = pq.ParquetFile(member_path(path.parent, shard["path"]))
        if parquet.metadata.num_rows != shard["count"]:
            raise ValueError("source shard count changed")
        count += shard["count"]
    if count != bundle.count:
        raise ValueError("source bundle count changed")
    for source in bundle.sources:
        for key in ("dataset_evidence", "origin_evidence"):
            if any(evidence not in bundle.members for evidence in source[key]):
                raise ValueError("source evidence is unbound")
    return bundle
