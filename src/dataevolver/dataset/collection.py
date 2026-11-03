"""Metadata-first COCO/VG acquisition; all remote I/O is server-only."""
from __future__ import annotations

import io
import json
import hashlib
import threading
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from .collection_spec import CollectionSpec, SourceBundle, load_source_bundle, resolve_hf_endpoint
from .contracts import ArtifactRef, canonical_json, file_sha256, payload_sha256
from .image_records import encode_records
from dataevolver.harness.data_budget import DataBudgetExceeded
from dataevolver.harness.state_store import atomic_write_json


def normalize_coco_record(image, instances, captions, categories, licenses):
    image_id = image.get("id", image.get("image_id"))
    if image_id is None:
        raise ValueError("COCO image metadata is missing image_id")
    split = image.get("source_split", image.get("split", "unknown"))
    if "/train2017/" in str(image.get("coco_url", "")) or str(image.get("file_name", "")).startswith("train2017/"):
        split = "train2017"
    license_record = licenses.get(image.get("license"))
    return {
        "source_row_id": str(image_id), "coco_id": int(image_id), "source_split": split,
        "width": image.get("width"), "height": image.get("height"),
        "file_name": image.get("file_name"), "image_url": image.get("coco_url"),
        "image_shard": image.get("image_shard"),
        "license": license_record.get("name") if isinstance(license_record, dict) else license_record,
        "image_license_evidence": {"license_id": image.get("license"), "license_record": license_record},
        "captions": [value["caption"] for value in captions if value.get("caption")],
        "objects": [{"id": str(value["id"]), "category": categories.get(value.get("category_id"), str(value.get("category_id"))),
                     "box_xywh": value["bbox"], "attributes": [], "iscrowd": value.get("iscrowd", 0)}
                    for value in instances],
        "attributes": [], "relations": [], "raw_annotation": {"image": image, "instances": instances, "captions": captions},
        "related_keys": [f"coco:{image_id}"],
    }


def normalize_vg_record(image, objects, attributes, relationships, coco_eval_ids=()):
    coco_id = image.get("coco_id")
    if coco_id is not None and int(coco_id) in set(coco_eval_ids):
        return None
    image_id = image.get("image_id", image.get("id"))
    if image_id is None:
        raise ValueError("VG image metadata is missing image_id")
    by_object = defaultdict(list)
    for value in attributes:
        by_object[str(value["object_id"])].extend(value.get("attributes", []))
    normalized = []
    for value in objects:
        object_id = str(value["object_id"])
        names = value.get("names") or [value.get("name", "")]
        normalized.append({"id": object_id, "category": names[0], "names": names,
                           "box_xywh": [value["x"], value["y"], value["w"], value["h"]],
                           "attributes": by_object[object_id]})
    relations = []
    for value in relationships:
        subject = value.get("subject_id", (value.get("subject") or {}).get("object_id"))
        target = value.get("object_id", (value.get("object") or {}).get("object_id"))
        relations.append({"subject_id": str(subject), "object_id": str(target),
                          "predicate": value.get("predicate", ""), "relationship_id": value.get("relationship_id")})
    return {
        "source_row_id": str(image_id), "coco_id": int(coco_id) if coco_id is not None else None,
        "source_split": image.get("source_split", "unspecified"), "width": image.get("width"), "height": image.get("height"),
        "image_url": image.get("url"), "image_shard": image.get("image_shard"), "file_name": image.get("file_name"),
        "license": image.get("license"), "image_license_evidence": image.get("license_evidence", {}),
        "captions": image.get("captions", []), "objects": normalized, "attributes": attributes, "relations": relations,
        "raw_annotation": {"image": image, "objects": objects, "attributes": attributes, "relationships": relationships},
        "related_keys": [f"vg:{image_id}"] + ([f"coco:{coco_id}"] if coco_id is not None else []),
    }


class DownloadLedger:
    def __init__(self, ctx, spec):
        self.ctx, self.spec = ctx, spec
        self.path = Path(ctx.parent.work_dir) / "downloads.json"
        self.cache = ctx.graph.dataset_root / "download_cache" / spec.compute_hash()
        self.lock = threading.RLock()
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {
            "binding": spec.compute_hash(), "transferred_bytes": 0, "items": {}, "pins": {}}
        if self.data["binding"] != spec.compute_hash():
            raise ValueError("download journal binding changed")

    def save(self):
        atomic_write_json(self.path, self.data)

    def progress(self, key, count):
        with self.lock:
            self.ctx.check()
            total = self.data["transferred_bytes"] + count
            if total > self.spec.max_download_bytes:
                raise DataBudgetExceeded("download byte budget exhausted")
            self.ctx.budget.reserve_file(f"download:{key}:{total}", count)
            self.data["transferred_bytes"] = total
            self.data["items"][key]["transferred_bytes"] += count
            self.save()

    def download(self, source, entry):
        if self.spec.mode == "fixture":
            return self.ctx.store.resolve(ArtifactRef.from_dict(entry["artifact"]))
        key = payload_sha256([source["repo_id"], source["revision"], entry])
        with self.lock:
            item = self.data["items"].setdefault(key, {"source_id": source["source_id"], "file": entry,
                "attempts": 0, "status": "PENDING", "transferred_bytes": 0, "cache_hits": 0})
            if item["status"] == "SUCCESS":
                path = Path(item["cache_path"])
                if not path.is_file() or file_sha256(path) != item["sha256"]:
                    raise ValueError("download cache is missing or changed")
                item["cache_hits"] += 1
                self.save()
                return path
        self.cache.mkdir(parents=True, exist_ok=True)
        while item["attempts"] < self.spec.max_attempts:
            with self.lock:
                item["attempts"] += 1
                self.save()
            try:
                if "url" in entry:
                    path = self._external(source, entry["url"], key, item)
                else:
                    path = self._hub(source, entry["path"], key)
                if entry.get("sha256") and file_sha256(path) != entry["sha256"]:
                    raise ValueError("download differs from its approved hash")
                with self.lock:
                    item.update(status="SUCCESS", cache_path=str(path), sha256=file_sha256(path),
                                file_bytes=path.stat().st_size, error="")
                    self.save()
                return path
            except DataBudgetExceeded as exc:
                with self.lock:
                    item.update(status="FAILED", error=str(exc))
                    self.save()
                raise
            except Exception as exc:
                with self.lock:
                    item.update(status="FAILED", error=f"{type(exc).__name__}: {str(exc)[:300]}")
                    self.save()
        raise ValueError(f"download attempts exhausted: {entry}")

    def _hub(self, source, filename, key):
        from huggingface_hub import hf_hub_download
        from tqdm.auto import tqdm
        ledger = self

        class TransferProgress(tqdm):
            def update_transfer(self, count):
                ledger.progress(key, count)

        return Path(hf_hub_download(repo_id=source["repo_id"], filename=filename, revision=source["revision"],
            repo_type="dataset", endpoint=resolve_hf_endpoint(self.spec.hf_endpoint), cache_dir=self.cache / "hf",
            tqdm_class=TransferProgress, token=None))

    def _external(self, source, url, key, item):
        allowed_hosts = source["allowed_hosts"]

        def check_url(value):
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or parsed.hostname not in allowed_hosts or parsed.username or parsed.password:
                raise ValueError("external download URL is outside the source allowlist")

        class SourceRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, request, response, code, message, headers, new_url):
                check_url(new_url)
                return super().redirect_request(request, response, code, message, headers, new_url)

        check_url(url)
        path, part = self.cache / key, self.cache / f"{key}.part"
        offset = part.stat().st_size if part.exists() and item.get("etag") else 0
        headers = {"Range": f"bytes={offset}-", "If-Range": item["etag"]} if offset else {}
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.build_opener(SourceRedirect()).open(request, timeout=60) as response:
            append = offset and response.status == 206
            if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("external partial response has a mismatched range")
            item["etag"] = response.headers.get("ETag")
            with self.lock:
                self.save()
            written = 0
            with part.open("ab" if append else "wb") as handle:
                while chunk := response.read(1024 * 1024):
                    self.progress(key, len(chunk))
                    handle.write(chunk)
                    written += len(chunk)
            expected = response.headers.get("Content-Length")
            if expected is not None and written != int(expected):
                raise OSError("external response ended before its advertised length")
        part.replace(path)
        return path

    def pin(self, source):
        if self.spec.mode == "fixture":
            return
        from huggingface_hub import HfApi
        identity = source["source_id"]
        if identity in self.data["pins"]:
            if self.data["pins"][identity]["revision"] != source["revision"]:
                raise ValueError("pinned source revision changed")
            return
        info = HfApi(endpoint=resolve_hf_endpoint(self.spec.hf_endpoint)).repo_info(
            source["repo_id"], repo_type="dataset", revision=source["revision"], files_metadata=True)
        if info.sha != source["revision"]:
            raise ValueError("endpoint did not return the exact pinned revision")
        self.data["pins"][identity] = {"revision": info.sha,
            "files": [{"path": value.rfilename, "size": value.size} for value in info.siblings]}
        self.save()


def table_rows(path, entry):
    import pyarrow.parquet as pq
    name = entry.get("member") or entry.get("path") or entry.get("url", "") or path.name
    if zipfile.is_zipfile(path):
        if not entry.get("member"):
            raise ValueError("ZIP metadata requires an explicit member")
        with zipfile.ZipFile(path) as archive:
            payload = json.loads(archive.read(entry["member"]))
    elif str(name).endswith(".parquet"):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            yield from batch.to_pylist()
        return
    else:
        payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        if "key" not in entry:
            raise ValueError("JSON metadata object requires an explicit table key")
        payload = payload[entry["key"]]
    if not isinstance(payload, list):
        raise ValueError("source table must contain records")
    yield from payload


def source_candidates(source, ledger, spec):
    tables = {}
    for role, entries in source["tables"].items():
        rows = []
        for entry in entries:
            for row in table_rows(ledger.download(source, entry), entry):
                mapping = source.get("field_mapping", {}).get(role, {})
                rows.append({mapping.get(key, key): value for key, value in row.items()})
        tables[role] = rows
    if "image_info" not in tables:
        raise ValueError("source requires an image_info table and an explicit field mapping")
    grouped = {}
    for role in ("instances", "captions", "objects", "attributes", "relationships"):
        values = defaultdict(list)
        for value in tables.get(role, []):
            image_id = str(value.get("image_id"))
            values[image_id].extend(value.get(role, [value]))
        grouped[role] = values
    categories = {value["id"]: value["name"] for value in tables.get("categories", [])}
    licenses = {value["id"]: value for value in tables.get("licenses", [])}
    for offset, image in enumerate(tables["image_info"]):
        identity = str(image.get("image_id", image.get("id"))) if isinstance(image, dict) else f"row-{offset}"
        try:
            if source["adapter"] == "coco":
                row = normalize_coco_record(image, grouped["instances"][identity], grouped["captions"][identity], categories, licenses)
                if row["source_split"] == "unknown" and source.get("upstream_split") == "train2017" and source.get("split_evidence"):
                    row["source_split"] = "train2017"
                if row["source_split"] != "train2017":
                    yield None, identity, "unverified_or_nontraining_upstream_split"
                    continue
            else:
                row = normalize_vg_record(image, grouped["objects"][identity], grouped["attributes"][identity],
                                          grouped["relationships"][identity], spec.exclude_coco_ids)
            if row is None or row.get("coco_id") in spec.exclude_coco_ids:
                yield None, identity, "excluded_coco_evaluation_id"
            elif row.get("source_split") in {"val", "test", "val2017", "test2017", "evaluation"}:
                yield None, identity, "excluded_upstream_evaluation_split"
            elif row["width"] is not None and row["height"] is not None and min(row["width"], row["height"]) < 256:
                yield None, identity, "metadata_image_too_small"
            elif ("instances" if source["adapter"] == "coco" else "objects") in tables and len(row["objects"]) < 2:
                yield None, identity, "metadata_subject_count_below_two"
            else:
                yield row, identity, ""
        except (ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
            yield None, identity, f"invalid_image_metadata: {type(exc).__name__}: {str(exc)[:200]}"


def collect(ctx):
    declared_spec = CollectionSpec.from_dict(ctx.read("spec"))
    spec = ctx.graph.collection_spec
    if declared_spec.collection_id != spec.collection_id:
        raise ValueError("collection source spec changed")
    if spec.mode != ctx.graph.intent.parameters["execution_mode"]:
        raise ValueError("collection mode differs from DatasetIntent")
    ledger = DownloadLedger(ctx, spec)
    for source in spec.sources:
        ledger.pin(source)
    sources, shards, members, item_log, batch = [], [], {}, [], []
    base = f"bundles/{spec.collection_id}/{spec.compute_hash()[:16]}"

    def write(relative, payload, media):
        ref = ctx.write_bytes(f"{base}/{relative}", payload, media)
        members[relative] = ref.sha256
        return ref

    def flush():
        if batch:
            relative = f"records/{len(shards):05d}.parquet"
            ref = write(relative, encode_records(batch), "application/vnd.apache.parquet")
            shards.append({"path": relative, "sha256": ref.sha256, "count": len(batch)})
            batch.clear()

    for source in spec.sources:
        evidence = {}
        for key in ("dataset_evidence", "origin_evidence", "split_evidence"):
            evidence[key] = []
            for number, value in enumerate(source.get(key, [])):
                ref = ArtifactRef.from_dict(value)
                relative = f"evidence/{source['source_id']}-{key}-{number}.bin"
                write(relative, ctx.store.resolve(ref).read_bytes(), ref.media_type)
                evidence[key].append(relative)
        selected, source_error = [], ""
        try:
            for row, identity, reason in source_candidates(source, ledger, spec):
                if row is None:
                    item_log.append({"source_id": source["source_id"], "item_id": identity, "status": "SKIPPED", "reason": reason})
                else:
                    ctx.budget.reserve_rows(f"{source['source_id']}:{identity}", 1)
                    selected.append(row)
                    if len(selected) == source["quota"]:
                        break
        except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
            selected = []
            source_error = f"source_metadata_failed: {type(exc).__name__}: {str(exc)[:200]}"
            item_log.append({"source_id": source["source_id"], "item_id": "source_metadata",
                             "status": "FAILED", "reason": source_error})
        wanted = {value["source_row_id"] for value in selected}
        image_paths = {}
        needed = {value["image_shard"] for value in selected if value.get("image_shard")}
        image_files = [entry for entry in source["image_files"] if selected and (not needed or entry.get("path") in needed)]
        for entry in image_files:
            path = ledger.download(source, entry)
            import pyarrow.parquet as pq
            for chunk in pq.ParquetFile(path).iter_batches(batch_size=128):
                for value in chunk.to_pylist():
                    identity = str(value.get("image_id", value.get("id")))
                    if identity in wanted:
                        embedded = value.get("image", {})
                        raw = embedded.get("bytes") if isinstance(embedded, dict) else embedded
                        if raw and len(raw) <= 32 * 1024 * 1024:
                            digest = hashlib.sha256(raw).hexdigest()
                            image_path = ledger.cache / "embedded_images" / digest
                            ctx.budget.reserve_file(f"embedded_image:{digest}", len(raw))
                            if not image_path.exists():
                                image_path.parent.mkdir(parents=True, exist_ok=True)
                                image_path.write_bytes(raw)
                            elif file_sha256(image_path) != digest:
                                raise ValueError("embedded-image cache changed")
                            image_paths[identity] = image_path

        def fetch_image(row):
            identity = row["source_row_id"]
            if identity in image_paths:
                return row, image_paths[identity].read_bytes(), ""
            if not row.get("image_url"):
                return row, None, "missing_image_bytes_or_url"
            try:
                path = ledger.download(source, {"url": row["image_url"]})
                if path.stat().st_size > 32 * 1024 * 1024:
                    return row, None, "encoded_image_too_large"
                return row, path.read_bytes(), ""
            except DataBudgetExceeded:
                raise
            except Exception as exc:
                return row, None, str(exc)

        successful = 0
        with ThreadPoolExecutor(max_workers=spec.concurrency) as pool:
            for offset in range(0, len(selected), spec.batch_size):
                results = list(pool.map(fetch_image, selected[offset:offset+spec.batch_size]))
                for row, raw, reason in results:
                    identity = row["source_row_id"]
                    item_log.append({"source_id": source["source_id"], "item_id": identity,
                                     "status": "SUCCESS" if raw else "FAILED", "reason": reason})
                    if not raw:
                        continue
                    successful += 1
                    row.update(id="img-" + payload_sha256([source["repo_id"], source["revision"], identity])[:24],
                        source_id=source["source_id"], source_revision=source["revision"], image_bytes=raw,
                        original_sha256=hashlib.sha256(raw).hexdigest(),
                        origin_type=source["origin_type"], caption_origin="upstream_annotation",
                        evidence_level="mock_verified" if spec.mode == "fixture" else "real_execution_verified")
                    batch.append(row)
                flush()
        sources.append({"source_id": source["source_id"], "repo_id": source["repo_id"], "revision": source["revision"],
            "quota": source["quota"], "selected_count": len(selected), "collected_count": successful,
            "collection_error": source_error,
            "shortfall": source["quota"] - successful, "dataset_license": source["dataset_license"],
            "origin_type": source["origin_type"], **evidence})
    flush()
    write("collection_spec.json", canonical_json(spec.to_dict()).encode(), "application/json")
    public_ledger = {key: {field: value for field, value in item.items() if field != "cache_path"}
                     for key, item in ledger.data["items"].items()}
    summary = {"transferred_bytes": ledger.data["transferred_bytes"],
               "materialized_file_bytes": sum(value.get("file_bytes", 0) for value in public_ledger.values()),
               "success_count": sum(value["status"] == "SUCCESS" for value in item_log),
               "failure_count": sum(value["status"] == "FAILED" for value in item_log),
               "skipped_count": sum(value["status"] == "SKIPPED" for value in item_log)}
    write("downloads.json", canonical_json({"summary": summary, "files": public_ledger, "items": item_log,
          "endpoint": resolve_hf_endpoint(spec.hf_endpoint), "pins": ledger.data["pins"]}).encode(), "application/json")
    bundle = SourceBundle(spec.collection_id, spec.compute_hash(), spec.mode, tuple(sources), tuple(shards), members,
                          sum(value["count"] for value in shards), summary)
    ref = ctx.write_json(f"{base}/bundle.json", bundle.to_dict())
    path = ctx.store.resolve(ref)
    load_source_bundle(path)
    for relative in [*members, "bundle.json"]:
        (path.parent / relative).chmod(0o444)
    return {"source_bundle": ref, "report": ctx.write_json("collection_report.json", {"sources": sources, **summary})}


STEPS = {"images.collect": ("data", {"spec"}, set(), {"source_bundle", "report"}, collect)}


def validate_step(node):
    if (node.operator != "images.collect" or node.operator_version != "1" or node.kind != "data"
            or set(node.inputs) != {"spec"} or node.parameters):
        raise ValueError("collection node differs from its registered contract")
