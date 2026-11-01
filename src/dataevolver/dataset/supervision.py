"""Subject-list and action-conditioned video supervision, independent of a model.

Media operations return ArtifactRefs. Review and target-side annotations are
kept outside the training input whitelist.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .contracts import ArtifactRef, ArtifactStore, canonical_json, file_sha256, identifier, payload_sha256


def refs(value):
    if isinstance(value, dict):
        if value.get("schema_version") == ArtifactRef.schema:
            yield ArtifactRef.from_dict(value)
        else:
            for child in value.values():
                yield from refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from refs(child)


def artifact_closure(value, store):
    """Keep references inside immutable JSON evidence portable as well."""
    pending = list(refs(value))
    seen = set()
    while pending:
        ref = pending.pop()
        key = (ref.store, ref.path, ref.sha256)
        if key in seen:
            continue
        path = store.resolve(ref)
        seen.add(key)
        yield ref
        if ref.media_type == 'application/json':
            pending.extend(refs(json.loads(path.read_text())))


def write_artifact(store, store_name, relative, data, media_type):
    import hashlib
    ref = ArtifactRef(store_name, relative, hashlib.sha256(data).hexdigest(), media_type)
    root = store.roots[store_name]
    path = (root / ref.path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("artifact output escapes dataset store")
    if path.exists():
        store.resolve(ref)
        return ref
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)
    store.resolve(ref)
    return ref


def write_image(store, name, relative, image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return write_artifact(store, name, relative, buffer.getvalue(), "image/png")


def extract_subject_references(store, name, family_id, target, instances):
    from PIL import Image
    import numpy as np
    identifier(family_id, "family_id")
    image = Image.open(store.resolve(target)).convert("RGB")
    subjects, transforms = [], []
    identities = set()
    for instance in instances:
        identity = instance["subject_id"]; identifier(identity, "subject_id")
        if identity in identities:
            raise ValueError("duplicate subject identity")
        identities.add(identity)
        mask_ref = ArtifactRef.from_dict(instance["mask"])
        mask = np.asarray(Image.open(store.resolve(mask_ref)).convert("L")) > 127
        if mask.shape != (image.height, image.width) or not mask.any():
            raise ValueError("subject mask is empty or in a different coordinate frame")
        y, x = np.where(mask)
        box = [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)]
        pixels = np.asarray(image).copy()
        pixels[~mask] = 0
        rgba = Image.fromarray(np.dstack((pixels,mask.astype("uint8")*255)),"RGBA")
        ref = write_image(store, name, f"{family_id}/references/{identity}.png", rgba.crop(box))
        subjects.append({"id": identity, "reference": ref.to_dict()})
        transforms.append({"subject_id": identity, "target_instance_id": instance["target_instance_id"],
                           "target_bbox_xyxy": box, "reference_size": list(rgba.crop(box).size),
                           "reference_to_target": {"translation_xy": box[:2], "scale_xy": [1.0, 1.0]}})
    if not subjects:
        raise ValueError("no subject reference extracted")
    return subjects, transforms


def dilate_binary_region(region,pixels):
    """Square binary dilation by integral sums, without a quadratic rank filter."""
    import numpy as np
    if type(pixels) is not int or not 0<=pixels<=128:raise ValueError('invalid region expansion')
    if pixels==0:return region.astype(bool).copy()
    padded=np.pad(region.astype(bool),pixels)
    integral=np.pad(padded,((1,0),(1,0))).cumsum(0,dtype=np.int64).cumsum(1,dtype=np.int64)
    width=2*pixels+1
    counts=integral[width:,width:]-integral[:-width,width:]-integral[width:,:-width]+integral[:-width,:-width]
    return counts>0


def rectangle_region(shape,boxes):
    import numpy as np
    import math
    height,width=shape;mask=np.zeros(shape,dtype=bool)
    for box in boxes:
        if (len(box)!=4 or any(type(x) not in {int,float} or not math.isfinite(x) or not 0<=x<=1 for x in box)
                or box[0]>=box[2] or box[1]>=box[3]):
            raise ValueError('observed support region must declare normalized, ordered coordinates')
        left,top=math.floor(box[0]*width),math.floor(box[1]*height)
        right,bottom=math.ceil(box[2]*width),math.ceil(box[3]*height)
        mask[top:bottom,left:right]=True
    return mask


def completion_region(store, name, family_id, target, selected, preserved, *, dilation_pixels,extra_boxes=()):
    from PIL import Image
    import numpy as np
    image = Image.open(store.resolve(target)).convert("RGB")
    region = np.zeros((image.height, image.width), dtype=bool)
    for instance in selected:
        mask = np.asarray(Image.open(store.resolve(ArtifactRef.from_dict(instance["mask"]))).convert("L")) > 127
        if mask.shape != region.shape:
            raise ValueError("selected mask coordinates differ from target")
        region |= mask
    region |= rectangle_region(region.shape,extra_boxes)
    if not region.any() or type(dilation_pixels) is not int or not 0 <= dilation_pixels <= 128:
        raise ValueError("empty selection or invalid background region expansion")
    region = dilate_binary_region(region,dilation_pixels)
    for instance in preserved:
        mask = np.asarray(Image.open(store.resolve(ArtifactRef.from_dict(instance["mask"]))).convert("L")) > 127
        if mask.shape != region.shape:
            raise ValueError("preserved mask coordinates differ from target")
        region &= ~mask
    visible = image.convert("RGBA")
    visible.putalpha(Image.fromarray((~region).astype("uint8")*255))
    return (write_image(store, name, f"{family_id}/background_visible.png", visible),
            write_image(store, name, f"{family_id}/completion_region.png", Image.fromarray(region.astype("uint8")*255)))


def complete_background_composite(store, name, family_id, target, generated_edit, region):
    from PIL import Image
    import numpy as np
    original = Image.open(store.resolve(target)).convert("RGB")
    proposal = Image.open(store.resolve(generated_edit)).convert("RGB")
    mask = Image.open(store.resolve(region)).convert("L")
    if original.size != proposal.size or mask.size != original.size:
        raise ValueError("background completion dimensions changed")
    result = Image.composite(proposal, original, mask)
    outside = np.asarray(mask) == 0
    if not np.array_equal(np.asarray(result)[outside], np.asarray(original)[outside]):
        raise ValueError("background completion changed protected pixels")
    return write_image(store, name, f"{family_id}/background_completed.png", result)


def rectangular_condition_region(store,name,family_id,region,preserved=()):
    """Hide object-shaped cues in the edit input, preserving named instances.

    This is a model-conditioning region only. The original completion region
    continues to control which generated pixels may enter the training input.
    """
    import numpy as np
    from PIL import Image
    with Image.open(store.resolve(region)) as image:mask=np.asarray(image.convert('L'))>127
    ys,xs=np.nonzero(mask)
    if not len(xs):raise ValueError('empty conditioning region')
    rectangular=np.zeros_like(mask)
    rectangular[ys.min():ys.max()+1,xs.min():xs.max()+1]=True
    for instance in preserved:
        with Image.open(store.resolve(ArtifactRef.from_dict(instance['mask']))) as image:
            kept=np.asarray(image.convert('L'))>127
        if kept.shape!=mask.shape:raise ValueError('conditioning preservation coordinates changed')
        rectangular &= ~kept
    if np.any(mask & ~rectangular):raise ValueError('completion region overlaps a preserved instance')
    return write_image(store,name,f'{family_id}/background_condition_region.png',
        Image.fromarray(rectangular.astype('uint8')*255))


def masked_background_condition(store,name,family_id,target,region):
    """Explicit derived model input; the original training target stays intact."""
    import numpy as np
    from PIL import Image
    with Image.open(store.resolve(target)) as image:source=np.array(image.convert("RGB"))
    with Image.open(store.resolve(region)) as image:mask=np.array(image.convert("L"))>127
    if source.shape[:2]!=mask.shape:raise ValueError("conditioning mask dimensions changed")
    conditioned=source.copy();conditioned[mask]=127
    return write_image(store,name,f"{family_id}/background_condition.png",Image.fromarray(conditioned))


def completion_blend_region(store,name,family_id,region,*,feather_pixels,selected=(),opaque_boxes=(),selected_edge_margin_pixels=0):
    """Feather inside the authorized region; protected pixels retain zero alpha."""
    from PIL import Image,ImageFilter
    import numpy as np
    if type(feather_pixels) is not int or not 0<=feather_pixels<=32:
        raise ValueError("background feather radius must be between zero and 32")
    if type(selected_edge_margin_pixels) is not int or not 0<=selected_edge_margin_pixels<=8:
        raise ValueError('selected edge margin must be between zero and eight pixels')
    if feather_pixels==0:return region
    with Image.open(store.resolve(region)) as image:mask=image.convert('L')
    core=mask.filter(ImageFilter.MinFilter(2*feather_pixels+1))
    softened=np.array(core.filter(ImageFilter.GaussianBlur(feather_pixels/2)))
    softened[np.asarray(mask)==0]=0
    # Feathering must not blend the original selected objects back in.
    for instance in selected:
        with Image.open(store.resolve(ArtifactRef.from_dict(instance['mask']))) as image:
            subject=np.asarray(image.convert('L'))>127
        if subject.shape!=softened.shape:raise ValueError('selected blend mask coordinates changed')
        subject=dilate_binary_region(subject,selected_edge_margin_pixels)
        softened[subject & (np.asarray(mask)>0)]=255
    softened[rectangle_region(softened.shape,opaque_boxes) & (np.asarray(mask)>0)]=255
    return write_image(store,name,f"{family_id}/completion_blend.png",Image.fromarray(softened))


def probe_video(path, *, decode=True):
    completed = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=60)
    stream = json.loads(completed.stdout)["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/")
    fps = float(numerator)/float(denominator)
    result = {"width": int(stream["width"]), "height": int(stream["height"]), "fps": fps,
              "container_frames": int(stream["nb_frames"]) if stream.get("nb_frames", "N/A").isdigit() else None}
    if not fps or min(result["width"], result["height"]) <= 0:
        raise ValueError("invalid video stream")
    if decode:
        import cv2
        capture = cv2.VideoCapture(str(path)); count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (result["height"], result["width"]):
                raise ValueError("decoded video frame dimensions changed")
            count += 1
        capture.release()
        if not count or (result["container_frames"] is not None and count != result["container_frames"]):
            raise ValueError("video is empty or has an incomplete frame decode")
        result.update(frames=count, duration_seconds=count/fps)
    return result


def _verify_decomposition_pixels(row,store):
    import numpy as np
    from PIL import Image
    annotations=row["annotations"]
    target=np.asarray(Image.open(store.resolve(ArtifactRef.from_dict(row["target"]["image"]))).convert("RGB"))
    background=np.asarray(Image.open(store.resolve(ArtifactRef.from_dict(row["input"]["background"]))).convert("RGB"))
    region=np.asarray(Image.open(store.resolve(ArtifactRef.from_dict(annotations["completion_region"]))).convert("L"))>127
    if target.shape!=background.shape or region.shape!=target.shape[:2] or not np.array_equal(target[~region],background[~region]):
        raise ValueError("decomposition background changed protected pixels")
    visible=np.asarray(Image.open(store.resolve(ArtifactRef.from_dict(annotations["background_visible"]))).convert("RGBA"))
    if visible.shape[:2]!=region.shape or not np.array_equal(visible[:,:,3],(~region).astype("uint8")*255):
        raise ValueError("visible background does not match the completion region")
    subjects={s["id"]:s for s in row["input"]["subjects"]}
    transforms={s["subject_id"]:s for s in annotations["reference_transforms"]}
    for instance in annotations["target_instances"]:
        mask=np.asarray(Image.open(store.resolve(ArtifactRef.from_dict(instance["mask"]))).convert("L"))>127
        if mask.shape!=region.shape or not mask.any():raise ValueError("target instance mask geometry changed")
        y,x=np.where(mask);box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
        pixels=target.copy();pixels[~mask]=0
        expected=np.dstack((pixels,mask.astype("uint8")*255))[box[1]:box[3],box[0]:box[2]]
        actual=np.asarray(Image.open(store.resolve(ArtifactRef.from_dict(subjects[instance["subject_id"]]["reference"]))).convert("RGBA"))
        if not np.array_equal(actual,expected) or transforms[instance["subject_id"]]["target_bbox_xyxy"]!=box:
            raise ValueError("reference pixels or transform do not bind their target instance")


def validate_record(row, store, *, require_accepted=True, allow_fixture=False):
    if row.get("schema_version") != "dataevolver.supervision.v1" or row.get("is_placeholder", False):
        raise ValueError("expected an actual supervision record")
    if row.get("evidence_kind") != "real_execution" and not (allow_fixture and row.get("evidence_kind") == "fixture"):
        raise ValueError("fixture or unknown evidence cannot be loaded as real data")
    identifier(row["id"], "record id"); identifier(row["family_id"], "family id")
    if row["split"] not in {"train", "validation", "test", "pilot"}:
        raise ValueError("invalid dataset split")
    annotations = row["annotations"]
    if require_accepted and (row.get("status") != "pass" or annotations.get("review", {}).get("status") != "pass"):
        raise ValueError("failed or uncertain record is not accepted")
    if not annotations.get("provenance") or not row.get("model_receipts"):
        raise ValueError("source provenance or real model receipts are absent")
    from dataevolver.adapters.model_contracts import ModelRequest, ModelReceipt
    identities = set()
    for record in row["model_receipts"]:
        request, receipt = ModelRequest.from_dict(record["request"]), ModelReceipt.from_dict(record["receipt"])
        if (receipt.request_id != request.request_id or request.request_id in identities or
                receipt.request_hash != payload_sha256({"request":request.to_dict(),"profile_hash":receipt.profile_hash}) or
                receipt.without_reuse_hash() != record["receipt_hash"] or
                (receipt.backend == "mock" and not allow_fixture)):
            raise ValueError("model request and receipt identity/hash are inconsistent")
        identities.add(request.request_id)
    inputs, target = row["input"], row["target"]
    if row["task_type"] == "text_to_image_generation":
        from .t2i_supervision import validate_t2i_binding
        validate_t2i_binding(row, store, require_accepted=require_accepted)
    elif row["task_type"] == "multi_subject_insertion":
        if set(inputs) != {"background", "subjects", "instruction"} or set(target) != {"image"}:
            raise ValueError("multi-subject training input/target whitelist changed")
        subjects = inputs["subjects"]
        identities = [s["id"] for s in subjects]
        bindings = annotations["target_instances"]
        if (not identities or len(set(identities)) != len(identities) or
                {b["subject_id"] for b in bindings} != set(identities) or len(bindings) != len(identities) or
                len({b["target_instance_id"] for b in bindings}) != len(bindings)):
            raise ValueError("subject references and target instances are not bound one to one")
        if not inputs["instruction"].strip() or any(set(s) != {"id", "reference"} for s in subjects):
            raise ValueError("subject input references are incomplete")
        required_checks = {"subject_correspondence", "reference_identity", "selected_removal",
                           "unselected_preservation", "background_completion"}
        criteria = annotations.get("review", {}).get("criteria", {})
        if require_accepted and any(criteria.get(k, {}).get("status") != "pass" or
                                    not criteria[k].get("evidence") for k in required_checks):
            raise ValueError("multi-subject quality criteria are failed, uncertain or unevidenced")
        if row["construction_mode"] == "scene_decomposition":
            if (annotations.get("background_completion", {}).get("inferred") is not True or
                    not annotations.get("background_visible") or not annotations.get("completion_region")):
                raise ValueError("inferred background completion provenance is missing")
            _verify_decomposition_pixels(row,store)
    elif row["task_type"] == "action_conditioned_physical_video_generation":
        if set(inputs) != {"initial_frame", "action_condition"} or set(target) != {"video"}:
            raise ValueError("video action/initial input or target whitelist changed")
        if not inputs["action_condition"] or not annotations.get("expected_events"):
            raise ValueError("video action or event plan missing")
        expected = {e["id"] for e in annotations["expected_events"] if e.get("required", True)}
        observed = annotations.get("observed_events", [])
        passed = {e["expected_event_id"] for e in observed if e.get("status") == "pass" and e.get("evidence")}
        if require_accepted and not expected.issubset(passed):
            raise ValueError("key video events lack confirmed observations and evidence")
        for event in observed:
            if event["status"] == "uncertain" and not event.get("reason"):
                raise ValueError("uncertain event lacks a reason")
        media = probe_video(store.resolve(ArtifactRef.from_dict(target["video"])))
        if media != annotations["actual_video"]:
            raise ValueError("actual video specification differs from decoded media")
    else:
        raise ValueError("unregistered supervision task")
    for ref in refs(row):
        path = store.resolve(ref)
        if ref.media_type.startswith("image/"):
            from PIL import Image
            with Image.open(path) as image:
                image.load()
    return row


def load_dataset(root, *, split=None, allow_fixture=False):
    root = Path(root).resolve()
    checksums = json.loads((root / "checksums.json").read_text())
    if not {"manifest.json", "stores.json"}.issubset(checksums):
        raise ValueError("bundle omits required metadata checksums")
    for relative, digest in checksums.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != digest:
            raise ValueError("portable supervision bundle hash changed")
    store_roots = {}
    for name, relative in json.loads((root / "stores.json").read_text()).items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("supervision store escapes bundle")
        store_roots[name] = path
    store = ArtifactStore(store_roots)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != "dataevolver.supervision_manifest.v1":
        raise ValueError("unknown supervision manifest schema")
    rows = manifest["records"]
    identities, families = set(), {}
    for row in rows:
        validate_record(row, store, allow_fixture=allow_fixture)
        if row["id"] in identities or families.setdefault(row["family_id"], row["split"]) != row["split"]:
            raise ValueError("duplicate record or source-family leakage")
        identities.add(row["id"])
    if manifest["split_counts"] != {s:sum(r["split"] == s for r in rows) for s in manifest["split_counts"]}:
        raise ValueError("manifest split counts disagree with loaded records")
    for _ in artifact_closure(rows, store):
        pass
    return [(row, store) for row in rows if split is None or row["split"] == split]


def export_dataset(rows, store, destination, *, expected_counts, metadata=None, allow_fixture=False):
    destination = Path(destination)
    counts = {split: sum(r["split"] == split for r in rows) for split in expected_counts}
    if counts != expected_counts or len(rows) != sum(expected_counts.values()):
        raise ValueError("qualified record count/split target not reached")
    for row in rows:
        validate_record(row, store, allow_fixture=allow_fixture)
    if destination.exists():
        loaded = load_dataset(destination, allow_fixture=allow_fixture)
        if [r for r, _ in loaded] != rows:
            raise ValueError("existing dataset bundle belongs to different records")
        return {"path": str(destination), "records": len(loaded), "reused": True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name+".partial-", dir=destination.parent))
    stores = {}
    for ref in artifact_closure(rows, store):
        stores[ref.store] = "stores/"+ref.store
        output = temporary/stores[ref.store]/ref.path
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            shutil.copy2(store.resolve(ref), output)
    (temporary/"manifest.json").write_text(canonical_json({"schema_version": "dataevolver.supervision_manifest.v1",
        "records": rows, "split_counts": counts, "metadata": metadata or {}, "training_benefit_verified": False}))
    (temporary/"stores.json").write_text(canonical_json(stores))
    (temporary/"checksums.json").write_text(canonical_json({p.relative_to(temporary).as_posix():file_sha256(p)
        for p in temporary.rglob("*") if p.is_file()}))
    load_dataset(temporary, allow_fixture=allow_fixture)
    temporary.rename(destination)
    return {"path": str(destination), "records": len(rows), "split_counts": counts,
            "manifest_sha256": file_sha256(destination/"manifest.json"), "reused": False}
