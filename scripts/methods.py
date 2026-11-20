"""Compare frozen LISTEN predictions with a supervised acoustic correction.

Inputs are JSONL, joined on unique, condition-specific 'id'.
Features: {id, label, condition, group, split, features:{name:number,...}}.
Optional recording or audio_hash identifies repeated recordings for audit and weighting.
A group that is a waveform SHA-256 also serves as its recording ID.
Use --protocol official_plus4 with the native HF train/test partitions. This
preserves the six native held-out audio conditions and adds a deterministic
20% type-4 holdout with cross-condition development collisions removed.
Predictions: {id, prediction, scores:{emotion:raw_logit,...}}. A score must exist
for each offered class. Labels are normalized following the upstream evaluator.
Use --protocol grouped to create out-of-group predictions, or --protocol split
with a test split and group-disjoint train (and optionally validation) splits.
Scores are natural-log-scale logits. Convert probabilities with log(p) first.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import joblib
from threadpoolctl import threadpool_limits
from sklearn.decomposition import PCA
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

TEXT = {"1_text", "2A", "3A"}
ALIASES = {
    "angry": "anger",
    "happy": "happiness",
    "sad": "sadness",
    "pleasant_surprise": "surprise",
}
NATIVE_AUDIO = ("1_audio", "1_audio_and_text", "2B", "2C", "3B", "3C")
MIXED_AUDIO = (*NATIVE_AUDIO, "4")
METHODS = ("base", "score_argmax", "calibrated", "acoustic", "fused")
INVALID = "<invalid>"


def label_name(value):
    name = str(value or "").strip().lower()
    return ALIASES.get(name, name)


def read_rows(path):
    result = {}
    with Path(path).open() as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("id")
            if not isinstance(key, str) or not key or key in result:
                raise ValueError(f"{path}:{lineno} has a missing or duplicate id: {key!r}")
            result[key] = row
    return result


def official_split(features, seed):
    """Determine holdout solely from the full feature manifest, before predictions or labels."""
    groups = {str(r["group"]) for r in features.values()}
    parent = {g: g for g in groups}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    recordings = {}
    for r in features.values():
        g = str(r["group"])
        key = r.get("audio_hash") or r.get("recording")
        if key:
            rep = recordings.setdefault(key, g)
            a, b = find(rep), find(g)
            if a != b:
                parent[max(a, b)] = min(a, b)
    for r in features.values():
        r["group"] = find(str(r["group"]))
    native_groups = {r["group"] for r in features.values() if r["split"] == "test"}
    type4 = {
        r["group"] for r in features.values() if r["split"] == "train" and r["condition"] == "4"
    } - native_groups

    def rank(group):
        return hashlib.sha256(f"LISTEN-type4-v1:{seed}:{group}".encode()).hexdigest()

    holdout = set(sorted(type4, key=rank)[: math.ceil(0.2 * len(type4))])
    excluded = Counter()
    kept = {}
    for k, r in features.items():
        if r["split"] == "test":
            r["origin"] = "native_test"
        elif r["condition"] == "4" and r["split"] == "train" and r["group"] in holdout:
            r["split"], r["origin"] = "test", "type4_holdout"
        elif r["group"] in native_groups or r["group"] in holdout:
            excluded[r["condition"]] += 1
            continue
        else:
            r["origin"] = "development"
        kept[k] = r
    info = {
        "type4_eligible_groups": len(type4),
        "type4_heldout_groups": len(holdout),
        "rank_rule": "first ceil(0.2*n) groups sorted by sha256(LISTEN-type4-v1:<seed>:<group>)",
        "purged_development_rows": dict(excluded),
        "seed": seed,
        "holdout_groups": sorted(holdout),
    }
    return kept, info


def load(
    features_path,
    predictions_path,
    conditions,
    allow_partial,
    protocol,
    seed,
    feature_names,
    embeddings,
    selected_ids=None,
    inference=False,
):
    features = read_rows(features_path)
    if protocol == "official_plus4":
        features, split_info = official_split(features, seed)
    else:
        split_info = {}
    predictions = read_rows(predictions_path)
    wanted = {i for i, r in features.items() if not conditions or r["condition"] in conditions}
    if selected_ids is not None:
        wanted &= set(selected_ids)
    if protocol == "official_plus4":
        wanted = {
            i
            for i in wanted
            if features[i]["condition"] not in TEXT or features[i]["split"] == "test"
        }
    missing = sorted(wanted - predictions.keys())
    if missing and not allow_partial:
        raise ValueError(
            f"{len(missing)} rows without predictions, e.g. {missing[:3]}. "
            "Use --allow-partial only for an explicitly partial evaluation."
        )
    if conditions:
        unknown = conditions - {r["condition"] for r in features.values()}
        if unknown:
            raise ValueError(f"Requested conditions absent from features: {unknown}")
    rows = []
    skipped = Counter()
    for key in sorted(wanted & predictions.keys()):
        f, p = features[key], predictions[key]
        condition = f["condition"]
        group = str(f.get("group") or (key if inference else "")).strip()
        target = label_name(f.get("label"))
        if not group or (not inference and not target):
            raise ValueError(f"Missing group or target in {key}")
        if not p.get("scores") and not p.get("prediction"):
            skipped[(str(f.get("origin", f.get("split", "train"))), condition)] += 1
            continue
        scores = defaultdict(list)
        for k, v in (p.get("scores") or {}).items():
            if not np.isfinite(v) or not label_name(k):
                raise ValueError(f"Invalid score in {key}: {k!r}, {v!r}")
            scores[label_name(k)].append(float(v))
        # Aliased options correspond to distinct displayed letters. Sum their mass.
        scores = {k: float(logsumexp(v)) for k, v in scores.items()}
        if condition not in TEXT and (not scores or (not inference and target not in scores)):
            raise ValueError(f"No candidate logit for acoustic target {target!r} in {key}")
        if "prediction" not in p:
            raise ValueError(f"Missing explicit original prediction in {key}")
        split = str(f.get("split") or "train").strip().lower()
        split = {"dev": "validation", "val": "validation", "eval": "test"}.get(split, split)
        feat = f.get("features", {})
        if not isinstance(feat, dict) or (condition not in TEXT and not feat):
            raise ValueError(f"Missing numeric acoustic features in {key}")
        if condition not in TEXT and feature_names:
            absent = set(feature_names) - feat.keys()
            if absent:
                raise ValueError(f"Requested features missing for {key}: {absent}")
            feat = {k: feat[k] for k in feature_names}
        emb_path = None
        if embeddings and condition not in TEXT:
            path = p.get("embedding_path")
            if not path:
                raise ValueError(f"No embedding_path in {key}")
            path = Path(path)
            if not path.is_absolute():
                path = Path(predictions_path).parent / path
            emb_path = str(path)
        recording = f.get("audio_hash") or f.get("recording") or ""
        if not recording and len(group) == 64 and all(c in "0123456789abcdef" for c in group):
            recording = group
        rows.append(
            {
                "id": key,
                "target": target,
                "condition": condition,
                "group": group,
                "origin": f.get("origin", split),
                "recording": str(recording),
                "split": split,
                "features": feat,
                "embedding_path": emb_path,
                "scores": scores,
                "base": label_name(p["prediction"]) or INVALID,
            }
        )
    if not rows:
        raise ValueError("No matched evaluation rows.")
    return rows, {
        "missing_prediction_rows": len(missing),
        "missing_by_condition": dict(Counter(features[i]["condition"] for i in missing)),
        "jointly_skipped_null_prediction_and_scores": {
            f"{sp}/{c}": v for (sp, c), v in skipped.items()
        },
        "official_plus4_manifest": split_info,
    }


def matrix(rows, names=None, labels=None):
    # Numeric schema can be read without consulting held-out labels or distributions.
    names = names or sorted({k for r in rows for k in r["features"]})
    if labels is None:
        labels = sorted({k for r in rows for k in r["scores"]})
    ids = {k: i for i, k in enumerate(labels)}
    emb_size = 0
    if rows[0]["embedding_path"]:
        sample = np.load(rows[0]["embedding_path"], mmap_mode="r", allow_pickle=False)
        if sample.ndim != 1:
            raise ValueError(f"Expected a pooled 1D embedding, found {sample.shape}")
        emb_size = sample.size
    x = np.full((len(rows), len(names) + emb_size), np.nan, dtype=np.float32)
    emb_rows = {}

    z = np.full((len(rows), len(labels)), -np.inf)
    for i, r in enumerate(rows):
        for j, k in enumerate(names):
            val = r["features"].get(k)
            if val is not None:
                x[i, j] = float(val)
        if emb_size:
            path = r["embedding_path"]
            if path in emb_rows:
                x[i, len(names) :] = x[emb_rows[path], len(names) :]
            else:
                v = np.load(path, allow_pickle=False)
                if v.shape != (emb_size,) or not np.isfinite(v).all():
                    raise ValueError(f"Invalid pooled embedding: {path}, shape={v.shape}")
                x[i, len(names) :] = v
                emb_rows[path] = i
        for k, v in r["scores"].items():
            z[i, ids[k]] = v
    x[~np.isfinite(x)] = np.nan
    y = np.array([ids.get(r["target"], -1) for r in rows])
    return x, z, y, np.array(labels), list(names)


def groups_array(rows):
    return np.array([r["group"] for r in rows])


def splits(idx, groups, n):
    num = min(n, len(set(groups[idx])))
    if num < 2:
        raise ValueError("Need at least two independent training groups to tune fusion.")
    return [(idx[tr], idx[te]) for tr, te in GroupKFold(num).split(idx, groups=groups[idx])]


def audit(rows):
    group_splits = defaultdict(set)
    recording_groups = defaultdict(set)
    recording_splits = defaultdict(set)
    for r in rows:
        group_splits[r["group"]].add(r["split"])
        if r["recording"]:
            recording_groups[r["recording"]].add(r["group"])
            recording_splits[r["recording"]].add(r["split"])
    return {
        "groups": len(group_splits),
        "known_recordings": len(recording_groups),
        "rows_with_recording_key": sum(bool(r["recording"]) for r in rows),
        "recordings_crossing_groups": sum(len(v) > 1 for v in recording_groups.values()),
        "recordings_crossing_input_splits": sum(len(v) > 1 for v in recording_splits.values()),
        "groups_crossing_input_splits": sum(len(v) > 1 for v in group_splits.values()),
    }


def training_weights(rows, idx):
    # The same physical audio can have 2B/2C or 3B/3C rows. Avoid double
    # weighting it when fitting the acoustic-only classifier.
    keys = [rows[i]["recording"] or rows[i]["id"] for i in idx]
    counts = Counter(keys)
    weights = np.array([1.0 / counts[k] for k in keys])
    return weights / weights.mean()


def prepare_x(x, bundle):
    raw = x[:, : bundle["prosody_width"]][:, bundle["finite"]]
    columns = [np.where(np.isnan(raw), bundle["median"], raw)]
    if bundle["missing"].any():
        columns.append(np.isnan(raw[:, bundle["missing"]]))
    if bundle["pca"] is not None:
        columns.append(bundle["pca"].transform(x[:, bundle["prosody_width"] :]))
    return np.column_stack(columns)


def fit_acoustic(x, y, tr, rows, prosody_width, use_emb, args):
    raw = x[tr, :prosody_width]
    finite = np.isfinite(raw).any(axis=0)
    if not finite.any() and not use_emb:
        raise ValueError("Every acoustic feature is missing on this training partition.")
    columns = raw[:, finite]
    model = {
        "prosody_width": prosody_width,
        "median": np.nanmedian(columns, axis=0),
        "finite": finite,
        "missing": np.isnan(columns).any(axis=0),
        "pca": None,
        "scaler": None,
        "clf": None,
        "classes": np.unique(y[tr]),
    }
    if use_emb:
        # Fit PCA once per unique training recording. Both an audio-only prompt
        # and the audio+text prompt can supply the identical encoder vector.
        unique = {}
        for i in tr:
            unique.setdefault(rows[i]["embedding_path"], i)
        embs = x[list(unique.values()), prosody_width:]
        k = min(64, len(embs) - 1, embs.shape[1])
        if k < 1:
            raise ValueError("Need at least two unique training embeddings.")
        model["pca"] = PCA(
            n_components=k, svd_solver="randomized", iterated_power=3, random_state=args.seed
        ).fit(embs)
    train = prepare_x(x[tr], model)
    weights = training_weights(rows, tr)
    if len(model["classes"]) > 1:
        if args.head == "linear":
            scaler = StandardScaler().fit(train, sample_weight=weights)
            model["scaler"] = scaler
            clf = LogisticRegression(C=1, max_iter=1000)
            train = scaler.transform(train)
        else:
            clf = ExtraTreesClassifier(
                n_estimators=128,
                min_samples_leaf=4,
                max_features=0.8,
                n_jobs=args.jobs,
                random_state=args.seed,
            )
        model["clf"] = clf.fit(train, y[tr], sample_weight=weights)
    return model


def predict_acoustic(x, model, n_classes):
    probs = np.full((len(x), n_classes), 1e-3 / n_classes)
    if model["clf"] is None:
        probs[:, model["classes"][0]] += 1
    else:
        values = prepare_x(x, model)
        if model["scaler"] is not None:
            values = model["scaler"].transform(values)
        probs[:, model["classes"]] += model["clf"].predict_proba(values)
    return np.log(probs / probs.sum(axis=1, keepdims=True))


def fit_calibrator(z, y, tr):
    # Temperature and shrunk class biases. Row-specific answer masks remain fixed.
    train = z[tr]
    finite = np.isfinite(train)
    centered = np.where(finite, train - np.max(train, axis=1, keepdims=True), 0.0)
    classes = train.shape[1]
    one_hot = np.eye(classes)[y[tr]]

    def objective(params):
        scale = params[0]
        biases = params[1:]
        s = np.where(finite, centered * scale + biases, -np.inf)
        log_norm = logsumexp(s, axis=1)
        prob = np.exp(s - log_norm[:, None])
        residual = (prob - one_hot) / len(tr)
        penalty = 0.01
        loss = np.mean(log_norm - s[np.arange(len(tr)), y[tr]])
        loss += penalty * (np.sum(biases**2) + (scale - 1) ** 2) / 2
        grad = np.r_[
            np.sum(residual * centered) + penalty * (scale - 1),
            residual.sum(axis=0) + penalty * biases,
        ]
        return loss, grad

    fit = minimize(
        objective,
        np.r_[1.0, np.zeros(classes)],
        jac=True,
        bounds=[(0.03, 10.0)] + [(-8.0, 8.0)] * classes,
        method="L-BFGS-B",
    )
    if not fit.success:
        raise RuntimeError(f"Class calibrator failed: {fit.message}")
    return {"scale": fit.x[0], "biases": fit.x[1:]}


def mixed_probabilities(cal, acoustic, w):
    # Normalize after masking: candidates differ across benchmark sources.
    a = np.where(np.isfinite(cal), acoustic, -np.inf)
    pa = np.exp(a - logsumexp(a, axis=1, keepdims=True))
    pb = np.exp(cal - logsumexp(cal, axis=1, keepdims=True))
    return (1 - w) * pb + w * pa


def combined(cal, acoustic, w):
    return np.argmax(mixed_probabilities(cal, acoustic, w), axis=1)


def inner_score(pred, truth, conditions, metric):
    by_condition = []
    for c in sorted(set(conditions)):
        keep = conditions == c
        if metric == "wa":
            by_condition.append(np.mean(truth[keep] == pred[keep]))
        else:
            by_condition.append(
                np.mean(
                    [np.mean(pred[keep][truth[keep] == k] == k) for k in np.unique(truth[keep])]
                )
            )
    return math.fsum(by_condition) / len(by_condition)


def predict_bundle(x, z, bundle):
    cal = z * bundle["calibrator"]["scale"] + bundle["calibrator"]["biases"]
    labels = np.asarray(bundle["classes"])
    acoustic = predict_acoustic(x, bundle["acoustic"], len(labels))
    return {
        "score_argmax": labels[np.argmax(z, axis=1)],
        "calibrated": labels[np.argmax(cal, axis=1)],
        "acoustic": labels[combined(cal, acoustic, 1.0)],
        "fused": labels[combined(cal, acoustic, bundle["fusion_weight"])],
    }


def evaluate_partition(rows, arrays, tr, te, validation, args):
    x, z, y, labels, features = arrays
    groups = groups_array(rows)
    conditions = np.array([r["condition"] for r in rows])
    inner = [(tr, validation)] if len(validation) else splits(tr, groups, args.inner_folds)
    ids = np.concatenate([j for _, j in inner])
    grid = np.arange(11) / 10
    if args.select_linear_head:
        seen = set(conditions[ids])
        if set(MIXED_AUDIO) != seen and not args.allow_partial:
            raise ValueError(
                f"Selection requires all seven training validation conditions, observed {sorted(seen)}"
            )
    candidates = [False, True] if args.select_linear_head else [args.embeddings]
    cv_cal = []
    for i, j in inner:
        cal = fit_calibrator(z, y, i)
        cv_cal.append(z[j] * cal["scale"] + cal["biases"])
    cv_cal = np.concatenate(cv_cal)
    search = {}
    choice = None
    # Selection only accesses training indices and their held-out validation predictions.
    for use_emb in candidates:
        name = "prosody_plus_encoder" if use_emb else "prosody"
        val_acoustic = []
        for i, j in inner:
            model = fit_acoustic(x, y, i, rows, len(features), use_emb, args)
            val_acoustic.append(predict_acoustic(x[j], model, len(labels)))
        val_acoustic = np.concatenate(val_acoustic)
        scores = [
            inner_score(
                combined(cv_cal, val_acoustic, w), y[ids], conditions[ids], args.selection_metric
            )
            for w in grid
        ]
        best = next(k for k, value in enumerate(scores) if value >= max(scores) - 1e-12)
        search[name] = {
            "fusion_weight": float(grid[best]),
            "best_validation_score": scores[best],
            "validation_grid": dict(zip(map(str, grid), scores)),
        }
        if choice is None or scores[best] > search[choice]["best_validation_score"] + 1e-12:
            choice = name
    # The winning candidate and its hyperparameter are now frozen.
    calibrator = fit_calibrator(z, y, tr)
    bundles, predictions = {}, {}
    for use_emb in candidates:
        name = "prosody_plus_encoder" if use_emb else "prosody"
        bundle = {
            "format_version": 1,
            "classes": labels.tolist(),
            "features": features,
            "uses_embeddings": use_emb,
            "embedding_size": (x.shape[1] - len(features)) if use_emb else 0,
            "head": args.head,
            "calibrator": calibrator,
            "acoustic": fit_acoustic(x, y, tr, rows, len(features), use_emb, args),
            "fusion_weight": search[name]["fusion_weight"],
        }
        bundles[name] = bundle
        predictions[name] = predict_bundle(x[te], z[te], bundle)
    selected = predictions[choice].copy()
    for name in predictions:
        selected[f"acoustic_{name}"] = predictions[name]["acoustic"]
        selected[f"fused_{name}"] = predictions[name]["fused"]
    meta = {
        "train_rows": len(tr),
        "validation_predictions": len(ids),
        "test_rows": len(te),
        "train_groups": len(set(groups[tr])),
        "test_groups": len(set(groups[te])),
        "validation_conditions": sorted(set(conditions[ids])),
        "selected_head": choice,
        "fusion_weight": search[choice]["fusion_weight"],
        "validation_grid": search[choice]["validation_grid"],
        "candidate_search": search,
    }
    return selected, meta, {"selected_head": choice, "candidates": bundles, "validation": meta}


def metrics(rows, method):
    true = np.array([r["target"] for r in rows])
    pred = np.array([r[method] for r in rows])
    recalls, f1 = [], []
    for k in sorted(set(true) | set(pred)):
        positives = true == k
        predicted = pred == k
        tp = np.sum(positives & predicted)
        if positives.any():
            recalls.append(tp / positives.sum())
        f1.append(2 * tp / (positives.sum() + predicted.sum()))
    return {
        "n": len(rows),
        "wa": float(np.mean(true == pred)),
        "uar": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1)),
    }


def paired(rows, ref, method, iterations, seed):
    # Resample whole independent groups with replacement, keeping all method
    # predictions paired. Estimate intervals conditional on the fitted CV outputs.
    groups = sorted({r["group"] for r in rows})
    counts = np.zeros((len(groups), 2))
    idx = {g: i for i, g in enumerate(groups)}
    discordant = Counter()
    for r in rows:
        old = int(r[ref] == r["target"])
        new = int(r[method] == r["target"])
        counts[idx[r["group"]]] += [new - old, 1]
        if new != old:
            discordant["corrected" if new else "broken"] += 1
    delta = counts[:, 0].sum() / len(rows)
    info = {
        "delta_wa": float(delta),
        "n_groups": len(groups),
        "corrected": discordant["corrected"],
        "broken": discordant["broken"],
    }
    if iterations and len(groups) > 1:
        rng = np.random.default_rng(seed)
        reps = []
        for start in range(0, iterations, 64):
            draw = rng.integers(len(groups), size=(min(64, iterations - start), len(groups)))
            acc = counts[draw, :2].sum(axis=1)
            reps.extend(acc[:, 0] / acc[:, 1])
        info["delta_wa_percentile95"] = np.quantile(reps, [0.025, 0.975]).tolist()
    return info


def report_condition_mean(output, names, allowed_origins, iterations, seed):
    selected = [r for r in output if r["condition"] in names and r["origin"] in allowed_origins]
    present = [c for c in names if any(r["condition"] == c for r in selected)]
    if not present:
        return {"conditions": [], "complete": False}
    groups = sorted({r["group"] for r in selected})
    gi, ci = {g: i for i, g in enumerate(groups)}, {c: i for i, c in enumerate(present)}
    n = np.zeros((len(groups), len(present)), dtype=int)
    deltas = {m: np.zeros_like(n) for m in ("base", "calibrated", "acoustic")}
    per = {}
    methods = (
        *METHODS,
        *sorted(k for k in selected[0] if k.startswith("fused_") or k.startswith("acoustic_")),
    )
    for c in present:
        subset = [r for r in selected if r["condition"] == c]
        per[c] = {m: metrics(subset, m) for m in methods}
    for r in selected:
        i, j = gi[r["group"]], ci[r["condition"]]
        n[i, j] += 1
        for m, diff in deltas.items():
            diff[i, j] += int(r["fused"] == r["target"]) - int(r[m] == r["target"])
    means = {
        m: {k: float(np.mean([per[c][m][k] for c in present])) for k in ("wa", "uar", "macro_f1")}
        for m in methods
    }
    comparisons = {
        m: {"delta_wa": float(np.mean(d.sum(axis=0) / n.sum(axis=0)))} for m, d in deltas.items()
    }
    # The same draw is used across every condition and every reference. This
    # retains dependencies for repeated clips appearing in, for example, 2B/2C.
    if iterations and len(groups) > 1:
        rng = np.random.default_rng(seed)
        sampled = defaultdict(list)
        prop = np.full(len(groups), 1 / len(groups))
        arr = np.column_stack((n, *deltas.values())).astype(float)
        for start in range(0, iterations, 64):
            weights = rng.multinomial(len(groups), prop, size=min(64, iterations - start))
            blocks = np.split(weights @ arr, 1 + len(deltas), axis=1)
            valid = (blocks[0] > 0).all(axis=1)
            for m, data in zip(deltas, blocks[1:]):
                sampled[m].extend(np.mean(data[valid] / blocks[0][valid], axis=1))
        for m, values in sampled.items():
            if values:
                comparisons[m]["delta_wa_percentile95"] = np.quantile(
                    values, [0.025, 0.975]
                ).tolist()
            comparisons[m]["bootstrap_complete_draws"] = len(values)
    return {
        "conditions": present,
        "complete": len(present) == len(names),
        "n_by_condition": {c: per[c]["base"]["n"] for c in present},
        "unweighted_condition_means": means,
        "paired_fused_minus": comparisons,
    }


def replay(
    bundle_path, features_path, predictions_path, expected=None, only_selected=False, jobs=4
):
    saved = joblib.load(bundle_path)
    models = [
        f["candidates"][name]
        for f in saved["folds"]
        for name in ([f["selected_head"]] if only_selected else f["candidates"])
    ]
    names = models[0]["features"] if models else None
    requires_emb = any(model["uses_embeddings"] for model in models)
    rows, _ = load(
        features_path,
        predictions_path,
        None,
        False,
        "apply",
        0,
        names,
        requires_emb,
        selected_ids=expected,
        inference=True,
    )
    output = {}
    audio_by_fold = defaultdict(list)
    extra = set()
    for f in saved["folds"]:
        for name in f["candidates"]:
            extra.update((f"fused_{name}", f"acoustic_{name}"))
    for r in rows:
        if r["condition"] in TEXT:
            methods = METHODS if only_selected else (*METHODS, *sorted(extra))
            output[r["id"]] = {m: r["base"] for m in methods}
        else:
            if len(saved["folds"]) == 1:
                fold = 0
            elif r["group"] in saved["group_to_fold"]:
                fold = saved["group_to_fold"][r["group"]]
            else:
                raise ValueError(
                    "No held-out fold known for this group; use a single training/test-split artifact for deployment."
                )
            audio_by_fold[fold].append(r)
    for fid, samples in audio_by_fold.items():
        f = saved["folds"][fid]
        selected = f["selected_head"]
        ref = f["candidates"][selected]
        x, z, _, _, _ = matrix(samples, names=ref["features"], labels=ref["classes"])
        for r in samples:
            output[r["id"]] = {"base": r["base"]}
        for name, bundle in f["candidates"].items():
            if only_selected and name != selected:
                continue
            if bundle["uses_embeddings"] and x.shape[1] != len(names) + bundle["embedding_size"]:
                raise ValueError(f"Unexpected embedding dimension for saved {name} head.")
            with threadpool_limits(limits=jobs):
                pred = predict_bundle(x, z, bundle)
            for i, r in enumerate(samples):
                dst = output[r["id"]]
                if name == selected:
                    dst.update({m: str(v[i]) for m, v in pred.items()})
                if not only_selected:
                    dst.update(
                        {
                            f"acoustic_{name}": str(pred["acoustic"][i]),
                            f"fused_{name}": str(pred["fused"][i]),
                        }
                    )
    checked = 0
    if expected is not None:
        if set(output) != set(expected):
            raise ValueError(
                f"Replay identifier mismatch: {len(set(expected)-set(output))} missing, "
                f"{len(set(output)-set(expected))} unexpected"
            )
        for key, methods in output.items():
            for m, value in methods.items():
                if expected[key].get(m) != value:
                    raise ValueError(
                        f"Replay differs on {key} / {m}: {value} vs {expected[key].get(m)}"
                    )
                checked += 1
    return output, {"rows": len(output), "method_predictions_checked": checked, "mismatches": 0}


def apply_cli():
    p = argparse.ArgumentParser(
        description="Apply a local trusted fitted_heads.joblib to frozen model outputs."
    )
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument(
        "--expected", help="Verify every ID/method in the supplied heldout_predictions.jsonl"
    )
    p.add_argument(
        "--selected-only",
        action="store_true",
        help="Deploy only the selected head; skips ablation fields.",
    )
    p.add_argument("--output", type=Path, help="Write new JSONL predictions")
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args(sys.argv[2:])
    expected = read_rows(args.expected) if args.expected else None
    output, summary = replay(
        args.bundle, args.features, args.predictions, expected, args.selected_only, args.jobs
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as stream:
            for key, values in sorted(output.items()):
                print(json.dumps({"id": key, **values}, ensure_ascii=False), file=stream)
    print(json.dumps(summary))


def checkpoint_identity(model):
    """Config/index signatures distinguish local 3B/7B checkpoints and repackaging."""
    root = Path(model)
    config = root / "config.json"
    if not config.is_file():
        raise ValueError(f"Expected a local checkpoint directory containing config.json: {model}")
    identity = {"config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()}
    index = root / "model.safetensors.index.json"
    if index.is_file():
        identity["weights_index_sha256"] = hashlib.sha256(index.read_bytes()).hexdigest()
    return identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--model",
        type=Path,
        help="Local frozen checkpoint used for base predictions; binds the fitted artifact to its architecture config and weight index.",
    )
    p.add_argument("--conditions", help="Comma-separated condition names")
    p.add_argument(
        "--protocol", choices=["auto", "grouped", "split", "official_plus4"], default="auto"
    )
    p.add_argument("--head", choices=["linear", "trees"], default="linear")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--selection-metric", choices=["wa", "uar"], default="wa")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--feature-names",
        help="Comma-separated predeclared numeric features; default uses all numeric features.",
    )
    p.add_argument(
        "--embeddings",
        action="store_true",
        help="Exploratory separate variant also uses pooled embeddings specified in each prediction's embedding_path.",
    )
    p.add_argument(
        "--select-linear-head",
        action="store_true",
        help="Select three prosody features vs those plus 64 train-fold PCA components of frozen audio embeddings by train-only seven-condition mean WA.",
    )
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    if args.jobs < 1:
        p.error("--jobs must be positive")
    if args.select_linear_head:
        if args.head != "linear" or args.selection_metric != "wa":
            p.error("--select-linear-head requires linear classifiers and mean WA")
        if not args.feature_names:
            args.feature_names = "pitch_span,rms_span,pause_fraction"
    conditions = set(args.conditions.split(",")) if args.conditions else None
    feature_names = args.feature_names.split(",") if args.feature_names else None
    rows, alignment = load(
        args.features,
        args.predictions,
        conditions,
        args.allow_partial,
        args.protocol,
        args.seed,
        feature_names,
        args.embeddings or args.select_linear_head,
    )
    audits = audit(rows)
    if audits["recordings_crossing_groups"]:
        raise ValueError(
            f"Recording identities span groups: {audits}; merge conflicting groups first."
        )
    acoustic = [r for r in rows if r["condition"] not in TEXT]
    text = [r for r in rows if r["condition"] in TEXT]
    protocol = args.protocol
    if protocol == "auto":
        protocol = "split" if any(r["split"] == "test" for r in rows) else "grouped"
    if protocol in {"split", "official_plus4"}:
        if audits["groups_crossing_input_splits"] or audits["recordings_crossing_input_splits"]:
            raise ValueError(f"Supplied partitions leak groups or recordings: {audits}")
        bad = {r["split"] for r in rows} - {"train", "validation", "test"}
        if bad:
            raise ValueError(f"Unexpected split labels: {bad}")
    output = []
    folds = []
    model_identity = checkpoint_identity(args.model) if args.model else None
    saved = {
        "folds": [],
        "group_to_fold": {},
        "protocol": protocol,
        "checkpoint_identity": model_identity,
    }
    if acoustic:
        groups = groups_array(acoustic)
        arrays = matrix(acoustic)
        if protocol == "grouped":
            outer = [
                (i, j, np.array([], dtype=int))
                for i, j in splits(np.arange(len(acoustic)), groups, args.folds)
            ]
        else:
            sp = np.array([r["split"] for r in acoustic])
            tr, va, te = (np.flatnonzero(sp == k) for k in ("train", "validation", "test"))
            if not len(tr) or not len(te):
                raise ValueError(
                    "Explicit-split mode requires acoustic training and test examples."
                )
            outer = [(tr, te, va)]
        for fold, (tr, te, val) in enumerate(outer):
            with threadpool_limits(limits=args.jobs):
                pred, meta, bundle = evaluate_partition(acoustic, arrays, tr, te, val, args)
            saved["folds"].append(bundle)
            saved["group_to_fold"].update({g: fold for g in groups[te]})
            folds.append({"fold": fold, **meta})
            for j, index in enumerate(te):
                r = acoustic[index]
                row = {k: r[k] for k in ("id", "condition", "target", "group", "origin", "base")}
                output.append({**row, "fold": fold, **{k: str(v[j]) for k, v in pred.items()}})
    extra_methods = sorted(
        {
            name
            for row in output
            for name in row
            if name.startswith("acoustic_") or name.startswith("fused_")
        }
    )
    all_methods = (*METHODS, *extra_methods)
    for r in text:
        if protocol in {"split", "official_plus4"} and r["split"] != "test":
            continue
        row = {k: r[k] for k in ("id", "condition", "target", "group", "origin", "base")}
        output.append({**row, "fold": None, **{m: r["base"] for m in all_methods}})
    if not output:
        raise ValueError("No evaluation predictions.")
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(saved, args.output / "fitted_heads.joblib", compress=3)
    with (args.output / "heldout_predictions.jsonl").open("w") as stream:
        for r in sorted(output, key=lambda r: r["id"]):
            print(json.dumps(r, ensure_ascii=False), file=stream)
    per = {}
    comparisons = {}
    for c in sorted({r["condition"] for r in output}):
        subset = [r for r in output if r["condition"] == c]
        per[c] = {m: metrics(subset, m) for m in all_methods}
        comparisons[c] = {
            ref: paired(subset, ref, "fused", args.bootstrap, args.seed)
            for ref in ("base", "calibrated", "acoustic")
        }
    averages = {
        key: {
            m: {
                metric: float(np.mean([per[c][m][metric] for c in cs]))
                for metric in ("wa", "uar", "macro_f1")
            }
            for m in all_methods
        }
        for key, cs in (
            ("all_conditions", list(per)),
            ("audio_conditions", [c for c in per if c not in TEXT]),
        )
        if cs
    }
    result = {
        "protocol": protocol,
        "head": args.head,
        "selection_metric": args.selection_metric,
        "seed": args.seed,
        "feature_names": feature_names or "all",
        "plus_frozen_embeddings": args.embeddings or args.select_linear_head,
        "head_selection": args.select_linear_head,
        "checkpoint_identity": model_identity,
        "embedding_reduction": "Fold-fitted PCA on unique training recordings, min(64,n_recordings-1,n_dimensions) components.",
        "alignment": alignment,
        "audit": audits,
        "folds": folds,
        "per_condition": per,
        "unweighted_condition_means": averages,
        "paired_delta_wa_fused_minus": comparisons,
        "bootstrap_description": "95% paired group-percentile intervals conditional on fitted predictions.",
    }
    if protocol == "official_plus4":
        result["native_six"] = report_condition_mean(
            output, NATIVE_AUDIO, {"native_test"}, args.bootstrap, args.seed
        )
        result["mixed_seven"] = report_condition_mean(
            output, MIXED_AUDIO, {"native_test", "type4_holdout"}, args.bootstrap, args.seed
        )
        if not (args.allow_partial or conditions or result["mixed_seven"]["complete"]):
            raise ValueError(
                f"Official-plus4 output lacks all seven audio conditions: {result['mixed_seven']['conditions']}"
            )
    _, result["artifact_replay"] = replay(
        args.output / "fitted_heads.joblib",
        args.features,
        args.predictions,
        {r["id"]: r for r in output},
        jobs=args.jobs,
    )
    with (args.output / "metrics.json").open("w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "protocol": protocol,
                "conditions": list(per),
                "condition_means": averages,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    if sys.argv[1:2] == ["apply"]:
        apply_cli()
    else:
        main()
