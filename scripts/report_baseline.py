#!/usr/bin/env python3
"""Independently score condition-expanded LISTEN JSONL and frozen predictions.

Example: python report_baseline.py --data data --results runs/train.jsonl runs/test.jsonl \
    --output results/frozen_baseline.json
Current author WA excludes undecodable letters. Counts and exact-letter/all-row
accuracy expose the effect of that rule; paper-neutral labels are scored separately.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
import string

CONDITIONS = ('1_text', '1_audio', '1_audio_and_text', '2A', '2B', '2C', '3A', '3B', '3C', '4')
AUDIO = ('1_audio', '1_audio_and_text', '2B', '2C', '3B', '3C', '4')
ALIASES = {'happy': 'happiness', 'sad': 'sadness', 'angry': 'anger', 'pleasant_surprise': 'surprise'}
INVALID = '<invalid>'
METRICS = ('wa', 'uar', 'macro_f1', 'micro_f1', 'prediction_marginal_accuracy')
METHODS = ('base', 'score_argmax', 'calibrated', 'acoustic', 'fused')


def normal(label):
    value = str(label or '').lower().strip()
    return ALIASES.get(value, value)


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_manifests(path, splits):
    by_id = {}
    paths = [(s, path / (s + '.jsonl')) for s in splits] if path.is_dir() else [(None, path)]
    for expected_split, file in paths:
        for row in read_jsonl(file):
            split = row.get('split', expected_split or (splits[0] if len(splits) == 1 else None))
            if not split or (expected_split and split != expected_split):
                raise ValueError(f'Invalid or missing split for {row["id"]} in {file}')
            if split not in splits:
                continue
            if row['id'] in by_id or row['condition'] not in CONDITIONS:
                raise ValueError(f'Duplicate id or unknown condition: {row["id"]}')
            by_id[row['id']] = dict(row, split=split)
    return by_id


def load_predictions(paths, splits):
    predictions, files = {}, []
    for item in paths:
        if not item.is_dir():
            files.append(item)
            continue
        for split in splits:
            names = [f'{split}.jsonl', f'{split}_predictions.jsonl', f'predictions_{split}.jsonl',
                     f'{split}/predictions.jsonl']
            hits = [item / name for name in names if (item / name).is_file()]
            if len(hits) != 1:
                raise ValueError(f'Provide explicit prediction JSONL paths; found {hits} for {split} in {item}')
            files.extend(hits)
    for path in files:
        for record in read_jsonl(path):
            key = record['id']
            if key in predictions:
                raise ValueError(f'Duplicate prediction {key} in {path}')
            if 'predicted_letter' not in record or 'expected_letter' not in record or 'choices' not in record:
                raise ValueError(f'Incomplete frozen prediction {key} in {path}')
            predictions[key] = (record, path.parent)
    return predictions


def metrics(pairs):
    if not pairs:
        return {k: None for k in METRICS}
    true = Counter(g for g, _ in pairs)
    pred = Counter(p for _, p in pairs)
    classes = set(true) | set(pred)
    tp = Counter(g for g, p in pairs if g == p)
    n = len(pairs)
    wa = sum(tp.values()) / n
    return dict(wa=wa, micro_f1=wa, uar=mean(tp[c] / true[c] for c in true),
                macro_f1=mean(2 * tp[c] / (true[c] + pred[c]) for c in classes),
                prediction_marginal_accuracy=sum(true[c] * pred[c] for c in classes) / n ** 2)


def decode_letter(letter, choices):
    if isinstance(letter, str) and len(letter) == 1 and letter in string.ascii_uppercase[:len(choices)]:
        return normal(choices[ord(letter) - 65])
    return None


def score_row(row, result):
    key, condition = row['id'], row['condition']
    choices = result['choices']
    if not isinstance(choices, list) or not 1 <= len(choices) <= 26:
        raise ValueError(f'Invalid candidate list on {key}')
    if result.get('condition') != condition or result.get('group', row['group']) != row['group']:
        raise ValueError(f'Manifest/result identity mismatch: {key}')
    raw_gold = row['explicit_emotion'] if condition in ('2A', '3A') else row['answer']
    gold = normal(raw_gold)
    true_label = decode_letter(result['expected_letter'], choices)
    if not gold or true_label != gold or gold != normal(row['label']):
        raise ValueError(f'Expected letter and scored manifest label disagree: {key}')
    predicted = decode_letter(result['predicted_letter'], choices)
    if 'prediction' in result and normal(result['prediction']) != (predicted or ''):
        raise ValueError(f'Predicted letter/label mismatch: {key}')
    return gold, predicted, result['expected_letter'] == result['predicted_letter'], 'neutral' in map(normal, choices)


def per_condition(records, n_manifest):
    all_pairs = [(g, p or INVALID) for g, p, _, _ in records]
    valid_pairs = [(g, p) for g, p, _, _ in records if p is not None]
    return dict(manifest_rows=n_manifest, recorded_rows=len(records), missing_rows=n_manifest - len(records),
                valid_letter_rows=len(valid_pairs), invalid_letter_rows=len(records) - len(valid_pairs),
                exact_letter_accuracy_recorded=sum(exact for _, _, exact, _ in records) / len(records) if records else None,
                normalized_accuracy_invalid_as_wrong=metrics(all_pairs)['wa'],
                author_valid_letter=metrics(valid_pairs))


def means(report, conditions):
    complete = all(report[c]['manifest_rows'] > 0 and report[c]['missing_rows'] == 0
                   and report[c]['valid_letter_rows'] > 0 for c in conditions)
    return dict(conditions=list(conditions), complete=complete, **{
        k: mean(report[c]['author_valid_letter'][k] for c in conditions) if complete else None for k in METRICS})


def neutral_diagnostic(rows):
    valid = [('neutral', p) for _, p, _, _ in rows if p]
    eligible = [('neutral', p) for _, p, _, offered in rows if p and offered]
    return dict(target='neutral', recorded=len(rows), neutral_offered=sum(s[3] for s in rows),
                valid_letter_rows=len(valid), valid_neutral_offered_rows=len(eligible),
                paper_target_accuracy_valid=metrics(valid)['wa'],
                paper_target_accuracy_all_recorded=sum(p == 'neutral' for _, p, _, _ in rows) / len(rows) if rows else None,
                paper_target_accuracy_if_neutral_offered=metrics(eligible)['wa'])


def check_heldout(path, manifests, scored):
    """Independently re-score the primary method's final, already-held-out predictions."""
    grouped = defaultdict(lambda: defaultdict(list)); seen = set()
    for r in read_jsonl(path):
        key = r['id']
        if key not in manifests or key not in scored or key in seen:
            raise ValueError(f'Unknown, missing or duplicate frozen record on method output {key}')
        seen.add(key)
        src = manifests[key]; condition = src['condition']; origin = r['origin']; gold, base, _, _ = scored[key]
        if r['condition'] != condition or normal(r['target']) != gold or normal(r['base']) != (base or INVALID):
            raise ValueError(f'Held-out method/frozen baseline mismatch: {key}')
        if origin == 'native_test' and src['split'] != 'test':
            raise ValueError(f'native_test refers to train data: {key}')
        if origin == 'type4_holdout' and (src['split'] != 'train' or condition != '4'):
            raise ValueError(f'Incorrect provenance on type4_holdout: {key}')
        for method in METHODS:
            grouped[(origin, condition)][method].append((gold, normal(r[method]) or INVALID))
    out = {f'{o}/{c}': {m: {'rows': len(p), **metrics(p)} for m, p in methods.items()}
           for (o, c), methods in grouped.items()}
    for name, pairs in [('native_six', [('native_test', c) for c in AUDIO[:-1]]),
                        ('mixed_seven', [('native_test', c) for c in AUDIO[:-1]] + [('type4_holdout', '4')])]:
        complete = all(k in grouped for k in pairs)
        out[name] = {'all_conditions_present': complete, 'wa': {m: mean(out[f'{o}/{c}'][m]['wa'] for o, c in pairs)
                     if complete else None for m in METHODS}}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path, help='Folder with train.jsonl/test.jsonl, or an expanded manifest JSONL')
    parser.add_argument('--results', required=True, nargs='+', type=Path, help='Frozen prediction JSONLs or one folder containing train.jsonl/test.jsonl')
    parser.add_argument('--split', nargs='+', choices=('train','test'), default=['train','test'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-partial', action='store_true', help='Write coverage diagnostics even when predictions are still arriving')
    parser.add_argument('--heldout', type=Path, help='Optional heldout_predictions.jsonl produced by methods.py')
    args = parser.parse_args()
    manifest = load_manifests(args.data, args.split)
    results = load_predictions(args.results, args.split)
    scored = {k: score_row(row, results[k][0]) for k, row in manifest.items() if k in results}
    if not scored:
        raise ValueError('No frozen predictions joined to the supplied manifests')
    if len(scored) != len(manifest) and not args.allow_partial:
        raise ValueError(f'{len(manifest)-len(scored)} manifest rows have no prediction. Use --allow-partial to describe incomplete coverage.')
    report = {'scale':'fractions', 'metric_definitions':{
        'author_valid_letter':'Normalized expected/predicted option from valid letters only. WA is correct divided by scored rows.',
        'uar':'Mean recall over classes present in the ground truth.', 'macro_f1':'Mean F1 over the union of true and predicted classes.',
        'prediction_marginal_accuracy':'Dot product of the true and prediction label distributions on scored rows.',
        'table2_mean':'Unweighted mean of the seven audio-containing condition scores; unavailable if any condition is missing.',
        'paper_neutral':'Re-scores the 1_text outputs using the paper’s fixed neutral target. Existing answer choices and prompts are preserved.'},
        'unjoined_result_ids':len(results.keys()-manifest.keys()), 'splits':{}}
    for split in args.split:
        groups = defaultdict(list); n = Counter()
        for k, row in manifest.items():
            if row['split'] == split:
                n[row['condition']] += 1
                if k in scored:
                    groups[row['condition']].append(scored[k])
        if not n:
            raise ValueError(f'No eligible records for {split}')
        per = {c: per_condition(groups[c], n[c]) for c in CONDITIONS}
        report['splits'][split] = {'per_condition':per,'table2_mean':means(per,AUDIO),
            'native_six_mean':means(per,AUDIO[:-1]),
            'condition_family_means':{name:means(per,cs) for name,cs in (
                ('Neutral-Text',AUDIO[0:2]),('Emotion-Matched',AUDIO[2:4]),
                ('Emotion-Mismatched',AUDIO[4:6]),('Paralinguistic',AUDIO[6:7]))},
            'paper_neutral_1_text':neutral_diagnostic(groups['1_text'])}
    if args.heldout:
        report['heldout_method_audit'] = check_heldout(args.heldout, manifest, scored)
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + '\n', encoding='utf-8')
    else:
        print(serialized)


if __name__ == '__main__':
    main()
