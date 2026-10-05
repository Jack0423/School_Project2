"""Evaluate grouping against independent human labels (pairwise and exact groups)."""
import argparse
import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from burst_metadata import DEFAULT_MAX_SECONDS, build_burst_check, paths_without_metadata
from photo_grouping import DEFAULT_THRESHOLD, group_photos, load_photos


def normalize(path):
    return str(Path(path).expanduser().resolve())


def read_labels(path):
    with open(path, encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'path', 'group_id'} <= set(reader.fieldnames or []):
            raise ValueError('CSV needs path,group_id columns')
        labels = {}
        for row in reader:
            source, group = (row.get('path') or '').strip(), (row.get('group_id') or '').strip()
            if not source or not group:
                raise ValueError('Every photo needs a nonempty path and human group_id')
            key = normalize(source)
            if key in labels:
                raise ValueError(f'Duplicate label: {source}')
            labels[key] = group
    if len(labels) < 2:
        raise ValueError('At least two labeled photos required')
    return labels


def grouping_metrics(truth, predicted):
    if truth.keys() != predicted.keys():
        raise ValueError('Labels and predictions must cover exactly the same photos')
    if len(truth) < 2:
        raise ValueError('At least two photos required')
    counts = Counter()
    examples = {'false_merge': [], 'false_split': []}
    for a, b in combinations(sorted(truth), 2):
        same_truth, same_pred = truth[a] == truth[b], predicted[a] == predicted[b]
        key = ('tp' if same_pred else 'fn') if same_truth else ('fp' if same_pred else 'tn')
        counts[key] += 1
        error = 'false_merge' if key == 'fp' else 'false_split' if key == 'fn' else None
        if error and len(examples[error]) < 50:
            examples[error].append([a, b])
    tp, fp, fn, tn = (counts[k] for k in ('tp', 'fp', 'fn', 'tn'))
    ratio = lambda n, d: n / d if d else None
    def clusters(labels):
        groups = {}
        for path, label in labels.items():
            groups.setdefault(label, set()).add(path)
        return {frozenset(group) for group in groups.values()}
    actual, inferred = clusters(truth), clusters(predicted)
    return {'photos': len(truth), 'pairs': tp+fp+fn+tn,
            'true_positive': tp, 'false_positive': fp, 'false_negative': fn, 'true_negative': tn,
            'pair_precision': ratio(tp, tp+fp), 'pair_recall': ratio(tp, tp+fn),
            'pair_f1': ratio(2*tp, 2*tp+fp+fn), 'pair_accuracy': ratio(tp+tn, tp+fp+fn+tn),
            'true_groups': len(actual), 'predicted_groups': len(inferred),
            'exact_groups': len(actual & inferred),
            'exact_group_recall': ratio(len(actual & inferred), len(actual)),
            'exact_group_precision': ratio(len(actual & inferred), len(inferred)),
            'error_examples_first_50': examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    template = sub.add_parser('template', help='Create blank CSV for independent annotation')
    template.add_argument('input', help='Batch JSONL')
    template.add_argument('--output', required=True)
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('input', help='Batch JSONL')
    evaluate.add_argument('labels', help='Human CSV: path,group_id')
    evaluate.add_argument('--metadata', help='ExifTool JSON; required for burst mode')
    evaluate.add_argument('--mode', choices=['burst', 'similarity'], default='burst')
    evaluate.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    evaluate.add_argument('--seconds', type=float, default=DEFAULT_MAX_SECONDS)
    evaluate.add_argument('--camera-match', choices=['serial', 'model', 'ignore'], default='model')
    evaluate.add_argument('--output', required=True)
    args = parser.parse_args()
    photos = load_photos(args.input)
    paths = [normalize(photo['path']) for photo in photos]
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate paths in batch')
    with open(args.input, encoding='utf-8') as handle:
        failed_count = sum(not json.loads(line).get('ok') for line in handle if line.strip())
    if not paths or failed_count:
        raise ValueError(f'Need a complete successful batch; failed={failed_count}, success={len(paths)}')
    if args.command == 'template':
        with open(args.output, 'x', encoding='utf-8-sig', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['path', 'group_id'])
            writer.writerows((path, '') for path in sorted(paths))
        return 0
    labels = read_labels(args.labels)
    if labels.keys() != set(paths):
        raise ValueError('Human labels must cover exactly every photo in this batch')
    can_pair, metadata_stats, missing = None, None, []
    if args.mode == 'burst':
        if not args.metadata:
            parser.error('--metadata is required for burst mode')
        can_pair, metadata_stats = build_burst_check(args.metadata, args.seconds, args.camera_match)
        missing = paths_without_metadata(paths, args.metadata)
    groups = group_photos(args.input, args.threshold, can_pair)
    predictions = {normalize(photo['path']): group['group_id']
                   for group in groups for photo in group['photos']}
    report = {'mode': args.mode, 'threshold': args.threshold,
              'max_span_seconds': args.seconds if args.mode == 'burst' else None,
              'camera_match': args.camera_match if args.mode == 'burst' else None,
              'metadata_stats': metadata_stats, 'missing_metadata_paths': missing,
              'input': str(Path(args.input).resolve()), 'labels': str(Path(args.labels).resolve()),
              'metrics': grouping_metrics(labels, predictions), 'groups': groups,
              'note': 'Human labels must be independent. Null metrics mean no denominator. '
                      'Accuracy can be inflated by unrelated pairs; report precision, recall and F1 together.'}
    with open(args.output, 'x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report['metrics'], ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
