import tempfile
import unittest
from pathlib import Path
from validate_grouping import grouping_metrics, read_labels


class GroupingValidationTests(unittest.TestCase):
    def test_exact_partition_ignores_group_names(self):
        result = grouping_metrics(dict(a='1', b='1', c='2'), dict(a=7, b=7, c=9))
        self.assertEqual(result['pair_f1'], 1)
        self.assertEqual(result['exact_group_recall'], 1)

    def test_false_merge_and_false_split(self):
        result = grouping_metrics(dict(a=1, b=1, c=2), dict(a=1, b=2, c=2))
        self.assertEqual([result[k] for k in ('true_positive','false_positive','false_negative','true_negative')], [0,1,1,1])
        self.assertAlmostEqual(result['pair_accuracy'], 1/3)
        self.assertEqual(result['pair_f1'], 0)
        self.assertEqual(result['exact_groups'], 0)

    def test_all_singletons_do_not_claim_perfect_f1(self):
        result = grouping_metrics(dict(a=1,b=2), dict(a=1,b=2))
        self.assertIsNone(result['pair_f1'])
        self.assertEqual(result['pair_accuracy'], 1)
        with self.assertRaises(ValueError):
            grouping_metrics(dict(a=1,b=2), dict(a=1))

    def test_burst_validation_cli(self):
        import contextlib
        import csv
        import io
        import json
        from unittest.mock import patch
        import validate_grouping
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, labels, metadata, report = [root/name for name in
                                                ['batch.jsonl','labels.csv','metadata.json','report.json']]
            paths = [str(root/f'{i}.jpg') for i in range(3)]
            rows = []
            for i, path in enumerate(paths):
                vector = [0.0]*1280
                vector[0 if i < 2 else 1] = 1.0
                rows.append(dict(path=path,ok=True,feature_vector=vector,
                                 aesthetic_weight=.6,technical_weight=.4,overall_score=80-i))
            source.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with patch('sys.argv',['validate_grouping.py','template',str(source),'--output',str(labels)]):
                self.assertEqual(validate_grouping.main(),0)
            with self.assertRaises(ValueError):
                read_labels(labels)
            with labels.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(['path','group_id'])
                writer.writerows((p,1 if i<2 else 2) for i,p in enumerate(paths))
            metadata.write_text(json.dumps([dict(SourceFile=p,DateTimeOriginal=f'2026:10:05 12:00:0{i}',
                                                Make='Test',Model='Test') for i,p in enumerate(paths)]))
            with patch('sys.argv',['validate_grouping.py','evaluate',str(source),str(labels),
                                  '--metadata',str(metadata),'--output',str(report)]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(validate_grouping.main(),0)
            result = json.loads(report.read_text(encoding='utf-8'))
            self.assertEqual(result['metrics']['pair_f1'],1)
            self.assertEqual(result['threshold'],.85)
            self.assertEqual(result['max_span_seconds'],5)

    def test_reject_blank_and_duplicate_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'labels.csv'
            for text in ['path,group_id\na.jpg,\nb.jpg,2\n',
                         'path,group_id\na.jpg,1\na.jpg,2\n']:
                path.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    read_labels(path)


if __name__ == '__main__':
    unittest.main()
