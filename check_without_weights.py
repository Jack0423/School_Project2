"""Run an explicit offline test suite without loading trained model weights.

Qt integration uses fake inference with real QThread/signals/SQLite. Model shape
checks use untrained weights only to verify interfaces, never to score photos.
"""
import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import unittest

TESTS = [
    'tests.test_batch_comparison', 'tests.test_grouping_validation',
    'tests.test_qt_background', 'tests.test_photo_grouping',
    'tests.test_model', 'tests.test_metrics', 'tests.test_datasets',
    'tests.test_transforms.TestEvalTransform',
    'tests.test_transforms.TestTrainTransforms', 'tests.test_startup_guards',
    'tests.test_shooting_info',
    'tests.test_console_encoding.TestSourcesAreCp950Encodable',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='New report directory; defaults to timestamped reports/no_weights directory')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    missing = [name for name in ['numpy','torch','torchvision','PIL','cv2','PyQt6']
               if importlib.util.find_spec(name) is None]
    if missing:
        parser.error('Missing test dependencies: '+', '.join(missing)+
                     '. Use the project .venv Python (see docs/no_weights_quickstart.md).')
    output = Path(args.output).resolve() if args.output else (
        root/'reports'/'no_weights'/datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    output.mkdir(parents=True, exist_ok=False)
    suite = unittest.defaultTestLoader.loadTestsFromNames(TESTS)
    with (output/'tests.log').open('w', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    summary = {
        'scope': 'Selected weight-independent tests; not the full project suite',
        'python': platform.python_version(), 'executable': sys.executable,
        'tests_run': result.testsRun, 'skipped': len(result.skipped),
        'failures': len(result.failures), 'errors': len(result.errors),
        'successful': result.wasSuccessful(),
        'skipped_tests': [{'test':str(test),'reason':reason} for test,reason in result.skipped],
        'real_photo_scores_measured': False,
        'real_photo_performance_measured': False,
        'real_grouping_accuracy_measured': False,
    }
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Tests: {result.testsRun}; skipped: {len(result.skipped)}; "
          f"failures: {len(result.failures)}; errors: {len(result.errors)}")
    print(f'Report: {output}')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
