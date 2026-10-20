"""Shared command-line paths and run records; no dataset access at import time."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path


def output_directory(value, fresh=False):
    """Keep runtime outputs out of saved evidence; optionally require a fresh run."""
    output = Path(value).resolve()
    saved = Path(__file__).resolve().parents[1] / 'results'
    if output == saved or saved in output.parents:
        raise ValueError('Saved results must not be overwritten; use a directory under build/')
    if fresh and output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output directory must be empty for a new run')
    output.mkdir(parents=True, exist_ok=True)
    return output


def arguments(description, cohort=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    if cohort:
        parser.add_argument('--cohort', type=Path, required=True)
    args = parser.parse_args()
    args.data_root = args.data_root.resolve(strict=True)
    try:
        args.output_dir = output_directory(args.output_dir)
    except ValueError as error:
        parser.error(str(error))
    return args


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def record_run(output, name, inputs=(), **details):
    """Write a completion record. Names/hashes are portable; private paths stay out."""
    record = {'analysis': name, 'status': 'completed', 'python': platform.python_version(),
              'packages': {p: importlib.metadata.version(p) for p in
                           ('numpy', 'pandas', 'scikit-learn', 'pyarrow')},
              'inputs': [{'name': Path(p).name, 'sha256': fingerprint(p)} for p in inputs],
              **details}
    (Path(output) / f'{name}.run.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')


def feature_counts(features, labels, corpus):
    """Describe each candidate's finite cohort and class-valid default CV folds."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    records = []
    for field, values in features.items():
        finite = np.isfinite(values)
        y = labels[finite]
        eligible = len(y) >= 50 and len(np.unique(values[finite])) >= 2 and len(np.unique(y)) == 2
        folds = 0
        if eligible:
            folds = sum(len(np.unique(y[tr])) == 2 and len(np.unique(y[te])) == 2
                        for tr, te in StratifiedKFold(5).split(values[finite], y))
        records.append(dict(corpus=corpus, field=field, input_accounts=len(labels),
                            finite_accounts=int(finite.sum()), finite_bots=int(y.sum()),
                            finite_humans=int((y == 0).sum()), candidate_eligible=eligible,
                            class_valid_folds=folds))
    return records
