"""Final archives must not include raw data or unvalidated output artifacts."""
import argparse
import importlib.util
import json
from pathlib import Path
import zipfile
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/package_final_submission.py'
spec=importlib.util.spec_from_file_location('package_final_submission',path)
package=importlib.util.module_from_spec(spec); spec.loader.exec_module(package)


def test_assets_reject_raw_dataset_and_parent_escape(tmp_path):
    for name in ['dataset/test/test_source1.tsv','../private-key','/tmp/private-key']:
        with pytest.raises(ValueError): package.relative_file(tmp_path,name)


def archive_inputs(tmp_path):
    stage=tmp_path/'stage'; core=stage/package.CORE; core.mkdir(parents=True)
    source=core/'src/example.py'; source.parent.mkdir(); source.write_text('value = 1\n')
    document=stage/'Documentation_template.md'; document.write_text('Method\n')
    marker={'selected_freeze_sha256':'frozen', 'files':{
        source.relative_to(stage).as_posix():package.digest(source),
        document.relative_to(stage).as_posix():package.digest(document)}}
    package.write_json(core/'package_manifest.json',marker)
    submission=tmp_path/'submission'; submission.mkdir()
    for n in package.OUTPUT_NAMES: (submission/n).write_text('header\n')
    report={'status':'complete','validation':{'strict':'PASS','official':'PASS'},
        'candidate_frozen_sha256':'frozen',
        'files_sha256':{n:package.digest(submission/n) for n in package.OUTPUT_NAMES}}
    package.write_json(submission/'submission_report.json',report)
    verification=tmp_path/'verification.json'
    package.write_json(verification,{'status':'packaged_actual_subset_passed',
        'package_manifest_sha256':package.digest(core/'package_manifest.json')})
    return argparse.Namespace(stage=stage,submission=submission,verification=verification,output=tmp_path/'final.zip')


def test_archive_uses_only_sealed_whitelist_and_exact_output_paths(tmp_path):
    a=archive_inputs(tmp_path)
    unwanted=a.stage/'dataset/raw.tsv'; unwanted.parent.mkdir(); unwanted.write_text('raw records')
    unsealed=a.stage/'code/private-key'; unsealed.write_text('secret')
    package.archive(a)
    with zipfile.ZipFile(a.output) as archive:
        names=set(archive.namelist())
    assert 'output/matching_results.tsv' in names
    assert 'output/candidate_pairs.tsv' in names
    assert 'Documentation_template.md' in names
    assert not any(n.startswith('dataset/') or n.endswith('private-key') for n in names)
    assert {n for n in names if n.startswith('output/')} == {'output/matching_results.tsv','output/candidate_pairs.tsv'}


def test_archive_rejects_source_outside_required_src_directory(tmp_path):
    a=archive_inputs(tmp_path)
    outside=a.stage/package.CORE/'run_final_submission.py'; outside.write_text('pass\n')
    marker_path=a.stage/package.CORE/'package_manifest.json'
    marker=json.loads(marker_path.read_text())
    marker['files'][outside.relative_to(a.stage).as_posix()]=package.digest(outside)
    package.write_json(marker_path,marker)
    package.write_json(a.verification,{'status':'packaged_actual_subset_passed',
        'package_manifest_sha256':package.digest(marker_path)})
    with pytest.raises(ValueError,match='All source must be under'):
        package.archive(a)


@pytest.mark.parametrize('change',['unvalidated','wrong_freeze','changed_output','wrong_verification'])
def test_archive_rejects_invalid_or_changed_release_artifacts(tmp_path,change):
    a=archive_inputs(tmp_path)
    rp=a.submission/'submission_report.json'; report=json.loads(rp.read_text())
    if change=='unvalidated': report['validation']['official']='FAIL'
    if change=='wrong_freeze': report['candidate_frozen_sha256']='other'
    if change in ('unvalidated','wrong_freeze'): package.write_json(rp,report)
    if change=='changed_output': (a.submission/'matching_results.tsv').write_text('changed\n')
    if change=='wrong_verification': package.write_json(a.verification,{'status':'packaged_actual_subset_passed','package_manifest_sha256':'other'})
    with pytest.raises(ValueError): package.archive(a)
    assert not a.output.exists()
