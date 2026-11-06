"""Lossless packaging tests; no model/tensor deserialization."""
import gzip
import io
import tarfile
from pathlib import Path

import pytest
from scripts.evidence_bundles import file_record, resolve_historical_archive, validate_review_bundle
from scripts.package_source_checkpoint import package_checkpoint


def archive(path, members):
    with tarfile.open(path, 'w:gz') as handle:
        for name, data in members:
            info=tarfile.TarInfo(name); info.size=len(data)
            handle.addfile(info,io.BytesIO(data))
    return path


def test_preserves_nested_compressed_and_raw_bytes_with_complete_partition(tmp_path):
    members=[('failed.json',b'{"status":"FAIL"}'),('nested.tar.gz',b'nested exact bytes'),
             ('lesson.md.gz',gzip.compress(b'original text')),('weights.pt',b'raw bytes')]
    source=archive(tmp_path/'review.tar.gz',members)
    original=tmp_path/'original.tar.gz';original.write_bytes(source.read_bytes())
    before=file_record(original)
    contract=package_checkpoint(source,original,'docs/checkpoint.tar.gz')
    assert file_record(original)==before
    assert set(contract['members'])=={'failed.json'}
    assert set(contract['external_payloads'])=={name for name,_ in members}-{'failed.json'}
    assert validate_review_bundle(source)==contract
    recovered=resolve_historical_archive(source,before['sha256'])
    with tarfile.open(recovered) as handle:
        assert [(m.name,handle.extractfile(m).read()) for m in handle]==members


@pytest.mark.parametrize('name,data',[('bad.json',b'not json'),('bad.png',b'not image'),('../bad.txt',b'x')])
def test_invalid_review_data_is_not_silently_externalized(tmp_path,name,data):
    source=archive(tmp_path/'review.tar.gz',[(name,data)])
    original=tmp_path/'original.tar.gz';original.write_bytes(source.read_bytes())
    before=file_record(source)
    with pytest.raises(ValueError): package_checkpoint(source,original,'docs/checkpoint.tar.gz')
    assert file_record(source)==before
