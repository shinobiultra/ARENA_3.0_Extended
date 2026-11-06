"""Fixture-only checks: no model imports, loads, training or downloads."""
import ast
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

ROOT=Path(__file__).resolve().parents[1]
SECTION=ROOT/'chapter12_vlm_interpretability/exercises/part1_clip_siglip'


def loader(section=SECTION):
    tree=ast.parse((SECTION/'utils.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_cifar10_sample')
    namespace={'np':np,'Image':Image,'SECTION_DIR':section}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'utils.py','exec'),namespace)
    return namespace['load_cifar10_sample']


def test_png_fixture_arrays_match_preserved_original_byte_for_byte():
    images,labels=loader()()
    before=json.loads((ROOT/'docs/mastery/evidence/cifar_fixture_20261019/before.json').read_text())
    with np.load(before['cifar10_sample.npz']['absolute_path'],allow_pickle=False) as original:
        for result,name in ((images,'images'),(labels,'labels')):
            assert result.dtype==original[name].dtype
            assert result.shape==original[name].shape
            assert result.tobytes()==original[name].tobytes()


@pytest.mark.parametrize('change',['order','label','pixel_hash'])
def test_loader_rejects_manifest_corruption(tmp_path,change):
    fixture=tmp_path/'fixtures/cifar10_sample';fixture.mkdir(parents=True)
    source=SECTION/'fixtures/cifar10_sample'
    manifest=json.loads((source/'manifest.json').read_text())
    if change=='order':manifest['records'][0]['fixture_index']=1
    elif change=='label':manifest['records'][0]['label']=10
    else:manifest['records'][0]['pixels_sha256']='0'*64
    (fixture/'manifest.json').write_text(json.dumps(manifest))
    (fixture/'000.png').write_bytes((source/'000.png').read_bytes())
    with pytest.raises(ValueError):loader(tmp_path)()
