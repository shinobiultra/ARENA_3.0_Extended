"""Source-only identity/schema tests: fake unregistered fixtures vs genuine custody."""
import builtins
import hashlib
import importlib.util
from pathlib import Path
import shutil

import pytest
from infrastructure.extension import build

_fixture_path=Path(__file__).with_name('test_extension_master_notebook_schema.py')
_spec=importlib.util.spec_from_file_location('_normal_generate_schema_fixtures',_fixture_path)
_fixture=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_fixture)


def snapshot(directory):
    return {path.relative_to(directory).as_posix(): ('directory' if path.is_dir() else (hashlib.sha256(path.read_bytes()).hexdigest(),path.stat().st_size,path.stat().st_mode&0o777)) for path in directory.rglob('*')}


@pytest.mark.parametrize('section',['0.6','1.6'])
def test_fabricated_registered_identity_still_rejects_before_any_write(tmp_path,monkeypatch,section):
    source,config=_fixture.isolated_generation(tmp_path,monkeypatch,section)
    before=snapshot(tmp_path)
    with pytest.raises(ValueError,match='Wrong registered'):
        build.generate(section,config)
    assert snapshot(tmp_path)==before
    assert source.read_text()==_fixture.small_source(section)


def actual_copy(tmp_path,section):
    config=build.load_config();mapping=config['conversion_map'][section];chapter=int(section.split('.')[0])
    genuine=build.find_master(section)
    directory=tmp_path/'copied_masters';directory.mkdir()
    copied=directory/genuine.name;shutil.copy2(genuine,copied)
    target=copied.with_suffix('.ipynb');target.write_text('private source-only master sentinel')
    generated=build.ExtensionMasterFile(master_path=target,chapter_dir=tmp_path/config['chapter_names'][chapter],
        chapter_name_long=config['chapter_names_long'][chapter],exercise_dir_name=mapping['exercise_dir'],streamlit_page_name=mapping['streamlit_page'])
    return generated,genuine


def forbid_methods(monkeypatch):
    original=builtins.__import__
    def checked(name,*args,**kwargs):
        if name.split('.')[0] in {'torch','numpy','torchvision','safetensors','nbclient','jupyter_client','ipykernel','arena_ext','matplotlib','pandas','jax','scipy'}:
            raise AssertionError('Source schema/custody test imported a runtime: '+name)
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',checked)


@pytest.mark.parametrize('section',['0.6','1.6'])
def test_genuine_registered_copy_preserves_schema_and_exact_canonical_custody(tmp_path,monkeypatch,section):
    forbid_methods(monkeypatch)
    generated,genuine=actual_copy(tmp_path,section)
    canonical_root=build.ROOT
    mapping=build.load_config()['conversion_map'][section]
    canonical_solutions=canonical_root/generated.chapter_dir.name/'exercises'/mapping['exercise_dir']/(generated.colab_name+'_solutions.ipynb')
    physical_before=canonical_solutions.read_bytes();source_before=genuine.read_bytes()
    generated.master_py_to_ipynb(overwrite=True)
    first=generated.master_path.read_bytes()
    assert not _fixture.errors(_fixture.json.loads(first))
    generated.exercises_dir.mkdir(parents=True);generated.instructions_dir.mkdir(parents=True)
    generated.generate_files(overwrite=True,verbose=False)
    assert generated.source_preservation_applied is True
    assert (generated.exercises_dir/(generated.colab_name+'_solutions.ipynb')).read_bytes()==physical_before
    generated.master_py_to_ipynb(overwrite=True)
    assert generated.master_path.read_bytes()==first
    assert genuine.read_bytes()==source_before and canonical_solutions.read_bytes()==physical_before
    assert build.ROOT==canonical_root


@pytest.mark.parametrize('section',['0.6','1.6'])
def test_fabricated_source_cannot_claim_genuine_registered_recipe_identity(tmp_path,monkeypatch,section):
    forbid_methods(monkeypatch)
    generated,genuine=actual_copy(tmp_path,section)
    generated.master_py_path.write_text(_fixture.small_source(section))
    generated.master_py_to_ipynb(overwrite=True)
    generated.exercises_dir.mkdir(parents=True);generated.instructions_dir.mkdir(parents=True)
    before=snapshot(generated.chapter_dir)
    with pytest.raises(ValueError,match='actual master bytes changed'):
        generated.generate_files(overwrite=True,verbose=False)
    assert snapshot(generated.chapter_dir)==before
