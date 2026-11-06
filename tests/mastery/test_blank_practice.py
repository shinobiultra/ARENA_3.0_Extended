"""Inert actual-download contracts. No notebook execution or model imports."""
from pathlib import Path
import importlib.util
import json
import os
import sys

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'infrastructure/extension/syllabus.yaml').is_file())
BASE = Path(__file__).resolve().parents[3]
READER = Path(os.environ.get('ARENA_READER_SOURCE', ROOT / 'mastery_hub/read.py'))

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.fixture
def reader(monkeypatch):
    candidate = Path(__file__).resolve().parents[2] / 'infrastructure/extension/blank_practice.py'
    if candidate.exists():
        monkeypatch.setitem(sys.modules, 'infrastructure.extension.blank_practice', load(candidate, 'blank_practice_candidate'))
    return load(READER, 'reader_blank_candidate')

class UI:
    def __init__(self, row):
        self.row = row; self.sidebar = self; self.downloads = []; self.text = []
    def set_page_config(self, **kwargs): pass
    def title(self, text): self.text.append(text)
    def markdown(self, text, **kwargs): self.text.append(text)
    def info(self, text): self.text.append(text)
    def selectbox(self, label, options, **kwargs):
        return self.row['chapter'] if label == 'Chapter' else (self.row if label == 'Lesson' else 0)
    def download_button(self, label, data, file_name, **kwargs): self.downloads.append((label, data, file_name))


def available():
    return [r for r in load(ROOT / 'mastery_hub/read.py', 'source_inventory').inventory(ROOT) if r['available']]

ROWS = available()

@pytest.mark.parametrize('row', ROWS, ids=lambda r:r['number'])
@pytest.mark.parametrize('role,label', [('exercises','Download exercises'),('reference','Download worked reference')])
def test_all_configured_source_payloads_exact(reader, row, role, label):
    ui=UI(row); reader.render(ui, ROOT)
    matches=[(data,name) for lab,data,name in ui.downloads if lab==label]
    assert matches == [(row[role].read_bytes(),row[role].name)]

@pytest.mark.parametrize('row', ROWS, ids=lambda r:r['number'])
def test_all86_native_blank_download(reader, row):
    ui=UI(row); before=row['exercises'].read_bytes(); reader.render(ui, ROOT)
    matches=[(data,name) for label,data,name in ui.downloads if label=='Download blank Practice exercises']
    assert len(matches)==1, 'Missing additive blank Practice download'
    data,name=matches[0]; n=json.loads(data); source=json.loads(before)
    assert name.endswith('_Blank_Practice_exercises.ipynb')
    assert n['nbformat']==source['nbformat'] and n['nbformat_minor']==source['nbformat_minor']
    assert all(c['outputs']==[] and c['execution_count'] is None for c in n['cells'] if c['cell_type']=='code')
    assert len(n['cells'])==len(source['cells'])+1+(6 if row['number'] in ('0.6','1.6') else 0)
    note=''.join(n['cells'][0]['source']); assert all(x in note for x in ['PUBLIC Practice','beside','Hints and solutions'])
    assert row['exercises'].read_bytes()==before
    restored=reader.restore_blank_practice(data,before,row['exercises'].relative_to(ROOT).as_posix(),reader.blank_config_identity(ROOT))
    assert json.dumps(restored,sort_keys=True)==json.dumps(source,sort_keys=True)

@pytest.fixture
def projection(reader):
    return sys.modules['infrastructure.extension.blank_practice']

def encode(value): return (json.dumps(value,ensure_ascii=False)+'\n').encode()

def example(minor=5):
    cell = {'cell_type': 'code',
     'metadata': {'historical': {'value': 1, 'list': [True, None, 'expected answers']},
                  'execution': {'time': 'old'}},
     'source': ['print(1)\n'],
     'outputs': [{'output_type': 'error',
                  'ename': 'OldError',
                  'evalue': 'historic',
                  'traceback': ['old']}],
     'execution_count': 17}
    markdown={'cell_type':'markdown','metadata':{'tags':['help']},'source':'<details>expected answers</details>','attachments':{'picture.png':{'image/png':'YWJj'}}}
    if minor==5:cell['id']='original-code';markdown['id']='original-markdown'
    return {'nbformat':4,'nbformat_minor':minor,'metadata':{'historical':{'number':1,'nullable':None}},'cells':[markdown,cell]}

REL='chapter5_modern_architectures/exercises/part1_gemma_from_scratch/test_exercises.ipynb'
AUTH={'infrastructure/extension/syllabus.yaml':'a'*64,'infrastructure/core/config_extension.yaml':'b'*64}

@pytest.mark.parametrize('minor',[2,5])
def test_strict_inverse_preserves_metadata_attachments_source_ids_and_errors(projection,minor):
    n=example(minor); raw=encode(n); after=projection.project(raw,REL,AUTH)
    assert projection.restore(after,raw,REL,AUTH)==n
    assert after==projection.project(raw,REL,AUTH)
    result=json.loads(after)
    assert result['cells'][1]==n['cells'][0]
    assert result['cells'][2]['metadata']==n['cells'][1]['metadata']
    if minor==2:assert all('id' not in c for c in result['cells'])

@pytest.mark.parametrize('mutation',[
 lambda n:n['metadata']['historical'].__setitem__('number',True),
 lambda n:n['cells'][1].__setitem__('source',['different']),
 lambda n:n['cells'][2]['metadata'].__setitem__('historical',{}),
 lambda n:n['cells'][2].__setitem__('outputs',[{'output_type':'stream','name':'stdout','text':'invented'}]),
 lambda n:n['cells'][2].__setitem__('execution_count',1),
 lambda n:n['cells'][2].__setitem__('id','another'),
 lambda n:n['cells'][0]['source'].append('changed provenance'),
 lambda n:n['metadata']['arena_blank_practice'].__setitem__('source_sha256','f'*64),
 lambda n:n.__setitem__('unlisted_root_field',{}),
 lambda n:n['cells'].reverse(),
])
def test_inverse_rejects_all_unlisted_mutations(projection,mutation):
    raw=encode(example()); n=json.loads(projection.project(raw,REL,AUTH));mutation(n)
    with pytest.raises(ValueError):projection.restore(encode(n),raw,REL,AUTH)

@pytest.mark.parametrize('mutation',[
 lambda n:n['metadata'].__setitem__('arena_blank_practice',{}),
 lambda n:n.__setitem__('metadata',[]),
 lambda n:n.__setitem__('nbformat',True),
 lambda n:n.__setitem__('nbformat_minor',6),
 lambda n:n['cells'][1].__setitem__('execution_count',True),
 lambda n:n['cells'][1].__setitem__('outputs',{}),
 lambda n:n['cells'][1].__setitem__('outputs',[{'output_type':'unknown'}]),
 lambda n:n['cells'][1].__setitem__('source',23),
 lambda n:n['cells'][1].__setitem__('metadata',None),
 lambda n:n['cells'][1].__setitem__('id',n['cells'][0]['id']),
 lambda n:n['cells'][1].__setitem__('id','invalid id'),
 lambda n:n['cells'][1].pop('id'),
])
def test_invalid_source_schema_and_collisions_fail_closed(projection,mutation):
    n=example();mutation(n)
    with pytest.raises(ValueError):projection.project(encode(n),REL,AUTH)

@pytest.mark.parametrize('raw',[b'{} garbage',b'\xff',b'[]',b'{"nbformat":4,"nbformat":4}',b'{"x":NaN}',b'{"x":Infinity}'])
def test_malformed_duplicate_nonfinite_json_rejected(projection,raw):
    with pytest.raises(ValueError):projection.project(raw,REL,AUTH)

@pytest.mark.parametrize('relative',['/etc/passwd','../x_exercises.ipynb','a/../x_exercises.ipynb','a//x_exercises.ipynb','a\\x_exercises.ipynb','x_solutions.ipynb',''])
def test_pure_projection_rejects_nonsource_path_identity(projection,relative):
    with pytest.raises(ValueError):projection.project(encode(example()),relative,AUTH)

@pytest.mark.parametrize('section',['/etc/passwd','../5.1','5.1/../../x','5.99','999.1','5.1.ipynb',''])
def test_only_configured_available_sections(reader,section):
    with pytest.raises(ValueError):reader.blank_practice_download(ROOT,section)


def test_inverse_rejects_stale_source_and_configuration(projection):
    raw=encode(example()); after=projection.project(raw,REL,AUTH)
    changed=example();changed['metadata']['new']='source changed'
    with pytest.raises(ValueError):projection.restore(after,encode(changed),REL,AUTH)
    with pytest.raises(ValueError):projection.restore(after,raw,REL,{**AUTH,'infrastructure/extension/syllabus.yaml':'c'*64})


def test_generated_id_collision_rejected(projection,monkeypatch):
    n=example(); n['cells'][0]['id']='blank-practice-'+('a'*24)
    original=projection._provenance
    def collide(*args):return {**original(*args),'source_sha256':'a'*64}
    monkeypatch.setattr(projection,'_provenance',collide)
    with pytest.raises(ValueError,match='ID collision'):projection.project(encode(n),REL,AUTH)


def test_configured_symlink_rejected_before_read(projection,tmp_path):
    f=tmp_path/'outside';f.write_bytes(b'private sentinel')
    link=tmp_path/'infrastructure';link.symlink_to(f)
    with pytest.raises(ValueError,match='symlink'):projection.config_identity(tmp_path)


def test_real_inputs_stay_read_only_and_round_trip(reader,monkeypatch):
    original_open=Path.open
    def readonly(path,mode='r',*args,**kwargs):
        assert not any(x in mode for x in ('w','a','x','+')), 'Projection attempted a file write'
        return original_open(path,mode,*args,**kwargs)
    monkeypatch.setattr(Path,'open',readonly)
    assert ROWS
    for row in ROWS:
        source=row['exercises'].read_bytes()
        raw,_=reader.blank_practice_download(ROOT,row['number'])
        reader.restore_blank_practice(raw,source,row['exercises'].relative_to(ROOT).as_posix(),reader.blank_config_identity(ROOT))


@pytest.mark.parametrize('section,appended',[('0.6',6),('1.6',6),('5.1',0),('7.1',0),('12.3',0)])
def test_foundation_downloads_append_live_signature_cells_and_others_do_not(reader,section,appended):
    row=next(r for r in ROWS if r['number']==section)
    source=json.loads(row['exercises'].read_bytes())
    data,_=reader.blank_practice_download(ROOT,section);n=json.loads(data)
    assert len(n['cells'])==len(source['cells'])+1+appended
    provider=load(ROOT/'mastery_hub/practice_addons/foundation_live_signatures/lesson.py','foundation_provider')
    if appended:
        assert [c['source'] for c in n['cells'][-appended:]]==[c['source'] for c in provider.additions(section)]
    assert all(c['outputs']==[] and c['execution_count'] is None for c in n['cells'] if c['cell_type']=='code')


def test_source_symlink_never_read_and_no_historical_fallback(reader,tmp_path,monkeypatch):
    import shutil
    row=next(r for r in ROWS if r['number']=='5.1')
    for rel in ('infrastructure/extension/syllabus.yaml','infrastructure/core/config_extension.yaml'):
        dest=tmp_path/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    master=tmp_path/'infrastructure/chapters'/row['chapter']/'master_5_1.py';master.parent.mkdir(parents=True);master.write_text('# not imported')
    for key in ('page','reference'):
        f=tmp_path/row[key].relative_to(ROOT);f.parent.mkdir(parents=True,exist_ok=True);f.write_text('{}')
    source=tmp_path/row['exercises'].relative_to(ROOT);source.symlink_to(tmp_path/'private.ipynb')
    (tmp_path/'private.ipynb').write_bytes(encode(example()))
    original=Path.read_bytes
    def guarded(path):
        assert path!=source and path!=tmp_path/'private.ipynb','A symlink payload was read'
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',guarded)
    with pytest.raises(ValueError,match='symlink'):reader.blank_practice_download(tmp_path,'5.1')


def test_symlink_ancestor_of_repository_rejected(projection,tmp_path):
    actual=tmp_path/'actual/repo'
    for rel in AUTH:
        path=actual/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('authority bytes')
    (tmp_path/'alias').symlink_to(tmp_path/'actual',target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):
        projection.config_identity(tmp_path/'alias/repo')
