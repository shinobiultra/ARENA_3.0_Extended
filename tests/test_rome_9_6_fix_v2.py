"""Actual inert imports and renderer checks. No taught numerical function is called."""
from pathlib import Path
import ast,base64,builtins,importlib.util,json,os,re,symtable,sys,types,unittest
from unittest.mock import patch
ROOT=Path(os.environ.get('ARENA_MIGRATION_ROOT',Path(__file__).resolve().parents[1]))
LEGACY=Path(os.environ.get('ARENA_LEGACY_ROOT',ROOT))
NEW=ROOT/'chapter9_alignment_interpretability/exercises/part6_rome_memit'
OLD=LEGACY/'chapter9_alignment_interpretability/exercises/part5_rome_model_editing'
PAGE=ROOT/'chapter9_alignment_interpretability/instructions/pages/06_[9.6]_Model_Editing_with_ROME_and_MEMIT.md'
PREFIX='chapter9_alignment_interpretability.exercises'

def dummy(name,**attrs):
    m=types.ModuleType(name);m.__dict__.update(attrs);return m

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

class RomeFix(unittest.TestCase):
    def test_actual_inert_import_exact_original_default_and_all_globals(self):
        events=[]
        def called(name):
            def fn(*args,**kwargs):events.append(name);return object()
            return fn
        torch=dummy('torch',Tensor=type('Tensor',(),{}),float32=object(),manual_seed=called('manual_seed'),device=called('device'),__version__='inert-v2')
        names=['chapter9_alignment_interpretability',PREFIX,PREFIX+'.part5_rome_model_editing',PREFIX+'.part6_rome_memit']
        stubs={n:dummy(n,__path__=[]) for n in names}
        stubs.update({'torch':torch,'pandas':dummy('pandas'),'matplotlib':dummy('matplotlib'),'matplotlib.pyplot':dummy('matplotlib.pyplot'),'IPython':dummy('IPython'),'IPython.display':dummy('IPython.display',display=called('display'))})
        with patch.dict(sys.modules,stubs):
            # Load the unchanged actual legacy source, not a fake reference implementation.
            ref=load_module(PREFIX+'.part5_rome_model_editing.solutions',OLD/'solutions.py')
            self.assertEqual(events,[])
            setattr(stubs[PREFIX+'.part5_rome_model_editing'],'solutions',ref)
            if (NEW/'rome_metadata.py').exists():
                metadata=load_module(PREFIX+'.part6_rome_memit.rome_metadata',NEW/'rome_metadata.py')
                self.assertIs(metadata.PromptRecord,ref.PromptRecord)
                self.assertIs(metadata.HELDOUT_PARAPHRASE_RECORDS,ref.HELDOUT_PARAPHRASE_RECORDS)
            tests=load_module(PREFIX+'.part6_rome_memit.tests',NEW/'tests.py')
            setattr(stubs[PREFIX+'.part6_rome_memit'],'tests',tests)
            solution=load_module(PREFIX+'.part6_rome_memit.solutions',NEW/'solutions.py')
            self.assertEqual(events,[])
            self.assertIs(solution.evaluate_weight_edit.__kwdefaults__['paraphrase_records'],ref.HELDOUT_PARAPHRASE_RECORDS)
            self.assertEqual(len(ref.HELDOUT_PARAPHRASE_RECORDS),20)
            self.assertTrue(all(type(row) is ref.PromptRecord for row in solution.evaluate_weight_edit.__kwdefaults__['paraphrase_records']))
            self.assertFalse(hasattr(solution,'device'));self.assertFalse(hasattr(solution,'model'))
            solution.initialize_notebook_plumbing()
            self.assertEqual(events,['manual_seed','device'])
            self.assertIs(solution.evaluate_weight_edit.__kwdefaults__['paraphrase_records'],solution.HELDOUT_PARAPHRASE_RECORDS)
            code=(NEW/'solutions.py').read_text();tree=ast.parse(code)
            bundle=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='learner_functions' for t in n.targets))
            self.assertEqual(len(bundle.keywords),8)
            for kw in bundle.keywords:
                fn=getattr(solution,kw.value.id);self.assertTrue(callable(fn));self.assertIsNot(fn,getattr(ref,kw.value.id))
            # Full global dependency scan, with only the explicit playground experiment state absent.
            missing={}
            def visit(t):
                if t.get_type()!='module':
                    absent=sorted(s.get_name() for s in t.get_symbols() if s.is_referenced() and s.is_global() and s.get_name() not in solution.__dict__ and not hasattr(builtins,s.get_name()))
                    if absent:missing[t.get_name()]=absent
                for child in t.get_children():visit(child)
            visit(symtable.symtable(code,str(NEW/'solutions.py'),'exec'))
            self.assertEqual(missing,{'score_fixed_louvre_edit':['model','result','tokenizer']})
            # Inspect every default/decorator/class-eager expression against actual available globals.
            for n in tree.body:
                if not isinstance(n,(ast.FunctionDef,ast.ClassDef)):continue
                values=[*n.decorator_list]
                if isinstance(n,ast.FunctionDef):values += [*n.args.defaults,*[x for x in n.args.kw_defaults if x is not None]]
                else:values += [*n.bases,*[x.value for x in n.keywords]]
                for value in values:
                    for x in ast.walk(value):
                        if isinstance(x,ast.Name):self.assertTrue(x.id in solution.__dict__ or hasattr(builtins,x.id),(n.name,x.id))

    def test_actual_renderer_and_standard_shell_embed_both_png_bytes(self):
        sys.path.insert(0,str(LEGACY))
        from infrastructure.extension.render_media import inline_page_media
        rendered=inline_page_media(PAGE.read_text(),PAGE)
        pngs=re.findall(r'src=["\']data:image/png;base64,([^"\']+)',rendered)
        expected=[(NEW/'media'/n).read_bytes() for n in ['rome_model_editing_signature.png','rome_rank_one_toy_oracle.png']]
        self.assertEqual(len(pngs),2)
        self.assertEqual([base64.b64decode(p) for p in pngs],expected)
        # Execute the actual standard page shell with inert UI surfaces; no browser/network.
        class Sidebar:
            def __enter__(self):return self
            def __exit__(self,*args):return False
        emitted=[];session={}
        st=dummy('streamlit',session_state=session,sidebar=Sidebar(),info=lambda *a,**k:None,markdown=lambda text,**k:emitted.append(text))
        sac=dummy('streamlit_antd_components',StepsItem=lambda **kw:kw,steps=lambda *a,**kw:0)
        chat=dummy('st_chat',display_chat_interface=lambda *a:None,display_content_with_exercise_chats=lambda text,*a:emitted.append(text),initialize_chat=lambda:None)
        dependencies=dummy('st_dependencies',generate_toc=lambda *a:'toc',styling=lambda *a:None)
        with patch.dict(sys.modules,{'streamlit':st,'streamlit_antd_components':sac,'st_chat':chat,'st_dependencies':dependencies}):
            ns={'__name__':'rome_page_fix','__file__':str(PAGE.with_suffix('.py'))}
            exec(compile(PAGE.with_suffix('.py').read_text(),str(PAGE.with_suffix('.py')),'exec'),ns)
            if 'show' in ns:ns['show']()
        combined='\n'.join(session.get('content',emitted))
        actual=re.findall(r'src=["\']data:image/png;base64,([^"\']+)',combined)
        self.assertEqual([base64.b64decode(p) for p in actual],expected)
        self.assertIn('inline_page_media(',PAGE.with_suffix('.py').read_text())

if __name__=='__main__':unittest.main()
