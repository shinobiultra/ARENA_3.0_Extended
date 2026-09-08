import importlib, json, pathlib, time, traceback, os
out = pathlib.Path(__file__).parent
results=[]
for name in [
    'numpy',
    'scipy',
    'pandas',
    'sklearn',
    'torch',
    'torchvision',
    'numba',
    'shap',
    'shapiq',
    'captum',
    'transformers',
    'transformer_lens',
    'diffusers',
    'peft',
    'sentence_transformers',
    'sae_lens',
    'sae_vis',
    'circuitsvis',
    'nnsight',
    'streamlit',
    'plotly',
    'matplotlib',
    'nbformat',
    'IPython',
    'anthropic',
    'openai',
    'inspect_ai',
    'instructor',
]:
    started=time.monotonic()
    try:
        mod=importlib.import_module(name)
        results.append({'module':name,'status':'PASS','version':str(getattr(mod,'__version__','unavailable')),'seconds':time.monotonic()-started})
    except Exception:
        results.append({'module':name,'status':'FAIL','error':traceback.format_exc(),'seconds':time.monotonic()-started})
    print(name,results[-1]['status'],flush=True)
(out/'package_imports.json').write_text(json.dumps(results,indent=2)+'\n')
import torch
from transformers import GPT2Config,GPT2LMHeadModel
from peft import LoraConfig,get_peft_model
model=GPT2LMHeadModel(GPT2Config(vocab_size=32,n_positions=16,n_ctx=16,n_embd=16,n_layer=1,n_head=2,bos_token_id=0,eos_token_id=1))
model=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,target_modules=['c_attn'],task_type='CAUSAL_LM'))
ids=torch.tensor([[2,3,4,5]])
loss=model(input_ids=ids,labels=ids).loss
loss.backward()
assert torch.isfinite(loss)
assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
print('Tiny GPT-2 + PEFT forward/backward PASS',flush=True)
from diffusers import UNet2DModel,DDPMScheduler
unet=UNet2DModel(sample_size=8,in_channels=1,out_channels=1,layers_per_block=1,block_out_channels=(8,),down_block_types=('DownBlock2D',),up_block_types=('UpBlock2D',),norm_num_groups=4)
x=torch.randn(1,1,8,8);pred=unet(x,torch.tensor(1)).sample
assert pred.shape==x.shape and torch.isfinite(pred).all()
schedule=DDPMScheduler(num_train_timesteps=10)
assert torch.isfinite(schedule.step(pred,1,x).prev_sample).all()
print('Tiny Diffusers UNet + scheduler PASS',flush=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots();ax.plot([0,1],[0,1],label='signal');ax.plot([0,1],[0,0],label='control');ax.legend();fig.savefig(out/'plot_smoke.png');plt.close(fig)
import plotly.graph_objects as go
fig=go.Figure(go.Heatmap(z=[[1.,0.],[0.,1.]]));fig.write_html(out/'plot_smoke.html',include_plotlyjs=True)
import nbformat
notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('1 + 1')]);nbformat.validate(notebook)
from jupyter_client import KernelManager
km=KernelManager();km.start_kernel();client=km.client();client.start_channels()
try:
    client.wait_for_ready(timeout=30);client.execute('1 + 1')
    found=False
    while True:
        msg=client.get_iopub_msg(timeout=30)
        if msg['msg_type']=='execute_result':found=msg['content']['data']['text/plain']=='2'
        if msg['msg_type']=='error':raise RuntimeError(msg['content'])
        if msg['msg_type']=='status' and msg['content']['execution_state']=='idle':break
    assert found
finally:
    client.stop_channels();km.shutdown_kernel(now=True)
print('Matplotlib, Plotly, notebook and live Jupyter kernel PASS',flush=True)
(out/'package_smoke.json').write_text(
    json.dumps(
        {
            'imports_passed': sum(r['status']=='PASS' for r in results),
            'imports_failed': sum(r['status']=='FAIL' for r in results),
            'tiny_gpt2_peft': 'PASS',
            'tiny_diffusers': 'PASS',
            'matplotlib': 'PASS',
            'plotly_html': 'PASS',
            'notebook_validation': 'PASS',
            'live_jupyter_kernel': 'PASS',
        },
        indent=2,
    )+'\n'
)
assert all(r['status']=='PASS' for r in results)
