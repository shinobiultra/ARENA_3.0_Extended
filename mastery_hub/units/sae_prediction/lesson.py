"""Visible fixed-forward exercise. Module import builds no numerical arrays."""
from . import records
METHOD_NAMES=records.METHOD_NAMES

SOLUTION_SOURCE='''def affine(X, W, b):
    require(X.ndim == W.ndim == 2 and b.ndim == 1 and X.shape[1] == W.shape[0] and W.shape[1] == len(b), 'affine shapes')
    return X @ W + b

def relu_encode(M):
    return M.clamp(min=0)

def topk_encode(M, k, original_feature_ids):
    require(type(k) is int and 0 <= k <= M.shape[1], 'integer k excluding bool')
    require(
        type(original_feature_ids) is tuple
        and len(original_feature_ids) == M.shape[1]
        and all(type(i) is int for i in original_feature_ids)
        and len(set(original_feature_ids)) == len(original_feature_ids),
        'immutable unique original feature IDs'
    )
    positive = M.clamp(min=0)
    Z = t.zeros_like(positive)
    for i in range(len(M)):
        ordered = sorted(range(M.shape[1]), key=lambda j: (-float(positive[i,j]), original_feature_ids[j]))
        for j in ordered[:k]:
            Z[i,j] = positive[i,j]
    return Z

def gated_encode(M, G):
    require(M.shape == G.shape, 'gate shape')
    return M.clamp(min=0) * (G > 0)

def jump_encode(M, theta):
    require(theta.ndim == 1 and theta.shape[0] == M.shape[1] and bool(t.isfinite(theta).all()) and bool((theta >= 0).all()), 'finite nonnegative per-feature threshold')
    positive = M.clamp(min=0)
    return positive * (positive > theta)

def decode(Z, D, b_dec):
    require(Z.ndim == D.ndim == 2 and b_dec.ndim == 1 and Z.shape[1] == D.shape[0] and D.shape[1] == len(b_dec), 'decoder shapes')
    return Z @ D + b_dec

def measure_rows(X, Z, reconstruction):
    require(X.shape == reconstruction.shape and X.ndim == Z.ndim == 2 and len(X) == len(Z), 'measurement shapes')
    errors = ((reconstruction-X)**2).mean(dim=1)
    counts = (Z > 0).sum(dim=1, dtype=t.int64)
    return errors, counts
'''

DEV_TESTS={
'affine':"""M = ns['affine'](T([[1.,2.],[-1.,0.]]), T([[1.,0.,1.],[0.,1.,-1.]]), T([0.,0.,.5]))
assert M.tolist() == [[1.,2.,-.5],[-1.,0.,-.5]]""",
'relu_encode':"""assert ns['relu_encode'](T([[-1.,0.,2.],[.5,-.5,0.]])).tolist() == [[0.,0.,2.],[.5,0.,0.]]""",
 'topk_encode':"""M = T([[2.,2.,-1.],[0.,-1.,-1.]])
assert ns['topk_encode'](M,1,(9,2,4)).tolist() == [[0.,2.,0.],[0.,0.,0.]]
assert ns['topk_encode'](M,0,(9,2,4)).tolist() == [[0.,0.,0.],[0.,0.,0.]]
assert ns['topk_encode'](M,3,(9,2,4)).tolist() == [[2.,2.,0.],[0.,0.,0.]]""",
'gated_encode':"""assert ns['gated_encode'](T([[2.,-1.,3.],[1.,2.,0.]]), T([[-1.,1.,0.],[1.,0.,1.]])).tolist() == [[0.,0.,0.],[1.,0.,0.]]""",
'jump_encode':"""assert ns['jump_encode'](T([[1.,.5,.75],[0.,2.,-.5]]), T([1.,.5,.25])).tolist() == [[0.,0.,.75],[0.,2.,0.]]""",
'decode':"""assert ns['decode'](T([[1.,2.,0.],[0.,0.,0.]]), T([[1.,0.],[0.,1.],[1.,1.]]), T([.5,-.5])).tolist() == [[1.5,1.5],[.5,-.5]]""",
'measure_rows':"""E,L = ns['measure_rows'](T([[1.,2.],[-1.,0.]]),T([[1.,2.,0.],[0.,0.,0.]]),T([[1.5,1.5],[.5,-.5]]))
assert E.tolist() == [.25,1.25] and L.tolist() == [2,0] and L.dtype == t.int64"""}

DESCRIPTIONS={
'affine': (
    'Build both preactivation branches',
    ('Implement X @ W + b with non-square matrices. Canonical 6.1 uses (X-b_dec) @ W + b; this target fixes '
     'b_dec=0, the same specialization. The nonzero decoder-bias DEV test later tests decoding, not altered '
     'centering.'),
    'For DEV row [1,2], the third coordinate is 1-2+.5=-.5. Bias broadcasts over examples.',
),
'relu_encode':('Keep positive magnitudes','Clamp negative preactivations to zero. L1 belongs in a training loss; it does not subtract a threshold here.','Zero is inactive; a magnitude of 2 stays 2.'),
'topk_encode': (
    'Spend at most k active coordinates',
    ('Keep the largest k positive magnitudes per example, breaking ties by immutable original feature ID. '
     'Validate k as an integer excluding bool, 0 <= k <= width.'),
    ('The tied DEV IDs 9 and 2 select ID 2. Zero ties never spend an active coordinate. Realized L0 can be below '
     'k.'),
),
'gated_encode':('Separate magnitude and gate','Use max(M,0) * (G>0), with independent G. Gate equality is inactive.','A positive magnitude with a negative gate is zero; a negative magnitude never becomes positive.'),
'jump_encode':('Jump without shrinking amplitude','Keep positive magnitude only when strictly above a finite nonnegative per-feature threshold.','A value exactly equal to theta is inactive; .75 over .25 remains .75.'),
'decode':('Put the coordinates back','Return Z @ D + b_dec, with decoder rows corresponding to code columns.','Zero codes decode to b_dec. Under p[j]=old feature at new j, permute D rows with code columns.'),
'measure_rows': (
    'Measure every example',
    ('Return (row MSE, active counts): float64 errors averaged over dimensions; int64 counts of Z>0. The overall '
     'mean later includes all examples, including zero-code rows.'),
    'DEV errors are .25 and 1.25; counts are 2 and 0. No normalization by reconstruction norm is involved.',
)}

def practice_test(ns,name):
    import torch as t
    scope={'ns':ns,'t':t,'T':lambda values:t.tensor(values,dtype=t.float64,device='cpu')}
    exec(DEV_TESTS[name],scope)
    return name+': hand-calculated DEV checks passed'

def solutions(commitment_text):
    """Disclose source only after a valid immutable numerical commitment."""
    d=records.read(commitment_text);records.require(d['kind']=='commitment','commit first')
    import ast,html,inspect
    from .runner import scalar_oracle
    dropdowns=''.join('<details><summary>Postcommit solution: '+node.name+'</summary><pre>'+html.escape(ast.get_source_segment(SOLUTION_SOURCE,node))+'</pre></details>' for node in ast.parse(SOLUTION_SOURCE).body)
    return dropdowns+'<details><summary>Independent scalar oracle (fixed i,j,k order)</summary><pre>'+html.escape(inspect.getsource(scalar_oracle))+'</pre></details>'

def plot(observation_text):
    """Plots retained arrays only. No live identities, numerical runner or oracle."""
    d=records.read(observation_text);records.require(d['kind']=='observation','stored observation required')
    commit=records.read(d['parent']);identity=records.read(commit['parent'])['payload'];case=identity['recipe'];p=d['payload'];conditions=p['conditions'];methods=case['method_order']
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,5,figsize=(19,10),layout='constrained')
    maximum=max(v for m in methods for v in records.flat(conditions['primary']['codes'][m]['values']))
    bound=max(abs(v) for label in ('primary','zero','mismatch') for m in methods for v in records.flat(conditions[label]['residuals'][m]['values']))
    mse_max=max(v for label in ('primary','zero','mismatch') for m in methods for v in conditions[label]['errors'][m]['values'])
    mse_upper=1.05*mse_max if mse_max>0 else 1.0
    for j,m in enumerate(methods):
        axes[0,j].imshow(conditions['primary']['codes'][m]['values'],vmin=0,vmax=maximum or 1,aspect='auto',cmap='viridis');axes[0,j].set_title(m+' code')
        # Every residual panel includes primary alongside mismatch; zero shown at right.
        joined=[a+b for a,b in zip(conditions['primary']['residuals'][m]['values'],conditions['mismatch']['residuals'][m]['values'])]
        axes[1,j].imshow(joined,vmin=-(bound or 1),vmax=bound or 1,cmap='RdBu_r',aspect='auto');axes[1,j].set_title(m+' residual | mismatch')
        ax=axes[2,j]
        for label in ('primary','zero','mismatch'):ax.plot(conditions[label]['errors'][m]['values'],'o-',label=label)
        ax.set_ylim(0,mse_upper)
        ax.axhline(p['means'][m],linestyle=':',label='primary mean');ax.set_title(m+' row MSE');ax.set_ylabel('squared activation units');ax.legend(fontsize=7)
    axes[0,4].imshow(conditions['zero']['codes']['relu']['values'],vmin=0,vmax=maximum or 1,aspect='auto',cmap='viridis');axes[0,4].set_title('zero-code control')
    axes[1,4].imshow(conditions['zero']['residuals']['relu']['values'],vmin=-(bound or 1),vmax=bound or 1,cmap='RdBu_r',aspect='auto');axes[1,4].set_title('zero-code residual')
    for m in methods:axes[2,4].plot(conditions['primary']['counts'][m]['values'],'o-',label=m)
    axes[2,4].plot([0]*len(case['X']),label='zero');axes[2,4].axhline(case['k'],linestyle=':',label='TopK k bound');axes[2,4].set_title('L0 active coordinates');axes[2,4].legend(fontsize=7)
    for ax in axes[0]:
        ax.set_xlabel('feature');ax.set_xticks(range(len(case['feature_order'])),case['feature_order'])
        ax.set_yticks(range(len(case['example_order'])),case['example_order']);ax.set_ylabel('example')
    for j,ax in enumerate(axes[1]):
        ax.set_xlabel('activation dimension');ax.set_ylabel('example');ax.set_yticks(range(len(case['example_order'])),case['example_order'])
        dims=len(case['X'][0]);labels=['x'+str(i) for i in range(dims)]
        if j<4:labels+=['mismatch x'+str(i) for i in range(dims)];ax.axvline(dims-.5,color='black',linewidth=1)
        ax.set_xticks(range(len(labels)),labels,rotation=25,ha='right')
    for ax in axes[2]:ax.set_xlabel('example row');ax.set_xticks(range(len(case['example_order'])),case['example_order'])
    fig.colorbar(axes[0,0].images[0],ax=list(axes[0]),label='nonnegative code magnitude')
    fig.colorbar(axes[1,0].images[0],ax=list(axes[1]),label='signed residual activation')
    fig.suptitle(case['id']+' — stored fixed-forward evidence; controls retained; scalar-oracle agreement; coherent relabeling preserved')
    return fig

def dev_mismatch_panel(source):
    """Hand-worked identity decoder control. Distinct 2x2/3-feature DEV only."""
    import matplotlib.pyplot as plt
    # First two decoder rows are an identity; third row is unused. Cyclic row
    # mismatch sends code [1,2,0] to [0,1], and [3,1,0] to [0,3].
    import torch as t
    ns=records.namespace(source);T=lambda v:t.tensor(v,dtype=t.float64)
    Z=T([[1.,2.,0.],[3.,1.,0.]]);D=T([[1.,0.],[0.,1.],[0.,0.]]);bias=T([0.,0.])
    correct=ns['decode'](Z,D,bias).tolist();mismatched=ns['decode'](Z,D[[1,2,0]],bias).tolist()
    assert correct==[[1.,2.],[3.,1.]] and mismatched==[[0.,1.],[0.,3.]]
    residual=[[b-a for a,b in zip(row,wrong)] for row,wrong in zip(correct,mismatched)]
    assert residual==[[-1.,-1.],[-3.,2.]]
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    reconstruction_max=max(v for rows in (correct,mismatched) for row in rows for v in row)
    residual_bound=max(abs(v) for row in residual for v in row)
    for j,(ax,values,title) in enumerate(zip(axes,(correct,mismatched,residual),('DEV identity decode: MSE 0','DEV row mismatch','Known failure: row MSE 1, 6.5'))):
        if j<2:ax.imshow(values,aspect='auto',vmin=0,vmax=reconstruction_max,cmap='viridis')
        else:ax.imshow(values,aspect='auto',vmin=-residual_bound,vmax=residual_bound,cmap='RdBu_r')
        ax.set_title(title);ax.set_xticks([0,1],['x0','x1']);ax.set_yticks([0,1],['u','v']);ax.set_ylabel('DEV example')
        ax.set_xlabel('activation dimension' if j<2 else 'activation dimension\nsigned residual: mismatched - reference')
    fig.colorbar(axes[0].images[0],ax=list(axes[:2]),label='reconstruction activation (shared scale)')
    fig.colorbar(axes[2].images[0],ax=axes[2],label='signed residual: mismatched - reference')
    fig.suptitle('Known-answer DEV alignment control — reconstruction and signed residuals')
    return fig

def build_cells():
    cells=[]
    def add(kind,source,tag):
        c={'cell_type':kind,'metadata':{'sae_prediction':tag},'source':source,'id':tag}
        if kind=='code':c.update(execution_count=None,outputs=[])
        cells.append(c)
    add('markdown',(
            '# [6.1 companion] One activation cloud, four encoder rules\n'
            '\n'
            'Can you predict the signed difference **TopK MSE minus JumpReLU MSE**, under the same fixed '
            'weights, before seeing any codes? Every row counts; the units are squared activation units.\n'
            '\n'
            'Learn to implement affine branches, four forward encoders, decoding and row measurements; diagnose '
            'alignment and support; explain a wrong forecast from retained rows.\n'
            '\n'
            'These public constants are illustrative and fixed before measurement. TRAIN: none. DEV: exposed '
            'canonical lesson and separate hand-worked fixtures. TEST: none. This is an analytic forward '
            'exercise, not a learned-SAE ranking. Existing 6.1 outputs have already been viewed by engineers. No '
            'local seal can certify unseen exposure.\n'
            '\n'
            'Target status: **UNEXECUTED — implementation-only preparation**. The target cell ships dormant and '
            'needs separately authorized execution. No training, models, cache, downloads, GPU or cluster are '
            'used.\n'
            '\n'
            'Each editable source string below contains one plain function. Its actual UTF-8 bytes enter the '
            'commitment; defining source runs no function body. Complete and test these functions before making '
            'your forecast.'
        ),'introduction')
    add('code',"""from pathlib import Path
import sys
root = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p/'mastery_hub').is_dir()), None)
if root is None:
    raise RuntimeError('Open this notebook inside the ARENA_3.0_Extended checkout.')
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
import torch as t
from IPython.display import display, HTML
from mastery_hub.units.sae_prediction import records, runner, lesson
t.set_num_threads(1)  # CPU-only bounded numerical contract, before freeze
# This read shows raw public literals only. It constructs no target tensors.
print((records.UNIT / 'protocol.json').read_text())
# Keep edited method bytes separate from the immutable pristine template.""",'setup')
    add('markdown',(
            'Raw X rows A–F and decoder rows f0–f3 above are the concrete prediction problem. D is an analytic '
            'dictionary, not a recovered feature set. Compare the magnitude and independent gate weights before '
            'coding. CPU float64, one thread, strict > boundaries; exactly-on thresholds are inactive. Numerical '
            'arrays have finite CPU float64 values. measure_rows alone returns int64 counts alongside float64 '
            'errors.'
        ),'background')
    for name,signature in zip(METHOD_NAMES,records.SIGNATURES):
        title,task,hint=DESCRIPTIONS[name]
        add('markdown',(f'## Exercise: {title}\n\n{task}\n\n<details><summary>Help / interpretation</summary>\n\n'
            f'{hint}\n\n</details>\n\nExpected DEV check below: hand-calculated arrays in the assertions, '
            f'followed by `{name}: hand-calculated DEV checks passed`. No target expected values are published.'),name+'-prompt')
        stub=f"def {name}({', '.join(signature)}):\n    raise NotImplementedError('Implement {name}')\n"
        add('code',f"{name.upper()}_SOURCE = '''\n{stub}'''",name+'-source')
        add('code',f"method_source = '\\n'.join([{', '.join(v.upper()+'_SOURCE' for v in METHOD_NAMES[:METHOD_NAMES.index(name)+1])}])\n"
            f"ns = records.namespace(method_source, complete=False)\n"
            f"T = lambda values: t.tensor(values, dtype=t.float64, device='cpu')\n"
            f"{DEV_TESTS[name]}\nprint({name!r} + ': hand-calculated DEV checks passed')",name+'-dev-test')
    add('markdown', '''## Exposed DEV ladder — support versus reconstruction (10–15 minutes)
**Implementation preparation + complete exposed DEV cycle; target remains dormant.**
Hand-predict the existing two-row/three-feature DEV play below: first row M=[1,2,−.5],
second row M=[−1,0,−.5]. With k=1, TopK keeps only feature1 on row0; strict JumpReLU
at theta=[1,.5,.25] also drops the exactly-on-threshold feature0. Predict both row MSEs.
Choose **A:** selecting more positive features always changes every row's reconstruction;
or **B:** only changed support with nonzero decoder contribution can change a row.
Name a paired observation ruling A out. After the original DEV plot, compare k1→k2,
keeping decoder, inputs and all gates fixed. Zero and mismatch controls remain visible.
This is exposed DEV, not an unseen target. Confidence is an **unscored self-rating**.''', 'dev-forecast-prompt')
    add('code', '''dev_forecast_rows = [.5,.5]  # TopK k1 and JumpReLU, hand-worked DEV
dev_self_rating = .7  # unscored reasoning self-rating
dev_explanation = 'B'
dev_disconfirming_observation = 'k2 changes row0 but leaves the zero-code row1 unchanged.' ''', 'dev-forecast-card')
    add('markdown',(
            '## DEV controls and interpretation — before exact freeze\n'
            '\n'
            'Zero codes decode to b_dec, and decoder-row permutation deliberately mismatches feature '
            'coordinates. Neither is required to worsen every target row. Keep any null or reversed result and '
            'inspect which coordinates cancel residuals. These are fixed forward rules, with no fitted fairness '
            'or optimization claim. Coherent relabeling and scalar-oracle checks diagnose implementation '
            'consistency; they do not establish meaningful learned features.\n'
            '\n'
            'The separate known-answer DEV control below has hand-written distinct codes and an identity decoder '
            '(plus an unused third feature). Its Z=[[1,2,0],[3,1,0]], D=[[1,0],[0,1],[0,0]] and cyclic p=[1,2,0] '
            'mismatch must visibly fail; expected residual rows [-1,-1] and [-3,2] give MSE 1 and 6.5. This '
            'proves the control can detect alignment errors without choosing the public target for a desired '
            'winner.'
        ),'controls')
    add('code','display(lesson.dev_mismatch_panel(method_source))','dev-mismatch')
    add('markdown',(
            '## Try It Yourself — DEV only\n'
            '\n'
            'Before any exact-target commitment, change DEV k from 1 to 0 or 3. Edit the separate 2D/3-feature literals in '
            'dev_play or copy the cell to vary a gate/threshold. Predict a row before re-running. This is '
            'exposed exploration, not a target transfer or another target attempt.'
            ' Copy the play cell before editing, or restore its supplied literals before the fixed k1→k2 '
            'comparison below; its hand-calculated checks describe those supplied DEV inputs.'
        ),'play-prompt')
    add('code',"""# Exposed DEV: two examples, two activation dimensions, three features.
# Change k, gate bias, or theta; predict which DEV coordinate changes first.
import matplotlib.pyplot as plt
T = lambda values: t.tensor(values, dtype=t.float64, device='cpu')
DEV_X = T([[1.,2.],[-1.,0.]])
DEV_D = T([[1.,0.],[0.,1.],[1.,1.]])
DEV_W = T([[1.,0.,1.],[0.,1.,-1.]])
DEV_GATE_W = T([[1.,0.,0.],[0.,1.,1.]])
DEV_GATE_BIAS = T([-.5,0.,-1.])
DEV_THETA = T([1.,.5,.25])
DEV_K = 1
ns = records.namespace(method_source)
M = ns['affine'](DEV_X, DEV_W, T([0.,0.,.5]))
G = ns['affine'](DEV_X, DEV_GATE_W, DEV_GATE_BIAS)
codes = {'relu':ns['relu_encode'](M), 'topk':ns['topk_encode'](M,DEV_K,(0,1,2)),
         'gated':ns['gated_encode'](M,G), 'jumprelu':ns['jump_encode'](M,DEV_THETA),
         'zero':t.zeros_like(M)}
fig, axes = plt.subplots(2,5,figsize=(15,5),layout='constrained')
for j,(name,Z) in enumerate(codes.items()):
    R = ns['decode'](Z,DEV_D,T([0.,0.]))
    E,L = ns['measure_rows'](DEV_X,Z,R)
    axes[0,j].imshow(Z.tolist(),vmin=0,vmax=3,aspect='auto');axes[0,j].set_title(name+' DEV code')
    axes[1,j].imshow((R-DEV_X).tolist(),vmin=-3,vmax=3,cmap='RdBu_r',aspect='auto')
    axes[1,j].set_title('row MSE '+str(E.tolist())+' / L0 '+str(L.tolist()))
display(fig)""",'dev-play')
    add('code', '''# One changed cause: k 1 -> 2. Existing ns, M, codes and decoder are DEV only.
dev_k1 = ns['topk_encode'](M,1,(0,1,2))
dev_k2 = ns['topk_encode'](M,2,(0,1,2))
dev_pair_errors = {}
for name,Z in [('TopK k1',dev_k1),('TopK k2',dev_k2),('JumpReLU',codes['jumprelu']),('Zero',codes['zero'])]:
    R=ns['decode'](Z,DEV_D,T([0.,0.]))
    E,L=ns['measure_rows'](DEV_X,Z,R)
    dev_pair_errors[name]=E
fig,ax=plt.subplots(figsize=(8,3),layout='constrained')
for name,errors in dev_pair_errors.items(): ax.plot([0,1],errors.tolist(),'o-',label=name)
ax.set(xlabel='DEV row',ylabel='MSE across both activation dimensions',title='DEV k intervention with fixed decoder',xticks=[0,1]);ax.legend();display(fig)
t.testing.assert_close(dev_pair_errors['TopK k1'],T([.5,.5]))
t.testing.assert_close(dev_pair_errors['TopK k2'],T([0.,.5]))
print('DEV row absolute forecast errors:',(dev_pair_errors['TopK k1']-T(dev_forecast_rows)).abs().tolist())
print('DEV k2 minus JumpReLU mean MSE:',float(dev_pair_errors['TopK k2'].mean()-dev_pair_errors['JumpReLU'].mean()))''', 'dev-paired')
    add('markdown', '''### Worked DEV reflection / self-check
TopK k1 and JumpReLU both have row MSE [.5,.5]. TopK k2 has [0,.5], so its mean-minus-Jump
contrast is −.25. A forecast of −.5 errs by .25 because the unchanged zero-code row still
counts in the mean. More available slots did not change row1: that rules A out.
Self-check: name the activated feature, its decoder row, the strict threshold equality,
and the unchanged row's contribution to the fixed denominator. Compare the deliberately
mismatched decoder panel; support size alone does not prove alignment or reconstruction quality.

### Try It Yourself — changed-condition DEV transfer
Freeze one change: JumpReLU theta0 1→0, same inputs, weights and other thresholds; keep TopK k2.
Predict the two JumpReLU row MSEs and the mean TopK-minus-Jump contrast before running.
Explanation B predicts [0,.5] and contrast0 because row0's feature0 can now contribute.''', 'dev-reflection')
    add('code', '''dev_transfer_theta=DEV_THETA.clone();dev_transfer_theta[0]=0.
dev_transfer_forecast={'row_mse':[0.,.5],'contrast':0.}
dev_transfer_Z=ns['jump_encode'](M,dev_transfer_theta)
dev_transfer_R=ns['decode'](dev_transfer_Z,DEV_D,T([0.,0.]))
dev_transfer_E,dev_transfer_L=ns['measure_rows'](DEV_X,dev_transfer_Z,dev_transfer_R)
dev_transfer_contrast=float(dev_pair_errors['TopK k2'].mean()-dev_transfer_E.mean())
print('DEV transfer row absolute forecast errors:',(dev_transfer_E-T(dev_transfer_forecast['row_mse'])).abs().tolist())
print('DEV transfer contrast absolute forecast error:',abs(dev_transfer_contrast-dev_transfer_forecast['contrast']))
t.testing.assert_close(dev_transfer_E,T([0.,.5]))  # Method check, independent of forecast.
assert dev_transfer_contrast==0.
fig,ax=plt.subplots(figsize=(7,3),layout='constrained')
for name,errors in [('Original JumpReLU',dev_pair_errors['JumpReLU']),('Lower theta0',dev_transfer_E),('Zero control',dev_pair_errors['Zero'])]:
    ax.plot([0,1],errors.tolist(),'o-',label=name)
ax.set(xlabel='DEV row',ylabel='Row MSE',title='Observed DEV threshold transfer',xticks=[0,1]);ax.legend();display(fig)
print('DEV threshold transfer row MSE:',dev_transfer_E.tolist(),'contrast:',dev_transfer_contrast)
dev_revision=''  # Cite strict threshold crossing, decoder contribution, and unchanged row1.''', 'dev-transfer')
    add('markdown',(
            '## Commit before reveal\n'
            '\n'
            'TopK carries immutable original feature IDs into its tie-breaker. Permutation p[j] means old '
            'feature at new column j. Permute encoder/gate columns, both feature biases, thresholds, decoder '
            'rows and IDs together; undo only for comparing codes. Coherent relabeling and scalar-oracle parity '
            'are correctness diagnostics.\n'
            '\n'
            'Forecast any finite signed number; confidence [0,1] is an unscored self-rating of your reasoning, not a '
            'calibrated probability interval. Write a mechanistic explanation. Correct forecasting is never a '
            'pass gate. Unknown exposure must stay unknown. Freeze binds recoverable source snapshots, raw bytes '
            "and runtime, and a stable session. A fixed authoritative session binding under the OS account's "
            '~/.local/share/arena-mastery/sae_prediction/sessions (independent of HOME/XDG environment values) '
            'chooses the first private attempt registry (using XDG_DATA_HOME at first binding). That session '
            'binding is resolved before each new freeze; it survives new commitments, export paths, processes, '
            'or changed environment paths. DEV/review harnesses root-bind its anchor explicitly to their '
            'sandbox. No new learner assembly method is needed.'
        ),'commit-prompt')
    add('code',"""prediction = None
confidence = None
reasoning = ''
prior_exposure = 'unknown'
session = 'learner-primary-v1'  # freeze this before any authorized attempt
# Fill the four forecast fields, then explicitly run this cell.
identity = records.freeze(method_source, session)
commitment = records.commit(identity, prediction, confidence, reasoning, prior_exposure)""",'forecast')
    add('code',"display(HTML(lesson.solutions(commitment)))  # postcommit solution dropdowns only",'postcommit-solutions')
    add('markdown',(
            '## One later attempt, retained evidence\n'
            '\n'
            'The run below is dormant. Separate authorization is still required. One CPU thread and one '
            '60-second wall/CPU attempt covers all conditions and finalization; use an external 60-second '
            'process limit for a future authorized dispatch. Python deadline checks cannot interrupt a hung '
            'student call. The exclusive stable recipe/session start survives termination; unknown starts remain '
            'consumed. A failure raises AttemptFailure carrying valid saved failure text and retains partial '
            'arrays. Error summaries are limited by UTF-8 bytes, with explicit truncation flags and recoverable '
            'exact messages (inline through 1 MB; private original-message artifacts through 8 MB). Larger '
            'messages or failed storage are explicitly marked incomplete; the original exception remains the '
            'cause. Primary, secondary, and storage errors stay separate. Do not rerun automatically.\n'
            '\n'
            'Expected signature: example-by-feature code heatmaps (shared color limits), example-by-dimension '
            'residual heatmaps (shared symmetric limits), row MSE including zero and mismatched decoder '
            'controls, and L0 alongside zero and TopK k. Null/reversed controls remain visible. A wrong forecast '
            'leads to a row-level explanation, not another attempt.'
        ),'run-prompt')
    add('code',"""RUN_AUTHORIZED_TARGET = False  # dormant: no authorization has been granted
observation = None
if RUN_AUTHORIZED_TARGET:
    observation = runner.run(commitment, method_source, 'private-sae-attempt', authorized=True)
    display(lesson.plot(observation))""",'dormant-run')
    add('markdown',(
            '## Reflection and replay\n'
            '\n'
            'From retained codes, identify which row and activation dimensions explain the signed forecast '
            'error. Explain any reversed decoder control. Failures receive an unscored reflection. Replay, plot '
            'and export use stored arrays/snapshots across changed runtime; they do not execute target methods '
            'or retry. Export creates a new private file exclusively. Seals detect accidental drift only.'
        ),'reflection-prompt')
    add('code',"""reflection_text = ''
# After a saved observation or failure, fill the explanation and explicitly run:
# reflection = records.reflect(observation_or_failure_text, reflection_text)
# records.export(reflection, 'sae-reflection.json')""",'reflection')
    add('markdown',(
            '## Canonical bridge and limitations\n'
            '\n'
            'Return to [6.1 SAE Variants](../../../chapter6_sparse_feature_methods/instructions/pages/01_%5B6.1%5'
            'D_SAE_Variants.md) and [encoder / loss '
            'source](../../../chapter6_sparse_feature_methods/exercises/part1_sae_variants/solutions.py). '
            'Canonical SparseAutoencoder.forward centers by decoder bias; our zero b_dec specializes that '
            'formula. L1 training pressure, auxiliary gated loss, threshold learning/pseudo-gradients, feature '
            'recovery, held-out tests and real-model use remain in 6.1.\n'
            '\n'
            'The [original ARENA SAE section](../../../chapter1_transformer_interp/exercises/part54_toy_models_of'
            '_superposition_and_saes/1.5.4_Toy_Models_of_Superposition_&_SAEs_exercises.ipynb) supplies the full '
            'progression. Paper connections: [Gated SAEs](https://arxiv.org/abs/2404.16014), [TopK '
            'SAEs](https://arxiv.org/abs/2406.04093), [JumpReLU SAEs](https://arxiv.org/abs/2407.14435). These '
            'references motivate forward rules; this companion does not reproduce their training/results or rank '
            'learned SAE families.\n'
            '\n'
            'No learned dictionary, generalization evaluation, BatchTopK, target transfer run, calibrated uncertainty, '
            'causal feature claim or human learner acceptance is supplied here. Human walkthrough is deferred. '
            'The target remains unexecuted until separately authorized. Original ARENA files are preserved '
            'append-only.'
        ),'limitations')
    return cells
