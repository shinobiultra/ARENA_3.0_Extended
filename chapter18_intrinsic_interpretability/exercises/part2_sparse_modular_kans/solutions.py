# %%


from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

_here = Path.cwd().resolve()
_candidates = [_here, *_here.parents]
_root = next((p for p in _candidates if (p / "chapter18_intrinsic_interpretability" / "exercises").exists()), None)
if _root is not None:
    exercises_dir = _root / "chapter18_intrinsic_interpretability" / "exercises"
else:
    exercises_dir = next(p for p in _candidates if (p / "part2_sparse_modular_kans").exists())
if str(exercises_dir) not in sys.path:
    sys.path.insert(0, str(exercises_dir))
from part2_sparse_modular_kans import tests as tests
MAIN = __name__ == "__main__"
np.set_printoptions(precision=4, suppress=True)

# %%

def mean_squared_error(predictions, targets):
    return float(np.mean((np.asarray(predictions)-np.asarray(targets))**2))


if MAIN:
    tests.test_mean_squared_error(mean_squared_error)

# %%

def connection_cost(input_positions, output_positions):
    return 1.0 + np.abs(np.asarray(input_positions)[:,None] - np.asarray(output_positions)[None,:])


if MAIN:
    tests.test_connection_cost(connection_cost)

# %%

def soft_threshold(values, thresholds):
    return np.sign(values)*np.maximum(np.abs(values)-np.asarray(thresholds),0.0)


if MAIN:
    tests.test_soft_threshold(soft_threshold)

# %%

def proximal_fit(features, targets, costs, strength, steps=80, learning_rate=0.5):
    x=np.asarray(features,dtype=float)
    y=np.asarray(targets,dtype=float)
    w=np.zeros((x.shape[1],y.shape[1]))
    for _ in range(steps):
        gradient=x.T@(x@w-y)/len(x)
        w=soft_threshold(w-learning_rate*gradient,learning_rate*strength*np.asarray(costs))
    return w


if MAIN:
    tests.test_proximal_fit(proximal_fit)

# %%

def off_module_fraction(weights, allowed):
    magnitude=np.abs(weights)
    total=magnitude.sum()
    return float(magnitude[~np.asarray(allowed,dtype=bool)].sum()/total) if total else 0.0


if MAIN:
    tests.test_off_module_fraction(off_module_fraction)

# %%

if MAIN:
    from itertools import product
    knots = np.linspace(-1.,1.,5)
    # Public construction grid: 5 x 5 x 2 = 50 points, with no sampling or data selection.
    train_x = np.array(list(product(knots,knots,[-1.,1.])))
    raw_features=np.column_stack([train_x[:,0]**2-.5,train_x[:,1],train_x[:,2]])
    feature_scale=np.sqrt(np.mean(raw_features**2,axis=0))
    X=raw_features/feature_scale
    assert np.allclose(X.T@X/len(X),np.eye(3))
    clean_components=np.column_stack([raw_features[:,0],.5*raw_features[:,1]])
    # A deliberately supplied, recoverable perturbation, not a natural dataset.
    training_components=clean_components + np.column_stack([.05*X[:,1]+.15*X[:,2],.15*X[:,2]])
    costs=connection_cost([-1.,1.,0.],[-1.,1.])
    allowed=np.array([[True,False],[False,True],[False,False]])
    dense_weights=proximal_fit(X,training_components,np.ones((3,2)),0.)
    l1_weights=proximal_fit(X,training_components,np.ones((3,2)),.08)
    local_weights=proximal_fit(X,training_components,costs,.08)
    for name,w in [('dense',dense_weights),('uniform L1',l1_weights),('distance L1',local_weights)]:
        print(name,'clean component MSE / off-module mass / nonzero edges:',round(mean_squared_error(X@w,clean_components),6),round(off_module_fraction(w,allowed),4),int(np.sum(np.abs(w)>1e-8)))
    fig,axes=plt.subplots(1,3,figsize=(14,3.6),layout="constrained")
    for ax,w,name in zip(axes,[dense_weights,l1_weights,local_weights],['Dense fit','Uniform L1 control','Distance-weighted L1']):
        im=ax.imshow(w,vmin=-.1,vmax=.7,cmap='coolwarm')
        ax.set(xticks=[0,1],xticklabels=['square','linear'],yticks=[0,1,2],yticklabels=['square feature','linear feature','nuisance'],xlabel='Output module',ylabel='Input feature',title=name)
        for i in range(3):
            for j in range(2): ax.text(j,i,f'{w[i,j]:.2f}',ha='center',va='center')
    fig.colorbar(im,ax=list(axes),label='Fitted standardized-feature weight',shrink=.8)
    plt.show()

# %%

def hat_basis(values, knots):
    values=np.asarray(values,dtype=float)
    knots=np.asarray(knots,dtype=float)
    if len(knots)<2 or np.any(np.diff(knots)<=0):
        raise ValueError("knots must be strictly increasing")
    if np.any(values<knots[0]) or np.any(values>knots[-1]):
        raise ValueError("outside spline domain")
    basis=np.zeros(values.shape+(len(knots),))
    for k in range(len(knots)):
        if k>0:
            left=(values>=knots[k-1])&(values<=knots[k])
            basis[...,k]+=np.where(left,(values-knots[k-1])/(knots[k]-knots[k-1]),0.)
        if k<len(knots)-1:
            right=(values>knots[k])&(values<=knots[k+1])
            basis[...,k]+=np.where(right,(knots[k+1]-values)/(knots[k+1]-knots[k]),0.)
    # The first knot has no left interval.
    basis[...,0]=np.where(values==knots[0],1.,basis[...,0])
    return basis


if MAIN:
    tests.test_hat_basis(hat_basis)

# %%

def additive_design(inputs, knots):
    inputs=np.asarray(inputs,dtype=float)
    knots=np.asarray(knots,dtype=float)
    anchor=np.flatnonzero(np.isclose(knots,0.))
    if len(anchor)!=1:
        raise ValueError("exactly one zero anchor knot is required")
    centered=hat_basis(inputs,knots)-hat_basis(np.array(0.),knots)
    reduced=np.delete(centered,int(anchor[0]),axis=-1)
    return np.column_stack([np.ones(len(inputs)),reduced.reshape(len(inputs),-1)])


if MAIN:
    tests.test_additive_design(additive_design)

# %%

def fit_additive_spline(inputs, targets, knots):
    return np.linalg.lstsq(additive_design(inputs,knots),np.asarray(targets),rcond=None)[0]


if MAIN:
    tests.test_fit_additive_spline(fit_additive_spline)

# %%

def edge_values(inputs, coefficients, knots):
    inputs=np.asarray(inputs,dtype=float)
    design=additive_design(inputs,knots)[:,1:]
    width=len(knots)-1
    return np.sum(design.reshape(len(inputs),inputs.shape[1],width)*np.asarray(coefficients)[1:].reshape(inputs.shape[1],width),axis=-1)


if MAIN:
    tests.test_edge_values(edge_values)

# %%

def symbolic_affine_fit(coordinate, edge, candidate):
    z=np.asarray(candidate(np.asarray(coordinate)),dtype=float)
    design=np.column_stack([np.ones(len(z)),z])
    params=np.linalg.lstsq(design,np.asarray(edge),rcond=None)[0]
    residual=mean_squared_error(design@params,edge)
    return params,residual


if MAIN:
    tests.test_symbolic_affine_fit(symbolic_affine_fit)

# %%

if MAIN:
    spline_coefficients=fit_additive_spline(train_x[:,:2],train_x[:,0]**2+.5*train_x[:,1],knots)
    edge_grid=np.column_stack([knots,np.zeros(len(knots))])
    square_edge=edge_values(edge_grid,spline_coefficients,knots)[:,0]
    line_grid=np.column_stack([np.zeros(len(knots)),knots])
    linear_edge=edge_values(line_grid,spline_coefficients,knots)[:,1]
    square_params,square_residual=symbolic_affine_fit(knots,square_edge,np.square)
    linear_params,linear_residual=symbolic_affine_fit(knots,linear_edge,lambda x:x)
    wrong_params,wrong_residual=symbolic_affine_fit(knots,square_edge,lambda x:x)
    print('Recovered square offset/scale, knot MSE:',square_params,round(square_residual,12))
    print('Recovered linear offset/scale, knot MSE:',linear_params,round(linear_residual,12))
    print('Wrong linear explanation of square, knot MSE:',round(wrong_residual,6))
    print('Off-knot x=.25: spline square / symbolic square / oracle:',edge_values(np.array([[.25,0.]]),spline_coefficients,knots)[0,0],square_params[0]+square_params[1]*.25**2,.25**2)

# %%

def best_split(inputs, targets):
    x=np.asarray(inputs);y=np.asarray(targets,dtype=float)
    parent=np.sum((y-y.mean())**2)
    best=(None,None,0.0)
    for feature in range(x.shape[1]):
        unique=np.unique(x[:,feature])
        for threshold in (unique[:-1]+unique[1:])/2:
            left=x[:,feature]<=threshold
            sse=np.sum((y[left]-y[left].mean())**2)+np.sum((y[~left]-y[~left].mean())**2)
            gain=float(parent-sse)
            if gain>best[2]+1e-12:
                best=(feature,float(threshold),gain)
    return best


if MAIN:
    tests.test_best_split(best_split)

# %%

def fit_sparse_tree(inputs, targets, penalty=0.01, max_depth=4, root_count=None):
    x=np.asarray(inputs);y=np.asarray(targets,dtype=float)
    root_count=len(y) if root_count is None else root_count
    leaf={"value":float(y.mean())}
    if max_depth==0 or len(y)<2:
        return leaf
    feature,threshold,gain=best_split(x,y)
    if feature is None or gain/root_count<=penalty:
        return leaf
    left=x[:,feature]<=threshold
    return {"feature":feature,"threshold":threshold,
            "left":fit_sparse_tree(x[left],y[left],penalty,max_depth-1,root_count),
            "right":fit_sparse_tree(x[~left],y[~left],penalty,max_depth-1,root_count)}


if MAIN:
    tests.test_fit_sparse_tree(fit_sparse_tree)

# %%

def predict_tree(tree, inputs):
    x=np.asarray(inputs)
    if "value" in tree:
        return np.full(len(x),tree["value"])
    left=x[:,tree["feature"]]<=tree["threshold"]
    pred=np.empty(len(x))
    pred[left]=predict_tree(tree["left"],x[left])
    pred[~left]=predict_tree(tree["right"],x[~left])
    return pred


if MAIN:
    tests.test_predict_tree(predict_tree)

# %%

def tree_leaves(tree):
    return 1 if "value" in tree else tree_leaves(tree["left"])+tree_leaves(tree["right"])


if MAIN:
    tests.test_tree_leaves(tree_leaves)

# %%

def tree_paths(tree, prefix=""):
    if "value" in tree:
        return [(prefix or "always",tree["value"])]
    name=f'x{tree["feature"]}'
    threshold=tree["threshold"]
    left=(prefix+" and " if prefix else "")+f'{name} <= {threshold:g}'
    right=(prefix+" and " if prefix else "")+f'{name} > {threshold:g}'
    return tree_paths(tree["left"],left)+tree_paths(tree["right"],right)


if MAIN:
    tests.test_tree_paths(tree_paths)

# %%

if MAIN:
    train_y=train_x[:,0]**2+.5*train_x[:,1]
    small_tree=fit_sparse_tree(train_x,train_y,penalty=.03,max_depth=4)
    large_tree=fit_sparse_tree(train_x,train_y,penalty=0.,max_depth=5)
    for name,tree in [('penalized',small_tree),('unpenalized depth-5',large_tree)]:
        print(name,'leaves / construction-grid MSE:',tree_leaves(tree),round(mean_squared_error(predict_tree(tree,train_x),train_y),6))
    print('Penalized model rules:')
    for condition,value in tree_paths(small_tree):
        print(f'  IF {condition} THEN {value:.4f}')

# %%

if MAIN:
    eval_axis=np.linspace(-1.,1.,9)
    eval_x=np.array(list(product(eval_axis,eval_axis,[-1.,1.])))
    eval_y=eval_x[:,0]**2+.5*eval_x[:,1]
    linear_w=np.linalg.lstsq(np.column_stack([np.ones(len(train_x)),train_x]),train_y,rcond=None)[0]
    linear_pred=np.column_stack([np.ones(len(eval_x)),eval_x])@linear_w
    spline_pred=additive_design(eval_x[:,:2],knots)@spline_coefficients
    wrong_feature_w=fit_additive_spline(train_x[:,2:3],train_y,knots)
    wrong_feature_pred=additive_design(eval_x[:,2:3],knots)@wrong_feature_w
    symbolic_pred=square_params[0]+square_params[1]*eval_x[:,0]**2+linear_params[0]+linear_params[1]*eval_x[:,1]+spline_coefficients[0]
    model_predictions={'linear baseline':linear_pred,'irrelevant-only control':wrong_feature_pred,'additive spline':spline_pred,'symbolic candidate':symbolic_pred,'penalized tree':predict_tree(small_tree,eval_x),'larger tree':predict_tree(large_tree,eval_x)}
    for name,prediction in model_predictions.items():
        print(f'{name:25s} interpolation MSE {mean_squared_error(prediction,eval_y):.6f}')
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    curve=np.linspace(-1.,1.,101)
    curve_x=np.column_stack([curve,np.zeros_like(curve)])
    learned_edge=edge_values(curve_x,spline_coefficients,knots)[:,0]
    axes[0].plot(curve,curve**2,'k--',label='known square oracle')
    axes[0].plot(curve,learned_edge,label='fitted degree-one edge')
    axes[0].plot(curve,wrong_params[0]+wrong_params[1]*curve,':',label='wrong linear symbolic control')
    axes[0].scatter(knots,square_edge,color='C0',s=25,label='construction knots')
    axes[0].set(xlabel='First input',ylabel='Square-edge contribution',title='An inspectable edge is an approximation')
    axes[0].legend(fontsize=8)
    for name in ['linear baseline','additive spline','penalized tree','irrelevant-only control']:
        axes[1].scatter(eval_y,model_predictions[name],s=12,alpha=.45,label=name)
    axes[1].plot([-.5,1.5],[-.5,1.5],'k--',label='oracle identity')
    axes[1].set(xlabel='Known target',ylabel='Prediction',title='Same public interpolation grid')
    axes[1].legend(fontsize=7)
    names=list(model_predictions)
    errors=[mean_squared_error(model_predictions[name],eval_y) for name in names]
    axes[2].barh(names,errors)
    axes[2].set(xlabel='Mean squared error (162 public rows)',title='Controls beside model-class comparison')
    fig.tight_layout();plt.show()
    print('Complexity currencies: linear',len(linear_w),'coefficients; spline',len(spline_coefficients),'coefficients;',tree_leaves(small_tree),'and',tree_leaves(large_tree),'tree leaves; symbolic two candidate forms plus fitted scales/offsets.')

# %%

if MAIN:
    interaction=1.0  # Change the task, not merely the random seed.
    train_changed=train_y+interaction*train_x[:,0]*train_x[:,1]
    eval_changed=eval_y+interaction*eval_x[:,0]*eval_x[:,1]
    changed_coefficients=fit_additive_spline(train_x[:,:2],train_changed,knots)
    changed_pred=additive_design(eval_x[:,:2],knots)@changed_coefficients
    changed_tree=fit_sparse_tree(train_x,train_changed,penalty=.01,max_depth=5)
    polynomial_oracle=eval_x[:,0]**2+.5*eval_x[:,1]+interaction*eval_x[:,0]*eval_x[:,1]
    print('Changed-task MSE: additive / tree / known polynomial',mean_squared_error(changed_pred,eval_changed),mean_squared_error(predict_tree(changed_tree,eval_x),eval_changed),mean_squared_error(polynomial_oracle,eval_changed))
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,pred,title in zip(axes,[changed_pred,polynomial_oracle],['Additive model: missing interaction','Known polynomial control']):
        residual=(pred-eval_changed).reshape(9,9,2).mean(-1)
        im=ax.imshow(residual,origin='lower',extent=[-1,1,-1,1],vmin=-1.1,vmax=1.1,cmap='coolwarm')
        ax.set(xlabel='Second input',ylabel='First input',title=title)
    fig.colorbar(im,ax=list(axes),label='Prediction minus known interaction target')
    plt.show()

# %%
