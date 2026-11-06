"""Independent mathematical oracles for sparse fitting, spline edges and greedy trees."""
import numpy as np


def passed(name):
    print(f"All tests in {name} passed!")


def test_mean_squared_error(fn):
    assert fn([[1,3],[2,4]],np.zeros((2,2)))==7.5
    assert fn([1,2],[1,2])==0
    passed('test_mean_squared_error')


def test_connection_cost(fn):
    np.testing.assert_allclose(fn([-1,1,0],[-1,1]),[[1,3],[3,1],[2,2]])
    np.testing.assert_allclose(fn([7],[7]),[[1]])
    passed('test_connection_cost')


def test_soft_threshold(fn):
    np.testing.assert_allclose(fn(np.array([-2.,-.1,0,.8]),.2),[-1.8,0,0,.6])
    np.testing.assert_allclose(fn(np.array([[2.,-2.],[.2,-.2]]),[1,.5]),[[1,-1.5],[0,0]])
    passed('test_soft_threshold')


def test_proximal_fit(fn):
    x=np.array([[1,1],[1,-1],[-1,1],[-1,-1]],dtype=float)
    teacher=np.array([[2.,-.1],[.05,-1.]])
    y=x@teacher;cost=np.array([[1.,3.],[4.,1.]])
    actual=fn(x,y,cost,.1)
    # X.T X/n is identity: each coordinate solves a scalar quadratic+L1 objective.
    expected=np.array([[1.9,0.],[0.,-.9]])
    np.testing.assert_allclose(actual,expected,atol=1e-10)
    np.testing.assert_allclose(fn(x,y,cost,0.),teacher,atol=1e-10)
    np.testing.assert_array_equal(x,np.array([[1,1],[1,-1],[-1,1],[-1,-1]]))
    # Nonwhitened, multi-output data distinguishes finite updates and loss denominators.
    rows=np.array([[1.,2.],[-1.,.5],[.5,-1.]])
    targets=np.array([[2.,-1.],[.3,1.5],[-.7,.2]])
    costs=np.array([[1.,2.],[.5,3.]])
    strength=.17
    def scalar_iteration(steps,rate):
        weights=[[0.,0.],[0.,0.]]
        for _ in range(steps):
            residual=[[sum(rows[i,j]*weights[j][o] for j in range(2))-targets[i,o] for o in range(2)] for i in range(3)]
            updated=[[0.,0.],[0.,0.]]
            for j in range(2):
                for o in range(2):
                    gradient=sum(rows[i,j]*residual[i][o] for i in range(3))/3
                    proposal=weights[j][o]-rate*gradient
                    threshold=rate*strength*costs[j,o]
                    updated[j][o]=max(proposal-threshold,0)-max(-proposal-threshold,0)
            weights=updated
        return np.array(weights)
    for steps in (0,1,2,7):
        for rate in (0.,.1,.3):
            np.testing.assert_allclose(fn(rows,targets,costs,strength,steps=steps,learning_rate=rate),scalar_iteration(steps,rate),rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(rows,[[1.,2.],[-1.,.5],[.5,-1.]])
    np.testing.assert_array_equal(targets,[[2.,-1.],[.3,1.5],[-.7,.2]])
    np.testing.assert_array_equal(costs,[[1.,2.],[.5,3.]])
    passed('test_proximal_fit')


def test_off_module_fraction(fn):
    assert np.isclose(fn(np.array([[2.,-1.],[0,1.]]),np.eye(2,dtype=bool)),.25)
    assert fn(np.zeros((2,2)),np.eye(2,dtype=bool))==0
    passed('test_off_module_fraction')


def test_hat_basis(fn):
    knots=np.array([-1.,-.2,0.,.7,1.])
    values=np.array([[-1.,-.8,0.],[.1,.8,1.]])
    expected=np.stack([np.interp(values,knots,np.eye(5)[k]) for k in range(5)],axis=-1)
    np.testing.assert_allclose(fn(values,knots),expected,atol=1e-14)
    np.testing.assert_allclose(fn(knots,knots),np.eye(5))
    np.testing.assert_allclose(fn(values,knots).sum(-1),np.ones(values.shape))
    for v,k in [([1.1],knots),([0],[-1,0,0,1])]:
        try:
            fn(v,k)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid domain or repeated knots accepted')
    passed('test_hat_basis')


def test_additive_design(fn):
    k=np.array([-1.,0.,1.]);x=np.array([[0,0],[-1,1],[.5,-.5]])
    np.testing.assert_allclose(fn(x,k),[[1,0,0,0,0],[1,1,0,0,1],[1,0,.5,.5,0]])
    passed('test_additive_design')


def test_fit_additive_spline(fn):
    x=np.array([[a,b] for a in [-1.,0.,1.] for b in [-1.,0.,1.]])
    coeff=fn(x,x[:,0]+2*x[:,1],np.array([-1.,0.,1.]))
    np.testing.assert_allclose(coeff,[0,-1,1,-2,2],atol=1e-13)
    passed('test_fit_additive_spline')


def test_edge_values(fn):
    k=np.array([-1.,0.,1.]);coeff=np.array([4.,-1.,1.,-2.,2.]);x=np.array([[.25,-.5],[0,0],[-1,1.]])
    np.testing.assert_allclose(fn(x,coeff,k),[[.25,-1.],[0,0],[-1,2.]],atol=1e-14)
    # Shared intercept must not be duplicated into each edge.
    passed('test_edge_values')


def test_symbolic_affine_fit(fn):
    x=np.array([-1.,-.5,0.,.5,1.]);edge=3+2*x*x
    params,mse=fn(x,edge,np.square)
    np.testing.assert_allclose(params,[3,2],atol=1e-13)
    assert mse<1e-24
    wrong,wrong_mse=fn(x,x*x,lambda z:z)
    np.testing.assert_allclose(wrong,[.5,0],atol=1e-13)
    assert np.isclose(wrong_mse,.175)
    passed('test_symbolic_affine_fit')


def test_best_split(fn):
    x=np.column_stack([np.arange(4),np.zeros(4)])
    f,t,g=fn(x,[0,0,2,2])
    assert f==0 and t==1.5 and np.isclose(g,4)
    assert fn(x,np.ones(4))==(None,None,0.)
    # Equal informative columns must choose feature 0, not an arbitrary last tie.
    assert fn(np.column_stack([np.arange(4),np.arange(4)]),[0,0,2,2])[0]==0
    passed('test_best_split')


def _independent_predict(tree,rows):
    values=[]
    for row in rows:
        node=tree
        while 'value' not in node:
            node=node['left'] if row[node['feature']]<=node['threshold'] else node['right']
        values.append(node['value'])
    return np.asarray(values)


def test_fit_sparse_tree(fn):
    x=np.arange(4.)[:,None];y=np.array([0.,0.,2.,2.])
    tree=fn(x,y,penalty=.9,max_depth=4)
    assert tree['feature']==0 and tree['threshold']==1.5
    np.testing.assert_allclose(_independent_predict(tree,x),y)
    assert fn(x,y,penalty=1.,max_depth=4)=={'value':1.}
    assert fn(x,y,penalty=0.,max_depth=0)=={'value':1.}
    # Root-normalized gain on a small descendant must not be exaggerated.
    x=np.arange(8.)[:,None];y=np.array([0,0,0,0,10,10,11,11],dtype=float)
    tree=fn(x,y,penalty=.2,max_depth=4)
    right=tree['right']
    assert 'value' in right, 'child gain divided by child size instead of root size'
    passed('test_fit_sparse_tree')


def test_predict_tree(fn):
    tree={'feature':0,'threshold':0.,'left':{'value':-1.},'right':{'value':2.}}
    np.testing.assert_allclose(fn(tree,np.array([[.1],[-.1],[0.]])),[2,-1,-1])
    assert fn(tree,np.empty((0,1))).shape==(0,)
    passed('test_predict_tree')


def test_tree_leaves(fn):
    leaf={'value':0.};stump={'feature':0,'threshold':0,'left':leaf,'right':leaf}
    assert fn(leaf)==1 and fn(stump)==2
    assert fn({'feature':0,'threshold':1,'left':stump,'right':leaf})==3
    passed('test_tree_leaves')


def test_tree_paths(fn):
    tree={'feature':0,'threshold':0.,'left':{'value':-1.},'right':{'value':2.}}
    assert fn(tree)==[('x0 <= 0',-1.),('x0 > 0',2.)]
    assert fn({'value':3})==[('always',3)]
    passed('test_tree_paths')
