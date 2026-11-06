import numpy as np
import torch as t

def passed(name):
    print(f"All tests in {name} passed!")


def test_sudoku_peers(fn):
    p=fn();assert p.shape==(16,16) and p.dtype==bool
    for i in range(16):
     r,c=divmod(i,4)
     expected={4*rr+cc for rr in range(4) for cc in range(4) if (rr==r or cc==c or (rr//2==r//2 and cc//2==c//2)) and (rr,cc)!=(r,c)}
     assert set(np.flatnonzero(p[i]))==expected
    assert np.all(p.sum(1)==7)
    passed("test_sudoku_peers")


def test_clue_tokens(fn):
    x=np.zeros((2,4,4),dtype=int);x[0,0,0]=3
    z=fn(x);assert z.shape==(2,16,4)
    t.testing.assert_close(z[0,0],t.tensor([0.,0.,1.,0.]));assert z[1].sum()==0
    passed("test_clue_tokens")


def test_legal_candidates(fn):
    b=np.array([[1,0,3,4],[3,4,1,2],[2,1,4,3],[4,3,2,1]])
    c=fn(b);assert set(np.flatnonzero(c[1])+1)=={2};assert c.sum()==1
    empty=np.zeros((4,4),int);assert fn(empty).all()
    for r in range(4):
     for col in range(4):
      for digit in range(1,5):
       expected=b[r,col]==0 and digit not in b[r,:] and digit not in b[:,col] and digit not in b[(r//2)*2:(r//2+1)*2,(col//2)*2:(col//2+1)*2]
       assert c[4*r+col,digit-1]==expected
    passed("test_legal_candidates")


def test_valid_partial(fn):
    good=np.array([[1,0,3,4],[3,4,1,2],[2,1,4,3],[4,3,2,1]])
    assert fn(good);bad=good.copy();bad[0,1]=3;assert not fn(bad)
    box=np.zeros((4,4),int);box[0,0]=1;box[1,1]=1;assert not fn(box)
    assert not fn(np.full((4,4),5))
    passed("test_valid_partial")


def test_peer_attention(fn):
    p=np.zeros((16,16),bool)
    for i in range(16):p[i,(i+1)%16]=True
    x=t.arange(64,dtype=t.float32).reshape(16,4)
    c,a=fn(x,p);t.testing.assert_close(c,t.roll(x,-1,0));t.testing.assert_close(a.sum(-1),t.ones(16))
    assert t.all(a[~t.tensor(p)]==0)
    passed("test_peer_attention")


def test_candidate_bce(fn):
    z=t.zeros((1,2,4),requires_grad=True);y=t.tensor([[[1,0,1,0],[0,0,0,0]]]);mask=t.tensor([[True,False]])
    loss=fn(z,y,mask);assert abs(loss.item()-np.log(2))<1e-6;loss.backward()
    t.testing.assert_close(z.grad[0,1],t.zeros(4));t.testing.assert_close(z.grad[0,0],t.tensor([-.125,.125,-.125,.125]))
    passed("test_candidate_bce")


def test_train_candidate_head(fn):
    counts=t.tensor([[[0.,1.,1.,1.],[1.,0.,1.,1.],[1.,1.,0.,1.],[1.,1.,1.,0.]]])
    target=t.eye(4)[None];blanks=t.ones((1,4),dtype=t.bool)
    h,loss=fn(counts,target,blanks,steps=60)
    assert len(loss)==60 and loss[-1]<.2*loss[0]
    assert t.equal((h(counts)>0),target.bool())
    passed("test_train_candidate_head")


def test_naked_single_move(fn):
    b=np.zeros((4,4),int);c=np.zeros((16,4),bool);c[5,2]=True;c[7,0]=True
    assert fn(c,b)==(5,3)
    b.ravel()[5]=2;assert fn(c,b)==(7,1)
    c[7,1]=True;assert fn(c,b) is None
    passed("test_naked_single_move")


def test_apply_move(fn):
    b=np.zeros((4,4),int);o=fn(b,(5,3));assert o[1,1]==3 and b.sum()==0
    try:fn(o,(5,4))
    except ValueError:pass
    else:raise AssertionError("overwriting a clue must fail")
    passed("test_apply_move")


def test_solve_exact(fn):
    b=np.array([[1,0,3,4],[3,4,1,2],[2,1,4,3],[4,3,2,1]])
    a=fn(b);assert a[0,1]==2 and np.all(a>0)
    for unit in list(a)+list(a.T):assert set(unit)=={1,2,3,4}
    b[0,1]=1;assert fn(b) is None
    passed("test_solve_exact")


def test_solve_by_head(fn):
    class OracleHead:
     def __call__(self,x):return .5-x
    b=np.array([[1,0,3,4],[3,4,1,2],[2,1,4,3],[4,3,2,1]])
    states,moves=fn(b,OracleHead());assert moves==[(1,2)] and states[-1,0,1]==2
    class NoCandidates:
     def __call__(self,x):return t.zeros_like(x)
    s,m=fn(b,NoCandidates());assert len(s)==1 and m==[]
    passed("test_solve_by_head")


def test_patch_candidate_state(fn):
    r=np.zeros((16,4),bool);r[1,1]=True;d=np.ones((16,4),bool)
    p=fn(r,d,1,1);np.testing.assert_array_equal(p[1],[1,0,0,0]);assert r[1,1]
    np.testing.assert_array_equal(p[2:],r[2:])
    passed("test_patch_candidate_state")


def test_candidate_agreement(fn):
    p=np.array([[1,0,0,0],[1,1,0,0],[0,0,0,0]],bool);q=np.array([[1,0,0,0],[1,0,0,0],[0,0,0,0]],bool)
    assert fn(p,q,[1,1,0])==.5
    passed("test_candidate_agreement")
