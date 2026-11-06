# %%


from pathlib import Path
import sys
import itertools
import numpy as np
import torch as t
import torch.nn.functional as F
import matplotlib.pyplot as plt
chapter = "chapter14_jepa_world_models"
section = "part6_sudoku_transformers"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.insert(0, str(exercises_dir))
import part6_sudoku_transformers.tests as tests
MAIN = __name__ == "__main__"
device = t.device("cpu")
np.set_printoptions(precision=4, suppress=True)

# %%

def sudoku_peers():
    row, col = np.divmod(np.arange(16), 4)
    same_row = row[:,None] == row[None,:]
    same_col = col[:,None] == col[None,:]
    same_box = (row[:,None]//2 == row[None,:]//2) & (col[:,None]//2 == col[None,:]//2)
    return (same_row | same_col | same_box) & ~np.eye(16,dtype=bool)


if MAIN:
    tests.test_sudoku_peers(sudoku_peers)

# %%

def clue_tokens(boards):
    boards = t.as_tensor(boards, dtype=t.long)
    flat = boards.reshape(*boards.shape[:-2],16)
    return F.one_hot(flat,num_classes=5)[...,1:].to(t.float32)


if MAIN:
    tests.test_clue_tokens(clue_tokens)

# %%

def legal_candidates(board):
    board=np.asarray(board,dtype=int)
    out=np.zeros((16,4),dtype=bool)
    flat=board.ravel();peers=sudoku_peers()
    for cell in range(16):
        if flat[cell] == 0:
            forbidden=set(flat[peers[cell]])
            out[cell]=[digit not in forbidden for digit in range(1,5)]
    return out


if MAIN:
    tests.test_legal_candidates(legal_candidates)

# %%

def valid_partial(board):
    board=np.asarray(board,dtype=int)
    if board.shape != (4,4) or np.any((board<0)|(board>4)):
        return False
    units=list(board)+list(board.T)+[board[r:r+2,c:c+2].ravel() for r in (0,2) for c in (0,2)]
    return all(len(v[v!=0])==len(set(v[v!=0])) for v in units)


if MAIN:
    tests.test_valid_partial(valid_partial)

# %%

if MAIN:
    solved=np.array([[1,2,3,4],[3,4,1,2],[2,1,4,3],[4,3,2,1]])
    puzzle=solved.copy()
    puzzle.ravel()[[0,1,4,6,9,10,12,15]]=0
    print("Cold-open puzzle:\n",puzzle)
    print("Candidate counts:\n",legal_candidates(puzzle).sum(1).reshape(4,4))

# %%

def peer_attention(tokens, peers):
    mask=t.as_tensor(peers,dtype=t.bool,device=tokens.device)
    logits=t.zeros((16,16),dtype=tokens.dtype,device=tokens.device).masked_fill(~mask,-t.inf)
    attention=logits.softmax(dim=-1)
    counts=t.einsum("ij,...jd->...id",attention,tokens)*mask.sum(-1)[:,None]
    return counts, attention


if MAIN:
    tests.test_peer_attention(peer_attention)

# %%

def candidate_bce(logits, candidate_targets, blank_mask):
    per_digit=F.binary_cross_entropy_with_logits(logits,candidate_targets.to(logits.dtype),reduction="none")
    mask=blank_mask.to(logits.dtype)[...,None]
    return (per_digit*mask).sum()/(mask.sum()*logits.shape[-1])


if MAIN:
    tests.test_candidate_bce(candidate_bce)

# %%

def train_candidate_head(counts, targets, blanks, steps=120):
    head=t.nn.Linear(4,4,device="cpu")
    with t.no_grad():
        head.weight.zero_();head.bias.zero_()
    optimizer=t.optim.Adam(head.parameters(),lr=.08)
    losses=[]
    for _ in range(steps):
        optimizer.zero_grad()
        loss=candidate_bce(head(counts),targets,blanks)
        loss.backward();optimizer.step();losses.append(float(loss.detach()))
    return head,np.array(losses)


if MAIN:
    tests.test_train_candidate_head(train_candidate_head)

# %%

if MAIN:
    # Limit CPU threading for this tiny dense fit; preserve the caller's thread count.
    previous_threads=t.get_num_threads();t.set_num_threads(1)
    fitting=[];checking=[]
    masks=[[0,1,4,6,9,10,12,15],[0,2,5,7,8,11,13,14],[1,3,4,7,8,10,12,14],[0,3,5,6,9,11,12,15]]
    for permutation in itertools.permutations([1,2,3,4]):
        mapped=np.array([0,*permutation])[solved]
        for mask in masks:
            b=mapped.copy();b.ravel()[mask]=0;fitting.append(b)
        b=mapped.copy();b.ravel()[[0,2,4,5,7,9,11,14]]=0;checking.append(b)
    fitting,checking=np.array(fitting),np.array(checking)
    peers=sudoku_peers()
    train_counts,attention=peer_attention(clue_tokens(fitting),peers)
    train_targets=t.tensor(np.array([legal_candidates(b) for b in fitting]))
    train_blanks=t.tensor(fitting.reshape(-1,16)==0)
    head,losses=train_candidate_head(train_counts,train_targets,train_blanks)
    check_counts,_=peer_attention(clue_tokens(checking),peers)
    check_targets=t.tensor(np.array([legal_candidates(b) for b in checking]))
    check_blanks=t.tensor(checking.reshape(-1,16)==0)
    with t.no_grad():
        predicted=(head(check_counts)>0)&check_blanks[...,None]
    check_bit_accuracy=float((predicted[check_blanks]==check_targets[check_blanks]).float().mean())
    print("candidate-bit accuracy on public changed-mask check:",check_bit_accuracy)
    print("loss initial/final:",round(float(losses[0]),6),round(float(losses[-1]),6))
    t.set_num_threads(previous_threads)

# %%

if MAIN:
    fig,axs=plt.subplots(1,2,figsize=(10,3.5))
    axs[0].plot(losses,label="fitted candidate head");axs[0].axhline(np.log(2),ls="--",color="grey",label="untrained zero-logit control")
    axs[0].set(xlabel="fixed CPU update",ylabel="blank-cell bit BCE",title="Learn the count→candidate readout");axs[0].legend(fontsize=8)
    im=axs[1].imshow(attention.numpy(),cmap="Blues");fig.colorbar(im,ax=axs[1],label="attention weight")
    axs[1].set(xlabel="source cell",ylabel="destination cell",title="Given row/column/box peer prior")
    fig.tight_layout();plt.show()

# %%

def naked_single_move(candidates, board):
    blank=np.asarray(board).ravel()==0
    counts=candidates.sum(axis=-1)
    rows=np.flatnonzero(blank & (counts==1))
    if len(rows)==0:
        return None
    cell=int(rows[0]);digit=int(np.flatnonzero(candidates[cell])[0]+1)
    return cell,digit


if MAIN:
    tests.test_naked_single_move(naked_single_move)

# %%

def apply_move(board, move):
    out=np.array(board,copy=True)
    cell,digit=move
    if out.ravel()[cell] != 0 or digit not in range(1,5):
        raise ValueError("move must fill a blank with a valid digit")
    out.ravel()[cell]=digit
    return out


if MAIN:
    tests.test_apply_move(apply_move)

# %%

def solve_exact(board):
    if not valid_partial(board):
        return None
    blank=np.flatnonzero(np.asarray(board).ravel()==0)
    if not len(blank):
        return np.array(board,copy=True)
    cell=int(blank[0]);candidates=legal_candidates(board)[cell]
    for digit in np.flatnonzero(candidates)+1:
        answer=solve_exact(apply_move(board,(cell,int(digit))))
        if answer is not None:
            return answer
    return None


if MAIN:
    tests.test_solve_exact(solve_exact)

# %%

def solve_by_head(board, head, limit=16):
    states=[np.array(board,copy=True)];moves=[]
    for _ in range(limit):
        counts,_=peer_attention(clue_tokens(states[-1]),sudoku_peers())
        with t.no_grad():
            candidates=(head(counts)>0).numpy()
        move=naked_single_move(candidates,states[-1])
        if move is None:
            break
        next_board=apply_move(states[-1],move)
        states.append(next_board);moves.append(move)
        if not valid_partial(next_board):
            break
    return np.array(states),moves


if MAIN:
    tests.test_solve_by_head(solve_by_head)

# %%

if MAIN:
    trajectory,moves=solve_by_head(puzzle,head)
    zero_head=t.nn.Linear(4,4)
    with t.no_grad():zero_head.weight.zero_();zero_head.bias.zero_()
    control_states,control_moves=solve_by_head(puzzle,zero_head)
    print("fitted move trajectory:",moves)
    print("remaining blanks fitted/untrained:",int((trajectory[-1]==0).sum()),int((control_states[-1]==0).sum()))
    print("exact completion exists:",solve_exact(puzzle) is not None)
    fig,axs=plt.subplots(1,2,figsize=(10,3.5))
    axs[0].plot([(b==0).sum() for b in trajectory],"o-",label="learned candidate head")
    axs[0].plot(range(len(trajectory)),np.repeat((puzzle==0).sum(),len(trajectory)),"--",label="untrained control")
    axs[0].set(xlabel="proposed move",ylabel="remaining blanks",title="Visible solving trajectory");axs[0].legend(fontsize=8)
    axs[1].imshow(trajectory[-1],vmin=0,vmax=4,cmap="Blues")
    for r in range(4):
        for c in range(4):axs[1].text(c,r,str(trajectory[-1,r,c]) or ".",ha="center",va="center")
    axs[1].set(xticks=range(4),yticks=range(4),xlabel="column",ylabel="row",title="Final fitted-policy board")
    fig.tight_layout();plt.show()

# %%

def patch_candidate_state(recipient, donor, cell, donor_digit):
    out=np.array(recipient,copy=True)
    out[cell]=False
    out[cell,donor_digit-1]=donor[cell,donor_digit-1]
    return out


if MAIN:
    tests.test_patch_candidate_state(patch_candidate_state)

# %%

def candidate_agreement(predicted, truth, blanks):
    selected=np.asarray(blanks,dtype=bool)
    return float(np.mean(np.all(np.asarray(predicted)[selected]==np.asarray(truth)[selected],axis=-1)))


if MAIN:
    tests.test_candidate_agreement(candidate_agreement)

# %%

if MAIN:
    recipient=solved.copy();recipient[0,1]=0
    counts,_=peer_attention(clue_tokens(recipient),peers)
    with t.no_grad():candidate_state=(head(counts)>0).numpy()
    donor=legal_candidates(np.zeros((4,4),int))
    edited=patch_candidate_state(candidate_state,donor,1,1)
    wrong_cell=patch_candidate_state(candidate_state,donor,0,1)
    conditions=[("clean",candidate_state),("candidate patch",edited),("wrong-cell control",wrong_cell)]
    fig,axs=plt.subplots(1,3,figsize=(12,3.2))
    for ax,(name,candidate) in zip(axs,conditions):
        move=naked_single_move(candidate,recipient);result=apply_move(recipient,move)
        ax.imshow(candidate.astype(int),aspect="auto",cmap="Blues",vmin=0,vmax=1)
        ax.set(xticks=range(4),xticklabels=[1,2,3,4],xlabel="candidate digit",ylabel="flat cell",title=f"{name}: move {move}\nlegal={valid_partial(result)}")
        print(name,"move",move,"legal",valid_partial(result))
    fig.tight_layout();plt.show()
    check_exact=candidate_agreement(predicted.numpy().reshape(-1,4),check_targets.numpy().reshape(-1,4),check_blanks.numpy().reshape(-1))
    print("changed-mask exact candidate-set agreement:",check_exact)

# %%

if MAIN:
    play=solved.copy();play[[0,1,2,3],[0,1,2,3]]=0
    play_states,play_moves=solve_by_head(play,head)
    print("play moves:",play_moves)
    print("play remaining blanks:",int((play_states[-1]==0).sum()))
    print("play completion exists:",solve_exact(play) is not None)

# %%
