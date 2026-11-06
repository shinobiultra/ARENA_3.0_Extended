# %%


from pathlib import Path
import sys
import itertools
import numpy as np
import torch as t
import torch.nn.functional as F
import matplotlib.pyplot as plt
chapter = "chapter14_jepa_world_models"
section = "part7_recurrent_world_models"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.insert(0, str(exercises_dir))
import part7_recurrent_world_models.tests as tests
MAIN = __name__ == "__main__"
device = t.device("cpu")
np.set_printoptions(precision=4, suppress=True)

# %%

def chain_step(state, action):
    next_state=np.clip(np.asarray(state)+np.asarray(action)-1,0,4).astype(int)
    return next_state,(next_state==4).astype(float)


if MAIN:
    tests.test_chain_step(chain_step)

# %%

def straight_through_state(logits, sample=True):
    probs=logits.softmax(dim=-1)
    if sample:
        index=t.multinomial(probs.reshape(-1,probs.shape[-1]),1).reshape(probs.shape[:-1])
    else:
        index=probs.argmax(dim=-1)
    hard=F.one_hot(index,probs.shape[-1]).to(probs.dtype)
    return probs+(hard-probs).detach()


if MAIN:
    tests.test_straight_through_state(straight_through_state)

# %%

def categorical_kl(posterior_logits, prior_logits):
    logq=posterior_logits.log_softmax(dim=-1)
    logp=prior_logits.log_softmax(dim=-1)
    return (logq.exp()*(logq-logp)).sum(dim=-1)


if MAIN:
    tests.test_categorical_kl(categorical_kl)

# %%

def rssm_transition(model, h, z, action, observation=None, sample=True):
    a=F.one_hot(action.long(),3).to(h.dtype)
    h=model["memory"](t.cat([z,a],dim=-1),h)
    prior=model["prior"](h)
    posterior=None if observation is None else model["posterior"](t.cat([h,observation],dim=-1))
    new_z=straight_through_state(prior if posterior is None else posterior,sample=sample)
    return h,new_z,prior,posterior


if MAIN:
    tests.test_rssm_transition(rssm_transition)

# %%

def balanced_kl(posterior, prior, free_nats=0.1):
    dynamics=categorical_kl(posterior.detach(),prior).clamp_min(free_nats).mean()
    representation=categorical_kl(posterior,prior.detach()).clamp_min(free_nats).mean()
    return .5*dynamics+.1*representation


if MAIN:
    tests.test_balanced_kl(balanced_kl)

# %%

def lambda_returns(rewards, values, gamma=0.95, lam=0.8):
    out=np.empty_like(rewards,dtype=float)
    carry=float(values[-1])
    for i in range(len(rewards)-1,-1,-1):
        carry=float(rewards[i])+gamma*((1-lam)*float(values[i+1])+lam*carry)
        out[i]=carry
    return out


if MAIN:
    tests.test_lambda_returns(lambda_returns)

# %%

def make_rssm():
    return t.nn.ModuleDict({"memory":t.nn.GRUCell(8,8),"prior":t.nn.Linear(8,5),"posterior":t.nn.Linear(13,5),"decode":t.nn.Linear(13,5),"reward":t.nn.Linear(13,1)})

if MAIN:
    previous_threads=t.get_num_threads();t.set_num_threads(1);t.manual_seed(43)
    # Every initial room paired with all 27 action sequences of length three.
    action_sequences=np.array(list(itertools.product(range(3),repeat=3)))
    actions_np=np.tile(action_sequences,(5,1));starts=np.repeat(np.arange(5),len(action_sequences))
    room_trace=[starts]
    for i in range(3):room_trace.append(chain_step(room_trace[-1],actions_np[:,i])[0])
    room_trace=np.stack(room_trace,axis=1)
    obs=F.one_hot(t.tensor(room_trace),5).float();acts=t.tensor(actions_np)
    rssm=make_rssm();optimizer=t.optim.Adam(rssm.parameters(),lr=.025);rssm_losses=[]
    for update in range(80):
        h=t.zeros(len(starts),8);z=obs[:,0];loss=t.tensor(0.)
        for i in range(3):
            h,z,prior,posterior=rssm_transition(rssm,h,z,acts[:,i],obs[:,i+1],sample=True)
            state=t.cat([h,z],-1)
            reconstruction=F.cross_entropy(rssm["decode"](state),t.tensor(room_trace[:,i+1]))
            reward_target=(obs[:,i+1,4]).unsqueeze(-1)
            loss=loss+reconstruction+F.mse_loss(rssm["reward"](state),reward_target)+balanced_kl(posterior,prior)
        optimizer.zero_grad();loss.backward();optimizer.step();rssm_losses.append(float(loss.detach()/3))
    print("RSSM mean step loss initial/final:",round(rssm_losses[0],5),round(rssm_losses[-1],5))

# %%

def linear_room_probe(features, room_labels, fitting_mask):
    labels=np.asarray(room_labels,dtype=int)
    fit=np.asarray(fitting_mask,dtype=bool)
    design=np.c_[features,np.ones(len(features))]
    target=np.eye(5)[labels]
    weights=np.linalg.lstsq(design[fit],target[fit],rcond=None)[0]
    prediction=(design@weights).argmax(axis=-1)
    accuracy=float(np.mean(prediction[~fit]==labels[~fit]))
    return prediction,accuracy


if MAIN:
    tests.test_linear_room_probe(linear_room_probe)

# %%

if MAIN:
    probe_features=[];probe_labels=[]
    h=t.zeros(len(starts),8);z=obs[:,0]
    with t.no_grad():
        for i in range(3):
            h,z,_,_=rssm_transition(rssm,h,z,acts[:,i],obs[:,i+1],sample=False)
            probe_features.append(t.cat([h,z],-1).numpy());probe_labels.append(room_trace[:,i+1])
    probe_features=np.concatenate(probe_features);probe_labels=np.concatenate(probe_labels)
    probe_fit=np.arange(len(probe_labels))%2==0
    _,probe_accuracy=linear_room_probe(probe_features,probe_labels,probe_fit)
    shuffled_probe_labels=probe_labels.copy()
    shuffled_probe_labels[probe_fit]=np.random.default_rng(49).permutation(shuffled_probe_labels[probe_fit])
    _,probe_control=linear_room_probe(probe_features,shuffled_probe_labels,probe_fit)
    print("frozen posterior-state probe / shuffled-label control:",round(probe_accuracy,4),round(probe_control,4))
    fig,ax=plt.subplots(figsize=(6,3))
    ax.bar(["true fitting labels","shuffled fitting labels"],[probe_accuracy,probe_control])
    ax.axhline(.2,color="grey",ls="--",label="uniform five-room reference")
    ax.set(ylim=(0,1.05),ylabel="public check-row room accuracy",title="What is linearly decodable from frozen RSSM states?");ax.legend(fontsize=8)
    fig.tight_layout();plt.show()

# %%

def simnorm(x, group_size=5):
    if x.shape[-1] % group_size:
        raise ValueError("feature size must divide into equal simplex groups")
    shape=x.shape
    return x.reshape(*shape[:-1],-1,group_size).softmax(dim=-1).reshape(shape)


if MAIN:
    tests.test_simnorm(simnorm)

# %%

def td_world_loss(predicted_z, target_z, predicted_reward, reward, q, next_q, gamma=0.95):
    consistency=F.mse_loss(predicted_z,target_z.detach())
    reward_loss=F.mse_loss(predicted_reward,reward)
    target=reward+gamma*next_q.detach()
    value_loss=F.mse_loss(q,target)
    return consistency+reward_loss+value_loss


if MAIN:
    tests.test_td_world_loss(td_world_loss)

# %%

def td_latent(room):
    return simnorm(4*F.one_hot(room.long(),5).float())

def make_td():
    def head(out):return t.nn.Sequential(t.nn.Linear(8,24),t.nn.Tanh(),t.nn.Linear(24,out))
    return t.nn.ModuleDict({"dynamics":head(5),"reward":head(1),"q":head(1)})

def td_components(model,z,a):
    inputs=t.cat([z,F.one_hot(a.long(),3).float()],-1)
    return simnorm(model["dynamics"](inputs)),model["reward"](inputs).squeeze(-1),model["q"](inputs).squeeze(-1)

if MAIN:
    t.manual_seed(44)
    states=t.arange(5).repeat_interleave(3);actions=t.arange(3).repeat(5)
    ns,reward_np=chain_step(states.numpy(),actions.numpy());next_states=t.tensor(ns);rewards=t.tensor(reward_np,dtype=t.float32)
    z,z_next=td_latent(states),td_latent(next_states)
    td=make_td();target_td=make_td();target_td.load_state_dict(td.state_dict())
    optimizer=t.optim.Adam(td.parameters(),lr=.02);td_losses=[]
    for update in range(120):
        predicted_z,predicted_r,q=td_components(td,z,actions)
        with t.no_grad():
            next_values=t.stack([td_components(target_td,z_next,t.full_like(actions,a))[2] for a in range(3)],-1).max(-1).values
        loss=td_world_loss(predicted_z,z_next,predicted_r,rewards,q,next_values)
        optimizer.zero_grad();loss.backward();optimizer.step();td_losses.append(float(loss.detach()))
        with t.no_grad():
            for target,source in zip(target_td.parameters(),td.parameters(),strict=True):target.lerp_(source,.05)
    print("TD objective initial/final:",round(td_losses[0],5),round(td_losses[-1],5))

# %%

def plan_latent_actions(model, initial_room, horizon=5, gamma=.95, reward_bonus=None):
    sequences=np.array(list(itertools.product(range(3),repeat=horizon)))
    actions=t.tensor(sequences,dtype=t.long)
    z=td_latent(t.full((len(sequences),),initial_room,dtype=t.long))
    returns=t.zeros(len(sequences));trace=[z.argmax(-1).numpy()]
    with t.no_grad():
        for i in range(horizon):
            next_z,reward,_=td_components(model,z,actions[:,i])
            if reward_bonus is not None:
                reward=reward+reward_bonus(z,actions[:,i])
            returns=returns+gamma**i*reward
            z=next_z;trace.append(z.argmax(-1).numpy())
        terminal=t.stack([td_components(model,z,t.full((len(sequences),),a,dtype=t.long))[2] for a in range(3)],-1).max(-1).values
        returns=returns+gamma**horizon*terminal
    best=int(returns.argmax())
    return sequences[best],float(returns[best]),np.stack(trace,axis=1)[best]


if MAIN:
    tests.test_plan_latent_actions(plan_latent_actions)

# %%

if MAIN:
    aligned_plan,aligned_value,aligned_imagined=plan_latent_actions(td,0)
    real_trace=[0];real_rewards=[]
    for a in aligned_plan:
        n,r=chain_step(real_trace[-1],a);real_trace.append(int(n));real_rewards.append(float(r))
    fig,axs=plt.subplots(1,2,figsize=(11,3.5))
    axs[0].plot(td_losses,label="consistency+reward+TD");axs[0].axhline(td_losses[0],ls="--",label="untrained objective")
    axs[0].set(xlabel="tiny CPU update",ylabel="objective",title="Decoder-free objective fit");axs[0].legend(fontsize=8)
    axs[1].plot(aligned_imagined,"o-",label="TD latent imagination");axs[1].plot(real_trace,"x--",label="actual rooms");axs[1].plot(np.zeros(6),":",label="stay-at-start control")
    axs[1].set(xlabel="step",ylabel="room",yticks=range(5),title="Actions selected in learned latent space");axs[1].legend(fontsize=8)
    fig.tight_layout();plt.show()
    print("TD selected actions:",aligned_plan,"real rooms:",real_trace,"real undiscounted return:",sum(real_rewards))

# %%

def quantize_observation(encoded, codebook):
    distance=((encoded[...,None,:]-codebook)**2).sum(-1)
    indices=distance.argmin(-1)
    return codebook[indices],indices


if MAIN:
    tests.test_quantize_observation(quantize_observation)

# %%

def vq_loss(encoded, quantized, beta=.25):
    codebook=F.mse_loss(quantized,encoded.detach())
    commitment=F.mse_loss(encoded,quantized.detach())
    return codebook+beta*commitment


if MAIN:
    tests.test_vq_loss(vq_loss)

# %%

def causal_attention(query, key, value):
    length=query.shape[-2]
    scores=query@key.transpose(-1,-2)/np.sqrt(query.shape[-1])
    mask=t.ones(length,length,dtype=t.bool,device=query.device).triu(1)
    weights=scores.masked_fill(mask,-t.inf).softmax(-1)
    return weights@value


if MAIN:
    tests.test_causal_attention(causal_attention)

# %%

def interleave_trace(rooms, actions):
    rooms=t.as_tensor(rooms,dtype=t.long);actions=t.as_tensor(actions,dtype=t.long)
    sequence=t.empty((*rooms.shape[:-1],rooms.shape[-1]+actions.shape[-1]),dtype=t.long)
    sequence[...,0::2]=rooms
    sequence[...,1::2]=actions+5
    return sequence


if MAIN:
    tests.test_interleave_trace(interleave_trace)

# %%

def iris_logits(model, tokens):
    length=tokens.shape[-1]
    x=model["embedding"](tokens)+model["position"](t.arange(length,device=tokens.device))[None]
    x=x+causal_attention(model["query"](x),model["key"](x),model["value"](x))
    x=x+model["mlp"](model["norm"](x))
    return model["readout"](x)


if MAIN:
    tests.test_iris_logits(iris_logits)

# %%

def make_iris():
    return t.nn.ModuleDict({"embedding":t.nn.Embedding(8,16),"position":t.nn.Embedding(16,16),"query":t.nn.Linear(16,16),"key":t.nn.Linear(16,16),"value":t.nn.Linear(16,16),"norm":t.nn.LayerNorm(16),"mlp":t.nn.Sequential(t.nn.Linear(16,32),t.nn.GELU(),t.nn.Linear(32,16)),"readout":t.nn.Linear(16,5)})

if MAIN:
    codebook=t.eye(5)
    quantized,ids=quantize_observation(obs,codebook)
    print("identity-codebook quantization error:",float(vq_loss(obs,quantized)))
    tokens=interleave_trace(ids,acts)
    t.manual_seed(45);iris=make_iris();optimizer=t.optim.Adam(iris.parameters(),lr=.015);iris_losses=[]
    for update in range(100):
        logits=iris_logits(iris,tokens[:,:-1])
        # Positions1,3,5 contain chosen actions; target rooms are2,4,6.
        loss=F.cross_entropy(logits[:,1::2].reshape(-1,5),tokens[:,2::2].reshape(-1))
        optimizer.zero_grad();loss.backward();optimizer.step();iris_losses.append(float(loss.detach()))
    with t.no_grad():
        accuracy=float((iris_logits(iris,tokens[:,:-1])[:,1::2].argmax(-1)==tokens[:,2::2]).float().mean())
    print("IRIS-style token CE initial/final:",round(iris_losses[0],5),round(iris_losses[-1],5),"teacher-forced accuracy:",round(accuracy,5))
    t.set_num_threads(previous_threads)

# %%

def compare_imaginations(rssm, iris, initial_room, chosen_actions):
    h=t.zeros(1,8);z=F.one_hot(t.tensor([initial_room]),5).float()
    recurrent=[initial_room];token_rooms=[initial_room];context=[initial_room];actual=[initial_room]
    with t.no_grad():
        for a in chosen_actions:
            h,z,_,_=rssm_transition(rssm,h,z,t.tensor([int(a)]),None,sample=False)
            recurrent.append(int(rssm["decode"](t.cat([h,z],-1)).argmax(-1).item()))
            context.append(int(a)+5)
            prediction=int(iris_logits(iris,t.tensor([context]))[0,-1].argmax().item())
            context.append(prediction);token_rooms.append(prediction)
            actual.append(int(chain_step(actual[-1],a)[0]))
    return np.array(actual),np.array(recurrent),np.array(token_rooms)

if MAIN:
    # Fixed four-step path: includes a room boundary and a direction reversal.
    chosen=np.array([2,2,0,2])
    actual,rssm_rooms,iris_rooms=compare_imaginations(rssm,iris,0,chosen)
    fig,axs=plt.subplots(1,2,figsize=(11,3.5))
    axs[0].plot(rssm_losses,label="RSSM reconstruction+reward+KL");axs[0].plot(iris_losses,label="IRIS-style state-token CE")
    axs[0].set(xlabel="fixed tiny update",ylabel="different objective units",title="Separate organisms, separate objectives");axs[0].legend(fontsize=7)
    for label,tr in [("actual",actual),("RSSM prior mode",rssm_rooms),("token transformer",iris_rooms),("persistence control",np.zeros(len(actual)))]:
        axs[1].plot(tr,"o-",label=label)
    axs[1].set(xlabel="imagined step",ylabel="room",yticks=range(5),title="Free rollout after teacher-forced fitting");axs[1].legend(fontsize=7)
    fig.tight_layout();plt.show()
    print("actual / RSSM / token rooms:",actual,rssm_rooms,iris_rooms)

# %%

def reward_imagination_gap(imagined_rewards, actual_rewards, gamma=.95):
    weights=gamma**np.arange(len(actual_rewards))
    return float(np.sum(weights*(np.asarray(imagined_rewards)-np.asarray(actual_rewards))))


if MAIN:
    tests.test_reward_imagination_gap(reward_imagination_gap)

# %%

if MAIN:
    def wall_bonus(z,a):return 10*((z.argmax(-1)==0)&(a==0)).float()
    exploit_plan,exploit_score,exploit_rooms=plan_latent_actions(td,0,reward_bonus=wall_bonus)
    audits=[]
    for name,sequence,bonus in [("aligned reward",aligned_plan,None),("corrupted wall reward",exploit_plan,wall_bonus),("stay baseline",np.ones(5,dtype=int),None)]:
        z=td_latent(t.tensor([0]));room=0;imagined_rewards=[];actual_rewards=[];room_path=[0]
        with t.no_grad():
            for a in sequence:
                next_z,r,_=td_components(td,z,t.tensor([int(a)]))
                if bonus is not None:r=r+bonus(z,t.tensor([int(a)]))
                imagined_rewards.append(float(r.item()));z=next_z
                room,real_reward=chain_step(room,a);actual_rewards.append(float(real_reward));room_path.append(int(room))
        discount=.95**np.arange(5)
        audits.append((name,np.dot(discount,imagined_rewards),np.dot(discount,actual_rewards),room_path))
        print(name,"actions",sequence,"imagined reward",round(audits[-1][1],4),"real reward",round(audits[-1][2],4),"gap",round(reward_imagination_gap(imagined_rewards,actual_rewards),4))
    fig,axs=plt.subplots(1,2,figsize=(12,3.5));xx=np.arange(3)
    axs[0].bar(xx-.18,[a[1] for a in audits],.36,label="imagined discounted reward")
    axs[0].bar(xx+.18,[a[2] for a in audits],.36,label="actual discounted reward")
    axs[0].set(xticks=xx,xticklabels=["aligned","corrupted","stay"],ylabel="unbootstrapped reward",title="Planner exploits an injected reward error");axs[0].legend(fontsize=7)
    for name,_,_,path in audits:axs[1].plot(path,"o-",label=name)
    axs[1].set(xlabel="executed step",ylabel="actual room",yticks=range(5),title="Same real environment for every plan");axs[1].legend(fontsize=7)
    fig.tight_layout();plt.show()

# %%

if MAIN:
    def play_bonus(z,a):return .2*((z.argmax(-1)==0)&(a==0)).float()
    play_actions,play_score,play_rooms=plan_latent_actions(td,0,reward_bonus=play_bonus)
    print("small-bonus selected actions:",play_actions,"imagined rooms:",play_rooms)
    print("lambda-return fixture:",lambda_returns(np.array([0.,0.,1.]),np.zeros(4)))

# %%
