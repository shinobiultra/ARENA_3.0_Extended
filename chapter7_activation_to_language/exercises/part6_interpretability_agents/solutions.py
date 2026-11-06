# %%


from pathlib import Path
import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

chapter = "chapter7_activation_to_language"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
if str(exercises_dir) not in sys.path:
    sys.path.insert(0, str(exercises_dir))
import part6_interpretability_agents.tests as tests

MAIN = __name__ == "__main__"
# NumPy arrays stay on CPU; float64 makes small analytic comparisons transparent.

# %%

def render_scene(scene):
    """Render a 7×7 RGB controlled object on an explicit background."""
    if set(scene)!={"color","shape","background"} or scene["color"] not in {"red","blue"} or scene["shape"] not in {"circle","square"} or scene["background"] not in {"plain","striped"}:
        raise ValueError("Invalid controlled scene")
    image=np.zeros((7,7,3),dtype=float)
    if scene["background"]=="striped":image[::2,0,:]=.5
    mask=np.ones((3,3),bool) if scene["shape"]=="square" else np.array([[0,1,0],[1,1,1],[0,1,0]],bool)
    color=np.array([1.,0.,0.]) if scene["color"]=="red" else np.array([0.,0.,1.])
    image[2:5,2:5][mask]=color
    return image


if MAIN:
    tests.test_render_scene(render_scene)

# %%

def image_features(image):
    """Extract red, circle and stripe indicators from the controlled renderer."""
    if np.asarray(image).shape!=(7,7,3) or not np.isfinite(image).all():
        raise ValueError("Expected finite 7×7 RGB image")
    center=image[3,3]
    corners=image[[2,2,4,4],[2,4,2,4]]
    return np.array([float(center[0]>center[2]),float(center.sum()>0 and np.all(corners==0)),float(np.any(image[:,0]>0))])


if MAIN:
    tests.test_image_features(image_features)

# %%

def hypothesis_predictions(name, features):
    """Predict unit activation from one frozen candidate explanation."""
    if name=="red":return features[:,0]
    if name=="circle":return features[:,1]
    if name=="red-and-circle":return features[:,0]*features[:,1]
    if name=="stripes":return features[:,2]
    raise ValueError("Unknown frozen hypothesis")


if MAIN:
    tests.test_hypothesis_predictions(hypothesis_predictions)

# %%

def simulation_error(predicted, observed):
    """Mean squared activation error over every frozen evaluation example."""
    p=np.asarray(predicted);y=np.asarray(observed)
    if p.shape!=y.shape or p.ndim!=1 or len(p)==0 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Finite aligned nonempty predictions required")
    return float(np.mean((p-y)**2))


if MAIN:
    tests.test_simulation_error(simulation_error)

# %%

if MAIN:
    SCENES=[dict(color=c,shape=s,background=b) for c in ["red","blue"] for s in ["circle","square"] for b in ["plain","striped"]]
    HYPOTHESES=["red","circle","red-and-circle","stripes"]
    def target_unit(features):
        return np.maximum(features[0]+features[1]-1.,0.)
    initial_scenes=[SCENES[0],SCENES[1]]
    initial_history=[dict(scene=dict(scene),activation=float(target_unit(image_features(render_scene(scene))))) for scene in initial_scenes]

# %%

if MAIN:
    fig,axes=plt.subplots(1,4,figsize=(10,3))
    for ax,scene in zip(axes,[SCENES[0],SCENES[2],SCENES[4],SCENES[6]]):
        activation=target_unit(image_features(render_scene(scene)))
        ax.imshow(render_scene(scene));ax.set_title(f"{scene['color']} {scene['shape']}\nactivation {activation:g}");ax.axis("off")
    fig.suptitle("Change color or shape; plain background is fixed");fig.tight_layout();plt.show()
    display(pd.DataFrame(initial_history))

# %%

def validate_action(action):
    """Validate a finite activation/ablation/steering action without execution."""
    if not isinstance(action,dict) or action.get("tool") not in {"activation","ablate","steer"}:
        raise ValueError("Unknown tool")
    expected={"tool","scene"}|({"feature"} if action["tool"]!="activation" else set())|({"amount"} if action["tool"]=="steer" else set())
    if set(action)!=expected or not isinstance(action["scene"],dict):raise ValueError("Invalid tool schema")
    render_scene(action["scene"])
    if "feature" in action and (type(action["feature"]) is not int or not 0<=action["feature"]<3):raise ValueError("Invalid feature")
    if "amount" in action and (type(action["amount"]) not in (int,float) or not np.isfinite(action["amount"]) or not -1<=action["amount"]<=1):raise ValueError("Invalid steering dose")
    return {**action,"scene":dict(action["scene"])}


if MAIN:
    tests.test_validate_action(validate_action)

# %%

def execute_tool(action, target):
    """Execute one validated local tool call against a supplied target."""
    action=validate_action(action);features=image_features(render_scene(action["scene"]))
    if action["tool"]=="ablate":features[action["feature"]]=0
    if action["tool"]=="steer":features[action["feature"]]=np.clip(features[action["feature"]]+action["amount"],0,1)
    value=float(target(features))
    if not np.isfinite(value):raise ValueError("Target returned a nonfinite activation")
    return dict(action=action,activation=value)


if MAIN:
    tests.test_execute_tool(execute_tool)

# %%

def parse_backend_action(text):
    """Parse JSON and validate an action; never evaluate model-produced code."""
    def pairs(items):
        out={}
        for key,value in items:
            if key in out:raise ValueError("Duplicate JSON key")
            out[key]=value
        return out
    def nonfinite(value):raise ValueError("Nonfinite JSON constant")
    try:action=json.loads(text,object_pairs_hook=pairs,parse_constant=nonfinite)
    except (TypeError,json.JSONDecodeError) as exc:raise ValueError("Expected one JSON action") from exc
    return validate_action(action)


if MAIN:
    tests.test_parse_backend_action(parse_backend_action)

# %%

if MAIN:
    scene=dict(color="red",shape="circle",background="striped")
    actions=[dict(tool="activation",scene=scene),dict(tool="ablate",scene=scene,feature=0),dict(tool="ablate",scene=scene,feature=2),dict(tool="steer",scene=scene,feature=0,amount=-.5)]
    observations=[execute_tool(a,target_unit)["activation"] for a in actions]
    fig,ax=plt.subplots(figsize=(8,3));ax.bar(["no edit","remove red","remove stripe control","red dose −.5"],observations);ax.set(ylabel="unit activation",title="The planted conjunction needs red, not a stripe",ylim=(0,1.1));ax.tick_params(axis="x",rotation=15);fig.tight_layout();plt.show()

# %%

def consistent_hypotheses(candidates, history, tolerance=1e-9):
    """Retain candidates consistent with every activation observation."""
    if tolerance<0:raise ValueError("Nonnegative tolerance required")
    if not history:return list(candidates)
    features=np.stack([image_features(render_scene(row["scene"])) for row in history])
    observed=np.array([row["activation"] for row in history])
    return [name for name in candidates if np.all(np.abs(hypothesis_predictions(name,features)-observed)<=tolerance)]


if MAIN:
    tests.test_consistent_hypotheses(consistent_hypotheses)

# %%

def probe_disagreement(candidates, scenes):
    """Variance across candidate predictions for each proposed scene."""
    if not candidates:raise ValueError("No candidate family remains")
    features=np.stack([image_features(render_scene(scene)) for scene in scenes])
    predictions=np.stack([hypothesis_predictions(name,features) for name in candidates])
    return predictions.var(axis=0)


if MAIN:
    tests.test_probe_disagreement(probe_disagreement)

# %%

def choose_probe(candidates, scenes, history):
    """Choose an unobserved maximally disagreeing scene, or return None."""
    unseen=[scene for scene in scenes if not any(scene==row["scene"] for row in history)]
    if not unseen:return None
    scores=probe_disagreement(candidates,unseen)
    if scores.max()<=0:return None
    return dict(unseen[int(scores.argmax())])


if MAIN:
    tests.test_choose_probe(choose_probe)

# %%

def run_agent(target, candidates, scenes, initial_history, budget=2):
    """Bounded hypothesis–probe–observation loop with a retained trace."""
    if type(budget) is not int or budget<0:raise ValueError("Nonnegative integer budget required")
    history=[dict(scene=dict(row["scene"]),activation=float(row["activation"])) for row in initial_history]
    remaining=consistent_hypotheses(candidates,history);rounds=[];calls=0
    while len(remaining)>1 and calls<budget:
        scene=choose_probe(remaining,scenes,history)
        if scene is None:break
        before=list(remaining)
        observation=execute_tool(dict(tool="activation",scene=scene),target);calls+=1
        row=dict(scene=observation["action"]["scene"],activation=observation["activation"])
        history.append(row);remaining=consistent_hypotheses(remaining,history)
        rounds.append(dict(before=before,scene=dict(scene),activation=row["activation"],after=list(remaining)))
    reason="identified" if len(remaining)==1 else "contradicted" if not remaining else "budget_exhausted" if calls>=budget else "indistinguishable"
    return dict(history=history,rounds=rounds,remaining=remaining,explanation=remaining[0] if len(remaining)==1 else None,reason=reason,calls=calls)


if MAIN:
    tests.test_run_agent(run_agent)

# %%

if MAIN:
    agent=run_agent(target_unit,HYPOTHESES,SCENES,initial_history,budget=2)
    display(pd.DataFrame(agent["rounds"]))
    fig,axes=plt.subplots(1,max(1,len(agent["rounds"])),figsize=(8,3));axes=np.atleast_1d(axes)
    for ax,row in zip(axes,agent["rounds"]):
        ax.imshow(render_scene(row["scene"]));ax.set_title(f"{row['scene']['color']} {row['scene']['shape']}\nresponse {row['activation']:g}");ax.axis("off")
    fig.suptitle("Actual counterfactual probes, not more top activations");fig.tight_layout();plt.show()
    print("result",agent["reason"],agent["explanation"],"additional calls",agent["calls"])
    print("zero-budget control",run_agent(target_unit,HYPOTHESES,SCENES,initial_history,0)["reason"])

# %%

def one_shot_explanation(candidates, history):
    """Choose the first consistent candidate without additional target calls."""
    remaining=consistent_hypotheses(candidates,history)
    return remaining[0] if remaining else None


if MAIN:
    tests.test_one_shot_explanation(one_shot_explanation)

# %%

def intervention_effect(scene, feature, target, amount=None):
    """Compare one feature edit against its own unedited activation."""
    base=execute_tool(dict(tool="activation",scene=scene),target)["activation"]
    action=dict(tool="ablate",scene=scene,feature=feature) if amount is None else dict(tool="steer",scene=scene,feature=feature,amount=amount)
    return base-execute_tool(action,target)["activation"]


if MAIN:
    tests.test_intervention_effect(intervention_effect)

# %%

def score_explanations(explanations, scenes, target):
    """Retain per-item errors and explicit abstentions for a frozen cohort."""
    features=np.stack([image_features(render_scene(scene)) for scene in scenes])
    observed=np.array([float(target(f)) for f in features])
    methods={}
    for name,explanation in explanations.items():
        if explanation is None:methods[name]=dict(predicted=None,mse=None,abstained=True)
        else:
            predicted=hypothesis_predictions(explanation,features)
            methods[name]=dict(predicted=predicted.tolist(),mse=simulation_error(predicted,observed),abstained=False)
    return dict(scenes=[dict(scene) for scene in scenes],observed=observed.tolist(),methods=methods)


if MAIN:
    tests.test_score_explanations(score_explanations)

# %%

if MAIN:
    probe_pool=[scene for scene in SCENES if scene["background"]=="plain"]
    evaluation=[scene for scene in SCENES if scene["background"]=="striped"]
    training_history=[dict(scene=SCENES[0],activation=1.)]
    final_agent=run_agent(target_unit,HYPOTHESES,probe_pool,training_history,2)
    one_shot=one_shot_explanation(HYPOTHESES,training_history)
    random_name=HYPOTHESES[int(np.random.default_rng(0).integers(len(HYPOTHESES)))]
    explanations={"two-probe controller":final_agent["explanation"],"one-shot":one_shot,"fixed random explanation":random_name}
    report=score_explanations(explanations,evaluation,target_unit)
    display(pd.DataFrame({"scene":[str(s) for s in evaluation],"observed":report["observed"],**{name:item["predicted"] for name,item in report["methods"].items()}}))
    fig,axes=plt.subplots(1,2,figsize=(11,4));names=list(explanations)
    axes[0].bar(names,[report["methods"][n]["mse"] for n in names]);axes[0].set(ylabel="mean squared error / 4 striped scenes",ylim=(0,1.05),title="Simulation checks the explanation");axes[0].tick_params(axis="x",rotation=20)
    scene=dict(color="red",shape="circle",background="striped")
    axes[1].bar(["red edit","shape edit","stripe control"],[intervention_effect(scene,j,target_unit) for j in range(3)]);axes[1].set(ylabel="baseline − ablated activation",title="The causal check needs two object properties");axes[1].tick_params(axis="x",rotation=20);fig.tight_layout();plt.show()
    print("frozen explanations",explanations)
    print("controller calls",final_agent["calls"],"reason",final_agent["reason"])
    print("simulation errors",{n:d["mse"] for n,d in report["methods"].items()})

# %%

if MAIN:
    def changed_target(features):
        return float(max(features[0],features[1]))
    play_history=[dict(scene=SCENES[0],activation=float(changed_target(image_features(render_scene(SCENES[0])))))]
    play=run_agent(changed_target,HYPOTHESES,probe_pool,play_history,budget=3)
    display(pd.DataFrame(play["rounds"]))
    print("changed target",play["reason"],play["remaining"])
    audit_scene=dict(color="blue",shape="circle",background="plain")
    audit_features=image_features(render_scene(audit_scene))[None,:]
    print("predeclared blue-circle audit",float(changed_target(audit_features[0])))
    print("selected explanation predicts",None if play["explanation"] is None else float(hypothesis_predictions(play["explanation"],audit_features)[0]))
    print("one-call baseline",run_agent(target_unit,HYPOTHESES,probe_pool,training_history,1)["reason"])

# %%
