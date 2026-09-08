# [8.6] AtP* from Scratch

This section asks one question: **when does first-order attribution patching miss
the components that exact patching says matter, and which AtP* corrections fix
those misses?**

Start with `8.6_AtP-Star_from_Scratch_exercises.ipynb`. The exact nonlinear toy
has eight named sites and exhaustive intervention ground truth. Learners build
clean/corrupt caches, batched exact patching, vanilla AtP, the AtP* Q/K softmax
correction, GradDrop, rankings, recovery@k, and joint faithfulness. Eleven answer dropdowns complete all twenty code cells. The notebook
then applies those ideas to 96 named final-position Q/K sites in the exact pinned
`EleutherAI/pythia-70m-deduped` revision over 24 fixed grammar pairs.

The visible signature results are:

- `../../instructions/assets/atp_star_toy_signature.png`: exact rankings against
  vanilla AtP, AtP*, an EAP-style edge baseline, and random order, together with
  recovery, joint faithfulness, and measured method cost;
- `../../instructions/assets/atp_star_transformerlens_signature.png`: exact and
  approximate Pythia rankings, verified top-k effect mass, and causal random and
  shuffled controls.

Run the focused CPU checks with:

```bash
uv run pytest -q chapter8_automated_circuits/exercises/part6_atp_star/tests.py
uv run jupyter nbconvert --to notebook --execute \
  chapter8_automated_circuits/exercises/part6_atp_star/8.6_AtP-Star_from_Scratch_solutions.ipynb \
  --output 8.6_AtP-Star_from_Scratch_solutions.ipynb \
  --output-dir chapter8_automated_circuits/exercises/part6_atp_star \
  --ExecutePreprocessor.timeout=600
```

`run_gpu_test(max_vram_gb=24.0)` is a strict CUDA entry point. It raises when
CUDA is unavailable and never substitutes CPU execution. The candidate report records the completed CPU learner run; its release status
remains unaccepted pending independent review and strict CUDA evidence.

The real gradient hooks retain the computation graph across layers. Independent
scalar-chain tests check derivatives of 48 for vanilla AtP and 8 after dropping
the last layer; the earlier implementation incorrectly returned 2 in both cases.
The real patching, Q/K correction, gradients, GradDrop aggregation, and experiment
composition are now visible and editable in the notebook. The playground runs as
a code cell and plots what happens when the cancellation path is omitted.

`lesson.md` and `build_notebooks.py` synchronize the two notebooks, all eleven
solution dropdowns, and the instruction page. The corrected CPU result gives
rank correlations about 0.943 (vanilla) and 0.993 (AtP*), with top-eight effect 0.489
versus random mean 0.189 and best 0.417. The release report remains unaccepted
pending independent review and strict CUDA verification.

The real-model claim is deliberately narrow: AtP* closely tracks exact causal
ranking for final-token rotary Q/K interventions on this fixed agreement task.
It is not a complete transformer circuit, a head-function interpretation, or a
replication of every AtP* diagnostic.
