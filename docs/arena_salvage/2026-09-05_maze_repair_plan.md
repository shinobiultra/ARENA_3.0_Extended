# 14.4 maze policy and state intervention repair

Inspected the paired notebooks, instruction page, reference source, tests,
artifact lock, and expected outputs. The original transformer implementation
and Othello-style behavior / probe / intervention progression are the comparison.

Current learner failure: several immediate tests call only reference functions;
the student path omits the `template_action` dependency; dropdown formatting is
suspect; the section calls a local legal-neighbor count a search frontier and
calls repeated next-action decisions latent planning. Its same-maze intervention
does not isolate route-specific information from next-action information.

Target section claim: a supervised small transformer learns a maze policy whose
next-action information can be probed and causally replaced. Measure the stronger
planning and route-state hypotheses without presupposing success.

Signature result: actual BFS / predicted routes, raw versus constrained path
scores, layerwise distance / legal-degree / action probes, all-position patching
heatmap, and a same-action donor from another maze beside the existing controls.

Student-implemented functions: preserve BFS, tokenization, attention, model,
training, rollout, probes, and patching in the notebook. Supply the small missing
template helper visibly. Rebuild solutions by filling only learner stubs with
their own dropdown answers; execute that complete learner in a fresh kernel.

Controls/baselines: preserve existing thresholds, split exclusion, template,
shuffled walls, random labels, wrong-position and norm-matched random patches.
Add a Manhattan-distance baseline and same-action other-maze donor to test whether
distance decoding / patch recovery warrant stronger internal-search claims.
Derive path metrics from actual cells and actions rather than trusting flags.

Play cells: retain the editable maze / decoder and layer / position interventions.
Expose failures without stopping students before they see the evidence.

Files to modify: extension section sources, readable builder and lesson source,
paired notebooks, instruction export, reference artifacts, and scope metadata.
Original ARENA chapters remain unchanged.

Tests to run: independent action/route scoring fixtures, exact attention and
patching oracles, learner function injection, dropdown/solution parity, complete
CPU learner execution and CUDA experiment if the candidate remains scientifically
valid, then preservation and relevant course checks. No report can stand in for
an executed learner path or independent review.
