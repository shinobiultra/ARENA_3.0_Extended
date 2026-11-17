# Deep Learning Alchemy companions

[Deep Learning Alchemy](https://deep-learning-alchemy.github.io/) (Stanford CS 312) teaches how
training choices change what a model learns, unit by unit, each ending in a quiz where you predict
the result of an experiment before you see it. This page lists the optional ARENA companions to its
units. CS 312's materials stay on its own site: we link them, at a fixed release where one exists,
and never copy them.

## What a companion is

A companion is an ARENA-style notebook for one CS 312 unit. It works in that unit's setting with a
small model of its own: you build the method, test it, run it, and see the result next to controls.
It ends with a 🎯 Held-out Quiz: predict a described variant of an experiment you ran, then run it
and compare. Companions are optional side trips from the main route; the
[study guide](mastery_hub/STUDY-GUIDE.md) shows where each fits.

## The seven units

| Unit | Topic | CS 312 materials | Companion |
| --- | --- | --- | --- |
| 1 | Basics | [assignment code](https://github.com/deep-learning-alchemy/assignments/tree/baafc3f2d364e5ddbba75f0fe8d1cb29326920e5), [handout](https://deep-learning-alchemy.github.io/assets/assignments/a1_basics.pdf), [quiz](https://deep-learning-alchemy.github.io/quiz-basics.html) | Not written yet |
| 2 | Hyperparameter scaling | [assignment code](https://github.com/deep-learning-alchemy/assignments/tree/069e18c52c93a0c6626b00f1c79de070ba30275f), [handout](https://deep-learning-alchemy.github.io/assets/assignments/a2_hyperparameter_scaling.pdf), [quiz](https://deep-learning-alchemy.github.io/quiz-hyperparameter-scaling.html) | The optimizer lab (below); its notebook isn't written yet |
| 3 | Optimization | [assignment code](https://github.com/deep-learning-alchemy/assignments/tree/57c60bbeef200ee20f5bdd847bfefaf7646fee78), [handout](https://deep-learning-alchemy.github.io/assets/assignments/a3_optimization2.pdf), [quiz](https://deep-learning-alchemy.github.io/quiz-optimization-2.html) | Not written yet |
| 4 | Architecture | [assignment code](https://github.com/deep-learning-alchemy/assignments/tree/4be74a55db2c9d87df59c637c9795377f3320b9b), [handout](https://deep-learning-alchemy.github.io/assets/assignments/a4_architecture1.pdf), [quiz](https://deep-learning-alchemy.github.io/quiz-stability.html) | Not written yet |
| 5 | Capacity | [assignment code](https://github.com/deep-learning-alchemy/assignments/tree/d2407116240a41790fa0ff860b96dae3b08b72ec), [handout](https://deep-learning-alchemy.github.io/assets/assignments/a5_capacity.pdf), [quiz](https://deep-learning-alchemy.github.io/quiz-capacity.html) | Not written yet |
| 6 | Generalization (DNA) | [assignment code](https://github.com/deep-learning-alchemy/assignments/tree/7f0db017cb89e31c7d387b907480f977dd033127), [handout](https://deep-learning-alchemy.github.io/assets/assignments/a6_generalization.pdf) | Not written yet |
| 7 | Not yet released | See the [CS 312 schedule](https://deep-learning-alchemy.github.io/#schedule) | Waits for the unit |

Do each quiz before you open its answer slides on the CS 312 site.

## The optimizer lab

The optimizer lab goes with unit 2. It is an optional app for exploring how learning rate,
momentum, curvature and gradient noise move gradient descent on a quadratic loss: change the
settings, play the run step by step, and compare the plots. Start it from the repository root:

```bash
uv run streamlit run mastery_hub/app.py --server.address=127.0.0.1 --server.port=18506
```
