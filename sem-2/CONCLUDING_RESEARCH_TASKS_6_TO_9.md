# Concluding research protocol — Tasks 6 to 9

## Document status

Tasks 6–9 are implemented and their output pipelines are ready. Their final
SUMO experiments have **not** been run in this workspace yet. Therefore this
document specifies the research questions, controls, commands, evidence and
decision rules; it does not invent numerical outcomes. A task becomes
result-complete only when its final validation file passes and all raw,
summary and plot artifacts are present.

These four tasks conclude the project by studying the already selected Task 5
shared Dueling Double DQN from four complementary perspectives:

1. component sensitivity;
2. operational robustness;
3. transfer and sample efficiency; and
4. explainability and auditability.

This is a stronger final-year narrative than adding four unrelated algorithms.
The same multi-intersection controller is subjected to controlled experiments
that test what it uses, where it fails, whether it adapts, and how it decides.

---

## 1. Common scientific protocol

### 1.1 Frozen source model

Tasks 6, 7 and 9 load the selected Task 5 final checkpoint without modifying
it. Task 8 reads the same checkpoint to define zero-shot and fine-tuned
conditions. Every experiment records a SHA-256 fingerprint, preventing results
from being attributed to a different checkpoint accidentally.

The source model must be described honestly as a shared Dueling Double DQN
trained with prioritised replay, max-pressure expert guidance and a pressure
safety mask. It is not a purely unconstrained DQN.

### 1.2 Paired simulation design

Whenever controllers are compared, they receive identical SUMO seeds and the
same network, route and horizon. Comparisons are paired by seed. Tasks 6–8 save
10,000-resample paired bootstrap intervals. With only five final seeds, these
intervals quantify finite-sample uncertainty but should not be described as
proof of population-wide statistical significance.

### 1.3 Main outcomes

All performance studies retain the same four primary outcomes:

| Outcome | Direction |
|---|---:|
| Mean system total waiting time | lower is better |
| Mean system stopped vehicles | lower is better |
| Mean system speed | higher is better |
| Completed trips per hour | higher is better |

Teleport counts, simulation horizon, monotonic arrivals/departures and finite
values are structural validity checks. A good congestion score is never used
to hide an invalid simulation.

### 1.4 Result integrity

- Raw episode CSVs are retained.
- Completed units are skipped on rerun, allowing interruption and resume.
- Experiment manifests reject incompatible seeds, horizons, routes,
  checkpoints or hyperparameters.
- Negative, neutral and unstable outcomes remain reportable results.
- Final evaluation seeds are never used for model selection.

---

## 2. Task 6 — Controlled component ablation

### 2.1 Research question

Which state-information groups materially influence the selected Task 5
controller under the original 2×2 traffic distribution?

### 2.2 Design

One frozen checkpoint is evaluated in six paired conditions:

| Condition | Change |
|---|---|
| `fixed` | fixed-time control baseline |
| `full_dqn` | complete shielded Task 5 controller |
| `raw_dqn` | same network without action shielding |
| `no_network_context` | mean/max/std network queues set to zero |
| `no_identity` | four-value intersection identity set to zero |
| `no_pressure` | four phase-pressure values set to zero |

The final protocol uses five seeds, producing 30 simulations. Each ablation is
compared both with fixed timing and the complete DQN. The component-degradation
plot uses paired percentage changes, so positive degradation means occlusion
worsened performance.

### 2.3 Hypotheses and interpretation

- H6a: the complete DQN retains its Task 5 improvement over fixed timing.
- H6b: removing at least one traffic-information group changes performance or
  action behaviour measurably.
- H6c: raw-versus-shielded differences reveal how much nominal performance
  depends on the safety layer.

This is deliberately labelled **post-training occlusion**, not architectural
causal ablation. It answers whether one frozen policy is sensitive to a feature
group. It does not prove how a separately retrained model would behave.

### 2.4 Final command and evidence

```powershell
python sem-2/06-component-ablation/run_ablation.py --seconds 1800 --seeds 1001 1002 1003 1004 1005 --device cpu
```

Expected evidence includes 30 raw episodes, structural validation, grouped
metric plots, paired tables, bootstrap summaries and an automatic Markdown
analysis under `06-component-ablation/results/` and `plots/`.

---

## 3. Task 7 — Robustness under operational stress

### 3.1 Research question

Does the Task 5 DQN remain beneficial when traffic demand, road capacity or
observations differ from nominal training conditions, and what protection is
provided by its safety shield?

### 3.2 Stress matrix

Three controllers—fixed, raw DQN and shielded DQN—are evaluated under six
controlled conditions and five seeds, producing 90 simulations.

| Scenario | Intervention |
|---|---|
| `nominal` | original route and observations |
| `demand_surge` | SUMO demand scale 1.4 |
| `partial_lane_blockage` | lane `-h11_0` limited to 1 m/s from 600–1,200 s |
| `gaussian_noise` | σ=0.15 noise on normalised sensor state |
| `sensor_dropout` | 20% independent sensor-value dropout |
| `delayed_observation` | two-decision sensor delay |

All stress randomness is deterministic for a scenario/seed pair. The physical
incident restores the original lane speed even if the episode raises an error.

### 3.3 Analytical contribution

The task reports:

- paired controller improvement within every scenario;
- each controller's degradation relative to its own nominal baseline;
- queue recovery time following the temporary blockage;
- sensor corruption and shield-intervention rates;
- bootstrap uncertainty; and
- scenario-level and overall predeclared success decisions.

A scenario passes when shielded DQN wins waiting and stopped-vehicle outcomes
on at least four of five paired seeds while losing no more than 5% average
throughput. The overall target is at least four of six scenarios with zero
teleports.

For sensor faults, the raw DQN sees corrupted observations directly. The
shielded DQN's pressure mask uses the simulator's true state. Thus shielded
results represent a layered controller with an ideal independent safety
channel, not an end-to-end detector fault. This limitation must remain in the
paper.

### 3.4 Final command and evidence

```powershell
python sem-2/07-robustness-stress-test/run_stress_test.py --seconds 1800 --seeds 1101 1102 1103 1104 1105 --device cpu
```

The command can be interrupted and rerun unchanged. Expected evidence includes
90 raw episodes, stress and degradation tables, incident recovery records,
paired bootstrap analyses, validation JSON and uncertainty plots.

---

## 4. Task 8 — Transfer learning and limited-budget adaptation

### 4.1 Research question

Does the Task 5 representation provide faster or better adaptation to a
directionally shifted demand distribution than zero-shot reuse or training the
same architecture from scratch?

### 4.2 Controlled domain shift

The source route has four straight flows with probability 0.10 each. Two
deterministic target routes preserve total flow probability while changing its
direction:

| Domain | Horizontal flows | Vertical flows | Role |
|---|---:|---:|---|
| `target_horizontal` | 0.16 each | 0.06 each | adaptation and target evaluation |
| `reverse_vertical` | 0.06 each | 0.16 each | post-adaptation generalisation |

Route files and hashes are generated automatically. Because total probability
is held constant, the experiment targets directional distribution shift rather
than merely higher volume.

### 4.3 Learning comparison

| Method | Initialisation | Target training budget |
|---|---|---:|
| `zero_shot` | selected Task 5 checkpoint | 0 episodes |
| `fine_tuned` | selected Task 5 checkpoint | 12 episodes |
| `scratch` | deterministic random weights | 12 episodes |

Fine-tuned and scratch conditions receive the same 12 SUMO seeds and the same
NumPy exploration stream, architecture, optimiser, replay configuration,
pressure mask and update budget. Neither receives imitation loss. Validation
uses seeds 1251–1252 at episodes 0, 3, 6, 9 and 12. The best checkpoint is
selected only from these validation seeds. Final seeds 1301–1305 are unseen.

### 4.4 Sample-efficiency and transfer tests

Validation score is `waiting + 10 × stopped`; lower is better. The area under
the validation-score learning curve measures limited-budget sample efficiency.
Final target performance is reported separately from AUC.

The automatic analysis keeps three distinct statements:

1. fine-tuned target score beats zero-shot;
2. fine-tuned target score beats scratch; and
3. fine-tuned validation AUC beats scratch.

All must be inspected independently. Failure can indicate negative transfer,
no advantage within 12 episodes, or a source policy already near its attainable
performance. Reverse-vertical results expose whether horizontal adaptation
damages retained generalisation.

### 4.5 Commands and resumability

```powershell
python sem-2/08-transfer-learning/generate_target_routes.py
python sem-2/08-transfer-learning/train_transfer.py --episodes 12 --seconds 1800 --device auto
```

The training command completes two episodes for each learning condition per
invocation. Rerun it unchanged until both conditions report complete. The
immutable manifest prevents resuming with changed hyperparameters.

```powershell
python sem-2/08-transfer-learning/evaluate_transfer.py --seconds 1800 --seeds 1301 1302 1303 1304 1305 --device cpu
```

Final evaluation contains 40 simulations: four controllers, two domains and
five seeds. Expected artifacts include full replay/optimizer/model checkpoints,
training episodes, validation histories, learning curves, route manifests, raw
evaluation episodes, paired bootstrap tables and final automatic analysis.

---

## 5. Task 9 — Explainability and decision audit

### 5.1 Research question

Which feature groups drive DQN Q-values and actions, how stable are those
explanations across traffic seeds, and where is the policy uncertain or in
disagreement with its training expert?

### 5.2 Complementary explanation methods

The task captures every state, Q-value, safety mask, raw action, executed
action and expert action. It combines:

1. gradient saliency for local model sensitivity;
2. zero occlusion for a simple counterfactual;
3. distribution-preserving group permutation, which breaks the state-feature
   association without inventing arbitrary magnitudes;
4. Q-margin, action distribution, expert agreement and shield intervention;
5. representative decisions selected by fixed rules; and
6. separate per-seed importance, 95% uncertainty and rank correlation.

The nine audited groups are phase, minimum green, pressure, incoming queue,
outgoing queue, local queue, network context, simulation progress and
intersection identity.

### 5.3 Validity and limits

The final run expects exactly 4,320 decisions: three seeds × 360 decision times
× four intersections. All state and Q values must be finite, all actions must
be valid and all feature groups must occur in pooled and seed-level analyses.

High sensitivity does not establish a causal traffic effect. Correlated queue
and pressure features can share or displace importance. The interpretation
must also disclose the model's max-pressure expert-guided origin.

### 5.4 Final command and evidence

```powershell
python sem-2/09-explainability/run_explainability.py --seconds 1800 --seeds 1401 1402 1403 --max-analysis-samples 4000 --device cpu
```

Expected evidence includes raw episode and decision traces, pooled and
per-seed importance, rank stability, local occlusion cases, behavioural
statistics, Q-value and action plots, validation JSON and an automatic report.

---

## 6. Recommended laptop execution order

Run the tasks sequentially after Task 5 is complete:

1. Task 6 quick check, then final ablation.
2. Task 7 quick check, then the resumable full stress command.
3. Task 8 isolated `--run-name quick` check, then final training over repeated
   sessions, then final evaluation.
4. Task 9 isolated `--run-name quick` check, then final audit.

Generous CPU-laptop allowances are:

| Task | Final allowance |
|---|---:|
| Task 6 | 2–6 hours |
| Task 7 | 5–14 hours |
| Task 8 | 7–18 hours total |
| Task 9 | 1–3 hours |

SUMO is primarily CPU-bound. A GPU can accelerate small neural updates, but an
RTX 3050 is not required for evaluation. Laptop thermals and background load
can make wall time vary substantially.

---

## 7. Suggested three-member ownership

| Member | Primary ownership | Cross-check |
|---|---|---|
| Member 1 | Task 6 ablation and Task 9 explainability | verify feature definitions and interpretation |
| Member 2 | Task 7 stress testing | verify incident/fault settings and recovery analysis |
| Member 3 | Task 8 transfer learning | verify paired budgets, checkpoint manifests and final seed separation |

Every member should review the common metrics, Task 5 expert dependence and
the difference between pipeline success and positive scientific outcome.

---

## 8. Final paper/report claim policy

After the experiments finish, claims should follow the actual saved evidence:

- Say “the pipeline completed” only when structural validation passes.
- Say “the DQN improved” only for metrics and seeds showing that improvement.
- Report confidence intervals and paired wins beside mean percentages.
- Report negative transfer, failed stress scenarios and unstable explanations.
- Do not call occlusion causal architecture ablation.
- Do not generalise SUMO findings to real roads without detector calibration
  and field validation.

The intended final contribution is an end-to-end, reproducible evaluation of a
coordinated deep-RL signal controller—not a claim that every learned policy or
every traffic network will behave identically.
