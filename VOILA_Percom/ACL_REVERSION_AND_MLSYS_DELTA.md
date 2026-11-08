# ACL reversion and focused MLSys delta

## Decision

The default manuscript is the ACL/EMNLP version in
`acl_legacy_example_paper.tex`. The PerCom manuscript remains available in
`percom2027.tex`, but is no longer the default `example_paper.tex` entry point.

## Why the ACL version is the better base

- Its central claim is coherent: question semantics predict the visual
  information required before image retrieval.
- The three observations, linguistic feature design, calibration, and VQA
  framing reinforce one another.
- The evaluation spans five datasets and six VLMs, which supports a broader
  empirical language claim than the FloodNet-only PerCom framing.
- The PerCom draft shifts the paper toward a systems venue, but its own notes
  identify unresolved claim/table mismatches, policy-definition drift, and
  overclaims about universal accuracy/cost dominance.

## Focused delta to improve the ACL paper

Keep the ACL organization and NLP-first thesis. Import only the strongest
validated material from the later work:

1. Make the policy definition exact and unambiguous. State whether the primary
   selector is global argmax over calibrated utility or sequential escalation;
   keep the other only as an explicitly named ablation.
2. Use one cost convention throughout each result table. Prefer measured
   per-item bytes where the data exist; do not mix normalized historical costs
   with measured-byte claims without a clear label.
3. Add the leakage-resistant split description, image-grouped confidence
   intervals, and the oracle-gap/error decomposition to the ACL evaluation.
4. Retain the deadline/mission experiments as consequences of the linguistic
   selector, not as the paper’s organizing system contribution. State plainly
   that the network experiment is a trace-based serialization simulation when
   that is the measured protocol.
5. Add one compact worked example of the decision path: question features,
   per-fidelity calibrated probabilities, byte costs, selected fidelity, and
   resulting VLM call.
6. Keep the honest boundary conditions prominent: template shift, visual
   ambiguity, per-VLM calibration, cached captions, and the accuracy gap on
   general VQA.

## What not to import from the MLSys-shaped rewrite

Do not add Tag2Graph-style components, multi-shard deployment claims, online
drift retraining, or a three-phase memory-system architecture unless new
measurements actually support them. Those ideas belong to the MLSys paper’s
ontology-guided conversational-memory problem, not to VOILA’s pre-retrieval
fidelity-selection claim.

## Immediate paper-cleanup queue

- Reconcile the VOILA acronym expansion and title.
- Replace “greedy” language wherever the implementation uses exact global
  argmax, or restore a genuinely sequential implementation and label it.
- Remove all reviewer-note commands from the submission output; source notes
  are now hidden by the ACL entry point.
- Audit every headline number against the result tables and ensure baseline
  costs include the same acquisition boundary.
- Keep the ACL master as the paper to edit; treat PerCom as a separate branch
  of the project rather than the source of truth.
