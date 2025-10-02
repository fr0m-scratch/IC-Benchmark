## Overview

This JSON-Schema specifies three **Interface Complexity (IC)** record shapes:  **tool-level** ,  **server-level** , and  **set-level** . Each record carries:

* `features`: numeric features (extensible),
* `components`: per-feature **weighted contributions** (for attribution),
* `score`: the final IC score (typically from a linear/calibrated model).

> Note: `log()` below means the natural logarithm (`ln`) unless your repo states otherwise.

---

## Top-Level Fields

* **`$schema`** : JSON-Schema version (`draft-07`) used for validation.
* **`$id`** : Canonical identifier/URL for this schema; helps `$ref` resolution and caching.
* **`title`** : Human-readable title (“Interface Complexity Feature Vector”).
* **`type`** : Top type is `"object"`.
* **`oneOf`** : Exactly one of: `Tool IC entry`, `Server IC entry`, or `Set IC entry`.
* **`definitions.weight_contributions`** : A generic map `{ feature_name: number }` holding **per-feature contributions** (usually `w_i * φ_i(x)`); may also include a `"bias"` key for the intercept.

---

## Tool IC entry

 **Meaning** : Complexity of **one callable tool** exposed by a server.

### Identity / Envelope

* **`scope`** : Constant `"tool"`.
* **`tool_id`** : Stable tool identifier (e.g., “serverA.get_user”).
* **`server_id`** : Owning server identifier (enables aggregation/drill-down).
* **`metadata`** (optional): Freeform provenance (extraction time, commit, doc URL, etc.).

### `features` (per-tool)

> Extracted from the tool’s **argument schema** (JSON-Schema/MCP descriptor).

* **`leaf_fields`**

  Meaning: Count of terminal, directly fillable fields (string/number/boolean/enum).

  Rationale: More fillable slots → higher chance of omissions/mistakes.
* **`object_fields`**

  Meaning: Number of `type: "object"` nodes.

  Rationale: More structural hierarchy and key constraints.
* **`array_fields`**

  Meaning: Number of `type: "array"` nodes.

  Rationale: Lists introduce length and element-shape burdens.
* **`total_fields`**

  Meaning: Total nodes (≈ `leaf + object + array`).

  Rationale: Coarse upper bound on structural complexity.
* **`required_fields`**

  Meaning: Count of fields marked required.

  Rationale: Higher obligation → higher failure cost.
* **`required_ratio`**

  Meaning: `required_fields / max(leaf_fields, 1)`.

  Rationale: Normalized constraint density.
* **`max_depth`**

  Meaning: Max nesting depth from root to any leaf.

  Rationale: Deeper trees → more reasoning steps/prompt alignment.
* **`union_ops`**

  Meaning: Number of union constructs (`oneOf`/`anyOf`/`allOf`).

  Rationale: Each union adds  **branching choices** .
* **`dependency_ops`**

  Meaning: Count of inter-field dependencies/conditionals (`dependentRequired`, `if/then/else`, mutual exclusion).

  Rationale: Captures  **coupling and preconditions** , raising error risk.
* **`enum_bits`**

  Meaning: Information mass from enums.

  Typical: For each enum with `|E|` options, add `log2(|E|)`; sum.

  Rationale: Discrete choice burden during tool filling.
* **`numeric_bits`**

  Meaning: Discretized burden for numeric fields.

  Typical: If `min/max/step`, let `choices≈floor((max−min)/step)+1`, add `log2(choices)`; if only bounds, assume a default resolution.

  Rationale: Captures the feasible numeric choice space.
* **`string_burden`**

  Meaning: Proxy for free-text generation difficulty.

  Typical: cost for unbounded strings + `log(maxLength)` + regex/pattern “entropy” (pattern length, character-class diversity), weighted.

  Rationale: Free text is error-prone; deserves explicit accounting.
* **`array_cost`**

  Meaning: Additional burden introduced by arrays.

  Typical: For each array, add `1 + log(expected_length+1)` and scale by **element complexity** (e.g., leaf/union counts of the element subtree). Use `expected_length`=1–3 if unconstrained.

  Rationale: Multiple elements expand the fill space.
* **`doc_tokens`**

  Meaning: Total tokens shown to the model for this tool (name, summary, arg docs).

  Rationale: Inflates prompt size and selection latency.
* **`missing_examples`**

  Meaning: Penalty for absent usage examples.

  Values: binary (0/1) or “how many examples are missing” (per-arg + end-to-end).

  Rationale: Examples reduce exploration/error.
* **`log_leaf_fields`**

  Meaning: Stabilized `log(leaf_fields + 1)` version.

  Rationale: Prevents outlier dominance.

> **Extensibility** : `"additionalProperties":{"type":"number"}` lets you add experimental metrics (e.g., `regex_entropy`, `fewshot_quality`) without changing the schema.

### Scoring

* **`components`** : Map of **weighted contributions** per feature (after any transforms), optionally with `"bias"`. For interpretability/diagnostics.
* **`score`** : Final tool IC score, commonly

  `score = Σ components + (bias)`

  Higher = **harder** to select/fill correctly.

---

## Server IC entry

 **Meaning** : Aggregation across **all tools on one server** (MCP server, HTTP service, etc.).

### Identity

* **`scope`** : Constant `"server"`.
* **`server_id`** : Stable server identifier.
* **`server_name`** (optional): Human-readable name.
* **`tool_ids`** : Tools included in this aggregation (for reproducibility/drill-down).

### `features` (per-server)

* **`tool_count`** : Number of tools on the server.
* **`log_tool_count`** : `log(tool_count + 1)`; stabilizes extremes.
* **`mean_ic_tool`** : Mean of tool-level `score` over `tool_ids` (typical per-tool difficulty).
* **`ambiguity`** : **Confusability** across tools.

  Typical: mean/top-k cosine similarity over embeddings of names/descriptions/arg docs; or overlapping arg names/semantics. Higher → harder  **tool selection** .

* **`naming_inconsistency`** :  **Inconsistency of naming styles** .

  Typical: mix of camelCase/snake_case/verb-vs-noun, fraction violating the dominant convention, variance of naming features. Higher → more cognitive load.

* **`statefulness`** :  **Degree of stateful interaction** .

  Typical: share of tools with session/cursor/page tokens; presence of begin/continue/commit patterns; explicit pre/post-conditions in docs. Higher → the agent must  **remember prior context** .

* **`doc_burden`** : **Token load** to present this server’s tool catalog. Either sum of `doc_tokens` over tools or “server overview + top-N tools” with a budget.

### Scoring

* **`components`** : Weighted contributions at the server level.
* **`score`** : Final server IC score (higher = harder).

---

## Set IC entry

 **Meaning** : Complexity of the **actual tool set** exposed to the model in a prompt/run. Useful for pruning/ablation/ablation-style studies.

### Identity

* **`scope`** : Constant `"set"`.
* **`tools`** : List of `tool_id`s in the set (may span multiple servers).

### `features` (per-set)

* **`k`** : Number of tools.
* **`log_k`** : `log(k + 1)`; stabilizes extremes.
* **`redundancy`** :  **Intra-set redundancy** .

  Typical: mean pairwise similarity over tool embeddings (names+descriptions+arg docs), or Jaccard overlap over capability tags. Higher → harder selection, wasted context.

* **`token_load`** : Tokens to present the set (names + short summaries + key arg schema). Consider capping per-tool display tokens.
* **`mean_ic_tool`** : Mean of **tool-level `score`** across the set.
* **`server_diversity`** :  **Diversity of source servers** .

  Typical: `|unique(server_id)| / k` or a normalized diversity index. Can reduce collisions but increase stylistic heterogeneity.

### Scoring

* **`components`** : Weighted contributions at the set level.
* **`score`** : Final set IC score (higher = harder to **pick** and **fill** correctly across the set).

---

## Practical Notes (repo alignment)

1. **Extraction pipeline** : Parse schemas → traverse AST for structural counts → estimate `enum_bits`/`numeric_bits`/`string_burden`/`array_cost` → compute `doc_tokens`/`missing_examples` → aggregate to server/set → apply linear/calibrated model → emit `components` + `score`.
2. **Interpretability** : Always emit `components` (with `bias`) to power dashboards and RCA.
3. **Extensibility** : Use `additionalProperties` to add experimental features (e.g., `regex_entropy`, `example_coverage`) without schema changes.

