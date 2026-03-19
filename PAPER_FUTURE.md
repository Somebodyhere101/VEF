# Future Directions: General Basis Circuits and Self-Improving Systems

## The Architecture of Meaning

The current VEF system demonstrates that a language model can generalize without gradient descent through the correct basis (PPMI+SVD) and closed-form equation (W* = (H'H + λI)⁻¹H'Y). But the individual mechanisms — retrieval, refinement, co-substitution, awareness — are only the first instances of a much more general framework.

We propose **Basis Circuits**: fundamental operations on learned embedding spaces that compose into arbitrary reasoning. Not task-specific modules, but general computational primitives — like how the brain has a small number of circuit motifs (lateral inhibition, winner-take-all, recurrent amplification) that compose into vision, language, motor control, and abstract thought.

---

## Five General Basis Circuits

### 1. The Projection Circuit

**What it is:** Project any input onto any learned subspace and measure how much information survives.

**Why it's general:** Attention is a special case — Q/K/V project tokens onto subspaces to compute relevance. The Awareness circuit (basis vs anti-basis) is another special case — projecting onto the learned basis vs its null space. The Projection Circuit generalizes both: given input X and subspace S, compute how much of X lives in S.

**What it enables:**
- **Awareness** (project onto basis → energy = knowledge, project onto null space → energy = ignorance)
- **Relevance** (project query onto a candidate response's subspace → how relevant is this response?)
- **Feature extraction** (project text onto the "sentiment" subspace, or the "factuality" subspace, or any discoverable dimension)
- **Gating** (if projection energy below threshold, the input doesn't activate this circuit — biological lateral inhibition)

**Mathematical form:**
```
projection(x, S) = S @ S' @ x
energy(x, S) = ||projection(x, S)|| / ||x||
```

This single operation, applied to different learned subspaces, produces attention, awareness, feature detection, and gating — all from the same primitive.

### 2. The Composition Circuit

**What it is:** Combine two or more concepts in embedding space to create a representation that never existed in the training data.

**Why it's general:** Current retrieval finds existing entries. Composition CREATES new meaning by blending embeddings. "joke + cats" → a point in embedding space at the intersection of humor structure and feline content. This isn't interpolation — it's the embedding-space equivalent of compositional semantics.

**What it enables:**
- **Novel generation** (combine form × content to produce text that never existed in the corpus)
- **Analogy** (A:B :: C:? → compute the relational vector A-B, apply to C)
- **Transfer** (learn a pattern in domain X, apply it to domain Y through shared embedding structure)
- **Bisociation** (Koestler's creativity theory: the intersection of two previously unconnected frames of reference — computed as element-wise product of two concept embeddings)

**Mathematical form:**
```
compose(a, b, mode) = {
  blend:     α·embed(a) + (1-α)·embed(b)    (weighted combination)
  intersect: embed(a) ⊙ embed(b)             (element-wise product)
  transfer:  embed(c) + (embed(b) - embed(a)) (relational shift)
}
```

The antonym axis (embed(hot) - embed(cold) ≈ embed(big) - embed(small)) is a special case of the transfer mode. Generalized, this discovers ANY consistent relational pattern from seed examples.

### 3. The Decomposition Circuit

**What it is:** Break a complex concept into its basis components — the dimensions that define it.

**Why it's general:** SVD already decomposes the embedding matrix into principal components. The Decomposition Circuit applies this to individual concepts: which basis dimensions does "gravity" activate most? Those dimensions ARE the concept's definition — force, mass, attraction, acceleration.

**What it enables:**
- **Explanation** (decompose a concept → its top components → natural language description of each)
- **Comparison** ("gravity" and "magnetism" share force/field components but differ on mass/charge — computed from basis overlap)
- **Hierarchy** (concepts that share many components are in the same category — emergent taxonomy from embedding geometry)
- **Debugging** (when the model gives a wrong answer, decompose the query and response into components → see exactly which component caused the error)

**Mathematical form:**
```
decompose(concept) = {(i, embed(concept)[i]) for i in top-k dimensions by magnitude}
similarity(a, b) = cosine(decompose(a), decompose(b))
explain(concept) = retrieve entries that maximize each top component
```

### 4. The Convergence Circuit

**What it is:** Iteratively refine a representation until it stops changing — the general form of the current Refinement loop.

**Why it's general:** The brain processes information through recurrent loops that settle into attractor states. The Convergence Circuit does the same: start with an initial embedding, repeatedly transform it through the basis, until it reaches a fixed point. The fixed point IS the stable interpretation.

**What it enables:**
- **Disambiguation** ("bank" near "river" converges to water-bank; "bank" near "money" converges to financial-bank — same word, different attractors)
- **Reasoning** (multi-step inference as iterative convergence: premise → intermediate → conclusion, each step refining through the basis)
- **Self-correction** (if the initial interpretation is wrong, the convergence loop pulls it toward the nearest correct attractor)
- **Depth** (number of convergence steps = depth of reasoning, emergent from the data, not fixed by architecture)

**Mathematical form:**
```
converge(x, max_steps):
  for step in range(max_steps):
    x_new = retrieve_and_blend(x, corpus)
    if cosine(x, x_new) > threshold: return x_new  # fixed point reached
    x = x_new
```

This is already implemented as the Refinement loop. Generalized with different retrieve_and_blend functions, it becomes multi-step reasoning, planning, and even hypothesis testing.

### 5. The Self-Model Circuit

**What it is:** The model's representation of its own knowledge, capabilities, and limitations — derived from the geometry of the basis and anti-basis.

**Why it's general:** The Awareness circuit (basis energy vs anti-basis concentration) is a specific measurement. The Self-Model Circuit generalizes this into a continuous map of the model's knowledge landscape. For any point in embedding space, the model knows:
- How much it knows (basis energy)
- How confident it is (energy concentration)
- What related things it knows (nearest basis components)
- What it would need to learn to know this (which anti-basis directions to fill)

**What it enables:**
- **Honest uncertainty** (refuse when anti-basis dominates — already demonstrated)
- **Targeted learning** (identify the specific anti-basis directions that need filling → request exactly the right data)
- **Curriculum** (learn concepts in order of basis proximity — start with what's close to existing knowledge, expand outward)
- **Meta-cognition** (the model reasons ABOUT its own knowledge structure — "I know about physics but not biology" expressed as a subspace relationship)

---

## Self-Improvement Through Basis Expansion

The most profound implication: a model that knows its own limits can systematically expand them.

### The Learning Loop

```
1. QUERY:    User asks about X
2. AWARENESS: Project X onto basis → anti-basis dominates
3. DIAGNOSE:  Which basis dimensions are missing?
4. ACQUIRE:   Find data that fills those dimensions
               (ask the user, search the web, retrieve from external corpus)
5. INTEGRATE: Add new Q/A pairs to corpus
6. UPDATE:    Incrementally update embeddings (rank-1 SVD update)
               Basis SHIFTS to cover the new knowledge
               Anti-basis SHRINKS in that region
7. VERIFY:    Re-project X → basis now dominates
8. ANSWER:    Respond with the newly acquired knowledge
```

Each interaction makes the model smarter. Not through gradient descent, but through DATA ADDITION — the basis grows to encompass new knowledge. This is mathematically principled online learning:

- **No catastrophic forgetting** (old corpus entries remain, old knowledge is preserved)
- **No retraining** (incremental SVD update is O(d²), not O(n·d²))
- **Fully auditable** (every new piece of knowledge is a specific corpus entry that can be inspected, verified, or removed)

### Incremental Basis Update

When new data is added, the PPMI matrix changes. The SVD can be updated incrementally:

```
Given: existing U, S, V' from SVD of PPMI matrix M
New data: Δ (change to PPMI from new corpus entries)
Update: U', S', V'' ≈ SVD(M + Δ) via rank-k update

This is O(V·k²) instead of O(V·k·n) for full recomputation.
```

The basis evolves continuously as the model learns. The anti-basis shrinks. The knowledge boundary expands. And at every step, the model KNOWS exactly where that boundary is.

---

## Full Interpretability

Every decision in this system is traceable:

| Decision | Mathematical Operation | Interpretable As |
|---|---|---|
| "I know this" | Basis energy > threshold | Query projects strongly onto learned space |
| "I don't know this" | Anti-basis concentration high | Query falls in null space of learned features |
| "This is the answer" | max cosine(query, corpus) | The most similar stored Q/A pair |
| "I'm not confident" | alignment × margin < threshold | Best match doesn't stand out from background |
| "These are opposites" | cosine(diff_vector, antonym_axis) > threshold | Word pair's difference aligns with learned opposite direction |
| "The answer is 42" | Digit decomposition + carry | Arithmetic from corpus-verified single-digit facts |
| "Let me think more" | Convergence not reached | Embedding is still changing between iterations |
| "I need to correct this" | Anti-basis detects noise → edit distance finds correction | Unknown word is close to known word in character space |

There are no hidden states. No attention heads whose function we can't explain. No layers that mysteriously transform representations. Every operation is a dot product, a projection, or a lookup — and each has a clear semantic interpretation.

---

## The Vision

A system where:
- **Generalization is a theorem**, not an empirical observation — given the correct basis, W* is the optimal mapping, provably.
- **Self-knowledge is geometric** — the boundary between what the model knows and doesn't know is the boundary between the basis and its null space.
- **Learning is data addition**, not parameter optimization — new knowledge is a new corpus entry, not a gradient step.
- **Every decision is auditable** — trace any answer back to the specific corpus entry, embedding dimension, or circuit that produced it.
- **Improvement is targeted** — the anti-basis tells the model exactly what it needs to learn next.

This is not a black box that happens to work. It's a transparent system where intelligence emerges from the correct choice of basis — and the basis itself tells you when it's correct and when it's not.

The opposite of a basis is not emptiness. It's the map of everything the model has yet to learn.
