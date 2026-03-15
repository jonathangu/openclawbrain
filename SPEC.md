# OpenClawBrain: MDP Specification

Reference: Gu, J. (2016). "Reinforcement Learning." Econometrics Field.

## The Routing MDP

### States
```
S = { s_0, s_1, ..., s_T }

s_t = (currentNodeId, queryEmbedding, visited, fired, budgetRemaining, hopCount, maxHops)
```

### Actions
```
A(s_t) = { traverse(neighbor) for neighbor ∈ N(currentNode) \ visited } ∪ { STOP }

At seed phase (t=0): A(s_0) = { traverse(seed_i) } ∪ { STOP }
```

### Terminal Conditions (Assumption 1: game ends in finite time)
1. Agent chooses STOP
2. Budget exhausted: budgetRemaining ≤ 0
3. Max hops reached: hopCount ≥ maxHops
4. Dead end: |A(s_t)| = 0

### Reward (Assumption 2: reward only at terminal state)
```
z ∈ [-1, +1]

Sources (trust ranking):
  human   (rank 4): explicit corrections, confirmed good context
  self    (rank 3): tool success/failure, task completion
  scanner (rank 2): structural heuristics
  teacher (rank 1): off-path LLM evaluation
```

### Policy
```
P_ρ(a_j | s_t) = exp(score(a_j) / τ) / Σ_k exp(score(a_k) / τ)

score(STOP) = stopBias + budgetPressure × (1 - remaining/total) + hopPressure × (hops/max)
score(traverse(j)) = edge.weight × edge.prior + cos(query, node_j) + edgeKindBias[edge.kind]
```

### Update Rule (Lemma 6.1)
```
∂/∂ρ v_ρ(s_t) = E[z · Σ_{l=t}^{T} ∂logP_ρ(a_l|s_l)/∂ρ]

Implementation:
  advantage = z - baseline
  For each step l in trajectory where chosenAction = traverse:
    gradLogP = 1 - P(a_l|s_l)
    Δw = learningRate × advantage × gradLogP

  baseline_{n+1} = α × z_n + (1 - α) × baseline_n
```

### Weight Decay
```
w_new = w × rate + prior × (1 - rate)

Default rate: 0.995 per tick
```
