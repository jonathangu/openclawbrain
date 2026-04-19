# Recorded Session Replay Proof Bundle

- trace id: `live-main-6bc67602-c4ee-4fc7-8fbc-3434b2aa2286-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f7b093cf106a437e24ba93bbfbea56317e62afd65cc282953b847c0fec17c90f`
- fixture hash: `sha256-f186a663337b28243cdd6e62a9c63e0bf0678cf05202237e1d19a1f17b82f110`
- score hash: `sha256-d4cf1de190b599fcf14920cd3bc3070c8ba95c6aeff5cc2e8fed21819355e2fd`
- bundle hash: `sha256-96683535b1c42a33b89dc9a184889ed7fb965594fa3a5ad2d658d5b9723d50eb`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/12
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
| learned_route | 1 | 1 | 0 | 0 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ea127012751163ce5c5c7b6a51409b045b05c15be13611d375e11b98fe528366 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-9ab1c998399ccb6495a2513c7deef312d8725a152928c059eb58809713f02dfa |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-74a3e5163cf598b82d732604a805ecba8aa59395e5c986ac96423834eb5ae25c |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-b13cb8066af87736cde3d1e0a9634c9c74a150d53fb8170b1e8c11731ddcc64b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-3749358f | sha256-710e257b35b6a87c672c2a0f3d31edae1571f9e9465ebf4452844214af2fea84 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-3749358f | sha256-8b8006074565d4b8ea4bb7469e03483697e850793ec86ca80db5fb20df46a71d |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-3749358f | sha256-2a790ffe97e4358075459f3b162d5c1373066f853cb92eb62ac6f109c502a73b |
