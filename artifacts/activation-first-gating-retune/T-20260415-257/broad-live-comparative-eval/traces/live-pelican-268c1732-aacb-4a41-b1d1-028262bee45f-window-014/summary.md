# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-82b94292f904129190996d09645352442519cd34f4a6fe4ddc3d8ccfdc15ed4f`
- fixture hash: `sha256-2b7971a9291be722d620678727dc2afe570e5b9dc9a97d0983cbb8375a8b4f0f`
- score hash: `sha256-13e8ffba2be5acd22d8ad94584a4e3f75c3f4bae11770f1038d2582b3f7f8ffc`
- bundle hash: `sha256-4c9b12d957f7147654c54c2b975c7a3a951071684009605ccca83e1e2bf5153d`

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
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2126389181abd46124f339c97d016b2e80dbdd1c3f4a30cb14b5104924e09f3e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2ab2409261cf8d1d434220bd9fefa5bc2d3ffeba59208cb466105df4779ae614 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3f2d66eb195d91b269c81cad6af686cc0be8c794943e309ed2119cae6146d07a |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-ba0cfb04f6841541dd76f8fe1d2f48bb7c32a19eed0f11a0e057057ad124ade2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6e346b85 | sha256-070740dbdcf745ad8f7a5d29d3658d60ac87c20e13e85980d8265ffcf9a24301 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6e346b85 | sha256-3b6317b6d24da7c87f550260a07f3c108f4255f98268eeebe3103ee99308a1df |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-03077562 | sha256-2cb46bf9955edf04858d4b2c60b9eb44e0a9e2ccf47c688541693f092685be6a |
