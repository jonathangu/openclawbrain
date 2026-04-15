# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-08b73891909d4362ec26f6fa9db500532bf1bc8c805c846177530e06134e3890`
- fixture hash: `sha256-8b9b0ca98fc7faf65751139ae1faf124a5228fc02a0f5bb6427265ff145c7a87`
- score hash: `sha256-4d9f8c4c85683a40cc5fe23f2148492a392a0909e59694a4ec00bafc54509d1e`
- bundle hash: `sha256-cf09436c310b8e505b1b8cbbe88766de4bd2508ec3e055715c1ba3f437fefce7`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1f871e8b1f54b24b9e075d5f4db6f8b41f6cb53e929f6d747d42ccbb2426d8d7 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d20a8141ff114defde345d7a425b95d567d814e2e6e68a2572d615ade0174644 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-459eb447ab48bae400738d32f99df6c5a79c18f59a845590d182d9a83b5a5571 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-5c7a08286f64263a1c43b209566fe62984be6c725618c8f9b6076abdb4432732 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-900c18ab | sha256-a79f0dbb20f858cfc72fa3320978aa421774c4179f3e28a5322a8a9948119ca4 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-900c18ab | sha256-b79731f385b55543972c52409ff7e19a7b8e566454bdcdae00614f1d25ac0159 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-900c18ab | sha256-a79f0dbb20f858cfc72fa3320978aa421774c4179f3e28a5322a8a9948119ca4 |
