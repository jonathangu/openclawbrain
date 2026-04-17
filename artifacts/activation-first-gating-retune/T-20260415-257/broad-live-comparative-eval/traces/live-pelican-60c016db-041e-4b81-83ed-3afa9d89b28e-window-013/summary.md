# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7e2a057c58ceb7779d689dcd4238dfbc3207e352fc341de03ac7a06d504301da`
- fixture hash: `sha256-737f6561e785d3bc05d3981f983d5cf16785ca63d2f46199fbc1baaeee1f2b69`
- score hash: `sha256-30f522c93f8fcfe69e18bea6200aa51587e41e466130d62976f574d2e3e66744`
- bundle hash: `sha256-8ed1ba7f229ccda0c0da503151280a394d475e0cd73889d484a0a34255904ca6`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1e7a79c157dc055e3ad83a213c22e42badb5ac82b3ed30aa50ada887959b805f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b2fa5b50300ee98389fe31384ec103dd2b9a722d69be4046e241c7373891e67d |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d62b87ab892636ace5a0a9310b9b9ca65ca777e824c8f05a85c2dc968a7d2436 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-885ffb79804cc081bb868243c6c5cf44c6e7043b0235c6ced2719fdc87ed5f13 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d45579fa | sha256-dad002bee2b6b4fd8ea83dbc2c22ae6a6991822a26457bc90e7fc77a2ecabb81 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d45579fa | sha256-5aa7ed08932c1a94db9d0449f0a4cbd94e0be90a5c3bd832e826aabbedbad7b1 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-f5c3a9cd | sha256-72617535def64eef07471a5ea3dac515e8abf01ac7757337719b3c2115d98ed6 |
