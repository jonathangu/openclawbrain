# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8226e38f2d583af41a4327f3b8df4e5b434ae18ebbdb89d67531a4a854359a44`
- fixture hash: `sha256-e3733e9aa09beb01fe43936408b2069d985913ff1742752483045d9debec0829`
- score hash: `sha256-6c702b0aa3541ba9eb9eed16f5e99add1a9a6818fd0b448ed367849b3533247f`
- bundle hash: `sha256-83c4af825496d1a41334cea1866ea849fc117bf6a3af4a72875812423b2d6f5c`

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
| learned_route | 1 | 1 | 0 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f1f95cc8e218fff5d5905cf899fc04d3d3c62a98c1d684ae5ae4dffaa6f7bd10 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c524a53c3590b899c6b2aaadb0ff6a5ad6f6624580c264666312a32c7f749b4f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c2d4c50dabe9f9e96648a93b832833a027027fd2a58ca632a9e506c40b150cce |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-f59c0752854d94b0f9aa401186921a5ecc80d33c8d6c8f1ab9fae075b9ff6cb6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5696e790 | sha256-d28ba75bc2a284cc81a6c9af62d1eb94b7bb99d3be2a524556a65950c41d2251 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5696e790 | sha256-178e771f933f262f58c74f0e0a005b637e5d8e7b7b82938788a0e5f56319274c |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-ffc9389d | sha256-5b43e0771e8613a3c53bfdafbc0a63f6b367c952fbed01322529e5ee922ba18e |
