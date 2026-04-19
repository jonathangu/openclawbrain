# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fd4a73ef0679d3bd5e8a41ecf8528eaf1056f459a2933d6bce7a274e1da6704d`
- fixture hash: `sha256-cdbe046df5ba47eb867d34f32f856111ce7f2bac423e41168b29efa3bc680b6e`
- score hash: `sha256-bf08c0f511dbe609083e01e7fbdd3fc99655afd8a13994a61ac99bbbc267a483`
- bundle hash: `sha256-0e89ae5a8b2ab7f43a828cab5cdb56f8f0feb76aae8f5e6098fa0e0469872071`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-195c8562b43d566f299d3b4d568af19c059fadcd5ad0dc52c1779f850a2eeca5 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-02c7f8867b66645bd65395ceb8eef24911200efa27dab5e54c11842e889c9b65 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-242ec2695c4b7da038aacefd6e30cec122770afaab8f6bd53d4314a1531c620e |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-9d3326d10fcd3e9a45657cd4410ae43a60191a314f2c0844ad507059e1718338 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-66dac898 | sha256-c482c4fec8388b39034d7db58cf468bcb0e364fca25ac29cd16f62b5e6246ebb |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-66dac898 | sha256-24f72fd8d6d38b1ceeeb7ea74480b6ce7c8723ae9fd061a151502e647c6ad489 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-66dac898 | sha256-c482c4fec8388b39034d7db58cf468bcb0e364fca25ac29cd16f62b5e6246ebb |
