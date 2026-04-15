# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fd4a73ef0679d3bd5e8a41ecf8528eaf1056f459a2933d6bce7a274e1da6704d`
- fixture hash: `sha256-cdbe046df5ba47eb867d34f32f856111ce7f2bac423e41168b29efa3bc680b6e`
- score hash: `sha256-c951b08e1814ad47787d559eaef4b036bed13ec3d6e655242307edbb9d5b4baf`
- bundle hash: `sha256-60e609990f3bfd8d0aa1339a0867762ce7e10b55bbe83769c3c2517e2b2ab10f`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-195c8562b43d566f299d3b4d568af19c059fadcd5ad0dc52c1779f850a2eeca5 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5df3d6021007e384b9fc4f6c5fe8c7bf6a48a2060d785d584a23fe51ac600634 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c7e9fb2b893cb92931ea6668644cfa455a1c2dbf7b00b81da300ab2eb005afbb |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-bc55a143ae10402e31463ce721a8909774e63ca79d57a7f0ebc6e101c2afda87 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-7477ae65 | sha256-7ebcb6342780b4b285e334a33d176fbb1f060ef1cfc3a1cd0a8cd54d3ad41d4f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-7477ae65 | sha256-648f5872f8db7f4c2a2e68b5b987e468442d06f9ef4668ea0b25f0c2bd99d1df |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-7477ae65 | sha256-7ebcb6342780b4b285e334a33d176fbb1f060ef1cfc3a1cd0a8cd54d3ad41d4f |
