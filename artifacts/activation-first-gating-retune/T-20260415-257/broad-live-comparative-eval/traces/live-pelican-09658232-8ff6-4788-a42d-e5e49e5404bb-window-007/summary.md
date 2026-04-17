# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-945c2e59668de577944ec7fc5dd5f9442c630538679d6020f6fdac64e2a21a17`
- fixture hash: `sha256-19b533ee2cadb7bef94e2f868a3d98284f247e98f26920f7fea15136681e3d11`
- score hash: `sha256-7beda594be7184871dd3aeee876f74258d90378ae49f16442591805d000ff7df`
- bundle hash: `sha256-f2a61e9538c123ef8dd683ba8543000fe912137724d3d4ed3213d49fa2bfb157`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d37994d9aa92d2e1c7fd5cea54b3093f268f662f580c5608c088fa86597acbc2 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6dc5b58f0406a57b5f5c632f77a31b078b56912f23e8e513d189115994e6ae60 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1797ffc3c3cd0508cd2ce426eb0ac1bb753a9b438584c152d61c5fe68c7d68f6 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-ff2fe16c77b289d4acc8a8820ad284f32dc9e7ba270c2a37150a0a2b032e2564 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-bac9f5a8 | sha256-3968516374b243410b870ab1fc9880407a220b4ca0e2f42905f8f0d84d2dd521 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-bac9f5a8 | sha256-7c9bb1dbb0540a9f3ba97752bb2a26b59da7974a2c82f2fe4b0dd3d29737abca |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-81c9df35 | sha256-2602288fc1b7d278c5005dc441d2f01826c34ca0dc0358b47a3f1187e1b0d5f0 |
