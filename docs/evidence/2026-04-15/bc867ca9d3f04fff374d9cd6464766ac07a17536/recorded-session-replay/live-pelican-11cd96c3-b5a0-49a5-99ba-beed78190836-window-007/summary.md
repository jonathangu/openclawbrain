# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d958774a8fc5556f6b626cb2afd5141be38b390f01c3f1c481f5689e5c67765c`
- fixture hash: `sha256-bf711d8c588faf57d4df6088b8652fb030ca7a163bb118e31c3e2f2768cad0f2`
- score hash: `sha256-2442a58d1bbe97856911962791262af08cac3fd6ff0e7029ed37868884422956`
- bundle hash: `sha256-a560c574b0199dcc257867b2c0a096575870f930b54e97c0d7c3921b169d5882`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7f7ca30e9c8433554610f300b068b172fcd1c7c716d277545f4d5940081fb358 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9d1ddb279b565c73fd0a553afd5b07fd686fe52203a3c55b4414eca26bb00273 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-29498f657128d61044e2b7a0c016af58823e36059d0c4bb5861faa1980203c6b |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-abf859fc019fcb571c4cbd10874ed2f07d1a59ab129f8e5fe10c5bc23739e8df |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-26d17906 | sha256-afb339231732d58f846d2a367d8a7a2fcf865a326733ca4cde3b59e5d37f5fe2 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-26d17906 | sha256-90ddec385b8bc831bbb84553d57fee3d4418cb739544978f0f195ec38590887f |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-26d17906 | sha256-afb339231732d58f846d2a367d8a7a2fcf865a326733ca4cde3b59e5d37f5fe2 |
