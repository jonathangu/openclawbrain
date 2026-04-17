# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-028`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3b96b2a97bdbabf2a2491696460b8da77dc242a25c47533759e1ca69d544c781`
- fixture hash: `sha256-32449d86eb6b142eb11e1d76d43e4c37d62e87233bae5b870977e6a064fa97e1`
- score hash: `sha256-00a55d4c7c578267d1699dcd62337fbbb0b63d6a978ed895750e5fd0192fc261`
- bundle hash: `sha256-bdb2b6e91fb33f70529da04ed2fdaf1434278b1fc5db5db31a80ecd282d1044e`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4b14dd3575bcaf16e76897e36504d083be01ba320a2077714c9a7749ba84f112 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5e06319f02fb6ef7b262de7499b9be03a03e54858bde1cb24e047054787e2f27 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-92f1225ee714a0f8dab6e877e8f654a5c4014aad176310e066f5870c554ed477 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-48728d7ecfa4a2fc346964030a9cece6afa81c05c632705798e1d8090abbe9be |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-817361ec | sha256-46393dd3e65c2c467bd6a6c4a3a44f708697de209be1f58a47fc70434459637f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-817361ec | sha256-30b161e0f0fd3043ec43e29aa42f5d26191d357f809a6cc2eafcc729024d199a |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-225db351 | sha256-b2d05dc45d0f4be9034c1823df905eba40e1c752ce2439aab1ad247be9bf4b8c |
