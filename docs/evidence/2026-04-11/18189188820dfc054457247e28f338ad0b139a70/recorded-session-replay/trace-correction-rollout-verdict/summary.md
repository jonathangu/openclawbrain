# Recorded Session Replay Proof Bundle

- trace id: `trace-correction-rollout-verdict`
- winner mode: `graph_prior_only`
- trace hash: `sha256-16b1c9786508756f0b9bc0745893317b2032d3d7f42af8b674ab6c96358a37bb`
- fixture hash: `sha256-33cce9db06a1c0557c61c637734ec6566a42558c018e5cd2eca45eb861553334`
- score hash: `sha256-99dc27aefe2ecbdb02563457f7032329cbb9d72957484d9beaf4fed7fe5f931a`
- bundle hash: `sha256-f92acff724f3616c2cb2dcab71c7755b27b835c807957b05dee6ce5a001abf5c`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | learned_route | 70 |
| 3 | vector_only | 70 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 3/8
- phrase hit rate: 0.375

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 2 | 1 | 0.5 | 0 | 1 |
| learned_route | 2 | 1 | 0.5 | 0.5 | 1 |

## Hardening Snapshot
- compile failures: 2/8
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 2 | 0 | 2 | 2 |
| vector_only | 0 | 0 | 0 | 2 | 2 |
| graph_prior_only | 0 | 0 | 0 | 2 | 2 |
| learned_route | 0 | 0 | 1 | 2 | 2 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 2 | 0 | 0/2 | 0 | 0 | 2 | 1 | 0 | sha256-d1c39961f2988b0187ccc97c6c8d166f325a88a382093d64e508703691a75ec8 |
| vector_only | 2 | 2 | 1/2 | 0 | 0 | 2 | 1 | 0 | sha256-46f98dc12ec56a2462b327ce257c9ed2bc545b4bc2ef9ff544aad55e1d5e00c9 |
| graph_prior_only | 2 | 2 | 1/2 | 0 | 0 | 2 | 1 | 0 | sha256-1a835ee263525e23df1de29b4f1a2cebffb99b5f76476b2d2ae972b29e9aff03 |
| learned_route | 2 | 2 | 1/2 | 1 | 1 | 2 | 1 | 0 | sha256-f85f91536d09cbd68429a9d5284aa9c6eae17960007211743977ff92e1b849dd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | rollout-verdict-turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | rollout-verdict-turn-2 | 0 | no | 0/1 | no | no | none | none |
| vector_only | rollout-verdict-turn-1 | 40 | yes | 0/1 | no | no | pack-45e9f087 | sha256-c26b96714c167ae4af69e9a3b7589f0a2d79be1543c0c50945e23ccaf42d5863 |
| vector_only | rollout-verdict-turn-2 | 100 | yes | 1/1 | no | no | pack-45e9f087 | sha256-c26b96714c167ae4af69e9a3b7589f0a2d79be1543c0c50945e23ccaf42d5863 |
| graph_prior_only | rollout-verdict-turn-1 | 40 | yes | 0/1 | no | no | pack-45e9f087 | sha256-c26b96714c167ae4af69e9a3b7589f0a2d79be1543c0c50945e23ccaf42d5863 |
| graph_prior_only | rollout-verdict-turn-2 | 100 | yes | 1/1 | no | no | pack-45e9f087 | sha256-c26b96714c167ae4af69e9a3b7589f0a2d79be1543c0c50945e23ccaf42d5863 |
| learned_route | rollout-verdict-turn-1 | 40 | yes | 0/1 | no | yes | pack-45e9f087 | sha256-c26b96714c167ae4af69e9a3b7589f0a2d79be1543c0c50945e23ccaf42d5863 |
| learned_route | rollout-verdict-turn-2 | 100 | yes | 1/1 | yes | no | pack-e3c94101 | sha256-66e0481578d5cdd6f07921d16ed31cdb6c0bd1d87fb4864a91de24ca3e9dcff2 |
