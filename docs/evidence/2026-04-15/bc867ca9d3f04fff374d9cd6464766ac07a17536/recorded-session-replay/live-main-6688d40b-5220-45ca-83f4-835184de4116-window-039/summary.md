# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-039`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1ec62e8076ee4d1e710644be210d5ded13133f83ba7cc0a283a8ff2ec6e4b13a`
- fixture hash: `sha256-208011a3d49bd10b0f228ef3f15f5d25a591b8469fe6d29ce8deec0246fbbb48`
- score hash: `sha256-60986be37ab714d6f3d9ad00168626aeea5ad94c8b64eb9193979ba763a4fe86`
- bundle hash: `sha256-a94311bf991a9c0dd34ebc4660d1ab0617157725a8a6598abdb1a4cb26286225`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-30241ba1cbd874d0509ab1e29b9c021ef1eb69d9f017747456f3594de63d356c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-439358f097b778c200ce007ae5bf4162466b8e11b7804ef3c1ba9c6ebc24f96b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e9bf0e2f4f19856966e6cb88568a7eb32fb045a3a2c8e4540eb2509f152c588c |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-785a77df3a8cad80f4763c3fe6eed3f890cd32a873ad4430b45a5646c3137b95 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-71d26a06 | sha256-14115433009bcc860d1c9dc836e85ffb98bb36b196a4eebdacdf6f6bef3ad6ee |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-71d26a06 | sha256-59a8658f131675bf679bd793804165fdcb436d8898046b8fde018ab943e594f4 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-71d26a06 | sha256-14115433009bcc860d1c9dc836e85ffb98bb36b196a4eebdacdf6f6bef3ad6ee |
