# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-178`
- winner mode: `graph_prior_only`
- trace hash: `sha256-dc837ac64ce4a5cb1d121e2bea7830254f5b1cd1faf9dd8be0505cf94fe18342`
- fixture hash: `sha256-555eb18092c7a3b48bf36359187522f84e12b063bd73ce65d859cb8f468c2af9`
- score hash: `sha256-1076dd760ae905de77a3585ca498b23aebd748d6ab39611f6c4ab88daa457be2`
- bundle hash: `sha256-38df0aa21e3c5997047ef86e1e925bac9e0e54637ac951b467865932fccaad49`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-94018db213a88670c23984311d9a8431beabced6aba3b25434ee10a70b79887e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e69e8dba88b487cc473005f7d8ca8e316c1d61abe4a7e0034bdb10d2d863261d |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e8676849b4fdb2b0da9727a9c26d26c27817aaee031562db947976b3c1c47f3a |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-91b262fac0fd0f0db93c00d2fa61993daa54452e492cb1136929e606f4725bb5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2aa71efc | sha256-dfb01da495592473cffe727d5accda507b230620b522752c8bd2d10e0a2f47f1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2aa71efc | sha256-d556521bdfd9580b71d73189118c3884f11f2c31cd3d6800b8a7a001bf63f80d |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-c759a4a1 | sha256-473bab863c5f8d65db5742fd48fb8ec1d5be1292a3789e20b533c523a1451338 |
