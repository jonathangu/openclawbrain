# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-164`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f337daaeed6bc47fc68765c0195f530bb4ce38ec076e00ac4c73412b426d85da`
- fixture hash: `sha256-5e5cab708ce5b294bac69d34a6279b47e648ad8d40ed85f35998caca6e589c7b`
- score hash: `sha256-d419ae00d8154fb7cacc5cf23cfa8235cc1ef99e2b4198d9491654b77691424b`
- bundle hash: `sha256-6bb0cf92a8297cee6370ef4b0efaeb34ab0750ff3b24b65c7f0d327954b23ea4`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-100c819b269094add6922ee0aca0d157fd41366c476c3703f8d276f1431d3315 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-901f70a1db1683785bb76b4bb1c93569dfa37b4ae78778be242dceab100231d8 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-35ed0721911dcc82438950593497ed474da86ffd6f438dabce2aa21851232d12 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-1273be1c29a155c552cf6f9068d02aa0c27bdd799abd74fb7f904a65153379d5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b54c5e05 | sha256-ebef44162c6462816c9bad34bda4338c346c4aa2cdeca29f4e3b70b7c2167e51 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b54c5e05 | sha256-d3a5207828dee68d0991231b80368a10145b513c74fe4e24c80b30e6563912f5 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-3ff57f14 | sha256-1738a140d12edb4d68540d6e78cfca9c00f85cebeae886caa48b93c072228476 |
