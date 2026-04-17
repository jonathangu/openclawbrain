# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d958774a8fc5556f6b626cb2afd5141be38b390f01c3f1c481f5689e5c67765c`
- fixture hash: `sha256-bf711d8c588faf57d4df6088b8652fb030ca7a163bb118e31c3e2f2768cad0f2`
- score hash: `sha256-ceabee34f55c57bcaf0fbd4f608d28465b9f10b33622e8e705666c597e4985d4`
- bundle hash: `sha256-172fbd4af078311c2ee254f4543cc7ad95ed1967a829a8002dfe80c209c43436`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7f7ca30e9c8433554610f300b068b172fcd1c7c716d277545f4d5940081fb358 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3006d09ad6cd18ef37618cf609fa88c9d4f5ed9cd5f70bc9449de986c2009f9d |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-72b31432863574b24e6887d04a5683e966b8ab1447eb9412fe67d1be55a19969 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-81ed4587cadd21db5de8835d4d32ff2ec559e0d068dd3a4a617d9d7a76abe210 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f677e268 | sha256-1ccdd055ecce7576c944afe8ec7146410b6fbe7c631159464157759610998d25 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f677e268 | sha256-fe59b1830de229bed124d2ebc352ca0555ac27979c63dd7f671fb31858110866 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-a0d577c7 | sha256-4c535d88a7f2bd011e52fee4bfbefb734a80f1e39dd366af325f9dd5a6cb4fda |
