# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-054`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4f8bd6e98ba409d4b92ff33d315c90158dc9f7928f49ee95918b29862594fc07`
- fixture hash: `sha256-f2d0f492e33718dcda5e95309dd8b8ae83d2a012ce623b86565c773255e59638`
- score hash: `sha256-42119755b9bf244964daec27a623e9945b5a2fc883e800416b6d55d053705be0`
- bundle hash: `sha256-7d1472571d8acee8f2111e16cb021005d4fa406097c83153b965a5527b4c4a5f`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3a49035e9fd3e0717342039595aabed753c46d3f982a6fbdc847832f0114d10f |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f7c88953f83b48f599ad0caa5d33b6e5da43b942ea6ea1ef3c496c4e19f5e7b3 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-bb814a63e862c1057ab2dad4a0fcedcdb509c06861ae16ecee7ec7eae27b71ab |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-df526b4b74375f1065c2acbe51e5602b1311c669c70aed0c5fcaf7e00d62fa72 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-0e306eff | sha256-e11cc98a6f717a1382e01c94d98588b0f461cd2d9f4616fc90bbbcce84150b50 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-0e306eff | sha256-907c4093c2c5034a8828ec4dafa4d8d8eebae619954e01acd632c2f309aafb8f |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-0e306eff | sha256-e11cc98a6f717a1382e01c94d98588b0f461cd2d9f4616fc90bbbcce84150b50 |
