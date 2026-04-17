# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-047`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1d8011faf3d69f32bcf0e92bcef735c94f96aebd8322b667cbe52a25917f1a6e`
- fixture hash: `sha256-f28ec0241ac4efd4c1f97733d381efba161e2d4c7cd778ddce2f415ed4529529`
- score hash: `sha256-d1574a51aacaf9634bb7c3d14dff6f17e5ed3968934b344830fb437240f46774`
- bundle hash: `sha256-559b0e4c4eb967c7812294c6e5ceede81c43f421dc6c66c305936e5f104b275b`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-64f932981a8a1428e017d3b3bf8eed9c04a8f1b43e3be668df16d36de77d3b6f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a62036dcd8eac965076e091453a28a3a8b0704b9ccbff5d245741ab4f0faa128 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-65909c630ab332e9a7e553f87b112abb4440c2ad4266678cd546d15f0872dbac |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-fc1b1f65b065618e676b036543a031e568cf3bb300a518ec565171e17ae980d3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-31bf62b7 | sha256-11cfcdc5962846b4ed2a6020609b0387ba7709378e40118c538e311ba08a0575 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-31bf62b7 | sha256-d3a5e66b003c9191ccbb0681b870333022268aee66a5eaf88de370b091a3252c |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-99f580c2 | sha256-b39e604cdaa56af3d751dcb62ec3118952ba2159fae7b4f777e750fd0c109a59 |
