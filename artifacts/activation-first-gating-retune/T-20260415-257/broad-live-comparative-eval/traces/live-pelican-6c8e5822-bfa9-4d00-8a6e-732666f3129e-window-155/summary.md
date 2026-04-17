# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-155`
- winner mode: `graph_prior_only`
- trace hash: `sha256-54c6a7f75aa98b64fa06de64444db8f288aa41bfaf9731cc070d54f577be960a`
- fixture hash: `sha256-899705aeb2321d03b6a0aee78d7cfb19ca0d976080db3e6a3f83db60267852fd`
- score hash: `sha256-0fc5a7f0d6188a355adf9de641b5e2741f8f59aa89455c02d30d12731d90a971`
- bundle hash: `sha256-63a0df346bd4575b5fed39d0ca441f473ab1ee2e4d7dcce3eb5c2b92dd5a4034`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ee198590c5a4a8c84e2f8fe36017d040fb15fc92428b4d0396417de634b42329 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-89e3d57868a44239c09e910ba3e1619a915a83d67abaec25775f3c161f9f597f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4256f71df00408abefa0474d8bc0b4fbd71aad86e6a884800d70e8b4dba12acf |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-45cc297adf51ab767e53dae999121a506fa602e11215d3148bedeabd31b48d68 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6051808e | sha256-8178ca304b37e73a11e3ddbc32044a04ddb6c862e51cec48cc47cc44d73fda57 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6051808e | sha256-9a81b859a2689dfb1aed2e1370f9005b22e5427416a8d6f19abfdc2bc3e57208 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-e5e7baf3 | sha256-03e99082a094a9c8f40600f0edd5117794f97eab19100bdd3dbe7807a88b3c05 |
