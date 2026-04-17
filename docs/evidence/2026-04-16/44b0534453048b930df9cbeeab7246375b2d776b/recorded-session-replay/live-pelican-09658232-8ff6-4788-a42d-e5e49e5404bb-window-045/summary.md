# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-045`
- winner mode: `graph_prior_only`
- trace hash: `sha256-42dd5ae1fbc52ab37ab26b7eff707ccc814072dbaeb4cf80246f57beb5474c7c`
- fixture hash: `sha256-dae1773b38ede59c62f735546227926063dcc22433a680794834acb15197b82c`
- score hash: `sha256-b3b7c3884fa70574247898de952c7cffb979b931030252a22435e75903de16a4`
- bundle hash: `sha256-2e6c9e14505d9d1f64205c790a2b2674f9e61575c70a917c7cf8b24253c4ebfb`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-97339b57eb3bff564bd492b91102981f9054b332cee78d9338b804ad8b646434 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-49f836b468c3a0645689746d37e38e113750c65c6da536a27f248aea8bda9946 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dde6616fb9f23d20cc3cce28ba201623f77dd76334fe9f3c3ea6a2bcaab4b28a |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-487c7a4d60f2dd82e29eaeada5769df8b4cb9e6fcf7bc287248efbccb0151c82 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3afd0ffc | sha256-0530d1066288ab83b04bb6aa3cd99d8c3c07a03c537e12d8b1c47835b7e72682 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3afd0ffc | sha256-4d960d54b2e207d79293f37c8546bb650012f2a94eaab2ad42f4bbd59aaa680c |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-7538a865 | sha256-72bc1094a03e1b5a53654b555ea8fcaacc6d029ff32d696002bb896a7a7a766b |
