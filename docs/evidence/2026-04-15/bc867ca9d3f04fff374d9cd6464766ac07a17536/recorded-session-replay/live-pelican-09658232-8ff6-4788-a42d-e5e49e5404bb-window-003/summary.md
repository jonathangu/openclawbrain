# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-15894236758cffd6885df088771bc9158a039d8e6dca7ba37e0c0ae93f2bb22c`
- fixture hash: `sha256-897b7fdc496e16305fc54601a8aba44f23b5322a6b7036c26e9f447dc3d9e950`
- score hash: `sha256-f4ad7b9293c901992418203b1a7cce4f6931efcba453d1ad533e51ce6cdb4bb4`
- bundle hash: `sha256-b24a8eedcab6fba9c3b82757e85a2ca3df7c5f8f86c742836317ef1e61bd1e1b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-05b9f1a0d0ad4a80c5a15a8f7ef9c5d2527f8753fe005026d39ad6af8199556b |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-82f105f0aa25f132c263fb004a786eb54dea473f92bdfe0bbb5b730459a3cdbe |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-54a27b7442bb4eb529e160f9d56aa69bd5e1fbcba35c4798ff4ce04575c63cb0 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-f4119fb65efbf2631a202de76ef4efcebb03b16c8075ba231fd569ea647bfe40 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-57a866d2 | sha256-acd63d4408ec8ec06529a05575490d8223195a2e7407aac353f4fd33a27969f7 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-57a866d2 | sha256-b070cdad8528fb3ef281f02ffaab53f44e6ca67302db333be2885a434165c1df |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-57a866d2 | sha256-acd63d4408ec8ec06529a05575490d8223195a2e7407aac353f4fd33a27969f7 |
