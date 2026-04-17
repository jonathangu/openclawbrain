# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-188`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2286f1962f858995a9d11d68ccd4ff744be8c0925ede8b9595870bdf0f8216d1`
- fixture hash: `sha256-7131fce3dd7f89b87927812976c9719dadea253e34115e7f37e0887827e9427e`
- score hash: `sha256-969fa07a4f066873c71c6776b03e08738a8fe6187fd17485f9b24db6dd227d78`
- bundle hash: `sha256-c044bd2ef5eb73d341cde9465aae4c9c78d8d7612c31edbaa14b61368de05f19`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4cce362e58248df18caee133abbb86ea37c7c8cc312d9027b572d5a719da7a87 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ebde483b6e186b453431b4023dc992b745dd35b6a079aaa04287b5d327ca1a01 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5ffa92a80119501fb3ffcf91928b2cc6b40dbec3e7415a1635cea677607ecdd5 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-f03e309253ac303c488b3fbbd67affca1c27df7343658a573693cbdfeb7f4f8d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d67e6c63 | sha256-97ac45ec26ea05a5f308da76bf702663c5e0d48a341dc2dd2b183b3042c0a062 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d67e6c63 | sha256-c99c97846640f9e628566f3b287abf7bed1258fe457e73f81c250863f0460564 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-a9dad85e | sha256-42171d5171e2e5f22796dfca8be69917f8d1ccfcf890cf6bbca433ba363b2d26 |
