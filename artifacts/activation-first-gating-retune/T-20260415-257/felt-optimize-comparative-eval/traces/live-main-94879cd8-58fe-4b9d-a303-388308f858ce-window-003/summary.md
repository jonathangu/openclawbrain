# Recorded Session Replay Proof Bundle

- trace id: `live-main-94879cd8-58fe-4b9d-a303-388308f858ce-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ab9e7c92fecacf60147f27b8c27fc049edab247767d390d6cfd938c3433d0a10`
- fixture hash: `sha256-924c4ed1186166bf0f6b342b0967c241c06f6d79d3f88b2eb96f947a6b1061b7`
- score hash: `sha256-ba203543e0ec40665d9113b3955d7fabf32a9d536d3def792bc7961ef220a39c`
- bundle hash: `sha256-e27066a7d3cac00455eecb14a64e4722a7c2f39e1f0061394e70c360a1e6300f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-af4d866d80b0149b53a0921726b1499466ad574098653e689171c2b2c56dcca1 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a03c3cd1c73591fecb5ae7d292ca3e3d476277f6d964bede10a1514f4d18154c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a8a5bdd8f11440b353d6dc9dd58d86cc9920fbb187250bff088d018c3c4d3400 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-bb778cd7487ce107bf6fd018a7f2ce123d41d1a50a35e78f7a1602b085a21797 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bbca8a6c | sha256-7a55e6f738917580a9f3ea75990a4048173ec38f7f2bd67fe939386cac246573 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bbca8a6c | sha256-55abaeaecc3e2ab95c862abfc7e395fde9ac03d5ba49e094c7d189466abdca20 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-ace0b2fd | sha256-277ed8609c82ccada6a6035fc8226cd787e903eee279a1a2dc5ee0db1e56f8db |
