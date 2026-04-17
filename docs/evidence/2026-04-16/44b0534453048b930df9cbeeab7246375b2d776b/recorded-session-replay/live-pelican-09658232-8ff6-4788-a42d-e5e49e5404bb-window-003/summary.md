# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-15894236758cffd6885df088771bc9158a039d8e6dca7ba37e0c0ae93f2bb22c`
- fixture hash: `sha256-897b7fdc496e16305fc54601a8aba44f23b5322a6b7036c26e9f447dc3d9e950`
- score hash: `sha256-ae6462c32c928e8791cdfc1be7c240ce5a3d74db6caafb1fa513882dc6b10601`
- bundle hash: `sha256-17fd14f305ac6a4f25036bbd50ef9e85415a513b3576be7522c2a56192d2a86e`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-05b9f1a0d0ad4a80c5a15a8f7ef9c5d2527f8753fe005026d39ad6af8199556b |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d9ded60cfe553963b601ef8f03393a04b235e9f5ae0352de22d59be138ba9f22 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c230a50e0119e6bb90b50163acb0aa3c2339b55ddf6be5afb21b87cf668f27c0 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-57368cb329c1f98d8ad308ca2591e01284c0b65ef04b2377226d48d2442e1e9f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b7cb86be | sha256-33c1a8908290dc88698e18036f93e0716b16a77b26cf8f25e2a9cf62f91ceb40 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b7cb86be | sha256-96fd74971164c5c9c531006b18f6cbaefb265f53c04a06e4e88bd2d94e130a81 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-7c4c1153 | sha256-7924215df4f8cd1f63f236c3ed842c408064d0820b9e35b32b5018e41883015c |
