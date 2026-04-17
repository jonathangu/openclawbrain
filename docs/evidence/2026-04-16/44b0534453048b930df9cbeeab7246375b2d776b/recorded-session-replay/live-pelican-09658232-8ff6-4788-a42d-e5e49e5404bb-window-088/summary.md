# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-088`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f6e0cde3a25a3b093a3328111ec29f970fc82068378dbcdb19446f77be2e4c1e`
- fixture hash: `sha256-a657cfcc3c13a64972df27ba9b34b582252db2226ee691420ef45e3b6a2bad38`
- score hash: `sha256-15bcbf9ebe79f47f22eddac2a5ddba8ea89170f88fbde33b0d6f4ec5caad05bd`
- bundle hash: `sha256-70087a4c48f4ce7a5ed8c9338ef124e082cc0dd28a90a8be8616b55be458fda4`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f0b3855bdbfdfdd31f2cf7aedce5b8a8e42a2e757ba398c67ba975aa86dd21ea |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2d42d26e10c51894cea9f820cee1da63a4b841b7663ddc36c9bd3f7c09f80944 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-240e91d8dda37d4237f8855debc76517fd833e1410fa08515abbc65d8a10d330 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-a28a4febca81da2f74c8f1206c2e09b69c09f3ddb8bb37a594515f974d4099c1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-931e822c | sha256-f783a00a8565305a9b2553617df222e85f0e80ff51f0a5ff0f05a8e75c0de41b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-931e822c | sha256-6746008ac5b0dbf735be6986e6d01c7aac1ea7a7cf1a9cc84dff5ef32f5e349a |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-100f7f8f | sha256-6380ccbb2c278028222f8685be9ecae894161d7afd9797a76659f96646bd15d9 |
