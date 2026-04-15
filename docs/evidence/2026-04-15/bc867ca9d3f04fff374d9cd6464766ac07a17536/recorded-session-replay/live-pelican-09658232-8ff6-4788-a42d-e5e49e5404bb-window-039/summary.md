# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-039`
- winner mode: `graph_prior_only`
- trace hash: `sha256-30a54314d984e83263bc7ddfcb852ce4d67a835461588938c047eabba74d7daa`
- fixture hash: `sha256-a669f6ac0947e4907b9b5ff0ba78d765904f903d2ac7c540eba1f40434878bd9`
- score hash: `sha256-96fe9d44a0a330c2d4187517becae5176601426acc505518ad79a37b9dbe4c95`
- bundle hash: `sha256-0cd35a7fe8e3e9b6daaba11223d7c5c6393f9d41f73b69ddf0e4d46ab2c65bc8`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0ddef39ad20fc1c3136dfb625c29bf78d555d4df3233592558f3107ec01752a7 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-357d5a2c022eda0cec945a5d007d9f118c91f93428423f4b23b3da8244934f06 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4aa324fa3ec3acf22523d4667f7bf928e67e711837f54195ee075e38448b0f3c |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-ee0f81992124ffca51a5c1354c469c7ec939bca2b8c2f8bcae8f820f61026385 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-16e2f6d8 | sha256-17ed713644e2930e9132cb88e0e5dfc3ed8ca1725974d11a90b9ddd91379b97a |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-16e2f6d8 | sha256-391dcd7d1709d1bcb382b9e3300bbe3fbe3adc3665f6fb0075ef64384b8a2985 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-16e2f6d8 | sha256-17ed713644e2930e9132cb88e0e5dfc3ed8ca1725974d11a90b9ddd91379b97a |
