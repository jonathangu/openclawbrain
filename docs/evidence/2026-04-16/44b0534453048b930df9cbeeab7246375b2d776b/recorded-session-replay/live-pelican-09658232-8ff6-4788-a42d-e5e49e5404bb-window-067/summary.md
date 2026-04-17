# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-067`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b511b4ac1b0719d0780ec2c9e18278c04d48d286fc6710bb079c3e2eba6029e7`
- fixture hash: `sha256-801021da403c0e7ffabb1f8a7bb11de0378ca6fc16e45764ab572505b0e2f302`
- score hash: `sha256-c6bafb7ff4010ea18434d8e616574f6c4baf0f9e82bee67f9a96d79acf508cc6`
- bundle hash: `sha256-6d4df84012e98cda2ada1ea3116064eef9f3c7126bae36438980ebb39d3b37c2`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-88537cfda5a24580056c21d94f97ea80249672c6a52c1a8bd0de62e2aead80ad |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-10d5614e70bc2cf0ffabb8fb5fb1d4d6c95a3f8dbd6d9e4ab97bda3a863ffe22 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-403e9b1b1a77f2440f02e2ddc095733823415279d39e6be98d22ea69176f500b |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-3086107a907f6aaeeaea4f0cb5f749257216b43f01fc1f8203f5ddab94eee5a0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a083dceb | sha256-75141bcade8f91828547b5aaf1dc7003f5e190ed6bb1299cdccca609438e0eac |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a083dceb | sha256-42830e2e6f438ba64ffe968702aec463a2fbed93f96285e5014212f52e618a96 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-f56d6e1a | sha256-f7b3b058658d29deaa736a5aae65a4f06052052d09ddc6a46bd3f5b81dd8f90b |
