# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-067`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b511b4ac1b0719d0780ec2c9e18278c04d48d286fc6710bb079c3e2eba6029e7`
- fixture hash: `sha256-801021da403c0e7ffabb1f8a7bb11de0378ca6fc16e45764ab572505b0e2f302`
- score hash: `sha256-af8d31b06170f9f0f07cf443144bbf73be3ab8e7a751ae384c50639998b66d84`
- bundle hash: `sha256-9efc1c6fe4461a79e2ad5b595745ddb0f89d957eabba82fdf058a1eb8ba45b27`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-88537cfda5a24580056c21d94f97ea80249672c6a52c1a8bd0de62e2aead80ad |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fe72c2353f35a0e83edc8c853a0c6ecaa8e1ba671c88e1b03b16e3313558fe08 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f8a600fbc6af651a6e0758063b5db1cb3966cc31abf5b4d57efd33cd9fed293d |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-27a9473cce74041c44eb4f6a28f55e21b166d7576f9dc6995ea8ce2ef46bfa08 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6a459aa9 | sha256-c8ba7c45143d01d8329d457683c60a1785d22995f6d85a3ad677f326602e23d7 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6a459aa9 | sha256-1508138291bde7dd64be51e4d4a73d3ddab256b2ca31c1f474a26e8844021804 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-bf2f2bd8 | sha256-f7e6261c18cfe651ac0dcdfda3292a94d05b9a4061daca1306f021ec47686639 |
