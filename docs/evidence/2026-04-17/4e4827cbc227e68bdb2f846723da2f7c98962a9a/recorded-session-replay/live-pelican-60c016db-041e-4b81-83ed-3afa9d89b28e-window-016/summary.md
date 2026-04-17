# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3356e25999580c1815d0ae49bf40bb5a370485dd768a3aa1572de8bdcc8cba97`
- fixture hash: `sha256-0729bfe4197c5261cfcc1c8ec0f8202300a63367aac183bee3a483ca417a77a1`
- score hash: `sha256-54e06b6974227f60d974ce72c0858dbf14f038d8a72456a026028bc72ce49664`
- bundle hash: `sha256-49fdbbedadb496329273e898f32ec3c2d502d9057463ce7250cc7d7a631ae9dd`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a4b42315dd1b1a176628a7fe8f13cc2dabd4068509144906d5e0b01e2bdfcba3 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d66c3f90b97329e1e88269bb59a30396b1adfcc7c5557ad5a9aef8ea4a34bbce |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-eddc93352eab3b3d4ade2eb5b27f41f56a05edd26b1a5e2aaf27779ed8130d5e |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-59de65338f72d72ad6e00fb37b20b7829f546308ce06c562d3baa09aab9f75da |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ec9c794c | sha256-22610f63ca6a19b9846c24d27a3d48bdd3dcda94a17096b27361cec33be1e265 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ec9c794c | sha256-ef2cf60d00d734224ea64c2b7fbe6245d4661234701be9ceb9b743e7fd0f3f81 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-78b43ddf | sha256-82ee32e85754690190047789498685aa2658679722e87cccbc1bd840ebe6f8da |
