# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-071`
- winner mode: `graph_prior_only`
- trace hash: `sha256-aa6afe07711fbc8a13484cd14e70ac82c78cc503ee5449452a36b775fa63c3d1`
- fixture hash: `sha256-bacff39860081979b6852dc7223e7e30d3e6e8700496899a8864e78cf3c36fa0`
- score hash: `sha256-ea447f56ef0f8f9b7af5c68133d125ec58f4ec69c0b41bbdbb8fd6d8e68a0827`
- bundle hash: `sha256-ce29b51f8f5ff8dd131410e903d22f59c4157dafcebb30a40c9732fc7d8adc40`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-080c0791d3c8d4b27935c18a06ca48413df84ee848ffe0bfd6099d007a81a298 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-08021bbd38920be89feadd8612e31c13ee5bf139f82964aa45614d020b2b90f2 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0dbfc26d2d9bc049092c2c16bf3e5d472186dcb8f64c2733ce68f442da48365f |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-244d586506bc90f702588a399800035ca662f77ebc7992f704e933146e6bf951 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-84853203 | sha256-623edda6e797ecd9c8a4bc60203a2e6a06142ac01272ec24b9924a21e1eb8533 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-84853203 | sha256-70de72081f220a023fe3ff99ec7c5a223fe3604d29cd6bb806370b3007ec36e2 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-b412c820 | sha256-da165993bc15381eb03dfdc5126238831d542b185ac066675ba98a3d01f9c01a |
