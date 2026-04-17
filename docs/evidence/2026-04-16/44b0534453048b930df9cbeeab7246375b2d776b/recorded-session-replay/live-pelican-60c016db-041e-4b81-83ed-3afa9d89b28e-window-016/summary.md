# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3356e25999580c1815d0ae49bf40bb5a370485dd768a3aa1572de8bdcc8cba97`
- fixture hash: `sha256-0729bfe4197c5261cfcc1c8ec0f8202300a63367aac183bee3a483ca417a77a1`
- score hash: `sha256-136d354c6f6998c5aab7db85d91ea95ce9b3270867d48f0818cae6cded1230ba`
- bundle hash: `sha256-1093ca24a15c2d53382765c9397675056b87e293ce828b57389b47e3c5fa178f`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7d7c7ee971059a51f0e36823aafb228f963680a8a0b828a8be4af2a00bb72b43 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a8226bd468f1357ee3cb7f12b9d11e8c04b74f55f4f32414ace8964fdf65f02f |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-bba875fb669472a712d47aca267ae6834f5be0db92c7ff452f933b3ab891bad8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7bb4a00f | sha256-92f86fe6b4560afcdf5c94bffd46688f5573b1835cf879ea0d1b6fdf47b453b2 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7bb4a00f | sha256-49b81861b4b4935d0dd0d8f95218a951b2fae99f178bb47128672ac95f8fae66 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-07cc64a2 | sha256-f9b50ed05ad874e672a7b0bd715a1b62706f4f572c56691f7940e3509693eb16 |
