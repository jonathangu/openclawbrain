# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fdd527bd79d12179b9a91214346f01f93616aaa30cfc7eab53977a331a071be6`
- fixture hash: `sha256-0aa39e409846ff84cb75f09fd340ba40a4ae31d0d07442053eabe16d211a0cbc`
- score hash: `sha256-20d812d90f0ff0ed60313154174f977998be1abfab63c73473caf0fcf340059f`
- bundle hash: `sha256-229a3e52c2926c0a4f80fd6745d59735b9d01c85bbee8aff25251d781bf4e6ed`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-349b3d6c28f24da121efce8d6fd84ec2564b6e3556e1440bc8512b8e1750cb4a |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-1a74bc151140bd15a3487b6831d7a2b73ab0b717957b285495e8bc1a76606557 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8457e7281fa12877c0298459becb166de20524959c752ea884497984b217966d |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-4f1e18e2422d366051616b0f7437874427a44416ca9ca6592e244ee0d09b0665 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-5be523b6 | sha256-0e0aa577512995e089e856eefc544f914ec66017f0478b935e359d888cc8eb56 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-5be523b6 | sha256-d5df9cb4995a77e15792d5c7589d07eabc73ec8c9eca30de2bc8ed5dec25d1e2 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-ae4d2981 | sha256-e93478512423c3a3b311bf3dff11172dcb2c235031c6507f7be8b85ce2b4fe4a |
