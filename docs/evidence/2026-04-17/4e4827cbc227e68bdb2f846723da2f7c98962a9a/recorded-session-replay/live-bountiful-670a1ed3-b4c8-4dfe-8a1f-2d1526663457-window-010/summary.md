# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c80a3decbe06cbf3c4af187d8a5af847ce341540f23d409b6e7d63d31df4bcc4`
- fixture hash: `sha256-741cbfbe2c3d2f3a4ab8e97bf7b8405a7d1cec581f3191dded735c7802b1e00f`
- score hash: `sha256-26931654b9e966f70747a987b5f3d9becd84b244bfc80eb50665f1766c46e3ea`
- bundle hash: `sha256-b7e5d2b8d2ad40930959d4a974b1eec15ddd87ea160968dbaa3459457cf6e8ab`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-16c0b7f4b283cf0cadf9518aed3354f26372dc3c9867fbbccefe14e243137800 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b08eb5fc1ff25ca9747f4e1a4b98986ccb4cf7eba6dcc6f2ba034a7efa3d01a3 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-dc378d3513a7ef27a50f5884310bda9229dabb2677868ee474db6b6f6e99eb6e |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-64379825e286e65d11b2be80a02c416ecc8ea71d782f8bb66f5a8aef8bd641b8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-abac3b2c | sha256-be4627d465e9cabcc95cb1e2440c3ddd7c582602df2c46690eb1b5d8d7aa6a3d |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-abac3b2c | sha256-b79056d5f485e5d0e5856c5027e46a67631afbe0058aa5b6d6c1d70c22205ae4 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-c0809a7d | sha256-4d19b0514b79a441a36bef1ffdebe998719ca9a8a8b9b9925b156ba3da9c014e |
