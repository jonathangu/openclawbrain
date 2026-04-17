# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-051`
- winner mode: `graph_prior_only`
- trace hash: `sha256-00bf2bd686f7cfc027d3b7749683ef5ae4ebe1a8b4b5f12763771285b87ec8ab`
- fixture hash: `sha256-1287af06cb4b83146712d66b78f07ce6e6ac74450d156f3cf86e05b95cfe0f1f`
- score hash: `sha256-5e4be0fcf5dcb1d93cd4f14d26bf72eaf67de11fcd8f96a2ba3421c862016444`
- bundle hash: `sha256-344b4543197357cf010da357d1adb05248909ea27eb67e325deb3cd513e1b808`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-80a7f9806b34ab4aca7f2c918d805e0ef978c8cb5147a44aad086817dfd7315e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1dc19ec9831e83f2ec5276f09e06f53bcb22276e7c093b27b0962c5838ec65d7 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1b105c793ed5183bbfae7e3b786ac357a7bb99634023ee31a2ba1e204a312f7a |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-477811c7735e078f77901c00ffd53f71359870fcdba660dd7fd43a41f3834d7f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f77059b2 | sha256-5b2b0906bae487b9ffe8e07cb2a36adb09f8135bd0c7aa44541522591e024942 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f77059b2 | sha256-b3a3ec2b60617ad5e6c06d422bb07a86373cf058889429daa7059e748a5c603e |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-d6e072fb | sha256-e2d09554b45694ae8fb71816a1c5a4cf22128d9f36731cf736f5294beb10e981 |
