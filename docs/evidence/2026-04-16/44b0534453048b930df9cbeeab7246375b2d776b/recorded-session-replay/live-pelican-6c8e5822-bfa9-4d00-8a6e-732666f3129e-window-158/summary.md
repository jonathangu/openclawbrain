# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-158`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1daacbdd680cf4033ed5d9fa2efa105e6544ffe1129c7600ff85b76c0c2f8393`
- fixture hash: `sha256-ef7e749ef838de36d236aa29e0590a88a86c6b42be12cb84bf00123ad9c263a6`
- score hash: `sha256-28c6ae873480c63e6c0d252c1bc6adf771a7766487667d9145d87ec40cf9ddb3`
- bundle hash: `sha256-d2821271ef369d87fa7764ba45bc2f066a02f68eb3546dd6708a2fb0e5970746`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-715391ab03706266e4dd92a9d6ff099345f003fb7379bf779cf731b9d18a7950 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-37b4201007d9509b49d5cad73ed67e01403631a355ffa079e97c47a342e4511b |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f39bfa1c00d726b3bfaa950f5380814cc7a0fba7b2adb8f6c898990aa23ba114 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-ad639c9bdc7f0f0d53b6f3e70254aff5ce8f2fe58a6a2cd2d8fa98c31d5db0b9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d489a9dd | sha256-f4553278b4754201392895b445855afa34bb0fde133e72a55ea79dfdb06bbb18 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d489a9dd | sha256-48913ea9705e158663f81ba56d2fe1a7d3336dac3e6765778824b69b4db2881c |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-f5e5b200 | sha256-abac1dc5332bb46215621df1a5e7912e822644ddc3a106dd068579e8b13f6df2 |
