# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-19cd6a701f3afe5404567d59955346d7cfc26c77deb7b29e61fccacc22d3bbfa`
- fixture hash: `sha256-4dda7357e5652f879faf39fc4f606d23e6674326c96ea6b533ba27ecfc72cf16`
- score hash: `sha256-c47e165f039e2f3a588d33d174d160548b1b7f03d05db03096771d2bca3ee5b9`
- bundle hash: `sha256-d153cfe52b9b8fa0fb48d200096489c0bd4e6d31dc01d34cb5c40ad084b81441`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-934729bde748377658ef5251e3c9784137a24d5cc133cff448c2ec475fa6a4b7 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b65d0b3efc551b9db6334b816f57e210ceb81303f1840175997d00cebff68411 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0fcdeb3d5b3d49eeeae18b792cbd31ee870131cceb043f9a432d87c1d35f74ad |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-2e453896d4d3609255828e3c8d85cffff5608d374c91ded3df4410fa815d2fec |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-c0ff643c | sha256-87033ced33d19701f6f8c8de265e24cefac145b178dc365397708c55f5452644 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-c0ff643c | sha256-8d823ce61dc74589a198383762fbf21f8e446e2ec9781e63dff1d77a10563ec2 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-c0ff643c | sha256-87033ced33d19701f6f8c8de265e24cefac145b178dc365397708c55f5452644 |
