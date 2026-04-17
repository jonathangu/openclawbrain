# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-060`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ba906605b22dc94a498f1f2a524326ba3cc034e63e0a380bea5c3ad692ee02f8`
- fixture hash: `sha256-a13d04d7740d5d386089602f56e23b4bb142c5bed9f7073eeb6516366a131246`
- score hash: `sha256-587d1ecb7b8d2dfcc17d273a3725168a2a7015bbfc6771c9d171aec0dc2836e8`
- bundle hash: `sha256-9fca5e64efd9ee160c197101de3465a57773953a0edc58bfe5ce5071d6856530`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-aec7ec8f6799a0ee01f6b4130aafafed76ac2a835827ba3dac19bfdac983b407 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-82468a8accc9a859f3fcb2fa9930e31d3d46e8b87153212526a662f4e65eda41 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7248f89318541b54c57092b05f60f136a6c7b50e2005be04a2f680e037781d35 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-bb2cbe56159ec034cdee032da73aa6acb4bff0a8c1ccd2288deedf4a3b515e87 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d8ceafce | sha256-7dfa21c19c4082a072225d51483cc1c6a3bb1040ff18980bbd9779b6f0f91581 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d8ceafce | sha256-2571d13334c6f32d85ad22cfcabd3472490f75a027ac650fea7afb2bb54f1f5b |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-7b4f5d2d | sha256-8164fe42b86d2b39e3b93b5ca91738c83095761f957e5378cc2615c3668d809b |
