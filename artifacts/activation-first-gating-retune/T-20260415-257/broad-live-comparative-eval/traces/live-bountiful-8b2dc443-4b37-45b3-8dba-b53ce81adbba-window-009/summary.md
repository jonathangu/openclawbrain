# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8eced4262f5a642239299c7c899085a7bd53ad7880d03357a10326803fe33aa8`
- fixture hash: `sha256-5aa5748a68c006cb4152d6b9766d43523c43872689382d99e9608f0fedb263a8`
- score hash: `sha256-6e0e0b14834143937184447daf440d5ae35e3dd026c6062a300b7d11c2bbac0f`
- bundle hash: `sha256-71542f41bbf8bcd7b23a96ff1612d7a894bc06a749392a67686a0f32ebea8d8b`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-86ead80920d9422dc3144931f0210740c8474d5a0351518c55316e7dfbfbffe7 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f14f1a0fa9321b8748d28de341200e15a554908ccd9431c9632026dbda21afd3 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-47cff7d09bacbd763331abb42fd9d7bbb03d60e1f8812682a5c08e6aed07c7ed |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-ec21c9d7f7561b32c501d2e69eaf2317efb57c602dae257622bde6d8f8899e2c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f568da39 | sha256-09a412a842b94166e0f11520efe06524b16a009d0558b129ae7e5e81b58e9ede |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f568da39 | sha256-346bb002b4eaabf98f600e80c63fecd09f3d02c2dc65a20d64dbab0d32c6ae49 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-f2ea182a | sha256-e7c8fd61037d6c113c9669a881b60e02171c504d0dc807e389e8cc50b95d942b |
