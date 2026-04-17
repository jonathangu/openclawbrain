# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-838b9295d0df32bf17309a7744670eaab3129f24a6dca2ca9110c4b4940f8ca0`
- fixture hash: `sha256-56f7d90cfb38f59327532bc9b6beae4801650c72b03cf0a3e492173ea24b06f6`
- score hash: `sha256-e6154cf5d3f7a8f69a9ce5f91ffb1d9af5d0822ee6dbb90584fac8f2ef81ca25`
- bundle hash: `sha256-ecaca9036d0235812e0bb1f51b857796de0de2021a7f6adbff95115808f0253b`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d12703af710851e5a23d60b1d20c78b1a6044ead7e09a16f607df5e76e23db43 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-9e121008350ec71632c890f8ff3185d33169f143e23bb81ff44eab93b8cfb0b3 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-7128290d8444ba1479973b85bbd7f600d83d3dc902e1518e3408ac18417d0333 |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-b18ebdd7e4a7599a5106d75ac23f7b7982451c3fdc505f5c4f857c5d1a0ff830 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-dfe795ea | sha256-d0284ce6b6bc8260a77440904057d6e965acfd3931850abfc9a7921c0fedd175 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-dfe795ea | sha256-963a52c0e8c081501641d9c595975974d56dff4d3405e7484c6e6b35c6d9ecbc |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-4b2b0bf5 | sha256-91f1b1f6983104bec1425df083dd68beace7053f080edfb9be997e48a22ee5ee |
