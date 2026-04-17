# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-954e7370fea342c102d12700848d801f9cb4f766c26a152991092af096940dab`
- fixture hash: `sha256-71ae1b688e319cb6bd41b60a17bd7289e838d68bd59d5831fb46d3db379a64f9`
- score hash: `sha256-15d3f3746470a9a2bc62859d1b3f9126be84779e49d16c35ed697660b82af6da`
- bundle hash: `sha256-366030ca41e85403e42263ce8b967f0e4ff450e15d18f51b69c484c8263397d0`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c508d0d953d2e20d8464c0689a6d8d9a5c0442d5d485367bc8ffc01689888e09 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a7c87b7d091453d61adec1b79a290ec612d6128d2b44c5cf98f1bbf38c91f8bb |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7ac6d49432b3fcdfd9d51d1828e8c9f308e712a124cb64550f187a774e42bacb |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-c2ad792aff3869ea7ef0818b8f712458f506cbe4837c5e052e6849e63c23311a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-64b79640 | sha256-7039043ac2088cb8a26d27ba19884792d0e04978673b3d0c4850fc19c6e541b1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-64b79640 | sha256-083ba976c7f0f6f9414fd641ba62f02161efe2c676413680f3c673173a3ab01f |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-c4944fcd | sha256-12b4fb090a377042aacb861172dc937b65a1dcdb960a7ee30d231c1632186bda |
