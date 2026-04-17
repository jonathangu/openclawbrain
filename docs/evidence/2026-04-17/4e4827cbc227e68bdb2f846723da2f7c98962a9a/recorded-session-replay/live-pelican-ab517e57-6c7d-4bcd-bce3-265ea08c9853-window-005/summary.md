# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9a8a2e6a63cc5912fb58030e76267c771c6d07671775935e13384022cf8e7c59`
- fixture hash: `sha256-d3b9199b3d1fba06ec6d727611496f93d92d13e1e28ef25defc3314d0f80c421`
- score hash: `sha256-68788495cb2e448056aea7709314fe03d3b73b1c1729e291f9f64caf9c50d8af`
- bundle hash: `sha256-9e313f0b00204224628f0c239359f4a996fedfedfea82f963deadb1a1b8ec3be`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | learned_route | 70 |
| 3 | vector_only | 70 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/8
- phrase hit rate: 0.375

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 0 | 1 |
| learned_route | 1 | 1 | 0.5 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-39cca038bdbd32b11125d0c6fba3b1b3a673e66a982ba05e8a320b541d748401 |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-04249846775e6348508e6c92d04b78d294ba0c5eeb6c2527d4a721c84afd3535 |
| graph_prior_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-4198a82b7230fa5dd1b10d5161c88c36b055ed04cfdb5142ba52604a60cc2ee6 |
| learned_route | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 2 | sha256-e6ab87904584dbff51ce9e94eacafc81bd4c606b6f15a7db6f63e019c43dfea8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-a9c82841 | sha256-4637fe089fba0f5f941cd1d2f2e78ab0ad143740a0b307c1a18e99a1aef2e6df |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | no | no | pack-a9c82841 | sha256-dd96965c2ecef49edc4fc29bcddf529f28388a227c45156a807024d4f25e798f |
| learned_route | turn-1 | 70 | yes | 1/2 | yes | no | pack-a6330ad4 | sha256-e1876f6edee13d4640729c7be32196f1b57de05b0e2f175dfbd8d1f464126f3c |
