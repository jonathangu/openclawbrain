# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fdd527bd79d12179b9a91214346f01f93616aaa30cfc7eab53977a331a071be6`
- fixture hash: `sha256-0aa39e409846ff84cb75f09fd340ba40a4ae31d0d07442053eabe16d211a0cbc`
- score hash: `sha256-504a5f1404184bc7a48223e27e2775a80624f0e621363212bcfaf6047b8e54fa`
- bundle hash: `sha256-2a35017bd98f5ca6a3e156c620adc89861babf73da16595d994057b123d2ff78`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-349b3d6c28f24da121efce8d6fd84ec2564b6e3556e1440bc8512b8e1750cb4a |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-dffeb5d49ef2dfa951916c970167d8326cb9c12b4747d519ae21117e81e6864f |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0f9eb57fc0571fc3c58eacdd8cfeeb95f64b092f56e0943984d5a1faff136d74 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-155918665ebf90aff5667b546fde11ce42d554aef87abbc5a0d27e1fa5647586 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-63fe712f | sha256-14c8d7f50e3dfc54c61f1fe9e733f0f786eec93c5774e3d216d0b183c98445b7 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-63fe712f | sha256-9d95f785d8ca7cc90727487b0ecfa9377f430447e3060a135bf7f02052fe8166 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-b66676fa | sha256-036a169d7569163080b637472a4a78cf9f2c1ffec76d64f9754124a9e2f08ea4 |
