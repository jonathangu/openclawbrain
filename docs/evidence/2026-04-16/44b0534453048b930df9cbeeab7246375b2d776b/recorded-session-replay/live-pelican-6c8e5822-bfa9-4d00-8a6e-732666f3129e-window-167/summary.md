# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-167`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9808f8fc2b34b9bdd2037973a4af69235ae13bd43fa03c06a9cd2c930faaaa29`
- fixture hash: `sha256-c59bcee0d7e5004e8699b7491ca609cee1baf21baa2824b5dbd8c966b365083b`
- score hash: `sha256-0338842912ba65f32f8741f0bc81c34e46e783a02fee8f60ac6d88e41efb9a86`
- bundle hash: `sha256-d2ec47a554be57511a051da08e9a75d7d80a3033db2e11a94ec63136ad718834`

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
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-68b72eac2031bede9aa8770eaa1f000f5f4b3a15976311e630a954289393b0bd |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-6a1e6052e4570ae54af6acf708858610aa4f84820aecd170a448a0256f3edd6d |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-918ecb7e4aff962603706bba13134f400cc25cdebc49bcd05821507d9bb7e3ed |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-021262d3a878de9cf80f6347d66d16d3dc4d88e699b99f843aa8aa556bef7e14 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-216b1653 | sha256-2df0a443a2509d107bf194a28e49cee9088e01f37eeeb3f7cfdb570d34da9dc5 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-216b1653 | sha256-bcf0f0da3f15c2be6d013185a86f2f3ae8cfff00af398197cedd31fd81541345 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-2a601d00 | sha256-65a3b7aad0819dfe830386788ce76c626a90b4d47bc80579988613b31506ff5a |
