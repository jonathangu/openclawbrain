# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4b29852ddcae763768818b925bfa513423dbca5c8ad934450c78f0838b90cfab`
- fixture hash: `sha256-f828a5ef63881667b78ea5f5530e5417bb5590176f57bdcf8c4590150136788a`
- score hash: `sha256-4c0f146cd9d3158517f25661b516d1821855aa0a75c61d849e3fc19e82a04627`
- bundle hash: `sha256-6000552ab95ebe03de620ab27f728e605d1f84b60dfeeb228d8845c46df9cf9b`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d23197d4b519cff22649347398dfab9ce049fcf294afab672a8b41fd8ebcbbad |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a2edb66367fa54ad3a22d7521ecfc66ff0c262888456aa848c18dd16079e9bd2 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-209d671fc7b7e8402e5194cbbc2fa0460577c592ab4a029180abb1409aa0f547 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-3a1620b0c615c993d9f3ba1b7c737a310c142105cd9272fe16a74063a3af6b57 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-72a35763 | sha256-e9a0a0c4f0c83c7501d7fd6640e48bcd0efce1a5731fd0f87a38f52ee9efa791 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-72a35763 | sha256-064ed980b1758cf923e2b858830a161bb1d04cf96181e87362b82df8dc103e9d |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-72a35763 | sha256-e9a0a0c4f0c83c7501d7fd6640e48bcd0efce1a5731fd0f87a38f52ee9efa791 |
