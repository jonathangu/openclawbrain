# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4e89d5a5-376f-4b67-9e88-3aea314ab451-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-11f89d0770e58c74a32e0ac08329b409440ac220cb647ec446567aadc15cbdd6`
- fixture hash: `sha256-7793c2d77fac055a1c7c47c9d026a76a01511a45ccb17bbe5db49943de3d0ea4`
- score hash: `sha256-092d111ea3c227ec9fa3ceaef4b6333f9ce45b0b8ea0076afd33b0d1224db144`
- bundle hash: `sha256-1b67f231a56686a01d9934ac8e45cfe1822445bf28d99c774cf2e90fb9469339`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3297dca5d83084645cc80493377a366cf545c5142415159b972c4f8430720ab7 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-591fc7062148c1fa3cfb19bb075aee5ef9b7b943d09152368c7abf063298ea53 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-61e0693198e231229e5ea965c9147555551cde0c6e7792828d244c69caaf4b86 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-f1dd2c31db3f48251d183de343c2fde895de62d6c36d24a4bf49fbbb4f023c53 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-7c3e48d9 | sha256-9d7d5f1035bad64bb9303be6cae79f8331fd6b2fcb3eca72637931747880d100 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-7c3e48d9 | sha256-755d833d17d7d8bdb215d46ebd5f08935c2406ec6638b3fd4348ae7ba4a7225a |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-d7af2ef2 | sha256-73de5dc49f1e46c30dfcd18fcbc3d05387039c56f3c439a702e854cda0b16e4e |
