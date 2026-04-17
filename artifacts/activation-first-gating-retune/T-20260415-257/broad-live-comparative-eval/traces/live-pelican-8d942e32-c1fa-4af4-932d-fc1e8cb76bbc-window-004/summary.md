# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-34bdf8f02ec779363fa4c8a951850c78f5f147002a61b16879fb9ea405e3f18a`
- fixture hash: `sha256-fcd33f2f91aff8f0b5df7411e8b655364e067521c4da8a9896b3ab460088a1d2`
- score hash: `sha256-5793a982c8feef1bd54e610773b0636c1a482861767142025359f5bff7c77536`
- bundle hash: `sha256-a7aebb76bbbeaedb587642ba50727b8a0db1748b3918c9d0fee9a7d94d961a90`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-59aa7e4d7b28a1f5c691ae41334f8392171bd8edbe01742338c29f7ed9b2609b |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-1d764c1eed47ff17da0dfe9f46219231e5b9a1236732c12d93d95209c2101d90 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-363fa8dd28a3ae6e3456dfcb66910373f05e41af678ea450d55401ee99729aee |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-1faaa78e1d57315f860da283e66757a59d9a94d5df70adb2b3014f4b28bacede |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-3b54fcbd | sha256-cb560bb4a9113f1252f8fc3629e81a76a65baf19ad42269afce776a3f3cfa0c0 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-3b54fcbd | sha256-90abe80253968502aa62ba028989973e0d35aa9ea5cf79a3abc056830282b4fc |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-e60fa006 | sha256-335002a4b5d1df27b04b846b3ef814d745ad1593e3207d55963b18ad1b788bcd |
