# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-15f74481afa0ad3c49942a752d93fa21610759dcd0f5184c05ee667b747607b5`
- fixture hash: `sha256-27569dfe07b6cf66e357fc072347afe0c073b0dd225ff6f7f6dbd4f6b53bd5c5`
- score hash: `sha256-af466f70bb8f0c1fa7041d267c15bf5aad473248e4bf21a34906f8f80b05dc54`
- bundle hash: `sha256-94fa598f816bd19002a40a71bcb7646ae9c24dce81e4a00e3c10c42121d31974`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d9780bcb02b4dddac9cfba41582ad72477a9d4e9b030a1ad3ced919c347c5d08 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d8244e5aea6f9ac8537d62c64e042c7fbc5a8c633899935ca66b443e5824a37f |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d12fb56b3f15ea19aa9481a168d81e38474c39e33b25e88e5d3cf3b1d13b0227 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-39c293845a470c46b76b5274db6442b9243b5cfb0863d198b4bd766880432498 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-11a719b5 | sha256-c45cf5b664fe9c2c470d7238213d07e5a0f4de12e8902e9607b30e560507734a |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-11a719b5 | sha256-e69835657c20e3e545d1cb1bab68360068d9a1f8b2a0758efd13ec1c15b41476 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-254d9614 | sha256-3165688d7ede04cd85b9bc31d8dfe6789251616526496e5b089be936e2692d04 |
