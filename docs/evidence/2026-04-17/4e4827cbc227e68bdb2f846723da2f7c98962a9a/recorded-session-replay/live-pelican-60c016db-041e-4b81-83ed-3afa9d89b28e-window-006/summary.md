# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-15f74481afa0ad3c49942a752d93fa21610759dcd0f5184c05ee667b747607b5`
- fixture hash: `sha256-27569dfe07b6cf66e357fc072347afe0c073b0dd225ff6f7f6dbd4f6b53bd5c5`
- score hash: `sha256-6aa0ca6444a75b59ecab9ce40e2a57eb0e4b4d1456abdb1f25403dcdcf10185c`
- bundle hash: `sha256-a0842ab7e1e24359847c8af5767d0551e2399fc434336ab22c0ce9b6b6e225cc`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d9780bcb02b4dddac9cfba41582ad72477a9d4e9b030a1ad3ced919c347c5d08 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f7243b9737299dc49d5bb1e0573389d62271ed8799f62e1b07005e0ed6cb00e3 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c5b4f8d3546eb8ff0108114d0c42fbbbe2c941d15adf507376fdf16c1359c516 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-9ec918081e143da11c58180a7f47b90f05a2f7ae2a11dff0652a0a20c1e2e569 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-460f9e50 | sha256-fa7b88fc26aee7e9f14c2ebf954ec8524001e5600aa6a1315dabcd454cb4d05f |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-460f9e50 | sha256-a76a4a8d31a4956875614a277890ad05639dd6c956feb1e1f29d799fb1059461 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-59b61aaf | sha256-2c8c7338f63d8ee659500e1ff4830065a6dfd89981ad1e14957d752704291c56 |
