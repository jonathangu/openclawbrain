# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-041`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cd17705850f5fd87f770e4757922f483be90c3dcc5bfff44d696c49e62560cb7`
- fixture hash: `sha256-743937076adce554085fa9dd3236567f573df76180477a11d06a07f43c4044bc`
- score hash: `sha256-fddedf4510d67b9747d8a9d5e4ff36162f8c4016f8a46c2f96a35495ed48b5ec`
- bundle hash: `sha256-5f3c3bed179392e48071c82d8c2dd1975a6b351b3bdf93e4e270a1a0a0508e51`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-38ffbd4329a21a765f40f1a44ad7d1cc0603504c91e4e697e7b573151d0b2478 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-df6501113490487c62a611bd9ad5c09a3c79a1653b6f63bd544c867f94261865 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-cff13cfdfb85fbb59219af817a0be3fbee1679f39ca5a1840b8c3202a8e5c471 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-39e8c9c76653af3283a9ec39fc204f9d08e5b8ea250419b25d061dd7878f3af2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5baed2e5 | sha256-857c678063e178e7cc6c53ef9bd5099c03eadf2056fedebb0172115c6f0e34a0 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5baed2e5 | sha256-f00158128b7914cd9a107a066966fd8d20b615673ef8def3d938ce4976e95f99 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-9b635fbe | sha256-e858c69c814da10f3e4394d53600dd8be3ef37f13411d39f90dede63a1317516 |
