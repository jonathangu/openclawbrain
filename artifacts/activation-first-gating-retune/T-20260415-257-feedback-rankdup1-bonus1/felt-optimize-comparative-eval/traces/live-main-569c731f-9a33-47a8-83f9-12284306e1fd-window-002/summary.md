# Recorded Session Replay Proof Bundle

- trace id: `live-main-569c731f-9a33-47a8-83f9-12284306e1fd-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d27db54f25bb1682fcfa202523b5f1c6efccc7e2753d8e02e54ba11f6e3abbc5`
- fixture hash: `sha256-0e0db1f3540c6bbafcaa45e48b36b0aa0cc986ef0dddf4d7e13951d4b175679f`
- score hash: `sha256-c2f5b02d79b35604cf8609fb7fe04f4c56c514e34a07b6fcd993f17e6ee732a7`
- bundle hash: `sha256-b2591d4b0d4dd49b5eba1730e3c998465909bd652dbbdc467e550c4044006748`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-71d12c78bbd92c17749c2ba921bc24d7594735564898b2d4c08d5a5f8badb93b |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-1d666d8a48c09e624f75dbf891eda2f1ae6197af96f09ffa4bfd1b6884f4e473 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-876f6d61240d7a55bcd6c3151ab5c331b5f29a3ddd0d3798028f409ea8d1d60f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-ce8113430f0f3bab423ff1709c06be08c982e81c5d825404c4720c035f30dea5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-5fe1d870 | sha256-b03c078b36bc22c6b9de92a3e3cbae9ddcaf1b6150e17336e669afe52caa5f07 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-5fe1d870 | sha256-1c9639d2108a31f9c2fdf28c877f04505968cbba17f3fdc3071c0320cd1044f3 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-5fe1d870 | sha256-6bde84669f794c9b011d4ea6eca68fef740e00da56cbf31f61658cc86a5db65d |
